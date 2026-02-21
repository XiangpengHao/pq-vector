use crate::hnsw::index::{HnswBuildConfig, build_hnsw_index};
use crate::ivf::{EmbeddingColumn, EmbeddingDim, Embeddings};
use arrow::array::{Array, Float32Array, Float64Array, ListArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowSchemaConverter, ArrowWriter};
use parquet::basic::{Compression, Encoding, PageType};
use parquet::file::FOOTER_SIZE;
use parquet::file::metadata::{
    ColumnChunkMetaData, FileMetaData, FooterTail, KeyValue, ParquetMetaDataBuilder,
    ParquetMetaDataWriter,
};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Build an HNSW index and embed it into a new parquet file.
#[derive(Debug, Clone)]
pub struct HnswBuilder {
    source: PathBuf,
    embedding_column: String,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
}

impl HnswBuilder {
    pub fn new(source: impl AsRef<Path>, embedding_column: impl AsRef<str>) -> Self {
        Self {
            source: source.as_ref().to_path_buf(),
            embedding_column: embedding_column.as_ref().to_string(),
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 42,
        }
    }

    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    pub fn ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn build_inplace(self) -> Result<(), Box<dyn std::error::Error>> {
        let config = self.build_config()?;
        let embedding_column = EmbeddingColumn::try_from(self.embedding_column)?;
        let parquet = read_parquet_with_embeddings(self.source.as_path(), &embedding_column)?;
        let index = build_hnsw_index(
            &parquet.embeddings,
            HnswBuildConfig {
                m: config.m,
                ef_construction: config.ef_construction,
                ef_search: config.ef_search,
                seed: config.seed,
            },
        )?;
        let plan = ParquetIndexAppend {
            path: self.source.as_path(),
            index: &index,
            embedding_column: &embedding_column,
        };
        append_index_inplace(plan)?;
        Ok(())
    }

    pub fn build_new(self, output: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let config = self.build_config()?;
        let embedding_column = EmbeddingColumn::try_from(self.embedding_column)?;
        let parquet = read_parquet_with_embeddings(self.source.as_path(), &embedding_column)?;
        let index = build_hnsw_index(
            &parquet.embeddings,
            HnswBuildConfig {
                m: config.m,
                ef_construction: config.ef_construction,
                ef_search: config.ef_search,
                seed: config.seed,
            },
        )?;
        let plan = ParquetWritePlan {
            source: self.source.as_path(),
            path: output.as_ref(),
            batches: &parquet.batches,
            schema: parquet.schema,
            index: &index,
            embedding_column: &embedding_column,
        };
        write_parquet_with_index(plan)?;
        Ok(())
    }

    fn build_config(&self) -> Result<HnswBuildConfig, Box<dyn std::error::Error>> {
        if self.m == 0 {
            return Err("m must be > 0".into());
        }
        if self.ef_construction == 0 {
            return Err("ef_construction must be > 0".into());
        }
        if self.ef_search == 0 {
            return Err("ef_search must be > 0".into());
        }
        Ok(HnswBuildConfig {
            m: self.m,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            seed: self.seed,
        })
    }
}

/// Magic bytes to identify our pq-vector HNSW index format.
const PQ_VECTOR_HNSW_MAGIC: &[u8] = b"PQ_VECTOR_HNSW1";

/// Metadata key for the index offset.
const PQ_VECTOR_INDEX_OFFSET_KEY: &str = "pq_vector_index_offset";

/// Metadata key for the embedding column name.
const PQ_VECTOR_EMBEDDING_COLUMN_KEY: &str = "pq_vector_embedding_column";

#[derive(Debug, Clone)]
struct IndexMetadata {
    offset: u64,
    embedding_column: EmbeddingColumn,
}

pub(crate) fn has_pq_vector_index(path: impl AsRef<Path>) -> Result<bool, Box<dyn std::error::Error>> {
    Ok(read_index_metadata(path.as_ref())?.is_some())
}

fn parse_index_metadata(
    metadata: &FileMetaData,
) -> Result<Option<IndexMetadata>, Box<dyn std::error::Error>> {
    let Some(kv) = metadata.key_value_metadata() else {
        return Ok(None);
    };
    let offset = kv
        .iter()
        .find(|k| k.key == PQ_VECTOR_INDEX_OFFSET_KEY)
        .and_then(|k| k.value.clone());
    let embedding = kv
        .iter()
        .find(|k| k.key == PQ_VECTOR_EMBEDDING_COLUMN_KEY)
        .and_then(|k| k.value.clone());
    let (Some(offset), Some(embedding)) = (offset, embedding) else {
        return Ok(None);
    };
    let offset: u64 = offset.parse()?;
    let embedding_column = EmbeddingColumn::try_from(embedding)?;
    Ok(Some(IndexMetadata {
        offset,
        embedding_column,
    }))
}

pub(crate) fn read_index_from_payload(
    payload: &[u8],
    embedding_column: EmbeddingColumn,
) -> Result<(crate::hnsw::index::HnswIndex, EmbeddingColumn), Box<dyn std::error::Error>> {
    let header_len = PQ_VECTOR_HNSW_MAGIC.len() + std::mem::size_of::<u64>();
    if payload.len() < header_len {
        return Err("pq-vector HNSW index payload is truncated".into());
    }
    if &payload[..PQ_VECTOR_HNSW_MAGIC.len()] != PQ_VECTOR_HNSW_MAGIC {
        return Err("Invalid pq-vector HNSW index magic".into());
    }
    let mut len_buf = [0u8; 8];
    len_buf.copy_from_slice(&payload[PQ_VECTOR_HNSW_MAGIC.len()..header_len]);
    let index_len = u64::from_le_bytes(len_buf) as usize;
    if payload.len() < header_len + index_len {
        return Err("pq-vector HNSW index bytes are truncated".into());
    }
    let index_bytes = &payload[header_len..header_len + index_len];
    let index = crate::hnsw::index::HnswIndex::from_json_bytes(index_bytes)?;
    Ok((index, embedding_column))
}

pub(crate) fn read_index_metadata_from_file_metadata(
    metadata: &FileMetaData,
) -> Result<Option<(u64, EmbeddingColumn)>, Box<dyn std::error::Error>> {
    Ok(parse_index_metadata(metadata)?.map(|meta| (meta.offset, meta.embedding_column)))
}

pub(crate) fn read_index_metadata(
    path: &Path,
) -> Result<Option<EmbeddingColumn>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata().file_metadata();
    Ok(read_index_metadata_from_file_metadata(metadata)?
        .map(|(_offset, embedding_column)| embedding_column))
}

pub(crate) fn read_index_from_parquet(
    path: &Path,
) -> Result<(crate::hnsw::index::HnswIndex, EmbeddingColumn), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file.try_clone()?)?;
    let metadata = reader.metadata().file_metadata();
    let (offset, embedding_column) = read_index_metadata_from_file_metadata(metadata)?
        .ok_or("Missing pq-vector index metadata in parquet footer")?;

    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;

    let mut payload = Vec::new();
    file.read_to_end(&mut payload)?;
    read_index_from_payload(&payload, embedding_column).map_err(|err| {
        format!("Failed to decode pq-vector HNSW index payload at offset {offset}: {err}").into()
    })
}

struct ParquetHnswEmbeddings {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    embeddings: Embeddings,
}

fn read_parquet_with_embeddings(
    path: &Path,
    embedding_column: &EmbeddingColumn,
) -> Result<ParquetHnswEmbeddings, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let mut batches = Vec::new();
    let mut all_embeddings = Vec::new();
    let mut dim: Option<EmbeddingDim> = None;

    for batch in reader {
        let batch = batch?;

        let embedding_col = batch
            .column_by_name(embedding_column.as_str())
            .ok_or_else(|| format!("Column '{}' not found", embedding_column.as_str()))?;

        let list_array = embedding_col
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("Embedding column is not a list array")?;

        if list_array.null_count() > 0 {
            return Err("Embedding column contains null rows".into());
        }

        let values = list_array.values();
        enum FloatValues<'a> {
            F32(&'a Float32Array),
            F64(&'a Float64Array),
        }
        let float_values = if let Some(array) = values.as_any().downcast_ref::<Float32Array>() {
            FloatValues::F32(array)
        } else if let Some(array) = values.as_any().downcast_ref::<Float64Array>() {
            FloatValues::F64(array)
        } else {
            return Err("Embedding values are not float32/float64".into());
        };

        let null_count = match &float_values {
            FloatValues::F32(array) => array.null_count(),
            FloatValues::F64(array) => array.null_count(),
        };
        if null_count > 0 {
            return Err("Embedding values contain nulls".into());
        }

        for row in 0..list_array.len() {
            let row_len = list_array.value_length(row) as usize;
            if row_len == 0 {
                return Err("Embedding row has zero length".into());
            }
            let row_dim = EmbeddingDim::new(row_len)?;
            if let Some(existing) = dim {
                if existing != row_dim {
                    return Err("Embedding vectors have inconsistent dimensions".into());
                }
            } else {
                dim = Some(row_dim);
            }
        }

        match float_values {
            FloatValues::F32(array) => {
                for i in 0..array.len() {
                    all_embeddings.push(array.value(i));
                }
            }
            FloatValues::F64(array) => {
                for i in 0..array.len() {
                    all_embeddings.push(array.value(i) as f32);
                }
            }
        }

        batches.push(batch);
    }

    let dim = dim.ok_or("Embedding column has no rows")?;
    let embeddings = Embeddings::new(all_embeddings, dim)?;

    Ok(ParquetHnswEmbeddings {
        batches,
        schema,
        embeddings,
    })
}

struct ParquetWritePlan<'a> {
    source: &'a Path,
    path: &'a Path,
    batches: &'a [RecordBatch],
    schema: SchemaRef,
    index: &'a crate::hnsw::index::HnswIndex,
    embedding_column: &'a EmbeddingColumn,
}

fn write_parquet_with_index(plan: ParquetWritePlan<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let vector_size = plan.index.dim() * std::mem::size_of::<f32>();

    let parquet_schema = ArrowSchemaConverter::new().convert(plan.schema.as_ref())?;
    let output_columns = parquet_schema.columns();
    let embedding_col_path = embedding_column_path(output_columns, plan.embedding_column)?;
    let column_options = collect_column_write_options(plan.source, output_columns.len())?;

    let mut props = WriterProperties::builder()
        .set_data_page_size_limit(vector_size)
        .set_data_page_row_count_limit(1);

    for (column, options) in output_columns.iter().zip(column_options.iter()) {
        let path = column.path().clone();
        props = props
            .set_column_compression(path.clone(), options.compression)
            .set_column_dictionary_enabled(path.clone(), options.dictionary_enabled);

        if let Some(encoding) = options.encoding {
            props = props.set_column_encoding(path.clone(), encoding);
        }

        props = props.set_column_statistics_enabled(path.clone(), options.statistics_enabled);
    }

    let props = props
        .set_column_dictionary_enabled(embedding_col_path.clone(), false)
        .set_column_statistics_enabled(embedding_col_path.clone(), EnabledStatistics::Chunk)
        .set_column_write_page_header_statistics(embedding_col_path, false)
        .build();

    let file = File::create(plan.path)?;
    let mut writer = ArrowWriter::try_new(file, plan.schema, Some(props))?;

    for batch in plan.batches {
        writer.write(batch)?;
    }

    writer.flush()?;

    let index_offset = writer.bytes_written();
    let index_json = plan.index.to_json_bytes()?;
    let index_len = index_json.len() as u64;

    writer.write_all(PQ_VECTOR_HNSW_MAGIC)?;
    writer.write_all(&index_len.to_le_bytes())?;
    writer.write_all(&index_json)?;

    writer.append_key_value_metadata(KeyValue::new(
        PQ_VECTOR_INDEX_OFFSET_KEY.to_string(),
        index_offset.to_string(),
    ));
    writer.append_key_value_metadata(KeyValue::new(
        PQ_VECTOR_EMBEDDING_COLUMN_KEY.to_string(),
        plan.embedding_column.as_str().to_string(),
    ));

    writer.close()?;

    Ok(())
}

fn embedding_column_path(
    columns: &[parquet::schema::types::ColumnDescPtr],
    embedding_column: &EmbeddingColumn,
) -> Result<parquet::schema::types::ColumnPath, Box<dyn std::error::Error>> {
    let mut matches = columns
        .iter()
        .filter(|col| {
            col.path()
                .parts()
                .first()
                .is_some_and(|root| root == embedding_column.as_str())
        })
        .map(|col| col.path().clone())
        .collect::<Vec<_>>();

    match matches.len() {
        1 => Ok(matches.swap_remove(0)),
        0 => Err(format!(
            "Embedding column '{}' not found in parquet schema",
            embedding_column.as_str()
        )
        .into()),
        _ => Err(format!(
            "Embedding column '{}' maps to multiple parquet leaf columns",
            embedding_column.as_str()
        )
        .into()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColumnWriteOptions {
    compression: Compression,
    dictionary_enabled: bool,
    encoding: Option<Encoding>,
    statistics_enabled: EnabledStatistics,
}

fn collect_column_write_options(
    source: &Path,
    expected_columns: usize,
) -> Result<Vec<ColumnWriteOptions>, Box<dyn std::error::Error>> {
    let reader = SerializedFileReader::new(File::open(source)?)?;
    let metadata = reader.metadata();
    let row_groups = metadata.row_groups();
    if row_groups.is_empty() {
        return Ok(Vec::new());
    }

    let first_columns = row_groups[0].columns();
    if first_columns.len() != expected_columns {
        return Err(format!(
            "Expected {expected_columns} columns in source parquet, found {}",
            first_columns.len()
        )
        .into());
    }

    let mut options = Vec::with_capacity(first_columns.len());
    for column in first_columns {
        options.push(column_write_options(column));
    }

    for (row_group_idx, row_group) in row_groups.iter().enumerate().skip(1) {
        let columns = row_group.columns();
        if columns.len() != expected_columns {
            return Err(format!(
                "Row group {row_group_idx} column count mismatch: expected {expected_columns}, found {}",
                columns.len()
            )
            .into());
        }

        for (col_idx, column) in columns.iter().enumerate() {
            let current = column_write_options(column);
            if current != options[col_idx] {
                return Err(format!(
                    "Column settings for leaf column {col_idx} differ between row groups"
                )
                .into());
            }
        }
    }

    Ok(options)
}

fn column_write_options(column: &ColumnChunkMetaData) -> ColumnWriteOptions {
    ColumnWriteOptions {
        compression: column.compression(),
        dictionary_enabled: column_uses_dictionary(column),
        encoding: data_page_encoding(column),
        statistics_enabled: column_statistics_level(column),
    }
}

fn column_uses_dictionary(column: &ColumnChunkMetaData) -> bool {
    column.dictionary_page_offset().is_some() || column.encodings().any(is_dictionary_encoding)
}

fn column_statistics_level(column: &ColumnChunkMetaData) -> EnabledStatistics {
    if column.column_index_offset().is_some() {
        EnabledStatistics::Page
    } else if column.statistics().is_some() {
        EnabledStatistics::Chunk
    } else {
        EnabledStatistics::None
    }
}

fn data_page_encoding(column: &ColumnChunkMetaData) -> Option<Encoding> {
    if let Some(stats) = column.page_encoding_stats() {
        let mut counts: HashMap<Encoding, i32> = HashMap::new();
        for stat in stats.iter() {
            if !matches!(stat.page_type, PageType::DATA_PAGE | PageType::DATA_PAGE_V2) {
                continue;
            }
            if is_level_encoding(stat.encoding) || is_dictionary_encoding(stat.encoding) {
                continue;
            }
            *counts.entry(stat.encoding).or_insert(0) += stat.count;
        }
        if let Some((encoding, _)) = counts.into_iter().max_by_key(|(_, count)| *count) {
            return Some(encoding);
        }
    }

    let encodings = if let Some(mask) = column.page_encoding_stats_mask() {
        mask.encodings().collect::<Vec<_>>()
    } else {
        column.encodings().collect::<Vec<_>>()
    };

    let mut encoding = encodings
        .iter()
        .copied()
        .find(|encoding| !is_level_encoding(*encoding) && !is_dictionary_encoding(*encoding));

    if encoding.is_none() && encodings.contains(&Encoding::PLAIN) {
        encoding = Some(Encoding::PLAIN);
    }

    encoding
}

#[allow(deprecated)]
fn is_level_encoding(encoding: Encoding) -> bool {
    matches!(encoding, Encoding::RLE | Encoding::BIT_PACKED)
}

fn is_dictionary_encoding(encoding: Encoding) -> bool {
    matches!(
        encoding,
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
    )
}

struct ParquetIndexAppend<'a> {
    path: &'a Path,
    index: &'a crate::hnsw::index::HnswIndex,
    embedding_column: &'a EmbeddingColumn,
}

fn append_index_inplace(plan: ParquetIndexAppend<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let reader = SerializedFileReader::new(File::open(plan.path)?)?;
    let metadata = reader.metadata().clone();

    let mut file = OpenOptions::new().read(true).write(true).open(plan.path)?;
    let file_len = file.metadata()?.len();
    if file_len < FOOTER_SIZE as u64 {
        return Err("Parquet file too small to contain a footer".into());
    }

    file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
    let mut footer_bytes = [0u8; FOOTER_SIZE];
    file.read_exact(&mut footer_bytes)?;
    let footer_tail = FooterTail::try_new(&footer_bytes)?;
    if footer_tail.is_encrypted_footer() {
        return Err("Encrypted parquet footers are not supported for in-place indexing".into());
    }

    let metadata_len = footer_tail.metadata_length() as u64;
    if metadata_len + FOOTER_SIZE as u64 > file_len {
        return Err("Parquet footer length exceeds file size".into());
    }

    let metadata_end = file_len - FOOTER_SIZE as u64;
    let index_offset = metadata_end;

    let mut key_values = metadata
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default();
    key_values.retain(|kv| {
        kv.key != PQ_VECTOR_INDEX_OFFSET_KEY && kv.key != PQ_VECTOR_EMBEDDING_COLUMN_KEY
    });
    key_values.push(KeyValue::new(
        PQ_VECTOR_INDEX_OFFSET_KEY.to_string(),
        index_offset.to_string(),
    ));
    key_values.push(KeyValue::new(
        PQ_VECTOR_EMBEDDING_COLUMN_KEY.to_string(),
        plan.embedding_column.as_str().to_string(),
    ));

    let file_metadata = FileMetaData::new(
        metadata.file_metadata().version(),
        metadata.file_metadata().num_rows(),
        metadata.file_metadata().created_by().map(str::to_string),
        Some(key_values),
        metadata.file_metadata().schema_descr_ptr(),
        metadata.file_metadata().column_orders().cloned(),
    );

    let new_metadata = ParquetMetaDataBuilder::new(file_metadata)
        .set_row_groups(metadata.row_groups().to_vec())
        .build();

    file.seek(SeekFrom::Start(metadata_end))?;

    let index_json = plan.index.to_json_bytes()?;
    let index_len = index_json.len() as u64;
    file.write_all(PQ_VECTOR_HNSW_MAGIC)?;
    file.write_all(&index_len.to_le_bytes())?;
    file.write_all(&index_json)?;

    let writer = ParquetMetaDataWriter::new(&mut file, &new_metadata);
    writer.finish()?;
    file.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::types::Float32Type;
    use arrow::array::{Int32Array, ListArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn test_hnsw_build_index_inplace_appends_footer() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("data.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vec",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]));

        let ids = Int32Array::from(vec![0, 1, 2]);
        let vectors = vec![
            Some(vec![Some(0.0), Some(0.0)]),
            Some(vec![Some(1.0), Some(0.0)]),
            Some(vec![Some(0.0), Some(2.0)]),
        ];
        let vec_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vectors);
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vec_array)]).unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let original_size = std::fs::metadata(&path).unwrap().len();
        HnswBuilder::new(&path, "vec").build_inplace().unwrap();
        let new_size = std::fs::metadata(&path).unwrap().len();
        assert!(new_size > original_size);

        let (index, embedding_column) = read_index_from_parquet(&path).unwrap();
        assert_eq!(embedding_column.as_str(), "vec");
        assert_eq!(index.dim(), 2);
    }
}
