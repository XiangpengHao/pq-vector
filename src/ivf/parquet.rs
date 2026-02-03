use crate::ivf::index::{IvfBuildConfig, build_ivf_index};
use crate::ivf::{ClusterCount, EmbeddingColumn, EmbeddingDim, Embeddings, IvfIndex};
use arrow::array::{Array, Float32Array, ListArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::FOOTER_SIZE;
use parquet::file::metadata::{
    FileMetaData, FooterTail, KeyValue, ParquetMetaDataBuilder, ParquetMetaDataWriter,
};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Build an IVF index and embed it into a new parquet file.
#[derive(Debug, Clone)]
pub struct IndexBuilder {
    source: PathBuf,
    embedding_column: String,
    n_clusters: Option<usize>,
    max_iters: usize,
    seed: u64,
}

impl IndexBuilder {
    pub fn new(source: impl AsRef<Path>, embedding_column: impl AsRef<str>) -> Self {
        Self {
            source: source.as_ref().to_path_buf(),
            embedding_column: embedding_column.as_ref().to_string(),
            n_clusters: None,
            max_iters: 7,
            seed: 42,
        }
    }

    pub fn n_clusters(
        mut self,
        n_clusters: usize,
    ) -> Self {
        self.n_clusters = Some(n_clusters);
        self
    }

    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
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
        let index = build_ivf_index(&parquet.embeddings, config)?;
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
        let index = build_ivf_index(&parquet.embeddings, config)?;
        let plan = ParquetWritePlan {
            path: output.as_ref(),
            batches: &parquet.batches,
            schema: parquet.schema,
            index: &index,
            embedding_column: &embedding_column,
        };
        write_parquet_with_index(plan)?;
        Ok(())
    }

    fn build_config(&self) -> Result<IvfBuildConfig, Box<dyn std::error::Error>> {
        if self.max_iters == 0 {
            return Err("max_iters must be > 0".into());
        }
        let n_clusters = match self.n_clusters {
            Some(0) => return Err("n_clusters must be > 0".into()),
            Some(value) => Some(ClusterCount::new(value)?),
            None => None,
        };
        Ok(IvfBuildConfig {
            n_clusters,
            max_iters: self.max_iters,
            seed: self.seed,
        })
    }
}

/// Magic bytes to identify our IVF index format.
const IVF_INDEX_MAGIC: &[u8] = b"IVF1";

/// Metadata key for the index offset.
const IVF_INDEX_OFFSET_KEY: &str = "ivf_index_offset";

/// Metadata key for the embedding column name.
const IVF_EMBEDDING_COLUMN_KEY: &str = "ivf_embedding_column";

#[derive(Debug, Clone)]
struct IndexMetadata {
    offset: u64,
    embedding_column: EmbeddingColumn,
}

fn parse_index_metadata(
    metadata: &FileMetaData,
) -> Result<Option<IndexMetadata>, Box<dyn std::error::Error>> {
    let Some(kv) = metadata.key_value_metadata() else {
        return Ok(None);
    };
    let offset = kv
        .iter()
        .find(|k| k.key == IVF_INDEX_OFFSET_KEY)
        .and_then(|k| k.value.clone());
    let embedding = kv
        .iter()
        .find(|k| k.key == IVF_EMBEDDING_COLUMN_KEY)
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

pub(crate) fn read_index_metadata(
    path: &Path,
) -> Result<Option<EmbeddingColumn>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata().file_metadata();
    let metadata = parse_index_metadata(metadata)?;
    Ok(metadata.map(|meta| meta.embedding_column))
}

pub(crate) fn read_index_from_parquet(
    path: &Path,
) -> Result<(IvfIndex, EmbeddingColumn), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file.try_clone()?)?;
    let metadata = reader.metadata().file_metadata();
    let metadata =
        parse_index_metadata(metadata)?.ok_or("Missing IVF index metadata in parquet footer")?;
    let offset = metadata.offset;
    let embedding_column = metadata.embedding_column;

    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;

    let mut magic_buf = [0u8; 4];
    file.read_exact(&mut magic_buf)?;
    if magic_buf != IVF_INDEX_MAGIC {
        return Err(format!("Invalid IVF index magic at offset {}", offset).into());
    }

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let index_len = u64::from_le_bytes(len_buf) as usize;

    let mut index_bytes = vec![0u8; index_len];
    file.read_exact(&mut index_bytes)?;

    let index = IvfIndex::from_bytes(&index_bytes)?;

    Ok((index, embedding_column))
}

struct ParquetEmbeddings {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    embeddings: Embeddings,
}

fn read_parquet_with_embeddings(
    path: &Path,
    embedding_column: &EmbeddingColumn,
) -> Result<ParquetEmbeddings, Box<dyn std::error::Error>> {
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
        let float_array = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or("Embedding values are not float32")?;

        if float_array.null_count() > 0 {
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

        for i in 0..float_array.len() {
            all_embeddings.push(float_array.value(i));
        }

        batches.push(batch);
    }

    let dim = dim.ok_or("Embedding column has no rows")?;
    let embeddings = Embeddings::new(all_embeddings, dim)?;

    Ok(ParquetEmbeddings {
        batches,
        schema,
        embeddings,
    })
}

struct ParquetWritePlan<'a> {
    path: &'a Path,
    batches: &'a [RecordBatch],
    schema: SchemaRef,
    index: &'a IvfIndex,
    embedding_column: &'a EmbeddingColumn,
}

fn write_parquet_with_index(plan: ParquetWritePlan<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let vector_size = plan.index.dim() * std::mem::size_of::<f32>();

    let embedding_col_path = parquet::schema::types::ColumnPath::new(vec![
        plan.embedding_column.as_str().to_string(),
        "list".to_string(),
        "element".to_string(),
    ]);

    let props = WriterProperties::builder()
        .set_data_page_size_limit(vector_size)
        .set_data_page_row_count_limit(1)
        .set_column_compression(embedding_col_path.clone(), Compression::LZ4_RAW)
        .set_column_dictionary_enabled(embedding_col_path, false)
        .build();

    let file = File::create(plan.path)?;
    let mut writer = ArrowWriter::try_new(file, plan.schema, Some(props))?;

    for batch in plan.batches {
        writer.write(batch)?;
    }

    writer.flush()?;

    let index_offset = writer.bytes_written();

    let index_bytes = plan.index.to_bytes();
    let index_len = index_bytes.len() as u64;

    writer.write_all(IVF_INDEX_MAGIC)?;
    writer.write_all(&index_len.to_le_bytes())?;
    writer.write_all(&index_bytes)?;

    writer.append_key_value_metadata(KeyValue::new(
        IVF_INDEX_OFFSET_KEY.to_string(),
        index_offset.to_string(),
    ));
    writer.append_key_value_metadata(KeyValue::new(
        IVF_EMBEDDING_COLUMN_KEY.to_string(),
        plan.embedding_column.as_str().to_string(),
    ));

    writer.close()?;

    Ok(())
}

struct ParquetIndexAppend<'a> {
    path: &'a Path,
    index: &'a IvfIndex,
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
    key_values.retain(|kv| kv.key != IVF_INDEX_OFFSET_KEY && kv.key != IVF_EMBEDDING_COLUMN_KEY);
    key_values.push(KeyValue::new(
        IVF_INDEX_OFFSET_KEY.to_string(),
        index_offset.to_string(),
    ));
    key_values.push(KeyValue::new(
        IVF_EMBEDDING_COLUMN_KEY.to_string(),
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

    let index_bytes = plan.index.to_bytes();
    let index_len = index_bytes.len() as u64;
    file.write_all(IVF_INDEX_MAGIC)?;
    file.write_all(&index_len.to_le_bytes())?;
    file.write_all(&index_bytes)?;

    let writer = ParquetMetaDataWriter::new(&mut file, &new_metadata);
    writer.finish()?;
    file.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivf::IndexBuilder;
    use arrow::array::types::Float32Type;
    use arrow::array::{Int32Array, ListArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn test_build_index_inplace_appends_footer() {
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
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let original_size = std::fs::metadata(&path).unwrap().len();
        IndexBuilder::new(&path, "vec").build_inplace().unwrap();
        let new_size = std::fs::metadata(&path).unwrap().len();
        assert!(new_size > original_size);

        let (index, embedding_column) = read_index_from_parquet(&path).unwrap();
        assert_eq!(embedding_column.as_str(), "vec");
        assert_eq!(index.dim(), 2);
    }
}
