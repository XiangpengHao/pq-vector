use crate::hnsw::read_index_from_parquet;
use arrow::array::{Array, Float32Array, Float64Array, ListArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct HeapItem {
    row_idx: u32,
    distance: f32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub row_idx: u32,
    pub distance: f32,
}

#[derive(Debug, Clone)]
pub struct TopkBuilder<'a> {
    parquet_path: PathBuf,
    query: &'a [f32],
    k: Option<NonZeroUsize>,
    ef_search: Option<NonZeroUsize>,
}

impl<'a> TopkBuilder<'a> {
    pub fn new(parquet_path: impl AsRef<Path>, query: &'a [f32]) -> Self {
        Self {
            parquet_path: parquet_path.as_ref().to_path_buf(),
            query,
            k: None,
            ef_search: None,
        }
    }

    pub fn k(mut self, k: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.k = Some(NonZeroUsize::new(k).ok_or("k must be > 0")?);
        Ok(self)
    }

    pub fn ef_search(mut self, ef_search: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.ef_search = Some(NonZeroUsize::new(ef_search).ok_or("ef_search must be > 0")?);
        Ok(self)
    }

    pub async fn search(self) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let k = self.k.ok_or("k must be set")?;
        let ef_search = self.ef_search.ok_or("ef_search must be set")?;
        topk(self.parquet_path.as_path(), self.query, k, ef_search).await
    }
}

async fn topk(
    parquet_path: &Path,
    query: &[f32],
    k: NonZeroUsize,
    ef_search: NonZeroUsize,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let (index, embedding_column) = read_index_from_parquet(parquet_path)?;
    if query.len() != index.dim() {
        return Err(format!(
            "Query dimension mismatch: expected {}, got {}",
            index.dim(),
            query.len()
        )
        .into());
    }

    let embeddings = read_all_embeddings(parquet_path, embedding_column.as_str()).await?;
    if embeddings.dim() != index.dim() {
        return Err(format!(
            "Embedding dimension mismatch: expected {}, got {}",
            index.dim(),
            embeddings.dim()
        )
        .into());
    }
    let rows_to_check = index.candidate_rows(query, embeddings.data(), ef_search.get());
    if rows_to_check.is_empty() {
        return Ok(Vec::new());
    }

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k.get() + 1);
    for &row_idx in &rows_to_check {
        let row_idx = row_idx as usize;
        let vec_start = row_idx
            .checked_mul(index.dim())
            .ok_or("HNSW candidate row index overflows during embedding access")?;
        let vec_end = vec_start + index.dim();
        if vec_end > embeddings.data().len() {
            return Err("HNSW candidate row index is out of bounds".into());
        }
        let vec = &embeddings.data()[vec_start..vec_end];
        let distance = squared_l2_distance(query, vec);
        if heap.len() < k.get() {
            heap.push(HeapItem { row_idx: row_idx as u32, distance });
        } else if let Some(top) = heap.peek() && distance < top.distance {
            heap.pop();
            heap.push(HeapItem {
                row_idx: row_idx as u32,
                distance,
            });
        }
    }

    let mut results: Vec<SearchResult> = heap
        .into_iter()
        .map(|item| SearchResult {
            row_idx: item.row_idx,
            distance: item.distance.sqrt(),
        })
        .collect();
    results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    Ok(results)
}

fn squared_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let len = a.len();
    let mut i = 0usize;
    while i < len {
        let delta = a[i] - b[i];
        sum += delta * delta;
        i += 1;
    }
    sum
}

async fn read_all_embeddings(
    path: &Path,
    embedding_column: &str,
) -> Result<ReadEmbeddingsResult, Box<dyn std::error::Error>> {
    let file = tokio::fs::File::open(path).await?;
    let options = ArrowReaderOptions::new().with_page_index(true);
    let builder = parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new_with_options(
        file, options,
    )
    .await?;

    let schema = builder.schema();
    let embedding_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == embedding_column)
        .ok_or_else(|| format!("Column '{}' not found", embedding_column))?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), [embedding_idx]);

    let mut stream = builder.with_projection(projection).build()?;
    use futures::StreamExt;

    let mut dim: Option<usize> = None;
    let mut all_embeddings = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("Embedding column is not a list array")?;
        if column.null_count() > 0 {
            return Err("Embedding column contains null rows".into());
        }
        let values = column.values();
        let float_values = if let Some(array) = values.as_any().downcast_ref::<Float32Array>() {
            FloatSource::F32(array)
        } else if let Some(array) = values.as_any().downcast_ref::<Float64Array>() {
            FloatSource::F64(array)
        } else {
            return Err("Embedding values are not float32/float64".into());
        };

        if float_values.null_count() > 0 {
            return Err("Embedding values contain nulls".into());
        }
        for row in 0..column.len() {
            let row_len = column.value_length(row) as usize;
            if row_len == 0 {
                return Err("Embedding row has zero length".into());
            }
            let row_dim = row_len;
            if let Some(existing) = dim {
                if existing != row_dim {
                    return Err("Embedding vectors have inconsistent dimensions".into());
                }
            } else {
                dim = Some(row_dim);
            }
            let offsets = column.value_offsets();
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            for i in start..end {
                all_embeddings.push(float_values.value(i));
            }
        }
    }

    let dim = dim.ok_or("Embedding column has no rows")?;
    let data = ReadEmbeddingsResult { data: all_embeddings, dim };
    Ok(data)
}

enum FloatSource<'a> {
    F32(&'a Float32Array),
    F64(&'a Float64Array),
}

impl<'a> FloatSource<'a> {
    fn null_count(&self) -> usize {
        match self {
            FloatSource::F32(values) => values.null_count(),
            FloatSource::F64(values) => values.null_count(),
        }
    }

    fn value(&self, i: usize) -> f32 {
        match self {
            FloatSource::F32(values) => values.value(i),
            FloatSource::F64(values) => values.value(i) as f32,
        }
    }
}

struct ReadEmbeddingsResult {
    data: Vec<f32>,
    dim: usize,
}

impl ReadEmbeddingsResult {
    fn data(&self) -> &[f32] {
        &self.data
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn total_rows(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::parquet::HnswBuilder;
    use arrow::array::types::Float32Type;
    use arrow::array::{Int32Array, ListArray, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::TempDir;
    use crate::hnsw::index::{build_hnsw_index, HnswBuildConfig};
    use crate::ivf::EmbeddingDim;

    #[tokio::test]
    async fn test_hnsw_topk_search() {
        let temp_dir = TempDir::new().unwrap();
        let source = temp_dir.path().join("source.parquet");
        let indexed = temp_dir.path().join("indexed.parquet");

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
            Some(vec![Some(0.0), Some(1.0)]),
        ];
        let vec_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vectors);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vec_array)]).unwrap();

        let file = std::fs::File::create(&source).unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        HnswBuilder::new(&source, "vec")
            .build_new(&indexed)
            .unwrap();

        let query = vec![0.0, 0.0];
        let results = TopkBuilder::new(indexed, &query)
            .k(1)
            .unwrap()
            .ef_search(2)
            .unwrap()
            .search()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_hnsw_index_builder_smoke() {
        let data = crate::ivf::Embeddings::new(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            EmbeddingDim::new(2).unwrap(),
        )
        .unwrap();
        let config = HnswBuildConfig {
            m: 2,
            ef_construction: 4,
            ef_search: 4,
            seed: 42,
        };
        let index = build_hnsw_index(&data, config).unwrap();
        let candidates = index.candidate_rows(&[0.0, 0.0], data.data(), 3);
        assert!(!candidates.is_empty());
    }
}
