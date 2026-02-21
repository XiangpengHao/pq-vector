use crate::ivf::index::squared_l2_distance;
use crate::ivf::{EmbeddingColumn, EmbeddingDim, IvfIndex, read_index_from_parquet};
use arrow::array::Array;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, RowSelection, RowSelector};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct FileIndex {
    path: PathBuf,
    dim: EmbeddingDim,
}

// For max-heap (we want to pop largest distances).
#[derive(Debug, Clone)]
struct HeapItem {
    row_idx: u32,
    distance: f32,
}

#[derive(Debug)]
struct MergeHeapItem {
    distance: f32,
    file_idx: usize,
    result_idx: usize,
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

impl PartialEq for MergeHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MergeHeapItem {}

impl PartialOrd for MergeHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Result item from top-k search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub row_idx: u32,
    pub distance: f32,
}

/// Result item from top-k search across multiple files.
#[derive(Debug, Clone)]
pub struct MultiSearchResult {
    pub file_path: PathBuf,
    pub row_idx: u32,
    pub distance: f32,
}

/// Multi-file IVF index referencing multiple indexed Parquet files.
#[derive(Debug, Clone)]
pub struct MultiFileIndex {
    files: Vec<FileIndex>,
}

impl MultiFileIndex {
    fn new(
        paths: Vec<PathBuf>,
        embedding_column: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if paths.is_empty() {
            return Err("No parquet files provided".into());
        }

        let expected_embedding_column = EmbeddingColumn::try_from(embedding_column)?;
        let mut files = Vec::with_capacity(paths.len());
        let mut expected_dim: Option<EmbeddingDim> = None;

        for path in paths {
            let (index, file_embedding_column) = read_index_from_parquet(path.as_path())?;
            if file_embedding_column != expected_embedding_column {
                return Err(format!(
                    "Embedding column mismatch in '{}': expected '{}', found '{}'",
                    path.display(),
                    expected_embedding_column.as_str(),
                    file_embedding_column.as_str()
                )
                .into());
            }

            let dim = EmbeddingDim::new(index.dim())?;
            if let Some(expected) = expected_dim {
                if expected != dim {
                    return Err(format!(
                        "Embedding dimension mismatch in '{}': expected {}, found {}",
                        path.display(),
                        expected.as_usize(),
                        dim.as_usize()
                    )
                    .into());
                }
            } else {
                expected_dim = Some(dim);
            }

            files.push(FileIndex { path, dim });
        }

        Ok(Self { files })
    }

    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<MultiSearchResult>, Box<dyn std::error::Error>> {
        let k = NonZeroUsize::new(k).ok_or("k must be > 0")?;
        let nprobe = NonZeroUsize::new(nprobe).ok_or("nprobe must be > 0")?;

        let expected_dim = self.files[0].dim;
        if query.len() != expected_dim.as_usize() {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                expected_dim.as_usize(),
                query.len()
            )
            .into());
        }

        let mut per_file_results: Vec<Vec<SearchResult>> = Vec::with_capacity(self.files.len());
        for file in &self.files {
            let results = topk(file.path.as_path(), query, k, nprobe).await?;
            per_file_results.push(results);
        }

        let mut heap: BinaryHeap<MergeHeapItem> = BinaryHeap::new();
        let mut total_results = 0usize;
        for (file_idx, results) in per_file_results.iter().enumerate() {
            total_results += results.len();
            if let Some(first) = results.first() {
                heap.push(MergeHeapItem {
                    distance: first.distance,
                    file_idx,
                    result_idx: 0,
                });
            }
        }

        let mut merged = Vec::with_capacity(k.get().min(total_results));
        while merged.len() < k.get() {
            let item = match heap.pop() {
                Some(item) => item,
                None => break,
            };

            let result = &per_file_results[item.file_idx][item.result_idx];
            merged.push(MultiSearchResult {
                file_path: self.files[item.file_idx].path.clone(),
                row_idx: result.row_idx,
                distance: result.distance,
            });

            let next_idx = item.result_idx + 1;
            if next_idx < per_file_results[item.file_idx].len() {
                let next = &per_file_results[item.file_idx][next_idx];
                heap.push(MergeHeapItem {
                    distance: next.distance,
                    file_idx: item.file_idx,
                    result_idx: next_idx,
                });
            }
        }

        Ok(merged)
    }
}

/// Builder for a multi-file IVF index.
#[derive(Debug, Clone)]
pub struct MultiFileIndexBuilder {
    files: Vec<PathBuf>,
    embedding_column: String,
}

impl MultiFileIndexBuilder {
    pub fn new(embedding_column: impl AsRef<str>) -> Self {
        Self {
            files: Vec::new(),
            embedding_column: embedding_column.as_ref().to_string(),
        }
    }

    pub fn add_file(mut self, path: impl AsRef<Path>) -> Self {
        self.files.push(path.as_ref().to_path_buf());
        self
    }

    /// Build IVF indexes in-place for all configured files.
    pub fn build_indexes_inplace(&self) -> Result<(), Box<dyn std::error::Error>> {
        for file in &self.files {
            super::parquet::IndexBuilder::new(file.as_path(), self.embedding_column.as_str())
                .build_inplace()?;
        }
        Ok(())
    }

    pub fn build(self) -> Result<MultiFileIndex, Box<dyn std::error::Error>> {
        MultiFileIndex::new(self.files, self.embedding_column.as_str())
    }
}

/// Builder for top-k nearest neighbor search.
#[derive(Debug, Clone)]
pub struct TopkBuilder<'a> {
    parquet_path: PathBuf,
    query: &'a [f32],
    k: Option<NonZeroUsize>,
    nprobe: Option<NonZeroUsize>,
}

impl<'a> TopkBuilder<'a> {
    pub fn new(parquet_path: impl AsRef<Path>, query: &'a [f32]) -> Self {
        Self {
            parquet_path: parquet_path.as_ref().to_path_buf(),
            query,
            k: None,
            nprobe: None,
        }
    }

    pub fn k(mut self, k: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.k = Some(NonZeroUsize::new(k).ok_or("k must be > 0")?);
        Ok(self)
    }

    pub fn nprobe(mut self, nprobe: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.nprobe = Some(NonZeroUsize::new(nprobe).ok_or("nprobe must be > 0")?);
        Ok(self)
    }

    pub async fn search(self) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let k = self.k.ok_or("k must be set")?;
        let nprobe = self.nprobe.ok_or("nprobe must be set")?;
        topk(self.parquet_path.as_path(), self.query, k, nprobe).await
    }
}

async fn topk(
    parquet_path: &Path,
    query: &[f32],
    k: NonZeroUsize,
    nprobe: NonZeroUsize,
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

    let rows_to_check: Vec<u32> = index.candidate_rows(query, nprobe.get());

    let embeddings = read_embeddings_for_rows(
        EmbeddingReadContext {
            path: parquet_path,
            embedding_column: &embedding_column,
            dim: index_dim(&index),
        },
        &rows_to_check,
    )
    .await?;

    let k = k.get();
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

    for (i, &row_idx) in rows_to_check.iter().enumerate() {
        let vec = &embeddings[i * index.dim()..(i + 1) * index.dim()];
        let distance = squared_l2_distance(query, vec);

        if heap.len() < k {
            heap.push(HeapItem { row_idx, distance });
        } else if let Some(top) = heap.peek()
            && distance < top.distance
        {
            heap.pop();
            heap.push(HeapItem { row_idx, distance });
        }
    }

    let mut results: Vec<SearchResult> = heap
        .into_iter()
        .map(|item| SearchResult {
            row_idx: item.row_idx,
            distance: item.distance.sqrt(),
        })
        .collect();
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });
    Ok(results)
}

struct EmbeddingReadContext<'a> {
    path: &'a std::path::Path,
    embedding_column: &'a EmbeddingColumn,
    dim: EmbeddingDim,
}

fn index_dim(index: &IvfIndex) -> EmbeddingDim {
    EmbeddingDim::new(index.dim()).expect("IVF index dimension must be > 0")
}

/// Read embeddings for specific rows using direct page reads.
async fn read_embeddings_for_rows(
    context: EmbeddingReadContext<'_>,
    rows: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let mut sorted_rows: Vec<u32> = rows.to_vec();
    sorted_rows.sort_unstable();

    let file = tokio::fs::File::open(context.path).await?;

    let options = ArrowReaderOptions::new().with_page_index(true);
    let builder = parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new_with_options(
        file, options,
    )
    .await?;

    let schema = builder.schema();
    let embedding_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == context.embedding_column.as_str())
        .ok_or_else(|| format!("Column '{}' not found", context.embedding_column.as_str()))?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), [embedding_col_idx]);

    let total_rows = builder.metadata().file_metadata().num_rows() as usize;
    let mut selectors = Vec::new();
    let mut current_pos = 0;

    for &row in &sorted_rows {
        let row = row as usize;
        if row > current_pos {
            selectors.push(RowSelector::skip(row - current_pos));
        }
        selectors.push(RowSelector::select(1));
        current_pos = row + 1;
    }
    if current_pos < total_rows {
        selectors.push(RowSelector::skip(total_rows - current_pos));
    }

    let selection = RowSelection::from(selectors);

    let mut stream = builder
        .with_projection(projection)
        .with_row_selection(selection)
        .build()?;

    use arrow::array::{Float32Array, ListArray};
    use futures::StreamExt;
    let mut sorted_embeddings = Vec::with_capacity(sorted_rows.len() * context.dim.as_usize());
    while let Some(batch) = stream.next().await {
        let batch: arrow::record_batch::RecordBatch = batch?;
        let embedding_col = batch.column(0);
        let list_array = embedding_col.as_any().downcast_ref::<ListArray>().unwrap();
        if list_array.null_count() > 0 {
            return Err("Embedding column contains null rows".into());
        }
        let values = list_array.values();
        let float_array = values.as_any().downcast_ref::<Float32Array>().unwrap();
        if float_array.null_count() > 0 {
            return Err("Embedding values contain nulls".into());
        }

        for i in 0..float_array.len() {
            sorted_embeddings.push(float_array.value(i));
        }
    }

    if sorted_embeddings.len() != sorted_rows.len() * context.dim.as_usize() {
        return Err("Selected embeddings do not match expected dimensions".into());
    }

    let row_to_sorted_idx: std::collections::HashMap<u32, usize> = sorted_rows
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i))
        .collect();

    let mut result = Vec::with_capacity(rows.len() * context.dim.as_usize());
    for &row in rows {
        let sorted_idx = row_to_sorted_idx[&row];
        let start = sorted_idx * context.dim.as_usize();
        result.extend_from_slice(&sorted_embeddings[start..start + context.dim.as_usize()]);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::MultiFileIndexBuilder;
    use crate::ivf::IndexBuilder;
    use arrow::array::types::Float32Type;
    use arrow::array::{Array, Int32Array, ListArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn build_indexed_file(
        path: &std::path::Path,
        vectors: Vec<Vec<f32>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vec",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]));

        let ids = Int32Array::from_iter_values(0..(vectors.len() as i32));
        let vector_values = vectors
            .into_iter()
            .map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let vectors = ListArray::from_iter_primitive::<Float32Type, _, _>(vector_values);

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vectors)])?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        IndexBuilder::new(path, "vec").build_inplace()?;
        Ok(())
    }

    #[tokio::test]
    async fn multi_file_index_search_merges_results() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let file_a = temp_dir.path().join("a.parquet");
        let file_b = temp_dir.path().join("b.parquet");

        build_indexed_file(&file_a, vec![vec![0.0, 0.0], vec![2.0, 2.0]])?;
        build_indexed_file(&file_b, vec![vec![0.1, 0.1], vec![0.2, 0.2]])?;

        let index = MultiFileIndexBuilder::new("vec")
            .add_file(&file_a)
            .add_file(&file_b)
            .build()?;

        let results = index.search(&[0.0, 0.0], 3, 64).await?;
        let expected: Vec<(std::path::PathBuf, u32)> = vec![
            (file_a.clone(), 0),
            (file_b.clone(), 0),
            (file_b.clone(), 1),
        ];

        let actual: Vec<(std::path::PathBuf, u32)> = results
            .into_iter()
            .map(|item| (item.file_path, item.row_idx))
            .collect();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn multi_file_index_builder_rejects_empty_file_list() {
        let result = MultiFileIndexBuilder::new("vec").build();
        assert!(result.is_err());
    }

    #[test]
    fn multi_file_index_builder_rejects_different_embedding_columns() {
        let temp_dir = TempDir::new().unwrap();
        let first = temp_dir.path().join("first.parquet");
        let second = temp_dir.path().join("second.parquet");

        build_alt_column_file(first.as_path(), "vec", vec![vec![0.0, 0.0], vec![1.0, 1.0]])
            .unwrap();
        build_alt_column_file(
            second.as_path(),
            "embedding",
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        )
        .unwrap();

        let result = MultiFileIndexBuilder::new("vec")
            .add_file(first)
            .add_file(second)
            .build();
        assert!(result.is_err());
    }

    fn build_alt_column_file(
        path: &std::path::Path,
        column: &str,
        vectors: Vec<Vec<f32>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                column,
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]));

        let ids = Int32Array::from_iter_values(0..(vectors.len() as i32));
        let vector_values = vectors
            .into_iter()
            .map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let vectors = ListArray::from_iter_primitive::<Float32Type, _, _>(vector_values);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vectors)])?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        IndexBuilder::new(path, column).build_inplace()?;
        Ok(())
    }
}
