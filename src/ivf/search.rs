use crate::ivf::index::squared_l2_distance;
use crate::ivf::{EmbeddingColumn, EmbeddingDim, IvfIndex, read_index_from_parquet};
use arrow::array::Array;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, RowSelection, RowSelector};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::path::Path;

// For max-heap (we want to pop largest distances).
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

/// Result item from top-k search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub row_idx: u32,
    pub distance: f32,
}

/// Builder for top-k nearest neighbor search.
#[derive(Debug, Clone)]
pub struct TopkBuilder<'a> {
    parquet_path: &'a Path,
    query: &'a [f32],
    k: Option<NonZeroUsize>,
    nprobe: Option<NonZeroUsize>,
}

impl<'a> TopkBuilder<'a> {
    pub fn new(parquet_path: &'a Path, query: &'a [f32]) -> Self {
        Self {
            parquet_path,
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
        topk(self.parquet_path, self.query, k, nprobe).await
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
        } else if let Some(top) = heap.peek() {
            if distance < top.distance {
                heap.pop();
                heap.push(HeapItem { row_idx, distance });
            }
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
