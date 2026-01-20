//! IVF (Inverted File) index embedded in Parquet files.
//!
//! The IVF index is stored directly in the parquet file body after the data pages,
//! with the index offset recorded in the footer metadata. This allows standard
//! parquet readers to ignore the index while specialized readers can use it.

use arrow::array::{Array, Float32Array, ListArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use rand::Rng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Magic bytes to identify our IVF index format
const IVF_INDEX_MAGIC: &[u8] = b"IVF1";

/// Metadata key for the index offset
const IVF_INDEX_OFFSET_KEY: &str = "ivf_index_offset";

/// Metadata key for the embedding column name
const IVF_EMBEDDING_COLUMN_KEY: &str = "ivf_embedding_column";

/// Parameters for building an IVF index
#[derive(Debug, Clone)]
pub struct IvfBuildParams {
    /// Number of clusters. If None, uses sqrt(n).
    pub n_clusters: Option<usize>,
    /// Max iterations for k-means
    pub max_iters: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for IvfBuildParams {
    fn default() -> Self {
        Self {
            n_clusters: None,
            max_iters: 20,
            seed: 42,
        }
    }
}

/// Result item from topk search
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub row_idx: u32,
    pub distance: f32,
}

/// IVF index structure (in-memory representation)
struct IvfIndex {
    dim: usize,
    n_clusters: usize,
    centroids: Vec<f32>,
    inverted_lists: Vec<Vec<u32>>,
}

// For max-heap (we want to pop largest distances)
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

impl IvfIndex {
    /// Serialize the index to bytes
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write header
        bytes.extend_from_slice(&(self.dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.n_clusters as u32).to_le_bytes());

        // Write centroids
        for &val in &self.centroids {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // Write inverted lists
        for list in &self.inverted_lists {
            bytes.extend_from_slice(&(list.len() as u32).to_le_bytes());
            for &idx in list {
                bytes.extend_from_slice(&idx.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize index from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut offset = 0;

        let dim = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
        offset += 4;

        let n_clusters = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
        offset += 4;

        // Read centroids
        let centroids_len = n_clusters * dim;
        let mut centroids = Vec::with_capacity(centroids_len);
        for _ in 0..centroids_len {
            let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
            centroids.push(val);
            offset += 4;
        }

        // Read inverted lists
        let mut inverted_lists = Vec::with_capacity(n_clusters);
        for _ in 0..n_clusters {
            let list_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
            offset += 4;

            let mut list = Vec::with_capacity(list_len);
            for _ in 0..list_len {
                let idx = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
                list.push(idx);
                offset += 4;
            }
            inverted_lists.push(list);
        }

        Ok(Self {
            dim,
            n_clusters,
            centroids,
            inverted_lists,
        })
    }

    /// Find the closest nprobe centroids to the query
    fn find_closest_centroids(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let nprobe = nprobe.min(self.n_clusters);

        let mut centroid_distances: Vec<(usize, f32)> = (0..self.n_clusters)
            .map(|i| {
                let centroid_start = i * self.dim;
                let centroid = &self.centroids[centroid_start..centroid_start + self.dim];
                let dist = squared_l2_distance(query, centroid);
                (i, dist)
            })
            .collect();

        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        centroid_distances
            .into_iter()
            .take(nprobe)
            .map(|(idx, _)| idx)
            .collect()
    }
}

/// Build an IVF index and embed it into a new parquet file.
///
/// This reads the source parquet file, builds an IVF index on the specified
/// embedding column, and writes a new parquet file with the index embedded.
pub fn build_index(
    source_path: &Path,
    output_path: &Path,
    embedding_column: &str,
    params: &IvfBuildParams,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read all data and embeddings from source
    let (batches, schema, embeddings, dim) =
        read_parquet_with_embeddings(source_path, embedding_column)?;

    let n_vectors = embeddings.len() / dim;
    let n_clusters = params
        .n_clusters
        .unwrap_or_else(|| (n_vectors as f64).sqrt().ceil() as usize);

    println!(
        "Building IVF index: {} vectors, dim={}, k={}",
        n_vectors, dim, n_clusters
    );

    // Run k-means clustering
    let (centroids, assignments) =
        kmeans(&embeddings, dim, n_clusters, params.max_iters, params.seed);

    // Build inverted lists
    let mut inverted_lists = vec![Vec::new(); n_clusters];
    for (row_idx, &cluster_idx) in assignments.iter().enumerate() {
        inverted_lists[cluster_idx].push(row_idx as u32);
    }

    // Print cluster sizes
    let sizes: Vec<usize> = inverted_lists.iter().map(|l| l.len()).collect();
    println!(
        "Cluster sizes: min={}, max={}, avg={:.1}",
        sizes.iter().min().unwrap_or(&0),
        sizes.iter().max().unwrap_or(&0),
        sizes.iter().sum::<usize>() as f64 / n_clusters as f64
    );

    let index = IvfIndex {
        dim,
        n_clusters,
        centroids,
        inverted_lists,
    };

    // Write output parquet with embedded index
    write_parquet_with_index(output_path, &batches, schema, &index, embedding_column)?;

    println!("Index embedded into {:?}", output_path);
    Ok(())
}

/// Search for top-k nearest neighbors in a parquet file with embedded IVF index.
///
/// # Arguments
/// * `parquet_path` - Path to parquet file with embedded IVF index
/// * `query` - Query vector
/// * `k` - Number of results to return
/// * `nprobe` - Number of clusters to search (higher = more accurate but slower)
pub fn topk(
    parquet_path: &Path,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    // Read index from parquet file
    let (index, embedding_column) = read_index_from_parquet(parquet_path)?;

    assert_eq!(
        query.len(),
        index.dim,
        "Query dimension mismatch: expected {}, got {}",
        index.dim,
        query.len()
    );

    // Find closest centroids
    let closest_clusters = index.find_closest_centroids(query, nprobe);

    // Read embeddings for the rows we need to check
    let rows_to_check: Vec<u32> = closest_clusters
        .iter()
        .flat_map(|&c| index.inverted_lists[c].iter().copied())
        .collect();

    // Read embeddings from parquet
    let embeddings = read_embeddings_for_rows(parquet_path, &embedding_column, &rows_to_check)?;

    // Use max-heap to track top-k
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

    for (i, &row_idx) in rows_to_check.iter().enumerate() {
        let vec = &embeddings[i * index.dim..(i + 1) * index.dim];
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

    // Convert heap to sorted results
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

/// Read parquet file and extract embeddings
fn read_parquet_with_embeddings(
    path: &Path,
    embedding_column: &str,
) -> Result<(Vec<RecordBatch>, SchemaRef, Vec<f32>, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let mut batches = Vec::new();
    let mut all_embeddings = Vec::new();
    let mut dim = 0;

    for batch in reader {
        let batch = batch?;

        // Extract embeddings
        let embedding_col = batch
            .column_by_name(embedding_column)
            .ok_or_else(|| format!("Column '{}' not found", embedding_column))?;

        let list_array = embedding_col
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("Embedding column is not a list array")?;

        let values = list_array.values();
        let float_array = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or("Embedding values are not float32")?;

        if dim == 0 && list_array.len() > 0 {
            dim = float_array.len() / list_array.len();
        }

        for i in 0..float_array.len() {
            all_embeddings.push(float_array.value(i));
        }

        batches.push(batch);
    }

    Ok((batches, schema, all_embeddings, dim))
}

/// Write parquet file with embedded IVF index
fn write_parquet_with_index(
    path: &Path,
    batches: &[RecordBatch],
    schema: SchemaRef,
    index: &IvfIndex,
    embedding_column: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Calculate page size for embedding column: one vector per page
    // This enables random access to individual vectors without decompressing large pages
    let vector_size = index.dim * std::mem::size_of::<f32>();

    // Configure writer properties:
    // - Set data page size to vector size for better random access
    // - Disable compression on embedding column for direct access
    // - Enable page index for efficient seeking
    let embedding_col_path = parquet::schema::types::ColumnPath::new(vec![
        embedding_column.to_string(),
        "list".to_string(),
        "item".to_string(),
    ]);

    let props = WriterProperties::builder()
        // Set data page size to fit roughly one vector
        .set_data_page_size_limit(vector_size)
        // One row per page for better random access
        .set_data_page_row_count_limit(1)
        // Disable compression for embedding column to allow direct random access
        .set_column_compression(embedding_col_path, Compression::UNCOMPRESSED)
        .build();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

    // Write all data batches
    for batch in batches {
        writer.write(batch)?;
    }

    // Flush data to ensure bytes_written is accurate
    writer.flush()?;

    // Get offset where we'll write the index
    let index_offset = writer.bytes_written();

    // Serialize index
    let index_bytes = index.to_bytes();
    let index_len = index_bytes.len() as u64;

    // Write magic, length, and index bytes
    writer.write_all(IVF_INDEX_MAGIC)?;
    writer.write_all(&index_len.to_le_bytes())?;
    writer.write_all(&index_bytes)?;

    // Add metadata about the index location
    writer.append_key_value_metadata(KeyValue::new(
        IVF_INDEX_OFFSET_KEY.to_string(),
        index_offset.to_string(),
    ));
    writer.append_key_value_metadata(KeyValue::new(
        IVF_EMBEDDING_COLUMN_KEY.to_string(),
        embedding_column.to_string(),
    ));

    writer.close()?;

    println!(
        "Wrote index at offset {}, size {} bytes (vector_size={})",
        index_offset, index_len, vector_size
    );

    Ok(())
}

/// Read IVF index from parquet file
fn read_index_from_parquet(path: &Path) -> Result<(IvfIndex, String), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file.try_clone()?)?;
    let metadata = reader.metadata().file_metadata();

    // Get index offset from metadata
    let offset_str = metadata
        .key_value_metadata()
        .and_then(|kv| kv.iter().find(|k| k.key == IVF_INDEX_OFFSET_KEY))
        .and_then(|k| k.value.clone())
        .ok_or("Missing IVF index offset in metadata")?;

    let offset: u64 = offset_str.parse()?;

    // Get embedding column name
    let embedding_column = metadata
        .key_value_metadata()
        .and_then(|kv| kv.iter().find(|k| k.key == IVF_EMBEDDING_COLUMN_KEY))
        .and_then(|k| k.value.clone())
        .ok_or("Missing embedding column name in metadata")?;

    // Read index from file
    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;

    // Read magic
    let mut magic_buf = [0u8; 4];
    file.read_exact(&mut magic_buf)?;
    if magic_buf != IVF_INDEX_MAGIC {
        return Err(format!("Invalid IVF index magic at offset {}", offset).into());
    }

    // Read length
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let index_len = u64::from_le_bytes(len_buf) as usize;

    // Read index bytes
    let mut index_bytes = vec![0u8; index_len];
    file.read_exact(&mut index_bytes)?;

    let index = IvfIndex::from_bytes(&index_bytes)?;

    println!(
        "Read IVF index: {} clusters, dim={}",
        index.n_clusters, index.dim
    );

    Ok((index, embedding_column))
}

/// Read embeddings for specific rows
fn read_embeddings_for_rows(
    path: &Path,
    embedding_column: &str,
    rows: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // For simplicity, read all embeddings and filter
    // A production implementation would use row group/page index for efficiency
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut all_embeddings = Vec::new();
    let mut dim = 0;

    for batch in reader {
        let batch = batch?;
        let embedding_col = batch.column_by_name(embedding_column).unwrap();
        let list_array = embedding_col.as_any().downcast_ref::<ListArray>().unwrap();
        let values = list_array.values();
        let float_array = values.as_any().downcast_ref::<Float32Array>().unwrap();

        if dim == 0 && list_array.len() > 0 {
            dim = float_array.len() / list_array.len();
        }

        for i in 0..float_array.len() {
            all_embeddings.push(float_array.value(i));
        }
    }

    // Extract only the rows we need
    let mut result = Vec::with_capacity(rows.len() * dim);
    for &row_idx in rows {
        let start = row_idx as usize * dim;
        let end = start + dim;
        result.extend_from_slice(&all_embeddings[start..end]);
    }

    Ok(result)
}

/// Compute squared L2 distance between two vectors
#[inline]
fn squared_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// K-means clustering implementation with k-means++ initialization
fn kmeans(
    data: &[f32],
    dim: usize,
    k: usize,
    max_iters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<usize>) {
    let n = data.len() / dim;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Initialize centroids using k-means++
    let mut centroids = vec![0.0f32; k * dim];

    // Pick first centroid randomly
    let first_idx = rng.gen_range(0..n);
    centroids[..dim].copy_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

    // Pick remaining centroids with probability proportional to distance
    for i in 1..k {
        let distances: Vec<f32> = (0..n)
            .map(|j| {
                let vec = &data[j * dim..(j + 1) * dim];
                (0..i)
                    .map(|c| {
                        let centroid = &centroids[c * dim..(c + 1) * dim];
                        squared_l2_distance(vec, centroid)
                    })
                    .fold(f32::INFINITY, |a, b| a.min(b))
            })
            .collect();

        let total: f32 = distances.iter().sum();
        if total > 0.0 {
            let threshold = rng.gen_range(0.0..1.0) * total;
            let mut cumsum = 0.0;
            for (j, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    centroids[i * dim..(i + 1) * dim]
                        .copy_from_slice(&data[j * dim..(j + 1) * dim]);
                    break;
                }
            }
        } else {
            let idx = rng.gen_range(0..n);
            centroids[i * dim..(i + 1) * dim].copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
        }
    }

    let mut assignments = vec![0usize; n];
    let mut cluster_sizes = vec![0usize; k];

    for iter in 0..max_iters {
        // Assignment step
        let mut changed = 0;
        cluster_sizes.fill(0);

        for i in 0..n {
            let vec = &data[i * dim..(i + 1) * dim];
            let mut best_cluster = 0;
            let mut best_dist = f32::INFINITY;

            for j in 0..k {
                let centroid = &centroids[j * dim..(j + 1) * dim];
                let dist = squared_l2_distance(vec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = j;
                }
            }

            if assignments[i] != best_cluster {
                changed += 1;
            }
            assignments[i] = best_cluster;
            cluster_sizes[best_cluster] += 1;
        }

        println!("K-means iter {}: {} assignments changed", iter + 1, changed);

        if changed == 0 {
            break;
        }

        // Update step
        centroids.fill(0.0);

        for i in 0..n {
            let cluster = assignments[i];
            let vec = &data[i * dim..(i + 1) * dim];
            for (j, &val) in vec.iter().enumerate() {
                centroids[cluster * dim + j] += val;
            }
        }

        for j in 0..k {
            if cluster_sizes[j] > 0 {
                let size = cluster_sizes[j] as f32;
                for d in 0..dim {
                    centroids[j * dim + d] /= size;
                }
            }
        }
    }

    (centroids, assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_l2_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = squared_l2_distance(&a, &b);
        assert!((dist - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_serialization() {
        let index = IvfIndex {
            dim: 3,
            n_clusters: 2,
            centroids: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            inverted_lists: vec![vec![0, 2, 4], vec![1, 3]],
        };

        let bytes = index.to_bytes();
        let restored = IvfIndex::from_bytes(&bytes).unwrap();

        assert_eq!(restored.dim, index.dim);
        assert_eq!(restored.n_clusters, index.n_clusters);
        assert_eq!(restored.centroids, index.centroids);
        assert_eq!(restored.inverted_lists, index.inverted_lists);
    }
}
