use crate::ivf::{EmbeddingDim, Embeddings};
use rand::Rng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::num::NonZeroU32;
use std::thread;

/// IVF index structure (in-memory representation).
pub(crate) struct IvfIndex {
    dim: EmbeddingDim,
    n_clusters: ClusterCount,
    centroids: Vec<f32>,
    inverted_lists: Vec<Vec<u32>>,
}

/// Non-zero cluster count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterCount(NonZeroU32);

impl ClusterCount {
    pub fn new(count: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let count_u32: u32 = count.try_into()?;
        let count =
            NonZeroU32::new(count_u32).ok_or_else(|| "Cluster count must be > 0".to_string())?;
        Ok(Self(count))
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0.get() as usize
    }

    pub(crate) fn as_u32(self) -> u32 {
        self.0.get()
    }
}

impl TryFrom<usize> for ClusterCount {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IvfBuildConfig {
    pub(crate) n_clusters: Option<ClusterCount>,
    pub(crate) max_iters: usize,
    pub(crate) seed: u64,
}

impl IvfIndex {
    pub(crate) fn dim(&self) -> usize {
        self.dim.as_usize()
    }

    pub(crate) fn candidate_rows(&self, query: &[f32], nprobe: usize) -> Vec<u32> {
        let closest_clusters = self.find_closest_centroids(query, nprobe);
        closest_clusters
            .iter()
            .flat_map(|&c| self.inverted_lists[c].iter().copied())
            .collect()
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&self.dim.as_u32().to_le_bytes());
        bytes.extend_from_slice(&self.n_clusters.as_u32().to_le_bytes());

        for &val in &self.centroids {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        for list in &self.inverted_lists {
            bytes.extend_from_slice(&(list.len() as u32).to_le_bytes());
            for &idx in list {
                bytes.extend_from_slice(&idx.to_le_bytes());
            }
        }

        bytes
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut offset = 0;

        if bytes.len() < 8 {
            return Err("IVF index buffer too small".into());
        }

        let dim = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        let n_clusters = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;

        let dim = EmbeddingDim::new(dim as usize)?;
        let n_clusters = ClusterCount::new(n_clusters as usize)?;

        let centroids_len = n_clusters.as_usize() * dim.as_usize();
        let mut centroids = Vec::with_capacity(centroids_len);
        for _ in 0..centroids_len {
            let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
            centroids.push(val);
            offset += 4;
        }

        let mut inverted_lists = Vec::with_capacity(n_clusters.as_usize());
        for _ in 0..n_clusters.as_usize() {
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

    pub(crate) fn find_closest_centroids(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let nprobe = nprobe.min(self.n_clusters.as_usize());

        let mut centroid_distances: Vec<(usize, f32)> = (0..self.n_clusters.as_usize())
            .map(|i| {
                let centroid_start = i * self.dim.as_usize();
                let centroid =
                    &self.centroids[centroid_start..centroid_start + self.dim.as_usize()];
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

pub(crate) fn build_ivf_index(
    embeddings: &Embeddings,
    config: IvfBuildConfig,
) -> Result<IvfIndex, Box<dyn std::error::Error>> {
    let n_vectors = embeddings.row_count();
    if n_vectors == 0 {
        return Err("Cannot build IVF index with zero vectors".into());
    }

    let n_clusters = match config.n_clusters {
        Some(value) => value,
        None => {
            let target = (n_vectors as f64).sqrt().ceil() as usize;
            ClusterCount::new(target)?
        }
    };
    if n_clusters.as_usize() > n_vectors {
        return Err("n_clusters cannot exceed number of vectors".into());
    }

    let mut sample_size = (n_vectors / 20).max(1); // 5%
    sample_size = sample_size.min(100_000);
    sample_size = sample_size.max(n_clusters.as_usize()).min(n_vectors);

    let kmeans_params = KMeansParams {
        n_clusters,
        max_iters: config.max_iters,
        seed: config.seed,
    };

    let (centroids, _) = if sample_size == n_vectors {
        k_means(embeddings, kmeans_params)
    } else {
        let sample = sample_embeddings(embeddings, sample_size, config.seed)?;
        k_means(&sample, kmeans_params)
    };

    let n_clusters_usize = n_clusters.as_usize();
    let dim = embeddings.dim().as_usize();
    let data = embeddings.data();
    let mut inverted_lists = vec![Vec::new(); n_clusters_usize];
    let locals = parallel_ranges(n_vectors, |start, end| {
        let mut local_lists = vec![Vec::new(); n_clusters_usize];
        for row_idx in start..end {
            let vec = &data[row_idx * dim..(row_idx + 1) * dim];
            let cluster_idx = nearest_centroid(vec, &centroids, dim);
            local_lists[cluster_idx].push(row_idx as u32);
        }
        local_lists
    });
    for mut local_lists in locals {
        for (cluster_idx, local) in local_lists.iter_mut().enumerate() {
            inverted_lists[cluster_idx].append(local);
        }
    }

    Ok(IvfIndex {
        dim: embeddings.dim(),
        n_clusters,
        centroids,
        inverted_lists,
    })
}

struct KMeansParams {
    n_clusters: ClusterCount,
    max_iters: usize,
    seed: u64,
}

fn sample_embeddings(
    embeddings: &Embeddings,
    sample_size: usize,
    seed: u64,
) -> Result<Embeddings, Box<dyn std::error::Error>> {
    use rand::seq::index::sample;

    let n = embeddings.row_count();
    let dim = embeddings.dim().as_usize();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let indices = sample(&mut rng, n, sample_size);

    let mut data = Vec::with_capacity(sample_size * dim);
    for idx in indices.iter() {
        let start = idx * dim;
        let end = start + dim;
        data.extend_from_slice(&embeddings.data()[start..end]);
    }

    Embeddings::new(data, embeddings.dim())
}

fn nearest_centroid(vec: &[f32], centroids: &[f32], dim: usize) -> usize {
    let mut best_cluster = 0usize;
    let mut best_dist = f32::INFINITY;
    let n_clusters = centroids.len() / dim;
    for i in 0..n_clusters {
        let start = i * dim;
        let dist = squared_l2_distance(vec, &centroids[start..start + dim]);
        if dist < best_dist {
            best_dist = dist;
            best_cluster = i;
        }
    }
    best_cluster
}

fn worker_count(len: usize) -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(len)
        .max(1)
}

fn parallel_ranges<R, F>(len: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize, usize) -> R + Sync,
{
    if len == 0 {
        return Vec::new();
    }
    let workers = worker_count(len);
    let chunk_size = len.div_ceil(workers);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let f = &f;
        for chunk_idx in 0..workers {
            let start = chunk_idx * chunk_size;
            if start >= len {
                break;
            }
            let end = (start + chunk_size).min(len);
            let handle = scope.spawn(move || f(start, end));
            handles.push(handle);
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker thread panicked"))
            .collect()
    })
}

fn parallel_chunks_mut<T, R, F>(data: &mut [T], f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(usize, &mut [T]) -> R + Sync,
{
    if data.is_empty() {
        return Vec::new();
    }
    let workers = worker_count(data.len());
    let chunk_size = data.len().div_ceil(workers);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let f = &f;
        for (chunk_idx, chunk) in data.chunks_mut(chunk_size).enumerate() {
            let start = chunk_idx * chunk_size;
            let handle = scope.spawn(move || f(start, chunk));
            handles.push(handle);
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker thread panicked"))
            .collect()
    })
}

/// K-means clustering implementation with k-means++ initialization.
fn k_means(embeddings: &Embeddings, params: KMeansParams) -> (Vec<f32>, Vec<usize>) {
    let dim = embeddings.dim().as_usize();
    let n = embeddings.row_count();
    let data = embeddings.data();
    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);

    let n_clusters = params.n_clusters.as_usize();
    let mut centroids = vec![0.0f32; n_clusters * dim];

    let init_sample_size = n.min(50_000).max(n_clusters);
    let init_indices: Vec<usize> = if init_sample_size == n {
        (0..n).collect()
    } else {
        use rand::seq::index::sample;
        sample(&mut rng, n, init_sample_size).iter().collect()
    };

    let first_choice = rng.gen_range(0..init_indices.len());
    let first_idx = init_indices[first_choice];
    centroids[..dim].copy_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

    let mut min_distances = vec![0.0f32; init_indices.len()];
    parallel_chunks_mut(&mut min_distances, |start, dist_chunk| {
        let indices = &init_indices[start..start + dist_chunk.len()];
        let centroid = &centroids[..dim];
        for (&row_idx, dist_slot) in indices.iter().zip(dist_chunk.iter_mut()) {
            let vec = &data[row_idx * dim..(row_idx + 1) * dim];
            *dist_slot = squared_l2_distance(vec, centroid);
        }
    });

    for i in 1..n_clusters {
        let centroid = &centroids[(i - 1) * dim..i * dim];
        let partial_sums: Vec<f32> =
            parallel_chunks_mut(&mut min_distances, |start, dist_chunk| {
                let indices = &init_indices[start..start + dist_chunk.len()];
                let mut local_sum = 0.0f32;
                for (&row_idx, dist_slot) in indices.iter().zip(dist_chunk.iter_mut()) {
                    let vec = &data[row_idx * dim..(row_idx + 1) * dim];
                    let dist = squared_l2_distance(vec, centroid);
                    if dist < *dist_slot {
                        *dist_slot = dist;
                    }
                    local_sum += *dist_slot;
                }
                local_sum
            });
        let total: f32 = partial_sums.into_iter().sum();

        if total > 0.0 {
            let threshold = rng.gen_range(0.0..1.0) * total;
            let mut cumsum = 0.0;
            for (slot, &d) in min_distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    let row_idx = init_indices[slot];
                    centroids[i * dim..(i + 1) * dim]
                        .copy_from_slice(&data[row_idx * dim..(row_idx + 1) * dim]);
                    break;
                }
            }
        } else {
            let choice = rng.gen_range(0..init_indices.len());
            let row_idx = init_indices[choice];
            centroids[i * dim..(i + 1) * dim]
                .copy_from_slice(&data[row_idx * dim..(row_idx + 1) * dim]);
        }
    }

    let mut assignments = vec![0usize; n];
    let mut cluster_sizes = vec![0usize; n_clusters];

    for _iter in 0..params.max_iters {
        let mut changed = 0;
        cluster_sizes.fill(0);
        let results: Vec<(usize, Vec<usize>)> =
            parallel_chunks_mut(&mut assignments, |start, assignment_chunk| {
                let mut local_changed = 0usize;
                let mut local_sizes = vec![0usize; n_clusters];
                for (offset, slot) in assignment_chunk.iter_mut().enumerate() {
                    let row_idx = start + offset;
                    let vec = &data[row_idx * dim..(row_idx + 1) * dim];
                    let mut best_cluster = 0;
                    let mut best_dist = f32::INFINITY;

                    for j in 0..n_clusters {
                        let centroid = &centroids[j * dim..(j + 1) * dim];
                        let dist = squared_l2_distance(vec, centroid);
                        if dist < best_dist {
                            best_dist = dist;
                            best_cluster = j;
                        }
                    }

                    if *slot != best_cluster {
                        local_changed += 1;
                    }
                    *slot = best_cluster;
                    local_sizes[best_cluster] += 1;
                }
                (local_changed, local_sizes)
            });
        for (local_changed, local_sizes) in results {
            changed += local_changed;
            for (idx, size) in local_sizes.into_iter().enumerate() {
                cluster_sizes[idx] += size;
            }
        }

        if changed == 0 {
            break;
        }

        centroids.fill(0.0);

        for i in 0..n {
            let cluster = assignments[i];
            let vec = &data[i * dim..(i + 1) * dim];
            for (j, &val) in vec.iter().enumerate() {
                centroids[cluster * dim + j] += val;
            }
        }

        for j in 0..n_clusters {
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

/// Compute squared L2 distance between two vectors.
#[inline]
pub(crate) fn squared_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    let mut i = 0usize;
    let len = a.len();
    while i + 4 <= len {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivf::EmbeddingDim;

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
            dim: EmbeddingDim::new(3).unwrap(),
            n_clusters: ClusterCount::new(2).unwrap(),
            centroids: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            inverted_lists: vec![vec![0, 2, 4], vec![1, 3]],
        };

        let bytes = index.to_bytes();
        let restored = IvfIndex::from_bytes(&bytes).unwrap();

        assert_eq!(restored.dim(), index.dim());
        assert_eq!(restored.n_clusters.as_usize(), index.n_clusters.as_usize());
        assert_eq!(restored.centroids, index.centroids);
        assert_eq!(restored.inverted_lists, index.inverted_lists);
    }
}
