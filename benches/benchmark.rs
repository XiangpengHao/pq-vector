use arrow::array::{Array, Float32Array, ListArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pq_vector::{EmbeddingColumn, IndexBuilder, IvfBuildParams, TopkBuilder};
use std::cmp::Ordering;
use std::fs::{self, File};
use std::path::Path;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compressed_path = Path::new("data/combined_lz4.parquet");
    let indexed_path = Path::new("data/combined_indexed.parquet");
    let embedding_column = EmbeddingColumn::try_from("embedding")?;

    let compressed_size = fs::metadata(compressed_path)?.len();
    println!(
        "Compressed file size: {:.2} MB",
        compressed_size as f64 / 1024.0 / 1024.0
    );

    // Step 1: Build the index
    println!("\n=== Building IVF Index ===");
    let start = Instant::now();
    let params = IvfBuildParams {
        n_clusters: None, // sqrt(n) ~ 70 clusters
        max_iters: 20,
        seed: 42,
    };
    IndexBuilder::new(compressed_path, indexed_path, embedding_column.clone())
        .params(params)
        .build()?;
    let build_time = start.elapsed();
    println!("Build time: {:.2}s", build_time.as_secs_f64());

    let indexed_size = fs::metadata(indexed_path)?.len();
    println!(
        "Indexed file size: {:.2} MB",
        indexed_size as f64 / 1024.0 / 1024.0
    );
    println!(
        "Index overhead: {:.2} MB ({:.1}%)",
        (indexed_size - compressed_size) as f64 / 1024.0 / 1024.0,
        (indexed_size - compressed_size) as f64 / compressed_size as f64 * 100.0
    );

    // Load all embeddings for brute force and recall calculation
    let (all_embeddings, dim) = read_all_embeddings(indexed_path, embedding_column.as_str())?;
    let n_vectors = all_embeddings.len() / dim;
    println!("\nDataset: {} vectors, dim={}", n_vectors, dim);

    // Use multiple query vectors for more reliable measurements
    let query_rows = vec![42, 100, 500, 1000, 2000, 3000, 4000];
    let k = 10;

    // Step 2: Benchmark brute force - reading ALL embeddings from parquet each time
    // This is the fair comparison: IVF reads some pages, brute force reads all pages
    println!("\n=== Brute Force Performance (reading from parquet) ===");
    let start = Instant::now();
    let iterations = 5;
    for _ in 0..iterations {
        for &qr in &query_rows {
            // Read all embeddings from parquet file (this is what brute force must do)
            let (embs, d) = read_all_embeddings(compressed_path, embedding_column.as_str())?;
            let query = &embs[qr * d..(qr + 1) * d];
            let _ = brute_force_topk(&embs, d, query, k);
        }
    }
    let bf_time = start.elapsed().as_secs_f64() / (iterations * query_rows.len()) as f64;
    println!(
        "Brute force (with I/O): {:.2}ms per query",
        bf_time * 1000.0
    );

    // Also show in-memory brute force for reference
    println!("\n=== In-Memory Brute Force (for reference) ===");
    let start = Instant::now();
    let iterations = 50;
    for _ in 0..iterations {
        for &qr in &query_rows {
            let query = &all_embeddings[qr * dim..(qr + 1) * dim];
            let _ = brute_force_topk(&all_embeddings, dim, query, k);
        }
    }
    let bf_mem_time = start.elapsed().as_secs_f64() / (iterations * query_rows.len()) as f64;
    println!(
        "Brute force (in-memory): {:.2}ms per query",
        bf_mem_time * 1000.0
    );

    // Step 3: Benchmark IVF with different nprobe values + measure recall
    println!("\n=== IVF Performance & Recall ===");
    println!(
        "{:>8} {:>12} {:>10} {:>10}",
        "nprobe", "time (ms)", "speedup", "recall@10"
    );
    println!("{}", "-".repeat(44));

    for nprobe in [1, 2, 5, 10, 20, 40, 70] {
        // Measure time
        let start = Instant::now();
        let iterations = 20;
        for _ in 0..iterations {
            for &qr in &query_rows {
                let query = &all_embeddings[qr * dim..(qr + 1) * dim];
                let _ = TopkBuilder::new(indexed_path, query)
                    .k(k)?
                    .nprobe(nprobe)?
                    .search()
                    .await?;
            }
        }
        let ivf_time = start.elapsed().as_secs_f64() / (iterations * query_rows.len()) as f64;

        // Measure recall
        let mut total_recall = 0.0;
        for &qr in &query_rows {
            let query = &all_embeddings[qr * dim..(qr + 1) * dim];

            // Ground truth from brute force
            let bf_results = brute_force_topk(&all_embeddings, dim, query, k);
            let bf_set: std::collections::HashSet<usize> =
                bf_results.iter().map(|(idx, _)| *idx).collect();

            // IVF results
            let ivf_results = TopkBuilder::new(indexed_path, query)
                .k(k)?
                .nprobe(nprobe)?
                .search()
                .await?;
            let ivf_set: std::collections::HashSet<usize> =
                ivf_results.iter().map(|r| r.row_idx as usize).collect();

            let overlap = bf_set.intersection(&ivf_set).count();
            total_recall += overlap as f64 / k as f64;
        }
        let avg_recall = total_recall / query_rows.len() as f64;

        let speedup = bf_time / ivf_time;
        println!(
            "{:>8} {:>12.2} {:>10.1}x {:>10.2}",
            nprobe,
            ivf_time * 1000.0,
            speedup,
            avg_recall
        );
    }

    Ok(())
}

fn read_all_embeddings(
    path: &Path,
    column_name: &str,
) -> Result<(Vec<f32>, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut all_embeddings = Vec::new();
    let mut dim = 0;

    for batch in reader {
        let batch = batch?;
        let embedding_col = batch.column_by_name(column_name).unwrap();
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

    Ok((all_embeddings, dim))
}

fn brute_force_topk(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = data.len() / dim;
    let mut distances: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let vec = &data[i * dim..(i + 1) * dim];
            let dist = squared_l2_distance(query, vec).sqrt();
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    distances.truncate(k);
    distances
}

fn squared_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}
