use arrow::array::{Array, Float32Array, ListArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pq_vector::{build_index, topk, IvfBuildParams};
use std::cmp::Ordering;
use std::fs::File;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source_path = Path::new("data/vldb_2025.parquet");
    let output_path = Path::new("data/vldb_2025_indexed.parquet");
    let embedding_column = "embedding";

    // Build the index if not already done
    if !output_path.exists() {
        let params = IvfBuildParams {
            max_iters: 10,
            ..Default::default()
        };
        build_index(source_path, output_path, embedding_column, &params)?;
    }

    // Load all embeddings for brute force comparison
    let (embeddings, dim) = read_embeddings(output_path, embedding_column)?;
    let n_vectors = embeddings.len() / dim;

    // Test multiple queries
    let test_rows = vec![0, 50, 100, 200, 300, 400];
    let k = 10;

    println!("=== Comparing IVF search vs Brute Force ===\n");
    println!("Dataset: {} vectors, dim={}, k={}\n", n_vectors, dim, k);

    for query_row in test_rows {
        let query = &embeddings[query_row * dim..(query_row + 1) * dim];

        // IVF search with full nprobe (should be exact)
        let ivf_results = topk(output_path, query, k, 23)?; // 23 = n_clusters

        // Brute force search
        let bf_results = brute_force_topk(&embeddings, dim, query, k);

        // Compare results
        let ivf_rows: Vec<u32> = ivf_results.iter().map(|r| r.row_idx).collect();
        let bf_rows: Vec<u32> = bf_results.iter().map(|(idx, _)| *idx as u32).collect();

        let recall = ivf_rows.iter().filter(|r| bf_rows.contains(r)).count() as f64 / k as f64;

        println!("Query row {}: recall@{}={:.2}", query_row, k, recall);

        if recall < 1.0 {
            println!("  IVF top 3: {:?}", &ivf_rows[..3]);
            println!("  BF  top 3: {:?}", &bf_rows[..3]);
        }
    }

    // Test recall vs nprobe
    println!("\n=== Recall vs nprobe ===\n");
    let query_row = 100;
    let query = &embeddings[query_row * dim..(query_row + 1) * dim];
    let bf_results = brute_force_topk(&embeddings, dim, query, k);
    let bf_rows: Vec<u32> = bf_results.iter().map(|(idx, _)| *idx as u32).collect();

    for nprobe in [1, 2, 3, 5, 10, 23] {
        let ivf_results = topk(output_path, query, k, nprobe)?;
        let ivf_rows: Vec<u32> = ivf_results.iter().map(|r| r.row_idx).collect();
        let recall = ivf_rows.iter().filter(|r| bf_rows.contains(r)).count() as f64 / k as f64;
        println!("nprobe={:2}: recall@{}={:.2}", nprobe, k, recall);
    }

    // Verify file is readable by standard parquet readers
    println!("\n=== Verifying Standard Parquet Compatibility ===\n");
    let file = File::open(output_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    println!("Schema: {:?}", builder.schema());
    let reader = builder.build()?;
    let mut total_rows = 0;
    for batch in reader {
        total_rows += batch?.num_rows();
    }
    println!(
        "Successfully read {} rows from indexed parquet file",
        total_rows
    );

    Ok(())
}

fn read_embeddings(
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
