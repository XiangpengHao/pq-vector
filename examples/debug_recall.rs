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

    // Rebuild with more clusters to make the test more challenging
    println!("=== Rebuilding with more clusters (50) ===\n");
    let params = IvfBuildParams {
        n_clusters: Some(50), // More clusters = smaller clusters = harder test
        max_iters: 20,
        seed: 42,
    };
    build_index(source_path, output_path, embedding_column, &params)?;

    // Load embeddings
    let (embeddings, dim) = read_embeddings(output_path, embedding_column)?;
    let n_vectors = embeddings.len() / dim;

    println!("\n=== Detailed Recall Analysis ===\n");
    println!("Dataset: {} vectors, dim={}", n_vectors, dim);
    println!(
        "Clusters: 50, avg cluster size: {:.1}\n",
        n_vectors as f64 / 50.0
    );

    // Test multiple queries and track detailed stats
    let k = 10;
    let test_rows: Vec<usize> = (0..n_vectors).step_by(10).collect(); // Test every 10th row

    for nprobe in [1, 2, 3, 5, 10, 20, 50] {
        let mut total_recall = 0.0;
        let mut perfect_count = 0;
        let mut worst_recall = 1.0;
        let mut worst_query = 0;

        for &query_row in &test_rows {
            let query = &embeddings[query_row * dim..(query_row + 1) * dim];

            // IVF search
            let ivf_results = topk(output_path, query, k, nprobe)?;
            let ivf_rows: Vec<u32> = ivf_results.iter().map(|r| r.row_idx).collect();

            // Brute force
            let bf_results = brute_force_topk(&embeddings, dim, query, k);
            let bf_rows: Vec<u32> = bf_results.iter().map(|(idx, _)| *idx as u32).collect();

            let recall = ivf_rows.iter().filter(|r| bf_rows.contains(r)).count() as f64 / k as f64;
            total_recall += recall;

            if recall == 1.0 {
                perfect_count += 1;
            }
            if recall < worst_recall {
                worst_recall = recall;
                worst_query = query_row;
            }
        }

        let avg_recall = total_recall / test_rows.len() as f64;
        println!(
            "nprobe={:2}: avg_recall={:.3}, perfect={:3}/{}, worst={:.2} (query {})",
            nprobe,
            avg_recall,
            perfect_count,
            test_rows.len(),
            worst_recall,
            worst_query
        );
    }

    // Deep dive on a specific query with low recall
    println!("\n=== Deep Dive: Finding a challenging query ===\n");

    // Find a query where nprobe=1 doesn't give perfect recall
    for query_row in 0..n_vectors {
        let query = &embeddings[query_row * dim..(query_row + 1) * dim];

        let ivf_results_1 = topk(output_path, query, k, 1)?;
        let ivf_rows_1: Vec<u32> = ivf_results_1.iter().map(|r| r.row_idx).collect();

        let bf_results = brute_force_topk(&embeddings, dim, query, k);
        let bf_rows: Vec<u32> = bf_results.iter().map(|(idx, _)| *idx as u32).collect();

        let recall_1 = ivf_rows_1.iter().filter(|r| bf_rows.contains(r)).count() as f64 / k as f64;

        if recall_1 < 1.0 {
            println!("Found challenging query: row {}", query_row);
            println!("  Brute force top-{}: {:?}", k, bf_rows);
            println!(
                "  IVF nprobe=1:      {:?} (recall={:.2})",
                ivf_rows_1, recall_1
            );

            // Test with increasing nprobe
            for nprobe in [2, 3, 5, 10, 20] {
                let ivf_results = topk(output_path, query, k, nprobe)?;
                let ivf_rows: Vec<u32> = ivf_results.iter().map(|r| r.row_idx).collect();
                let recall =
                    ivf_rows.iter().filter(|r| bf_rows.contains(r)).count() as f64 / k as f64;
                println!(
                    "  IVF nprobe={:2}:     {:?} (recall={:.2})",
                    nprobe, ivf_rows, recall
                );
                if recall == 1.0 {
                    break;
                }
            }
            println!();
            break;
        }
    }

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
