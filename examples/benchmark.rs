use arrow::array::{Array, Float32Array, ListArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pq_vector::{build_index, topk, IvfBuildParams};
use std::fs::{self, File};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source_path = Path::new("data/combined.parquet");
    let output_path = Path::new("data/combined_indexed.parquet");
    let embedding_column = "embedding";

    // File size before indexing
    let original_size = fs::metadata(source_path)?.len();
    println!("Original file size: {:.2} MB", original_size as f64 / 1024.0 / 1024.0);

    // Build the index
    println!("\n=== Building IVF Index ===");
    let start = Instant::now();
    let params = IvfBuildParams {
        n_clusters: None, // sqrt(n) ~ 70 clusters
        max_iters: 20,
        seed: 42,
    };
    build_index(source_path, output_path, embedding_column, &params)?;
    let build_time = start.elapsed();
    println!("Build time: {:.2}s", build_time.as_secs_f64());

    // File size after indexing
    let indexed_size = fs::metadata(output_path)?.len();
    println!("\nIndexed file size: {:.2} MB", indexed_size as f64 / 1024.0 / 1024.0);
    println!("Index overhead: {:.2} MB ({:.1}%)", 
        (indexed_size - original_size) as f64 / 1024.0 / 1024.0,
        (indexed_size - original_size) as f64 / original_size as f64 * 100.0
    );

    // Get a query vector
    let query = get_embedding_at_row(output_path, embedding_column, 42)?;
    println!("\nQuery vector dimension: {}", query.len());

    // Benchmark searches with different nprobe values
    println!("\n=== Search Performance ===");
    for nprobe in [1, 5, 10, 20] {
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = topk(output_path, &query, 10, nprobe)?;
        }
        let total_time = start.elapsed();
        let avg_time = total_time.as_secs_f64() / iterations as f64;
        println!("nprobe={:2}: {:.2}ms per query", nprobe, avg_time * 1000.0);
    }

    // Show sample results
    println!("\n=== Sample Results (k=5, nprobe=10) ===");
    let results = topk(output_path, &query, 5, 10)?;
    for (i, r) in results.iter().enumerate() {
        println!("{}. row_idx={}, distance={:.4}", i + 1, r.row_idx, r.distance);
    }

    Ok(())
}

fn get_embedding_at_row(
    path: &Path,
    column_name: &str,
    row: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut current_row = 0;
    for batch in reader {
        let batch = batch?;
        let embedding_col = batch.column_by_name(column_name).unwrap();
        let list_array = embedding_col.as_any().downcast_ref::<ListArray>().unwrap();

        if row < current_row + list_array.len() {
            let local_row = row - current_row;
            let start = list_array.value_offsets()[local_row] as usize;
            let end = list_array.value_offsets()[local_row + 1] as usize;
            let values = list_array.values();
            let float_array = values.as_any().downcast_ref::<Float32Array>().unwrap();
            return Ok((start..end).map(|i| float_array.value(i)).collect());
        }
        current_row += list_array.len();
    }
    Err("Row not found".into())
}
