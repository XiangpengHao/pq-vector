//! Example demonstrating DataFusion SQL integration with the topk() table function.
//!
//! Run with: cargo run --example datafusion_sql

use arrow::array::{Array, Float32Array, ListArray};
use datafusion::prelude::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pq_vector::{encode_query_vector, TopkBinaryTableFunction, TopkTableFunction};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let indexed_path = Path::new("data/vldb_2025_indexed.parquet");

    // Check if indexed file exists
    if !indexed_path.exists() {
        eprintln!("Error: {} does not exist.", indexed_path.display());
        eprintln!("Run 'cargo run --example build_and_query' first to create the indexed file.");
        return Ok(());
    }

    // Get a query vector from the file
    let query_vector = get_embedding_at_row(indexed_path, "embedding", 42)?;
    println!(
        "Using embedding from row 42 as query vector (dim={})",
        query_vector.len()
    );

    // Create DataFusion context and register the topk functions
    let ctx = SessionContext::new();
    ctx.register_udtf("topk", Arc::new(TopkTableFunction));
    ctx.register_udtf("topk_bin", Arc::new(TopkBinaryTableFunction));

    // Encode the query vector as base64 for the binary UDTF
    let query_b64 = encode_query_vector(&query_vector);
    println!(
        "Base64 encoded query vector: {}...{} ({} bytes)",
        &query_b64[..20],
        &query_b64[query_b64.len() - 20..],
        query_b64.len()
    );

    // Query using the binary table function (works with large vectors!)
    println!("\n=== Query with topk_bin (base64 encoded vector) ===\n");
    let sql = format!(
        "SELECT _distance, title FROM topk_bin('{}', '{}', 10, 5)",
        indexed_path.display(),
        query_b64
    );
    println!(
        "SQL: SELECT _distance, title FROM topk_bin('...', '<base64>', 10, 5)\n"
    );

    let df = ctx.sql(&sql).await?;
    df.show().await?;

    // Query with LIMIT
    println!("\n=== Query with LIMIT ===\n");
    let sql = format!(
        "SELECT title, _distance FROM topk_bin('{}', '{}', 20, 10) LIMIT 5",
        indexed_path.display(),
        query_b64
    );

    let df = ctx.sql(&sql).await?;
    df.show().await?;

    // Demonstrate using a different query vector (row 0)
    println!("\n=== Search with row 0's embedding ===\n");
    let query_vector_0 = get_embedding_at_row(indexed_path, "embedding", 0)?;
    let query_b64_0 = encode_query_vector(&query_vector_0);

    let sql = format!(
        "SELECT _distance, title FROM topk_bin('{}', '{}', 5, 5)",
        indexed_path.display(),
        query_b64_0
    );

    let df = ctx.sql(&sql).await?;
    df.show().await?;

    // For small vectors, regular topk() with ARRAY syntax also works
    println!("\n=== Small vector with regular topk() ===\n");
    println!("For small dimension vectors, you can use ARRAY[] syntax:");
    println!("  SELECT * FROM topk('file.parquet', ARRAY[1.0, 2.0, 3.0], 10, 5)");
    println!("\nFor large vectors (like 4096-dim), use topk_bin() with base64 encoding.");

    println!("\nAll queries completed successfully!");
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
