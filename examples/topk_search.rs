//! Run a top-k search against an indexed Parquet file.
//!
//! Usage:
//!   cargo run --example topk_search
//!
//! Optional env vars:
//! - PQ_VECTOR_SOURCE: source parquet file (default: data/vldb_2025.parquet)
//! - PQ_VECTOR_INDEXED: indexed parquet file (default: data/vldb_2025_indexed.parquet)
//! - PQ_VECTOR_QUERY_ROW: which row to use as the query (default: 0)

mod common;

use common::{ensure_indexed, read_embedding_at_row};
use pq_vector::TopkBuilder;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = std::env::var("PQ_VECTOR_SOURCE")
        .unwrap_or_else(|_| "data/vldb_2025.parquet".to_string());
    let indexed = std::env::var("PQ_VECTOR_INDEXED")
        .unwrap_or_else(|_| "data/vldb_2025_indexed.parquet".to_string());
    let query_row: usize = std::env::var("PQ_VECTOR_QUERY_ROW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    ensure_indexed(&source, &indexed)?;

    let query = read_embedding_at_row(Path::new(&indexed), "embedding", query_row)?;
    let results = TopkBuilder::new(&indexed, &query)
        .k(5)?
        .nprobe(5)?
        .search()
        .await?;

    println!("Top 5 neighbors for row {query_row}:");
    for (rank, result) in results.iter().enumerate() {
        println!(
            "{}. row {} distance {:.4}",
            rank + 1,
            result.row_idx,
            result.distance
        );
    }

    Ok(())
}
