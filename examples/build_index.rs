//! Build an IVF index and write it to a new Parquet file.
//!
//! Usage:
//!   cargo run --example build_index
//!
//! Optional env vars:
//! - PQ_VECTOR_SOURCE: source parquet file (default: data/vldb_2025.parquet)
//! - PQ_VECTOR_INDEXED: output parquet file (default: data/vldb_2025_indexed.parquet)

mod common;
use pq_vector::IndexBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source =
        std::env::var("PQ_VECTOR_SOURCE").unwrap_or_else(|_| "data/vldb_2025.parquet".to_string());
    let indexed = std::env::var("PQ_VECTOR_INDEXED")
        .unwrap_or_else(|_| "data/vldb_2025_indexed.parquet".to_string());

    println!("Building IVF index from {source}...");
    IndexBuilder::new(&source, "embedding").build_new(&indexed)?;
    println!("Wrote indexed parquet to {indexed}");

    Ok(())
}
