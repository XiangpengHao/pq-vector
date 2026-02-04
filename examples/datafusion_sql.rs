//! Run vector top-k search using DataFusion SQL.
//!
//! Usage:
//!   cargo run --example datafusion_sql
//!
//! Optional env vars:
//! - PQ_VECTOR_SOURCE: source parquet file (default: data/vldb_2025.parquet)
//! - PQ_VECTOR_INDEXED: indexed parquet file (default: data/vldb_2025_indexed.parquet)
//! - PQ_VECTOR_QUERY_ROW: which row to use as the query (default: 0)

mod common;

use common::{ensure_indexed, read_embedding_at_row};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use pq_vector::df_vector::{PqVectorSessionBuilderExt, VectorTopKOptions};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source =
        std::env::var("PQ_VECTOR_SOURCE").unwrap_or_else(|_| "data/vldb_2025.parquet".to_string());
    let indexed = std::env::var("PQ_VECTOR_INDEXED")
        .unwrap_or_else(|_| "data/vldb_2025_indexed.parquet".to_string());
    let query_row: usize = std::env::var("PQ_VECTOR_QUERY_ROW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    ensure_indexed(&source, &indexed)?;

    let options = VectorTopKOptions {
        nprobe: 8,
        max_candidates: None,
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_pq_vector(options)
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_parquet("t", &indexed, ParquetReadOptions::default())
        .await?;

    let query_vec = read_embedding_at_row(Path::new(&indexed), "embedding", query_row)?;
    let query_literal = format!(
        "[{}]",
        query_vec
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let sql =
        format!("SELECT title FROM t ORDER BY array_distance(embedding, {query_literal}) LIMIT 5");
    let batches = ctx.sql(&sql).await?.collect().await?;
    println!("{}", arrow::util::pretty::pretty_format_batches(&batches)?);

    Ok(())
}
