mod bench_util;

use std::fs;
use std::path::Path;
use std::time::Instant;

use arrow::array::Int32Array;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use pq_vector::IndexBuilder;
use pq_vector::df_vector::{PqVectorSessionBuilderExt, VectorTopKOptions};

use bench_util::{array_literal, generate_parquet, random_query, to_mb};

const ROWS: usize = 1_000_000;
const DIM: usize = 1024;
const BATCH_ROWS: usize = 2048;
const K: usize = 100;
const NPROBE: usize = 8;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new("data");
    fs::create_dir_all(data_dir)?;

    let parquet_path = data_dir.join("benchmark.parquet");
    if parquet_path.exists() {
        fs::remove_file(&parquet_path)?;
    }

    println!("=== Generating synthetic dataset ===");
    println!("rows={ROWS}, dim={DIM}, batch_rows={BATCH_ROWS}");

    let gen_start = Instant::now();
    generate_parquet(&parquet_path, ROWS, DIM, BATCH_ROWS)?;
    let gen_time = gen_start.elapsed();

    let original_size = fs::metadata(&parquet_path)?.len();
    println!("Generated parquet in {:.2?}", gen_time);
    println!("Original parquet size: {:.2} MB", to_mb(original_size));

    println!("\n=== Vector search without index (DataFusion) ===");
    let query = random_query(DIM, 7);
    let query_literal = array_literal(&query);
    let sql =
        format!("SELECT id FROM t ORDER BY array_distance(embedding, {query_literal}) LIMIT {K}");

    let plain_ctx = SessionContext::new();
    plain_ctx
        .register_parquet(
            "t",
            parquet_path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    let plain_start = Instant::now();
    let plain_batches = plain_ctx.sql(&sql).await?.collect().await?;
    let plain_time = plain_start.elapsed();
    let plain_ids = extract_ids(&plain_batches);
    let plain_rows: usize = plain_batches.iter().map(|b| b.num_rows()).sum();
    println!("Query time (no index): {:.2?}", plain_time);
    println!("Returned rows: {plain_rows}");

    println!("\n=== Building IVF index (in-place, defaults) ===");
    let index_start = Instant::now();
    IndexBuilder::new(&parquet_path, "embedding").build_inplace()?;
    let index_time = index_start.elapsed();

    let indexed_size = fs::metadata(&parquet_path)?.len();
    println!("Index build time: {:.2?}", index_time);
    println!("Indexed parquet size: {:.2} MB", to_mb(indexed_size));
    println!(
        "Index overhead: {:.2} MB ({:.1}%)",
        to_mb(indexed_size - original_size),
        (indexed_size - original_size) as f64 / original_size as f64 * 100.0
    );

    println!("\n=== Vector search with index (DataFusion) ===");
    let options = VectorTopKOptions {
        nprobe: NPROBE,
        max_candidates: None,
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_pq_vector(options)
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_parquet(
        "t",
        parquet_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    let indexed_start = Instant::now();
    let indexed_batches = ctx.sql(&sql).await?.collect().await?;
    let indexed_time = indexed_start.elapsed();
    let indexed_ids = extract_ids(&indexed_batches);
    let indexed_rows: usize = indexed_batches.iter().map(|b| b.num_rows()).sum();
    println!("Query time (with index): {:.2?}", indexed_time);
    println!("Returned rows: {indexed_rows}");

    let recall = recall_at_k(&plain_ids, &indexed_ids);
    println!("Recall@{K}: {:.2}%", recall * 100.0);

    Ok(())
}

fn extract_ids(batches: &[arrow::record_batch::RecordBatch]) -> Vec<i32> {
    let mut ids = Vec::new();
    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id column should be int32");
        for i in 0..array.len() {
            ids.push(array.value(i));
        }
    }
    ids
}

fn recall_at_k(ground_truth: &[i32], candidates: &[i32]) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let truth: std::collections::HashSet<i32> = ground_truth.iter().copied().collect();
    let hits = candidates.iter().filter(|id| truth.contains(id)).count();
    hits as f64 / ground_truth.len() as f64
}
