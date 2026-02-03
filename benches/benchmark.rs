use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Float32Builder, Int32Builder, ListBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use parquet::arrow::ArrowWriter;
use pq_vector::IndexBuilder;
use pq_vector::df_vector::{VectorTopKOptions, VectorTopKPhysicalOptimizerRule};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
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
    let indexed_rows: usize = indexed_batches.iter().map(|b| b.num_rows()).sum();
    println!("Query time (with index): {:.2?}", indexed_time);
    println!("Returned rows: {indexed_rows}");

    Ok(())
}

fn generate_parquet(
    path: &Path,
    rows: usize,
    dim: usize,
    batch_rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
    ]));

    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    let mut rng = StdRng::seed_from_u64(1234);

    let mut row = 0usize;
    while row < rows {
        let count = (rows - row).min(batch_rows);
        let mut id_builder = Int32Builder::with_capacity(count);
        let mut list_builder = ListBuilder::new(Float32Builder::with_capacity(count * dim));

        for i in 0..count {
            id_builder.append_value((row + i) as i32);
            for _ in 0..dim {
                list_builder.values().append_value(rng.r#gen::<f32>());
            }
            list_builder.append(true);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(list_builder.finish()),
            ],
        )?;
        writer.write(&batch)?;
        row += count;
    }

    writer.close()?;
    Ok(())
}

fn random_query(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn array_literal(values: &[f32]) -> String {
    let mut out = String::with_capacity(values.len() * 10);
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

fn to_mb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}
