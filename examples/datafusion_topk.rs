use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, ListArray};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use pq_vector::df_vector::{VectorTopKOptions, VectorTopKPhysicalOptimizerRule};
use pq_vector::{IvfBuildParams, build_index};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source_path = Path::new("data/vldb_2025.parquet");
    let indexed_path = Path::new("data/vldb_2025_indexed.parquet");
    let embedding_column = "embedding";

    if !indexed_path.exists() {
        build_index(
            source_path,
            indexed_path,
            embedding_column,
            &IvfBuildParams::default(),
        )?;
    }

    let options = VectorTopKOptions {
        nprobe: 64,
        batch_size: 1024,
        max_candidates: None,
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_parquet(
        "t",
        indexed_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    let query_vec = get_embedding_at_row(indexed_path, embedding_column, 0)?;
    let query_literal = format!(
        "[{}]",
        query_vec
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let sql = format!(
        "SELECT title FROM t \
         ORDER BY array_distance(embedding, {query_literal}) \
         LIMIT 5"
    );
    let df = ctx.sql(&sql).await?;
    let batches = df.collect().await?;
    let formatted = arrow::util::pretty::pretty_format_batches(&batches)?;
    println!("{formatted}");

    Ok(())
}

fn get_embedding_at_row(
    path: &Path,
    column_name: &str,
    row: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
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
