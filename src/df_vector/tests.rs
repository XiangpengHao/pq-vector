use std::sync::Arc;

use arrow::array::{Int32Array, ListArray};
use arrow::array::types::Float32Type;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use tempfile::TempDir;

use crate::ivf::{build_index, IvfBuildParams};
use super::{VectorTopKOptimizerRule, VectorTopKOptions, VectorTopKQueryPlanner};

#[tokio::test]
async fn vector_topk_end_to_end() -> datafusion::common::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let source_path = temp_dir.path().join("source.parquet");
    let indexed_path = temp_dir.path().join("indexed.parquet");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
    ]));

    let ids = Int32Array::from(vec![0, 1, 2, 3, 4, 5]);
    let vectors = vec![
        Some(vec![Some(0.0), Some(0.0)]),
        Some(vec![Some(1.0), Some(0.0)]),
        Some(vec![Some(0.0), Some(2.0)]),
        Some(vec![Some(5.0), Some(5.0)]),
        Some(vec![Some(2.0), Some(2.0)]),
        Some(vec![Some(0.1), Some(0.1)]),
    ];
    let vec_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vectors);
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(vec_array)],
    )?;

    let file = std::fs::File::create(&source_path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None)
        .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    build_index(
        source_path.as_path(),
        indexed_path.as_path(),
        "vec",
        &IvfBuildParams::default(),
    )
    .unwrap();

    let options = VectorTopKOptions {
        nprobe: 64,
        batch_size: 1024,
        max_candidates: None,
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(VectorTopKQueryPlanner::new(options)))
        .with_optimizer_rule(Arc::new(VectorTopKOptimizerRule::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_parquet(
        "t",
        indexed_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let df = ctx
        .sql(
            "SELECT id, vec FROM t \
             WHERE id >= 2 \
             ORDER BY array_distance(vec, [0.0, 0.0]) \
             LIMIT 2",
        )
        .await
        .unwrap();

    let plan = df.clone().create_physical_plan().await?;
    let plan_str = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_str.contains("VectorTopKExec"),
        "expected VectorTopKExec in plan, got: {plan_str}"
    );

    let batches = df.collect().await?;
    let mut result_ids = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..ids.len() {
            result_ids.push(ids.value(i));
        }
    }

    assert_eq!(result_ids, vec![5, 2]);
    Ok(())
}
