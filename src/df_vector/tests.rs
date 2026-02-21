use std::sync::Arc;

use arrow::array::types::Float32Type;
use arrow::array::{Array, Float32Array, Int32Array, ListArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use insta::assert_snapshot;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

use super::{VectorTopKOptions, VectorTopKPhysicalOptimizerRule};
use crate::ivf::IndexBuilder;

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
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    IndexBuilder::new(source_path.as_path(), "vec")
        .build_new(indexed_path.as_path())
        .unwrap();

    let options = VectorTopKOptions {
        nprobe: 64,
        max_candidates: None,
        ..VectorTopKOptions::default()
    };
    let config = SessionConfig::new().with_target_partitions(2);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
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
    let batches = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
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

    let tree_str = displayable(plan.as_ref()).tree_render().to_string();
    assert_snapshot!("vector_topk_plan_tree", tree_str);
    Ok(())
}

#[tokio::test]
async fn vector_topk_vldb_tree_snapshot() -> datafusion::common::Result<()> {
    let source_path = std::path::Path::new("data/vldb_2025.parquet");
    let indexed_path = std::path::Path::new("data/vldb_2025_indexed.parquet");
    if !source_path.exists() || !indexed_path.exists() {
        return Ok(());
    }

    let options = VectorTopKOptions {
        nprobe: 32,
        max_candidates: Some(2048),
        ..VectorTopKOptions::default()
    };
    let ctx = build_context(options);
    ctx.register_parquet(
        "t",
        indexed_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let query_vec = get_embedding_at_row(indexed_path, "embedding", 0)?;
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
         LIMIT 3"
    );
    let df = ctx.sql(&sql).await.unwrap();
    let plan = df.clone().create_physical_plan().await?;
    let batches = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
    assert!(!batches.is_empty());

    let tree_str = displayable(plan.as_ref()).tree_render().to_string();
    assert_snapshot!("vector_topk_vldb_tree", tree_str);
    Ok(())
}

#[tokio::test]
async fn vector_topk_applies_filters_after_candidate_pruning() -> datafusion::common::Result<()> {
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
        Some(vec![Some(0.05), Some(0.05)]),
        Some(vec![Some(0.2), Some(0.2)]),
        Some(vec![Some(1.0), Some(1.0)]),
        Some(vec![Some(1.1), Some(1.1)]),
        Some(vec![Some(1.4), Some(1.4)]),
    ];
    let vec_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vectors);
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(vec_array)],
    )?;

    let file = std::fs::File::create(&source_path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    IndexBuilder::new(source_path.as_path(), "vec")
        .build_new(indexed_path.as_path())
        .unwrap();

    let options = VectorTopKOptions {
        nprobe: 64,
        max_candidates: None,
        ..VectorTopKOptions::default()
    };
    let config = SessionConfig::new().with_target_partitions(2);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
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
            "SELECT id FROM t \
             WHERE id >= 3 \
             ORDER BY array_distance(vec, [0.0, 0.0]) \
             LIMIT 2",
        )
        .await
        .unwrap();

    let plan = df.clone().create_physical_plan().await?;
    let batches = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;

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

    assert_eq!(result_ids, vec![3, 4]);

    let tree_str = displayable(plan.as_ref()).tree_render().to_string();
    assert_snapshot!("vector_topk_filter_plan_tree", tree_str);

    Ok(())
}

#[tokio::test]
async fn vector_topk_supports_complex_where_clause() -> datafusion::common::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let source_path = temp_dir.path().join("source.parquet");
    let indexed_path = temp_dir.path().join("indexed.parquet");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "group",
            DataType::Int32,
            true,
        ),
        Field::new(
            "vec",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
    ]));

    let ids = Int32Array::from(vec![0, 1, 2, 3, 4, 5]);
    let groups = Int32Array::from(vec![Some(1), None, Some(2), Some(3), None, Some(5)]);
    let vectors = vec![
        Some(vec![Some(0.0), Some(0.0)]),
        Some(vec![Some(0.1), Some(0.1)]),
        Some(vec![Some(0.2), Some(0.2)]),
        Some(vec![Some(1.0), Some(1.0)]),
        Some(vec![Some(0.05), Some(0.0)]),
        Some(vec![Some(10.0), Some(10.0)]),
    ];
    let vec_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vectors);
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(groups), Arc::new(vec_array)],
    )?;

    let file = std::fs::File::create(&source_path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    IndexBuilder::new(source_path.as_path(), "vec")
        .build_new(indexed_path.as_path())
        .unwrap();

    let options = VectorTopKOptions {
        nprobe: 64,
        post_filter_selectivity_threshold: 0.001,
        ..VectorTopKOptions::default()
    };
    let config = SessionConfig::new().with_target_partitions(2);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
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
            "SELECT id FROM t \
             WHERE (group IN (1, 3) OR group IS NULL) AND id <> 5 \
             ORDER BY array_distance(vec, [0.0, 0.0]) \
             LIMIT 2",
        )
        .await
        .unwrap();

    let batches = datafusion::physical_plan::collect(df.clone().create_physical_plan().await?, ctx.task_ctx()).await?;
    let ids = collect_i32_ids(batches);
    assert_eq!(ids, vec![0, 4]);

    let plan = df.create_physical_plan().await?;
    let tree_str = displayable(plan.as_ref()).tree_render().to_string();
    assert!(tree_str.contains("FilterExec"));
    assert!(tree_str.contains("VectorTopKExec"));
    Ok(())
}

#[tokio::test]
async fn vector_topk_uses_post_filter_for_selective_predicate() -> datafusion::common::Result<()> {
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
        Some(vec![Some(0.1), Some(0.1)]),
        Some(vec![Some(0.2), Some(0.2)]),
        Some(vec![Some(1.0), Some(1.0)]),
        Some(vec![Some(1.1), Some(1.1)]),
        Some(vec![Some(1.5), Some(1.5)]),
    ];
    let vec_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vectors);
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(vec_array)],
    )?;

    let file = std::fs::File::create(&source_path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    IndexBuilder::new(source_path.as_path(), "vec")
        .build_new(indexed_path.as_path())
        .unwrap();

    let options = VectorTopKOptions {
        nprobe: 64,
        post_filter_selectivity_threshold: 0.2,
        ..VectorTopKOptions::default()
    };
    let config = SessionConfig::new().with_target_partitions(2);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
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
            "SELECT id FROM t \
             WHERE id = 1 \
             ORDER BY array_distance(vec, [0.0, 0.0]) \
             LIMIT 2",
        )
        .await
        .unwrap();

    let batches = datafusion::physical_plan::collect(df.clone().create_physical_plan().await?, ctx.task_ctx()).await?;
    let ids = collect_i32_ids(batches);
    assert_eq!(ids, vec![1]);

    let plan = df.create_physical_plan().await?;
    let tree_str = displayable(plan.as_ref()).tree_render().to_string();
    let filter_pos = tree_str.find("FilterExec");
    let topk_pos = tree_str.find("VectorTopKExec");
    assert!(filter_pos.is_some() && topk_pos.is_some());
    assert!(filter_pos.unwrap() < topk_pos.unwrap());
    Ok(())
}

#[tokio::test]
async fn vector_topk_no_filter_fallback_and_empty_where() -> datafusion::common::Result<()> {
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

    let ids = Int32Array::from(vec![0, 1, 2]);
    let vectors = vec![
        Some(vec![Some(0.0), Some(0.0)]),
        Some(vec![Some(1.0), Some(1.0)]),
        Some(vec![Some(2.0), Some(2.0)]),
    ];
    let vec_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vectors);
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(vec_array)],
    )?;

    let file = std::fs::File::create(&source_path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    IndexBuilder::new(source_path.as_path(), "vec")
        .build_new(indexed_path.as_path())
        .unwrap();

    let options = VectorTopKOptions::default();
    let config = SessionConfig::new().with_target_partitions(2);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_parquet(
        "t",
        indexed_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let no_filter_plan = ctx
        .sql(
            "SELECT id FROM t \
             ORDER BY array_distance(vec, [0.0, 0.0]) \
             LIMIT 2",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await?;

    let no_filter_tree = displayable(no_filter_plan.as_ref()).tree_render().to_string();
    assert!(no_filter_tree.contains("VectorTopKExec"));
    assert!(!no_filter_tree.contains("FilterExec"));

    let empty_filter_plan = ctx
        .sql(
            "SELECT id FROM t \
             WHERE id IN (99, 100) \
             ORDER BY array_distance(vec, [0.0, 0.0]) \
             LIMIT 2",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await?;
    let empty_filter_tree = displayable(empty_filter_plan.as_ref()).tree_render().to_string();
    assert!(empty_filter_tree.contains("VectorTopKExec"));
    assert!(empty_filter_tree.contains("FilterExec"));

    let no_filter_result = datafusion::physical_plan::collect(
        ctx.sql(
            "SELECT id FROM t \
             WHERE id IN (99, 100) \
             ORDER BY array_distance(vec, [0.0, 0.0]) \
             LIMIT 2",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await?,
        ctx.task_ctx(),
    )
    .await?;
    let ids = collect_i32_ids(no_filter_result);
    assert_eq!(ids, Vec::<i32>::new());

    Ok(())
}

fn collect_i32_ids(batches: Vec<arrow::record_batch::RecordBatch>) -> Vec<i32> {
    let mut ids = Vec::new();
    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..array.len() {
            ids.push(array.value(i));
        }
    }
    ids
}

fn build_context(options: VectorTopKOptions) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(2);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
        .build();
    SessionContext::new_with_state(state)
}

fn get_embedding_at_row(
    path: &std::path::Path,
    column_name: &str,
    row: usize,
) -> datafusion::common::Result<Vec<f32>> {
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
    Err(datafusion::common::DataFusionError::Execution(
        "Row not found".to_string(),
    ))
}
