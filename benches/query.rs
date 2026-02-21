mod bench_util;

use mimalloc::MiMalloc;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
};
use arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pq_vector::IndexBuilder;
use pq_vector::df_vector::{PqVectorSessionBuilderExt, VectorTopKOptions};

use bench_util::{array_literal, generate_parquet, random_query, to_mb};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const ROWS: usize = 1_000_000;
const DIM: usize = 1024;
const BATCH_ROWS: usize = 2048;
const K: usize = 100;
const NPROBE: usize = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = BenchArgs::from_env();
    let data_dir = Path::new("data");
    fs::create_dir_all(data_dir)?;

    let paths = resolve_parquet_paths(&args, data_dir)?;
    if args.source_path.is_none() {
        println!("=== Generating synthetic dataset ===");
        println!("rows={ROWS}, dim={DIM}, batch_rows={BATCH_ROWS}");

        let gen_start = Instant::now();
        generate_parquet(&paths.source, ROWS, DIM, BATCH_ROWS)?;
        let gen_time = gen_start.elapsed();

        let original_size = fs::metadata(&paths.source)?.len();
        println!("Generated parquet in {:.2?}", gen_time);
        println!("Original parquet size: {:.2} MB", to_mb(original_size));
    }

    let schema = read_schema(&paths.source)?;
    let vector_column = args
        .vector_column
        .clone()
        .or_else(|| infer_vector_column(&schema))
        .unwrap_or_else(|| "embedding".to_string());
    if schema.field_with_name(&vector_column).is_err() {
        return Err(format!("vector column '{vector_column}' not found in schema").into());
    }
    let id_column = args
        .id_column
        .clone()
        .filter(|col| schema.field_with_name(col).is_ok())
        .or_else(|| schema.field_with_name("id").ok().map(|_| "id".to_string()));
    let query = if let Some(row) = args.query_row {
        read_vector_at_row(&paths.source, &vector_column, row)?
    } else {
        let query_dim = infer_vector_dim(&paths.source, &vector_column)?;
        random_query(query_dim, 7)
    };
    let k = args.k.unwrap_or(K);
    let nprobe = args.nprobe.unwrap_or(NPROBE);

    println!("\n=== Vector search without index (DataFusion) ===");
    let query_literal = array_literal(&query);
    let select_expr = id_column.as_deref().unwrap_or(&vector_column).to_string();
    let sql = format!(
        "SELECT {select_expr} FROM t ORDER BY array_distance({vector_column}, {query_literal}) LIMIT {k}"
    );

    let plain_ctx = SessionContext::new();
    plain_ctx
        .register_parquet(
            "t",
            paths.source.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    let plain_df = plain_ctx.sql(&sql).await?;
    let plain_plan = plain_df.create_physical_plan().await?;
    let plain_start = Instant::now();
    let plain_batches = collect(plain_plan.clone(), plain_ctx.task_ctx()).await?;
    let plain_time = plain_start.elapsed();
    let plain_keys = extract_keys(&plain_batches, id_column.is_some())?;
    let plain_rows: usize = plain_batches.iter().map(|b| b.num_rows()).sum();
    println!("Query time (no index): {:.2?}", plain_time);
    println!("Returned rows: {plain_rows}");
    if args.metrics {
        println!("--- Metrics (no index) ---");
        print_plan_metrics(&plain_plan, 0);
    }

    let source_size = fs::metadata(&paths.source)?.len();
    let mut inplace_path = None;
    let mut rewrite_path = None;

    if matches!(args.build_mode, BuildMode::Rewrite | BuildMode::Both) {
        println!("\n=== Building IVF index (rewrite, defaults) ===");
        if paths.rewrite.exists() {
            fs::remove_file(&paths.rewrite)?;
        }
        let index_start = Instant::now();
        let mut builder = IndexBuilder::new(&paths.source_for_rewrite, &vector_column);
        if let Some(n_clusters) = args.n_clusters {
            builder = builder.n_clusters(n_clusters);
        }
        builder.build_new(&paths.rewrite)?;
        let index_time = index_start.elapsed();

        let indexed_size = fs::metadata(&paths.rewrite)?.len();
        println!("Index build time: {:.2?}", index_time);
        println!("Indexed parquet size: {:.2} MB", to_mb(indexed_size));
        println!(
            "Index overhead: {:.2} MB ({:.1}%)",
            to_mb(indexed_size.saturating_sub(source_size)),
            indexed_size.saturating_sub(source_size) as f64 / source_size as f64 * 100.0
        );
        rewrite_path = Some(paths.rewrite.clone());
    }

    if matches!(args.build_mode, BuildMode::Inplace | BuildMode::Both) {
        println!("\n=== Building IVF index (in-place, defaults) ===");
        let index_start = Instant::now();
        let mut builder = IndexBuilder::new(&paths.inplace, &vector_column);
        if let Some(n_clusters) = args.n_clusters {
            builder = builder.n_clusters(n_clusters);
        }
        builder.build_inplace()?;
        let index_time = index_start.elapsed();

        let indexed_size = fs::metadata(&paths.inplace)?.len();
        println!("Index build time: {:.2?}", index_time);
        println!("Indexed parquet size: {:.2} MB", to_mb(indexed_size));
        println!(
            "Index overhead: {:.2} MB ({:.1}%)",
            to_mb(indexed_size.saturating_sub(source_size)),
            indexed_size.saturating_sub(source_size) as f64 / source_size as f64 * 100.0
        );
        inplace_path = Some(paths.inplace.clone());
    }

    let options = VectorTopKOptions {
        nprobe,
        max_candidates: args.max_candidates,
        ..VectorTopKOptions::default()
    };
    let indexed_paths = [inplace_path, rewrite_path];
    for (idx, path) in indexed_paths.into_iter().enumerate() {
        let Some(path) = path else { continue };
        let label = match (idx, args.build_mode) {
            (0, BuildMode::Both) => "in-place",
            (1, BuildMode::Both) => "rewrite",
            (_, BuildMode::Inplace) => "in-place",
            (_, BuildMode::Rewrite) => "rewrite",
            _ => "index",
        };
        println!("\n=== Vector search with index ({label}) ===");

        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_pq_vector(options.clone())
            .build();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
            .await?;
        let indexed_df = ctx.sql(&sql).await?;
        let indexed_plan = indexed_df.create_physical_plan().await?;
        let indexed_start = Instant::now();
        let indexed_batches = collect(indexed_plan.clone(), ctx.task_ctx()).await?;
        let indexed_time = indexed_start.elapsed();
        let indexed_keys = extract_keys(&indexed_batches, id_column.is_some())?;
        let indexed_rows: usize = indexed_batches.iter().map(|b| b.num_rows()).sum();
        println!("Query time (with index): {:.2?}", indexed_time);
        println!("Returned rows: {indexed_rows}");
        if args.metrics {
            println!("--- Metrics (with index) ---");
            print_plan_metrics(&indexed_plan, 0);
        }

        let recall = recall_at_k(&plain_keys, &indexed_keys);
        println!("Recall@{k}: {:.2}%", recall * 100.0);
    }

    Ok(())
}

#[derive(Debug, Default)]
struct BenchArgs {
    source_path: Option<PathBuf>,
    vector_column: Option<String>,
    id_column: Option<String>,
    in_place: bool,
    build_mode: BuildMode,
    nprobe: Option<usize>,
    max_candidates: Option<usize>,
    n_clusters: Option<usize>,
    k: Option<usize>,
    query_row: Option<usize>,
    metrics: bool,
}

impl BenchArgs {
    fn from_env() -> Self {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self::default();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--path" => {
                    if let Some(value) = args.next() {
                        parsed.source_path = Some(PathBuf::from(value));
                    }
                }
                "--vector-column" => {
                    if let Some(value) = args.next() {
                        parsed.vector_column = Some(value);
                    }
                }
                "--id-column" => {
                    if let Some(value) = args.next() {
                        parsed.id_column = Some(value);
                    }
                }
                "--in-place" => {
                    parsed.in_place = true;
                }
                "--build-mode" => {
                    if let Some(value) = args.next() {
                        parsed.build_mode = BuildMode::from_str(&value);
                    }
                }
                "--rewrite" => {
                    parsed.build_mode = BuildMode::Rewrite;
                }
                "--both" => {
                    parsed.build_mode = BuildMode::Both;
                }
                "--nprobe" => {
                    if let Some(value) = args.next() {
                        parsed.nprobe = value.parse().ok();
                    }
                }
                "--max-candidates" => {
                    if let Some(value) = args.next() {
                        parsed.max_candidates = value.parse().ok();
                    }
                }
                "--n-clusters" => {
                    if let Some(value) = args.next() {
                        parsed.n_clusters = value.parse().ok();
                    }
                }
                "--k" => {
                    if let Some(value) = args.next() {
                        parsed.k = value.parse().ok();
                    }
                }
                "--query-row" => {
                    if let Some(value) = args.next() {
                        parsed.query_row = value.parse().ok();
                    }
                }
                "--metrics" => {
                    parsed.metrics = true;
                }
                _ => {}
            }
        }
        parsed
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum BuildMode {
    #[default]
    Inplace,
    Rewrite,
    Both,
}

impl BuildMode {
    fn from_str(value: &str) -> Self {
        match value {
            "rewrite" => Self::Rewrite,
            "both" => Self::Both,
            _ => Self::Inplace,
        }
    }
}

struct ParquetPaths {
    source: PathBuf,
    inplace: PathBuf,
    rewrite: PathBuf,
    source_for_rewrite: PathBuf,
}

fn resolve_parquet_paths(
    args: &BenchArgs,
    data_dir: &Path,
) -> Result<ParquetPaths, Box<dyn std::error::Error>> {
    let Some(source) = &args.source_path else {
        let source = data_dir.join("benchmark.parquet");
        if source.exists() {
            fs::remove_file(&source)?;
        }
        let inplace = source.clone();
        let rewrite = data_dir.join("benchmark_rewrite.parquet");
        let source_for_rewrite = source.clone();
        return Ok(ParquetPaths {
            source,
            inplace,
            rewrite,
            source_for_rewrite,
        });
    };

    let source = source.clone();
    let needs_inplace_copy = matches!(args.build_mode, BuildMode::Inplace | BuildMode::Both);
    let inplace = if args.in_place || !needs_inplace_copy {
        source.clone()
    } else {
        let work_path = data_dir.join("benchmark_custom.parquet");
        if work_path.exists() {
            fs::remove_file(&work_path)?;
        }
        fs::copy(&source, &work_path)?;
        work_path
    };

    let rewrite = data_dir.join("benchmark_custom_rewrite.parquet");
    let source_for_rewrite =
        if !args.in_place && matches!(args.build_mode, BuildMode::Inplace | BuildMode::Both) {
            inplace.clone()
        } else {
            source.clone()
        };

    Ok(ParquetPaths {
        source,
        inplace,
        rewrite,
        source_for_rewrite,
    })
}

fn read_schema(path: &Path) -> Result<SchemaRef, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder.schema().clone())
}

fn infer_vector_column(schema: &SchemaRef) -> Option<String> {
    schema.fields().iter().find_map(|field| {
        if is_vector_field(field) {
            Some(field.name().to_string())
        } else {
            None
        }
    })
}

fn is_vector_field(field: &Field) -> bool {
    match field.data_type() {
        DataType::List(inner) | DataType::LargeList(inner) => {
            matches!(inner.data_type(), DataType::Float32 | DataType::Float64)
        }
        DataType::FixedSizeList(inner, _) => {
            matches!(inner.data_type(), DataType::Float32 | DataType::Float64)
        }
        _ => false,
    }
}

fn infer_vector_dim(path: &Path, vector_column: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;
    let Some(batch) = reader.next().transpose()? else {
        return Err("parquet file is empty".into());
    };
    let array = batch
        .column_by_name(vector_column)
        .ok_or_else(|| format!("vector column '{vector_column}' not found"))?;

    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        return Ok(list.value_length().try_into()?);
    }
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        return first_list_length(list);
    }
    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        return first_large_list_length(list);
    }

    Err("vector column must be list or fixed-size list".into())
}

fn read_vector_at_row(
    path: &Path,
    vector_column: &str,
    row: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;
    let mut current_row = 0usize;
    while let Some(batch) = reader.next().transpose()? {
        let array = batch
            .column_by_name(vector_column)
            .ok_or_else(|| format!("vector column '{vector_column}' not found"))?;
        let row_count = batch.num_rows();
        if row < current_row + row_count {
            let local_row = row - current_row;
            return extract_vector_row(array.as_ref(), local_row);
        }
        current_row += row_count;
    }
    Err("query row not found".into())
}

fn extract_vector_row(
    array: &dyn Array,
    row: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        return extract_vector_values(&list.value(row));
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        return extract_vector_values(&list.value(row));
    }
    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        return extract_vector_values(&list.value(row));
    }
    Err("vector column must be list or fixed-size list".into())
}

fn extract_vector_values(
    values: &arrow::array::ArrayRef,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if let Some(floats) = values.as_any().downcast_ref::<Float32Array>() {
        return Ok(floats.values().to_vec());
    }
    if let Some(floats) = values.as_any().downcast_ref::<Float64Array>() {
        return Ok(floats.values().iter().map(|v| *v as f32).collect());
    }
    Err("vector column must be float32 or float64 list".into())
}

fn print_plan_metrics(plan: &Arc<dyn ExecutionPlan>, indent: usize) {
    let prefix = " ".repeat(indent);
    if let Some(metrics) = plan.metrics() {
        let mut lines = Vec::new();
        for metric in metrics.iter() {
            lines.push(format!("{metric}"));
        }
        if !lines.is_empty() {
            println!("{prefix}{} metrics:", plan.name());
            for line in lines {
                println!("{prefix}  {line}");
            }
        }
    }
    for child in plan.children() {
        print_plan_metrics(child, indent + 2);
    }
}

fn first_list_length(list: &ListArray) -> Result<usize, Box<dyn std::error::Error>> {
    for row in 0..list.len() {
        if !list.is_null(row) {
            return Ok(list.value_length(row) as usize);
        }
    }
    Err("vector column contains only nulls".into())
}

fn first_large_list_length(list: &LargeListArray) -> Result<usize, Box<dyn std::error::Error>> {
    for row in 0..list.len() {
        if !list.is_null(row) {
            return Ok(list.value_length(row) as usize);
        }
    }
    Err("vector column contains only nulls".into())
}

fn extract_keys(
    batches: &[arrow::record_batch::RecordBatch],
    use_id: bool,
) -> Result<Vec<Key>, Box<dyn std::error::Error>> {
    let mut keys = Vec::new();
    for batch in batches {
        let array = batch.column(0);
        for row in 0..batch.num_rows() {
            let key = if use_id {
                let value = ScalarValue::try_from_array(array.as_ref(), row)?;
                Key::Id(value.to_string())
            } else {
                Key::Hash(hash_vector(array.as_ref(), row)?)
            };
            keys.push(key);
        }
    }
    Ok(keys)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Id(String),
    Hash(u64),
}

fn hash_vector(array: &dyn Array, row: usize) -> Result<u64, Box<dyn std::error::Error>> {
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        let values = list.value(row);
        return hash_vector_values(&values);
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let values = list.value(row);
        return hash_vector_values(&values);
    }
    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        let values = list.value(row);
        return hash_vector_values(&values);
    }
    Err("vector column must be list or fixed-size list".into())
}

fn hash_vector_values(values: &arrow::array::ArrayRef) -> Result<u64, Box<dyn std::error::Error>> {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;

    if let Some(floats) = values.as_any().downcast_ref::<Float32Array>() {
        for value in floats.values() {
            hash ^= value.to_bits() as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        return Ok(hash);
    }
    if let Some(floats) = values.as_any().downcast_ref::<Float64Array>() {
        for value in floats.values() {
            hash ^= value.to_bits();
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        return Ok(hash);
    }
    Err("vector column must be float32 or float64 list".into())
}

fn recall_at_k(ground_truth: &[Key], candidates: &[Key]) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let truth: HashSet<&Key> = ground_truth.iter().collect();
    let hits = candidates.iter().filter(|id| truth.contains(id)).count();
    hits as f64 / ground_truth.len() as f64
}
