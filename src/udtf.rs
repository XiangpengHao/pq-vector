//! DataFusion UDTF (User-Defined Table Function) for IVF vector search.
//!
//! This module provides table functions for vector similarity search:
//!
//! ## `topk` - Array-based query vector
//! ```sql
//! SELECT * FROM topk('/path/to/file.parquet', ARRAY[1.0, 2.0, ...], 10, 5)
//! ```
//!
//! ## `topk_bin` - Binary (base64) query vector for large dimensions
//! ```sql
//! SELECT * FROM topk_bin('/path/to/file.parquet', 'base64_encoded_floats', 10, 5)
//! ```
//!
//! Arguments:
//! - parquet_path: Path to parquet file with embedded IVF index
//! - query_vector: Query vector (array literal or base64-encoded f32 bytes)
//! - k: Number of results to return
//! - nprobe: Number of clusters to search (optional, default 10)

use std::any::Any;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, Float32Array, ListArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{exec_err, plan_err, DFSchema, ScalarValue};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::physical_plan::{FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    Statistics,
};
use futures::StreamExt;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::file::metadata::ParquetMetaData;

use crate::ivf::{topk, SearchResult};

/// Table function that performs IVF vector search on parquet files.
///
/// Usage in SQL: `topk(path, query_vector, k, nprobe)`
///
/// For large dimension vectors (e.g., 4096), use `TopkBinaryTableFunction` instead
/// to avoid SQL parsing limitations with large array literals.
#[derive(Debug)]
pub struct TopkTableFunction;

impl TableFunctionImpl for TopkTableFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        // Parse arguments
        if exprs.len() < 3 {
            return plan_err!(
                "topk requires at least 3 arguments: path, query_vector, k. Got {}",
                exprs.len()
            );
        }

        // Argument 1: parquet path (string)
        let path = parse_parquet_path(&exprs[0])?;

        // Argument 2: query vector (array of floats)
        let query_vector = extract_float_array(&exprs[1])?;

        // Argument 3: k (integer)
        let k = parse_usize_literal(&exprs[2], "Third argument (k)")?;

        // Argument 4: nprobe (optional integer, default 10)
        let nprobe = parse_nprobe(exprs)?;

        let provider = TopkResultProvider::new(path, query_vector, k, nprobe)?;

        Ok(Arc::new(provider))
    }
}

/// Extract a float array from a literal expression.
fn extract_float_array(expr: &Expr) -> Result<Vec<f32>> {
    match expr {
        // Handle array literal: ARRAY[1.0, 2.0, ...]
        Expr::Literal(ScalarValue::List(array), _) => {
            let list_array = array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                datafusion::error::DataFusionError::Plan("Expected list array".to_string())
            })?;

            if list_array.is_empty() {
                return plan_err!("Query vector cannot be empty");
            }

            let values = list_array.value(0);
            extract_floats_from_array(&values)
        }
        // Handle FixedSizeList
        Expr::Literal(ScalarValue::FixedSizeList(array), _) => {
            let values = array.values();
            extract_floats_from_array(values)
        }
        _ => plan_err!(
            "Query vector must be an array literal (ARRAY[1.0, 2.0, ...]). Got: {:?}",
            expr
        ),
    }
}

fn extract_floats_from_array(array: &ArrayRef) -> Result<Vec<f32>> {
    // Try Float32
    if let Some(f32_array) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(f32_array.iter().filter_map(|v| v).collect());
    }

    // Try Float64 and convert
    if let Some(f64_array) = array
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
    {
        return Ok(f64_array.iter().filter_map(|v| v.map(|x| x as f32)).collect());
    }

    plan_err!("Query vector values must be float32 or float64")
}

/// Table provider that runs IVF search and returns matching parquet rows.
///
/// This provider builds a row selection from the IVF index and returns a
/// DataSourceExec that reads only the selected rows from parquet.
#[derive(Debug)]
struct TopkResultProvider {
    schema: SchemaRef,
    file_schema: SchemaRef,
    metadata: Arc<ParquetMetaData>,
    parquet_path: PathBuf,
    query_vector: Vec<f32>,
    k: usize,
    nprobe: usize,
}

impl TopkResultProvider {
    fn new(
        parquet_path: PathBuf,
        query_vector: Vec<f32>,
        k: usize,
        nprobe: usize,
    ) -> Result<Self> {
        let file = std::fs::File::open(&parquet_path).map_err(|e| {
            datafusion::error::DataFusionError::External(Box::new(e))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let parquet_schema = builder.schema().clone();
        let metadata = builder.metadata().clone();

        // Build schema: original columns + _distance
        let mut fields: Vec<Field> = parquet_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(DISTANCE_COLUMN, DataType::Float32, false));
        let schema = Arc::new(Schema::new(fields));

        Ok(Self {
            schema,
            file_schema: parquet_schema,
            metadata,
            parquet_path,
            query_vector,
            k,
            nprobe,
        })
    }
}

fn parse_parquet_path(expr: &Expr) -> Result<PathBuf> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _) => Ok(PathBuf::from(s)),
        _ => plan_err!("First argument (path) must be a string literal"),
    }
}

fn parse_usize_literal(expr: &Expr, label: &str) -> Result<usize> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(v)), _) => Ok(*v as usize),
        Expr::Literal(ScalarValue::Int32(Some(v)), _) => Ok(*v as usize),
        Expr::Literal(ScalarValue::UInt64(Some(v)), _) => Ok(*v as usize),
        _ => plan_err!("{} must be an integer literal", label),
    }
}

fn parse_nprobe(exprs: &[Expr]) -> Result<usize> {
    if exprs.len() > 3 {
        parse_usize_literal(&exprs[3], "Fourth argument (nprobe)")
    } else {
        Ok(10)
    }
}

const DISTANCE_COLUMN: &str = "_distance";

fn expr_references_distance(expr: &Expr) -> bool {
    let mut found = false;
    let _ = expr.apply(|expr| {
        if let Expr::Column(column) = expr {
            if column.name == DISTANCE_COLUMN {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn build_access_plan(
    metadata: &ParquetMetaData,
    results: &[SearchResult],
) -> Result<(ParquetAccessPlan, Vec<f32>)> {
    let num_row_groups = metadata.num_row_groups();
    if results.is_empty() {
        return Ok((ParquetAccessPlan::new_none(num_row_groups), Vec::new()));
    }

    let mut sorted = results.to_vec();
    sorted.sort_by_key(|r| r.row_idx);
    sorted.dedup_by_key(|r| r.row_idx);

    let mut distances = Vec::with_capacity(sorted.len());
    let mut row_group_starts = Vec::with_capacity(num_row_groups);
    let mut current_start = 0u64;
    for i in 0..num_row_groups {
        row_group_starts.push(current_start);
        current_start += metadata.row_group(i).num_rows() as u64;
    }

    let mut rows_per_group: Vec<Vec<u32>> = vec![Vec::new(); num_row_groups];
    for result in &sorted {
        distances.push(result.distance);
        let row = result.row_idx as u64;
        let mut group_idx = None;
        for (idx, start) in row_group_starts.iter().enumerate() {
            let end = if idx + 1 < num_row_groups {
                row_group_starts[idx + 1]
            } else {
                current_start
            };
            if row >= *start && row < end {
                group_idx = Some(idx);
                rows_per_group[idx].push((row - *start) as u32);
                break;
            }
        }
        if group_idx.is_none() {
            return plan_err!("Row index {} is out of bounds", row);
        }
    }

    let mut plan = ParquetAccessPlan::new_none(num_row_groups);
    for (group_idx, mut rows) in rows_per_group.into_iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        rows.sort_unstable();
        rows.dedup();

        let total_rows = metadata.row_group(group_idx).num_rows() as usize;
        let mut selectors = Vec::new();
        let mut current_pos = 0usize;
        for row in rows {
            let row = row as usize;
            if row > current_pos {
                selectors.push(RowSelector::skip(row - current_pos));
            }
            selectors.push(RowSelector::select(1));
            current_pos = row + 1;
        }
        if current_pos < total_rows {
            selectors.push(RowSelector::skip(total_rows - current_pos));
        }
        let selection = RowSelection::from(selectors);
        plan.scan_selection(group_idx, selection);
    }

    Ok((plan, distances))
}

#[derive(Debug, Clone)]
enum OutputColumn {
    Input(usize),
    Distance,
}

#[derive(Debug, Clone)]
struct ProjectionPlan {
    file_projection: Option<Vec<usize>>,
    output_schema: SchemaRef,
    output_columns: Vec<OutputColumn>,
    needs_distance: bool,
}

impl ProjectionPlan {
    fn try_new(
        table_schema: SchemaRef,
        file_schema_len: usize,
        projection: Option<&Vec<usize>>,
    ) -> Result<Self> {
        let (file_projection, output_schema, output_columns, needs_distance) = match projection {
            None => {
                let mut output_columns = Vec::with_capacity(file_schema_len + 1);
                for idx in 0..file_schema_len {
                    output_columns.push(OutputColumn::Input(idx));
                }
                output_columns.push(OutputColumn::Distance);
                (
                    None,
                    table_schema,
                    output_columns,
                    true,
                )
            }
            Some(indices) => {
                let mut file_projection = Vec::new();
                for idx in indices {
                    if *idx < file_schema_len {
                        file_projection.push(*idx);
                    }
                }
                let output_schema = Arc::new(table_schema.project(indices)?);
                let mut input_positions = vec![None; file_schema_len];
                for (pos, idx) in file_projection.iter().enumerate() {
                    input_positions[*idx] = Some(pos);
                }
                let mut output_columns = Vec::with_capacity(indices.len());
                let mut needs_distance = false;
                for idx in indices {
                    if *idx < file_schema_len {
                        let input_pos = input_positions[*idx].ok_or_else(|| {
                            datafusion::error::DataFusionError::Plan(format!(
                                "Missing projection for column {}",
                                idx
                            ))
                        })?;
                        output_columns.push(OutputColumn::Input(input_pos));
                    } else if *idx == file_schema_len {
                        output_columns.push(OutputColumn::Distance);
                        needs_distance = true;
                    } else {
                        return plan_err!("Invalid projection index {}", idx);
                    }
                }
                (
                    Some(file_projection),
                    output_schema,
                    output_columns,
                    needs_distance,
                )
            }
        };

        Ok(Self {
            file_projection,
            output_schema,
            output_columns,
            needs_distance,
        })
    }
}

#[derive(Debug, Clone)]
struct DistanceProjection {
    output_schema: SchemaRef,
    output_columns: Vec<OutputColumn>,
    distances: Arc<Vec<f32>>,
}

impl DistanceProjection {
    fn new(plan: ProjectionPlan, distances: Arc<Vec<f32>>) -> Self {
        Self {
            output_schema: plan.output_schema,
            output_columns: plan.output_columns,
            distances,
        }
    }
}

#[derive(Debug)]
struct AddDistanceExec {
    input: Arc<dyn ExecutionPlan>,
    projection: DistanceProjection,
    cache: PlanProperties,
}

impl AddDistanceExec {
    fn try_new(input: Arc<dyn ExecutionPlan>, projection: DistanceProjection) -> Result<Self> {
        let input_props = input.properties();
        let partition_count = input_props.partitioning.partition_count();
        if partition_count != 1 {
            return plan_err!(
                "AddDistanceExec requires a single partition, got {}",
                partition_count
            );
        }
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&projection.output_schema)),
            input_props.partitioning.clone(),
            input_props.emission_type,
            input_props.boundedness,
        );
        Ok(Self {
            input,
            projection,
            cache,
        })
    }
}

impl DisplayAs for AddDistanceExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "AddDistanceExec")
            }
            DisplayFormatType::TreeRender => write!(f, ""),
        }
    }
}

impl ExecutionPlan for AddDistanceExec {
    fn name(&self) -> &str {
        "AddDistanceExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return plan_err!("AddDistanceExec expects a single child");
        }
        Ok(Arc::new(AddDistanceExec::try_new(
            Arc::clone(&children[0]),
            self.projection.clone(),
        )?))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let output_schema = Arc::clone(&self.projection.output_schema);
        let stream_schema = Arc::clone(&output_schema);
        let output_columns = self.projection.output_columns.clone();
        let distances = Arc::clone(&self.projection.distances);
        let offset = Arc::new(Mutex::new(0usize));

        let stream = input.map(move |batch| {
            let batch = batch?;
            let num_rows = batch.num_rows();
            let distance_array = if output_columns
                .iter()
                .any(|col| matches!(col, OutputColumn::Distance))
            {
                let mut guard = offset.lock().unwrap();
                let start = *guard;
                let end = start + num_rows;
                if end > distances.len() {
                    return exec_err!(
                        "Distance values exhausted: need {}, have {}",
                        end,
                        distances.len()
                    );
                }
                *guard = end;
                Some(Arc::new(Float32Array::from(
                    distances[start..end].to_vec(),
                )) as ArrayRef)
            } else {
                None
            };

            let mut columns = Vec::with_capacity(output_columns.len());
            for col in &output_columns {
                match col {
                    OutputColumn::Input(idx) => columns.push(batch.column(*idx).clone()),
                    OutputColumn::Distance => {
                        let array = distance_array.clone().ok_or_else(|| {
                            datafusion::error::DataFusionError::Execution(
                                "Distance column requested but not available".to_string(),
                            )
                        })?;
                        columns.push(array);
                    }
                }
            }
            Ok(arrow::record_batch::RecordBatch::try_new(
                Arc::clone(&stream_schema),
                columns,
            )?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema()))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema()))
    }
}

#[async_trait]
impl TableProvider for TopkResultProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let results = topk(&self.parquet_path, &self.query_vector, self.k, self.nprobe)
            .await
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

        let (access_plan, distances) = build_access_plan(&self.metadata, &results)?;
        let projection_plan = ProjectionPlan::try_new(
            Arc::clone(&self.schema),
            self.file_schema.fields().len(),
            projection,
        )?;

        let file_size = std::fs::metadata(&self.parquet_path)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            .len();
        let partitioned_file = PartitionedFile::new(
            self.parquet_path.to_string_lossy().to_string(),
            file_size,
        )
        .with_extensions(Arc::new(access_plan));

        let pushdown_filters: Vec<Expr> = filters
            .iter()
            .cloned()
            .filter(|expr| !expr_references_distance(expr))
            .collect();

        let predicate = if pushdown_filters.is_empty() {
            None
        } else {
            let df_schema = DFSchema::try_from(Arc::clone(&self.file_schema))?;
            conjunction(pushdown_filters)
                .map(|expr| state.create_physical_expr(expr, &df_schema))
                .transpose()?
        };

        let mut source =
            ParquetSource::new(Arc::clone(&self.file_schema)).with_enable_page_index(true);
        if let Some(predicate) = predicate {
            source = source.with_predicate(predicate);
        }
        let source = Arc::new(source);

        let file_scan_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            source,
        )
        .with_file(partitioned_file)
        .with_projection_indices(projection_plan.file_projection.clone())?
        .with_limit(limit)
        .build();

        let scan = DataSourceExec::from_data_source(file_scan_config);
        if projection_plan.needs_distance {
            let projection =
                DistanceProjection::new(projection_plan, Arc::new(distances));
            return Ok(Arc::new(AddDistanceExec::try_new(scan, projection)?));
        }

        Ok(scan)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

/// Table function that performs IVF vector search using base64-encoded query vector.
///
/// This is useful for large dimension vectors (e.g., 4096) where SQL array literals
/// hit parsing limitations.
///
/// Usage in SQL: `topk_bin(path, base64_query_vector, k, nprobe)`
///
/// The query vector should be base64-encoded little-endian f32 bytes.
/// Use `encode_query_vector()` to create the base64 string from a `Vec<f32>`.
#[derive(Debug)]
pub struct TopkBinaryTableFunction;

impl TableFunctionImpl for TopkBinaryTableFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if exprs.len() < 3 {
            return plan_err!(
                "topk_bin requires at least 3 arguments: path, base64_query_vector, k. Got {}",
                exprs.len()
            );
        }

        // Argument 1: parquet path (string)
        let path = parse_parquet_path(&exprs[0])?;

        // Argument 2: base64-encoded query vector
        let query_vector = match &exprs[1] {
            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => decode_query_vector(s)?,
            _ => return plan_err!("Second argument (query_vector) must be a base64-encoded string"),
        };

        // Argument 3: k (integer)
        let k = parse_usize_literal(&exprs[2], "Third argument (k)")?;

        // Argument 4: nprobe (optional integer, default 10)
        let nprobe = parse_nprobe(exprs)?;

        let provider = TopkResultProvider::new(path, query_vector, k, nprobe)?;
        Ok(Arc::new(provider))
    }
}

/// Encode a query vector as a base64 string for use with `topk_bin`.
///
/// # Example
/// ```
/// use pq_vector::encode_query_vector;
///
/// let query = vec![1.0f32, 2.0, 3.0];
/// let encoded = encode_query_vector(&query);
/// // Use `encoded` in SQL: SELECT * FROM topk_bin('file.parquet', '{encoded}', 10, 5)
/// ```
pub fn encode_query_vector(query: &[f32]) -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    let bytes: Vec<u8> = query.iter().flat_map(|f| f.to_le_bytes()).collect();
    STANDARD.encode(&bytes)
}

/// Decode a base64-encoded query vector.
fn decode_query_vector(encoded: &str) -> Result<Vec<f32>> {
    use base64::{Engine, engine::general_purpose::STANDARD};

    let bytes = STANDARD.decode(encoded).map_err(|e| {
        datafusion::error::DataFusionError::Plan(format!("Invalid base64 encoding: {}", e))
    })?;

    if bytes.len() % 4 != 0 {
        return plan_err!(
            "Invalid query vector: byte length {} is not a multiple of 4",
            bytes.len()
        );
    }

    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok(floats)
}
