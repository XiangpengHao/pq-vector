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
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, ListArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::{plan_err, ScalarValue};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;

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
        let path = match &exprs[0] {
            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => PathBuf::from(s),
            _ => return plan_err!("First argument (path) must be a string literal"),
        };

        // Argument 2: query vector (array of floats)
        let query_vector = extract_float_array(&exprs[1])?;

        // Argument 3: k (integer)
        let k = match &exprs[2] {
            Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v as usize,
            Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as usize,
            Expr::Literal(ScalarValue::UInt64(Some(v)), _) => *v as usize,
            _ => return plan_err!("Third argument (k) must be an integer literal"),
        };

        // Argument 4: nprobe (optional integer, default 10)
        let nprobe = if exprs.len() > 3 {
            match &exprs[3] {
                Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v as usize,
                Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as usize,
                Expr::Literal(ScalarValue::UInt64(Some(v)), _) => *v as usize,
                _ => return plan_err!("Fourth argument (nprobe) must be an integer literal"),
            }
        } else {
            10 // default nprobe
        };

        // Execute the search
        let results = topk(&path, &query_vector, k, nprobe).map_err(|e| {
            datafusion::error::DataFusionError::Execution(e.to_string())
        })?;

        // Read the original data to join with results
        let provider = TopkResultProvider::new(path, results, query_vector)?;

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

/// Table provider that holds the results of an IVF search.
///
/// This provider joins the search results (row_idx, distance) with the
/// original parquet data to return complete rows.
#[derive(Debug)]
struct TopkResultProvider {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl TopkResultProvider {
    fn new(
        parquet_path: PathBuf,
        results: Vec<SearchResult>,
        _query_vector: Vec<f32>,
    ) -> Result<Self> {
        use arrow::compute::concat_batches;

        // Read original parquet data
        let file = std::fs::File::open(&parquet_path).map_err(|e| {
            datafusion::error::DataFusionError::External(Box::new(e))
        })?;

        let builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
        let parquet_schema = builder.schema().clone();
        let reader = builder.build()?;

        // Read all batches into memory and concatenate
        let mut all_batches: Vec<RecordBatch> = Vec::new();
        for batch in reader {
            all_batches.push(batch?);
        }

        // Build schema: original columns + _distance + _row_idx
        let mut fields: Vec<Field> = parquet_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new("_distance", DataType::Float32, false));
        fields.push(Field::new("_row_idx", DataType::UInt32, false));
        let schema = Arc::new(Schema::new(fields));

        // Handle empty results
        if results.is_empty() {
            return Ok(Self {
                schema,
                batches: vec![],
            });
        }

        // Concatenate all batches into one
        let combined = concat_batches(&parquet_schema, &all_batches)?;

        // Create indices array for take operation (sorted by distance, i.e., result order)
        let indices: Vec<u32> = results.iter().map(|r| r.row_idx).collect();
        let indices_array = arrow::array::UInt32Array::from(indices.clone());

        // Use take to extract rows in the desired order
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(combined.num_columns() + 2);
        for col in combined.columns() {
            let taken = arrow::compute::take(col, &indices_array, None)?;
            columns.push(taken);
        }

        // Add _distance column
        let distances: Vec<f32> = results.iter().map(|r| r.distance).collect();
        columns.push(Arc::new(Float32Array::from(distances)));

        // Add _row_idx column
        columns.push(Arc::new(arrow::array::UInt32Array::from(indices)));

        let result_batch = RecordBatch::try_new(schema.clone(), columns)?;

        Ok(Self {
            schema,
            batches: vec![result_batch],
        })
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
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(MemorySourceConfig::try_new_exec(
            &[self.batches.clone()],
            self.schema.clone(),
            projection.cloned(),
        )?)
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
/// Use `encode_query_vector()` to create the base64 string from a Vec<f32>.
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
        let path = match &exprs[0] {
            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => PathBuf::from(s),
            _ => return plan_err!("First argument (path) must be a string literal"),
        };

        // Argument 2: base64-encoded query vector
        let query_vector = match &exprs[1] {
            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => decode_query_vector(s)?,
            _ => return plan_err!("Second argument (query_vector) must be a base64-encoded string"),
        };

        // Argument 3: k (integer)
        let k = match &exprs[2] {
            Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v as usize,
            Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as usize,
            Expr::Literal(ScalarValue::UInt64(Some(v)), _) => *v as usize,
            _ => return plan_err!("Third argument (k) must be an integer literal"),
        };

        // Argument 4: nprobe (optional integer, default 10)
        let nprobe = if exprs.len() > 3 {
            match &exprs[3] {
                Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v as usize,
                Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as usize,
                Expr::Literal(ScalarValue::UInt64(Some(v)), _) => *v as usize,
                _ => return plan_err!("Fourth argument (nprobe) must be an integer literal"),
            }
        } else {
            10
        };

        // Execute the search
        let results = topk(&path, &query_vector, k, nprobe).map_err(|e| {
            datafusion::error::DataFusionError::Execution(e.to_string())
        })?;

        let provider = TopkResultProvider::new(path, results, query_vector)?;
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
