//! Physical execution for IVF candidate row-id scan.

use std::any::Any;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Float64Array, ListArray, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result, assert_eq_or_internal_err};
use datafusion::execution::TaskContext;
use datafusion::object_store::path::Path as ObjectStorePath;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricsSet,
    RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use futures::stream;
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::ProjectionMask;

use crate::hnsw::{
    read_index_from_payload as read_hnsw_index_from_payload, HnswIndex,
};
use crate::ivf::{
    read_index_from_payload, read_index_metadata_from_file_metadata, EmbeddingColumn, EmbeddingDim,
    Embeddings,
};

use super::access::ScanFile;
use super::options::VectorTopKOptions;

enum ParsedIndex {
    Ivf(crate::ivf::IvfIndex),
    Hnsw(HnswIndex),
}

pub(crate) const INDEX_PATH_COL: &str = "pq_vector_object_path";
pub(crate) const INDEX_ROW_ID_COL: &str = "pq_vector_row_id";

struct IndexFileCandidates {
    object_path: String,
    candidates: Vec<u32>,
}

/// Execution plan that reads IVF indexes and emits candidate `(file_path, row_id)` rows.
#[derive(Clone)]
pub(crate) struct VectorIndexScanExec {
    files: Vec<ScanFile>,
    vector_column: String,
    query: Vec<f32>,
    options: VectorTopKOptions,
    schema: SchemaRef,
    cache: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
    metric_handles: VectorIndexScanMetricHandles,
}

impl VectorIndexScanExec {
    pub(crate) fn new(
        files: Vec<ScanFile>,
        vector_column: String,
        query: Vec<f32>,
        options: VectorTopKOptions,
    ) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new(INDEX_PATH_COL, DataType::Utf8, false),
            Field::new(INDEX_ROW_ID_COL, DataType::UInt32, false),
        ]));
        let cache = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        let metrics = ExecutionPlanMetricsSet::new();
        let metric_handles = VectorIndexScanMetricHandles::new(&metrics, 0);
        Self {
            files,
            vector_column,
            query,
            options,
            schema,
            cache,
            metrics,
            metric_handles,
        }
    }

    async fn execute_index_scan(&self, context: Arc<TaskContext>) -> Result<RecordBatch> {
        let mut files = Vec::new();
        for file in self.files.iter() {
            let object_path = file.object_path.clone();
            let location = ObjectStorePath::parse(&object_path).map_err(|err| {
                DataFusionError::Execution(format!(
                    "Invalid object-store path '{}': {}",
                    object_path, err
                ))
            })?;
            let store = context
                .runtime_env()
                .object_store(file.object_store_url.clone())?;
            let mut reader = ParquetObjectReader::new(store.clone(), location.clone())
                .with_file_size(file.file_size);
            if let Some(hint) = file.metadata_size_hint {
                reader = reader.with_footer_size_hint(hint);
            }

            let metadata = reader.get_metadata(None).await.map_err(|err| {
                DataFusionError::Execution(format!(
                    "Failed to read parquet metadata from '{}': {}",
                    object_path, err
                ))
            })?;
            let (offset, embedding_column) =
                read_index_metadata_from_file_metadata(metadata.file_metadata())
                    .map_err(|err| {
                        DataFusionError::Execution(format!(
                            "Failed to parse pq-vector metadata from '{}': {}",
                            object_path, err
                        ))
                    })?
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Missing pq-vector index metadata in '{}'",
                            object_path
                        ))
                    })?;

            if embedding_column.as_str() != self.vector_column {
                return Err(DataFusionError::Execution(format!(
                    "IVF index column mismatch: expected '{}', found '{}'",
                    self.vector_column,
                    embedding_column.as_str()
                )));
            }
            if offset >= file.file_size {
                return Err(DataFusionError::Execution(format!(
                    "Invalid pq-vector index offset {} for '{}' with size {}",
                    offset, object_path, file.file_size
                )));
            }

            let range: Range<u64> = offset..file.file_size;
            let payload = store.get_range(&location, range).await.map_err(|err| {
                DataFusionError::Execution(format!(
                    "Failed to fetch pq-vector payload from '{}': {}",
                    object_path, err
                ))
            })?;
            let index = parse_index_payload(payload.as_ref(), &embedding_column).map_err(|err| {
                    DataFusionError::Execution(format!(
                        "Failed to decode pq-vector payload from '{}': {}",
                        object_path, err
                    ))
                })?;

            let (candidates, index_dim) = match index {
                ParsedIndex::Ivf(index) => (
                    index.candidate_rows(&self.query, self.options.nprobe),
                    index.dim(),
                ),
                ParsedIndex::Hnsw(index) => {
                    let all_embeddings = read_all_embeddings(
                        &object_path,
                        file.file_size,
                        &store,
                        file.metadata_size_hint,
                        embedding_column.as_str(),
                    )
                    .await?;
                    if all_embeddings.dim().as_usize() != self.query.len() {
                        return Err(DataFusionError::Plan(format!(
                            "Embedding dimension mismatch: expected {}, got {}",
                            all_embeddings.dim().as_usize(),
                            self.query.len()
                        )));
                    }
                    (
                        index.candidate_rows(&self.query, all_embeddings.data(), self.options.nprobe),
                        index.dim(),
                    )
                }
            };

            if index_dim != self.query.len() {
                return Err(DataFusionError::Plan(format!(
                    "Query dimension mismatch: expected {}, got {}",
                    index_dim,
                    self.query.len()
                )));
            }

            files.push(IndexFileCandidates {
                object_path,
                candidates,
            });
        }

        self.metric_handles.files_scanned.add(files.len());
        let candidate_rows = files.iter().map(|f| f.candidates.len()).sum::<usize>();
        self.metric_handles.candidate_rows.add(candidate_rows);

        let total_rows = files.iter().map(|f| f.candidates.len()).sum::<usize>();
        let mut paths = Vec::with_capacity(total_rows);
        let mut row_ids = Vec::with_capacity(total_rows);
        for file in files {
            for row in file.candidates {
                paths.push(file.object_path.clone());
                row_ids.push(row);
            }
        }

        RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(StringArray::from(paths)),
                Arc::new(UInt32Array::from(row_ids)),
            ],
        )
        .map_err(DataFusionError::from)
    }
}

impl fmt::Debug for VectorIndexScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorIndexScanExec")
    }
}

impl DisplayAs for VectorIndexScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "VectorIndexScanExec")
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "vector_index_scan")?;
                writeln!(f, "files={}", self.files.len())?;
                writeln!(
                    f,
                    "files_scanned={}",
                    self.metric_handles.files_scanned.value()
                )?;
                writeln!(
                    f,
                    "candidate_rows={}",
                    self.metric_handles.candidate_rows.value()
                )?;
                Ok(())
            }
        }
    }
}

#[async_trait::async_trait]
impl ExecutionPlan for VectorIndexScanExec {
    fn name(&self) -> &'static str {
        "VectorIndexScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Plan(
                "VectorIndexScanExec does not accept children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_eq_or_internal_err!(partition, 0, "VectorIndexScanExec invalid partition");
        let schema = self.schema();
        let this = self.clone();
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let stream = stream::once(async move {
            let batch = this.execute_index_scan(context).await?;
            let batch = batch.record_output(&baseline_metrics);
            baseline_metrics.done();
            Ok(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema()))
    }
}

#[derive(Debug, Clone)]
struct VectorIndexScanMetricHandles {
    files_scanned: Count,
    candidate_rows: Count,
}

impl VectorIndexScanMetricHandles {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            files_scanned: MetricBuilder::new(metrics)
                .with_type(MetricType::SUMMARY)
                .counter("files_scanned", partition),
            candidate_rows: MetricBuilder::new(metrics)
                .with_type(MetricType::SUMMARY)
                .counter("candidate_rows", partition),
        }
    }
}

fn parse_index_payload(
    payload: &[u8],
    embedding_column: &EmbeddingColumn,
) -> Result<ParsedIndex, Box<dyn std::error::Error>> {
    if let Ok((index, _)) = read_index_from_payload(payload, embedding_column.clone()) {
        return Ok(ParsedIndex::Ivf(index));
    }
    let (index, _) = read_hnsw_index_from_payload(payload, embedding_column.clone())?;
    Ok(ParsedIndex::Hnsw(index))
}

async fn read_all_embeddings(
    object_path: &str,
    file_size: u64,
    store: &std::sync::Arc<dyn datafusion::object_store::ObjectStore>,
    metadata_size_hint: Option<usize>,
    embedding_column: &str,
) -> Result<Embeddings, DataFusionError> {
    let location = ObjectStorePath::parse(object_path).map_err(|err| {
        DataFusionError::Execution(format!(
            "Invalid object-store path '{}': {}",
            object_path, err
        ))
    })?;

    let mut reader = ParquetObjectReader::new(store.clone(), location).with_file_size(file_size);
    if let Some(hint) = metadata_size_hint {
        reader = reader.with_footer_size_hint(hint);
    }

    let options = ArrowReaderOptions::new().with_page_index(true);
    let builder = parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new_with_options(
        reader, options,
    )
    .await?;

    let schema = builder.schema();
    let embedding_col_idx = schema
        .fields()
        .iter()
        .position(|field| field.name() == embedding_column)
        .ok_or_else(|| {
            DataFusionError::Execution(format!("Embedding column '{}' not found", embedding_column))
        })?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), [embedding_col_idx]);
    use futures::StreamExt;
    let mut stream = builder.with_projection(projection).build()?;

    let mut dim = None;
    let mut all_embeddings = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let embedding_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Embedding column is not a list array".to_string())
            })?;
        if embedding_col.null_count() > 0 {
            return Err(DataFusionError::Execution(
                "Embedding column contains null rows".to_string(),
            ));
        }

        let values = embedding_col.values();
        enum FloatValues<'a> {
            F32(&'a Float32Array),
            F64(&'a Float64Array),
        }
        let float_values = if let Some(array) = values.as_any().downcast_ref::<Float32Array>() {
            FloatValues::F32(array)
        } else if let Some(array) = values.as_any().downcast_ref::<Float64Array>() {
            FloatValues::F64(array)
        } else {
            return Err(DataFusionError::Execution(
                "Embedding values are not float32/float64".to_string(),
            ));
        };

        if match &float_values {
            FloatValues::F32(array) => array.null_count(),
            FloatValues::F64(array) => array.null_count(),
        } > 0
        {
            return Err(DataFusionError::Execution(
                "Embedding values contain nulls".to_string(),
            ));
        }

        for row in 0..embedding_col.len() {
            let row_len = embedding_col.value_length(row) as usize;
            if row_len == 0 {
                return Err(DataFusionError::Execution(
                    "Embedding row has zero length".to_string(),
                ));
            }
            let row_dim = EmbeddingDim::new(row_len).map_err(|err| {
                DataFusionError::Execution(format!(
                    "Embedding vector has invalid row length {}: {}",
                    row_len, err
                ))
            })?;
            if let Some(existing) = dim {
                if existing != row_dim {
                    return Err(DataFusionError::Execution(
                        "Embedding vectors have inconsistent dimensions".to_string(),
                    ));
                }
            } else {
                dim = Some(row_dim);
            }
            let offsets = embedding_col.value_offsets();
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            match &float_values {
                FloatValues::F32(values) => {
                    for i in start..end {
                        all_embeddings.push(values.value(i));
                    }
                }
                FloatValues::F64(values) => {
                    for i in start..end {
                        all_embeddings.push(values.value(i) as f32);
                    }
                }
            }
        }
    }

    let dim = dim.ok_or_else(|| {
        DataFusionError::Execution("Embedding column has no rows".to_string())
    })?;

    Embeddings::new(all_embeddings, dim)
        .map_err(|err| DataFusionError::Execution(err.to_string()))
}
