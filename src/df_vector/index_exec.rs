//! Physical execution for IVF candidate row-id scan.

use std::any::Any;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result, assert_eq_or_internal_err};
use datafusion::execution::TaskContext;
use datafusion::object_store::path::Path as ObjectStorePath;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use futures::stream;
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};

use crate::ivf::{read_index_from_payload, read_index_metadata_from_file_metadata};

use super::access::ScanFile;
use super::options::VectorTopKOptions;

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
        Self {
            files,
            vector_column,
            query,
            options,
            schema,
            cache,
            metrics: ExecutionPlanMetricsSet::new(),
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
            let (index, _) =
                read_index_from_payload(payload.as_ref(), embedding_column).map_err(|err| {
                    DataFusionError::Execution(format!(
                        "Failed to decode pq-vector payload from '{}': {}",
                        object_path, err
                    ))
                })?;

            if index.dim() != self.query.len() {
                return Err(DataFusionError::Plan(format!(
                    "Query dimension mismatch: expected {}, got {}",
                    index.dim(),
                    self.query.len()
                )));
            }

            files.push(IndexFileCandidates {
                object_path,
                candidates: index.candidate_rows(&self.query, self.options.nprobe),
            });
        }

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
                writeln!(f, "column={}", self.vector_column)?;
                writeln!(f, "query_dim={}", self.query.len())?;
                writeln!(f, "nprobe={}", self.options.nprobe)?;
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
