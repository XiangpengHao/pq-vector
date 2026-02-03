//! Physical execution for VectorTopK.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
    RecordBatch,
};
use arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result, ScalarValue, assert_eq_or_internal_err};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricsSet,
    RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics, collect,
};
use futures::stream;

use crate::ivf::read_index_from_parquet;

use super::access::{
    CandidateCursor, FileEntry, ParquetScanInfo, build_access_plans, gather_single_parquet_scan,
    local_path_from_object_store, read_row_group_row_counts, rewrite_with_access_plans,
};
use super::options::VectorTopKOptions;

/// Execution plan for VectorTopK.
#[derive(Clone)]
pub(crate) struct VectorTopKExec {
    scan_plan: Arc<dyn ExecutionPlan>,
    vector_column: String,
    query: Vec<f32>,
    k: usize,
    options: VectorTopKOptions,
    cache: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
    metric_handles: VectorTopKMetricHandles,
}

impl VectorTopKExec {
    pub(crate) fn new(
        scan_plan: Arc<dyn ExecutionPlan>,
        vector_column: String,
        query: Vec<f32>,
        k: usize,
        options: VectorTopKOptions,
    ) -> Self {
        let metrics = ExecutionPlanMetricsSet::new();
        let metric_handles = VectorTopKMetricHandles::new(&metrics, 0);
        let cache = PlanProperties::new(
            EquivalenceProperties::new(scan_plan.schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            scan_plan,
            vector_column,
            query,
            k,
            options,
            cache,
            metrics,
            metric_handles,
        }
    }

    async fn execute_topk(
        &self,
        context: Arc<TaskContext>,
        metrics: &VectorTopKMetricHandles,
    ) -> Result<RecordBatch> {
        let scan_info = gather_single_parquet_scan(&self.scan_plan)?.ok_or_else(|| {
            DataFusionError::Plan("VectorTopKExec requires a single parquet scan input".to_string())
        })?;
        self.execute_with_index(&scan_info, context, metrics).await
    }

    async fn execute_with_index(
        &self,
        scan: &ParquetScanInfo,
        context: Arc<TaskContext>,
        metrics: &VectorTopKMetricHandles,
    ) -> Result<RecordBatch> {
        let schema = self.scan_plan.schema();
        let vector_idx = schema.index_of(&self.vector_column).map_err(|_| {
            DataFusionError::Plan(format!(
                "Vector column '{}' not found in schema",
                self.vector_column
            ))
        })?;

        let mut file_entries = Vec::new();
        let mut candidate_rows = 0usize;
        for file_group in scan.file_groups.iter() {
            for file in file_group.files().iter() {
                let object_path = file.path().as_ref().to_string();
                let local_path =
                    local_path_from_object_store(&scan.object_store_url, file.path().as_ref())
                        .ok_or_else(|| {
                            DataFusionError::Plan(
                                "VectorTopK only supports local file:// paths".to_string(),
                            )
                        })?;
                let (index, embedding_column) =
                    read_index_from_parquet(&local_path).map_err(|err| {
                        DataFusionError::Plan(format!(
                            "Failed to read IVF index from {}: {}",
                            local_path.display(),
                            err
                        ))
                    })?;
                if embedding_column.as_str() != self.vector_column {
                    return Err(DataFusionError::Plan(format!(
                        "IVF index column mismatch: expected '{}', found '{}'",
                        self.vector_column,
                        embedding_column.as_str()
                    )));
                }
                if index.dim() != self.query.len() {
                    return Err(DataFusionError::Plan(format!(
                        "Query dimension mismatch: expected {}, got {}",
                        index.dim(),
                        self.query.len()
                    )));
                }
                let candidates = index.candidate_rows(&self.query, self.options.nprobe);
                candidate_rows += candidates.len();
                let row_groups = read_row_group_row_counts(&local_path)?;
                file_entries.push(FileEntry {
                    object_path,
                    row_groups,
                    candidates,
                });
            }
        }

        if file_entries.is_empty() {
            return Err(DataFusionError::Plan(
                "VectorTopKExec requires at least one indexed parquet file".to_string(),
            ));
        }

        metrics.index_used.add(1);
        metrics.candidate_rows.add(candidate_rows);

        let mut cursor = CandidateCursor::new(file_entries.len());
        for (idx, entry) in file_entries.iter().enumerate() {
            cursor.add_candidates(idx, entry.candidates.clone());
        }

        let max_candidates = self.options.max_candidates.unwrap_or(usize::MAX);
        let mut heap = BinaryHeap::new();
        let mut scanned = 0usize;
        let batch_size = context.session_config().batch_size();

        while heap.len() < self.k && scanned < max_candidates {
            let batch = cursor.next_batch(batch_size);
            if batch.is_empty() {
                break;
            }

            let mut selections: HashMap<String, Vec<u32>> = HashMap::new();
            for (file_idx, row) in batch {
                selections
                    .entry(file_entries[file_idx].object_path.clone())
                    .or_default()
                    .push(row);
            }

            let access_plans = build_access_plans(&file_entries, &selections)?;
            let plan = rewrite_with_access_plans(self.scan_plan.clone(), &access_plans)?;
            let batches = collect(plan, context.clone()).await?;
            scanned += selections.values().map(|v| v.len()).sum::<usize>();

            for batch in batches {
                update_topk_heap(&mut heap, &batch, vector_idx, &self.query, self.k, metrics)?;
            }
        }

        let mut results: Vec<TopKRow> = heap.into_iter().collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        let rows = results.into_iter().map(|r| r.values).collect::<Vec<_>>();
        let schema = self.scan_plan.schema();
        build_batch_from_rows(schema, rows)
    }
}

impl fmt::Debug for VectorTopKExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorTopKExec")
    }
}

impl DisplayAs for VectorTopKExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "VectorTopKExec: k={}", self.k)
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "vector_topk")?;
                writeln!(f, "k={}", self.k)?;
                writeln!(f, "column={}", self.vector_column)?;
                writeln!(f, "query_dim={}", self.query.len())?;
                writeln!(f, "nprobe={}", self.options.nprobe)?;
                if let Some(max_candidates) = self.options.max_candidates {
                    writeln!(f, "max_candidates={max_candidates}")?;
                }
                writeln!(
                    f,
                    "index_used={}",
                    self.metric_handles.index_used.value() > 0
                )?;
                writeln!(
                    f,
                    "candidate_rows={}",
                    self.metric_handles.candidate_rows.value()
                )?;
                writeln!(
                    f,
                    "embeddings_fetched={}",
                    self.metric_handles.embeddings_fetched.value()
                )?;
                writeln!(
                    f,
                    "batches_fetched={}",
                    self.metric_handles.batches_fetched.value()
                )?;
                Ok(())
            }
        }
    }
}

#[async_trait::async_trait]
impl ExecutionPlan for VectorTopKExec {
    fn name(&self) -> &'static str {
        "VectorTopKExec"
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
                "VectorTopKExec does not accept children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_eq_or_internal_err!(partition, 0, "VectorTopKExec invalid partition");
        let schema = self.schema();
        let this = self.clone();
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let metric_handles = self.metric_handles.clone();
        let stream = stream::once(async move {
            let batch = this.execute_topk(context, &metric_handles).await?;
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
struct VectorTopKMetricHandles {
    index_used: Count,
    candidate_rows: Count,
    embeddings_fetched: Count,
    batches_fetched: Count,
}

impl VectorTopKMetricHandles {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            index_used: MetricBuilder::new(metrics)
                .with_type(MetricType::SUMMARY)
                .counter("index_used", partition),
            candidate_rows: MetricBuilder::new(metrics)
                .with_type(MetricType::SUMMARY)
                .counter("candidate_rows", partition),
            embeddings_fetched: MetricBuilder::new(metrics)
                .with_type(MetricType::SUMMARY)
                .counter("embeddings_fetched", partition),
            batches_fetched: MetricBuilder::new(metrics)
                .with_type(MetricType::DEV)
                .counter("batches_fetched", partition),
        }
    }

    fn record_batch(&self, batch: &RecordBatch) {
        self.embeddings_fetched.add(batch.num_rows());
        self.batches_fetched.add(1);
    }
}

#[derive(Debug)]
struct TopKRow {
    distance: f32,
    values: Vec<ScalarValue>,
}

impl PartialEq for TopKRow {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for TopKRow {}

impl PartialOrd for TopKRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

fn update_topk_heap(
    heap: &mut BinaryHeap<TopKRow>,
    batch: &RecordBatch,
    vector_idx: usize,
    query: &[f32],
    k: usize,
    metrics: &VectorTopKMetricHandles,
) -> Result<()> {
    metrics.record_batch(batch);
    let vector_array = batch.column(vector_idx);
    for row in 0..batch.num_rows() {
        let distance = match compute_distance(vector_array, row, query)? {
            Some(distance) => distance,
            None => continue,
        };
        let values = row_to_scalar_values(batch, row)?;
        let item = TopKRow { distance, values };
        if heap.len() < k {
            heap.push(item);
        } else if let Some(top) = heap.peek()
            && distance < top.distance
        {
            heap.pop();
            heap.push(item);
        }
    }
    Ok(())
}

fn row_to_scalar_values(batch: &RecordBatch, row: usize) -> Result<Vec<ScalarValue>> {
    batch
        .columns()
        .iter()
        .map(|array| ScalarValue::try_from_array(array.as_ref(), row))
        .collect()
}

fn compute_distance(array: &ArrayRef, row: usize, query: &[f32]) -> Result<Option<f32>> {
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        if list.is_null(row) {
            return Ok(None);
        }
        let values = list.value(row);
        return compute_distance_values(&values, query);
    }

    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        if list.is_null(row) {
            return Ok(None);
        }
        let values = list.value(row);
        return compute_distance_values(&values, query);
    }

    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        if list.is_null(row) {
            return Ok(None);
        }
        let values = list.value(row);
        return compute_distance_values(&values, query);
    }

    Err(DataFusionError::Plan(
        "Vector column must be list or fixed-size list".to_string(),
    ))
}

fn compute_distance_values(values: &ArrayRef, query: &[f32]) -> Result<Option<f32>> {
    if let Some(floats) = values.as_any().downcast_ref::<Float32Array>() {
        if floats.len() != query.len() {
            return Ok(None);
        }
        let mut dist = 0.0f32;
        for (&value, &q) in floats.values().iter().zip(query) {
            let diff = value - q;
            dist += diff * diff;
        }
        return Ok(Some(dist));
    }
    if let Some(floats) = values.as_any().downcast_ref::<Float64Array>() {
        if floats.len() != query.len() {
            return Ok(None);
        }
        let mut dist = 0.0f32;
        for (&value, &q) in floats.values().iter().zip(query) {
            let diff = value as f32 - q;
            dist += diff * diff;
        }
        return Ok(Some(dist));
    }
    Err(DataFusionError::Plan(
        "Vector column must be Float32 or Float64 list".to_string(),
    ))
}

fn build_batch_from_rows(schema: SchemaRef, rows: Vec<Vec<ScalarValue>>) -> Result<RecordBatch> {
    if rows.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }
    let num_cols = schema.fields().len();
    let mut columns: Vec<Vec<ScalarValue>> = vec![Vec::with_capacity(rows.len()); num_cols];
    for row in rows {
        for (col_idx, value) in row.into_iter().enumerate() {
            columns[col_idx].push(value);
        }
    }
    let arrays = columns
        .into_iter()
        .map(|col| ScalarValue::iter_to_array(col.into_iter()))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema, arrays).map_err(DataFusionError::from)
}
