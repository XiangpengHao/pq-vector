//! Extension planner and query planner wiring.

use std::fmt;
use std::sync::Arc;

use datafusion::common::{DataFusionError, Result};
use datafusion::execution::context::SessionState;
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::UserDefinedLogicalNode;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use super::exec::VectorTopKExec;
use super::expr::extract_array_distance;
use super::logical::VectorTopKPlanNode;
use super::options::VectorTopKOptions;

/// Extension planner for VectorTopK nodes.
#[derive(Debug)]
pub(crate) struct VectorTopKPlanner {
    options: VectorTopKOptions,
}

impl VectorTopKPlanner {
    /// Create a new planner with the provided options.
    pub fn new(options: VectorTopKOptions) -> Self {
        Self { options }
    }
}

#[async_trait::async_trait]
impl ExtensionPlanner for VectorTopKPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(topk) = node.as_any().downcast_ref::<VectorTopKPlanNode>() {
            let Some((column, query_vector)) = extract_array_distance(topk.distance_expr()) else {
                return Err(DataFusionError::Plan(
                    "VectorTopK requires array_distance(column, const)".to_string(),
                ));
            };
            if physical_inputs.len() != 1 {
                return Err(DataFusionError::Plan(
                    "VectorTopK expects exactly one input".to_string(),
                ));
            }
            return Ok(Some(Arc::new(VectorTopKExec::new(
                physical_inputs[0].clone(),
                column.name.clone(),
                query_vector,
                topk.k(),
                self.options.clone(),
            ))));
        }
        Ok(None)
    }
}

/// Build a query planner with VectorTopK extension support.
pub struct VectorTopKQueryPlanner {
    planner: DefaultPhysicalPlanner,
}

impl VectorTopKQueryPlanner {
    /// Create a new query planner with VectorTopK support.
    pub fn new(options: VectorTopKOptions) -> Self {
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            VectorTopKPlanner::new(options),
        )]);
        Self { planner }
    }
}

impl fmt::Debug for VectorTopKQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorTopKQueryPlanner")
    }
}

#[async_trait::async_trait]
impl datafusion::execution::context::QueryPlanner for VectorTopKQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}
