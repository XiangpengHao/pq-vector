//! Logical optimizer rule and plan node for VectorTopK.

use std::fmt;
use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use datafusion::logical_expr::{FetchType, SkipType};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use super::expr::{extract_array_distance, normalize_sort_expr};

/// Optimizer rule that replaces Limit->Sort(array_distance) with VectorTopK.
#[derive(Debug, Default)]
pub struct VectorTopKOptimizerRule {}

impl VectorTopKOptimizerRule {
    /// Create a new optimizer rule.
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for VectorTopKOptimizerRule {
    fn name(&self) -> &str {
        "vector_topk"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up_with_subqueries(|plan| match plan {
            LogicalPlan::Limit(limit) => {
                let SkipType::Literal(skip) = limit.get_skip_type()? else {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                };
                let FetchType::Literal(fetch) = limit.get_fetch_type()? else {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                };
                if skip != 0 {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                }
                let Some(fetch) = fetch else {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                };

                let LogicalPlan::Sort(sort) = limit.input.as_ref() else {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                };
                if sort.expr.len() != 1 {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                }
                if !sort.expr[0].asc {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                }

                let distance_expr = normalize_sort_expr(&sort.expr[0]);
                if extract_array_distance(&distance_expr).is_none() {
                    return Ok(Transformed::no(LogicalPlan::Limit(limit)));
                }

                let k = sort.fetch.map(|f| f.min(fetch)).unwrap_or(fetch);
                let node = VectorTopKPlanNode::try_new(
                    (*sort.input).clone(),
                    distance_expr,
                    k as usize,
                )?;
                Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                    node: Arc::new(node),
                })))
            }
            other => Ok(Transformed::no(other)),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct VectorTopKPlanNode {
    input: LogicalPlan,
    distance_expr: Expr,
    k: usize,
}

impl VectorTopKPlanNode {
    pub(crate) fn try_new(input: LogicalPlan, distance_expr: Expr, k: usize) -> Result<Self> {
        extract_array_distance(&distance_expr).ok_or_else(|| {
            DataFusionError::Plan(
                "VectorTopK requires array_distance(column, const)".to_string(),
            )
        })?;
        Ok(Self {
            input,
            distance_expr,
            k,
        })
    }

    pub(crate) fn distance_expr(&self) -> &Expr {
        &self.distance_expr
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }
}

impl UserDefinedLogicalNodeCore for VectorTopKPlanNode {
    fn name(&self) -> &str {
        "VectorTopK"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &datafusion::common::DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.distance_expr.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VectorTopK: k={}, expr={}", self.k, self.distance_expr)
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 || inputs.len() != 1 {
            return Err(DataFusionError::Plan(
                "VectorTopK expects exactly one expr and one input".to_string(),
            ));
        }
        VectorTopKPlanNode::try_new(inputs[0].clone(), exprs[0].clone(), self.k)
    }
}
