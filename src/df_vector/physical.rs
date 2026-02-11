//! Physical optimizer rule for VectorTopK.

use std::sync::Arc;

use datafusion::common::ScalarValue;
use datafusion::common::{Result, config::ConfigOptions};
use datafusion::physical_expr::expressions::{CastExpr, Column, Literal, TryCastExpr};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

use super::access::gather_single_parquet_scan;
use super::exec::VectorTopKExec;
use super::expr::scalar_to_f32_list;
use super::options::VectorTopKOptions;

/// Physical optimizer rule that replaces TopK sorts with VectorTopKExec.
#[derive(Debug, Clone)]
pub struct VectorTopKPhysicalOptimizerRule {
    options: VectorTopKOptions,
}

impl VectorTopKPhysicalOptimizerRule {
    /// Create a new optimizer rule.
    pub fn new(options: VectorTopKOptions) -> Self {
        Self { options }
    }

    fn rewrite_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        has_offset: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(merge) = plan.as_any().downcast_ref::<SortPreservingMergeExec>() {
            if !has_offset
                && let Some(sort) = merge.input().as_any().downcast_ref::<SortExec>()
                && merge.expr().len() == sort.expr().len()
                && merge.expr().len() == 1
                && merge.expr()[0] == sort.expr()[0]
                && sort.preserve_partitioning()
                && let Some(topk) = self.build_topk_from_sort_merge(sort, merge.fetch())?
            {
                return Ok(topk);
            }
            let child = self.rewrite_plan(merge.input().clone(), has_offset)?;
            if Arc::ptr_eq(&child, merge.input()) {
                return Ok(plan);
            }
            return plan.with_new_children(vec![child]);
        }

        if let Some(limit) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
            let skip = limit.skip();
            if skip == 0
                && let Some(sort) = limit.input().as_any().downcast_ref::<SortExec>()
                && let Some(topk) = self.build_topk_from_sort(sort, limit.fetch())?
            {
                return Ok(topk);
            }
            let child = self.rewrite_plan(limit.input().clone(), has_offset || skip > 0)?;
            if Arc::ptr_eq(&child, limit.input()) {
                return Ok(plan);
            }
            return plan.with_new_children(vec![child]);
        }

        if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
            if let Some(sort) = limit.input().as_any().downcast_ref::<SortExec>()
                && let Some(topk) = self.build_topk_from_sort(sort, Some(limit.fetch()))?
            {
                return Ok(topk);
            }
            let child = self.rewrite_plan(limit.input().clone(), has_offset)?;
            if Arc::ptr_eq(&child, limit.input()) {
                return Ok(plan);
            }
            return plan.with_new_children(vec![child]);
        }

        if let Some(sort) = plan.as_any().downcast_ref::<SortExec>()
            && !has_offset
            && let Some(topk) = self.build_topk_from_sort(sort, None)?
        {
            return Ok(topk);
        }

        self.rewrite_children(plan, has_offset)
    }

    fn rewrite_children(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        has_offset: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if plan.children().is_empty() {
            return Ok(plan);
        }
        let mut new_children = Vec::with_capacity(plan.children().len());
        let mut changed = false;
        for child in plan.children() {
            let new_child = self.rewrite_plan(child.clone(), has_offset)?;
            changed |= !Arc::ptr_eq(child, &new_child);
            new_children.push(new_child);
        }
        if changed {
            plan.with_new_children(new_children)
        } else {
            Ok(plan)
        }
    }

    fn build_topk_from_sort(
        &self,
        sort: &SortExec,
        limit_fetch: Option<usize>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if sort.preserve_partitioning() {
            return Ok(None);
        }
        self.build_topk_from_sort_inner(sort, limit_fetch)
    }

    fn build_topk_from_sort_merge(
        &self,
        sort: &SortExec,
        limit_fetch: Option<usize>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        self.build_topk_from_sort_inner(sort, limit_fetch)
    }

    fn build_topk_from_sort_inner(
        &self,
        sort: &SortExec,
        limit_fetch: Option<usize>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if sort.expr().len() != 1 {
            return Ok(None);
        }
        let sort_expr = &sort.expr()[0];
        if sort_expr.options.descending {
            return Ok(None);
        }
        let Some((column, query_vector)) = extract_array_distance_physical(&sort_expr.expr) else {
            return Ok(None);
        };
        let Some(scan) = gather_single_parquet_scan(sort.input())? else {
            return Ok(None);
        };
        if scan
            .file_groups
            .iter()
            .map(|group| group.files().len())
            .sum::<usize>()
            == 0
        {
            return Ok(None);
        }
        let k = match limit_fetch {
            Some(limit) => sort.fetch().map(|f| f.min(limit)).unwrap_or(limit),
            None => {
                let Some(fetch) = sort.fetch() else {
                    return Ok(None);
                };
                fetch
            }
        };
        Ok(Some(Arc::new(VectorTopKExec::try_new(
            sort.input().clone(),
            column,
            query_vector,
            k,
            self.options.clone(),
        )?)))
    }
}

impl PhysicalOptimizerRule for VectorTopKPhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.rewrite_plan(plan, false)
    }

    fn name(&self) -> &str {
        "vector_topk_physical"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn extract_array_distance_physical(expr: &Arc<dyn PhysicalExpr>) -> Option<(String, Vec<f32>)> {
    let expr = strip_wrappers(expr);
    let scalar = expr.as_any().downcast_ref::<ScalarFunctionExpr>()?;
    if scalar.name() != "array_distance" || scalar.args().len() != 2 {
        return None;
    }
    let left = strip_wrappers(&scalar.args()[0]);
    let right = strip_wrappers(&scalar.args()[1]);
    let (column, literal) = match (column_from_expr(left), literal_from_expr(right)) {
        (Some(col), Some(lit)) => (col, lit),
        _ => match (column_from_expr(right), literal_from_expr(left)) {
            (Some(col), Some(lit)) => (col, lit),
            _ => return None,
        },
    };
    scalar_to_f32_list(&literal).map(|vector| (column, vector))
}

fn strip_wrappers(expr: &Arc<dyn PhysicalExpr>) -> &Arc<dyn PhysicalExpr> {
    let mut current = expr;
    loop {
        if let Some(cast) = current.as_any().downcast_ref::<CastExpr>() {
            current = cast.expr();
            continue;
        }
        if let Some(cast) = current.as_any().downcast_ref::<TryCastExpr>() {
            current = cast.expr();
            continue;
        }
        return current;
    }
}

fn column_from_expr(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    let expr = strip_wrappers(expr);
    expr.as_any()
        .downcast_ref::<Column>()
        .map(|col| col.name().to_string())
}

fn literal_from_expr(expr: &Arc<dyn PhysicalExpr>) -> Option<ScalarValue> {
    let expr = strip_wrappers(expr);
    expr.as_any()
        .downcast_ref::<Literal>()
        .map(|lit| lit.value().clone())
}
