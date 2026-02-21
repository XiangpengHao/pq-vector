//! Filter extraction and strategy selection for vector top-k optimization.

use std::sync::Arc;

use datafusion::common::{Result};
use datafusion::physical_expr::expressions::{
    BinaryExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal,
};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::expressions::TryCastExpr;
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::filter::FilterExec;

use super::options::VectorTopKOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FilterStrategy {
    PreFilter,
    PostFilter,
}

const DEFAULT_EQ_SELECTIVITY: f64 = 0.1;
const DEFAULT_IN_VALUE_SELECTIVITY: f64 = 0.03;
const DEFAULT_RANGE_SELECTIVITY: f64 = 0.45;
const DEFAULT_IS_NULL_SELECTIVITY: f64 = 0.05;
const DEFAULT_NOT_SELECTIVITY: f64 = 0.9;

const MIN_SELECTIVITY: f64 = 1e-6;
const MAX_SELECTIVITY: f64 = 1.0;

pub(crate) fn select_filter_strategy(
    filters: &[Arc<dyn PhysicalExpr>],
    options: &VectorTopKOptions,
) -> FilterStrategy {
    if filters.is_empty() {
        return FilterStrategy::PreFilter;
    }
    let selectivity = estimate_filters_selectivity(filters);
    if selectivity <= options.post_filter_selectivity_threshold {
        FilterStrategy::PostFilter
    } else {
        FilterStrategy::PreFilter
    }
}

pub(crate) fn split_filter_chain(
    plan: &Arc<dyn ExecutionPlan>,
) -> (Vec<Arc<dyn PhysicalExpr>>, Arc<dyn ExecutionPlan>) {
    let mut filters = Vec::new();
    let mut current = plan.clone();
    while let Some(filter) = current.as_any().downcast_ref::<FilterExec>() {
        filters.push(filter.predicate().clone());
        current = filter.input().clone();
    }
    (filters, current)
}

pub(crate) fn rewrap_filters(
    plan: Arc<dyn ExecutionPlan>,
    predicates: &[Arc<dyn PhysicalExpr>],
) -> Result<Arc<dyn ExecutionPlan>> {
    predicates.iter().rev().try_fold(plan, |current, predicate| {
        FilterExec::try_new(predicate.clone(), current).map(|f| Arc::new(f) as Arc<dyn ExecutionPlan>)
    })
}

fn estimate_filters_selectivity(filters: &[Arc<dyn PhysicalExpr>]) -> f64 {
    let mut estimate = 1.0;
    for filter in filters {
        estimate *= estimate_filter_selectivity(filter);
    }
    clamp_selectivity(estimate)
}

fn estimate_filter_selectivity(expr: &Arc<dyn PhysicalExpr>) -> f64 {
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        return match binary.op() {
            Operator::And => {
                let left = estimate_filter_selectivity(&binary.left());
                let right = estimate_filter_selectivity(&binary.right());
                clamp_selectivity(left * right)
            }
            Operator::Or => {
                let left = estimate_filter_selectivity(&binary.left());
                let right = estimate_filter_selectivity(&binary.right());
                clamp_selectivity(1.0 - (1.0 - left) * (1.0 - right))
            }
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => {
                estimate_comparison_selectivity(binary)
            }
            _ => DEFAULT_NOT_SELECTIVITY,
        };
    }

    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        let list_len = in_list.list().len();
        if list_len == 0 {
            return if in_list.negated() { 1.0 } else { 0.0 };
        }
        let selectivity = DEFAULT_IN_VALUE_SELECTIVITY * (list_len as f64);
        return if in_list.negated() {
            clamp_selectivity(1.0 - selectivity)
        } else {
            clamp_selectivity(selectivity)
        };
    }

    if expr.as_any().downcast_ref::<IsNullExpr>().is_some() {
        return DEFAULT_IS_NULL_SELECTIVITY;
    }
    if expr.as_any().downcast_ref::<IsNotNullExpr>().is_some() {
        return 1.0 - DEFAULT_IS_NULL_SELECTIVITY;
    }

    DEFAULT_NOT_SELECTIVITY
}

fn estimate_comparison_selectivity(expr: &BinaryExpr) -> f64 {
    if !is_column_scalar_filter(expr.left(), expr.right()) {
        return DEFAULT_NOT_SELECTIVITY;
    }
    match expr.op() {
        Operator::Eq => DEFAULT_EQ_SELECTIVITY,
        Operator::NotEq => 1.0 - DEFAULT_EQ_SELECTIVITY,
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => DEFAULT_RANGE_SELECTIVITY,
        _ => DEFAULT_NOT_SELECTIVITY,
    }
}

fn is_column_scalar_filter(left: &Arc<dyn PhysicalExpr>, right: &Arc<dyn PhysicalExpr>) -> bool {
    (as_column_expr(left).is_some() && as_scalar_expr(right))
        || (as_column_expr(right).is_some() && as_scalar_expr(left))
}

fn as_scalar_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let expr = strip_wrappers(expr);
    expr.as_any().downcast_ref::<Literal>().is_some()
}

fn as_column_expr(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    let expr = strip_wrappers(expr);
    expr.as_any()
        .downcast_ref::<Column>()
        .map(|col| col.name().to_string())
}

fn as_cast_expr(expr: &Arc<dyn PhysicalExpr>) -> Option<&Arc<dyn PhysicalExpr>> {
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        return Some(cast.expr());
    }
    expr.as_any()
        .downcast_ref::<TryCastExpr>()
        .map(|cast: &TryCastExpr| cast.expr())
}

fn strip_wrappers(expr: &Arc<dyn PhysicalExpr>) -> &Arc<dyn PhysicalExpr> {
    let mut current = expr;
    loop {
        if let Some(inner) = as_cast_expr(current) {
            current = inner;
            continue;
        }
        return current;
    }
}

fn clamp_selectivity(value: f64) -> f64 {
    value.clamp(MIN_SELECTIVITY, MAX_SELECTIVITY)
}
