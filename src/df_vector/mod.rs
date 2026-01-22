//! DataFusion integration for vector top-k using IVF indexes in Parquet.

mod access;
mod exec;
mod expr;
mod logical;
mod options;
mod planner;

#[cfg(test)]
mod tests;

pub use logical::VectorTopKOptimizerRule;
pub use options::VectorTopKOptions;
pub use planner::VectorTopKQueryPlanner;

use std::sync::Arc;

use datafusion::prelude::SessionContext;

/// Helper to register the optimizer rule on a SessionContext.
pub fn register_vector_topk_rule(ctx: &SessionContext) {
    ctx.add_optimizer_rule(Arc::new(VectorTopKOptimizerRule::new()));
}
