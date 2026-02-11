//! DataFusion integration for vector top-k using IVF indexes in Parquet.

mod access;
mod exec;
mod expr;
mod index_exec;
mod options;
mod physical;
mod session;

#[cfg(test)]
mod tests;

pub use options::VectorTopKOptions;
pub use physical::VectorTopKPhysicalOptimizerRule;
pub use session::{PqVectorSessionBuilderExt, PqVectorSessionConfigExt};
