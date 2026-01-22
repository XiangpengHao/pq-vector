//! DataFusion integration for vector top-k using IVF indexes in Parquet.

mod access;
mod exec;
mod expr;
mod options;
mod physical;

#[cfg(test)]
mod tests;

pub use options::VectorTopKOptions;
pub use physical::VectorTopKPhysicalOptimizerRule;
