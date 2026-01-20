//! pq-vector: Parquet with embedded IVF vector index
//!
//! This crate provides IVF (Inverted File) indexing for vector embeddings
//! embedded directly into Parquet files using the user-defined index mechanism.

pub mod ivf;

pub use ivf::{build_index, topk, IvfBuildParams, SearchResult};
