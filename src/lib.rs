//! pq-vector: Parquet with embedded IVF vector index
//!
//! This crate provides IVF (Inverted File) indexing for vector embeddings
//! embedded directly into Parquet files using the user-defined index mechanism.
//!
//! ## Features
//!
//! - **IVF Index**: Build and embed IVF indexes directly into Parquet files
//! - **DataFusion Integration**: Use `topk()` table function in SQL queries
//!
//! ## Example
//!
//! ```ignore
//! use datafusion::prelude::*;
//! use pq_vector::{TopkTableFunction, build_index, IvfBuildParams};
//! use std::sync::Arc;
//!
//! // Build index
//! build_index(source, output, "embedding", &IvfBuildParams::default())?;
//!
//! // Query with DataFusion
//! let ctx = SessionContext::new();
//! ctx.register_udtf("topk", Arc::new(TopkTableFunction));
//!
//! let df = ctx.sql("SELECT * FROM topk('file.parquet', ARRAY[1.0, 2.0], 10, 5)").await?;
//! ```

pub mod ivf;
pub mod udtf;

pub use ivf::{build_index, topk, IvfBuildParams, SearchResult};
pub use udtf::TopkTableFunction;
