//! pq-vector: Parquet with embedded IVF vector index
//!
//! This crate provides IVF (Inverted File) indexing for vector embeddings
//! embedded directly into Parquet files using the user-defined index mechanism.
//!
//! ## Features
//!
//! - **IVF Index**: Build and embed IVF indexes directly into Parquet files
//!
//! ## Example
//!
//! ```ignore
//! use pq_vector::{build_index, topk, IvfBuildParams};
//! use std::path::Path;
//!
//! // Build index
//! build_index(source, output, "embedding", &IvfBuildParams::default())?;
//!
//! // Query with Rust API
//! let query_vector = vec![1.0f32, 2.0f32];
//! let results = topk(Path::new("file.parquet"), &query_vector, 10, 5).await?;
//! ```

pub mod df_vector;
pub mod ivf;

pub use ivf::{IvfBuildParams, SearchResult, build_index, topk};
