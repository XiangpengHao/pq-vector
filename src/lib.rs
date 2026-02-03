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
//! use pq_vector::{
//!     IndexBuilder,
//!     TopkBuilder,
//! };
//! use std::path::Path;
//!
//! // Build index
//! IndexBuilder::new(source, "embedding")
//!     .build_inplace()?;
//! // Optional: write to a new file
//! IndexBuilder::new(source, "embedding")
//!     .build_new(output)?;
//!
//! // Query with Rust API
//! let query_vector = vec![1.0f32, 2.0f32];
//! let results = TopkBuilder::new(Path::new("file.parquet"), &query_vector)
//!     .k(10)?
//!     .nprobe(5)?
//!     .search()
//!     .await?;
//! ```

pub mod df_vector;
pub mod ivf;

pub use ivf::{
    ClusterCount, IndexBuilder, SearchResult, TopkBuilder,
};
