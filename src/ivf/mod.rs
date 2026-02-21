//! IVF (Inverted File) index embedded in Parquet files.
//!
//! The IVF index is stored directly in the parquet file body after the data pages,
//! with the index offset recorded in the footer metadata. This allows standard
//! parquet readers to ignore the index while specialized readers can use it.

mod index;
mod parquet;
mod search;

use std::num::NonZeroU32;

pub use index::ClusterCount;
pub use parquet::{IndexBuilder, has_pq_vector_index};
pub use search::{
    MultiFileIndex, MultiFileIndexBuilder, MultiSearchResult, SearchResult, TopkBuilder,
};

/// Non-empty embedding column name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingColumn(String);

impl EmbeddingColumn {
    pub fn new(name: impl Into<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err("Embedding column name cannot be empty".into());
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for EmbeddingColumn {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for EmbeddingColumn {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Non-zero embedding dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmbeddingDim(NonZeroU32);

impl EmbeddingDim {
    pub(crate) fn new(dim: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let dim_u32: u32 = dim.try_into()?;
        let dim = NonZeroU32::new(dim_u32)
            .ok_or_else(|| "Embedding dimension must be > 0".to_string())?;
        Ok(Self(dim))
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0.get() as usize
    }

    pub(crate) fn as_u32(self) -> u32 {
        self.0.get()
    }
}

/// Validated embedding matrix (row-major).
#[derive(Debug, Clone)]
pub(crate) struct Embeddings {
    data: Vec<f32>,
    dim: EmbeddingDim,
}

impl Embeddings {
    pub(crate) fn new(
        data: Vec<f32>,
        dim: EmbeddingDim,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dim_usize = dim.as_usize();
        if !data.len().is_multiple_of(dim_usize) {
            return Err("Embedding data length must be a multiple of dimension".into());
        }
        Ok(Self { data, dim })
    }

    pub(crate) fn data(&self) -> &[f32] {
        &self.data
    }

    pub(crate) fn dim(&self) -> EmbeddingDim {
        self.dim
    }

    pub(crate) fn row_count(&self) -> usize {
        self.data.len() / self.dim.as_usize()
    }
}

pub(crate) use index::IvfIndex;
pub(crate) use parquet::{
    read_index_from_parquet, read_index_from_payload, read_index_metadata_from_file_metadata,
};
