//! Configuration for VectorTopK execution.

/// Options controlling VectorTopK execution.
#[derive(Debug, Clone)]
pub struct VectorTopKOptions {
    /// Number of IVF clusters to probe.
    pub nprobe: usize,
    /// Max number of candidate rows to read per batch.
    pub batch_size: usize,
    /// Optional hard cap on total candidates to scan.
    pub max_candidates: Option<usize>,
}

impl Default for VectorTopKOptions {
    fn default() -> Self {
        Self {
            nprobe: 5,
            batch_size: 1024,
            max_candidates: None,
        }
    }
}
