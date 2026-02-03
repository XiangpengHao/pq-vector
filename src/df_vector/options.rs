//! Configuration for VectorTopK execution.

/// Options controlling VectorTopK execution.
#[derive(Debug, Clone)]
pub struct VectorTopKOptions {
    /// Number of IVF clusters to probe.
    pub nprobe: usize,
    /// Optional hard cap on total candidates to scan.
    pub max_candidates: Option<usize>,
}

impl Default for VectorTopKOptions {
    fn default() -> Self {
        Self {
            nprobe: 5,
            max_candidates: None,
        }
    }
}
