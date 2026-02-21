//! Configuration for VectorTopK execution.

/// Options controlling VectorTopK execution.
#[derive(Debug, Clone)]
pub struct VectorTopKOptions {
    /// Number of IVF clusters to probe.
    pub nprobe: usize,
    /// Optional hard cap on total candidates to scan.
    pub max_candidates: Option<usize>,
    /// Selectivity threshold that triggers post-filtering.
    ///
    /// Lower values increase pre-filtering usage. Smaller `selectivity` estimates are
    /// considered more selective, and trigger post-filtering.
    pub post_filter_selectivity_threshold: f64,
}

impl Default for VectorTopKOptions {
    fn default() -> Self {
        Self {
            nprobe: 5,
            max_candidates: None,
            post_filter_selectivity_threshold: 0.2,
        }
    }
}
