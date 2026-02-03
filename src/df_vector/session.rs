//! Session configuration helpers for pq-vector with DataFusion.

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;

use super::{VectorTopKOptions, VectorTopKPhysicalOptimizerRule};

/// Extensions for configuring `SessionConfig` to work with pq-vector.
pub trait PqVectorSessionConfigExt {
    /// Enable parquet metadata reads required by pq-vector.
    fn with_pq_vector(self) -> Self;
}

impl PqVectorSessionConfigExt for SessionConfig {
    fn with_pq_vector(mut self) -> Self {
        self.options_mut().execution.parquet.skip_metadata = false;
        self
    }
}

/// Extensions for configuring `SessionStateBuilder` to work with pq-vector.
pub trait PqVectorSessionBuilderExt {
    /// Configure pq-vector defaults and register the physical optimizer rule.
    fn with_pq_vector(self, options: VectorTopKOptions) -> Self;
}

impl PqVectorSessionBuilderExt for SessionStateBuilder {
    fn with_pq_vector(mut self, options: VectorTopKOptions) -> Self {
        let config = self.config().get_or_insert_with(SessionConfig::new);
        config.options_mut().execution.parquet.skip_metadata = false;
        self.with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
    }
}
