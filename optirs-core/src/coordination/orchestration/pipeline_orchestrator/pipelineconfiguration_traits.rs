//! # PipelineConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `PipelineConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::types::{
    CheckpointSettings, ExecutionMode, MonitoringConfiguration, ParallelismSettings,
    PipelineConfiguration, ResourceLimits, SecuritySettings, TimeoutSettings,
};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for PipelineConfiguration<T> {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Sequential,
            parallelism: ParallelismSettings::default(),
            resource_limits: ResourceLimits::default(),
            timeouts: TimeoutSettings::default(),
            retry_policies: HashMap::new(),
            checkpoint_settings: CheckpointSettings::default(),
            monitoring: MonitoringConfiguration::default(),
            security: SecuritySettings::default(),
        }
    }
}
