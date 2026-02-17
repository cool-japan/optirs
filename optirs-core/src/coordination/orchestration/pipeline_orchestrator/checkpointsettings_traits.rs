//! # CheckpointSettings - Trait Implementations
//!
//! This module contains trait implementations for `CheckpointSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::{
    CheckpointFrequency, CheckpointRetentionPolicy, CheckpointSettings, CheckpointStorage,
    CompressionSettings,
};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for CheckpointSettings<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency: CheckpointFrequency::TimeBased(Duration::from_secs(300)),
            storage: CheckpointStorage::default(),
            compression: CompressionSettings::default(),
            retention: CheckpointRetentionPolicy::default(),
        }
    }
}
