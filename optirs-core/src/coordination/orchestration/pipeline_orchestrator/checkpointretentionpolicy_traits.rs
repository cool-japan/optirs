//! # CheckpointRetentionPolicy - Trait Implementations
//!
//! This module contains trait implementations for `CheckpointRetentionPolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::{CheckpointRetentionPolicy, CleanupStrategy};

impl<T: Float + Debug + Send + Sync + 'static> Default for CheckpointRetentionPolicy<T> {
    fn default() -> Self {
        Self {
            max_checkpoints: 10,
            retention_duration: Some(Duration::from_secs(86400 * 7)),
            cleanup_strategy: CleanupStrategy::FIFO,
            retention_rules: Vec::new(),
        }
    }
}
