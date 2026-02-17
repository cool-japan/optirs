//! # StorageConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `StorageConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::{
    CompressionSettings, IndexConfiguration, PartitioningStrategy, StorageConfiguration,
};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for StorageConfiguration<T> {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(86400 * 30),
            compression: CompressionSettings::default(),
            partitioning: PartitioningStrategy::default(),
            indexing: IndexConfiguration::default(),
            custom_params: HashMap::new(),
        }
    }
}
