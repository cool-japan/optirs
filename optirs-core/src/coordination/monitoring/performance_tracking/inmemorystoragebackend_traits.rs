//! # InMemoryStorageBackend - Trait Implementations
//!
//! This module contains trait implementations for `InMemoryStorageBackend`.
//!
//! ## Implemented Traits
//!
//! - `StorageBackend`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::functions::StorageBackend;
use super::types::{InMemoryStorageBackend, MetricQuery, PerformanceMetrics, StorageBackendStats};

impl<T: Float + Debug + Send + Sync + 'static> StorageBackend<T> for InMemoryStorageBackend<T> {
    fn store(&mut self, metrics: &PerformanceMetrics<T>) -> Result<()> {
        self.storage
            .push((metrics.timestamp, "metrics".to_string()));
        Ok(())
    }
    fn retrieve(&self, _query: &MetricQuery<T>) -> Result<Vec<PerformanceMetrics<T>>> {
        Ok(Vec::new())
    }
    fn delete(&mut self, _query: &MetricQuery<T>) -> Result<usize> {
        Ok(0)
    }
    fn get_statistics(&self) -> Result<StorageBackendStats> {
        Ok(StorageBackendStats {
            total_metrics: self.storage.len(),
            storage_size_bytes: self.storage.len() * 100,
            average_query_time: Duration::from_millis(1),
            utilization_percentage: 0.5,
        })
    }
}
