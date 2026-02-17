//! # MetricsConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `MetricsConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::{AggregationSettings, ExportSettings, MetricType, MetricsConfiguration};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for MetricsConfiguration<T> {
    fn default() -> Self {
        Self {
            metrics: vec![MetricType::ResourceUtilization, MetricType::Performance],
            collection_frequency: Duration::from_secs(5),
            aggregation: AggregationSettings::default(),
            export: ExportSettings::default(),
        }
    }
}
