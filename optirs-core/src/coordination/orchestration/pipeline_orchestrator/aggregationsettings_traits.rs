//! # AggregationSettings - Trait Implementations
//!
//! This module contains trait implementations for `AggregationSettings`.
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

use super::types::{AggregationFunction, AggregationSettings};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for AggregationSettings<T> {
    fn default() -> Self {
        Self {
            functions: vec![
                AggregationFunction::Mean,
                AggregationFunction::Max,
                AggregationFunction::Min,
            ],
            window: Duration::from_secs(60),
            interval: Duration::from_secs(10),
            retention: Duration::from_secs(86400),
            custom: HashMap::new(),
        }
    }
}
