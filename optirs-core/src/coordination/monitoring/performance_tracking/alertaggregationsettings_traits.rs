//! # AlertAggregationSettings - Trait Implementations
//!
//! This module contains trait implementations for `AlertAggregationSettings`.
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

use super::types::AlertAggregationSettings;

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for AlertAggregationSettings<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(60),
            custom_params: HashMap::new(),
        }
    }
}
