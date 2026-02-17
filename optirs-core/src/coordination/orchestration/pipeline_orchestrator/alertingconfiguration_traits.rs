//! # AlertingConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `AlertingConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{AlertAggregation, AlertingConfiguration};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for AlertingConfiguration<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: Vec::new(),
            channels: Vec::new(),
            aggregation: AlertAggregation::default(),
        }
    }
}
