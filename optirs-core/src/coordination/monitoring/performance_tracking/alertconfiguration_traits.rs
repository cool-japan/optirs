//! # AlertConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `AlertConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{AlertAggregationSettings, AlertConfiguration, NotificationSettings};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for AlertConfiguration<T> {
    fn default() -> Self {
        Self {
            default_rules: Vec::new(),
            notification_settings: NotificationSettings::default(),
            aggregation_settings: AlertAggregationSettings::default(),
        }
    }
}
