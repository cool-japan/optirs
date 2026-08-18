//! # `AnomalyConfig` - Trait Implementations
//!
//! This module contains trait implementations for `AnomalyConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(dead_code)]
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types_3::{AnomalyConfig, OutlierMethod};

impl<T: Float + Debug + Send + Sync + 'static> Default for AnomalyConfig<T> {
    fn default() -> Self {
        Self {
            statistical_threshold: T::from(2.5).unwrap_or_else(|| T::zero()),
            trend_sensitivity: T::from(0.1).unwrap_or_else(|| T::zero()),
            pattern_window: 50,
            baseline_window: 100,
            min_data_points: 10,
            confidence_threshold: T::from(0.8).unwrap_or_else(|| T::zero()),
            enable_adaptive_thresholds: true,
            seasonal_analysis: false,
            outlier_methods: vec![
                OutlierMethod::ZScore,
                OutlierMethod::IQR,
                OutlierMethod::IsolationForest,
                OutlierMethod::LocalOutlierFactor,
            ],
            alert_cooldown: Duration::from_secs(60),
        }
    }
}
