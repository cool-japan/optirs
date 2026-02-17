//! # MonitoringConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringConfiguration`.
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
    AlertingConfiguration, LoggingConfiguration, MetricsConfiguration, MonitoringConfiguration,
};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for MonitoringConfiguration<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(10),
            metrics: MetricsConfiguration::default(),
            alerting: AlertingConfiguration::default(),
            logging: LoggingConfiguration::default(),
        }
    }
}
