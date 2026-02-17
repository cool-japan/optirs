//! # DashboardConfig - Trait Implementations
//!
//! This module contains trait implementations for `DashboardConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{DashboardConfig, DashboardLayout};

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            refresh_interval: 30,
            history_retention_hours: 24,
            alert_thresholds: HashMap::new(),
            layout: DashboardLayout {
                columns: 3,
                widgets: Vec::new(),
            },
        }
    }
}
