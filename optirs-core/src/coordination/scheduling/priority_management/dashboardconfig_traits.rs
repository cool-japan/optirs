//! # `DashboardConfig` - Trait Implementations
//!
//! This module contains trait implementations for `DashboardConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::DashboardConfig;

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_frequency: Duration::from_secs(30),
            display_options: HashMap::new(),
            layout: Vec::new(),
        }
    }
}
