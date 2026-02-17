//! # NotificationSettings - Trait Implementations
//!
//! This module contains trait implementations for `NotificationSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::NotificationSettings;

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            default_channels: vec!["console".to_string()],
            channel_configs: HashMap::new(),
            templates: HashMap::new(),
        }
    }
}
