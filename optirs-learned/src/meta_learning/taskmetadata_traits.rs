//! # TaskMetadata - Trait Implementations
//!
//! This module contains trait implementations for `TaskMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::types::TaskMetadata;

impl Default for TaskMetadata {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: "default task".to_string(),
            properties: HashMap::new(),
            created_at: Instant::now(),
            source: "default".to_string(),
        }
    }
}
