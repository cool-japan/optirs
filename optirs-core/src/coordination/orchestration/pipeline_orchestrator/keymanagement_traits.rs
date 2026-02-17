//! # KeyManagement - Trait Implementations
//!
//! This module contains trait implementations for `KeyManagement`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{KeyManagement, KeyRotation};

impl Default for KeyManagement {
    fn default() -> Self {
        Self {
            provider: String::from("local"),
            config: HashMap::new(),
            rotation: KeyRotation::default(),
        }
    }
}
