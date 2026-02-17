//! # CheckpointStorage - Trait Implementations
//!
//! This module contains trait implementations for `CheckpointStorage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{CheckpointStorage, StorageType};

impl Default for CheckpointStorage {
    fn default() -> Self {
        Self {
            storage_type: StorageType::Local,
            location: String::from("/tmp/checkpoints"),
            credentials: None,
            options: HashMap::new(),
        }
    }
}
