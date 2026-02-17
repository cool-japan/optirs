//! # MetaTask - Trait Implementations
//!
//! This module contains trait implementations for `MetaTask`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{MetaTask, TaskDataset, TaskMetadata, TaskType};

impl<T: Float + Debug + Send + Sync + 'static> Default for MetaTask<T> {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            support_set: TaskDataset::default(),
            query_set: TaskDataset::default(),
            metadata: TaskMetadata::default(),
            difficulty: scirs2_core::numeric::NumCast::from(1.0).unwrap_or_else(|| T::zero()),
            domain: "default".to_string(),
            task_type: TaskType::Classification,
        }
    }
}
