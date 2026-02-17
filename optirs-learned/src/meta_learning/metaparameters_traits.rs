//! # MetaParameters - Trait Implementations
//!
//! This module contains trait implementations for `MetaParameters`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::types::MetaParameters;

impl<T: Float + Debug + Send + Sync + 'static> Default for MetaParameters<T> {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}
