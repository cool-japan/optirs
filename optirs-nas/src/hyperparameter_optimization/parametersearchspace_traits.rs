//! # ParameterSearchSpace - Trait Implementations
//!
//! This module contains trait implementations for `ParameterSearchSpace`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque, BTreeMap};

use super::types::ParameterSearchSpace;

impl Default for ParameterSearchSpace {
    fn default() -> Self {
        Self {
            continuous_params: HashMap::new(),
            discrete_params: HashMap::new(),
            categorical_params: HashMap::new(),
            conditional_params: HashMap::new(),
            dependencies: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

