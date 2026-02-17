//! # AuthorizationSettings - Trait Implementations
//!
//! This module contains trait implementations for `AuthorizationSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{AuthorizationModel, AuthorizationSettings};

impl Default for AuthorizationSettings {
    fn default() -> Self {
        Self {
            model: AuthorizationModel::None,
            rules: Vec::new(),
            roles: HashMap::new(),
        }
    }
}
