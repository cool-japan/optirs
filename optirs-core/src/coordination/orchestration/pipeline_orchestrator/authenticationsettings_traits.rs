//! # AuthenticationSettings - Trait Implementations
//!
//! This module contains trait implementations for `AuthenticationSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{AuthenticationSettings, AuthenticationType};

impl Default for AuthenticationSettings {
    fn default() -> Self {
        Self {
            auth_type: AuthenticationType::None,
            provider: String::from("none"),
            config: HashMap::new(),
        }
    }
}
