//! # `OptimizerIdentifier` - Trait Implementations
//!
//! This module contains trait implementations for `OptimizerIdentifier`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OptimizerIdentifier;

impl std::fmt::Display for OptimizerIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref version) = self.version {
            write!(f, "{}-{}-v{}", self.framework, self.name, version)
        } else {
            write!(f, "{}-{}", self.framework, self.name)
        }
    }
}
