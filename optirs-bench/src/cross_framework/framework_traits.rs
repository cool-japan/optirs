//! # `Framework` - Trait Implementations
//!
//! This module contains trait implementations for `Framework`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Framework;

impl std::fmt::Display for Framework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Framework::SciRS2 => write!(f, "SciRS2"),
            Framework::PyTorch => write!(f, "PyTorch"),
            Framework::TensorFlow => write!(f, "TensorFlow"),
        }
    }
}
