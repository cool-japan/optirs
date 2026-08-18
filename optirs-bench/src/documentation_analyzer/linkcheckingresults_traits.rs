//! # `LinkCheckingResults` - Trait Implementations
//!
//! This module contains trait implementations for `LinkCheckingResults`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::LinkCheckingResults;

impl Default for LinkCheckingResults {
    fn default() -> Self {
        Self {
            total_links: 0,
            valid_links: 0,
            broken_links: Vec::new(),
            external_link_status: HashMap::new(),
            internal_link_consistency: 1.0,
        }
    }
}
