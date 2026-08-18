//! # `DocumentationDebt` - Trait Implementations
//!
//! This module contains trait implementations for `DocumentationDebt`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::DocumentationDebt;

impl Default for DocumentationDebt {
    fn default() -> Self {
        Self {
            total_debt_score: 0.0,
            debt_by_category: HashMap::new(),
            high_priority_items: Vec::new(),
            estimated_effort_hours: 0.0,
        }
    }
}
