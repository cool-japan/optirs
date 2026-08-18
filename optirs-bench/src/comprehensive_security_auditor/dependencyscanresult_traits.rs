//! # `DependencyScanResult` - Trait Implementations
//!
//! This module contains trait implementations for `DependencyScanResult`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DependencyScanResult, DependencyTree};

impl Default for DependencyScanResult {
    fn default() -> Self {
        Self {
            total_dependencies: 0,
            vulnerable_dependencies: Vec::new(),
            outdated_dependencies: Vec::new(),
            license_violations: Vec::new(),
            supply_chain_risks: Vec::new(),
            dependency_tree: DependencyTree::default(),
            risk_score: 0.0,
        }
    }
}
