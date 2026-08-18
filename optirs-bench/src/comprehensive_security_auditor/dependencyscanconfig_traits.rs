//! # `DependencyScanConfig` - Trait Implementations
//!
//! This module contains trait implementations for `DependencyScanConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};

use super::types::DependencyScanConfig;

impl Default for DependencyScanConfig {
    fn default() -> Self {
        Self {
            scan_direct_deps: true,
            scan_transitive_deps: true,
            max_depth: 10,
            check_outdated: true,
            min_versions: HashMap::new(),
            blocked_dependencies: HashSet::new(),
            allowed_licenses: HashSet::new(),
            blocked_licenses: HashSet::new(),
        }
    }
}
