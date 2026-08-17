//! # `AccessibilityAnalysis` - Trait Implementations
//!
//! This module contains trait implementations for `AccessibilityAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AccessibilityAnalysis;

impl Default for AccessibilityAnalysis {
    fn default() -> Self {
        Self {
            alttext_coverage: 1.0,
            color_contrast_compliance: 1.0,
            screen_reader_compatibility: 1.0,
            keyboard_navigation_support: 1.0,
            overall_accessibility_score: 1.0,
        }
    }
}
