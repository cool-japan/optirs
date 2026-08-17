//! # `StatisticalProperties` - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalProperties`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StatisticalProperties;
use super::types_7::HjorthParameters;

impl Default for StatisticalProperties {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            entropy: 0.0,
            autocorrelation: Vec::new(),
            partial_autocorrelation: Vec::new(),
            hjorth_parameters: HjorthParameters {
                activity: 0.0,
                mobility: 0.0,
                complexity: 0.0,
            },
        }
    }
}
