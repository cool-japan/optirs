//! # `FrequencyCharacteristics` - Trait Implementations
//!
//! This module contains trait implementations for `FrequencyCharacteristics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_7::FrequencyCharacteristics;

impl Default for FrequencyCharacteristics {
    fn default() -> Self {
        Self {
            dominant_frequencies: Vec::new(),
            power_spectrum: Vec::new(),
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            spectral_rolloff: 0.0,
            spectral_flux: 0.0,
            zero_crossing_rate: 0.0,
        }
    }
}
