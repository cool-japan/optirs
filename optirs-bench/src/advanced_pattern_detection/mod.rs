// Advanced-advanced pattern detection for memory leak analysis
//
// This module implements cutting-edge pattern detection algorithms using machine learning
// techniques, statistical analysis, and advanced signal processing for memory usage patterns.
//
// Split into submodules by splitrs; see individual files for details.

pub mod advancedpatternconfig_traits;
pub mod constants;
pub mod frequencycharacteristics_traits;
pub mod functions;
pub mod patternevolution_traits;
pub mod statisticalproperties_traits;
pub mod trendinfo_traits;
pub mod types;
pub mod types_7;

// Re-export all types
pub use types::*;
pub use types_7::*;
