// Cross-framework benchmarking against PyTorch and TensorFlow optimizers
//
// This module provides comprehensive benchmarking capabilities to compare
// SciRS2 optimizers against their PyTorch and TensorFlow counterparts.
//
// Split into submodules by splitrs; see individual files for details.

pub mod constants;
pub mod crossframeworkbenchmark_impl;
pub mod crossframeworkbenchmark_parsing;
pub mod crossframeworkbenchmark_type;
pub mod crossframeworkconfig_traits;
pub mod framework_traits;
pub mod functions;
pub mod optimizeridentifier_traits;
pub mod type_aliases;
pub mod types;

// Re-export all types
pub use crossframeworkbenchmark_type::*;
pub use types::*;
