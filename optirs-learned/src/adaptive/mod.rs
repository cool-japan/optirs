//! Auto-generated module structure

pub mod adaptiveconfig_traits;
pub mod architecture_adapter;
pub mod architecturesearchspace_traits;
pub mod functions;
pub mod landscape;
pub mod landscapefeatures_traits;
pub mod performance_predictor;
pub mod predictor;
pub mod resourceconstraints_traits;
pub mod types;

// Re-export all types.
//
// The `*_traits` submodules and `functions` hold only trait `impl`s (and tests)
// for types declared in `types`, so they export no names of their own; glob
// re-exporting them was a no-op. The `impl`s stay active through `pub mod`.
pub use architecture_adapter::*;
pub use landscape::*;
pub use performance_predictor::*;
pub use predictor::*;
pub use types::*;
