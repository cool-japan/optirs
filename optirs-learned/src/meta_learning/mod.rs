//! Auto-generated module structure

pub mod datasetmetadata_traits;
pub mod framework;
pub mod functions;
pub mod linear_model;
pub mod mamllearner_traits;
pub mod meta_sgd_learner;
pub mod metaoptimizationtracker_traits;
pub mod metaparameters_traits;
pub mod metatask_traits;
pub mod metrics;
pub mod reptile_learner;
pub mod taskdataset_traits;
pub mod taskmetadata_traits;
pub mod types;

// Re-export all types.
//
// The `*_traits` submodules hold only trait `impl`s for types declared in
// `types`, so they export no names of their own; glob re-exporting them was a
// no-op. The `impl`s stay active through `pub mod`.
pub use framework::*;
pub use functions::*;
pub use linear_model::{BIAS_KEY, WEIGHTS_KEY};
pub use meta_sgd_learner::*;
pub use reptile_learner::*;
pub use types::*;
