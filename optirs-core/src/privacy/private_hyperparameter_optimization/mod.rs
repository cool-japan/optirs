//! Differentially private hyperparameter optimization.
//!
//! # Module layout
//!
//! | module | responsibility |
//! |---|---|
//! | [`types`] | configuration and the search-component data types |
//! | [`budget_manager`] | privacy budget allocation and accounting |
//! | [`optimizer`] | the `PrivateHyperparameterOptimizer` driver |
//! | [`results`] | private aggregation and the selection report |
//! | [`selection`] | the private selection mechanisms and noisy statistics |
//! | [`gaussian_process`] | the GP surrogate and acquisition function |
//! | [`random_search`] | `NoisyOptimizer` for random search |
//! | [`bayesian_optimization`] | `NoisyOptimizer` for Bayesian optimization |
//! | [`functions`] | the `NoisyOptimizer` trait and function aliases |
//! | [`trait_impls`] | the `Default` impls |
//!
//! # 0.3.2 notes
//!
//! Hyperparameter *selection* is now differentially private. Previously
//! `HyperparameterNoiseMechanism` was stored and never matched on: the choice
//! was the exact argmax over utilities computed from the private data, which is
//! precisely what private HPO exists to avoid.
//!
//! Behavioural and API changes:
//!
//! * [`types::PrivateHyperparameterOptimizer::new`] rejects a configuration with
//!   `private_model_selection: true` that declares no objective sensitivity.
//! * [`types::PrivateHPOResults`] gained `selection`, which records whether the
//!   returned configuration was chosen privately, by which mechanism, and at
//!   what cost.
//! * [`types::SelectionParameters`] gained `delta` (needed to calibrate Gaussian
//!   selection).
//! * `PrivateResultsAggregator::aggregate_results` takes `&mut self`, because
//!   selecting now spends budget.
//! * The 16 `*_traits.rs` shells were collapsed into [`trait_impls`]; the two
//!   that held real `NoisyOptimizer` implementations were renamed to
//!   [`random_search`] and [`bayesian_optimization`].

pub mod bayesian_optimization;
pub mod budget_manager;
pub mod functions;
pub mod gaussian_process;
pub mod optimizer;
pub mod random_search;
pub mod results;
pub mod selection;
pub mod trait_impls;
pub mod types;

pub use budget_manager::{
    AdaptiveBudgetController, HPOBudgetManager, DEFAULT_SELECTION_BUDGET_FRACTION,
};
pub use functions::*;
pub use gaussian_process::{
    encode_configuration, ConfigurationEncoding, ExpectedImprovement, GaussianProcessFit,
};
pub use optimizer::PrivateHyperparameterOptimizer;
pub use results::{PrivateResultsAggregator, SelectionReport, PRIVATE_TOP_K};
pub use selection::{
    exponential_mechanism_index, exponential_mechanism_probabilities, gaussian_sigma,
    laplace_sample, mechanism_name, noisy_summary_statistics, report_noisy_max_gaussian,
    report_noisy_max_gumbel, report_noisy_max_laplace, summary_mean_noise_scale, NoisySummary,
    SelectionOutcome, OBJECTIVE_SENSITIVITY_KEY, SUMMARY_QUANTILES,
};
pub use types::*;
