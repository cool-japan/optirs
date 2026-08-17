//! Privacy-utility tradeoff analysis.
//!
//! [`types::PrivacyUtilityAnalyzer`] explores a space of differential-privacy
//! configurations, evaluates a caller supplied utility oracle on each of them,
//! and reports the Pareto frontier together with sensitivity, robustness,
//! budget, risk and statistical-test results.

mod analysis;
pub mod functions;
mod risk;
mod robustness;
mod stats;
pub mod trait_impls;
pub mod types;
pub mod types_3;

// Re-export all types
pub use types::*;
pub use types_3::*;
