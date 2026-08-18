// Global Sensitivity Analysis (GSA) for Hyperparameter Exploration
//
// This module provides a suite of methods for analyzing how a model's output
// responds to changes in its input hyperparameters. Sensitivity analysis is a
// crucial step in understanding which hyperparameters are most influential and
// can guide optimization, model interpretability and resource allocation.
//
// # Methods
//
// - [`SobolAnalyzer`] — Variance-based **global** sensitivity analysis using the
//   Saltelli sampling scheme. Decomposes the variance of the model output into
//   contributions from individual parameters (first-order indices) and their
//   interactions (total-order, optionally second-order indices).
// - [`MorrisAnalyzer`] — Elementary Effects (Morris) **screening** method. Useful
//   when the cost of evaluating the model is high; identifies which inputs are
//   negligible, linear, or non-linear/interacting.
// - [`OatAnalyzer`] — One-At-a-Time **local** sensitivity around a baseline.
//   Computes central- and forward-difference gradients to quantify the
//   instantaneous response of the model in the neighborhood of a point.
//
// # Typical workflow
//
// 1. Screen with Morris to filter out non-influential parameters.
// 2. Quantify variance attribution for the surviving parameters with Sobol.
// 3. Use the OAT analyzer for local diagnostics around the optimum.
//
// All analyzers operate on a black-box model `Fn(&Array1<F>) -> F` and the
// rectangular parameter bounds `&[(F, F)]`.

use crate::error::Result;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

pub mod morris;
pub mod oat;
pub mod sobol;

pub use morris::{MorrisAnalyzer, MorrisIndices};
pub use oat::{OatAnalyzer, OatResult};
pub use sobol::SobolAnalyzer;

/// Variance-based sensitivity indices.
///
/// `first_order[i]` is the fraction of the output variance that can be
/// attributed to parameter `i` alone. `total_order[i]` additionally
/// includes all interactions involving parameter `i`. When
/// `second_order` is `Some`, `second_order[i][j]` quantifies the
/// pure interaction between parameters `i` and `j` (excluding their
/// own first-order contributions).
#[derive(Debug, Clone)]
pub struct SensitivityIndices<F: Float> {
    /// First-order Sobol indices `S_i`, one per parameter.
    pub first_order: Vec<F>,
    /// Total-order Sobol indices `S_Ti`, one per parameter.
    pub total_order: Vec<F>,
    /// Optional second-order interaction indices `S_ij`.
    pub second_order: Option<Vec<Vec<F>>>,
    /// Human-readable names for the parameters; same length as `first_order`.
    pub parameter_names: Vec<String>,
}

impl<F: Float> SensitivityIndices<F> {
    /// Number of parameters described by the indices.
    pub fn num_parameters(&self) -> usize {
        self.first_order.len()
    }
}

/// Trait implemented by all sensitivity-analysis algorithms in this module.
///
/// Implementors evaluate `model` at a number of sample points within the
/// rectangular domain defined by `bounds` and return a populated
/// [`SensitivityIndices`] structure.
pub trait SensitivityAnalyzer<F: Float> {
    /// Analyze the sensitivity of `model` over the rectangular domain
    /// specified by `bounds`.
    ///
    /// `bounds[i]` is the `(min, max)` range for parameter `i`.
    fn analyze(
        &mut self,
        model: &dyn Fn(&Array1<F>) -> F,
        bounds: &[(F, F)],
    ) -> Result<SensitivityIndices<F>>;
}
