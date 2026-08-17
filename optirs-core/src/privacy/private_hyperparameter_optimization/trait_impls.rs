//! `Default` implementations for the private-HPO components.
//!
//! This file replaces the 16 auto-generated `*_traits.rs` shells listed in the
//! module documentation, each of which held a single
//! `Default::default() -> Self::new()` forwarding impl. No implementation was
//! dropped, so the change is source-compatible; only the empty module paths are
//! gone.

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::budget_manager::AdaptiveBudgetController;
use super::types::{
    AcquisitionFunction, AnomalyDetector, BootstrapEstimator, ConfidenceEstimation,
    GaussianProcessModel, KernelFunction, ObjectiveNoiseMechanism, ObjectiveSensitivityAnalyzer,
    PrivateCrossValidation, PrivateFoldAggregation, ResultValidator,
    SampleBasedSensitivityEstimator, SearchStrategy, SelectionMechanism, UtilityFunction,
};

impl Default for AdaptiveBudgetController {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! default_via_new {
    ($($type_name:ident),* $(,)?) => {
        $(
            impl<T: Float + Debug + Send + Sync + 'static> Default for $type_name<T> {
                fn default() -> Self {
                    Self::new()
                }
            }
        )*
    };
}

default_via_new!(
    AcquisitionFunction,
    AnomalyDetector,
    BootstrapEstimator,
    ConfidenceEstimation,
    GaussianProcessModel,
    KernelFunction,
    ObjectiveNoiseMechanism,
    ObjectiveSensitivityAnalyzer,
    PrivateCrossValidation,
    PrivateFoldAggregation,
    ResultValidator,
    SampleBasedSensitivityEstimator,
    SearchStrategy,
    SelectionMechanism,
    UtilityFunction,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::private_hyperparameter_optimization::types::HyperparameterNoiseMechanism;

    #[test]
    fn every_component_still_has_a_default() {
        let _ = AdaptiveBudgetController::default();
        let _ = AcquisitionFunction::<f64>::default();
        let _ = AnomalyDetector::<f64>::default();
        let _ = BootstrapEstimator::<f64>::default();
        let _ = ConfidenceEstimation::<f64>::default();
        let _ = GaussianProcessModel::<f64>::default();
        let _ = KernelFunction::<f64>::default();
        let _ = ObjectiveSensitivityAnalyzer::<f64>::default();
        let _ = PrivateCrossValidation::<f64>::default();
        let _ = PrivateFoldAggregation::<f64>::default();
        let _ = ResultValidator::<f64>::default();
        let _ = SampleBasedSensitivityEstimator::<f64>::default();
        let _ = SearchStrategy::<f64>::default();
        let _ = UtilityFunction::<f64>::default();

        let noise = ObjectiveNoiseMechanism::<f64>::default();
        assert!(matches!(
            noise.mechanism_type(),
            HyperparameterNoiseMechanism::Laplace
        ));
        let selection = SelectionMechanism::<f64>::default();
        assert!(matches!(
            selection.mechanism_type(),
            HyperparameterNoiseMechanism::Exponential
        ));
    }
}
