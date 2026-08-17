//! Default configurations for the privacy-utility analysis.
//!
//! The literal bounds used here are statically known to satisfy
//! [`ParameterRange::validate`], so they are built through the internal
//! unchecked constructor; every value is validated again before it is sampled.

use super::types_3::{
    AnalysisConfig, EpsilonSemantics, ParameterRange, PrivacyParameterSpace, SamplingStrategy,
};

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            privacy_parameters: PrivacyParameterSpace::default(),
            // 32 replicates give a usable standard error for the sensitivity
            // intervals at 32 extra model evaluations per parameter.
            monte_carlo_samples: 32,
            enable_sensitivity_analysis: true,
            enable_robustness_evaluation: true,
            pareto_resolution: 100,
            confidence_level: 0.95,
            target_power: 0.8,
            utility_degradation_threshold: 0.25,
            epsilon_semantics: EpsilonSemantics::Total,
            random_seed: None,
        }
    }
}

impl Default for PrivacyParameterSpace {
    fn default() -> Self {
        Self {
            epsilon_range: ParameterRange::new_unchecked(
                0.1,
                10.0,
                50,
                SamplingStrategy::Logarithmic,
            ),
            delta_range: ParameterRange::new_unchecked(
                1e-6,
                1e-3,
                20,
                SamplingStrategy::Logarithmic,
            ),
            noise_multiplier_range: ParameterRange::new_unchecked(
                0.1,
                5.0,
                30,
                SamplingStrategy::Linear,
            ),
            clipping_threshold_range: ParameterRange::new_unchecked(
                0.1,
                10.0,
                25,
                SamplingStrategy::Linear,
            ),
            sampling_probability_range: ParameterRange::new_unchecked(
                0.01,
                1.0,
                20,
                SamplingStrategy::Linear,
            ),
            iterations_range: ParameterRange::new_unchecked(
                100.0,
                10000.0,
                20,
                SamplingStrategy::Logarithmic,
            ),
            batch_size_range: ParameterRange::new_unchecked(
                16.0,
                1024.0,
                15,
                SamplingStrategy::Logarithmic,
            ),
            learning_rate_range: ParameterRange::new_unchecked(
                1e-5,
                1e-1,
                25,
                SamplingStrategy::Logarithmic,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        AnalysisConfig::default()
            .validate()
            .expect("the library default configuration must validate");
    }

    #[test]
    fn test_default_parameter_space_ranges_are_valid() {
        PrivacyParameterSpace::default()
            .validate()
            .expect("the library default parameter space must validate");
    }
}
