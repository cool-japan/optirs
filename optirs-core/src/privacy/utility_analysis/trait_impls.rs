//! # AnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::UtilityMetric;
use super::types_3::{
    AnalysisConfig, AnalysisGranularity, BudgetOptimizationMethod, ParameterRange,
    PrivacyParameterSpace, SamplingStrategy,
};

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            privacy_parameters: PrivacyParameterSpace::default(),
            utility_metrics: vec![UtilityMetric::Accuracy, UtilityMetric::F1Score],
            monte_carlo_samples: 1000,
            analysis_granularity: AnalysisGranularity::Medium,
            enable_sensitivity_analysis: true,
            enable_robustness_evaluation: true,
            pareto_resolution: 100,
            budget_optimization_method: BudgetOptimizationMethod::BayesianOptimization,
            confidence_level: 0.95,
            adaptive_analysis: true,
        }
    }
}

impl Default for PrivacyParameterSpace {
    fn default() -> Self {
        Self {
            epsilon_range: ParameterRange {
                min: 0.1,
                max: 10.0,
                num_samples: 50,
                sampling_strategy: SamplingStrategy::Logarithmic,
            },
            delta_range: ParameterRange {
                min: 1e-6,
                max: 1e-3,
                num_samples: 20,
                sampling_strategy: SamplingStrategy::Logarithmic,
            },
            noise_multiplier_range: ParameterRange {
                min: 0.1,
                max: 5.0,
                num_samples: 30,
                sampling_strategy: SamplingStrategy::Linear,
            },
            clipping_threshold_range: ParameterRange {
                min: 0.1,
                max: 10.0,
                num_samples: 25,
                sampling_strategy: SamplingStrategy::Linear,
            },
            sampling_probability_range: ParameterRange {
                min: 0.01,
                max: 1.0,
                num_samples: 20,
                sampling_strategy: SamplingStrategy::Linear,
            },
            iterations_range: ParameterRange {
                min: 100.0,
                max: 10000.0,
                num_samples: 20,
                sampling_strategy: SamplingStrategy::Logarithmic,
            },
            batch_size_range: ParameterRange {
                min: 16.0,
                max: 1024.0,
                num_samples: 15,
                sampling_strategy: SamplingStrategy::Logarithmic,
            },
            learning_rate_range: ParameterRange {
                min: 1e-5,
                max: 1e-1,
                num_samples: 25,
                sampling_strategy: SamplingStrategy::Logarithmic,
            },
        }
    }
}
