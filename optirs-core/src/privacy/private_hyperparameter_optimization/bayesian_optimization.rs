//! `NoisyOptimizer` for private Bayesian optimization.
//!
//! # The defect this replaces
//!
//! ```text
//! Ok(ParameterConfiguration {
//!     values: HashMap::new(),                       // no parameters at all
//!     id: format!("bayesianconfig_{}", history.len()),
//!     metadata: HashMap::new(),
//! })
//! ```
//!
//! was returned for every trial after the first, and `update` was `Ok(())` so
//! the history never grew. There was no surrogate, no kernel and no acquisition
//! function.
//!
//! # What runs now
//!
//! 1. `update` records every evaluation.
//! 2. Until [`COLD_START_TRIALS`] evaluations exist, proposals are drawn
//!    uniformly at random from the encoded space (all five parameter kinds, not
//!    a constant 0.5).
//! 3. After that, a [`GaussianProcessFit`] is fitted to the **already-released**
//!    noisy objectives and [`ExpectedImprovement`] is maximised over a random
//!    candidate pool, which is a real randomized acquisition maximisation.
//!
//! Fitting the surrogate to values that were released under differential privacy
//! is post-processing, so it consumes no additional privacy budget. The
//! observation-noise variance handed to the GP is derived from the DP noise scale
//! recorded on each evaluation (`HPOResult::standard_error`), so the surrogate
//! does not treat a heavily perturbed release as exact.

use crate::error::{OptimError, Result};
use crate::privacy::PrivacyBudget;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::NoisyOptimizer;
use super::gaussian_process::{ConfigurationEncoding, ExpectedImprovement, GaussianProcessFit};
use super::types::{
    HPOEvaluation, HPOResult, ParameterConfiguration, ParameterSpace, PrivateBayesianOptimization,
};

/// Number of random trials before the surrogate is trusted.
pub const COLD_START_TRIALS: usize = 3;

/// Candidate pool size per encoded dimension when maximising the acquisition.
pub const CANDIDATES_PER_DIMENSION: usize = 64;

/// Upper bound on the candidate pool.
pub const MAX_CANDIDATES: usize = 2_048;

impl<T: Float + Debug + Send + Sync + 'static> PrivateBayesianOptimization<T> {
    /// Number of recorded evaluations.
    pub fn history_len(&self) -> usize {
        self.history().len()
    }

    /// Whether a surrogate has been fitted at least once.
    pub fn has_surrogate(&self) -> bool {
        self.gp_model_is_fitted()
    }

    /// Propose the next configuration by maximising expected improvement.
    fn propose_from_surrogate(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        evaluation_history: &[HPOEvaluation<T>],
    ) -> Result<ParameterConfiguration<T>> {
        let encoding = ConfigurationEncoding::from_space(parameterspace)?;

        let mut inputs = Vec::with_capacity(evaluation_history.len());
        let mut targets = Vec::with_capacity(evaluation_history.len());
        let mut noise_scales = Vec::with_capacity(evaluation_history.len());
        for evaluation in evaluation_history {
            inputs.push(encoding.encode(&evaluation.configuration)?);
            let target = evaluation.result.objective_value.to_f64().ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "the objective of evaluation `{}` cannot be represented as f64",
                    evaluation.id
                ))
            })?;
            if !target.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "evaluation `{}` released a non-finite objective, so the surrogate cannot be \
                     fitted",
                    evaluation.id
                )));
            }
            targets.push(target);
            if let Some(scale) = evaluation
                .result
                .standard_error
                .and_then(|scale| scale.to_f64())
            {
                if scale.is_finite() && scale > 0.0 {
                    noise_scales.push(scale);
                }
            }
        }

        // Laplace noise of scale b has variance 2 b^2; averaging the recorded
        // scales gives the surrogate an honest picture of how noisy the
        // observations are. With nothing recorded, fall back to a small jitter
        // that keeps the kernel matrix invertible.
        let noise_variance = if noise_scales.is_empty() {
            1e-6
        } else {
            let mean_scale = noise_scales.iter().sum::<f64>() / noise_scales.len() as f64;
            (2.0 * mean_scale * mean_scale).max(1e-9)
        };

        let surrogate =
            GaussianProcessFit::fit_with_median_heuristic(&inputs, &targets, noise_variance)?;
        let incumbent = targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let acquisition = ExpectedImprovement::new();

        let candidate_count =
            (CANDIDATES_PER_DIMENSION * encoding.dimension()).clamp(32, MAX_CANDIDATES);
        let mut best_point: Option<Vec<f64>> = None;
        let mut best_value = f64::NEG_INFINITY;
        for _ in 0..candidate_count {
            let candidate = encoding.sample_point(self.rng_mut());
            let (mean, variance) = surrogate.predict(&candidate)?;
            let value = acquisition.evaluate(mean, variance, incumbent)?;
            if value > best_value {
                best_value = value;
                best_point = Some(candidate);
            }
        }

        let point = best_point.ok_or_else(|| {
            OptimError::InvalidState(
                "the acquisition maximisation produced no candidate".to_string(),
            )
        })?;
        let mut config: ParameterConfiguration<T> = encoding.decode(
            &point,
            format!("bayesianconfig_{}", evaluation_history.len()),
        )?;
        config.metadata.insert(
            "acquisition".to_string(),
            format!("expected_improvement={best_value:.6}"),
        );
        config.metadata.insert(
            "surrogate_observations".to_string(),
            surrogate.observation_count().to_string(),
        );
        config.metadata.insert(
            "surrogate_length_scale".to_string(),
            format!("{:.6}", surrogate.length_scale()),
        );
        self.record_surrogate(surrogate);
        Ok(config)
    }

    /// Draw a uniformly random configuration from the encoded space.
    fn propose_at_random(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        trial: usize,
    ) -> Result<ParameterConfiguration<T>> {
        let encoding = ConfigurationEncoding::from_space(parameterspace)?;
        let point = encoding.sample_point(self.rng_mut());
        let mut config: ParameterConfiguration<T> =
            encoding.decode(&point, format!("initialconfig_{trial}"))?;
        config
            .metadata
            .insert("proposal".to_string(), "uniform_cold_start".to_string());
        Ok(config)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> NoisyOptimizer<T>
    for PrivateBayesianOptimization<T>
{
    fn suggest_next(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        evaluation_history: &[HPOEvaluation<T>],
        _privacy_budget: &PrivacyBudget,
    ) -> Result<ParameterConfiguration<T>> {
        // Prefer the caller's history (the optimizer loop owns it) and fall back
        // to what `update` recorded, so a caller that only calls `update` still
        // gets a surrogate-driven proposal.
        let history: Vec<HPOEvaluation<T>> = if evaluation_history.is_empty() {
            self.history().to_vec()
        } else {
            evaluation_history.to_vec()
        };

        if history.len() < COLD_START_TRIALS {
            return self.propose_at_random(parameterspace, history.len());
        }
        self.propose_from_surrogate(parameterspace, &history)
    }

    fn update(
        &mut self,
        config: &ParameterConfiguration<T>,
        result: &HPOResult<T>,
        privacy_budget: &PrivacyBudget,
    ) -> Result<()> {
        // `Ok(())` here is what made the history permanently empty, so the
        // surrogate never had anything to fit.
        self.push_history(HPOEvaluation {
            id: format!("bayes_update_{}", self.history().len()),
            configuration: config.clone(),
            result: result.clone(),
            privacy_cost: privacy_budget.clone(),
            timestamp: super::types::unix_timestamp()?,
            metadata: HashMap::new(),
        });
        Ok(())
    }

    fn name(&self) -> &str {
        "PrivateBayesianOptimization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::private_hyperparameter_optimization::types::{
        BudgetAllocationStrategy, EarlyStoppingConfig, EvaluationStatus,
        HyperparameterNoiseMechanism, ParameterBounds, ParameterDefinition, ParameterType,
        ParameterValue, PrivateHPOConfig, SearchAlgorithm, SensitivityBounds, ValidationStrategy,
    };
    use crate::privacy::DifferentialPrivacyConfig;

    fn hpo_config() -> PrivateHPOConfig<f64> {
        PrivateHPOConfig {
            base_privacyconfig: DifferentialPrivacyConfig::default(),
            budget_allocation: BudgetAllocationStrategy::Equal,
            search_algorithm: SearchAlgorithm::BayesianOptimization,
            num_evaluations: 12,
            cv_folds: 3,
            early_stopping: EarlyStoppingConfig {
                enabled: false,
                patience: 2,
                min_improvement: 1e-3,
                max_evaluations: 12,
            },
            noise_mechanism: HyperparameterNoiseMechanism::Laplace,
            sensitivity_bounds: SensitivityBounds {
                global_sensitivity: HashMap::new(),
                local_sensitivity: HashMap::new(),
                smooth_sensitivity: HashMap::new(),
            },
            private_model_selection: false,
            validation_strategy: ValidationStrategy::HoldOut,
        }
    }

    fn two_parameter_space() -> ParameterSpace<f64> {
        let mut parameters = HashMap::new();
        parameters.insert(
            "learning_rate".to_string(),
            ParameterDefinition {
                name: "learning_rate".to_string(),
                param_type: ParameterType::Continuous,
                bounds: ParameterBounds {
                    min: Some(0.0),
                    max: Some(1.0),
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        parameters.insert(
            "optimizer".to_string(),
            ParameterDefinition {
                name: "optimizer".to_string(),
                param_type: ParameterType::Categorical(vec!["sgd".to_string(), "adam".to_string()]),
                bounds: ParameterBounds {
                    min: None,
                    max: None,
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        ParameterSpace {
            parameters,
            constraints: Vec::new(),
            defaultconfig: None,
        }
    }

    fn result(value: f64) -> HPOResult<f64> {
        HPOResult {
            objective_value: value,
            standard_error: Some(0.01),
            cv_scores: None,
            training_time: None,
            complexity_metrics: HashMap::new(),
            additional_metrics: HashMap::new(),
            status: EvaluationStatus::Success,
        }
    }

    fn learning_rate(config: &ParameterConfiguration<f64>) -> f64 {
        match config.values.get("learning_rate") {
            Some(ParameterValue::Continuous(value)) => *value,
            other => panic!("expected a continuous learning_rate, got {other:?}"),
        }
    }

    #[test]
    fn every_proposal_sets_every_parameter() {
        // Regression: with a non-empty history the proposal used to be an
        // *empty* configuration, so every trial after the first proposed
        // nothing at all.
        let space = two_parameter_space();
        let mut optimizer = match PrivateBayesianOptimization::<f64>::new_with_seed(hpo_config(), 5)
        {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        let budget = PrivacyBudget::default();
        let mut history: Vec<HPOEvaluation<f64>> = Vec::new();

        for trial in 0..8usize {
            let config = match optimizer.suggest_next(&space, &history, &budget) {
                Ok(config) => config,
                Err(err) => panic!("trial {trial} failed: {err}"),
            };
            assert_eq!(
                config.values.len(),
                2,
                "trial {trial} proposed {} parameters",
                config.values.len()
            );
            assert!(config.values.contains_key("learning_rate"));
            assert!(config.values.contains_key("optimizer"));

            let score = -(learning_rate(&config) - 0.7).powi(2);
            let evaluation_result = result(score);
            let ok = optimizer.update(&config, &evaluation_result, &budget);
            assert!(ok.is_ok(), "update failed");
            history.push(HPOEvaluation {
                id: format!("eval_{trial}"),
                configuration: config,
                result: evaluation_result,
                privacy_cost: budget.clone(),
                timestamp: trial as u64,
                metadata: HashMap::new(),
            });
        }
    }

    #[test]
    fn update_records_the_history_it_used_to_discard() {
        let space = two_parameter_space();
        let mut optimizer = match PrivateBayesianOptimization::<f64>::new_with_seed(hpo_config(), 6)
        {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        let budget = PrivacyBudget::default();
        assert_eq!(optimizer.history_len(), 0);
        for _ in 0..4 {
            let config = match optimizer.suggest_next(&space, &[], &budget) {
                Ok(config) => config,
                Err(err) => panic!("suggest failed: {err}"),
            };
            let ok = optimizer.update(&config, &result(0.5), &budget);
            assert!(ok.is_ok());
        }
        assert_eq!(optimizer.history_len(), 4);
        assert!(
            optimizer.has_surrogate(),
            "with four recorded evaluations a surrogate must have been fitted"
        );
    }

    #[test]
    fn the_surrogate_steers_proposals_towards_the_optimum() {
        // A deterministic quadratic objective peaking at 0.7. After enough
        // trials the GP-driven proposals must cluster near it -- something an
        // empty configuration could never do.
        let space = two_parameter_space();
        let mut optimizer =
            match PrivateBayesianOptimization::<f64>::new_with_seed(hpo_config(), 17) {
                Ok(optimizer) => optimizer,
                Err(err) => panic!("construction failed: {err}"),
            };
        let budget = PrivacyBudget::default();
        let mut history: Vec<HPOEvaluation<f64>> = Vec::new();
        let mut late_distances = Vec::new();

        for trial in 0..40usize {
            let config = match optimizer.suggest_next(&space, &history, &budget) {
                Ok(config) => config,
                Err(err) => panic!("trial {trial} failed: {err}"),
            };
            let rate = learning_rate(&config);
            if trial >= 20 {
                late_distances.push((rate - 0.7).abs());
            }
            let score = -(rate - 0.7).powi(2);
            let evaluation_result = HPOResult {
                objective_value: score,
                // Tiny observation noise so the surrogate can actually learn.
                standard_error: Some(1e-4),
                cv_scores: None,
                training_time: None,
                complexity_metrics: HashMap::new(),
                additional_metrics: HashMap::new(),
                status: EvaluationStatus::Success,
            };
            history.push(HPOEvaluation {
                id: format!("eval_{trial}"),
                configuration: config,
                result: evaluation_result,
                privacy_cost: budget.clone(),
                timestamp: trial as u64,
                metadata: HashMap::new(),
            });
        }

        let mean_late_distance = late_distances.iter().sum::<f64>() / late_distances.len() as f64;
        assert!(
            mean_late_distance < 0.25,
            "late proposals averaged {mean_late_distance} away from the optimum; the surrogate is \
             not steering the search"
        );
    }

    #[test]
    fn proposals_carry_the_acquisition_value_they_were_chosen_for() {
        let space = two_parameter_space();
        let mut optimizer = match PrivateBayesianOptimization::<f64>::new_with_seed(hpo_config(), 9)
        {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        let budget = PrivacyBudget::default();
        let mut history: Vec<HPOEvaluation<f64>> = Vec::new();
        for trial in 0..COLD_START_TRIALS {
            let config = match optimizer.suggest_next(&space, &history, &budget) {
                Ok(config) => config,
                Err(err) => panic!("cold start failed: {err}"),
            };
            assert_eq!(
                config.metadata.get("proposal").map(String::as_str),
                Some("uniform_cold_start")
            );
            history.push(HPOEvaluation {
                id: format!("eval_{trial}"),
                configuration: config,
                result: result(0.1 * trial as f64),
                privacy_cost: budget.clone(),
                timestamp: trial as u64,
                metadata: HashMap::new(),
            });
        }
        let config = match optimizer.suggest_next(&space, &history, &budget) {
            Ok(config) => config,
            Err(err) => panic!("surrogate proposal failed: {err}"),
        };
        assert!(config.metadata.contains_key("acquisition"));
        assert_eq!(
            config
                .metadata
                .get("surrogate_observations")
                .map(String::as_str),
            Some("3")
        );
    }

    #[test]
    fn a_non_finite_released_objective_is_refused_rather_than_fitted() {
        let space = two_parameter_space();
        let mut optimizer = match PrivateBayesianOptimization::<f64>::new_with_seed(hpo_config(), 3)
        {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        let budget = PrivacyBudget::default();
        let mut history: Vec<HPOEvaluation<f64>> = Vec::new();
        for trial in 0..COLD_START_TRIALS {
            let config = match optimizer.suggest_next(&space, &history, &budget) {
                Ok(config) => config,
                Err(err) => panic!("cold start failed: {err}"),
            };
            history.push(HPOEvaluation {
                id: format!("eval_{trial}"),
                configuration: config,
                result: result(if trial == 1 { f64::NAN } else { 0.5 }),
                privacy_cost: budget.clone(),
                timestamp: trial as u64,
                metadata: HashMap::new(),
            });
        }
        assert!(optimizer.suggest_next(&space, &history, &budget).is_err());
    }

    #[test]
    fn two_optimizers_do_not_walk_the_same_trajectory() {
        // The cold-start path used to build `Random::seed(42)` inside
        // `suggest_next`, so the first proposal was a compile-time constant.
        let space = two_parameter_space();
        let budget = PrivacyBudget::default();
        let mut first = match PrivateBayesianOptimization::<f64>::new(hpo_config()) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        let mut second = match PrivateBayesianOptimization::<f64>::new(hpo_config()) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        let left: Vec<f64> = (0..6)
            .map(|_| match first.suggest_next(&space, &[], &budget) {
                Ok(config) => learning_rate(&config),
                Err(err) => panic!("suggest failed: {err}"),
            })
            .collect();
        let right: Vec<f64> = (0..6)
            .map(|_| match second.suggest_next(&space, &[], &budget) {
                Ok(config) => learning_rate(&config),
                Err(err) => panic!("suggest failed: {err}"),
            })
            .collect();
        assert_ne!(left, right);
    }

    #[test]
    fn an_explicit_seed_is_reproducible() {
        let space = two_parameter_space();
        let budget = PrivacyBudget::default();
        let propose = |seed: u64| -> Vec<f64> {
            let mut optimizer =
                match PrivateBayesianOptimization::<f64>::new_with_seed(hpo_config(), seed) {
                    Ok(optimizer) => optimizer,
                    Err(err) => panic!("construction failed: {err}"),
                };
            (0..6)
                .map(|_| match optimizer.suggest_next(&space, &[], &budget) {
                    Ok(config) => learning_rate(&config),
                    Err(err) => panic!("suggest failed: {err}"),
                })
                .collect()
        };
        assert_eq!(propose(21), propose(21));
    }
}
