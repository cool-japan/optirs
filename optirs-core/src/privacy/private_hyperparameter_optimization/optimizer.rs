//! The private hyperparameter optimizer driver.
//!
//! Extracted from `types.rs` to keep every file under the 2000-line limit.

use crate::error::{OptimError, Result};
use crate::privacy::PrivacyBudget;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::budget_manager::{HPOBudgetManager, DEFAULT_SELECTION_BUDGET_FRACTION};
use super::functions::{NoisyOptimizer, ObjectiveFn};
use super::results::{PrivateResultsAggregator, SelectionReport, PRIVATE_TOP_K};
use super::types::{
    unix_timestamp, EvaluationStatus, HPOEvaluation, HyperparameterNoiseMechanism, NoiseParameters,
    ObjectiveNoiseMechanism, OptimizationStats, ParameterSpace, PrivateBayesianOptimization,
    PrivateHPOConfig, PrivateHPOResults, PrivateObjective, PrivateRandomSearch, SearchAlgorithm,
};

/// The registry key of the private optimizer that implements `algorithm`.
///
/// Returns [`OptimError::UnsupportedOperation`] for the five `SearchAlgorithm`
/// variants that have no private implementation here. They used to be mapped to
/// `"random_search"` by a `_ =>` arm, so the configured algorithm never ran and
/// nothing said so.
pub(crate) fn optimizer_key(algorithm: SearchAlgorithm) -> Result<&'static str> {
    match algorithm {
        SearchAlgorithm::RandomSearch => Ok("random_search"),
        SearchAlgorithm::BayesianOptimization => Ok("bayesian_opt"),
        other => Err(OptimError::UnsupportedOperation(format!(
            "SearchAlgorithm::{other:?} has no differentially private implementation in this \
             crate; configure SearchAlgorithm::RandomSearch or \
             SearchAlgorithm::BayesianOptimization"
        ))),
    }
}

/// Privacy-preserving hyperparameter optimizer
pub struct PrivateHyperparameterOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for privacy-preserving hyperparameter optimization
    config: PrivateHPOConfig<T>,
    /// Privacy budget manager
    budget_manager: HPOBudgetManager,
    /// Noisy optimization algorithms
    noisy_optimizers: HashMap<String, Box<dyn NoisyOptimizer<T>>>,
    /// Hyperparameter space definition
    parameterspace: ParameterSpace<T>,
    /// Objective function with privacy guarantees
    private_objective: PrivateObjective<T>,
    /// Results aggregator with privacy
    results_aggregator: PrivateResultsAggregator<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> PrivateHyperparameterOptimizer<T> {
    /// Create new private hyperparameter optimizer.
    ///
    /// The objective's global sensitivity **must** be declared in
    /// `config.sensitivity_bounds` (under `"objective"`, or as the only entry).
    /// Every evaluation releases its objective under a differentially private
    /// noise mechanism whose scale is `sensitivity / epsilon`, so an undeclared
    /// sensitivity has no safe default: substituting `1.0` silently rescales the
    /// noise, and every epsilon reported afterwards would describe a guarantee
    /// the run did not deliver. Construction therefore fails instead of guessing,
    /// whether or not `private_model_selection` is set.
    pub fn new(config: PrivateHPOConfig<T>, parameterspace: ParameterSpace<T>) -> Result<Self> {
        if parameterspace.parameters.is_empty() {
            return Err(OptimError::InvalidConfig(
                "the parameter space declares no hyperparameter to search".to_string(),
            ));
        }

        // The objective release always needs a sensitivity; the private selection
        // needs the same number to calibrate the exponential mechanism.
        let objective_sensitivity = match config.sensitivity_bounds.objective_sensitivity() {
            Some(sensitivity) => {
                let as_f64 = sensitivity.to_f64().unwrap_or(f64::NAN);
                if !as_f64.is_finite() || as_f64 <= 0.0 {
                    return Err(OptimError::InvalidPrivacyConfig(format!(
                        "the declared objective sensitivity is {as_f64}; it must be positive \
                         and finite"
                    )));
                }
                sensitivity
            }
            None => {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "no objective sensitivity is declared in \
                     sensitivity_bounds.global_sensitivity (expected the key `{}`); the \
                     objective's noise scale is sensitivity / epsilon and the exponential \
                     mechanism is calibrated with the same number, so neither can be derived \
                     without it",
                    super::selection::OBJECTIVE_SENSITIVITY_KEY
                )))
            }
        };
        let selection_sensitivity = if config.private_model_selection {
            Some(objective_sensitivity)
        } else {
            None
        };

        let selection_fraction = if config.private_model_selection {
            DEFAULT_SELECTION_BUDGET_FRACTION
        } else {
            0.0
        };
        let budget_manager = HPOBudgetManager::with_selection_reserve(
            config.base_privacyconfig.clone(),
            config.budget_allocation,
            config.num_evaluations,
            selection_fraction,
        )?;
        // Only two search algorithms have a private implementation. The other
        // five used to fall through to `PrivateRandomSearch`, so a caller asking
        // for TPE (or a genetic search, or simulated annealing) silently got
        // uniform random proposals and no indication that the algorithm it
        // configured had never run.
        let optimizer_name = optimizer_key(config.search_algorithm)?;
        let mut noisy_optimizers: HashMap<String, Box<dyn NoisyOptimizer<T>>> = HashMap::new();
        match config.search_algorithm {
            SearchAlgorithm::RandomSearch => {
                noisy_optimizers.insert(
                    optimizer_name.to_string(),
                    Box::new(PrivateRandomSearch::new(config.clone())?),
                );
            }
            SearchAlgorithm::BayesianOptimization => {
                noisy_optimizers.insert(
                    optimizer_name.to_string(),
                    Box::new(PrivateBayesianOptimization::new(config.clone())?),
                );
            }
            other => {
                return Err(OptimError::UnsupportedOperation(format!(
                    "SearchAlgorithm::{other:?} has no differentially private implementation in \
                     this crate; configure SearchAlgorithm::RandomSearch or \
                     SearchAlgorithm::BayesianOptimization"
                )))
            }
        }

        // The Gaussian selection mechanism is an (epsilon, delta) mechanism. Its
        // delta comes from the base configuration's reporting delta -- never from
        // a hardcoded constant -- and the classic Gaussian bound additionally
        // requires each draw's epsilon to be at most 1, which is checked here so
        // the failure surfaces at construction rather than mid-run.
        let selection_delta = if matches!(
            config.noise_mechanism,
            HyperparameterNoiseMechanism::Gaussian
        ) {
            let delta = config.base_privacyconfig.target_delta;
            if !delta.is_finite() || !(0.0..1.0).contains(&delta) || delta <= 0.0 {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "Gaussian hyperparameter selection needs a reporting delta in (0, 1), but \
                     base_privacyconfig.target_delta is {delta}"
                )));
            }
            // `aggregate_results` draws `k = PRIVATE_TOP_K.min(evaluations)` times
            // and splits half the selection epsilon across them, so the per-draw
            // epsilon is largest when the run produces the fewest evaluations.
            // Using `num_evaluations` here matches the k the run will actually
            // use; a run cut short by budget exhaustion draws fewer times, and
            // `gaussian_sigma` refuses the oversized epsilon at that point rather
            // than quietly widening the guarantee.
            let draws = PRIVATE_TOP_K.min(config.num_evaluations.max(1));
            let per_draw_epsilon = budget_manager.selection_epsilon() * 0.5 / draws as f64;
            if per_draw_epsilon > 1.0 {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "Gaussian selection would draw at epsilon {per_draw_epsilon} per selection, \
                     but the classic Gaussian bound requires epsilon <= 1; lower target_epsilon \
                     or choose HyperparameterNoiseMechanism::Exponential"
                )));
            }
            Some(delta)
        } else {
            None
        };

        let results_aggregator = match selection_sensitivity {
            Some(sensitivity) => PrivateResultsAggregator::with_selection_budget(
                budget_manager.selection_epsilon(),
                selection_delta,
                config.noise_mechanism,
                sensitivity,
                1.0,
            )?,
            None => PrivateResultsAggregator::new()?,
        };

        // The objective release is charged out of each evaluation's grant, so
        // it must use a scalar-perturbation mechanism. `noise_mechanism`
        // configures the *selection*; the objective always uses Laplace unless
        // the user asked for Gaussian.
        let objective_mechanism = match config.noise_mechanism {
            HyperparameterNoiseMechanism::Gaussian => HyperparameterNoiseMechanism::Gaussian,
            _ => HyperparameterNoiseMechanism::Laplace,
        };
        let private_objective =
            PrivateObjective::with_noise_mechanism(ObjectiveNoiseMechanism::with_parameters(
                objective_mechanism,
                NoiseParameters {
                    scale: T::one(),
                    sensitivity: objective_sensitivity,
                    epsilon: 1.0,
                    delta: Some(
                        config
                            .base_privacyconfig
                            .target_delta
                            .max(f64::MIN_POSITIVE),
                    ),
                },
            )?)?;

        Ok(Self {
            config,
            budget_manager,
            noisy_optimizers,
            parameterspace,
            private_objective,
            results_aggregator,
        })
    }

    /// Seed every stochastic component deterministically (tests only).
    ///
    /// The sub-seeds are domain-separated. Seeding the objective's noise
    /// mechanism and the selection mechanism from the *same* seed makes both
    /// draw the same underlying uniform stream, so the evaluation that receives
    /// the largest objective noise also receives the largest selection noise --
    /// the private selection then reproduces the exact argmax and looks
    /// deterministic when it is not. That correlation is an artefact of the test
    /// harness, not of the mechanisms, and this is where it is avoided.
    pub fn seed_for_tests(&mut self, seed: u64) {
        const OBJECTIVE_DOMAIN: u64 = 0x9E37_79B9_7F4A_7C15;
        const SELECTION_DOMAIN: u64 = 0xC2B2_AE3D_27D4_EB4F;
        self.results_aggregator
            .seed_for_tests(seed.wrapping_mul(SELECTION_DOMAIN) | 1);
        self.private_objective
            .seed_for_tests(seed.wrapping_mul(OBJECTIVE_DOMAIN) | 1);
    }

    /// The epsilon spent so far across every objective release and the private
    /// selection.
    ///
    /// This used to be `privacy_accountant() -> &MomentsAccountant`. That
    /// accountant was constructed from `base_privacyconfig`'s DP-SGD parameters
    /// (`noise_multiplier`, `batch_size`, `dataset_size`) and then **never
    /// stepped**, so it reported the spend of a training run that had not
    /// happened while the hyperparameter search's real, pure-epsilon spend was
    /// tracked entirely by [`HPOBudgetManager`]. A moments accountant models
    /// subsampled-Gaussian composition and is the wrong primitive for the
    /// Laplace / exponential releases this optimizer performs, so it is gone
    /// rather than fed fabricated `(sigma, q)` pairs. Read the real ledger here
    /// or in [`PrivateHPOResults::total_privacy_cost`].
    pub fn total_privacy_cost(&self) -> PrivacyBudget {
        self.budget_manager.get_total_consumed_budget()
    }

    /// The budget manager.
    pub fn budget_manager(&self) -> &HPOBudgetManager {
        &self.budget_manager
    }

    /// The private objective, including the noise mechanism and the scale it
    /// last used.
    pub fn private_objective(&self) -> &PrivateObjective<T> {
        &self.private_objective
    }

    /// Optimize hyperparameters with differential privacy.
    ///
    /// The final configuration is chosen by the configured private selection
    /// mechanism when `private_model_selection` is set. When it is not, the
    /// exact argmax is returned and [`PrivateHPOResults::selection`] records
    /// `was_private: false` so the caller cannot mistake it for a private
    /// choice.
    pub fn optimize(&mut self, objective_fn: ObjectiveFn<T>) -> Result<PrivateHPOResults<T>> {
        self.private_objective.set_objective(objective_fn)?;
        let started = std::time::Instant::now();
        let mut evaluations = Vec::new();
        let mut evaluation_durations: Vec<f64> = Vec::new();
        let mut failed_evaluations = 0usize;
        let mut last_error: Option<OptimError> = None;
        let mut best_score_so_far = T::neg_infinity();
        let mut convergence_iteration = None;
        // `new` already refused every algorithm without an implementation, so
        // this cannot silently pick a different optimizer than the caller asked
        // for; it propagates rather than defaulting all the same.
        let optimizer_name = optimizer_key(self.config.search_algorithm)?;
        for iteration in 0..self.config.num_evaluations {
            if !self.budget_manager.has_budget_remaining()? {
                break;
            }
            let evaluation_budget = self.budget_manager.get_evaluation_budget(iteration)?;
            let config = if let Some(optimizer) = self.noisy_optimizers.get_mut(optimizer_name) {
                optimizer.suggest_next(&self.parameterspace, &evaluations, &evaluation_budget)?
            } else {
                return Err(OptimError::InvalidConfig(
                    "No optimizer available".to_string(),
                ));
            };
            let evaluation_started = std::time::Instant::now();
            let result = match self.private_objective.evaluate(&config, &evaluation_budget) {
                Ok(result) => result,
                Err(err) => {
                    // A failing objective still touched the data, so charge its
                    // grant (fail closed) and record the failure rather than
                    // pretending the trial never happened. If every trial fails
                    // the error is propagated below.
                    failed_evaluations += 1;
                    last_error = Some(err);
                    self.budget_manager
                        .record_evaluation(&evaluation_budget, 0.0)?;
                    continue;
                }
            };
            evaluation_durations.push(evaluation_started.elapsed().as_secs_f64());
            let evaluation = HPOEvaluation {
                id: format!("eval_{}", iteration),
                configuration: config.clone(),
                result: result.clone(),
                privacy_cost: evaluation_budget.clone(),
                timestamp: unix_timestamp()?,
                metadata: HashMap::new(),
            };
            if result.objective_value > best_score_so_far {
                best_score_so_far = result.objective_value;
                convergence_iteration = Some(iteration);
            }
            if let Some(optimizer) = self.noisy_optimizers.get_mut(optimizer_name) {
                optimizer.update(&config, &result, &evaluation_budget)?;
            }
            evaluations.push(evaluation);
            self.budget_manager.record_evaluation(
                &evaluation_budget,
                result.objective_value.to_f64().unwrap_or(0.0),
            )?;
            if self.should_stop_early(&evaluations)? {
                break;
            }
        }

        if evaluations.is_empty() {
            return Err(last_error.unwrap_or(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.budget_manager.epsilon_spent(),
                target_epsilon: self.config.base_privacyconfig.target_epsilon,
            }));
        }

        let final_results = self.results_aggregator.aggregate_results(&evaluations)?;

        let (bestconfiguration, best_score, selection) = if self.config.private_model_selection {
            let spent = self
                .results_aggregator
                .selection_mechanism()
                .epsilon_spent();
            self.budget_manager.record_selection_spend(spent)?;
            let mut report = self.results_aggregator.selection_report();
            let chosen = final_results
                .topconfigurations
                .first()
                .cloned()
                .ok_or_else(|| {
                    OptimError::InvalidState(
                        "the private selection returned no configuration".to_string(),
                    )
                })?;
            report.selected_probability = final_results
                .model_selection
                .as_ref()
                .map(|selection| selection.selection_confidence);
            (Some(chosen.0), chosen.1, report)
        } else {
            // Non-private fallback: the exact argmax. Recorded as such.
            let mut best_index = 0usize;
            let mut best = T::neg_infinity();
            for (index, evaluation) in evaluations.iter().enumerate() {
                if evaluation.result.objective_value > best {
                    best = evaluation.result.objective_value;
                    best_index = index;
                }
            }
            (
                Some(evaluations[best_index].configuration.clone()),
                best,
                SelectionReport {
                    was_private: false,
                    mechanism: "exact_argmax".to_string(),
                    epsilon_spent: 0.0,
                    delta_spent: 0.0,
                    utility_sensitivity: f64::NAN,
                    selected_probability: None,
                },
            )
        };

        let optimization_stats = self.compute_optimization_stats(
            &evaluations,
            &evaluation_durations,
            failed_evaluations,
            started.elapsed().as_secs_f64(),
            convergence_iteration,
        )?;

        Ok(PrivateHPOResults {
            bestconfiguration,
            best_score,
            all_evaluations: evaluations,
            final_results,
            total_privacy_cost: self.budget_manager.get_total_consumed_budget(),
            optimization_stats,
            selection,
        })
    }
    /// Check early stopping criteria
    fn should_stop_early(&self, evaluations: &[HPOEvaluation<T>]) -> Result<bool> {
        if !self.config.early_stopping.enabled {
            return Ok(false);
        }
        if evaluations.len() < self.config.early_stopping.patience {
            return Ok(false);
        }
        let recent_scores: Vec<T> = evaluations
            .iter()
            .rev()
            .take(self.config.early_stopping.patience)
            .map(|eval| eval.result.objective_value)
            .collect();
        let best_recent =
            recent_scores
                .iter()
                .fold(T::neg_infinity(), |acc, &x| if x > acc { x } else { acc });
        let best_overall = evaluations
            .iter()
            .map(|eval| eval.result.objective_value)
            .fold(T::neg_infinity(), |acc, x| if x > acc { x } else { acc });
        let improvement = best_recent - best_overall;
        Ok(improvement
            < T::from(self.config.early_stopping.min_improvement).unwrap_or_else(|| T::zero()))
    }
    /// Compute optimization statistics from the run that just finished.
    ///
    /// The previous implementation returned all zeros regardless of what the
    /// run did, so every reported statistic was fabricated.
    fn compute_optimization_stats(
        &self,
        evaluations: &[HPOEvaluation<T>],
        durations: &[f64],
        failed_evaluations: usize,
        total_time: f64,
        convergence_iteration: Option<usize>,
    ) -> Result<OptimizationStats<T>> {
        let successful = evaluations
            .iter()
            .filter(|evaluation| matches!(evaluation.result.status, EvaluationStatus::Success))
            .count();
        let average_evaluation_time = if durations.is_empty() {
            0.0
        } else {
            durations.iter().sum::<f64>() / durations.len() as f64
        };

        // Budget efficiency: score improvement achieved per unit epsilon spent.
        let epsilon_spent = self.budget_manager.epsilon_spent();
        let budget_efficiency = if epsilon_spent > 0.0 && evaluations.len() >= 2 {
            let first = evaluations[0]
                .result
                .objective_value
                .to_f64()
                .unwrap_or(0.0);
            let best = evaluations
                .iter()
                .filter_map(|evaluation| evaluation.result.objective_value.to_f64())
                .fold(f64::NEG_INFINITY, f64::max);
            if best.is_finite() {
                (best - first) / epsilon_spent
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(OptimizationStats {
            total_evaluations: evaluations.len() + failed_evaluations,
            successful_evaluations: successful,
            failed_evaluations,
            average_evaluation_time,
            total_optimization_time: total_time,
            convergence_iteration,
            budget_efficiency,
            _phantom: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::ParameterConfiguration;
    use super::*;
    use crate::privacy::private_hyperparameter_optimization::selection::OBJECTIVE_SENSITIVITY_KEY;
    use crate::privacy::private_hyperparameter_optimization::types::{
        BudgetAllocationStrategy, EarlyStoppingConfig, ParameterBounds, ParameterDefinition,
        ParameterType, ParameterValue, SensitivityBounds, ValidationStrategy,
    };
    use crate::privacy::DifferentialPrivacyConfig;

    fn sensitivity_bounds(declared: Option<f64>) -> SensitivityBounds<f64> {
        let mut global_sensitivity = HashMap::new();
        if let Some(value) = declared {
            global_sensitivity.insert(OBJECTIVE_SENSITIVITY_KEY.to_string(), value);
        }
        SensitivityBounds {
            global_sensitivity,
            local_sensitivity: HashMap::new(),
            smooth_sensitivity: HashMap::new(),
        }
    }

    fn config(private_selection: bool, declared: Option<f64>) -> PrivateHPOConfig<f64> {
        PrivateHPOConfig {
            base_privacyconfig: DifferentialPrivacyConfig {
                target_epsilon: 4.0,
                ..DifferentialPrivacyConfig::default()
            },
            budget_allocation: BudgetAllocationStrategy::Equal,
            search_algorithm: SearchAlgorithm::RandomSearch,
            num_evaluations: 10,
            cv_folds: 3,
            early_stopping: EarlyStoppingConfig {
                enabled: false,
                patience: 3,
                min_improvement: 1e-4,
                max_evaluations: 10,
            },
            noise_mechanism: HyperparameterNoiseMechanism::Laplace,
            sensitivity_bounds: sensitivity_bounds(declared),
            private_model_selection: private_selection,
            validation_strategy: ValidationStrategy::HoldOut,
        }
    }

    fn space() -> ParameterSpace<f64> {
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
        ParameterSpace {
            parameters,
            constraints: Vec::new(),
            defaultconfig: None,
        }
    }

    /// A deterministic objective peaking at learning_rate = 0.75.
    fn objective() -> ObjectiveFn<f64> {
        Box::new(
            |config: &ParameterConfiguration<f64>| match config.values.get("learning_rate") {
                Some(ParameterValue::Continuous(rate)) => Ok(1.0 - (rate - 0.75).abs()),
                other => Err(crate::error::OptimError::InvalidParameter(format!(
                    "expected a continuous learning_rate, got {other:?}"
                ))),
            },
        )
    }

    fn learning_rate(config: &ParameterConfiguration<f64>) -> f64 {
        match config.values.get("learning_rate") {
            Some(ParameterValue::Continuous(rate)) => *rate,
            other => panic!("expected a continuous learning_rate, got {other:?}"),
        }
    }

    #[test]
    fn private_selection_requires_a_declared_objective_sensitivity() {
        // Guessing the sensitivity would silently invalidate the epsilon the
        // exponential mechanism is calibrated with.
        let outcome = PrivateHyperparameterOptimizer::new(config(true, None), space());
        let message = match outcome {
            Err(err) => err.to_string(),
            Ok(_) => panic!("an undeclared sensitivity must be refused"),
        };
        assert!(
            message.contains(OBJECTIVE_SENSITIVITY_KEY),
            "got: {message}"
        );
        assert!(
            PrivateHyperparameterOptimizer::new(config(true, Some(1.0)), space()).is_ok(),
            "a declared sensitivity must be accepted"
        );
        for bad in [0.0f64, -1.0, f64::NAN] {
            assert!(
                PrivateHyperparameterOptimizer::new(config(true, Some(bad)), space()).is_err(),
                "sensitivity {bad} must be refused"
            );
        }
    }

    #[test]
    fn an_empty_parameter_space_is_refused() {
        let empty = ParameterSpace {
            parameters: HashMap::new(),
            constraints: Vec::new(),
            defaultconfig: None,
        };
        // A declared sensitivity keeps this isolated to the empty-space check:
        // with `None` it would now also fail for the missing sensitivity.
        assert!(PrivateHyperparameterOptimizer::new(config(false, Some(1.0)), empty).is_err());
    }

    #[test]
    fn an_end_to_end_run_selects_privately_and_charges_for_it() {
        let mut optimizer =
            match PrivateHyperparameterOptimizer::new(config(true, Some(1.0)), space()) {
                Ok(optimizer) => optimizer,
                Err(err) => panic!("construction failed: {err}"),
            };
        optimizer.seed_for_tests(11);
        let results = match optimizer.optimize(objective()) {
            Ok(results) => results,
            Err(err) => panic!("optimize failed: {err}"),
        };

        assert_eq!(results.all_evaluations.len(), 10);
        assert!(results.bestconfiguration.is_some());
        assert!(
            results.selection.was_private,
            "the selection must be reported as private"
        );
        // The configured `noise_mechanism` drives the selection, so a Laplace
        // configuration selects by Laplace report-noisy-max.
        assert_eq!(results.selection.mechanism, "laplace_report_noisy_max");
        assert!(
            results.selection.epsilon_spent > 0.0,
            "the selection must cost epsilon, got {}",
            results.selection.epsilon_spent
        );
        assert_eq!(results.selection.utility_sensitivity, 1.0);
        match results.selection.selected_probability {
            Some(probability) => {
                assert!(
                    (0.0..=1.0).contains(&probability),
                    "probability {probability} is not a probability"
                );
            }
            None => panic!("the mechanism's own selection probability must be reported"),
        }

        // The whole budget is accounted for and never exceeded.
        let spent = results.total_privacy_cost.epsilon_consumed;
        assert!(
            spent > 0.0 && spent <= 4.0 + 1e-9,
            "spent {spent} of a 4.0 budget"
        );
        assert!(results.total_privacy_cost.epsilon_remaining >= 0.0);
        assert_eq!(results.total_privacy_cost.delta_consumed, 0.0);

        // The statistics are measured, not the fabricated zeros they used to be.
        assert_eq!(results.optimization_stats.total_evaluations, 10);
        assert_eq!(results.optimization_stats.successful_evaluations, 10);
        assert_eq!(results.optimization_stats.failed_evaluations, 0);
        assert!(results.optimization_stats.total_optimization_time > 0.0);
        assert!(results.optimization_stats.convergence_iteration.is_some());

        // The summary statistics are noisy, not a copy of the mean.
        assert!(results.final_results.summary_stats.noisy_std > 0.0);
        assert!(results.final_results.confidence_intervals.is_some());
        assert!(!results.final_results.topconfigurations.is_empty());
        assert!(results.final_results.model_selection.is_some());
    }

    #[test]
    fn the_reported_configuration_is_not_always_the_exact_argmax() {
        // The core regression: an exact argmax over utilities computed from the
        // private data leaks the selection. Across independent runs of the same
        // deterministic objective the reported configuration must sometimes
        // differ from the best evaluated one.
        let mut deviations = 0usize;
        for seed in 0..24u64 {
            let mut optimizer =
                match PrivateHyperparameterOptimizer::new(config(true, Some(1.0)), space()) {
                    Ok(optimizer) => optimizer,
                    Err(err) => panic!("construction failed: {err}"),
                };
            optimizer.seed_for_tests(seed);
            let results = match optimizer.optimize(objective()) {
                Ok(results) => results,
                Err(err) => panic!("optimize failed: {err}"),
            };
            let best_observed = results
                .all_evaluations
                .iter()
                .map(|evaluation| evaluation.result.objective_value)
                .fold(f64::NEG_INFINITY, f64::max);
            if (results.best_score - best_observed).abs() > 1e-12 {
                deviations += 1;
            }
        }
        // At epsilon/draw = 0.04 with Delta_u = 1 the mechanism is close to
        // uniform over the 10 candidates, so the argmax should be returned about
        // 1 run in 10; requiring at least half the runs to deviate fails with
        // probability well under 1e-6 under that model, and fails *always* if the
        // selection is the exact argmax.
        assert!(
            deviations >= 12,
            "only {deviations}/24 runs deviated from the exact argmax; the selection is not \
             behaving like a private mechanism"
        );
    }

    #[test]
    fn a_non_private_selection_is_reported_as_such() {
        let mut optimizer =
            match PrivateHyperparameterOptimizer::new(config(false, Some(1.0)), space()) {
                Ok(optimizer) => optimizer,
                Err(err) => panic!("construction failed: {err}"),
            };
        optimizer.seed_for_tests(5);
        let results = match optimizer.optimize(objective()) {
            Ok(results) => results,
            Err(err) => panic!("optimize failed: {err}"),
        };
        assert!(
            !results.selection.was_private,
            "an exact argmax must not be reported as private"
        );
        assert_eq!(results.selection.mechanism, "exact_argmax");
        assert_eq!(results.selection.epsilon_spent, 0.0);

        // With no private selection the argmax is exactly what is returned.
        let best_observed = results
            .all_evaluations
            .iter()
            .map(|evaluation| evaluation.result.objective_value)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((results.best_score - best_observed).abs() < 1e-12);
    }

    #[test]
    fn the_released_objectives_are_noisy_not_the_raw_values() {
        // `add_noise` used to fall through to `Ok(value)` for every mechanism
        // except Gaussian, and even then used a constant scale.
        let mut optimizer =
            match PrivateHyperparameterOptimizer::new(config(false, Some(1.0)), space()) {
                Ok(optimizer) => optimizer,
                Err(err) => panic!("construction failed: {err}"),
            };
        optimizer.seed_for_tests(3);
        let results = match optimizer.optimize(objective()) {
            Ok(results) => results,
            Err(err) => panic!("optimize failed: {err}"),
        };
        let mut noisy_count = 0usize;
        for evaluation in &results.all_evaluations {
            let rate = learning_rate(&evaluation.configuration);
            let exact = 1.0 - (rate - 0.75).abs();
            if (evaluation.result.objective_value - exact).abs() > 1e-9 {
                noisy_count += 1;
            }
            assert!(
                evaluation.result.standard_error.is_some(),
                "the release must report the noise scale that was applied"
            );
        }
        assert_eq!(
            noisy_count,
            results.all_evaluations.len(),
            "every released objective must be perturbed"
        );
    }

    #[test]
    fn a_failing_objective_is_counted_and_propagated_when_nothing_succeeds() {
        let mut optimizer =
            match PrivateHyperparameterOptimizer::new(config(false, Some(1.0)), space()) {
                Ok(optimizer) => optimizer,
                Err(err) => panic!("construction failed: {err}"),
            };
        let always_fails: ObjectiveFn<f64> = Box::new(|_| {
            Err(crate::error::OptimError::ComputationError(
                "the trial crashed".to_string(),
            ))
        });
        let message = match optimizer.optimize(always_fails) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("a run in which every trial failed must not succeed"),
        };
        assert!(message.contains("the trial crashed"), "got: {message}");
    }

    #[test]
    fn an_unset_objective_errors_instead_of_scoring_zero() {
        // `PrivateObjective::new` used to default to `|_| Ok(0.0)`.
        let mut objective: PrivateObjective<f64> = match PrivateObjective::new() {
            Ok(objective) => objective,
            Err(err) => panic!("construction failed: {err}"),
        };
        let config = ParameterConfiguration {
            values: HashMap::new(),
            id: "c".to_string(),
            metadata: HashMap::new(),
        };
        let budget = crate::privacy::PrivacyBudget {
            epsilon_consumed: 0.5,
            ..crate::privacy::PrivacyBudget::default()
        };
        let message = match objective.evaluate(&config, &budget) {
            Err(err) => err.to_string(),
            Ok(result) => panic!("an unset objective scored {:?}", result.objective_value),
        };
        assert!(message.contains("no objective function"), "got: {message}");
    }

    #[test]
    fn a_zero_epsilon_grant_is_refused_by_the_objective_release() {
        let mut objective: PrivateObjective<f64> = match PrivateObjective::new() {
            Ok(objective) => objective,
            Err(err) => panic!("construction failed: {err}"),
        };
        let ok = objective.set_objective(Box::new(|_| Ok(1.0)));
        assert!(ok.is_ok());
        let config = ParameterConfiguration {
            values: HashMap::new(),
            id: "c".to_string(),
            metadata: HashMap::new(),
        };
        for epsilon in [0.0f64, -1.0, f64::NAN] {
            let budget = crate::privacy::PrivacyBudget {
                epsilon_consumed: epsilon,
                ..crate::privacy::PrivacyBudget::default()
            };
            assert!(
                objective.evaluate(&config, &budget).is_err(),
                "an epsilon of {epsilon} must not buy a release"
            );
        }
    }

    #[test]
    fn the_configured_noise_mechanism_drives_the_selection() {
        for (mechanism, expected) in [
            (
                HyperparameterNoiseMechanism::Exponential,
                "exponential_mechanism",
            ),
            (
                HyperparameterNoiseMechanism::NoisyMax,
                "gumbel_report_noisy_max",
            ),
            (
                HyperparameterNoiseMechanism::Laplace,
                "laplace_report_noisy_max",
            ),
            (
                HyperparameterNoiseMechanism::Gaussian,
                "gaussian_report_noisy_max",
            ),
        ] {
            let mut hpo_config = config(true, Some(1.0));
            hpo_config.noise_mechanism = mechanism;
            let mut optimizer = match PrivateHyperparameterOptimizer::new(hpo_config, space()) {
                Ok(optimizer) => optimizer,
                Err(err) => panic!("{mechanism:?} construction failed: {err}"),
            };
            optimizer.seed_for_tests(23);
            let results = match optimizer.optimize(objective()) {
                Ok(results) => results,
                Err(err) => panic!("{mechanism:?} optimize failed: {err}"),
            };
            assert_eq!(results.selection.mechanism, expected);
            assert!(results.selection.was_private);
            assert!(results.selection.epsilon_spent > 0.0);
        }
    }

    #[test]
    fn the_bayesian_search_path_also_runs_end_to_end() {
        let mut hpo_config = config(true, Some(1.0));
        hpo_config.search_algorithm = SearchAlgorithm::BayesianOptimization;
        let mut optimizer = match PrivateHyperparameterOptimizer::new(hpo_config, space()) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        optimizer.seed_for_tests(19);
        let results = match optimizer.optimize(objective()) {
            Ok(results) => results,
            Err(err) => panic!("optimize failed: {err}"),
        };
        assert_eq!(results.all_evaluations.len(), 10);
        for evaluation in &results.all_evaluations {
            assert_eq!(
                evaluation.configuration.values.len(),
                1,
                "every Bayesian proposal must set the parameter"
            );
        }
        assert!(results.selection.was_private);
    }

    #[test]
    fn the_objective_release_also_requires_a_declared_sensitivity() {
        // Regression: the objective sensitivity used to be read with
        // `.unwrap_or_else(T::one)`, so an undeclared sensitivity silently became
        // 1.0 whenever `private_model_selection` was off. The objective's noise
        // scale is `sensitivity / epsilon`, so a true sensitivity of 8 would have
        // been noised eight times too weakly while the run still reported the
        // configured epsilon. The old code constructed happily here.
        let outcome = PrivateHyperparameterOptimizer::new(config(false, None), space());
        let message = match outcome {
            Err(err) => err.to_string(),
            Ok(_) => panic!("an undeclared objective sensitivity must be refused"),
        };
        assert!(
            message.contains(OBJECTIVE_SENSITIVITY_KEY),
            "the error must name the key the sensitivity is expected under, got: {message}"
        );
        assert!(
            PrivateHyperparameterOptimizer::new(config(false, Some(2.0)), space()).is_ok(),
            "a declared sensitivity must be accepted with private selection off"
        );
    }

    #[test]
    fn the_declared_sensitivity_reaches_the_objective_noise_scale() {
        // The scale recorded after a release must be the one the mechanism
        // actually used: `sensitivity / epsilon` for Laplace. Two runs differing
        // only in the declared sensitivity must record scales in that ratio, so a
        // constant substituted for the declaration cannot pass.
        let mut scales = Vec::new();
        for declared in [1.0f64, 4.0] {
            let mut optimizer =
                match PrivateHyperparameterOptimizer::new(config(false, Some(declared)), space()) {
                    Ok(optimizer) => optimizer,
                    Err(err) => panic!("construction failed for sensitivity {declared}: {err}"),
                };
            optimizer.seed_for_tests(7);
            if let Err(err) = optimizer.optimize(objective()) {
                panic!("optimize failed for sensitivity {declared}: {err}");
            }
            let params = optimizer
                .private_objective()
                .noise_mechanism()
                .noise_params();
            assert_eq!(params.sensitivity, declared);
            let epsilon = params.epsilon;
            assert!(epsilon > 0.0, "the release must have been charged epsilon");
            let expected = declared / epsilon;
            assert!(
                (params.scale - expected).abs() < 1e-12,
                "recorded scale {} is not sensitivity/epsilon = {expected}",
                params.scale
            );
            scales.push(params.scale);
        }
        assert!(
            (scales[1] / scales[0] - 4.0).abs() < 1e-9,
            "quadrupling the declared sensitivity must quadruple the noise scale, got {scales:?}"
        );
    }

    #[test]
    fn the_reported_privacy_cost_is_the_real_ledger_not_an_unstepped_accountant() {
        // Regression: `privacy_accountant()` handed out a `MomentsAccountant`
        // built from the DP-SGD parameters in `base_privacyconfig` and never
        // stepped, so it reported zero spend after a full search while the real
        // spend sat in `HPOBudgetManager`. The accessor now reads that ledger.
        let mut optimizer =
            match PrivateHyperparameterOptimizer::new(config(true, Some(1.0)), space()) {
                Ok(optimizer) => optimizer,
                Err(err) => panic!("construction failed: {err}"),
            };
        assert_eq!(
            optimizer.total_privacy_cost().epsilon_consumed,
            0.0,
            "nothing has been released yet"
        );
        optimizer.seed_for_tests(31);
        let results = match optimizer.optimize(objective()) {
            Ok(results) => results,
            Err(err) => panic!("optimize failed: {err}"),
        };

        let reported = optimizer.total_privacy_cost();
        assert!(
            reported.epsilon_consumed > 0.0,
            "a completed search must report a positive spend, got {}",
            reported.epsilon_consumed
        );
        assert!(
            (reported.epsilon_consumed - results.total_privacy_cost.epsilon_consumed).abs() < 1e-12,
            "the accessor and the results must read the same ledger: {} vs {}",
            reported.epsilon_consumed,
            results.total_privacy_cost.epsilon_consumed
        );
        assert!(
            reported.epsilon_consumed <= 4.0 + 1e-9,
            "the spend must not exceed the 4.0 target, got {}",
            reported.epsilon_consumed
        );
        // The private selection is part of that spend, so the ledger must cover
        // at least what the selection itself reports charging.
        assert!(
            reported.epsilon_consumed >= results.selection.epsilon_spent,
            "the ledger {} does not cover the selection's own charge {}",
            reported.epsilon_consumed,
            results.selection.epsilon_spent
        );
    }

    #[test]
    fn search_algorithms_without_a_private_implementation_are_refused() {
        // Regression: a `_ =>` arm mapped GridSearch, GeneticAlgorithm,
        // ParticleSwarm, SimulatedAnnealing and TPE onto `PrivateRandomSearch`,
        // so the configured algorithm never ran and the caller was never told.
        for algorithm in [
            SearchAlgorithm::GridSearch,
            SearchAlgorithm::GeneticAlgorithm,
            SearchAlgorithm::ParticleSwarm,
            SearchAlgorithm::SimulatedAnnealing,
            SearchAlgorithm::TPE,
        ] {
            let mut hpo_config = config(true, Some(1.0));
            hpo_config.search_algorithm = algorithm;
            let message = match PrivateHyperparameterOptimizer::new(hpo_config, space()) {
                Err(err) => err.to_string(),
                Ok(_) => panic!(
                    "{algorithm:?} has no private implementation and must not be substituted"
                ),
            };
            assert!(
                message.contains(&format!("{algorithm:?}")),
                "the error must name the refused algorithm, got: {message}"
            );
            assert!(
                optimizer_key(algorithm).is_err(),
                "{algorithm:?} must not resolve to an optimizer key"
            );
        }

        assert_eq!(
            optimizer_key(SearchAlgorithm::RandomSearch).ok(),
            Some("random_search")
        );
        assert_eq!(
            optimizer_key(SearchAlgorithm::BayesianOptimization).ok(),
            Some("bayesian_opt")
        );
    }
}
