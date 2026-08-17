//! Private aggregation and reporting of hyperparameter search results.
//!
//! Extracted from `types.rs` to keep every file under the 2000-line limit. See
//! [`PrivateResultsAggregator`] for the non-private aggregation it replaces.

use crate::error::{OptimError, Result};
use crate::privacy::PrivacyBudget;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{
    AggregatedResults, HPOEvaluation, HyperparameterNoiseMechanism, ModelSelectionResults,
    ResultAggregationStrategy, ResultValidator, SelectionMechanism, SelectionParameters,
};

/// Number of configurations reported in the private top-k.
pub const PRIVATE_TOP_K: usize = 5;

/// Report of how the final configuration was chosen.
#[derive(Debug, Clone)]
pub struct SelectionReport {
    /// Whether a differentially private mechanism produced the choice.
    ///
    /// `false` means the exact argmax was returned, which leaks the selection.
    /// It happens only when `PrivateHPOConfig::private_model_selection` is off,
    /// and is recorded here so a caller cannot mistake the result for private.
    pub was_private: bool,
    /// Name of the mechanism used, or `"exact_argmax"`.
    pub mechanism: String,
    /// Epsilon charged for the selection.
    pub epsilon_spent: f64,
    /// Delta charged for the selection.
    pub delta_spent: f64,
    /// Utility sensitivity the mechanism was calibrated with.
    pub utility_sensitivity: f64,
    /// Probability the mechanism assigned to the configuration it returned.
    pub selected_probability: Option<f64>,
}

/// Private results aggregator.
///
/// # The defect this replaces
///
/// `aggregate_results` sorted the evaluations **exactly** and returned the exact
/// top five, then reported `noisy_std: T::zero()` and `noisy_median: mean` --
/// neither noisy nor a median. The selection budget it carried was never spent
/// and `model_selection` was always `None`.
pub struct PrivateResultsAggregator<T: Float + Debug + Send + Sync + 'static> {
    /// Aggregation strategy
    aggregation_strategy: ResultAggregationStrategy,
    /// Privacy budget available to the selection, and the record of what it spent
    selection_budget: PrivacyBudget,
    /// Selection mechanism
    selection_mechanism: SelectionMechanism<T>,
    /// Result validation
    result_validator: ResultValidator<T>,
    /// Public a-priori range of a single objective value, used to calibrate the
    /// noisy summary statistics
    objective_range: f64,
    /// Epsilon split between the top-k selection and the summary statistics
    summary_epsilon_fraction: f64,
}

impl<T: Float + Debug + Send + Sync + 'static> PrivateResultsAggregator<T> {
    /// An aggregator with a 0.1 selection epsilon over a unit objective range.
    pub fn new() -> Result<Self> {
        Self::with_selection_budget(
            0.1,
            HyperparameterNoiseMechanism::Exponential,
            T::one(),
            1.0,
        )
    }

    /// An aggregator with an explicit selection budget and mechanism.
    ///
    /// `objective_range` is the *public* a-priori range of a single objective
    /// value (for example 1.0 for an accuracy in `[0, 1]`); it is what the
    /// summary statistics' sensitivity is derived from.
    pub fn with_selection_budget(
        epsilon: f64,
        mechanism_type: HyperparameterNoiseMechanism,
        utility_sensitivity: T,
        objective_range: f64,
    ) -> Result<Self> {
        if !objective_range.is_finite() || objective_range <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the public objective range must be positive and finite, got {objective_range}"
            )));
        }
        let mut selection_mechanism = SelectionMechanism::new();
        selection_mechanism.set_mechanism_type(mechanism_type);
        let mut params = SelectionParameters::pure_epsilon(epsilon, utility_sensitivity);
        if matches!(mechanism_type, HyperparameterNoiseMechanism::Gaussian) {
            params.delta = Some(1e-6);
        }
        selection_mechanism.set_selection_parameters(params)?;

        Ok(Self {
            aggregation_strategy: ResultAggregationStrategy::SelectBest,
            selection_budget: PrivacyBudget {
                epsilon_consumed: 0.0,
                delta_consumed: 0.0,
                epsilon_remaining: epsilon,
                delta_remaining: 0.0,
                steps_taken: 0,
                accounting_method: crate::privacy::AccountingMethod::RenyiDP,
                estimated_steps_remaining: 1,
            },
            selection_mechanism,
            result_validator: ResultValidator::new(),
            objective_range,
            summary_epsilon_fraction: 0.5,
        })
    }

    /// Seed every stochastic component deterministically (tests only).
    pub fn seed_for_tests(&mut self, seed: u64) {
        self.selection_mechanism.seed_for_tests(seed);
    }

    /// The aggregation strategy.
    pub fn aggregation_strategy(&self) -> ResultAggregationStrategy {
        self.aggregation_strategy
    }

    /// Replace the aggregation strategy.
    pub fn set_aggregation_strategy(&mut self, strategy: ResultAggregationStrategy) {
        self.aggregation_strategy = strategy;
    }

    /// Read-only access to the selection mechanism.
    pub fn selection_mechanism(&self) -> &SelectionMechanism<T> {
        &self.selection_mechanism
    }

    /// The selection budget and what it has spent.
    pub fn selection_budget(&self) -> &PrivacyBudget {
        &self.selection_budget
    }

    /// Read-only access to the result validator.
    pub fn result_validator(&self) -> &ResultValidator<T> {
        &self.result_validator
    }

    /// Aggregate the evaluations, selecting the reported configurations with a
    /// differentially private mechanism.
    ///
    /// The selection epsilon is split: half over `PRIVATE_TOP_K` sequential
    /// selections without replacement (basic composition), half over the noisy
    /// summary statistics.
    pub fn aggregate_results(
        &mut self,
        evaluations: &[HPOEvaluation<T>],
    ) -> Result<AggregatedResults<T>> {
        if evaluations.is_empty() {
            return Err(OptimError::InvalidParameter(
                "there are no evaluations to aggregate".to_string(),
            ));
        }

        let objective_values: Vec<T> = evaluations
            .iter()
            .map(|eval| eval.result.objective_value)
            .collect();
        let utilities: Vec<T> = objective_values
            .iter()
            .map(|value| self.selection_mechanism.utility_function().evaluate(*value))
            .collect::<Result<Vec<T>>>()?;

        let total_epsilon = self.selection_budget.epsilon_remaining;
        if !total_epsilon.is_finite() || total_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the aggregator was given a selection epsilon of {total_epsilon}"
            )));
        }
        let summary_epsilon = total_epsilon * self.summary_epsilon_fraction;
        let selection_epsilon = total_epsilon - summary_epsilon;

        // Private top-k without replacement: k sequential exponential-mechanism
        // draws, each at epsilon / k, composed linearly.
        let k = PRIVATE_TOP_K.min(evaluations.len());
        let per_draw_epsilon = selection_epsilon / k as f64;
        let mut params = self.selection_mechanism.selection_params().clone();
        params.epsilon = per_draw_epsilon;
        self.selection_mechanism.set_selection_parameters(params)?;

        let sensitivity = self
            .selection_mechanism
            .selection_params()
            .utility_sensitivity
            .to_f64()
            .unwrap_or(1.0);
        let probabilities = super::selection::exponential_mechanism_probabilities(
            &utilities
                .iter()
                .map(|utility| utility.to_f64().unwrap_or(f64::NEG_INFINITY))
                .collect::<Vec<f64>>(),
            sensitivity,
            per_draw_epsilon,
        )?;

        let mut remaining: Vec<usize> = (0..evaluations.len()).collect();
        let mut topconfigurations = Vec::with_capacity(k);
        let mut first_probability = None;
        for _ in 0..k {
            let candidate_utilities: Vec<T> =
                remaining.iter().map(|index| utilities[*index]).collect();
            let outcome = self
                .selection_mechanism
                .select_index(&candidate_utilities)?;
            let chosen = remaining.remove(outcome.index);
            if first_probability.is_none() {
                first_probability = probabilities.get(chosen).copied();
            }
            self.selection_budget.epsilon_consumed += outcome.epsilon_spent;
            self.selection_budget.epsilon_remaining =
                (self.selection_budget.epsilon_remaining - outcome.epsilon_spent).max(0.0);
            self.selection_budget.steps_taken += 1;
            topconfigurations.push((
                evaluations[chosen].configuration.clone(),
                evaluations[chosen].result.objective_value,
            ));
        }

        // Noisy summary statistics over the observed objectives. The summary
        // reports the epsilon it actually consumed, which is what gets charged.
        let summary = super::selection::noisy_summary_statistics(
            &objective_values,
            self.objective_range,
            summary_epsilon,
            self.selection_mechanism.rng_mut(),
        )?;
        let summary_stats = summary.statistics;
        self.selection_budget.epsilon_consumed += summary.epsilon_spent;
        self.selection_budget.epsilon_remaining =
            (self.selection_budget.epsilon_remaining - summary.epsilon_spent).max(0.0);

        // Confidence interval for the released mean, accounting for both the
        // sampling error and the Laplace noise that was added to it.
        let mean_noise_scale = summary.mean_noise_scale;
        let sample_std = summary_stats.noisy_std.to_f64().unwrap_or(0.0);
        let count = objective_values.len() as f64;
        let combined_std =
            (sample_std * sample_std / count + 2.0 * mean_noise_scale * mean_noise_scale).sqrt();
        let noisy_mean = summary_stats.noisy_mean.to_f64().unwrap_or(0.0);
        let confidence_intervals = match (
            T::from(noisy_mean - 1.96 * combined_std),
            T::from(noisy_mean + 1.96 * combined_std),
        ) {
            (Some(low), Some(high)) => Some((low, high)),
            _ => None,
        };

        let model_selection = topconfigurations.first().map(|(config, _)| {
            ModelSelectionResults {
                selectedconfig: config.clone(),
                // The mechanism's own probability of returning this
                // configuration -- a real number, not a placeholder.
                selection_confidence: first_probability.unwrap_or(0.0),
                alternatives: topconfigurations
                    .iter()
                    .skip(1)
                    .map(|(config, _)| config.clone())
                    .collect(),
            }
        });

        Ok(AggregatedResults {
            topconfigurations,
            confidence_intervals,
            summary_stats,
            model_selection,
        })
    }

    /// A report of how the final selection was made.
    pub fn selection_report(&self) -> SelectionReport {
        SelectionReport {
            was_private: true,
            mechanism: super::selection::mechanism_name(self.selection_mechanism.mechanism_type())
                .to_string(),
            epsilon_spent: self.selection_mechanism.epsilon_spent(),
            delta_spent: self.selection_mechanism.delta_spent(),
            utility_sensitivity: self
                .selection_mechanism
                .selection_params()
                .utility_sensitivity
                .to_f64()
                .unwrap_or(f64::NAN),
            selected_probability: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::private_hyperparameter_optimization::types::{
        EvaluationStatus, HPOResult, ParameterConfiguration, ParameterValue,
    };
    use std::collections::HashMap;

    fn evaluation(index: usize, objective: f64) -> HPOEvaluation<f64> {
        let mut values = HashMap::new();
        values.insert(
            "learning_rate".to_string(),
            ParameterValue::Continuous(index as f64 / 10.0),
        );
        HPOEvaluation {
            id: format!("eval_{index}"),
            configuration: ParameterConfiguration {
                values,
                id: format!("config_{index}"),
                metadata: HashMap::new(),
            },
            result: HPOResult {
                objective_value: objective,
                standard_error: Some(0.01),
                cv_scores: None,
                training_time: None,
                complexity_metrics: HashMap::new(),
                additional_metrics: HashMap::new(),
                status: EvaluationStatus::Success,
            },
            privacy_cost: PrivacyBudget::default(),
            timestamp: index as u64,
            metadata: HashMap::new(),
        }
    }

    fn evaluations() -> Vec<HPOEvaluation<f64>> {
        (0..10)
            .map(|index| evaluation(index, index as f64 / 10.0))
            .collect()
    }

    fn aggregator(epsilon: f64, seed: u64) -> PrivateResultsAggregator<f64> {
        let mut aggregator = match PrivateResultsAggregator::with_selection_budget(
            epsilon,
            HyperparameterNoiseMechanism::Exponential,
            1.0,
            1.0,
        ) {
            Ok(aggregator) => aggregator,
            Err(err) => panic!("construction failed: {err}"),
        };
        aggregator.seed_for_tests(seed);
        aggregator
    }

    #[test]
    fn the_top_k_is_not_the_exact_descending_sort() {
        // Regression: `aggregate_results` used to sort the evaluations exactly
        // and take the first five, which leaks the ranking.
        let exact_top: Vec<String> = {
            let mut sorted = evaluations();
            sorted.sort_by(|left, right| {
                right
                    .result
                    .objective_value
                    .partial_cmp(&left.result.objective_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted
                .iter()
                .take(PRIVATE_TOP_K)
                .map(|evaluation| evaluation.configuration.id.clone())
                .collect()
        };

        let mut deviations = 0usize;
        for seed in 0..16u64 {
            let mut aggregator = aggregator(0.5, seed);
            let results = match aggregator.aggregate_results(&evaluations()) {
                Ok(results) => results,
                Err(err) => panic!("aggregation failed: {err}"),
            };
            assert_eq!(results.topconfigurations.len(), PRIVATE_TOP_K);
            let reported: Vec<String> = results
                .topconfigurations
                .iter()
                .map(|(config, _)| config.id.clone())
                .collect();
            // No configuration may appear twice: the draws are without
            // replacement.
            let unique: std::collections::BTreeSet<&String> = reported.iter().collect();
            assert_eq!(unique.len(), reported.len(), "a duplicate was reported");
            if reported != exact_top {
                deviations += 1;
            }
        }
        assert!(
            deviations > 0,
            "16 aggregations all reproduced the exact descending sort"
        );
    }

    #[test]
    fn the_summary_statistics_are_noisy_and_distinct() {
        let mut aggregator = aggregator(2.0, 4);
        let results = match aggregator.aggregate_results(&evaluations()) {
            Ok(results) => results,
            Err(err) => panic!("aggregation failed: {err}"),
        };
        // `noisy_std: T::zero()` and `noisy_median: mean` were literals.
        assert!(
            results.summary_stats.noisy_std > 0.0,
            "the standard deviation must be measured, got {}",
            results.summary_stats.noisy_std
        );
        assert_ne!(
            results.summary_stats.noisy_median, results.summary_stats.noisy_mean,
            "the median must not be a copy of the mean"
        );
        assert_eq!(results.summary_stats.noisy_quantiles.len(), 3);
        let true_mean = 0.45f64;
        assert!(
            (results.summary_stats.noisy_mean - true_mean).abs() < 0.5,
            "noisy mean {} is implausible",
            results.summary_stats.noisy_mean
        );
    }

    #[test]
    fn the_confidence_interval_widens_with_the_noise() {
        let width = |epsilon: f64| -> f64 {
            let mut aggregator = aggregator(epsilon, 8);
            let results = match aggregator.aggregate_results(&evaluations()) {
                Ok(results) => results,
                Err(err) => panic!("aggregation failed: {err}"),
            };
            match results.confidence_intervals {
                Some((low, high)) => high - low,
                None => panic!("a confidence interval must be reported"),
            }
        };
        let tight = width(8.0);
        let loose = width(0.2);
        assert!(
            loose > tight,
            "a smaller epsilon must widen the interval: {loose} vs {tight}"
        );
    }

    #[test]
    fn the_selection_budget_is_charged_and_reported() {
        let mut aggregator = aggregator(1.0, 2);
        assert_eq!(aggregator.selection_budget().epsilon_consumed, 0.0);
        let _ = match aggregator.aggregate_results(&evaluations()) {
            Ok(results) => results,
            Err(err) => panic!("aggregation failed: {err}"),
        };
        let budget = aggregator.selection_budget();
        assert!(
            (budget.epsilon_consumed - 1.0).abs() < 1e-9,
            "the whole selection epsilon must be charged, got {}",
            budget.epsilon_consumed
        );
        assert!(budget.epsilon_remaining < 1e-9);
        assert_eq!(budget.steps_taken, PRIVATE_TOP_K);

        let report = aggregator.selection_report();
        assert!(report.was_private);
        assert_eq!(report.mechanism, "exponential_mechanism");
        assert!(report.epsilon_spent > 0.0);
    }

    #[test]
    fn the_model_selection_reports_a_real_probability() {
        let mut aggregator = aggregator(1.0, 6);
        let results = match aggregator.aggregate_results(&evaluations()) {
            Ok(results) => results,
            Err(err) => panic!("aggregation failed: {err}"),
        };
        let selection = match results.model_selection {
            Some(selection) => selection,
            None => panic!("model selection must be reported"),
        };
        assert!(
            (0.0..=1.0).contains(&selection.selection_confidence),
            "confidence {} is not a probability",
            selection.selection_confidence
        );
        assert!(
            selection.selection_confidence > 0.0,
            "the mechanism assigned zero probability to its own choice"
        );
        assert_eq!(selection.alternatives.len(), PRIVATE_TOP_K - 1);
    }

    #[test]
    fn aggregating_nothing_is_an_error() {
        let mut aggregator = aggregator(1.0, 1);
        assert!(aggregator.aggregate_results(&[]).is_err());
    }

    #[test]
    fn an_invalid_objective_range_is_refused() {
        for range in [0.0f64, -1.0, f64::NAN] {
            assert!(
                PrivateResultsAggregator::<f64>::with_selection_budget(
                    1.0,
                    HyperparameterNoiseMechanism::Exponential,
                    1.0,
                    range
                )
                .is_err(),
                "range {range} must be refused"
            );
        }
    }

    #[test]
    fn fewer_evaluations_than_k_still_aggregates() {
        let mut aggregator = aggregator(1.0, 3);
        let two = vec![evaluation(0, 0.1), evaluation(1, 0.9)];
        let results = match aggregator.aggregate_results(&two) {
            Ok(results) => results,
            Err(err) => panic!("aggregation failed: {err}"),
        };
        assert_eq!(results.topconfigurations.len(), 2);
    }
}
