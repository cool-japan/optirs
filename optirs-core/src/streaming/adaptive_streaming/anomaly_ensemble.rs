// Ensemble voting for the streaming anomaly detector.
//
// Each strategy computes the quantity it names. Two of them used to fall
// through to a `_` arm and silently behave as plain min-consensus voting; that
// is gone. `Adaptive` is now backed by real per-detector confusion counters fed
// from the ground-truth feedback path, and `Stacking` — the one strategy that
// genuinely needs a trained second-stage model — returns an honest error.

use super::anomaly_detection::{AnomalyDetectionResult, AnomalySeverity, AnomalyType};
use super::anomaly_scoring::DetectionCounters;

use crate::utils::try_scalar_str;
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Ensemble anomaly detector combining multiple methods
pub struct EnsembleAnomalyDetector<A: Float + Send + Sync> {
    /// Ensemble voting strategy
    voting_strategy: EnsembleVotingStrategy,
    /// Per-detector weights used by
    /// [`EnsembleVotingStrategy::Weighted`]. Detectors with no explicit weight
    /// count as `1`, so an unconfigured ensemble weights every detector
    /// equally rather than ignoring them.
    detector_weights: HashMap<String, A>,
    /// Per-detector confusion counters, the measurement
    /// [`EnsembleVotingStrategy::Adaptive`] derives its weights from.
    ///
    /// These are the ensemble's own counters, not the ML detectors': the ML
    /// detectors are told the *ensemble's* verdict when ground truth arrives
    /// (that is what "the outcome of the most recent detection" means to them),
    /// and the statistical detectors carry no counters at all. Weighting a vote
    /// needs each member's own verdict scored against the truth, which is
    /// exactly what these record.
    detector_counters: HashMap<String, DetectionCounters>,
    /// Each member's verdict on the most recently combined point, awaiting
    /// ground truth. Cleared once labelled, so one detection is scored once.
    pending_verdicts: HashMap<String, bool>,
    /// Ensemble configuration
    ensemble_config: EnsembleConfig<A>,
}

/// Ensemble voting strategies
#[derive(Debug, Clone)]
pub enum EnsembleVotingStrategy {
    /// Simple majority voting
    Majority,
    /// Weighted voting using the operator-supplied per-detector weights
    Weighted,
    /// Maximum anomaly score
    MaxScore,
    /// Average anomaly score
    AverageScore,
    /// Median anomaly score
    MedianScore,
    /// Weighted voting whose weights are *measured* from each detector's own
    /// confusion matrix (balanced accuracy), rather than configured.
    Adaptive,
    /// Stacking with meta-learner
    Stacking,
}

/// Ensemble configuration
#[derive(Debug, Clone)]
pub struct EnsembleConfig<A: Float + Send + Sync> {
    /// Minimum number of detectors that must agree
    pub min_consensus: usize,
    /// Threshold for ensemble anomaly score
    pub ensemble_threshold: A,
    /// Enable dynamic detector weighting
    pub dynamic_weighting: bool,
    /// Performance evaluation window
    pub evaluation_window: usize,
    /// Enable detector selection based on context
    pub context_based_selection: bool,
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> EnsembleAnomalyDetector<A> {
    pub(super) fn new(voting_strategy: EnsembleVotingStrategy) -> Result<Self, String> {
        Ok(Self {
            voting_strategy,
            detector_weights: HashMap::new(),
            detector_counters: HashMap::new(),
            pending_verdicts: HashMap::new(),
            ensemble_config: EnsembleConfig {
                min_consensus: 2,
                ensemble_threshold: try_scalar_str::<A, _>(0.5)?,
                dynamic_weighting: true,
                evaluation_window: 100,
                context_based_selection: false,
            },
        })
    }

    /// Replaces the voting strategy.
    pub(super) fn set_voting_strategy(&mut self, strategy: EnsembleVotingStrategy) {
        self.voting_strategy = strategy;
    }

    /// Strategy currently in force.
    pub fn voting_strategy(&self) -> &EnsembleVotingStrategy {
        &self.voting_strategy
    }

    /// Sets the weight [`EnsembleVotingStrategy::Weighted`] gives one detector.
    pub(super) fn set_detector_weight(&mut self, detector_name: &str, weight: A) {
        self.detector_weights
            .insert(detector_name.to_string(), weight);
    }

    /// Records ground truth for the most recently combined point against every
    /// member's own verdict.
    ///
    /// This is the ground-truth feedback path the adaptive weighting needs: a
    /// detector that flagged the point is scored as a true or false positive,
    /// one that did not is scored as a true or false negative.
    pub(super) fn record_outcome(&mut self, was_true_anomaly: bool) {
        for (name, verdict) in self.pending_verdicts.drain() {
            self.detector_counters
                .entry(name)
                .or_default()
                .record_outcome(verdict, was_true_anomaly);
        }
    }

    /// Balanced accuracy measured for one member, or `None` before any ground
    /// truth has been recorded for it.
    pub fn detector_balanced_accuracy(&self, detector_name: &str) -> Option<f64> {
        self.detector_counters
            .get(detector_name)?
            .balanced_accuracy()
    }

    /// Weights derived from the measured per-detector confusion matrices, or
    /// `None` while no member has any labelled outcome.
    ///
    /// The weight is balanced accuracy `(TPR + TNR) / 2`, which is insensitive
    /// to how rare anomalies are; plain accuracy would hand a detector that
    /// flags nothing on a 1%-anomaly stream a weight of 0.99.
    pub fn measured_weights(&self) -> Option<HashMap<String, A>> {
        let mut weights = HashMap::new();
        let mut measured_any = false;
        for (name, counters) in &self.detector_counters {
            if let Some(balanced) = counters.balanced_accuracy() {
                measured_any = true;
                if let Some(weight) = A::from(balanced) {
                    weights.insert(name.clone(), weight);
                }
            }
        }
        measured_any.then_some(weights)
    }

    /// Weighted vote and weighted mean score under an arbitrary weighting.
    ///
    /// Shared by [`EnsembleVotingStrategy::Weighted`] (configured weights) and
    /// [`EnsembleVotingStrategy::Adaptive`] (measured weights), so the two
    /// cannot drift apart.
    fn weighted_vote(
        results: &HashMap<String, AnomalyDetectionResult<A>>,
        weight_of: &dyn Fn(&str) -> A,
    ) -> Result<(bool, A), String> {
        let mut weight_sum = A::zero();
        let mut weighted_score = A::zero();
        let mut weighted_votes = A::zero();
        for (name, result) in results {
            let weight = weight_of(name);
            weight_sum = weight_sum + weight;
            weighted_score = weighted_score + weight * result.anomaly_score;
            if result.is_anomaly {
                weighted_votes = weighted_votes + weight;
            }
        }
        if weight_sum <= A::zero() {
            return Err(
                "weighted ensemble voting needs a positive total detector weight".to_string(),
            );
        }
        let score = weighted_score / weight_sum;
        let vote_share = weighted_votes / weight_sum;
        Ok((vote_share > try_scalar_str::<A, _>(0.5)?, score))
    }

    pub(super) fn combine_results(
        &mut self,
        results: HashMap<String, AnomalyDetectionResult<A>>,
    ) -> Result<AnomalyDetectionResult<A>, String> {
        if results.is_empty() {
            return Ok(AnomalyDetectionResult {
                is_anomaly: false,
                anomaly_score: A::zero(),
                confidence: A::zero(),
                anomaly_type: None,
                severity: AnomalySeverity::Low,
                metadata: HashMap::new(),
            });
        }

        // Record each member's own verdict, so that a later
        // `record_outcome` can score it against the truth. This is the only
        // place the ensemble sees the individual verdicts.
        self.pending_verdicts.clear();
        for (name, result) in &results {
            self.detector_counters
                .entry(name.clone())
                .or_default()
                .record_prediction(result.is_anomaly, result.anomaly_score);
            self.pending_verdicts
                .insert(name.clone(), result.is_anomaly);
        }

        let anomaly_count = results.values().filter(|r| r.is_anomaly).count();
        let total_count = results.len();

        let avg_score = results.values().map(|r| r.anomaly_score).sum::<A>()
            / try_scalar_str::<A, _>(total_count)?;
        let avg_confidence = results.values().map(|r| r.confidence).sum::<A>()
            / try_scalar_str::<A, _>(total_count)?;

        // The ensemble the detector actually builds is `Weighted`, which used
        // to fall through to the `_` arm and behave as plain min-consensus
        // voting -- the weights were never consulted at all. Each strategy now
        // computes the quantity it names, and the one that has no
        // implementation behind it says so instead of silently pretending to
        // be a different strategy.
        let threshold = self.ensemble_config.ensemble_threshold;
        let (is_anomaly, ensemble_score) = match self.voting_strategy {
            EnsembleVotingStrategy::Majority => (anomaly_count > total_count / 2, avg_score),
            EnsembleVotingStrategy::MaxScore => {
                let max_score = results
                    .values()
                    .map(|r| r.anomaly_score)
                    .fold(A::zero(), |acc, s| if s > acc { s } else { acc });
                (max_score > threshold, max_score)
            }
            EnsembleVotingStrategy::AverageScore => (avg_score > threshold, avg_score),
            EnsembleVotingStrategy::MedianScore => {
                let mut scores: Vec<A> = results.values().map(|r| r.anomaly_score).collect();
                scores.sort_by(crate::utils::total_order);
                let median = if scores.len().is_multiple_of(2) {
                    (scores[scores.len() / 2 - 1] + scores[scores.len() / 2])
                        / try_scalar_str::<A, _>(2.0)?
                } else {
                    scores[scores.len() / 2]
                };
                (median > threshold, median)
            }
            EnsembleVotingStrategy::Weighted => {
                let one = A::one();
                let weights = &self.detector_weights;
                Self::weighted_vote(&results, &|name| *weights.get(name).unwrap_or(&one))?
            }
            EnsembleVotingStrategy::Adaptive => {
                // Measured weights when ground truth has arrived; uniform
                // otherwise. Uniform is the honest default: before any label
                // exists there is no evidence that one member deserves more say
                // than another, and inventing a prior ranking would be a
                // fabrication dressed as a measurement.
                match self.measured_weights() {
                    Some(measured) if measured.values().any(|w| *w > A::zero()) => {
                        Self::weighted_vote(&results, &|name| {
                            measured.get(name).copied().unwrap_or_else(A::zero)
                        })?
                    }
                    // Either no ground truth yet, or every member has measured
                    // zero skill — in both cases there is nothing to rank the
                    // members by, so the vote stays uniform.
                    _ => Self::weighted_vote(&results, &|_| A::one())?,
                }
            }
            EnsembleVotingStrategy::Stacking => {
                return Err(
                    "ensemble voting strategy `Stacking` is not implemented: stacking \
                     feeds the member scores into a second-stage model trained on held-out \
                     labelled data, and this detector carries no meta-learner and no \
                     held-out split to train one on. Use `Adaptive`, which weights the \
                     members by their measured balanced accuracy from the same feedback \
                     path"
                        .to_string(),
                );
            }
        };
        let avg_score = ensemble_score;

        Ok(AnomalyDetectionResult {
            is_anomaly,
            anomaly_score: avg_score,
            confidence: avg_confidence,
            anomaly_type: if is_anomaly {
                Some(AnomalyType::StatisticalOutlier)
            } else {
                None
            },
            severity: if avg_score > try_scalar_str::<A, _>(0.8)? {
                AnomalySeverity::High
            } else if avg_score > try_scalar_str::<A, _>(0.5)? {
                AnomalySeverity::Medium
            } else {
                AnomalySeverity::Low
            },
            metadata: HashMap::new(),
        })
    }

    pub(super) fn adjust_sensitivity(&mut self, adjustment: A) -> Result<(), String> {
        self.ensemble_config.ensemble_threshold = (self.ensemble_config.ensemble_threshold
            + adjustment)
            .max(try_scalar_str::<A, _>(0.1)?)
            .min(try_scalar_str::<A, _>(0.9)?);
        Ok(())
    }
}

#[cfg(test)]
#[path = "anomaly_ensemble_tests.rs"]
mod anomaly_ensemble_tests;
