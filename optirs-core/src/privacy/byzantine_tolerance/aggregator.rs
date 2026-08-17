//! [`ByzantineTolerantAggregator`] itself: construction, accessors, and the
//! `byzantine_robust_aggregate` orchestration pipeline (cohort validation,
//! reputation filtering, anomaly/outlier/verification detection, Byzantine
//! identification and the final dispatch into an aggregation algorithm).

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::types::{
    AnomalyDetector, AnomalyScore, BehaviorHistory, ByzantineAggregationMethod,
    ByzantineAggregationResult, ByzantineConfig, GradientProperties, GradientVerifier,
    OutlierScore, ReputationScore, StatisticalAnalysis, TrustLevel, VerificationScore,
};

/// Byzantine fault tolerant aggregator
pub struct ByzantineTolerantAggregator<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for Byzantine tolerance
    pub(super) config: ByzantineConfig,
    /// Participant reputation scores
    pub(super) reputation_scores: HashMap<String, ReputationScore>,
    /// History of participant behavior
    pub(super) behavior_history: HashMap<String, BehaviorHistory>,
    /// Anomaly detection engine
    pub(super) anomaly_detector: AnomalyDetector<T>,
    /// Statistical analysis engine
    pub(super) statistics_engine: StatisticalAnalysis<T>,
    /// Gradient verification system
    pub(super) gradient_verifier: GradientVerifier<T>,
    /// Per-participant sum of all historical updates, used by FoolsGold.
    pub(super) fools_gold_history: HashMap<String, Array1<T>>,
    /// Number of completed aggregation rounds (seeds FLAME's noise).
    pub(super) round: u64,
}

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Create new Byzantine tolerant aggregator.
    ///
    /// Returns an error when `config` is inconsistent (see
    /// [`ByzantineConfig::validate`]).
    pub fn new(config: ByzantineConfig) -> Result<Self> {
        config.validate()?;
        let anomaly_threshold = config.anomaly_threshold;
        Ok(Self {
            config,
            reputation_scores: HashMap::new(),
            behavior_history: HashMap::new(),
            anomaly_detector: AnomalyDetector::new(anomaly_threshold),
            statistics_engine: StatisticalAnalysis::new(),
            gradient_verifier: GradientVerifier::new(),
            fools_gold_history: HashMap::new(),
            round: 0,
        })
    }
    /// Configuration in use.
    pub fn config(&self) -> &ByzantineConfig {
        &self.config
    }
    /// Number of completed aggregation rounds.
    pub fn round(&self) -> u64 {
        self.round
    }
    /// Reputation record of a participant, if one exists.
    pub fn reputation(&self, participant_id: &str) -> Option<&ReputationScore> {
        self.reputation_scores.get(participant_id)
    }
    /// Recorded behaviour history of a participant, if one exists.
    pub fn behavior_history(&self, participant_id: &str) -> Option<&BehaviorHistory> {
        self.behavior_history.get(participant_id)
    }
    /// Replace the gradient properties enforced by the verification stage.
    pub fn set_gradient_properties(&mut self, properties: GradientProperties<T>) {
        let reference = self.gradient_verifier.reference_direction.clone();
        self.gradient_verifier = GradientVerifier::with_properties(properties);
        self.gradient_verifier.reference_direction = reference;
    }
    /// Perform Byzantine-robust aggregation.
    pub fn byzantine_robust_aggregate(
        &mut self,
        participant_gradients: &HashMap<String, Array1<T>>,
    ) -> Result<ByzantineAggregationResult<T>> {
        Self::validate_cohort(participant_gradients)?;
        let total_submitted = participant_gradients.len();
        let filtered_participants = self.filter_by_reputation(participant_gradients)?;
        if filtered_participants.is_empty() {
            return Err(OptimError::InvalidState(
                "every submitting participant is blacklisted".to_string(),
            ));
        }
        let anomaly_results = self.detect_anomalies(&filtered_participants)?;
        let outlier_results = self.detect_statistical_outliers(&filtered_participants)?;
        let verification_results = if self.config.gradient_verification {
            self.verify_gradients(&filtered_participants)?
        } else {
            HashMap::new()
        };
        let byzantine_participants = self.identify_byzantine_participants(
            &anomaly_results,
            &outlier_results,
            &verification_results,
        )?;
        let honest_participants =
            self.select_honest_participants(&filtered_participants, &byzantine_participants)?;
        let consensus_ratio = honest_participants.len() as f64 / total_submitted as f64;
        if consensus_ratio < self.config.consensus_threshold {
            return Err(OptimError::InvalidState(format!(
                "only {:.1}% of the cohort survived Byzantine detection, below the configured \
                 consensus threshold of {:.1}%",
                consensus_ratio * 100.0,
                self.config.consensus_threshold * 100.0
            )));
        }
        let aggregate = self.perform_robust_aggregation(&honest_participants)?;
        let confidence_score =
            self.calculate_confidence_score(&honest_participants, total_submitted, &aggregate)?;
        self.record_behavior(
            participant_gradients,
            &filtered_participants,
            &anomaly_results,
            &aggregate,
        )?;
        self.update_reputations(&honest_participants, &byzantine_participants)?;
        self.learn_patterns(
            &honest_participants,
            &filtered_participants,
            &byzantine_participants,
        )?;
        self.gradient_verifier
            .set_reference_direction(aggregate.clone());
        self.round = self.round.saturating_add(1);
        let mut honest_ids: Vec<String> = honest_participants.keys().cloned().collect();
        honest_ids.sort();
        Ok(ByzantineAggregationResult {
            aggregate,
            honest_participants: honest_ids,
            byzantine_participants,
            reputation_updates: self.get_reputation_updates(),
            aggregation_method: self.config.aggregation_method,
            consensus_ratio,
            confidence_score,
        })
    }
    /// Reject empty, ragged or non-finite cohorts.
    pub(super) fn validate_cohort(gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }
        let mut expected_dim: Option<usize> = None;
        for (participant_id, gradient) in Self::ordered_cohort(gradients) {
            if gradient.is_empty() {
                return Err(OptimError::InvalidConfig(format!(
                    "participant '{participant_id}' submitted an empty gradient"
                )));
            }
            match expected_dim {
                None => expected_dim = Some(gradient.len()),
                Some(dim) if dim != gradient.len() => {
                    return Err(OptimError::DimensionMismatch(format!(
                        "participant '{participant_id}' submitted a gradient of length {} \
                         while the cohort uses {dim}",
                        gradient.len()
                    )));
                }
                Some(_) => {}
            }
            if !gradient.iter().all(|x| x.is_finite()) {
                return Err(OptimError::InvalidParameter(format!(
                    "participant '{participant_id}' submitted a non-finite gradient"
                )));
            }
        }
        Ok(())
    }
    /// Cohort ordered by participant id, making every algorithm in this module
    /// independent of `HashMap` iteration order.
    pub(super) fn ordered_cohort(
        gradients: &HashMap<String, Array1<T>>,
    ) -> Vec<(&String, &Array1<T>)> {
        let mut items: Vec<(&String, &Array1<T>)> = gradients.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        items
    }
    /// Filter participants based on reputation scores
    pub(super) fn filter_by_reputation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut filtered = HashMap::new();
        for (participant_id, gradient) in gradients {
            match self.reputation_scores.get(participant_id) {
                Some(reputation) if reputation.trust_level == TrustLevel::Blacklisted => {}
                _ => {
                    filtered.insert(participant_id.clone(), gradient.clone());
                }
            }
        }
        Ok(filtered)
    }
    /// Detect anomalies in gradients
    pub(super) fn detect_anomalies(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, AnomalyScore>> {
        let ordered: Vec<(String, Array1<T>)> = Self::ordered_cohort(gradients)
            .into_iter()
            .map(|(id, g)| (id.clone(), g.clone()))
            .collect();
        let mut anomaly_results = HashMap::new();
        for (participant_id, gradient) in ordered {
            let anomaly_score = self
                .anomaly_detector
                .detect_anomaly(&participant_id, &gradient)?;
            anomaly_results.insert(participant_id, anomaly_score);
        }
        Ok(anomaly_results)
    }
    /// Detect statistical outliers across the current round's cohort.
    pub(super) fn detect_statistical_outliers(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, OutlierScore>> {
        let ordered = Self::ordered_cohort(gradients);
        let cohort: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let stats = self.statistics_engine.compute_statistics(&cohort)?;
        let mut outlier_results = HashMap::new();
        for (index, (participant_id, _)) in ordered.iter().enumerate() {
            let outlier_score = self.compute_outlier_score(index, &cohort, &stats)?;
            outlier_results.insert((*participant_id).clone(), outlier_score);
        }
        Ok(outlier_results)
    }
    /// Verify gradients using verification rules
    pub(super) fn verify_gradients(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, VerificationScore>> {
        let mut verification_results = HashMap::new();
        for (participant_id, gradient) in gradients {
            let verification_score = self.gradient_verifier.verify_gradient(gradient)?;
            verification_results.insert(participant_id.clone(), verification_score);
        }
        Ok(verification_results)
    }
    /// Identify Byzantine participants based on multiple criteria.
    ///
    /// Every participant whose combined score exceeds `anomaly_threshold` is
    /// reported; the list is deliberately not capped at `max_byzantine`, because a
    /// round in which more than `f` participants look Byzantine is exactly the case
    /// the caller must be told about (the downstream consensus gate then rejects it).
    pub(super) fn identify_byzantine_participants(
        &self,
        anomaly_results: &HashMap<String, AnomalyScore>,
        outlier_results: &HashMap<String, OutlierScore>,
        verification_results: &HashMap<String, VerificationScore>,
    ) -> Result<Vec<String>> {
        let mut participant_ids: Vec<&String> = anomaly_results.keys().collect();
        participant_ids.sort();
        let mut byzantine_participants = Vec::new();
        for participant_id in participant_ids {
            let anomaly_score = anomaly_results.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing anomaly score for '{participant_id}'"))
            })?;
            let outlier_score = outlier_results.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing outlier score for '{participant_id}'"))
            })?;
            let verification_score = verification_results.get(participant_id);
            let combined_score =
                self.compute_byzantine_score(anomaly_score, outlier_score, verification_score);
            if combined_score > self.config.anomaly_threshold {
                byzantine_participants.push(participant_id.clone());
            }
        }
        Ok(byzantine_participants)
    }
    /// Select honest participants for aggregation
    pub(super) fn select_honest_participants(
        &self,
        all_participants: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut honest_participants = HashMap::new();
        for (participant_id, gradient) in all_participants {
            if !byzantine_participants.contains(participant_id) {
                honest_participants.insert(participant_id.clone(), gradient.clone());
            }
        }
        if honest_participants.len() < self.config.min_participants {
            return Err(OptimError::InvalidState(format!(
                "only {} honest participants remain, but {} are required",
                honest_participants.len(),
                self.config.min_participants
            )));
        }
        Ok(honest_participants)
    }
    /// Perform robust aggregation using the configured method.
    ///
    /// `f` is always `config.max_byzantine`: the detection stages that ran before
    /// this point are heuristics, so their removals are not assumed sound and the
    /// configured tolerance is kept in full. `n` is the number of gradients handed
    /// to this call, i.e. the round's participants minus those filtered out by
    /// reputation and minus those flagged Byzantine.
    pub(super) fn perform_robust_aggregation(
        &mut self,
        honest_gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        let f = self.config.max_byzantine;
        match self.config.aggregation_method {
            ByzantineAggregationMethod::TrimmedMean => {
                self.trimmed_mean_aggregation(honest_gradients, f)
            }
            ByzantineAggregationMethod::CoordinateMedian => {
                self.coordinate_median_aggregation(honest_gradients)
            }
            ByzantineAggregationMethod::Krum => self.krum_aggregation(honest_gradients, f),
            ByzantineAggregationMethod::MultiKrum => {
                self.multi_krum_aggregation(honest_gradients, f)
            }
            ByzantineAggregationMethod::Bulyan => self.bulyan_aggregation(honest_gradients, f),
            ByzantineAggregationMethod::FoolsGold => self.fools_gold_aggregation(honest_gradients),
            ByzantineAggregationMethod::FLAME => self.flame_aggregation(honest_gradients),
            ByzantineAggregationMethod::Median => self.median_aggregation(honest_gradients),
            ByzantineAggregationMethod::GeometricMedian => {
                self.geometric_median_aggregation(honest_gradients)
            }
        }
    }
}
