//! Scoring, reputation and behaviour tracking

use crate::error::Result;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

use super::aggregator::ByzantineTolerantAggregator;
use super::helpers::{cosine_similarity, euclidean_distance, from_scalar, l2_norm};
use super::types::{
    AnomalyScore, BehaviorHistory, OutlierScore, ReputationScore, TrustLevel, VerificationScore,
};

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Combined Byzantine score in `[0, 1]`; every input is already bounded to
    /// `[0, 1]`, so the weights below are a genuine convex combination.
    pub(super) fn compute_byzantine_score(
        &self,
        anomaly_score: &AnomalyScore,
        outlier_score: &OutlierScore,
        verification_score: Option<&VerificationScore>,
    ) -> f64 {
        let mut combined_score = anomaly_score.score.clamp(0.0, 1.0) * 0.4;
        combined_score += outlier_score.score.clamp(0.0, 1.0) * 0.3;
        if let Some(verification) = verification_score {
            combined_score += (1.0 - verification.score.clamp(0.0, 1.0)) * 0.3;
        }
        combined_score
    }
    /// Confidence in the round's aggregate, in `[0, 1]`.
    ///
    /// The product of two measured quantities: the fraction of the submitting
    /// cohort that survived detection, and how tightly the survivors cluster around
    /// the aggregate (their mean distance to it, relative to the median update
    /// norm).
    pub(super) fn calculate_confidence_score(
        &self,
        honest_participants: &HashMap<String, Array1<T>>,
        total_submitted: usize,
        aggregate: &Array1<T>,
    ) -> Result<f64> {
        if total_submitted == 0 || honest_participants.is_empty() {
            return Ok(0.0);
        }
        let survivor_fraction =
            (honest_participants.len() as f64 / total_submitted as f64).min(1.0);
        let ordered = Self::ordered_cohort(honest_participants);
        let mut dispersion = 0.0f64;
        let mut norms: Vec<f64> = Vec::with_capacity(ordered.len());
        for (_, gradient) in &ordered {
            dispersion += from_scalar(euclidean_distance(gradient, aggregate)?)?;
            norms.push(from_scalar(l2_norm(gradient))?);
        }
        dispersion /= ordered.len() as f64;
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let scale = if norms.len().is_multiple_of(2) {
            let mid = norms.len() / 2;
            (norms[mid - 1] + norms[mid]) / 2.0
        } else {
            norms[norms.len() / 2]
        };
        let consensus = if scale > 0.0 {
            1.0 / (1.0 + dispersion / scale)
        } else if dispersion == 0.0 {
            1.0
        } else {
            0.0
        };
        Ok((survivor_fraction * consensus).clamp(0.0, 1.0))
    }
    /// Record this round's observed behaviour for every submitting participant.
    pub(super) fn record_behavior(
        &mut self,
        submitted: &HashMap<String, Array1<T>>,
        filtered: &HashMap<String, Array1<T>>,
        anomaly_results: &HashMap<String, AnomalyScore>,
        aggregate: &Array1<T>,
    ) -> Result<()> {
        let ordered: Vec<(String, Array1<T>)> = Self::ordered_cohort(submitted)
            .into_iter()
            .map(|(id, g)| (id.clone(), g.clone()))
            .collect();
        for (participant_id, gradient) in ordered {
            let participated = filtered.contains_key(&participant_id);
            let norm = from_scalar(l2_norm(&gradient))?;
            let similarity = from_scalar(cosine_similarity(&gradient, aggregate)?)?;
            let anomaly = anomaly_results
                .get(&participant_id)
                .map(|score| score.score.clamp(0.0, 1.0));
            let history = self.behavior_history.entry(participant_id).or_default();
            BehaviorHistory::push_bounded(&mut history.participation_pattern, participated);
            if participated {
                history.rounds_participated = history.rounds_participated.saturating_add(1);
                BehaviorHistory::push_bounded(&mut history.gradient_norms, norm);
                BehaviorHistory::push_bounded(&mut history.gradient_similarities, similarity);
                if let Some(anomaly) = anomaly {
                    BehaviorHistory::push_bounded(&mut history.anomaly_scores, anomaly);
                }
            }
        }
        Ok(())
    }
    /// Update participant reputations based on aggregation results.
    ///
    /// Reputation is an exponential moving average driven by `reputation_decay`;
    /// see [`super::types::ByzantineConfig::reputation_decay`].
    pub(super) fn update_reputations(
        &mut self,
        honest_participants: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<()> {
        let decay = self.config.reputation_decay;
        let honest_rate = 1.0 - decay;
        let byzantine_rate = (honest_rate * 5.0).min(1.0);
        let mut honest_ids: Vec<&String> = honest_participants.keys().collect();
        honest_ids.sort();
        for participant_id in honest_ids {
            let quality = self.derived_quality(participant_id);
            let reputation = self
                .reputation_scores
                .entry(participant_id.clone())
                .or_default();
            reputation.successful_aggregations =
                reputation.successful_aggregations.saturating_add(1);
            reputation.score = (decay * reputation.score + honest_rate).clamp(0.0, 1.0);
            if let Some((gradient_quality, consistency)) = quality {
                reputation.gradient_quality = gradient_quality;
                reputation.consistency_score = consistency;
            }
            reputation.trust_level = match reputation.score {
                s if s >= 0.8 => TrustLevel::High,
                s if s >= 0.5 => TrustLevel::Medium,
                _ => TrustLevel::Low,
            };
        }
        for participant_id in byzantine_participants {
            let quality = self.derived_quality(participant_id);
            let reputation = self
                .reputation_scores
                .entry(participant_id.clone())
                .or_default();
            reputation.detected_anomalies = reputation.detected_anomalies.saturating_add(1);
            reputation.score = (reputation.score * (1.0 - byzantine_rate)).clamp(0.0, 1.0);
            if let Some((gradient_quality, consistency)) = quality {
                reputation.gradient_quality = gradient_quality;
                reputation.consistency_score = consistency;
            }
            reputation.trust_level = if reputation.score < 0.1 {
                TrustLevel::Blacklisted
            } else if reputation.score < 0.3 {
                TrustLevel::Low
            } else {
                TrustLevel::Medium
            };
        }
        Ok(())
    }
    /// Gradient quality and consistency derived from the recorded behaviour history.
    pub(super) fn derived_quality(&self, participant_id: &str) -> Option<(f64, f64)> {
        let history = self.behavior_history.get(participant_id)?;
        let quality = 1.0 - history.mean_anomaly_score()?;
        let consistency = (history.mean_similarity()? + 1.0) / 2.0;
        Some((quality.clamp(0.0, 1.0), consistency.clamp(0.0, 1.0)))
    }
    /// Feed this round's outcome to the behaviour pattern model.
    pub(super) fn learn_patterns(
        &mut self,
        honest_participants: &HashMap<String, Array1<T>>,
        filtered: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<()> {
        let honest: Vec<Array1<T>> = Self::ordered_cohort(honest_participants)
            .into_iter()
            .map(|(_, g)| g.clone())
            .collect();
        for gradient in honest {
            self.anomaly_detector.learn_normal(&gradient)?;
        }
        let attacks: Vec<Array1<T>> = byzantine_participants
            .iter()
            .filter_map(|id| filtered.get(id).cloned())
            .collect();
        for gradient in attacks {
            self.anomaly_detector.learn_attack(&gradient)?;
        }
        Ok(())
    }
    /// Compute Euclidean distance between two gradients
    pub fn compute_euclidean_distance(&self, a: &Array1<T>, b: &Array1<T>) -> Result<T> {
        euclidean_distance(a, b)
    }
    /// Compute cosine similarity between two gradients
    pub fn compute_cosine_similarity(&self, a: &Array1<T>, b: &Array1<T>) -> Result<T> {
        cosine_similarity(a, b)
    }
    /// Get reputation updates
    pub(super) fn get_reputation_updates(&self) -> HashMap<String, ReputationScore> {
        self.reputation_scores.clone()
    }
}
