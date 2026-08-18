// Transfer learning across streaming optimization tasks.
//
// Split out of `meta_learning.rs` so that file stays under the 2000-line policy
// limit.
//
// This module implements *instance transfer*: experiences gathered on one stream
// (a "source domain") are replayed into the meta-learner training a different
// stream (the "target domain"), weighted by how similar the two domains look.
// That is the one transfer family this crate has the material for — it owns
// stored `MetaExperience` values but not the model weights or feature
// extractors the other `TransferStrategy` variants would need.
//
// What changed: `TransferLearning::new` used to fabricate its own report card —
// `success_rate: 0.5`, `improvement: 0.1`, `efficiency: 0.7`,
// `domain_similarity: 0.5` — while `source_experiences` was never written and
// nothing ever read any of it. Every metric here is now either measured or
// absent, and the struct only reports a domain similarity it actually computed.

use super::meta_learning::MetaExperience;
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Transfer learning strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    /// Direct parameter transfer: copy the source model's parameters.
    ParameterTransfer,
    /// Feature transfer: reuse the source feature representation.
    FeatureTransfer,
    /// Instance transfer: replay source-domain experiences into the target.
    InstanceTransfer,
    /// Relational transfer: carry over relations between entities.
    RelationalTransfer,
    /// Meta-transfer learning: transfer the learning procedure itself.
    MetaTransfer,
}

impl TransferStrategy {
    /// Whether this crate can carry out the strategy.
    ///
    /// Only [`TransferStrategy::InstanceTransfer`] is supported: the meta-learner
    /// stores experiences, not model weights, feature extractors or relational
    /// structure, so the other four have nothing here to transfer. Reporting
    /// that plainly is better than accepting the setting and quietly doing
    /// instance transfer (or nothing) instead.
    pub fn is_supported(self) -> bool {
        matches!(self, TransferStrategy::InstanceTransfer)
    }
}

/// Domain adaptation state: the characteristics of the source and target
/// domains and the resulting per-feature adaptation weights.
#[derive(Debug, Clone)]
pub struct DomainAdaptation<A: Float + Send + Sync> {
    /// Source domain characteristics, as last registered.
    source_characteristics: Vec<A>,
    /// Target domain characteristics, as last reported.
    target_characteristics: Vec<A>,
    /// Per-feature adaptation weights: `target_i / source_i` where the source
    /// coordinate is non-zero, so a source feature on a different scale is
    /// rescaled rather than transferred verbatim.
    adaptation_weights: Vec<A>,
    /// Cosine similarity between the two characteristic vectors, or `None`
    /// before both have been supplied.
    domain_similarity: Option<A>,
}

impl<A: Float + Send + Sync> Default for DomainAdaptation<A> {
    fn default() -> Self {
        Self {
            source_characteristics: Vec::new(),
            target_characteristics: Vec::new(),
            adaptation_weights: Vec::new(),
            domain_similarity: None,
        }
    }
}

impl<A: Float + Send + Sync> DomainAdaptation<A> {
    /// Records the source domain's characteristic vector.
    pub fn set_source_characteristics(&mut self, characteristics: Vec<A>) {
        self.source_characteristics = characteristics;
        self.recompute();
    }

    /// Records the target domain's characteristic vector.
    pub fn set_target_characteristics(&mut self, characteristics: Vec<A>) {
        self.target_characteristics = characteristics;
        self.recompute();
    }

    /// Cosine similarity between source and target domains, or `None` until
    /// both characteristic vectors are available and non-degenerate.
    pub fn domain_similarity(&self) -> Option<A> {
        self.domain_similarity
    }

    /// Per-feature adaptation weights derived from the two characteristic
    /// vectors.
    pub fn adaptation_weights(&self) -> &[A] {
        &self.adaptation_weights
    }

    fn recompute(&mut self) {
        let shared = self
            .source_characteristics
            .len()
            .min(self.target_characteristics.len());
        if shared == 0 {
            self.domain_similarity = None;
            self.adaptation_weights.clear();
            return;
        }

        let mut dot = A::zero();
        let mut source_norm = A::zero();
        let mut target_norm = A::zero();
        self.adaptation_weights.clear();
        for index in 0..shared {
            let source = self.source_characteristics[index];
            let target = self.target_characteristics[index];
            dot = dot + source * target;
            source_norm = source_norm + source * source;
            target_norm = target_norm + target * target;
            self.adaptation_weights.push(if source == A::zero() {
                A::one()
            } else {
                target / source
            });
        }

        self.domain_similarity = if source_norm > A::zero() && target_norm > A::zero() {
            Some(dot / (source_norm.sqrt() * target_norm.sqrt()))
        } else {
            None
        };
    }
}

/// Measured transfer-learning outcomes.
///
/// Every field starts at its neutral value and only moves when a transfer has
/// actually been evaluated, so a fresh instance reports "nothing measured yet"
/// rather than a flattering constant.
#[derive(Debug, Clone)]
pub struct TransferMetrics<A: Float + Send + Sync> {
    /// Transfers that improved the target's reward, as a fraction of those
    /// evaluated. `None` until at least one transfer has been evaluated.
    pub success_rate: Option<A>,
    /// Mean reward improvement across evaluated transfers.
    pub improvement: Option<A>,
    /// Fraction of transferred experiences that were retained (not discarded
    /// for low domain similarity).
    pub efficiency: Option<A>,
    /// Transfers that made the target worse.
    pub negative_transfer_count: usize,
    /// Transfers evaluated so far.
    pub evaluated_transfers: usize,
}

impl<A: Float + Send + Sync> Default for TransferMetrics<A> {
    fn default() -> Self {
        Self {
            success_rate: None,
            improvement: None,
            efficiency: None,
            negative_transfer_count: 0,
            evaluated_transfers: 0,
        }
    }
}

/// Minimum cosine similarity between domains before source experiences are
/// replayed into the target at all. Below this, transfer is more likely to be
/// negative than helpful, so nothing is transferred.
pub const MIN_TRANSFER_SIMILARITY: f64 = 0.5;

/// Transfer learning system: stores per-source-domain experiences and replays
/// them into a target domain, weighted by measured domain similarity.
#[derive(Debug, Clone)]
pub struct TransferLearning<A: Float + Send + Sync> {
    /// Source domain experiences, keyed by source task id.
    source_experiences: HashMap<String, Vec<MetaExperience<A>>>,
    /// Strategies this instance was asked to use.
    transfer_strategies: Vec<TransferStrategy>,
    /// Domain adaptation state.
    domain_adaptation: DomainAdaptation<A>,
    /// Measured transfer outcomes.
    transfer_metrics: TransferMetrics<A>,
    /// Cumulative reward improvement across evaluated transfers, used to derive
    /// `TransferMetrics::improvement`.
    improvement_total: A,
    /// Evaluated transfers that improved the target.
    successful_transfers: usize,
}

impl<A: Float + Send + Sync + Clone> Default for TransferLearning<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Send + Sync + Clone> TransferLearning<A> {
    /// Creates a transfer-learning system using instance transfer.
    pub fn new() -> Self {
        Self {
            source_experiences: HashMap::new(),
            transfer_strategies: vec![TransferStrategy::InstanceTransfer],
            domain_adaptation: DomainAdaptation::default(),
            transfer_metrics: TransferMetrics::default(),
            improvement_total: A::zero(),
            successful_transfers: 0,
        }
    }

    /// Requests a specific set of transfer strategies.
    ///
    /// Returns an error naming any strategy this crate cannot carry out, rather
    /// than accepting it and silently substituting instance transfer.
    pub fn set_strategies(&mut self, strategies: Vec<TransferStrategy>) -> Result<(), String> {
        if let Some(unsupported) = strategies.iter().find(|s| !s.is_supported()) {
            return Err(format!(
                "transfer strategy {unsupported:?} is not implemented: the meta-learner \
                 stores experiences, not model parameters, feature extractors or \
                 relational structure; only InstanceTransfer is supported"
            ));
        }
        if strategies.is_empty() {
            return Err("at least one transfer strategy is required".to_string());
        }
        self.transfer_strategies = strategies;
        Ok(())
    }

    /// Strategies currently in force.
    pub fn strategies(&self) -> &[TransferStrategy] {
        &self.transfer_strategies
    }

    /// Registers a source domain's experiences and characteristic vector.
    pub fn register_source(
        &mut self,
        source_id: String,
        experiences: Vec<MetaExperience<A>>,
        characteristics: Vec<A>,
    ) {
        self.domain_adaptation
            .set_source_characteristics(characteristics);
        self.source_experiences.insert(source_id, experiences);
    }

    /// Number of registered source domains.
    pub fn source_domain_count(&self) -> usize {
        self.source_experiences.len()
    }

    /// Measured transfer outcomes.
    pub fn metrics(&self) -> &TransferMetrics<A> {
        &self.transfer_metrics
    }

    /// Similarity between the registered source and the last reported target.
    pub fn domain_similarity(&self) -> Option<A> {
        self.domain_adaptation.domain_similarity()
    }

    /// Selects source experiences to replay into a target domain described by
    /// `target_characteristics`.
    ///
    /// Returns an empty vector when there is no registered source, when the
    /// domains are not similar enough (below [`MIN_TRANSFER_SIMILARITY`]), or
    /// when the similarity cannot be computed — all of which are honest
    /// "nothing to transfer" outcomes rather than errors.
    pub fn select_transfer_batch(
        &mut self,
        target_characteristics: Vec<A>,
        limit: usize,
    ) -> Vec<MetaExperience<A>> {
        self.domain_adaptation
            .set_target_characteristics(target_characteristics);

        let available: usize = self.source_experiences.values().map(Vec::len).sum();
        if available == 0 || limit == 0 {
            return Vec::new();
        }

        let Some(similarity) = self.domain_adaptation.domain_similarity() else {
            return Vec::new();
        };
        let threshold = A::from(MIN_TRANSFER_SIMILARITY).unwrap_or_else(A::zero);
        if similarity < threshold {
            self.record_efficiency(0, available);
            return Vec::new();
        }

        // Take the highest-priority source experiences first: those are the ones
        // prioritized replay would have picked in their own domain.
        let mut candidates: Vec<MetaExperience<A>> = self
            .source_experiences
            .values()
            .flat_map(|batch| batch.iter().cloned())
            .collect();
        candidates.sort_by(|a, b| crate::utils::total_order(&b.priority, &a.priority));
        candidates.truncate(limit);

        // Scale each transferred experience's replay priority by the measured
        // domain similarity, so a marginally similar domain contributes
        // correspondingly less.
        for experience in &mut candidates {
            experience.priority = experience.priority * similarity;
        }

        self.record_efficiency(candidates.len(), available);
        candidates
    }

    /// Records the outcome of a completed transfer: the target's reward before
    /// and after replaying the transferred experiences.
    pub fn record_transfer_outcome(&mut self, reward_before: A, reward_after: A) {
        let improvement = reward_after - reward_before;
        self.transfer_metrics.evaluated_transfers += 1;
        self.improvement_total = self.improvement_total + improvement;

        if improvement > A::zero() {
            self.successful_transfers += 1;
        } else if improvement < A::zero() {
            self.transfer_metrics.negative_transfer_count += 1;
        }

        if let Some(evaluated) = A::from(self.transfer_metrics.evaluated_transfers) {
            if evaluated > A::zero() {
                self.transfer_metrics.improvement = Some(self.improvement_total / evaluated);
                self.transfer_metrics.success_rate =
                    A::from(self.successful_transfers).map(|successes| successes / evaluated);
            }
        }
    }

    fn record_efficiency(&mut self, retained: usize, available: usize) {
        if available == 0 {
            self.transfer_metrics.efficiency = None;
            return;
        }
        let (Some(retained), Some(available)) = (A::from(retained), A::from(available)) else {
            return;
        };
        self.transfer_metrics.efficiency = Some(retained / available);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::adaptive_streaming::meta_learning::{
        EpisodeContext, EpisodeOutcome, MetaAction, MetaExperience, MetaState,
    };
    use crate::streaming::adaptive_streaming::optimizer::AdaptationType;
    use std::time::{Duration, Instant};

    fn experience(priority: f64, reward: f64) -> MetaExperience<f64> {
        MetaExperience {
            id: 1,
            state: MetaState {
                performance_metrics: vec![reward],
                resource_state: vec![1.0],
                drift_indicators: vec![0.0],
                adaptation_history: 0,
                timestamp: Instant::now(),
            },
            action: MetaAction {
                adaptation_magnitudes: vec![0.1],
                adaptation_types: vec![AdaptationType::LearningRate],
                learning_rate_change: 0.1,
                buffer_size_change: 0.0,
                timestamp: Instant::now(),
            },
            reward,
            next_state: None,
            timestamp: Instant::now(),
            episode_context: EpisodeContext {
                episode_id: 0,
                start_time: Instant::now(),
                duration: Duration::ZERO,
                initial_performance: 0.0,
                final_performance: reward,
                adaptation_count: 1,
                outcome: EpisodeOutcome::Neutral,
            },
            priority,
            replay_count: 0,
        }
    }

    /// A fresh instance must report "nothing measured" — not the old fabricated
    /// `success_rate: 0.5` / `improvement: 0.1` / `efficiency: 0.7`.
    #[test]
    fn fresh_metrics_are_absent_not_fabricated() {
        let transfer = TransferLearning::<f64>::new();
        let metrics = transfer.metrics();
        assert!(metrics.success_rate.is_none());
        assert!(metrics.improvement.is_none());
        assert!(metrics.efficiency.is_none());
        assert_eq!(metrics.evaluated_transfers, 0);
        assert!(
            transfer.domain_similarity().is_none(),
            "no domain has been described yet, so there is no similarity to report"
        );
    }

    /// Similar domains transfer; the transferred priorities are scaled by the
    /// measured similarity rather than copied verbatim.
    #[test]
    fn similar_domains_transfer_with_similarity_scaled_priorities() {
        let mut transfer = TransferLearning::<f64>::new();
        transfer.register_source(
            "source".to_string(),
            vec![experience(1.0, 1.0), experience(0.2, 0.0)],
            vec![1.0, 1.0],
        );

        let batch = transfer.select_transfer_batch(vec![1.0, 1.0], 2);
        assert_eq!(batch.len(), 2, "identical domains must transfer everything");
        let similarity = transfer.domain_similarity().expect("similarity");
        assert!((similarity - 1.0).abs() < 1e-9, "similarity = {similarity}");
        // Highest source priority first, scaled by similarity (1.0 here).
        assert!((batch[0].priority - 1.0).abs() < 1e-9);
        assert_eq!(transfer.metrics().efficiency, Some(1.0));
    }

    /// Dissimilar domains must transfer nothing rather than risk negative
    /// transfer, and must say so through `efficiency`.
    #[test]
    fn dissimilar_domains_transfer_nothing() {
        let mut transfer = TransferLearning::<f64>::new();
        transfer.register_source(
            "source".to_string(),
            vec![experience(1.0, 1.0)],
            vec![1.0, 0.0],
        );

        // Orthogonal characteristics: cosine similarity 0.
        let batch = transfer.select_transfer_batch(vec![0.0, 1.0], 4);
        assert!(batch.is_empty(), "orthogonal domains must not transfer");
        assert_eq!(transfer.metrics().efficiency, Some(0.0));
    }

    /// Outcomes are measured from the rewards actually observed.
    #[test]
    fn transfer_outcomes_are_measured() {
        let mut transfer = TransferLearning::<f64>::new();
        transfer.record_transfer_outcome(1.0, 2.0); // +1.0, success
        transfer.record_transfer_outcome(1.0, 0.0); // -1.0, negative transfer

        let metrics = transfer.metrics();
        assert_eq!(metrics.evaluated_transfers, 2);
        assert_eq!(metrics.negative_transfer_count, 1);
        assert_eq!(metrics.success_rate, Some(0.5));
        assert_eq!(metrics.improvement, Some(0.0));
    }

    /// Unsupported strategies are refused by name instead of being accepted and
    /// silently downgraded to instance transfer.
    #[test]
    fn unsupported_strategies_are_refused() {
        let mut transfer = TransferLearning::<f64>::new();
        let err = transfer
            .set_strategies(vec![TransferStrategy::ParameterTransfer])
            .expect_err("ParameterTransfer must be refused");
        assert!(err.contains("ParameterTransfer"), "{err}");
        transfer
            .set_strategies(vec![TransferStrategy::InstanceTransfer])
            .expect("InstanceTransfer is supported");
        assert_eq!(transfer.strategies(), [TransferStrategy::InstanceTransfer]);
    }
}
