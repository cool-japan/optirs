//! The dynamic architecture adapter.
//!
//! Split out of `adaptive/types.rs` to keep every file under the 2000-line cap.
//! The interesting part is [`DynamicArchitectureAdapter::adapt_architecture`],
//! which used to ignore all three of its arguments and always return
//! `changes = [LayerCountChange(6)]`, `expected_improvement = 0.1`,
//! `confidence = 0.8`.

use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::Result;
use crate::transformer_based_optimizer::TransformerOptimizerConfig;

use super::types::{
    AdaptationStrategy, ArchitectureAdaptation, ArchitectureChange, ArchitecturePerformance,
    ArchitectureSearchSpace, AttentionOptimization, LandscapeAnalysis, ResourceConstraints,
    SequenceAdaptation,
};
use super::AdaptiveConfig;

/// Dynamic architecture adapter
#[derive(Debug)]
pub struct DynamicArchitectureAdapter<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Current architecture configuration
    current_config: TransformerOptimizerConfig<T>,
    /// Architecture performance history
    pub(super) performance_history: VecDeque<ArchitecturePerformance<T>>,
    /// Adaptation strategy
    adaptation_strategy: AdaptationStrategy,
    /// Resource constraints
    resource_constraints: ResourceConstraints,
    /// Architecture search space
    search_space: ArchitectureSearchSpace,
    /// Whether layer-count changes are permitted, from
    /// `AdaptiveConfig::layer_adaptation`
    layer_adaptation_enabled: bool,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    DynamicArchitectureAdapter<T>
{
    /// Build the architecture adapter from the adaptive configuration.
    ///
    /// `layer_adaptation` selects the adaptation strategy (`Gradual` when layer
    /// changes are allowed, `Conservative` when they are not) and gates the
    /// layer-count moves in [`Self::adapt_architecture`]; `memory_budget`
    /// becomes the adapter's memory constraint. Previously the config was
    /// ignored entirely.
    pub fn new(config: &AdaptiveConfig<T>) -> Result<Self> {
        let mut resource_constraints = ResourceConstraints::default();
        if config.memory_budget > 0 {
            resource_constraints.max_memory = config.memory_budget;
        }
        Ok(Self {
            current_config: TransformerOptimizerConfig::<T>::default(),
            performance_history: VecDeque::new(),
            adaptation_strategy: if config.layer_adaptation {
                AdaptationStrategy::Gradual
            } else {
                AdaptationStrategy::Conservative
            },
            resource_constraints,
            search_space: ArchitectureSearchSpace::default(),
            layer_adaptation_enabled: config.layer_adaptation,
        })
    }

    /// Propose an architecture change for the measured landscape.
    ///
    /// The proposal is derived, not fabricated:
    ///
    /// * **Layer count** moves one step within
    ///   `ArchitectureSearchSpace::layer_count_range` — up when the landscape is
    ///   complex, down when it is simple — and only when
    ///   `AdaptiveConfig::layer_adaptation` is enabled.
    /// * **Attention heads** follow the head count the attention manager
    ///   actually chose, snapped to the nearest legal option that divides the
    ///   model dimension.
    /// * **Dropout** rises with difficulty (more regularization when
    ///   optimization is struggling), bounded to `[0, 0.5]`.
    /// * **Expected improvement** is the measured headroom
    ///   `difficulty · (1 - complexity) · efficiency_gain`, i.e. large only when
    ///   there is something to gain *and* the surface is tractable *and* the
    ///   sequence adaptation actually bought efficiency.
    /// * **Confidence** is the landscape analysis confidence discounted by how
    ///   many changes are being proposed at once.
    ///
    /// The previous implementation ignored all three inputs and always returned
    /// `changes = [LayerCountChange(6)]`, `expected_improvement = 0.1`,
    /// `confidence = 0.8`.
    pub fn adapt_architecture(
        &mut self,
        landscape: &LandscapeAnalysis<T>,
        sequence: &SequenceAdaptation<T>,
        attention: &AttentionOptimization<T>,
    ) -> Result<ArchitectureAdaptation<T>> {
        let complexity = landscape.complexity.to_f64().unwrap_or(0.0);
        let difficulty = landscape.difficulty.to_f64().unwrap_or(0.0);

        let mut adapted = self.current_config.clone();
        let mut changes = Vec::new();

        // --- layers ---------------------------------------------------------
        //
        // The target depth is a *function of the measured complexity*, not of the
        // current depth:
        //
        //     target = min + round(complexity · (max − min))
        //
        // That matters because the adapter is re-seeded from the optimizer's live
        // configuration on every `enhance_optimizer` call. The previous rule
        // (`current + 1` whenever complexity > 0.6) therefore **ratcheted**: eight
        // calls on an unchanged history walked the depth 2→3→…→10, and because a
        // depth change is structural, each step rebuilt — and re-initialized —
        // the whole transformer. A depth that depends only on the landscape is
        // idempotent: once the architecture matches the measurement, further calls
        // propose nothing and no weights are lost.
        //
        // `AdaptationStrategy::Gradual` still approaches the target one layer at a
        // time (so it converges in a bounded number of steps and then stops);
        // `Conservative` leaves the depth alone entirely; the remaining strategies
        // jump straight to the target.
        if self.layer_adaptation_enabled {
            let (min_layers, max_layers) = self.search_space.layer_count_range;
            let lo = min_layers.max(1);
            let hi = max_layers.max(lo);
            let span = (hi - lo) as f64;
            let target = lo + (complexity.clamp(0.0, 1.0) * span).round() as usize;
            let target = target.clamp(lo, hi);

            let current = adapted.num_transformer_layers;
            let proposed = match self.adaptation_strategy {
                AdaptationStrategy::Conservative => current,
                AdaptationStrategy::Gradual => match target.cmp(&current) {
                    std::cmp::Ordering::Greater => current + 1,
                    std::cmp::Ordering::Less => current.saturating_sub(1).max(lo),
                    std::cmp::Ordering::Equal => current,
                },
                _ => target,
            };
            // A deeper transformer is the one proposal that costs memory, so it
            // is the one the configured budget has to veto. Without this check
            // `resource_constraints` was populated from
            // `AdaptiveConfig::memory_budget` and then never consulted: the
            // adapter would happily propose a depth whose weights could not fit.
            let within_budget = |layers: usize| {
                layers <= current
                    || Self::estimated_parameter_megabytes(&adapted, layers)
                        <= self.resource_constraints.max_memory
            };
            if proposed != current && proposed > 0 && within_budget(proposed) {
                adapted.num_transformer_layers = proposed;
                changes.push(ArchitectureChange::LayerCountChange(proposed));
            }
        }

        // --- attention heads ------------------------------------------------
        let requested_heads = attention.attention_patterns.dim().0;
        if requested_heads > 0 && requested_heads != adapted.num_attention_heads {
            // Only accept a head count that evenly divides the model dimension.
            let candidate = self
                .search_space
                .attention_head_options
                .iter()
                .copied()
                .filter(|&h| h > 0 && adapted.model_dimension.is_multiple_of(h))
                .min_by_key(|&h| h.abs_diff(requested_heads));
            if let Some(heads) = candidate {
                if heads != adapted.num_attention_heads {
                    adapted.num_attention_heads = heads;
                    adapted.attention_head_dimension = adapted.model_dimension / heads;
                    changes.push(ArchitectureChange::AttentionHeadChange(heads));
                }
            }
        }

        // --- dropout --------------------------------------------------------
        let proposed_dropout = (0.5 * difficulty).clamp(0.0, 0.5);
        if (proposed_dropout - adapted.dropout_rate).abs() > 1e-6 {
            adapted.dropout_rate = proposed_dropout;
            changes.push(ArchitectureChange::DropoutChange(proposed_dropout));
        }

        // --- expected improvement & confidence ------------------------------
        let efficiency = sequence
            .efficiency_gain
            .to_f64()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let headroom = (difficulty * (1.0 - complexity) * efficiency).clamp(0.0, 1.0);
        // Each simultaneous change dilutes confidence: more moving parts, less
        // certainty that the estimate holds.
        let dilution = 1.0 / (1.0 + changes.len() as f64);
        let confidence = (landscape.confidence.to_f64().unwrap_or(0.0) * dilution).clamp(0.0, 1.0);

        Ok(ArchitectureAdaptation {
            adapted_config: adapted,
            changes,
            expected_improvement: scirs2_core::numeric::NumCast::from(headroom)
                .unwrap_or_else(|| T::zero()),
            confidence: scirs2_core::numeric::NumCast::from(confidence)
                .unwrap_or_else(|| T::zero()),
        })
    }

    /// Adopt `config` as the baseline every future proposal is a delta from.
    ///
    /// Without this the adapter always started from
    /// `TransformerOptimizerConfig::default()` and proposed changes relative to
    /// that, regardless of how the optimizer it was supposed to be adapting was
    /// actually configured.
    pub fn sync_with(&mut self, config: &TransformerOptimizerConfig<T>) {
        self.current_config = config.clone();
    }

    /// The configuration the adapter currently proposes.
    pub fn current_config(&self) -> &TransformerOptimizerConfig<T> {
        &self.current_config
    }

    /// Adaptation strategy selected from the configuration.
    pub fn adaptation_strategy(&self) -> AdaptationStrategy {
        self.adaptation_strategy
    }

    /// Memory budget proposals are checked against, in **megabytes**, from
    /// `AdaptiveConfig::memory_budget` (which is documented in MB).
    pub fn memory_budget(&self) -> usize {
        self.resource_constraints.max_memory
    }

    /// Weight-memory estimate, in **megabytes**, for `config` at `layers` layers.
    ///
    /// Per encoder layer a standard transformer holds four `d × d` attention
    /// projections and two `d × ff` feed-forward matrices, i.e.
    /// `4d² + 2·d·ff` scalars of `T`. Biases and layer-norm parameters are
    /// `O(d)` and are ignored; this is a lower bound used only to veto a depth
    /// increase that clearly cannot fit the configured budget.
    pub fn estimated_parameter_megabytes(
        config: &TransformerOptimizerConfig<T>,
        layers: usize,
    ) -> usize {
        let d = config.model_dimension;
        let ff = config.feedforward_dimension;
        let per_layer = 4usize
            .saturating_mul(d)
            .saturating_mul(d)
            .saturating_add(2usize.saturating_mul(d).saturating_mul(ff));
        per_layer
            .saturating_mul(layers)
            .saturating_mul(std::mem::size_of::<T>())
            / (1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(memory_budget: usize) -> DynamicArchitectureAdapter<f32> {
        let config = AdaptiveConfig::<f32> {
            memory_budget,
            layer_adaptation: true,
            ..AdaptiveConfig::default()
        };
        DynamicArchitectureAdapter::new(&config).expect("adapter")
    }

    #[test]
    fn memory_budget_is_taken_from_the_adaptive_config() {
        assert_eq!(adapter(256).memory_budget(), 256);
        // A zero budget keeps the `ResourceConstraints` default rather than
        // vetoing every proposal.
        assert_eq!(
            adapter(0).memory_budget(),
            ResourceConstraints::default().max_memory
        );
    }

    /// `resource_constraints` was populated from `AdaptiveConfig::memory_budget`
    /// and then never read, so a depth increase was proposed regardless of the
    /// configured budget.
    #[test]
    fn a_depth_increase_that_exceeds_the_budget_is_vetoed() {
        let cfg = TransformerOptimizerConfig::<f32>::default();
        let deeper = cfg.num_transformer_layers + 1;
        let needed = DynamicArchitectureAdapter::<f32>::estimated_parameter_megabytes(&cfg, deeper);
        assert!(needed > 0, "default config should need measurable memory");

        // One megabyte short of what the deeper architecture needs: the veto
        // predicate `estimate <= budget` must reject it.
        let tight = adapter(needed - 1);
        assert!(
            needed > tight.memory_budget(),
            "a budget below the requirement must not admit the deeper architecture"
        );

        // Ample budget: the same estimate must fit.
        let roomy = adapter(needed * 4);
        assert!(needed <= roomy.memory_budget());
    }

    /// The estimate must grow with depth, width and feed-forward size, or the
    /// veto would be insensitive to the thing it is guarding.
    #[test]
    fn parameter_estimate_grows_with_the_architecture() {
        let base = TransformerOptimizerConfig::<f32>::default();
        let est = |c: &TransformerOptimizerConfig<f32>, l: usize| {
            DynamicArchitectureAdapter::<f32>::estimated_parameter_megabytes(c, l)
        };
        assert!(est(&base, 8) > est(&base, 4));

        let wide = TransformerOptimizerConfig::<f32> {
            model_dimension: base.model_dimension * 2,
            ..base.clone()
        };
        assert!(est(&wide, 4) > est(&base, 4));

        let fat = TransformerOptimizerConfig::<f32> {
            feedforward_dimension: base.feedforward_dimension * 2,
            ..base.clone()
        };
        assert!(est(&fat, 4) > est(&base, 4));
    }
}
