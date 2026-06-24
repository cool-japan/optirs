//! The [`MultimodalNasEngine`] orchestration object and the
//! [`MultimodalEvaluation`] record, implementing propose / mutate /
//! crossover / record / Pareto-front operations over the search space.

use super::architecture::MultimodalArchitecture;
use super::search_space::MultimodalSearchSpace;
use super::types::{
    encoder_capacity, fusion_supports, layer_capacity, ActivationKind, FusionOp, Modality,
    ModalityEncoder, MultimodalLayer,
};
use crate::error::{OptimError, Result};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

// =======================================================================
// Evaluation
// =======================================================================

/// Evaluation result for a single multimodal architecture.
///
/// * `accuracy` — task accuracy in `[0, 1]`. Higher is better.
/// * `latency_ms` — wall-clock inference latency on the target device.
/// * `memory_mb` — peak resident memory during inference.
/// * `param_count` — total trainable parameter count.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MultimodalEvaluation {
    /// Task accuracy in `[0, 1]`. Higher is better.
    pub accuracy: f64,
    /// Wall-clock inference latency in milliseconds. Lower is better.
    pub latency_ms: f64,
    /// Peak resident memory in megabytes. Lower is better.
    pub memory_mb: f64,
    /// Total trainable parameter count.
    pub param_count: u64,
}

// =======================================================================
// Engine
// =======================================================================

/// Probability of substituting (vs. perturbing) a layer in
/// [`MultimodalNasEngine::mutate`].
const SUBSTITUTE_PROB: f64 = 0.4;
/// Probability of perturbing a layer once substitution is not chosen.
const PERTURB_PROB: f64 = 0.7;
/// Probability of mutating the fusion operator once layer edits are not
/// chosen.
const FUSION_MUTATE_PROB: f64 = 0.85;

/// The multimodal NAS engine.
///
/// `MultimodalNasEngine` owns the search space, the running history of
/// evaluated architectures, and the current best-so-far record. Metrics
/// are inherently `f64`-valued, so (like [`crate::speech_nas::SpeechNasEngine`])
/// it is non-generic over a numeric type.
#[derive(Debug, Clone)]
pub struct MultimodalNasEngine {
    search_space: MultimodalSearchSpace,
    evaluated: Vec<(MultimodalArchitecture, MultimodalEvaluation)>,
    best_model: Option<MultimodalArchitecture>,
    best_score: f64,
    next_id: usize,
}

impl MultimodalNasEngine {
    /// Construct a new engine from an explicit search space.
    pub fn new(search_space: MultimodalSearchSpace) -> Self {
        Self {
            search_space,
            evaluated: Vec::new(),
            best_model: None,
            best_score: f64::NEG_INFINITY,
            next_id: 0,
        }
    }

    /// Construct a new engine using [`MultimodalSearchSpace::default`].
    pub fn with_default_search_space() -> Self {
        Self::new(MultimodalSearchSpace::default())
    }

    /// Borrow the search space.
    pub fn search_space(&self) -> &MultimodalSearchSpace {
        &self.search_space
    }

    /// Borrow the evaluation history.
    pub fn evaluated(&self) -> &[(MultimodalArchitecture, MultimodalEvaluation)] {
        &self.evaluated
    }

    /// Return the best architecture recorded so far, if any.
    pub fn best(&self) -> Option<&MultimodalArchitecture> {
        self.best_model.as_ref()
    }

    /// Return the evaluation associated with [`Self::best`].
    pub fn best_evaluation(&self) -> Option<MultimodalEvaluation> {
        let best_id = self.best_model.as_ref()?.model_id.clone();
        self.evaluated
            .iter()
            .find(|(m, _)| m.model_id == best_id)
            .map(|(_, e)| *e)
    }

    /// Reset the engine to a freshly-constructed state, preserving the
    /// search space.
    pub fn reset(&mut self) {
        self.evaluated.clear();
        self.best_model = None;
        self.best_score = f64::NEG_INFINITY;
        self.next_id = 0;
    }

    // -------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------

    /// Allocate the next architecture identifier.
    fn allocate_model_id(&mut self) -> String {
        let id = format!("multimodal_arch_{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Candidate layers usable in a given modality's encoder
    /// (modality-specific layers plus the generic layers).
    fn candidate_layers_for(&self, modality: Modality) -> Vec<MultimodalLayer> {
        let mut v = match modality {
            Modality::Image => self.search_space.image_layers.clone(),
            Modality::Text => self.search_space.text_layers.clone(),
            Modality::Audio => self.search_space.audio_layers.clone(),
        };
        v.extend(self.search_space.generic_layers.iter().copied());
        v
    }

    /// The heaviest legal layer for a modality (used as rebalance filler).
    fn heaviest_candidate(&self, modality: Modality) -> Option<MultimodalLayer> {
        self.candidate_layers_for(modality)
            .into_iter()
            .max_by(|a, b| {
                layer_capacity(a)
                    .partial_cmp(&layer_capacity(b))
                    .unwrap_or(Ordering::Equal)
            })
    }

    /// Validate that the configured search space can produce a valid arch.
    fn ensure_search_space_valid(&self) -> Result<()> {
        if self.search_space.available_modalities.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "available_modalities is empty".to_string(),
            ));
        }
        // No duplicate available modalities.
        for (i, &m) in self.search_space.available_modalities.iter().enumerate() {
            if self.search_space.available_modalities[i + 1..].contains(&m) {
                return Err(OptimError::SearchSpaceError(format!(
                    "duplicate available modality {}",
                    m
                )));
            }
        }
        for &m in &self.search_space.available_modalities {
            if self.candidate_layers_for(m).is_empty() {
                return Err(OptimError::SearchSpaceError(format!(
                    "no candidate encoder layers for modality {}",
                    m
                )));
            }
        }
        if self.search_space.head_layers.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "head_layers is empty".to_string(),
            ));
        }
        if self.search_space.fusion_ops.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "fusion_ops is empty".to_string(),
            ));
        }
        if self.search_space.min_encoder_len == 0 {
            return Err(OptimError::SearchSpaceError(
                "min_encoder_len must be at least 1".to_string(),
            ));
        }
        if self.search_space.max_encoder_len < self.search_space.min_encoder_len {
            return Err(OptimError::SearchSpaceError(format!(
                "max_encoder_len ({}) must be >= min_encoder_len ({})",
                self.search_space.max_encoder_len, self.search_space.min_encoder_len
            )));
        }
        if self.search_space.min_head_len == 0 {
            return Err(OptimError::SearchSpaceError(
                "min_head_len must be at least 1".to_string(),
            ));
        }
        if self.search_space.max_head_len < self.search_space.min_head_len {
            return Err(OptimError::SearchSpaceError(format!(
                "max_head_len ({}) must be >= min_head_len ({})",
                self.search_space.max_head_len, self.search_space.min_head_len
            )));
        }
        if !self.search_space.max_capacity_ratio.is_finite()
            || self.search_space.max_capacity_ratio < 1.0
        {
            return Err(OptimError::SearchSpaceError(format!(
                "max_capacity_ratio must be a finite value >= 1.0, got {}",
                self.search_space.max_capacity_ratio
            )));
        }
        Ok(())
    }

    /// The rebalance target ratio (slightly stricter than the validation
    /// ratio so that produced architectures clear `validate` with margin).
    fn rebalance_target(&self) -> f64 {
        (self.search_space.max_capacity_ratio * 0.9).max(1.0)
    }

    /// Sample a layer index uniformly from a non-empty candidate slice.
    fn sample_layer<R: Rng>(
        candidates: &[MultimodalLayer],
        rng: &mut Random<R>,
    ) -> Option<MultimodalLayer> {
        if candidates.is_empty() {
            None
        } else {
            Some(candidates[rng.gen_range(0..candidates.len())])
        }
    }

    /// Build a fresh encoder of the given length for a modality.
    fn build_encoder<R: Rng>(
        &self,
        modality: Modality,
        length: usize,
        rng: &mut Random<R>,
    ) -> Result<ModalityEncoder> {
        let candidates = self.candidate_layers_for(modality);
        if candidates.is_empty() {
            return Err(OptimError::SearchSpaceError(format!(
                "no candidate encoder layers for modality {}",
                modality
            )));
        }
        let mut layers = Vec::with_capacity(length);
        for _ in 0..length.max(1) {
            match Self::sample_layer(&candidates, rng) {
                Some(layer) => layers.push(layer),
                None => {
                    return Err(OptimError::SearchSpaceError(format!(
                        "failed to sample encoder layer for modality {}",
                        modality
                    )))
                }
            }
        }
        Ok(ModalityEncoder { modality, layers })
    }

    /// Pick a fusion operator compatible with `num_modalities`, falling
    /// back to the always-valid [`FusionOp::EarlyConcat`].
    fn pick_compatible_fusion<R: Rng>(
        &self,
        num_modalities: usize,
        rng: &mut Random<R>,
    ) -> FusionOp {
        let compatible: Vec<FusionOp> = self
            .search_space
            .fusion_ops
            .iter()
            .copied()
            .filter(|f| fusion_supports(f, num_modalities))
            .collect();
        if compatible.is_empty() {
            // EarlyConcat is valid for any modality count >= 1.
            FusionOp::EarlyConcat
        } else {
            compatible[rng.gen_range(0..compatible.len())]
        }
    }

    /// Rebalance encoders so the max/min capacity ratio drops to at most
    /// `target_ratio`, by replacing the lightest encoder's cheapest layer
    /// with the heaviest legal filler. Terminates in at most one pass per
    /// non-filler layer.
    fn rebalance(&self, encoders: &mut [ModalityEncoder], target_ratio: f64) {
        if encoders.len() < 2 {
            return;
        }
        let total_layers: usize = encoders.iter().map(|e| e.layers.len()).sum();
        for _ in 0..=total_layers {
            // Locate lightest and heaviest encoder capacities.
            let mut min_idx = 0usize;
            let mut min_c = f64::INFINITY;
            let mut max_c = f64::NEG_INFINITY;
            for (i, e) in encoders.iter().enumerate() {
                let c = encoder_capacity(e);
                if c < min_c {
                    min_c = c;
                    min_idx = i;
                }
                if c > max_c {
                    max_c = c;
                }
            }
            if min_c <= 0.0 || !max_c.is_finite() {
                break;
            }
            if max_c / min_c <= target_ratio {
                break;
            }

            let modality = encoders[min_idx].modality;
            let filler = match self.heaviest_candidate(modality) {
                Some(f) => f,
                None => break,
            };
            let filler_cap = layer_capacity(&filler);

            let layers = &mut encoders[min_idx].layers;
            if layers.is_empty() {
                break;
            }
            // Replace the cheapest layer in the lightest encoder.
            let mut cheap_idx = 0usize;
            let mut cheap_c = f64::INFINITY;
            for (j, l) in layers.iter().enumerate() {
                let c = layer_capacity(l);
                if c < cheap_c {
                    cheap_c = c;
                    cheap_idx = j;
                }
            }
            if cheap_c >= filler_cap {
                // The lightest encoder is already all-filler; cannot improve.
                break;
            }
            layers[cheap_idx] = filler;
        }
    }

    /// Single-point crossover of two layer sequences. The result inherits
    /// the prefix of `a` and the suffix of `b`, yielding a sequence the
    /// length of `b` (which is valid when `b` came from a valid parent).
    fn crossover_layers<R: Rng>(
        a: &[MultimodalLayer],
        b: &[MultimodalLayer],
        rng: &mut Random<R>,
    ) -> Vec<MultimodalLayer> {
        if a.is_empty() {
            return b.to_vec();
        }
        if b.is_empty() {
            return a.to_vec();
        }
        let min_len = a.len().min(b.len());
        if min_len < 2 {
            return b.to_vec();
        }
        let split = rng.gen_range(1..min_len);
        let mut v = a[..split].to_vec();
        v.extend_from_slice(&b[split..]);
        v
    }

    // -------------------------------------------------------------------
    // Public NAS operations
    // -------------------------------------------------------------------

    /// Propose a fresh, valid multimodal architecture from the search space.
    ///
    /// A random non-empty subset of the available modalities is drawn, each
    /// modality receives a freshly-sampled encoder of a shared random
    /// length, a compatible fusion operator and a random head are chosen,
    /// and a final rebalance pass restores capacity balance. The result is
    /// guaranteed to pass [`MultimodalArchitecture::validate`].
    pub fn propose<R: Rng>(&mut self, rng: &mut Random<R>) -> Result<MultimodalArchitecture> {
        self.ensure_search_space_valid()?;

        // Sample a random non-empty subset of modalities via a partial
        // Fisher-Yates shuffle.
        let mut pool: Vec<Modality> = self.search_space.available_modalities.clone();
        let n = pool.len();
        let k = rng.gen_range(1..=n);
        for i in 0..k {
            let j = rng.gen_range(i..n);
            pool.swap(i, j);
        }
        pool.truncate(k);

        // Shared encoder length across modalities aids capacity balance.
        let enc_len =
            rng.gen_range(self.search_space.min_encoder_len..=self.search_space.max_encoder_len);

        let mut encoders = Vec::with_capacity(k);
        for &modality in &pool {
            encoders.push(self.build_encoder(modality, enc_len, rng)?);
        }

        // Restore capacity balance.
        let target = self.rebalance_target();
        self.rebalance(&mut encoders, target);

        // Compatible fusion operator.
        let fusion = self.pick_compatible_fusion(k, rng);

        // Head.
        let head_len =
            rng.gen_range(self.search_space.min_head_len..=self.search_space.max_head_len);
        let mut head = Vec::with_capacity(head_len);
        for _ in 0..head_len {
            match Self::sample_layer(&self.search_space.head_layers, rng) {
                Some(layer) => head.push(layer),
                None => {
                    return Err(OptimError::SearchSpaceError(
                        "failed to sample head layer".to_string(),
                    ))
                }
            }
        }

        let model_id = self.allocate_model_id();
        let arch = MultimodalArchitecture {
            modalities: pool,
            encoders,
            fusion,
            head,
            parameters: HashMap::new(),
            model_id,
        };

        arch.validate_with(self.search_space.max_capacity_ratio)
            .map_err(|e| OptimError::ArchitectureError(e.to_string()))?;
        Ok(arch)
    }

    /// Mutate an architecture into a valid neighbour.
    ///
    /// One of four edits is applied: substitute an encoder layer with a
    /// different candidate for the same modality; perturb an encoder
    /// layer's hyperparameters; swap the fusion operator for another
    /// compatible one; or edit a head layer. The modality set is preserved,
    /// and a final rebalance pass keeps the architecture valid.
    pub fn mutate<R: Rng>(
        &self,
        arch: &MultimodalArchitecture,
        rng: &mut Random<R>,
    ) -> Result<MultimodalArchitecture> {
        self.ensure_search_space_valid()?;
        if arch.encoders.is_empty() {
            return Err(OptimError::ArchitectureError(
                "cannot mutate an architecture with no encoders".to_string(),
            ));
        }

        let mut encoders = arch.encoders.clone();
        let mut fusion = arch.fusion;
        let mut head = arch.head.clone();

        let choice = rng.gen_range(0.0_f64..1.0);
        if choice < SUBSTITUTE_PROB {
            // Substitute a random encoder layer with a different candidate.
            let enc_idx = rng.gen_range(0..encoders.len());
            let modality = encoders[enc_idx].modality;
            let layer_count = encoders[enc_idx].layers.len();
            if layer_count > 0 {
                let pos = rng.gen_range(0..layer_count);
                let candidates = self.candidate_layers_for(modality);
                let original = encoders[enc_idx].layers[pos];
                let mut replacement = Self::sample_layer(&candidates, rng).unwrap_or(original);
                for _ in 0..8 {
                    if replacement != original {
                        break;
                    }
                    replacement = Self::sample_layer(&candidates, rng).unwrap_or(original);
                }
                encoders[enc_idx].layers[pos] = replacement;
            }
        } else if choice < PERTURB_PROB {
            // Perturb a random encoder layer's hyperparameters.
            let enc_idx = rng.gen_range(0..encoders.len());
            let layer_count = encoders[enc_idx].layers.len();
            if layer_count > 0 {
                let pos = rng.gen_range(0..layer_count);
                encoders[enc_idx].layers[pos] =
                    Self::perturb_layer(encoders[enc_idx].layers[pos], rng);
            }
        } else if choice < FUSION_MUTATE_PROB {
            // Swap the fusion operator for another compatible one.
            fusion = self.pick_compatible_fusion(arch.modalities.len(), rng);
        } else {
            // Edit a head layer (substitute or perturb).
            if !head.is_empty() {
                let pos = rng.gen_range(0..head.len());
                if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    if let Some(layer) = Self::sample_layer(&self.search_space.head_layers, rng) {
                        head[pos] = layer;
                    }
                } else {
                    head[pos] = Self::perturb_layer(head[pos], rng);
                }
            }
        }

        // Restore capacity balance after the structural edit.
        let target = self.rebalance_target();
        self.rebalance(&mut encoders, target);

        let mutated = MultimodalArchitecture {
            modalities: arch.modalities.clone(),
            encoders,
            fusion,
            head,
            parameters: arch.parameters.clone(),
            model_id: format!("{}_mut", arch.model_id),
        };
        mutated
            .validate_with(self.search_space.max_capacity_ratio)
            .map_err(|e| OptimError::ArchitectureError(e.to_string()))?;
        Ok(mutated)
    }

    /// Apply a bounded hyperparameter perturbation to a single layer.
    ///
    /// The mutation preserves the layer variant (and therefore its
    /// modality), so an `ImageConv` stays an `ImageConv`. Sizes drift by a
    /// bounded factor and never collapse below a small floor.
    fn perturb_layer<R: Rng>(layer: MultimodalLayer, rng: &mut Random<R>) -> MultimodalLayer {
        let scale = |rng: &mut Random<R>| -> f64 {
            if rng.gen_range(0.0_f64..1.0) < 0.5 {
                0.75
            } else {
                1.25
            }
        };
        match layer {
            MultimodalLayer::ImageConv { channels, kernel } => {
                let new_channels = ((f64::from(channels)) * scale(rng)).round().max(1.0) as u32;
                MultimodalLayer::ImageConv {
                    channels: new_channels,
                    kernel,
                }
            }
            MultimodalLayer::ImagePool { kernel } => {
                let new_kernel = if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    kernel.saturating_sub(1).max(2)
                } else {
                    (kernel + 1).min(9)
                };
                MultimodalLayer::ImagePool { kernel: new_kernel }
            }
            MultimodalLayer::TextEmbed { dim } => {
                let new_dim = ((f64::from(dim)) * scale(rng)).round().max(8.0) as u32;
                MultimodalLayer::TextEmbed { dim: new_dim }
            }
            MultimodalLayer::TextTransformer { heads, dim } => {
                if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    let new_heads = if rng.gen_range(0.0_f64..1.0) < 0.5 && heads > 1 {
                        heads - 1
                    } else {
                        heads + 1
                    };
                    MultimodalLayer::TextTransformer {
                        heads: new_heads,
                        dim,
                    }
                } else {
                    let new_dim = ((f64::from(dim)) * scale(rng)).round().max(8.0) as u32;
                    MultimodalLayer::TextTransformer {
                        heads,
                        dim: new_dim,
                    }
                }
            }
            MultimodalLayer::AudioConv { channels, kernel } => {
                let new_channels = ((f64::from(channels)) * scale(rng)).round().max(1.0) as u32;
                MultimodalLayer::AudioConv {
                    channels: new_channels,
                    kernel,
                }
            }
            MultimodalLayer::AudioRecurrent { hidden } => {
                let new_hidden = ((f64::from(hidden)) * scale(rng)).round().max(8.0) as u32;
                MultimodalLayer::AudioRecurrent { hidden: new_hidden }
            }
            MultimodalLayer::Dense { out } => {
                let new_out = ((f64::from(out)) * scale(rng)).round().max(8.0) as u32;
                MultimodalLayer::Dense { out: new_out }
            }
            MultimodalLayer::Norm => MultimodalLayer::Norm,
            MultimodalLayer::Activation { kind } => {
                let next = match rng.gen_range(0_u32..4) {
                    0 => ActivationKind::Relu,
                    1 => ActivationKind::Gelu,
                    2 => ActivationKind::Tanh,
                    _ => ActivationKind::Sigmoid,
                };
                // Avoid a no-op when possible by nudging to the next kind.
                let chosen = if next == kind {
                    match kind {
                        ActivationKind::Relu => ActivationKind::Gelu,
                        ActivationKind::Gelu => ActivationKind::Tanh,
                        ActivationKind::Tanh => ActivationKind::Sigmoid,
                        ActivationKind::Sigmoid => ActivationKind::Relu,
                    }
                } else {
                    next
                };
                MultimodalLayer::Activation { kind: chosen }
            }
        }
    }

    /// Recombine two parents into a valid offspring.
    ///
    /// The child's modality set is the union of the parents'. For a
    /// modality owned by both parents the encoder is a single-point
    /// crossover of the two; for a modality owned by one parent that
    /// encoder is inherited. A compatible fusion operator and a
    /// crossed-over head complete the child, and a final rebalance pass
    /// restores capacity balance.
    pub fn crossover<R: Rng>(
        &self,
        a: &MultimodalArchitecture,
        b: &MultimodalArchitecture,
        rng: &mut Random<R>,
    ) -> Result<MultimodalArchitecture> {
        self.ensure_search_space_valid()?;
        if a.modalities.is_empty() || b.modalities.is_empty() {
            return Err(OptimError::ArchitectureError(
                "cannot cross over an architecture with no modalities".to_string(),
            ));
        }

        // Union of declared modalities (a first, then b's extras).
        let mut child_modalities: Vec<Modality> = a.modalities.clone();
        for &m in &b.modalities {
            if !child_modalities.contains(&m) {
                child_modalities.push(m);
            }
        }

        let mut encoders = Vec::with_capacity(child_modalities.len());
        for &m in &child_modalities {
            let a_enc = a.encoder_for(m);
            let b_enc = b.encoder_for(m);
            let layers = match (a_enc, b_enc) {
                (Some(ea), Some(eb)) => Self::crossover_layers(&ea.layers, &eb.layers, rng),
                (Some(ea), None) => ea.layers.clone(),
                (None, Some(eb)) => eb.layers.clone(),
                (None, None) => {
                    // Neither parent supplied an encoder for a declared
                    // modality (only reachable from malformed parents);
                    // synthesise a fresh minimal encoder.
                    self.build_encoder(m, self.search_space.min_encoder_len, rng)?
                        .layers
                }
            };
            encoders.push(ModalityEncoder {
                modality: m,
                layers,
            });
        }

        // Restore capacity balance.
        let target = self.rebalance_target();
        self.rebalance(&mut encoders, target);

        // Fusion: prefer a compatible parent fusion, else resample.
        let k = child_modalities.len();
        let a_ok = fusion_supports(&a.fusion, k);
        let b_ok = fusion_supports(&b.fusion, k);
        let fusion = if a_ok && b_ok {
            if rng.gen_range(0.0_f64..1.0) < 0.5 {
                a.fusion
            } else {
                b.fusion
            }
        } else if a_ok {
            a.fusion
        } else if b_ok {
            b.fusion
        } else {
            self.pick_compatible_fusion(k, rng)
        };

        // Head: single-point crossover (falls back to a non-empty parent).
        let mut head = Self::crossover_layers(&a.head, &b.head, rng);
        if head.is_empty() {
            head = if !a.head.is_empty() {
                a.head.clone()
            } else {
                b.head.clone()
            };
        }

        // Merge parameter maps (parent b wins on conflict).
        let mut parameters = a.parameters.clone();
        for (key, value) in &b.parameters {
            parameters.insert(key.clone(), *value);
        }

        let child = MultimodalArchitecture {
            modalities: child_modalities,
            encoders,
            fusion,
            head,
            parameters,
            model_id: format!("{}_x_{}", a.model_id, b.model_id),
        };
        child
            .validate_with(self.search_space.max_capacity_ratio)
            .map_err(|e| OptimError::ArchitectureError(e.to_string()))?;
        Ok(child)
    }

    // -------------------------------------------------------------------
    // Evaluation tracking
    // -------------------------------------------------------------------

    /// Record an evaluation result for a previously-proposed architecture.
    ///
    /// Validates that `accuracy ∈ [0, 1]`, `latency_ms ≥ 0`, and
    /// `memory_mb ≥ 0` (all finite). Updates the best-so-far record when
    /// the new evaluation has a strictly higher accuracy than the current
    /// best.
    pub fn record_evaluation(
        &mut self,
        arch: MultimodalArchitecture,
        eval: MultimodalEvaluation,
    ) -> Result<()> {
        if !eval.accuracy.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "accuracy must be finite, got {}",
                eval.accuracy
            )));
        }
        if !(0.0..=1.0).contains(&eval.accuracy) {
            return Err(OptimError::InvalidParameter(format!(
                "accuracy must lie in [0, 1], got {}",
                eval.accuracy
            )));
        }
        if !eval.latency_ms.is_finite() || eval.latency_ms < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "latency_ms must be non-negative finite, got {}",
                eval.latency_ms
            )));
        }
        if !eval.memory_mb.is_finite() || eval.memory_mb < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "memory_mb must be non-negative finite, got {}",
                eval.memory_mb
            )));
        }

        if eval.accuracy > self.best_score {
            self.best_score = eval.accuracy;
            self.best_model = Some(arch.clone());
        }
        self.evaluated.push((arch, eval));
        Ok(())
    }

    /// Compute the Pareto front over `(accuracy, latency_ms, memory_mb)`.
    ///
    /// Accuracy is maximised while latency and memory are minimised. An
    /// architecture is *dominated* if another evaluated architecture is no
    /// worse on every axis and strictly better on at least one. The
    /// returned vector borrows from [`Self::evaluated`] and contains only
    /// non-dominated architectures.
    pub fn pareto_front(&self) -> Vec<&MultimodalArchitecture> {
        let mut front: Vec<&MultimodalArchitecture> = Vec::new();
        for (i, (arch, eval)) in self.evaluated.iter().enumerate() {
            let mut dominated = false;
            for (j, (_, other)) in self.evaluated.iter().enumerate() {
                if i == j {
                    continue;
                }
                if Self::dominates(other, eval) {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                front.push(arch);
            }
        }
        front
    }

    /// Return `true` iff `a` strictly dominates `b` on
    /// `(accuracy↑, latency_ms↓, memory_mb↓)`.
    fn dominates(a: &MultimodalEvaluation, b: &MultimodalEvaluation) -> bool {
        let no_worse =
            a.accuracy >= b.accuracy && a.latency_ms <= b.latency_ms && a.memory_mb <= b.memory_mb;
        let strictly_better =
            a.accuracy > b.accuracy || a.latency_ms < b.latency_ms || a.memory_mb < b.memory_mb;
        no_worse && strictly_better
    }
}
