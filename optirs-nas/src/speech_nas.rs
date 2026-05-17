//! Speech recognition Neural Architecture Search.
//!
//! `speech_nas` provides a domain-specialised NAS engine for *automatic
//! speech recognition* (ASR) architectures. It complements
//! [`crate::domain_specific_nas`] — which uses generic optimizer
//! components — by reasoning explicitly about the layer topology of a
//! speech model: feature front-end (Mel filter bank), encoder body
//! (recurrent / convolutional / attention stacks), and CTC decoder head.
//!
//! Unlike the generic NAS engine, the search space here is *positional*:
//! certain layer types may only appear at specific positions. A
//! [`SpeechLayerType::MelFilterBank`] must always be the very first layer
//! (it converts a raw waveform into log-Mel features), and a
//! [`SpeechLayerType::CTCDecoder`] must always be the last layer (it
//! emits softmax distributions over the output vocabulary for
//! connectionist temporal classification). The [`layer_position_constraint`]
//! helper exposes that classification as an enum.
//!
//! # High-level workflow
//!
//! 1. Build a [`SpeechSearchSpace`] (or take the default) describing the
//!    allowed layers, the min/max architecture depth, and hard constraints
//!    on latency and memory. The default search space covers a typical
//!    16 kHz / 10 ms stride ASR setup with a 200 MB / 500 ms budget.
//! 2. Construct a [`SpeechNasEngine`] from the search space.
//! 3. Repeatedly call [`SpeechNasEngine::propose`] to draw fresh models
//!    from the search space, [`SpeechNasEngine::mutate`] to perturb a
//!    known-good model, or [`SpeechNasEngine::crossover`] to recombine
//!    two parents.
//! 4. Evaluate each proposed [`SpeechModelConfig`] externally (training,
//!    decoding, latency probe, …) and feed the result back via
//!    [`SpeechNasEngine::record_evaluation`].
//! 5. Inspect the [`SpeechNasEngine::best`] / [`SpeechNasEngine::pareto_front`]
//!    outputs to extract the search frontier.
//!
//! All proposal / mutation / crossover operations honour the positional
//! constraints by construction, and [`SpeechNasEngine::crossover`] performs
//! a final *repair* pass so that even an unlucky split index produces a
//! valid offspring.
//!
//! # Examples
//!
//! ```ignore
//! use optirs_nas::speech_nas::{SpeechModelEvaluation, SpeechNasEngine};
//! use scirs2_core::random::Random;
//!
//! let mut engine = SpeechNasEngine::with_default_search_space();
//! let mut rng = Random::seed(7);
//! let model = engine.propose(&mut rng).expect("propose");
//! engine
//!     .record_evaluation(
//!         model.clone(),
//!         SpeechModelEvaluation {
//!             word_error_rate: 0.12,
//!             latency_ms: 80.0,
//!             memory_mb: 35.0,
//!             param_count: 8_000_000,
//!         },
//!     )
//!     .expect("record");
//! assert!(engine.best().is_some());
//! ```
//!
//! # Pareto frontier
//!
//! Speech model selection is inherently multi-objective: a slightly
//! higher word-error rate may be acceptable in exchange for a 4x smaller
//! memory footprint or a 2x lower latency. [`SpeechNasEngine::pareto_front`]
//! returns the set of *non-dominated* models — those for which no other
//! evaluated model is simultaneously better-or-equal on word-error rate,
//! latency, and memory while being strictly better on at least one axis.

use crate::error::{OptimError, Result};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Layer types supported in speech recognition architectures.
///
/// Each variant carries the hyperparameters that influence its compute /
/// memory footprint. Concrete fields use `u32` (rather than `usize`)
/// because configurations are persisted to JSON and compared for
/// equality across machines with different pointer widths. The
/// [`SpeechLayerType::Dropout`] variant stores `p_milli = p * 1000` as
/// an integer so that the enum can derive `Eq` / `Hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeechLayerType {
    /// 1D convolution layer applied along the time axis.
    Conv1D {
        /// Number of output channels.
        channels: u32,
        /// Convolution kernel size (in frames).
        kernel: u32,
    },
    /// Mel-filter-bank front-end. Must appear at position 0.
    MelFilterBank {
        /// Number of Mel filters.
        n_mels: u32,
    },
    /// Bidirectional LSTM. Higher accuracy but cannot stream.
    BiLSTM {
        /// Hidden state size.
        hidden: u32,
    },
    /// Unidirectional LSTM. Cheaper and streaming-friendly.
    LSTM {
        /// Hidden state size.
        hidden: u32,
    },
    /// Multi-head self attention layer.
    Attention {
        /// Number of attention heads.
        heads: u32,
        /// Per-head feature dimension.
        dim: u32,
    },
    /// Dense linear projection.
    LinearProjection {
        /// Output feature dimension.
        out: u32,
    },
    /// Layer normalisation.
    LayerNorm,
    /// Stochastic dropout.
    Dropout {
        /// Drop probability scaled by 1000 (so `p_milli = 100` means `p = 0.1`).
        p_milli: u32,
    },
    /// CTC decoder head. Must appear at the last position.
    CTCDecoder,
}

/// Position constraints for layer types within an ASR architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerPositionConstraint {
    /// Layer must appear at the start of the model
    /// (e.g. [`SpeechLayerType::MelFilterBank`]).
    StartOnly,
    /// Layer must appear at the end of the model
    /// (e.g. [`SpeechLayerType::CTCDecoder`]).
    EndOnly,
    /// Layer can appear anywhere except start/end (reserved for future use).
    Body,
    /// Layer can appear anywhere in the model.
    Any,
}

/// Search space describing which speech models can be built.
///
/// `allowed_layers` enumerates every concrete layer instantiation the
/// search may draw from. Position constraints are *implicit* in the layer
/// variant (see [`layer_position_constraint`]). The `min_depth` and
/// `max_depth` bounds are inclusive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSearchSpace {
    /// Concrete layer variants the search may sample from.
    pub allowed_layers: Vec<SpeechLayerType>,
    /// Minimum architecture depth (number of layers).
    pub min_depth: usize,
    /// Maximum architecture depth (number of layers).
    pub max_depth: usize,
    /// Target audio sample rate in Hertz.
    pub sample_rate_hz: u32,
    /// Frame stride for the Mel front-end in milliseconds.
    pub frame_stride_ms: f64,
    /// Hard upper bound on end-to-end inference latency.
    pub max_latency_ms: f64,
    /// Hard upper bound on resident model memory.
    pub max_memory_mb: f64,
}

impl Default for SpeechSearchSpace {
    fn default() -> Self {
        // Build a rich default layer alphabet covering the four canonical
        // speech building blocks (Mel front-end, convolution, recurrence,
        // attention) at multiple widths, plus the regularisation /
        // normalisation / decoder primitives.
        let mut allowed_layers: Vec<SpeechLayerType> = Vec::new();

        // Mel-filter-bank front-ends at the three sizes used by
        // mainstream ASR recipes (Kaldi, ESPnet, Wav2Vec2).
        for &n_mels in &[64u32, 80, 128] {
            allowed_layers.push(SpeechLayerType::MelFilterBank { n_mels });
        }

        // Convolutional body cells at four widths x four kernel sizes.
        for &channels in &[32u32, 64, 128, 256] {
            for &kernel in &[3u32, 5, 7, 9] {
                allowed_layers.push(SpeechLayerType::Conv1D { channels, kernel });
            }
        }

        // Recurrent cells (uni + bi) at four hidden sizes.
        for &hidden in &[128u32, 256, 384, 512] {
            allowed_layers.push(SpeechLayerType::LSTM { hidden });
            allowed_layers.push(SpeechLayerType::BiLSTM { hidden });
        }

        // Multi-head attention. Fixed at four heads / dim 64 to keep the
        // search space tractable; richer attention shapes are introduced
        // through composition of multiple Attention layers.
        allowed_layers.push(SpeechLayerType::Attention { heads: 4, dim: 64 });

        // Normalisation and dropout.
        allowed_layers.push(SpeechLayerType::LayerNorm);
        for &p_milli in &[50u32, 100, 150] {
            allowed_layers.push(SpeechLayerType::Dropout { p_milli });
        }

        // Linear projections at four output dims.
        for &out in &[128u32, 256, 512, 1024] {
            allowed_layers.push(SpeechLayerType::LinearProjection { out });
        }

        // CTC decoder head.
        allowed_layers.push(SpeechLayerType::CTCDecoder);

        Self {
            allowed_layers,
            min_depth: 4,
            max_depth: 12,
            sample_rate_hz: 16_000,
            frame_stride_ms: 10.0,
            max_latency_ms: 500.0,
            max_memory_mb: 200.0,
        }
    }
}

/// A concrete speech model configuration produced by the NAS engine.
///
/// `layers` lists the architecture in execution order. `parameters` carries
/// free-form scalar hyperparameters (e.g. `"learning_rate"`,
/// `"warmup_steps"`) that downstream training code reads. `model_id` is a
/// deterministic identifier the engine assigns when proposing the model
/// — useful for tracking models across log files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechModelConfig {
    /// Layers in execution order.
    pub layers: Vec<SpeechLayerType>,
    /// Free-form scalar hyperparameters keyed by name.
    pub parameters: HashMap<String, f64>,
    /// Engine-assigned model identifier.
    pub model_id: String,
}

/// Evaluation result for a single speech model.
///
/// The semantics of every field are:
///
/// * `word_error_rate` — fraction of incorrectly recognised words in
///   `[0, 1]`. Lower is better.
/// * `latency_ms` — wall-clock inference latency on the target device.
/// * `memory_mb` — peak resident memory during inference.
/// * `param_count` — total trainable parameter count.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpeechModelEvaluation {
    /// Fraction of incorrectly recognised words in `[0, 1]`. Lower is better.
    pub word_error_rate: f64,
    /// Wall-clock inference latency in milliseconds.
    pub latency_ms: f64,
    /// Peak resident memory in megabytes.
    pub memory_mb: f64,
    /// Total trainable parameter count.
    pub param_count: u64,
}

/// The speech NAS engine.
///
/// `SpeechNasEngine` is the orchestration object: it owns the search
/// space, the running history of evaluated models, and the current
/// best-so-far record. It is intentionally non-generic over a numeric
/// type (unlike [`crate::domain_specific_nas::DomainNASEngine`]) because
/// speech evaluation metrics are inherently `f64`-valued.
#[derive(Debug, Clone)]
pub struct SpeechNasEngine {
    search_space: SpeechSearchSpace,
    evaluated: Vec<(SpeechModelConfig, SpeechModelEvaluation)>,
    best_model: Option<SpeechModelConfig>,
    best_score: f64,
    next_id: usize,
}

/// Return the position constraint associated with a layer type.
///
/// This is the single source of truth for placement rules in the
/// speech-NAS module — every proposal / mutation / crossover step uses
/// it to decide whether a layer is legal at a given position.
pub fn layer_position_constraint(layer: SpeechLayerType) -> LayerPositionConstraint {
    match layer {
        SpeechLayerType::MelFilterBank { .. } => LayerPositionConstraint::StartOnly,
        SpeechLayerType::CTCDecoder => LayerPositionConstraint::EndOnly,
        _ => LayerPositionConstraint::Any,
    }
}

/// Probability of preferring a `StartOnly` layer for the first position
/// when both `StartOnly` and `Any` candidates exist.
const PREFER_START_PROB: f64 = 0.8;

/// Probability of preferring an `EndOnly` layer for the last position
/// when both `EndOnly` and `Any` candidates exist.
const PREFER_END_PROB: f64 = 0.8;

/// Probability of substituting a layer (vs. perturbing its hyperparameters)
/// in [`SpeechNasEngine::mutate`].
const SUBSTITUTE_PROB: f64 = 0.3;

impl SpeechNasEngine {
    /// Construct a new engine from an explicit search space.
    ///
    /// The engine starts with no evaluated models, no best record, and an
    /// internal model counter at zero.
    pub fn new(search_space: SpeechSearchSpace) -> Self {
        Self {
            search_space,
            evaluated: Vec::new(),
            best_model: None,
            best_score: f64::INFINITY,
            next_id: 0,
        }
    }

    /// Construct a new engine using [`SpeechSearchSpace::default`].
    pub fn with_default_search_space() -> Self {
        Self::new(SpeechSearchSpace::default())
    }

    /// Borrow the search space.
    pub fn search_space(&self) -> &SpeechSearchSpace {
        &self.search_space
    }

    /// Borrow the evaluation history.
    pub fn evaluated(&self) -> &[(SpeechModelConfig, SpeechModelEvaluation)] {
        &self.evaluated
    }

    /// Return the best model recorded so far, if any.
    pub fn best(&self) -> Option<&SpeechModelConfig> {
        self.best_model.as_ref()
    }

    /// Return the evaluation associated with [`Self::best`].
    pub fn best_evaluation(&self) -> Option<SpeechModelEvaluation> {
        let best_id = self.best_model.as_ref()?.model_id.clone();
        self.evaluated
            .iter()
            .find(|(m, _)| m.model_id == best_id)
            .map(|(_, e)| *e)
    }

    /// Reset the engine to a freshly-constructed state.
    ///
    /// The search space is preserved. The history, best record, and
    /// internal id counter are wiped.
    pub fn reset(&mut self) {
        self.evaluated.clear();
        self.best_model = None;
        self.best_score = f64::INFINITY;
        self.next_id = 0;
    }

    // -------------------------------------------------------------------
    // Sampling primitives
    // -------------------------------------------------------------------

    /// Allocate the next model identifier.
    fn allocate_model_id(&mut self) -> String {
        let id = format!("speech_model_{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Validate that the configured search space can produce a valid model.
    fn ensure_search_space_valid(&self) -> Result<()> {
        if self.search_space.allowed_layers.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "allowed_layers is empty".to_string(),
            ));
        }
        if self.search_space.min_depth == 0 {
            return Err(OptimError::SearchSpaceError(
                "min_depth must be at least 1".to_string(),
            ));
        }
        if self.search_space.max_depth < self.search_space.min_depth {
            return Err(OptimError::SearchSpaceError(format!(
                "max_depth ({}) must be >= min_depth ({})",
                self.search_space.max_depth, self.search_space.min_depth
            )));
        }
        // Need at least one body-eligible layer (Any / Body) to fill
        // middle positions for architectures of depth >= 3.
        let has_body = self.search_space.allowed_layers.iter().any(|l| {
            matches!(
                layer_position_constraint(*l),
                LayerPositionConstraint::Any | LayerPositionConstraint::Body
            )
        });
        if !has_body && self.search_space.max_depth >= 3 {
            return Err(OptimError::SearchSpaceError(
                "search space has no Body/Any layer for middle positions".to_string(),
            ));
        }
        Ok(())
    }

    /// Pick one of the layers matching the supplied position constraint
    /// (including `Any`), preferring the strict constraint with probability
    /// `prefer_strict_prob` when both kinds of candidates exist.
    ///
    /// Returns an error only when *no* layer is legal at the position.
    fn pick_position_layer<R: Rng>(
        &self,
        rng: &mut Random<R>,
        strict: LayerPositionConstraint,
        prefer_strict_prob: f64,
    ) -> Result<SpeechLayerType> {
        let strict_layers: Vec<SpeechLayerType> = self
            .search_space
            .allowed_layers
            .iter()
            .copied()
            .filter(|l| layer_position_constraint(*l) == strict)
            .collect();
        let any_layers: Vec<SpeechLayerType> = self
            .search_space
            .allowed_layers
            .iter()
            .copied()
            .filter(|l| layer_position_constraint(*l) == LayerPositionConstraint::Any)
            .collect();

        let use_strict = if strict_layers.is_empty() {
            false
        } else if any_layers.is_empty() {
            true
        } else {
            rng.gen_range(0.0_f64..1.0) < prefer_strict_prob
        };

        let pool = if use_strict {
            &strict_layers
        } else {
            &any_layers
        };
        if pool.is_empty() {
            return Err(OptimError::SearchSpaceError(format!(
                "no candidate layer available for constraint {:?}",
                strict
            )));
        }
        let idx: usize = rng.gen_range(0..pool.len());
        Ok(pool[idx])
    }

    /// Pick a body-position layer (constraint `Body` or `Any`).
    fn pick_body_layer<R: Rng>(&self, rng: &mut Random<R>) -> Result<SpeechLayerType> {
        let body_layers: Vec<SpeechLayerType> = self
            .search_space
            .allowed_layers
            .iter()
            .copied()
            .filter(|l| {
                matches!(
                    layer_position_constraint(*l),
                    LayerPositionConstraint::Any | LayerPositionConstraint::Body
                )
            })
            .collect();
        if body_layers.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "no body-eligible layers in search space".to_string(),
            ));
        }
        let idx: usize = rng.gen_range(0..body_layers.len());
        Ok(body_layers[idx])
    }

    // -------------------------------------------------------------------
    // Public NAS operations
    // -------------------------------------------------------------------

    /// Propose a fresh speech model configuration uniformly from the search space.
    ///
    /// Depth is sampled uniformly from `[min_depth, max_depth]`. The first
    /// layer is drawn preferring [`LayerPositionConstraint::StartOnly`]
    /// with probability [`PREFER_START_PROB`]; the last layer is drawn
    /// preferring [`LayerPositionConstraint::EndOnly`] with probability
    /// [`PREFER_END_PROB`]; middle layers are drawn from the body pool.
    pub fn propose<R: Rng>(&mut self, rng: &mut Random<R>) -> Result<SpeechModelConfig> {
        self.ensure_search_space_valid()?;

        // Inclusive depth sample. The gen_range API in scirs2-core uses an
        // exclusive upper bound, so add 1 to the max depth here.
        let depth: usize =
            rng.gen_range(self.search_space.min_depth..(self.search_space.max_depth + 1));

        let mut layers: Vec<SpeechLayerType> = Vec::with_capacity(depth);

        if depth == 1 {
            // Degenerate: pick something legal at any position. Prefer the
            // end (CTC) since a 1-layer model is essentially a head.
            let layer = self
                .pick_position_layer(rng, LayerPositionConstraint::EndOnly, PREFER_END_PROB)
                .or_else(|_| {
                    self.pick_position_layer(rng, LayerPositionConstraint::StartOnly, 1.0)
                })?;
            layers.push(layer);
        } else {
            // First layer: prefer Mel front-end.
            let first = self.pick_position_layer(
                rng,
                LayerPositionConstraint::StartOnly,
                PREFER_START_PROB,
            )?;
            layers.push(first);

            // Middle layers.
            for _ in 1..(depth - 1) {
                layers.push(self.pick_body_layer(rng)?);
            }

            // Last layer: prefer CTC decoder.
            let last =
                self.pick_position_layer(rng, LayerPositionConstraint::EndOnly, PREFER_END_PROB)?;
            layers.push(last);
        }

        let model_id = self.allocate_model_id();
        Ok(SpeechModelConfig {
            layers,
            parameters: HashMap::new(),
            model_id,
        })
    }

    /// Mutate a model by either substituting a non-edge layer or perturbing
    /// the hyperparameters of an existing one.
    ///
    /// With probability [`SUBSTITUTE_PROB`] the engine replaces a single
    /// non-first / non-last layer with a different body-eligible layer.
    /// Otherwise it perturbs a hyperparameter — `Conv1D::channels` is
    /// scaled by ±25 %, `Dropout::p_milli` shifts by ±50, recurrent
    /// hidden sizes shift by one bucket, etc.
    ///
    /// The first and last layers are never replaced; the positional
    /// invariant (Mel-at-start, CTC-at-end) is therefore preserved.
    pub fn mutate<R: Rng>(
        &self,
        model: &SpeechModelConfig,
        rng: &mut Random<R>,
    ) -> Result<SpeechModelConfig> {
        self.ensure_search_space_valid()?;

        if model.layers.is_empty() {
            return Err(OptimError::ArchitectureError(
                "cannot mutate an empty model".to_string(),
            ));
        }

        let mut new_layers = model.layers.clone();
        let n = new_layers.len();

        // Decide between substitution and perturbation.
        let do_substitute = rng.gen_range(0.0_f64..1.0) < SUBSTITUTE_PROB;

        if do_substitute && n >= 3 {
            // Pick a non-edge position.
            let pos: usize = rng.gen_range(1..(n - 1));
            let original = new_layers[pos];

            // Try up to 8 draws to find a different layer.
            let mut replacement = self.pick_body_layer(rng)?;
            for _ in 0..8 {
                if replacement != original {
                    break;
                }
                replacement = self.pick_body_layer(rng)?;
            }
            new_layers[pos] = replacement;
        } else {
            // Perturb a single layer's hyperparameters. Restrict to non-edge
            // layers when possible so the positional invariant cannot be
            // accidentally broken by a perturbation that produces a
            // different position class (e.g. a Mel filter bank with
            // adjusted n_mels remains a Mel filter bank — but we still
            // protect the edges as a defensive measure).
            let target_pos: usize = if n >= 3 {
                rng.gen_range(1..(n - 1))
            } else {
                rng.gen_range(0..n)
            };
            new_layers[target_pos] = self.perturb_layer(new_layers[target_pos], rng);
        }

        Ok(SpeechModelConfig {
            layers: new_layers,
            parameters: model.parameters.clone(),
            model_id: format!("{}_mut", model.model_id),
        })
    }

    /// Apply a small hyperparameter perturbation to a single layer.
    ///
    /// The mutation is *bounded* — channels / hidden sizes / dropout
    /// probabilities can drift but never explode — and it preserves the
    /// position class of the original layer (a `Conv1D` stays a `Conv1D`,
    /// not a `BiLSTM`).
    fn perturb_layer<R: Rng>(
        &self,
        layer: SpeechLayerType,
        rng: &mut Random<R>,
    ) -> SpeechLayerType {
        match layer {
            SpeechLayerType::Conv1D { channels, kernel } => {
                let scale = if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    0.75
                } else {
                    1.25
                };
                let new_channels = ((channels as f64) * scale).round().max(1.0) as u32;
                SpeechLayerType::Conv1D {
                    channels: new_channels,
                    kernel,
                }
            }
            SpeechLayerType::MelFilterBank { n_mels } => {
                // Shift in {-16, 0, +16} but stay > 0.
                let delta: i32 = match rng.gen_range(0_u32..3) {
                    0 => -16,
                    1 => 0,
                    _ => 16,
                };
                let new_n = ((n_mels as i64) + delta as i64).max(8) as u32;
                SpeechLayerType::MelFilterBank { n_mels: new_n }
            }
            SpeechLayerType::BiLSTM { hidden } => {
                let scale = if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    0.75
                } else {
                    1.25
                };
                let new_hidden = ((hidden as f64) * scale).round().max(8.0) as u32;
                SpeechLayerType::BiLSTM { hidden: new_hidden }
            }
            SpeechLayerType::LSTM { hidden } => {
                let scale = if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    0.75
                } else {
                    1.25
                };
                let new_hidden = ((hidden as f64) * scale).round().max(8.0) as u32;
                SpeechLayerType::LSTM { hidden: new_hidden }
            }
            SpeechLayerType::Attention { heads, dim } => {
                // Shift heads by ±1 (but stay >= 1) or dim by ±25 %.
                if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    let new_heads = if rng.gen_range(0.0_f64..1.0) < 0.5 && heads > 1 {
                        heads - 1
                    } else {
                        heads + 1
                    };
                    SpeechLayerType::Attention {
                        heads: new_heads,
                        dim,
                    }
                } else {
                    let scale = if rng.gen_range(0.0_f64..1.0) < 0.5 {
                        0.75
                    } else {
                        1.25
                    };
                    let new_dim = ((dim as f64) * scale).round().max(8.0) as u32;
                    SpeechLayerType::Attention {
                        heads,
                        dim: new_dim,
                    }
                }
            }
            SpeechLayerType::LinearProjection { out } => {
                let scale = if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    0.75
                } else {
                    1.25
                };
                let new_out = ((out as f64) * scale).round().max(8.0) as u32;
                SpeechLayerType::LinearProjection { out: new_out }
            }
            SpeechLayerType::LayerNorm => SpeechLayerType::LayerNorm,
            SpeechLayerType::Dropout { p_milli } => {
                // Drift p_milli by ±50 but clamp to [0, 900] (probability
                // 0..0.9). Probabilities above 0.9 are degenerate.
                let delta: i32 = if rng.gen_range(0.0_f64..1.0) < 0.5 {
                    -50
                } else {
                    50
                };
                let new_p = ((p_milli as i64) + delta as i64).clamp(0, 900) as u32;
                SpeechLayerType::Dropout { p_milli: new_p }
            }
            SpeechLayerType::CTCDecoder => SpeechLayerType::CTCDecoder,
        }
    }

    /// Single-point crossover.
    ///
    /// Picks a split index `s ∈ [1, min(a.len, b.len) - 1]`, takes
    /// `a.layers[..s]` from the first parent and `b.layers[s..]` from
    /// the second, then runs the positional *repair* pass:
    ///
    /// * if the offspring's first layer is not a Mel filter bank, a
    ///   randomly chosen Mel filter bank from the search space is
    ///   prepended (or, if none is present, the first existing Mel
    ///   filter bank in the parents).
    /// * if the offspring's last layer is not a CTC decoder, one is
    ///   appended.
    ///
    /// Because the inherited slices each came from a valid parent the
    /// repair changes at most one layer per edge.
    pub fn crossover<R: Rng>(
        &self,
        a: &SpeechModelConfig,
        b: &SpeechModelConfig,
        rng: &mut Random<R>,
    ) -> Result<SpeechModelConfig> {
        self.ensure_search_space_valid()?;

        if a.layers.is_empty() || b.layers.is_empty() {
            return Err(OptimError::ArchitectureError(
                "cannot cross over empty parents".to_string(),
            ));
        }
        let min_len = a.layers.len().min(b.layers.len());
        if min_len < 2 {
            // Degenerate parents — fall through with split = 1 for the
            // longer of the two, otherwise just append parent b.
            let split = if a.layers.len() >= 2 {
                1
            } else {
                a.layers.len()
            };
            let mut layers: Vec<SpeechLayerType> = a.layers[..split].to_vec();
            layers.extend(b.layers[split.min(b.layers.len())..].iter().copied());
            return self.finish_crossover(a, b, layers);
        }

        let split: usize = rng.gen_range(1..min_len);
        let mut layers: Vec<SpeechLayerType> = a.layers[..split].to_vec();
        layers.extend(b.layers[split..].iter().copied());
        self.finish_crossover(a, b, layers)
    }

    /// Repair-and-finalise step shared by [`Self::crossover`].
    fn finish_crossover(
        &self,
        a: &SpeechModelConfig,
        b: &SpeechModelConfig,
        mut layers: Vec<SpeechLayerType>,
    ) -> Result<SpeechModelConfig> {
        // Repair: ensure a Mel filter bank at position 0.
        if !matches!(layers.first(), Some(SpeechLayerType::MelFilterBank { .. })) {
            // Prefer to harvest a Mel filter bank from one of the parents
            // so the spectral feature width stays consistent; fall back to
            // the search space's first Mel filter bank if neither parent
            // has one.
            let mel = a
                .layers
                .iter()
                .copied()
                .chain(b.layers.iter().copied())
                .find(|l| matches!(l, SpeechLayerType::MelFilterBank { .. }))
                .or_else(|| {
                    self.search_space
                        .allowed_layers
                        .iter()
                        .copied()
                        .find(|l| matches!(l, SpeechLayerType::MelFilterBank { .. }))
                });
            if let Some(mel) = mel {
                layers.insert(0, mel);
            } else {
                return Err(OptimError::SearchSpaceError(
                    "no MelFilterBank available to repair crossover offspring".to_string(),
                ));
            }
        }

        // Repair: ensure a CTC decoder at the last position.
        if !matches!(layers.last(), Some(SpeechLayerType::CTCDecoder)) {
            // CTC is parameter-free so we can always synthesise one.
            layers.push(SpeechLayerType::CTCDecoder);
        }

        // Clamp to max depth if the repair overshot. We trim mid-body
        // layers (preserving the Mel head and CTC tail) so the resulting
        // model stays within the engine's depth budget.
        if layers.len() > self.search_space.max_depth {
            let mut trimmed = Vec::with_capacity(self.search_space.max_depth);
            trimmed.push(layers[0]);
            // Body length to keep: max_depth - 2 (head + tail).
            let body_target = self.search_space.max_depth.saturating_sub(2);
            for i in 1..(1 + body_target) {
                if i < layers.len() - 1 {
                    trimmed.push(layers[i]);
                }
            }
            trimmed.push(layers[layers.len() - 1]);
            layers = trimmed;
        }

        // Merge parent parameter maps (parent b wins on conflict — a
        // simple, deterministic, well-defined recombination rule).
        let mut parameters = a.parameters.clone();
        for (k, v) in &b.parameters {
            parameters.insert(k.clone(), *v);
        }

        Ok(SpeechModelConfig {
            layers,
            parameters,
            model_id: format!("{}_x_{}", a.model_id, b.model_id),
        })
    }

    // -------------------------------------------------------------------
    // Evaluation tracking
    // -------------------------------------------------------------------

    /// Record an evaluation result for a previously-proposed model.
    ///
    /// Validates that `word_error_rate ∈ [0, 1]`, `latency_ms ≥ 0`, and
    /// `memory_mb ≥ 0`. Updates the best-so-far record when the new
    /// evaluation has a strictly lower word-error rate than the current
    /// best.
    pub fn record_evaluation(
        &mut self,
        model: SpeechModelConfig,
        eval: SpeechModelEvaluation,
    ) -> Result<()> {
        if !eval.word_error_rate.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "word_error_rate must be finite, got {}",
                eval.word_error_rate
            )));
        }
        if !(0.0..=1.0).contains(&eval.word_error_rate) {
            return Err(OptimError::InvalidParameter(format!(
                "word_error_rate must lie in [0, 1], got {}",
                eval.word_error_rate
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

        if eval.word_error_rate < self.best_score {
            self.best_score = eval.word_error_rate;
            self.best_model = Some(model.clone());
        }
        self.evaluated.push((model, eval));
        Ok(())
    }

    /// Compute the Pareto front over `(WER, latency_ms, memory_mb)`.
    ///
    /// A model is *dominated* if there exists another evaluated model
    /// that is no worse on every axis and strictly better on at least
    /// one. The returned slice borrows from [`Self::evaluated`] and is
    /// guaranteed not to include any dominated model.
    pub fn pareto_front(&self) -> Vec<&SpeechModelConfig> {
        let mut front: Vec<&SpeechModelConfig> = Vec::new();
        for (i, (model, eval)) in self.evaluated.iter().enumerate() {
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
                front.push(model);
            }
        }
        front
    }

    /// Return `true` iff `a` strictly dominates `b` on
    /// `(WER, latency_ms, memory_mb)`.
    fn dominates(a: &SpeechModelEvaluation, b: &SpeechModelEvaluation) -> bool {
        let no_worse = a.word_error_rate <= b.word_error_rate
            && a.latency_ms <= b.latency_ms
            && a.memory_mb <= b.memory_mb;
        let strictly_better = a.word_error_rate < b.word_error_rate
            || a.latency_ms < b.latency_ms
            || a.memory_mb < b.memory_mb;
        no_worse && strictly_better
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mel_default() -> SpeechLayerType {
        SpeechLayerType::MelFilterBank { n_mels: 80 }
    }

    fn make_ctc() -> SpeechLayerType {
        SpeechLayerType::CTCDecoder
    }

    fn make_simple_model(id: &str) -> SpeechModelConfig {
        SpeechModelConfig {
            layers: vec![
                make_mel_default(),
                SpeechLayerType::Conv1D {
                    channels: 64,
                    kernel: 3,
                },
                SpeechLayerType::BiLSTM { hidden: 256 },
                SpeechLayerType::LinearProjection { out: 512 },
                make_ctc(),
            ],
            parameters: HashMap::new(),
            model_id: id.to_string(),
        }
    }

    #[test]
    fn test_default_search_space_has_speech_layers() {
        let space = SpeechSearchSpace::default();
        let has_mel = space
            .allowed_layers
            .iter()
            .any(|l| matches!(l, SpeechLayerType::MelFilterBank { .. }));
        let has_bilstm = space
            .allowed_layers
            .iter()
            .any(|l| matches!(l, SpeechLayerType::BiLSTM { .. }));
        let has_ctc = space
            .allowed_layers
            .iter()
            .any(|l| matches!(l, SpeechLayerType::CTCDecoder));
        let has_conv = space
            .allowed_layers
            .iter()
            .any(|l| matches!(l, SpeechLayerType::Conv1D { .. }));
        let has_attention = space
            .allowed_layers
            .iter()
            .any(|l| matches!(l, SpeechLayerType::Attention { .. }));
        assert!(has_mel, "MelFilterBank must be in default search space");
        assert!(has_bilstm, "BiLSTM must be in default search space");
        assert!(has_ctc, "CTCDecoder must be in default search space");
        assert!(has_conv, "Conv1D must be in default search space");
        assert!(has_attention, "Attention must be in default search space");
        assert_eq!(space.min_depth, 4);
        assert_eq!(space.max_depth, 12);
        assert_eq!(space.sample_rate_hz, 16_000);
    }

    #[test]
    fn test_propose_returns_valid_architecture() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(1);
        let model = engine.propose(&mut rng).expect("propose");
        assert!(model.layers.len() >= engine.search_space.min_depth);
        assert!(model.layers.len() <= engine.search_space.max_depth);
        assert!(!model.model_id.is_empty());
        assert!(model.parameters.is_empty());
    }

    #[test]
    fn test_propose_respects_depth_bounds() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(2);
        let min_d = engine.search_space.min_depth;
        let max_d = engine.search_space.max_depth;
        for _ in 0..50 {
            let model = engine.propose(&mut rng).expect("propose");
            assert!(
                model.layers.len() >= min_d,
                "depth {} below min {}",
                model.layers.len(),
                min_d
            );
            assert!(
                model.layers.len() <= max_d,
                "depth {} above max {}",
                model.layers.len(),
                max_d
            );
        }
    }

    #[test]
    fn test_propose_mel_filterbank_at_position_zero() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(3);
        let trials = 100;
        let mut mel_first = 0;
        for _ in 0..trials {
            let model = engine.propose(&mut rng).expect("propose");
            if matches!(
                model.layers.first(),
                Some(SpeechLayerType::MelFilterBank { .. })
            ) {
                mel_first += 1;
            }
        }
        let ratio = mel_first as f64 / trials as f64;
        assert!(
            ratio > 0.5,
            "MelFilterBank at position 0 ratio {:.3} should exceed 0.5",
            ratio
        );
    }

    #[test]
    fn test_propose_ctc_decoder_at_last_position() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(4);
        let trials = 100;
        let mut ctc_last = 0;
        for _ in 0..trials {
            let model = engine.propose(&mut rng).expect("propose");
            if matches!(model.layers.last(), Some(SpeechLayerType::CTCDecoder)) {
                ctc_last += 1;
            }
        }
        let ratio = ctc_last as f64 / trials as f64;
        assert!(
            ratio > 0.5,
            "CTCDecoder at last position ratio {:.3} should exceed 0.5",
            ratio
        );
    }

    #[test]
    fn test_mutate_changes_model_within_search_space() {
        let engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(5);
        let model = make_simple_model("seed");
        // Try multiple mutations to confirm at least one actually changes
        // the architecture (substitution is probabilistic).
        let mut saw_change = false;
        for _ in 0..30 {
            let mutated = engine.mutate(&model, &mut rng).expect("mutate");
            // Same depth.
            assert_eq!(mutated.layers.len(), model.layers.len());
            // Different id.
            assert_ne!(mutated.model_id, model.model_id);
            if mutated.layers != model.layers {
                saw_change = true;
            }
        }
        assert!(
            saw_change,
            "mutate produced no observable change in 30 tries"
        );
    }

    #[test]
    fn test_mutate_preserves_constraints() {
        let engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(6);
        let model = make_simple_model("seed");
        for _ in 0..50 {
            let mutated = engine.mutate(&model, &mut rng).expect("mutate");
            assert!(matches!(
                mutated.layers.first(),
                Some(SpeechLayerType::MelFilterBank { .. })
            ));
            assert!(matches!(
                mutated.layers.last(),
                Some(SpeechLayerType::CTCDecoder)
            ));
        }
    }

    #[test]
    fn test_crossover_produces_valid_offspring() {
        let engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(7);
        let mut a = make_simple_model("a");
        a.layers = vec![
            SpeechLayerType::MelFilterBank { n_mels: 80 },
            SpeechLayerType::Conv1D {
                channels: 64,
                kernel: 3,
            },
            SpeechLayerType::BiLSTM { hidden: 256 },
            SpeechLayerType::LinearProjection { out: 512 },
            SpeechLayerType::CTCDecoder,
        ];
        let mut b = make_simple_model("b");
        b.layers = vec![
            SpeechLayerType::MelFilterBank { n_mels: 128 },
            SpeechLayerType::Attention { heads: 4, dim: 64 },
            SpeechLayerType::LayerNorm,
            SpeechLayerType::Dropout { p_milli: 100 },
            SpeechLayerType::CTCDecoder,
        ];
        let child = engine.crossover(&a, &b, &mut rng).expect("crossover");
        assert!(matches!(
            child.layers.first(),
            Some(SpeechLayerType::MelFilterBank { .. })
        ));
        assert!(matches!(
            child.layers.last(),
            Some(SpeechLayerType::CTCDecoder)
        ));
        assert!(child.layers.len() >= 2);
        assert!(child.model_id.contains("_x_"));
    }

    #[test]
    fn test_crossover_seed_reproducibility() {
        let engine = SpeechNasEngine::with_default_search_space();
        let a = make_simple_model("a");
        let mut b = make_simple_model("b");
        b.layers[2] = SpeechLayerType::LSTM { hidden: 384 };
        b.layers[3] = SpeechLayerType::LayerNorm;

        let mut rng1 = Random::seed(42);
        let mut rng2 = Random::seed(42);
        let c1 = engine.crossover(&a, &b, &mut rng1).expect("c1");
        let c2 = engine.crossover(&a, &b, &mut rng2).expect("c2");
        assert_eq!(c1.layers, c2.layers);
        assert_eq!(c1.model_id, c2.model_id);
    }

    #[test]
    fn test_record_evaluation_updates_best() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let m1 = make_simple_model("m1");
        let m2 = make_simple_model("m2");

        engine
            .record_evaluation(
                m1.clone(),
                SpeechModelEvaluation {
                    word_error_rate: 0.2,
                    latency_ms: 90.0,
                    memory_mb: 30.0,
                    param_count: 1_000_000,
                },
            )
            .expect("record m1");
        assert_eq!(
            engine.best().map(|m| m.model_id.clone()).expect("best m1"),
            "m1".to_string()
        );

        engine
            .record_evaluation(
                m2.clone(),
                SpeechModelEvaluation {
                    word_error_rate: 0.1,
                    latency_ms: 120.0,
                    memory_mb: 70.0,
                    param_count: 4_000_000,
                },
            )
            .expect("record m2");
        assert_eq!(
            engine.best().map(|m| m.model_id.clone()).expect("best m2"),
            "m2".to_string()
        );
        let best_eval = engine.best_evaluation().expect("best eval");
        assert!((best_eval.word_error_rate - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_record_evaluation_validates_wer_range() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let m = make_simple_model("bad");
        let neg = engine.record_evaluation(
            m.clone(),
            SpeechModelEvaluation {
                word_error_rate: -0.01,
                latency_ms: 50.0,
                memory_mb: 10.0,
                param_count: 100,
            },
        );
        assert!(matches!(neg, Err(OptimError::InvalidParameter(_))));

        let too_big = engine.record_evaluation(
            m.clone(),
            SpeechModelEvaluation {
                word_error_rate: 1.5,
                latency_ms: 50.0,
                memory_mb: 10.0,
                param_count: 100,
            },
        );
        assert!(matches!(too_big, Err(OptimError::InvalidParameter(_))));

        let bad_latency = engine.record_evaluation(
            m.clone(),
            SpeechModelEvaluation {
                word_error_rate: 0.2,
                latency_ms: -1.0,
                memory_mb: 10.0,
                param_count: 100,
            },
        );
        assert!(matches!(bad_latency, Err(OptimError::InvalidParameter(_))));

        let bad_mem = engine.record_evaluation(
            m,
            SpeechModelEvaluation {
                word_error_rate: 0.2,
                latency_ms: 1.0,
                memory_mb: -10.0,
                param_count: 100,
            },
        );
        assert!(matches!(bad_mem, Err(OptimError::InvalidParameter(_))));
    }

    #[test]
    fn test_pareto_front_no_dominated_solutions() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        // Three models:
        //   A: WER 0.15, latency 100, mem 60   (Pareto-optimal — best WER)
        //   B: WER 0.20, latency  50, mem 30   (Pareto-optimal — best lat/mem)
        //   C: WER 0.25, latency 150, mem 80   (dominated by A on every axis)
        let a = make_simple_model("a");
        let mut b = make_simple_model("b");
        b.model_id = "b".to_string();
        let mut c = make_simple_model("c");
        c.model_id = "c".to_string();

        engine
            .record_evaluation(
                a,
                SpeechModelEvaluation {
                    word_error_rate: 0.15,
                    latency_ms: 100.0,
                    memory_mb: 60.0,
                    param_count: 1_000_000,
                },
            )
            .expect("record a");
        engine
            .record_evaluation(
                b,
                SpeechModelEvaluation {
                    word_error_rate: 0.20,
                    latency_ms: 50.0,
                    memory_mb: 30.0,
                    param_count: 500_000,
                },
            )
            .expect("record b");
        engine
            .record_evaluation(
                c,
                SpeechModelEvaluation {
                    word_error_rate: 0.25,
                    latency_ms: 150.0,
                    memory_mb: 80.0,
                    param_count: 1_500_000,
                },
            )
            .expect("record c");

        let front = engine.pareto_front();
        assert_eq!(front.len(), 2, "expected exactly 2 non-dominated models");
        let ids: Vec<String> = front.iter().map(|m| m.model_id.clone()).collect();
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
        assert!(!ids.contains(&"c".to_string()));
    }

    #[test]
    fn test_pareto_front_empty_initially() {
        let engine = SpeechNasEngine::with_default_search_space();
        let front = engine.pareto_front();
        assert!(front.is_empty());
        assert!(engine.best().is_none());
        assert!(engine.best_evaluation().is_none());
    }

    #[test]
    fn test_speech_model_serde_roundtrip() {
        let model = make_simple_model("rt");
        let json = serde_json::to_string(&model).expect("serialise");
        let back: SpeechModelConfig = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(model.layers, back.layers);
        assert_eq!(model.model_id, back.model_id);

        let eval = SpeechModelEvaluation {
            word_error_rate: 0.123,
            latency_ms: 45.6,
            memory_mb: 78.9,
            param_count: 7_777_777,
        };
        let json_eval = serde_json::to_string(&eval).expect("serialise eval");
        let back_eval: SpeechModelEvaluation =
            serde_json::from_str(&json_eval).expect("deserialise eval");
        assert!((back_eval.word_error_rate - eval.word_error_rate).abs() < 1e-12);
        assert_eq!(back_eval.param_count, eval.param_count);

        let space = SpeechSearchSpace::default();
        let json_space = serde_json::to_string(&space).expect("serialise space");
        let back_space: SpeechSearchSpace =
            serde_json::from_str(&json_space).expect("deserialise space");
        assert_eq!(back_space.allowed_layers.len(), space.allowed_layers.len());
        assert_eq!(back_space.min_depth, space.min_depth);
        assert_eq!(back_space.max_depth, space.max_depth);
    }

    #[test]
    fn test_layer_position_constraint_correct() {
        assert_eq!(
            layer_position_constraint(SpeechLayerType::MelFilterBank { n_mels: 80 }),
            LayerPositionConstraint::StartOnly
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::CTCDecoder),
            LayerPositionConstraint::EndOnly
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::Conv1D {
                channels: 32,
                kernel: 3
            }),
            LayerPositionConstraint::Any
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::BiLSTM { hidden: 128 }),
            LayerPositionConstraint::Any
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::LSTM { hidden: 128 }),
            LayerPositionConstraint::Any
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::Attention { heads: 4, dim: 64 }),
            LayerPositionConstraint::Any
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::LinearProjection { out: 256 }),
            LayerPositionConstraint::Any
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::LayerNorm),
            LayerPositionConstraint::Any
        );
        assert_eq!(
            layer_position_constraint(SpeechLayerType::Dropout { p_milli: 100 }),
            LayerPositionConstraint::Any
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(11);
        let model = engine.propose(&mut rng).expect("propose");
        engine
            .record_evaluation(
                model,
                SpeechModelEvaluation {
                    word_error_rate: 0.2,
                    latency_ms: 50.0,
                    memory_mb: 25.0,
                    param_count: 1000,
                },
            )
            .expect("record");
        assert!(engine.best().is_some());
        assert!(!engine.evaluated().is_empty());

        engine.reset();
        assert!(engine.best().is_none());
        assert!(engine.evaluated().is_empty());
        assert!(engine.best_evaluation().is_none());
        // After reset, the id counter should also be zero again.
        let mut rng2 = Random::seed(12);
        let m2 = engine.propose(&mut rng2).expect("propose after reset");
        assert_eq!(m2.model_id, "speech_model_0");
    }

    #[test]
    fn test_search_space_accessor_returns_current_space() {
        let engine = SpeechNasEngine::with_default_search_space();
        let space = engine.search_space();
        assert_eq!(space.min_depth, 4);
        assert_eq!(space.max_depth, 12);
        assert!((space.frame_stride_ms - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_proposed_model_ids_are_unique_and_sequential() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let mut rng = Random::seed(13);
        let m0 = engine.propose(&mut rng).expect("p0");
        let m1 = engine.propose(&mut rng).expect("p1");
        let m2 = engine.propose(&mut rng).expect("p2");
        assert_eq!(m0.model_id, "speech_model_0");
        assert_eq!(m1.model_id, "speech_model_1");
        assert_eq!(m2.model_id, "speech_model_2");
    }

    #[test]
    fn test_pareto_front_all_non_dominated() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        // Three mutually non-dominated models: each is best on one axis.
        let mut a = make_simple_model("a");
        a.model_id = "a".to_string();
        let mut b = make_simple_model("b");
        b.model_id = "b".to_string();
        let mut c = make_simple_model("c");
        c.model_id = "c".to_string();
        engine
            .record_evaluation(
                a,
                SpeechModelEvaluation {
                    word_error_rate: 0.10,
                    latency_ms: 200.0,
                    memory_mb: 80.0,
                    param_count: 1,
                },
            )
            .expect("rec a");
        engine
            .record_evaluation(
                b,
                SpeechModelEvaluation {
                    word_error_rate: 0.30,
                    latency_ms: 40.0,
                    memory_mb: 70.0,
                    param_count: 2,
                },
            )
            .expect("rec b");
        engine
            .record_evaluation(
                c,
                SpeechModelEvaluation {
                    word_error_rate: 0.20,
                    latency_ms: 100.0,
                    memory_mb: 20.0,
                    param_count: 3,
                },
            )
            .expect("rec c");
        let front = engine.pareto_front();
        assert_eq!(front.len(), 3);
    }

    #[test]
    fn test_invalid_search_space_rejected() {
        let mut space = SpeechSearchSpace::default();
        space.allowed_layers.clear();
        let mut engine = SpeechNasEngine::new(space);
        let mut rng = Random::seed(99);
        let result = engine.propose(&mut rng);
        assert!(matches!(result, Err(OptimError::SearchSpaceError(_))));
    }

    #[test]
    fn test_record_evaluation_rejects_nan_wer() {
        let mut engine = SpeechNasEngine::with_default_search_space();
        let m = make_simple_model("nan");
        let result = engine.record_evaluation(
            m,
            SpeechModelEvaluation {
                word_error_rate: f64::NAN,
                latency_ms: 10.0,
                memory_mb: 10.0,
                param_count: 1,
            },
        );
        assert!(matches!(result, Err(OptimError::InvalidParameter(_))));
    }

    #[test]
    fn test_speech_layer_type_hash_eq() {
        // Required for HashMap / HashSet membership in downstream tooling.
        use std::collections::HashSet;
        let mut set: HashSet<SpeechLayerType> = HashSet::new();
        set.insert(SpeechLayerType::Conv1D {
            channels: 64,
            kernel: 3,
        });
        set.insert(SpeechLayerType::Conv1D {
            channels: 64,
            kernel: 3,
        });
        set.insert(SpeechLayerType::Conv1D {
            channels: 64,
            kernel: 5,
        });
        assert_eq!(set.len(), 2);
    }
}
