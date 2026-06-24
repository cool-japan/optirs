//! Multimodal Neural Architecture Search.
//!
//! `multimodal_nas` provides a domain-specialised NAS engine for
//! *multimodal* models — architectures that ingest several input
//! modalities (image, text, audio), encode each one with a dedicated
//! per-modality sub-network, fuse the resulting representations with a
//! fusion operator, and emit a prediction through a shared post-fusion
//! head. It is the multimodal analogue of [`crate::speech_nas`].
//!
//! Unlike the generic NAS engine, the search space here is *structured*:
//! the architecture is split into one encoder sub-sequence per declared
//! modality, a single [`FusionOp`] that combines all encoders, and a head.
//! This structure makes the dominant validity rules explicit:
//!
//! * every declared modality must own a non-empty encoder *before* fusion
//!   (an empty / missing encoder means fusion would run before that
//!   modality is encoded);
//! * an encoder may only contain layers compatible with its modality
//!   (generic [`MultimodalLayer::Dense`] / [`MultimodalLayer::Norm`] /
//!   [`MultimodalLayer::Activation`] layers are allowed everywhere);
//! * the chosen fusion operator must be compatible with the number of
//!   modalities present (for example [`FusionOp::Bilinear`] pools exactly
//!   two modalities, while [`FusionOp::CrossAttention`] needs at least
//!   two);
//! * no modality's encoder may *dwarf* the others beyond a configurable
//!   capacity ratio (rough compute / parameter balance).
//!
//! All of these are checked by [`MultimodalArchitecture::validate`], which
//! returns a structured [`MultimodalValidationError`].
//!
//! # High-level workflow
//!
//! 1. Build a [`MultimodalSearchSpace`] (or take the default) describing
//!    the available modalities, the per-modality candidate encoder layers,
//!    the allowed fusion operators, the head layers, the length bounds, and
//!    the capacity-balance ratio.
//! 2. Construct a [`MultimodalNasEngine`] from the search space.
//! 3. Repeatedly call [`MultimodalNasEngine::propose`] to draw fresh
//!    architectures, [`MultimodalNasEngine::mutate`] to perturb a
//!    known-good architecture, or [`MultimodalNasEngine::crossover`] to
//!    recombine two parents. Every operation returns a `validate`-passing
//!    architecture by construction (a final *rebalance* pass restores the
//!    capacity balance after structural edits).
//! 4. Evaluate each proposed [`MultimodalArchitecture`] externally and feed
//!    the result back via [`MultimodalNasEngine::record_evaluation`].
//! 5. Inspect [`MultimodalNasEngine::best`] /
//!    [`MultimodalNasEngine::pareto_front`] to extract the search frontier.
//!
//! # Pareto frontier
//!
//! Multimodal model selection is multi-objective: higher accuracy may be
//! traded against latency or memory. [`MultimodalNasEngine::pareto_front`]
//! returns the set of *non-dominated* architectures over
//! `(accuracy, latency_ms, memory_mb)` — accuracy is *maximised* while
//! latency and memory are *minimised*.
//!
//! # Examples
//!
//! ```ignore
//! use optirs_nas::multimodal_nas::{MultimodalEvaluation, MultimodalNasEngine};
//! use scirs2_core::random::Random;
//!
//! let mut engine = MultimodalNasEngine::with_default_search_space();
//! let mut rng = Random::seed(7);
//! let arch = engine.propose(&mut rng).expect("propose");
//! assert!(arch.validate().is_ok());
//! engine
//!     .record_evaluation(
//!         arch,
//!         MultimodalEvaluation {
//!             accuracy: 0.91,
//!             latency_ms: 80.0,
//!             memory_mb: 140.0,
//!             param_count: 12_000_000,
//!         },
//!     )
//!     .expect("record");
//! assert!(engine.best().is_some());
//! ```

mod architecture;
mod engine;
mod search_space;
mod types;

pub use architecture::{
    MultimodalArchitecture, MultimodalValidationError, DEFAULT_MAX_CAPACITY_RATIO,
};
pub use engine::{MultimodalEvaluation, MultimodalNasEngine};
pub use search_space::MultimodalSearchSpace;
pub use types::{
    encoder_capacity, fusion_max_modalities, fusion_min_modalities, fusion_supports,
    layer_capacity, layer_modality, ActivationKind, FusionOp, Modality, ModalityEncoder,
    MultimodalLayer,
};

#[cfg(test)]
mod tests;
