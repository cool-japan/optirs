//! Core value types for the multimodal NAS module: modalities, layer and
//! fusion-operator enums, the per-modality encoder, and the helper
//! functions that classify layers and fusion operators.

use serde::{Deserialize, Serialize};
use std::fmt;

// =======================================================================
// Modalities
// =======================================================================

/// Input modality consumed by a multimodal architecture.
///
/// `Modality` is `Ord` so that declared-modality lists have a stable,
/// deterministic ordering and so the type can key ordered collections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual input (pixels / feature maps).
    Image,
    /// Textual / token input.
    Text,
    /// Audio waveform / spectrogram input.
    Audio,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::Image => write!(f, "image"),
            Modality::Text => write!(f, "text"),
            Modality::Audio => write!(f, "audio"),
        }
    }
}

// =======================================================================
// Activation kinds
// =======================================================================

/// Non-linear activation applied by [`MultimodalLayer::Activation`].
///
/// Kept as a fieldless enum so [`MultimodalLayer`] can derive `Eq` / `Hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationKind {
    /// Rectified linear unit.
    Relu,
    /// Gaussian error linear unit.
    Gelu,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid.
    Sigmoid,
}

// =======================================================================
// Layers
// =======================================================================

/// Encoder / head layer types available in multimodal architectures.
///
/// Each variant carries the hyperparameters that influence its compute /
/// memory footprint. Concrete fields use `u32` (rather than `usize`)
/// because configurations are persisted to JSON and compared for equality
/// across machines with different pointer widths.
///
/// Layers fall into three groups:
///
/// * *Image* encoders ([`MultimodalLayer::ImageConv`],
///   [`MultimodalLayer::ImagePool`]),
/// * *Text* encoders ([`MultimodalLayer::TextEmbed`],
///   [`MultimodalLayer::TextTransformer`]),
/// * *Audio* encoders ([`MultimodalLayer::AudioConv`],
///   [`MultimodalLayer::AudioRecurrent`]),
///
/// plus the *generic* layers ([`MultimodalLayer::Dense`],
/// [`MultimodalLayer::Norm`], [`MultimodalLayer::Activation`]) which may
/// appear in any modality's encoder and in the head. The mapping from a
/// layer to its owning modality (or `None` for generic layers) is exposed
/// by [`layer_modality`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultimodalLayer {
    /// 2D convolution for the image encoder.
    ImageConv {
        /// Number of output channels.
        channels: u32,
        /// Square convolution kernel size.
        kernel: u32,
    },
    /// Spatial pooling for the image encoder.
    ImagePool {
        /// Pooling window size.
        kernel: u32,
    },
    /// Token-embedding layer for the text encoder.
    TextEmbed {
        /// Embedding dimension.
        dim: u32,
    },
    /// Transformer block for the text encoder.
    TextTransformer {
        /// Number of attention heads.
        heads: u32,
        /// Per-head feature dimension.
        dim: u32,
    },
    /// 1D convolution for the audio encoder.
    AudioConv {
        /// Number of output channels.
        channels: u32,
        /// Convolution kernel size (in frames).
        kernel: u32,
    },
    /// Recurrent cell for the audio encoder.
    AudioRecurrent {
        /// Hidden state size.
        hidden: u32,
    },
    /// Generic dense projection (usable in any encoder or in the head).
    Dense {
        /// Output feature dimension.
        out: u32,
    },
    /// Generic normalisation layer.
    Norm,
    /// Generic non-linear activation.
    Activation {
        /// The activation function applied.
        kind: ActivationKind,
    },
}

/// Return the modality a layer belongs to, or `None` for generic layers.
///
/// This is the single source of truth for encoder membership in the
/// multimodal-NAS module — [`super::MultimodalArchitecture::validate`] uses
/// it to reject foreign layers placed in the wrong encoder.
pub fn layer_modality(layer: &MultimodalLayer) -> Option<Modality> {
    match layer {
        MultimodalLayer::ImageConv { .. } | MultimodalLayer::ImagePool { .. } => {
            Some(Modality::Image)
        }
        MultimodalLayer::TextEmbed { .. } | MultimodalLayer::TextTransformer { .. } => {
            Some(Modality::Text)
        }
        MultimodalLayer::AudioConv { .. } | MultimodalLayer::AudioRecurrent { .. } => {
            Some(Modality::Audio)
        }
        MultimodalLayer::Dense { .. }
        | MultimodalLayer::Norm
        | MultimodalLayer::Activation { .. } => None,
    }
}

/// Rough compute / parameter "capacity" of a single layer.
///
/// The value is a unitless cost used solely for the modality-balance
/// constraint; it is intentionally monotone in the dominant size
/// hyperparameter of each layer. Every layer has capacity `>= 1.0`, so
/// encoder capacities are always strictly positive.
pub fn layer_capacity(layer: &MultimodalLayer) -> f64 {
    match layer {
        MultimodalLayer::ImageConv { channels, .. } => f64::from(*channels),
        MultimodalLayer::ImagePool { .. } => 8.0,
        MultimodalLayer::TextEmbed { dim } => f64::from(*dim),
        MultimodalLayer::TextTransformer { heads, dim } => f64::from(*heads) * f64::from(*dim),
        MultimodalLayer::AudioConv { channels, .. } => f64::from(*channels),
        MultimodalLayer::AudioRecurrent { hidden } => f64::from(*hidden),
        MultimodalLayer::Dense { out } => f64::from(*out),
        MultimodalLayer::Norm => 4.0,
        MultimodalLayer::Activation { .. } => 1.0,
    }
}

// =======================================================================
// Fusion operators
// =======================================================================

/// Fusion operators combining the per-modality encoder outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionOp {
    /// Early fusion: concatenate encoder features before the head.
    EarlyConcat,
    /// Late fusion: combine per-modality predictions (e.g. averaging).
    LateFusion,
    /// Cross-modal attention between encoder streams.
    CrossAttention {
        /// Number of attention heads.
        heads: u32,
        /// Per-head feature dimension.
        dim: u32,
    },
    /// Gated fusion: one modality gates another.
    Gated {
        /// Gate / projection dimension.
        dim: u32,
    },
    /// Bilinear pooling of exactly two modality features.
    Bilinear {
        /// Bilinear projection dimension.
        dim: u32,
    },
}

/// Minimum number of modalities a fusion operator can combine.
pub fn fusion_min_modalities(fusion: &FusionOp) -> usize {
    match fusion {
        FusionOp::EarlyConcat | FusionOp::LateFusion => 1,
        FusionOp::CrossAttention { .. } | FusionOp::Gated { .. } | FusionOp::Bilinear { .. } => 2,
    }
}

/// Maximum number of modalities a fusion operator can combine, if bounded.
pub fn fusion_max_modalities(fusion: &FusionOp) -> Option<usize> {
    match fusion {
        FusionOp::Bilinear { .. } => Some(2),
        _ => None,
    }
}

/// Return `true` iff `fusion` is compatible with `num_modalities` inputs.
pub fn fusion_supports(fusion: &FusionOp, num_modalities: usize) -> bool {
    let min_m = fusion_min_modalities(fusion);
    match fusion_max_modalities(fusion) {
        None => num_modalities >= min_m,
        Some(max_m) => num_modalities >= min_m && num_modalities <= max_m,
    }
}

// =======================================================================
// Modality encoder
// =======================================================================

/// A per-modality encoder: the modality it encodes and its layer sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityEncoder {
    /// The modality this encoder consumes.
    pub modality: Modality,
    /// Layers applied in execution order.
    pub layers: Vec<MultimodalLayer>,
}

/// Sum of the per-layer capacities of an encoder.
pub fn encoder_capacity(encoder: &ModalityEncoder) -> f64 {
    encoder.layers.iter().map(layer_capacity).sum()
}
