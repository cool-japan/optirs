//! The [`MultimodalSearchSpace`] describing which multimodal architectures
//! can be built, plus its rich default instantiation.

use super::architecture::DEFAULT_MAX_CAPACITY_RATIO;
use super::types::{ActivationKind, FusionOp, Modality, MultimodalLayer};
use serde::{Deserialize, Serialize};

/// Search space describing which multimodal architectures can be built.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalSearchSpace {
    /// Modalities the search may sample from.
    pub available_modalities: Vec<Modality>,
    /// Candidate layers for the image encoder.
    pub image_layers: Vec<MultimodalLayer>,
    /// Candidate layers for the text encoder.
    pub text_layers: Vec<MultimodalLayer>,
    /// Candidate layers for the audio encoder.
    pub audio_layers: Vec<MultimodalLayer>,
    /// Generic layers usable in any encoder and as the heaviest filler.
    pub generic_layers: Vec<MultimodalLayer>,
    /// Fusion operators the search may sample from.
    pub fusion_ops: Vec<FusionOp>,
    /// Candidate layers for the post-fusion head.
    pub head_layers: Vec<MultimodalLayer>,
    /// Minimum encoder length per modality (inclusive).
    pub min_encoder_len: usize,
    /// Maximum encoder length per modality (inclusive).
    pub max_encoder_len: usize,
    /// Minimum head length (inclusive).
    pub min_head_len: usize,
    /// Maximum head length (inclusive).
    pub max_head_len: usize,
    /// Maximum allowed max/min capacity ratio across modalities.
    pub max_capacity_ratio: f64,
    /// Hard upper bound on end-to-end inference latency.
    pub max_latency_ms: f64,
    /// Hard upper bound on resident model memory.
    pub max_memory_mb: f64,
}

impl Default for MultimodalSearchSpace {
    fn default() -> Self {
        // Image encoder alphabet: convolutions at four widths x three
        // kernels plus two pooling windows.
        let mut image_layers: Vec<MultimodalLayer> = Vec::new();
        for &channels in &[16u32, 32, 64, 128] {
            for &kernel in &[3u32, 5, 7] {
                image_layers.push(MultimodalLayer::ImageConv { channels, kernel });
            }
        }
        for &kernel in &[2u32, 3] {
            image_layers.push(MultimodalLayer::ImagePool { kernel });
        }

        // Text encoder alphabet: embeddings at four widths plus transformer
        // blocks. Transformer capacities (heads x dim) top out at 256 to
        // stay within the global per-layer ceiling.
        let mut text_layers: Vec<MultimodalLayer> = Vec::new();
        for &dim in &[32u32, 64, 128, 256] {
            text_layers.push(MultimodalLayer::TextEmbed { dim });
        }
        for &heads in &[2u32, 4] {
            for &dim in &[32u32, 64] {
                text_layers.push(MultimodalLayer::TextTransformer { heads, dim });
            }
        }

        // Audio encoder alphabet: convolutions plus recurrent cells.
        let mut audio_layers: Vec<MultimodalLayer> = Vec::new();
        for &channels in &[16u32, 32, 64, 128] {
            for &kernel in &[3u32, 5] {
                audio_layers.push(MultimodalLayer::AudioConv { channels, kernel });
            }
        }
        for &hidden in &[64u32, 128, 256] {
            audio_layers.push(MultimodalLayer::AudioRecurrent { hidden });
        }

        // Generic layers. The heaviest (Dense { out: 256 }) matches the
        // global per-layer capacity ceiling so any modality can always be
        // re-balanced up to any other.
        let mut generic_layers: Vec<MultimodalLayer> = Vec::new();
        for &out in &[32u32, 64, 128, 256] {
            generic_layers.push(MultimodalLayer::Dense { out });
        }
        generic_layers.push(MultimodalLayer::Norm);
        for &kind in &[
            ActivationKind::Relu,
            ActivationKind::Gelu,
            ActivationKind::Tanh,
        ] {
            generic_layers.push(MultimodalLayer::Activation { kind });
        }

        // Head alphabet (post-fusion): a few dense widths plus
        // normalisation / activation.
        let mut head_layers: Vec<MultimodalLayer> = Vec::new();
        for &out in &[64u32, 128, 256] {
            head_layers.push(MultimodalLayer::Dense { out });
        }
        head_layers.push(MultimodalLayer::Norm);
        for &kind in &[ActivationKind::Relu, ActivationKind::Gelu] {
            head_layers.push(MultimodalLayer::Activation { kind });
        }

        let fusion_ops = vec![
            FusionOp::EarlyConcat,
            FusionOp::LateFusion,
            FusionOp::CrossAttention { heads: 4, dim: 64 },
            FusionOp::Gated { dim: 128 },
            FusionOp::Bilinear { dim: 128 },
        ];

        Self {
            available_modalities: vec![Modality::Image, Modality::Text, Modality::Audio],
            image_layers,
            text_layers,
            audio_layers,
            generic_layers,
            fusion_ops,
            head_layers,
            min_encoder_len: 2,
            max_encoder_len: 5,
            min_head_len: 1,
            max_head_len: 3,
            max_capacity_ratio: DEFAULT_MAX_CAPACITY_RATIO,
            max_latency_ms: 300.0,
            max_memory_mb: 4096.0,
        }
    }
}
