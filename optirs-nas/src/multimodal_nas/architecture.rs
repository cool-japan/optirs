//! The [`MultimodalArchitecture`] description, its structured validation
//! error type, and the [`MultimodalArchitecture::validate`] logic enforcing
//! the modality / fusion / capacity-balance constraints.

use super::types::{
    encoder_capacity, fusion_supports, layer_modality, FusionOp, Modality, ModalityEncoder,
    MultimodalLayer,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Default maximum max/min encoder-capacity ratio across modalities.
pub const DEFAULT_MAX_CAPACITY_RATIO: f64 = 8.0;

// =======================================================================
// Validation errors
// =======================================================================

/// Structured validity error returned by
/// [`MultimodalArchitecture::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MultimodalValidationError {
    /// No modality is declared.
    NoModalities,
    /// The same modality is declared more than once.
    DuplicateModality {
        /// The duplicated modality.
        modality: Modality,
    },
    /// A declared modality has no corresponding encoder (fusion would run
    /// before that modality is encoded).
    MissingEncoder {
        /// The modality whose encoder is absent.
        modality: Modality,
    },
    /// A declared modality's encoder is present but empty.
    EmptyEncoder {
        /// The modality whose encoder has no layers.
        modality: Modality,
    },
    /// An encoder contains a layer belonging to a different modality.
    ForeignLayerInEncoder {
        /// The modality of the encoder.
        modality: Modality,
        /// The offending foreign layer.
        layer: MultimodalLayer,
    },
    /// An encoder is present for a modality that was not declared.
    UndeclaredEncoder {
        /// The undeclared modality.
        modality: Modality,
    },
    /// The post-fusion head has no layers.
    EmptyHead,
    /// The fusion operator is incompatible with the modality count.
    FusionModalityMismatch {
        /// The fusion operator.
        fusion: FusionOp,
        /// The number of declared modalities.
        modalities: usize,
    },
    /// One modality's encoder dwarfs another beyond the allowed ratio.
    CapacityImbalance {
        /// The observed max/min capacity ratio.
        ratio: f64,
        /// The maximum allowed ratio.
        max_ratio: f64,
    },
}

impl fmt::Display for MultimodalValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultimodalValidationError::NoModalities => {
                write!(f, "architecture declares no modalities")
            }
            MultimodalValidationError::DuplicateModality { modality } => {
                write!(f, "modality {} is declared more than once", modality)
            }
            MultimodalValidationError::MissingEncoder { modality } => {
                write!(f, "modality {} has no encoder before fusion", modality)
            }
            MultimodalValidationError::EmptyEncoder { modality } => {
                write!(f, "modality {} has an empty encoder", modality)
            }
            MultimodalValidationError::ForeignLayerInEncoder { modality, layer } => {
                write!(
                    f,
                    "encoder for {} contains foreign layer {:?}",
                    modality, layer
                )
            }
            MultimodalValidationError::UndeclaredEncoder { modality } => {
                write!(f, "encoder present for undeclared modality {}", modality)
            }
            MultimodalValidationError::EmptyHead => write!(f, "post-fusion head is empty"),
            MultimodalValidationError::FusionModalityMismatch { fusion, modalities } => write!(
                f,
                "fusion {:?} is incompatible with {} modalities",
                fusion, modalities
            ),
            MultimodalValidationError::CapacityImbalance { ratio, max_ratio } => write!(
                f,
                "modality capacity ratio {:.3} exceeds maximum {:.3}",
                ratio, max_ratio
            ),
        }
    }
}

impl std::error::Error for MultimodalValidationError {}

// =======================================================================
// Architecture
// =======================================================================

/// A concrete multimodal architecture produced by the NAS engine.
///
/// The architecture is structured: `modalities` declares which modalities
/// are present, `encoders` carries one encoder sub-sequence per declared
/// modality, `fusion` combines them, and `head` runs after fusion.
/// `parameters` carries free-form scalar hyperparameters and `model_id` is
/// the engine-assigned identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalArchitecture {
    /// Declared modalities present in this architecture.
    pub modalities: Vec<Modality>,
    /// Per-modality encoder sub-sequences (one per declared modality).
    pub encoders: Vec<ModalityEncoder>,
    /// Fusion operator applied after all encoders.
    pub fusion: FusionOp,
    /// Post-fusion head layers in execution order.
    pub head: Vec<MultimodalLayer>,
    /// Free-form scalar hyperparameters keyed by name.
    pub parameters: HashMap<String, f64>,
    /// Engine-assigned model identifier.
    pub model_id: String,
}

impl MultimodalArchitecture {
    /// Validate the architecture using the default capacity ratio
    /// ([`DEFAULT_MAX_CAPACITY_RATIO`]).
    pub fn validate(&self) -> std::result::Result<(), MultimodalValidationError> {
        self.validate_with(DEFAULT_MAX_CAPACITY_RATIO)
    }

    /// Validate the architecture against an explicit capacity ratio.
    ///
    /// Checks, in order: at least one modality; no duplicate modalities;
    /// every declared modality owns exactly one non-empty encoder whose
    /// layers are all compatible with that modality; no encoder for an
    /// undeclared modality; a non-empty head; a fusion operator compatible
    /// with the modality count; and rough capacity balance across
    /// modalities.
    pub fn validate_with(
        &self,
        max_capacity_ratio: f64,
    ) -> std::result::Result<(), MultimodalValidationError> {
        // 1. At least one modality.
        if self.modalities.is_empty() {
            return Err(MultimodalValidationError::NoModalities);
        }

        // 2. No duplicate modalities.
        for (i, &m) in self.modalities.iter().enumerate() {
            if self.modalities[i + 1..].contains(&m) {
                return Err(MultimodalValidationError::DuplicateModality { modality: m });
            }
        }

        // 3. Each declared modality owns exactly one non-empty, well-typed
        //    encoder.
        for &m in &self.modalities {
            let matching: Vec<&ModalityEncoder> =
                self.encoders.iter().filter(|e| e.modality == m).collect();
            match matching.as_slice() {
                [] => return Err(MultimodalValidationError::MissingEncoder { modality: m }),
                [enc] => {
                    if enc.layers.is_empty() {
                        return Err(MultimodalValidationError::EmptyEncoder { modality: m });
                    }
                    for &layer in &enc.layers {
                        if let Some(owner) = layer_modality(&layer) {
                            if owner != m {
                                return Err(MultimodalValidationError::ForeignLayerInEncoder {
                                    modality: m,
                                    layer,
                                });
                            }
                        }
                    }
                }
                _ => return Err(MultimodalValidationError::DuplicateModality { modality: m }),
            }
        }

        // 4. No encoder for an undeclared modality.
        for enc in &self.encoders {
            if !self.modalities.contains(&enc.modality) {
                return Err(MultimodalValidationError::UndeclaredEncoder {
                    modality: enc.modality,
                });
            }
        }

        // 5. Non-empty head.
        if self.head.is_empty() {
            return Err(MultimodalValidationError::EmptyHead);
        }

        // 6. Fusion compatible with the modality count.
        let num_modalities = self.modalities.len();
        if !fusion_supports(&self.fusion, num_modalities) {
            return Err(MultimodalValidationError::FusionModalityMismatch {
                fusion: self.fusion,
                modalities: num_modalities,
            });
        }

        // 7. Rough capacity balance across modalities.
        if num_modalities >= 2 {
            let mut min_c = f64::INFINITY;
            let mut max_c = f64::NEG_INFINITY;
            for &m in &self.modalities {
                if let Some(enc) = self.encoders.iter().find(|e| e.modality == m) {
                    let c = encoder_capacity(enc);
                    if c < min_c {
                        min_c = c;
                    }
                    if c > max_c {
                        max_c = c;
                    }
                }
            }
            if min_c > 0.0 && max_c.is_finite() {
                let ratio = max_c / min_c;
                if ratio > max_capacity_ratio {
                    return Err(MultimodalValidationError::CapacityImbalance {
                        ratio,
                        max_ratio: max_capacity_ratio,
                    });
                }
            }
        }

        Ok(())
    }

    /// Borrow the encoder for a given modality, if present.
    pub fn encoder_for(&self, modality: Modality) -> Option<&ModalityEncoder> {
        self.encoders.iter().find(|e| e.modality == modality)
    }
}
