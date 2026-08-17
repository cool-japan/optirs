use std::fmt::Debug;
// Positional encoding mechanisms for transformer optimization
//
// This module implements various positional encoding strategies used in the
// transformer optimizer to provide position information to the attention mechanisms.

#[allow(dead_code)]
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng as SCRRng};

use super::super::TransformerOptimizerConfig;
use crate::error::{OptimError, Result};

/// Types of positional encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncodingType {
    /// Sinusoidal position encoding
    Sinusoidal,
    /// Learned position embedding
    Learned,
    /// Rotary position embedding (RoPE)
    Rotary,
    /// Relative position encoding
    Relative,
    /// ALiBi (Attention with Linear Biases)
    ALiBi,
}

/// Positional encoder for transformer inputs
#[derive(Debug, Clone)]
pub struct PositionalEncoder<T: Float + Debug + Send + Sync + 'static> {
    /// Encoding type
    encoding_type: PositionalEncodingType,

    /// Cached encodings
    cached_encodings: Option<Array2<T>>,

    /// Maximum sequence length
    max_seqlen: usize,

    /// Model dimension
    modeldim: usize,

    /// Learned position embeddings (if applicable)
    position_embeddings: Option<Array2<T>>,

    /// ALiBi slopes (if applicable)
    alibi_slopes: Option<Array1<T>>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> PositionalEncoder<T> {
    /// Create new positional encoder
    pub fn new(config: &TransformerOptimizerConfig) -> Result<Self> {
        let max_seqlen = config.max_sequence_length;
        let modeldim = config.modeldim;

        let mut cached_encodings = None;
        let mut position_embeddings = None;
        let mut alibi_slopes = None;

        match config.pos_encoding_type {
            PositionalEncodingType::Sinusoidal => {
                cached_encodings = Some(Self::sinusoidal_table(max_seqlen, modeldim));
            }
            PositionalEncodingType::Learned => {
                // Initialize learnable position embeddings
                let mut rng = scirs2_core::random::thread_rng();
                let mut embeddings = Array2::zeros((max_seqlen, modeldim));

                // Xavier initialization
                let bound = (6.0 / (max_seqlen + modeldim) as f64).sqrt();
                for elem in embeddings.iter_mut() {
                    *elem = scirs2_core::numeric::NumCast::from(
                        (rng.random::<f64>() - 0.5) * 2.0 * bound,
                    )
                    .unwrap_or_else(|| T::zero());
                }
                position_embeddings = Some(embeddings);
            }
            PositionalEncodingType::ALiBi => {
                // Initialize ALiBi slopes
                let numheads = config.numheads.max(1);
                let mut slopes = Array1::zeros(numheads);

                for h in 0..numheads {
                    slopes[h] = scirs2_core::numeric::NumCast::from(
                        2.0_f64.powf(-8.0 * (h + 1) as f64 / numheads as f64),
                    )
                    .unwrap_or_else(|| T::zero());
                }
                alibi_slopes = Some(slopes);
            }
            _ => {
                // Rotary and relative encodings modify the attention scores
                // rather than the inputs, but a sinusoidal table is still
                // precomputed so `compute_sinusoidal_position` stays usable.
                cached_encodings = Some(Self::sinusoidal_table(max_seqlen, modeldim));
            }
        }

        Ok(Self {
            encoding_type: config.pos_encoding_type,
            cached_encodings,
            max_seqlen,
            modeldim,
            position_embeddings,
            alibi_slopes,
        })
    }

    /// Angular frequency for dimension `i` of a `modeldim`-wide sinusoidal
    /// encoding: `1 / 10000^(2 * floor(i / 2) / d)`.
    ///
    /// The `floor(i / 2)` is what makes `(2k, 2k+1)` a genuine
    /// `(sin, cos)` pair at the *same* frequency, as in Vaswani et al. (2017).
    /// Using `i` directly - as the previous implementation did - gave the sine
    /// and cosine halves of each pair different frequencies, so the encoding
    /// was not a rotation and adjacent dimensions carried unrelated phases.
    fn inverse_frequency(i: usize, modeldim: usize) -> f64 {
        let exponent = 2.0 * ((i / 2) as f64) / modeldim.max(1) as f64;
        1.0 / 10000.0_f64.powf(exponent)
    }

    /// Build the full sinusoidal encoding table.
    fn sinusoidal_table(max_seqlen: usize, modeldim: usize) -> Array2<T> {
        let mut encodings = Array2::zeros((max_seqlen, modeldim));

        for pos in 0..max_seqlen {
            for i in 0..modeldim {
                let angle = pos as f64 * Self::inverse_frequency(i, modeldim);
                let value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                encodings[[pos, i]] =
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero());
            }
        }

        encodings
    }

    /// Encode input with positional information
    pub fn encode(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (seq_len, modeldim) = input.dim();

        if seq_len > self.max_seqlen {
            return Err(OptimError::InvalidConfig(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_seqlen
            )));
        }

        if modeldim != self.modeldim {
            return Err(OptimError::InvalidConfig(format!(
                "Model dimension {} doesn't match expected {}",
                modeldim, self.modeldim
            )));
        }

        let mut output = input.clone();

        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => {
                if let Some(ref encodings) = self.cached_encodings {
                    let pos_enc = encodings.slice(s![..seq_len, ..]);
                    output = output + pos_enc;
                }
            }
            PositionalEncodingType::Learned => {
                if let Some(ref embeddings) = self.position_embeddings {
                    let pos_emb = embeddings.slice(s![..seq_len, ..]);
                    output = output + pos_emb;
                }
            }
            PositionalEncodingType::Rotary => {
                // Rotary position embedding (RoPE) doesn't add to input,
                // it modifies attention computation
                // For now, just return input unchanged
            }
            PositionalEncodingType::Relative => {
                // Relative position encoding doesn't add to input,
                // it modifies attention computation
                // For now, just return input unchanged
            }
            PositionalEncodingType::ALiBi => {
                // ALiBi doesn't add to input, it modifies attention scores
                // For now, just return input unchanged
            }
        }

        Ok(output)
    }

    /// Get ALiBi slopes for attention bias calculation
    pub fn get_alibi_slopes(&self) -> Option<&Array1<T>> {
        self.alibi_slopes.as_ref()
    }

    /// Get encoding type
    pub fn encoding_type(&self) -> PositionalEncodingType {
        self.encoding_type
    }

    /// Get maximum sequence length
    pub fn max_sequence_length(&self) -> usize {
        self.max_seqlen
    }

    /// Get model dimension
    pub fn model_dimension(&self) -> usize {
        self.modeldim
    }

    /// Update position embeddings (for learned encoding)
    pub fn update_embeddings(&mut self, new_embeddings: Array2<T>) -> Result<()> {
        match self.encoding_type {
            PositionalEncodingType::Learned => {
                let (pos_len, model_dim) = new_embeddings.dim();
                if pos_len != self.max_seqlen || model_dim != self.modeldim {
                    return Err(OptimError::InvalidConfig(
                        "New embeddings dimensions don't match encoder configuration".to_string(),
                    ));
                }
                self.position_embeddings = Some(new_embeddings);
                Ok(())
            }
            _ => Err(OptimError::InvalidConfig(
                "Position embeddings can only be updated for learned encoding type".to_string(),
            )),
        }
    }

    /// Compute sinusoidal encoding for a specific position
    pub fn compute_sinusoidal_position(&self, position: usize) -> Result<Array1<T>> {
        if position >= self.max_seqlen {
            return Err(OptimError::InvalidConfig(
                "Position exceeds maximum sequence length".to_string(),
            ));
        }

        let mut encoding = Array1::zeros(self.modeldim);
        for i in 0..self.modeldim {
            let angle = position as f64 * Self::inverse_frequency(i, self.modeldim);
            let value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            encoding[i] = scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero());
        }

        Ok(encoding)
    }

    /// Apply ALiBi bias to attention scores
    pub fn apply_alibi_bias(
        &self,
        attention_scores: &mut Array2<T>,
        head_idx: usize,
    ) -> Result<()> {
        if self.encoding_type != PositionalEncodingType::ALiBi {
            return Ok(()); // No-op for non-ALiBi encoding
        }

        if let Some(ref slopes) = self.alibi_slopes {
            if head_idx >= slopes.len() {
                return Err(OptimError::InvalidConfig(
                    "Head index exceeds number of ALiBi slopes".to_string(),
                ));
            }

            let slope = slopes[head_idx];
            let (rows, cols) = attention_scores.dim();

            for i in 0..rows {
                for j in 0..cols {
                    let distance: T =
                        scirs2_core::numeric::NumCast::from((i as i64 - j as i64).abs() as f64)
                            .unwrap_or_else(|| T::zero());
                    attention_scores[[i, j]] = attention_scores[[i, j]] - slope * distance;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerOptimizerConfig;

    fn config(encoding: PositionalEncodingType) -> TransformerOptimizerConfig {
        TransformerOptimizerConfig {
            modeldim: 8,
            numheads: 2,
            max_sequence_length: 16,
            pos_encoding_type: encoding,
            ..Default::default()
        }
    }

    #[test]
    fn sinusoidal_pairs_share_a_frequency() {
        let encoder = PositionalEncoder::<f64>::new(&config(PositionalEncodingType::Sinusoidal))
            .expect("encoder creation");
        let encoding = encoder
            .compute_sinusoidal_position(3)
            .expect("position encoding");

        // d = 8; dimension pair (2, 3) shares the frequency 1 / 10000^(2/8).
        let omega = 1.0 / 10000.0_f64.powf(2.0 * 1.0 / 8.0);
        assert!((encoding[2] - (3.0 * omega).sin()).abs() < 1e-12);
        assert!((encoding[3] - (3.0 * omega).cos()).abs() < 1e-12);

        // sin^2 + cos^2 == 1 holds exactly for a genuine (sin, cos) pair.
        assert!((encoding[2] * encoding[2] + encoding[3] * encoding[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn position_zero_is_alternating_zero_one() {
        let encoder = PositionalEncoder::<f64>::new(&config(PositionalEncodingType::Sinusoidal))
            .expect("encoder creation");
        let encoding = encoder
            .compute_sinusoidal_position(0)
            .expect("position encoding");
        for i in 0..8 {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert!((encoding[i] - expected).abs() < 1e-12, "dim {i}");
        }
    }

    #[test]
    fn encode_adds_the_cached_table() {
        let encoder = PositionalEncoder::<f64>::new(&config(PositionalEncodingType::Sinusoidal))
            .expect("encoder creation");
        let input = Array2::<f64>::zeros((4, 8));
        let encoded = encoder.encode(&input).expect("encode");
        let reference = encoder
            .compute_sinusoidal_position(2)
            .expect("position encoding");
        for i in 0..8 {
            assert!((encoded[[2, i]] - reference[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn alibi_bias_penalizes_distance() {
        let encoder = PositionalEncoder::<f64>::new(&config(PositionalEncodingType::ALiBi))
            .expect("encoder creation");
        let mut scores = Array2::<f64>::zeros((4, 4));
        encoder
            .apply_alibi_bias(&mut scores, 0)
            .expect("alibi bias");
        assert_eq!(scores[[0, 0]], 0.0);
        assert!(
            scores[[0, 3]] < scores[[0, 1]],
            "farther must be penalized more"
        );
    }

    #[test]
    fn oversized_sequences_are_rejected() {
        let encoder = PositionalEncoder::<f64>::new(&config(PositionalEncodingType::Sinusoidal))
            .expect("encoder creation");
        let input = Array2::<f64>::zeros((64, 8));
        assert!(encoder.encode(&input).is_err());
    }
}
