// Positional encoding implementations for transformer sequences

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Angular frequency of dimension `i` for a `model_dimension`-wide sinusoidal
/// encoding with the given base: `1 / base^(2 * floor(i / 2) / d)`.
///
/// The `floor(i / 2)` makes `(2k, 2k+1)` a real `(sin, cos)` pair at one shared
/// frequency, as in Vaswani et al. (2017).
fn inverse_frequency(i: usize, model_dimension: usize, base: f64) -> f64 {
    let exponent = 2.0 * ((i / 2) as f64) / model_dimension.max(1) as f64;
    1.0 / base.powf(exponent)
}

/// Positional encoding types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding (original Transformer)
    Sinusoidal,
    /// Learned positional encoding
    Learned,
    /// Rotary Position Embedding (RoPE)
    Rotary,
    /// No positional encoding
    None,
}

/// Positional encoding implementation
pub struct PositionalEncoding<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Type of positional encoding
    encoding_type: PositionalEncodingType,

    /// Maximum sequence length
    max_sequence_length: usize,

    /// Model dimension
    model_dimension: usize,

    /// Precomputed encoding matrix
    encoding_matrix: Array2<T>,

    /// Learned parameters (for learned encoding)
    learned_embeddings: Option<Array2<T>>,

    /// RoPE frequency base
    rope_base: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    PositionalEncoding<T>
{
    /// Create new positional encoding
    pub fn new(
        max_sequence_length: usize,
        model_dimension: usize,
        encoding_type: PositionalEncodingType,
    ) -> Result<Self> {
        let rope_base = scirs2_core::numeric::NumCast::from(10000.0).unwrap_or_else(|| T::zero());
        let mut encoding = Self {
            encoding_type,
            max_sequence_length,
            model_dimension,
            encoding_matrix: Array2::zeros((max_sequence_length, model_dimension)),
            learned_embeddings: None,
            rope_base,
        };

        encoding.initialize_encoding()?;
        Ok(encoding)
    }

    /// Initialize the encoding based on type
    fn initialize_encoding(&mut self) -> Result<()> {
        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => self.initialize_sinusoidal(),
            PositionalEncodingType::Learned => self.initialize_learned(),
            PositionalEncodingType::Rotary => self.initialize_rotary(),
            PositionalEncodingType::None => Ok(()),
        }
    }

    /// Initialize sinusoidal positional encoding
    fn initialize_sinusoidal(&mut self) -> Result<()> {
        let base = self.rope_base.to_f64().unwrap_or(10000.0);
        self.fill_sinusoidal(base);
        Ok(())
    }

    /// Fill the encoding matrix with a sinusoidal table for the given base.
    fn fill_sinusoidal(&mut self, base: f64) {
        for pos in 0..self.max_sequence_length {
            for i in 0..self.model_dimension {
                let angle = pos as f64 * inverse_frequency(i, self.model_dimension, base);
                let value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                self.encoding_matrix[[pos, i]] =
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero());
            }
        }
    }

    /// Initialize learned positional encoding.
    ///
    /// Embeddings are seeded from the sinusoidal table rather than from zeros,
    /// so an untrained learned encoding still carries position information
    /// instead of collapsing every position onto the same (zero) vector.
    fn initialize_learned(&mut self) -> Result<()> {
        let base = self.rope_base.to_f64().unwrap_or(10000.0);
        self.fill_sinusoidal(base);
        self.learned_embeddings = Some(self.encoding_matrix.clone());
        Ok(())
    }

    /// Initialize rotary positional encoding.
    ///
    /// RoPE has no additive table; row 0 of the encoding matrix caches the
    /// inverse frequency shared by each `(2k, 2k+1)` coordinate pair.
    fn initialize_rotary(&mut self) -> Result<()> {
        let base = self.rope_base.to_f64().unwrap_or(10000.0);
        self.encoding_matrix.fill(T::zero());
        for i in (0..self.model_dimension).step_by(2) {
            let freq: T = scirs2_core::numeric::NumCast::from(inverse_frequency(
                i,
                self.model_dimension,
                base,
            ))
            .unwrap_or_else(|| T::zero());
            self.encoding_matrix[[0, i]] = freq;
            if i + 1 < self.model_dimension {
                self.encoding_matrix[[0, i + 1]] = freq;
            }
        }
        Ok(())
    }

    /// Validate that `input` is a `(sequence_length, model_dimension)` matrix
    /// that fits within the configured maximum sequence length.
    fn validate_input(&self, input: &Array2<T>) -> Result<(usize, usize)> {
        let (sequence_length, model_dimension) = input.dim();
        if model_dimension != self.model_dimension {
            return Err(OptimError::InvalidConfig(format!(
                "Input width {model_dimension} does not match the model dimension {}",
                self.model_dimension
            )));
        }
        if sequence_length > self.max_sequence_length {
            return Err(OptimError::InvalidConfig(format!(
                "Sequence length {sequence_length} exceeds the maximum {}",
                self.max_sequence_length
            )));
        }
        Ok((sequence_length, model_dimension))
    }

    /// Apply positional encoding to input
    pub fn encode(&self, input: &Array2<T>) -> Result<Array2<T>> {
        match self.encoding_type {
            PositionalEncodingType::None => Ok(input.clone()),
            PositionalEncodingType::Sinusoidal => self.apply_sinusoidal(input),
            PositionalEncodingType::Learned => self.apply_learned(input),
            PositionalEncodingType::Rotary => self.apply_rotary(input),
        }
    }

    /// Apply sinusoidal encoding elementwise: `output[pos, dim] += table[pos, dim]`.
    fn apply_sinusoidal(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (sequence_length, model_dimension) = self.validate_input(input)?;
        let mut output = input.clone();

        for pos in 0..sequence_length {
            for dim in 0..model_dimension {
                output[[pos, dim]] = output[[pos, dim]] + self.encoding_matrix[[pos, dim]];
            }
        }

        Ok(output)
    }

    /// Apply learned encoding elementwise.
    fn apply_learned(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let Some(ref learned) = self.learned_embeddings else {
            return Err(OptimError::InvalidState(
                "Learned embeddings not initialized".to_string(),
            ));
        };

        let (sequence_length, model_dimension) = self.validate_input(input)?;
        let mut output = input.clone();

        for pos in 0..sequence_length {
            for dim in 0..model_dimension {
                output[[pos, dim]] = output[[pos, dim]] + learned[[pos, dim]];
            }
        }

        Ok(output)
    }

    /// Apply rotary position embedding.
    ///
    /// Each `(2k, 2k+1)` coordinate pair of every row is rotated by
    /// `theta = position * inv_freq_k`:
    /// `x' = x cos(theta) - y sin(theta)`, `y' = x sin(theta) + y cos(theta)`.
    /// This is a genuine rotation, so it preserves the norm of every pair.
    fn apply_rotary(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (sequence_length, model_dimension) = self.validate_input(input)?;
        let mut output = input.clone();

        for pos in 0..sequence_length {
            let position: T =
                scirs2_core::numeric::NumCast::from(pos as f64).unwrap_or_else(|| T::zero());

            for i in (0..model_dimension.saturating_sub(1)).step_by(2) {
                let freq = self.encoding_matrix[[0, i]];
                let angle = position * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let x = input[[pos, i]];
                let y = input[[pos, i + 1]];

                output[[pos, i]] = x * cos_val - y * sin_val;
                output[[pos, i + 1]] = x * sin_val + y * cos_val;
            }
        }

        Ok(output)
    }

    /// Get encoding for specific position
    pub fn get_position_encoding(&self, position: usize) -> Result<Array1<T>> {
        if position >= self.max_sequence_length {
            return Err(crate::error::OptimError::Other(
                "Position exceeds maximum sequence length".to_string(),
            ));
        }

        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => Ok(self.encoding_matrix.row(position).to_owned()),
            PositionalEncodingType::Learned => {
                if let Some(ref learned) = self.learned_embeddings {
                    Ok(learned.row(position).to_owned())
                } else {
                    Err(crate::error::OptimError::Other(
                        "Learned embeddings not available".to_string(),
                    ))
                }
            }
            PositionalEncodingType::Rotary => {
                // Return frequency information for RoPE
                Ok(self.encoding_matrix.row(0).to_owned())
            }
            PositionalEncodingType::None => Ok(Array1::zeros(self.model_dimension)),
        }
    }

    /// Update learned embeddings (for training)
    pub fn update_learned_embeddings(&mut self, gradients: &Array2<T>) -> Result<()> {
        let Some(ref mut learned) = self.learned_embeddings else {
            return Err(OptimError::InvalidState(
                "No learned embeddings to update".to_string(),
            ));
        };
        if learned.dim() != gradients.dim() {
            return Err(OptimError::InvalidConfig(format!(
                "Gradient shape {:?} does not match the embedding shape {:?}",
                gradients.dim(),
                learned.dim()
            )));
        }
        *learned = &*learned - gradients;
        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        match self.encoding_type {
            PositionalEncodingType::Learned => self.max_sequence_length * self.model_dimension,
            _ => 0, // Sinusoidal, RoPE, and None have no learnable parameters
        }
    }

    /// Reset parameters
    pub fn reset(&mut self) -> Result<()> {
        self.initialize_encoding()
    }

    /// Get encoding type
    pub fn get_encoding_type(&self) -> PositionalEncodingType {
        self.encoding_type
    }

    /// Get maximum sequence length
    pub fn get_max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    /// Get model dimension
    pub fn get_model_dimension(&self) -> usize {
        self.model_dimension
    }

    /// Create sinusoidal encoding with custom base
    pub fn sinusoidal_with_base(
        max_sequence_length: usize,
        model_dimension: usize,
        base: T,
    ) -> Result<Self> {
        let mut encoding = Self::new(
            max_sequence_length,
            model_dimension,
            PositionalEncodingType::Sinusoidal,
        )?;

        // Reinitialize with custom base
        encoding.rope_base = base;
        encoding.fill_sinusoidal(base.to_f64().unwrap_or(10000.0));

        Ok(encoding)
    }

    /// Create rotary encoding with custom base
    pub fn rotary_with_base(
        max_sequence_length: usize,
        model_dimension: usize,
        base: T,
    ) -> Result<Self> {
        let mut encoding = Self::new(
            max_sequence_length,
            model_dimension,
            PositionalEncodingType::Rotary,
        )?;

        encoding.rope_base = base;
        encoding.initialize_rotary()?;

        Ok(encoding)
    }
}

/// Relative positional encoding for local attention patterns
pub struct RelativePositionalEncoding<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Maximum relative distance
    max_relative_distance: usize,

    /// Model dimension
    model_dimension: usize,

    /// Relative encoding table
    relative_encoding: Array2<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    RelativePositionalEncoding<T>
{
    /// Create new relative positional encoding
    pub fn new(max_relative_distance: usize, model_dimension: usize) -> Result<Self> {
        let table_size = 2 * max_relative_distance + 1;
        let relative_encoding = Array2::zeros((table_size, model_dimension));

        Ok(Self {
            max_relative_distance,
            model_dimension,
            relative_encoding,
        })
    }

    /// Get relative encoding between two positions
    pub fn get_relative_encoding(&self, from_pos: usize, to_pos: usize) -> Array1<T> {
        let relative_distance = (to_pos as i32 - from_pos as i32)
            .max(-(self.max_relative_distance as i32))
            .min(self.max_relative_distance as i32);

        let index = (relative_distance + self.max_relative_distance as i32) as usize;
        self.relative_encoding.row(index).to_owned()
    }

    /// Initialize with sinusoidal patterns
    pub fn initialize_sinusoidal(&mut self) -> Result<()> {
        let table_size = 2 * self.max_relative_distance + 1;

        for i in 0..table_size {
            let relative_pos = i as i32 - self.max_relative_distance as i32;
            let position =
                scirs2_core::numeric::NumCast::from(relative_pos).unwrap_or_else(|| T::zero());

            let _ = position;
            for j in 0..self.model_dimension {
                let angle =
                    relative_pos as f64 * inverse_frequency(j, self.model_dimension, 10000.0);
                let value = if j % 2 == 0 { angle.sin() } else { angle.cos() };
                self.relative_encoding[[i, j]] =
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero());
            }
        }

        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        (2 * self.max_relative_distance + 1) * self.model_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_pairs_share_a_frequency() {
        let pe = PositionalEncoding::<f64>::new(64, 8, PositionalEncodingType::Sinusoidal)
            .expect("encoding creation");
        let row = pe.get_position_encoding(3).expect("position encoding");

        // d = 8; the pair (2, 3) shares the frequency 1 / 10000^(2/8).
        let omega = 1.0 / 10000.0_f64.powf(2.0 * 1.0 / 8.0);
        assert!((row[2] - (3.0 * omega).sin()).abs() < 1e-12);
        assert!((row[3] - (3.0 * omega).cos()).abs() < 1e-12);
        assert!((row[2] * row[2] + row[3] * row[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn position_zero_alternates_zero_and_one() {
        let pe = PositionalEncoding::<f64>::new(16, 8, PositionalEncodingType::Sinusoidal)
            .expect("encoding creation");
        let row = pe.get_position_encoding(0).expect("position encoding");
        for i in 0..8 {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert!((row[i] - expected).abs() < 1e-12, "dim {i} was {}", row[i]);
        }
    }

    #[test]
    fn sinusoidal_encoding_adds_per_cell_not_per_row() {
        let pe = PositionalEncoding::<f64>::new(16, 8, PositionalEncodingType::Sinusoidal)
            .expect("encoding creation");
        let input = Array2::<f64>::zeros((4, 8));
        let encoded = pe.encode(&input).expect("encode");
        assert_eq!(encoded.dim(), (4, 8));

        for pos in 0..4 {
            let reference = pe.get_position_encoding(pos).expect("position encoding");
            for dim in 0..8 {
                assert!(
                    (encoded[[pos, dim]] - reference[dim]).abs() < 1e-12,
                    "cell ({pos}, {dim}) was {}",
                    encoded[[pos, dim]]
                );
            }
        }
        // Distinct positions must produce distinct rows.
        assert!(encoded.row(0) != encoded.row(1));
    }

    #[test]
    fn wrong_width_is_rejected() {
        let pe = PositionalEncoding::<f64>::new(16, 8, PositionalEncodingType::Sinusoidal)
            .expect("encoding creation");
        assert!(pe.encode(&Array2::<f64>::zeros((4, 5))).is_err());
        assert!(pe.encode(&Array2::<f64>::zeros((32, 8))).is_err());
    }

    #[test]
    fn rotary_encoding_is_a_rotation() {
        let pe = PositionalEncoding::<f64>::new(16, 4, PositionalEncodingType::Rotary)
            .expect("encoding creation");
        let input = Array2::<f64>::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 1.0, //
                2.0, 1.0, -1.0, 3.0, //
                0.5, 0.5, 0.5, 0.5,
            ],
        )
        .expect("valid shape");

        let output = pe.encode(&input).expect("encode");
        assert_eq!(output.dim(), (3, 4));

        for pos in 0..3 {
            for pair in 0..2 {
                let before = input[[pos, 2 * pair]].powi(2) + input[[pos, 2 * pair + 1]].powi(2);
                let after = output[[pos, 2 * pair]].powi(2) + output[[pos, 2 * pair + 1]].powi(2);
                assert!(
                    (before - after).abs() < 1e-10,
                    "pair {pair} at position {pos} changed norm: {before} -> {after}"
                );
            }
        }

        // Position 0 is the identity rotation.
        for dim in 0..4 {
            assert!((output[[0, dim]] - input[[0, dim]]).abs() < 1e-12);
        }
        // Later positions really rotate.
        assert!((output[[2, 0]] - input[[2, 0]]).abs() > 1e-9);
    }

    #[test]
    fn rotary_matches_hand_computed_values() {
        let pe = PositionalEncoding::<f64>::new(16, 2, PositionalEncodingType::Rotary)
            .expect("encoding creation");
        // d = 2 => the single pair has inverse frequency 1.
        let input =
            Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 0.0]).expect("valid shape");
        let output = pe.encode(&input).expect("encode");
        assert!((output[[1, 0]] - 1.0_f64.cos()).abs() < 1e-12);
        assert!((output[[1, 1]] - 1.0_f64.sin()).abs() < 1e-12);
    }

    #[test]
    fn learned_encoding_starts_position_aware() {
        let pe = PositionalEncoding::<f64>::new(32, 8, PositionalEncodingType::Learned)
            .expect("encoding creation");
        assert_eq!(pe.parameter_count(), 32 * 8);
        let a = pe.get_position_encoding(1).expect("position encoding");
        let b = pe.get_position_encoding(2).expect("position encoding");
        assert!(a != b, "learned embeddings collapsed onto one vector");
    }

    #[test]
    fn no_encoding_is_the_identity() {
        let pe = PositionalEncoding::<f64>::new(16, 4, PositionalEncodingType::None)
            .expect("encoding creation");
        let input = Array2::<f64>::from_elem((3, 4), 0.25);
        assert_eq!(pe.encode(&input).expect("encode"), input);
    }

    #[test]
    fn relative_encoding_pairs_share_a_frequency() {
        let mut rel = RelativePositionalEncoding::<f64>::new(8, 8).expect("creation");
        rel.initialize_sinusoidal().expect("initialization");
        let row = rel.get_relative_encoding(0, 3);
        let omega = 1.0 / 10000.0_f64.powf(2.0 * 1.0 / 8.0);
        assert!((row[2] - (3.0 * omega).sin()).abs() < 1e-12);
        assert!((row[3] - (3.0 * omega).cos()).abs() < 1e-12);
        assert_eq!(rel.parameter_count(), 17 * 8);
    }

    #[test]
    fn all_encoding_types_construct() {
        for encoding_type in [
            PositionalEncodingType::Sinusoidal,
            PositionalEncodingType::Learned,
            PositionalEncodingType::Rotary,
            PositionalEncodingType::None,
        ] {
            let pe = PositionalEncoding::<f64>::new(50, 32, encoding_type);
            assert!(pe.is_ok(), "{encoding_type:?} failed to construct");
        }
    }
}
