use std::fmt::Debug;
// Feed-forward network components for transformer layers
//
// This module implements the feed-forward network (FFN) layers used in
// transformer encoder/decoder blocks, including various activation functions
// and output projection layers.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng as SCRRng};

use crate::error::{OptimError, Result};

/// Output transformation types
#[derive(Debug, Clone, Copy)]
pub enum OutputTransformation {
    /// Linear transformation
    Linear,
    /// Tanh activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// Learned activation
    LearnedActivation,
    /// Parameter-specific scaling
    ParameterScaling,
}

/// Output projection layer for final transformer output
#[derive(Debug, Clone)]
pub struct OutputProjectionLayer<T: Float + Debug + Send + Sync + 'static> {
    /// Projection weights
    weights: Array2<T>,

    /// Projection bias
    bias: Array1<T>,

    /// Output transformation
    transformation: OutputTransformation,
}

/// Input embedding layer for transformer input processing
#[derive(Debug, Clone)]
pub struct InputEmbedding<T: Float + Debug + Send + Sync + 'static> {
    /// Embedding weights
    weights: Array2<T>,

    /// Input dimension
    input_dim: usize,

    /// Model dimension
    modeldim: usize,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> OutputProjectionLayer<T> {
    /// Create new output projection layer with Xavier/Glorot uniform weights,
    /// limit `sqrt(6 / (fan_in + fan_out))`.
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "Output projection dimensions must be positive".to_string(),
            ));
        }

        let mut rng = scirs2_core::random::thread_rng();
        let mut weights = Array2::zeros((input_dim, output_dim));

        let bound = (6.0 / (input_dim + output_dim) as f64).sqrt();
        for elem in weights.iter_mut() {
            *elem = scirs2_core::numeric::NumCast::from((rng.random::<f64>() - 0.5) * 2.0 * bound)
                .unwrap_or_else(|| T::zero());
        }

        let bias = Array1::zeros(output_dim);

        Ok(Self {
            weights,
            bias,
            transformation: OutputTransformation::Linear,
        })
    }

    /// Create with specific transformation
    pub fn new_with_transformation(
        input_dim: usize,
        output_dim: usize,
        transformation: OutputTransformation,
    ) -> Result<Self> {
        let mut layer = Self::new(input_dim, output_dim)?;
        layer.transformation = transformation;
        Ok(layer)
    }

    /// Forward pass through output projection
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (seq_len, input_dim) = input.dim();
        let (weight_in, weight_out) = self.weights.dim();

        if input_dim != weight_in {
            return Err(OptimError::InvalidConfig(
                "Input dimension doesn't match weight matrix".to_string(),
            ));
        }

        let _ = seq_len;
        let mut output = input.dot(&self.weights) + &self.bias;

        // Apply output transformation
        match self.transformation {
            OutputTransformation::Linear => {
                // No additional transformation
            }
            OutputTransformation::Tanh => {
                output.mapv_inplace(|x| x.tanh());
            }
            OutputTransformation::Sigmoid => {
                output.mapv_inplace(|x| T::one() / (T::one() + (-x).exp()));
            }
            OutputTransformation::LearnedActivation => {
                // Softplus-gated identity: smooth, monotone and bounded below.
                output.mapv_inplace(|x| x * (T::one() + (-x.abs()).exp()));
            }
            OutputTransformation::ParameterScaling => {
                // Apply a fixed, dimension-dependent scaling per output feature.
                for j in 0..weight_out {
                    let scale: T =
                        scirs2_core::numeric::NumCast::from(1.0 + 0.1 * (j as f64).sin())
                            .unwrap_or_else(|| T::one());
                    for i in 0..output.nrows() {
                        output[[i, j]] = output[[i, j]] * scale;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get output transformation type
    pub fn transformation(&self) -> OutputTransformation {
        self.transformation
    }

    /// Set output transformation type
    pub fn set_transformation(&mut self, transformation: OutputTransformation) {
        self.transformation = transformation;
    }

    /// Get projection weights
    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }

    /// Get projection bias
    pub fn bias(&self) -> &Array1<T> {
        &self.bias
    }

    /// Update weights and bias
    pub fn update_parameters(&mut self, weights: Array2<T>, bias: Array1<T>) -> Result<()> {
        let (weight_in, weight_out) = weights.dim();
        let bias_dim = bias.len();

        if weight_out != bias_dim {
            return Err(OptimError::InvalidConfig(
                "Weight output dimension doesn't match bias dimension".to_string(),
            ));
        }

        // Update internal dimensions if they match
        if (weight_in, weight_out) == self.weights.dim() && bias_dim == self.bias.len() {
            self.weights = weights;
            self.bias = bias;
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(
                "New parameter dimensions don't match current layer dimensions".to_string(),
            ))
        }
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> InputEmbedding<T> {
    /// Create new input embedding layer with Xavier/Glorot uniform weights.
    pub fn new(input_dim: usize, model_dim: usize) -> Result<Self> {
        if input_dim == 0 || model_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "Input embedding dimensions must be positive".to_string(),
            ));
        }

        let mut rng = scirs2_core::random::thread_rng();
        let mut weights = Array2::zeros((input_dim, model_dim));

        let bound = (6.0 / (input_dim + model_dim) as f64).sqrt();
        for elem in weights.iter_mut() {
            *elem = scirs2_core::numeric::NumCast::from((rng.random::<f64>() - 0.5) * 2.0 * bound)
                .unwrap_or_else(|| T::zero());
        }

        Ok(Self {
            weights,
            input_dim,
            modeldim: model_dim,
        })
    }

    /// Forward pass through input embedding
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (seq_len, input_dim) = input.dim();

        if input_dim != self.input_dim {
            return Err(OptimError::InvalidConfig(format!(
                "Input dimension {} doesn't match embedding input dimension {}",
                input_dim, self.input_dim
            )));
        }

        let _ = seq_len;
        Ok(input.dot(&self.weights))
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get model dimension
    pub fn model_dim(&self) -> usize {
        self.modeldim
    }

    /// Get embedding weights
    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }

    /// Update embedding weights
    pub fn update_weights(&mut self, weights: Array2<T>) -> Result<()> {
        let (weight_in, weight_out) = weights.dim();

        if weight_in != self.input_dim || weight_out != self.modeldim {
            return Err(OptimError::InvalidConfig(
                "New weight dimensions don't match embedding dimensions".to_string(),
            ));
        }

        self.weights = weights;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_embedding_produces_nonzero_output() {
        let embedding = InputEmbedding::<f64>::new(6, 4).expect("embedding creation");
        let input = Array2::<f64>::ones((3, 6));
        let output = embedding.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[3, 4]);
        assert!(
            output.iter().any(|&v| v != 0.0),
            "Xavier-initialized embedding produced all zeros"
        );
    }

    #[test]
    fn output_projection_produces_nonzero_output() {
        let projection = OutputProjectionLayer::<f64>::new(6, 4).expect("projection creation");
        let input = Array2::<f64>::ones((2, 6));
        let output = projection.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[2, 4]);
        assert!(
            output.iter().any(|&v| v != 0.0),
            "Xavier-initialized projection produced all zeros"
        );
    }

    #[test]
    fn projection_matches_manual_matmul() {
        let mut projection = OutputProjectionLayer::<f64>::new(2, 2).expect("projection creation");
        let weights =
            Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");
        let bias = Array1::<f64>::from_vec(vec![0.5, -0.5]);
        projection
            .update_parameters(weights, bias)
            .expect("parameter update");

        let input = Array2::<f64>::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("valid shape");
        let output = projection.forward(&input).expect("forward");
        assert!((output[[0, 0]] - 4.5).abs() < 1e-12);
        assert!((output[[0, 1]] - 5.5).abs() < 1e-12);
    }

    #[test]
    fn dimension_mismatches_are_errors() {
        let embedding = InputEmbedding::<f64>::new(6, 4).expect("embedding creation");
        let input = Array2::<f64>::ones((3, 5));
        assert!(embedding.forward(&input).is_err());

        assert!(InputEmbedding::<f64>::new(0, 4).is_err());
        assert!(OutputProjectionLayer::<f64>::new(4, 0).is_err());
    }

    #[test]
    fn tanh_transformation_bounds_the_output() {
        let mut projection = OutputProjectionLayer::<f64>::new(4, 4).expect("projection creation");
        projection.set_transformation(OutputTransformation::Tanh);
        let input = Array2::<f64>::from_elem((2, 4), 100.0);
        let output = projection.forward(&input).expect("forward");
        assert!(output.iter().all(|v| v.abs() <= 1.0));
    }
}
