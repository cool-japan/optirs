// Core transformer layer implementations

use super::config::ActivationFunction;
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng as SCRRng};
use std::fmt::Debug;

/// Draw a Xavier/Glorot uniform matrix with limit `sqrt(6 / (fan_in + fan_out))`.
fn xavier_uniform<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
>(
    fan_in: usize,
    fan_out: usize,
) -> Array2<T> {
    let mut rng = scirs2_core::random::thread_rng();
    let bound = (6.0 / (fan_in + fan_out).max(1) as f64).sqrt();
    let mut weights = Array2::zeros((fan_in, fan_out));
    for elem in weights.iter_mut() {
        *elem = scirs2_core::numeric::NumCast::from((rng.random::<f64>() - 0.5) * 2.0 * bound)
            .unwrap_or_else(|| T::zero());
    }
    weights
}

/// Input projection layer.
///
/// Optimization traces are continuous vectors, not token ids, so this is a
/// dense linear projection `input @ W` rather than a vocabulary lookup. The
/// previous implementation cast each float to a `usize` index, which mapped
/// every value in `(-1, 1)` onto row 0 and discarded the input entirely.
pub struct EmbeddingLayer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Projection matrix, shape `(input_dimension, output_dimension)`
    embedding_matrix: Array2<T>,

    /// Input dimension
    input_dimension: usize,

    /// Output dimension (model dimension)
    output_dimension: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    EmbeddingLayer<T>
{
    /// Create a new input projection with Xavier-initialized weights.
    pub fn new(input_dimension: usize, output_dimension: usize) -> Result<Self> {
        if input_dimension == 0 || output_dimension == 0 {
            return Err(OptimError::InvalidConfig(
                "Embedding dimensions must be positive".to_string(),
            ));
        }

        Ok(Self {
            embedding_matrix: xavier_uniform(input_dimension, output_dimension),
            input_dimension,
            output_dimension,
        })
    }

    /// Project a `(sequence_length, input_dimension)` matrix to the model dimension.
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (_sequence_length, width) = input.dim();
        if width != self.input_dimension {
            return Err(OptimError::InvalidConfig(format!(
                "Embedding input width {width} does not match the expected {}",
                self.input_dimension
            )));
        }

        Ok(input.dot(&self.embedding_matrix))
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_dimension * self.output_dimension
    }

    /// Projection weights
    pub fn weights(&self) -> &Array2<T> {
        &self.embedding_matrix
    }

    /// Replace the projection weights
    pub fn set_weights(&mut self, weights: Array2<T>) -> Result<()> {
        if weights.dim() != (self.input_dimension, self.output_dimension) {
            return Err(OptimError::InvalidConfig(format!(
                "Weight shape {:?} does not match ({}, {})",
                weights.dim(),
                self.input_dimension,
                self.output_dimension
            )));
        }
        self.embedding_matrix = weights;
        Ok(())
    }

    /// Gradient of the projection with respect to its weights, and an SGD step.
    ///
    /// For `y = x W`, `dL/dW = x^T (dL/dy)`.
    pub fn backward(
        &mut self,
        input: &Array2<T>,
        grad_output: &Array2<T>,
        learning_rate: T,
    ) -> Result<Array2<T>> {
        if input.ncols() != self.input_dimension || grad_output.ncols() != self.output_dimension {
            return Err(OptimError::InvalidConfig(
                "Embedding backward received mismatched shapes".to_string(),
            ));
        }
        if input.nrows() != grad_output.nrows() {
            return Err(OptimError::InvalidConfig(
                "Embedding backward received mismatched row counts".to_string(),
            ));
        }

        let grad_weights = input.t().dot(grad_output);
        self.embedding_matrix = &self.embedding_matrix - &(grad_weights.clone() * learning_rate);
        Ok(grad_weights)
    }

    /// Re-randomize the projection weights.
    pub fn reset(&mut self) -> Result<()> {
        self.embedding_matrix = xavier_uniform(self.input_dimension, self.output_dimension);
        Ok(())
    }
}

/// Layer normalization implementation
pub struct LayerNormalization<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Layer dimension
    dimension: usize,

    /// Learnable scale parameters
    gamma: Array1<T>,

    /// Learnable shift parameters
    beta: Array1<T>,

    /// Small constant for numerical stability
    epsilon: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    LayerNormalization<T>
{
    /// Create new layer normalization
    pub fn new(dimension: usize) -> Result<Self> {
        let gamma = Array1::ones(dimension);
        let beta = Array1::zeros(dimension);
        let epsilon = scirs2_core::numeric::NumCast::from(1e-5).unwrap_or_else(|| T::zero());

        Ok(Self {
            dimension,
            gamma,
            beta,
            epsilon,
        })
    }

    /// Forward pass through layer normalization.
    ///
    /// Rejects inputs whose width differs from the configured dimension instead
    /// of indexing out of bounds.
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (rows, width) = input.dim();
        if width != self.dimension {
            return Err(OptimError::InvalidConfig(format!(
                "Layer norm input width {width} does not match its dimension {}",
                self.dimension
            )));
        }
        if width == 0 {
            return Ok(input.clone());
        }

        let mut output = input.clone();
        let width_t: T =
            scirs2_core::numeric::NumCast::from(width as f64).unwrap_or_else(|| T::one());

        for i in 0..rows {
            let row = input.row(i);

            let mean = row.sum() / width_t;
            let variance = row
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
                / width_t;

            let std_dev = (variance + self.epsilon).sqrt();

            for j in 0..self.dimension {
                let normalized = (input[[i, j]] - mean) / std_dev;
                output[[i, j]] = self.gamma[j] * normalized + self.beta[j];
            }
        }

        Ok(output)
    }

    /// Normalized dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Backward pass through layer normalization.
    ///
    /// With `xhat = (x - mu) / sqrt(var + eps)` and `y = gamma * xhat + beta`,
    /// the exact input gradient for one row is
    /// `dL/dx = (1/sigma) * (g' - mean(g') - xhat * mean(g' * xhat))`
    /// where `g' = dL/dy * gamma`. `gamma` and `beta` are updated in place with
    /// a plain SGD step and the input gradient is returned.
    pub fn backward(
        &mut self,
        input: &Array2<T>,
        grad_output: &Array2<T>,
        learning_rate: T,
    ) -> Result<Array2<T>> {
        if input.dim() != grad_output.dim() {
            return Err(OptimError::InvalidConfig(format!(
                "Layer norm backward shape mismatch: {:?} vs {:?}",
                input.dim(),
                grad_output.dim()
            )));
        }
        let (rows, width) = input.dim();
        if width != self.dimension {
            return Err(OptimError::InvalidConfig(format!(
                "Layer norm backward width {width} does not match dimension {}",
                self.dimension
            )));
        }
        if width == 0 {
            return Ok(Array2::zeros(input.dim()));
        }

        let width_t: T =
            scirs2_core::numeric::NumCast::from(width as f64).unwrap_or_else(|| T::one());
        let mut grad_input = Array2::zeros(input.dim());
        let mut grad_gamma = Array1::zeros(width);
        let mut grad_beta = Array1::zeros(width);

        for i in 0..rows {
            let row = input.row(i);
            let mean = row.sum() / width_t;
            let variance = row
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(T::zero(), |acc, x| acc + x)
                / width_t;
            let sigma = (variance + self.epsilon).sqrt();

            let mut xhat = Array1::zeros(width);
            for j in 0..width {
                xhat[j] = (input[[i, j]] - mean) / sigma;
            }

            let mut g_scaled = Array1::zeros(width);
            for j in 0..width {
                let g = grad_output[[i, j]];
                grad_gamma[j] = grad_gamma[j] + g * xhat[j];
                grad_beta[j] = grad_beta[j] + g;
                g_scaled[j] = g * self.gamma[j];
            }

            let mean_g = g_scaled.iter().fold(T::zero(), |a, &b| a + b) / width_t;
            let mean_gx = g_scaled
                .iter()
                .zip(xhat.iter())
                .map(|(&g, &x)| g * x)
                .fold(T::zero(), |a, b| a + b)
                / width_t;

            for j in 0..width {
                grad_input[[i, j]] = (g_scaled[j] - mean_g - xhat[j] * mean_gx) / sigma;
            }
        }

        self.gamma = &self.gamma - &(grad_gamma * learning_rate);
        self.beta = &self.beta - &(grad_beta * learning_rate);

        Ok(grad_input)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        2 * self.dimension // gamma + beta
    }

    /// Reset normalization parameters
    pub fn reset(&mut self) -> Result<()> {
        self.gamma.fill(T::one());
        self.beta.fill(T::zero());
        Ok(())
    }
}

/// Dropout layer for regularization, backed by a seeded generator so training
/// runs are reproducible.
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    /// Dropout probability
    dropout_rate: f64,

    /// Training mode flag. Defaults to `false`, so inference is the safe default.
    training: bool,

    /// xorshift64* state (never zero)
    rng_state: u64,
}

impl DropoutLayer {
    /// Create a new dropout layer with a default seed. Starts in inference mode.
    pub fn new(dropout_rate: f64) -> Self {
        Self::with_seed(dropout_rate, 0x9E37_79B9_7F4A_7C15)
    }

    /// Create a new dropout layer with an explicit seed. Starts in inference mode.
    pub fn with_seed(dropout_rate: f64, seed: u64) -> Self {
        Self {
            dropout_rate: dropout_rate.clamp(0.0, 1.0),
            training: false,
            rng_state: seed | 1,
        }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let value = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Inverted dropout: zero entries with probability `dropout_rate` and scale
    /// the survivors by `1 / (1 - rate)` so the expectation is preserved.
    pub fn forward<
        T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
    >(
        &mut self,
        input: &Array2<T>,
    ) -> Array2<T> {
        if !self.training || self.dropout_rate <= 0.0 {
            return input.clone();
        }
        if self.dropout_rate >= 1.0 {
            return Array2::zeros(input.dim());
        }

        let mut output = input.clone();
        let keep_prob = 1.0 - self.dropout_rate;
        let scale: T =
            scirs2_core::numeric::NumCast::from(1.0 / keep_prob).unwrap_or_else(|| T::one());

        for elem in output.iter_mut() {
            if self.next_f64() < self.dropout_rate {
                *elem = T::zero();
            } else {
                *elem = *elem * scale;
            }
        }

        output
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Dropout probability
    pub fn dropout_rate(&self) -> f64 {
        self.dropout_rate
    }
}

/// Output projection layer
pub struct OutputProjection<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Weight matrix
    weight: Array2<T>,

    /// Bias vector
    bias: Array1<T>,

    /// Input dimension
    input_dim: usize,

    /// Output dimension
    output_dim: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    OutputProjection<T>
{
    /// Create a new output projection with Xavier-initialized weights.
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "Output projection dimensions must be positive".to_string(),
            ));
        }

        Ok(Self {
            weight: xavier_uniform(input_dim, output_dim),
            bias: Array1::zeros(output_dim),
            input_dim,
            output_dim,
        })
    }

    /// Forward pass: `input @ weight + bias`.
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (_rows, width) = input.dim();
        if width != self.input_dim {
            return Err(OptimError::InvalidConfig(format!(
                "Projection input width {width} does not match the expected {}",
                self.input_dim
            )));
        }

        Ok(input.dot(&self.weight) + &self.bias)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }

    /// Projection weights
    pub fn weights(&self) -> &Array2<T> {
        &self.weight
    }

    /// Projection bias
    pub fn bias(&self) -> &Array1<T> {
        &self.bias
    }

    /// Replace the projection parameters
    pub fn set_parameters(&mut self, weight: Array2<T>, bias: Array1<T>) -> Result<()> {
        if weight.dim() != (self.input_dim, self.output_dim) || bias.len() != self.output_dim {
            return Err(OptimError::InvalidConfig(format!(
                "Parameter shapes {:?}/{} do not match ({}, {})",
                weight.dim(),
                bias.len(),
                self.input_dim,
                self.output_dim
            )));
        }
        self.weight = weight;
        self.bias = bias;
        Ok(())
    }

    /// Backward pass for `y = x W + b`.
    ///
    /// Returns `dL/dx` and applies an SGD step to `W` and `b`.
    pub fn backward(
        &mut self,
        input: &Array2<T>,
        grad_output: &Array2<T>,
        learning_rate: T,
    ) -> Result<Array2<T>> {
        if input.ncols() != self.input_dim || grad_output.ncols() != self.output_dim {
            return Err(OptimError::InvalidConfig(
                "Output projection backward received mismatched shapes".to_string(),
            ));
        }
        if input.nrows() != grad_output.nrows() {
            return Err(OptimError::InvalidConfig(
                "Output projection backward received mismatched row counts".to_string(),
            ));
        }

        let grad_weight = input.t().dot(grad_output);
        let mut grad_bias = Array1::zeros(self.output_dim);
        for j in 0..self.output_dim {
            grad_bias[j] = grad_output.column(j).iter().fold(T::zero(), |a, &b| a + b);
        }

        let grad_input = grad_output.dot(&self.weight.t());

        self.weight = &self.weight - &(grad_weight * learning_rate);
        self.bias = &self.bias - &(grad_bias * learning_rate);

        Ok(grad_input)
    }

    /// Re-randomize the projection weights and zero the bias.
    pub fn reset(&mut self) -> Result<()> {
        self.weight = xavier_uniform(self.input_dim, self.output_dim);
        self.bias.fill(T::zero());
        Ok(())
    }
}

/// Residual connections manager
pub struct ResidualConnections<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Model dimension
    dimension: usize,

    /// Optional learnable scaling factor
    scale_factor: Option<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    ResidualConnections<T>
{
    /// Create new residual connections
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            scale_factor: None,
        }
    }

    /// Create with learnable scaling
    pub fn new_with_scaling(dimension: usize, initial_scale: T) -> Self {
        Self {
            dimension,
            scale_factor: Some(initial_scale),
        }
    }

    /// Add residual connection
    pub fn add(&self, input: &Array2<T>, residual: &Array2<T>) -> Result<Array2<T>> {
        if input.shape() != residual.shape() {
            return Err(crate::error::OptimError::Other(
                "Shape mismatch in residual connection".to_string(),
            ));
        }

        let mut output = input + residual;

        if let Some(scale) = self.scale_factor {
            output.mapv_inplace(|x| x * scale);
        }

        Ok(output)
    }

    /// Set scaling factor
    pub fn set_scale_factor(&mut self, scale: T) {
        self.scale_factor = Some(scale);
    }

    /// Get scaling factor
    pub fn get_scale_factor(&self) -> Option<T> {
        self.scale_factor
    }
}

/// Activation layer with various activation functions
pub struct ActivationLayer;

impl ActivationLayer {
    /// Apply activation function
    pub fn apply<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>(
        input: &Array2<T>,
        activation: ActivationFunction,
    ) -> Array2<T> {
        match activation {
            ActivationFunction::ReLU => Self::relu(input),
            ActivationFunction::GELU => Self::gelu(input),
            ActivationFunction::Swish => Self::swish(input),
            ActivationFunction::Tanh => Self::tanh(input),
            ActivationFunction::Sigmoid => Self::sigmoid(input),
            ActivationFunction::LeakyReLU => Self::leaky_relu(
                input,
                scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero()),
            ),
        }
    }

    /// ReLU activation
    fn relu<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>(
        input: &Array2<T>,
    ) -> Array2<T> {
        input.map(|&x| if x > T::zero() { x } else { T::zero() })
    }

    /// GELU activation (approximation)
    fn gelu<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>(
        input: &Array2<T>,
    ) -> Array2<T> {
        input.map(|&x| {
            let half = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
            let one = T::one();
            let sqrt_2_pi =
                scirs2_core::numeric::NumCast::from(0.797884560802865).unwrap_or_else(|| T::zero()); // sqrt(2/π)
            let coeff = scirs2_core::numeric::NumCast::from(0.044715).unwrap_or_else(|| T::zero());

            let tanh_arg = sqrt_2_pi * (x + coeff * x * x * x);
            let tanh_val = tanh_arg.tanh();

            half * x * (one + tanh_val)
        })
    }

    /// Swish activation
    fn swish<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>(
        input: &Array2<T>,
    ) -> Array2<T> {
        input.map(|&x| x * Self::sigmoid_scalar(x))
    }

    /// Tanh activation
    fn tanh<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>(
        input: &Array2<T>,
    ) -> Array2<T> {
        input.map(|&x| x.tanh())
    }

    /// Sigmoid activation
    fn sigmoid<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>(
        input: &Array2<T>,
    ) -> Array2<T> {
        input.map(|&x| Self::sigmoid_scalar(x))
    }

    /// Leaky ReLU activation
    fn leaky_relu<
        T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
    >(
        input: &Array2<T>,
        alpha: T,
    ) -> Array2<T> {
        input.map(|&x| if x > T::zero() { x } else { alpha * x })
    }

    /// Numerically stable sigmoid.
    fn sigmoid_scalar<
        T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
    >(
        x: T,
    ) -> T {
        let one = T::one();
        if x >= T::zero() {
            one / (one + (-x).exp())
        } else {
            let e = x.exp();
            e / (one + e)
        }
    }

    /// Elementwise derivative of the activation, evaluated at its *input*.
    pub fn derivative<
        T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
    >(
        pre_activation: &Array2<T>,
        activation: ActivationFunction,
    ) -> Array2<T> {
        match activation {
            ActivationFunction::ReLU => {
                pre_activation.map(|&x| if x > T::zero() { T::one() } else { T::zero() })
            }
            ActivationFunction::LeakyReLU => {
                let alpha: T =
                    scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero());
                pre_activation.map(|&x| if x > T::zero() { T::one() } else { alpha })
            }
            ActivationFunction::Tanh => pre_activation.map(|&x| {
                let t = x.tanh();
                T::one() - t * t
            }),
            ActivationFunction::Sigmoid => pre_activation.map(|&x| {
                let s = Self::sigmoid_scalar(x);
                s * (T::one() - s)
            }),
            ActivationFunction::Swish => pre_activation.map(|&x| {
                let s = Self::sigmoid_scalar(x);
                s + x * s * (T::one() - s)
            }),
            ActivationFunction::GELU => pre_activation.map(|&x| {
                // Derivative of 0.5 x (1 + tanh(a (x + c x^3)))
                let a: T = scirs2_core::numeric::NumCast::from(0.797_884_560_802_865)
                    .unwrap_or_else(|| T::one());
                let c: T =
                    scirs2_core::numeric::NumCast::from(0.044715).unwrap_or_else(|| T::zero());
                let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::one());
                let three: T = scirs2_core::numeric::NumCast::from(3.0).unwrap_or_else(|| T::one());
                let inner = a * (x + c * x * x * x);
                let t = inner.tanh();
                let sech2 = T::one() - t * t;
                half * (T::one() + t) + half * x * sech2 * a * (T::one() + three * c * x * x)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_layer_projects_continuous_inputs() {
        let embedding = EmbeddingLayer::<f64>::new(6, 4).expect("embedding creation");
        assert_eq!(embedding.parameter_count(), 24);

        // Values inside (-1, 1) used to collapse onto embedding row 0.
        let input = Array2::<f64>::from_shape_fn((3, 6), |(i, j)| 0.1 * (i + j) as f64 - 0.3);
        let output = embedding.forward(&input).expect("forward");
        assert_eq!(output.dim(), (3, 4));
        assert!(
            output.iter().any(|&v| v != 0.0),
            "projection produced all zeros"
        );

        // Different rows must yield different projections.
        assert!(output.row(0) != output.row(2));
    }

    #[test]
    fn embedding_layer_matches_a_manual_matmul() {
        let mut embedding = EmbeddingLayer::<f64>::new(2, 2).expect("embedding creation");
        embedding
            .set_weights(
                Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape"),
            )
            .expect("weights");
        let input = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("valid shape");
        let output = embedding.forward(&input).expect("forward");
        assert!((output[[0, 0]] - 4.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn embedding_reset_re_randomizes() {
        let mut embedding = EmbeddingLayer::<f64>::new(8, 8).expect("embedding creation");
        let before = embedding.weights().clone();
        embedding.reset().expect("reset");
        assert!(embedding.weights().iter().any(|&v| v != 0.0));
        assert_ne!(*embedding.weights(), before);
    }

    #[test]
    fn embedding_rejects_wrong_width() {
        let embedding = EmbeddingLayer::<f64>::new(6, 4).expect("embedding creation");
        assert!(embedding.forward(&Array2::<f64>::zeros((2, 5))).is_err());
        assert!(EmbeddingLayer::<f64>::new(0, 4).is_err());
    }

    #[test]
    fn output_projection_produces_nonzero_output() {
        let projection = OutputProjection::<f64>::new(8, 4).expect("projection creation");
        let input = Array2::<f64>::ones((2, 8));
        let output = projection.forward(&input).expect("forward");
        assert_eq!(output.dim(), (2, 4));
        assert!(
            output.iter().any(|&v| v != 0.0),
            "zero-initialized projection produced all zeros"
        );
        assert_eq!(projection.parameter_count(), 8 * 4 + 4);
    }

    #[test]
    fn output_projection_reset_re_randomizes() {
        let mut projection = OutputProjection::<f64>::new(8, 4).expect("projection creation");
        let before = projection.weights().clone();
        projection.reset().expect("reset");
        assert!(projection.weights().iter().any(|&v| v != 0.0));
        assert_ne!(*projection.weights(), before);
    }

    #[test]
    fn layer_normalization_standardizes_and_validates() {
        let norm = LayerNormalization::<f64>::new(4).expect("layer norm creation");
        let input =
            Array2::<f64>::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");
        let output = norm.forward(&input).expect("forward");
        let mean: f64 = output.row(0).iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-9, "mean {mean}");
        assert_eq!(norm.parameter_count(), 8);

        // Mismatched widths are an error, not an out-of-bounds index.
        assert!(norm.forward(&Array2::<f64>::zeros((2, 8))).is_err());
    }

    #[test]
    fn dropout_defaults_to_inference_and_is_reproducible() {
        let mut dropout = DropoutLayer::with_seed(0.5, 11);
        let input = Array2::<f64>::ones((8, 8));

        assert!(!dropout.is_training());
        assert_eq!(dropout.forward(&input), input);

        dropout.set_training(true);
        let first = dropout.forward(&input);
        let zeros = first.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0 && zeros < input.len());
        assert!(first.iter().all(|&v| v == 0.0 || (v - 2.0).abs() < 1e-12));

        let mut replay = DropoutLayer::with_seed(0.5, 11);
        replay.set_training(true);
        assert_eq!(replay.forward(&input), first);
    }

    #[test]
    fn residual_connections_validate_shapes() {
        let residual = ResidualConnections::<f64>::new(4);
        let input = Array2::<f64>::ones((2, 4));
        let other = Array2::<f64>::from_elem((2, 4), 0.5);
        let output = residual.add(&input, &other).expect("residual add");
        assert!((output[[0, 0]] - 1.5).abs() < 1e-12);
        assert!(residual.add(&input, &Array2::<f64>::ones((3, 4))).is_err());
    }

    #[test]
    fn activation_functions_behave() {
        let input =
            Array2::<f64>::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.5, 1.0]).expect("valid shape");

        let relu = ActivationLayer::apply(&input, ActivationFunction::ReLU);
        assert_eq!(relu[[0, 0]], 0.0);
        assert_eq!(relu[[1, 1]], 1.0);

        let gelu = ActivationLayer::apply(&input, ActivationFunction::GELU);
        assert_eq!(gelu.shape(), input.shape());

        let sigmoid = ActivationLayer::apply(&input, ActivationFunction::Sigmoid);
        assert!(sigmoid.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn activations_stay_finite_for_extreme_inputs() {
        let input =
            Array2::<f64>::from_shape_vec((1, 3), vec![-1e6, 0.0, 1e6]).expect("valid shape");
        for activation in [
            ActivationFunction::ReLU,
            ActivationFunction::GELU,
            ActivationFunction::Swish,
            ActivationFunction::Tanh,
            ActivationFunction::Sigmoid,
            ActivationFunction::LeakyReLU,
        ] {
            let output = ActivationLayer::apply(&input, activation);
            assert!(
                output.iter().all(|v| v.is_finite()),
                "{activation:?} produced {output:?}"
            );
        }
    }
}
