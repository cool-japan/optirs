use std::fmt::Debug;
// Transformer encoder layers and components
//
// This module implements the encoder components of the transformer optimizer,
// including the transformer layer, feed-forward network, and layer normalization.

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{CoreRandom as Random, Rng as SCRRng};

use super::super::TransformerOptimizerConfig;
use super::attention::MultiHeadAttention;
use crate::error::{OptimError, Result};

/// Activation functions for feed-forward networks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivationFunction {
    /// ReLU activation
    ReLU,
    /// GELU activation
    #[default]
    GELU,
    /// Swish/SiLU activation
    Swish,
    /// GLU (Gated Linear Unit): splits the input and gates one half by
    /// the sigmoid of the other, halving the width.
    GLU,
    /// GeGLU (GELU variant of GLU): splits the input and gates one half by
    /// the GELU of the other, halving the width.
    GeGLU,
}

impl ActivationFunction {
    /// Whether this activation halves the feature width (gated variants do).
    pub fn is_gated(self) -> bool {
        matches!(self, ActivationFunction::GLU | ActivationFunction::GeGLU)
    }

    /// Width produced by this activation given `input_width`.
    pub fn output_width(self, input_width: usize) -> usize {
        if self.is_gated() {
            input_width / 2
        } else {
            input_width
        }
    }
}

/// Single transformer encoder layer
#[derive(Debug, Clone)]
pub struct TransformerLayer<T: Float + Debug + Send + Sync + 'static> {
    /// Multi-head self-attention
    self_attention: MultiHeadAttention<T>,

    /// Cross-attention (for multi-task learning)
    cross_attention: Option<MultiHeadAttention<T>>,

    /// Feed-forward network
    feed_forward: FeedForwardNetwork<T>,

    /// Layer normalization layers
    ln1: LayerNorm<T>,
    ln2: LayerNorm<T>,
    ln3: Option<LayerNorm<T>>, // For cross-attention

    /// Dropout layers
    dropout1: DropoutLayer,
    dropout2: DropoutLayer,
    dropout3: Option<DropoutLayer>,

    /// Use pre-layer normalization
    pre_layer_norm: bool,
}

/// Feed-forward network
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork<T: Float + Debug + Send + Sync + 'static> {
    /// First linear layer weights
    linear1: Array2<T>,

    /// First linear layer bias
    bias1: Array1<T>,

    /// Second linear layer weights
    linear2: Array2<T>,

    /// Second linear layer bias
    bias2: Array1<T>,

    /// Activation function
    activation: ActivationFunction,

    /// Model dimension (input and output width)
    modeldim: usize,

    /// Hidden width produced by the first linear layer
    ff_dim: usize,

    /// Dropout layer
    dropout: DropoutLayer,
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm<T: Float + Debug + Send + Sync + 'static> {
    /// Scale parameters (gamma)
    gamma: Array1<T>,

    /// Shift parameters (beta)
    beta: Array1<T>,

    /// Epsilon for numerical stability
    eps: T,

    /// Dimension
    dim: usize,
}

/// Dropout layer with a deterministic, seeded generator.
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    /// Dropout probability
    prob: f64,

    /// Training mode. Defaults to `false` so that inference is the safe default;
    /// call [`DropoutLayer::set_training`] to enable dropout during training.
    training: bool,

    /// xorshift64* state (never zero)
    rng_state: u64,
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync,
    > TransformerLayer<T>
{
    /// Build a transformer encoder layer from the optimizer configuration.
    pub fn new(config: &TransformerOptimizerConfig, _rng: &mut Random) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(config)?;
        let cross_attention = if config.cross_attention {
            Some(MultiHeadAttention::new(config)?)
        } else {
            None
        };

        let feed_forward = FeedForwardNetwork::new(config)?;

        let ln1 = LayerNorm::new(config.modeldim, config.layer_norm_eps);
        let ln2 = LayerNorm::new(config.modeldim, config.layer_norm_eps);
        let ln3 = if config.cross_attention {
            Some(LayerNorm::new(config.modeldim, config.layer_norm_eps))
        } else {
            None
        };

        let dropout1 = DropoutLayer::with_seed(config.attention_dropout, config.dropout_seed);
        let dropout2 = DropoutLayer::with_seed(config.ff_dropout, config.dropout_seed ^ 0x9E37);
        let dropout3 = if config.cross_attention {
            Some(DropoutLayer::with_seed(
                config.attention_dropout,
                config.dropout_seed ^ 0x7F4A,
            ))
        } else {
            None
        };

        Ok(Self {
            self_attention,
            cross_attention,
            feed_forward,
            ln1,
            ln2,
            ln3,
            dropout1,
            dropout2,
            dropout3,
            pre_layer_norm: config.pre_layer_norm,
        })
    }

    /// Forward pass through the layer.
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        let mut x = input.clone();

        // Self-attention with residual connection
        let residual = x.clone();
        if self.pre_layer_norm {
            x = self.ln1.forward(&x)?;
        }

        x = self.self_attention.forward(&x, &x, &x)?;
        x = self.dropout1.forward(&x);
        x = x + &residual;

        if !self.pre_layer_norm {
            x = self.ln1.forward(&x)?;
        }

        // Cross-attention (if enabled)
        if let Some(ref mut cross_attn) = self.cross_attention {
            let residual = x.clone();
            if self.pre_layer_norm {
                if let Some(ref ln3) = self.ln3 {
                    x = ln3.forward(&x)?;
                }
            }

            x = cross_attn.forward(&x, &x, &x)?;
            if let Some(ref mut dropout3) = self.dropout3 {
                x = dropout3.forward(&x);
            }
            x = x + &residual;

            if !self.pre_layer_norm {
                if let Some(ref ln3) = self.ln3 {
                    x = ln3.forward(&x)?;
                }
            }
        }

        // Feed-forward with residual connection
        let residual = x.clone();
        if self.pre_layer_norm {
            x = self.ln2.forward(&x)?;
        }

        x = self.feed_forward.forward(&x)?;
        x = self.dropout2.forward(&x);
        x = x + &residual;

        if !self.pre_layer_norm {
            x = self.ln2.forward(&x)?;
        }

        Ok(x)
    }

    /// Get attention patterns for analysis
    pub fn get_attention_patterns(&self) -> Option<&scirs2_core::ndarray::Array3<T>> {
        self.self_attention.get_attention_patterns()
    }

    /// Switch every dropout in the layer between training and inference mode.
    pub fn set_training(&mut self, training: bool) {
        self.dropout1.set_training(training);
        self.dropout2.set_training(training);
        if let Some(ref mut dropout3) = self.dropout3 {
            dropout3.set_training(training);
        }
        self.feed_forward.set_training(training);
    }

    /// Mutable access to the feed-forward network
    pub fn feed_forward_mut(&mut self) -> &mut FeedForwardNetwork<T> {
        &mut self.feed_forward
    }

    /// Access the self-attention block
    pub fn self_attention(&self) -> &MultiHeadAttention<T> {
        &self.self_attention
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> FeedForwardNetwork<T> {
    /// Build a feed-forward network from the optimizer configuration.
    pub fn new(config: &TransformerOptimizerConfig) -> Result<Self> {
        Self::with_dimensions(
            config.modeldim,
            config.ff_dim,
            config.activation,
            config.ff_dropout,
            config.dropout_seed ^ 0x1357,
        )
    }

    /// Build a feed-forward network with explicit dimensions.
    ///
    /// The second linear layer is sized from the *activation output* width, so
    /// gated activations (which halve the width) are handled correctly.
    pub fn with_dimensions(
        modeldim: usize,
        ff_dim: usize,
        activation: ActivationFunction,
        dropout: f64,
        dropout_seed: u64,
    ) -> Result<Self> {
        if modeldim == 0 || ff_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "Feed-forward dimensions must be positive".to_string(),
            ));
        }
        if activation.is_gated() && !ff_dim.is_multiple_of(2) {
            return Err(OptimError::InvalidConfig(format!(
                "Gated activation {activation:?} requires an even feed-forward dimension, got {ff_dim}"
            )));
        }

        let hidden_out = activation.output_width(ff_dim);
        let linear1 = Self::xavier(modeldim, ff_dim);
        let linear2 = Self::xavier(hidden_out, modeldim);

        Ok(Self {
            linear1,
            bias1: Array1::zeros(ff_dim),
            linear2,
            bias2: Array1::zeros(modeldim),
            activation,
            modeldim,
            ff_dim,
            dropout: DropoutLayer::with_seed(dropout, dropout_seed),
        })
    }

    /// Xavier/Glorot uniform initialization with limit `sqrt(6 / (fan_in + fan_out))`.
    fn xavier(fan_in: usize, fan_out: usize) -> Array2<T> {
        let mut rng = scirs2_core::random::thread_rng();
        let bound = (6.0 / (fan_in + fan_out).max(1) as f64).sqrt();
        let mut weights = Array2::zeros((fan_in, fan_out));
        for elem in weights.iter_mut() {
            *elem = scirs2_core::numeric::NumCast::from((rng.random::<f64>() - 0.5) * 2.0 * bound)
                .unwrap_or_else(|| T::zero());
        }
        weights
    }

    /// Forward pass through the feed-forward network.
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        let x1 = Self::linear_transform(input, &self.linear1, &self.bias1)?;
        let x2 = self.apply_activation(&x1);
        let x3 = self.dropout.forward(&x2);
        Self::linear_transform(&x3, &self.linear2, &self.bias2)
    }

    fn linear_transform(
        input: &Array2<T>,
        weights: &Array2<T>,
        bias: &Array1<T>,
    ) -> Result<Array2<T>> {
        let (_seq_len, input_dim) = input.dim();
        let (weight_in, weight_out) = weights.dim();

        if input_dim != weight_in {
            return Err(OptimError::InvalidConfig(format!(
                "Input dimension {input_dim} doesn't match weight matrix input {weight_in}"
            )));
        }
        if bias.len() != weight_out {
            return Err(OptimError::InvalidConfig(format!(
                "Bias dimension {} doesn't match output dimension {weight_out}",
                bias.len()
            )));
        }

        Ok(input.dot(weights) + &*bias)
    }

    /// Numerically stable GELU (tanh approximation).
    fn gelu(x: T) -> T {
        let sqrt_2_pi: T =
            scirs2_core::numeric::NumCast::from(0.797_884_560_802_865).unwrap_or_else(|| T::one());
        let coeff: T = scirs2_core::numeric::NumCast::from(0.044715).unwrap_or_else(|| T::zero());
        let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::one());
        let inner = sqrt_2_pi * (x + coeff * x * x * x);
        half * x * (T::one() + inner.tanh())
    }

    /// Numerically stable logistic sigmoid.
    fn sigmoid(x: T) -> T {
        if x >= T::zero() {
            T::one() / (T::one() + (-x).exp())
        } else {
            let e = x.exp();
            e / (T::one() + e)
        }
    }

    fn apply_activation(&self, input: &Array2<T>) -> Array2<T> {
        match self.activation {
            ActivationFunction::ReLU => input.map(|&x| if x > T::zero() { x } else { T::zero() }),
            ActivationFunction::GELU => input.map(|&x| Self::gelu(x)),
            // Swish / SiLU: x * sigmoid(x). The previous `x * exp(x) / (1 +
            // exp(x))` form overflowed to NaN for x greater than about 710.
            ActivationFunction::Swish => input.map(|&x| x * Self::sigmoid(x)),
            ActivationFunction::GLU => self.gated(input, Self::sigmoid),
            ActivationFunction::GeGLU => self.gated(input, Self::gelu),
        }
    }

    /// Gated linear unit: split the last axis in half and gate `a` by `g(b)`.
    fn gated(&self, input: &Array2<T>, gate: fn(T) -> T) -> Array2<T> {
        let (rows, cols) = input.dim();
        let half = cols / 2;
        let mut output = Array2::zeros((rows, half));
        for i in 0..rows {
            for j in 0..half {
                output[[i, j]] = input[[i, j]] * gate(input[[i, half + j]]);
            }
        }
        output
    }

    /// Set the activation function, resizing the second linear layer when the
    /// activation changes the hidden width (gated vs. elementwise).
    pub fn set_activation(&mut self, activation: ActivationFunction) -> Result<()> {
        if activation == self.activation {
            return Ok(());
        }
        if activation.is_gated() && !self.ff_dim.is_multiple_of(2) {
            return Err(OptimError::InvalidConfig(format!(
                "Gated activation {activation:?} requires an even feed-forward dimension, got {}",
                self.ff_dim
            )));
        }

        let new_width = activation.output_width(self.ff_dim);
        if new_width != self.linear2.nrows() {
            self.linear2 = Self::xavier(new_width, self.modeldim);
        }
        self.activation = activation;
        Ok(())
    }

    /// Get the active activation function
    pub fn activation(&self) -> ActivationFunction {
        self.activation
    }

    /// Model (input/output) dimension
    pub fn model_dim(&self) -> usize {
        self.modeldim
    }

    /// Hidden dimension of the first linear layer
    pub fn hidden_dim(&self) -> usize {
        self.ff_dim
    }

    /// Enable or disable dropout
    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }
}

impl<T: Float + Debug + Default + Clone + std::iter::Sum + Send + Sync> LayerNorm<T> {
    /// Create a layer normalization block with the given epsilon.
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: scirs2_core::numeric::NumCast::from(eps).unwrap_or_else(|| T::zero()),
            dim,
        }
    }

    /// Create a layer normalization block with the default epsilon of 1e-6.
    pub fn with_default_eps(dim: usize) -> Self {
        Self::new(dim, 1e-6)
    }

    /// Normalize each row: `(x - mean) / sqrt(var + eps) * gamma + beta`.
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let (seq_len, input_dim) = input.dim();

        if input_dim != self.dim {
            return Err(OptimError::InvalidConfig(format!(
                "Input dimension {} doesn't match layer norm dimension {}",
                input_dim, self.dim
            )));
        }
        if input_dim == 0 {
            return Ok(Array2::zeros((seq_len, 0)));
        }

        let mut output = Array2::zeros((seq_len, input_dim));
        let width: T =
            scirs2_core::numeric::NumCast::from(input_dim as f64).unwrap_or_else(|| T::one());

        for i in 0..seq_len {
            let row = input.slice(s![i, ..]);

            let mean = row.iter().cloned().sum::<T>() / width;
            let variance = row
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<T>()
                / width;

            let std = (variance + self.eps).sqrt();

            for j in 0..input_dim {
                let normalized = (input[[i, j]] - mean) / std;
                output[[i, j]] = self.gamma[j] * normalized + self.beta[j];
            }
        }

        Ok(output)
    }

    /// Get layer normalization parameters
    pub fn parameters(&self) -> (&Array1<T>, &Array1<T>) {
        (&self.gamma, &self.beta)
    }

    /// Normalized dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Set layer normalization parameters
    pub fn set_parameters(&mut self, gamma: Array1<T>, beta: Array1<T>) -> Result<()> {
        if gamma.len() != self.dim || beta.len() != self.dim {
            return Err(OptimError::InvalidConfig(format!(
                "Parameter dimensions ({}, {}) don't match layer norm dimension {}",
                gamma.len(),
                beta.len(),
                self.dim
            )));
        }
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }
}

impl DropoutLayer {
    /// Create a dropout layer with a default seed. Starts in inference mode.
    pub fn new(prob: f64) -> Self {
        Self::with_seed(prob, 0x243F_6A88_85A3_08D3)
    }

    /// Create a dropout layer with an explicit seed. Starts in inference mode.
    pub fn with_seed(prob: f64, seed: u64) -> Self {
        Self {
            prob: prob.clamp(0.0, 1.0),
            training: false,
            // xorshift64* requires a non-zero state.
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

    /// Inverted dropout: zero each entry with probability `prob` and rescale the
    /// survivors by `1 / (1 - prob)` so the expected activation is unchanged.
    /// In inference mode the input is returned untouched.
    pub fn forward<T: Float + Clone>(&mut self, input: &Array2<T>) -> Array2<T> {
        if !self.training || self.prob <= 0.0 {
            return input.clone();
        }
        if self.prob >= 1.0 {
            return Array2::zeros(input.dim());
        }

        let keep_prob = 1.0 - self.prob;
        let scale: T = scirs2_core::numeric::NumCast::from(1.0 / keep_prob).unwrap_or_else(T::one);

        let mut output = input.clone();
        for elem in output.iter_mut() {
            if self.next_f64() < self.prob {
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

    /// Whether the layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get dropout probability
    pub fn prob(&self) -> f64 {
        self.prob
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swish_is_stable_for_large_inputs() {
        let input =
            Array2::<f64>::from_shape_vec((1, 3), vec![-1000.0, 0.0, 1000.0]).expect("valid shape");
        let mut ffn =
            FeedForwardNetwork::<f64>::with_dimensions(3, 4, ActivationFunction::Swish, 0.0, 1)
                .expect("ffn creation");
        let activated = ffn.apply_activation(&input);
        assert!(activated.iter().all(|v| v.is_finite()), "{activated:?}");
        assert!((activated[[0, 2]] - 1000.0).abs() < 1e-6);
        assert!(activated[[0, 0]].abs() < 1e-6);
        ffn.set_training(true);
    }

    #[test]
    fn gated_activations_halve_the_width() {
        let ffn = FeedForwardNetwork::<f64>::with_dimensions(4, 8, ActivationFunction::GLU, 0.0, 1)
            .expect("ffn creation");
        let hidden = Array2::<f64>::from_shape_fn((2, 8), |(_, j)| j as f64 - 3.5);
        let activated = ffn.apply_activation(&hidden);
        assert_eq!(activated.shape(), &[2, 4]);
        // GLU: a * sigmoid(b)
        let expected = hidden[[0, 0]] * (1.0 / (1.0 + (-hidden[[0, 4]]).exp()));
        assert!((activated[[0, 0]] - expected).abs() < 1e-12);
    }

    #[test]
    fn gated_feed_forward_round_trips_to_model_dim() {
        let mut ffn =
            FeedForwardNetwork::<f64>::with_dimensions(4, 8, ActivationFunction::GeGLU, 0.0, 1)
                .expect("ffn creation");
        let input = Array2::<f64>::from_shape_fn((3, 4), |(i, j)| (i + j) as f64 * 0.1);
        let output = ffn.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn switching_to_a_gated_activation_resizes_the_output_layer() {
        let mut ffn =
            FeedForwardNetwork::<f64>::with_dimensions(4, 8, ActivationFunction::GELU, 0.0, 1)
                .expect("ffn creation");
        ffn.set_activation(ActivationFunction::GLU)
            .expect("activation switch");
        let input = Array2::<f64>::ones((2, 4));
        let output = ffn.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn odd_hidden_width_rejects_gated_activation() {
        let result =
            FeedForwardNetwork::<f64>::with_dimensions(4, 7, ActivationFunction::GLU, 0.0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn dropout_is_identity_at_inference_and_active_in_training() {
        let mut dropout = DropoutLayer::with_seed(0.5, 42);
        let input = Array2::<f64>::ones((8, 8));

        // Default is inference mode.
        assert!(!dropout.is_training());
        assert_eq!(dropout.forward(&input), input);

        dropout.set_training(true);
        let output = dropout.forward(&input);
        let zeros = output.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "training-mode dropout dropped nothing");
        assert!(
            zeros < input.len(),
            "training-mode dropout dropped everything"
        );
        // Inverted dropout rescales survivors by 1 / (1 - p) = 2.
        assert!(output.iter().all(|&v| v == 0.0 || (v - 2.0).abs() < 1e-12));
    }

    #[test]
    fn dropout_is_reproducible_for_a_given_seed() {
        let input = Array2::<f64>::ones((4, 4));
        let run = || {
            let mut dropout = DropoutLayer::with_seed(0.3, 7);
            dropout.set_training(true);
            dropout.forward(&input)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn layer_norm_rejects_mismatched_width() {
        let norm = LayerNorm::<f64>::with_default_eps(8);
        let input = Array2::<f64>::zeros((2, 4));
        assert!(norm.forward(&input).is_err());
    }

    #[test]
    fn layer_norm_standardizes_rows() {
        let norm = LayerNorm::<f64>::with_default_eps(4);
        let input =
            Array2::<f64>::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");
        let output = norm.forward(&input).expect("forward");
        let mean: f64 = output.row(0).iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-9, "mean {mean}");
    }
}
