use std::fmt::Debug;
// Transformer-based learned optimizer
//
// This module provides a modular implementation of a transformer-based neural optimizer
// that uses self-attention mechanisms to adaptively update optimization parameters.
// The implementation is organized into architectural components, optimization strategies,
// and training infrastructure for improved maintainability and extensibility.

pub mod architecture;
pub mod strategies;
pub mod training;

// Re-export key types for convenience
pub use architecture::{
    ActivationFunction, AttentionOptimization, FeedForwardNetwork, InputEmbedding, LayerNorm,
    MultiHeadAttention, OutputProjectionLayer, PositionalEncoder, PositionalEncodingType,
    TransformerLayer,
};

pub use strategies::{
    GradientProcessingStrategy, GradientProcessor, LearningRateAdaptationStrategy,
    LearningRateAdapter, MomentumIntegrator, MomentumStrategy, RegularizationStrategy,
    TransformerRegularizer,
};

pub use training::{
    CurriculumLearner, CurriculumStrategy, EvaluationStrategy, MetaLearningStrategy,
    TransformerEvaluator, TransformerMetaLearner,
};

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};

use super::LearnedOptimizerConfig;
use crate::error::{OptimError, Result};

/// Configuration specific to Transformer optimizer
#[derive(Debug, Clone)]
pub struct TransformerOptimizerConfig {
    /// Base learned optimizer config
    pub base_config: LearnedOptimizerConfig,

    /// Model dimension (d_model)
    pub modeldim: usize,

    /// Number of attention heads
    pub numheads: usize,

    /// Feed-forward network dimension
    pub ff_dim: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Maximum sequence length, which is also the optimization-history window
    /// fed to the network on every step
    pub max_sequence_length: usize,

    /// Attention dropout rate
    pub attention_dropout: f64,

    /// Feed-forward dropout rate
    pub ff_dropout: f64,

    /// Layer normalization epsilon
    pub layer_norm_eps: f64,

    /// Use pre-layer normalization
    pub pre_layer_norm: bool,

    /// Positional encoding type
    pub pos_encoding_type: PositionalEncodingType,

    /// Enable relative position bias
    pub relative_position_bias: bool,

    /// Use rotary position embedding
    pub use_rope: bool,

    /// Mask out future positions in self-attention
    pub causal_attention: bool,

    /// Feed-forward activation function
    pub activation: ActivationFunction,

    /// Seed used by the dropout layers so training runs are reproducible
    pub dropout_seed: u64,

    /// Amplitude of the transformer's modulation of the gradient step.
    ///
    /// The applied per-coordinate scale is `1 + modulation * tanh(network
    /// output)`, so values in `(0, 1)` keep the scale strictly positive and the
    /// update in the descent half-space.
    pub update_modulation: f64,

    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,

    /// Attention pattern optimization
    pub attention_optimization: AttentionOptimization,

    /// Multi-scale attention
    pub multi_scale_attention: bool,

    /// Cross-attention for multi-task learning
    pub cross_attention: bool,

    /// Memory efficiency mode
    pub memory_efficient: bool,
}

impl Default for TransformerOptimizerConfig {
    fn default() -> Self {
        Self {
            base_config: LearnedOptimizerConfig::default(),
            modeldim: 32,
            numheads: 4,
            ff_dim: 64,
            num_layers: 2,
            max_sequence_length: 32,
            attention_dropout: 0.1,
            ff_dropout: 0.1,
            layer_norm_eps: 1e-6,
            pre_layer_norm: true,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            relative_position_bias: false,
            use_rope: false,
            causal_attention: true,
            activation: ActivationFunction::GELU,
            dropout_seed: 0x0D15_EA5E,
            update_modulation: 0.5,
            gradient_checkpointing: false,
            attention_optimization: AttentionOptimization::Full,
            multi_scale_attention: false,
            cross_attention: false,
            memory_efficient: false,
        }
    }
}

impl TransformerOptimizerConfig {
    /// Validate the configuration, returning a descriptive error for any
    /// combination that would otherwise fail (or silently misbehave) later.
    pub fn validate(&self) -> Result<()> {
        if self.modeldim == 0 {
            return Err(OptimError::InvalidConfig(
                "modeldim must be positive".to_string(),
            ));
        }
        if self.numheads == 0 {
            return Err(OptimError::InvalidConfig(
                "numheads must be positive".to_string(),
            ));
        }
        if !self.modeldim.is_multiple_of(self.numheads) {
            return Err(OptimError::InvalidConfig(format!(
                "modeldim ({}) must be divisible by numheads ({})",
                self.modeldim, self.numheads
            )));
        }
        if self.ff_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "ff_dim must be positive".to_string(),
            ));
        }
        if self.num_layers == 0 {
            return Err(OptimError::InvalidConfig(
                "num_layers must be positive".to_string(),
            ));
        }
        if self.max_sequence_length == 0 {
            return Err(OptimError::InvalidConfig(
                "max_sequence_length must be positive".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.attention_dropout) {
            return Err(OptimError::InvalidConfig(format!(
                "attention_dropout must lie in [0, 1), got {}",
                self.attention_dropout
            )));
        }
        if !(0.0..1.0).contains(&self.ff_dropout) {
            return Err(OptimError::InvalidConfig(format!(
                "ff_dropout must lie in [0, 1), got {}",
                self.ff_dropout
            )));
        }
        if self.layer_norm_eps <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "layer_norm_eps must be positive, got {}",
                self.layer_norm_eps
            )));
        }
        if !(0.0..1.0).contains(&self.update_modulation) {
            return Err(OptimError::InvalidConfig(format!(
                "update_modulation must lie in [0, 1), got {}",
                self.update_modulation
            )));
        }
        if self.activation.is_gated() && !self.ff_dim.is_multiple_of(2) {
            return Err(OptimError::InvalidConfig(format!(
                "gated activation {:?} requires an even ff_dim, got {}",
                self.activation, self.ff_dim
            )));
        }
        let head_dim = self.modeldim / self.numheads;
        if (self.use_rope || self.pos_encoding_type == PositionalEncodingType::Rotary)
            && !head_dim.is_multiple_of(2)
        {
            return Err(OptimError::InvalidConfig(format!(
                "rotary embeddings require an even head dimension, got {head_dim}"
            )));
        }

        Ok(())
    }
}

/// Transformer network architecture
#[derive(Debug, Clone)]
pub struct TransformerNetwork<
    T: Float
        + Debug
        + Default
        + Clone
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
> {
    /// Input embedding layer
    input_embedding: InputEmbedding<T>,

    /// Transformer layers
    layers: Vec<TransformerLayer<T>>,

    /// Output projection
    output_projection: OutputProjectionLayer<T>,

    /// Layer normalization for output
    output_layer_norm: LayerNorm<T>,

    /// Position encoder
    position_encoder: PositionalEncoder<T>,

    /// Configuration
    config: TransformerOptimizerConfig,
}

/// Transformer-based neural optimizer with self-attention mechanisms
#[derive(Debug)]
pub struct TransformerOptimizer<
    T: Float
        + Debug
        + Default
        + Clone
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
> {
    /// Configuration for the Transformer optimizer
    config: TransformerOptimizerConfig,

    /// Transformer network architecture
    transformer_network: TransformerNetwork<T>,

    /// Gradient processing strategies
    gradient_processor: GradientProcessor<T>,

    /// Learning rate adaptation
    lr_adapter: LearningRateAdapter<T>,

    /// Momentum integration
    momentum_integrator: MomentumIntegrator<T>,

    /// Regularization strategies
    regularizer: TransformerRegularizer<T>,

    /// Meta-learning components
    meta_learner: TransformerMetaLearner<T>,

    /// Curriculum learning
    curriculum_learner: CurriculumLearner<T>,

    /// Evaluation framework
    evaluator: TransformerEvaluator<T>,

    /// Sequence buffer for maintaining optimization history
    sequence_buffer: SequenceBuffer<T>,

    /// Performance metrics
    metrics: TransformerOptimizerMetrics,

    /// Current optimization step
    step_count: usize,
}

/// Per-parameter sequence buffer for optimization history
#[derive(Debug, Clone)]
pub struct SequenceBuffer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Gradient sequences, keyed by parameter name
    gradient_sequences: HashMap<String, VecDeque<Array1<T>>>,

    /// Parameter sequences, keyed by parameter name
    parameter_sequences: HashMap<String, VecDeque<Array1<T>>>,

    /// Loss sequences
    loss_sequences: VecDeque<T>,

    /// Learning rate sequences
    lr_sequences: VecDeque<T>,

    /// Buffer capacity per key
    capacity: usize,
}

/// Performance metrics for transformer optimizer
#[derive(Debug, Clone)]
pub struct TransformerOptimizerMetrics {
    /// Total optimization steps
    total_steps: usize,

    /// Convergence history
    convergence_history: Vec<f64>,

    /// Attention pattern statistics
    attention_stats: HashMap<String, f64>,

    /// Strategy usage statistics
    strategy_stats: HashMap<String, f64>,

    /// Performance comparisons
    performance_comparisons: HashMap<String, f64>,
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static,
    > TransformerNetwork<T>
{
    /// Create new transformer network
    pub fn new(config: &TransformerOptimizerConfig) -> Result<Self> {
        config.validate()?;

        let input_embedding = InputEmbedding::new(config.modeldim, config.modeldim)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        let mut rng = scirs2_core::random::thread_rng();
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(config, &mut rng)?);
        }

        let output_projection = OutputProjectionLayer::new(config.modeldim, config.modeldim)?;
        let output_layer_norm = LayerNorm::new(config.modeldim, config.layer_norm_eps);
        let position_encoder = PositionalEncoder::new(config)?;

        Ok(Self {
            input_embedding,
            layers,
            output_projection,
            output_layer_norm,
            position_encoder,
            config: config.clone(),
        })
    }

    /// Forward pass through transformer network
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        // Input embedding
        let mut x = self.input_embedding.forward(input)?;

        // Add positional encoding
        x = self.position_encoder.encode(&x)?;

        // Pass through transformer layers
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        // Output layer normalization
        x = self.output_layer_norm.forward(&x)?;

        // Output projection
        let output = self.output_projection.forward(&x)?;

        Ok(output)
    }

    /// Get attention patterns from all layers
    pub fn get_attention_patterns(&self) -> Vec<Option<&scirs2_core::ndarray::Array3<T>>> {
        self.layers
            .iter()
            .map(|layer| layer.get_attention_patterns())
            .collect()
    }

    /// Switch every dropout in the network between training and inference mode.
    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    /// Number of transformer layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Configuration this network was built from
    pub fn config(&self) -> &TransformerOptimizerConfig {
        &self.config
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static,
    > TransformerOptimizer<T>
{
    /// Create new transformer optimizer
    pub fn new(config: TransformerOptimizerConfig) -> Result<Self> {
        // Reject invalid configurations before any allocation happens.
        config.validate()?;

        let transformer_network = TransformerNetwork::new(&config)?;
        let gradient_processor = GradientProcessor::new(GradientProcessingStrategy::Adaptive);
        let base_lr: T = scirs2_core::numeric::NumCast::from(config.base_config.learning_rate)
            .unwrap_or_else(|| T::zero());
        let lr_adapter = LearningRateAdapter::new(
            LearningRateAdaptationStrategy::TransformerPredicted,
            base_lr,
        );
        let momentum_integrator = MomentumIntegrator::new(MomentumStrategy::Adam);
        let regularizer = TransformerRegularizer::new(RegularizationStrategy::Adaptive);
        let meta_learner = TransformerMetaLearner::new(MetaLearningStrategy::GradientBased)?;
        let curriculum_learner = CurriculumLearner::new(CurriculumStrategy::Adaptive)?;
        let evaluator = TransformerEvaluator::new(EvaluationStrategy::Comprehensive)?;
        let sequence_buffer = SequenceBuffer::new(config.max_sequence_length.max(1));
        let metrics = TransformerOptimizerMetrics::new();

        Ok(Self {
            config,
            transformer_network,
            gradient_processor,
            lr_adapter,
            momentum_integrator,
            regularizer,
            meta_learner,
            curriculum_learner,
            evaluator,
            sequence_buffer,
            metrics,
            step_count: 0,
        })
    }

    /// Perform one optimization step, writing the updated values back into
    /// `parameters` in place.
    ///
    /// The pipeline is: regularize -> process -> adapt learning rate ->
    /// integrate momentum -> run the recent optimization history through the
    /// transformer -> apply `param -= lr * modulation * momentum_gradient`.
    ///
    /// The transformer output is used as a bounded, strictly positive
    /// per-coordinate *scale* on the momentum-integrated gradient rather than
    /// as the update itself. This is the standard learned-preconditioner
    /// formulation: the network can speed up or slow down individual
    /// coordinates but cannot, by itself, reverse the descent direction, so an
    /// untrained network degrades gracefully to plain gradient descent instead
    /// of diverging.
    ///
    /// Returns the loss it was given, so callers can chain steps.
    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
        loss: T,
    ) -> Result<T> {
        self.step_count += 1;

        // Deterministic ordering so repeated runs are reproducible.
        let mut names: Vec<String> = gradients.keys().cloned().collect();
        names.sort();

        for name in &names {
            let Some(gradient) = gradients.get(name) else {
                continue;
            };
            let Some(param_matrix) = parameters.get(name) else {
                return Err(OptimError::InvalidConfig(format!(
                    "Gradient '{name}' has no matching parameter"
                )));
            };
            if param_matrix.dim() != gradient.dim() {
                return Err(OptimError::InvalidConfig(format!(
                    "Parameter '{}' has shape {:?} but its gradient has shape {:?}",
                    name,
                    param_matrix.dim(),
                    gradient.dim()
                )));
            }

            let shape = param_matrix.dim();
            let param_matrix = param_matrix.clone();
            let gradient = gradient.clone();

            // 1. Regularization operates on the real parameter/gradient pair and
            //    its effect is kept.
            let mut param_map = HashMap::new();
            param_map.insert(name.clone(), param_matrix.clone());
            let mut grad_map = HashMap::new();
            grad_map.insert(name.clone(), gradient);

            let _reg_loss =
                self.regularizer
                    .apply_regularization(&mut param_map, &mut grad_map, None)?;

            let regularized = grad_map.remove(name).ok_or_else(|| {
                OptimError::ComputationError(format!(
                    "Regularizer dropped the gradient for '{name}'"
                ))
            })?;
            // Spectral normalization may have rescaled the weights.
            let param_matrix = param_map.remove(name).unwrap_or(param_matrix);

            let flat_gradient = Array1::from_iter(regularized.iter().cloned());
            let flat_param = Array1::from_iter(param_matrix.iter().cloned());

            // 2. Gradient processing.
            let processed_gradient = self
                .gradient_processor
                .process_gradients(name, &flat_gradient)?;

            // 3. Learning rate adaptation.
            let current_lr = self
                .lr_adapter
                .update_learning_rate(Some(loss), Some(&processed_gradient))?;

            // 4. Momentum integration.
            let momentum_gradient =
                self.momentum_integrator
                    .integrate_momentum(name, &processed_gradient, None)?;

            // 5. Record the history the transformer consumes.
            self.sequence_buffer
                .add_gradient(name, momentum_gradient.clone());
            self.sequence_buffer
                .add_parameters(name, flat_param.clone());

            // 6. Transformer forward pass over the recent history.
            let modulation = self.transformer_modulation(name, flat_param.len())?;

            // 7. Apply the update in place.
            let mut updated = Vec::with_capacity(flat_param.len());
            for i in 0..flat_param.len() {
                updated.push(flat_param[i] - current_lr * momentum_gradient[i] * modulation[i]);
            }
            let updated = Array2::from_shape_vec(shape, updated).map_err(|e| {
                OptimError::ComputationError(format!(
                    "Failed to reshape the update for '{name}': {e}"
                ))
            })?;

            parameters.insert(name.clone(), updated);
            gradients.insert(name.clone(), regularized);
        }

        // Update sequence buffer with loss and learning rate
        self.sequence_buffer.add_loss(loss);
        self.sequence_buffer
            .add_learning_rate(self.lr_adapter.current_learning_rate());

        // Update curriculum learning
        self.curriculum_learner
            .update_curriculum("current_task", loss, self.step_count)?;

        // Update metrics
        self.metrics
            .update_step(loss.to_f64().unwrap_or(0.0), self.step_count);

        Ok(loss)
    }

    /// Build the `(seq_len, modeldim)` feature matrix for `param_name` from the
    /// recorded optimization history.
    ///
    /// Each row summarises one historical step: the flattened gradient is
    /// average-pooled into `modeldim` buckets by index modulo `modeldim`, which
    /// is well defined for parameter tensors both smaller and larger than the
    /// model dimension.
    fn build_feature_sequence(&self, param_name: &str) -> Array2<T> {
        let modeldim = self.config.modeldim;
        let history = self.sequence_buffer.recent_gradients(param_name);

        if history.is_empty() {
            return Array2::zeros((1, modeldim));
        }

        let seq_len = history.len().min(self.config.max_sequence_length).max(1);
        let start = history.len() - seq_len;
        let mut features = Array2::zeros((seq_len, modeldim));

        for (row, gradient) in history[start..].iter().enumerate() {
            let pooled = Self::pool_to_width(gradient, modeldim);
            for j in 0..modeldim {
                features[[row, j]] = pooled[j];
            }
        }

        features
    }

    /// Average-pool an arbitrary-length vector into exactly `width` buckets.
    fn pool_to_width(values: &Array1<T>, width: usize) -> Array1<T> {
        let mut sums = Array1::zeros(width);
        if width == 0 {
            return sums;
        }
        let mut counts = vec![0usize; width];

        for (i, &value) in values.iter().enumerate() {
            let bucket = i % width;
            sums[bucket] = sums[bucket] + value;
            counts[bucket] += 1;
        }

        for (j, &count) in counts.iter().enumerate() {
            if count > 0 {
                let denominator: T =
                    scirs2_core::numeric::NumCast::from(count as f64).unwrap_or_else(|| T::one());
                sums[j] = sums[j] / denominator;
            }
        }

        sums
    }

    /// Run the history through the transformer and expand its output into a
    /// per-coordinate, strictly positive modulation factor.
    fn transformer_modulation(&mut self, param_name: &str, target_len: usize) -> Result<Array1<T>> {
        let features = self.build_feature_sequence(param_name);
        let output = self.transformer_network.forward(&features)?;

        let rows = output.nrows();
        if rows == 0 || output.ncols() == 0 {
            return Ok(Array1::from_elem(target_len, T::one()));
        }

        // The last row summarises the most recent optimization state.
        let code = output.row(rows - 1);
        let width = code.len();
        let amplitude: T = scirs2_core::numeric::NumCast::from(self.config.update_modulation)
            .unwrap_or_else(|| T::zero());

        let mut modulation = Array1::from_elem(target_len, T::one());
        for i in 0..target_len {
            modulation[i] = T::one() + amplitude * code[i % width].tanh();
        }

        Ok(modulation)
    }

    /// Get current optimization statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        stats.insert("step_count".to_string(), self.step_count as f64);
        stats.insert(
            "current_lr".to_string(),
            self.lr_adapter
                .current_learning_rate()
                .to_f64()
                .unwrap_or(0.0),
        );

        // Add gradient processor statistics
        let grad_stats = self.gradient_processor.statistics();
        stats.insert(
            "mean_gradient_magnitude".to_string(),
            grad_stats.mean_magnitude().to_f64().unwrap_or(0.0),
        );
        stats.insert(
            "gradient_sparsity".to_string(),
            grad_stats.sparsity().to_f64().unwrap_or(0.0),
        );

        // Add momentum statistics
        let momentum_stats = self.momentum_integrator.statistics();
        stats.insert(
            "momentum_magnitude".to_string(),
            momentum_stats
                .avg_momentum_magnitude
                .to_f64()
                .unwrap_or(0.0),
        );
        stats.insert(
            "momentum_direction_consistency".to_string(),
            momentum_stats.direction_consistency.to_f64().unwrap_or(0.0),
        );

        // Add curriculum statistics
        let curriculum_stats = self.curriculum_learner.get_curriculum_statistics();
        for (key, value) in curriculum_stats {
            stats.insert(format!("curriculum_{}", key), value.to_f64().unwrap_or(0.0));
        }

        stats
    }

    /// Switch the network between training mode (dropout active) and inference
    /// mode. Optimizers are created in inference mode.
    pub fn set_training(&mut self, training: bool) {
        self.transformer_network.set_training(training);
    }

    /// Override the base learning rate used by the adapter.
    pub fn set_learning_rate(&mut self, lr: T) {
        self.lr_adapter.set_base_learning_rate(lr);
    }

    /// Current (post-adaptation) learning rate.
    pub fn current_learning_rate(&self) -> T {
        self.lr_adapter.current_learning_rate()
    }

    /// Number of steps taken so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Configuration in use.
    pub fn config(&self) -> &TransformerOptimizerConfig {
        &self.config
    }

    /// Mutable access to the learning rate adapter.
    pub fn lr_adapter_mut(&mut self) -> &mut LearningRateAdapter<T> {
        &mut self.lr_adapter
    }

    /// Mutable access to the gradient processor.
    pub fn gradient_processor_mut(&mut self) -> &mut GradientProcessor<T> {
        &mut self.gradient_processor
    }

    /// Mutable access to the momentum integrator.
    pub fn momentum_integrator_mut(&mut self) -> &mut MomentumIntegrator<T> {
        &mut self.momentum_integrator
    }

    /// Mutable access to the regularizer.
    pub fn regularizer_mut(&mut self) -> &mut TransformerRegularizer<T> {
        &mut self.regularizer
    }

    /// Mutable access to the meta-learner.
    pub fn meta_learner_mut(&mut self) -> &mut TransformerMetaLearner<T> {
        &mut self.meta_learner
    }

    /// Mutable access to the evaluator.
    pub fn evaluator_mut(&mut self) -> &mut TransformerEvaluator<T> {
        &mut self.evaluator
    }

    /// Reset optimizer state
    pub fn reset(&mut self) -> Result<()> {
        self.step_count = 0;
        self.gradient_processor.reset();
        self.lr_adapter.reset();
        self.momentum_integrator.reset();
        self.regularizer.reset();
        self.meta_learner.reset();
        self.curriculum_learner.reset();
        self.evaluator.reset();
        self.sequence_buffer.clear();
        self.metrics = TransformerOptimizerMetrics::new();

        Ok(())
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static,
    > SequenceBuffer<T>
{
    /// Create new sequence buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            gradient_sequences: HashMap::new(),
            parameter_sequences: HashMap::new(),
            loss_sequences: VecDeque::new(),
            lr_sequences: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn push_bounded(queue: &mut VecDeque<Array1<T>>, value: Array1<T>, capacity: usize) {
        queue.push_back(value);
        while queue.len() > capacity {
            queue.pop_front();
        }
    }

    /// Record a gradient for a named parameter
    pub fn add_gradient(&mut self, param_name: &str, gradient: Array1<T>) {
        let capacity = self.capacity;
        let queue = self
            .gradient_sequences
            .entry(param_name.to_string())
            .or_default();
        Self::push_bounded(queue, gradient, capacity);
    }

    /// Record parameter values for a named parameter
    pub fn add_parameters(&mut self, param_name: &str, parameters: Array1<T>) {
        let capacity = self.capacity;
        let queue = self
            .parameter_sequences
            .entry(param_name.to_string())
            .or_default();
        Self::push_bounded(queue, parameters, capacity);
    }

    /// Add loss to buffer
    pub fn add_loss(&mut self, loss: T) {
        self.loss_sequences.push_back(loss);
        while self.loss_sequences.len() > self.capacity {
            self.loss_sequences.pop_front();
        }
    }

    /// Add learning rate to buffer
    pub fn add_learning_rate(&mut self, lr: T) {
        self.lr_sequences.push_back(lr);
        while self.lr_sequences.len() > self.capacity {
            self.lr_sequences.pop_front();
        }
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.gradient_sequences.clear();
        self.parameter_sequences.clear();
        self.loss_sequences.clear();
        self.lr_sequences.clear();
    }

    /// Recorded gradients for a parameter, oldest first
    pub fn recent_gradients(&self, param_name: &str) -> Vec<&Array1<T>> {
        match self.gradient_sequences.get(param_name) {
            Some(queue) => queue.iter().collect(),
            None => Vec::new(),
        }
    }

    /// Recorded parameter values for a parameter, oldest first
    pub fn recent_parameters(&self, param_name: &str) -> Vec<&Array1<T>> {
        match self.parameter_sequences.get(param_name) {
            Some(queue) => queue.iter().collect(),
            None => Vec::new(),
        }
    }

    /// Recorded losses, oldest first
    pub fn losses(&self) -> &VecDeque<T> {
        &self.loss_sequences
    }

    /// Recorded learning rates, oldest first
    pub fn learning_rates(&self) -> &VecDeque<T> {
        &self.lr_sequences
    }

    /// Per-key capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for TransformerOptimizerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformerOptimizerMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            convergence_history: Vec::new(),
            attention_stats: HashMap::new(),
            strategy_stats: HashMap::new(),
            performance_comparisons: HashMap::new(),
        }
    }

    /// Update metrics after optimization step
    pub fn update_step(&mut self, loss: f64, step: usize) {
        self.total_steps = step;
        self.convergence_history.push(loss);

        // Keep only recent history
        if self.convergence_history.len() > 10000 {
            self.convergence_history.remove(0);
        }
    }

    /// Total steps recorded
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Recorded loss history
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Attention statistics
    pub fn attention_stats(&self) -> &HashMap<String, f64> {
        &self.attention_stats
    }

    /// Strategy usage statistics
    pub fn strategy_stats(&self) -> &HashMap<String, f64> {
        &self.strategy_stats
    }

    /// Baseline comparison statistics
    pub fn performance_comparisons(&self) -> &HashMap<String, f64> {
        &self.performance_comparisons
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_loss(params: &Array2<f64>) -> f64 {
        params.iter().map(|&x| x * x).sum()
    }

    fn quadratic_gradient(params: &Array2<f64>) -> Array2<f64> {
        params.mapv(|x| 2.0 * x)
    }

    #[test]
    fn default_config_is_valid() {
        TransformerOptimizerConfig::default()
            .validate()
            .expect("the default configuration must be valid");
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        let mut config = TransformerOptimizerConfig::default();
        config.numheads = 5; // 32 is not divisible by 5
        assert!(config.validate().is_err());
        assert!(TransformerOptimizer::<f64>::new(config).is_err());

        let mut config = TransformerOptimizerConfig::default();
        config.modeldim = 0;
        assert!(TransformerOptimizer::<f64>::new(config).is_err());

        let mut config = TransformerOptimizerConfig::default();
        config.ff_dropout = 1.5;
        assert!(TransformerOptimizer::<f64>::new(config).is_err());

        let mut config = TransformerOptimizerConfig::default();
        config.num_layers = 0;
        assert!(TransformerOptimizer::<f64>::new(config).is_err());
    }

    #[test]
    fn every_layer_count_is_supported() {
        for num_layers in 1..=4 {
            let config = TransformerOptimizerConfig {
                num_layers,
                ..Default::default()
            };
            let mut optimizer =
                TransformerOptimizer::<f64>::new(config).expect("optimizer creation");

            let mut parameters = HashMap::new();
            parameters.insert("w".to_string(), Array2::from_elem((2, 5), 0.5));
            let mut gradients = HashMap::new();
            gradients.insert("w".to_string(), Array2::from_elem((2, 5), 0.1));

            optimizer
                .step(&mut parameters, &mut gradients, 1.0)
                .unwrap_or_else(|e| panic!("{num_layers} layers failed: {e}"));
        }
    }

    #[test]
    fn step_writes_parameters_in_place() {
        let mut optimizer = TransformerOptimizer::<f64>::new(TransformerOptimizerConfig::default())
            .expect("optimizer creation");
        optimizer.set_learning_rate(1e-2);

        let initial = Array2::from_elem((3, 4), 1.0);
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), initial.clone());
        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), quadratic_gradient(&initial));

        optimizer
            .step(&mut parameters, &mut gradients, quadratic_loss(&initial))
            .expect("step must succeed");

        let updated = parameters.get("w").expect("parameter present");
        assert_ne!(*updated, initial, "step left the parameters untouched");
        assert!(updated.iter().all(|v| v.is_finite()));
        // Descent on a convex quadratic moves positive weights downward.
        assert!(updated[[0, 0]] < initial[[0, 0]]);
    }

    #[test]
    fn optimizing_a_quadratic_reduces_the_loss() {
        // Several seeds' worth of starting points: the update is gradient
        // anchored, so convergence must not depend on a lucky initialization.
        for scale in [0.25_f64, 0.5, 1.0, 2.0] {
            let mut optimizer =
                TransformerOptimizer::<f64>::new(TransformerOptimizerConfig::default())
                    .expect("optimizer creation");
            optimizer.set_learning_rate(1e-2);

            let initial =
                Array2::from_shape_fn((2, 4), |(i, j)| scale * (1.0 + (i + j) as f64 * 0.25));
            let initial_loss = quadratic_loss(&initial);

            let mut parameters = HashMap::new();
            parameters.insert("w".to_string(), initial.clone());

            for _ in 0..100 {
                let current = parameters.get("w").expect("parameter present").clone();
                let loss = quadratic_loss(&current);
                let mut gradients = HashMap::new();
                gradients.insert("w".to_string(), quadratic_gradient(&current));
                optimizer
                    .step(&mut parameters, &mut gradients, loss)
                    .expect("step must succeed");
            }

            let updated = parameters.get("w").expect("parameter present");
            let end_loss = quadratic_loss(updated);

            assert!(
                end_loss < initial_loss,
                "scale {scale}: loss {end_loss} did not improve on {initial_loss}"
            );
            assert!(
                updated.iter().zip(initial.iter()).any(|(a, b)| a != b),
                "scale {scale}: parameters never changed"
            );
            assert!(end_loss.is_finite());
        }
    }

    #[test]
    fn parameters_of_different_shapes_are_handled() {
        let mut optimizer = TransformerOptimizer::<f64>::new(TransformerOptimizerConfig::default())
            .expect("optimizer creation");

        let mut parameters = HashMap::new();
        parameters.insert("small".to_string(), Array2::from_elem((2, 3), 0.5));
        parameters.insert("large".to_string(), Array2::from_elem((8, 16), 0.25));
        let mut gradients = HashMap::new();
        gradients.insert("small".to_string(), Array2::from_elem((2, 3), 0.1));
        gradients.insert("large".to_string(), Array2::from_elem((8, 16), 0.05));

        optimizer
            .step(&mut parameters, &mut gradients, 1.0)
            .expect("heterogeneous shapes must work");

        assert_eq!(parameters["small"].dim(), (2, 3));
        assert_eq!(parameters["large"].dim(), (8, 16));
    }

    #[test]
    fn shape_mismatch_between_parameter_and_gradient_is_an_error() {
        let mut optimizer = TransformerOptimizer::<f64>::new(TransformerOptimizerConfig::default())
            .expect("optimizer creation");

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), Array2::from_elem((2, 3), 0.5));
        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), Array2::from_elem((3, 2), 0.1));

        assert!(optimizer
            .step(&mut parameters, &mut gradients, 1.0)
            .is_err());
    }

    #[test]
    fn zero_gradients_leave_parameters_finite() {
        let mut optimizer = TransformerOptimizer::<f64>::new(TransformerOptimizerConfig::default())
            .expect("optimizer creation");

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), Array2::from_elem((2, 3), 0.5));
        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), Array2::zeros((2, 3)));

        optimizer
            .step(&mut parameters, &mut gradients, 0.0)
            .expect("step must succeed");
        assert!(parameters["w"].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn network_forward_is_reachable_and_nonzero() {
        let config = TransformerOptimizerConfig::default();
        let mut network = TransformerNetwork::<f64>::new(&config).expect("network creation");
        let input = Array2::from_shape_fn((4, config.modeldim), |(i, j)| {
            ((i * config.modeldim + j) as f64).sin()
        });
        let output = network.forward(&input).expect("forward");
        assert_eq!(output.dim(), (4, config.modeldim));
        assert!(output.iter().any(|&v| v != 0.0));
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn training_mode_toggles_dropout() {
        let config = TransformerOptimizerConfig::default();
        let mut network = TransformerNetwork::<f64>::new(&config).expect("network creation");
        let input = Array2::from_elem((4, config.modeldim), 0.1);

        let inference_a = network.forward(&input).expect("forward");
        let inference_b = network.forward(&input).expect("forward");
        assert_eq!(inference_a, inference_b, "inference must be deterministic");

        network.set_training(true);
        let training_out = network.forward(&input).expect("forward");
        assert_ne!(
            training_out, inference_a,
            "training mode must actually apply dropout"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut optimizer = TransformerOptimizer::<f64>::new(TransformerOptimizerConfig::default())
            .expect("optimizer creation");
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), Array2::from_elem((2, 3), 0.5));
        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), Array2::from_elem((2, 3), 0.1));

        optimizer
            .step(&mut parameters, &mut gradients, 1.0)
            .expect("step");
        assert_eq!(optimizer.step_count(), 1);
        optimizer.reset().expect("reset");
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn sequence_buffer_is_bounded_per_parameter() {
        let mut buffer = SequenceBuffer::<f64>::new(3);
        for i in 0..10 {
            buffer.add_gradient("a", Array1::from_elem(2, i as f64));
            buffer.add_gradient("b", Array1::from_elem(5, i as f64));
        }
        assert_eq!(buffer.recent_gradients("a").len(), 3);
        assert_eq!(buffer.recent_gradients("b").len(), 3);
        assert_eq!(buffer.recent_gradients("a")[2].len(), 2);
        assert_eq!(buffer.recent_gradients("b")[2].len(), 5);
        assert!(buffer.recent_gradients("missing").is_empty());
    }
}
