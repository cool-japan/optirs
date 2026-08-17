// Transformer-Based Meta-Learning for Optimization
//
// This module implements transformer architectures specifically designed for
// meta-learning in optimization tasks. It includes attention mechanisms,
// sequence modeling for optimization trajectories, and advanced transformer
// architectures tailored for learning optimization strategies.

pub mod architecture;
pub mod attention;
pub mod config;
pub mod feedforward;
pub mod layers;
pub mod memory_manager;
pub mod meta_learning;
pub mod performance_tracker;
pub mod positional_encoding;
pub mod sequence_processor;
pub mod state;

// Re-export main types for backward compatibility
pub use architecture::{ArchitectureCache, TransformerArchitecture, TransformerLayer};
pub use attention::{AttentionMechanism, MultiHeadAttention};
pub use config::{TransformerArchConfig, TransformerBasedOptimizerConfig};
pub use feedforward::{ActivationFunction, FeedForwardNetwork};
pub use layers::{
    DropoutLayer, EmbeddingLayer, LayerNormalization, OutputProjection, ResidualConnections,
};
pub use memory_manager::{MemoryManagementStrategy, TransformerMemoryManager};
pub use meta_learning::{MetaLearningStrategy, TransformerMetaLearning};
pub use performance_tracker::{PerformanceMetrics, TransformerPerformanceTracker};
pub use positional_encoding::{PositionalEncoding, PositionalEncodingType};
pub use sequence_processor::{OptimizationSequenceProcessor, SequenceProcessingStrategy};
pub use state::{OptimizerStateSnapshot, TransformerOptimizerState};

// Re-export for backward compatibility - create alias for the old name
pub use TransformerBasedOptimizerConfig as TransformerOptimizerConfig;

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayBase, Axis, Data, Dimension};
use scirs2_core::numeric::{Float, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{
    LearnedOptimizerConfig, MetaOptimizationStrategy, NeuralOptimizerMetrics, NeuralOptimizerType,
    OptimizerState, TaskContext, TaskPerformance,
};
use crate::error::{OptimError, Result};
use optirs_core::adaptive_selection::OptimizerType;

// Import for external compatibility

/// Transformer-based meta-learning optimizer
pub struct TransformerOptimizer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Core transformer architecture
    transformer: TransformerArchitecture<T>,

    /// Positional encoding for sequence modeling
    positional_encoding: PositionalEncoding<T>,

    // NOTE: this struct used to also own a standalone `MultiHeadAttention` and
    // one `FeedForwardNetwork` per layer. Nothing ever read them: every
    // attention and feed-forward computation happens inside
    // `TransformerArchitecture`'s own `TransformerLayer`s. At the default
    // configuration (model_dimension 512, feedforward_dimension 2048, 6 layers)
    // those duplicates allocated ~12.6M dead parameters, which is why
    // constructing this optimizer took ~13s and its construction test was
    // `#[ignore]`d. They are gone.
    /// Meta-learning components
    meta_learning: TransformerMetaLearning<T>,

    /// Sequence processor for optimization trajectories
    sequence_processor: OptimizationSequenceProcessor<T>,

    /// Memory management for long sequences
    memory_manager: TransformerMemoryManager<T>,

    /// Configuration
    config: TransformerBasedOptimizerConfig<T>,

    /// Performance tracking
    performance_tracker: TransformerPerformanceTracker<T>,

    /// State management
    state: TransformerOptimizerState<T>,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + scirs2_core::ndarray::ScalarOperand
            + scirs2_core::numeric::FromPrimitive,
    > TransformerOptimizer<T>
{
    /// Create new transformer optimizer.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] if `config` (or any of its three
    /// sub-configurations — memory, meta-learning, performance) is inconsistent.
    ///
    /// F46: `TransformerBasedOptimizerConfig::validate` and the three sub-config
    /// validators it delegates to existed and were correct, but *nothing in the
    /// crate ever called them*. A `model_dimension` of 0, an
    /// `attention_head_dimension · num_attention_heads` that did not equal
    /// `model_dimension`, a `dropout_rate` of 2.0 or a zero `batch_size` were all
    /// accepted here and then produced either a wrong answer or a panic several
    /// layers down. The check now runs before anything is allocated.
    pub fn new(config: TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        config.validate().map_err(OptimError::InvalidConfig)?;

        let transformer_config = TransformerArchConfig::from_optimizer_config(&config);
        let transformer = TransformerArchitecture::new(transformer_config)?;

        let positional_encoding = PositionalEncoding::new(
            config.sequence_length,
            config.model_dimension,
            config.positional_encoding_type,
        )?;

        let meta_learning = TransformerMetaLearning::new(&config)?;
        let sequence_processor = OptimizationSequenceProcessor::new(&config)?;
        let memory_manager = TransformerMemoryManager::new(&config)?;
        let performance_tracker = TransformerPerformanceTracker::new();
        let state = TransformerOptimizerState::new(&config)?;

        Ok(Self {
            transformer,
            positional_encoding,
            meta_learning,
            sequence_processor,
            memory_manager,
            config,
            performance_tracker,
            state,
        })
    }

    /// Generate optimization step using transformer
    pub fn generate_optimization_step(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array1<T>> {
        let start_time = Instant::now();

        // Process input sequences
        let processed_sequence = self.sequence_processor.process_optimization_sequence(
            gradient_history,
            parameter_history,
            loss_history,
        )?;

        // Apply positional encoding
        let encoded_sequence = self.positional_encoding.encode(&processed_sequence)?;

        // Forward pass through transformer
        let transformer_output = self.transformer.forward(&encoded_sequence)?;

        // Generate optimization step
        let optimization_step = self
            .meta_learning
            .generate_update(&transformer_output, &self.state.current_parameters)?;

        // Update state
        self.state
            .update_with_step(&optimization_step, loss_history.last().copied())?;

        // Track performance
        let elapsed = start_time.elapsed();
        self.performance_tracker
            .record_optimization_step(elapsed, &optimization_step);

        Ok(optimization_step)
    }

    /// Train the transformer on optimization trajectories
    pub fn train_on_trajectories(
        &mut self,
        trajectories: &[OptimizationTrajectory<T>],
    ) -> Result<TrainingMetrics> {
        let start_time = Instant::now();
        let mut total_loss = T::zero();
        let mut batch_count = 0;
        let mut applied_gradient_energy = T::zero();

        for trajectory in trajectories {
            // Process trajectory into sequences
            let sequences = self
                .sequence_processor
                .trajectory_to_sequences(trajectory)?;

            for sequence in sequences {
                // Forward pass, recording the activations the backward pass needs
                let encoded = self.positional_encoding.encode(&sequence.input)?;
                let (prediction, cache) = self.transformer.forward_with_cache(&encoded)?;

                // Calculate loss
                let loss = self.calculate_sequence_loss(&prediction, &sequence.target)?;
                total_loss = total_loss + loss;

                // Backward pass: real gradients, real parameter update
                applied_gradient_energy = applied_gradient_energy
                    + self.backward_pass(&prediction, &sequence.target, &cache, loss)?;

                batch_count += 1;
            }
        }

        let avg_loss = if batch_count > 0 {
            total_loss
                / scirs2_core::numeric::NumCast::from(batch_count).unwrap_or_else(|| T::zero())
        } else {
            T::zero()
        };

        let training_time = start_time.elapsed();

        let metrics = TrainingMetrics {
            loss: avg_loss.to_f64().unwrap_or(0.0),
            training_time,
            num_sequences: batch_count,
            convergence_rate: self.calculate_convergence_rate()?,
            applied_gradient_energy: applied_gradient_energy.to_f64().unwrap_or(0.0),
        };

        self.performance_tracker
            .record_training_epoch(metrics.clone());

        Ok(metrics)
    }

    /// Forward pass through the transformer for a sequence (inference path).
    pub fn forward_sequence(&mut self, sequence: &Array2<T>) -> Result<Array2<T>> {
        // Apply positional encoding
        let encoded_sequence = self.positional_encoding.encode(sequence)?;

        // Forward through transformer
        self.transformer.forward(&encoded_sequence)
    }

    /// Calculate loss for sequence prediction
    fn calculate_sequence_loss(&self, prediction: &Array2<T>, target: &Array2<T>) -> Result<T> {
        if prediction.shape() != target.shape() {
            return Err(OptimError::Other(
                "Shape mismatch in loss calculation".to_string(),
            ));
        }

        if prediction.is_empty() {
            return Err(OptimError::InvalidConfig(
                "cannot take the mean squared error of an empty prediction".to_string(),
            ));
        }

        // Mean squared error
        let diff = prediction - target;
        let squared_diff = &diff * &diff;
        let sum = squared_diff.sum();
        let count = T::from(prediction.len()).ok_or_else(|| {
            OptimError::ComputationError(format!(
                "cannot represent the element count {} in the value type",
                prediction.len()
            ))
        })?;

        Ok(sum / count)
    }

    /// Backward pass: differentiate the sequence MSE and update the transformer.
    ///
    /// The loss is `L = mean((pred - target)²)` over all `N` cells, so the
    /// gradient handed to the network is `dL/dpred = 2·(pred - target)/N`. That
    /// gradient is propagated by [`TransformerArchitecture::backward`], which
    /// updates the output projection, the final layer norm, every layer's
    /// feed-forward sub-layer (with its layer norms) and the input embedding.
    /// The attention sub-layers ride their residual identity branch and stay
    /// frozen — the first-order approximation documented on that method.
    ///
    /// `enable_gradient_clipping` / `gradient_clip_value` are honoured by
    /// rescaling `dL/dpred` when its Frobenius norm exceeds the threshold.
    ///
    /// Returns `Err` on a shape mismatch instead of silently doing nothing.
    fn backward_pass(
        &mut self,
        prediction: &Array2<T>,
        target: &Array2<T>,
        cache: &ArchitectureCache<T>,
        loss: T,
    ) -> Result<T> {
        if prediction.shape() != target.shape() {
            return Err(OptimError::Other(format!(
                "Shape mismatch in backward pass: prediction {:?} vs target {:?}",
                prediction.shape(),
                target.shape()
            )));
        }

        let count = T::from(prediction.len()).unwrap_or_else(|| T::one());
        let two = T::one() + T::one();
        let mut grad_output = (prediction - target).mapv(|d| two * d / count);

        if self.config.enable_gradient_clipping {
            let norm = grad_output
                .iter()
                .map(|&g| g * g)
                .fold(T::zero(), |a, b| a + b)
                .sqrt();
            let threshold = self.config.gradient_clip_value;
            if threshold > T::zero() && norm > threshold {
                let scale = threshold / norm;
                grad_output.mapv_inplace(|g| g * scale);
            }
        }

        let applied = self
            .transformer
            .backward(cache, &grad_output, self.config.learning_rate)?;

        self.meta_learning.update_from_loss(loss)?;
        Ok(applied)
    }

    /// Calculate convergence rate
    fn calculate_convergence_rate(&self) -> Result<f64> {
        let loss_history = self.performance_tracker.get_loss_history();
        if loss_history.len() < 2 {
            return Ok(0.0);
        }

        // Most recent first; the window is the last 10 recorded losses.
        let recent_losses: Vec<_> = loss_history.iter().rev().take(10).collect();
        // `first`/`last` cannot fail after this guard, but pattern-matching says so
        // to the compiler instead of asserting it at runtime.
        let (Some(final_loss), Some(initial_loss)) = (recent_losses.first(), recent_losses.last())
        else {
            return Ok(0.0);
        };
        let (initial_loss, final_loss) = (**initial_loss, **final_loss);
        if initial_loss == 0.0 {
            // Already at zero loss: no relative improvement is definable.
            return Ok(0.0);
        }

        let improvement = (initial_loss - final_loss) / initial_loss;
        Ok(improvement.clamp(0.0, 1.0))
    }

    /// Adopt a new architecture configuration.
    ///
    /// The return value says exactly what happened, because the two cases differ
    /// in a way callers must not be left guessing about:
    ///
    /// * [`ArchitectureUpdate::Unchanged`] — the proposal matched the live
    ///   configuration; nothing was touched.
    /// * [`ArchitectureUpdate::InPlace`] — only non-structural fields differed
    ///   (`learning_rate`, `dropout_rate`, clipping, weight decay). They were
    ///   adopted and **all learned weights were kept**.
    /// * [`ArchitectureUpdate::Rebuilt`] — a structural dimension differed
    ///   (`model_dimension`, `num_transformer_layers`, `num_attention_heads`,
    ///   `attention_head_dimension`, `feedforward_dimension`, `sequence_length`,
    ///   `activation_function`, `positional_encoding_type`). Weights of one shape
    ///   cannot be reinterpreted at another, so the transformer stack, positional
    ///   encoding, meta-learning networks and optimizer state were **rebuilt from
    ///   a fresh initialization — every learned parameter was discarded.**
    ///
    /// That last case is destructive, which is why it is reported rather than
    /// signalled by a bare `true`. `AdaptiveTransformerEnhancement`'s architecture
    /// proposal is a function of the measured landscape rather than of the current
    /// architecture, so repeated enhancement converges and stops rebuilding
    /// instead of wiping the weights on every call.
    ///
    /// This is also what makes
    /// `AdaptiveTransformerEnhancement::enhance_optimizer` actually enhance the
    /// optimizer it is given; before, its `&mut TransformerOptimizer` argument
    /// was never touched.
    ///
    /// # Errors
    /// Propagates construction errors from the new configuration (e.g. a head
    /// count that does not divide the model dimension), leaving `self` unchanged.
    pub fn apply_architecture_config(
        &mut self,
        config: &TransformerBasedOptimizerConfig<T>,
    ) -> Result<ArchitectureUpdate> {
        let structural_change = config.model_dimension != self.config.model_dimension
            || config.num_transformer_layers != self.config.num_transformer_layers
            || config.num_attention_heads != self.config.num_attention_heads
            || config.attention_head_dimension != self.config.attention_head_dimension
            || config.feedforward_dimension != self.config.feedforward_dimension
            || config.sequence_length != self.config.sequence_length
            || config.activation_function != self.config.activation_function
            || config.positional_encoding_type != self.config.positional_encoding_type;

        if structural_change {
            // Build everything first so a failure leaves `self` untouched.
            let transformer_config = TransformerArchConfig::from_optimizer_config(config);
            let transformer = TransformerArchitecture::new(transformer_config)?;
            let positional_encoding = PositionalEncoding::new(
                config.sequence_length,
                config.model_dimension,
                config.positional_encoding_type,
            )?;
            let meta_learning = TransformerMetaLearning::new(config)?;
            let sequence_processor = OptimizationSequenceProcessor::new(config)?;
            let memory_manager = TransformerMemoryManager::new(config)?;
            let state = TransformerOptimizerState::new(config)?;

            self.transformer = transformer;
            self.positional_encoding = positional_encoding;
            self.meta_learning = meta_learning;
            self.sequence_processor = sequence_processor;
            self.memory_manager = memory_manager;
            self.state = state;
            self.config = config.clone();
            return Ok(ArchitectureUpdate::Rebuilt);
        }

        let non_structural_change = config.learning_rate != self.config.learning_rate
            || config.dropout_rate != self.config.dropout_rate
            || config.enable_gradient_clipping != self.config.enable_gradient_clipping
            || config.gradient_clip_value != self.config.gradient_clip_value
            || config.weight_decay != self.config.weight_decay;
        if non_structural_change {
            self.config = config.clone();
            return Ok(ArchitectureUpdate::InPlace);
        }

        Ok(ArchitectureUpdate::Unchanged)
    }

    /// The configuration currently in force.
    pub fn config(&self) -> &TransformerBasedOptimizerConfig<T> {
        &self.config
    }

    /// Total number of learnable parameters this optimizer owns.
    ///
    /// Exactly the transformer architecture plus the meta-learning adaptation
    /// networks — the only two subsystems that hold weights.
    pub fn parameter_count(&self) -> usize {
        self.transformer.parameter_count() + self.meta_learning.parameter_count()
    }

    /// Get current state
    pub fn get_state(&self) -> &TransformerOptimizerState<T> {
        &self.state
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &TransformerPerformanceTracker<T> {
        &self.performance_tracker
    }

    /// Reset optimizer state
    pub fn reset_state(&mut self) -> Result<()> {
        self.state = TransformerOptimizerState::new(&self.config)?;
        self.performance_tracker.reset();
        Ok(())
    }
}

/// Optimization trajectory for training
#[derive(Debug, Clone)]
pub struct OptimizationTrajectory<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub gradient_sequence: Array2<T>,
    pub parameter_sequence: Array2<T>,
    pub loss_sequence: Array1<T>,
    pub metadata: TrajectoryMetadata,
}

/// Trajectory metadata
#[derive(Debug, Clone)]
pub struct TrajectoryMetadata {
    pub task_id: String,
    pub optimizer_type: String,
    pub convergence_achieved: bool,
    pub total_steps: usize,
}

/// Training sequence
#[derive(Debug, Clone)]
pub struct TrainingSequence<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub input: Array2<T>,
    pub target: Array2<T>,
    pub sequence_length: usize,
}

/// Outcome of [`TransformerOptimizer::apply_architecture_config`].
///
/// Distinguishes a no-op from a weight-preserving update from a **destructive**
/// rebuild, so a caller can never mistake the third for the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureUpdate {
    /// The proposal already matched the live configuration.
    Unchanged,
    /// Only non-structural fields changed; all learned weights were kept.
    InPlace,
    /// A structural dimension changed, so every learned parameter was discarded
    /// and re-initialized.
    Rebuilt,
}

impl ArchitectureUpdate {
    /// Whether anything changed at all.
    pub fn changed(self) -> bool {
        !matches!(self, ArchitectureUpdate::Unchanged)
    }

    /// Whether learned parameters were discarded.
    pub fn discarded_weights(self) -> bool {
        matches!(self, ArchitectureUpdate::Rebuilt)
    }
}

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub training_time: Duration,
    pub num_sequences: usize,
    pub convergence_rate: f64,
    /// Total squared magnitude of the gradient that reached the input embedding
    /// across every sequence in this epoch.
    ///
    /// This is strictly positive whenever a backward pass actually propagated a
    /// non-zero gradient, and exactly zero when nothing was propagated — which
    /// is what distinguishes the real backward pass from the previous no-op
    /// that only recorded the loss.
    pub applied_gradient_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small but structurally complete configuration for the training tests.
    fn small_config() -> TransformerBasedOptimizerConfig<f64> {
        TransformerBasedOptimizerConfig::<f64> {
            model_dimension: 8,
            num_transformer_layers: 2,
            num_attention_heads: 2,
            attention_head_dimension: 4,
            feedforward_dimension: 16,
            sequence_length: 8,
            learning_rate: 1e-2,
            dropout_rate: 0.0,
            ..Default::default()
        }
    }

    fn trajectory(steps: usize, width: usize, scale: f64) -> OptimizationTrajectory<f64> {
        let gradient_sequence = Array2::from_shape_fn((steps, width), |(i, j)| {
            scale * ((i * width + j) as f64 * 0.017).sin()
        });
        let parameter_sequence = Array2::from_shape_fn((steps, width), |(i, j)| {
            scale * ((i * width + j) as f64 * 0.031).cos()
        });
        let loss_sequence = Array1::from_shape_fn(steps, |i| scale / (1.0 + i as f64));
        OptimizationTrajectory {
            gradient_sequence,
            parameter_sequence,
            loss_sequence,
            metadata: TrajectoryMetadata {
                task_id: "regression".to_string(),
                optimizer_type: "adam".to_string(),
                convergence_achieved: false,
                total_steps: steps,
            },
        }
    }

    #[test]
    fn test_transformer_optimizer_creation() {
        let config = TransformerBasedOptimizerConfig::default();
        let optimizer = TransformerOptimizer::<f32>::new(config);
        assert!(optimizer.is_ok());
    }

    /// F46: `TransformerBasedOptimizerConfig::validate` (and the memory /
    /// meta-learning / performance validators it delegates to) had **zero call
    /// sites in the crate**. Every one of the configurations below used to be
    /// accepted by `TransformerOptimizer::new`, which then either produced a
    /// silently wrong answer or panicked somewhere downstream.
    ///
    /// The list deliberately covers the top-level validator *and* all three
    /// sub-config validators, because delegation is the part that is easy to
    /// drop.
    #[test]
    fn construction_rejects_every_invalid_configuration() {
        // A small but valid baseline, so each case below differs in exactly one
        // field and the default's own consistency is not what is under test.
        let base = || TransformerBasedOptimizerConfig::<f32> {
            model_dimension: 8,
            num_transformer_layers: 1,
            num_attention_heads: 2,
            attention_head_dimension: 4,
            feedforward_dimension: 16,
            sequence_length: 4,
            batch_size: 2,
            ..Default::default()
        };
        assert!(
            TransformerOptimizer::<f32>::new(base()).is_ok(),
            "the baseline configuration must itself be valid"
        );

        let mut cases: Vec<(&str, TransformerBasedOptimizerConfig<f32>)> = Vec::new();

        // --- top-level validator ---
        cases.push(("model_dimension = 0", {
            let mut c = base();
            c.model_dimension = 0;
            c
        }));
        cases.push(("num_transformer_layers = 0", {
            let mut c = base();
            c.num_transformer_layers = 0;
            c
        }));
        cases.push(("num_attention_heads = 0", {
            let mut c = base();
            c.num_attention_heads = 0;
            c
        }));
        cases.push(("heads do not divide model_dimension", {
            let mut c = base();
            c.num_attention_heads = 3;
            c
        }));
        cases.push(("head_dim * heads != model_dimension", {
            let mut c = base();
            c.attention_head_dimension = 3;
            c
        }));
        cases.push(("sequence_length = 0", {
            let mut c = base();
            c.sequence_length = 0;
            c
        }));
        cases.push(("dropout_rate above 1", {
            let mut c = base();
            c.dropout_rate = 2.0;
            c
        }));
        cases.push(("negative learning_rate", {
            let mut c = base();
            c.learning_rate = -1.0;
            c
        }));
        cases.push(("batch_size = 0", {
            let mut c = base();
            c.batch_size = 0;
            c
        }));

        // --- delegated: MemoryConfig ---
        cases.push(("max_cache_size = 0", {
            let mut c = base();
            c.memory_config.max_cache_size = 0;
            c
        }));
        cases.push(("allocation_block_size = 0", {
            let mut c = base();
            c.memory_config.allocation_block_size = 0;
            c
        }));

        // --- delegated: MetaLearningConfig ---
        cases.push(("meta_learning_rate = 0", {
            let mut c = base();
            c.meta_learning_config.meta_learning_rate = 0.0;
            c
        }));
        cases.push(("inner_steps = 0", {
            let mut c = base();
            c.meta_learning_config.inner_steps = 0;
            c
        }));
        cases.push(("num_support = 0", {
            let mut c = base();
            c.meta_learning_config.num_support = 0;
            c
        }));

        // --- delegated: PerformanceConfig ---
        cases.push(("metrics_interval = 0", {
            let mut c = base();
            c.performance_config.metrics_interval = 0;
            c
        }));
        cases.push(("max_history_size = 0", {
            let mut c = base();
            c.performance_config.max_history_size = 0;
            c
        }));

        for (label, config) in cases {
            let outcome = TransformerOptimizer::<f32>::new(config);
            assert!(
                outcome.is_err(),
                "`{label}` was accepted by TransformerOptimizer::new"
            );
        }
    }

    /// F62: this struct used to also own a standalone `MultiHeadAttention`
    /// (4·512² = 1.05M) plus one `FeedForwardNetwork` per layer
    /// (6 · 2·512·2048 = 12.6M) that nothing ever read. Those duplicates are
    /// gone, so the optimizer's live parameter count must now equal exactly the
    /// sum of the two subsystems that actually compute — no more, no less.
    #[test]
    fn no_dead_duplicate_parameter_blocks() {
        let config = TransformerBasedOptimizerConfig::<f32>::default();
        let model_dim = config.model_dimension;
        let ff_dim = config.feedforward_dimension;
        let layers = config.num_transformer_layers;
        let optimizer = TransformerOptimizer::<f32>::new(config).expect("default construction");

        let live = optimizer.parameter_count();
        let dead_attention = 4 * model_dim * model_dim;
        let dead_feedforward = layers * 2 * model_dim * ff_dim;

        assert!(
            live > 0,
            "the optimizer must own some parameters ({live} found)"
        );
        // The removed duplicates were >13M parameters; if they came back the
        // live count would exceed the architecture+meta total by that much.
        let expected =
            optimizer.transformer.parameter_count() + optimizer.meta_learning.parameter_count();
        assert_eq!(
            live,
            expected,
            "live parameter count {live} does not match \
             transformer+meta {expected}; something allocated {} extra \
             (dead attention {dead_attention}, dead feed-forward {dead_feedforward})",
            live.saturating_sub(expected)
        );
    }

    /// F26: `backward_pass` used to only call `update_from_loss` — no gradients,
    /// no parameter update. A real backward pass must (a) report non-zero
    /// applied gradient energy and (b) actually move the network's outputs.
    #[test]
    fn train_on_trajectories_propagates_real_gradients() {
        let config = small_config();
        let width = config.model_dimension;
        let mut optimizer =
            TransformerOptimizer::<f64>::new(config).expect("optimizer construction");

        let probe = Array2::from_shape_fn((4, width), |(i, j)| ((i + j) as f64 * 0.05).tanh());
        let before = optimizer
            .forward_sequence(&probe)
            .expect("forward before training");

        let trajectories = vec![trajectory(12, width, 1.0)];
        let metrics = optimizer
            .train_on_trajectories(&trajectories)
            .expect("training");

        assert!(
            metrics.num_sequences > 0,
            "the trajectory must produce at least one training sequence"
        );
        assert!(
            metrics.applied_gradient_energy > 0.0,
            "backward pass propagated no gradient (energy {})",
            metrics.applied_gradient_energy
        );

        let after = optimizer
            .forward_sequence(&probe)
            .expect("forward after training");
        let moved = (&after - &before)
            .iter()
            .map(|d| d.abs())
            .fold(0.0, f64::max);
        assert!(
            moved > 1e-12,
            "training left the network unchanged (max output delta {moved})"
        );
    }

    /// F26: repeated epochs on the same data must reduce the training loss —
    /// the old no-op backward pass produced a flat loss curve forever.
    #[test]
    fn repeated_training_epochs_reduce_the_loss() {
        let config = small_config();
        let width = config.model_dimension;
        let mut optimizer =
            TransformerOptimizer::<f64>::new(config).expect("optimizer construction");

        let trajectories = vec![trajectory(12, width, 1.0)];
        let first = optimizer
            .train_on_trajectories(&trajectories)
            .expect("epoch 1")
            .loss;
        let mut last = first;
        for _ in 0..12 {
            last = optimizer
                .train_on_trajectories(&trajectories)
                .expect("later epoch")
                .loss;
        }

        assert!(
            last < first,
            "loss did not decrease across epochs: {first} -> {last}"
        );
    }

    /// F26: a shape mismatch must be an error, not a silently ignored no-op.
    #[test]
    fn sequence_loss_rejects_shape_mismatch() {
        let config = small_config();
        let optimizer = TransformerOptimizer::<f64>::new(config).expect("construction");
        let prediction = Array2::<f64>::zeros((3, 8));
        let target = Array2::<f64>::zeros((4, 8));
        assert!(optimizer
            .calculate_sequence_loss(&prediction, &target)
            .is_err());
    }

    #[test]
    fn test_trajectory_creation() {
        let trajectory = OptimizationTrajectory::<f32> {
            gradient_sequence: Array2::zeros((10, 5)),
            parameter_sequence: Array2::zeros((10, 5)),
            loss_sequence: Array1::zeros(10),
            metadata: TrajectoryMetadata {
                task_id: "test".to_string(),
                optimizer_type: "adam".to_string(),
                convergence_achieved: true,
                total_steps: 10,
            },
        };

        assert_eq!(trajectory.gradient_sequence.shape(), &[10, 5]);
        assert_eq!(trajectory.loss_sequence.len(), 10);
    }
}
