// LSTM-based Neural Optimizer
//
// This module implements a learned optimizer using LSTM networks to adaptively
// update optimization parameters. The LSTM learns optimization strategies through
// meta-learning, enabling automatic discovery of effective optimization patterns.

use scirs2_core::ndarray::{s, Array, Array1, Array2, ArrayBase, Data, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::random::{rngs::StdRng, seeded_rng, thread_rng, CoreRandom};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::LearnedOptimizerConfig;
use crate::common::cast_scalar;
use crate::error::{OptimError, Result};

pub mod bptt;
pub mod components;
pub mod features;
pub mod introspection;
pub mod trainer;

pub use bptt::{
    BpttConfig, DiagonalQuadraticTask, FrozenRollout, MetaTrainingTask, NetworkGradients,
};
pub use features::build_lstm_features;
pub use trainer::MetaTrainer;

/// LSTM-based neural optimizer with meta-learning capabilities
#[derive(Debug)]
pub struct LSTMOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for the LSTM optimizer
    config: LearnedOptimizerConfig,

    /// LSTM network architecture
    lstm_network: LSTMNetwork<T>,

    /// Gradient and parameter history for context
    history_buffer: HistoryBuffer<T>,

    /// Meta-learning components
    meta_learner: MetaLearner<T>,

    /// Adaptive learning rate controller
    lr_controller: AdaptiveLearningRateController<T>,

    /// Optimization state tracker
    state_tracker: OptimizationStateTracker<T>,

    /// Performance metrics
    metrics: LSTMOptimizerMetrics,

    /// Current optimization step
    step_count: usize,
}

/// LSTM network architecture for optimization
#[derive(Debug, Clone)]
pub struct LSTMNetwork<T: Float + Debug + Send + Sync + 'static> {
    /// LSTM layers
    layers: Vec<LSTMLayer<T>>,

    /// Output projection layer
    output_projection: OutputProjection<T>,

    /// Attention mechanism (optional)
    attention: Option<AttentionMechanism<T>>,

    /// Normalization layers
    layer_norms: Vec<LayerNormalization<T>>,

    /// Dropout for regularization
    dropout_rate: f64,

    /// Whether dropout is active.
    ///
    /// Defaults to `false` (evaluation mode). Dropout used to be applied
    /// unconditionally whenever `dropout_rate > 0.0`, which made
    /// `LSTMOptimizer::lstm_step` non-deterministic and injected training-time
    /// noise into deployed updates. It also made meta-training impossible to
    /// verify: the truncated-BPTT gradient in [`bptt`] is the gradient of a
    /// *deterministic* rollout. Call [`LSTMNetwork::set_training`] to turn it on
    /// deliberately.
    training: bool,
}

/// Individual LSTM layer
#[derive(Debug, Clone)]
pub struct LSTMLayer<T: Float + Debug + Send + Sync + 'static> {
    /// Input-to-hidden weights (for i, f, g, o gates)
    weight_ih: Array2<T>,

    /// Hidden-to-hidden weights (for i, f, g, o gates)
    weight_hh: Array2<T>,

    /// Input biases
    bias_ih: Array1<T>,

    /// Hidden biases
    bias_hh: Array1<T>,

    /// Hidden state
    hidden_state: Array1<T>,

    /// Cell state
    cell_state: Array1<T>,

    /// Hidden size
    hiddensize: usize,
}

/// Output projection for generating parameter updates
#[derive(Debug, Clone)]
pub struct OutputProjection<T: Float + Debug + Send + Sync + 'static> {
    /// Projection weights
    weights: Array2<T>,

    /// Projection biases
    bias: Array1<T>,

    /// Output transformation
    output_transform: OutputTransform,
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> OutputProjection<T> {
    /// Create a new output projection with Xavier-initialized weights.
    ///
    /// The output-side transform (tanh / scaling / ...) named by
    /// `output_transform` is applied by the caller (see
    /// `LSTMOptimizer::generate_updates`); this projection itself computes the
    /// underlying linear map `W x + b`.
    pub fn new(
        input_size: usize,
        output_size: usize,
        output_transform: OutputTransform,
    ) -> Result<Self> {
        Self::new_with_rng(
            input_size,
            output_size,
            output_transform,
            &mut seeded_rng(thread_rng().random::<u64>()),
        )
    }

    /// [`Self::new`] with the caller's generator, for reproducible construction
    /// (see [`LSTMNetwork::new_seeded`]).
    pub(crate) fn new_with_rng(
        input_size: usize,
        output_size: usize,
        output_transform: OutputTransform,
        rng: &mut CoreRandom<StdRng>,
    ) -> Result<Self> {
        // Xavier/Glorot *uniform* limit is sqrt(6 / (fan_in + fan_out)); the
        // sqrt(2 / ...) that used to be here is the limit for a Xavier *normal*
        // draw, and `xavier_init` samples uniformly. Uniform[-b, b] has variance
        // b^2/3, so the old constant gave exactly one third of the intended
        // variance in every LSTM weight matrix in this file (finding F64).
        let limit = LSTMLayer::<T>::xavier_limit(input_size, output_size);
        let weights = LSTMLayer::<T>::xavier_init(rng, output_size, input_size, limit);
        let bias = Array1::zeros(output_size);

        Ok(Self {
            weights,
            bias,
            output_transform,
        })
    }

    /// Forward pass through output projection: `output = W * input + b`.
    pub fn forward(&self, input: &Array1<T>) -> Result<Array1<T>> {
        if input.len() != self.weights.ncols() {
            return Err(OptimError::InvalidConfig(format!(
                "OutputProjection expected {} input features, got {}",
                self.weights.ncols(),
                input.len()
            )));
        }
        Ok(self.weights.dot(input) + &self.bias)
    }

    /// Re-randomize the projection weights and zero the bias.
    ///
    /// Used when the target output dimension changes at runtime (e.g. the
    /// flattened parameter count the optimizer is asked to update does not
    /// match this projection's configured `output_features`): a fresh Xavier
    /// draw at the new shape is the honest response, since there is no
    /// meaningful way to reuse weights trained for a different output size.
    pub fn reset(&mut self, input_size: usize, output_size: usize) {
        let limit = LSTMLayer::<T>::xavier_limit(input_size, output_size);
        // A runtime reshape is inherently a fresh draw; entropy seeding is the
        // honest choice here (reproducible runs should avoid triggering it).
        let mut rng = seeded_rng(thread_rng().random::<u64>());
        self.weights = LSTMLayer::<T>::xavier_init(&mut rng, output_size, input_size, limit);
        self.bias = Array1::zeros(output_size);
    }

    /// Current output dimension.
    pub fn output_size(&self) -> usize {
        self.weights.nrows()
    }
}

/// Output transformation types
#[derive(Debug, Clone, Copy)]
pub enum OutputTransform {
    /// Direct output (no transformation)
    Identity,

    /// Tanh activation
    Tanh,

    /// Scaled tanh for bounded updates
    ScaledTanh { scale: f64 },

    /// Adaptive scaling based on gradient norms
    AdaptiveScale,

    /// Learned nonlinear transformation
    LearnedNonlinear,
}

/// Attention mechanism for focusing on relevant history
#[derive(Debug, Clone)]
pub struct AttentionMechanism<T: Float + Debug + Send + Sync + 'static> {
    /// Query projection
    query_proj: Array2<T>,

    /// Key projection
    key_proj: Array2<T>,

    /// Value projection
    value_proj: Array2<T>,

    /// Output projection
    output_proj: Array2<T>,

    /// Number of attention heads
    num_heads: usize,

    /// Attention head size
    head_size: usize,

    /// Attention weights from last forward pass
    attentionweights: Option<Array2<T>>,

    /// Raw pre-softmax logits from the last forward pass, one per head.
    ///
    /// At sequence length 1 the softmax weights are identically `1.0` for every
    /// head, so they carry no information about how the heads differ. The scaled
    /// dot products `qₕ·kₕ / √d` do, which is what
    /// [`LSTMOptimizer::compute_attention_stats`] measures head diversity from.
    head_logits: Option<Array1<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> AttentionMechanism<T> {
    /// Create a new attention mechanism with Xavier-initialized projections.
    pub fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        Self::new_with_rng(config, &mut seeded_rng(thread_rng().random::<u64>()))
    }

    /// [`Self::new`] with the caller's generator, for reproducible construction
    /// (see [`LSTMNetwork::new_seeded`]).
    pub(crate) fn new_with_rng(
        config: &LearnedOptimizerConfig,
        rng: &mut CoreRandom<StdRng>,
    ) -> Result<Self> {
        let hiddensize = config.hidden_size;
        let num_heads = config.attention_heads.max(1);
        if !hiddensize.is_multiple_of(num_heads) {
            return Err(OptimError::InvalidConfig(format!(
                "attention_heads ({num_heads}) must evenly divide hidden_size ({hiddensize})"
            )));
        }
        let head_size = hiddensize / num_heads;
        // Square projections: fan_in == fan_out == hiddensize.
        let limit = LSTMLayer::<T>::xavier_limit(hiddensize, hiddensize);

        Ok(Self {
            query_proj: LSTMLayer::<T>::xavier_init(rng, hiddensize, hiddensize, limit),
            key_proj: LSTMLayer::<T>::xavier_init(rng, hiddensize, hiddensize, limit),
            value_proj: LSTMLayer::<T>::xavier_init(rng, hiddensize, hiddensize, limit),
            output_proj: LSTMLayer::<T>::xavier_init(rng, hiddensize, hiddensize, limit),
            num_heads,
            head_size,
            attentionweights: None,
            head_logits: None,
        })
    }

    /// Scaled dot-product self-attention.
    ///
    /// The optimizer calls this once per optimization step with a *single*
    /// hidden-state vector (there is no sequence axis at this call site), so
    /// per head the attention distribution is always over exactly one key and
    /// therefore softmaxes to the constant `1.0`: the attended value degenerates
    /// to `V` itself. The formula is exactly standard SDPA at sequence length 1
    /// rather than an ad hoc shortcut.
    ///
    /// The degenerate weights are recorded in the private `attentionweights`
    /// field, and the *pre-softmax* scaled dot products — which, unlike the
    /// weights, do vary with the input and between heads — in `head_logits`, so
    /// the optimizer's attention statistics (surfaced through
    /// [`LSTMOptimizer::get_metrics`]) have something informative to report for
    /// head diversity.
    pub fn forward(&mut self, input: &Array1<T>) -> Result<Array1<T>> {
        let dim = self.query_proj.nrows();
        if input.len() != dim {
            return Err(OptimError::InvalidConfig(format!(
                "AttentionMechanism expected {dim} features, got {}",
                input.len()
            )));
        }

        let query = self.query_proj.dot(input);
        let key = self.key_proj.dot(input);
        let value = self.value_proj.dot(input);

        let head_scale = T::one()
            / scirs2_core::numeric::NumCast::from(self.head_size)
                .unwrap_or_else(T::one)
                .sqrt();
        let mut weights = Array2::zeros((self.num_heads, 1));
        let mut logits = Array1::zeros(self.num_heads);
        for h in 0..self.num_heads {
            let start = h * self.head_size;
            let end = start + self.head_size;
            let score = query
                .slice(s![start..end])
                .iter()
                .zip(key.slice(s![start..end]).iter())
                .fold(T::zero(), |acc, (&q, &k)| acc + q * k)
                * head_scale;
            logits[h] = score;
            // softmax over a single logit: exp(score) / exp(score) == 1,
            // independent of `score`'s value. The score itself is kept in
            // `head_logits` because it is the only per-head quantity that
            // survives the degenerate softmax.
            weights[[h, 0]] = T::one();
        }
        self.attentionweights = Some(weights);
        self.head_logits = Some(logits);

        Ok(self.output_proj.dot(&value))
    }
}

/// Layer normalization for stable training
#[derive(Debug, Clone)]
pub struct LayerNormalization<T: Float + Debug + Send + Sync + 'static> {
    /// Scale parameters
    gamma: Array1<T>,

    /// Shift parameters
    beta: Array1<T>,

    /// Epsilon for numerical stability
    epsilon: T,
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> LayerNormalization<T> {
    /// Create a new layer normalization
    pub fn new(features: usize) -> Result<Self> {
        Ok(Self {
            gamma: Array1::ones(features),
            beta: Array1::zeros(features),
            epsilon: scirs2_core::numeric::NumCast::from(1e-5).unwrap_or_else(|| T::zero()),
        })
    }

    /// Forward pass through layer normalization: standardise `input` to zero
    /// mean / unit variance across its features, then apply the learned
    /// affine transform `gamma * x_hat + beta`.
    pub fn forward(&self, input: &Array1<T>) -> Result<Array1<T>> {
        if input.len() != self.gamma.len() {
            return Err(OptimError::InvalidConfig(format!(
                "LayerNormalization expected {} features, got {}",
                self.gamma.len(),
                input.len()
            )));
        }
        if input.is_empty() {
            return Ok(input.clone());
        }
        let n = scirs2_core::numeric::NumCast::from(input.len()).unwrap_or_else(T::one);
        let mean = input.iter().copied().fold(T::zero(), |a, b| a + b) / n;
        let variance = input
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |a, b| a + b)
            / n;
        let inv_std = T::one() / (variance + self.epsilon).sqrt();

        let mut output = Array1::zeros(input.len());
        for i in 0..input.len() {
            output[i] = (input[i] - mean) * inv_std * self.gamma[i] + self.beta[i];
        }
        Ok(output)
    }
}

/// History buffer for maintaining context
#[derive(Debug, Clone)]
pub struct HistoryBuffer<T: Float + Debug + Send + Sync + 'static> {
    /// Gradient history
    gradients: VecDeque<Array1<T>>,

    /// Parameter history
    parameters: VecDeque<Array1<T>>,

    /// Loss history
    losses: VecDeque<T>,

    /// Maximum history length
    _maxlength: usize,

    /// Preprocessed features cache
    feature_cache: Option<Array2<T>>,
}

/// Meta-learning component for optimizer adaptation.
///
/// `MetaLearner::step` implements one fixed rule: a truncated-BPTT unroll of
/// the controller over each task, i.e. the second-order / MAML-style inner loop.
/// It carried a `strategy: MetaOptimizationStrategy` field that was hard-wired to
/// `MAML` at construction and never consulted, which advertised a choice the
/// implementation does not offer; the alternatives (Reptile, first-order,
/// custom) are not implemented for this controller.
#[derive(Debug, Clone)]
pub struct MetaLearner<T: Float + Debug + Send + Sync + 'static> {
    /// Meta-parameters (optimizer parameters)
    meta_parameters: HashMap<String, Array1<T>>,

    /// Meta-gradients accumulator
    meta_gradients: HashMap<String, Array1<T>>,

    /// Task history for meta-learning
    task_history: VecDeque<MetaTask<T>>,

    /// Meta-learning state
    meta_state: MetaLearningState<T>,

    /// Transfer learning capabilities
    transfer_learner: TransferLearner<T>,
}

/// Meta-learning task
#[derive(Debug, Clone)]
pub struct MetaTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    pub id: String,

    /// Task type
    pub task_type: TaskType,

    /// Training trajectory
    pub training_trajectory: Vec<TrajectoryPoint<T>>,

    /// Final performance
    pub final_performance: T,

    /// Task characteristics
    pub characteristics: TaskCharacteristics<T>,

    /// Task weight for meta-learning
    pub weight: T,
}

/// Types of optimization tasks
#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    /// Standard supervised learning
    SupervisedLearning,

    /// Reinforcement learning
    ReinforcementLearning,

    /// Unsupervised learning
    UnsupervisedLearning,

    /// Few-shot learning
    FewShotLearning,

    /// Online learning
    OnlineLearning,

    /// Adversarial training
    AdversarialTraining,
}

/// Point in optimization trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryPoint<T: Float + Debug + Send + Sync + 'static> {
    /// Step number
    pub step: usize,

    /// Gradient at this step
    pub gradient: Array1<T>,

    /// Parameters at this step
    pub parameters: Array1<T>,

    /// Loss at this step
    pub loss: T,

    /// Learning rate used
    pub learning_rate: T,

    /// Update direction
    pub update: Array1<T>,
}

/// Task characteristics for meta-learning
#[derive(Debug, Clone)]
pub struct TaskCharacteristics<T: Float + Debug + Send + Sync + 'static> {
    /// Problem dimensionality
    pub dimensionality: usize,

    /// Loss landscape curvature estimate
    pub curvature: T,

    /// Noise level estimate
    pub noise_level: T,

    /// Conditioning number estimate
    pub conditioning: T,

    /// Convergence difficulty
    pub difficulty: T,

    /// Task domain features
    pub domain_features: Array1<T>,
}

/// Meta-learning state
#[derive(Debug, Clone)]
pub struct MetaLearningState<T: Float + Debug + Send + Sync + 'static> {
    /// Current meta-learning step
    pub meta_step: usize,

    /// Meta-learning rate
    pub meta_lr: T,

    /// Adaptation rate
    pub adaptation_rate: T,

    /// Meta-validation performance
    pub meta_validation_performance: T,

    /// Task adaptation history
    pub adaptation_history: VecDeque<AdaptationEvent<T>>,

    /// Inner loop state
    pub inner_loop_state: InnerLoopState<T>,
}

/// Adaptation event tracking
#[derive(Debug, Clone)]
pub struct AdaptationEvent<T: Float + Debug + Send + Sync + 'static> {
    /// Source task
    pub source_task: String,

    /// Target task
    pub target_task: String,

    /// Adaptation steps required
    pub adaptation_steps: usize,

    /// Transfer efficiency
    pub transfer_efficiency: T,

    /// Final performance improvement
    pub performance_improvement: T,
}

/// Inner loop optimization state
#[derive(Debug, Clone)]
pub struct InnerLoopState<T: Float + Debug + Send + Sync + 'static> {
    /// Current inner step
    pub inner_step: usize,

    /// Inner loop parameters
    pub inner_parameters: Array1<T>,

    /// Inner loop optimizer state
    pub inner_optimizer_state: HashMap<String, Array1<T>>,

    /// Inner loop performance
    pub inner_performance: T,
}

/// Transfer learning component
#[derive(Debug, Clone)]
pub struct TransferLearner<T: Float + Debug + Send + Sync + 'static> {
    /// Source domain knowledge
    pub source_knowledge: HashMap<String, Array1<T>>,

    /// Domain adaptation parameters
    pub adaptation_parameters: Array1<T>,

    /// Transfer efficiency metrics
    pub transfer_metrics: TransferMetrics<T>,

    /// Domain similarity estimator
    pub similarity_estimator: DomainSimilarityEstimator<T>,
}

/// Transfer learning metrics
#[derive(Debug, Clone)]
pub struct TransferMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Transfer efficiency
    pub efficiency: T,

    /// Adaptation speed
    pub adaptation_speed: T,

    /// Knowledge retention
    pub knowledge_retention: T,

    /// Negative transfer detection
    pub negative_transfer_score: T,
}

/// Domain similarity estimator
#[derive(Debug, Clone)]
pub struct DomainSimilarityEstimator<T: Float + Debug + Send + Sync + 'static> {
    /// Domain embeddings
    pub domain_embeddings: HashMap<String, Array1<T>>,

    /// Similarity metric parameters
    pub similarity_params: Array1<T>,

    /// Learned similarity function
    pub similarity_function: SimilarityFunction,
}

/// Similarity function types
#[derive(Debug, Clone, Copy)]
pub enum SimilarityFunction {
    /// Cosine similarity
    Cosine,

    /// Euclidean distance
    Euclidean,

    /// Learned metric
    LearnedMetric,

    /// Task-specific similarity
    TaskSpecific,
}

/// Adaptive learning rate controller
#[derive(Debug, Clone)]
pub struct AdaptiveLearningRateController<T: Float + Debug + Send + Sync + 'static> {
    /// Base learning rate
    base_lr: T,

    /// Current learning rate
    current_lr: T,

    /// Learning rate adaptation parameters
    adaptation_params: LRAdaptationParams<T>,

    /// Learning rate history
    lr_history: VecDeque<T>,

    /// Performance-based adaptation
    performance_tracker: PerformanceTracker<T>,
}

/// Learning rate adaptation parameters
#[derive(Debug, Clone)]
pub struct LRAdaptationParams<T: Float + Debug + Send + Sync + 'static> {
    /// Momentum for LR adaptation
    pub momentum: T,

    /// Sensitivity to gradient changes
    pub gradient_sensitivity: T,

    /// Sensitivity to loss changes
    pub loss_sensitivity: T,

    /// Minimum learning rate
    pub min_lr: T,

    /// Maximum learning rate
    pub max_lr: T,

    /// Adaptation rate
    pub adaptation_rate: T,
}

/// Performance tracker for adaptive learning rate
#[derive(Debug, Clone)]
pub struct PerformanceTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Recent loss values
    recent_losses: VecDeque<T>,

    /// Performance trend
    trend: PerformanceTrend,

    /// Stagnation detection
    stagnation_counter: usize,

    /// Best performance seen
    best_performance: T,

    /// Performance improvement rate
    improvement_rate: T,
}

/// Performance trend indicators
#[derive(Debug, Clone, Copy)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving,

    /// Performance is stagnating
    Stagnating,

    /// Performance is degrading
    Degrading,

    /// Performance is oscillating
    Oscillating,

    /// Insufficient data
    Unknown,
}

/// Optimization state tracker
#[derive(Debug, Clone)]
pub struct OptimizationStateTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Current optimization phase
    phase: OptimizationPhase,

    /// Convergence indicators
    convergence_indicators: ConvergenceIndicators<T>,

    /// Gradient analysis
    gradient_analyzer: GradientAnalyzer<T>,

    /// Loss landscape analysis
    landscape_analyzer: LossLandscapeAnalyzer<T>,

    /// Stability metrics
    stability_metrics: StabilityMetrics<T>,

    /// Previous step's gradient, for the direction-consistency, noise and
    /// secant-curvature estimates computed by
    /// [`OptimizationStateTracker::update`].
    previous_gradient: Option<Array1<T>>,

    /// Previous step's loss, for the loss-change trend.
    previous_loss: Option<T>,

    /// Number of observed steps, for the running moments.
    step_count: usize,
}

/// Optimization phases
#[derive(Debug, Clone, Copy)]
pub enum OptimizationPhase {
    /// Initial rapid descent
    InitialDescent,

    /// Steady progress
    SteadyProgress,

    /// Fine-tuning
    FineTuning,

    /// Converged
    Converged,

    /// Stuck/Plateau
    Plateau,

    /// Diverging
    Diverging,
}

/// Convergence indicators
#[derive(Debug, Clone)]
pub struct ConvergenceIndicators<T: Float + Debug + Send + Sync + 'static> {
    /// Gradient norm trend
    pub gradient_norm_trend: Vec<T>,

    /// Loss change trend
    pub loss_change_trend: Vec<T>,

    /// Parameter change magnitude
    pub parameter_change_magnitude: T,

    /// Convergence probability
    pub convergence_probability: T,

    /// Estimated steps to convergence
    pub estimated_steps_to_convergence: Option<usize>,
}

/// Gradient analysis component
#[derive(Debug, Clone)]
pub struct GradientAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Gradient statistics
    pub gradient_stats: GradientStatistics<T>,

    /// Gradient correlation tracking
    pub correlation_tracker: GradientCorrelationTracker<T>,

    /// Gradient noise estimation
    pub noise_estimator: GradientNoiseEstimator<T>,

    /// Gradient flow analysis
    pub flow_analyzer: GradientFlowAnalyzer<T>,
}

/// Gradient statistics
#[derive(Debug, Clone)]
pub struct GradientStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Mean gradient norm
    pub mean_norm: T,

    /// Gradient norm variance
    pub norm_variance: T,

    /// Gradient direction consistency
    pub direction_consistency: T,

    /// Gradient magnitude distribution
    pub magnitude_distribution: Vec<T>,

    /// Component-wise statistics
    pub component_stats: Array1<T>,
}

/// Gradient correlation tracker
#[derive(Debug, Clone)]
pub struct GradientCorrelationTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Correlation matrix
    pub correlation_matrix: Array2<T>,

    /// Temporal correlations
    pub temporal_correlations: VecDeque<T>,

    /// Cross-parameter correlations
    pub cross_correlations: HashMap<String, T>,
}

/// Gradient noise estimator
#[derive(Debug, Clone)]
pub struct GradientNoiseEstimator<T: Float + Debug + Send + Sync + 'static> {
    /// Estimated noise level
    pub noise_level: T,

    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: T,

    /// Noise characteristics
    pub noise_characteristics: NoiseCharacteristics<T>,
}

/// Noise characteristics
#[derive(Debug, Clone)]
pub struct NoiseCharacteristics<T: Float + Debug + Send + Sync + 'static> {
    /// Noise type
    pub noise_type: NoiseType,

    /// Noise scale
    pub scale: T,

    /// Temporal correlation
    pub temporal_correlation: T,

    /// Spatial correlation
    pub spatial_correlation: T,
}

/// Types of gradient noise
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    /// White noise (uncorrelated)
    White,

    /// Colored noise (correlated)
    Colored,

    /// Structured noise
    Structured,

    /// Adaptive noise
    Adaptive,
}

/// Gradient flow analyzer
#[derive(Debug, Clone)]
pub struct GradientFlowAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Flow field estimation
    pub flow_field: Array2<T>,

    /// Critical points
    pub critical_points: Vec<Array1<T>>,

    /// Flow stability
    pub stability: FlowStability,

    /// Attractors and repellers
    pub attractors: Vec<Array1<T>>,
    pub repellers: Vec<Array1<T>>,
}

/// Flow stability indicators
#[derive(Debug, Clone, Copy)]
pub enum FlowStability {
    /// Stable flow
    Stable,

    /// Unstable flow
    Unstable,

    /// Chaotic flow
    Chaotic,

    /// Unknown stability
    Unknown,
}

/// Loss landscape analyzer
#[derive(Debug, Clone)]
pub struct LossLandscapeAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Local curvature estimation
    pub local_curvature: T,

    /// Hessian eigenvalue estimates
    pub hessian_eigenvalues: Option<Array1<T>>,

    /// Landscape roughness
    pub roughness: T,

    /// Basin of attraction size
    pub basin_size: T,

    /// Barrier heights
    pub barrier_heights: Vec<T>,
}

/// Stability metrics
#[derive(Debug, Clone)]
pub struct StabilityMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Lyapunov exponents
    pub lyapunov_exponents: Array1<T>,

    /// Stability margin
    pub stability_margin: T,

    /// Perturbation sensitivity
    pub perturbation_sensitivity: T,

    /// Robustness score
    pub robustness_score: T,
}

/// Performance metrics for LSTM optimizer
#[derive(Debug, Clone)]
pub struct LSTMOptimizerMetrics {
    /// Meta-learning performance
    pub meta_learning_loss: f64,

    /// Average convergence speed
    pub avg_convergence_speed: f64,

    /// Generalization performance
    pub generalization_performance: f64,

    /// Adaptation efficiency
    pub adaptation_efficiency: f64,

    /// Transfer learning success rate
    pub transfer_success_rate: f64,

    /// Memory usage
    pub memory_usage_mb: f64,

    /// Computational overhead
    pub computational_overhead: f64,

    /// LSTM network statistics
    pub lstm_stats: LSTMNetworkStats,

    /// Attention statistics (if using attention)
    pub attention_stats: Option<AttentionStats>,
}

/// LSTM network statistics
#[derive(Debug, Clone)]
pub struct LSTMNetworkStats {
    /// Gate activation statistics
    pub gate_activations: GateActivationStats,

    /// Hidden state statistics
    pub hidden_state_stats: StateStatistics,

    /// Cell state statistics
    pub cell_state_stats: StateStatistics,

    /// Gradient flow statistics
    pub gradient_flow_stats: GradientFlowStats,
}

/// Gate activation statistics
#[derive(Debug, Clone)]
pub struct GateActivationStats {
    /// Input gate activations
    pub input_gate: StateStatistics,

    /// Forget gate activations
    pub forget_gate: StateStatistics,

    /// Output gate activations
    pub output_gate: StateStatistics,

    /// Cell gate activations
    pub cell_gate: StateStatistics,
}

/// State statistics
#[derive(Debug, Clone)]
pub struct StateStatistics {
    /// Mean activation
    pub mean: f64,

    /// Standard deviation
    pub std: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Saturation percentage
    pub saturation_percent: f64,
}

/// Gradient flow statistics
#[derive(Debug, Clone)]
pub struct GradientFlowStats {
    /// Gradient norm through layers
    pub layer_gradient_norms: Vec<f64>,

    /// Gradient correlation between layers
    pub layer_correlations: Vec<f64>,

    /// Vanishing gradient indicator
    pub vanishing_gradient_score: f64,

    /// Exploding gradient indicator
    pub exploding_gradient_score: f64,
}

/// Attention mechanism statistics
#[derive(Debug, Clone)]
pub struct AttentionStats {
    /// Attention entropy
    pub attention_entropy: f64,

    /// Attention concentration
    pub attention_concentration: f64,

    /// Head diversity
    pub head_diversity: f64,

    /// Temporal attention patterns
    pub temporal_patterns: Vec<f64>,
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + for<'a> std::iter::Sum<&'a T>
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug,
    > LSTMOptimizer<T>
{
    /// Create a new LSTM optimizer
    pub fn new(config: LearnedOptimizerConfig) -> Result<Self> {
        // Validate configuration
        Self::validate_config(&config)?;

        // Initialize LSTM network
        let lstm_network = LSTMNetwork::new(&config)?;

        // Initialize history buffer
        let history_buffer = HistoryBuffer::new(config.gradient_history_size);

        // Initialize meta-learner
        let meta_learner = MetaLearner::new(&config)?;

        // Initialize learning rate controller
        let lr_controller = AdaptiveLearningRateController::new(&config)?;

        // Initialize state tracker
        let state_tracker = OptimizationStateTracker::new();

        // Initialize metrics
        let metrics = LSTMOptimizerMetrics::new();

        Ok(Self {
            config,
            lstm_network,
            history_buffer,
            meta_learner,
            lr_controller,
            state_tracker,
            metrics,
            step_count: 0,
        })
    }

    /// Perform LSTM-based optimization step
    pub fn lstm_step<S, D>(
        &mut self,
        parameters: &ArrayBase<S, D>,
        gradients: &ArrayBase<S, D>,
        loss: Option<T>,
    ) -> Result<Array<T, D>>
    where
        S: Data<Elem = T>,
        D: Dimension + Clone,
    {
        // Convert to flat arrays for processing
        let flat_params = self.flatten_to_1d(parameters)?;
        let flat_gradients = self.flatten_to_1d(gradients)?;

        // The output projection must produce exactly one update per
        // flattened parameter, or the `flat_params - updates` subtraction
        // below panics on mismatched shapes. Re-initialize it at the correct
        // size the moment the optimizer sees a parameter count different from
        // its `output_features` configuration, rather than requiring every
        // caller to pre-size their model to match a fixed constant.
        if self.lstm_network.output_projection.output_size() != flat_params.len() {
            self.lstm_network
                .output_projection
                .reset(self.config.hidden_size, flat_params.len());
        }

        // Update history buffer
        self.history_buffer
            .update(&flat_params, &flat_gradients, loss);

        // Prepare LSTM input features
        let lstm_input = self.prepare_lstm_input(&flat_gradients)?;

        // Forward pass through LSTM
        let lstm_output = self.lstm_network.forward(&lstm_input)?;

        // Compute adaptive learning rate
        let learning_rate =
            self.lr_controller
                .compute_lr(&flat_gradients, loss, &self.history_buffer)?;

        // Generate parameter updates
        let updates = self.generate_updates(&lstm_output, &flat_gradients, learning_rate)?;

        // Apply updates to parameters
        let updated_flat = &flat_params - &updates;

        // Update state tracking
        self.state_tracker.update(&flat_gradients, &updates, loss);

        // Update metrics
        self.update_metrics(&flat_gradients, &updates, learning_rate);

        // Reshape back to original dimensions
        let updated_params = self.reshape_from_1d(&updated_flat, parameters.raw_dim())?;

        self.step_count += 1;

        Ok(updated_params)
    }

    /// Meta-learning step for optimizer adaptation
    pub fn meta_learning_step(&mut self, tasks: &[MetaTask<T>]) -> Result<T> {
        // Perform meta-learning update
        let meta_loss = self.meta_learner.step(tasks, &mut self.lstm_network)?;

        // Update meta-learning metrics
        self.metrics.meta_learning_loss = meta_loss.to_f64().unwrap_or(0.0);

        Ok(meta_loss)
    }

    /// Transfer learning to new optimization domain
    pub fn transfer_to_domain(
        &mut self,
        target_tasks: &[MetaTask<T>],
    ) -> Result<TransferResults<T>> {
        self.meta_learner
            .transfer_learner
            .transfer_to_domain(target_tasks, &mut self.lstm_network)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &LSTMOptimizerMetrics {
        &self.metrics
    }

    /// Read-only access to the LSTM controller network.
    pub fn network(&self) -> &LSTMNetwork<T> {
        &self.lstm_network
    }

    /// Read-only access to the meta-learner, whose `adaptation_history` records
    /// what each [`Self::meta_learning_step`] achieved.
    pub fn meta_learner(&self) -> &MetaLearner<T> {
        &self.meta_learner
    }

    /// Mutable access to the LSTM controller network.
    ///
    /// This is the handle a caller needs to meta-train the controller directly
    /// with [`trainer::MetaTrainer`], rather than going through
    /// [`Self::meta_learning_step`]'s trajectory-surrogate path.
    pub fn network_mut(&mut self) -> &mut LSTMNetwork<T> {
        &mut self.lstm_network
    }

    /// The learning rate the adaptive controller last produced.
    ///
    /// Meaningful only after at least one [`Self::lstm_step`]; before that it is
    /// the configured base rate.
    pub fn current_learning_rate(&self) -> T {
        self.lr_controller.current_lr()
    }

    /// Consecutive steps without a new best loss, as tracked by the adaptive
    /// learning-rate controller.
    pub fn stagnation_counter(&self) -> usize {
        self.lr_controller.stagnation_counter()
    }

    /// Get optimization state analysis
    pub fn get_state_analysis(&self) -> OptimizationStateAnalysis<T> {
        OptimizationStateAnalysis {
            current_phase: self.state_tracker.phase,
            convergence_indicators: self.state_tracker.convergence_indicators.clone(),
            gradient_analysis: self.state_tracker.gradient_analyzer.clone(),
            landscape_analysis: self.state_tracker.landscape_analyzer.clone(),
            stability_metrics: self.state_tracker.stability_metrics.clone(),
        }
    }

    /// Prepare input features for the LSTM.
    ///
    /// Delegates to [`features::build_lstm_features`] so that inference and
    /// truncated-BPTT meta-training consume byte-identical features — training a
    /// different function than the one that ships would make the meta-training
    /// worthless. That shared builder also removed the
    /// `gradients.as_slice().expect(...)` in the previous body, which panicked on
    /// any gradient that was not in contiguous standard layout.
    fn prepare_lstm_input(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let recent = self
            .history_buffer
            .get_recent_gradients(5)
            .unwrap_or_default();
        let loss_features = self.history_buffer.get_loss_features();
        features::build_lstm_features(
            gradients,
            &recent,
            loss_features.as_deref(),
            self.config.input_features,
        )
    }

    /// Generate parameter updates from LSTM output
    fn generate_updates(
        &self,
        lstm_output: &Array1<T>,
        gradients: &Array1<T>,
        learning_rate: T,
    ) -> Result<Array1<T>> {
        // Apply _output transformation
        let transformed_output = match self.lstm_network.output_projection.output_transform {
            OutputTransform::Identity => lstm_output.clone(),
            OutputTransform::Tanh => lstm_output.mapv(|x| x.tanh()),
            OutputTransform::ScaledTanh { scale } => {
                let scale_t =
                    scirs2_core::numeric::NumCast::from(scale).unwrap_or_else(|| T::zero());
                lstm_output.mapv(|x| x.tanh() * scale_t)
            }
            OutputTransform::AdaptiveScale => {
                let grad_norm = gradients.iter().map(|&g| g * g).sum::<T>().sqrt();
                let adaptive_scale = T::one() / (T::one() + grad_norm);
                lstm_output.mapv(|x| x * adaptive_scale)
            }
            OutputTransform::LearnedNonlinear => {
                // Apply learned nonlinear transformation
                lstm_output.mapv(|x| {
                    let exp_x = x.exp();
                    (exp_x - (-x).exp()) / (exp_x + (-x).exp()) // tanh via exp
                })
            }
        };

        // Combine with gradient information
        let updates = &transformed_output * learning_rate;

        Ok(updates)
    }

    /// Update performance metrics.
    ///
    /// `adaptation_efficiency` is the *effective step-size ratio*
    /// `‖Δθ‖ / (lr · ‖g‖)`: the learned optimizer's step measured against the
    /// plain SGD step of the same learning rate that it replaces. `1.0` means
    /// "the same size as SGD would have taken", above `1.0` means the learned
    /// rule is more aggressive than its nominal learning rate, below `1.0` more
    /// conservative. Without dividing by `lr` the figure just tracked the
    /// learning rate itself and was not comparable across schedules.
    fn update_metrics(&mut self, gradients: &Array1<T>, updates: &Array1<T>, lr: T) {
        // Compute gradient statistics
        let grad_norm = gradients.iter().map(|&g| g * g).sum::<T>().sqrt();
        let update_norm = updates.iter().map(|&u| u * u).sum::<T>().sqrt();

        // Update LSTM statistics
        self.update_lstm_stats();

        // Update efficiency metrics. A zero gradient (or a zero learning rate)
        // makes the ratio undefined rather than 0/0-shaped, so report the
        // neutral 1.0 instead of dividing.
        let sgd_step_norm = lr.abs() * grad_norm;
        self.metrics.adaptation_efficiency = if sgd_step_norm > T::zero() {
            (update_norm / sgd_step_norm).to_f64().unwrap_or(1.0)
        } else {
            1.0
        };

        // Update computational overhead
        self.metrics.computational_overhead = self.estimate_computational_overhead();

        // Update memory usage
        self.metrics.memory_usage_mb = self.estimate_memory_usage();
    }

    /// Update LSTM network statistics
    fn update_lstm_stats(&mut self) {
        // Update gate activation statistics
        for layer in self.lstm_network.layers.iter() {
            let hidden_stats = self.compute_state_stats(&layer.hidden_state);
            let cell_stats = self.compute_state_stats(&layer.cell_state);

            // Update statistics (simplified)
            self.metrics.lstm_stats.hidden_state_stats = hidden_stats;
            self.metrics.lstm_stats.cell_state_stats = cell_stats;
        }

        // Update attention statistics if available
        if let Some(ref attention) = self.lstm_network.attention {
            if let Some(ref attentionweights) = attention.attentionweights {
                self.metrics.attention_stats = Some(
                    self.compute_attention_stats(attentionweights, attention.head_logits.as_ref()),
                );
            }
        }
    }

    /// Compute state statistics
    fn compute_state_stats(&self, state: &Array1<T>) -> StateStatistics {
        let values: Vec<f64> = state.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let saturation_count = values.iter().filter(|&&x| x.abs() > 0.95).count();
        let saturation_percent = saturation_count as f64 / values.len() as f64 * 100.0;

        StateStatistics {
            mean,
            std,
            min,
            max,
            saturation_percent,
        }
    }

    /// Compute attention statistics.
    ///
    /// `head_logits`, when present, holds the pre-softmax scaled dot product per
    /// head. Head diversity is the population standard deviation of those
    /// logits: at sequence length 1 the softmax weights are all `1.0`, so any
    /// diversity measure computed from the *weights* is a constant and tells the
    /// caller nothing. The logits are the quantity that actually varies with the
    /// input and between heads.
    ///
    /// `temporal_patterns` is the per-head logit sequence itself, which is the
    /// full temporal record available from one forward pass; an empty vector when
    /// no logits were recorded, rather than a zero-filled placeholder.
    fn compute_attention_stats(
        &self,
        attentionweights: &Array2<T>,
        head_logits: Option<&Array1<T>>,
    ) -> AttentionStats {
        let weights: Vec<f64> = attentionweights
            .iter()
            .map(|&w| w.to_f64().unwrap_or(0.0))
            .collect();

        // Compute entropy
        let entropy = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum::<f64>();

        // Compute concentration (inverse of entropy)
        let concentration = 1.0 / (1.0 + entropy);

        let logits: Vec<f64> = head_logits
            .map(|l| l.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect())
            .unwrap_or_default();
        let head_diversity = if logits.is_empty() {
            0.0
        } else {
            let mean = logits.iter().sum::<f64>() / logits.len() as f64;
            (logits.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / logits.len() as f64)
                .sqrt()
        };

        AttentionStats {
            attention_entropy: entropy,
            attention_concentration: concentration,
            head_diversity,
            temporal_patterns: logits,
        }
    }

    /// Estimate computational overhead
    fn estimate_computational_overhead(&self) -> f64 {
        // Simplified overhead estimation
        let lstm_overhead = self.config.num_layers as f64 * 0.1;
        let attention_overhead = if self.config.use_attention { 0.2 } else { 0.0 };
        let meta_learning_overhead = 0.1;

        1.0 + lstm_overhead + attention_overhead + meta_learning_overhead
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> f64 {
        // Simplified memory estimation in MB
        let parameter_memory =
            self.config.hidden_size as f64 * self.config.num_layers as f64 * 8.0 / 1024.0 / 1024.0;
        let history_memory =
            self.config.gradient_history_size as f64 * self.config.input_features as f64 * 8.0
                / 1024.0
                / 1024.0;
        let lstm_state_memory =
            self.config.hidden_size as f64 * self.config.num_layers as f64 * 2.0 * 8.0
                / 1024.0
                / 1024.0;

        parameter_memory + history_memory + lstm_state_memory
    }

    /// Validate configuration
    fn validate_config(config: &LearnedOptimizerConfig) -> Result<()> {
        if config.hidden_size == 0 {
            return Err(OptimError::InvalidConfig(
                "Hidden size must be positive".to_string(),
            ));
        }

        if config.num_layers == 0 {
            return Err(OptimError::InvalidConfig(
                "Number of layers must be positive".to_string(),
            ));
        }

        if config.input_features == 0 {
            return Err(OptimError::InvalidConfig(
                "Input features must be positive".to_string(),
            ));
        }

        if config.meta_learning_rate <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "Meta learning rate must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Utility functions for array manipulation
    fn flatten_to_1d<S, D>(&self, array: &ArrayBase<S, D>) -> Result<Array1<T>>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        Ok(Array1::from_iter(array.iter().cloned()))
    }

    fn reshape_from_1d<D>(&self, flat: &Array1<T>, shape: D) -> Result<Array<T, D>>
    where
        D: Dimension + Clone,
    {
        Array::from_shape_vec(shape, flat.to_vec())
            .map_err(|e| OptimError::InvalidConfig(format!("Reshape error: {}", e)))
    }
}

// Implementation of major components

impl<T: Float + Debug + Default + Clone + 'static + Send + Sync> LSTMNetwork<T> {
    /// Create a new LSTM controller network from a configuration, with weights
    /// drawn from thread-local entropy.
    ///
    /// Public because meta-training operates on a controller directly: see
    /// [`trainer::MetaTrainer`] and [`LSTMOptimizer::network_mut`]. For
    /// reproducible construction (tests, checkpoint-comparable experiments) use
    /// [`Self::new_seeded`].
    pub fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        Self::new_seeded(config, thread_rng().random::<u64>())
    }

    /// Create a controller whose initial weights are a deterministic function of
    /// `seed` (and the configuration).
    ///
    /// Everything downstream of construction — rollout capture, truncated BPTT,
    /// meta-training — is already deterministic with dropout off, so seeding the
    /// initialization makes an entire meta-training run reproducible.
    pub fn new_seeded(config: &LearnedOptimizerConfig, seed: u64) -> Result<Self> {
        let mut rng = seeded_rng(seed);
        let mut layers = Vec::new();

        // Create LSTM layers
        for i in 0..config.num_layers {
            let input_size = if i == 0 {
                config.input_features
            } else {
                config.hidden_size
            };
            let layer = LSTMLayer::new(input_size, config.hidden_size, &mut rng)?;
            layers.push(layer);
        }

        // Create output projection
        let output_projection = OutputProjection::new_with_rng(
            config.hidden_size,
            config.output_features,
            OutputTransform::ScaledTanh { scale: 0.1 },
            &mut rng,
        )?;

        // Create attention mechanism if enabled
        let attention = if config.use_attention {
            Some(AttentionMechanism::new_with_rng(config, &mut rng)?)
        } else {
            None
        };

        // Create layer normalization
        let layer_norms = (0..config.num_layers)
            .map(|_| LayerNormalization::new(config.hidden_size))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            layers,
            output_projection,
            attention,
            layer_norms,
            dropout_rate: config.dropout_rate,
            training: false,
        })
    }

    /// Enable or disable dropout.
    ///
    /// Meta-training and inference both run with dropout **off**; turn it on only
    /// for a training regime that wants the regularization and can tolerate the
    /// nondeterminism.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Whether dropout is currently active.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Zero every layer's hidden and cell state.
    ///
    /// Required before a meta-training rollout: the recurrent state is what makes
    /// two rollouts of the same controller differ, so a reproducible rollout must
    /// start from a known state.
    pub fn reset_state(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.hidden_state.fill(T::zero());
            layer.cell_state.fill(T::zero());
        }
    }

    /// Forward pass through the LSTM controller network.
    ///
    /// Public because the truncated-BPTT engine's taped forward
    /// ([`bptt`]) must be checkable against it — see
    /// `taped_forward_matches_the_deployed_forward` in
    /// `tests/lstm_meta_training.rs`. If the two ever drift apart,
    /// meta-training would optimize a different function than the one that runs
    /// at inference time, and the finite-difference check alone would not notice
    /// (it is self-consistent by construction).
    pub fn forward(&mut self, input: &Array1<T>) -> Result<Array1<T>> {
        let mut current_input = input.clone();

        // Forward through LSTM layers
        for i in 0..self.layers.len() {
            current_input = self.layers[i].forward(&current_input)?;

            // Apply layer normalization
            current_input = self.layer_norms[i].forward(&current_input)?;

            // Apply dropout during training only.
            if self.training && self.dropout_rate > 0.0 {
                current_input = self.apply_dropout(&current_input)?;
            }
        }

        // Apply attention if enabled
        if let Some(ref mut attention) = self.attention {
            current_input = attention.forward(&current_input)?;
        }

        // Final output projection
        let output = self.output_projection.forward(&current_input)?;

        Ok(output)
    }

    /// Apply inverted dropout for regularization.
    ///
    /// Each element is zeroed with probability `dropout_rate` and the survivors
    /// are divided by `1 - dropout_rate`, so the expected activation is
    /// unchanged. A rate of `0` is a no-op; a rate of `1` would divide by zero
    /// and is rejected.
    ///
    /// # Errors
    /// Returns `Err` when `dropout_rate` is not in `[0, 1)` or when the drawn
    /// uniform sample cannot be represented in `T`.
    fn apply_dropout(&self, input: &Array1<T>) -> Result<Array1<T>> {
        if !(0.0..1.0).contains(&self.dropout_rate) {
            return Err(OptimError::InvalidConfig(format!(
                "dropout_rate must be in [0, 1), got {}",
                self.dropout_rate
            )));
        }
        if self.dropout_rate == 0.0 {
            return Ok(input.clone());
        }
        let threshold: T = cast_scalar(self.dropout_rate)?;
        let keep_scale: T = cast_scalar(1.0 - self.dropout_rate)?;
        let mut rng = scirs2_core::random::thread_rng();
        let mut out = input.clone();
        for value in out.iter_mut() {
            let sample: T = cast_scalar(rng.gen_range(0.0..1.0))?;
            *value = if sample < threshold {
                T::zero()
            } else {
                *value / keep_scale
            };
        }
        Ok(out)
    }
}

impl<T: Float + Debug + Default + Clone + 'static + Send + Sync> LSTMLayer<T> {
    /// Create new LSTM layer
    fn new(_input_size: usize, hiddensize: usize, rng: &mut CoreRandom<StdRng>) -> Result<Self> {
        // Xavier/Glorot uniform, with the fan pair taken *per matrix*: the four
        // gates are independent maps stacked along the rows, so the fan-out of
        // each is `hiddensize`, not `4 * hiddensize`. The input-to-hidden and
        // hidden-to-hidden matrices therefore have different fan-ins and must not
        // share one limit (they did, computed from `_input_size + hiddensize` for
        // both).
        let limit_ih = Self::xavier_limit(_input_size, hiddensize);
        let limit_hh = Self::xavier_limit(hiddensize, hiddensize);

        Ok(Self {
            weight_ih: Self::xavier_init(rng, 4 * hiddensize, _input_size, limit_ih),
            weight_hh: Self::xavier_init(rng, 4 * hiddensize, hiddensize, limit_hh),
            bias_ih: Array1::zeros(4 * hiddensize),
            bias_hh: Array1::zeros(4 * hiddensize),
            hidden_state: Array1::zeros(hiddensize),
            cell_state: Array1::zeros(hiddensize),
            hiddensize,
        })
    }

    /// Forward pass through LSTM layer
    fn forward(&mut self, input: &Array1<T>) -> Result<Array1<T>> {
        // LSTM computation: i, f, g, o = σ(W_ih @ x + W_hh @ h + b)
        let ih_linear = self.weight_ih.dot(input) + &self.bias_ih;
        let hh_linear = self.weight_hh.dot(&self.hidden_state) + &self.bias_hh;
        let gates = ih_linear + hh_linear;

        // Split into gates
        let input_gate = Self::sigmoid(&gates.slice(s![0..self.hiddensize]).to_owned());
        let forget_gate = Self::sigmoid(
            &gates
                .slice(s![self.hiddensize..2 * self.hiddensize])
                .to_owned(),
        );
        let cell_gate = Self::tanh(
            &gates
                .slice(s![2 * self.hiddensize..3 * self.hiddensize])
                .to_owned(),
        );
        let output_gate = Self::sigmoid(
            &gates
                .slice(s![3 * self.hiddensize..4 * self.hiddensize])
                .to_owned(),
        );

        // Update cell state
        self.cell_state = &forget_gate * &self.cell_state + &input_gate * &cell_gate;

        // Update hidden state
        self.hidden_state = &output_gate * &Self::tanh(&self.cell_state);

        Ok(self.hidden_state.clone())
    }

    /// Xavier/Glorot **uniform** limit for a layer with the given fans:
    /// `sqrt(6 / (fan_in + fan_out))`.
    ///
    /// `Uniform[-b, b]` has variance `b^2 / 3`, so reaching Glorot's target
    /// variance `2 / (fan_in + fan_out)` needs `b = sqrt(6 / (fan_in + fan_out))`.
    /// Every call site in this file previously passed `sqrt(2 / (fan_in +
    /// fan_out))` — the limit for a *normal* draw — to the uniform sampler
    /// below, so every LSTM weight matrix started with one third of the intended
    /// variance and correspondingly attenuated signal and gradient (finding F64).
    pub(crate) fn xavier_limit(fan_in: usize, fan_out: usize) -> f64 {
        (6.0 / (fan_in + fan_out).max(1) as f64).sqrt()
    }

    /// Draw a matrix from `Uniform[-limit, limit]` using the caller's generator.
    ///
    /// `limit` is the *uniform half-width*, not a standard deviation — use
    /// [`Self::xavier_limit`] to compute it from the layer's fans. Threading the
    /// generator through the constructors (rather than grabbing `thread_rng`
    /// here) is what makes [`LSTMNetwork::new_seeded`] reproducible.
    fn xavier_init(
        rng: &mut CoreRandom<StdRng>,
        rows: usize,
        cols: usize,
        limit: f64,
    ) -> Array2<T> {
        Array2::from_shape_fn((rows, cols), |_| {
            let val = (rng.gen_range(0.0..1.0) - 0.5) * 2.0 * limit;
            scirs2_core::numeric::NumCast::from(val).unwrap_or_else(|| T::zero())
        })
    }

    /// Sigmoid activation
    fn sigmoid(x: &Array1<T>) -> Array1<T> {
        x.mapv(|xi| T::one() / (T::one() + (-xi).exp()))
    }

    /// Tanh activation
    fn tanh(x: &Array1<T>) -> Array1<T> {
        x.mapv(|xi| xi.tanh())
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> HistoryBuffer<T> {
    /// Create new history buffer
    fn new(_maxlength: usize) -> Self {
        Self {
            gradients: VecDeque::with_capacity(_maxlength),
            parameters: VecDeque::with_capacity(_maxlength),
            losses: VecDeque::with_capacity(_maxlength),
            _maxlength,
            feature_cache: None,
        }
    }

    /// Update history with new data
    fn update(&mut self, params: &Array1<T>, grads: &Array1<T>, loss: Option<T>) {
        // Add new entries
        self.parameters.push_back(params.clone());
        self.gradients.push_back(grads.clone());

        if let Some(l) = loss {
            self.losses.push_back(l);
        }

        // Maintain size limits
        while self.parameters.len() > self._maxlength {
            self.parameters.pop_front();
        }
        while self.gradients.len() > self._maxlength {
            self.gradients.pop_front();
        }
        while self.losses.len() > self._maxlength {
            self.losses.pop_front();
        }

        // Invalidate cache
        self.feature_cache = None;
    }

    /// Get recent gradients
    fn get_recent_gradients(&self, count: usize) -> Option<Vec<&Array1<T>>> {
        if self.gradients.len() < count {
            return None;
        }

        Some(self.gradients.iter().rev().take(count).collect())
    }

    /// Get loss-based features
    fn get_loss_features(&self) -> Option<Vec<T>> {
        if self.losses.len() < 2 {
            return None;
        }

        // `len() >= 2` was checked above, so both reads are in range; using the
        // fallible accessor keeps that fact local instead of relying on an
        // `expect` five lines away from its guard.
        let current_loss = *self.losses.back()?;
        let prev_loss = *self.losses.get(self.losses.len() - 2)?;

        let loss_change = current_loss - prev_loss;
        let loss_ratio = if prev_loss.abs()
            > scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero())
        {
            current_loss / prev_loss
        } else {
            T::one()
        };

        Some(vec![loss_change, loss_ratio])
    }
}

/// Additional implementations for other components...
/// Results from optimization state analysis
#[derive(Debug, Clone)]
pub struct OptimizationStateAnalysis<T: Float + Debug + Send + Sync + 'static> {
    pub current_phase: OptimizationPhase,
    pub convergence_indicators: ConvergenceIndicators<T>,
    pub gradient_analysis: GradientAnalyzer<T>,
    pub landscape_analysis: LossLandscapeAnalyzer<T>,
    pub stability_metrics: StabilityMetrics<T>,
}

/// Transfer learning results
#[derive(Debug, Clone)]
pub struct TransferResults<T: Float + Debug + Send + Sync + 'static> {
    pub initial_performance: T,
    pub final_performance: T,
    pub adaptation_steps: usize,
    pub transfer_efficiency: T,
}

// Additional default implementations and stubs for remaining components...

impl Default for LSTMOptimizerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl LSTMOptimizerMetrics {
    fn new() -> Self {
        Self {
            meta_learning_loss: 0.0,
            avg_convergence_speed: 0.0,
            generalization_performance: 0.0,
            adaptation_efficiency: 0.0,
            transfer_success_rate: 0.0,
            memory_usage_mb: 0.0,
            computational_overhead: 1.0,
            lstm_stats: LSTMNetworkStats {
                gate_activations: GateActivationStats {
                    input_gate: StateStatistics::default(),
                    forget_gate: StateStatistics::default(),
                    output_gate: StateStatistics::default(),
                    cell_gate: StateStatistics::default(),
                },
                hidden_state_stats: StateStatistics::default(),
                cell_state_stats: StateStatistics::default(),
                gradient_flow_stats: GradientFlowStats {
                    layer_gradient_norms: Vec::new(),
                    layer_correlations: Vec::new(),
                    vanishing_gradient_score: 0.0,
                    exploding_gradient_score: 0.0,
                },
            },
            attention_stats: None,
        }
    }
}

impl Default for StateStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            saturation_percent: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_optimizer_creation() {
        let config = LearnedOptimizerConfig::default();
        let optimizer = LSTMOptimizer::<f64>::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_lstm_layer_creation() {
        let layer = LSTMLayer::<f64>::new(10, 20, &mut seeded_rng(7));
        assert!(layer.is_ok());

        let layer = layer.expect("LSTMLayer::new should succeed");
        assert_eq!(layer.hiddensize, 20);
        assert_eq!(layer.weight_ih.shape(), &[80, 10]); // 4 * hiddensize, input_size
        assert_eq!(layer.weight_hh.shape(), &[80, 20]); // 4 * hiddensize, hiddensize
    }

    #[test]
    fn test_history_buffer() {
        let mut buffer = HistoryBuffer::<f64>::new(5);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        buffer.update(&params, &grads, Some(0.5));

        assert_eq!(buffer.gradients.len(), 1);
        assert_eq!(buffer.parameters.len(), 1);
        assert_eq!(buffer.losses.len(), 1);
    }

    #[test]
    fn test_config_validation() {
        let mut config = LearnedOptimizerConfig::default();
        assert!(LSTMOptimizer::<f64>::validate_config(&config).is_ok());

        config.hidden_size = 0;
        assert!(LSTMOptimizer::<f64>::validate_config(&config).is_err());
    }

    #[test]
    fn test_lstm_network_creation() {
        let config = LearnedOptimizerConfig::default();
        let network = LSTMNetwork::<f64>::new(&config);
        assert!(network.is_ok());

        let network = network.expect("LSTMNetwork::new should succeed");
        assert_eq!(network.layers.len(), config.num_layers);
        assert!(network.attention.is_some()); // attention enabled by default
    }

    #[test]
    fn test_metrics_initialization() {
        let metrics = LSTMOptimizerMetrics::new();
        assert_eq!(metrics.meta_learning_loss, 0.0);
        assert_eq!(metrics.computational_overhead, 1.0);
        assert!(metrics.attention_stats.is_none());
    }
}
