// Differentiable Architecture Search (DARTS) and variants
//
// Implements the original DARTS algorithm along with enhanced variants:
// - MemoryEfficientDARTS: partial channel connections, edge normalization
// - RobustDARTS: perturbation regularization for stability
// - DARTSConfig: flexible configuration builder

use scirs2_core::ndarray::{s, Array1, Array3};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::architecture::{ComponentPosition, ComponentType};
#[allow(unused_imports)]
use crate::error::Result;
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::{SearchStrategy, SearchStrategyStatistics};

/// Differentiable Architecture Search (DARTS)
pub struct DifferentiableSearch<T: Float + Debug + Send + Sync + 'static> {
    architecture_weights: Array3<T>,
    weight_optimizer: WeightOptimizer<T>,
    temperature: T,
    gumbel_softmax: bool,
    continuous_relaxation: bool,
    statistics: SearchStrategyStatistics<T>,
    discretization_strategy: DiscretizationStrategy,
    /// Running (EMA) reward baseline for the REINFORCE advantage.
    ///
    /// Read *before* it is updated on every step so that even a single-sample
    /// batch yields a non-zero advantage (a batch-mean baseline would cancel to
    /// exactly zero for a one-element batch, silently disabling learning).
    reward_baseline: T,
}

/// SGD-with-momentum optimizer for the DARTS architecture logits.
///
/// Applies momentum-smoothed gradient ascent together with decoupled weight
/// decay, mutating the architecture-weight tensor in place. Both `momentum` and
/// `weight_decay` are genuinely applied by [`WeightOptimizer::step`].
#[derive(Debug)]
pub struct WeightOptimizer<T: Float + Debug + Send + Sync + 'static> {
    learning_rate: T,
    momentum: T,
    weight_decay: T,
    velocity: Array3<T>,
}

/// Discretization strategies for DARTS
#[derive(Debug, Clone, Copy)]
pub enum DiscretizationStrategy {
    /// Select operation with highest weight
    Greedy,
    /// Sample proportional to weights
    Sampling,
    /// Progressive discretization
    Progressive,
    /// Threshold-based discretization
    Threshold,
}

/// Temperature schedule for architecture weight annealing
#[derive(Debug, Clone, Copy)]
pub enum TemperatureSchedule {
    /// Constant temperature
    Constant,
    /// Linear decay from initial to final temperature
    Linear { initial: f64, final_temp: f64 },
    /// Exponential decay
    Exponential { initial: f64, decay_rate: f64 },
    /// Cosine annealing
    Cosine { initial: f64, final_temp: f64 },
}

/// Memory-Efficient DARTS with partial channel connections and edge normalization
///
/// Based on "PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search"
/// Reduces memory consumption by only operating on a subset of channels while maintaining
/// search quality through edge normalization.
pub struct MemoryEfficientDARTS<T: Float + Debug + Send + Sync + 'static> {
    /// Architecture weights (edges x operations x 1)
    architecture_weights: Array3<T>,
    /// Weight optimizer for architecture parameters
    weight_optimizer: WeightOptimizer<T>,
    /// Current temperature for softmax
    temperature: T,
    /// Whether to use Gumbel-Softmax sampling
    gumbel_softmax: bool,
    /// Fraction of channels to use in partial connections (0.0, 1.0]
    partial_channel_ratio: T,
    /// Whether to apply edge normalization
    edge_normalization: bool,
    /// Edge normalization weights (one per edge)
    edge_weights: Array1<T>,
    /// Discretization strategy
    discretization_strategy: DiscretizationStrategy,
    /// Search statistics
    statistics: SearchStrategyStatistics<T>,
    /// Channel sampling seed for reproducibility
    channel_seed: u64,
    /// Running (EMA) reward baseline for the REINFORCE advantage.
    reward_baseline: T,
}

/// Robust DARTS with perturbation-based regularization
///
/// Based on "Understanding and Robustifying Differentiable Architecture Search"
/// Addresses the performance collapse problem in DARTS through perturbation-based
/// regularization and early stopping based on eigenvalue analysis.
pub struct RobustDARTS<T: Float + Debug + Send + Sync + 'static> {
    /// Architecture weights (edges x operations x 1)
    architecture_weights: Array3<T>,
    /// Weight optimizer for architecture parameters
    weight_optimizer: WeightOptimizer<T>,
    /// Current temperature for softmax
    temperature: T,
    /// Whether to use Gumbel-Softmax sampling
    gumbel_softmax: bool,
    /// Perturbation strength for regularization
    perturbation_strength: T,
    /// Early stopping patience (number of steps without improvement)
    early_stopping_patience: usize,
    /// Regularization weight for perturbation loss
    regularization_weight: T,
    /// Discretization strategy
    discretization_strategy: DiscretizationStrategy,
    /// Search statistics
    statistics: SearchStrategyStatistics<T>,
    /// History of dominant eigenvalues for early stopping
    eigenvalue_history: Vec<T>,
    /// Steps since last improvement
    steps_without_improvement: usize,
    /// Best validation performance seen so far
    best_validation_performance: T,
    /// Whether search has been early-stopped
    early_stopped: bool,
    /// Running (EMA) reward baseline for the REINFORCE advantage.
    reward_baseline: T,
}

/// Configuration builder for DARTS variants
#[derive(Debug, Clone)]
pub struct DARTSConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Number of operations in the search space
    pub num_operations: usize,
    /// Number of edges in the DAG
    pub num_edges: usize,
    /// Initial temperature for softmax
    pub initial_temperature: f64,
    /// Whether to use Gumbel-Softmax
    pub use_gumbel: bool,
    /// Temperature schedule
    pub temperature_schedule: TemperatureSchedule,
    /// Discretization strategy
    pub discretization_strategy: DiscretizationStrategy,
    /// Learning rate for architecture weights
    pub architecture_lr: f64,
    /// Weight decay for architecture optimization
    pub weight_decay: f64,
    /// Momentum for optimizer
    pub momentum: f64,
    _marker: std::marker::PhantomData<T>,
}

// ─── DifferentiableSearch impl ───────────────────────────────────────────────

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + scirs2_core::ndarray::ScalarOperand
            + std::iter::Sum,
    > DifferentiableSearch<T>
{
    pub fn new(
        num_operations: usize,
        num_edges: usize,
        temperature: f64,
        use_gumbel: bool,
    ) -> Self {
        Self {
            architecture_weights: Array3::zeros((num_edges, num_operations, 1)),
            weight_optimizer: WeightOptimizer::new(
                scirs2_core::numeric::NumCast::from(0.025).unwrap_or_else(|| T::zero()),
            ),
            temperature: scirs2_core::numeric::NumCast::from(temperature)
                .unwrap_or_else(|| T::zero()),
            gumbel_softmax: use_gumbel,
            continuous_relaxation: true,
            statistics: SearchStrategyStatistics::default(),
            discretization_strategy: DiscretizationStrategy::Progressive,
            reward_baseline: T::zero(),
        }
    }

    fn gumbel_softmax_sample(&self, logits: &Array1<T>) -> Array1<T> {
        if !self.gumbel_softmax {
            return softmax(logits);
        }

        let gumbel_noise: Array1<T> = Array1::from_shape_fn(logits.len(), |_| {
            let u = scirs2_core::random::Random::default().random::<f64>();
            T::from(gumbel_noise_from_uniform(u)).unwrap_or_else(|| T::zero())
        });

        let gumbel_logits = logits + &gumbel_noise;
        let scaled_logits = gumbel_logits / self.temperature;
        softmax(&scaled_logits)
    }
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > SearchStrategy<T> for DifferentiableSearch<T>
{
    fn initialize(&mut self, _searchspace: &SearchSpaceConfig) -> Result<()> {
        // Initialize architecture weights with small random values
        self.architecture_weights =
            Array3::from_shape_fn(self.architecture_weights.raw_dim(), |_| {
                T::from(scirs2_core::random::Random::default().random::<f64>() * 0.1 - 0.05)
                    .unwrap_or_else(|| T::zero())
            });
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        _search_space: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        if self.continuous_relaxation {
            // Generate continuous relaxation of architecture
            let mut sampled_weights = Array3::zeros(self.architecture_weights.raw_dim());

            for edge_idx in 0..self.architecture_weights.dim().0 {
                let edge_weights = self.architecture_weights.slice(s![edge_idx, .., 0]);
                let sampled = self.gumbel_softmax_sample(&edge_weights.to_owned());

                for (op_idx, &weight) in sampled.iter().enumerate() {
                    sampled_weights[[edge_idx, op_idx, 0]] = weight;
                }
            }

            self.statistics.total_architectures_generated += 1;
            Ok(discretize_architecture(
                &sampled_weights,
                &self.discretization_strategy,
            ))
        } else {
            // Direct discretization
            self.statistics.total_architectures_generated += 1;
            Ok(discretize_architecture(
                &self.architecture_weights,
                &self.discretization_strategy,
            ))
        }
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        // Keep every score paired with the architecture that produced it so the
        // policy gradient can attribute reward to the operations actually chosen.
        let scored = collect_scored(results);

        if scored.is_empty() {
            return Ok(());
        }

        let performances: Vec<T> = scored.iter().map(|(_, p)| *p).collect();

        // Update statistics.
        self.statistics.best_performance = performances
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::zero());

        let sum: T = performances.iter().cloned().sum();
        let count = T::from(performances.len()).unwrap_or_else(|| T::one());
        let batch_mean = sum / count;
        self.statistics.average_performance = batch_mean;

        // REINFORCE advantage against a running baseline read BEFORE it is
        // updated, so a single-sample batch still produces a non-zero signal.
        let baseline = self.reward_baseline;
        let advantages: Vec<T> = performances.iter().map(|&p| p - baseline).collect();

        // Real per-entry policy gradient (non-uniform across operations) applied
        // through the momentum/weight-decay optimizer.
        let grad = reinforce_logit_gradient(&self.architecture_weights, &scored, &advantages);
        self.weight_optimizer
            .step(&mut self.architecture_weights, &grad);

        // Move the baseline toward the observed batch mean.
        let beta: T = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());
        self.reward_baseline = self.reward_baseline * (T::one() - beta) + batch_mean * beta;

        // Anneal temperature
        self.temperature = self.temperature
            * scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero());

        Ok(())
    }

    fn name(&self) -> &str {
        "DifferentiableSearch"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate = self.temperature;
        stats.exploitation_rate = T::one() - self.temperature;
        stats
    }
}

// ─── MemoryEfficientDARTS impl ──────────────────────────────────────────────

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + scirs2_core::ndarray::ScalarOperand
            + std::iter::Sum,
    > MemoryEfficientDARTS<T>
{
    /// Create a new MemoryEfficientDARTS instance
    ///
    /// # Arguments
    /// * `num_operations` - Number of candidate operations per edge
    /// * `num_edges` - Number of edges in the search DAG
    /// * `temperature` - Initial softmax temperature
    /// * `use_gumbel` - Whether to use Gumbel-Softmax relaxation
    /// * `partial_channel_ratio` - Fraction of channels to use (0.0, 1.0]
    /// * `edge_normalization` - Whether to apply edge normalization
    pub fn new(
        num_operations: usize,
        num_edges: usize,
        temperature: f64,
        use_gumbel: bool,
        partial_channel_ratio: f64,
        edge_normalization: bool,
    ) -> Self {
        let ratio = partial_channel_ratio.clamp(0.01, 1.0);
        Self {
            architecture_weights: Array3::zeros((num_edges, num_operations, 1)),
            weight_optimizer: WeightOptimizer::new(
                scirs2_core::numeric::NumCast::from(0.025).unwrap_or_else(|| T::zero()),
            ),
            temperature: scirs2_core::numeric::NumCast::from(temperature)
                .unwrap_or_else(|| T::zero()),
            gumbel_softmax: use_gumbel,
            partial_channel_ratio: scirs2_core::numeric::NumCast::from(ratio)
                .unwrap_or_else(|| T::one()),
            edge_normalization,
            edge_weights: Array1::ones(num_edges),
            discretization_strategy: DiscretizationStrategy::Progressive,
            statistics: SearchStrategyStatistics::default(),
            channel_seed: 42,
            reward_baseline: T::zero(),
        }
    }

    /// Indicator tensor (`1` kept, `0` dropped) of the partial-channel selection.
    ///
    /// Exposed separately from [`Self::apply_partial_channels`] because the
    /// policy gradient has to chain through this mask: a dropped operation does
    /// not appear in the sampling logits, so its architecture weight receives no
    /// credit at all.
    fn partial_channel_mask(&self, weights: &Array3<T>) -> Array3<T> {
        let num_edges = weights.dim().0;
        let num_ops = weights.dim().1;
        let mut mask = Array3::zeros(weights.raw_dim());

        // For each edge, only activate a fraction of operations based on channel ratio
        let active_ops =
            ((num_ops as f64) * self.partial_channel_ratio.to_f64().unwrap_or(1.0)).ceil() as usize;
        let active_ops = active_ops.max(1).min(num_ops);

        for edge_idx in 0..num_edges {
            // Select top-k operations by weight magnitude (deterministic channel selection)
            let edge_slice = weights.slice(s![edge_idx, .., 0]);
            let mut indexed: Vec<(usize, T)> = edge_slice.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| {
                b.abs()
                    .partial_cmp(&a.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (rank, (op_idx, _)) in indexed.iter().enumerate() {
                if rank < active_ops {
                    mask[[edge_idx, *op_idx, 0]] = T::one();
                }
            }
        }

        mask
    }

    /// Apply partial channel selection to architecture weights
    fn apply_partial_channels(&self, weights: &Array3<T>) -> Array3<T> {
        let mask = self.partial_channel_mask(weights);
        let mut masked = Array3::zeros(weights.raw_dim());
        let (num_edges, num_ops, depth) = weights.dim();
        for e in 0..num_edges {
            for o in 0..num_ops {
                for d in 0..depth {
                    masked[[e, o, d]] = mask[[e, o, d]] * weights[[e, o, d]];
                }
            }
        }
        masked
    }

    /// Apply edge normalization to the architecture weights
    fn apply_edge_normalization(&self, weights: &Array3<T>) -> Array3<T> {
        if !self.edge_normalization {
            return weights.clone();
        }

        let num_edges = weights.dim().0;
        let mut normalized = weights.clone();

        // Normalize edge weights using softmax
        let edge_probs = softmax(&self.edge_weights);

        for edge_idx in 0..num_edges {
            let edge_weight = edge_probs[edge_idx];
            for op_idx in 0..weights.dim().1 {
                normalized[[edge_idx, op_idx, 0]] = normalized[[edge_idx, op_idx, 0]] * edge_weight;
            }
        }

        normalized
    }

    fn gumbel_softmax_sample(&self, logits: &Array1<T>) -> Array1<T> {
        if !self.gumbel_softmax {
            return softmax(logits);
        }

        let gumbel_noise: Array1<T> = Array1::from_shape_fn(logits.len(), |_| {
            let u = scirs2_core::random::Random::default().random::<f64>();
            T::from(gumbel_noise_from_uniform(u)).unwrap_or_else(|| T::zero())
        });

        let gumbel_logits = logits + &gumbel_noise;
        let scaled_logits = gumbel_logits / self.temperature;
        softmax(&scaled_logits)
    }
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > SearchStrategy<T> for MemoryEfficientDARTS<T>
{
    fn initialize(&mut self, _searchspace: &SearchSpaceConfig) -> Result<()> {
        self.architecture_weights =
            Array3::from_shape_fn(self.architecture_weights.raw_dim(), |_| {
                T::from(scirs2_core::random::Random::default().random::<f64>() * 0.1 - 0.05)
                    .unwrap_or_else(|| T::zero())
            });
        // Initialize edge weights uniformly
        self.edge_weights = Array1::ones(self.architecture_weights.dim().0);
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        _search_space: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        // Apply partial channel selection
        let partial_weights = self.apply_partial_channels(&self.architecture_weights);

        // Apply edge normalization
        let normalized_weights = self.apply_edge_normalization(&partial_weights);

        // Sample using Gumbel-Softmax
        let mut sampled_weights = Array3::zeros(normalized_weights.raw_dim());
        for edge_idx in 0..normalized_weights.dim().0 {
            let edge_weights = normalized_weights.slice(s![edge_idx, .., 0]);
            let sampled = self.gumbel_softmax_sample(&edge_weights.to_owned());
            for (op_idx, &weight) in sampled.iter().enumerate() {
                sampled_weights[[edge_idx, op_idx, 0]] = weight;
            }
        }

        self.statistics.total_architectures_generated += 1;
        Ok(discretize_architecture(
            &sampled_weights,
            &self.discretization_strategy,
        ))
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        let scored = collect_scored(results);
        if scored.is_empty() {
            return Ok(());
        }

        let performances: Vec<T> = scored.iter().map(|(_, p)| *p).collect();

        self.statistics.best_performance = performances
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::zero());

        let sum: T = performances.iter().cloned().sum();
        let count = T::from(performances.len()).unwrap_or_else(|| T::one());
        let batch_mean = sum / count;
        self.statistics.average_performance = batch_mean;

        // Advantage against the running baseline read *before* it is updated, so
        // a single-sample batch still carries signal (a batch-mean baseline
        // cancels to exactly zero there).
        let baseline = self.reward_baseline;
        let advantages: Vec<T> = performances.iter().map(|&p| p - baseline).collect();

        // The sampling logits are `beta_e * mask_[e,o] * w_[e,o]`, so the score
        // function w.r.t. the raw architecture weights chains through both the
        // partial-channel mask and the edge-normalization factor.
        let mask = self.partial_channel_mask(&self.architecture_weights);
        let partial = self.apply_partial_channels(&self.architecture_weights);
        let effective = self.apply_edge_normalization(&partial);
        let grad_effective = reinforce_logit_gradient(&effective, &scored, &advantages);

        let edge_probs = softmax(&self.edge_weights);
        let (num_edges, num_ops, _depth) = self.architecture_weights.dim();
        let mut grad_weights = Array3::zeros(self.architecture_weights.raw_dim());
        for e in 0..num_edges {
            // d effective / d w = beta_e * mask (beta_e == 1 when edge
            // normalization is disabled, i.e. `apply_edge_normalization` is the
            // identity).
            let edge_scale = if self.edge_normalization {
                edge_probs.get(e).copied().unwrap_or_else(T::one)
            } else {
                T::one()
            };
            for o in 0..num_ops {
                grad_weights[[e, o, 0]] = grad_effective[[e, o, 0]] * edge_scale * mask[[e, o, 0]];
            }
        }
        self.weight_optimizer
            .step(&mut self.architecture_weights, &grad_weights);

        // Edge-normalization weights participate in the same policy, so they get
        // the same score function chained through the softmax that produces
        // `edge_probs`: dJ/draw_k = beta_k * (s_k - sum_e s_e * beta_e), with
        // s_e = sum_o dJ/deffective[e, o] * partial[e, o].
        if self.edge_normalization && num_edges > 0 {
            let edge_scores: Vec<T> = (0..num_edges)
                .map(|e| {
                    (0..num_ops)
                        .map(|o| grad_effective[[e, o, 0]] * partial[[e, o, 0]])
                        .fold(T::zero(), |acc, v| acc + v)
                })
                .collect();
            let mean_score = edge_scores
                .iter()
                .enumerate()
                .fold(T::zero(), |acc, (e, &s)| {
                    acc + s * edge_probs.get(e).copied().unwrap_or_else(T::zero)
                });
            let edge_lr: T = scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero());
            let updatable = num_edges.min(self.edge_weights.len());
            for (e, &score) in edge_scores.iter().enumerate().take(updatable) {
                let beta = edge_probs.get(e).copied().unwrap_or_else(T::zero);
                self.edge_weights[e] = self.edge_weights[e] + edge_lr * beta * (score - mean_score);
            }
        }

        // Move the baseline toward the observed batch mean.
        let beta: T = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());
        self.reward_baseline = self.reward_baseline * (T::one() - beta) + batch_mean * beta;

        // Anneal temperature
        self.temperature = self.temperature
            * scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero());

        Ok(())
    }

    fn name(&self) -> &str {
        "MemoryEfficientDARTS"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate = self.temperature;
        stats.exploitation_rate = T::one() - self.temperature;
        stats
    }
}

// ─── RobustDARTS impl ───────────────────────────────────────────────────────

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + scirs2_core::ndarray::ScalarOperand
            + std::iter::Sum,
    > RobustDARTS<T>
{
    /// Create a new RobustDARTS instance
    ///
    /// # Arguments
    /// * `num_operations` - Number of candidate operations per edge
    /// * `num_edges` - Number of edges in the search DAG
    /// * `temperature` - Initial softmax temperature
    /// * `use_gumbel` - Whether to use Gumbel-Softmax relaxation
    /// * `perturbation_strength` - Magnitude of perturbation for regularization
    /// * `early_stopping_patience` - Steps without improvement before stopping
    /// * `regularization_weight` - Weight of the perturbation regularization term
    pub fn new(
        num_operations: usize,
        num_edges: usize,
        temperature: f64,
        use_gumbel: bool,
        perturbation_strength: f64,
        early_stopping_patience: usize,
        regularization_weight: f64,
    ) -> Self {
        Self {
            architecture_weights: Array3::zeros((num_edges, num_operations, 1)),
            weight_optimizer: WeightOptimizer::new(
                scirs2_core::numeric::NumCast::from(0.025).unwrap_or_else(|| T::zero()),
            ),
            temperature: scirs2_core::numeric::NumCast::from(temperature)
                .unwrap_or_else(|| T::zero()),
            gumbel_softmax: use_gumbel,
            perturbation_strength: scirs2_core::numeric::NumCast::from(perturbation_strength)
                .unwrap_or_else(|| T::zero()),
            early_stopping_patience,
            regularization_weight: scirs2_core::numeric::NumCast::from(regularization_weight)
                .unwrap_or_else(|| T::zero()),
            discretization_strategy: DiscretizationStrategy::Greedy,
            statistics: SearchStrategyStatistics::default(),
            eigenvalue_history: Vec::new(),
            steps_without_improvement: 0,
            best_validation_performance: T::neg_infinity(),
            early_stopped: false,
            reward_baseline: T::zero(),
        }
    }

    /// Compute perturbation regularization loss
    ///
    /// Measures sensitivity of the architecture to weight perturbations.
    /// High sensitivity indicates potential performance collapse.
    fn compute_perturbation_loss(&self, weights: &Array3<T>) -> T {
        let perturbation = Array3::from_shape_fn(weights.raw_dim(), |_| {
            let noise = Random::default().random::<f64>();
            self.perturbation_strength * T::from(noise * 2.0 - 1.0).unwrap_or_else(|| T::zero())
        });

        let perturbed_weights = weights + &perturbation;

        // Compute KL divergence between original and perturbed softmax distributions
        let mut total_kl = T::zero();
        for edge_idx in 0..weights.dim().0 {
            let original = softmax(&weights.slice(s![edge_idx, .., 0]).to_owned());
            let perturbed = softmax(&perturbed_weights.slice(s![edge_idx, .., 0]).to_owned());

            // KL(original || perturbed)
            for (p, q) in original.iter().zip(perturbed.iter()) {
                let epsilon: T =
                    scirs2_core::numeric::NumCast::from(1e-10).unwrap_or_else(|| T::zero());
                if *p > epsilon && *q > epsilon {
                    total_kl = total_kl + *p * (*p / *q).ln();
                }
            }
        }

        total_kl
    }

    /// Estimate the dominant eigenvalue of the Hessian (simplified)
    ///
    /// Used for early stopping: rapid growth of eigenvalues indicates
    /// the search is approaching performance collapse.
    fn estimate_dominant_eigenvalue(&self) -> T {
        // Simplified: use the variance of architecture weights as a proxy
        let num_elements = self.architecture_weights.len();
        if num_elements == 0 {
            return T::zero();
        }
        let mean =
            self.architecture_weights.sum() / T::from(num_elements).unwrap_or_else(|| T::one());
        let variance = self
            .architecture_weights
            .mapv(|x| (x - mean) * (x - mean))
            .sum()
            / T::from(num_elements).unwrap_or_else(|| T::one());
        variance.sqrt()
    }

    /// Check if early stopping criteria is met
    fn should_early_stop(&self) -> bool {
        self.early_stopped || self.steps_without_improvement >= self.early_stopping_patience
    }

    fn gumbel_softmax_sample(&self, logits: &Array1<T>) -> Array1<T> {
        if !self.gumbel_softmax {
            return softmax(logits);
        }

        let gumbel_noise: Array1<T> = Array1::from_shape_fn(logits.len(), |_| {
            let u = scirs2_core::random::Random::default().random::<f64>();
            T::from(gumbel_noise_from_uniform(u)).unwrap_or_else(|| T::zero())
        });

        let gumbel_logits = logits + &gumbel_noise;
        let scaled_logits = gumbel_logits / self.temperature;
        softmax(&scaled_logits)
    }
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > SearchStrategy<T> for RobustDARTS<T>
{
    fn initialize(&mut self, _searchspace: &SearchSpaceConfig) -> Result<()> {
        self.architecture_weights =
            Array3::from_shape_fn(self.architecture_weights.raw_dim(), |_| {
                T::from(scirs2_core::random::Random::default().random::<f64>() * 0.1 - 0.05)
                    .unwrap_or_else(|| T::zero())
            });
        self.eigenvalue_history.clear();
        self.steps_without_improvement = 0;
        self.best_validation_performance = T::neg_infinity();
        self.early_stopped = false;
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        _search_space: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        // If early-stopped, return the best architecture found so far
        if self.should_early_stop() {
            self.statistics.total_architectures_generated += 1;
            return Ok(discretize_architecture(
                &self.architecture_weights,
                &DiscretizationStrategy::Greedy,
            ));
        }

        // Sample using Gumbel-Softmax
        let mut sampled_weights = Array3::zeros(self.architecture_weights.raw_dim());
        for edge_idx in 0..self.architecture_weights.dim().0 {
            let edge_weights = self.architecture_weights.slice(s![edge_idx, .., 0]);
            let sampled = self.gumbel_softmax_sample(&edge_weights.to_owned());
            for (op_idx, &weight) in sampled.iter().enumerate() {
                sampled_weights[[edge_idx, op_idx, 0]] = weight;
            }
        }

        self.statistics.total_architectures_generated += 1;
        Ok(discretize_architecture(
            &sampled_weights,
            &self.discretization_strategy,
        ))
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        if results.is_empty() || self.early_stopped {
            return Ok(());
        }

        let scored = collect_scored(results);
        let performances: Vec<T> = scored.iter().map(|(_, p)| *p).collect();

        if !performances.is_empty() {
            let current_best = performances
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(T::zero());

            self.statistics.best_performance = current_best;

            let sum: T = performances.iter().cloned().sum();
            let count = T::from(performances.len()).unwrap_or_else(|| T::one());
            let batch_mean = sum / count;
            self.statistics.average_performance = batch_mean;

            // Check for improvement (early stopping logic)
            if current_best > self.best_validation_performance {
                self.best_validation_performance = current_best;
                self.steps_without_improvement = 0;
            } else {
                self.steps_without_improvement += 1;
            }

            // Track eigenvalue for collapse detection
            let eigenvalue = self.estimate_dominant_eigenvalue();
            self.eigenvalue_history.push(eigenvalue);

            // Check for rapid eigenvalue growth (indicates collapse)
            if self.eigenvalue_history.len() >= 10 {
                let recent_len = self.eigenvalue_history.len();
                let recent_eigenvalue = self.eigenvalue_history[recent_len - 1];
                let earlier_eigenvalue = self.eigenvalue_history[recent_len - 10];
                let growth_ratio = if earlier_eigenvalue
                    > scirs2_core::numeric::NumCast::from(1e-10).unwrap_or_else(|| T::zero())
                {
                    recent_eigenvalue / earlier_eigenvalue
                } else {
                    T::one()
                };

                let collapse_threshold: T =
                    scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero());
                if growth_ratio > collapse_threshold {
                    self.early_stopped = true;
                    return Ok(());
                }
            }

            // Early stopping check
            if self.should_early_stop() {
                self.early_stopped = true;
                return Ok(());
            }

            // Compute perturbation regularization. The penalty is shared by the
            // whole batch (it is a property of the current logits, not of any
            // one candidate), so it is folded into every advantage.
            let perturbation_loss = self.compute_perturbation_loss(&self.architecture_weights);
            let penalty = self.regularization_weight * perturbation_loss;

            // Advantage against the running baseline read *before* it is
            // updated, so a single-sample batch still carries signal.
            let baseline = self.reward_baseline;
            let advantages: Vec<T> = performances
                .iter()
                .map(|&p| p - baseline - penalty)
                .collect();

            // Real per-entry REINFORCE gradient (non-uniform across operations)
            // applied through the momentum / weight-decay optimizer. The former
            // uniform `weights + ones * lr * reward` update was provably inert:
            // adding the same scalar to every logit leaves the per-edge softmax
            // — and therefore the sampled architecture — unchanged.
            let grad = reinforce_logit_gradient(&self.architecture_weights, &scored, &advantages);
            self.weight_optimizer
                .step(&mut self.architecture_weights, &grad);

            // Move the baseline toward the observed batch mean.
            let beta: T = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());
            self.reward_baseline = self.reward_baseline * (T::one() - beta) + batch_mean * beta;

            // Anneal temperature
            self.temperature = self.temperature
                * scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero());
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "RobustDARTS"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate = self.temperature;
        stats.exploitation_rate = T::one() - self.temperature;
        stats
    }
}

// ─── DARTSConfig impl ───────────────────────────────────────────────────────

impl<T: Float + Debug + Default + Send + Sync + 'static> Default for DARTSConfig<T> {
    fn default() -> Self {
        Self {
            num_operations: 4,
            num_edges: 8,
            initial_temperature: 1.0,
            use_gumbel: true,
            temperature_schedule: TemperatureSchedule::Exponential {
                initial: 1.0,
                decay_rate: 0.999,
            },
            discretization_strategy: DiscretizationStrategy::Progressive,
            architecture_lr: 0.025,
            weight_decay: 1e-4,
            momentum: 0.9,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + 'static
            + scirs2_core::ndarray::ScalarOperand
            + std::iter::Sum,
    > DARTSConfig<T>
{
    /// Create a new DARTSConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of operations
    pub fn num_operations(mut self, n: usize) -> Self {
        self.num_operations = n;
        self
    }

    /// Set the number of edges
    pub fn num_edges(mut self, n: usize) -> Self {
        self.num_edges = n;
        self
    }

    /// Set the initial temperature
    pub fn temperature(mut self, t: f64) -> Self {
        self.initial_temperature = t;
        self
    }

    /// Set whether to use Gumbel-Softmax
    pub fn gumbel_softmax(mut self, use_gumbel: bool) -> Self {
        self.use_gumbel = use_gumbel;
        self
    }

    /// Set the temperature schedule
    pub fn temperature_schedule(mut self, schedule: TemperatureSchedule) -> Self {
        self.temperature_schedule = schedule;
        self
    }

    /// Set the discretization strategy
    pub fn discretization_strategy(mut self, strategy: DiscretizationStrategy) -> Self {
        self.discretization_strategy = strategy;
        self
    }

    /// Set the architecture learning rate
    pub fn architecture_lr(mut self, lr: f64) -> Self {
        self.architecture_lr = lr;
        self
    }

    /// Set the weight decay
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set the momentum
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }

    /// Build a standard DifferentiableSearch from this config
    pub fn build(self) -> DifferentiableSearch<T> {
        DifferentiableSearch::new(
            self.num_operations,
            self.num_edges,
            self.initial_temperature,
            self.use_gumbel,
        )
    }

    /// Build a MemoryEfficientDARTS from this config
    pub fn build_memory_efficient(
        self,
        partial_channel_ratio: f64,
        edge_normalization: bool,
    ) -> MemoryEfficientDARTS<T> {
        MemoryEfficientDARTS::new(
            self.num_operations,
            self.num_edges,
            self.initial_temperature,
            self.use_gumbel,
            partial_channel_ratio,
            edge_normalization,
        )
    }

    /// Build a RobustDARTS from this config
    pub fn build_robust(
        self,
        perturbation_strength: f64,
        early_stopping_patience: usize,
        regularization_weight: f64,
    ) -> RobustDARTS<T> {
        RobustDARTS::new(
            self.num_operations,
            self.num_edges,
            self.initial_temperature,
            self.use_gumbel,
            perturbation_strength,
            early_stopping_patience,
            regularization_weight,
        )
    }
}

// ─── WeightOptimizer impl ───────────────────────────────────────────────────

impl<T: Float + Debug + Default + Send + Sync> WeightOptimizer<T> {
    fn new(learningrate: T) -> Self {
        Self {
            learning_rate: learningrate,
            momentum: scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero()),
            weight_decay: scirs2_core::numeric::NumCast::from(1e-4).unwrap_or_else(|| T::zero()),
            velocity: Array3::zeros((0, 0, 0)),
        }
    }

    /// One step of momentum SGD **ascent** with decoupled weight decay.
    ///
    /// Updates `velocity ← momentum·velocity + grad` then
    /// `weights ← weights − lr·weight_decay·weights + lr·velocity`, mutating
    /// `weights` in place. This *ascends* `grad` because REINFORCE maximises
    /// reward, so callers pass the raw policy-gradient estimate. The velocity
    /// buffer is lazily resized to match `weights` on first use.
    fn step(&mut self, weights: &mut Array3<T>, grad: &Array3<T>) {
        if self.velocity.raw_dim() != weights.raw_dim() {
            self.velocity = Array3::zeros(weights.raw_dim());
        }
        let (edges, ops, depth) = weights.dim();
        let lr = self.learning_rate;
        let momentum = self.momentum;
        let weight_decay = self.weight_decay;
        for e in 0..edges {
            for o in 0..ops {
                for d in 0..depth {
                    let g = grad[[e, o, d]];
                    let v = momentum * self.velocity[[e, o, d]] + g;
                    self.velocity[[e, o, d]] = v;
                    let w = weights[[e, o, d]];
                    weights[[e, o, d]] = w - lr * weight_decay * w + lr * v;
                }
            }
        }
    }
}

// ─── Shared utility functions ────────────────────────────────────────────────

/// Compute softmax of a 1D array
fn softmax<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>(
    x: &Array1<T>,
) -> Array1<T> {
    let max_val = x
        .iter()
        .cloned()
        .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });
    let exp_x = x.mapv(|xi| (xi - max_val).exp());
    let sum_exp = exp_x.sum();
    exp_x / sum_exp
}

/// Sample-free Gumbel noise from a uniform draw `u ∈ (0, 1)`.
///
/// Returns `−ln(−ln(u))`, the inverse-CDF transform that turns a uniform
/// variate into a standard Gumbel(0, 1) sample used by the Gumbel-softmax
/// relaxation. `u` is clamped away from `0` and `1` so the two logarithms
/// stay finite.
fn gumbel_noise_from_uniform(u: f64) -> f64 {
    let u = u.clamp(1e-12, 1.0 - 1e-12);
    -(-u.ln()).ln()
}

/// Metadata key prefix under which [`discretize_architecture`] records the raw
/// operation index selected on each edge. The full key is
/// `"{DARTS_OP_INDEX_PREFIX}{edge}"`.
const DARTS_OP_INDEX_PREFIX: &str = "darts_op_index_";

/// Metadata key recording how many candidate operations the relaxation had when
/// the architecture was discretized. Used to reject an encoding produced under a
/// different search space instead of trusting an out-of-range index.
const DARTS_NUM_OPS_KEY: &str = "darts_num_ops";

/// Map an operation index onto the concrete optimizer [`ComponentType`] that
/// [`discretize_architecture`] emits for it.
///
/// This is the exact inverse of
/// [`crate::search_strategies::component_type_to_u8`] over `0..=40`, which makes
/// the index → name direction injective for every operation index the component
/// vocabulary can express. Indices past the vocabulary collapse onto
/// [`ComponentType::Custom`]; that collision is deliberate and visible, and
/// [`recover_chosen_op`] declines to guess when it hits one instead of silently
/// crediting the wrong operation.
fn op_index_to_component_type(idx: usize) -> ComponentType {
    match idx {
        0 => ComponentType::SGD,
        1 => ComponentType::Adam,
        2 => ComponentType::AdaGrad,
        3 => ComponentType::RMSprop,
        4 => ComponentType::AdamW,
        5 => ComponentType::LAMB,
        6 => ComponentType::LARS,
        7 => ComponentType::Lion,
        8 => ComponentType::RAdam,
        9 => ComponentType::Lookahead,
        10 => ComponentType::SAM,
        11 => ComponentType::LBFGS,
        12 => ComponentType::SparseAdam,
        13 => ComponentType::GroupedAdam,
        14 => ComponentType::MAML,
        15 => ComponentType::Reptile,
        16 => ComponentType::MetaSGD,
        17 => ComponentType::ConstantLR,
        18 => ComponentType::ExponentialLR,
        19 => ComponentType::StepLR,
        20 => ComponentType::CosineAnnealingLR,
        21 => ComponentType::OneCycleLR,
        22 => ComponentType::CyclicLR,
        23 => ComponentType::L1Regularizer,
        24 => ComponentType::L2Regularizer,
        25 => ComponentType::ElasticNetRegularizer,
        26 => ComponentType::DropoutRegularizer,
        27 => ComponentType::GradientClipping,
        28 => ComponentType::WeightDecay,
        29 => ComponentType::AdaptiveLR,
        30 => ComponentType::AdaptiveMomentum,
        31 => ComponentType::AdaptiveRegularization,
        32 => ComponentType::LSTMOptimizer,
        33 => ComponentType::TransformerOptimizer,
        34 => ComponentType::AttentionOptimizer,
        35 => ComponentType::AdaDelta,
        36 => ComponentType::Momentum,
        37 => ComponentType::Nesterov,
        38 => ComponentType::LRScheduler,
        39 => ComponentType::BatchNorm,
        40 => ComponentType::Dropout,
        _ => ComponentType::Custom,
    }
}

/// Pair every result that reports a final-performance score with that score.
///
/// The architecture is kept alongside its reward so the policy gradient can
/// attribute credit to the operations that result actually selected; dropping to
/// a bare `Vec<T>` of scores is what forces the uniform (softmax-invariant, and
/// therefore inert) update this module used to apply.
fn collect_scored<T: Float + Debug + Send + Sync + 'static>(
    results: &[SearchResult<T>],
) -> Vec<(&SearchResult<T>, T)> {
    results
        .iter()
        .filter_map(|r| {
            r.evaluation_results
                .metric_scores
                .get(&EvaluationMetric::FinalPerformance)
                .map(|p| (r, *p))
        })
        .collect()
}

/// Index of the largest entry in `v` (first on ties), or `0` when empty.
fn argmax_index<T: Float>(v: &Array1<T>) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Recover the operation chosen on `edge` from a produced architecture.
///
/// Two sources are consulted, in order:
///
/// 1. **The lossless encoding.** [`discretize_architecture`] records the raw
///    selected index in `architecture.metadata` under
///    `"{DARTS_OP_INDEX_PREFIX}{edge}"`. It is accepted only when it is in range
///    for `num_ops`, when the recorded operation count agrees with `num_ops`,
///    and when it still agrees with the component name actually present on that
///    edge — so an architecture whose components were rewritten after
///    discretization (a mutation, a crossover) cannot smuggle a stale index
///    through.
/// 2. **The component name**, for architectures produced before or outside this
///    encoding. The name is only trusted when it identifies *exactly one*
///    operation index in `0..num_ops`; an ambiguous name (several indices share
///    it, which happens once `num_ops` exceeds the component vocabulary) yields
///    `None` rather than the first match.
///
/// `None` tells the caller the choice is genuinely unknown so it can fall back
/// to an explicit proxy instead of crediting an arbitrary operation.
fn recover_chosen_op<T: Float + Debug + Send + Sync + 'static>(
    result: &SearchResult<T>,
    edge: usize,
    num_ops: usize,
) -> Option<usize> {
    let architecture = &result.architecture;
    let comp = architecture.components.get(edge)?;

    // 1. Lossless per-edge index recorded at discretization time.
    let encoded_num_ops = architecture
        .metadata
        .get(DARTS_NUM_OPS_KEY)
        .and_then(|value| value.parse::<usize>().ok());
    if encoded_num_ops.is_none_or(|encoded| encoded == num_ops) {
        let encoded_index = architecture
            .metadata
            .get(&format!("{}{}", DARTS_OP_INDEX_PREFIX, edge))
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&idx| idx < num_ops)
            .filter(|&idx| &op_index_to_component_type(idx).to_string() == comp);
        if let Some(idx) = encoded_index {
            return Some(idx);
        }
    }

    // 2. Name round-trip, accepted only when it is unambiguous.
    let mut matches =
        (0..num_ops).filter(|&op| &op_index_to_component_type(op).to_string() == comp);
    let first = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(first)
}

/// REINFORCE score-function gradient for the per-edge categorical softmax
/// policy defined by `weights` (logits of shape `[edges, ops, 1]`).
///
/// For each scored sample the operation actually chosen on every edge is
/// recovered from the produced architecture and credited with that sample's
/// advantage, giving the standard estimator
/// `∇logit[e, o] = mean_i A_i · (1{o = a_{i,e}} − p[e, o])`, where `p` is the
/// current per-edge softmax. The term is genuinely non-uniform across
/// operations (the `−p[e, o]` subtracts probability mass from the alternatives
/// not chosen). When an edge's choice cannot be recovered the policy's current
/// argmax is used so the estimate stays defined rather than silently zero.
fn reinforce_logit_gradient<T>(
    weights: &Array3<T>,
    scored: &[(&SearchResult<T>, T)],
    advantages: &[T],
) -> Array3<T>
where
    T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand,
{
    let mut grad = Array3::zeros(weights.raw_dim());
    let (num_edges, num_ops, _depth) = weights.dim();
    if num_edges == 0 || num_ops == 0 || scored.is_empty() {
        return grad;
    }

    // Per-edge softmax probabilities of the current logits.
    let probs: Vec<Array1<T>> = (0..num_edges)
        .map(|e| softmax(&weights.slice(s![e, .., 0]).to_owned()))
        .collect();

    let count = T::from(scored.len()).unwrap_or_else(|| T::one());
    for (i, (result, _)) in scored.iter().enumerate() {
        let advantage = advantages.get(i).copied().unwrap_or_else(|| T::zero());
        for (e, p) in probs.iter().enumerate() {
            let chosen = recover_chosen_op(result, e, num_ops).unwrap_or_else(|| argmax_index(p));
            for o in 0..num_ops {
                let indicator = if o == chosen { T::one() } else { T::zero() };
                grad[[e, o, 0]] = grad[[e, o, 0]] + advantage * (indicator - p[o]) / count;
            }
        }
    }
    grad
}

/// Discretize continuous architecture weights into a concrete architecture
fn discretize_architecture<
    T: Float + Debug + Default + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand,
>(
    weights: &Array3<T>,
    strategy: &DiscretizationStrategy,
) -> OptimizerArchitecture<T> {
    use crate::architecture::OptimizerComponent;

    let num_ops = weights.dim().1;
    let mut components = Vec::new();
    // Raw per-edge operation indices, carried into the architecture's metadata so
    // credit assignment can recover the exact choice instead of guessing it back
    // from a component name (which is not injective once `num_ops` exceeds the
    // component vocabulary).
    let mut selected_ops: Vec<usize> = Vec::with_capacity(weights.dim().0);

    for edge_idx in 0..weights.dim().0 {
        let edge_weights = weights.slice(s![edge_idx, .., 0]);

        let selected_op_idx = match strategy {
            DiscretizationStrategy::Greedy => edge_weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            DiscretizationStrategy::Sampling => {
                let probs = softmax(&edge_weights.to_owned());
                let rand_val = scirs2_core::random::Random::default().random::<f64>();
                let mut cumsum = 0.0;

                let mut selected_idx = 0;
                for (idx, prob) in probs.iter().enumerate() {
                    cumsum += prob.to_f64().unwrap_or(0.0);
                    if cumsum >= rand_val {
                        selected_idx = idx;
                        break;
                    }
                }
                selected_idx
            }
            DiscretizationStrategy::Threshold => {
                let threshold: T =
                    scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
                edge_weights
                    .iter()
                    .enumerate()
                    .find(|(_, &weight)| weight > threshold)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
            DiscretizationStrategy::Progressive => {
                // Take the operation with the largest weight.
                //
                // This used to square every entry first ("gradually sharpen the
                // distribution") and then argmax. Squaring is a monotone
                // transform only on non-negative values: on the raw-logit paths
                // (`DifferentiableSearch` with `continuous_relaxation == false`
                // and `RobustDARTS`'s early-stop branch) the weights are signed,
                // so squaring turns magnitude into rank and the **most negative**
                // logit won — i.e. the operation the relaxation liked least.
                // `Progressive` is the default strategy for `DifferentiableSearch`
                // and `MemoryEfficientDARTS`, so this silently mis-selected on the
                // default configuration.
                //
                // Argmax is invariant to any *monotone* sharpening anyway, so the
                // squaring could only ever change the answer by being wrong; the
                // honest implementation is the plain argmax over the weights as
                // given.
                edge_weights
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        };

        // Map operation index to component type (kept in sync with the inverse
        // used by `recover_chosen_op` for the REINFORCE credit assignment).
        let component_type = op_index_to_component_type(selected_op_idx);
        selected_ops.push(selected_op_idx);

        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("_learningrate".to_string(), 0.001f64);

        components.push(OptimizerComponent {
            id: format!("comp_{}", edge_idx),
            component_type,
            hyperparameters,
            enabled: true,
            position: ComponentPosition {
                layer: 0,
                index: edge_idx as u32,
                x: 0.0,
                y: 0.0,
            },
        });
    }

    OptimizerArchitecture {
        components: components
            .iter()
            .map(|c| c.component_type.to_string())
            .collect(),
        parameters: components
            .iter()
            .enumerate()
            .flat_map(|(i, c)| {
                c.hyperparameters.iter().map(move |(k, v)| {
                    (
                        format!("{}_{}", i, k),
                        scirs2_core::numeric::NumCast::from(*v).unwrap_or_else(|| T::zero()),
                    )
                })
            })
            .collect(),
        hyperparameters: components
            .iter()
            .enumerate()
            .flat_map(|(i, c)| {
                c.hyperparameters.iter().map(move |(k, v)| {
                    (
                        format!("{}_{}", i, k),
                        scirs2_core::numeric::NumCast::from(*v).unwrap_or_else(|| T::zero()),
                    )
                })
            })
            .collect(),
        connections: Vec::new(),
        metadata: {
            // Lossless per-edge encoding of the discretization decision. Without
            // it the only record of the chosen operation is the component name,
            // which several operation indices share once `num_ops` outgrows the
            // component vocabulary — silently misattributing REINFORCE credit.
            let mut metadata = HashMap::with_capacity(selected_ops.len() + 1);
            metadata.insert(DARTS_NUM_OPS_KEY.to_string(), num_ops.to_string());
            for (edge_idx, op_idx) in selected_ops.iter().enumerate() {
                metadata.insert(
                    format!("{}{}", DARTS_OP_INDEX_PREFIX, edge_idx),
                    op_idx.to_string(),
                );
            }
            metadata
        },
        architecture_id: format!("arch_{}", Random::default().random::<u64>()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::results::{
        ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
    };
    use std::time::Duration;

    /// `DiscretizationStrategy::Progressive` must select the operation with the
    /// **largest** weight, including on the raw-logit paths where weights are
    /// signed. The pre-fix implementation squared every entry before the argmax,
    /// which ranked by magnitude and therefore picked the most negative logit.
    #[test]
    fn progressive_discretization_picks_the_most_positive_raw_logit() {
        // One edge, three signed logits: index 1 is the largest, index 2 has the
        // largest magnitude.
        let mut weights = Array3::<f64>::zeros((1, 3, 1));
        weights[[0, 0, 0]] = -3.0;
        weights[[0, 1, 0]] = -0.5;
        weights[[0, 2, 0]] = -5.0;

        let progressive = discretize_architecture(&weights, &DiscretizationStrategy::Progressive);
        let greedy = discretize_architecture(&weights, &DiscretizationStrategy::Greedy);

        let progressive_idx = recorded_op_index(&progressive, 0).expect("op index recorded");
        let greedy_idx = recorded_op_index(&greedy, 0).expect("op index recorded");

        assert_eq!(
            progressive_idx, 1,
            "Progressive must choose the largest logit (index 1); squaring the \
             signed logits selects index 2, the most negative one"
        );
        assert_eq!(
            progressive_idx, greedy_idx,
            "argmax is invariant to monotone sharpening, so Progressive and \
             Greedy must agree"
        );
        assert_ne!(progressive_idx, 2, "the most negative logit must never win");
    }

    /// The same must hold across a multi-edge relaxation, and on probability-valued
    /// weights (where squaring happened to be harmless).
    #[test]
    fn progressive_discretization_agrees_with_greedy_on_every_edge() {
        let mut weights = Array3::<f64>::zeros((4, 5, 1));
        // Mixed signs per edge; the intended winner is different on each edge.
        let table = [
            [-1.0, -2.0, 0.5, -9.0, -0.25],
            [3.0, 0.1, -7.0, 2.9, -0.5],
            [-0.2, -0.1, -0.3, -0.4, -0.5],
            [0.05, 0.10, 0.60, 0.20, 0.05],
        ];
        for (edge, row) in table.iter().enumerate() {
            for (op, value) in row.iter().enumerate() {
                weights[[edge, op, 0]] = *value;
            }
        }

        let progressive = discretize_architecture(&weights, &DiscretizationStrategy::Progressive);
        for (edge, row) in table.iter().enumerate() {
            let expected = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .expect("non-empty row");
            let selected = recorded_op_index(&progressive, edge).expect("op index recorded");
            assert_eq!(
                selected, expected,
                "edge {edge}: expected op {expected}, got {selected}"
            );
        }
    }

    /// Read back the raw operation index `discretize_architecture` records in the
    /// architecture metadata for `edge`.
    fn recorded_op_index(architecture: &OptimizerArchitecture<f64>, edge: usize) -> Option<usize> {
        architecture
            .metadata
            .get(&format!("darts_op_index_{}", edge))
            .and_then(|raw| raw.parse::<usize>().ok())
    }

    /// Wrap an already-built architecture in a `SearchResult` carrying a single
    /// final-performance score.
    fn search_result_for(
        architecture: OptimizerArchitecture<f64>,
        performance: f64,
    ) -> SearchResult<f64> {
        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::FinalPerformance, performance);

        SearchResult {
            architecture,
            evaluation_results: EvaluationResults {
                metric_scores,
                overall_score: performance,
                confidence_intervals: HashMap::new(),
                evaluation_time: Duration::from_secs(0),
                success: true,
                error_message: None,
                cv_results: None,
                benchmark_results: HashMap::new(),
                training_trajectory: Vec::new(),
            },
            generation: 0,
            search_time: 0.0,
            resource_usage: ResourceUsage::default(),
            encoding: ArchitectureEncoding::default(),
            metadata: SearchResultMetadata::default(),
        }
    }

    #[test]
    fn test_darts_creation() {
        let search = DifferentiableSearch::<f64>::new(4, 8, 1.0, true);
        assert_eq!(search.name(), "DifferentiableSearch");
    }

    #[test]
    fn test_memory_efficient_darts_creation() {
        let search = MemoryEfficientDARTS::<f64>::new(4, 8, 1.0, true, 0.25, true);
        assert_eq!(search.name(), "MemoryEfficientDARTS");
    }

    #[test]
    fn test_robust_darts_creation() {
        let search = RobustDARTS::<f64>::new(4, 8, 1.0, true, 0.1, 20, 0.01);
        assert_eq!(search.name(), "RobustDARTS");
        assert!(!search.should_early_stop());
    }

    #[test]
    fn test_darts_config_builder() {
        let config = DARTSConfig::<f64>::new()
            .num_operations(6)
            .num_edges(12)
            .temperature(0.5)
            .gumbel_softmax(false)
            .discretization_strategy(DiscretizationStrategy::Greedy);

        assert_eq!(config.num_operations, 6);
        assert_eq!(config.num_edges, 12);

        let search = config.build();
        assert_eq!(search.name(), "DifferentiableSearch");
    }

    #[test]
    fn test_darts_config_build_variants() {
        let config = DARTSConfig::<f64>::new();
        let _me = config.clone().build_memory_efficient(0.25, true);
        let _robust = config.build_robust(0.1, 20, 0.01);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Highest logit should have highest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    /// The operation-index encoding must be the exact inverse of the crate-wide
    /// canonical `component_type_to_u8` table. If either side drifts, the DARTS
    /// credit assignment silently starts attributing reward to the wrong
    /// operation, so pin the round-trip.
    #[test]
    fn test_op_index_component_type_round_trip_is_a_bijection() {
        use crate::search_strategies::component_type_to_u8;
        let mut seen = std::collections::HashSet::new();
        for idx in 0..=40usize {
            let component = op_index_to_component_type(idx);
            assert_eq!(
                component_type_to_u8(&component) as usize,
                idx,
                "op index {idx} does not round-trip through the canonical table"
            );
            assert!(
                seen.insert(component.to_string()),
                "op index {idx} collides with an earlier operation name"
            );
        }
        // Past the component vocabulary the mapping is deliberately collapsed.
        assert_eq!(op_index_to_component_type(41), ComponentType::Custom);
    }

    /// FC2 regression: with more than four operations the chosen op used to be
    /// recovered by round-tripping through the component *name*, and every index
    /// from 4 upwards stringified to "Adam" — so REINFORCE credit landed on op 1
    /// whatever was actually selected. The per-edge index is now carried
    /// losslessly in the architecture metadata.
    #[test]
    fn test_reinforce_credit_lands_on_chosen_op_with_six_operations() {
        const NUM_OPS: usize = 6;
        const CHOSEN: usize = 5;

        // Uniform logits so the softmax is exactly 1/NUM_OPS everywhere and the
        // gradient sign is decided purely by the recovered choice.
        let logits: Array3<f64> = Array3::zeros((1, NUM_OPS, 1));

        // Distinct sampled weights whose argmax is the last operation.
        let mut sampled: Array3<f64> = Array3::zeros((1, NUM_OPS, 1));
        for op in 0..NUM_OPS {
            sampled[[0, op, 0]] = 0.01 * op as f64;
        }

        let architecture = discretize_architecture(&sampled, &DiscretizationStrategy::Greedy);
        assert_eq!(
            architecture.metadata.get("darts_op_index_0"),
            Some(&CHOSEN.to_string()),
            "discretization must record the raw operation index"
        );

        let result = search_result_for(architecture, 1.0);
        assert_eq!(
            recover_chosen_op(&result, 0, NUM_OPS),
            Some(CHOSEN),
            "the chosen op must be recovered exactly, not via the lossy name map"
        );

        let scored = vec![(&result, 1.0f64)];
        let grad = reinforce_logit_gradient(&logits, &scored, &[1.0]);

        let credited = argmax_index(&grad.slice(s![0, .., 0]).to_owned());
        assert_eq!(credited, CHOSEN, "credit must land on the chosen operation");
        assert!(
            grad[[0, CHOSEN, 0]] > 0.0,
            "the chosen op must receive positive credit"
        );
        // The pre-fix failure mode: op 1 ("Adam") absorbed the credit.
        assert!(
            grad[[0, 1, 0]] < 0.0,
            "op 1 must lose probability mass, not gain it"
        );
        // Score-function gradients sum to zero over the categorical support.
        let column_sum: f64 = (0..NUM_OPS).map(|o| grad[[0, o, 0]]).sum();
        assert!(column_sum.abs() < 1e-12);
    }

    /// Behaviour for <= 4 operations is unchanged: the name map is injective
    /// there, so an architecture without the encoding still recovers exactly.
    #[test]
    fn test_recover_chosen_op_name_fallback_is_exact_for_four_operations() {
        let mut architecture: OptimizerArchitecture<f64> =
            discretize_architecture(&Array3::zeros((1, 4, 1)), &DiscretizationStrategy::Greedy);
        architecture.components = vec!["AdaGrad".to_string()];
        architecture.metadata.clear();

        let result = search_result_for(architecture, 1.0);
        assert_eq!(recover_chosen_op(&result, 0, 4), Some(2));
    }

    /// Without the encoding and with more operations than the vocabulary can
    /// name uniquely, recovery declines instead of guessing — the caller then
    /// falls back to the policy's own argmax.
    #[test]
    fn test_recover_chosen_op_declines_when_name_is_ambiguous() {
        let mut architecture: OptimizerArchitecture<f64> =
            discretize_architecture(&Array3::zeros((1, 50, 1)), &DiscretizationStrategy::Greedy);
        // "Custom" is shared by every index past the component vocabulary.
        architecture.components = vec!["Custom".to_string()];
        architecture.metadata.clear();

        let result = search_result_for(architecture, 1.0);
        assert_eq!(recover_chosen_op(&result, 0, 50), None);

        // The argmax proxy still produces a defined, non-uniform gradient.
        let mut logits: Array3<f64> = Array3::zeros((1, 50, 1));
        logits[[0, 7, 0]] = 5.0;
        let scored = vec![(&result, 1.0f64)];
        let grad = reinforce_logit_gradient(&logits, &scored, &[1.0]);
        assert_eq!(argmax_index(&grad.slice(s![0, .., 0]).to_owned()), 7);
    }

    /// A stale index (components rewritten after discretization) must not be
    /// trusted; recovery falls through to the name check.
    #[test]
    fn test_recover_chosen_op_rejects_stale_encoding() {
        let mut architecture: OptimizerArchitecture<f64> =
            discretize_architecture(&Array3::zeros((1, 6, 1)), &DiscretizationStrategy::Greedy);
        architecture
            .metadata
            .insert("darts_op_index_0".to_string(), "4".to_string());
        // A mutation rewrote the component but left the metadata behind.
        architecture.components = vec!["Lion".to_string()];

        let result = search_result_for(architecture, 1.0);
        // "Lion" is op 7, which is out of range for 6 operations -> unknown.
        assert_eq!(recover_chosen_op(&result, 0, 6), None);
        // With room for op 7 the name identifies it unambiguously.
        assert_eq!(recover_chosen_op(&result, 0, 8), Some(7));
    }

    /// An out-of-range recorded index must never be handed back as the chosen
    /// operation (it would silently zero the positive term of the estimator).
    #[test]
    fn test_recover_chosen_op_rejects_out_of_range_encoding() {
        let mut architecture: OptimizerArchitecture<f64> =
            discretize_architecture(&Array3::zeros((1, 6, 1)), &DiscretizationStrategy::Greedy);
        architecture
            .metadata
            .insert("darts_op_index_0".to_string(), "99".to_string());
        architecture.components = vec!["SGD".to_string()];

        let result = search_result_for(architecture, 1.0);
        assert_eq!(recover_chosen_op(&result, 0, 6), Some(0));
    }

    /// An encoding produced under a different operation count is ignored rather
    /// than reinterpreted against the current one.
    #[test]
    fn test_recover_chosen_op_ignores_encoding_from_another_search_space() {
        let mut architecture: OptimizerArchitecture<f64> =
            discretize_architecture(&Array3::zeros((1, 6, 1)), &DiscretizationStrategy::Greedy);
        architecture
            .metadata
            .insert("darts_num_ops".to_string(), "12".to_string());
        architecture
            .metadata
            .insert("darts_op_index_0".to_string(), "5".to_string());
        architecture.components = vec!["LAMB".to_string()];

        let result = search_result_for(architecture, 1.0);
        // Metadata rejected; the name still resolves LAMB unambiguously to 5.
        assert_eq!(recover_chosen_op(&result, 0, 6), Some(5));
    }

    /// Every edge of a multi-edge architecture must be encoded and recovered
    /// independently.
    #[test]
    fn test_encoding_round_trips_every_edge() {
        const NUM_EDGES: usize = 4;
        const NUM_OPS: usize = 9;

        let mut sampled: Array3<f64> = Array3::zeros((NUM_EDGES, NUM_OPS, 1));
        let expected = [8usize, 0, 5, 3];
        for (edge, &op) in expected.iter().enumerate() {
            sampled[[edge, op, 0]] = 1.0;
        }

        let architecture = discretize_architecture(&sampled, &DiscretizationStrategy::Greedy);
        let result = search_result_for(architecture, 0.5);
        for (edge, &op) in expected.iter().enumerate() {
            assert_eq!(recover_chosen_op(&result, edge, NUM_OPS), Some(op));
        }
    }

    /// F13 regression: the DARTS variants used to add the *same* scalar to every
    /// logit, which a softmax is invariant to — the update could not change any
    /// sampling probability. Every variant must now move the per-edge
    /// distribution.
    #[test]
    fn test_variant_updates_change_the_sampling_distribution() {
        fn edge_probabilities(weights: &Array3<f64>) -> Vec<f64> {
            softmax(&weights.slice(s![0, .., 0]).to_owned()).to_vec()
        }

        let mut sampled: Array3<f64> = Array3::zeros((1, 6, 1));
        sampled[[0, 4, 0]] = 1.0;
        let architecture = discretize_architecture(&sampled, &DiscretizationStrategy::Greedy);
        let results = vec![search_result_for(architecture, 1.0)];

        let mut plain = DifferentiableSearch::<f64>::new(6, 1, 1.0, false);
        let before = edge_probabilities(&plain.architecture_weights);
        plain.update_with_results(&results).expect("plain update");
        let after = edge_probabilities(&plain.architecture_weights);
        assert!(after[4] > before[4], "chosen op must gain probability mass");

        let mut memory_efficient = MemoryEfficientDARTS::<f64>::new(6, 1, 1.0, false, 1.0, true);
        let before = edge_probabilities(&memory_efficient.architecture_weights);
        let edge_weights_before = memory_efficient.edge_weights.to_vec();
        memory_efficient
            .update_with_results(&results)
            .expect("memory-efficient update");
        let after = edge_probabilities(&memory_efficient.architecture_weights);
        assert!(
            after[4] > before[4],
            "PC-DARTS chosen op must gain probability mass"
        );
        // A single edge carries all of the edge-normalization mass, so its
        // centered gradient is exactly zero; the update must at least stay
        // finite and defined.
        assert!(memory_efficient
            .edge_weights
            .iter()
            .zip(edge_weights_before.iter())
            .all(|(a, b)| a.is_finite() && b.is_finite()));

        let mut robust = RobustDARTS::<f64>::new(6, 1, 1.0, false, 0.1, 20, 0.0);
        let before = edge_probabilities(&robust.architecture_weights);
        robust.update_with_results(&results).expect("robust update");
        let after = edge_probabilities(&robust.architecture_weights);
        assert!(
            after[4] > before[4],
            "RobustDARTS chosen op must gain probability mass"
        );
    }

    /// The multi-edge edge-normalization gradient must be genuinely non-uniform
    /// (the old uniform update left every edge weight identical forever).
    #[test]
    fn test_edge_normalization_weights_become_non_uniform() {
        let mut sampled: Array3<f64> = Array3::zeros((3, 5, 1));
        sampled[[0, 1, 0]] = 1.0;
        sampled[[1, 3, 0]] = 1.0;
        sampled[[2, 0, 0]] = 1.0;
        let architecture = discretize_architecture(&sampled, &DiscretizationStrategy::Greedy);
        let results = vec![search_result_for(architecture, 2.0)];

        let mut search = MemoryEfficientDARTS::<f64>::new(5, 3, 1.0, false, 1.0, true);
        search
            .initialize(&SearchSpaceConfig::default())
            .expect("initialize");
        search.update_with_results(&results).expect("update");

        let weights = search.edge_weights.to_vec();
        assert!(weights.iter().all(|w| w.is_finite()));
        let spread = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - weights.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            spread > 0.0,
            "edge-normalization weights must differentiate between edges"
        );
    }
}
