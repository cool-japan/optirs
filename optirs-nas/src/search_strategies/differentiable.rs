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
}

/// Weight optimizer for DARTS
#[derive(Debug)]
pub struct WeightOptimizer<T: Float + Debug + Send + Sync + 'static> {
    _learningrate: T,
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
        }
    }

    fn gumbel_softmax_sample(&self, logits: &Array1<T>) -> Array1<T> {
        if !self.gumbel_softmax {
            return softmax(logits);
        }

        let gumbel_noise: Array1<T> = Array1::from_shape_fn(logits.len(), |_| {
            let u = scirs2_core::random::Random::default().random::<f64>();
            T::from(-(-u.ln()).ln()).expect("gumbel noise conversion failed")
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
                    .expect("weight initialization conversion failed")
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

        // Compute gradients based on performance
        let performances: Vec<T> = results
            .iter()
            .filter_map(|r| {
                r.evaluation_results
                    .metric_scores
                    .get(&EvaluationMetric::FinalPerformance)
            })
            .cloned()
            .collect();

        if !performances.is_empty() {
            // Update statistics
            self.statistics.best_performance = performances
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(T::zero());

            let sum: T = performances.iter().cloned().sum();
            self.statistics.average_performance =
                sum / T::from(performances.len()).expect("conversion from usize to T failed");

            // Compute gradient estimate (simplified REINFORCE-style)
            let baseline = self.statistics.average_performance;
            let reward = performances[0] - baseline;

            // Update architecture weights
            let lr: T = scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero());
            self.architecture_weights = &self.architecture_weights
                + &(Array3::ones(self.architecture_weights.raw_dim()) * lr * reward);

            // Anneal temperature
            self.temperature = self.temperature
                * scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero());
        }

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
        }
    }

    /// Apply partial channel selection to architecture weights
    fn apply_partial_channels(&self, weights: &Array3<T>) -> Array3<T> {
        let num_edges = weights.dim().0;
        let num_ops = weights.dim().1;
        let mut masked = Array3::zeros(weights.raw_dim());

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
                    masked[[edge_idx, *op_idx, 0]] = weights[[edge_idx, *op_idx, 0]];
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
            T::from(-(-u.ln()).ln()).expect("gumbel noise conversion failed")
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
                    .expect("weight initialization conversion failed")
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

        let performances: Vec<T> = results
            .iter()
            .filter_map(|r| {
                r.evaluation_results
                    .metric_scores
                    .get(&EvaluationMetric::FinalPerformance)
            })
            .cloned()
            .collect();

        if !performances.is_empty() {
            self.statistics.best_performance = performances
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(T::zero());

            let sum: T = performances.iter().cloned().sum();
            self.statistics.average_performance =
                sum / T::from(performances.len()).expect("conversion from usize to T failed");

            let baseline = self.statistics.average_performance;
            let reward = performances[0] - baseline;

            // Update architecture weights with partial channel gradient
            let lr: T = scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero());
            self.architecture_weights = &self.architecture_weights
                + &(Array3::ones(self.architecture_weights.raw_dim()) * lr * reward);

            // Update edge normalization weights
            if self.edge_normalization {
                let edge_lr: T =
                    scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero());
                self.edge_weights = &self.edge_weights
                    + &(Array1::ones(self.edge_weights.len()) * edge_lr * reward);
            }

            // Anneal temperature
            self.temperature = self.temperature
                * scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero());
        }

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
        }
    }

    /// Compute perturbation regularization loss
    ///
    /// Measures sensitivity of the architecture to weight perturbations.
    /// High sensitivity indicates potential performance collapse.
    fn compute_perturbation_loss(&self, weights: &Array3<T>) -> T {
        let perturbation = Array3::from_shape_fn(weights.raw_dim(), |_| {
            let noise = Random::default().random::<f64>();
            self.perturbation_strength
                * T::from(noise * 2.0 - 1.0).expect("perturbation noise conversion failed")
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
        let mean = self.architecture_weights.sum()
            / T::from(num_elements).expect("conversion from usize to T failed");
        let variance = self
            .architecture_weights
            .mapv(|x| (x - mean) * (x - mean))
            .sum()
            / T::from(num_elements).expect("conversion from usize to T failed");
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
            T::from(-(-u.ln()).ln()).expect("gumbel noise conversion failed")
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
                    .expect("weight initialization conversion failed")
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

        let performances: Vec<T> = results
            .iter()
            .filter_map(|r| {
                r.evaluation_results
                    .metric_scores
                    .get(&EvaluationMetric::FinalPerformance)
            })
            .cloned()
            .collect();

        if !performances.is_empty() {
            let current_best = performances
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(T::zero());

            self.statistics.best_performance = current_best;

            let sum: T = performances.iter().cloned().sum();
            self.statistics.average_performance =
                sum / T::from(performances.len()).expect("conversion from usize to T failed");

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

            // Compute perturbation regularization
            let perturbation_loss = self.compute_perturbation_loss(&self.architecture_weights);

            let baseline = self.statistics.average_performance;
            let reward = performances[0] - baseline;

            // Update with regularized gradient
            let lr: T = scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero());
            let regularized_reward = reward - self.regularization_weight * perturbation_loss;

            self.architecture_weights = &self.architecture_weights
                + &(Array3::ones(self.architecture_weights.raw_dim()) * lr * regularized_reward);

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
            _learningrate: learningrate,
            momentum: scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero()),
            weight_decay: scirs2_core::numeric::NumCast::from(1e-4).unwrap_or_else(|| T::zero()),
            velocity: Array3::zeros((0, 0, 0)),
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

/// Discretize continuous architecture weights into a concrete architecture
fn discretize_architecture<
    T: Float + Debug + Default + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand,
>(
    weights: &Array3<T>,
    strategy: &DiscretizationStrategy,
) -> OptimizerArchitecture<T> {
    use crate::architecture::OptimizerComponent;

    let mut components = Vec::new();

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
                // Gradually sharpen the distribution
                let sharpened = edge_weights.mapv(|x| {
                    x.powf(scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero()))
                });
                sharpened
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        };

        // Map operation index to component type (simplified)
        let component_type = match selected_op_idx {
            0 => ComponentType::SGD,
            1 => ComponentType::Adam,
            2 => ComponentType::AdaGrad,
            3 => ComponentType::RMSprop,
            _ => ComponentType::Adam,
        };

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
        metadata: HashMap::new(),
        architecture_id: format!("arch_{}", Random::default().random::<u64>()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
