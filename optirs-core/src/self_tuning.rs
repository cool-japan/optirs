// Self-tuning optimization strategies
//
// This module provides adaptive optimization strategies that automatically
// tune hyperparameters, select optimizers, and adjust configurations based
// on training dynamics and problem characteristics.

use crate::error::{OptimError, Result};
use crate::optimizers::*;
use crate::utils::{scalar_or, total_order, try_f64, try_scalar};
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::thread_rng;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Configuration for self-tuning optimization
#[derive(Debug, Clone)]
pub struct SelfTuningConfig {
    /// Window size for performance evaluation
    pub evaluation_window: usize,

    /// Minimum improvement threshold for parameter updates
    pub improvement_threshold: f64,

    /// Maximum number of optimizer switches per epoch
    pub max_switches_per_epoch: usize,

    /// Enable automatic learning rate adjustment
    pub auto_lr_adjustment: bool,

    /// Enable automatic optimizer selection
    pub auto_optimizer_selection: bool,

    /// Enable automatic batch size tuning
    pub auto_batch_size_tuning: bool,

    /// Warmup period before starting adaptations
    pub warmup_steps: usize,

    /// Exploration probability for optimizer selection
    pub exploration_rate: f64,

    /// Decay rate for exploration
    pub exploration_decay: f64,

    /// Performance metric to optimize
    pub target_metric: TargetMetric,

    /// Minimum wall-clock time that must pass between two optimizer switches.
    ///
    /// Switching optimizers throws away momentum/accumulator state that has
    /// not transferred, so back-to-back switches can cost more than they gain.
    /// `Duration::ZERO` disables the throttle and restores step-count-only
    /// gating.
    pub min_adaptation_interval: Duration,
}

impl Default for SelfTuningConfig {
    fn default() -> Self {
        Self {
            evaluation_window: 100,
            improvement_threshold: 0.01,
            max_switches_per_epoch: 3,
            auto_lr_adjustment: true,
            auto_optimizer_selection: true,
            auto_batch_size_tuning: false,
            warmup_steps: 1000,
            exploration_rate: 0.1,
            exploration_decay: 0.99,
            target_metric: TargetMetric::Loss,
            min_adaptation_interval: Duration::from_secs(1),
        }
    }
}

/// Target optimization metric
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetMetric {
    /// Minimize loss
    Loss,
    /// Maximize accuracy
    Accuracy,
    /// Minimize convergence time
    ConvergenceTime,
    /// Maximize training throughput
    Throughput,
    /// Custom metric (user-defined)
    Custom,
}

/// Performance statistics for tracking optimization progress
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Current loss value
    pub loss: f64,

    /// Current accuracy (if available)
    pub accuracy: Option<f64>,

    /// Gradient norm
    pub gradient_norm: f64,

    /// Training throughput (samples/second)
    pub throughput: f64,

    /// Memory usage (MB)
    pub memory_usage: f64,

    /// Wall clock time for this step
    pub step_time: Duration,

    /// Learning rate used
    pub learning_rate: f64,

    /// Optimizer type used
    pub optimizer_type: String,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Adaptive optimizer that automatically tunes hyperparameters
pub struct SelfTuningOptimizer<A: Float, D: Dimension> {
    /// Configuration
    config: SelfTuningConfig,

    /// Current active optimizer
    current_optimizer: Box<dyn OptimizerTrait<A, D>>,

    /// Available optimizer candidates
    optimizer_candidates: Vec<OptimizerCandidate<A, D>>,

    /// Performance history
    performance_history: VecDeque<PerformanceStats>,

    /// Hyperparameter search state
    search_state: HyperparameterSearchState,

    /// Optimizer selection strategy
    selection_strategy: OptimizerSelectionStrategy,

    /// Index into `optimizer_candidates` of the optimizer currently running.
    ///
    /// Without it there was no way to attribute an observed performance back
    /// to the candidate that produced it, which is why every candidate's
    /// `average_reward` stayed pinned at its initial `0.0`.
    current_candidate_idx: usize,

    /// Current step count
    step_count: usize,

    /// Number of optimizer switches in current epoch
    switches_this_epoch: usize,

    /// Best performance seen so far
    best_performance: Option<f64>,

    /// Time of last adaptation
    last_adaptation_time: Instant,

    /// Exploration state for multi-armed bandit
    bandit_state: BanditState,
}

/// Optimizer candidate with its configuration
struct OptimizerCandidate<A: Float, D: Dimension> {
    /// Name/identifier
    name: String,

    /// Factory function to create the optimizer
    factory: Box<dyn Fn() -> Box<dyn OptimizerTrait<A, D>>>,

    /// Rewards observed while this candidate was the active optimizer, most
    /// recent last and bounded by the configured evaluation window.
    performance_history: Vec<f64>,

    /// Usage count
    usage_count: usize,

    /// Mean of `performance_history`, in *reward* orientation: always
    /// higher-is-better, with loss-like target metrics negated. Selection
    /// maximises this, so storing the raw metric would make the tuner prefer
    /// the *worst* optimizer whenever the target metric is a loss.
    average_reward: f64,

    /// 95% confidence interval around `average_reward`, as
    /// `(lower, upper)`. Width feeds the bandit's exploration bonus.
    confidence_interval: (f64, f64),
}

/// Hyperparameter search state
#[derive(Debug)]
struct HyperparameterSearchState {
    /// Current learning rate
    learning_rate: f64,

    /// Learning rate search bounds
    lr_bounds: (f64, f64),

    /// Current batch size
    batch_size: usize,

    /// Batch size search bounds
    batch_size_bounds: (usize, usize),

    /// Number of search iterations (reported optimization steps) folded into
    /// the search state so far.
    search_iterations: usize,

    /// Target-metric values observed for the configurations tried so far, most
    /// recent last, bounded to a few evaluation windows.
    observed_metrics: Vec<f64>,

    /// Best hyperparameters found: the learning rate and batch size in force
    /// when the best target-metric value so far was observed.
    best_hyperparameters: HashMap<String, f64>,
}

/// Public snapshot of the hyperparameter-search state.
#[derive(Debug, Clone)]
pub struct HyperparameterSearchSummary {
    /// Reported optimization steps folded into the search state.
    pub search_iterations: usize,
    /// Number of retained target-metric observations.
    pub observations: usize,
    /// Learning rate currently in force.
    pub learning_rate: f64,
    /// Inclusive learning-rate search bounds.
    pub lr_bounds: (f64, f64),
    /// Batch size currently in force.
    pub batch_size: usize,
    /// Inclusive batch-size search bounds.
    pub batch_size_bounds: (usize, usize),
    /// Best configuration observed so far, keyed by hyperparameter name plus a
    /// `target_metric` entry holding the metric value it achieved.
    pub best_hyperparameters: HashMap<String, f64>,
}

/// Optimizer selection strategies.
///
/// Every variant below has a working implementation in
/// [`SelfTuningOptimizer`]; this enum is public so a caller can actually
/// choose between them via
/// [`SelfTuningOptimizer::set_selection_strategy`]. It used to be private
/// with the constructor hard-coding `MultiArmedBandit { UCB1 }`, which left
/// three fully-implemented strategies unreachable.
#[derive(Debug, Clone)]
pub enum OptimizerSelectionStrategy {
    /// Multi-armed bandit approach
    MultiArmedBandit {
        /// Bandit algorithm type
        algorithm: BanditAlgorithm,
    },

    /// Performance-based selection
    PerformanceBased {
        /// Minimum performance difference for switching
        min_difference: f64,
    },

    /// Round-robin testing
    RoundRobin {
        /// Current optimizer index
        current_index: usize,
    },

    /// Meta-learning based selection
    MetaLearning {
        /// Problem characteristics
        problem_features: Vec<f64>,
        /// Learned optimizer mappings
        optimizer_mappings: HashMap<String, f64>,
    },
}

/// Multi-armed bandit algorithms.
///
/// All four are implemented by `select_optimizer_bandit`; public so
/// [`OptimizerSelectionStrategy::MultiArmedBandit`] can be built with any of
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BanditAlgorithm {
    /// Explore uniformly at random with probability `exploration_rate`,
    /// otherwise take the current best reward estimate.
    EpsilonGreedy,
    /// Deterministic upper-confidence-bound selection (Auer et al. 2002).
    UCB1,
    /// Sample each arm from its estimated reward interval and take the best
    /// draw.
    ThompsonSampling,
    /// Contextual UCB variant.
    LinUCB,
}

/// Multi-armed bandit state
#[derive(Debug)]
struct BanditState {
    /// Reward estimates for each optimizer
    reward_estimates: Vec<f64>,

    /// Confidence bounds
    confidence_bounds: Vec<f64>,

    /// Selection counts
    selection_counts: Vec<usize>,

    /// Total selections
    total_selections: usize,

    /// Exploration parameter
    exploration_param: f64,
}

/// Trait for optimizer implementations that can be used with self-tuning
pub trait OptimizerTrait<A: Float + ScalarOperand + Debug, D: Dimension>: Send + Sync {
    /// Get optimizer name
    fn name(&self) -> &str;

    /// Perform optimization step
    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()>;

    /// Get current learning rate
    fn learning_rate(&self) -> A;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: A);

    /// Get optimizer state for serialization
    fn get_state(&self) -> HashMap<String, Vec<u8>>;

    /// Set optimizer state from serialization
    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()>;

    /// Clone the optimizer
    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>>;
}

impl<
        A: Float + ScalarOperand + Debug + Send + Sync + 'static + scirs2_core::numeric::FromPrimitive,
        D: Dimension + 'static,
    > SelfTuningOptimizer<A, D>
{
    /// Create new self-tuning optimizer
    pub fn new(config: SelfTuningConfig) -> Result<Self> {
        let mut optimizer_candidates = Vec::new();

        // Add default optimizer candidates
        optimizer_candidates.push(OptimizerCandidate {
            name: "Adam".to_string(),
            factory: Box::new(|| Box::new(AdamOptimizerWrapper::new(0.001, 0.9, 0.999, 1e-8, 0.0))),
            performance_history: Vec::new(),
            usage_count: 0,
            average_reward: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        optimizer_candidates.push(OptimizerCandidate {
            name: "SGD".to_string(),
            factory: Box::new(|| Box::new(SGDOptimizerWrapper::new(0.01, 0.9, 0.0))),
            performance_history: Vec::new(),
            usage_count: 0,
            average_reward: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        optimizer_candidates.push(OptimizerCandidate {
            name: "AdamW".to_string(),
            factory: Box::new(|| {
                Box::new(AdamWOptimizerWrapper::new(0.001, 0.9, 0.999, 1e-8, 0.01))
            }),
            performance_history: Vec::new(),
            usage_count: 0,
            average_reward: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        // Start with Adam as default
        let current_optimizer = (optimizer_candidates[0].factory)();

        let search_state = HyperparameterSearchState {
            learning_rate: 0.001,
            lr_bounds: (1e-6, 1.0),
            batch_size: 32,
            batch_size_bounds: (8, 512),
            search_iterations: 0,
            observed_metrics: Vec::new(),
            best_hyperparameters: HashMap::new(),
        };

        let selection_strategy = OptimizerSelectionStrategy::MultiArmedBandit {
            algorithm: BanditAlgorithm::UCB1,
        };

        let bandit_state = BanditState {
            reward_estimates: vec![0.0; optimizer_candidates.len()],
            confidence_bounds: vec![1.0; optimizer_candidates.len()],
            selection_counts: vec![0; optimizer_candidates.len()],
            total_selections: 0,
            exploration_param: 2.0,
        };

        Ok(Self {
            config,
            current_optimizer,
            optimizer_candidates,
            performance_history: VecDeque::new(),
            search_state,
            selection_strategy,
            current_candidate_idx: 0,
            step_count: 0,
            switches_this_epoch: 0,
            best_performance: None,
            last_adaptation_time: Instant::now(),
            bandit_state,
        })
    }

    /// Add a custom optimizer candidate
    pub fn add_optimizer_candidate<F>(&mut self, name: String, factory: F)
    where
        F: Fn() -> Box<dyn OptimizerTrait<A, D>> + 'static,
    {
        self.optimizer_candidates.push(OptimizerCandidate {
            name,
            factory: Box::new(factory),
            performance_history: Vec::new(),
            usage_count: 0,
            average_reward: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        // Update bandit state
        self.bandit_state.reward_estimates.push(0.0);
        self.bandit_state.confidence_bounds.push(1.0);
        self.bandit_state.selection_counts.push(0);
    }

    /// Choose how the next optimizer is selected.
    ///
    /// All four [`OptimizerSelectionStrategy`] variants are implemented; before
    /// this setter existed the constructor's `MultiArmedBandit { UCB1 }` was
    /// the only reachable one.
    pub fn set_selection_strategy(&mut self, strategy: OptimizerSelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// The strategy currently in force.
    pub fn selection_strategy(&self) -> &OptimizerSelectionStrategy {
        &self.selection_strategy
    }

    /// Perform optimization step with automatic tuning
    pub fn step(
        &mut self,
        params: &mut [Array<A, D>],
        grads: &[Array<A, D>],
        stats: PerformanceStats,
    ) -> Result<()> {
        self.step_count += 1;

        // Record performance
        self.performance_history.push_back(stats.clone());
        if self.performance_history.len() > self.config.evaluation_window {
            self.performance_history.pop_front();
        }

        // Perform optimization step
        self.current_optimizer.step(params, grads)?;

        // Attribute this step's observation to the optimizer that produced it,
        // *before* any adaptation can switch which optimizer is current.
        // Without this every candidate's reward estimate stayed at its initial
        // 0.0, so `PerformanceBased` selection ranked a constant and the bandit
        // chose between identical arms forever; recording it after the switch
        // would credit the incoming optimizer with the outgoing one's result,
        // which is worse than not recording at all -- it would systematically
        // reward whichever optimizer was switched *to* after a bad step.
        self.record_candidate_performance(&stats);

        // Self-tuning adaptations
        if self.step_count > self.config.warmup_steps {
            self.maybe_adapt_optimizer(&stats)?;
            self.maybe_adapt_learning_rate(&stats)?;
            self.maybe_adapt_hyperparameters(&stats)?;
        }

        // Update best performance
        if let Some(performance) = self.extract_performance_metric(&stats) {
            let improved = match self.best_performance {
                None => true,
                Some(best) => self.is_better_performance(performance, best),
            };
            if improved {
                self.best_performance = Some(performance);
            }
        }

        Ok(())
    }

    /// Check if we should adapt the optimizer
    fn maybe_adapt_optimizer(&mut self, stats: &PerformanceStats) -> Result<()> {
        if !self.config.auto_optimizer_selection {
            return Ok(());
        }

        if self.switches_this_epoch >= self.config.max_switches_per_epoch {
            return Ok(());
        }

        // Respect the cool-down since the last switch. `last_adaptation_time`
        // was recorded but never consulted, so the only limit on switching was
        // the per-epoch count.
        if self.switches_this_epoch > 0
            && self.last_adaptation_time.elapsed() < self.config.min_adaptation_interval
        {
            return Ok(());
        }

        let should_adapt = self.should_adapt_optimizer(stats);

        if should_adapt {
            self.adapt_optimizer(stats)?;
            self.switches_this_epoch += 1;
        }

        Ok(())
    }

    /// Determine if optimizer should be adapted
    fn should_adapt_optimizer(&self, stats: &PerformanceStats) -> bool {
        if self.performance_history.len() < self.config.evaluation_window / 2 {
            return false;
        }

        // Check for performance degradation or stagnation. The freshest
        // observation is the `stats` just reported, which is not yet in
        // `performance_history`; including it is what makes this decision react
        // to the current step rather than lagging a full window behind it.
        let mut recent_performance: Vec<f64> = self
            .performance_history
            .iter()
            .rev()
            .take(self.config.evaluation_window / 4)
            .filter_map(|s| self.extract_performance_metric(s))
            .collect();
        if let Some(current) = self.extract_performance_metric(stats) {
            recent_performance.insert(0, current);
        }

        let older_performance: Vec<f64> = self
            .performance_history
            .iter()
            .rev()
            .skip(self.config.evaluation_window / 4)
            .take(self.config.evaluation_window / 4)
            .filter_map(|s| self.extract_performance_metric(s))
            .collect();

        if recent_performance.is_empty() || older_performance.is_empty() {
            return false;
        }

        let recent_avg = recent_performance.iter().sum::<f64>() / recent_performance.len() as f64;
        let older_avg = older_performance.iter().sum::<f64>() / older_performance.len() as f64;

        // Check for stagnation or degradation
        match self.config.target_metric {
            TargetMetric::Loss => {
                (recent_avg - older_avg).abs() < self.config.improvement_threshold
                    || recent_avg > older_avg
            }
            TargetMetric::Accuracy | TargetMetric::Throughput => {
                (recent_avg - older_avg).abs() < self.config.improvement_threshold
                    || recent_avg < older_avg
            }
            _ => false,
        }
    }

    /// Adapt the optimizer based on performance
    fn adapt_optimizer(&mut self, stats: &PerformanceStats) -> Result<()> {
        let new_optimizer_idx = match &self.selection_strategy {
            OptimizerSelectionStrategy::MultiArmedBandit { algorithm } => {
                self.select_optimizer_bandit(*algorithm)
            }
            OptimizerSelectionStrategy::PerformanceBased { .. } => {
                self.select_optimizer_performance_based()
            }
            // Advance from the optimizer that is actually running. The
            // strategy's own `current_index` was never written back, so this
            // used to return the same successor on every call and round-robin
            // never got past the second candidate.
            OptimizerSelectionStrategy::RoundRobin { .. } => {
                (self.current_candidate_idx + 1) % self.optimizer_candidates.len()
            }
            OptimizerSelectionStrategy::MetaLearning { .. } => {
                self.select_optimizer_meta_learning(stats)
            }
        };

        // Switch to new optimizer
        if new_optimizer_idx < self.optimizer_candidates.len() {
            let current_lr = self.current_optimizer.learning_rate();
            let current_state = self.current_optimizer.get_state();

            self.current_optimizer = (self.optimizer_candidates[new_optimizer_idx].factory)();
            self.current_optimizer.set_learning_rate(current_lr);

            // Try to transfer compatible state
            if self.current_optimizer.set_state(current_state).is_err() {
                // State transfer failed, continue with fresh state
            }

            // Update usage statistics
            self.optimizer_candidates[new_optimizer_idx].usage_count += 1;
            self.current_candidate_idx = new_optimizer_idx;
            self.last_adaptation_time = Instant::now();
            // Keep the strategy's own cursor in step with reality so
            // `selection_strategy()` reports where the rotation actually is.
            if let OptimizerSelectionStrategy::RoundRobin { current_index } =
                &mut self.selection_strategy
            {
                *current_index = new_optimizer_idx;
            }
        }

        Ok(())
    }

    /// Select optimizer using multi-armed bandit
    fn select_optimizer_bandit(&mut self, algorithm: BanditAlgorithm) -> usize {
        match algorithm {
            BanditAlgorithm::UCB1 => self.select_ucb1(),
            BanditAlgorithm::EpsilonGreedy => self.select_epsilon_greedy(),
            BanditAlgorithm::ThompsonSampling => self.select_thompson_sampling(),
            BanditAlgorithm::LinUCB => self.select_linucb(),
        }
    }

    /// UCB1 optimizer selection
    fn select_ucb1(&self) -> usize {
        if self.bandit_state.total_selections == 0 {
            return 0;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for i in 0..self.optimizer_candidates.len() {
            let ucb_score = if self.bandit_state.selection_counts[i] == 0 {
                f64::INFINITY
            } else {
                let mean_reward = self.bandit_state.reward_estimates[i];
                let confidence = (self.bandit_state.exploration_param
                    * (self.bandit_state.total_selections as f64).ln()
                    / self.bandit_state.selection_counts[i] as f64)
                    .sqrt();
                mean_reward + confidence
            };

            if ucb_score > best_score {
                best_score = ucb_score;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Epsilon-greedy optimizer selection
    fn select_epsilon_greedy(&self) -> usize {
        let mut rng = thread_rng();

        if scalar_or(rng.random::<f64>(), A::zero())
            < scalar_or(self.config.exploration_rate, A::zero())
        {
            // Explore: random selection
            rng.gen_range(0..self.optimizer_candidates.len())
        } else {
            // Exploit: best performing optimizer
            self.bandit_state
                .reward_estimates
                .iter()
                .enumerate()
                .max_by(|a, b| total_order(a.1, b.1))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Thompson sampling optimizer selection
    fn select_thompson_sampling(&self) -> usize {
        // Simplified Thompson sampling - in practice would use Beta distributions
        let mut rng = thread_rng();

        let mut best_sample = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, _) in self.optimizer_candidates.iter().enumerate() {
            let mean = self.bandit_state.reward_estimates[i];
            let std = self.bandit_state.confidence_bounds[i];
            let sample = rng.gen_range(mean - std..mean + std);

            if sample > best_sample {
                best_sample = sample;
                best_idx = i;
            }
        }

        best_idx
    }

    /// LinUCB optimizer selection (contextual bandit)
    fn select_linucb(&self) -> usize {
        // Simplified LinUCB - would use contextual features in practice
        self.select_ucb1()
    }

    /// Performance-based optimizer selection
    fn select_optimizer_performance_based(&self) -> usize {
        self.optimizer_candidates
            .iter()
            .enumerate()
            .max_by(|a, b| total_order(&a.1.average_reward, &b.1.average_reward))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Feature vector describing the current optimization problem, in the same
    /// order as `OptimizerSelectionStrategy::MetaLearning::problem_features`:
    /// `[loss, gradient_norm, throughput, memory_usage, learning_rate]`.
    fn problem_feature_vector(stats: &PerformanceStats) -> [f64; 5] {
        [
            stats.loss,
            stats.gradient_norm,
            stats.throughput,
            stats.memory_usage,
            stats.learning_rate,
        ]
    }

    /// Meta-learning based optimizer selection.
    ///
    /// The strategy carries a learned `optimizer_mappings` table (optimizer name
    /// to expected quality) that was fitted on a problem described by
    /// `problem_features`. That table is only trustworthy for problems that
    /// resemble the one it was fitted on, so the current problem's features are
    /// compared to the stored ones by cosine similarity and the learned ranking
    /// is used only above a similarity threshold; otherwise selection falls back
    /// to the measured `average_reward` of each candidate.
    ///
    /// This replaces a stub that ignored `stats` and unconditionally returned
    /// candidate 0, which made `MetaLearning` silently equivalent to "never
    /// switch away from the first optimizer".
    fn select_optimizer_meta_learning(&self, stats: &PerformanceStats) -> usize {
        /// Minimum cosine similarity between the current problem and the one the
        /// mappings were learned on before the learned ranking is trusted.
        const SIMILARITY_THRESHOLD: f64 = 0.9;

        let OptimizerSelectionStrategy::MetaLearning {
            problem_features,
            optimizer_mappings,
        } = &self.selection_strategy
        else {
            return self.select_optimizer_performance_based();
        };
        if optimizer_mappings.is_empty() {
            return self.select_optimizer_performance_based();
        }

        let current = Self::problem_feature_vector(stats);
        let shared = problem_features.len().min(current.len());
        let (mut dot, mut norm_stored, mut norm_current) = (0.0, 0.0, 0.0);
        for i in 0..shared {
            dot += problem_features[i] * current[i];
            norm_stored += problem_features[i] * problem_features[i];
            norm_current += current[i] * current[i];
        }
        let similarity = if norm_stored > 0.0 && norm_current > 0.0 {
            dot / (norm_stored.sqrt() * norm_current.sqrt())
        } else {
            // No usable feature vector on one side: the mappings cannot be
            // shown to apply here.
            0.0
        };
        if similarity < SIMILARITY_THRESHOLD {
            return self.select_optimizer_performance_based();
        }

        self.optimizer_candidates
            .iter()
            .enumerate()
            .filter_map(|(idx, candidate)| {
                optimizer_mappings
                    .get(&candidate.name)
                    .map(|score| (idx, *score))
            })
            .max_by(|a, b| total_order(&a.1, &b.1))
            .map(|(idx, _)| idx)
            .unwrap_or_else(|| self.select_optimizer_performance_based())
    }

    /// Adapt learning rate based on performance
    fn maybe_adapt_learning_rate(&mut self, stats: &PerformanceStats) -> Result<()> {
        if !self.config.auto_lr_adjustment {
            return Ok(());
        }

        // Simple adaptive learning rate based on gradient norm
        let current_lr = try_f64(self.current_optimizer.learning_rate())?;
        let gradient_norm = stats.gradient_norm;

        let new_lr = if gradient_norm > 10.0 {
            // Large gradients - reduce learning rate
            current_lr * 0.9
        } else if gradient_norm < 0.1 {
            // Small gradients - increase learning rate
            current_lr * 1.1
        } else {
            current_lr
        };

        let clamped_lr = new_lr
            .max(self.search_state.lr_bounds.0)
            .min(self.search_state.lr_bounds.1);

        if (clamped_lr - current_lr).abs() > current_lr * 0.01 {
            self.current_optimizer
                .set_learning_rate(try_scalar::<A, _>(clamped_lr)?);
            self.search_state.learning_rate = clamped_lr;
        }

        Ok(())
    }

    /// Record the reported statistics against the active hyperparameter search
    /// trial.
    ///
    /// Only the *observation* half of hyperparameter search is implemented here:
    /// the metric for the current configuration is appended to the search
    /// state's history so a search driver has real data to work from. Proposing
    /// the next configuration (Bayesian optimization / grid / successive
    /// halving) is not implemented — see `HyperparameterSearchState` and the
    /// unimplemented `SearchStrategy` variants — so nothing is mutated behind
    /// the caller's back and no configuration change is fabricated.
    fn maybe_adapt_hyperparameters(&mut self, stats: &PerformanceStats) -> Result<()> {
        let Some(metric) = self.extract_performance_metric(stats) else {
            return Ok(());
        };

        let improved = match self.best_observed_metric() {
            Some(best) => self.metric_is_better(metric, best),
            None => true,
        };

        self.search_state.observed_metrics.push(metric);
        let cap = self.config.evaluation_window.max(1) * 4;
        if self.search_state.observed_metrics.len() > cap {
            self.search_state.observed_metrics.remove(0);
        }
        self.search_state.search_iterations += 1;

        if improved {
            self.search_state
                .best_hyperparameters
                .insert("learning_rate".to_string(), self.search_state.learning_rate);
            self.search_state.best_hyperparameters.insert(
                "batch_size".to_string(),
                self.search_state.batch_size as f64,
            );
            self.search_state
                .best_hyperparameters
                .insert("target_metric".to_string(), metric);
        }

        Ok(())
    }

    /// Whether `candidate` is a better target-metric value than `incumbent`,
    /// respecting the configured metric's direction.
    fn metric_is_better(&self, candidate: f64, incumbent: f64) -> bool {
        match self.config.target_metric {
            TargetMetric::Loss => candidate < incumbent,
            _ => candidate > incumbent,
        }
    }

    /// Best target-metric value recorded by the hyperparameter search so far.
    fn best_observed_metric(&self) -> Option<f64> {
        self.search_state
            .best_hyperparameters
            .get("target_metric")
            .copied()
    }

    /// Snapshot of the hyperparameter-search state: the number of reported
    /// steps folded in, the best configuration observed, and the batch-size
    /// search bounds the (not yet implemented) proposal step would respect.
    pub fn hyperparameter_search_summary(&self) -> HyperparameterSearchSummary {
        HyperparameterSearchSummary {
            search_iterations: self.search_state.search_iterations,
            observations: self.search_state.observed_metrics.len(),
            learning_rate: self.search_state.learning_rate,
            lr_bounds: self.search_state.lr_bounds,
            batch_size: self.search_state.batch_size,
            batch_size_bounds: self.search_state.batch_size_bounds,
            best_hyperparameters: self.search_state.best_hyperparameters.clone(),
        }
    }

    /// Extract performance metric from stats
    fn extract_performance_metric(&self, stats: &PerformanceStats) -> Option<f64> {
        match self.config.target_metric {
            TargetMetric::Loss => Some(stats.loss),
            TargetMetric::Accuracy => stats.accuracy,
            TargetMetric::Throughput => Some(stats.throughput),
            TargetMetric::ConvergenceTime => Some(stats.step_time.as_secs_f64()),
            TargetMetric::Custom => stats.custom_metrics.values().next().copied(),
        }
    }

    /// Whether the target metric is one where a *smaller* value is better.
    fn lower_is_better(&self) -> bool {
        matches!(
            self.config.target_metric,
            TargetMetric::Loss | TargetMetric::ConvergenceTime
        )
    }

    /// The observed metric expressed as a reward: always higher-is-better, so
    /// that every selection rule can simply maximise it.
    fn reward_from_metric(&self, metric: f64) -> f64 {
        if self.lower_is_better() {
            -metric
        } else {
            metric
        }
    }

    /// Fold this step's observation into the active candidate's statistics and
    /// the bandit's estimate of that arm.
    ///
    /// This is the update that makes `performance_history`, `average_reward`,
    /// `confidence_interval`, `reward_estimates` and `confidence_bounds`
    /// carry real measurements instead of their initial constants.
    fn record_candidate_performance(&mut self, stats: &PerformanceStats) {
        let Some(metric) = self.extract_performance_metric(stats) else {
            return;
        };
        if !metric.is_finite() {
            return;
        }
        let reward = self.reward_from_metric(metric);

        let window = self.config.evaluation_window.max(1);
        let idx = self.current_candidate_idx;
        let Some(candidate) = self.optimizer_candidates.get_mut(idx) else {
            return;
        };

        candidate.performance_history.push(reward);
        if candidate.performance_history.len() > window {
            let excess = candidate.performance_history.len() - window;
            candidate.performance_history.drain(0..excess);
        }

        let samples = candidate.performance_history.len();
        let count = samples as f64;
        let mean = candidate.performance_history.iter().sum::<f64>() / count;
        // Sample standard error; a single observation has no spread to report,
        // so its interval is a point.
        let half_width = if samples > 1 {
            let variance = candidate
                .performance_history
                .iter()
                .map(|&r| (r - mean) * (r - mean))
                .sum::<f64>()
                / (count - 1.0);
            1.96 * (variance / count).sqrt()
        } else {
            0.0
        };

        candidate.average_reward = mean;
        candidate.confidence_interval = (mean - half_width, mean + half_width);

        // Mirror into the bandit arms, which select on exactly these numbers.
        if let Some(estimate) = self.bandit_state.reward_estimates.get_mut(idx) {
            *estimate = mean;
        }
        if let Some(bound) = self.bandit_state.confidence_bounds.get_mut(idx) {
            // A never-measured arm keeps its optimistic initial bound so it
            // still gets explored; a measured one reports its real spread.
            *bound = if samples > 1 { half_width } else { 1.0 };
        }
    }

    /// Check if performance is better
    fn is_better_performance(&self, new_perf: f64, oldperf: f64) -> bool {
        match self.config.target_metric {
            TargetMetric::Loss | TargetMetric::ConvergenceTime => new_perf < oldperf,
            TargetMetric::Accuracy | TargetMetric::Throughput => new_perf > oldperf,
            TargetMetric::Custom => new_perf > oldperf, // Assume higher is better for custom
        }
    }

    /// Reset epoch counters
    pub fn reset_epoch(&mut self) {
        self.switches_this_epoch = 0;
    }

    /// Get current optimizer information
    pub fn get_optimizer_info(&self) -> OptimizerInfo {
        OptimizerInfo {
            name: self.current_optimizer.name().to_string(),
            // A learning rate with no `f64` image cannot be reported; `NaN`
            // marks it as unavailable rather than panicking an info getter.
            learning_rate: try_f64(self.current_optimizer.learning_rate()).unwrap_or(f64::NAN),
            step_count: self.step_count,
            switches_this_epoch: self.switches_this_epoch,
            performance_window_size: self.performance_history.len(),
            best_performance: self.best_performance,
        }
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> SelfTuningStatistics {
        let optimizer_usage: HashMap<String, usize> = self
            .optimizer_candidates
            .iter()
            .map(|c| (c.name.clone(), c.usage_count))
            .collect();

        SelfTuningStatistics {
            total_steps: self.step_count,
            total_optimizer_switches: self
                .optimizer_candidates
                .iter()
                .map(|c| c.usage_count)
                .sum(),
            optimizer_usage,
            current_learning_rate: self.search_state.learning_rate,
            average_step_time: self
                .performance_history
                .iter()
                .map(|s| s.step_time.as_secs_f64())
                .sum::<f64>()
                / self.performance_history.len().max(1) as f64,
            exploration_rate: self.config.exploration_rate,
        }
    }
}

/// Information about current optimizer state
#[derive(Debug, Clone)]
pub struct OptimizerInfo {
    pub name: String,
    pub learning_rate: f64,
    pub step_count: usize,
    pub switches_this_epoch: usize,
    pub performance_window_size: usize,
    pub best_performance: Option<f64>,
}

/// Statistics about self-tuning optimization
#[derive(Debug, Clone)]
pub struct SelfTuningStatistics {
    pub total_steps: usize,
    pub total_optimizer_switches: usize,
    pub optimizer_usage: HashMap<String, usize>,
    pub current_learning_rate: f64,
    pub average_step_time: f64,
    pub exploration_rate: f64,
}

// Wrapper implementations for existing optimizers
struct AdamOptimizerWrapper<A: Float + ScalarOperand + Debug, D: Dimension> {
    inner: crate::optimizers::Adam<A>,
    _phantom: std::marker::PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    AdamOptimizerWrapper<A, D>
{
    fn new(_lr: f64, beta1: f64, beta2: f64, eps: f64, weightdecay: f64) -> Self {
        Self {
            inner: crate::optimizers::Adam::new_with_config(
                scalar_or(_lr, A::zero()),
                scalar_or(beta1, A::zero()),
                scalar_or(beta2, A::zero()),
                scalar_or(eps, A::zero()),
                scalar_or(weightdecay, A::zero()),
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static, D: Dimension + 'static>
    OptimizerTrait<A, D> for AdamOptimizerWrapper<A, D>
{
    fn name(&self) -> &str {
        "Adam"
    }

    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::OptimError::InvalidParameter(
                "Mismatched number of parameters and gradients".to_string(),
            ));
        }

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let updated = self.inner.step(param, grad)?;
            *param = updated;
        }
        Ok(())
    }

    fn learning_rate(&self) -> A {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: A) {
        <crate::optimizers::Adam<A> as crate::optimizers::Optimizer<A, D>>::set_learning_rate(
            &mut self.inner,
            lr,
        );
    }

    /// Returns an empty map: the wrapped optimizer does not expose its internal
    /// moment/accumulator arrays, so there is genuinely nothing to serialize.
    /// This is reported honestly rather than emitting a partial snapshot that
    /// would silently lose state on restore.
    fn get_state(&self) -> HashMap<String, Vec<u8>> {
        HashMap::new()
    }

    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()> {
        if state.is_empty() {
            return Ok(());
        }
        Err(OptimError::UnsupportedOperation(format!(
            "{} does not expose serializable moment state, so a {}-entry state \
             snapshot cannot be restored; the optimizer starts from a fresh state",
            self.name(),
            state.len()
        )))
    }

    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>> {
        Box::new(AdamOptimizerWrapper {
            inner: self.inner.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

struct SGDOptimizerWrapper<A: Float + ScalarOperand + Debug, D: Dimension> {
    inner: crate::optimizers::SGD<A>,
    _phantom: std::marker::PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    SGDOptimizerWrapper<A, D>
{
    /// Build an SGD wrapper.
    ///
    /// There is deliberately no `nesterov` parameter: `crate::optimizers::SGD`
    /// implements classical (heavy-ball) momentum only, with no Nesterov
    /// look-ahead term, so a flag here could not be honoured. It previously
    /// accepted `nesterov: bool` and discarded it, which advertised support
    /// that does not exist.
    fn new(lr: f64, momentum: f64, weightdecay: f64) -> Self {
        Self {
            inner: crate::optimizers::SGD::new_with_config(
                scalar_or(lr, A::zero()),
                scalar_or(momentum, A::zero()),
                scalar_or(weightdecay, A::zero()),
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static, D: Dimension + 'static>
    OptimizerTrait<A, D> for SGDOptimizerWrapper<A, D>
{
    fn name(&self) -> &str {
        "SGD"
    }

    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::OptimError::InvalidParameter(
                "Mismatched number of parameters and gradients".to_string(),
            ));
        }

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let updated = self.inner.step(param, grad)?;
            *param = updated;
        }
        Ok(())
    }

    fn learning_rate(&self) -> A {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: A) {
        <crate::optimizers::SGD<A> as crate::optimizers::Optimizer<A, D>>::set_learning_rate(
            &mut self.inner,
            lr,
        );
    }

    /// Returns an empty map: the wrapped optimizer does not expose its internal
    /// moment/accumulator arrays, so there is genuinely nothing to serialize.
    /// This is reported honestly rather than emitting a partial snapshot that
    /// would silently lose state on restore.
    fn get_state(&self) -> HashMap<String, Vec<u8>> {
        HashMap::new()
    }

    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()> {
        if state.is_empty() {
            return Ok(());
        }
        Err(OptimError::UnsupportedOperation(format!(
            "{} does not expose serializable moment state, so a {}-entry state \
             snapshot cannot be restored; the optimizer starts from a fresh state",
            self.name(),
            state.len()
        )))
    }

    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>> {
        Box::new(SGDOptimizerWrapper {
            inner: self.inner.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

struct AdamWOptimizerWrapper<A: Float + ScalarOperand + Debug, D: Dimension> {
    inner: crate::optimizers::AdamW<A>,
    _phantom: std::marker::PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    AdamWOptimizerWrapper<A, D>
{
    fn new(_lr: f64, beta1: f64, beta2: f64, eps: f64, weightdecay: f64) -> Self {
        Self {
            inner: crate::optimizers::AdamW::new_with_config(
                scalar_or(_lr, A::zero()),
                scalar_or(beta1, A::zero()),
                scalar_or(beta2, A::zero()),
                scalar_or(eps, A::zero()),
                scalar_or(weightdecay, A::zero()),
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static, D: Dimension + 'static>
    OptimizerTrait<A, D> for AdamWOptimizerWrapper<A, D>
{
    fn name(&self) -> &str {
        "AdamW"
    }

    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::OptimError::InvalidParameter(
                "Mismatched number of parameters and gradients".to_string(),
            ));
        }

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let updated = self.inner.step(param, grad)?;
            *param = updated;
        }
        Ok(())
    }

    fn learning_rate(&self) -> A {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: A) {
        <crate::optimizers::AdamW<A> as crate::optimizers::Optimizer<A, D>>::set_learning_rate(
            &mut self.inner,
            lr,
        );
    }

    /// Returns an empty map: the wrapped optimizer does not expose its internal
    /// moment/accumulator arrays, so there is genuinely nothing to serialize.
    /// This is reported honestly rather than emitting a partial snapshot that
    /// would silently lose state on restore.
    fn get_state(&self) -> HashMap<String, Vec<u8>> {
        HashMap::new()
    }

    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()> {
        if state.is_empty() {
            return Ok(());
        }
        Err(OptimError::UnsupportedOperation(format!(
            "{} does not expose serializable moment state, so a {}-entry state \
             snapshot cannot be restored; the optimizer starts from a fresh state",
            self.name(),
            state.len()
        )))
    }

    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>> {
        Box::new(AdamWOptimizerWrapper {
            inner: self.inner.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::time::Duration;

    #[test]
    fn test_self_tuning_config_default() {
        let config = SelfTuningConfig::default();
        assert_eq!(config.evaluation_window, 100);
        assert!(config.auto_lr_adjustment);
        assert!(config.auto_optimizer_selection);
    }

    #[test]
    fn test_self_tuning_optimizer_creation() {
        let config = SelfTuningConfig::default();
        let optimizer: Result<SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1>> =
            SelfTuningOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats {
            loss: 0.5,
            accuracy: Some(0.9),
            gradient_norm: 1.2,
            throughput: 100.0,
            memory_usage: 1024.0,
            step_time: Duration::from_millis(50),
            learning_rate: 0.001,
            optimizer_type: "Adam".to_string(),
            custom_metrics: HashMap::new(),
        };

        assert_eq!(stats.loss, 0.5);
        assert_eq!(stats.accuracy, Some(0.9));
    }

    #[test]
    fn test_optimizer_step() {
        let config = SelfTuningConfig::default();
        let mut optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(config).expect("default config must construct");

        let mut params = vec![Array1::zeros(10)];
        let grads = vec![Array1::ones(10)];

        let stats = PerformanceStats {
            loss: 1.0,
            accuracy: None,
            gradient_norm: 1.0,
            throughput: 50.0,
            memory_usage: 512.0,
            step_time: Duration::from_millis(10),
            learning_rate: 0.001,
            optimizer_type: "Adam".to_string(),
            custom_metrics: HashMap::new(),
        };

        let result = optimizer.step(&mut params, &grads, stats);
        assert!(result.is_ok());

        let info = optimizer.get_optimizer_info();
        assert_eq!(info.name, "Adam");
        assert_eq!(info.step_count, 1);
    }

    #[test]
    fn test_bandit_selection() {
        let config = SelfTuningConfig::default();
        let optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(config).expect("default config must construct");

        let selection = optimizer.select_ucb1();
        assert!(selection < optimizer.optimizer_candidates.len());
    }

    #[test]
    fn test_performance_metric_extraction() {
        let config = SelfTuningConfig {
            target_metric: TargetMetric::Loss,
            ..Default::default()
        };
        let optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(config).expect("default config must construct");

        let stats = PerformanceStats {
            loss: 0.8,
            accuracy: Some(0.85),
            gradient_norm: 1.1,
            throughput: 75.0,
            memory_usage: 800.0,
            step_time: Duration::from_millis(20),
            learning_rate: 0.001,
            optimizer_type: "Adam".to_string(),
            custom_metrics: HashMap::new(),
        };

        let metric = optimizer.extract_performance_metric(&stats);
        assert_eq!(metric, Some(0.8));
    }

    #[test]
    fn test_statistics() {
        let config = SelfTuningConfig::default();
        let optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(config).expect("default config must construct");

        let stats = optimizer.get_statistics();
        assert_eq!(stats.total_steps, 0);
        assert!(stats.optimizer_usage.contains_key("Adam"));
    }
    // ------------------------------------------------- candidate accounting --

    fn stats_with_loss(loss: f64) -> PerformanceStats {
        PerformanceStats {
            loss,
            accuracy: None,
            gradient_norm: 1.0,
            throughput: 50.0,
            memory_usage: 512.0,
            step_time: Duration::from_millis(10),
            learning_rate: 0.001,
            optimizer_type: "Adam".to_string(),
            custom_metrics: HashMap::new(),
        }
    }

    /// The active candidate's reward statistics must track the observations
    /// that were actually reported. They used to be pinned at their initial
    /// `0.0` forever, which made every selection rule rank a constant.
    #[test]
    fn observed_performance_reaches_the_active_candidate() {
        let mut optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(SelfTuningConfig::default())
                .expect("default config must construct");
        let mut params = vec![Array1::zeros(4)];
        let grads = vec![Array1::ones(4)];

        for loss in [1.0, 0.8, 0.6, 0.4] {
            optimizer
                .step(&mut params, &grads, stats_with_loss(loss))
                .expect("step");
        }

        let active = &optimizer.optimizer_candidates[optimizer.current_candidate_idx];
        assert_eq!(
            active.performance_history.len(),
            4,
            "every reported observation must be attributed to the active candidate"
        );
        // Target metric is Loss (lower is better), so rewards are negated:
        // mean of -1.0, -0.8, -0.6, -0.4 is -0.7.
        assert!(
            (active.average_reward - (-0.7)).abs() < 1e-12,
            "average reward must be the mean of the negated losses, got {}",
            active.average_reward
        );
        assert_ne!(
            active.average_reward, 0.0,
            "regression: candidate statistics are still frozen at their initial 0.0"
        );
        let (lo, hi) = active.confidence_interval;
        assert!(
            lo < active.average_reward && active.average_reward < hi,
            "the confidence interval must bracket the mean, got ({lo}, {hi})"
        );
        assert!(
            (optimizer.bandit_state.reward_estimates[optimizer.current_candidate_idx] - (-0.7))
                .abs()
                < 1e-12,
            "the bandit arm must see the same estimate as the candidate"
        );
    }

    /// Rewards must be higher-is-better regardless of the target metric,
    /// otherwise `PerformanceBased` selection would prefer the *worst*
    /// optimizer whenever the target metric is a loss.
    #[test]
    fn reward_orientation_follows_the_target_metric() {
        let loss_tuner: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(SelfTuningConfig {
                target_metric: TargetMetric::Loss,
                ..Default::default()
            })
            .expect("construct");
        assert_eq!(loss_tuner.reward_from_metric(0.3), -0.3);

        let acc_tuner: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(SelfTuningConfig {
                target_metric: TargetMetric::Accuracy,
                ..Default::default()
            })
            .expect("construct");
        assert_eq!(acc_tuner.reward_from_metric(0.3), 0.3);
    }

    /// A non-finite observation must not poison the running statistics.
    #[test]
    fn non_finite_observations_are_ignored() {
        let mut optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(SelfTuningConfig::default()).expect("construct");
        let mut params = vec![Array1::zeros(2)];
        let grads = vec![Array1::ones(2)];

        optimizer
            .step(&mut params, &grads, stats_with_loss(1.0))
            .expect("step");
        optimizer
            .step(&mut params, &grads, stats_with_loss(f64::NAN))
            .expect("step");

        let active = &optimizer.optimizer_candidates[optimizer.current_candidate_idx];
        assert_eq!(active.performance_history.len(), 1);
        assert!(active.average_reward.is_finite());
    }

    /// All four selection strategies must be reachable by a caller. Only
    /// `MultiArmedBandit { UCB1 }` used to be constructible.
    #[test]
    fn every_selection_strategy_is_reachable() {
        let mut optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(SelfTuningConfig::default()).expect("construct");

        for strategy in [
            OptimizerSelectionStrategy::MultiArmedBandit {
                algorithm: BanditAlgorithm::EpsilonGreedy,
            },
            OptimizerSelectionStrategy::MultiArmedBandit {
                algorithm: BanditAlgorithm::ThompsonSampling,
            },
            OptimizerSelectionStrategy::MultiArmedBandit {
                algorithm: BanditAlgorithm::LinUCB,
            },
            OptimizerSelectionStrategy::PerformanceBased {
                min_difference: 0.01,
            },
            OptimizerSelectionStrategy::RoundRobin { current_index: 0 },
            OptimizerSelectionStrategy::MetaLearning {
                problem_features: vec![0.0; 5],
                optimizer_mappings: HashMap::new(),
            },
        ] {
            optimizer.set_selection_strategy(strategy);
            let picked = optimizer.adapt_optimizer(&stats_with_loss(1.0));
            assert!(picked.is_ok(), "strategy must be usable end to end");
            assert!(optimizer.current_candidate_idx < optimizer.optimizer_candidates.len());
        }
    }

    /// `PerformanceBased` selection must pick the candidate with the best
    /// measured reward, which is only possible now that rewards are recorded.
    #[test]
    fn performance_based_selection_picks_the_best_measured_candidate() {
        let mut optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(SelfTuningConfig::default()).expect("construct");
        // Candidate 1 measured better (less negative reward) than 0 and 2.
        optimizer.optimizer_candidates[0].average_reward = -1.0;
        optimizer.optimizer_candidates[1].average_reward = -0.1;
        optimizer.optimizer_candidates[2].average_reward = -0.5;

        assert_eq!(optimizer.select_optimizer_performance_based(), 1);
    }
    /// An observation must be credited to the optimizer that produced it, not
    /// to whichever optimizer a switch on the same step happened to select.
    ///
    /// `record_candidate_performance` used to run *after* `maybe_adapt_optimizer`,
    /// so on every switching step the incoming optimizer was handed the outgoing
    /// one's result — systematically rewarding whichever optimizer was switched
    /// *to* after a bad step, which is exactly backwards.
    #[test]
    fn an_observation_is_credited_to_the_optimizer_that_produced_it() {
        let mut optimizer: SelfTuningOptimizer<f64, scirs2_core::ndarray::Ix1> =
            SelfTuningOptimizer::new(SelfTuningConfig {
                warmup_steps: 0,
                evaluation_window: 4,
                max_switches_per_epoch: 10,
                min_adaptation_interval: Duration::ZERO,
                improvement_threshold: 10.0, // any step counts as "stagnating"
                ..Default::default()
            })
            .expect("construct");
        // Round-robin makes every adaptation move to a different candidate, so
        // a mis-ordered recording is guaranteed to land on the wrong one.
        optimizer
            .set_selection_strategy(OptimizerSelectionStrategy::RoundRobin { current_index: 0 });

        let mut params = vec![Array1::zeros(3)];
        let grads = vec![Array1::ones(3)];

        // Each step carries a unique loss, so each reward identifies its step.
        // Run until a step actually switches optimizers, then assert on that
        // step: that is the only step where the ordering is observable.
        let mut switched_on: Option<(usize, usize, f64)> = None;
        for i in 0..40 {
            let active_before = optimizer.current_candidate_idx;
            let loss = 1.0 + i as f64;
            optimizer
                .step(&mut params, &grads, stats_with_loss(loss))
                .expect("step");
            if optimizer.current_candidate_idx != active_before {
                switched_on = Some((active_before, optimizer.current_candidate_idx, -loss));
                break;
            }
        }

        let (produced_by, switched_to, reward) =
            switched_on.expect("no switch occurred, so the ordering is not under test");
        assert_ne!(produced_by, switched_to);
        assert_eq!(
            optimizer.optimizer_candidates[produced_by]
                .performance_history
                .last()
                .copied(),
            Some(reward),
            "the observation must sit on the candidate that was active when it \
             was measured (candidate {produced_by}), not on the one switched to"
        );
        assert_ne!(
            optimizer.optimizer_candidates[switched_to]
                .performance_history
                .last()
                .copied(),
            Some(reward),
            "the candidate switched *to* (candidate {switched_to}) must not be \
             credited with the outgoing optimizer's result"
        );
    }
}
