use std::fmt::Debug;
// Neural architecture search strategies and algorithms
//
// This module implements various search strategies for neural architecture search,
// including evolutionary algorithms, Bayesian optimization, reinforcement learning,
// and other advanced search methods.

pub mod strategies;
pub mod evolutionary;
pub mod bayesian;
pub mod reinforcement;
pub mod differentiable;
pub mod progressive;

use std::collections::{HashMap, VecDeque};
use scirs2_core::numeric::Float;
use scirs2_core::ndarray::{Array1, Array2};

use super::architecture::{ArchitectureSpec, ArchitectureCandidate};

pub use strategies::*;
pub use evolutionary::*;
pub use bayesian::*;
pub use reinforcement::*;
pub use differentiable::*;
pub use progressive::*;

/// Search strategy implementation
pub struct SearchStrategy<T: Float + Debug + Send + Sync + 'static> {
    /// Strategy type
    strategy_type: SearchStrategyType,

    /// Random number generator
    #[allow(dead_code)]
    rng: Box<dyn scirs2_core::random::RngCore + Send>,

    /// Strategy-specific state
    state: SearchStrategyState<T>,

    /// Optimization history
    optimization_history: Vec<OptimizationStep<T>>,

    /// Current best architectures
    best_architectures: Vec<ArchitectureCandidate>,
}

impl<T: Float + Debug + std::fmt::Debug + Send + Sync> std::fmt::Debug for SearchStrategy<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchStrategy")
            .field("strategy_type", &self.strategy_type)
            .field("state", &self.state)
            .field("optimization_history", &self.optimization_history)
            .field("best_architectures", &self.best_architectures)
            .finish()
    }
}

/// Types of search strategies
#[derive(Debug, Clone, Copy, Default)]
pub enum SearchStrategyType {
    /// Random search
    #[default]
    Random,

    /// Evolutionary algorithm
    Evolutionary,

    /// Bayesian optimization
    BayesianOptimization,

    /// Reinforcement learning
    ReinforcementLearning,

    /// Differentiable NAS
    DifferentiableNAS,

    /// Progressive search
    Progressive,

    /// Multi-objective search
    MultiObjective,

    /// Hyperband-based search
    Hyperband,
}

/// Search strategy state
#[derive(Debug)]
pub enum SearchStrategyState<T: Float + Debug + Send + Sync + 'static> {
    Random(RandomSearchState),
    Evolutionary(EvolutionarySearchState<T>),
    Bayesian(BayesianOptimizationState<T>),
    ReinforcementLearning(RLSearchState<T>),
    Differentiable(DifferentiableNASState<T>),
    Progressive(ProgressiveSearchState<T>),
    MultiObjective(MultiObjectiveState<T>),
}

/// Random search state
#[derive(Debug, Default)]
pub struct RandomSearchState {
    /// Sampling budget remaining
    pub budget_remaining: usize,

    /// Sampling history
    pub sampling_history: Vec<String>,
}

/// Optimization step record
#[derive(Debug, Clone)]
pub struct OptimizationStep<T: Float + Debug + Send + Sync + 'static> {
    /// Step number
    pub step: usize,

    /// Architecture evaluated
    pub architecture: ArchitectureSpec,

    /// Performance achieved
    pub performance: T,

    /// Step timestamp
    pub timestamp: std::time::Instant,

    /// Step type
    pub step_type: OptimizationStepType,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of optimization steps
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationStepType {
    Random,
    Mutation,
    Crossover,
    Selection,
    Evaluation,
    Acquisition,
    PolicyUpdate,
    ArchitectureUpdate,
}

// Implementation methods
impl<T: Float + Debug + Default + std::fmt::Debug + Send + Sync> SearchStrategy<T> {
    /// Create new search strategy
    pub fn new(strategy_type: SearchStrategyType, config: SearchConfig) -> Self {
        let rng = Box::new(scirs2_core::random::thread_rng());
        
        let state = match strategy_type {
            SearchStrategyType::Random => {
                SearchStrategyState::Random(RandomSearchState {
                    budget_remaining: config.budget,
                    sampling_history: Vec::new(),
                })
            }
            SearchStrategyType::Evolutionary => {
                SearchStrategyState::Evolutionary(EvolutionarySearchState::new(config.population_size))
            }
            SearchStrategyType::BayesianOptimization => {
                SearchStrategyState::Bayesian(BayesianOptimizationState::new())
            }
            SearchStrategyType::ReinforcementLearning => {
                SearchStrategyState::ReinforcementLearning(RLSearchState::new())
            }
            SearchStrategyType::DifferentiableNAS => {
                SearchStrategyState::Differentiable(DifferentiableNASState::new())
            }
            SearchStrategyType::Progressive => {
                SearchStrategyState::Progressive(ProgressiveSearchState::new())
            }
            SearchStrategyType::MultiObjective => {
                SearchStrategyState::MultiObjective(MultiObjectiveState::new())
            }
            SearchStrategyType::Hyperband => {
                // Default to progressive for now
                SearchStrategyState::Progressive(ProgressiveSearchState::new())
            }
        };

        Self {
            strategy_type,
            rng,
            state,
            optimization_history: Vec::new(),
            best_architectures: Vec::new(),
        }
    }

    /// Generate next architecture candidate
    pub fn generate_candidate(&mut self) -> Result<ArchitectureCandidate, SearchError> {
        match &mut self.state {
            SearchStrategyState::Random(state) => {
                self.generate_random_candidate(state)
            }
            SearchStrategyState::Evolutionary(state) => {
                self.generate_evolutionary_candidate(state)
            }
            SearchStrategyState::Bayesian(state) => {
                self.generate_bayesian_candidate(state)
            }
            SearchStrategyState::ReinforcementLearning(state) => {
                self.generate_rl_candidate(state)
            }
            SearchStrategyState::Differentiable(state) => {
                self.generate_differentiable_candidate(state)
            }
            SearchStrategyState::Progressive(state) => {
                self.generate_progressive_candidate(state)
            }
            SearchStrategyState::MultiObjective(state) => {
                self.generate_multiobjective_candidate(state)
            }
        }
    }

    /// Update strategy with evaluation result
    pub fn update_with_result(&mut self, candidate: &ArchitectureCandidate, performance: T) {
        // Record optimization step
        let step = OptimizationStep {
            step: self.optimization_history.len(),
            architecture: candidate.architecture.clone(),
            performance,
            timestamp: std::time::Instant::now(),
            step_type: OptimizationStepType::Evaluation,
            metadata: HashMap::new(),
        };
        
        self.optimization_history.push(step);

        // Update best architectures
        self.update_best_architectures(candidate.clone());

        // Strategy-specific updates
        match &mut self.state {
            SearchStrategyState::Evolutionary(state) => {
                self.update_evolutionary_state(state, candidate, performance);
            }
            SearchStrategyState::Bayesian(state) => {
                self.update_bayesian_state(state, candidate, performance);
            }
            SearchStrategyState::ReinforcementLearning(state) => {
                self.update_rl_state(state, candidate, performance);
            }
            SearchStrategyState::Differentiable(state) => {
                self.update_differentiable_state(state, candidate, performance);
            }
            SearchStrategyState::Progressive(state) => {
                self.update_progressive_state(state, candidate, performance);
            }
            SearchStrategyState::MultiObjective(state) => {
                self.update_multiobjective_state(state, candidate, performance);
            }
            _ => {} // No updates needed for random search
        }
    }

    /// Get current best architectures
    pub fn get_best_architectures(&self) -> &Vec<ArchitectureCandidate> {
        &self.best_architectures
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &Vec<OptimizationStep<T>> {
        &self.optimization_history
    }

    /// Check if search should terminate
    pub fn should_terminate(&self) -> bool {
        match &self.state {
            SearchStrategyState::Random(state) => {
                state.budget_remaining == 0
            }
            SearchStrategyState::Evolutionary(state) => {
                state.generation >= 100 // Default max generations
            }
            _ => false // Continue indefinitely for other strategies
        }
    }

    // Strategy-specific generation methods
    fn generate_random_candidate(&mut self, state: &mut RandomSearchState) -> Result<ArchitectureCandidate, SearchError> {
        if state.budget_remaining == 0 {
            return Err(SearchError::BudgetExhausted);
        }

        state.budget_remaining -= 1;
        
        // Generate random architecture (simplified)
        let id = format!("random_{}", state.sampling_history.len());
        let arch_spec = ArchitectureSpec::new(vec![], super::architecture::GlobalArchitectureConfig::default());
        let candidate = ArchitectureCandidate::new(id.clone(), arch_spec);
        
        state.sampling_history.push(id);
        
        Ok(candidate)
    }

    fn generate_evolutionary_candidate(&mut self, state: &mut EvolutionarySearchState<T>) -> Result<ArchitectureCandidate, SearchError> {
        use super::architecture::{LayerSpec, LayerDimensions, LayerType, ActivationType, GlobalArchitectureConfig};
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Select the best existing individual to mutate, or create a base architecture
        let parent = if !state.population.is_empty() {
            // Find the individual with the highest optimization performance
            let best_idx = state.population
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.performance.optimization_performance
                        .partial_cmp(&b.performance.optimization_performance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            state.population[best_idx].architecture.clone()
        } else {
            ArchitectureSpec::new(
                vec![
                    LayerSpec::new(LayerType::Linear, LayerDimensions { input_dim: 128, output_dim: 64, hidden_dims: vec![] }, ActivationType::ReLU),
                ],
                GlobalArchitectureConfig::default(),
            )
        };

        // Deterministic perturbation seed from generation number and history size
        let mut hasher = DefaultHasher::new();
        state.generation.hash(&mut hasher);
        state.fitness_history.len().hash(&mut hasher);
        let seed = hasher.finish();

        // Mutate layers: perturb output dims within ±25% using seed-based offsets
        let mut new_layers = parent.layers.clone();
        if !new_layers.is_empty() {
            let layer_idx = (seed as usize) % new_layers.len();
            let layer = &mut new_layers[layer_idx];

            // Perturb output_dim by ±12 (bounded to [16, 512])
            let delta = if (seed >> 8) & 1 == 0 { 12usize } else { 0usize };
            let dir = if (seed >> 9) & 1 == 0 { 1i64 } else { -1i64 };
            let new_out = ((layer.dimensions.output_dim as i64 + dir * delta as i64).max(16)).min(512) as usize;
            layer.dimensions.output_dim = new_out;

            // Also perturb activation based on seed
            layer.activation = match (seed >> 4) % 3 {
                0 => ActivationType::ReLU,
                1 => ActivationType::GELU,
                _ => ActivationType::Tanh,
            };
        }

        let mutated_arch = ArchitectureSpec::new(new_layers, parent.global_config.clone());
        let candidate_id = format!("evo_gen{}_mut{}", state.generation, state.fitness_history.len());
        let candidate = ArchitectureCandidate::new(candidate_id, mutated_arch);

        state.generation += 1;
        Ok(candidate)
    }

    fn generate_bayesian_candidate(&mut self, state: &mut BayesianOptimizationState<T>) -> Result<ArchitectureCandidate, SearchError> {
        use super::architecture::{LayerSpec, LayerDimensions, LayerType, ActivationType, GlobalArchitectureConfig};
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // UCB acquisition: mean + kappa * std, kappa = 2.0
        // Fall back to random if fewer than 3 observations
        let n_obs = state.observations.len();

        if n_obs < 3 {
            // Random sampling fallback
            let mut hasher = DefaultHasher::new();
            n_obs.hash(&mut hasher);
            let seed = hasher.finish();

            let hidden = 32 + (seed as usize % 96); // [32, 128)
            let layers = vec![
                LayerSpec::new(
                    LayerType::Linear,
                    LayerDimensions { input_dim: 128, output_dim: hidden, hidden_dims: vec![] },
                    ActivationType::ReLU,
                )
            ];
            let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
            let candidate = ArchitectureCandidate::new(format!("bo_random_{}", n_obs), arch);
            return Ok(candidate);
        }

        // Compute mean and std from observed scores (simple GP surrogate via moments)
        let scores: Vec<f64> = state.observations.iter().map(|(_, s)| *s).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt().max(1e-8);

        // UCB = mean + kappa * std_dev; kappa = 2.0
        let kappa = 2.0_f64;
        let ucb_score = mean + kappa * std_dev;

        // Translate UCB score to architecture parameters:
        // Higher ucb_score → wider layers, more depth
        let hidden_dim = ((ucb_score.abs() * 32.0) as usize).clamp(32, 256);
        let n_layers = (1 + (ucb_score.abs() as usize) % 3).min(4);

        let mut layers = Vec::new();
        for i in 0..n_layers {
            let in_dim = if i == 0 { 128 } else { hidden_dim };
            layers.push(LayerSpec::new(
                LayerType::Linear,
                LayerDimensions { input_dim: in_dim, output_dim: hidden_dim, hidden_dims: vec![] },
                ActivationType::GELU,
            ));
        }

        let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
        let candidate = ArchitectureCandidate::new(format!("bo_ucb_{}", n_obs), arch);
        Ok(candidate)
    }

    fn generate_rl_candidate(&mut self, state: &mut RLSearchState<T>) -> Result<ArchitectureCandidate, SearchError> {
        use super::architecture::{LayerSpec, LayerDimensions, LayerType, ActivationType, GlobalArchitectureConfig};

        // ε-greedy multi-armed bandit over discrete hyperparameter choices.
        // Each "arm" corresponds to a layer configuration. ε = 0.1 → explore every 10th step.
        // Use replay buffer size as a pseudo-random source (deterministic w.r.t. state)
        let step = state.reward_history.len();
        let explore = (step % 10) == 0; // deterministic ε = 0.1 schedule

        // Arm definitions: (LayerType, hidden_dim, ActivationType)
        let arm_configs: &[(LayerType, usize, ActivationType)] = &[
            (LayerType::Linear,    64,  ActivationType::ReLU),
            (LayerType::Linear,   128,  ActivationType::GELU),
            (LayerType::LSTM,      64,  ActivationType::Tanh),
            (LayerType::Attention, 128, ActivationType::ReLU),
            (LayerType::Linear,   256,  ActivationType::GELU),
        ];
        let n_arms = arm_configs.len();

        let chosen_arm = if explore || state.replay_buffer.size == 0 {
            // Exploration: cycle through arms
            step % n_arms
        } else {
            // Exploitation: pick arm with highest estimated reward from reward history
            // Divide history into n_arms buckets and pick best mean
            let history: Vec<f64> = state.reward_history.iter()
                .map(|&r| r)
                .collect();
            let bucket_size = (history.len() / n_arms).max(1);

            let mut best_arm = 0usize;
            let mut best_mean = f64::NEG_INFINITY;
            for arm in 0..n_arms {
                let start = arm * bucket_size;
                let end = ((arm + 1) * bucket_size).min(history.len());
                if start >= history.len() {
                    break;
                }
                let bucket = &history[start..end];
                let arm_mean = bucket.iter().sum::<f64>() / bucket.len() as f64;
                if arm_mean > best_mean {
                    best_mean = arm_mean;
                    best_arm = arm;
                }
            }
            best_arm
        };

        let (layer_type, hidden_dim, activation) = arm_configs[chosen_arm];
        let layers = vec![
            LayerSpec::new(
                layer_type,
                LayerDimensions { input_dim: 128, output_dim: hidden_dim, hidden_dims: vec![] },
                activation,
            )
        ];
        let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
        let candidate = ArchitectureCandidate::new(format!("rl_bandit_arm{}_step{}", chosen_arm, step), arch);
        Ok(candidate)
    }

    fn generate_differentiable_candidate(&mut self, state: &mut DifferentiableNASState<T>) -> Result<ArchitectureCandidate, SearchError> {
        use super::architecture::{LayerSpec, LayerDimensions, LayerType, ActivationType, GlobalArchitectureConfig};

        // REINFORCE / finite-differences gradient approximation over arch_parameters.
        // If no parameters yet, return a random perturbation as the initial point.
        if state.arch_parameters.is_empty() {
            // Initialise with a simple 2-layer candidate
            let layers = vec![
                LayerSpec::new(LayerType::Linear, LayerDimensions { input_dim: 128, output_dim: 64, hidden_dims: vec![] }, ActivationType::ReLU),
                LayerSpec::new(LayerType::Linear, LayerDimensions { input_dim: 64, output_dim: 32, hidden_dims: vec![] }, ActivationType::GELU),
            ];
            let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
            return Ok(ArchitectureCandidate::new(format!("dnas_init_{}", state.current_epoch), arch));
        }

        // Finite-difference gradient step: δ = 0.1
        // positive_score ≈ current best loss_history value (or 0 if empty)
        let delta = 0.1_f64;

        let last_loss = state.loss_history.last()
            .and_then(|v| scirs2_core::numeric::NumCast::from(*v))
            .unwrap_or(0.0_f64);

        // Approximate gradient direction: increase params where loss decreased
        // Use sign of (previous - current) loss to determine update direction
        let prev_loss = state.loss_history.len().checked_sub(2)
            .and_then(|i| state.loss_history.get(i))
            .and_then(|v| scirs2_core::numeric::NumCast::from(*v))
            .unwrap_or(last_loss);

        let grad_sign = if prev_loss > last_loss { 1.0_f64 } else { -1.0_f64 };

        // Translate arch_parameters[0] to hidden_dim (perturbed by delta * grad_sign)
        let base_param: f64 = scirs2_core::numeric::NumCast::from(state.arch_parameters[0])
            .unwrap_or(0.5_f64);
        let updated_param = (base_param + delta * grad_sign).clamp(0.0, 1.0);

        // Map [0,1] param to hidden_dim in [32, 256]
        let hidden_dim = (32.0 + updated_param * 224.0) as usize;

        let layers = vec![
            LayerSpec::new(
                LayerType::Linear,
                LayerDimensions { input_dim: 128, output_dim: hidden_dim, hidden_dims: vec![] },
                ActivationType::GELU,
            )
        ];
        let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
        let candidate = ArchitectureCandidate::new(
            format!("dnas_epoch{}_hd{}", state.current_epoch, hidden_dim),
            arch,
        );
        state.current_epoch += 1;
        Ok(candidate)
    }

    fn generate_progressive_candidate(&mut self, state: &mut ProgressiveSearchState<T>) -> Result<ArchitectureCandidate, SearchError> {
        use super::architecture::{LayerSpec, LayerDimensions, LayerType, ActivationType, GlobalArchitectureConfig};

        // Progressive grid search: coarse → fine.
        // Resolution doubles each phase: phase 0 = 3 values/HP, phase k = 3 * 2^k values/HP.
        let phase_idx = state.current_phase;
        let resolution = 3usize << phase_idx; // 3, 6, 12, 24, ...

        // Hyperparameter grid for hidden_dim in [32, 256] and n_layers in [1, 4]
        let evaluated_in_phase = state.phase_candidates
            .get(&phase_idx)
            .map(|v| v.len())
            .unwrap_or(0);

        let total_grid_points = resolution * 4; // resolution hidden_dims × 4 layer counts

        if evaluated_in_phase >= total_grid_points {
            // Advance to next phase and return the first grid point of the new phase.
            // Grid point 0 always maps to: hidden_dim = 32 (index 0), n_layers = 1 (index 0).
            state.current_phase += 1;
            let new_phase = state.current_phase;
            let hidden_dim = 32usize;
            let n_layers = 1usize;

            let mut layers = Vec::new();
            for i in 0..n_layers {
                let in_dim = if i == 0 { 128 } else { hidden_dim };
                layers.push(LayerSpec::new(
                    LayerType::Linear,
                    LayerDimensions { input_dim: in_dim, output_dim: hidden_dim, hidden_dims: vec![] },
                    ActivationType::ReLU,
                ));
            }
            let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
            let new_candidate_id = format!("prog_phase{}_pt0", new_phase);
            let new_candidate = ArchitectureCandidate::new(new_candidate_id.clone(), arch);

            // Register the first point in the new phase
            state.phase_candidates
                .entry(new_phase)
                .or_insert_with(Vec::new)
                .push(new_candidate_id);

            return Ok(new_candidate);
        }

        // Decode grid point index into (hidden_dim, n_layers)
        let grid_step = 224 / resolution.max(1); // span [32..256] / resolution
        let hd_idx = evaluated_in_phase % resolution;
        let layer_idx = evaluated_in_phase / resolution;
        let hidden_dim = (32 + hd_idx * grid_step).min(256);
        let n_layers = (1 + layer_idx % 4).min(4);

        let mut layers = Vec::new();
        for i in 0..n_layers {
            let in_dim = if i == 0 { 128 } else { hidden_dim };
            layers.push(LayerSpec::new(
                LayerType::Linear,
                LayerDimensions { input_dim: in_dim, output_dim: hidden_dim, hidden_dims: vec![] },
                ActivationType::ReLU,
            ));
        }
        let candidate_id = format!("prog_phase{}_pt{}", phase_idx, evaluated_in_phase);
        let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
        let candidate = ArchitectureCandidate::new(candidate_id.clone(), arch);

        // Track this candidate in the phase so evaluated_in_phase advances per call
        state.phase_candidates
            .entry(phase_idx)
            .or_insert_with(Vec::new)
            .push(candidate_id);

        Ok(candidate)
    }

    fn generate_multiobjective_candidate(&mut self, state: &mut MultiObjectiveState<T>) -> Result<ArchitectureCandidate, SearchError> {
        use super::architecture::{LayerSpec, LayerDimensions, LayerType, ActivationType, GlobalArchitectureConfig};

        // NSGA-II selection: rank by Pareto dominance then crowding distance.
        // If no Pareto front yet, return a baseline candidate.
        if state.pareto_front.is_empty() {
            // Initial candidate spanning all objectives
            let layers = vec![
                LayerSpec::new(LayerType::Linear, LayerDimensions { input_dim: 128, output_dim: 128, hidden_dims: vec![] }, ActivationType::ReLU),
                LayerSpec::new(LayerType::Linear, LayerDimensions { input_dim: 128, output_dim: 64, hidden_dims: vec![] }, ActivationType::GELU),
            ];
            let arch = ArchitectureSpec::new(layers, GlobalArchitectureConfig::default());
            return Ok(ArchitectureCandidate::new("mo_init_0".to_string(), arch));
        }

        // Compute dominance rank for each individual in the Pareto front
        let n = state.pareto_front.len();
        let mut dom_count = vec![0usize; n];

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let pi = &state.pareto_front[i].performance;
                let pj = &state.pareto_front[j].performance;
                // j dominates i if it is at least as good in all objectives and strictly better in one
                let j_dom_i = (pj.optimization_performance >= pi.optimization_performance
                    && pj.convergence_speed >= pi.convergence_speed
                    && pj.generalization >= pi.generalization)
                    && (pj.optimization_performance > pi.optimization_performance
                        || pj.convergence_speed > pi.convergence_speed
                        || pj.generalization > pi.generalization);
                if j_dom_i {
                    dom_count[i] += 1;
                }
            }
        }

        // Select the individual with the lowest domination count (Pareto rank 0 is best)
        // Break ties by highest optimization_performance (crowding distance proxy)
        let best_idx = (0..n)
            .min_by(|&a, &b| {
                dom_count[a].cmp(&dom_count[b]).then_with(|| {
                    state.pareto_front[b].performance.optimization_performance
                        .partial_cmp(&state.pareto_front[a].performance.optimization_performance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .unwrap_or(0);

        // Crossover: blend best individual with its neighbour
        let next_idx = (best_idx + 1) % n;
        let parent_a = &state.pareto_front[best_idx].architecture;
        let parent_b = &state.pareto_front[next_idx].architecture;

        // Take layers from both parents (first half from a, second half from b)
        let half_a = parent_a.layers.len() / 2 + 1;
        let mut new_layers: Vec<LayerSpec> = parent_a.layers.iter().take(half_a).cloned().collect();
        new_layers.extend(parent_b.layers.iter().skip(half_a).cloned());

        if new_layers.is_empty() {
            // Fallback: single layer from parent_a config
            new_layers.push(LayerSpec::new(
                LayerType::Linear,
                LayerDimensions { input_dim: 128, output_dim: 64, hidden_dims: vec![] },
                ActivationType::ReLU,
            ));
        }

        let arch = ArchitectureSpec::new(new_layers, parent_a.global_config.clone());
        let candidate_id = format!("mo_nsga2_cross_{}", state.pareto_front.len());
        Ok(ArchitectureCandidate::new(candidate_id, arch))
    }

    // Strategy-specific update methods
    fn update_evolutionary_state(&mut self, _state: &mut EvolutionarySearchState<T>, _candidate: &ArchitectureCandidate, _performance: T) {
        // Implementation in evolutionary module
    }

    fn update_bayesian_state(&mut self, _state: &mut BayesianOptimizationState<T>, _candidate: &ArchitectureCandidate, _performance: T) {
        // Implementation in bayesian module
    }

    fn update_rl_state(&mut self, _state: &mut RLSearchState<T>, _candidate: &ArchitectureCandidate, _performance: T) {
        // Implementation in reinforcement module
    }

    fn update_differentiable_state(&mut self, _state: &mut DifferentiableNASState<T>, _candidate: &ArchitectureCandidate, _performance: T) {
        // Implementation in differentiable module
    }

    fn update_progressive_state(&mut self, _state: &mut ProgressiveSearchState<T>, _candidate: &ArchitectureCandidate, _performance: T) {
        // Implementation in progressive module
    }

    fn update_multiobjective_state(&mut self, _state: &mut MultiObjectiveState<T>, _candidate: &ArchitectureCandidate, _performance: T) {
        // Implementation in multi-objective module
    }

    fn update_best_architectures(&mut self, candidate: ArchitectureCandidate) {
        self.best_architectures.push(candidate);
        
        // Keep only top N architectures
        self.best_architectures.sort_by(|a, b| {
            b.performance.optimization_performance
                .partial_cmp(&a.performance.optimization_performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        if self.best_architectures.len() > 10 {
            self.best_architectures.truncate(10);
        }
    }
}

/// Search configuration
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Search budget (number of evaluations)
    pub budget: usize,

    /// Population size (for evolutionary strategies)
    pub population_size: usize,

    /// Maximum iterations
    pub max_iterations: usize,

    /// Convergence threshold
    pub convergence_threshold: f64,

    /// Early stopping patience
    pub early_stopping_patience: usize,

    /// Random seed
    pub random_seed: Option<u64>,

    /// Strategy-specific parameters
    pub strategy_params: HashMap<String, f64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            budget: 1000,
            population_size: 50,
            max_iterations: 100,
            convergence_threshold: 0.001,
            early_stopping_patience: 10,
            random_seed: None,
            strategy_params: HashMap::new(),
        }
    }
}

/// Search errors
#[derive(Debug, Clone)]
pub enum SearchError {
    BudgetExhausted,
    InvalidConfiguration(String),
    GenerationFailed(String),
    EvaluationFailed(String),
    NotImplemented(String),
    StrategyError(String),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::BudgetExhausted => write!(f, "Search budget exhausted"),
            SearchError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            SearchError::GenerationFailed(msg) => write!(f, "Generation failed: {}", msg),
            SearchError::EvaluationFailed(msg) => write!(f, "Evaluation failed: {}", msg),
            SearchError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            SearchError::StrategyError(msg) => write!(f, "Strategy error: {}", msg),
        }
    }
}

impl std::error::Error for SearchError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_strategy_creation() {
        let config = SearchConfig::default();
        let strategy: SearchStrategy<f64> = SearchStrategy::new(SearchStrategyType::Random, config);
        
        match strategy.state {
            SearchStrategyState::Random(_) => {}
            _ => panic!("Expected random search state"),
        }
    }

    #[test]
    fn test_random_candidate_generation() {
        let config = SearchConfig { budget: 5, ..Default::default() };
        let mut strategy: SearchStrategy<f64> = SearchStrategy::new(SearchStrategyType::Random, config);
        
        for _ in 0..5 {
            let result = strategy.generate_candidate();
            assert!(result.is_ok());
        }
        
        // Should fail when budget exhausted
        let result = strategy.generate_candidate();
        assert!(matches!(result, Err(SearchError::BudgetExhausted)));
    }

    #[test]
    fn test_termination_condition() {
        let config = SearchConfig { budget: 0, ..Default::default() };
        let strategy: SearchStrategy<f64> = SearchStrategy::new(SearchStrategyType::Random, config);
        
        assert!(strategy.should_terminate());
    }

    #[test]
    fn test_best_architecture_tracking() {
        let config = SearchConfig::default();
        let mut strategy: SearchStrategy<f64> = SearchStrategy::new(SearchStrategyType::Random, config);
        
        let arch = ArchitectureSpec::new(vec![], super::architecture::GlobalArchitectureConfig::default());
        let candidate = ArchitectureCandidate::new("test".to_string(), arch);
        
        strategy.update_with_result(&candidate, 0.8);
        assert_eq!(strategy.get_best_architectures().len(), 1);
    }
}