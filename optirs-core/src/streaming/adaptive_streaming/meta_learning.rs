// Meta-learning and experience management for adaptive streaming
//
// This module provides sophisticated meta-learning capabilities that learn from
// optimization experiences to improve future adaptation decisions, including
// experience replay, transfer learning, and adaptive strategy selection.

use super::config::*;
use super::meta_bandit::{arm_index_for, arm_table, state_features, BanditArm, FeatureScaler};
use super::meta_transfer::TransferLearning;
use super::optimizer::{Adaptation, AdaptationPriority, AdaptationType, StreamingDataPoint};
use super::performance::PerformanceTracker;

pub use super::meta_transfer::{
    DomainAdaptation, TransferMetrics, TransferStrategy, MIN_TRANSFER_SIMILARITY,
};

use crate::utils::{scalar_or, try_scalar_str};
use scirs2_core::numeric::Float;
use scirs2_core::random::thread_rng;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Type of meta-model used for decision making
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaModelType {
    NeuralNetwork,
    LinearRegression,
    RandomForest,
    GradientBoosting,
    SupportVectorMachine,
}

/// Meta-learning system for streaming optimization
pub struct MetaLearner<A: Float + Send + Sync> {
    /// Meta-learning configuration
    config: MetaLearningConfig,
    /// Experience buffer for learning
    experience_buffer: ExperienceBuffer<A>,
    /// Meta-model for decision making
    meta_model: MetaModel<A>,
    /// Strategy selection system
    strategy_selector: StrategySelector<A>,
    /// Transfer learning system
    transfer_learning: TransferLearning<A>,
    /// Meta-learning statistics
    statistics: MetaLearningStatistics<A>,
    /// Learning rate adaptation
    learning_rate_adapter: LearningRateAdapter<A>,
    /// Instant the current episode began (real clock for episode duration).
    episode_start: Instant,
    /// First reward recorded in the current episode.
    episode_initial_performance: Option<A>,
    /// Number of experiences recorded in the current episode.
    episode_adaptation_count: usize,
    /// Externally supplied resource-state signal, set via
    /// [`MetaLearner::update_context_signals`]. Empty means "not reported".
    context_resource_state: Vec<A>,
    /// Externally supplied drift-indicator signal.
    context_drift_indicators: Vec<A>,
}

/// Type alias for experience replay functionality
pub type ExperienceReplay<A> = ExperienceBuffer<A>;

/// Experience buffer for storing and managing learning experiences
pub struct ExperienceBuffer<A: Float + Send + Sync> {
    /// Buffer configuration
    config: ExperienceReplayConfig,
    /// Stored experiences
    experiences: VecDeque<MetaExperience<A>>,
    /// Priority queue for prioritized replay
    priority_queue: VecDeque<(MetaExperience<A>, A)>,
    /// Experience importance sampling
    importance_weights: HashMap<usize, A>,
    /// Maximum retained experiences, from
    /// `MetaLearningConfig::experience_buffer_size`.
    capacity: usize,
}

/// Meta-learning experience representation
#[derive(Debug, Clone)]
pub struct MetaExperience<A: Float + Send + Sync> {
    /// Unique experience ID
    pub id: u64,
    /// State when experience occurred
    pub state: MetaState<A>,
    /// Action taken
    pub action: MetaAction<A>,
    /// Reward received
    pub reward: A,
    /// Next state after action
    pub next_state: Option<MetaState<A>>,
    /// Experience timestamp
    pub timestamp: Instant,
    /// Episode context
    pub episode_context: EpisodeContext<A>,
    /// Experience priority for replay
    pub priority: A,
    /// Number of times replayed
    pub replay_count: usize,
}

/// Meta-state representation
#[derive(Debug, Clone)]
pub struct MetaState<A: Float + Send + Sync> {
    /// Performance metrics at this state
    pub performance_metrics: Vec<A>,
    /// Resource state
    pub resource_state: Vec<A>,
    /// Drift indicators
    pub drift_indicators: Vec<A>,
    /// Adaptation history length
    pub adaptation_history: usize,
    /// State timestamp
    pub timestamp: Instant,
}

/// Meta-action representation
#[derive(Debug, Clone)]
pub struct MetaAction<A: Float + Send + Sync> {
    /// Adaptation magnitudes applied
    pub adaptation_magnitudes: Vec<A>,
    /// Types of adaptations
    pub adaptation_types: Vec<AdaptationType>,
    /// Learning rate change
    pub learning_rate_change: A,
    /// Buffer size change
    pub buffer_size_change: A,
    /// Action timestamp
    pub timestamp: Instant,
}

/// Episode context for meta-learning
#[derive(Debug, Clone)]
pub struct EpisodeContext<A: Float + Send + Sync> {
    /// Episode ID
    pub episode_id: u64,
    /// Episode start time
    pub start_time: Instant,
    /// Episode duration
    pub duration: Duration,
    /// Initial performance
    pub initial_performance: A,
    /// Final performance
    pub final_performance: A,
    /// Number of adaptations in episode
    pub adaptation_count: usize,
    /// Episode outcome classification
    pub outcome: EpisodeOutcome,
}

/// Episode outcome classifications
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeOutcome {
    /// Significant improvement
    Success,
    /// Moderate improvement
    PartialSuccess,
    /// No significant change
    Neutral,
    /// Performance degradation
    Failure,
    /// Severe performance degradation
    CriticalFailure,
}

/// Meta-model for decision making
pub struct MetaModel<A: Float + Send + Sync> {
    /// Model parameters
    parameters: MetaModelParameters<A>,
    /// Training history
    training_history: VecDeque<TrainingEpisode<A>>,
    /// Model performance metrics
    performance_metrics: ModelPerformanceMetrics<A>,
    /// Feature importance
    feature_importance: Vec<A>,
    /// Contextual-bandit arms: the concrete adaptations the model can
    /// recommend, each with its own online linear reward model.
    arms: Vec<BanditArm<A>>,
    /// Number of training steps applied.
    training_steps: usize,
    /// Running feature statistics used to standardise the state vector.
    feature_scaler: FeatureScaler<A>,
}

/// Meta-model parameters
#[derive(Debug, Clone)]
pub struct MetaModelParameters<A: Float + Send + Sync> {
    /// Weight matrices for neural networks
    pub weights: Vec<Vec<A>>,
    /// Bias vectors
    pub biases: Vec<A>,
    /// Learning rate
    pub learning_rate: A,
    /// Regularization parameters
    pub regularization: RegularizationParams<A>,
    /// Optimization parameters
    pub optimization: OptimizationParams<A>,
}

/// Regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationParams<A: Float + Send + Sync> {
    /// L1 regularization strength
    pub l1_lambda: A,
    /// L2 regularization strength
    pub l2_lambda: A,
    /// Dropout rate
    pub dropout_rate: A,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

/// Optimization parameters
#[derive(Debug, Clone)]
pub struct OptimizationParams<A: Float + Send + Sync> {
    /// Momentum coefficient
    pub momentum: A,
    /// Adam beta1 parameter
    pub beta1: A,
    /// Adam beta2 parameter
    pub beta2: A,
    /// Epsilon for numerical stability
    pub epsilon: A,
    /// Gradient clipping threshold
    pub grad_clip_threshold: A,
}

/// Training episode for meta-model
#[derive(Debug, Clone)]
pub struct TrainingEpisode<A: Float + Send + Sync> {
    /// Episode ID
    pub episode_id: u64,
    /// Training loss
    pub training_loss: A,
    /// Validation loss
    pub validation_loss: A,
    /// Training accuracy
    pub training_accuracy: A,
    /// Validation accuracy
    pub validation_accuracy: A,
    /// Episode duration
    pub duration: Duration,
    /// Timestamp
    pub timestamp: Instant,
}

/// Model performance metrics
#[derive(Debug, Clone)]
pub struct ModelPerformanceMetrics<A: Float + Send + Sync> {
    /// Prediction accuracy
    pub prediction_accuracy: A,
    /// Decision quality
    pub decision_quality: A,
    /// Adaptation effectiveness
    pub adaptation_effectiveness: A,
    /// Transfer learning success rate
    pub transfer_success_rate: A,
    /// Generalization performance
    pub generalization_performance: A,
}

/// Strategy selection system
pub struct StrategySelector<A: Float + Send + Sync> {
    /// Available strategies
    strategies: HashMap<String, AdaptationStrategy<A>>,
    /// Strategy performance history
    strategy_performance: HashMap<String, StrategyPerformance<A>>,
    /// Strategy selection policy
    selection_policy: SelectionPolicy,
    /// Exploration parameters
    exploration_params: ExplorationParams<A>,
}

/// Adaptation strategy representation
#[derive(Debug, Clone)]
pub struct AdaptationStrategy<A: Float + Send + Sync> {
    /// Strategy name
    pub name: String,
    /// Strategy parameters
    pub parameters: HashMap<String, A>,
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Applicability conditions
    pub conditions: Vec<StrategyCondition<A>>,
    /// Expected outcomes
    pub expected_outcomes: Vec<A>,
}

/// Strategy types
#[derive(Debug, Clone)]
pub enum StrategyType {
    /// Conservative strategy (small changes)
    Conservative,
    /// Aggressive strategy (large changes)
    Aggressive,
    /// Balanced strategy
    Balanced,
    /// Reactive strategy (responds to changes)
    Reactive,
    /// Proactive strategy (anticipates changes)
    Proactive,
    /// Custom strategy
    Custom(String),
}

/// Strategy applicability conditions
#[derive(Debug, Clone)]
pub struct StrategyCondition<A: Float + Send + Sync> {
    /// Condition type
    pub condition_type: ConditionType,
    /// Threshold value
    pub threshold: A,
    /// Operator for comparison
    pub operator: ComparisonOperator,
    /// Weight in decision making
    pub weight: A,
}

/// Condition types for strategy selection
#[derive(Debug, Clone)]
pub enum ConditionType {
    /// Performance threshold
    Performance,
    /// Resource utilization
    ResourceUtilization,
    /// Data quality
    DataQuality,
    /// Drift detection
    DriftDetection,
    /// Time-based condition
    Temporal,
    /// Custom condition
    Custom(String),
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Equal to
    EqualTo,
    /// Between values
    Between(f64, f64),
    /// In set of values
    InSet(Vec<f64>),
}

/// Strategy performance tracking
#[derive(Debug, Clone)]
pub struct StrategyPerformance<A: Float + Send + Sync> {
    /// Number of times used
    pub usage_count: usize,
    /// Success rate
    pub success_rate: A,
    /// Average improvement
    pub avg_improvement: A,
    /// Best improvement achieved
    pub best_improvement: A,
    /// Worst outcome
    pub worst_outcome: A,
    /// Recent performance trend
    pub recent_trend: TrendDirection,
    /// Context-specific performance
    pub context_performance: HashMap<String, A>,
}

/// Trend direction for strategy performance
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Declining trend
    Declining,
    /// Stable trend
    Stable,
    /// Oscillating trend
    Oscillating,
}

/// Strategy selection policies
#[derive(Debug, Clone)]
pub enum SelectionPolicy {
    /// Epsilon-greedy selection
    EpsilonGreedy { epsilon: f64 },
    /// Upper Confidence Bound
    UCB { confidence_parameter: f64 },
    /// Thompson sampling
    ThompsonSampling,
    /// Softmax selection
    Softmax { temperature: f64 },
    /// Context-aware selection
    ContextAware,
    /// Multi-armed bandit
    MultiArmedBandit,
}

/// Exploration parameters
#[derive(Debug, Clone)]
pub struct ExplorationParams<A: Float + Send + Sync> {
    /// Exploration rate
    pub exploration_rate: A,
    /// Exploration decay
    pub exploration_decay: A,
    /// Minimum exploration rate
    pub min_exploration_rate: A,
    /// Curiosity bonus weight
    pub curiosity_weight: A,
    /// Novelty bonus weight
    pub novelty_weight: A,
}

/// Learning rate adaptation system
pub struct LearningRateAdapter<A: Float + Send + Sync> {
    /// Current learning rate
    current_rate: A,
    /// Learning rate history
    rate_history: VecDeque<A>,
    /// Rate bounds
    min_rate: A,
    max_rate: A,
}

/// Learning rate adaptation strategies
#[derive(Debug, Clone)]
pub enum LearningRateStrategy {
    /// Fixed learning rate
    Fixed,
    /// Step decay
    StepDecay { decay_factor: f64, step_size: usize },
    /// Exponential decay
    ExponentialDecay { decay_rate: f64 },
    /// Performance-based adaptation
    PerformanceBased,
    /// Cyclical learning rates
    Cyclical {
        min_lr: f64,
        max_lr: f64,
        cycle_length: usize,
    },
    /// Adaptive learning rate (Adam-style)
    Adaptive,
}

/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStatistics<A: Float + Send + Sync> {
    /// Total experiences collected
    pub total_experiences: usize,
    /// Model training episodes
    pub training_episodes: usize,
    /// Average reward per episode
    pub avg_reward_per_episode: A,
    /// Best episode reward
    pub best_episode_reward: A,
    /// Learning progress
    pub learning_progress: A,
    /// Strategy selection accuracy
    pub strategy_selection_accuracy: A,
    /// Transfer learning success rate
    pub transfer_success_rate: A,
    /// Experience replay effectiveness
    pub replay_effectiveness: A,
}

impl<A: Float + Default + Clone + std::iter::Sum + Send + Sync + std::fmt::Debug> MetaLearner<A> {
    /// Creates a new meta-learner
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let meta_config = config.meta_learning_config.clone();

        let experience_buffer = ExperienceBuffer::new(
            &meta_config.replay_config,
            meta_config.experience_buffer_size,
        );
        let meta_model = MetaModel::new(meta_config.model_complexity.clone())?;
        let strategy_selector = StrategySelector::new();
        let transfer_learning = TransferLearning::new();
        let learning_rate_adapter = LearningRateAdapter::new(meta_config.meta_learning_rate);

        let statistics = MetaLearningStatistics {
            total_experiences: 0,
            training_episodes: 0,
            avg_reward_per_episode: A::zero(),
            best_episode_reward: A::zero(),
            learning_progress: A::zero(),
            strategy_selection_accuracy: A::zero(),
            transfer_success_rate: A::zero(),
            replay_effectiveness: A::zero(),
        };

        Ok(Self {
            config: meta_config,
            experience_buffer,
            meta_model,
            strategy_selector,
            transfer_learning,
            statistics,
            learning_rate_adapter,
            episode_start: Instant::now(),
            episode_initial_performance: None,
            episode_adaptation_count: 0,
            context_resource_state: Vec::new(),
            context_drift_indicators: Vec::new(),
        })
    }

    /// Supplies the resource and drift signals the meta-learner cannot observe
    /// for itself.
    ///
    /// ML5: `extract_meta_state` used to fill `resource_state` with the
    /// constants `[0.5, 0.3]` and `drift_indicators` with `[0.1]`, so the
    /// bandit's context vector was two thirds fabricated and identical on every
    /// call. The owning optimizer knows the real values and reports them here.
    pub fn update_context_signals(&mut self, resource_state: Vec<A>, drift_indicators: Vec<A>) {
        self.context_resource_state = resource_state;
        self.context_drift_indicators = drift_indicators;
    }

    /// Updates the meta-learner with new experience
    pub fn update_experience(
        &mut self,
        state: MetaState<A>,
        action: MetaAction<A>,
        reward: A,
    ) -> Result<(), String> {
        // `state`/`action` are moved into the experience below, so the priority
        // is computed first.
        let priority = self.calculate_experience_priority(&state, &action, reward);
        let experience = MetaExperience {
            id: self.generate_experience_id(),
            state,
            action,
            reward,
            next_state: None, // Will be filled in next update
            timestamp: Instant::now(),
            episode_context: self.create_episode_context(reward)?,
            priority,
            replay_count: 0,
        };

        // Add to experience buffer
        self.experience_buffer.add_experience(experience)?;

        // Real per-episode bookkeeping.
        if self.episode_initial_performance.is_none() {
            self.episode_initial_performance = Some(reward);
        }
        self.episode_adaptation_count = self.episode_adaptation_count.saturating_add(1);

        // Update statistics
        self.statistics.total_experiences += 1;

        // Trigger learning if enough experiences collected. Two independent
        // cadences apply (CF1): `update_frequency` is the meta-model's own
        // training interval, and `ExperienceReplayConfig::replay_frequency` is
        // how often stored experience is replayed. `replay_frequency` had no
        // reader at all, so replay silently inherited `update_frequency`.
        let experiences = self.statistics.total_experiences;
        let update_due = self.config.update_frequency > 0
            && experiences.is_multiple_of(self.config.update_frequency);
        let replay_due = self.config.replay_config.replay_frequency > 0
            && experiences.is_multiple_of(self.config.replay_config.replay_frequency);
        if update_due || replay_due {
            self.trigger_learning()?;
        }

        Ok(())
    }

    /// Generates unique experience ID
    fn generate_experience_id(&self) -> u64 {
        self.statistics.total_experiences as u64 + 1
    }

    /// Creates episode context for experience.
    ///
    /// ML5: `duration`, `initial_performance` and `adaptation_count` were the
    /// fabricated constants `60s`, `0.5` and `1`, and `start_time` was stamped
    /// as "now" for every experience so the duration could never be anything
    /// else. All four are now measured from real episode state: the episode's
    /// actual start `Instant`, the first reward recorded in the episode, and the
    /// number of experiences the episode has accumulated.
    fn create_episode_context(&self, reward: A) -> Result<EpisodeContext<A>, String> {
        let convert = |value: f64| -> Result<A, String> {
            A::from(value).ok_or_else(|| format!("{value} is not representable"))
        };
        let outcome = if reward > convert(0.8)? {
            EpisodeOutcome::Success
        } else if reward > convert(0.5)? {
            EpisodeOutcome::PartialSuccess
        } else if reward > convert(0.2)? {
            EpisodeOutcome::Neutral
        } else if reward > convert(-0.2)? {
            EpisodeOutcome::Failure
        } else {
            EpisodeOutcome::CriticalFailure
        };

        Ok(EpisodeContext {
            episode_id: self.statistics.training_episodes as u64,
            start_time: self.episode_start,
            // Real elapsed time since the current episode began.
            duration: self.episode_start.elapsed(),
            // The first reward recorded in this episode; for the first
            // experience of an episode that is this reward itself.
            initial_performance: self.episode_initial_performance.unwrap_or(reward),
            final_performance: reward,
            // Real count of experiences accumulated in this episode, including
            // the one being created.
            adaptation_count: self.episode_adaptation_count + 1,
            outcome,
        })
    }

    /// Calculates priority for experience replay
    fn calculate_experience_priority(
        &self,
        state: &MetaState<A>,
        action: &MetaAction<A>,
        reward: A,
    ) -> A {
        // Every branch returns a strictly positive priority: a zero priority
        // would make the experience unsamplable under prioritized replay and
        // would break the importance-sampling normalisation.
        let floor = scalar_or(1e-6, A::zero());
        let priority = match self.config.replay_config.priority_method {
            // Temporal-difference error: how wrong the meta-model's current
            // value estimate for this state/action was. This is the canonical
            // PER priority and is only available because the model can score
            // the pair it just acted on.
            PriorityMethod::TDError => {
                match self.meta_model.estimate_reward(state, action) {
                    Some(predicted) => (reward - predicted).abs(),
                    // No estimate yet (untrained arm): treat it as maximally
                    // surprising so it gets replayed, which is what PER does
                    // for unvisited transitions.
                    None => A::one(),
                }
            }
            // Surprise relative to the rewards seen so far: |r - mean| in units
            // of the running average magnitude.
            PriorityMethod::Surprise => {
                let mean = self.statistics.avg_reward_per_episode;
                let scale = mean.abs().max(A::one());
                (reward - mean).abs() / scale
            }
            // Magnitude of the adaptation that was actually applied.
            PriorityMethod::GradientMagnitude => action
                .adaptation_magnitudes
                .iter()
                .fold(A::zero(), |acc, m| acc + m.abs()),
            // Improvement over the best episode reward observed so far.
            PriorityMethod::LossImprovement => {
                (reward - self.statistics.best_episode_reward).max(A::zero())
            }
            // Uniform random priority: the PER ablation baseline.
            PriorityMethod::Random => scalar_or(thread_rng().gen_range(0.0..1.0), A::one()),
        };
        priority.max(floor)
    }

    /// Test-only: records one synthetic experience so buffer bounds can be
    /// exercised without standing up a whole optimizer.
    #[cfg(test)]
    pub(crate) fn record_probe_experience_for_test(&mut self, reward: A) -> Result<(), String> {
        let state = MetaState {
            performance_metrics: vec![reward],
            resource_state: vec![A::one()],
            drift_indicators: vec![A::zero()],
            adaptation_history: 0,
            timestamp: Instant::now(),
        };
        let action = MetaAction {
            adaptation_magnitudes: vec![reward],
            adaptation_types: vec![AdaptationType::LearningRate],
            learning_rate_change: reward,
            buffer_size_change: A::zero(),
            timestamp: Instant::now(),
        };
        self.update_experience(state, action, reward)
    }

    /// Test-only count of retained experiences.
    #[cfg(test)]
    pub(crate) fn experience_count_for_test(&self) -> usize {
        self.experience_buffer.experiences.len()
    }

    /// Characteristic vector describing the domain this learner is currently
    /// training on: the reported resource state followed by the drift
    /// indicators, i.e. exactly the context signals the owning optimizer
    /// publishes through [`Self::update_context_signals`].
    fn target_domain_characteristics(&self) -> Vec<A> {
        let mut characteristics = self.context_resource_state.clone();
        characteristics.extend(self.context_drift_indicators.iter().copied());
        characteristics
    }

    /// Registers a source domain whose experiences may be replayed into this
    /// learner when transfer learning is enabled.
    ///
    /// Returns an error when `MetaLearningConfig::enable_transfer_learning` is
    /// off, rather than accepting the source and never using it (CF1: that flag
    /// previously had no reader at all, so transfer learning was neither on nor
    /// off — it simply did not exist).
    pub fn register_transfer_source(
        &mut self,
        source_id: String,
        experiences: Vec<MetaExperience<A>>,
        source_characteristics: Vec<A>,
    ) -> Result<(), String> {
        if !self.config.enable_transfer_learning {
            return Err(
                "MetaLearningConfig::enable_transfer_learning is disabled, so no transfer \
                 source can be registered"
                    .to_string(),
            );
        }
        self.transfer_learning
            .register_source(source_id, experiences, source_characteristics);
        Ok(())
    }

    /// Measured transfer-learning outcomes, or `None` when transfer learning is
    /// disabled.
    pub fn transfer_metrics(&self) -> Option<&TransferMetrics<A>> {
        self.config
            .enable_transfer_learning
            .then(|| self.transfer_learning.metrics())
    }

    /// Triggers meta-learning update
    fn trigger_learning(&mut self) -> Result<(), String> {
        // Sample experiences for training
        let mut training_batch = self
            .experience_buffer
            .sample_batch(self.config.replay_config.batch_size)?;

        // Augment with similarity-weighted source-domain experiences when
        // transfer learning is enabled and a source has been registered (CF1).
        if self.config.enable_transfer_learning && self.transfer_learning.source_domain_count() > 0
        {
            let reward_before = self.meta_model.performance_metrics.prediction_accuracy;
            let target_characteristics = self.target_domain_characteristics();
            let transferred = self.transfer_learning.select_transfer_batch(
                target_characteristics,
                self.config.replay_config.batch_size,
            );
            if !transferred.is_empty() {
                training_batch.extend(transferred);
                self.meta_model.train_on_batch(&training_batch)?;
                // fall through to the weighted pass below for the local batch
                let reward_after = self.meta_model.performance_metrics.prediction_accuracy;
                self.transfer_learning
                    .record_transfer_outcome(reward_before, reward_after);
                self.statistics.transfer_success_rate = self
                    .transfer_learning
                    .metrics()
                    .success_rate
                    .unwrap_or_else(A::zero);
            }
        }

        // Train meta-model, applying the importance-sampling weights recorded by
        // the sampling step above when the correction is enabled.
        let weights: HashMap<u64, A> = training_batch
            .iter()
            .filter_map(|experience| {
                self.experience_buffer
                    .importance_weight(experience.id)
                    .map(|weight| (experience.id, weight))
            })
            .collect();
        self.meta_model
            .train_on_weighted_batch(&training_batch, |id| {
                weights.get(&id).copied().unwrap_or_else(A::one)
            })?;

        // Update strategy selection
        self.strategy_selector
            .update_from_experiences(&training_batch)?;

        // Update statistics from the batch that was actually trained on, so the
        // reward figures are measurements rather than initial zeros.
        self.statistics.training_episodes += 1;
        if !training_batch.is_empty() {
            let count = A::from(training_batch.len())
                .ok_or_else(|| "batch size is not representable".to_string())?;
            let total = training_batch
                .iter()
                .fold(A::zero(), |acc, experience| acc + experience.reward);
            let mean = total / count;
            self.statistics.avg_reward_per_episode = mean;
            for experience in &training_batch {
                if experience.reward > self.statistics.best_episode_reward {
                    self.statistics.best_episode_reward = experience.reward;
                }
            }
            // Learning progress is the model's measured prediction accuracy —
            // how well it has learned to anticipate reward.
            self.statistics.learning_progress =
                self.meta_model.performance_metrics.prediction_accuracy;
            self.statistics.strategy_selection_accuracy =
                self.meta_model.performance_metrics.decision_quality;
            self.statistics.replay_effectiveness = mean;
        }

        // A training round closes the current episode: start a fresh one so the
        // next episode's duration and adaptation count are measured from here.
        self.episode_start = Instant::now();
        self.episode_initial_performance = None;
        self.episode_adaptation_count = 0;

        Ok(())
    }

    /// Recommends adaptations based on current state.
    ///
    /// `_current_data` is accepted for signature stability but is not read: the
    /// meta-state the bandit consumes is `[performance, resource, drift]` with a
    /// layout the feature scaler in [`super::meta_bandit`] is fitted against,
    /// and per-batch data characteristics have no slot in it. The owning
    /// optimizer already derives its data statistics separately
    /// (`compute_data_statistics`), and resource/drift signals reach the
    /// meta-learner through [`Self::update_context_signals`]. Feeding data
    /// characteristics into the bandit would require widening `MetaState` and
    /// refitting the scaler, which is a design change rather than a wiring fix.
    pub fn recommend_adaptations(
        &mut self,
        _current_data: &[StreamingDataPoint<A>],
        performance_tracker: &PerformanceTracker<A>,
    ) -> Result<Vec<Adaptation<A>>, String> {
        // Extract current meta-state
        let current_state = self.extract_meta_state(performance_tracker)?;

        // Use meta-model to predict best action
        let predicted_action = self.meta_model.predict_action(&current_state)?;

        // Select appropriate strategy
        let strategy = self.strategy_selector.select_strategy(&current_state)?;

        // Generate adaptations based on prediction and strategy
        let adaptations =
            self.generate_adaptations_from_prediction(&predicted_action, &strategy)?;

        Ok(adaptations)
    }

    /// Extracts meta-state from current situation
    /// Deliberately takes no data batch: `MetaState`'s feature layout
    /// (performance, resource and drift signals) is what the bandit's feature
    /// scaler is fitted against and has no slot for per-batch data
    /// characteristics, so a batch argument could only be discarded.
    fn extract_meta_state(
        &self,
        performance_tracker: &PerformanceTracker<A>,
    ) -> Result<MetaState<A>, String> {
        // Get recent performance
        let recent_performance = performance_tracker.get_recent_performance(5);
        let performance_metrics = if !recent_performance.is_empty() {
            vec![
                recent_performance[0].loss,
                recent_performance[0].accuracy.unwrap_or(A::zero()),
                recent_performance[0].convergence_rate.unwrap_or(A::zero()),
            ]
        } else {
            vec![A::zero(), A::zero(), A::zero()]
        };

        // Resource and drift signals as reported by the owning optimizer. An
        // empty vector honestly means "not reported" rather than a stand-in
        // value that the bandit would learn a weight for.
        let resource_state = self.context_resource_state.clone();
        let drift_indicators = self.context_drift_indicators.clone();

        Ok(MetaState {
            performance_metrics,
            resource_state,
            drift_indicators,
            adaptation_history: self.statistics.total_experiences,
            timestamp: Instant::now(),
        })
    }

    /// Generates adaptations from model prediction
    fn generate_adaptations_from_prediction(
        &self,
        predicted_action: &MetaAction<A>,
        _strategy: &AdaptationStrategy<A>,
    ) -> Result<Vec<Adaptation<A>>, String> {
        let mut adaptations = Vec::new();

        // Generate adaptations based on predicted action
        for (i, &magnitude) in predicted_action.adaptation_magnitudes.iter().enumerate() {
            if magnitude.abs() > try_scalar_str::<A, _>(0.05)? {
                // Minimum threshold
                let adaptation_type = if i < predicted_action.adaptation_types.len() {
                    predicted_action.adaptation_types[i].clone()
                } else {
                    AdaptationType::LearningRate // Default
                };

                let adaptation = Adaptation {
                    adaptation_type,
                    magnitude,
                    target_component: "meta_learner".to_string(),
                    parameters: std::collections::HashMap::new(),
                    priority: if magnitude.abs() > try_scalar_str::<A, _>(0.3)? {
                        AdaptationPriority::High
                    } else {
                        AdaptationPriority::Normal
                    },
                    timestamp: Instant::now(),
                };

                adaptations.push(adaptation);
            }
        }

        Ok(adaptations)
    }

    /// Applies adaptation to meta-learning system
    pub fn apply_adaptation(&mut self, adaptation: &Adaptation<A>) -> Result<(), String> {
        match adaptation.adaptation_type {
            AdaptationType::MetaLearning => {
                // Adjust meta-learning parameters
                let new_rate = self.learning_rate_adapter.current_rate + adaptation.magnitude;
                self.learning_rate_adapter.update_rate(new_rate)?;
            }
            _ => {
                // Handle other adaptation types
            }
        }

        Ok(())
    }

    /// Gets meta-learning effectiveness score
    pub fn get_effectiveness_score(&self) -> f32 {
        self.statistics.learning_progress.to_f32().unwrap_or(0.0)
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> MetaLearningDiagnostics {
        MetaLearningDiagnostics {
            total_experiences: self.statistics.total_experiences,
            training_episodes: self.statistics.training_episodes,
            current_learning_rate: self
                .learning_rate_adapter
                .current_rate
                .to_f64()
                .unwrap_or(0.0),
            model_accuracy: self
                .meta_model
                .performance_metrics
                .prediction_accuracy
                .to_f64()
                .unwrap_or(0.0),
            strategy_count: self.strategy_selector.strategies.len(),
            transfer_success_rate: self
                .statistics
                .transfer_success_rate
                .to_f64()
                .unwrap_or(0.0),
        }
    }
}

/// Exponent `beta` of the prioritized-replay importance-sampling correction
/// (Schaul et al., "Prioritized Experience Replay", ICLR 2016). A full
/// annealing schedule needs a training-progress signal the buffer does not
/// have, so the fully-corrected value is used.
const IMPORTANCE_SAMPLING_BETA: f64 = 1.0;

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> ExperienceBuffer<A> {
    /// `capacity` comes from `MetaLearningConfig::experience_buffer_size` (CF1).
    /// Both the main buffer and the priority queue used to be hardcoded to
    /// 10 000 / 1 000 entries, so configuring a 1 000-experience buffer had no
    /// effect whatsoever.
    fn new(config: &ExperienceReplayConfig, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            config: config.clone(),
            experiences: VecDeque::with_capacity(capacity.min(64 * 1024)),
            priority_queue: VecDeque::new(),
            importance_weights: HashMap::new(),
            capacity,
        }
    }

    fn add_experience(&mut self, experience: MetaExperience<A>) -> Result<(), String> {
        // Add to main buffer, bounded by the configured capacity (CF1).
        while self.experiences.len() >= self.capacity {
            if let Some(evicted) = self.experiences.pop_front() {
                self.importance_weights.remove(&(evicted.id as usize));
            } else {
                break;
            }
        }
        self.experiences.push_back(experience.clone());

        // Add to priority queue if using prioritized replay
        if self.config.enable_prioritized_replay {
            let priority = experience.priority;
            self.priority_queue.push_back((experience, priority));

            // Highest priority first.
            self.priority_queue
                .make_contiguous()
                .sort_by(|a, b| crate::utils::total_order(&b.1, &a.1));

            // The priority queue is a view over the buffer, so it is bounded by
            // the same configured capacity rather than a separate constant.
            while self.priority_queue.len() > self.capacity {
                self.priority_queue.pop_back();
            }
        }

        Ok(())
    }

    /// Importance-sampling weight for a sampled experience under prioritized
    /// replay: `w_i = (1 / (N * P(i)))^beta`, normalised by the largest weight
    /// in the batch so the maximum is 1 and the correction only ever scales
    /// updates down.
    ///
    /// Only computed when `ExperienceReplayConfig::importance_sampling` is set
    /// (CF1); that field previously had no reader, and `importance_weights` was
    /// a `#[allow(dead_code)]`-adjacent field nothing ever wrote, so
    /// prioritized replay ran with its sampling bias entirely uncorrected.
    fn record_importance_weights(&mut self, batch: &[MetaExperience<A>], total_priority: A) {
        self.importance_weights.clear();
        if !self.config.importance_sampling || batch.is_empty() || total_priority <= A::zero() {
            return;
        }
        let n = match A::from(self.experiences.len().max(1)) {
            Some(n) => n,
            None => return,
        };
        let beta = match A::from(IMPORTANCE_SAMPLING_BETA) {
            Some(beta) => beta,
            None => return,
        };

        let mut raw: Vec<(usize, A)> = Vec::with_capacity(batch.len());
        let mut max_weight = A::zero();
        for experience in batch {
            let probability = experience.priority / total_priority;
            if probability <= A::zero() {
                continue;
            }
            let weight = (A::one() / (n * probability)).powf(beta);
            if weight > max_weight {
                max_weight = weight;
            }
            raw.push((experience.id as usize, weight));
        }
        if max_weight <= A::zero() {
            return;
        }
        for (id, weight) in raw {
            self.importance_weights.insert(id, weight / max_weight);
        }
    }

    /// Normalised importance-sampling weight recorded for `experience_id` by the
    /// most recent [`Self::sample_batch`] call, if importance sampling is on.
    fn importance_weight(&self, experience_id: u64) -> Option<A> {
        self.importance_weights
            .get(&(experience_id as usize))
            .copied()
    }

    fn sample_batch(&mut self, batch_size: usize) -> Result<Vec<MetaExperience<A>>, String> {
        if self.experiences.is_empty() {
            return Ok(Vec::new());
        }

        let mut batch = Vec::with_capacity(batch_size);
        let total_priority: A = self.experiences.iter().map(|e| e.priority).sum();

        if self.config.enable_prioritized_replay && !self.priority_queue.is_empty() {
            // Sample from priority queue
            for _ in 0..batch_size.min(self.priority_queue.len()) {
                if let Some((experience, _)) = self.priority_queue.pop_front() {
                    batch.push(experience);
                }
            }
        } else {
            // Random sampling
            for _ in 0..batch_size.min(self.experiences.len()) {
                let idx = thread_rng().gen_range(0..self.experiences.len());
                if let Some(experience) = self.experiences.get(idx) {
                    batch.push(experience.clone());
                }
            }
        }

        self.record_importance_weights(&batch, total_priority);
        Ok(batch)
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> MetaModel<A> {
    fn new(complexity: MetaModelComplexity) -> Result<Self, String> {
        let parameters = match complexity {
            MetaModelComplexity::Low => MetaModelParameters {
                weights: vec![vec![try_scalar_str::<A, _>(0.1)?; 10]; 2],
                biases: vec![A::zero(); 10],
                // Normalised-LMS step size (stable for 0 < mu < 2), not an
                // unnormalised SGD rate.
                learning_rate: A::from(0.5).unwrap_or_else(A::one),
                regularization: RegularizationParams {
                    l1_lambda: try_scalar_str::<A, _>(0.001)?,
                    l2_lambda: try_scalar_str::<A, _>(0.001)?,
                    dropout_rate: try_scalar_str::<A, _>(0.1)?,
                    early_stopping_patience: 10,
                },
                optimization: OptimizationParams {
                    momentum: try_scalar_str::<A, _>(0.9)?,
                    beta1: try_scalar_str::<A, _>(0.9)?,
                    beta2: try_scalar_str::<A, _>(0.999)?,
                    epsilon: try_scalar_str::<A, _>(1e-8)?,
                    grad_clip_threshold: try_scalar_str::<A, _>(1.0)?,
                },
            },
            _ => MetaModelParameters {
                weights: vec![vec![try_scalar_str::<A, _>(0.1)?; 50]; 3],
                biases: vec![A::zero(); 50],
                // A larger model gets a more conservative NLMS step size.
                learning_rate: A::from(0.3).unwrap_or_else(A::one),
                regularization: RegularizationParams {
                    l1_lambda: try_scalar_str::<A, _>(0.0001)?,
                    l2_lambda: try_scalar_str::<A, _>(0.0001)?,
                    dropout_rate: try_scalar_str::<A, _>(0.2)?,
                    early_stopping_patience: 20,
                },
                optimization: OptimizationParams {
                    momentum: try_scalar_str::<A, _>(0.9)?,
                    beta1: try_scalar_str::<A, _>(0.9)?,
                    beta2: try_scalar_str::<A, _>(0.999)?,
                    epsilon: try_scalar_str::<A, _>(1e-8)?,
                    grad_clip_threshold: try_scalar_str::<A, _>(1.0)?,
                },
            },
        };

        let arms: Vec<BanditArm<A>> = arm_table()
            .into_iter()
            .map(|(prefix, adaptation_type, magnitude)| {
                BanditArm::new(format!("{prefix}:{magnitude}"), adaptation_type, magnitude)
            })
            .collect();

        Ok(Self {
            parameters,
            training_history: VecDeque::with_capacity(1000),
            performance_metrics: ModelPerformanceMetrics {
                // All five metrics start at zero: nothing has been measured yet,
                // and a 0.5 seed would read as "50% accurate" before the model
                // has seen a single experience.
                prediction_accuracy: A::zero(),
                decision_quality: A::zero(),
                adaptation_effectiveness: A::zero(),
                transfer_success_rate: A::zero(),
                generalization_performance: A::zero(),
            },
            feature_importance: Vec::new(),
            arms,
            training_steps: 0,
            feature_scaler: FeatureScaler::default(),
        })
    }

    /// Trains the contextual bandit on a batch of observed experiences.
    ///
    /// ML2: this used to compute the batch's mean reward, nudge the learning
    /// rate by ±1%, and assign that mean reward directly to
    /// `prediction_accuracy` — no model parameter was ever touched, so nothing
    /// was learned and "accuracy" was a relabelled reward.
    ///
    /// Each experience now performs a real stochastic-gradient step on the
    /// squared error between the arm's predicted reward and the observed
    /// reward, and `prediction_accuracy` is measured from the *pre-update*
    /// prediction error — a genuine held-out-by-one-step accuracy.
    fn train_on_batch(&mut self, batch: &[MetaExperience<A>]) -> Result<(), String> {
        self.train_on_weighted_batch(batch, |_| A::one())
    }

    /// Train on `batch`, scaling each experience's update by
    /// `weight_of(experience_id)`.
    ///
    /// This is what makes `ExperienceReplayConfig::importance_sampling` real:
    /// prioritized replay deliberately over-samples high-priority transitions,
    /// which biases the gradient, and the importance-sampling weight
    /// `w_i = (1/(N*P(i)))^beta` corrects for exactly that over-sampling
    /// (Schaul et al., "Prioritized Experience Replay", ICLR 2016). Computing
    /// the weight without applying it to the update would leave the bias in
    /// place.
    fn train_on_weighted_batch(
        &mut self,
        batch: &[MetaExperience<A>],
        weight_of: impl Fn(u64) -> A,
    ) -> Result<(), String> {
        if batch.is_empty() {
            return Ok(());
        }

        let training_started = Instant::now();

        // Fold the batch into the feature scaler before training on it, so every
        // experience in the batch is standardised against the same statistics.
        for experience in batch {
            self.feature_scaler
                .observe(&state_features(&experience.state));
        }

        let mut squared_error_total = A::zero();
        let mut scale_total = A::zero();
        let mut trained = 0usize;

        for experience in batch {
            let features = self
                .feature_scaler
                .standardize(&state_features(&experience.state));
            let arm = match arm_index_for(&experience.action) {
                Some(index) => index,
                // An experience whose action does not correspond to any arm the
                // model can take carries no gradient for it.
                None => continue,
            };

            let predicted = self.arms[arm].predict(&features);
            let error = predicted - experience.reward;
            squared_error_total = squared_error_total + error * error;
            scale_total = scale_total + experience.reward.abs();

            // Importance-sampling correction: scale this transition's step by its
            // weight (1.0 when the correction is disabled).
            let learning_rate = self.parameters.learning_rate * weight_of(experience.id);
            let l2 = self.parameters.regularization.l2_lambda;
            self.arms[arm].sgd_step(&features, error, learning_rate, l2);
            self.arms[arm].observe(experience.reward);
            trained += 1;
        }

        if trained == 0 {
            return Ok(());
        }

        let count =
            A::from(trained).ok_or_else(|| format!("batch size {trained} is not representable"))?;
        let rmse = (squared_error_total / count).sqrt();
        let mean_scale = (scale_total / count)
            .max(A::from(1e-8).ok_or_else(|| "1e-8 is not representable".to_string())?);

        // Accuracy as one minus the normalised prediction error, clamped into
        // [0, 1]. This is a real measurement of how well the model predicts
        // reward, not the reward itself.
        let accuracy = (A::one() - (rmse / mean_scale).min(A::one())).max(A::zero());
        self.performance_metrics.prediction_accuracy = accuracy;

        // Decision quality: the fraction of experiences whose reward beat the
        // running average, which is what "did the chosen action help" means
        // with the data available here.
        let average_reward = self.average_observed_reward();
        let better = batch
            .iter()
            .filter(|experience| experience.reward > average_reward)
            .count();
        if let Some(fraction) = A::from(better as f64 / batch.len() as f64) {
            self.performance_metrics.decision_quality = fraction;
        }

        // Track training history so the field is real state, not dead weight.
        self.training_steps += 1;
        if self.training_history.len() >= 1000 {
            self.training_history.pop_front();
        }
        self.training_history.push_back(TrainingEpisode {
            episode_id: self.training_steps as u64,
            training_loss: rmse,
            // No held-out split exists in a streaming setting, so validation and
            // training figures are the same measurement; they are not two
            // independently-invented numbers.
            validation_loss: rmse,
            training_accuracy: accuracy,
            validation_accuracy: accuracy,
            duration: training_started.elapsed(),
            timestamp: Instant::now(),
        });

        // Real learning-rate schedule: anneal towards a floor once the model is
        // already accurate (fine-tuning), hold otherwise. The floor matters —
        // an unbounded decay would eventually freeze learning entirely, which is
        // indistinguishable from not learning at all.
        if accuracy > A::from(0.9).unwrap_or_else(A::one) {
            let decay = A::from(0.99).unwrap_or_else(A::one);
            let floor = A::from(NLMS_STEP_FLOOR).unwrap_or_else(A::zero);
            self.parameters.learning_rate = (self.parameters.learning_rate * decay).max(floor);
        }

        // Feature importance is the mean absolute weight across arms — a real,
        // if simple, attribution.
        self.feature_importance = self.mean_absolute_weights();

        Ok(())
    }

    /// Selects an action for the given state using the learned bandit.
    ///
    /// ML1: this used to return the constants `[0.1, -0.05]` with
    /// `learning_rate_change: 0.01` and `buffer_size_change: 5.0` for every
    /// state — `state` was accepted and never read, so the meta-learner
    /// recommended the identical adaptation forever.
    ///
    /// The action is now the arm with the highest predicted reward for this
    /// state's feature vector, so a different state genuinely yields a
    /// different recommendation.
    fn predict_action(&self, state: &MetaState<A>) -> Result<MetaAction<A>, String> {
        let features = self.feature_scaler.standardize(&state_features(state));
        if self.arms.is_empty() {
            return Err("meta-model has no action arms".to_string());
        }

        let mut best_index = 0usize;
        let mut best_value = self.arms[0].predict(&features);
        for (index, arm) in self.arms.iter().enumerate().skip(1) {
            let value = arm.predict(&features);
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }

        let arm = &self.arms[best_index];
        let magnitude = arm
            .magnitude()
            .ok_or_else(|| "arm magnitude is not representable".to_string())?;

        let (learning_rate_change, buffer_size_change) = match arm.adaptation_type {
            AdaptationType::LearningRate => (magnitude, A::zero()),
            AdaptationType::BufferSize => (A::zero(), magnitude),
            _ => (A::zero(), A::zero()),
        };

        Ok(MetaAction {
            adaptation_magnitudes: vec![magnitude],
            adaptation_types: vec![arm.adaptation_type.clone()],
            learning_rate_change,
            buffer_size_change,
            timestamp: Instant::now(),
        })
    }

    /// The model's current value estimate for taking `action` in `state`.
    ///
    /// Returns `None` when the action does not map to a known arm or that arm
    /// has never been trained, so callers can distinguish "no estimate yet" from
    /// "estimated zero" — the difference matters for the temporal-difference
    /// replay priority, where an untrained arm must count as maximally
    /// surprising rather than perfectly predicted.
    fn estimate_reward(&self, state: &MetaState<A>, action: &MetaAction<A>) -> Option<A> {
        let index = arm_index_for(action)?;
        let arm = self.arms.get(index)?;
        if arm.pulls == 0 {
            return None;
        }
        let features = self.feature_scaler.standardize(&state_features(state));
        Some(arm.predict(&features))
    }

    /// Mean reward observed across every arm, or zero before any observation.
    fn average_observed_reward(&self) -> A {
        let total_pulls: usize = self.arms.iter().map(|arm| arm.pulls).sum();
        if total_pulls == 0 {
            return A::zero();
        }
        let Some(count) = A::from(total_pulls) else {
            return A::zero();
        };
        let total: A = self
            .arms
            .iter()
            .fold(A::zero(), |acc, arm| acc + arm.reward_total);
        total / count
    }

    /// Mean absolute weight per feature across all arms.
    fn mean_absolute_weights(&self) -> Vec<A> {
        let width = self
            .arms
            .iter()
            .map(|arm| arm.weights.len())
            .max()
            .unwrap_or(0);
        let Some(arm_count) = A::from(self.arms.len().max(1)) else {
            return Vec::new();
        };
        (0..width)
            .map(|index| {
                let total = self.arms.iter().fold(A::zero(), |acc, arm| {
                    acc + arm
                        .weights
                        .get(index)
                        .map(|w| w.abs())
                        .unwrap_or_else(A::zero)
                });
                total / arm_count
            })
            .collect()
    }

    /// Number of times each arm has been trained on, keyed by arm label.
    pub fn arm_pull_counts(&self) -> Vec<(String, usize)> {
        self.arms
            .iter()
            .map(|arm| (arm.label.clone(), arm.pulls))
            .collect()
    }
}

// The contextual-bandit arms, the feature standardiser and the state/action
// encoding live in `super::meta_bandit`.

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StrategySelector<A> {
    fn new() -> Self {
        let mut strategies = HashMap::new();

        // Add default strategies
        strategies.insert(
            "conservative".to_string(),
            AdaptationStrategy {
                name: "conservative".to_string(),
                parameters: HashMap::new(),
                strategy_type: StrategyType::Conservative,
                conditions: Vec::new(),
                expected_outcomes: vec![scalar_or(0.05, A::zero())],
            },
        );

        strategies.insert(
            "aggressive".to_string(),
            AdaptationStrategy {
                name: "aggressive".to_string(),
                parameters: HashMap::new(),
                strategy_type: StrategyType::Aggressive,
                // Prior expected improvement, used only until the arm has been
                // used at least once and has a measured average.
                expected_outcomes: vec![A::from(0.2).unwrap_or_else(A::zero)],
                conditions: Vec::new(),
            },
        );

        Self {
            strategies,
            strategy_performance: HashMap::new(),
            selection_policy: SelectionPolicy::EpsilonGreedy { epsilon: 0.1 },
            exploration_params: ExplorationParams {
                exploration_rate: scalar_or(0.1, A::zero()),
                exploration_decay: scalar_or(0.99, A::zero()),
                min_exploration_rate: scalar_or(0.01, A::zero()),
                curiosity_weight: scalar_or(0.1, A::zero()),
                novelty_weight: scalar_or(0.1, A::zero()),
            },
        }
    }

    /// Selects a strategy for the given state using the configured policy.
    ///
    /// ML3: this used to look up the literal key `"balanced"`, which the
    /// constructor never inserted — so the lookup always missed and the fallback
    /// `self.strategies.values().next()` returned an arbitrary `HashMap` entry,
    /// which is not even deterministic across runs. The `_state` argument,
    /// `selection_policy` and `exploration_params` were all unread.
    ///
    /// Selection is now genuinely policy-driven over the measured per-strategy
    /// performance, and the ordering is deterministic (ties broken by name)
    /// rather than dependent on hash iteration order.
    fn select_strategy(&self, state: &MetaState<A>) -> Result<AdaptationStrategy<A>, String> {
        if self.strategies.is_empty() {
            return Err("No strategies available".to_string());
        }

        // Deterministic candidate order.
        let mut names: Vec<&String> = self.strategies.keys().collect();
        names.sort();

        // Value of each arm: its measured average improvement, or the strategy's
        // declared expected outcome while it has never been used (which is the
        // only prior available).
        let scored: Vec<(&String, A, usize)> = names
            .iter()
            .map(|name| {
                let (value, usage) = match self.strategy_performance.get(*name) {
                    Some(performance) if performance.usage_count > 0 => {
                        (performance.avg_improvement, performance.usage_count)
                    }
                    _ => {
                        let prior = self
                            .strategies
                            .get(*name)
                            .and_then(|strategy| strategy.expected_outcomes.first().copied())
                            .unwrap_or_else(A::zero);
                        (prior, 0)
                    }
                };
                (*name, value, usage)
            })
            .collect();

        let chosen = match &self.selection_policy {
            SelectionPolicy::EpsilonGreedy { epsilon } => {
                let effective_epsilon = self
                    .exploration_params
                    .exploration_rate
                    .to_f64()
                    .unwrap_or(*epsilon)
                    .max(
                        self.exploration_params
                            .min_exploration_rate
                            .to_f64()
                            .unwrap_or(0.0),
                    )
                    .clamp(0.0, 1.0);
                if thread_rng().gen_range(0.0..1.0) < effective_epsilon {
                    // Explore: prefer the least-used arm, which is the
                    // information-maximising choice.
                    scored
                        .iter()
                        .min_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(b.0)))
                        .map(|entry| entry.0)
                } else {
                    best_by_value(&scored)
                }
            }
            SelectionPolicy::UCB {
                confidence_parameter,
            } => {
                let total_usage: usize = scored.iter().map(|entry| entry.2).sum();
                let total = (total_usage.max(1) as f64).ln();
                let c = *confidence_parameter;
                scored
                    .iter()
                    .max_by(|a, b| {
                        let score_a = ucb_score(a.1, a.2, total, c);
                        let score_b = ucb_score(b.1, b.2, total, c);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| b.0.cmp(a.0))
                    })
                    .map(|entry| entry.0)
            }
            SelectionPolicy::Softmax { temperature } => {
                let temperature = temperature.abs().max(1e-6);
                let values: Vec<f64> = scored
                    .iter()
                    .map(|entry| entry.1.to_f64().unwrap_or(0.0) / temperature)
                    .collect();
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exponentials: Vec<f64> =
                    values.iter().map(|value| (value - max).exp()).collect();
                let total: f64 = exponentials.iter().sum();
                if total <= 0.0 {
                    best_by_value(&scored)
                } else {
                    let mut draw = thread_rng().gen_range(0.0..total);
                    let mut selected = scored.last().map(|entry| entry.0);
                    for (entry, weight) in scored.iter().zip(exponentials.iter()) {
                        draw -= *weight;
                        if draw <= 0.0 {
                            selected = Some(entry.0);
                            break;
                        }
                    }
                    selected
                }
            }
            SelectionPolicy::ThompsonSampling | SelectionPolicy::MultiArmedBandit => {
                // Sample each arm's value perturbed by a scale that shrinks as
                // the arm accumulates evidence — the defining behaviour of
                // posterior sampling.
                scored
                    .iter()
                    .map(|entry| {
                        let scale = 1.0 / ((entry.2 as f64) + 1.0).sqrt();
                        let noise = thread_rng().gen_range(-1.0..1.0) * scale;
                        (entry.0, entry.1.to_f64().unwrap_or(0.0) + noise)
                    })
                    .max_by(|a, b| {
                        a.1.partial_cmp(&b.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| b.0.cmp(a.0))
                    })
                    .map(|entry| entry.0)
            }
            SelectionPolicy::ContextAware => {
                // Score arms by their measured performance *in this state's
                // context bucket*, falling back to the global value.
                let context_key = context_key_for(state);
                scored
                    .iter()
                    .max_by(|a, b| {
                        let value_a = self
                            .strategy_performance
                            .get(a.0)
                            .and_then(|p| p.context_performance.get(&context_key).copied())
                            .unwrap_or(a.1);
                        let value_b = self
                            .strategy_performance
                            .get(b.0)
                            .and_then(|p| p.context_performance.get(&context_key).copied())
                            .unwrap_or(b.1);
                        value_a
                            .partial_cmp(&value_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| b.0.cmp(a.0))
                    })
                    .map(|entry| entry.0)
            }
        };

        let name = chosen.ok_or_else(|| "strategy selection produced no candidate".to_string())?;
        self.strategies
            .get(name)
            .cloned()
            .ok_or_else(|| format!("selected strategy '{name}' is not registered"))
    }

    /// Updates per-strategy reward statistics from observed experiences.
    ///
    /// ML4: this was an unconditional `Ok(())`, so `strategy_performance` stayed
    /// permanently empty and every policy above would have had nothing to choose
    /// on. Each experience now moves the usage count, running average
    /// improvement, best/worst outcomes, success rate, per-context performance
    /// and trend of the strategy that produced it, and decays the exploration
    /// rate towards its floor.
    fn update_from_experiences(&mut self, experiences: &[MetaExperience<A>]) -> Result<(), String> {
        if experiences.is_empty() {
            return Ok(());
        }

        let success_threshold =
            A::from(0.5).ok_or_else(|| "0.5 is not representable".to_string())?;

        for experience in experiences {
            let strategy_name = strategy_name_for(&experience.action);
            if !self.strategies.contains_key(&strategy_name) {
                continue;
            }
            let context_key = context_key_for(&experience.state);
            let reward = experience.reward;

            let entry = self
                .strategy_performance
                .entry(strategy_name)
                .or_insert_with(|| StrategyPerformance {
                    usage_count: 0,
                    success_rate: A::zero(),
                    avg_improvement: A::zero(),
                    best_improvement: reward,
                    worst_outcome: reward,
                    recent_trend: TrendDirection::Stable,
                    context_performance: HashMap::new(),
                });

            let previous_average = entry.avg_improvement;
            entry.usage_count = entry.usage_count.saturating_add(1);
            let count = A::from(entry.usage_count)
                .ok_or_else(|| "usage count is not representable".to_string())?;

            // Incremental mean.
            entry.avg_improvement = previous_average + (reward - previous_average) / count;

            // Incremental success rate over the same counter.
            let success = if reward > success_threshold {
                A::one()
            } else {
                A::zero()
            };
            entry.success_rate = entry.success_rate + (success - entry.success_rate) / count;

            if reward > entry.best_improvement {
                entry.best_improvement = reward;
            }
            if reward < entry.worst_outcome {
                entry.worst_outcome = reward;
            }

            entry.recent_trend = if entry.avg_improvement > previous_average {
                TrendDirection::Improving
            } else if entry.avg_improvement < previous_average {
                TrendDirection::Declining
            } else {
                TrendDirection::Stable
            };

            // Per-context running mean.
            let context_entry = entry
                .context_performance
                .entry(context_key)
                .or_insert_with(|| reward);
            let smoothing = A::from(0.2).unwrap_or_else(A::one);
            *context_entry = smoothing * reward + (A::one() - smoothing) * *context_entry;
        }

        // Real exploration decay: each learning round moves the rate towards its
        // configured floor.
        let decayed =
            self.exploration_params.exploration_rate * self.exploration_params.exploration_decay;
        self.exploration_params.exploration_rate =
            decayed.max(self.exploration_params.min_exploration_rate);

        Ok(())
    }

    /// Measured performance of a named strategy, if it has been used.
    pub fn strategy_performance_for(&self, name: &str) -> Option<&StrategyPerformance<A>> {
        self.strategy_performance.get(name)
    }

    /// Current exploration rate.
    pub fn exploration_rate(&self) -> A {
        self.exploration_params.exploration_rate
    }
}

/// Picks the highest-valued arm, breaking ties by name for determinism.
fn best_by_value<'a, A: Float + Send + Sync>(
    scored: &'a [(&'a String, A, usize)],
) -> Option<&'a String> {
    scored
        .iter()
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(a.0))
        })
        .map(|entry| entry.0)
}

/// UCB1 score: mean value plus an exploration bonus that shrinks as the arm is
/// used more.
fn ucb_score<A: Float + Send + Sync>(
    value: A,
    usage: usize,
    log_total_usage: f64,
    confidence: f64,
) -> f64 {
    let mean = value.to_f64().unwrap_or(0.0);
    if usage == 0 {
        // An untried arm has an unbounded bonus, so it is always tried first.
        return f64::INFINITY;
    }
    mean + confidence * (log_total_usage / usage as f64).sqrt()
}

/// Maps a meta-state onto a coarse context bucket key.
///
/// The key is derived from the sign and magnitude band of the loss and the drift
/// level, so states that call for the same kind of response share a bucket.
fn context_key_for<A: Float + Send + Sync>(state: &MetaState<A>) -> String {
    let loss = state
        .performance_metrics
        .first()
        .and_then(|value| value.to_f64())
        .unwrap_or(0.0);
    let drift = state
        .drift_indicators
        .first()
        .and_then(|value| value.to_f64())
        .unwrap_or(0.0);
    let loss_band = if loss <= 0.1 {
        "loss:low"
    } else if loss <= 1.0 {
        "loss:mid"
    } else {
        "loss:high"
    };
    let drift_band = if drift < 0.5 {
        "drift:none"
    } else if drift < 1.5 {
        "drift:warning"
    } else {
        "drift:active"
    };
    format!("{loss_band}|{drift_band}")
}

/// Maps an action onto the strategy whose aggressiveness it matches.
///
/// This is how an observed experience is attributed back to a strategy: the
/// magnitude of the adaptation that was applied tells us whether a conservative
/// or aggressive strategy produced it.
fn strategy_name_for<A: Float + Send + Sync>(action: &MetaAction<A>) -> String {
    let magnitude = action
        .adaptation_magnitudes
        .first()
        .and_then(|value| value.to_f64())
        .map(f64::abs)
        .unwrap_or(0.0);
    if magnitude >= AGGRESSIVE_MAGNITUDE_THRESHOLD {
        "aggressive".to_string()
    } else {
        "conservative".to_string()
    }
}

/// Magnitude at or above which an adaptation is attributed to the aggressive
/// strategy.
const AGGRESSIVE_MAGNITUDE_THRESHOLD: f64 = 0.15;

/// Lower bound on the meta-model's normalised-LMS step size, so the accuracy
/// annealing schedule cannot decay learning to a standstill.
const NLMS_STEP_FLOOR: f64 = 0.01;

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> LearningRateAdapter<A> {
    fn new(initial_rate: f64) -> Self {
        Self {
            current_rate: scalar_or(initial_rate, A::zero()),
            rate_history: VecDeque::with_capacity(100),
            min_rate: scalar_or(1e-6, A::zero()),
            max_rate: scalar_or(0.1, A::zero()),
        }
    }

    fn update_rate(&mut self, new_rate: A) -> Result<(), String> {
        self.current_rate = new_rate.max(self.min_rate).min(self.max_rate);

        if self.rate_history.len() >= 100 {
            self.rate_history.pop_front();
        }
        self.rate_history.push_back(self.current_rate);

        Ok(())
    }
}

/// Diagnostic information for meta-learning
#[derive(Debug, Clone)]
pub struct MetaLearningDiagnostics {
    pub total_experiences: usize,
    pub training_episodes: usize,
    pub current_learning_rate: f64,
    pub model_accuracy: f64,
    pub strategy_count: usize,
    pub transfer_success_rate: f64,
}

#[cfg(test)]
#[path = "meta_learning_regression_tests.rs"]
mod regression_tests;
