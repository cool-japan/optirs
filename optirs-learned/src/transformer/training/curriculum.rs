// Curriculum learning strategies for transformer optimization
//
// This module implements various curriculum learning approaches that progressively
// introduce optimization challenges of increasing difficulty to improve learning.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Curriculum learning strategies
#[derive(Debug, Clone, Copy)]
pub enum CurriculumStrategy {
    /// No curriculum learning
    None,
    /// Difficulty-based progression
    DifficultyProgression,
    /// Diversity-based curriculum
    DiversityBased,
    /// Self-paced learning
    SelfPaced,
    /// Teacher-student curriculum
    TeacherStudent,
    /// Adversarial curriculum
    Adversarial,
    /// Multi-task curriculum
    MultiTask,
    /// Adaptive curriculum
    Adaptive,
}

/// Curriculum learning manager
#[derive(Debug, Clone)]
pub struct CurriculumLearner<T: Float + Debug + Send + Sync + 'static> {
    /// Curriculum strategy
    strategy: CurriculumStrategy,

    /// Curriculum parameters
    curriculum_params: CurriculumParams<T>,

    /// Task difficulty estimator
    difficulty_estimator: TaskDifficultyEstimator<T>,

    /// Learning progress tracker
    progress_tracker: LearningProgressTracker<T>,

    /// Current curriculum state
    curriculum_state: CurriculumState<T>,

    /// Task scheduling policy
    task_scheduler: TaskScheduler<T>,

    /// Performance history
    performance_history: VecDeque<PerformanceRecord<T>>,
}

/// Curriculum parameters
#[derive(Debug, Clone)]
pub struct CurriculumParams<T: Float + Debug + Send + Sync + 'static> {
    /// Initial difficulty threshold
    initial_difficulty: T,

    /// Maximum difficulty threshold
    max_difficulty: T,

    /// Difficulty increment per epoch
    difficulty_increment: T,

    /// Performance threshold for progression
    progression_threshold: T,

    /// Patience for difficulty increases
    patience: usize,

    /// Self-pacing factor
    self_pacing_factor: T,

    /// Diversity weight in curriculum
    diversity_weight: T,

    /// Teacher model confidence threshold
    teacher_confidence: T,
}

/// Task difficulty estimator
#[derive(Debug, Clone)]
pub struct TaskDifficultyEstimator<T: Float + Debug + Send + Sync + 'static> {
    /// Learned difficulty predictor
    difficulty_predictor: DifficultyPredictor<T>,

    /// Feature extractors for tasks
    task_features: HashMap<String, Array1<T>>,

    /// Historical difficulty measurements
    difficulty_history: HashMap<String, Vec<T>>,

    /// Difficulty estimation method
    estimation_method: DifficultyEstimationMethod,
}

/// Learning progress tracker
#[derive(Debug, Clone)]
pub struct LearningProgressTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Performance metrics over time
    performance_timeline: VecDeque<T>,

    /// Learning rate estimates
    learning_rates: VecDeque<T>,

    /// Competency levels for different task types
    competency_levels: HashMap<String, T>,

    /// Progress milestones
    milestones: Vec<ProgressMilestone<T>>,
}

/// Current curriculum state
#[derive(Debug, Clone)]
pub struct CurriculumState<T: Float + Debug + Send + Sync + 'static> {
    /// Current difficulty level
    current_difficulty: T,

    /// Active task types
    active_tasks: Vec<String>,

    /// Recent performance
    recent_performance: T,

    /// Epochs since last difficulty increase
    epochs_since_increase: usize,

    /// Current learning phase
    learning_phase: LearningPhase,

    /// Adaptive parameters
    adaptive_params: HashMap<String, T>,
}

/// Task scheduler for curriculum
#[derive(Debug, Clone)]
pub struct TaskScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Task queue with priorities
    task_queue: VecDeque<ScheduledTask<T>>,

    /// Scheduling policy
    scheduling_policy: SchedulingPolicy,

    /// Task weights for sampling
    task_weights: HashMap<String, T>,

    /// Load balancing factors
    load_balancing: HashMap<String, T>,
}

/// Performance record for curriculum tracking
#[derive(Debug, Clone)]
pub struct PerformanceRecord<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    task_id: String,

    /// Performance score
    performance: T,

    /// Difficulty level when task was attempted
    difficulty_level: T,

    /// Number of training steps
    training_steps: usize,

    /// Timestamp
    timestamp: usize,

    /// Additional metrics
    metrics: HashMap<String, T>,
}

/// Difficulty predictor network
#[derive(Debug, Clone)]
pub struct DifficultyPredictor<T: Float + Debug + Send + Sync + 'static> {
    /// Input features dimension
    input_dim: usize,

    /// Hidden layers
    hidden_layers: Vec<Array2<T>>,

    /// Output layer
    output_layer: Array1<T>,

    /// Training history
    training_history: Vec<(Array1<T>, T)>,
}

/// Progress milestone
#[derive(Debug, Clone)]
pub struct ProgressMilestone<T: Float + Debug + Send + Sync + 'static> {
    /// Milestone name
    name: String,

    /// Performance threshold
    threshold: T,

    /// Whether milestone is achieved
    achieved: bool,

    /// Achievement timestamp
    achieved_at: Option<usize>,
}

/// Scheduled task with priority
#[derive(Debug, Clone)]
pub struct ScheduledTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    task_id: String,

    /// Task priority
    priority: T,

    /// Estimated difficulty
    difficulty: T,

    /// Required competency level
    required_competency: T,

    /// Task parameters
    parameters: HashMap<String, T>,
}

/// Learning phases in curriculum
#[derive(Debug, Clone, Copy)]
pub enum LearningPhase {
    /// Initial exploration phase
    Exploration,
    /// Skill building phase
    SkillBuilding,
    /// Mastery phase
    Mastery,
    /// Transfer phase
    Transfer,
    /// Generalization phase
    Generalization,
}

/// Difficulty estimation methods
#[derive(Debug, Clone, Copy)]
pub enum DifficultyEstimationMethod {
    /// Performance-based estimation
    PerformanceBased,
    /// Feature-based prediction
    FeatureBased,
    /// Gradient-based estimation
    GradientBased,
    /// Uncertainty-based estimation
    UncertaintyBased,
    /// Multi-modal estimation
    MultiModal,
}

/// Scheduling policies
#[derive(Debug, Clone, Copy)]
pub enum SchedulingPolicy {
    /// First-in-first-out
    FIFO,
    /// Priority-based scheduling
    Priority,
    /// Weighted random sampling
    WeightedRandom,
    /// Balanced sampling
    Balanced,
    /// Adaptive scheduling
    Adaptive,
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> CurriculumLearner<T> {
    /// Create new curriculum learner
    pub fn new(strategy: CurriculumStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            curriculum_params: CurriculumParams::default(),
            difficulty_estimator: TaskDifficultyEstimator::new()?,
            progress_tracker: LearningProgressTracker::new(),
            curriculum_state: CurriculumState::new()?,
            task_scheduler: TaskScheduler::new()?,
            performance_history: VecDeque::new(),
        })
    }

    /// Update curriculum based on performance
    pub fn update_curriculum(
        &mut self,
        task_id: &str,
        performance: T,
        training_steps: usize,
    ) -> Result<()> {
        // Record performance
        let record = PerformanceRecord {
            task_id: task_id.to_string(),
            performance,
            difficulty_level: self.curriculum_state.current_difficulty,
            training_steps,
            timestamp: self.performance_history.len(),
            metrics: HashMap::new(),
        };

        self.performance_history.push_back(record);
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Update progress tracker
        self.progress_tracker.update_performance(performance);

        // Update curriculum state based on strategy. Every variant has its own
        // progression rule; none of them share an implementation.
        match self.strategy {
            CurriculumStrategy::None => Ok(()),
            CurriculumStrategy::DifficultyProgression => {
                self.update_difficulty_progression(performance)
            }
            CurriculumStrategy::DiversityBased => {
                self.update_diversity_curriculum(task_id, performance)
            }
            CurriculumStrategy::SelfPaced => self.update_self_paced_curriculum(performance),
            CurriculumStrategy::TeacherStudent => {
                self.update_teacher_student_curriculum(performance)
            }
            CurriculumStrategy::Adversarial => self.update_adversarial_curriculum(performance),
            CurriculumStrategy::MultiTask => {
                self.update_multi_task_curriculum(task_id, performance)
            }
            CurriculumStrategy::Adaptive => self.update_adaptive_curriculum(task_id, performance),
        }
    }

    /// Clamp a difficulty proposal into the configured range.
    fn clamp_difficulty(&self, value: T) -> T {
        value
            .max(self.curriculum_params.initial_difficulty)
            .min(self.curriculum_params.max_difficulty)
    }

    /// Diversity-based curriculum: difficulty is driven by how many *distinct*
    /// tasks have been seen relative to the number of attempts, so a learner
    /// grinding a single task progresses more slowly than one covering many.
    fn update_diversity_curriculum(&mut self, task_id: &str, performance: T) -> Result<()> {
        self.curriculum_state.recent_performance = performance;

        if !self
            .curriculum_state
            .active_tasks
            .iter()
            .any(|t| t == task_id)
        {
            self.curriculum_state.active_tasks.push(task_id.to_string());
        }

        let attempts = self.performance_history.len().max(1);
        let distinct = self.curriculum_state.active_tasks.len();
        let diversity: T = scirs2_core::numeric::NumCast::from(distinct as f64 / attempts as f64)
            .unwrap_or_else(|| T::zero());

        let increment = self.curriculum_params.difficulty_increment
            * self.curriculum_params.diversity_weight
            * diversity;
        self.curriculum_state.current_difficulty =
            self.clamp_difficulty(self.curriculum_state.current_difficulty + increment);
        self.update_learning_phase();

        Ok(())
    }

    /// Teacher-student curriculum: the student only advances once its
    /// performance clears the teacher's confidence bar; falling short pulls the
    /// difficulty back down.
    fn update_teacher_student_curriculum(&mut self, performance: T) -> Result<()> {
        self.curriculum_state.recent_performance = performance;

        let increment = self.curriculum_params.difficulty_increment;
        let proposal = if performance >= self.curriculum_params.teacher_confidence {
            self.curriculum_state.epochs_since_increase = 0;
            self.curriculum_state.current_difficulty + increment
        } else {
            self.curriculum_state.epochs_since_increase += 1;
            let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::one());
            self.curriculum_state.current_difficulty - increment * half
        };

        self.curriculum_state.current_difficulty = self.clamp_difficulty(proposal);
        self.update_learning_phase();
        Ok(())
    }

    /// Adversarial curriculum: the environment pushes back hardest exactly when
    /// the learner is doing well, and relents when it collapses.
    fn update_adversarial_curriculum(&mut self, performance: T) -> Result<()> {
        self.curriculum_state.recent_performance = performance;

        let two: T = scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::one());
        let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::one());
        let threshold = self.curriculum_params.progression_threshold;

        let proposal = if performance > threshold {
            self.curriculum_state.current_difficulty
                + self.curriculum_params.difficulty_increment * two
        } else if performance < threshold * half {
            self.curriculum_state.current_difficulty - self.curriculum_params.difficulty_increment
        } else {
            self.curriculum_state.current_difficulty
        };

        self.curriculum_state.current_difficulty = self.clamp_difficulty(proposal);
        self.update_learning_phase();
        Ok(())
    }

    /// Multi-task curriculum: the difficulty follows the *weakest* task, so a
    /// single lagging task holds the whole curriculum back.
    fn update_multi_task_curriculum(&mut self, task_id: &str, performance: T) -> Result<()> {
        self.curriculum_state.recent_performance = performance;

        let alpha: T = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());
        let competency = self
            .progress_tracker
            .competency_levels
            .get(task_id)
            .copied()
            .unwrap_or(T::zero());
        let updated = competency * (T::one() - alpha) + performance * alpha;
        self.progress_tracker
            .competency_levels
            .insert(task_id.to_string(), updated);

        let weakest = self
            .progress_tracker
            .competency_levels
            .values()
            .cloned()
            .fold(T::infinity(), |a, b| a.min(b));

        let proposal = if weakest.is_finite()
            && weakest > self.curriculum_params.progression_threshold
        {
            self.curriculum_state.current_difficulty + self.curriculum_params.difficulty_increment
        } else {
            self.curriculum_state.current_difficulty
        };

        self.curriculum_state.current_difficulty = self.clamp_difficulty(proposal);
        self.update_learning_phase();
        Ok(())
    }

    /// Get next task according to curriculum
    pub fn get_next_task(&mut self) -> Result<Option<String>> {
        match self.strategy {
            CurriculumStrategy::None => Ok(None),
            _ => Ok(self.task_scheduler.schedule_next_task()),
        }
    }

    /// Update difficulty progression curriculum
    fn update_difficulty_progression(&mut self, performance: T) -> Result<()> {
        self.curriculum_state.recent_performance = performance;

        if performance > self.curriculum_params.progression_threshold {
            self.curriculum_state.epochs_since_increase += 1;

            if self.curriculum_state.epochs_since_increase >= self.curriculum_params.patience {
                // Increase difficulty
                let new_difficulty = (self.curriculum_state.current_difficulty
                    + self.curriculum_params.difficulty_increment)
                    .min(self.curriculum_params.max_difficulty);

                self.curriculum_state.current_difficulty = new_difficulty;
                self.curriculum_state.epochs_since_increase = 0;

                // Update learning phase
                self.update_learning_phase();
            }
        } else {
            self.curriculum_state.epochs_since_increase = 0;
        }

        Ok(())
    }

    /// Update self-paced curriculum
    fn update_self_paced_curriculum(&mut self, performance: T) -> Result<()> {
        let pacing_factor = self.curriculum_params.self_pacing_factor;

        // Adjust difficulty based on performance
        let performance_ratio = performance / self.get_expected_performance();
        let difficulty_adjustment = (performance_ratio - T::one()) * pacing_factor;

        let new_difficulty = (self.curriculum_state.current_difficulty + difficulty_adjustment)
            .max(self.curriculum_params.initial_difficulty)
            .min(self.curriculum_params.max_difficulty);

        self.curriculum_state.current_difficulty = new_difficulty;

        Ok(())
    }

    /// Update adaptive curriculum
    fn update_adaptive_curriculum(&mut self, task_id: &str, performance: T) -> Result<()> {
        // Update task-specific competency
        let competency = self
            .progress_tracker
            .competency_levels
            .get(task_id)
            .copied()
            .unwrap_or(T::zero());

        let alpha = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());
        let new_competency = competency * (T::one() - alpha) + performance * alpha;

        self.progress_tracker
            .competency_levels
            .insert(task_id.to_string(), new_competency);

        // Adapt curriculum parameters
        self.adapt_curriculum_parameters(task_id, performance)?;

        Ok(())
    }

    /// Update learning phase
    fn update_learning_phase(&mut self) {
        let difficulty_ratio =
            self.curriculum_state.current_difficulty / self.curriculum_params.max_difficulty;

        self.curriculum_state.learning_phase = match difficulty_ratio {
            x if x < scirs2_core::numeric::NumCast::from(0.2).unwrap_or_else(|| T::zero()) => {
                LearningPhase::Exploration
            }
            x if x < scirs2_core::numeric::NumCast::from(0.4).unwrap_or_else(|| T::zero()) => {
                LearningPhase::SkillBuilding
            }
            x if x < scirs2_core::numeric::NumCast::from(0.7).unwrap_or_else(|| T::zero()) => {
                LearningPhase::Mastery
            }
            x if x < scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero()) => {
                LearningPhase::Transfer
            }
            _ => LearningPhase::Generalization,
        };
    }

    /// Adapt curriculum parameters based on performance
    fn adapt_curriculum_parameters(&mut self, task_id: &str, performance: T) -> Result<()> {
        // Adapt patience based on task performance variance
        let performance_variance = self.calculate_performance_variance(task_id);
        if performance_variance
            > scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero())
        {
            self.curriculum_params.patience = self.curriculum_params.patience.max(5);
        } else {
            self.curriculum_params.patience =
                (self.curriculum_params.patience.saturating_sub(1)).max(1);
        }

        // Adapt progression threshold based on recent performance trend
        let trend = self.calculate_performance_trend();
        if trend > T::zero() {
            // Performance is improving, can be more aggressive
            self.curriculum_params.progression_threshold =
                (self.curriculum_params.progression_threshold
                    * scirs2_core::numeric::NumCast::from(0.95).unwrap_or_else(|| T::zero()))
                .max(scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()));
        } else {
            // Performance declining, be more conservative
            self.curriculum_params.progression_threshold =
                (self.curriculum_params.progression_threshold
                    * scirs2_core::numeric::NumCast::from(1.05).unwrap_or_else(|| T::zero()))
                .min(scirs2_core::numeric::NumCast::from(0.95).unwrap_or_else(|| T::zero()));
        }

        Ok(())
    }

    /// Calculate performance variance for a task
    fn calculate_performance_variance(&self, task_id: &str) -> T {
        let task_performances: Vec<T> = self
            .performance_history
            .iter()
            .filter(|record| record.task_id == task_id)
            .map(|record| record.performance)
            .collect();

        if task_performances.len() < 2 {
            return T::zero();
        }

        let mean = task_performances
            .iter()
            .cloned()
            .fold(T::zero(), |a, b| a + b)
            / scirs2_core::numeric::NumCast::from(task_performances.len() as f64)
                .unwrap_or_else(|| T::one());

        let variance = task_performances
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |a, b| a + b)
            / scirs2_core::numeric::NumCast::from((task_performances.len() - 1) as f64)
                .unwrap_or_else(|| T::one());

        variance
    }

    /// Calculate recent performance trend
    fn calculate_performance_trend(&self) -> T {
        if self.performance_history.len() < 10 {
            return T::zero();
        }

        let recent: Vec<T> = self
            .performance_history
            .iter()
            .rev()
            .take(10)
            .map(|record| record.performance)
            .collect();

        let first_half_avg = recent[5..].iter().cloned().fold(T::zero(), |a, b| a + b)
            / scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero());
        let second_half_avg = recent[..5].iter().cloned().fold(T::zero(), |a, b| a + b)
            / scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero());

        second_half_avg - first_half_avg
    }

    /// Get expected performance for current difficulty
    fn get_expected_performance(&self) -> T {
        // Simple model: expected performance decreases with difficulty
        let difficulty_factor =
            self.curriculum_state.current_difficulty / self.curriculum_params.max_difficulty;
        T::one()
            - difficulty_factor
                * scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero())
    }

    /// Add task to curriculum
    pub fn add_task(
        &mut self,
        task_id: String,
        estimated_difficulty: T,
        required_competency: T,
    ) -> Result<()> {
        let scheduled_task = ScheduledTask {
            task_id: task_id.clone(),
            priority: T::one() / estimated_difficulty, // Higher priority for easier tasks initially
            difficulty: estimated_difficulty,
            required_competency,
            parameters: HashMap::new(),
        };

        self.task_scheduler.add_task(scheduled_task);

        // Initialize competency tracking
        self.progress_tracker
            .competency_levels
            .insert(task_id, T::zero());

        Ok(())
    }

    /// Get curriculum statistics
    pub fn get_curriculum_statistics(&self) -> HashMap<String, T> {
        let mut stats = HashMap::new();

        stats.insert(
            "current_difficulty".to_string(),
            self.curriculum_state.current_difficulty,
        );
        stats.insert(
            "recent_performance".to_string(),
            self.curriculum_state.recent_performance,
        );
        stats.insert(
            "epochs_since_increase".to_string(),
            scirs2_core::numeric::NumCast::from(self.curriculum_state.epochs_since_increase as f64)
                .unwrap_or_else(|| T::zero()),
        );
        stats.insert(
            "active_tasks_count".to_string(),
            scirs2_core::numeric::NumCast::from(self.curriculum_state.active_tasks.len() as f64)
                .unwrap_or_else(|| T::zero()),
        );

        // Average competency across all tasks
        if !self.progress_tracker.competency_levels.is_empty() {
            let avg_competency = self
                .progress_tracker
                .competency_levels
                .values()
                .cloned()
                .fold(T::zero(), |a, b| a + b)
                / scirs2_core::numeric::NumCast::from(
                    self.progress_tracker.competency_levels.len() as f64,
                )
                .unwrap_or_else(|| T::one());
            stats.insert("average_competency".to_string(), avg_competency);
        }

        stats
    }

    /// Get the active curriculum strategy
    pub fn strategy(&self) -> CurriculumStrategy {
        self.strategy
    }

    /// Current difficulty level
    pub fn current_difficulty(&self) -> T {
        self.curriculum_state.current_difficulty
    }

    /// Current learning phase
    pub fn learning_phase(&self) -> LearningPhase {
        self.curriculum_state.learning_phase
    }

    /// Change the task scheduling policy
    pub fn set_scheduling_policy(&mut self, policy: SchedulingPolicy) {
        self.task_scheduler.set_policy(policy);
    }

    /// Reset curriculum state
    pub fn reset(&mut self) {
        self.curriculum_state = CurriculumState::new_state();
        self.progress_tracker.reset();
        self.performance_history.clear();
        self.task_scheduler.reset();
    }
}

// Supporting type implementations
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> TaskDifficultyEstimator<T> {
    fn new() -> Result<Self> {
        Ok(Self {
            difficulty_predictor: DifficultyPredictor::new(10)?, // Default 10-dim input
            task_features: HashMap::new(),
            difficulty_history: HashMap::new(),
            estimation_method: DifficultyEstimationMethod::PerformanceBased,
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> LearningProgressTracker<T> {
    fn new() -> Self {
        Self {
            performance_timeline: VecDeque::new(),
            learning_rates: VecDeque::new(),
            competency_levels: HashMap::new(),
            milestones: Vec::new(),
        }
    }

    fn update_performance(&mut self, performance: T) {
        self.performance_timeline.push_back(performance);
        if self.performance_timeline.len() > 1000 {
            self.performance_timeline.pop_front();
        }
    }

    fn reset(&mut self) {
        self.performance_timeline.clear();
        self.learning_rates.clear();
        self.competency_levels.clear();
        self.milestones.clear();
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> CurriculumState<T> {
    fn new() -> Result<Self> {
        Ok(Self::new_state())
    }

    fn new_state() -> Self {
        Self {
            current_difficulty: scirs2_core::numeric::NumCast::from(0.1)
                .unwrap_or_else(|| T::zero()),
            active_tasks: Vec::new(),
            recent_performance: T::zero(),
            epochs_since_increase: 0,
            learning_phase: LearningPhase::Exploration,
            adaptive_params: HashMap::new(),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> TaskScheduler<T> {
    fn new() -> Result<Self> {
        Ok(Self {
            task_queue: VecDeque::new(),
            scheduling_policy: SchedulingPolicy::Priority,
            task_weights: HashMap::new(),
            load_balancing: HashMap::new(),
        })
    }

    fn add_task(&mut self, task: ScheduledTask<T>) {
        self.task_queue.push_back(task);
    }

    /// Pick the next task according to the configured scheduling policy.
    fn schedule_next_task(&mut self) -> Option<String> {
        if self.task_queue.is_empty() {
            return None;
        }

        let index = match self.scheduling_policy {
            SchedulingPolicy::FIFO => 0,
            SchedulingPolicy::Priority => Self::arg_extreme(&self.task_queue, true),
            SchedulingPolicy::Balanced => Self::arg_extreme(&self.task_queue, false),
            SchedulingPolicy::WeightedRandom => {
                // Deterministic weighted choice driven by the queue state so the
                // scheduler stays reproducible without an external generator.
                let total: f64 = self
                    .task_queue
                    .iter()
                    .map(|t| t.priority.to_f64().unwrap_or(0.0).max(0.0))
                    .sum();
                if total <= 0.0 {
                    0
                } else {
                    let mut target =
                        (self.task_weights.len() as f64 * 0.618_033_988_75).fract() * total;
                    let mut chosen = self.task_queue.len() - 1;
                    for (i, task) in self.task_queue.iter().enumerate() {
                        target -= task.priority.to_f64().unwrap_or(0.0).max(0.0);
                        if target <= 0.0 {
                            chosen = i;
                            break;
                        }
                    }
                    chosen
                }
            }
            SchedulingPolicy::Adaptive => {
                // Prefer the task whose difficulty is closest to the required
                // competency, i.e. the best-matched challenge.
                let mut best = 0;
                let mut best_gap = T::infinity();
                for (i, task) in self.task_queue.iter().enumerate() {
                    let gap = (task.difficulty - task.required_competency).abs();
                    if gap < best_gap {
                        best_gap = gap;
                        best = i;
                    }
                }
                best
            }
        };

        let task = self.task_queue.remove(index)?;
        let counter = self
            .load_balancing
            .entry(task.task_id.clone())
            .or_insert(T::zero());
        *counter = *counter + T::one();
        self.task_weights
            .insert(task.task_id.clone(), task.priority);
        Some(task.task_id)
    }

    /// Index of the highest (or lowest) priority task.
    fn arg_extreme(queue: &VecDeque<ScheduledTask<T>>, highest: bool) -> usize {
        let mut best = 0;
        let mut best_priority = match queue.front() {
            Some(task) => task.priority,
            None => return 0,
        };
        for (i, task) in queue.iter().enumerate() {
            let better = if highest {
                task.priority > best_priority
            } else {
                task.priority < best_priority
            };
            if better {
                best_priority = task.priority;
                best = i;
            }
        }
        best
    }

    /// Change the scheduling policy.
    fn set_policy(&mut self, policy: SchedulingPolicy) {
        self.scheduling_policy = policy;
    }

    fn reset(&mut self) {
        self.task_queue.clear();
        self.task_weights.clear();
        self.load_balancing.clear();
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> DifficultyPredictor<T> {
    fn new(input_dim: usize) -> Result<Self> {
        Ok(Self {
            input_dim,
            hidden_layers: vec![Array2::eye(input_dim)],
            output_layer: Array1::ones(input_dim),
            training_history: Vec::new(),
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> Default for CurriculumParams<T> {
    fn default() -> Self {
        Self {
            initial_difficulty: scirs2_core::numeric::NumCast::from(0.1)
                .unwrap_or_else(|| T::zero()),
            max_difficulty: scirs2_core::numeric::NumCast::from(1.0).unwrap_or_else(|| T::zero()),
            difficulty_increment: scirs2_core::numeric::NumCast::from(0.05)
                .unwrap_or_else(|| T::zero()),
            progression_threshold: scirs2_core::numeric::NumCast::from(0.8)
                .unwrap_or_else(|| T::zero()),
            patience: 5,
            self_pacing_factor: scirs2_core::numeric::NumCast::from(0.1)
                .unwrap_or_else(|| T::zero()),
            diversity_weight: scirs2_core::numeric::NumCast::from(0.2).unwrap_or_else(|| T::zero()),
            teacher_confidence: scirs2_core::numeric::NumCast::from(0.9)
                .unwrap_or_else(|| T::zero()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_strategies() -> [CurriculumStrategy; 8] {
        [
            CurriculumStrategy::None,
            CurriculumStrategy::DifficultyProgression,
            CurriculumStrategy::DiversityBased,
            CurriculumStrategy::SelfPaced,
            CurriculumStrategy::TeacherStudent,
            CurriculumStrategy::Adversarial,
            CurriculumStrategy::MultiTask,
            CurriculumStrategy::Adaptive,
        ]
    }

    #[test]
    fn every_strategy_runs_and_stays_in_range() {
        for strategy in all_strategies() {
            let mut learner = CurriculumLearner::<f64>::new(strategy).expect("curriculum creation");
            for step in 0..50 {
                let task = if step % 3 == 0 { "a" } else { "b" };
                learner
                    .update_curriculum(task, 0.9, step)
                    .unwrap_or_else(|e| panic!("{strategy:?} failed: {e}"));
            }
            let difficulty = learner.current_difficulty();
            assert!(
                (0.1..=1.0).contains(&difficulty),
                "{strategy:?} produced difficulty {difficulty}"
            );
        }
    }

    #[test]
    fn strategies_do_not_all_behave_identically() {
        let run = |strategy| {
            let mut learner = CurriculumLearner::<f64>::new(strategy).expect("curriculum creation");
            for step in 0..30 {
                learner
                    .update_curriculum("task", 0.95, step)
                    .expect("update");
            }
            learner.current_difficulty()
        };

        let teacher = run(CurriculumStrategy::TeacherStudent);
        let adversarial = run(CurriculumStrategy::Adversarial);
        let diversity = run(CurriculumStrategy::DiversityBased);
        let multi_task = run(CurriculumStrategy::MultiTask);

        let mut distinct = vec![teacher, adversarial, diversity, multi_task];
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        assert!(
            distinct.len() >= 3,
            "strategies collapsed to the same behaviour: {distinct:?}"
        );
    }

    #[test]
    fn zero_performance_keeps_difficulty_finite() {
        for strategy in all_strategies() {
            let mut learner = CurriculumLearner::<f64>::new(strategy).expect("curriculum creation");
            for step in 0..20 {
                learner.update_curriculum("t", 0.0, step).expect("update");
            }
            assert!(
                learner.current_difficulty().is_finite(),
                "{strategy:?} produced a non-finite difficulty"
            );
        }
    }

    #[test]
    fn scheduling_policies_pick_different_tasks() {
        let mut learner =
            CurriculumLearner::<f64>::new(CurriculumStrategy::Adaptive).expect("creation");
        learner.add_task("hard".to_string(), 4.0, 0.9).expect("add");
        learner.add_task("easy".to_string(), 1.0, 0.9).expect("add");

        learner.set_scheduling_policy(SchedulingPolicy::FIFO);
        assert_eq!(
            learner.get_next_task().expect("schedule"),
            Some("hard".to_string())
        );

        let mut learner =
            CurriculumLearner::<f64>::new(CurriculumStrategy::Adaptive).expect("creation");
        learner.add_task("hard".to_string(), 4.0, 0.9).expect("add");
        learner.add_task("easy".to_string(), 1.0, 0.9).expect("add");
        learner.set_scheduling_policy(SchedulingPolicy::Priority);
        // Priority is 1 / difficulty, so the easy task wins.
        assert_eq!(
            learner.get_next_task().expect("schedule"),
            Some("easy".to_string())
        );
    }

    #[test]
    fn statistics_are_reported() {
        let mut learner =
            CurriculumLearner::<f64>::new(CurriculumStrategy::Adaptive).expect("creation");
        learner.update_curriculum("t", 0.7, 1).expect("update");
        let stats = learner.get_curriculum_statistics();
        assert!(stats.contains_key("current_difficulty"));
        assert!(stats.contains_key("recent_performance"));
    }

    #[test]
    fn reset_restores_the_initial_state() {
        let mut learner =
            CurriculumLearner::<f64>::new(CurriculumStrategy::Adversarial).expect("creation");
        for step in 0..20 {
            learner.update_curriculum("t", 0.99, step).expect("update");
        }
        assert!(learner.current_difficulty() > 0.1);
        learner.reset();
        assert!((learner.current_difficulty() - 0.1).abs() < 1e-12);
    }
}
