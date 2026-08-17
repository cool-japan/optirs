// Task scheduling for optimization coordination
//
// This module provides task scheduling capabilities for optimization processes,
// including priority-based scheduling, resource-aware task allocation, and
// dynamic load balancing.

#[allow(dead_code)]
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant, SystemTime};

use crate::error::{OptimError, Result};

/// Parse a `usize` value out of a task's metadata map, if present and valid.
fn parse_metadata_usize(metadata: &HashMap<String, String>, key: &str) -> Option<usize> {
    metadata.get(key).and_then(|v| v.parse::<usize>().ok())
}

/// Parse an `f64` value out of a task's metadata map, if present and valid.
fn parse_metadata_f64(metadata: &HashMap<String, String>, key: &str) -> Option<f64> {
    metadata.get(key).and_then(|v| v.parse::<f64>().ok())
}

/// Average of `extract(record)` across a task type's resource-usage history,
/// or `None` if there is no history yet.
fn history_average_usize<T, F>(
    history: Option<&VecDeque<ResourceUsageRecord<T>>>,
    extract: F,
) -> Option<usize>
where
    T: Float + Debug + Send + Sync + 'static,
    F: Fn(&ResourceUsageRecord<T>) -> usize,
{
    let history = history?;
    if history.is_empty() {
        return None;
    }
    let sum: usize = history.iter().map(&extract).sum();
    Some(sum / history.len())
}

/// Average of `extract(record)` across a task type's resource-usage history,
/// or `None` if there is no history yet.
fn history_average_f64<T, F>(
    history: Option<&VecDeque<ResourceUsageRecord<T>>>,
    extract: F,
) -> Option<f64>
where
    T: Float + Debug + Send + Sync + 'static,
    F: Fn(&ResourceUsageRecord<T>) -> f64,
{
    let history = history?;
    if history.is_empty() {
        return None;
    }
    let sum: f64 = history.iter().map(&extract).sum();
    Some(sum / history.len() as f64)
}

/// Conservative default CPU core count by task type, used only when neither
/// task metadata nor historical data is available.
fn default_cpu_cores_for(task_type: &TaskType) -> usize {
    match task_type {
        TaskType::ArchitectureSearch | TaskType::MetaLearning | TaskType::EnsembleCoordination => 4,
        TaskType::PerformanceEvaluation | TaskType::KnowledgeDistillation => 2,
        TaskType::GradientComputation
        | TaskType::ParameterUpdate
        | TaskType::ResourceOptimization
        | TaskType::Custom(_) => 1,
    }
}

/// Conservative default memory (MB) by task type, used only when neither
/// task metadata nor historical data is available.
fn default_memory_mb_for(task_type: &TaskType) -> usize {
    match task_type {
        TaskType::ArchitectureSearch | TaskType::MetaLearning => 4096,
        TaskType::EnsembleCoordination | TaskType::KnowledgeDistillation => 2048,
        _ => 1024,
    }
}

/// Conservative default GPU device count by task type, used only when
/// neither task metadata nor historical data is available.
fn default_gpu_devices_for(task_type: &TaskType) -> usize {
    match task_type {
        TaskType::ArchitectureSearch | TaskType::MetaLearning | TaskType::GradientComputation => 1,
        _ => 0,
    }
}

/// Task scheduler for optimization processes
#[derive(Debug)]
pub struct TaskScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Pending tasks queue
    pending_tasks: VecDeque<ScheduledTask<T>>,

    /// Currently executing tasks
    executing_tasks: HashMap<String, ExecutingTask<T>>,

    /// Completed tasks history
    completed_tasks: VecDeque<CompletedTask<T>>,

    /// Scheduling strategy
    strategy: SchedulingStrategy,

    /// Task priority calculator
    priority_calculator: PriorityCalculator<T>,

    /// Resource requirements estimator
    resource_estimator: ResourceRequirementEstimator<T>,

    /// Load balancer
    load_balancer: TaskLoadBalancer<T>,

    /// Scheduler configuration
    config: SchedulerConfig<T>,

    /// Scheduler statistics
    stats: SchedulerStatistics<T>,
}

/// Scheduled task representation
#[derive(Debug, Clone)]
pub struct ScheduledTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    pub task_id: String,

    /// Task type
    pub task_type: TaskType,

    /// Task priority
    pub priority: TaskPriority<T>,

    /// Estimated resource requirements
    pub resource_requirements: TaskResourceRequirements,

    /// Task parameters
    pub parameters: HashMap<String, T>,

    /// Expected execution time
    pub estimated_duration: Duration,

    /// Task dependencies
    pub dependencies: Vec<String>,

    /// Creation timestamp
    pub created_at: SystemTime,

    /// Deadline (if any)
    pub deadline: Option<SystemTime>,

    /// Task metadata
    pub metadata: HashMap<String, String>,
}

impl<T: Float + Debug + Send + Sync + 'static> ScheduledTask<T> {
    /// Create a new scheduled task with default values
    pub fn new(name: String) -> Self {
        use std::time::SystemTime;
        use uuid::Uuid;

        Self {
            task_id: Uuid::new_v4().to_string(),
            task_type: TaskType::ParameterUpdate,
            priority: TaskPriority {
                base_priority: 5,
                urgency: T::from(0.5).unwrap_or_else(|| T::zero()),
                importance: T::from(0.5).unwrap_or_else(|| T::zero()),
                efficiency: T::one(),
                dynamic_adjustment: T::one(),
                composite_score: T::from(5.0).unwrap_or_else(|| T::zero()), // Initial composite score
            },
            resource_requirements: TaskResourceRequirements {
                cpu_cores: 1,
                memory_mb: 1024,
                gpu_devices: 0,
                storage_gb: 1,
                network_bandwidth: 1.0,
                special_hardware: Vec::new(),
            },
            parameters: HashMap::new(),
            estimated_duration: Duration::from_secs(60),
            dependencies: Vec::new(),
            created_at: SystemTime::now(),
            deadline: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("name".to_string(), name);
                meta
            },
        }
    }
}

/// Task priority with multiple dimensions
#[derive(Debug, Clone)]
pub struct TaskPriority<T: Float + Debug + Send + Sync + 'static> {
    /// Base priority level
    pub base_priority: u8,

    /// Urgency factor (0.0 to 1.0)
    pub urgency: T,

    /// Importance factor (0.0 to 1.0)
    pub importance: T,

    /// Resource efficiency factor
    pub efficiency: T,

    /// Dynamic adjustment factor
    pub dynamic_adjustment: T,

    /// Composite priority score
    pub composite_score: T,
}

/// Types of optimization tasks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Gradient computation task
    GradientComputation,

    /// Parameter update task
    ParameterUpdate,

    /// Architecture search task
    ArchitectureSearch,

    /// Meta-learning task
    MetaLearning,

    /// Performance evaluation task
    PerformanceEvaluation,

    /// Resource optimization task
    ResourceOptimization,

    /// Knowledge distillation task
    KnowledgeDistillation,

    /// Ensemble coordination task
    EnsembleCoordination,

    /// Custom task
    Custom(String),
}

/// Scheduling strategies
#[derive(Debug, Clone, Copy)]
pub enum SchedulingStrategy {
    /// First-In-First-Out
    FIFO,

    /// Priority-based scheduling
    PriorityBased,

    /// Shortest Job First
    ShortestJobFirst,

    /// Round Robin
    RoundRobin,

    /// Fair Share scheduling
    FairShare,

    /// Resource-aware scheduling
    ResourceAware,

    /// Deadline-aware scheduling
    DeadlineAware,

    /// Learning-based scheduling
    LearningBased,
}

/// Task resource requirements
#[derive(Debug, Clone)]
pub struct TaskResourceRequirements {
    /// CPU cores required
    pub cpu_cores: usize,

    /// Memory required (MB)
    pub memory_mb: usize,

    /// GPU devices required
    pub gpu_devices: usize,

    /// Storage required (GB)
    pub storage_gb: usize,

    /// Network bandwidth required (Mbps)
    pub network_bandwidth: f64,

    /// Special hardware requirements
    pub special_hardware: Vec<String>,
}

/// Currently executing task
#[derive(Debug)]
pub struct ExecutingTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task information
    pub task: ScheduledTask<T>,

    /// Execution start time
    pub start_time: SystemTime,

    /// Assigned resources
    pub assigned_resources: AssignedResources,

    /// Current progress (0.0 to 1.0)
    pub progress: T,

    /// Performance metrics
    pub metrics: ExecutionMetrics<T>,

    /// Resource utilization
    pub resource_utilization: ResourceUtilization<T>,
}

/// Completed task record
#[derive(Debug, Clone)]
pub struct CompletedTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task information
    pub task: ScheduledTask<T>,

    /// Execution start time
    pub start_time: SystemTime,

    /// Execution end time
    pub end_time: SystemTime,

    /// Actual duration
    pub duration: Duration,

    /// Final performance metrics
    pub final_metrics: ExecutionMetrics<T>,

    /// Resource efficiency
    pub resource_efficiency: T,

    /// Success status
    pub success: bool,

    /// Error information (if failed)
    pub error_info: Option<String>,
}

/// Assigned resources for a task
#[derive(Debug, Clone)]
pub struct AssignedResources {
    /// Assigned CPU cores
    pub cpu_cores: Vec<usize>,

    /// Assigned memory (MB)
    pub memory_mb: usize,

    /// Assigned GPU devices
    pub gpu_devices: Vec<usize>,

    /// Assigned storage (GB)
    pub storage_gb: usize,

    /// Assigned network bandwidth (Mbps)
    pub network_bandwidth: f64,
}

/// Execution metrics for running tasks
#[derive(Debug, Clone)]
pub struct ExecutionMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Throughput metric
    pub throughput: T,

    /// Latency metric
    pub latency: T,

    /// CPU utilization
    pub cpu_utilization: T,

    /// Memory utilization
    pub memory_utilization: T,

    /// GPU utilization
    pub gpu_utilization: T,

    /// Quality metric
    pub quality: T,
}

/// Resource utilization tracking
#[derive(Debug, Clone)]
pub struct ResourceUtilization<T: Float + Debug + Send + Sync + 'static> {
    /// Current CPU usage
    pub cpu_usage: T,

    /// Current memory usage
    pub memory_usage: T,

    /// Current GPU usage
    pub gpu_usage: T,

    /// Current network usage
    pub network_usage: T,

    /// Efficiency score
    pub efficiency_score: T,
}

/// Priority calculator for tasks
#[derive(Debug)]
pub struct PriorityCalculator<T: Float + Debug + Send + Sync + 'static> {
    /// Priority weights
    weights: PriorityWeights<T>,

    /// Historical priority performance
    priority_performance: HashMap<String, T>,

    /// Dynamic adjustment algorithm
    adjustment_algorithm: PriorityAdjustmentAlgorithm,
}

/// Priority calculation weights
#[derive(Debug, Clone)]
pub struct PriorityWeights<T: Float + Debug + Send + Sync + 'static> {
    /// Base priority weight
    pub base_weight: T,

    /// Urgency weight
    pub urgency_weight: T,

    /// Importance weight
    pub importance_weight: T,

    /// Efficiency weight
    pub efficiency_weight: T,

    /// Historical performance weight
    pub history_weight: T,
}

/// Priority adjustment algorithms
#[derive(Debug, Clone, Copy)]
pub enum PriorityAdjustmentAlgorithm {
    /// Static priorities
    Static,

    /// Linear adjustment
    Linear,

    /// Exponential adjustment
    Exponential,

    /// Learning-based adjustment
    LearningBased,

    /// Feedback-based adjustment
    FeedbackBased,
}

/// Resource requirement estimator
#[derive(Debug)]
pub struct ResourceRequirementEstimator<T: Float + Debug + Send + Sync + 'static> {
    /// Historical resource usage data
    resource_history: HashMap<TaskType, VecDeque<ResourceUsageRecord<T>>>,

    /// Estimation models
    estimation_models: HashMap<TaskType, EstimationModel<T>>,

    /// Estimation accuracy tracker
    accuracy_tracker: EstimationAccuracyTracker<T>,
}

/// Resource usage record for learning
#[derive(Debug, Clone)]
pub struct ResourceUsageRecord<T: Float + Debug + Send + Sync + 'static> {
    /// Task parameters
    pub task_params: HashMap<String, T>,

    /// Actual resource usage
    pub actual_usage: TaskResourceRequirements,

    /// Execution time
    pub execution_time: Duration,

    /// Performance achieved
    pub performance: T,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Estimation model for resource requirements
#[derive(Debug)]
pub struct EstimationModel<T: Float + Debug + Send + Sync + 'static> {
    /// Model type
    model_type: EstimationModelType,

    /// Model parameters
    parameters: HashMap<String, Array1<T>>,

    /// Model accuracy
    accuracy: T,

    /// Training data size
    training_size: usize,
}

/// Types of estimation models
#[derive(Debug, Clone, Copy)]
pub enum EstimationModelType {
    /// Linear regression
    LinearRegression,

    /// Polynomial regression
    PolynomialRegression,

    /// Neural network
    NeuralNetwork,

    /// Ensemble model
    Ensemble,

    /// Historical average
    HistoricalAverage,
}

/// Estimation accuracy tracker
#[derive(Debug)]
pub struct EstimationAccuracyTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Accuracy per task type
    task_accuracies: HashMap<TaskType, T>,

    /// Overall accuracy
    overall_accuracy: T,

    /// Accuracy trend
    accuracy_trend: VecDeque<T>,

    /// Accuracy improvement rate
    improvement_rate: T,
}

/// Task load balancer
#[derive(Debug)]
pub struct TaskLoadBalancer<T: Float + Debug + Send + Sync + 'static> {
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,

    /// Current load per resource
    resource_loads: HashMap<String, T>,

    /// Load history
    load_history: VecDeque<LoadSnapshot<T>>,

    /// Load prediction model
    prediction_model: LoadPredictionModel<T>,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Round robin assignment
    RoundRobin,

    /// Least loaded first
    LeastLoaded,

    /// Weighted round robin
    WeightedRoundRobin,

    /// Resource-aware balancing
    ResourceAware,

    /// Predictive load balancing
    Predictive,

    /// Learning-based balancing
    LearningBased,
}

/// Load snapshot for tracking
#[derive(Debug, Clone)]
pub struct LoadSnapshot<T: Float + Debug + Send + Sync + 'static> {
    /// Timestamp
    pub timestamp: SystemTime,

    /// CPU load per core
    pub cpu_loads: Vec<T>,

    /// Memory usage
    pub memory_usage: T,

    /// GPU loads per device
    pub gpu_loads: Vec<T>,

    /// Network utilization
    pub network_utilization: T,

    /// Overall system load
    pub overall_load: T,
}

/// Load prediction model
#[derive(Debug)]
pub struct LoadPredictionModel<T: Float + Debug + Send + Sync + 'static> {
    /// Prediction horizon
    horizon: Duration,

    /// Model parameters
    parameters: HashMap<String, Array1<T>>,

    /// Prediction accuracy
    accuracy: T,

    /// Update frequency
    update_frequency: Duration,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,

    /// Queue size limit
    pub queue_size_limit: usize,

    /// Task timeout duration
    pub task_timeout: Duration,

    /// Priority recalculation interval
    pub priority_update_interval: Duration,

    /// Load balancing update interval
    pub load_balance_interval: Duration,

    /// Resource estimation accuracy threshold
    pub estimation_threshold: T,

    /// Enable adaptive scheduling
    pub enable_adaptive_scheduling: bool,

    /// Enable performance learning
    pub enable_performance_learning: bool,
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total tasks scheduled
    pub total_tasks_scheduled: usize,

    /// Total tasks completed
    pub total_tasks_completed: usize,

    /// Total tasks failed
    pub total_tasks_failed: usize,

    /// Average execution time
    pub average_execution_time: Duration,

    /// Average waiting time
    pub average_waiting_time: Duration,

    /// Resource utilization efficiency
    pub resource_efficiency: T,

    /// Scheduling overhead
    pub scheduling_overhead: T,

    /// Throughput (tasks per second)
    pub throughput: T,
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> TaskScheduler<T> {
    /// Create new task scheduler
    pub fn new(config: SchedulerConfig<T>) -> Result<Self> {
        Ok(Self {
            pending_tasks: VecDeque::new(),
            executing_tasks: HashMap::new(),
            completed_tasks: VecDeque::new(),
            strategy: SchedulingStrategy::PriorityBased,
            priority_calculator: PriorityCalculator::new()?,
            resource_estimator: ResourceRequirementEstimator::new()?,
            load_balancer: TaskLoadBalancer::new()?,
            config,
            stats: SchedulerStatistics::default(),
        })
    }

    /// Submit a new task for scheduling
    pub fn submit_task(&mut self, mut task: ScheduledTask<T>) -> Result<()> {
        // Calculate task priority from real signals: base priority,
        // deadline slack, dependents in the current pending queue, and
        // learned resource-usage history for this task type.
        task.priority = self.priority_calculator.calculate_priority(
            &task,
            &self.pending_tasks,
            &self.resource_estimator.resource_history,
        )?;

        // Estimate resource requirements if not provided
        if task.resource_requirements.cpu_cores == 0 {
            task.resource_requirements = self.resource_estimator.estimate_requirements(&task)?;
        }

        // Add to pending queue with priority ordering
        self.insert_task_by_priority(task)?;

        // Update statistics
        self.stats.total_tasks_scheduled += 1;

        Ok(())
    }

    /// Get next task to execute based on scheduling strategy
    pub fn get_next_task(&mut self) -> Option<ScheduledTask<T>> {
        match self.strategy {
            SchedulingStrategy::FIFO => self.pending_tasks.pop_front(),
            SchedulingStrategy::PriorityBased => self.get_highest_priority_task(),
            SchedulingStrategy::ShortestJobFirst => self.get_shortest_job(),
            SchedulingStrategy::DeadlineAware => self.get_most_urgent_task(),
            _ => self.pending_tasks.pop_front(),
        }
    }

    /// Start executing a task
    pub fn start_task_execution(
        &mut self,
        task: ScheduledTask<T>,
        assigned_resources: AssignedResources,
    ) -> Result<()> {
        let executing_task = ExecutingTask {
            task: task.clone(),
            start_time: SystemTime::now(),
            assigned_resources,
            progress: T::zero(),
            metrics: ExecutionMetrics::default(),
            resource_utilization: ResourceUtilization::default(),
        };

        self.executing_tasks
            .insert(task.task_id.clone(), executing_task);
        Ok(())
    }

    /// Update task execution progress
    pub fn update_task_progress(
        &mut self,
        task_id: &str,
        progress: T,
        metrics: ExecutionMetrics<T>,
    ) -> Result<()> {
        if let Some(executing_task) = self.executing_tasks.get_mut(task_id) {
            executing_task.progress = progress;
            executing_task.metrics = metrics;
        }
        Ok(())
    }

    /// Complete task execution
    pub fn complete_task(
        &mut self,
        task_id: &str,
        success: bool,
        error_info: Option<String>,
    ) -> Result<()> {
        if let Some(executing_task) = self.executing_tasks.remove(task_id) {
            let completed_task = CompletedTask {
                task: executing_task.task,
                start_time: executing_task.start_time,
                end_time: SystemTime::now(),
                duration: SystemTime::now()
                    .duration_since(executing_task.start_time)
                    .unwrap_or_default(),
                final_metrics: executing_task.metrics,
                resource_efficiency: executing_task.resource_utilization.efficiency_score,
                success,
                error_info,
            };

            // Update statistics
            if success {
                self.stats.total_tasks_completed += 1;
            } else {
                self.stats.total_tasks_failed += 1;
            }

            // Learn from completed task
            self.resource_estimator
                .learn_from_completion(&completed_task)?;

            // Store in history
            self.completed_tasks.push_back(completed_task);

            // Limit history size
            if self.completed_tasks.len() > 1000 {
                self.completed_tasks.pop_front();
            }
        }

        Ok(())
    }

    /// Get pending tasks count
    pub fn pending_tasks_count(&self) -> usize {
        self.pending_tasks.len()
    }

    /// Get executing tasks count
    pub fn executing_tasks_count(&self) -> usize {
        self.executing_tasks.len()
    }

    /// Get scheduler statistics
    pub fn get_statistics(&self) -> &SchedulerStatistics<T> {
        &self.stats
    }

    /// Update scheduler configuration
    pub fn update_config(&mut self, config: SchedulerConfig<T>) {
        self.config = config;
    }

    /// Insert task maintaining priority order
    fn insert_task_by_priority(&mut self, task: ScheduledTask<T>) -> Result<()> {
        let position = self
            .pending_tasks
            .iter()
            .position(|t| t.priority.composite_score < task.priority.composite_score)
            .unwrap_or(self.pending_tasks.len());

        self.pending_tasks.insert(position, task);
        Ok(())
    }

    /// Get highest priority task
    fn get_highest_priority_task(&mut self) -> Option<ScheduledTask<T>> {
        if !self.pending_tasks.is_empty() {
            self.pending_tasks.pop_front()
        } else {
            None
        }
    }

    /// Get shortest job first
    fn get_shortest_job(&mut self) -> Option<ScheduledTask<T>> {
        if self.pending_tasks.is_empty() {
            return None;
        }

        let shortest_idx = self
            .pending_tasks
            .iter()
            .enumerate()
            .min_by_key(|(_, task)| task.estimated_duration)
            .map(|(idx, _)| idx)?;

        self.pending_tasks.remove(shortest_idx)
    }

    /// Get most urgent task (closest deadline)
    fn get_most_urgent_task(&mut self) -> Option<ScheduledTask<T>> {
        if self.pending_tasks.is_empty() {
            return None;
        }

        let now = SystemTime::now();
        let most_urgent_idx = self
            .pending_tasks
            .iter()
            .enumerate()
            .filter_map(|(idx, task)| task.deadline.map(|deadline| (idx, deadline)))
            .min_by_key(|(_, deadline)| *deadline)
            .map(|(idx, _)| idx);

        if let Some(idx) = most_urgent_idx {
            self.pending_tasks.remove(idx)
        } else {
            self.pending_tasks.pop_front()
        }
    }
}

// Implementation of helper structs

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> PriorityCalculator<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            weights: PriorityWeights::default(),
            priority_performance: HashMap::new(),
            adjustment_algorithm: PriorityAdjustmentAlgorithm::LearningBased,
        })
    }

    /// Calculate the full priority (urgency/importance/efficiency and their
    /// weighted composite) for `task`. `all_tasks` is the current pending
    /// queue, used to count how many other tasks depend on this one, and
    /// `resource_history` is the scheduler's learned resource-usage history
    /// per task type, used to derive an efficiency estimate from real past
    /// performance instead of a constant.
    pub fn calculate_priority(
        &self,
        task: &ScheduledTask<T>,
        all_tasks: &VecDeque<ScheduledTask<T>>,
        resource_history: &HashMap<TaskType, VecDeque<ResourceUsageRecord<T>>>,
    ) -> Result<TaskPriority<T>> {
        let base_priority = T::from(task.priority.base_priority).unwrap_or_else(|| T::zero());
        let urgency = self.calculate_urgency(task)?;
        let importance = self.calculate_importance(task, all_tasks)?;
        let efficiency = self.calculate_efficiency(task, resource_history)?;

        let composite_score = base_priority * self.weights.base_weight
            + urgency * self.weights.urgency_weight
            + importance * self.weights.importance_weight
            + efficiency * self.weights.efficiency_weight;

        Ok(TaskPriority {
            base_priority: task.priority.base_priority,
            urgency,
            importance,
            efficiency,
            dynamic_adjustment: T::zero(),
            composite_score,
        })
    }

    fn calculate_urgency(&self, task: &ScheduledTask<T>) -> Result<T> {
        if let Some(deadline) = task.deadline {
            let now = SystemTime::now();
            let time_to_deadline = deadline.duration_since(now).unwrap_or_default();
            let urgency = T::one()
                - T::from(time_to_deadline.as_secs_f64() / 3600.0).unwrap_or_else(|| T::zero());
            Ok(urgency.max(T::zero()).min(T::one()))
        } else {
            Ok(T::from(0.5).unwrap_or_else(|| T::zero()))
        }
    }

    /// Importance derived from three real signals: the task's declared base
    /// priority, how much slack its deadline leaves relative to its own
    /// estimated duration (tighter slack = more important), and how many
    /// other currently-known tasks depend on it (more dependents = more
    /// important, since delaying this task blocks others).
    fn calculate_importance(
        &self,
        task: &ScheduledTask<T>,
        all_tasks: &VecDeque<ScheduledTask<T>>,
    ) -> Result<T> {
        // Base priority is a u8; treat 10 as a "very high" reference point.
        let base_component = T::from(task.priority.base_priority as f64 / 10.0)
            .unwrap_or_else(|| T::zero())
            .min(T::one());

        let deadline_component = match task.deadline {
            Some(deadline) => match deadline.duration_since(SystemTime::now()) {
                Ok(remaining) => {
                    let estimated_secs = task.estimated_duration.as_secs_f64().max(1.0);
                    let slack_ratio = (remaining.as_secs_f64() / estimated_secs).max(0.0);
                    // Little slack beyond the estimated runtime -> high
                    // importance; ample slack -> low importance.
                    T::from((1.0 / (1.0 + slack_ratio)).clamp(0.0, 1.0))
                        .unwrap_or_else(|| T::zero())
                }
                Err(_) => T::one(), // deadline already passed: maximally important
            },
            None => T::from(0.3).unwrap_or_else(|| T::zero()),
        };

        let dependents = all_tasks
            .iter()
            .filter(|other| other.dependencies.iter().any(|dep| dep == &task.task_id))
            .count();
        let dependents_component =
            T::from((dependents as f64 / 5.0).min(1.0)).unwrap_or_else(|| T::zero());

        let weight_base = T::from(0.4).unwrap_or_else(|| T::zero());
        let weight_deadline = T::from(0.35).unwrap_or_else(|| T::zero());
        let weight_dependents = T::from(0.25).unwrap_or_else(|| T::zero());

        let importance = base_component * weight_base
            + deadline_component * weight_deadline
            + dependents_component * weight_dependents;

        Ok(importance.max(T::zero()).min(T::one()))
    }

    /// Efficiency derived from the historical performance of previously
    /// completed tasks of the same type (`resource_history`, populated by
    /// `ResourceRequirementEstimator::learn_from_completion`). Falls back to
    /// a neutral 0.5 only when no history exists yet for this task type.
    fn calculate_efficiency(
        &self,
        task: &ScheduledTask<T>,
        resource_history: &HashMap<TaskType, VecDeque<ResourceUsageRecord<T>>>,
    ) -> Result<T> {
        match resource_history.get(&task.task_type) {
            Some(history) if !history.is_empty() => {
                let count = T::from(history.len()).unwrap_or_else(|| T::one());
                let sum = history
                    .iter()
                    .fold(T::zero(), |acc, record| acc + record.performance);
                Ok((sum / count).max(T::zero()).min(T::one()))
            }
            _ => Ok(T::from(0.5).unwrap_or_else(|| T::zero())),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> ResourceRequirementEstimator<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            resource_history: HashMap::new(),
            estimation_models: HashMap::new(),
            accuracy_tracker: EstimationAccuracyTracker::default(),
        })
    }

    /// Estimate resource requirements for `task`, preferring (in order):
    /// 1. Explicit values in `task.metadata` (the caller's stated intent);
    /// 2. The average actual usage recorded in `resource_history` for tasks
    ///    of the same `task_type` (real learned data, previously collected
    ///    by `learn_from_completion` but never read anywhere);
    /// 3. A conservative task-type-based heuristic default.
    pub fn estimate_requirements(
        &self,
        task: &ScheduledTask<T>,
    ) -> Result<TaskResourceRequirements> {
        let history = self.resource_history.get(&task.task_type);

        let cpu_cores = parse_metadata_usize(&task.metadata, "cpu_cores")
            .or_else(|| history_average_usize(history, |r| r.actual_usage.cpu_cores))
            .unwrap_or_else(|| default_cpu_cores_for(&task.task_type));
        let memory_mb = parse_metadata_usize(&task.metadata, "memory_mb")
            .or_else(|| history_average_usize(history, |r| r.actual_usage.memory_mb))
            .unwrap_or_else(|| default_memory_mb_for(&task.task_type));
        let gpu_devices = parse_metadata_usize(&task.metadata, "gpu_devices")
            .or_else(|| history_average_usize(history, |r| r.actual_usage.gpu_devices))
            .unwrap_or_else(|| default_gpu_devices_for(&task.task_type));
        let storage_gb = parse_metadata_usize(&task.metadata, "storage_gb")
            .or_else(|| history_average_usize(history, |r| r.actual_usage.storage_gb))
            .unwrap_or(1);
        let network_bandwidth = parse_metadata_f64(&task.metadata, "network_bandwidth_mbps")
            .or_else(|| history_average_f64(history, |r| r.actual_usage.network_bandwidth))
            .unwrap_or(10.0);
        let special_hardware = task
            .metadata
            .get("special_hardware")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        Ok(TaskResourceRequirements {
            cpu_cores,
            memory_mb,
            gpu_devices,
            storage_gb,
            network_bandwidth,
            special_hardware,
        })
    }

    pub fn learn_from_completion(&mut self, completed_task: &CompletedTask<T>) -> Result<()> {
        // Learn from actual resource usage vs estimates
        let record = ResourceUsageRecord {
            task_params: completed_task.task.parameters.clone(),
            actual_usage: completed_task.task.resource_requirements.clone(),
            execution_time: completed_task.duration,
            performance: completed_task.final_metrics.quality,
            timestamp: completed_task.end_time,
        };

        self.resource_history
            .entry(completed_task.task.task_type.clone())
            .or_default()
            .push_back(record);

        Ok(())
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> TaskLoadBalancer<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            strategy: LoadBalancingStrategy::LeastLoaded,
            resource_loads: HashMap::new(),
            load_history: VecDeque::new(),
            prediction_model: LoadPredictionModel::default(),
        })
    }
}

// Default implementations

impl<T: Float + Debug + Default + Send + Sync> Default for PriorityWeights<T> {
    fn default() -> Self {
        Self {
            base_weight: T::from(0.3).unwrap_or_else(|| T::zero()),
            urgency_weight: T::from(0.25).unwrap_or_else(|| T::zero()),
            importance_weight: T::from(0.25).unwrap_or_else(|| T::zero()),
            efficiency_weight: T::from(0.15).unwrap_or_else(|| T::zero()),
            history_weight: T::from(0.05).unwrap_or_else(|| T::zero()),
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> Default for ExecutionMetrics<T> {
    fn default() -> Self {
        Self {
            throughput: T::zero(),
            latency: T::zero(),
            cpu_utilization: T::zero(),
            memory_utilization: T::zero(),
            gpu_utilization: T::zero(),
            quality: T::zero(),
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> Default for ResourceUtilization<T> {
    fn default() -> Self {
        Self {
            cpu_usage: T::zero(),
            memory_usage: T::zero(),
            gpu_usage: T::zero(),
            network_usage: T::zero(),
            efficiency_score: T::zero(),
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> Default for EstimationAccuracyTracker<T> {
    fn default() -> Self {
        Self {
            task_accuracies: HashMap::new(),
            overall_accuracy: T::from(0.5).unwrap_or_else(|| T::zero()),
            accuracy_trend: VecDeque::new(),
            improvement_rate: T::zero(),
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> Default for LoadPredictionModel<T> {
    fn default() -> Self {
        Self {
            horizon: Duration::from_secs(300), // 5 minutes
            parameters: HashMap::new(),
            accuracy: T::from(0.5).unwrap_or_else(|| T::zero()),
            update_frequency: Duration::from_secs(60),
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> Default for SchedulerStatistics<T> {
    fn default() -> Self {
        Self {
            total_tasks_scheduled: 0,
            total_tasks_completed: 0,
            total_tasks_failed: 0,
            average_execution_time: Duration::from_secs(0),
            average_waiting_time: Duration::from_secs(0),
            resource_efficiency: T::zero(),
            scheduling_overhead: T::zero(),
            throughput: T::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler() -> TaskScheduler<f64> {
        TaskScheduler::<f64>::new(SchedulerConfig {
            max_concurrent_tasks: 4,
            queue_size_limit: 100,
            task_timeout: Duration::from_secs(60),
            priority_update_interval: Duration::from_secs(5),
            load_balance_interval: Duration::from_secs(10),
            estimation_threshold: 0.9,
            enable_adaptive_scheduling: true,
            enable_performance_learning: true,
        })
        .expect("scheduler construction failed")
    }

    #[test]
    fn priority_queue_pops_highest_composite_score_first() {
        // Regression test for F63: pending tasks must be served in
        // descending priority order.
        let mut sched = scheduler();

        let mut low = ScheduledTask::<f64>::new("low".to_string());
        low.priority.base_priority = 1;
        let mut high = ScheduledTask::<f64>::new("high".to_string());
        high.priority.base_priority = 9;
        let mut mid = ScheduledTask::<f64>::new("mid".to_string());
        mid.priority.base_priority = 5;

        sched.submit_task(low).expect("submit low");
        sched.submit_task(high).expect("submit high");
        sched.submit_task(mid).expect("submit mid");

        let first = sched.get_next_task().expect("first task");
        let second = sched.get_next_task().expect("second task");
        let third = sched.get_next_task().expect("third task");

        assert!(
            first.priority.composite_score >= second.priority.composite_score,
            "expected descending order: {:?} >= {:?}",
            first.priority.composite_score,
            second.priority.composite_score
        );
        assert!(
            second.priority.composite_score >= third.priority.composite_score,
            "expected descending order: {:?} >= {:?}",
            second.priority.composite_score,
            third.priority.composite_score
        );
        // The task submitted with the highest base_priority must come out
        // first under the default PriorityBased strategy.
        assert_eq!(first.metadata.get("name"), Some(&"high".to_string()));
    }

    #[test]
    fn importance_reflects_dependents_not_a_constant() {
        // Regression test for F63: calculate_importance must vary with real
        // signals (here: number of dependents), not return a fixed 0.5.
        let calc = PriorityCalculator::<f64>::new().expect("calculator construction failed");

        let base = ScheduledTask::<f64>::new("base".to_string());
        let mut dependent = ScheduledTask::<f64>::new("dependent".to_string());
        dependent.dependencies.push(base.task_id.clone());

        let empty_queue: VecDeque<ScheduledTask<f64>> = VecDeque::new();
        let with_dependent: VecDeque<ScheduledTask<f64>> = VecDeque::from(vec![dependent]);

        let importance_no_deps = calc
            .calculate_importance(&base, &empty_queue)
            .expect("importance calc");
        let importance_with_deps = calc
            .calculate_importance(&base, &with_dependent)
            .expect("importance calc");

        assert!(
            importance_with_deps > importance_no_deps,
            "a task with a dependent must be considered more important: {importance_with_deps} vs {importance_no_deps}"
        );
    }

    #[test]
    fn efficiency_uses_resource_history_not_a_constant() {
        // Regression test for F63: calculate_efficiency must derive from
        // resource_history when present, not always return 0.5.
        let calc = PriorityCalculator::<f64>::new().expect("calculator construction failed");
        let task = ScheduledTask::<f64>::new("t".to_string());

        let mut history: HashMap<TaskType, VecDeque<ResourceUsageRecord<f64>>> = HashMap::new();
        let mut records = VecDeque::new();
        records.push_back(ResourceUsageRecord {
            task_params: HashMap::new(),
            actual_usage: TaskResourceRequirements {
                cpu_cores: 1,
                memory_mb: 512,
                gpu_devices: 0,
                storage_gb: 1,
                network_bandwidth: 1.0,
                special_hardware: Vec::new(),
            },
            execution_time: Duration::from_secs(10),
            performance: 0.9,
            timestamp: SystemTime::now(),
        });
        history.insert(task.task_type.clone(), records);

        let empty_history: HashMap<TaskType, VecDeque<ResourceUsageRecord<f64>>> = HashMap::new();

        let efficiency_no_history = calc
            .calculate_efficiency(&task, &empty_history)
            .expect("efficiency calc");
        let efficiency_with_history = calc
            .calculate_efficiency(&task, &history)
            .expect("efficiency calc");

        assert_eq!(efficiency_no_history, 0.5);
        assert!(
            (efficiency_with_history - 0.9).abs() < 1e-9,
            "efficiency should reflect the recorded performance: got {efficiency_with_history}"
        );
    }

    #[test]
    fn estimate_requirements_uses_metadata_when_present() {
        // Regression test for F63: estimate_requirements must honor
        // explicit task metadata instead of always returning fixed values.
        let estimator =
            ResourceRequirementEstimator::<f64>::new().expect("estimator construction failed");

        let mut task = ScheduledTask::<f64>::new("custom".to_string());
        task.metadata
            .insert("cpu_cores".to_string(), "16".to_string());
        task.metadata
            .insert("memory_mb".to_string(), "32768".to_string());
        task.metadata
            .insert("gpu_devices".to_string(), "2".to_string());

        let requirements = estimator
            .estimate_requirements(&task)
            .expect("estimate_requirements");

        assert_eq!(requirements.cpu_cores, 16);
        assert_eq!(requirements.memory_mb, 32768);
        assert_eq!(requirements.gpu_devices, 2);
    }

    #[test]
    fn estimate_requirements_learns_from_history_when_no_metadata() {
        let mut estimator =
            ResourceRequirementEstimator::<f64>::new().expect("estimator construction failed");

        let completed = CompletedTask {
            task: ScheduledTask::<f64>::new("historical".to_string()),
            start_time: SystemTime::now(),
            end_time: SystemTime::now(),
            duration: Duration::from_secs(5),
            final_metrics: ExecutionMetrics::default(),
            resource_efficiency: 0.8,
            success: true,
            error_info: None,
        };
        let mut historical_task = completed.task.clone();
        historical_task.resource_requirements = TaskResourceRequirements {
            cpu_cores: 8,
            memory_mb: 8192,
            gpu_devices: 1,
            storage_gb: 5,
            network_bandwidth: 50.0,
            special_hardware: Vec::new(),
        };
        let mut completed = completed;
        completed.task = historical_task;

        estimator
            .learn_from_completion(&completed)
            .expect("learn_from_completion");

        let mut new_task = ScheduledTask::<f64>::new("same_type".to_string());
        new_task.task_type = completed.task.task_type.clone();

        let requirements = estimator
            .estimate_requirements(&new_task)
            .expect("estimate_requirements");

        assert_eq!(requirements.cpu_cores, 8);
        assert_eq!(requirements.memory_mb, 8192);
    }

    #[test]
    fn estimate_requirements_falls_back_to_type_heuristic() {
        let estimator =
            ResourceRequirementEstimator::<f64>::new().expect("estimator construction failed");

        let mut search_task = ScheduledTask::<f64>::new("search".to_string());
        search_task.task_type = TaskType::ArchitectureSearch;

        let requirements = estimator
            .estimate_requirements(&search_task)
            .expect("estimate_requirements");

        // ArchitectureSearch's heuristic default is heavier than the
        // previous universal constant of 1 core / 1024MB.
        assert_eq!(requirements.cpu_cores, 4);
        assert_eq!(requirements.memory_mb, 4096);
    }
}
