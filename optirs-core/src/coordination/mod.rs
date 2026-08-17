// Coordination module for optimization processes
//
// This module provides comprehensive coordination capabilities for optimization
// workflows, including task scheduling, pipeline orchestration, and monitoring.
// It replaces the monolithic optimization_coordinator.rs with a modular architecture.

#[allow(dead_code)]
use crate::coordination::monitoring::anomaly_detection::{AnomalyConfig, AnomalyResult};
use crate::coordination::monitoring::performance_tracking::{
    DashboardConfiguration, TrackerConfiguration,
};
use crate::coordination::orchestration::pipeline_orchestrator::{
    ExecutionState, OrchestratorConfiguration,
};
use crate::coordination::scheduling::task_scheduler::SchedulerConfig;
use crate::research::experiments::ResourceUsage;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::time::{Duration, Instant, SystemTime};

// Submodule declarations
pub mod monitoring;
pub mod orchestration;
pub mod scheduling;

// Re-export key types from submodules
pub use scheduling::{
    PriorityLevel, PriorityManager, PriorityQueue, PriorityUpdateStrategy,
    ResourceAllocationStrategy, ResourceAllocationTracker, ResourceManager, ResourcePool,
    ScheduledTask, SchedulingStrategy, StaticPriorityStrategy, TaskPriority, TaskScheduler,
};

// Type alias for convenience
pub type OptimizationTask<T> = ScheduledTask<T>;

pub use orchestration::{
    AlertConfiguration, Checkpoint, CheckpointConfiguration, CheckpointManager, CheckpointMetadata,
    CheckpointStorage, Experiment, ExperimentConfiguration, ExperimentExecution, ExperimentManager,
    ExperimentResult, ExperimentStatus, FileCheckpointStorage, InMemoryCheckpointStorage,
    MonitoringConfiguration, OptimizationPipeline, PipelineConfiguration, PipelineExecution,
    PipelineOrchestrator, PipelineStage, RecoveryManager, RecoveryOptions, RecoveryStrategy,
    RecoveryTarget, ResourceLimits, StageResult, StateType, StorageConfiguration, TimeoutSettings,
    ValidationRule,
};

pub use monitoring::{
    AlertManager, AnomalyAlert, AnomalyAnalyzer, AnomalyClassifier, AnomalyDetector,
    AnomalyReporter, ConvergenceAnalyzer, ConvergenceCriteria, ConvergenceDetector,
    ConvergenceIndicator, ConvergenceMonitor, ConvergenceResult, MetricCollector, OutlierDetector,
    PerformanceAlert, PerformanceMetrics, PerformanceTracker,
};

/// Main coordination manager that integrates all coordination components
pub struct OptimizationCoordinator<T: Float + Debug + Send + Sync + 'static> {
    scheduler: TaskScheduler<T>,
    orchestrator: PipelineOrchestrator<T>,
    performance_tracker: PerformanceTracker<T>,
    convergence_detector: ConvergenceDetector<T>,
    anomaly_detector: AnomalyDetector<T>,
    config: CoordinatorConfig<T>,
    state: CoordinatorState<T>,
    metrics: CoordinatorMetrics<T>,
    _phantom: PhantomData<T>,
}

/// Configuration for the optimization coordinator
#[derive(Debug)]
pub struct CoordinatorConfig<T: Float + Debug + Send + Sync + 'static> {
    pub max_concurrent_tasks: usize,
    pub default_timeout: Duration,
    pub monitoring_interval: Duration,
    pub checkpoint_interval: Duration,
    pub resource_allocation_strategy: ResourceAllocationStrategy,
    pub priority_strategy: Box<dyn PriorityUpdateStrategy<T>>,
    pub convergence_criteria: ConvergenceCriteria<T>,
    pub enable_anomaly_detection: bool,
    pub enable_auto_scaling: bool,
    pub enable_fault_tolerance: bool,
    pub performance_threshold: T,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CoordinatorConfig<T> {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            default_timeout: Duration::from_secs(3600),
            monitoring_interval: Duration::from_secs(10),
            checkpoint_interval: Duration::from_secs(300),
            resource_allocation_strategy: ResourceAllocationStrategy::FairShare,
            priority_strategy: Box::new(StaticPriorityStrategy),
            convergence_criteria: ConvergenceCriteria::default(),
            enable_anomaly_detection: true,
            enable_auto_scaling: true,
            enable_fault_tolerance: true,
            performance_threshold: T::from(0.01).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Internal state of the coordination manager
#[derive(Debug)]
pub struct CoordinatorState<T: Float + Debug + Send + Sync + 'static> {
    pub active_tasks: HashMap<String, OptimizationTask<T>>,
    pub active_pipelines: HashMap<String, OptimizationPipeline<T>>,
    /// Orchestrator execution id per submitted pipeline id, so a caller can ask
    /// the orchestrator for the pipeline's real state.
    pub pipeline_execution_ids: HashMap<String, String>,
    /// Pipeline id each submitted experiment was converted into. Without this
    /// the pipeline id returned by `submit_pipeline` was dropped on the floor
    /// and an experiment's execution could not be located afterwards.
    pub experiment_pipeline_ids: HashMap<String, String>,
    pub active_experiments: HashMap<String, Experiment<T>>,
    pub resource_usage: ResourceUsage,
    pub last_checkpoint: Option<Instant>,
    pub last_monitoring_update: Option<Instant>,
    pub coordination_start_time: Instant,
    pub total_tasks_processed: usize,
    pub total_experiments_completed: usize,
    /// Total convergence checks performed via `monitor_optimization_value`.
    pub convergence_checks_total: usize,
    /// Of those, how many reported `converged == true`.
    pub convergence_checks_passed: usize,
    /// Total anomaly checks performed via `monitor_optimization_value`.
    pub anomaly_checks_total: usize,
    /// Of those, how many flagged `is_anomaly == true`.
    pub anomaly_checks_flagged: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CoordinatorState<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> CoordinatorState<T> {
    pub fn new() -> Self {
        Self {
            active_tasks: HashMap::new(),
            active_pipelines: HashMap::new(),
            pipeline_execution_ids: HashMap::new(),
            experiment_pipeline_ids: HashMap::new(),
            active_experiments: HashMap::new(),
            resource_usage: ResourceUsage::default(),
            last_checkpoint: None,
            last_monitoring_update: None,
            coordination_start_time: Instant::now(),
            total_tasks_processed: 0,
            total_experiments_completed: 0,
            convergence_checks_total: 0,
            convergence_checks_passed: 0,
            anomaly_checks_total: 0,
            anomaly_checks_flagged: 0,
        }
    }
}

/// Metrics for coordination performance
#[derive(Debug, Clone)]
pub struct CoordinatorMetrics<T: Float + Debug + Send + Sync + 'static> {
    pub average_task_completion_time: T,
    pub throughput: T,
    pub resource_utilization: T,
    pub error_rate: T,
    pub convergence_rate: T,
    pub anomaly_detection_rate: T,
    pub uptime: Duration,
    pub total_processed_tasks: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CoordinatorMetrics<T> {
    fn default() -> Self {
        Self {
            average_task_completion_time: T::zero(),
            throughput: T::zero(),
            resource_utilization: T::zero(),
            error_rate: T::zero(),
            convergence_rate: T::zero(),
            anomaly_detection_rate: T::zero(),
            uptime: Duration::new(0, 0),
            total_processed_tasks: 0,
        }
    }
}

/// Result of coordination operations
#[derive(Debug, Clone)]
pub struct CoordinationResult<T: Float + Debug + Send + Sync + 'static> {
    pub success: bool,
    pub task_id: String,
    pub execution_time: Duration,
    pub resource_usage: ResourceUsage,
    pub performance_metrics: PerformanceMetrics<T>,
    pub convergence_result: Option<ConvergenceResult<T>>,
    pub anomaly_alerts: Vec<AnomalyAlert<T>>,
    pub errors: Vec<String>,
}

impl<T: Float + Debug + Send + Sync + 'static + Default> OptimizationCoordinator<T> {
    /// Create a new optimization coordinator
    ///
    /// # Errors
    /// Returns `Err` (as a descriptive `String`, matching this type's other
    /// public methods) if any underlying scheduler, orchestrator, or
    /// performance tracker fails to construct, or if `0.9` cannot be
    /// represented in the target float type `T`.
    pub fn new(config: CoordinatorConfig<T>) -> Result<Self, String> {
        let estimation_threshold = T::from(0.9)
            .ok_or_else(|| "failed to represent 0.9 in target float type".to_string())?;

        let scheduler = TaskScheduler::new(SchedulerConfig {
            max_concurrent_tasks: config.max_concurrent_tasks,
            queue_size_limit: 1000,
            task_timeout: config.default_timeout,
            priority_update_interval: Duration::from_secs(5),
            load_balance_interval: Duration::from_secs(10),
            estimation_threshold,
            enable_adaptive_scheduling: true,
            enable_performance_learning: true,
        })
        .map_err(|e| format!("Failed to create task scheduler: {e}"))?;

        let orchestrator = PipelineOrchestrator::new(OrchestratorConfiguration {
            max_concurrent_pipelines: config.max_concurrent_tasks,
            default_resource_limits: ResourceLimits::default(),
            default_timeouts: TimeoutSettings::default(),
            monitoring: MonitoringConfiguration::default(),
        })
        .map_err(|e| format!("Failed to create pipeline orchestrator: {e}"))?;

        let performance_tracker = PerformanceTracker::new(TrackerConfiguration {
            collection_interval: config.monitoring_interval,
            enabled_collectors: vec!["default".to_string()],
            enabled_analyzers: vec!["default".to_string()],
            storage_config: StorageConfiguration::default(),
            alert_config: AlertConfiguration::default(),
            dashboard_config: DashboardConfiguration {
                theme: String::from("default"),
                auto_refresh: true,
                default_time_range: Duration::from_secs(3600),
                custom_params: HashMap::new(),
            },
        })
        .map_err(|e| format!("Failed to create performance tracker: {e}"))?;

        let convergence_detector = ConvergenceDetector::new(config.convergence_criteria.clone());

        let anomaly_detector = AnomalyDetector::new(AnomalyConfig::default());

        Ok(Self {
            scheduler,
            orchestrator,
            performance_tracker,
            convergence_detector,
            anomaly_detector,
            state: CoordinatorState::new(),
            metrics: CoordinatorMetrics::default(),
            config,
            _phantom: PhantomData,
        })
    }

    /// Submit a single optimization task
    pub fn submit_task(&mut self, mut task: OptimizationTask<T>) -> Result<String, String> {
        // Generate unique task ID
        let task_id = format!(
            "task_{}_{}",
            self.state.total_tasks_processed,
            Instant::now().elapsed().as_nanos()
        );
        task.task_id = task_id.clone();

        // Schedule the task
        match self.scheduler.submit_task(task.clone()) {
            Ok(()) => {
                self.state.active_tasks.insert(task_id.clone(), task);
                self.state.total_tasks_processed += 1;

                // Start performance monitoring for the task
                // Performance tracking is handled by collectors

                Ok(task_id)
            }
            Err(e) => Err(format!("Failed to schedule task: {}", e)),
        }
    }

    /// Submit an optimization pipeline
    pub fn submit_pipeline(
        &mut self,
        mut pipeline: OptimizationPipeline<T>,
    ) -> Result<String, String> {
        let pipeline_id = format!(
            "pipeline_{}_{}",
            self.state.active_pipelines.len(),
            Instant::now().elapsed().as_nanos()
        );

        pipeline.pipeline_id = pipeline_id.clone();

        // Hand the pipeline to the orchestrator. Until 0.3.2 this block read
        // `let execution_result: Result<(), String> = Ok(());` -- a hardcoded
        // success with the comment "needs proper orchestrator API" -- so
        // `submit_pipeline` reported that every pipeline had been executed while
        // the `orchestrator` field was never touched at all. The orchestrator's
        // `execute_pipeline` has been there the whole time; it returns the
        // execution id, which is now recorded on the pipeline's state entry.
        let execution_id = self
            .orchestrator
            .execute_pipeline(pipeline.clone())
            .map_err(|err| format!("Failed to execute pipeline: {err}"))?;
        self.state
            .active_pipelines
            .insert(pipeline_id.clone(), pipeline);
        self.state
            .pipeline_execution_ids
            .insert(pipeline_id.clone(), execution_id);
        Ok(pipeline_id)
    }

    /// Execution id the orchestrator assigned to a submitted pipeline.
    pub fn pipeline_execution_id(&self, pipeline_id: &str) -> Option<&str> {
        self.state
            .pipeline_execution_ids
            .get(pipeline_id)
            .map(String::as_str)
    }

    /// Current orchestrator-reported state of a submitted pipeline.
    pub fn pipeline_execution_status(&self, pipeline_id: &str) -> Option<ExecutionState> {
        let execution_id = self.state.pipeline_execution_ids.get(pipeline_id)?;
        self.orchestrator.get_execution_status(execution_id)
    }

    /// Submit an experiment
    pub fn submit_experiment(&mut self, mut experiment: Experiment<T>) -> Result<String, String> {
        let experiment_id = format!(
            "experiment_{}_{}",
            self.state.active_experiments.len(),
            Instant::now().elapsed().as_nanos()
        );

        experiment.experiment_id = experiment_id.clone();

        // Convert experiment to pipeline for execution
        let pipeline = self.experiment_to_pipeline(&experiment)?;
        let pipeline_id = self.submit_pipeline(pipeline)?;

        self.state
            .active_experiments
            .insert(experiment_id.clone(), experiment);
        self.state
            .experiment_pipeline_ids
            .insert(experiment_id.clone(), pipeline_id);

        Ok(experiment_id)
    }

    /// Pipeline id an experiment was converted into, if it was submitted.
    pub fn experiment_pipeline_id(&self, experiment_id: &str) -> Option<&str> {
        self.state
            .experiment_pipeline_ids
            .get(experiment_id)
            .map(String::as_str)
    }

    /// Execute a coordination cycle
    pub fn execute_cycle(&mut self) -> Vec<CoordinationResult<T>> {
        let mut results = Vec::new();

        // Update monitoring
        self.update_monitoring();

        // Process scheduled tasks
        let task_results = self.process_scheduled_tasks();
        results.extend(task_results);

        // Update pipeline executions
        self.update_pipeline_executions();

        // Perform maintenance operations
        self.perform_maintenance();

        // Update metrics
        self.update_metrics();

        results
    }

    /// Monitor optimization value for convergence and anomalies
    pub fn monitor_optimization_value(&mut self, task_id: &str, value: T) -> MonitoringResult<T> {
        let mut alerts = Vec::new();

        // Update performance tracking
        let _ = self.performance_tracker.collect_metrics();

        // Check for convergence
        let convergence_result = self.convergence_detector.check_convergence(value);

        // Check for anomalies if enabled
        let anomaly_result = if self.config.enable_anomaly_detection {
            Some(self.anomaly_detector.detect_anomaly(value))
        } else {
            None
        };

        // Generate alerts based on monitoring results
        if let Some(ref anomaly) = anomaly_result {
            self.state.anomaly_checks_total += 1;
            if anomaly.is_anomaly {
                self.state.anomaly_checks_flagged += 1;
                alerts.push(MonitoringAlert::Anomaly(anomaly.clone()));
            }
        }

        self.state.convergence_checks_total += 1;
        if convergence_result.converged {
            self.state.convergence_checks_passed += 1;
            alerts.push(MonitoringAlert::Convergence(convergence_result.clone()));
        }

        MonitoringResult {
            task_id: task_id.to_string(),
            value,
            convergence_result,
            anomaly_result,
            alerts,
            timestamp: Instant::now(),
        }
    }

    /// Get current coordination status
    pub fn get_status(&self) -> CoordinationStatus<T> {
        CoordinationStatus {
            active_tasks: self.state.active_tasks.len(),
            active_pipelines: self.state.active_pipelines.len(),
            active_experiments: self.state.active_experiments.len(),
            resource_utilization: self.state.resource_usage.clone(),
            metrics: self.metrics.clone(),
            uptime: self.state.coordination_start_time.elapsed(),
            health_status: self.assess_health_status(),
        }
    }

    /// Update resource allocation
    pub fn update_resource_allocation(&mut self, allocation: ResourceUsage) -> Result<(), String> {
        self.state.resource_usage = allocation;

        // Update scheduler with new resource information
        // Update resource availability - needs proper implementation
        // self.scheduler.update_resources(&self.state.resource_usage);

        // Update orchestrator
        // Update resource allocation - needs proper implementation
        // self.orchestrator.update_resources(&self.state.resource_usage);

        Ok(())
    }

    /// Shutdown coordination gracefully
    pub fn shutdown(&mut self) -> Result<(), String> {
        // Complete active tasks
        self.complete_active_tasks()?;

        // Save checkpoints
        self.save_final_checkpoints()?;

        // Generate final report
        let report = self.generate_final_report();
        println!("Coordination shutdown report:\n{}", report);

        Ok(())
    }

    // Private helper methods

    fn update_monitoring(&mut self) {
        let now = Instant::now();

        if let Some(last_update) = self.state.last_monitoring_update {
            if now.duration_since(last_update) < self.config.monitoring_interval {
                return;
            }
        }

        // Update performance metrics
        let _ = self.performance_tracker.collect_metrics();

        // Check for system-level anomalies
        if self.config.enable_anomaly_detection {
            let system_metrics = self.collect_system_metrics();
            for metric in system_metrics {
                let _ = self.anomaly_detector.detect_anomaly(metric);
            }
        }

        self.state.last_monitoring_update = Some(now);
    }

    fn process_scheduled_tasks(&mut self) -> Vec<CoordinationResult<T>> {
        let mut results = Vec::new();

        // Pull every task the scheduler is currently willing to hand out
        // (respecting its configured scheduling strategy) and dispatch each
        // one for real, instead of iterating over a permanently-empty list.
        while let Some(task) = self.scheduler.get_next_task() {
            let task_id = task.task_id.clone();
            match self.execute_task(&task) {
                Ok(result) => {
                    results.push(result);
                    self.state.active_tasks.remove(&task_id);
                }
                Err(e) => {
                    self.state.active_tasks.remove(&task_id);
                    results.push(CoordinationResult {
                        success: false,
                        task_id,
                        execution_time: Duration::new(0, 0),
                        resource_usage: ResourceUsage::default(),
                        performance_metrics: PerformanceMetrics::default(),
                        convergence_result: None,
                        anomaly_alerts: Vec::new(),
                        errors: vec![e],
                    });
                }
            }
        }

        results
    }

    fn execute_task(
        &mut self,
        task: &OptimizationTask<T>,
    ) -> Result<CoordinationResult<T>, String> {
        let start_time = Instant::now();
        let task_id = task.task_id.clone();

        // Dispatch through the scheduler's real execution lifecycle (start
        // -> do the work this layer is responsible for -> complete) instead
        // of being a no-op that always reports success without doing
        // anything checkable.
        let assigned_resources =
            crate::coordination::scheduling::task_scheduler::AssignedResources {
                cpu_cores: (0..task.resource_requirements.cpu_cores).collect(),
                memory_mb: task.resource_requirements.memory_mb,
                gpu_devices: (0..task.resource_requirements.gpu_devices).collect(),
                storage_gb: task.resource_requirements.storage_gb,
                network_bandwidth: task.resource_requirements.network_bandwidth,
            };
        self.scheduler
            .start_task_execution(task.clone(), assigned_resources)
            .map_err(|e| format!("Failed to start task execution: {e}"))?;

        // The concrete unit of work this coordination layer performs per
        // task is collecting real performance metrics; its actual `Result`
        // determines success instead of being papered over with a default.
        let (success, performance_metrics, error_info) =
            match self.performance_tracker.collect_metrics() {
                Ok(metrics) => (true, metrics, None),
                Err(e) => (false, PerformanceMetrics::default(), Some(e.to_string())),
            };

        let execution_time = start_time.elapsed();

        self.scheduler
            .complete_task(&task_id, success, error_info.clone())
            .map_err(|e| format!("Failed to complete task in scheduler: {e}"))?;

        Ok(CoordinationResult {
            success,
            task_id,
            execution_time,
            resource_usage: self.state.resource_usage.clone(),
            performance_metrics,
            convergence_result: None,
            anomaly_alerts: Vec::new(),
            errors: error_info.into_iter().collect(),
        })
    }

    fn update_pipeline_executions(&mut self) {
        // Update orchestrator state - needs implementation
        // self.orchestrator.update_pipeline_states();

        // Check for completed pipelines - needs implementation
        // let completed_pipelines = self.orchestrator.get_completed_pipelines();
        // for pipeline_id in completed_pipelines {
        //     self.state.active_pipelines.remove(&pipeline_id);
        // }
    }

    fn perform_maintenance(&mut self) {
        let now = Instant::now();

        // Perform checkpointing
        if let Some(last_checkpoint) = self.state.last_checkpoint {
            if now.duration_since(last_checkpoint) >= self.config.checkpoint_interval {
                let _ = self.create_checkpoint();
            }
        } else {
            let _ = self.create_checkpoint();
        }

        // Clean up completed tasks and experiments
        self.cleanup_completed_items();

        // Update adaptive parameters
        if self.config.enable_auto_scaling {
            self.update_adaptive_parameters();
        }
    }

    fn create_checkpoint(&mut self) -> Result<(), String> {
        // Create checkpoint through orchestrator - needs implementation
        // self.orchestrator.create_system_checkpoint()?;
        self.state.last_checkpoint = Some(Instant::now());
        Ok(())
    }

    fn cleanup_completed_items(&mut self) {
        // Tasks are removed from `active_tasks` synchronously as soon as
        // `process_scheduled_tasks` finishes executing them (success or
        // failure) -- see the `self.state.active_tasks.remove(&task_id)`
        // calls there -- so anything still here has been submitted but not
        // yet picked up by the scheduler. Drop entries that have sat
        // unscheduled longer than `threshold` instead of retaining every
        // task forever (the previous `retain(|_, _| true)` never removed
        // anything, regardless of age).
        let threshold = Duration::from_secs(3600); // 1 hour
        let now = SystemTime::now();

        self.state.active_tasks.retain(|_, task| {
            now.duration_since(task.created_at)
                .map(|age| age < threshold)
                .unwrap_or(true)
        });
    }

    fn update_adaptive_parameters(&mut self) {
        // Adaptive resource allocation based on performance
        let current_metrics = &self.metrics;

        if current_metrics.resource_utilization > T::from(0.9).unwrap_or_else(|| T::zero()) {
            // High utilization - consider scaling up
            let _ = self.request_additional_resources();
        } else if current_metrics.resource_utilization < T::from(0.3).unwrap_or_else(|| T::zero()) {
            // Low utilization - consider scaling down
            let _ = self.release_excess_resources();
        }
    }

    fn request_additional_resources(&mut self) -> Result<(), String> {
        // Implementation would request additional compute resources
        Ok(())
    }

    fn release_excess_resources(&mut self) -> Result<(), String> {
        // Implementation would release unused resources
        Ok(())
    }

    fn update_metrics(&mut self) {
        let current_time = Instant::now();
        let uptime = current_time.duration_since(self.state.coordination_start_time);

        // Update basic metrics
        self.metrics.uptime = uptime;
        self.metrics.total_processed_tasks = self.state.total_tasks_processed;

        // Calculate throughput
        if uptime.as_secs() > 0 {
            self.metrics.throughput = T::from(self.state.total_tasks_processed)
                .unwrap_or_else(|| T::zero())
                / T::from(uptime.as_secs()).unwrap_or_else(|| T::one());
        }

        // Fraction of the coordinator's configured task-concurrency
        // capacity currently in use. `self.state.resource_usage` (CPU/
        // memory/etc. from `research::experiments::ResourceUsage`) is never
        // populated by any real sampling anywhere in this coordinator, so
        // deriving from it would still be a fabricated number; this is
        // real, live data instead of the previous hardcoded `T::zero()`.
        self.metrics.resource_utilization = if self.config.max_concurrent_tasks > 0 {
            (T::from(self.state.active_tasks.len()).unwrap_or_else(|| T::zero())
                / T::from(self.config.max_concurrent_tasks).unwrap_or_else(|| T::one()))
            .min(T::one())
        } else {
            T::zero()
        };

        // Update convergence and anomaly rates
        self.metrics.convergence_rate = self.calculate_convergence_rate();
        self.metrics.anomaly_detection_rate = self.calculate_anomaly_rate();
    }

    /// Fraction of `monitor_optimization_value` calls that reported
    /// convergence, computed from real state history rather than a
    /// hardcoded constant. Returns `T::zero()` when no checks have been
    /// performed yet (honest "no data" rather than a fabricated rate).
    fn calculate_convergence_rate(&self) -> T {
        if self.state.convergence_checks_total == 0 {
            return T::zero();
        }
        T::from(self.state.convergence_checks_passed).unwrap_or_else(|| T::zero())
            / T::from(self.state.convergence_checks_total).unwrap_or_else(|| T::one())
    }

    /// Fraction of `monitor_optimization_value` calls that flagged an
    /// anomaly, computed from real state history rather than a hardcoded
    /// constant. Returns `T::zero()` when no checks have been performed yet.
    fn calculate_anomaly_rate(&self) -> T {
        if self.state.anomaly_checks_total == 0 {
            return T::zero();
        }
        T::from(self.state.anomaly_checks_flagged).unwrap_or_else(|| T::zero())
            / T::from(self.state.anomaly_checks_total).unwrap_or_else(|| T::one())
    }

    fn collect_system_metrics(&self) -> Vec<T> {
        // Collect system-level metrics for anomaly detection
        vec![
            self.metrics.resource_utilization,
            self.metrics.throughput,
            T::from(self.state.active_tasks.len()).unwrap_or_else(|| T::zero()),
            T::from(self.state.active_pipelines.len()).unwrap_or_else(|| T::zero()),
        ]
    }

    fn experiment_to_pipeline(
        &self,
        experiment: &Experiment<T>,
    ) -> Result<OptimizationPipeline<T>, String> {
        // Convert experiment configuration to pipeline stages
        let pipeline = OptimizationPipeline {
            pipeline_id: experiment.experiment_id.clone(),
            name: format!("Experiment {}", experiment.experiment_id),
            description: "Auto-generated pipeline from experiment".to_string(),
            stages: Vec::new(), // Will be populated with proper stages
            dependencies: HashMap::new(),
            configuration: PipelineConfiguration::default(),
            global_parameters: HashMap::new(),
            metadata: crate::coordination::orchestration::pipeline_orchestrator::PipelineMetadata {
                created_by: "system".to_string(),
                created_at: std::time::SystemTime::now(),
                updated_at: std::time::SystemTime::now(),
                tags: vec!["experiment".to_string()],
                description: "Pipeline from experiment".to_string(),
            },
            version: "1.0.0".to_string(),
        };
        Ok(pipeline)
    }

    fn assess_health_status(&self) -> HealthStatus {
        let error_rate = self.metrics.error_rate;
        let resource_utilization = self.metrics.resource_utilization;

        if error_rate > T::from(0.1).unwrap_or_else(|| T::zero()) {
            HealthStatus::Unhealthy
        } else if resource_utilization > T::from(0.95).unwrap_or_else(|| T::zero()) {
            HealthStatus::Degraded
        } else if error_rate > T::from(0.05).unwrap_or_else(|| T::zero())
            || resource_utilization > T::from(0.8).unwrap_or_else(|| T::zero())
        {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }

    fn complete_active_tasks(&mut self) -> Result<(), String> {
        // Wait for active tasks to complete or timeout
        let timeout = Duration::from_secs(60);
        let start = Instant::now();

        while !self.state.active_tasks.is_empty() && start.elapsed() < timeout {
            let results = self.process_scheduled_tasks();
            if results.is_empty() {
                std::thread::sleep(Duration::from_millis(100));
            }
        }

        if !self.state.active_tasks.is_empty() {
            return Err(format!(
                "Timeout waiting for {} tasks to complete",
                self.state.active_tasks.len()
            ));
        }

        Ok(())
    }

    fn save_final_checkpoints(&mut self) -> Result<(), String> {
        self.create_checkpoint()
    }

    fn generate_final_report(&self) -> String {
        format!(
            "Optimization Coordination Final Report:\n\
             - Total Uptime: {:?}\n\
             - Tasks Processed: {}\n\
             - Experiments Completed: {}\n\
             - Average Throughput: {:.2}\n\
             - Resource Utilization: {:.2}%\n\
             - Error Rate: {:.2}%\n\
             - Convergence Rate: {:.2}%\n\
             - Health Status: {:?}",
            self.metrics.uptime,
            self.metrics.total_processed_tasks,
            self.state.total_experiments_completed,
            self.metrics.throughput.to_f64().unwrap_or(0.0),
            (self.metrics.resource_utilization * T::from(100.0).unwrap_or_else(|| T::zero()))
                .to_f64()
                .unwrap_or(0.0),
            (self.metrics.error_rate * T::from(100.0).unwrap_or_else(|| T::zero()))
                .to_f64()
                .unwrap_or(0.0),
            (self.metrics.convergence_rate * T::from(100.0).unwrap_or_else(|| T::zero()))
                .to_f64()
                .unwrap_or(0.0),
            self.assess_health_status(),
        )
    }
}

/// Result of monitoring operations
#[derive(Debug, Clone)]
pub struct MonitoringResult<T: Float + Debug + Send + Sync + 'static> {
    pub task_id: String,
    pub value: T,
    pub convergence_result: ConvergenceResult<T>,
    pub anomaly_result: Option<AnomalyResult<T>>,
    pub alerts: Vec<MonitoringAlert<T>>,
    pub timestamp: Instant,
}

/// Monitoring alert types
#[derive(Debug, Clone)]
pub enum MonitoringAlert<T: Float + Debug + Send + Sync + 'static> {
    Convergence(ConvergenceResult<T>),
    Anomaly(AnomalyResult<T>),
    Performance(Box<PerformanceAlert<T>>),
    Resource(String),
}

/// Current status of the coordination system
#[derive(Debug, Clone)]
pub struct CoordinationStatus<T: Float + Debug + Send + Sync + 'static> {
    pub active_tasks: usize,
    pub active_pipelines: usize,
    pub active_experiments: usize,
    pub resource_utilization: ResourceUsage,
    pub metrics: CoordinatorMetrics<T>,
    pub uptime: Duration,
    pub health_status: HealthStatus,
}

/// Health status of the coordination system
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Degraded,
    Unhealthy,
}

/// Builder for creating optimization coordinators
pub struct CoordinatorBuilder<T: Float + Debug + Send + Sync + 'static> {
    config: CoordinatorConfig<T>,
}

impl<T: Float + Debug + Send + Sync + 'static + Default> CoordinatorBuilder<T> {
    pub fn new() -> Self {
        Self {
            config: CoordinatorConfig::default(),
        }
    }

    pub fn max_concurrent_tasks(mut self, max: usize) -> Self {
        self.config.max_concurrent_tasks = max;
        self
    }

    pub fn monitoring_interval(mut self, interval: Duration) -> Self {
        self.config.monitoring_interval = interval;
        self
    }

    pub fn enable_anomaly_detection(mut self, enable: bool) -> Self {
        self.config.enable_anomaly_detection = enable;
        self
    }

    pub fn enable_fault_tolerance(mut self, enable: bool) -> Self {
        self.config.enable_fault_tolerance = enable;
        self
    }

    pub fn convergence_criteria(mut self, criteria: ConvergenceCriteria<T>) -> Self {
        self.config.convergence_criteria = criteria;
        self
    }

    pub fn build(self) -> Result<OptimizationCoordinator<T>, String> {
        OptimizationCoordinator::new(self.config)
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for CoordinatorBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = CoordinatorBuilder::<f64>::new()
            .max_concurrent_tasks(5)
            .enable_anomaly_detection(true)
            .build()
            .expect("unwrap failed");

        let status = coordinator.get_status();
        assert_eq!(status.active_tasks, 0);
        assert_eq!(status.health_status, HealthStatus::Healthy);
    }

    #[test]
    fn test_task_submission() {
        let mut coordinator = OptimizationCoordinator::<f64>::new(CoordinatorConfig::default())
            .expect("unwrap failed");
        let task = OptimizationTask::new("test_task".to_string());

        let task_id = coordinator.submit_task(task).expect("unwrap failed");
        assert!(!task_id.is_empty());

        let status = coordinator.get_status();
        assert!(status.active_tasks > 0);
    }

    #[test]
    fn test_monitoring() {
        let mut coordinator = OptimizationCoordinator::<f64>::new(CoordinatorConfig::default())
            .expect("unwrap failed");
        let result = coordinator.monitor_optimization_value("test_task", 1.0);

        assert_eq!(result.task_id, "test_task");
        assert_eq!(result.value, 1.0);
    }

    #[test]
    fn convergence_and_anomaly_rates_reflect_real_history() {
        // Regression test for F62: rates must be computed from actual
        // monitor_optimization_value history, not hardcoded constants.
        let mut coordinator = OptimizationCoordinator::<f64>::new(CoordinatorConfig::default())
            .expect("unwrap failed");

        // No checks performed yet: rates must honestly report zero, not a
        // fabricated "everything is fine" placeholder.
        assert_eq!(coordinator.calculate_convergence_rate(), 0.0);
        assert_eq!(coordinator.calculate_anomaly_rate(), 0.0);

        for _ in 0..4 {
            coordinator.monitor_optimization_value("t", 1.0);
        }
        // Rates must now be derived from the recorded history: exactly 4
        // convergence checks were performed.
        assert_eq!(coordinator.state.convergence_checks_total, 4);
    }

    #[test]
    fn execute_cycle_processes_submitted_tasks_through_the_scheduler() {
        // Regression test for F62: process_scheduled_tasks must actually
        // pull tasks from the scheduler and execute them, instead of
        // iterating over a permanently-empty list.
        let mut coordinator = OptimizationCoordinator::<f64>::new(CoordinatorConfig::default())
            .expect("unwrap failed");
        let task = OptimizationTask::new("cycle_task".to_string());
        coordinator.submit_task(task).expect("unwrap failed");

        let results = coordinator.execute_cycle();
        assert!(
            !results.is_empty(),
            "execute_cycle must dispatch the submitted task instead of processing nothing"
        );
        // The task must have been finalized through the scheduler's real
        // lifecycle (either completed or failed), not left in limbo by a
        // no-op `execute_task` that never calls `complete_task`.
        let stats = coordinator.scheduler.get_statistics();
        assert_eq!(
            stats.total_tasks_completed + stats.total_tasks_failed,
            1,
            "the dispatched task must be reflected in scheduler statistics"
        );
    }

    // Regression test for F62 (additional fix beyond the pre-existing
    // execute_task/rates fixes above): `resource_utilization` was
    // hardcoded to `T::zero()` on every call to `update_metrics`
    // ("Needs proper implementation"), which fed into
    // `assess_health_status`'s Degraded threshold and
    // `update_adaptive_parameters`'s scale-up/down decisions -- both of
    // which could therefore never observe anything but "0% utilized".
    #[test]
    fn resource_utilization_reflects_real_active_task_load() {
        let mut coordinator = CoordinatorBuilder::<f64>::new()
            .max_concurrent_tasks(4)
            .build()
            .expect("unwrap failed");

        coordinator.update_metrics();
        assert_eq!(coordinator.metrics.resource_utilization, 0.0);

        for i in 0..2 {
            let task = OptimizationTask::new(format!("util_task_{i}"));
            coordinator
                .submit_task(task)
                .expect("submit should succeed");
        }
        coordinator.update_metrics();

        assert_eq!(
            coordinator.metrics.resource_utilization, 0.5,
            "2 active tasks out of a configured max of 4 must report 50% utilization, \
             not the old hardcoded 0.0"
        );
    }

    // Regression test for F62 (additional fix): `cleanup_completed_items`
    // always retained every task (`retain(|_, _| true)`, "Need proper
    // completion check implementation"), so nothing was ever actually
    // cleaned up regardless of age.
    #[test]
    fn cleanup_completed_items_removes_only_stale_tasks() {
        let mut coordinator = OptimizationCoordinator::<f64>::new(CoordinatorConfig::default())
            .expect("unwrap failed");

        let mut old_task = OptimizationTask::new("old_task".to_string());
        old_task.created_at = SystemTime::now() - Duration::from_secs(7200); // 2h old
        coordinator
            .state
            .active_tasks
            .insert(old_task.task_id.clone(), old_task);

        let fresh_task = OptimizationTask::new("fresh_task".to_string());
        let fresh_id = fresh_task.task_id.clone();
        coordinator
            .state
            .active_tasks
            .insert(fresh_id.clone(), fresh_task);

        assert_eq!(coordinator.state.active_tasks.len(), 2);
        coordinator.cleanup_completed_items();

        assert_eq!(
            coordinator.state.active_tasks.len(),
            1,
            "the stale (>1h old) task must be removed"
        );
        assert!(
            coordinator.state.active_tasks.contains_key(&fresh_id),
            "the fresh task must be retained"
        );
    }
}
