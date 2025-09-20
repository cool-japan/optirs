// Experiment tracking and management for research projects
//
// This module provides comprehensive tools for designing, executing, and tracking
// machine learning optimization experiments with full reproducibility support.

use crate::error::{OptimError, Result};
use crate::optimizers::*;
use crate::unified_api::OptimizerConfig;
use chrono::{DateTime, Utc};
use num_traits::Float;
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive experiment definition and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Experiment identifier
    pub id: String,
    /// Experiment name
    pub name: String,
    /// Research hypothesis
    pub hypothesis: String,
    /// Experiment description
    pub description: String,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Experiment configuration
    pub config: ExperimentConfig,
    /// Optimizer configurations being tested
    pub optimizer_configs: HashMap<String, OptimizerConfig<f64>>,
    /// Dataset information
    pub dataset_info: DatasetInfo,
    /// Metrics to track
    pub metrics: Vec<String>,
    /// Experiment results
    pub results: Vec<ExperimentResult>,
    /// Reproducibility information
    pub reproducibility: ReproducibilityInfo,
    /// Experiment timeline
    pub timeline: ExperimentTimeline,
    /// Analysis notes
    pub notes: Vec<ExperimentNote>,
    /// Experiment metadata
    pub metadata: ExperimentMetadata,
}

/// Experiment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Planning phase
    Planning,
    /// Ready to run
    Ready,
    /// Currently running
    Running,
    /// Completed successfully
    Completed,
    /// Failed with errors
    Failed,
    /// Paused/suspended
    Paused,
    /// Cancelled
    Cancelled,
    /// Under analysis
    Analyzing,
    /// Results published
    Published,
}

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Number of runs/repetitions
    pub num_runs: usize,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Hardware configuration
    pub hardware_config: HardwareConfig,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Validation split
    pub validation_split: f64,
    /// Test split
    pub test_split: f64,
    /// Cross-validation folds
    pub cv_folds: Option<usize>,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Metric to monitor
    pub monitor_metric: String,
    /// Patience (epochs without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Mode (minimize or maximize)
    pub mode: OptimizationMode,
}

/// Optimization mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationMode {
    /// Minimize the metric
    Minimize,
    /// Maximize the metric
    Maximize,
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareConfig {
    /// CPU information
    pub cpu_info: CpuInfo,
    /// GPU information
    pub gpu_info: Option<GpuInfo>,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Parallel processing settings
    pub parallel_config: ParallelConfig,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU model
    pub model: String,
    /// Number of cores
    pub cores: usize,
    /// Number of threads
    pub threads: usize,
    /// CPU frequency (MHz)
    pub frequency_mhz: u32,
    /// Cache sizes
    pub cache_sizes: Vec<String>,
    /// SIMD capabilities
    pub simd_capabilities: Vec<String>,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU model
    pub model: String,
    /// Memory size (MB)
    pub memory_mb: usize,
    /// Compute capability
    pub compute_capability: String,
    /// CUDA version
    pub cuda_version: Option<String>,
    /// Driver version
    pub driver_version: String,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Total system memory (MB)
    pub total_memory_mb: usize,
    /// Available memory (MB)
    pub available_memory_mb: usize,
    /// Memory allocation strategy
    pub allocation_strategy: MemoryAllocationStrategy,
    /// Memory pool size (MB)
    pub pool_size_mb: Option<usize>,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryAllocationStrategy {
    /// Standard allocation
    Standard,
    /// Pool-based allocation
    Pooled,
    /// Memory-mapped allocation
    MemoryMapped,
    /// Compressed allocation
    Compressed,
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of threads
    pub num_threads: usize,
    /// Thread affinity
    pub thread_affinity: Option<Vec<usize>>,
    /// Work stealing enabled
    pub work_stealing: bool,
    /// Chunk size for parallel operations
    pub chunk_size: Option<usize>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Dataset description
    pub description: String,
    /// Dataset source/URL
    pub source: String,
    /// Dataset version
    pub version: String,
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
    /// Data type
    pub data_type: DataType,
    /// Dataset statistics
    pub statistics: DatasetStatistics,
    /// Data preprocessing steps
    pub preprocessing: Vec<PreprocessingStep>,
}

/// Data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    /// Tabular data
    Tabular,
    /// Image data
    Image,
    /// Text data
    Text,
    /// Audio data
    Audio,
    /// Video data
    Video,
    /// Time series data
    TimeSeries,
    /// Graph data
    Graph,
    /// Multi-modal data
    MultiModal,
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetStatistics {
    /// Feature means
    pub feature_means: Vec<f64>,
    /// Feature standard deviations
    pub feature_stds: Vec<f64>,
    /// Feature ranges
    pub feature_ranges: Vec<(f64, f64)>,
    /// Class distribution (for classification)
    pub class_distribution: Option<HashMap<String, usize>>,
    /// Missing value counts
    pub missing_values: Vec<usize>,
    /// Correlation matrix
    pub correlation_matrix: Option<Array2<f64>>,
}

/// Preprocessing step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStep {
    /// Step name
    pub name: String,
    /// Step description
    pub description: String,
    /// Parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Order in preprocessing pipeline
    pub order: usize,
}

/// Experiment result for a single run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Run identifier
    pub run_id: String,
    /// Optimizer name
    pub optimizer_name: String,
    /// Start timestamp
    pub start_time: DateTime<Utc>,
    /// End timestamp
    pub end_time: Option<DateTime<Utc>>,
    /// Run status
    pub status: RunStatus,
    /// Final metrics
    pub final_metrics: HashMap<String, f64>,
    /// Training history
    pub training_history: TrainingHistory,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Error information (if failed)
    pub error_info: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Run status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunStatus {
    /// Run completed successfully
    Success,
    /// Run failed with error
    Failed,
    /// Run was terminated early
    Terminated,
    /// Run timed out
    Timeout,
    /// Run was cancelled
    Cancelled,
}

/// Training history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Epoch numbers
    pub epochs: Vec<usize>,
    /// Training metrics by epoch
    pub train_metrics: HashMap<String, Vec<f64>>,
    /// Validation metrics by epoch
    pub val_metrics: HashMap<String, Vec<f64>>,
    /// Learning rates by epoch
    pub learning_rates: Vec<f64>,
    /// Gradient norms by epoch
    pub gradient_norms: Vec<f64>,
    /// Parameter norms by epoch
    pub parameter_norms: Vec<f64>,
    /// Step times by epoch
    pub step_times: Vec<f64>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak CPU usage (%)
    pub peak_cpu_usage: f64,
    /// Average CPU usage (%)
    pub avg_cpu_usage: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: usize,
    /// Average memory usage (MB)
    pub avg_memory_mb: usize,
    /// Peak GPU memory usage (MB)
    pub peak_gpu_memory_mb: Option<usize>,
    /// Total training time (seconds)
    pub total_time_seconds: f64,
    /// Energy consumption (Joules)
    pub energy_consumption_joules: Option<f64>,
}

/// Reproducibility information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReproducibilityInfo {
    /// Environment hash for reproducibility
    pub environment_hash: String,
    /// Git commit hash
    pub git_commit: Option<String>,
    /// Code checksum
    pub code_checksum: String,
    /// Dependency versions
    pub dependency_versions: HashMap<String, String>,
    /// System information
    pub system_info: SystemInfo,
    /// Reproducibility checklist
    pub checklist: ReproducibilityChecklist,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// OS version
    pub os_version: String,
    /// Architecture
    pub architecture: String,
    /// Hostname
    pub hostname: String,
    /// Username
    pub username: String,
    /// Timezone
    pub timezone: String,
}

/// Reproducibility checklist
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReproducibilityChecklist {
    /// Random seed set
    pub random_seed_set: bool,
    /// Dependencies pinned
    pub dependencies_pinned: bool,
    /// Data version controlled
    pub data_version_controlled: bool,
    /// Code version controlled
    pub code_version_controlled: bool,
    /// Environment documented
    pub environment_documented: bool,
    /// Hardware documented
    pub hardware_documented: bool,
    /// Results archived
    pub results_archived: bool,
}

/// Experiment timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTimeline {
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Started timestamp
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Estimated duration
    pub estimated_duration: Option<chrono::Duration>,
    /// Actual duration
    pub actual_duration: Option<chrono::Duration>,
}

/// Experiment note
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentNote {
    /// Note timestamp
    pub timestamp: DateTime<Utc>,
    /// Note author
    pub author: String,
    /// Note content
    pub content: String,
    /// Note type
    pub note_type: NoteType,
    /// Associated run ID
    pub run_id: Option<String>,
}

/// Note types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NoteType {
    /// General observation
    Observation,
    /// Issue or problem
    Issue,
    /// Solution or fix
    Solution,
    /// Hypothesis
    Hypothesis,
    /// Conclusion
    Conclusion,
    /// Question
    Question,
    /// Reminder
    Reminder,
}

/// Experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExperimentMetadata {
    /// Experiment tags
    pub tags: Vec<String>,
    /// Research question
    pub research_question: String,
    /// Expected outcomes
    pub expected_outcomes: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Related experiments
    pub related_experiments: Vec<String>,
    /// References/citations
    pub references: Vec<String>,
}

/// Experiment runner for executing experiments
pub struct ExperimentRunner {
    /// Current experiment
    experiment: Experiment,
    /// Resource monitor
    resource_monitor: ResourceMonitor,
    /// Progress callback
    progress_callback: Option<Box<dyn Fn(f64) + Send + Sync>>,
}

impl std::fmt::Debug for ExperimentRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExperimentRunner")
            .field("experiment", &self.experiment)
            .field("resource_monitor", &self.resource_monitor)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

/// Resource monitoring
#[derive(Debug)]
pub struct ResourceMonitor {
    /// CPU usage history
    cpu_usage: Vec<f64>,
    /// Memory usage history  
    memory_usage: Vec<usize>,
    /// GPU memory usage history
    gpu_memory_usage: Vec<Option<usize>>,
    /// Monitoring interval (seconds)
    interval_seconds: u64,
}

impl Experiment {
    /// Create a new experiment
    pub fn new(name: &str) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            hypothesis: String::new(),
            description: String::new(),
            status: ExperimentStatus::Planning,
            config: ExperimentConfig::default(),
            optimizer_configs: HashMap::new(),
            dataset_info: DatasetInfo::default(),
            metrics: Vec::new(),
            results: Vec::new(),
            reproducibility: ReproducibilityInfo::default(),
            timeline: ExperimentTimeline {
                created_at: now,
                started_at: None,
                completed_at: None,
                estimated_duration: None,
                actual_duration: None,
            },
            notes: Vec::new(),
            metadata: ExperimentMetadata::default(),
        }
    }

    /// Set experiment hypothesis
    pub fn hypothesis(mut self, hypothesis: &str) -> Self {
        self.hypothesis = hypothesis.to_string();
        self
    }

    /// Set experiment description
    pub fn description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Add optimizer configuration
    pub fn add_optimizer_config(mut self, name: &str, config: OptimizerConfig<f64>) -> Self {
        self.optimizer_configs.insert(name.to_string(), config);
        self
    }

    /// Set dataset information
    pub fn dataset(mut self, datasetinfo: DatasetInfo) -> Self {
        self.dataset_info = datasetinfo;
        self
    }

    /// Set metrics to track
    pub fn metrics(mut self, metrics: Vec<String>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Add a note to the experiment
    pub fn add_note(&mut self, author: &str, content: &str, notetype: NoteType) {
        let note = ExperimentNote {
            timestamp: Utc::now(),
            author: author.to_string(),
            content: content.to_string(),
            note_type: notetype,
            run_id: None,
        };
        self.notes.push(note);
    }

    /// Start the experiment
    pub fn start(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Ready && self.status != ExperimentStatus::Planning {
            return Err(OptimError::InvalidConfig(format!(
                "Cannot start experiment in status {:?}",
                self.status
            )));
        }

        self.status = ExperimentStatus::Running;
        self.timeline.started_at = Some(Utc::now());

        Ok(())
    }

    /// Complete the experiment
    pub fn complete(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Running {
            return Err(OptimError::InvalidConfig(format!(
                "Cannot complete experiment in status {:?}",
                self.status
            )));
        }

        self.status = ExperimentStatus::Completed;
        self.timeline.completed_at = Some(Utc::now());

        if let (Some(start), Some(end)) = (self.timeline.started_at, self.timeline.completed_at) {
            self.timeline.actual_duration = Some(end - start);
        }

        Ok(())
    }

    /// Generate experiment report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("# Experiment Report: {}\n\n", self.name));
        report.push_str(&format!("**ID**: {}\n", self.id));
        report.push_str(&format!("**Status**: {:?}\n", self.status));
        report.push_str(&format!("**Hypothesis**: {}\n\n", self.hypothesis));

        if !self.description.is_empty() {
            report.push_str(&format!("## Description\n\n{}\n\n", self.description));
        }

        report.push_str("## Configuration\n\n");
        report.push_str(&format!("- **Random Seed**: {}\n", self.config.random_seed));
        report.push_str(&format!("- **Number of Runs**: {}\n", self.config.num_runs));
        report.push_str(&format!("- **Max Epochs**: {}\n", self.config.max_epochs));

        report.push_str("\n## Optimizers\n\n");
        for name in self.optimizer_configs.keys() {
            report.push_str(&format!("- {}\n", name));
        }

        report.push_str("\n## Dataset\n\n");
        report.push_str(&format!("- **Name**: {}\n", self.dataset_info.name));
        report.push_str(&format!(
            "- **Samples**: {}\n",
            self.dataset_info.num_samples
        ));
        report.push_str(&format!(
            "- **Features**: {}\n",
            self.dataset_info.num_features
        ));

        report.push_str("\n## Results\n\n");
        report.push_str(&format!("**Total Runs**: {}\n\n", self.results.len()));

        // Group results by optimizer
        let mut optimizer_results: HashMap<String, Vec<&ExperimentResult>> = HashMap::new();
        for result in &self.results {
            optimizer_results
                .entry(result.optimizer_name.clone())
                .or_default()
                .push(result);
        }

        for (optimizer, results) in optimizer_results {
            report.push_str(&format!("### {}\n\n", optimizer));

            if !results.is_empty() {
                // Calculate statistics
                let successful_runs: Vec<&ExperimentResult> = results
                    .iter()
                    .filter(|r| r.status == RunStatus::Success)
                    .copied()
                    .collect();

                report.push_str(&format!(
                    "- **Successful Runs**: {}/{}\n",
                    successful_runs.len(),
                    results.len()
                ));

                if !successful_runs.is_empty() {
                    for metric in &self.metrics {
                        if let Some(values) = self.get_metric_values(&successful_runs, metric) {
                            let mean = values.iter().sum::<f64>() / values.len() as f64;
                            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                / values.len() as f64)
                                .sqrt();
                            report
                                .push_str(&format!("- **{}**: {:.4} ± {:.4}\n", metric, mean, std));
                        }
                    }
                }
            }
            report.push('\n');
        }

        if !self.notes.is_empty() {
            report.push_str("## Notes\n\n");
            for note in &self.notes {
                report.push_str(&format!(
                    "**{}** ({}): {}\n\n",
                    note.author,
                    note.timestamp.format("%Y-%m-%d %H:%M"),
                    note.content
                ));
            }
        }

        report
    }

    fn get_metric_values(&self, results: &[&ExperimentResult], metric: &str) -> Option<Vec<f64>> {
        let mut values = Vec::new();
        for result in results {
            if let Some(&value) = result.final_metrics.get(metric) {
                values.push(value);
            }
        }
        if values.is_empty() {
            None
        } else {
            Some(values)
        }
    }
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            random_seed: 42,
            num_runs: 1,
            max_epochs: 100,
            early_stopping: None,
            hardware_config: HardwareConfig::default(),
            environment: HashMap::new(),
            validation_split: 0.2,
            test_split: 0.1,
            cv_folds: None,
        }
    }
}

impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            model: "Unknown".to_string(),
            cores: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            frequency_mhz: 0,
            cache_sizes: Vec::new(),
            simd_capabilities: Vec::new(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            total_memory_mb: 8192,     // 8GB default
            available_memory_mb: 6144, // 6GB default
            allocation_strategy: MemoryAllocationStrategy::Standard,
            pool_size_mb: None,
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            thread_affinity: None,
            work_stealing: true,
            chunk_size: None,
        }
    }
}

impl Default for DatasetInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            description: String::new(),
            source: String::new(),
            version: "1.0".to_string(),
            num_samples: 0,
            num_features: 0,
            num_classes: None,
            data_type: DataType::Tabular,
            statistics: DatasetStatistics::default(),
            preprocessing: Vec::new(),
        }
    }
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            os_version: String::new(),
            architecture: std::env::consts::ARCH.to_string(),
            hostname: String::new(),
            username: std::env::var("USER").unwrap_or_else(|_| "unknown".to_string()),
            timezone: String::new(),
        }
    }
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(_intervalseconds: u64) -> Self {
        Self {
            cpu_usage: Vec::new(),
            memory_usage: Vec::new(),
            gpu_memory_usage: Vec::new(),
            interval_seconds: _intervalseconds,
        }
    }

    /// Start monitoring resources
    pub fn start_monitoring(&mut self) {
        // Implementation would use system monitoring libraries
        // This is a placeholder for the actual monitoring logic
    }

    /// Stop monitoring and return resource usage summary
    pub fn stop_monitoring(&self) -> ResourceUsage {
        let peak_cpu = self.cpu_usage.iter().fold(0.0f64, |a, &b| a.max(b));
        let avg_cpu = if self.cpu_usage.is_empty() {
            0.0
        } else {
            self.cpu_usage.iter().sum::<f64>() / self.cpu_usage.len() as f64
        };

        let peak_memory = self.memory_usage.iter().fold(0usize, |a, &b| a.max(b));
        let avg_memory = if self.memory_usage.is_empty() {
            0
        } else {
            self.memory_usage.iter().sum::<usize>() / self.memory_usage.len()
        };

        ResourceUsage {
            peak_cpu_usage: peak_cpu,
            avg_cpu_usage: avg_cpu,
            peak_memory_mb: peak_memory,
            avg_memory_mb: avg_memory,
            peak_gpu_memory_mb: None, // Would be calculated from gpu_memory_usage
            total_time_seconds: 0.0,  // Would be calculated from monitoring duration
            energy_consumption_joules: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_creation() {
        let experiment = Experiment::new("Test Experiment")
            .hypothesis("Test hypothesis")
            .description("Test description")
            .metrics(vec!["accuracy".to_string(), "loss".to_string()]);

        assert_eq!(experiment.name, "Test Experiment");
        assert_eq!(experiment.hypothesis, "Test hypothesis");
        assert_eq!(experiment.description, "Test description");
        assert_eq!(experiment.metrics.len(), 2);
        assert_eq!(experiment.status, ExperimentStatus::Planning);
    }

    #[test]
    fn test_experiment_lifecycle() {
        let mut experiment = Experiment::new("Lifecycle Test");

        // Start experiment
        experiment.status = ExperimentStatus::Ready;
        assert!(experiment.start().is_ok());
        assert_eq!(experiment.status, ExperimentStatus::Running);
        assert!(experiment.timeline.started_at.is_some());

        // Complete experiment
        assert!(experiment.complete().is_ok());
        assert_eq!(experiment.status, ExperimentStatus::Completed);
        assert!(experiment.timeline.completed_at.is_some());
        assert!(experiment.timeline.actual_duration.is_some());
    }

    #[test]
    fn test_experiment_notes() {
        let mut experiment = Experiment::new("Notes Test");

        experiment.add_note("Researcher", "Initial observation", NoteType::Observation);
        experiment.add_note("Researcher", "Found an issue", NoteType::Issue);

        assert_eq!(experiment.notes.len(), 2);
        assert_eq!(experiment.notes[0].note_type, NoteType::Observation);
        assert_eq!(experiment.notes[1].note_type, NoteType::Issue);
    }
}
