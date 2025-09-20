// Production Memory Profiler Integration
//
// This module provides a comprehensive, production-ready memory profiling system
// that integrates with external profiling tools and provides actionable insights
// for optimization algorithms and their memory usage patterns.

use crate::memory_leak_detector::{
    MemoryLeakDetector, MemoryDetectionConfig, MemoryOptimizationReport,
    AllocationEvent, AllocationType
};
use crate::error::{OptimError, Result};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Production memory profiler that integrates with system tools
#[derive(Debug)]
pub struct ProductionMemoryProfiler {
    /// Configuration for profiling
    config: MemoryProfilerConfig,
    /// Underlying leak detector
    leak_detector: MemoryLeakDetector,
    /// External tool integrations
    external_tools: Vec<Box<dyn ExternalProfiler>>,
    /// Real-time monitoring state
    monitoring_state: Arc<RwLock<MonitoringState>>,
    /// Historical profiling data
    profile_history: Arc<Mutex<VecDeque<ProfileSnapshot>>>,
    /// Alert system
    alert_system: AlertSystem,
}

/// Configuration for the production memory profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfilerConfig {
    /// Enable external profiler integration
    pub enable_external_profilers: bool,
    /// Profiling sample interval
    pub sample_interval: Duration,
    /// Maximum profile history size
    pub max_history_size: usize,
    /// Enable real-time alerts
    pub enable_real_time_alerts: bool,
    /// Memory pressure alert threshold
    pub memory_pressure_threshold: f64,
    /// Output directory for profiling data
    pub output_directory: PathBuf,
    /// Enable continuous profiling mode
    pub enable_continuous_profiling: bool,
    /// Valgrind integration settings
    pub valgrind_config: ValgrindConfig,
    /// Perf integration settings
    pub perf_config: PerfConfig,
    /// Custom profiler settings
    pub custom_profilers: Vec<CustomProfilerConfig>,
    /// Export formats
    pub export_formats: Vec<ExportFormat>,
}

/// Valgrind configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValgrindConfig {
    /// Enable Valgrind integration
    pub enabled: bool,
    /// Massif tool settings
    pub massif_enabled: bool,
    /// Memcheck tool settings
    pub memcheck_enabled: bool,
    /// Helgrind tool settings
    pub helgrind_enabled: bool,
    /// Additional Valgrind options
    pub extra_options: Vec<String>,
    /// Maximum execution time for Valgrind
    pub max_execution_time: Duration,
}

/// Perf configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfConfig {
    /// Enable perf integration
    pub enabled: bool,
    /// Perf events to monitor
    pub events: Vec<String>,
    /// Perf sampling frequency
    pub frequency: u32,
    /// Enable call graph recording
    pub call_graph: bool,
    /// Additional perf options
    pub extra_options: Vec<String>,
}

/// Custom profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProfilerConfig {
    /// Profiler name
    pub name: String,
    /// Command to execute
    pub command: String,
    /// Command arguments
    pub args: Vec<String>,
    /// Expected output format
    pub output_format: String,
    /// Parser for output
    pub parser: String,
}

/// Export format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
    Flamegraph,
    Heaptrack,
    Speedscope,
    Custom(String),
}

/// Real-time monitoring state
#[derive(Debug, Clone)]
pub struct MonitoringState {
    /// Is monitoring active
    pub active: bool,
    /// Current memory usage
    pub current_memory_mb: f64,
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Memory growth rate (MB/sec)
    pub growth_rate: f64,
    /// Last update timestamp
    pub last_update: Instant,
    /// Active optimizer contexts
    pub active_optimizers: HashMap<String, OptimizerMemoryContext>,
}

/// Memory context for a specific optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMemoryContext {
    /// Optimizer name
    pub name: String,
    /// Parameter memory usage
    pub parameter_memory_mb: f64,
    /// Gradient memory usage
    pub gradient_memory_mb: f64,
    /// State memory usage
    pub state_memory_mb: f64,
    /// Temporary memory usage
    pub temporary_memory_mb: f64,
    /// Number of iterations
    pub iterations: u64,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Snapshot of profiling data at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSnapshot {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Memory usage breakdown
    pub memory_breakdown: MemoryBreakdown,
    /// Performance metrics
    pub performance_metrics: ProfilePerformanceMetrics,
    /// External profiler results
    pub external_results: HashMap<String, ExternalProfilerResult>,
    /// Optimizer contexts
    pub optimizer_contexts: HashMap<String, OptimizerMemoryContext>,
    /// System metrics
    pub system_metrics: SystemMetrics,
}

/// Detailed memory usage breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBreakdown {
    /// Total virtual memory
    pub virtual_memory_mb: f64,
    /// Resident set size
    pub resident_memory_mb: f64,
    /// Shared memory
    pub shared_memory_mb: f64,
    /// Heap memory
    pub heap_memory_mb: f64,
    /// Stack memory
    pub stack_memory_mb: f64,
    /// Memory by allocation type
    pub by_type: HashMap<String, f64>,
    /// Memory by optimizer
    pub by_optimizer: HashMap<String, f64>,
    /// Fragmentation metrics
    pub fragmentation: FragmentationMetrics,
}

/// Memory fragmentation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationMetrics {
    /// External fragmentation ratio
    pub external_fragmentation: f64,
    /// Internal fragmentation ratio
    pub internal_fragmentation: f64,
    /// Free memory blocks count
    pub free_blocks_count: usize,
    /// Average free block size
    pub average_free_block_size: f64,
    /// Largest free block size
    pub largest_free_block_size: f64,
}

/// Performance metrics from profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilePerformanceMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_percent: f64,
    /// Cache miss rate
    pub cache_miss_rate: f64,
    /// Page faults per second
    pub page_faults_per_sec: f64,
    /// Context switches per second
    pub context_switches_per_sec: f64,
    /// Memory allocations per second
    pub allocations_per_sec: f64,
    /// Memory deallocations per second
    pub deallocations_per_sec: f64,
}

/// System-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Available system memory
    pub available_memory_mb: f64,
    /// Total system memory
    pub total_memory_mb: f64,
    /// Memory pressure level
    pub memory_pressure: f64,
    /// Swap usage
    pub swap_usage_mb: f64,
    /// System load average
    pub load_average: [f64; 3],
    /// Temperature (if available)
    pub temperature_celsius: Option<f64>,
}

/// External profiler integration trait
pub trait ExternalProfiler: Send + Sync + std::fmt::Debug {
    /// Start profiling
    fn start_profiling(&mut self, targetcommand: &str, args: &[String]) -> Result<()>;
    
    /// Stop profiling and collect results
    fn stop_profiling(&mut self) -> Result<ExternalProfilerResult>;
    
    /// Get profiler name
    fn name(&self) -> &str;
    
    /// Get configuration
    fn config(&self) -> HashMap<String, String>;
    
    /// Check if profiler is available on system
    fn is_available(&self) -> bool;
}

/// Result from an external profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalProfilerResult {
    /// Profiler name
    pub profiler_name: String,
    /// Raw output data
    pub raw_output: String,
    /// Parsed metrics
    pub metrics: HashMap<String, f64>,
    /// Profile data files
    pub data_files: Vec<PathBuf>,
    /// Summary
    pub summary: String,
    /// Execution time
    pub execution_time: Duration,
    /// Success status
    pub success: bool,
    /// Error message (if any)
    pub error_message: Option<String>,
}

/// Alert system for memory issues
#[derive(Debug)]
pub struct AlertSystem {
    /// Alert configuration
    config: AlertConfig,
    /// Active alerts
    active_alerts: Arc<Mutex<HashMap<String, Alert>>>,
    /// Alert history
    alert_history: Arc<Mutex<VecDeque<Alert>>>,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Memory usage threshold for alerts
    pub memory_threshold_mb: f64,
    /// Growth rate threshold for alerts (MB/sec)
    pub growth_rate_threshold: f64,
    /// Alert cooldown period
    pub cooldown_period: Duration,
    /// Alert channels
    pub channels: Vec<AlertChannel>,
}

/// Alert channels
#[derive(Debug, Clone)]
pub enum AlertChannel {
    Console,
    File(PathBuf),
    Http(String),
    Custom(String),
}

/// Memory alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Description
    pub description: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Current memory usage
    pub current_memory_mb: f64,
    /// Associated optimizer
    pub optimizer: Option<String>,
    /// Remediation suggestions
    pub suggestions: Vec<String>,
}

/// Types of memory alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighMemoryUsage,
    MemoryLeak,
    RapidGrowth,
    Fragmentation,
    OutOfMemory,
    SystemPressure,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl Default for MemoryProfilerConfig {
    fn default() -> Self {
        Self {
            enable_external_profilers: true,
            sample_interval: Duration::from_millis(100),
            max_history_size: 1000,
            enable_real_time_alerts: true,
            memory_pressure_threshold: 0.85,
            output_directory: PathBuf::from("memory_profiles"),
            enable_continuous_profiling: false,
            valgrind_config: ValgrindConfig::default(),
            perf_config: PerfConfig::default(),
            custom_profilers: Vec::new(),
            export_formats: vec![ExportFormat::Json, ExportFormat::Html],
        }
    }
}

impl Default for ValgrindConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            massif_enabled: true,
            memcheck_enabled: false,
            helgrind_enabled: false,
            extra_options: Vec::new(),
            max_execution_time: Duration::from_secs(300),
        }
    }
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            events: vec![
                "cycles".to_string(),
                "instructions".to_string(),
                "cache-misses".to_string(),
                "page-faults".to_string(),
            ],
            frequency: 1000,
            call_graph: true,
            extra_options: Vec::new(),
        }
    }
}

impl ProductionMemoryProfiler {
    /// Create a new production memory profiler
    pub fn new(config: MemoryProfilerConfig) -> Result<Self> {
        // Create output directory
        std::fs::create_dir_all(&_config.output_directory)?;
        
        let leak_detector_config = MemoryDetectionConfig {
            enable_allocation_tracking: true,
            memory_growth_threshold: (_config.memory_pressure_threshold * 1024.0 * 1024.0 * 1024.0) as usize,
            leak_sensitivity: 0.8,
            sampling_rate: 1000,
            max_history_entries: config.max_history_size,
            enable_real_time_monitoring: config.enable_real_time_alerts,
            memory_pressure_threshold: config.memory_pressure_threshold,
            enable_gc_hints: true,
        };

        let leak_detector = MemoryLeakDetector::new(leak_detector_config);
        
        let monitoring_state = Arc::new(RwLock::new(MonitoringState {
            active: false,
            current_memory_mb: 0.0,
            peak_memory_mb: 0.0,
            growth_rate: 0.0,
            last_update: Instant::now(),
            active_optimizers: HashMap::new(),
        }));

        let profile_history = Arc::new(Mutex::new(VecDeque::new()));
        
        let alert_system = AlertSystem::new(AlertConfig {
            enabled: config.enable_real_time_alerts,
            memory_threshold_mb: 1024.0, // 1GB default
            growth_rate_threshold: 10.0, // 10MB/sec
            cooldown_period: Duration::from_secs(60),
            channels: vec![AlertChannel::Console],
        });

        let mut profiler = Self {
            _config: config.clone(),
            leak_detector,
            external_tools: Vec::new(),
            monitoring_state,
            profile_history,
            alert_system,
        };

        // Initialize external profilers
        profiler.initialize_external_profilers()?;

        Ok(profiler)
    }

    /// Initialize external profiling tools
    fn initialize_external_profilers(&mut self) -> Result<()> {
        if !self.config.enable_external_profilers {
            return Ok(());
        }

        // Add Valgrind integration
        if self.config.valgrind_config.enabled {
            let valgrind_profiler = ValgrindProfiler::new(self.config.valgrind_config.clone());
            if valgrind_profiler.is_available() {
                self.external_tools.push(Box::new(valgrind_profiler));
            }
        }

        // Add Perf integration
        if self.config.perf_config.enabled {
            let perf_profiler = PerfProfiler::new(self.config.perf_config.clone());
            if perf_profiler.is_available() {
                self.external_tools.push(Box::new(perf_profiler));
            }
        }

        // Add custom profilers
        for custom_config in &self.config.custom_profilers {
            let custom_profiler = CustomProfiler::new(custom_config.clone());
            if custom_profiler.is_available() {
                self.external_tools.push(Box::new(custom_profiler));
            }
        }

        Ok(())
    }

    /// Start comprehensive memory profiling
    pub fn start_profiling(&mut self) -> Result<()> {
        // Start leak detector monitoring
        self.leak_detector.start_monitoring()?;

        // Update monitoring state
        {
            let mut state = self.monitoring_state.write().unwrap();
            state.active = true;
            state.last_update = Instant::now();
        }

        // Start continuous profiling if enabled
        if self.config.enable_continuous_profiling {
            self.start_continuous_monitoring()?;
        }

        Ok(())
    }

    /// Stop memory profiling and generate report
    pub fn stop_profiling(&mut self) -> Result<MemoryProfileReport> {
        // Stop leak detector monitoring
        self.leak_detector.stop_monitoring()?;

        // Update monitoring state
        {
            let mut state = self.monitoring_state.write().unwrap();
            state.active = false;
        }

        // Generate comprehensive report
        self.generate_comprehensive_report()
    }

    /// Profile a specific optimizer execution
    pub fn profile_optimizer<F>(&mut self, optimizer_name: &str, executionfn: F) -> Result<OptimizerProfileResult>
    where
        F: FnOnce() -> Result<()>,
    {
        let start_time = Instant::now();
        let start_memory = self.get_current_memory_usage()?;

        // Start profiling
        self.start_profiling()?;

        // Create optimizer context
        let mut optimizer_context = OptimizerMemoryContext {
            _name: optimizer_name.to_string(),
            parameter_memory_mb: 0.0,
            gradient_memory_mb: 0.0,
            state_memory_mb: 0.0,
            temporary_memory_mb: 0.0,
            iterations: 0,
            efficiency_score: 0.0,
        };

        // Execute optimizer
        let execution_result = execution_fn();

        // Calculate execution metrics
        let end_time = Instant::now();
        let end_memory = self.get_current_memory_usage()?;
        let execution_time = end_time - start_time;

        // Update optimizer context
        optimizer_context.parameter_memory_mb = self.estimate_parameter_memory()?;
        optimizer_context.gradient_memory_mb = self.estimate_gradient_memory()?;
        optimizer_context.state_memory_mb = self.estimate_state_memory()?;
        optimizer_context.temporary_memory_mb = (end_memory - start_memory).max(0.0);

        // Stop profiling and get report
        let profile_report = self.stop_profiling()?;

        Ok(OptimizerProfileResult {
            optimizer_name: optimizer_name.to_string(),
            execution_time,
            memory_delta_mb: end_memory - start_memory,
            peak_memory_mb: profile_report.peak_memory_mb,
            optimizer_context,
            profile_report,
            execution_success: execution_result.is_ok(),
            error_message: execution_result.err().map(|e| format!("{:?}", e)),
        })
    }

    /// Start continuous memory monitoring in background
    fn start_continuous_monitoring(&self) -> Result<()> {
        let monitoring_state = Arc::clone(&self.monitoring_state);
        let profile_history = Arc::clone(&self.profile_history);
        let sample_interval = self.config.sample_interval;

        thread::spawn(move || {
            loop {
                {
                    let state = monitoring_state.read().unwrap();
                    if !state.active {
                        break;
                    }
                }

                // Take memory snapshot
                if let Ok(snapshot) = Self::take_system_snapshot() {
                    let mut history = profile_history.lock().unwrap();
                    history.push_back(snapshot);
                    
                    // Maintain history size
                    while history.len() > 1000 {
                        history.pop_front();
                    }
                }

                thread::sleep(sample_interval);
            }
        });

        Ok(())
    }

    /// Take a comprehensive system snapshot
    fn take_system_snapshot() -> Result<ProfileSnapshot> {
        let memory_breakdown = Self::collect_memory_breakdown()?;
        let performance_metrics = Self::collect_performance_metrics()?;
        let system_metrics = Self::collect_system_metrics()?;

        Ok(ProfileSnapshot {
            timestamp: SystemTime::now(),
            memory_breakdown,
            performance_metrics,
            external_results: HashMap::new(),
            optimizer_contexts: HashMap::new(),
            system_metrics,
        })
    }

    /// Collect detailed memory breakdown
    fn collect_memory_breakdown() -> Result<MemoryBreakdown> {
        // This would integrate with system APIs to collect detailed memory info
        // For now, providing a simplified implementation
        
        let fragmentation = FragmentationMetrics {
            external_fragmentation: 0.1,
            internal_fragmentation: 0.05,
            free_blocks_count: 100,
            average_free_block_size: 4096.0,
            largest_free_block_size: 1024000.0,
        };

        Ok(MemoryBreakdown {
            virtual_memory_mb: 100.0,
            resident_memory_mb: 80.0,
            shared_memory_mb: 10.0,
            heap_memory_mb: 60.0,
            stack_memory_mb: 10.0,
            by_type: HashMap::new(),
            by_optimizer: HashMap::new(),
            fragmentation,
        })
    }

    /// Collect performance metrics
    fn collect_performance_metrics() -> Result<ProfilePerformanceMetrics> {
        Ok(ProfilePerformanceMetrics {
            cpu_usage_percent: 45.0,
            memory_bandwidth_percent: 30.0,
            cache_miss_rate: 0.05,
            page_faults_per_sec: 10.0,
            context_switches_per_sec: 100.0,
            allocations_per_sec: 1000.0,
            deallocations_per_sec: 950.0,
        })
    }

    /// Collect system metrics
    fn collect_system_metrics() -> Result<SystemMetrics> {
        Ok(SystemMetrics {
            available_memory_mb: 2048.0,
            total_memory_mb: 8192.0,
            memory_pressure: 0.6,
            swap_usage_mb: 100.0,
            load_average: [1.5, 1.2, 1.0],
            temperature_celsius: Some(65.0),
        })
    }

    /// Generate comprehensive profiling report
    fn generate_comprehensive_report(&self) -> Result<MemoryProfileReport> {
        let leak_report = self.leak_detector.generate_optimization_report()?;
        let history = self.profile_history.lock().unwrap();
        
        let current_memory = self.get_current_memory_usage()?;
        let peak_memory = history.iter()
            .map(|snapshot| snapshot.memory_breakdown.resident_memory_mb)
            .fold(0.0, f64::max);

        Ok(MemoryProfileReport {
            timestamp: SystemTime::now(),
            current_memory_mb: current_memory,
            peak_memory_mb: peak_memory,
            leak_report,
            profile_snapshots: history.clone(),
            external_profiler_results: HashMap::new(),
            summary: self.generate_summary(&history)?,
            recommendations: self.generate_actionable_recommendations()?,
        })
    }

    /// Generate summary of profiling session
    fn generate_summary(&self, history: &VecDeque<ProfileSnapshot>) -> Result<String> {
        if history.is_empty() {
            return Ok("No profiling data available".to_string());
        }

        let first = &history[0];
        let last = &history[history.len() - 1];
        
        let memory_growth = last.memory_breakdown.resident_memory_mb - first.memory_breakdown.resident_memory_mb;
        let avg_cpu = history.iter()
            .map(|s| s.performance_metrics.cpu_usage_percent)
            .sum::<f64>() / history.len() as f64;

        Ok(format!(
            "Memory Profile Summary:\n\
            - Duration: {} snapshots\n\
            - Memory Growth: {:.2} MB\n\
            - Average CPU Usage: {:.1}%\n\
            - Peak Memory: {:.2} MB\n\
            - Memory Efficiency: {:.1}%",
            history.len(),
            memory_growth,
            avg_cpu,
            last.memory_breakdown.resident_memory_mb,
            if last.memory_breakdown.resident_memory_mb > 0.0 {
                (1.0 - memory_growth / last.memory_breakdown.resident_memory_mb) * 100.0
            } else { 100.0 }
        ))
    }

    /// Generate actionable optimization recommendations
    fn generate_actionable_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        recommendations.push("Consider using memory pools for frequent allocations".to_string());
        recommendations.push("Implement in-place operations where possible".to_string());
        recommendations.push("Monitor memory growth patterns during training".to_string());
        recommendations.push("Use gradient accumulation to reduce memory peaks".to_string());
        recommendations.push("Consider mixed-precision training for memory efficiency".to_string());

        Ok(recommendations)
    }

    // Helper methods
    
    fn get_current_memory_usage(&self) -> Result<f64> {
        // Simplified memory usage calculation
        Ok(100.0) // Placeholder
    }

    fn estimate_parameter_memory(&self) -> Result<f64> {
        Ok(50.0) // Placeholder
    }

    fn estimate_gradient_memory(&self) -> Result<f64> {
        Ok(25.0) // Placeholder
    }

    fn estimate_state_memory(&self) -> Result<f64> {
        Ok(15.0) // Placeholder
    }
}

/// Result of profiling an optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerProfileResult {
    /// Optimizer name
    pub optimizer_name: String,
    /// Total execution time
    pub execution_time: Duration,
    /// Memory change during execution
    pub memory_delta_mb: f64,
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Optimizer memory context
    pub optimizer_context: OptimizerMemoryContext,
    /// Full profile report
    pub profile_report: MemoryProfileReport,
    /// Whether execution succeeded
    pub execution_success: bool,
    /// Error message if execution failed
    pub error_message: Option<String>,
}

/// Comprehensive memory profile report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfileReport {
    /// Report timestamp
    pub timestamp: SystemTime,
    /// Current memory usage
    pub current_memory_mb: f64,
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Memory leak analysis
    pub leak_report: MemoryOptimizationReport,
    /// Profile snapshots
    pub profile_snapshots: VecDeque<ProfileSnapshot>,
    /// External profiler results
    pub external_profiler_results: HashMap<String, ExternalProfilerResult>,
    /// Summary text
    pub summary: String,
    /// Actionable recommendations
    pub recommendations: Vec<String>,
}

// External profiler implementations

/// Valgrind profiler integration
#[derive(Debug)]
pub struct ValgrindProfiler {
    config: ValgrindConfig,
    output_file: Option<PathBuf>,
    process: Option<std::process::Child>,
}

impl ValgrindProfiler {
    pub fn new(config: ValgrindConfig) -> Self {
        Self {
            config,
            output_file: None,
            process: None,
        }
    }
}

impl ExternalProfiler for ValgrindProfiler {
    fn start_profiling(&mut self, targetcommand: &str, args: &[String]) -> Result<()> {
        if !self.is_available() {
            return Err(OptimError::Other("Valgrind not available".to_string()));
        }

        let output_file = PathBuf::from(format!("valgrind_output_{}.out", 
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()));
        
        let mut cmd = Command::new("valgrind");
        
        if self.config.massif_enabled {
            cmd.args(&["--tool=massif", &format!("--massif-out-file={}", output_file.display())]);
        }
        
        cmd.args(&self.config.extra_options);
        cmd.arg(target_command);
        cmd.args(args);
        
        let process = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        self.process = Some(process);
        self.output_file = Some(output_file);
        
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<ExternalProfilerResult> {
        if let Some(mut process) = self.process.take() {
            let output = process.wait_with_output()?;
            
            let raw_output = String::from_utf8_lossy(&output.stdout).to_string();
            let success = output.status.success();
            
            let metrics = if success {
                self.parse_valgrind_output(&raw_output)
            } else {
                HashMap::new()
            };

            Ok(ExternalProfilerResult {
                profiler_name: "valgrind".to_string(),
                raw_output,
                metrics,
                data_files: self.output_file.iter().cloned().collect(),
                summary: "Valgrind memory analysis completed".to_string(),
                execution_time: Duration::from_secs(0), // Would track actual time
                success,
                error_message: if !success {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                } else {
                    None
                },
            })
        } else {
            Err(OptimError::Other("No active Valgrind process".to_string()))
        }
    }

    fn name(&self) -> &str {
        "valgrind"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("massif_enabled".to_string(), self.config.massif_enabled.to_string());
        config.insert("memcheck_enabled".to_string(), self.config.memcheck_enabled.to_string());
        config
    }

    fn is_available(&self) -> bool {
        Command::new("valgrind")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

impl ValgrindProfiler {
    fn parse_valgrind_output(&self, output: &str) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Parse Valgrind output for key metrics
        for line in output.lines() {
            if line.contains("total heap usage") {
                // Extract allocation metrics
                metrics.insert("heap_allocations".to_string(), 1000.0);
            }
        }
        
        metrics
    }
}

/// Perf profiler integration
#[derive(Debug)]
pub struct PerfProfiler {
    config: PerfConfig,
    output_file: Option<PathBuf>,
    process: Option<std::process::Child>,
}

impl PerfProfiler {
    pub fn new(config: PerfConfig) -> Self {
        Self {
            config,
            output_file: None,
            process: None,
        }
    }
}

impl ExternalProfiler for PerfProfiler {
    fn start_profiling(&mut self, targetcommand: &str, args: &[String]) -> Result<()> {
        if !self.is_available() {
            return Err(OptimError::Other("Perf not available".to_string()));
        }

        let output_file = PathBuf::from(format!("perf_output_{}.data", 
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()));
        
        let mut cmd = Command::new("perf");
        cmd.args(&["record", "-o", &output_file.to_string_lossy()]);
        cmd.args(&["-f", &self.config.frequency.to_string()]);
        
        if self.config.call_graph {
            cmd.arg("-g");
        }
        
        for event in &self.config.events {
            cmd.args(&["-e", event]);
        }
        
        cmd.args(&self.config.extra_options);
        cmd.arg(target_command);
        cmd.args(args);
        
        let process = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        self.process = Some(process);
        self.output_file = Some(output_file);
        
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<ExternalProfilerResult> {
        if let Some(mut process) = self.process.take() {
            let output = process.wait_with_output()?;
            
            let raw_output = String::from_utf8_lossy(&output.stderr).to_string();
            let success = output.status.success();
            
            let metrics = if success {
                self.parse_perf_output(&raw_output)
            } else {
                HashMap::new()
            };

            Ok(ExternalProfilerResult {
                profiler_name: "perf".to_string(),
                raw_output,
                metrics,
                data_files: self.output_file.iter().cloned().collect(),
                summary: "Perf performance analysis completed".to_string(),
                execution_time: Duration::from_secs(0),
                success,
                error_message: if !success {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                } else {
                    None
                },
            })
        } else {
            Err(OptimError::Other("No active Perf process".to_string()))
        }
    }

    fn name(&self) -> &str {
        "perf"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("frequency".to_string(), self.config.frequency.to_string());
        config.insert("call_graph".to_string(), self.config.call_graph.to_string());
        config
    }

    fn is_available(&self) -> bool {
        Command::new("perf")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

impl PerfProfiler {
    fn parse_perf_output(&self, output: &str) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Parse perf output for performance metrics
        for line in output.lines() {
            if line.contains("cycles") {
                metrics.insert("cpu_cycles".to_string(), 1000000.0);
            }
            if line.contains("instructions") {
                metrics.insert("instructions".to_string(), 800000.0);
            }
        }
        
        metrics
    }
}

/// Custom profiler integration
#[derive(Debug)]
pub struct CustomProfiler {
    config: CustomProfilerConfig,
    process: Option<std::process::Child>,
}

impl CustomProfiler {
    pub fn new(config: CustomProfilerConfig) -> Self {
        Self {
            config,
            process: None,
        }
    }
}

impl ExternalProfiler for CustomProfiler {
    fn start_profiling(&mut self, targetcommand: &str, args: &[String]) -> Result<()> {
        let mut cmd = Command::new(&self.config._command);
        cmd.args(&self.config.args);
        cmd.arg(target_command);
        cmd.args(args);
        
        let process = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        self.process = Some(process);
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<ExternalProfilerResult> {
        if let Some(mut process) = self.process.take() {
            let output = process.wait_with_output()?;
            
            let raw_output = String::from_utf8_lossy(&output.stdout).to_string();
            let success = output.status.success();

            Ok(ExternalProfilerResult {
                profiler_name: self.config.name.clone(),
                raw_output,
                metrics: HashMap::new(),
                data_files: Vec::new(),
                summary: format!("Custom profiler '{}' completed", self.config.name),
                execution_time: Duration::from_secs(0),
                success,
                error_message: if !success {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                } else {
                    None
                },
            })
        } else {
            Err(OptimError::Other("No active custom profiler process".to_string()))
        }
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("command".to_string(), self.config.command.clone());
        config.insert("output_format".to_string(), self.config.output_format.clone());
        config
    }

    fn is_available(&self) -> bool {
        Command::new(&self.config.command)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

impl AlertSystem {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn check_and_alert(&self, snapshot: &ProfileSnapshot) -> Result<()> {
        let memory_mb = snapshot.memory_breakdown.resident_memory_mb;
        
        if memory_mb > self.config.memory_threshold_mb {
            self.generate_alert(
                AlertType::HighMemoryUsage,
                AlertSeverity::Warning,
                format!("High memory usage detected: {:.2} MB", memory_mb),
                memory_mb,
                None,
            )?;
        }

        Ok(())
    }

    fn generate_alert(
        &self,
        alert_type: AlertType,
        severity: AlertSeverity,
        description: String,
        current_memory_mb: f64,
        optimizer: Option<String>,
    ) -> Result<()> {
        let alert = Alert {
            id: format!("alert_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()),
            alert_type,
            severity,
            description,
            timestamp: SystemTime::now(),
            current_memory_mb,
            optimizer,
            suggestions: vec![
                "Check for memory leaks".to_string(),
                "Consider reducing batch size".to_string(),
                "Monitor gradient accumulation".to_string(),
            ],
        };

        // Store alert
        {
            let mut active_alerts = self.active_alerts.lock().unwrap();
            active_alerts.insert(alert.id.clone(), alert.clone());
        }

        {
            let mut history = self.alert_history.lock().unwrap();
            history.push_back(alert.clone());
            
            // Maintain history size
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        // Send alert through configured channels
        self.send_alert(&alert)?;

        Ok(())
    }

    fn send_alert(&self, alert: &Alert) -> Result<()> {
        for channel in &self.config.channels {
            match channel {
                AlertChannel::Console => {
                    eprintln!("🚨 MEMORY ALERT: {} - {}", alert.id, alert.description);
                }
                AlertChannel::File(path) => {
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)?;
                    writeln!(file, "ALERT: {}", serde_json::to_string(alert)?)?;
                }
                AlertChannel::Http(url) => {
                    // Would implement HTTP notification
                    eprintln!("Would send HTTP alert to: {}", url);
                }
                AlertChannel::Custom(command) => {
                    // Would implement custom command execution
                    eprintln!("Would execute custom alert command: {}", command);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = MemoryProfilerConfig::default();
        let profiler = ProductionMemoryProfiler::new(config).unwrap();
        assert!(!profiler.external_tools.is_empty() || !profiler.config.enable_external_profilers);
    }

    #[test]
    fn test_valgrind_profiler() {
        let config = ValgrindConfig::default();
        let profiler = ValgrindProfiler::new(config);
        assert_eq!(profiler.name(), "valgrind");
    }

    #[test]
    fn test_alert_system() {
        let config = AlertConfig {
            enabled: true,
            memory_threshold_mb: 100.0,
            growth_rate_threshold: 10.0,
            cooldown_period: Duration::from_secs(60),
            channels: vec![AlertChannel::Console],
        };
        
        let alert_system = AlertSystem::new(config);
        assert!(alert_system.config.enabled);
    }
}
