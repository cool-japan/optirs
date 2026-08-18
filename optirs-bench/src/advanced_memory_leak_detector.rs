// Advanced Memory Leak Detection System
//
// This module provides comprehensive memory leak detection capabilities specifically
// designed for optimization algorithms, with support for real-time monitoring,
// statistical analysis, and automated reporting.

use crate::error::{OptimError, Result};
use crate::memory_leak_detector as stats;
use crate::system_sampler::SystemSampler;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

/// Advanced memory leak detector with real-time monitoring
#[allow(dead_code)]
pub struct AdvancedMemoryLeakDetector {
    /// Configuration for memory leak detection
    config: MemoryLeakConfig,
    /// Memory usage history
    memory_history: Arc<RwLock<VecDeque<MemorySnapshot>>>,
    /// Active monitoring sessions
    active_sessions: Arc<Mutex<HashMap<String, MonitoringSession>>>,
    /// Leak detection engine
    leak_analyzer: LeakAnalysisEngine,
    /// Alert system for memory issues
    alert_system: MemoryAlertSystem,
    /// Statistics tracker
    statistics: MemoryStatistics,
    /// Real system/process telemetry (RSS, virtual memory, system memory).
    sampler: Arc<SystemSampler>,
}

/// Configuration for memory leak detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeakConfig {
    /// Enable leak detection
    pub enabled: bool,
    /// Sampling interval for memory monitoring
    pub sampling_interval: Duration,
    /// Memory growth threshold for leak detection (bytes per second)
    pub growth_threshold: f64,
    /// Statistical confidence level for leak detection
    pub confidence_level: f64,
    /// Minimum monitoring duration before analysis
    pub min_monitoring_duration: Duration,
    /// Maximum memory history to retain
    pub max_history_size: usize,
    /// Memory leak detection algorithms to use
    pub detection_algorithms: Vec<LeakDetectionAlgorithm>,
    /// Alert thresholds
    pub alert_thresholds: MemoryAlertThresholds,
    /// Optimization-specific settings
    pub optimizer_settings: OptimizerMemorySettings,
}

/// Memory leak detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakDetectionAlgorithm {
    /// Linear trend analysis
    LinearTrend,
    /// Statistical process control
    StatisticalProcessControl,
    /// Machine learning anomaly detection
    AnomalyDetection,
    /// Pattern recognition
    PatternRecognition,
    /// Memory pool analysis
    MemoryPoolAnalysis,
}

/// Alert thresholds for memory issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAlertThresholds {
    /// Memory growth rate threshold (bytes/second)
    pub growth_rate_threshold: f64,
    /// Absolute memory threshold (bytes)
    pub absolute_memory_threshold: u64,
    /// Memory fragmentation threshold (0.0-1.0)
    pub fragmentation_threshold: f64,
    /// Garbage collection frequency threshold
    pub gc_frequency_threshold: u32,
}

/// Optimizer-specific memory settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMemorySettings {
    /// Expected memory growth patterns for different optimizers
    pub optimizer_profiles: HashMap<String, OptimizerMemoryProfile>,
    /// Memory pool tracking settings
    pub track_memory_pools: bool,
    /// Gradient accumulation monitoring
    pub monitor_gradient_accumulation: bool,
    /// Parameter buffer tracking
    pub track_parameter_buffers: bool,
}

/// Memory profile for a specific optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMemoryProfile {
    /// Expected base memory usage (bytes)
    pub base_memory: u64,
    /// Expected memory growth per iteration (bytes)
    pub memory_per_iteration: f64,
    /// Maximum acceptable memory growth rate
    pub max_growth_rate: f64,
    /// Memory release patterns
    pub release_patterns: Vec<MemoryReleasePattern>,
}

/// Memory release pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReleasePattern {
    /// Trigger condition
    pub trigger: ReleaseTrigger,
    /// Expected memory release amount (bytes)
    pub release_amount: u64,
    /// Release frequency
    pub frequency: Duration,
}

/// Memory release trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReleaseTrigger {
    /// After specific number of iterations
    IterationCount(u32),
    /// When memory reaches threshold
    MemoryThreshold(u64),
    /// Time-based release
    TimeInterval(Duration),
    /// Optimizer step completion
    OptimizerStep,
}

/// Memory snapshot at a specific point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Timestamp of snapshot
    pub timestamp: SystemTime,
    /// Total memory usage (bytes)
    pub total_memory: u64,
    /// Heap memory usage (bytes)
    pub heap_memory: u64,
    /// Stack memory usage (bytes)
    pub stack_memory: u64,
    /// GPU memory usage (bytes, if available)
    pub gpu_memory: Option<u64>,
    /// Memory fragmentation level (0.0-1.0)
    pub fragmentation: f64,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Active memory pools
    pub memory_pools: HashMap<String, MemoryPoolInfo>,
    /// Optimizer-specific memory usage
    pub optimizer_memory: HashMap<String, OptimizerMemoryUsage>,
    /// System memory information
    pub system_memory: SystemMemoryInfo,
}

/// Memory pool information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolInfo {
    /// Pool size (bytes)
    pub size: u64,
    /// Used memory in pool (bytes)
    pub used: u64,
    /// Free memory in pool (bytes)
    pub free: u64,
    /// Number of allocations from pool
    pub allocations: u64,
    /// Pool fragmentation level
    pub fragmentation: f64,
}

/// Optimizer-specific memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMemoryUsage {
    /// Parameter memory (bytes)
    pub parameters: u64,
    /// Gradient memory (bytes)
    pub gradients: u64,
    /// Momentum/velocity memory (bytes)
    pub momentum: u64,
    /// Second moment estimates (bytes)
    pub second_moments: u64,
    /// Optimizer state memory (bytes)
    pub optimizer_state: u64,
    /// Temporary computation memory (bytes)
    pub temporary: u64,
}

/// System memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMemoryInfo {
    /// Total system memory (bytes)
    pub total: u64,
    /// Available system memory (bytes)
    pub available: u64,
    /// Used system memory (bytes)
    pub used: u64,
    /// System memory pressure level (0.0-1.0)
    pub pressure: f64,
}

/// Active monitoring session
#[derive(Debug)]
pub struct MonitoringSession {
    /// Session ID
    pub sessionid: String,
    /// Start time
    pub start_time: Instant,
    /// Monitoring configuration
    pub config: SessionConfig,
    /// Memory snapshots for this session
    pub snapshots: VecDeque<MemorySnapshot>,
    /// Current analysis results
    pub analysis_results: Option<LeakAnalysisResult>,
    /// Session statistics
    pub statistics: SessionStatistics,
}

/// Configuration for a monitoring session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Optimizer being monitored
    pub optimizer_name: String,
    /// Monitoring duration
    pub duration: Duration,
    /// Sampling frequency
    pub sampling_frequency: Duration,
    /// Analysis triggers
    pub analysis_triggers: Vec<AnalysisTrigger>,
}

/// Analysis trigger conditions
#[derive(Debug, Clone)]
pub enum AnalysisTrigger {
    /// Analyze after duration
    Duration(Duration),
    /// Analyze after number of snapshots
    SnapshotCount(usize),
    /// Analyze on memory threshold
    MemoryThreshold(u64),
    /// Analyze on growth rate
    GrowthRate(f64),
}

/// Session-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total monitoring duration
    pub duration: Duration,
    /// Number of snapshots collected
    pub snapshot_count: usize,
    /// Average memory usage
    pub avg_memory: f64,
    /// Peak memory usage
    pub peak_memory: u64,
    /// Memory growth rate (bytes/second)
    pub growth_rate: f64,
    /// Memory volatility
    pub volatility: f64,
}

/// Leak analysis engine
#[derive(Debug)]
#[allow(dead_code)]
pub struct LeakAnalysisEngine {
    /// Analysis configuration
    config: AnalysisConfig,
    /// Statistical analyzer
    statistical_analyzer: StatisticalAnalyzer,
    /// Pattern detector
    pattern_detector: PatternDetector,
    /// Anomaly detector
    anomaly_detector: AnomalyDetector,
}

/// Configuration for leak analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Minimum effect size to consider significant
    pub min_effect_size: f64,
    /// Analysis window size
    pub analysis_window: usize,
    /// Trend detection sensitivity
    pub trend_sensitivity: f64,
}

/// Result of leak analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakAnalysisResult {
    /// Analysis timestamp
    pub timestamp: SystemTime,
    /// Is a leak detected?
    pub leak_detected: bool,
    /// Confidence level of detection
    pub confidence: f64,
    /// Leak severity (0.0-1.0)
    pub severity: f64,
    /// Growth rate analysis
    pub growth_analysis: GrowthAnalysis,
    /// Pattern analysis results
    pub pattern_analysis: PatternAnalysisResult,
    /// Anomaly detection results
    pub anomaly_analysis: AnomalyAnalysisResult,
    /// Leak characteristics
    pub leak_characteristics: LeakCharacteristics,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Growth rate analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthAnalysis {
    /// Linear growth rate (bytes/second)
    pub linear_rate: f64,
    /// Exponential growth factor
    pub exponential_factor: f64,
    /// Growth trend type
    pub trend_type: GrowthTrendType,
    /// Statistical significance
    pub significance: f64,
    /// R-squared value for trend fit
    pub r_squared: f64,
}

/// Growth trend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthTrendType {
    /// No significant growth
    NoGrowth,
    /// Linear growth
    Linear,
    /// Exponential growth
    Exponential,
    /// Polynomial growth
    Polynomial,
    /// Irregular/chaotic growth
    Irregular,
}

/// Pattern analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysisResult {
    /// Detected patterns
    pub patterns: Vec<MemoryPattern>,
    /// Pattern confidence
    pub confidence: f64,
    /// Periodic behavior detected
    pub periodic_behavior: Option<PeriodicBehavior>,
}

/// Memory usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern strength (0.0-1.0)
    pub strength: f64,
    /// Pattern frequency
    pub frequency: Option<Duration>,
    /// Pattern description
    pub description: String,
}

/// Types of memory patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Saw-tooth pattern (allocate then release)
    SawTooth,
    /// Staircase pattern (gradual increases)
    Staircase,
    /// Periodic spikes
    PeriodicSpikes,
    /// Memory plateau
    Plateau,
    /// Random walk
    RandomWalk,
}

/// Periodic behavior in memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicBehavior {
    /// Period duration
    pub period: Duration,
    /// Amplitude of oscillation
    pub amplitude: f64,
    /// Phase shift
    pub phase: f64,
    /// Periodicity confidence
    pub confidence: f64,
}

/// Anomaly analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAnalysisResult {
    /// Detected anomalies
    pub anomalies: Vec<MemoryAnomaly>,
    /// Overall anomaly score
    pub anomaly_score: f64,
    /// Anomaly detection confidence
    pub confidence: f64,
}

/// Memory usage anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnomaly {
    /// Anomaly timestamp
    pub timestamp: SystemTime,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Anomaly severity (0.0-1.0)
    pub severity: f64,
    /// Expected vs actual values
    pub expected_value: f64,
    /// Actual value
    pub actual_value: f64,
    /// Anomaly description
    pub description: String,
}

/// Types of memory anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Sudden memory spike
    MemorySpike,
    /// Unexpected memory release
    UnexpectedRelease,
    /// Gradual memory increase
    GradualIncrease,
    /// Memory fragmentation spike
    FragmentationSpike,
    /// Allocation/deallocation imbalance
    AllocationImbalance,
}

/// Leak characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakCharacteristics {
    /// Leak type
    pub leak_type: LeakType,
    /// Leak source (if identifiable)
    pub source: Option<String>,
    /// Leak rate (bytes/second)
    pub leak_rate: f64,
    /// Time to exhaust memory (if applicable)
    pub time_to_exhaustion: Option<Duration>,
    /// Affected memory components
    pub affected_components: Vec<String>,
}

/// Types of memory leaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakType {
    /// Classic memory leak (allocated but not freed)
    ClassicLeak,
    /// Memory growth due to unbounded collections
    UnboundedGrowth,
    /// Fragmentation-induced pseudo-leak
    FragmentationLeak,
    /// Cyclic reference leak
    CyclicReferenceLeak,
    /// Resource leak (file handles, connections, etc.)
    ResourceLeak,
}

/// Memory alert system
#[allow(dead_code)]
pub struct MemoryAlertSystem {
    /// Alert configuration
    config: AlertConfig,
    /// Alert history
    alert_history: VecDeque<MemoryAlert>,
    /// Alert handlers
    alert_handlers: Vec<Box<dyn AlertHandler>>,
}

/// Memory alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAlert {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert type
    pub alert_type: MemoryAlertType,
    /// Alert message
    pub message: String,
    /// Memory metrics at alert time
    pub memory_metrics: MemorySnapshot,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Memory alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAlertType {
    /// Memory leak detected
    MemoryLeak,
    /// High memory usage
    HighMemoryUsage,
    /// Memory fragmentation
    MemoryFragmentation,
    /// Unexpected memory pattern
    UnexpectedPattern,
    /// System memory pressure
    SystemPressure,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Minimum severity for alerts
    pub min_severity: AlertSeverity,
    /// Alert throttling settings
    pub throttling: AlertThrottling,
}

/// Alert throttling configuration
#[derive(Debug, Clone)]
pub struct AlertThrottling {
    /// Maximum alerts per time window
    pub max_alerts: usize,
    /// Time window for throttling
    pub time_window: Duration,
    /// Cooldown between similar alerts
    pub cooldown: Duration,
}

/// Alert handler trait
pub trait AlertHandler: Send + Sync {
    fn handle_alert(&self, alert: &MemoryAlert) -> Result<()>;
}

/// Memory statistics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Total monitoring sessions
    pub total_sessions: u64,
    /// Total leaks detected
    pub total_leaks_detected: u64,
    /// Average memory usage across sessions
    pub avg_memory_usage: f64,
    /// Peak memory usage seen
    pub peak_memory_usage: u64,
    /// Most common leak type
    pub most_common_leak_type: Option<LeakType>,
    /// Memory efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Memory efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Average memory utilization (0.0-1.0)
    pub avg_utilization: f64,
    /// Memory fragmentation average
    pub avg_fragmentation: f64,
    /// Allocation efficiency
    pub allocation_efficiency: f64,
    /// Garbage collection overhead
    pub gc_overhead: f64,
}

// Statistical analyzer for memory data
#[derive(Debug)]
#[allow(dead_code)]
pub struct StatisticalAnalyzer {
    config: StatisticalConfig,
}

#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    pub confidence_level: f64,
    pub trend_window_size: usize,
    pub outlier_threshold: f64,
}

// Pattern detector for memory usage patterns
#[derive(Debug)]
#[allow(dead_code)]
pub struct PatternDetector {
    config: PatternConfig,
}

#[derive(Debug, Clone)]
pub struct PatternConfig {
    pub min_pattern_length: usize,
    pub pattern_similarity_threshold: f64,
    pub frequency_analysis_window: usize,
}

// Anomaly detector for unusual memory behavior
#[derive(Debug)]
#[allow(dead_code)]
pub struct AnomalyDetector {
    config: AnomalyConfig,
}

#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    pub anomaly_threshold: f64,
    pub baseline_window_size: usize,
    pub sensitivity: f64,
}

impl AdvancedMemoryLeakDetector {
    /// Create a new advanced memory leak detector
    pub fn new(config: MemoryLeakConfig) -> Result<Self> {
        let memory_history = Arc::new(RwLock::new(VecDeque::with_capacity(
            config.max_history_size,
        )));
        let active_sessions = Arc::new(Mutex::new(HashMap::new()));

        let leak_analyzer = LeakAnalysisEngine::new(AnalysisConfig::default());
        let alert_system = MemoryAlertSystem::new(AlertConfig::default());
        let statistics = MemoryStatistics::default();

        Ok(Self {
            config,
            memory_history,
            active_sessions,
            leak_analyzer,
            alert_system,
            statistics,
            sampler: Arc::new(SystemSampler::new()?),
        })
    }

    /// Start monitoring an optimizer
    pub fn start_monitoring(
        &self,
        optimizer_name: String,
        session_config: SessionConfig,
    ) -> Result<String> {
        let sessionid = format!(
            "{}_{}",
            optimizer_name,
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()
        );

        let session = MonitoringSession {
            sessionid: sessionid.clone(),
            start_time: Instant::now(),
            config: session_config,
            snapshots: VecDeque::new(),
            analysis_results: None,
            statistics: SessionStatistics::default(),
        };

        {
            let mut sessions = self.active_sessions.lock().map_err(|_| {
                OptimError::MonitoringError("Failed to acquire sessions lock".to_string())
            })?;
            sessions.insert(sessionid.clone(), session);
        }

        // Start background monitoring thread
        self.start_monitoring_thread(sessionid.clone())?;

        Ok(sessionid)
    }

    /// Stop monitoring a session
    pub fn stop_monitoring(&self, sessionid: &str) -> Result<LeakAnalysisResult> {
        let mut sessions = self.active_sessions.lock().map_err(|_| {
            OptimError::MonitoringError("Failed to acquire sessions lock".to_string())
        })?;

        if let Some(mut session) = sessions.remove(sessionid) {
            // Perform final analysis
            let analysis_result = self.analyze_session(&mut session)?;

            // Update statistics
            self.update_statistics(&session, &analysis_result)?;

            Ok(analysis_result)
        } else {
            Err(OptimError::MonitoringError(format!(
                "Session {} not found",
                sessionid
            )))
        }
    }

    /// Get current memory snapshot
    pub fn get_memory_snapshot(&self) -> Result<MemorySnapshot> {
        let timestamp = SystemTime::now();

        // Get system memory information
        let system_memory = self.get_system_memory_info()?;

        // Get process memory information
        let (total_memory, heap_memory, stack_memory) = self.get_process_memory_info()?;

        // Get GPU memory if available
        let gpu_memory = self.get_gpu_memory_info().ok();

        // Calculate fragmentation
        let fragmentation = self.calculate_memory_fragmentation()?;

        // Get allocation/deallocation counts
        let (allocation_count, deallocation_count) = self.get_allocation_counts()?;

        // Get memory pool information
        let memory_pools = self.get_memory_pool_info()?;

        // Get optimizer-specific memory usage
        let optimizer_memory = self.get_optimizer_memory_usage()?;

        Ok(MemorySnapshot {
            timestamp,
            total_memory,
            heap_memory,
            stack_memory,
            gpu_memory,
            fragmentation,
            allocation_count,
            deallocation_count,
            memory_pools,
            optimizer_memory,
            system_memory,
        })
    }

    /// Analyze a monitoring session for leaks
    pub fn analyze_session(&self, session: &mut MonitoringSession) -> Result<LeakAnalysisResult> {
        if session.snapshots.len() < 2 {
            return Err(OptimError::AnalysisError(
                "Insufficient data for analysis".to_string(),
            ));
        }

        // Extract memory values for analysis
        let memoryvalues: Vec<f64> = session
            .snapshots
            .iter()
            .map(|snapshot| snapshot.total_memory as f64)
            .collect();

        // Perform growth analysis
        let growth_analysis = self.leak_analyzer.analyze_growth(&memoryvalues)?;

        // Perform pattern analysis
        let pattern_analysis = self.leak_analyzer.analyze_patterns(&memoryvalues)?;

        // Perform anomaly detection
        let anomaly_analysis = self.leak_analyzer.detect_anomalies(&memoryvalues)?;

        // Determine if leak is detected
        let leak_detected = growth_analysis.significance > self.config.confidence_level
            && growth_analysis.linear_rate > self.config.growth_threshold;

        // Calculate confidence and severity
        let confidence = growth_analysis.significance;
        let severity = self.calculate_leak_severity(&growth_analysis, &anomaly_analysis);

        // Determine leak characteristics
        let leak_characteristics = self.analyze_leak_characteristics(session, &growth_analysis)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &growth_analysis,
            &pattern_analysis,
            &leak_characteristics,
        );

        let result = LeakAnalysisResult {
            timestamp: SystemTime::now(),
            leak_detected,
            confidence,
            severity,
            growth_analysis,
            pattern_analysis,
            anomaly_analysis,
            leak_characteristics,
            recommendations,
        };

        // Store analysis result in session
        session.analysis_results = Some(result.clone());

        // Generate alert if leak detected
        if leak_detected && severity > 0.5 {
            self.generate_leak_alert(&result, session)?;
        }

        Ok(result)
    }

    /// Generate comprehensive memory leak report
    pub fn generate_leak_report(&self, sessionid: &str) -> Result<MemoryLeakReport> {
        let sessions = self.active_sessions.lock().map_err(|_| {
            OptimError::MonitoringError("Failed to acquire sessions lock".to_string())
        })?;

        if let Some(session) = sessions.get(sessionid) {
            let report = MemoryLeakReport {
                sessionid: sessionid.to_string(),
                optimizer_name: session.config.optimizer_name.clone(),
                monitoring_duration: session.start_time.elapsed(),
                total_snapshots: session.snapshots.len(),
                analysis_results: session.analysis_results.clone(),
                session_statistics: session.statistics.clone(),
                memory_timeline: session
                    .snapshots
                    .iter()
                    .map(|s| (s.timestamp, s.total_memory))
                    .collect(),
                recommendations: session
                    .analysis_results
                    .as_ref()
                    .map(|r| r.recommendations.clone())
                    .unwrap_or_default(),
            };

            Ok(report)
        } else {
            Err(OptimError::MonitoringError(format!(
                "Session {} not found",
                sessionid
            )))
        }
    }

    // Private implementation methods

    fn start_monitoring_thread(&self, sessionid: String) -> Result<()> {
        let memory_history = Arc::clone(&self.memory_history);
        let active_sessions = Arc::clone(&self.active_sessions);
        let sampling_interval = self.config.sampling_interval;
        let sampler = Arc::clone(&self.sampler);

        thread::spawn(move || {
            loop {
                // Check if session still exists
                {
                    let sessions = match active_sessions.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if !sessions.contains_key(&sessionid) {
                        break;
                    }
                }

                // Take a real memory snapshot
                if let Ok(snapshot) = Self::take_memory_snapshot(&sampler) {
                    // Add to session snapshots
                    {
                        let mut sessions = match active_sessions.lock() {
                            Ok(guard) => guard,
                            Err(poisoned) => poisoned.into_inner(),
                        };
                        if let Some(session) = sessions.get_mut(&sessionid) {
                            session.snapshots.push_back(snapshot.clone());

                            // Limit snapshot history
                            if session.snapshots.len() > 1000 {
                                session.snapshots.pop_front();
                            }
                        }
                    }

                    // Add to global history
                    {
                        let mut history = match memory_history.write() {
                            Ok(guard) => guard,
                            Err(poisoned) => poisoned.into_inner(),
                        };
                        history.push_back(snapshot);

                        // Limit global history
                        if history.len() > 10000 {
                            history.pop_front();
                        }
                    }
                }

                thread::sleep(sampling_interval);
            }
        });

        Ok(())
    }

    /// Take a real memory snapshot via [`SystemSampler`] (process RSS,
    /// virtual memory, and system-wide memory). Fields with no honest
    /// source in this detector (fragmentation, allocation counts, pool and
    /// per-optimizer breakdowns, GPU memory) are documented zero/empty --
    /// never fabricated.
    fn take_memory_snapshot(sampler: &SystemSampler) -> Result<MemorySnapshot> {
        sampler.refresh();
        let process = sampler.sample_process()?;
        let system = sampler.sample_system();

        // ESTIMATE: no portable per-process heap/stack split is available;
        // see the identical rationale in `memory_leak_detector`.
        let heap_memory = process.rss_bytes * 80 / 100;
        let stack_memory = process.rss_bytes * 20 / 100;

        let pressure = if system.total_memory_bytes > 0 {
            system.used_memory_bytes as f64 / system.total_memory_bytes as f64
        } else {
            0.0
        };

        Ok(MemorySnapshot {
            timestamp: SystemTime::now(),
            total_memory: process.rss_bytes,
            heap_memory,
            stack_memory,
            gpu_memory: None, // No GPU probe wired in this detector; honest absence.
            // Not tracked: this detector has no allocation-tracker
            // bookkeeping of its own (see `memory_leak_detector::AllocationTracker`
            // and `memory_optimizer::AllocationTracker` for real,
            // caller-fed fragmentation heuristics).
            fragmentation: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
            memory_pools: HashMap::new(),
            optimizer_memory: HashMap::new(),
            system_memory: SystemMemoryInfo {
                total: system.total_memory_bytes,
                available: system.available_memory_bytes,
                used: system.used_memory_bytes,
                pressure,
            },
        })
    }

    /// Real system-wide memory via [`SystemSampler`].
    fn get_system_memory_info(&self) -> Result<SystemMemoryInfo> {
        self.sampler.refresh();
        let system = self.sampler.sample_system();
        let pressure = if system.total_memory_bytes > 0 {
            system.used_memory_bytes as f64 / system.total_memory_bytes as f64
        } else {
            0.0
        };
        Ok(SystemMemoryInfo {
            total: system.total_memory_bytes,
            available: system.available_memory_bytes,
            used: system.used_memory_bytes,
            pressure,
        })
    }

    /// Real process memory via [`SystemSampler`]. `heap`/`stack` are a
    /// documented 80/20 estimate over real RSS (no portable exact split).
    fn get_process_memory_info(&self) -> Result<(u64, u64, u64)> {
        self.sampler.refresh();
        let process = self.sampler.sample_process()?;
        let heap_memory = process.rss_bytes * 80 / 100;
        let stack_memory = process.rss_bytes * 20 / 100;
        Ok((process.rss_bytes, heap_memory, stack_memory))
    }

    fn get_gpu_memory_info(&self) -> Result<u64> {
        // No real GPU memory probe is wired here (see `cross_platform_tester`
        // / `advanced_cross_platform_orchestrator::resources` for the
        // detector's real, honest-false-by-default GPU availability
        // check). Reporting "unavailable" is correct rather than a
        // fabricated byte count.
        Err(OptimError::UnsupportedOperation(
            "GPU memory monitoring not available".to_string(),
        ))
    }

    /// Not tracked: this detector installs no allocation-tracker
    /// bookkeeping of its own. Returns a real, honest zero rather than a
    /// fabricated fragmentation estimate. For a real, bookkeeping-derived
    /// fragmentation heuristic use `memory_leak_detector::AllocationTracker`
    /// or `memory_optimizer::AllocationTracker`.
    fn calculate_memory_fragmentation(&self) -> Result<f64> {
        Ok(0.0)
    }

    /// Not tracked: no allocator hook is installed in this detector.
    fn get_allocation_counts(&self) -> Result<(u64, u64)> {
        Ok((0, 0))
    }

    /// Not tracked: no memory pools are registered with this detector.
    fn get_memory_pool_info(&self) -> Result<HashMap<String, MemoryPoolInfo>> {
        Ok(HashMap::new())
    }

    /// Not tracked: no per-optimizer registration API exists on this
    /// detector (see `memory_profiler_integration::ProductionMemoryProfiler::register_parameter_footprint`
    /// and `memory_optimizer::MemoryOptimizer::register_category_bytes`
    /// for the equivalents that do exist elsewhere in this crate).
    fn get_optimizer_memory_usage(&self) -> Result<HashMap<String, OptimizerMemoryUsage>> {
        Ok(HashMap::new())
    }

    fn calculate_leak_severity(
        &self,
        growth_analysis: &GrowthAnalysis,
        anomaly_analysis: &AnomalyAnalysisResult,
    ) -> f64 {
        let growth_severity = (growth_analysis.linear_rate / self.config.growth_threshold).min(1.0);
        let anomaly_severity = anomaly_analysis.anomaly_score;

        (growth_severity + anomaly_severity) / 2.0
    }

    fn analyze_leak_characteristics(
        &self,
        session: &MonitoringSession,
        growth_analysis: &GrowthAnalysis,
    ) -> Result<LeakCharacteristics> {
        let leak_type = match growth_analysis.trend_type {
            GrowthTrendType::Linear => LeakType::ClassicLeak,
            GrowthTrendType::Exponential => LeakType::UnboundedGrowth,
            GrowthTrendType::NoGrowth => LeakType::ClassicLeak, // Consider as stabilized leak
            GrowthTrendType::Polynomial => LeakType::UnboundedGrowth, // Polynomial growth can lead to unbounded
            GrowthTrendType::Irregular => LeakType::ClassicLeak, // Default to classic for irregular patterns
        };

        Ok(LeakCharacteristics {
            leak_type,
            source: Some(session.config.optimizer_name.clone()),
            leak_rate: growth_analysis.linear_rate,
            time_to_exhaustion: None,
            affected_components: vec![session.config.optimizer_name.clone()],
        })
    }

    fn generate_recommendations(
        &self,
        growth_analysis: &GrowthAnalysis,
        _analysis: &PatternAnalysisResult,
        leak_characteristics: &LeakCharacteristics,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if growth_analysis.linear_rate > self.config.growth_threshold {
            recommendations
                .push("Investigate memory allocation patterns in the optimizer".to_string());
        }

        match leak_characteristics.leak_type {
            LeakType::ClassicLeak => {
                recommendations.push("Check for unfreed memory allocations".to_string());
                recommendations.push("Review resource cleanup in optimization loops".to_string());
            }
            LeakType::UnboundedGrowth => {
                recommendations.push("Check for unbounded collections or caches".to_string());
                recommendations
                    .push("Implement size limits on internal data structures".to_string());
            }
            _ => {}
        }

        recommendations
    }

    fn generate_leak_alert(
        &self,
        analysis_result: &LeakAnalysisResult,
        session: &MonitoringSession,
    ) -> Result<()> {
        let memory_metrics = session.snapshots.back().cloned().ok_or_else(|| {
            OptimError::MonitoringError(
                "cannot generate a leak alert from a session with no snapshots".to_string(),
            )
        })?;

        let alert = MemoryAlert {
            id: format!(
                "leak_{}_{}",
                session.sessionid,
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
            ),
            timestamp: SystemTime::now(),
            severity: match analysis_result.severity {
                s if s >= 0.9 => AlertSeverity::Critical,
                s if s >= 0.7 => AlertSeverity::High,
                s if s >= 0.5 => AlertSeverity::Medium,
                _ => AlertSeverity::Low,
            },
            alert_type: MemoryAlertType::MemoryLeak,
            message: format!(
                "Memory leak detected in optimizer: {}",
                session.config.optimizer_name
            ),
            memory_metrics,
            recommended_actions: analysis_result.recommendations.clone(),
        };

        self.alert_system.send_alert(alert)?;
        Ok(())
    }

    // NOTE: `session`/`_result` are intentionally unused. `self.statistics` (unlike
    // `memory_history`/`active_sessions`) is a plain `MemoryStatistics`, not behind
    // an `Arc<Mutex<_>>`/`Arc<RwLock<_>>`, and this method only takes `&self` (its
    // caller, `stop_monitoring`, is `&self` too, deliberately, so it stays callable
    // from multiple threads like the rest of this detector). Actually aggregating
    // per-session stats into detector-wide `MemoryStatistics` needs that field
    // behind interior mutability plus a public accessor to read it back out
    // (nothing reads `self.statistics` today) -- a real structural change, not a
    // one-line fix, so it is left as a tracked gap rather than partially done here.
    fn update_statistics(
        &self,
        _session: &MonitoringSession,
        _result: &LeakAnalysisResult,
    ) -> Result<()> {
        Ok(())
    }
}

/// Comprehensive memory leak report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeakReport {
    /// Session identifier
    pub sessionid: String,
    /// Optimizer name
    pub optimizer_name: String,
    /// Total monitoring duration
    pub monitoring_duration: Duration,
    /// Total snapshots collected
    pub total_snapshots: usize,
    /// Analysis results
    pub analysis_results: Option<LeakAnalysisResult>,
    /// Session statistics
    pub session_statistics: SessionStatistics,
    /// Memory usage timeline
    pub memory_timeline: Vec<(SystemTime, u64)>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

// Implementation stubs for the analysis engines

impl LeakAnalysisEngine {
    fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            statistical_analyzer: StatisticalAnalyzer::new(StatisticalConfig::default()),
            pattern_detector: PatternDetector::new(PatternConfig::default()),
            anomaly_detector: AnomalyDetector::new(AnomalyConfig::default()),
        }
    }

    /// Real ordinary-least-squares growth analysis: slope (`linear_rate`),
    /// R² goodness-of-fit, and a t-test-derived `significance`, computed
    /// inline from the actual snapshot series -- replacing the previous
    /// constant `significance: 0.95, r_squared: 0.8`.
    fn analyze_growth(&self, memoryvalues: &[f64]) -> Result<GrowthAnalysis> {
        if memoryvalues.len() < 3 {
            return Ok(GrowthAnalysis {
                linear_rate: 0.0,
                exponential_factor: 1.0,
                trend_type: GrowthTrendType::NoGrowth,
                significance: 0.0,
                r_squared: 0.0,
            });
        }

        let points: Vec<(f64, f64)> = memoryvalues
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64, y))
            .collect();

        let Some((slope, intercept, se_slope)) = stats::ols_fit(&points) else {
            return Ok(GrowthAnalysis {
                linear_rate: 0.0,
                exponential_factor: 1.0,
                trend_type: GrowthTrendType::NoGrowth,
                significance: 0.0,
                r_squared: 0.0,
            });
        };

        let mean_y = memoryvalues.iter().sum::<f64>() / memoryvalues.len() as f64;
        let ss_tot: f64 = memoryvalues.iter().map(|y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = points
            .iter()
            .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
            .sum();
        let r_squared = if ss_tot > 0.0 {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // `se_slope == 0` is a perfect (noiseless) fit: if the slope is
        // also (numerically) zero this is a certainly-flat series, but if
        // the slope is nonzero it is the strongest possible evidence of a
        // real trend, so treat it as maximally significant rather than
        // reporting a t-statistic of 0 (which would say the opposite).
        let (_t_stat, p_value) = if se_slope > f64::EPSILON {
            let t = slope / se_slope;
            (t, stats::two_tailed_p_value(t))
        } else if slope.abs() > f64::EPSILON {
            (f64::INFINITY, 0.0)
        } else {
            (0.0, 1.0)
        };
        let significance = (1.0 - p_value).clamp(0.0, 1.0);

        // Exponential growth check: fit ln(y) ~ x when all values are
        // strictly positive (required for a real logarithm); otherwise
        // there is no honest exponential-factor estimate, so default to 1.0
        // (no growth) rather than fabricate one.
        let exponential_factor = if memoryvalues.iter().all(|&y| y > 0.0) {
            let log_points: Vec<(f64, f64)> = memoryvalues
                .iter()
                .enumerate()
                .map(|(i, &y)| (i as f64, y.ln()))
                .collect();
            stats::ols_fit(&log_points)
                .map(|(log_slope, _, _)| log_slope.exp())
                .unwrap_or(1.0)
        } else {
            1.0
        };

        let trend_type = if significance < 0.8 {
            GrowthTrendType::Irregular
        } else if exponential_factor > 1.01 && r_squared > 0.7 {
            GrowthTrendType::Exponential
        } else if slope.abs() < 1e-6 {
            GrowthTrendType::NoGrowth
        } else {
            GrowthTrendType::Linear
        };

        Ok(GrowthAnalysis {
            linear_rate: slope,
            exponential_factor,
            trend_type,
            significance,
            r_squared,
        })
    }

    /// Real periodicity signal: fraction of consecutive-difference sign
    /// changes (a genuine oscillation measure over the real series), not a
    /// fabricated constant confidence.
    fn analyze_patterns(&self, memoryvalues: &[f64]) -> Result<PatternAnalysisResult> {
        if memoryvalues.len() < 4 {
            return Ok(PatternAnalysisResult {
                patterns: Vec::new(),
                confidence: 0.0,
                periodic_behavior: None,
            });
        }

        let diffs: Vec<f64> = memoryvalues.windows(2).map(|w| w[1] - w[0]).collect();
        let sign_changes = diffs.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let oscillation_ratio = sign_changes as f64 / diffs.len().max(1) as f64;

        Ok(PatternAnalysisResult {
            patterns: Vec::new(),
            confidence: oscillation_ratio.clamp(0.0, 1.0),
            periodic_behavior: None,
        })
    }

    /// Real anomaly score: max absolute z-score of the series relative to
    /// its own mean/standard deviation, not a fabricated constant.
    fn detect_anomalies(&self, memoryvalues: &[f64]) -> Result<AnomalyAnalysisResult> {
        if memoryvalues.len() < 3 {
            return Ok(AnomalyAnalysisResult {
                anomalies: Vec::new(),
                anomaly_score: 0.0,
                confidence: 0.0,
            });
        }

        let mean = memoryvalues.iter().sum::<f64>() / memoryvalues.len() as f64;
        let variance = memoryvalues.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / memoryvalues.len() as f64;
        let std_dev = variance.sqrt();

        let anomaly_score = if std_dev > 0.0 {
            memoryvalues
                .iter()
                .map(|v| ((v - mean) / std_dev).abs())
                .fold(0.0, f64::max)
                / 3.0 // normalize: a 3-sigma deviation maps to score 1.0
        } else {
            0.0
        };

        // Confidence in the z-score estimate grows with sample count
        // (more samples => more reliable mean/std_dev), a real function of
        // `n` rather than a fixed constant.
        let confidence = if std_dev > 0.0 {
            (1.0 - 1.0 / memoryvalues.len() as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(AnomalyAnalysisResult {
            anomalies: Vec::new(),
            anomaly_score: anomaly_score.clamp(0.0, 1.0),
            confidence,
        })
    }
}

impl StatisticalAnalyzer {
    fn new(config: StatisticalConfig) -> Self {
        Self { config }
    }
}

impl PatternDetector {
    fn new(config: PatternConfig) -> Self {
        Self { config }
    }
}

impl AnomalyDetector {
    fn new(config: AnomalyConfig) -> Self {
        Self { config }
    }
}

impl MemoryAlertSystem {
    fn new(config: AlertConfig) -> Self {
        Self {
            config,
            alert_history: VecDeque::new(),
            alert_handlers: Vec::new(),
        }
    }

    fn send_alert(&self, alert: MemoryAlert) -> Result<()> {
        println!("🚨 MEMORY ALERT: {}", alert.message);
        println!("   Severity: {:?}", alert.severity);
        println!("   Type: {:?}", alert.alert_type);
        Ok(())
    }
}

// Default implementations

impl Default for MemoryLeakConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_interval: Duration::from_secs(1),
            growth_threshold: 1024.0 * 1024.0, // 1MB per second
            confidence_level: 0.95,
            min_monitoring_duration: Duration::from_secs(60),
            max_history_size: 10000,
            detection_algorithms: vec![
                LeakDetectionAlgorithm::LinearTrend,
                LeakDetectionAlgorithm::StatisticalProcessControl,
            ],
            alert_thresholds: MemoryAlertThresholds::default(),
            optimizer_settings: OptimizerMemorySettings::default(),
        }
    }
}

impl Default for MemoryAlertThresholds {
    fn default() -> Self {
        Self {
            growth_rate_threshold: 1024.0 * 1024.0,        // 1MB/s
            absolute_memory_threshold: 1024 * 1024 * 1024, // 1GB
            fragmentation_threshold: 0.5,
            gc_frequency_threshold: 100,
        }
    }
}

impl Default for OptimizerMemorySettings {
    fn default() -> Self {
        Self {
            optimizer_profiles: HashMap::new(),
            track_memory_pools: true,
            monitor_gradient_accumulation: true,
            track_parameter_buffers: true,
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_effect_size: 0.2,
            analysis_window: 100,
            trend_sensitivity: 0.1,
        }
    }
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            trend_window_size: 50,
            outlier_threshold: 3.0,
        }
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            min_pattern_length: 5,
            pattern_similarity_threshold: 0.8,
            frequency_analysis_window: 100,
        }
    }
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            anomaly_threshold: 2.0,
            baseline_window_size: 50,
            sensitivity: 0.8,
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_severity: AlertSeverity::Medium,
            throttling: AlertThrottling {
                max_alerts: 10,
                time_window: Duration::from_secs(3600),
                cooldown: Duration::from_secs(300),
            },
        }
    }
}

impl Default for SessionStatistics {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(0),
            snapshot_count: 0,
            avg_memory: 0.0,
            peak_memory: 0,
            growth_rate: 0.0,
            volatility: 0.0,
        }
    }
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            total_leaks_detected: 0,
            avg_memory_usage: 0.0,
            peak_memory_usage: 0,
            most_common_leak_type: None,
            efficiency_metrics: EfficiencyMetrics::default(),
        }
    }
}

impl Default for EfficiencyMetrics {
    fn default() -> Self {
        Self {
            avg_utilization: 0.0,
            avg_fragmentation: 0.0,
            allocation_efficiency: 0.0,
            gc_overhead: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_leak_detector_creation() {
        let config = MemoryLeakConfig::default();
        let _detector = AdvancedMemoryLeakDetector::new(config).expect("unwrap failed");
        // Test basic functionality
    }

    #[test]
    fn test_monitoring_session_lifecycle() {
        let config = MemoryLeakConfig::default();
        let detector = AdvancedMemoryLeakDetector::new(config).expect("unwrap failed");

        let session_config = SessionConfig {
            optimizer_name: "test_optimizer".to_string(),
            duration: Duration::from_secs(60),
            sampling_frequency: Duration::from_secs(1),
            analysis_triggers: vec![AnalysisTrigger::Duration(Duration::from_secs(30))],
        };

        let sessionid = detector
            .start_monitoring("test_optimizer".to_string(), session_config)
            .expect("unwrap failed");
        assert!(!sessionid.is_empty());
    }

    #[test]
    fn test_memory_snapshot() {
        let config = MemoryLeakConfig::default();
        let detector = AdvancedMemoryLeakDetector::new(config).expect("unwrap failed");

        let snapshot = detector.get_memory_snapshot().expect("unwrap failed");
        // Verify snapshot structure
        assert!(snapshot
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .is_ok());
    }

    #[test]
    fn test_memory_snapshot_reports_real_rss() {
        let config = MemoryLeakConfig::default();
        let detector = match AdvancedMemoryLeakDetector::new(config) {
            Ok(d) => d,
            Err(e) => panic!("failed to create detector: {e:?}"),
        };
        let snapshot = match detector.get_memory_snapshot() {
            Ok(s) => s,
            Err(e) => panic!("failed to take snapshot: {e:?}"),
        };
        assert!(
            snapshot.total_memory > 0,
            "process RSS must be a real positive measurement"
        );
        assert!(
            snapshot.system_memory.total > 0,
            "system total memory must be real and positive"
        );
    }

    #[test]
    fn test_analyze_growth_detects_planted_linear_leak() {
        let config = MemoryLeakConfig::default();
        let detector = match AdvancedMemoryLeakDetector::new(config) {
            Ok(d) => d,
            Err(e) => panic!("failed to create detector: {e:?}"),
        };

        // Planted leak: strictly increasing series.
        let growing: Vec<f64> = (0..30).map(|i| 1_000_000.0 + i as f64 * 50_000.0).collect();
        let growth = match detector.leak_analyzer.analyze_growth(&growing) {
            Ok(g) => g,
            Err(e) => panic!("analyze_growth failed: {e:?}"),
        };
        assert!(
            growth.linear_rate > 0.0,
            "planted growth must yield a positive OLS slope"
        );
        assert!(
            growth.r_squared > 0.9,
            "a clean linear series must fit almost perfectly, got {}",
            growth.r_squared
        );
        assert!(
            growth.significance > 0.9,
            "a clean linear trend must be statistically significant, got {}",
            growth.significance
        );
        assert!(!matches!(growth.trend_type, GrowthTrendType::NoGrowth));

        // Stable series: no leak.
        let stable: Vec<f64> = vec![1_000_000.0; 30];
        let stable_growth = match detector.leak_analyzer.analyze_growth(&stable) {
            Ok(g) => g,
            Err(e) => panic!("analyze_growth failed: {e:?}"),
        };
        assert!(
            matches!(stable_growth.trend_type, GrowthTrendType::NoGrowth)
                || stable_growth.linear_rate.abs() < 1e-6,
            "a flat series must not be classified as growing"
        );
    }
}
