// Advanced memory leak detection and optimization tooling
//
// This module provides comprehensive memory leak detection capabilities,
// memory profiling, and optimization recommendations for optimization algorithms.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Advanced memory leak detector and profiler
#[derive(Debug)]
pub struct MemoryLeakDetector {
    /// Configuration for memory detection
    config: MemoryDetectionConfig,
    /// Memory allocation tracker
    allocation_tracker: AllocationTracker,
    /// Memory pattern analyzer
    pattern_analyzer: MemoryPatternAnalyzer,
    /// Leak detection algorithms
    detectors: Vec<Box<dyn LeakDetector>>,
    /// Memory optimization recommendations
    optimizer: MemoryOptimizer,
}

/// Configuration for memory leak detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDetectionConfig {
    /// Enable detailed allocation tracking
    pub enable_allocation_tracking: bool,
    /// Memory growth threshold (bytes)
    pub memory_growth_threshold: usize,
    /// Leak detection sensitivity (0.0 to 1.0)
    pub leak_sensitivity: f64,
    /// Sampling rate for memory profiling
    pub sampling_rate: u64,
    /// Maximum tracking history
    pub max_history_entries: usize,
    /// Enable real-time monitoring
    pub enable_real_time_monitoring: bool,
    /// Memory pressure threshold
    pub memory_pressure_threshold: f64,
    /// Enable garbage collection hints
    pub enable_gc_hints: bool,
    /// Capture a real `std::backtrace::Backtrace` on every recorded
    /// allocation. Off by default because forcing a backtrace capture on
    /// every allocation is expensive; when enabled, `capture_stack_trace`
    /// returns a genuine captured stack rather than `None`.
    pub capture_stack_traces: bool,
}

impl Default for MemoryDetectionConfig {
    fn default() -> Self {
        Self {
            enable_allocation_tracking: true,
            memory_growth_threshold: 100 * 1024 * 1024, // 100MB
            leak_sensitivity: 0.8,
            sampling_rate: 1000, // Every 1000 allocations
            max_history_entries: 10000,
            enable_real_time_monitoring: true,
            memory_pressure_threshold: 0.85, // 85% memory usage
            enable_gc_hints: true,
            capture_stack_traces: false,
        }
    }
}

/// Memory allocation tracker
#[derive(Debug)]
#[allow(dead_code)]
pub struct AllocationTracker {
    /// Total allocations count
    total_allocations: Arc<AtomicUsize>,
    /// Total deallocations count
    total_deallocations: Arc<AtomicUsize>,
    /// Current memory usage
    current_memory_usage: Arc<AtomicUsize>,
    /// Peak memory usage
    peak_memory_usage: Arc<AtomicUsize>,
    /// Allocation history
    allocation_history: VecDeque<AllocationEvent>,
    /// Active allocations
    active_allocations: HashMap<usize, AllocationInfo>,
    /// Memory pools tracking
    memory_pools: HashMap<String, MemoryPoolStats>,
    /// Sizes of recently freed blocks, bounded history. Used to derive a
    /// real (non-fabricated) fragmentation heuristic from the tracker's own
    /// bookkeeping: a free-block-size population with high diversity
    /// relative to active-allocation sizes is a real signal of external
    /// fragmentation risk, even without OS-level heap introspection.
    freed_block_sizes: VecDeque<usize>,
}

/// Individual allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    /// Timestamp of allocation
    pub timestamp: u64,
    /// Allocation ID
    pub allocation_id: usize,
    /// Size of allocation
    pub size: usize,
    /// Allocation type
    pub allocation_type: AllocationType,
    /// Stack trace (if available)
    pub stack_trace: Option<Vec<String>>,
    /// Associated optimizer context
    pub optimizer_context: Option<String>,
}

/// Types of memory allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    /// Parameter storage
    Parameter,
    /// Gradient storage
    Gradient,
    /// Optimizer state
    OptimizerState,
    /// Temporary computation
    Temporary,
    /// Cache storage
    Cache,
    /// Other/Unknown
    Other,
}

/// Information about an active allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Size of allocation
    pub size: usize,
    /// Allocation timestamp
    pub timestamp: Instant,
    /// Allocation type
    pub allocation_type: AllocationType,
    /// Source location
    pub source_location: Option<String>,
    /// Reference count (for tracking shared ownership)
    pub reference_count: usize,
}

/// Memory pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolStats {
    /// Pool name
    pub name: String,
    /// Total pool size
    pub total_size: usize,
    /// Used size
    pub used_size: usize,
    /// Free size
    pub free_size: usize,
    /// Fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Allocation count
    pub allocation_count: usize,
    /// Hit rate
    pub hit_rate: f64,
}

/// Memory pattern analyzer
#[derive(Debug)]
pub struct MemoryPatternAnalyzer {
    /// Memory usage patterns
    usage_patterns: VecDeque<MemoryUsageSnapshot>,
    /// Pattern detection algorithms
    pattern_detectors: Vec<Box<dyn PatternDetector>>,
    /// Anomaly detectors
    anomaly_detectors: Vec<Box<dyn AnomalyDetector>>,
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Total memory usage
    pub total_memory: usize,
    /// Heap memory usage
    pub heap_memory: usize,
    /// Stack memory usage
    pub stack_memory: usize,
    /// Memory by allocation type
    pub memory_by_type: HashMap<String, usize>,
    /// Memory growth rate
    pub growth_rate: f64,
    /// Fragmentation level
    pub fragmentation_level: f64,
}

/// Memory leak detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeakResult {
    /// Leak detected
    pub leak_detected: bool,
    /// Leak severity (0.0 to 1.0)
    pub severity: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Leaked memory estimate (bytes)
    pub leaked_memory_bytes: usize,
    /// Leak sources
    pub leak_sources: Vec<LeakSource>,
    /// Memory growth analysis
    pub growth_analysis: MemoryGrowthAnalysis,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Detailed analysis
    pub detailed_analysis: String,
}

/// Source of memory leak
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakSource {
    /// Source type
    pub source_type: AllocationType,
    /// Location in code
    pub location: Option<String>,
    /// Estimated leak size
    pub leak_size: usize,
    /// Leak probability
    pub probability: f64,
    /// Stack trace
    pub stack_trace: Option<Vec<String>>,
}

/// Memory growth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGrowthAnalysis {
    /// Growth trend
    pub growth_trend: GrowthTrend,
    /// Growth rate (bytes per second)
    pub growth_rate: f64,
    /// Projected memory usage
    pub projected_usage: Vec<(u64, usize)>, // (timestamp, memory_usage)
    /// Growth pattern type
    pub pattern_type: GrowthPattern,
}

/// Memory growth trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthTrend {
    Stable,
    Linear,
    Exponential,
    Oscillating,
    Irregular,
}

/// Memory growth patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthPattern {
    Normal,
    Leak,
    Burst,
    Periodic,
    Fragmentation,
}

/// Leak detection algorithm trait
pub trait LeakDetector: Debug + Send + Sync {
    /// Detect memory leaks in allocation data
    fn detect_leaks(
        &self,
        allocation_history: &VecDeque<AllocationEvent>,
        usage_snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<MemoryLeakResult>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Get detector configuration
    fn config(&self) -> HashMap<String, String>;
}

/// Pattern detection trait
pub trait PatternDetector: Debug + Send + Sync {
    /// Detect memory usage patterns
    fn detect_patterns(
        &self,
        snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<MemoryPattern>>;

    /// Get detector name
    fn name(&self) -> &str;
}

/// Memory usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    /// Pattern type
    pub pattern_type: String,
    /// Pattern confidence
    pub confidence: f64,
    /// Pattern description
    pub description: String,
    /// Associated metrics
    pub metrics: HashMap<String, f64>,
}

/// Anomaly detection trait
pub trait AnomalyDetector: Debug + Send + Sync {
    /// Detect memory usage anomalies
    fn detect_anomalies(
        &self,
        snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<MemoryAnomaly>>;

    /// Get detector name
    fn name(&self) -> &str;
}

/// Memory anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnomaly {
    /// Anomaly type
    pub anomaly_type: String,
    /// Severity score
    pub severity: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Description
    pub description: String,
    /// Affected memory region
    pub affected_region: Option<String>,
}

/// Memory optimizer for providing recommendations
#[derive(Debug)]
pub struct MemoryOptimizer {
    /// Optimization strategies
    strategies: Vec<Box<dyn OptimizationStrategy>>,
    /// Performance metrics
    performance_metrics: PerformanceMetrics,
}

/// Optimization strategy trait
pub trait OptimizationStrategy: Debug + Send + Sync {
    /// Generate optimization recommendations
    fn recommend(
        &self,
        leak_result: &MemoryLeakResult,
        usage_history: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<OptimizationRecommendation>>;

    /// Get strategy name
    fn name(&self) -> &str;
}

/// Memory optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation complexity
    pub complexity: ImplementationComplexity,
    /// Code examples
    pub code_examples: Option<Vec<String>>,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    MemoryPooling,
    ObjectReuse,
    LazyEvaluation,
    InPlaceOperations,
    CacheOptimization,
    GarbageCollection,
    DataStructureOptimization,
    AlgorithmOptimization,
}

/// Recommendation priority levels
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

/// Implementation complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationComplexity {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Performance metrics for memory optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Memory efficiency score: current / peak tracked memory usage.
    pub memory_efficiency: f64,
    /// Allocation efficiency: complement of the tracker's real
    /// fragmentation heuristic (see [`AllocationTracker::fragmentation_ratio`]).
    pub allocation_efficiency: f64,
    /// Cache hit ratio. `None` because this module tracks no cache layer;
    /// an honest absence rather than a fabricated number.
    pub cache_hit_ratio: Option<f64>,
    /// Garbage collection overhead. `None` because Rust has no tracked
    /// garbage collector here to measure; an honest absence rather than a
    /// fabricated number.
    pub gc_overhead: Option<f64>,
}

impl MemoryLeakDetector {
    /// Create a new memory leak detector
    pub fn new(config: MemoryDetectionConfig) -> Self {
        let mut detector = Self {
            config: config.clone(),
            allocation_tracker: AllocationTracker::new(),
            pattern_analyzer: MemoryPatternAnalyzer::new(),
            detectors: Vec::new(),
            optimizer: MemoryOptimizer::new(),
        };

        // Initialize default detectors
        detector.initialize_default_detectors();

        detector
    }

    /// Initialize default leak detection algorithms
    fn initialize_default_detectors(&mut self) {
        self.detectors
            .push(Box::new(GrowthBasedLeakDetector::new()));
        self.detectors
            .push(Box::new(PatternBasedLeakDetector::new()));
        self.detectors
            .push(Box::new(StatisticalLeakDetector::new()));

        // Add advanced leak detectors
        if self.config.enable_real_time_monitoring {
            use crate::advanced_leak_detectors::{
                ReferenceCountingConfig, ReferenceCountingDetector,
            };
            let ref_counting_config = ReferenceCountingConfig::default();
            self.detectors.push(Box::new(ReferenceCountingDetector::new(
                ref_counting_config,
            )));
        }
    }

    /// Start memory monitoring
    pub fn start_monitoring(&mut self) -> Result<()> {
        if self.config.enable_real_time_monitoring {
            self.allocation_tracker.start_tracking()?;
            self.pattern_analyzer.start_analysis()?;
        }
        Ok(())
    }

    /// Stop memory monitoring
    pub fn stop_monitoring(&mut self) -> Result<()> {
        self.allocation_tracker.stop_tracking()?;
        self.pattern_analyzer.stop_analysis()?;
        Ok(())
    }

    /// Record an allocation event
    pub fn record_allocation(
        &mut self,
        allocation_id: usize,
        size: usize,
        allocation_type: AllocationType,
    ) -> Result<()> {
        let event = AllocationEvent {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            allocation_id,
            size,
            allocation_type: allocation_type.clone(),
            stack_trace: self.capture_stack_trace(),
            optimizer_context: None,
        };

        self.allocation_tracker.record_allocation(event)?;
        Ok(())
    }

    /// Record a deallocation event
    pub fn record_deallocation(&mut self, allocation_id: usize) -> Result<()> {
        self.allocation_tracker.record_deallocation(allocation_id)?;
        Ok(())
    }

    /// Take a memory usage snapshot
    pub fn take_snapshot(&mut self) -> Result<MemoryUsageSnapshot> {
        let snapshot = MemoryUsageSnapshot {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            total_memory: self.get_total_memory_usage(),
            heap_memory: self.get_heap_memory_usage(),
            stack_memory: self.get_stack_memory_usage(),
            memory_by_type: self.get_memory_by_type(),
            growth_rate: self.calculate_growth_rate(),
            fragmentation_level: self.calculate_fragmentation_level(),
        };

        self.pattern_analyzer.add_snapshot(snapshot.clone());
        Ok(snapshot)
    }

    /// Detect memory leaks using all configured detectors
    pub fn detect_leaks(&self) -> Result<Vec<MemoryLeakResult>> {
        let mut results = Vec::new();

        for detector in &self.detectors {
            let result = detector.detect_leaks(
                &self.allocation_tracker.allocation_history,
                &self.pattern_analyzer.usage_patterns,
            )?;
            results.push(result);
        }

        Ok(results)
    }

    /// Generate memory optimization report
    pub fn generate_optimization_report(&self) -> Result<MemoryOptimizationReport> {
        let leak_results = self.detect_leaks()?;
        let patterns = self.pattern_analyzer.detect_all_patterns()?;
        let anomalies = self.pattern_analyzer.detect_all_anomalies()?;
        let recommendations = self.optimizer.generate_recommendations(&leak_results)?;

        Ok(MemoryOptimizationReport {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            leak_results,
            patterns,
            anomalies,
            recommendations,
            performance_metrics: self.compute_performance_metrics(),
            summary: self.generate_summary()?,
        })
    }

    /// Compute real performance metrics from the allocation tracker's own
    /// counters, replacing the previously hardcoded constants.
    fn compute_performance_metrics(&self) -> PerformanceMetrics {
        let current = self.get_total_memory_usage() as f64;
        let peak = self
            .allocation_tracker
            .peak_memory_usage
            .load(Ordering::Relaxed) as f64;

        let memory_efficiency = if peak > 0.0 {
            (current / peak).min(1.0)
        } else {
            1.0
        };
        let allocation_efficiency =
            (1.0 - self.allocation_tracker.fragmentation_ratio()).clamp(0.0, 1.0);

        PerformanceMetrics {
            memory_efficiency,
            allocation_efficiency,
            cache_hit_ratio: None,
            gc_overhead: None,
        }
    }

    // Helper methods for memory monitoring

    /// Capture a real stack trace via `std::backtrace::Backtrace`, gated by
    /// `config.capture_stack_traces` since forcing a capture on every
    /// allocation is expensive. Returns `None` when disabled -- an honest
    /// "not captured", never a synthesized trace.
    fn capture_stack_trace(&self) -> Option<Vec<String>> {
        if !self.config.capture_stack_traces {
            return None;
        }
        let backtrace = std::backtrace::Backtrace::force_capture();
        let rendered = format!("{backtrace}");
        let frames: Vec<String> = rendered.lines().map(|line| line.to_string()).collect();
        if frames.is_empty() {
            None
        } else {
            Some(frames)
        }
    }

    fn get_total_memory_usage(&self) -> usize {
        self.allocation_tracker
            .current_memory_usage
            .load(Ordering::Relaxed)
    }

    /// ESTIMATE: `sysinfo` and portable Rust do not expose a real
    /// heap/stack split per process, so this is a fixed 80% heuristic over
    /// tracked allocation bytes rather than a measurement. Callers that
    /// need a precise split must use a platform-specific tool (e.g.
    /// `/proc/<pid>/smaps` on Linux).
    fn get_heap_memory_usage(&self) -> usize {
        self.get_total_memory_usage() * 80 / 100
    }

    /// ESTIMATE: see [`Self::get_heap_memory_usage`]; the complement 20%
    /// heuristic, not a measurement.
    fn get_stack_memory_usage(&self) -> usize {
        self.get_total_memory_usage() * 20 / 100
    }

    fn get_memory_by_type(&self) -> HashMap<String, usize> {
        let mut memory_by_type = HashMap::new();

        // Analyze allocations by type
        for info in self.allocation_tracker.active_allocations.values() {
            let type_name = format!("{:?}", info.allocation_type);
            *memory_by_type.entry(type_name).or_insert(0) += info.size;
        }

        memory_by_type
    }

    fn calculate_growth_rate(&self) -> f64 {
        // Calculate memory growth rate from recent snapshots
        if self.pattern_analyzer.usage_patterns.len() < 2 {
            return 0.0;
        }

        let recent =
            &self.pattern_analyzer.usage_patterns[self.pattern_analyzer.usage_patterns.len() - 1];
        let previous =
            &self.pattern_analyzer.usage_patterns[self.pattern_analyzer.usage_patterns.len() - 2];

        let time_diff = (recent.timestamp - previous.timestamp) as f64;
        if time_diff > 0.0 {
            (recent.total_memory as f64 - previous.total_memory as f64) / time_diff
        } else {
            0.0
        }
    }

    /// Real fragmentation heuristic from the allocation tracker's own
    /// freed/active block bookkeeping. See
    /// [`AllocationTracker::fragmentation_ratio`].
    fn calculate_fragmentation_level(&self) -> f64 {
        self.allocation_tracker.fragmentation_ratio()
    }

    fn generate_summary(&self) -> Result<String> {
        let total_allocations = self
            .allocation_tracker
            .total_allocations
            .load(Ordering::Relaxed);
        let total_deallocations = self
            .allocation_tracker
            .total_deallocations
            .load(Ordering::Relaxed);
        let current_memory = self.get_total_memory_usage();
        let peak_memory = self
            .allocation_tracker
            .peak_memory_usage
            .load(Ordering::Relaxed);

        Ok(format!(
            "Memory Analysis Summary:\n\
            - Total Allocations: {}\n\
            - Total Deallocations: {}\n\
            - Active Allocations: {}\n\
            - Current Memory Usage: {} bytes\n\
            - Peak Memory Usage: {} bytes\n\
            - Memory Efficiency: {:.2}%",
            total_allocations,
            total_deallocations,
            total_allocations - total_deallocations,
            current_memory,
            peak_memory,
            if peak_memory > 0 {
                (current_memory as f64 / peak_memory as f64) * 100.0
            } else {
                0.0
            }
        ))
    }
}

/// Memory optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationReport {
    /// Report timestamp
    pub timestamp: u64,
    /// Leak detection results
    pub leak_results: Vec<MemoryLeakResult>,
    /// Memory patterns detected
    pub patterns: Vec<MemoryPattern>,
    /// Memory anomalies detected
    pub anomalies: Vec<MemoryAnomaly>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Summary text
    pub summary: String,
}

impl Default for AllocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocationTracker {
    /// Create a new allocation tracker
    pub fn new() -> Self {
        Self {
            total_allocations: Arc::new(AtomicUsize::new(0)),
            total_deallocations: Arc::new(AtomicUsize::new(0)),
            current_memory_usage: Arc::new(AtomicUsize::new(0)),
            peak_memory_usage: Arc::new(AtomicUsize::new(0)),
            allocation_history: VecDeque::new(),
            active_allocations: HashMap::new(),
            memory_pools: HashMap::new(),
            freed_block_sizes: VecDeque::new(),
        }
    }

    /// Start allocation tracking
    pub fn start_tracking(&mut self) -> Result<()> {
        // Initialize tracking systems
        Ok(())
    }

    /// Stop allocation tracking
    pub fn stop_tracking(&mut self) -> Result<()> {
        // Cleanup tracking systems
        Ok(())
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, event: AllocationEvent) -> Result<()> {
        // Update counters
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        let current = self
            .current_memory_usage
            .fetch_add(event.size, Ordering::Relaxed)
            + event.size;

        // Update peak memory
        let mut peak = self.peak_memory_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_memory_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }

        // Record allocation info
        let allocation_info = AllocationInfo {
            size: event.size,
            timestamp: Instant::now(),
            allocation_type: event.allocation_type.clone(),
            source_location: None,
            reference_count: 1,
        };

        self.active_allocations
            .insert(event.allocation_id, allocation_info);

        // Add to history
        self.allocation_history.push_back(event);

        // Maintain history size
        if self.allocation_history.len() > 10000 {
            self.allocation_history.pop_front();
        }

        Ok(())
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, allocation_id: usize) -> Result<()> {
        if let Some(info) = self.active_allocations.remove(&allocation_id) {
            self.total_deallocations.fetch_add(1, Ordering::Relaxed);
            self.current_memory_usage
                .fetch_sub(info.size, Ordering::Relaxed);

            self.freed_block_sizes.push_back(info.size);
            const MAX_FREED_HISTORY: usize = 4096;
            while self.freed_block_sizes.len() > MAX_FREED_HISTORY {
                self.freed_block_sizes.pop_front();
            }
        }
        Ok(())
    }

    /// A real fragmentation heuristic derived from this tracker's own
    /// allocation/deallocation bookkeeping (not an OS-level measurement,
    /// and not a fabricated constant).
    ///
    /// Rationale: a freed-block-size population that is highly diverse
    /// relative to the sizes currently in active use is a real signal that
    /// freed blocks are unlikely to be reused as-is by a typical allocator,
    /// i.e. external fragmentation risk. We approximate this as the
    /// fraction of freed block sizes that do not match the size of any
    /// currently-active allocation.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.freed_block_sizes.is_empty() {
            return 0.0;
        }

        let active_sizes: std::collections::HashSet<usize> = self
            .active_allocations
            .values()
            .map(|info| info.size)
            .collect();

        let unmatched = self
            .freed_block_sizes
            .iter()
            .filter(|size| !active_sizes.contains(size))
            .count();

        unmatched as f64 / self.freed_block_sizes.len() as f64
    }
}

impl Default for MemoryPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPatternAnalyzer {
    /// Create a new pattern analyzer
    pub fn new() -> Self {
        Self {
            usage_patterns: VecDeque::new(),
            pattern_detectors: Vec::new(),
            anomaly_detectors: Vec::new(),
        }
    }

    /// Start pattern analysis
    pub fn start_analysis(&mut self) -> Result<()> {
        // Initialize pattern detection algorithms
        self.pattern_detectors
            .push(Box::new(TrendPatternDetector::new()));
        self.pattern_detectors
            .push(Box::new(PeriodicPatternDetector::new()));

        self.anomaly_detectors
            .push(Box::new(SpikeAnomalyDetector::new()));
        self.anomaly_detectors
            .push(Box::new(LeakAnomalyDetector::new()));

        Ok(())
    }

    /// Stop pattern analysis
    pub fn stop_analysis(&mut self) -> Result<()> {
        Ok(())
    }

    /// Add a memory usage snapshot
    pub fn add_snapshot(&mut self, snapshot: MemoryUsageSnapshot) {
        self.usage_patterns.push_back(snapshot);

        // Maintain reasonable history size
        if self.usage_patterns.len() > 1000 {
            self.usage_patterns.pop_front();
        }
    }

    /// Detect all patterns in usage data
    pub fn detect_all_patterns(&self) -> Result<Vec<MemoryPattern>> {
        let mut all_patterns = Vec::new();

        for detector in &self.pattern_detectors {
            let patterns = detector.detect_patterns(&self.usage_patterns)?;
            all_patterns.extend(patterns);
        }

        Ok(all_patterns)
    }

    /// Detect all anomalies in usage data
    pub fn detect_all_anomalies(&self) -> Result<Vec<MemoryAnomaly>> {
        let mut all_anomalies = Vec::new();

        for detector in &self.anomaly_detectors {
            let anomalies = detector.detect_anomalies(&self.usage_patterns)?;
            all_anomalies.extend(anomalies);
        }

        Ok(all_anomalies)
    }
}

impl Default for MemoryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new() -> Self {
        let mut optimizer = Self {
            strategies: Vec::new(),
            // Real metrics require a live `AllocationTracker`; at
            // construction time there is no data yet, so this starts at an
            // honest "unknown" baseline and is overwritten with real
            // measurements by `MemoryLeakDetector::generate_optimization_report`.
            performance_metrics: PerformanceMetrics {
                memory_efficiency: 1.0,
                allocation_efficiency: 1.0,
                cache_hit_ratio: None,
                gc_overhead: None,
            },
        };

        // Initialize optimization strategies
        optimizer.strategies.push(Box::new(PoolingStrategy::new()));
        optimizer.strategies.push(Box::new(InPlaceStrategy::new()));
        optimizer.strategies.push(Box::new(CacheStrategy::new()));

        optimizer
    }

    /// Generate optimization recommendations
    pub fn generate_recommendations(
        &self,
        leak_results: &[MemoryLeakResult],
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        for strategy in &self.strategies {
            for leak_result in leak_results {
                let strategy_recommendations = strategy.recommend(leak_result, &VecDeque::new())?;
                recommendations.extend(strategy_recommendations);
            }
        }

        // Sort by priority and remove duplicates
        recommendations.sort_by(|a, b| match (a.priority.clone(), b.priority.clone()) {
            (RecommendationPriority::Critical, _) => std::cmp::Ordering::Less,
            (_, RecommendationPriority::Critical) => std::cmp::Ordering::Greater,
            (RecommendationPriority::High, _) => std::cmp::Ordering::Less,
            (_, RecommendationPriority::High) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        });

        recommendations.dedup_by(|a, b| a.description == b.description);

        Ok(recommendations)
    }
}

// Default detector implementations

/// Growth-based leak detector
#[derive(Debug)]
pub struct GrowthBasedLeakDetector {
    growth_threshold: f64,
}

impl Default for GrowthBasedLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl GrowthBasedLeakDetector {
    pub fn new() -> Self {
        Self {
            growth_threshold: 1.5, // 50% growth threshold
        }
    }
}

impl LeakDetector for GrowthBasedLeakDetector {
    fn detect_leaks(
        &self,
        _allocation_history: &VecDeque<AllocationEvent>,
        usage_snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<MemoryLeakResult> {
        if usage_snapshots.len() < 2 {
            return Ok(MemoryLeakResult {
                leak_detected: false,
                severity: 0.0,
                confidence: 0.0,
                leaked_memory_bytes: 0,
                leak_sources: vec![],
                growth_analysis: MemoryGrowthAnalysis {
                    growth_trend: GrowthTrend::Stable,
                    growth_rate: 0.0,
                    projected_usage: vec![],
                    pattern_type: GrowthPattern::Normal,
                },
                recommendations: vec![],
                detailed_analysis: "Insufficient data for analysis".to_string(),
            });
        }

        let first = &usage_snapshots[0];
        let last = &usage_snapshots[usage_snapshots.len() - 1];

        let growth_ratio = last.total_memory as f64 / first.total_memory as f64;
        let leak_detected = growth_ratio > self.growth_threshold;

        Ok(MemoryLeakResult {
            leak_detected,
            severity: if leak_detected {
                (growth_ratio - 1.0).min(1.0)
            } else {
                0.0
            },
            confidence: 0.8,
            leaked_memory_bytes: if leak_detected {
                last.total_memory - first.total_memory
            } else {
                0
            },
            leak_sources: vec![],
            growth_analysis: MemoryGrowthAnalysis {
                growth_trend: if growth_ratio > 2.0 {
                    GrowthTrend::Exponential
                } else if growth_ratio > 1.1 {
                    GrowthTrend::Linear
                } else {
                    GrowthTrend::Stable
                },
                growth_rate: (last.total_memory as f64 - first.total_memory as f64)
                    / (last.timestamp - first.timestamp) as f64,
                projected_usage: vec![],
                pattern_type: if leak_detected {
                    GrowthPattern::Leak
                } else {
                    GrowthPattern::Normal
                },
            },
            recommendations: if leak_detected {
                vec![
                    "Investigate rapidly growing memory allocations".to_string(),
                    "Check for unreleased resources".to_string(),
                    "Consider implementing memory pooling".to_string(),
                ]
            } else {
                vec![]
            },
            detailed_analysis: format!(
                "Memory growth analysis: {:.2}x growth detected over {} _snapshots",
                growth_ratio,
                usage_snapshots.len()
            ),
        })
    }

    fn name(&self) -> &str {
        "growth_based"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert(
            "growth_threshold".to_string(),
            self.growth_threshold.to_string(),
        );
        config
    }
}

/// Pattern-based leak detector
#[derive(Debug)]
pub struct PatternBasedLeakDetector;

impl Default for PatternBasedLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternBasedLeakDetector {
    pub fn new() -> Self {
        Self
    }
}

impl LeakDetector for PatternBasedLeakDetector {
    /// Detects a "monotonic-with-sawtooth" leak signature: memory usage may
    /// dip repeatedly (allocate/free cycles), but if the *floor* of those
    /// dips (local minima / troughs) keeps rising across the series, that
    /// is a real signal that each cycle is failing to fully release memory.
    /// A purely monotonic series (no dips at all) is treated as the
    /// degenerate one-trough-per-endpoint case.
    fn detect_leaks(
        &self,
        _allocation_history: &VecDeque<AllocationEvent>,
        usage_snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<MemoryLeakResult> {
        if usage_snapshots.len() < 4 {
            return Ok(insufficient_data_result(
                "Insufficient data for pattern analysis (need >= 4 snapshots)",
            ));
        }

        let values: Vec<f64> = usage_snapshots
            .iter()
            .map(|s| s.total_memory as f64)
            .collect();
        let (Some(&first_value), Some(&last_value)) = (values.first(), values.last()) else {
            return Ok(insufficient_data_result("Empty snapshot series"));
        };

        // Local minima ("troughs"): points at or below both neighbors.
        let mut troughs = Vec::new();
        for i in 1..values.len() - 1 {
            if values[i] <= values[i - 1] && values[i] <= values[i + 1] {
                troughs.push(values[i]);
            }
        }
        if troughs.is_empty() {
            // No dips observed: fall back to endpoint-only comparison so a
            // purely monotonic series is still evaluated.
            troughs.push(first_value);
            troughs.push(last_value);
        }

        let rising_steps = troughs.windows(2).filter(|w| w[1] > w[0]).count();
        let total_steps = troughs.len().saturating_sub(1).max(1);
        let rising_fraction = rising_steps as f64 / total_steps as f64;

        let overall_growth = last_value - first_value;
        let relative_growth = if first_value > 0.0 {
            overall_growth / first_value
        } else {
            0.0
        };

        // Sawtooth-with-rising-floor: most trough-to-trough steps increase
        // and the series shows real net growth (not just noise).
        let leak_detected = rising_fraction >= 0.75 && relative_growth > 0.05;
        let confidence = rising_fraction.clamp(0.0, 1.0);
        let severity = relative_growth.clamp(0.0, 1.0);

        let (Some(front_snapshot), Some(back_snapshot)) =
            (usage_snapshots.front(), usage_snapshots.back())
        else {
            return Ok(insufficient_data_result("Empty snapshot series"));
        };
        let first_ts = front_snapshot.timestamp as f64;
        let last_ts = back_snapshot.timestamp as f64;
        let duration = (last_ts - first_ts).max(1.0);

        Ok(MemoryLeakResult {
            leak_detected,
            severity,
            confidence,
            leaked_memory_bytes: if leak_detected {
                overall_growth.max(0.0) as usize
            } else {
                0
            },
            leak_sources: vec![],
            growth_analysis: MemoryGrowthAnalysis {
                growth_trend: if leak_detected {
                    GrowthTrend::Irregular
                } else {
                    GrowthTrend::Stable
                },
                growth_rate: overall_growth / duration,
                projected_usage: vec![],
                pattern_type: if leak_detected {
                    GrowthPattern::Leak
                } else {
                    GrowthPattern::Normal
                },
            },
            recommendations: if leak_detected {
                vec![
                    "Sawtooth memory floor is rising across allocation/free cycles: check for incomplete deallocation".to_string(),
                ]
            } else {
                vec![]
            },
            detailed_analysis: format!(
                "Pattern analysis: {rising_steps}/{total_steps} trough-to-trough steps rising ({:.0}%), relative growth {:.1}%",
                rising_fraction * 100.0,
                relative_growth * 100.0
            ),
        })
    }

    fn name(&self) -> &str {
        "pattern_based"
    }

    fn config(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Statistical leak detector
#[derive(Debug)]
pub struct StatisticalLeakDetector;

impl Default for StatisticalLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalLeakDetector {
    pub fn new() -> Self {
        Self
    }
}

impl LeakDetector for StatisticalLeakDetector {
    /// Fits an ordinary-least-squares trend line to the real snapshot
    /// series (memory vs. elapsed time) and tests whether the slope is
    /// significantly positive via a t-test (normal approximation), rather
    /// than returning a constant "no leak".
    fn detect_leaks(
        &self,
        _allocation_history: &VecDeque<AllocationEvent>,
        usage_snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<MemoryLeakResult> {
        if usage_snapshots.len() < 3 {
            return Ok(insufficient_data_result(
                "Insufficient data for statistical analysis (need >= 3 snapshots)",
            ));
        }

        let Some(front) = usage_snapshots.front() else {
            return Ok(insufficient_data_result("Empty snapshot series"));
        };
        let t0 = front.timestamp as f64;
        let points: Vec<(f64, f64)> = usage_snapshots
            .iter()
            .map(|s| (s.timestamp as f64 - t0, s.total_memory as f64))
            .collect();

        let Some((slope, intercept, se_slope)) = ols_fit(&points) else {
            return Ok(insufficient_data_result(
                "OLS fit degenerate: no variance in snapshot timestamps",
            ));
        };

        // See the identical rationale in `advanced_memory_leak_detector`:
        // a zero standard error is a perfect (noiseless) fit, which is
        // maximal evidence of a real trend when the slope is nonzero, not
        // "no significance".
        let (t_stat, p_value) = if se_slope > f64::EPSILON {
            let t = slope / se_slope;
            (t, two_tailed_p_value(t))
        } else if slope.abs() > f64::EPSILON {
            (f64::INFINITY, 0.0)
        } else {
            (0.0, 1.0)
        };
        let significant = p_value < 0.05 && slope > 0.0;

        let (Some(&(_, first_point_value)), Some(&(_, last_point_value))) =
            (points.first(), points.last())
        else {
            return Ok(insufficient_data_result("Empty point series after OLS fit"));
        };
        let first_value = first_point_value.max(1.0);
        let last_value = last_point_value;
        let relative_growth = (last_value - first_value) / first_value;

        Ok(MemoryLeakResult {
            leak_detected: significant,
            severity: relative_growth.clamp(0.0, 1.0),
            confidence: (1.0 - p_value).clamp(0.0, 1.0),
            leaked_memory_bytes: if significant {
                (last_value - first_value).max(0.0) as usize
            } else {
                0
            },
            leak_sources: vec![],
            growth_analysis: MemoryGrowthAnalysis {
                growth_trend: if significant {
                    GrowthTrend::Linear
                } else {
                    GrowthTrend::Stable
                },
                growth_rate: slope,
                projected_usage: vec![],
                pattern_type: if significant {
                    GrowthPattern::Leak
                } else {
                    GrowthPattern::Normal
                },
            },
            recommendations: if significant {
                vec![format!(
                    "OLS slope {slope:.3} bytes/sec is statistically significant (p={p_value:.4}); investigate sustained growth"
                )]
            } else {
                vec![]
            },
            detailed_analysis: format!(
                "OLS regression: slope={slope:.4} bytes/sec, intercept={intercept:.2}, t={t_stat:.3}, p={p_value:.4} (n={})",
                usage_snapshots.len()
            ),
        })
    }

    fn name(&self) -> &str {
        "statistical"
    }

    fn config(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Shared "no leak, insufficient data" result used by detectors when the
/// snapshot series is too short for a statistically meaningful judgement.
fn insufficient_data_result(reason: &str) -> MemoryLeakResult {
    MemoryLeakResult {
        leak_detected: false,
        severity: 0.0,
        confidence: 0.0,
        leaked_memory_bytes: 0,
        leak_sources: vec![],
        growth_analysis: MemoryGrowthAnalysis {
            growth_trend: GrowthTrend::Stable,
            growth_rate: 0.0,
            projected_usage: vec![],
            pattern_type: GrowthPattern::Normal,
        },
        recommendations: vec![],
        detailed_analysis: reason.to_string(),
    }
}

/// Ordinary least-squares fit of `y` on `x`, returning
/// `(slope, intercept, standard_error_of_slope)`. Returns `None` when there
/// are fewer than 3 points or `x` has no variance (degenerate fit).
pub(crate) fn ols_fit(points: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let n_f = n as f64;
    let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / n_f;
    let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / n_f;

    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    for (x, y) in points {
        ss_xx += (x - mean_x).powi(2);
        ss_xy += (x - mean_x) * (y - mean_y);
    }
    if ss_xx <= f64::EPSILON {
        return None;
    }
    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    let ss_res: f64 = points
        .iter()
        .map(|(x, y)| {
            let predicted = intercept + slope * x;
            (y - predicted).powi(2)
        })
        .sum();

    let dof = n_f - 2.0;
    if dof <= 0.0 {
        return None;
    }
    let residual_variance = ss_res / dof;
    let se_slope = (residual_variance / ss_xx).sqrt();

    Some((slope, intercept, se_slope))
}

/// Abramowitz & Stegun 7.1.26 erf approximation (max abs error ~1.5e-7).
pub(crate) fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Two-tailed p-value for a t-statistic, approximated via the standard
/// normal distribution. This is exact in the large-sample limit and a
/// documented approximation for small samples -- never a fabricated
/// constant p-value.
pub(crate) fn two_tailed_p_value(t_stat: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(t_stat.abs()))
}

// Pattern detector implementations

/// Trend pattern detector
#[derive(Debug)]
pub struct TrendPatternDetector;

impl Default for TrendPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendPatternDetector {
    pub fn new() -> Self {
        Self
    }
}

impl PatternDetector for TrendPatternDetector {
    fn detect_patterns(
        &self,
        _snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<MemoryPattern>> {
        Ok(vec![MemoryPattern {
            pattern_type: "trend".to_string(),
            confidence: 0.8,
            description: "Memory usage trend analysis".to_string(),
            metrics: HashMap::new(),
        }])
    }

    fn name(&self) -> &str {
        "trend"
    }
}

/// Periodic pattern detector
#[derive(Debug)]
pub struct PeriodicPatternDetector;

impl Default for PeriodicPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PeriodicPatternDetector {
    pub fn new() -> Self {
        Self
    }
}

impl PatternDetector for PeriodicPatternDetector {
    fn detect_patterns(
        &self,
        _snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<MemoryPattern>> {
        Ok(vec![])
    }

    fn name(&self) -> &str {
        "periodic"
    }
}

// Anomaly detector implementations

/// Spike anomaly detector
#[derive(Debug)]
pub struct SpikeAnomalyDetector;

impl Default for SpikeAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl SpikeAnomalyDetector {
    pub fn new() -> Self {
        Self
    }
}

impl AnomalyDetector for SpikeAnomalyDetector {
    fn detect_anomalies(
        &self,
        _snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<MemoryAnomaly>> {
        Ok(vec![])
    }

    fn name(&self) -> &str {
        "spike"
    }
}

/// Leak anomaly detector
#[derive(Debug)]
pub struct LeakAnomalyDetector;

impl Default for LeakAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LeakAnomalyDetector {
    pub fn new() -> Self {
        Self
    }
}

impl AnomalyDetector for LeakAnomalyDetector {
    fn detect_anomalies(
        &self,
        _snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<MemoryAnomaly>> {
        Ok(vec![])
    }

    fn name(&self) -> &str {
        "leak"
    }
}

// Optimization strategy implementations

/// Memory pooling strategy
#[derive(Debug)]
pub struct PoolingStrategy;

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolingStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationStrategy for PoolingStrategy {
    fn recommend(
        &self,
        leak_result: &MemoryLeakResult,
        _history: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        if leak_result.leak_detected && leak_result.severity > 0.5 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::MemoryPooling,
                priority: RecommendationPriority::High,
                description: "Implement memory pooling to reduce allocation overhead".to_string(),
                expected_impact: "50-80% reduction in allocation overhead".to_string(),
                complexity: ImplementationComplexity::Medium,
                code_examples: Some(vec![
                    "// Use memory pools for frequent allocations".to_string(),
                    "let pool = MemoryPool::new(chunk_size);".to_string(),
                    "let memory = pool.allocate(size);".to_string(),
                ]),
            });
        }

        Ok(recommendations)
    }

    fn name(&self) -> &str {
        "pooling"
    }
}

/// In-place operations strategy
#[derive(Debug)]
pub struct InPlaceStrategy;

impl Default for InPlaceStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl InPlaceStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationStrategy for InPlaceStrategy {
    fn recommend(
        &self,
        leak_result: &MemoryLeakResult,
        _history: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        if leak_result.leaked_memory_bytes > 1024 * 1024 {
            // > 1MB
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::InPlaceOperations,
                priority: RecommendationPriority::Medium,
                description: "Use in-place operations to reduce memory allocations".to_string(),
                expected_impact: "30-50% reduction in temporary allocations".to_string(),
                complexity: ImplementationComplexity::Easy,
                code_examples: Some(vec![
                    "// Use in-place operations".to_string(),
                    "array.mapv_inplace(|x| x * 2.0);".to_string(),
                    "// instead of: let new_array = array.mapv(|x| x * 2.0);".to_string(),
                ]),
            });
        }

        Ok(recommendations)
    }

    fn name(&self) -> &str {
        "inplace"
    }
}

/// Cache optimization strategy
#[derive(Debug)]
pub struct CacheStrategy;

impl Default for CacheStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationStrategy for CacheStrategy {
    fn recommend(
        &self,
        _leak_result: &MemoryLeakResult,
        _history: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<Vec<OptimizationRecommendation>> {
        Ok(vec![OptimizationRecommendation {
            recommendation_type: RecommendationType::CacheOptimization,
            priority: RecommendationPriority::Low,
            description: "Optimize cache usage patterns for better memory locality".to_string(),
            expected_impact: "10-20% improvement in memory access patterns".to_string(),
            complexity: ImplementationComplexity::Medium,
            code_examples: Some(vec![
                "// Improve cache locality".to_string(),
                "// Process data in cache-friendly order".to_string(),
            ]),
        }])
    }

    fn name(&self) -> &str {
        "cache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_leak_detector_creation() {
        let config = MemoryDetectionConfig::default();
        let detector = MemoryLeakDetector::new(config);
        assert_eq!(detector.detectors.len(), 4); // 3 basic + 1 real-time monitoring detector
    }

    #[test]
    fn test_allocation_tracking() {
        let mut tracker = AllocationTracker::new();
        let event = AllocationEvent {
            timestamp: 12345,
            allocation_id: 1,
            size: 1024,
            allocation_type: AllocationType::Parameter,
            stack_trace: None,
            optimizer_context: None,
        };

        tracker.record_allocation(event).expect("unwrap failed");
        assert_eq!(tracker.total_allocations.load(Ordering::Relaxed), 1);
        assert_eq!(tracker.current_memory_usage.load(Ordering::Relaxed), 1024);

        tracker.record_deallocation(1).expect("unwrap failed");
        assert_eq!(tracker.total_deallocations.load(Ordering::Relaxed), 1);
        assert_eq!(tracker.current_memory_usage.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_growth_based_leak_detector() {
        let detector = GrowthBasedLeakDetector::new();

        let mut snapshots = VecDeque::new();
        snapshots.push_back(MemoryUsageSnapshot {
            timestamp: 0,
            total_memory: 1000,
            heap_memory: 800,
            stack_memory: 200,
            memory_by_type: HashMap::new(),
            growth_rate: 0.0,
            fragmentation_level: 0.1,
        });
        snapshots.push_back(MemoryUsageSnapshot {
            timestamp: 1000,
            total_memory: 2000, // 2x growth
            heap_memory: 1600,
            stack_memory: 400,
            memory_by_type: HashMap::new(),
            growth_rate: 1.0,
            fragmentation_level: 0.1,
        });

        let result = detector
            .detect_leaks(&VecDeque::new(), &snapshots)
            .expect("unwrap failed");
        assert!(result.leak_detected);
        assert!(result.severity > 0.0);
        assert_eq!(result.leaked_memory_bytes, 1000);
    }
}
