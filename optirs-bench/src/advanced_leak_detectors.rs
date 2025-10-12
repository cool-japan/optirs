// Advanced memory leak detection algorithms
//
// This module implements sophisticated memory leak detection algorithms including
// reference counting analysis, cycle detection, real-time monitoring, and
// pattern-based leak identification.

use super::memory_leak_detector::{
    AllocationEvent, AllocationType, GrowthPattern, GrowthTrend, LeakDetector, LeakSource,
    MemoryGrowthAnalysis, MemoryLeakResult, MemoryUsageSnapshot,
};
use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Reference counting leak detector
///
/// Detects memory leaks by analyzing reference count patterns and identifying
/// potential circular references or unreleased references.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ReferenceCountingDetector {
    /// Configuration for reference counting analysis
    config: ReferenceCountingConfig,
    /// Reference tracking state
    reference_tracker: Arc<RwLock<ReferenceTracker>>,
    /// Cycle detection engine
    cycle_detector: CycleDetector,
}

/// Configuration for reference counting detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceCountingConfig {
    /// Minimum reference count threshold for suspicion
    pub min_suspicious_refcount: usize,
    /// Maximum reference count before flagging
    pub max_normal_refcount: usize,
    /// Cycle detection depth
    pub cycle_detection_depth: usize,
    /// Reference age threshold (seconds)
    pub reference_age_threshold: u64,
    /// Enable strong reference analysis
    pub enable_strong_ref_analysis: bool,
    /// Enable weak reference analysis
    pub enable_weak_ref_analysis: bool,
}

impl Default for ReferenceCountingConfig {
    fn default() -> Self {
        Self {
            min_suspicious_refcount: 5,
            max_normal_refcount: 100,
            cycle_detection_depth: 10,
            reference_age_threshold: 300, // 5 minutes
            enable_strong_ref_analysis: true,
            enable_weak_ref_analysis: true,
        }
    }
}

/// Reference tracking state
#[derive(Debug)]
pub struct ReferenceTracker {
    /// Active references by allocation ID
    active_references: HashMap<usize, ReferenceInfo>,
    /// Reference graph for cycle detection
    reference_graph: HashMap<usize, HashSet<usize>>,
    /// Reference history
    reference_history: VecDeque<ReferenceEvent>,
    /// Suspected leaks
    suspectedleaks: Vec<SuspectedLeak>,
}

/// Information about a reference
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    /// Allocation ID
    pub allocation_id: usize,
    /// Current reference count
    pub reference_count: usize,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last accessed timestamp
    pub last_accessed: Instant,
    /// Reference type
    pub reference_type: ReferenceType,
    /// Source location
    pub source_location: Option<String>,
    /// References to other allocations
    pub references_to: HashSet<usize>,
    /// Referenced by other allocations
    pub referenced_by: HashSet<usize>,
}

/// Types of references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    /// Strong reference (prevents deallocation)
    Strong,
    /// Weak reference (doesn't prevent deallocation)
    Weak,
    /// Shared reference (shared ownership)
    Shared,
    /// Unique reference (exclusive ownership)
    Unique,
}

/// Reference event for tracking
#[derive(Debug, Clone)]
pub struct ReferenceEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Allocation ID
    pub allocation_id: usize,
    /// Event type
    pub event_type: ReferenceEventType,
    /// Reference count after event
    pub reference_count: usize,
}

/// Types of reference events
#[derive(Debug, Clone)]
pub enum ReferenceEventType {
    /// Reference created
    Created,
    /// Reference incremented
    Incremented,
    /// Reference decremented
    Decremented,
    /// Reference destroyed
    Destroyed,
    /// Reference accessed
    Accessed,
}

/// Suspected memory leak
#[derive(Debug, Clone)]
pub struct SuspectedLeak {
    /// Allocation ID
    pub allocation_id: usize,
    /// Leak confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Leak type
    pub leak_type: LeakType,
    /// Evidence for the leak
    pub evidence: Vec<String>,
    /// Detection timestamp
    pub detected_at: Instant,
}

/// Types of memory leaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakType {
    /// Circular reference
    CircularReference,
    /// Dangling reference
    DanglingReference,
    /// Unreleased reference
    UnreleasedReference,
    /// Reference count overflow
    ReferenceCountOverflow,
    /// Stale reference
    StaleReference,
}

impl ReferenceCountingDetector {
    /// Create a new reference counting detector
    pub fn new(config: ReferenceCountingConfig) -> Self {
        Self {
            config,
            reference_tracker: Arc::new(RwLock::new(ReferenceTracker::new())),
            cycle_detector: CycleDetector::new(),
        }
    }

    /// Track a reference operation
    pub fn track_reference_operation(
        &self,
        allocation_id: usize,
        event_type: ReferenceEventType,
        reference_count: usize,
    ) -> Result<()> {
        let mut tracker = self.reference_tracker.write().map_err(|_| {
            OptimError::InvalidState("Failed to acquire reference tracker lock".to_string())
        })?;

        let now = Instant::now();

        // Update reference info
        let ref_info = tracker
            .active_references
            .entry(allocation_id)
            .or_insert_with(|| ReferenceInfo {
                allocation_id,
                reference_count: 0,
                created_at: now,
                last_accessed: now,
                reference_type: ReferenceType::Strong,
                source_location: None,
                references_to: HashSet::new(),
                referenced_by: HashSet::new(),
            });

        ref_info.reference_count = reference_count;
        ref_info.last_accessed = now;

        // Record the event
        let event = ReferenceEvent {
            timestamp: now,
            allocation_id,
            event_type,
            reference_count,
        };

        tracker.reference_history.push_back(event);

        // Limit history size
        while tracker.reference_history.len() > 10000 {
            tracker.reference_history.pop_front();
        }

        // Analyze for potential leaks
        self.analyze_reference_patterns(&mut tracker)?;

        Ok(())
    }

    /// Analyze reference patterns for potential leaks
    fn analyze_reference_patterns(&self, tracker: &mut ReferenceTracker) -> Result<()> {
        let now = Instant::now();

        // Check for suspicious reference counts
        for (allocation_id, ref_info) in &tracker.active_references {
            let mut evidence = Vec::new();
            let mut confidence = 0.0;
            let mut leak_type = LeakType::UnreleasedReference;

            // Check reference count thresholds
            if ref_info.reference_count > self.config.max_normal_refcount {
                evidence.push(format!(
                    "Reference count {} exceeds normal threshold {}",
                    ref_info.reference_count, self.config.max_normal_refcount
                ));
                confidence += 0.3;
                leak_type = LeakType::ReferenceCountOverflow;
            }

            // Check reference age
            let age = now.duration_since(ref_info.created_at).as_secs();
            if age > self.config.reference_age_threshold {
                evidence.push(format!(
                    "Reference age {} seconds exceeds threshold {}",
                    age, self.config.reference_age_threshold
                ));
                confidence += 0.2;
            }

            // Check for stale references (not accessed recently)
            let last_access_age = now.duration_since(ref_info.last_accessed).as_secs();
            if last_access_age > self.config.reference_age_threshold / 2 {
                evidence.push(format!(
                    "Reference not accessed for {} seconds",
                    last_access_age
                ));
                confidence += 0.15;
                leak_type = LeakType::StaleReference;
            }

            // Check for circular references
            if self.has_circular_reference(*allocation_id, &tracker.reference_graph) {
                evidence.push("Circular reference detected".to_string());
                confidence += 0.4;
                leak_type = LeakType::CircularReference;
            }

            // Add suspected leak if confidence is high enough
            if confidence > 0.5 && !evidence.is_empty() {
                let suspected_leak = SuspectedLeak {
                    allocation_id: *allocation_id,
                    confidence,
                    leak_type,
                    evidence,
                    detected_at: now,
                };

                tracker.suspectedleaks.push(suspected_leak);
            }
        }

        // Clean up old suspected leaks
        tracker.suspectedleaks.retain(|leak| {
            now.duration_since(leak.detected_at).as_secs() < 3600 // Keep for 1 hour
        });

        Ok(())
    }

    /// Check for circular references using DFS
    fn has_circular_reference(
        &self,
        allocation_id: usize,
        reference_graph: &HashMap<usize, HashSet<usize>>,
    ) -> bool {
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();

        Self::dfs_cycle_detection(
            allocation_id,
            reference_graph,
            &mut visited,
            &mut recursion_stack,
        )
    }

    /// Depth-first search for cycle detection
    fn dfs_cycle_detection(
        node: usize,
        graph: &HashMap<usize, HashSet<usize>>,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> bool {
        visited.insert(node);
        recursion_stack.insert(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    if Self::dfs_cycle_detection(neighbor, graph, visited, recursion_stack) {
                        return true;
                    }
                } else if recursion_stack.contains(&neighbor) {
                    // Found a back edge (cycle)
                    return true;
                }
            }
        }

        recursion_stack.remove(&node);
        false
    }
}

impl LeakDetector for ReferenceCountingDetector {
    fn detect_leaks(
        &self,
        allocation_history: &VecDeque<AllocationEvent>,
        usage_snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<MemoryLeakResult> {
        let tracker = self.reference_tracker.read().map_err(|_| {
            OptimError::InvalidState("Failed to acquire reference tracker lock".to_string())
        })?;

        let mut leak_sources = Vec::new();
        let mut total_leaked_bytes = 0;
        let mut max_confidence = 0.0;

        // Analyze suspected leaks
        for suspected_leak in &tracker.suspectedleaks {
            if let Some(ref_info) = tracker.active_references.get(&suspected_leak.allocation_id) {
                // Estimate leak size from allocation _history
                let leak_size = allocation_history
                    .iter()
                    .find(|event| event.allocation_id == suspected_leak.allocation_id)
                    .map(|event| event.size)
                    .unwrap_or(0);

                let leak_source = LeakSource {
                    source_type: AllocationType::OptimizerState, // Default type
                    location: ref_info.source_location.clone(),
                    leak_size,
                    probability: suspected_leak.confidence,
                    stack_trace: None,
                };

                leak_sources.push(leak_source);
                total_leaked_bytes += leak_size;
                max_confidence = max_confidence.max(suspected_leak.confidence);
            }
        }

        // Perform memory growth analysis
        let growth_analysis = self.analyze_memory_growth(usage_snapshots)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&tracker.suspectedleaks);

        // Create detailed analysis
        let detailed_analysis = format!(
            "Reference Counting Analysis:\n\
             - Active References: {}\n\
             - Suspected Leaks: {}\n\
             - Circular References Detected: {}\n\
             - Average Reference Count: {:.2}\n\
             - Maximum Reference Count: {}",
            tracker.active_references.len(),
            tracker.suspectedleaks.len(),
            tracker
                .suspectedleaks
                .iter()
                .filter(|l| matches!(l.leak_type, LeakType::CircularReference))
                .count(),
            tracker
                .active_references
                .values()
                .map(|r| r.reference_count)
                .sum::<usize>() as f64
                / tracker.active_references.len().max(1) as f64,
            tracker
                .active_references
                .values()
                .map(|r| r.reference_count)
                .max()
                .unwrap_or(0)
        );

        Ok(MemoryLeakResult {
            leak_detected: !leak_sources.is_empty(),
            severity: max_confidence,
            confidence: max_confidence,
            leaked_memory_bytes: total_leaked_bytes,
            leak_sources,
            growth_analysis,
            recommendations,
            detailed_analysis,
        })
    }

    fn name(&self) -> &str {
        "ReferenceCountingDetector"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert(
            "min_suspicious_refcount".to_string(),
            self.config.min_suspicious_refcount.to_string(),
        );
        config.insert(
            "max_normal_refcount".to_string(),
            self.config.max_normal_refcount.to_string(),
        );
        config.insert(
            "cycle_detection_depth".to_string(),
            self.config.cycle_detection_depth.to_string(),
        );
        config.insert(
            "reference_age_threshold".to_string(),
            self.config.reference_age_threshold.to_string(),
        );
        config
    }
}

impl ReferenceCountingDetector {
    /// Analyze memory growth patterns
    fn analyze_memory_growth(
        &self,
        snapshots: &VecDeque<MemoryUsageSnapshot>,
    ) -> Result<MemoryGrowthAnalysis> {
        if snapshots.len() < 2 {
            return Ok(MemoryGrowthAnalysis {
                growth_trend: GrowthTrend::Stable,
                growth_rate: 0.0,
                projected_usage: Vec::new(),
                pattern_type: GrowthPattern::Normal,
            });
        }

        let memory_values: Vec<f64> = snapshots.iter().map(|s| s.total_memory as f64).collect();

        // Calculate growth rate using linear regression
        let n = memory_values.len() as f64;
        let sum_x = (0..memory_values.len()).sum::<usize>() as f64;
        let sum_y = memory_values.iter().sum::<f64>();
        let sum_xy = memory_values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let sum_x2 = (0..memory_values.len())
            .map(|i| (i * i) as f64)
            .sum::<f64>();

        let growth_rate = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Determine growth trend
        let growth_trend = if growth_rate.abs() < 1000.0 {
            GrowthTrend::Stable
        } else if growth_rate > 0.0 {
            if growth_rate > 100000.0 {
                GrowthTrend::Exponential
            } else {
                GrowthTrend::Linear
            }
        } else {
            GrowthTrend::Irregular
        };

        // Determine pattern type
        let pattern_type = if growth_rate > 50000.0 {
            GrowthPattern::Leak
        } else if memory_values
            .windows(2)
            .any(|w| (w[1] - w[0]).abs() > 10000000.0)
        {
            GrowthPattern::Burst
        } else {
            GrowthPattern::Normal
        };

        // Project future memory usage
        let last_timestamp = snapshots.back().unwrap().timestamp;
        let projected_usage = (1..=10)
            .map(|i| {
                let future_timestamp = last_timestamp + (i * 60); // 1 minute intervals
                let future_memory = memory_values.last().unwrap() + (growth_rate * i as f64);
                (future_timestamp, future_memory.max(0.0) as usize)
            })
            .collect();

        Ok(MemoryGrowthAnalysis {
            growth_trend,
            growth_rate,
            projected_usage,
            pattern_type,
        })
    }

    /// Generate recommendations based on detected issues
    fn generate_recommendations(&self, suspectedleaks: &[SuspectedLeak]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let circular_ref_count = suspectedleaks
            .iter()
            .filter(|l| matches!(l.leak_type, LeakType::CircularReference))
            .count();

        let stale_ref_count = suspectedleaks
            .iter()
            .filter(|l| matches!(l.leak_type, LeakType::StaleReference))
            .count();

        let overflow_count = suspectedleaks
            .iter()
            .filter(|l| matches!(l.leak_type, LeakType::ReferenceCountOverflow))
            .count();

        if circular_ref_count > 0 {
            recommendations.push(format!(
                "Detected {} circular references. Consider using weak references to break cycles.",
                circular_ref_count
            ));
        }

        if stale_ref_count > 0 {
            recommendations.push(format!(
                "Detected {} stale references. Implement automatic cleanup for unused references.",
                stale_ref_count
            ));
        }

        if overflow_count > 0 {
            recommendations.push(format!(
                "Detected {} reference count overflows. Review reference sharing patterns.",
                overflow_count
            ));
        }

        if suspectedleaks.len() > 10 {
            recommendations.push(
                "High number of suspected _leaks detected. Consider implementing reference pooling."
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Reference counting patterns appear healthy.".to_string());
        }

        recommendations
    }
}

/// Cycle detector for identifying circular references
#[derive(Debug)]
#[allow(dead_code)]
pub struct CycleDetector {
    /// Maximum detection depth
    max_depth: usize,
}

/// State for Tarjan's algorithm
struct TarjanState {
    index: usize,
    stack: Vec<usize>,
    indices: HashMap<usize, usize>,
    lowlinks: HashMap<usize, usize>,
    on_stack: HashSet<usize>,
    strongly_connected_components: Vec<Vec<usize>>,
}

impl TarjanState {
    fn new() -> Self {
        Self {
            index: 0,
            stack: Vec::new(),
            indices: HashMap::new(),
            lowlinks: HashMap::new(),
            on_stack: HashSet::new(),
            strongly_connected_components: Vec::new(),
        }
    }
}

impl CycleDetector {
    /// Create a new cycle detector
    pub fn new() -> Self {
        Self { max_depth: 20 }
    }

    /// Detect cycles in reference graph using Tarjan's algorithm
    pub fn detect_cycles(&self, graph: &HashMap<usize, HashSet<usize>>) -> Vec<Vec<usize>> {
        let mut state = TarjanState::new();

        for &node in graph.keys() {
            if !state.indices.contains_key(&node) {
                Self::strongconnect(node, &mut state, graph);
            }
        }

        // Filter to return only cycles (SCCs with more than one node or self-loops)
        state
            .strongly_connected_components
            .into_iter()
            .filter(|scc| {
                scc.len() > 1
                    || (scc.len() == 1
                        && graph
                            .get(&scc[0])
                            .is_some_and(|neighbors| neighbors.contains(&scc[0])))
            })
            .collect()
    }

    /// Tarjan's strongly connected components algorithm
    fn strongconnect(v: usize, state: &mut TarjanState, graph: &HashMap<usize, HashSet<usize>>) {
        state.indices.insert(v, state.index);
        state.lowlinks.insert(v, state.index);
        state.index += 1;
        state.stack.push(v);
        state.on_stack.insert(v);

        if let Some(neighbors) = graph.get(&v) {
            for &w in neighbors {
                if !state.indices.contains_key(&w) {
                    Self::strongconnect(w, state, graph);
                    let v_lowlink = *state.lowlinks.get(&v).unwrap();
                    let w_lowlink = *state.lowlinks.get(&w).unwrap();
                    state.lowlinks.insert(v, v_lowlink.min(w_lowlink));
                } else if state.on_stack.contains(&w) {
                    let v_lowlink = *state.lowlinks.get(&v).unwrap();
                    let w_index = *state.indices.get(&w).unwrap();
                    state.lowlinks.insert(v, v_lowlink.min(w_index));
                }
            }
        }

        if state.lowlinks.get(&v) == state.indices.get(&v) {
            let mut component = Vec::new();
            loop {
                let w = state.stack.pop().unwrap();
                state.on_stack.remove(&w);
                component.push(w);
                if w == v {
                    break;
                }
            }
            state.strongly_connected_components.push(component);
        }
    }
}

impl Default for ReferenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceTracker {
    /// Create a new reference tracker
    pub fn new() -> Self {
        Self {
            active_references: HashMap::new(),
            reference_graph: HashMap::new(),
            reference_history: VecDeque::new(),
            suspectedleaks: Vec::new(),
        }
    }
}

impl Default for CycleDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time memory monitor
///
/// Provides continuous monitoring of memory usage patterns and leak detection
/// in real-time with configurable sampling rates and alert thresholds.
#[allow(dead_code)]
pub struct RealTimeMemoryMonitor {
    /// Monitor configuration
    config: RealTimeMonitorConfig,
    /// Current monitoring state
    state: Arc<Mutex<MonitorState>>,
    /// Alert system
    alert_system: AlertSystem,
    /// Is monitoring active
    is_active: Arc<Mutex<bool>>,
}

/// Real-time monitor configuration
#[derive(Debug, Clone)]
pub struct RealTimeMonitorConfig {
    /// Sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    /// Memory threshold for alerts (bytes)
    pub memory_threshold_bytes: usize,
    /// Growth rate threshold (bytes/second)
    pub growth_rate_threshold: f64,
    /// Window size for trend analysis
    pub trend_window_size: usize,
    /// Enable automatic garbage collection hints
    pub enable_gc_hints: bool,
}

impl Default for RealTimeMonitorConfig {
    fn default() -> Self {
        Self {
            sampling_interval_ms: 1000,                // 1 second
            memory_threshold_bytes: 100 * 1024 * 1024, // 100MB
            growth_rate_threshold: 1024.0 * 1024.0,    // 1MB/s
            trend_window_size: 60,                     // 1 minute of samples
            enable_gc_hints: true,
        }
    }
}

/// Monitor state
#[derive(Debug)]
pub struct MonitorState {
    /// Recent memory samples
    memory_samples: VecDeque<MemorySample>,
    /// Current memory usage
    current_memory_usage: usize,
    /// Peak memory usage
    peak_memory_usage: usize,
    /// Memory growth rate
    current_growth_rate: f64,
    /// Alert history
    alert_history: VecDeque<MemoryAlert>,
}

/// Memory sample for monitoring
#[derive(Debug, Clone)]
pub struct MemorySample {
    /// Sample timestamp
    pub timestamp: Instant,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Allocation rate
    pub allocation_rate: f64,
    /// Deallocation rate
    pub deallocation_rate: f64,
}

/// Memory alert
#[derive(Debug, Clone)]
pub struct MemoryAlert {
    /// Alert timestamp
    pub timestamp: Instant,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Severity level
    pub severity: AlertSeverity,
}

/// Types of memory alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    /// Memory threshold exceeded
    MemoryThreshold,
    /// Growth rate exceeded
    GrowthRate,
    /// Potential leak detected
    PotentialLeak,
    /// Memory fragmentation
    Fragmentation,
    /// System memory pressure
    SystemPressure,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Type alias for alert callbacks
type AlertCallback = Box<dyn Fn(&MemoryAlert) + Send + Sync>;

/// Alert system for memory monitoring
pub struct AlertSystem {
    /// Alert callbacks
    alert_callbacks: Vec<AlertCallback>,
}

impl RealTimeMemoryMonitor {
    /// Create a new real-time memory monitor
    pub fn new(config: RealTimeMonitorConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(MonitorState::new())),
            alert_system: AlertSystem::new(),
            is_active: Arc::new(Mutex::new(false)),
        }
    }

    /// Start monitoring
    pub fn start_monitoring(&self) -> Result<()> {
        let mut is_active = self
            .is_active
            .lock()
            .map_err(|_| OptimError::InvalidState("Failed to acquire monitor lock".to_string()))?;

        if *is_active {
            return Ok(()); // Already monitoring
        }

        *is_active = true;

        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let is_active_flag = Arc::clone(&self.is_active);

        thread::spawn(move || {
            let mut _last_sample_time = Instant::now();

            loop {
                // Check if monitoring should continue
                {
                    let active = is_active_flag.lock().unwrap();
                    if !*active {
                        break;
                    }
                }

                // Sleep for sampling interval
                thread::sleep(Duration::from_millis(config.sampling_interval_ms));

                // Take memory sample
                let now = Instant::now();
                let sample = Self::take_memory_sample(now);

                // Update state
                {
                    let mut monitor_state = state.lock().unwrap();
                    monitor_state.add_sample(sample);
                    monitor_state.update_metrics(&config);
                }

                _last_sample_time = now;
            }
        });

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) -> Result<()> {
        let mut is_active = self
            .is_active
            .lock()
            .map_err(|_| OptimError::InvalidState("Failed to acquire monitor lock".to_string()))?;

        *is_active = false;
        Ok(())
    }

    /// Take a memory sample
    fn take_memory_sample(timestamp: Instant) -> MemorySample {
        // In a real implementation, this would use system APIs to get actual memory usage
        // For now, we'll simulate memory sampling
        MemorySample {
            timestamp,
            memory_usage: Self::get_current_memory_usage(),
            allocation_rate: 0.0,   // Would be calculated from real data
            deallocation_rate: 0.0, // Would be calculated from real data
        }
    }

    /// Get current memory usage (simulated)
    fn get_current_memory_usage() -> usize {
        // In a real implementation, this would use:
        // - On Linux: /proc/self/status or mallinfo
        // - On macOS: task_info or malloc_zone_statistics
        // - On Windows: GetProcessMemoryInfo

        // For now, return a simulated value
        64 * 1024 * 1024 // 64MB
    }

    /// Get current monitoring statistics
    pub fn get_statistics(&self) -> Result<MonitoringStatistics> {
        let state = self.state.lock().map_err(|_| {
            OptimError::InvalidState("Failed to acquire monitor state lock".to_string())
        })?;

        Ok(MonitoringStatistics {
            current_memory_usage: state.current_memory_usage,
            peak_memory_usage: state.peak_memory_usage,
            current_growth_rate: state.current_growth_rate,
            sample_count: state.memory_samples.len(),
            recent_alerts: state.alert_history.iter().cloned().collect(),
        })
    }
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct MonitoringStatistics {
    /// Current memory usage
    pub current_memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Current growth rate
    pub current_growth_rate: f64,
    /// Number of samples collected
    pub sample_count: usize,
    /// Recent alerts
    pub recent_alerts: Vec<MemoryAlert>,
}

impl MonitorState {
    /// Create new monitor state
    pub fn new() -> Self {
        Self {
            memory_samples: VecDeque::new(),
            current_memory_usage: 0,
            peak_memory_usage: 0,
            current_growth_rate: 0.0,
            alert_history: VecDeque::new(),
        }
    }

    /// Add a memory sample
    pub fn add_sample(&mut self, sample: MemorySample) {
        self.current_memory_usage = sample.memory_usage;
        self.peak_memory_usage = self.peak_memory_usage.max(sample.memory_usage);

        self.memory_samples.push_back(sample);

        // Limit sample history
        while self.memory_samples.len() > 3600 {
            // 1 hour at 1s intervals
            self.memory_samples.pop_front();
        }
    }

    /// Update monitoring metrics
    pub fn update_metrics(&mut self, config: &RealTimeMonitorConfig) {
        // Calculate growth rate
        if self.memory_samples.len() >= 2 {
            let window_size = config.trend_window_size.min(self.memory_samples.len());
            let recent_samples: Vec<_> =
                self.memory_samples.iter().rev().take(window_size).collect();

            if recent_samples.len() >= 2 {
                let first = recent_samples.last().unwrap();
                let last = recent_samples.first().unwrap();
                let time_diff = last.timestamp.duration_since(first.timestamp).as_secs_f64();

                if time_diff > 0.0 {
                    let memory_diff = last.memory_usage as f64 - first.memory_usage as f64;
                    self.current_growth_rate = memory_diff / time_diff;
                }
            }
        }

        // Check for alerts
        self.check_for_alerts(config);
    }

    /// Check for alert conditions
    fn check_for_alerts(&mut self, config: &RealTimeMonitorConfig) {
        let now = Instant::now();

        // Memory threshold alert
        if self.current_memory_usage > config.memory_threshold_bytes {
            let alert = MemoryAlert {
                timestamp: now,
                alert_type: AlertType::MemoryThreshold,
                message: format!(
                    "Memory usage {} exceeds threshold {}",
                    self.current_memory_usage, config.memory_threshold_bytes
                ),
                severity: AlertSeverity::High,
            };
            self.alert_history.push_back(alert);
        }

        // Growth rate alert
        if self.current_growth_rate > config.growth_rate_threshold {
            let alert = MemoryAlert {
                timestamp: now,
                alert_type: AlertType::GrowthRate,
                message: format!(
                    "Memory growth rate {:.2} bytes/s exceeds threshold {:.2}",
                    self.current_growth_rate, config.growth_rate_threshold
                ),
                severity: AlertSeverity::Medium,
            };
            self.alert_history.push_back(alert);
        }

        // Limit alert history
        while self.alert_history.len() > 1000 {
            self.alert_history.pop_front();
        }
    }
}

impl AlertSystem {
    /// Create new alert system
    pub fn new() -> Self {
        Self {
            alert_callbacks: Vec::new(),
        }
    }

    /// Add alert callback
    pub fn add_callback<F>(&mut self, callback: F)
    where
        F: Fn(&MemoryAlert) + Send + Sync + 'static,
    {
        self.alert_callbacks.push(Box::new(callback));
    }

    /// Trigger alert
    pub fn trigger_alert(&self, alert: &MemoryAlert) {
        for callback in &self.alert_callbacks {
            callback(alert);
        }
    }
}

impl Default for MonitorState {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AlertSystem {
    fn default() -> Self {
        Self::new()
    }
}
