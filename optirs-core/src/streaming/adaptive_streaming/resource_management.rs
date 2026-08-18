// Resource allocation and monitoring for streaming optimization
//
// This module provides comprehensive resource management capabilities including
// dynamic resource allocation, monitoring, budgeting, and optimization for
// streaming optimization workloads.

use super::config::*;
use super::optimizer::{Adaptation, AdaptationPriority, AdaptationType};

use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

mod alerts;
mod optimization;
mod prediction;
mod probe;

#[cfg(test)]
mod regression_tests;

pub use probe::SystemProbe;

/// How often the monitoring thread checks its shutdown flag (R4). Kept short
/// so `stop_monitoring` returns promptly even when `monitoring_frequency` is
/// measured in minutes.
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Recovers a mutex guard even when the lock was poisoned. The values guarded
/// in this module are plain snapshots with no cross-field invariant a panic
/// could leave half-written, so continuing with the recovered value is better
/// than propagating the panic into every monitoring call.
pub(crate) fn lock_recovered<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Current resource usage information
#[derive(Debug, Clone, Serialize)]
pub struct ResourceUsage {
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// Total system memory in MB (R2 fix). Populated from the real system
    /// total at collection time so `memory_usage_mb` can be turned into an
    /// honest percentage instead of assuming a fixed 1024 MB (1 GB) total,
    /// which pinned `memory_percent`/`memory_utilization` above 100% (and
    /// therefore alert severity at `Emergency`) on essentially every real
    /// machine. `0` (the `Default` value) means "unknown"; callers must
    /// treat that as "cannot compute a percentage", not as "0 MB total".
    pub total_memory_mb: usize,
    /// Memory this *process* is using, in MB, or `None` when the process
    /// counters are unavailable.
    ///
    /// `memory_usage_mb`/`total_memory_mb` are system-wide and only meaningful
    /// as a ratio (which is what the memory alert thresholds compare). The
    /// allocation budget, by contrast, is a per-process figure, so comparing
    /// system-wide usage against it reports several hundred percent utilization
    /// on any real machine — the same class of error as R2, one level up. That
    /// comparison was inert only because `start_monitoring` was never called
    /// and `current_usage` stayed all-zero; every budget consumer now reads this
    /// field instead.
    pub process_memory_mb: Option<usize>,
    /// CPU usage percentage (0-100).
    ///
    /// R1: this used to be the hardcoded literal `50.0`. It is now the real
    /// system-wide CPU usage read from a *persistent* `sysinfo::System`,
    /// because CPU usage is a delta between two refreshes: a freshly built
    /// `System` always reports 0. Until two refreshes at least
    /// `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL` apart have happened this field
    /// holds `0.0` and `cpu_usage_percent_valid` is `false`; consumers must
    /// check the flag rather than treating the placeholder as a measurement.
    pub cpu_usage_percent: f64,
    /// Whether `cpu_usage_percent` holds a real measurement.
    pub cpu_usage_percent_valid: bool,
    /// GPU usage percentage (0-100) if applicable. There is no pure-Rust,
    /// vendor-neutral way to read this, so it stays `None` unless an external
    /// probe supplies it.
    pub gpu_usage_percent: Option<f64>,
    /// Network I/O rate in MB/s, or `None` before two interface refreshes have
    /// produced a usable byte delta (R1: this used to be the literal `1.0`).
    pub network_io_mbps: Option<f64>,
    /// Disk I/O rate in MB/s for this process, or `None` before two process
    /// refreshes have produced a usable byte delta (R1: this used to be the
    /// literal `5.0`).
    pub disk_io_mbps: Option<f64>,
    /// Number of active threads
    pub active_threads: usize,
    /// Timestamp of measurement
    #[serde(skip)]
    pub timestamp: Instant,
}

impl ResourceUsage {
    /// Memory this process is using, in MB, when it could be measured.
    pub fn process_memory(&self) -> Option<usize> {
        self.process_memory_mb
    }

    /// Real CPU usage, or `None` when no valid measurement has been taken yet.
    pub fn cpu_usage(&self) -> Option<f64> {
        if self.cpu_usage_percent_valid {
            Some(self.cpu_usage_percent)
        } else {
            None
        }
    }

    /// Real memory-usage percentage (R2), `used / total * 100`. Returns
    /// `None` when `total_memory_mb` is unknown (0) rather than fabricating
    /// a value against a wrong assumed total.
    pub fn memory_usage_percent(&self) -> Option<f64> {
        if self.total_memory_mb == 0 {
            None
        } else {
            Some((self.memory_usage_mb as f64 / self.total_memory_mb as f64) * 100.0)
        }
    }
}

/// Resource budget and constraints
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    /// Memory budget constraints
    pub memory_budget: MemoryBudget,
    /// CPU budget constraints
    pub cpu_budget: CpuBudget,
    /// Network budget constraints
    pub network_budget: NetworkBudget,
    /// Time budget constraints
    pub time_budget: TimeBudget,
    /// Enforcement strategy
    pub enforcement_strategy: BudgetEnforcementStrategy,
    /// Budget flexibility (0.0 = strict, 1.0 = flexible)
    pub flexibility: f64,
}

/// Memory budget configuration
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum memory allocation in MB
    pub max_allocation_mb: usize,
    /// Soft limit for memory usage in MB
    pub soft_limit_mb: usize,
    /// Memory cleanup threshold (percentage)
    pub cleanup_threshold: f64,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Memory priority levels
    pub priority_levels: Vec<MemoryPriority>,
}

/// Memory allocation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPriority {
    /// Critical memory for core operations
    Critical,
    /// High priority memory for optimization
    High,
    /// Normal priority memory for buffering
    Normal,
    /// Low priority memory for caching
    Low,
    /// Temporary memory that can be freed immediately
    Temporary,
}

/// CPU budget configuration
#[derive(Debug, Clone)]
pub struct CpuBudget {
    /// Maximum CPU utilization percentage
    pub max_utilization: f64,
    /// Target CPU utilization percentage
    pub target_utilization: f64,
    /// Maximum number of worker threads
    pub max_threads: usize,
    /// Thread priority management
    pub thread_priority: ThreadPriorityConfig,
    /// CPU affinity settings
    pub cpu_affinity: Option<Vec<usize>>,
}

/// Thread priority configuration
#[derive(Debug, Clone)]
pub struct ThreadPriorityConfig {
    /// High priority thread count
    pub high_priority_threads: usize,
    /// Normal priority thread count
    pub normal_priority_threads: usize,
    /// Background thread count
    pub background_threads: usize,
    /// Enable dynamic priority adjustment
    pub dynamic_priority: bool,
}

/// Network budget configuration
#[derive(Debug, Clone)]
pub struct NetworkBudget {
    /// Maximum bandwidth usage in MB/s
    pub max_bandwidth_mbps: f64,
    /// Bandwidth priority allocation
    pub priority_allocation: HashMap<String, f64>,
    /// Enable traffic shaping
    pub enable_traffic_shaping: bool,
    /// Quality of Service settings
    pub qos_settings: QoSSettings,
}

/// Quality of Service settings for network traffic
#[derive(Debug, Clone)]
pub struct QoSSettings {
    /// Latency requirements in milliseconds
    pub max_latency_ms: u64,
    /// Jitter tolerance in milliseconds
    pub jitter_tolerance_ms: u64,
    /// Packet loss tolerance (percentage)
    pub packet_loss_tolerance: f64,
    /// Traffic classes
    pub traffic_classes: Vec<TrafficClass>,
}

/// Network traffic classification
#[derive(Debug, Clone)]
pub struct TrafficClass {
    /// Class name
    pub name: String,
    /// Priority level (0 = highest)
    pub priority: u8,
    /// Bandwidth guarantee (percentage)
    pub bandwidth_guarantee: f64,
    /// Maximum bandwidth (percentage)
    pub max_bandwidth: f64,
}

/// Time budget configuration
#[derive(Debug, Clone)]
pub struct TimeBudget {
    /// Maximum processing time per batch
    pub max_batch_processing_time: Duration,
    /// Target processing time per batch
    pub target_batch_processing_time: Duration,
    /// Timeout for long-running operations
    pub operation_timeout: Duration,
    /// Deadline enforcement strategy
    pub deadline_enforcement: DeadlineEnforcement,
}

/// Deadline enforcement strategies
#[derive(Debug, Clone)]
pub enum DeadlineEnforcement {
    /// Strict deadline enforcement (fail if exceeded)
    Strict,
    /// Soft deadline with warnings
    Soft,
    /// Best effort (informational only)
    BestEffort,
    /// Adaptive deadline based on system load
    Adaptive,
}

/// Budget enforcement strategies
#[derive(Debug, Clone)]
pub enum BudgetEnforcementStrategy {
    /// Strict enforcement (fail if budget exceeded)
    Strict,
    /// Throttling (reduce resource usage)
    Throttling,
    /// Load shedding (drop low priority work)
    LoadShedding,
    /// Graceful degradation
    GracefulDegradation,
    /// Adaptive enforcement based on system state
    Adaptive,
}

/// Resource manager for streaming optimization
pub struct ResourceManager {
    /// Resource configuration
    config: ResourceConfig,
    /// Current resource usage
    current_usage: Arc<Mutex<ResourceUsage>>,
    /// Resource usage history
    usage_history: Arc<Mutex<VecDeque<ResourceUsage>>>,
    /// Resource budget
    budget: ResourceBudget,
    /// Resource allocations by component
    allocations: Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    /// Resource monitoring thread handle
    monitoring_handle: Option<std::thread::JoinHandle<()>>,
    /// Shutdown flag for the monitoring thread (R4). Without it the thread
    /// looped forever, outliving the manager that spawned it.
    shutdown: Arc<AtomicBool>,
    /// Long-lived OS probe shared with the monitoring thread (R1)
    probe: Arc<Mutex<SystemProbe>>,
    /// When `update_utilization` last collected a sample itself
    last_synchronous_sample: Option<Instant>,
    /// Resource prediction model
    predictor: ResourcePredictor,
    /// Resource optimizer
    optimizer: ResourceOptimizer,
    /// Alert system
    alert_system: ResourceAlertSystem,
    /// Real budget-violation counter (R7: `get_diagnostics` used to report a
    /// hardcoded `0` with a "would be calculated" comment)
    budget_violations: Arc<AtomicU64>,
    /// Accumulated penalty from budget violations, scaled by
    /// `ResourceBudgetConstraints::violation_penalty`
    budget_penalty: f64,
}

/// Resource allocation for a specific component
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Component name
    pub component_name: String,
    /// Allocated memory in MB
    pub allocated_memory_mb: usize,
    /// Allocated CPU percentage
    pub allocated_cpu_percent: f64,
    /// Allocated network bandwidth in MB/s
    pub allocated_bandwidth_mbps: f64,
    /// Priority level
    pub priority: ResourcePriority,
    /// Allocation timestamp
    pub allocation_time: Instant,
    /// Last access timestamp
    pub last_access: Instant,
    /// Usage statistics
    pub usage_stats: ComponentUsageStats,
}

/// Resource priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourcePriority {
    /// Critical system resources
    Critical = 0,
    /// High priority operations
    High = 1,
    /// Normal priority operations
    Normal = 2,
    /// Low priority background operations
    Low = 3,
    /// Temporary or cache operations
    Temporary = 4,
}

/// Usage statistics for a component
#[derive(Debug, Clone)]
pub struct ComponentUsageStats {
    /// Peak memory usage
    pub peak_memory_mb: usize,
    /// Average memory usage
    pub avg_memory_mb: usize,
    /// Peak CPU usage
    pub peak_cpu_percent: f64,
    /// Average CPU usage
    pub avg_cpu_percent: f64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Number of operations performed
    pub operation_count: u64,
    /// Efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

/// Resource usage prediction model
pub struct ResourcePredictor {
    /// Historical usage patterns
    pub(crate) usage_patterns: VecDeque<ResourceUsage>,
    /// Prediction horizon (steps ahead)
    pub(crate) prediction_horizon: usize,
    /// Mean absolute percentage error per resource, measured by scoring each
    /// prediction against the sample that actually arrived (R7)
    pub(crate) prediction_accuracy: HashMap<String, f64>,
    /// Per-minute-of-hour seasonal profile, keyed by resource
    pub(crate) seasonal_patterns: HashMap<String, Vec<f64>>,
    /// Trend analysis
    pub(crate) trend_analysis: ResourceTrendAnalysis,
    /// Whether prediction is enabled (CF1:
    /// `ResourceConfig::enable_resource_prediction`)
    pub(crate) enabled: bool,
    /// Observation counts backing the seasonal profile
    pub(crate) seasonal_counts: HashMap<String, Vec<u64>>,
    /// The prediction awaiting a real observation, and how many samples remain
    pub(crate) pending_prediction: Option<(usize, ResourceUsage)>,
}

/// Resource trend analysis
#[derive(Debug, Clone)]
pub struct ResourceTrendAnalysis {
    /// Memory usage trend
    pub memory_trend: TrendDirection,
    /// CPU usage trend
    pub cpu_trend: TrendDirection,
    /// Network usage trend
    pub network_trend: TrendDirection,
    /// Trend confidence
    pub trend_confidence: f64,
    /// Trend stability
    pub trend_stability: f64,
}

/// Trend direction indicators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
    /// Oscillating trend
    Oscillating,
    /// Unknown trend
    Unknown,
}

/// Resource optimizer for dynamic allocation
pub struct ResourceOptimizer {
    /// Optimization strategy
    pub(crate) strategy: ResourceOptimizationStrategy,
    /// Optimization history
    pub(crate) optimization_history: VecDeque<OptimizationEvent>,
    /// Performance impact tracking
    pub(crate) performance_impact: HashMap<String, f64>,
    /// Optimization constraints
    pub(crate) constraints: OptimizationConstraints,
    /// When each component last had a change applied, and its sign, used to
    /// enforce `StabilityRequirements` (R7)
    pub(crate) last_change: HashMap<String, (Instant, f64)>,
    /// Pending change magnitude per component, set by `clamp_change`
    pub(crate) pending_change: HashMap<String, f64>,
}

/// Resource optimization strategies
#[derive(Debug, Clone)]
pub enum ResourceOptimizationStrategy {
    /// Conservative optimization (minimize changes)
    Conservative,
    /// Aggressive optimization (maximize performance)
    Aggressive,
    /// Balanced optimization
    Balanced,
    /// Power-efficient optimization
    PowerEfficient,
    /// Latency-optimized
    LatencyOptimized,
    /// Throughput-optimized
    ThroughputOptimized,
}

/// Resource optimization event
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Optimization type
    pub optimization_type: String,
    /// Resources affected
    pub affected_resources: Vec<String>,
    /// Resource deltas
    pub resource_deltas: HashMap<String, f64>,
    /// Performance impact
    pub performance_impact: f64,
    /// Success indicator
    pub success: bool,
}

/// Constraints for resource optimization
#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    /// Minimum resource guarantees
    pub min_guarantees: HashMap<String, f64>,
    /// Maximum resource limits
    pub max_limits: HashMap<String, f64>,
    /// Resource change rate limits
    pub change_rate_limits: HashMap<String, f64>,
    /// Stability requirements
    pub stability_requirements: StabilityRequirements,
}

/// Stability requirements for resource allocation
#[derive(Debug, Clone)]
pub struct StabilityRequirements {
    /// Minimum stable period before changes
    pub min_stable_period: Duration,
    /// Maximum change frequency
    pub max_change_frequency: f64,
    /// Oscillation prevention
    pub prevent_oscillation: bool,
    /// Hysteresis factor (0.0 to 1.0)
    pub hysteresis_factor: f64,
}

/// Resource alert system
pub struct ResourceAlertSystem {
    /// Alert thresholds
    pub(crate) thresholds: ResourceThresholds,
    /// Active alerts
    pub(crate) active_alerts: VecDeque<ResourceAlert>,
    /// Alert history
    pub(crate) alert_history: VecDeque<ResourceAlert>,
    /// Alert handlers
    pub(crate) alert_handlers: Vec<Box<dyn AlertHandler>>,
    /// Monotonic counter backing collision-free alert identifiers (R6). The
    /// previous scheme was `Instant::now().elapsed().as_nanos()`, which is
    /// always ~0 because the instant is created on the same line, so every
    /// alert for a resource shared the same id.
    pub(crate) next_alert_id: u64,
}

/// Resource alert thresholds
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// Memory usage thresholds
    pub memory_thresholds: ThresholdSet,
    /// CPU usage thresholds
    pub cpu_thresholds: ThresholdSet,
    /// Network usage thresholds
    pub network_thresholds: ThresholdSet,
    /// Response time thresholds
    pub response_time_thresholds: ThresholdSet,
}

/// Threshold set for a resource type
#[derive(Debug, Clone)]
pub struct ThresholdSet {
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
    /// Emergency threshold
    pub emergency: f64,
    /// Recovery threshold (for clearing alerts)
    pub recovery: f64,
}

/// Resource alert
#[derive(Debug, Clone)]
pub struct ResourceAlert {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: Instant,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Resource type
    pub resource_type: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert message
    pub message: String,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
    /// Auto-resolution attempts
    pub auto_resolution_attempts: u32,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

/// Trait for handling resource alerts
pub trait AlertHandler: Send + Sync {
    /// Handles a resource alert
    fn handle_alert(&self, alert: &ResourceAlert) -> Result<(), String>;

    /// Gets handler priority (lower number = higher priority)
    fn priority(&self) -> u32;

    /// Checks if this handler can handle the given alert
    fn can_handle(&self, alert: &ResourceAlert) -> bool;
}

impl ResourceManager {
    /// Creates a new resource manager
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let resource_config = config.resource_config.clone();
        // CF1: `ResourceConfig::budget_constraints` was never read. Every one
        // of its five fields now drives real behaviour below.
        let constraints = resource_config.budget_constraints.clone();

        let available_cpus = num_cpus::get().max(1);
        // The old expression `num_cpus::get() - 2` underflows and panics on a
        // one- or two-core machine.
        let high_priority_threads = available_cpus.min(2);
        let normal_priority_threads = available_cpus.saturating_sub(high_priority_threads);

        // The soft limit is the tighter of "80% of the hard maximum" and the
        // operator-supplied memory budget.
        let soft_limit_mb = ((resource_config.max_memory_mb as f64 * 0.8) as usize)
            .min(constraints.memory_budget_mb.max(1));

        let budget = ResourceBudget {
            memory_budget: MemoryBudget {
                max_allocation_mb: resource_config.max_memory_mb,
                soft_limit_mb,
                cleanup_threshold: resource_config.cleanup_threshold,
                enable_compression: true,
                priority_levels: vec![
                    MemoryPriority::Critical,
                    MemoryPriority::High,
                    MemoryPriority::Normal,
                    MemoryPriority::Low,
                ],
            },
            cpu_budget: CpuBudget {
                max_utilization: resource_config.max_cpu_percent,
                // Target the operator's CPU budget, never above the hard cap.
                target_utilization: constraints
                    .cpu_budget_percent
                    .min(resource_config.max_cpu_percent),
                max_threads: available_cpus,
                thread_priority: ThreadPriorityConfig {
                    high_priority_threads,
                    normal_priority_threads,
                    background_threads: 1,
                    dynamic_priority: true,
                },
                cpu_affinity: None,
            },
            network_budget: NetworkBudget {
                max_bandwidth_mbps: 100.0, // Default limit
                priority_allocation: HashMap::new(),
                enable_traffic_shaping: false,
                qos_settings: QoSSettings {
                    max_latency_ms: 100,
                    jitter_tolerance_ms: 10,
                    packet_loss_tolerance: 0.1,
                    traffic_classes: Vec::new(),
                },
            },
            time_budget: TimeBudget {
                // The operator's per-operation time budget is the target; the
                // hard cap is three times that, and the operation timeout is
                // the configured budget itself.
                max_batch_processing_time: constraints.time_budget.saturating_mul(3),
                target_batch_processing_time: constraints.time_budget,
                operation_timeout: constraints.time_budget,
                deadline_enforcement: if constraints.strict_enforcement {
                    DeadlineEnforcement::Strict
                } else {
                    DeadlineEnforcement::Soft
                },
            },
            enforcement_strategy: if constraints.strict_enforcement {
                BudgetEnforcementStrategy::Strict
            } else {
                match resource_config.allocation_strategy {
                    ResourceAllocationStrategy::Static => BudgetEnforcementStrategy::Strict,
                    ResourceAllocationStrategy::Dynamic => BudgetEnforcementStrategy::Throttling,
                    ResourceAllocationStrategy::Adaptive => BudgetEnforcementStrategy::Adaptive,
                    _ => BudgetEnforcementStrategy::GracefulDegradation,
                }
            },
            // Strict enforcement means zero head-room above the budget.
            flexibility: if constraints.strict_enforcement {
                0.0
            } else {
                0.2
            },
        };

        let predictor = ResourcePredictor::new(resource_config.enable_resource_prediction);
        let optimizer = ResourceOptimizer::new(match resource_config.allocation_strategy {
            ResourceAllocationStrategy::Static => ResourceOptimizationStrategy::Conservative,
            ResourceAllocationStrategy::Dynamic => {
                ResourceOptimizationStrategy::ThroughputOptimized
            }
            ResourceAllocationStrategy::PriorityBased => {
                ResourceOptimizationStrategy::LatencyOptimized
            }
            _ => ResourceOptimizationStrategy::Balanced,
        });
        // R8: the alert thresholds used to be hardcoded percentages with no
        // relationship to the configured limits.
        let alert_system = ResourceAlertSystem::from_config(&resource_config, &budget);

        Ok(Self {
            config: resource_config,
            current_usage: Arc::new(Mutex::new(ResourceUsage::default())),
            usage_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            budget,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            monitoring_handle: None,
            shutdown: Arc::new(AtomicBool::new(false)),
            probe: Arc::new(Mutex::new(SystemProbe::new())),
            last_synchronous_sample: None,
            predictor,
            optimizer,
            alert_system,
            budget_violations: Arc::new(AtomicU64::new(0)),
            budget_penalty: 0.0,
        })
    }

    /// Starts resource monitoring.
    ///
    /// R4: the previous loop had no way to stop, so the thread outlived the
    /// manager. It now polls a shutdown flag on a short tick (independent of
    /// `monitoring_frequency`, which may be minutes) and
    /// [`Self::stop_monitoring`] joins it.
    pub fn start_monitoring(&mut self) -> Result<(), String> {
        if self.monitoring_handle.is_some() {
            return Ok(()); // Already monitoring
        }

        let current_usage = Arc::clone(&self.current_usage);
        let usage_history = Arc::clone(&self.usage_history);
        let shutdown = Arc::clone(&self.shutdown);
        let probe = Arc::clone(&self.probe);
        let monitoring_frequency = self.config.monitoring_frequency;

        shutdown.store(false, Ordering::SeqCst);
        let handle = std::thread::Builder::new()
            .name("optirs-resource-monitor".to_string())
            .spawn(move || {
                let mut next_sample = Instant::now();
                while !shutdown.load(Ordering::SeqCst) {
                    let now = Instant::now();
                    if now >= next_sample {
                        let usage = lock_recovered(&probe).sample();
                        {
                            let mut current = lock_recovered(&current_usage);
                            *current = usage.clone();
                        }
                        {
                            let mut history = lock_recovered(&usage_history);
                            if history.len() >= 1000 {
                                history.pop_front();
                            }
                            history.push_back(usage);
                        }
                        next_sample = now + monitoring_frequency.max(SHUTDOWN_POLL_INTERVAL);
                    }
                    std::thread::sleep(SHUTDOWN_POLL_INTERVAL);
                }
            })
            .map_err(|error| format!("failed to spawn the resource monitor thread: {error}"))?;

        self.monitoring_handle = Some(handle);
        Ok(())
    }

    /// Signals the monitoring thread to stop and waits for it.
    pub fn stop_monitoring(&mut self) -> Result<(), String> {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.monitoring_handle.take() {
            handle
                .join()
                .map_err(|_| "the resource monitor thread panicked".to_string())?;
        }
        Ok(())
    }

    /// Whether the background monitor is running.
    pub fn is_monitoring(&self) -> bool {
        self.monitoring_handle.is_some()
    }

    /// Collects current resource usage from the long-lived probe.
    pub fn collect_resource_usage(&self) -> ResourceUsage {
        lock_recovered(&self.probe).sample()
    }

    /// Allocates resources for a component
    pub fn allocate_resources(
        &mut self,
        component_name: &str,
        memory_mb: usize,
        cpu_percent: f64,
        priority: ResourcePriority,
    ) -> Result<(), String> {
        // Check budget constraints
        self.check_budget_constraints(memory_mb, cpu_percent)?;

        let allocation = ResourceAllocation {
            component_name: component_name.to_string(),
            allocated_memory_mb: memory_mb,
            allocated_cpu_percent: cpu_percent,
            allocated_bandwidth_mbps: 0.0, // Default
            priority,
            allocation_time: Instant::now(),
            last_access: Instant::now(),
            usage_stats: ComponentUsageStats {
                peak_memory_mb: 0,
                avg_memory_mb: 0,
                peak_cpu_percent: 0.0,
                avg_cpu_percent: 0.0,
                total_processing_time: Duration::ZERO,
                operation_count: 0,
                efficiency_score: 1.0,
            },
        };

        let mut allocations = lock_recovered(&self.allocations);
        allocations.insert(component_name.to_string(), allocation);

        Ok(())
    }

    /// Checks budget constraints for resource allocation.
    ///
    /// CF1: `ResourceBudgetConstraints::strict_enforcement` selects whether
    /// the budget has head-room (`flexibility`) and `violation_penalty` feeds
    /// the accumulated penalty surfaced by [`Self::budget_penalty`].
    fn check_budget_constraints(&self, memory_mb: usize, cpu_percent: f64) -> Result<(), String> {
        let allocations = lock_recovered(&self.allocations);

        // Calculate total allocated resources
        let total_memory: usize = allocations
            .values()
            .map(|a| a.allocated_memory_mb)
            .sum::<usize>()
            + memory_mb;

        let total_cpu: f64 = allocations
            .values()
            .map(|a| a.allocated_cpu_percent)
            .sum::<f64>()
            + cpu_percent;

        let flexibility = 1.0 + self.budget.flexibility;
        let memory_limit =
            (self.budget.memory_budget.max_allocation_mb as f64 * flexibility) as usize;
        let cpu_limit = self.budget.cpu_budget.max_utilization * flexibility;

        // Check constraints
        if total_memory > memory_limit {
            self.record_budget_violation();
            return Err(format!(
                "Memory allocation would exceed budget: {} MB > {} MB",
                total_memory, memory_limit
            ));
        }

        if total_cpu > cpu_limit {
            self.record_budget_violation();
            return Err(format!(
                "CPU allocation would exceed budget: {:.2}% > {:.2}%",
                total_cpu, cpu_limit
            ));
        }

        Ok(())
    }

    fn record_budget_violation(&self) {
        self.budget_violations.fetch_add(1, Ordering::Relaxed);
    }

    /// Number of budget violations observed so far (R7).
    pub fn budget_violations(&self) -> u64 {
        self.budget_violations.load(Ordering::Relaxed)
    }

    /// Accumulated budget-violation penalty (CF1:
    /// `ResourceBudgetConstraints::violation_penalty`).
    pub fn budget_penalty(&self) -> f64 {
        self.budget_penalty
    }

    /// Updates resource utilization tracking.
    ///
    /// When no monitoring thread is running this collects a sample itself
    /// (throttled to `monitoring_frequency`). Previously `start_monitoring`
    /// was never called from anywhere, so `current_usage` stayed at its
    /// all-zero `Default` forever and every consumer of it read zeros.
    pub fn update_utilization(&mut self) -> Result<(), String> {
        if self.monitoring_handle.is_none() {
            let due = self
                .last_synchronous_sample
                .map(|last| last.elapsed() >= self.config.monitoring_frequency)
                .unwrap_or(true);
            if due {
                let usage = self.collect_resource_usage();
                self.last_synchronous_sample = Some(Instant::now());
                {
                    let mut current = lock_recovered(&self.current_usage);
                    *current = usage.clone();
                }
                let mut history = lock_recovered(&self.usage_history);
                if history.len() >= 1000 {
                    history.pop_front();
                }
                history.push_back(usage);
            }
        }

        let current_usage = lock_recovered(&self.current_usage).clone();

        // Raise new alerts and clear the ones that recovered (R5).
        self.alert_system.update(&current_usage)?;

        // Count real budget violations against the observed usage (R7). The
        // allocation budget is per-process, so it is compared against this
        // process's own footprint, never against system-wide usage.
        let mut violated = false;
        if let Some(process_mb) = current_usage.process_memory() {
            if process_mb > self.budget.memory_budget.max_allocation_mb {
                violated = true;
            }
        }
        if let Some(cpu) = current_usage.cpu_usage() {
            if cpu > self.budget.cpu_budget.max_utilization {
                violated = true;
            }
        }
        if violated {
            self.record_budget_violation();
            self.budget_penalty += self.config.budget_constraints.violation_penalty;
        }

        // Update predictor (CF1: gated by `enable_resource_prediction`).
        self.predictor.update(&current_usage)?;

        // Check for optimization opportunities
        if self.config.enable_dynamic_allocation {
            self.optimizer
                .check_optimization_opportunities(&current_usage, &self.allocations)?;
        }

        Ok(())
    }

    /// Checks if sufficient resources are available for processing.
    ///
    /// The CPU term is only applied when a real CPU measurement exists; before
    /// the probe has two refreshes to compare, the placeholder `0.0` must not
    /// be mistaken for "idle".
    pub fn has_sufficient_resources_for_processing(&self) -> Result<bool, String> {
        let current_usage = lock_recovered(&self.current_usage);

        // Check memory availability against this process's own footprint; the
        // soft limit is a per-process allocation budget.
        let memory_available = match current_usage.process_memory() {
            Some(process_mb) => {
                process_mb < (self.budget.memory_budget.soft_limit_mb as f64 * 0.9) as usize
            }
            None => true,
        };

        // Check CPU availability
        let cpu_available = match current_usage.cpu_usage() {
            Some(cpu) => cpu < self.budget.cpu_budget.target_utilization * 0.9,
            None => true,
        };

        Ok(memory_available && cpu_available)
    }

    /// Computes resource allocation adaptation
    pub fn compute_allocation_adaptation(&mut self) -> Result<Option<Adaptation<f32>>, String> {
        let current_usage = lock_recovered(&self.current_usage);

        // Check if we need to adapt resource allocation. Again per-process:
        // shrinking this optimizer's buffers cannot help with memory another
        // process is holding.
        let process_memory_mb = current_usage.process_memory();
        if process_memory_mb
            .is_some_and(|process_mb| process_mb > self.budget.memory_budget.soft_limit_mb)
        {
            // Memory pressure - suggest reducing buffer sizes. The magnitude is
            // proportional to the real overshoot and clamped by the
            // optimizer's configured change-rate limit rather than being a
            // fixed -20%.
            let overshoot = (process_memory_mb.unwrap_or(0) as f64
                - self.budget.memory_budget.soft_limit_mb as f64)
                / (self.budget.memory_budget.soft_limit_mb.max(1) as f64);
            let magnitude = self.optimizer.clamp_change("memory", -overshoot);
            let adaptation = Adaptation {
                adaptation_type: AdaptationType::ResourceAllocation,
                magnitude: magnitude as f32,
                target_component: "memory_manager".to_string(),
                parameters: std::collections::HashMap::new(),
                priority: AdaptationPriority::High,
                timestamp: Instant::now(),
            };

            drop(current_usage);
            if self.optimizer.accept_change("memory_manager") {
                return Ok(Some(adaptation));
            }
            return Ok(None);
        }

        // Only react to a real CPU measurement; the not-yet-measured
        // placeholder must not be read as "0% busy".
        if let Some(cpu) = current_usage.cpu_usage() {
            if cpu > self.budget.cpu_budget.target_utilization {
                let overshoot = (cpu - self.budget.cpu_budget.target_utilization)
                    / self.budget.cpu_budget.target_utilization.max(1.0);
                let magnitude = self.optimizer.clamp_change("cpu", -overshoot);
                let adaptation = Adaptation {
                    adaptation_type: AdaptationType::ResourceAllocation,
                    magnitude: magnitude as f32,
                    target_component: "cpu_manager".to_string(),
                    parameters: std::collections::HashMap::new(),
                    priority: AdaptationPriority::High,
                    timestamp: Instant::now(),
                };

                drop(current_usage);
                if self.optimizer.accept_change("cpu_manager") {
                    return Ok(Some(adaptation));
                }
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Predicted resource usage `prediction_horizon` samples ahead, or `None`
    /// when prediction is disabled or there is not enough history (R7).
    pub fn predict_usage(&self) -> Option<ResourceUsage> {
        self.predictor.predict()
    }

    /// Mean absolute percentage error of the predictor, per resource (R7).
    pub fn prediction_accuracy(&self) -> &HashMap<String, f64> {
        self.predictor.accuracy()
    }

    /// Current resource trend analysis.
    pub fn trend_analysis(&self) -> &ResourceTrendAnalysis {
        self.predictor.trend_analysis()
    }

    /// Registers a handler invoked for every raised alert.
    pub fn register_alert_handler(&mut self, handler: Box<dyn AlertHandler>) {
        self.alert_system.register_handler(handler);
    }

    /// Currently unresolved alerts.
    pub fn active_alerts(&self) -> Vec<ResourceAlert> {
        self.alert_system.active_alerts.iter().cloned().collect()
    }

    /// Resolved alerts, oldest first.
    pub fn alert_history(&self) -> Vec<ResourceAlert> {
        self.alert_system.alert_history.iter().cloned().collect()
    }

    /// Applies resource allocation adaptation
    pub fn apply_allocation_adaptation(
        &mut self,
        adaptation: &Adaptation<f32>,
    ) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::ResourceAllocation {
            match adaptation.target_component.as_str() {
                "memory_manager" => {
                    // Adjust memory allocations
                    let factor = (1.0 + adaptation.magnitude).max(0.0);
                    let mut allocations = lock_recovered(&self.allocations);

                    for allocation in allocations.values_mut() {
                        if allocation.priority >= ResourcePriority::Normal {
                            allocation.allocated_memory_mb =
                                ((allocation.allocated_memory_mb as f32) * factor) as usize;
                        }
                    }
                    drop(allocations);
                    self.optimizer.record_applied_change("memory_manager");
                }
                "cpu_manager" => {
                    // Adjust CPU allocations
                    let factor = (1.0 + adaptation.magnitude).max(0.0);
                    let mut allocations = lock_recovered(&self.allocations);

                    for allocation in allocations.values_mut() {
                        if allocation.priority >= ResourcePriority::Normal {
                            allocation.allocated_cpu_percent *= factor as f64;
                        }
                    }
                    drop(allocations);
                    self.optimizer.record_applied_change("cpu_manager");
                }
                other => {
                    return Err(format!(
                        "no resource adaptation is defined for target component '{other}'"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Gets current resource usage
    pub fn current_usage(&self) -> Result<ResourceUsage, String> {
        Ok(lock_recovered(&self.current_usage).clone())
    }

    /// Gets resource usage history
    pub fn get_usage_history(&self, count: usize) -> Vec<ResourceUsage> {
        let history = lock_recovered(&self.usage_history);
        history.iter().rev().take(count).cloned().collect()
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> ResourceDiagnostics {
        let current_usage = lock_recovered(&self.current_usage);
        let allocations = lock_recovered(&self.allocations);

        ResourceDiagnostics {
            current_usage: current_usage.clone(),
            total_allocations: allocations.len(),
            // Per-process footprint against the per-process allocation budget.
            memory_utilization: current_usage.process_memory().map(|process_mb| {
                (process_mb as f64 / self.budget.memory_budget.max_allocation_mb.max(1) as f64)
                    * 100.0
            }),
            // System-wide pressure, which is what the memory alerts watch.
            system_memory_percent: current_usage.memory_usage_percent(),
            cpu_utilization: current_usage.cpu_usage(),
            active_alerts: self.alert_system.active_alerts.len(),
            // R7: a real count, not the hardcoded zero it used to be.
            budget_violations: self.budget_violations.load(Ordering::Relaxed) as usize,
            budget_penalty: self.budget_penalty,
            // Accumulated magnitude of the resource changes actually applied
            // per component. The optimizer tracked this from the start but
            // nothing surfaced it, so callers had no way to see whether the
            // optimization engine was doing anything at all.
            applied_change_magnitude: self.optimizer.performance_impact().clone(),
        }
    }
}

impl Drop for ResourceManager {
    /// R4: joins the monitoring thread so it cannot outlive the manager.
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.monitoring_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Diagnostic information for resource management
#[derive(Debug, Clone)]
pub struct ResourceDiagnostics {
    pub current_usage: ResourceUsage,
    pub total_allocations: usize,
    /// This process's footprint as a percentage of its allocation budget, or
    /// `None` when the process counters are unavailable.
    pub memory_utilization: Option<f64>,
    /// System-wide memory pressure as a percentage, or `None` when the system
    /// total is unknown.
    pub system_memory_percent: Option<f64>,
    /// Real CPU utilization, or `None` when the probe has not produced a
    /// measurement yet (R1).
    pub cpu_utilization: Option<f64>,
    pub active_alerts: usize,
    /// Real budget-violation count (R7).
    pub budget_violations: usize,
    /// Accumulated budget-violation penalty.
    pub budget_penalty: f64,
    /// Total absolute resource change applied per component by the
    /// optimization engine, keyed by component name.
    pub applied_change_magnitude: HashMap<String, f64>,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            memory_usage_mb: 0,
            total_memory_mb: 0,
            process_memory_mb: None,
            cpu_usage_percent: 0.0,
            cpu_usage_percent_valid: false,
            gpu_usage_percent: None,
            network_io_mbps: None,
            disk_io_mbps: None,
            active_threads: 0,
            timestamp: Instant::now(),
        }
    }
}

#[cfg(test)]
mod r2_memory_percent_tests {
    use super::*;

    fn usage_with(memory_usage_mb: usize, total_memory_mb: usize) -> ResourceUsage {
        ResourceUsage {
            memory_usage_mb,
            total_memory_mb,
            ..Default::default()
        }
    }

    /// R2: on a realistic machine (far more than 1 GB total memory), a
    /// moderate absolute memory usage must NOT be reported as a >100%
    /// (and therefore permanently `Emergency`-severity) utilization. The
    /// previous `(memory_usage_mb / 1024.0) * 100.0` assumed exactly 1 GB
    /// of total system memory.
    #[test]
    fn memory_percent_is_correct_on_realistic_machine() {
        // 4 GB used out of 32 GB total: a real, moderate 12.5% utilization.
        let usage = usage_with(4096, 32768);
        let percent = usage
            .memory_usage_percent()
            .expect("total_memory_mb is set, so this must be Some");
        assert!(
            (percent - 12.5).abs() < 1e-9,
            "R2 regression: expected ~12.5%, got {percent}%"
        );
        assert!(
            percent < 100.0,
            "R2 regression: realistic usage reported as over 100% (got {percent}%), \
             which pins alert severity at Emergency regardless of real pressure"
        );
    }

    /// R2: `memory_usage_percent` must return `None` (not a fabricated
    /// value) when the real total is unknown, so callers can skip the
    /// check instead of manufacturing a false alert.
    #[test]
    fn memory_percent_is_none_when_total_unknown() {
        let usage = usage_with(4096, 0);
        assert_eq!(usage.memory_usage_percent(), None);
    }

    /// R2 (via `ResourceAlertSystem::check_thresholds`): a moderate,
    /// realistic memory load on a large machine must not raise a memory
    /// alert at all, since it is nowhere near any real threshold. Before
    /// the fix, this scenario would compute `(4096 / 1024.0) * 100.0 =
    /// 400%`, which is above every threshold including `emergency: 95.0`.
    #[test]
    fn realistic_memory_load_raises_no_alert() {
        let mut alert_system = ResourceAlertSystem::new();
        let usage = usage_with(4096, 32768); // 12.5%, far below `warning: 70.0`
        let alerts = alert_system
            .check_thresholds(&usage)
            .expect("check_thresholds");
        assert!(
            alerts.is_empty(),
            "R2 regression: realistic 12.5% memory usage raised alert(s): {alerts:?}"
        );
    }

    /// R2: a genuinely high memory load (relative to the real total) must
    /// still raise an alert — the fix must not make the detector blind to
    /// real pressure, only correct about what "high" means.
    #[test]
    fn genuinely_high_memory_load_raises_alert() {
        let mut alert_system = ResourceAlertSystem::new();
        let usage = usage_with(31000, 32768); // ~94.6%, above `critical: 85.0`
        let alerts = alert_system
            .check_thresholds(&usage)
            .expect("check_thresholds");
        assert!(
            !alerts.is_empty(),
            "genuinely high memory usage (~94.6%) should raise an alert"
        );
        // The exact band depends on the configured `cleanup_threshold` (R8:
        // the thresholds are derived from configuration now, not hardcoded), so
        // assert on the severity floor rather than one specific level.
        assert!(alerts
            .iter()
            .any(|a| a.resource_type == "memory" && a.severity >= AlertSeverity::Critical));
    }
}
