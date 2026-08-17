// Resource Monitoring and Management for Neural Architecture Search
//
// This module provides comprehensive resource monitoring, constraint enforcement,
// and resource optimization functionality for the NAS system.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::config::{
    HardwareResources, ResourceConstraints, ResourceViolationHandling, TimeConstraints,
};
use super::results::{ResourceUsage, ResourceUsageSummary};
use super::telemetry::{StdTelemetry, TelemetrySample, TelemetrySource};
use crate::error::{OptimError, Result};

/// Resource monitor for tracking and managing system resources
pub struct ResourceMonitor<T: Float + Debug + Send + Sync + 'static> {
    /// Resource constraints
    constraints: ResourceConstraints<T>,

    /// Current resource usage
    current_usage: Arc<Mutex<ResourceUsage<T>>>,

    /// Resource usage history
    usage_history: Arc<Mutex<Vec<ResourceSnapshot<T>>>>,

    /// Monitoring configuration
    monitoring_config: MonitoringConfig,

    /// Resource monitors
    monitors: Vec<Box<dyn ResourceTracker<T>>>,

    /// Alert handlers
    alert_handlers: Vec<Box<dyn AlertHandler<T>>>,

    /// Monitoring state
    monitoring_state: MonitoringState,

    /// Resource optimization strategies
    optimization_strategies: Vec<Box<dyn ResourceOptimizer<T>>>,
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for ResourceMonitor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceMonitor")
            .field("constraints", &self.constraints)
            .field("current_usage", &self.current_usage)
            .field("usage_history", &self.usage_history)
            .field("monitoring_config", &self.monitoring_config)
            .field("monitoring_state", &self.monitoring_state)
            .field("monitors_count", &self.monitors.len())
            .field("alert_handlers_count", &self.alert_handlers.len())
            .field(
                "optimization_strategies_count",
                &self.optimization_strategies.len(),
            )
            .finish()
    }
}

/// Lock a mutex, recovering the guarded value if the mutex was poisoned (F22).
///
/// Every mutex in this module guards a plain data snapshot: a `ResourceUsage`
/// total or an append-only history vector. Neither has an invariant that a
/// panicking writer could leave broken, so `PoisonError::into_inner` is the correct
/// recovery — the previous `lock().expect("lock poisoned")` turned one unrelated
/// panic into a permanently panicking getter.
fn lock_recovering<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Derive memory pressure from a telemetry sample, or `None` when memory is not
/// measurable. Shared by [`SystemResourceTracker`] and [`ResourceMonitor`].
fn memory_pressure_from_sample(sample: &TelemetrySample) -> Option<MemoryPressure> {
    let total = sample.total_memory_gb?;
    let available = sample.available_memory_gb?;
    if total <= 0.0 {
        return None;
    }
    let used_fraction = ((total - available) / total).clamp(0.0, 1.0);
    Some(if used_fraction >= 0.95 {
        MemoryPressure::Critical
    } else if used_fraction >= 0.85 {
        MemoryPressure::High
    } else if used_fraction >= 0.7 {
        MemoryPressure::Medium
    } else {
        MemoryPressure::Low
    })
}

/// Resource snapshot for history tracking
#[derive(Debug, Clone)]
pub struct ResourceSnapshot<T: Float + Debug + Send + Sync + 'static> {
    /// Timestamp
    pub timestamp: Instant,

    /// Resource usage at this time
    pub usage: ResourceUsage<T>,

    /// System metrics
    pub system_metrics: SystemMetrics<T>,

    /// Number of processes visible on the system, when measurable.
    pub active_processes: Option<usize>,

    /// Memory pressure level, derived from measured available/total memory. `None`
    /// when memory is not measurable — it used to be hard-coded to
    /// [`MemoryPressure::Low`] regardless.
    pub memory_pressure: Option<MemoryPressure>,

    /// 1-minute load average per logical CPU, when measurable. Previously a
    /// hard-coded `0.6`.
    pub cpu_load_average: Option<T>,

    /// GPU utilization in `[0, 1]`, when measurable. Previously a hard-coded `0.8`.
    pub gpu_utilization: Option<T>,
}

/// System metrics.
///
/// Every field is `Option` because most of them cannot be measured in pure Rust
/// (F12). `None` means **not measured**, and consumers must treat it as
/// *unconstrained* — never as zero, and never as a limit violation. The previous
/// version made every field mandatory, which forced
/// [`SystemResourceTracker`] to fabricate values (32 GB memory, 4 GPUs, 65 C,
/// 250 W) that were then compared against the caller's real constraints.
///
/// Supply your own [`crate::nas_engine::telemetry::TelemetrySource`] to fill in
/// the fields the default source cannot reach.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Memory the OS reports as available for new allocations (GB).
    pub available_memory_gb: Option<T>,

    /// Logical CPUs usable by this process.
    pub available_cpu_cores: Option<usize>,

    /// GPU devices visible to this process.
    pub available_gpu_devices: Option<usize>,

    /// Free space on the working-directory filesystem (GB).
    pub available_disk_gb: Option<T>,

    /// Network bandwidth (MB/s).
    pub network_bandwidth: Option<T>,

    /// System temperature (Celsius).
    pub system_temperature: Option<T>,

    /// Power consumption (watts).
    pub power_consumption: Option<T>,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// History retention period
    pub history_retention: Duration,

    /// Enable detailed monitoring
    pub enable_detailed_monitoring: bool,

    /// Enable predictive monitoring
    pub enable_predictive_monitoring: bool,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,

    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Memory usage threshold (0.0 to 1.0)
    pub memory_threshold: f64,

    /// CPU usage threshold (0.0 to 1.0)
    pub cpu_threshold: f64,

    /// GPU usage threshold (0.0 to 1.0)
    pub gpu_threshold: f64,

    /// Disk usage threshold (0.0 to 1.0)
    pub disk_threshold: f64,

    /// Temperature threshold (Celsius)
    pub temperature_threshold: f64,

    /// Power consumption threshold (watts)
    pub power_threshold: f64,
}

/// Monitoring state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitoringState {
    Stopped,
    Starting,
    Running,
    Paused,
    Stopping,
    Error,
}

/// Resource tracker trait
pub trait ResourceTracker<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Get current resource usage
    fn get_current_usage(&self) -> Result<ResourceUsage<T>>;

    /// Get system metrics
    fn get_system_metrics(&self) -> Result<SystemMetrics<T>>;

    /// Check if resource limits are exceeded
    fn check_limits(
        &self,
        constraints: &ResourceConstraints<T>,
    ) -> Result<Vec<ResourceViolation<T>>>;

    /// Get tracker name
    fn name(&self) -> &str;

    /// Initialize tracker
    fn initialize(&mut self) -> Result<()>;

    /// Cleanup tracker
    fn cleanup(&mut self) -> Result<()>;

    /// The raw telemetry reading behind [`ResourceTracker::get_system_metrics`], if
    /// this tracker has one.
    ///
    /// [`ResourceMonitor::update_usage`] uses it for the snapshot fields that are not
    /// part of [`SystemMetrics`] (process count, load average, GPU utilization).
    /// Trackers with no underlying sample return `None`, and those snapshot fields
    /// then stay `None` too — which is the honest outcome.
    fn telemetry_sample(&self) -> Option<TelemetrySample> {
        None
    }
}

/// Resource violation
#[derive(Debug, Clone)]
pub struct ResourceViolation<T: Float + Debug + Send + Sync + 'static> {
    /// Violation type
    pub violation_type: ViolationType,

    /// Current value
    pub current_value: T,

    /// Limit value
    pub limit_value: T,

    /// Severity
    pub severity: ViolationSeverity,

    /// Violation time
    pub violation_time: Instant,

    /// Affected resources
    pub affected_resources: Vec<String>,

    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Types of resource violations
#[derive(Debug, Clone, Copy)]
pub enum ViolationType {
    MemoryExceeded,
    CPUExceeded,
    GPUExceeded,
    DiskExceeded,
    TimeExceeded,
    EnergyExceeded,
    TemperatureExceeded,
    PowerExceeded,
    NetworkExceeded,
    ComputationTimeExceeded,
    CostExceeded,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert handler trait
pub trait AlertHandler<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Handle resource violation
    fn handle_violation(&self, violation: &ResourceViolation<T>) -> Result<()>;

    /// Handle resource warning
    fn handle_warning(&self, warning: &ResourceWarning<T>) -> Result<()>;

    /// Get handler name
    fn name(&self) -> &str;
}

/// Resource warning
#[derive(Debug, Clone)]
pub struct ResourceWarning<T: Float + Debug + Send + Sync + 'static> {
    /// Warning type
    pub warning_type: WarningType,

    /// Current value
    pub current_value: T,

    /// Threshold value
    pub threshold_value: T,

    /// Trend direction
    pub trend: TrendDirection,

    /// Estimated time to violation
    pub time_to_violation: Option<Duration>,

    /// Warning message
    pub message: String,
}

/// Types of resource warnings
#[derive(Debug, Clone, Copy)]
pub enum WarningType {
    MemoryApproachingLimit,
    CPUApproachingLimit,
    GPUApproachingLimit,
    DiskApproachingLimit,
    TimeApproachingLimit,
    EnergyApproachingLimit,
    TemperatureRising,
    PowerIncreasing,
}

/// Trend directions
#[derive(Debug, Clone, Copy)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Resource optimizer trait
pub trait ResourceOptimizer<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Optimize resource usage
    fn optimize(
        &self,
        current_usage: &ResourceUsage<T>,
        constraints: &ResourceConstraints<T>,
    ) -> Result<OptimizationAction<T>>;

    /// Get optimizer name
    fn name(&self) -> &str;

    /// Get optimization priority
    fn priority(&self) -> OptimizationPriority;
}

/// Optimization actions
#[derive(Debug, Clone)]
pub struct OptimizationAction<T: Float + Debug + Send + Sync + 'static> {
    /// Action type
    pub action_type: ActionType,

    /// Action parameters
    pub parameters: HashMap<String, T>,

    /// Expected resource savings
    pub expected_savings: ResourceUsage<T>,

    /// Implementation cost
    pub implementation_cost: T,

    /// Action description
    pub description: String,

    /// Priority
    pub priority: OptimizationPriority,
}

/// Types of optimization actions
#[derive(Debug, Clone, Copy)]
pub enum ActionType {
    ReduceMemoryUsage,
    OptimizeCPUUsage,
    OptimizeGPUUsage,
    ClearCaches,
    GarbageCollection,
    ProcessThrottling,
    ResourceReallocation,
    TaskMigration,
    PowerManagement,
    TemperatureControl,
}

/// Optimization priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// System resource tracker backed by an injectable
/// [`TelemetrySource`] (F12).
///
/// It previously returned hard-coded readings (32 GB memory / 16 GB used, 60% CPU,
/// 4 GPUs at 80%, 1 TB disk, 1 GB/s network, 65 C, 250 W) from seven
/// `get_*_info` helpers, each commented "in a real implementation, this would
/// query system APIs". Those numbers reached [`ResourceTracker::check_limits`] and
/// were compared against the caller's real constraints, so a tight memory budget
/// aborted the search on invented data.
///
/// Now it reports whatever its source can actually measure and `None` for the rest,
/// and `check_limits` **skips** any check whose measurement is missing.
pub struct SystemResourceTracker {
    /// Tracker name
    name: String,

    /// Monitoring interval
    monitoring_interval: Duration,

    /// Last update time
    last_update: Instant,

    /// Telemetry source. Defaults to
    /// [`StdTelemetry`](crate::nas_engine::telemetry::StdTelemetry).
    telemetry: Box<dyn TelemetrySource>,

    /// Most recent reading, refreshed on every query.
    cached_sample: Option<TelemetrySample>,
}

impl Debug for SystemResourceTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemResourceTracker")
            .field("name", &self.name)
            .field("monitoring_interval", &self.monitoring_interval)
            .field("telemetry", &self.telemetry.name())
            .field("cached_sample", &self.cached_sample)
            .finish()
    }
}

impl SystemResourceTracker {
    /// Create a tracker using the default (honest, sparse) telemetry source.
    pub fn new(name: String, monitoring_interval: Duration) -> Self {
        Self::with_telemetry(name, monitoring_interval, Box::new(StdTelemetry::new()))
    }

    /// Create a tracker backed by a caller-supplied telemetry source. This is how
    /// GPU / thermal / power constraints become enforceable: provide a source that
    /// can measure them.
    pub fn with_telemetry(
        name: String,
        monitoring_interval: Duration,
        telemetry: Box<dyn TelemetrySource>,
    ) -> Self {
        Self {
            name,
            monitoring_interval,
            last_update: Instant::now(),
            telemetry,
            cached_sample: None,
        }
    }

    /// Name of the telemetry source in use.
    pub fn telemetry_name(&self) -> &str {
        self.telemetry.name()
    }

    /// The most recent reading, if one has been taken.
    pub fn last_sample(&self) -> Option<&TelemetrySample> {
        self.cached_sample.as_ref()
    }

    /// Take a fresh reading.
    pub fn sample(&self) -> TelemetrySample {
        self.telemetry.sample()
    }

    /// Memory pressure derived from the current reading, or `None` when memory is
    /// not measurable here.
    pub fn memory_pressure(&self) -> Option<MemoryPressure> {
        memory_pressure_from_sample(&self.telemetry.sample())
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ResourceTracker<T> for SystemResourceTracker
where
    T: From<f64> + std::fmt::Debug,
{
    /// Resource usage attributable to **this search**.
    ///
    /// Only the process's own resident set size is reported, and only where it can
    /// be measured; everything else stays at the accumulator identity (`0`), which
    /// is honest for a *usage* total that nothing has been attributed to. The
    /// previous version reported system-wide invented figures here, which is a
    /// category error as well as a fabrication: `ResourceUsage` is compared against
    /// a budget for the search, so charging it for the whole machine aborts the
    /// search whenever the host is busy.
    fn get_current_usage(&self) -> Result<ResourceUsage<T>> {
        let sample = self.telemetry.sample();
        let mut usage = ResourceUsage::default();
        if let Some(process_memory_gb) = sample.process_memory_gb {
            let value: T =
                scirs2_core::numeric::NumCast::from(process_memory_gb).unwrap_or_else(|| T::zero());
            usage.memory_gb = value;
            usage.peak_memory_gb = value;
        }
        Ok(usage)
    }

    fn get_system_metrics(&self) -> Result<SystemMetrics<T>> {
        let sample = self.telemetry.sample();
        let convert = |value: Option<f64>| -> Option<T> {
            value.and_then(|v| scirs2_core::numeric::NumCast::from(v))
        };
        Ok(SystemMetrics {
            available_memory_gb: convert(sample.available_memory_gb),
            available_cpu_cores: sample.logical_cpus,
            available_gpu_devices: sample.gpu_devices,
            available_disk_gb: convert(sample.available_disk_gb),
            network_bandwidth: convert(sample.network_bandwidth_mbps),
            system_temperature: convert(sample.temperature_celsius),
            power_consumption: convert(sample.power_watts),
        })
    }

    /// Report only violations backed by an actual measurement (F12).
    ///
    /// Every check is guarded by the presence of its measurement, so an
    /// unmeasurable resource is treated as **unlimited** and can never abort a
    /// search. This is the whole point: `check_violations` is called from
    /// `NeuralArchitectureSearch::should_stop_search` on every iteration, so a
    /// fabricated reading here terminated real searches.
    fn check_limits(
        &self,
        constraints: &ResourceConstraints<T>,
    ) -> Result<Vec<ResourceViolation<T>>> {
        let sample = self.telemetry.sample();
        let mut violations = Vec::new();

        // Memory: this process's RSS against the hardware memory budget.
        if let Some(process_memory_gb) = sample.process_memory_gb {
            let current: T =
                scirs2_core::numeric::NumCast::from(process_memory_gb).unwrap_or_else(|| T::zero());
            let limit: T =
                scirs2_core::numeric::NumCast::from(constraints.hardware_resources.max_memory_gb)
                    .unwrap_or_else(|| T::zero());
            if limit > T::zero() && current > limit {
                violations.push(ResourceViolation {
                    violation_type: ViolationType::MemoryExceeded,
                    current_value: current,
                    limit_value: limit,
                    severity: ViolationSeverity::High,
                    violation_time: Instant::now(),
                    affected_resources: vec!["Memory".to_string()],
                    suggested_actions: vec![
                        "Clear caches".to_string(),
                        "Reduce batch size".to_string(),
                        "Enable memory optimization".to_string(),
                    ],
                });
            }
        }

        // Power: only when a source can actually read it.
        if let Some(power_watts) = sample.power_watts {
            let current: T =
                scirs2_core::numeric::NumCast::from(power_watts).unwrap_or_else(|| T::zero());
            let limit: T =
                scirs2_core::numeric::NumCast::from(POWER_LIMIT_WATTS).unwrap_or_else(|| T::zero());
            if current > limit {
                violations.push(ResourceViolation {
                    violation_type: ViolationType::PowerExceeded,
                    current_value: current,
                    limit_value: limit,
                    severity: ViolationSeverity::Medium,
                    violation_time: Instant::now(),
                    affected_resources: vec!["Power".to_string()],
                    suggested_actions: vec![
                        "Reduce clock speeds".to_string(),
                        "Throttle processes".to_string(),
                        "Enable power saving mode".to_string(),
                    ],
                });
            }
        }

        // Temperature: likewise.
        if let Some(temperature_celsius) = sample.temperature_celsius {
            let current: T = scirs2_core::numeric::NumCast::from(temperature_celsius)
                .unwrap_or_else(|| T::zero());
            let limit: T = scirs2_core::numeric::NumCast::from(TEMPERATURE_LIMIT_CELSIUS)
                .unwrap_or_else(|| T::zero());
            if current > limit {
                violations.push(ResourceViolation {
                    violation_type: ViolationType::TemperatureExceeded,
                    current_value: current,
                    limit_value: limit,
                    severity: ViolationSeverity::Critical,
                    violation_time: Instant::now(),
                    affected_resources: vec!["CPU".to_string(), "GPU".to_string()],
                    suggested_actions: vec![
                        "Increase cooling".to_string(),
                        "Reduce workload".to_string(),
                        "Enable thermal throttling".to_string(),
                    ],
                });
            }
        }

        Ok(violations)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn initialize(&mut self) -> Result<()> {
        self.last_update = Instant::now();
        let sample = self.telemetry.sample();
        if sample.is_fully_unknown() {
            log::warn!(
                "resource tracker {} initialized with telemetry source '{}', which can measure \
                 nothing on this platform; every resource constraint will be treated as \
                 unlimited",
                self.name,
                self.telemetry.name()
            );
        } else {
            log::info!(
                "resource tracker {} initialized with telemetry source '{}': {:?}",
                self.name,
                self.telemetry.name(),
                sample
            );
        }
        self.cached_sample = Some(sample);
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        self.cached_sample = None;
        log::info!("cleaned up system resource tracker: {}", self.name);
        Ok(())
    }

    fn telemetry_sample(&self) -> Option<TelemetrySample> {
        Some(self.telemetry.sample())
    }
}

/// Package power above which a [`ViolationType::PowerExceeded`] is reported —
/// checked **only** when a telemetry source can actually read power draw.
const POWER_LIMIT_WATTS: f64 = 1000.0;

/// Die temperature above which a [`ViolationType::TemperatureExceeded`] is
/// reported — checked only when a source can actually read temperature.
const TEMPERATURE_LIMIT_CELSIUS: f64 = 80.0;

/// Console alert handler implementation
pub struct ConsoleAlertHandler {
    /// Handler name
    name: String,

    /// Enable verbose logging
    verbose: bool,
}

impl ConsoleAlertHandler {
    /// Create a new console alert handler
    pub fn new(name: String, verbose: bool) -> Self {
        Self { name, verbose }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AlertHandler<T> for ConsoleAlertHandler
where
    T: std::fmt::Display + std::fmt::Debug,
{
    /// Report a violation through `log`, at a level matching its severity (F24).
    ///
    /// This used to `println!` unconditionally, which writes to a library consumer's
    /// stdout with no way to filter or redirect it.
    fn handle_violation(&self, violation: &ResourceViolation<T>) -> Result<()> {
        let summary = format!(
            "resource violation {:?}: current={}, limit={}, severity={:?}",
            violation.violation_type,
            violation.current_value,
            violation.limit_value,
            violation.severity
        );
        match violation.severity {
            ViolationSeverity::Critical | ViolationSeverity::High => log::error!("{}", summary),
            ViolationSeverity::Medium => log::warn!("{}", summary),
            ViolationSeverity::Low => log::info!("{}", summary),
        }

        if self.verbose {
            log::debug!(
                "violation detail: time={:?}, affected={:?}, suggested={:?}",
                violation.violation_time,
                violation.affected_resources,
                violation.suggested_actions
            );
        }

        Ok(())
    }

    /// Report a warning through `log` (F24).
    fn handle_warning(&self, warning: &ResourceWarning<T>) -> Result<()> {
        log::warn!(
            "resource warning {:?}: current={}, threshold={}, trend={:?}",
            warning.warning_type,
            warning.current_value,
            warning.threshold_value,
            warning.trend
        );

        if self.verbose {
            log::debug!(
                "warning detail: time_to_violation={:?}, message={}",
                warning.time_to_violation,
                warning.message
            );
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Memory optimizer implementation
pub struct MemoryOptimizer {
    /// Optimizer name
    name: String,

    /// Optimization aggressiveness
    aggressiveness: f64,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(name: String, aggressiveness: f64) -> Self {
        Self {
            name,
            aggressiveness,
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ResourceOptimizer<T> for MemoryOptimizer
where
    T: From<f64> + std::cmp::PartialOrd,
{
    fn optimize(
        &self,
        current_usage: &ResourceUsage<T>,
        constraints: &ResourceConstraints<T>,
    ) -> Result<OptimizationAction<T>> {
        let memory_limit =
            scirs2_core::numeric::NumCast::from(constraints.hardware_resources.max_memory_gb)
                .unwrap_or_else(|| T::zero());

        if current_usage.memory_gb
            > memory_limit * scirs2_core::numeric::NumCast::from(0.8).unwrap_or_else(|| T::zero())
        {
            let reduction_target = current_usage.memory_gb
                * scirs2_core::numeric::NumCast::from(self.aggressiveness * 0.2)
                    .unwrap_or_else(|| T::zero());

            let mut parameters = HashMap::new();
            parameters.insert("memory_reduction_gb".to_string(), reduction_target);
            parameters.insert(
                "aggressiveness".to_string(),
                scirs2_core::numeric::NumCast::from(self.aggressiveness)
                    .unwrap_or_else(|| T::zero()),
            );

            Ok(OptimizationAction {
                action_type: ActionType::ReduceMemoryUsage,
                parameters,
                expected_savings: ResourceUsage {
                    memory_gb: reduction_target,
                    cpu_time_seconds: scirs2_core::numeric::NumCast::from(0.0)
                        .unwrap_or_else(|| T::zero()),
                    gpu_time_seconds: scirs2_core::numeric::NumCast::from(0.0)
                        .unwrap_or_else(|| T::zero()),
                    energy_kwh: scirs2_core::numeric::NumCast::from(0.0)
                        .unwrap_or_else(|| T::zero()),
                    network_io_gb: scirs2_core::numeric::NumCast::from(0.0)
                        .unwrap_or_else(|| T::zero()),
                    disk_io_gb: scirs2_core::numeric::NumCast::from(0.0)
                        .unwrap_or_else(|| T::zero()),
                    peak_memory_gb: reduction_target,
                    efficiency_score: scirs2_core::numeric::NumCast::from(0.1)
                        .unwrap_or_else(|| T::zero()),
                    cost_usd: scirs2_core::numeric::NumCast::from(0.0).unwrap_or_else(|| T::zero()),
                    network_gb: scirs2_core::numeric::NumCast::from(0.0)
                        .unwrap_or_else(|| T::zero()),
                },
                implementation_cost: scirs2_core::numeric::NumCast::from(0.05)
                    .unwrap_or_else(|| T::zero()),
                description: "Reduce memory usage through cache clearing and optimization"
                    .to_string(),
                priority: OptimizationPriority::High,
            })
        } else {
            Ok(OptimizationAction {
                action_type: ActionType::ReduceMemoryUsage,
                parameters: HashMap::new(),
                expected_savings: ResourceUsage::default(),
                implementation_cost: scirs2_core::numeric::NumCast::from(0.0)
                    .unwrap_or_else(|| T::zero()),
                description: "No memory optimization needed".to_string(),
                priority: OptimizationPriority::Low,
            })
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> OptimizationPriority {
        OptimizationPriority::High
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ResourceMonitor<T>
where
    T: From<f64> + std::fmt::Display,
{
    /// Create a new resource monitor
    pub fn new(constraints: ResourceConstraints<T>) -> Self {
        let monitoring_config = MonitoringConfig {
            monitoring_interval: Duration::from_secs(5),
            history_retention: Duration::from_secs(3600), // 1 hour
            enable_detailed_monitoring: true,
            enable_predictive_monitoring: false,
            alert_thresholds: AlertThresholds {
                memory_threshold: 0.8,
                cpu_threshold: 0.9,
                gpu_threshold: 0.9,
                disk_threshold: 0.9,
                temperature_threshold: 75.0,
                power_threshold: 800.0,
            },
            enable_auto_optimization: true,
        };

        let monitors: Vec<Box<dyn ResourceTracker<T>>> =
            vec![Box::new(SystemResourceTracker::new(
                "SystemTracker".to_string(),
                monitoring_config.monitoring_interval,
            ))];

        let alert_handlers: Vec<Box<dyn AlertHandler<T>>> = vec![Box::new(
            ConsoleAlertHandler::new("ConsoleHandler".to_string(), true),
        )];

        let optimization_strategies: Vec<Box<dyn ResourceOptimizer<T>>> = vec![Box::new(
            MemoryOptimizer::new("MemoryOptimizer".to_string(), 0.8),
        )];

        Self {
            constraints,
            current_usage: Arc::new(Mutex::new(ResourceUsage::default())),
            usage_history: Arc::new(Mutex::new(Vec::new())),
            monitoring_config,
            monitors,
            alert_handlers,
            monitoring_state: MonitoringState::Stopped,
            optimization_strategies,
        }
    }

    /// Start resource monitoring
    pub fn start_monitoring(&mut self) -> Result<()> {
        self.monitoring_state = MonitoringState::Starting;

        // Initialize all monitors
        for monitor in &mut self.monitors {
            monitor.initialize()?;
        }

        self.monitoring_state = MonitoringState::Running;
        log::info!(
            "resource monitoring started with {} tracker(s)",
            self.monitors.len()
        );
        Ok(())
    }

    /// Stop resource monitoring
    pub fn stop_monitoring(&mut self) -> Result<()> {
        self.monitoring_state = MonitoringState::Stopping;

        // Cleanup all monitors
        for monitor in &mut self.monitors {
            monitor.cleanup()?;
        }

        self.monitoring_state = MonitoringState::Stopped;
        log::info!("resource monitoring stopped");
        Ok(())
    }

    /// Sample every tracker and record a snapshot (F12).
    ///
    /// This is called once per generation from
    /// `NeuralArchitectureSearch::check_resource_constraints`; before that wiring it
    /// was never called at all, so `current_usage` stayed at
    /// `ResourceUsage::default()` forever and `check_resource_violations` could
    /// never fire.
    ///
    /// The snapshot's derived fields are all measured or absent: `active_processes`,
    /// `memory_pressure`, `cpu_load_average` and `gpu_utilization` used to be
    /// hard-coded to `1` / `Low` / `0.6` / `0.8`.
    ///
    /// The clock is read once and reused for the timestamp and the retention cutoff,
    /// so a snapshot cannot be pruned by an `Instant::now()` taken microseconds later.
    pub fn update_usage(&self) -> Result<()> {
        if self.monitoring_state != MonitoringState::Running {
            return Ok(());
        }

        let mut total_usage = ResourceUsage::default();
        let mut system_metrics = None;
        let mut telemetry_sample: Option<TelemetrySample> = None;

        for monitor in &self.monitors {
            let usage = monitor.get_current_usage()?;
            total_usage.memory_gb = total_usage.memory_gb + usage.memory_gb;
            total_usage.cpu_time_seconds = total_usage.cpu_time_seconds + usage.cpu_time_seconds;
            total_usage.gpu_time_seconds = total_usage.gpu_time_seconds + usage.gpu_time_seconds;
            total_usage.energy_kwh = total_usage.energy_kwh + usage.energy_kwh;
            if total_usage.peak_memory_gb < usage.peak_memory_gb {
                total_usage.peak_memory_gb = usage.peak_memory_gb;
            }

            if system_metrics.is_none() {
                system_metrics = Some(monitor.get_system_metrics()?);
            }
            if telemetry_sample.is_none() {
                telemetry_sample = monitor.telemetry_sample();
            }
        }

        {
            let mut current = lock_recovering(&self.current_usage);
            *current = total_usage.clone();
        }

        if let Some(metrics) = system_metrics {
            let sample = telemetry_sample.unwrap_or_default();
            let now = Instant::now();
            let snapshot = ResourceSnapshot {
                timestamp: now,
                usage: total_usage,
                system_metrics: metrics,
                active_processes: sample.process_count,
                memory_pressure: memory_pressure_from_sample(&sample),
                cpu_load_average: sample
                    .load_average_per_cpu
                    .and_then(scirs2_core::numeric::NumCast::from),
                gpu_utilization: sample
                    .gpu_utilization
                    .and_then(scirs2_core::numeric::NumCast::from),
            };

            {
                let mut history = lock_recovering(&self.usage_history);
                history.push(snapshot);
                let cutoff_time = now
                    .checked_sub(self.monitoring_config.history_retention)
                    .unwrap_or(now);
                history.retain(|s| s.timestamp >= cutoff_time);
            }
        }

        Ok(())
    }

    /// Check for resource violations
    pub fn check_violations(&self) -> Result<Vec<ResourceViolation<T>>> {
        let mut all_violations = Vec::new();

        for monitor in &self.monitors {
            let violations = monitor.check_limits(&self.constraints)?;
            all_violations.extend(violations);
        }

        // Handle violations
        for violation in &all_violations {
            for handler in &self.alert_handlers {
                handler.handle_violation(violation)?;
            }
        }

        Ok(all_violations)
    }

    /// Optimize resource usage
    pub fn optimize_resources(&self) -> Result<Vec<OptimizationAction<T>>> {
        if !self.monitoring_config.enable_auto_optimization {
            return Ok(Vec::new());
        }

        let current_usage = {
            let usage = lock_recovering(&self.current_usage);
            usage.clone()
        };

        let mut optimization_actions = Vec::new();

        for optimizer in &self.optimization_strategies {
            let action = optimizer.optimize(&current_usage, &self.constraints)?;
            if action.priority >= OptimizationPriority::Medium {
                optimization_actions.push(action);
            }
        }

        // Most urgent first (F25). `OptimizationPriority` derives `Ord` with
        // `Low < Medium < High < Critical`, so the previous ascending
        // `sort_by_key(|a| a.priority)` put the *least* urgent action at the front —
        // the opposite of what a caller applying a prefix of the list needs.
        optimization_actions.sort_by_key(|action| std::cmp::Reverse(action.priority));

        Ok(optimization_actions)
    }

    /// Get current resource usage.
    ///
    /// Recovers from a poisoned mutex instead of panicking (F22): the guarded data
    /// is a plain snapshot with no invariants that a panicking writer could have
    /// broken, so one panic elsewhere must not make this getter panic forever.
    pub fn get_current_usage(&self) -> ResourceUsage<T> {
        lock_recovering(&self.current_usage).clone()
    }

    /// Get usage history. Poison-recovering, for the same reason as
    /// [`ResourceMonitor::get_current_usage`].
    pub fn get_usage_history(&self) -> Vec<ResourceSnapshot<T>> {
        lock_recovering(&self.usage_history).clone()
    }

    /// Get monitoring state
    pub fn get_monitoring_state(&self) -> MonitoringState {
        self.monitoring_state
    }

    /// Replace the resource trackers. This is the injection point for real
    /// telemetry: hand in a [`SystemResourceTracker::with_telemetry`] built over a
    /// source that can measure what the default cannot (F12).
    pub fn set_trackers(&mut self, monitors: Vec<Box<dyn ResourceTracker<T>>>) {
        self.monitors = monitors;
    }

    /// Replace the alert handlers.
    pub fn set_alert_handlers(&mut self, handlers: Vec<Box<dyn AlertHandler<T>>>) {
        self.alert_handlers = handlers;
    }

    /// Replace the optimization strategies.
    pub fn set_optimization_strategies(&mut self, strategies: Vec<Box<dyn ResourceOptimizer<T>>>) {
        self.optimization_strategies = strategies;
    }

    /// Names of the telemetry sources currently backing this monitor.
    pub fn telemetry_source_names(&self) -> Vec<&str> {
        self.monitors.iter().map(|monitor| monitor.name()).collect()
    }

    /// Deliberately poison both internal mutexes, so a test can prove the getters
    /// recover (F22) rather than panicking forever.
    #[cfg(test)]
    pub(crate) fn poison_locks_for_test(&self) {
        for _ in 0..2 {
            let usage = std::sync::Arc::clone(&self.current_usage);
            let history = std::sync::Arc::clone(&self.usage_history);
            let handle = std::thread::spawn(move || {
                let _usage_guard = usage.lock();
                let _history_guard = history.lock();
                panic!("deliberate panic to poison the resource-monitor mutexes");
            });
            let _ = handle.join();
        }
        debug_assert!(self.current_usage.is_poisoned());
        debug_assert!(self.usage_history.is_poisoned());
    }

    /// Update constraints
    pub fn update_constraints(&mut self, constraints: ResourceConstraints<T>) {
        self.constraints = constraints;
    }

    /// Generate resource report
    pub fn generate_report(&self) -> ResourceReport<T> {
        let current_usage = self.get_current_usage();
        let history = self.get_usage_history();

        let usage_trend = self.calculate_usage_trend(&history);
        let efficiency_score = self.calculate_efficiency_score(&current_usage);
        let recommendations = self.generate_recommendations(&current_usage, &history);

        ResourceReport {
            timestamp: Instant::now(),
            current_usage,
            usage_trend,
            efficiency_score,
            total_samples: history.len(),
            monitoring_duration: match (history.first(), history.last()) {
                // `duration_since` on `Instant` panics if the argument is later;
                // history is append-only and monotonic, but `saturating_duration_since`
                // makes that structural rather than an assumption.
                (Some(first), Some(last)) => {
                    last.timestamp.saturating_duration_since(first.timestamp)
                }
                _ => Duration::from_secs(0),
            },
            recommendations,
            constraint_violations: self.check_violations().unwrap_or_default(),
        }
    }

    /// Calculate usage trend
    fn calculate_usage_trend(&self, history: &[ResourceSnapshot<T>]) -> UsageTrend<T> {
        if history.len() < 2 {
            return UsageTrend::default();
        }

        let recent_window = history.len().min(10);
        let recent = &history[history.len() - recent_window..];

        let memory_trend =
            self.calculate_metric_trend(recent.iter().map(|s| s.usage.memory_gb).collect());
        let cpu_trend =
            self.calculate_metric_trend(recent.iter().map(|s| s.usage.cpu_time_seconds).collect());
        let gpu_trend =
            self.calculate_metric_trend(recent.iter().map(|s| s.usage.gpu_time_seconds).collect());

        UsageTrend {
            memory_trend,
            cpu_trend,
            gpu_trend,
            energy_trend: TrendDirection::Stable,  // Simplified
            overall_trend: TrendDirection::Stable, // Simplified
            _phantom: std::marker::PhantomData,
        }
    }

    /// Calculate metric trend
    fn calculate_metric_trend(&self, values: Vec<T>) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Stable;
        }

        let first_half_avg = values.iter().take(values.len() / 2).fold(
            scirs2_core::numeric::NumCast::from(0.0).unwrap_or_else(|| T::zero()),
            |acc, &x| acc + x,
        ) / scirs2_core::numeric::NumCast::from(values.len() / 2)
            .unwrap_or_else(|| T::one());
        let second_half_avg =
            values.iter().skip(values.len() / 2).fold(
                scirs2_core::numeric::NumCast::from(0.0).unwrap_or_else(|| T::zero()),
                |acc, &x| acc + x,
            ) / scirs2_core::numeric::NumCast::from(values.len() - values.len() / 2)
                .unwrap_or_else(|| T::one());

        let change_threshold =
            scirs2_core::numeric::NumCast::from(0.05).unwrap_or_else(|| T::zero()); // 5% change threshold

        if second_half_avg > first_half_avg + change_threshold {
            TrendDirection::Increasing
        } else if second_half_avg < first_half_avg - change_threshold {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate efficiency score
    fn calculate_efficiency_score(&self, usage: &ResourceUsage<T>) -> T {
        // Simplified efficiency calculation
        usage.efficiency_score
    }

    /// Generate recommendations
    fn generate_recommendations(
        &self,
        _usage: &ResourceUsage<T>,
        _history: &[ResourceSnapshot<T>],
    ) -> Vec<String> {
        vec![
            "Consider enabling memory optimization".to_string(),
            "Monitor GPU utilization for potential improvements".to_string(),
            "Review energy consumption patterns".to_string(),
        ]
    }

    /// Check for resource constraint violations
    pub fn check_resource_violations(&self) -> Result<Vec<ResourceViolation<T>>> {
        let current_usage = self.get_current_usage();
        let mut violations = Vec::new();

        // Check memory constraints
        if current_usage.memory_gb > self.constraints.max_memory_gb {
            violations.push(ResourceViolation {
                violation_type: ViolationType::MemoryExceeded,
                severity: ViolationSeverity::High,
                current_value: current_usage.memory_gb,
                limit_value: self.constraints.max_memory_gb,
                violation_time: Instant::now(),
                affected_resources: vec!["System Memory".to_string(), "RAM".to_string()],
                suggested_actions: vec![
                    "Reduce batch size".to_string(),
                    "Clear memory cache".to_string(),
                    "Enable memory optimization".to_string(),
                ],
            });
        }

        // Check CPU time constraints
        if current_usage.cpu_time_seconds
            > self.constraints.max_computation_hours
                * scirs2_core::numeric::NumCast::from(3600.0).unwrap_or_else(|| T::zero())
        {
            violations.push(ResourceViolation {
                violation_type: ViolationType::ComputationTimeExceeded,
                severity: ViolationSeverity::Medium,
                current_value: current_usage.cpu_time_seconds,
                limit_value: self.constraints.max_computation_hours
                    * scirs2_core::numeric::NumCast::from(3600.0).unwrap_or_else(|| T::zero()),
                violation_time: Instant::now(),
                affected_resources: vec!["CPU".to_string(), "Computation Time".to_string()],
                suggested_actions: vec![
                    "Optimize algorithms".to_string(),
                    "Enable parallelization".to_string(),
                    "Reduce search space".to_string(),
                ],
            });
        }

        // Check energy constraints
        if current_usage.energy_kwh > self.constraints.max_energy_kwh {
            violations.push(ResourceViolation {
                violation_type: ViolationType::EnergyExceeded,
                severity: ViolationSeverity::Low,
                current_value: current_usage.energy_kwh,
                limit_value: self.constraints.max_energy_kwh,
                violation_time: Instant::now(),
                affected_resources: vec!["Energy".to_string(), "Power Consumption".to_string()],
                suggested_actions: vec![
                    "Enable power-saving mode".to_string(),
                    "Reduce GPU usage".to_string(),
                    "Optimize computation schedule".to_string(),
                ],
            });
        }

        // Check cost constraints
        if current_usage.cost_usd > self.constraints.max_cost_usd {
            violations.push(ResourceViolation {
                violation_type: ViolationType::CostExceeded,
                severity: ViolationSeverity::High,
                current_value: current_usage.cost_usd,
                limit_value: self.constraints.max_cost_usd,
                violation_time: Instant::now(),
                affected_resources: vec!["Budget".to_string(), "Cost".to_string()],
                suggested_actions: vec![
                    "Stop non-critical tasks".to_string(),
                    "Switch to cheaper resources".to_string(),
                    "Optimize resource allocation".to_string(),
                ],
            });
        }

        Ok(violations)
    }

    /// Get a summary of resource usage
    pub fn get_usage_summary(&self) -> ResourceUsageSummary<T> {
        let current_usage = self.get_current_usage();
        let history = self.get_usage_history();

        // Calculate totals from history
        let mut total_cpu_hours = T::zero();
        let mut total_gpu_hours = T::zero();
        let mut total_energy_kwh = T::zero();
        let mut total_cost_usd = T::zero();

        for snapshot in &history {
            total_cpu_hours = total_cpu_hours
                + snapshot.usage.cpu_time_seconds
                    / scirs2_core::numeric::NumCast::from(3600.0).unwrap_or_else(|| T::one());
            total_gpu_hours = total_gpu_hours
                + snapshot.usage.gpu_time_seconds
                    / scirs2_core::numeric::NumCast::from(3600.0).unwrap_or_else(|| T::one());
            total_energy_kwh = total_energy_kwh + snapshot.usage.energy_kwh;
            total_cost_usd = total_cost_usd + snapshot.usage.cost_usd;
        }

        ResourceUsageSummary {
            total_memory_gb: current_usage.memory_gb,
            total_cpu_hours,
            total_gpu_hours,
            total_energy_kwh,
            total_cost_usd,
            average_efficiency: current_usage.efficiency_score,
        }
    }
}

/// Resource usage trend
#[derive(Debug, Clone)]
pub struct UsageTrend<T: Float + Debug + Send + Sync + 'static> {
    pub memory_trend: TrendDirection,
    pub cpu_trend: TrendDirection,
    pub gpu_trend: TrendDirection,
    pub energy_trend: TrendDirection,
    pub overall_trend: TrendDirection,
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for UsageTrend<T> {
    fn default() -> Self {
        Self {
            memory_trend: TrendDirection::Stable,
            cpu_trend: TrendDirection::Stable,
            gpu_trend: TrendDirection::Stable,
            energy_trend: TrendDirection::Stable,
            overall_trend: TrendDirection::Stable,
            _phantom: PhantomData,
        }
    }
}

/// Resource monitoring report
#[derive(Debug, Clone)]
pub struct ResourceReport<T: Float + Debug + Send + Sync + 'static> {
    /// Report timestamp
    pub timestamp: Instant,

    /// Current resource usage
    pub current_usage: ResourceUsage<T>,

    /// Usage trends
    pub usage_trend: UsageTrend<T>,

    /// Efficiency score
    pub efficiency_score: T,

    /// Total samples in history
    pub total_samples: usize,

    /// Monitoring duration
    pub monitoring_duration: Duration,

    /// Optimization recommendations
    pub recommendations: Vec<String>,

    /// Current constraint violations
    pub constraint_violations: Vec<ResourceViolation<T>>,
}

// Default implementations
impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(5),
            history_retention: Duration::from_secs(3600),
            enable_detailed_monitoring: true,
            enable_predictive_monitoring: false,
            alert_thresholds: AlertThresholds::default(),
            enable_auto_optimization: true,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_threshold: 0.8,
            cpu_threshold: 0.9,
            gpu_threshold: 0.9,
            disk_threshold: 0.9,
            temperature_threshold: 75.0,
            power_threshold: 800.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_monitor_creation() {
        let constraints = ResourceConstraints::default();
        let monitor = ResourceMonitor::<f64>::new(constraints);
        assert_eq!(monitor.get_monitoring_state(), MonitoringState::Stopped);
    }

    #[test]
    fn test_system_resource_tracker() {
        use super::ResourceTracker;

        let mut tracker =
            SystemResourceTracker::new("TestTracker".to_string(), Duration::from_secs(1));

        assert!(<SystemResourceTracker as ResourceTracker<f64>>::initialize(&mut tracker).is_ok());

        let usage = <SystemResourceTracker as ResourceTracker<f64>>::get_current_usage(&tracker);
        assert!(usage.is_ok());

        let metrics = <SystemResourceTracker as ResourceTracker<f64>>::get_system_metrics(&tracker);
        assert!(metrics.is_ok());

        assert!(<SystemResourceTracker as ResourceTracker<f64>>::cleanup(&mut tracker).is_ok());
    }

    #[test]
    fn test_memory_optimizer() {
        let optimizer = MemoryOptimizer::new("TestOptimizer".to_string(), 0.5);

        let usage = ResourceUsage {
            memory_gb: 20.0,
            cpu_time_seconds: 100.0,
            gpu_time_seconds: 50.0,
            energy_kwh: 1.0,
            network_io_gb: 1.0,
            disk_io_gb: 2.0,
            peak_memory_gb: 22.0,
            efficiency_score: 0.8,
            cost_usd: 0.0,
            network_gb: 1.0,
        };

        let constraints = ResourceConstraints::default();
        let action = optimizer.optimize(&usage, &constraints);
        assert!(action.is_ok());
    }

    #[test]
    fn test_alert_handler() {
        let handler = ConsoleAlertHandler::new("TestHandler".to_string(), false);

        let violation = ResourceViolation {
            violation_type: ViolationType::MemoryExceeded,
            current_value: 25.0,
            limit_value: 20.0,
            severity: ViolationSeverity::High,
            violation_time: Instant::now(),
            affected_resources: vec!["Memory".to_string()],
            suggested_actions: vec!["Clear cache".to_string()],
        };

        assert!(handler.handle_violation(&violation).is_ok());
    }

    #[test]
    fn test_trend_calculation() {
        let constraints = ResourceConstraints::default();
        let monitor = ResourceMonitor::<f64>::new(constraints);

        let values = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let trend = monitor.calculate_metric_trend(values);
        assert!(matches!(trend, TrendDirection::Increasing));

        let stable_values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let stable_trend = monitor.calculate_metric_trend(stable_values);
        assert!(matches!(stable_trend, TrendDirection::Stable));
    }
    use super::super::telemetry::FixedTelemetry;

    /// Build a monitor whose single tracker reports exactly `sample`.
    fn monitor_with(
        constraints: ResourceConstraints<f64>,
        sample: TelemetrySample,
    ) -> ResourceMonitor<f64> {
        let mut monitor = ResourceMonitor::<f64>::new(constraints);
        monitor.set_trackers(vec![Box::new(SystemResourceTracker::with_telemetry(
            "fixed".to_string(),
            Duration::from_secs(1),
            Box::new(FixedTelemetry::new("fixed", sample)),
        ))]);
        monitor
    }

    // ---- F12: no fabricated telemetry -----------------------------------

    /// The default tracker must not report the old invented constants, and every
    /// unmeasurable metric must be `None`.
    #[test]
    fn the_default_tracker_reports_no_fabricated_metrics() {
        use super::ResourceTracker;
        let tracker = SystemResourceTracker::new("t".to_string(), Duration::from_secs(1));
        assert_eq!(tracker.telemetry_name(), "std");

        let metrics = <SystemResourceTracker as ResourceTracker<f64>>::get_system_metrics(&tracker)
            .expect("metrics");

        // These were 4 GPUs, 1 TB disk (500 GB free), 1 GB/s, 65 C and 250 W.
        assert_eq!(metrics.available_gpu_devices, None);
        assert_eq!(metrics.available_disk_gb, None);
        assert_eq!(metrics.network_bandwidth, None);
        assert_eq!(metrics.system_temperature, None);
        assert_eq!(metrics.power_consumption, None);
        // CPU count is genuinely measurable.
        assert!(metrics.available_cpu_cores.is_some_and(|cores| cores >= 1));

        let usage = <SystemResourceTracker as ResourceTracker<f64>>::get_current_usage(&tracker)
            .expect("usage");
        // Memory was a fixed 16.0 GB with a 19.2 GB "peak"; now it is either a real
        // measurement or the accumulator identity.
        assert_ne!(usage.memory_gb, 16.0);
        assert_ne!(usage.peak_memory_gb, 19.2);
        assert!(usage.memory_gb >= 0.0 && usage.memory_gb.is_finite());
        // These were 0.25 kWh, 1 GB network, 2 GB disk I/O, 0.8 efficiency, $0.50.
        assert_eq!(usage.energy_kwh, 0.0);
        assert_eq!(usage.network_io_gb, 0.0);
        assert_eq!(usage.disk_io_gb, 0.0);
        assert_eq!(usage.cost_usd, 0.0);
    }

    /// The core F12 hazard: a tight constraint must not abort a search on the
    /// strength of an unmeasured resource.
    #[test]
    fn unknown_measurements_are_treated_as_unlimited() {
        use super::ResourceTracker;
        let tracker = SystemResourceTracker::with_telemetry(
            "blind".to_string(),
            Duration::from_secs(1),
            Box::new(FixedTelemetry::new("blind", TelemetrySample::unknown())),
        );

        // An absurdly tight budget on every axis.
        let mut constraints = ResourceConstraints::<f64>::default();
        constraints.hardware_resources.max_memory_gb = 0.001;
        constraints.max_memory_gb = 0.001;
        constraints.max_computation_hours = 0.0;
        constraints.max_energy_kwh = 0.0;
        constraints.max_cost_usd = 0.0;

        let violations = tracker.check_limits(&constraints).expect("check_limits");
        assert!(
            violations.is_empty(),
            "an unmeasurable resource must never produce a violation, got {:?}",
            violations
                .iter()
                .map(|v| v.violation_type)
                .collect::<Vec<_>>()
        );
    }

    /// A measured value over budget still must be reported — the guard must not
    /// have disabled enforcement altogether.
    #[test]
    fn measured_overruns_are_still_reported() {
        use super::ResourceTracker;
        let sample = TelemetrySample {
            process_memory_gb: Some(9.0),
            temperature_celsius: Some(95.0),
            power_watts: Some(1500.0),
            ..TelemetrySample::unknown()
        };
        let tracker = SystemResourceTracker::with_telemetry(
            "hot".to_string(),
            Duration::from_secs(1),
            Box::new(FixedTelemetry::new("hot", sample)),
        );

        let mut constraints = ResourceConstraints::<f64>::default();
        constraints.hardware_resources.max_memory_gb = 4.0;

        let violations = tracker.check_limits(&constraints).expect("check_limits");
        let kinds: Vec<String> = violations
            .iter()
            .map(|v| format!("{:?}", v.violation_type))
            .collect();
        assert!(kinds.contains(&"MemoryExceeded".to_string()), "{kinds:?}");
        assert!(
            kinds.contains(&"TemperatureExceeded".to_string()),
            "{kinds:?}"
        );
        assert!(kinds.contains(&"PowerExceeded".to_string()), "{kinds:?}");

        // A generous budget produces no memory violation.
        constraints.hardware_resources.max_memory_gb = 64.0;
        let relaxed = tracker.check_limits(&constraints).expect("check_limits");
        assert!(!relaxed
            .iter()
            .any(|v| matches!(v.violation_type, ViolationType::MemoryExceeded)));
    }

    /// `update_usage` must actually record what the tracker measured, and the
    /// snapshot's derived fields must be measured or `None` — never the old
    /// hard-coded `1` / `Low` / `0.6` / `0.8`.
    #[test]
    fn update_usage_records_measured_values_and_no_hardcoded_ones() {
        let sample = TelemetrySample {
            process_memory_gb: Some(2.5),
            total_memory_gb: Some(16.0),
            available_memory_gb: Some(2.0),
            logical_cpus: Some(8),
            load_average_per_cpu: Some(0.375),
            process_count: Some(412),
            ..TelemetrySample::unknown()
        };
        let mut monitor = monitor_with(ResourceConstraints::default(), sample);

        // Nothing is recorded while monitoring is stopped.
        monitor.update_usage().expect("update while stopped");
        assert!(monitor.get_usage_history().is_empty());

        monitor.start_monitoring().expect("start");
        monitor.update_usage().expect("update");

        let usage = monitor.get_current_usage();
        assert_eq!(usage.memory_gb, 2.5, "the measured RSS must be recorded");

        let history = monitor.get_usage_history();
        assert_eq!(history.len(), 1);
        let snapshot = &history[0];
        assert_eq!(snapshot.active_processes, Some(412));
        assert_eq!(snapshot.cpu_load_average, Some(0.375));
        assert_ne!(
            snapshot.cpu_load_average,
            Some(0.6),
            "0.6 was the hardcoded load average"
        );
        assert_eq!(
            snapshot.gpu_utilization, None,
            "GPU utilization is unmeasurable here; 0.8 was fabricated"
        );
        // 14/16 = 87.5% used -> High, derived, not the hardcoded Low.
        assert!(
            matches!(snapshot.memory_pressure, Some(MemoryPressure::High)),
            "got {:?}",
            snapshot.memory_pressure
        );
        assert_eq!(snapshot.system_metrics.available_cpu_cores, Some(8));

        // The summary now reflects real data instead of an untouched default.
        let summary = monitor.get_usage_summary();
        assert_eq!(summary.total_memory_gb, 2.5);
    }

    #[test]
    fn a_blind_monitor_records_a_snapshot_with_everything_unknown() {
        let mut monitor = monitor_with(ResourceConstraints::default(), TelemetrySample::unknown());
        monitor.start_monitoring().expect("start");
        monitor.update_usage().expect("update");

        let history = monitor.get_usage_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].active_processes, None);
        assert_eq!(history[0].memory_pressure, None);
        assert_eq!(history[0].cpu_load_average, None);
        assert_eq!(history[0].gpu_utilization, None);
        assert!(monitor
            .check_resource_violations()
            .expect("violations")
            .is_empty());
    }

    // ---- F22: poisoned locks are recovered, not fatal --------------------

    #[test]
    fn a_poisoned_lock_is_recovered_instead_of_panicking_forever() {
        use std::sync::{Arc, Mutex};

        let shared: Arc<Mutex<u32>> = Arc::new(Mutex::new(7));
        let poisoner = Arc::clone(&shared);
        let handle = std::thread::spawn(move || {
            let _guard = poisoner.lock().expect("first lock succeeds");
            panic!("deliberate panic to poison the mutex");
        });
        assert!(
            handle.join().is_err(),
            "the helper thread must have panicked"
        );
        assert!(shared.is_poisoned(), "the mutex must be poisoned");

        // `lock().expect("lock poisoned")` would panic here forever.
        assert_eq!(*lock_recovering(&shared), 7);
    }

    #[test]
    fn getters_survive_a_poisoned_usage_lock() {
        let sample = TelemetrySample {
            process_memory_gb: Some(1.25),
            ..TelemetrySample::unknown()
        };
        let mut monitor = monitor_with(ResourceConstraints::default(), sample);
        monitor.start_monitoring().expect("start");
        monitor.update_usage().expect("update");

        monitor.poison_locks_for_test();

        // Every getter must still work.
        assert_eq!(monitor.get_current_usage().memory_gb, 1.25);
        assert_eq!(monitor.get_usage_history().len(), 1);
        let report = monitor.generate_report();
        assert_eq!(report.total_samples, 1);
        assert!(monitor.optimize_resources().is_ok());
    }

    // ---- F25: priority ordering ------------------------------------------

    /// An optimizer that always emits an action at a chosen priority, so the sort
    /// direction can be observed. `MemoryOptimizer` alone cannot do this: it emits
    /// either `High` or `Low`, and `optimize_resources` filters `Low` out, so every
    /// surviving action has the same priority and any ordering passes.
    #[derive(Debug)]
    struct FixedPriorityOptimizer {
        name: String,
        priority: OptimizationPriority,
    }

    impl ResourceOptimizer<f64> for FixedPriorityOptimizer {
        fn optimize(
            &self,
            _current_usage: &ResourceUsage<f64>,
            _constraints: &ResourceConstraints<f64>,
        ) -> Result<OptimizationAction<f64>> {
            Ok(OptimizationAction {
                action_type: ActionType::ClearCaches,
                parameters: HashMap::new(),
                expected_savings: ResourceUsage::default(),
                implementation_cost: 0.0,
                description: format!("{} action", self.name),
                priority: self.priority,
            })
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> OptimizationPriority {
            self.priority
        }
    }

    #[test]
    fn optimization_actions_are_returned_most_urgent_first() {
        let mut monitor = monitor_with(ResourceConstraints::default(), TelemetrySample::unknown());
        // Registered in an order that an ascending sort would preserve and a
        // descending sort must reverse.
        monitor.set_optimization_strategies(vec![
            Box::new(FixedPriorityOptimizer {
                name: "medium".to_string(),
                priority: OptimizationPriority::Medium,
            }),
            Box::new(FixedPriorityOptimizer {
                name: "critical".to_string(),
                priority: OptimizationPriority::Critical,
            }),
            Box::new(FixedPriorityOptimizer {
                name: "high".to_string(),
                priority: OptimizationPriority::High,
            }),
        ]);
        monitor.start_monitoring().expect("start");
        monitor.update_usage().expect("update");

        let actions = monitor.optimize_resources().expect("optimize");
        let priorities: Vec<OptimizationPriority> =
            actions.iter().map(|action| action.priority).collect();
        assert_eq!(
            priorities,
            vec![
                OptimizationPriority::Critical,
                OptimizationPriority::High,
                OptimizationPriority::Medium
            ],
            "the previous ascending `sort_by_key(|a| a.priority)` yielded \
             [Medium, High, Critical] — least urgent first"
        );
    }

    #[test]
    fn optimization_can_be_disabled() {
        let mut monitor = monitor_with(ResourceConstraints::default(), TelemetrySample::unknown());
        monitor.set_optimization_strategies(vec![Box::new(FixedPriorityOptimizer {
            name: "critical".to_string(),
            priority: OptimizationPriority::Critical,
        })]);
        monitor.monitoring_config.enable_auto_optimization = false;
        assert!(monitor.optimize_resources().expect("optimize").is_empty());
    }
}
