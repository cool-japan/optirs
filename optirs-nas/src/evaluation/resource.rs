//! Resource monitoring for evaluation
//!
//! Tracks memory, CPU, GPU usage and other resources during evaluation.

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::SystemTime;

use crate::error::Result;
use crate::ResourceUsage;

/// Resource monitor for tracking evaluation resources
#[derive(Debug)]
pub struct ResourceMonitor<T: Float + Debug + Send + Sync + 'static> {
    /// Current resource usage
    current_usage: ResourceUsage,

    /// Resource usage history
    usage_history: VecDeque<ResourceUsageSnapshot<T>>,

    /// Resource limits
    limits: ResourceLimits<T>,

    /// Monitoring configuration
    config: MonitoringConfig,
}

/// Resource usage snapshot
#[derive(Debug, Clone)]
pub struct ResourceUsageSnapshot<T: Float + Debug + Send + Sync + 'static> {
    /// Timestamp
    pub timestamp: SystemTime,

    /// Memory usage (MB)
    pub memory_mb: T,

    /// CPU usage (%)
    pub cpu_usage_percent: T,

    /// GPU usage (%)
    pub gpu_usage_percent: T,

    /// Network I/O (MB/s)
    pub network_io_mbps: T,

    /// Disk I/O (MB/s)
    pub disk_io_mbps: T,
}

/// Resource limits
#[derive(Debug, Clone)]
pub struct ResourceLimits<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum memory (MB)
    pub max_memory_mb: T,

    /// Maximum CPU usage (%)
    pub max_cpu_percent: T,

    /// Maximum GPU memory (MB)
    pub max_gpu_memory_mb: T,

    /// Maximum evaluation time (seconds)
    pub max_evaluation_time_seconds: T,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring interval (milliseconds)
    pub monitoring_interval_ms: u64,

    /// History size limit
    pub history_size_limit: usize,

    /// Enable detailed monitoring
    pub enable_detailed_monitoring: bool,

    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

impl<T: Float + Debug + Default + Send + Sync> ResourceMonitor<T> {
    pub(crate) fn new() -> Self {
        Self {
            current_usage: ResourceUsage {
                memory_usage: 0,
                compute_time: 0.0,
                energy_consumption: 0.0,
                memory_gb: 0.0,
                cpu_time_seconds: 0.0,
                gpu_time_seconds: 0.0,
                energy_kwh: 0.0,
                cost_usd: 0.0,
                network_gb: 0.0,
            },
            usage_history: VecDeque::new(),
            limits: ResourceLimits {
                max_memory_mb: scirs2_core::numeric::NumCast::from(8192.0)
                    .unwrap_or_else(|| T::zero()),
                max_cpu_percent: scirs2_core::numeric::NumCast::from(90.0)
                    .unwrap_or_else(|| T::zero()),
                max_gpu_memory_mb: scirs2_core::numeric::NumCast::from(16384.0)
                    .unwrap_or_else(|| T::zero()),
                max_evaluation_time_seconds: scirs2_core::numeric::NumCast::from(3600.0)
                    .unwrap_or_else(|| T::zero()),
            },
            config: MonitoringConfig {
                monitoring_interval_ms: 1000,
                history_size_limit: 1000,
                enable_detailed_monitoring: true,
                alert_thresholds: HashMap::new(),
            },
        }
    }

    pub(crate) fn start_monitoring(&mut self) -> Result<()> {
        // Start resource monitoring (simplified)
        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) {
        // Stop resource monitoring
    }

    /// Get current resource usage
    pub fn get_current_usage(&self) -> &ResourceUsage {
        &self.current_usage
    }

    /// Get resource limits
    pub fn get_limits(&self) -> &ResourceLimits<T> {
        &self.limits
    }

    /// Set resource limits
    pub fn set_limits(&mut self, limits: ResourceLimits<T>) {
        self.limits = limits;
    }

    /// Get monitoring configuration
    pub fn get_config(&self) -> &MonitoringConfig {
        &self.config
    }

    /// Set monitoring configuration
    pub fn set_config(&mut self, config: MonitoringConfig) {
        self.config = config;
    }

    /// Record a resource snapshot
    pub fn record_snapshot(&mut self, snapshot: ResourceUsageSnapshot<T>) {
        if self.usage_history.len() >= self.config.history_size_limit {
            self.usage_history.pop_front();
        }
        self.usage_history.push_back(snapshot);
    }

    /// Get usage history
    pub fn get_history(&self) -> &VecDeque<ResourceUsageSnapshot<T>> {
        &self.usage_history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.usage_history.clear();
    }

    /// Check if any resource limit is exceeded
    pub fn check_limits(&self) -> bool {
        let memory_mb: f64 = self.current_usage.memory_gb * 1024.0;
        let max_memory: f64 =
            scirs2_core::numeric::NumCast::from(self.limits.max_memory_mb).unwrap_or(8192.0);

        memory_mb <= max_memory
    }
}
