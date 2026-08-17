// Resource alert system with real lifecycle management (findings R5, R6, R8).
//
// * R5: `active_alerts` was pushed to and never cleared, so a single transient
//   spike left an alert "active" for the lifetime of the process and
//   `get_diagnostics().active_alerts` only ever grew.
// * R6: alert ids were `format!("{resource}_{}", Instant::now().elapsed().as_nanos())`,
//   and `Instant::now().elapsed()` is ~0 by construction, so every alert for a
//   resource collided on the same id.
// * R8: the thresholds were hardcoded percentages unrelated to the configured
//   limits.

use super::{
    AlertHandler, AlertSeverity, ResourceAlert, ResourceAlertSystem, ResourceBudget,
    ResourceConfig, ResourceThresholds, ResourceUsage, ThresholdSet,
};
use std::collections::VecDeque;
use std::time::Instant;

/// Maximum retained resolved alerts.
const MAX_ALERT_HISTORY: usize = 1000;

impl ResourceAlertSystem {
    /// Thresholds derived from the default configuration.
    ///
    /// Test-only: production builds always have a real `ResourceConfig` and
    /// budget to hand, and construct through [`Self::from_config`].
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        let config = ResourceConfig::default();
        let budget = default_budget(&config);
        Self::from_config(&config, &budget)
    }

    /// R8: build the thresholds from the operator's configuration instead of
    /// hardcoded percentages, so raising `max_cpu_percent` or tightening
    /// `cleanup_threshold` actually moves the alerting bands.
    pub(crate) fn from_config(config: &ResourceConfig, budget: &ResourceBudget) -> Self {
        // Memory: the cleanup threshold is the point at which the system
        // already wants to reclaim, so treat it as the emergency band and work
        // downwards from there.
        let memory_emergency = (config.cleanup_threshold * 100.0).clamp(10.0, 99.0);
        let memory_thresholds = ThresholdSet {
            emergency: memory_emergency,
            critical: memory_emergency * 0.95,
            warning: memory_emergency * 0.85,
            recovery: memory_emergency * 0.75,
        };

        // CPU: the configured maximum is the emergency band, the operator's
        // budget (`target_utilization`) is the warning band.
        let cpu_emergency = config.max_cpu_percent.clamp(10.0, 100.0);
        let cpu_warning = budget
            .cpu_budget
            .target_utilization
            .clamp(1.0, cpu_emergency * 0.95);
        let cpu_thresholds = ThresholdSet {
            emergency: cpu_emergency,
            critical: (cpu_emergency + cpu_warning) / 2.0,
            warning: cpu_warning,
            recovery: cpu_warning * 0.9,
        };

        // Network thresholds are absolute MB/s bands off the bandwidth budget.
        let max_bandwidth = budget.network_budget.max_bandwidth_mbps.max(1.0);
        let network_thresholds = ThresholdSet {
            emergency: max_bandwidth * 0.98,
            critical: max_bandwidth * 0.9,
            warning: max_bandwidth * 0.7,
            recovery: max_bandwidth * 0.6,
        };

        // Response-time thresholds come from the configured time budget.
        let target_ms = budget
            .time_budget
            .target_batch_processing_time
            .as_millis()
            .max(1) as f64;
        let max_ms = budget
            .time_budget
            .max_batch_processing_time
            .as_millis()
            .max(1) as f64;
        let response_time_thresholds = ThresholdSet {
            warning: target_ms,
            critical: (target_ms + max_ms) / 2.0,
            emergency: max_ms,
            recovery: target_ms * 0.5,
        };

        Self {
            thresholds: ResourceThresholds {
                memory_thresholds,
                cpu_thresholds,
                network_thresholds,
                response_time_thresholds,
            },
            active_alerts: VecDeque::new(),
            alert_history: VecDeque::with_capacity(MAX_ALERT_HISTORY),
            alert_handlers: Vec::new(),
            next_alert_id: 0,
        }
    }

    /// Register a handler invoked for every raised alert.
    pub(crate) fn register_handler(&mut self, handler: Box<dyn AlertHandler>) {
        self.alert_handlers.push(handler);
        // Highest priority (lowest number) first.
        self.alert_handlers
            .sort_by_key(|handler| handler.priority());
    }

    /// Full alert cycle: raise what breached, clear what recovered (R5).
    pub(crate) fn update(&mut self, usage: &ResourceUsage) -> Result<(), String> {
        for (resource, value) in observed_values(usage) {
            let thresholds = self.thresholds_for(&resource);
            match severity_for(value, &thresholds) {
                Some(severity) => {
                    let threshold_value = threshold_for(severity.clone(), &thresholds);
                    self.raise(&resource, value, threshold_value, severity)?;
                }
                None => {
                    if value <= thresholds.recovery {
                        self.clear(&resource);
                    }
                }
            }
        }
        Ok(())
    }

    fn thresholds_for(&self, resource: &str) -> ThresholdSet {
        match resource {
            "memory" => self.thresholds.memory_thresholds.clone(),
            "cpu" => self.thresholds.cpu_thresholds.clone(),
            "network" => self.thresholds.network_thresholds.clone(),
            _ => self.thresholds.response_time_thresholds.clone(),
        }
    }

    /// Raise a new alert, or update the one already active for this resource.
    fn raise(
        &mut self,
        resource_type: &str,
        current_value: f64,
        threshold_value: f64,
        severity: AlertSeverity,
    ) -> Result<(), String> {
        if let Some(existing) = self
            .active_alerts
            .iter_mut()
            .find(|alert| alert.resource_type == resource_type)
        {
            existing.current_value = current_value;
            existing.auto_resolution_attempts += 1;
            if existing.severity != severity {
                existing.severity = severity.clone();
                existing.threshold_value = threshold_value;
                existing.message = format!(
                    "{} usage is {:.2} (threshold: {:.2})",
                    resource_type, current_value, threshold_value
                );
            }
            return Ok(());
        }

        // R6: a monotonic counter, so repeated firings never share an id.
        self.next_alert_id += 1;
        let alert = ResourceAlert {
            id: format!("{}#{}", resource_type, self.next_alert_id),
            timestamp: Instant::now(),
            severity: severity.clone(),
            resource_type: resource_type.to_string(),
            current_value,
            threshold_value,
            message: format!(
                "{} usage is {:.2} (threshold: {:.2})",
                resource_type, current_value, threshold_value
            ),
            suggested_actions: self.generate_suggested_actions(resource_type, &severity),
            auto_resolution_attempts: 0,
        };
        self.handle_alert(alert)
    }

    /// Clear the active alert for a resource that dropped back below its
    /// recovery threshold, moving it into the history (R5).
    fn clear(&mut self, resource_type: &str) {
        let Some(position) = self
            .active_alerts
            .iter()
            .position(|alert| alert.resource_type == resource_type)
        else {
            return;
        };
        if let Some(alert) = self.active_alerts.remove(position) {
            if self.alert_history.len() >= MAX_ALERT_HISTORY {
                self.alert_history.pop_front();
            }
            self.alert_history.push_back(alert);
        }
    }

    /// Threshold evaluation kept as a pure query for callers that only want to
    /// know what *would* fire.
    /// Test-only: returns the alerts the current usage *would* raise without
    /// mutating the active/​historical alert state. Production code calls
    /// [`Self::update`], which raises and clears alerts for real.
    #[cfg(test)]
    pub(crate) fn check_thresholds(
        &mut self,
        usage: &ResourceUsage,
    ) -> Result<Vec<ResourceAlert>, String> {
        let mut alerts = Vec::new();
        for (resource, value) in observed_values(usage) {
            let thresholds = self.thresholds_for(&resource);
            if let Some(alert) = self.check_threshold(&resource, value, &thresholds)? {
                alerts.push(alert);
            }
        }
        Ok(alerts)
    }

    #[cfg(test)]
    fn check_threshold(
        &mut self,
        resource_type: &str,
        current_value: f64,
        thresholds: &ThresholdSet,
    ) -> Result<Option<ResourceAlert>, String> {
        let Some(severity) = severity_for(current_value, thresholds) else {
            return Ok(None);
        };
        let threshold_value = threshold_for(severity.clone(), thresholds);
        let suggested_actions = self.generate_suggested_actions(resource_type, &severity);

        self.next_alert_id += 1;
        Ok(Some(ResourceAlert {
            id: format!("{}#{}", resource_type, self.next_alert_id),
            timestamp: Instant::now(),
            severity,
            resource_type: resource_type.to_string(),
            current_value,
            threshold_value,
            message: format!(
                "{} usage is {:.2} (threshold: {:.2})",
                resource_type, current_value, threshold_value
            ),
            suggested_actions,
            auto_resolution_attempts: 0,
        }))
    }

    fn generate_suggested_actions(
        &self,
        resource_type: &str,
        severity: &AlertSeverity,
    ) -> Vec<String> {
        match (resource_type, severity) {
            ("memory", AlertSeverity::Critical | AlertSeverity::Emergency) => vec![
                "Reduce buffer sizes".to_string(),
                "Clear caches".to_string(),
                "Reduce batch sizes".to_string(),
            ],
            ("memory", AlertSeverity::Warning) => vec![
                "Monitor memory usage trends".to_string(),
                "Consider reducing buffer sizes".to_string(),
            ],
            ("cpu", AlertSeverity::Critical | AlertSeverity::Emergency) => vec![
                "Reduce processing frequency".to_string(),
                "Lower thread count".to_string(),
                "Defer non-critical operations".to_string(),
            ],
            ("cpu", AlertSeverity::Warning) => vec![
                "Monitor CPU usage patterns".to_string(),
                "Consider load balancing".to_string(),
            ],
            ("network", AlertSeverity::Critical | AlertSeverity::Emergency) => vec![
                "Enable traffic shaping".to_string(),
                "Batch outbound updates".to_string(),
            ],
            ("network", AlertSeverity::Warning) => {
                vec!["Monitor bandwidth utilization".to_string()]
            }
            _ => vec!["Monitor resource usage".to_string()],
        }
    }

    pub(crate) fn handle_alert(&mut self, alert: ResourceAlert) -> Result<(), String> {
        // Add to active alerts
        self.active_alerts.push_back(alert.clone());

        // Notify handlers
        for handler in &self.alert_handlers {
            if handler.can_handle(&alert) {
                handler.handle_alert(&alert)?;
            }
        }

        Ok(())
    }
}

/// The measured values available for threshold evaluation. Unmeasured
/// quantities are simply absent, so an unknown CPU reading can never be
/// mistaken for an idle one.
fn observed_values(usage: &ResourceUsage) -> Vec<(String, f64)> {
    let mut observed = Vec::new();
    if let Some(memory_percent) = usage.memory_usage_percent() {
        observed.push(("memory".to_string(), memory_percent));
    }
    if let Some(cpu) = usage.cpu_usage() {
        observed.push(("cpu".to_string(), cpu));
    }
    if let Some(network) = usage.network_io_mbps {
        observed.push(("network".to_string(), network));
    }
    observed
}

fn severity_for(value: f64, thresholds: &ThresholdSet) -> Option<AlertSeverity> {
    if value >= thresholds.emergency {
        Some(AlertSeverity::Emergency)
    } else if value >= thresholds.critical {
        Some(AlertSeverity::Critical)
    } else if value >= thresholds.warning {
        Some(AlertSeverity::Warning)
    } else {
        None
    }
}

fn threshold_for(severity: AlertSeverity, thresholds: &ThresholdSet) -> f64 {
    match severity {
        AlertSeverity::Emergency => thresholds.emergency,
        AlertSeverity::Critical => thresholds.critical,
        _ => thresholds.warning,
    }
}

/// The budget the manager would build from `config`, used by
/// [`ResourceAlertSystem::new`] so the standalone constructor derives the same
/// thresholds as the configured one.
#[cfg(test)]
fn default_budget(config: &ResourceConfig) -> ResourceBudget {
    use super::{
        BudgetEnforcementStrategy, CpuBudget, DeadlineEnforcement, MemoryBudget, MemoryPriority,
        NetworkBudget, QoSSettings, ThreadPriorityConfig, TimeBudget,
    };
    let constraints = config.budget_constraints.clone();
    let available_cpus = num_cpus::get().max(1);
    ResourceBudget {
        memory_budget: MemoryBudget {
            max_allocation_mb: config.max_memory_mb,
            soft_limit_mb: ((config.max_memory_mb as f64 * 0.8) as usize)
                .min(constraints.memory_budget_mb.max(1)),
            cleanup_threshold: config.cleanup_threshold,
            enable_compression: true,
            priority_levels: vec![MemoryPriority::Critical, MemoryPriority::Normal],
        },
        cpu_budget: CpuBudget {
            max_utilization: config.max_cpu_percent,
            target_utilization: constraints.cpu_budget_percent.min(config.max_cpu_percent),
            max_threads: available_cpus,
            thread_priority: ThreadPriorityConfig {
                high_priority_threads: available_cpus.min(2),
                normal_priority_threads: available_cpus.saturating_sub(available_cpus.min(2)),
                background_threads: 1,
                dynamic_priority: true,
            },
            cpu_affinity: None,
        },
        network_budget: NetworkBudget {
            max_bandwidth_mbps: 100.0,
            priority_allocation: std::collections::HashMap::new(),
            enable_traffic_shaping: false,
            qos_settings: QoSSettings {
                max_latency_ms: 100,
                jitter_tolerance_ms: 10,
                packet_loss_tolerance: 0.1,
                traffic_classes: Vec::new(),
            },
        },
        time_budget: TimeBudget {
            max_batch_processing_time: constraints.time_budget.saturating_mul(3),
            target_batch_processing_time: constraints.time_budget,
            operation_timeout: constraints.time_budget,
            deadline_enforcement: DeadlineEnforcement::Soft,
        },
        enforcement_strategy: BudgetEnforcementStrategy::Adaptive,
        flexibility: 0.2,
    }
}
