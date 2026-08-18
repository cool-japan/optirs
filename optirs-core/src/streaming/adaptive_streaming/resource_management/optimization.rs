// Resource optimizer with real constraint enforcement (finding R7).
//
// `OptimizationConstraints` (change-rate limits, minimum stable period,
// oscillation prevention, hysteresis) and `performance_impact` were never
// read: adaptations were emitted as fixed -20% / -15% magnitudes with no
// stability control at all.

use super::{
    lock_recovered, OptimizationConstraints, OptimizationEvent, ResourceAllocation,
    ResourceOptimizationStrategy, ResourceOptimizer, ResourceUsage, StabilityRequirements,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Default per-resource change-rate limit applied when the constraint map has
/// no entry for a resource.
const DEFAULT_CHANGE_RATE_LIMIT: f64 = 0.25;

impl ResourceOptimizer {
    pub(crate) fn new(strategy: ResourceOptimizationStrategy) -> Self {
        // The stability envelope depends on the strategy: an aggressive or
        // throughput-oriented optimizer is allowed to move faster and further.
        let (min_stable_period, max_change_frequency, change_rate_limit, hysteresis) =
            match strategy {
                ResourceOptimizationStrategy::Conservative => {
                    (Duration::from_secs(300), 0.02, 0.05, 0.05)
                }
                ResourceOptimizationStrategy::Aggressive
                | ResourceOptimizationStrategy::ThroughputOptimized => {
                    (Duration::from_secs(15), 0.5, 0.5, 0.01)
                }
                ResourceOptimizationStrategy::LatencyOptimized => {
                    (Duration::from_secs(10), 0.5, 0.35, 0.01)
                }
                ResourceOptimizationStrategy::PowerEfficient => {
                    (Duration::from_secs(120), 0.05, 0.1, 0.05)
                }
                ResourceOptimizationStrategy::Balanced => (
                    Duration::from_secs(60),
                    0.1,
                    DEFAULT_CHANGE_RATE_LIMIT,
                    0.02,
                ),
            };

        let mut change_rate_limits = HashMap::new();
        change_rate_limits.insert("memory".to_string(), change_rate_limit);
        change_rate_limits.insert("cpu".to_string(), change_rate_limit);
        change_rate_limits.insert("network".to_string(), change_rate_limit);

        Self {
            strategy,
            optimization_history: VecDeque::with_capacity(100),
            performance_impact: HashMap::new(),
            constraints: OptimizationConstraints {
                min_guarantees: HashMap::new(),
                max_limits: HashMap::new(),
                change_rate_limits,
                stability_requirements: StabilityRequirements {
                    min_stable_period,
                    max_change_frequency,
                    prevent_oscillation: true,
                    hysteresis_factor: hysteresis,
                },
            },
            last_change: HashMap::new(),
            pending_change: HashMap::new(),
        }
    }

    /// Clamp a requested relative change by the configured change-rate limit.
    pub(crate) fn clamp_change(&mut self, resource: &str, requested: f64) -> f64 {
        let limit = self
            .constraints
            .change_rate_limits
            .get(resource)
            .copied()
            .unwrap_or(DEFAULT_CHANGE_RATE_LIMIT)
            .abs();
        let clamped = requested.clamp(-limit, limit);
        self.pending_change.insert(resource.to_string(), clamped);
        clamped
    }

    /// Whether a change to `component` is allowed right now, honouring the
    /// minimum stable period, the hysteresis band and oscillation prevention.
    pub(crate) fn accept_change(&self, component: &str) -> bool {
        let requirements = &self.constraints.stability_requirements;
        let resource = component.split('_').next().unwrap_or(component);
        let magnitude = self.pending_change.get(resource).copied().unwrap_or(0.0);

        // Hysteresis: ignore changes too small to be worth the disruption.
        if magnitude.abs() < requirements.hysteresis_factor {
            return false;
        }

        match self.last_change.get(component) {
            None => true,
            Some((at, previous_magnitude)) => {
                let elapsed = at.elapsed();
                if elapsed < requirements.min_stable_period {
                    // Inside the stable period, a sign flip is exactly the
                    // oscillation the constraint exists to prevent.
                    if requirements.prevent_oscillation && magnitude * previous_magnitude < 0.0 {
                        return false;
                    }
                    // Otherwise limit how often changes may land at all.
                    let allowed_interval = if requirements.max_change_frequency > 0.0 {
                        Duration::from_secs_f64(1.0 / requirements.max_change_frequency)
                    } else {
                        requirements.min_stable_period
                    };
                    return elapsed >= allowed_interval;
                }
                true
            }
        }
    }

    /// Record that a change was applied to `component`.
    pub(crate) fn record_applied_change(&mut self, component: &str) {
        let resource = component.split('_').next().unwrap_or(component).to_string();
        let magnitude = self.pending_change.get(&resource).copied().unwrap_or(0.0);
        self.last_change
            .insert(component.to_string(), (Instant::now(), magnitude));
        let impact = self
            .performance_impact
            .entry(component.to_string())
            .or_insert(0.0);
        *impact += magnitude.abs();
        if let Some(event) = self.optimization_history.back_mut() {
            if event.affected_resources.contains(&resource) {
                event.success = true;
            }
        }
    }

    /// Accumulated absolute change applied per component.
    pub(crate) fn performance_impact(&self) -> &HashMap<String, f64> {
        &self.performance_impact
    }

    pub(crate) fn check_optimization_opportunities(
        &mut self,
        current_usage: &ResourceUsage,
        allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Check for optimization opportunities based on current strategy
        match self.strategy {
            ResourceOptimizationStrategy::Balanced
            | ResourceOptimizationStrategy::Aggressive
            | ResourceOptimizationStrategy::Conservative => {
                self.check_balanced_optimization(current_usage, allocations)?;
            }
            ResourceOptimizationStrategy::PowerEfficient => {
                self.check_power_optimization(current_usage, allocations)?;
            }
            ResourceOptimizationStrategy::LatencyOptimized
            | ResourceOptimizationStrategy::ThroughputOptimized => {
                self.check_latency_optimization(current_usage, allocations)?;
            }
        }

        Ok(())
    }

    fn record_event(
        &mut self,
        optimization_type: &str,
        affected: Vec<String>,
        deltas: HashMap<String, f64>,
        expected_impact: f64,
    ) {
        let event = OptimizationEvent {
            timestamp: Instant::now(),
            optimization_type: optimization_type.to_string(),
            affected_resources: affected,
            resource_deltas: deltas,
            performance_impact: expected_impact,
            success: false, // set by `record_applied_change`
        };
        if self.optimization_history.len() >= 100 {
            self.optimization_history.pop_front();
        }
        self.optimization_history.push_back(event);
    }

    fn check_balanced_optimization(
        &mut self,
        current_usage: &ResourceUsage,
        _allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Check for resource imbalances. R2 fix: this compares against
        // 80.0/40.0 as percentages, so it needs a real used/total ratio
        // (see `ResourceUsage::memory_usage_percent`), not `used_mb / 1024`
        // (which silently assumed exactly 1 GB of total system memory).
        // With no real total, and with CPU not yet measured, there is nothing
        // to compare and no event is recorded.
        let Some(memory_utilization) = current_usage.memory_usage_percent() else {
            return Ok(());
        };
        let Some(cpu_utilization) = current_usage.cpu_usage() else {
            return Ok(());
        };

        // If one resource is heavily utilized while others are underutilized, suggest rebalancing
        if (memory_utilization > 80.0 && cpu_utilization < 40.0)
            || (cpu_utilization > 80.0 && memory_utilization < 40.0)
        {
            let mut deltas = HashMap::new();
            deltas.insert("memory".to_string(), memory_utilization - cpu_utilization);
            deltas.insert("cpu".to_string(), cpu_utilization - memory_utilization);
            // The expected gain is proportional to the real imbalance rather
            // than a fixed 5%.
            let imbalance = (memory_utilization - cpu_utilization).abs() / 100.0;
            self.record_event(
                "resource_rebalancing",
                vec!["memory".to_string(), "cpu".to_string()],
                deltas,
                imbalance,
            );
        }

        Ok(())
    }

    fn check_power_optimization(
        &mut self,
        current_usage: &ResourceUsage,
        allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Power efficiency here means: idle capacity that is still allocated.
        // Any component whose allocation has not been touched for longer than
        // the minimum stable period, while the machine is not busy, is a real
        // reclaim candidate.
        let Some(cpu) = current_usage.cpu_usage() else {
            return Ok(());
        };
        if cpu > 25.0 {
            return Ok(());
        }
        let stale_period = self.constraints.stability_requirements.min_stable_period;
        let idle: Vec<String> = {
            let guard = lock_recovered(allocations);
            guard
                .values()
                .filter(|allocation| allocation.last_access.elapsed() > stale_period)
                .map(|allocation| allocation.component_name.clone())
                .collect()
        };
        if idle.is_empty() {
            return Ok(());
        }
        let mut deltas = HashMap::new();
        for component in &idle {
            deltas.insert(component.clone(), -1.0);
        }
        let expected = (idle.len() as f64 / 10.0).min(1.0);
        self.record_event("idle_reclaim", idle, deltas, expected);
        Ok(())
    }

    fn check_latency_optimization(
        &mut self,
        current_usage: &ResourceUsage,
        allocations: &Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    ) -> Result<(), String> {
        // Latency suffers once the machine saturates, so the opportunity here
        // is to shed the lowest-priority allocations before that happens.
        let Some(cpu) = current_usage.cpu_usage() else {
            return Ok(());
        };
        if cpu < 75.0 {
            return Ok(());
        }
        let sheddable: Vec<String> = {
            let guard = lock_recovered(allocations);
            guard
                .values()
                .filter(|allocation| allocation.priority >= super::ResourcePriority::Low)
                .map(|allocation| allocation.component_name.clone())
                .collect()
        };
        if sheddable.is_empty() {
            return Ok(());
        }
        let mut deltas = HashMap::new();
        for component in &sheddable {
            deltas.insert(component.clone(), -(cpu - 75.0) / 100.0);
        }
        self.record_event(
            "latency_load_shedding",
            sheddable,
            deltas,
            (cpu - 75.0) / 100.0,
        );
        Ok(())
    }
}
