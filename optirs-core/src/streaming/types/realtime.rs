//! Real-time performance optimizer (T11).
//!
//! The previous `optimize_realtime` ignored its latency budget entirely and
//! returned the same fabricated triple — "applied, 1.2x faster, 5ms saved" —
//! on every call, regardless of whether anything had been observed, changed,
//! or improved. This module replaces it with a controller that only reports
//! what it has actually measured.

use crate::error::Result;
use std::time::Duration;

use super::primitives::{RTOptimizationResult, RTOptimizationState, RealTimeMetrics};
use super::types_3::RealTimeConfig;

/// Highest optimization level the controller will escalate to.
pub(super) const MAX_RT_OPTIMIZATION_LEVEL: u8 = 3;

/// Latency observations required at a level before its effect can be judged
/// (and before the controller is allowed to move again).
pub(super) const RT_OBSERVATIONS_PER_LEVEL: usize = 8;

/// Smoothing factor for the measured average processing time.
const RT_LATENCY_EWMA_ALPHA: f64 = 0.2;

/// Deadline-miss rate above which the controller escalates.
const RT_MISS_RATE_ESCALATION: f64 = 0.1;

/// Fraction of the deadline that must be left unused before the controller
/// relaxes back down a level.
const RT_RELAX_HEADROOM: f64 = 0.5;

/// Real-time performance optimizer.
///
/// Drives a single discrete knob — [`RTOptimizationState::optimization_level`]
/// — from *measured* latency behaviour: it escalates while the observed
/// deadline-miss rate is too high, relaxes while there is ample headroom, and
/// reports the improvement it can actually attribute to its last move by
/// comparing the current measured average against the average recorded at the
/// moment that move was made.
pub struct RealTimeOptimizer {
    /// Configuration
    pub(super) config: RealTimeConfig,
    /// Performance metrics
    pub(super) performance_metrics: RealTimeMetrics,
    /// Optimization state
    pub(super) optimization_state: RTOptimizationState,
    /// Total latency observations recorded.
    pub(super) observations: usize,
    /// Observations recorded since the current level was entered.
    pub(super) observations_at_level: usize,
    /// Deadline misses since the current level was entered.
    pub(super) misses_at_level: usize,
    /// Measured average processing time (us) at the moment the current level
    /// was entered; `None` until the first level change.
    pub(super) baseline_avg_us: Option<f64>,
}

impl RealTimeOptimizer {
    pub fn new(config: RealTimeConfig) -> Result<Self> {
        let cpu_affinity_mask = config
            .cpu_affinity
            .as_ref()
            .map(|cores| {
                cores.iter().fold(0u64, |mask, &core| {
                    if core < 64 {
                        mask | (1u64 << core)
                    } else {
                        mask
                    }
                })
            })
            .unwrap_or(0);
        Ok(Self {
            optimization_state: RTOptimizationState {
                current_priority: config.scheduling_priority,
                cpu_affinity_mask,
                memory_pools: Vec::new(),
                optimization_level: 0,
            },
            config,
            performance_metrics: RealTimeMetrics::default(),
            observations: 0,
            observations_at_level: 0,
            misses_at_level: 0,
            baseline_avg_us: None,
        })
    }

    /// Record one real end-to-end processing latency.
    ///
    /// This is what turns the optimizer from a constructed-and-forgotten
    /// object into a controller: without observations it has nothing to
    /// optimize, and [`Self::optimize_realtime`] honestly reports "nothing
    /// applied, nothing measured".
    pub fn observe_latency(&mut self, latency: Duration) {
        let micros = latency.as_secs_f64() * 1.0e6;
        if !micros.is_finite() {
            return;
        }
        if self.observations == 0 {
            self.performance_metrics.avg_processing_time_us = micros;
        } else {
            self.performance_metrics.avg_processing_time_us =
                self.performance_metrics.avg_processing_time_us * (1.0 - RT_LATENCY_EWMA_ALPHA)
                    + micros * RT_LATENCY_EWMA_ALPHA;
        }
        if micros > self.performance_metrics.worst_case_latency_us {
            self.performance_metrics.worst_case_latency_us = micros;
        }
        if micros > self.config.deadline_us as f64 {
            self.performance_metrics.deadline_misses += 1;
            self.misses_at_level += 1;
        }
        self.observations += 1;
        self.observations_at_level += 1;
    }

    /// Record the real resource utilisation observed by the caller.
    pub fn observe_utilization(&mut self, cpu_utilization: f64, memory_pressure: f64) {
        if cpu_utilization.is_finite() {
            self.performance_metrics.cpu_utilization = cpu_utilization;
        }
        if memory_pressure.is_finite() {
            self.performance_metrics.memory_pressure = memory_pressure;
        }
    }

    /// Measured real-time metrics.
    pub fn performance_metrics(&self) -> &RealTimeMetrics {
        &self.performance_metrics
    }

    /// Current optimization state.
    pub fn optimization_state(&self) -> &RTOptimizationState {
        &self.optimization_state
    }

    /// Number of latency observations recorded so far.
    pub fn observation_count(&self) -> usize {
        self.observations
    }

    /// Re-evaluate the real-time optimization level against `latency_budget`.
    ///
    /// The returned figures are measurements, not estimates:
    ///
    /// * `optimization_applied` is true only when this call actually changed
    ///   the optimization level.
    /// * `performance_gain` is `baseline_avg / current_avg`, where the
    ///   baseline is the measured average at the moment the current level was
    ///   entered — exactly 1.0 while nothing has been changed yet.
    /// * `latency_reduction_ms` is the corresponding measured difference, and
    ///   is negative if the last change made things worse.
    pub fn optimize_realtime(&mut self, latency_budget: Duration) -> Result<RTOptimizationResult> {
        let budget_us = {
            let from_argument = latency_budget.as_secs_f64() * 1.0e6;
            let configured = self.config.deadline_us as f64;
            if from_argument > 0.0 && from_argument.is_finite() {
                from_argument.min(configured)
            } else {
                configured
            }
        };
        if self.observations == 0 || budget_us <= 0.0 {
            // Nothing measured yet: report honestly rather than inventing a
            // gain that was never observed.
            return Ok(RTOptimizationResult {
                optimization_applied: false,
                performance_gain: 1.0,
                latency_reduction_ms: 0.0,
            });
        }

        let current_avg_us = self.performance_metrics.avg_processing_time_us;
        let (performance_gain, latency_reduction_ms) = match self.baseline_avg_us {
            Some(baseline) if current_avg_us > 0.0 && baseline > 0.0 => (
                baseline / current_avg_us,
                (baseline - current_avg_us) / 1000.0,
            ),
            _ => (1.0, 0.0),
        };

        let current_level = self.optimization_state.optimization_level;
        let mut new_level = current_level;
        if self.observations_at_level >= RT_OBSERVATIONS_PER_LEVEL {
            let miss_rate = self.misses_at_level as f64 / self.observations_at_level as f64;
            let headroom = (budget_us - current_avg_us) / budget_us;
            if miss_rate > RT_MISS_RATE_ESCALATION || current_avg_us > budget_us {
                new_level = current_level
                    .saturating_add(1)
                    .min(MAX_RT_OPTIMIZATION_LEVEL);
            } else if miss_rate == 0.0 && headroom > RT_RELAX_HEADROOM {
                new_level = current_level.saturating_sub(1);
            }
        }

        let optimization_applied = new_level != current_level;
        if optimization_applied {
            self.optimization_state.optimization_level = new_level;
            // Both of the following are *recommendations* derived from real
            // configuration, not applied settings: setting an OS scheduling
            // priority or a CPU affinity mask requires platform FFI, which
            // this crate deliberately does not link. They are exposed so a
            // host that does have those privileges can act on them.
            self.optimization_state.current_priority =
                self.config.scheduling_priority + new_level as i32;
            self.optimization_state.memory_pools =
                Self::recommended_memory_pools(self.config.memory_preallocation_mb, new_level);
            self.baseline_avg_us = Some(current_avg_us);
            self.observations_at_level = 0;
            self.misses_at_level = 0;
        }

        Ok(RTOptimizationResult {
            optimization_applied,
            performance_gain,
            latency_reduction_ms,
        })
    }

    /// Split the configured pre-allocation budget into `level + 1` equal
    /// pools (in bytes). More aggressive levels favour more, smaller pools so
    /// that a burst can be served without touching the allocator.
    fn recommended_memory_pools(preallocation_mb: usize, level: u8) -> Vec<usize> {
        let pools = level as usize + 1;
        let total_bytes = preallocation_mb.saturating_mul(1024 * 1024);
        let per_pool = total_bytes / pools.max(1);
        vec![per_pool; pools]
    }
}

#[cfg(test)]
mod realtime_optimizer_tests {
    use super::*;

    fn optimizer(deadline_us: u64) -> RealTimeOptimizer {
        RealTimeOptimizer::new(RealTimeConfig {
            deadline_us,
            ..Default::default()
        })
        .expect("construct")
    }

    /// T11: with nothing observed, the optimizer must say so. The previous
    /// implementation claimed a 1.2x gain and a 5ms latency reduction before
    /// a single sample had been processed.
    #[test]
    fn reports_no_gain_before_anything_has_been_measured() {
        let mut rt = optimizer(10_000);
        let result = rt
            .optimize_realtime(Duration::from_millis(10))
            .expect("optimize");
        assert!(
            !result.optimization_applied,
            "T11 regression: claimed an optimization was applied with zero observations"
        );
        assert_eq!(
            result.performance_gain, 1.0,
            "T11 regression: fabricated a performance gain with no measurements"
        );
        assert_eq!(
            result.latency_reduction_ms, 0.0,
            "T11 regression: fabricated a latency reduction with no measurements"
        );
    }

    /// A system comfortably inside its deadline must not be "optimized" at
    /// all: the level stays at zero and every reported figure stays neutral.
    #[test]
    fn healthy_latency_leaves_the_optimization_level_untouched() {
        let mut rt = optimizer(10_000);
        for _ in 0..64 {
            rt.observe_latency(Duration::from_micros(500));
            let result = rt
                .optimize_realtime(Duration::from_millis(10))
                .expect("optimize");
            assert!(
                !result.optimization_applied,
                "escalated despite 500us latency against a 10ms deadline"
            );
        }
        assert_eq!(rt.optimization_state().optimization_level, 0);
        assert_eq!(rt.performance_metrics().deadline_misses, 0);
    }

    /// Persistent deadline misses must escalate the level, and the reported
    /// gain must then be the *measured* effect of that escalation.
    #[test]
    fn deadline_misses_escalate_and_the_reported_gain_is_measured() {
        let mut rt = optimizer(1_000);

        // Phase 1: every batch blows the 1ms deadline.
        let mut escalated = false;
        for _ in 0..32 {
            rt.observe_latency(Duration::from_micros(4_000));
            let result = rt
                .optimize_realtime(Duration::from_millis(10))
                .expect("optimize");
            escalated |= result.optimization_applied;
        }
        assert!(
            escalated,
            "T11 regression: never escalated despite continuous deadline misses"
        );
        assert!(rt.optimization_state().optimization_level > 0);
        assert!(rt.performance_metrics().deadline_misses > 0);
        assert!(
            !rt.optimization_state().memory_pools.is_empty(),
            "escalating must produce a real pool recommendation"
        );

        // Phase 2: latency genuinely improves; the reported gain must reflect
        // the measured improvement over the baseline captured at escalation.
        for _ in 0..32 {
            rt.observe_latency(Duration::from_micros(200));
        }
        let result = rt
            .optimize_realtime(Duration::from_millis(10))
            .expect("optimize");
        assert!(
            result.performance_gain > 1.5,
            "measured gain should be large after latency dropped 20x, got {}",
            result.performance_gain
        );
        assert!(
            result.latency_reduction_ms > 0.0,
            "measured latency reduction should be positive, got {}",
            result.latency_reduction_ms
        );
    }

    /// A configured CPU affinity must be turned into a real mask rather than
    /// being left at the default zero.
    #[test]
    fn cpu_affinity_is_translated_into_a_real_mask() {
        let rt = RealTimeOptimizer::new(RealTimeConfig {
            cpu_affinity: Some(vec![0, 3, 5]),
            ..Default::default()
        })
        .expect("construct");
        assert_eq!(
            rt.optimization_state().cpu_affinity_mask,
            0b101001,
            "affinity mask was not derived from the configured core list"
        );
    }
}
