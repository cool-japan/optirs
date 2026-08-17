//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(dead_code)]
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use super::functions::to_a_or;
use super::primitives::{
    FusedOptimizationStep, FusionStrategy, QoSMetric, ResourceAllocation, ResourceConstraints,
    ResourceReservationStrategy, ResourceUsage, ServiceLevelObjective, StreamPriority,
    StreamingConfig, StreamingMetrics,
};
use super::types_2::ResourceAllocationStrategy;

/// Maximum number of fused steps retained for inspection.
const MAX_FUSION_HISTORY: usize = 100;

/// Stream fusion optimizer for combining multiple optimization streams
pub struct StreamFusionOptimizer<A: Float + Send + Sync> {
    /// Fusion strategy
    pub(super) fusion_strategy: FusionStrategy,
    /// Stream weights for weighted fusion
    pub(super) stream_weights: HashMap<String, A>,
    /// Fusion buffer
    pub(super) fusion_buffer: VecDeque<FusedOptimizationStep<A>>,
    /// Consensus mechanism
    pub(super) consensus_mechanism: ConsensusAlgorithm,
}
impl<
        A: Float
            + std::ops::DivAssign
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + Send
            + Sync,
    > StreamFusionOptimizer<A>
{
    pub fn new(config: &StreamingConfig) -> Result<Self> {
        Ok(Self {
            fusion_strategy: FusionStrategy::WeightedAverage,
            stream_weights: HashMap::new(),
            fusion_buffer: VecDeque::with_capacity(config.buffer_size),
            consensus_mechanism: ConsensusAlgorithm::MajorityVoting,
        })
    }
    /// Select the strategy used to combine per-stream steps.
    pub fn set_fusion_strategy(&mut self, strategy: FusionStrategy) {
        self.fusion_strategy = strategy;
    }

    /// The strategy currently in force.
    pub fn fusion_strategy(&self) -> FusionStrategy {
        self.fusion_strategy
    }

    /// Select the consensus algorithm used by
    /// [`FusionStrategy::ConsensusBased`].
    pub fn set_consensus_mechanism(&mut self, mechanism: ConsensusAlgorithm) {
        self.consensus_mechanism = mechanism;
    }

    /// Replace the per-stream weights used by the weighted strategies.
    pub fn set_stream_weights(&mut self, weights: HashMap<String, A>) {
        self.stream_weights = weights;
    }

    /// Weight currently assigned to a stream, if any.
    pub fn stream_weight(&self, stream_id: &str) -> Option<A> {
        self.stream_weights.get(stream_id).copied()
    }

    /// Steps fused so far, most recent last (capped).
    pub fn fusion_history(&self) -> &VecDeque<FusedOptimizationStep<A>> {
        &self.fusion_buffer
    }

    /// Fuse optimization steps from multiple streams
    pub fn fuse_optimization_steps(&mut self, steps: &[(String, Array1<A>)]) -> Result<Array1<A>> {
        if steps.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No optimization steps to fuse".to_string(),
            ));
        }
        let fused = match self.fusion_strategy {
            FusionStrategy::WeightedAverage => {
                let mut fused_step = Array1::zeros(steps[0].1.len());
                let mut total_weight = A::zero();
                for (stream_id, step) in steps {
                    let weight = self
                        .stream_weights
                        .get(stream_id)
                        .copied()
                        .unwrap_or(A::one());
                    fused_step = fused_step + step * weight;
                    total_weight = total_weight + weight;
                }
                if total_weight > A::zero() {
                    fused_step /= total_weight;
                }
                Ok(fused_step)
            }
            FusionStrategy::MedianFusion => Self::coordinate_wise_median(steps),
            FusionStrategy::ConsensusBased => self.apply_consensus(steps),
            FusionStrategy::AdaptiveFusion => {
                // Adaptive fusion re-derives the weights from how far each
                // stream sits from the robust (median) consensus, so a stream
                // that disagrees with the majority loses influence instead of
                // being averaged in as an equal.
                let robust = Self::coordinate_wise_median(steps)?;
                let mut fused_step = Array1::zeros(steps[0].1.len());
                let mut total_weight = A::zero();
                for (stream_id, step) in steps {
                    let configured = self
                        .stream_weights
                        .get(stream_id)
                        .copied()
                        .unwrap_or(A::one());
                    let deviation = step
                        .iter()
                        .zip(robust.iter())
                        .fold(A::zero(), |acc, (&value, &reference)| {
                            let difference = value - reference;
                            acc + difference * difference
                        })
                        .sqrt();
                    let weight = configured / (A::one() + deviation);
                    fused_step = fused_step + step * weight;
                    total_weight = total_weight + weight;
                }
                if total_weight > A::zero() {
                    fused_step /= total_weight;
                }
                Ok(fused_step)
            }
        }?;
        // Record the fused step together with a real agreement measure: the
        // mean distance of the contributing proposals from the fused result,
        // normalised into a `[0, 1]` confidence.
        let mut dispersion = A::zero();
        for (_, step) in steps {
            let distance = step
                .iter()
                .zip(fused.iter())
                .fold(A::zero(), |acc, (&value, &reference)| {
                    let difference = value - reference;
                    acc + difference * difference
                })
                .sqrt();
            dispersion = dispersion + distance;
        }
        let count = to_a_or(steps.len() as f64, A::one());
        let confidence = A::one() / (A::one() + dispersion / count);
        self.fusion_buffer.push_back(FusedOptimizationStep {
            step: fused.clone(),
            confidence,
            contributing_streams: steps.iter().map(|(id, _)| id.clone()).collect(),
            timestamp: Instant::now(),
        });
        while self.fusion_buffer.len() > MAX_FUSION_HISTORY {
            self.fusion_buffer.pop_front();
        }
        Ok(fused)
    }
    /// Validate that every stream's step has the same dimensionality,
    /// returning it. Shared by the fusion/consensus reducers below so a
    /// mismatched stream produces an honest error instead of an
    /// out-of-bounds panic or silently-wrong fusion.
    pub(super) fn checked_step_dim(steps: &[(String, Array1<A>)]) -> Result<usize> {
        let (first_id, first_step) = steps.first().ok_or_else(|| {
            OptimError::InvalidConfig("No optimization steps to fuse".to_string())
        })?;
        let dim = first_step.len();
        for (stream_id, step) in steps {
            if step.len() != dim {
                return Err(OptimError::InvalidConfig(format!(
                    "stream '{stream_id}' step has dimension {} but stream '{first_id}' has {dim}",
                    step.len()
                )));
            }
        }
        Ok(dim)
    }
    /// T9: coordinate-wise median across all streams' proposed steps.
    /// Robust to a minority of outlier ("faulty") streams, which is the
    /// property `FusionStrategy::MedianFusion` and
    /// `ConsensusAlgorithm::MajorityVoting` are meant to provide — the
    /// previous implementation just returned `steps[0]` unconditionally.
    pub(super) fn coordinate_wise_median(steps: &[(String, Array1<A>)]) -> Result<Array1<A>> {
        let dim = Self::checked_step_dim(steps)?;
        let mut fused = Array1::zeros(dim);
        let mut column: Vec<A> = Vec::with_capacity(steps.len());
        let two = to_a_or(2.0, A::one());
        for i in 0..dim {
            column.clear();
            column.extend(steps.iter().map(|(_, step)| step[i]));
            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = column.len();
            fused[i] = if n % 2 == 1 {
                column[n / 2]
            } else {
                (column[n / 2 - 1] + column[n / 2]) / two
            };
        }
        Ok(fused)
    }
    /// Coordinate-wise trimmed mean: drop the smallest and largest `f`
    /// contributions per coordinate before averaging, where `f` is the
    /// standard Byzantine safety bound `f < n / 3`. This is the actual
    /// aggregation rule PBFT/Byzantine-tolerant *distributed optimization*
    /// uses to stay correct under up to `f` faulty proposals; a literal
    /// wire-protocol replay isn't meaningful here since `steps` are
    /// already-collected local proposals, not independent network peers.
    pub(super) fn coordinate_wise_trimmed_mean(steps: &[(String, Array1<A>)]) -> Result<Array1<A>> {
        let dim = Self::checked_step_dim(steps)?;
        let n = steps.len();
        let trim = n.saturating_sub(1) / 3;
        let mut fused = Array1::zeros(dim);
        let mut column: Vec<A> = Vec::with_capacity(n);
        for i in 0..dim {
            column.clear();
            column.extend(steps.iter().map(|(_, step)| step[i]));
            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let kept = &column[trim..n - trim];
            let sum = kept.iter().copied().fold(A::zero(), |acc, v| acc + v);
            fused[i] = sum / to_a_or(kept.len() as f64, A::one());
        }
        Ok(fused)
    }
    /// Leader-based (Raft-style) consensus: defer to the proposal from the
    /// most-trusted ("elected leader") stream, approximated here as the
    /// stream with the highest configured weight.
    pub(super) fn leader_step(&self, steps: &[(String, Array1<A>)]) -> Result<Array1<A>> {
        Self::checked_step_dim(steps)?;
        steps
            .iter()
            .max_by(|(id_a, _), (id_b, _)| {
                let wa = self.stream_weights.get(id_a).copied().unwrap_or(A::zero());
                let wb = self.stream_weights.get(id_b).copied().unwrap_or(A::zero());
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, step)| step.clone())
            .ok_or_else(|| OptimError::InvalidConfig("No optimization steps to fuse".to_string()))
    }
    pub(super) fn apply_consensus(&self, steps: &[(String, Array1<A>)]) -> Result<Array1<A>> {
        match self.consensus_mechanism {
            ConsensusAlgorithm::MajorityVoting => Self::coordinate_wise_median(steps),
            ConsensusAlgorithm::PBFT | ConsensusAlgorithm::Byzantine => {
                Self::coordinate_wise_trimmed_mean(steps)
            }
            ConsensusAlgorithm::Raft => self.leader_step(steps),
        }
    }
}
/// Real-time optimization configuration
#[derive(Debug, Clone)]
pub struct RealTimeConfig {
    /// Real-time scheduling priority
    pub scheduling_priority: i32,
    /// CPU affinity mask
    pub cpu_affinity: Option<Vec<usize>>,
    /// Memory pre-allocation size
    pub memory_preallocation_mb: usize,
    /// Enable NUMA optimization
    pub numa_optimization: bool,
    /// Real-time deadline (microseconds)
    pub deadline_us: u64,
    /// Enable lock-free data structures
    pub lock_free_structures: bool,
    /// Interrupt handling strategy
    pub interrupt_strategy: InterruptStrategy,
}
/// Advanced Quality of Service configuration
#[derive(Debug, Clone)]
pub struct AdvancedQoSConfig {
    /// Strict latency guarantees
    pub strict_latency_bounds: bool,
    /// Quality degradation tolerance
    pub quality_degradation_tolerance: f64,
    /// Resource reservation strategy
    pub resource_reservation: ResourceReservationStrategy,
    /// Adaptive QoS adjustment
    pub adaptive_adjustment: bool,
    /// Priority-based scheduling
    pub priority_scheduling: bool,
    /// Service level objectives
    pub service_level_objectives: Vec<ServiceLevelObjective>,
}
/// Health status of streaming optimizer
#[derive(Debug, Clone)]
pub struct StreamingHealthStatus {
    pub is_healthy: bool,
    pub warnings: Vec<String>,
    pub metrics: StreamingMetrics,
}
/// Consensus algorithms for distributed optimization
#[derive(Debug, Clone, Copy)]
pub enum ConsensusAlgorithm {
    MajorityVoting,
    PBFT,
    Raft,
    Byzantine,
}
/// Maximum number of QoS violations retained for inspection.
const MAX_QOS_VIOLATION_HISTORY: usize = 1000;

/// Smoothing factor for the adaptive (spike-suppressing) QoS thresholds.
const QOS_EWMA_ALPHA: f64 = 0.2;

/// One round of observations to evaluate the configured service level
/// objectives against.
///
/// `None` means *not measured* — the corresponding objective is then skipped
/// rather than being assumed satisfied, so a disabled subsystem never
/// silently produces a clean bill of health.
#[derive(Debug, Clone, Default)]
pub struct QoSObservation {
    /// Average end-to-end latency (ms).
    pub latency_ms: f64,
    /// Achieved throughput (samples/second).
    pub throughput_samples_per_sec: f64,
    /// Memory in use (MB).
    pub memory_usage_mb: f64,
    /// Measured CPU duty cycle of the streaming path (percent).
    pub cpu_utilization_percent: Option<f64>,
    /// Measured accuracy of the predictive engine in `[0, 1]`.
    pub prediction_accuracy: Option<f64>,
    /// Measured inter-stream synchronization skew (ms).
    pub stream_synchronization_delay_ms: Option<f64>,
}

impl QoSObservation {
    /// Build an observation from the streaming metrics alone; the fields the
    /// metrics cannot speak to are left unmeasured.
    pub fn from_metrics(metrics: &StreamingMetrics) -> Self {
        Self {
            latency_ms: metrics.avg_latency_ms,
            throughput_samples_per_sec: metrics.processing_rate,
            memory_usage_mb: metrics.memory_usage_mb,
            cpu_utilization_percent: None,
            prediction_accuracy: None,
            stream_synchronization_delay_ms: None,
        }
    }
}

/// Advanced QoS manager for quality of service guarantees.
pub struct AdvancedQoSManager {
    /// QoS configuration
    pub(super) config: AdvancedQoSConfig,
    /// Current QoS status
    pub(super) current_status: QoSStatus,
    /// QoS violation history
    pub(super) violation_history: VecDeque<QoSViolation>,
    /// Smoothed value of every observed metric, keyed by metric name. Used
    /// when `config.adaptive_adjustment` is enabled to suppress violations
    /// caused by a single outlying sample.
    pub(super) adaptive_thresholds: HashMap<String, f64>,
}
impl AdvancedQoSManager {
    pub fn new(config: AdvancedQoSConfig) -> Self {
        Self {
            config,
            current_status: QoSStatus {
                is_compliant: true,
                violations: Vec::new(),
                timestamp: Instant::now(),
            },
            violation_history: VecDeque::with_capacity(MAX_QOS_VIOLATION_HISTORY),
            adaptive_thresholds: HashMap::new(),
        }
    }

    /// Evaluate the configured service level objectives against `observation`.
    ///
    /// The thresholds now come from `config.service_level_objectives` — the
    /// objectives the caller actually configured — instead of the three
    /// hardcoded constants (50ms, 100MB, 10 violations) the previous
    /// implementation compared against no matter what was asked for. The
    /// tolerance band around each target is widened by
    /// `config.quality_degradation_tolerance` and collapsed entirely when
    /// `config.strict_latency_bounds` is set.
    pub fn monitor_qos(&mut self, observation: &QoSObservation) -> QoSStatus {
        let mut violations = Vec::new();
        let objectives = self.config.service_level_objectives.clone();
        for slo in &objectives {
            let (key, observed) = match slo.metric {
                QoSMetric::Latency => ("latency_ms", Some(observation.latency_ms)),
                QoSMetric::Throughput => (
                    "throughput_samples_per_sec",
                    Some(observation.throughput_samples_per_sec),
                ),
                QoSMetric::MemoryUsage => ("memory_usage_mb", Some(observation.memory_usage_mb)),
                QoSMetric::CpuUtilization => (
                    "cpu_utilization_percent",
                    observation.cpu_utilization_percent,
                ),
                QoSMetric::PredictionAccuracy => {
                    ("prediction_accuracy", observation.prediction_accuracy)
                }
                QoSMetric::StreamSynchronization => (
                    "stream_synchronization_delay_ms",
                    observation.stream_synchronization_delay_ms,
                ),
            };
            let Some(observed) = observed else {
                // Not measured this round: an objective that cannot be
                // evaluated must not be reported as met.
                continue;
            };
            if !observed.is_finite() {
                continue;
            }
            let smoothed = self.smooth(key, observed);
            let tolerance = if self.config.strict_latency_bounds {
                0.0
            } else {
                slo.tolerance.max(self.config.quality_degradation_tolerance)
            };
            // "Lower is better" for latency, memory and synchronization delay;
            // "higher is better" for throughput, utilization and accuracy.
            let lower_is_better = matches!(
                slo.metric,
                QoSMetric::Latency | QoSMetric::MemoryUsage | QoSMetric::StreamSynchronization
            );
            let breached = |value: f64| {
                if lower_is_better {
                    value > slo.target_value * (1.0 + tolerance)
                } else {
                    value < slo.target_value * (1.0 - tolerance)
                }
            };
            // With adaptive adjustment on, a single outlying sample is not a
            // violation: the smoothed trend has to agree.
            let violated = if self.config.adaptive_adjustment {
                breached(observed) && breached(smoothed)
            } else {
                breached(observed)
            };
            if !violated {
                continue;
            }
            violations.push(match slo.metric {
                QoSMetric::Latency => QoSViolation::LatencyExceeded {
                    actual: observed,
                    target: slo.target_value,
                },
                QoSMetric::MemoryUsage => QoSViolation::MemoryExceeded {
                    actual: observed,
                    target: slo.target_value,
                },
                QoSMetric::Throughput => QoSViolation::ThroughputDegraded {
                    violation_rate: if slo.target_value > 0.0 {
                        ((slo.target_value - observed) / slo.target_value).clamp(0.0, 1.0)
                    } else {
                        0.0
                    },
                },
                QoSMetric::CpuUtilization => QoSViolation::ResourceUtilizationLow {
                    utilization: observed,
                    target: slo.target_value,
                },
                QoSMetric::PredictionAccuracy => QoSViolation::PredictionAccuracyDegraded {
                    current: observed,
                    target: slo.target_value,
                },
                QoSMetric::StreamSynchronization => {
                    QoSViolation::StreamSynchronizationLoss { delay_ms: observed }
                }
            });
        }

        for violation in &violations {
            self.violation_history.push_back(violation.clone());
            if self.violation_history.len() > MAX_QOS_VIOLATION_HISTORY {
                self.violation_history.pop_front();
            }
        }
        self.current_status = QoSStatus {
            is_compliant: violations.is_empty(),
            violations,
            timestamp: Instant::now(),
        };
        self.current_status.clone()
    }

    fn smooth(&mut self, key: &str, observed: f64) -> f64 {
        let entry = self.adaptive_thresholds.entry(key.to_string());
        let smoothed = entry
            .and_modify(|value| {
                *value = *value * (1.0 - QOS_EWMA_ALPHA) + observed * QOS_EWMA_ALPHA
            })
            .or_insert(observed);
        *smoothed
    }

    /// The most recent evaluated status.
    pub fn current_status(&self) -> &QoSStatus {
        &self.current_status
    }

    /// Every violation observed so far (most recent last, capped).
    pub fn violation_history(&self) -> &VecDeque<QoSViolation> {
        &self.violation_history
    }

    /// Smoothed value of each observed metric.
    pub fn smoothed_metrics(&self) -> &HashMap<String, f64> {
        &self.adaptive_thresholds
    }

    /// A scalar summary of how far outside its objectives the system is,
    /// used to drive resource re-allocation: `0.0` when compliant, and
    /// growing with the relative size of each breach.
    pub fn pressure(&self) -> f64 {
        let mut pressure: f64 = 0.0;
        for violation in &self.current_status.violations {
            let relative = match violation {
                QoSViolation::LatencyExceeded { actual, target }
                | QoSViolation::MemoryExceeded { actual, target } => {
                    if *target > 0.0 {
                        (actual - target) / target
                    } else {
                        1.0
                    }
                }
                QoSViolation::ThroughputDegraded { violation_rate } => *violation_rate,
                QoSViolation::PredictionAccuracyDegraded { current, target } => {
                    if *target > 0.0 {
                        ((target - current) / target).max(0.0)
                    } else {
                        0.0
                    }
                }
                QoSViolation::ResourceUtilizationLow {
                    utilization,
                    target,
                } => {
                    if *target > 0.0 {
                        ((target - utilization) / target).max(0.0)
                    } else {
                        0.0
                    }
                }
                QoSViolation::StreamSynchronizationLoss { delay_ms } => {
                    (delay_ms / 1000.0).min(1.0)
                }
            };
            if relative.is_finite() {
                pressure += relative.max(0.0);
            }
        }
        pressure
    }
}
/// Quality of Service status
#[derive(Debug, Clone)]
pub struct QoSStatus {
    pub is_compliant: bool,
    pub violations: Vec<QoSViolation>,
    pub timestamp: Instant,
}
/// Synchronization barrier
#[derive(Debug, Clone)]
pub struct SyncBarrier {
    pub barrier_id: String,
    pub wait_count: usize,
    pub timestamp: Instant,
}
/// Learning rate adaptation state
#[derive(Debug, Clone)]
pub(super) struct LearningRateAdaptationState<A: Float + Send + Sync> {
    /// Current learning rate
    pub(super) current_lr: A,
    /// Accumulated squared gradients (for AdaGrad)
    pub(super) accumulated_gradients: Option<Array1<A>>,
    /// Exponential moving average of squared gradients (for RMSprop)
    pub(super) ema_squared_gradients: Option<Array1<A>>,
    /// Performance history
    pub(super) performance_history: VecDeque<A>,
    /// Last adaptation time
    pub(super) last_adaptation: Instant,
    /// Adaptation frequency
    pub(super) adaptation_frequency: Duration,
}
/// Interrupt handling strategies for real-time processing
#[derive(Debug, Clone, Copy)]
pub enum InterruptStrategy {
    Immediate,
    Deferred,
    Batched,
    Adaptive,
}
/// Quality of Service violation types
#[derive(Debug, Clone)]
pub enum QoSViolation {
    LatencyExceeded { actual: f64, target: f64 },
    MemoryExceeded { actual: f64, target: f64 },
    ThroughputDegraded { violation_rate: f64 },
    PredictionAccuracyDegraded { current: f64, target: f64 },
    ResourceUtilizationLow { utilization: f64, target: f64 },
    StreamSynchronizationLoss { delay_ms: f64 },
}
/// Maximum number of allocation decisions retained for inspection.
const MAX_ALLOCATION_HISTORY: usize = 100;

/// Headroom kept above the live memory footprint when reserving memory.
const MEMORY_RESERVATION_HEADROOM: f64 = 1.5;

/// Adaptive resource manager.
pub struct AdaptiveResourceManager {
    /// Resource allocation strategy
    pub(super) allocation_strategy: ResourceAllocationStrategy,
    /// Current resource usage
    pub(super) current_usage: ResourceUsage,
    /// Resource constraints
    pub(super) constraints: ResourceConstraints,
    /// Allocation history
    pub(super) allocation_history: VecDeque<ResourceAllocation>,
}
impl AdaptiveResourceManager {
    pub fn new(config: &StreamingConfig) -> Result<Self> {
        // The CPU ceiling is the machine's real parallelism, not a guess.
        let max_cpu_cores = std::thread::available_parallelism()
            .map(|degree| degree.get())
            .unwrap_or(1);
        Ok(Self {
            allocation_strategy: ResourceAllocationStrategy::Adaptive,
            current_usage: ResourceUsage::default(),
            constraints: ResourceConstraints {
                max_memory_mb: config.memory_budget_mb,
                max_cpu_cores,
                max_latency_ms: config.latency_budget_ms,
            },
            allocation_history: VecDeque::with_capacity(MAX_ALLOCATION_HISTORY),
        })
    }

    /// Record the measured resource usage of the streaming path.
    pub fn observe_usage(&mut self, usage: ResourceUsage) {
        self.current_usage = usage;
    }

    /// Measured resource usage most recently observed.
    pub fn current_usage(&self) -> &ResourceUsage {
        &self.current_usage
    }

    /// Configured resource ceilings.
    pub fn constraints(&self) -> &ResourceConstraints {
        &self.constraints
    }

    /// Every allocation decision made so far (most recent last, capped).
    pub fn allocation_history(&self) -> &VecDeque<ResourceAllocation> {
        &self.allocation_history
    }

    /// Decide the next allocation from the measured load, the QoS pressure
    /// and the stream's processing priority.
    ///
    /// Every field of the result is now derived from a real input: the
    /// previous implementation reported a constant `cpu_allocation: 2` and
    /// `priority_adjustment: 0` regardless of what the system was doing, and
    /// reserved memory with no floor (so an idle stream was "allocated" zero
    /// megabytes).
    pub fn adapt_allocation(
        &mut self,
        load_metrics: &StreamingMetrics,
        qos_pressure: f64,
        processing_priority: StreamPriority,
    ) -> Result<ResourceAllocation> {
        let max_memory = self.constraints.max_memory_mb.max(1);
        let reserved = (load_metrics.memory_usage_mb * MEMORY_RESERVATION_HEADROOM).ceil();
        let memory_allocation_mb = if reserved.is_finite() && reserved > 1.0 {
            (reserved as usize).min(max_memory)
        } else {
            1.min(max_memory)
        };

        // Latency pressure: how much of the configured budget each batch is
        // already consuming. Combined with the QoS breach pressure, this is
        // the real demand signal for more cores.
        let latency_pressure = if self.constraints.max_latency_ms > 0 {
            load_metrics.avg_latency_ms / self.constraints.max_latency_ms as f64
        } else {
            0.0
        };
        let demand = latency_pressure.max(qos_pressure.max(0.0));
        let max_cores = self.constraints.max_cpu_cores.max(1);
        let cpu_allocation = match self.allocation_strategy {
            // A static allocation deliberately ignores the live load.
            ResourceAllocationStrategy::Static => max_cores.min(2),
            _ => {
                if demand.is_finite() {
                    let requested = (demand * max_cores as f64).ceil();
                    let requested = if requested.is_finite() && requested >= 1.0 {
                        requested as usize
                    } else {
                        1
                    };
                    requested.clamp(1, max_cores)
                } else {
                    1
                }
            }
        };

        // Priority relative to `Normal`, nudged up while the system is
        // breaching its objectives.
        let base_priority = processing_priority as i32 - StreamPriority::Normal as i32;
        let pressure_bump = if qos_pressure > 1.0 {
            2
        } else if qos_pressure > 0.0 {
            1
        } else {
            0
        };
        let priority_adjustment = (base_priority + pressure_bump).clamp(-2, 5);

        let allocation = ResourceAllocation {
            memory_allocation_mb,
            cpu_allocation,
            priority_adjustment,
            timestamp: Instant::now(),
        };
        self.allocation_history.push_back(allocation.clone());
        if self.allocation_history.len() > MAX_ALLOCATION_HISTORY {
            self.allocation_history.pop_front();
        }
        Ok(allocation)
    }
}
