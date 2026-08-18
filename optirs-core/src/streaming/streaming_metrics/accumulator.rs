// Real metric accumulation for the streaming metrics collector (finding M1).
//
// The four `update_*_metrics` methods used to be `Ok(())` with a comment. They
// now derive every metric from the rolling raw observations kept here, and
// leave an `Option` at `None` whenever the underlying measurement was never
// supplied.

use super::*;
use std::collections::VecDeque;

/// OS-level resource measurements the collector cannot take itself.
#[derive(Debug, Clone, Default)]
pub struct ResourceProbe {
    /// CPU utilization percentage (0-100)
    pub cpu_utilization: Option<f64>,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization: Option<f64>,
    /// Network bandwidth in MB/s
    pub network_bandwidth_mbps: Option<f64>,
    /// Disk I/O in MB/s
    pub disk_io_mbps: Option<f64>,
    /// Fraction of available threads busy (0-1)
    pub thread_utilization: Option<f64>,
    /// Bytes handed out by the allocator
    pub total_allocated_bytes: Option<u64>,
    /// Allocator fragmentation ratio (0-1)
    pub fragmentation_ratio: Option<f64>,
}

/// Robustness measurements obtained from deliberate perturbation experiments.
#[derive(Debug, Clone)]
pub struct RobustnessProbe<A: Float + Send + Sync> {
    /// Retained accuracy under input noise
    pub noise_tolerance: Option<A>,
    /// Retained accuracy under adversarial perturbation
    pub adversarial_robustness: Option<A>,
    /// Output sensitivity to a unit input perturbation
    pub perturbation_sensitivity: Option<A>,
    /// Fraction of performance recovered after a shock
    pub recovery_capability: Option<A>,
    /// Fraction of injected faults survived
    pub fault_tolerance: Option<A>,
}

impl<A: Float + Send + Sync> Default for RobustnessProbe<A> {
    fn default() -> Self {
        Self {
            noise_tolerance: None,
            adversarial_robustness: None,
            perturbation_sensitivity: None,
            recovery_capability: None,
            fault_tolerance: None,
        }
    }
}

/// Most recent externally reported drift event.
#[derive(Debug, Clone)]
struct DriftReport<A: Float + Send + Sync> {
    magnitude: A,
    confidence: A,
    detection_latency: Duration,
    adaptation_effectiveness: Option<A>,
}

/// Rolling raw observations every aggregate metric is derived from.
#[derive(Debug, Clone)]
pub struct MetricsAccumulator<A: Float + Send + Sync> {
    window: usize,

    losses: VecDeque<A>,
    gradients: VecDeque<A>,
    latencies: VecDeque<Duration>,
    inter_arrival: VecDeque<Duration>,
    memory: VecDeque<u64>,
    gradient_times: VecDeque<Duration>,
    update_times: VecDeque<Duration>,
    communication_times: VecDeque<Duration>,
    queue_times: VecDeque<Duration>,

    first_timestamp: Option<SystemTime>,
    last_timestamp: Option<SystemTime>,
    first_loss: Option<A>,

    sample_count: u64,
    valid_sample_count: u64,
    peak_memory: u64,
    peak_rate: Option<f64>,
    min_rate: Option<f64>,

    // Welford accumulator over the loss stream, used for the anomaly z-score.
    loss_n: u64,
    loss_mean: f64,
    loss_m2: f64,
    anomaly_count: u64,

    slo_evaluated: u64,
    slo_met: u64,

    downtime: Duration,
    drift_events: u64,
    last_drift: Option<DriftReport<A>>,
    energy_joules: Option<f64>,

    resource_probe: Option<ResourceProbe>,
    robustness_probe: Option<RobustnessProbe<A>>,
}

fn push_capped<T>(queue: &mut VecDeque<T>, value: T, window: usize) {
    queue.push_back(value);
    while queue.len() > window {
        queue.pop_front();
    }
}

pub(crate) fn to_scalar<A: Float>(value: f64) -> A {
    A::from(value).unwrap_or_else(A::zero)
}

pub(crate) fn from_scalar<A: Float>(value: A) -> f64 {
    value.to_f64().unwrap_or(0.0)
}

pub(crate) fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub(crate) fn variance_f64(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean_f64(values);
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Ordinary-least-squares slope of `values` against their index.
pub(crate) fn ols_slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let x_mean = (n - 1) as f64 / 2.0;
    let y_mean = mean_f64(values);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (index, value) in values.iter().enumerate() {
        let dx = index as f64 - x_mean;
        numerator += dx * (value - y_mean);
        denominator += dx * dx;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Full latency statistics over a sample window, or `None` when empty.
pub(crate) fn duration_stats(samples: &VecDeque<Duration>) -> Option<LatencyStats> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted: Vec<Duration> = samples.iter().copied().collect();
    sorted.sort();
    let last = sorted.len() - 1;
    let quantile = |q: f64| sorted[((sorted.len() as f64 * q) as usize).min(last)];

    let seconds: Vec<f64> = sorted.iter().map(|d| d.as_secs_f64()).collect();
    let mean_seconds = mean_f64(&seconds);
    let std_seconds = variance_f64(&seconds).sqrt();

    Some(LatencyStats {
        mean: Duration::from_secs_f64(mean_seconds.max(0.0)),
        median: quantile(0.50),
        p95: quantile(0.95),
        p99: quantile(0.99),
        p999: quantile(0.999),
        max: sorted[last],
        min: sorted[0],
        std_dev: Duration::from_secs_f64(std_seconds.max(0.0)),
    })
}

impl<A: Float + Send + Sync> MetricsAccumulator<A> {
    /// Create an accumulator retaining `window` observations per series.
    pub fn new(window: usize) -> Self {
        let window = window.max(2);
        Self {
            window,
            losses: VecDeque::with_capacity(window),
            gradients: VecDeque::with_capacity(window),
            latencies: VecDeque::with_capacity(window),
            inter_arrival: VecDeque::with_capacity(window),
            memory: VecDeque::with_capacity(window),
            gradient_times: VecDeque::with_capacity(window),
            update_times: VecDeque::with_capacity(window),
            communication_times: VecDeque::with_capacity(window),
            queue_times: VecDeque::with_capacity(window),
            first_timestamp: None,
            last_timestamp: None,
            first_loss: None,
            sample_count: 0,
            valid_sample_count: 0,
            peak_memory: 0,
            peak_rate: None,
            min_rate: None,
            loss_n: 0,
            loss_mean: 0.0,
            loss_m2: 0.0,
            anomaly_count: 0,
            slo_evaluated: 0,
            slo_met: 0,
            downtime: Duration::ZERO,
            drift_events: 0,
            last_drift: None,
            energy_joules: None,
            resource_probe: None,
            robustness_probe: None,
        }
    }

    /// Fold one sample into the rolling state.
    pub fn ingest(&mut self, sample: &MetricsSample<A>) {
        self.sample_count += 1;

        let loss = from_scalar(sample.loss);
        let gradient = from_scalar(sample.gradient_magnitude);
        if loss.is_finite() && gradient.is_finite() && gradient >= 0.0 {
            self.valid_sample_count += 1;
        }

        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(sample.timestamp);
        }
        if self.first_loss.is_none() && loss.is_finite() {
            self.first_loss = Some(sample.loss);
        }

        if let Some(previous) = self.last_timestamp {
            let gap = saturating_elapsed(sample.timestamp, previous);
            push_capped(&mut self.inter_arrival, gap, self.window);
            let seconds = gap.as_secs_f64();
            if seconds > 0.0 {
                let rate = 1.0 / seconds;
                self.peak_rate = Some(self.peak_rate.map_or(rate, |p| p.max(rate)));
                self.min_rate = Some(self.min_rate.map_or(rate, |p| p.min(rate)));
            }
        }
        self.last_timestamp = Some(sample.timestamp);

        push_capped(&mut self.losses, sample.loss, self.window);
        push_capped(&mut self.gradients, sample.gradient_magnitude, self.window);
        push_capped(&mut self.latencies, sample.processing_time, self.window);
        push_capped(&mut self.memory, sample.memory_usage, self.window);
        self.peak_memory = self.peak_memory.max(sample.memory_usage);

        if let Some(value) = sample.gradient_computation_time {
            push_capped(&mut self.gradient_times, value, self.window);
        }
        if let Some(value) = sample.update_application_time {
            push_capped(&mut self.update_times, value, self.window);
        }
        if let Some(value) = sample.communication_time {
            push_capped(&mut self.communication_times, value, self.window);
        }
        if let Some(value) = sample.queue_wait_time {
            push_capped(&mut self.queue_times, value, self.window);
        }

        // Welford update plus anomaly counting against the *previous*
        // statistics, so a sample never suppresses its own anomaly score.
        if loss.is_finite() {
            let previous_std = self.loss_std();
            if self.loss_n >= 2 && previous_std > 0.0 {
                let z = (loss - self.loss_mean).abs() / previous_std;
                if z >= 3.0 {
                    self.anomaly_count += 1;
                }
            }
            self.loss_n += 1;
            let delta = loss - self.loss_mean;
            self.loss_mean += delta / self.loss_n as f64;
            self.loss_m2 += delta * (loss - self.loss_mean);
        }
    }

    fn loss_std(&self) -> f64 {
        if self.loss_n < 2 {
            0.0
        } else {
            (self.loss_m2 / self.loss_n as f64).sqrt()
        }
    }

    /// Robust z-score of the most recent loss against the running statistics.
    pub(crate) fn anomaly_score(&self) -> f64 {
        let Some(&latest) = self.losses.back() else {
            return 0.0;
        };
        let latest = from_scalar(latest);
        let std = self.loss_std();
        if !latest.is_finite() || std <= 0.0 {
            0.0
        } else {
            (latest - self.loss_mean).abs() / std
        }
    }

    pub(crate) fn anomaly_frequency(&self) -> f64 {
        if self.loss_n == 0 {
            0.0
        } else {
            self.anomaly_count as f64 / self.loss_n as f64
        }
    }

    pub(crate) fn observed_span(&self) -> Duration {
        match (self.first_timestamp, self.last_timestamp) {
            (Some(first), Some(last)) => saturating_elapsed(last, first),
            _ => Duration::ZERO,
        }
    }

    pub(crate) fn losses_f64(&self) -> Vec<f64> {
        self.losses.iter().map(|v| from_scalar(*v)).collect()
    }

    pub(crate) fn gradients_f64(&self) -> Vec<f64> {
        self.gradients.iter().map(|v| from_scalar(*v)).collect()
    }

    pub(crate) fn rates_f64(&self) -> Vec<f64> {
        self.inter_arrival
            .iter()
            .map(|d| d.as_secs_f64())
            .filter(|s| *s > 0.0)
            .map(|s| 1.0 / s)
            .collect()
    }

    pub(crate) fn total_processing_time(&self) -> f64 {
        self.latencies.iter().map(|d| d.as_secs_f64()).sum()
    }

    pub(crate) fn total_communication_time(&self) -> f64 {
        self.communication_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum()
    }

    /// Record an SLO evaluation outcome for the sample just ingested.
    pub(crate) fn record_slo_outcome(&mut self, met: bool) {
        self.slo_evaluated += 1;
        if met {
            self.slo_met += 1;
        }
    }

    pub(crate) fn slo_compliance(&self) -> Option<f64> {
        if self.slo_evaluated == 0 {
            None
        } else {
            Some(self.slo_met as f64 / self.slo_evaluated as f64)
        }
    }

    /// Report observed downtime.
    pub fn record_outage(&mut self, downtime: Duration) {
        self.downtime = self.downtime.saturating_add(downtime);
    }

    pub(crate) fn availability(&self) -> Option<f64> {
        if self.downtime.is_zero() {
            // Without a single reported outage there is no evidence about
            // availability; reporting 100% would be an invention.
            return None;
        }
        let span = self.observed_span().as_secs_f64() + self.downtime.as_secs_f64();
        if span <= 0.0 {
            return None;
        }
        Some((span - self.downtime.as_secs_f64()) / span)
    }

    /// Report a detected concept-drift event.
    pub fn record_drift_event(
        &mut self,
        magnitude: A,
        confidence: A,
        detection_latency: Duration,
        adaptation_effectiveness: Option<A>,
    ) {
        self.drift_events += 1;
        self.last_drift = Some(DriftReport {
            magnitude,
            confidence,
            detection_latency,
            adaptation_effectiveness,
        });
    }

    /// Report measured energy consumption.
    pub fn record_energy(&mut self, joules: f64) {
        self.energy_joules = Some(self.energy_joules.unwrap_or(0.0) + joules);
    }

    pub fn record_resource_probe(&mut self, probe: ResourceProbe) {
        self.resource_probe = Some(probe);
    }

    pub fn record_robustness_probe(&mut self, probe: RobustnessProbe<A>) {
        self.robustness_probe = Some(probe);
    }
}

impl<A: Float + Default + Clone + std::fmt::Debug + Send + Sync> StreamingMetricsCollector<A> {
    pub(crate) fn update_performance_metrics(&mut self, sample: &MetricsSample<A>) -> Result<()> {
        let rates = self.accumulator.rates_f64();
        let losses = self.accumulator.losses_f64();
        let gradients = self.accumulator.gradients_f64();

        // ---- throughput -------------------------------------------------
        let throughput = &mut self.performance_metrics.throughput;
        if !rates.is_empty() {
            let mean_rate = mean_f64(&rates);
            throughput.samples_per_second = mean_rate;
            throughput.updates_per_second = mean_rate;
            throughput.gradients_per_second = mean_rate;
            throughput.throughput_variance = variance_f64(&rates);
            throughput.throughput_trend = ols_slope(&rates);
        }
        if let Some(peak) = self.accumulator.peak_rate {
            throughput.peak_throughput = peak;
        }
        if let Some(min) = self.accumulator.min_rate {
            throughput.min_throughput = min;
        }

        // ---- latency ----------------------------------------------------
        let latency = &mut self.performance_metrics.latency;
        if let Some(stats) = duration_stats(&self.accumulator.latencies) {
            latency.end_to_end = stats;
        }
        latency.gradient_computation = duration_stats(&self.accumulator.gradient_times);
        latency.update_application = duration_stats(&self.accumulator.update_times);
        latency.communication = duration_stats(&self.accumulator.communication_times);
        latency.queue_wait_time = duration_stats(&self.accumulator.queue_times);
        latency.jitter = {
            let seconds: Vec<f64> = self
                .accumulator
                .latencies
                .iter()
                .map(|d| d.as_secs_f64())
                .collect();
            if seconds.len() < 2 {
                0.0
            } else {
                seconds
                    .windows(2)
                    .map(|pair| (pair[1] - pair[0]).abs())
                    .sum::<f64>()
                    / (seconds.len() - 1) as f64
            }
        };

        // ---- accuracy / convergence --------------------------------------
        let span_seconds = self.accumulator.observed_span().as_secs_f64();
        let first_loss = self.accumulator.first_loss.map(from_scalar);
        let current_loss = from_scalar(sample.loss);
        let loss_slope = ols_slope(&losses);

        let accuracy = &mut self.performance_metrics.accuracy;
        accuracy.current_loss = sample.loss;
        accuracy.gradient_magnitude = sample.gradient_magnitude;
        accuracy.loss_reduction_rate = match (first_loss, span_seconds > 0.0) {
            (Some(first), true) => to_scalar((first - current_loss) / span_seconds),
            _ => A::zero(),
        };
        // Positive when the loss is trending down.
        accuracy.convergence_rate = to_scalar(-loss_slope);
        accuracy.prediction_accuracy = sample.custom_metrics.get("accuracy").copied();
        accuracy.parameter_stability = {
            let spread = variance_f64(&gradients).sqrt();
            to_scalar(1.0 / (1.0 + spread))
        };
        accuracy.learning_progress = match first_loss {
            Some(first) if first.abs() > f64::EPSILON => {
                to_scalar(((first - current_loss) / first.abs()).clamp(-1.0, 1.0))
            }
            _ => A::zero(),
        };

        // ---- stability ---------------------------------------------------
        let increases = losses.windows(2).filter(|pair| pair[1] > pair[0]).count() as f64;
        let sign_changes = losses
            .windows(3)
            .filter(|triple| {
                let first = triple[1] - triple[0];
                let second = triple[2] - triple[1];
                first * second < 0.0
            })
            .count() as f64;

        let stability = &mut self.performance_metrics.stability;
        stability.loss_variance = to_scalar(variance_f64(&losses));
        stability.gradient_variance = to_scalar(variance_f64(&gradients));
        stability.parameter_drift = to_scalar(mean_f64(&gradients));
        stability.oscillation_score = if losses.len() >= 3 {
            to_scalar(sign_changes / (losses.len() - 2) as f64)
        } else {
            A::zero()
        };
        stability.divergence_probability = if losses.len() >= 2 {
            to_scalar(increases / (losses.len() - 1) as f64)
        } else {
            A::zero()
        };
        stability.stability_confidence = {
            let mean_loss = mean_f64(&losses).abs();
            let spread = variance_f64(&losses).sqrt();
            if mean_loss > f64::EPSILON {
                to_scalar((1.0 - (spread / mean_loss)).clamp(0.0, 1.0))
            } else {
                A::zero()
            }
        };

        // ---- efficiency ---------------------------------------------------
        let processing_seconds = self.accumulator.total_processing_time();
        let communication_seconds = self.accumulator.total_communication_time();
        let memory_values: Vec<f64> = self
            .accumulator
            .memory
            .iter()
            .map(|bytes| *bytes as f64)
            .collect();
        let peak_memory = self.accumulator.peak_memory as f64;

        let efficiency = &mut self.performance_metrics.efficiency;
        efficiency.computational_efficiency = match (first_loss, processing_seconds > 0.0) {
            (Some(first), true) => Some(to_scalar((first - current_loss) / processing_seconds)),
            _ => None,
        };
        efficiency.memory_efficiency = if peak_memory > 0.0 && !memory_values.is_empty() {
            Some(to_scalar(mean_f64(&memory_values) / peak_memory))
        } else {
            None
        };
        efficiency.communication_efficiency =
            if !self.accumulator.communication_times.is_empty() && processing_seconds > 0.0 {
                Some(to_scalar(
                    (1.0 - communication_seconds / processing_seconds).clamp(0.0, 1.0),
                ))
            } else {
                None
            };
        efficiency.energy_efficiency = match (self.accumulator.energy_joules, first_loss) {
            (Some(joules), Some(first)) if joules > 0.0 => {
                Some(to_scalar((first - current_loss) / joules))
            }
            _ => None,
        };
        efficiency.resource_utilization = if span_seconds > 0.0 {
            to_scalar((processing_seconds / span_seconds).clamp(0.0, 1.0))
        } else {
            A::zero()
        };

        Ok(())
    }

    pub(crate) fn update_resource_metrics(&mut self, sample: &MetricsSample<A>) -> Result<()> {
        let memory_values: Vec<f64> = self
            .accumulator
            .memory
            .iter()
            .map(|bytes| *bytes as f64)
            .collect();
        let peak = self.accumulator.peak_memory;

        self.resource_metrics.memory_usage = MemoryUsage {
            total_allocated: self
                .accumulator
                .resource_probe
                .as_ref()
                .and_then(|probe| probe.total_allocated_bytes),
            current_used: sample.memory_usage,
            peak_usage: peak,
            fragmentation_ratio: self
                .accumulator
                .resource_probe
                .as_ref()
                .and_then(|probe| probe.fragmentation_ratio),
            // Rust is not garbage collected; there is no overhead to report.
            gc_overhead: None,
            efficiency: if peak > 0 && !memory_values.is_empty() {
                Some(mean_f64(&memory_values) / peak as f64)
            } else {
                None
            },
        };

        if let Some(probe) = self.accumulator.resource_probe.as_ref() {
            self.resource_metrics.cpu_utilization = probe.cpu_utilization;
            self.resource_metrics.gpu_utilization = probe.gpu_utilization;
            self.resource_metrics.network_bandwidth = probe.network_bandwidth_mbps;
            self.resource_metrics.disk_io = probe.disk_io_mbps;
            self.resource_metrics.thread_utilization = probe.thread_utilization;
        }

        Ok(())
    }

    pub(crate) fn update_quality_metrics(&mut self, sample: &MetricsSample<A>) -> Result<()> {
        let losses = self.accumulator.losses_f64();
        let current_loss = from_scalar(sample.loss);
        let first_loss = self.accumulator.first_loss.map(from_scalar);

        self.quality_metrics.data_quality = if self.accumulator.sample_count == 0 {
            A::zero()
        } else {
            to_scalar(
                self.accumulator.valid_sample_count as f64 / self.accumulator.sample_count as f64,
            )
        };

        let validation_loss = sample
            .custom_metrics
            .get("val_loss")
            .copied()
            .map(from_scalar);
        let model_quality = &mut self.quality_metrics.model_quality;
        model_quality.training_quality = match first_loss {
            Some(first) if first.abs() > f64::EPSILON => {
                to_scalar(((first - current_loss) / first.abs()).clamp(0.0, 1.0))
            }
            _ => A::zero(),
        };
        model_quality.generalization_score = validation_loss.map(|validation| {
            // 1 when validation matches training loss, decaying as the gap grows.
            let gap = (validation - current_loss).abs();
            to_scalar(1.0 / (1.0 + gap))
        });
        model_quality.overfitting_score = validation_loss.map(|validation| {
            let denominator = current_loss.abs().max(f64::EPSILON);
            to_scalar(((validation - current_loss) / denominator).max(0.0))
        });
        model_quality.underfitting_score = validation_loss.map(|validation| {
            // Both losses high and close together indicates underfitting.
            let denominator = current_loss.abs().max(f64::EPSILON);
            let gap = ((validation - current_loss) / denominator).abs();
            match first_loss {
                Some(first) if first.abs() > f64::EPSILON => {
                    to_scalar(((current_loss / first.abs()) * (1.0 - gap)).clamp(0.0, 1.0))
                }
                _ => to_scalar((1.0 - gap).clamp(0.0, 1.0)),
            }
        });
        model_quality.complexity_score =
            sample
                .custom_metrics
                .get("parameter_count")
                .copied()
                .map(|count| {
                    let count = from_scalar(count).max(1.0);
                    to_scalar(count.ln() / (1.0 + count.ln()))
                });

        let span_seconds = self.accumulator.observed_span().as_secs_f64();
        let drift = &mut self.quality_metrics.concept_drift;
        drift.drift_frequency = if span_seconds > 0.0 {
            self.accumulator.drift_events as f64 / span_seconds
        } else {
            0.0
        };
        if let Some(report) = self.accumulator.last_drift.as_ref() {
            drift.drift_confidence = Some(report.confidence);
            drift.drift_magnitude = Some(report.magnitude);
            drift.detection_latency = Some(report.detection_latency);
            drift.adaptation_effectiveness = report.adaptation_effectiveness;
        }

        let anomaly = &mut self.quality_metrics.anomaly_detection;
        anomaly.anomaly_score = to_scalar(self.accumulator.anomaly_score());
        anomaly.anomaly_frequency = self.accumulator.anomaly_frequency();
        // The remaining anomaly metrics need labelled ground truth, which a
        // streaming collector never observes; they stay `None`.

        if let Some(probe) = self.accumulator.robustness_probe.as_ref() {
            self.quality_metrics.robustness = RobustnessMetrics {
                noise_tolerance: probe.noise_tolerance,
                adversarial_robustness: probe.adversarial_robustness,
                perturbation_sensitivity: probe.perturbation_sensitivity,
                recovery_capability: probe.recovery_capability,
                fault_tolerance: probe.fault_tolerance,
            };
        }

        let _ = losses;
        Ok(())
    }

    pub(crate) fn update_business_metrics(&mut self, sample: &MetricsSample<A>) -> Result<()> {
        if let Some(targets) = self.slo.clone() {
            let mut met = true;
            if let Some(limit) = targets.max_processing_time {
                met &= sample.processing_time <= limit;
            }
            if let Some(limit) = targets.max_loss {
                met &= from_scalar(sample.loss) <= limit;
            }
            if let Some(limit) = targets.max_memory_bytes {
                met &= sample.memory_usage <= limit;
            }
            self.accumulator.record_slo_outcome(met);
        }

        self.business_metrics.slo_compliance = self.accumulator.slo_compliance();
        self.business_metrics.availability = self.accumulator.availability();
        self.business_metrics.user_satisfaction =
            sample.custom_metrics.get("user_satisfaction").copied();

        if let Some(model) = self.cost_model.clone() {
            let compute_seconds = self.accumulator.total_processing_time();
            let memory_gb_hours = {
                let bytes: Vec<f64> = self
                    .accumulator
                    .memory
                    .iter()
                    .map(|value| *value as f64)
                    .collect();
                let mean_gb = mean_f64(&bytes) / (1024.0 * 1024.0 * 1024.0);
                mean_gb * (self.accumulator.observed_span().as_secs_f64() / 3600.0)
            };
            let energy = self.accumulator.energy_joules;

            let computational_cost =
                to_scalar::<A>(compute_seconds) * model.compute_cost_per_second;
            let infrastructure_cost =
                to_scalar::<A>(memory_gb_hours) * model.memory_cost_per_gb_hour;
            let energy_cost = energy.map(|j| to_scalar::<A>(j) * model.energy_cost_per_joule);

            let first_loss = self.accumulator.first_loss.map(from_scalar);
            let loss_reduction = match first_loss {
                Some(first) => (first - from_scalar(sample.loss)).max(0.0),
                None => 0.0,
            };
            let business_value = to_scalar::<A>(loss_reduction) * model.value_per_loss_unit;
            let total =
                computational_cost + infrastructure_cost + energy_cost.unwrap_or_else(A::zero);

            self.business_metrics.cost_metrics = CostMetrics {
                computational_cost: Some(computational_cost),
                infrastructure_cost: Some(infrastructure_cost),
                energy_cost,
                // The value forgone by spending this compute elsewhere is the
                // value it produced here, at the margin.
                opportunity_cost: Some(business_value),
                total_cost: Some(total),
            };
            self.business_metrics.business_value = Some(business_value);
            self.performance_metrics.efficiency.cost_efficiency = if total > A::zero() {
                Some(business_value / total)
            } else {
                None
            };
        }

        Ok(())
    }
}
