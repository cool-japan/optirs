// Advanced performance profiling for optimizers
//
// This module provides comprehensive performance analysis capabilities including
// memory profiling, gradient flow analysis, computational efficiency metrics,
// and hardware utilization monitoring.

use crate::error::Result;
use crate::system_sampler::SystemSampler;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Comprehensive performance profiler for optimizers
#[derive(Debug)]
pub struct PerformanceProfiler<A: Float> {
    /// Profiling configuration
    config: ProfilerConfig,
    /// Performance metrics collection
    metrics: PerformanceMetrics<A>,
    /// Memory usage tracking
    memory_tracker: MemoryTracker,
    /// Computational efficiency analyzer
    efficiency_analyzer: EfficiencyAnalyzer<A>,
    /// Hardware utilization monitor
    hardware_monitor: HardwareMonitor,
    /// Profiling session start time
    session_start: Instant,
    /// Current profiling step
    current_step: usize,
    /// Real system/process telemetry (RSS, CPU time deltas).
    sampler: SystemSampler,
}

/// Configuration for performance profiling
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable computational efficiency analysis
    pub enable_efficiency_analysis: bool,
    /// Enable hardware monitoring
    pub enable_hardware_monitoring: bool,
    /// Sample interval for hardware monitoring (milliseconds)
    pub hardware_sample_interval_ms: u64,
    /// Maximum history to keep for analysis
    pub max_history_length: usize,
    /// Enable detailed gradient analysis
    pub enable_gradient_analysis: bool,
    /// Enable convergence pattern detection
    pub enable_convergence_analysis: bool,
    /// Enable performance regression detection
    pub enable_regression_detection: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_memory_profiling: true,
            enable_efficiency_analysis: true,
            enable_hardware_monitoring: true,
            hardware_sample_interval_ms: 100,
            max_history_length: 10000,
            enable_gradient_analysis: true,
            enable_convergence_analysis: true,
            enable_regression_detection: true,
        }
    }
}

/// Comprehensive performance metrics
#[derive(Debug)]
pub struct PerformanceMetrics<A: Float> {
    /// Step timing information
    pub step_timings: VecDeque<StepTiming>,
    /// Memory usage metrics
    pub memory_metrics: MemoryMetrics,
    /// Computational metrics
    pub computational_metrics: ComputationalMetrics<A>,
    /// Gradient flow metrics
    pub gradient_metrics: GradientMetrics<A>,
    /// Convergence analysis
    pub convergence_metrics: ConvergenceMetrics<A>,
    /// Hardware utilization metrics
    pub hardware_metrics: HardwareMetrics,
}

/// Timing information for a single optimization step
#[derive(Debug, Clone)]
pub struct StepTiming {
    /// Step number
    pub step: usize,
    /// Total step duration
    pub total_duration: Duration,
    /// Gradient computation time
    pub gradient_computation_time: Duration,
    /// Parameter update time
    pub parameter_update_time: Duration,
    /// Memory allocation time
    pub memory_allocation_time: Duration,
    /// Timestamp
    pub timestamp: Instant,
    /// Caller-declared floating-point operation count for this step (via
    /// [`StepProfiler::record_op_count`]). `None` when the caller did not
    /// declare a count -- FLOPS is only ever derived from this real,
    /// caller-supplied figure divided by real elapsed time, never
    /// fabricated.
    pub declared_ops: Option<u64>,
}

/// Memory usage tracking
#[derive(Debug)]
#[allow(dead_code)]
pub struct MemoryTracker {
    /// Peak memory usage (bytes)
    peak_memory_bytes: usize,
    /// Current memory usage (bytes)
    current_memory_bytes: usize,
    /// Memory allocation count
    allocation_count: usize,
    /// Memory deallocation count
    deallocation_count: usize,
    /// Memory usage history
    memory_history: VecDeque<MemorySnapshot>,
    /// Memory fragmentation metrics
    fragmentation_metrics: FragmentationMetrics,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Number of allocations
    pub allocations: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// Memory fragmentation analysis
#[derive(Debug, Clone)]
pub struct FragmentationMetrics {
    /// Current fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub current_ratio: f64,
    /// Average fragmentation over time
    pub average_ratio: f64,
    /// Peak fragmentation observed
    pub peak_ratio: f64,
    /// Fragmentation trend (positive = increasing, negative = decreasing)
    pub trend: f64,
}

/// Memory metrics summary
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Peak memory usage
    pub peak_memory_bytes: usize,
    /// Average memory usage
    pub average_memory_bytes: f64,
    /// Memory efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
    /// Total allocations
    pub total_allocations: usize,
    /// Memory leak indicators
    pub leak_indicators: MemoryLeakIndicators,
    /// Fragmentation analysis
    pub fragmentation: FragmentationMetrics,
}

/// Memory leak detection indicators
#[derive(Debug, Clone)]
pub struct MemoryLeakIndicators {
    /// Suspected memory leak
    pub suspected_leak: bool,
    /// Memory growth rate (bytes per step)
    pub growth_rate: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Evidence description
    pub evidence: Vec<String>,
}

/// Computational efficiency analyzer
#[derive(Debug)]
#[allow(dead_code)]
pub struct EfficiencyAnalyzer<A: Float> {
    /// FLOPS (Floating Point Operations Per Second) history
    flops_history: VecDeque<f64>,
    /// Arithmetic intensity history
    arithmetic_intensity_history: VecDeque<f64>,
    /// Cache performance metrics
    cache_metrics: CacheMetrics,
    /// Vectorization efficiency
    vectorization_metrics: VectorizationMetrics,
    /// Algorithm complexity analysis
    complexity_analysis: ComplexityAnalysis<A>,
}

/// Cache performance metrics
#[derive(Debug, Clone)]
pub struct CacheMetrics {
    /// Cache hit ratio
    pub hit_ratio: f64,
    /// Cache miss penalty (nanoseconds)
    pub miss_penalty_ns: f64,
    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,
}

/// Vectorization efficiency metrics
#[derive(Debug, Clone)]
pub struct VectorizationMetrics {
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Vector width efficiency
    pub vector_width_efficiency: f64,
    /// Vectorization speedup factor
    pub speedup_factor: f64,
}

/// Algorithm complexity analysis
#[derive(Debug)]
pub struct ComplexityAnalysis<A: Float> {
    /// Estimated time complexity
    pub time_complexity: String,
    /// Estimated space complexity
    pub space_complexity: String,
    /// Scaling factor analysis
    pub scaling_factors: Vec<(usize, f64)>, // (problem_size, time_per_step)
    /// Efficiency trends
    pub efficiency_trends: EfficiencyTrends<A>,
}

/// Efficiency trend analysis
#[derive(Debug, Clone)]
pub struct EfficiencyTrends<A: Float> {
    /// Performance degradation rate
    pub degradation_rate: f64,
    /// Improvement opportunities
    pub improvement_opportunities: Vec<String>,
    /// Bottleneck identification
    pub bottlenecks: Vec<PerformanceBottleneck<A>>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck<A: Float> {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity (0.0 to 1.0)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Suggested optimizations
    pub optimizations: Vec<String>,
    /// Impact estimation
    pub estimated_impact: A,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone)]
pub enum BottleneckType {
    /// Memory bandwidth limitation
    MemoryBandwidth,
    /// Compute bound
    ComputeBound,
    /// Memory allocation overhead
    MemoryAllocation,
    /// Poor cache locality
    CacheLocality,
    /// Insufficient vectorization
    Vectorization,
    /// Algorithm inefficiency
    Algorithm,
    /// Hardware underutilization
    HardwareUnderutilization,
}

/// Computational efficiency metrics
#[derive(Debug, Clone)]
pub struct ComputationalMetrics<A: Float> {
    /// Average FLOPS achieved
    pub average_flops: f64,
    /// Peak FLOPS achieved
    pub peak_flops: f64,
    /// Arithmetic intensity
    pub arithmetic_intensity: f64,
    /// Cache efficiency
    pub cache_efficiency: f64,
    /// Vectorization efficiency
    pub vectorization_efficiency: f64,
    /// Overall efficiency score
    pub efficiency_score: f64,
    /// Bottleneck analysis
    pub bottlenecks: Vec<PerformanceBottleneck<A>>,
}

/// Gradient flow analysis metrics
#[derive(Debug, Clone)]
pub struct GradientMetrics<A: Float> {
    /// Gradient magnitude statistics
    pub magnitude_stats: GradientMagnitudeStats<A>,
    /// Gradient direction analysis
    pub direction_analysis: GradientDirectionAnalysis<A>,
    /// Gradient stability metrics
    pub stability_metrics: GradientStabilityMetrics<A>,
    /// Learning dynamics analysis
    pub learning_dynamics: LearningDynamicsAnalysis<A>,
}

/// Gradient magnitude statistics
#[derive(Debug, Clone)]
pub struct GradientMagnitudeStats<A: Float> {
    /// Mean gradient magnitude
    pub mean_magnitude: A,
    /// Standard deviation
    pub std_magnitude: A,
    /// Magnitude trend (growing/shrinking)
    pub magnitude_trend: A,
    /// Gradient explosion indicators
    pub explosion_indicators: Vec<String>,
    /// Vanishing gradient indicators
    pub vanishing_indicators: Vec<String>,
}

/// Gradient direction analysis
#[derive(Debug, Clone)]
pub struct GradientDirectionAnalysis<A: Float> {
    /// Direction consistency score
    pub consistency_score: A,
    /// Oscillation frequency
    pub oscillation_frequency: f64,
    /// Direction change patterns
    pub change_patterns: Vec<String>,
}

/// Gradient stability metrics
#[derive(Debug, Clone)]
pub struct GradientStabilityMetrics<A: Float> {
    /// Stability score (0.0 to 1.0)
    pub stability_score: f64,
    /// Noise level estimation
    pub noise_level: A,
    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: A,
}

/// Learning dynamics analysis
#[derive(Debug, Clone)]
pub struct LearningDynamicsAnalysis<A: Float> {
    /// Learning rate adaptation effectiveness
    pub lr_adaptation_effectiveness: f64,
    /// Momentum effectiveness
    pub momentum_effectiveness: f64,
    /// Second-order information utilization
    pub second_order_utilization: f64,
    /// Convergence velocity
    pub convergence_velocity: A,
}

/// Convergence analysis metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics<A: Float> {
    /// Convergence status
    pub status: ConvergenceStatus,
    /// Convergence rate estimation
    pub convergence_rate: f64,
    /// Time to convergence estimation
    pub estimated_time_to_convergence: Option<Duration>,
    /// Convergence quality score
    pub quality_score: f64,
    /// Convergence patterns
    pub patterns: Vec<ConvergencePattern<A>>,
}

/// Convergence status
#[derive(Debug, Clone)]
pub enum ConvergenceStatus {
    /// Rapidly converging
    RapidConvergence,
    /// Steady convergence
    SteadyConvergence,
    /// Slow convergence
    SlowConvergence,
    /// Oscillating
    Oscillating,
    /// Stagnated
    Stagnated,
    /// Diverging
    Diverging,
}

/// Convergence pattern identification
#[derive(Debug, Clone)]
pub struct ConvergencePattern<A: Float> {
    /// Pattern type
    pub pattern_type: String,
    /// Pattern strength (0.0 to 1.0)
    pub strength: f64,
    /// Pattern description
    pub description: String,
    /// Associated characteristics
    pub characteristics: Vec<A>,
}

/// Hardware utilization monitor
#[derive(Debug)]
#[allow(dead_code)]
pub struct HardwareMonitor {
    /// CPU utilization history
    cpu_utilization: VecDeque<f64>,
    /// Memory bandwidth utilization
    memory_bandwidth: VecDeque<f64>,
    /// GPU utilization (if available)
    gpu_utilization: Option<VecDeque<f64>>,
    /// Cache performance counters
    cache_counters: CacheCounters,
    /// Hardware efficiency metrics
    efficiency_metrics: HardwareEfficiencyMetrics,
}

/// Cache performance counters
#[derive(Debug, Clone, Default)]
pub struct CacheCounters {
    /// L1 cache hits
    pub l1_hits: u64,
    /// L1 cache misses
    pub l1_misses: u64,
    /// L2 cache hits
    pub l2_hits: u64,
    /// L2 cache misses
    pub l2_misses: u64,
    /// L3 cache hits
    pub l3_hits: u64,
    /// L3 cache misses
    pub l3_misses: u64,
}

/// Hardware efficiency metrics
#[derive(Debug, Clone)]
pub struct HardwareEfficiencyMetrics {
    /// Overall hardware utilization
    pub overall_utilization: f64,
    /// CPU efficiency
    pub cpu_efficiency: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
    /// Cache efficiency
    pub cache_efficiency: f64,
    /// GPU efficiency (if available)
    pub gpu_efficiency: Option<f64>,
}

/// Hardware metrics summary
#[derive(Debug, Clone)]
pub struct HardwareMetrics {
    /// Average CPU utilization
    pub avg_cpu_utilization: f64,
    /// Peak CPU utilization
    pub peak_cpu_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// GPU utilization (if available)
    pub gpu_utilization: Option<f64>,
    /// Hardware efficiency summary
    pub efficiency_summary: HardwareEfficiencyMetrics,
}

impl<A: Float + Debug + Send + Sync> PerformanceProfiler<A> {
    /// Create a new performance profiler
    pub fn new(config: ProfilerConfig) -> Result<Self> {
        Ok(Self {
            config,
            metrics: PerformanceMetrics::new(),
            memory_tracker: MemoryTracker::new(),
            efficiency_analyzer: EfficiencyAnalyzer::new(),
            hardware_monitor: HardwareMonitor::new(),
            session_start: Instant::now(),
            current_step: 0,
            sampler: SystemSampler::new()?,
        })
    }

    /// Start profiling an optimization step
    pub fn start_step(&mut self) -> StepProfiler<A> {
        self.current_step += 1;
        StepProfiler::new(self.current_step, &self.config)
    }

    /// Complete a profiling step
    pub fn complete_step(&mut self, step_profiler: StepProfiler<A>) -> Result<()> {
        let step_timing = step_profiler.finalize()?;

        // Update metrics
        self.metrics.step_timings.push_back(step_timing.clone());

        // Maintain history size
        if self.metrics.step_timings.len() > self.config.max_history_length {
            self.metrics.step_timings.pop_front();
        }

        // Update memory metrics if enabled
        if self.config.enable_memory_profiling {
            self.update_memory_metrics()?;
        }

        // Update efficiency metrics if enabled
        if self.config.enable_efficiency_analysis {
            self.update_efficiency_metrics(&step_timing)?;
        }

        // Update hardware metrics if enabled
        if self.config.enable_hardware_monitoring {
            self.update_hardware_metrics()?;
        }

        Ok(())
    }

    /// Update memory profiling metrics using real process RSS.
    fn update_memory_metrics(&mut self) -> Result<()> {
        let current_memory = self.estimate_memory_usage();

        self.memory_tracker.current_memory_bytes = current_memory;
        self.memory_tracker.peak_memory_bytes =
            self.memory_tracker.peak_memory_bytes.max(current_memory);

        // Create memory snapshot
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            memory_bytes: current_memory,
            allocations: self.memory_tracker.allocation_count,
            fragmentation_ratio: self.estimate_fragmentation(),
        };

        // Keep the aggregate fragmentation metrics in sync with the
        // snapshot's own value (previously always stuck at its `Default`
        // and never updated).
        self.memory_tracker.fragmentation_metrics.current_ratio = snapshot.fragmentation_ratio;
        self.memory_tracker.fragmentation_metrics.peak_ratio = self
            .memory_tracker
            .fragmentation_metrics
            .peak_ratio
            .max(snapshot.fragmentation_ratio);

        self.memory_tracker.memory_history.push_back(snapshot);

        // Maintain history size
        if self.memory_tracker.memory_history.len() > self.config.max_history_length {
            self.memory_tracker.memory_history.pop_front();
        }

        // Real average over history.
        if !self.memory_tracker.memory_history.is_empty() {
            self.memory_tracker.fragmentation_metrics.average_ratio = self
                .memory_tracker
                .memory_history
                .iter()
                .map(|s| s.fragmentation_ratio)
                .sum::<f64>()
                / self.memory_tracker.memory_history.len() as f64;
        }

        Ok(())
    }

    /// Update computational efficiency metrics. FLOPS is only recorded
    /// when the caller declared a real op count for this step (via
    /// [`StepProfiler::record_op_count`]) -- gates on missing data skip
    /// the sample rather than fabricating one.
    fn update_efficiency_metrics(&mut self, steptiming: &StepTiming) -> Result<()> {
        if let Some(flops) = self.estimate_flops(steptiming) {
            self.efficiency_analyzer.flops_history.push_back(flops);
            if self.efficiency_analyzer.flops_history.len() > self.config.max_history_length {
                self.efficiency_analyzer.flops_history.pop_front();
            }
        }

        Ok(())
    }

    /// Update hardware monitoring metrics. CPU utilization is only
    /// recorded when a real sample is available (skip on failure rather
    /// than fabricate). Memory bandwidth is not tracked: no portable
    /// hardware performance counters are available from safe Rust.
    fn update_hardware_metrics(&mut self) -> Result<()> {
        if let Some(cpu_util) = self.measure_cpu_utilization() {
            self.hardware_monitor.cpu_utilization.push_back(cpu_util);
            if self.hardware_monitor.cpu_utilization.len() > self.config.max_history_length {
                self.hardware_monitor.cpu_utilization.pop_front();
            }
        }

        Ok(())
    }

    /// Generate comprehensive performance report
    pub fn generate_performance_report(&self) -> PerformanceReport<A> {
        PerformanceReport {
            session_duration: self.session_start.elapsed(),
            total_steps: self.current_step,
            memory_analysis: self.analyze_memory_performance(),
            computational_analysis: self.analyze_computational_performance(),
            hardware_analysis: self.analyze_hardware_performance(),
            efficiency_recommendations: self.generate_efficiency_recommendations(),
            performance_score: self.calculate_overall_performance_score(),
        }
    }

    /// Analyze memory performance
    fn analyze_memory_performance(&self) -> MemoryAnalysis {
        let avg_memory = if !self.memory_tracker.memory_history.is_empty() {
            self.memory_tracker
                .memory_history
                .iter()
                .map(|s| s.memory_bytes as f64)
                .sum::<f64>()
                / self.memory_tracker.memory_history.len() as f64
        } else {
            0.0
        };

        let efficiency_score = self.calculate_memory_efficiency_score();
        let leak_indicators = self.detect_memory_leaks();

        MemoryAnalysis {
            peak_usage_bytes: self.memory_tracker.peak_memory_bytes,
            average_usage_bytes: avg_memory,
            efficiency_score,
            leak_indicators,
            fragmentation_analysis: self.memory_tracker.fragmentation_metrics.clone(),
            optimization_recommendations: self.generate_memory_optimizations(),
        }
    }

    /// Analyze computational performance
    fn analyze_computational_performance(&self) -> ComputationalAnalysis<A> {
        let avg_flops = if !self.efficiency_analyzer.flops_history.is_empty() {
            self.efficiency_analyzer.flops_history.iter().sum::<f64>()
                / self.efficiency_analyzer.flops_history.len() as f64
        } else {
            0.0
        };

        let peak_flops = self
            .efficiency_analyzer
            .flops_history
            .iter()
            .fold(0.0, |acc, &x| acc.max(x));

        ComputationalAnalysis {
            average_flops: avg_flops,
            peak_flops,
            arithmetic_intensity: None,
            vectorization_efficiency: None,
            bottlenecks: self.identify_computational_bottlenecks(),
            optimization_opportunities: self.identify_optimization_opportunities(),
        }
    }

    /// Analyze hardware performance
    fn analyze_hardware_performance(&self) -> HardwareAnalysis {
        let avg_cpu = if !self.hardware_monitor.cpu_utilization.is_empty() {
            self.hardware_monitor.cpu_utilization.iter().sum::<f64>()
                / self.hardware_monitor.cpu_utilization.len() as f64
        } else {
            0.0
        };

        let peak_cpu = self
            .hardware_monitor
            .cpu_utilization
            .iter()
            .fold(0.0, |acc, &x| acc.max(x));

        HardwareAnalysis {
            cpu_utilization_avg: avg_cpu,
            cpu_utilization_peak: peak_cpu,
            memory_bandwidth_utilization: None,
            cache_performance: None,
            hardware_efficiency_score: self.calculate_hardware_efficiency_score(),
            underutilization_analysis: self.analyze_hardware_underutilization(),
        }
    }

    /// Generate efficiency recommendations
    fn generate_efficiency_recommendations(&self) -> Vec<EfficiencyRecommendation> {
        let mut recommendations = Vec::new();

        // Memory-related recommendations
        if self.memory_tracker.fragmentation_metrics.current_ratio > 0.3 {
            recommendations.push(EfficiencyRecommendation {
                category: RecommendationCategory::Memory,
                priority: RecommendationPriority::High,
                title: "High Memory Fragmentation Detected".to_string(),
                description: "Consider using memory pools or pre-allocating arrays".to_string(),
                estimated_impact: 0.2,
            });
        }

        // Computational recommendations. Only fire with real FLOPS data --
        // an empty history must not silently read as "0 FLOPS, therefore
        // low throughput".
        if !self.efficiency_analyzer.flops_history.is_empty() {
            let avg_flops = self.efficiency_analyzer.flops_history.iter().sum::<f64>()
                / self.efficiency_analyzer.flops_history.len() as f64;
            if avg_flops < 1e9 {
                // Less than 1 GFLOPS
                recommendations.push(EfficiencyRecommendation {
                    category: RecommendationCategory::Computation,
                    priority: RecommendationPriority::Medium,
                    title: "Low Computational Throughput".to_string(),
                    description: "Consider enabling SIMD optimizations or GPU acceleration"
                        .to_string(),
                    estimated_impact: 0.3,
                });
            }
        }

        // Hardware utilization recommendations. Only fire with real CPU
        // samples, for the same reason.
        if !self.hardware_monitor.cpu_utilization.is_empty() {
            let avg_cpu = self.hardware_monitor.cpu_utilization.iter().sum::<f64>()
                / self.hardware_monitor.cpu_utilization.len() as f64;
            if avg_cpu < 0.5 {
                recommendations.push(EfficiencyRecommendation {
                    category: RecommendationCategory::Hardware,
                    priority: RecommendationPriority::Medium,
                    title: "Low CPU Utilization".to_string(),
                    description: "Consider increasing parallelism or batch size".to_string(),
                    estimated_impact: 0.25,
                });
            }
        }

        recommendations
    }

    /// Calculate overall performance score
    fn calculate_overall_performance_score(&self) -> f64 {
        let memory_score = self.calculate_memory_efficiency_score();
        let computational_score = self.calculate_computational_efficiency_score();
        let hardware_score = self.calculate_hardware_efficiency_score();

        // Weighted average
        (memory_score * 0.3 + computational_score * 0.4 + hardware_score * 0.3).clamp(0.0, 1.0)
    }

    // Helper methods for calculations and estimations

    /// Real process RSS via [`SystemSampler`].
    fn estimate_memory_usage(&self) -> usize {
        self.sampler.refresh();
        self.sampler
            .sample_process()
            .map(|sample| sample.rss_bytes as usize)
            .unwrap_or(self.memory_tracker.current_memory_bytes)
    }

    /// Not tracked: this profiler has no allocation-tracker bookkeeping of
    /// its own. Returns a real, honest zero rather than a step-count-based
    /// formula with no grounding in actual memory behavior. For a real,
    /// bookkeeping-derived fragmentation heuristic use
    /// `memory_optimizer::AllocationTracker` or
    /// `memory_leak_detector::AllocationTracker`.
    fn estimate_fragmentation(&self) -> f64 {
        0.0
    }

    /// Real FLOPS: caller-declared op count (via
    /// [`StepProfiler::record_op_count`]) divided by real elapsed step
    /// time. `None` when the caller declared no op count -- never a
    /// fabricated estimate.
    fn estimate_flops(&self, steptiming: &StepTiming) -> Option<f64> {
        let ops = steptiming.declared_ops?;
        let elapsed = steptiming.total_duration.as_secs_f64();
        if elapsed > 0.0 {
            Some(ops as f64 / elapsed)
        } else {
            None
        }
    }

    /// Real process CPU utilization (as a 0.0-1.0 fraction) derived from
    /// process time deltas via [`SystemSampler`]. `None` on the first
    /// sample (no prior delta to compare against) or if sampling fails --
    /// never a fabricated sine wave.
    fn measure_cpu_utilization(&self) -> Option<f64> {
        self.sampler.refresh();
        let sample = self.sampler.sample_process().ok()?;
        sample
            .cpu_percent
            .map(|percent| (percent / 100.0).clamp(0.0, 1.0))
    }

    fn calculate_memory_efficiency_score(&self) -> f64 {
        // Simplified memory efficiency calculation
        1.0 - self.memory_tracker.fragmentation_metrics.current_ratio
    }

    /// Real leak signal over the real (RSS-derived) memory history: growth
    /// rate is a genuine slope between the first and most recent
    /// snapshots, and `confidence` scales continuously with how far growth
    /// exceeds the 1KB/step threshold, rather than a fixed 0.7/0.1 binary
    /// switch.
    fn detect_memory_leaks(&self) -> MemoryLeakIndicators {
        const GROWTH_THRESHOLD_BYTES_PER_STEP: f64 = 1024.0;

        let growth_rate = if self.memory_tracker.memory_history.len() > 2 {
            let recent =
                &self.memory_tracker.memory_history[self.memory_tracker.memory_history.len() - 1];
            let earlier = &self.memory_tracker.memory_history[0];
            (recent.memory_bytes as f64 - earlier.memory_bytes as f64)
                / self.memory_tracker.memory_history.len() as f64
        } else {
            0.0
        };

        let suspected_leak = growth_rate > GROWTH_THRESHOLD_BYTES_PER_STEP;
        // Confidence grows with the ratio of observed growth to the
        // threshold, saturating at 1.0 for >= 3x threshold.
        let confidence = if suspected_leak {
            (growth_rate / (GROWTH_THRESHOLD_BYTES_PER_STEP * 3.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        MemoryLeakIndicators {
            suspected_leak,
            growth_rate,
            confidence,
            evidence: if suspected_leak {
                vec![format!(
                    "Memory grew by {growth_rate:.1} bytes/step (threshold {GROWTH_THRESHOLD_BYTES_PER_STEP:.0})"
                )]
            } else {
                vec![]
            },
        }
    }

    fn generate_memory_optimizations(&self) -> Vec<String> {
        let mut optimizations = Vec::new();

        if self.memory_tracker.fragmentation_metrics.current_ratio > 0.2 {
            optimizations.push("Use object pooling to reduce fragmentation".to_string());
        }

        if self.memory_tracker.peak_memory_bytes > 1024 * 1024 * 100 {
            // 100MB
            optimizations.push("Consider streaming or chunked processing".to_string());
        }

        optimizations
    }

    fn identify_computational_bottlenecks(&self) -> Vec<PerformanceBottleneck<A>> {
        let mut bottlenecks = Vec::new();

        // Only report a compute-bound bottleneck when real FLOPS data
        // exists: with an empty history, `avg_flops` would be a fabricated
        // 0.0 that always looks "low throughput" even though nothing was
        // measured at all.
        if self.efficiency_analyzer.flops_history.is_empty() {
            return bottlenecks;
        }
        let avg_flops = self.efficiency_analyzer.flops_history.iter().sum::<f64>()
            / self.efficiency_analyzer.flops_history.len() as f64;

        if avg_flops < 1e9 {
            if let Some(estimated_impact) = A::from(0.3) {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::ComputeBound,
                    severity: 0.6,
                    description: "Low computational throughput detected".to_string(),
                    optimizations: vec![
                        "Enable SIMD optimizations".to_string(),
                        "Consider GPU acceleration".to_string(),
                    ],
                    estimated_impact,
                });
            }
        }

        bottlenecks
    }

    fn identify_optimization_opportunities(&self) -> Vec<String> {
        vec![
            "Enable advanced SIMD operations".to_string(),
            "Optimize memory access patterns".to_string(),
            "Consider parallel processing".to_string(),
        ]
    }

    /// Real: FLOPS observed relative to a documented reference baseline of
    /// 1 GFLOPS (a conservative "reasonably efficient scalar numeric code"
    /// reference point), clamped to `[0, 1]`. `0.0` (an honest "no data",
    /// not a fabricated placeholder) when no step declared an op count.
    fn calculate_computational_efficiency_score(&self) -> f64 {
        const REFERENCE_FLOPS: f64 = 1e9;
        if self.efficiency_analyzer.flops_history.is_empty() {
            return 0.0;
        }
        let avg_flops = self.efficiency_analyzer.flops_history.iter().sum::<f64>()
            / self.efficiency_analyzer.flops_history.len() as f64;
        (avg_flops / REFERENCE_FLOPS).clamp(0.0, 1.0)
    }

    /// Real: mean of process-CPU-time-delta samples. Memory bandwidth is
    /// not folded in here since it is not tracked (no portable hardware
    /// counters available); `0.0` on no samples.
    fn calculate_hardware_efficiency_score(&self) -> f64 {
        if self.hardware_monitor.cpu_utilization.is_empty() {
            0.0
        } else {
            self.hardware_monitor.cpu_utilization.iter().sum::<f64>()
                / self.hardware_monitor.cpu_utilization.len() as f64
        }
    }

    fn analyze_hardware_underutilization(&self) -> Vec<String> {
        let mut issues = Vec::new();

        let avg_cpu = if !self.hardware_monitor.cpu_utilization.is_empty() {
            self.hardware_monitor.cpu_utilization.iter().sum::<f64>()
                / self.hardware_monitor.cpu_utilization.len() as f64
        } else {
            0.0
        };

        if avg_cpu < 0.5 {
            issues.push("CPU underutilization detected".to_string());
        }

        issues
    }
}

/// Step-level profiler for detailed timing
pub struct StepProfiler<A: Float> {
    step_number: usize,
    start_time: Instant,
    gradient_start: Option<Instant>,
    gradient_duration: Option<Duration>,
    update_start: Option<Instant>,
    update_duration: Option<Duration>,
    memory_start: Option<Instant>,
    memory_duration: Option<Duration>,
    declared_ops: Option<u64>,
    _config: ProfilerConfig,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> StepProfiler<A> {
    fn new(_stepnumber: usize, config: &ProfilerConfig) -> Self {
        Self {
            step_number: _stepnumber,
            start_time: Instant::now(),
            gradient_start: None,
            gradient_duration: None,
            update_start: None,
            update_duration: None,
            memory_start: None,
            memory_duration: None,
            declared_ops: None,
            _config: config.clone(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Declare the real number of floating-point operations performed in
    /// this step (e.g. from a caller-computed FLOP count for the
    /// optimizer's update rule). Enables real FLOPS reporting; without a
    /// declaration, FLOPS for this step is `None` rather than a
    /// fabricated estimate.
    pub fn record_op_count(&mut self, ops: u64) {
        self.declared_ops = Some(ops);
    }

    /// Mark the start of gradient computation
    pub fn start_gradient_computation(&mut self) {
        self.gradient_start = Some(Instant::now());
    }

    /// Mark the end of gradient computation
    pub fn end_gradient_computation(&mut self) {
        if let Some(start) = self.gradient_start {
            self.gradient_duration = Some(start.elapsed());
        }
    }

    /// Mark the start of parameter update
    pub fn start_parameter_update(&mut self) {
        self.update_start = Some(Instant::now());
    }

    /// Mark the end of parameter update
    pub fn end_parameter_update(&mut self) {
        if let Some(start) = self.update_start {
            self.update_duration = Some(start.elapsed());
        }
    }

    /// Mark the start of memory operation
    pub fn start_memory_operation(&mut self) {
        self.memory_start = Some(Instant::now());
    }

    /// Mark the end of memory operation
    pub fn end_memory_operation(&mut self) {
        if let Some(start) = self.memory_start {
            self.memory_duration = Some(start.elapsed());
        }
    }

    /// Finalize the step profiling
    fn finalize(self) -> Result<StepTiming> {
        Ok(StepTiming {
            step: self.step_number,
            total_duration: self.start_time.elapsed(),
            gradient_computation_time: self.gradient_duration.unwrap_or(Duration::from_nanos(0)),
            parameter_update_time: self.update_duration.unwrap_or(Duration::from_nanos(0)),
            memory_allocation_time: self.memory_duration.unwrap_or(Duration::from_nanos(0)),
            timestamp: self.start_time,
            declared_ops: self.declared_ops,
        })
    }
}

// Additional analysis structures

/// Comprehensive performance report
#[derive(Debug)]
pub struct PerformanceReport<A: Float> {
    pub session_duration: Duration,
    pub total_steps: usize,
    pub memory_analysis: MemoryAnalysis,
    pub computational_analysis: ComputationalAnalysis<A>,
    pub hardware_analysis: HardwareAnalysis,
    pub efficiency_recommendations: Vec<EfficiencyRecommendation>,
    pub performance_score: f64,
}

/// Memory performance analysis
#[derive(Debug)]
pub struct MemoryAnalysis {
    pub peak_usage_bytes: usize,
    pub average_usage_bytes: f64,
    pub efficiency_score: f64,
    pub leak_indicators: MemoryLeakIndicators,
    pub fragmentation_analysis: FragmentationMetrics,
    pub optimization_recommendations: Vec<String>,
}

/// Computational performance analysis
#[derive(Debug)]
pub struct ComputationalAnalysis<A: Float> {
    /// Real: mean of caller-declared FLOPS samples (0.0 if none recorded).
    pub average_flops: f64,
    /// Real: max of caller-declared FLOPS samples (0.0 if none recorded).
    pub peak_flops: f64,
    /// `None`: requires real memory-traffic byte tracking (FLOPs / bytes
    /// moved), which is not implemented -- never a fabricated formula.
    pub arithmetic_intensity: Option<f64>,
    /// `None`: no real SIMD/vectorization introspection is available from
    /// portable Rust.
    pub vectorization_efficiency: Option<f64>,
    pub bottlenecks: Vec<PerformanceBottleneck<A>>,
    pub optimization_opportunities: Vec<String>,
}

/// Hardware performance analysis
#[derive(Debug)]
pub struct HardwareAnalysis {
    /// Real: mean of process-CPU-time-delta samples (0.0 if none recorded).
    pub cpu_utilization_avg: f64,
    /// Real: max of process-CPU-time-delta samples (0.0 if none recorded).
    pub cpu_utilization_peak: f64,
    /// `None`: no portable hardware memory-bandwidth counters are
    /// available from safe Rust.
    pub memory_bandwidth_utilization: Option<f64>,
    /// `None`: no portable hardware cache-performance counters are
    /// available from safe Rust.
    pub cache_performance: Option<CachePerformanceAnalysis>,
    pub hardware_efficiency_score: f64,
    pub underutilization_analysis: Vec<String>,
}

/// Cache performance analysis
#[derive(Debug)]
pub struct CachePerformanceAnalysis {
    pub l1_hit_ratio: f64,
    pub l2_hit_ratio: f64,
    pub l3_hit_ratio: f64,
    pub cache_efficiency_score: f64,
    pub miss_penalty_impact: f64,
}

/// Efficiency recommendation
#[derive(Debug)]
pub struct EfficiencyRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub estimated_impact: f64,
}

/// Recommendation categories
#[derive(Debug)]
pub enum RecommendationCategory {
    Memory,
    Computation,
    Hardware,
    Algorithm,
}

/// Recommendation priorities
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum RecommendationPriority {
    High,
    Medium,
    Low,
}

// Default implementations for metrics structures

impl<A: Float + Send + Sync> PerformanceMetrics<A> {
    fn new() -> Self {
        Self {
            step_timings: VecDeque::new(),
            memory_metrics: MemoryMetrics::default(),
            computational_metrics: ComputationalMetrics::default(),
            gradient_metrics: GradientMetrics::default(),
            convergence_metrics: ConvergenceMetrics::default(),
            hardware_metrics: HardwareMetrics::default(),
        }
    }
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
            memory_history: VecDeque::new(),
            fragmentation_metrics: FragmentationMetrics::default(),
        }
    }
}

impl<A: Float + Send + Sync> EfficiencyAnalyzer<A> {
    fn new() -> Self {
        Self {
            flops_history: VecDeque::new(),
            arithmetic_intensity_history: VecDeque::new(),
            cache_metrics: CacheMetrics::default(),
            vectorization_metrics: VectorizationMetrics::default(),
            complexity_analysis: ComplexityAnalysis::default(),
        }
    }
}

impl HardwareMonitor {
    fn new() -> Self {
        Self {
            cpu_utilization: VecDeque::new(),
            memory_bandwidth: VecDeque::new(),
            gpu_utilization: None,
            cache_counters: CacheCounters::default(),
            efficiency_metrics: HardwareEfficiencyMetrics::default(),
        }
    }
}

// Default trait implementations for various metrics structures

impl Default for FragmentationMetrics {
    fn default() -> Self {
        Self {
            current_ratio: 0.0,
            average_ratio: 0.0,
            peak_ratio: 0.0,
            trend: 0.0,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            peak_memory_bytes: 0,
            average_memory_bytes: 0.0,
            efficiency_score: 1.0,
            total_allocations: 0,
            leak_indicators: MemoryLeakIndicators::default(),
            fragmentation: FragmentationMetrics::default(),
        }
    }
}

impl Default for MemoryLeakIndicators {
    fn default() -> Self {
        Self {
            suspected_leak: false,
            growth_rate: 0.0,
            confidence: 0.0,
            evidence: Vec::new(),
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hit_ratio: 1.0,
            miss_penalty_ns: 0.0,
            bandwidth_utilization: 0.0,
        }
    }
}

impl Default for VectorizationMetrics {
    fn default() -> Self {
        Self {
            simd_utilization: 0.0,
            vector_width_efficiency: 0.0,
            speedup_factor: 1.0,
        }
    }
}

impl<A: Float + Send + Sync> Default for ComplexityAnalysis<A> {
    fn default() -> Self {
        Self {
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(n)".to_string(),
            scaling_factors: Vec::new(),
            efficiency_trends: EfficiencyTrends::default(),
        }
    }
}

impl<A: Float + Send + Sync> Default for EfficiencyTrends<A> {
    fn default() -> Self {
        Self {
            degradation_rate: 0.0,
            improvement_opportunities: Vec::new(),
            bottlenecks: Vec::new(),
        }
    }
}

impl<A: Float + Send + Sync> Default for ComputationalMetrics<A> {
    fn default() -> Self {
        Self {
            average_flops: 0.0,
            peak_flops: 0.0,
            arithmetic_intensity: 0.0,
            cache_efficiency: 1.0,
            vectorization_efficiency: 0.0,
            efficiency_score: 1.0,
            bottlenecks: Vec::new(),
        }
    }
}

impl<A: Float + Send + Sync> Default for GradientMetrics<A> {
    fn default() -> Self {
        Self {
            magnitude_stats: GradientMagnitudeStats::default(),
            direction_analysis: GradientDirectionAnalysis::default(),
            stability_metrics: GradientStabilityMetrics::default(),
            learning_dynamics: LearningDynamicsAnalysis::default(),
        }
    }
}

impl<A: Float + Send + Sync> Default for GradientMagnitudeStats<A> {
    fn default() -> Self {
        Self {
            mean_magnitude: A::zero(),
            std_magnitude: A::zero(),
            magnitude_trend: A::zero(),
            explosion_indicators: Vec::new(),
            vanishing_indicators: Vec::new(),
        }
    }
}

impl<A: Float + Send + Sync> Default for GradientDirectionAnalysis<A> {
    fn default() -> Self {
        Self {
            consistency_score: A::one(),
            oscillation_frequency: 0.0,
            change_patterns: Vec::new(),
        }
    }
}

impl<A: Float + Send + Sync> Default for GradientStabilityMetrics<A> {
    fn default() -> Self {
        Self {
            stability_score: 1.0,
            noise_level: A::zero(),
            signal_to_noise_ratio: A::infinity(),
        }
    }
}

impl<A: Float + Send + Sync> Default for LearningDynamicsAnalysis<A> {
    fn default() -> Self {
        Self {
            lr_adaptation_effectiveness: 1.0,
            momentum_effectiveness: 1.0,
            second_order_utilization: 0.0,
            convergence_velocity: A::zero(),
        }
    }
}

impl<A: Float + Send + Sync> Default for ConvergenceMetrics<A> {
    fn default() -> Self {
        Self {
            status: ConvergenceStatus::SteadyConvergence,
            convergence_rate: 0.0,
            estimated_time_to_convergence: None,
            quality_score: 1.0,
            patterns: Vec::new(),
        }
    }
}

impl Default for HardwareEfficiencyMetrics {
    fn default() -> Self {
        Self {
            overall_utilization: 0.0,
            cpu_efficiency: 0.0,
            memory_efficiency: 0.0,
            cache_efficiency: 1.0,
            gpu_efficiency: None,
        }
    }
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self {
            avg_cpu_utilization: 0.0,
            peak_cpu_utilization: 0.0,
            memory_bandwidth_utilization: 0.0,
            gpu_utilization: None,
            efficiency_summary: HardwareEfficiencyMetrics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_profiler(config: ProfilerConfig) -> PerformanceProfiler<f64> {
        match PerformanceProfiler::<f64>::new(config) {
            Ok(p) => p,
            Err(e) => panic!("failed to create PerformanceProfiler: {e:?}"),
        }
    }

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = new_profiler(config);
        assert_eq!(profiler.current_step, 0);
    }

    #[test]
    fn test_step_profiling() {
        let config = ProfilerConfig::default();
        let mut profiler = new_profiler(config);

        let mut step_profiler = profiler.start_step();
        step_profiler.start_gradient_computation();
        std::thread::sleep(std::time::Duration::from_millis(1));
        step_profiler.end_gradient_computation();

        step_profiler.start_parameter_update();
        std::thread::sleep(std::time::Duration::from_millis(1));
        step_profiler.end_parameter_update();

        if let Err(e) = profiler.complete_step(step_profiler) {
            panic!("complete_step failed: {e:?}");
        }
        assert_eq!(profiler.current_step, 1);
    }

    #[test]
    fn test_performance_report_generation() {
        let config = ProfilerConfig::default();
        let profiler = new_profiler(config);

        let report = profiler.generate_performance_report();
        assert!(report.performance_score >= 0.0 && report.performance_score <= 1.0);
    }

    #[test]
    fn test_memory_leak_detection() {
        let config = ProfilerConfig::default();
        let profiler = new_profiler(config);

        let leak_indicators = profiler.detect_memory_leaks();
        assert!(leak_indicators.confidence >= 0.0 && leak_indicators.confidence <= 1.0);
    }

    /// With no steps run, there is no real data, so honest recommendations
    /// must be empty (previously this always fired even with zero data,
    /// because the fallback "no history" values of 0.0 always compared as
    /// "low"). Once a step with a real, low, declared op count is
    /// recorded, a genuine low-throughput recommendation should appear.
    #[test]
    fn test_efficiency_recommendations_require_real_data() {
        let config = ProfilerConfig::default();
        let profiler = new_profiler(config);
        let recommendations = profiler.generate_efficiency_recommendations();
        assert!(
            recommendations.is_empty(),
            "no steps have been recorded yet, so there must be no fabricated recommendations"
        );
    }

    #[test]
    fn test_efficiency_recommendations_fire_on_real_low_throughput() {
        let config = ProfilerConfig::default();
        let mut profiler = new_profiler(config);

        let mut step_profiler = profiler.start_step();
        step_profiler.record_op_count(10); // trivially low real op count
        if let Err(e) = profiler.complete_step(step_profiler) {
            panic!("complete_step failed: {e:?}");
        }

        let recommendations = profiler.generate_efficiency_recommendations();
        assert!(
            recommendations
                .iter()
                .any(|r| matches!(r.category, RecommendationCategory::Computation)),
            "a real, tiny declared op count must yield a genuine low-throughput recommendation"
        );
    }

    #[test]
    fn test_flops_require_declared_op_count() {
        let config = ProfilerConfig::default();
        let mut profiler = new_profiler(config);

        // A step with no declared op count must not contribute a
        // fabricated FLOPS sample.
        let step_profiler = profiler.start_step();
        if let Err(e) = profiler.complete_step(step_profiler) {
            panic!("complete_step failed: {e:?}");
        }
        assert!(profiler.efficiency_analyzer.flops_history.is_empty());

        // A step with a declared op count must contribute a real sample.
        let mut step_profiler = profiler.start_step();
        step_profiler.record_op_count(1_000_000);
        if let Err(e) = profiler.complete_step(step_profiler) {
            panic!("complete_step failed: {e:?}");
        }
        assert_eq!(profiler.efficiency_analyzer.flops_history.len(), 1);
    }
}
