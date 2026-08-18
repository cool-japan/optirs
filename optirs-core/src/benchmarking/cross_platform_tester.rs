// Cross-platform performance testing and benchmarking
//
// This module provides cross-platform performance testing capabilities
// for optimization algorithms across different hardware targets.

use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

// SciRS2 Integration - ESSENTIAL for benchmarking

use crate::error::{OptimError, Result};

/// Platform target for cross-platform testing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlatformTarget {
    /// CPU-based execution
    CPU,
    /// CUDA GPU
    CUDA,
    /// Metal GPU (macOS)
    Metal,
    /// OpenCL GPU
    OpenCL,
    /// WebGPU
    WebGPU,
    /// TPU
    TPU,
    /// Custom platform
    Custom(String),
}

/// Performance baseline for comparison
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub target: PlatformTarget,
    pub throughput_ops_per_sec: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub energy_consumption_joules: Option<f64>,
    pub accuracy_metrics: HashMap<String, f64>,
}

impl PerformanceBaseline {
    pub fn new(target: PlatformTarget) -> Self {
        Self {
            target,
            throughput_ops_per_sec: 0.0,
            latency_ms: 0.0,
            memory_usage_mb: 0.0,
            energy_consumption_joules: None,
            accuracy_metrics: HashMap::new(),
        }
    }

    pub fn with_throughput(mut self, ops_per_sec: f64) -> Self {
        self.throughput_ops_per_sec = ops_per_sec;
        self
    }

    pub fn with_latency(mut self, latency_ms: f64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    pub fn with_memory_usage(mut self, memory_mb: f64) -> Self {
        self.memory_usage_mb = memory_mb;
        self
    }
}

/// Cross-platform performance tester
#[derive(Debug)]
pub struct CrossPlatformTester {
    baselines: HashMap<PlatformTarget, PerformanceBaseline>,
    test_configurations: HashMap<String, TestConfiguration>,
}

/// Test configuration for benchmarking
#[derive(Debug, Clone)]
pub struct TestConfiguration {
    pub name: String,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub data_size: usize,
    pub timeout: Duration,
}

impl CrossPlatformTester {
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            test_configurations: HashMap::new(),
        }
    }

    pub fn add_baseline(&mut self, baseline: PerformanceBaseline) {
        self.baselines.insert(baseline.target.clone(), baseline);
    }

    pub fn add_test_config(&mut self, config: TestConfiguration) {
        self.test_configurations.insert(config.name.clone(), config);
    }

    /// Run a benchmark for the named test configuration against `target`,
    /// executing `op` for both warmup and timed iterations.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] if the test configuration is
    /// unknown or requests zero iterations, [`OptimError::UnsupportedOperation`]
    /// if `target` is not a backend compiled into `optirs-core` (only
    /// [`PlatformTarget::CPU`] is supported here; accelerator targets belong
    /// to the dedicated `optirs-gpu` / `optirs-tpu` crates), and
    /// [`OptimError::ExecutionError`] if the configured timeout elapses
    /// before all iterations complete.
    pub fn run_benchmark<Op>(
        &self,
        target: &PlatformTarget,
        test_name: &str,
        mut op: Op,
    ) -> Result<PerformanceBaseline>
    where
        Op: FnMut(),
    {
        if !matches!(target, PlatformTarget::CPU) {
            return Err(OptimError::UnsupportedOperation(format!(
                "platform target {target:?} is not compiled into optirs-core (CPU-only); \
                 use the dedicated accelerator crate (optirs-gpu / optirs-tpu) for this target"
            )));
        }

        let config = self.test_configurations.get(test_name).ok_or_else(|| {
            OptimError::InvalidConfig(format!("Test configuration '{}' not found", test_name))
        })?;

        if config.iterations == 0 {
            return Err(OptimError::InvalidConfig(format!(
                "test configuration '{}' must run at least one iteration",
                test_name
            )));
        }

        let overall_start = Instant::now();

        // Untimed warmup iterations (still bounded by the overall timeout).
        for _ in 0..config.warmup_iterations {
            if overall_start.elapsed() > config.timeout {
                return Err(OptimError::ExecutionError(format!(
                    "benchmark '{}' exceeded timeout {:?} during warmup",
                    test_name, config.timeout
                )));
            }
            op();
        }

        // Timed iterations, measured as ONE batch. Timing each iteration
        // separately quantizes a sub-tick `op` to a zero `Duration` on coarse
        // monotonic clocks, and the zeros used to sum to a zero total that was
        // then reported as *infinite* throughput — a fabricated number that
        // intermittently failed the finiteness test under scheduler jitter.
        // (The per-iteration timeout check is inside the measured window; this
        // helper benchmarks coarse workloads, not nanosecond kernels.)
        let mut completed = 0usize;
        let timed_start = Instant::now();
        for _ in 0..config.iterations {
            if overall_start.elapsed() > config.timeout {
                return Err(OptimError::ExecutionError(format!(
                    "benchmark '{}' exceeded timeout {:?} after {} of {} iterations",
                    test_name, config.timeout, completed, config.iterations
                )));
            }
            op();
            completed += 1;
        }

        let total_secs = timed_start.elapsed().as_secs_f64();
        if total_secs <= 0.0 {
            // Faster than the clock can resolve: the honest answer is that no
            // throughput was measured, not that it was infinite.
            return Err(OptimError::ExecutionError(format!(
                "benchmark '{}' completed {} iterations faster than the \
                 monotonic clock can resolve; increase iterations or data_size \
                 to get a measurable run",
                test_name, completed
            )));
        }
        let (throughput, latency_ms) = (
            config.iterations as f64 / total_secs,
            total_secs * 1000.0 / config.iterations as f64,
        );

        Ok(PerformanceBaseline::new(target.clone())
            .with_throughput(throughput)
            .with_latency(latency_ms))
    }

    pub fn compare_performance(
        &self,
        target1: &PlatformTarget,
        target2: &PlatformTarget,
    ) -> Result<f64> {
        let baseline1 = self.baselines.get(target1).ok_or_else(|| {
            OptimError::InvalidConfig("Baseline for target1 not found".to_string())
        })?;
        let baseline2 = self.baselines.get(target2).ok_or_else(|| {
            OptimError::InvalidConfig("Baseline for target2 not found".to_string())
        })?;

        Ok(baseline1.throughput_ops_per_sec / baseline2.throughput_ops_per_sec)
    }
}

impl Default for CrossPlatformTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_tester(
        iterations: usize,
        warmup_iterations: usize,
        timeout: Duration,
    ) -> CrossPlatformTester {
        let mut tester = CrossPlatformTester::new();
        tester.add_test_config(TestConfiguration {
            name: "op".to_string(),
            iterations,
            warmup_iterations,
            data_size: 1,
            timeout,
        });
        tester
    }

    #[test]
    fn run_benchmark_invokes_closure_for_warmup_and_timed_iterations() {
        let tester = make_tester(5, 3, Duration::from_secs(10));
        let calls = AtomicUsize::new(0);

        let result = tester.run_benchmark(&PlatformTarget::CPU, "op", || {
            calls.fetch_add(1, Ordering::SeqCst);
            // Measurable, optimization-proof work: a bare atomic increment can
            // finish inside one tick of a coarse monotonic clock, which is the
            // zero-total case run_benchmark now rejects as unmeasurable.
            let mut acc = 0u64;
            for i in 0..10_000u64 {
                acc = acc.wrapping_add(std::hint::black_box(i));
            }
            std::hint::black_box(acc);
        });

        assert!(result.is_ok());
        // 3 warmup + 5 timed = 8 total invocations of the closure.
        assert_eq!(calls.load(Ordering::SeqCst), 8);
        let baseline = result.expect("unwrap failed");
        assert!(baseline.throughput_ops_per_sec.is_finite());
        assert!(baseline.throughput_ops_per_sec > 0.0);
    }

    #[test]
    fn run_benchmark_rejects_uncompiled_backend_targets() {
        let tester = make_tester(1, 0, Duration::from_secs(10));

        for target in [
            PlatformTarget::CUDA,
            PlatformTarget::Metal,
            PlatformTarget::OpenCL,
            PlatformTarget::WebGPU,
            PlatformTarget::TPU,
        ] {
            let result = tester.run_benchmark(&target, "op", || {});
            assert!(
                result.is_err(),
                "expected {:?} to be rejected as an uncompiled backend",
                target
            );
        }
    }

    #[test]
    fn run_benchmark_honors_timeout() {
        // A tight timeout with many iterations that individually sleep past it
        // must abort with an error instead of running to completion.
        let tester = make_tester(1_000_000, 0, Duration::from_millis(5));

        let result = tester.run_benchmark(&PlatformTarget::CPU, "op", || {
            std::thread::sleep(Duration::from_millis(2));
        });

        assert!(result.is_err());
    }

    #[test]
    fn run_benchmark_rejects_unknown_test_config() {
        let tester = CrossPlatformTester::new();
        let result = tester.run_benchmark(&PlatformTarget::CPU, "missing", || {});
        assert!(result.is_err());
    }

    #[test]
    fn run_benchmark_rejects_zero_iterations() {
        let tester = make_tester(0, 0, Duration::from_secs(10));
        let result = tester.run_benchmark(&PlatformTarget::CPU, "op", || {});
        assert!(result.is_err());
    }
}
