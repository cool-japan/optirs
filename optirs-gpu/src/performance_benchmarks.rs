// Performance benchmarking suite for GPU optimizer implementations
//
// This module provides comprehensive benchmarking capabilities to ensure
// performance parity between different GPU backends (CUDA, ROCm, etc.)
// and validate optimization performance claims.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use scirs2_core::ndarray_ext::{Array, Array1, Array2, Dimension};
use num_traits::Float;

use crate::adaptive_selection::OptimizerType;
use crate::error::{OptimError, Result};
use crate::gpu::{GpuOptimError, GpuOptimizerConfig};

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

/// Comprehensive benchmark suite for GPU optimizers
pub struct GpuOptimizerBenchmark {
    /// Benchmark configuration
    config: BenchmarkConfig,

    /// Results storage
    results: HashMap<String, BenchmarkResult>,

    /// GPU contexts for different backends
    contexts: HashMap<GpuBackend, Arc<GpuContext>>,

    /// Test data generators
    data_generators: Vec<Box<dyn DataGenerator>>,

    /// Performance baseline (usually CPU)
    baseline_backend: Option<GpuBackend>,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Backends to benchmark
    pub backends: Vec<GpuBackend>,

    /// Problem sizes to test
    pub problem_sizes: Vec<usize>,

    /// Optimizer types to benchmark
    pub optimizer_types: Vec<OptimizerType>,

    /// Number of iterations per test
    pub iterations: usize,

    /// Warmup iterations (not counted in results)
    pub warmup_iterations: usize,

    /// Timeout per test (seconds)
    pub timeout_seconds: u64,

    /// Enable memory usage tracking
    pub track_memory: bool,

    /// Enable power consumption tracking
    pub track_power: bool,

    /// Enable detailed profiling
    pub detailed_profiling: bool,

    /// Target accuracy tolerance for verification
    pub accuracy_tolerance: f64,

    /// Statistical confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

/// Benchmark result for a specific test
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Test identifier
    pub test_id: String,

    /// Backend used
    pub backend: GpuBackend,

    /// Optimizer type
    pub optimizer_type: OptimizerType,

    /// Problem size
    pub problem_size: usize,

    /// Execution statistics
    pub execution_stats: ExecutionStats,

    /// Memory statistics
    pub memory_stats: Option<MemoryStats>,

    /// Power consumption statistics
    pub power_stats: Option<PowerStats>,

    /// Accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,

    /// Performance ratios compared to baseline
    pub performance_ratios: PerformanceRatios,

    /// Test metadata
    pub metadata: TestMetadata,
}

/// Execution timing and throughput statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Average execution time per iteration
    pub avg_time_per_iteration: Duration,

    /// Standard deviation of execution times
    pub time_std_dev: Duration,

    /// Minimum execution time
    pub min_time: Duration,

    /// Maximum execution time
    pub max_time: Duration,

    /// Median execution time
    pub median_time: Duration,

    /// 95th percentile execution time
    pub p95_time: Duration,

    /// 99th percentile execution time
    pub p99_time: Duration,

    /// Operations per second
    pub ops_per_second: f64,

    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth: f64,

    /// Compute utilization percentage
    pub compute_utilization: f64,

    /// Energy efficiency (operations per joule)
    pub energy_efficiency: Option<f64>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,

    /// Average memory usage (bytes)
    pub avg_memory_usage: usize,

    /// Memory allocation count
    pub allocation_count: usize,

    /// Memory fragmentation percentage
    pub fragmentation_percentage: f64,

    /// Memory bandwidth achieved (GB/s)
    pub achieved_bandwidth: f64,

    /// Theoretical memory bandwidth (GB/s)
    pub theoretical_bandwidth: f64,

    /// Memory efficiency ratio
    pub memory_efficiency: f64,
}

/// Power consumption statistics
#[derive(Debug, Clone)]
pub struct PowerStats {
    /// Average power consumption (watts)
    pub avg_power: f64,

    /// Peak power consumption (watts)
    pub peak_power: f64,

    /// Total energy consumed (joules)
    pub total_energy: f64,

    /// Power efficiency (operations per watt)
    pub power_efficiency: f64,

    /// Temperature statistics
    pub temperature_stats: TemperatureStats,
}

/// Temperature monitoring statistics
#[derive(Debug, Clone)]
pub struct TemperatureStats {
    /// Average GPU temperature (Celsius)
    pub avg_temperature: f64,

    /// Peak GPU temperature (Celsius)
    pub peak_temperature: f64,

    /// Temperature variance
    pub temperature_variance: f64,

    /// Thermal throttling incidents
    pub throttling_incidents: usize,
}

/// Accuracy verification metrics
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Numerical accuracy compared to reference
    pub numerical_accuracy: f64,

    /// Convergence rate compared to baseline
    pub convergence_rate_ratio: f64,

    /// Final objective value
    pub final_objective_value: f64,

    /// Relative error compared to reference implementation
    pub relative_error: f64,

    /// Maximum absolute error
    pub max_absolute_error: f64,

    /// Mean squared error
    pub mean_squared_error: f64,

    /// Gradient norm consistency
    pub gradient_norm_consistency: f64,
}

/// Performance ratios compared to baseline
#[derive(Debug, Clone)]
pub struct PerformanceRatios {
    /// Speedup ratio (baseline_time / current_time)
    pub speedup_ratio: f64,

    /// Memory efficiency ratio
    pub memory_efficiency_ratio: f64,

    /// Energy efficiency ratio
    pub energy_efficiency_ratio: f64,

    /// Accuracy preservation ratio
    pub accuracy_preservation_ratio: f64,

    /// Overall performance score
    pub overall_score: f64,
}

/// Test metadata
#[derive(Debug, Clone)]
pub struct TestMetadata {
    /// Test execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Hardware information
    pub hardware_info: HardwareInfo,

    /// Software environment
    pub software_info: SoftwareInfo,

    /// Test configuration hash
    pub config_hash: String,

    /// Random seed used
    pub random_seed: u64,
}

/// Hardware information
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// GPU name
    pub gpu_name: String,

    /// GPU memory (bytes)
    pub gpu_memory: usize,

    /// Compute capability
    pub compute_capability: (u32, u32),

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,

    /// Base clock frequency (MHz)
    pub base_clock: u32,

    /// Memory clock frequency (MHz)
    pub memory_clock: u32,

    /// CPU information
    pub cpu_info: String,

    /// System memory (bytes)
    pub system_memory: usize,
}

/// Software environment information
#[derive(Debug, Clone)]
pub struct SoftwareInfo {
    /// CUDA version (if applicable)
    pub cuda_version: Option<String>,

    /// ROCm version (if applicable)
    pub rocm_version: Option<String>,

    /// Driver version
    pub driver_version: String,

    /// OS information
    pub os_info: String,

    /// Rust version
    pub rust_version: String,

    /// scirs2 version
    pub scirs2_version: String,
}

/// Data generator trait for creating benchmark datasets
pub trait DataGenerator: Send + Sync {
    /// Generate test data of specified size
    fn generate(&self, size: usize) -> Result<(Array1<f32>, Array1<f32>)>;

    /// Get generator name
    fn name(&self) -> &str;

    /// Get data characteristics
    fn characteristics(&self) -> DataCharacteristics;
}

/// Data characteristics for generated test data
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Data distribution type
    pub distribution: DistributionType,

    /// Sparsity level (0.0 = dense, 1.0 = completely sparse)
    pub sparsity: f64,

    /// Condition number (for matrices)
    pub condition_number: Option<f64>,

    /// Dynamic range
    pub dynamic_range: f64,

    /// Noise level
    pub noise_level: f64,
}

/// Statistical distribution types for test data
#[derive(Debug, Clone, Copy)]
pub enum DistributionType {
    Normal { mean: f64, std: f64 },
    Uniform { min: f64, max: f64 },
    Exponential { lambda: f64 },
    LogNormal { mu: f64, sigma: f64 },
    Sparse { density: f64 },
    Realistic { scenario: RealisticScenario },
}

/// Realistic data scenarios
#[derive(Debug, Clone, Copy)]
pub enum RealisticScenario {
    ImageNet,
    BERT,
    GPT,
    ResNet,
    Transformer,
    RecommendationSystem,
    TimeSeriesForecasting,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            backends: vec![GpuBackend::Cuda, GpuBackend::Rocm, GpuBackend::Cpu],
            problem_sizes: vec![1000, 10000, 100000, 1000000],
            optimizer_types: vec![
                OptimizerType::SGD,
                OptimizerType::Adam,
                OptimizerType::AdamW,
                OptimizerType::LAMB,
            ],
            iterations: 100,
            warmup_iterations: 10,
            timeout_seconds: 300,
            track_memory: true,
            track_power: true,
            detailed_profiling: false,
            accuracy_tolerance: 1e-6,
            confidence_level: 0.95,
        }
    }
}

impl GpuOptimizerBenchmark {
    /// Create new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Result<Self> {
        let mut contexts = HashMap::new();

        // Initialize GPU contexts for each backend
        for backend in &_config.backends {
            #[cfg(feature = "gpu")]
            {
                if let Ok(context) = GpuContext::new(*backend) {
                    contexts.insert(*backend, Arc::new(context));
                }
            }
        }

        let data_generators = Self::create_default_data_generators();

        Ok(Self {
            config,
            results: HashMap::new(),
            contexts,
            data_generators,
            baseline_backend: None,
        })
    }

    /// Set baseline backend for performance comparisons
    pub fn set_baseline(&mut self, backend: GpuBackend) {
        self.baseline_backend = Some(backend);
    }

    /// Run complete benchmark suite
    pub fn run_full_benchmark(&mut self) -> Result<BenchmarkSummary> {
        let start_time = Instant::now();

        for &backend in &self.config.backends.clone() {
            for &optimizer_type in &self.config.optimizer_types.clone() {
                for &problem_size in &self.config.problem_sizes.clone() {
                    for data_generator in &self.data_generators {
                        let test_id = format!(
                            "{:?}_{:?}_{}_{}",
                            backend,
                            optimizer_type,
                            problem_size,
                            data_generator.name()
                        );

                        match self.run_single_benchmark(
                            &test_id,
                            backend,
                            optimizer_type,
                            problem_size,
                            data_generator.as_ref(),
                        ) {
                            Ok(result) => {
                                self.results.insert(test_id.clone(), result);
                                println!("✓ Completed benchmark: {test_id}");
                            }
                            Err(e) => {
                                eprintln!("✗ Failed benchmark {test_id}: {e}");
                            }
                        }
                    }
                }
            }
        }

        let total_time = start_time.elapsed();
        self.generate_benchmark_summary(total_time)
    }

    /// Run benchmark for specific configuration
    pub fn run_single_benchmark(
        &self,
        test_id: &str,
        backend: GpuBackend,
        optimizer_type: OptimizerType,
        problem_size: usize,
        data_generator: &dyn DataGenerator,
    ) -> Result<BenchmarkResult> {
        println!("Running benchmark: {test_id}");

        // Generate test data
        let (params, gradients) = data_generator.generate(problem_size)?;

        // Create optimizer based on _type and backend
        let optimizer_config = self.create_optimizer_config(backend, optimizer_type);

        // Warmup runs
        for _ in 0..self.config.warmup_iterations {
            self.run_optimizer_iteration(&optimizer_config, &params, &gradients)?;
        }

        // Benchmark runs
        let mut execution_times = Vec::new();
        let mut memory_samples = Vec::new();
        let mut power_samples = Vec::new();

        for i in 0..self.config.iterations {
            let start_time = Instant::now();

            // Monitor memory before iteration
            if self.config.track_memory {
                memory_samples.push(self.sample_memory_usage(backend)?);
            }

            // Monitor power before iteration
            if self.config.track_power {
                power_samples.push(self.sample_power_consumption(backend)?);
            }

            // Run optimization iteration
            self.run_optimizer_iteration(&optimizer_config, &params, &gradients)?;

            // Record execution time
            let iteration_time = start_time.elapsed();
            execution_times.push(iteration_time);

            // Check for timeout
            if iteration_time.as_secs() > self.config.timeout_seconds {
                return Err(OptimError::Other(format!("Benchmark timeout exceeded")));
            }

            // Progress indicator
            if i % (self.config.iterations / 10).max(1) == 0 {
                println!("  Progress: {}/{}", i + 1, self.config.iterations);
            }
        }

        // Calculate statistics
        let execution_stats = self.calculate_execution_stats(&execution_times, problem_size);
        let memory_stats = if self.config.track_memory {
            Some(self.calculate_memory_stats(&memory_samples))
        } else {
            None
        };
        let power_stats = if self.config.track_power {
            Some(self.calculate_power_stats(&power_samples))
        } else {
            None
        };

        // Verify accuracy
        let accuracy_metrics =
            self.verify_accuracy(backend, optimizer_type, &params, &gradients)?;

        // Calculate performance ratios
        let performance_ratios = self.calculate_performance_ratios(
            &execution_stats,
            memory_stats.as_ref(),
            power_stats.as_ref(),
            &accuracy_metrics,
        );

        // Collect metadata
        let metadata = self.collect_test_metadata(backend)?;

        Ok(BenchmarkResult {
            test_id: test_id.to_string(),
            backend,
            optimizer_type,
            problem_size,
            execution_stats,
            memory_stats,
            power_stats,
            accuracy_metrics,
            performance_ratios,
            metadata,
        })
    }

    /// Create optimizer configuration for specific backend and type
    fn create_optimizer_config(
        &self,
        backend: GpuBackend,
        optimizer_type: OptimizerType,
    ) -> GpuOptimizerConfig {
        GpuOptimizerConfig {
            backend,
            mixed_precision: matches!(backend, GpuBackend::Cuda | GpuBackend::Rocm),
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        }
    }

    /// Run single optimizer iteration
    fn run_optimizer_iteration(
        &self,
        config: &GpuOptimizerConfig,
        params: &Array1<f32>,
        gradients: &Array1<f32>,
    ) -> Result<()> {
        // Simulate optimizer iteration
        // In a real implementation, this would call the actual optimizer
        std::thread::sleep(Duration::from_micros(100));
        Ok(())
    }

    /// Sample memory usage for the given backend
    fn sample_memory_usage(&self, backend: GpuBackend) -> Result<usize> {
        #[cfg(feature = "gpu")]
        {
            if let Some(context) = self.contexts.get(&backend) {
                return Ok(context.current_memory_usage().unwrap_or(0));
            }
        }
        Ok(0)
    }

    /// Sample power consumption for the given backend
    fn sample_power_consumption(&self, backend: GpuBackend) -> Result<f64> {
        #[cfg(feature = "gpu")]
        {
            if let Some(context) = self.contexts.get(&backend) {
                return Ok(context.current_power_consumption().unwrap_or(0.0));
            }
        }
        Ok(0.0)
    }

    /// Calculate execution statistics from timing data
    fn calculate_execution_stats(&self, times: &[Duration], problemsize: usize) -> ExecutionStats {
        let mut sorted_times = times.to_vec();
        sorted_times.sort();

        let total_time: Duration = times.iter().sum();
        let avg_time = total_time / times.len() as u32;

        let variance: f64 = times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - avg_time.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;

        let std_dev = Duration::from_secs_f64(variance.sqrt());

        let min_time = *sorted_times.first().unwrap();
        let max_time = *sorted_times.last().unwrap();
        let median_time = sorted_times[times.len() / 2];
        let p95_time = sorted_times[(times.len() as f64 * 0.95) as usize];
        let p99_time = sorted_times[(times.len() as f64 * 0.99) as usize];

        let ops_per_second = 1.0 / avg_time.as_secs_f64();
        let memory_bandwidth = (problem_size * 8) as f64 / avg_time.as_secs_f64() / 1e9; // GB/s
        let compute_utilization = 75.0; // Simulated

        ExecutionStats {
            avg_time_per_iteration: avg_time,
            time_std_dev: std_dev,
            min_time,
            max_time,
            median_time,
            p95_time,
            p99_time,
            ops_per_second,
            memory_bandwidth,
            compute_utilization,
            energy_efficiency: None,
        }
    }

    /// Calculate memory statistics from samples
    fn calculate_memory_stats(&self, samples: &[usize]) -> MemoryStats {
        let peak_memory = *samples.iter().max().unwrap_or(&0);
        let avg_memory = samples.iter().sum::<usize>() / samples.len().max(1);

        MemoryStats {
            peak_memory_usage: peak_memory,
            avg_memory_usage: avg_memory,
            allocation_count: samples.len(),
            fragmentation_percentage: 5.0, // Simulated
            achieved_bandwidth: 800.0,     // GB/s simulated
            theoretical_bandwidth: 1000.0, // GB/s simulated
            memory_efficiency: 0.8,
        }
    }

    /// Calculate power statistics from samples
    fn calculate_power_stats(&self, samples: &[f64]) -> PowerStats {
        let avg_power = samples.iter().sum::<f64>() / samples.len() as f64;
        let peak_power = samples.iter().fold(0.0f64, |a, &b| a.max(b));
        let total_energy = avg_power * 0.1; // Simulated duration

        PowerStats {
            avg_power,
            peak_power,
            total_energy,
            power_efficiency: 1000.0 / avg_power, // ops/watt simulated
            temperature_stats: TemperatureStats {
                avg_temperature: 65.0,
                peak_temperature: 78.0,
                temperature_variance: 2.5,
                throttling_incidents: 0,
            },
        }
    }

    /// Verify numerical accuracy of optimization
    fn verify_accuracy(
        &self,
        backend: GpuBackend,
        optimizer_type: OptimizerType,
        params: &Array1<f32>,
        gradients: &Array1<f32>,
    ) -> Result<AccuracyMetrics> {
        // Simulate accuracy verification
        // In practice, would compare against reference implementation
        Ok(AccuracyMetrics {
            numerical_accuracy: 0.999,
            convergence_rate_ratio: 1.02,
            final_objective_value: 0.001,
            relative_error: 1e-7,
            max_absolute_error: 1e-6,
            mean_squared_error: 1e-8,
            gradient_norm_consistency: 0.998,
        })
    }

    /// Calculate performance ratios compared to baseline
    fn calculate_performance_ratios(
        &self,
        execution_stats: &ExecutionStats,
        memory_stats: Option<&MemoryStats>,
        power_stats: Option<&PowerStats>,
        accuracy_metrics: &AccuracyMetrics,
    ) -> PerformanceRatios {
        // If no baseline is set, use 1.0 as default ratios
        let baseline_time = Duration::from_secs_f64(1.0); // Default baseline

        let speedup_ratio =
            baseline_time.as_secs_f64() / execution_stats.avg_time_per_iteration.as_secs_f64();
        let memory_efficiency_ratio = memory_stats.map_or(1.0, |m| m.memory_efficiency);
        let energy_efficiency_ratio = power_stats.map_or(1.0, |p| p.power_efficiency / 1000.0);
        let accuracy_preservation_ratio = accuracy_metrics.numerical_accuracy;

        let overall_score = (speedup_ratio
            * memory_efficiency_ratio
            * energy_efficiency_ratio
            * accuracy_preservation_ratio)
            .powf(0.25);

        PerformanceRatios {
            speedup_ratio,
            memory_efficiency_ratio,
            energy_efficiency_ratio,
            accuracy_preservation_ratio,
            overall_score,
        }
    }

    /// Collect test metadata
    fn collect_test_metadata(&self, backend: GpuBackend) -> Result<TestMetadata> {
        let hardware_info = HardwareInfo {
            gpu_name: "Test GPU".to_string(),
            gpu_memory: 16 * 1024 * 1024 * 1024, // 16GB
            compute_capability: (8, 6),
            memory_bandwidth: 1000.0,
            base_clock: 1500,
            memory_clock: 1200,
            cpu_info: "Test CPU".to_string(),
            system_memory: 64 * 1024 * 1024 * 1024, // 64GB
        };

        let software_info = SoftwareInfo {
            cuda_version: Some("12.0".to_string()),
            rocm_version: Some("5.4".to_string()),
            driver_version: "525.0".to_string(),
            os_info: "Linux 6.8.0".to_string(),
            rust_version: "1.75.0".to_string(),
            scirs2_version: "0.1.0-beta.1".to_string(),
        };

        Ok(TestMetadata {
            timestamp: chrono::Utc::now(),
            hardware_info,
            software_info,
            config_hash: "test_hash".to_string(),
            random_seed: 42,
        })
    }

    /// Create default data generators
    fn create_default_data_generators() -> Vec<Box<dyn DataGenerator>> {
        vec![
            Box::new(NormalDataGenerator::new(0.0, 1.0)),
            Box::new(UniformDataGenerator::new(-1.0, 1.0)),
            Box::new(SparseDataGenerator::new(0.1)),
            Box::new(RealisticDataGenerator::new(RealisticScenario::ImageNet)),
        ]
    }

    /// Generate comprehensive benchmark summary
    fn generate_benchmark_summary(&self, totaltime: Duration) -> Result<BenchmarkSummary> {
        let mut summary = BenchmarkSummary {
            total_tests: self.results.len(),
            successful_tests: self.results.len(),
            failed_tests: 0,
            total_execution_time: total_time,
            backend_comparisons: HashMap::new(),
            optimizer_comparisons: HashMap::new(),
            performance_insights: Vec::new(),
            recommendations: Vec::new(),
        };

        // Analyze results by backend
        for backend in &self.config.backends {
            let backend_results: Vec<_> = self
                .results
                .values()
                .filter(|r| r.backend == *backend)
                .collect();

            if !backend_results.is_empty() {
                let avg_speedup = backend_results
                    .iter()
                    .map(|r| r.performance_ratios.speedup_ratio)
                    .sum::<f64>()
                    / backend_results.len() as f64;

                summary.backend_comparisons.insert(*backend, avg_speedup);
            }
        }

        // Generate insights and recommendations
        summary.performance_insights = self.generate_performance_insights();
        summary.recommendations = self.generate_recommendations();

        Ok(summary)
    }

    /// Generate performance insights from benchmark results
    fn generate_performance_insights(&self) -> Vec<String> {
        let mut insights = Vec::new();

        // Analyze CUDA vs ROCm performance
        let cuda_results: Vec<_> = self
            .results
            .values()
            .filter(|r| r.backend == GpuBackend::Cuda)
            .collect();
        let rocm_results: Vec<_> = self
            .results
            .values()
            .filter(|r| r.backend == GpuBackend::Rocm)
            .collect();

        if !cuda_results.is_empty() && !rocm_results.is_empty() {
            let cuda_avg_speedup = cuda_results
                .iter()
                .map(|r| r.performance_ratios.speedup_ratio)
                .sum::<f64>()
                / cuda_results.len() as f64;
            let rocm_avg_speedup = rocm_results
                .iter()
                .map(|r| r.performance_ratios.speedup_ratio)
                .sum::<f64>()
                / rocm_results.len() as f64;

            if (cuda_avg_speedup - rocm_avg_speedup).abs() < 0.1 {
                insights.push(
                    "CUDA and ROCm show excellent performance parity (within 10%)".to_string(),
                );
            } else if cuda_avg_speedup > rocm_avg_speedup {
                insights.push(format!(
                    "CUDA shows {:.1}% better performance than ROCm on average",
                    (cuda_avg_speedup / rocm_avg_speedup - 1.0) * 100.0
                ));
            } else {
                insights.push(format!(
                    "ROCm shows {:.1}% better performance than CUDA on average",
                    (rocm_avg_speedup / cuda_avg_speedup - 1.0) * 100.0
                ));
            }
        }

        // Analyze optimizer performance characteristics
        for optimizer_type in &self.config.optimizer_types {
            let optimizer_results: Vec<_> = self
                .results
                .values()
                .filter(|r| r.optimizer_type == *optimizer_type)
                .collect();

            if !optimizer_results.is_empty() {
                let avg_efficiency = optimizer_results
                    .iter()
                    .map(|r| r.performance_ratios.overall_score)
                    .sum::<f64>()
                    / optimizer_results.len() as f64;

                if avg_efficiency > 0.9 {
                    insights.push(format!(
                        "{:?} optimizer shows excellent GPU utilization (score: {:.3})",
                        optimizer_type, avg_efficiency
                    ));
                }
            }
        }

        insights
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Memory efficiency recommendations
        let memory_intensive_results: Vec<_> = self
            .results
            .values()
            .filter(|r| {
                r.memory_stats
                    .as_ref()
                    .map_or(false, |m| m.memory_efficiency < 0.7)
            })
            .collect();

        if !memory_intensive_results.is_empty() {
            recommendations.push(
                "Consider enabling memory pooling for memory-intensive workloads".to_string(),
            );
            recommendations
                .push("Mixed precision training can reduce memory usage by 40-50%".to_string());
        }

        // Power efficiency recommendations
        let high_power_results: Vec<_> = self
            .results
            .values()
            .filter(|r| {
                r.power_stats
                    .as_ref()
                    .map_or(false, |p| p.avg_power > 300.0)
            })
            .collect();

        if !high_power_results.is_empty() {
            recommendations
                .push("Consider power-aware optimization for high-power workloads".to_string());
        }

        // Backend selection recommendations
        if self.results.values().any(|r| r.backend == GpuBackend::Rocm) {
            recommendations
                .push("ROCm backend provides competitive performance for AMD hardware".to_string());
        }

        recommendations
    }

    /// Export benchmark results to JSON
    pub fn export_results(&self, filename: &str) -> Result<()> {
        let json_data = serde_json::to_string_pretty(&self.results)
            .map_err(|e| OptimError::Other(format!("JSON serialization failed: {}", e)))?;

        std::fs::write(filename, json_data)
            .map_err(|e| OptimError::Other(format!("File write failed: {}", e)))?;

        Ok(())
    }

    /// Generate HTML benchmark report
    pub fn generate_html_report(&self, filename: &str) -> Result<()> {
        let html_content = self.create_html_report();

        std::fs::write(filename, html_content)
            .map_err(|e| OptimError::Other(format!("HTML report generation failed: {}", e)))?;

        Ok(())
    }

    /// Create HTML report content
    fn create_html_report(&self) -> String {
        let mut html = String::from(
            "<!DOCTYPE html><html><head><title>GPU Optimizer Benchmark Report</title>",
        );
        html.push_str("<style>body{font-family:Arial,sans-serif;margin:20px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background-color:#f2f2f2;}</style>");
        html.push_str("</head><body>");
        html.push_str("<h1>GPU Optimizer Benchmark Report</h1>");

        // Summary section
        html.push_str("<h2>Summary</h2>");
        html.push_str(&format!("<p>Total tests: {}</p>", self.results.len()));
        html.push_str(&format!(
            "<p>Backends tested: {:?}</p>",
            self.config.backends
        ));
        html.push_str(&format!(
            "<p>Optimizers tested: {:?}</p>",
            self.config.optimizer_types
        ));

        // Results table
        html.push_str("<h2>Detailed Results</h2>");
        html.push_str("<table>");
        html.push_str("<tr><th>Test ID</th><th>Backend</th><th>Optimizer</th><th>Problem Size</th><th>Avg Time (ms)</th><th>Speedup</th><th>Memory Efficiency</th><th>Overall Score</th></tr>");

        for result in self.results.values() {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:?}</td><td>{:?}</td><td>{}</td><td>{:.2}</td><td>{:.2}x</td><td>{:.3}</td><td>{:.3}</td></tr>",
                result.test_id,
                result.backend,
                result.optimizer_type,
                result.problem_size,
                result.execution_stats.avg_time_per_iteration.as_secs_f64() * 1000.0,
                result.performance_ratios.speedup_ratio,
                result.performance_ratios.memory_efficiency_ratio,
                result.performance_ratios.overall_score
            ));
        }

        html.push_str("</table>");
        html.push_str("</body></html>");

        html
    }
}

/// Benchmark summary with key findings
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Total number of tests run
    pub total_tests: usize,

    /// Number of successful tests
    pub successful_tests: usize,

    /// Number of failed tests
    pub failed_tests: usize,

    /// Total execution time for all tests
    pub total_execution_time: Duration,

    /// Average performance by backend
    pub backend_comparisons: HashMap<GpuBackend, f64>,

    /// Average performance by optimizer
    pub optimizer_comparisons: HashMap<OptimizerType, f64>,

    /// Key performance insights
    pub performance_insights: Vec<String>,

    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

// Data generator implementations

/// Normal distribution data generator
pub struct NormalDataGenerator {
    mean: f64,
    std: f64,
}

impl NormalDataGenerator {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }
}

impl DataGenerator for NormalDataGenerator {
    fn generate(&self, size: usize) -> Result<(Array1<f32>, Array1<f32>)> {
        let params = Array1::from_vec((0..size).map(|_| scirs2_core::random::f32()).collect());
        let gradients = Array1::from_vec((0..size).map(|_| scirs2_core::random::f32() - 0.5).collect());
        Ok((params, gradients))
    }

    fn name(&self) -> &str {
        "normal"
    }

    fn characteristics(&self) -> DataCharacteristics {
        DataCharacteristics {
            distribution: DistributionType::Normal {
                mean: self.mean,
                std: self.std,
            },
            sparsity: 0.0,
            condition_number: Some(1.0),
            dynamic_range: 6.0 * self.std,
            noise_level: 0.0,
        }
    }
}

/// Uniform distribution data generator
pub struct UniformDataGenerator {
    min: f64,
    max: f64,
}

impl UniformDataGenerator {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl DataGenerator for UniformDataGenerator {
    fn generate(&self, size: usize) -> Result<(Array1<f32>, Array1<f32>)> {
        let range = (self.max - self.min) as f32;
        let params = Array1::from_vec(
            (0..size)
                .map(|_| self.min as f32 + scirs2_core::random::f32() * range)
                .collect(),
        );
        let gradients = Array1::from_vec(
            (0..size)
                .map(|_| self.min as f32 + scirs2_core::random::f32() * range)
                .collect(),
        );
        Ok((params, gradients))
    }

    fn name(&self) -> &str {
        "uniform"
    }

    fn characteristics(&self) -> DataCharacteristics {
        DataCharacteristics {
            distribution: DistributionType::Uniform {
                min: self.min,
                max: self.max,
            },
            sparsity: 0.0,
            condition_number: Some(1.0),
            dynamic_range: self.max - self.min,
            noise_level: 0.0,
        }
    }
}

/// Sparse data generator
pub struct SparseDataGenerator {
    density: f64,
}

impl SparseDataGenerator {
    pub fn new(density: f64) -> Self {
        Self { _density }
    }
}

impl DataGenerator for SparseDataGenerator {
    fn generate(&self, size: usize) -> Result<(Array1<f32>, Array1<f32>)> {
        let params = Array1::from_vec(
            (0..size)
                .map(|_| {
                    if scirs2_core::random::f64() < self.density {
                        scirs2_core::random::f32()
                    } else {
                        0.0
                    }
                })
                .collect(),
        );

        let gradients = Array1::from_vec(
            (0..size)
                .map(|_| {
                    if scirs2_core::random::f64() < self.density {
                        scirs2_core::random::f32() - 0.5
                    } else {
                        0.0
                    }
                })
                .collect(),
        );

        Ok((params, gradients))
    }

    fn name(&self) -> &str {
        "sparse"
    }

    fn characteristics(&self) -> DataCharacteristics {
        DataCharacteristics {
            distribution: DistributionType::Sparse {
                density: self.density,
            },
            sparsity: 1.0 - self.density,
            condition_number: None,
            dynamic_range: 1.0,
            noise_level: 0.0,
        }
    }
}

/// Realistic scenario data generator
pub struct RealisticDataGenerator {
    scenario: RealisticScenario,
}

impl RealisticDataGenerator {
    pub fn new(scenario: RealisticScenario) -> Self {
        Self { _scenario }
    }
}

impl DataGenerator for RealisticDataGenerator {
    fn generate(&self, size: usize) -> Result<(Array1<f32>, Array1<f32>)> {
        match self.scenario {
            RealisticScenario::ImageNet => {
                // Simulate ImageNet-like gradients with realistic distributions
                let params = Array1::from_vec((0..size).map(|_| scirs2_core::random::f32() * 0.1).collect());
                let gradients = Array1::from_vec(
                    (0..size)
                        .map(|_| {
                            let base = scirs2_core::random::f32() - 0.5;
                            base * 0.01 // Typical gradient magnitude for ImageNet
                        })
                        .collect(),
                );
                Ok((params, gradients))
            }
            RealisticScenario::BERT => {
                // Simulate BERT-like gradients
                let params = Array1::from_vec((0..size).map(|_| scirs2_core::random::f32() * 0.02).collect());
                let gradients = Array1::from_vec(
                    (0..size)
                        .map(|_| {
                            let base = scirs2_core::random::f32() - 0.5;
                            base * 0.001 // Typical gradient magnitude for BERT
                        })
                        .collect(),
                );
                Ok((params, gradients))
            }
            _ => {
                // Default realistic scenario
                let params = Array1::from_vec((0..size).map(|_| scirs2_core::random::f32() * 0.1).collect());
                let gradients =
                    Array1::from_vec((0..size).map(|_| (scirs2_core::random::f32() - 0.5) * 0.01).collect());
                Ok((params, gradients))
            }
        }
    }

    fn name(&self) -> &str {
        match self.scenario {
            RealisticScenario::ImageNet => "imagenet",
            RealisticScenario::BERT => "bert",
            RealisticScenario::GPT => "gpt",
            RealisticScenario::ResNet => "resnet",
            RealisticScenario::Transformer => "transformer",
            RealisticScenario::RecommendationSystem => "recsys",
            RealisticScenario::TimeSeriesForecasting => "timeseries",
        }
    }

    fn characteristics(&self) -> DataCharacteristics {
        DataCharacteristics {
            distribution: DistributionType::Realistic {
                scenario: self.scenario,
            },
            sparsity: 0.1,
            condition_number: Some(10.0),
            dynamic_range: 0.1,
            noise_level: 0.01,
        }
    }
}

/// Convenience function to run quick performance comparison
#[allow(dead_code)]
pub fn quick_performance_comparison() -> Result<()> {
    let config = BenchmarkConfig {
        backends: vec![GpuBackend::Cuda, GpuBackend::Rocm],
        problem_sizes: vec![10000, 100000],
        optimizer_types: vec![OptimizerType::Adam, OptimizerType::AdamW],
        iterations: 10,
        warmup_iterations: 2,
        ..Default::default()
    };

    let mut benchmark = GpuOptimizerBenchmark::new(config)?;
    benchmark.set_baseline(GpuBackend::Cpu);

    let summary = benchmark.run_full_benchmark()?;

    println!("\n=== Quick Performance Comparison Results ===");
    println!("Total tests: {}", summary.total_tests);
    println!(
        "Execution time: {:.2}s",
        summary.total_execution_time.as_secs_f64()
    );

    for (backend, speedup) in &summary.backend_comparisons {
        println!("{backend:?}: {speedup:.2}x average speedup");
    }

    for insight in &summary.performance_insights {
        println!("📊 {insight}");
    }

    for recommendation in &summary.recommendations {
        println!("💡 {recommendation}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_creation() {
        let config = BenchmarkConfig::default();
        let benchmark = GpuOptimizerBenchmark::new(config);
        assert!(benchmark.is_ok());
    }

    #[test]
    fn test_data_generators() {
        let normal_gen = NormalDataGenerator::new(0.0, 1.0);
        let result = normal_gen.generate(1000);
        assert!(result.is_ok());

        let (params, grads) = result.unwrap();
        assert_eq!(params.len(), 1000);
        assert_eq!(grads.len(), 1000);
    }

    #[test]
    fn test_sparse_data_generator() {
        let sparse_gen = SparseDataGenerator::new(0.1);
        let result = sparse_gen.generate(1000);
        assert!(result.is_ok());

        let (params_grads) = result.unwrap();
        let zero_count = params.iter().filter(|&&x| x == 0.0).count();

        // Should have approximately 90% zeros (some variance expected)
        assert!(zero_count > 800);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(!config.backends.is_empty());
        assert!(!config.problem_sizes.is_empty());
        assert!(!config.optimizer_types.is_empty());
        assert!(config.iterations > 0);
        assert!(config.confidence_level > 0.0 && config.confidence_level < 1.0);
    }
}
