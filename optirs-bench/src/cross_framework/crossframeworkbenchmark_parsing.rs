//! # `CrossFrameworkBenchmark` - parsing Methods
//!
//! This module contains method implementations for `CrossFrameworkBenchmark`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use crate::regression_tester::distributions::{sample_variance, welch_t_test};
use crate::TestFunction;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::process::Command;
use std::time::{Duration, Instant};

use super::constants::MISSING_DEPENDENCY_MARKER;
use super::type_aliases::OptimizerFn;
use super::types::{
    CpuStats, CrossFrameworkBenchmarkResult, CrossFrameworkConfig, ExternalFrameworkOutcome,
    Framework, GpuStats, MemoryStats, OptimizerBenchmarkSummary, OptimizerIdentifier,
    PythonScriptTemplates, ResourceUsageComparison, SkippedFramework, StatisticalComparison,
    TTestResult,
};

use super::crossframeworkbenchmark_type::CrossFrameworkBenchmark;
use std::collections::HashSet;

impl<A: Float + Debug + Send + Sync> CrossFrameworkBenchmark<A> {
    /// Create a new cross-framework benchmark suite
    pub fn new(config: CrossFrameworkConfig) -> Result<Self> {
        let python_scripts = PythonScriptTemplates::new();

        // Create temporary directory
        std::fs::create_dir_all(&config.temp_dir).map_err(|e| {
            OptimError::InvalidConfig(format!("Failed to create temp directory: {}", e))
        })?;

        Ok(Self {
            config,
            test_functions: Vec::new(),
            python_scripts,
            results: Vec::new(),
        })
    }

    /// Add a test function to the benchmark suite
    pub fn add_test_function(&mut self, test_function: TestFunction<A>) {
        self.test_functions.push(test_function);
    }

    /// Add standard optimization test functions
    pub fn add_standard_test_functions(&mut self) {
        // Quadratic function
        self.add_test_function(TestFunction {
            name: "Quadratic".to_string(),
            dimension: 10,
            function: Box::new(|x: &Array1<A>| x.mapv(|val| val * val).sum()),
            gradient: Box::new(|x: &Array1<A>| {
                x.mapv(|val| A::from(2.0).expect("unwrap failed") * val)
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::zeros(10)),
        });

        // Rosenbrock function
        self.add_test_function(TestFunction {
            name: "Rosenbrock".to_string(),
            dimension: 2,
            function: Box::new(|x: &Array1<A>| {
                let a = A::one();
                let b = A::from(100.0).expect("unwrap failed");
                let term1 = (a - x[0]) * (a - x[0]);
                let term2 = b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
                term1 + term2
            }),
            gradient: Box::new(|x: &Array1<A>| {
                let a = A::one();
                let b = A::from(100.0).expect("unwrap failed");
                let grad_x = A::from(-2.0).expect("unwrap failed") * (a - x[0])
                    - A::from(4.0).expect("unwrap failed") * b * x[0] * (x[1] - x[0] * x[0]);
                let grad_y = A::from(2.0).expect("unwrap failed") * b * (x[1] - x[0] * x[0]);
                Array1::from_vec(vec![grad_x, grad_y])
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::from_vec(vec![A::one(), A::one()])),
        });

        // Beale function
        self.add_test_function(TestFunction {
            name: "Beale".to_string(),
            dimension: 2,
            function: Box::new(|x: &Array1<A>| {
                let x1 = x[0];
                let x2 = x[1];
                let term1 = (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2)
                    * (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2);
                let term2 = (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2)
                    * (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2);
                let term3 = (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2)
                    * (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2);
                term1 + term2 + term3
            }),
            gradient: Box::new(|x: &Array1<A>| {
                let x1 = x[0];
                let x2 = x[1];
                let dx1 = A::from(2.0).expect("unwrap failed")
                    * (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2)
                    * (x2 - A::one())
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2)
                        * (x2 * x2 - A::one())
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2)
                        * (x2 * x2 * x2 - A::one());
                let dx2 = A::from(2.0).expect("unwrap failed")
                    * (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2)
                    * x1
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2)
                        * (A::from(2.0).expect("unwrap failed") * x1 * x2)
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2)
                        * (A::from(3.0).expect("unwrap failed") * x1 * x2 * x2);
                Array1::from_vec(vec![dx1, dx2])
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::from_vec(vec![
                A::from(3.0).expect("unwrap failed"),
                A::from(0.5).expect("unwrap failed"),
            ])),
        });
    }

    /// Run comprehensive cross-framework benchmark
    pub fn run_comprehensive_benchmark(
        &mut self,
        scirs2_optimizers: Vec<(String, OptimizerFn<A>)>,
    ) -> Result<Vec<CrossFrameworkBenchmarkResult<A>>> {
        let mut all_results = Vec::new();

        for test_function in &self.test_functions {
            for &problem_dim in &self.config.problem_dimensions {
                for &batch_size in &self.config.batch_sizes {
                    let result = self.run_single_benchmark(
                        test_function,
                        problem_dim,
                        batch_size,
                        &scirs2_optimizers,
                    )?;
                    all_results.push(result);
                }
            }
        }

        self.results.extend(all_results.clone());
        Ok(all_results)
    }

    /// Run benchmark for a single configuration
    fn run_single_benchmark(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        batch_size: usize,
        scirs2_optimizers: &[(String, OptimizerFn<A>)],
    ) -> Result<CrossFrameworkBenchmarkResult<A>> {
        let mut optimizer_results = HashMap::new();
        let mut skipped_frameworks = Vec::new();

        // Run SciRS2 _optimizers
        for (name, optimizer) in scirs2_optimizers {
            let identifier = OptimizerIdentifier {
                framework: Framework::SciRS2,
                name: name.clone(),
                version: Some("0.1.0".to_string()),
            };

            let summary =
                self.benchmark_scirs2_optimizer(test_function, problem_dim, batch_size, optimizer)?;
            optimizer_results.insert(identifier, summary);
        }

        // Run PyTorch _optimizers
        if self.config.enable_pytorch {
            match self.benchmark_pytorch_optimizers(test_function, problem_dim, batch_size)? {
                ExternalFrameworkOutcome::Completed(results) => {
                    optimizer_results.extend(results);
                }
                ExternalFrameworkOutcome::Skipped(skip) => skipped_frameworks.push(skip),
            }
        }

        // Run TensorFlow _optimizers
        if self.config.enable_tensorflow {
            match self.benchmark_tensorflow_optimizers(test_function, problem_dim, batch_size)? {
                ExternalFrameworkOutcome::Completed(results) => {
                    optimizer_results.extend(results);
                }
                ExternalFrameworkOutcome::Skipped(skip) => skipped_frameworks.push(skip),
            }
        }

        // Perform statistical analysis
        let statistical_comparison = self.perform_statistical_analysis(&optimizer_results)?;

        // Rank _optimizers by performance
        let performance_ranking = self.rank_optimizers(&optimizer_results);

        // Analyze resource usage
        let resource_usage = self.analyze_resource_usage(&optimizer_results);

        Ok(CrossFrameworkBenchmarkResult {
            config: self.config.clone(),
            function_name: test_function.name.clone(),
            problem_dim,
            batch_size,
            optimizer_results,
            statistical_comparison,
            performance_ranking,
            resource_usage,
            skipped_frameworks,
            timestamp: std::time::Instant::now(),
        })
    }

    /// Benchmark SciRS2 optimizer
    fn benchmark_scirs2_optimizer(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        _batch_size: usize,
        optimizer: &OptimizerFn<A>,
    ) -> Result<OptimizerBenchmarkSummary<A>> {
        let mut convergence_times = Vec::new();
        let mut final_values = Vec::new();
        let mut iterations_counts = Vec::new();
        let mut gradient_norms = Vec::new();
        let mut convergence_curves = Vec::new();
        let mut successful_runs = 0;

        for run in 0..self.config.num_runs {
            // Set random seed for reproducibility
            let mut rng_seed = self.config.random_seed + run as u64;

            // Initialize parameters
            let mut x = Array1::from_vec(
                (0..problem_dim)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        A::from((rng_seed % 1000) as f64 / 1000.0 - 0.5).unwrap_or_else(A::zero)
                    })
                    .collect(),
            );

            let start_time = Instant::now();
            let mut convergence_curve = Vec::new();
            let mut converged = false;

            for iteration in 0..self.config.max_iterations {
                let f_val = (test_function.function)(&x);
                let grad = (test_function.gradient)(&x);
                let grad_norm = grad.mapv(|g| g * g).sum().sqrt();

                convergence_curve.push(f_val);

                // Check convergence
                if grad_norm.to_f64().unwrap_or(f64::INFINITY) < self.config.tolerance {
                    let elapsed = start_time.elapsed();
                    convergence_times.push(elapsed);
                    final_values.push(f_val);
                    iterations_counts.push(iteration as f64);
                    gradient_norms.push(grad_norm);
                    convergence_curves.push(convergence_curve.clone());
                    successful_runs += 1;
                    converged = true;
                    break;
                }

                // Perform optimization step
                x = optimizer(&x, &grad);
            }

            // If didn't converge, record final state
            if !converged {
                let elapsed = start_time.elapsed();
                let f_val = (test_function.function)(&x);
                let grad = (test_function.gradient)(&x);
                let grad_norm = grad.mapv(|g| g * g).sum().sqrt();

                convergence_times.push(elapsed);
                final_values.push(f_val);
                iterations_counts.push(self.config.max_iterations as f64);
                gradient_norms.push(grad_norm);
                convergence_curves.push(convergence_curve);
            }
        }

        // Calculate statistics
        let success_rate = successful_runs as f64 / self.config.num_runs as f64;

        let mean_convergence_time = if !convergence_times.is_empty() {
            convergence_times.iter().sum::<Duration>() / convergence_times.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let mean_final_value = match A::from(final_values.len()) {
            Some(count) if !final_values.is_empty() => {
                final_values.iter().fold(A::zero(), |acc, &x| acc + x) / count
            }
            _ => A::zero(),
        };

        let mean_iterations = if !iterations_counts.is_empty() {
            iterations_counts.iter().sum::<f64>() / iterations_counts.len() as f64
        } else {
            0.0
        };

        let mean_gradient_norm = match A::from(gradient_norms.len()) {
            Some(count) if !gradient_norms.is_empty() => {
                gradient_norms.iter().fold(A::zero(), |acc, &x| acc + x) / count
            }
            _ => A::zero(),
        };

        // Calculate standard deviations
        let std_convergence_time =
            self.calculate_duration_std(&convergence_times, mean_convergence_time);
        let std_final_value = self.calculate_std(&final_values, mean_final_value);
        let std_iterations = self.calculate_f64std(&iterations_counts, mean_iterations);
        let std_gradient_norm = self.calculate_std(&gradient_norms, mean_gradient_norm);

        Ok(OptimizerBenchmarkSummary {
            optimizer: OptimizerIdentifier {
                framework: Framework::SciRS2,
                name: "SciRS2".to_string(),
                version: Some("0.1.0".to_string()),
            },
            successful_runs,
            total_runs: self.config.num_runs,
            success_rate,
            mean_convergence_time,
            std_convergence_time,
            mean_final_value,
            std_final_value,
            mean_iterations,
            std_iterations,
            mean_gradient_norm,
            std_gradient_norm,
            convergence_curves,
            run_convergence_times_secs: convergence_times.iter().map(|d| d.as_secs_f64()).collect(),
            run_final_values: final_values
                .iter()
                .filter_map(|value| value.to_f64())
                .collect(),
            memory_stats: MemoryStats {
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                allocation_count: 0,
                fragmentation_ratio: 0.0,
            },
            gpu_utilization: None,
        })
    }

    /// Benchmark PyTorch optimizers.
    ///
    /// Requires a Python interpreter with `torch` installed. When either is
    /// missing the run is reported as
    /// [`ExternalFrameworkOutcome::Skipped`] - never as fabricated results.
    pub(super) fn benchmark_pytorch_optimizers(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        batch_size: usize,
    ) -> Result<ExternalFrameworkOutcome<A>> {
        let script_path = format!("{}/pytorch_benchmark.py", self.config.temp_dir);
        let script_content = self.python_scripts.generate_pytorch_script(
            &test_function.name,
            problem_dim,
            batch_size,
            &self.config,
        );

        self.run_python_benchmark(Framework::PyTorch, &script_path, &script_content)
    }

    /// Benchmark TensorFlow optimizers.
    ///
    /// Requires a Python interpreter with `tensorflow` installed. When either
    /// is missing the run is reported as
    /// [`ExternalFrameworkOutcome::Skipped`] - never as fabricated results.
    fn benchmark_tensorflow_optimizers(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        batch_size: usize,
    ) -> Result<ExternalFrameworkOutcome<A>> {
        let script_path = format!("{}/tensorflow_benchmark.py", self.config.temp_dir);
        let script_content = self.python_scripts.generate_tensorflow_script(
            &test_function.name,
            problem_dim,
            batch_size,
            &self.config,
        );

        self.run_python_benchmark(Framework::TensorFlow, &script_path, &script_content)
    }

    /// Write a generated script, execute it, and parse the JSON it prints.
    fn run_python_benchmark(
        &self,
        framework: Framework,
        script_path: &str,
        script_content: &str,
    ) -> Result<ExternalFrameworkOutcome<A>> {
        std::fs::write(script_path, script_content).map_err(|e| {
            OptimError::InvalidConfig(format!(
                "Failed to write {} benchmark script to {}: {}",
                framework, script_path, e
            ))
        })?;

        let output = match Command::new(&self.config.python_path)
            .arg(script_path)
            .output()
        {
            Ok(output) => output,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ExternalFrameworkOutcome::Skipped(SkippedFramework {
                    framework: framework.clone(),
                    reason: format!(
                        "Python interpreter '{}' was not found; install Python to enable this comparison",
                        self.config.python_path
                    ),
                }));
            }
            Err(e) => {
                return Ok(ExternalFrameworkOutcome::Skipped(SkippedFramework {
                    framework: framework.clone(),
                    reason: format!("Failed to launch '{}': {}", self.config.python_path, e),
                }));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        if !output.status.success() {
            // The generated scripts exit with code 2 and a machine-readable
            // marker when the framework package itself is unavailable.
            if stderr.contains(MISSING_DEPENDENCY_MARKER)
                || stdout.contains(MISSING_DEPENDENCY_MARKER)
            {
                return Ok(ExternalFrameworkOutcome::Skipped(SkippedFramework {
                    reason: format!(
                        "{} is not installed for interpreter '{}'",
                        framework, self.config.python_path
                    ),
                    framework,
                }));
            }
            return Err(OptimError::InvalidConfig(format!(
                "{} benchmark failed (exit status {:?}): {}",
                framework,
                output.status.code(),
                stderr.trim()
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
            OptimError::InvalidConfig(format!(
                "Failed to parse {} results as JSON: {} (stdout was: {})",
                framework,
                e,
                stdout.trim()
            ))
        })?;

        let framework_version = parsed
            .get("framework_version")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());

        let results = parsed
            .get("optimizers")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "{} results are missing the required 'optimizers' object",
                    framework
                ))
            })?;

        let mut optimizer_results = HashMap::new();
        for (optimizer_name, result_data) in results {
            let identifier = OptimizerIdentifier {
                framework: framework.clone(),
                name: optimizer_name.clone(),
                version: framework_version.clone(),
            };
            let summary = self.parse_python_results(identifier.clone(), result_data)?;
            optimizer_results.insert(identifier, summary);
        }

        Ok(ExternalFrameworkOutcome::Completed(optimizer_results))
    }

    /// Parse one optimizer's results out of the JSON produced by a generated
    /// Python benchmark script.
    ///
    /// Parsing is strict: every field the summary needs must be present and of
    /// the right type. A missing field is an error, because silently
    /// substituting "100 ms" or "0.1" for an absent measurement produces a
    /// comparison table that looks authoritative and is entirely fictional.
    pub(super) fn parse_python_results(
        &self,
        identifier: OptimizerIdentifier,
        result_data: &serde_json::Value,
    ) -> Result<OptimizerBenchmarkSummary<A>> {
        let context = identifier.to_string();

        let required_f64 = |field: &str| -> Result<f64> {
            result_data
                .get(field)
                .and_then(|value| value.as_f64())
                .ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "{}: required numeric field '{}' is missing from the Python results",
                        context, field
                    ))
                })
        };
        let required_array = |field: &str| -> Result<Vec<f64>> {
            let array = result_data
                .get(field)
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "{}: required array field '{}' is missing from the Python results",
                        context, field
                    ))
                })?;
            array
                .iter()
                .map(|value| {
                    value.as_f64().ok_or_else(|| {
                        OptimError::InvalidConfig(format!(
                            "{}: field '{}' contains a non-numeric entry",
                            context, field
                        ))
                    })
                })
                .collect()
        };

        let total_runs = required_f64("total_runs")? as usize;
        let successful_runs = required_f64("successful_runs")? as usize;
        if total_runs == 0 {
            return Err(OptimError::InvalidConfig(format!(
                "{}: 'total_runs' must be greater than zero",
                context
            )));
        }
        let success_rate = successful_runs as f64 / total_runs as f64;

        // Wall-clock convergence time is recorded per run by the script.
        let run_convergence_times_secs = required_array("convergence_times_secs")?;
        let run_final_values = required_array("final_values")?;
        let run_iterations = required_array("iterations")?;
        let run_gradient_norms = required_array("gradient_norms")?;

        if run_convergence_times_secs.len() != total_runs
            || run_final_values.len() != total_runs
            || run_iterations.len() != total_runs
            || run_gradient_norms.len() != total_runs
        {
            return Err(OptimError::InvalidConfig(format!(
                "{}: per-run arrays must all have exactly 'total_runs' = {} entries",
                context, total_runs
            )));
        }

        let convergence_curves: Vec<Vec<A>> = result_data
            .get("convergence_curves")
            .and_then(|value| value.as_array())
            .map(|curves| {
                curves
                    .iter()
                    .filter_map(|curve| curve.as_array())
                    .map(|curve| {
                        curve
                            .iter()
                            .filter_map(|value| value.as_f64())
                            .filter_map(A::from)
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mean_of = |values: &[f64]| -> f64 {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            }
        };
        let std_of =
            |values: &[f64]| -> f64 { sample_variance(values).map(f64::sqrt).unwrap_or(0.0) };

        let mean_convergence_secs = mean_of(&run_convergence_times_secs);
        let std_convergence_secs = std_of(&run_convergence_times_secs);
        let mean_final = mean_of(&run_final_values);
        let std_final = std_of(&run_final_values);
        let mean_gradient = mean_of(&run_gradient_norms);
        let std_gradient = std_of(&run_gradient_norms);

        let memory_stats = MemoryStats {
            // Memory is optional: `0` means the script did not measure it.
            peak_memory_bytes: result_data
                .get("peak_memory_bytes")
                .and_then(|value| value.as_f64())
                .map(|value| value.max(0.0) as usize)
                .unwrap_or(0),
            avg_memory_bytes: result_data
                .get("avg_memory_bytes")
                .and_then(|value| value.as_f64())
                .map(|value| value.max(0.0) as usize)
                .unwrap_or(0),
            allocation_count: result_data
                .get("allocation_count")
                .and_then(|value| value.as_f64())
                .map(|value| value.max(0.0) as usize)
                .unwrap_or(0),
            fragmentation_ratio: result_data
                .get("fragmentation_ratio")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
        };

        Ok(OptimizerBenchmarkSummary {
            optimizer: identifier,
            successful_runs,
            total_runs,
            success_rate,
            mean_convergence_time: Duration::from_secs_f64(mean_convergence_secs.max(0.0)),
            std_convergence_time: Duration::from_secs_f64(std_convergence_secs.max(0.0)),
            mean_final_value: A::from(mean_final).unwrap_or_else(A::zero),
            std_final_value: A::from(std_final).unwrap_or_else(A::zero),
            mean_iterations: mean_of(&run_iterations),
            std_iterations: std_of(&run_iterations),
            mean_gradient_norm: A::from(mean_gradient).unwrap_or_else(A::zero),
            std_gradient_norm: A::from(std_gradient).unwrap_or_else(A::zero),
            convergence_curves,
            run_convergence_times_secs,
            run_final_values,
            memory_stats,
            gpu_utilization: result_data
                .get("gpu_utilization")
                .and_then(|value| value.as_f64()),
        })
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        results: &HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>,
    ) -> Result<StatisticalComparison<A>> {
        let mut convergence_time_tests = HashMap::new();
        let mut final_value_tests = HashMap::new();
        let mut effect_sizes = HashMap::new();
        let mut confidence_intervals = HashMap::new();

        // Pairwise comparisons
        let optimizers: Vec<_> = results.keys().collect();
        for i in 0..optimizers.len() {
            for j in (i + 1)..optimizers.len() {
                let opt1 = optimizers[i];
                let opt2 = optimizers[j];

                let result1 = &results[opt1];
                let result2 = &results[opt2];

                // Welch t-test on the recorded per-run wall-clock times.
                let time_test = self.perform_t_test(
                    &result1.run_convergence_times_secs,
                    &result2.run_convergence_times_secs,
                );
                convergence_time_tests.insert((opt1.clone(), opt2.clone()), time_test);

                // Welch t-test on the recorded per-run final objective values.
                let value_test =
                    self.perform_t_test(&result1.run_final_values, &result2.run_final_values);
                final_value_tests.insert((opt1.clone(), opt2.clone()), value_test);

                // Effect size (Cohen's d)
                let effect_size =
                    self.calculate_cohens_d(&result1.run_final_values, &result2.run_final_values);
                effect_sizes.insert((opt1.clone(), opt2.clone()), effect_size);
            }

            // Confidence interval for the mean final objective value.
            let result = &results[optimizers[i]];
            let ci = self.calculate_confidence_interval(
                &result.run_final_values,
                self.config.confidence_level(),
            );
            confidence_intervals.insert(optimizers[i].clone(), ci);
        }

        // ANOVA across the per-run final objective values of every optimizer.
        let all_final_values: Vec<Vec<f64>> = results
            .values()
            .map(|result| result.run_final_values.clone())
            .collect();
        let anova_results = self.perform_anova(&all_final_values);

        Ok(StatisticalComparison {
            convergence_time_tests,
            final_value_tests,
            anova_results,
            effect_sizes,
            confidence_intervals,
        })
    }

    /// Rank optimizers by performance
    fn rank_optimizers(
        &self,
        results: &HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>,
    ) -> Vec<(OptimizerIdentifier, f64)> {
        let mut rankings: Vec<_> = results
            .iter()
            .map(|(identifier, summary)| {
                // Composite score: success_rate * (1 / mean_final_value) * (1 / mean_convergence_time)
                let time_factor = 1.0 / (summary.mean_convergence_time.as_millis() as f64 + 1.0);
                let value_factor = 1.0 / (summary.mean_final_value.to_f64().unwrap_or(1.0) + 1e-10);
                let score = summary.success_rate * time_factor * value_factor;
                (identifier.clone(), score)
            })
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }

    /// Analyze resource usage
    fn analyze_resource_usage(
        &self,
        results: &HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>,
    ) -> ResourceUsageComparison {
        let memory_usage = results
            .iter()
            .map(|(id, summary)| (id.clone(), summary.memory_stats.clone()))
            .collect();

        // CPU counters are not instrumented by this harness; zeros here mean
        // "not measured" and are documented as such on `CpuStats`.
        let cpu_usage = results
            .iter()
            .map(|(id, _summary)| (id.clone(), CpuStats::unmeasured()))
            .collect();

        let gpu_usage = results
            .iter()
            .map(|(id, summary)| {
                let gpu_stats = summary.gpu_utilization.map(|gpu_percent| GpuStats {
                    gpu_percent,
                    memory_usage_bytes: 0,
                    kernel_launches: 0,
                    avg_kernel_time_us: 0.0,
                });
                (id.clone(), gpu_stats)
            })
            .collect();

        ResourceUsageComparison {
            memory_usage,
            cpu_usage,
            gpu_usage,
        }
    }

    /// Generate comprehensive benchmark report
    pub fn generate_comprehensive_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Cross-Framework Optimizer Benchmark Report\n\n");
        report.push_str(&format!(
            "Generated: {:?}\n\n",
            std::time::SystemTime::now()
        ));

        if self.results.is_empty() {
            report.push_str("No benchmark results available.\n");
            return report;
        }

        // Executive summary
        report.push_str("## Executive Summary\n\n");
        report.push_str(&format!(
            "Total test configurations: {}\n",
            self.results.len()
        ));

        // Framework coverage
        let frameworks: std::collections::HashSet<_> = self
            .results
            .iter()
            .flat_map(|result| result.optimizer_results.keys())
            .map(|id| &id.framework)
            .collect();
        report.push_str(&format!("Frameworks tested: {:?}\n\n", frameworks));

        // Performance rankings
        report.push_str("## Overall Performance Rankings\n\n");
        for result in &self.results {
            report.push_str(&format!(
                "### {} ({}D, batch={})\n\n",
                result.function_name, result.problem_dim, result.batch_size
            ));

            for (rank, (optimizer, score)) in result.performance_ranking.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} - Score: {:.6}\n",
                    rank + 1,
                    optimizer,
                    score
                ));
            }
            report.push('\n');
        }

        // Statistical significance
        report.push_str("## Statistical Analysis\n\n");
        for result in &self.results {
            report.push_str(&format!("### {} Results\n\n", result.function_name));

            // ANOVA results
            let anova = &result.statistical_comparison.anova_results;
            report.push_str(&format!(
                "ANOVA F-statistic: {:.4}, p-value: {:.6}\n",
                anova.f_statistic, anova.p_value
            ));

            if anova.p_value < 0.05 {
                report.push_str(
                    "**Statistically significant differences found between optimizers.**\n\n",
                );
            } else {
                report.push_str("No statistically significant differences found.\n\n");
            }
        }

        report
    }

    /// Calculate standard deviation for Duration values
    fn calculate_duration_std(&self, values: &[Duration], mean: Duration) -> Duration {
        if values.len() <= 1 {
            return Duration::from_millis(0);
        }

        let variance = values
            .iter()
            .map(|&v| {
                let diff = v.as_millis() as i64 - mean.as_millis() as i64;
                (diff * diff) as f64
            })
            .sum::<f64>()
            / (values.len() - 1) as f64;

        Duration::from_millis(variance.sqrt() as u64)
    }

    /// Calculate standard deviation for Float values
    fn calculate_std(&self, values: &[A], mean: A) -> A {
        if values.len() <= 1 {
            return A::zero();
        }

        let variance = values
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .fold(A::zero(), |acc, x| acc + x)
            / A::from(values.len() - 1).expect("unwrap failed");

        variance.sqrt()
    }

    /// Calculate standard deviation for f64 values
    fn calculate_f64std(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
            / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Welch's unequal-variance t-test between two samples.
    ///
    /// Both the statistic and the p-value are exact: the p-value comes from the
    /// regularized incomplete beta representation of the Student's t
    /// distribution, so it spans the full `(0, 1]` range instead of saturating
    /// around `0.25` as the previous closed-form approximation did.
    ///
    /// Degenerate inputs (fewer than two observations in a group, or a zero
    /// pooled standard error because both groups are constant) yield
    /// `p = 1` and `is_significant = false` rather than a division by zero.
    pub(super) fn perform_t_test(&self, sample1: &[f64], sample2: &[f64]) -> TTestResult {
        match welch_t_test(sample1, sample2) {
            Some(test) => TTestResult {
                t_statistic: test.t_statistic,
                p_value: test.p_value,
                degrees_of_freedom: test.degrees_of_freedom,
                is_significant: test.p_value < 1.0 - self.config.confidence_level(),
            },
            None => TTestResult {
                t_statistic: 0.0,
                p_value: 1.0,
                degrees_of_freedom: 0.0,
                is_significant: false,
            },
        }
    }

    /// Student's t cumulative distribution function.
    ///
    /// Delegates to the shared, table-checked implementation.
    pub(super) fn t_distribution_cdf(&self, t: f64, df: f64) -> f64 {
        crate::regression_tester::distributions::student_t_cdf(t, df)
    }
}
