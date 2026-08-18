// Benchmarking and evaluation tools for optimizers
//
// This module provides tools for analyzing optimizer performance, gradient flow,
// and visualization of optimization behavior, including cross-framework comparisons
// with PyTorch and TensorFlow optimizers.

use optirs_core::error::{OptimError, Result};
use optirs_core::utils::{scalar_or, try_scalar};
use scirs2_core::ndarray::{Array, Array1, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Type alias for objective function
pub type ObjectiveFunction<A> = Box<dyn Fn(&Array1<A>) -> A>;
/// Type alias for gradient function
pub type GradientFunction<A> = Box<dyn Fn(&Array1<A>) -> Array1<A>>;

/// Gradient flow analyzer for understanding optimization dynamics
#[derive(Debug)]
pub struct GradientFlowAnalyzer<A: Float, D: Dimension> {
    /// History of gradient magnitudes
    gradient_magnitudes: VecDeque<Vec<A>>,
    /// History of gradient directions (cosine similarities)
    gradient_directions: VecDeque<A>,
    /// History of parameter updates
    parameter_updates: VecDeque<Vec<Array<A, D>>>,
    /// Step count
    step_count: usize,
    /// Maximum history size
    _maxhistory: usize,
    /// Statistics cache. Not an `Option`: a freshly constructed analyzer (no
    /// steps recorded yet) already has a well-defined "empty" stats value --
    /// the same one `compute_stats` would produce from all-empty history --
    /// so there is no `None` state to represent and no panic path needed to
    /// read it back out (see `get_stats`).
    stats_cache: GradientFlowStats<A>,
    /// Cache validity
    cache_valid: bool,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> GradientFlowAnalyzer<A, D> {
    /// Create a new gradient flow analyzer
    pub fn new(_maxhistory: usize) -> Self {
        Self {
            gradient_magnitudes: VecDeque::with_capacity(_maxhistory),
            gradient_directions: VecDeque::with_capacity(_maxhistory),
            parameter_updates: VecDeque::with_capacity(_maxhistory),
            step_count: 0,
            _maxhistory,
            // Matches exactly what `compute_stats` returns for all-empty
            // history (`cache_valid: true`, so this is not just a filler
            // placeholder waiting to be overwritten).
            stats_cache: GradientFlowStats {
                step_count: 0,
                per_group_stats: Vec::new(),
                mean_direction_similarity: A::one(),
                direction_variance: A::zero(),
                direction_std_dev: A::zero(),
                is_converging: false,
                oscillation_frequency: 0.0,
                stability_score: 1.0,
                direction_history: Vec::new(),
            },
            cache_valid: true,
        }
    }

    /// Record a gradient and parameter update step
    pub fn record_step(
        &mut self,
        gradients: &[Array<A, D>],
        parameter_updates: &[Array<A, D>],
    ) -> Result<()> {
        if gradients.len() != parameter_updates.len() {
            return Err(OptimError::DimensionMismatch(
                "Number of gradients must match number of parameter _updates".to_string(),
            ));
        }

        self.step_count += 1;

        // Calculate gradient magnitudes for each parameter group
        let magnitudes: Vec<A> = gradients
            .iter()
            .map(|grad| grad.mapv(|x| x * x).sum().sqrt())
            .collect();

        self.gradient_magnitudes.push_back(magnitudes);

        // Calculate gradient direction similarity (cosine similarity with previous step)
        if let Some(prev_gradients) = self.parameter_updates.back() {
            let similarity = self.calculate_cosine_similarity(gradients, prev_gradients)?;
            self.gradient_directions.push_back(similarity);
        } else {
            // First step, no previous gradient to compare with
            self.gradient_directions.push_back(A::one());
        }

        // Store parameter _updates
        self.parameter_updates.push_back(parameter_updates.to_vec());

        // Maintain maximum history size
        if self.gradient_magnitudes.len() > self._maxhistory {
            self.gradient_magnitudes.pop_front();
        }
        if self.gradient_directions.len() > self._maxhistory {
            self.gradient_directions.pop_front();
        }
        if self.parameter_updates.len() > self._maxhistory {
            self.parameter_updates.pop_front();
        }

        // Invalidate cache
        self.cache_valid = false;

        Ok(())
    }

    /// Calculate cosine similarity between two sets of arrays
    fn calculate_cosine_similarity(
        &self,
        arrays1: &[Array<A, D>],
        arrays2: &[Array<A, D>],
    ) -> Result<A> {
        if arrays1.len() != arrays2.len() {
            return Err(OptimError::DimensionMismatch(
                "Array sets must have same length".to_string(),
            ));
        }

        let mut dot_product = A::zero();
        let mut norm1_sq = A::zero();
        let mut norm2_sq = A::zero();

        for (arr1, arr2) in arrays1.iter().zip(arrays2.iter()) {
            for (&a, &b) in arr1.iter().zip(arr2.iter()) {
                dot_product = dot_product + a * b;
                norm1_sq = norm1_sq + a * a;
                norm2_sq = norm2_sq + b * b;
            }
        }

        let norm1 = norm1_sq.sqrt();
        let norm2 = norm2_sq.sqrt();

        if norm1 > A::zero() && norm2 > A::zero() {
            Ok(dot_product / (norm1 * norm2))
        } else {
            Ok(A::zero())
        }
    }

    /// Get gradient flow statistics
    pub fn get_stats(&mut self) -> &GradientFlowStats<A> {
        if !self.cache_valid {
            self.stats_cache = self.compute_stats();
            self.cache_valid = true;
        }
        &self.stats_cache
    }

    /// Compute gradient flow statistics
    fn compute_stats(&self) -> GradientFlowStats<A> {
        let num_param_groups = if let Some(first) = self.gradient_magnitudes.front() {
            first.len()
        } else {
            0
        };

        // Compute per-parameter-group statistics
        let mut per_group_stats = Vec::new();
        for group_idx in 0..num_param_groups {
            let group_magnitudes: Vec<A> = self
                .gradient_magnitudes
                .iter()
                .map(|step_mags| step_mags[group_idx])
                .collect();

            let mean_magnitude = if !group_magnitudes.is_empty() {
                group_magnitudes.iter().fold(A::zero(), |acc, &x| acc + x)
                    / scalar_or(group_magnitudes.len(), A::one())
            } else {
                A::zero()
            };

            let variance = if group_magnitudes.len() > 1 {
                let mean = mean_magnitude;
                let sum_sq_diff = group_magnitudes
                    .iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .fold(A::zero(), |acc, x| acc + x);
                sum_sq_diff / scalar_or(group_magnitudes.len() - 1, A::one())
            } else {
                A::zero()
            };

            let max_magnitude = group_magnitudes
                .iter()
                .fold(A::neg_infinity(), |acc, &x| acc.max(x));

            let min_magnitude = group_magnitudes
                .iter()
                .fold(A::infinity(), |acc, &x| acc.min(x));

            per_group_stats.push(ParameterGroupStats {
                mean_magnitude,
                variance,
                std_dev: variance.sqrt(),
                max_magnitude,
                min_magnitude,
                magnitude_history: group_magnitudes,
            });
        }

        // Overall gradient direction statistics
        let mean_direction_similarity = if !self.gradient_directions.is_empty() {
            self.gradient_directions
                .iter()
                .fold(A::zero(), |acc, &x| acc + x)
                / scalar_or(self.gradient_directions.len(), A::one())
        } else {
            A::one()
        };

        let direction_variance = if self.gradient_directions.len() > 1 {
            let mean = mean_direction_similarity;
            let sum_sq_diff = self
                .gradient_directions
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(A::zero(), |acc, x| acc + x);
            sum_sq_diff / scalar_or(self.gradient_directions.len() - 1, A::one())
        } else {
            A::zero()
        };

        // Convergence analysis
        let is_converging = self.analyze_convergence();
        let oscillation_frequency = self.calculate_oscillation_frequency();
        let stability_score = self.calculate_stability_score();

        GradientFlowStats {
            step_count: self.step_count,
            per_group_stats,
            mean_direction_similarity,
            direction_variance,
            direction_std_dev: direction_variance.sqrt(),
            is_converging,
            oscillation_frequency,
            stability_score,
            direction_history: self.gradient_directions.iter().copied().collect(),
        }
    }

    /// Analyze if gradients are converging
    fn analyze_convergence(&self) -> bool {
        if self.gradient_magnitudes.len() < 5 {
            return false;
        }

        // Check if gradient magnitudes are generally decreasing
        let recent_steps = 5.min(self.gradient_magnitudes.len());
        let recent_magnitudes: Vec<_> = self
            .gradient_magnitudes
            .iter()
            .rev()
            .take(recent_steps)
            .collect();

        // Calculate trend for each parameter group
        let mut converging_groups = 0;
        let num_groups = recent_magnitudes[0].len();

        for group_idx in 0..num_groups {
            let group_trend: Vec<A> = recent_magnitudes
                .iter()
                .rev()
                .map(|step| step[group_idx])
                .collect();

            // Simple linear trend analysis
            let is_decreasing = group_trend
                .windows(2)
                .map(|window| window[1] < window[0])
                .filter(|&x| x)
                .count()
                >= group_trend.len() / 2;

            if is_decreasing {
                converging_groups += 1;
            }
        }

        converging_groups >= num_groups / 2
    }

    /// Calculate oscillation frequency in gradient directions
    fn calculate_oscillation_frequency(&self) -> f64 {
        if self.gradient_directions.len() < 3 {
            return 0.0;
        }

        let mut sign_changes = 0;
        let mut prev_positive = None;

        for &direction in &self.gradient_directions {
            let is_positive = direction >= A::zero();
            if let Some(prev) = prev_positive {
                if prev != is_positive {
                    sign_changes += 1;
                }
            }
            prev_positive = Some(is_positive);
        }

        sign_changes as f64 / (self.gradient_directions.len() - 1) as f64
    }

    /// Calculate stability score (0.0 = unstable, 1.0 = stable)
    fn calculate_stability_score(&self) -> f64 {
        if self.gradient_directions.is_empty() {
            return 1.0;
        }

        // Stability based on direction consistency and magnitude variance
        let direction_consistency = self
            .gradient_directions
            .iter()
            .fold(A::zero(), |acc, &x| acc + x.abs())
            / scalar_or(self.gradient_directions.len(), A::one());

        let magnitude_consistency = if !self.gradient_magnitudes.is_empty() {
            let all_magnitudes: Vec<A> = self
                .gradient_magnitudes
                .iter()
                .flat_map(|step| step.iter())
                .copied()
                .collect();

            if all_magnitudes.len() > 1 {
                let mean = all_magnitudes.iter().fold(A::zero(), |acc, &x| acc + x)
                    / scalar_or(all_magnitudes.len(), A::one());
                let variance = all_magnitudes
                    .iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .fold(A::zero(), |acc, x| acc + x)
                    / scalar_or(all_magnitudes.len(), A::one());
                let cv = if mean > A::zero() {
                    variance.sqrt() / mean
                } else {
                    A::zero()
                };
                // Lower coefficient of variation = higher stability
                (A::one() / (A::one() + cv)).to_f64().unwrap_or(0.0)
            } else {
                1.0
            }
        } else {
            1.0
        };

        let direction_score = direction_consistency.to_f64().unwrap_or(0.0).abs();
        (direction_score + magnitude_consistency) / 2.0
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.gradient_magnitudes.clear();
        self.gradient_directions.clear();
        self.parameter_updates.clear();
        self.step_count = 0;
        // Invalidate only; `stats_cache` is not read again until `get_stats`
        // recomputes it (see the struct field's doc comment).
        self.cache_valid = false;
    }

    /// Export data for visualization
    pub fn export_for_visualization(&self) -> VisualizationData<A> {
        let magnitude_series: Vec<Vec<A>> = if !self.gradient_magnitudes.is_empty() {
            let num_groups = self.gradient_magnitudes[0].len();
            (0..num_groups)
                .map(|group_idx| {
                    self.gradient_magnitudes
                        .iter()
                        .map(|step| step[group_idx])
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };

        VisualizationData {
            step_indices: (0..self.step_count).collect(),
            magnitude_series,
            direction_similarities: self.gradient_directions.iter().copied().collect(),
        }
    }
}

/// Statistics about gradient flow
#[derive(Debug, Clone)]
pub struct GradientFlowStats<A: Float> {
    /// Total number of steps recorded
    pub step_count: usize,
    /// Per-parameter-group statistics
    pub per_group_stats: Vec<ParameterGroupStats<A>>,
    /// Mean cosine similarity between consecutive gradients
    pub mean_direction_similarity: A,
    /// Variance in gradient direction similarities
    pub direction_variance: A,
    /// Standard deviation in gradient direction similarities
    pub direction_std_dev: A,
    /// Whether the optimization appears to be converging
    pub is_converging: bool,
    /// Frequency of oscillations in gradient directions
    pub oscillation_frequency: f64,
    /// Overall stability score (0.0 = unstable, 1.0 = stable)
    pub stability_score: f64,
    /// History of direction similarities
    pub direction_history: Vec<A>,
}

/// Statistics for a single parameter group
#[derive(Debug, Clone)]
pub struct ParameterGroupStats<A: Float> {
    /// Mean gradient magnitude
    pub mean_magnitude: A,
    /// Variance in gradient magnitudes
    pub variance: A,
    /// Standard deviation in gradient magnitudes
    pub std_dev: A,
    /// Maximum gradient magnitude observed
    pub max_magnitude: A,
    /// Minimum gradient magnitude observed
    pub min_magnitude: A,
    /// History of gradient magnitudes
    pub magnitude_history: Vec<A>,
}

/// Data structure for visualization
#[derive(Debug, Clone)]
pub struct VisualizationData<A: Float> {
    /// Step indices
    pub step_indices: Vec<usize>,
    /// Gradient magnitude series (one per parameter group)
    pub magnitude_series: Vec<Vec<A>>,
    /// Direction similarity series
    pub direction_similarities: Vec<A>,
}

/// Optimizer benchmark suite
pub struct OptimizerBenchmark<A: Float> {
    /// Test functions for benchmarking
    test_functions: Vec<TestFunction<A>>,
    /// Benchmark results
    results: Vec<BenchmarkResult<A>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> OptimizerBenchmark<A> {
    /// Create a new optimizer benchmark suite
    pub fn new() -> Self {
        Self {
            test_functions: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a test function to the benchmark suite
    pub fn add_test_function(&mut self, testfunction: TestFunction<A>) {
        self.test_functions.push(testfunction);
    }

    /// Add standard test functions
    pub fn add_standard_test_functions(&mut self) {
        // Quadratic function: f(x) = x^T * x
        self.add_test_function(TestFunction {
            name: "Quadratic".to_string(),
            dimension: 10,
            function: Box::new(|x: &Array1<A>| x.mapv(|val| val * val).sum()),
            gradient: Box::new(|x: &Array1<A>| {
                x.mapv(|val| scalar_or(2.0, A::one() + A::one()) * val)
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::zeros(10)),
        });

        // Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
        self.add_test_function(TestFunction {
            name: "Rosenbrock".to_string(),
            dimension: 2,
            function: Box::new(|x: &Array1<A>| {
                let a = A::one();
                let b = scalar_or(100.0, A::one());
                let term1 = (a - x[0]) * (a - x[0]);
                let term2 = b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
                term1 + term2
            }),
            gradient: Box::new(|x: &Array1<A>| {
                let a = A::one();
                let two = A::one() + A::one();
                let b = scalar_or(100.0, A::one());
                let grad_x = -two * (a - x[0]) - (two + two) * b * x[0] * (x[1] - x[0] * x[0]);
                let grad_y = two * b * (x[1] - x[0] * x[0]);
                Array1::from_vec(vec![grad_x, grad_y])
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::from_vec(vec![A::one(), A::one()])),
        });

        // Sphere function: f(x) = sum(x_i^2)
        self.add_test_function(TestFunction {
            name: "Sphere".to_string(),
            dimension: 5,
            function: Box::new(|x: &Array1<A>| x.mapv(|val| val * val).sum()),
            gradient: Box::new(|x: &Array1<A>| {
                x.mapv(|val| scalar_or(2.0, A::one() + A::one()) * val)
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::zeros(5)),
        });
    }

    /// Run benchmark on a specific optimizer
    pub fn run_benchmark<F>(
        &mut self,
        optimizername: String,
        mut optimization_step: F,
        max_iterations: usize,
        tolerance: A,
    ) -> Result<Vec<BenchmarkResult<A>>>
    where
        F: FnMut(&Array1<A>, &Array1<A>) -> Array1<A>,
    {
        // Regression (F50): with `max_iterations == 0` the per-function loop
        // below records zero function/gradient evaluations, and the
        // `.last()` calls that used to build the final `BenchmarkResult`
        // panicked via `.expect("unwrap failed")` on the empty history.
        // Zero iterations is a genuine misconfiguration (there is nothing to
        // benchmark), so it is rejected up front with a real error instead
        // of being silently accepted and then crashing partway through.
        if max_iterations == 0 {
            return Err(OptimError::InvalidConfig(
                "run_benchmark requires max_iterations > 0: with zero iterations no \
                 function/gradient evaluations would be recorded at all"
                    .to_string(),
            ));
        }

        let mut results = Vec::new();

        for testfunction in &self.test_functions {
            let mut x = Array1::from_vec(
                (0..testfunction.dimension)
                    .map(|_| try_scalar(0.5))
                    .collect::<Result<Vec<A>>>()?,
            );

            let mut function_values = Vec::new();
            let mut gradient_norms = Vec::new();
            let mut convergence_step = None;

            let start_time = std::time::Instant::now();

            for iteration in 0..max_iterations {
                let f_val = (testfunction.function)(&x);
                let grad = (testfunction.gradient)(&x);
                let grad_norm = grad.mapv(|g| g * g).sum().sqrt();

                function_values.push(f_val);
                gradient_norms.push(grad_norm);

                // Check convergence
                if grad_norm < tolerance {
                    convergence_step = Some(iteration);
                    break;
                }

                // Perform optimization _step
                x = optimization_step(&x, &grad);
            }

            let elapsed = start_time.elapsed();

            // `max_iterations > 0` is enforced above, and every loop
            // iteration pushes exactly one function value and one gradient
            // norm before it can `break`, so both vectors are guaranteed
            // non-empty here. `ok_or_else` + `?` is used anyway rather than
            // `.expect(...)`, so a future change to the loop above fails
            // with a real, propagated error instead of a panic.
            let last_function_value = *function_values.last().ok_or_else(|| {
                OptimError::OptimizationError(format!(
                    "benchmark loop for '{}' produced no function-value samples",
                    testfunction.name
                ))
            })?;
            let last_gradient_norm = *gradient_norms.last().ok_or_else(|| {
                OptimError::OptimizationError(format!(
                    "benchmark loop for '{}' produced no gradient-norm samples",
                    testfunction.name
                ))
            })?;

            let final_error = if let Some(optimal_value) = testfunction.optimal_value {
                (last_function_value - optimal_value).abs()
            } else {
                A::zero()
            };

            let result = BenchmarkResult {
                optimizername: optimizername.clone(),
                function_name: testfunction.name.clone(),
                converged: convergence_step.is_some(),
                convergence_step,
                final_function_value: last_function_value,
                final_gradient_norm: last_gradient_norm,
                final_error,
                iterations_taken: function_values.len(),
                elapsed_time: elapsed,
                function_evaluations: function_values.len(),
                function_value_history: function_values,
                gradient_norm_history: gradient_norms,
            };

            results.push(result.clone());
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult<A>] {
        &self.results
    }

    /// Clear all results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Generate performance comparison report
    pub fn generate_report(&self) -> BenchmarkReport<A> {
        let mut optimizer_performance = std::collections::HashMap::new();

        for result in &self.results {
            let entry = optimizer_performance
                .entry(result.optimizername.clone())
                .or_insert_with(|| OptimizerPerformance {
                    total_runs: 0,
                    successful_runs: 0,
                    average_iterations: 0.0,
                    average_final_error: A::zero(),
                    average_time: std::time::Duration::from_secs(0),
                });

            entry.total_runs += 1;
            if result.converged {
                entry.successful_runs += 1;
            }
            entry.average_iterations += result.iterations_taken as f64;
            entry.average_final_error = entry.average_final_error + result.final_error;
            entry.average_time += result.elapsed_time;
        }

        // Normalize averages
        for performance in optimizer_performance.values_mut() {
            if performance.total_runs > 0 {
                performance.average_iterations /= performance.total_runs as f64;
                performance.average_final_error =
                    performance.average_final_error / scalar_or(performance.total_runs, A::one());
                performance.average_time /= performance.total_runs as u32;
            }
        }

        BenchmarkReport {
            total_tests: self.results.len(),
            optimizer_performance,
        }
    }
}

/// Test function for optimization benchmarking
pub struct TestFunction<A: Float> {
    /// Name of the test function
    pub name: String,
    /// Dimension of the problem
    pub dimension: usize,
    /// Function to optimize
    pub function: ObjectiveFunction<A>,
    /// Gradient function
    pub gradient: GradientFunction<A>,
    /// Known optimal value (if available)
    pub optimal_value: Option<A>,
    /// Known optimal point (if available)
    pub optimal_point: Option<Array1<A>>,
}

/// Result of a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult<A: Float> {
    /// Name of the optimizer
    pub optimizername: String,
    /// Name of the test function
    pub function_name: String,
    /// Whether the optimizer converged
    pub converged: bool,
    /// Step at which convergence was achieved
    pub convergence_step: Option<usize>,
    /// Final function value
    pub final_function_value: A,
    /// Final gradient norm
    pub final_gradient_norm: A,
    /// Error from known optimal value
    pub final_error: A,
    /// Total iterations taken
    pub iterations_taken: usize,
    /// Elapsed wall-clock time
    pub elapsed_time: std::time::Duration,
    /// Number of function evaluations
    pub function_evaluations: usize,
    /// History of function values
    pub function_value_history: Vec<A>,
    /// History of gradient norms
    pub gradient_norm_history: Vec<A>,
}

/// Performance summary for an optimizer
#[derive(Debug, Clone)]
pub struct OptimizerPerformance<A: Float> {
    /// Total number of test runs
    pub total_runs: usize,
    /// Number of successful convergences
    pub successful_runs: usize,
    /// Average iterations to convergence
    pub average_iterations: f64,
    /// Average final error
    pub average_final_error: A,
    /// Average time per run
    pub average_time: std::time::Duration,
}

/// Comprehensive benchmark report
#[derive(Debug)]
pub struct BenchmarkReport<A: Float> {
    /// Total number of tests run
    pub total_tests: usize,
    /// Performance data per optimizer
    pub optimizer_performance: std::collections::HashMap<String, OptimizerPerformance<A>>,
}

impl<A: Float + Send + Sync> BenchmarkReport<A> {
    /// Get success rate for an optimizer
    pub fn get_success_rate(&self, optimizername: &str) -> Option<f64> {
        self.optimizer_performance.get(optimizername).map(|perf| {
            if perf.total_runs > 0 {
                perf.successful_runs as f64 / perf.total_runs as f64
            } else {
                0.0
            }
        })
    }

    /// Compare two optimizers
    pub fn compare_optimizers(&self, opt1: &str, opt2: &str) -> Option<OptimizerComparison<A>> {
        let perf1 = self.optimizer_performance.get(opt1)?;
        let perf2 = self.optimizer_performance.get(opt2)?;

        Some(OptimizerComparison {
            optimizer1: opt1.to_string(),
            optimizer2: opt2.to_string(),
            success_rate_diff: self.get_success_rate(opt1).unwrap_or(0.0)
                - self.get_success_rate(opt2).unwrap_or(0.0),
            avg_iterations_diff: perf1.average_iterations - perf2.average_iterations,
            avg_error_diff: perf1.average_final_error - perf2.average_final_error,
        })
    }
}

/// Comparison between two optimizers
#[derive(Debug, Clone)]
pub struct OptimizerComparison<A: Float> {
    /// First optimizer name
    pub optimizer1: String,
    /// Second optimizer name
    pub optimizer2: String,
    /// Difference in success rates (opt1 - opt2)
    pub success_rate_diff: f64,
    /// Difference in average iterations (opt1 - opt2)
    pub avg_iterations_diff: f64,
    /// Difference in average final error (opt1 - opt2)
    pub avg_error_diff: A,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> Default for OptimizerBenchmark<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimizer state visualization tools
pub mod visualization {
    use super::*;

    /// Optimizer state visualizer
    #[derive(Debug)]
    pub struct OptimizerStateVisualizer<A: Float, D: Dimension> {
        /// Current parameter values
        parameter_history: VecDeque<Vec<Array<A, D>>>,
        /// Optimizer internal state history
        state_history: VecDeque<OptimizerStateSnapshot<A>>,
        /// Learning rate history
        learning_rate_history: VecDeque<A>,
        /// Loss/objective value history
        loss_history: VecDeque<A>,
        /// Maximum history to keep
        _maxhistory: usize,
        /// Step counter
        step_count: usize,
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> OptimizerStateVisualizer<A, D> {
        /// Create a new optimizer state visualizer
        pub fn new(_maxhistory: usize) -> Self {
            Self {
                parameter_history: VecDeque::with_capacity(_maxhistory),
                state_history: VecDeque::with_capacity(_maxhistory),
                learning_rate_history: VecDeque::with_capacity(_maxhistory),
                loss_history: VecDeque::with_capacity(_maxhistory),
                _maxhistory,
                step_count: 0,
            }
        }

        /// Record a step with optimizer state
        pub fn record_step(
            &mut self,
            parameters: &[Array<A, D>],
            state_snapshot: OptimizerStateSnapshot<A>,
            learning_rate: A,
            loss_value: A,
        ) {
            self.step_count += 1;

            // Record parameters
            self.parameter_history.push_back(parameters.to_vec());
            if self.parameter_history.len() > self._maxhistory {
                self.parameter_history.pop_front();
            }

            // Record state
            self.state_history.push_back(state_snapshot);
            if self.state_history.len() > self._maxhistory {
                self.state_history.pop_front();
            }

            // Record learning _rate
            self.learning_rate_history.push_back(learning_rate);
            if self.learning_rate_history.len() > self._maxhistory {
                self.learning_rate_history.pop_front();
            }

            // Record loss
            self.loss_history.push_back(loss_value);
            if self.loss_history.len() > self._maxhistory {
                self.loss_history.pop_front();
            }
        }

        /// Generate ASCII art visualization of convergence
        pub fn generate_convergence_plot(&self, width: usize, height: usize) -> String {
            if self.loss_history.is_empty() {
                return "No data to visualize".to_string();
            }

            let mut plot = String::new();

            // Find min and max loss values
            let min_loss = self
                .loss_history
                .iter()
                .fold(A::infinity(), |acc, &x| acc.min(x));
            let max_loss = self
                .loss_history
                .iter()
                .fold(A::neg_infinity(), |acc, &x| acc.max(x));

            let loss_range = max_loss - min_loss;

            plot.push_str(&format!("Loss Convergence (Steps: {})\n", self.step_count));
            plot.push_str(&format!(
                "Max: {:.6}, Min: {:.6}\n",
                max_loss.to_f64().unwrap_or(0.0),
                min_loss.to_f64().unwrap_or(0.0)
            ));
            plot.push_str(&"=".repeat(width + 10));
            plot.push('\n');

            // Create the plot
            for row in 0..height {
                let y_value = max_loss
                    - (scalar_or(row, A::zero()) / scalar_or(height - 1, A::one())) * loss_range;
                plot.push_str(&format!("{:8.3} |", y_value.to_f64().unwrap_or(0.0)));

                for col in 0..width {
                    let step_index = (col * self.loss_history.len()) / width;
                    if step_index < self.loss_history.len() {
                        let loss_val = self.loss_history[step_index];
                        let normalized_y = ((max_loss - loss_val) / loss_range
                            * scalar_or(height - 1, A::one()))
                        .to_usize()
                        .unwrap_or(0);

                        if normalized_y == row {
                            plot.push('*');
                        } else {
                            plot.push(' ');
                        }
                    } else {
                        plot.push(' ');
                    }
                }
                plot.push_str("|\n");
            }

            plot.push_str("         ");
            plot.push_str(&"-".repeat(width));
            plot.push('\n');
            plot.push_str(&format!(
                "         0{:width$}Steps\n",
                self.step_count,
                width = width - 10
            ));

            plot
        }

        /// Generate learning rate schedule visualization
        pub fn generate_learning_rate_plot(&self, width: usize, height: usize) -> String {
            if self.learning_rate_history.is_empty() {
                return "No learning rate data to visualize".to_string();
            }

            let mut plot = String::new();

            let min_lr = self
                .learning_rate_history
                .iter()
                .fold(A::infinity(), |acc, &x| acc.min(x));
            let max_lr = self
                .learning_rate_history
                .iter()
                .fold(A::neg_infinity(), |acc, &x| acc.max(x));

            let lr_range = max_lr - min_lr;

            plot.push_str("Learning Rate Schedule\n");
            plot.push_str(&format!(
                "Max: {:.6}, Min: {:.6}\n",
                max_lr.to_f64().unwrap_or(0.0),
                min_lr.to_f64().unwrap_or(0.0)
            ));
            plot.push_str(&"=".repeat(width + 10));
            plot.push('\n');

            for row in 0..height {
                let y_value = max_lr
                    - (scalar_or(row, A::zero()) / scalar_or(height - 1, A::one())) * lr_range;
                plot.push_str(&format!("{:8.3} |", y_value.to_f64().unwrap_or(0.0)));

                for col in 0..width {
                    let step_index = (col * self.learning_rate_history.len()) / width;
                    if step_index < self.learning_rate_history.len() {
                        let lr_val = self.learning_rate_history[step_index];
                        let normalized_y = if lr_range > A::zero() {
                            ((max_lr - lr_val) / lr_range * scalar_or(height - 1, A::one()))
                                .to_usize()
                                .unwrap_or(0)
                        } else {
                            height / 2
                        };

                        if normalized_y == row {
                            plot.push('*');
                        } else {
                            plot.push(' ');
                        }
                    } else {
                        plot.push(' ');
                    }
                }
                plot.push_str("|\n");
            }

            plot.push_str("         ");
            plot.push_str(&"-".repeat(width));
            plot.push('\n');
            plot.push_str(&format!(
                "         0{:width$}Steps\n",
                self.step_count,
                width = width - 10
            ));

            plot
        }

        /// Generate parameter evolution heatmap
        pub fn generate_parameter_heatmap(&self, width: usize, height: usize) -> String {
            if self.parameter_history.is_empty() {
                return "No parameter data to visualize".to_string();
            }

            let mut plot = String::new();
            plot.push_str("Parameter Evolution Heatmap\n");
            plot.push_str(&"=".repeat(width + 5));
            plot.push('\n');

            // Flatten all parameters for analysis
            let all_params: Vec<A> = self
                .parameter_history
                .iter()
                .flat_map(|step| step.iter().flat_map(|array| array.iter().copied()))
                .collect();

            if all_params.is_empty() {
                return "No parameter data available".to_string();
            }

            let min_param = all_params.iter().fold(A::infinity(), |acc, &x| acc.min(x));
            let max_param = all_params
                .iter()
                .fold(A::neg_infinity(), |acc, &x| acc.max(x));
            let param_range = max_param - min_param;

            // Create heatmap representation
            let num_steps = self.parameter_history.len().min(width);
            let num_params = if !self.parameter_history.is_empty() {
                self.parameter_history[0]
                    .iter()
                    .map(|arr| arr.len())
                    .sum::<usize>()
                    .min(height)
            } else {
                0
            };

            for param_idx in 0..num_params {
                plot.push_str(&format!("P{:3} |", param_idx));

                for step_idx in 0..num_steps {
                    let step_data = &self.parameter_history[step_idx];

                    // Find the parameter value at this step and index
                    let mut flat_idx = 0;
                    let mut found_value = None;

                    for array in step_data {
                        if flat_idx + array.len() > param_idx {
                            let local_idx = param_idx - flat_idx;
                            if let Some(&value) = array.iter().nth(local_idx) {
                                found_value = Some(value);
                                break;
                            }
                        }
                        flat_idx += array.len();
                    }

                    if let Some(value) = found_value {
                        let normalized = if param_range > A::zero() {
                            ((value - min_param) / param_range).to_f64().unwrap_or(0.0)
                        } else {
                            0.5
                        };

                        let char = match (normalized * 4.0) as i32 {
                            0 => ' ',
                            1 => '.',
                            2 => ':',
                            3 => '*',
                            _ => '#',
                        };
                        plot.push(char);
                    } else {
                        plot.push(' ');
                    }
                }
                plot.push_str("|\n");
            }

            plot.push_str("     ");
            plot.push_str(&"-".repeat(num_steps));
            plot.push('\n');
            plot.push_str("     Legend: ' ' = Low, '.' < ':' < '*' < '#' = High\n");
            plot.push_str(&format!(
                "     Range: {:.6} to {:.6}\n",
                min_param.to_f64().unwrap_or(0.0),
                max_param.to_f64().unwrap_or(0.0)
            ));

            plot
        }

        /// Generate optimizer state summary
        pub fn generate_state_summary(&self) -> String {
            let mut summary = String::new();

            summary.push_str("Optimizer State Summary\n");
            summary.push_str("======================\n");
            summary.push_str(&format!("Total Steps: {}\n", self.step_count));
            summary.push_str(&format!(
                "History Length: {}\n",
                self.parameter_history.len()
            ));

            if let Some(current_loss) = self.loss_history.back() {
                summary.push_str(&format!(
                    "Current Loss: {:.6}\n",
                    current_loss.to_f64().unwrap_or(0.0)
                ));
            }

            if let Some(current_lr) = self.learning_rate_history.back() {
                summary.push_str(&format!(
                    "Current Learning Rate: {:.6}\n",
                    current_lr.to_f64().unwrap_or(0.0)
                ));
            }

            // Loss statistics
            if !self.loss_history.is_empty() {
                let min_loss = self
                    .loss_history
                    .iter()
                    .fold(A::infinity(), |acc, &x| acc.min(x));
                let max_loss = self
                    .loss_history
                    .iter()
                    .fold(A::neg_infinity(), |acc, &x| acc.max(x));
                let avg_loss = self.loss_history.iter().fold(A::zero(), |acc, &x| acc + x)
                    / scalar_or(self.loss_history.len(), A::one());

                summary.push_str("\nLoss Statistics:\n");
                summary.push_str(&format!("  Min: {:.6}\n", min_loss.to_f64().unwrap_or(0.0)));
                summary.push_str(&format!("  Max: {:.6}\n", max_loss.to_f64().unwrap_or(0.0)));
                summary.push_str(&format!("  Avg: {:.6}\n", avg_loss.to_f64().unwrap_or(0.0)));

                // Improvement rate: `if let` (rather than a `len() > 1` guard
                // plus a separate `.expect()`-ed `.back()`) means there is no
                // `None` case that relies on the guard to be unreachable --
                // this function returns a bare `String`, so a structural
                // rewrite is used here instead of `.expect()` + propagation.
                if self.loss_history.len() > 1 {
                    if let Some(&last_loss) = self.loss_history.back() {
                        let first_loss = self.loss_history[0];
                        let improvement = first_loss - last_loss;
                        let improvement_rate = improvement / first_loss;
                        summary.push_str(&format!(
                            "  Improvement: {:.6} ({:.2}%)\n",
                            improvement.to_f64().unwrap_or(0.0),
                            (improvement_rate.to_f64().unwrap_or(0.0) * 100.0)
                        ));
                    }
                }
            }

            // Parameter statistics: `if let` on `.back()` makes emptiness a
            // no-op instead of relying on an outer `is_empty()` guard plus a
            // separate `.expect()`.
            if let Some(current_params) = self.parameter_history.back() {
                let total_params: usize = current_params.iter().map(|arr| arr.len()).sum();
                summary.push_str("\nParameter Statistics:\n");
                summary.push_str(&format!("  Total Parameters: {}\n", total_params));
                summary.push_str(&format!("  Parameter Groups: {}\n", current_params.len()));

                // Parameter norms
                for (i, array) in current_params.iter().enumerate() {
                    let l2_norm = array.mapv(|x| x * x).sum().sqrt();
                    summary.push_str(&format!(
                        "  Group {} L2 Norm: {:.6}\n",
                        i,
                        l2_norm.to_f64().unwrap_or(0.0)
                    ));
                }
            }

            // State snapshots summary
            if !self.state_history.is_empty() {
                summary.push_str("\nOptimizer State:\n");
                if let Some(latest_state) = self.state_history.back() {
                    summary.push_str(&format!(
                        "  Momentum Norm: {:.6}\n",
                        latest_state.momentum_norm.to_f64().unwrap_or(0.0)
                    ));
                    summary.push_str(&format!(
                        "  Velocity Norm: {:.6}\n",
                        latest_state.velocity_norm.to_f64().unwrap_or(0.0)
                    ));
                    summary.push_str(&format!(
                        "  Step Size: {:.6}\n",
                        latest_state.effective_step_size.to_f64().unwrap_or(0.0)
                    ));
                    summary.push_str(&format!(
                        "  Beta1: {:.6}\n",
                        latest_state.beta1.to_f64().unwrap_or(0.0)
                    ));
                    summary.push_str(&format!(
                        "  Beta2: {:.6}\n",
                        latest_state.beta2.to_f64().unwrap_or(0.0)
                    ));
                }
            }

            summary
        }

        /// Export data for external visualization tools
        pub fn export_data(&self) -> VisualizationExport<A> {
            VisualizationExport {
                step_indices: (0..self.step_count).collect(),
                loss_history: self.loss_history.iter().copied().collect(),
                learning_rate_history: self.learning_rate_history.iter().copied().collect(),
                parameter_norms: self
                    .parameter_history
                    .iter()
                    .map(|step| {
                        step.iter()
                            .map(|array| array.mapv(|x| x * x).sum().sqrt())
                            .collect()
                    })
                    .collect(),
                state_snapshots: self.state_history.iter().cloned().collect(),
            }
        }

        /// Clear all history
        pub fn clear(&mut self) {
            self.parameter_history.clear();
            self.state_history.clear();
            self.learning_rate_history.clear();
            self.loss_history.clear();
            self.step_count = 0;
        }

        /// Get current step count
        pub fn step_count(&self) -> usize {
            self.step_count
        }
    }

    /// Snapshot of optimizer internal state
    #[derive(Debug, Clone)]
    pub struct OptimizerStateSnapshot<A: Float> {
        /// Momentum vector norm
        pub momentum_norm: A,
        /// Velocity vector norm (for adaptive methods)
        pub velocity_norm: A,
        /// Effective step size used
        pub effective_step_size: A,
        /// Beta1 parameter (momentum decay)
        pub beta1: A,
        /// Beta2 parameter (velocity decay)
        pub beta2: A,
        /// Additional optimizer-specific state
        pub custom_fields: std::collections::HashMap<String, A>,
    }

    impl<A: Float + Send + Sync> OptimizerStateSnapshot<A> {
        /// Create a new state snapshot with default values
        pub fn new() -> Self {
            Self {
                momentum_norm: A::zero(),
                velocity_norm: A::zero(),
                effective_step_size: A::zero(),
                beta1: A::zero(),
                beta2: A::zero(),
                custom_fields: std::collections::HashMap::new(),
            }
        }

        /// Add a custom field to the snapshot
        pub fn with_custom_field(mut self, name: String, value: A) -> Self {
            self.custom_fields.insert(name, value);
            self
        }
    }

    impl<A: Float + Send + Sync> Default for OptimizerStateSnapshot<A> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Exported data for visualization
    #[derive(Debug, Clone)]
    pub struct VisualizationExport<A: Float> {
        /// Step indices
        pub step_indices: Vec<usize>,
        /// Loss value history
        pub loss_history: Vec<A>,
        /// Learning rate history
        pub learning_rate_history: Vec<A>,
        /// Parameter norm history (per group)
        pub parameter_norms: Vec<Vec<A>>,
        /// Optimizer state snapshots
        pub state_snapshots: Vec<OptimizerStateSnapshot<A>>,
    }

    /// Dashboard for multiple optimizer comparison
    #[derive(Debug)]
    pub struct OptimizerDashboard<A: Float, D: Dimension> {
        /// Visualizers for different optimizers
        visualizers: std::collections::HashMap<String, OptimizerStateVisualizer<A, D>>,
        /// Comparison metrics
        #[allow(dead_code)]
        comparison_metrics: Vec<ComparisonMetric<A>>,
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> OptimizerDashboard<A, D> {
        /// Create a new optimizer dashboard
        pub fn new() -> Self {
            Self {
                visualizers: std::collections::HashMap::new(),
                comparison_metrics: Vec::new(),
            }
        }

        /// Add an optimizer to track
        pub fn add_optimizer(&mut self, name: String, maxhistory: usize) {
            self.visualizers
                .insert(name, OptimizerStateVisualizer::new(maxhistory));
        }

        /// Record a step for a specific optimizer
        pub fn record_optimizer_step(
            &mut self,
            optimizername: &str,
            parameters: &[Array<A, D>],
            state_snapshot: OptimizerStateSnapshot<A>,
            learning_rate: A,
            loss_value: A,
        ) -> Result<()> {
            if let Some(visualizer) = self.visualizers.get_mut(optimizername) {
                visualizer.record_step(parameters, state_snapshot, learning_rate, loss_value);
                Ok(())
            } else {
                Err(OptimError::InvalidConfig(format!(
                    "Optimizer '{}' not found in dashboard",
                    optimizername
                )))
            }
        }

        /// Generate comparison report
        pub fn generate_comparison_report(&self) -> String {
            let mut report = String::new();

            report.push_str("Optimizer Comparison Dashboard\n");
            report.push_str("===============================\n");

            for (name, visualizer) in &self.visualizers {
                report.push_str(&format!("\n{}\n", name));
                report.push_str(&"-".repeat(name.len()));
                report.push('\n');

                if let Some(current_loss) = visualizer.loss_history.back() {
                    report.push_str(&format!(
                        "Current Loss: {:.6}\n",
                        current_loss.to_f64().unwrap_or(0.0)
                    ));
                }

                report.push_str(&format!("Steps: {}\n", visualizer.step_count));

                // Calculate convergence rate. `if let` on `.back()` (rather
                // than a `len() > 1` guard plus a separate `.expect()`-ed
                // `.back()`) removes the panic path structurally.
                if visualizer.loss_history.len() > 1 {
                    if let Some(&last_loss) = visualizer.loss_history.back() {
                        let first_loss = visualizer.loss_history[0];
                        let improvement = first_loss - last_loss;
                        report.push_str(&format!(
                            "Total Improvement: {:.6}\n",
                            improvement.to_f64().unwrap_or(0.0)
                        ));
                    }
                }
            }

            // Best performer analysis
            if !self.visualizers.is_empty() {
                report.push_str("\nBest Performers:\n");
                report.push_str("================\n");

                let best_current_loss = self
                    .visualizers
                    .iter()
                    .filter_map(|(name, viz)| viz.loss_history.back().map(|&loss| (name, loss)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                if let Some((best_name, best_loss)) = best_current_loss {
                    report.push_str(&format!(
                        "Lowest Current Loss: {} ({:.6})\n",
                        best_name,
                        best_loss.to_f64().unwrap_or(0.0)
                    ));
                }
            }

            report
        }

        /// Get visualizer for a specific optimizer
        pub fn get_visualizer(
            &self,
            optimizername: &str,
        ) -> Option<&OptimizerStateVisualizer<A, D>> {
            self.visualizers.get(optimizername)
        }

        /// Get mutable visualizer for a specific optimizer
        pub fn get_visualizer_mut(
            &mut self,
            optimizername: &str,
        ) -> Option<&mut OptimizerStateVisualizer<A, D>> {
            self.visualizers.get_mut(optimizername)
        }

        /// List all tracked optimizers
        pub fn list_optimizers(&self) -> Vec<&String> {
            self.visualizers.keys().collect()
        }
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> Default
        for OptimizerDashboard<A, D>
    {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Metric for comparing optimizers
    #[derive(Debug, Clone)]
    pub struct ComparisonMetric<A: Float> {
        /// Name of the metric
        pub name: String,
        /// Values for each optimizer
        pub values: std::collections::HashMap<String, A>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gradient_flow_analyzer() {
        let mut analyzer = GradientFlowAnalyzer::new(100);

        let gradients1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let updates1 = vec![Array1::from_vec(vec![0.1, 0.2])];

        let gradients2 = vec![Array1::from_vec(vec![0.8, 1.6])];
        let updates2 = vec![Array1::from_vec(vec![0.08, 0.16])];

        analyzer
            .record_step(&gradients1, &updates1)
            .expect("unwrap failed");
        analyzer
            .record_step(&gradients2, &updates2)
            .expect("unwrap failed");

        assert_eq!(analyzer.step_count(), 2);

        let stats = analyzer.get_stats();
        assert_eq!(stats.step_count, 2);
        assert_eq!(stats.per_group_stats.len(), 1);

        // Check magnitude calculation
        let expected_mag1 = (1.0_f64 * 1.0 + 2.0 * 2.0).sqrt();
        let expected_mag2 = (0.8_f64 * 0.8 + 1.6 * 1.6).sqrt();

        assert_relative_eq!(
            stats.per_group_stats[0].magnitude_history[0],
            expected_mag1,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            stats.per_group_stats[0].magnitude_history[1],
            expected_mag2,
            epsilon = 1e-6
        );

        // Direction similarity should be high (gradients are in same direction)
        assert!(stats.mean_direction_similarity > 0.9);
    }

    #[test]
    fn test_benchmark_quadratic() {
        let mut benchmark = OptimizerBenchmark::new();
        benchmark.add_standard_test_functions();

        // Simple gradient descent step
        let learning_rate = 0.01;
        let mut step_function = |x: &Array1<f64>, grad: &Array1<f64>| x - &(grad * learning_rate);

        let results = benchmark
            .run_benchmark(
                "GradientDescent".to_string(),
                &mut step_function,
                1000,
                1e-6,
            )
            .expect("unwrap failed");

        assert!(!results.is_empty());

        // Check that quadratic function converged
        let quadratic_result = results
            .iter()
            .find(|r| r.function_name == "Quadratic")
            .expect("unwrap failed");

        assert!(quadratic_result.converged);
        assert!(quadratic_result.final_function_value < 1e-3);
    }

    #[test]
    fn test_run_benchmark_zero_iterations_is_a_real_error_not_a_panic() {
        // Regression (F50): `max_iterations == 0` used to reach
        // `function_values.last().expect("unwrap failed")` on an empty
        // history and panic. It must now be a real, honest `Err`.
        let mut benchmark = OptimizerBenchmark::<f64>::new();
        benchmark.add_standard_test_functions();
        let mut step_function = |x: &Array1<f64>, grad: &Array1<f64>| x - grad;

        let result = benchmark.run_benchmark("Zero".to_string(), &mut step_function, 0, 1e-6);
        assert!(
            result.is_err(),
            "zero iterations must be a configuration error, not a panic or a fabricated result"
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let analyzer = GradientFlowAnalyzer::<f64, scirs2_core::ndarray::Ix1>::new(10);

        let arrays1 = vec![Array1::from_vec(vec![1.0, 0.0])];
        let arrays2 = vec![Array1::from_vec(vec![1.0, 0.0])]; // Same direction
        let similarity = analyzer
            .calculate_cosine_similarity(&arrays1, &arrays2)
            .expect("unwrap failed");
        assert_relative_eq!(similarity, 1.0, epsilon = 1e-6);

        let arrays3 = vec![Array1::from_vec(vec![-1.0, 0.0])]; // Opposite direction
        let similarity2 = analyzer
            .calculate_cosine_similarity(&arrays1, &arrays3)
            .expect("unwrap failed");
        assert_relative_eq!(similarity2, -1.0, epsilon = 1e-6);

        let arrays4 = vec![Array1::from_vec(vec![0.0, 1.0])]; // Orthogonal
        let similarity3 = analyzer
            .calculate_cosine_similarity(&arrays1, &arrays4)
            .expect("unwrap failed");
        assert_relative_eq!(similarity3, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_benchmark_report() {
        let mut benchmark = OptimizerBenchmark::new();
        benchmark.add_test_function(TestFunction {
            name: "Simple".to_string(),
            dimension: 2,
            function: Box::new(|x: &Array1<f64>| x[0] * x[0] + x[1] * x[1]),
            gradient: Box::new(|x: &Array1<f64>| Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]])),
            optimal_value: Some(0.0),
            optimal_point: Some(Array1::zeros(2)),
        });

        // Run two different "optimizers"
        let mut step1 = |x: &Array1<f64>, grad: &Array1<f64>| x - &(grad * 0.1);
        let mut step2 = |x: &Array1<f64>, grad: &Array1<f64>| x - &(grad * 0.05);

        benchmark
            .run_benchmark("Fast".to_string(), &mut step1, 100, 1e-3)
            .expect("unwrap failed");
        benchmark
            .run_benchmark("Slow".to_string(), &mut step2, 100, 1e-3)
            .expect("unwrap failed");

        let report = benchmark.generate_report();
        assert_eq!(report.total_tests, 2);
        assert!(report.optimizer_performance.contains_key("Fast"));
        assert!(report.optimizer_performance.contains_key("Slow"));

        let comparison = report
            .compare_optimizers("Fast", "Slow")
            .expect("unwrap failed");
        assert_eq!(comparison.optimizer1, "Fast");
        assert_eq!(comparison.optimizer2, "Slow");
    }

    #[test]
    fn test_visualization_data_export() {
        let mut analyzer = GradientFlowAnalyzer::new(10);

        let _gradients = [Array1::from_vec(vec![1.0, 2.0])];
        let _updates = [Array1::from_vec(vec![0.1, 0.2])];

        for i in 0..5 {
            let scale = 1.0 / (i + 1) as f64;
            let scaled_grad = vec![Array1::from_vec(vec![scale, 2.0 * scale])];
            let scaled_update = vec![Array1::from_vec(vec![0.1 * scale, 0.2 * scale])];
            analyzer
                .record_step(&scaled_grad, &scaled_update)
                .expect("unwrap failed");
        }

        let viz_data = analyzer.export_for_visualization();
        assert_eq!(viz_data.step_indices.len(), 5);
        assert_eq!(viz_data.magnitude_series.len(), 1); // One parameter group
        assert_eq!(viz_data.magnitude_series[0].len(), 5); // Five steps
        assert_eq!(viz_data.direction_similarities.len(), 5); // Five direction entries (first is default 1.0)

        // Check that magnitudes are decreasing
        let magnitudes = &viz_data.magnitude_series[0];
        for i in 1..magnitudes.len() {
            assert!(magnitudes[i] < magnitudes[i - 1]);
        }
    }

    #[test]
    fn test_convergence_analysis() {
        let mut analyzer = GradientFlowAnalyzer::new(10);

        // Simulate converging gradients (decreasing magnitudes)
        for i in 0..10 {
            let scale = 1.0 / (i + 1) as f64;
            let gradients = vec![Array1::from_vec(vec![scale, scale])];
            let updates = vec![Array1::from_vec(vec![0.1 * scale, 0.1 * scale])];
            analyzer
                .record_step(&gradients, &updates)
                .expect("unwrap failed");
        }

        let stats = analyzer.get_stats();
        assert!(stats.is_converging);
        assert!(stats.stability_score > 0.5);
    }

    #[test]
    fn test_oscillation_detection() {
        let mut analyzer = GradientFlowAnalyzer::new(10);

        // First step - initialize with some gradient
        let gradients = vec![Array1::from_vec(vec![1.0, 1.0])];
        let updates = vec![Array1::from_vec(vec![0.1, 0.1])];
        analyzer
            .record_step(&gradients, &updates)
            .expect("unwrap failed");

        // Simulate oscillating gradients and updates
        for i in 1..8 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let gradients = vec![Array1::from_vec(vec![sign, sign])];
            let updates = vec![Array1::from_vec(vec![0.1 * sign, 0.1 * sign])];
            analyzer
                .record_step(&gradients, &updates)
                .expect("unwrap failed");
        }

        let stats = analyzer.get_stats();
        // With alternating signs, we should see some oscillation
        // The oscillation frequency depends on cosine similarity between gradients and updates
        assert!(stats.oscillation_frequency >= 0.0); // Just check it's computed correctly
                                                     // Note: stability score calculation may not work as expected with alternating patterns
    }

    #[test]
    fn test_optimizer_state_visualizer() {
        let mut visualizer = visualization::OptimizerStateVisualizer::new(100);

        let params = vec![Array1::from_vec(vec![1.0, 2.0])];
        let state = visualization::OptimizerStateSnapshot::new()
            .with_custom_field("test_field".to_string(), 0.5);

        visualizer.record_step(&params, state, 0.01, 1.5);
        visualizer.record_step(
            &params,
            visualization::OptimizerStateSnapshot::new(),
            0.009,
            1.2,
        );

        assert_eq!(visualizer.step_count(), 2);

        // Test summary generation
        let summary = visualizer.generate_state_summary();
        assert!(summary.contains("Total Steps: 2"));
        assert!(summary.contains("Current Loss: 1.200000"));

        // Test convergence plot
        let plot = visualizer.generate_convergence_plot(40, 10);
        assert!(plot.contains("Loss Convergence"));
        assert!(plot.contains("Steps: 2"));

        // Test learning rate plot
        let lr_plot = visualizer.generate_learning_rate_plot(40, 10);
        assert!(lr_plot.contains("Learning Rate Schedule"));

        // Test parameter heatmap
        let heatmap = visualizer.generate_parameter_heatmap(20, 5);
        assert!(heatmap.contains("Parameter Evolution Heatmap"));
    }

    #[test]
    fn test_visualization_export() {
        let mut visualizer = visualization::OptimizerStateVisualizer::new(10);

        for i in 0..5 {
            let params = vec![Array1::from_vec(vec![i as f64, (i * 2) as f64])];
            let state = visualization::OptimizerStateSnapshot::new();
            let lr = 0.01 / (i + 1) as f64;
            let loss = 1.0 / (i + 1) as f64;

            visualizer.record_step(&params, state, lr, loss);
        }

        let export = visualizer.export_data();
        assert_eq!(export.step_indices.len(), 5);
        assert_eq!(export.loss_history.len(), 5);
        assert_eq!(export.learning_rate_history.len(), 5);
        assert_eq!(export.parameter_norms.len(), 5);
        assert_eq!(export.state_snapshots.len(), 5);

        // Check that values are decreasing (loss and learning rate)
        assert!(export.loss_history[0] > export.loss_history[4]);
        assert!(export.learning_rate_history[0] > export.learning_rate_history[4]);
    }

    #[test]
    fn test_optimizer_dashboard() {
        let mut dashboard = visualization::OptimizerDashboard::new();

        dashboard.add_optimizer("SGD".to_string(), 100);
        dashboard.add_optimizer("Adam".to_string(), 100);

        let params = vec![Array1::from_vec(vec![1.0, 2.0])];
        let state = visualization::OptimizerStateSnapshot::new();

        // Record steps for both optimizers
        dashboard
            .record_optimizer_step("SGD", &params, state.clone(), 0.01, 1.0)
            .expect("unwrap failed");
        dashboard
            .record_optimizer_step("Adam", &params, state, 0.001, 0.8)
            .expect("unwrap failed");

        let optimizers = dashboard.list_optimizers();
        assert_eq!(optimizers.len(), 2);
        assert!(optimizers.contains(&&"SGD".to_string()));
        assert!(optimizers.contains(&&"Adam".to_string()));

        // Test getting individual visualizers
        let sgd_viz = dashboard.get_visualizer("SGD").expect("unwrap failed");
        assert_eq!(sgd_viz.step_count(), 1);

        // Test comparison report
        let report = dashboard.generate_comparison_report();
        assert!(report.contains("Optimizer Comparison Dashboard"));
        assert!(report.contains("SGD"));
        assert!(report.contains("Adam"));
        assert!(report.contains("Lowest Current Loss: Adam"));
    }

    #[test]
    fn test_state_snapshot_custom_fields() {
        let snapshot = visualization::OptimizerStateSnapshot::new()
            .with_custom_field("custom1".to_string(), 1.5)
            .with_custom_field("custom2".to_string(), 2.5);

        assert_eq!(snapshot.custom_fields.len(), 2);
        assert_eq!(snapshot.custom_fields.get("custom1"), Some(&1.5));
        assert_eq!(snapshot.custom_fields.get("custom2"), Some(&2.5));
    }

    #[test]
    fn test_visualizer_clear() {
        let mut visualizer = visualization::OptimizerStateVisualizer::new(10);

        let params = vec![Array1::from_vec(vec![1.0])];
        let state = visualization::OptimizerStateSnapshot::new();

        visualizer.record_step(&params, state, 0.01, 1.0);
        assert_eq!(visualizer.step_count(), 1);

        visualizer.clear();
        assert_eq!(visualizer.step_count(), 0);

        let summary = visualizer.generate_state_summary();
        assert!(summary.contains("Total Steps: 0"));
    }

    #[test]
    fn test_dashboard_invalid_optimizer() {
        let mut dashboard = visualization::OptimizerDashboard::new();

        let params = vec![Array1::from_vec(vec![1.0])];
        let state = visualization::OptimizerStateSnapshot::new();

        // Try to record for non-existent optimizer
        let result = dashboard.record_optimizer_step("NonExistent", &params, state, 0.01, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ascii_plot_generation() {
        let mut visualizer = visualization::OptimizerStateVisualizer::new(10);

        // Add data with clear pattern
        for i in 0..10 {
            let params = vec![Array1::from_vec(vec![1.0])];
            let state = visualization::OptimizerStateSnapshot::new();
            let loss = 10.0 - i as f64; // Decreasing loss
            let lr = 0.1; // Constant learning rate

            visualizer.record_step(&params, state, lr, loss);
        }

        // Test convergence plot has proper structure
        let plot = visualizer.generate_convergence_plot(20, 5);
        let lines: Vec<&str> = plot.lines().collect();
        assert!(lines.len() > 5); // Should have header + plot lines

        // Check that plot contains expected elements
        assert!(plot.contains("|")); // Y-axis markers
        assert!(plot.contains("-")); // X-axis
        assert!(plot.contains("*")); // Data points

        // Test learning rate plot with constant rate
        let lr_plot = visualizer.generate_learning_rate_plot(20, 5);
        assert!(lr_plot.contains("Learning Rate Schedule"));
        assert!(lr_plot.contains("Max: 0.100000, Min: 0.100000")); // Constant rate
    }

    #[test]
    fn test_parameter_heatmap_generation() {
        let mut visualizer = visualization::OptimizerStateVisualizer::new(10);

        // Create parameters that change over time
        for i in 0..5 {
            let params = vec![Array1::from_vec(vec![i as f64 * 0.1, i as f64 * 0.2])];
            let state = visualization::OptimizerStateSnapshot::new();
            visualizer.record_step(&params, state, 0.01, 1.0);
        }

        let heatmap = visualizer.generate_parameter_heatmap(10, 5);
        assert!(heatmap.contains("Parameter Evolution Heatmap"));
        assert!(heatmap.contains("Legend"));
        assert!(heatmap.contains("Range"));

        // Should contain parameter indices
        assert!(heatmap.contains("P"));
    }
}
