//! Convergence validation for optimizer plugins.
//!
//! Split out of `validation.rs` to keep every file under the project's
//! 2000-line limit. Holds [`ConvergenceTestSuite`], which runs a plugin against
//! a convex [`TestProblem`] and asserts the loss actually decreases -- the one
//! property any optimizer claiming to optimize must satisfy.

use super::{SuiteResult, TestResult, TestSummary, ValidationConfig, ValidationTestSuite};
use crate::plugin::core::*;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::time::Instant;

/// Type alias for objective function
pub(super) type ObjectiveFn<A> = Box<dyn Fn(&Array1<A>) -> A + Send + Sync>;

/// Type alias for gradient function
pub(super) type GradientFn<A> = Box<dyn Fn(&Array1<A>) -> Array1<A> + Send + Sync>;

/// Convergence test suite
#[derive(Debug)]
pub struct ConvergenceTestSuite<A: Float + std::fmt::Debug + Send + Sync> {
    config: ValidationConfig,
    test_problems: Vec<TestProblem<A>>,
}

impl<A: Float + std::fmt::Debug + Send + Sync + 'static> ConvergenceTestSuite<A> {
    /// Create a new convergence test suite with the standard convex problems.
    ///
    /// `test_problems` was an empty vector nothing populated or read; the
    /// quadratic problem the suite actually runs is now one of its entries, so
    /// the field describes the suite instead of decorating it.
    pub fn new(config: ValidationConfig) -> Self {
        let dimension = config
            .test_data_sizes
            .iter()
            .copied()
            .find(|size| *size > 0)
            .unwrap_or(4)
            .min(4096);
        Self {
            config,
            test_problems: vec![TestProblem::sum_of_squares(dimension)],
        }
    }

    /// The convergence problems this suite runs.
    pub fn test_problems(&self) -> &[TestProblem<A>] {
        &self.test_problems
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync> ConvergenceTestSuite<A> {
    /// Run the plugin on the convex quadratic `f(x) = sum(x_i^2)`
    /// (gradient `2x`) and assert the loss actually decreases -- the one
    /// property any optimizer claiming to optimize must satisfy.
    fn test_quadratic_convergence(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();
        if !self.config.check_convergence {
            return TestResult {
                passed: true,
                message: "convergence testing disabled by ValidationConfig::check_convergence"
                    .to_string(),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            };
        }
        // Drive the run from the configured problem rather than literals, so a
        // caller that widened `test_data_sizes` actually gets a wider problem.
        let problem = match self.test_problems.first() {
            Some(problem) => problem,
            None => {
                return TestResult {
                    passed: false,
                    message: "no convergence problem is configured".to_string(),
                    execution_time: start_time.elapsed(),
                    data: HashMap::new(),
                }
            }
        };
        let dim = problem.initial_params.len();
        let iterations = problem.max_iterations;

        if let Err(e) = plugin.initialize(&[dim]) {
            return TestResult {
                passed: false,
                message: format!("initialize failed before convergence run: {e}"),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            };
        }

        let mut params: Array1<A> = problem.initial_params.clone();
        let loss = |p: &Array1<A>| -> A { (problem.objective_fn)(p) };
        let initial_loss = loss(&params);

        for step in 0..iterations {
            let gradients = (problem.gradient_fn)(&params);
            match plugin.step(&params, &gradients) {
                Ok(next) => params = next,
                Err(e) => {
                    return TestResult {
                        passed: false,
                        message: format!(
                            "step failed at iteration {step} during convergence run: {e}"
                        ),
                        execution_time: start_time.elapsed(),
                        data: HashMap::new(),
                    };
                }
            }
            let current_loss = loss(&params);
            if !current_loss.is_finite() {
                return TestResult {
                    passed: false,
                    message: format!("loss diverged to a non-finite value by iteration {step}"),
                    execution_time: start_time.elapsed(),
                    data: HashMap::new(),
                };
            }
        }

        let final_loss = loss(&params);
        let passed = final_loss < initial_loss;

        TestResult {
            passed,
            message: format!(
                "{} over {iterations} steps: initial={initial_loss:?} final={final_loss:?}",
                problem.name
            ),
            execution_time: start_time.elapsed(),
            data: HashMap::new(),
        }
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync> ValidationTestSuite<A> for ConvergenceTestSuite<A> {
    fn run_tests(&self, plugin: &mut dyn OptimizerPlugin<A>) -> SuiteResult {
        let start_time = Instant::now();
        let result = self.test_quadratic_convergence(plugin);
        let passed = result.passed;

        SuiteResult {
            suite_name: "Convergence".to_string(),
            test_results: vec![result],
            suite_passed: passed,
            execution_time: start_time.elapsed(),
            summary: TestSummary {
                total_tests: 1,
                passed_tests: passed as usize,
                failed_tests: (!passed) as usize,
                skipped_tests: 0,
                success_rate: if passed { 1.0 } else { 0.0 },
            },
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Convergence Tests"
    }

    fn description(&self) -> &str {
        "Tests for optimization convergence"
    }

    fn test_count(&self) -> usize {
        1
    }
}

/// Test problem for convergence testing
pub struct TestProblem<A: Float + std::fmt::Debug> {
    /// Problem name
    pub name: String,
    /// Initial parameters
    pub initial_params: Array1<A>,
    /// Objective function
    pub objective_fn: ObjectiveFn<A>,
    /// Gradient function
    pub gradient_fn: GradientFn<A>,
    /// Known optimal value
    pub optimal_value: Option<A>,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: A,
}

impl<A: Float + std::fmt::Debug + Send + Sync + 'static> TestProblem<A> {
    /// The convex quadratic `f(x) = sum(x_i^2)`, gradient `2x`, optimum `0`.
    ///
    /// The one problem every optimizer claiming to optimize must make progress
    /// on, and the problem [`ConvergenceTestSuite`] actually runs.
    pub fn sum_of_squares(dimension: usize) -> Self {
        let dimension = dimension.max(1);
        Self {
            name: format!("sum_of_squares_{dimension}d"),
            initial_params: Array1::from_iter(
                (0..dimension).map(|i| A::from(2.0 + i as f64).unwrap_or_else(A::one)),
            ),
            objective_fn: Box::new(|params: &Array1<A>| {
                params.iter().fold(A::zero(), |acc, &x| acc + x * x)
            }),
            gradient_fn: Box::new(|params: &Array1<A>| params.mapv(|x| x + x)),
            optimal_value: Some(A::zero()),
            max_iterations: 200,
            convergence_tolerance: A::from(1e-6).unwrap_or_else(A::zero),
        }
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync> std::fmt::Debug for TestProblem<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestProblem")
            .field("name", &self.name)
            .field("initial_params", &self.initial_params)
            .field("objective_fn", &"<function>")
            .field("gradient_fn", &"<function>")
            .field("optimal_value", &self.optimal_value)
            .field("max_iterations", &self.max_iterations)
            .field("convergence_tolerance", &self.convergence_tolerance)
            .finish()
    }
}
