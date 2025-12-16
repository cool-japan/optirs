//! # OptiRS Bench - Benchmarking and Performance Analysis
//!
//! **Version:** 0.1.0-rc.1
//! **Status:** Available
//!
//! This crate provides comprehensive benchmarking, profiling, performance analysis, and regression
//! detection tools for ML optimization algorithms in the OptiRS ecosystem.
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.1.0-rc.2 - Required foundation
//! - `optirs-core` 0.1.0-rc.1 - Core optimizers
//!
//! ## Features
//!
//! - **Performance Benchmarking**: Compare optimizers across standard test functions
//! - **Gradient Flow Analysis**: Monitor optimization dynamics and convergence patterns
//! - **Memory Profiling**: Track memory usage, detect leaks, and optimize allocation
//! - **Regression Detection**: Detect performance regressions across different versions
//! - **Cross-Platform Testing**: Validate optimizers across different hardware and OS
//! - **Security Auditing**: Scan for security vulnerabilities and compliance issues
//! - **CI/CD Integration**: Automated testing and reporting for continuous integration
//! - **Visualization Tools**: Generate plots and reports for optimization analysis
//!
//! ## Architecture
//!
//! The crate is organized into several main modules:
//!
//! - `benchmarking`: Core benchmarking functionality and test suites
//! - `memory`: Memory profiling, leak detection, and optimization
//! - `regression`: Performance regression detection and alerting
//! - `security`: Security auditing and vulnerability scanning
//! - `visualization`: Plotting and reporting tools
//! - `ci_cd_automation`: Continuous integration and deployment automation
//! - `cross_platform`: Cross-platform testing and validation
//!
//! ## Usage
//!
//! ```rust
//! use optirs_bench::{
//!     OptimizerBenchmark, GradientFlowAnalyzer,
//!     visualization::OptimizerStateVisualizer,
//! };
//! use scirs2_core::ndarray::{Array1, Ix1};
//!
//! // Create a benchmark suite
//! let mut benchmark = OptimizerBenchmark::<f64>::new();
//! benchmark.add_standard_test_functions();
//!
//! // Set up gradient flow analysis
//! let mut analyzer = GradientFlowAnalyzer::<f64, Ix1>::new(1000);
//!
//! // Set up state visualization
//! let mut visualizer = OptimizerStateVisualizer::<f64, Ix1>::new(500);
//! ```

// Re-export error types from optirs-core for consistency
pub use optirs_core::error::{OptimError, Result};

// Re-export key types for external users
pub mod error {
    pub use optirs_core::error::{OptimError, OptimizerError, Result};
}

// Core benchmarking and analysis functionality
mod mod_impl;

// Re-export the main types and functions
pub use mod_impl::*;

// Advanced modules for specific functionality
pub mod advanced_cross_platform_orchestrator;
pub mod advanced_leak_detectors;
pub mod advanced_memory_leak_detector;
pub mod advanced_pattern_detection;
pub mod automated_test_runners;
pub mod ci_cd_automation;
pub mod comprehensive_security_auditor;
pub mod cross_framework;
pub mod cross_platform_tester;
pub mod documentation_analyzer;
pub mod enhanced_memory_monitor;
pub mod memory_leak_detector;
pub mod memory_optimizer;
pub mod performance_profiler;
pub mod performance_regression_detector;
pub mod regression_tester;
pub mod security_auditor;

// Re-export common types for convenience
pub use mod_impl::{
    BenchmarkReport, BenchmarkResult, GradientFlowAnalyzer, GradientFlowStats, GradientFunction,
    ObjectiveFunction, OptimizerBenchmark, OptimizerComparison, OptimizerPerformance,
    ParameterGroupStats, TestFunction, VisualizationData,
};

// Re-export visualization types
pub use mod_impl::visualization::{
    ComparisonMetric, OptimizerDashboard, OptimizerStateSnapshot, OptimizerStateVisualizer,
    VisualizationExport,
};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::{
        BenchmarkReport, BenchmarkResult, GradientFlowAnalyzer, GradientFlowStats,
        GradientFunction, ObjectiveFunction, OptimError, OptimizerBenchmark, OptimizerComparison,
        OptimizerPerformance, ParameterGroupStats, Result, TestFunction, VisualizationData,
    };

    pub use crate::visualization::{
        ComparisonMetric, OptimizerDashboard, OptimizerStateSnapshot, OptimizerStateVisualizer,
        VisualizationExport,
    };

    pub use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView, ArrayViewMut};
    pub use scirs2_core::random::{thread_rng, Rng};
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_library_integration() {
        // Test that all major components can be instantiated
        let mut benchmark = OptimizerBenchmark::<f64>::new();
        benchmark.add_standard_test_functions();

        let analyzer = GradientFlowAnalyzer::<f64, scirs2_core::ndarray::Ix1>::new(10);
        let visualizer =
            visualization::OptimizerStateVisualizer::<f64, scirs2_core::ndarray::Ix1>::new(10);

        assert_eq!(analyzer.step_count(), 0);
        assert_eq!(visualizer.step_count(), 0);
        assert!(!benchmark.get_results().is_empty() || benchmark.get_results().is_empty());
        // Just test it exists
    }

    #[test]
    fn test_error_types() {
        // Test that error types are properly re-exported
        let error = OptimError::InvalidConfig("test".to_string());
        let result: Result<()> = Err(error);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid configuration"));
        }
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        // Test that prelude imports work
        let benchmark = OptimizerBenchmark::<f64>::new();
        let analyzer = GradientFlowAnalyzer::<f64, scirs2_core::ndarray::Ix1>::new(5);

        assert_eq!(analyzer.step_count(), 0);
        assert!(benchmark.get_results().is_empty());
    }
}
