//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Only exercised by the unit tests below -- gated so a non-test build does not
// warn about unused imports (this module currently has no non-test functions).
#[cfg(test)]
use super::constants::MISSING_DEPENDENCY_MARKER;
#[cfg(test)]
use super::crossframeworkbenchmark_type::CrossFrameworkBenchmark;
#[cfg(test)]
use super::types::{
    CrossFrameworkConfig, ExternalFrameworkOutcome, Framework, OptimizerIdentifier, Precision,
    PythonScriptTemplates,
};

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn test_cross_framework_config() {
        let config = CrossFrameworkConfig::default();
        assert!(config.enable_pytorch);
        assert!(config.enable_tensorflow);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
    }

    #[test]
    fn test_optimizer_identifier() {
        let id = OptimizerIdentifier {
            framework: Framework::SciRS2,
            name: "Adam".to_string(),
            version: Some("0.1.0".to_string()),
        };
        assert_eq!(id.to_string(), "SciRS2-Adam-v0.1.0");
    }

    #[test]
    fn test_precision_enum() {
        let precision = Precision::F64;
        assert!(matches!(precision, Precision::F64));
    }

    #[test]
    fn test_framework_display() {
        assert_eq!(Framework::SciRS2.to_string(), "SciRS2");
        assert_eq!(Framework::PyTorch.to_string(), "PyTorch");
        assert_eq!(Framework::TensorFlow.to_string(), "TensorFlow");
    }

    #[test]
    fn test_python_script_generation() {
        let templates = PythonScriptTemplates::new();
        let config = CrossFrameworkConfig::default();

        let script = templates.generate_pytorch_script("Quadratic", 10, 32, &config);
        assert!(script.contains("PROBLEM_DIM = 10"));
        assert!(script.contains("BATCH_SIZE = 32"));
        assert!(script.contains("MAX_ITERATIONS = 1000"));
        // The script must actually construct optimizers and emit JSON, not
        // just print a banner.
        assert!(script.contains("torch.optim.Adam"));
        assert!(script.contains("value.backward()"));
        assert!(script.contains("optimizer.step()"));
        assert!(script.contains("json.dump"));
        assert!(script.contains("convergence_times_secs"));
        assert!(script.contains(MISSING_DEPENDENCY_MARKER));
        assert!(script.contains("OPTIMIZERS = [\"Adam\", \"SGD\", \"RMSprop\"]"));

        let tf_script = templates.generate_tensorflow_script("Rosenbrock", 2, 1, &config);
        assert!(tf_script.contains("tf.GradientTape"));
        assert!(tf_script.contains("apply_gradients"));
        assert!(tf_script.contains("optimizers.Adam"));
        assert!(tf_script.contains("json.dump"));
    }

    fn test_benchmark() -> CrossFrameworkBenchmark<f64> {
        let config = CrossFrameworkConfig {
            temp_dir: std::env::temp_dir()
                .join("optirs_cross_framework_tests")
                .to_string_lossy()
                .into_owned(),
            ..Default::default()
        };
        CrossFrameworkBenchmark::<f64>::new(config).expect("temp dir is creatable")
    }

    /// F47: the previous t-distribution CDF saturated, so no comparison could
    /// ever be significant. Two clearly separated samples must now be.
    #[test]
    fn t_test_reports_real_p_values() {
        let benchmark = test_benchmark();

        let fast = [0.100, 0.102, 0.098, 0.101, 0.099, 0.100];
        let slow = [0.200, 0.202, 0.198, 0.201, 0.199, 0.200];
        let separated = benchmark.perform_t_test(&fast, &slow);
        assert!(separated.p_value < 1e-6, "p = {}", separated.p_value);
        assert!(separated.is_significant);
        assert!(separated.degrees_of_freedom > 0.0);

        let overlapping = benchmark.perform_t_test(&fast, &fast);
        assert!(overlapping.p_value > 0.5);
        assert!(!overlapping.is_significant);

        // Degenerate inputs must not divide by a zero pooled standard error.
        let degenerate = benchmark.perform_t_test(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]);
        assert_eq!(degenerate.p_value, 1.0);
        assert!(!degenerate.is_significant);
        assert!(!benchmark.perform_t_test(&[], &[1.0, 2.0]).is_significant);
    }

    #[test]
    fn t_distribution_cdf_matches_tables() {
        let benchmark = test_benchmark();
        // df = 10: P(T <= 3) = 0.993328
        assert!((benchmark.t_distribution_cdf(3.0, 10.0) - 0.993_328).abs() < 1e-5);
        assert!((benchmark.t_distribution_cdf(0.0, 10.0) - 0.5).abs() < 1e-12);
    }

    /// F48: the ANOVA p-value must come from the F distribution.
    #[test]
    fn anova_uses_the_f_distribution() {
        let benchmark = test_benchmark();

        let separated = vec![
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
            vec![5.0, 5.1, 4.9, 5.05, 4.95],
            vec![9.0, 9.1, 8.9, 9.05, 8.95],
        ];
        let result = benchmark.perform_anova(&separated);
        assert!(result.f_statistic > 100.0);
        assert!(result.p_value < 1e-9, "p = {}", result.p_value);
        assert_eq!(result.df_between, 2);
        assert_eq!(result.df_within, 12);

        let identical = vec![
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
        ];
        let flat = benchmark.perform_anova(&identical);
        assert!(flat.p_value > 0.9, "p = {}", flat.p_value);

        // Guards.
        assert_eq!(benchmark.perform_anova(&[]).p_value, 1.0);
        assert_eq!(benchmark.perform_anova(&[vec![1.0, 2.0]]).p_value, 1.0);
        assert_eq!(
            benchmark.perform_anova(&[vec![1.0], vec![2.0]]).p_value,
            1.0
        );
    }

    /// F49: the critical value must follow the requested confidence level.
    #[test]
    fn confidence_intervals_honour_the_confidence_level() {
        let benchmark = test_benchmark();
        let values = [10.0, 12.0, 9.0, 11.0, 13.0, 8.0, 10.5, 11.5];

        let ci95 = benchmark.calculate_confidence_interval(&values, 0.95);
        let ci99 = benchmark.calculate_confidence_interval(&values, 0.99);
        assert_eq!(ci95.confidence_level, 0.95);
        assert!(ci99.upper - ci99.lower > ci95.upper - ci95.lower);
        assert!(ci95.lower < 10.6 && ci95.upper > 10.6);

        // With n = 8 the 95% t critical value is 2.365, clearly wider than the
        // hardcoded 1.96 the previous implementation always used.
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let half_width = ci95.upper - mean;
        let normal_half_width = 1.96
            * (1.767_766_952_966_369 / (values.len() as f64).sqrt() * 0.0 + {
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                variance.sqrt() / (values.len() as f64).sqrt()
            });
        assert!(half_width > normal_half_width);

        // Degenerate inputs.
        let empty = benchmark.calculate_confidence_interval(&[], 0.95);
        assert_eq!(empty.lower, 0.0);
        assert_eq!(empty.upper, 0.0);
        let single = benchmark.calculate_confidence_interval(&[4.0], 0.95);
        assert_eq!(single.lower, 4.0);
        assert_eq!(single.upper, 4.0);
    }

    #[test]
    fn cohens_d_guards_degenerate_samples() {
        let benchmark = test_benchmark();
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert!((benchmark.calculate_cohens_d(&a, &b) + 3.098_386_676_965_933).abs() < 1e-9);

        assert_eq!(benchmark.calculate_cohens_d(&[1.0], &b), 0.0);
        assert_eq!(benchmark.calculate_cohens_d(&[], &b), 0.0);
        assert_eq!(
            benchmark.calculate_cohens_d(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]),
            0.0
        );
    }

    /// F45: parsing must be strict - a missing field is an error, never a
    /// silently substituted default.
    #[test]
    fn python_result_parsing_is_strict() {
        let benchmark = test_benchmark();
        let identifier = OptimizerIdentifier {
            framework: Framework::PyTorch,
            name: "Adam".to_string(),
            version: Some("2.3.0".to_string()),
        };

        let complete = serde_json::json!({
            "total_runs": 3,
            "successful_runs": 2,
            "convergence_times_secs": [0.10, 0.12, 0.11],
            "final_values": [1e-8, 2e-8, 1.5e-8],
            "iterations": [120.0, 140.0, 130.0],
            "gradient_norms": [1e-7, 2e-7, 1.5e-7],
        });
        let summary = benchmark
            .parse_python_results(identifier.clone(), &complete)
            .expect("complete results parse");
        assert_eq!(summary.total_runs, 3);
        assert_eq!(summary.successful_runs, 2);
        assert!((summary.success_rate - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(summary.run_convergence_times_secs.len(), 3);
        assert!((summary.mean_iterations - 130.0).abs() < 1e-9);
        // Memory is optional and defaults to "not measured" (zero), never 1 MB.
        assert_eq!(summary.memory_stats.peak_memory_bytes, 0);
        assert!(summary.gpu_utilization.is_none());

        // Every required field must be present.
        for field in [
            "total_runs",
            "successful_runs",
            "convergence_times_secs",
            "final_values",
            "iterations",
            "gradient_norms",
        ] {
            let mut partial = complete.clone();
            if let Some(object) = partial.as_object_mut() {
                object.remove(field);
            }
            assert!(
                benchmark
                    .parse_python_results(identifier.clone(), &partial)
                    .is_err(),
                "missing '{}' must be an error",
                field
            );
        }

        // Length mismatches are rejected too.
        let mut mismatched = complete.clone();
        if let Some(object) = mismatched.as_object_mut() {
            object.insert("iterations".to_string(), serde_json::json!([1.0]));
        }
        assert!(benchmark
            .parse_python_results(identifier, &mismatched)
            .is_err());
    }

    /// F43/F44: a missing interpreter must skip the comparison explicitly.
    #[test]
    fn missing_python_interpreter_is_skipped_not_faked() {
        let config = CrossFrameworkConfig {
            temp_dir: std::env::temp_dir()
                .join("optirs_cross_framework_missing_python")
                .to_string_lossy()
                .into_owned(),
            python_path: "optirs-definitely-not-a-real-python-interpreter".to_string(),
            ..Default::default()
        };
        let benchmark = CrossFrameworkBenchmark::<f64>::new(config).expect("temp dir is creatable");

        let mut functions = CrossFrameworkBenchmark::<f64>::new(CrossFrameworkConfig {
            temp_dir: std::env::temp_dir()
                .join("optirs_cross_framework_missing_python_fn")
                .to_string_lossy()
                .into_owned(),
            ..Default::default()
        })
        .expect("temp dir is creatable");
        functions.add_standard_test_functions();
        let test_function = functions
            .test_functions
            .first()
            .expect("standard functions were added");

        let outcome = benchmark
            .benchmark_pytorch_optimizers(test_function, 10, 1)
            .expect("a missing interpreter is not a hard error");
        match outcome {
            ExternalFrameworkOutcome::Skipped(skip) => {
                assert_eq!(skip.framework, Framework::PyTorch);
                assert!(skip.reason.contains("not found"), "reason: {}", skip.reason);
            }
            ExternalFrameworkOutcome::Completed(results) => {
                panic!("expected a skip, got {} results", results.len())
            }
        }
    }
}
