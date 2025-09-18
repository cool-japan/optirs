// Performance regression testing binary for CI/CD integration
//
// This binary provides command-line interface for running comprehensive
// performance regression tests and generating CI-compatible reports.

use clap::{Arg, ArgMatches, Command};
use scirs2_core::ndarray_ext::Array1;
use scirs2_optim::benchmarking::regression_tester::{
    CiReportFormat, RegressionConfig, RegressionTester,
};
use scirs2_optim::benchmarking::{BenchmarkResult, OptimizerBenchmark};
use scirs2_optim::error::{OptimError, Result};
use std::fs;
use std::path::PathBuf;

#[allow(dead_code)]
fn main() -> Result<()> {
    let matches = Command::new("performance_regression_tester")
        .version("0.1.0")
        .author("SCIRS2 Team")
        .about("Performance regression testing for CI/CD integration")
        .arg(
            Arg::new("baseline-dir")
                .long("baseline-dir")
                .value_name("DIR")
                .help("Directory containing performance baselines")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("FILE")
                .help("Output file for regression report")
                .required(true),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .value_name("FORMAT")
                .help("Output format (json, junit-xml, markdown, github-actions)")
                .default_value("json"),
        )
        .arg(
            Arg::new("degradation-threshold")
                .long("degradation-threshold")
                .value_name("PERCENT")
                .help("Performance degradation threshold percentage")
                .default_value("5.0"),
        )
        .arg(
            Arg::new("memory-threshold")
                .long("memory-threshold")
                .value_name("PERCENT")
                .help("Memory regression threshold percentage")
                .default_value("10.0"),
        )
        .arg(
            Arg::new("significance-threshold")
                .long("significance-threshold")
                .value_name("ALPHA")
                .help("Statistical significance threshold")
                .default_value("0.05"),
        )
        .arg(
            Arg::new("update-baseline")
                .long("update-baseline")
                .help("Update baseline after successful tests")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Parse configuration from command line arguments
    let config = parse_config(&matches)?;

    if matches.get_flag("verbose") {
        println!(
            "Starting performance regression testing with config: {:#?}",
            config
        );
    }

    // Create regression tester
    let mut tester = RegressionTester::<f64>::new(config.clone())?;

    // Run regression tests for different optimizers
    if matches.get_flag("verbose") {
        println!("Running comprehensive optimizer regression tests...");
    }

    let mut regression_results = Vec::new();

    // Test different optimizers
    let optimizers = vec![
        ("SGD", create_sgd_step_function()),
        ("Adam", create_adam_step_function()),
        ("AdaGrad", create_adagrad_step_function()),
        ("RMSprop", create_rmsprop_step_function()),
        ("LAMB", create_lamb_step_function()),
    ];

    for (name, mut step_fn) in optimizers {
        if matches.get_flag("verbose") {
            println!("  Testing {} optimizer...", name);
        }

        // Run regression test for this optimizer
        let test_result = tester.run_regression_test("comprehensive_benchmark", name, || {
            let mut benchmark = OptimizerBenchmark::new();
            benchmark.add_standard_test_functions();
            let results = benchmark.run_benchmark(
                name.to_string(),
                &mut step_fn,
                1000, // max iterations
                1e-6, // tolerance
            )?;
            Ok(results
                .into_iter()
                .next()
                .unwrap_or_else(|| BenchmarkResult {
                    optimizername: name.to_string(),
                    function_name: "comprehensive_benchmark".to_string(),
                    converged: false,
                    convergence_step: None,
                    final_function_value: f64::INFINITY,
                    final_gradient_norm: f64::INFINITY,
                    final_error: f64::INFINITY,
                    iterations_taken: 0,
                    elapsed_time: std::time::Duration::from_secs(0),
                    function_evaluations: 0,
                    function_value_history: Vec::new(),
                    gradient_norm_history: Vec::new(),
                }))
        })?;

        regression_results.push(test_result);
    }

    // Generate CI report
    let outputpath = PathBuf::from(matches.get_one::<String>("output").unwrap());

    if matches.get_flag("verbose") {
        println!("Generating report to: {}", outputpath.display());
    }

    let cireport = tester.generate_ci_report(&regression_results)?;
    fs::write(&outputpath, cireport)?;

    // Check for critical regressions
    let has_regressions = regression_results.iter().any(|r| r.has_regressions());

    if has_regressions {
        eprintln!("Performance regressions detected!");
        std::process::exit(1);
    }

    if matches.get_flag("verbose") {
        println!("Performance regression testing completed successfully!");
    }

    Ok(())
}

#[allow(dead_code)]
fn parse_config(matches: &ArgMatches) -> Result<RegressionConfig> {
    let baseline_dir = PathBuf::from(matches.get_one::<String>("baseline-dir").unwrap());

    let degradation_threshold: f64 = matches
        .get_one::<String>("degradation-threshold")
        .unwrap()
        .parse()
        .map_err(|_| OptimError::InvalidConfig("Invalid degradation threshold".to_string()))?;

    let memory_threshold: f64 = matches
        .get_one::<String>("memory-threshold")
        .unwrap()
        .parse()
        .map_err(|_| OptimError::InvalidConfig("Invalid memory threshold".to_string()))?;

    let significance_threshold: f64 = matches
        .get_one::<String>("significance-threshold")
        .unwrap()
        .parse()
        .map_err(|_| OptimError::InvalidConfig("Invalid significance threshold".to_string()))?;

    Ok(RegressionConfig {
        baseline_dir,
        max_history_length: 1000,
        min_baseline_samples: 10,
        significance_threshold,
        degradation_threshold,
        memory_threshold,
        enable_ci_integration: true,
        enable_alerts: true,
        outlier_sensitivity: 2.0,
        detection_algorithms: vec![
            "statistical_test".to_string(),
            "sliding_window".to_string(),
            "change_point".to_string(),
        ],
        ci_report_format: parsereport_format(matches.get_one::<String>("format").unwrap())
            .unwrap_or(CiReportFormat::Json),
    })
}

#[allow(dead_code)]
fn parsereport_format(_formatstr: &str) -> Result<CiReportFormat> {
    match _formatstr.to_lowercase().as_str() {
        "json" => Ok(CiReportFormat::Json),
        "junit-xml" => Ok(CiReportFormat::JunitXml),
        "markdown" => Ok(CiReportFormat::Markdown),
        "github-actions" => Ok(CiReportFormat::GitHubActions),
        _ => Err(OptimError::InvalidConfig(format!(
            "Unknown format: {}",
            _formatstr
        ))),
    }
}

// Function removed - regression testing is now handled directly by RegressionTester

// Helper functions to create optimizer step functions
#[allow(dead_code)]
fn create_sgd_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let learning_rate = 0.01;
    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| x - &(grad * learning_rate))
}

#[allow(dead_code)]
fn create_adam_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut m: Array1<f64> = Array1::zeros(0);
    let mut v: Array1<f64> = Array1::zeros(0);
    let mut t = 0;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let learning_rate = 0.001;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if m.len() != x.len() {
            m = Array1::zeros(x.len());
            v = Array1::zeros(x.len());
        }

        t += 1;

        // Update biased first moment estimate
        m = &m * beta1 + &(grad * (1.0 - beta1));

        // Update biased second raw moment estimate
        let grad_squared = grad.mapv(|g| g * g);
        v = &v * beta2 + &(&grad_squared * (1.0 - beta2));

        // Compute bias-corrected first moment estimate
        let m_hat = &m / (1.0 - beta1.powi(t));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &v / (1.0 - beta2.powi(t));

        // Update parameters
        let denominator = v_hat.mapv(|v: f64| v.sqrt() + epsilon);
        x - &(&m_hat / &denominator * learning_rate)
    })
}

#[allow(dead_code)]
fn create_adagrad_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut g: Array1<f64> = Array1::zeros(0);
    let learning_rate = 0.01;
    let epsilon = 1e-8;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if g.len() != x.len() {
            g = Array1::<f64>::zeros(x.len());
        }

        // Accumulate squared gradients
        let grad_squared = grad.mapv(|g| g * g);
        g = &g + &grad_squared;

        // Update parameters
        let denominator = g.mapv(|g: f64| g.sqrt() + epsilon);
        x - &(grad / &denominator * learning_rate)
    })
}

#[allow(dead_code)]
fn create_rmsprop_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut s: Array1<f64> = Array1::zeros(0);
    let learning_rate = 0.001;
    let epsilon = 1e-8;
    let decay_rate = 0.9;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if s.len() != x.len() {
            s = Array1::<f64>::zeros(x.len());
        }

        // Update moving average of squared gradients
        let grad_squared = grad.mapv(|g| g * g);
        s = &s * decay_rate + &(&grad_squared * (1.0 - decay_rate));

        // Update parameters
        let denominator = s.mapv(|s: f64| s.sqrt() + epsilon);
        x - &(grad / &denominator * learning_rate)
    })
}

#[allow(dead_code)]
fn create_lamb_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut m: Array1<f64> = Array1::zeros(0);
    let mut v: Array1<f64> = Array1::zeros(0);
    let mut t = 0;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-6;
    let learning_rate = 0.001;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if m.len() != x.len() {
            m = Array1::<f64>::zeros(x.len());
            v = Array1::<f64>::zeros(x.len());
        }

        t += 1;

        // Update biased first moment estimate
        m = &m * beta1 + &(grad * (1.0 - beta1));

        // Update biased second raw moment estimate
        let grad_squared = grad.mapv(|g| g * g);
        v = &v * beta2 + &(&grad_squared * (1.0 - beta2));

        // Compute bias-corrected first moment estimate
        let m_hat = &m / (1.0 - beta1.powi(t));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &v / (1.0 - beta2.powi(t));

        // Compute update
        let denominator = v_hat.mapv(|v: f64| v.sqrt() + epsilon);
        let update = &m_hat / &denominator;

        // Compute trust ratio (simplified)
        let param_norm = x.mapv(|p: f64| p * p).sum().sqrt();
        let update_norm = update.mapv(|u: f64| u * u).sum().sqrt();

        let trust_ratio = if update_norm > 0.0 {
            (param_norm / update_norm).min(1.0)
        } else {
            1.0
        };

        // Apply update with trust ratio
        x - &(&update * (learning_rate * trust_ratio))
    })
}

// Custom structs removed - using RegressionTester's built-in types and CI report generation
