//! `optirs-bench` CLI: run the built-in optimizer benchmark suite, compare
//! optimizer variants for regressions, and render reports.
//!
//! Every subcommand below performs a real run of
//! [`optirs_bench::OptimizerBenchmark`] against the crate's standard test
//! functions (Quadratic, Rosenbrock, Sphere) -- there is no fabricated or
//! placeholder output.

use optirs_bench::report_templates::{save_report, ReportFormat, ReportTemplate};
use optirs_bench::{BenchmarkReport, OptimizerBenchmark};
use scirs2_core::ndarray::Array1;
use std::path::PathBuf;
use std::process::ExitCode;

const BASELINE_NAME: &str = "gradient-descent";
const CANDIDATE_NAME: &str = "gradient-descent-decay";
const DEFAULT_MAX_ITERATIONS: usize = 500;
const DEFAULT_TOLERANCE: f64 = 1e-6;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        return ExitCode::FAILURE;
    }

    let result = match args[1].as_str() {
        "benchmark" => run_benchmark(&args[2..]),
        "analyze" => run_analyze(&args[2..]),
        "report" => run_report(&args[2..]),
        "-h" | "--help" | "help" => {
            print_usage(&args[0]);
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("Unknown command: {other}");
            print_usage(&args[0]);
            return ExitCode::FAILURE;
        }
    };

    match result {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage(program: &str) {
    println!("Usage: {program} <command> [options]");
    println!();
    println!("Commands:");
    println!("  benchmark [--iterations N] [--tolerance T]");
    println!("      Run the built-in optimizer benchmark suite and print a summary.");
    println!();
    println!("  analyze [--max-regression PCT] [--iterations N] [--tolerance T]");
    println!("      Compare '{CANDIDATE_NAME}' against the '{BASELINE_NAME}' baseline and");
    println!("      exit with a non-zero status if its success rate regresses by more than");
    println!("      PCT percentage points (default: 10.0).");
    println!();
    println!("  report [--format md|txt|csv] [--output PATH] [--iterations N] [--tolerance T]");
    println!("      Render a full benchmark report to stdout, or to PATH if given.");
}

/// A parsed `--iterations`/`--tolerance` pair shared by all subcommands.
struct RunOptions {
    max_iterations: usize,
    tolerance: f64,
}

fn parse_run_options(args: &[String]) -> Result<RunOptions, String> {
    let mut max_iterations = DEFAULT_MAX_ITERATIONS;
    let mut tolerance = DEFAULT_TOLERANCE;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                let value = args.get(i + 1).ok_or("--iterations requires a value")?;
                max_iterations = value
                    .parse()
                    .map_err(|_| format!("invalid --iterations value: {value}"))?;
                i += 2;
            }
            "--tolerance" => {
                let value = args.get(i + 1).ok_or("--tolerance requires a value")?;
                tolerance = value
                    .parse()
                    .map_err(|_| format!("invalid --tolerance value: {value}"))?;
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    Ok(RunOptions {
        max_iterations,
        tolerance,
    })
}

fn find_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|idx| args.get(idx + 1))
        .map(|s| s.as_str())
}

/// Plain gradient descent: `x_{t+1} = x_t - lr * grad`.
fn gradient_descent_step(lr: f64) -> impl FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64> {
    move |x, grad| x - &(grad * lr)
}

/// Gradient descent with an inverse-time learning-rate decay schedule.
/// Dimension-agnostic (the only state carried across steps is a scalar
/// iteration counter), so it is safe to reuse across test functions of
/// different dimensionality within a single `run_benchmark` call.
fn gradient_descent_decay_step(
    initial_lr: f64,
    decay: f64,
) -> impl FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64> {
    let mut step: u32 = 0;
    move |x, grad| {
        step += 1;
        let lr = initial_lr / (1.0 + decay * step as f64);
        x - &(grad * lr)
    }
}

/// Run both baseline and candidate optimizers through the standard test
/// suite and return the resulting report.
fn run_suite(options: &RunOptions) -> Result<BenchmarkReport<f64>, String> {
    let mut benchmark = OptimizerBenchmark::<f64>::new();
    benchmark.add_standard_test_functions();

    benchmark
        .run_benchmark(
            BASELINE_NAME.to_string(),
            gradient_descent_step(0.01),
            options.max_iterations,
            options.tolerance,
        )
        .map_err(|e| format!("failed running '{BASELINE_NAME}': {e}"))?;

    benchmark
        .run_benchmark(
            CANDIDATE_NAME.to_string(),
            gradient_descent_decay_step(0.05, 0.01),
            options.max_iterations,
            options.tolerance,
        )
        .map_err(|e| format!("failed running '{CANDIDATE_NAME}': {e}"))?;

    Ok(benchmark.generate_report())
}

fn run_benchmark(args: &[String]) -> Result<ExitCode, String> {
    let options = parse_run_options(args)?;
    let report = run_suite(&options)?;

    println!(
        "Ran {} benchmark result(s) across {} optimizer(s).",
        report.total_tests,
        report.optimizer_performance.len()
    );
    for (name, perf) in &report.optimizer_performance {
        let success_rate = report.get_success_rate(name).unwrap_or(0.0) * 100.0;
        println!(
            "  {name}: {}/{} converged ({:.1}%), avg iterations {:.1}, avg final error {:.3e}",
            perf.successful_runs,
            perf.total_runs,
            success_rate,
            perf.average_iterations,
            perf.average_final_error
        );
    }

    Ok(ExitCode::SUCCESS)
}

fn run_analyze(args: &[String]) -> Result<ExitCode, String> {
    let options = parse_run_options(args)?;
    let max_regression_points: f64 = find_flag_value(args, "--max-regression")
        .map(|v| {
            v.parse()
                .map_err(|_| format!("invalid --max-regression value: {v}"))
        })
        .transpose()?
        .unwrap_or(10.0);

    let report = run_suite(&options)?;

    let comparison = report
        .compare_optimizers(CANDIDATE_NAME, BASELINE_NAME)
        .ok_or_else(|| format!("missing results for '{BASELINE_NAME}' or '{CANDIDATE_NAME}'"))?;

    let success_rate_diff_points = comparison.success_rate_diff * 100.0;

    println!(
        "Comparing '{}' against baseline '{}':",
        comparison.optimizer1, comparison.optimizer2
    );
    println!(
        "  success rate diff: {:+.1} percentage points",
        success_rate_diff_points
    );
    println!(
        "  avg iterations diff: {:+.1}",
        comparison.avg_iterations_diff
    );
    println!("  avg final error diff: {:+.3e}", comparison.avg_error_diff);

    if success_rate_diff_points < -max_regression_points {
        println!(
            "REGRESSION: '{CANDIDATE_NAME}' success rate is {:.1} points below baseline \
             (threshold: -{max_regression_points:.1})",
            success_rate_diff_points.abs()
        );
        return Ok(ExitCode::FAILURE);
    }

    println!("OK: no regression beyond the configured threshold.");
    Ok(ExitCode::SUCCESS)
}

fn run_report(args: &[String]) -> Result<ExitCode, String> {
    let options = parse_run_options(args)?;
    let format = match find_flag_value(args, "--format").unwrap_or("md") {
        "md" | "markdown" => ReportFormat::Markdown,
        "txt" | "text" | "plain" => ReportFormat::PlainText,
        "csv" => ReportFormat::Csv,
        other => {
            return Err(format!(
                "unknown --format value: {other} (expected md|txt|csv)"
            ))
        }
    };
    let output_path = find_flag_value(args, "--output").map(PathBuf::from);

    let report = run_suite(&options)?;
    let comparison = report.compare_optimizers(CANDIDATE_NAME, BASELINE_NAME);

    let rendered = ReportTemplate::new()
        .with_title("OptiRS Bench Report")
        .render(&report, comparison.as_ref(), format);

    match output_path {
        Some(path) => {
            save_report(&path, &rendered).map_err(|e| format!("failed to write report: {e}"))?;
            println!("Report written to {}", path.display());
        }
        None => println!("{rendered}"),
    }

    Ok(ExitCode::SUCCESS)
}
