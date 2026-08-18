//! Performance regression detector binary for OptiRS benchmarking suite.
//!
//! Compares a baseline and a current metrics file and reports tests whose
//! mean value grew by more than a relative threshold. Each line has the form
//! `test_name: value[, value, ...]`; when several samples are given the mean
//! is used, so the comparison is over averages rather than a single reading.
//! The threshold is relative (fractional) and configurable; values are treated
//! as unit-agnostic magnitudes (higher == slower/worse).

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <baseline_file> <current_file> [threshold_fraction]",
            args[0]
        );
        eprintln!("  threshold_fraction defaults to 0.05 (5% relative growth).");
        std::process::exit(1);
    }

    let baseline_file = &args[1];
    let current_file = &args[2];
    let threshold = match args.get(3) {
        Some(raw) => match raw.parse::<f64>() {
            Ok(t) if t.is_finite() && t >= 0.0 => t,
            _ => {
                eprintln!("Invalid threshold '{raw}': expected a non-negative number.");
                std::process::exit(1);
            }
        },
        None => 0.05,
    };

    let regressions = detect_regressions(baseline_file, current_file, threshold)?;

    if regressions.is_empty() {
        println!(
            "No performance regressions detected (threshold {:.1}%).",
            threshold * 100.0
        );
    } else {
        println!(
            "Performance regressions detected (threshold {:.1}%):",
            threshold * 100.0
        );
        for regression in regressions {
            println!("- {regression}");
        }
    }

    Ok(())
}

fn detect_regressions(
    baseline_file: &str,
    current_file: &str,
    threshold: f64,
) -> io::Result<Vec<String>> {
    let mut regressions = Vec::new();

    let baseline_metrics = load_metrics(baseline_file)?;
    let current_metrics = load_metrics(current_file)?;

    let mut names: Vec<&String> = baseline_metrics.keys().collect();
    names.sort();

    for test_name in names {
        let baseline_mean = match baseline_metrics.get(test_name) {
            Some(v) => *v,
            None => continue,
        };
        if let Some(current_mean) = current_metrics.get(test_name) {
            if baseline_mean == 0.0 {
                continue;
            }
            let regression_ratio = (*current_mean - baseline_mean) / baseline_mean;

            if regression_ratio > threshold {
                regressions.push(format!(
                    "{}: {:.2}% slower ({:.4} -> {:.4})",
                    test_name,
                    regression_ratio * 100.0,
                    baseline_mean,
                    current_mean
                ));
            }
        }
    }

    Ok(regressions)
}

/// Load `test_name: value[, value, ...]` lines, storing the mean of the
/// samples for each test. Lines without a parseable value are skipped.
fn load_metrics(file_path: &str) -> io::Result<HashMap<String, f64>> {
    let mut metrics = HashMap::new();
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if let Some((test_name, values_str)) = line.split_once(':') {
            let samples: Vec<f64> = values_str
                .split([',', ' ', '\t'])
                .filter(|token| !token.trim().is_empty())
                .filter_map(|token| token.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite())
                .collect();
            if !samples.is_empty() {
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                metrics.insert(test_name.trim().to_string(), mean);
            }
        }
    }

    Ok(metrics)
}
