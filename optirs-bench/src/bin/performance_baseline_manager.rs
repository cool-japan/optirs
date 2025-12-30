// Performance Baseline Manager Binary
//
// This binary provides a command-line interface for managing performance baselines,
// including creating, updating, and validating baseline performance data.

use chrono::{DateTime, Utc};
use clap::{Arg, ArgMatches, Command};
use optirs_core::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
struct BaselineMetrics {
    metrics: HashMap<String, MetricValue>,
    metadata: BaselineMetadata,
    statistical_summary: StatisticalSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricValue {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    samples: Vec<f64>,
    confidence_interval: (f64, f64),
}

#[derive(Debug, Serialize, Deserialize)]
struct BaselineMetadata {
    version: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    commit_hash: String,
    branch: String,
    features: String,
    sample_count: usize,
    baseline_id: String,
    platform_info: PlatformInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlatformInfo {
    os: String,
    arch: String,
    cpu_cores: usize,
    rust_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StatisticalSummary {
    total_benchmarks: usize,
    stable_benchmarks: usize,
    variable_benchmarks: usize,
    overall_confidence: f64,
    quality_score: f64,
}

#[allow(dead_code)]
fn main() {
    let matches = Command::new("performance-baseline-manager")
        .version("0.1.0")
        .author("SciRS2 Development Team")
        .about("Performance baseline management for continuous integration")
        .subcommand(
            Command::new("create")
                .about("Create a new performance baseline")
                .arg(
                    Arg::new("results-file")
                        .long("results-file")
                        .value_name("FILE")
                        .help("Path to performance results JSON file")
                        .required(true),
                )
                .arg(
                    Arg::new("baseline-dir")
                        .long("baseline-dir")
                        .value_name("DIR")
                        .help("Directory to store baseline files")
                        .required(true),
                )
                .arg(
                    Arg::new("features")
                        .long("features")
                        .value_name("STRING")
                        .help("Feature set being tested")
                        .required(true),
                )
                .arg(
                    Arg::new("commit-hash")
                        .long("commit-hash")
                        .value_name("HASH")
                        .help("Git commit hash")
                        .required(true),
                )
                .arg(
                    Arg::new("branch")
                        .long("branch")
                        .value_name("BRANCH")
                        .help("Git branch name")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("update")
                .about("Update an existing performance baseline")
                .arg(
                    Arg::new("results-file")
                        .long("results-file")
                        .value_name("FILE")
                        .help("Path to performance results JSON file")
                        .required(true),
                )
                .arg(
                    Arg::new("baseline-dir")
                        .long("baseline-dir")
                        .value_name("DIR")
                        .help("Directory containing baseline files")
                        .required(true),
                )
                .arg(
                    Arg::new("features")
                        .long("features")
                        .value_name("STRING")
                        .help("Feature set being tested")
                        .required(true),
                )
                .arg(
                    Arg::new("commit-hash")
                        .long("commit-hash")
                        .value_name("HASH")
                        .help("Git commit hash")
                        .required(true),
                )
                .arg(
                    Arg::new("branch")
                        .long("branch")
                        .value_name("BRANCH")
                        .help("Git branch name")
                        .required(true),
                )
                .arg(
                    Arg::new("merge-strategy")
                        .long("merge-strategy")
                        .value_name("STRATEGY")
                        .help("Strategy for merging with existing baseline")
                        .value_parser(["replace", "merge", "weighted"])
                        .default_value("weighted"),
                ),
        )
        .subcommand(
            Command::new("validate")
                .about("Validate a performance baseline")
                .arg(
                    Arg::new("baseline-dir")
                        .long("baseline-dir")
                        .value_name("DIR")
                        .help("Directory containing baseline files")
                        .required(true),
                )
                .arg(
                    Arg::new("features")
                        .long("features")
                        .value_name("STRING")
                        .help("Feature set to validate")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("list")
                .about("List available performance baselines")
                .arg(
                    Arg::new("baseline-dir")
                        .long("baseline-dir")
                        .value_name("DIR")
                        .help("Directory containing baseline files")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("info")
                .about("Show information about a specific baseline")
                .arg(
                    Arg::new("baseline-dir")
                        .long("baseline-dir")
                        .value_name("DIR")
                        .help("Directory containing baseline files")
                        .required(true),
                )
                .arg(
                    Arg::new("features")
                        .long("features")
                        .value_name("STRING")
                        .help("Feature set to show info for")
                        .required(true),
                ),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue)
                .global(true),
        )
        .get_matches();

    let verbose = matches.get_flag("verbose");

    match matches.subcommand() {
        Some(("create", sub_m)) => {
            if let Err(e) = handle_create_baseline(sub_m, verbose) {
                eprintln!("❌ Error creating baseline: {}", e);
                process::exit(1);
            }
        }
        Some(("update", sub_m)) => {
            if let Err(e) = handle_update_baseline(sub_m, verbose) {
                eprintln!("❌ Error updating baseline: {}", e);
                process::exit(1);
            }
        }
        Some(("validate", sub_m)) => {
            if let Err(e) = handle_validate_baseline(sub_m, verbose) {
                eprintln!("❌ Error validating baseline: {}", e);
                process::exit(1);
            }
        }
        Some(("list", sub_m)) => {
            if let Err(e) = handle_list_baselines(sub_m, verbose) {
                eprintln!("❌ Error listing baselines: {}", e);
                process::exit(1);
            }
        }
        Some(("info", sub_m)) => {
            if let Err(e) = handle_show_baseline_info(sub_m, verbose) {
                eprintln!("❌ Error showing baseline info: {}", e);
                process::exit(1);
            }
        }
        _ => {
            eprintln!("No subcommand provided. Use --help for usage information.");
            process::exit(1);
        }
    }
}

#[allow(dead_code)]
fn handle_create_baseline(matches: &ArgMatches, verbose: bool) -> Result<()> {
    let results_file = matches.get_one::<String>("results-file").unwrap();
    let baseline_dir = matches.get_one::<String>("baseline-dir").unwrap();
    let features = matches.get_one::<String>("features").unwrap();
    let commit_hash = matches.get_one::<String>("commit-hash").unwrap();
    let branch = matches.get_one::<String>("branch").unwrap();

    if verbose {
        println!("🏗️  Creating new performance baseline:");
        println!("  Results File: {}", results_file);
        println!("  Baseline Directory: {}", baseline_dir);
        println!("  Features: {}", features);
        println!("  Commit: {}", commit_hash);
        println!("  Branch: {}", branch);
    }

    // Load performance results
    let results = loadperformance_results(results_file)?;

    // Create baseline directory if it doesn't exist
    fs::create_dir_all(baseline_dir).map_err(|e| {
        OptimError::ResourceError(format!("Failed to create baseline directory: {}", e))
    })?;

    // Generate baseline metrics
    let baseline = create_baseline_from_results(&results, features, commit_hash, branch)?;

    // Save baseline
    let baselinepath = PathBuf::from(baseline_dir).join(format!("baseline_{}.json", features));
    save_baseline(&baseline, &baselinepath)?;

    if verbose {
        println!("✅ Baseline created successfully");
        println!("📄 Saved to: {}", baselinepath.display());
        println!("📊 Summary:");
        println!(
            "  Total Benchmarks: {}",
            baseline.statistical_summary.total_benchmarks
        );
        println!("  Sample Count: {}", baseline.metadata.sample_count);
        println!(
            "  Quality Score: {:.2}",
            baseline.statistical_summary.quality_score
        );
    } else {
        println!("✅ Baseline created: {}", baselinepath.display());
    }

    Ok(())
}

#[allow(dead_code)]
fn handle_update_baseline(matches: &ArgMatches, verbose: bool) -> Result<()> {
    let results_file = matches.get_one::<String>("results-file").unwrap();
    let baseline_dir = matches.get_one::<String>("baseline-dir").unwrap();
    let features = matches.get_one::<String>("features").unwrap();
    let commit_hash = matches.get_one::<String>("commit-hash").unwrap();
    let branch = matches.get_one::<String>("branch").unwrap();
    let merge_strategy = matches.get_one::<String>("merge-strategy").unwrap();

    if verbose {
        println!("🔄 Updating performance baseline:");
        println!("  Results File: {}", results_file);
        println!("  Baseline Directory: {}", baseline_dir);
        println!("  Features: {}", features);
        println!("  Commit: {}", commit_hash);
        println!("  Branch: {}", branch);
        println!("  Merge Strategy: {}", merge_strategy);
    }

    let baselinepath = PathBuf::from(baseline_dir).join(format!("baseline_{}.json", features));

    // Load existing baseline if it exists
    let existing_baseline = if baselinepath.exists() {
        Some(load_baseline(&baselinepath)?)
    } else {
        if verbose {
            println!("⚠️  No existing baseline found, creating new one");
        }
        None
    };

    // Load new performance results
    let results = loadperformance_results(results_file)?;

    // Create updated baseline
    let updated_baseline = if let Some(existing) = existing_baseline {
        merge_baseline_with_results(
            &existing,
            &results,
            features,
            commit_hash,
            branch,
            merge_strategy,
        )?
    } else {
        create_baseline_from_results(&results, features, commit_hash, branch)?
    };

    // Save updated baseline
    save_baseline(&updated_baseline, &baselinepath)?;

    if verbose {
        println!("✅ Baseline updated successfully");
        println!("📄 Saved to: {}", baselinepath.display());
        println!("📊 Summary:");
        println!(
            "  Total Benchmarks: {}",
            updated_baseline.statistical_summary.total_benchmarks
        );
        println!("  Sample Count: {}", updated_baseline.metadata.sample_count);
        println!(
            "  Quality Score: {:.2}",
            updated_baseline.statistical_summary.quality_score
        );
    } else {
        println!("✅ Baseline updated: {}", baselinepath.display());
    }

    Ok(())
}

#[allow(dead_code)]
fn handle_validate_baseline(matches: &ArgMatches, verbose: bool) -> Result<()> {
    let baseline_dir = matches.get_one::<String>("baseline-dir").unwrap();
    let features = matches.get_one::<String>("features").unwrap();

    let baselinepath = PathBuf::from(baseline_dir).join(format!("baseline_{}.json", features));

    if !baselinepath.exists() {
        return Err(OptimError::ConfigurationError(format!(
            "Baseline file not found: {}",
            baselinepath.display()
        )));
    }

    if verbose {
        println!("🔍 Validating baseline: {}", baselinepath.display());
    }

    let baseline = load_baseline(&baselinepath)?;
    let validation_result = validate_baseline(&baseline)?;

    if verbose {
        println!("📊 Baseline Validation Results:");
        println!("  Baseline ID: {}", baseline.metadata.baseline_id);
        println!("  Created: {}", baseline.metadata.created_at);
        println!("  Updated: {}", baseline.metadata.updated_at);
        println!("  Branch: {}", baseline.metadata.branch);
        println!("  Commit: {}", baseline.metadata.commit_hash);
        println!("  Features: {}", baseline.metadata.features);
        println!(
            "  Platform: {} {}",
            baseline.metadata.platform_info.os, baseline.metadata.platform_info.arch
        );
        println!(
            "  Total Benchmarks: {}",
            baseline.statistical_summary.total_benchmarks
        );
        println!("  Sample Count: {}", baseline.metadata.sample_count);
        println!(
            "  Quality Score: {:.2}/100",
            baseline.statistical_summary.quality_score
        );
        println!(
            "  Overall Confidence: {:.2}%",
            baseline.statistical_summary.overall_confidence * 100.0
        );
        println!(
            "  Validation Status: {}",
            if validation_result {
                "✅ VALID"
            } else {
                "❌ INVALID"
            }
        );
    }

    if validation_result {
        println!("✅ Baseline is valid");
    } else {
        println!("❌ Baseline validation failed");
        process::exit(1);
    }

    Ok(())
}

#[allow(dead_code)]
fn handle_list_baselines(matches: &ArgMatches, verbose: bool) -> Result<()> {
    let baseline_dir = matches.get_one::<String>("baseline-dir").unwrap();

    if verbose {
        println!("📋 Listing baselines in: {}", baseline_dir);
    }

    let baseline_files = find_baseline_files(baseline_dir)?;

    if baseline_files.is_empty() {
        println!("No baseline files found in {}", baseline_dir);
        return Ok(());
    }

    println!("Available baselines:");
    for (i, filepath) in baseline_files.iter().enumerate() {
        let baseline = load_baseline(filepath)?;

        if verbose {
            println!(
                "{}. {} ({})",
                i + 1,
                baseline.metadata.features,
                filepath.display()
            );
            println!("   Created: {}", baseline.metadata.created_at);
            println!("   Branch: {}", baseline.metadata.branch);
            println!("   Commit: {}", &baseline.metadata.commit_hash[..8]);
            println!(
                "   Benchmarks: {}",
                baseline.statistical_summary.total_benchmarks
            );
            println!(
                "   Quality: {:.1}/100",
                baseline.statistical_summary.quality_score
            );
            println!();
        } else {
            println!(
                "  {} - {} benchmarks ({})",
                baseline.metadata.features,
                baseline.statistical_summary.total_benchmarks,
                baseline.metadata.created_at.format("%Y-%m-%d")
            );
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn handle_show_baseline_info(matches: &ArgMatches, verbose: bool) -> Result<()> {
    let baseline_dir = matches.get_one::<String>("baseline-dir").unwrap();
    let features = matches.get_one::<String>("features").unwrap();

    let baselinepath = PathBuf::from(baseline_dir).join(format!("baseline_{}.json", features));

    if !baselinepath.exists() {
        return Err(OptimError::ConfigurationError(format!(
            "Baseline file not found: {}",
            baselinepath.display()
        )));
    }

    let baseline = load_baseline(&baselinepath)?;

    println!("📊 Baseline Information: {}", features);
    println!("=====================================");
    println!("ID: {}", baseline.metadata.baseline_id);
    println!("Features: {}", baseline.metadata.features);
    println!("Created: {}", baseline.metadata.created_at);
    println!("Updated: {}", baseline.metadata.updated_at);
    println!("Branch: {}", baseline.metadata.branch);
    println!("Commit: {}", baseline.metadata.commit_hash);
    println!(
        "Platform: {} {} ({} cores)",
        baseline.metadata.platform_info.os,
        baseline.metadata.platform_info.arch,
        baseline.metadata.platform_info.cpu_cores
    );
    println!(
        "Rust Version: {}",
        baseline.metadata.platform_info.rust_version
    );
    println!();
    println!("Statistics:");
    println!(
        "  Total Benchmarks: {}",
        baseline.statistical_summary.total_benchmarks
    );
    println!(
        "  Stable Benchmarks: {}",
        baseline.statistical_summary.stable_benchmarks
    );
    println!(
        "  Variable Benchmarks: {}",
        baseline.statistical_summary.variable_benchmarks
    );
    println!("  Sample Count: {}", baseline.metadata.sample_count);
    println!(
        "  Quality Score: {:.1}/100",
        baseline.statistical_summary.quality_score
    );
    println!(
        "  Overall Confidence: {:.1}%",
        baseline.statistical_summary.overall_confidence * 100.0
    );

    if verbose {
        println!();
        println!("📈 Metric Details:");
        for (name, metric) in &baseline.metrics {
            println!("  {}:", name);
            println!("    Mean: {:.6}", metric.mean);
            println!("    Std Dev: {:.6}", metric.std_dev);
            println!("    Range: [{:.6}, {:.6}]", metric.min, metric.max);
            println!("    Samples: {}", metric.samples.len());
            println!(
                "    Confidence Interval: [{:.6}, {:.6}]",
                metric.confidence_interval.0, metric.confidence_interval.1
            );
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn loadperformance_results(path: &str) -> Result<serde_json::Value> {
    let content = fs::read_to_string(path)
        .map_err(|e| OptimError::ResourceError(format!("Failed to read results file: {}", e)))?;

    serde_json::from_str(&content)
        .map_err(|e| OptimError::OptimizationError(format!("Failed to parse results: {}", e)))
}

#[allow(dead_code)]
fn load_baseline(path: &PathBuf) -> Result<BaselineMetrics> {
    let content = fs::read_to_string(path)
        .map_err(|e| OptimError::ResourceError(format!("Failed to read baseline file: {}", e)))?;

    serde_json::from_str(&content)
        .map_err(|e| OptimError::OptimizationError(format!("Failed to parse baseline: {}", e)))
}

#[allow(dead_code)]
fn save_baseline(baseline: &BaselineMetrics, path: &PathBuf) -> Result<()> {
    let content = serde_json::to_string_pretty(baseline).map_err(|e| {
        OptimError::OptimizationError(format!("Failed to serialize baseline: {}", e))
    })?;

    fs::write(path, content)
        .map_err(|e| OptimError::ResourceError(format!("Failed to write baseline file: {}", e)))
}

#[allow(dead_code)]
fn create_baseline_from_results(
    results: &serde_json::Value,
    features: &str,
    commit_hash: &str,
    branch: &str,
) -> Result<BaselineMetrics> {
    let mut metrics = HashMap::new();
    let mut total_benchmarks = 0;

    // Extract metrics from results
    if let Some(current_results) = results.get("current_results").and_then(|v| v.as_object()) {
        for (metric_name, metricdata) in current_results {
            if let Some(value) = extract_numeric_value(metricdata) {
                let metric_value = MetricValue {
                    mean: value,
                    std_dev: 0.0, // Initial baseline has no variance
                    min: value,
                    max: value,
                    samples: vec![value],
                    confidence_interval: (value * 0.95, value * 1.05), // ±5% initial confidence
                };
                metrics.insert(metric_name.clone(), metric_value);
                total_benchmarks += 1;
            }
        }
    }

    let now = Utc::now();
    let platform_info = get_platform_info();

    let metadata = BaselineMetadata {
        version: "1.0".to_string(),
        created_at: now,
        updated_at: now,
        commit_hash: commit_hash.to_string(),
        branch: branch.to_string(),
        features: features.to_string(),
        sample_count: 1,
        baseline_id: Uuid::new_v4().to_string(),
        platform_info,
    };

    let statistical_summary = StatisticalSummary {
        total_benchmarks,
        stable_benchmarks: total_benchmarks, // All are considered stable initially
        variable_benchmarks: 0,
        overall_confidence: 0.7, // Initial confidence is moderate
        quality_score: 70.0,     // Initial quality score
    };

    Ok(BaselineMetrics {
        metrics,
        metadata,
        statistical_summary,
    })
}

#[allow(dead_code)]
fn merge_baseline_with_results(
    existing: &BaselineMetrics,
    results: &serde_json::Value,
    features: &str,
    commit_hash: &str,
    branch: &str,
    merge_strategy: &str,
) -> Result<BaselineMetrics> {
    let mut updated_metrics = existing.metrics.clone();
    let mut total_benchmarks = existing.statistical_summary.total_benchmarks;

    // Extract new metrics from results
    if let Some(current_results) = results.get("current_results").and_then(|v| v.as_object()) {
        for (metric_name, metricdata) in current_results {
            if let Some(new_value) = extract_numeric_value(metricdata) {
                if let Some(existing_metric) = updated_metrics.get_mut(metric_name) {
                    // Update existing metric
                    match merge_strategy {
                        "replace" => {
                            existing_metric.mean = new_value;
                            existing_metric.std_dev = 0.0;
                            existing_metric.min = new_value;
                            existing_metric.max = new_value;
                            existing_metric.samples = vec![new_value];
                        }
                        "merge" | "weighted" => {
                            existing_metric.samples.push(new_value);

                            // Limit sample history
                            if existing_metric.samples.len() > 100 {
                                existing_metric.samples.remove(0);
                            }

                            // Recalculate statistics
                            let samples = &existing_metric.samples;
                            existing_metric.mean =
                                samples.iter().sum::<f64>() / samples.len() as f64;
                            existing_metric.min =
                                samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            existing_metric.max =
                                samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                            if samples.len() > 1 {
                                let variance = samples
                                    .iter()
                                    .map(|&x| (x - existing_metric.mean).powi(2))
                                    .sum::<f64>()
                                    / (samples.len() - 1) as f64;
                                existing_metric.std_dev = variance.sqrt();
                            }

                            // Update confidence interval (±2 standard deviations)
                            let margin = existing_metric.std_dev * 2.0;
                            existing_metric.confidence_interval =
                                (existing_metric.mean - margin, existing_metric.mean + margin);
                        }
                        _ => {
                            return Err(OptimError::ConfigurationError(format!(
                                "Unknown merge _strategy: {}",
                                merge_strategy
                            )));
                        }
                    }
                } else {
                    // Add new metric
                    let metric_value = MetricValue {
                        mean: new_value,
                        std_dev: 0.0,
                        min: new_value,
                        max: new_value,
                        samples: vec![new_value],
                        confidence_interval: (new_value * 0.95, new_value * 1.05),
                    };
                    updated_metrics.insert(metric_name.clone(), metric_value);
                    total_benchmarks += 1;
                }
            }
        }
    }

    let now = Utc::now();
    let sample_count = existing.metadata.sample_count + 1;

    // Calculate updated statistical summary
    let stable_benchmarks = updated_metrics
        .values()
        .filter(|m| m.std_dev / m.mean < 0.1) // CV < 10%
        .count();
    let variable_benchmarks = total_benchmarks - stable_benchmarks;

    let overall_confidence = if total_benchmarks > 0 {
        stable_benchmarks as f64 / total_benchmarks as f64
    } else {
        0.0
    };

    let quality_score = (overall_confidence * 100.0).min(100.0);

    let metadata = BaselineMetadata {
        version: existing.metadata.version.clone(),
        created_at: existing.metadata.created_at,
        updated_at: now,
        commit_hash: commit_hash.to_string(),
        branch: branch.to_string(),
        features: features.to_string(),
        sample_count,
        baseline_id: existing.metadata.baseline_id.clone(),
        platform_info: existing.metadata.platform_info.clone(),
    };

    let statistical_summary = StatisticalSummary {
        total_benchmarks,
        stable_benchmarks,
        variable_benchmarks,
        overall_confidence,
        quality_score,
    };

    Ok(BaselineMetrics {
        metrics: updated_metrics,
        metadata,
        statistical_summary,
    })
}

#[allow(dead_code)]
fn extract_numeric_value(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::Object(obj) => {
            // Try common numeric fields
            for field in &["execution_time", "mean_time", "peak_memory", "throughput"] {
                if let Some(v) = obj.get(*field).and_then(|v| v.as_f64()) {
                    return Some(v);
                }
            }
            None
        }
        _ => None,
    }
}

#[allow(dead_code)]
fn validate_baseline(baseline: &BaselineMetrics) -> Result<bool> {
    // Basic validation checks
    if baseline.metrics.is_empty() {
        return Ok(false);
    }

    if baseline.metadata.sample_count == 0 {
        return Ok(false);
    }

    if baseline.statistical_summary.total_benchmarks == 0 {
        return Ok(false);
    }

    // Check for reasonable quality score
    if baseline.statistical_summary.quality_score < 50.0 {
        return Ok(false);
    }

    // Validate individual metrics
    for metric in baseline.metrics.values() {
        if metric.samples.is_empty() {
            return Ok(false);
        }

        if !metric.mean.is_finite() || metric.mean < 0.0 {
            return Ok(false);
        }

        if !metric.std_dev.is_finite() || metric.std_dev < 0.0 {
            return Ok(false);
        }
    }

    Ok(true)
}

#[allow(dead_code)]
fn find_baseline_files(_baselinedir: &str) -> Result<Vec<PathBuf>> {
    let dirpath = PathBuf::from(_baselinedir);

    if !dirpath.exists() {
        return Ok(vec![]);
    }

    let mut baseline_files = Vec::new();

    for entry in fs::read_dir(&dirpath)
        .map_err(|e| OptimError::ResourceError(format!("Failed to read directory: {}", e)))?
    {
        let entry = entry.map_err(|e| {
            OptimError::ResourceError(format!("Failed to read directory entry: {}", e))
        })?;

        let path = entry.path();

        if path.is_file()
            && path.extension().is_some_and(|ext| ext == "json")
            && path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("baseline_"))
        {
            baseline_files.push(path);
        }
    }

    baseline_files.sort();
    Ok(baseline_files)
}

#[allow(dead_code)]
fn get_platform_info() -> PlatformInfo {
    PlatformInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_cores: num_cpus::get(),
        rust_version: "stable".to_string(), // Simplified since RUSTC_VERSION env var not available
    }
}
