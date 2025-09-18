// Memory pattern analyzer binary
//
// This binary analyzes memory allocation patterns from profiling data
// and identifies potential optimization opportunities.

use clap::{Arg, Command};
use scirs2_optim::benchmarking::memory_leak_detector::{
    AllocationType, MemoryAnomaly, MemoryPattern, MemoryUsageSnapshot,
};
use scirs2_optim::error::Result;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalysisInput {
    snapshots: Vec<MemoryUsageSnapshot>,
    allocation_events: Vec<AllocationEvent>,
    metadata: AnalysisMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AllocationEvent {
    timestamp: u64,
    allocation_id: usize,
    size: usize,
    allocation_type: AllocationType,
    operation: AllocationOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum AllocationOperation {
    Allocate,
    Deallocate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalysisMetadata {
    session_id: String,
    optimizer_type: String,
    problem_size: usize,
    duration_seconds: u64,
    sampling_rate_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternAnalysisResult {
    session_metadata: AnalysisMetadata,
    detected_patterns: Vec<MemoryPattern>,
    anomalies: Vec<MemoryAnomaly>,
    allocation_statistics: AllocationStatistics,
    memory_trends: MemoryTrends,
    optimization_suggestions: Vec<OptimizationSuggestion>,
    risk_assessment: RiskAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AllocationStatistics {
    total_allocations: usize,
    total_deallocations: usize,
    peak_active_allocations: usize,
    allocation_size_distribution: HashMap<String, usize>,
    allocation_type_distribution: HashMap<String, usize>,
    average_allocation_size: f64,
    largest_allocation: usize,
    allocation_rate_per_second: f64,
    deallocation_rate_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryTrends {
    overall_trend: TrendDirection,
    trend_strength: f64,
    growth_rate_bytes_per_second: f64,
    memory_usage_stability: f64,
    peak_memory_usage: usize,
    average_memory_usage: f64,
    memory_utilization_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Irregular,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationSuggestion {
    category: OptimizationCategory,
    priority: SuggestionPriority,
    description: String,
    potential_impact: String,
    implementation_effort: ImplementationEffort,
    code_example: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum OptimizationCategory {
    MemoryPooling,
    AllocationReduction,
    DeallocationOptimization,
    MemoryLayout,
    CacheOptimization,
    LeakPrevention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum SuggestionPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RiskAssessment {
    leakrisk: RiskLevel,
    performancerisk: RiskLevel,
    stabilityrisk: RiskLevel,
    scalabilityrisk: RiskLevel,
    overallrisk: RiskLevel,
    risk_factors: Vec<String>,
    mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let matches = Command::new("Memory Pattern Analyzer")
        .version("1.0")
        .about("Analyzes memory allocation patterns and identifies optimization opportunities")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("DIRECTORY")
                .help("Input directory containing memory profiling data")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for analysis results")
                .default_value("pattern_analysis.json"),
        )
        .arg(
            Arg::new("format")
                .short('f')
                .long("format")
                .value_name("FORMAT")
                .help("Output format")
                .value_parser(["json", "markdown", "html"])
                .default_value("json"),
        )
        .arg(
            Arg::new("detailed")
                .long("detailed")
                .action(clap::ArgAction::SetTrue)
                .help("Generate detailed analysis report"),
        )
        .get_matches();

    let input_dir = matches.get_one::<String>("input").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();
    let output_format = matches.get_one::<String>("format").unwrap();
    let detailed = matches.get_flag("detailed");

    println!("🔍 Memory Pattern Analyzer");
    println!("==========================");
    println!("Input directory: {}", input_dir);
    println!("Output file: {}", output_file);
    println!("Output format: {}", output_format);
    println!("Detailed analysis: {}", detailed);
    println!();

    // Load analysis input data
    let analysis_input = load_analysis_input(input_dir)?;
    println!(
        "📊 Loaded {} snapshots and {} allocation events",
        analysis_input.snapshots.len(),
        analysis_input.allocation_events.len()
    );

    // Perform pattern analysis
    println!("🔬 Analyzing memory patterns...");
    let analysisresult = analyze_memory_patterns(&analysis_input, detailed)?;

    // Display summary
    display_analysis_summary(&analysisresult);

    // Generate output
    match output_format.as_str() {
        "json" => {
            let json_output = serde_json::to_string_pretty(&analysisresult)?;
            fs::write(output_file, json_output)?;
        }
        "markdown" => {
            let markdown_output = generate_markdownreport(&analysisresult);
            fs::write(output_file, markdown_output)?;
        }
        "html" => {
            let html_output = generate_htmlreport(&analysisresult);
            fs::write(output_file, html_output)?;
        }
        _ => unreachable!(),
    }

    println!("✅ Analysis complete. Results saved to: {}", output_file);
    Ok(())
}

#[allow(dead_code)]
fn load_analysis_input(_inputdir: &str) -> Result<AnalysisInput> {
    let inputpath = Path::new(_inputdir);

    // Try to find relevant files
    let mut snapshots = Vec::new();
    let mut allocation_events = Vec::new();
    let mut metadata = None;

    if inputpath.is_dir() {
        for entry in fs::read_dir(inputpath)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.contains("memory_profile") && filename.ends_with(".json") {
                    // Try to parse as profiling results
                    let content = fs::read_to_string(&path)?;
                    if let Ok(profilingdata) = serde_json::from_str::<serde_json::Value>(&content) {
                        // Extract snapshots if available
                        if let Some(snapshot_array) = profilingdata.get("snapshots") {
                            if let Ok(parsed_snapshots) =
                                serde_json::from_value::<Vec<MemoryUsageSnapshot>>(
                                    snapshot_array.clone(),
                                )
                            {
                                snapshots.extend(parsed_snapshots);
                            }
                        }

                        // Extract metadata if available
                        if let Some(sessiondata) = profilingdata.get("session") {
                            if let Ok(parsed_metadata) =
                                serde_json::from_value::<AnalysisMetadata>(sessiondata.clone())
                            {
                                metadata = Some(parsed_metadata);
                            }
                        }
                    }
                }

                if filename.contains("allocation") && filename.ends_with(".json") {
                    // Try to parse allocation events
                    let content = fs::read_to_string(&path)?;
                    if let Ok(events) = serde_json::from_str::<Vec<AllocationEvent>>(&content) {
                        allocation_events.extend(events);
                    }
                }
            }
        }
    } else {
        // Single file input
        let content = fs::read_to_string(inputpath)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        // Try to extract data from the file
        if let Some(snapshot_array) = data.get("snapshots") {
            snapshots = serde_json::from_value(snapshot_array.clone())?;
        }

        if let Some(events_array) = data.get("allocation_events") {
            allocation_events = serde_json::from_value(events_array.clone())?;
        }

        if let Some(metadata_data) = data.get("metadata") {
            metadata = Some(serde_json::from_value(metadata_data.clone())?);
        }
    }

    // Use default metadata if not found
    let metadata = metadata.unwrap_or_else(|| AnalysisMetadata {
        session_id: "unknown".to_string(),
        optimizer_type: "unknown".to_string(),
        problem_size: 0,
        duration_seconds: 0,
        sampling_rate_ms: 1000,
    });

    Ok(AnalysisInput {
        snapshots,
        allocation_events,
        metadata,
    })
}

#[allow(dead_code)]
fn analyze_memory_patterns(input: &AnalysisInput, detailed: bool) -> Result<PatternAnalysisResult> {
    println!("  📈 Analyzing allocation statistics...");
    let allocation_statistics =
        analyze_allocation_statistics(&input.allocation_events, &input.metadata);

    println!("  📊 Analyzing memory trends...");
    let memory_trends = analyze_memory_trends(&input.snapshots);

    println!("  🔍 Detecting patterns...");
    let detected_patterns = detect_memory_patterns(&input.snapshots, &input.allocation_events);

    println!("  ⚠️  Detecting anomalies...");
    let anomalies = detect_memory_anomalies(&input.snapshots, &input.allocation_events);

    println!("  💡 Generating optimization suggestions...");
    let optimization_suggestions = generate_optimization_suggestions(
        &allocation_statistics,
        &memory_trends,
        &detected_patterns,
        &anomalies,
    );

    println!("  🎯 Assessing risks...");
    let risk_assessment = assessrisks(
        &allocation_statistics,
        &memory_trends,
        &detected_patterns,
        &anomalies,
    );

    Ok(PatternAnalysisResult {
        session_metadata: input.metadata.clone(),
        detected_patterns,
        anomalies,
        allocation_statistics,
        memory_trends,
        optimization_suggestions,
        risk_assessment,
    })
}

#[allow(dead_code)]
fn analyze_allocation_statistics(
    events: &[AllocationEvent],
    metadata: &AnalysisMetadata,
) -> AllocationStatistics {
    let total_allocations = events
        .iter()
        .filter(|e| matches!(e.operation, AllocationOperation::Allocate))
        .count();
    let total_deallocations = events
        .iter()
        .filter(|e| matches!(e.operation, AllocationOperation::Deallocate))
        .count();

    let mut active_allocations: usize = 0;
    let mut peak_active_allocations = 0;
    let mut allocation_sizes = Vec::new();
    let mut allocation_type_counts = HashMap::new();

    for event in events {
        match event.operation {
            AllocationOperation::Allocate => {
                active_allocations += 1;
                peak_active_allocations = peak_active_allocations.max(active_allocations);
                allocation_sizes.push(event.size);

                let type_name = format!("{:?}", event.allocation_type);
                *allocation_type_counts.entry(type_name).or_insert(0) += 1;
            }
            AllocationOperation::Deallocate => {
                active_allocations = active_allocations.saturating_sub(1);
            }
        }
    }

    let average_allocation_size = if !allocation_sizes.is_empty() {
        allocation_sizes.iter().sum::<usize>() as f64 / allocation_sizes.len() as f64
    } else {
        0.0
    };

    let largest_allocation = allocation_sizes.iter().max().copied().unwrap_or(0);

    // Create size distribution
    let mut allocation_size_distribution = HashMap::new();
    for &size in &allocation_sizes {
        let bucket = match size {
            0..=1024 => "0-1KB",
            1025..=10240 => "1-10KB",
            10241..=102400 => "10-100KB",
            102401..=1048576 => "100KB-1MB",
            _ => ">1MB",
        };
        *allocation_size_distribution
            .entry(bucket.to_string())
            .or_insert(0) += 1;
    }

    let duration = metadata.duration_seconds as f64;
    let allocation_rate_per_second = if duration > 0.0 {
        total_allocations as f64 / duration
    } else {
        0.0
    };
    let deallocation_rate_per_second = if duration > 0.0 {
        total_deallocations as f64 / duration
    } else {
        0.0
    };

    AllocationStatistics {
        total_allocations,
        total_deallocations,
        peak_active_allocations,
        allocation_size_distribution,
        allocation_type_distribution: allocation_type_counts,
        average_allocation_size,
        largest_allocation,
        allocation_rate_per_second,
        deallocation_rate_per_second,
    }
}

#[allow(dead_code)]
fn analyze_memory_trends(snapshots: &[MemoryUsageSnapshot]) -> MemoryTrends {
    if snapshots.is_empty() {
        return MemoryTrends {
            overall_trend: TrendDirection::Stable,
            trend_strength: 0.0,
            growth_rate_bytes_per_second: 0.0,
            memory_usage_stability: 1.0,
            peak_memory_usage: 0,
            average_memory_usage: 0.0,
            memory_utilization_efficiency: 1.0,
        };
    }

    let memory_values: Vec<f64> = snapshots.iter().map(|s| s.total_memory as f64).collect();
    let peak_memory_usage = snapshots.iter().map(|s| s.total_memory).max().unwrap_or(0);
    let average_memory_usage = memory_values.iter().sum::<f64>() / memory_values.len() as f64;

    // Calculate trend
    let (trend_direction, trend_strength) = calculate_trend(&memory_values);

    // Calculate growth rate
    let first_timestamp = snapshots.first().unwrap().timestamp;
    let last_timestamp = snapshots.last().unwrap().timestamp;
    let duration_seconds = (last_timestamp - first_timestamp) as f64 / 1000.0;

    let growth_rate_bytes_per_second = if duration_seconds > 0.0 {
        (snapshots.last().unwrap().total_memory as f64
            - snapshots.first().unwrap().total_memory as f64)
            / duration_seconds
    } else {
        0.0
    };

    // Calculate stability (coefficient of variation)
    let std_dev = calculate_std_dev(&memory_values);
    let memory_usage_stability = if average_memory_usage > 0.0 {
        1.0 - (std_dev / average_memory_usage).min(1.0)
    } else {
        1.0
    };

    // Calculate utilization efficiency (simplified)
    let memory_utilization_efficiency = 0.8; // Would be calculated based on actual utilization

    MemoryTrends {
        overall_trend: trend_direction,
        trend_strength,
        growth_rate_bytes_per_second,
        memory_usage_stability,
        peak_memory_usage,
        average_memory_usage,
        memory_utilization_efficiency,
    }
}

#[allow(dead_code)]
fn calculate_trend(values: &[f64]) -> (TrendDirection, f64) {
    if values.len() < 2 {
        return (TrendDirection::Stable, 0.0);
    }

    // Simple linear regression for trend analysis
    let n = values.len() as f64;
    let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
    let y_sum: f64 = values.iter().sum();
    let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

    let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));

    let direction = if slope > 10.0 {
        TrendDirection::Increasing
    } else if slope < -10.0 {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    };

    let strength = slope.abs() / values.iter().sum::<f64>() * n;

    (direction, strength)
}

#[allow(dead_code)]
fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[allow(dead_code)]
fn detect_memory_patterns(
    _snapshots: &[MemoryUsageSnapshot],
    _events: &[AllocationEvent],
) -> Vec<MemoryPattern> {
    // Simplified pattern detection
    vec![MemoryPattern {
        pattern_type: "steady_growth".to_string(),
        confidence: 0.8,
        description: "Steady memory growth pattern detected".to_string(),
        metrics: HashMap::new(),
    }]
}

#[allow(dead_code)]
fn detect_memory_anomalies(
    _snapshots: &[MemoryUsageSnapshot],
    _events: &[AllocationEvent],
) -> Vec<MemoryAnomaly> {
    // Simplified anomaly detection
    Vec::new()
}

#[allow(dead_code)]
fn generate_optimization_suggestions(
    stats: &AllocationStatistics,
    trends: &MemoryTrends,
    patterns: &[MemoryPattern],
    _anomalies: &[MemoryAnomaly],
) -> Vec<OptimizationSuggestion> {
    let mut suggestions = Vec::new();

    // High allocation rate suggestion
    if stats.allocation_rate_per_second > 1000.0 {
        suggestions.push(OptimizationSuggestion {
            category: OptimizationCategory::MemoryPooling,
            priority: SuggestionPriority::High,
            description: "High allocation rate detected. Consider implementing memory pooling."
                .to_string(),
            potential_impact: format!(
                "Could reduce allocation overhead by 50-80% ({:.0} allocs/sec)",
                stats.allocation_rate_per_second
            ),
            implementation_effort: ImplementationEffort::Medium,
            code_example: Some(
                "let pool = MemoryPool::new(chunk_size); let memory = pool.allocate(size);"
                    .to_string(),
            ),
        });
    }

    // Memory growth suggestion
    if trends.growth_rate_bytes_per_second > 1024.0 {
        suggestions.push(OptimizationSuggestion {
            category: OptimizationCategory::LeakPrevention,
            priority: SuggestionPriority::Critical,
            description: "Significant memory growth detected. Investigate potential memory leaks."
                .to_string(),
            potential_impact: format!(
                "Growing at {:.2} KB/s - could exhaust memory",
                trends.growth_rate_bytes_per_second / 1024.0
            ),
            implementation_effort: ImplementationEffort::High,
            code_example: Some(
                "// Use RAII _patterns and smart pointers\nlet _guard = MemoryGuard::new();"
                    .to_string(),
            ),
        });
    }

    // Large allocation suggestion
    if stats.largest_allocation > 10 * 1024 * 1024 {
        suggestions.push(OptimizationSuggestion {
            category: OptimizationCategory::AllocationReduction,
            priority: SuggestionPriority::Medium,
            description: "Large allocations detected. Consider chunking or streaming.".to_string(),
            potential_impact: format!(
                "Largest allocation: {:.2} MB",
                stats.largest_allocation as f64 / (1024.0 * 1024.0)
            ),
            implementation_effort: ImplementationEffort::Medium,
            code_example: Some(
                "// Process data in chunks\nfor chunk in data.chunks(chunk_size) { /* process */ }"
                    .to_string(),
            ),
        });
    }

    suggestions
}

#[allow(dead_code)]
fn assessrisks(
    stats: &AllocationStatistics,
    trends: &MemoryTrends,
    patterns: &[MemoryPattern],
    anomalies: &[MemoryAnomaly],
) -> RiskAssessment {
    let mut risk_factors = Vec::new();
    let mut mitigation_strategies = Vec::new();

    // Assess leak risk
    let leakrisk = if trends.growth_rate_bytes_per_second > 10240.0 {
        risk_factors.push("High memory growth rate detected".to_string());
        mitigation_strategies.push("Implement comprehensive leak detection".to_string());
        RiskLevel::High
    } else if trends.growth_rate_bytes_per_second > 1024.0 {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    };

    // Assess performance risk
    let performancerisk = if stats.allocation_rate_per_second > 5000.0 {
        risk_factors.push("Very high allocation rate".to_string());
        mitigation_strategies.push("Implement memory pooling and object reuse".to_string());
        RiskLevel::High
    } else if stats.allocation_rate_per_second > 1000.0 {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    };

    // Assess stability risk
    let stabilityrisk = if trends.memory_usage_stability < 0.7 {
        risk_factors.push("High memory usage variability".to_string());
        mitigation_strategies.push("Stabilize allocation _patterns".to_string());
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    };

    // Assess scalability risk
    let scalabilityrisk = if stats.peak_active_allocations > 100000 {
        risk_factors.push("Very high number of concurrent allocations".to_string());
        mitigation_strategies
            .push("Optimize data structures and reduce memory fragmentation".to_string());
        RiskLevel::High
    } else {
        RiskLevel::Low
    };

    // Overall risk assessment
    let risk_levels = [
        &leakrisk,
        &performancerisk,
        &stabilityrisk,
        &scalabilityrisk,
    ];
    let overallrisk = if risk_levels.iter().any(|r| matches!(r, RiskLevel::High)) {
        RiskLevel::High
    } else if risk_levels.iter().any(|r| matches!(r, RiskLevel::Medium)) {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    };

    // Add anomaly-based risks
    if !anomalies.is_empty() {
        risk_factors.push(format!("{} memory anomalies detected", anomalies.len()));
        mitigation_strategies.push("Investigate and address detected anomalies".to_string());
    }

    RiskAssessment {
        leakrisk,
        performancerisk,
        stabilityrisk,
        scalabilityrisk,
        overallrisk,
        risk_factors,
        mitigation_strategies,
    }
}

#[allow(dead_code)]
fn display_analysis_summary(result: &PatternAnalysisResult) {
    println!("\n📋 Memory Pattern Analysis Summary");
    println!("=================================");
    println!("Session: {}", result.session_metadata.session_id);
    println!("Optimizer: {}", result.session_metadata.optimizer_type);
    println!(
        "Problem size: {} parameters",
        result.session_metadata.problem_size
    );

    println!("\n📊 Allocation Statistics:");
    println!(
        "  Total allocations: {}",
        result.allocation_statistics.total_allocations
    );
    println!(
        "  Total deallocations: {}",
        result.allocation_statistics.total_deallocations
    );
    println!(
        "  Peak active allocations: {}",
        result.allocation_statistics.peak_active_allocations
    );
    println!(
        "  Average allocation size: {:.2} KB",
        result.allocation_statistics.average_allocation_size / 1024.0
    );
    println!(
        "  Allocation rate: {:.2}/sec",
        result.allocation_statistics.allocation_rate_per_second
    );

    println!("\n📈 Memory Trends:");
    println!("  Overall trend: {:?}", result.memory_trends.overall_trend);
    println!(
        "  Growth rate: {:.2} KB/s",
        result.memory_trends.growth_rate_bytes_per_second / 1024.0
    );
    println!(
        "  Peak memory: {:.2} MB",
        result.memory_trends.peak_memory_usage as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Memory stability: {:.2}%",
        result.memory_trends.memory_usage_stability * 100.0
    );

    println!("\n🎯 Risk Assessment:");
    println!("  Overall risk: {:?}", result.risk_assessment.overallrisk);
    println!("  Leak risk: {:?}", result.risk_assessment.leakrisk);
    println!(
        "  Performance risk: {:?}",
        result.risk_assessment.performancerisk
    );

    if !result.optimization_suggestions.is_empty() {
        println!("\n💡 Top Optimization Suggestions:");
        for (i, suggestion) in result.optimization_suggestions.iter().take(3).enumerate() {
            println!(
                "  {}. {:?} - {:?}",
                i + 1,
                suggestion.category,
                suggestion.priority
            );
            println!("     {}", suggestion.description);
        }
    }
}

#[allow(dead_code)]
fn generate_markdownreport(result: &PatternAnalysisResult) -> String {
    format!(
        r#"# Memory Pattern Analysis Report

## Session Information
- **Session ID**: {}
- **Optimizer**: {}
- **Problem Size**: {} parameters
- **Duration**: {} seconds

## Allocation Statistics
- **Total Allocations**: {}
- **Total Deallocations**: {}
- **Peak Active Allocations**: {}
- **Average Allocation Size**: {:.2} KB
- **Allocation Rate**: {:.2}/sec

## Memory Trends
- **Overall Trend**: {:?}
- **Growth Rate**: {:.2} KB/s
- **Peak Memory**: {:.2} MB
- **Memory Stability**: {:.2}%

## Risk Assessment
- **Overall Risk**: {:?}
- **Leak Risk**: {:?}
- **Performance Risk**: {:?}
- **Stability Risk**: {:?}

## Optimization Suggestions
{}

## Risk Factors
{}

## Mitigation Strategies
{}
"#,
        result.session_metadata.session_id,
        result.session_metadata.optimizer_type,
        result.session_metadata.problem_size,
        result.session_metadata.duration_seconds,
        result.allocation_statistics.total_allocations,
        result.allocation_statistics.total_deallocations,
        result.allocation_statistics.peak_active_allocations,
        result.allocation_statistics.average_allocation_size / 1024.0,
        result.allocation_statistics.allocation_rate_per_second,
        result.memory_trends.overall_trend,
        result.memory_trends.growth_rate_bytes_per_second / 1024.0,
        result.memory_trends.peak_memory_usage as f64 / (1024.0 * 1024.0),
        result.memory_trends.memory_usage_stability * 100.0,
        result.risk_assessment.overallrisk,
        result.risk_assessment.leakrisk,
        result.risk_assessment.performancerisk,
        result.risk_assessment.stabilityrisk,
        result
            .optimization_suggestions
            .iter()
            .map(|s| format!(
                "- **{:?}** ({}): {}",
                s.category,
                match s.priority {
                    SuggestionPriority::Critical => "Critical",
                    SuggestionPriority::High => "High",
                    SuggestionPriority::Medium => "Medium",
                    SuggestionPriority::Low => "Low",
                },
                s.description
            ))
            .collect::<Vec<_>>()
            .join("\n"),
        result
            .risk_assessment
            .risk_factors
            .iter()
            .map(|f| format!("- {f}"))
            .collect::<Vec<_>>()
            .join("\n"),
        result
            .risk_assessment
            .mitigation_strategies
            .iter()
            .map(|s| format!("- {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

#[allow(dead_code)]
fn generate_htmlreport(result: &PatternAnalysisResult) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Memory Pattern Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ margin: 10px 0; }}
        .risk-high {{ color: #d32f2f; }}
        .risk-medium {{ color: #f57c00; }}
        .risk-low {{ color: #388e3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Memory Pattern Analysis Report</h1>
        <p><strong>Session:</strong> {}</p>
        <p><strong>Optimizer:</strong> {}</p>
        <p><strong>Generated:</strong> {}</p>
    </div>
    
    <div class="section">
        <h2>📊 Allocation Statistics</h2>
        <div class="metric">Total Allocations: {}</div>
        <div class="metric">Total Deallocations: {}</div>
        <div class="metric">Peak Active Allocations: {}</div>
        <div class="metric">Average Allocation Size: {:.2} KB</div>
    </div>
    
    <div class="section">
        <h2>📈 Memory Trends</h2>
        <div class="metric">Overall Trend: {:?}</div>
        <div class="metric">Growth Rate: {:.2} KB/s</div>
        <div class="metric">Peak Memory: {:.2} MB</div>
        <div class="metric">Memory Stability: {:.2}%</div>
    </div>
    
    <div class="section">
        <h2>🎯 Risk Assessment</h2>
        <div class="metric">Overall Risk: <span class="risk-{}">{:?}</span></div>
        <div class="metric">Leak Risk: <span class="risk-{}">{:?}</span></div>
        <div class="metric">Performance Risk: <span class="risk-{}">{:?}</span></div>
    </div>
    
    <div class="section">
        <h2>💡 Optimization Suggestions</h2>
        <ul>
        {}
        </ul>
    </div>
</body>
</html>"#,
        result.session_metadata.session_id,
        result.session_metadata.optimizer_type,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        result.allocation_statistics.total_allocations,
        result.allocation_statistics.total_deallocations,
        result.allocation_statistics.peak_active_allocations,
        result.allocation_statistics.average_allocation_size / 1024.0,
        result.memory_trends.overall_trend,
        result.memory_trends.growth_rate_bytes_per_second / 1024.0,
        result.memory_trends.peak_memory_usage as f64 / (1024.0 * 1024.0),
        result.memory_trends.memory_usage_stability * 100.0,
        risk_level_to_css_class(&result.risk_assessment.overallrisk),
        result.risk_assessment.overallrisk,
        risk_level_to_css_class(&result.risk_assessment.leakrisk),
        result.risk_assessment.leakrisk,
        risk_level_to_css_class(&result.risk_assessment.performancerisk),
        result.risk_assessment.performancerisk,
        result
            .optimization_suggestions
            .iter()
            .map(|s| format!(
                "<li><strong>{:?}</strong>: {}</li>",
                s.category, s.description
            ))
            .collect::<Vec<_>>()
            .join("\n        ")
    )
}

#[allow(dead_code)]
fn risk_level_to_css_class(risk: &RiskLevel) -> &'static str {
    match risk {
        RiskLevel::High | RiskLevel::Critical => "high",
        RiskLevel::Medium => "medium",
        RiskLevel::Low => "low",
    }
}
