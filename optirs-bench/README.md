# OptiRS Bench

Benchmarking, profiling, and performance analysis tools for the OptiRS machine learning optimization library.

## Overview

OptiRS-Bench provides comprehensive benchmarking and performance analysis capabilities for the OptiRS ecosystem. This crate includes tools for measuring optimization performance, detecting performance regressions, monitoring system resources, and ensuring the reliability and security of optimization workloads.

## Features

- **Performance Benchmarking**: Comprehensive optimization performance measurement
- **Regression Detection**: Automated detection of performance regressions
- **Memory Profiling**: Memory usage analysis and leak detection
- **System Monitoring**: Real-time system resource monitoring
- **Security Auditing**: Security analysis of optimization pipelines
- **Cross-Platform Support**: Benchmarking across different platforms and hardware
- **Continuous Integration**: Integration with CI/CD pipelines for automated testing
- **Comparative Analysis**: Side-by-side comparison of optimization strategies

## Benchmarking Tools

### Performance Measurement
- **Throughput Analysis**: Operations per second measurement
- **Latency Profiling**: Step-by-step timing analysis
- **Convergence Tracking**: Optimization convergence rate measurement
- **Resource Utilization**: CPU, memory, and GPU usage monitoring
- **Scalability Testing**: Performance across different problem sizes
- **Hardware-Specific Benchmarks**: Platform-optimized performance tests

### Regression Detection
- **Automated Testing**: Continuous performance regression detection
- **Statistical Analysis**: Statistical significance testing for performance changes
- **Threshold Monitoring**: Configurable performance degradation alerts
- **Historical Tracking**: Long-term performance trend analysis
- **Bisection Analysis**: Automated identification of regression-causing changes

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-bench = "0.1.0"
scirs2-core = "0.1.1"  # Required foundation
```

### Feature Selection

Enable specific benchmarking features:

```toml
[dependencies]
optirs-bench = { version = "0.1.0", features = ["profiling", "regression_detection", "security_auditing"] }
```

Available features:
- `profiling`: Memory and performance profiling tools (enabled by default)
- `regression_detection`: Automated regression detection
- `security_auditing`: Security analysis tools
- `ci_integration`: Continuous integration support

## Command-Line Tools

OptiRS-Bench includes several command-line utilities:

### Main Benchmarking Tool
```bash
# Run comprehensive benchmark suite
optirs-bench --optimizer adam --dataset cifar10 --iterations 1000

# Compare multiple optimizers
optirs-bench compare --optimizers adam,sgd,adamw --dataset imagenet

# Hardware-specific benchmarking
optirs-bench --hardware gpu --device nvidia-rtx-4090
```

### Performance Regression Detector
```bash
# Detect performance regressions
performance-regression-detector --baseline v1.0.0 --current HEAD

# Continuous monitoring
performance-regression-detector --monitor --threshold 5% --alert email
```

### Memory Leak Reporter
```bash
# Memory leak detection
memory-leak-reporter --duration 1h --sample-rate 1s

# Generate memory usage report
memory-leak-reporter --report --output memory_report.html
```

### Security Audit Scanner
```bash
# Security vulnerability scanning
security-audit-scanner --scan-dependencies --check-versions

# Plugin security analysis
security-audit-scanner --verify-plugins --sandbox-test
```

## Usage

### Basic Performance Benchmarking

```rust
use optirs_bench::{BenchmarkSuite, OptimizerBenchmark, BenchmarkConfig};
use optirs_core::optimizers::{Adam, SGD};
use criterion::Criterion;

// Create benchmark configuration
let config = BenchmarkConfig::new()
    .with_iterations(1000)
    .with_warmup_iterations(100)
    .with_dataset_size(10000)
    .with_batch_size(32)
    .build();

// Setup benchmark suite
let mut benchmark_suite = BenchmarkSuite::new()
    .with_config(config)
    .add_optimizer("Adam", Adam::new(0.001))
    .add_optimizer("SGD", SGD::new(0.01))
    .build()?;

// Run benchmarks
let results = benchmark_suite.run().await?;

// Generate report
results.generate_report("benchmark_results.html")?;
results.print_summary();
```

### Memory Profiling

```rust
use optirs_bench::{MemoryProfiler, AllocationTracker};

// Setup memory profiling
let mut profiler = MemoryProfiler::new()
    .with_sampling_rate(Duration::from_millis(100))
    .with_stack_trace_depth(10)
    .build()?;

// Start profiling
profiler.start_profiling()?;

// Your optimization code here
let mut optimizer = Adam::new(0.001);
for epoch in 0..100 {
    // Training loop
    optimizer.step(&mut params, &grads).await?;
    
    // Record memory usage
    profiler.record_memory_snapshot(&format!("epoch_{}", epoch))?;
}

// Stop profiling and generate report
let memory_report = profiler.stop_and_generate_report()?;
memory_report.save_to_file("memory_profile.json")?;
```

### Regression Detection

```rust
use optirs_bench::{RegressionDetector, PerformanceBaseline, StatisticalTest};

// Load performance baseline
let baseline = PerformanceBaseline::from_file("baseline_v1.0.0.json")?;

// Setup regression detector
let detector = RegressionDetector::new()
    .with_baseline(baseline)
    .with_significance_threshold(0.05)
    .with_effect_size_threshold(0.1)
    .with_statistical_test(StatisticalTest::WelchTTest)
    .build()?;

// Run current performance tests
let current_results = run_performance_tests().await?;

// Check for regressions
let regression_analysis = detector.analyze(&current_results)?;

if regression_analysis.has_regressions() {
    println!("Performance regressions detected:");
    for regression in regression_analysis.regressions() {
        println!("  - {}: {:.2}% slower (p-value: {:.4})", 
                 regression.metric_name, 
                 regression.performance_delta * 100.0,
                 regression.p_value);
    }
}
```

### System Resource Monitoring

```rust
use optirs_bench::{SystemMonitor, ResourceAlert};

// Setup system monitoring
let monitor = SystemMonitor::new()
    .with_sampling_interval(Duration::from_secs(1))
    .monitor_cpu(true)
    .monitor_memory(true)
    .monitor_gpu(true)
    .monitor_disk_io(true)
    .monitor_network_io(true)
    .build()?;

// Configure alerts
let alerts = ResourceAlert::new()
    .cpu_threshold(90.0)  // Alert if CPU usage > 90%
    .memory_threshold(8_000_000_000)  // Alert if memory usage > 8GB
    .gpu_memory_threshold(0.95)  // Alert if GPU memory > 95%
    .build();

monitor.set_alerts(alerts);

// Start monitoring
let monitoring_handle = monitor.start_monitoring().await?;

// Your optimization workload
run_training_workload().await?;

// Stop monitoring and get report
let resource_report = monitor.stop_and_report().await?;
resource_report.save_to_file("resource_usage.json")?;
```

### Comparative Analysis

```rust
use optirs_bench::{ComparativeAnalysis, OptimizerComparison, StatisticalComparison};

// Compare multiple optimizers
let comparison = ComparativeAnalysis::new()
    .add_optimizer("Adam", adam_results)
    .add_optimizer("SGD", sgd_results)
    .add_optimizer("AdamW", adamw_results)
    .with_metrics(&["convergence_speed", "final_accuracy", "memory_usage"])
    .build()?;

// Perform statistical comparison
let statistical_analysis = comparison.statistical_comparison()?;

// Generate visualization
comparison.generate_comparison_plots("optimizer_comparison.html")?;

// Print summary
for result in statistical_analysis.significant_differences() {
    println!("{} vs {}: {} is significantly better (p < 0.05)", 
             result.optimizer_a, 
             result.optimizer_b, 
             result.better_performer);
}
```

## Security Auditing

### Dependency Scanning

```rust
use optirs_bench::security::{SecurityAuditor, VulnerabilityScanner};

// Setup security auditor
let auditor = SecurityAuditor::new()
    .with_vulnerability_database(VulnerabilityDB::latest())
    .with_severity_threshold(Severity::Medium)
    .build()?;

// Scan dependencies
let scan_results = auditor.scan_dependencies().await?;

if scan_results.has_vulnerabilities() {
    println!("Security vulnerabilities found:");
    for vuln in scan_results.vulnerabilities() {
        println!("  - {}: {} ({})", 
                 vuln.crate_name, 
                 vuln.description, 
                 vuln.severity);
    }
}

// Generate security report
scan_results.generate_security_report("security_audit.html")?;
```

## Continuous Integration Integration

### GitHub Actions

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Run benchmarks
        run: |
          cargo run --bin optirs-bench -- --output benchmark_results.json
          
      - name: Check for regressions
        run: |
          cargo run --bin performance-regression-detector -- \
            --baseline benchmark_baseline.json \
            --current benchmark_results.json \
            --fail-on-regression
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                sh 'cargo build --release'
            }
        }
        
        stage('Benchmark') {
            steps {
                sh 'cargo run --bin optirs-bench -- --ci-mode'
                archiveArtifacts 'benchmark_results.json'
            }
        }
        
        stage('Regression Check') {
            steps {
                script {
                    def regressionCheck = sh(
                        script: 'cargo run --bin performance-regression-detector',
                        returnStatus: true
                    )
                    if (regressionCheck != 0) {
                        error "Performance regression detected!"
                    }
                }
            }
        }
    }
}
```

## Configuration

### Benchmark Configuration File

```yaml
# bench_config.yaml
benchmark:
  iterations: 1000
  warmup_iterations: 100
  timeout: 300  # seconds
  
optimizers:
  - name: "Adam"
    learning_rate: 0.001
    beta1: 0.9
    beta2: 0.999
  - name: "SGD"
    learning_rate: 0.01
    momentum: 0.9
    
datasets:
  - name: "CIFAR-10"
    size: 50000
    batch_size: 32
  - name: "ImageNet"
    size: 1281167
    batch_size: 64
    
monitoring:
  sample_rate: 1000  # milliseconds
  metrics:
    - cpu_usage
    - memory_usage
    - gpu_utilization
    - disk_io
    
regression_detection:
  significance_threshold: 0.05
  effect_size_threshold: 0.1
  baseline_file: "baseline.json"
```

## Platform Support

| Platform | CPU Profiling | GPU Profiling | Memory Profiling | Security Scanning |
|----------|---------------|---------------|------------------|-------------------|
| Linux    | ✅           | ✅ (CUDA/ROCm)| ✅              | ✅               |
| macOS    | ✅           | ✅ (Metal)   | ✅              | ✅               |
| Windows  | ✅           | ✅ (CUDA/DX)  | ✅              | ✅               |
| Web      | ⚠️ (Limited)  | ❌            | ⚠️ (Limited)    | ⚠️ (Limited)     |

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS repository for contribution guidelines.

## License

This project is licensed under either of:
- Apache License, Version 2.0
- MIT License

at your option.