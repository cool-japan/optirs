# OptiRS Bench

Benchmarking, profiling, and performance analysis tools for the OptiRS machine learning optimization library.

## Overview

OptiRS-Bench provides comprehensive benchmarking and performance analysis capabilities for the OptiRS ecosystem. This crate includes tools for measuring optimization performance, detecting performance regressions, monitoring system resources, and ensuring the reliability and security of optimization workloads.

## Features

- **Performance Benchmarking**: Comprehensive optimization performance measurement
- **Regression Detection**: Automated detection of performance regressions
- **Memory Profiling**: Memory usage analysis and leak detection
- **System Monitoring**: Real process/system resource monitoring (CPU, memory) via `sysinfo`
- **Security Auditing**: Security analysis of optimization pipelines
- **Cross-Platform Support**: Orchestrated benchmarking across platforms via local, Docker,
  and SSH execution, with an explicit error when the runtime is absent rather than a
  fabricated pass
- **Continuous Integration**: Integration with CI/CD pipelines for automated testing
- **Comparative Analysis**: Side-by-side comparison of optimization strategies

## Benchmarking Tools

### Performance Measurement
- **Throughput Analysis**: Operations per second measurement
- **Latency Profiling**: Step-by-step timing analysis
- **Convergence Tracking**: Optimization convergence rate measurement
- **Resource Utilization**: CPU and memory usage monitoring via `sysinfo` (real process
  RSS/virtual memory and system-wide load); GPU presence is detected (for cross-platform
  test-matrix purposes) but there is no GPU utilization/memory telemetry

### Regression Detection
- **Automated Testing**: Continuous performance regression detection
- **Statistical Analysis**: Statistical significance testing for performance changes
- **Threshold Monitoring**: Configurable performance degradation alerts
- **Historical Tracking**: Long-term performance trend analysis

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-bench = "0.3.3"
scirs2-core = "0.6.5"  # Required foundation
```

### Feature Selection

Enable specific benchmarking features:

```toml
[dependencies]
optirs-bench = { version = "0.3.3", features = ["profiling", "regression_detection", "security_auditing"] }
```

Available features:
- `profiling`: Memory and performance profiling tools (enabled by default)
- `regression_detection`: Automated regression detection
- `security_auditing`: Security analysis tools (pulls in `rsa`/`x509-parser`)
- `ci_integration`: Continuous integration support

## Command-Line Tools

`optirs-bench` ships 10 binaries (`autobins = false` in `Cargo.toml`; the `[[bin]]` list
there is authoritative). Two pairs of names differ only by hyphen vs. underscore and are
genuinely different tools -- read the binary path, not just the name.

### Main benchmarking tool: `optirs-bench`

Runs `optirs_bench::OptimizerBenchmark` against the crate's three built-in test functions
(Quadratic, Rosenbrock, Sphere) comparing a baseline gradient-descent step against a
decayed-learning-rate variant. There is no dataset loading, no `--optimizer`/`--dataset`
selection, and no GPU/hardware targeting flag.

```bash
# Run the built-in benchmark suite and print a summary
optirs-bench benchmark [--iterations N] [--tolerance T]

# Compare the decayed-LR candidate against the baseline; exits non-zero if its
# success rate regresses by more than PCT percentage points (default: 10.0)
optirs-bench analyze [--max-regression PCT] [--iterations N] [--tolerance T]

# Render a full report to stdout, or to a file
optirs-bench report [--format md|txt|csv] [--output PATH] [--iterations N] [--tolerance T]
```

### Performance regression detector: `performance-regression-detector`

Two positional metrics files, not flags. Each line is `test_name: value[, value, ...]`;
when several samples are given per line, their mean is compared. This tool only prints
its findings -- its process always exits `0`, so it cannot gate a CI build by itself
(use `optirs-bench analyze` or `security_audit_scanner` below for that).

```bash
performance-regression-detector <baseline_file> <current_file> [threshold_fraction]
# threshold_fraction is relative and defaults to 0.05 (5% growth)
```

### Memory reporting -- two different binaries

- **`memory-leak-reporter`** (hyphen; `src/bin/memory_reporter.rs`): a one-shot snapshot
  of the *current* process' real RSS/virtual memory, via `system_sampler::SystemSampler`
  (backed by `sysinfo`). Despite the name, it does not detect leaks.
  ```bash
  memory-leak-reporter <output_file>
  ```
- **`memory_leak_reporter`** (underscore; `src/bin/memory_leak_reporter.rs`): parses an
  *existing* report from an external memory tool (Valgrind XML, Massif, HeapTrack, or a
  custom JSON profiler) and renders a leak analysis.
  ```bash
  memory_leak_reporter --input <path> --format json|markdown|github-actions \
      [--output PATH] [--severity-threshold N] [--confidence-threshold N] \
      [--include-recommendations] [--verbose]
  ```

### Security scanning -- two different binaries

- **`security-audit-scanner`** (hyphen; `src/bin/security_scanner.rs`): a dependency-free
  recursive scan of a directory for suspicious source patterns (potential secrets, weak
  crypto, `unsafe`, command injection), word-boundary matched to avoid substring false
  positives.
  ```bash
  security-audit-scanner <directory>
  ```
- **`security_audit_scanner`** (underscore; `src/bin/security_audit_scanner.rs`): the full
  engine -- RustSec-range vulnerability matching against an embedded offline advisory
  snapshot, license lookups, static-analysis pattern checks, Shannon-entropy secret
  detection, and supply-chain risk analysis, via `ComprehensiveSecurityAuditor`. Exits with
  a non-zero status when a finding meets `--severity`, so this is the tool suited to CI
  gating.
  ```bash
  security_audit_scanner --project <path> [--format json|yaml|html|markdown] \
      [--output PATH] [--severity info|low|medium|high|critical] \
      [--scan-dependencies] [--scan-secrets] [--scan-code] [--check-licenses] \
      [--all] [--verbose] [--exclude PATHS]
  ```

### Other binaries

| Binary | Purpose (flags: `--help`) |
|---|---|
| `dependency_vulnerability_scanner` | Dependency vulnerability / outdated-package / license scanning (`--project`, `--output`, `--format`, `--update-database`, `--check-outdated`, `--check-licenses`, `--min-severity`, `--advisory-db`, `--verbose`) |
| `longrun_analyzer` | Analyzes long-running stability/endurance test result files (`--input`, `--output`, `--format`, threshold flags, `--verbose`) |
| `stress_test_analyzer` | Analyzes stress-test result files (same flag shape as `longrun_analyzer`) |
| `performance_baseline_manager` | Creates/updates/validates performance baselines (`--results-file`, `--baseline-dir`, `--features`, `--commit-hash`, `--branch`, plus subcommand-specific flags) |

## Usage

### Basic Performance Benchmarking

`OptimizerBenchmark::run_benchmark` accepts any `FnMut(&Array1<A>, &Array1<A>) -> Array1<A>`
step function -- a plain closure, as below, or an `optirs-core` optimizer's `step` with its
`Result` handled inside the closure.

```rust
use optirs_bench::OptimizerBenchmark;
use scirs2_core::ndarray::Array1;

fn main() -> optirs_bench::Result<()> {
    let mut benchmark = OptimizerBenchmark::<f64>::new();
    benchmark.add_standard_test_functions(); // Quadratic, Rosenbrock, Sphere

    // Plain gradient descent: x_{t+1} = x_t - lr * grad
    let lr = 0.01;
    benchmark.run_benchmark(
        "gradient-descent".to_string(),
        move |x: &Array1<f64>, grad: &Array1<f64>| x - &(grad * lr),
        500,  // max_iterations
        1e-6, // tolerance
    )?;

    let report = benchmark.generate_report();
    println!(
        "Ran {} benchmark result(s) across {} optimizer(s).",
        report.total_tests,
        report.optimizer_performance.len()
    );
    Ok(())
}
```

Render the same report to Markdown/plain-text/CSV with `report_templates` (this pattern
is a passing doctest on `report_templates`, given a `report: &BenchmarkReport<A>`):

```rust
use optirs_bench::report_templates::{ReportFormat, ReportTemplate, save_report};
use std::path::Path;

let markdown = ReportTemplate::new()
    .with_title("Nightly Benchmark")
    .render(&report, None, ReportFormat::Markdown);
save_report(Path::new("/tmp/benchmark_report.md"), &markdown)?;
```

For regression detection and per-run system-resource monitoring, use the
`performance-regression-detector` / `optirs-bench analyze` CLI tools and the
`system_sampler::SystemSampler` type documented above and in rustdoc -- see the
[Command-Line Tools](#command-line-tools) section for the verified, real invocations.

## Security Auditing

```rust
use optirs_bench::security_auditor::SecurityAuditor;

fn main() -> optirs_bench::Result<()> {
    // `new()` is fallible: it validates its own default `SecurityAuditConfig`.
    let mut auditor = SecurityAuditor::new()?;
    auditor.run_complete_audit()?;
    println!("{}", auditor.generate_report());
    Ok(())
}
```

`SecurityAuditor::lightweight()` / `::comprehensive()` select a lighter or fuller preset
config; `auditor.export_json()` renders the results as JSON instead of the text report.

For dependency/license/static-analysis auditing scoped to a Cargo project on disk (the
engine behind the `security_audit_scanner` binary above), use
`comprehensive_security_auditor::ComprehensiveSecurityAuditor::audit_project(path)` --
see rustdoc for its full, real API.

## Continuous Integration Integration

### GitHub Actions

Two commands here actually gate the build with a real non-zero exit status:
`optirs-bench analyze` (regresses beyond `--max-regression`) and `security_audit_scanner`
(a finding meets `--severity`). `performance-regression-detector` is reporting-only (see
above) and is intentionally not used as a gate here.

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
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run the benchmark suite
        run: cargo run --release --bin optirs-bench -- benchmark

      - name: Fail on a benchmark regression
        run: cargo run --release --bin optirs-bench -- analyze --max-regression 10

      - name: Fail on a high/critical security finding
        run: |
          cargo run --release --bin security_audit_scanner -- \
            --project . --all --severity high
```

## Platform Support

| Platform | CPU Profiling | Memory Profiling | Security Scanning |
|----------|---------------|-------------------|--------------------|
| Linux    | Yes (`sysinfo`) | Yes (`sysinfo`) | Yes |
| macOS    | Yes (`sysinfo`) | Yes (`sysinfo`) | Yes |
| Windows  | Yes (`sysinfo`) | Yes (`sysinfo`) | Yes |

This is a native crate (binaries + library); it does not target `wasm32`. There is no GPU
utilization/memory telemetry on any platform -- see Resource Utilization above. Browser/WASM
optimizer bindings are a separate crate, `optirs-wasm`.

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS repository for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0.
