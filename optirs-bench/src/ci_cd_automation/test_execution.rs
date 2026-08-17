// Test Execution and Management
//
// This module provides comprehensive test execution capabilities for CI/CD automation,
// including test suite management, performance test cases, test execution contexts,
// and result handling.

use crate::error::{OptimError, Result};
use crate::performance_regression_detector::{
    EnvironmentInfo, MetricType, MetricValue, PerformanceMeasurement, TestConfiguration,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use crate::system_sampler::SystemSampler;

use super::config::CiCdPlatform;

/// Performance test suite for CI/CD automation
#[derive(Debug, Clone)]
pub struct PerformanceTestSuite {
    /// Test cases in the suite
    pub test_cases: Vec<PerformanceTestCase>,
    /// Test suite configuration
    pub config: TestSuiteConfig,
    /// Execution context
    pub context: Option<CiCdContext>,
    /// Test results
    pub results: Vec<CiCdTestResult>,
}

/// Individual performance test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestCase {
    /// Test case name
    pub name: String,
    /// Test category
    pub category: TestCategory,
    /// Test executor type
    pub executor: TestExecutor,
    /// Test parameters
    pub parameters: HashMap<String, String>,
    /// Expected baseline metrics
    pub baseline: Option<BaselineMetrics>,
    /// Test timeout in seconds
    pub timeout: Option<u64>,
    /// Number of iterations
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Test dependencies
    pub dependencies: Vec<String>,
    /// Test tags for filtering
    pub tags: Vec<String>,
    /// Test environment requirements
    pub environment_requirements: EnvironmentRequirements,
    /// Custom test configuration
    pub custom_config: HashMap<String, String>,
}

/// Test categories for organization and filtering
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TestCategory {
    /// Unit performance tests
    Unit,
    /// Integration performance tests
    Integration,
    /// System-wide performance tests
    System,
    /// Load testing
    Load,
    /// Stress testing
    Stress,
    /// Endurance testing
    Endurance,
    /// Spike testing
    Spike,
    /// Volume testing
    Volume,
    /// Security performance tests
    Security,
    /// Regression testing
    Regression,
    /// Benchmark testing
    Benchmark,
    /// Custom test category
    Custom(String),
}

/// Test executors for different types of performance tests
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestExecutor {
    /// Criterion.rs benchmark executor
    Criterion,
    /// Custom benchmark executor
    Custom(String),
    /// Shell command executor
    Shell,
    /// Docker container executor
    Docker { image: String, options: Vec<String> },
    /// External tool executor
    ExternalTool { tool: String, args: Vec<String> },
    /// Rust binary executor
    RustBinary { binary: String, args: Vec<String> },
    /// Python script executor
    Python { script: String, args: Vec<String> },
}

/// Test suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteConfig {
    /// Include unit tests
    pub include_unit: bool,
    /// Include integration tests
    pub include_integration: bool,
    /// Include stress tests
    pub include_stress: bool,
    /// Include load tests
    pub include_load: bool,
    /// Include security tests
    pub include_security: bool,
    /// Test timeout in seconds
    pub default_timeout: u64,
    /// Parallel execution settings
    pub parallel_execution: ParallelExecutionConfig,
    /// Resource monitoring
    pub resource_monitoring: ResourceMonitoringConfig,
    /// Test filtering
    pub filtering: TestFilteringConfig,
    /// Retry configuration
    pub retry_config: TestRetryConfig,
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionConfig {
    /// Enable parallel execution
    pub enabled: bool,
    /// Maximum concurrent tests
    pub max_concurrent: usize,
    /// Thread pool size
    pub thread_pool_size: Option<usize>,
    /// Test grouping strategy
    pub grouping_strategy: TestGroupingStrategy,
    /// Resource allocation per test
    pub resource_allocation: ResourceAllocationConfig,
}

/// Test grouping strategies for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestGroupingStrategy {
    /// Group by test category
    ByCategory,
    /// Group by execution time
    ByExecutionTime,
    /// Group by resource requirements
    ByResourceRequirements,
    /// No grouping (random)
    None,
    /// Custom grouping
    Custom(String),
}

/// Resource allocation configuration per test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationConfig {
    /// CPU cores per test
    pub cpu_cores: Option<usize>,
    /// Memory limit per test (MB)
    pub memory_limit_mb: Option<usize>,
    /// Disk space limit per test (MB)
    pub disk_limit_mb: Option<usize>,
    /// Network bandwidth limit per test (MB/s)
    pub network_limit_mbps: Option<f64>,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Enable CPU monitoring
    pub monitor_cpu: bool,
    /// Enable memory monitoring
    pub monitor_memory: bool,
    /// Enable disk I/O monitoring
    pub monitor_disk_io: bool,
    /// Enable network monitoring
    pub monitor_network: bool,
    /// Monitoring frequency in milliseconds
    pub monitoring_frequency_ms: u64,
    /// Resource alert thresholds
    pub alert_thresholds: ResourceAlertThresholds,
}

/// Resource alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlertThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_threshold: f64,
    /// Memory usage threshold (percentage)
    pub memory_threshold: f64,
    /// Disk usage threshold (percentage)
    pub disk_threshold: f64,
    /// Network usage threshold (MB/s)
    pub network_threshold: f64,
}

/// Test filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestFilteringConfig {
    /// Include specific test categories
    pub include_categories: Vec<TestCategory>,
    /// Exclude specific test categories
    pub exclude_categories: Vec<TestCategory>,
    /// Include tests with specific tags
    pub include_tags: Vec<String>,
    /// Exclude tests with specific tags
    pub exclude_tags: Vec<String>,
    /// Test name patterns to include
    pub include_patterns: Vec<String>,
    /// Test name patterns to exclude
    pub exclude_patterns: Vec<String>,
    /// Only run tests that match platform
    pub platform_specific: bool,
}

/// Test retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRetryConfig {
    /// Enable test retries
    pub enabled: bool,
    /// Maximum number of retries
    pub max_retries: u32,
    /// Delay between retries in seconds
    pub retry_delay_sec: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Retry on specific failure types
    pub retry_on_failures: Vec<TestFailureType>,
}

/// Types of test failures that can trigger retries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestFailureType {
    /// Timeout failures
    Timeout,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Network failures
    Network,
    /// Transient system errors
    TransientError,
    /// Environment setup failures
    EnvironmentSetup,
    /// All failure types
    All,
}

/// Environment requirements for test execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentRequirements {
    /// Required operating system
    pub os: Option<String>,
    /// Required architecture
    pub architecture: Option<String>,
    /// Minimum CPU cores
    pub min_cpu_cores: Option<usize>,
    /// Minimum memory in MB
    pub min_memory_mb: Option<usize>,
    /// Required environment variables
    pub required_env_vars: Vec<String>,
    /// Required software dependencies
    pub dependencies: Vec<SoftwareDependency>,
    /// Required network access
    pub network_access: NetworkAccessRequirements,
    /// Required file system permissions
    pub file_permissions: Vec<FilePermissionRequirement>,
}

/// Software dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareDependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub version: Option<String>,
    /// Installation source
    pub source: DependencySource,
    /// Optional installation
    pub optional: bool,
}

/// Dependency installation sources
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DependencySource {
    /// System package manager
    System,
    /// Cargo for Rust crates
    Cargo,
    /// npm for Node.js packages
    Npm,
    /// pip for Python packages
    Pip,
    /// apt for Debian/Ubuntu
    Apt,
    /// yum for RedHat/CentOS
    Yum,
    /// brew for macOS
    Homebrew,
    /// Custom installation script
    Custom(String),
}

/// Network access requirements
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkAccessRequirements {
    /// Requires internet access
    pub internet_access: bool,
    /// Required network ports
    pub required_ports: Vec<u16>,
    /// Required domains/hosts
    pub required_hosts: Vec<String>,
    /// Maximum allowed latency in ms
    pub max_latency_ms: Option<u64>,
    /// Minimum required bandwidth in MB/s
    pub min_bandwidth_mbps: Option<f64>,
}

/// File permission requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePermissionRequirement {
    /// File or directory path
    pub path: String,
    /// Required permissions (Unix-style)
    pub permissions: u32,
    /// Must be readable
    pub readable: bool,
    /// Must be writable
    pub writable: bool,
    /// Must be executable
    pub executable: bool,
}

/// Baseline metrics for performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    /// Execution time baseline
    pub execution_time: Option<MetricBaseline>,
    /// Memory usage baseline
    pub memory_usage: Option<MetricBaseline>,
    /// CPU usage baseline
    pub cpu_usage: Option<MetricBaseline>,
    /// Throughput baseline
    pub throughput: Option<MetricBaseline>,
    /// Custom metrics baselines
    pub custom_metrics: HashMap<String, MetricBaseline>,
}

/// Individual metric baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBaseline {
    /// Expected value
    pub expected_value: f64,
    /// Acceptable variance (percentage)
    pub variance_threshold: f64,
    /// Upper bound (fail if exceeded)
    pub upper_bound: Option<f64>,
    /// Lower bound (warn if below)
    pub lower_bound: Option<f64>,
    /// Unit of measurement
    pub unit: String,
}

/// CI/CD test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdTestResult {
    /// Test case name
    pub test_name: String,
    /// Test execution status
    pub status: TestExecutionStatus,
    /// Execution start time
    pub start_time: SystemTime,
    /// Execution end time
    pub end_time: Option<SystemTime>,
    /// Test duration
    pub duration: Option<Duration>,
    /// Performance measurements
    pub measurements: Vec<PerformanceMeasurement>,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Test output/logs
    pub output: String,
    /// Resource usage during test
    pub resource_usage: ResourceUsageReport,
    /// Test metadata
    pub metadata: TestExecutionMetadata,
    /// Regression analysis results
    pub regression_analysis: Option<RegressionAnalysisResult>,
}

/// Test execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TestExecutionStatus {
    /// Test passed successfully
    Passed,
    /// Test failed
    Failed,
    /// Test was skipped
    Skipped,
    /// Test timed out
    TimedOut,
    /// Test encountered an error
    Error,
    /// Test execution is pending
    Pending,
    /// Test execution is in progress
    Running,
}

/// Resource usage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageReport {
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Average CPU usage percentage
    pub avg_cpu_percent: f64,
    /// Peak CPU usage percentage
    pub peak_cpu_percent: f64,
    /// Total disk I/O in MB
    pub disk_io_mb: f64,
    /// Network usage in MB
    pub network_usage_mb: f64,
    /// Detailed resource timeline
    pub timeline: Vec<ResourceSnapshot>,
}

/// Resource usage snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: SystemTime,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// CPU usage percentage
    pub cpu_percent: f64,
    /// Disk I/O rate in MB/s
    pub disk_io_mbps: f64,
    /// Network I/O rate in MB/s
    pub network_mbps: f64,
}

/// Background sampler that records real per-process resource usage while a test
/// body executes.
///
/// Backed by [`SystemSampler`], which reads genuine OS counters (resident set
/// size, process CPU%, cumulative disk bytes). A dedicated thread appends a
/// [`ResourceSnapshot`] every `frequency_ms` until [`ResourceMonitor::stop`]
/// joins it and returns the captured timeline. Nothing is fabricated: the
/// disk rate is differenced from real byte counters, and metrics that are not
/// portably observable per-process (network throughput) are left at zero.
struct ResourceMonitor {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<Vec<ResourceSnapshot>>>,
}

impl ResourceMonitor {
    /// Start monitoring, sampling every `frequency_ms` milliseconds (clamped to
    /// a small floor so a misconfigured `0` cannot busy-spin). Returns an error
    /// when a system sampler cannot be constructed on this platform, letting the
    /// caller proceed with an empty report rather than fabricated numbers.
    fn start(frequency_ms: u64) -> Result<Self> {
        let sampler = SystemSampler::new()?;
        let interval = Duration::from_millis(frequency_ms.max(10));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("optirs-resource-monitor".to_string())
            .spawn(move || Self::run(sampler, interval, thread_stop))
            .map_err(|e| {
                OptimError::ResourceUnavailable(format!("failed to spawn resource monitor: {e}"))
            })?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }

    /// Sampling loop; owns the sampler and returns every captured snapshot.
    fn run(
        sampler: SystemSampler,
        interval: Duration,
        stop: Arc<AtomicBool>,
    ) -> Vec<ResourceSnapshot> {
        let mut snapshots = Vec::new();
        // Prime the CPU delta: the first `sample_process` yields `cpu_percent
        // == None` because a rate needs two refreshes.
        sampler.refresh();
        let mut prev_disk: Option<(u64, Instant)> = None;
        while !stop.load(Ordering::Relaxed) {
            // Sleep the full sampling interval, but in short slices so `stop`
            // is observed promptly: waiting out a whole (possibly one second)
            // interval would delay every join by that much. The sampling
            // cadence itself is unchanged — only the responsiveness of the
            // stop flag improves, so the disk-rate differencing below still
            // sees `interval`-spaced samples.
            if !Self::sleep_interruptibly(interval, &stop) {
                break;
            }
            sampler.refresh();
            let sample = match sampler.sample_process() {
                Ok(sample) => sample,
                Err(_) => continue,
            };
            let disk_bytes = sample
                .disk_read_bytes
                .saturating_add(sample.disk_written_bytes);
            let disk_io_mbps = match prev_disk {
                Some((prev_bytes, prev_instant)) => {
                    let secs = sample
                        .timestamp
                        .saturating_duration_since(prev_instant)
                        .as_secs_f64();
                    if secs > 0.0 {
                        (disk_bytes.saturating_sub(prev_bytes) as f64 / 1_000_000.0) / secs
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            };
            prev_disk = Some((disk_bytes, sample.timestamp));
            snapshots.push(ResourceSnapshot {
                timestamp: SystemTime::now(),
                memory_mb: sample.rss_bytes as f64 / 1_000_000.0,
                cpu_percent: sample.cpu_percent.unwrap_or(0.0),
                disk_io_mbps,
                network_mbps: 0.0,
            });
        }
        snapshots
    }

    /// Longest uninterrupted nap the sampling thread takes; the stop flag is
    /// re-checked at least this often.
    const STOP_POLL_TICK: Duration = Duration::from_millis(25);

    /// Sleep for `total`, waking every [`Self::STOP_POLL_TICK`] to re-check the
    /// stop flag. Returns `false` as soon as a stop is requested, so the caller
    /// can leave the sampling loop without waiting out the remaining time.
    fn sleep_interruptibly(total: Duration, stop: &Arc<AtomicBool>) -> bool {
        let mut remaining = total;
        while !remaining.is_zero() {
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            let slice = remaining.min(Self::STOP_POLL_TICK);
            thread::sleep(slice);
            remaining -= slice;
        }
        !stop.load(Ordering::Relaxed)
    }

    /// Stop sampling, join the thread, and return the captured timeline.
    fn stop(mut self) -> Vec<ResourceSnapshot> {
        self.stop.store(true, Ordering::Relaxed);
        match self.handle.take() {
            Some(handle) => handle.join().unwrap_or_default(),
            None => Vec::new(),
        }
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        // Ensure the sampling thread is never leaked if the monitor is dropped
        // without an explicit `stop` (e.g. on an early return or panic).
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Test execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionMetadata {
    /// Test executor used
    pub executor: TestExecutor,
    /// Test parameters
    pub parameters: HashMap<String, String>,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Git information
    pub git_info: Option<GitInfo>,
    /// CI/CD context
    pub ci_context: Option<CiCdContext>,
    /// Test configuration
    pub test_config: TestConfiguration,
}

/// Git repository information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInfo {
    /// Current commit hash
    pub commit_hash: String,
    /// Current branch name
    pub branch: String,
    /// Commit message
    pub commit_message: Option<String>,
    /// Commit author
    pub author: Option<String>,
    /// Commit timestamp
    pub commit_time: Option<SystemTime>,
    /// Repository URL
    pub repository_url: Option<String>,
    /// Is working directory clean
    pub is_clean: bool,
}

/// CI/CD execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdContext {
    /// CI/CD platform
    pub platform: CiCdPlatform,
    /// Build/job ID
    pub build_id: String,
    /// Build number
    pub build_number: Option<u64>,
    /// Trigger event
    pub trigger: TriggerEvent,
    /// Environment variables
    pub environment_vars: HashMap<String, String>,
    /// Build URL
    pub build_url: Option<String>,
    /// Pull request information
    pub pull_request: Option<PullRequestInfo>,
    /// Triggered by user
    pub triggered_by: Option<String>,
}

/// Events that can trigger CI/CD execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TriggerEvent {
    /// Triggered by code push
    Push,
    /// Triggered by pull request
    PullRequest,
    /// Triggered by release/tag
    Release,
    /// Triggered by scheduled event
    Schedule,
    /// Manually triggered
    Manual,
    /// Triggered by API call
    Api,
    /// Triggered by webhook
    Webhook,
}

/// Pull request information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequestInfo {
    /// Pull request number
    pub number: u64,
    /// Source branch
    pub source_branch: String,
    /// Target branch
    pub target_branch: String,
    /// Pull request title
    pub title: String,
    /// Pull request author
    pub author: String,
    /// Pull request URL
    pub url: Option<String>,
}

/// Regression analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysisResult {
    /// Whether regression was detected
    pub regression_detected: bool,
    /// Confidence level of the detection
    pub confidence: f64,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Performance change percentage
    pub performance_change_percent: f64,
    /// Statistical significance
    pub statistical_significance: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl PerformanceTestSuite {
    /// Create a new performance test suite
    pub fn new(config: TestSuiteConfig) -> Result<Self> {
        Ok(Self {
            test_cases: Vec::new(),
            config,
            context: None,
            results: Vec::new(),
        })
    }

    /// Add a test case to the suite
    pub fn add_test_case(&mut self, test_case: PerformanceTestCase) {
        self.test_cases.push(test_case);
    }

    /// Set the execution context
    pub fn set_context(&mut self, context: CiCdContext) {
        self.context = Some(context);
    }

    /// Execute all test cases in the suite
    pub fn execute(&mut self) -> Result<Vec<CiCdTestResult>> {
        // Filter test cases based on configuration
        let filtered_tests = self.filter_test_cases()?;

        // Execute tests based on parallel configuration
        let results = if self.config.parallel_execution.enabled {
            self.execute_parallel(&filtered_tests)?
        } else {
            self.execute_sequential(&filtered_tests)?
        };

        self.results = results.clone();
        Ok(results)
    }

    /// Filter test cases based on configuration
    fn filter_test_cases(&self) -> Result<Vec<&PerformanceTestCase>> {
        let mut filtered = Vec::new();

        for test_case in &self.test_cases {
            if self.should_include_test_case(test_case) {
                filtered.push(test_case);
            }
        }

        Ok(filtered)
    }

    /// Check if a test case should be included based on filtering configuration
    fn should_include_test_case(&self, test_case: &PerformanceTestCase) -> bool {
        let filtering = &self.config.filtering;

        // Check category inclusion
        if !filtering.include_categories.is_empty()
            && !filtering.include_categories.contains(&test_case.category)
        {
            return false;
        }

        // Check category exclusion
        if filtering.exclude_categories.contains(&test_case.category) {
            return false;
        }

        // Check tag inclusion
        if !filtering.include_tags.is_empty() {
            let has_included_tag = filtering
                .include_tags
                .iter()
                .any(|tag| test_case.tags.contains(tag));
            if !has_included_tag {
                return false;
            }
        }

        // Check tag exclusion
        for excluded_tag in &filtering.exclude_tags {
            if test_case.tags.contains(excluded_tag) {
                return false;
            }
        }

        // Check name patterns (simplified regex matching)
        if !filtering.include_patterns.is_empty() {
            let matches_pattern = filtering
                .include_patterns
                .iter()
                .any(|pattern| test_case.name.contains(pattern));
            if !matches_pattern {
                return false;
            }
        }

        for excluded_pattern in &filtering.exclude_patterns {
            if test_case.name.contains(excluded_pattern) {
                return false;
            }
        }

        true
    }

    /// Execute test cases sequentially
    fn execute_sequential(
        &self,
        test_cases: &[&PerformanceTestCase],
    ) -> Result<Vec<CiCdTestResult>> {
        let mut results = Vec::with_capacity(test_cases.len());

        for test_case in test_cases {
            // Execution currently continues even on failure/error; making this
            // configurable (fail-fast) is left as a future enhancement.
            results.push(self.execute_single_test(test_case)?);
        }

        Ok(results)
    }

    /// Execute test cases with real, bounded parallelism.
    ///
    /// Test cases are processed in chunks of `max_concurrent`; within each
    /// chunk every test runs on its own scoped thread (`std::thread::scope`),
    /// so the configured concurrency is genuinely respected rather than merely
    /// documented. Results are collected back in the original test order.
    ///
    /// `execute_single_test` only borrows `&self` immutably and returns an
    /// owned result, so no per-thread config cloning is required; the shared
    /// `&self` is safe to borrow across the scoped threads.
    fn execute_parallel(&self, test_cases: &[&PerformanceTestCase]) -> Result<Vec<CiCdTestResult>> {
        // `chunks(0)` panics, and `max_concurrent` is an unvalidated `usize`.
        let max_concurrent = self.config.parallel_execution.max_concurrent.max(1);
        let mut results = Vec::with_capacity(test_cases.len());

        for chunk in test_cases.chunks(max_concurrent) {
            let chunk_results = thread::scope(|scope| -> Result<Vec<CiCdTestResult>> {
                // Spawn one thread per test in the chunk for true concurrency.
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|test_case| scope.spawn(move || self.execute_single_test(test_case)))
                    .collect();

                // Join in spawn order so results keep the original ordering.
                let mut chunk_out = Vec::with_capacity(handles.len());
                for handle in handles {
                    match handle.join() {
                        Ok(result) => chunk_out.push(result?),
                        Err(_) => {
                            return Err(OptimError::ComputationError(
                                "a parallel test-execution thread panicked".to_string(),
                            ))
                        }
                    }
                }
                Ok(chunk_out)
            })?;

            results.extend(chunk_results);
        }

        Ok(results)
    }

    /// Execute a single test case
    fn execute_single_test(&self, test_case: &PerformanceTestCase) -> Result<CiCdTestResult> {
        let start_time = SystemTime::now();
        let mut result = CiCdTestResult {
            test_name: test_case.name.clone(),
            status: TestExecutionStatus::Running,
            start_time,
            end_time: None,
            duration: None,
            measurements: Vec::new(),
            error_message: None,
            output: String::new(),
            resource_usage: ResourceUsageReport::default(),
            metadata: self.create_test_metadata(test_case)?,
            regression_analysis: None,
        };

        // Check environment requirements
        if let Err(e) = self.check_environment_requirements(&test_case.environment_requirements) {
            result.status = TestExecutionStatus::Error;
            result.error_message = Some(format!("Environment requirements not met: {}", e));
            result.end_time = Some(SystemTime::now());
            return Ok(result);
        }

        // Start real resource monitoring for the duration of the test body. If
        // a sampler cannot be created on this platform we proceed without a
        // timeline rather than fabricating numbers.
        let monitor =
            match ResourceMonitor::start(self.config.resource_monitoring.monitoring_frequency_ms) {
                Ok(monitor) => Some(monitor),
                Err(e) => {
                    log::warn!("resource monitoring unavailable; report will be empty: {e}");
                    None
                }
            };

        // Execute the test based on its executor type
        match self.execute_test_by_executor(test_case) {
            Ok((measurements, output)) => {
                result.status = TestExecutionStatus::Passed;
                result.measurements = measurements;
                result.output = output;
            }
            Err(e) => {
                result.status = TestExecutionStatus::Failed;
                result.error_message = Some(e.to_string());
            }
        }

        // Stop monitoring and turn the real captured samples into a report.
        let snapshots = monitor.map(ResourceMonitor::stop).unwrap_or_default();

        let end_time = SystemTime::now();
        result.end_time = Some(end_time);
        result.duration = end_time.duration_since(start_time).ok();
        result.resource_usage = Self::build_resource_usage_report(snapshots);

        Ok(result)
    }

    /// Execute test based on its executor type
    fn execute_test_by_executor(
        &self,
        test_case: &PerformanceTestCase,
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        match &test_case.executor {
            TestExecutor::Criterion => self.execute_criterion_test(test_case),
            TestExecutor::Custom(cmd) => self.execute_custom_test(test_case, cmd),
            TestExecutor::Shell => self.execute_shell_test(test_case),
            TestExecutor::Docker { image, options } => {
                self.execute_docker_test(test_case, image, options)
            }
            TestExecutor::ExternalTool { tool, args } => {
                self.execute_external_tool_test(test_case, tool, args)
            }
            TestExecutor::RustBinary { binary, args } => {
                self.execute_rust_binary_test(test_case, binary, args)
            }
            TestExecutor::Python { script, args } => {
                self.execute_python_test(test_case, script, args)
            }
        }
    }

    /// Execute a Criterion.rs benchmark test.
    ///
    /// When the test case names a criterion target (parameter `bench` or
    /// `bench_target`) the real benchmark is executed with `cargo bench` and its
    /// reported timings are parsed. Without such a target there is no criterion
    /// harness to drive, so the case falls back to an in-process arithmetic loop
    /// whose *measured* wall-clock time is reported — and the output says so
    /// rather than claiming a criterion run happened.
    fn execute_criterion_test(
        &self,
        test_case: &PerformanceTestCase,
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        if let Some(target) = test_case
            .parameters
            .get("bench")
            .or_else(|| test_case.parameters.get("bench_target"))
        {
            return self.execute_cargo_bench(test_case, target);
        }

        let iterations = test_case
            .parameters
            .get("iterations")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(test_case.iterations);

        let loop_seconds = self.execute_simple_loop_benchmark(iterations);
        let measurement = self.build_time_measurement(test_case, loop_seconds);
        let output = format!(
            "no criterion target configured (set the 'bench' parameter to run one); measured an \
             in-process loop of {iterations} iterations in {loop_seconds:.9} s"
        );
        Ok((vec![measurement], output))
    }

    /// Run `cargo bench --bench <target> -- <filter>` and parse the timings
    /// criterion prints.
    ///
    /// A missing `cargo`, or a benchmark that fails to build or run, is an
    /// honest error: no measurement is invented for it.
    fn execute_cargo_bench(
        &self,
        test_case: &PerformanceTestCase,
        target: &str,
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        let filter = test_case
            .parameters
            .get("bench_filter")
            .cloned()
            .unwrap_or_else(|| test_case.name.clone());

        let output = Command::new("cargo")
            .arg("bench")
            .arg("--bench")
            .arg(target)
            .arg("--")
            .arg(&filter)
            .output()
            .map_err(|e| {
                OptimError::ResourceUnavailable(format!(
                    "cannot run criterion benchmark '{target}': failed to spawn cargo ({e})"
                ))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            return Err(OptimError::ExecutionError(format!(
                "criterion benchmark '{}' (filter '{}') failed with status {}: {}",
                target,
                filter,
                output.status,
                stderr.trim()
            )));
        }

        let combined = format!("{stdout}{stderr}");
        let measurements = self.parse_performance_output(&combined, test_case)?;
        Ok((measurements, combined))
    }

    /// Execute a small arithmetic loop and return its measured wall-clock time
    /// in seconds.
    ///
    /// [`std::hint::black_box`] is applied to the accumulator: without it the
    /// entire loop is dead code that the optimizer removes, and the
    /// "measurement" degenerates into timing an empty block.
    fn execute_simple_loop_benchmark(&self, iterations: usize) -> f64 {
        let start = Instant::now();
        let mut sum = 0u64;
        for i in 0..iterations {
            sum = std::hint::black_box(sum.wrapping_add(std::hint::black_box(i as u64)));
        }
        std::hint::black_box(sum);
        start.elapsed().as_secs_f64()
    }

    /// Execute a custom test command
    fn execute_custom_test(
        &self,
        test_case: &PerformanceTestCase,
        command: &str,
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to execute custom test: {}", e))
            })?;

        let output_str = String::from_utf8_lossy(&output.stdout).to_string();

        // Parse output for performance measurements (simplified)
        let measurements = self.parse_performance_output(&output_str, test_case)?;

        Ok((measurements, output_str))
    }

    /// Execute a shell test
    fn execute_shell_test(
        &self,
        test_case: &PerformanceTestCase,
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        let command = test_case.parameters.get("command").ok_or_else(|| {
            OptimError::InvalidConfig("Shell test requires 'command' parameter".to_string())
        })?;

        self.execute_custom_test(test_case, command)
    }

    /// Execute a Docker-based test
    fn execute_docker_test(
        &self,
        test_case: &PerformanceTestCase,
        image: &str,
        options: &[String],
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        let mut cmd = Command::new("docker");
        cmd.arg("run");

        for option in options {
            cmd.arg(option);
        }

        cmd.arg(image);

        if let Some(command) = test_case.parameters.get("command") {
            cmd.args(command.split_whitespace());
        }

        let output = cmd.output().map_err(|e| {
            OptimError::InvalidConfig(format!("Failed to execute Docker test: {}", e))
        })?;

        let output_str = String::from_utf8_lossy(&output.stdout).to_string();
        let measurements = self.parse_performance_output(&output_str, test_case)?;

        Ok((measurements, output_str))
    }

    /// Execute an external tool test
    fn execute_external_tool_test(
        &self,
        test_case: &PerformanceTestCase,
        tool: &str,
        args: &[String],
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        let mut cmd = Command::new(tool);
        cmd.args(args);

        let output = cmd.output().map_err(|e| {
            OptimError::InvalidConfig(format!("Failed to execute external tool: {}", e))
        })?;

        let output_str = String::from_utf8_lossy(&output.stdout).to_string();
        let measurements = self.parse_performance_output(&output_str, test_case)?;

        Ok((measurements, output_str))
    }

    /// Execute a Rust binary test
    fn execute_rust_binary_test(
        &self,
        test_case: &PerformanceTestCase,
        binary: &str,
        args: &[String],
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        let mut cmd = Command::new("cargo");
        cmd.arg("run").arg("--bin").arg(binary).arg("--");
        cmd.args(args);

        let output = cmd.output().map_err(|e| {
            OptimError::InvalidConfig(format!("Failed to execute Rust binary: {}", e))
        })?;

        let output_str = String::from_utf8_lossy(&output.stdout).to_string();
        let measurements = self.parse_performance_output(&output_str, test_case)?;

        Ok((measurements, output_str))
    }

    /// Execute a Python script test
    fn execute_python_test(
        &self,
        test_case: &PerformanceTestCase,
        script: &str,
        args: &[String],
    ) -> Result<(Vec<PerformanceMeasurement>, String)> {
        let mut cmd = Command::new("python");
        cmd.arg(script);
        cmd.args(args);

        let output = cmd.output().map_err(|e| {
            OptimError::InvalidConfig(format!("Failed to execute Python script: {}", e))
        })?;

        let output_str = String::from_utf8_lossy(&output.stdout).to_string();
        let measurements = self.parse_performance_output(&output_str, test_case)?;

        Ok((measurements, output_str))
    }

    /// Parse performance output into measurements.
    ///
    /// Only timings that are actually present in the output become
    /// measurements, and their units are honoured: `ms`/`us`/`ns` are converted
    /// to seconds instead of being stripped, which previously reported every
    /// millisecond timing as if it were whole seconds (a 1000x error straight
    /// into regression analysis).
    ///
    /// When the output contains no recognisable timing an **empty** vector is
    /// returned. Synthesising a one-second measurement, as this used to do, fed
    /// a fabricated data point into the baseline and regression detectors;
    /// "nothing was measured" is the honest answer and callers record it as an
    /// empty measurement set.
    fn parse_performance_output(
        &self,
        output: &str,
        test_case: &PerformanceTestCase,
    ) -> Result<Vec<PerformanceMeasurement>> {
        let mut measurements = Vec::new();

        for line in output.lines() {
            if let Some(duration_seconds) = Self::extract_duration_seconds(line) {
                measurements.push(self.build_time_measurement(test_case, duration_seconds));
            }
        }

        Ok(measurements)
    }

    /// Build a single execution-time measurement from a genuinely observed
    /// duration (in seconds).
    fn build_time_measurement(
        &self,
        test_case: &PerformanceTestCase,
        duration_seconds: f64,
    ) -> PerformanceMeasurement {
        let mut metrics = HashMap::new();
        metrics.insert(
            crate::performance_regression_detector::MetricType::ExecutionTime,
            MetricValue {
                value: duration_seconds,
                std_dev: None,
                sample_count: 1,
                min_value: duration_seconds,
                max_value: duration_seconds,
                percentiles: None,
            },
        );

        PerformanceMeasurement {
            timestamp: SystemTime::now(),
            commithash: "unknown".to_string(),
            branch: "unknown".to_string(),
            build_config: "unknown".to_string(),
            environment: crate::performance_regression_detector::EnvironmentInfo {
                os: std::env::consts::OS.to_string(),
                cpu_model: std::env::consts::ARCH.to_string(),
                cpu_cores: num_cpus::get(),
                total_memory_mb: 0,
                gpu_info: None,
                compiler_version: "unknown".to_string(),
                rust_version: "unknown".to_string(),
                env_vars: HashMap::new(),
            },
            metrics,
            test_config: crate::performance_regression_detector::TestConfiguration {
                test_name: test_case.name.clone(),
                parameters: test_case.parameters.clone(),
                dataset_size: None,
                iterations: Some(1),
                batch_size: None,
                precision: "f64".to_string(),
            },
            metadata: HashMap::new(),
        }
    }

    /// Extract a duration **in seconds** from a text line such as
    /// `time: 1.5 ms`, `duration = 250us`, or criterion's
    /// `time:   [1.2340 ms 1.2350 ms 1.2360 ms]`.
    ///
    /// Returns `None` when the line carries no recognisable timing, or when the
    /// unit is not a known time unit — an unrecognised unit is skipped rather
    /// than assumed to be seconds.
    fn extract_duration_seconds(line: &str) -> Option<f64> {
        let lower = line.to_ascii_lowercase();

        for label in ["duration", "elapsed", "runtime", "time"] {
            let mut search_from = 0usize;
            while let Some(offset) = lower[search_from..].find(label) {
                let start = search_from + offset;
                let after = start + label.len();
                // Require a word boundary before the label so `lifetime` or
                // `downtime` do not masquerade as a `time` metric.
                let boundary_ok = start == 0
                    || !matches!(lower.as_bytes()[start - 1], b'a'..=b'z' | b'0'..=b'9' | b'_');
                if boundary_ok {
                    if let Some(seconds) = Self::parse_duration_after(&lower[after..]) {
                        return Some(seconds);
                    }
                }
                search_from = after;
            }
        }

        None
    }

    /// Parse `value unit` (in seconds) out of the text following a duration
    /// label. `rest` must already be lowercased.
    fn parse_duration_after(rest: &str) -> Option<f64> {
        let trimmed = rest
            .trim_start_matches(|c: char| matches!(c, ':' | '=' | '[' | '(') || c.is_whitespace());

        let mut number_end = 0usize;
        for (idx, ch) in trimmed.char_indices() {
            if ch.is_ascii_digit() || ch == '.' || ((ch == '-' || ch == '+') && idx == 0) {
                number_end = idx + ch.len_utf8();
            } else {
                break;
            }
        }
        if number_end == 0 {
            return None;
        }

        let value: f64 = trimmed[..number_end].parse().ok()?;
        let unit: String = trimmed[number_end..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphabetic() || *c == 'µ')
            .collect();

        match unit.as_str() {
            "ns" | "nsec" | "nsecs" | "nanos" | "nanoseconds" => Some(value * 1e-9),
            "us" | "µs" | "usec" | "usecs" | "micros" | "microseconds" => Some(value * 1e-6),
            "ms" | "msec" | "msecs" | "millis" | "milliseconds" => Some(value * 1e-3),
            "m" | "min" | "mins" | "minutes" => Some(value * 60.0),
            "" | "s" | "sec" | "secs" | "second" | "seconds" => Some(value),
            _ => None,
        }
    }

    /// Check environment requirements
    fn check_environment_requirements(&self, requirements: &EnvironmentRequirements) -> Result<()> {
        // Check OS requirement
        if let Some(required_os) = &requirements.os {
            let current_os = std::env::consts::OS;
            if current_os != required_os {
                return Err(OptimError::InvalidConfig(format!(
                    "Required OS: {}, Current OS: {}",
                    required_os, current_os
                )));
            }
        }

        // Check architecture requirement
        if let Some(required_arch) = &requirements.architecture {
            let current_arch = std::env::consts::ARCH;
            if current_arch != required_arch {
                return Err(OptimError::InvalidConfig(format!(
                    "Required architecture: {}, Current architecture: {}",
                    required_arch, current_arch
                )));
            }
        }

        // Check CPU cores requirement
        if let Some(min_cores) = requirements.min_cpu_cores {
            let available_cores = num_cpus::get();
            if available_cores < min_cores {
                return Err(OptimError::InvalidConfig(format!(
                    "Required CPU cores: {}, Available: {}",
                    min_cores, available_cores
                )));
            }
        }

        // Check environment variables
        for env_var in &requirements.required_env_vars {
            if std::env::var(env_var).is_err() {
                return Err(OptimError::InvalidConfig(format!(
                    "Required environment variable not set: {}",
                    env_var
                )));
            }
        }

        Ok(())
    }

    /// Create test execution metadata
    fn create_test_metadata(
        &self,
        test_case: &PerformanceTestCase,
    ) -> Result<TestExecutionMetadata> {
        Ok(TestExecutionMetadata {
            executor: test_case.executor.clone(),
            parameters: test_case.parameters.clone(),
            environment: self.gather_environment_info()?,
            git_info: self.gather_git_info().ok(),
            ci_context: self.context.clone(),
            test_config: crate::performance_regression_detector::TestConfiguration {
                test_name: test_case.name.clone(),
                parameters: test_case.parameters.clone(),
                dataset_size: None,
                iterations: Some(test_case.iterations),
                batch_size: None,
                precision: "f64".to_string(),
            },
        })
    }

    /// Gather environment information from real platform measurements.
    ///
    /// CPU model, core count, installed memory and the toolchain version all
    /// come from [`SystemSampler`] (which reads genuine OS counters and
    /// `rustc -vV`) instead of the previous `"unknown"`/`0` placeholders. Values
    /// the platform genuinely does not expose stay `"unknown"`/`0` rather than
    /// being invented.
    fn gather_environment_info(
        &self,
    ) -> Result<crate::performance_regression_detector::EnvironmentInfo> {
        let platform = SystemSampler::platform_info_static();

        let os = match &platform.os_version {
            Some(version) => format!("{} {}", platform.os_name, version),
            None => platform.os_name.clone(),
        };
        let cpu_model = platform
            .cpu_brand
            .clone()
            .unwrap_or_else(|| platform.arch.to_string());
        let rust_version = platform
            .rustc_version
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        // Installed memory needs a live sampler; if one cannot be created on
        // this platform the field stays 0, which by convention means
        // "not measured".
        let total_memory_mb = match SystemSampler::new() {
            Ok(sampler) => (sampler.sample_system().total_memory_bytes / 1_000_000) as usize,
            Err(e) => {
                log::debug!("total memory unavailable: {e}");
                0
            }
        };

        Ok(crate::performance_regression_detector::EnvironmentInfo {
            os,
            cpu_model,
            cpu_cores: platform.logical_core_count,
            total_memory_mb,
            gpu_info: None,
            compiler_version: rust_version.clone(),
            rust_version,
            env_vars: std::env::vars().collect(),
        })
    }

    /// Gather Git repository information for the current working directory.
    ///
    /// Each `git` invocation is checked for a successful exit status: a command
    /// that ran but failed (for example outside a repository) yields
    /// `"unknown"`, not its empty stdout. `is_clean` is determined by an actual
    /// `git status --porcelain` run; when that cannot be established it is
    /// reported as `false`, because claiming a verified-clean tree we never
    /// verified is exactly the fabrication this replaces.
    fn gather_git_info(&self) -> Result<GitInfo> {
        fn git_field(args: &[&str]) -> String {
            match Command::new("git").args(args).output() {
                Ok(output) if output.status.success() => {
                    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if value.is_empty() {
                        "unknown".to_string()
                    } else {
                        value
                    }
                }
                _ => "unknown".to_string(),
            }
        }

        let commit_hash = git_field(&["rev-parse", "HEAD"]);
        let branch = git_field(&["rev-parse", "--abbrev-ref", "HEAD"]);
        // `%s` (subject line) rather than `%B` (full body): the subject is what
        // CI metadata wants, and a single-line value cannot smuggle newlines
        // into the CSV/HTML reports built from this metadata.
        let commit_message = match git_field(&["log", "-1", "--pretty=%s"]).as_str() {
            "unknown" => None,
            message => Some(message.to_string()),
        };
        let author = match git_field(&["log", "-1", "--pretty=%an"]).as_str() {
            "unknown" => None,
            name => Some(name.to_string()),
        };
        let repository_url = match git_field(&["config", "--get", "remote.origin.url"]).as_str() {
            "unknown" => None,
            url => Some(url.to_string()),
        };

        let is_clean = crate::system_sampler::git_is_clean(".").unwrap_or_else(|e| {
            log::debug!("working-tree cleanliness could not be determined: {e}");
            false
        });

        Ok(GitInfo {
            commit_hash,
            branch,
            commit_message,
            author,
            commit_time: None,
            repository_url,
            is_clean,
        })
    }

    /// Build a resource-usage report from real captured snapshots.
    ///
    /// Every field is derived from the timeline of real [`SystemSampler`]
    /// measurements; there are no placeholder constants. An empty timeline
    /// (monitoring unavailable) yields the honest all-zero default rather than
    /// fabricated values.
    fn build_resource_usage_report(snapshots: Vec<ResourceSnapshot>) -> ResourceUsageReport {
        if snapshots.is_empty() {
            return ResourceUsageReport::default();
        }

        let count = snapshots.len() as f64;
        let peak_memory_mb = snapshots
            .iter()
            .map(|snapshot| snapshot.memory_mb)
            .fold(0.0_f64, f64::max);
        let avg_cpu_percent = snapshots
            .iter()
            .map(|snapshot| snapshot.cpu_percent)
            .sum::<f64>()
            / count;
        let peak_cpu_percent = snapshots
            .iter()
            .map(|snapshot| snapshot.cpu_percent)
            .fold(0.0_f64, f64::max);

        // Total disk I/O over the run: integrate the per-snapshot rates across
        // the real time intervals between successive snapshots.
        let mut disk_io_mb = 0.0_f64;
        for pair in snapshots.windows(2) {
            let interval = match pair[1].timestamp.duration_since(pair[0].timestamp) {
                Ok(delta) => delta.as_secs_f64(),
                Err(_) => 0.0,
            };
            disk_io_mb += pair[1].disk_io_mbps * interval;
        }

        ResourceUsageReport {
            peak_memory_mb,
            avg_cpu_percent,
            peak_cpu_percent,
            disk_io_mb,
            // Per-process network throughput is not portably measurable (see
            // `ResourceMonitor::run`), so it is reported as zero, not invented.
            network_usage_mb: 0.0,
            timeline: snapshots,
        }
    }

    /// Get test suite statistics
    pub fn get_statistics(&self) -> TestSuiteStatistics {
        let total_tests = self.results.len();
        let passed = self
            .results
            .iter()
            .filter(|r| r.status == TestExecutionStatus::Passed)
            .count();
        let failed = self
            .results
            .iter()
            .filter(|r| r.status == TestExecutionStatus::Failed)
            .count();
        let skipped = self
            .results
            .iter()
            .filter(|r| r.status == TestExecutionStatus::Skipped)
            .count();
        let errors = self
            .results
            .iter()
            .filter(|r| r.status == TestExecutionStatus::Error)
            .count();

        let total_duration = self
            .results
            .iter()
            .filter_map(|r| r.duration)
            .fold(Duration::ZERO, |acc, d| acc + d);

        TestSuiteStatistics {
            total_tests,
            passed,
            failed,
            skipped,
            errors,
            total_duration,
            success_rate: if total_tests > 0 {
                passed as f64 / total_tests as f64
            } else {
                0.0
            },
        }
    }
}

/// Test suite execution statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestSuiteStatistics {
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed: usize,
    /// Number of failed tests
    pub failed: usize,
    /// Number of skipped tests
    pub skipped: usize,
    /// Number of error tests
    pub errors: usize,
    /// Total execution duration
    pub total_duration: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

// Default implementations

impl Default for TestSuiteConfig {
    fn default() -> Self {
        Self {
            include_unit: true,
            include_integration: true,
            include_stress: false,
            include_load: false,
            include_security: false,
            default_timeout: 300, // 5 minutes
            parallel_execution: ParallelExecutionConfig::default(),
            resource_monitoring: ResourceMonitoringConfig::default(),
            filtering: TestFilteringConfig::default(),
            retry_config: TestRetryConfig::default(),
        }
    }
}

impl Default for ParallelExecutionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent: num_cpus::get(),
            thread_pool_size: None,
            grouping_strategy: TestGroupingStrategy::ByCategory,
            resource_allocation: ResourceAllocationConfig::default(),
        }
    }
}

impl Default for ResourceAllocationConfig {
    fn default() -> Self {
        Self {
            cpu_cores: None,
            memory_limit_mb: Some(1024), // 1GB per test
            disk_limit_mb: Some(10240),  // 10GB per test
            network_limit_mbps: None,
        }
    }
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            monitor_cpu: true,
            monitor_memory: true,
            monitor_disk_io: false,
            monitor_network: false,
            monitoring_frequency_ms: 1000, // 1 second
            alert_thresholds: ResourceAlertThresholds::default(),
        }
    }
}

impl Default for ResourceAlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 90.0,      // 90% CPU
            memory_threshold: 85.0,   // 85% memory
            disk_threshold: 90.0,     // 90% disk
            network_threshold: 100.0, // 100 MB/s
        }
    }
}

impl Default for TestRetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            retry_delay_sec: 5,
            backoff_multiplier: 2.0,
            retry_on_failures: vec![
                TestFailureType::Timeout,
                TestFailureType::TransientError,
                TestFailureType::Network,
            ],
        }
    }
}

impl Default for ResourceUsageReport {
    fn default() -> Self {
        Self {
            peak_memory_mb: 0.0,
            avg_cpu_percent: 0.0,
            peak_cpu_percent: 0.0,
            disk_io_mb: 0.0,
            network_usage_mb: 0.0,
            timeline: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_suite_creation() {
        let config = TestSuiteConfig::default();
        let suite = PerformanceTestSuite::new(config);
        assert!(suite.is_ok());
    }

    #[test]
    fn test_test_case_filtering() {
        let mut suite =
            PerformanceTestSuite::new(TestSuiteConfig::default()).expect("unwrap failed");

        let test_case = PerformanceTestCase {
            name: "test1".to_string(),
            category: TestCategory::Unit,
            executor: TestExecutor::Criterion,
            parameters: HashMap::new(),
            baseline: None,
            timeout: None,
            iterations: 5,
            warmup_iterations: 1,
            dependencies: Vec::new(),
            tags: vec!["fast".to_string()],
            environment_requirements: EnvironmentRequirements::default(),
            custom_config: HashMap::new(),
        };

        suite.add_test_case(test_case);
        assert_eq!(suite.test_cases.len(), 1);
    }

    #[test]
    fn test_environment_requirements_validation() {
        let suite = PerformanceTestSuite::new(TestSuiteConfig::default()).expect("unwrap failed");
        let requirements = EnvironmentRequirements::default();
        assert!(suite.check_environment_requirements(&requirements).is_ok());
    }

    #[test]
    fn test_test_execution_status() {
        assert_eq!(TestExecutionStatus::Passed, TestExecutionStatus::Passed);
        assert_ne!(TestExecutionStatus::Passed, TestExecutionStatus::Failed);
    }

    #[test]
    fn test_resource_monitoring_config() {
        let config = ResourceMonitoringConfig::default();
        assert!(config.monitor_cpu);
        assert!(config.monitor_memory);
        assert_eq!(config.monitoring_frequency_ms, 1000);
    }

    fn sample_test_case(name: &str) -> PerformanceTestCase {
        PerformanceTestCase {
            name: name.to_string(),
            category: TestCategory::Unit,
            executor: TestExecutor::Criterion,
            parameters: HashMap::new(),
            baseline: None,
            timeout: None,
            iterations: 1,
            warmup_iterations: 0,
            dependencies: Vec::new(),
            tags: Vec::new(),
            environment_requirements: EnvironmentRequirements::default(),
            custom_config: HashMap::new(),
        }
    }

    #[test]
    fn test_extract_duration_seconds_honours_units() {
        // Regression (F30): `ms` used to be stripped rather than converted, so
        // a 1.5 ms timing was reported as 1.5 SECONDS — a 1000x error.
        let one_and_a_half_ms = PerformanceTestSuite::extract_duration_seconds("time: 1.5 ms")
            .expect("a millisecond timing parses");
        assert!(
            (one_and_a_half_ms - 0.0015).abs() < 1e-12,
            "1.5 ms must be 0.0015 s, got {one_and_a_half_ms}"
        );

        assert!(
            (PerformanceTestSuite::extract_duration_seconds("duration = 250us")
                .expect("microseconds parse")
                - 250e-6)
                .abs()
                < 1e-15
        );
        assert!(
            (PerformanceTestSuite::extract_duration_seconds("elapsed: 2 s")
                .expect("seconds parse")
                - 2.0)
                .abs()
                < 1e-12
        );
        assert!(
            (PerformanceTestSuite::extract_duration_seconds("time:   [1.2340 ms 1.2350 ms]")
                .expect("criterion output parses")
                - 0.001234)
                .abs()
                < 1e-12
        );

        // Lines without a timing must not yield one.
        assert_eq!(
            PerformanceTestSuite::extract_duration_seconds("running 3 tests"),
            None
        );
        assert_eq!(
            PerformanceTestSuite::extract_duration_seconds("timestamp: 1700000000"),
            None
        );
        assert_eq!(
            PerformanceTestSuite::extract_duration_seconds("estimated time 12 units"),
            None,
            "an unrecognised unit must not be assumed to be seconds"
        );
    }

    #[test]
    fn test_parse_performance_output_does_not_fabricate_measurements() {
        // Regression (F30): output with no timing used to be turned into a
        // fabricated 1-second measurement that then fed regression analysis.
        let suite = PerformanceTestSuite::new(TestSuiteConfig::default()).expect("suite creation");
        let test_case = sample_test_case("no_timing");

        let empty = suite
            .parse_performance_output("running 3 tests\nall good\n", &test_case)
            .expect("parsing succeeds");
        assert!(
            empty.is_empty(),
            "output without a timing must produce no measurements, got {}",
            empty.len()
        );

        let parsed = suite
            .parse_performance_output("bench foo\ntime: 2.5 ms\n", &test_case)
            .expect("parsing succeeds");
        assert_eq!(parsed.len(), 1);
        let value = parsed[0]
            .metrics
            .get(&crate::performance_regression_detector::MetricType::ExecutionTime)
            .map(|metric| metric.value)
            .expect("execution time recorded");
        assert!(
            (value - 0.0025).abs() < 1e-12,
            "2.5 ms must be recorded as 0.0025 s, got {value}"
        );
    }

    #[test]
    fn test_environment_and_git_info_are_measured_not_placeholders() {
        // Regression (F33/F17): environment info used to be all placeholders
        // ("unknown" toolchain, 0 memory) and `is_clean` was hardcoded `true`.
        let suite = PerformanceTestSuite::new(TestSuiteConfig::default()).expect("suite creation");

        let env = suite
            .gather_environment_info()
            .expect("environment info is gathered");
        assert!(!env.os.is_empty());
        assert!(!env.cpu_model.is_empty());
        assert!(env.cpu_cores >= 1, "core count must be a real measurement");
        if SystemSampler::new().is_ok() {
            assert!(
                env.total_memory_mb > 0,
                "installed memory must be measured when a sampler is available"
            );
        }
        // The test itself is built by cargo, so rustc is on PATH and the
        // toolchain version must have been read rather than left "unknown".
        assert_ne!(env.rust_version, "unknown");
        assert_eq!(env.compiler_version, env.rust_version);

        let git = suite.gather_git_info().expect("git info is gathered");
        assert!(!git.commit_hash.is_empty());
        assert!(!git.branch.is_empty());
        // Single-line fields only: a multi-line value would corrupt the CSV and
        // HTML reports built from this metadata.
        for field in [&git.commit_hash, &git.branch] {
            assert!(!field.contains('\n'));
        }
        if let Some(message) = &git.commit_message {
            assert!(
                !message.contains('\n'),
                "the commit subject must be a single line, got {message:?}"
            );
        }
        // `is_clean` must agree with a real `git status` when one can be run,
        // and must never be an unverified `true`.
        match crate::system_sampler::git_is_clean(".") {
            Ok(clean) => assert_eq!(git.is_clean, clean),
            Err(_) => assert!(
                !git.is_clean,
                "an unverifiable working tree must not be reported as clean"
            ),
        }
    }

    #[test]
    fn test_criterion_without_target_reports_an_in_process_loop() {
        // Regression (F31/F32): a Criterion test case with no configured bench
        // target must not claim a criterion run happened, and the loop it does
        // measure must survive the optimizer (black_box), producing a real,
        // finite duration.
        let suite = PerformanceTestSuite::new(TestSuiteConfig::default()).expect("suite creation");
        let mut test_case = sample_test_case("loop_only");
        test_case
            .parameters
            .insert("iterations".to_string(), "10000".to_string());

        let (measurements, output) = suite
            .execute_criterion_test(&test_case)
            .expect("in-process loop executes");

        assert_eq!(measurements.len(), 1);
        let value = measurements[0]
            .metrics
            .get(&crate::performance_regression_detector::MetricType::ExecutionTime)
            .map(|metric| metric.value)
            .expect("execution time recorded");
        assert!(
            value.is_finite() && value >= 0.0,
            "the measured loop time must be a real finite duration, got {value}"
        );
        assert!(
            !output.contains("Criterion benchmark completed"),
            "the output must not claim a criterion run that did not happen: {output}"
        );
        assert!(output.contains("in-process loop"));
    }

    #[test]
    fn test_resource_monitor_stops_without_waiting_a_full_interval() {
        // Regression (carryover nit): the sampling thread slept the whole
        // interval before re-checking the stop flag, so `stop()` blocked for up
        // to `frequency_ms` on every test execution. It must now observe the
        // flag on a short tick.
        let monitor = match ResourceMonitor::start(5_000) {
            Ok(monitor) => monitor,
            // No system sampler on this platform: nothing to assert.
            Err(_) => return,
        };

        let started = Instant::now();
        let snapshots = monitor.stop();
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(1_000),
            "stopping must not wait out the 5s sampling interval, took {elapsed:?}"
        );
        // Stopped immediately, so no full interval elapsed and nothing was
        // sampled — an empty timeline is the honest result.
        assert!(snapshots.is_empty());
    }
}
