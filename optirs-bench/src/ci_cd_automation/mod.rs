// # CI/CD Automation System
//
// This module provides comprehensive CI/CD automation for performance testing,
// benchmarking, and regression detection in the SciRS2 ecosystem.
//
// ## Architecture
//
// The CI/CD automation system is organized into 7 specialized modules:
//
// - **config**: Configuration management and platform settings
// - **test_execution**: Test suite management and execution logic
// - **reporting**: Report generation, templates, and formatting
// - **artifact_management**: Storage providers and artifact handling
// - **integrations**: External service integrations (GitHub, Slack, Email, Webhooks)
// - **performance_gates**: Performance monitoring and gate evaluation
// - **core_automation**: Main automation engine and orchestration
//
// ## Key Features
//
// - Multi-platform CI/CD support (GitHub Actions, GitLab CI, Jenkins, etc.)
// - Performance regression detection with statistical analysis
// - Comprehensive reporting with multiple output formats
// - Artifact storage with multiple cloud providers
// - Real-time integration with external services
// - Performance gates with trend analysis and alerting
// - Automated baseline management and historical tracking
//
// ## Usage
//
// ```rust
// use optirs_core::benchmarking::ci_cd_automation::{
//     CiCdAutomation, CiCdAutomationConfig, CiCdPlatform
// };
//
// // Create automation configuration
// let config = CiCdAutomationConfig {
//     enable_automation: true,
//     platform: CiCdPlatform::GitHubActions,
//     // ... other configuration
// };
//
// // Initialize automation system
// let automation = CiCdAutomation::new(config)?;
//
// // Run performance tests with CI/CD integration
// let results = automation.run_automated_tests().await?;
// ```

pub mod artifact_management;
pub mod config;
pub mod core_automation;
pub mod integrations;
pub mod performance_gates;
pub mod reporting;
pub mod test_execution;

// Re-export all public types and functions

// Configuration types and enums
pub use config::{
    ArtifactStorageConfig, BaselineManagementConfig, BaselineStorageConfig,
    BaselineStorageProvider, BaselineValidationConfig, CiCdAutomationConfig, CiCdPlatform,
    ComparisonOperator, CompressionConfig, CronSchedule, EmailIntegration, EncryptionAlgorithm,
    EncryptionConfig, GateFailureHandling, GateSeverity, GateType, GitHubIntegration,
    IntegrationConfig, KeyManagementConfig, MetricGate, MetricType, PerformanceGatesConfig,
    PlatformSpecificConfig, ReportDistributionConfig, ReportStylingConfig, ReportTemplateConfig,
    ReportingConfig, ResourceLimits, SlackIntegration, TestExecutionConfig, TestIsolationLevel,
    WebhookIntegration,
};

// Test execution types and functionality
pub use test_execution::{
    BaselineMetrics, CiCdContext, CiCdTestResult, DependencySource, EnvironmentRequirements,
    GitInfo, MetricBaseline, NetworkAccessRequirements, ParallelExecutionConfig,
    PerformanceTestCase, PerformanceTestSuite, PullRequestInfo, RegressionAnalysisResult,
    ResourceAllocationConfig, ResourceMonitoringConfig, ResourceSnapshot, ResourceUsageReport,
    SoftwareDependency, TestCategory, TestExecutionMetadata, TestExecutionStatus, TestExecutor,
    TestFailureType, TestFilteringConfig, TestGroupingStrategy, TestRetryConfig, TestSuiteConfig,
    TestSuiteStatistics, TriggerEvent,
};

// Reporting types and functionality
pub use reporting::{
    AnimationConfig, AxisConfig, ChartConfig, ChartData, ChartType, DataPoint, DataSeries,
    DataValue, EasingFunction, GeneratedReport, GeneratorInfo, GridConfig, JsonReportData,
    LegendConfig, LegendOrientation, LegendPosition, PerformanceTrendAnalysis, ReportGenerator,
    ReportMetadata, ReportSummary, ReportType, ScaleType, SeriesStyle, StrokeStyle, TemplateEngine,
    TemplateFunction, TrendDataPoint, TrendDirection,
};

// Artifact management types and functionality
pub use artifact_management::{
    ArtifactInfo, ArtifactManager, ArtifactMetadata, ArtifactRecord, ArtifactRegistry,
    ArtifactStatus, ArtifactStorage, CacheEntry, CacheStatistics, ChecksumAlgorithm, CleanupAction,
    CleanupCondition, CleanupResult, CleanupRule, CleanupScheduler, CompressionInfo, DownloadCache,
    DownloadManager, DownloadProgress, DownloadStatus, DownloadTask, EncryptionInfo,
    LocalArtifactStorage, LocalStorageConfig, RegistryMetadata, RegistryStatistics,
    RetentionManager, StorageStatistics, UploadManager, UploadProgress, UploadResult, UploadStatus,
    UploadTask, UploadTaskConfig,
};

// Integration types and functionality
pub use integrations::*;

// Performance gates types and functionality
pub use performance_gates::*;

// Core automation types and functionality
pub use core_automation::*;

// Convenience type aliases
pub type CiCdResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type AutomationResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// Constants
pub const DEFAULT_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3600); // 1 hour
pub const DEFAULT_RETRY_ATTEMPTS: usize = 3;
pub const DEFAULT_PARALLEL_JOBS: usize = 4;
pub const DEFAULT_COMPRESSION_LEVEL: u8 = 6;
pub const DEFAULT_RETENTION_DAYS: u32 = 30;
pub const MAX_ARTIFACT_SIZE: u64 = 1024 * 1024 * 1024; // 1 GB
pub const MAX_REPORT_SIZE: usize = 100 * 1024 * 1024; // 100 MB
pub const DEFAULT_BASELINE_WINDOW: usize = 10;
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 0.05; // 5%

// Error types - removed non-existent error types

// Utility functions
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn build_info() -> String {
    format!(
        "SciRS2 CI/CD Automation v{} ({})",
        version(),
        env!("CARGO_PKG_REPOSITORY")
    )
}
