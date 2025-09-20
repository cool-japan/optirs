// Advanced Cross-Platform Testing Orchestrator
//
// This module provides comprehensive cross-platform testing capabilities including:
// - Multi-platform test execution (Linux, Windows, macOS, mobile, embedded)
// - Cloud provider integration (AWS, Azure, GCP, GitHub Actions)
// - Container runtime management (Docker, Podman)
// - Test matrix generation and execution
// - Resource allocation and cost tracking
// - Result aggregation and compatibility analysis
//
// # Architecture
//
// The orchestrator is built with a modular architecture:
//
// - **config**: Configuration management for all components
// - **types**: Core data types and platform definitions
// - **orchestrator**: Main orchestration logic and workflow
// - **matrix**: Test matrix generation and filtering
// - **container**: Container runtime abstraction and management
// - **cloud**: Cloud provider implementations and abstractions
// - **resources**: Resource allocation, usage tracking, and cost management
// - **aggregator**: Result collection and compatibility analysis
//
// # Usage Example
//
// ```rust
// use optirs_core::benchmarking::advanced_cross_platform_orchestrator::{
//     CrossPlatformOrchestrator, OrchestratorConfig, PlatformTarget
// };
//
// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
// let config = OrchestratorConfig::default();
// let mut orchestrator = CrossPlatformOrchestrator::new(config)?;
//
// // Execute cross-platform testing
// let summary = orchestrator.execute_cross_platform_testing().await?;
// println!("Testing completed with {} results", summary.total_tests);
// # Ok(())
// # }
// ```

pub mod aggregator;
pub mod cloud;
pub mod config;
pub mod container;
pub mod matrix;
pub mod orchestrator;
pub mod resources;
pub mod types;

use std::collections::HashMap;

// Re-export core configuration types
pub use config::{
    AwsConfig, AzureConfig, BuildProfile, CloudConfig, ComplianceConfig, ContainerConfig,
    ContainerRegistryConfig, CustomCloudConfig, FeatureCombination, GcpConfig, GitHubActionsConfig,
    MetricsConfig, MonitoringConfig, NetworkConfig, OrchestratorConfig,
    PlatformResourceRequirements, PlatformSpec, ResourceLimits, SecurityConfig, StorageConfig,
    TestMatrixConfig, TestScenario,
};

// Re-export core data types
pub use types::{
    CloudInstance, CloudInstanceStatus, CloudProviderType, CompatibilityIssue, CompatibilityMatrix,
    ComplianceReport, ContainerInfo, ContainerRuntime, ContainerStats, ContainerStatus,
    CrossPlatformMetrics, CrossPlatformTestingSummary, EnvironmentVariables, HardwareInfo,
    NetworkPort, NumericalResult, OptimizationLevel, Performance, PerformanceMetrics,
    PlatformCapabilities, PlatformTarget, PlatformTestResult, ResourceType, ResourceUsage,
    SecurityContext, SoftwareInfo, TestEnvironment, TestExecutionResult, TestMatrixEntry,
    TestResult, TestStatus,
};

// Import BenchmarkResult from the main crate
pub use crate::BenchmarkResult;

// Re-export main orchestrator
pub use orchestrator::{
    CloudTestExecutor, ContainerTestExecutor, CrossPlatformOrchestrator, ExecutionContext,
    PlatformTestExecutor, TestExecutor,
};

// Re-export test matrix functionality
pub use matrix::{MatrixFilter, MatrixStatistics, TestMatrixGenerator};

// Re-export container management
pub use container::{ContainerManager, ContainerRuntimeTrait, DockerRuntime, PodmanRuntime};

// Re-export cloud provider abstractions
pub use cloud::{
    AwsProvider, AzureProvider, CloudProvider, CustomProvider, GcpProvider, GitHubActionsProvider,
};

// Re-export resource management
pub use resources::{CostEntry, CostTracker, PlatformResourceManager, ResourceUsageTracker};

// Re-export result aggregation
pub use aggregator::{AggregationStatistics, ResultAggregator};

/// Convenience function to create a default orchestrator
pub fn create_default_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let config = OrchestratorConfig::default();
    CrossPlatformOrchestrator::new(config)
}

/// Convenience function to create an orchestrator with cloud testing enabled
pub fn create_cloud_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let mut config = OrchestratorConfig::default();
    config.enable_cloud_testing = true;
    // Enable cloud providers by setting up basic configs
    config.cloud_config.aws = Some(config::AwsConfig::default());
    config.cloud_config.azure = Some(config::AzureConfig::default());
    config.cloud_config.gcp = Some(config::GcpConfig::default());
    config.cloud_config.github_actions = Some(config::GitHubActionsConfig::default());

    CrossPlatformOrchestrator::new(config)
}

/// Convenience function to create an orchestrator with container testing enabled
pub fn create_container_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let mut config = OrchestratorConfig::default();
    config.enable_container_testing = true;
    config.container_config.runtime = ContainerRuntime::Docker;
    // Container configuration will use defaults

    CrossPlatformOrchestrator::new(config)
}

/// Convenience function to create a minimal orchestrator for CI/CD
pub fn create_ci_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let mut config = OrchestratorConfig::default();
    config.enable_parallel_testing = true;
    config.max_concurrent_jobs = 4;
    config.enable_container_testing = true;
    config.enable_cloud_testing = false; // Disable cloud for CI to save costs

    // Configure for essential platforms only
    config.matrix_config.platforms = vec![
        PlatformSpec {
            target: PlatformTarget::LinuxX86_64,
            priority: 10,
            required_for_release: true,
            is_baseline: true,
            config: HashMap::new(),
            resource_requirements: PlatformResourceRequirements::default(),
        },
        PlatformSpec {
            target: PlatformTarget::WindowsX86_64,
            priority: 9,
            required_for_release: true,
            is_baseline: false,
            config: HashMap::new(),
            resource_requirements: PlatformResourceRequirements::default(),
        },
        PlatformSpec {
            target: PlatformTarget::MacOSX86_64,
            priority: 8,
            required_for_release: true,
            is_baseline: false,
            config: HashMap::new(),
            resource_requirements: PlatformResourceRequirements::default(),
        },
    ];

    CrossPlatformOrchestrator::new(config)
}

/// High-level async function to run complete cross-platform testing
pub async fn run_cross_platform_testing(
    config: Option<OrchestratorConfig>,
) -> crate::error::Result<CrossPlatformTestingSummary> {
    let config = config.unwrap_or_default();
    let mut orchestrator = CrossPlatformOrchestrator::new(config)?;
    orchestrator.execute_cross_platform_testing().await
}

/// High-level async function to run testing on specific platforms
pub async fn run_platform_specific_testing(
    platforms: Vec<PlatformTarget>,
    config: Option<OrchestratorConfig>,
) -> crate::error::Result<CrossPlatformTestingSummary> {
    let mut config = config.unwrap_or_default();

    // Filter platforms to only include requested ones
    config.matrix_config.platforms = config
        .matrix_config
        .platforms
        .into_iter()
        .filter(|spec| platforms.contains(&spec.target))
        .collect();

    let mut orchestrator = CrossPlatformOrchestrator::new(config)?;
    orchestrator.execute_cross_platform_testing().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_orchestrator_creation() {
        let result = create_default_orchestrator();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cloud_orchestrator_creation() {
        let result = create_cloud_orchestrator();
        assert!(result.is_ok());
    }

    #[test]
    fn test_container_orchestrator_creation() {
        let result = create_container_orchestrator();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ci_orchestrator_creation() {
        let result = create_ci_orchestrator();
        assert!(result.is_ok());

        if let Ok(orchestrator) = result {
            // Verify CI-specific configuration
            let config = orchestrator.get_config();
            assert!(config.enable_parallel_testing);
            assert_eq!(config.max_concurrent_jobs, 4);
            assert!(config.enable_container_testing);
            assert!(!config.enable_cloud_testing);
            assert_eq!(config.matrix_config.platforms.len(), 3);
        }
    }

    #[tokio::test]
    async fn test_cross_platform_testing_execution() {
        let config = OrchestratorConfig::default();
        let result = run_cross_platform_testing(Some(config)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_platform_specific_testing() {
        let platforms = vec![PlatformTarget::LinuxX86_64, PlatformTarget::WindowsX86_64];
        let result = run_platform_specific_testing(platforms, None).await;
        assert!(result.is_ok());
    }
}
