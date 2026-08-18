// Main orchestrator implementation for cross-platform testing
//
// This module contains the primary CrossPlatformOrchestrator struct and its core
// orchestration logic for managing distributed, cross-platform testing workflows.

use crate::ci_cd_automation::{CiCdAutomation, CiCdAutomationConfig};
use crate::error::{OptimError, Result};
use futures::stream::{self, StreamExt};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};
// Only exercised by the unit tests below -- gated so a non-test build does not
// warn about unused imports.
#[cfg(test)]
use std::time::Instant;

use super::aggregator::ResultAggregator;
use super::cloud::{
    AwsProvider, AzureProvider, CloudProvider, CloudProviderEnum, CustomProvider, GcpProvider,
    GitHubActionsProvider,
};
use super::config::*;
use super::container::ContainerManager;
use super::execution::{
    host_platform, is_windows_platform, parse_metrics_from_output, run_process,
    scenario_environment, shell_invocation, ExecutionTarget, ResolvedTarget, SshAccess,
};
use super::matrix::TestMatrixGenerator;
use super::resources::PlatformResourceManager;
use super::types::platform_target_to_string;
use super::types::*;
use crate::cross_platform_tester::{
    PlatformTarget as CrossPlatformTarget, RecommendationPriority, RecommendationType,
};

/// Key under which a test result carries its declared target platform (as a serde
/// JSON encoding of `PlatformTarget`) inside `TestResult::platform_details`. This is
/// the authoritative, declared platform metadata — never guessed from the test name.
const PLATFORM_DETAIL_KEY: &str = "target_platform";

/// Convert between PlatformTarget types
fn convert_platform_target(platform: &PlatformTarget) -> CrossPlatformTarget {
    match platform {
        PlatformTarget::LinuxX86_64 => CrossPlatformTarget::LinuxX64,
        PlatformTarget::LinuxAarch64 => CrossPlatformTarget::LinuxArm64,
        PlatformTarget::WindowsX86_64 => CrossPlatformTarget::WindowsX64,
        PlatformTarget::MacOSX86_64 => CrossPlatformTarget::MacOSX64,
        PlatformTarget::MacOSAarch64 => CrossPlatformTarget::MacOSArm64,
        _ => CrossPlatformTarget::LinuxX64, // Default fallback
    }
}

/// Advanced cross-platform testing orchestrator
#[derive(Debug)]
pub struct CrossPlatformOrchestrator {
    /// Orchestrator configuration
    config: OrchestratorConfig,
    /// Cloud provider integrations
    cloud_providers: Vec<CloudProviderEnum>,
    /// Container runtime manager
    container_manager: ContainerManager,
    /// Test matrix generator
    matrix_generator: TestMatrixGenerator,
    /// Result aggregator
    result_aggregator: ResultAggregator,
    /// Platform resource manager
    resource_manager: PlatformResourceManager,
    /// CI/CD integration
    ci_cd_integration: Option<CiCdAutomation>,
}

// CrossPlatformTestingSummary is now defined in types.rs

/// Resource allocation information
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocation ID
    pub id: String,
    /// Platform
    pub platform: PlatformTarget,
    /// Resource type
    pub resource_type: AllocatedResourceType,
    /// Allocated time
    pub allocated_at: SystemTime,
    /// Estimated completion time
    pub estimated_completion: SystemTime,
    /// Allocation status
    pub status: AllocationStatus,
    /// Resource usage
    pub usage: ResourceUsage,
}

/// Allocated resource types
#[derive(Debug, Clone)]
pub enum AllocatedResourceType {
    CloudInstance(CloudInstance),
    Container(ContainerInfo),
    Local,
}

/// Compatibility analysis result
#[derive(Debug, Clone)]
pub struct CompatibilityAnalysis {
    /// Overall compatibility score
    pub overall_score: f64,
    /// Platform-specific scores
    pub platform_scores: HashMap<PlatformTarget, f64>,
    /// Feature compatibility
    pub feature_compatibility: HashMap<String, HashMap<PlatformTarget, bool>>,
    /// Issues by platform
    pub issues_by_platform: HashMap<PlatformTarget, Vec<CompatibilityIssue>>,
    /// Cross-platform issues
    pub cross_platform_issues: Vec<CompatibilityIssue>,
}

impl CrossPlatformOrchestrator {
    /// Create a new cross-platform orchestrator
    pub fn new(config: OrchestratorConfig) -> Result<Self> {
        let cloud_providers = Self::initialize_cloud_providers(&config.cloud_config)?;
        let container_manager = ContainerManager::new(config.container_config.clone())?;
        let matrix_generator = TestMatrixGenerator::new(config.matrix_config.clone())?;
        let result_aggregator = ResultAggregator::new();
        let resource_manager = PlatformResourceManager::new(config.resource_limits.clone())?;

        // NOTE: `config.ci_cd_config` (a `CiCdIntegrationConfig`: platform/settings/
        // webhooks/status_checks) and `CiCdAutomationConfig` (enable_automation/
        // platform/test_execution/baseline_management/reporting/artifact_storage/
        // integrations/performance_gates) are different, non-isomorphic schemas with
        // no lossless field-for-field mapping; only `platform` lines up directly.
        // Deriving a faithful `CiCdAutomationConfig` from the lighter integration
        // config is a real design task, not a mechanical conversion, so it is left
        // as a tracked gap rather than invented here: CI/CD automation is enabled
        // with defaults whenever integration is requested, but does not yet inherit
        // the caller's declared platform/webhooks/status-check settings.
        let ci_cd_integration = if config.ci_cd_config.is_some() {
            Some(CiCdAutomation::new(CiCdAutomationConfig::default())?)
        } else {
            None
        };

        Ok(Self {
            config,
            cloud_providers,
            container_manager,
            matrix_generator,
            result_aggregator,
            resource_manager,
            ci_cd_integration,
        })
    }

    /// The configured CI/CD automation engine, if `OrchestratorConfig::ci_cd_config`
    /// requested one. `execute_cross_platform_testing` does not drive this itself —
    /// `CiCdAutomation::execute_automation` needs a `TriggerEvent` (push/PR/release/
    /// schedule/manual/api/webhook) that only the surrounding CI environment knows,
    /// not something this orchestrator can infer — so callers that do know the
    /// trigger use this accessor to run it explicitly.
    pub fn ci_cd_automation(&self) -> Option<&CiCdAutomation> {
        self.ci_cd_integration.as_ref()
    }

    /// Mutable access to the configured CI/CD automation engine; see
    /// [`Self::ci_cd_automation`]. `CiCdAutomation::execute_automation` takes
    /// `&mut self`, so callers driving it need this rather than the shared accessor.
    pub fn ci_cd_automation_mut(&mut self) -> Option<&mut CiCdAutomation> {
        self.ci_cd_integration.as_mut()
    }

    /// Execute comprehensive cross-platform testing
    pub async fn execute_cross_platform_testing(&mut self) -> Result<CrossPlatformTestingSummary> {
        log::info!("🚀 Starting comprehensive cross-platform testing...");

        // Generate test matrix
        let matrix = self.matrix_generator.generate_matrix()?;
        log::info!("📋 Generated test matrix with {} entries", matrix.len());

        // Allocate resources
        let allocations = self.allocate_resources_for_matrix(&matrix).await?;
        log::info!("🔧 Allocated resources for {} platforms", allocations.len());

        // Execute tests
        let results = if self.config.enable_parallel_testing {
            self.execute_parallel_testing(&matrix, &allocations).await?
        } else {
            self.execute_sequential_testing(&matrix, &allocations)
                .await?
        };

        // Aggregate results
        self.result_aggregator.aggregate_results(results.clone())?;

        // Analyze cross-platform compatibility
        let compatibility_analysis = self.analyze_compatibility(&results).await?;

        // Generate performance comparisons
        let performance_comparisons = self.generate_performance_comparisons(&results).await?;

        // Detect trends
        let trends = self.analyze_performance_trends(&results).await?;

        // Generate recommendations
        let recommendations = self
            .generate_recommendations(&compatibility_analysis, &results)
            .await?;

        // Cleanup resources
        self.cleanup_resources(&allocations).await?;

        // Generate comprehensive report
        let total_platforms = matrix
            .iter()
            .map(|e| &e.platform)
            .collect::<HashSet<_>>()
            .len();
        let successful_platforms = results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Passed))
            .count();
        let failed_platforms = results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Failed))
            .count();

        // Create platform results map using the declared platform metadata carried on
        // each result (not a parse of the test name, which never matched).
        let mut platform_results = HashMap::new();
        for result in &results {
            if let Some(platform) = self.extract_platform_from_result(result) {
                platform_results.insert(platform, result.clone());
            }
        }

        let overall_status = if failed_platforms == 0 {
            TestStatus::Passed
        } else if successful_platforms == 0 {
            TestStatus::Failed
        } else {
            TestStatus::Skipped // Mixed results
        };

        let issues_summary = self.generate_issues_summary(&results);

        let summary = CrossPlatformTestingSummary {
            total_platforms,
            successful_platforms,
            failed_platforms,
            platform_results,
            overall_status,
            execution_time: self.resource_manager.get_total_execution_time(),
            performance_comparisons,
            trends,
            recommendations,
            issues_summary,
        };

        log::info!("✅ Cross-platform testing completed!");
        Ok(summary)
    }

    /// Initialize cloud providers based on configuration
    fn initialize_cloud_providers(config: &CloudConfig) -> Result<Vec<CloudProviderEnum>> {
        let mut providers: Vec<CloudProviderEnum> = Vec::new();

        if let Some(aws_config) = &config.aws {
            providers.push(CloudProviderEnum::Aws(AwsProvider::new(
                aws_config.clone(),
            )?));
        }

        if let Some(azure_config) = &config.azure {
            providers.push(CloudProviderEnum::Azure(AzureProvider::new(
                azure_config.clone(),
            )?));
        }

        if let Some(gcp_config) = &config.gcp {
            providers.push(CloudProviderEnum::Gcp(GcpProvider::new(
                gcp_config.clone(),
            )?));
        }

        if let Some(github_config) = &config.github_actions {
            providers.push(CloudProviderEnum::GitHub(GitHubActionsProvider::new(
                github_config.clone(),
            )?));
        }

        for custom_config in &config.custom_providers {
            providers.push(CloudProviderEnum::Custom(CustomProvider::new(
                custom_config.clone(),
            )?));
        }

        Ok(providers)
    }

    /// Allocate resources for test matrix
    async fn allocate_resources_for_matrix(
        &mut self,
        matrix: &[TestMatrixEntry],
    ) -> Result<HashMap<String, ResourceAllocation>> {
        let mut allocations = HashMap::new();

        for entry in matrix {
            let allocation_id = format!(
                "{}_{}",
                platform_target_to_string(&entry.platform),
                entry.priority
            );

            if self.config.enable_cloud_testing {
                if let Some(provider) = self.find_provider_for_platform(&entry.platform) {
                    // Cloud provisioning depends on external SDKs/credentials. When
                    // they are unavailable we skip that platform (it is reported as
                    // untested in the summary) rather than aborting the whole run.
                    match provider.provision_instance(&entry.platform).await {
                        Ok(instance) => {
                            let allocation = ResourceAllocation {
                                id: allocation_id.clone(),
                                platform: entry.platform.clone(),
                                resource_type: AllocatedResourceType::CloudInstance(instance),
                                allocated_at: SystemTime::now(),
                                estimated_completion: SystemTime::now() + entry.estimated_duration,
                                status: AllocationStatus::Allocated,
                                usage: ResourceUsage::default(),
                            };
                            allocations.insert(allocation_id, allocation);
                        }
                        Err(e) => {
                            log::warn!(
                                "skipping platform {}: cloud provisioning unavailable: {}",
                                platform_target_to_string(&entry.platform),
                                e
                            );
                        }
                    }
                }
            } else if self.config.enable_container_testing {
                // Container creation shells out to a real runtime (docker/podman).
                // If the daemon is not reachable, skip this platform with a warning
                // instead of failing the entire matrix — the run still returns a
                // summary describing what could and could not be exercised.
                match self
                    .container_manager
                    .create_container_for_platform(&entry.platform)
                    .await
                {
                    Ok(container) => {
                        let allocation = ResourceAllocation {
                            id: allocation_id.clone(),
                            platform: entry.platform.clone(),
                            resource_type: AllocatedResourceType::Container(container),
                            allocated_at: SystemTime::now(),
                            estimated_completion: SystemTime::now() + entry.estimated_duration,
                            status: AllocationStatus::Allocated,
                            usage: ResourceUsage::default(),
                        };
                        allocations.insert(allocation_id, allocation);
                    }
                    Err(e) => {
                        log::warn!(
                            "skipping platform {}: container runtime unavailable: {}",
                            platform_target_to_string(&entry.platform),
                            e
                        );
                    }
                }
            } else {
                // Local testing. Mirror the cloud/container branches above: an
                // entry that declares a hard requirement (GPU, network) the local
                // host cannot actually satisfy is skipped with a warning instead
                // of being silently allocated and left to fail its own scenarios
                // later with a more confusing error.
                let needs_gpu = entry.resource_requirements.contains_key(&ResourceType::GPU);
                let needs_network = entry
                    .resource_requirements
                    .contains_key(&ResourceType::Network);
                if needs_gpu && !super::resources::gpu_available() {
                    log::warn!(
                        "skipping platform {}: entry requires a GPU and none was detected locally",
                        platform_target_to_string(&entry.platform)
                    );
                } else if needs_network && !super::resources::network_available() {
                    log::warn!(
                        "skipping platform {}: entry requires network access and none was detected locally",
                        platform_target_to_string(&entry.platform)
                    );
                } else {
                    let allocation = ResourceAllocation {
                        id: allocation_id.clone(),
                        platform: entry.platform.clone(),
                        resource_type: AllocatedResourceType::Local,
                        allocated_at: SystemTime::now(),
                        estimated_completion: SystemTime::now() + entry.estimated_duration,
                        status: AllocationStatus::Available,
                        usage: ResourceUsage::default(),
                    };
                    allocations.insert(allocation_id, allocation);
                }
            }
        }

        Ok(allocations)
    }

    /// Execute tests in parallel.
    ///
    /// Genuinely concurrent (not merely batched-and-awaited-serially, which is
    /// what this used to do despite the name -- see FC1 findings): up to
    /// `max_concurrent_jobs` matrix entries are in flight at once via
    /// `buffer_unordered`. Result order does not need to match matrix order --
    /// `ResultAggregator::update_compatibility_matrix` groups results by
    /// platform through a `HashMap`, not by position.
    async fn execute_parallel_testing(
        &self,
        matrix: &[TestMatrixEntry],
        allocations: &HashMap<String, ResourceAllocation>,
    ) -> Result<Vec<TestResult>> {
        // `buffer_unordered(0)` panics; a non-positive configured limit is
        // treated as "no concurrency limit configured" -> run one at a time
        // rather than fail the whole matrix over a config typo.
        let max_concurrent = self.config.max_concurrent_jobs.max(1);

        // Only entries with a live resource allocation are runnable at all --
        // an entry without one was already skipped (with its reason logged)
        // during `allocate_resources_for_matrix` and must not appear here,
        // matching the previous chunked implementation's lookup-and-skip.
        let runnable: Vec<(&TestMatrixEntry, &ResourceAllocation)> = matrix
            .iter()
            .filter_map(|entry| {
                let allocation_id = format!(
                    "{}_{}",
                    platform_target_to_string(&entry.platform),
                    entry.priority
                );
                allocations
                    .get(&allocation_id)
                    .map(|allocation| (entry, allocation))
            })
            .collect();

        let results: Vec<TestResult> = stream::iter(runnable)
            .map(|(entry, allocation)| async move {
                let outcome = self.execute_matrix_entry(entry, allocation).await;
                self.result_or_skip(entry, outcome)
            })
            .buffer_unordered(max_concurrent)
            .collect()
            .await;

        Ok(results)
    }

    /// Convert a failed entry execution into an honest skipped result.
    ///
    /// The error is preserved verbatim in `error_message`; the status is
    /// `Skipped` (never `Passed`, and never `Failed`, which would misattribute
    /// an environment problem to the code under test).
    fn result_or_skip(&self, entry: &TestMatrixEntry, outcome: Result<TestResult>) -> TestResult {
        match outcome {
            Ok(result) => result,
            Err(e) => {
                log::warn!(
                    "skipping matrix entry {} on {}: {}",
                    entry.id,
                    platform_target_to_string(&entry.platform),
                    e
                );
                TestResult {
                    test_name: entry.id.clone(),
                    status: TestStatus::Skipped,
                    execution_time: Duration::from_secs(0),
                    performance_metrics: PerformanceMetrics::default(),
                    error_message: Some(e.to_string()),
                    platform_details: self.platform_details_for(&entry.platform),
                    numerical_results: None,
                }
            }
        }
    }

    /// Execute tests sequentially
    async fn execute_sequential_testing(
        &self,
        matrix: &[TestMatrixEntry],
        allocations: &HashMap<String, ResourceAllocation>,
    ) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();

        for entry in matrix {
            let allocation_id = format!(
                "{}_{}",
                platform_target_to_string(&entry.platform),
                entry.priority
            );
            if let Some(allocation) = allocations.get(&allocation_id) {
                let outcome = self.execute_matrix_entry(entry, allocation).await;
                results.push(self.result_or_skip(entry, outcome));
            }
        }

        Ok(results)
    }

    /// Execute a single matrix entry
    async fn execute_matrix_entry(
        &self,
        entry: &TestMatrixEntry,
        allocation: &ResourceAllocation,
    ) -> Result<TestResult> {
        let start_time = SystemTime::now();

        // Create test execution context
        let context = TestExecutionContext {
            execution_id: entry.id.clone(),
            platform: entry.platform.clone(),
            rust_version: entry.rust_version.clone(),
            features: entry.features.clone(),
            optimization: entry.optimization.clone(),
            build_profile: entry.build_profile.clone(),
            scenarios: entry.scenarios.clone(),
            start_time,
            expected_duration: entry.estimated_duration,
        };

        // Execute based on resource type
        let result = match &allocation.resource_type {
            AllocatedResourceType::CloudInstance(instance) => {
                self.execute_cloud_test(&context, instance).await
            }
            AllocatedResourceType::Container(container) => {
                self.execute_container_test(&context, container).await
            }
            AllocatedResourceType::Local => self.execute_local_test(&context).await,
        };

        result
    }

    /// Execute the matrix entry's scenario commands on a provisioned cloud
    /// instance.
    ///
    /// Commands run over the instance's configured access method (an SSH login
    /// described by `CloudInstance::config` plus its addresses). When no access
    /// method is configured — no reachable host, or no login user — this returns
    /// an honest error: the test genuinely could not be executed, and reporting
    /// `Passed` would fabricate a result.
    async fn execute_cloud_test(
        &self,
        context: &TestExecutionContext,
        instance: &CloudInstance,
    ) -> Result<TestResult> {
        self.execute_scenarios(context, ExecutionTarget::Cloud(instance))
            .await
    }

    /// Execute the matrix entry's scenario commands inside a provisioned
    /// container, through the configured container runtime (`docker exec` /
    /// `podman exec`).
    ///
    /// The container is inspected first: if the runtime is not installed, the
    /// daemon is unreachable, or the container is not running, this returns
    /// [`OptimError::ResourceUnavailable`] instead of pretending the test ran.
    async fn execute_container_test(
        &self,
        context: &TestExecutionContext,
        container: &ContainerInfo,
    ) -> Result<TestResult> {
        self.execute_scenarios(context, ExecutionTarget::Container(container))
            .await
    }

    /// Execute the matrix entry's scenario commands directly on this host.
    ///
    /// A matrix entry targeting a different platform than the host cannot be run
    /// locally; it is reported as [`TestStatus::PlatformNotSupported`] rather
    /// than being silently "passed" on the wrong platform.
    async fn execute_local_test(&self, context: &TestExecutionContext) -> Result<TestResult> {
        self.execute_scenarios(context, ExecutionTarget::Local)
            .await
    }

    /// Resolve the concrete commands a matrix entry has to run.
    ///
    /// Each of the entry's scenario names is looked up in the configured test
    /// scenarios. An unknown scenario name is a configuration error and is
    /// reported as such — guessing a command would be worse than failing.
    fn resolve_scenario_commands(
        &self,
        context: &TestExecutionContext,
    ) -> Result<Vec<(String, String, Duration)>> {
        let mut resolved = Vec::new();

        for scenario_name in &context.scenarios {
            let scenario = self
                .config
                .matrix_config
                .test_scenarios
                .iter()
                .find(|candidate| &candidate.name == scenario_name)
                .ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "test scenario '{}' referenced by execution {} is not defined in the \
                         orchestrator's matrix configuration",
                        scenario_name, context.execution_id
                    ))
                })?;

            let timeout = self.effective_timeout(scenario.timeout);
            for command in &scenario.commands {
                resolved.push((scenario_name.clone(), command.clone(), timeout));
            }
        }

        Ok(resolved)
    }

    /// Clamp a scenario timeout to the configured maximum test duration, and
    /// substitute the maximum when a scenario declares no timeout at all.
    fn effective_timeout(&self, scenario_timeout: Duration) -> Duration {
        let max_duration = self.config.resource_limits.max_test_duration;
        if scenario_timeout.is_zero() {
            max_duration
        } else if max_duration.is_zero() {
            scenario_timeout
        } else {
            scenario_timeout.min(max_duration)
        }
    }

    /// Build a `TestResult` for an entry that was not executed, carrying the
    /// honest reason. Never used for `Passed`.
    fn non_executed_result(
        &self,
        context: &TestExecutionContext,
        status: TestStatus,
        reason: String,
    ) -> TestResult {
        TestResult {
            test_name: context.execution_id.clone(),
            status,
            execution_time: SystemTime::now()
                .duration_since(context.start_time)
                .unwrap_or_default(),
            performance_metrics: PerformanceMetrics::default(),
            error_message: Some(reason),
            platform_details: self.platform_details_for(&context.platform),
            numerical_results: None,
        }
    }

    /// Verify that the container is actually reachable and running before any
    /// command is dispatched into it.
    ///
    /// This is what separates "the test failed" from "the test never ran": a
    /// missing runtime binary, an unreachable daemon or a stopped container all
    /// produce [`OptimError::ResourceUnavailable`], never a test status.
    async fn ensure_container_running(&self, container: &ContainerInfo) -> Result<()> {
        let runtime = self.container_manager.runtime_binary().to_string();
        let args = vec![
            "inspect".to_string(),
            "--format".to_string(),
            "{{.State.Running}}".to_string(),
            container.container_id.clone(),
        ];

        let outcome = run_process(&runtime, &args, &[], Duration::from_secs(60)).await?;

        if outcome.timed_out {
            return Err(OptimError::ResourceUnavailable(format!(
                "'{} inspect {}' timed out; the container runtime is not responding",
                runtime, container.container_id
            )));
        }
        if !outcome.success {
            return Err(OptimError::ResourceUnavailable(format!(
                "container '{}' cannot be inspected with '{}': {}",
                container.container_id,
                runtime,
                outcome.stderr.trim()
            )));
        }
        if outcome.stdout.trim() != "true" {
            return Err(OptimError::ResourceUnavailable(format!(
                "container '{}' is not running (runtime '{}' reports state '{}'), so no test \
                 command can be executed inside it",
                container.container_id,
                runtime,
                outcome.stdout.trim()
            )));
        }

        Ok(())
    }

    /// Resolve the SSH access configured for a provisioned cloud instance.
    ///
    /// The host is taken from `ssh_host`, the public address, or the private
    /// address (in that order); the login user from `ssh_user`/`username`. Both
    /// are mandatory: without them the instance has no configured access method
    /// and no test can be executed on it.
    fn resolve_ssh_access(instance: &CloudInstance) -> Result<SshAccess> {
        if !matches!(instance.status, CloudInstanceStatus::Running) {
            return Err(OptimError::ResourceUnavailable(format!(
                "cloud instance {} (provider {}) is in state {:?}, not Running; no test command \
                 can be executed on it",
                instance.instance_id, instance.provider, instance.status
            )));
        }

        let host = instance
            .config
            .get("ssh_host")
            .cloned()
            .or_else(|| instance.public_ip.clone())
            .or_else(|| instance.private_ip.clone())
            .ok_or_else(|| {
                OptimError::UnsupportedOperation(format!(
                    "cloud instance {} (provider {}) has no reachable address: set 'ssh_host' in \
                     its configuration, or provide a public/private IP",
                    instance.instance_id, instance.provider
                ))
            })?;

        let user = instance
            .config
            .get("ssh_user")
            .or_else(|| instance.config.get("username"))
            .cloned()
            .ok_or_else(|| {
                OptimError::UnsupportedOperation(format!(
                    "cloud instance {} (provider {}) has no configured access method: set \
                     'ssh_user' in its configuration to execute tests over SSH",
                    instance.instance_id, instance.provider
                ))
            })?;

        let port = instance
            .config
            .get("ssh_port")
            .and_then(|value| value.parse::<u16>().ok());

        let identity_file = instance
            .config
            .get("ssh_key")
            .or_else(|| instance.config.get("ssh_identity_file"))
            .cloned();

        Ok(SshAccess {
            host,
            user,
            port,
            identity_file,
        })
    }

    /// Turn a scenario command string into the concrete program and arguments
    /// used to execute it on the resolved target.
    fn build_invocation(
        target: &ResolvedTarget,
        platform: &PlatformTarget,
        command: &str,
        envs: &[(String, String)],
    ) -> (String, Vec<String>) {
        match target {
            ResolvedTarget::Local => shell_invocation(is_windows_platform(platform), command),
            ResolvedTarget::Container {
                runtime,
                container_id,
                windows,
            } => {
                let mut args = vec!["exec".to_string()];
                for (key, value) in envs {
                    args.push("-e".to_string());
                    args.push(format!("{}={}", key, value));
                }
                args.push(container_id.clone());
                let (shell, shell_args) = shell_invocation(*windows, command);
                args.push(shell);
                args.extend(shell_args);
                (runtime.clone(), args)
            }
            ResolvedTarget::Cloud(access) => {
                // Environment variables are intentionally NOT injected here: `ssh`
                // does not forward them without server-side `AcceptEnv`, and
                // splicing assignments into the remote command string would be a
                // quoting hazard. The remote side sees its own environment.
                let mut args = vec![
                    "-o".to_string(),
                    "BatchMode=yes".to_string(),
                    "-o".to_string(),
                    "StrictHostKeyChecking=accept-new".to_string(),
                ];
                if let Some(identity) = &access.identity_file {
                    args.push("-i".to_string());
                    args.push(identity.clone());
                }
                if let Some(port) = access.port {
                    args.push("-p".to_string());
                    args.push(port.to_string());
                }
                args.push(format!("{}@{}", access.user, access.host));
                args.push(command.to_string());
                ("ssh".to_string(), args)
            }
        }
    }

    /// Execute every command of the entry's scenarios on `target` and map the
    /// real process results onto a [`TestResult`].
    ///
    /// Commands run in order and stop at the first non-zero exit or timeout, the
    /// same way a shell-based CI step behaves. The resulting status is derived
    /// only from measured facts: exit codes, timeouts, and the metrics actually
    /// printed by the commands.
    async fn execute_scenarios(
        &self,
        context: &TestExecutionContext,
        target: ExecutionTarget<'_>,
    ) -> Result<TestResult> {
        let commands = self.resolve_scenario_commands(context)?;
        if commands.is_empty() {
            return Ok(self.non_executed_result(
                context,
                TestStatus::Skipped,
                format!(
                    "no test commands are configured for scenarios [{}]; nothing was executed",
                    context.scenarios.join(", ")
                ),
            ));
        }

        // Reachability pre-flight. A target we cannot reach yields an honest
        // error (cloud/container) or an explicit not-supported status (local on a
        // foreign platform) — never a fabricated pass.
        let resolved = match &target {
            ExecutionTarget::Local => {
                let host = host_platform();
                if host.as_ref() != Some(&context.platform) {
                    return Ok(self.non_executed_result(
                        context,
                        TestStatus::PlatformNotSupported,
                        format!(
                            "local execution cannot run a {} test: this host is {}-{}",
                            context.platform,
                            std::env::consts::OS,
                            std::env::consts::ARCH
                        ),
                    ));
                }
                ResolvedTarget::Local
            }
            ExecutionTarget::Container(container) => {
                self.ensure_container_running(container).await?;
                ResolvedTarget::Container {
                    runtime: self.container_manager.runtime_binary().to_string(),
                    container_id: container.container_id.clone(),
                    windows: is_windows_platform(&container.platform),
                }
            }
            ExecutionTarget::Cloud(instance) => {
                ResolvedTarget::Cloud(Self::resolve_ssh_access(instance)?)
            }
        };

        let envs = scenario_environment(context);
        let mut aggregated_stdout = String::new();
        let mut aggregated_stderr = String::new();
        let mut executed = 0usize;
        let mut status = TestStatus::Passed;
        let mut error_message = None;
        let mut last_exit_code = None;

        for (scenario_name, command, timeout) in &commands {
            let (program, args) =
                Self::build_invocation(&resolved, &context.platform, command, &envs);
            log::info!(
                "executing scenario '{}' for {}: {} {}",
                scenario_name,
                context.execution_id,
                program,
                args.join(" ")
            );

            let outcome = run_process(&program, &args, &envs, *timeout).await?;
            log::debug!(
                "scenario '{}' ran `{}` in {:?} (exit: {:?})",
                scenario_name,
                outcome.command_line,
                outcome.duration,
                outcome.exit_code
            );
            executed += 1;
            aggregated_stdout.push_str(&outcome.stdout);
            aggregated_stderr.push_str(&outcome.stderr);
            last_exit_code = outcome.exit_code;

            if outcome.timed_out {
                status = TestStatus::Timeout;
                error_message = Some(format!(
                    "scenario '{}' command '{}' exceeded its {:?} timeout and was terminated",
                    scenario_name, command, timeout
                ));
                break;
            }
            if !outcome.success {
                status = TestStatus::Failed;
                error_message = Some(format!(
                    "scenario '{}' command '{}' exited with status {}: {}",
                    scenario_name,
                    command,
                    outcome
                        .exit_code
                        .map(|code| code.to_string())
                        .unwrap_or_else(|| "signal".to_string()),
                    outcome.stderr.trim()
                ));
                break;
            }
        }

        let mut platform_details = self.platform_details_for(&context.platform);
        platform_details.insert("commands_total".to_string(), commands.len().to_string());
        platform_details.insert("commands_executed".to_string(), executed.to_string());
        if let Some(code) = last_exit_code {
            platform_details.insert("last_exit_code".to_string(), code.to_string());
        }
        match &resolved {
            ResolvedTarget::Local => {
                platform_details.insert("execution_target".to_string(), "local".to_string());
            }
            ResolvedTarget::Container {
                runtime,
                container_id,
                ..
            } => {
                platform_details.insert("execution_target".to_string(), "container".to_string());
                platform_details.insert("container_runtime".to_string(), runtime.clone());
                platform_details.insert("container_id".to_string(), container_id.clone());
            }
            ResolvedTarget::Cloud(access) => {
                platform_details.insert("execution_target".to_string(), "cloud".to_string());
                platform_details.insert("ssh_host".to_string(), access.host.clone());
            }
        }
        if !aggregated_stderr.trim().is_empty() {
            platform_details.insert(
                "stderr_bytes".to_string(),
                aggregated_stderr.len().to_string(),
            );
        }

        Ok(TestResult {
            test_name: context.execution_id.clone(),
            status,
            execution_time: SystemTime::now()
                .duration_since(context.start_time)
                .unwrap_or_default(),
            performance_metrics: parse_metrics_from_output(&aggregated_stdout),
            error_message,
            platform_details,
            numerical_results: None,
        })
    }

    /// Find cloud provider for platform
    // NOTE: `platform` is intentionally unused. None of the provider config structs
    // (AwsConfig/AzureConfig/GcpConfig/GitHubActionsConfig) declare which
    // `PlatformTarget`s they support, so there is no configured data to match
    // against; a real per-platform selection would mean inventing a provider ->
    // platform capability matrix rather than wiring existing data through. Tracked
    // as a gap; for now the first configured provider is used and
    // `provision_instance` is trusted to reject platforms it cannot serve.
    fn find_provider_for_platform(&self, _platform: &PlatformTarget) -> Option<&CloudProviderEnum> {
        self.cloud_providers.first()
    }

    /// Analyze cross-platform compatibility
    async fn analyze_compatibility(&self, results: &[TestResult]) -> Result<CompatibilityAnalysis> {
        let mut platform_scores = HashMap::new();
        let mut overall_score = 0.0;

        // Group results by platform
        let mut platform_results: HashMap<PlatformTarget, Vec<&TestResult>> = HashMap::new();
        for result in results {
            // Extract platform from test name (simplified)
            if let Some(platform) = self.extract_platform_from_result(result) {
                platform_results.entry(platform).or_default().push(result);
            }
        }

        // Calculate platform-specific scores
        for (platform, platform_tests) in &platform_results {
            let passed = platform_tests
                .iter()
                .filter(|r| matches!(r.status, TestStatus::Passed))
                .count();
            let total = platform_tests.len();
            let score = if total > 0 {
                passed as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            platform_scores.insert(platform.clone(), score);
        }

        // Calculate overall score
        if !platform_scores.is_empty() {
            overall_score = platform_scores.values().sum::<f64>() / platform_scores.len() as f64;
        }

        Ok(CompatibilityAnalysis {
            overall_score,
            platform_scores,
            feature_compatibility: HashMap::new(), // Simplified
            issues_by_platform: HashMap::new(),
            cross_platform_issues: Vec::new(),
        })
    }

    /// Build the `platform_details` map for a test result, recording the declared
    /// target platform as authoritative metadata (serde JSON encoding). Downstream
    /// analysis reads this back instead of guessing the platform from the test name.
    fn platform_details_for(&self, platform: &PlatformTarget) -> HashMap<String, String> {
        let mut details = HashMap::new();
        if let Ok(encoded) = serde_json::to_string(platform) {
            details.insert(PLATFORM_DETAIL_KEY.to_string(), encoded);
        }
        details
    }

    /// Recover the declared target platform from a test result's metadata.
    ///
    /// The platform comes from the test case's declared platform (carried in
    /// `platform_details`), NOT from a substring guess over the test name. If the
    /// metadata is absent or unparseable, returns `None` honestly rather than
    /// fabricating a platform.
    fn extract_platform_from_result(&self, result: &TestResult) -> Option<PlatformTarget> {
        result
            .platform_details
            .get(PLATFORM_DETAIL_KEY)
            .and_then(|encoded| serde_json::from_str::<PlatformTarget>(encoded).ok())
    }

    /// Generate performance comparisons
    async fn generate_performance_comparisons(
        &self,
        results: &[TestResult],
    ) -> Result<HashMap<PlatformTarget, PerformanceMetrics>> {
        let mut comparisons = HashMap::new();

        // Group by platform and aggregate metrics
        let mut platform_metrics: HashMap<PlatformTarget, Vec<&PerformanceMetrics>> =
            HashMap::new();
        for result in results {
            if let Some(platform) = self.extract_platform_from_result(result) {
                platform_metrics
                    .entry(platform)
                    .or_default()
                    .push(&result.performance_metrics);
            }
        }

        // Calculate average metrics per platform
        for (platform, metrics) in platform_metrics {
            if !metrics.is_empty() {
                // For simplicity, just use the first metric (in real implementation, would average)
                comparisons.insert(platform, metrics[0].clone());
            }
        }

        Ok(comparisons)
    }

    /// Analyze performance trends
    async fn analyze_performance_trends(
        &self,
        results: &[TestResult],
    ) -> Result<HashMap<PlatformTarget, TrendDirection>> {
        let mut trends = HashMap::new();

        // Simplified trend analysis
        for result in results {
            if let Some(platform) = self.extract_platform_from_result(result) {
                let trend = match result.status {
                    TestStatus::Passed => TrendDirection::Stable,
                    TestStatus::Failed => TrendDirection::Degrading,
                    _ => TrendDirection::Unknown,
                };
                trends.insert(platform, trend);
            }
        }

        Ok(trends)
    }

    /// Generate recommendations
    // NOTE: `results` is intentionally unused today. Recommendations are derived
    // solely from `compatibility.platform_scores`; folding raw pass/fail counts in
    // too (e.g. flagging a platform whose tests fail outright even when its score
    // stays above the 80% threshold below) needs a `RecommendationType` variant
    // that fits "test failure" — none of the existing ones
    // (Optimization/Configuration/FeatureEnablement/PlatformSpecificImplementation/
    // PerformanceTuning) do — so this is left as a tracked gap rather than an
    // invented mapping.
    async fn generate_recommendations(
        &self,
        compatibility: &CompatibilityAnalysis,
        _results: &[TestResult],
    ) -> Result<Vec<PlatformRecommendation>> {
        let mut recommendations = Vec::new();

        // Generate recommendations based on analysis
        for (platform, score) in &compatibility.platform_scores {
            if *score < 80.0 {
                recommendations.push(PlatformRecommendation {
                    platform: convert_platform_target(platform),
                    recommendation_type: RecommendationType::Optimization,
                    description: format!(
                        "Platform {} has low compatibility score: {:.1}%",
                        platform_target_to_string(platform),
                        score
                    ),
                    priority: if *score < 50.0 {
                        RecommendationPriority::High
                    } else {
                        RecommendationPriority::Medium
                    },
                    estimated_impact: 100.0 - score,
                });
            }
        }

        Ok(recommendations)
    }

    /// Generate issues summary
    fn generate_issues_summary(&self, results: &[TestResult]) -> IssueSummary {
        let total_issues = results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Failed))
            .count();
        let mut issues_by_platform = HashMap::new();

        for result in results {
            if matches!(result.status, TestStatus::Failed) {
                if let Some(platform) = self.extract_platform_from_result(result) {
                    *issues_by_platform.entry(platform).or_insert(0) += 1;
                }
            }
        }

        IssueSummary {
            total_issues,
            issues_by_severity: HashMap::new(),
            issues_by_platform,
            issues_by_category: HashMap::new(),
            blocking_issues: 0,
        }
    }

    /// Cleanup allocated resources
    async fn cleanup_resources(
        &self,
        allocations: &HashMap<String, ResourceAllocation>,
    ) -> Result<()> {
        log::info!("Cleaning up {} resource allocations", allocations.len());

        for (allocation_id, allocation) in allocations {
            match &allocation.resource_type {
                AllocatedResourceType::CloudInstance(instance) => {
                    log::info!("Terminating cloud instance {}", instance.instance_id);
                    // Would terminate cloud instance here
                }
                AllocatedResourceType::Container(container) => {
                    log::info!("Removing container {}", container.container_id);
                    // Would remove container here
                }
                AllocatedResourceType::Local => {
                    log::info!("Local resource {} cleanup complete", allocation_id);
                }
            }
        }

        Ok(())
    }

    /// Get configuration
    pub fn get_config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Get resource manager
    pub fn get_resource_manager(&self) -> &PlatformResourceManager {
        &self.resource_manager
    }

    /// Get result aggregator
    pub fn get_result_aggregator(&self) -> &ResultAggregator {
        &self.result_aggregator
    }
}

/// Execution context for test runs
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub platform: PlatformTarget,
    pub environment: HashMap<String, String>,
    pub resource_allocation: ResourceAllocation,
    pub timeout: Duration,
    pub retry_count: usize,
}

/// Cloud-based test executor
#[derive(Debug)]
pub struct CloudTestExecutor {
    pub provider: String,
    pub region: String,
    pub instance_type: String,
    pub context: ExecutionContext,
}

impl CloudTestExecutor {
    pub fn new(
        provider: String,
        region: String,
        instance_type: String,
        context: ExecutionContext,
    ) -> Self {
        Self {
            provider,
            region,
            instance_type,
            context,
        }
    }

    pub async fn execute_test(&self, test_name: &str) -> Result<String> {
        // Implementation would interact with cloud provider APIs
        Ok(format!(
            "Executed {} on {} in {}",
            test_name, self.instance_type, self.region
        ))
    }
}

/// Container-based test executor
#[derive(Debug)]
pub struct ContainerTestExecutor {
    pub image: String,
    pub runtime: String,
    pub context: ExecutionContext,
}

impl ContainerTestExecutor {
    pub fn new(image: String, runtime: String, context: ExecutionContext) -> Self {
        Self {
            image,
            runtime,
            context,
        }
    }

    pub async fn execute_test(&self, test_name: &str) -> Result<String> {
        // Implementation would run tests in containers
        Ok(format!(
            "Executed {} in container {}",
            test_name, self.image
        ))
    }
}

/// Platform-specific test executor
#[derive(Debug)]
pub struct PlatformTestExecutor {
    pub platform: PlatformTarget,
    pub context: ExecutionContext,
}

impl PlatformTestExecutor {
    pub fn new(platform: PlatformTarget, context: ExecutionContext) -> Self {
        Self { platform, context }
    }

    pub async fn execute_test(&self, test_name: &str) -> Result<String> {
        // Implementation would run tests on specific platforms
        Ok(format!("Executed {} on {:?}", test_name, self.platform))
    }
}

/// Generic test executor trait
#[async_trait::async_trait]
pub trait TestExecutor {
    async fn execute(&self, test_name: &str, context: &ExecutionContext) -> Result<String>;
    fn get_platform(&self) -> PlatformTarget;
    fn supports_parallel_execution(&self) -> bool {
        true
    }
}

#[async_trait::async_trait]
impl TestExecutor for CloudTestExecutor {
    async fn execute(&self, test_name: &str, _context: &ExecutionContext) -> Result<String> {
        self.execute_test(test_name).await
    }

    fn get_platform(&self) -> PlatformTarget {
        self.context.platform.clone()
    }
}

#[async_trait::async_trait]
impl TestExecutor for ContainerTestExecutor {
    async fn execute(&self, test_name: &str, _context: &ExecutionContext) -> Result<String> {
        self.execute_test(test_name).await
    }

    fn get_platform(&self) -> PlatformTarget {
        self.context.platform.clone()
    }
}

#[async_trait::async_trait]
impl TestExecutor for PlatformTestExecutor {
    async fn execute(&self, test_name: &str, _context: &ExecutionContext) -> Result<String> {
        self.execute_test(test_name).await
    }

    fn get_platform(&self) -> PlatformTarget {
        self.context.platform.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let config = OrchestratorConfig::default();
        let orchestrator = CrossPlatformOrchestrator::new(config);
        assert!(orchestrator.is_ok());
    }

    #[test]
    fn test_platform_from_declared_metadata() {
        // Regression (F82): the target platform must come from the test case's declared
        // platform metadata (carried in `platform_details`), never from a substring of
        // the test name.
        let config = OrchestratorConfig::default();
        let orchestrator = CrossPlatformOrchestrator::new(config).expect("unwrap failed");

        // A result whose declared platform is Windows round-trips back to Windows,
        // even though its test name contains no platform hint.
        let details = orchestrator.platform_details_for(&PlatformTarget::WindowsX86_64);
        let declared = TestResult {
            test_name: "opaque_execution_id".to_string(),
            status: TestStatus::Passed,
            execution_time: Duration::from_secs(0),
            performance_metrics: PerformanceMetrics::default(),
            error_message: None,
            platform_details: details,
            numerical_results: None,
        };
        assert_eq!(
            orchestrator.extract_platform_from_result(&declared),
            Some(PlatformTarget::WindowsX86_64)
        );

        // A result WITHOUT declared metadata returns None honestly — the name contains
        // "linux" but that must NOT be used to guess the platform.
        let bare = TestResult {
            test_name: "test_linux_build".to_string(),
            status: TestStatus::Passed,
            execution_time: Duration::from_secs(0),
            performance_metrics: PerformanceMetrics::default(),
            error_message: None,
            platform_details: HashMap::new(),
            numerical_results: None,
        };
        assert_eq!(orchestrator.extract_platform_from_result(&bare), None);
    }

    /// Build an orchestrator whose matrix declares a single scenario running
    /// `commands`, with local execution selected.
    fn orchestrator_with_scenario(name: &str, commands: Vec<String>) -> CrossPlatformOrchestrator {
        let mut config = OrchestratorConfig {
            enable_cloud_testing: false,
            enable_container_testing: false,
            ..Default::default()
        };
        config.matrix_config.test_scenarios = vec![TestScenario {
            name: name.to_string(),
            commands,
            category: TestCategory::Functionality,
            timeout: Duration::from_secs(60),
            expected_results: HashMap::new(),
        }];
        CrossPlatformOrchestrator::new(config).expect("orchestrator construction succeeds")
    }

    fn context_for(platform: PlatformTarget, scenario: &str) -> TestExecutionContext {
        TestExecutionContext {
            execution_id: format!("exec_{}", scenario),
            platform,
            rust_version: "stable".to_string(),
            features: vec!["default".to_string()],
            optimization: OptimizationLevel::Debug,
            build_profile: "test".to_string(),
            scenarios: vec![scenario.to_string()],
            start_time: SystemTime::now(),
            expected_duration: Duration::from_secs(60),
        }
    }

    fn container_for(id: &str) -> ContainerInfo {
        ContainerInfo {
            container_id: id.to_string(),
            name: id.to_string(),
            image: "ubuntu:22.04".to_string(),
            platform: PlatformTarget::LinuxX86_64,
            status: ContainerStatus::Running,
            ports: vec![],
            resource_usage: ContainerStats::default(),
            created_at: SystemTime::now(),
            started_at: Some(SystemTime::now()),
        }
    }

    fn cloud_instance_with(
        status: CloudInstanceStatus,
        config: HashMap<String, String>,
        public_ip: Option<String>,
    ) -> CloudInstance {
        CloudInstance {
            instance_id: "i-testinstance".to_string(),
            provider: "test".to_string(),
            instance_type: "t3.micro".to_string(),
            platform: PlatformTarget::LinuxX86_64,
            status,
            public_ip,
            private_ip: None,
            launch_time: SystemTime::now(),
            cost_per_hour: 0.0,
            config,
        }
    }

    #[tokio::test]
    async fn test_local_execution_maps_real_exit_status() {
        // Regression (FC1): execute_local_test must really run the scenario's
        // command and derive the status from its exit code, instead of sleeping
        // and reporting a fabricated `Passed`.
        let Some(host) = host_platform() else {
            // Unmapped host: the local path honestly refuses to run, which the
            // dedicated foreign-platform test already covers.
            return;
        };

        let orchestrator = orchestrator_with_scenario("ok", vec!["exit 0".to_string()]);
        let passed = orchestrator
            .execute_local_test(&context_for(host.clone(), "ok"))
            .await
            .expect("local execution succeeds");
        assert!(
            matches!(passed.status, TestStatus::Passed),
            "a command exiting 0 must be reported as Passed, got {:?}",
            passed.status
        );
        assert_eq!(passed.error_message, None);
        assert_eq!(
            passed.platform_details.get("last_exit_code"),
            Some(&"0".to_string())
        );
        assert_eq!(
            passed.platform_details.get("execution_target"),
            Some(&"local".to_string())
        );

        let orchestrator = orchestrator_with_scenario("bad", vec!["exit 3".to_string()]);
        let failed = orchestrator
            .execute_local_test(&context_for(host.clone(), "bad"))
            .await
            .expect("local execution succeeds");
        assert!(
            matches!(failed.status, TestStatus::Failed),
            "a command exiting non-zero must be reported as Failed, got {:?}",
            failed.status
        );
        assert_eq!(
            failed.platform_details.get("last_exit_code"),
            Some(&"3".to_string())
        );
        assert!(failed
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("exited with status 3")));
    }

    #[tokio::test]
    async fn test_local_execution_stops_at_first_failing_command() {
        let Some(host) = host_platform() else {
            return;
        };

        let orchestrator =
            orchestrator_with_scenario("chain", vec!["exit 1".to_string(), "exit 0".to_string()]);
        let result = orchestrator
            .execute_local_test(&context_for(host, "chain"))
            .await
            .expect("local execution succeeds");

        assert!(matches!(result.status, TestStatus::Failed));
        assert_eq!(
            result.platform_details.get("commands_total"),
            Some(&"2".to_string())
        );
        assert_eq!(
            result.platform_details.get("commands_executed"),
            Some(&"1".to_string()),
            "execution must stop at the first failing command"
        );
    }

    #[tokio::test]
    async fn test_execute_parallel_testing_runs_concurrently() {
        // Regression: execute_parallel_testing used to `await` every entry
        // inline inside its "batch" loop, so despite the name it never ran
        // more than one command at a time (see FC1 deferred notes). With N
        // independent local entries that each take ~0.4s and
        // max_concurrent_jobs >= N, real concurrency keeps the wall-clock time
        // close to a single entry's duration; sequential execution would take
        // roughly N * 0.4s and fails the bound below.
        let Some(host) = host_platform() else {
            return;
        };

        const N: usize = 4;
        let sleep_cmd = if cfg!(windows) {
            "ping -n 1 -w 400 127.0.0.1 > NUL".to_string()
        } else {
            "sleep 0.4".to_string()
        };

        let mut config = OrchestratorConfig {
            enable_cloud_testing: false,
            enable_container_testing: false,
            enable_parallel_testing: true,
            max_concurrent_jobs: N,
            ..Default::default()
        };
        config.matrix_config.test_scenarios = vec![TestScenario {
            name: "slow".to_string(),
            commands: vec![sleep_cmd],
            category: TestCategory::Functionality,
            timeout: Duration::from_secs(10),
            expected_results: HashMap::new(),
        }];
        let orchestrator =
            CrossPlatformOrchestrator::new(config).expect("orchestrator construction succeeds");

        let matrix: Vec<TestMatrixEntry> = (0..N)
            .map(|i| TestMatrixEntry {
                id: format!("entry_{i}"),
                platform: host.clone(),
                rust_version: "stable".to_string(),
                features: vec!["default".to_string()],
                optimization: OptimizationLevel::Debug,
                build_profile: "test".to_string(),
                scenarios: vec!["slow".to_string()],
                priority: i as u8,
                required_for_release: false,
                estimated_duration: Duration::from_secs(10),
                resource_requirements: HashMap::new(),
            })
            .collect();

        let mut allocations = HashMap::new();
        for entry in &matrix {
            let allocation_id = format!(
                "{}_{}",
                platform_target_to_string(&entry.platform),
                entry.priority
            );
            allocations.insert(
                allocation_id.clone(),
                ResourceAllocation {
                    id: allocation_id,
                    platform: entry.platform.clone(),
                    resource_type: AllocatedResourceType::Local,
                    allocated_at: SystemTime::now(),
                    estimated_completion: SystemTime::now() + entry.estimated_duration,
                    status: AllocationStatus::Available,
                    usage: ResourceUsage::default(),
                },
            );
        }

        let start = Instant::now();
        let results = orchestrator
            .execute_parallel_testing(&matrix, &allocations)
            .await
            .expect("parallel execution succeeds");
        let elapsed = start.elapsed();

        assert_eq!(results.len(), N);
        for result in &results {
            assert!(
                matches!(result.status, TestStatus::Passed),
                "expected Passed, got {:?} ({:?})",
                result.status,
                result.error_message
            );
        }
        // Sequential execution of N * 0.4s commands takes >= 1.6s (N=4).
        // Real bounded concurrency (max_concurrent_jobs=N) keeps wall clock
        // close to one command's duration; 1.2s leaves a wide margin above a
        // single 0.4s run while staying well under the 1.6s sequential floor.
        assert!(
            elapsed < Duration::from_millis(1200),
            "execute_parallel_testing took {:?} for {N} x 0.4s entries with \
             max_concurrent_jobs={N}; expected real concurrency to keep this \
             well under {N} * 0.4s",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_local_execution_refuses_foreign_platform() {
        // A matrix entry for a platform this host is not cannot be run locally;
        // it must be reported as PlatformNotSupported, never silently passed.
        let foreign = match host_platform() {
            Some(PlatformTarget::Custom(_)) | None => PlatformTarget::LinuxS390X,
            Some(PlatformTarget::LinuxS390X) => PlatformTarget::WindowsX86_64,
            Some(_) => PlatformTarget::LinuxS390X,
        };

        let orchestrator = orchestrator_with_scenario("foreign", vec!["exit 0".to_string()]);
        let result = orchestrator
            .execute_local_test(&context_for(foreign, "foreign"))
            .await
            .expect("local execution returns a result");

        assert!(
            matches!(result.status, TestStatus::PlatformNotSupported),
            "a foreign target platform must not be reported as Passed, got {:?}",
            result.status
        );
        assert!(result.error_message.is_some());
    }

    #[tokio::test]
    async fn test_container_execution_without_reachable_runtime_is_error() {
        // Regression (FC1): the container path must actually reach the container
        // runtime. Whether the runtime binary is missing, its daemon is down, or
        // the container does not exist, the result is an honest
        // ResourceUnavailable — not a fabricated `Passed`, and not `Failed`
        // (which would blame the code under test for an infrastructure problem).
        let orchestrator = orchestrator_with_scenario("unit", vec!["exit 0".to_string()]);
        let container = container_for("optirs_nonexistent_container_for_tests");

        let error = orchestrator
            .execute_container_test(
                &context_for(PlatformTarget::LinuxX86_64, "unit"),
                &container,
            )
            .await
            .expect_err("an unreachable container must not produce a test result");

        assert!(
            matches!(error, OptimError::ResourceUnavailable(_)),
            "expected ResourceUnavailable for an unreachable container runtime, got {:?}",
            error
        );
    }

    #[tokio::test]
    async fn test_cloud_execution_without_access_method_is_error() {
        // Regression (FC1): a provisioned instance with no configured access
        // method cannot run anything. Returning `Passed` would fabricate a
        // result, so an honest error is required.
        let orchestrator = orchestrator_with_scenario("unit", vec!["exit 0".to_string()]);
        let context = context_for(PlatformTarget::LinuxX86_64, "unit");

        let no_address = cloud_instance_with(
            CloudInstanceStatus::Running,
            HashMap::new(),
            None, // no public IP either
        );
        let error = orchestrator
            .execute_cloud_test(&context, &no_address)
            .await
            .expect_err("an instance with no address must not produce a test result");
        assert!(matches!(error, OptimError::UnsupportedOperation(_)));

        // Reachable address but no login user is still "no access method".
        let no_user = cloud_instance_with(
            CloudInstanceStatus::Running,
            HashMap::new(),
            Some("203.0.113.10".to_string()),
        );
        let error = orchestrator
            .execute_cloud_test(&context, &no_user)
            .await
            .expect_err("an instance with no ssh user must not produce a test result");
        assert!(matches!(error, OptimError::UnsupportedOperation(_)));

        // A non-running instance is an unavailable resource.
        let mut config = HashMap::new();
        config.insert("ssh_user".to_string(), "runner".to_string());
        let pending = cloud_instance_with(
            CloudInstanceStatus::Pending,
            config,
            Some("203.0.113.10".to_string()),
        );
        let error = orchestrator
            .execute_cloud_test(&context, &pending)
            .await
            .expect_err("a non-running instance must not produce a test result");
        assert!(matches!(error, OptimError::ResourceUnavailable(_)));
    }

    #[test]
    fn test_ssh_invocation_uses_configured_access() {
        let mut config = HashMap::new();
        config.insert("ssh_user".to_string(), "runner".to_string());
        config.insert("ssh_port".to_string(), "2222".to_string());
        config.insert("ssh_key".to_string(), "/keys/id_ed25519".to_string());
        let instance = cloud_instance_with(
            CloudInstanceStatus::Running,
            config,
            Some("203.0.113.10".to_string()),
        );

        let access = CrossPlatformOrchestrator::resolve_ssh_access(&instance)
            .expect("configured access resolves");
        assert_eq!(access.user, "runner");
        assert_eq!(access.host, "203.0.113.10");
        assert_eq!(access.port, Some(2222));

        let (program, args) = CrossPlatformOrchestrator::build_invocation(
            &ResolvedTarget::Cloud(access),
            &PlatformTarget::LinuxX86_64,
            "cargo test",
            &[],
        );
        assert_eq!(program, "ssh");
        assert!(args.contains(&"runner@203.0.113.10".to_string()));
        assert!(args.contains(&"cargo test".to_string()));
        assert!(args.contains(&"/keys/id_ed25519".to_string()));
        assert!(args.contains(&"2222".to_string()));
    }

    #[tokio::test]
    async fn test_run_process_missing_program_is_resource_unavailable() {
        // Spawning a program that does not exist is a capability problem, not a
        // test failure, and must surface as ResourceUnavailable.
        let error = run_process(
            "optirs_definitely_missing_binary_for_tests",
            &[],
            &[],
            Duration::from_secs(5),
        )
        .await
        .expect_err("spawning a missing binary must fail");
        assert!(matches!(error, OptimError::ResourceUnavailable(_)));
    }

    #[tokio::test]
    async fn test_unknown_scenario_is_configuration_error() {
        let orchestrator = orchestrator_with_scenario("known", vec!["exit 0".to_string()]);
        let context = context_for(PlatformTarget::LinuxX86_64, "unknown_scenario");
        let error = orchestrator
            .execute_local_test(&context)
            .await
            .expect_err("an undefined scenario must not silently pass");
        assert!(matches!(error, OptimError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_scenario_without_commands_is_skipped_not_passed() {
        let orchestrator = orchestrator_with_scenario("empty", vec![]);
        let result = orchestrator
            .execute_local_test(&context_for(PlatformTarget::LinuxX86_64, "empty"))
            .await
            .expect("an empty scenario returns a result");
        assert!(
            matches!(result.status, TestStatus::Skipped),
            "a scenario with no commands must be Skipped, not Passed"
        );
        assert!(result.error_message.is_some());
    }

    #[test]
    fn test_parse_metrics_from_output_uses_real_units() {
        // Metrics come from what the command actually printed, with correct unit
        // conversion; anything absent stays at zero (= not measured).
        let output = "\
throughput: 2500 ops/s
latency: 1.5 ms
peak_memory: 128 MiB
cpu: 42.5%
";
        let metrics = parse_metrics_from_output(output);
        assert!((metrics.throughput - 2500.0).abs() < 1e-9);
        assert!(
            (metrics.latency - 0.0015).abs() < 1e-12,
            "1.5 ms must become 0.0015 s, got {}",
            metrics.latency
        );
        assert_eq!(metrics.memory_usage, 128 * 1024 * 1024);
        assert!((metrics.cpu_usage - 42.5).abs() < 1e-9);
        assert_eq!(metrics.energy_consumption, None);

        // Nothing recognisable: every field stays at the "not measured" zero
        // rather than acquiring an invented default.
        let empty = parse_metrics_from_output("running 3 tests\nall good\n");
        assert_eq!(empty.throughput, 0.0);
        assert_eq!(empty.latency, 0.0);
        assert_eq!(empty.memory_usage, 0);
        assert_eq!(empty.cpu_usage, 0.0);

        // A `timestamp:` prefix must not be mistaken for a `time:` metric.
        let decoy = parse_metrics_from_output("timestamp: 1700000000\n");
        assert_eq!(decoy.latency, 0.0);

        // The last matching line wins, and criterion-style brackets parse.
        let repeated = parse_metrics_from_output("time: 5 s\ntime: [250.0 ms 251.0 ms]\n");
        assert!((repeated.latency - 0.25).abs() < 1e-12);
    }
}
