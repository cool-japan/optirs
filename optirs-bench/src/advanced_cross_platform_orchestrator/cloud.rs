// Cloud provider implementations for cross-platform testing
//
// This module provides cloud provider abstractions and implementations
// for AWS, Azure, GCP, GitHub Actions, and custom cloud providers.

use crate::error::{OptimError, Result};
// Only exercised by the unit tests below -- gated so a non-test build does not
// warn about unused imports.
#[cfg(test)]
use std::collections::HashMap;

use super::config::*;
use super::types::*;

/// Cloud provider trait
#[async_trait::async_trait]
pub trait CloudProvider: Send + Sync + std::fmt::Debug {
    async fn provision_instance(&self, platform: &PlatformTarget) -> Result<CloudInstance>;
    async fn terminate_instance(&self, instance_id: &str) -> Result<()>;
    async fn get_instance_status(&self, instance_id: &str) -> Result<CloudInstanceStatus>;
    fn get_provider_name(&self) -> &str;
}

/// AWS provider implementation
#[derive(Debug)]
pub struct AwsProvider {
    config: AwsConfig,
}

/// Azure provider implementation
#[derive(Debug)]
pub struct AzureProvider {
    config: AzureConfig,
}

/// GCP provider implementation
#[derive(Debug)]
pub struct GcpProvider {
    config: GcpConfig,
}

/// GitHub Actions provider implementation
#[derive(Debug)]
pub struct GitHubActionsProvider {
    config: GitHubActionsConfig,
}

/// Custom cloud provider implementation
#[derive(Debug)]
pub struct CustomProvider {
    config: CustomCloudConfig,
}

impl AwsProvider {
    pub fn new(config: AwsConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

#[async_trait::async_trait]
impl CloudProvider for AwsProvider {
    async fn provision_instance(&self, platform: &PlatformTarget) -> Result<CloudInstance> {
        // Real EC2 provisioning requires the AWS SDK plus valid credentials, neither
        // of which is available in this build. Refuse to fabricate an instance with a
        // made-up IP and status; return an explicit, honest error instead.
        Err(OptimError::UnsupportedOperation(format!(
            "AWS EC2 provisioning for platform {} in region {} is not available: it \
             requires a configured AWS SDK and credentials, which are not present in \
             this build",
            platform, self.config.region
        )))
    }

    async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot terminate AWS instance {} in region {}: a configured AWS SDK and \
             credentials are not available in this build",
            instance_id, self.config.region
        )))
    }

    async fn get_instance_status(&self, instance_id: &str) -> Result<CloudInstanceStatus> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot query AWS status for instance {} in region {}: a configured AWS \
             SDK and credentials are not available in this build",
            instance_id, self.config.region
        )))
    }

    fn get_provider_name(&self) -> &str {
        "aws"
    }
}

impl AzureProvider {
    pub fn new(config: AzureConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

#[async_trait::async_trait]
impl CloudProvider for AzureProvider {
    async fn provision_instance(&self, platform: &PlatformTarget) -> Result<CloudInstance> {
        // Real Azure VM provisioning requires the Azure SDK plus valid credentials;
        // neither is available here. Do not fabricate a VM with a made-up IP/status.
        Err(OptimError::UnsupportedOperation(format!(
            "Azure VM provisioning for platform {} (subscription {}, resource group \
             {}) is not available: it requires a configured Azure SDK and \
             credentials, which are not present in this build",
            platform, self.config.subscription_id, self.config.resource_group
        )))
    }

    async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot terminate Azure VM {} (subscription {}): a configured Azure SDK \
             and credentials are not available in this build",
            instance_id, self.config.subscription_id
        )))
    }

    async fn get_instance_status(&self, instance_id: &str) -> Result<CloudInstanceStatus> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot query Azure status for VM {} (subscription {}): a configured \
             Azure SDK and credentials are not available in this build",
            instance_id, self.config.subscription_id
        )))
    }

    fn get_provider_name(&self) -> &str {
        "azure"
    }
}

impl GcpProvider {
    pub fn new(config: GcpConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

#[async_trait::async_trait]
impl CloudProvider for GcpProvider {
    async fn provision_instance(&self, platform: &PlatformTarget) -> Result<CloudInstance> {
        // Real GCE provisioning requires the Google Cloud SDK plus valid
        // credentials; neither is available here. Do not fabricate an instance.
        Err(OptimError::UnsupportedOperation(format!(
            "GCP Compute Engine provisioning for platform {} (project {}, zone {}) is \
             not available: it requires a configured Google Cloud SDK and \
             credentials, which are not present in this build",
            platform, self.config.project_id, self.config.zone
        )))
    }

    async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot terminate GCP instance {} (project {}): a configured Google Cloud \
             SDK and credentials are not available in this build",
            instance_id, self.config.project_id
        )))
    }

    async fn get_instance_status(&self, instance_id: &str) -> Result<CloudInstanceStatus> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot query GCP status for instance {} (project {}): a configured \
             Google Cloud SDK and credentials are not available in this build",
            instance_id, self.config.project_id
        )))
    }

    fn get_provider_name(&self) -> &str {
        "gcp"
    }
}

impl GitHubActionsProvider {
    pub fn new(config: GitHubActionsConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

#[async_trait::async_trait]
impl CloudProvider for GitHubActionsProvider {
    async fn provision_instance(&self, platform: &PlatformTarget) -> Result<CloudInstance> {
        // Dispatching a GitHub Actions runner requires the GitHub API plus a token,
        // neither of which is available here. Do not fabricate a runner instance.
        Err(OptimError::UnsupportedOperation(format!(
            "GitHub Actions runner provisioning for platform {} on repository {} is \
             not available: it requires the GitHub API and an access token, which are \
             not present in this build",
            platform, self.config.repository
        )))
    }

    async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        // GitHub Actions ephemeral runners are torn down by the platform itself; there
        // is genuinely nothing to terminate from here, so this is an honest no-op.
        log::info!(
            "GitHub Actions runner {} is managed and cleaned up by GitHub; no local \
             termination required",
            instance_id
        );
        Ok(())
    }

    async fn get_instance_status(&self, instance_id: &str) -> Result<CloudInstanceStatus> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot query GitHub Actions runner status for {} on repository {}: the \
             GitHub API and an access token are not available in this build",
            instance_id, self.config.repository
        )))
    }

    fn get_provider_name(&self) -> &str {
        "github"
    }
}

impl CustomProvider {
    pub fn new(config: CustomCloudConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

#[async_trait::async_trait]
impl CloudProvider for CustomProvider {
    async fn provision_instance(&self, platform: &PlatformTarget) -> Result<CloudInstance> {
        // A custom provider would need a real client for its endpoint plus
        // credentials; none is implemented here. Do not fabricate an instance.
        Err(OptimError::UnsupportedOperation(format!(
            "Custom provider '{}' provisioning for platform {} is not available: it \
             requires a real client for endpoint '{}' and credentials, which are not \
             implemented in this build",
            self.config.name, platform, self.config.endpoint
        )))
    }

    async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot terminate custom provider '{}' instance {}: a real client for \
             endpoint '{}' is not implemented in this build",
            self.config.name, instance_id, self.config.endpoint
        )))
    }

    async fn get_instance_status(&self, instance_id: &str) -> Result<CloudInstanceStatus> {
        Err(OptimError::UnsupportedOperation(format!(
            "Cannot query custom provider '{}' status for instance {}: a real client \
             for endpoint '{}' is not implemented in this build",
            self.config.name, instance_id, self.config.endpoint
        )))
    }

    fn get_provider_name(&self) -> &str {
        &self.config.name
    }
}

/// Enum wrapper for cloud providers to avoid trait object issues
#[derive(Debug)]
pub enum CloudProviderEnum {
    Aws(AwsProvider),
    Azure(AzureProvider),
    Gcp(GcpProvider),
    GitHub(GitHubActionsProvider),
    Custom(CustomProvider),
}

#[async_trait::async_trait]
impl CloudProvider for CloudProviderEnum {
    async fn provision_instance(&self, platform: &PlatformTarget) -> Result<CloudInstance> {
        match self {
            CloudProviderEnum::Aws(provider) => provider.provision_instance(platform).await,
            CloudProviderEnum::Azure(provider) => provider.provision_instance(platform).await,
            CloudProviderEnum::Gcp(provider) => provider.provision_instance(platform).await,
            CloudProviderEnum::GitHub(provider) => provider.provision_instance(platform).await,
            CloudProviderEnum::Custom(provider) => provider.provision_instance(platform).await,
        }
    }

    async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        match self {
            CloudProviderEnum::Aws(provider) => provider.terminate_instance(instance_id).await,
            CloudProviderEnum::Azure(provider) => provider.terminate_instance(instance_id).await,
            CloudProviderEnum::Gcp(provider) => provider.terminate_instance(instance_id).await,
            CloudProviderEnum::GitHub(provider) => provider.terminate_instance(instance_id).await,
            CloudProviderEnum::Custom(provider) => provider.terminate_instance(instance_id).await,
        }
    }

    async fn get_instance_status(&self, instance_id: &str) -> Result<CloudInstanceStatus> {
        match self {
            CloudProviderEnum::Aws(provider) => provider.get_instance_status(instance_id).await,
            CloudProviderEnum::Azure(provider) => provider.get_instance_status(instance_id).await,
            CloudProviderEnum::Gcp(provider) => provider.get_instance_status(instance_id).await,
            CloudProviderEnum::GitHub(provider) => provider.get_instance_status(instance_id).await,
            CloudProviderEnum::Custom(provider) => provider.get_instance_status(instance_id).await,
        }
    }

    fn get_provider_name(&self) -> &str {
        match self {
            CloudProviderEnum::Aws(provider) => provider.get_provider_name(),
            CloudProviderEnum::Azure(provider) => provider.get_provider_name(),
            CloudProviderEnum::Gcp(provider) => provider.get_provider_name(),
            CloudProviderEnum::GitHub(provider) => provider.get_provider_name(),
            CloudProviderEnum::Custom(provider) => provider.get_provider_name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_aws_provider_provisioning_is_honest_error() {
        // Regression (F76): without a cloud SDK + credentials, provisioning must NOT
        // fabricate an instance with a made-up IP and "Running" status. It must return
        // an explicit error, and status/termination queries must be honest errors too.
        let config = AwsConfig {
            region: "us-east-1".to_string(),
            instance_types: HashMap::new(),
            ami_mappings: HashMap::new(),
            vpc_id: None,
            subnet_id: None,
            security_groups: vec![],
            key_pair: None,
            iam_role: None,
            use_spot_instances: false,
            max_spot_price: None,
        };

        let provider = AwsProvider::new(config).expect("provider construction succeeds");
        assert!(
            provider
                .provision_instance(&PlatformTarget::LinuxX86_64)
                .await
                .is_err(),
            "AWS provisioning must not fabricate an instance without a cloud SDK"
        );
        assert!(
            provider
                .get_instance_status("i-doesnotexist")
                .await
                .is_err(),
            "AWS status query must be an honest error, not unconditional Running"
        );
        assert!(provider.terminate_instance("i-doesnotexist").await.is_err());
    }

    #[tokio::test]
    async fn test_github_provider_provisioning_is_honest_error() {
        // Regression (F76): the GitHub Actions runner path also fabricated an instance
        // id and unconditional Running status; both must now be honest.
        let config = GitHubActionsConfig {
            repository: "test/repo".to_string(),
            workflow_templates: HashMap::new(),
            runner_labels: HashMap::new(),
            secrets: vec![],
            matrix_strategy: "matrix".to_string(),
        };

        let provider = GitHubActionsProvider::new(config).expect("provider construction succeeds");
        assert!(
            provider
                .provision_instance(&PlatformTarget::LinuxX86_64)
                .await
                .is_err(),
            "GitHub Actions provisioning must not fabricate a runner"
        );
        assert!(
            provider
                .get_instance_status("gh-doesnotexist")
                .await
                .is_err(),
            "GitHub Actions status query must be an honest error"
        );
    }
}
