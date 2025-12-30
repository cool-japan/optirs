// Reproducibility tools for research experiments
//
// This module provides tools to ensure research experiments can be reproduced
// exactly, including environment capture, dependency tracking, and result verification.

use crate::error::{OptimError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Reproducibility manager for tracking experiment reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityManager {
    /// Environment snapshots
    pub environments: HashMap<String, EnvironmentSnapshot>,
    /// Reproducibility reports
    pub reports: Vec<ReproducibilityReport>,
    /// Verification results
    pub verifications: Vec<VerificationResult>,
    /// Configuration
    pub config: ReproducibilityConfig,
}

/// Complete environment snapshot for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    /// Snapshot identifier
    pub id: String,
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// System information
    pub system_info: SystemInfo,
    /// Software dependencies
    pub dependencies: Vec<Dependency>,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
    /// Hardware configuration
    pub hardware_config: HardwareConfig,
    /// Random seeds
    pub random_seeds: Vec<u64>,
    /// Data checksums
    pub data_checksums: HashMap<String, String>,
    /// Configuration hashes
    pub config_hashes: HashMap<String, String>,
}

/// System information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// OS version
    pub os_version: String,
    /// Kernel version
    pub kernel_version: Option<String>,
    /// Architecture
    pub architecture: String,
    /// Hostname
    pub hostname: String,
    /// Timezone
    pub timezone: String,
    /// Locale settings
    pub locale: HashMap<String, String>,
}

/// Software dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Package name
    pub name: String,
    /// Version
    pub version: String,
    /// Source/registry
    pub source: String,
    /// Checksum
    pub checksum: Option<String>,
    /// Installation path
    pub install_path: Option<String>,
}

/// Hardware configuration for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// CPU information
    pub cpu: CpuSpec,
    /// Memory information
    pub memory: MemorySpec,
    /// GPU information
    pub gpu: Option<GpuSpec>,
    /// Storage information
    pub storage: Vec<StorageSpec>,
}

/// CPU specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSpec {
    /// CPU model
    pub model: String,
    /// Number of cores
    pub cores: usize,
    /// Number of threads
    pub threads: usize,
    /// Base frequency (MHz)
    pub base_frequency: u32,
    /// Max frequency (MHz)
    pub max_frequency: u32,
    /// Cache information
    pub cache: HashMap<String, String>,
    /// CPU flags/features
    pub flags: Vec<String>,
}

/// Memory specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySpec {
    /// Total memory (bytes)
    pub total_bytes: u64,
    /// Available memory (bytes)
    pub available_bytes: u64,
    /// Memory type (DDR4, etc.)
    pub memory_type: String,
    /// Memory speed (MHz)
    pub speed_mhz: u32,
}

/// GPU specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSpec {
    /// GPU model
    pub model: String,
    /// GPU memory (bytes)
    pub memory_bytes: u64,
    /// Driver version
    pub driver_version: String,
    /// CUDA version (if applicable)
    pub cuda_version: Option<String>,
    /// Compute capability
    pub compute_capability: Option<String>,
}

/// Storage specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSpec {
    /// Device name
    pub device: String,
    /// Storage type (SSD, HDD, etc.)
    pub storage_type: String,
    /// Total size (bytes)
    pub size_bytes: u64,
    /// Available space (bytes)
    pub available_bytes: u64,
    /// File system
    pub filesystem: String,
}

/// Reproducibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityReport {
    /// Report ID
    pub id: String,
    /// Experiment ID
    pub experiment_id: String,
    /// Environment snapshot ID
    pub environment_id: String,
    /// Reproducibility score
    pub reproducibility_score: f64,
    /// Checklist results
    pub checklist: ReproducibilityChecklist,
    /// Issues found
    pub issues: Vec<ReproducibilityIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Reproducibility checklist
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityChecklist {
    /// Random seed documented
    pub random_seed_documented: bool,
    /// Dependencies pinned
    pub dependencies_pinned: bool,
    /// Environment captured
    pub environment_captured: bool,
    /// Data versioned
    pub data_versioned: bool,
    /// Code versioned
    pub code_versioned: bool,
    /// Hardware documented
    pub hardware_documented: bool,
    /// Configuration hashed
    pub configuration_hashed: bool,
    /// Results verified
    pub results_verified: bool,
}

/// Reproducibility issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityIssue {
    /// Issue type
    pub issue_type: IssueType,
    /// Severity level
    pub severity: IssueSeverity,
    /// Description
    pub description: String,
    /// Affected component
    pub component: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Types of reproducibility issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IssueType {
    /// Missing random seed
    MissingRandomSeed,
    /// Unpinned dependencies
    UnpinnedDependencies,
    /// Missing environment info
    MissingEnvironment,
    /// Data not versioned
    DataNotVersioned,
    /// Code not versioned
    CodeNotVersioned,
    /// Hardware not documented
    HardwareNotDocumented,
    /// Configuration not hashed
    ConfigurationNotHashed,
    /// Non-deterministic behavior
    NonDeterministic,
    /// Platform-specific code
    PlatformSpecific,
    /// External dependencies
    ExternalDependencies,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Critical issue - prevents reproducibility
    Critical,
    /// High severity - likely to affect reproducibility
    High,
    /// Medium severity - may affect reproducibility
    Medium,
    /// Low severity - minor impact on reproducibility
    Low,
    /// Info only
    Info,
}

/// Verification result for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Verification ID
    pub id: String,
    /// Original experiment ID
    pub original_experiment_id: String,
    /// Reproduction experiment ID
    pub reproduction_experiment_id: String,
    /// Verification status
    pub status: VerificationStatus,
    /// Similarity metrics
    pub similarity_metrics: SimilarityMetrics,
    /// Differences found
    pub differences: Vec<Difference>,
    /// Verification timestamp
    pub verified_at: DateTime<Utc>,
}

/// Verification status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Exact reproduction
    ExactMatch,
    /// Close reproduction (within tolerance)
    CloseMatch,
    /// Partial reproduction
    PartialMatch,
    /// No match
    NoMatch,
    /// Verification failed
    VerificationFailed,
}

/// Similarity metrics between original and reproduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMetrics {
    /// Overall similarity score (0.0 to 1.0)
    pub overall_similarity: f64,
    /// Result similarity
    pub result_similarity: f64,
    /// Performance similarity
    pub performance_similarity: f64,
    /// Configuration similarity
    pub configuration_similarity: f64,
    /// Environment similarity
    pub environment_similarity: f64,
}

/// Difference between original and reproduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Difference {
    /// Category of difference
    pub category: DifferenceCategory,
    /// Field or metric name
    pub field: String,
    /// Original value
    pub original_value: String,
    /// Reproduction value
    pub reproduction_value: String,
    /// Difference magnitude
    pub magnitude: f64,
    /// Significance
    pub significant: bool,
}

/// Categories of differences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DifferenceCategory {
    /// Difference in results
    Results,
    /// Difference in performance
    Performance,
    /// Difference in configuration
    Configuration,
    /// Difference in environment
    Environment,
    /// Difference in dependencies
    Dependencies,
    /// Difference in hardware
    Hardware,
}

/// Reproducibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityConfig {
    /// Tolerance for numerical comparisons
    pub numerical_tolerance: f64,
    /// Tolerance for performance comparisons
    pub performance_tolerance: f64,
    /// Minimum reproducibility score
    pub min_reproducibility_score: f64,
    /// Auto-capture environment
    pub auto_capture_environment: bool,
    /// Auto-verify results
    pub auto_verify_results: bool,
    /// Storage settings
    pub storage: ReproducibilityStorage,
}

/// Storage settings for reproducibility data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityStorage {
    /// Base storage directory
    pub base_directory: PathBuf,
    /// Compress snapshots
    pub compress_snapshots: bool,
    /// Retention period (days)
    pub retention_days: u32,
    /// Maximum storage size (bytes)
    pub max_storage_bytes: u64,
}

impl ReproducibilityManager {
    /// Create a new reproducibility manager
    pub fn new(config: ReproducibilityConfig) -> Self {
        Self {
            environments: HashMap::new(),
            reports: Vec::new(),
            verifications: Vec::new(),
            config,
        }
    }

    /// Capture current environment snapshot
    pub fn capture_environment(&mut self) -> Result<String> {
        let snapshot_id = uuid::Uuid::new_v4().to_string();
        let snapshot = EnvironmentSnapshot {
            id: snapshot_id.clone(),
            timestamp: Utc::now(),
            system_info: self.capture_system_info()?,
            dependencies: self.capture_dependencies()?,
            environment_variables: self.capture_environment_variables(),
            hardware_config: self.capture_hardware_config()?,
            random_seeds: vec![42], // Default seed
            data_checksums: HashMap::new(),
            config_hashes: HashMap::new(),
        };

        self.environments.insert(snapshot_id.clone(), snapshot);
        Ok(snapshot_id)
    }

    /// Generate reproducibility report for experiment
    pub fn generate_report(&mut self, experiment_id: &str, environment_id: &str) -> Result<String> {
        let environment = self.environments.get(environment_id).ok_or_else(|| {
            OptimError::InvalidConfig("Environment snapshot not found".to_string())
        })?;

        let checklist = self.evaluate_checklist(environment);
        let (score, issues) = self.calculate_reproducibility_score(&checklist, environment);
        let recommendations = self.generate_recommendations(&issues);

        let report_id = uuid::Uuid::new_v4().to_string();
        let report = ReproducibilityReport {
            id: report_id.clone(),
            experiment_id: experiment_id.to_string(),
            environment_id: environment_id.to_string(),
            reproducibility_score: score,
            checklist,
            issues,
            recommendations,
            generated_at: Utc::now(),
        };

        self.reports.push(report);
        Ok(report_id)
    }

    /// Verify reproducibility between two experiments
    pub fn verify_reproducibility(
        &mut self,
        original_experiment_id: &str,
        reproduction_experiment_id: &str,
    ) -> Result<String> {
        // This would compare the actual experiment results
        // For now, we'll create a placeholder verification

        let verification_id = uuid::Uuid::new_v4().to_string();
        let verification = VerificationResult {
            id: verification_id.clone(),
            original_experiment_id: original_experiment_id.to_string(),
            reproduction_experiment_id: reproduction_experiment_id.to_string(),
            status: VerificationStatus::CloseMatch, // Placeholder
            similarity_metrics: SimilarityMetrics {
                overall_similarity: 0.95,
                result_similarity: 0.98,
                performance_similarity: 0.92,
                configuration_similarity: 1.0,
                environment_similarity: 0.90,
            },
            differences: Vec::new(),
            verified_at: Utc::now(),
        };

        self.verifications.push(verification);
        Ok(verification_id)
    }

    fn capture_system_info(&self) -> Result<SystemInfo> {
        Ok(SystemInfo {
            os: std::env::consts::OS.to_string(),
            os_version: "Unknown".to_string(), // Would use system APIs
            kernel_version: None,
            architecture: std::env::consts::ARCH.to_string(),
            hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
            timezone: "UTC".to_string(), // Would detect actual timezone
            locale: HashMap::new(),
        })
    }

    fn capture_dependencies(&self) -> Result<Vec<Dependency>> {
        // In a real implementation, this would parse Cargo.lock, requirements.txt, etc.
        Ok(vec![Dependency {
            name: "scirs2-optim".to_string(),
            version: "0.1.0".to_string(),
            source: "local".to_string(),
            checksum: None,
            install_path: None,
        }])
    }

    fn capture_environment_variables(&self) -> HashMap<String, String> {
        std::env::vars().collect()
    }

    fn capture_hardware_config(&self) -> Result<HardwareConfig> {
        Ok(HardwareConfig {
            cpu: CpuSpec {
                model: "Unknown CPU".to_string(),
                cores: std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1),
                threads: std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1),
                base_frequency: 0,
                max_frequency: 0,
                cache: HashMap::new(),
                flags: Vec::new(),
            },
            memory: MemorySpec {
                total_bytes: 8 * 1024 * 1024 * 1024,     // 8GB default
                available_bytes: 6 * 1024 * 1024 * 1024, // 6GB default
                memory_type: "Unknown".to_string(),
                speed_mhz: 0,
            },
            gpu: None,
            storage: Vec::new(),
        })
    }

    fn evaluate_checklist(&self, environment: &EnvironmentSnapshot) -> ReproducibilityChecklist {
        ReproducibilityChecklist {
            random_seed_documented: !environment.random_seeds.is_empty(),
            dependencies_pinned: !environment.dependencies.is_empty(),
            environment_captured: true, // We have the snapshot
            data_versioned: !environment.data_checksums.is_empty(),
            code_versioned: false,     // Would check git info
            hardware_documented: true, // We captured hardware info
            configuration_hashed: !environment.config_hashes.is_empty(),
            results_verified: false, // Would check verification status
        }
    }

    fn calculate_reproducibility_score(
        &self,
        checklist: &ReproducibilityChecklist,
        environment: &EnvironmentSnapshot,
    ) -> (f64, Vec<ReproducibilityIssue>) {
        let mut score = 0.0;
        let mut issues = Vec::new();
        let total_checks = 8.0;

        if checklist.random_seed_documented {
            score += 1.0;
        } else {
            issues.push(ReproducibilityIssue {
                issue_type: IssueType::MissingRandomSeed,
                severity: IssueSeverity::High,
                description: "Random seed not documented".to_string(),
                component: "Random Number Generation".to_string(),
                suggested_fix: Some("Set and document random seeds for all RNGs".to_string()),
            });
        }

        if checklist.dependencies_pinned {
            score += 1.0;
        } else {
            issues.push(ReproducibilityIssue {
                issue_type: IssueType::UnpinnedDependencies,
                severity: IssueSeverity::Critical,
                description: "Dependencies not pinned to specific versions".to_string(),
                component: "Dependencies".to_string(),
                suggested_fix: Some("Pin all dependencies to exact versions".to_string()),
            });
        }

        if checklist.environment_captured {
            score += 1.0;
        }

        if checklist.data_versioned {
            score += 1.0;
        } else {
            issues.push(ReproducibilityIssue {
                issue_type: IssueType::DataNotVersioned,
                severity: IssueSeverity::High,
                description: "Data not versioned or checksummed".to_string(),
                component: "Data Management".to_string(),
                suggested_fix: Some("Version control data or provide checksums".to_string()),
            });
        }

        if checklist.code_versioned {
            score += 1.0;
        } else {
            issues.push(ReproducibilityIssue {
                issue_type: IssueType::CodeNotVersioned,
                severity: IssueSeverity::Critical,
                description: "Code not under version control".to_string(),
                component: "Source Code".to_string(),
                suggested_fix: Some("Use Git or other version control system".to_string()),
            });
        }

        if checklist.hardware_documented {
            score += 1.0;
        }

        if checklist.configuration_hashed {
            score += 1.0;
        } else {
            issues.push(ReproducibilityIssue {
                issue_type: IssueType::ConfigurationNotHashed,
                severity: IssueSeverity::Medium,
                description: "Configuration not hashed for integrity".to_string(),
                component: "Configuration".to_string(),
                suggested_fix: Some("Generate and store configuration hashes".to_string()),
            });
        }

        if checklist.results_verified {
            score += 1.0;
        }

        (score / total_checks, issues)
    }

    fn generate_recommendations(&self, issues: &[ReproducibilityIssue]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for issue in issues {
            if let Some(fix) = &issue.suggested_fix {
                recommendations.push(format!("{}: {}", issue.component, fix));
            }
        }

        if issues
            .iter()
            .any(|i| i.issue_type == IssueType::MissingRandomSeed)
        {
            recommendations.push("Use consistent random seeds across all components".to_string());
        }

        if issues
            .iter()
            .any(|i| i.issue_type == IssueType::UnpinnedDependencies)
        {
            recommendations.push("Create a lockfile with exact dependency versions".to_string());
        }

        recommendations.push("Document the complete experimental procedure".to_string());
        recommendations.push("Provide clear instructions for reproduction".to_string());

        recommendations
    }
}

impl Default for ReproducibilityConfig {
    fn default() -> Self {
        Self {
            numerical_tolerance: 1e-6,
            performance_tolerance: 0.1,     // 10%
            min_reproducibility_score: 0.8, // 80%
            auto_capture_environment: true,
            auto_verify_results: false,
            storage: ReproducibilityStorage {
                base_directory: PathBuf::from("./reproducibility"),
                compress_snapshots: true,
                retention_days: 365,
                max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10GB
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducibility_manager_creation() {
        let config = ReproducibilityConfig::default();
        let manager = ReproducibilityManager::new(config);

        assert!(manager.environments.is_empty());
        assert!(manager.reports.is_empty());
        assert!(manager.verifications.is_empty());
    }

    #[test]
    fn test_environment_capture() {
        let config = ReproducibilityConfig::default();
        let mut manager = ReproducibilityManager::new(config);

        let snapshot_id = manager.capture_environment().unwrap();

        assert!(manager.environments.contains_key(&snapshot_id));
        let snapshot = &manager.environments[&snapshot_id];
        assert_eq!(snapshot.system_info.os, std::env::consts::OS);
    }

    #[test]
    fn test_reproducibility_report() {
        let config = ReproducibilityConfig::default();
        let mut manager = ReproducibilityManager::new(config);

        let env_id = manager.capture_environment().unwrap();
        let report_id = manager.generate_report("test_experiment", &env_id).unwrap();

        assert!(!manager.reports.is_empty());
        let report = &manager.reports[0];
        assert_eq!(report.id, report_id);
        assert_eq!(report.experiment_id, "test_experiment");
    }
}
