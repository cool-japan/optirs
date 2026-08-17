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
    /// Overall similarity score (0.0 to 1.0), averaged from whichever of
    /// the dimensions below were actually measured.
    pub overall_similarity: f64,
    /// Similarity of non-performance result metrics (accuracy, loss, final
    /// objective, ...), computed from the metrics maps passed to
    /// [`ReproducibilityManager::verify_reproducibility`].
    pub result_similarity: f64,
    /// Similarity of performance-labeled metrics (execution time, memory,
    /// throughput, ...) within the same metrics maps.
    pub performance_similarity: f64,
    /// Similarity of experiment configuration. `None` when no
    /// configuration comparison was performed, rather than a fabricated
    /// number presented as a real measurement.
    pub configuration_similarity: Option<f64>,
    /// Similarity of the captured environment snapshots. `None` unless both
    /// environment snapshot IDs were supplied to `verify_reproducibility`
    /// and found.
    pub environment_similarity: Option<f64>,
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

    /// Capture current environment snapshot.
    ///
    /// `random_seeds` should be the actual seed(s) the experiment being
    /// snapshotted used; pass an empty slice if none were recorded. This
    /// used to be hardcoded to `vec![42]` regardless of what the experiment
    /// actually did, which made the "random seed documented" checklist item
    /// trivially true for every snapshot rather than reflecting reality.
    pub fn capture_environment(&mut self, random_seeds: &[u64]) -> Result<String> {
        let snapshot_id = uuid::Uuid::new_v4().to_string();
        let snapshot = EnvironmentSnapshot {
            id: snapshot_id.clone(),
            timestamp: Utc::now(),
            system_info: self.capture_system_info()?,
            dependencies: self.capture_dependencies()?,
            environment_variables: self.capture_environment_variables(),
            hardware_config: self.capture_hardware_config()?,
            random_seeds: random_seeds.to_vec(),
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

        let checklist = self.evaluate_checklist(environment, experiment_id);
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

    /// Verify reproducibility between two experiment runs by comparing
    /// their actual final metrics.
    ///
    /// Every key present in either `original_metrics` or
    /// `reproduction_metrics` is compared (relative to
    /// `config.numerical_tolerance`); a key named with a performance-ish
    /// marker (time/memory/latency/throughput/duration/speed) is scored as
    /// `performance_similarity`, everything else as `result_similarity`, so
    /// wall-clock jitter between runs cannot mask (or be masked by) an
    /// actual difference in the computed result, or vice versa.
    ///
    /// When `original_environment_id`/`reproduction_environment_id` are
    /// both supplied and resolve to a captured [`EnvironmentSnapshot`],
    /// `environment_similarity` is computed for real from OS/architecture
    /// and pinned-dependency-set overlap; otherwise it is `None` rather
    /// than a fabricated number. `configuration_similarity` is always
    /// `None`: this function has no experiment-configuration data to
    /// compare against.
    pub fn verify_reproducibility(
        &mut self,
        original_experiment_id: &str,
        reproduction_experiment_id: &str,
        original_metrics: &HashMap<String, f64>,
        reproduction_metrics: &HashMap<String, f64>,
        original_environment_id: Option<&str>,
        reproduction_environment_id: Option<&str>,
    ) -> Result<String> {
        const PERFORMANCE_MARKERS: &[&str] = &[
            "time",
            "memory",
            "latency",
            "throughput",
            "duration",
            "speed",
        ];

        let mut keys: Vec<&String> = original_metrics
            .keys()
            .chain(reproduction_metrics.keys())
            .collect();
        keys.sort();
        keys.dedup();

        let mut differences = Vec::new();
        let mut result_diffs = Vec::new();
        let mut performance_diffs = Vec::new();

        for key in keys {
            let is_performance = PERFORMANCE_MARKERS
                .iter()
                .any(|marker| key.to_lowercase().contains(marker));
            let category = if is_performance {
                DifferenceCategory::Performance
            } else {
                DifferenceCategory::Results
            };

            let relative = match (original_metrics.get(key), reproduction_metrics.get(key)) {
                (Some(&orig), Some(&repro)) => {
                    let magnitude = orig
                        .abs()
                        .max(repro.abs())
                        .max(self.config.numerical_tolerance);
                    let relative = ((orig - repro).abs() / magnitude).min(1.0);
                    if relative > self.config.numerical_tolerance {
                        differences.push(Difference {
                            category,
                            field: key.clone(),
                            original_value: orig.to_string(),
                            reproduction_value: repro.to_string(),
                            magnitude: relative,
                            significant: relative > self.config.performance_tolerance,
                        });
                    }
                    relative
                }
                (orig, repro) => {
                    differences.push(Difference {
                        category,
                        field: key.clone(),
                        original_value: orig
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "<missing>".to_string()),
                        reproduction_value: repro
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "<missing>".to_string()),
                        magnitude: 1.0,
                        significant: true,
                    });
                    1.0
                }
            };

            if is_performance {
                performance_diffs.push(relative);
            } else {
                result_diffs.push(relative);
            }
        }

        // Nothing to compare in a bucket is vacuously "no difference found"
        // (1.0), which is distinct from the old bug of a fixed similarity
        // presented regardless of whether -- or how badly -- inputs
        // actually differed.
        let mean_similarity = |diffs: &[f64]| -> f64 {
            if diffs.is_empty() {
                1.0
            } else {
                1.0 - (diffs.iter().sum::<f64>() / diffs.len() as f64)
            }
        };
        let result_similarity = mean_similarity(&result_diffs);
        let performance_similarity = mean_similarity(&performance_diffs);

        let environment_similarity = match (original_environment_id, reproduction_environment_id) {
            (Some(orig_id), Some(repro_id)) => match (
                self.environments.get(orig_id),
                self.environments.get(repro_id),
            ) {
                (Some(orig_env), Some(repro_env)) => {
                    Some(environment_similarity(orig_env, repro_env))
                }
                _ => None,
            },
            _ => None,
        };

        let overall_similarity = {
            let mut parts = vec![result_similarity, performance_similarity];
            parts.extend(environment_similarity);
            parts.iter().sum::<f64>() / parts.len() as f64
        };

        let status = if differences.is_empty() {
            VerificationStatus::ExactMatch
        } else if overall_similarity >= 1.0 - self.config.performance_tolerance {
            VerificationStatus::CloseMatch
        } else if overall_similarity >= self.config.min_reproducibility_score {
            VerificationStatus::PartialMatch
        } else {
            VerificationStatus::NoMatch
        };

        let verification_id = uuid::Uuid::new_v4().to_string();
        let verification = VerificationResult {
            id: verification_id.clone(),
            original_experiment_id: original_experiment_id.to_string(),
            reproduction_experiment_id: reproduction_experiment_id.to_string(),
            status,
            similarity_metrics: SimilarityMetrics {
                overall_similarity,
                result_similarity,
                performance_similarity,
                configuration_similarity: None,
                environment_similarity,
            },
            differences,
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

    /// Parse the workspace's `Cargo.lock` for the exact locked version of
    /// every dependency (transitive included), which is what "pinned
    /// dependencies" actually means for a Rust project. Returns an empty
    /// list -- not a fabricated placeholder entry -- when no `Cargo.lock`
    /// can be found (e.g. running outside a checked-out repository).
    fn capture_dependencies(&self) -> Result<Vec<Dependency>> {
        let Some(lock_path) = find_cargo_lock() else {
            return Ok(Vec::new());
        };
        let content = std::fs::read_to_string(&lock_path).map_err(|e| {
            OptimError::InvalidConfig(format!("failed to read {}: {e}", lock_path.display()))
        })?;

        Ok(parse_cargo_lock_dependencies(&content))
    }

    /// Capture the subset of environment variables relevant to reproducing
    /// a run (toolchain/build configuration, locale, thread counts,
    /// accelerator visibility, ...), never the full process environment.
    ///
    /// A `EnvironmentSnapshot` is `Serialize`/`Deserialize` and is intended
    /// to be written to disk or shared between machines when debugging a
    /// reproducibility gap, so capturing `std::env::vars()` unfiltered would
    /// leak whatever secrets (API keys, tokens, cloud credentials, database
    /// URLs, ...) happen to be set in the researcher's shell into that
    /// artifact. Instead this uses an explicit allowlist of
    /// reproducibility-relevant names, and additionally redacts the value
    /// of any allowlisted variable whose name still looks secret-shaped (as
    /// defense in depth against e.g. a CI variable named `RUSTC_WRAPPER`
    /// being repurposed to smuggle a token).
    fn capture_environment_variables(&self) -> HashMap<String, String> {
        const ALLOWED_EXACT: &[&str] = &[
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "LC_NUMERIC",
            "TZ",
            "PATH",
            "HOSTNAME",
            "USER",
            "SHELL",
            "PWD",
            "OS",
            "OSTYPE",
            "HOSTTYPE",
            "RUSTC_VERSION",
            "RUSTFLAGS",
            "RUST_BACKTRACE",
            "RUST_LOG",
            "CARGO_HOME",
            "RUSTUP_HOME",
            "RUSTUP_TOOLCHAIN",
            "OMP_NUM_THREADS",
            "RAYON_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ];
        const ALLOWED_PREFIXES: &[&str] = &["CARGO_", "RUSTC_"];
        const SECRET_MARKERS: &[&str] = &[
            "KEY",
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "PASSWD",
            "CREDENTIAL",
            "AUTH",
            "PRIVATE",
            "APIKEY",
            "ACCESS",
            "COOKIE",
            "SESSION",
        ];

        std::env::vars()
            .filter(|(name, _)| {
                let upper = name.to_uppercase();
                ALLOWED_EXACT.contains(&upper.as_str())
                    || ALLOWED_PREFIXES.iter().any(|p| upper.starts_with(p))
            })
            .map(|(name, value)| {
                let upper = name.to_uppercase();
                if SECRET_MARKERS.iter().any(|marker| upper.contains(marker)) {
                    (name, "<redacted>".to_string())
                } else {
                    (name, value)
                }
            })
            .collect()
    }

    /// Capture real hardware facts via portable, pure-Rust means (spawning
    /// the OS's own introspection tools / reading its own `/proc` files --
    /// no FFI, no linked C libraries). Previously this returned a fixed
    /// "8GB / 6GB available" `MemorySpec` on every machine regardless of
    /// its actual capacity; `0` now means "not detected" rather than a
    /// specific, plausible-looking but wrong number being reported as fact.
    fn capture_hardware_config(&self) -> Result<HardwareConfig> {
        let cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let (model, base_frequency, max_frequency) = detect_cpu_info();
        let (total_bytes, available_bytes) = detect_memory_info();

        Ok(HardwareConfig {
            cpu: CpuSpec {
                model,
                cores,
                threads: cores,
                base_frequency,
                max_frequency,
                cache: HashMap::new(),
                flags: Vec::new(),
            },
            memory: MemorySpec {
                total_bytes,
                available_bytes,
                memory_type: "Unknown".to_string(),
                speed_mhz: 0,
            },
            gpu: None,
            storage: Vec::new(),
        })
    }

    fn evaluate_checklist(
        &self,
        environment: &EnvironmentSnapshot,
        experiment_id: &str,
    ) -> ReproducibilityChecklist {
        ReproducibilityChecklist {
            random_seed_documented: !environment.random_seeds.is_empty(),
            dependencies_pinned: !environment.dependencies.is_empty(),
            environment_captured: true, // We have the snapshot
            data_versioned: !environment.data_checksums.is_empty(),
            code_versioned: is_code_versioned(),
            // "Documented" means detection actually found real hardware
            // facts, not merely that a (possibly all-unknown) HardwareConfig
            // struct exists.
            hardware_documented: environment.hardware_config.cpu.model != "Unknown CPU"
                || environment.hardware_config.memory.total_bytes > 0,
            configuration_hashed: !environment.config_hashes.is_empty(),
            // True only if this specific experiment has actually been
            // through `verify_reproducibility` with a non-failed outcome,
            // not merely because *some* verification exists somewhere.
            results_verified: self.verifications.iter().any(|v| {
                (v.original_experiment_id == experiment_id
                    || v.reproduction_experiment_id == experiment_id)
                    && v.status != VerificationStatus::VerificationFailed
            }),
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

/// Locate `Cargo.lock` by walking up from this crate's own manifest
/// directory (its build-time `CARGO_MANIFEST_DIR`) toward the filesystem
/// root -- the same direction Cargo itself searches for a workspace root
/// from a member crate.
fn find_cargo_lock() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dir: &std::path::Path = &manifest_dir;
    loop {
        let candidate = dir.join("Cargo.lock");
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
}

/// Extract every `[[package]]` entry's `name`/`version`/`source`/`checksum`
/// from raw `Cargo.lock` content. `Cargo.lock` is machine-generated TOML
/// with a very regular, single-line-per-field structure for these keys, so
/// a full TOML parser is not needed to read them correctly.
fn parse_cargo_lock_dependencies(content: &str) -> Vec<Dependency> {
    let mut dependencies = Vec::new();
    let mut current: Option<(String, String, Option<String>, Option<String>)> = None;

    for raw_line in content.lines() {
        let line = raw_line.trim();

        if line == "[[package]]" {
            if let Some((name, version, source, checksum)) = current.take() {
                if !name.is_empty() {
                    dependencies.push(Dependency {
                        name,
                        version,
                        source: source.unwrap_or_else(|| "local".to_string()),
                        checksum,
                        install_path: None,
                    });
                }
            }
            current = Some((String::new(), String::new(), None, None));
            continue;
        }

        let Some(entry) = current.as_mut() else {
            continue;
        };

        if let Some(value) = parse_toml_string_field(line, "name") {
            entry.0 = value;
        } else if let Some(value) = parse_toml_string_field(line, "version") {
            entry.1 = value;
        } else if let Some(value) = parse_toml_string_field(line, "source") {
            entry.2 = Some(value);
        } else if let Some(value) = parse_toml_string_field(line, "checksum") {
            entry.3 = Some(value);
        }
    }

    if let Some((name, version, source, checksum)) = current {
        if !name.is_empty() {
            dependencies.push(Dependency {
                name,
                version,
                source: source.unwrap_or_else(|| "local".to_string()),
                checksum,
                install_path: None,
            });
        }
    }

    dependencies
}

/// Parse a single `key = "value"` TOML line for `key`, returning the value
/// (unquoted) if this line defines it. Only handles the plain-string form
/// `Cargo.lock` actually uses for `name`/`version`/`source`/`checksum`.
fn parse_toml_string_field(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=')?;
    let rest = rest.trim();
    let rest = rest.strip_prefix('"')?;
    let value = rest.strip_suffix('"')?;
    Some(value.to_string())
}

/// Best-effort CPU model + (base, max) frequency in MHz, detected by
/// spawning the operating system's own introspection tools or reading its
/// own procfs -- no FFI, no linked C library. Returns `("Unknown CPU", 0,
/// 0)` when detection is unavailable rather than presenting a fabricated
/// model/speed as if it were measured.
fn detect_cpu_info() -> (String, u32, u32) {
    #[cfg(target_os = "macos")]
    {
        if let Some(brand) = run_system_command("sysctl", &["-n", "machdep.cpu.brand_string"]) {
            let brand = brand.trim();
            if !brand.is_empty() {
                return (brand.to_string(), 0, 0);
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            let model = cpuinfo
                .lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string());
            if let Some(model) = model {
                if !model.is_empty() {
                    return (model, 0, 0);
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(output) = run_system_command("wmic", &["cpu", "get", "name"]) {
            if let Some(model) = output.lines().nth(1).map(str::trim) {
                if !model.is_empty() {
                    return (model.to_string(), 0, 0);
                }
            }
        }
    }
    ("Unknown CPU".to_string(), 0, 0)
}

/// Best-effort (total, available) memory in bytes. Returns `(0, 0)` when
/// detection is unavailable rather than presenting a fabricated capacity as
/// if it were measured.
fn detect_memory_info() -> (u64, u64) {
    #[cfg(target_os = "macos")]
    {
        if let Some(total) = run_system_command("sysctl", &["-n", "hw.memsize"])
            .and_then(|s| s.trim().parse::<u64>().ok())
        {
            // macOS has no single simple sysctl for "currently available";
            // report total for both rather than guessing at a fake split.
            return (total, total);
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            let total = parse_meminfo_kb(&meminfo, "MemTotal:");
            let available = parse_meminfo_kb(&meminfo, "MemAvailable:");
            if total > 0 {
                return (total * 1024, available * 1024);
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(output) = run_system_command(
            "wmic",
            &[
                "OS",
                "get",
                "TotalVisibleMemorySize,FreePhysicalMemory",
                "/value",
            ],
        ) {
            let total = parse_wmic_kb_field(&output, "TotalVisibleMemorySize");
            let available = parse_wmic_kb_field(&output, "FreePhysicalMemory");
            if total > 0 {
                return (total * 1024, available * 1024);
            }
        }
    }
    (0, 0)
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn run_system_command(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(meminfo: &str, key: &str) -> u64 {
    meminfo
        .lines()
        .find(|line| line.starts_with(key))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(target_os = "windows")]
fn parse_wmic_kb_field(output: &str, key: &str) -> u64 {
    output
        .lines()
        .find_map(|line| line.trim().strip_prefix(&format!("{key}=")))
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Compare two environment snapshots and return a similarity in `[0, 1]`:
/// matching OS + architecture contributes half the score, and the Jaccard
/// similarity of the two snapshots' `name@version` dependency sets
/// contributes the other half.
fn environment_similarity(a: &EnvironmentSnapshot, b: &EnvironmentSnapshot) -> f64 {
    let os_match = if a.system_info.os == b.system_info.os
        && a.system_info.architecture == b.system_info.architecture
    {
        1.0
    } else {
        0.0
    };

    let deps_a: std::collections::HashSet<String> = a
        .dependencies
        .iter()
        .map(|d| format!("{}@{}", d.name, d.version))
        .collect();
    let deps_b: std::collections::HashSet<String> = b
        .dependencies
        .iter()
        .map(|d| format!("{}@{}", d.name, d.version))
        .collect();

    let dependency_similarity = if deps_a.is_empty() && deps_b.is_empty() {
        1.0
    } else {
        let intersection = deps_a.intersection(&deps_b).count() as f64;
        let union = deps_a.union(&deps_b).count().max(1) as f64;
        intersection / union
    };

    0.5 * os_match + 0.5 * dependency_similarity
}

/// Whether the current working directory is inside a Git work tree, used
/// as a real (rather than hardcoded) signal for the "code versioned"
/// reproducibility checklist item.
fn is_code_versioned() -> bool {
    std::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
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

        let snapshot_id = manager.capture_environment(&[123]).expect("unwrap failed");

        assert!(manager.environments.contains_key(&snapshot_id));
        let snapshot = &manager.environments[&snapshot_id];
        assert_eq!(snapshot.system_info.os, std::env::consts::OS);
        assert_eq!(snapshot.random_seeds, vec![123]);
    }

    #[test]
    fn test_reproducibility_report() {
        let config = ReproducibilityConfig::default();
        let mut manager = ReproducibilityManager::new(config);

        let env_id = manager.capture_environment(&[42]).expect("unwrap failed");
        let report_id = manager
            .generate_report("test_experiment", &env_id)
            .expect("unwrap failed");

        assert!(!manager.reports.is_empty());
        let report = &manager.reports[0];
        assert_eq!(report.id, report_id);
        assert_eq!(report.experiment_id, "test_experiment");
    }

    // Regression test for F76: `capture_environment_variables` used to
    // return `std::env::vars()` unfiltered, which would capture and
    // persist (this snapshot is `Serialize`) any secret the researcher
    // happened to have set in their shell.
    #[test]
    fn test_environment_variables_are_allowlisted_and_redacted() {
        // SAFETY: test-only env mutation; no other test in this process
        // reads these specific names.
        unsafe {
            std::env::set_var("OPTIRS_TEST_SECRET_API_KEY", "super-secret-value");
            std::env::set_var("LANG", "en_US.UTF-8");
        }

        let config = ReproducibilityConfig::default();
        let manager = ReproducibilityManager::new(config);
        let captured = manager.capture_environment_variables();

        assert!(
            !captured.contains_key("OPTIRS_TEST_SECRET_API_KEY"),
            "a variable outside the allowlist must not be captured at all"
        );

        unsafe {
            std::env::set_var("CARGO_TEST_SECRET_KEY", "another-secret");
        }
        let captured = manager.capture_environment_variables();
        if let Some(value) = captured.get("CARGO_TEST_SECRET_KEY") {
            assert_eq!(
                value, "<redacted>",
                "an allowlisted-by-prefix variable whose name looks secret-shaped must be redacted"
            );
        }

        if let Some(lang) = captured.get("LANG") {
            assert_eq!(
                lang, "en_US.UTF-8",
                "ordinary allowlisted values pass through"
            );
        }

        unsafe {
            std::env::remove_var("OPTIRS_TEST_SECRET_API_KEY");
            std::env::remove_var("CARGO_TEST_SECRET_KEY");
        }
    }

    // Regression test for F23: dependency capture used to always return a
    // single hardcoded fake `Dependency` regardless of the real
    // `Cargo.lock`; hardware capture used to always return a fixed "8GB /
    // 6GB available" `MemorySpec` regardless of the real machine.
    #[test]
    fn test_capture_dependencies_reads_real_cargo_lock() {
        let config = ReproducibilityConfig::default();
        let manager = ReproducibilityManager::new(config);

        let dependencies = manager
            .capture_dependencies()
            .expect("dependency capture should not error");

        // This workspace has a real Cargo.lock with many real packages;
        // the old code always returned exactly one ("scirs2-optim").
        assert!(
            dependencies.len() > 1,
            "expected real Cargo.lock contents, got {} entries",
            dependencies.len()
        );
        assert!(
            dependencies.iter().any(|d| d.name == "serde"),
            "expected to find a real, well-known dependency (serde) in the parsed lockfile"
        );
        assert!(
            !dependencies.iter().any(|d| d.name == "scirs2-optim"),
            "must not still contain the old fabricated placeholder entry"
        );
    }

    #[test]
    fn test_capture_hardware_config_is_not_the_old_fixed_placeholder() {
        let config = ReproducibilityConfig::default();
        let manager = ReproducibilityManager::new(config);

        let hardware = manager
            .capture_hardware_config()
            .expect("hardware capture should not error");

        // The old code always reported exactly 8GiB / 6GiB regardless of
        // the real machine.
        assert_ne!(hardware.memory.total_bytes, 8 * 1024 * 1024 * 1024);
        assert_ne!(hardware.memory.available_bytes, 6 * 1024 * 1024 * 1024);
        // On macOS/Linux (this test's CI targets) real detection should
        // succeed; elsewhere `0` honestly means "not detected".
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            assert!(hardware.memory.total_bytes > 0);
            assert_ne!(hardware.cpu.model, "Unknown CPU");
        }
    }

    // Regression test for F21: the reproducibility score used to always be
    // exactly 0.5 because half the checklist items were tautologically
    // fixed (a hardcoded non-empty seed list, a hardcoded non-empty fake
    // dependency list) and the other half were hardcoded constants
    // (`code_versioned: false`, `hardware_documented: true`).
    #[test]
    fn test_reproducibility_score_reflects_real_environment_not_a_fixed_constant() {
        let config = ReproducibilityConfig::default();
        let mut manager = ReproducibilityManager::new(config);

        // A snapshot with no documented seed and no checksummed data
        // should score strictly lower than one with both, proving the
        // score is not a fixed constant.
        let sparse_env_id = manager.capture_environment(&[]).expect("capture");
        let rich_env_id = manager.capture_environment(&[7]).expect("capture");
        if let Some(env) = manager.environments.get_mut(&rich_env_id) {
            env.data_checksums
                .insert("dataset.csv".to_string(), "deadbeef".to_string());
            env.config_hashes
                .insert("config.json".to_string(), "cafebabe".to_string());
        }

        let sparse_report_id = manager
            .generate_report("exp_sparse", &sparse_env_id)
            .expect("report");
        let rich_report_id = manager
            .generate_report("exp_rich", &rich_env_id)
            .expect("report");

        let sparse_score = manager
            .reports
            .iter()
            .find(|r| r.id == sparse_report_id)
            .unwrap()
            .reproducibility_score;
        let rich_score = manager
            .reports
            .iter()
            .find(|r| r.id == rich_report_id)
            .unwrap()
            .reproducibility_score;

        assert!(
            rich_score > sparse_score,
            "richer environment ({rich_score}) should score higher than sparse ({sparse_score})"
        );
        // Since we're running these tests inside a real git checkout,
        // `code_versioned` must be real (true) rather than the old
        // hardcoded `false`.
        assert!(is_code_versioned());
    }

    // Regression test for F22: `verify_reproducibility` used to always
    // record `CloseMatch` with fixed similarity scores (0.95/0.98/0.92/...)
    // no matter what was being "verified" -- it never looked at the
    // experiments' actual results at all.
    #[test]
    fn test_verify_reproducibility_reflects_real_metric_differences() {
        let config = ReproducibilityConfig::default();
        let mut manager = ReproducibilityManager::new(config);

        let mut identical_metrics = HashMap::new();
        identical_metrics.insert("accuracy".to_string(), 0.95);
        identical_metrics.insert("execution_time_seconds".to_string(), 12.0);

        let exact_id = manager
            .verify_reproducibility(
                "orig",
                "repro_exact",
                &identical_metrics,
                &identical_metrics.clone(),
                None,
                None,
            )
            .expect("verification should succeed");
        let exact = manager
            .verifications
            .iter()
            .find(|v| v.id == exact_id)
            .unwrap();
        assert_eq!(exact.status, VerificationStatus::ExactMatch);
        assert_eq!(exact.similarity_metrics.result_similarity, 1.0);
        assert!(exact.differences.is_empty());
        let exact_overall_similarity = exact.similarity_metrics.overall_similarity;

        let mut divergent_metrics = HashMap::new();
        divergent_metrics.insert("accuracy".to_string(), 0.10);
        divergent_metrics.insert("execution_time_seconds".to_string(), 999.0);

        let divergent_id = manager
            .verify_reproducibility(
                "orig",
                "repro_divergent",
                &identical_metrics,
                &divergent_metrics,
                None,
                None,
            )
            .expect("verification should succeed");
        let divergent = manager
            .verifications
            .iter()
            .find(|v| v.id == divergent_id)
            .unwrap();

        assert_ne!(
            divergent.status,
            VerificationStatus::ExactMatch,
            "a run with wildly different metrics must not be reported as an exact match"
        );
        assert!(!divergent.differences.is_empty());
        assert!(
            divergent.similarity_metrics.overall_similarity < exact_overall_similarity,
            "the divergent run must score lower than the identical run, not a fixed constant"
        );
        // Configuration was never supplied, so it must be honestly `None`,
        // not a fabricated 1.0.
        assert!(divergent
            .similarity_metrics
            .configuration_similarity
            .is_none());
    }

    #[test]
    fn test_verify_reproducibility_computes_real_environment_similarity() {
        let config = ReproducibilityConfig::default();
        let mut manager = ReproducibilityManager::new(config);

        let env_a = manager.capture_environment(&[1]).expect("capture");
        let env_b = manager.capture_environment(&[2]).expect("capture");

        let metrics = HashMap::new();
        let verification_id = manager
            .verify_reproducibility(
                "orig",
                "repro",
                &metrics,
                &metrics,
                Some(env_a.as_str()),
                Some(env_b.as_str()),
            )
            .expect("verification should succeed");

        let verification = manager
            .verifications
            .iter()
            .find(|v| v.id == verification_id)
            .unwrap();
        // Both snapshots were captured on the same machine in the same
        // process, so they must be recognized as identical (1.0), not left
        // as `None` when the data to compare them was clearly available.
        assert_eq!(
            verification.similarity_metrics.environment_similarity,
            Some(1.0)
        );
    }
}
