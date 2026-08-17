//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
#[allow(dead_code)]
use crate::plugin::core::*;
use crate::plugin::registry::*;
#[cfg(feature = "crypto")]
use rsa::{pkcs1v15::Pkcs1v15Sign, pkcs8::DecodePublicKey, RsaPublicKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

#[cfg(feature = "crypto")]
use super::functions::{decode_hex, sha256_file};
use super::types_7::{
    CpuRequirements, KeyUsage, PluginMetadata, PluginSourceConfig, SandboxConfig,
    SignatureAlgorithm,
};

/// Cryptographic validator for signature verification
#[derive(Debug)]
pub struct CryptographicValidator {
    /// Trusted CAs
    pub(super) _trustedcas: Vec<TrustedCA>,
    /// Signature verification configuration
    pub(super) config: SignatureVerificationConfig,
}
impl CryptographicValidator {
    pub(super) fn new(_trustedcas: Vec<TrustedCA>, config: SignatureVerificationConfig) -> Self {
        Self {
            _trustedcas,
            config,
        }
    }
    pub(super) fn verify_plugin_signature(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<SignatureVerificationResult> {
        let sig_path = path
            .parent()
            .map(|p| p.join("plugin.sig"))
            .unwrap_or_else(|| PathBuf::from("plugin.sig"));
        if !self.config.enabled {
            return Ok(SignatureVerificationResult {
                valid: true,
                errors: Vec::new(),
                warnings: vec!["Signature verification disabled".to_string()],
                chain_valid: true,
                signer_info: None,
                algorithm: None,
            });
        }
        if !sig_path.exists() {
            return Ok(SignatureVerificationResult {
                valid: false,
                errors: vec![format!(
                    "No signature file found for plugin '{}' at {}",
                    metadata.plugin.name,
                    sig_path.display()
                )],
                warnings: Vec::new(),
                chain_valid: false,
                signer_info: None,
                algorithm: None,
            });
        }
        #[cfg(feature = "crypto")]
        {
            let signature_hex = std::fs::read_to_string(&sig_path).map_err(|e| {
                OptimError::InvalidConfig(format!(
                    "failed to read signature file {}: {e}",
                    sig_path.display()
                ))
            })?;
            if self._trustedcas.is_empty() {
                return Ok(SignatureVerificationResult {
                    valid: false,
                    errors: vec![
                        "No trusted public keys configured; cannot verify signature".to_string()
                    ],
                    warnings: Vec::new(),
                    chain_valid: false,
                    signer_info: None,
                    algorithm: Some(self.config.required_algorithm),
                });
            }
            let digest = match sha256_file(path) {
                Ok(d) => d,
                Err(e) => {
                    return Ok(SignatureVerificationResult {
                        valid: false,
                        errors: vec![format!("failed to hash plugin file: {e}")],
                        warnings: Vec::new(),
                        chain_valid: false,
                        signer_info: None,
                        algorithm: Some(self.config.required_algorithm),
                    });
                }
            };
            let sig_bytes = match decode_hex(&signature_hex) {
                Ok(b) => b,
                Err(e) => {
                    return Ok(SignatureVerificationResult {
                        valid: false,
                        errors: vec![format!("malformed signature file: {e}")],
                        warnings: Vec::new(),
                        chain_valid: false,
                        signer_info: None,
                        algorithm: Some(self.config.required_algorithm),
                    });
                }
            };
            let mut errors = Vec::new();
            let mut valid = false;
            for ca in &self._trustedcas {
                match RsaPublicKey::from_public_key_pem(&ca.public_key) {
                    Ok(public_key) => {
                        if public_key
                            .verify(Pkcs1v15Sign::new_unprefixed(), &digest, &sig_bytes)
                            .is_ok()
                        {
                            valid = true;
                            break;
                        }
                    }
                    Err(e) => errors.push(format!(
                        "trusted CA '{}' has an unparseable public key: {e}",
                        ca.name
                    )),
                }
            }
            if !valid {
                errors.push(
                    "signature did not verify against any configured trusted public key"
                        .to_string(),
                );
            }
            Ok(SignatureVerificationResult {
                valid,
                errors: if valid { Vec::new() } else { errors },
                warnings: Vec::new(),
                chain_valid: valid,
                signer_info: None,
                algorithm: Some(self.config.required_algorithm),
            })
        }
        #[cfg(not(feature = "crypto"))]
        {
            Ok(SignatureVerificationResult {
                valid: false,
                errors: vec![format!(
                    "Cryptographic features not enabled; cannot verify signature for plugin '{}'",
                    metadata.plugin.name
                )],
                warnings: vec![
                    "Build with --features crypto for signature verification".to_string()
                ],
                chain_valid: false,
                signer_info: None,
                algorithm: None,
            })
        }
    }
}
/// Plugin load result
#[derive(Debug)]
pub struct PluginLoadResult {
    /// Whether loading was successful
    pub success: bool,
    /// Loaded plugin information
    pub plugin_info: Option<PluginInfo>,
    /// Load errors
    pub errors: Vec<String>,
    /// Load warnings
    pub warnings: Vec<String>,
    /// Load time
    pub load_time: std::time::Duration,
    /// Security scan results
    pub security_results: SecurityScanResult,
}
impl PluginLoadResult {
    /// Create a failed result with errors
    pub fn failed_with_errors(errors: Vec<String>) -> Self {
        Self {
            success: false,
            plugin_info: None,
            errors,
            warnings: Vec::new(),
            load_time: std::time::Duration::from_secs(0),
            security_results: SecurityScanResult::default(),
        }
    }
}
/// Security threat information
#[derive(Debug, Clone)]
pub struct SecurityThreat {
    /// Threat type
    pub threat_type: ThreatType,
    /// Threat description
    pub description: String,
    /// Severity level
    pub severity: ScanSeverity,
    /// Location in code
    pub location: Option<String>,
}
/// Types of security threats
#[derive(Debug, Clone)]
pub enum ThreatType {
    /// Suspicious function call
    SuspiciousFunction,
    /// Unsafe code block
    UnsafeCode,
    /// Network access
    NetworkAccess,
    /// File system access
    FileSystemAccess,
    /// Process execution
    ProcessExecution,
    /// Invalid cryptographic signature
    InvalidSignature,
    /// Unsigned plugin when signature required
    UnsignedPlugin,
    /// Plugin not in allowlist
    UnauthorizedPlugin,
    /// Expired certificate
    ExpiredCertificate,
    /// Revoked certificate
    RevokedCertificate,
    /// Weak cryptographic algorithms
    WeakCryptography,
    /// File integrity violation
    IntegrityViolation,
    /// Unknown/custom threat
    Unknown(String),
}
/// Plugin configuration for loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin source
    pub source: PluginSourceConfig,
    /// Plugin name
    pub name: String,
    /// Plugin version requirement
    pub version: Option<String>,
    /// Configuration parameters
    pub config: HashMap<String, serde_json::Value>,
    /// Enable automatic updates
    pub auto_update: bool,
}
/// Runtime requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    /// Minimum Rust version
    pub min_rust_version: String,
    /// Required system libraries
    pub system_libraries: Vec<String>,
    /// Environment variables
    pub environment_variables: Vec<String>,
    /// Memory requirements (MB)
    pub memory_mb: Option<usize>,
    /// CPU requirements
    pub cpu_requirements: CpuRequirements,
}
/// Plugin handle for managing dynamic libraries
#[derive(Debug)]
pub struct PluginHandle {
    /// Library path
    pub library_path: PathBuf,
    /// Entry point function name
    pub entry_point: String,
    /// Plugin metadata
    pub metadata: PluginMetadata,
}
/// Configuration for plugin loader
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Enable dynamic loading
    pub enable_dynamic_loading: bool,
    /// Plugin directories to scan
    pub plugin_directories: Vec<PathBuf>,
    /// Maximum plugins to load
    pub max_plugins: usize,
    /// Load timeout
    pub load_timeout: std::time::Duration,
    /// Enable plugin sandboxing
    pub enable_sandboxing: bool,
    /// Allowed plugin sources
    pub allowed_sources: Vec<PluginSource>,
    /// Security policy
    pub security_policy: SecurityPolicy,
}
/// Signature verification result
#[derive(Debug, Clone)]
pub struct SignatureVerificationResult {
    /// Signature is valid
    pub valid: bool,
    /// Verification errors
    pub errors: Vec<String>,
    /// Verification warnings
    pub warnings: Vec<String>,
    /// Certificate chain validation result
    pub chain_valid: bool,
    /// Signer information
    pub signer_info: Option<SignerInfo>,
    /// Signature algorithm used
    pub algorithm: Option<SignatureAlgorithm>,
}
/// Security policy configuration
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Allow unsigned plugins
    pub allow_unsigned: bool,
    /// Require specific permissions
    pub required_permissions: Vec<Permission>,
    /// Forbidden permissions
    pub forbidden_permissions: Vec<Permission>,
    /// Maximum plugin size (bytes)
    pub max_plugin_size: usize,
    /// Enable code scanning
    pub enable_code_scanning: bool,
    /// Sandbox configuration
    pub sandbox_config: SandboxConfig,
    /// Cryptographic signature verification
    pub signature_verification: SignatureVerificationConfig,
    /// Trusted certificate authorities
    pub _trustedcas: Vec<TrustedCA>,
    /// Plugin allowlist (hashes of approved plugins)
    pub plugin_allowlist: Vec<String>,
    /// Enable plugin integrity monitoring
    pub integrity_monitoring: bool,
}
/// Dependency graph for managing plugin dependencies
#[derive(Debug)]
pub struct DependencyGraph {
    /// Node dependencies
    pub(super) dependencies: HashMap<String, Vec<String>>,
    /// Reverse dependencies
    pub(super) dependents: HashMap<String, Vec<String>>,
}
impl DependencyGraph {
    pub(super) fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
        }
    }
    pub(super) fn add_plugin(&mut self, name: &str, dependencies: &[String]) {
        self.dependencies
            .insert(name.to_string(), dependencies.to_vec());
        for dep in dependencies {
            self.dependents
                .entry(dep.clone())
                .or_default()
                .push(name.to_string());
        }
    }
    pub(super) fn remove_plugin(&mut self, name: &str) {
        if let Some(dependencies) = self.dependencies.remove(name) {
            for dep in dependencies {
                if let Some(dependents) = self.dependents.get_mut(&dep) {
                    dependents.retain(|x| x != name);
                }
            }
        }
        self.dependents.remove(name);
    }
    pub(super) fn get_dependents(&self, name: &str) -> Option<&Vec<String>> {
        self.dependents.get(name)
    }
}
/// Security scan result
#[derive(Debug, Clone)]
pub struct SecurityScanResult {
    /// Scan successful
    pub scan_successful: bool,
    /// Security threats found
    pub threats: Vec<SecurityThreat>,
    /// Permission violations
    pub permission_violations: Vec<String>,
    /// Overall security score (0.0 to 1.0)
    pub security_score: f64,
    /// Signature verification result
    pub signature_verification: Option<SignatureVerificationResult>,
    /// Plugin hash (for allowlist checking)
    pub plugin_hash: String,
    /// Integrity check result
    pub integrity_valid: bool,
}
/// Permission validator
#[derive(Debug)]
pub struct PermissionValidator {
    /// Validation rules
    pub(super) rules: Vec<ValidationRule>,
}
impl PermissionValidator {
    /// A validator carrying only the built-in rules.
    pub(super) fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Register an additional rule.
    ///
    /// `permission_pattern` selects which permissions the rule applies to: it
    /// is matched as a case-insensitive substring against the permission's
    /// rendered form, and an empty pattern applies the rule to every
    /// permission. Until 0.3.2 `rules` was an empty vector that nothing could
    /// populate and `validate_permission` never consulted, so a deployment's
    /// own policy had no way in.
    pub fn add_rule(&mut self, rule: ValidationRule) -> Result<()> {
        if rule.name.is_empty() {
            return Err(OptimError::InvalidParameter(
                "a permission validation rule must be named".to_string(),
            ));
        }
        self.rules.push(rule);
        Ok(())
    }

    /// Registered rules, in the order they will be applied.
    pub fn rules(&self) -> &[ValidationRule] {
        &self.rules
    }

    /// Whether `permission` is acceptable.
    ///
    /// The built-in checks run first (no path traversal, no absolute paths, no
    /// empty network target); every registered rule whose `permission_pattern`
    /// matches must then also accept. Rules can only *tighten* the policy --
    /// a rule cannot re-permit something the built-in checks rejected.
    pub(super) fn validate_permission(&self, permission: &Permission) -> bool {
        let builtin_ok = match permission {
            Permission::FileSystem(path) => !path.contains("..") && !path.starts_with('/'),
            Permission::Network(addr) => !addr.is_empty(),
            _ => true,
        };
        if !builtin_ok {
            return false;
        }

        let rendered = format!("{permission:?}").to_ascii_lowercase();
        self.rules.iter().all(|rule| {
            let pattern = rule.permission_pattern.to_ascii_lowercase();
            if !pattern.is_empty() && !rendered.contains(&pattern) {
                return true;
            }
            (rule.validator)(permission)
        })
    }
}
/// Trusted Certificate Authority
#[derive(Debug, Clone)]
pub struct TrustedCA {
    /// CA name
    pub name: String,
    /// CA public key (PEM format)
    pub public_key: String,
    /// CA certificate (PEM format)
    pub certificate: String,
    /// Key usage constraints
    pub key_usage: Vec<KeyUsage>,
    /// Valid from date
    pub valid_from: std::time::SystemTime,
    /// Valid until date
    pub valid_until: std::time::SystemTime,
}
/// Cryptographic signature verification configuration
#[derive(Debug, Clone)]
pub struct SignatureVerificationConfig {
    /// Enable signature verification
    pub enabled: bool,
    /// Required signature algorithm
    pub required_algorithm: SignatureAlgorithm,
    /// Minimum key size (bits)
    pub min_key_size: usize,
    /// Allow self-signed certificates
    pub allow_self_signed: bool,
    /// Certificate chain validation depth
    pub max_chain_depth: usize,
    /// Certificate revocation checking
    pub check_revocation: bool,
    /// Signature validation timeout
    pub validation_timeout: std::time::Duration,
}
/// Code scanning rule
#[derive(Debug)]
pub struct ScanningRule {
    /// Rule name
    pub name: String,
    /// Pattern to match
    pub pattern: String,
    /// Severity level
    pub severity: ScanSeverity,
}
/// Scan severity levels
#[derive(Debug, Clone)]
pub enum ScanSeverity {
    Info,
    Warning,
    Error,
    Critical,
}
/// Security permissions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Permission {
    /// File system access
    FileSystem(String),
    /// Network access
    Network(String),
    /// Process execution
    ProcessExecution,
    /// System information access
    SystemInfo,
    /// Hardware access
    Hardware(String),
    /// Custom permission
    Custom(String),
}
/// Validation rule for permissions
#[derive(Debug)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Permission pattern
    pub permission_pattern: String,
    /// Validation function
    pub validator: fn(&Permission) -> bool,
}
/// Signer information
#[derive(Debug, Clone)]
pub struct SignerInfo {
    /// Signer name
    pub name: String,
    /// Signer email
    pub email: String,
    /// Organization
    pub organization: String,
    /// Country
    pub country: String,
}
/// Build information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Rust version used
    pub rust_version: String,
    /// Target triple
    pub target: String,
    /// Build profile (debug/release)
    pub profile: String,
    /// Build timestamp
    pub timestamp: String,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
}
