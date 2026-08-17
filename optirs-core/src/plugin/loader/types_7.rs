//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
#[allow(dead_code)]
use crate::plugin::core::*;
use crate::plugin::registry::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use super::functions::system_library_exists;
use super::manifest::parse_manifest_toml;
use super::types::{
    BuildInfo, CryptographicValidator, DependencyGraph, LoaderConfig, Permission,
    PermissionValidator, PluginConfig, PluginHandle, PluginLoadResult, RuntimeRequirements,
    ScanSeverity, ScanningRule, SecurityPolicy, SecurityScanResult, SecurityThreat,
    SignatureVerificationResult, SignerInfo, ThreatType,
};

/// Plugin loader for managing plugin loading and unloading
#[derive(Debug)]
pub struct PluginLoader {
    /// Loader configuration
    pub(super) config: LoaderConfig,
    /// Loaded plugins
    pub(super) loaded_plugins: HashMap<String, LoadedPlugin>,
    /// Plugin dependencies
    pub(super) dependency_graph: DependencyGraph,
    /// Security manager
    pub(super) security_manager: SecurityManager,
}
impl PluginLoader {
    /// Create a new plugin loader
    pub fn new(config: LoaderConfig) -> Self {
        let security_manager = SecurityManager::new(config.security_policy.clone());
        Self {
            config,
            security_manager,
            loaded_plugins: HashMap::new(),
            dependency_graph: DependencyGraph::new(),
        }
    }
    /// Load plugin from file
    pub fn load_plugin_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<PluginLoadResult> {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let warnings = Vec::new();
        let path = path.as_ref();
        if !path.exists() {
            return Ok(PluginLoadResult {
                success: false,
                plugin_info: None,
                errors: vec![format!("Plugin file not found: {}", path.display())],
                warnings,
                load_time: start_time.elapsed(),
                security_results: SecurityScanResult::default(),
            });
        }
        let metadata = match self.load_plugin_metadata(path) {
            Ok(metadata) => metadata,
            Err(e) => {
                return Ok(PluginLoadResult {
                    success: false,
                    plugin_info: None,
                    errors: vec![format!("Failed to load metadata: {}", e)],
                    warnings,
                    load_time: start_time.elapsed(),
                    security_results: SecurityScanResult::default(),
                });
            }
        };
        let security_results = self.security_manager.scan_plugin(path, &metadata)?;
        if !security_results.scan_successful || security_results.security_score < 0.5 {
            errors.push("Plugin failed security scan".to_string());
            return Ok(PluginLoadResult {
                success: false,
                plugin_info: None,
                errors,
                warnings,
                load_time: start_time.elapsed(),
                security_results,
            });
        }
        if let Err(e) = self.check_dependencies(&metadata.plugin.dependencies) {
            errors.push(format!("Dependency check failed: {}", e));
        }
        let plugin_info = PluginInfo {
            name: metadata.plugin.name.clone(),
            version: metadata.plugin.version.clone(),
            author: metadata.plugin.author.clone(),
            description: metadata.plugin.description.clone(),
            homepage: metadata.plugin.homepage.clone(),
            license: metadata.plugin.license.clone(),
            supported_types: vec![DataType::F32, DataType::F64],
            category: PluginCategory::FirstOrder,
            tags: Vec::new(),
            min_sdk_version: metadata.runtime.min_rust_version.clone(),
            dependencies: metadata.plugin.dependencies.clone(),
        };
        let loaded_plugin = LoadedPlugin {
            info: plugin_info.clone(),
            source: PluginSource::Local(path.to_path_buf()),
            loaded_at: std::time::SystemTime::now(),
            handle: Some(PluginHandle {
                library_path: path.to_path_buf(),
                entry_point: metadata.plugin.entry_point.clone(),
                metadata: metadata.clone(),
            }),
            initialized: false,
            dependencies: metadata
                .plugin
                .dependencies
                .iter()
                .map(|dep| dep.name.clone())
                .collect(),
        };
        self.loaded_plugins
            .insert(metadata.plugin.name.clone(), loaded_plugin);
        self.dependency_graph.add_plugin(
            &metadata.plugin.name,
            &metadata
                .plugin
                .dependencies
                .iter()
                .map(|dep| dep.name.clone())
                .collect::<Vec<_>>(),
        );
        Ok(PluginLoadResult {
            success: errors.is_empty(),
            plugin_info: Some(plugin_info),
            errors,
            warnings,
            load_time: start_time.elapsed(),
            security_results,
        })
    }
    /// Load plugin from configuration
    pub fn load_plugin_from_config(&mut self, config: PluginConfig) -> Result<PluginLoadResult> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();
        if let Some(loaded) = self.loaded_plugins.get(&config.name) {
            warnings.push(format!("Plugin '{}' is already loaded", config.name));
            return Ok(PluginLoadResult {
                success: true,
                plugin_info: Some(loaded.info.clone()),
                errors: Vec::new(),
                warnings,
                load_time: start_time.elapsed(),
                security_results: SecurityScanResult::default(),
            });
        }
        let result = match &config.source {
            PluginSourceConfig::File(path) => self.load_plugin_from_file(path),
            PluginSourceConfig::Git { url, branch } => {
                self.load_plugin_from_git(url, branch.as_deref(), &config)
            }
            PluginSourceConfig::Registry { name, version } => {
                self.load_plugin_from_registry(name, version.as_deref(), &config)
            }
            PluginSourceConfig::Http(url) => self.load_plugin_from_http(url, &config),
        }?;
        if result.success {
            if let Some(plugin_info) = &result.plugin_info {
                if let Some(loaded) = self.loaded_plugins.get_mut(&plugin_info.name) {
                    loaded.initialized = true;
                }
            }
        }
        Ok(result)
    }
    /// Unload plugin
    pub fn unload_plugin(&mut self, name: &str) -> Result<()> {
        if let Some(dependents) = self.dependency_graph.get_dependents(name) {
            if !dependents.is_empty() {
                return Err(OptimError::PluginStillInUse(format!(
                    "Plugin '{}' is still used by: {}",
                    name,
                    dependents.join(", ")
                )));
            }
        }
        self.loaded_plugins.remove(name);
        self.dependency_graph.remove_plugin(name);
        Ok(())
    }
    /// List loaded plugins
    pub fn list_loaded_plugins(&self) -> Vec<&PluginInfo> {
        self.loaded_plugins.values().map(|p| &p.info).collect()
    }
    /// Get plugin load status
    pub fn get_load_status(&self, name: &str) -> Option<&LoadedPlugin> {
        self.loaded_plugins.get(name)
    }
    /// Discover plugins in configured directories
    pub fn discover_plugins(&mut self) -> Result<Vec<PluginLoadResult>> {
        let mut results = Vec::new();
        let directories = self.config.plugin_directories.clone();
        for directory in directories {
            if directory.exists() && directory.is_dir() {
                let discovered = self.discover_plugins_in_directory(&directory)?;
                results.extend(discovered);
            }
        }
        Ok(results)
    }
    /// Load plugin from Git repository
    /// Git-sourced plugin loading would require compiling arbitrary
    /// downloaded code into this process (dlopen or equivalent), which this
    /// crate does not implement (see the module-level note on dynamic
    /// loading). Rather than fabricate a clone/build pipeline that can never
    /// produce a callable optimizer, this returns an honest, immediate
    /// error. Register optimizers at compile time instead via
    /// `PluginRegistry::register_plugin`.
    pub(super) fn load_plugin_from_git(
        &mut self,
        _url: &str,
        _branch: Option<&str>,
        _config: &PluginConfig,
    ) -> Result<PluginLoadResult> {
        Err(OptimError::UnsupportedOperation(
            "dynamic loading from a Git repository is not supported; use static registration \
             via PluginRegistry::register_plugin instead"
                .to_string(),
        ))
    }
    /// Load plugin from package registry
    ///
    /// Registry-sourced plugin loading has the same fundamental gap as
    /// `load_plugin_from_git`: this crate has no dynamic-loading backend
    /// (`libloading`/`dlopen`), so even a successfully downloaded package
    /// could never be turned into a callable `OptimizerPlugin` -- there is
    /// nowhere to resolve the entry point into. The previous implementation
    /// fabricated a registry HTTP response (`query_registry_api`), wrote a
    /// literal `b"dummy plugin package content"` in place of a download,
    /// unconditionally verified any signature as valid, and wrote
    /// `b"dummy plugin binary"` as the "extracted" plugin -- five straight
    /// simulated steps that could never fail, always ending in the same
    /// `errors.push("...not yet fully implemented")` this now returns
    /// immediately and honestly instead.
    pub(super) fn load_plugin_from_registry(
        &mut self,
        _name: &str,
        _version: Option<&str>,
        _config: &PluginConfig,
    ) -> Result<PluginLoadResult> {
        Err(OptimError::UnsupportedOperation(
            "dynamic loading from a package registry is not supported; use static \
             registration via PluginRegistry::register_plugin instead"
                .to_string(),
        ))
    }
    /// Load plugin from HTTP URL
    ///
    /// Same gap as `load_plugin_from_registry`: this crate has no HTTP
    /// client dependency and no dynamic-loading backend, so a downloaded
    /// file could never become a callable optimizer even if fetched. The
    /// previous implementation wrote a literal `b"downloaded plugin
    /// content"` file and reported it as a successful download before
    /// eventually failing anyway.
    pub(super) fn load_plugin_from_http(
        &mut self,
        _url: &str,
        _config: &PluginConfig,
    ) -> Result<PluginLoadResult> {
        Err(OptimError::UnsupportedOperation(
            "dynamic loading from an HTTP URL is not supported; use static registration \
             via PluginRegistry::register_plugin instead"
                .to_string(),
        ))
    }
    pub(super) fn load_plugin_metadata(&self, path: &Path) -> Result<PluginMetadata> {
        let manifest_path = path
            .parent()
            .map(|p| p.join("plugin.toml"))
            .unwrap_or_else(|| PathBuf::from("plugin.toml"));
        if manifest_path.exists() {
            let content = std::fs::read_to_string(&manifest_path)?;
            let parsed_metadata = self.parse_plugin_toml(&content, path)?;
            Ok(parsed_metadata)
        } else {
            Ok(PluginMetadata::default_for_path(path))
        }
    }
    pub(super) fn check_dependencies(&self, dependencies: &[PluginDependency]) -> Result<()> {
        for dep in dependencies {
            if !dep.optional && !self.is_dependency_satisfied(dep) {
                return Err(OptimError::MissingDependency(dep.name.clone()));
            }
        }
        Ok(())
    }
    /// Check whether a declared dependency is actually satisfied.
    ///
    /// `SystemLibrary` gets a real (if best-effort) filesystem probe below.
    /// `Crate` and `Runtime` have no verifiable signal available to this
    /// crate without new dependencies (cargo metadata access, a version
    /// registry) that are out of scope here, so they return `false` rather
    /// than the previous unconditional `true`. Reporting "satisfied" for a
    /// dependency this function never actually checked is the failure mode
    /// this exists to prevent: a caller declaring a mandatory `Crate`
    /// dependency would have had `check_dependencies` silently pass
    /// regardless of whether that crate exists. `false` for a mandatory
    /// dependency of these kinds now surfaces as `MissingDependency` (see
    /// `check_dependencies`) instead of loading silently; a plugin author
    /// who cannot get a verifiable check should mark the dependency
    /// `optional: true` instead, which this function still honours (an
    /// unverifiable optional dependency does not block loading).
    pub(super) fn is_dependency_satisfied(&self, dependency: &PluginDependency) -> bool {
        match dependency.dependency_type {
            DependencyType::Plugin => self.loaded_plugins.contains_key(&dependency.name),
            DependencyType::SystemLibrary => system_library_exists(&dependency.name),
            DependencyType::Crate | DependencyType::Runtime => false,
        }
    }
    pub(super) fn discover_plugins_in_directory(
        &mut self,
        directory: &Path,
    ) -> Result<Vec<PluginLoadResult>> {
        let mut results = Vec::new();
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if self.is_plugin_file(&path) {
                let result = self.load_plugin_from_file(&path)?;
                results.push(result);
            }
        }
        Ok(results)
    }
    pub(super) fn is_plugin_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            match extension.to_str() {
                Some("so") | Some("dylib") | Some("dll") => true,
                Some("toml") if path.file_stem().and_then(|s| s.to_str()) == Some("plugin") => true,
                _ => false,
            }
        } else {
            false
        }
    }
    /// Parse a plugin TOML manifest.
    ///
    /// Real TOML parsing via the `toml`/`serde` crates (see
    /// `super::manifest` for the full schema, defaulting rules, and
    /// leniency notes). This replaces the previous hand-rolled,
    /// dependency-free line scanner, which could not represent
    /// `[[plugin.dependencies]]` or `[[plugin.permissions]]` array-of-
    /// tables entries at all -- both always parsed as empty `Vec`s
    /// regardless of what the manifest declared. Both are now fully
    /// supported, including `dependency_type`/`optional`/`version` on each
    /// dependency and every `Permission` variant (unit and payload-
    /// carrying) on each permission.
    pub(super) fn parse_plugin_toml(&self, content: &str, path: &Path) -> Result<PluginMetadata> {
        parse_manifest_toml(content, path)
    }
}
/// Security manager for plugin validation
#[derive(Debug)]
pub struct SecurityManager {
    /// Security policy
    pub(super) policy: SecurityPolicy,
    /// Permission validator
    pub(super) permission_validator: PermissionValidator,
    /// Code scanner
    pub(super) code_scanner: CodeScanner,
    /// Cryptographic validator
    pub(super) crypto_validator: CryptographicValidator,
}
impl SecurityManager {
    pub(super) fn new(policy: SecurityPolicy) -> Self {
        let crypto_validator = CryptographicValidator::new(
            policy._trustedcas.clone(),
            policy.signature_verification.clone(),
        );
        Self {
            policy,
            permission_validator: PermissionValidator::new(),
            code_scanner: CodeScanner::new(),
            crypto_validator,
        }
    }
    pub(super) fn scan_plugin(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<SecurityScanResult> {
        let mut threats = Vec::new();
        let mut permission_violations = Vec::new();
        let plugin_hash = self.calculate_plugin_hash(path)?;
        let mut integrity_valid = true;
        if !self.policy.plugin_allowlist.is_empty() {
            integrity_valid = self.policy.plugin_allowlist.contains(&plugin_hash);
            if !integrity_valid {
                threats.push(SecurityThreat {
                    threat_type: ThreatType::UnauthorizedPlugin,
                    description: "Plugin not in approved allowlist".to_string(),
                    severity: ScanSeverity::Critical,
                    location: Some(path.to_string_lossy().to_string()),
                });
            }
        }
        let signature_verification = if self.policy.signature_verification.enabled {
            Some(
                self.crypto_validator
                    .verify_plugin_signature(path, metadata)?,
            )
        } else {
            None
        };
        if !self.policy.allow_unsigned {
            match &signature_verification {
                Some(sig_result) if !sig_result.valid => {
                    threats.push(SecurityThreat {
                        threat_type: ThreatType::InvalidSignature,
                        description: "Plugin signature verification failed".to_string(),
                        severity: ScanSeverity::Critical,
                        location: Some(path.to_string_lossy().to_string()),
                    });
                }
                None => {
                    threats.push(SecurityThreat {
                        threat_type: ThreatType::UnsignedPlugin,
                        description: "Plugin is unsigned but policy requires signatures"
                            .to_string(),
                        severity: ScanSeverity::Critical,
                        location: Some(path.to_string_lossy().to_string()),
                    });
                }
                _ => {}
            }
        }
        for permission in &metadata.plugin.permissions {
            if !self.permission_validator.validate_permission(permission) {
                permission_violations.push(format!("Invalid permission: {:?}", permission));
            }
            if self.policy.forbidden_permissions.contains(permission) {
                permission_violations.push(format!("Forbidden permission: {:?}", permission));
            }
        }
        if self.policy.enable_code_scanning {
            let scan_threats = self.code_scanner.scan_code(path)?;
            threats.extend(scan_threats);
        }
        let security_score = self.calculate_comprehensive_security_score(
            &threats,
            &permission_violations,
            &signature_verification,
            integrity_valid,
        );
        let scan_successful = permission_violations.is_empty()
            && threats
                .iter()
                .all(|t| !matches!(t.severity, ScanSeverity::Critical))
            && integrity_valid;
        Ok(SecurityScanResult {
            scan_successful,
            threats,
            permission_violations,
            security_score,
            signature_verification,
            plugin_hash,
            integrity_valid,
        })
    }
    pub(super) fn calculate_comprehensive_security_score(
        &self,
        threats: &[SecurityThreat],
        violations: &[String],
        signature_verification: &Option<SignatureVerificationResult>,
        integrity_valid: bool,
    ) -> f64 {
        let mut score = 1.0;
        for threat in threats {
            let penalty = match threat.severity {
                ScanSeverity::Critical => 0.5,
                ScanSeverity::Error => 0.3,
                ScanSeverity::Warning => 0.1,
                ScanSeverity::Info => 0.05,
            };
            score -= penalty;
        }
        score -= violations.len() as f64 * 0.1;
        if let Some(sig_result) = signature_verification {
            if !sig_result.valid {
                score -= 0.4;
            }
            if !sig_result.chain_valid {
                score -= 0.2;
            }
        }
        if !integrity_valid {
            score -= 0.3;
        }
        score.clamp(0.0, 1.0)
    }
    pub(super) fn calculate_plugin_hash(&self, path: &Path) -> Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        #[cfg(feature = "crypto")]
        {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&buffer);
            let digest = hasher.finalize();
            Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
        }
        #[cfg(not(feature = "crypto"))]
        {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            buffer.hash(&mut hasher);
            Ok(format!("{:x}", hasher.finish()))
        }
    }
}
impl SecurityManager {
    /// Scan an individual file for security issues.
    ///
    /// Delegates to the configured [`CodeScanner`]. Until 0.3.2 this returned
    /// `SecurityFileScanResult { safe: true, issues: vec![] }` for every input
    /// without reading the file at all -- a scanner that unconditionally
    /// certifies its input as safe is worse than no scanner, because callers
    /// act on the certificate.
    ///
    /// # Errors
    ///
    /// Propagates [`CodeScanner::scan_code`], which refuses to report a clean
    /// result when it has no rules or signatures to check against.
    pub fn scan_file(&self, path: &Path) -> Result<SecurityFileScanResult> {
        let threats = self.code_scanner.scan_code(path)?;
        Ok(SecurityFileScanResult {
            safe: threats.is_empty(),
            issues: threats
                .iter()
                .map(|threat| match &threat.location {
                    Some(location) => {
                        format!("{:?}: {} ({location})", threat.severity, threat.description)
                    }
                    None => format!("{:?}: {}", threat.severity, threat.description),
                })
                .collect(),
        })
    }
}
/// Plugin signature information
#[derive(Debug, Clone)]
pub struct PluginSignature {
    /// Signature algorithm used
    pub algorithm: SignatureAlgorithm,
    /// Signature bytes
    pub signature: Vec<u8>,
    /// Signing certificate chain
    pub certificate_chain: Vec<String>,
    /// Signature timestamp
    pub timestamp: std::time::SystemTime,
    /// Signer information
    pub signer_info: SignerInfo,
}
/// Sandbox configuration
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Enable process isolation
    pub process_isolation: bool,
    /// Memory limit (bytes)
    pub memory_limit: usize,
    /// CPU time limit (seconds)
    pub cpu_time_limit: f64,
    /// Network access allowed
    pub network_access: bool,
    /// File system access paths
    pub filesystem_access: Vec<PathBuf>,
}
/// CPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuRequirements {
    /// Minimum cores
    pub min_cores: Option<usize>,
    /// Required instruction sets
    pub instruction_sets: Vec<String>,
    /// Architecture requirements
    pub architectures: Vec<String>,
}
/// Plugin metadata from manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin manifest version
    pub manifest_version: String,
    /// Plugin information
    pub plugin: PluginManifest,
    /// Build information
    pub build: BuildInfo,
    /// Runtime requirements
    pub runtime: RuntimeRequirements,
}
impl PluginMetadata {
    pub(super) fn default_for_path(path: &Path) -> Self {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        Self {
            manifest_version: "1.0".to_string(),
            plugin: PluginManifest {
                name: name.clone(),
                version: "0.1.0".to_string(),
                description: "Auto-generated plugin manifest".to_string(),
                author: "Unknown".to_string(),
                license: "MIT".to_string(),
                homepage: None,
                entry_point: "plugin_main".to_string(),
                dependencies: Vec::new(),
                platforms: vec!["*".to_string()],
                permissions: Vec::new(),
            },
            build: BuildInfo {
                rust_version: "1.70.0".to_string(),
                target: std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
                profile: "release".to_string(),
                timestamp: format!("{:?}", std::time::SystemTime::now()),
                compiler_flags: Vec::new(),
            },
            runtime: RuntimeRequirements {
                min_rust_version: "1.70.0".to_string(),
                system_libraries: Vec::new(),
                environment_variables: Vec::new(),
                memory_mb: None,
                cpu_requirements: CpuRequirements {
                    min_cores: None,
                    instruction_sets: Vec::new(),
                    architectures: Vec::new(),
                },
            },
        }
    }
}
/// Signature algorithms
#[derive(Debug, Clone, Copy)]
pub enum SignatureAlgorithm {
    Rsa2048Sha256,
    Rsa3072Sha256,
    Rsa4096Sha256,
    EcdsaP256Sha256,
    EcdsaP384Sha384,
    Ed25519,
}
/// Code scanner for malware detection
#[derive(Debug)]
pub struct CodeScanner {
    /// Scanning rules
    pub(super) rules: Vec<ScanningRule>,
    /// Signature database
    pub(super) signatures: Vec<MalwareSignature>,
}
impl CodeScanner {
    /// An empty scanner. Configure it with [`Self::add_rule`] and
    /// [`Self::add_signature`] before scanning; an unconfigured scanner refuses
    /// to scan rather than reporting a clean result it cannot justify.
    pub(super) fn new() -> Self {
        Self {
            rules: Vec::new(),
            signatures: Vec::new(),
        }
    }

    /// Register a byte-pattern rule. An empty pattern is refused: it matches
    /// every file at offset zero.
    pub fn add_rule(&mut self, rule: ScanningRule) -> Result<()> {
        if rule.pattern.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "scanning rule '{}' has an empty pattern, which matches everything",
                rule.name
            )));
        }
        self.rules.push(rule);
        Ok(())
    }

    /// Register a known-bad file digest, as lowercase hex.
    pub fn add_signature(&mut self, signature: MalwareSignature) -> Result<()> {
        let hash = signature.hash.trim().to_ascii_lowercase();
        if hash.is_empty() || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(OptimError::InvalidParameter(format!(
                "malware signature '{}' must carry a non-empty hex digest, got '{}'",
                signature.name, signature.hash
            )));
        }
        self.signatures.push(MalwareSignature { hash, ..signature });
        Ok(())
    }

    /// Configured pattern rules.
    pub fn rules(&self) -> &[ScanningRule] {
        &self.rules
    }

    /// Configured malware digests.
    pub fn signatures(&self) -> &[MalwareSignature] {
        &self.signatures
    }

    /// Scan `path` against every configured rule and signature.
    ///
    /// * **Rules** are literal byte patterns searched over the file's contents;
    ///   a hit yields a [`ThreatType::SuspiciousFunction`] threat carrying the
    ///   byte offset.
    /// * **Signatures** are whole-file digests; a match yields a
    ///   [`ThreatType::IntegrityViolation`] threat at
    ///   [`ScanSeverity::Critical`].
    ///
    /// Until 0.3.2 this returned `Ok(Vec::new())` unconditionally, so
    /// `rules` and `signatures` were written at construction and never read and
    /// every plugin passed the scan.
    ///
    /// # Errors
    ///
    /// * [`OptimError::InvalidState`] -- the scanner has neither rules nor
    ///   signatures, so "no threats found" would assert something it never
    ///   checked.
    /// * An I/O error if `path` cannot be read.
    pub(super) fn scan_code(&self, path: &Path) -> Result<Vec<SecurityThreat>> {
        if self.rules.is_empty() && self.signatures.is_empty() {
            return Err(OptimError::InvalidState(format!(
                "the code scanner has no rules and no malware signatures configured, so it cannot \
                 certify {} as clean; register them with CodeScanner::add_rule / add_signature",
                path.display()
            )));
        }

        let contents = std::fs::read(path)?;
        let mut threats = Vec::new();

        for rule in &self.rules {
            let pattern = rule.pattern.as_bytes();
            if let Some(offset) = contents
                .windows(pattern.len())
                .position(|window| window == pattern)
            {
                threats.push(SecurityThreat {
                    threat_type: ThreatType::SuspiciousFunction,
                    description: format!(
                        "rule '{}' matched the pattern {:?}",
                        rule.name, rule.pattern
                    ),
                    severity: rule.severity.clone(),
                    location: Some(format!("{}:+{offset}", path.display())),
                });
            }
        }

        if !self.signatures.is_empty() {
            let digest = file_digest_hex(&contents);
            for signature in &self.signatures {
                if signature.hash == digest {
                    threats.push(SecurityThreat {
                        threat_type: ThreatType::IntegrityViolation,
                        description: format!(
                            "file digest matches malware signature '{}': {}",
                            signature.name, signature.description
                        ),
                        severity: ScanSeverity::Critical,
                        location: Some(path.display().to_string()),
                    });
                }
            }
        }

        Ok(threats)
    }
}

/// Lowercase hex digest of `contents`.
///
/// SHA-256 under the `crypto` feature; a non-cryptographic 64-bit hash
/// otherwise, matching what [`SecurityManager::calculate_plugin_hash`] does so
/// digests recorded by one are comparable with the other in the same build.
fn file_digest_hex(contents: &[u8]) -> String {
    #[cfg(feature = "crypto")]
    {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(contents);
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
    #[cfg(not(feature = "crypto"))]
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        contents.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
/// Security scan result for individual files
#[derive(Debug)]
pub struct SecurityFileScanResult {
    pub safe: bool,
    pub issues: Vec<String>,
}
/// Plugin manifest information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin license
    pub license: String,
    /// Plugin homepage
    pub homepage: Option<String>,
    /// Plugin entry point
    pub entry_point: String,
    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// Required permissions
    pub permissions: Vec<Permission>,
}
/// Malware signature
#[derive(Debug)]
pub struct MalwareSignature {
    /// Signature name
    pub name: String,
    /// Signature hash
    pub hash: String,
    /// Description
    pub description: String,
}
/// Plugin source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginSourceConfig {
    /// Local file path
    File(PathBuf),
    /// Git repository
    Git { url: String, branch: Option<String> },
    /// Package registry
    Registry {
        name: String,
        version: Option<String>,
    },
    /// HTTP/HTTPS URL
    Http(String),
}
/// Key usage types
#[derive(Debug, Clone, Copy)]
pub enum KeyUsage {
    DigitalSignature,
    ContentCommitment,
    KeyEncipherment,
    DataEncipherment,
    KeyAgreement,
    KeyCertSign,
    CRLSign,
    CodeSigning,
}
/// Loaded plugin information
#[derive(Debug)]
pub struct LoadedPlugin {
    /// Plugin info
    pub info: PluginInfo,
    /// Load source
    pub source: PluginSource,
    /// Load timestamp
    pub loaded_at: std::time::SystemTime,
    /// Plugin handle (for dynamic libraries)
    pub handle: Option<PluginHandle>,
    /// Initialization status
    pub initialized: bool,
    /// Dependencies
    pub dependencies: Vec<String>,
}
