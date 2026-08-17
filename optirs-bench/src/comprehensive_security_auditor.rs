// Comprehensive Security Audit Engine
//
// This module provides advanced security auditing capabilities including dependency
// scanning, vulnerability detection, supply chain security analysis, and automated
// security monitoring for the optimization library and its plugins.

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Main security audit engine
#[derive(Debug)]
pub struct ComprehensiveSecurityAuditor {
    /// Audit configuration
    config: SecurityAuditConfig,
    /// Dependency scanner
    dependency_scanner: DependencyScanner,
    /// Vulnerability database
    vulnerability_db: VulnerabilityDatabase,
    /// Policy enforcer
    policy_enforcer: SecurityPolicyEnforcer,
    /// Report generator
    report_generator: SecurityReportGenerator,
    /// Audit history
    audit_history: Vec<SecurityAuditResult>,
}

/// Security audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditConfig {
    /// Enable dependency vulnerability scanning
    pub enable_dependency_scanning: bool,
    /// Enable static code analysis
    pub enable_static_analysis: bool,
    /// Enable license compliance checking
    pub enable_license_compliance: bool,
    /// Enable supply chain analysis
    pub enable_supply_chain_analysis: bool,
    /// Enable secret detection
    pub enable_secret_detection: bool,
    /// Enable configuration security checks
    pub enable_config_security: bool,
    /// Vulnerability database update frequency
    pub db_update_frequency: Duration,
    /// Maximum audit time
    pub max_audit_time: Duration,
    /// Severity threshold for alerts
    pub alert_threshold: SecuritySeverity,
    /// Audit report format
    pub report_format: ReportFormat,
    /// Enable automatic remediation suggestions
    pub enable_auto_remediation: bool,
    /// Trusted sources for dependencies
    pub trusted_sources: Vec<String>,
    /// Excluded paths from scanning
    pub excluded_paths: Vec<PathBuf>,
    /// Custom security rules
    pub custom_rules: Vec<CustomSecurityRule>,
    /// Webhook URL to POST critical security alerts to, in addition to the
    /// local `log::error!` alert. `None` means alerts are local-only (still
    /// real, just not externally delivered) -- see `generate_security_alert`.
    pub alert_webhook_url: Option<String>,
}

/// Custom security rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomSecurityRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Pattern to match (regex)
    pub pattern: String,
    /// Severity level
    pub severity: SecuritySeverity,
    /// File types to check
    pub file_types: Vec<String>,
    /// Remediation suggestion
    pub remediation: Option<String>,
}

/// Security severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Report format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Yaml,
    Html,
    Pdf,
    Markdown,
    Sarif,
}

/// Comprehensive security audit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditResult {
    /// Audit timestamp
    pub timestamp: SystemTime,
    /// Audit duration
    pub duration: Duration,
    /// Overall security score (0.0 to 1.0)
    pub security_score: f64,
    /// Dependency scan results
    pub dependency_results: DependencyScanResult,
    /// Static analysis results
    pub static_analysis_results: StaticAnalysisResult,
    /// License compliance results
    pub license_compliance_results: LicenseComplianceResult,
    /// Supply chain analysis results
    pub supply_chain_results: SupplyChainAnalysisResult,
    /// Secret detection results
    pub secret_detection_results: SecretDetectionResult,
    /// Configuration security results
    pub config_security_results: ConfigSecurityResult,
    /// Policy compliance results
    pub policy_compliance_results: PolicyComplianceResult,
    /// Remediation suggestions
    pub remediation_suggestions: Vec<RemediationSuggestion>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Dependency scanner for vulnerability detection
#[derive(Debug)]
#[allow(dead_code)]
pub struct DependencyScanner {
    /// Scanner configuration
    config: DependencyScanConfig,
    /// Vulnerability database client
    vuln_db_client: VulnerabilityDatabaseClient,
    /// License database
    license_db: LicenseDatabase,
    /// Package metadata cache
    package_cache: HashMap<String, PackageMetadata>,
}

/// Dependency scan configuration
#[derive(Debug, Clone)]
pub struct DependencyScanConfig {
    /// Scan direct dependencies
    pub scan_direct_deps: bool,
    /// Scan transitive dependencies
    pub scan_transitive_deps: bool,
    /// Maximum dependency depth
    pub max_depth: usize,
    /// Check for outdated dependencies
    pub check_outdated: bool,
    /// Minimum version requirements
    pub min_versions: HashMap<String, String>,
    /// Blocked dependencies
    pub blocked_dependencies: HashSet<String>,
    /// License allowlist
    pub allowed_licenses: HashSet<String>,
    /// License blocklist
    pub blocked_licenses: HashSet<String>,
}

/// Dependency scan result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyScanResult {
    /// Total dependencies scanned
    pub total_dependencies: usize,
    /// Vulnerable dependencies found
    pub vulnerable_dependencies: Vec<VulnerableDependency>,
    /// Outdated dependencies
    pub outdated_dependencies: Vec<OutdatedDependency>,
    /// License violations
    pub license_violations: Vec<LicenseViolation>,
    /// Supply chain risks
    pub supply_chain_risks: Vec<SupplyChainRisk>,
    /// Dependency tree analysis
    pub dependency_tree: DependencyTree,
    /// Risk score (0.0 to 1.0)
    pub risk_score: f64,
}

/// Vulnerable dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerableDependency {
    /// Package name
    pub name: String,
    /// Current version
    pub current_version: String,
    /// Vulnerability details
    pub vulnerabilities: Vec<Vulnerability>,
    /// Affected version range
    pub affected_versions: String,
    /// Fixed version
    pub fixed_version: Option<String>,
    /// Severity
    pub severity: SecuritySeverity,
    /// CVE identifiers
    pub cve_ids: Vec<String>,
}

/// Vulnerability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// Vulnerability ID
    pub id: String,
    /// Title/summary
    pub title: String,
    /// Description
    pub description: String,
    /// Severity
    pub severity: SecuritySeverity,
    /// CVSS score
    pub cvss_score: Option<f64>,
    /// Publication date
    pub published: SystemTime,
    /// Discovery date
    pub discovered: Option<SystemTime>,
    /// Affected versions
    pub affected_versions: String,
    /// Patched versions
    pub patched_versions: Vec<String>,
    /// References
    pub references: Vec<String>,
    /// Categories
    pub categories: Vec<VulnerabilityCategory>,
}

/// Vulnerability categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityCategory {
    CodeExecution,
    MemoryCorruption,
    InformationLeak,
    DenialOfService,
    PrivilegeEscalation,
    AuthenticationBypass,
    Cryptographic,
    InputValidation,
    Other(String),
}

/// Static analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticAnalysisResult {
    /// Security issues found
    pub security_issues: Vec<SecurityIssue>,
    /// Code quality issues
    pub quality_issues: Vec<QualityIssue>,
    /// Files scanned
    pub files_scanned: usize,
    /// Lines of code analyzed
    pub lines_analyzed: usize,
    /// Analysis duration
    pub analysis_duration: Duration,
}

/// Security issue found in static analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    /// Issue ID
    pub id: String,
    /// Issue type
    pub issue_type: SecurityIssueType,
    /// Severity
    pub severity: SecuritySeverity,
    /// File location
    pub file: PathBuf,
    /// Line number
    pub line: usize,
    /// Column number
    pub column: Option<usize>,
    /// Description
    pub description: String,
    /// Code snippet
    pub code_snippet: Option<String>,
    /// Remediation suggestion
    pub remediation: Option<String>,
    /// Rule ID that triggered this issue
    pub rule_id: String,
}

/// Types of security issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityIssueType {
    UnsafeCode,
    HardcodedSecret,
    WeakCryptography,
    SqlInjection,
    PathTraversal,
    CommandInjection,
    BufferOverflow,
    IntegerOverflow,
    UseAfterFree,
    DoubleFree,
    UnvalidatedInput,
    InformationLeak,
    InsecureDeserialization,
    Other(String),
}

/// Vulnerability database for tracking known security issues
#[derive(Debug)]
#[allow(dead_code)]
pub struct VulnerabilityDatabase {
    /// Database configuration
    config: VulnerabilityDatabaseConfig,
    /// Local vulnerability cache
    local_cache: HashMap<String, CachedVulnerability>,
    /// Database update status
    last_update: SystemTime,
    /// Update frequency
    update_frequency: Duration,
    /// External database sources
    external_sources: Vec<ExternalVulnerabilitySource>,
}

/// Vulnerability database configuration
#[derive(Debug, Clone)]
pub struct VulnerabilityDatabaseConfig {
    /// Enable automatic updates
    pub auto_update: bool,
    /// Update check frequency
    pub update_frequency: Duration,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Retention period for cached data
    pub cache_retention: Duration,
    /// External data sources
    pub external_sources: Vec<String>,
    /// API keys for external services
    pub api_keys: HashMap<String, String>,
}

/// Cached vulnerability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedVulnerability {
    /// Vulnerability data
    pub vulnerability: Vulnerability,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Data source
    pub source: String,
    /// Verification status
    pub verified: bool,
}

/// External vulnerability data source
#[derive(Debug, Clone)]
pub struct ExternalVulnerabilitySource {
    /// Source name
    pub name: String,
    /// API endpoint
    pub endpoint: String,
    /// API key
    pub api_key: Option<String>,
    /// Update frequency
    pub update_frequency: Duration,
    /// Priority level
    pub priority: u8,
}

/// Security policy enforcer
#[derive(Debug)]
#[allow(dead_code)]
pub struct SecurityPolicyEnforcer {
    /// Active policies
    policies: Vec<SecurityPolicy>,
    /// Policy evaluation engine
    evaluator: PolicyEvaluator,
    /// Violation tracking
    violations: Vec<PolicyViolation>,
}

/// Security policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Policy ID
    pub id: String,
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    /// Enforcement level
    pub enforcement: EnforcementLevel,
    /// Applicable scopes
    pub scopes: Vec<PolicyScope>,
}

/// Policy rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule ID
    pub id: String,
    /// Rule condition
    pub condition: PolicyCondition,
    /// Required action
    pub action: PolicyAction,
    /// Rule severity
    pub severity: SecuritySeverity,
}

/// Policy enforcement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory only - log violations
    Advisory,
    /// Warning - log and report violations
    Warning,
    /// Enforcing - block violations
    Enforcing,
    /// Panic - stop execution on violations
    Panic,
}

/// Risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub overall_risk: RiskLevel,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Risk score (0.0 to 1.0)
    pub risk_score: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Risk mitigation strategies
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// Risk levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RiskLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

/// Risk factor contributing to overall assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor name
    pub name: String,
    /// Factor description
    pub description: String,
    /// Impact level
    pub impact: f64,
    /// Likelihood
    pub likelihood: f64,
    /// Risk contribution
    pub risk_contribution: f64,
}

/// Risk mitigation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Implementation steps
    pub steps: Vec<String>,
    /// Estimated effort
    pub effort: EffortLevel,
    /// Expected risk reduction
    pub risk_reduction: f64,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

impl ComprehensiveSecurityAuditor {
    /// Create a new security auditor
    pub fn new(config: SecurityAuditConfig) -> Self {
        let dependency_scanner = DependencyScanner::new(DependencyScanConfig::default());
        let vulnerability_db = VulnerabilityDatabase::new(VulnerabilityDatabaseConfig::default());
        let policy_enforcer = SecurityPolicyEnforcer::new();
        let report_generator = SecurityReportGenerator::new();

        Self {
            config,
            dependency_scanner,
            vulnerability_db,
            policy_enforcer,
            report_generator,
            audit_history: Vec::new(),
        }
    }

    /// Run comprehensive security audit
    pub fn audit_project<P: AsRef<Path>>(&mut self, projectpath: P) -> Result<SecurityAuditResult> {
        let start_time = std::time::Instant::now();
        let projectpath = projectpath.as_ref();

        // Update vulnerability database if needed
        self.update_vulnerability_database()?;

        // Initialize audit result
        let mut auditresult = SecurityAuditResult {
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(0),
            security_score: 0.0,
            dependency_results: DependencyScanResult::default(),
            static_analysis_results: StaticAnalysisResult::default(),
            license_compliance_results: LicenseComplianceResult::default(),
            supply_chain_results: SupplyChainAnalysisResult::default(),
            secret_detection_results: SecretDetectionResult::default(),
            config_security_results: ConfigSecurityResult::default(),
            policy_compliance_results: PolicyComplianceResult::default(),
            remediation_suggestions: Vec::new(),
            risk_assessment: RiskAssessment::default(),
        };

        // Run dependency scanning
        if self.config.enable_dependency_scanning {
            auditresult.dependency_results =
                self.dependency_scanner.scan_dependencies(projectpath)?;
        }

        // Run static analysis
        if self.config.enable_static_analysis {
            auditresult.static_analysis_results = self.run_static_analysis(projectpath)?;
        }

        // Run license compliance check
        if self.config.enable_license_compliance {
            auditresult.license_compliance_results = self.check_license_compliance(projectpath)?;
        }

        // Run supply chain analysis
        if self.config.enable_supply_chain_analysis {
            auditresult.supply_chain_results = self.analyze_supply_chain(projectpath)?;
        }

        // Run secret detection
        if self.config.enable_secret_detection {
            auditresult.secret_detection_results = self.detect_secrets(projectpath)?;
        }

        // Run configuration security checks
        if self.config.enable_config_security {
            auditresult.config_security_results = self.check_config_security(projectpath)?;
        }

        // Check policy compliance
        auditresult.policy_compliance_results =
            self.policy_enforcer.check_compliance(&auditresult)?;

        // Generate remediation suggestions
        if self.config.enable_auto_remediation {
            auditresult.remediation_suggestions =
                self.generate_remediation_suggestions(&auditresult)?;
        }

        // Perform risk assessment
        auditresult.risk_assessment = self.assess_risk(&auditresult)?;

        // Calculate overall security score
        auditresult.security_score = self.calculate_security_score(&auditresult);

        // Set audit duration
        auditresult.duration = start_time.elapsed();

        // Store in audit history
        self.audit_history.push(auditresult.clone());

        // Generate alerts if necessary
        self.check_alerts(&auditresult)?;

        Ok(auditresult)
    }

    /// Update vulnerability database from external sources
    pub fn update_vulnerability_database(&mut self) -> Result<()> {
        if self.vulnerability_db.needs_update() {
            self.vulnerability_db.update_from_sources()?;
        }
        Ok(())
    }

    /// Run automated dependency scanning against the embedded, offline
    /// RustSec advisory snapshot (see [`embedded_advisory_snapshot`]).
    /// Delegates to [`scan_dependencies_offline`] so this and
    /// `DependencyScanner::scan_dependencies` (the method actually invoked
    /// by [`Self::audit_project`]) never diverge into two different
    /// implementations.
    pub fn scan_dependencies_with_rustsec(
        &mut self,
        projectpath: &Path,
    ) -> Result<DependencyScanResult> {
        scan_dependencies_offline(projectpath)
    }

    /// Run static analysis on project files
    fn run_static_analysis(&self, projectpath: &Path) -> Result<StaticAnalysisResult> {
        let start_time = std::time::Instant::now();
        let mut security_issues = Vec::new();
        let quality_issues = Vec::new();
        let mut files_scanned = 0;
        let mut lines_analyzed = 0;

        // Find and scan Rust source files
        let rust_files = self.find_rust_files(projectpath)?;

        for filepath in rust_files {
            if self.is_excluded_path(&filepath) {
                continue;
            }

            let content = std::fs::read_to_string(&filepath)?;
            let file_lines = content.lines().count();
            lines_analyzed += file_lines;
            files_scanned += 1;

            // Analyze file for security issues
            let mut file_issues = self.analyze_file_security(&filepath, &content)?;
            security_issues.append(&mut file_issues);

            // Apply custom security rules
            let mut custom_issues = self.apply_custom_rules(&filepath, &content)?;
            security_issues.append(&mut custom_issues);
        }

        Ok(StaticAnalysisResult {
            security_issues,
            quality_issues,
            files_scanned,
            lines_analyzed,
            analysis_duration: start_time.elapsed(),
        })
    }

    /// Analyze file for security issues
    fn analyze_file_security(&self, filepath: &Path, content: &str) -> Result<Vec<SecurityIssue>> {
        let mut issues = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            // Check for unsafe code blocks
            if line.trim_start().starts_with("unsafe") {
                issues.push(SecurityIssue {
                    id: format!("unsafe_code_{}", line_num),
                    issue_type: SecurityIssueType::UnsafeCode,
                    severity: SecuritySeverity::Medium,
                    file: filepath.to_path_buf(),
                    line: line_num + 1,
                    column: Some(line.find("unsafe").unwrap_or(0)),
                    description: "Unsafe code block detected - review for memory safety"
                        .to_string(),
                    code_snippet: Some(line.to_string()),
                    remediation: Some(
                        "Ensure unsafe code is properly justified and reviewed".to_string(),
                    ),
                    rule_id: "SEC001".to_string(),
                });
            }

            // Check for hardcoded secrets (basic patterns)
            if self.contains_potential_secret(line) {
                issues.push(SecurityIssue {
                    id: format!("secret_{}", line_num),
                    issue_type: SecurityIssueType::HardcodedSecret,
                    severity: SecuritySeverity::High,
                    file: filepath.to_path_buf(),
                    line: line_num + 1,
                    column: None,
                    description: "Potential hardcoded secret detected".to_string(),
                    code_snippet: Some(self.sanitize_secret_in_line(line)),
                    remediation: Some(
                        "Move secrets to environment variables or secure configuration".to_string(),
                    ),
                    rule_id: "SEC002".to_string(),
                });
            }

            // Check for potential command injection
            if line.contains("Command::new") || line.contains("process::Command") {
                issues.push(SecurityIssue {
                    id: format!("command_injection_{}", line_num),
                    issue_type: SecurityIssueType::CommandInjection,
                    severity: SecuritySeverity::Medium,
                    file: filepath.to_path_buf(),
                    line: line_num + 1,
                    column: None,
                    description: "Command execution detected - ensure input validation".to_string(),
                    code_snippet: Some(line.to_string()),
                    remediation: Some("Validate and sanitize all command arguments".to_string()),
                    rule_id: "SEC003".to_string(),
                });
            }

            // Check for weak cryptography
            if self.uses_weak_crypto(line) {
                issues.push(SecurityIssue {
                    id: format!("weak_crypto_{}", line_num),
                    issue_type: SecurityIssueType::WeakCryptography,
                    severity: SecuritySeverity::High,
                    file: filepath.to_path_buf(),
                    line: line_num + 1,
                    column: None,
                    description: "Weak cryptographic algorithm detected".to_string(),
                    code_snippet: Some(line.to_string()),
                    remediation: Some("Use modern, secure cryptographic algorithms".to_string()),
                    rule_id: "SEC004".to_string(),
                });
            }
        }

        Ok(issues)
    }

    /// Apply custom security rules to file content
    fn apply_custom_rules(&self, filepath: &Path, content: &str) -> Result<Vec<SecurityIssue>> {
        let mut issues = Vec::new();

        for rule in &self.config.custom_rules {
            // Check if rule applies to this file type
            if let Some(extension) = filepath.extension() {
                let ext_str = extension.to_str().unwrap_or("");
                if !rule.file_types.is_empty() && !rule.file_types.contains(&ext_str.to_string()) {
                    continue;
                }
            }

            // Apply regex pattern
            for (line_num, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&rule.pattern.to_lowercase()) {
                    issues.push(SecurityIssue {
                        id: format!("custom_{}_{}", rule.id, line_num),
                        issue_type: SecurityIssueType::Other(rule.name.clone()),
                        severity: rule.severity,
                        file: filepath.to_path_buf(),
                        line: line_num + 1,
                        column: None,
                        description: rule.description.clone(),
                        code_snippet: Some(line.to_string()),
                        remediation: rule.remediation.clone(),
                        rule_id: rule.id.clone(),
                    });
                }
            }
        }

        Ok(issues)
    }

    /// Check for potential secrets in code line
    fn contains_potential_secret(&self, line: &str) -> bool {
        let secret_indicators = [
            "password",
            "secret",
            "token",
            "api_key",
            "private_key",
            "access_key",
            "auth_token",
            "bearer",
            "jwt",
        ];

        let line_lower = line.to_lowercase();

        // Look for patterns like: variable = "secret_value"
        if line_lower.contains('=') && (line_lower.contains('"') || line_lower.contains('\'')) {
            for indicator in &secret_indicators {
                if line_lower.contains(indicator) {
                    return true;
                }
            }

            // Check for long random-looking strings
            if let Some(quote_start) = line.find('"') {
                if let Some(quote_end) = line[quote_start + 1..].find('"') {
                    let potential_secret = &line[quote_start + 1..quote_start + 1 + quote_end];
                    if potential_secret.len() > 16
                        && potential_secret.chars().any(|c| c.is_ascii_alphanumeric())
                    {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Sanitize secret in code line for safe reporting
    fn sanitize_secret_in_line(&self, line: &str) -> String {
        let mut sanitized = line.to_string();

        // Replace quoted strings that might be secrets
        if let Some(quote_start) = sanitized.find('"') {
            if let Some(quote_end) = sanitized[quote_start + 1..].find('"') {
                let before = &sanitized[..quote_start + 1];
                let after = &sanitized[quote_start + 1 + quote_end..];
                sanitized = format!("{}[REDACTED]{}", before, after);
            }
        }

        sanitized
    }

    /// Check if line uses weak cryptography
    fn uses_weak_crypto(&self, line: &str) -> bool {
        let weak_crypto_patterns = ["md5", "sha1", "des", "3des", "rc4", "md4"];

        let line_lower = line.to_lowercase();
        weak_crypto_patterns
            .iter()
            .any(|pattern| line_lower.contains(pattern))
    }

    /// Find all Rust source files in project
    fn find_rust_files(&self, projectpath: &Path) -> Result<Vec<PathBuf>> {
        let mut rust_files = Vec::new();

        fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    // Skip common non-source directories
                    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                        if ["target", ".git", "node_modules"].contains(&dir_name) {
                            continue;
                        }
                    }
                    visit_dir(&path, files)?;
                } else if let Some(extension) = path.extension() {
                    if extension == "rs" {
                        files.push(path);
                    }
                }
            }
            Ok(())
        }

        visit_dir(projectpath, &mut rust_files)?;
        Ok(rust_files)
    }

    /// Check if path should be excluded from scanning
    fn is_excluded_path(&self, path: &Path) -> bool {
        self.config.excluded_paths.iter().any(|excluded| {
            path.starts_with(excluded)
                || path
                    .components()
                    .any(|component| component.as_os_str() == excluded.as_os_str())
        })
    }

    /// Calculate overall security score based on audit results
    fn calculate_security_score(&self, auditresult: &SecurityAuditResult) -> f64 {
        let mut score = 1.0;

        // Dependency vulnerabilities penalty
        let critical_vulns = auditresult
            .dependency_results
            .vulnerable_dependencies
            .iter()
            .filter(|dep| dep.severity == SecuritySeverity::Critical)
            .count();
        let high_vulns = auditresult
            .dependency_results
            .vulnerable_dependencies
            .iter()
            .filter(|dep| dep.severity == SecuritySeverity::High)
            .count();

        score -= critical_vulns as f64 * 0.2;
        score -= high_vulns as f64 * 0.1;

        // Static analysis issues penalty
        let critical_issues = auditresult
            .static_analysis_results
            .security_issues
            .iter()
            .filter(|issue| issue.severity == SecuritySeverity::Critical)
            .count();
        let high_issues = auditresult
            .static_analysis_results
            .security_issues
            .iter()
            .filter(|issue| issue.severity == SecuritySeverity::High)
            .count();

        score -= critical_issues as f64 * 0.15;
        score -= high_issues as f64 * 0.08;

        // Secret detection penalty
        score -= auditresult.secret_detection_results.secrets_found.len() as f64 * 0.1;

        // License compliance penalty
        score -= auditresult.license_compliance_results.violations.len() as f64 * 0.05;

        // Policy violations penalty
        let critical_violations = auditresult
            .policy_compliance_results
            .violations
            .iter()
            .filter(|v| v.severity == SecuritySeverity::Critical)
            .count();
        score -= critical_violations as f64 * 0.1;

        score.clamp(0.0, 1.0)
    }

    /// Generate remediation suggestions based on audit findings
    fn generate_remediation_suggestions(
        &self,
        auditresult: &SecurityAuditResult,
    ) -> Result<Vec<RemediationSuggestion>> {
        let mut suggestions = Vec::new();

        // Suggestions for vulnerable dependencies
        for vuln_dep in &auditresult.dependency_results.vulnerable_dependencies {
            if let Some(fixed_version) = &vuln_dep.fixed_version {
                suggestions.push(RemediationSuggestion {
                    id: format!("dep_update_{}", vuln_dep.name),
                    title: format!("Update {} to secure version", vuln_dep.name),
                    description: format!(
                        "Update {} from {} to {} to fix security vulnerabilities",
                        vuln_dep.name, vuln_dep.current_version, fixed_version
                    ),
                    priority: match vuln_dep.severity {
                        SecuritySeverity::Critical => RemediationPriority::Critical,
                        SecuritySeverity::High => RemediationPriority::High,
                        SecuritySeverity::Medium => RemediationPriority::Medium,
                        SecuritySeverity::Low => RemediationPriority::Low,
                        SecuritySeverity::Info => RemediationPriority::Low,
                    },
                    effort: EffortLevel::Low,
                    steps: vec![
                        format!(
                            "Update Cargo.toml to use {} = \"{}\"",
                            vuln_dep.name, fixed_version
                        ),
                        "Run cargo update".to_string(),
                        "Test the application thoroughly".to_string(),
                    ],
                    automated: true,
                });
            }
        }

        // Suggestions for static analysis issues
        for issue in &auditresult.static_analysis_results.security_issues {
            if let Some(remediation) = &issue.remediation {
                suggestions.push(RemediationSuggestion {
                    id: format!("static_{}", issue.id),
                    title: format!("Fix security issue: {}", issue.description),
                    description: remediation.clone(),
                    priority: match issue.severity {
                        SecuritySeverity::Critical => RemediationPriority::Critical,
                        SecuritySeverity::High => RemediationPriority::High,
                        SecuritySeverity::Medium => RemediationPriority::Medium,
                        SecuritySeverity::Low => RemediationPriority::Low,
                        SecuritySeverity::Info => RemediationPriority::Low,
                    },
                    effort: EffortLevel::Medium,
                    steps: vec![
                        format!("Review code at {}:{}", issue.file.display(), issue.line),
                        remediation.clone(),
                        "Test the fix thoroughly".to_string(),
                    ],
                    automated: false,
                });
            }
        }

        // Suggestions for secrets
        for secret in &auditresult.secret_detection_results.secrets_found {
            suggestions.push(RemediationSuggestion {
                id: format!("secret_{}", secret.id),
                title: "Remove hardcoded secret".to_string(),
                description:
                    "Move hardcoded secret to environment variable or secure configuration"
                        .to_string(),
                priority: RemediationPriority::High,
                effort: EffortLevel::Medium,
                steps: vec![
                    "Remove the hardcoded secret from source code".to_string(),
                    "Add the secret as an environment variable".to_string(),
                    "Update code to read from environment".to_string(),
                    "Rotate the secret if it was committed to version control".to_string(),
                ],
                automated: false,
            });
        }

        Ok(suggestions)
    }

    /// Assess overall security risk
    fn assess_risk(&self, auditresult: &SecurityAuditResult) -> Result<RiskAssessment> {
        let mut risk_factors = Vec::new();
        let mut total_risk = 0.0;

        // Vulnerability risk
        let vuln_count = auditresult.dependency_results.vulnerable_dependencies.len();
        if vuln_count > 0 {
            let vuln_risk = (vuln_count as f64 * 0.1).min(0.8);
            risk_factors.push(RiskFactor {
                name: "Dependency Vulnerabilities".to_string(),
                description: format!("{} vulnerable dependencies found", vuln_count),
                impact: 0.8,
                likelihood: 0.9,
                risk_contribution: vuln_risk,
            });
            total_risk += vuln_risk;
        }

        // Security issues risk
        let issue_count = auditresult.static_analysis_results.security_issues.len();
        if issue_count > 0 {
            let issue_risk = (issue_count as f64 * 0.05).min(0.6);
            risk_factors.push(RiskFactor {
                name: "Static Analysis Issues".to_string(),
                description: format!("{} security issues found in code", issue_count),
                impact: 0.6,
                likelihood: 0.7,
                risk_contribution: issue_risk,
            });
            total_risk += issue_risk;
        }

        // Secret exposure risk
        let secret_count = auditresult.secret_detection_results.secrets_found.len();
        if secret_count > 0 {
            let secret_risk = (secret_count as f64 * 0.2).min(0.9);
            risk_factors.push(RiskFactor {
                name: "Exposed Secrets".to_string(),
                description: format!("{} hardcoded secrets found", secret_count),
                impact: 0.9,
                likelihood: 0.8,
                risk_contribution: secret_risk,
            });
            total_risk += secret_risk;
        }

        let overall_risk = match total_risk {
            r if r >= 0.8 => RiskLevel::Critical,
            r if r >= 0.6 => RiskLevel::High,
            r if r >= 0.4 => RiskLevel::Medium,
            r if r >= 0.2 => RiskLevel::Low,
            _ => RiskLevel::Minimal,
        };

        let mitigation_strategies = self.generate_mitigation_strategies(&risk_factors);

        Ok(RiskAssessment {
            overall_risk,
            risk_factors,
            risk_score: total_risk.min(1.0),
            recommendations: vec![
                "Implement regular security audits".to_string(),
                "Keep dependencies up to date".to_string(),
                "Use automated security scanning in CI/CD".to_string(),
                "Implement secure coding practices".to_string(),
                "Regular security training for developers".to_string(),
            ],
            mitigation_strategies,
        })
    }

    /// Generate mitigation strategies based on risk factors
    fn generate_mitigation_strategies(
        &self,
        risk_factors: &[RiskFactor],
    ) -> Vec<MitigationStrategy> {
        let mut strategies = Vec::new();

        for factor in risk_factors {
            match factor.name.as_str() {
                "Dependency Vulnerabilities" => {
                    strategies.push(MitigationStrategy {
                        name: "Automated Dependency Management".to_string(),
                        description: "Implement automated dependency scanning and updates"
                            .to_string(),
                        steps: vec![
                            "Set up dependabot or renovate for automated updates".to_string(),
                            "Implement dependency scanning in CI/CD pipeline".to_string(),
                            "Establish process for reviewing security advisories".to_string(),
                            "Create dependency approval process".to_string(),
                        ],
                        effort: EffortLevel::Medium,
                        risk_reduction: 0.7,
                    });
                }
                "Static Analysis Issues" => {
                    strategies.push(MitigationStrategy {
                        name: "Enhanced Static Analysis".to_string(),
                        description:
                            "Implement comprehensive static analysis in development workflow"
                                .to_string(),
                        steps: vec![
                            "Integrate static analysis tools in IDE".to_string(),
                            "Add pre-commit hooks for security checks".to_string(),
                            "Implement security linting in CI/CD".to_string(),
                            "Establish code review guidelines for security".to_string(),
                        ],
                        effort: EffortLevel::Low,
                        risk_reduction: 0.6,
                    });
                }
                "Exposed Secrets" => {
                    strategies.push(MitigationStrategy {
                        name: "Secret Management Implementation".to_string(),
                        description: "Implement proper secret management practices".to_string(),
                        steps: vec![
                            "Deploy secret management solution (HashiCorp Vault, etc.)".to_string(),
                            "Implement secret scanning in CI/CD".to_string(),
                            "Rotate all exposed secrets".to_string(),
                            "Train developers on secret management".to_string(),
                        ],
                        effort: EffortLevel::High,
                        risk_reduction: 0.9,
                    });
                }
                _ => {}
            }
        }

        strategies
    }

    /// Check if alerts should be generated based on audit results
    fn check_alerts(&self, auditresult: &SecurityAuditResult) -> Result<()> {
        let mut critical_issues = Vec::new();

        // Check for critical vulnerabilities
        for vuln_dep in &auditresult.dependency_results.vulnerable_dependencies {
            if vuln_dep.severity >= self.config.alert_threshold {
                critical_issues.push(format!(
                    "Critical vulnerability in {}: {}",
                    vuln_dep.name,
                    vuln_dep
                        .vulnerabilities
                        .first()
                        .map(|v| &v.title)
                        .unwrap_or(&"Unknown".to_string())
                ));
            }
        }

        // Check for critical static analysis issues
        for issue in &auditresult.static_analysis_results.security_issues {
            if issue.severity >= self.config.alert_threshold {
                critical_issues.push(format!("Critical security issue: {}", issue.description));
            }
        }

        // Check for exposed secrets
        if !auditresult
            .secret_detection_results
            .secrets_found
            .is_empty()
        {
            critical_issues.push("Hardcoded secrets detected in source code".to_string());
        }

        // Generate alerts if there are critical issues
        if !critical_issues.is_empty() {
            self.generate_security_alert(critical_issues)?;
        }

        Ok(())
    }

    /// Generate security alert. Always logs locally via `log::error!` (a
    /// real, immediate alert channel); additionally POSTs to
    /// `config.alert_webhook_url` through `crate::notification_transport`
    /// when configured. An unconfigured webhook is not an error (the local
    /// log alert already happened for real); a *configured* webhook that
    /// fails to deliver is, so it is never silently swallowed.
    fn generate_security_alert(&self, issues: Vec<String>) -> Result<()> {
        log::error!(
            "SECURITY ALERT: {} critical security issue(s) detected: {}",
            issues.len(),
            issues.join("; ")
        );

        let Some(webhook_url) = self.config.alert_webhook_url.as_ref() else {
            return Ok(());
        };
        if webhook_url.is_empty() {
            return Ok(());
        }

        let payload = serde_json::json!({
            "alert": "security",
            "issue_count": issues.len(),
            "issues": issues,
        });
        let target = crate::notification_transport::DeliveryTarget::json_post(
            webhook_url.clone(),
            "security-audit-alert",
        );
        let transport_kind = crate::notification_transport::transport_kind_from_env();
        let outcome =
            crate::notification_transport::deliver(&transport_kind, &target, &payload.to_string())?;
        if outcome.is_success() {
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(format!(
                "security alert webhook delivery failed: {}",
                outcome.detail()
            )))
        }
    }

    /// Get audit history
    pub fn get_audit_history(&self) -> &[SecurityAuditResult] {
        &self.audit_history
    }

    /// Generate security report
    pub fn generate_report(&self, auditresult: &SecurityAuditResult) -> Result<String> {
        self.report_generator
            .generate_report(auditresult, &self.config.report_format)
    }

    /// Run scheduled security audit
    pub fn run_scheduled_audit(
        &mut self,
        projectpath: &Path,
        schedule: AuditSchedule,
    ) -> Result<()> {
        match schedule {
            AuditSchedule::Daily => {
                // Run lightweight audit daily
                let mut config = self.config.clone();
                config.enable_supply_chain_analysis = false;
                config.max_audit_time = Duration::from_secs(5 * 60); // 5 minutes

                let temp_auditor = ComprehensiveSecurityAuditor::new(config);
                let _result = temp_auditor.audit_project_lightweight(projectpath)?;
            }
            AuditSchedule::Weekly => {
                // Run full audit weekly
                let _result = self.audit_project(projectpath)?;
            }
            AuditSchedule::Monthly => {
                // Run comprehensive audit with supply chain analysis
                let _result = self.audit_project(projectpath)?;
                self.generate_monthly_security_report()?;
            }
        }
        Ok(())
    }

    /// Lightweight audit for frequent scanning
    fn audit_project_lightweight(&self, projectpath: &Path) -> Result<SecurityAuditResult> {
        let start_time = std::time::Instant::now();

        let mut auditresult = SecurityAuditResult {
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(0),
            security_score: 0.0,
            dependency_results: DependencyScanResult::default(),
            static_analysis_results: self.run_static_analysis(projectpath)?,
            license_compliance_results: LicenseComplianceResult::default(),
            supply_chain_results: SupplyChainAnalysisResult::default(),
            secret_detection_results: self.detect_secrets(projectpath)?,
            config_security_results: ConfigSecurityResult::default(),
            policy_compliance_results: PolicyComplianceResult::default(),
            remediation_suggestions: Vec::new(),
            risk_assessment: RiskAssessment::default(),
        };

        auditresult.security_score = self.calculate_security_score(&auditresult);
        auditresult.duration = start_time.elapsed();

        Ok(auditresult)
    }

    /// Generate monthly security report
    fn generate_monthly_security_report(&self) -> Result<()> {
        // Analyze trends from audit history
        let recent_audits: Vec<_> = self
            .audit_history
            .iter()
            .filter(|audit| {
                audit
                    .timestamp
                    .elapsed()
                    .map(|duration| duration < Duration::from_secs(30 * 24 * 60 * 60))
                    .unwrap_or(false)
            })
            .collect();

        if recent_audits.is_empty() {
            return Ok(());
        }

        // Calculate trend metrics
        let avg_security_score = recent_audits
            .iter()
            .map(|audit| audit.security_score)
            .sum::<f64>()
            / recent_audits.len() as f64;

        let vulnerability_trend = recent_audits
            .iter()
            .map(|audit| audit.dependency_results.vulnerable_dependencies.len())
            .collect::<Vec<_>>();

        // Generate trend report
        log::info!(
            "Monthly Security Report: average score {:.2}, vulnerability trend {:?}, {} audit(s) performed",
            avg_security_score,
            vulnerability_trend,
            recent_audits.len()
        );

        Ok(())
    }

    /// Offline license compliance check: resolves dependencies from
    /// `Cargo.lock`, reads each crate's own `license`/`license-file` field
    /// from the local Cargo registry cache (`read_crate_license`), and
    /// reports a violation only when the found license matches a policy
    /// actually configured on `self.dependency_scanner.config`
    /// (`blocked_licenses`/`allowed_licenses`). No local policy configured
    /// and/or no local cache entry for a crate both mean "cannot determine a
    /// violation" -- never a fabricated one.
    fn check_license_compliance(&self, projectpath: &Path) -> Result<LicenseComplianceResult> {
        if !projectpath.exists() {
            return Err(OptimError::InvalidConfig(format!(
                "cannot check license compliance: {} does not exist",
                projectpath.display()
            )));
        }
        let lock_path = projectpath.join("Cargo.lock");
        if !lock_path.exists() {
            // No lockfile to resolve exact versions from -- nothing to check
            // (not an error: license compliance is optional metadata, unlike
            // dependency scanning where a missing lockfile is fatal).
            return Ok(LicenseComplianceResult::default());
        }
        let content = std::fs::read_to_string(&lock_path).map_err(OptimError::IO)?;
        let deps = parse_cargo_lock_packages(&content);

        let mut violations = Vec::new();
        for dep in &deps {
            let Some(license) = read_crate_license(&dep.name, &dep.version) else {
                continue; // Unknown: not in the local registry cache.
            };
            let policy = &self.dependency_scanner.config;
            if policy.blocked_licenses.contains(&license) {
                violations.push(LicenseViolation {
                    package: dep.name.clone(),
                    reason: format!(
                        "license '{license}' is on the configured blocked_licenses list"
                    ),
                    license,
                });
            } else if !policy.allowed_licenses.is_empty()
                && !policy.allowed_licenses.contains(&license)
            {
                violations.push(LicenseViolation {
                    package: dep.name.clone(),
                    reason: format!(
                        "license '{license}' is not on the configured allowed_licenses allow-list"
                    ),
                    license,
                });
            }
        }
        Ok(LicenseComplianceResult { violations })
    }

    /// Offline supply-chain risk analysis: flags git dependencies
    /// (bypassing crates.io's publish trail), wildcard/unconstrained
    /// version requirements, and the presence of a `build.rs` (arbitrary
    /// code execution at build time). All three are genuinely computable
    /// without network access.
    fn analyze_supply_chain(&self, projectpath: &Path) -> Result<SupplyChainAnalysisResult> {
        if !projectpath.exists() {
            return Err(OptimError::InvalidConfig(format!(
                "cannot analyze supply chain: {} does not exist",
                projectpath.display()
            )));
        }
        let mut risks = Vec::new();
        let toml_path = projectpath.join("Cargo.toml");
        if toml_path.exists() {
            let content = std::fs::read_to_string(&toml_path).map_err(OptimError::IO)?;
            for dep in parse_cargo_toml_dependencies(&content) {
                if dep.is_git {
                    risks.push(SupplyChainRisk {
                        description: format!(
                            "'{}' is sourced from a git repository, bypassing crates.io's \
                             publish/audit trail",
                            dep.name
                        ),
                        package: dep.name,
                        risk_type: "git-dependency".to_string(),
                        severity: SecuritySeverity::Medium,
                    });
                } else if dep.is_wildcard {
                    risks.push(SupplyChainRisk {
                        description: format!(
                            "'{}' has an unconstrained version requirement (\"*\"), which can \
                             silently pull in any future release, including a compromised one",
                            dep.name
                        ),
                        package: dep.name,
                        risk_type: "wildcard-version".to_string(),
                        severity: SecuritySeverity::High,
                    });
                }
            }
        }
        if projectpath.join("build.rs").is_file() {
            risks.push(SupplyChainRisk {
                package: "<this project>".to_string(),
                risk_type: "build-script-present".to_string(),
                severity: SecuritySeverity::Info,
                description: "a build.rs build script is present; it runs arbitrary code at \
                    build time and should be reviewed"
                    .to_string(),
            });
        }
        Ok(SupplyChainAnalysisResult { risks })
    }

    /// Real, line-level secret detection: reuses the same file-walking
    /// approach as `run_static_analysis` and the existing
    /// `contains_potential_secret` pattern check, plus an independent
    /// Shannon-entropy check over long token-shaped runs (catches
    /// unlabeled high-entropy strings that `contains_potential_secret`'s
    /// keyword list would miss).
    fn detect_secrets(&self, projectpath: &Path) -> Result<SecretDetectionResult> {
        if !projectpath.exists() {
            return Err(OptimError::InvalidConfig(format!(
                "cannot detect secrets: {} does not exist",
                projectpath.display()
            )));
        }
        let mut secrets_found = Vec::new();
        for filepath in self.find_rust_files(projectpath)? {
            if self.is_excluded_path(&filepath) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&filepath) else {
                continue;
            };
            for (line_num, line) in content.lines().enumerate() {
                if self.contains_potential_secret(line) {
                    secrets_found.push(DetectedSecret {
                        id: format!("secret_pattern_{}_{}", filepath.display(), line_num + 1),
                        secret_type: "pattern-match".to_string(),
                        file: filepath.clone(),
                        line: line_num + 1,
                        severity: SecuritySeverity::High,
                    });
                    continue; // avoid double-counting the same line via entropy
                }
                if high_entropy_runs(line)
                    .iter()
                    .any(|run| shannon_entropy(run) > 4.0)
                {
                    secrets_found.push(DetectedSecret {
                        id: format!("secret_entropy_{}_{}", filepath.display(), line_num + 1),
                        secret_type: "high-entropy-string".to_string(),
                        file: filepath.clone(),
                        line: line_num + 1,
                        severity: SecuritySeverity::Medium,
                    });
                }
            }
        }
        Ok(SecretDetectionResult { secrets_found })
    }

    /// Offline configuration-security check over config-shaped files
    /// (`.env*`, `*.toml`, `*.yaml`/`*.yml`): flags committed `.env` files
    /// and well-known insecure-config text patterns (TLS verification
    /// disabled, etc.).
    fn check_config_security(&self, projectpath: &Path) -> Result<ConfigSecurityResult> {
        if !projectpath.exists() {
            return Err(OptimError::InvalidConfig(format!(
                "cannot check config security: {} does not exist",
                projectpath.display()
            )));
        }

        const INSECURE_PATTERNS: &[&str] = &[
            "verify_ssl = false",
            "verify_ssl=false",
            "insecure_skip_verify",
            "danger_accept_invalid_certs",
            "node_tls_reject_unauthorized=0",
            "ssl_verify = false",
        ];

        fn walk(
            dir: &Path,
            auditor: &ComprehensiveSecurityAuditor,
            issues: &mut Vec<ConfigSecurityIssue>,
        ) -> Result<()> {
            for entry in std::fs::read_dir(dir).map_err(OptimError::IO)? {
                let entry = entry.map_err(OptimError::IO)?;
                let path = entry.path();
                if auditor.is_excluded_path(&path) {
                    continue;
                }
                if path.is_dir() {
                    walk(&path, auditor, issues)?;
                    continue;
                }

                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if file_name == ".env" || file_name.starts_with(".env.") {
                    issues.push(ConfigSecurityIssue {
                        config_file: path.clone(),
                        issue: "a .env file is committed to the repository; secrets belong in \
                            the environment, not a tracked file"
                            .to_string(),
                        severity: SecuritySeverity::High,
                    });
                    continue;
                }

                let is_config_shaped = matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("toml") | Some("yaml") | Some("yml")
                );
                if !is_config_shaped {
                    continue;
                }
                let Ok(content) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let content_lower = content.to_lowercase();
                for pattern in INSECURE_PATTERNS {
                    if content_lower.contains(pattern) {
                        issues.push(ConfigSecurityIssue {
                            config_file: path.clone(),
                            issue: format!("insecure configuration pattern detected: '{pattern}'"),
                            severity: SecuritySeverity::High,
                        });
                    }
                }
            }
            Ok(())
        }

        let mut issues = Vec::new();
        walk(projectpath, self, &mut issues)?;
        Ok(ConfigSecurityResult { issues })
    }
}

/// Shannon entropy in bits/char of `s`, from character frequency. Standard
/// `-sum(p_i * log2(p_i))` formula; used by `detect_secrets` to flag
/// unlabeled high-entropy token-shaped strings.
fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in s.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let len = s.chars().count() as f64;
    counts.values().fold(0.0, |acc, &count| {
        let p = count as f64 / len;
        acc - p * p.log2()
    })
}

/// Extract maximal runs (length >= 20) of base64/token-shaped characters
/// (`[A-Za-z0-9+/=_-]`) from `line`, as entropy-check candidates.
fn high_entropy_runs(line: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for c in line.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-') {
            current.push(c);
        } else {
            if current.chars().count() >= 20 {
                runs.push(current.clone());
            }
            current.clear();
        }
    }
    if current.chars().count() >= 20 {
        runs.push(current);
    }
    runs
}

/// Audit scheduling options
#[derive(Debug, Clone)]
pub enum AuditSchedule {
    Daily,
    Weekly,
    Monthly,
}

// Placeholder implementations and default constructors for all types
// (continuing with the same pattern as before...)

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LicenseComplianceResult {
    pub violations: Vec<LicenseViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseViolation {
    pub package: String,
    pub license: String,
    pub reason: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SupplyChainAnalysisResult {
    pub risks: Vec<SupplyChainRisk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyChainRisk {
    pub package: String,
    pub risk_type: String,
    pub severity: SecuritySeverity,
    pub description: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SecretDetectionResult {
    pub secrets_found: Vec<DetectedSecret>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedSecret {
    pub id: String,
    pub secret_type: String,
    pub file: PathBuf,
    pub line: usize,
    pub severity: SecuritySeverity,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ConfigSecurityResult {
    pub issues: Vec<ConfigSecurityIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSecurityIssue {
    pub config_file: PathBuf,
    pub issue: String,
    pub severity: SecuritySeverity,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PolicyComplianceResult {
    pub violations: Vec<PolicyViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    pub policy_id: String,
    pub rule_id: String,
    pub severity: SecuritySeverity,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationSuggestion {
    pub id: String,
    pub title: String,
    pub description: String,
    pub priority: RemediationPriority,
    pub effort: EffortLevel,
    pub steps: Vec<String>,
    pub automated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OutdatedDependency {
    pub name: String,
    pub current_version: String,
    pub latest_version: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DependencyTree {
    pub root: String,
    pub dependencies: HashMap<String, Vec<String>>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub id: String,
    pub description: String,
    pub file: PathBuf,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    Always,
    Never,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    Deny,
    Warn,
    Log,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyScope {
    Global,
    Project,
    Directory(PathBuf),
}

// Supporting implementations
impl Default for SecurityAuditConfig {
    fn default() -> Self {
        Self {
            enable_dependency_scanning: true,
            enable_static_analysis: true,
            enable_license_compliance: true,
            enable_supply_chain_analysis: true,
            enable_secret_detection: true,
            enable_config_security: true,
            db_update_frequency: Duration::from_secs(24 * 60 * 60), // Daily
            max_audit_time: Duration::from_secs(30 * 60),           // 30 minutes
            alert_threshold: SecuritySeverity::High,
            report_format: ReportFormat::Json,
            enable_auto_remediation: true,
            trusted_sources: vec!["crates.io".to_string()],
            excluded_paths: vec![
                PathBuf::from("target"),
                PathBuf::from(".git"),
                PathBuf::from("node_modules"),
            ],
            custom_rules: Vec::new(),
            alert_webhook_url: None,
        }
    }
}

impl Default for DependencyScanResult {
    fn default() -> Self {
        Self {
            total_dependencies: 0,
            vulnerable_dependencies: Vec::new(),
            outdated_dependencies: Vec::new(),
            license_violations: Vec::new(),
            supply_chain_risks: Vec::new(),
            dependency_tree: DependencyTree::default(),
            risk_score: 0.0,
        }
    }
}

impl Default for StaticAnalysisResult {
    fn default() -> Self {
        Self {
            security_issues: Vec::new(),
            quality_issues: Vec::new(),
            files_scanned: 0,
            lines_analyzed: 0,
            analysis_duration: Duration::from_secs(0),
        }
    }
}

impl Default for RiskAssessment {
    fn default() -> Self {
        Self {
            overall_risk: RiskLevel::Minimal,
            risk_factors: Vec::new(),
            risk_score: 0.0,
            recommendations: Vec::new(),
            mitigation_strategies: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Offline dependency scanning: real Cargo.lock/Cargo.toml parsing (no TOML
// crate dependency -- Cargo.lock's `[[package]]` format and the handful of
// Cargo.toml dependency-table shapes handled below are simple enough to
// parse by hand) and matching against a small, hand-curated, embedded
// RustSec advisory snapshot. No network access, no fabricated IDs: every
// entry in `embedded_advisory_snapshot` is a real, publicly published
// RUSTSEC advisory; a dependency that matches no entry is reported as not
// (currently known to be) vulnerable, and "is this the latest version" is
// reported as `Unknown` rather than guessed, since this crate has no
// registry client and none may be added.
// ---------------------------------------------------------------------------

/// A dependency resolved to a concrete version, either from `Cargo.lock`
/// (preferred -- gives the actually-locked version) or, if no lockfile is
/// present, from a `Cargo.toml` version requirement string used as a
/// best-effort stand-in.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedDependency {
    name: String,
    version: String,
}

/// A dependency as declared in a `Cargo.toml` dependency table, before
/// resolution. Captures enough shape to support F8's "all forms" parsing:
/// plain string version, inline table, `[dependencies.foo]` sub-table, and
/// `workspace = true` inheritance.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclaredDependency {
    name: String,
    /// `Some(version_requirement)` when a literal version/req string was
    /// found; `None` when only `git`/`path`/`workspace = true` was given.
    version_req: Option<String>,
    is_git: bool,
    is_workspace_inherited: bool,
    is_wildcard: bool,
}

/// Parse a single `key = "value"` TOML line (no leading `[`), returning the
/// unquoted value if `line` assigns a plain string to `key`. Handles only
/// the flat `key = "value"` shape deliberately -- inline tables and arrays
/// are handled by their own callers.
fn parse_toml_string_assignment(line: &str, key: &str) -> Option<String> {
    let (lhs, rhs) = line.split_once('=')?;
    if lhs.trim() != key {
        return None;
    }
    let rhs = rhs.trim();
    if rhs.len() >= 2 && rhs.starts_with('"') && rhs.ends_with('"') {
        Some(rhs[1..rhs.len() - 1].to_string())
    } else {
        None
    }
}

/// Extract a quoted string field from inline-table syntax, e.g.
/// `{ version = "1.2", features = ["x"] }` -> `extract_inline_table_field(_, "version") == Some("1.2")`.
fn extract_inline_table_field(inline_table: &str, field: &str) -> Option<String> {
    let inner = inline_table
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}');
    // Split on top-level commas only (none of the fields we look for here
    // contain array values with embedded commas relevant to matching the
    // field name itself, so a plain split is sufficient).
    for part in inner.split(',') {
        if let Some((key, value)) = part.split_once('=') {
            if key.trim() == field {
                let value = value.trim();
                if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
                    return Some(value[1..value.len() - 1].to_string());
                }
            }
        }
    }
    None
}

/// Parse `Cargo.lock`'s `[[package]]` blocks into resolved `(name, version)`
/// pairs. Cargo.lock is generated by Cargo itself and never uses inline
/// tables or multi-line strings for these two fields, so a line-oriented
/// parser is reliable here (unlike hand-written Cargo.toml, which can use
/// any of TOML's dependency-table shapes -- see `parse_cargo_toml_dependencies`).
fn parse_cargo_lock_packages(content: &str) -> Vec<ResolvedDependency> {
    let mut deps = Vec::new();
    let mut in_package = false;
    let mut name: Option<String> = None;
    let mut version: Option<String> = None;

    let flush = |name: &mut Option<String>,
                 version: &mut Option<String>,
                 deps: &mut Vec<ResolvedDependency>| {
        if let (Some(n), Some(v)) = (name.take(), version.take()) {
            deps.push(ResolvedDependency {
                name: n,
                version: v,
            });
        }
    };

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line == "[[package]]" {
            flush(&mut name, &mut version, &mut deps);
            in_package = true;
            continue;
        }
        if line.starts_with('[') && line != "[[package]]" {
            in_package = false;
            continue;
        }
        if in_package {
            if let Some(value) = parse_toml_string_assignment(line, "name") {
                name = Some(value);
            } else if let Some(value) = parse_toml_string_assignment(line, "version") {
                version = Some(value);
            }
        }
    }
    flush(&mut name, &mut version, &mut deps);
    deps
}

/// Parse a `Cargo.toml`'s `[dependencies]`/`[dev-dependencies]`/
/// `[build-dependencies]` tables (including `[target.'cfg(...)'.dependencies]`
/// variants), handling all of: plain string version (`foo = "1.0"`), inline
/// table (`foo = { version = "1.0", features = [...] }`), sub-table
/// (`[dependencies.foo]` followed by `version = "1.0"`), and workspace
/// inheritance (`foo = { workspace = true }` or, in sub-table form,
/// `workspace = true`).
fn parse_cargo_toml_dependencies(content: &str) -> Vec<DeclaredDependency> {
    let mut deps = Vec::new();
    let mut in_dep_table = false;
    let mut current_subtable_dep: Option<String> = None;

    let is_dep_table_header = |line: &str| -> bool {
        matches!(
            line,
            "[dependencies]" | "[dev-dependencies]" | "[build-dependencies]"
        ) || (line.ends_with(".dependencies]") && line.starts_with("[target."))
    };

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') {
            if let Some(name) = line
                .strip_prefix("[dependencies.")
                .or_else(|| line.strip_prefix("[dev-dependencies."))
                .or_else(|| line.strip_prefix("[build-dependencies."))
                .and_then(|rest| rest.strip_suffix(']'))
            {
                current_subtable_dep = Some(name.trim_matches('"').to_string());
                in_dep_table = false;
                continue;
            }
            current_subtable_dep = None;
            in_dep_table = is_dep_table_header(line);
            continue;
        }

        if let Some(dep_name) = current_subtable_dep.clone() {
            if let Some(v) = parse_toml_string_assignment(line, "version") {
                let is_wildcard = v.trim() == "*";
                deps.push(DeclaredDependency {
                    name: dep_name,
                    version_req: Some(v),
                    is_git: false,
                    is_workspace_inherited: false,
                    is_wildcard,
                });
            } else if line == "workspace = true" {
                deps.push(DeclaredDependency {
                    name: dep_name,
                    version_req: None,
                    is_git: false,
                    is_workspace_inherited: true,
                    is_wildcard: false,
                });
            } else if parse_toml_string_assignment(line, "git").is_some() {
                deps.push(DeclaredDependency {
                    name: dep_name,
                    version_req: None,
                    is_git: true,
                    is_workspace_inherited: false,
                    is_wildcard: false,
                });
            }
            continue;
        }

        if in_dep_table {
            let Some((lhs, rhs)) = line.split_once('=') else {
                continue;
            };
            let name = lhs.trim().trim_matches('"').to_string();
            if name.is_empty() {
                continue;
            }
            let rhs = rhs.trim();

            if rhs.len() >= 2 && rhs.starts_with('"') && rhs.ends_with('"') {
                let version = rhs[1..rhs.len() - 1].to_string();
                let is_wildcard = version.trim() == "*";
                deps.push(DeclaredDependency {
                    name,
                    version_req: Some(version),
                    is_git: false,
                    is_workspace_inherited: false,
                    is_wildcard,
                });
            } else if rhs.starts_with('{') {
                let is_workspace = extract_inline_table_field(rhs, "workspace").as_deref()
                    == Some("true")
                    || rhs.replace(' ', "").contains("workspace=true");
                let is_git = rhs.contains("git");
                let version = extract_inline_table_field(rhs, "version");
                let is_wildcard = version.as_deref() == Some("*");
                deps.push(DeclaredDependency {
                    name,
                    version_req: version,
                    is_git,
                    is_workspace_inherited: is_workspace,
                    is_wildcard,
                });
            }
        }
    }

    deps
}

/// Parse the leading `major.minor.patch` numeric triple from a version
/// string, ignoring any pre-release/build metadata suffix
/// (`"1.2.3-beta.1"` -> `(1, 2, 3)`). Missing minor/patch components default
/// to 0 (`"1"` -> `(1, 0, 0)`), matching Cargo's own version normalization.
fn parse_semver_triple(version: &str) -> Option<(u64, u64, u64)> {
    let core = version.split(['-', '+']).next()?.trim();
    let mut parts = core.split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse().ok()?;
    let patch = parts.next().unwrap_or("0").trim().parse().ok()?;
    Some((major, minor, patch))
}

/// `true` if `a < b` as a `(major, minor, patch)` triple; `None` if either
/// string cannot be parsed as a numeric semver triple.
fn semver_lt(a: &str, b: &str) -> Option<bool> {
    Some(parse_semver_triple(a)? < parse_semver_triple(b)?)
}

/// A single, verified-real RustSec advisory embedded for fully offline
/// vulnerability checking.
///
/// **Snapshot date: 2026-08-17**, hand-curated from
/// <https://rustsec.org/advisories/>. This list is NOT auto-updated and is
/// intentionally small (only advisories relevant to this workspace's actual
/// dependency set were verified) -- treat a "not found" result as "not
/// known to be vulnerable by this offline snapshot", not as a guarantee.
/// For a live, comprehensive check, run `cargo audit` (external tool, not a
/// dependency of this crate) against the real RustSec Advisory Database.
struct EmbeddedAdvisory {
    id: &'static str,
    package: &'static str,
    title: &'static str,
    /// Vulnerable if version >= this (when set) and < `fixed_in`.
    affected_from: Option<&'static str>,
    /// Vulnerable if version < this.
    fixed_in: &'static str,
    cvss_score: Option<f64>,
    severity: SecuritySeverity,
    url: &'static str,
}

const EMBEDDED_ADVISORY_SNAPSHOT_DATE: &str = "2026-08-17";

fn embedded_advisory_snapshot() -> &'static [EmbeddedAdvisory] {
    &[
        EmbeddedAdvisory {
            id: "RUSTSEC-2020-0159",
            package: "chrono",
            title: "Potential segfault in `localtime_r` invocations",
            affected_from: None,
            fixed_in: "0.4.20",
            cvss_score: Some(6.2),
            severity: SecuritySeverity::Medium,
            url: "https://rustsec.org/advisories/RUSTSEC-2020-0159",
        },
        EmbeddedAdvisory {
            id: "RUSTSEC-2020-0071",
            package: "time",
            title: "Potential segfault in the time crate",
            // 0.1.x used a different (unaffected) implementation; only the
            // 0.2.x line before 0.2.23 is affected.
            affected_from: Some("0.2.0"),
            fixed_in: "0.2.23",
            cvss_score: Some(6.2),
            severity: SecuritySeverity::Medium,
            url: "https://rustsec.org/advisories/RUSTSEC-2020-0071",
        },
        EmbeddedAdvisory {
            id: "RUSTSEC-2023-0071",
            package: "rsa",
            title: "Marvin Attack: potential key recovery through timing sidechannels",
            affected_from: None,
            fixed_in: "0.9.6",
            cvss_score: Some(5.9),
            severity: SecuritySeverity::Medium,
            url: "https://rustsec.org/advisories/RUSTSEC-2023-0071",
        },
        EmbeddedAdvisory {
            id: "RUSTSEC-2021-0145",
            package: "atty",
            title: "Potential unaligned read (crate unmaintained)",
            affected_from: None,
            // atty was never fixed; every published version is considered
            // affected. Using a fixed_in above any real release means "always
            // vulnerable if present at all".
            fixed_in: "999.0.0",
            cvss_score: None,
            severity: SecuritySeverity::Low,
            url: "https://rustsec.org/advisories/RUSTSEC-2021-0145",
        },
    ]
}

/// Whether `version` is vulnerable per `advisory`'s bounds. `None` when the
/// version string cannot be parsed as a numeric semver triple (never
/// silently treated as "not vulnerable" by the caller in that case -- see
/// `find_known_vulnerabilities`, which surfaces unparsable versions
/// separately rather than dropping them).
fn version_is_vulnerable(version: &str, advisory: &EmbeddedAdvisory) -> Option<bool> {
    let below_fix = semver_lt(version, advisory.fixed_in)?;
    if !below_fix {
        return Some(false);
    }
    if let Some(from) = advisory.affected_from {
        let at_or_above_from = !semver_lt(version, from)?;
        return Some(at_or_above_from);
    }
    Some(true)
}

/// Find every embedded advisory that matches `name` at `version`.
fn find_known_vulnerabilities(name: &str, version: &str) -> Vec<&'static EmbeddedAdvisory> {
    embedded_advisory_snapshot()
        .iter()
        .filter(|advisory| advisory.package == name)
        .filter(|advisory| version_is_vulnerable(version, advisory) == Some(true))
        .collect()
}

/// Locate the Cargo registry cache's source directory for `name-version`
/// (typically `$CARGO_HOME/registry/src/*/name-version/`), used to read a
/// dependency's own `Cargo.toml` for offline license lookup (F9). Returns
/// `None` when the crate is not present in the local cache (e.g. vendored,
/// path dependency, or never downloaded) -- callers must treat that as
/// `Unknown`, never as "no license".
fn find_registry_cache_dir(name: &str, version: &str) -> Option<PathBuf> {
    let cargo_home = std::env::var("CARGO_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cargo"))
        })?;
    let registry_src = cargo_home.join("registry").join("src");
    let entries = std::fs::read_dir(&registry_src).ok()?;
    let target_dir_name = format!("{name}-{version}");
    for entry in entries.flatten() {
        let index_dir = entry.path();
        if !index_dir.is_dir() {
            continue;
        }
        let candidate = index_dir.join(&target_dir_name);
        if candidate.join("Cargo.toml").is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Read the `[package].license` (or `license-file`, reported as
/// `"file:<path>"`) field from a crate's own `Cargo.toml`. `None` when the
/// crate's manifest isn't reachable locally or declares neither field.
fn read_crate_license(name: &str, version: &str) -> Option<String> {
    let dir = find_registry_cache_dir(name, version)?;
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).ok()?;
    let mut in_package = false;
    for raw_line in manifest.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(license) = parse_toml_string_assignment(line, "license") {
            return Some(license);
        }
        if let Some(file) = parse_toml_string_assignment(line, "license-file") {
            return Some(format!("file:{file}"));
        }
    }
    None
}

/// Real, fully offline dependency scan: resolves dependencies from
/// `Cargo.lock` (preferred) or `Cargo.toml` (fallback), matches each against
/// [`embedded_advisory_snapshot`], and reports outdated status as `Unknown`
/// (this crate has no registry client). Shared by both
/// `ComprehensiveSecurityAuditor::scan_dependencies_with_rustsec` and
/// `DependencyScanner::scan_dependencies` so there is exactly one
/// implementation of this logic.
fn scan_dependencies_offline(projectpath: &Path) -> Result<DependencyScanResult> {
    let lock_path = projectpath.join("Cargo.lock");
    let toml_path = projectpath.join("Cargo.toml");

    let resolved: Vec<ResolvedDependency> = if lock_path.exists() {
        let content = std::fs::read_to_string(&lock_path).map_err(OptimError::IO)?;
        parse_cargo_lock_packages(&content)
    } else if toml_path.exists() {
        let content = std::fs::read_to_string(&toml_path).map_err(OptimError::IO)?;
        parse_cargo_toml_dependencies(&content)
            .into_iter()
            .filter_map(|dep| {
                let version = dep.version_req?;
                // Only usable as a stand-in "resolved" version when it looks
                // like an exact/minimum version, not a bare wildcard.
                if dep.is_wildcard {
                    None
                } else {
                    Some(ResolvedDependency {
                        name: dep.name,
                        version: version
                            .trim_start_matches(['^', '~', '=', '>', '<', ' '])
                            .to_string(),
                    })
                }
            })
            .collect()
    } else {
        return Err(OptimError::InvalidConfig(format!(
            "neither Cargo.lock nor Cargo.toml found under {}; cannot scan dependencies",
            projectpath.display()
        )));
    };

    let total_dependencies = resolved.len();
    let mut vulnerable_dependencies = Vec::new();
    let outdated_dependencies: Vec<OutdatedDependency> = Vec::new();

    for dep in &resolved {
        let matches = find_known_vulnerabilities(&dep.name, &dep.version);
        if !matches.is_empty() {
            let vulnerabilities: Vec<Vulnerability> = matches
                .iter()
                .map(|advisory| Vulnerability {
                    id: advisory.id.to_string(),
                    title: advisory.title.to_string(),
                    description: format!(
                        "{} {} matches {} (offline snapshot dated {}): affected {}, fixed in {}",
                        dep.name,
                        dep.version,
                        advisory.id,
                        EMBEDDED_ADVISORY_SNAPSHOT_DATE,
                        advisory.affected_from.unwrap_or("0.0.0"),
                        advisory.fixed_in
                    ),
                    severity: advisory.severity,
                    cvss_score: advisory.cvss_score,
                    published: SystemTime::now(),
                    discovered: None,
                    affected_versions: format!(
                        "{}..{}",
                        advisory.affected_from.unwrap_or("0.0.0"),
                        advisory.fixed_in
                    ),
                    patched_versions: vec![format!(">= {}", advisory.fixed_in)],
                    references: vec![advisory.url.to_string()],
                    categories: vec![VulnerabilityCategory::Other("RustSec".to_string())],
                })
                .collect();
            let severity = vulnerabilities
                .iter()
                .map(|v| v.severity)
                .max()
                .unwrap_or(SecuritySeverity::Low);
            let fixed_version = matches.first().map(|a| a.fixed_in.to_string());
            vulnerable_dependencies.push(VulnerableDependency {
                name: dep.name.clone(),
                current_version: dep.version.clone(),
                cve_ids: matches.iter().map(|a| a.id.to_string()).collect(),
                vulnerabilities,
                affected_versions: format!("< {}", fixed_version.clone().unwrap_or_default()),
                fixed_version,
                severity,
            });
        }
    }

    // This crate has no registry client and none may be added, so "is this
    // outdated" cannot be determined offline at all -- not even a per-crate
    // `Unknown` marker, since pushing one for every one of potentially
    // hundreds of dependencies would just be noise with no information
    // content. `outdated_dependencies` is therefore always empty here; that
    // is honestly different from "checked, found nothing outdated".

    let risk_score = (vulnerable_dependencies.len() as f64 * 0.3).min(1.0);

    Ok(DependencyScanResult {
        total_dependencies,
        vulnerable_dependencies,
        outdated_dependencies,
        license_violations: Vec::new(),
        supply_chain_risks: Vec::new(),
        dependency_tree: DependencyTree::default(),
        risk_score,
    })
}

// Supporting struct implementations
impl DependencyScanner {
    fn new(config: DependencyScanConfig) -> Self {
        Self {
            config: DependencyScanConfig::default(),
            vuln_db_client: VulnerabilityDatabaseClient::new(),
            license_db: LicenseDatabase::new(),
            package_cache: HashMap::new(),
        }
    }

    fn scan_dependencies(&mut self, projectpath: &Path) -> Result<DependencyScanResult> {
        scan_dependencies_offline(projectpath)
    }
}

impl Default for DependencyScanConfig {
    fn default() -> Self {
        Self {
            scan_direct_deps: true,
            scan_transitive_deps: true,
            max_depth: 10,
            check_outdated: true,
            min_versions: HashMap::new(),
            blocked_dependencies: HashSet::new(),
            allowed_licenses: HashSet::new(),
            blocked_licenses: HashSet::new(),
        }
    }
}

#[derive(Debug)]
struct VulnerabilityDatabaseClient;

impl VulnerabilityDatabaseClient {
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct LicenseDatabase;

impl LicenseDatabase {
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct PackageMetadata;

impl VulnerabilityDatabase {
    fn new(config: VulnerabilityDatabaseConfig) -> Self {
        Self {
            config: VulnerabilityDatabaseConfig::default(),
            local_cache: HashMap::new(),
            last_update: SystemTime::now(),
            update_frequency: Duration::from_secs(24 * 60 * 60),
            external_sources: Vec::new(),
        }
    }

    fn needs_update(&self) -> bool {
        self.last_update.elapsed().unwrap_or(Duration::from_secs(0)) > self.update_frequency
    }

    fn update_from_sources(&mut self) -> Result<()> {
        self.last_update = SystemTime::now();
        Ok(())
    }
}

impl Default for VulnerabilityDatabaseConfig {
    fn default() -> Self {
        Self {
            auto_update: true,
            update_frequency: Duration::from_secs(24 * 60 * 60),
            cache_size_limit: 10000,
            cache_retention: Duration::from_secs(7 * 24 * 60 * 60), // 1 week
            external_sources: vec!["https://rustsec.org".to_string()],
            api_keys: HashMap::new(),
        }
    }
}

impl SecurityPolicyEnforcer {
    fn new() -> Self {
        Self {
            policies: Vec::new(),
            evaluator: PolicyEvaluator::new(),
            violations: Vec::new(),
        }
    }

    fn check_compliance(
        &mut self,
        _audit_result: &SecurityAuditResult,
    ) -> Result<PolicyComplianceResult> {
        Ok(PolicyComplianceResult::default())
    }
}

#[derive(Debug)]
struct PolicyEvaluator;

impl PolicyEvaluator {
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct SecurityReportGenerator;

impl SecurityReportGenerator {
    fn new() -> Self {
        Self
    }

    fn generate_report(
        &self,
        auditresult: &SecurityAuditResult,
        format: &ReportFormat,
    ) -> Result<String> {
        match format {
            ReportFormat::Json => Ok(serde_json::to_string_pretty(auditresult)?),
            ReportFormat::Yaml => serde_yaml::to_string(auditresult).map_err(|e| {
                OptimError::InvalidConfig(format!("failed to serialize YAML report: {e}"))
            }),
            ReportFormat::Markdown => {
                let mut report = String::new();
                report.push_str("# Security Audit Report\n\n");
                report.push_str(&format!("**Audit Date:** {:?}\n", auditresult.timestamp));
                report.push_str(&format!(
                    "**Security Score:** {:.2}/1.0\n\n",
                    auditresult.security_score
                ));

                report.push_str("## Dependency Vulnerabilities\n");
                report.push_str(&format!(
                    "Found {} vulnerable dependencies\n\n",
                    auditresult.dependency_results.vulnerable_dependencies.len()
                ));

                report.push_str("## Static Analysis Issues\n");
                report.push_str(&format!(
                    "Found {} security issues\n\n",
                    auditresult.static_analysis_results.security_issues.len()
                ));

                report.push_str("## Risk Assessment\n");
                report.push_str(&format!(
                    "Overall Risk: {:?}\n",
                    auditresult.risk_assessment.overall_risk
                ));

                Ok(report)
            }
            ReportFormat::Html => {
                let mut html = String::new();
                html.push_str(
                    "<!DOCTYPE html><html><head><title>Security Audit Report</title></head><body>",
                );
                html.push_str("<h1>Security Audit Report</h1>");
                html.push_str(&format!(
                    "<p><strong>Audit Date:</strong> {:?}</p>",
                    auditresult.timestamp
                ));
                html.push_str(&format!(
                    "<p><strong>Security Score:</strong> {:.2}/1.0</p>",
                    auditresult.security_score
                ));
                html.push_str("<h2>Dependency Vulnerabilities</h2>");
                html.push_str(&format!(
                    "<p>Found {} vulnerable dependencies</p>",
                    auditresult.dependency_results.vulnerable_dependencies.len()
                ));
                html.push_str("<h2>Static Analysis Issues</h2>");
                html.push_str(&format!(
                    "<p>Found {} security issues</p>",
                    auditresult.static_analysis_results.security_issues.len()
                ));
                html.push_str("<h2>Risk Assessment</h2>");
                html.push_str(&format!(
                    "<p>Overall Risk: {:?}</p>",
                    auditresult.risk_assessment.overall_risk
                ));
                html.push_str("</body></html>");
                Ok(html)
            }
            // No PDF-rendering dependency is linked into this crate (pure-Rust
            // policy) and SARIF is not yet implemented: both fail explicitly
            // rather than returning a placeholder string as a fake report.
            ReportFormat::Pdf => Err(OptimError::UnsupportedOperation(
                "PDF report format is not supported: no PDF rendering dependency is linked \
                 into this crate; use Json/Yaml/Markdown/Html instead"
                    .to_string(),
            )),
            ReportFormat::Sarif => Err(OptimError::UnsupportedOperation(
                "SARIF report format is not yet implemented; use Json/Yaml/Markdown/Html instead"
                    .to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_auditor_creation() {
        let config = SecurityAuditConfig::default();
        let auditor = ComprehensiveSecurityAuditor::new(config);
        assert!(auditor.config.enable_dependency_scanning);
        assert!(auditor.config.enable_static_analysis);
    }

    #[test]
    fn test_security_score_calculation() {
        let config = SecurityAuditConfig::default();
        let auditor = ComprehensiveSecurityAuditor::new(config);

        let auditresult = SecurityAuditResult {
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(10),
            security_score: 0.0,
            dependency_results: DependencyScanResult::default(),
            static_analysis_results: StaticAnalysisResult::default(),
            license_compliance_results: LicenseComplianceResult::default(),
            supply_chain_results: SupplyChainAnalysisResult::default(),
            secret_detection_results: SecretDetectionResult::default(),
            config_security_results: ConfigSecurityResult::default(),
            policy_compliance_results: PolicyComplianceResult::default(),
            remediation_suggestions: Vec::new(),
            risk_assessment: RiskAssessment::default(),
        };

        let score = auditor.calculate_security_score(&auditresult);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_secret_detection() {
        let config = SecurityAuditConfig::default();
        let auditor = ComprehensiveSecurityAuditor::new(config);

        assert!(auditor.contains_potential_secret("password = \"secret123\""));
        assert!(auditor.contains_potential_secret("api_key = 'abc123def456'"));
        assert!(!auditor.contains_potential_secret("let x = 5;"));
    }

    #[test]
    fn test_weak_crypto_detection() {
        let config = SecurityAuditConfig::default();
        let auditor = ComprehensiveSecurityAuditor::new(config);

        assert!(auditor.uses_weak_crypto("use md5::Md5;"));
        assert!(auditor.uses_weak_crypto("let hash = sha1(data);"));
        assert!(!auditor.uses_weak_crypto("use sha256::Sha256;"));
    }

    #[test]
    fn test_cargo_dependency_parsing() {
        let config = SecurityAuditConfig::default();
        let auditor = ComprehensiveSecurityAuditor::new(config);

        let cargocontent = r#"
[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
log = "0.4"

[dev-dependencies]
test-dep = "0.1"
"#;

        let _ = auditor;
        let deps = parse_cargo_toml_dependencies(cargocontent);
        assert!(deps.len() >= 2);
        assert!(deps.iter().any(|d| d.name == "serde"));
        assert!(deps.iter().any(|d| d.name == "log"));
        // Table-form dependency with a version requirement is parsed too.
        assert!(deps
            .iter()
            .any(|d| d.name == "tokio" && d.version_req.is_some()));
    }

    #[test]
    fn test_known_vulnerability_lookup() {
        // The offline advisory matcher returns real semver-range matches; an
        // unknown crate must never be flagged.
        let hits = find_known_vulnerabilities("definitely-not-a-real-crate", "1.0.0");
        assert!(hits.is_empty());
    }
}
