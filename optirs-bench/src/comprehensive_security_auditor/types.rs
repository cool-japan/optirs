//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::functions::scan_dependencies_offline;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseViolation {
    pub package: String,
    pub license: String,
    pub reason: String,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SecretDetectionResult {
    pub secrets_found: Vec<DetectedSecret>,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DependencyTree {
    pub root: String,
    pub dependencies: HashMap<String, Vec<String>>,
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
/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
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
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LicenseComplianceResult {
    pub violations: Vec<LicenseViolation>,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ConfigSecurityResult {
    pub issues: Vec<ConfigSecurityIssue>,
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
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SupplyChainAnalysisResult {
    pub risks: Vec<SupplyChainRisk>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    pub policy_id: String,
    pub rule_id: String,
    pub severity: SecuritySeverity,
    pub description: String,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OutdatedDependency {
    pub name: String,
    pub current_version: String,
    pub latest_version: String,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub id: String,
    pub description: String,
    pub file: PathBuf,
    pub line: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyScope {
    Global,
    Project,
    Directory(PathBuf),
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
/// A dependency as declared in a `Cargo.toml` dependency table, before
/// resolution. Captures enough shape to support F8's "all forms" parsing:
/// plain string version, inline table, `[dependencies.foo]` sub-table, and
/// `workspace = true` inheritance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DeclaredDependency {
    pub(super) name: String,
    /// `Some(version_requirement)` when a literal version/req string was
    /// found; `None` when only `git`/`path`/`workspace = true` was given.
    pub(super) version_req: Option<String>,
    pub(super) is_git: bool,
    pub(super) is_workspace_inherited: bool,
    pub(super) is_wildcard: bool,
}
/// A dependency resolved to a concrete version, either from `Cargo.lock`
/// (preferred -- gives the actually-locked version) or, if no lockfile is
/// present, from a `Cargo.toml` version requirement string used as a
/// best-effort stand-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResolvedDependency {
    pub(super) name: String,
    pub(super) version: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    Deny,
    Warn,
    Log,
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
pub(super) struct EmbeddedAdvisory {
    pub(super) id: &'static str,
    pub(super) package: &'static str,
    pub(super) title: &'static str,
    /// Vulnerable if version >= this (when set) and < `fixed_in`.
    pub(super) affected_from: Option<&'static str>,
    /// Vulnerable if version < this.
    pub(super) fixed_in: &'static str,
    pub(super) cvss_score: Option<f64>,
    pub(super) severity: SecuritySeverity,
    pub(super) url: &'static str,
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
/// Dependency scanner for vulnerability detection.
///
/// `vuln_db_client`/`license_db`/`package_cache` fields from an earlier,
/// stateful-online-lookup design were removed: [`scan_dependencies_offline`]
/// (the real, working implementation `scan_dependencies` delegates to) is a
/// pure function of a project path and this `config`, so those fields were
/// never read by anything -- genuinely superseded scaffolding, not a feature
/// gap.
#[derive(Debug)]
pub struct DependencyScanner {
    /// Scanner configuration
    pub(super) config: DependencyScanConfig,
}
impl DependencyScanner {
    pub(super) fn new(config: DependencyScanConfig) -> Self {
        Self { config }
    }
    pub(super) fn scan_dependencies(&mut self, projectpath: &Path) -> Result<DependencyScanResult> {
        scan_dependencies_offline(projectpath, &self.config)
    }
}
/// Vulnerability database for tracking known security issues.
///
/// `local_cache`/`external_sources` fields from an earlier online-lookup
/// design were removed along with the `config` fields that only ever
/// initialized them: nothing populates or reads a local cache, and fetching
/// from `external_sources` would mean giving this offline-only scanner (see
/// [`scan_dependencies_offline`]) a registry client, which is a real feature
/// addition, not a mechanical wire-up. `auto_update` and `update_frequency`
/// -- the two `VulnerabilityDatabaseConfig` fields [`Self::needs_update`]
/// can actually honor without that -- are kept.
#[derive(Debug)]
pub struct VulnerabilityDatabase {
    /// Whether automatic updates are enabled.
    pub(super) auto_update: bool,
    /// Database update status
    pub(super) last_update: SystemTime,
    /// Update frequency
    pub(super) update_frequency: Duration,
}
impl VulnerabilityDatabase {
    pub(super) fn new(config: VulnerabilityDatabaseConfig) -> Self {
        Self {
            auto_update: config.auto_update,
            last_update: SystemTime::now(),
            update_frequency: config.update_frequency,
        }
    }
    pub(super) fn needs_update(&self) -> bool {
        self.auto_update
            && self.last_update.elapsed().unwrap_or(Duration::from_secs(0)) > self.update_frequency
    }
    pub(super) fn update_from_sources(&mut self) -> Result<()> {
        self.last_update = SystemTime::now();
        Ok(())
    }
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
#[derive(Debug)]
pub(super) struct PolicyEvaluator;
impl PolicyEvaluator {
    pub(super) fn new() -> Self {
        Self
    }
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
pub enum RemediationPriority {
    Low,
    Medium,
    High,
    Critical,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedSecret {
    pub id: String,
    pub secret_type: String,
    pub file: PathBuf,
    pub line: usize,
    pub severity: SecuritySeverity,
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
/// Security policy enforcer
#[derive(Debug)]
#[allow(dead_code)]
pub struct SecurityPolicyEnforcer {
    /// Active policies
    pub(super) policies: Vec<SecurityPolicy>,
    /// Policy evaluation engine
    pub(super) evaluator: PolicyEvaluator,
    /// Violation tracking
    pub(super) violations: Vec<PolicyViolation>,
}
impl SecurityPolicyEnforcer {
    pub(super) fn new() -> Self {
        Self {
            policies: Vec::new(),
            evaluator: PolicyEvaluator::new(),
            violations: Vec::new(),
        }
    }
    pub(super) fn check_compliance(
        &mut self,
        _audit_result: &SecurityAuditResult,
    ) -> Result<PolicyComplianceResult> {
        Ok(PolicyComplianceResult::default())
    }
}
#[derive(Debug)]
pub(super) struct SecurityReportGenerator;
impl SecurityReportGenerator {
    pub(super) fn new() -> Self {
        Self
    }
    pub(super) fn generate_report(
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyChainRisk {
    pub package: String,
    pub risk_type: String,
    pub severity: SecuritySeverity,
    pub description: String,
}
/// Audit scheduling options
#[derive(Debug, Clone)]
pub enum AuditSchedule {
    Daily,
    Weekly,
    Monthly,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    Always,
    Never,
    Custom(String),
}
