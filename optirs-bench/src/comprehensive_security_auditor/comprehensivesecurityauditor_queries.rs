//! # `ComprehensiveSecurityAuditor` - queries Methods
//!
//! This module contains method implementations for `ComprehensiveSecurityAuditor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::functions::{
    contains_keyword, parse_cargo_lock_packages, read_crate_license, scan_dependencies_offline,
};
use super::types::{
    AuditSchedule, ConfigSecurityResult, DependencyScanConfig, DependencyScanResult,
    DependencyScanner, EffortLevel, LicenseComplianceResult, LicenseViolation, MitigationStrategy,
    PolicyComplianceResult, RemediationPriority, RemediationSuggestion, RiskAssessment, RiskFactor,
    RiskLevel, SecretDetectionResult, SecurityAuditConfig, SecurityAuditResult, SecurityIssue,
    SecurityIssueType, SecurityPolicyEnforcer, SecurityReportGenerator, SecuritySeverity,
    StaticAnalysisResult, SupplyChainAnalysisResult, VulnerabilityDatabase,
    VulnerabilityDatabaseConfig,
};

use super::comprehensivesecurityauditor_type::ComprehensiveSecurityAuditor;

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
    /// RustSec advisory snapshot (see `embedded_advisory_snapshot`).
    /// Delegates to `scan_dependencies_offline` so this and
    /// `DependencyScanner::scan_dependencies` (the method actually invoked
    /// by [`Self::audit_project`]) never diverge into two different
    /// implementations.
    pub fn scan_dependencies_with_rustsec(
        &mut self,
        projectpath: &Path,
    ) -> Result<DependencyScanResult> {
        scan_dependencies_offline(projectpath, &self.dependency_scanner.config)
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
    pub(super) fn contains_potential_secret(&self, line: &str) -> bool {
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

    /// Check if line uses weak cryptography.
    ///
    /// Matches are bounded by identifier boundaries (via `contains_keyword`),
    /// not plain substring search (F14): a bare `line.contains("md5")` also
    /// fires on identifiers and URLs that merely contain the pattern -- e.g.
    /// `let cmd5_result = …` or a comment linking
    /// `https://example.com/md5sum-tool` -- neither of which uses MD5 as a
    /// cryptographic primitive.
    pub(super) fn uses_weak_crypto(&self, line: &str) -> bool {
        let weak_crypto_patterns = ["md5", "sha1", "des", "3des", "rc4", "md4"];

        let line_lower = line.to_lowercase();
        weak_crypto_patterns
            .iter()
            .any(|pattern| contains_keyword(&line_lower, pattern))
    }

    /// Find all Rust source files in project
    pub(super) fn find_rust_files(&self, projectpath: &Path) -> Result<Vec<PathBuf>> {
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
    pub(super) fn is_excluded_path(&self, path: &Path) -> bool {
        self.config.excluded_paths.iter().any(|excluded| {
            path.starts_with(excluded)
                || path
                    .components()
                    .any(|component| component.as_os_str() == excluded.as_os_str())
        })
    }

    /// Calculate overall security score based on audit results
    pub(super) fn calculate_security_score(&self, auditresult: &SecurityAuditResult) -> f64 {
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
}
