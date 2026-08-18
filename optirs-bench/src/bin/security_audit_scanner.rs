//! Security Audit Scanner CLI Tool
//!
//! A command-line front-end over
//! [`optirs_bench::comprehensive_security_auditor::ComprehensiveSecurityAuditor`]
//! -- the crate's real, offline-capable security engine: RustSec-range
//! vulnerability matching against an embedded advisory snapshot, license
//! lookups from the local Cargo registry cache, static-analysis pattern
//! checks (unsafe code, hardcoded secrets, command injection, weak crypto),
//! Shannon-entropy secret detection, and supply-chain risk analysis (git
//! dependencies, wildcard versions, build scripts).
//!
//! This binary previously carried its own, entirely separate implementation
//! that faked all of the above: a two-crate hardcoded "vulnerability
//! database" whose `version_is_vulnerable` always returned `false`
//! ("Most dependencies are likely not vulnerable"), a license checker that
//! reported the same invented `"example-gpl-dep"` GPL violation on every
//! run regardless of the project scanned, and substring-only crypto/secret
//! heuristics prone to false positives on ordinary identifiers and URLs
//! (optirs-bench findings F7/F11/F12/F14/F15). It now runs the same real
//! engine as the rest of the crate instead of maintaining a second,
//! divergent copy of this logic.

use clap::{Arg, ArgMatches, Command};
use optirs_bench::comprehensive_security_auditor::{
    ComprehensiveSecurityAuditor, ReportFormat, SecurityAuditConfig, SecurityAuditResult,
    SecuritySeverity,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn main() {
    let matches = build_cli().get_matches();
    if let Err(e) = run(&matches) {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

fn build_cli() -> Command {
    Command::new("Security Audit Scanner")
        .version("0.1.0")
        .about("Comprehensive, offline security auditing for Rust projects")
        .arg(
            Arg::new("project-path")
                .short('p')
                .long("project")
                .value_name("PATH")
                .help("Path to the Rust project to audit")
                .required(false)
                .default_value("."),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for the audit report (stdout if omitted)")
                .required(false),
        )
        .arg(
            Arg::new("format")
                .short('f')
                .long("format")
                .value_name("FORMAT")
                .help("Output format")
                .default_value("markdown")
                .value_parser(["json", "yaml", "html", "markdown"]),
        )
        .arg(
            Arg::new("severity")
                .short('s')
                .long("severity")
                .value_name("LEVEL")
                .help("Minimum severity level that causes a non-zero exit code")
                .default_value("low")
                .value_parser(["info", "low", "medium", "high", "critical"]),
        )
        .arg(
            Arg::new("scan-deps")
                .long("scan-dependencies")
                .help("Enable dependency vulnerability scanning")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("scan-secrets")
                .long("scan-secrets")
                .help("Enable secret detection scanning")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("scan-code")
                .long("scan-code")
                .help("Enable static code analysis")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("check-licenses")
                .long("check-licenses")
                .help("Enable license compliance checking")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("all")
                .short('a')
                .long("all")
                .help(
                    "Enable all scanning options, including supply-chain analysis and \
                       configuration security checks (which have no individual flag)",
                )
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("exclude")
                .long("exclude")
                .value_name("PATHS")
                .help(
                    "Extra excluded path components, comma-separated (target/.git/node_modules \
                       are always excluded)",
                )
                .required(false),
        )
}

fn run(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let projectpath = Path::new(
        matches
            .get_one::<String>("project-path")
            .ok_or("project path is required")?,
    );
    let verbose = matches.get_flag("verbose");
    let min_severity = parse_severity(
        matches
            .get_one::<String>("severity")
            .ok_or("severity is required")?,
    )?;

    let config = build_audit_config(matches, min_severity)?;

    if verbose {
        println!(
            "Running security audit for project: {}",
            projectpath.display()
        );
    }

    let mut auditor = ComprehensiveSecurityAuditor::new(config);
    let auditresult = auditor.audit_project(projectpath)?;
    let report = auditor.generate_report(&auditresult)?;

    if let Some(output_file) = matches.get_one::<String>("output") {
        fs::write(output_file, &report)?;
        if verbose {
            println!("Audit report written to: {output_file}");
        }
    } else {
        println!("{report}");
    }

    if verbose {
        println!(
            "Security score: {:.2}/1.0; overall risk: {:?}",
            auditresult.security_score, auditresult.risk_assessment.overall_risk
        );
    }

    let exit_code = exit_code_for(min_severity, &auditresult);
    if exit_code != 0 {
        process::exit(exit_code);
    }
    Ok(())
}

/// Build a [`SecurityAuditConfig`] from CLI flags, starting from
/// [`SecurityAuditConfig::default`] for every field this CLI does not expose
/// (db update cadence, max audit time, trusted sources, custom rules, alert
/// webhook). Nothing runs unless explicitly requested via a scan flag or
/// `--all`, matching this CLI's historical behavior.
fn build_audit_config(
    matches: &ArgMatches,
    min_severity: SecuritySeverity,
) -> Result<SecurityAuditConfig, Box<dyn std::error::Error>> {
    let all = matches.get_flag("all");
    let mut config = SecurityAuditConfig {
        enable_dependency_scanning: all || matches.get_flag("scan-deps"),
        enable_static_analysis: all || matches.get_flag("scan-code"),
        enable_license_compliance: all || matches.get_flag("check-licenses"),
        enable_secret_detection: all || matches.get_flag("scan-secrets"),
        // No dedicated CLI flags for these two; only `--all` turns them on,
        // since both are engine features this CLI's original flag set never
        // exposed on their own.
        enable_supply_chain_analysis: all,
        enable_config_security: all,
        alert_threshold: min_severity,
        report_format: parse_report_format(
            matches
                .get_one::<String>("format")
                .ok_or("format is required")?,
        )?,
        ..SecurityAuditConfig::default()
    };

    if let Some(exclude_str) = matches.get_one::<String>("exclude") {
        config.excluded_paths.extend(
            exclude_str
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
        );
    }

    Ok(config)
}

fn parse_severity(text: &str) -> Result<SecuritySeverity, Box<dyn std::error::Error>> {
    match text.to_lowercase().as_str() {
        "info" => Ok(SecuritySeverity::Info),
        "low" => Ok(SecuritySeverity::Low),
        "medium" => Ok(SecuritySeverity::Medium),
        "high" => Ok(SecuritySeverity::High),
        "critical" => Ok(SecuritySeverity::Critical),
        other => Err(format!("invalid severity level: {other}").into()),
    }
}

fn parse_report_format(text: &str) -> Result<ReportFormat, Box<dyn std::error::Error>> {
    match text.to_lowercase().as_str() {
        "json" => Ok(ReportFormat::Json),
        "yaml" => Ok(ReportFormat::Yaml),
        "html" => Ok(ReportFormat::Html),
        "markdown" => Ok(ReportFormat::Markdown),
        other => Err(format!("invalid report format: {other}").into()),
    }
}

/// Real maximum severity found across every part of the audit result that
/// carries one: dependency vulnerabilities, static-analysis issues,
/// detected secrets, configuration-security issues, policy violations, and
/// supply-chain risks. `LicenseViolation` carries no severity field, so any
/// license violation contributes `Medium` (matching this CLI's historical
/// treatment of license issues as medium-severity findings).
fn max_severity_found(result: &SecurityAuditResult) -> Option<SecuritySeverity> {
    let mut found: Option<SecuritySeverity> = None;
    let bump = |s: SecuritySeverity, found: &mut Option<SecuritySeverity>| {
        *found = Some(found.map_or(s, |current| current.max(s)));
    };

    for vuln in &result.dependency_results.vulnerable_dependencies {
        bump(vuln.severity, &mut found);
    }
    for risk in &result.dependency_results.supply_chain_risks {
        bump(risk.severity, &mut found);
    }
    for issue in &result.static_analysis_results.security_issues {
        bump(issue.severity, &mut found);
    }
    for secret in &result.secret_detection_results.secrets_found {
        bump(secret.severity, &mut found);
    }
    for issue in &result.config_security_results.issues {
        bump(issue.severity, &mut found);
    }
    for violation in &result.policy_compliance_results.violations {
        bump(violation.severity, &mut found);
    }
    if !result.license_compliance_results.violations.is_empty() {
        bump(SecuritySeverity::Medium, &mut found);
    }

    found
}

/// Exit with a non-zero status only when a real finding at or above
/// `min_severity` was recorded -- never unconditionally, and never derived
/// from a fabricated count.
fn exit_code_for(min_severity: SecuritySeverity, result: &SecurityAuditResult) -> i32 {
    match max_severity_found(result) {
        Some(actual) if actual >= min_severity => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    /// Regression (F7/F11/F12): the old bin's `get_known_vulnerabilities()`
    /// only ever contained hardcoded entries for "serde"/"tokio", and even
    /// for those `version_is_vulnerable` always returned `false` -- no
    /// project could ever be flagged. This audits a temp project declaring a
    /// dependency the real embedded advisory snapshot knows is vulnerable at
    /// the declared version, through the exact same `ComprehensiveSecurityAuditor`
    /// entry point this binary calls, and asserts it is actually flagged.
    #[test]
    fn test_audit_project_flags_a_real_known_vulnerability() {
        let dir = std::env::temp_dir().join(format!(
            "optirs_bench_security_scanner_test_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp project dir");
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"scratch\"\nversion = \"0.1.0\"\n\n[dependencies]\n\
             chrono = \"0.4.19\"\n",
        )
        .expect("write Cargo.toml");

        let mut config = SecurityAuditConfig {
            enable_dependency_scanning: true,
            enable_static_analysis: false,
            enable_license_compliance: false,
            enable_supply_chain_analysis: false,
            enable_secret_detection: false,
            enable_config_security: false,
            enable_auto_remediation: false,
            ..SecurityAuditConfig::default()
        };
        config.excluded_paths.clear();

        let mut auditor = ComprehensiveSecurityAuditor::new(config);
        let result = auditor
            .audit_project(&dir)
            .expect("audit_project succeeds on a well-formed temp project");

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(result.dependency_results.total_dependencies, 1);
        assert!(
            result
                .dependency_results
                .vulnerable_dependencies
                .iter()
                .any(|v| v.name == "chrono"
                    && v.cve_ids.iter().any(|id| id == "RUSTSEC-2020-0159")),
            "chrono 0.4.19 must be flagged via the real embedded advisory snapshot, got: {:?}",
            result.dependency_results.vulnerable_dependencies
        );
        assert!(exit_code_for(SecuritySeverity::Low, &result) != 0);
        // A stricter minimum than the actual Medium-severity finding must not fire.
        assert_eq!(exit_code_for(SecuritySeverity::Critical, &result), 0);
    }

    #[test]
    fn test_exit_code_is_zero_on_a_clean_result() {
        let clean = SecurityAuditResult {
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(0),
            security_score: 1.0,
            dependency_results: Default::default(),
            static_analysis_results: Default::default(),
            license_compliance_results: Default::default(),
            supply_chain_results: Default::default(),
            secret_detection_results: Default::default(),
            config_security_results: Default::default(),
            policy_compliance_results: Default::default(),
            remediation_suggestions: Vec::new(),
            risk_assessment: Default::default(),
        };
        assert_eq!(exit_code_for(SecuritySeverity::Info, &clean), 0);
        assert_eq!(max_severity_found(&clean), None);
    }

    #[test]
    fn test_parse_severity_and_format_reject_garbage() {
        assert!(parse_severity("not-a-level").is_err());
        assert!(parse_report_format("not-a-format").is_err());
        assert_eq!(parse_severity("HIGH").unwrap(), SecuritySeverity::High);
    }
}
