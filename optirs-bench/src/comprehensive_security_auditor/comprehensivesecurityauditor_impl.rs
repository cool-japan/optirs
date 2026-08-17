//! # `ComprehensiveSecurityAuditor` - analyze_supply_chain_group Methods
//!
//! This module contains method implementations for `ComprehensiveSecurityAuditor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use std::path::Path;

use super::functions::{high_entropy_runs, parse_cargo_toml_dependencies, shannon_entropy};
use super::types::{
    ConfigSecurityIssue, ConfigSecurityResult, DetectedSecret, SecretDetectionResult,
    SecuritySeverity, SupplyChainAnalysisResult, SupplyChainRisk,
};

use super::comprehensivesecurityauditor_type::ComprehensiveSecurityAuditor;

impl ComprehensiveSecurityAuditor {
    /// Offline supply-chain risk analysis: flags git dependencies
    /// (bypassing crates.io's publish trail), wildcard/unconstrained
    /// version requirements, and the presence of a `build.rs` (arbitrary
    /// code execution at build time). All three are genuinely computable
    /// without network access.
    pub(super) fn analyze_supply_chain(
        &self,
        projectpath: &Path,
    ) -> Result<SupplyChainAnalysisResult> {
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
    pub(super) fn detect_secrets(&self, projectpath: &Path) -> Result<SecretDetectionResult> {
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
    pub(super) fn check_config_security(&self, projectpath: &Path) -> Result<ConfigSecurityResult> {
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
