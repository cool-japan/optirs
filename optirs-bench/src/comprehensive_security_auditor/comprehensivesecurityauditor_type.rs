//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    DependencyScanner, SecurityAuditConfig, SecurityAuditResult, SecurityPolicyEnforcer,
    SecurityReportGenerator, VulnerabilityDatabase,
};

/// Main security audit engine
#[derive(Debug)]
pub struct ComprehensiveSecurityAuditor {
    /// Audit configuration
    pub(super) config: SecurityAuditConfig,
    /// Dependency scanner
    pub(super) dependency_scanner: DependencyScanner,
    /// Vulnerability database
    pub(super) vulnerability_db: VulnerabilityDatabase,
    /// Policy enforcer
    pub(super) policy_enforcer: SecurityPolicyEnforcer,
    /// Report generator
    pub(super) report_generator: SecurityReportGenerator,
    /// Audit history
    pub(super) audit_history: Vec<SecurityAuditResult>,
}
