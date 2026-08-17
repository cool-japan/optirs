//! # `SecurityAuditConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SecurityAuditConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::path::PathBuf;
use std::time::Duration;

use super::types::{ReportFormat, SecurityAuditConfig, SecuritySeverity};

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
