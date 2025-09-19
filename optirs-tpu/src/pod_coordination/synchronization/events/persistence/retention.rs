// Retention Policies and Lifecycle Management
//
// This module provides retention policy management, lifecycle rules, and cleanup
// configurations for event persistence systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::storage::{ArchiveDestination, CompressionAlgorithm, IndexingStrategy, StorageBackend};

/// Retention policies configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicies {
    /// Default retention policy
    pub default_policy: RetentionPolicy,
    /// Event-specific policies
    pub event_policies: HashMap<String, RetentionPolicy>,
    /// Enforcement settings
    pub enforcement: RetentionEnforcement,
    /// Lifecycle management
    pub lifecycle: LifecycleManagement,
}

impl Default for RetentionPolicies {
    fn default() -> Self {
        Self {
            default_policy: RetentionPolicy::default(),
            event_policies: HashMap::new(),
            enforcement: RetentionEnforcement::default(),
            lifecycle: LifecycleManagement::default(),
        }
    }
}

/// Retention policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Policy name
    pub name: String,
    /// Retention duration
    pub duration: RetentionDuration,
    /// Retention criteria
    pub criteria: RetentionCriteria,
    /// Action on expiration
    pub expiration_action: ExpirationAction,
    /// Policy priority
    pub priority: u32,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            duration: RetentionDuration::Days(30),
            criteria: RetentionCriteria::Age,
            expiration_action: ExpirationAction::Delete,
            priority: 100,
        }
    }
}

/// Retention duration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionDuration {
    /// Retain for specified number of days
    Days(u32),
    /// Retain for specified number of weeks
    Weeks(u32),
    /// Retain for specified number of months
    Months(u32),
    /// Retain for specified number of years
    Years(u32),
    /// Retain indefinitely
    Indefinite,
    /// Custom duration
    Custom(Duration),
}

/// Retention criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionCriteria {
    /// Age-based retention
    Age,
    /// Size-based retention
    Size(usize),
    /// Count-based retention
    Count(usize),
    /// Access-based retention
    Access { last_accessed: Duration },
    /// Composite criteria
    Composite {
        operator: LogicalOperator,
        criteria: Vec<RetentionCriteria>,
    },
    /// Custom criteria
    Custom(String),
}

/// Logical operators for composite criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

/// Actions to take on expiration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpirationAction {
    /// Delete the data
    Delete,
    /// Archive the data
    Archive { destination: ArchiveDestination },
    /// Move to different storage tier
    Migrate { destination: StorageBackend },
    /// Compress the data
    Compress { algorithm: CompressionAlgorithm },
    /// Custom action
    Custom(String),
}

/// Retention enforcement settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionEnforcement {
    /// Enable enforcement
    pub enabled: bool,
    /// Enforcement frequency
    pub frequency: Duration,
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum processing time per batch
    pub max_processing_time: Duration,
    /// Enforcement reporting
    pub reporting: EnforcementReporting,
}

impl Default for RetentionEnforcement {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(3600), // 1 hour
            batch_size: 1000,
            max_processing_time: Duration::from_secs(300), // 5 minutes
            reporting: EnforcementReporting::default(),
        }
    }
}

/// Enforcement reporting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementReporting {
    /// Enable reporting
    pub enabled: bool,
    /// Report frequency
    pub frequency: Duration,
    /// Report format
    pub format: ReportFormat,
    /// Report destination
    pub destination: String,
}

impl Default for EnforcementReporting {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(86400), // Daily
            format: ReportFormat::Json,
            destination: "logs/retention_enforcement.log".to_string(),
        }
    }
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Yaml,
    Csv,
    Html,
    Xml,
}

/// Lifecycle management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleManagement {
    /// Lifecycle rules
    pub rules: Vec<LifecycleRule>,
    /// Transition policies
    pub transitions: Vec<TransitionPolicy>,
    /// Cleanup policies
    pub cleanup: CleanupPolicy,
}

impl Default for LifecycleManagement {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            transitions: Vec::new(),
            cleanup: CleanupPolicy::default(),
        }
    }
}

/// Lifecycle rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: LifecycleCondition,
    /// Rule action
    pub action: LifecycleAction,
    /// Rule status
    pub status: RuleStatus,
}

/// Lifecycle conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleCondition {
    /// Age condition
    Age(Duration),
    /// Size condition
    Size(usize),
    /// Access pattern condition
    AccessPattern {
        last_accessed: Duration,
        access_count: usize,
    },
    /// Storage tier condition
    StorageTier(String),
    /// Composite condition
    Composite {
        operator: LogicalOperator,
        conditions: Vec<LifecycleCondition>,
    },
}

/// Lifecycle actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleAction {
    /// Transition to different storage tier
    Transition { destination: StorageBackend },
    /// Archive data
    Archive { destination: ArchiveDestination },
    /// Delete data
    Delete,
    /// Compress data
    Compress { algorithm: CompressionAlgorithm },
    /// Index data
    Index { strategy: IndexingStrategy },
}

/// Rule status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleStatus {
    Active,
    Inactive,
    Testing,
}

/// Transition policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionPolicy {
    /// Policy name
    pub name: String,
    /// Source storage tier
    pub source: String,
    /// Destination storage tier
    pub destination: String,
    /// Transition criteria
    pub criteria: TransitionCriteria,
    /// Transition schedule
    pub schedule: TransitionSchedule,
}

/// Transition criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionCriteria {
    /// Minimum age
    pub min_age: Option<Duration>,
    /// Maximum size
    pub max_size: Option<usize>,
    /// Access frequency threshold
    pub access_frequency: Option<f32>,
    /// Cost optimization threshold
    pub cost_threshold: Option<f32>,
}

/// Transition schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionSchedule {
    /// Immediate transition
    Immediate,
    /// Scheduled transition
    Scheduled {
        frequency: Duration,
        batch_size: usize,
    },
    /// Event-driven transition
    EventDriven { events: Vec<String> },
}

/// Cleanup policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupPolicy {
    /// Enable cleanup
    pub enabled: bool,
    /// Cleanup frequency
    pub frequency: Duration,
    /// Cleanup targets
    pub targets: Vec<CleanupTarget>,
    /// Cleanup thresholds
    pub thresholds: CleanupThresholds,
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(86400), // Daily
            targets: vec![
                CleanupTarget::TempFiles,
                CleanupTarget::Logs,
                CleanupTarget::Cache,
            ],
            thresholds: CleanupThresholds::default(),
        }
    }
}

/// Cleanup targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupTarget {
    /// Temporary files
    TempFiles,
    /// Log files
    Logs,
    /// Cache files
    Cache,
    /// Backup files
    Backups,
    /// Archive files
    Archives,
    /// Custom target
    Custom(String),
}

/// Cleanup thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupThresholds {
    /// Maximum disk usage
    pub max_disk_usage: f32,
    /// Maximum file age
    pub max_file_age: Duration,
    /// Maximum file count
    pub max_file_count: usize,
    /// Maximum total size
    pub max_total_size: usize,
}

impl Default for CleanupThresholds {
    fn default() -> Self {
        Self {
            max_disk_usage: 0.8,                          // 80%
            max_file_age: Duration::from_secs(86400 * 7), // 7 days
            max_file_count: 10000,
            max_total_size: 10 * 1024 * 1024 * 1024, // 10GB
        }
    }
}
