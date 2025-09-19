// Backup and Recovery Configuration
//
// This module provides backup strategies, recovery configurations, disaster recovery,
// and point-in-time recovery capabilities for event persistence systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::storage::{CloudProvider, CompressionConfig};

/// Backup and recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecoveryConfig {
    /// Backup settings
    pub backup: BackupConfig,
    /// Recovery settings
    pub recovery: RecoveryConfig,
    /// Disaster recovery
    pub disaster_recovery: DisasterRecoveryConfig,
    /// Point-in-time recovery
    pub point_in_time: PointInTimeRecoveryConfig,
}

impl Default for BackupRecoveryConfig {
    fn default() -> Self {
        Self {
            backup: BackupConfig::default(),
            recovery: RecoveryConfig::default(),
            disaster_recovery: DisasterRecoveryConfig::default(),
            point_in_time: PointInTimeRecoveryConfig::default(),
        }
    }
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable backups
    pub enabled: bool,
    /// Backup strategy
    pub strategy: BackupStrategy,
    /// Backup schedule
    pub schedule: BackupSchedule,
    /// Backup destinations
    pub destinations: Vec<BackupDestination>,
    /// Backup retention
    pub retention: BackupRetention,
    /// Backup encryption
    pub encryption: BackupEncryption,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: BackupStrategy::Incremental,
            schedule: BackupSchedule::default(),
            destinations: vec![BackupDestination::Local {
                path: "/var/backups/scirs2/events".to_string(),
            }],
            retention: BackupRetention::default(),
            encryption: BackupEncryption::default(),
        }
    }
}

/// Backup strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupStrategy {
    /// Full backup
    Full,
    /// Incremental backup
    Incremental,
    /// Differential backup
    Differential,
    /// Continuous backup
    Continuous,
    /// Snapshot backup
    Snapshot,
}

/// Backup schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSchedule {
    /// Full backup frequency
    pub full_backup_frequency: Duration,
    /// Incremental backup frequency
    pub incremental_frequency: Duration,
    /// Backup window
    pub backup_window: BackupWindow,
    /// Maximum backup duration
    pub max_duration: Duration,
}

impl Default for BackupSchedule {
    fn default() -> Self {
        Self {
            full_backup_frequency: Duration::from_secs(86400 * 7), // Weekly
            incremental_frequency: Duration::from_secs(3600),      // Hourly
            backup_window: BackupWindow {
                start_time: (2, 0), // 2:00 AM
                end_time: (4, 0),   // 4:00 AM
            },
            max_duration: Duration::from_secs(7200), // 2 hours
        }
    }
}

/// Backup window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupWindow {
    /// Start time (hour, minute)
    pub start_time: (u8, u8),
    /// End time (hour, minute)
    pub end_time: (u8, u8),
}

/// Backup destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupDestination {
    /// Local backup
    Local { path: String },
    /// Network backup
    Network {
        url: String,
        credentials: Option<String>,
    },
    /// Cloud backup
    Cloud {
        provider: CloudProvider,
        bucket: String,
        storage_class: Option<String>,
    },
    /// Tape backup
    Tape { library: String, pool: String },
}

/// Backup retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRetention {
    /// Daily backups to keep
    pub daily_count: usize,
    /// Weekly backups to keep
    pub weekly_count: usize,
    /// Monthly backups to keep
    pub monthly_count: usize,
    /// Yearly backups to keep
    pub yearly_count: usize,
    /// Maximum backup age
    pub max_age: Duration,
}

impl Default for BackupRetention {
    fn default() -> Self {
        Self {
            daily_count: 7,
            weekly_count: 4,
            monthly_count: 12,
            yearly_count: 5,
            max_age: Duration::from_secs(86400 * 365 * 7), // 7 years
        }
    }
}

/// Backup encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupEncryption {
    /// Enable encryption
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key management
    pub key_management: KeyManagement,
    /// Compression before encryption
    pub compress_before_encrypt: bool,
}

impl Default for BackupEncryption {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: EncryptionAlgorithm::AES256,
            key_management: KeyManagement::default(),
            compress_before_encrypt: true,
        }
    }
}

/// Encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES128,
    AES256,
    ChaCha20,
    Blowfish,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagement {
    /// Key source
    pub key_source: KeySource,
    /// Key rotation
    pub key_rotation: KeyRotation,
    /// Key escrow
    pub key_escrow: bool,
}

impl Default for KeyManagement {
    fn default() -> Self {
        Self {
            key_source: KeySource::Generated,
            key_rotation: KeyRotation::default(),
            key_escrow: false,
        }
    }
}

/// Key sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeySource {
    /// Generated key
    Generated,
    /// User-provided key
    UserProvided(String),
    /// Key management service
    KMS { provider: String, key_id: String },
    /// Hardware security module
    HSM { module: String, slot: u32 },
}

/// Key rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotation {
    /// Enable key rotation
    pub enabled: bool,
    /// Rotation frequency
    pub frequency: Duration,
    /// Keep old keys
    pub keep_old_keys: bool,
    /// Maximum key age
    pub max_key_age: Duration,
}

impl Default for KeyRotation {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(86400 * 90), // 90 days
            keep_old_keys: true,
            max_key_age: Duration::from_secs(86400 * 365), // 1 year
        }
    }
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Recovery strategies
    pub strategies: Vec<RecoveryStrategy>,
    /// Recovery validation
    pub validation: RecoveryValidation,
    /// Recovery testing
    pub testing: RecoveryTesting,
    /// Recovery automation
    pub automation: RecoveryAutomation,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            strategies: vec![
                RecoveryStrategy::BackupRestore,
                RecoveryStrategy::PointInTime,
            ],
            validation: RecoveryValidation::default(),
            testing: RecoveryTesting::default(),
            automation: RecoveryAutomation::default(),
        }
    }
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Restore from backup
    BackupRestore,
    /// Point-in-time recovery
    PointInTime,
    /// Hot standby
    HotStandby,
    /// Cold standby
    ColdStandby,
    /// Replication recovery
    Replication,
}

/// Recovery validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryValidation {
    /// Enable validation
    pub enabled: bool,
    /// Validation checks
    pub checks: Vec<ValidationCheck>,
    /// Validation timeout
    pub timeout: Duration,
}

impl Default for RecoveryValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            checks: vec![
                ValidationCheck::DataIntegrity,
                ValidationCheck::Completeness,
                ValidationCheck::Consistency,
            ],
            timeout: Duration::from_secs(1800), // 30 minutes
        }
    }
}

/// Validation checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCheck {
    /// Data integrity check
    DataIntegrity,
    /// Completeness check
    Completeness,
    /// Consistency check
    Consistency,
    /// Performance check
    Performance,
    /// Custom check
    Custom(String),
}

/// Recovery testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryTesting {
    /// Enable testing
    pub enabled: bool,
    /// Testing frequency
    pub frequency: Duration,
    /// Test scenarios
    pub scenarios: Vec<TestScenario>,
    /// Test environment
    pub environment: TestEnvironment,
}

impl Default for RecoveryTesting {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(86400 * 30), // Monthly
            scenarios: vec![
                TestScenario::FullRestore,
                TestScenario::PartialRestore,
                TestScenario::PointInTimeRestore,
            ],
            environment: TestEnvironment::Isolated,
        }
    }
}

/// Test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestScenario {
    /// Full system restore
    FullRestore,
    /// Partial restore
    PartialRestore,
    /// Point-in-time restore
    PointInTimeRestore,
    /// Disaster recovery
    DisasterRecovery,
    /// Custom scenario
    Custom(String),
}

/// Test environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestEnvironment {
    /// Isolated test environment
    Isolated,
    /// Production clone
    ProductionClone,
    /// Staging environment
    Staging,
    /// Custom environment
    Custom(String),
}

/// Recovery automation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAutomation {
    /// Enable automation
    pub enabled: bool,
    /// Automation triggers
    pub triggers: Vec<AutomationTrigger>,
    /// Automation actions
    pub actions: Vec<AutomationAction>,
    /// Notification settings
    pub notifications: NotificationSettings,
}

impl Default for RecoveryAutomation {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for safety
            triggers: Vec::new(),
            actions: Vec::new(),
            notifications: NotificationSettings::default(),
        }
    }
}

/// Automation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomationTrigger {
    /// Data corruption detected
    DataCorruption,
    /// Storage failure
    StorageFailure,
    /// Performance degradation
    PerformanceDegradation,
    /// Custom trigger
    Custom(String),
}

/// Automation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomationAction {
    /// Initiate backup restore
    BackupRestore,
    /// Switch to standby
    SwitchToStandby,
    /// Notify administrators
    NotifyAdministrators,
    /// Custom action
    Custom(String),
}

/// Notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// Enable notifications
    pub enabled: bool,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Notification levels
    pub levels: Vec<NotificationLevel>,
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: Vec::new(),
            levels: vec![NotificationLevel::Error, NotificationLevel::Critical],
        }
    }
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email {
        addresses: Vec<String>,
    },
    SMS {
        numbers: Vec<String>,
    },
    Slack {
        webhook: String,
        channel: String,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
}

/// Notification levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Disaster recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisasterRecoveryConfig {
    /// Enable disaster recovery
    pub enabled: bool,
    /// Recovery sites
    pub recovery_sites: Vec<RecoverySite>,
    /// Failover strategy
    pub failover_strategy: FailoverStrategy,
    /// Recovery objectives
    pub objectives: RecoveryObjectives,
}

impl Default for DisasterRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            recovery_sites: Vec::new(),
            failover_strategy: FailoverStrategy::Manual,
            objectives: RecoveryObjectives::default(),
        }
    }
}

/// Recovery site configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverySite {
    /// Site name
    pub name: String,
    /// Site location
    pub location: String,
    /// Site type
    pub site_type: RecoverySiteType,
    /// Site capacity
    pub capacity: SiteCapacity,
    /// Site status
    pub status: SiteStatus,
}

/// Recovery site types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoverySiteType {
    /// Hot site (ready to take over immediately)
    Hot,
    /// Warm site (can be made ready quickly)
    Warm,
    /// Cold site (requires setup time)
    Cold,
}

/// Site capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteCapacity {
    /// Storage capacity
    pub storage: u64,
    /// Compute capacity
    pub compute: ComputeCapacity,
    /// Network capacity
    pub network: NetworkCapacity,
}

/// Compute capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapacity {
    /// CPU cores
    pub cpu_cores: u32,
    /// Memory (bytes)
    pub memory: u64,
    /// GPU count
    pub gpu_count: u32,
}

/// Network capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapacity {
    /// Bandwidth (bits per second)
    pub bandwidth: u64,
    /// Latency (milliseconds)
    pub latency: u32,
    /// Availability percentage
    pub availability: f32,
}

/// Site status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SiteStatus {
    Active,
    Standby,
    Maintenance,
    Offline,
}

/// Failover strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverStrategy {
    /// Manual failover
    Manual,
    /// Automatic failover
    Automatic {
        triggers: Vec<FailoverTrigger>,
        timeout: Duration,
    },
    /// Semi-automatic failover
    SemiAutomatic {
        approval_required: bool,
        timeout: Duration,
    },
}

/// Failover triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverTrigger {
    /// Primary site unavailable
    PrimarySiteDown,
    /// Performance threshold breached
    PerformanceThreshold(f32),
    /// Data center failure
    DataCenterFailure,
    /// Network partition
    NetworkPartition,
}

/// Recovery objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryObjectives {
    /// Recovery Time Objective (RTO)
    pub rto: Duration,
    /// Recovery Point Objective (RPO)
    pub rpo: Duration,
    /// Mean Time To Recovery (MTTR)
    pub mttr: Duration,
    /// Target availability
    pub availability: f32,
}

impl Default for RecoveryObjectives {
    fn default() -> Self {
        Self {
            rto: Duration::from_secs(3600),  // 1 hour
            rpo: Duration::from_secs(300),   // 5 minutes
            mttr: Duration::from_secs(1800), // 30 minutes
            availability: 0.999,             // 99.9%
        }
    }
}

/// Point-in-time recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointInTimeRecoveryConfig {
    /// Enable point-in-time recovery
    pub enabled: bool,
    /// Transaction log configuration
    pub transaction_log: TransactionLogConfig,
    /// Snapshot configuration
    pub snapshots: SnapshotConfig,
    /// Recovery granularity
    pub granularity: RecoveryGranularity,
}

impl Default for PointInTimeRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            transaction_log: TransactionLogConfig::default(),
            snapshots: SnapshotConfig::default(),
            granularity: RecoveryGranularity::Second,
        }
    }
}

/// Transaction log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionLogConfig {
    /// Log retention period
    pub retention_period: Duration,
    /// Log rotation settings
    pub rotation: LogRotation,
    /// Log compression
    pub compression: CompressionConfig,
    /// Log replication
    pub replication: LogReplication,
}

impl Default for TransactionLogConfig {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(86400 * 30), // 30 days
            rotation: LogRotation::default(),
            compression: CompressionConfig::default(),
            replication: LogReplication::default(),
        }
    }
}

/// Log rotation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotation {
    /// Rotation strategy
    pub strategy: RotationStrategy,
    /// Maximum log size
    pub max_size: usize,
    /// Maximum log age
    pub max_age: Duration,
    /// Compression after rotation
    pub compress_after_rotation: bool,
}

impl Default for LogRotation {
    fn default() -> Self {
        Self {
            strategy: RotationStrategy::SizeBased,
            max_size: 100 * 1024 * 1024,         // 100MB
            max_age: Duration::from_secs(86400), // 1 day
            compress_after_rotation: true,
        }
    }
}

/// Rotation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationStrategy {
    /// Size-based rotation
    SizeBased,
    /// Time-based rotation
    TimeBased,
    /// Combined rotation
    Combined,
}

/// Log replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogReplication {
    /// Enable replication
    pub enabled: bool,
    /// Replication targets
    pub targets: Vec<ReplicationTarget>,
    /// Replication mode
    pub mode: ReplicationMode,
}

impl Default for LogReplication {
    fn default() -> Self {
        Self {
            enabled: false,
            targets: Vec::new(),
            mode: ReplicationMode::Asynchronous,
        }
    }
}

/// Replication targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationTarget {
    /// Target name
    pub name: String,
    /// Target location
    pub location: String,
    /// Target type
    pub target_type: ReplicationTargetType,
    /// Target priority
    pub priority: u32,
}

/// Replication target types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationTargetType {
    /// File system
    FileSystem,
    /// Database
    Database,
    /// Cloud storage
    CloudStorage,
    /// Remote server
    RemoteServer,
}

/// Replication modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationMode {
    /// Synchronous replication
    Synchronous,
    /// Asynchronous replication
    Asynchronous,
    /// Semi-synchronous replication
    SemiSynchronous,
}

/// Snapshot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    /// Snapshot frequency
    pub frequency: Duration,
    /// Snapshot retention
    pub retention: SnapshotRetention,
    /// Snapshot compression
    pub compression: CompressionConfig,
    /// Incremental snapshots
    pub incremental: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(3600), // 1 hour
            retention: SnapshotRetention::default(),
            compression: CompressionConfig::default(),
            incremental: true,
        }
    }
}

/// Snapshot retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRetention {
    /// Maximum number of snapshots
    pub max_count: usize,
    /// Maximum snapshot age
    pub max_age: Duration,
    /// Retention strategy
    pub strategy: SnapshotRetentionStrategy,
}

impl Default for SnapshotRetention {
    fn default() -> Self {
        Self {
            max_count: 168,                           // 7 days * 24 hours
            max_age: Duration::from_secs(86400 * 30), // 30 days
            strategy: SnapshotRetentionStrategy::TimeBase,
        }
    }
}

/// Snapshot retention strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotRetentionStrategy {
    /// Time-based retention
    TimeBase,
    /// Count-based retention
    CountBase,
    /// Hierarchical retention
    Hierarchical {
        hourly: usize,
        daily: usize,
        weekly: usize,
        monthly: usize,
    },
}

/// Recovery granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryGranularity {
    /// Second-level granularity
    Second,
    /// Minute-level granularity
    Minute,
    /// Hour-level granularity
    Hour,
    /// Transaction-level granularity
    Transaction,
}
