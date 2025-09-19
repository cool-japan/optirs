// Filter Storage and Caching
//
// This module provides filter storage mechanisms, caching strategies,
// and persistence capabilities for event filtering systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::rules::FilterRule;

/// Filter storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterStorage {
    /// Storage backend configuration
    pub backend: StorageBackend,
    /// Caching configuration
    pub caching: FilterCaching,
    /// Persistence configuration
    pub persistence: FilterPersistence,
    /// Synchronization settings
    pub synchronization: StorageSynchronization,
    /// Backup and recovery
    pub backup_recovery: BackupRecovery,
}

impl Default for FilterStorage {
    fn default() -> Self {
        Self {
            backend: StorageBackend::default(),
            caching: FilterCaching::default(),
            persistence: FilterPersistence::default(),
            synchronization: StorageSynchronization::default(),
            backup_recovery: BackupRecovery::default(),
        }
    }
}

/// Storage backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    /// In-memory storage
    Memory {
        capacity: usize,
        eviction_policy: EvictionPolicy,
    },
    /// File-based storage
    File {
        directory: String,
        format: StorageFormat,
        compression: bool,
    },
    /// Database storage
    Database {
        connection_string: String,
        table_name: String,
        schema: DatabaseSchema,
    },
    /// Distributed storage
    Distributed {
        nodes: Vec<StorageNode>,
        replication_factor: usize,
        consistency_level: ConsistencyLevel,
    },
    /// Cloud storage
    Cloud {
        provider: CloudProvider,
        bucket: String,
        region: String,
    },
    /// Hybrid storage
    Hybrid {
        primary: Box<StorageBackend>,
        secondary: Vec<StorageBackend>,
        routing_strategy: RoutingStrategy,
    },
}

impl Default for StorageBackend {
    fn default() -> Self {
        Self::Memory {
            capacity: 10000,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Eviction policies for memory storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
    /// Time-based eviction
    TTL(Duration),
}

/// Storage formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    /// JSON format
    Json,
    /// Binary format
    Binary,
    /// YAML format
    Yaml,
    /// MessagePack format
    MessagePack,
    /// Custom format
    Custom(String),
}

/// Database schema configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    /// Primary key field
    pub primary_key: String,
    /// Field definitions
    pub fields: HashMap<String, FieldDefinition>,
    /// Indexes
    pub indexes: Vec<IndexDefinition>,
}

/// Field definition for database schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field type
    pub field_type: FieldType,
    /// Is nullable
    pub nullable: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Maximum length (for strings)
    pub max_length: Option<usize>,
}

/// Field types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Integer,
    BigInteger,
    Float,
    Double,
    Boolean,
    DateTime,
    Binary,
    Json,
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    /// Index name
    pub name: String,
    /// Indexed fields
    pub fields: Vec<String>,
    /// Index type
    pub index_type: IndexType,
    /// Is unique index
    pub unique: bool,
}

/// Index types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    FullText,
    Spatial,
}

/// Storage node for distributed storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageNode {
    /// Node identifier
    pub id: String,
    /// Node address
    pub address: String,
    /// Node weight
    pub weight: f32,
    /// Node status
    pub status: NodeStatus,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Inactive,
    Maintenance,
    Failed,
}

/// Consistency levels for distributed storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency
    Strong,
    /// Eventual consistency
    Eventual,
    /// Causal consistency
    Causal,
    /// Session consistency
    Session,
}

/// Cloud storage providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS,
    GCP,
    Azure,
    Custom(String),
}

/// Routing strategies for hybrid storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route by rule type
    ByRuleType,
    /// Route by priority
    ByPriority,
    /// Route by size
    BySize,
    /// Round robin
    RoundRobin,
    /// Custom routing
    Custom(String),
}

/// Filter caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCaching {
    /// Enable caching
    pub enabled: bool,
    /// Cache layers
    pub layers: Vec<CacheLayer>,
    /// Cache policies
    pub policies: CachePolicies,
    /// Cache statistics
    pub statistics: CacheStatistics,
    /// Cache warming
    pub warming: CacheWarming,
}

impl Default for FilterCaching {
    fn default() -> Self {
        Self {
            enabled: true,
            layers: vec![
                CacheLayer::L1 {
                    size: 1000,
                    ttl: Duration::from_secs(300), // 5 minutes
                },
                CacheLayer::L2 {
                    size: 10000,
                    ttl: Duration::from_secs(3600), // 1 hour
                },
            ],
            policies: CachePolicies::default(),
            statistics: CacheStatistics::default(),
            warming: CacheWarming::default(),
        }
    }
}

/// Cache layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheLayer {
    /// L1 cache (fastest, smallest)
    L1 { size: usize, ttl: Duration },
    /// L2 cache (slower, larger)
    L2 { size: usize, ttl: Duration },
    /// L3 cache (slowest, largest)
    L3 { size: usize, ttl: Duration },
    /// Custom cache layer
    Custom {
        name: String,
        config: HashMap<String, String>,
    },
}

/// Cache policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicies {
    /// Cache key strategy
    pub key_strategy: CacheKeyStrategy,
    /// Invalidation policy
    pub invalidation: InvalidationPolicy,
    /// Replacement policy
    pub replacement: ReplacementPolicy,
    /// Write policy
    pub write_policy: WritePolicy,
}

impl Default for CachePolicies {
    fn default() -> Self {
        Self {
            key_strategy: CacheKeyStrategy::RuleId,
            invalidation: InvalidationPolicy::TTL,
            replacement: ReplacementPolicy::LRU,
            write_policy: WritePolicy::WriteThrough,
        }
    }
}

/// Cache key strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheKeyStrategy {
    /// Use rule ID as key
    RuleId,
    /// Use rule hash as key
    RuleHash,
    /// Use composite key
    Composite(Vec<String>),
    /// Custom key generation
    Custom(String),
}

/// Cache invalidation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidationPolicy {
    /// Time-to-live based
    TTL,
    /// Event-driven invalidation
    EventDriven,
    /// Manual invalidation
    Manual,
    /// Version-based invalidation
    VersionBased,
}

/// Cache replacement policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplacementPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random replacement
    Random,
    /// Adaptive replacement
    Adaptive,
}

/// Cache write policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WritePolicy {
    /// Write through to storage
    WriteThrough,
    /// Write back (lazy write)
    WriteBack,
    /// Write around (bypass cache)
    WriteAround,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Cache evictions
    pub evictions: u64,
    /// Cache invalidations
    pub invalidations: u64,
    /// Hit ratio
    pub hit_ratio: f64,
    /// Miss ratio
    pub miss_ratio: f64,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            invalidations: 0,
            hit_ratio: 0.0,
            miss_ratio: 0.0,
        }
    }
}

/// Cache warming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheWarming {
    /// Enable cache warming
    pub enabled: bool,
    /// Warming strategy
    pub strategy: WarmingStrategy,
    /// Warming schedule
    pub schedule: WarmingSchedule,
    /// Warming rules
    pub rules: Vec<WarmingRule>,
}

impl Default for CacheWarming {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: WarmingStrategy::Proactive,
            schedule: WarmingSchedule::OnStartup,
            rules: Vec::new(),
        }
    }
}

/// Cache warming strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmingStrategy {
    /// Proactive warming
    Proactive,
    /// Reactive warming
    Reactive,
    /// Predictive warming
    Predictive,
    /// Manual warming
    Manual,
}

/// Cache warming schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmingSchedule {
    /// Warm on startup
    OnStartup,
    /// Periodic warming
    Periodic(Duration),
    /// Event-driven warming
    EventDriven(Vec<String>),
    /// Manual warming
    Manual,
}

/// Cache warming rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingRule {
    /// Rule name
    pub name: String,
    /// Warming condition
    pub condition: WarmingCondition,
    /// Rules to warm
    pub rules_to_warm: Vec<String>,
    /// Priority
    pub priority: i32,
}

/// Cache warming conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmingCondition {
    /// Time-based condition
    TimeBased(Duration),
    /// Access pattern based
    AccessPattern(String),
    /// Rule usage based
    RuleUsage(f64), // Threshold
    /// Custom condition
    Custom(String),
}

/// Filter persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterPersistence {
    /// Enable persistence
    pub enabled: bool,
    /// Persistence mode
    pub mode: PersistenceMode,
    /// Persistence interval
    pub interval: Duration,
    /// Persistence format
    pub format: PersistenceFormat,
    /// Compression settings
    pub compression: CompressionSettings,
    /// Versioning
    pub versioning: VersioningSettings,
}

impl Default for FilterPersistence {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: PersistenceMode::Automatic,
            interval: Duration::from_secs(300), // 5 minutes
            format: PersistenceFormat::Json,
            compression: CompressionSettings::default(),
            versioning: VersioningSettings::default(),
        }
    }
}

/// Persistence modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceMode {
    /// Automatic persistence
    Automatic,
    /// Manual persistence
    Manual,
    /// Event-driven persistence
    EventDriven,
    /// Periodic persistence
    Periodic(Duration),
}

/// Persistence formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceFormat {
    Json,
    Binary,
    Yaml,
    MessagePack,
    Custom(String),
}

/// Compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: CompressionLevel,
    /// Compression threshold
    pub threshold: usize,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Gzip,
            level: CompressionLevel::Default,
            threshold: 1024, // 1KB
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zlib,
    Lz4,
    Zstd,
    Brotli,
}

/// Compression levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionLevel {
    Fastest,
    Default,
    Best,
    Custom(i32),
}

/// Versioning settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningSettings {
    /// Enable versioning
    pub enabled: bool,
    /// Versioning strategy
    pub strategy: VersioningStrategy,
    /// Maximum versions to keep
    pub max_versions: usize,
    /// Version retention period
    pub retention_period: Duration,
}

impl Default for VersioningSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: VersioningStrategy::Incremental,
            max_versions: 10,
            retention_period: Duration::from_secs(86400 * 30), // 30 days
        }
    }
}

/// Versioning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersioningStrategy {
    /// Incremental versioning
    Incremental,
    /// Semantic versioning
    Semantic,
    /// Timestamp-based versioning
    Timestamp,
    /// Hash-based versioning
    Hash,
}

/// Storage synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSynchronization {
    /// Synchronization mode
    pub mode: SynchronizationMode,
    /// Conflict resolution
    pub conflict_resolution: ConflictResolution,
    /// Merge strategies
    pub merge_strategies: MergeStrategies,
    /// Lock management
    pub lock_management: LockManagement,
}

impl Default for StorageSynchronization {
    fn default() -> Self {
        Self {
            mode: SynchronizationMode::Optimistic,
            conflict_resolution: ConflictResolution::LastWriteWins,
            merge_strategies: MergeStrategies::default(),
            lock_management: LockManagement::default(),
        }
    }
}

/// Synchronization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronizationMode {
    /// Optimistic locking
    Optimistic,
    /// Pessimistic locking
    Pessimistic,
    /// No locking
    None,
    /// Custom synchronization
    Custom(String),
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Last write wins
    LastWriteWins,
    /// First write wins
    FirstWriteWins,
    /// Manual resolution
    Manual,
    /// Merge conflicts
    Merge,
    /// Reject on conflict
    Reject,
}

/// Merge strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStrategies {
    /// Default merge strategy
    pub default_strategy: MergeStrategy,
    /// Field-specific strategies
    pub field_strategies: HashMap<String, MergeStrategy>,
}

impl Default for MergeStrategies {
    fn default() -> Self {
        Self {
            default_strategy: MergeStrategy::Union,
            field_strategies: HashMap::new(),
        }
    }
}

/// Merge strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Union of values
    Union,
    /// Intersection of values
    Intersection,
    /// Use newer value
    UseNewer,
    /// Use older value
    UseOlder,
    /// Custom merge logic
    Custom(String),
}

/// Lock management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockManagement {
    /// Lock timeout
    pub timeout: Duration,
    /// Lock granularity
    pub granularity: LockGranularity,
    /// Deadlock detection
    pub deadlock_detection: bool,
    /// Lock escalation
    pub escalation: LockEscalation,
}

impl Default for LockManagement {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            granularity: LockGranularity::Rule,
            deadlock_detection: true,
            escalation: LockEscalation::default(),
        }
    }
}

/// Lock granularity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LockGranularity {
    /// Global lock
    Global,
    /// Rule-level lock
    Rule,
    /// Field-level lock
    Field,
    /// Custom granularity
    Custom(String),
}

/// Lock escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockEscalation {
    /// Enable escalation
    pub enabled: bool,
    /// Escalation threshold
    pub threshold: usize,
    /// Escalation strategy
    pub strategy: EscalationStrategy,
}

impl Default for LockEscalation {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 10,
            strategy: EscalationStrategy::ToParent,
        }
    }
}

/// Lock escalation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationStrategy {
    /// Escalate to parent level
    ToParent,
    /// Escalate to global
    ToGlobal,
    /// Custom escalation
    Custom(String),
}

/// Backup and recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecovery {
    /// Backup configuration
    pub backup: BackupConfiguration,
    /// Recovery configuration
    pub recovery: RecoveryConfiguration,
    /// Disaster recovery
    pub disaster_recovery: DisasterRecovery,
}

impl Default for BackupRecovery {
    fn default() -> Self {
        Self {
            backup: BackupConfiguration::default(),
            recovery: RecoveryConfiguration::default(),
            disaster_recovery: DisasterRecovery::default(),
        }
    }
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfiguration {
    /// Enable backups
    pub enabled: bool,
    /// Backup frequency
    pub frequency: Duration,
    /// Backup destinations
    pub destinations: Vec<BackupDestination>,
    /// Backup retention
    pub retention: BackupRetention,
    /// Backup compression
    pub compression: bool,
    /// Backup encryption
    pub encryption: BackupEncryption,
}

impl Default for BackupConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency: Duration::from_secs(86400), // Daily
            destinations: Vec::new(),
            retention: BackupRetention::default(),
            compression: true,
            encryption: BackupEncryption::default(),
        }
    }
}

/// Backup destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupDestination {
    /// Local file system
    Local(String),
    /// Remote file system
    Remote {
        url: String,
        credentials: Option<String>,
    },
    /// Cloud storage
    Cloud {
        provider: CloudProvider,
        bucket: String,
    },
}

/// Backup retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRetention {
    /// Daily backups to keep
    pub daily: usize,
    /// Weekly backups to keep
    pub weekly: usize,
    /// Monthly backups to keep
    pub monthly: usize,
    /// Yearly backups to keep
    pub yearly: usize,
}

impl Default for BackupRetention {
    fn default() -> Self {
        Self {
            daily: 7,
            weekly: 4,
            monthly: 12,
            yearly: 3,
        }
    }
}

/// Backup encryption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupEncryption {
    /// Enable encryption
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key management
    pub key_management: KeyManagement,
}

impl Default for BackupEncryption {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: EncryptionAlgorithm::AES256,
            key_management: KeyManagement::default(),
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
    pub source: KeySource,
    /// Key rotation
    pub rotation: KeyRotation,
}

impl Default for KeyManagement {
    fn default() -> Self {
        Self {
            source: KeySource::Generated,
            rotation: KeyRotation::default(),
        }
    }
}

/// Key sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeySource {
    /// Generated key
    Generated,
    /// Environment variable
    Environment(String),
    /// File-based key
    File(String),
    /// Key management service
    KMS(String),
}

/// Key rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotation {
    /// Enable rotation
    pub enabled: bool,
    /// Rotation frequency
    pub frequency: Duration,
    /// Keep old keys
    pub keep_old_keys: bool,
}

impl Default for KeyRotation {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency: Duration::from_secs(86400 * 90), // 90 days
            keep_old_keys: true,
        }
    }
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfiguration {
    /// Recovery strategies
    pub strategies: Vec<RecoveryStrategy>,
    /// Recovery validation
    pub validation: RecoveryValidation,
    /// Recovery timeout
    pub timeout: Duration,
}

impl Default for RecoveryConfiguration {
    fn default() -> Self {
        Self {
            strategies: vec![RecoveryStrategy::FromBackup, RecoveryStrategy::FromReplica],
            validation: RecoveryValidation::default(),
            timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Recover from backup
    FromBackup,
    /// Recover from replica
    FromReplica,
    /// Point-in-time recovery
    PointInTime(Instant),
    /// Manual recovery
    Manual,
}

/// Recovery validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryValidation {
    /// Enable validation
    pub enabled: bool,
    /// Validation checks
    pub checks: Vec<ValidationCheck>,
}

impl Default for RecoveryValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            checks: vec![
                ValidationCheck::Integrity,
                ValidationCheck::Completeness,
                ValidationCheck::Consistency,
            ],
        }
    }
}

/// Validation checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCheck {
    /// Data integrity check
    Integrity,
    /// Data completeness check
    Completeness,
    /// Data consistency check
    Consistency,
    /// Custom validation
    Custom(String),
}

/// Disaster recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisasterRecovery {
    /// Enable disaster recovery
    pub enabled: bool,
    /// Recovery sites
    pub sites: Vec<RecoverySite>,
    /// Failover strategy
    pub failover: FailoverStrategy,
    /// Recovery objectives
    pub objectives: RecoveryObjectives,
}

impl Default for DisasterRecovery {
    fn default() -> Self {
        Self {
            enabled: false,
            sites: Vec::new(),
            failover: FailoverStrategy::Manual,
            objectives: RecoveryObjectives::default(),
        }
    }
}

/// Recovery site configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverySite {
    /// Site identifier
    pub id: String,
    /// Site location
    pub location: String,
    /// Site type
    pub site_type: SiteType,
    /// Site capacity
    pub capacity: SiteCapacity,
}

/// Site types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SiteType {
    /// Hot standby site
    Hot,
    /// Warm standby site
    Warm,
    /// Cold standby site
    Cold,
}

/// Site capacity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteCapacity {
    /// Storage capacity (bytes)
    pub storage: u64,
    /// Compute capacity
    pub compute: f64,
    /// Network bandwidth (bytes/sec)
    pub bandwidth: u64,
}

/// Failover strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverStrategy {
    /// Manual failover
    Manual,
    /// Automatic failover
    Automatic { triggers: Vec<FailoverTrigger> },
    /// Hybrid failover
    Hybrid {
        auto_triggers: Vec<FailoverTrigger>,
        manual_approval: bool,
    },
}

/// Failover triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverTrigger {
    /// Primary site unreachable
    PrimarySiteDown,
    /// Performance degradation
    PerformanceDegradation(f64),
    /// Error rate threshold
    ErrorRateThreshold(f64),
    /// Custom trigger
    Custom(String),
}

/// Recovery objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryObjectives {
    /// Recovery Time Objective (RTO)
    pub rto: Duration,
    /// Recovery Point Objective (RPO)
    pub rpo: Duration,
    /// Maximum tolerable downtime
    pub mtd: Duration,
}

impl Default for RecoveryObjectives {
    fn default() -> Self {
        Self {
            rto: Duration::from_secs(3600), // 1 hour
            rpo: Duration::from_secs(300),  // 5 minutes
            mtd: Duration::from_secs(7200), // 2 hours
        }
    }
}
