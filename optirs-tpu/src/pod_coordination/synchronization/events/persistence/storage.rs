// Storage Backends and Configuration
//
// This module provides storage backend configurations including file storage,
// database storage, cloud storage, and distributed storage systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Storage backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageBackendConfig {
    /// Primary storage backend
    pub primary: StorageBackend,
    /// Secondary storage backends
    pub secondary: Vec<StorageBackend>,
    /// Replication strategy
    pub replication: ReplicationStrategy,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Transaction support
    pub transaction_support: TransactionSupport,
    /// Partitioning strategy
    pub partitioning: PartitioningStrategy,
}

impl Default for StorageBackendConfig {
    fn default() -> Self {
        Self {
            primary: StorageBackend::File {
                path: "/var/lib/scirs2/events".to_string(),
                format: StorageFormat::Binary,
                compression: CompressionConfig::default(),
            },
            secondary: Vec::new(),
            replication: ReplicationStrategy::Synchronous,
            consistency_level: ConsistencyLevel::Strong,
            transaction_support: TransactionSupport::default(),
            partitioning: PartitioningStrategy::default(),
        }
    }
}

/// Storage backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    /// File-based storage
    File {
        path: String,
        format: StorageFormat,
        compression: CompressionConfig,
    },
    /// Database storage
    Database {
        connection: DatabaseConnection,
        schema: DatabaseSchema,
        indexing: IndexingStrategy,
    },
    /// In-memory storage
    Memory {
        capacity: usize,
        persistence: MemoryPersistence,
        eviction_policy: EvictionPolicy,
    },
    /// Distributed storage
    Distributed {
        nodes: Vec<StorageNode>,
        consistency: DistributedConsistency,
        sharding: ShardingStrategy,
    },
    /// Cloud storage
    Cloud {
        provider: CloudProvider,
        bucket: String,
        credentials: CloudCredentials,
        region: Option<String>,
    },
    /// Hybrid storage
    Hybrid {
        backends: Vec<HybridBackend>,
        routing: RoutingStrategy,
        fallback: FallbackStrategy,
    },
}

/// Storage formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    /// JSON format
    Json,
    /// Binary format
    Binary,
    /// Protocol Buffers
    Protobuf,
    /// MessagePack
    MessagePack,
    /// Apache Avro
    Avro,
    /// Apache Parquet
    Parquet,
    /// Custom format
    Custom {
        name: String,
        serializer: String,
        deserializer: String,
    },
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: CompressionLevel,
    /// Compression threshold
    pub threshold: usize,
    /// Adaptive compression
    pub adaptive: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Zstd,
            level: CompressionLevel::Balanced,
            threshold: 1024, // 1KB
            adaptive: true,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Zstandard compression
    Zstd,
    /// LZ4 compression
    Lz4,
    /// Brotli compression
    Brotli,
    /// Snappy compression
    Snappy,
}

/// Compression levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionLevel {
    /// Fastest compression
    Fastest,
    /// Balanced compression
    Balanced,
    /// Best compression ratio
    Best,
    /// Custom level
    Custom(i32),
}

/// Database connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConnection {
    /// Database type
    pub database_type: DatabaseType,
    /// Connection string
    pub connection_string: String,
    /// Connection pool settings
    pub pool_settings: ConnectionPoolSettings,
    /// Connection timeout
    pub timeout: Duration,
    /// SSL/TLS configuration
    pub ssl_config: Option<SslConfig>,
}

/// Database types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseType {
    PostgreSQL,
    MySQL,
    SQLite,
    MongoDB,
    Redis,
    Cassandra,
    InfluxDB,
    TimescaleDB,
    ClickHouse,
}

/// Connection pool settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolSettings {
    /// Minimum connections
    pub min_connections: u32,
    /// Maximum connections
    pub max_connections: u32,
    /// Connection idle timeout
    pub idle_timeout: Duration,
    /// Connection lifetime
    pub max_lifetime: Duration,
}

/// SSL/TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslConfig {
    /// Enable SSL/TLS
    pub enabled: bool,
    /// Certificate path
    pub cert_path: Option<String>,
    /// Key path
    pub key_path: Option<String>,
    /// CA path
    pub ca_path: Option<String>,
    /// Verify certificates
    pub verify_certificates: bool,
}

/// Database schema configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    /// Table/collection name
    pub table_name: String,
    /// Schema definition
    pub schema_definition: SchemaDefinition,
    /// Indexes
    pub indexes: Vec<IndexDefinition>,
    /// Constraints
    pub constraints: Vec<ConstraintDefinition>,
}

/// Schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    /// Fields definition
    pub fields: HashMap<String, FieldDefinition>,
    /// Version
    pub version: u32,
    /// Migration scripts
    pub migrations: Vec<MigrationScript>,
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field type
    pub field_type: FieldType,
    /// Nullable
    pub nullable: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Constraints
    pub constraints: Vec<String>,
}

/// Field types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    Integer,
    BigInteger,
    Float,
    Double,
    String,
    Text,
    Boolean,
    DateTime,
    Binary,
    Json,
    Array(Box<FieldType>),
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    /// Index name
    pub name: String,
    /// Index type
    pub index_type: IndexType,
    /// Indexed fields
    pub fields: Vec<String>,
    /// Unique index
    pub unique: bool,
    /// Partial index condition
    pub condition: Option<String>,
}

/// Index types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    Gin,
    Gist,
    FullText,
    Spatial,
}

/// Constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDefinition {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint definition
    pub definition: String,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    PrimaryKey,
    ForeignKey,
    Unique,
    Check,
    NotNull,
}

/// Migration script
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationScript {
    /// Migration version
    pub version: u32,
    /// Up script
    pub up_script: String,
    /// Down script
    pub down_script: String,
    /// Description
    pub description: String,
}

/// Indexing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingStrategy {
    /// Primary indexes
    pub primary_indexes: Vec<IndexConfiguration>,
    /// Secondary indexes
    pub secondary_indexes: Vec<IndexConfiguration>,
    /// Full-text indexes
    pub fulltext_indexes: Vec<FullTextIndexConfiguration>,
    /// Spatial indexes
    pub spatial_indexes: Vec<SpatialIndexConfiguration>,
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfiguration {
    /// Index name
    pub name: String,
    /// Indexed fields
    pub fields: Vec<String>,
    /// Index options
    pub options: IndexOptions,
}

/// Index options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexOptions {
    /// Unique index
    pub unique: bool,
    /// Sparse index
    pub sparse: bool,
    /// Background creation
    pub background: bool,
    /// TTL (time to live)
    pub ttl: Option<Duration>,
}

/// Full-text index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextIndexConfiguration {
    /// Index name
    pub name: String,
    /// Text fields
    pub text_fields: Vec<String>,
    /// Language
    pub language: Option<String>,
    /// Weights
    pub weights: HashMap<String, f32>,
}

/// Spatial index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialIndexConfiguration {
    /// Index name
    pub name: String,
    /// Geometry field
    pub geometry_field: String,
    /// Spatial reference system
    pub srs: Option<String>,
}

/// Memory persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPersistence {
    /// Enable persistence to disk
    pub enabled: bool,
    /// Persistence interval
    pub interval: Duration,
    /// Persistence file
    pub file_path: String,
    /// Synchronous writes
    pub synchronous: bool,
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
    /// Size-based eviction
    SizeBased(usize),
}

/// Storage node for distributed storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageNode {
    /// Node ID
    pub node_id: String,
    /// Node address
    pub address: String,
    /// Node weight
    pub weight: f32,
    /// Node status
    pub status: NodeStatus,
    /// Node capabilities
    pub capabilities: NodeCapabilities,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Inactive,
    Maintenance,
    Failed,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Storage capacity
    pub storage_capacity: u64,
    /// Available storage
    pub available_storage: u64,
    /// Read IOPS
    pub read_iops: u32,
    /// Write IOPS
    pub write_iops: u32,
    /// Network bandwidth
    pub network_bandwidth: u64,
}

/// Distributed consistency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConsistency {
    /// Consistency model
    pub model: ConsistencyModel,
    /// Read quorum
    pub read_quorum: usize,
    /// Write quorum
    pub write_quorum: usize,
    /// Conflict resolution
    pub conflict_resolution: ConflictResolution,
}

/// Consistency models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyModel {
    StrongConsistency,
    EventualConsistency,
    CausalConsistency,
    MonotonicReadConsistency,
    MonotonicWriteConsistency,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    LastWriteWins,
    FirstWriteWins,
    Merge,
    Manual,
    Custom(String),
}

/// Sharding strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingStrategy {
    /// Sharding method
    pub method: ShardingMethod,
    /// Number of shards
    pub shard_count: usize,
    /// Shard key
    pub shard_key: String,
    /// Rebalancing policy
    pub rebalancing: RebalancingPolicy,
}

/// Sharding methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingMethod {
    Hash,
    Range,
    Directory,
    Consistent,
}

/// Rebalancing policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingPolicy {
    /// Enable automatic rebalancing
    pub enabled: bool,
    /// Rebalancing threshold
    pub threshold: f32,
    /// Rebalancing frequency
    pub frequency: Duration,
    /// Maximum concurrent transfers
    pub max_concurrent_transfers: usize,
}

/// Cloud provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS {
        service: AwsService,
        region: String,
    },
    GCP {
        service: GcpService,
        project: String,
    },
    Azure {
        service: AzureService,
        subscription: String,
    },
    Custom {
        name: String,
        endpoint: String,
    },
}

/// AWS services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AwsService {
    S3,
    DynamoDB,
    RDS,
    DocumentDB,
}

/// GCP services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GcpService {
    CloudStorage,
    Firestore,
    CloudSQL,
    BigQuery,
}

/// Azure services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AzureService {
    BlobStorage,
    CosmosDB,
    SQLDatabase,
}

/// Cloud credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudCredentials {
    AccessKey {
        access_key_id: String,
        secret_access_key: String,
    },
    ServiceAccount {
        key_file: String,
    },
    IAMRole {
        role_arn: String,
    },
    Default,
}

/// Hybrid backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridBackend {
    /// Backend
    pub backend: StorageBackend,
    /// Priority
    pub priority: u32,
    /// Selection criteria
    pub criteria: SelectionCriteria,
}

/// Selection criteria for hybrid backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Data size threshold
    pub size_threshold: Option<usize>,
    /// Access frequency threshold
    pub frequency_threshold: Option<f32>,
    /// Retention period threshold
    pub retention_threshold: Option<Duration>,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum latency
    pub max_latency: Duration,
    /// Minimum throughput
    pub min_throughput: u64,
    /// Availability requirement
    pub availability: f32,
    /// Durability requirement
    pub durability: f32,
}

/// Routing strategy for hybrid storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route by data characteristics
    DataBased,
    /// Route by performance requirements
    PerformanceBased,
    /// Route by cost optimization
    CostOptimized,
    /// Round-robin routing
    RoundRobin,
    /// Custom routing logic
    Custom(String),
}

/// Fallback strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackStrategy {
    /// Enable fallback
    pub enabled: bool,
    /// Fallback backends in order
    pub fallback_order: Vec<usize>,
    /// Fallback timeout
    pub timeout: Duration,
    /// Retry policy
    pub retry_policy: RetryPolicy,
}

/// Retry policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retries
    pub max_retries: usize,
    /// Base delay
    pub base_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f32,
    /// Maximum delay
    pub max_delay: Duration,
    /// Jitter
    pub jitter: bool,
}

/// Replication strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// No replication
    None,
    /// Synchronous replication
    Synchronous,
    /// Asynchronous replication
    Asynchronous,
    /// Semi-synchronous replication
    SemiSynchronous,
    /// Master-slave replication
    MasterSlave,
    /// Multi-master replication
    MultiMaster,
}

/// Consistency level
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
    /// Bounded staleness
    BoundedStaleness(Duration),
}

/// Transaction support configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSupport {
    /// Enable transactions
    pub enabled: bool,
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Transaction timeout
    pub timeout: Duration,
    /// Deadlock detection
    pub deadlock_detection: bool,
    /// Auto-retry on conflicts
    pub auto_retry: bool,
}

impl Default for TransactionSupport {
    fn default() -> Self {
        Self {
            enabled: true,
            isolation_level: IsolationLevel::ReadCommitted,
            timeout: Duration::from_secs(30),
            deadlock_detection: true,
            auto_retry: true,
        }
    }
}

/// Transaction isolation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Partitioning strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningStrategy {
    /// Partitioning method
    pub method: PartitioningMethod,
    /// Partition key
    pub partition_key: String,
    /// Number of partitions
    pub partition_count: usize,
    /// Partition size limit
    pub size_limit: Option<usize>,
    /// Auto-partitioning
    pub auto_partitioning: bool,
}

impl Default for PartitioningStrategy {
    fn default() -> Self {
        Self {
            method: PartitioningMethod::TimeRange,
            partition_key: "timestamp".to_string(),
            partition_count: 12,                  // Monthly partitions
            size_limit: Some(1024 * 1024 * 1024), // 1GB
            auto_partitioning: true,
        }
    }
}

/// Partitioning methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitioningMethod {
    /// Time-based partitioning
    TimeRange,
    /// Hash-based partitioning
    Hash,
    /// Range-based partitioning
    Range,
    /// List-based partitioning
    List,
    /// Composite partitioning
    Composite(Vec<PartitioningMethod>),
}
