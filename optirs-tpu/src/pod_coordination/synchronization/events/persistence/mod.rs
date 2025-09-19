// Event Persistence Module - Modular Interface
//
// This module provides comprehensive event persistence capabilities for TPU synchronization
// including multiple storage backends, retention policies, data lifecycle management,
// backup and recovery mechanisms, and performance optimization for storage operations.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

// Import and re-export all submodules
pub mod archival;
pub mod backup;
pub mod monitoring;
pub mod performance;
pub mod retention;
pub mod storage;

// Re-export commonly used types from submodules
pub use storage::{
    CloudCredentials, CloudProvider, CompressionAlgorithm, CompressionConfig, CompressionLevel,
    ConsistencyLevel, DatabaseConnection, DatabaseType, DistributedConsistency, EvictionPolicy,
    FallbackStrategy, HybridBackend, IndexingStrategy, MemoryPersistence, PartitioningStrategy,
    ReplicationStrategy, RetryPolicy, RoutingStrategy, ShardingStrategy, StorageBackend,
    StorageBackendConfig, StorageFormat, StorageNode, TransactionSupport,
};

pub use retention::{
    CleanupPolicy, CleanupTarget, CleanupThresholds, ExpirationAction, LifecycleAction,
    LifecycleCondition, LifecycleManagement, LifecycleRule, RetentionCriteria, RetentionDuration,
    RetentionEnforcement, RetentionPolicies, RetentionPolicy, TransitionPolicy,
};

pub use backup::{
    BackupConfig, BackupDestination, BackupEncryption, BackupRecoveryConfig, BackupRetention,
    BackupSchedule, BackupStrategy, DisasterRecoveryConfig, EncryptionAlgorithm, KeyManagement,
    NotificationChannel, NotificationLevel, NotificationSettings, PointInTimeRecoveryConfig,
    RecoveryAutomation, RecoveryConfig, RecoveryStrategy, RecoveryTesting, RecoveryValidation,
};

pub use performance::{
    AsyncOperations, BatchProcessing, CacheInvalidation, CacheLayer, CacheWarming, CachingConfig,
    ConnectionPooling, CpuOptimization, HardwareOptimization, MemoryOptimization,
    NetworkOptimization, PerformanceOptimization, PoolConfig, StorageOptimization,
};

pub use monitoring::{
    AlertAction, AlertSeverity, CapacityAlert, CapacityForecasting, CapacityMetric,
    CapacityMonitoring, ErrorAlert, ErrorAnalysis, ErrorMonitoring, ErrorTracking, HealthAlert,
    HealthCheck, HealthMonitoring, PerformanceAlert, PerformanceMetric, PerformanceMonitoring,
    PersistenceMonitoring,
};

pub use archival::{
    ArchiveCriteria, ArchiveDestination, ArchiveFormat, ArchiveIndexType, ArchiveIndexing,
    ArchiveManagement, ArchivePolicy, ArchiveRetrieval, ArchiveStorage, IndexMaintenance,
    IndexSearch, RetrievalStrategy, SearchAlgorithm, VerificationMethod,
};

/// Errors that can occur during persistence operations
#[derive(Error, Debug)]
pub enum PersistenceError {
    #[error("Storage backend error: {0}")]
    StorageBackendError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    #[error("Retention policy error: {0}")]
    RetentionPolicyError(String),
    #[error("Backup operation failed: {0}")]
    BackupError(String),
    #[error("Recovery operation failed: {0}")]
    RecoveryError(String),
    #[error("Storage capacity exceeded: {0}")]
    CapacityExceeded(String),
    #[error("Archive operation failed: {0}")]
    ArchiveError(String),
    #[error("Index operation failed: {0}")]
    IndexError(String),
}

/// Result type for persistence operations
pub type PersistenceResult<T> = Result<T, PersistenceError>;

/// Main event persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPersistence {
    /// Storage backend configuration
    pub storage_backend: StorageBackendConfig,
    /// Retention policies
    pub retention_policies: RetentionPolicies,
    /// Backup and recovery settings
    pub backup_recovery: BackupRecoveryConfig,
    /// Performance optimization
    pub performance_optimization: PerformanceOptimization,
    /// Monitoring and health checks
    pub monitoring: PersistenceMonitoring,
    /// Archive management
    pub archive_management: ArchiveManagement,
}

impl Default for EventPersistence {
    fn default() -> Self {
        Self {
            storage_backend: StorageBackendConfig::default(),
            retention_policies: RetentionPolicies::default(),
            backup_recovery: BackupRecoveryConfig::default(),
            performance_optimization: PerformanceOptimization::default(),
            monitoring: PersistenceMonitoring::default(),
            archive_management: ArchiveManagement::default(),
        }
    }
}

impl EventPersistence {
    /// Create new event persistence configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create event persistence with custom storage backend
    pub fn with_storage_backend(mut self, storage_backend: StorageBackendConfig) -> Self {
        self.storage_backend = storage_backend;
        self
    }

    /// Create event persistence with custom retention policies
    pub fn with_retention_policies(mut self, retention_policies: RetentionPolicies) -> Self {
        self.retention_policies = retention_policies;
        self
    }

    /// Create event persistence with custom backup configuration
    pub fn with_backup_config(mut self, backup_recovery: BackupRecoveryConfig) -> Self {
        self.backup_recovery = backup_recovery;
        self
    }

    /// Create event persistence with custom performance optimization
    pub fn with_performance_optimization(
        mut self,
        performance_optimization: PerformanceOptimization,
    ) -> Self {
        self.performance_optimization = performance_optimization;
        self
    }

    /// Create event persistence with custom monitoring
    pub fn with_monitoring(mut self, monitoring: PersistenceMonitoring) -> Self {
        self.monitoring = monitoring;
        self
    }

    /// Create event persistence with custom archive management
    pub fn with_archive_management(mut self, archive_management: ArchiveManagement) -> Self {
        self.archive_management = archive_management;
        self
    }

    /// Validate the persistence configuration
    pub fn validate(&self) -> PersistenceResult<()> {
        // Validate storage backend configuration
        self.validate_storage_backend()?;

        // Validate retention policies
        self.validate_retention_policies()?;

        // Validate backup configuration
        self.validate_backup_config()?;

        // Validate performance settings
        self.validate_performance_settings()?;

        // Validate monitoring configuration
        self.validate_monitoring_config()?;

        // Validate archive management
        self.validate_archive_management()?;

        Ok(())
    }

    /// Get storage backend configuration
    pub fn storage_backend(&self) -> &StorageBackendConfig {
        &self.storage_backend
    }

    /// Get retention policies
    pub fn retention_policies(&self) -> &RetentionPolicies {
        &self.retention_policies
    }

    /// Get backup and recovery configuration
    pub fn backup_recovery(&self) -> &BackupRecoveryConfig {
        &self.backup_recovery
    }

    /// Get performance optimization settings
    pub fn performance_optimization(&self) -> &PerformanceOptimization {
        &self.performance_optimization
    }

    /// Get monitoring configuration
    pub fn monitoring(&self) -> &PersistenceMonitoring {
        &self.monitoring
    }

    /// Get archive management configuration
    pub fn archive_management(&self) -> &ArchiveManagement {
        &self.archive_management
    }

    // Private validation methods
    fn validate_storage_backend(&self) -> PersistenceResult<()> {
        // Validate storage backend configuration
        match &self.storage_backend.primary {
            StorageBackend::File { path, .. } => {
                if path.is_empty() {
                    return Err(PersistenceError::StorageBackendError(
                        "File storage path cannot be empty".to_string(),
                    ));
                }
            }
            StorageBackend::Database { connection, .. } => {
                if connection.connection_string.is_empty() {
                    return Err(PersistenceError::StorageBackendError(
                        "Database connection string cannot be empty".to_string(),
                    ));
                }
            }
            StorageBackend::Memory { capacity, .. } => {
                if *capacity == 0 {
                    return Err(PersistenceError::StorageBackendError(
                        "Memory storage capacity must be greater than zero".to_string(),
                    ));
                }
            }
            StorageBackend::Distributed { nodes, .. } => {
                if nodes.is_empty() {
                    return Err(PersistenceError::StorageBackendError(
                        "Distributed storage must have at least one node".to_string(),
                    ));
                }
            }
            StorageBackend::Cloud { bucket, .. } => {
                if bucket.is_empty() {
                    return Err(PersistenceError::StorageBackendError(
                        "Cloud storage bucket cannot be empty".to_string(),
                    ));
                }
            }
            StorageBackend::Hybrid { backends, .. } => {
                if backends.is_empty() {
                    return Err(PersistenceError::StorageBackendError(
                        "Hybrid storage must have at least one backend".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_retention_policies(&self) -> PersistenceResult<()> {
        // Validate default retention policy
        if self.retention_policies.default_policy.name.is_empty() {
            return Err(PersistenceError::RetentionPolicyError(
                "Default retention policy name cannot be empty".to_string(),
            ));
        }

        // Validate retention enforcement settings
        if self.retention_policies.enforcement.enabled
            && self.retention_policies.enforcement.batch_size == 0
        {
            return Err(PersistenceError::RetentionPolicyError(
                "Retention enforcement batch size must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_backup_config(&self) -> PersistenceResult<()> {
        // Validate backup configuration
        if self.backup_recovery.backup.enabled
            && self.backup_recovery.backup.destinations.is_empty()
        {
            return Err(PersistenceError::BackupError(
                "Backup destinations cannot be empty when backup is enabled".to_string(),
            ));
        }

        // Validate recovery configuration
        if self.backup_recovery.recovery.strategies.is_empty() {
            return Err(PersistenceError::RecoveryError(
                "Recovery strategies cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_performance_settings(&self) -> PersistenceResult<()> {
        // Validate caching configuration
        if self.performance_optimization.caching.enabled
            && self.performance_optimization.caching.layers.is_empty()
        {
            return Err(PersistenceError::StorageBackendError(
                "Cache layers cannot be empty when caching is enabled".to_string(),
            ));
        }

        // Validate batch processing settings
        if self.performance_optimization.batch_processing.enabled
            && self.performance_optimization.batch_processing.batch_size == 0
        {
            return Err(PersistenceError::StorageBackendError(
                "Batch processing size must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_monitoring_config(&self) -> PersistenceResult<()> {
        // Validate performance monitoring
        if self.monitoring.performance.enabled && self.monitoring.performance.metrics.is_empty() {
            return Err(PersistenceError::StorageBackendError(
                "Performance metrics cannot be empty when monitoring is enabled".to_string(),
            ));
        }

        // Validate health monitoring
        if self.monitoring.health.enabled && self.monitoring.health.checks.is_empty() {
            return Err(PersistenceError::StorageBackendError(
                "Health checks cannot be empty when health monitoring is enabled".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_archive_management(&self) -> PersistenceResult<()> {
        // Validate archive indexing
        if self.archive_management.indexing.enabled
            && self.archive_management.indexing.index_types.is_empty()
        {
            return Err(PersistenceError::ArchiveError(
                "Archive index types cannot be empty when indexing is enabled".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder pattern for EventPersistence configuration
pub struct EventPersistenceBuilder {
    config: EventPersistence,
}

impl EventPersistenceBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: EventPersistence::default(),
        }
    }

    /// Set storage backend
    pub fn storage_backend(mut self, storage_backend: StorageBackendConfig) -> Self {
        self.config.storage_backend = storage_backend;
        self
    }

    /// Set retention policies
    pub fn retention_policies(mut self, retention_policies: RetentionPolicies) -> Self {
        self.config.retention_policies = retention_policies;
        self
    }

    /// Set backup configuration
    pub fn backup_recovery(mut self, backup_recovery: BackupRecoveryConfig) -> Self {
        self.config.backup_recovery = backup_recovery;
        self
    }

    /// Set performance optimization
    pub fn performance_optimization(
        mut self,
        performance_optimization: PerformanceOptimization,
    ) -> Self {
        self.config.performance_optimization = performance_optimization;
        self
    }

    /// Set monitoring configuration
    pub fn monitoring(mut self, monitoring: PersistenceMonitoring) -> Self {
        self.config.monitoring = monitoring;
        self
    }

    /// Set archive management
    pub fn archive_management(mut self, archive_management: ArchiveManagement) -> Self {
        self.config.archive_management = archive_management;
        self
    }

    /// Build the configuration
    pub fn build(self) -> PersistenceResult<EventPersistence> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for EventPersistenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
