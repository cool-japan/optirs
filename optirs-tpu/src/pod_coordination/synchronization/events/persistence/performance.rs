// Performance Optimization Configuration
//
// This module provides performance optimization settings including caching,
// connection pooling, batch processing, async operations, and hardware optimization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimization {
    /// Caching configuration
    pub caching: CachingConfig,
    /// Connection pooling
    pub connection_pooling: ConnectionPooling,
    /// Batch processing
    pub batch_processing: BatchProcessing,
    /// Async operations
    pub async_operations: AsyncOperations,
    /// Hardware optimization
    pub hardware_optimization: HardwareOptimization,
}

impl Default for PerformanceOptimization {
    fn default() -> Self {
        Self {
            caching: CachingConfig::default(),
            connection_pooling: ConnectionPooling::default(),
            batch_processing: BatchProcessing::default(),
            async_operations: AsyncOperations::default(),
            hardware_optimization: HardwareOptimization::default(),
        }
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache layers
    pub layers: Vec<CacheLayer>,
    /// Cache invalidation
    pub invalidation: CacheInvalidation,
    /// Cache warming
    pub warming: CacheWarming,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            layers: vec![
                CacheLayer::Memory {
                    size: 100 * 1024 * 1024, // 100MB
                    ttl: Duration::from_secs(3600),
                },
                CacheLayer::Disk {
                    path: "/tmp/scirs2_cache".to_string(),
                    size: 1024 * 1024 * 1024, // 1GB
                    ttl: Duration::from_secs(86400),
                },
            ],
            invalidation: CacheInvalidation::default(),
            warming: CacheWarming::default(),
        }
    }
}

/// Cache layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheLayer {
    /// Memory cache
    Memory { size: usize, ttl: Duration },
    /// Disk cache
    Disk {
        path: String,
        size: usize,
        ttl: Duration,
    },
    /// Distributed cache
    Distributed {
        nodes: Vec<String>,
        replication_factor: usize,
    },
}

/// Cache invalidation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInvalidation {
    /// Invalidation strategy
    pub strategy: InvalidationStrategy,
    /// Invalidation triggers
    pub triggers: Vec<InvalidationTrigger>,
    /// Batch invalidation
    pub batch_invalidation: bool,
}

impl Default for CacheInvalidation {
    fn default() -> Self {
        Self {
            strategy: InvalidationStrategy::TTL,
            triggers: vec![
                InvalidationTrigger::DataUpdate,
                InvalidationTrigger::SchemaChange,
            ],
            batch_invalidation: true,
        }
    }
}

/// Invalidation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidationStrategy {
    /// Time-to-live based
    TTL,
    /// Event-driven
    EventDriven,
    /// Manual
    Manual,
    /// Write-through
    WriteThrough,
    /// Write-behind
    WriteBehind,
}

/// Invalidation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidationTrigger {
    /// Data update
    DataUpdate,
    /// Schema change
    SchemaChange,
    /// Cache full
    CacheFull,
    /// Memory pressure
    MemoryPressure,
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
}

impl Default for CacheWarming {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: WarmingStrategy::Predictive,
            schedule: WarmingSchedule::OnStartup,
        }
    }
}

/// Warming strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmingStrategy {
    /// Predictive warming
    Predictive,
    /// Access pattern based
    AccessPattern,
    /// Time-based
    TimeBased,
    /// Manual
    Manual,
}

/// Warming schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmingSchedule {
    /// On startup
    OnStartup,
    /// Scheduled
    Scheduled(Duration),
    /// Continuous
    Continuous,
    /// Event-driven
    EventDriven(Vec<String>),
}

/// Connection pooling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPooling {
    /// Enable connection pooling
    pub enabled: bool,
    /// Pool configuration per backend
    pub pools: HashMap<String, PoolConfig>,
    /// Global pool settings
    pub global_settings: GlobalPoolSettings,
}

impl Default for ConnectionPooling {
    fn default() -> Self {
        Self {
            enabled: true,
            pools: HashMap::new(),
            global_settings: GlobalPoolSettings::default(),
        }
    }
}

/// Pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Minimum connections
    pub min_size: u32,
    /// Maximum connections
    pub max_size: u32,
    /// Connection timeout
    pub acquire_timeout: Duration,
    /// Idle timeout
    pub idle_timeout: Duration,
    /// Maximum lifetime
    pub max_lifetime: Duration,
}

/// Global pool settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalPoolSettings {
    /// Total connection limit
    pub total_connection_limit: u32,
    /// Pool monitoring
    pub monitoring: bool,
    /// Pool health checks
    pub health_checks: bool,
}

impl Default for GlobalPoolSettings {
    fn default() -> Self {
        Self {
            total_connection_limit: 1000,
            monitoring: true,
            health_checks: true,
        }
    }
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessing {
    /// Enable batch processing
    pub enabled: bool,
    /// Batch size
    pub batch_size: usize,
    /// Batch timeout
    pub batch_timeout: Duration,
    /// Parallel batches
    pub parallel_batches: usize,
    /// Batch optimization
    pub optimization: BatchOptimization,
}

impl Default for BatchProcessing {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_size: 1000,
            batch_timeout: Duration::from_secs(10),
            parallel_batches: 4,
            optimization: BatchOptimization::default(),
        }
    }
}

/// Batch optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOptimization {
    /// Dynamic batch sizing
    pub dynamic_sizing: bool,
    /// Size adjustment factor
    pub size_adjustment_factor: f32,
    /// Performance monitoring
    pub performance_monitoring: bool,
}

impl Default for BatchOptimization {
    fn default() -> Self {
        Self {
            dynamic_sizing: true,
            size_adjustment_factor: 1.2,
            performance_monitoring: true,
        }
    }
}

/// Async operations configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncOperations {
    /// Enable async operations
    pub enabled: bool,
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Queue size
    pub queue_size: usize,
    /// Timeout settings
    pub timeouts: AsyncTimeouts,
}

impl Default for AsyncOperations {
    fn default() -> Self {
        Self {
            enabled: true,
            thread_pool_size: num_cpus::get(),
            queue_size: 10000,
            timeouts: AsyncTimeouts::default(),
        }
    }
}

/// Async timeout settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncTimeouts {
    /// Operation timeout
    pub operation_timeout: Duration,
    /// Queue timeout
    pub queue_timeout: Duration,
    /// Shutdown timeout
    pub shutdown_timeout: Duration,
}

impl Default for AsyncTimeouts {
    fn default() -> Self {
        Self {
            operation_timeout: Duration::from_secs(30),
            queue_timeout: Duration::from_secs(5),
            shutdown_timeout: Duration::from_secs(60),
        }
    }
}

/// Hardware optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimization {
    /// CPU optimization
    pub cpu: CpuOptimization,
    /// Memory optimization
    pub memory: MemoryOptimization,
    /// Storage optimization
    pub storage: StorageOptimization,
    /// Network optimization
    pub network: NetworkOptimization,
}

impl Default for HardwareOptimization {
    fn default() -> Self {
        Self {
            cpu: CpuOptimization::default(),
            memory: MemoryOptimization::default(),
            storage: StorageOptimization::default(),
            network: NetworkOptimization::default(),
        }
    }
}

/// CPU optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimization {
    /// CPU affinity
    pub affinity: Option<Vec<usize>>,
    /// NUMA optimization
    pub numa_optimization: bool,
    /// Vector instructions
    pub vector_instructions: bool,
}

impl Default for CpuOptimization {
    fn default() -> Self {
        Self {
            affinity: None,
            numa_optimization: true,
            vector_instructions: true,
        }
    }
}

/// Memory optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Memory mapping
    pub memory_mapping: bool,
    /// Huge pages
    pub huge_pages: bool,
    /// Memory prefetching
    pub prefetching: bool,
    /// Memory alignment
    pub alignment: usize,
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        Self {
            memory_mapping: true,
            huge_pages: false,
            prefetching: true,
            alignment: 64, // Cache line size
        }
    }
}

/// Storage optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageOptimization {
    /// Direct I/O
    pub direct_io: bool,
    /// Async I/O
    pub async_io: bool,
    /// Read-ahead
    pub read_ahead: usize,
    /// Write-behind
    pub write_behind: bool,
}

impl Default for StorageOptimization {
    fn default() -> Self {
        Self {
            direct_io: false,
            async_io: true,
            read_ahead: 128 * 1024, // 128KB
            write_behind: true,
        }
    }
}

/// Network optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimization {
    /// TCP no delay
    pub tcp_no_delay: bool,
    /// Socket buffer sizes
    pub socket_buffer_size: Option<usize>,
    /// Connection keep-alive
    pub keep_alive: bool,
    /// Compression
    pub compression: bool,
}

impl Default for NetworkOptimization {
    fn default() -> Self {
        Self {
            tcp_no_delay: true,
            socket_buffer_size: Some(64 * 1024), // 64KB
            keep_alive: true,
            compression: true,
        }
    }
}
