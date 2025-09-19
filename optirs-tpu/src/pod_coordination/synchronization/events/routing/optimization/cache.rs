// Routing cache implementation

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Routing cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingCache {
    /// Cache enabled
    pub enabled: bool,
    /// Cache size
    pub size: usize,
    /// Cache TTL
    pub ttl: Duration,
    /// Cache strategy
    pub strategy: CacheStrategy,
}

impl Default for RoutingCache {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 1000,
            ttl: Duration::from_secs(300),
            strategy: CacheStrategy::LRU,
        }
    }
}

/// Cache strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// Least Recently Used
    LRU,
    /// First In First Out
    FIFO,
    /// Time-based expiration
    TTL,
    /// Custom strategy
    Custom(String),
}
