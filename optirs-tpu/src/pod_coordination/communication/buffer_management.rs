// Buffer Management Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BufferPoolConfig {
    pub initial_size: usize,
    pub max_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferStatus {
    Available,
    InUse,
    Full,
}

impl Default for BufferStatus {
    fn default() -> Self {
        Self::Available
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GarbageCollector;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryAllocator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryManagementStrategy {
    Dynamic,
    Static,
    Hybrid,
}

impl Default for MemoryManagementStrategy {
    fn default() -> Self {
        Self::Dynamic
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageBufferPool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePriority {
    High,
    Normal,
    Low,
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolGrowthStrategy {
    Linear,
    Exponential,
    Fixed,
}

impl Default for PoolGrowthStrategy {
    fn default() -> Self {
        Self::Linear
    }
}
