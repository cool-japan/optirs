// Buffer Management Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BufferPoolConfig {
    pub initial_size: usize,
    pub max_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum BufferStatus {
    #[default]
    Available,
    InUse,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GarbageCollector;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryAllocator;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MemoryManagementStrategy {
    #[default]
    Dynamic,
    Static,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageBufferPool;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MessagePriority {
    High,
    #[default]
    Normal,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PoolGrowthStrategy {
    #[default]
    Linear,
    Exponential,
    Fixed,
}
