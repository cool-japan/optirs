// Resource Scheduling Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct AllocationMetrics {
    pub utilization_percent: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceReservation {
    pub device_id: DeviceId,
    pub reserved_percent: f64,
}

#[derive(Debug, Clone, Default)]
pub struct QueueMetrics {
    pub queue_depth: usize,
    pub avg_wait_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl Default for RequestStatus {
    fn default() -> Self {
        Self::Pending
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReservationPolicy {
    FirstCome,
    Priority,
    Fair,
}

impl Default for ReservationPolicy {
    fn default() -> Self {
        Self::FirstCome
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReservationType {
    Exclusive,
    Shared,
    Preemptible,
}

impl Default for ReservationType {
    fn default() -> Self {
        Self::Shared
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResourceAllocation {
    pub device_id: DeviceId,
    pub amount: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ResourcePoolConfig {
    pub max_resources: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourcePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for ResourcePriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_gb: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceScheduler {
    pub strategy: SchedulingStrategy,
    pub pool_config: ResourcePoolConfig,
}

#[derive(Debug, Clone, Default)]
pub struct SchedulingRequest {
    pub requirements: ResourceRequirements,
    pub priority: ResourcePriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    FIFO,
    Priority,
    ShortestJobFirst,
    RoundRobin,
}

impl Default for SchedulingStrategy {
    fn default() -> Self {
        Self::FIFO
    }
}
