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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RequestStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ReservationPolicy {
    #[default]
    FirstCome,
    Priority,
    Fair,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ReservationType {
    Exclusive,
    #[default]
    Shared,
    Preemptible,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ResourcePriority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SchedulingStrategy {
    #[default]
    FIFO,
    Priority,
    ShortestJobFirst,
    RoundRobin,
}
