// Load Balancing Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceAvailability {
    Available,
    Busy,
    Offline,
}

impl Default for DeviceAvailability {
    fn default() -> Self {
        Self::Available
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceLoad {
    pub device_id: DeviceId,
    pub utilization: f64,
}

#[derive(Debug, Clone, Default)]
pub struct LoadBalancer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastLoaded,
    Random,
}

impl Default for LoadBalancingAlgorithm {
    fn default() -> Self {
        Self::RoundRobin
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadSnapshot {
    pub loads: HashMap<DeviceId, f64>,
}

#[derive(Debug, Clone, Default)]
pub struct PodLoadBalancer;

#[derive(Debug, Clone)]
pub enum RebalancingAction {
    Migrate,
    Redistribute,
    NoOp,
}

#[derive(Debug, Clone)]
pub enum RebalancingPolicy {
    Periodic,
    Threshold,
    Adaptive,
}

#[derive(Debug, Clone)]
pub enum RebalancingTrigger {
    Load,
    Time,
    Manual,
}

#[derive(Debug, Clone, Default)]
pub struct LoadBalanceStatistics {
    pub rebalance_count: u64,
}
