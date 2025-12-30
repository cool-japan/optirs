// Load Balancing Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DeviceAvailability {
    #[default]
    Available,
    Busy,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceLoad {
    pub device_id: DeviceId,
    pub utilization: f64,
}

#[derive(Debug, Clone, Default)]
pub struct LoadBalancer;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LoadBalancingAlgorithm {
    #[default]
    RoundRobin,
    LeastLoaded,
    Random,
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
