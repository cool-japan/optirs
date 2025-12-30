// Routing Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LoadBalancingAlgorithm {
    #[default]
    RoundRobin,
    LeastConnections,
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkTopology;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OptimizationObjective {
    #[default]
    MinLatency,
    MaxThroughput,
    MinCost,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Route {
    pub path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouteMetrics {
    pub latency_ms: u64,
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RouteState {
    #[default]
    Active,
    Inactive,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingTable {
    pub routes: HashMap<String, Route>,
}
