// Routing Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    Random,
}

impl Default for LoadBalancingAlgorithm {
    fn default() -> Self {
        Self::RoundRobin
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkTopology;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MinLatency,
    MaxThroughput,
    MinCost,
}

impl Default for OptimizationObjective {
    fn default() -> Self {
        Self::MinLatency
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouteState {
    Active,
    Inactive,
    Failed,
}

impl Default for RouteState {
    fn default() -> Self {
        Self::Active
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingTable {
    pub routes: HashMap<String, Route>,
}
