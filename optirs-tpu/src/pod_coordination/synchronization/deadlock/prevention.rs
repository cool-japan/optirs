// Deadlock Prevention Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PreventionTechnique;

pub type WoundWaitStrategy = PreventionTechnique;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreventionPolicy {
    WoundWait,
    WaitDie,
    NoWait,
}

impl Default for PreventionPolicy {
    fn default() -> Self {
        Self::WoundWait
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeadlockPreventer {
    pub policy: PreventionPolicy,
    pub technique: PreventionTechnique,
}

// Additional prevention types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPolicy {
    Conservative,
    Optimistic,
    Adaptive,
}

impl Default for AllocationPolicy {
    fn default() -> Self {
        Self::Conservative
    }
}

#[derive(Debug, Clone, Default)]
pub struct AvoidanceStrategy {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct BankersAlgorithmConfig {
    pub max_resources: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CircularWaitPrevention {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ConservativeStrategy {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DeadlockPrevention {
    pub policy: PreventionPolicy,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DeadlockPreventionSystem {
    pub prevention: DeadlockPrevention,
}

impl DeadlockPreventionSystem {
    /// Create a new deadlock prevention system
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct HoldAndWaitPrevention {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct MutualExclusionPrevention {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct NoPreemptionPrevention {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct OptimisticStrategy {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderingStrategy {
    Total,
    Partial,
    None,
}

impl Default for OrderingStrategy {
    fn default() -> Self {
        Self::Total
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    Allow,
    Deny,
    Conditional,
}

impl Default for PreemptionPolicy {
    fn default() -> Self {
        Self::Conditional
    }
}

#[derive(Debug, Clone, Default)]
pub struct PreemptionStrategy {
    pub policy: PreemptionPolicy,
}

#[derive(Debug, Clone, Default)]
pub struct PreventionStatistics {
    pub prevented_count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceOrdering {
    pub strategy: OrderingStrategy,
}

#[derive(Debug, Clone, Default)]
pub struct TimeoutStrategy {
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationStatistics {
    pub validations_performed: u64,
}
