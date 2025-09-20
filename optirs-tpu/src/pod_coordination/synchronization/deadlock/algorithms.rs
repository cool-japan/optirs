// Deadlock Algorithms Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Linear,
    Exponential,
    Random,
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self::Exponential
    }
}

#[derive(Debug, Clone, Default)]
pub struct CacheOptimization {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    LRU,
    LFU,
    FIFO,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::LRU
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombinationStrategy {
    Voting,
    Weighted,
    Priority,
}

impl Default for CombinationStrategy {
    fn default() -> Self {
        Self::Voting
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    WaitDie,
    WoundWait,
    NoWait,
}

impl Default for ConflictResolution {
    fn default() -> Self {
        Self::WaitDie
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CycleDetectionMethod {
    DFS,
    BFS,
    Tarjan,
}

impl Default for CycleDetectionMethod {
    fn default() -> Self {
        Self::DFS
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeadlockCriteria {
    pub max_wait_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeadlockDetectionAlgorithm {
    WaitForGraph,
    ResourceAllocation,
    Timestamp,
}

impl Default for DeadlockDetectionAlgorithm {
    fn default() -> Self {
        Self::WaitForGraph
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    Retry,
    Abort,
    Fallback,
}

impl Default for ErrorHandling {
    fn default() -> Self {
        Self::Retry
    }
}

#[derive(Debug, Clone, Default)]
pub struct GraphOptimization {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphReductionMethod {
    Transitive,
    Component,
    Hierarchical,
}

impl Default for GraphReductionMethod {
    fn default() -> Self {
        Self::Transitive
    }
}

#[derive(Debug, Clone, Default)]
pub struct ParallelProcessing {
    pub num_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchingStrategy {
    Aggressive,
    Conservative,
    Adaptive,
}

impl Default for PrefetchingStrategy {
    fn default() -> Self {
        Self::Conservative
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropagationStrategy {
    Immediate,
    Batch,
    Lazy,
}

impl Default for PropagationStrategy {
    fn default() -> Self {
        Self::Immediate
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceAllocationMethod {
    BankersAlgorithm,
    Priority,
    FIFO,
}

impl Default for ResourceAllocationMethod {
    fn default() -> Self {
        Self::BankersAlgorithm
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseHandling {
    Synchronous,
    Asynchronous,
    Callback,
}

impl Default for ResponseHandling {
    fn default() -> Self {
        Self::Asynchronous
    }
}

#[derive(Debug, Clone, Default)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff: BackoffStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafeStateMethod {
    Checkpoint,
    Rollback,
    Reset,
}

impl Default for SafeStateMethod {
    fn default() -> Self {
        Self::Checkpoint
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronizationMethod {
    Lock,
    Semaphore,
    Monitor,
}

impl Default for SynchronizationMethod {
    fn default() -> Self {
        Self::Lock
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimestampOrdering {
    Lamport,
    Vector,
    Physical,
}

impl Default for TimestampOrdering {
    fn default() -> Self {
        Self::Lamport
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkDistribution {
    Static,
    Dynamic,
    Adaptive,
}

impl Default for WorkDistribution {
    fn default() -> Self {
        Self::Dynamic
    }
}
