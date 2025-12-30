// Deadlock Algorithms Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum BackoffStrategy {
    Linear,
    #[default]
    Exponential,
    Random,
}

#[derive(Debug, Clone, Default)]
pub struct CacheOptimization {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CachePolicy {
    #[default]
    LRU,
    LFU,
    FIFO,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CombinationStrategy {
    #[default]
    Voting,
    Weighted,
    Priority,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ConflictResolution {
    #[default]
    WaitDie,
    WoundWait,
    NoWait,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CycleDetectionMethod {
    #[default]
    DFS,
    BFS,
    Tarjan,
}

#[derive(Debug, Clone, Default)]
pub struct DeadlockCriteria {
    pub max_wait_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DeadlockDetectionAlgorithm {
    #[default]
    WaitForGraph,
    ResourceAllocation,
    Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ErrorHandling {
    #[default]
    Retry,
    Abort,
    Fallback,
}

#[derive(Debug, Clone, Default)]
pub struct GraphOptimization {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum GraphReductionMethod {
    #[default]
    Transitive,
    Component,
    Hierarchical,
}

#[derive(Debug, Clone, Default)]
pub struct ParallelProcessing {
    pub num_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PrefetchingStrategy {
    Aggressive,
    #[default]
    Conservative,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PropagationStrategy {
    #[default]
    Immediate,
    Batch,
    Lazy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ResourceAllocationMethod {
    #[default]
    BankersAlgorithm,
    Priority,
    FIFO,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ResponseHandling {
    Synchronous,
    #[default]
    Asynchronous,
    Callback,
}

#[derive(Debug, Clone, Default)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff: BackoffStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SafeStateMethod {
    #[default]
    Checkpoint,
    Rollback,
    Reset,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SynchronizationMethod {
    #[default]
    Lock,
    Semaphore,
    Monitor,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TimestampOrdering {
    #[default]
    Lamport,
    Vector,
    Physical,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum WorkDistribution {
    Static,
    #[default]
    Dynamic,
    Adaptive,
}
