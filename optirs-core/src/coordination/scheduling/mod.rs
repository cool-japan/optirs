// Task scheduling and resource management for optimization coordination
//
// This module provides task scheduling, resource allocation, and priority management
// for coordinating multiple optimization processes.

#[allow(dead_code)]
pub mod priority_management;
pub mod resource_allocation;
pub mod task_scheduler;

// Re-export key types
pub use priority_management::{
    PriorityLevel, PriorityManager, PriorityQueue, PriorityUpdateStrategy, StaticPriorityStrategy,
};
pub use resource_allocation::{
    ResourceAllocationStrategy, ResourceAllocationTracker, ResourceManager,
    ResourceOptimizationEngine, ResourcePool,
};
pub use task_scheduler::{ScheduledTask, SchedulingStrategy, TaskPriority, TaskScheduler};
