//! # OptiRS TPU
//!
//! TPU coordination and pod management for OptiRS.
//!
//! This crate provides:
//! - TPU pod coordination and synchronization
//! - XLA compilation and optimization
//! - Distributed training coordination
//! - Fault tolerance and recovery
//! - Performance monitoring and profiling

pub mod coordination;
pub mod error;
pub mod fault_tolerance;
pub mod monitoring;
pub mod pod_coordination;
pub mod synchronization;
pub mod tpu_backend;
pub mod xla;
pub mod xla_compilation;

// Re-export main types from mod.rs
mod main_types;
pub use main_types::*;

pub use coordination::PodCoordinator;
pub use tpu_backend::DeviceId;
