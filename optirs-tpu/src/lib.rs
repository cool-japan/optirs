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
pub mod fault_tolerance;
pub mod monitoring;
pub mod synchronization;
pub mod xla;

pub use coordination::PodCoordinator;
