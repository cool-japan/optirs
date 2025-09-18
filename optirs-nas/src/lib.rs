//! # OptiRS NAS
//!
//! Neural Architecture Search and hyperparameter optimization for OptiRS.
//!
//! This crate provides:
//! - Neural architecture search algorithms
//! - Hyperparameter optimization
//! - Multi-objective optimization
//! - Progressive search strategies
//! - Architecture encoding and evaluation

pub mod architecture;
pub mod evaluation;
pub mod hyperparameter;
pub mod multi_objective;
pub mod progressive;
pub mod search_strategies;

pub use architecture::ArchitectureSpace;
pub use search_strategies::SearchStrategy;
