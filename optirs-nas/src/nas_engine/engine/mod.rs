// Neural Architecture Search Engine
//
// This module implements the core NAS engine that coordinates the entire
// architecture search process, including candidate generation, evaluation,
// and optimization strategy execution.

pub mod controller;
pub mod core;
pub mod mo_optimizers;
pub mod strategies;
pub mod support;

// Re-export all types
pub use core::*;
pub use support::*;

#[cfg(test)]
mod tests;
