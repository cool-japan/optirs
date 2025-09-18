// Core routing functionality
//
// This module contains the core routing engine, routing tables, and strategies.

pub mod router;
pub mod table;
pub mod strategy;

// Re-export core types
pub use router::*;
pub use table::*;
pub use strategy::*;