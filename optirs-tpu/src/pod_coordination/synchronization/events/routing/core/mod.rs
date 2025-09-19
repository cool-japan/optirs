// Core routing functionality
//
// This module contains the core routing engine, routing tables, and strategies.

pub mod router;
pub mod strategy;
pub mod table;

// Re-export core types
pub use router::*;
pub use strategy::*;
pub use table::*;
