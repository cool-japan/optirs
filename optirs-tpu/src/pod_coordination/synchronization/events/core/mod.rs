// Core event synchronization components
//
// This module contains the foundational types, configurations, and builders
// for the event synchronization system.

pub mod types;
pub mod config;
pub mod builder;

// Re-export core types for convenience
pub use types::*;
pub use config::*;
pub use builder::*;