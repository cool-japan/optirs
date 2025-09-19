// Core event synchronization components
//
// This module contains the foundational types, configurations, and builders
// for the event synchronization system.

pub mod builder;
pub mod config;
pub mod types;

// Re-export core types for convenience
pub use builder::*;
pub use config::*;
pub use types::*;
