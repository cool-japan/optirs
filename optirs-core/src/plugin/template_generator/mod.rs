// Template generator module
//
// This module provides comprehensive template generation capabilities
// for creating plugin templates with advanced features.

pub mod config;
pub mod templates;
pub mod registry;
pub mod generators;
pub mod validation;

// Re-export key types
pub use config::*;
pub use templates::*;
pub use registry::*;
pub use generators::*;
pub use validation::*;