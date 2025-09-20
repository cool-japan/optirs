// Core Types Module for Event Synchronization System
//
// This module provides comprehensive type definitions for event synchronization,
// including core configurations, global settings, and integration configurations.
//
// Refactored for modularity and maintainability.

pub mod core;
pub mod global_settings;
pub mod integrations;

pub use self::core::*;
pub use self::global_settings::*;
pub use self::integrations::*;
