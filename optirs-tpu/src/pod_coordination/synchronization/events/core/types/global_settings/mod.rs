// Global Settings Module for Event Synchronization System
//
// This module provides comprehensive global configuration settings for the entire
// event synchronization system, including coordination, error handling, monitoring,
// and performance optimization.
//
// Refactored for modularity and maintainability.

pub mod coordination;
pub mod error_handling;
pub mod monitoring;
pub mod performance;

pub use self::coordination::*;
pub use self::error_handling::*;
pub use self::monitoring::*;
pub use self::performance::*;
