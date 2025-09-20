// Global Performance Configuration Module
//
// This module provides comprehensive global performance configuration for TPU systems.
// All functionality has been decomposed into focused modules for better maintainability.

pub mod caching_config;
pub mod io_config;
pub mod memory_config;
pub mod network_config;
pub mod threading_config;

pub use self::caching_config::*;
pub use self::io_config::*;
pub use self::memory_config::*;
pub use self::network_config::*;
pub use self::threading_config::*;
