// Clock Synchronization and Time Management
//
// This module provides comprehensive clock synchronization capabilities including
// time source management, drift compensation, accuracy tracking, and distributed clock coordination.
//
// The functionality has been modularized and is available through the clocks subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular clocks implementation
pub use self::clocks::*;

// Import the modular clocks module
pub mod clocks;