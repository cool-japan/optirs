// TPU Coordination and Orchestration Primitives
//
// This module provides synchronization and coordination primitives specifically designed for TPU pod coordination,
// including coordination strategies, orchestration mechanisms, and pod-level synchronization primitives.
//
// The functionality has been modularized and is available through the coordination subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular coordination implementation
pub use self::coordination::*;

// Import the modular coordination module
pub mod coordination;