// TPU Pod Synchronization
//
// This module handles synchronization barriers, events, clock synchronization,
// and coordination mechanisms for TPU pod operations.
//
// The functionality has been modularized and is available through the synchronization subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular synchronization implementation
pub use self::synchronization::*;

// Import the modular synchronization module
pub mod synchronization;