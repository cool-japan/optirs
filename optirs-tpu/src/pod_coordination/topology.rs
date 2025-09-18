// TPU Pod Topology Management
//
// This module handles topology management, device layout, communication topology,
// and network configuration for TPU pod coordination.
//
// The functionality has been modularized and is available through the topology subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular topology implementation
pub use self::topology::*;

// Import the modular topology module
pub mod topology;