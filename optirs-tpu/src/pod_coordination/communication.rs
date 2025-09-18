// TPU Communication Management
//
// This module handles communication management, message buffering, compression,
// and communication optimization for TPU pod coordination.
//
// The functionality has been modularized and is available through the communication subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular communication implementation
pub use self::communication::*;

// Import the modular communication module
pub mod communication;