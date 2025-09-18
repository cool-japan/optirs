// TPU Pod Network Communication and Topology Management
//
// This module handles network communication, topology management, routing protocols,
// QoS settings, traffic management, and network monitoring for TPU pod coordination.
//
// The functionality has been modularized and is available through the network subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular network implementation
pub use self::network::*;

// Import the modular network module
pub mod network;