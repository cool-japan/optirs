// Event Compression Algorithms and Adaptive Compression
//
// This module provides comprehensive event compression capabilities for TPU synchronization
// including multiple compression algorithms, adaptive compression strategies, real-time
// streaming compression, compression analytics, and performance optimization.
//
// The functionality has been modularized and is available through the compression subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular compression implementation
pub use self::compression::*;

// Import the modular compression module
pub mod compression;