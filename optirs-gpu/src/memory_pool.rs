// CUDA memory pool management (compatibility redirect)
//
// This module provides backward compatibility access to the refactored memory pool components.
// The implementation has been moved to the `memory_pool` submodule for better organization.

// Re-export everything from the memory_pool module to maintain backward compatibility
pub use self::memory_pool::*;

// Include the memory_pool submodule
pub mod memory_pool;