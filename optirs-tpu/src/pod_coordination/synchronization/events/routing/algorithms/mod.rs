// Routing algorithms

pub mod load_balancing;
pub mod failover;
pub mod topology_aware;

// Re-export algorithm types
pub use load_balancing::*;
pub use failover::*;
pub use topology_aware::*;