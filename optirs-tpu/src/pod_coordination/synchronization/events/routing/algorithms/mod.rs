// Routing algorithms

pub mod failover;
pub mod load_balancing;
pub mod topology_aware;

// Re-export algorithm types
pub use failover::*;
pub use load_balancing::*;
pub use topology_aware::*;
