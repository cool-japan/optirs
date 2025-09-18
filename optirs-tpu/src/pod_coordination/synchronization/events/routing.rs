// Event Routing Strategies, Load Balancing, and Failover
//
// This module provides comprehensive event routing capabilities for TPU synchronization
// including intelligent routing strategies, load balancing algorithms, failover mechanisms,
// health monitoring, traffic management, and routing analytics.
//
// The functionality has been modularized and is available through the routing subdirectory.
// This file serves as a convenience re-export for backward compatibility.

// Re-export everything from the modular routing implementation
pub use self::routing::*;

// Import the modular routing module
pub mod routing;