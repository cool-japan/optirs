// Physical Layout Module for TPU Pod Coordination
//
// This module provides comprehensive physical layout management functionality including
// 3D positioning, node management, physical connections, network interfaces, optimization,
// validation, metrics, and layout management.
//
// Refactored for modularity and maintainability.

pub mod metrics;
pub mod nodes;
pub mod positioning;

pub use self::metrics::*;
pub use self::nodes::*;
pub use self::positioning::*;
