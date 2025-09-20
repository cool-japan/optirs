// Physical Layout Module for TPU Pod Coordination
//
// This module provides comprehensive physical layout management functionality including
// 3D positioning, node management, physical connections, network interfaces, optimization,
// validation, metrics, and layout management.
//
// Refactored for modularity and maintainability.

pub mod connections;
pub mod interfaces;
pub mod layout;
pub mod metrics;
pub mod nodes;
pub mod optimization;
pub mod positioning;
pub mod validation;

pub use self::connections::*;
pub use self::interfaces::*;
pub use self::layout::*;
pub use self::metrics::*;
pub use self::nodes::*;
pub use self::optimization::*;
pub use self::positioning::*;
pub use self::validation::*;
