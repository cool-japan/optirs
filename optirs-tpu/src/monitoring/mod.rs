// Comprehensive monitoring and performance tracking for TPU pod topology management
//
// This module provides extensive monitoring capabilities including performance metrics,
// health monitoring, traffic analysis, anomaly detection, alerting systems, and
// comprehensive statistics collection for topology management systems.
//
// Split into focused submodules (originally a single ~2000-line file):
// - `types`: settings/config/record structs and enums (all-`pub`-field data bags).
// - `defaults`: their `Default` implementations.
// - `monitor_impl`: the behavior of `TopologyPerformanceMonitor` (metrics
//   collection, health checks, anomaly detection, report generation).
//
// `TopologyPerformanceMonitor::perform_health_check` is `pub(super)` here
// solely because it is now reached from the sibling `tests` module (see its
// doc comment); this does not widen the crate's public API. Every type that
// was `pub` at the top of the original file is re-exported below at the same
// `monitoring::` path.

mod defaults;
mod monitor_impl;
#[cfg(test)]
mod tests;
mod types;

pub use types::*;
