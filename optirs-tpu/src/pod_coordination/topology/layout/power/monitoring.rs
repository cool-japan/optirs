// Power Monitoring Module
//
// This module handles power monitoring, metrics collection, alerting,
// and telemetry for TPU pod coordination systems. All functionality has been
// decomposed into focused sub-modules for maintainability while preserving
// backward compatibility.
//
// Refactored using aggressive decomposition methodology.

mod monitoring;

pub use self::monitoring::*;