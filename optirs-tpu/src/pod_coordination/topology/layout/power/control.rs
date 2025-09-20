// Power Control Module
//
// This module handles power control systems, optimization algorithms,
// and coordination for TPU pod power management. All functionality has been
// decomposed into focused sub-modules for maintainability while preserving
// backward compatibility.
//
// Refactored using aggressive decomposition methodology.

mod control;

pub use self::control::*;