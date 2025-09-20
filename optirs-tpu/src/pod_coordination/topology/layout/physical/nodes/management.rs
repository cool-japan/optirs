// Node Management Module
//
// This module provides comprehensive node management functionality for TPU pod
// coordination systems. All functionality has been decomposed into focused
// sub-modules for maintainability while preserving backward compatibility.
//
// Refactored using aggressive decomposition methodology.

mod management;

pub use self::management::*;