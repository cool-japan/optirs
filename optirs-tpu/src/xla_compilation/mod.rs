// XLA Compilation modules for TPU optimization
//
// This module contains the refactored XLA compilation system, split into
// focused modules for better maintainability and organization.

pub mod config;
pub mod graph;
pub mod types;

// Re-export commonly used types
pub use config::*;
pub use graph::*;
pub use types::*;

// TODO: Additional modules to be created:
// pub mod optimization;  // OptimizationPipeline, PassManager, etc.
// pub mod memory;        // MemoryPlanner, memory allocation
// pub mod performance;   // PerformanceAnalyzer, profiling
// pub mod codegen;       // TPUCodeGenerator
// pub mod cache;         // CompilationCache
// pub mod parallel;      // ParallelCompilationManager
