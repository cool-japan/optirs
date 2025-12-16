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

// ROADMAP (v1.1.0+): Additional modules planned for future releases:
// pub mod optimization;  // OptimizationPipeline, PassManager, etc.
// pub mod memory;        // MemoryPlanner, memory allocation
// pub mod performance;   // PerformanceAnalyzer, profiling
// pub mod codegen;       // TPUCodeGenerator
// pub mod cache;         // CompilationCache
// pub mod parallel;      // ParallelCompilationManager
//
// v1.0.0 provides the core XLA compilation infrastructure.
// Advanced features above will be added in future releases.
