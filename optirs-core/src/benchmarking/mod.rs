// Benchmarking and performance testing module
//
// This module provides comprehensive benchmarking capabilities for optimization
// algorithms across different platforms and hardware targets.

pub mod cross_platform_tester;

// Re-export key types
pub use cross_platform_tester::{
    CrossPlatformTester, PerformanceBaseline, PlatformTarget, TestConfiguration,
};
