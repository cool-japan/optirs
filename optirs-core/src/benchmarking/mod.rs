// Benchmarking and performance testing module
//
// This module provides comprehensive benchmarking capabilities for optimization
// algorithms across different platforms and hardware targets.

#[cfg(feature = "cross-platform-testing")]
pub mod cross_platform_tester;

// Re-export key types
#[cfg(feature = "cross-platform-testing")]
pub use cross_platform_tester::{
    CrossPlatformTester, PerformanceBaseline, PlatformTarget, TestConfiguration,
};
