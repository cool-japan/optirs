// Event Compression Algorithms and Adaptive Compression
//
// This module provides comprehensive event compression capabilities for TPU synchronization
// including multiple compression algorithms, adaptive compression strategies, real-time
// streaming compression, compression analytics, and performance optimization.
//
// The functionality has been modularized and is available through focused sub-modules.
// This file serves as a convenience re-export for backward compatibility.

// Import and re-export all sub-modules
pub mod adaptive;
pub mod algorithms;
pub mod analytics;
pub mod pipelines;
pub mod streaming;

// Re-export core types and functionality
pub use algorithms::{
    Algorithm, ArithmeticConfig, BrotliConfig, BrotliMode, BurrowsWheelerConfig,
    CompressionAlgorithms, DeflateConfig, DeflateStrategy, GzipConfig, GzipStrategy, HuffmanConfig,
    Lz4BlockSize, Lz4Config, Lz77Config, Lz78Config, LzoAlgorithm, LzoConfig, Lzw2Config,
    PpmConfig, SnappyConfig, ZstdConfig, ZstdDictionary,
};

pub use adaptive::{
    AdaptiveCompression, AdaptiveController, AdaptiveStrategy, AlgorithmSelector,
    OptimizationObjective, ParameterOptimizer, PerformanceMonitor, SelectionAlgorithm,
    SelectionCriteria, SelectionFrequency, SelectionStrategy,
};

pub use streaming::{
    BufferManagement, ErrorHandling, FlowControl, StreamingAnalytics, StreamingBuffer,
    StreamingCompression, StreamingConfig, StreamingOptimization, StreamingPerformance,
};

pub use analytics::{
    AnalyticsConfig, AnalyticsStorage, CompressionAnalytics, CompressionMetrics, MetricsCollector,
    PerformanceAnalyzer, PerformanceMetrics, QualityMetrics, ReportGenerator,
};

pub use pipelines::{
    CompressionPipelines, ExecutionMode, MonitoringConfig, Pipeline, PipelineConfig,
    PipelineExecution, PipelineMonitoring, PipelineOptimization, PipelineStage, StageType,
};

// Core error and result types
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during compression operations
#[derive(Error, Debug)]
pub enum CompressionError {
    #[error("Compression algorithm error: {0}")]
    AlgorithmError(String),
    #[error("Decompression error: {0}")]
    DecompressionError(String),
    #[error("Unsupported compression format: {0}")]
    UnsupportedFormat(String),
    #[error("Compression buffer overflow: {0}")]
    BufferOverflow(String),
    #[error("Compression ratio threshold not met: {0}")]
    CompressionRatioError(String),
    #[error("Adaptive compression error: {0}")]
    AdaptiveCompressionError(String),
    #[error("Streaming compression error: {0}")]
    StreamingError(String),
    #[error("Compression pipeline error: {0}")]
    PipelineError(String),
}

/// Result type for compression operations
pub type CompressionResult<T> = Result<T, CompressionError>;

/// Event compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCompression {
    /// Compression algorithms configuration
    pub algorithms: CompressionAlgorithms,
    /// Adaptive compression settings
    pub adaptive_compression: AdaptiveCompression,
    /// Streaming compression configuration
    pub streaming: StreamingCompression,
    /// Compression analytics
    pub analytics: CompressionAnalytics,
    /// Compression pipelines
    pub pipelines: CompressionPipelines,
}

impl Default for EventCompression {
    fn default() -> Self {
        Self {
            algorithms: CompressionAlgorithms::default(),
            adaptive_compression: AdaptiveCompression::default(),
            streaming: StreamingCompression::default(),
            analytics: CompressionAnalytics::default(),
            pipelines: CompressionPipelines::default(),
        }
    }
}

/// Event compression builder for easy configuration
pub struct EventCompressionBuilder {
    config: EventCompression,
}

impl EventCompressionBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EventCompression::default(),
        }
    }

    /// Configure compression algorithms
    pub fn with_algorithms(mut self, algorithms: CompressionAlgorithms) -> Self {
        self.config.algorithms = algorithms;
        self
    }

    /// Configure adaptive compression
    pub fn with_adaptive_compression(mut self, adaptive: AdaptiveCompression) -> Self {
        self.config.adaptive_compression = adaptive;
        self
    }

    /// Configure streaming compression
    pub fn with_streaming(mut self, streaming: StreamingCompression) -> Self {
        self.config.streaming = streaming;
        self
    }

    /// Configure analytics
    pub fn with_analytics(mut self, analytics: CompressionAnalytics) -> Self {
        self.config.analytics = analytics;
        self
    }

    /// Configure pipelines
    pub fn with_pipelines(mut self, pipelines: CompressionPipelines) -> Self {
        self.config.pipelines = pipelines;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> EventCompression {
        self.config
    }
}

impl Default for EventCompressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression presets for common configurations
pub struct CompressionPresets;

impl CompressionPresets {
    /// High-performance compression configuration
    pub fn high_performance() -> EventCompression {
        EventCompressionBuilder::new()
            .with_algorithms(CompressionAlgorithms::high_performance())
            .with_adaptive_compression(AdaptiveCompression::performance_optimized())
            .with_streaming(StreamingCompression::high_throughput())
            .with_analytics(CompressionAnalytics::performance_focused())
            .build()
    }

    /// High-compression ratio configuration
    pub fn high_compression() -> EventCompression {
        EventCompressionBuilder::new()
            .with_algorithms(CompressionAlgorithms::high_compression())
            .with_adaptive_compression(AdaptiveCompression::ratio_optimized())
            .with_streaming(StreamingCompression::compression_focused())
            .with_analytics(CompressionAnalytics::ratio_focused())
            .build()
    }

    /// Balanced compression configuration
    pub fn balanced() -> EventCompression {
        EventCompressionBuilder::new()
            .with_algorithms(CompressionAlgorithms::balanced())
            .with_adaptive_compression(AdaptiveCompression::balanced())
            .with_streaming(StreamingCompression::balanced())
            .with_analytics(CompressionAnalytics::balanced())
            .build()
    }

    /// Low-latency compression configuration
    pub fn low_latency() -> EventCompression {
        EventCompressionBuilder::new()
            .with_algorithms(CompressionAlgorithms::low_latency())
            .with_adaptive_compression(AdaptiveCompression::latency_optimized())
            .with_streaming(StreamingCompression::low_latency())
            .with_analytics(CompressionAnalytics::latency_focused())
            .build()
    }
}
