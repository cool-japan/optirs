// Power Efficiency Module
//
// This module provides comprehensive power efficiency optimization and analysis capabilities
// for TPU pod coordination. All functionality has been decomposed into focused modules
// for better maintainability and performance.

pub mod efficiency_config;
pub mod efficiency_metrics;
pub mod efficiency_optimizer;
pub mod efficiency_reporting;
pub mod energy_saver;
pub mod performance_analyzer;

pub use self::efficiency_config::*;
pub use self::efficiency_metrics::*;
pub use self::efficiency_optimizer::*;
pub use self::efficiency_reporting::*;
pub use self::energy_saver::*;
pub use self::performance_analyzer::*;
