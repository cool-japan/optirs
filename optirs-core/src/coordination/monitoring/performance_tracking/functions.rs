//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::types::{
    AnalysisResult, BufferedMetric, MetricQuery, PerformanceAlert, PerformanceMetrics,
    StorageBackendStats,
};

/// Metric filter trait
pub trait MetricFilter<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Filter metric data
    fn filter(&self, metric: &BufferedMetric<T>) -> bool;
    /// Get filter name
    fn name(&self) -> &str;
    /// Get filter configuration
    fn configuration(&self) -> HashMap<String, String>;
}
/// Aggregation function trait
pub trait AggregationFunction<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Aggregate metric values
    fn aggregate(&self, values: &[T]) -> Result<T>;
    /// Get function name
    fn name(&self) -> &str;
    /// Get function parameters
    fn parameters(&self) -> HashMap<String, String>;
}
/// Notification channel trait
pub trait NotificationChannel<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Send alert notification
    fn send_notification(&mut self, alert: &PerformanceAlert<T>) -> Result<()>;
    /// Get channel name
    fn name(&self) -> &str;
    /// Get channel configuration
    fn configuration(&self) -> HashMap<String, String>;
    /// Test channel connectivity
    fn test_connection(&self) -> Result<bool>;
}
/// Performance analyzer trait
pub trait PerformanceAnalyzer<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Analyze performance metrics
    fn analyze(&mut self, metrics: &PerformanceMetrics<T>) -> Result<AnalysisResult<T>>;
    /// Get analyzer name
    fn name(&self) -> &str;
    /// Get analyzer configuration
    fn configuration(&self) -> HashMap<String, String>;
}
/// Storage backend trait
pub trait StorageBackend<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Store metrics
    fn store(&mut self, metrics: &PerformanceMetrics<T>) -> Result<()>;
    /// Retrieve metrics
    fn retrieve(&self, query: &MetricQuery<T>) -> Result<Vec<PerformanceMetrics<T>>>;
    /// Delete metrics
    fn delete(&mut self, query: &MetricQuery<T>) -> Result<usize>;
    /// Get storage statistics
    fn get_statistics(&self) -> Result<StorageBackendStats>;
}
