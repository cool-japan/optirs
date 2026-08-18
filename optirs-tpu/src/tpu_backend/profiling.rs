//! Error handler and performance monitor: the bookkeeping components owned by
//! [`super::backend::TPUBackend`].
//!
//! Both keep real state. The monitor maintains an always-on execution summary
//! plus a `collection_interval`-decimated sample history; the error handler owns
//! the retry/recovery policy that `execute_computation` actually consults, and
//! the error statistics it reports come from the failures it saw.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::error::{OptimError, Result};

use super::types::{
    ComputationId, ErrorStatistics, ErrorType, ExecutionUtilization, PerformanceSample,
    RecoveryStrategy, TPUBackendConfig, TaskExecutionResult,
};

/// Longest sample history the monitor retains before evicting the oldest entry.
const MAX_RETAINED_SAMPLES: usize = 1024;

/// TPU error handler
#[derive(Debug)]
pub struct TPUErrorHandler {
    /// Error recovery enabled
    recovery_enabled: bool,

    /// Error statistics
    error_statistics: ErrorStatistics,

    /// Recovery strategies
    recovery_strategies: HashMap<ErrorType, RecoveryStrategy>,

    /// Max retry attempts
    max_retry_attempts: usize,

    /// Total operations observed, so `error_rate` is errors per operation
    /// rather than an unanchored number.
    operations_observed: usize,

    /// Recovery attempts and how many of them ended in a successful retry,
    /// backing `recovery_success_rate`.
    recovery_attempts: usize,

    /// Successful recoveries.
    recoveries_succeeded: usize,
}

impl TPUErrorHandler {
    /// Create a new TPU error handler
    pub fn new(config: &TPUBackendConfig) -> Self {
        // Default policy per error class. Only classes that a retry can plausibly
        // fix are marked `Retry`; the rest are surfaced to the caller. These are
        // the strategies `should_retry` consults, so the map is the real policy
        // rather than a description of one.
        let mut recovery_strategies = HashMap::new();
        recovery_strategies.insert(ErrorType::TimeoutError, RecoveryStrategy::Retry);
        recovery_strategies.insert(ErrorType::ResourceError, RecoveryStrategy::Retry);
        recovery_strategies.insert(ErrorType::CommunicationError, RecoveryStrategy::Retry);
        recovery_strategies.insert(ErrorType::MemoryError, RecoveryStrategy::Fallback);
        recovery_strategies.insert(ErrorType::DeviceError, RecoveryStrategy::Migrate);
        recovery_strategies.insert(ErrorType::ComputationError, RecoveryStrategy::Restart);

        Self {
            recovery_enabled: config.enable_error_recovery,
            error_statistics: ErrorStatistics::default(),
            recovery_strategies,
            max_retry_attempts: config.max_retry_attempts,
            operations_observed: 0,
            recovery_attempts: 0,
            recoveries_succeeded: 0,
        }
    }

    /// Get current error rate
    pub fn get_error_rate(&self) -> f64 {
        self.error_statistics.error_rate
    }

    /// Full error statistics, including the per-class breakdown.
    pub fn error_statistics(&self) -> &ErrorStatistics {
        &self.error_statistics
    }

    /// Classify an error into the class the recovery policy is keyed on.
    pub fn classify(error: &OptimError) -> ErrorType {
        match error {
            OptimError::DeviceError(_) => ErrorType::DeviceError,
            OptimError::MemoryError(_) | OptimError::AllocationError(_) => ErrorType::MemoryError,
            OptimError::TimeoutError(_) => ErrorType::TimeoutError,
            OptimError::MutexError(_) => ErrorType::ResourceError,
            _ => ErrorType::ComputationError,
        }
    }

    /// Record that one operation was attempted (successful or not), so the error
    /// rate has a denominator.
    pub fn record_operation(&mut self) {
        self.operations_observed += 1;
        self.refresh_error_rate();
    }

    /// Record a real failure and return the retry decision for it.
    ///
    /// `attempt` is the zero-based index of the attempt that just failed. A
    /// retry is offered only when recovery is enabled, the class's strategy is
    /// [`RecoveryStrategy::Retry`], and the configured attempt budget has not
    /// been spent. Every other class is surfaced to the caller unchanged --
    /// silently downgrading, say, a `MemoryError` into a retry loop would just
    /// hide a real capacity problem.
    pub fn record_error(&mut self, error: &OptimError, attempt: usize) -> bool {
        let class = Self::classify(error);
        self.error_statistics.total_errors += 1;
        *self
            .error_statistics
            .errors_by_type
            .entry(class)
            .or_insert(0) += 1;
        self.refresh_error_rate();

        let retry = self.recovery_enabled
            && attempt + 1 < self.max_retry_attempts.max(1)
            && matches!(
                self.recovery_strategies.get(&class),
                Some(RecoveryStrategy::Retry)
            );
        if retry {
            self.recovery_attempts += 1;
        }
        retry
    }

    /// Record that a retry offered by [`Self::record_error`] eventually
    /// produced a successful execution.
    pub fn record_recovery_success(&mut self) {
        self.recoveries_succeeded += 1;
        self.error_statistics.recovery_success_rate = if self.recovery_attempts == 0 {
            0.0
        } else {
            self.recoveries_succeeded as f64 / self.recovery_attempts as f64
        };
    }

    /// The configured attempt budget (always at least one attempt).
    pub fn max_attempts(&self) -> usize {
        if self.recovery_enabled {
            self.max_retry_attempts.max(1)
        } else {
            1
        }
    }

    fn refresh_error_rate(&mut self) {
        let denominator = self
            .operations_observed
            .max(self.error_statistics.total_errors);
        self.error_statistics.error_rate = if denominator == 0 {
            0.0
        } else {
            self.error_statistics.total_errors as f64 / denominator as f64
        };
    }
}

/// Performance monitor
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Monitoring enabled
    enabled: bool,

    /// Total executions
    pub total_executions: usize,

    /// Average execution time
    pub average_execution_time: Duration,

    /// Performance history
    performance_history: VecDeque<PerformanceSample>,

    /// Metrics collection interval
    collection_interval: Duration,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: &TPUBackendConfig) -> Self {
        Self {
            enabled: config.enable_performance_monitoring,
            total_executions: 0,
            average_execution_time: Duration::from_millis(0),
            performance_history: VecDeque::new(),
            collection_interval: Duration::from_millis(1000),
        }
    }

    /// Whether detailed sample retention is switched on.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The retained (interval-decimated) sample history.
    pub fn performance_history(&self) -> &VecDeque<PerformanceSample> {
        &self.performance_history
    }

    /// Fold one finished execution into the monitor.
    ///
    /// The execution count and mean execution time are always maintained --
    /// they are what [`super::backend::TPUBackend::get_performance_statistics`]
    /// reports. The per-execution sample is additionally retained when
    /// monitoring is enabled *and* at least `collection_interval` has passed
    /// since the last retained sample, which is what makes the interval a real
    /// sampling rate rather than an unused number. The first execution always
    /// produces a sample so an enabled monitor is never empty.
    pub fn record_execution(
        &mut self,
        computation_id: ComputationId,
        time: Duration,
        results: &TaskExecutionResult,
        utilization: ExecutionUtilization,
    ) {
        self.total_executions += 1;

        // Running mean over every execution seen so far.
        let count = self.total_executions as u32;
        let previous_total = self
            .average_execution_time
            .saturating_mul(count.saturating_sub(1));
        self.average_execution_time = (previous_total + time) / count.max(1);

        if !self.enabled {
            return;
        }

        let due = match self.performance_history.back() {
            Some(last) => last.timestamp.elapsed() >= self.collection_interval,
            None => true,
        };
        if !due {
            return;
        }

        // Throughput in bytes per second of real produced output; zero-length
        // spans report zero rather than an infinity.
        let seconds = time.as_secs_f64();
        let throughput = if seconds > 0.0 {
            results.output_data.len() as f64 / seconds
        } else {
            0.0
        };

        self.performance_history.push_back(PerformanceSample {
            timestamp: Instant::now(),
            computation: computation_id,
            execution_time: results.execution_time,
            throughput,
            device_utilization: utilization.device,
            memory_utilization: utilization.memory,
        });
        while self.performance_history.len() > MAX_RETAINED_SAMPLES {
            self.performance_history.pop_front();
        }
    }

    /// Drop the retained per-sample detail.
    ///
    /// The aggregate summary (`total_executions`, `average_execution_time`)
    /// deliberately survives: flushing releases the history buffer, it does not
    /// erase the fact that the executions happened.
    pub fn flush_metrics(&mut self) -> Result<()> {
        self.performance_history.clear();
        Ok(())
    }
}
