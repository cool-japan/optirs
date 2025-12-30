//! Profiling and metrics integration using SciRS2-Core
//!
//! This module provides comprehensive profiling and monitoring for
//! optimization processes using scirs2_core's production-ready features.

use scirs2_core::metrics::{Counter, Gauge, Histogram, Timer};
use scirs2_core::profiling::Profiler;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Optimizer profiler for tracking performance metrics
///
/// This struct integrates with scirs2_core's profiling infrastructure
/// to provide comprehensive monitoring of optimization processes.
pub struct OptimizerProfiler {
    profiler: Profiler,
    step_counter: Counter,
    learning_rate_gauge: Gauge,
    gradient_norm_histogram: Histogram,
    step_duration_histogram: Histogram,
    memory_usage_gauge: Gauge,
    custom_timers: HashMap<String, Timer>,
}

impl OptimizerProfiler {
    /// Creates a new optimizer profiler
    ///
    /// # Arguments
    ///
    /// * `optimizer_name` - Name of the optimizer being profiled
    pub fn new(optimizer_name: &str) -> Self {
        let profiler = Profiler::new();

        Self {
            profiler,
            step_counter: Counter::new(format!("{}_steps", optimizer_name)),
            learning_rate_gauge: Gauge::new(format!("{}_learning_rate", optimizer_name)),
            gradient_norm_histogram: Histogram::new(format!("{}_gradient_norm", optimizer_name)),
            step_duration_histogram: Histogram::new(format!("{}_step_duration_ms", optimizer_name)),
            memory_usage_gauge: Gauge::new(format!("{}_memory_mb", optimizer_name)),
            custom_timers: HashMap::new(),
        }
    }

    /// Start profiling an optimization step
    ///
    /// # Arguments
    ///
    /// * `step_name` - Name of the step (e.g., "forward_pass", "backward_pass")
    ///
    /// # Returns
    ///
    /// A handle that automatically stops timing when dropped
    pub fn start_step(&self, step_name: &str) -> ProfileHandle {
        self.profiler.start(step_name);
        ProfileHandle {
            name: step_name.to_string(),
            start_time: Instant::now(),
            profiler: &self.profiler,
        }
    }

    /// Record a completed optimization step
    ///
    /// # Arguments
    ///
    /// * `duration` - Duration of the step
    /// * `gradient_norm` - L2 norm of gradients
    /// * `learning_rate` - Current learning rate
    pub fn record_step(&mut self, duration: Duration, gradient_norm: f64, learning_rate: f64) {
        // Increment step counter
        self.step_counter.increment();

        // Update metrics
        self.learning_rate_gauge.set(learning_rate);
        self.gradient_norm_histogram.observe(gradient_norm);
        self.step_duration_histogram
            .observe(duration.as_secs_f64() * 1000.0); // Convert to milliseconds

        // Update memory usage
        if let Ok(mem_usage_mb) = Self::get_memory_usage_mb() {
            self.memory_usage_gauge.set(mem_usage_mb);
        }
    }

    /// Get current memory usage in megabytes
    fn get_memory_usage_mb() -> Result<f64, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status")?;
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: f64 = parts[1].parse()?;
                        return Ok(kb / 1024.0); // Convert KB to MB
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let output = Command::new("ps")
                .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()?;
            let rss_kb: f64 = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()?;
            return Ok(rss_kb / 1024.0); // Convert KB to MB
        }

        // Default fallback - estimate based on system info
        Ok(0.0)
    }

    /// Create a custom timer
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the timer
    ///
    /// # Returns
    ///
    /// A timer that can be started and stopped
    pub fn create_timer(&mut self, name: String) {
        self.custom_timers.insert(name.clone(), Timer::new(name));
    }

    /// Start a custom timer
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the timer to start
    pub fn start_timer(&self, name: &str) {
        if let Some(timer) = self.custom_timers.get(name) {
            timer.start();
        }
    }

    /// Stop a custom timer and record the duration
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the timer to stop
    pub fn stop_timer(&self, name: &str) {
        if let Some(timer) = self.custom_timers.get(name) {
            timer.stop();
        }
    }

    /// Get current step count
    pub fn get_step_count(&self) -> u64 {
        self.step_counter.get()
    }

    /// Get current learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate_gauge.get()
    }

    /// Get statistics for gradient norms
    pub fn get_gradient_norm_stats(&self) -> HistogramStats {
        self.gradient_norm_histogram.get_stats()
    }

    /// Get statistics for step durations
    pub fn get_step_duration_stats(&self) -> HistogramStats {
        self.step_duration_histogram.get_stats()
    }

    /// Generate a comprehensive profiling report
    pub fn generate_report(&self) -> ProfilingReport {
        ProfilingReport {
            total_steps: self.get_step_count(),
            current_learning_rate: self.get_learning_rate(),
            gradient_norm_stats: self.get_gradient_norm_stats(),
            step_duration_stats: self.get_step_duration_stats(),
            memory_usage_mb: self.memory_usage_gauge.get(),
        }
    }
}

/// Handle for a profiling session
///
/// Automatically stops profiling when dropped
pub struct ProfileHandle<'a> {
    name: String,
    start_time: Instant,
    profiler: &'a Profiler,
}

impl<'a> Drop for ProfileHandle<'a> {
    fn drop(&mut self) {
        self.profiler.stop(&self.name);
    }
}

/// Statistics from a histogram
#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
}

/// Comprehensive profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    pub total_steps: u64,
    pub current_learning_rate: f64,
    pub gradient_norm_stats: HistogramStats,
    pub step_duration_stats: HistogramStats,
    pub memory_usage_mb: f64,
}

impl ProfilingReport {
    /// Format the report as a human-readable string
    pub fn to_string(&self) -> String {
        format!(
            r#"Profiling Report
================
Total Steps: {}
Learning Rate: {:.6e}

Gradient Norms:
  Count: {}
  Mean: {:.4e}
  Range: [{:.4e}, {:.4e}]

Step Duration (ms):
  Count: {}
  Mean: {:.2}
  Range: [{:.2}, {:.2}]

Memory Usage: {:.2} MB
"#,
            self.total_steps,
            self.current_learning_rate,
            self.gradient_norm_stats.count,
            self.gradient_norm_stats.mean,
            self.gradient_norm_stats.min,
            self.gradient_norm_stats.max,
            self.step_duration_stats.count,
            self.step_duration_stats.mean,
            self.step_duration_stats.min,
            self.step_duration_stats.max,
            self.memory_usage_mb
        )
    }
}

/// Lightweight performance tracker
///
/// A simpler alternative to OptimizerProfiler for cases
/// where full profiling is not needed.
pub struct PerformanceTracker {
    step_times: Vec<Duration>,
    gradient_norms: Vec<f64>,
    max_history: usize,
}

impl PerformanceTracker {
    /// Create a new performance tracker
    ///
    /// # Arguments
    ///
    /// * `max_history` - Maximum number of steps to keep in history
    pub fn new(max_history: usize) -> Self {
        Self {
            step_times: Vec::with_capacity(max_history),
            gradient_norms: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Record a step
    ///
    /// # Arguments
    ///
    /// * `duration` - Duration of the step
    /// * `gradient_norm` - L2 norm of gradients
    pub fn record(&mut self, duration: Duration, gradient_norm: f64) {
        // Add new values
        self.step_times.push(duration);
        self.gradient_norms.push(gradient_norm);

        // Trim to max history
        if self.step_times.len() > self.max_history {
            self.step_times.remove(0);
        }
        if self.gradient_norms.len() > self.max_history {
            self.gradient_norms.remove(0);
        }
    }

    /// Get average step duration
    pub fn avg_step_duration(&self) -> Option<Duration> {
        if self.step_times.is_empty() {
            return None;
        }

        let total: Duration = self.step_times.iter().sum();
        Some(total / self.step_times.len() as u32)
    }

    /// Get average gradient norm
    pub fn avg_gradient_norm(&self) -> Option<f64> {
        if self.gradient_norms.is_empty() {
            return None;
        }

        let sum: f64 = self.gradient_norms.iter().sum();
        Some(sum / self.gradient_norms.len() as f64)
    }

    /// Check if gradients are exploding
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold for gradient explosion
    ///
    /// # Returns
    ///
    /// True if recent gradients exceed threshold
    pub fn is_gradient_exploding(&self, threshold: f64) -> bool {
        if let Some(recent_norm) = self.gradient_norms.last() {
            *recent_norm > threshold
        } else {
            false
        }
    }

    /// Check if gradients are vanishing
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold for gradient vanishing
    ///
    /// # Returns
    ///
    /// True if recent gradients are below threshold
    pub fn is_gradient_vanishing(&self, threshold: f64) -> bool {
        if let Some(recent_norm) = self.gradient_norms.last() {
            *recent_norm < threshold
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_profiler() {
        let mut profiler = OptimizerProfiler::new("test_optimizer");

        // Record some steps
        for i in 0..10 {
            let duration = Duration::from_millis(100 + i * 10);
            let gradient_norm = 0.1 * (i as f64);
            let learning_rate = 0.001;

            profiler.record_step(duration, gradient_norm, learning_rate);
        }

        assert_eq!(profiler.get_step_count(), 10);
        assert_eq!(profiler.get_learning_rate(), 0.001);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(5);

        // Record some steps
        tracker.record(Duration::from_millis(100), 0.5);
        tracker.record(Duration::from_millis(120), 0.6);
        tracker.record(Duration::from_millis(110), 0.55);

        assert!(tracker.avg_step_duration().is_some());
        assert!(tracker.avg_gradient_norm().is_some());
    }

    #[test]
    fn test_gradient_explosion_detection() {
        let mut tracker = PerformanceTracker::new(5);

        tracker.record(Duration::from_millis(100), 0.5);
        assert!(!tracker.is_gradient_exploding(1.0));

        tracker.record(Duration::from_millis(100), 10.0);
        assert!(tracker.is_gradient_exploding(1.0));
    }

    #[test]
    fn test_gradient_vanishing_detection() {
        let mut tracker = PerformanceTracker::new(5);

        tracker.record(Duration::from_millis(100), 0.5);
        assert!(!tracker.is_gradient_vanishing(0.1));

        tracker.record(Duration::from_millis(100), 0.01);
        assert!(tracker.is_gradient_vanishing(0.1));
    }
}
