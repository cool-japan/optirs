// Memory pressure monitoring and management

use std::collections::VecDeque;
use std::time::Instant;

/// Memory pressure monitoring
#[derive(Debug, Clone)]
pub struct MemoryPressureMonitor {
    /// Enable pressure monitoring
    pub enable_monitoring: bool,
    /// Memory pressure threshold (0.0-1.0)
    pub pressure_threshold: f32,
    /// Monitoring interval (milliseconds)
    pub monitor_interval_ms: u64,
    /// Current memory pressure level
    pub current_pressure: f32,
    /// Pressure history
    pub pressure_history: VecDeque<PressureReading>,
    /// Maximum history size
    pub max_history_size: usize,
    /// Enable automatic cleanup under pressure
    pub auto_cleanup: bool,
    /// Cleanup threshold (pressure level)
    pub cleanup_threshold: f32,
    /// Last monitoring timestamp
    last_monitor_time: Instant,
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            pressure_threshold: 0.9,
            monitor_interval_ms: 1000,
            current_pressure: 0.0,
            pressure_history: VecDeque::new(),
            max_history_size: 3600, // 1 hour at 1s intervals
            auto_cleanup: true,
            cleanup_threshold: 0.95,
            last_monitor_time: Instant::now(),
        }
    }
}

/// Memory pressure reading
#[derive(Debug, Clone)]
pub struct PressureReading {
    /// Timestamp of reading
    pub timestamp: Instant,
    /// Memory pressure level (0.0-1.0)
    pub pressure: f32,
    /// Available memory (bytes)
    pub available_memory: usize,
    /// Total allocated memory (bytes)
    pub allocated_memory: usize,
}

impl MemoryPressureMonitor {
    /// Create new pressure monitor with custom settings
    pub fn new(
        pressure_threshold: f32,
        monitor_interval_ms: u64,
        auto_cleanup: bool,
    ) -> Self {
        Self {
            enable_monitoring: true,
            pressure_threshold,
            monitor_interval_ms,
            current_pressure: 0.0,
            pressure_history: VecDeque::new(),
            max_history_size: 3600,
            auto_cleanup,
            cleanup_threshold: pressure_threshold + 0.05,
            last_monitor_time: Instant::now(),
        }
    }

    /// Update memory pressure based on current pool state
    pub fn update_pressure(&mut self, used_memory: usize, total_memory: usize) {
        // Check if it's time to monitor
        if !self.should_monitor_now() {
            return;
        }

        let pressure = if total_memory > 0 {
            used_memory as f32 / total_memory as f32
        } else {
            0.0
        };

        self.current_pressure = pressure;

        // Record pressure reading
        let reading = PressureReading {
            timestamp: Instant::now(),
            pressure,
            available_memory: total_memory.saturating_sub(used_memory),
            allocated_memory: used_memory,
        };

        self.pressure_history.push_back(reading);

        // Maintain history size
        while self.pressure_history.len() > self.max_history_size {
            self.pressure_history.pop_front();
        }

        self.last_monitor_time = Instant::now();
    }

    /// Check if pressure monitoring should run now
    fn should_monitor_now(&self) -> bool {
        if !self.enable_monitoring {
            return false;
        }

        let elapsed = self.last_monitor_time.elapsed().as_millis() as u64;
        elapsed >= self.monitor_interval_ms
    }

    /// Check if system is under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        self.current_pressure > self.pressure_threshold
    }

    /// Check if automatic cleanup should be triggered
    pub fn should_trigger_cleanup(&self) -> bool {
        self.auto_cleanup && self.current_pressure > self.cleanup_threshold
    }

    /// Get pressure trend over recent history
    pub fn get_pressure_trend(&self) -> PressureTrend {
        if self.pressure_history.len() < 3 {
            return PressureTrend::Stable;
        }

        let recent_readings: Vec<_> = self.pressure_history
            .iter()
            .rev()
            .take(5) // Look at last 5 readings
            .collect();

        let first_pressure = recent_readings.last().unwrap().pressure;
        let last_pressure = recent_readings[0].pressure;
        let pressure_change = last_pressure - first_pressure;

        if pressure_change > 0.1 {
            PressureTrend::Increasing
        } else if pressure_change < -0.1 {
            PressureTrend::Decreasing
        } else {
            PressureTrend::Stable
        }
    }

    /// Get pressure statistics
    pub fn get_pressure_stats(&self) -> PressureStatistics {
        if self.pressure_history.is_empty() {
            return PressureStatistics::default();
        }

        let pressures: Vec<f32> = self.pressure_history.iter().map(|r| r.pressure).collect();

        let min_pressure = pressures.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_pressure = pressures.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_pressure = pressures.iter().sum::<f32>() / pressures.len() as f32;

        // Calculate standard deviation
        let variance = pressures.iter()
            .map(|p| (p - avg_pressure).powi(2))
            .sum::<f32>() / pressures.len() as f32;
        let std_deviation = variance.sqrt();

        let time_under_pressure = self.pressure_history.iter()
            .filter(|r| r.pressure > self.pressure_threshold)
            .count() as f32 / self.pressure_history.len() as f32;

        PressureStatistics {
            current_pressure: self.current_pressure,
            min_pressure,
            max_pressure,
            average_pressure: avg_pressure,
            std_deviation,
            time_under_pressure_ratio: time_under_pressure,
            total_readings: self.pressure_history.len(),
        }
    }

    /// Predict future pressure based on trend
    pub fn predict_future_pressure(&self, seconds_ahead: u64) -> f32 {
        let trend = self.get_pressure_trend();
        let current = self.current_pressure;

        match trend {
            PressureTrend::Increasing => {
                // Simple linear extrapolation
                let rate = self.calculate_pressure_rate();
                (current + rate * seconds_ahead as f32).min(1.0)
            }
            PressureTrend::Decreasing => {
                let rate = self.calculate_pressure_rate();
                (current + rate * seconds_ahead as f32).max(0.0)
            }
            PressureTrend::Stable => current,
        }
    }

    /// Calculate rate of pressure change per second
    fn calculate_pressure_rate(&self) -> f32 {
        if self.pressure_history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<_> = self.pressure_history.iter().rev().take(10).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let time_span = recent[0].timestamp.duration_since(recent.last().unwrap().timestamp).as_secs_f32();
        if time_span == 0.0 {
            return 0.0;
        }

        let pressure_change = recent[0].pressure - recent.last().unwrap().pressure;
        pressure_change / time_span
    }

    /// Get recommendations based on pressure analysis
    pub fn get_recommendations(&self) -> Vec<PressureRecommendation> {
        let mut recommendations = Vec::new();
        let stats = self.get_pressure_stats();

        if stats.current_pressure > 0.9 {
            recommendations.push(PressureRecommendation::ImmediateCleanup);
        } else if stats.current_pressure > self.pressure_threshold {
            recommendations.push(PressureRecommendation::TriggerCleanup);
        }

        if stats.time_under_pressure_ratio > 0.5 {
            recommendations.push(PressureRecommendation::IncreasePoolSize);
        }

        if self.get_pressure_trend() == PressureTrend::Increasing {
            recommendations.push(PressureRecommendation::MonitorClosely);
        }

        if stats.std_deviation > 0.3 {
            recommendations.push(PressureRecommendation::StabilizeWorkload);
        }

        if recommendations.is_empty() {
            recommendations.push(PressureRecommendation::Continue);
        }

        recommendations
    }

    /// Reset pressure monitoring
    pub fn reset(&mut self) {
        self.pressure_history.clear();
        self.current_pressure = 0.0;
        self.last_monitor_time = Instant::now();
    }

    /// Enable/disable monitoring
    pub fn set_monitoring_enabled(&mut self, enabled: bool) {
        self.enable_monitoring = enabled;
    }

    /// Update monitoring interval
    pub fn set_monitor_interval(&mut self, interval_ms: u64) {
        self.monitor_interval_ms = interval_ms;
    }
}

/// Memory pressure trend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressureTrend {
    Increasing,
    Decreasing,
    Stable,
}

/// Pressure statistics
#[derive(Debug, Clone, Default)]
pub struct PressureStatistics {
    pub current_pressure: f32,
    pub min_pressure: f32,
    pub max_pressure: f32,
    pub average_pressure: f32,
    pub std_deviation: f32,
    pub time_under_pressure_ratio: f32,
    pub total_readings: usize,
}

/// Pressure-based recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum PressureRecommendation {
    Continue,
    MonitorClosely,
    TriggerCleanup,
    ImmediateCleanup,
    IncreasePoolSize,
    StabilizeWorkload,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_monitor_creation() {
        let monitor = MemoryPressureMonitor::default();
        assert!(monitor.enable_monitoring);
        assert_eq!(monitor.current_pressure, 0.0);
    }

    #[test]
    fn test_pressure_update() {
        let mut monitor = MemoryPressureMonitor::default();
        monitor.monitor_interval_ms = 0; // Force immediate monitoring

        monitor.update_pressure(800, 1000);
        assert_eq!(monitor.current_pressure, 0.8);
        assert_eq!(monitor.pressure_history.len(), 1);
    }

    #[test]
    fn test_pressure_threshold() {
        let mut monitor = MemoryPressureMonitor::default();
        monitor.pressure_threshold = 0.7;
        monitor.monitor_interval_ms = 0;

        monitor.update_pressure(600, 1000);
        assert!(!monitor.is_under_pressure());

        monitor.update_pressure(800, 1000);
        assert!(monitor.is_under_pressure());
    }

    #[test]
    fn test_pressure_trend() {
        let mut monitor = MemoryPressureMonitor::default();
        monitor.monitor_interval_ms = 0;

        // Create increasing trend
        monitor.update_pressure(500, 1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.update_pressure(600, 1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.update_pressure(700, 1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.update_pressure(800, 1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.update_pressure(900, 1000);

        let trend = monitor.get_pressure_trend();
        assert_eq!(trend, PressureTrend::Increasing);
    }

    #[test]
    fn test_pressure_recommendations() {
        let mut monitor = MemoryPressureMonitor::default();
        monitor.monitor_interval_ms = 0;

        // High pressure scenario
        monitor.update_pressure(950, 1000);
        let recommendations = monitor.get_recommendations();
        assert!(recommendations.contains(&PressureRecommendation::ImmediateCleanup));
    }
}