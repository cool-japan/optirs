// Real resource-usage prediction and trend analysis (findings R3, R7, CF1).
//
// * R3: `update_trend_analysis` built its window with `.rev().take(10)`, i.e.
//   newest-first, and then treated the *first* half as "older". Every reported
//   trend direction was therefore inverted.
// * R7: `prediction_horizon`, `prediction_accuracy` and `seasonal_patterns`
//   were never read; there was no prediction at all.
// * CF1: `ResourceConfig::enable_resource_prediction` was never read.

use super::{ResourcePredictor, ResourceTrendAnalysis, ResourceUsage, TrendDirection};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Slots in the seasonal profile: one per minute of the hour.
const SEASONAL_SLOTS: usize = 60;

/// Samples retained for trend and prediction work.
const MAX_PATTERNS: usize = 1000;

/// Trend window length.
const TREND_WINDOW: usize = 10;

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
}

/// Ordinary-least-squares slope against the sample index.
fn ols_slope(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let x_mean = (values.len() - 1) as f64 / 2.0;
    let y_mean = mean(values);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (index, value) in values.iter().enumerate() {
        let dx = index as f64 - x_mean;
        numerator += dx * (value - y_mean);
        denominator += dx * dx;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn current_seasonal_slot() -> usize {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    ((seconds / 60) % SEASONAL_SLOTS as u64) as usize
}

impl ResourcePredictor {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            usage_patterns: VecDeque::with_capacity(MAX_PATTERNS),
            prediction_horizon: 10,
            prediction_accuracy: HashMap::new(),
            seasonal_patterns: HashMap::new(),
            trend_analysis: ResourceTrendAnalysis {
                memory_trend: TrendDirection::Unknown,
                cpu_trend: TrendDirection::Unknown,
                network_trend: TrendDirection::Unknown,
                trend_confidence: 0.0,
                trend_stability: 0.0,
            },
            enabled,
            seasonal_counts: HashMap::new(),
            pending_prediction: None,
        }
    }

    pub(crate) fn accuracy(&self) -> &HashMap<String, f64> {
        &self.prediction_accuracy
    }

    pub(crate) fn trend_analysis(&self) -> &ResourceTrendAnalysis {
        &self.trend_analysis
    }

    pub(crate) fn update(&mut self, usage: &ResourceUsage) -> Result<(), String> {
        if !self.enabled {
            // CF1: prediction is opt-in; without it we do no bookkeeping at all
            // rather than quietly maintaining state nobody can read.
            return Ok(());
        }

        self.score_pending_prediction(usage);

        if self.usage_patterns.len() >= MAX_PATTERNS {
            self.usage_patterns.pop_front();
        }
        self.usage_patterns.push_back(usage.clone());

        self.update_seasonal_profile(usage);

        if self.usage_patterns.len() >= 3 {
            self.update_trend_analysis()?;
        }

        // Stage a fresh prediction for the horizon so its error can be
        // measured against reality later.
        if let Some(prediction) = self.predict() {
            self.pending_prediction = Some((self.prediction_horizon.max(1), prediction));
        }

        Ok(())
    }

    /// Compare the outstanding prediction with the sample that arrived and fold
    /// the absolute percentage error into a running mean.
    fn score_pending_prediction(&mut self, actual: &ResourceUsage) {
        let Some((remaining, prediction)) = self.pending_prediction.take() else {
            return;
        };
        if remaining > 1 {
            self.pending_prediction = Some((remaining - 1, prediction));
            return;
        }

        let mut record = |key: &str, predicted: f64, observed: f64| {
            if observed.abs() <= f64::EPSILON {
                return;
            }
            let error = (predicted - observed).abs() / observed.abs();
            let entry = self
                .prediction_accuracy
                .entry(key.to_string())
                .or_insert(0.0);
            // Exponential moving average of the absolute percentage error.
            *entry = *entry * 0.8 + error * 0.2;
        };

        record(
            "memory",
            prediction.memory_usage_mb as f64,
            actual.memory_usage_mb as f64,
        );
        if let (Some(predicted), Some(observed)) = (prediction.cpu_usage(), actual.cpu_usage()) {
            record("cpu", predicted, observed);
        }
        if let (Some(predicted), Some(observed)) =
            (prediction.network_io_mbps, actual.network_io_mbps)
        {
            record("network", predicted, observed);
        }
    }

    fn update_seasonal_profile(&mut self, usage: &ResourceUsage) {
        let slot = current_seasonal_slot();
        let mut fold = |key: &str, value: f64| {
            let profile = self
                .seasonal_patterns
                .entry(key.to_string())
                .or_insert_with(|| vec![0.0; SEASONAL_SLOTS]);
            let counts = self
                .seasonal_counts
                .entry(key.to_string())
                .or_insert_with(|| vec![0u64; SEASONAL_SLOTS]);
            counts[slot] += 1;
            let count = counts[slot] as f64;
            profile[slot] += (value - profile[slot]) / count;
        };

        fold("memory", usage.memory_usage_mb as f64);
        if let Some(cpu) = usage.cpu_usage() {
            fold("cpu", cpu);
        }
        if let Some(network) = usage.network_io_mbps {
            fold("network", network);
        }
    }

    /// Seasonal offset for a resource: how far the current slot usually sits
    /// from the overall profile mean. `0.0` when there is no profile yet.
    fn seasonal_offset(&self, key: &str) -> f64 {
        let Some(profile) = self.seasonal_patterns.get(key) else {
            return 0.0;
        };
        let Some(counts) = self.seasonal_counts.get(key) else {
            return 0.0;
        };
        let slot = current_seasonal_slot();
        if counts.get(slot).copied().unwrap_or(0) == 0 {
            return 0.0;
        }
        let observed: Vec<f64> = profile
            .iter()
            .zip(counts.iter())
            .filter(|(_, count)| **count > 0)
            .map(|(value, _)| *value)
            .collect();
        if observed.len() < 2 {
            return 0.0;
        }
        profile[slot] - mean(&observed)
    }

    /// Linear extrapolation over the trend window plus the seasonal offset.
    pub(crate) fn predict(&self) -> Option<ResourceUsage> {
        if !self.enabled || self.usage_patterns.len() < 3 {
            return None;
        }
        let horizon = self.prediction_horizon.max(1) as f64;

        let memory: Vec<f64> = self
            .chronological_window()
            .map(|usage| usage.memory_usage_mb as f64)
            .collect();
        let predicted_memory =
            (mean_last(&memory) + ols_slope(&memory) * horizon + self.seasonal_offset("memory"))
                .max(0.0);

        let cpu: Vec<f64> = self
            .chronological_window()
            .filter_map(|usage| usage.cpu_usage())
            .collect();
        let predicted_cpu = if cpu.len() >= 3 {
            Some(
                (mean_last(&cpu) + ols_slope(&cpu) * horizon + self.seasonal_offset("cpu"))
                    .clamp(0.0, 100.0),
            )
        } else {
            None
        };

        let network: Vec<f64> = self
            .chronological_window()
            .filter_map(|usage| usage.network_io_mbps)
            .collect();
        let predicted_network = if network.len() >= 3 {
            Some(
                (mean_last(&network)
                    + ols_slope(&network) * horizon
                    + self.seasonal_offset("network"))
                .max(0.0),
            )
        } else {
            None
        };

        let latest = self.usage_patterns.back()?;
        Some(ResourceUsage {
            memory_usage_mb: predicted_memory as usize,
            total_memory_mb: latest.total_memory_mb,
            // Not predicted: the process footprint has its own series and
            // extrapolating it from the system-wide one would be an invention.
            process_memory_mb: None,
            cpu_usage_percent: predicted_cpu.unwrap_or(0.0),
            cpu_usage_percent_valid: predicted_cpu.is_some(),
            gpu_usage_percent: None,
            network_io_mbps: predicted_network,
            disk_io_mbps: None,
            active_threads: latest.active_threads,
            timestamp: latest.timestamp,
        })
    }

    /// The trend window in chronological (oldest-first) order.
    ///
    /// R3: the previous code iterated `.rev().take(10)`, i.e. newest-first, and
    /// then compared "first half" against "second half" — inverting every
    /// reported trend.
    fn chronological_window(&self) -> impl Iterator<Item = &ResourceUsage> {
        let skip = self.usage_patterns.len().saturating_sub(TREND_WINDOW);
        self.usage_patterns.iter().skip(skip)
    }

    pub(crate) fn update_trend_analysis(&mut self) -> Result<(), String> {
        let memory_values: Vec<f64> = self
            .chronological_window()
            .map(|usage| usage.memory_usage_mb as f64)
            .collect();
        let cpu_values: Vec<f64> = self
            .chronological_window()
            .filter_map(|usage| usage.cpu_usage())
            .collect();
        let network_values: Vec<f64> = self
            .chronological_window()
            .filter_map(|usage| usage.network_io_mbps)
            .collect();

        self.trend_analysis.memory_trend = analyze_trend(&memory_values);
        self.trend_analysis.cpu_trend = analyze_trend(&cpu_values);
        self.trend_analysis.network_trend = analyze_trend(&network_values);
        self.trend_analysis.trend_confidence = trend_confidence(&memory_values, &cpu_values);
        self.trend_analysis.trend_stability = trend_stability(&memory_values);

        Ok(())
    }
}

fn mean_last(values: &[f64]) -> f64 {
    values.last().copied().unwrap_or(0.0)
}

/// Classify a chronologically ordered series.
pub(crate) fn analyze_trend(values: &[f64]) -> TrendDirection {
    if values.len() < 3 {
        return TrendDirection::Unknown;
    }

    let split = values.len() / 2;
    let older = mean(&values[..split]);
    let newer = mean(&values[split..]);

    let change_threshold = 0.05; // 5% change threshold
    let relative_change = (newer - older) / older.abs().max(1.0);

    // A series that keeps reversing direction is oscillating, not trending.
    let reversals = values
        .windows(3)
        .filter(|window| {
            let first = window[1] - window[0];
            let second = window[2] - window[1];
            first * second < 0.0
        })
        .count();
    if values.len() >= 5 && reversals * 2 >= values.len() - 2 {
        return TrendDirection::Oscillating;
    }

    if relative_change > change_threshold {
        TrendDirection::Increasing
    } else if relative_change < -change_threshold {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    }
}

fn trend_confidence(memory_values: &[f64], cpu_values: &[f64]) -> f64 {
    // Lower variance = higher confidence.
    let memory_confidence = 1.0 / (1.0 + variance(memory_values) / 100.0);
    let cpu_confidence = if cpu_values.len() >= 2 {
        1.0 / (1.0 + variance(cpu_values) / 100.0)
    } else {
        // No CPU measurements yet: do not let a fabricated 1.0 inflate the
        // confidence, fall back to the memory term alone.
        return memory_confidence;
    };
    (memory_confidence + cpu_confidence) / 2.0
}

/// Fraction of consecutive deltas that keep the same sign.
fn trend_stability(values: &[f64]) -> f64 {
    if values.len() < 3 {
        return 0.0;
    }
    let deltas: Vec<f64> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let agreeing = deltas
        .windows(2)
        .filter(|pair| pair[0] * pair[1] >= 0.0)
        .count();
    agreeing as f64 / (deltas.len() - 1) as f64
}
