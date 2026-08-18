// Performance Forecast Modeling
//
// This module implements **univariate time-series forecasters** for projecting
// future performance metrics (wall-clock time, peak memory, throughput, etc.)
// from historical observations. It is **time-series forecasting** and is
// deliberately distinct from `performance_prediction.rs` which solves a
// **cross-sectional regression** problem (predicting a metric from a feature
// vector at a single point in time).
//
// Four forecaster flavors are exposed via the [`PerformanceForecaster`] trait:
//
// 1. [`MovingAverageForecaster`]       - constant prediction equal to the mean
//    of the trailing window. Useful as a baseline.
// 2. [`ExponentialSmoothingForecaster`] - single exponential smoothing (EWMA).
//    Tracks the local level of a stationary series.
// 3. [`HoltLinearForecaster`]          - double exponential smoothing with
//    additive trend (Holt's linear method).
// 4. [`HoltWintersForecaster`]         - triple exponential smoothing with
//    additive seasonality (Hyndman additive Holt-Winters).
//
// All implementations follow the workspace policies: snake_case naming,
// `scirs2_core` only (no direct ndarray/rand), no `.unwrap()` in production
// code, and a single file kept under 2000 lines.
//
// Confidence intervals are computed from the one-step-ahead in-sample residual
// standard deviation. The half-width at horizon `h` is
// `z * residual_stddev * sqrt(h)` where `z` is the normal quantile for the
// configured confidence level (1.645 / 1.96 / 2.576 for 90% / 95% / 99%).

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// -----------------------------------------------------------------------------
// Public data types
// -----------------------------------------------------------------------------

/// A univariate time-series of observed performance measurements.
///
/// The values and their associated timestamps are stored in parallel vectors.
/// Timestamps must be **monotonically non-decreasing** when pushed via
/// [`TimeSeries::push`]; the constructor allows starting from an empty series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Observed scalar values in arrival order.
    pub values: Vec<f64>,
    /// Seconds since some reference epoch (must be monotonic).
    pub timestamps_secs: Vec<f64>,
    /// Human-readable metric label (e.g. `"wall_clock_seconds"`).
    pub metric_name: String,
}

impl TimeSeries {
    /// Create an empty time series with the supplied metric name.
    pub fn new(metric_name: impl Into<String>) -> Self {
        Self {
            values: Vec::new(),
            timestamps_secs: Vec::new(),
            metric_name: metric_name.into(),
        }
    }

    /// Append a new observation, validating that the timestamp is monotonic
    /// (greater than or equal to the last timestamp).
    pub fn push(&mut self, value: f64, timestamp_secs: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "TimeSeries::push: value must be finite, got {value}"
            )));
        }
        if !timestamp_secs.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "TimeSeries::push: timestamp must be finite, got {timestamp_secs}"
            )));
        }
        if let Some(&last_ts) = self.timestamps_secs.last() {
            if timestamp_secs < last_ts {
                return Err(OptimError::InvalidParameter(format!(
                    "TimeSeries::push: timestamps must be monotonic, got {timestamp_secs} after {last_ts}"
                )));
            }
        }
        self.values.push(value);
        self.timestamps_secs.push(timestamp_secs);
        Ok(())
    }

    /// Number of observations in the series.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the series contains no observations.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Immutable view of the underlying value slice.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Immutable view of the timestamp slice.
    pub fn timestamps(&self) -> &[f64] {
        &self.timestamps_secs
    }
}

/// A single forecasted future point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastPoint {
    /// 1-indexed horizon offset (`1` = the very next step).
    pub horizon_step: usize,
    /// Point prediction.
    pub predicted_value: f64,
    /// Lower bound of the confidence interval.
    pub lower_bound: f64,
    /// Upper bound of the confidence interval.
    pub upper_bound: f64,
}

/// A complete forecast over a horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Forecaster name (e.g. `"holt_linear"`).
    pub method: String,
    /// Forecasted points, ordered by horizon (1..=h).
    pub points: Vec<ForecastPoint>,
    /// In-sample one-step-ahead residual standard deviation used for the CI.
    pub fit_residual_stddev: f64,
    /// Confidence level used to compute the bounds (e.g. `0.95`).
    pub confidence_level: f64,
}

/// Common configuration shared by every forecaster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastConfig {
    /// Two-sided confidence level (e.g. `0.95`).
    pub confidence_level: f64,
    /// Minimum number of training samples required before a fit succeeds.
    pub min_samples: usize,
    /// Optional seasonal period (only used by [`HoltWintersForecaster`]).
    pub seasonality_period: Option<usize>,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_samples: 10,
            seasonality_period: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Forecaster trait
// -----------------------------------------------------------------------------

/// Unified interface for time-series forecasters.
pub trait PerformanceForecaster: Send + Sync + Debug {
    /// Fit the forecaster on `series` and store internal state for prediction.
    fn fit(&mut self, series: &TimeSeries) -> Result<()>;

    /// Produce a forecast of length `horizon` (must be `>= 1`).
    fn forecast(&self, horizon: usize) -> Result<ForecastResult>;

    /// Whether [`Self::fit`] has been successfully called.
    fn is_fitted(&self) -> bool;

    /// Human-readable name of the method.
    fn method_name(&self) -> &str;
}

// -----------------------------------------------------------------------------
// Standalone helpers (also exported for downstream callers)
// -----------------------------------------------------------------------------

/// Arithmetic mean of a slice. Returns `0.0` for an empty slice.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().copied().sum();
    sum / values.len() as f64
}

/// Sample variance with `(n - 1)` denominator (Bessel-corrected).
///
/// Returns `0.0` for `n <= 1`.
pub fn variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(values);
    let mut acc = 0.0_f64;
    for &v in values {
        let d = v - m;
        acc += d * d;
    }
    acc / (n as f64 - 1.0)
}

/// Pearson auto-correlation at the supplied non-zero `lag`.
///
/// The validity constraints are:
/// * `lag > 0`
/// * `lag < values.len() / 2` (ensures enough overlap for stability)
///
/// Returns `0.0` when either window has zero variance (constant series), and
/// an [`OptimError::InvalidParameter`] when the lag is out of range.
pub fn auto_correlation(values: &[f64], lag: usize) -> Result<f64> {
    let n = values.len();
    if lag == 0 {
        return Err(OptimError::InvalidParameter(
            "auto_correlation: lag must be >= 1".to_string(),
        ));
    }
    if lag >= n / 2 {
        return Err(OptimError::InvalidParameter(format!(
            "auto_correlation: lag={lag} must be < n/2 = {} (n={n})",
            n / 2
        )));
    }

    let left = &values[..n - lag];
    let right = &values[lag..];

    let m_left = mean(left);
    let m_right = mean(right);

    let mut num = 0.0_f64;
    let mut sq_left = 0.0_f64;
    let mut sq_right = 0.0_f64;
    for i in 0..left.len() {
        let dl = left[i] - m_left;
        let dr = right[i] - m_right;
        num += dl * dr;
        sq_left += dl * dl;
        sq_right += dr * dr;
    }

    let denom = (sq_left * sq_right).sqrt();
    if denom < 1e-30 {
        // One of the windows is essentially constant => undefined correlation.
        return Ok(0.0);
    }
    Ok(num / denom)
}

/// Heuristic seasonality detector based on the **first local maximum** of the
/// auto-correlation function exceeding `0.5`.
///
/// Scans lags in `2..=max_lag` (capped at `(n - 1) / 2`) and looks for the
/// smallest lag whose autocorrelation:
/// * exceeds the threshold (`0.5`), and
/// * is a local maximum (strictly greater than the autocorrelations at
///   `lag - 1` and `lag + 1`).
///
/// This avoids returning multiples of the true period (e.g. for a pure sine
/// with period 12 the autocorrelation also peaks at 24, 36, ...; we want 12).
/// Returns [`None`] otherwise, or when the series is too short.
pub fn detect_seasonality_period(values: &[f64], max_lag: usize) -> Option<usize> {
    let n = values.len();
    if n < 4 || max_lag < 2 {
        return None;
    }
    let cap = (n - 1) / 2;
    if cap < 2 {
        return None;
    }
    let upper = max_lag.min(cap);
    if upper < 3 {
        // Need at least three points to evaluate a local maximum.
        return None;
    }

    const THRESHOLD: f64 = 0.5;

    // Pre-compute autocorrelations for the scan range plus one neighbour on
    // each side so that we can test for a local maximum.
    let mut corrs: Vec<f64> = Vec::with_capacity(upper + 1);
    // Index `lag` holds the true autocorrelation at that lag. At lag 0 a series
    // is perfectly correlated with itself, so the value is unity by definition
    // (the `auto_correlation` helper rejects lag 0, hence the closed form here).
    corrs.push(1.0);
    corrs.push(auto_correlation(values, 1).unwrap_or(1.0));
    for lag in 2..=upper {
        let c = auto_correlation(values, lag).unwrap_or(0.0);
        corrs.push(c);
    }

    // Walk lags 2..=upper-1 and return the first strict local max above the
    // threshold. We require `corrs[lag] > corrs[lag-1]` AND
    // `corrs[lag] > corrs[lag+1]`.
    for lag in 2..upper {
        let c_prev = corrs[lag - 1];
        let c = corrs[lag];
        let c_next = corrs[lag + 1];
        if c > THRESHOLD && c > c_prev && c > c_next {
            return Some(lag);
        }
    }
    // Boundary case: lag = upper has no right neighbour we can compare to;
    // require monotonic ascent on the left side and accept it if the
    // correlation is strictly greater than the previous lag and above the
    // threshold.
    if upper >= 3 {
        let c_prev = corrs[upper - 1];
        let c = corrs[upper];
        if c > THRESHOLD && c > c_prev {
            // Only accept the boundary if the entire interior was below
            // threshold or strictly increasing - otherwise we'd return spurious
            // tail peaks.
            let mut monotonic = true;
            for lag in 3..upper {
                if corrs[lag] < corrs[lag - 1] {
                    monotonic = false;
                    break;
                }
            }
            if monotonic {
                return Some(upper);
            }
        }
    }
    None
}

/// Approximate two-sided normal quantile for a confidence level.
///
/// Only the three canonical levels are supported; arbitrary inputs are clamped
/// to the closest of `0.90`, `0.95`, `0.99` (returning `1.645`, `1.96`, or
/// `2.576` respectively).
pub fn z_score_for(confidence: f64) -> f64 {
    // Canonical anchors -> z-scores.
    const ANCHORS: [(f64, f64); 3] = [(0.90, 1.645), (0.95, 1.96), (0.99, 2.576)];
    let mut best_idx = 0usize;
    let mut best_dist = (confidence - ANCHORS[0].0).abs();
    for (i, (level, _)) in ANCHORS.iter().enumerate().skip(1) {
        let d = (confidence - level).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    ANCHORS[best_idx].1
}

/// Validate the common prerequisites of every forecaster: enough samples and a
/// finite-valued series.
fn validate_series(series: &TimeSeries, min_samples: usize) -> Result<()> {
    if series.is_empty() {
        return Err(OptimError::InvalidParameter(
            "TimeSeries is empty; cannot fit forecaster".to_string(),
        ));
    }
    if series.len() < min_samples {
        return Err(OptimError::InvalidParameter(format!(
            "TimeSeries has {} samples but min_samples = {min_samples}",
            series.len()
        )));
    }
    for (i, &v) in series.values.iter().enumerate() {
        if !v.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "TimeSeries value at index {i} is not finite: {v}"
            )));
        }
    }
    Ok(())
}

/// Compute RMS of a residual slice (root-mean-square, denominator `n`).
fn rms(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let mut s = 0.0_f64;
    for r in residuals {
        s += r * r;
    }
    (s / residuals.len() as f64).sqrt()
}

/// Build a [`ForecastResult`] given an iterator of predicted values, the
/// residual standard deviation, the confidence level, and the method label.
///
/// The CI half-width at horizon `h` is `z * residual_stddev * sqrt(h)`. The
/// caller is responsible for supplying the predictions in horizon order.
fn build_forecast_result(
    method: &str,
    predictions: Vec<f64>,
    residual_stddev: f64,
    confidence_level: f64,
) -> ForecastResult {
    let z = z_score_for(confidence_level);
    let mut points = Vec::with_capacity(predictions.len());
    for (i, p) in predictions.into_iter().enumerate() {
        let h = i + 1;
        let half = z * residual_stddev * (h as f64).sqrt();
        points.push(ForecastPoint {
            horizon_step: h,
            predicted_value: p,
            lower_bound: p - half,
            upper_bound: p + half,
        });
    }
    ForecastResult {
        method: method.to_string(),
        points,
        fit_residual_stddev: residual_stddev,
        confidence_level,
    }
}

// -----------------------------------------------------------------------------
// MovingAverageForecaster
// -----------------------------------------------------------------------------

/// Constant-prediction forecaster equal to the mean of the trailing window.
///
/// The width of the window is configurable; the residual standard deviation is
/// computed from the in-sample one-step-ahead residual (current value minus
/// rolling mean of the previous window).
#[derive(Debug, Clone)]
pub struct MovingAverageForecaster {
    config: ForecastConfig,
    window_size: usize,
    last_mean: Option<f64>,
    residual_stddev: f64,
    fitted: bool,
}

impl MovingAverageForecaster {
    /// Create a forecaster with the supplied window size (default 10).
    pub fn new(window_size: usize) -> Self {
        Self {
            config: ForecastConfig::default(),
            window_size: window_size.max(1),
            last_mean: None,
            residual_stddev: 0.0,
            fitted: false,
        }
    }

    /// Override the shared configuration.
    pub fn with_config(mut self, config: ForecastConfig) -> Self {
        self.config = config;
        self
    }

    /// Window size used during fitting.
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

impl Default for MovingAverageForecaster {
    fn default() -> Self {
        Self::new(10)
    }
}

impl PerformanceForecaster for MovingAverageForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series(series, self.config.min_samples)?;
        let values = series.values();
        let n = values.len();
        let w = self.window_size.min(n);

        // Mean of the trailing window.
        let window = &values[n - w..];
        let m = mean(window);
        self.last_mean = Some(m);

        // In-sample residuals: predict y_t = mean(y_{t-w..t}); compute residual
        // for every index where a full window is available.
        let mut residuals = Vec::new();
        if n > w {
            for t in w..n {
                let win = &values[t - w..t];
                let pred = mean(win);
                residuals.push(values[t] - pred);
            }
        }
        self.residual_stddev = rms(&residuals);
        self.fitted = true;
        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<ForecastResult> {
        if !self.fitted {
            return Err(OptimError::InvalidState(
                "MovingAverageForecaster::forecast called before fit".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(OptimError::InvalidParameter(
                "horizon must be >= 1".to_string(),
            ));
        }
        let m = self.last_mean.unwrap_or(0.0);
        let preds = vec![m; horizon];
        Ok(build_forecast_result(
            self.method_name(),
            preds,
            self.residual_stddev,
            self.config.confidence_level,
        ))
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn method_name(&self) -> &str {
        "moving_average"
    }
}

// -----------------------------------------------------------------------------
// ExponentialSmoothingForecaster (single / EWMA)
// -----------------------------------------------------------------------------

/// Single exponential smoothing forecaster.
///
/// Updates a running `level_t = alpha * y_t + (1 - alpha) * level_{t-1}`,
/// initialising `level_0 = y_0`. The forecast at every horizon equals the
/// final level; the CI widens with `sqrt(h)`.
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingForecaster {
    config: ForecastConfig,
    alpha: f64,
    level: Option<f64>,
    residual_stddev: f64,
    fitted: bool,
}

impl ExponentialSmoothingForecaster {
    /// Create a forecaster with smoothing parameter `alpha` (default 0.3).
    pub fn new(alpha: f64) -> Self {
        Self {
            config: ForecastConfig::default(),
            alpha: alpha.clamp(0.0, 1.0),
            level: None,
            residual_stddev: 0.0,
            fitted: false,
        }
    }

    /// Override the shared configuration.
    pub fn with_config(mut self, config: ForecastConfig) -> Self {
        self.config = config;
        self
    }

    /// Smoothing factor `alpha`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Current level estimate (only meaningful after fitting).
    pub fn level(&self) -> Option<f64> {
        self.level
    }
}

impl Default for ExponentialSmoothingForecaster {
    fn default() -> Self {
        Self::new(0.3)
    }
}

impl PerformanceForecaster for ExponentialSmoothingForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series(series, self.config.min_samples)?;
        let values = series.values();
        let alpha = self.alpha;
        let mut level = values[0];
        let mut residuals: Vec<f64> = Vec::with_capacity(values.len().saturating_sub(1));
        for &y in values.iter().skip(1) {
            // One-step-ahead prediction at t is the previous level.
            let pred = level;
            residuals.push(y - pred);
            level = alpha * y + (1.0 - alpha) * level;
        }
        self.level = Some(level);
        self.residual_stddev = rms(&residuals);
        self.fitted = true;
        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<ForecastResult> {
        if !self.fitted {
            return Err(OptimError::InvalidState(
                "ExponentialSmoothingForecaster::forecast called before fit".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(OptimError::InvalidParameter(
                "horizon must be >= 1".to_string(),
            ));
        }
        let l = self.level.unwrap_or(0.0);
        let preds = vec![l; horizon];
        Ok(build_forecast_result(
            self.method_name(),
            preds,
            self.residual_stddev,
            self.config.confidence_level,
        ))
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn method_name(&self) -> &str {
        "exponential_smoothing"
    }
}

// -----------------------------------------------------------------------------
// HoltLinearForecaster (double exponential smoothing)
// -----------------------------------------------------------------------------

/// Double exponential smoothing forecaster (Holt's linear method).
///
/// Tracks an additive trend on top of the level. The level/trend recurrences
/// are:
///   * `level_t = alpha * y_t + (1 - alpha) * (level_{t-1} + trend_{t-1})`
///   * `trend_t = beta  * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}`
///
/// Initial conditions: `level_0 = y_0`, `trend_0 = y_1 - y_0` if `n >= 2`,
/// otherwise zero.
#[derive(Debug, Clone)]
pub struct HoltLinearForecaster {
    config: ForecastConfig,
    alpha: f64,
    beta: f64,
    level: Option<f64>,
    trend: Option<f64>,
    residual_stddev: f64,
    fitted: bool,
}

impl HoltLinearForecaster {
    /// Create a Holt linear forecaster with the supplied smoothing factors.
    /// Defaults: `alpha = 0.3`, `beta = 0.1`.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            config: ForecastConfig::default(),
            alpha: alpha.clamp(0.0, 1.0),
            beta: beta.clamp(0.0, 1.0),
            level: None,
            trend: None,
            residual_stddev: 0.0,
            fitted: false,
        }
    }

    /// Override the shared configuration.
    pub fn with_config(mut self, config: ForecastConfig) -> Self {
        self.config = config;
        self
    }

    /// Smoothing factor for the level component.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Smoothing factor for the trend component.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Current level (only meaningful after fitting).
    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// Current trend (only meaningful after fitting).
    pub fn trend(&self) -> Option<f64> {
        self.trend
    }
}

impl Default for HoltLinearForecaster {
    fn default() -> Self {
        Self::new(0.3, 0.1)
    }
}

impl PerformanceForecaster for HoltLinearForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series(series, self.config.min_samples.max(2))?;
        let values = series.values();
        let n = values.len();
        let alpha = self.alpha;
        let beta = self.beta;

        let mut level = values[0];
        let mut trend = if n >= 2 { values[1] - values[0] } else { 0.0 };
        let mut residuals: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));
        for &y in values.iter().skip(1) {
            // One-step-ahead prediction at t given history up to t-1.
            let pred = level + trend;
            residuals.push(y - pred);
            let prev_level = level;
            level = alpha * y + (1.0 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
        }
        self.level = Some(level);
        self.trend = Some(trend);
        self.residual_stddev = rms(&residuals);
        self.fitted = true;
        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<ForecastResult> {
        if !self.fitted {
            return Err(OptimError::InvalidState(
                "HoltLinearForecaster::forecast called before fit".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(OptimError::InvalidParameter(
                "horizon must be >= 1".to_string(),
            ));
        }
        let l = self.level.unwrap_or(0.0);
        let b = self.trend.unwrap_or(0.0);
        let mut preds = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            preds.push(l + (h as f64) * b);
        }
        Ok(build_forecast_result(
            self.method_name(),
            preds,
            self.residual_stddev,
            self.config.confidence_level,
        ))
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn method_name(&self) -> &str {
        "holt_linear"
    }
}

// -----------------------------------------------------------------------------
// HoltWintersForecaster (triple exponential smoothing, additive seasonality)
// -----------------------------------------------------------------------------

/// Triple exponential smoothing forecaster (Hyndman additive Holt-Winters).
///
/// State variables:
/// * `level_t`     - running additive level estimate.
/// * `trend_t`     - running additive trend estimate.
/// * `seasonal_t`  - rolling seasonal index of length `period`.
///
/// Initialisation follows Hyndman:
/// * `level_0 = mean(y_0..y_{p-1})`
/// * `trend_0 = (mean(y_p..y_{2p-1}) - mean(y_0..y_{p-1})) / p`
/// * `seasonal_i = y_i - level_0`  for `i in 0..p`
///
/// The model requires `series.len() >= 2 * period` for stable initialisation.
#[derive(Debug, Clone)]
pub struct HoltWintersForecaster {
    config: ForecastConfig,
    alpha: f64,
    beta: f64,
    gamma: f64,
    period: Option<usize>,
    level: Option<f64>,
    trend: Option<f64>,
    seasonal: Vec<f64>,
    /// Number of observations consumed during fit; used for forecasting the
    /// correct seasonal index for horizon `h`.
    last_index: Option<usize>,
    residual_stddev: f64,
    fitted: bool,
}

impl HoltWintersForecaster {
    /// Create a Holt-Winters forecaster with the supplied smoothing factors.
    /// Defaults: `alpha = 0.3`, `beta = 0.1`, `gamma = 0.1`.
    pub fn new(alpha: f64, beta: f64, gamma: f64) -> Self {
        Self {
            config: ForecastConfig::default(),
            alpha: alpha.clamp(0.0, 1.0),
            beta: beta.clamp(0.0, 1.0),
            gamma: gamma.clamp(0.0, 1.0),
            period: None,
            level: None,
            trend: None,
            seasonal: Vec::new(),
            last_index: None,
            residual_stddev: 0.0,
            fitted: false,
        }
    }

    /// Override the shared configuration. The seasonal period **must** be set
    /// on the config (`config.seasonality_period = Some(p)`) before fitting.
    pub fn with_config(mut self, config: ForecastConfig) -> Self {
        self.config = config;
        self
    }

    /// Smoothing factor for the level component.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Smoothing factor for the trend component.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Smoothing factor for the seasonal component.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Seasonal period actually used during fitting.
    pub fn period(&self) -> Option<usize> {
        self.period
    }

    /// Current level (only meaningful after fitting).
    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// Current trend (only meaningful after fitting).
    pub fn trend(&self) -> Option<f64> {
        self.trend
    }

    /// Seasonal component vector (length = period after fitting).
    pub fn seasonal_components(&self) -> &[f64] {
        &self.seasonal
    }
}

impl Default for HoltWintersForecaster {
    fn default() -> Self {
        Self::new(0.3, 0.1, 0.1)
    }
}

impl PerformanceForecaster for HoltWintersForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let period = self.config.seasonality_period.ok_or_else(|| {
            OptimError::InvalidParameter(
                "HoltWintersForecaster requires config.seasonality_period = Some(p)".to_string(),
            )
        })?;
        if period < 2 {
            return Err(OptimError::InvalidParameter(format!(
                "HoltWintersForecaster seasonality_period must be >= 2, got {period}"
            )));
        }
        // Require enough data for stable initialisation.
        let min_required = (2 * period).max(self.config.min_samples);
        validate_series(series, min_required)?;
        let values = series.values();
        let n = values.len();
        if n < 2 * period {
            return Err(OptimError::InvalidParameter(format!(
                "HoltWintersForecaster needs at least 2 * period = {} samples, got {n}",
                2 * period
            )));
        }

        // Hyndman additive initialisation -----------------------------------
        let mean_first: f64 = values[..period].iter().copied().sum::<f64>() / period as f64;
        let mean_second: f64 =
            values[period..2 * period].iter().copied().sum::<f64>() / period as f64;
        let mut level = mean_first;
        let mut trend = (mean_second - mean_first) / period as f64;
        let mut seasonal: Vec<f64> = (0..period).map(|i| values[i] - level).collect();

        let alpha = self.alpha;
        let beta = self.beta;
        let gamma = self.gamma;

        // Recurrences ------------------------------------------------------
        let mut residuals: Vec<f64> = Vec::with_capacity(n.saturating_sub(period));
        for t in period..n {
            let s_tp = seasonal[t % period];
            // One-step-ahead prediction at time t given history up to t-1.
            let pred = level + trend + s_tp;
            residuals.push(values[t] - pred);

            let prev_level = level;
            level = alpha * (values[t] - s_tp) + (1.0 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
            let new_s = gamma * (values[t] - level) + (1.0 - gamma) * s_tp;
            seasonal[t % period] = new_s;
        }

        self.period = Some(period);
        self.level = Some(level);
        self.trend = Some(trend);
        self.seasonal = seasonal;
        self.last_index = Some(n);
        self.residual_stddev = rms(&residuals);
        self.fitted = true;
        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<ForecastResult> {
        if !self.fitted {
            return Err(OptimError::InvalidState(
                "HoltWintersForecaster::forecast called before fit".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(OptimError::InvalidParameter(
                "horizon must be >= 1".to_string(),
            ));
        }
        let period = self.period.ok_or_else(|| {
            OptimError::InvalidState("HoltWintersForecaster period missing".to_string())
        })?;
        let last_index = self.last_index.ok_or_else(|| {
            OptimError::InvalidState("HoltWintersForecaster last_index missing".to_string())
        })?;
        let l = self.level.unwrap_or(0.0);
        let b = self.trend.unwrap_or(0.0);

        let mut preds = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            // Seasonal index for horizon h: position (last_index + h - 1) mod period.
            let idx = (last_index + h - 1) % period;
            let s = self.seasonal.get(idx).copied().unwrap_or(0.0);
            preds.push(l + (h as f64) * b + s);
        }
        Ok(build_forecast_result(
            self.method_name(),
            preds,
            self.residual_stddev,
            self.config.confidence_level,
        ))
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn method_name(&self) -> &str {
        "holt_winters"
    }
}

// -----------------------------------------------------------------------------
// Generic-Float interop note
// -----------------------------------------------------------------------------
//
// The forecasters above operate on `f64` for numerical stability of the
// in-sample residual stddev. For callers driving generic-Float pipelines we
// expose a tiny conversion helper that downcasts a `Float`-typed value to
// `f64` without loss for the common scalar types.

/// Convert a generic [`Float`] value to `f64`, using `to_f64()`.
pub fn float_to_f64<F: Float>(v: F) -> f64 {
    v.to_f64().unwrap_or(0.0)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn series_from_values(name: &str, values: &[f64]) -> TimeSeries {
        let mut s = TimeSeries::new(name);
        for (i, &v) in values.iter().enumerate() {
            s.push(v, i as f64).expect("push should succeed");
        }
        s
    }

    // ---- TimeSeries -------------------------------------------------------

    #[test]
    fn test_time_series_basic_construction() {
        let mut s = TimeSeries::new("wall_clock");
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        s.push(1.0, 0.0).expect("push 1");
        s.push(2.0, 1.0).expect("push 2");
        assert_eq!(s.len(), 2);
        assert_eq!(s.values(), &[1.0, 2.0]);
        assert_eq!(s.timestamps(), &[0.0, 1.0]);
        assert_eq!(s.metric_name, "wall_clock");
    }

    #[test]
    fn test_time_series_push_validates_monotonic_timestamps() {
        let mut s = TimeSeries::new("metric");
        s.push(1.0, 10.0).expect("first push ok");
        s.push(2.0, 10.0).expect("equal timestamp ok");
        let err = s.push(3.0, 5.0).expect_err("non-monotonic must error");
        match err {
            OptimError::InvalidParameter(_) => {}
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_time_series_push_rejects_non_finite() {
        let mut s = TimeSeries::new("m");
        let err = s
            .push(f64::NAN, 0.0)
            .expect_err("non-finite value must error");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
        let err2 = s
            .push(1.0, f64::INFINITY)
            .expect_err("non-finite timestamp must error");
        assert!(matches!(err2, OptimError::InvalidParameter(_)));
    }

    // ---- Helper functions -------------------------------------------------

    #[test]
    fn test_mean_and_variance() {
        assert_eq!(mean(&[]), 0.0);
        assert_relative_eq!(mean(&[1.0, 2.0, 3.0]), 2.0, epsilon = 1e-12);
        // Sample variance with denominator (n-1):
        // values = [1, 2, 3], mean = 2, var = ((1)^2 + 0 + (1)^2) / 2 = 1
        assert_relative_eq!(variance(&[1.0, 2.0, 3.0]), 1.0, epsilon = 1e-12);
        // n=1 returns 0.
        assert_eq!(variance(&[7.0]), 0.0);
    }

    #[test]
    fn test_autocorrelation_lag_zero_is_invalid() {
        let vs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let err = auto_correlation(&vs, 0).expect_err("lag=0 must error");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
    }

    #[test]
    fn test_autocorrelation_periodic_signal_detects_period() {
        // Sine with period 10 over 60 samples.
        let n = 60usize;
        let period = 10usize;
        let vs: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * (i as f64) / period as f64).sin())
            .collect();
        let c10 = auto_correlation(&vs, period).expect("auto-correlation ok");
        let c5 = auto_correlation(&vs, period / 2).expect("auto-correlation ok");
        assert!(c10 > 0.8, "lag-10 correlation should be high, got {c10}");
        assert!(
            c5 < 0.0,
            "lag-5 correlation should be strongly negative for sine, got {c5}"
        );
    }

    #[test]
    fn test_detect_seasonality_period_finds_correct_period() {
        // Synthetic seasonal data: y = sin(2 pi t / 12) over 60 samples.
        let n = 60usize;
        let period = 12usize;
        let vs: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * (i as f64) / period as f64).sin())
            .collect();
        let detected = detect_seasonality_period(&vs, 24).expect("should detect");
        assert_eq!(detected, period);
    }

    #[test]
    fn test_detect_seasonality_returns_none_for_random_data() {
        // Pseudo-random sequence via a small linear congruential generator -
        // deterministic but exhibits no periodicity at the scale we scan.
        let n = 128usize;
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let vs: Vec<f64> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 33) as f64) / ((1u64 << 31) as f64);
                u - 1.0
            })
            .collect();
        let detected = detect_seasonality_period(&vs, 16);
        assert!(
            detected.is_none(),
            "expected None for noisy data, got {detected:?}"
        );
    }

    #[test]
    fn test_detect_seasonality_short_series_returns_none() {
        let vs = [1.0, 2.0, 3.0];
        assert!(detect_seasonality_period(&vs, 4).is_none());
        let vs2 = [1.0, 2.0, 3.0, 4.0, 5.0];
        // max_lag < 2 forbidden.
        assert!(detect_seasonality_period(&vs2, 1).is_none());
    }

    #[test]
    fn test_z_score_for_95_is_1_96() {
        assert_relative_eq!(z_score_for(0.95), 1.96, epsilon = 1e-12);
        assert_relative_eq!(z_score_for(0.99), 2.576, epsilon = 1e-12);
        assert_relative_eq!(z_score_for(0.90), 1.645, epsilon = 1e-12);
        // Clamping: 0.97 is closer to 0.95 than to 0.99.
        assert_relative_eq!(z_score_for(0.97), 1.96, epsilon = 1e-12);
        // 0.92 is closer to 0.90.
        assert_relative_eq!(z_score_for(0.92), 1.645, epsilon = 1e-12);
    }

    // ---- Moving average ---------------------------------------------------

    #[test]
    fn test_moving_average_constant_series_predicts_constant() {
        let s = series_from_values("c", &[5.0; 10]);
        let mut f = MovingAverageForecaster::new(5);
        f.fit(&s).expect("fit ok");
        let out = f.forecast(3).expect("forecast ok");
        assert_eq!(out.points.len(), 3);
        for (i, p) in out.points.iter().enumerate() {
            assert_eq!(p.horizon_step, i + 1);
            assert_relative_eq!(p.predicted_value, 5.0, epsilon = 1e-12);
            // Residual stddev should be zero for a constant series.
            assert_relative_eq!(p.lower_bound, 5.0, epsilon = 1e-12);
            assert_relative_eq!(p.upper_bound, 5.0, epsilon = 1e-12);
        }
        assert_relative_eq!(out.fit_residual_stddev, 0.0, epsilon = 1e-12);
        assert_relative_eq!(out.confidence_level, 0.95, epsilon = 1e-12);
        assert_eq!(out.method, "moving_average");
    }

    #[test]
    fn test_moving_average_decreasing_trend_predicts_mean() {
        // Window size 10 over [10, 9, ..., 1] -> mean = 5.5
        let vs: Vec<f64> = (1..=10).rev().map(|i| i as f64).collect();
        let s = series_from_values("dec", &vs);
        let mut f = MovingAverageForecaster::new(10);
        f.fit(&s).expect("fit ok");
        let out = f.forecast(2).expect("forecast ok");
        assert_relative_eq!(out.points[0].predicted_value, 5.5, epsilon = 1e-9);
        assert_relative_eq!(out.points[1].predicted_value, 5.5, epsilon = 1e-9);
    }

    // ---- Exponential smoothing -------------------------------------------

    #[test]
    fn test_exponential_smoothing_recovers_recent_level() {
        // Step function 1->5 should bias level toward 5 because alpha=0.3 still
        // weights recent observations heavily after several updates.
        let vs = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let s = series_from_values("step", &vs);
        let mut f = ExponentialSmoothingForecaster::new(0.3);
        f.fit(&s).expect("fit ok");
        let out = f.forecast(1).expect("forecast ok");
        let predicted = out.points[0].predicted_value;
        // It cannot reach 5 with only 5 steps but must be > 3 (above the
        // halfway point) and < 5.
        assert!(
            predicted > 3.0 && predicted < 5.0,
            "predicted={predicted} should be between 3 and 5"
        );
        // And the prediction must be strictly closer to 5 than to 1.
        assert!(
            (predicted - 5.0).abs() < (predicted - 1.0).abs(),
            "predicted={predicted} should be closer to 5 than to 1"
        );
    }

    // ---- Holt linear ------------------------------------------------------

    #[test]
    fn test_holt_linear_recovers_trend() {
        // y = 2 + 0.5 * t for t in 0..30
        let vs: Vec<f64> = (0..30).map(|i| 2.0 + 0.5 * i as f64).collect();
        let s = series_from_values("lin", &vs);
        let mut f = HoltLinearForecaster::new(0.6, 0.4);
        f.fit(&s).expect("fit ok");
        let out = f.forecast(5).expect("forecast ok");
        // After fitting, level should be near y[29] = 2 + 0.5*29 = 16.5 and
        // trend near 0.5. So forecast at h should be near 16.5 + h*0.5.
        for (i, p) in out.points.iter().enumerate() {
            let h = (i + 1) as f64;
            let expected = 2.0 + 0.5 * (29.0 + h);
            assert!(
                (p.predicted_value - expected).abs() < 0.5,
                "horizon {h}: predicted {} vs expected {}",
                p.predicted_value,
                expected
            );
        }
    }

    #[test]
    fn test_holt_linear_ci_widens_with_horizon() {
        // Generate a noisy trend so the residual stddev is non-zero.
        let n = 40usize;
        let vs: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                2.0 + 0.5 * t + 0.05 * (t * 0.7).sin()
            })
            .collect();
        let s = series_from_values("noisy_lin", &vs);
        let mut f = HoltLinearForecaster::new(0.4, 0.2);
        f.fit(&s).expect("fit ok");
        assert!(f.is_fitted());
        let out = f.forecast(8).expect("forecast ok");
        // CI half-widths should grow strictly with horizon (sqrt(h)).
        let mut prev_half = -1.0;
        for p in &out.points {
            let half = p.upper_bound - p.predicted_value;
            assert!(half >= prev_half, "CI half-width should be non-decreasing");
            prev_half = half;
        }
        // Last CI must be strictly larger than the first (since stddev > 0).
        let first = out.points[0].upper_bound - out.points[0].predicted_value;
        let last = out.points.last().expect("nonempty").upper_bound
            - out.points.last().expect("nonempty").predicted_value;
        assert!(
            last > first,
            "last CI ({last}) should exceed first ({first})"
        );
    }

    // ---- Holt-Winters -----------------------------------------------------

    #[test]
    fn test_holt_winters_recovers_seasonal_pattern() {
        // y_t = 1.0 + sin(2 pi t / 12) for t in 0..36
        let period = 12usize;
        let n = 36usize;
        let vs: Vec<f64> = (0..n)
            .map(|i| 1.0 + (2.0 * PI * (i as f64) / period as f64).sin())
            .collect();
        let s = series_from_values("seasonal", &vs);
        let config = ForecastConfig {
            seasonality_period: Some(period),
            ..Default::default()
        };
        let mut f = HoltWintersForecaster::new(0.3, 0.05, 0.5).with_config(config);
        f.fit(&s).expect("fit ok");
        let out = f.forecast(period).expect("forecast ok");
        let expected: Vec<f64> = (0..period)
            .map(|h| {
                let t = (n + h) as f64;
                1.0 + (2.0 * PI * t / period as f64).sin()
            })
            .collect();
        let mae: f64 = out
            .points
            .iter()
            .zip(expected.iter())
            .map(|(p, e)| (p.predicted_value - *e).abs())
            .sum::<f64>()
            / period as f64;
        assert!(mae < 0.3, "Holt-Winters seasonal MAE too large: {mae}");
    }

    #[test]
    fn test_holt_winters_requires_two_periods() {
        let period = 7usize;
        let vs: Vec<f64> = (0..(2 * period - 1)).map(|i| i as f64).collect();
        let s = series_from_values("short", &vs);
        let config = ForecastConfig {
            seasonality_period: Some(period),
            min_samples: 2, // ensure validate_series doesn't fire first
            ..Default::default()
        };
        let mut f = HoltWintersForecaster::new(0.3, 0.1, 0.1).with_config(config);
        let err = f.fit(&s).expect_err("must error when n < 2 * period");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
    }

    #[test]
    fn test_holt_winters_requires_seasonality_period() {
        let vs: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let s = series_from_values("nosea", &vs);
        let mut f = HoltWintersForecaster::new(0.3, 0.1, 0.1);
        let err = f.fit(&s).expect_err("must require period");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
    }

    // ---- Cross-cutting tests ----------------------------------------------

    #[test]
    fn test_forecast_before_fit_errors() {
        let ma = MovingAverageForecaster::new(5);
        let err = ma.forecast(3).expect_err("forecast before fit");
        assert!(matches!(err, OptimError::InvalidState(_)));

        let es = ExponentialSmoothingForecaster::new(0.3);
        let err = es.forecast(3).expect_err("forecast before fit");
        assert!(matches!(err, OptimError::InvalidState(_)));

        let hl = HoltLinearForecaster::new(0.3, 0.1);
        let err = hl.forecast(3).expect_err("forecast before fit");
        assert!(matches!(err, OptimError::InvalidState(_)));

        let hw = HoltWintersForecaster::new(0.3, 0.1, 0.1);
        let err = hw.forecast(3).expect_err("forecast before fit");
        assert!(matches!(err, OptimError::InvalidState(_)));
    }

    #[test]
    fn test_forecast_zero_horizon_errors() {
        let s = series_from_values("c", &[1.0; 12]);
        let mut f = MovingAverageForecaster::new(5);
        f.fit(&s).expect("fit ok");
        let err = f.forecast(0).expect_err("horizon=0 must error");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
    }

    #[test]
    fn test_forecast_result_ci_widens_with_horizon() {
        // Use exponential smoothing on a slightly noisy series and check that
        // CI half-widths grow with horizon (sqrt(h)).
        let n = 30usize;
        let vs: Vec<f64> = (0..n)
            .map(|i| 3.0 + 0.1 * ((i as f64) * 0.5).sin())
            .collect();
        let s = series_from_values("noisy", &vs);
        let mut f = ExponentialSmoothingForecaster::new(0.3);
        f.fit(&s).expect("fit ok");
        let out = f.forecast(5).expect("forecast ok");
        let half_widths: Vec<f64> = out
            .points
            .iter()
            .map(|p| p.upper_bound - p.predicted_value)
            .collect();
        // Strictly increasing (since residual stddev > 0).
        for i in 1..half_widths.len() {
            assert!(
                half_widths[i] > half_widths[i - 1],
                "half-widths not increasing: {:?}",
                half_widths
            );
        }
        // Ratios should approximately follow sqrt(h)/sqrt(1) = sqrt(h).
        let h2_ratio = half_widths[1] / half_widths[0];
        assert!(
            (h2_ratio - (2.0_f64).sqrt()).abs() < 1e-9,
            "ratio at h=2: got {h2_ratio}, expected {}",
            (2.0_f64).sqrt()
        );
    }

    #[test]
    fn test_min_samples_validation_errors() {
        // Fewer samples than min_samples on each forecaster should error out.
        let vs = [1.0, 2.0, 3.0]; // 3 samples
        let s = series_from_values("tiny", &vs);

        let mut ma = MovingAverageForecaster::new(3); // default min_samples=10
        let err = ma.fit(&s).expect_err("not enough samples");
        assert!(matches!(err, OptimError::InvalidParameter(_)));

        let mut es = ExponentialSmoothingForecaster::new(0.3);
        let err = es.fit(&s).expect_err("not enough samples");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
    }

    #[test]
    fn test_constant_zero_series_handled_gracefully() {
        // Constant zero must fit without error and produce zero forecasts.
        let s = series_from_values("zero", &[0.0; 24]);
        let config = ForecastConfig {
            seasonality_period: Some(4),
            ..Default::default()
        };
        let methods: Vec<Box<dyn PerformanceForecaster>> = vec![
            Box::new(MovingAverageForecaster::new(6)),
            Box::new(ExponentialSmoothingForecaster::new(0.3)),
            Box::new(HoltLinearForecaster::new(0.3, 0.1)),
            Box::new(HoltWintersForecaster::new(0.3, 0.1, 0.1).with_config(config)),
        ];
        for mut m in methods {
            m.fit(&s).expect("fit zero series");
            let out = m.forecast(3).expect("forecast zero series");
            for p in &out.points {
                assert_relative_eq!(p.predicted_value, 0.0, epsilon = 1e-9);
                assert_relative_eq!(p.lower_bound, 0.0, epsilon = 1e-9);
                assert_relative_eq!(p.upper_bound, 0.0, epsilon = 1e-9);
            }
            assert_relative_eq!(out.fit_residual_stddev, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_forecast_config_default_values() {
        let c = ForecastConfig::default();
        assert_relative_eq!(c.confidence_level, 0.95, epsilon = 1e-12);
        assert_eq!(c.min_samples, 10);
        assert_eq!(c.seasonality_period, None);
    }

    #[test]
    fn test_moving_average_window_clamps_to_one() {
        // window_size=0 is illegal at the API level but is clamped to 1 for safety.
        let s = series_from_values("c", &[3.0; 12]);
        let mut f = MovingAverageForecaster::new(0);
        assert_eq!(f.window_size(), 1);
        f.fit(&s).expect("fit ok");
        let out = f.forecast(2).expect("forecast ok");
        assert_relative_eq!(out.points[0].predicted_value, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_method_names_are_unique() {
        let names = [
            MovingAverageForecaster::new(5).method_name().to_string(),
            ExponentialSmoothingForecaster::new(0.3)
                .method_name()
                .to_string(),
            HoltLinearForecaster::new(0.3, 0.1)
                .method_name()
                .to_string(),
            HoltWintersForecaster::new(0.3, 0.1, 0.1)
                .method_name()
                .to_string(),
        ];
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                assert_ne!(names[i], names[j], "method names must be unique");
            }
        }
    }

    #[test]
    fn test_float_to_f64_conversion() {
        let v_f32: f32 = 1.5;
        let v_f64: f64 = 2.25;
        assert_relative_eq!(float_to_f64(v_f32), 1.5, epsilon = 1e-6);
        assert_relative_eq!(float_to_f64(v_f64), 2.25, epsilon = 1e-12);
    }
}
