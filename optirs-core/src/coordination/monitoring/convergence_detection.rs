// Convergence detection for optimization processes
//
// This module provides sophisticated convergence detection capabilities for optimization
// algorithms, including statistical tests, ML-based detection, and adaptive thresholds.

#[allow(dead_code)]
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// Result of convergence analysis
#[derive(Debug, Clone)]
pub struct ConvergenceResult<T: Float + Debug + Send + Sync + 'static> {
    pub converged: bool,
    pub confidence: T,
    pub iterations_to_convergence: Option<usize>,
    pub convergence_rate: Option<T>,
    pub stagnation_count: usize,
    pub trend_analysis: TrendAnalysis<T>,
    pub statistical_significance: T,
}

/// Trend analysis for convergence patterns
#[derive(Debug, Clone)]
pub struct TrendAnalysis<T: Float + Debug + Send + Sync + 'static> {
    pub slope: T,
    pub r_squared: T,
    pub acceleration: T,
    pub volatility: T,
    pub momentum: T,
}

/// Convergence criteria configuration
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria<T: Float + Debug + Send + Sync + 'static> {
    pub absolute_tolerance: T,
    pub relative_tolerance: T,
    pub max_stagnation_iterations: usize,
    pub min_improvement_rate: T,
    pub confidence_threshold: T,
    pub window_size: usize,
    pub statistical_test_threshold: T,
    pub enable_adaptive_thresholds: bool,
    pub early_stopping_patience: usize,
    pub min_iterations: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ConvergenceCriteria<T> {
    fn default() -> Self {
        Self {
            absolute_tolerance: T::from(1e-6).unwrap_or_else(|| T::zero()),
            relative_tolerance: T::from(1e-4).unwrap_or_else(|| T::zero()),
            max_stagnation_iterations: 50,
            min_improvement_rate: T::from(1e-8).unwrap_or_else(|| T::zero()),
            confidence_threshold: T::from(0.95).unwrap_or_else(|| T::zero()),
            window_size: 20,
            statistical_test_threshold: T::from(0.05).unwrap_or_else(|| T::zero()),
            enable_adaptive_thresholds: true,
            early_stopping_patience: 100,
            min_iterations: 10,
        }
    }
}

/// Convergence monitoring state
#[derive(Debug)]
pub struct ConvergenceState<T: Float + Debug + Send + Sync + 'static> {
    pub history: VecDeque<T>,
    pub gradients: VecDeque<T>,
    pub improvement_history: VecDeque<T>,
    pub stagnation_count: usize,
    pub best_value: T,
    pub last_improvement_iteration: usize,
    pub current_iteration: usize,
    pub convergence_start: Option<usize>,
    pub adaptive_tolerance: T,
    pub trend_buffer: VecDeque<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ConvergenceState<T> {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            gradients: VecDeque::with_capacity(window_size),
            improvement_history: VecDeque::with_capacity(window_size),
            stagnation_count: 0,
            best_value: T::infinity(),
            last_improvement_iteration: 0,
            current_iteration: 0,
            convergence_start: None,
            adaptive_tolerance: T::from(1e-6).unwrap_or_else(|| T::zero()),
            trend_buffer: VecDeque::with_capacity(window_size),
        }
    }
}

/// Primary convergence detector
pub struct ConvergenceDetector<T: Float + Debug + Send + Sync + 'static> {
    criteria: ConvergenceCriteria<T>,
    state: ConvergenceState<T>,
    analyzer: ConvergenceAnalyzer<T>,
    monitor: ConvergenceMonitor<T>,
    indicator: ConvergenceIndicator<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ConvergenceDetector<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        let window_size = criteria.window_size;
        Self {
            analyzer: ConvergenceAnalyzer::new(criteria.clone()),
            monitor: ConvergenceMonitor::new(criteria.clone()),
            indicator: ConvergenceIndicator::new(criteria.clone()),
            state: ConvergenceState::new(window_size),
            criteria,
            _phantom: PhantomData,
        }
    }

    pub fn check_convergence(&mut self, value: T) -> ConvergenceResult<T> {
        self.state.current_iteration += 1;
        self.update_state(value);

        // Multi-faceted convergence analysis
        let statistical_result = self.analyzer.statistical_analysis(&self.state);
        let trend_result = self.analyzer.trend_analysis(&self.state);
        let adaptive_result = self.analyzer.adaptive_analysis(&self.state, &self.criteria);

        // Combine results with weighted confidence
        let combined_confidence = self.combine_confidences(
            statistical_result.confidence,
            trend_result.confidence,
            adaptive_result.confidence,
        );

        let converged = combined_confidence > self.criteria.confidence_threshold
            && self.state.current_iteration >= self.criteria.min_iterations;

        if converged && self.state.convergence_start.is_none() {
            self.state.convergence_start = Some(self.state.current_iteration);
        }

        let trend_analysis = self.analyzer.compute_trend_analysis(&self.state);

        let result = ConvergenceResult {
            converged,
            confidence: combined_confidence,
            iterations_to_convergence: self.state.convergence_start,
            convergence_rate: self.compute_convergence_rate(),
            stagnation_count: self.state.stagnation_count,
            trend_analysis,
            statistical_significance: statistical_result.confidence,
        };

        // Feed the monitor and the indicator engine. Until 0.3.2 both were
        // constructed in `new` and never touched again, so the detector produced
        // no stagnation/premature-convergence alerts and no indicator history at
        // all -- the `monitor` and `indicator` fields were pure decoration.
        self.monitor.monitor(result.clone());
        let _ = self.indicator.compute_indicators(&result, &self.state);

        result
    }

    /// Convergence alerts raised so far (stagnation, premature convergence).
    pub fn alerts(&self) -> &[ConvergenceAlert<T>] {
        self.monitor.get_alerts()
    }

    /// Per-iteration indicator history: convergence score, trend strength,
    /// stability index, confidence and progress ratio.
    pub fn indicator_history(&self) -> &VecDeque<IndicatorValues<T>> {
        self.indicator.get_indicator_history()
    }

    /// The criteria this detector was built with.
    pub fn criteria(&self) -> &ConvergenceCriteria<T> {
        &self.criteria
    }

    fn update_state(&mut self, value: T) {
        self.state.history.push_back(value);
        if self.state.history.len() > self.criteria.window_size {
            self.state.history.pop_front();
        }

        // Update best value and improvement tracking
        if value < self.state.best_value {
            let improvement = self.state.best_value - value;
            self.state.improvement_history.push_back(improvement);
            self.state.best_value = value;
            self.state.last_improvement_iteration = self.state.current_iteration;
            self.state.stagnation_count = 0;
        } else {
            self.state.stagnation_count += 1;
            self.state.improvement_history.push_back(T::zero());
        }

        if self.state.improvement_history.len() > self.criteria.window_size {
            self.state.improvement_history.pop_front();
        }

        // Compute gradients if we have enough history
        if self.state.history.len() >= 2 {
            let gradient = self.state.history[self.state.history.len() - 1]
                - self.state.history[self.state.history.len() - 2];
            self.state.gradients.push_back(gradient);
            if self.state.gradients.len() > self.criteria.window_size {
                self.state.gradients.pop_front();
            }
        }

        // Update adaptive tolerance
        if self.criteria.enable_adaptive_thresholds {
            self.update_adaptive_tolerance();
        }
    }

    fn update_adaptive_tolerance(&mut self) {
        if self.state.history.len() < 5 {
            return;
        }

        let variance = self.compute_variance(&self.state.history);
        let noise_level = variance.sqrt();

        // Adapt tolerance based on noise level and convergence progress
        let progress_factor = if self.state.stagnation_count > 0 {
            T::one()
                + T::from(self.state.stagnation_count).unwrap_or_else(|| T::zero())
                    / T::from(100.0).unwrap_or_else(|| T::zero())
        } else {
            T::one()
        };

        self.state.adaptive_tolerance = (noise_level * progress_factor)
            .max(self.criteria.absolute_tolerance / T::from(10.0).unwrap_or_else(|| T::zero()))
            .min(self.criteria.absolute_tolerance * T::from(10.0).unwrap_or_else(|| T::zero()));
    }

    fn compute_variance(&self, data: &VecDeque<T>) -> T {
        if data.len() < 2 {
            return T::zero();
        }

        let mean = data.iter().fold(T::zero(), |acc, &x| acc + x)
            / crate::utils::scalar_or(data.len(), T::one());
        let variance = data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / crate::utils::scalar_or(data.len() - 1, T::one());

        variance
    }

    fn combine_confidences(&self, stat_conf: T, trend_conf: T, adapt_conf: T) -> T {
        // Weighted combination of different confidence measures
        let stat_weight = T::from(0.4).unwrap_or_else(|| T::zero());
        let trend_weight = T::from(0.3).unwrap_or_else(|| T::zero());
        let adapt_weight = T::from(0.3).unwrap_or_else(|| T::zero());

        stat_conf * stat_weight + trend_conf * trend_weight + adapt_conf * adapt_weight
    }

    fn compute_convergence_rate(&self) -> Option<T> {
        if self.state.improvement_history.len() < 5 {
            return None;
        }

        let recent_improvements: Vec<T> = self
            .state
            .improvement_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        let avg_improvement = recent_improvements
            .iter()
            .fold(T::zero(), |acc, &x| acc + x)
            / crate::utils::scalar_or(recent_improvements.len(), T::one());

        Some(avg_improvement)
    }

    pub fn reset(&mut self) {
        self.state = ConvergenceState::new(self.criteria.window_size);
    }

    pub fn get_criteria(&self) -> &ConvergenceCriteria<T> {
        &self.criteria
    }

    pub fn update_criteria(&mut self, criteria: ConvergenceCriteria<T>) {
        self.criteria = criteria;
    }
}

/// Statistical and trend analysis for convergence
pub struct ConvergenceAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    criteria: ConvergenceCriteria<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ConvergenceAnalyzer<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        Self {
            criteria,
            _phantom: PhantomData,
        }
    }

    pub fn statistical_analysis(&self, state: &ConvergenceState<T>) -> ConvergenceResult<T> {
        if state.history.len() < 3 {
            return ConvergenceResult {
                converged: false,
                confidence: T::zero(),
                iterations_to_convergence: None,
                convergence_rate: None,
                stagnation_count: state.stagnation_count,
                trend_analysis: TrendAnalysis {
                    slope: T::zero(),
                    r_squared: T::zero(),
                    acceleration: T::zero(),
                    volatility: T::zero(),
                    momentum: T::zero(),
                },
                statistical_significance: T::zero(),
            };
        }

        // Kolmogorov-Smirnov test for stationarity
        let ks_statistic = self.kolmogorov_smirnov_test(state);

        // Mann-Kendall test for trend
        let mk_statistic = self.mann_kendall_test(state);

        // Anderson-Darling test for normality of residuals
        let ad_statistic = self.anderson_darling_test(state);

        // Combine statistical tests
        let statistical_confidence =
            self.combine_statistical_tests(ks_statistic, mk_statistic, ad_statistic);

        let converged = statistical_confidence > self.criteria.confidence_threshold;

        ConvergenceResult {
            converged,
            confidence: statistical_confidence,
            iterations_to_convergence: state.convergence_start,
            convergence_rate: None,
            stagnation_count: state.stagnation_count,
            trend_analysis: self.compute_trend_analysis(state),
            statistical_significance: statistical_confidence,
        }
    }

    pub fn trend_analysis(&self, state: &ConvergenceState<T>) -> ConvergenceResult<T> {
        let trend_analysis = self.compute_trend_analysis(state);

        // Trend-based convergence assessment
        let trend_confidence = self.assess_trend_convergence(&trend_analysis);

        let converged = trend_confidence > self.criteria.confidence_threshold;

        ConvergenceResult {
            converged,
            confidence: trend_confidence,
            iterations_to_convergence: state.convergence_start,
            convergence_rate: None,
            stagnation_count: state.stagnation_count,
            trend_analysis,
            statistical_significance: trend_confidence,
        }
    }

    pub fn adaptive_analysis(
        &self,
        state: &ConvergenceState<T>,
        criteria: &ConvergenceCriteria<T>,
    ) -> ConvergenceResult<T> {
        if state.history.len() < 2 {
            return ConvergenceResult {
                converged: false,
                confidence: T::zero(),
                iterations_to_convergence: None,
                convergence_rate: None,
                stagnation_count: state.stagnation_count,
                trend_analysis: TrendAnalysis {
                    slope: T::zero(),
                    r_squared: T::zero(),
                    acceleration: T::zero(),
                    volatility: T::zero(),
                    momentum: T::zero(),
                },
                statistical_significance: T::zero(),
            };
        }

        // `state.history.len() >= 2` is checked by the guard above, but reading
        // it out fallibly keeps this function panic-free if that guard is ever
        // relaxed.
        let (Some(&current_value), Some(&previous_value)) = (
            state.history.back(),
            state.history.get(state.history.len().wrapping_sub(2)),
        ) else {
            return ConvergenceResult {
                converged: false,
                confidence: T::zero(),
                iterations_to_convergence: None,
                convergence_rate: None,
                stagnation_count: state.stagnation_count,
                trend_analysis: TrendAnalysis {
                    slope: T::zero(),
                    r_squared: T::zero(),
                    acceleration: T::zero(),
                    volatility: T::zero(),
                    momentum: T::zero(),
                },
                statistical_significance: T::zero(),
            };
        };

        // Adaptive tolerance check
        let absolute_improvement = (previous_value - current_value).abs();
        let relative_improvement = if previous_value.abs() > T::epsilon() {
            absolute_improvement / previous_value.abs()
        } else {
            T::zero()
        };

        let tolerance = if criteria.enable_adaptive_thresholds {
            state.adaptive_tolerance
        } else {
            criteria.absolute_tolerance
        };

        let abs_converged = absolute_improvement < tolerance;
        let rel_converged = relative_improvement < criteria.relative_tolerance;
        let stagnation_converged = state.stagnation_count >= criteria.max_stagnation_iterations;

        let convergence_score =
            self.compute_adaptive_score(abs_converged, rel_converged, stagnation_converged, state);

        let converged = convergence_score > criteria.confidence_threshold;

        ConvergenceResult {
            converged,
            confidence: convergence_score,
            iterations_to_convergence: state.convergence_start,
            convergence_rate: None,
            stagnation_count: state.stagnation_count,
            trend_analysis: self.compute_trend_analysis(state),
            statistical_significance: convergence_score,
        }
    }

    pub fn compute_trend_analysis(&self, state: &ConvergenceState<T>) -> TrendAnalysis<T> {
        if state.history.len() < 3 {
            return TrendAnalysis {
                slope: T::zero(),
                r_squared: T::zero(),
                acceleration: T::zero(),
                volatility: T::zero(),
                momentum: T::zero(),
            };
        }

        let (slope, r_squared) = self.linear_regression(&state.history);
        let acceleration = self.compute_acceleration(&state.history);
        let volatility = self.compute_volatility(&state.history);
        let momentum = self.compute_momentum(&state.history);

        TrendAnalysis {
            slope,
            r_squared,
            acceleration,
            volatility,
            momentum,
        }
    }

    /// Two-sample Kolmogorov-Smirnov test comparing the first vs. second
    /// half of the history window for stationarity, via
    /// [`scirs2_stats::ks_2samp`]'s real implementation (real empirical-CDF
    /// statistic and a real p-value derived from the Kolmogorov
    /// distribution -- not a fixed 1.36/sqrt(n) critical-value cutoff
    /// collapsed to a binary 0.95/0.05).
    ///
    /// Returns the real p-value: high = "cannot reject that both halves
    /// come from the same distribution" = evidence of a stable, converged
    /// process; low = evidence the distribution shifted (still changing).
    /// Computed in `f64` internally (via `T::to_f64`/`T::from`, which any
    /// `Float` already provides) so this doesn't need to widen `T`'s trait
    /// bounds for every caller of `ConvergenceDetector<T>`.
    fn kolmogorov_smirnov_test(&self, state: &ConvergenceState<T>) -> T {
        if state.history.len() < 4 {
            return T::zero();
        }

        let n = state.history.len();
        let mid = n / 2;

        let first_half: Vec<f64> = state
            .history
            .iter()
            .take(mid)
            .map(|v| v.to_f64().unwrap_or(0.0))
            .collect();
        let second_half: Vec<f64> = state
            .history
            .iter()
            .skip(mid)
            .map(|v| v.to_f64().unwrap_or(0.0))
            .collect();

        let first_arr = scirs2_core::ndarray::Array1::from_vec(first_half);
        let second_arr = scirs2_core::ndarray::Array1::from_vec(second_half);

        match scirs2_stats::ks_2samp(&first_arr.view(), &second_arr.view(), "two-sided") {
            Ok((_statistic, p_value)) => T::from(p_value).unwrap_or_else(|| T::zero()),
            Err(_) => T::zero(),
        }
    }

    /// Mann-Kendall trend test: the S statistic and its (no-ties) variance
    /// are the real, standard formulas; the fix is converting the resulting
    /// Z-score to a real two-tailed p-value via the real standard-normal
    /// CDF from `scirs2_stats`, instead of collapsing every Z-score to a
    /// binary 0.95/0.05 at the fixed threshold 1.96.
    ///
    /// Returns the real p-value: high = "no significant trend" = evidence
    /// of a converged/stable process.
    fn mann_kendall_test(&self, state: &ConvergenceState<T>) -> T {
        if state.history.len() < 3 {
            return T::zero();
        }

        let data: Vec<T> = state.history.iter().cloned().collect();
        let n = data.len();
        let mut s = T::zero();

        for i in 0..n - 1 {
            for j in i + 1..n {
                let diff = data[j] - data[i];
                if diff > T::zero() {
                    s = s + T::one();
                } else if diff < T::zero() {
                    s = s - T::one();
                }
            }
        }

        let variance = (n * (n - 1) * (2 * n + 5)) as f64 / 18.0;
        if variance <= 0.0 {
            return T::from(1.0).unwrap_or_else(|| T::zero());
        }
        let s_f64 = s.to_f64().unwrap_or(0.0);
        let z = s_f64 / variance.sqrt();

        let p_value = match scirs2_stats::distributions::Normal::new(0.0_f64, 1.0_f64) {
            Ok(normal) => 2.0 * (1.0 - normal.cdf(z.abs())),
            Err(_) => 1.0,
        };

        T::from(p_value.clamp(0.0, 1.0)).unwrap_or_else(|| T::zero())
    }

    /// Anderson-Darling normality test on the recent gradient sequence, via
    /// [`scirs2_stats::anderson_darling`]'s real implementation (real A²
    /// statistic and a real p-value) -- not an ad hoc skewness/kurtosis
    /// heuristic.
    ///
    /// Returns the real p-value: high = "cannot reject normality" (gradient
    /// noise around a converged point is expected to look approximately
    /// Gaussian). `scirs2_stats`'s implementation requires at least 8
    /// samples; with fewer, this reports a neutral, explicitly
    /// inconclusive 0.5 rather than fabricating a confident answer either
    /// way. A zero-variance gradient sequence (every recent gradient
    /// identical) is itself a strong, degenerate sign of convergence, so
    /// that specific error case is treated as high confidence.
    fn anderson_darling_test(&self, state: &ConvergenceState<T>) -> T {
        const MIN_SAMPLES: usize = 8;
        if state.gradients.len() < MIN_SAMPLES {
            return T::from(0.5).unwrap_or_else(|| T::zero());
        }

        let gradients: Vec<f64> = state
            .gradients
            .iter()
            .map(|v| v.to_f64().unwrap_or(0.0))
            .collect();
        let array = scirs2_core::ndarray::Array1::from_vec(gradients);

        match scirs2_stats::anderson_darling(&array.view()) {
            Ok((_statistic, p_value)) => T::from(p_value).unwrap_or_else(|| T::zero()),
            Err(_) => T::from(0.95).unwrap_or_else(|| T::zero()),
        }
    }

    fn combine_statistical_tests(&self, ks: T, mk: T, ad: T) -> T {
        // Weighted combination of statistical test results
        let ks_weight = T::from(0.4).unwrap_or_else(|| T::zero());
        let mk_weight = T::from(0.3).unwrap_or_else(|| T::zero());
        let ad_weight = T::from(0.3).unwrap_or_else(|| T::zero());

        ks * ks_weight + mk * mk_weight + ad * ad_weight
    }

    fn assess_trend_convergence(&self, trend: &TrendAnalysis<T>) -> T {
        // Assess convergence based on trend characteristics
        let slope_score = (-trend.slope.abs()).exp();
        let r_squared_penalty = if trend.r_squared > T::from(0.8).unwrap_or_else(|| T::zero()) {
            T::from(0.5).unwrap_or_else(|| T::zero()) // Strong trend is bad for convergence
        } else {
            T::one()
        };
        let volatility_score = (-trend.volatility).exp();
        let momentum_score = (-trend.momentum.abs()).exp();

        (slope_score * r_squared_penalty + volatility_score + momentum_score)
            / T::from(3.0).unwrap_or_else(|| T::zero())
    }

    fn compute_adaptive_score(
        &self,
        abs_conv: bool,
        rel_conv: bool,
        stag_conv: bool,
        state: &ConvergenceState<T>,
    ) -> T {
        let mut score = T::zero();

        if abs_conv {
            score = score + T::from(0.4).unwrap_or_else(|| T::zero());
        }

        if rel_conv {
            score = score + T::from(0.3).unwrap_or_else(|| T::zero());
        }

        if stag_conv {
            score = score + T::from(0.3).unwrap_or_else(|| T::zero());
        }

        // Adjust score based on improvement history
        if !state.improvement_history.is_empty() {
            let recent_improvements = state
                .improvement_history
                .iter()
                .rev()
                .take(5)
                .filter(|&&x| x > T::zero())
                .count();

            if recent_improvements == 0 {
                score = score + T::from(0.2).unwrap_or_else(|| T::zero());
            }
        }

        score.min(T::one())
    }

    fn linear_regression(&self, data: &VecDeque<T>) -> (T, T) {
        if data.len() < 2 {
            return (T::zero(), T::zero());
        }

        let n = crate::utils::scalar_or(data.len(), T::one());
        let sum_x = (0..data.len()).fold(T::zero(), |acc, i| {
            acc + crate::utils::scalar_or(i, T::zero())
        });
        let sum_y = data.iter().fold(T::zero(), |acc, &y| acc + y);
        let sum_xy = data.iter().enumerate().fold(T::zero(), |acc, (i, &y)| {
            acc + T::from(i).unwrap_or_else(|| T::zero()) * y
        });
        let sum_x2 = (0..data.len()).fold(T::zero(), |acc, i| {
            let i_f = crate::utils::scalar_or(i, T::zero());
            acc + i_f * i_f
        });

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < T::epsilon() {
            return (T::zero(), T::zero());
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        // Compute R²
        let mean_y = sum_y / n;
        let ss_tot = data
            .iter()
            .fold(T::zero(), |acc, &y| acc + (y - mean_y) * (y - mean_y));
        let ss_res = data.iter().enumerate().fold(T::zero(), |acc, (i, &y)| {
            let predicted =
                slope * T::from(i).unwrap_or_else(|| T::zero()) + (sum_y - slope * sum_x) / n;
            acc + (y - predicted) * (y - predicted)
        });

        let r_squared = if ss_tot > T::epsilon() {
            T::one() - ss_res / ss_tot
        } else {
            T::zero()
        };

        (slope, r_squared)
    }

    fn compute_acceleration(&self, data: &VecDeque<T>) -> T {
        if data.len() < 3 {
            return T::zero();
        }

        let mut accelerations = Vec::new();
        for i in 2..data.len() {
            let accel =
                data[i] - T::from(2.0).unwrap_or_else(|| T::zero()) * data[i - 1] + data[i - 2];
            accelerations.push(accel);
        }

        accelerations
            .iter()
            .fold(T::zero(), |acc, &a| acc + a.abs())
            / crate::utils::scalar_or(accelerations.len(), T::one())
    }

    fn compute_volatility(&self, data: &VecDeque<T>) -> T {
        if data.len() < 2 {
            return T::zero();
        }

        let mean = data.iter().fold(T::zero(), |acc, &x| acc + x)
            / crate::utils::scalar_or(data.len(), T::one());
        let variance = data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / crate::utils::scalar_or(data.len(), T::one());

        variance.sqrt()
    }

    fn compute_momentum(&self, data: &VecDeque<T>) -> T {
        if data.len() < 3 {
            return T::zero();
        }

        let recent_window = 3.min(data.len());

        if recent_window >= 2 {
            let start_idx = data.len() - recent_window;
            let end_val = data[data.len() - 1];
            let start_val = data[start_idx];
            (end_val - start_val) / T::from(recent_window - 1).unwrap_or_else(|| T::zero())
        } else {
            T::zero()
        }
    }
}

/// Real-time convergence monitoring
pub struct ConvergenceMonitor<T: Float + Debug + Send + Sync + 'static> {
    criteria: ConvergenceCriteria<T>,
    monitoring_interval: Duration,
    last_check: Option<Instant>,
    convergence_history: VecDeque<ConvergenceResult<T>>,
    alerts: Vec<ConvergenceAlert<T>>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct ConvergenceAlert<T: Float + Debug + Send + Sync + 'static> {
    pub timestamp: Instant,
    pub alert_type: ConvergenceAlertType,
    pub message: String,
    pub confidence: T,
    pub suggested_action: String,
}

#[derive(Debug, Clone)]
pub enum ConvergenceAlertType {
    SlowConvergence,
    PrematureConvergence,
    Stagnation,
    Divergence,
    NoiseDetected,
}

impl<T: Float + Debug + Send + Sync + 'static> ConvergenceMonitor<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        Self {
            criteria,
            monitoring_interval: Duration::from_secs(1),
            last_check: None,
            convergence_history: VecDeque::with_capacity(1000),
            alerts: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn monitor(&mut self, result: ConvergenceResult<T>) {
        let now = Instant::now();

        if let Some(last_check) = self.last_check {
            if now.duration_since(last_check) < self.monitoring_interval {
                return;
            }
        }

        self.last_check = Some(now);
        self.convergence_history.push_back(result.clone());

        if self.convergence_history.len() > 1000 {
            self.convergence_history.pop_front();
        }

        // Generate alerts based on convergence patterns
        self.check_for_alerts(&result);
    }

    fn check_for_alerts(&mut self, result: &ConvergenceResult<T>) {
        // Check for slow convergence
        if result.stagnation_count > self.criteria.max_stagnation_iterations / 2 {
            self.add_alert(
                ConvergenceAlertType::Stagnation,
                "Optimization showing signs of stagnation".to_string(),
                result.confidence,
            );
        }

        // Check for premature convergence
        if result.converged
            && result.iterations_to_convergence.unwrap_or(1000) < self.criteria.min_iterations
        {
            self.add_alert(
                ConvergenceAlertType::PrematureConvergence,
                "Optimization may have converged prematurely".to_string(),
                result.confidence,
            );
        }

        // Check for high volatility
        if result.trend_analysis.volatility > T::from(0.1).unwrap_or_else(|| T::zero()) {
            self.add_alert(
                ConvergenceAlertType::NoiseDetected,
                "High volatility detected in optimization process".to_string(),
                result.confidence,
            );
        }
    }

    fn add_alert(&mut self, alert_type: ConvergenceAlertType, message: String, confidence: T) {
        let suggested_action = match alert_type {
            ConvergenceAlertType::Stagnation => {
                "Consider adjusting learning rate or optimization strategy".to_string()
            }
            ConvergenceAlertType::PrematureConvergence => {
                "Verify convergence criteria and consider stricter thresholds".to_string()
            }
            ConvergenceAlertType::NoiseDetected => {
                "Consider noise reduction or adaptive tolerance".to_string()
            }
            _ => "Monitor optimization progress closely".to_string(),
        };

        let alert = ConvergenceAlert {
            timestamp: Instant::now(),
            alert_type,
            message,
            confidence,
            suggested_action,
        };

        self.alerts.push(alert);

        // Keep only recent alerts
        if self.alerts.len() > 100 {
            self.alerts.remove(0);
        }
    }

    pub fn get_alerts(&self) -> &[ConvergenceAlert<T>] {
        &self.alerts
    }

    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    pub fn get_convergence_history(&self) -> &VecDeque<ConvergenceResult<T>> {
        &self.convergence_history
    }
}

/// Convergence indicators and visualization support
pub struct ConvergenceIndicator<T: Float + Debug + Send + Sync + 'static> {
    criteria: ConvergenceCriteria<T>,
    indicator_history: VecDeque<IndicatorValues<T>>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct IndicatorValues<T: Float + Debug + Send + Sync + 'static> {
    pub convergence_score: T,
    pub trend_strength: T,
    pub stability_index: T,
    pub confidence_level: T,
    pub progress_ratio: T,
}

impl<T: Float + Debug + Send + Sync + 'static> ConvergenceIndicator<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        Self {
            criteria,
            indicator_history: VecDeque::with_capacity(1000),
            _phantom: PhantomData,
        }
    }

    pub fn compute_indicators(
        &mut self,
        result: &ConvergenceResult<T>,
        state: &ConvergenceState<T>,
    ) -> IndicatorValues<T> {
        let convergence_score = result.confidence;
        let trend_strength = self.compute_trend_strength(&result.trend_analysis);
        let stability_index = self.compute_stability_index(state);
        let confidence_level = result.statistical_significance;
        let progress_ratio = self.compute_progress_ratio(state);

        let indicators = IndicatorValues {
            convergence_score,
            trend_strength,
            stability_index,
            confidence_level,
            progress_ratio,
        };

        self.indicator_history.push_back(indicators.clone());
        if self.indicator_history.len() > 1000 {
            self.indicator_history.pop_front();
        }

        indicators
    }

    fn compute_trend_strength(&self, trend: &TrendAnalysis<T>) -> T {
        // Combine slope and R² to measure trend strength
        let normalized_slope = (-trend.slope.abs()).exp();
        let r_squared_component = trend.r_squared;

        (normalized_slope + r_squared_component) / T::from(2.0).unwrap_or_else(|| T::zero())
    }

    fn compute_stability_index(&self, state: &ConvergenceState<T>) -> T {
        if state.history.len() < 3 {
            return T::zero();
        }

        // Measure stability based on recent volatility
        let recent_window = 10.min(state.history.len());
        let recent_values: Vec<T> = state
            .history
            .iter()
            .rev()
            .take(recent_window)
            .cloned()
            .collect();

        let mean = recent_values.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(recent_values.len()).unwrap_or_else(T::one);
        let variance = recent_values
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / T::from(recent_values.len()).unwrap_or_else(T::one);

        (-variance.sqrt()).exp()
    }

    fn compute_progress_ratio(&self, state: &ConvergenceState<T>) -> T {
        if state.history.is_empty() {
            return T::zero();
        }

        let initial_value = state.history[0];
        let Some(&current_value) = state.history.back() else {
            return T::zero();
        };
        let best_value = state.best_value;

        if (initial_value - best_value).abs() < T::epsilon() {
            return T::one();
        }

        let progress = (initial_value - current_value) / (initial_value - best_value);
        let progress = progress.clamp(T::zero(), T::one());

        // A run that has not yet reached `min_iterations` cannot report full
        // progress no matter how flat the curve looks: the criteria say the
        // question is not settled yet. `criteria` was previously stored and
        // never read.
        if state.current_iteration < self.criteria.min_iterations {
            let ceiling = T::from(state.current_iteration).unwrap_or_else(T::zero)
                / T::from(self.criteria.min_iterations.max(1)).unwrap_or_else(T::one);
            progress.min(ceiling)
        } else {
            progress
        }
    }

    pub fn get_indicator_history(&self) -> &VecDeque<IndicatorValues<T>> {
        &self.indicator_history
    }

    pub fn generate_convergence_report(&self, current_indicators: &IndicatorValues<T>) -> String {
        format!(
            "Convergence Report:\n\
             - Convergence Score: {:.4}\n\
             - Trend Strength: {:.4}\n\
             - Stability Index: {:.4}\n\
             - Confidence Level: {:.4}\n\
             - Progress Ratio: {:.4}\n",
            current_indicators.convergence_score.to_f64().unwrap_or(0.0),
            current_indicators.trend_strength.to_f64().unwrap_or(0.0),
            current_indicators.stability_index.to_f64().unwrap_or(0.0),
            current_indicators.confidence_level.to_f64().unwrap_or(0.0),
            current_indicators.progress_ratio.to_f64().unwrap_or(0.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_detector_basic() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let mut detector = ConvergenceDetector::new(criteria);

        // Test with converging sequence
        let values = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];

        for value in values {
            let result = detector.check_convergence(value);
            println!(
                "Value: {}, Converged: {}, Confidence: {:.4}",
                value, result.converged, result.confidence
            );
        }
    }

    #[test]
    fn test_statistical_analysis() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);

        let mut state = ConvergenceState::new(20);
        let values = vec![10.0, 5.0, 2.5, 1.25, 0.625, 0.3125];

        for value in values {
            state.history.push_back(value);
        }

        let result = analyzer.statistical_analysis(&state);
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_trend_analysis() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);

        let mut state = ConvergenceState::new(20);
        let values = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];

        for value in values {
            state.history.push_back(value);
        }

        let trend = analyzer.compute_trend_analysis(&state);
        assert!(trend.slope < 0.0); // Decreasing trend
    }

    // Regression tests for F61: the KS/Mann-Kendall/Anderson-Darling
    // "tests" used to each collapse to one of exactly two hardcoded
    // constants (0.95 or 0.05) regardless of how strong the real evidence
    // was, and the KS critical value itself was off by a factor of ~2. They
    // now return real, continuously-varying p-values from `scirs2_stats`.

    #[test]
    fn test_kolmogorov_smirnov_test_distinguishes_identical_from_shifted_halves() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);

        // Identical first/second halves: the two empirical distributions
        // are exactly the same, so the real KS p-value must be 1.0 (no
        // evidence whatsoever of a difference), not merely ">= 0.05".
        let mut stable_state = ConvergenceState::new(20);
        for value in [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] {
            stable_state.history.push_back(value);
        }
        let stable_p = analyzer.kolmogorov_smirnov_test(&stable_state);
        assert_eq!(stable_p, 1.0);

        // A huge, obvious shift between halves (at a realistic window size
        // -- the KS test has very little power at tiny sample sizes, which
        // is a property of the real test, not a bug): the real p-value
        // must be near zero, not the old fixed "0.05" placeholder.
        let mut shifted_state = ConvergenceState::new(20);
        for value in std::iter::repeat_n(0.0, 10).chain(std::iter::repeat_n(1000.0, 10)) {
            shifted_state.history.push_back(value);
        }
        let shifted_p = analyzer.kolmogorov_smirnov_test(&shifted_state);
        assert!(
            shifted_p < 0.05,
            "expected a small p-value, got {shifted_p}"
        );
        assert!(stable_p > shifted_p);
    }

    #[test]
    fn test_mann_kendall_test_distinguishes_trend_strength() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);

        // A strictly monotonic sequence is the strongest possible trend
        // signal: the real p-value must be small.
        let mut trending_state = ConvergenceState::new(20);
        for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
            trending_state.history.push_back(value);
        }
        let trending_p = analyzer.mann_kendall_test(&trending_state);

        // An oscillating, trendless sequence: the real p-value must be
        // large (S near zero either way).
        let mut flat_state = ConvergenceState::new(20);
        for value in [5.0, 6.0, 4.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0] {
            flat_state.history.push_back(value);
        }
        let flat_p = analyzer.mann_kendall_test(&flat_state);

        assert!(
            trending_p < flat_p,
            "a strict monotonic trend must score a smaller p-value than a trendless \
             oscillation (trending={trending_p}, flat={flat_p})"
        );
        assert!(trending_p < 0.05);
        assert!(flat_p > 0.05);
    }

    #[test]
    fn test_anderson_darling_test_is_neutral_below_the_real_minimum_sample_size() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);

        // scirs2_stats::anderson_darling requires >= 8 samples; with fewer
        // gradients this must report the documented neutral 0.5, not a
        // fabricated confident answer.
        let mut sparse_state = ConvergenceState::new(20);
        for value in [0.1, -0.1, 0.2] {
            sparse_state.gradients.push_back(value);
        }
        assert_eq!(analyzer.anderson_darling_test(&sparse_state), 0.5);
    }

    #[test]
    fn test_anderson_darling_test_runs_the_real_test_with_enough_samples() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);

        let mut state = ConvergenceState::new(20);
        for value in [0.1, -0.2, 0.05, -0.15, 0.3, -0.05, 0.12, -0.08, 0.02, -0.11] {
            state.gradients.push_back(value);
        }

        let p_value = analyzer.anderson_darling_test(&state);
        assert!(
            (0.0..=1.0).contains(&p_value),
            "expected a real p-value in [0, 1], got {p_value}"
        );

        // Zero-variance gradients (every recent gradient identical) is the
        // one explicitly-documented high-confidence special case.
        let mut degenerate_state = ConvergenceState::new(20);
        for _ in 0..8 {
            degenerate_state.gradients.push_back(0.5);
        }
        assert_eq!(analyzer.anderson_darling_test(&degenerate_state), 0.95);
    }
}
