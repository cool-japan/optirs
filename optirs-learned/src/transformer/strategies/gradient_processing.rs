use std::fmt::Debug;
// Gradient processing strategies for transformer optimization
//
// This module implements various gradient transformation and processing strategies
// used by the transformer optimizer to improve optimization performance.
//
// All stateful strategies (smoothing, accumulation, adaptive scaling) keep their
// state per parameter name, so that a model with parameters of differing shapes
// can be processed without shape conflicts.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};

use crate::error::Result;

/// Gradient processing strategies
#[derive(Debug, Clone, Copy)]
pub enum GradientProcessingStrategy {
    /// Raw gradients without processing
    Raw,
    /// Gradient clipping
    Clipping,
    /// Gradient normalization
    Normalization,
    /// Adaptive gradient scaling
    AdaptiveScaling,
    /// Adaptive processing (general)
    Adaptive,
    /// Gradient smoothing
    Smoothing,
    /// Gradient accumulation
    Accumulation,
    /// Gradient dropout
    Dropout,
    /// Gradient compression
    Compression,
}

/// Per-parameter processing state.
#[derive(Debug, Clone)]
struct PerParameterState<T: Float + Debug + Default + Clone + Send + Sync + 'static> {
    /// Gradient history for smoothing
    gradient_history: VecDeque<Array1<T>>,

    /// Accumulated gradients
    accumulated_gradients: Option<Array1<T>>,

    /// Statistics for this parameter only
    stats: GradientStatistics<T>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> PerParameterState<T> {
    fn new() -> Self {
        Self {
            gradient_history: VecDeque::new(),
            accumulated_gradients: None,
            stats: GradientStatistics::new(),
        }
    }

    /// Drop state whose width no longer matches the incoming gradient.
    fn ensure_width(&mut self, width: usize) {
        if self
            .gradient_history
            .back()
            .is_some_and(|g| g.len() != width)
        {
            self.gradient_history.clear();
        }
        if self
            .accumulated_gradients
            .as_ref()
            .is_some_and(|a| a.len() != width)
        {
            self.accumulated_gradients = None;
        }
    }
}

/// Gradient processor for transformer optimizer
#[derive(Debug, Clone)]
pub struct GradientProcessor<
    T: Float
        + Debug
        + Default
        + Clone
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
> {
    /// Processing strategy
    strategy: GradientProcessingStrategy,

    /// Per-parameter state (history, accumulation, statistics)
    param_states: HashMap<String, PerParameterState<T>>,

    /// Aggregate gradient statistics across all parameters
    gradient_stats: GradientStatistics<T>,

    /// Processing parameters
    processing_params: GradientProcessingParams<T>,
}

/// Gradient processing parameters
#[derive(Debug, Clone)]
pub struct GradientProcessingParams<T: Float + Debug + Send + Sync + 'static> {
    /// Clipping threshold
    pub clip_threshold: T,

    /// Smoothing factor: weight applied to the *new* gradient
    pub smoothing_factor: T,

    /// Accumulation steps
    pub accumulation_steps: usize,

    /// Dropout probability
    pub dropout_prob: f64,

    /// Compression ratio
    pub compression_ratio: f64,

    /// Normalization epsilon
    pub norm_eps: T,
}

/// Gradient statistics tracking
#[derive(Debug, Clone)]
pub struct GradientStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Running mean of gradient magnitudes
    mean_magnitude: T,

    /// Running sum of squared deviations (Welford aggregate)
    var_magnitude: T,

    /// Maximum gradient magnitude seen
    max_magnitude: T,

    /// Minimum gradient magnitude seen
    min_magnitude: T,

    /// Update count
    update_count: usize,

    /// Gradient sparsity
    sparsity: T,
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static,
    > GradientProcessor<T>
{
    /// Create new gradient processor
    pub fn new(strategy: GradientProcessingStrategy) -> Self {
        Self {
            strategy,
            param_states: HashMap::new(),
            gradient_stats: GradientStatistics::new(),
            processing_params: GradientProcessingParams::default(),
        }
    }

    /// Create with custom parameters
    pub fn new_with_params(
        strategy: GradientProcessingStrategy,
        params: GradientProcessingParams<T>,
    ) -> Self {
        Self {
            strategy,
            param_states: HashMap::new(),
            gradient_stats: GradientStatistics::new(),
            processing_params: params,
        }
    }

    /// Process gradients according to the selected strategy.
    ///
    /// `param_name` selects the per-parameter state slot; callers that only have a
    /// single parameter tensor may pass any stable identifier.
    pub fn process_gradients(
        &mut self,
        param_name: &str,
        gradients: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Update aggregate statistics first
        self.gradient_stats.update(gradients);

        let params = self.processing_params.clone();
        let strategy = self.strategy;

        let state = self
            .param_states
            .entry(param_name.to_string())
            .or_insert_with(PerParameterState::new);
        state.ensure_width(gradients.len());
        state.stats.update(gradients);

        match strategy {
            GradientProcessingStrategy::Raw => Ok(gradients.clone()),
            GradientProcessingStrategy::Clipping => Ok(Self::clip_gradients(&params, gradients)),
            GradientProcessingStrategy::Normalization => {
                Ok(Self::normalize_gradients(&params, gradients))
            }
            GradientProcessingStrategy::AdaptiveScaling | GradientProcessingStrategy::Adaptive => {
                Ok(Self::adaptive_scale_gradients(state, gradients))
            }
            GradientProcessingStrategy::Smoothing => {
                Ok(Self::smooth_gradients(state, &params, gradients))
            }
            GradientProcessingStrategy::Accumulation => {
                Ok(Self::accumulate_gradients(state, &params, gradients))
            }
            GradientProcessingStrategy::Dropout => Ok(Self::dropout_gradients(&params, gradients)),
            GradientProcessingStrategy::Compression => {
                Ok(Self::compress_gradients(&params, gradients))
            }
        }
    }

    /// Clip gradients to prevent explosion
    fn clip_gradients(params: &GradientProcessingParams<T>, gradients: &Array1<T>) -> Array1<T> {
        let grad_norm = Self::compute_gradient_norm(gradients);

        if grad_norm > params.clip_threshold && grad_norm > T::zero() {
            let scale = params.clip_threshold / grad_norm;
            gradients * scale
        } else {
            gradients.clone()
        }
    }

    /// Normalize gradients
    fn normalize_gradients(
        params: &GradientProcessingParams<T>,
        gradients: &Array1<T>,
    ) -> Array1<T> {
        let grad_norm = Self::compute_gradient_norm(gradients);

        if grad_norm > params.norm_eps {
            gradients / grad_norm
        } else {
            gradients.clone()
        }
    }

    /// Adaptively scale gradients based on this parameter's magnitude statistics.
    ///
    /// The scale is always strictly positive, so the descent direction is preserved.
    fn adaptive_scale_gradients(state: &PerParameterState<T>, gradients: &Array1<T>) -> Array1<T> {
        let current_norm = Self::compute_gradient_norm(gradients);
        let mean_norm = state.stats.mean_magnitude;

        // Guard against division by (near) zero: a vanishing gradient is passed
        // through unchanged instead of being blown up to infinity/NaN.
        let eps: T = scirs2_core::numeric::NumCast::from(1e-12).unwrap_or_else(T::zero);
        if mean_norm > T::zero() && current_norm > eps {
            let adaptive_scale = scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(T::zero)
                * mean_norm
                / current_norm
                + scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(T::zero);
            gradients * adaptive_scale
        } else {
            gradients.clone()
        }
    }

    /// Smooth gradients using an exponential moving average
    fn smooth_gradients(
        state: &mut PerParameterState<T>,
        params: &GradientProcessingParams<T>,
        gradients: &Array1<T>,
    ) -> Array1<T> {
        let alpha = params.smoothing_factor;

        let smoothed = match state.gradient_history.back() {
            Some(prev_grad) => gradients * alpha + prev_grad * (T::one() - alpha),
            None => gradients.clone(),
        };

        state.gradient_history.push_back(smoothed.clone());
        if state.gradient_history.len() > 10 {
            state.gradient_history.pop_front();
        }

        smoothed
    }

    /// Accumulate gradients over multiple steps
    fn accumulate_gradients(
        state: &mut PerParameterState<T>,
        params: &GradientProcessingParams<T>,
        gradients: &Array1<T>,
    ) -> Array1<T> {
        // Zero accumulation steps would make the modulo below panic.
        let steps = params.accumulation_steps.max(1);

        match state.accumulated_gradients {
            Some(ref mut accumulated) => *accumulated = accumulated.clone() + gradients,
            None => state.accumulated_gradients = Some(gradients.clone()),
        }

        if state.stats.update_count.is_multiple_of(steps) {
            match state.accumulated_gradients.take() {
                Some(accumulated) => {
                    let scale = scirs2_core::numeric::NumCast::from(1.0 / steps as f64)
                        .unwrap_or_else(T::one);
                    accumulated * scale
                }
                None => gradients.clone(),
            }
        } else {
            // Intermediate steps contribute nothing until the window closes.
            Array1::zeros(gradients.len())
        }
    }

    /// Apply a deterministic structured dropout pattern to gradients
    fn dropout_gradients(params: &GradientProcessingParams<T>, gradients: &Array1<T>) -> Array1<T> {
        let mut result = gradients.clone();
        let keep_prob = (1.0 - params.dropout_prob).max(f64::EPSILON);
        let scale: T = scirs2_core::numeric::NumCast::from(1.0 / keep_prob).unwrap_or_else(T::one);
        let dropped = (params.dropout_prob * 10.0) as usize;

        // Deterministic "dropout" pattern for reproducibility, with the inverted
        // dropout rescaling applied to the surviving entries.
        for (i, elem) in result.iter_mut().enumerate() {
            if (i % 10) < dropped {
                *elem = T::zero();
            } else {
                *elem = *elem * scale;
            }
        }

        result
    }

    /// Compress gradients (magnitude based sparsification)
    fn compress_gradients(
        params: &GradientProcessingParams<T>,
        gradients: &Array1<T>,
    ) -> Array1<T> {
        let mut result = gradients.clone();
        let threshold = Self::compute_gradient_norm(gradients)
            * scirs2_core::numeric::NumCast::from(params.compression_ratio).unwrap_or_else(T::zero);

        for elem in result.iter_mut() {
            if elem.abs() < threshold {
                *elem = T::zero();
            }
        }

        result
    }

    /// Compute L2 norm of gradients
    fn compute_gradient_norm(gradients: &Array1<T>) -> T {
        gradients
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |a, b| a + b)
            .sqrt()
    }

    /// Get aggregate gradient statistics
    pub fn statistics(&self) -> &GradientStatistics<T> {
        &self.gradient_stats
    }

    /// Get gradient statistics for a specific parameter
    pub fn statistics_for(&self, param_name: &str) -> Option<&GradientStatistics<T>> {
        self.param_states.get(param_name).map(|s| &s.stats)
    }

    /// Get the current processing strategy
    pub fn strategy(&self) -> GradientProcessingStrategy {
        self.strategy
    }

    /// Get the current processing parameters
    pub fn parameters(&self) -> &GradientProcessingParams<T> {
        &self.processing_params
    }

    /// Update processing strategy
    pub fn set_strategy(&mut self, strategy: GradientProcessingStrategy) {
        self.strategy = strategy;
    }

    /// Update processing parameters
    pub fn set_parameters(&mut self, params: GradientProcessingParams<T>) {
        self.processing_params = params;
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.param_states.clear();
        self.gradient_stats = GradientStatistics::new();
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> Default for GradientStatistics<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> GradientStatistics<T> {
    /// Create new gradient statistics
    pub fn new() -> Self {
        Self {
            mean_magnitude: T::zero(),
            var_magnitude: T::zero(),
            max_magnitude: T::zero(),
            min_magnitude: scirs2_core::numeric::NumCast::from(f64::INFINITY)
                .unwrap_or_else(T::zero),
            update_count: 0,
            sparsity: T::zero(),
        }
    }

    /// Update statistics with new gradients (Welford's online algorithm)
    pub fn update(&mut self, gradients: &Array1<T>) {
        let magnitude = gradients
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |a, b| a + b)
            .sqrt();

        self.update_count += 1;
        let count: T =
            scirs2_core::numeric::NumCast::from(self.update_count as f64).unwrap_or_else(T::one);

        // Welford update for the running mean and the sum of squared deviations
        let delta = magnitude - self.mean_magnitude;
        self.mean_magnitude = self.mean_magnitude + delta / count;
        let delta2 = magnitude - self.mean_magnitude;
        self.var_magnitude = self.var_magnitude + delta * delta2;

        if magnitude > self.max_magnitude {
            self.max_magnitude = magnitude;
        }
        if magnitude < self.min_magnitude {
            self.min_magnitude = magnitude;
        }

        // Update sparsity (fraction of near-zero elements)
        if !gradients.is_empty() {
            let zero_threshold: T =
                scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(T::zero);
            let zero_count = gradients
                .iter()
                .filter(|&&x| x.abs() < zero_threshold)
                .count();
            let current_sparsity: T =
                scirs2_core::numeric::NumCast::from(zero_count as f64 / gradients.len() as f64)
                    .unwrap_or_else(T::zero);
            let alpha: T = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(T::zero);
            self.sparsity = self.sparsity * (T::one() - alpha) + current_sparsity * alpha;
        }
    }

    /// Get mean magnitude
    pub fn mean_magnitude(&self) -> T {
        self.mean_magnitude
    }

    /// Get maximum magnitude
    pub fn max_magnitude(&self) -> T {
        self.max_magnitude
    }

    /// Get minimum magnitude observed (zero before the first update)
    pub fn min_magnitude(&self) -> T {
        if self.update_count == 0 {
            T::zero()
        } else {
            self.min_magnitude
        }
    }

    /// Get number of updates
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Get variance of magnitude
    pub fn variance_magnitude(&self) -> T {
        if self.update_count > 1 {
            let denominator: T =
                scirs2_core::numeric::NumCast::from((self.update_count - 1) as f64)
                    .unwrap_or_else(T::one);
            self.var_magnitude / denominator
        } else {
            T::zero()
        }
    }

    /// Get standard deviation of magnitude
    pub fn std_magnitude(&self) -> T {
        self.variance_magnitude().sqrt()
    }

    /// Get gradient sparsity
    pub fn sparsity(&self) -> T {
        self.sparsity
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> Default
    for GradientProcessingParams<T>
{
    fn default() -> Self {
        Self {
            clip_threshold: scirs2_core::numeric::NumCast::from(1.0).unwrap_or_else(T::one),
            smoothing_factor: scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(T::one),
            accumulation_steps: 4,
            dropout_prob: 0.1,
            compression_ratio: 0.1,
            norm_eps: scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(T::zero),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grad(values: &[f64]) -> Array1<f64> {
        Array1::from_vec(values.to_vec())
    }

    #[test]
    fn clipping_caps_the_norm() {
        let mut processor = GradientProcessor::<f64>::new(GradientProcessingStrategy::Clipping);
        let out = processor
            .process_gradients("w", &grad(&[3.0, 4.0]))
            .expect("clipping must succeed");
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!(norm <= 1.0 + 1e-9, "norm {norm} exceeds clip threshold");
        // Direction preserved
        assert!((out[0] / out[1] - 0.75).abs() < 1e-9);
    }

    #[test]
    fn clipping_leaves_small_gradients_untouched() {
        let mut processor = GradientProcessor::<f64>::new(GradientProcessingStrategy::Clipping);
        let input = grad(&[0.1, -0.2]);
        let out = processor
            .process_gradients("w", &input)
            .expect("clipping must succeed");
        assert_eq!(out, input);
    }

    #[test]
    fn adaptive_scaling_stays_finite_for_zero_gradients() {
        let mut processor = GradientProcessor::<f64>::new(GradientProcessingStrategy::Adaptive);
        // Prime the statistics with a non-zero gradient so mean_magnitude > 0.
        let _ = processor.process_gradients("w", &grad(&[1.0, 1.0]));
        let out = processor
            .process_gradients("w", &grad(&[0.0, 0.0]))
            .expect("adaptive scaling must succeed");
        assert!(out.iter().all(|v| v.is_finite()), "produced {out:?}");
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn adaptive_scaling_preserves_descent_direction() {
        let mut processor = GradientProcessor::<f64>::new(GradientProcessingStrategy::Adaptive);
        let _ = processor.process_gradients("w", &grad(&[2.0, -2.0]));
        let out = processor
            .process_gradients("w", &grad(&[1.0, -1.0]))
            .expect("adaptive scaling must succeed");
        assert!(out[0] > 0.0 && out[1] < 0.0);
    }

    #[test]
    fn differing_parameter_widths_do_not_conflict() {
        let mut processor = GradientProcessor::<f64>::new(GradientProcessingStrategy::Smoothing);
        let a = processor
            .process_gradients("a", &grad(&[1.0, 2.0, 3.0]))
            .expect("processing a");
        let b = processor
            .process_gradients("b", &grad(&[1.0, 2.0]))
            .expect("processing b");
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn zero_accumulation_steps_do_not_panic() {
        let mut params = GradientProcessingParams::<f64>::default();
        params.accumulation_steps = 0;
        let mut processor = GradientProcessor::<f64>::new_with_params(
            GradientProcessingStrategy::Accumulation,
            params,
        );
        let out = processor
            .process_gradients("w", &grad(&[1.0, 1.0]))
            .expect("accumulation must succeed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn statistics_track_welford_variance() {
        let mut stats = GradientStatistics::<f64>::new();
        stats.update(&grad(&[3.0, 4.0])); // magnitude 5
        stats.update(&grad(&[0.0, 1.0])); // magnitude 1
        assert!((stats.mean_magnitude() - 3.0).abs() < 1e-12);
        // Sample variance of {5, 1} is 8
        assert!((stats.variance_magnitude() - 8.0).abs() < 1e-12);
    }
}
