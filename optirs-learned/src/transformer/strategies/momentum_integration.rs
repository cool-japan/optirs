use std::fmt::Debug;
// Momentum integration strategies for transformer optimization
//
// This module implements various momentum-based optimization strategies that
// integrate with the transformer optimizer's attention mechanisms.
//
// Momentum state is kept per parameter name so that models whose parameter
// tensors have different shapes can share a single integrator.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

use crate::error::Result;

/// Momentum integration strategies
#[derive(Debug, Clone, Copy)]
pub enum MomentumStrategy {
    /// Standard momentum
    Standard,
    /// Nesterov accelerated gradient
    Nesterov,
    /// Adam-style adaptive momentum
    Adam,
    /// RMSprop-style momentum
    RMSprop,
    /// Transformer-predicted momentum
    TransformerPredicted,
    /// Adaptive momentum based on attention patterns
    AttentionAdaptive,
    /// Hierarchical momentum for different scales
    Hierarchical,
    /// Momentum with variance tracking
    VarianceTracking,
}

/// Per-parameter momentum state slots.
#[derive(Debug, Clone)]
struct MomentumSlots<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
{
    first_moment: Option<Array1<T>>,
    second_moment: Option<Array1<T>>,
    velocity: Option<Array1<T>>,
    hierarchical_moments: Vec<MomentumState<T>>,
    step_count: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for MomentumSlots<T>
{
    fn default() -> Self {
        Self {
            first_moment: None,
            second_moment: None,
            velocity: None,
            hierarchical_moments: Vec::new(),
            step_count: 0,
        }
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    MomentumSlots<T>
{
    /// Discard state that no longer matches the parameter width.
    fn ensure_width(&mut self, width: usize) {
        let mismatched = |slot: &Option<Array1<T>>| slot.as_ref().is_some_and(|a| a.len() != width);
        if mismatched(&self.first_moment)
            || mismatched(&self.second_moment)
            || mismatched(&self.velocity)
        {
            self.first_moment = None;
            self.second_moment = None;
            self.velocity = None;
        }
        let hierarchical_width: usize = self.hierarchical_moments.iter().map(|s| s.m.len()).sum();
        if !self.hierarchical_moments.is_empty() && hierarchical_width != width {
            self.hierarchical_moments.clear();
        }
    }
}

/// Momentum integrator for transformer optimizer
#[derive(Debug, Clone)]
pub struct MomentumIntegrator<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Integration strategy
    strategy: MomentumStrategy,

    /// Momentum parameters
    momentum_params: MomentumParams<T>,

    /// First moment estimates (momentum) for the parameter being processed
    first_moment: Option<Array1<T>>,

    /// Second moment estimates (for Adam-style)
    second_moment: Option<Array1<T>>,

    /// Velocity for standard/Nesterov momentum
    velocity: Option<Array1<T>>,

    /// Attention-based momentum scaling
    attention_scaling: Option<Array1<T>>,

    /// Step counter of the parameter currently being processed
    step_count: usize,

    /// Total number of integration calls (all parameters)
    total_steps: usize,

    /// Hierarchical momentum for different parameter groups
    hierarchical_moments: Vec<MomentumState<T>>,

    /// Per-parameter persisted state
    states: HashMap<String, MomentumSlots<T>>,
}

/// Momentum parameters
#[derive(Debug, Clone)]
pub struct MomentumParams<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Beta1 (momentum coefficient)
    pub beta1: T,

    /// Beta2 (second moment coefficient, for Adam-style)
    pub beta2: T,

    /// Epsilon for numerical stability
    pub epsilon: T,

    /// Decay rate for momentum
    pub decay_rate: T,

    /// Adaptive scaling factor
    pub adaptive_scale: T,

    /// Attention integration weight
    pub attention_weight: T,

    /// Variance tracking weight
    pub variance_weight: T,

    /// Number of hierarchical parameter groups
    pub hierarchical_groups: usize,
}

/// Momentum state for hierarchical momentum
#[derive(Debug, Clone)]
pub struct MomentumState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// First moment
    m: Array1<T>,

    /// Second moment
    v: Array1<T>,

    /// Parameter group identifier
    group_id: usize,

    /// Group-specific momentum coefficient
    group_beta1: T,

    /// Group-specific second moment coefficient
    group_beta2: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    MomentumState<T>
{
    /// Group identifier
    pub fn group_id(&self) -> usize {
        self.group_id
    }

    /// Group-specific first moment coefficient
    pub fn group_beta1(&self) -> T {
        self.group_beta1
    }

    /// Group-specific second moment coefficient
    pub fn group_beta2(&self) -> T {
        self.group_beta2
    }

    /// Number of parameters in this group
    pub fn len(&self) -> usize {
        self.m.len()
    }

    /// Whether this group is empty
    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }
}

/// Momentum statistics for analysis
#[derive(Debug, Clone)]
pub struct MomentumStatistics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Average momentum magnitude
    pub avg_momentum_magnitude: T,

    /// Momentum variance
    pub momentum_variance: T,

    /// Direction consistency score (cosine of the mean update direction)
    pub direction_consistency: T,

    /// Number of parameters with tracked momentum
    pub tracked_parameters: usize,

    /// Update count
    pub update_count: usize,
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand
            + std::iter::Sum
            + Send
            + Sync
            + 'static,
    > MomentumIntegrator<T>
{
    /// Create new momentum integrator
    pub fn new(strategy: MomentumStrategy) -> Self {
        Self::new_with_params(strategy, MomentumParams::default())
    }

    /// Create with custom parameters
    pub fn new_with_params(strategy: MomentumStrategy, params: MomentumParams<T>) -> Self {
        Self {
            strategy,
            momentum_params: params,
            first_moment: None,
            second_moment: None,
            velocity: None,
            attention_scaling: None,
            step_count: 0,
            total_steps: 0,
            hierarchical_moments: Vec::new(),
            states: HashMap::new(),
        }
    }

    /// Integrate momentum with gradients for the named parameter.
    pub fn integrate_momentum(
        &mut self,
        param_name: &str,
        gradients: &Array1<T>,
        attention_weights: Option<&Array2<T>>,
    ) -> Result<Array1<T>> {
        // Load the per-parameter state into the working slots.
        let mut slots = self.states.remove(param_name).unwrap_or_default();
        slots.ensure_width(gradients.len());
        self.first_moment = slots.first_moment.take();
        self.second_moment = slots.second_moment.take();
        self.velocity = slots.velocity.take();
        self.hierarchical_moments = std::mem::take(&mut slots.hierarchical_moments);
        self.step_count = slots.step_count + 1;
        self.total_steps += 1;

        // Update attention-based scaling if provided
        if let Some(attn) = attention_weights {
            self.update_attention_scaling(attn)?;
        }

        let result = match self.strategy {
            MomentumStrategy::Standard => self.standard_momentum(gradients),
            MomentumStrategy::Nesterov => self.nesterov_momentum(gradients),
            MomentumStrategy::Adam => self.adam_momentum(gradients),
            MomentumStrategy::RMSprop => self.rmsprop_momentum(gradients),
            MomentumStrategy::TransformerPredicted => {
                self.transformer_predicted_momentum(gradients)
            }
            MomentumStrategy::AttentionAdaptive => self.attention_adaptive_momentum(gradients),
            MomentumStrategy::Hierarchical => self.hierarchical_momentum(gradients),
            MomentumStrategy::VarianceTracking => self.variance_tracking_momentum(gradients),
        };

        // Persist the state back into the per-parameter slot.
        self.states.insert(
            param_name.to_string(),
            MomentumSlots {
                first_moment: self.first_moment.take(),
                second_moment: self.second_moment.take(),
                velocity: self.velocity.take(),
                hierarchical_moments: std::mem::take(&mut self.hierarchical_moments),
                step_count: self.step_count,
            },
        );

        result
    }

    /// Standard momentum implementation: `v <- beta * v + g`.
    fn standard_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let v = self
            .velocity
            .get_or_insert_with(|| Array1::zeros(gradients.len()));
        *v = v.clone() * self.momentum_params.beta1 + gradients;
        Ok(v.clone())
    }

    /// Nesterov accelerated gradient.
    ///
    /// The velocity is updated exactly as for classical momentum,
    /// `v_new = beta * v + g`, but the applied update uses the look-ahead
    /// direction `beta * v_new + g`, which is what distinguishes NAG from
    /// classical momentum.
    fn nesterov_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let beta = self.momentum_params.beta1;
        let v = self
            .velocity
            .get_or_insert_with(|| Array1::zeros(gradients.len()));
        *v = v.clone() * beta + gradients;
        let update = v.clone() * beta + gradients;
        Ok(update)
    }

    /// Adam-style momentum with second moments
    fn adam_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let beta1 = self.momentum_params.beta1;
        let beta2 = self.momentum_params.beta2;
        let eps = self.momentum_params.epsilon;
        let step: T =
            scirs2_core::numeric::NumCast::from(self.step_count as f64).unwrap_or_else(T::one);

        if self.first_moment.is_none() {
            self.first_moment = Some(Array1::zeros(gradients.len()));
        }
        if self.second_moment.is_none() {
            self.second_moment = Some(Array1::zeros(gradients.len()));
        }

        let (Some(m), Some(v)) = (self.first_moment.as_mut(), self.second_moment.as_mut()) else {
            return Ok(gradients.clone());
        };

        *m = m.clone() * beta1 + gradients * (T::one() - beta1);
        let grad_squared = gradients.mapv(|x| x * x);
        *v = v.clone() * beta2 + &grad_squared * (T::one() - beta2);

        // Bias correction, guarded against a degenerate zero denominator
        let bias_correction1 = (T::one() - beta1.powf(step)).max(eps);
        let bias_correction2 = (T::one() - beta2.powf(step)).max(eps);

        let m_hat = m.clone() / bias_correction1;
        let v_hat = v.clone() / bias_correction2;

        let update = m_hat
            .iter()
            .zip(v_hat.iter())
            .map(|(&m_val, &v_val)| m_val / (v_val.sqrt() + eps))
            .collect::<Vec<_>>();

        Ok(Array1::from_vec(update))
    }

    /// RMSprop-style momentum
    fn rmsprop_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let beta2 = self.momentum_params.beta2;
        let eps = self.momentum_params.epsilon;

        let v = self
            .second_moment
            .get_or_insert_with(|| Array1::zeros(gradients.len()));

        let grad_squared = gradients.mapv(|x| x * x);
        *v = v.clone() * beta2 + &grad_squared * (T::one() - beta2);

        let update = gradients
            .iter()
            .zip(v.iter())
            .map(|(&g, &v_val)| g / (v_val.sqrt() + eps))
            .collect::<Vec<_>>();

        Ok(Array1::from_vec(update))
    }

    /// Transformer-predicted momentum coefficients derived from gradient scale.
    fn transformer_predicted_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let grad_norm = gradients
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |a, b| a + b)
            .sqrt();
        let adaptive_beta: T = self.momentum_params.beta1 * (T::one() / (T::one() + grad_norm));

        let v = self
            .velocity
            .get_or_insert_with(|| Array1::zeros(gradients.len()));
        *v = v.clone() * adaptive_beta + gradients;
        Ok(v.clone())
    }

    /// Attention-adaptive momentum
    fn attention_adaptive_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let base_beta = self.momentum_params.beta1;

        // Scale momentum based on attention patterns when the widths line up.
        let momentum_update = match self.attention_scaling.as_ref() {
            Some(scaling) if scaling.len() == gradients.len() => {
                let scaled = gradients
                    .iter()
                    .zip(scaling.iter())
                    .map(|(&g, &s)| g * s)
                    .collect::<Vec<_>>();
                Array1::from_vec(scaled)
            }
            _ => gradients.clone(),
        };

        let v = self
            .velocity
            .get_or_insert_with(|| Array1::zeros(gradients.len()));
        *v = v.clone() * base_beta + &momentum_update;
        Ok(v.clone())
    }

    /// Hierarchical momentum for different parameter groups.
    ///
    /// The parameter vector is split into contiguous groups; the final group
    /// absorbs the remainder so that every coordinate is covered.
    fn hierarchical_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        if gradients.is_empty() {
            return Ok(Array1::zeros(0));
        }

        if self.hierarchical_moments.is_empty() {
            self.initialize_hierarchical_states(gradients.len())?;
        }

        let num_groups = self.hierarchical_moments.len().max(1);
        let base = gradients.len() / num_groups;
        let mut update = Array1::zeros(gradients.len());

        for (i, state) in self.hierarchical_moments.iter_mut().enumerate() {
            let start_idx = i * base;
            // The last group absorbs the remainder (len % num_groups).
            let end_idx = if i + 1 == num_groups {
                gradients.len()
            } else {
                (i + 1) * base
            };
            if start_idx >= end_idx {
                continue;
            }

            let group_gradients = gradients.slice(scirs2_core::ndarray::s![start_idx..end_idx]);
            if state.m.len() != group_gradients.len() {
                state.m = Array1::zeros(group_gradients.len());
                state.v = Array1::zeros(group_gradients.len());
            }

            state.m = state.m.clone() * state.group_beta1 + group_gradients;

            for (j, &val) in state.m.iter().enumerate() {
                update[start_idx + j] = val;
            }
        }

        Ok(update)
    }

    /// Momentum with variance tracking
    fn variance_tracking_momentum(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let beta1 = self.momentum_params.beta1;
        let var_weight = self.momentum_params.variance_weight;
        let epsilon = self.momentum_params.epsilon;

        if self.first_moment.is_none() {
            self.first_moment = Some(Array1::zeros(gradients.len()));
        }
        if self.second_moment.is_none() {
            self.second_moment = Some(Array1::zeros(gradients.len()));
        }

        let (Some(m), Some(v)) = (self.first_moment.as_mut(), self.second_moment.as_mut()) else {
            return Ok(gradients.clone());
        };

        *m = m.clone() * beta1 + gradients * (T::one() - beta1);

        let grad_diff = gradients - &m.clone();
        let grad_var = grad_diff.mapv(|x| x * x);
        *v = v.clone() * var_weight + &grad_var * (T::one() - var_weight);

        let adjusted_momentum = m
            .iter()
            .zip(v.iter())
            .map(|(&m_val, &var)| m_val / (var.sqrt() + epsilon))
            .collect::<Vec<_>>();

        Ok(Array1::from_vec(adjusted_momentum))
    }

    /// Update attention-based scaling factors
    fn update_attention_scaling(&mut self, attention_weights: &Array2<T>) -> Result<()> {
        let (num_heads, seq_len) = attention_weights.dim();
        if num_heads == 0 || seq_len == 0 {
            return Ok(());
        }

        let total_attention = attention_weights.iter().cloned().sum::<T>();

        if total_attention > T::zero() {
            let mut scaling = Array1::zeros(seq_len);
            let seq_len_t: T =
                scirs2_core::numeric::NumCast::from(seq_len as f64).unwrap_or_else(T::one);
            for i in 0..seq_len {
                let column_sum = (0..num_heads).map(|h| attention_weights[[h, i]]).sum::<T>();
                scaling[i] = column_sum / total_attention * seq_len_t;
            }

            self.attention_scaling = Some(scaling);
        }

        Ok(())
    }

    /// Initialize hierarchical momentum states
    fn initialize_hierarchical_states(&mut self, param_count: usize) -> Result<()> {
        let num_groups = self.momentum_params.hierarchical_groups.max(1);
        let base = param_count / num_groups;

        for i in 0..num_groups {
            // Last group absorbs the remainder.
            let actual_size = if i + 1 == num_groups {
                param_count - i * base
            } else {
                base
            };

            let group_progress = i as f64 / num_groups as f64;
            let state = MomentumState {
                m: Array1::zeros(actual_size),
                v: Array1::zeros(actual_size),
                group_id: i,
                group_beta1: self.momentum_params.beta1
                    * scirs2_core::numeric::NumCast::from(0.8 + 0.2 * group_progress)
                        .unwrap_or_else(T::one),
                group_beta2: self.momentum_params.beta2
                    * scirs2_core::numeric::NumCast::from(0.9 + 0.1 * group_progress)
                        .unwrap_or_else(T::one),
            };

            self.hierarchical_moments.push(state);
        }

        Ok(())
    }

    /// Get the current momentum vector for a parameter, if any is tracked.
    pub fn current_momentum(&self, param_name: &str) -> Option<&Array1<T>> {
        self.states
            .get(param_name)
            .and_then(|s| s.first_moment.as_ref().or(s.velocity.as_ref()))
    }

    /// Get momentum statistics aggregated over all tracked parameters.
    pub fn statistics(&self) -> MomentumStatistics<T> {
        let mut magnitude_sum = T::zero();
        let mut variance_sum = T::zero();
        let mut consistency_sum = T::zero();
        let mut tracked = 0usize;

        for slots in self.states.values() {
            let Some(momentum) = slots.first_moment.as_ref().or(slots.velocity.as_ref()) else {
                continue;
            };
            if momentum.is_empty() {
                continue;
            }
            tracked += 1;

            let magnitude = momentum
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |a, b| a + b)
                .sqrt();
            magnitude_sum = magnitude_sum + magnitude;

            let len_t: T =
                scirs2_core::numeric::NumCast::from(momentum.len() as f64).unwrap_or_else(T::one);
            let mean = momentum.iter().cloned().sum::<T>() / len_t;
            if momentum.len() > 1 {
                let denominator: T =
                    scirs2_core::numeric::NumCast::from((momentum.len() - 1) as f64)
                        .unwrap_or_else(T::one);
                variance_sum = variance_sum
                    + momentum
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .fold(T::zero(), |a, b| a + b)
                        / denominator;
            }

            // Direction consistency: |sum(m)| / (sqrt(n) * ||m||), which is 1 when
            // every coordinate points the same way and ~0 for uncorrelated signs.
            if magnitude > T::zero() {
                consistency_sum =
                    consistency_sum + (mean * len_t).abs() / (len_t.sqrt() * magnitude);
            }
        }

        if tracked == 0 {
            return MomentumStatistics {
                avg_momentum_magnitude: T::zero(),
                momentum_variance: T::zero(),
                direction_consistency: T::zero(),
                tracked_parameters: 0,
                update_count: self.total_steps,
            };
        }

        let tracked_t: T =
            scirs2_core::numeric::NumCast::from(tracked as f64).unwrap_or_else(T::one);
        MomentumStatistics {
            avg_momentum_magnitude: magnitude_sum / tracked_t,
            momentum_variance: variance_sum / tracked_t,
            direction_consistency: consistency_sum / tracked_t,
            tracked_parameters: tracked,
            update_count: self.total_steps,
        }
    }

    /// Get the configured strategy
    pub fn strategy(&self) -> MomentumStrategy {
        self.strategy
    }

    /// Get the configured parameters
    pub fn parameters(&self) -> &MomentumParams<T> {
        &self.momentum_params
    }

    /// Reset integrator state
    pub fn reset(&mut self) {
        self.first_moment = None;
        self.second_moment = None;
        self.velocity = None;
        self.attention_scaling = None;
        self.step_count = 0;
        self.total_steps = 0;
        self.hierarchical_moments.clear();
        self.states.clear();
    }

    /// Update strategy
    pub fn set_strategy(&mut self, strategy: MomentumStrategy) {
        self.strategy = strategy;
    }

    /// Update parameters
    pub fn set_parameters(&mut self, params: MomentumParams<T>) {
        self.momentum_params = params;
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand
            + std::iter::Sum
            + Send
            + Sync
            + 'static,
    > Default for MomentumParams<T>
{
    fn default() -> Self {
        Self {
            beta1: scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(T::one),
            beta2: scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(T::one),
            epsilon: scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(T::zero),
            decay_rate: scirs2_core::numeric::NumCast::from(0.99).unwrap_or_else(T::one),
            adaptive_scale: T::one(),
            attention_weight: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(T::zero),
            variance_weight: scirs2_core::numeric::NumCast::from(0.99).unwrap_or_else(T::one),
            hierarchical_groups: 4,
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
    fn nesterov_differs_from_standard_momentum() {
        let mut standard = MomentumIntegrator::<f64>::new(MomentumStrategy::Standard);
        let mut nesterov = MomentumIntegrator::<f64>::new(MomentumStrategy::Nesterov);
        let g = grad(&[1.0, -1.0]);

        let mut standard_out = grad(&[0.0, 0.0]);
        let mut nesterov_out = grad(&[0.0, 0.0]);
        for _ in 0..3 {
            standard_out = standard
                .integrate_momentum("w", &g, None)
                .expect("standard momentum");
            nesterov_out = nesterov
                .integrate_momentum("w", &g, None)
                .expect("nesterov momentum");
        }

        assert!(
            (standard_out[0] - nesterov_out[0]).abs() > 1e-9,
            "nesterov {nesterov_out:?} must differ from standard {standard_out:?}"
        );
        // Nesterov look-ahead: beta * v_new + g with v_new = beta*v + g
        let beta = 0.9;
        let mut v = 0.0;
        for _ in 0..3 {
            v = beta * v + 1.0;
        }
        assert!((nesterov_out[0] - (beta * v + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn hierarchical_momentum_covers_the_remainder() {
        let mut integrator = MomentumIntegrator::<f64>::new(MomentumStrategy::Hierarchical);
        // 10 elements into 4 groups leaves a remainder of 2.
        let g = grad(&[1.0; 10]);
        let out = integrator
            .integrate_momentum("w", &g, None)
            .expect("hierarchical momentum");
        assert_eq!(out.len(), 10);
        assert!(
            out.iter().all(|&x| x > 0.0),
            "tail coordinates were dropped: {out:?}"
        );
    }

    #[test]
    fn zero_gradients_stay_finite() {
        for strategy in [
            MomentumStrategy::Standard,
            MomentumStrategy::Nesterov,
            MomentumStrategy::Adam,
            MomentumStrategy::RMSprop,
            MomentumStrategy::TransformerPredicted,
            MomentumStrategy::AttentionAdaptive,
            MomentumStrategy::Hierarchical,
            MomentumStrategy::VarianceTracking,
        ] {
            let mut integrator = MomentumIntegrator::<f64>::new(strategy);
            let out = integrator
                .integrate_momentum("w", &grad(&[0.0; 6]), None)
                .expect("momentum integration");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{strategy:?} produced {out:?}"
            );
        }
    }

    #[test]
    fn differing_parameter_widths_do_not_conflict() {
        let mut integrator = MomentumIntegrator::<f64>::new(MomentumStrategy::Adam);
        let a = integrator
            .integrate_momentum("a", &grad(&[1.0, 2.0, 3.0]), None)
            .expect("param a");
        let b = integrator
            .integrate_momentum("b", &grad(&[1.0, 2.0]), None)
            .expect("param b");
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn statistics_report_tracked_parameters() {
        let mut integrator = MomentumIntegrator::<f64>::new(MomentumStrategy::Standard);
        let _ = integrator.integrate_momentum("a", &grad(&[1.0, 1.0]), None);
        let _ = integrator.integrate_momentum("b", &grad(&[2.0, 2.0]), None);
        let stats = integrator.statistics();
        assert_eq!(stats.tracked_parameters, 2);
        assert_eq!(stats.update_count, 2);
        assert!(stats.avg_momentum_magnitude > 0.0);
    }
}
