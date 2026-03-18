//! Computer Vision Optimizer
//!
//! A domain-specific optimizer for computer vision tasks that incorporates
//! spatial awareness, channel-wise gradient normalization, and progressive
//! resolution training support.
//!
//! # Key Features
//! - **Spatial LR scaling**: Different learning rates for spatial vs non-spatial params
//! - **Channel normalization**: Per-channel gradient normalization for stable training
//! - **Progressive resolution**: Gradual resolution increases during training
//! - **Warmup schedule**: Linear warmup for stable early training
//! - **Momentum**: Classical momentum for smoother updates

use crate::domain_optimizers::{l2_norm, AdvancedOptimizer, OptimizerStateInfo};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Computer Vision optimizer with spatial awareness and progressive training.
///
/// This optimizer is designed for convolutional neural networks and vision
/// transformers. It applies domain-specific heuristics such as spatial gradient
/// scaling, per-channel normalization, and progressive resolution support.
#[derive(Debug, Clone)]
pub struct CVOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Base learning rate (before schedules)
    base_lr: T,
    /// Current effective learning rate (after schedules)
    current_lr: T,
    /// Scale factor applied to spatial dimension gradients
    spatial_lr_scale: T,
    /// Whether to normalize gradients per channel group
    channel_normalization: bool,
    /// Whether progressive resolution training is enabled
    progressive_resolution: bool,
    /// Current resolution phase (0-indexed)
    resolution_phase: usize,
    /// Maximum number of resolution phases
    max_resolution_phases: usize,
    /// Number of warmup steps with linear LR ramp
    warmup_steps: usize,
    /// Current optimization step
    step_count: usize,
    /// Momentum coefficient
    momentum: T,
    /// Velocity buffer for momentum
    velocity: Option<Array1<T>>,
    /// Exponential moving average of gradient norms
    grad_norm_ema: T,
    /// Decay factor for gradient norm EMA
    ema_decay: T,
}

impl<T: Float + Debug + Send + Sync + 'static> CVOptimizer<T> {
    /// Create a new CVOptimizer with the given base learning rate.
    ///
    /// Defaults: no spatial scaling, no channel normalization, no progressive
    /// resolution, no warmup, momentum=0.9, EMA decay=0.999.
    pub fn new(base_lr: T) -> Self {
        Self {
            base_lr,
            current_lr: base_lr,
            spatial_lr_scale: T::one(),
            channel_normalization: false,
            progressive_resolution: false,
            resolution_phase: 0,
            max_resolution_phases: 1,
            warmup_steps: 0,
            step_count: 0,
            momentum: T::from(0.9).expect("0.9 should convert to T"),
            velocity: None,
            grad_norm_ema: T::zero(),
            ema_decay: T::from(0.999).expect("0.999 should convert to T"),
        }
    }

    /// Set the spatial learning rate scale factor (builder pattern).
    ///
    /// Spatial gradients (simulated as the first half of the gradient vector)
    /// will be multiplied by this factor.
    pub fn with_spatial_lr_scale(mut self, scale: T) -> Self {
        self.spatial_lr_scale = scale;
        self
    }

    /// Enable or disable per-channel gradient normalization (builder pattern).
    pub fn with_channel_normalization(mut self, enable: bool) -> Self {
        self.channel_normalization = enable;
        self
    }

    /// Enable progressive resolution training (builder pattern).
    ///
    /// `max_phases` is the total number of resolution phases. The learning
    /// rate is scaled by `(current_phase + 1) / max_phases` to allow the
    /// model to stabilise at each resolution.
    pub fn with_progressive_resolution(mut self, enable: bool, max_phases: usize) -> Self {
        self.progressive_resolution = enable;
        self.max_resolution_phases = if max_phases == 0 { 1 } else { max_phases };
        self
    }

    /// Set the number of linear warmup steps (builder pattern).
    pub fn with_warmup(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set the momentum coefficient (builder pattern).
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    /// Advance to the next resolution phase.
    ///
    /// Returns an error if the optimizer is already at the maximum phase.
    pub fn advance_resolution_phase(&mut self) -> Result<()> {
        if !self.progressive_resolution {
            return Err(OptimError::InvalidConfig(
                "Progressive resolution is not enabled".to_string(),
            ));
        }
        if self.resolution_phase + 1 >= self.max_resolution_phases {
            return Err(OptimError::InvalidState(format!(
                "Already at maximum resolution phase {}",
                self.resolution_phase
            )));
        }
        self.resolution_phase += 1;
        Ok(())
    }

    /// Compute the warmup multiplier for the current step.
    fn warmup_factor(&self) -> T {
        if self.warmup_steps == 0 || self.step_count >= self.warmup_steps {
            return T::one();
        }
        let step_t = T::from(self.step_count + 1).expect("step should convert");
        let warmup_t = T::from(self.warmup_steps).expect("warmup should convert");
        step_t / warmup_t
    }

    /// Compute the resolution phase scaling factor.
    fn resolution_factor(&self) -> T {
        if !self.progressive_resolution || self.max_resolution_phases <= 1 {
            return T::one();
        }
        let phase_t = T::from(self.resolution_phase + 1).expect("phase should convert");
        let max_t = T::from(self.max_resolution_phases).expect("max_phases should convert");
        phase_t / max_t
    }

    /// Apply spatial scaling to gradients.
    ///
    /// Treats the first half of the gradient vector as "spatial" parameters
    /// and scales them by `spatial_lr_scale`.
    fn apply_spatial_scaling(&self, gradients: &Array1<T>) -> Array1<T> {
        if (self.spatial_lr_scale - T::one()).abs() < T::from(1e-12).expect("epsilon convert") {
            return gradients.clone();
        }
        let len = gradients.len();
        let spatial_end = len / 2;
        let mut scaled = gradients.clone();
        for i in 0..spatial_end {
            scaled[i] = scaled[i] * self.spatial_lr_scale;
        }
        scaled
    }

    /// Apply per-channel normalization to gradients.
    ///
    /// Splits gradients into 4 equal groups (simulating channels) and
    /// normalises each group to have unit norm, preserving direction.
    fn apply_channel_normalization(&self, gradients: &Array1<T>) -> Array1<T> {
        if !self.channel_normalization {
            return gradients.clone();
        }
        let len = gradients.len();
        let num_channels: usize = 4; // simulated channel count
        let chunk_size = if len >= num_channels {
            len / num_channels
        } else {
            return gradients.clone();
        };

        let mut normalised = gradients.clone();
        for ch in 0..num_channels {
            let start = ch * chunk_size;
            let end = if ch == num_channels - 1 {
                len
            } else {
                start + chunk_size
            };

            // Compute norm of this channel slice
            let channel_norm = {
                let mut sum_sq = T::zero();
                for i in start..end {
                    sum_sq = sum_sq + normalised[i] * normalised[i];
                }
                sum_sq.sqrt()
            };

            let epsilon = T::from(1e-8).expect("epsilon convert");
            if channel_norm > epsilon {
                for i in start..end {
                    normalised[i] = normalised[i] / (channel_norm + epsilon);
                }
            }
        }
        normalised
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdvancedOptimizer<T> for CVOptimizer<T> {
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {
        if params.len() != gradients.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Parameter length {} != gradient length {}",
                params.len(),
                gradients.len()
            )));
        }
        if params.is_empty() {
            return Err(OptimError::InsufficientData(
                "Empty parameter array".to_string(),
            ));
        }

        // 1. Apply warmup schedule
        let warmup = self.warmup_factor();

        // 2. Scale spatial gradients
        let grad = self.apply_spatial_scaling(gradients);

        // 3. Normalize per-channel if enabled
        let grad = self.apply_channel_normalization(&grad);

        // 4. Update gradient norm EMA
        let norm = l2_norm(&grad);
        self.grad_norm_ema =
            self.ema_decay * self.grad_norm_ema + (T::one() - self.ema_decay) * norm;

        // 5. Resolution phase factor
        let res_factor = self.resolution_factor();

        // 6. Effective LR
        let effective_lr = self.base_lr * warmup * res_factor;
        self.current_lr = effective_lr;

        // 7. Initialize or update velocity (momentum)
        let velocity = match self.velocity.take() {
            Some(v) if v.len() == params.len() => v,
            _ => Array1::zeros(params.len()),
        };

        let velocity =
            velocity.mapv(|v| v * self.momentum) + &grad.mapv(|g| g * (T::one() - self.momentum));
        let updated = params - &velocity.mapv(|v| v * effective_lr);

        self.velocity = Some(velocity);
        self.step_count += 1;

        Ok(updated)
    }

    fn get_learning_rate(&self) -> T {
        self.current_lr
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.base_lr = lr;
        self.current_lr = lr;
    }

    fn name(&self) -> &str {
        "CVOptimizer"
    }

    fn get_state(&self) -> OptimizerStateInfo<T> {
        OptimizerStateInfo {
            step_count: self.step_count,
            current_lr: self.current_lr,
            grad_norm_ema: self.grad_norm_ema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_cv_optimizer_basic_step() {
        let mut opt = CVOptimizer::new(0.01_f64);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let updated = opt.step(&params, &grads).expect("step should succeed");
        // Params should decrease in the gradient direction
        for i in 0..4 {
            assert!(updated[i] < params[i], "param {} should decrease", i);
        }
        assert_eq!(opt.step_count, 1);
    }

    #[test]
    fn test_cv_optimizer_warmup() {
        let mut opt = CVOptimizer::new(0.1_f64).with_warmup(10);
        let params = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let grads = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        // First step: warmup factor = 1/10 = 0.1, effective lr = 0.01
        let u1 = opt.step(&params, &grads).expect("step should succeed");
        // Second step: warmup factor = 2/10 = 0.2, effective lr = 0.02
        let u2 = opt.step(&params, &grads).expect("step should succeed");

        // Second step should have larger update magnitude
        let diff1 = (&params - &u1).mapv(|x| x.abs()).sum();
        let diff2 = (&params - &u2).mapv(|x| x.abs()).sum();
        assert!(
            diff2 > diff1,
            "warmup should increase update magnitude over time"
        );
    }

    #[test]
    fn test_cv_optimizer_spatial_scaling() {
        let mut opt_scaled = CVOptimizer::new(0.01_f64)
            .with_spatial_lr_scale(2.0)
            .with_momentum(0.0);
        let mut opt_base = CVOptimizer::new(0.01_f64).with_momentum(0.0);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1]);

        let u_scaled = opt_scaled
            .step(&params, &grads)
            .expect("step should succeed");
        let u_base = opt_base.step(&params, &grads).expect("step should succeed");

        // First half (spatial) should have larger updates with scaling
        let delta_scaled_spatial = (params[0] - u_scaled[0]).abs();
        let delta_base_spatial = (params[0] - u_base[0]).abs();
        assert!(
            delta_scaled_spatial > delta_base_spatial,
            "spatial scaling should amplify spatial updates"
        );
    }

    #[test]
    fn test_cv_optimizer_progressive_resolution() {
        let mut opt = CVOptimizer::new(0.1_f64).with_progressive_resolution(true, 4);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);

        // Phase 0: factor = 1/4
        let _u0 = opt.step(&params, &grads).expect("step should succeed");
        let lr0 = opt.get_learning_rate();

        opt.advance_resolution_phase()
            .expect("advance should succeed");
        // Phase 1: factor = 2/4
        let _u1 = opt.step(&params, &grads).expect("step should succeed");
        let lr1 = opt.get_learning_rate();

        assert!(lr1 > lr0, "LR should increase with resolution phase");
    }

    #[test]
    fn test_cv_optimizer_advance_phase_error() {
        let mut opt = CVOptimizer::new(0.01_f64).with_progressive_resolution(true, 2);
        opt.advance_resolution_phase().expect("first advance ok");
        let result = opt.advance_resolution_phase();
        assert!(result.is_err(), "should error when at max phase");
    }

    #[test]
    fn test_cv_optimizer_channel_normalization() {
        let mut opt = CVOptimizer::new(0.01_f64)
            .with_channel_normalization(true)
            .with_momentum(0.0);
        let params = Array1::from_vec(vec![1.0; 8]);
        // Large gradients in first channel, small in others
        let grads = Array1::from_vec(vec![100.0, 100.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]);

        let updated = opt.step(&params, &grads).expect("step should succeed");
        // With channel normalization, the update magnitudes across channels
        // should be more balanced than without
        let diff_ch0 = (params[0] - updated[0]).abs();
        let diff_ch1 = (params[2] - updated[2]).abs();
        // Without normalization, diff_ch0 would be ~10000x larger.
        // With normalization, they should be in the same ballpark.
        let ratio = diff_ch0 / (diff_ch1 + 1e-15);
        assert!(
            ratio < 100.0,
            "channel normalization should balance updates, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_cv_optimizer_state_info() {
        let mut opt = CVOptimizer::new(0.05_f64);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let _ = opt.step(&params, &grads).expect("step should succeed");
        let state = opt.get_state();
        assert_eq!(state.step_count, 1);
        assert!(approx_eq(state.current_lr, 0.05, 1e-9));
        assert!(state.grad_norm_ema > 0.0);
    }

    #[test]
    fn test_cv_optimizer_dimension_mismatch() {
        let mut opt = CVOptimizer::new(0.01_f64);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let result = opt.step(&params, &grads);
        assert!(result.is_err(), "should error on dimension mismatch");
    }
}
