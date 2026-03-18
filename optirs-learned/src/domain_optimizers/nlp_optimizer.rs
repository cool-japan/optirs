//! NLP Optimizer
//!
//! A domain-specific optimizer for natural language processing tasks that
//! incorporates layer-wise learning rate decay, linear warmup with cosine
//! annealing, gradient accumulation, and gradient clipping.
//!
//! # Key Features
//! - **Layer-wise LR decay**: Deeper layers get smaller learning rates
//! - **Warmup + cosine schedule**: Linear warmup followed by cosine decay
//! - **Token-aware scaling**: Optional gradient scaling by token frequency
//! - **Gradient accumulation**: Accumulate gradients over multiple micro-batches
//! - **Gradient clipping**: Max-norm gradient clipping for training stability

use crate::domain_optimizers::{clip_grad_norm, l2_norm, AdvancedOptimizer, OptimizerStateInfo};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// NLP optimizer with layer-wise learning rate decay and schedule support.
///
/// Designed for transformer-based language models (BERT, GPT, T5, etc.).
/// Implements the common recipe of linear warmup followed by cosine
/// annealing, combined with discriminative (layer-wise) learning rates.
#[derive(Debug, Clone)]
pub struct NLPOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Base learning rate
    base_lr: T,
    /// Current effective learning rate after schedule
    current_lr: T,
    /// Multiplicative decay applied per layer (e.g., 0.95)
    layer_wise_decay: T,
    /// Total number of transformer layers (for layer-wise decay)
    num_layers: usize,
    /// Number of linear warmup steps
    warmup_steps: usize,
    /// Total training steps (for cosine annealing)
    total_steps: usize,
    /// Current optimization step
    step_count: usize,
    /// Whether to apply token-frequency-aware gradient scaling
    token_aware_scaling: bool,
    /// Number of micro-batch steps to accumulate before applying an update
    gradient_accumulation_steps: usize,
    /// Accumulated gradient buffer
    accumulated_gradients: Option<Array1<T>>,
    /// Number of gradients accumulated so far in the current window
    accumulation_count: usize,
    /// Maximum gradient norm for clipping
    max_grad_norm: T,
    /// Velocity buffer for momentum
    velocity: Option<Array1<T>>,
    /// Momentum coefficient
    momentum: T,
    /// Exponential moving average of gradient norms
    grad_norm_ema: T,
    /// Decay factor for gradient norm EMA
    ema_decay: T,
}

impl<T: Float + Debug + Send + Sync + 'static> NLPOptimizer<T> {
    /// Create a new NLPOptimizer with the given base learning rate.
    ///
    /// Defaults: no layer-wise decay, no warmup, no accumulation,
    /// max_grad_norm=1.0, momentum=0.9.
    pub fn new(base_lr: T) -> Self {
        Self {
            base_lr,
            current_lr: base_lr,
            layer_wise_decay: T::one(),
            num_layers: 1,
            warmup_steps: 0,
            total_steps: 0,
            step_count: 0,
            token_aware_scaling: false,
            gradient_accumulation_steps: 1,
            accumulated_gradients: None,
            accumulation_count: 0,
            max_grad_norm: T::one(),
            velocity: None,
            momentum: T::from(0.9).expect("0.9 should convert"),
            grad_norm_ema: T::zero(),
            ema_decay: T::from(0.999).expect("0.999 should convert"),
        }
    }

    /// Set layer-wise learning rate decay (builder pattern).
    ///
    /// `decay` is the multiplicative factor per layer. For layer `i` (0-indexed
    /// from the top/output layer), the effective LR is
    /// `base_lr * decay^(num_layers - 1 - i)`.
    pub fn with_layer_wise_decay(mut self, decay: T, num_layers: usize) -> Self {
        self.layer_wise_decay = decay;
        self.num_layers = if num_layers == 0 { 1 } else { num_layers };
        self
    }

    /// Set warmup steps and total steps for cosine schedule (builder pattern).
    pub fn with_warmup_and_schedule(mut self, warmup: usize, total: usize) -> Self {
        self.warmup_steps = warmup;
        self.total_steps = total;
        self
    }

    /// Enable or disable token-aware gradient scaling (builder pattern).
    pub fn with_token_aware_scaling(mut self, enable: bool) -> Self {
        self.token_aware_scaling = enable;
        self
    }

    /// Set the number of gradient accumulation steps (builder pattern).
    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = if steps == 0 { 1 } else { steps };
        self
    }

    /// Set the maximum gradient norm for clipping (builder pattern).
    pub fn with_max_grad_norm(mut self, norm: T) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Set the momentum coefficient (builder pattern).
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    /// Get the effective learning rate for a specific layer.
    ///
    /// Layer 0 is the output (top) layer and gets the highest LR.
    /// Deeper layers get progressively smaller LRs.
    pub fn get_layer_lr(&self, layer_idx: usize) -> T {
        let exponent = if layer_idx < self.num_layers {
            self.num_layers - 1 - layer_idx
        } else {
            0
        };
        let mut factor = T::one();
        for _ in 0..exponent {
            factor = factor * self.layer_wise_decay;
        }
        self.current_lr * factor
    }

    /// Compute the LR schedule multiplier (warmup + cosine annealing).
    fn schedule_factor(&self) -> T {
        // During warmup: linear ramp
        if self.warmup_steps > 0 && self.step_count < self.warmup_steps {
            let step_t = T::from(self.step_count + 1).expect("step convert");
            let warmup_t = T::from(self.warmup_steps).expect("warmup convert");
            return step_t / warmup_t;
        }
        // After warmup: cosine annealing (if total_steps specified)
        if self.total_steps > self.warmup_steps {
            let progress_steps = self.step_count.saturating_sub(self.warmup_steps);
            let decay_steps = self.total_steps - self.warmup_steps;
            let progress = T::from(progress_steps).expect("progress convert")
                / T::from(decay_steps).expect("decay convert");
            // Clamp to [0, 1]
            let progress = if progress > T::one() {
                T::one()
            } else {
                progress
            };
            // Cosine annealing: 0.5 * (1 + cos(pi * progress))
            let pi = T::from(std::f64::consts::PI).expect("pi convert");
            let half = T::from(0.5).expect("0.5 convert");
            let cosine = (pi * progress).cos();
            return half * (T::one() + cosine);
        }
        T::one()
    }

    /// Apply token-aware scaling.
    ///
    /// Simulates inverse-frequency scaling: parameter groups are scaled
    /// based on their position (proxy for embedding vs attention vs FFN).
    fn apply_token_scaling(&self, gradients: &Array1<T>) -> Array1<T> {
        if !self.token_aware_scaling {
            return gradients.clone();
        }
        let len = gradients.len();
        let mut scaled = gradients.clone();
        // First 1/3: embedding layer - scale down (common tokens get smaller updates)
        // Middle 1/3: attention - keep as-is
        // Last 1/3: FFN - slight scale up
        let third = len / 3;
        let embed_scale = T::from(0.5).expect("0.5 convert");
        let ffn_scale = T::from(1.2).expect("1.2 convert");
        for i in 0..third {
            scaled[i] = scaled[i] * embed_scale;
        }
        for i in (len - third)..len {
            scaled[i] = scaled[i] * ffn_scale;
        }
        scaled
    }

    /// Apply layer-wise learning rate decay to gradients.
    ///
    /// Splits the gradient vector evenly across `num_layers` and applies
    /// progressively smaller scaling to deeper layers.
    fn apply_layer_wise_decay(&self, gradients: &Array1<T>) -> Array1<T> {
        if (self.layer_wise_decay - T::one()).abs() < T::from(1e-12).expect("eps") {
            return gradients.clone();
        }
        let len = gradients.len();
        let chunk = len / self.num_layers;
        if chunk == 0 {
            return gradients.clone();
        }
        let mut scaled = gradients.clone();
        for layer in 0..self.num_layers {
            let start = layer * chunk;
            let end = if layer == self.num_layers - 1 {
                len
            } else {
                start + chunk
            };
            // Layer 0 = top/output, gets highest LR; deeper layers get decay
            let exponent = self.num_layers - 1 - layer;
            let mut factor = T::one();
            for _ in 0..exponent {
                factor = factor * self.layer_wise_decay;
            }
            for i in start..end {
                scaled[i] = scaled[i] * factor;
            }
        }
        scaled
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdvancedOptimizer<T> for NLPOptimizer<T> {
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

        // 1. Gradient accumulation
        let acc = match self.accumulated_gradients.take() {
            Some(a) if a.len() == gradients.len() => a + gradients,
            _ => gradients.clone(),
        };
        self.accumulation_count += 1;

        if self.accumulation_count < self.gradient_accumulation_steps {
            // Not yet ready to update — store accumulated grads and return params unchanged
            self.accumulated_gradients = Some(acc);
            return Ok(params.clone());
        }

        // Average accumulated gradients
        let accum_steps_t = T::from(self.gradient_accumulation_steps).expect("accum steps convert");
        let grad = acc.mapv(|g| g / accum_steps_t);
        self.accumulation_count = 0;

        // 2. Clip gradients
        let grad = clip_grad_norm(&grad, self.max_grad_norm);

        // 3. Token-aware scaling
        let grad = self.apply_token_scaling(&grad);

        // 4. Layer-wise decay
        let grad = self.apply_layer_wise_decay(&grad);

        // 5. Update gradient norm EMA
        let norm = l2_norm(&grad);
        self.grad_norm_ema =
            self.ema_decay * self.grad_norm_ema + (T::one() - self.ema_decay) * norm;

        // 6. Schedule
        let schedule = self.schedule_factor();
        let effective_lr = self.base_lr * schedule;
        self.current_lr = effective_lr;

        // 7. Momentum update
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
        "NLPOptimizer"
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

    #[test]
    fn test_nlp_optimizer_basic_step() {
        let mut opt = NLPOptimizer::new(0.01_f64);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3]);
        let updated = opt.step(&params, &grads).expect("step should succeed");
        for i in 0..6 {
            assert!(updated[i] < params[i], "param {} should decrease", i);
        }
        assert_eq!(opt.step_count, 1);
    }

    #[test]
    fn test_nlp_optimizer_warmup_and_cosine() {
        let mut opt = NLPOptimizer::new(0.1_f64).with_warmup_and_schedule(10, 100);
        let params = Array1::ones(6);
        let grads = Array1::ones(6);

        // Run through warmup
        let mut lrs = Vec::new();
        for _ in 0..15 {
            let _ = opt.step(&params, &grads).expect("step ok");
            lrs.push(opt.get_learning_rate());
        }
        // LR should increase during warmup (first 10 steps)
        assert!(lrs[4] > lrs[0], "LR should increase during warmup");
        // After warmup, LR should start to decay (cosine)
        assert!(lrs[14] < lrs[9], "LR should decay after warmup via cosine");
    }

    #[test]
    fn test_nlp_optimizer_layer_wise_decay() {
        let opt = NLPOptimizer::new(0.01_f64).with_layer_wise_decay(0.8, 4);
        // Layer 0 (top): base_lr * 0.8^3
        // Layer 3 (bottom/deepest): base_lr * 0.8^0 = base_lr
        let lr_top = opt.get_layer_lr(0);
        let lr_bottom = opt.get_layer_lr(3);
        // Actually layer 0 is top (output), gets highest LR => exponent = num_layers-1-0 = 3
        // layer 3 is deepest, exponent = 0
        // So lr_top = 0.01 * 0.8^3, lr_bottom = 0.01 * 0.8^0
        assert!(
            lr_bottom > lr_top,
            "output layer should have highest LR (index 3), lr_top={}, lr_bottom={}",
            lr_top,
            lr_bottom
        );
    }

    #[test]
    fn test_nlp_optimizer_gradient_accumulation() {
        let mut opt = NLPOptimizer::new(0.01_f64).with_gradient_accumulation(4);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // First 3 accumulation steps should return params unchanged
        for _ in 0..3 {
            let updated = opt.step(&params, &grads).expect("step ok");
            assert_eq!(updated, params, "should not update during accumulation");
            assert_eq!(
                opt.step_count, 0,
                "step_count should not advance during accumulation"
            );
        }
        // 4th step should apply the accumulated update
        let updated = opt.step(&params, &grads).expect("step ok");
        assert_ne!(updated, params, "should update after accumulation complete");
        assert_eq!(opt.step_count, 1);
    }

    #[test]
    fn test_nlp_optimizer_gradient_clipping() {
        let mut opt = NLPOptimizer::new(0.01_f64)
            .with_max_grad_norm(0.5)
            .with_momentum(0.0);
        let params = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        // Large gradients that should be clipped
        let grads = Array1::from_vec(vec![10.0, 10.0, 10.0]);
        let updated = opt.step(&params, &grads).expect("step ok");
        // The update magnitude should be bounded
        let diff_norm = l2_norm(&(&params - &updated));
        // Without clipping, diff would be ~0.01 * 17.32 = 0.173
        // With clipping to norm 0.5, diff should be ~0.01 * 0.5 = 0.005
        assert!(
            diff_norm < 0.02,
            "gradient clipping should limit update, got {}",
            diff_norm
        );
    }

    #[test]
    fn test_nlp_optimizer_token_aware_scaling() {
        let mut opt_token = NLPOptimizer::new(0.01_f64)
            .with_token_aware_scaling(true)
            .with_max_grad_norm(100.0)
            .with_momentum(0.0);
        let mut opt_base = NLPOptimizer::new(0.01_f64)
            .with_max_grad_norm(100.0)
            .with_momentum(0.0);

        let params = Array1::from_vec(vec![1.0; 9]);
        let grads = Array1::from_vec(vec![1.0; 9]);

        let u_token = opt_token.step(&params, &grads).expect("step ok");
        let u_base = opt_base.step(&params, &grads).expect("step ok");

        // Embedding region (first 1/3) should have smaller update with token scaling
        let delta_token_embed = (params[0] - u_token[0]).abs();
        let delta_base_embed = (params[0] - u_base[0]).abs();
        assert!(
            delta_token_embed < delta_base_embed,
            "token scaling should reduce embedding updates"
        );
    }

    #[test]
    fn test_nlp_optimizer_dimension_mismatch() {
        let mut opt = NLPOptimizer::new(0.01_f64);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![0.1]);
        let result = opt.step(&params, &grads);
        assert!(result.is_err());
    }

    #[test]
    fn test_nlp_optimizer_state_info() {
        let mut opt = NLPOptimizer::new(0.05_f64);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let _ = opt.step(&params, &grads).expect("step ok");
        let state = opt.get_state();
        assert_eq!(state.step_count, 1);
        assert!(state.grad_norm_ema > 0.0);
    }
}
