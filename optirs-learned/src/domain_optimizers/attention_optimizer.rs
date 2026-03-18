//! Attention Optimizer
//!
//! A domain-specific optimizer for attention mechanisms that incorporates
//! per-head gradient scaling, attention entropy regularization, and
//! head-importance-aware updates.
//!
//! # Key Features
//! - **Head-wise scaling**: Scale gradients per attention head based on importance
//! - **Entropy regularization**: Encourage diverse attention distributions
//! - **Adaptive head scales**: Automatically adjust per-head scales from attention weights
//! - **Warmup support**: Linear learning rate warmup

use crate::domain_optimizers::{l2_norm, AdvancedOptimizer, OptimizerStateInfo};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Attention-specific optimizer with per-head scaling and entropy regularization.
///
/// This optimizer is designed for multi-head attention layers. It can:
/// - Scale gradients differently for each attention head
/// - Apply entropy-based regularization to prevent attention collapse
/// - Dynamically adjust head importance from observed attention weights
#[derive(Debug, Clone)]
pub struct AttentionOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Base learning rate
    base_lr: T,
    /// Current effective learning rate
    current_lr: T,
    /// Number of attention heads
    num_heads: usize,
    /// Whether to apply per-head gradient scaling
    head_wise_scaling: bool,
    /// Strength of attention entropy regularization
    attention_entropy_reg: T,
    /// Current optimization step
    step_count: usize,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Per-head gradient scaling factors
    head_gradient_scales: Vec<T>,
    /// Velocity buffer for momentum
    velocity: Option<Array1<T>>,
    /// Momentum coefficient
    momentum: T,
    /// Exponential moving average of gradient norms
    grad_norm_ema: T,
    /// Decay factor for gradient norm EMA
    ema_decay: T,
}

impl<T: Float + Debug + Send + Sync + 'static> AttentionOptimizer<T> {
    /// Create a new AttentionOptimizer.
    ///
    /// All heads start with equal gradient scale (1.0).
    pub fn new(base_lr: T, num_heads: usize) -> Self {
        let num_heads = if num_heads == 0 { 1 } else { num_heads };
        Self {
            base_lr,
            current_lr: base_lr,
            num_heads,
            head_wise_scaling: false,
            attention_entropy_reg: T::zero(),
            step_count: 0,
            warmup_steps: 0,
            head_gradient_scales: vec![T::one(); num_heads],
            velocity: None,
            momentum: T::from(0.9).expect("0.9 convert"),
            grad_norm_ema: T::zero(),
            ema_decay: T::from(0.999).expect("0.999 convert"),
        }
    }

    /// Enable or disable per-head gradient scaling (builder pattern).
    pub fn with_head_wise_scaling(mut self, enable: bool) -> Self {
        self.head_wise_scaling = enable;
        self
    }

    /// Set the attention entropy regularization strength (builder pattern).
    ///
    /// A positive value encourages higher entropy (more uniform attention).
    /// A value of 0 disables entropy regularization.
    pub fn with_attention_entropy_reg(mut self, strength: T) -> Self {
        self.attention_entropy_reg = strength;
        self
    }

    /// Set the number of warmup steps (builder pattern).
    pub fn with_warmup(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set the momentum coefficient (builder pattern).
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    /// Update per-head gradient scales based on attention weight distributions.
    ///
    /// Computes the entropy of each head's attention weights. Heads with low
    /// entropy (concentrated attention) get *higher* gradient scales to encourage
    /// them to diversify. Heads with high entropy get lower scales.
    ///
    /// # Arguments
    /// * `attention_weights` - Slice of `num_heads` arrays, each representing
    ///   the attention distribution for one head (should sum to ~1).
    pub fn update_head_scales(&mut self, attention_weights: &[Array1<T>]) -> Result<()> {
        if attention_weights.len() != self.num_heads {
            return Err(OptimError::InvalidConfig(format!(
                "Expected {} attention weight arrays, got {}",
                self.num_heads,
                attention_weights.len()
            )));
        }

        let entropies: Vec<T> = attention_weights
            .iter()
            .map(Self::compute_attention_entropy)
            .collect();

        // Compute mean entropy
        let sum_entropy = entropies.iter().fold(T::zero(), |acc, &e| acc + e);
        let mean_entropy = sum_entropy / T::from(self.num_heads).expect("num_heads convert");
        let epsilon = T::from(1e-8).expect("epsilon convert");

        // Heads with below-average entropy get scale > 1, above-average get scale < 1
        // scale_i = 1 + alpha * (mean_entropy - entropy_i) / (mean_entropy + eps)
        let alpha = T::from(0.5).expect("alpha convert");
        for (i, &ent) in entropies.iter().enumerate() {
            let deviation = (mean_entropy - ent) / (mean_entropy + epsilon);
            self.head_gradient_scales[i] = T::one() + alpha * deviation;
            // Clamp to reasonable range [0.5, 2.0]
            let lower = T::from(0.5).expect("lower convert");
            let upper = T::from(2.0).expect("upper convert");
            if self.head_gradient_scales[i] < lower {
                self.head_gradient_scales[i] = lower;
            }
            if self.head_gradient_scales[i] > upper {
                self.head_gradient_scales[i] = upper;
            }
        }

        Ok(())
    }

    /// Compute the Shannon entropy of an attention distribution.
    ///
    /// H = -sum(p * log(p)) for p > 0
    ///
    /// Returns 0 if the distribution is empty.
    pub fn compute_attention_entropy(attention_weights: &Array1<T>) -> T {
        if attention_weights.is_empty() {
            return T::zero();
        }
        let epsilon = T::from(1e-12).expect("epsilon convert");
        let mut entropy = T::zero();
        for &p in attention_weights.iter() {
            if p > epsilon {
                entropy = entropy - p * p.ln();
            }
        }
        // Ensure non-negative (numerical errors can cause tiny negatives)
        if entropy < T::zero() {
            T::zero()
        } else {
            entropy
        }
    }

    /// Apply per-head gradient scaling.
    ///
    /// Splits the gradient vector evenly into `num_heads` segments and
    /// scales each by the corresponding head scale factor.
    fn apply_head_scaling(&self, gradients: &Array1<T>) -> Array1<T> {
        if !self.head_wise_scaling {
            return gradients.clone();
        }
        let len = gradients.len();
        let chunk = len / self.num_heads;
        if chunk == 0 {
            return gradients.clone();
        }
        let mut scaled = gradients.clone();
        for head in 0..self.num_heads {
            let start = head * chunk;
            let end = if head == self.num_heads - 1 {
                len
            } else {
                start + chunk
            };
            let scale = self.head_gradient_scales[head];
            for i in start..end {
                scaled[i] = scaled[i] * scale;
            }
        }
        scaled
    }

    /// Compute entropy regularization gradient.
    ///
    /// Adds a small gradient signal that encourages higher entropy
    /// (more uniform attention) in the parameter space. This is a
    /// simplified proxy: it adds a constant push proportional to
    /// `attention_entropy_reg` toward zero (weight decay-like).
    fn entropy_regularization_gradient(&self, params: &Array1<T>) -> Array1<T> {
        if self.attention_entropy_reg <= T::from(1e-12).expect("eps") {
            return Array1::zeros(params.len());
        }
        // L2-style regularization as proxy for entropy regularization
        params.mapv(|p| p * self.attention_entropy_reg)
    }

    /// Compute the warmup factor.
    fn warmup_factor(&self) -> T {
        if self.warmup_steps == 0 || self.step_count >= self.warmup_steps {
            return T::one();
        }
        let step_t = T::from(self.step_count + 1).expect("step convert");
        let warmup_t = T::from(self.warmup_steps).expect("warmup convert");
        step_t / warmup_t
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdvancedOptimizer<T> for AttentionOptimizer<T> {
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

        // 1. Warmup
        let warmup = self.warmup_factor();
        let effective_lr = self.base_lr * warmup;
        self.current_lr = effective_lr;

        // 2. Per-head gradient scaling
        let grad = self.apply_head_scaling(gradients);

        // 3. Entropy regularization gradient
        let entropy_grad = self.entropy_regularization_gradient(params);
        let grad = &grad + &entropy_grad;

        // 4. Update gradient norm EMA
        let norm = l2_norm(&grad);
        self.grad_norm_ema =
            self.ema_decay * self.grad_norm_ema + (T::one() - self.ema_decay) * norm;

        // 5. Momentum update
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
        "AttentionOptimizer"
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
    fn test_attention_optimizer_basic_step() {
        let mut opt = AttentionOptimizer::new(0.01_f64, 4);
        let params = Array1::from_vec(vec![1.0; 8]);
        let grads = Array1::from_vec(vec![0.1; 8]);
        let updated = opt.step(&params, &grads).expect("step should succeed");
        for i in 0..8 {
            assert!(updated[i] < params[i], "param {} should decrease", i);
        }
        assert_eq!(opt.step_count, 1);
        assert_eq!(opt.name(), "AttentionOptimizer");
    }

    #[test]
    fn test_attention_optimizer_head_wise_scaling() {
        let mut opt = AttentionOptimizer::new(0.01_f64, 2)
            .with_head_wise_scaling(true)
            .with_momentum(0.0);

        // Manually set head scales: head 0 = 2.0, head 1 = 0.5
        opt.head_gradient_scales = vec![2.0, 0.5];

        let params = Array1::from_vec(vec![1.0; 4]);
        let grads = Array1::from_vec(vec![1.0; 4]);
        let updated = opt.step(&params, &grads).expect("step ok");

        // Head 0 (first 2 params) should have larger update than head 1
        let delta_h0 = (params[0] - updated[0]).abs();
        let delta_h1 = (params[2] - updated[2]).abs();
        assert!(
            delta_h0 > delta_h1,
            "head 0 (scale=2.0) should have larger update than head 1 (scale=0.5)"
        );
    }

    #[test]
    fn test_attention_entropy_computation() {
        // Uniform distribution: entropy = ln(n)
        let uniform = Array1::from_vec(vec![0.25_f64, 0.25, 0.25, 0.25]);
        let entropy = AttentionOptimizer::<f64>::compute_attention_entropy(&uniform);
        let expected = (4.0_f64).ln(); // ln(4) ≈ 1.386
        assert!(
            (entropy - expected).abs() < 1e-6,
            "uniform entropy should be ln(4), got {}",
            entropy
        );

        // Peaked distribution: low entropy
        let peaked = Array1::from_vec(vec![0.97_f64, 0.01, 0.01, 0.01]);
        let entropy_peaked = AttentionOptimizer::<f64>::compute_attention_entropy(&peaked);
        assert!(
            entropy_peaked < entropy,
            "peaked distribution should have lower entropy"
        );

        // Empty distribution
        let empty = Array1::from_vec(vec![]);
        let entropy_empty = AttentionOptimizer::<f64>::compute_attention_entropy(&empty);
        assert!((entropy_empty - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_attention_update_head_scales() {
        let mut opt = AttentionOptimizer::new(0.01_f64, 3);

        // Head 0: uniform (high entropy), Head 1: peaked (low entropy), Head 2: medium
        let weights = vec![
            Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]),
            Array1::from_vec(vec![0.97, 0.01, 0.01, 0.01]),
            Array1::from_vec(vec![0.5, 0.3, 0.1, 0.1]),
        ];

        opt.update_head_scales(&weights).expect("update ok");

        // Head 1 (lowest entropy) should get highest scale (encouraged to diversify)
        assert!(
            opt.head_gradient_scales[1] > opt.head_gradient_scales[0],
            "low-entropy head should get higher scale: head1={}, head0={}",
            opt.head_gradient_scales[1],
            opt.head_gradient_scales[0]
        );
    }

    #[test]
    fn test_attention_update_head_scales_wrong_count() {
        let mut opt = AttentionOptimizer::new(0.01_f64, 2);
        let weights = vec![Array1::from_vec(vec![0.5, 0.5])]; // only 1, need 2
        let result = opt.update_head_scales(&weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_optimizer_warmup() {
        let mut opt = AttentionOptimizer::new(0.1_f64, 2).with_warmup(10);
        let params = Array1::from_vec(vec![1.0; 4]);
        let grads = Array1::from_vec(vec![1.0; 4]);

        let _ = opt.step(&params, &grads).expect("step ok");
        let lr_early = opt.get_learning_rate();

        for _ in 0..9 {
            let _ = opt.step(&params, &grads).expect("step ok");
        }
        let lr_after_warmup = opt.get_learning_rate();

        assert!(
            lr_after_warmup > lr_early,
            "LR should increase during warmup"
        );
    }

    #[test]
    fn test_attention_entropy_regularization() {
        let mut opt_reg = AttentionOptimizer::new(0.01_f64, 2)
            .with_attention_entropy_reg(0.1)
            .with_momentum(0.0);
        let mut opt_base = AttentionOptimizer::new(0.01_f64, 2).with_momentum(0.0);

        let params = Array1::from_vec(vec![2.0; 4]);
        let grads = Array1::zeros(4); // zero gradients

        let u_reg = opt_reg.step(&params, &grads).expect("step ok");
        let u_base = opt_base.step(&params, &grads).expect("step ok");

        // With zero gradients, the base optimizer should not move params
        // but the regularized one should push toward zero
        let diff_reg = l2_norm(&(&params - &u_reg));
        let diff_base = l2_norm(&(&params - &u_base));
        assert!(
            diff_reg > diff_base,
            "entropy regularization should cause updates even with zero gradients"
        );
    }

    #[test]
    fn test_attention_optimizer_dimension_mismatch() {
        let mut opt = AttentionOptimizer::new(0.01_f64, 2);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![0.1]);
        let result = opt.step(&params, &grads);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_optimizer_state_info() {
        let mut opt = AttentionOptimizer::new(0.05_f64, 4);
        let params = Array1::from_vec(vec![1.0; 8]);
        let grads = Array1::from_vec(vec![0.1; 8]);
        let _ = opt.step(&params, &grads).expect("step ok");
        let state = opt.get_state();
        assert_eq!(state.step_count, 1);
        assert!(state.grad_norm_ema > 0.0);
    }
}
