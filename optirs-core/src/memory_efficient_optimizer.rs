//! Memory-efficient optimizer operations
//!
//! This module provides memory-efficient optimization for very large models
//! through gradient accumulation, chunked processing, and memory usage estimation.
//!
//! # Features
//!
//! - Gradient accumulation to reduce memory pressure
//! - Chunked parameter processing for large models
//! - Memory usage estimation and recommendations
//! - Streaming gradient computation
//!
//! # Performance
//!
//! Enables optimization of models with billions of parameters through efficient memory management.

use scirs2_core::ndarray::{s, Array1, ArrayView1, Ix1, ScalarOperand};
use scirs2_core::numeric::{Float, Zero};
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Gradient accumulator for memory-efficient training
///
/// Accumulates gradients over multiple micro-batches before applying updates,
/// reducing memory requirements for large batch training.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::memory_efficient_optimizer::GradientAccumulator;
///
/// let mut accumulator = GradientAccumulator::<f32>::new(1000);
///
/// // Accumulate gradients from 4 micro-batches
/// for _ in 0..4 {
///     let micro_batch_grads = Array1::from_elem(1000, 0.1);
///     accumulator.accumulate(&micro_batch_grads.view()).unwrap();
/// }
///
/// // Get averaged gradients
/// let avg_grads = accumulator.average().unwrap();
/// ```
pub struct GradientAccumulator<A: Float> {
    accumulated: Array1<A>,
    count: usize,
}

impl<A: Float + ScalarOperand + Debug + Zero> GradientAccumulator<A> {
    /// Creates a new gradient accumulator
    ///
    /// # Arguments
    ///
    /// * `size` - Size of gradient vectors
    pub fn new(size: usize) -> Self {
        Self {
            accumulated: Array1::zeros(size),
            count: 0,
        }
    }

    /// Accumulate a gradient vector
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradients to accumulate
    pub fn accumulate(&mut self, gradients: &ArrayView1<A>) -> Result<()> {
        if gradients.len() != self.accumulated.len() {
            return Err(crate::error::OptimError::DimensionMismatch(format!(
                "Gradient size ({}) doesn't match accumulator size ({})",
                gradients.len(),
                self.accumulated.len()
            )));
        }

        self.accumulated = &self.accumulated + gradients;
        self.count += 1;

        Ok(())
    }

    /// Get the number of accumulated gradients
    pub fn count(&self) -> usize {
        self.count
    }

    /// Compute the average of accumulated gradients
    ///
    /// Returns the averaged gradients and resets the accumulator.
    pub fn average(&mut self) -> Result<Array1<A>> {
        if self.count == 0 {
            return Err(crate::error::OptimError::InvalidConfig(
                "No gradients accumulated".to_string(),
            ));
        }

        let scale = A::from(self.count).unwrap();
        let averaged = &self.accumulated / scale;

        // Reset accumulator
        self.reset();

        Ok(averaged)
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.accumulated.fill(A::zero());
        self.count = 0;
    }

    /// Check if accumulator has reached target count
    ///
    /// # Arguments
    ///
    /// * `target` - Target number of accumulations
    pub fn is_ready(&self, target: usize) -> bool {
        self.count >= target
    }
}

/// Chunked optimizer for processing large parameter arrays in chunks
///
/// Enables optimization of very large models by processing parameters
/// in manageable chunks, reducing peak memory usage.
pub struct ChunkedOptimizer<O, A>
where
    O: Optimizer<A, Ix1> + Clone,
    A: Float + ScalarOperand + Debug,
{
    base_optimizer: O,
    chunk_size: usize,
    _phantom: std::marker::PhantomData<A>,
}

impl<O, A> ChunkedOptimizer<O, A>
where
    O: Optimizer<A, Ix1> + Clone,
    A: Float + ScalarOperand + Debug,
{
    /// Creates a new chunked optimizer
    ///
    /// # Arguments
    ///
    /// * `base_optimizer` - Base optimizer to use for each chunk
    /// * `chunk_size` - Size of each chunk (default: 1M elements)
    pub fn new(base_optimizer: O, chunk_size: Option<usize>) -> Self {
        let chunk_size = chunk_size.unwrap_or(1_000_000);

        Self {
            base_optimizer,
            chunk_size,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process parameters in chunks
    ///
    /// # Arguments
    ///
    /// * `params` - Full parameter array
    /// * `gradients` - Full gradient array
    ///
    /// # Returns
    ///
    /// Updated parameters
    pub fn step_chunked(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>> {
        if params.len() != gradients.len() {
            return Err(crate::error::OptimError::DimensionMismatch(format!(
                "Parameters ({}) and gradients ({}) must have same size",
                params.len(),
                gradients.len()
            )));
        }

        let total_size = params.len();
        let mut updated = Array1::zeros(total_size);

        // Process in chunks
        let num_chunks = total_size.div_ceil(self.chunk_size);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * self.chunk_size;
            let end = (start + self.chunk_size).min(total_size);

            // Extract chunk views
            let params_chunk = params.slice(s![start..end]).to_owned();
            let grads_chunk = gradients.slice(s![start..end]).to_owned();

            // Update chunk
            let updated_chunk = self.base_optimizer.step(&params_chunk, &grads_chunk)?;

            // Copy back to result
            updated.slice_mut(s![start..end]).assign(&updated_chunk);
        }

        Ok(updated)
    }

    /// Get the chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Calculate number of chunks for given size
    pub fn num_chunks(&self, total_size: usize) -> usize {
        total_size.div_ceil(self.chunk_size)
    }
}

/// Memory usage estimator for optimizers
///
/// Provides utilities for estimating memory requirements and recommending
/// optimal configurations for different optimizer types.
pub struct MemoryUsageEstimator;

impl MemoryUsageEstimator {
    /// Estimate memory usage for SGD without momentum
    ///
    /// # Arguments
    ///
    /// * `num_params` - Number of parameters
    /// * `dtype_size` - Size of data type in bytes (4 for f32, 8 for f64)
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes
    pub fn sgd(num_params: usize, dtype_size: usize) -> usize {
        // Parameters + gradients
        num_params * dtype_size * 2
    }

    /// Estimate memory usage for SGD with momentum
    ///
    /// # Arguments
    ///
    /// * `num_params` - Number of parameters
    /// * `dtype_size` - Size of data type in bytes (4 for f32, 8 for f64)
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes
    pub fn sgd_with_momentum(num_params: usize, dtype_size: usize) -> usize {
        // Parameters + gradients + velocity
        num_params * dtype_size * 3
    }

    /// Estimate memory usage for Adam optimizer
    ///
    /// # Arguments
    ///
    /// * `num_params` - Number of parameters
    /// * `dtype_size` - Size of data type in bytes (4 for f32, 8 for f64)
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes
    pub fn adam(num_params: usize, dtype_size: usize) -> usize {
        // Parameters + gradients + first moment + second moment
        num_params * dtype_size * 4
    }

    /// Recommend chunk size based on available memory
    ///
    /// # Arguments
    ///
    /// * `total_params` - Total number of parameters
    /// * `available_memory_bytes` - Available memory in bytes
    /// * `dtype_size` - Size of data type in bytes (4 for f32, 8 for f64)
    /// * `optimizer_state_multiplier` - Memory multiplier for optimizer state
    ///
    /// # Returns
    ///
    /// Recommended chunk size
    pub fn recommend_chunk_size(
        total_params: usize,
        available_memory_bytes: usize,
        dtype_size: usize,
        optimizer_state_multiplier: usize,
    ) -> usize {
        let memory_per_param = dtype_size * optimizer_state_multiplier;
        let max_params = available_memory_bytes / memory_per_param;

        // Use 80% of available memory to leave headroom
        let safe_params = (max_params * 80) / 100;

        safe_params.min(total_params).max(1024)
    }

    /// Get recommended accumulation steps for given batch size
    ///
    /// # Arguments
    ///
    /// * `target_batch_size` - Desired effective batch size
    /// * `max_micro_batch_size` - Maximum micro-batch that fits in memory
    ///
    /// # Returns
    ///
    /// Number of gradient accumulation steps
    pub fn recommend_accumulation_steps(
        target_batch_size: usize,
        max_micro_batch_size: usize,
    ) -> usize {
        target_batch_size.div_ceil(max_micro_batch_size)
    }

    /// Estimate peak memory usage during training
    ///
    /// # Arguments
    ///
    /// * `num_params` - Number of parameters
    /// * `batch_size` - Batch size
    /// * `sequence_length` - Sequence length (for transformers, 1 otherwise)
    /// * `dtype_size` - Size of data type in bytes
    /// * `optimizer_type` - Type of optimizer ("sgd", "adam", etc.)
    ///
    /// # Returns
    ///
    /// Estimated peak memory in bytes
    pub fn estimate_peak_memory(
        num_params: usize,
        batch_size: usize,
        sequence_length: usize,
        dtype_size: usize,
        optimizer_type: &str,
    ) -> usize {
        // Model parameters
        let param_memory = num_params * dtype_size;

        // Gradients
        let grad_memory = num_params * dtype_size;

        // Optimizer state
        let optimizer_memory = match optimizer_type {
            "sgd" => num_params * dtype_size,
            "adam" | "adamw" => num_params * dtype_size * 2,
            _ => num_params * dtype_size,
        };

        // Activations (rough estimate: batch_size * sequence_length * hidden_dim)
        let hidden_dim = (num_params as f64).sqrt() as usize;
        let activation_memory = batch_size * sequence_length * hidden_dim * dtype_size;

        param_memory + grad_memory + optimizer_memory + activation_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;
    use approx::assert_relative_eq;

    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = GradientAccumulator::<f32>::new(100);

        // Accumulate some gradients
        let grad1 = Array1::from_elem(100, 1.0);
        let grad2 = Array1::from_elem(100, 2.0);

        accumulator.accumulate(&grad1.view()).unwrap();
        accumulator.accumulate(&grad2.view()).unwrap();

        assert_eq!(accumulator.count(), 2);
        assert!(accumulator.is_ready(2));

        // Get average
        let avg = accumulator.average().unwrap();
        assert_relative_eq!(avg[0], 1.5, epsilon = 1e-6);

        // After average, accumulator should be reset
        assert_eq!(accumulator.count(), 0);
    }

    #[test]
    fn test_chunked_optimizer() {
        let optimizer = SGD::new(0.01);
        let mut chunked_opt = ChunkedOptimizer::new(optimizer, Some(10));

        let params = Array1::from_vec((0..25).map(|i| i as f32).collect());
        let gradients = Array1::from_elem(25, 0.1);

        let updated = chunked_opt.step_chunked(&params, &gradients).unwrap();

        // Verify updates
        assert_eq!(updated.len(), 25);
        assert_relative_eq!(updated[0], 0.0 - 0.01 * 0.1, epsilon = 1e-6);

        // Check number of chunks
        assert_eq!(chunked_opt.num_chunks(25), 3);
    }

    #[test]
    fn test_memory_estimator_sgd() {
        // SGD for 1M parameters (f32)
        let mem = MemoryUsageEstimator::sgd(1_000_000, 4);
        assert_eq!(mem, 8_000_000); // 8 MB

        // SGD with momentum
        let mem = MemoryUsageEstimator::sgd_with_momentum(1_000_000, 4);
        assert_eq!(mem, 12_000_000); // 12 MB
    }

    #[test]
    fn test_memory_estimator_adam() {
        // Adam for 1M parameters (f32)
        let mem = MemoryUsageEstimator::adam(1_000_000, 4);
        assert_eq!(mem, 16_000_000); // 16 MB
    }

    #[test]
    fn test_recommend_chunk_size() {
        // 1GB available, f32, Adam optimizer
        let chunk_size = MemoryUsageEstimator::recommend_chunk_size(
            100_000_000,   // 100M total params
            1_000_000_000, // 1GB available
            4,             // f32
            4,             // Adam state multiplier
        );

        // Should be around 50M params (80% of 62.5M that fits in 1GB)
        assert!(chunk_size > 40_000_000);
        assert!(chunk_size < 60_000_000);
    }

    #[test]
    fn test_recommend_accumulation_steps() {
        let steps = MemoryUsageEstimator::recommend_accumulation_steps(128, 32);
        assert_eq!(steps, 4);

        let steps = MemoryUsageEstimator::recommend_accumulation_steps(100, 32);
        assert_eq!(steps, 4); // Rounds up
    }

    #[test]
    fn test_estimate_peak_memory() {
        let peak = MemoryUsageEstimator::estimate_peak_memory(
            10_000_000, // 10M params
            32,         // batch size
            512,        // sequence length
            4,          // f32
            "adam",
        );

        // Should be substantial (model + optimizer + activations)
        assert!(peak > 100_000_000); // > 100MB
    }
}
