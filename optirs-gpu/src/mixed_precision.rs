// Mixed precision training support with tensor cores
//
// This module provides automatic mixed precision (AMP) training capabilities
// leveraging tensor cores on modern GPUs for accelerated computation.

use scirs2_core::ndarray_ext::{Array, Dimension};
use num_traits::Float;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::gpu::{GpuOptimError, GpuOptimizerConfig};

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBuffer, GpuContext};

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Initial loss scale factor
    pub init_scale: f32,

    /// Growth factor for loss scaling
    pub growth_factor: f32,

    /// Backoff factor when overflow detected
    pub backoff_factor: f32,

    /// Growth interval (steps between scale increases)
    pub growth_interval: i32,

    /// Minimum loss scale
    pub min_scale: f32,

    /// Maximum loss scale
    pub max_scale: f32,

    /// Enable gradient clipping
    pub gradient_clipping: bool,

    /// Maximum gradient norm
    pub max_grad_norm: f32,

    /// Use bfloat16 instead of float16
    pub use_bfloat16: bool,

    /// Enable tensor core operations
    pub use_tensor_cores: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            min_scale: 1.0,
            max_scale: 65536.0 * 128.0,
            gradient_clipping: true,
            max_grad_norm: 1.0,
            use_bfloat16: false,
            use_tensor_cores: true,
        }
    }
}

/// Dynamic loss scaler for mixed precision training
pub struct DynamicLossScaler {
    /// Current scale factor
    scale: f32,

    /// Configuration
    config: MixedPrecisionConfig,

    /// Steps since last scale update
    growth_tracker: i32,

    /// Overflow history for debugging
    overflow_history: Vec<bool>,
}

impl DynamicLossScaler {
    /// Create a new dynamic loss scaler
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            scale: config.init_scale,
            config,
            growth_tracker: 0,
            overflow_history: Vec::with_capacity(100),
        }
    }

    /// Get current scale
    pub fn get_scale(&self) -> f32 {
        self.scale
    }

    /// Update scale based on overflow status
    pub fn update(&mut self, hasoverflow: bool) {
        self.overflow_history.push(has_overflow);
        if self.overflow_history.len() > 100 {
            self.overflow_history.remove(0);
        }

        if has_overflow {
            // Decrease scale on _overflow
            self.scale = (self.scale * self.config.backoff_factor).max(self.config.min_scale);
            self.growth_tracker = 0;
        } else {
            // Increase scale if stable
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.scale = (self.scale * self.config.growth_factor).min(self.config.max_scale);
                self.growth_tracker = 0;
            }
        }
    }

    /// Get overflow statistics
    pub fn get_overflow_stats(&self) -> OverflowStats {
        let total = self.overflow_history.len();
        let overflows = self.overflow_history.iter().filter(|&&x| x).count();

        OverflowStats {
            total_steps: total,
            overflow_count: overflows,
            overflow_rate: if total > 0 {
                overflows as f32 / total as f32
            } else {
                0.0
            },
            current_scale: self.scale,
        }
    }
}

/// Overflow statistics
#[derive(Debug, Clone)]
pub struct OverflowStats {
    pub total_steps: usize,
    pub overflow_count: usize,
    pub overflow_rate: f32,
    pub current_scale: f32,
}

/// Mixed precision optimizer wrapper
pub struct MixedPrecisionOptimizer<O, A: Float> {
    /// Underlying optimizer
    optimizer: O,

    /// Loss scaler
    scaler: DynamicLossScaler,

    /// GPU context
    gpu_context: Option<Arc<GpuContext>>,

    /// FP32 master weights
    master_weights: Option<GpuBuffer<f32>>,

    /// FP16/BF16 model weights
    model_weights_half: Option<GpuBuffer<u16>>,

    /// Gradient buffer (FP16/BF16)
    gradients_half: Option<GpuBuffer<u16>>,

    /// Overflow detection buffer
    overflow_flag: Option<GpuBuffer<i32>>,

    /// Configuration
    config: MixedPrecisionConfig,

    /// Phantom data
    _phantom: PhantomData<A>,
}

impl<O, A: Float + Send + Sync> MixedPrecisionOptimizer<O, A> {
    /// Create a new mixed precision optimizer
    pub fn new(optimizer: O, config: MixedPrecisionConfig) -> Self {
        let scaler = DynamicLossScaler::new(config.clone());

        Self {
            optimizer,
            scaler,
            gpu_context: None,
            master_weights: None,
            model_weights_half: None,
            gradients_half: None,
            overflow_flag: None,
            config_phantom: PhantomData,
        }
    }

    /// Initialize GPU resources
    pub fn initialize_gpu(
        &mut self,
        param_count: usize,
        gpu_config: GpuOptimizerConfig,
    ) -> Result<(), GpuOptimError> {
        #[cfg(feature = "gpu")]
        {
            let context = Arc::new(GpuContext::new(gpu_config.backend)?);

            // Allocate buffers
            self.master_weights = Some(context.create_buffer::<f32>(param_count));
            self.model_weights_half = Some(context.create_buffer::<u16>(param_count));
            self.gradients_half = Some(context.create_buffer::<u16>(param_count));
            self.overflow_flag = Some(context.create_buffer::<i32>(1));

            self.gpu_context = Some(context);
        }

        Ok(())
    }

    /// Scale gradients before backward pass
    pub fn scale_loss(&self, loss: A) -> A {
        loss * A::from(self.scaler.get_scale()).unwrap()
    }

    /// Unscale gradients and check for overflow
    pub fn unscale_and_check_overflow<D>(
        &mut self,
        gradients: &mut Array<A, D>,
    ) -> Result<bool, GpuOptimError>
    where
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref context) = self.gpu_context {
                let kernel = context.get_kernel("scale_gradients_check_overflow_f16")?;

                // Set kernel parameters
                kernel.set_buffer("gradients", self.gradients_half.as_ref().unwrap());
                kernel.set_buffer("has_overflow", self.overflow_flag.as_ref().unwrap());
                kernel.set_f32("scale_factor", 1.0 / self.scaler.get_scale());
                kernel.set_i32("n", gradients.len() as i32);

                // Launch kernel
                let (grid_size, block_size) =
                    crate::gpu::utils::calculate_block_size(gradients.len(), 256);
                kernel.dispatch([grid_size as u32, 1, 1]);

                // Check overflow flag
                let mut overflow_flag = vec![0i32; 1];
                self.overflow_flag
                    .as_ref()
                    .unwrap()
                    .copy_to_host(&mut overflow_flag);

                let has_overflow = overflow_flag[0] != 0;
                self.scaler.update(has_overflow);

                return Ok(has_overflow);
            }
        }

        // CPU fallback
        let scale = A::from(self.scaler.get_scale()).unwrap();
        let mut has_overflow = false;

        gradients.mapv_inplace(|g| {
            let unscaled = g / scale;
            if !unscaled.is_finite() {
                has_overflow = true;
            }
            unscaled
        });

        self.scaler.update(has_overflow);
        Ok(has_overflow)
    }

    /// Get overflow statistics
    pub fn get_overflow_stats(&self) -> OverflowStats {
        self.scaler.get_overflow_stats()
    }

    /// Get current loss scale
    pub fn get_scale(&self) -> f32 {
        self.scaler.get_scale()
    }
}

/// Tensor core optimization utilities
pub mod tensor_core_utils {
    use super::*;

    /// Pad tensor dimensions for tensor core alignment
    pub fn pad_for_tensor_cores(size: usize, alignment: usize) -> usize {
        (_size + alignment - 1) / alignment * alignment
    }

    /// Check if dimensions are tensor core friendly
    pub fn is_tensor_core_friendly(m: usize, n: usize, k: usize) -> bool {
        // Tensor cores work best with dimensions divisible by 16
        m % 16 == 0 && n % 16 == 0 && k % 16 == 0
    }

    /// Get optimal matrix multiplication configuration
    pub fn get_optimal_gemm_config(m: usize, n: usize, k: usize) -> GemmConfig {
        if is_tensor_core_friendly(m, n, k) {
            GemmConfig {
                use_tensor_cores: true,
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                threads_per_block: 256,
            }
        } else {
            GemmConfig {
                use_tensor_cores: false,
                tile_m: 64,
                tile_n: 64,
                tile_k: 16,
                threads_per_block: 256,
            }
        }
    }
}

/// GEMM configuration for tensor cores
#[derive(Debug, Clone)]
pub struct GemmConfig {
    pub use_tensor_cores: bool,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub threads_per_block: usize,
}

/// Mixed precision training utilities
pub struct MixedPrecisionUtils;

impl MixedPrecisionUtils {
    /// Convert FP32 to FP16 with saturation
    pub fn float_to_half(values: &[f32]) -> Vec<u16> {
        _values
            .iter()
            .map(|&v| {
                let clamped = v.max(-65504.0).min(65504.0);
                F16::from_f32(clamped).to_bits()
            })
            .collect()
    }

    /// Convert FP16 to FP32
    pub fn half_to_float(values: &[u16]) -> Vec<f32> {
        _values
            .iter()
            .map(|&v| F16::from_bits(v).to_f32())
            .collect()
    }

    /// Check if GPU supports tensor cores
    pub fn has_tensor_core_support(_gpucontext: &GpuContext) -> bool {
        // Check compute capability
        // Volta (7.0), Turing (7.5), Ampere (8.0+) have tensor cores
        true // Placeholder
    }
}

// F16 type placeholder (would use half crate in real implementation)
struct F16(u16);

impl F16 {
    fn from_f32(v: f32) -> Self {
        // Simplified conversion
        F16(0)
    }

    fn to_f32(&self) -> f32 {
        0.0
    }

    fn from_bits(bits: u16) -> Self {
        F16(_bits)
    }

    fn to_bits(&self) -> u16 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.init_scale, 65536.0);
        assert_eq!(config.growth_factor, 2.0);
        assert!(config.use_tensor_cores);
    }

    #[test]
    fn test_dynamic_loss_scaler() {
        let config = MixedPrecisionConfig::default();
        let mut scaler = DynamicLossScaler::new(config);

        assert_eq!(scaler.get_scale(), 65536.0);

        // Test scale decrease on overflow
        scaler.update(true);
        assert_eq!(scaler.get_scale(), 32768.0);

        // Test scale increase after stable steps
        for _ in 0..2000 {
            scaler.update(false);
        }
        assert!(scaler.get_scale() > 32768.0);
    }

    #[test]
    fn test_tensor_core_alignment() {
        assert_eq!(tensor_core_utils::pad_for_tensor_cores(100, 16), 112);
        assert_eq!(tensor_core_utils::pad_for_tensor_cores(128, 16), 128);

        assert!(tensor_core_utils::is_tensor_core_friendly(256, 256, 128));
        assert!(!tensor_core_utils::is_tensor_core_friendly(100, 100, 100));
    }
}
