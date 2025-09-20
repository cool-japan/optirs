// GPU-accelerated LAMB optimizer implementation

use scirs2_core::ndarray_ext::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::sync::Arc;

use crate::error::Result as OptimResult;
use crate::gpu::{GpuOptimError, GpuOptimizerConfig, GpuOptimizerMemory};
use crate::optimizers::{Optimizer, LAMB};

#[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
use scirs2_core::gpu::{GpuContext, GpuKernelHandle};

/// GPU-accelerated LAMB optimizer
pub struct LAMBGpu<A: Float + ScalarOperand + Debug> {
    /// CPU LAMB optimizer for fallback
    cpu_optimizer: LAMB<A>,

    /// GPU memory manager
    gpu_memory: Option<GpuOptimizerMemory<A>>,

    /// GPU kernel handle for LAMB update
    kernel_handle: Option<Arc<GpuKernelHandle>>,

    /// GPU kernel handle for norm computation
    norm_kernel_handle: Option<Arc<GpuKernelHandle>>,

    /// Whether optimizer is on GPU
    on_gpu: bool,

    /// Step count for bias correction
    step_count: usize,

    /// Buffer for parameter norms (per layer)
    param_norms_gpu: Option<scirs2_core::gpu::GpuBuffer<A>>,

    /// Buffer for update norms (per layer)
    update_norms_gpu: Option<scirs2_core::gpu::GpuBuffer<A>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> LAMBGpu<A> {
    /// Create a new GPU-accelerated LAMB optimizer
    pub fn new(_learningrate: A) -> Self {
        Self {
            cpu_optimizer: LAMB::new(_learning_rate),
            gpu_memory: None,
            kernel_handle: None,
            norm_kernel_handle: None,
            on_gpu: false,
            step_count: 0,
            param_norms_gpu: None,
            update_norms_gpu: None,
        }
    }

    /// Create with full configuration
    pub fn new_with_config(
        learning_rate: A,
        beta1: A,
        beta2: A,
        epsilon: A,
        weight_decay: A,
    ) -> Self {
        Self {
            cpu_optimizer: LAMB::new_with_config(
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            ),
            gpu_memory: None,
            kernel_handle: None,
            norm_kernel_handle: None,
            on_gpu: false,
            step_count: 0,
            param_norms_gpu: None,
            update_norms_gpu: None,
        }
    }

    /// Initialize GPU resources
    pub fn initialize_gpu(
        &mut self,
        size: usize,
        config: GpuOptimizerConfig,
    ) -> Result<(), GpuOptimError> {
        // Create GPU memory manager
        let mut gpu_memory = GpuOptimizerMemory::new(size, config)?;
        gpu_memory.allocate()?;

        // Load LAMB kernels
        #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
        {
            let kernel_name = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                "lamb_update_fused_f32"
            } else if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f64>() {
                "lamb_update_fused_f64"
            } else {
                return Err(GpuOptimError::UnsupportedOperation(
                    "Unsupported data type for GPU LAMB".to_string(),
                ));
            };

            let kernel_handle = gpu_memory.context().get_kernel(kernel_name)?;
            self.kernel_handle = Some(Arc::new(kernel_handle));

            // Load norm computation kernel
            let norm_kernel = gpu_memory.context().get_kernel("compute_layer_norms_f32")?;
            self.norm_kernel_handle = Some(Arc::new(norm_kernel));

            // Allocate norm buffers (assuming single layer for now)
            self.param_norms_gpu = Some(gpu_memory.context().create_buffer::<A>(2)); // [param_norm, trust_ratio]
            self.update_norms_gpu = Some(gpu_memory.context().create_buffer::<A>(2));
        }

        self.gpu_memory = Some(gpu_memory);
        Ok(())
    }

    /// Move optimizer state to GPU
    pub fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
        if self.gpu_memory.is_none() {
            return Err(GpuOptimError::NotInitialized);
        }

        self.on_gpu = true;
        Ok(())
    }

    /// Move optimizer state back to CPU
    pub fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.on_gpu = false;
        Ok(())
    }

    /// Perform optimization step on GPU
    pub fn step_gpu<S1, S2, D>(
        &mut self,
        params: &mut ArrayBase<S1, D>,
        gradients: &ArrayBase<S2, D>,
    ) -> Result<(), GpuOptimError>
    where
        S1: DataMut<Elem = A>,
        S2: Data<Elem = A>,
        D: Dimension,
    {
        if !self.on_gpu {
            return Err(GpuOptimError::InvalidState(
                "Optimizer not on GPU".to_string(),
            ));
        }

        let gpu_memory = self
            .gpu_memory
            .as_mut()
            .ok_or(GpuOptimError::NotInitialized)?;

        let kernel = self
            .kernel_handle
            .as_ref()
            .ok_or(GpuOptimError::NotInitialized)?;

        // Copy data to GPU
        gpu_memory.copy_params_to_gpu(params)?;

        // Copy gradients to GPU
        if let Some(ref grads_gpu) = gpu_memory.grads_gpu {
            let grads_slice = gradients.as_slice().ok_or_else(|| {
                GpuOptimError::InvalidState("Gradients must be contiguous".to_string())
            })?;
            grads_gpu.copy_from_host(grads_slice);
        }

        self.step_count += 1;

        // Compute bias correction terms
        let bias_correction1 = A::one() - self.cpu_optimizer.beta1.powi(self.step_count as i32);
        let bias_correction2 = A::one() - self.cpu_optimizer.beta2.powi(self.step_count as i32);

        // Set kernel parameters
        #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
        {
            kernel.set_buffer("params", gpu_memory.params_gpu.as_ref().unwrap());
            kernel.set_buffer("grads", gpu_memory.grads_gpu.as_ref().unwrap());
            kernel.set_buffer("m", gpu_memory.m_gpu.as_ref().unwrap());
            kernel.set_buffer("v", gpu_memory.v_gpu.as_ref().unwrap());

            // Convert Float values to concrete types for kernel
            if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                kernel.set_f32(
                    "lr",
                    self.cpu_optimizer.get_learning_rate().to_f32().unwrap(),
                );
                kernel.set_f32("beta1", self.cpu_optimizer.beta1.to_f32().unwrap());
                kernel.set_f32("beta2", self.cpu_optimizer.beta2.to_f32().unwrap());
                kernel.set_f32("eps", self.cpu_optimizer.epsilon.to_f32().unwrap());
                kernel.set_f32(
                    "weight_decay",
                    self.cpu_optimizer.weight_decay.to_f32().unwrap(),
                );
                kernel.set_f32("bias_correction1", bias_correction1.to_f32().unwrap());
                kernel.set_f32("bias_correction2", bias_correction2.to_f32().unwrap());
            } else {
                kernel.set_f64(
                    "lr",
                    self.cpu_optimizer.get_learning_rate().to_f64().unwrap(),
                );
                kernel.set_f64("beta1", self.cpu_optimizer.beta1.to_f64().unwrap());
                kernel.set_f64("beta2", self.cpu_optimizer.beta2.to_f64().unwrap());
                kernel.set_f64("eps", self.cpu_optimizer.epsilon.to_f64().unwrap());
                kernel.set_f64(
                    "weight_decay",
                    self.cpu_optimizer.weight_decay.to_f64().unwrap(),
                );
                kernel.set_f64("bias_correction1", bias_correction1.to_f64().unwrap());
                kernel.set_f64("bias_correction2", bias_correction2.to_f64().unwrap());
            }

            kernel.set_i32("n", params.len() as i32);

            // Calculate grid and block dimensions
            let (grid_size, block_size) =
                crate::utils::calculate_block_size(params.len(), 256);

            // Launch kernel with shared memory for norm computation
            let shared_mem_size = (block_size / 32) * 2 * std::mem::size_of::<f32>();
            kernel.dispatch([grid_size as u32, 1, 1]);
        }

        // Copy results back to CPU
        gpu_memory.copy_params_from_gpu(params)?;

        Ok(())
    }

    /// Perform multi-GPU optimization step
    pub fn step_multi_gpu<S1, S2, D>(
        &mut self,
        params: &mut ArrayBase<S1, D>,
        gradients: &ArrayBase<S2, D>,
        num_gpus: usize,
        gpu_id: usize,
    ) -> Result<(), GpuOptimError>
    where
        S1: DataMut<Elem = A>,
        S2: Data<Elem = A>,
        D: Dimension,
    {
        if !self.on_gpu {
            return Err(GpuOptimError::InvalidState(
                "Optimizer not on GPU".to_string(),
            ));
        }

        let gpu_memory = self
            .gpu_memory
            .as_mut()
            .ok_or(GpuOptimError::NotInitialized)?;

        // Load multi-GPU kernel
        #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
        {
            let kernel_name = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                "lamb_update_multi_gpu_f32"
            } else {
                "lamb_update_multi_gpu_f64"
            };

            let kernel = gpu_memory.context().get_kernel(kernel_name)?;

            // Set additional parameters for multi-GPU
            kernel.set_i32("num_gpus", num_gpus as i32);
            kernel.set_i32("gpu_id", gpu_id as i32);

            // Rest of the implementation similar to single GPU
            // but with gradient synchronization across GPUs
        }

        Ok(())
    }
}

// Implement standard Optimizer trait with CPU fallback
impl<A, D> Optimizer<A, D> for LAMBGpu<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> OptimResult<Array<A, D>> {
        if self.on_gpu {
            // Try GPU execution
            let mut params_mut = params.clone();
            match self.step_gpu(&mut params_mut, gradients) {
                Ok(()) => Ok(params_mut),
                Err(e) => {
                    // Fall back to CPU on error
                    eprintln!("GPU execution failed, falling back to CPU: {}", e);
                    self.cpu_optimizer.step(params, gradients)
                }
            }
        } else {
            // Use CPU implementation
            self.cpu_optimizer.step(params, gradients)
        }
    }

    fn get_learning_rate(&self) -> A {
        self.cpu_optimizer.get_learning_rate()
    }

    fn set_learning_rate(&mut self, learningrate: A) {
        self.cpu_optimizer.set_learning_rate(learning_rate);
    }
}

/// Batch LAMB optimizer for multiple parameter groups
pub struct BatchLAMBGpu<A: Float + ScalarOperand + Debug> {
    /// Individual LAMB optimizers for each parameter group
    optimizers: Vec<LAMBGpu<A>>,

    /// Shared GPU context
    gpu_context: Option<Arc<GpuContext>>,

    /// Configuration
    config: GpuOptimizerConfig,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> BatchLAMBGpu<A> {
    /// Create a new batch LAMB optimizer
    pub fn new(_learning_rate: A, numgroups: usize) -> Self {
        let optimizers = (0..num_groups)
            .map(|_| LAMBGpu::new(_learning_rate))
            .collect();

        Self {
            optimizers,
            gpu_context: None,
            config: GpuOptimizerConfig::default(),
        }
    }

    /// Initialize GPU for all parameter groups
    pub fn initialize_gpu(&mut self, groupsizes: &[usize]) -> Result<(), GpuOptimError> {
        // Create shared GPU context
        let gpu_context = Arc::new(GpuContext::new(self.config.backend)?);
        self.gpu_context = Some(gpu_context.clone());

        // Initialize each optimizer
        for (opt, &size) in self.optimizers.iter_mut().zip(group_sizes.iter()) {
            opt.initialize_gpu(size, self.config.clone())?;
        }

        Ok(())
    }

    /// Perform optimization step for all groups
    pub fn step_all<S, D>(
        &mut self,
        params_list: &mut [ArrayBase<S, D>],
        gradients_list: &[ArrayBase<S, D>],
    ) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        if params_list.len() != gradients_list.len() || params_list.len() != self.optimizers.len() {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![self.optimizers.len()],
                actual: vec![params_list.len(), gradients_list.len()],
            });
        }

        // Update each parameter group
        for (i, (opt, (params, grads))) in self
            .optimizers
            .iter_mut()
            .zip(params_list.iter_mut().zip(gradients_list.iter()))
            .enumerate()
        {
            opt.step_gpu(params, grads)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array1;

    #[test]
    fn test_lamb_gpu_creation() {
        let optimizer = LAMBGpu::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert!(!optimizer.on_gpu);
    }

    #[test]
    fn test_lamb_gpu_cpu_fallback() {
        let mut optimizer = LAMBGpu::new(0.001);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Should use CPU implementation when not on GPU
        let result = optimizer.step(&params, &grads);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.len(), 3);
    }

    #[test]
    fn test_batch_lamb_creation() {
        let batch_optimizer = BatchLAMBGpu::<f32>::new(0.001, 3);
        assert_eq!(batch_optimizer.optimizers.len(), 3);
    }
}
