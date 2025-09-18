// GPU-accelerated RMSprop optimizer implementation

use scirs2_core::ndarray_ext::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::sync::Arc;

use crate::error::Result as OptimResult;
use crate::gpu::{GpuOptimError, GpuOptimizerConfig, GpuOptimizerMemory};
use crate::optimizers::{Optimizer, RMSprop};

#[cfg(feature = "gpu")]
use scirs2_core::gpu::GpuKernelHandle;

/// GPU-accelerated RMSprop optimizer
pub struct RMSpropGpu<A: Float + ScalarOperand + Debug> {
    /// CPU RMSprop optimizer for fallback
    cpu_optimizer: RMSprop<A>,

    /// GPU memory manager
    gpu_memory: Option<GpuOptimizerMemory<A>>,

    /// GPU kernel handle
    kernel_handle: Option<Arc<GpuKernelHandle>>,

    /// Whether optimizer is on GPU
    on_gpu: bool,

    /// Step count
    step_count: usize,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> RMSpropGpu<A> {
    /// Create a new GPU-accelerated RMSprop optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            cpu_optimizer: RMSprop::new(learning_rate),
            gpu_memory: None,
            kernel_handle: None,
            on_gpu: false,
            step_count: 0,
        }
    }

    /// Create with full configuration
    pub fn new_with_config(
        learning_rate: A,
        alpha: A,
        epsilon: A,
        weight_decay: A,
        momentum: A,
        centered: bool,
    ) -> Self {
        Self {
            cpu_optimizer: RMSprop::new_with_config(
                learning_rate,
                alpha,
                epsilon,
                weight_decay,
                momentum,
                centered,
            ),
            gpu_memory: None,
            kernel_handle: None,
            on_gpu: false,
            step_count: 0,
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

        // Load RMSprop kernel
        #[cfg(feature = "gpu")]
        {
            let kernel_name = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                if self.cpu_optimizer.centered {
                    "rmsprop_centered_update_f32"
                } else {
                    "rmsprop_update_f32"
                }
            } else if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f64>() {
                if self.cpu_optimizer.centered {
                    "rmsprop_centered_update_f64"
                } else {
                    "rmsprop_update_f64"
                }
            } else {
                return Err(GpuOptimError::UnsupportedOperation(
                    "Unsupported data type for GPU RMSprop".to_string(),
                ));
            };

            let kernel_handle = gpu_memory.context().get_kernel(kernel_name)?;
            self.kernel_handle = Some(Arc::new(kernel_handle));
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

        // Set kernel parameters
        #[cfg(feature = "gpu")]
        {
            kernel.set_buffer("params", gpu_memory.params_gpu.as_ref().unwrap());
            kernel.set_buffer("grads", gpu_memory.grads_gpu.as_ref().unwrap());
            kernel.set_buffer("v", gpu_memory.v_gpu.as_ref().unwrap()); // Mean square accumulator

            if self.cpu_optimizer.momentum != A::zero() {
                kernel.set_buffer("m", gpu_memory.m_gpu.as_ref().unwrap()); // Momentum buffer
            }

            // Convert Float values to concrete types for kernel
            if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
                kernel.set_f32(
                    "lr",
                    self.cpu_optimizer.get_learning_rate().to_f32().unwrap(),
                );
                kernel.set_f32("alpha", self.cpu_optimizer.alpha.to_f32().unwrap());
                kernel.set_f32("eps", self.cpu_optimizer.epsilon.to_f32().unwrap());
                kernel.set_f32(
                    "weight_decay",
                    self.cpu_optimizer.weight_decay.to_f32().unwrap(),
                );
                kernel.set_f32("momentum", self.cpu_optimizer.momentum.to_f32().unwrap());
            } else {
                kernel.set_f64(
                    "lr",
                    self.cpu_optimizer.get_learning_rate().to_f64().unwrap(),
                );
                kernel.set_f64("alpha", self.cpu_optimizer.alpha.to_f64().unwrap());
                kernel.set_f64("eps", self.cpu_optimizer.epsilon.to_f64().unwrap());
                kernel.set_f64(
                    "weight_decay",
                    self.cpu_optimizer.weight_decay.to_f64().unwrap(),
                );
                kernel.set_f64("momentum", self.cpu_optimizer.momentum.to_f64().unwrap());
            }

            kernel.set_i32("n", params.len() as i32);

            // Calculate grid and block dimensions
            let (grid_size, block_size) =
                crate::gpu::utils::calculate_block_size(params.len(), 256);

            // Launch kernel
            kernel.dispatch([grid_size as u32, 1, 1]);
        }

        // Copy results back to CPU
        gpu_memory.copy_params_from_gpu(params)?;

        Ok(())
    }
}

// Implement standard Optimizer trait with CPU fallback
impl<A, D> Optimizer<A, D> for RMSpropGpu<A>
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray_ext::Array1;

    #[test]
    fn test_rmsprop_gpu_creation() {
        let optimizer = RMSpropGpu::<f32>::new(0.001);
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert!(!optimizer.on_gpu);
    }

    #[test]
    fn test_rmsprop_gpu_cpu_fallback() {
        let mut optimizer = RMSpropGpu::new(0.001);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Should use CPU implementation when not on GPU
        let result = optimizer.step(&params, &grads);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.len(), 3);
    }

    #[test]
    fn test_gpu_initialization() {
        let mut optimizer = RMSpropGpu::<f32>::new(0.001);
        let config = GpuOptimizerConfig {
            backend: scirs2,
            core: gpu::GpuBackend::Cpu,
            ..Default::default()
        };

        let result = optimizer.initialize_gpu(1000, config);
        assert!(result.is_ok());
    }
}
