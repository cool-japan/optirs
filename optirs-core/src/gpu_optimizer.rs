//! GPU optimizer scaffolding
//!
//! # Status: no GPU backend is wired up yet
//!
//! This module defines the API surface for GPU-accelerated optimization, but **no
//! device backend is currently implemented**. [`GpuUtils::detect_backends`] returns
//! an empty list, [`GpuUtils::device_count`] returns `0`, and consequently
//! [`GpuOptimizer::is_gpu_available`] reports `false` and every optimization step
//! executes on the CPU through the wrapped base optimizer.
//!
//! The wrapper is still useful today: it lets calling code be written once against
//! the GPU-aware API and keep working unchanged when a backend lands. It will not,
//! however, make anything faster right now — treat it as a compatibility shim, not
//! as an accelerator.
//!
//! # What a real backend must provide
//!
//! When SciRS2's GPU abstractions become available, the integration points are:
//! - `scirs2_core::gpu::GpuContext` for GPU context management
//! - `scirs2_core::gpu::GpuBuffer` for GPU memory allocation
//! - `scirs2_core::gpu::GpuKernel` for GPU kernel execution
//! - `scirs2_core::tensor_cores` for mixed-precision optimization
//! - `scirs2_core::array_protocol::GPUArray` for the GPU array interface
//!
//! Wiring those up means implementing [`GpuUtils::detect_backends`],
//! [`GpuUtils::device_count`] and `GpuOptimizer::step_gpu`; the availability
//! reporting below then becomes truthful automatically.

use scirs2_core::ndarray::{Array1, ArrayView1, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// GPU optimizer configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Enable tensor core acceleration
    pub use_tensor_cores: bool,
    /// Enable mixed-precision training (FP16/FP32)
    pub use_mixed_precision: bool,
    /// Preferred GPU backend (auto-detected if None)
    pub preferred_backend: Option<String>,
    /// Maximum GPU memory usage (bytes)
    pub max_gpu_memory: Option<usize>,
    /// Enable GPU memory tracking
    pub track_memory: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            use_tensor_cores: true,
            use_mixed_precision: false,
            preferred_backend: None,
            max_gpu_memory: None,
            track_memory: true,
        }
    }
}

/// GPU-accelerated optimizer wrapper
///
/// Wraps any CPU optimizer to provide GPU acceleration using SciRS2's GPU abstractions.
/// Automatically handles host-device data transfer and GPU memory management.
///
/// # Examples
///
/// ```
/// use optirs_core::optimizers::SGD;
/// use optirs_core::gpu_optimizer::{GpuOptimizer, GpuConfig};
/// use scirs2_core::ndarray::Array1;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let optimizer = SGD::new(0.01);
/// let config = GpuConfig::default();
///
/// // Create GPU-accelerated optimizer
/// let mut gpu_opt = GpuOptimizer::new(optimizer, config)?;
///
/// // Use like a normal optimizer - GPU acceleration is automatic
/// let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);
///
/// let updated = gpu_opt.step(&params, &grads)?;
/// # Ok(())
/// # }
/// ```
pub struct GpuOptimizer<O, A>
where
    O: Optimizer<A, scirs2_core::ndarray::Ix1>,
    A: Float + ScalarOperand + Debug,
{
    /// Base CPU optimizer
    base_optimizer: O,
    /// GPU configuration
    config: GpuConfig,
    /// GPU context (lazily initialized)
    gpu_context: Option<GpuContextWrapper>,
    /// Phantom data for type parameter
    _phantom: PhantomData<A>,
}

/// Wrapper for GPU context to handle initialization
struct GpuContextWrapper {
    /// Whether a GPU backend was actually found and initialized
    available: bool,
    /// GPU backend name (CUDA, Metal, OpenCL, WebGPU); `None` when unavailable
    backend: Option<String>,
}

impl<O, A> GpuOptimizer<O, A>
where
    O: Optimizer<A, scirs2_core::ndarray::Ix1> + Clone,
    A: Float + ScalarOperand + Debug,
{
    /// Creates a new GPU-accelerated optimizer
    ///
    /// # Arguments
    ///
    /// * `base_optimizer` - The CPU optimizer to accelerate
    /// * `config` - GPU configuration settings
    ///
    /// # Returns
    ///
    /// A GPU-accelerated optimizer or an error if GPU initialization fails
    pub fn new(base_optimizer: O, config: GpuConfig) -> Result<Self> {
        // Initialize GPU context
        let gpu_context = Self::initialize_gpu(&config)?;

        Ok(Self {
            base_optimizer,
            config,
            gpu_context: Some(gpu_context),
            _phantom: PhantomData,
        })
    }

    /// Creates a new GPU optimizer with default configuration
    pub fn with_default_config(base_optimizer: O) -> Result<Self> {
        Self::new(base_optimizer, GpuConfig::default())
    }

    /// Initialize the GPU context, if a backend is actually present
    ///
    /// No backend is implemented yet, so this reports "unavailable" and the optimizer
    /// transparently runs on the CPU. Construction still succeeds: an unavailable GPU
    /// is a supported configuration, not an error.
    fn initialize_gpu(config: &GpuConfig) -> Result<GpuContextWrapper> {
        let available_backends = GpuUtils::detect_backends();

        let backend = match &config.preferred_backend {
            // A specific backend was requested: honour it only if it is present.
            Some(preferred) => available_backends
                .iter()
                .find(|candidate| candidate.eq_ignore_ascii_case(preferred))
                .cloned(),
            // Otherwise take whatever the platform offers first.
            None => available_backends.first().cloned(),
        };

        Ok(GpuContextWrapper {
            available: backend.is_some() && GpuUtils::device_count() > 0,
            backend,
        })
    }

    /// Perform GPU-accelerated optimization step
    ///
    /// # Arguments
    ///
    /// * `params` - Current parameters
    /// * `gradients` - Gradients
    ///
    /// # Returns
    ///
    /// Updated parameters after GPU-accelerated optimization
    pub fn step(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>> {
        // Use the device path only when a backend really exists. Today this is never
        // taken; the CPU fallback below is the executed path.
        let gpu_ready = self
            .gpu_context
            .as_ref()
            .map(|ctx| ctx.available)
            .unwrap_or(false);

        if gpu_ready {
            return self.step_gpu(params, gradients);
        }

        self.base_optimizer.step(params, gradients)
    }

    /// Device-side step implementation
    ///
    /// Unimplemented: a real version would transfer `params`/`gradients` into
    /// `scirs2_core::gpu::GpuBuffer`s, run a `scirs2_core::gpu::GpuKernel` (optionally
    /// through `scirs2_core::tensor_cores`), copy the result back and account the
    /// memory. Until then this returns an error rather than pretending the work
    /// happened on a device.
    fn step_gpu(&mut self, _params: &Array1<A>, _gradients: &Array1<A>) -> Result<Array1<A>> {
        Err(crate::error::OptimError::InvalidConfig(
            "GPU execution path is not implemented; no device backend is available".to_string(),
        ))
    }

    /// Transfer array to GPU
    ///
    /// Unimplemented: returns an error because there is no device to copy to.
    /// A real version would use `scirs2_core::gpu::GpuBuffer::from_slice()`.
    pub fn to_gpu(&self, _data: &ArrayView1<A>) -> Result<()> {
        Err(crate::error::OptimError::InvalidConfig(
            "GPU transfer is not implemented; no device backend is available".to_string(),
        ))
    }

    /// Transfer array from GPU
    ///
    /// Unimplemented: returns an error because there is no device to copy from.
    /// A real version would use `scirs2_core::gpu::GpuBuffer::to_host()`.
    pub fn from_gpu(&self) -> Result<Array1<A>> {
        Err(crate::error::OptimError::InvalidConfig(
            "GPU transfer is not implemented; no device backend is available".to_string(),
        ))
    }

    /// Check whether a GPU backend is actually available and initialized
    ///
    /// Currently always `false`: no device backend is implemented, so all work runs
    /// on the CPU. This is guaranteed to agree with
    /// [`GpuUtils::device_count`] / [`GpuUtils::detect_backends`].
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_context
            .as_ref()
            .map(|ctx| ctx.available)
            .unwrap_or(false)
    }

    /// Get the active GPU backend name, or `None` when running on the CPU
    pub fn gpu_backend(&self) -> Option<&str> {
        self.gpu_context
            .as_ref()
            .and_then(|ctx| ctx.backend.as_deref())
    }

    /// Get GPU configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Enable/disable tensor core acceleration
    pub fn set_use_tensor_cores(&mut self, enable: bool) {
        self.config.use_tensor_cores = enable;
    }

    /// Enable/disable mixed-precision training
    pub fn set_use_mixed_precision(&mut self, enable: bool) {
        self.config.use_mixed_precision = enable;
    }

    /// Get estimated GPU memory usage for given parameter count
    pub fn estimate_gpu_memory(
        num_params: usize,
        dtype_size: usize,
        optimizer_states: usize,
    ) -> usize {
        // Parameters + gradients + optimizer states
        num_params * dtype_size * (2 + optimizer_states)
    }
}

/// GPU memory statistics
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    /// Total GPU memory (bytes)
    pub total: usize,
    /// Used GPU memory (bytes)
    pub used: usize,
    /// Free GPU memory (bytes)
    pub free: usize,
    /// Memory used by optimizer (bytes)
    pub optimizer_usage: usize,
}

impl GpuMemoryStats {
    /// Create memory stats
    pub fn new(total: usize, used: usize) -> Self {
        Self {
            total,
            used,
            free: total.saturating_sub(used),
            optimizer_usage: 0,
        }
    }

    /// Get memory utilization percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }
}

/// GPU optimizer utilities
pub struct GpuUtils;

impl GpuUtils {
    /// Detect available GPU backends
    ///
    /// Returns the list of usable backends (CUDA, Metal, OpenCL, WebGPU).
    ///
    /// No backend is implemented yet, so this is always empty. It must stay empty
    /// until a backend can genuinely execute kernels — reporting a phantom "auto"
    /// backend would make [`GpuOptimizer::is_gpu_available`] lie.
    pub fn detect_backends() -> Vec<String> {
        // Note: Full implementation would use scirs2_core::gpu::detect_backends()
        Vec::new()
    }

    /// Check if tensor cores are available
    pub fn has_tensor_cores() -> bool {
        // Note: Full implementation would use scirs2_core::tensor_cores::is_available()
        false
    }

    /// Get GPU device count
    pub fn device_count() -> usize {
        // Note: Full implementation would use scirs2_core::gpu::device_count()
        0
    }

    /// Get GPU memory stats for device
    pub fn memory_stats(device_id: usize) -> Result<GpuMemoryStats> {
        // Note: Full implementation would use scirs2_core::gpu::get_memory_info()
        let _ = device_id;
        Ok(GpuMemoryStats::new(0, 0))
    }

    /// Synchronize GPU operations
    pub fn synchronize() -> Result<()> {
        // Note: Full implementation would use scirs2_core::gpu::synchronize()
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert!(config.use_tensor_cores);
        assert!(!config.use_mixed_precision);
        assert!(config.track_memory);
    }

    #[test]
    fn test_gpu_optimizer_creation() {
        let optimizer = SGD::new(0.01);
        let config = GpuConfig::default();
        let gpu_opt = GpuOptimizer::new(optimizer, config);
        assert!(gpu_opt.is_ok());
    }

    #[test]
    fn test_gpu_optimizer_with_default_config() {
        let optimizer = SGD::new(0.01);
        let gpu_opt = GpuOptimizer::with_default_config(optimizer);
        assert!(gpu_opt.is_ok());
    }

    #[test]
    fn test_gpu_optimizer_step() {
        let optimizer = SGD::new(0.01);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer)
            .expect("GpuOptimizer::with_default_config succeeds in test_gpu_optimizer_step");

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let result = gpu_opt.step(&params, &grads);
        assert!(result.is_ok());
    }

    /// Regression test: availability reporting must match the actual device inventory.
    ///
    /// `is_gpu_available()` used to return `true` unconditionally while
    /// `device_count()` returned `0` and every step ran on the CPU.
    #[test]
    fn test_gpu_availability_matches_device_inventory() {
        let optimizer = SGD::new(0.01);
        let gpu_opt = GpuOptimizer::with_default_config(optimizer).expect("GpuOptimizer::with_default_config succeeds in test_gpu_availability_matches_device_inventory");

        let devices = GpuUtils::device_count();
        let backends = GpuUtils::detect_backends();

        assert_eq!(
            gpu_opt.is_gpu_available(),
            devices > 0 && !backends.is_empty(),
            "availability must agree with the detected device inventory"
        );
        assert_eq!(backends.is_empty(), devices == 0);

        // No backend is implemented yet, so the honest answer is "not available".
        assert!(!gpu_opt.is_gpu_available());
    }

    #[test]
    fn test_gpu_backend() {
        let optimizer = SGD::new(0.01);
        let gpu_opt = GpuOptimizer::with_default_config(optimizer)
            .expect("GpuOptimizer::with_default_config succeeds in test_gpu_backend");

        // No device => no backend name to report.
        assert_eq!(gpu_opt.gpu_backend().is_some(), gpu_opt.is_gpu_available());
        assert!(gpu_opt.gpu_backend().is_none());
    }

    /// Requesting a backend that does not exist must not fabricate one.
    #[test]
    fn test_gpu_preferred_backend_is_not_fabricated() {
        let optimizer = SGD::new(0.01);
        let config = GpuConfig {
            preferred_backend: Some("cuda".to_string()),
            ..GpuConfig::default()
        };
        let gpu_opt = GpuOptimizer::new(optimizer, config).expect("construction must succeed");

        assert!(!gpu_opt.is_gpu_available());
        assert!(gpu_opt.gpu_backend().is_none());
    }

    /// The device transfer helpers must report that they are unimplemented.
    #[test]
    fn test_gpu_transfers_report_unavailable() {
        let optimizer = SGD::new(0.01);
        let gpu_opt = GpuOptimizer::with_default_config(optimizer).expect(
            "GpuOptimizer::with_default_config succeeds in test_gpu_transfers_report_unavailable",
        );

        let data = Array1::from_vec(vec![1.0f64, 2.0]);
        assert!(gpu_opt.to_gpu(&data.view()).is_err());
        assert!(gpu_opt.from_gpu().is_err());
    }

    #[test]
    fn test_gpu_config_mutations() {
        let optimizer = SGD::new(0.01);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer)
            .expect("GpuOptimizer::with_default_config succeeds in test_gpu_config_mutations");

        gpu_opt.set_use_tensor_cores(false);
        assert!(!gpu_opt.config().use_tensor_cores);

        gpu_opt.set_use_mixed_precision(true);
        assert!(gpu_opt.config().use_mixed_precision);
    }

    #[test]
    fn test_estimate_gpu_memory() {
        // SGD: params + gradients + velocity = 3 states
        let mem = GpuOptimizer::<SGD<f32>, f32>::estimate_gpu_memory(1_000_000, 4, 1);
        assert_eq!(mem, 12_000_000); // 12 MB

        // Adam: params + gradients + m + v = 4 states
        let mem = GpuOptimizer::<SGD<f32>, f32>::estimate_gpu_memory(1_000_000, 4, 2);
        assert_eq!(mem, 16_000_000); // 16 MB
    }

    #[test]
    fn test_gpu_memory_stats() {
        let stats = GpuMemoryStats::new(1_000_000_000, 500_000_000);
        assert_eq!(stats.total, 1_000_000_000);
        assert_eq!(stats.used, 500_000_000);
        assert_eq!(stats.free, 500_000_000);
        assert_eq!(stats.utilization_percent(), 50.0);
    }

    #[test]
    fn test_gpu_utils_detect_backends() {
        // No backend is implemented, so nothing may be advertised.
        let backends = GpuUtils::detect_backends();
        assert!(backends.is_empty());
        assert_eq!(GpuUtils::device_count(), 0);
    }

    #[test]
    fn test_gpu_utils_synchronize() {
        let result = GpuUtils::synchronize();
        assert!(result.is_ok());
    }
}
