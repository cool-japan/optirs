//! GPU-resident optimizer steps executed through `scirs2_core::gpu`.
//!
//! Every optimizer here implements [`crate::GpuOptimizer`] for `f32` and runs a
//! real compute shader: parameters and gradients are uploaded to device
//! buffers, a WGSL kernel from [`crate::shaders::wgsl`] is dispatched, and the updated
//! parameters are read back. The per-parameter optimizer state (Adam's `m`/`v`,
//! SGD's momentum buffer, ...) stays resident in device memory between steps;
//! [`crate::GpuOptimizer::move_to_cpu`] genuinely downloads it and
//! [`crate::GpuOptimizer::move_to_gpu`] genuinely uploads it again.
//!
//! # Backend support
//!
//! Only the WebGPU backend (`wgpu` feature → Vulkan / Metal / DX12) has a
//! complete compute path in scirs2-core 0.6.x. Constructing an optimizer with
//! any other backend returns [`GpuOptimError::UnsupportedOperation`] rather
//! than silently computing nothing — in particular the Metal backend registers
//! *empty* kernel sources for the optimizer kernels, and CUDA was removed from
//! scirs2-core in 0.6.x.
//!
//! # Precision
//!
//! WGSL compute shaders are `f32`. The trait is therefore implemented for
//! `f32` only; there is no `f64` GPU path and none is faked.

use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuKernelHandle};
use scirs2_core::ndarray::{Array, Dimension};

use crate::shaders::{OptimizerKernel, WORKGROUP_SIZE};
use crate::{GpuOptimError, GpuOptimizer};

/// Configuration shared by every GPU optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuOptimizerConfig {
    /// Backend to use, or `None` to probe the supported backends in order.
    ///
    /// Auto-selection tries [`GpuBackend::Wgpu`] first (portable: Vulkan /
    /// Metal / DX12) and falls back to [`GpuBackend::Metal`] on macOS.
    pub backend: Option<GpuBackend>,
}

impl GpuOptimizerConfig {
    /// Pin the optimizer to one backend.
    pub fn with_backend(backend: GpuBackend) -> Self {
        Self {
            backend: Some(backend),
        }
    }
}

/// Backends that can execute the optimizer kernels, in preference order.
///
/// `Cuda` and `Rocm` are absent because `scirs2-core` 0.6.x has no working
/// compute path for them; `OpenCL` is absent because this crate ships no
/// OpenCL C sources. Asking for any of those is an explicit error rather than
/// a silent no-op.
pub const SUPPORTED_BACKENDS: [GpuBackend; 2] = [GpuBackend::Wgpu, GpuBackend::Metal];

/// Round `n` up to a whole number of workgroups of [`WORKGROUP_SIZE`].
fn workgroup_count(n: usize) -> Result<u32, GpuOptimError> {
    let groups = n.div_ceil(WORKGROUP_SIZE);
    u32::try_from(groups).map_err(|_| {
        GpuOptimError::UnsupportedOperation(format!(
            "{n} elements need {groups} workgroups, which exceeds the u32 dispatch limit"
        ))
    })
}

/// Encode a `usize` element count into an `f32` slot bit-for-bit.
///
/// The kernels recover it with `bitcast<u32>` / `as_type<uint>`, so counts
/// above 2^24 stay exact (a plain `n as f32` would not).
fn encode_u32(value: usize) -> Result<f32, GpuOptimError> {
    let raw = u32::try_from(value).map_err(|_| {
        GpuOptimError::UnsupportedOperation(format!("{value} does not fit in a u32 kernel operand"))
    })?;
    Ok(f32::from_bits(raw))
}

/// Owns the GPU context plus the pipeline it currently holds.
///
/// `GpuCompiler::compile` on the WebGPU backend of scirs2-core 0.6.5 registers
/// every compiled shader under one fixed internal name per context, so a
/// context can only have one live pipeline. This cache makes that explicit: a
/// kernel is recompiled only when a *different* shader is requested on the same
/// context, and the handle is always resolved immediately before the dispatch
/// that uses it.
struct KernelCache {
    context: GpuContext,
    backend: GpuBackend,
    installed: Option<&'static str>,
    handle: Option<GpuKernelHandle>,
}

impl KernelCache {
    fn new(requested: Option<GpuBackend>) -> Result<Self, GpuOptimError> {
        match requested {
            Some(backend) => {
                if !SUPPORTED_BACKENDS.contains(&backend) {
                    return Err(GpuOptimError::UnsupportedOperation(format!(
                        "backend {backend} cannot run optirs-gpu optimizer kernels; \
                         supported backends are {SUPPORTED_BACKENDS:?}"
                    )));
                }
                let context = GpuContext::new(backend)?;
                Ok(Self {
                    context,
                    backend,
                    installed: None,
                    handle: None,
                })
            }
            None => {
                let mut reasons = Vec::new();
                for backend in SUPPORTED_BACKENDS {
                    match GpuContext::new(backend) {
                        Ok(context) => {
                            return Ok(Self {
                                context,
                                backend,
                                installed: None,
                                handle: None,
                            })
                        }
                        Err(e) => reasons.push(format!("{backend}: {e}")),
                    }
                }
                Err(GpuOptimError::UnsupportedOperation(format!(
                    "no GPU backend available for optimizer kernels ({})",
                    reasons.join("; ")
                )))
            }
        }
    }

    fn context(&self) -> &GpuContext {
        &self.context
    }

    fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Return a handle for `kernel`, compiling it if it is not the pipeline
    /// currently installed on this context.
    fn kernel(&mut self, kernel: OptimizerKernel) -> Result<&GpuKernelHandle, GpuOptimError> {
        let key = kernel.cache_key(self.backend);
        if self.installed != Some(key) || self.handle.is_none() {
            let source = kernel.source_for(self.backend).ok_or_else(|| {
                GpuOptimError::UnsupportedOperation(format!(
                    "no {} shader source for backend {}",
                    kernel.id(),
                    self.backend
                ))
            })?;
            let handle = self.context.execute(|compiler| compiler.compile(source))?;
            self.handle = Some(handle);
            self.installed = Some(key);
        }
        self.handle.as_ref().ok_or_else(|| {
            GpuOptimError::InvalidState("kernel compilation produced no handle".into())
        })
    }
}

/// Device-side per-parameter state buffers plus their host mirror.
///
/// The host mirror is authoritative while the optimizer is on the CPU; the
/// device buffers are authoritative while it is on the GPU. Exactly one of the
/// two is live at a time, and transitions copy the data for real.
struct StateBuffers {
    /// Number of independent state vectors (2 for Adam: `m` and `v`).
    slots: usize,
    /// Element count of each state vector.
    len: usize,
    host: Vec<Vec<f32>>,
    device: Vec<GpuBuffer<f32>>,
}

impl StateBuffers {
    fn new(slots: usize, len: usize) -> Self {
        Self {
            slots,
            len,
            host: vec![vec![0.0f32; len]; slots],
            device: Vec::new(),
        }
    }

    fn is_resident(&self) -> bool {
        self.device.len() == self.slots
    }

    /// Upload the host mirror into fresh device buffers.
    fn upload(&mut self, context: &GpuContext) -> Result<(), GpuOptimError> {
        if self.len == 0 {
            return Err(GpuOptimError::InvalidState(
                "cannot allocate zero-length optimizer state".into(),
            ));
        }
        let mut device = Vec::with_capacity(self.slots);
        for slot in &self.host {
            let buffer = context.create_buffer::<f32>(self.len);
            buffer.copy_from_host(slot)?;
            device.push(buffer);
        }
        self.device = device;
        Ok(())
    }

    /// Download the device buffers back into the host mirror and release them.
    fn download(&mut self) -> Result<(), GpuOptimError> {
        if !self.is_resident() {
            return Ok(());
        }
        for (slot, buffer) in self.host.iter_mut().zip(self.device.iter()) {
            buffer.copy_to_host(slot)?;
        }
        self.device.clear();
        Ok(())
    }

    fn device_slot(&self, index: usize) -> Result<&GpuBuffer<f32>, GpuOptimError> {
        self.device.get(index).ok_or(GpuOptimError::NotInitialized)
    }

    /// Reset both mirrors to zero for a new parameter length.
    fn resize(&mut self, len: usize) {
        self.len = len;
        self.host = vec![vec![0.0f32; len]; self.slots];
        self.device.clear();
    }
}

/// Shared plumbing for the concrete optimizers below.
struct GpuStepEngine {
    cache: KernelCache,
    state: StateBuffers,
    on_gpu: bool,
    step_count: u64,
}

impl GpuStepEngine {
    fn new(config: GpuOptimizerConfig, slots: usize) -> Result<Self, GpuOptimError> {
        Ok(Self {
            cache: KernelCache::new(config.backend)?,
            state: StateBuffers::new(slots, 0),
            on_gpu: false,
            step_count: 0,
        })
    }

    fn backend(&self) -> GpuBackend {
        self.cache.backend()
    }

    fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        if self.on_gpu {
            return Ok(());
        }
        if self.state.len > 0 {
            let context = &self.cache.context;
            self.state.upload(context)?;
        }
        self.on_gpu = true;
        Ok(())
    }

    fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        if !self.on_gpu {
            return Ok(());
        }
        self.state.download()?;
        self.on_gpu = false;
        Ok(())
    }

    /// Make sure the device-side state matches `len` elements and is resident.
    fn prepare(&mut self, len: usize) -> Result<(), GpuOptimError> {
        if !self.on_gpu {
            return Err(GpuOptimError::InvalidState(
                "optimizer is on the CPU; call move_to_gpu() before step_gpu()".into(),
            ));
        }
        if len == 0 {
            return Err(GpuOptimError::InvalidState(
                "cannot run a GPU step on an empty parameter array".into(),
            ));
        }
        // scirs2-core's Metal buffers silently clamp allocations at 1 GiB and
        // then `assert!` in `copy_from_host` when the requested copy exceeds the
        // clamped size. Reject oversized parameter vectors here with a real
        // error instead of tripping that assert.
        const MAX_BUFFER_BYTES: usize = 1024 * 1024 * 1024;
        let bytes = len.saturating_mul(std::mem::size_of::<f32>());
        if bytes > MAX_BUFFER_BYTES {
            return Err(GpuOptimError::UnsupportedOperation(format!(
                "{len} f32 parameters need {bytes} bytes, above the {MAX_BUFFER_BYTES}-byte \
                 per-buffer limit of the GPU backends this crate supports"
            )));
        }
        if self.state.len != len {
            self.state.resize(len);
            self.step_count = 0;
        }
        if !self.state.is_resident() {
            let context = &self.cache.context;
            self.state.upload(context)?;
        }
        Ok(())
    }
}

/// Flatten an array of any layout into a contiguous host vector.
fn to_host_vec<D: Dimension>(array: &Array<f32, D>) -> Vec<f32> {
    match array.as_slice() {
        Some(slice) => slice.to_vec(),
        None => array.iter().copied().collect(),
    }
}

/// Write a contiguous host vector back into an array of any layout.
fn from_host_vec<D: Dimension>(
    array: &mut Array<f32, D>,
    values: &[f32],
) -> Result<(), GpuOptimError> {
    if values.len() != array.len() {
        return Err(GpuOptimError::DimensionMismatch {
            expected: array.shape().to_vec(),
            actual: vec![values.len()],
        });
    }
    for (dst, src) in array.iter_mut().zip(values.iter()) {
        *dst = *src;
    }
    Ok(())
}

/// Validate that parameters and gradients agree in shape.
fn check_shapes<D: Dimension>(
    params: &Array<f32, D>,
    gradients: &Array<f32, D>,
) -> Result<(), GpuOptimError> {
    if params.shape() != gradients.shape() {
        return Err(GpuOptimError::DimensionMismatch {
            expected: params.shape().to_vec(),
            actual: gradients.shape().to_vec(),
        });
    }
    Ok(())
}

// ─── Adam / AdamW ───────────────────────────────────────────────────────────

/// Hyper-parameters shared by [`GpuAdam`] and [`GpuAdamW`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamParams {
    /// Step size.
    pub learning_rate: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Denominator stabiliser.
    pub epsilon: f32,
    /// Weight decay (coupled L2 for Adam, decoupled for AdamW).
    pub weight_decay: f32,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl AdamParams {
    /// Build a validated parameter set.
    ///
    /// Returns [`GpuOptimError::InvalidState`] instead of panicking when a
    /// value is outside its admissible range.
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<Self, GpuOptimError> {
        let params = Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        };
        params.validate()?;
        Ok(params)
    }

    fn validate(&self) -> Result<(), GpuOptimError> {
        let invalid = |what: &str| GpuOptimError::InvalidState(format!("invalid Adam {what}"));
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(invalid("learning rate (must be finite and > 0)"));
        }
        if !(self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1)) {
            return Err(invalid("beta1 (must be in [0, 1))"));
        }
        if !(self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2)) {
            return Err(invalid("beta2 (must be in [0, 1))"));
        }
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(invalid("epsilon (must be finite and > 0)"));
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            return Err(invalid("weight decay (must be finite and >= 0)"));
        }
        Ok(())
    }

    /// Bias-correction denominators for a 1-based timestep.
    fn bias_corrections(&self, step: u64) -> (f32, f32) {
        let exp = step.min(i32::MAX as u64) as i32;
        (1.0 - self.beta1.powi(exp), 1.0 - self.beta2.powi(exp))
    }
}

macro_rules! adam_family {
    ($name:ident, $kernel:expr, $doc:literal) => {
        #[doc = $doc]
        pub struct $name {
            engine: GpuStepEngine,
            params: AdamParams,
        }

        impl $name {
            /// Create the optimizer with the default WebGPU backend.
            pub fn new(params: AdamParams) -> Result<Self, GpuOptimError> {
                Self::with_config(params, GpuOptimizerConfig::default())
            }

            /// Create the optimizer on an explicit backend.
            pub fn with_config(
                params: AdamParams,
                config: GpuOptimizerConfig,
            ) -> Result<Self, GpuOptimError> {
                params.validate()?;
                Ok(Self {
                    engine: GpuStepEngine::new(config, 2)?,
                    params,
                })
            }

            /// Backend actually in use.
            pub fn backend(&self) -> GpuBackend {
                self.engine.backend()
            }

            /// Upload the optimizer state to device memory.
            ///
            /// Inherent alias for [`GpuOptimizer::move_to_gpu`] so callers do
            /// not need a turbofish to pin the unused dimension parameter.
            pub fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
                self.engine.move_to_gpu()
            }

            /// Download the optimizer state back to host memory.
            ///
            /// Inherent alias for [`GpuOptimizer::move_to_cpu`]; see
            /// [`Self::move_to_gpu`].
            pub fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
                self.engine.move_to_cpu()
            }

            /// Deprecated alias for [`Self::move_to_gpu`].
            #[deprecated(since = "0.3.2", note = "renamed to `move_to_gpu`")]
            pub fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
                self.move_to_gpu()
            }

            /// Deprecated alias for [`Self::move_to_cpu`].
            #[deprecated(since = "0.3.2", note = "renamed to `move_to_cpu`")]
            pub fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
                self.move_to_cpu()
            }

            /// Whether a real device (not the CPU fallback) is backing this optimizer.
            pub fn is_gpu_available(&self) -> bool {
                self.engine.backend() != GpuBackend::Cpu
            }

            /// Number of updates applied so far (1-based bias correction input).
            pub fn step_count(&self) -> u64 {
                self.engine.step_count
            }

            /// Current hyper-parameters.
            pub fn params(&self) -> &AdamParams {
                &self.params
            }

            /// Replace the hyper-parameters, validating them first.
            pub fn set_params(&mut self, params: AdamParams) -> Result<(), GpuOptimError> {
                params.validate()?;
                self.params = params;
                Ok(())
            }
        }

        impl<D: Dimension> GpuOptimizer<f32, D> for $name {
            fn is_gpu_available(&self) -> bool {
                self.engine.backend() != GpuBackend::Cpu
            }

            fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
                self.engine.move_to_gpu()
            }

            fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
                self.engine.move_to_cpu()
            }

            fn step_gpu(
                &mut self,
                params: &mut Array<f32, D>,
                gradients: &Array<f32, D>,
            ) -> Result<(), GpuOptimError> {
                check_shapes(params, gradients)?;
                let n = params.len();
                self.engine.prepare(n)?;
                self.engine.step_count = self.engine.step_count.saturating_add(1);

                let (bc1, bc2) = self.params.bias_corrections(self.engine.step_count);
                let hyper = [
                    self.params.learning_rate,
                    self.params.beta1,
                    self.params.beta2,
                    self.params.epsilon,
                    self.params.weight_decay,
                    bc1,
                    bc2,
                    encode_u32(n)?,
                ];

                let host_params = to_host_vec(params);
                let host_grads = to_host_vec(gradients);
                let groups = workgroup_count(n)?;

                let updated = {
                    let context = self.engine.cache.context();
                    let params_buf = context.create_buffer::<f32>(n);
                    params_buf.copy_from_host(&host_params)?;
                    let grads_buf = context.create_buffer::<f32>(n);
                    grads_buf.copy_from_host(&host_grads)?;
                    let hyper_buf = context.create_buffer::<f32>(hyper.len());
                    hyper_buf.copy_from_host(&hyper)?;

                    let m_buf = self.engine.state.device_slot(0)?.clone();
                    let v_buf = self.engine.state.device_slot(1)?.clone();

                    let kernel = self.engine.cache.kernel($kernel)?;
                    kernel.set_buffer("x", &params_buf);
                    kernel.set_buffer("y", &grads_buf);
                    kernel.set_buffer("a", &m_buf);
                    kernel.set_buffer("b", &v_buf);
                    kernel.set_buffer("result", &hyper_buf);
                    kernel.dispatch([groups, 1, 1]);

                    let mut out = vec![0.0f32; n];
                    params_buf.copy_to_host(&mut out)?;
                    out
                };

                from_host_vec(params, &updated)
            }
        }
    };
}

adam_family!(
    GpuAdam,
    OptimizerKernel::Adam,
    "GPU Adam with coupled L2 weight decay, numerically matching \
     `optirs_core::optimizers::Adam`."
);
adam_family!(
    GpuAdamW,
    OptimizerKernel::AdamW,
    "GPU AdamW with *decoupled* weight decay: the decay term is applied to the \
     parameter and never enters the moment estimates."
);

// ─── SGD ────────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`GpuSgd`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdParams {
    /// Step size.
    pub learning_rate: f32,
    /// Momentum factor (0 disables the momentum buffer).
    pub momentum: f32,
    /// Dampening applied to the gradient inside the momentum buffer.
    pub dampening: f32,
    /// Coupled L2 weight decay.
    pub weight_decay: f32,
    /// Use Nesterov accelerated gradient.
    pub nesterov: bool,
}

impl Default for SgdParams {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

impl SgdParams {
    fn validate(&self) -> Result<(), GpuOptimError> {
        let invalid = |what: &str| GpuOptimError::InvalidState(format!("invalid SGD {what}"));
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(invalid("learning rate (must be finite and > 0)"));
        }
        if !(self.momentum.is_finite() && self.momentum >= 0.0) {
            return Err(invalid("momentum (must be finite and >= 0)"));
        }
        if !(self.dampening.is_finite() && (0.0..1.0).contains(&self.dampening)) {
            return Err(invalid("dampening (must be in [0, 1))"));
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            return Err(invalid("weight decay (must be finite and >= 0)"));
        }
        if self.nesterov && (self.momentum <= 0.0 || self.dampening != 0.0) {
            return Err(invalid(
                "Nesterov mode (requires momentum > 0 and dampening == 0)",
            ));
        }
        Ok(())
    }
}

/// GPU stochastic gradient descent with optional momentum / Nesterov.
pub struct GpuSgd {
    engine: GpuStepEngine,
    params: SgdParams,
}

impl GpuSgd {
    /// Create the optimizer with the default WebGPU backend.
    pub fn new(params: SgdParams) -> Result<Self, GpuOptimError> {
        Self::with_config(params, GpuOptimizerConfig::default())
    }

    /// Create the optimizer on an explicit backend.
    pub fn with_config(
        params: SgdParams,
        config: GpuOptimizerConfig,
    ) -> Result<Self, GpuOptimError> {
        params.validate()?;
        Ok(Self {
            engine: GpuStepEngine::new(config, 1)?,
            params,
        })
    }

    /// Backend actually in use.
    pub fn backend(&self) -> GpuBackend {
        self.engine.backend()
    }

    /// Upload the optimizer state to device memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_gpu`] so callers do not
    /// need a turbofish to pin the unused dimension parameter.
    pub fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    /// Download the optimizer state back to host memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_cpu`]; see
    /// [`Self::move_to_gpu`].
    pub fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    /// Deprecated alias for [`Self::move_to_gpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_gpu`")]
    pub fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_gpu()
    }

    /// Deprecated alias for [`Self::move_to_cpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_cpu`")]
    pub fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_cpu()
    }

    /// Whether a real device (not the CPU fallback) is backing this optimizer.
    pub fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    /// Number of updates applied so far.
    pub fn step_count(&self) -> u64 {
        self.engine.step_count
    }
}

impl<D: Dimension> GpuOptimizer<f32, D> for GpuSgd {
    fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    fn step_gpu(
        &mut self,
        params: &mut Array<f32, D>,
        gradients: &Array<f32, D>,
    ) -> Result<(), GpuOptimError> {
        check_shapes(params, gradients)?;
        let n = params.len();
        self.engine.prepare(n)?;
        let first = self.engine.step_count == 0;
        self.engine.step_count = self.engine.step_count.saturating_add(1);

        let hyper = [
            self.params.learning_rate,
            self.params.momentum,
            self.params.dampening,
            self.params.weight_decay,
            if self.params.nesterov { 1.0 } else { 0.0 },
            if first { 1.0 } else { 0.0 },
            encode_u32(n)?,
        ];

        let host_params = to_host_vec(params);
        let host_grads = to_host_vec(gradients);
        let groups = workgroup_count(n)?;

        let updated = {
            let context = self.engine.cache.context();
            let params_buf = context.create_buffer::<f32>(n);
            params_buf.copy_from_host(&host_params)?;
            let grads_buf = context.create_buffer::<f32>(n);
            grads_buf.copy_from_host(&host_grads)?;
            let hyper_buf = context.create_buffer::<f32>(hyper.len());
            hyper_buf.copy_from_host(&hyper)?;
            let buf = self.engine.state.device_slot(0)?.clone();

            let kernel = self.engine.cache.kernel(OptimizerKernel::Sgd)?;
            kernel.set_buffer("x", &params_buf);
            kernel.set_buffer("y", &grads_buf);
            kernel.set_buffer("a", &buf);
            kernel.set_buffer("b", &hyper_buf);
            kernel.dispatch([groups, 1, 1]);

            let mut out = vec![0.0f32; n];
            params_buf.copy_to_host(&mut out)?;
            out
        };

        from_host_vec(params, &updated)
    }
}

// ─── RMSprop ────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`GpuRmsprop`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmspropParams {
    /// Step size.
    pub learning_rate: f32,
    /// Smoothing constant for the squared-gradient average.
    pub alpha: f32,
    /// Denominator stabiliser.
    pub epsilon: f32,
    /// Coupled L2 weight decay.
    pub weight_decay: f32,
    /// Momentum factor (0 disables the momentum buffer).
    pub momentum: f32,
    /// Subtract the squared mean gradient from the variance estimate.
    pub centered: bool,
}

impl Default for RmspropParams {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }
}

impl RmspropParams {
    fn validate(&self) -> Result<(), GpuOptimError> {
        let invalid = |what: &str| GpuOptimError::InvalidState(format!("invalid RMSprop {what}"));
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(invalid("learning rate (must be finite and > 0)"));
        }
        if !(self.alpha.is_finite() && (0.0..1.0).contains(&self.alpha)) {
            return Err(invalid("alpha (must be in [0, 1))"));
        }
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(invalid("epsilon (must be finite and > 0)"));
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            return Err(invalid("weight decay (must be finite and >= 0)"));
        }
        if !(self.momentum.is_finite() && self.momentum >= 0.0) {
            return Err(invalid("momentum (must be finite and >= 0)"));
        }
        Ok(())
    }
}

/// GPU RMSprop with optional centering and momentum.
///
/// The centered variant keeps a dedicated running mean-gradient buffer
/// (`g_avg`), which is what makes `E[g^2] - E[g]^2` a real variance estimate.
pub struct GpuRmsprop {
    engine: GpuStepEngine,
    params: RmspropParams,
}

impl GpuRmsprop {
    /// Create the optimizer with the default WebGPU backend.
    pub fn new(params: RmspropParams) -> Result<Self, GpuOptimError> {
        Self::with_config(params, GpuOptimizerConfig::default())
    }

    /// Create the optimizer on an explicit backend.
    pub fn with_config(
        params: RmspropParams,
        config: GpuOptimizerConfig,
    ) -> Result<Self, GpuOptimError> {
        params.validate()?;
        // Slots: 0 = sq_avg, 1 = g_avg (centered), 2 = momentum buffer.
        Ok(Self {
            engine: GpuStepEngine::new(config, 3)?,
            params,
        })
    }

    /// Backend actually in use.
    pub fn backend(&self) -> GpuBackend {
        self.engine.backend()
    }

    /// Upload the optimizer state to device memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_gpu`] so callers do not
    /// need a turbofish to pin the unused dimension parameter.
    pub fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    /// Download the optimizer state back to host memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_cpu`]; see
    /// [`Self::move_to_gpu`].
    pub fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    /// Deprecated alias for [`Self::move_to_gpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_gpu`")]
    pub fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_gpu()
    }

    /// Deprecated alias for [`Self::move_to_cpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_cpu`")]
    pub fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_cpu()
    }

    /// Whether a real device (not the CPU fallback) is backing this optimizer.
    pub fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    /// Number of updates applied so far.
    pub fn step_count(&self) -> u64 {
        self.engine.step_count
    }
}

impl<D: Dimension> GpuOptimizer<f32, D> for GpuRmsprop {
    fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    fn step_gpu(
        &mut self,
        params: &mut Array<f32, D>,
        gradients: &Array<f32, D>,
    ) -> Result<(), GpuOptimError> {
        check_shapes(params, gradients)?;
        let n = params.len();
        self.engine.prepare(n)?;
        self.engine.step_count = self.engine.step_count.saturating_add(1);

        let hyper = [
            self.params.learning_rate,
            self.params.alpha,
            self.params.epsilon,
            self.params.weight_decay,
            self.params.momentum,
            if self.params.centered { 1.0 } else { 0.0 },
            encode_u32(n)?,
        ];

        let host_params = to_host_vec(params);
        let host_grads = to_host_vec(gradients);
        let groups = workgroup_count(n)?;

        let updated = {
            let context = self.engine.cache.context();
            let params_buf = context.create_buffer::<f32>(n);
            params_buf.copy_from_host(&host_params)?;
            let grads_buf = context.create_buffer::<f32>(n);
            grads_buf.copy_from_host(&host_grads)?;
            let hyper_buf = context.create_buffer::<f32>(hyper.len());
            hyper_buf.copy_from_host(&hyper)?;
            let sq_avg = self.engine.state.device_slot(0)?.clone();
            let g_avg = self.engine.state.device_slot(1)?.clone();
            let buf = self.engine.state.device_slot(2)?.clone();

            let kernel = self.engine.cache.kernel(OptimizerKernel::Rmsprop)?;
            kernel.set_buffer("x", &params_buf);
            kernel.set_buffer("y", &grads_buf);
            kernel.set_buffer("a", &sq_avg);
            kernel.set_buffer("b", &g_avg);
            kernel.set_buffer("result", &buf);
            kernel.set_buffer("output", &hyper_buf);
            kernel.dispatch([groups, 1, 1]);

            let mut out = vec![0.0f32; n];
            params_buf.copy_to_host(&mut out)?;
            out
        };

        from_host_vec(params, &updated)
    }
}

// ─── Adagrad ────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`GpuAdagrad`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdagradParams {
    /// Base step size.
    pub learning_rate: f32,
    /// Learning-rate decay applied per step.
    pub lr_decay: f32,
    /// Denominator stabiliser.
    pub epsilon: f32,
    /// Coupled L2 weight decay.
    pub weight_decay: f32,
}

impl Default for AdagradParams {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            lr_decay: 0.0,
            epsilon: 1e-10,
            weight_decay: 0.0,
        }
    }
}

impl AdagradParams {
    fn validate(&self) -> Result<(), GpuOptimError> {
        let invalid = |what: &str| GpuOptimError::InvalidState(format!("invalid Adagrad {what}"));
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(invalid("learning rate (must be finite and > 0)"));
        }
        if !(self.lr_decay.is_finite() && self.lr_decay >= 0.0) {
            return Err(invalid("lr_decay (must be finite and >= 0)"));
        }
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(invalid("epsilon (must be finite and > 0)"));
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            return Err(invalid("weight decay (must be finite and >= 0)"));
        }
        Ok(())
    }

    /// Effective step size for a 1-based timestep.
    ///
    /// The first update (`step == 1`) uses exactly `learning_rate`; the decay
    /// denominator counts *completed* steps, so it is `step - 1`, not `step`.
    fn effective_lr(&self, step: u64) -> f32 {
        let completed = step.saturating_sub(1) as f32;
        self.learning_rate / (1.0 + completed * self.lr_decay)
    }
}

/// GPU Adagrad with learning-rate decay.
pub struct GpuAdagrad {
    engine: GpuStepEngine,
    params: AdagradParams,
}

impl GpuAdagrad {
    /// Create the optimizer with the default WebGPU backend.
    pub fn new(params: AdagradParams) -> Result<Self, GpuOptimError> {
        Self::with_config(params, GpuOptimizerConfig::default())
    }

    /// Create the optimizer on an explicit backend.
    pub fn with_config(
        params: AdagradParams,
        config: GpuOptimizerConfig,
    ) -> Result<Self, GpuOptimError> {
        params.validate()?;
        Ok(Self {
            engine: GpuStepEngine::new(config, 1)?,
            params,
        })
    }

    /// Backend actually in use.
    pub fn backend(&self) -> GpuBackend {
        self.engine.backend()
    }

    /// Upload the optimizer state to device memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_gpu`] so callers do not
    /// need a turbofish to pin the unused dimension parameter.
    pub fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    /// Download the optimizer state back to host memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_cpu`]; see
    /// [`Self::move_to_gpu`].
    pub fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    /// Deprecated alias for [`Self::move_to_gpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_gpu`")]
    pub fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_gpu()
    }

    /// Deprecated alias for [`Self::move_to_cpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_cpu`")]
    pub fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_cpu()
    }

    /// Whether a real device (not the CPU fallback) is backing this optimizer.
    pub fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    /// Number of updates applied so far.
    pub fn step_count(&self) -> u64 {
        self.engine.step_count
    }

    /// Effective learning rate that the next step will use.
    pub fn next_effective_lr(&self) -> f32 {
        self.params.effective_lr(self.engine.step_count + 1)
    }
}

impl<D: Dimension> GpuOptimizer<f32, D> for GpuAdagrad {
    fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    fn step_gpu(
        &mut self,
        params: &mut Array<f32, D>,
        gradients: &Array<f32, D>,
    ) -> Result<(), GpuOptimError> {
        check_shapes(params, gradients)?;
        let n = params.len();
        self.engine.prepare(n)?;
        self.engine.step_count = self.engine.step_count.saturating_add(1);

        let hyper = [
            self.params.effective_lr(self.engine.step_count),
            self.params.epsilon,
            self.params.weight_decay,
            encode_u32(n)?,
        ];

        let host_params = to_host_vec(params);
        let host_grads = to_host_vec(gradients);
        let groups = workgroup_count(n)?;

        let updated = {
            let context = self.engine.cache.context();
            let params_buf = context.create_buffer::<f32>(n);
            params_buf.copy_from_host(&host_params)?;
            let grads_buf = context.create_buffer::<f32>(n);
            grads_buf.copy_from_host(&host_grads)?;
            let hyper_buf = context.create_buffer::<f32>(hyper.len());
            hyper_buf.copy_from_host(&hyper)?;
            let sum = self.engine.state.device_slot(0)?.clone();

            let kernel = self.engine.cache.kernel(OptimizerKernel::Adagrad)?;
            kernel.set_buffer("x", &params_buf);
            kernel.set_buffer("y", &grads_buf);
            kernel.set_buffer("a", &sum);
            kernel.set_buffer("b", &hyper_buf);
            kernel.dispatch([groups, 1, 1]);

            let mut out = vec![0.0f32; n];
            params_buf.copy_to_host(&mut out)?;
            out
        };

        from_host_vec(params, &updated)
    }
}

// ─── LAMB ───────────────────────────────────────────────────────────────────

/// GPU LAMB with a real layer-wise trust ratio.
///
/// One step is two dispatches of the same pipeline: the first advances the
/// moments, materialises the Adam-style update direction and reduces
/// `||params||^2` / `||update||^2` per workgroup; the host finishes the
/// reduction and computes `trust = ||params|| / ||update||`; the second applies
/// `p -= lr * trust * update`. When either norm is zero the ratio degenerates
/// to `1`, matching the reference implementation.
pub struct GpuLamb {
    engine: GpuStepEngine,
    params: AdamParams,
}

impl GpuLamb {
    /// Create the optimizer with the default WebGPU backend.
    pub fn new(params: AdamParams) -> Result<Self, GpuOptimError> {
        Self::with_config(params, GpuOptimizerConfig::default())
    }

    /// Create the optimizer on an explicit backend.
    pub fn with_config(
        params: AdamParams,
        config: GpuOptimizerConfig,
    ) -> Result<Self, GpuOptimError> {
        params.validate()?;
        // Persistent slots: 0 = m, 1 = v. The update direction and the norm
        // partials live in a per-step scratch buffer instead, because their
        // length depends on the workgroup count, not just on `n`.
        Ok(Self {
            engine: GpuStepEngine::new(config, 2)?,
            params,
        })
    }

    /// Backend actually in use.
    pub fn backend(&self) -> GpuBackend {
        self.engine.backend()
    }

    /// Upload the optimizer state to device memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_gpu`] so callers do not
    /// need a turbofish to pin the unused dimension parameter.
    pub fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    /// Download the optimizer state back to host memory.
    ///
    /// Inherent alias for [`GpuOptimizer::move_to_cpu`]; see
    /// [`Self::move_to_gpu`].
    pub fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    /// Deprecated alias for [`Self::move_to_gpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_gpu`")]
    pub fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_gpu()
    }

    /// Deprecated alias for [`Self::move_to_cpu`].
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_cpu`")]
    pub fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_cpu()
    }

    /// Whether a real device (not the CPU fallback) is backing this optimizer.
    pub fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    /// Number of updates applied so far.
    pub fn step_count(&self) -> u64 {
        self.engine.step_count
    }

    /// Trust ratio implied by a finished norm reduction.
    fn trust_ratio(param_norm: f32, update_norm: f32) -> f32 {
        if param_norm > 0.0 && update_norm > 0.0 {
            param_norm / update_norm
        } else {
            1.0
        }
    }
}

impl<D: Dimension> GpuOptimizer<f32, D> for GpuLamb {
    fn is_gpu_available(&self) -> bool {
        self.engine.backend() != GpuBackend::Cpu
    }

    fn move_to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_gpu()
    }

    fn move_to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.engine.move_to_cpu()
    }

    fn step_gpu(
        &mut self,
        params: &mut Array<f32, D>,
        gradients: &Array<f32, D>,
    ) -> Result<(), GpuOptimError> {
        check_shapes(params, gradients)?;
        let n = params.len();
        self.engine.prepare(n)?;
        self.engine.step_count = self.engine.step_count.saturating_add(1);

        let (bc1, bc2) = self.params.bias_corrections(self.engine.step_count);
        let groups = workgroup_count(n)?;
        let partial_len = (groups as usize).saturating_mul(2);

        let host_params = to_host_vec(params);
        let host_grads = to_host_vec(gradients);

        let mut hyper = [
            self.params.learning_rate,
            self.params.beta1,
            self.params.beta2,
            self.params.epsilon,
            self.params.weight_decay,
            bc1,
            bc2,
            encode_u32(n)?,
            encode_u32(0)?,
            1.0,
        ];

        let updated = {
            let context = self.engine.cache.context();
            let params_buf = context.create_buffer::<f32>(n);
            params_buf.copy_from_host(&host_params)?;
            let grads_buf = context.create_buffer::<f32>(n);
            grads_buf.copy_from_host(&host_grads)?;

            // Scratch layout: `update[0..n]` followed by `partials[n..]`, two
            // f32 per workgroup (sum of p^2, sum of update^2). Keeping them in
            // one buffer holds the binding count at six, which is the limit of
            // the deterministic Metal argument-table mapping.
            let scratch_len = n.saturating_add(partial_len);
            let scratch_buf = context.create_buffer::<f32>(scratch_len);
            scratch_buf.copy_from_host(&vec![0.0f32; scratch_len])?;
            let hyper_buf = context.create_buffer::<f32>(hyper.len());
            hyper_buf.copy_from_host(&hyper)?;

            let m_buf = self.engine.state.device_slot(0)?.clone();
            let v_buf = self.engine.state.device_slot(1)?.clone();

            // Phase 0: moments + update direction + partial norms.
            {
                let kernel = self.engine.cache.kernel(OptimizerKernel::Lamb)?;
                kernel.set_buffer("x", &params_buf);
                kernel.set_buffer("y", &grads_buf);
                kernel.set_buffer("a", &m_buf);
                kernel.set_buffer("b", &v_buf);
                kernel.set_buffer("result", &scratch_buf);
                kernel.set_buffer("output", &hyper_buf);
                kernel.dispatch([groups, 1, 1]);
            }

            let mut scratch = vec![0.0f32; scratch_len];
            scratch_buf.copy_to_host(&mut scratch)?;
            let mut param_sq = 0.0f64;
            let mut update_sq = 0.0f64;
            for pair in scratch[n..].chunks_exact(2) {
                param_sq += f64::from(pair[0]);
                update_sq += f64::from(pair[1]);
            }
            let trust = Self::trust_ratio(
                param_sq.max(0.0).sqrt() as f32,
                update_sq.max(0.0).sqrt() as f32,
            );

            // Phase 1: apply the trusted update.
            hyper[8] = encode_u32(1)?;
            hyper[9] = trust;
            hyper_buf.copy_from_host(&hyper)?;
            {
                let kernel = self.engine.cache.kernel(OptimizerKernel::Lamb)?;
                kernel.set_buffer("x", &params_buf);
                kernel.set_buffer("y", &grads_buf);
                kernel.set_buffer("a", &m_buf);
                kernel.set_buffer("b", &v_buf);
                kernel.set_buffer("result", &scratch_buf);
                kernel.set_buffer("output", &hyper_buf);
                kernel.dispatch([groups, 1, 1]);
            }

            let mut out = vec![0.0f32; n];
            params_buf.copy_to_host(&mut out)?;
            out
        };

        from_host_vec(params, &updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_params_reject_invalid_values() {
        assert!(AdamParams::new(1e-3, 0.9, 0.999, 1e-8, 0.0).is_ok());
        assert!(AdamParams::new(0.0, 0.9, 0.999, 1e-8, 0.0).is_err());
        assert!(AdamParams::new(1e-3, 1.0, 0.999, 1e-8, 0.0).is_err());
        assert!(AdamParams::new(1e-3, 0.9, f32::NAN, 1e-8, 0.0).is_err());
        assert!(AdamParams::new(1e-3, 0.9, 0.999, 0.0, 0.0).is_err());
        assert!(AdamParams::new(1e-3, 0.9, 0.999, 1e-8, -1.0).is_err());
    }

    #[test]
    fn adagrad_lr_decay_is_not_off_by_one() {
        let params = AdagradParams {
            learning_rate: 0.1,
            lr_decay: 0.5,
            ..AdagradParams::default()
        };
        // First step must use the base rate exactly.
        assert!((params.effective_lr(1) - 0.1).abs() < 1e-9);
        // Second step divides by (1 + 1 * 0.5).
        assert!((params.effective_lr(2) - 0.1 / 1.5).abs() < 1e-9);
        assert!((params.effective_lr(3) - 0.1 / 2.0).abs() < 1e-9);
    }

    #[test]
    fn encode_u32_round_trips_large_counts() {
        let n = 20_000_000usize; // > 2^24, not exactly representable as f32
        let encoded = encode_u32(n).expect("encodable");
        assert_eq!(encoded.to_bits() as usize, n);
    }

    #[test]
    fn workgroup_count_covers_the_tail() {
        assert_eq!(workgroup_count(1).expect("ok"), 1);
        assert_eq!(workgroup_count(256).expect("ok"), 1);
        assert_eq!(workgroup_count(257).expect("ok"), 2);
        assert_eq!(workgroup_count(0).expect("ok"), 0);
    }

    #[test]
    fn lamb_trust_ratio_degenerates_to_one() {
        assert_eq!(GpuLamb::trust_ratio(0.0, 1.0), 1.0);
        assert_eq!(GpuLamb::trust_ratio(1.0, 0.0), 1.0);
        assert!((GpuLamb::trust_ratio(4.0, 2.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn state_buffers_start_zeroed_on_the_host() {
        let state = StateBuffers::new(2, 8);
        assert!(!state.is_resident());
        assert_eq!(state.host.len(), 2);
        assert!(state.host.iter().all(|s| s.iter().all(|&x| x == 0.0)));
    }

    #[test]
    fn non_wgpu_backends_are_rejected_explicitly() {
        for backend in [
            GpuBackend::Cpu,
            GpuBackend::Cuda,
            GpuBackend::Rocm,
            GpuBackend::OpenCL,
        ] {
            let err = KernelCache::new(Some(backend))
                .err()
                .unwrap_or_else(|| panic!("backend {backend} must be rejected"));
            assert!(
                matches!(err, GpuOptimError::UnsupportedOperation(_)),
                "backend {backend} produced the wrong error: {err}"
            );
        }
    }
}
