//! Analytical hardware-aware cost models for Neural Architecture Search.
//!
//! `hardware_cost` provides **closed-form** estimates of the inference cost of a
//! candidate neural architecture: floating-point operations (FLOPs), parameter
//! count, activation memory, a **roofline** latency bound, and energy. Every
//! number produced here is the output of an *honest mathematical model*
//! parameterised by a [`HardwareProfile`] — it is **not** a measurement read
//! back from a real device. This module is the analytical *predictor* that
//! complements the runtime [`crate::nas_engine::resources::ResourceMonitor`]
//! (which observes actual usage on a live machine).
//!
//! # Why analytical models?
//!
//! During architecture search, thousands of candidate models must be ranked
//! before any of them is trained. Probing real hardware for each candidate is
//! far too slow. Instead, NAS frameworks rely on cheap, deterministic surrogate
//! cost models. The models here are the textbook ones used by FLOP counters and
//! roofline analyses; they are accurate enough to *rank* architectures while
//! being computable in microseconds.
//!
//! # FLOP model (counting each multiply-accumulate as two operations)
//!
//! - **Conv2d**: `2 · H_out · W_out · C_out · C_in · kH · kW`, with
//!   `H_out = ⌊(H_in + 2·pad − kH) / stride⌋ + 1` (and analogously for `W_out`).
//! - **Linear**: `2 · in_features · out_features`.
//! - **Attention** (`D = embed_dim`, `S = seq_len`, `h = num_heads`):
//!   `8·S·D²` (Q/K/V plus output projections) `+ 4·S²·D` (scores `Q·Kᵀ` plus
//!   the `context = softmaxᵀ·V` product, each `2·S²·D`) `+ 5·h·S²` (softmax over
//!   the `h·S·S` attention-weight elements).
//! - **Pooling**: `C · H_out · W_out · kH · kW` (one reduce op per window cell;
//!   no multiply-accumulate, hence no factor of two).
//! - **Activation / elementwise**: `N · ops_per_element` (ReLU = 1, … GELU = 8).
//! - **BatchNorm** (inference, affine fold): `2 · C · H · W`.
//!
//! # Roofline latency model
//!
//! For each layer the *operational (arithmetic) intensity* is
//! `I = FLOPs / bytes_moved`, where `bytes_moved` counts the input tensor, the
//! weights (incl. bias), and the output tensor, each read/written once at the
//! profile's numeric precision. The two roofline bounds are
//! `t_compute = FLOPs / peak_flops` and `t_memory = bytes_moved / bandwidth`;
//! the predicted latency is `max(t_compute, t_memory)` and the binding
//! [`Bottleneck`] is whichever term is larger. A layer is compute-bound exactly
//! when `I` exceeds the profile *ridge point* `peak_flops / bandwidth`.
//!
//! # Energy model
//!
//! `energy = FLOPs · energy_per_flop + bytes_moved · energy_per_byte`.
//!
//! # Latency lookup tables
//!
//! When real measurements are available they should be preferred over the
//! roofline bound. [`LatencyLookupTable`] stores measured
//! `(LayerSignature → latency)` entries; [`LatencyLookupTable::predict`] returns
//! an exact table hit when present, a FLOP-interpolated estimate from
//! neighbouring entries of the same layer kind, or — failing both — the
//! analytical roofline bound.
//!
//! # Example
//!
//! ```ignore
//! use optirs_nas::hardware_cost::{HardwareCostModel, HardwareProfile, LayerSpec};
//!
//! let model = HardwareCostModel::new();
//! let profile = HardwareProfile::server_gpu();
//! let layers = vec![
//!     LayerSpec::Conv2d {
//!         in_channels: 3,
//!         out_channels: 64,
//!         kernel: (3, 3),
//!         stride: (1, 1),
//!         padding: (1, 1),
//!         input_hw: (224, 224),
//!     },
//!     LayerSpec::Linear { in_features: 2048, out_features: 1000 },
//! ];
//! let report = model.estimate(&layers, &profile).expect("estimate");
//! println!("latency = {} s, energy = {} J", report.latency_s, report.energy_j);
//! ```

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of scalar operations attributed to each element of the attention
/// weight matrix during softmax (max-subtract, exp, running sum, divide).
const SOFTMAX_FLOPS_PER_ELEMENT: f64 = 5.0;

/// Operations per element for inference-time batch normalisation once the
/// statistics are folded into an affine transform (one multiply, one add).
const BATCHNORM_FLOPS_PER_ELEMENT: f64 = 2.0;

/// Which side of the roofline bounds a layer (or whole model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Bottleneck {
    /// Limited by arithmetic throughput (`t_compute ≥ t_memory`).
    Compute,
    /// Limited by memory bandwidth (`t_memory > t_compute`).
    Bandwidth,
}

/// Spatial pooling reduction variant. Both variants share the same FLOP and
/// memory footprint; the discriminant is kept so lookup-table signatures can
/// distinguish them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoolKind {
    /// Max pooling.
    Max,
    /// Average pooling.
    Average,
}

/// Elementwise activation function, used to weight the per-element FLOP cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationKind {
    /// Rectified linear unit (one compare).
    Relu,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Gaussian error linear unit.
    Gelu,
    /// Numerically-stabilised softmax (treated as an elementwise cost here).
    Softmax,
}

impl ActivationKind {
    /// Representative number of scalar floating-point operations per element.
    ///
    /// These are nominal complexity weights (transcendental functions are not a
    /// single hardware op); they are used only to keep the relative ordering of
    /// activations sensible inside the cost model.
    pub fn flops_per_element(self) -> f64 {
        match self {
            ActivationKind::Relu => 1.0,
            ActivationKind::Sigmoid => 4.0,
            ActivationKind::Tanh => 6.0,
            ActivationKind::Gelu => 8.0,
            ActivationKind::Softmax => 5.0,
        }
    }
}

/// Coarse layer family, used to group [`LatencyLookupTable`] entries for
/// interpolation and to tag a [`PerLayerCost`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerKind {
    /// 2-D convolution.
    Conv2d,
    /// Dense / fully-connected projection.
    Linear,
    /// Multi-head self attention block.
    Attention,
    /// Spatial pooling.
    Pooling,
    /// Elementwise activation.
    Activation,
    /// Batch normalisation.
    BatchNorm,
}

/// Specification of a single network layer with exactly the dimensions needed
/// to compute its analytical cost.
///
/// All fields are integral (`usize`) so a layer is hashable and serialisable;
/// derived cost quantities are reported as `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerSpec {
    /// 2-D convolution over an `(C_in, H_in, W_in)` feature map.
    Conv2d {
        /// Number of input channels.
        in_channels: usize,
        /// Number of output channels (filters).
        out_channels: usize,
        /// Kernel size `(kH, kW)`.
        kernel: (usize, usize),
        /// Stride `(sH, sW)`.
        stride: (usize, usize),
        /// Zero-padding `(pH, pW)` applied to each spatial side.
        padding: (usize, usize),
        /// Input spatial size `(H_in, W_in)`.
        input_hw: (usize, usize),
    },
    /// Dense projection mapping `in_features → out_features` for one token.
    Linear {
        /// Input feature dimension.
        in_features: usize,
        /// Output feature dimension.
        out_features: usize,
    },
    /// Multi-head self attention over a length-`seq_len` sequence.
    Attention {
        /// Sequence length `S`.
        seq_len: usize,
        /// Model / embedding dimension `D` (must be divisible by `num_heads`).
        embed_dim: usize,
        /// Number of attention heads `h`.
        num_heads: usize,
    },
    /// Spatial pooling over an `(C, H_in, W_in)` feature map.
    Pooling {
        /// Pooling reduction variant.
        kind: PoolKind,
        /// Number of channels (preserved by pooling).
        channels: usize,
        /// Pooling window `(kH, kW)`.
        kernel: (usize, usize),
        /// Stride `(sH, sW)`.
        stride: (usize, usize),
        /// Zero-padding `(pH, pW)`.
        padding: (usize, usize),
        /// Input spatial size `(H_in, W_in)`.
        input_hw: (usize, usize),
    },
    /// Elementwise activation applied to `num_elements` values.
    Activation {
        /// Activation function (sets the per-element FLOP weight).
        kind: ActivationKind,
        /// Number of elements in the tensor.
        num_elements: usize,
    },
    /// Batch normalisation over a `(C, H, W)` feature map (cheap compute, but a
    /// full-size activation tensor and `2·C` affine parameters).
    BatchNorm {
        /// Number of normalised feature channels `C`.
        num_features: usize,
        /// Spatial size `(H, W)` (use `(1, 1)` for a 1-D / token-wise norm).
        spatial_hw: (usize, usize),
    },
}

/// Compute the output extent of one spatial axis of a sliding-window layer.
fn windowed_output_dim(
    in_dim: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize> {
    if stride == 0 {
        return Err(OptimError::InvalidParameter(
            "stride must be >= 1".to_string(),
        ));
    }
    let padded = in_dim + 2 * padding;
    if kernel == 0 || kernel > padded {
        return Err(OptimError::InvalidParameter(format!(
            "kernel {kernel} exceeds padded input extent {padded}"
        )));
    }
    Ok((padded - kernel) / stride + 1)
}

/// Compute both output spatial extents of a sliding-window layer.
fn windowed_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(usize, usize)> {
    let out_h = windowed_output_dim(input_hw.0, kernel.0, stride.0, padding.0)?;
    let out_w = windowed_output_dim(input_hw.1, kernel.1, stride.1, padding.1)?;
    Ok((out_h, out_w))
}

/// Reject a zero dimension with a descriptive error.
fn require_positive(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        Err(OptimError::InvalidParameter(format!("{name} must be >= 1")))
    } else {
        Ok(())
    }
}

impl LayerSpec {
    /// The coarse family this layer belongs to.
    pub fn kind(&self) -> LayerKind {
        match self {
            LayerSpec::Conv2d { .. } => LayerKind::Conv2d,
            LayerSpec::Linear { .. } => LayerKind::Linear,
            LayerSpec::Attention { .. } => LayerKind::Attention,
            LayerSpec::Pooling { .. } => LayerKind::Pooling,
            LayerSpec::Activation { .. } => LayerKind::Activation,
            LayerSpec::BatchNorm { .. } => LayerKind::BatchNorm,
        }
    }

    /// Validate that every dimension is well-formed (positive, and for
    /// windowed layers that the kernel fits the padded input).
    ///
    /// Returns [`OptimError::InvalidParameter`] describing the first offending
    /// field. All cost methods validate before computing, so a malformed layer
    /// yields an honest error rather than a fabricated zero cost.
    pub fn validate(&self) -> Result<()> {
        match *self {
            LayerSpec::Conv2d {
                in_channels,
                out_channels,
                kernel,
                stride,
                padding,
                input_hw,
            } => {
                require_positive("conv in_channels", in_channels)?;
                require_positive("conv out_channels", out_channels)?;
                require_positive("conv kernel height", kernel.0)?;
                require_positive("conv kernel width", kernel.1)?;
                require_positive("conv input height", input_hw.0)?;
                require_positive("conv input width", input_hw.1)?;
                windowed_output_hw(input_hw, kernel, stride, padding)?;
            }
            LayerSpec::Linear {
                in_features,
                out_features,
            } => {
                require_positive("linear in_features", in_features)?;
                require_positive("linear out_features", out_features)?;
            }
            LayerSpec::Attention {
                seq_len,
                embed_dim,
                num_heads,
            } => {
                require_positive("attention seq_len", seq_len)?;
                require_positive("attention embed_dim", embed_dim)?;
                require_positive("attention num_heads", num_heads)?;
                if embed_dim % num_heads != 0 {
                    return Err(OptimError::InvalidParameter(format!(
                        "embed_dim {embed_dim} is not divisible by num_heads {num_heads}"
                    )));
                }
            }
            LayerSpec::Pooling {
                channels,
                kernel,
                stride,
                padding,
                input_hw,
                ..
            } => {
                require_positive("pool channels", channels)?;
                require_positive("pool kernel height", kernel.0)?;
                require_positive("pool kernel width", kernel.1)?;
                require_positive("pool input height", input_hw.0)?;
                require_positive("pool input width", input_hw.1)?;
                windowed_output_hw(input_hw, kernel, stride, padding)?;
            }
            LayerSpec::Activation { num_elements, .. } => {
                require_positive("activation num_elements", num_elements)?;
            }
            LayerSpec::BatchNorm {
                num_features,
                spatial_hw,
            } => {
                require_positive("batchnorm num_features", num_features)?;
                require_positive("batchnorm spatial height", spatial_hw.0)?;
                require_positive("batchnorm spatial width", spatial_hw.1)?;
            }
        }
        Ok(())
    }

    /// Number of elements in the layer's input tensor.
    fn input_elements(&self) -> f64 {
        match *self {
            LayerSpec::Conv2d {
                in_channels,
                input_hw,
                ..
            } => in_channels as f64 * input_hw.0 as f64 * input_hw.1 as f64,
            LayerSpec::Linear { in_features, .. } => in_features as f64,
            LayerSpec::Attention {
                seq_len, embed_dim, ..
            } => seq_len as f64 * embed_dim as f64,
            LayerSpec::Pooling {
                channels, input_hw, ..
            } => channels as f64 * input_hw.0 as f64 * input_hw.1 as f64,
            LayerSpec::Activation { num_elements, .. } => num_elements as f64,
            LayerSpec::BatchNorm {
                num_features,
                spatial_hw,
            } => num_features as f64 * spatial_hw.0 as f64 * spatial_hw.1 as f64,
        }
    }

    /// Number of elements in the layer's output tensor.
    ///
    /// Fails for windowed layers whose geometry is invalid.
    pub fn output_elements(&self) -> Result<f64> {
        let elements = match *self {
            LayerSpec::Conv2d {
                out_channels,
                kernel,
                stride,
                padding,
                input_hw,
                ..
            } => {
                let (out_h, out_w) = windowed_output_hw(input_hw, kernel, stride, padding)?;
                out_channels as f64 * out_h as f64 * out_w as f64
            }
            LayerSpec::Linear { out_features, .. } => out_features as f64,
            LayerSpec::Attention {
                seq_len, embed_dim, ..
            } => seq_len as f64 * embed_dim as f64,
            LayerSpec::Pooling {
                channels,
                kernel,
                stride,
                padding,
                input_hw,
                ..
            } => {
                let (out_h, out_w) = windowed_output_hw(input_hw, kernel, stride, padding)?;
                channels as f64 * out_h as f64 * out_w as f64
            }
            LayerSpec::Activation { num_elements, .. } => num_elements as f64,
            LayerSpec::BatchNorm {
                num_features,
                spatial_hw,
            } => num_features as f64 * spatial_hw.0 as f64 * spatial_hw.1 as f64,
        };
        Ok(elements)
    }

    /// Trainable parameter count for the layer (weights plus biases).
    ///
    /// Pooling and activation layers are parameter-free. Attention counts the
    /// three Q/K/V projections plus the output projection (`4·D²` weights and
    /// `4·D` biases); the head count only reshapes those parameters.
    pub fn parameter_count(&self) -> f64 {
        match *self {
            LayerSpec::Conv2d {
                in_channels,
                out_channels,
                kernel,
                ..
            } => {
                out_channels as f64 * in_channels as f64 * kernel.0 as f64 * kernel.1 as f64
                    + out_channels as f64
            }
            LayerSpec::Linear {
                in_features,
                out_features,
            } => in_features as f64 * out_features as f64 + out_features as f64,
            LayerSpec::Attention { embed_dim, .. } => {
                let d = embed_dim as f64;
                4.0 * d * d + 4.0 * d
            }
            LayerSpec::Pooling { .. } | LayerSpec::Activation { .. } => 0.0,
            LayerSpec::BatchNorm { num_features, .. } => 2.0 * num_features as f64,
        }
    }

    /// Floating-point operation count (multiply-accumulate counted as two).
    ///
    /// See the module-level documentation for the per-family formulas. Fails for
    /// windowed layers whose geometry is invalid.
    pub fn flops(&self) -> Result<f64> {
        self.validate()?;
        let flops = match *self {
            LayerSpec::Conv2d {
                in_channels,
                out_channels,
                kernel,
                stride,
                padding,
                input_hw,
            } => {
                let (out_h, out_w) = windowed_output_hw(input_hw, kernel, stride, padding)?;
                2.0 * out_h as f64
                    * out_w as f64
                    * out_channels as f64
                    * in_channels as f64
                    * kernel.0 as f64
                    * kernel.1 as f64
            }
            LayerSpec::Linear {
                in_features,
                out_features,
            } => 2.0 * in_features as f64 * out_features as f64,
            LayerSpec::Attention {
                seq_len,
                embed_dim,
                num_heads,
            } => {
                let s = seq_len as f64;
                let d = embed_dim as f64;
                let h = num_heads as f64;
                let qkv_and_out_proj = 8.0 * s * d * d;
                let scores_and_context = 4.0 * s * s * d;
                let softmax = SOFTMAX_FLOPS_PER_ELEMENT * h * s * s;
                qkv_and_out_proj + scores_and_context + softmax
            }
            LayerSpec::Pooling {
                channels,
                kernel,
                stride,
                padding,
                input_hw,
                ..
            } => {
                let (out_h, out_w) = windowed_output_hw(input_hw, kernel, stride, padding)?;
                channels as f64 * out_h as f64 * out_w as f64 * kernel.0 as f64 * kernel.1 as f64
            }
            LayerSpec::Activation { kind, num_elements } => {
                num_elements as f64 * kind.flops_per_element()
            }
            LayerSpec::BatchNorm {
                num_features,
                spatial_hw,
            } => {
                num_features as f64
                    * spatial_hw.0 as f64
                    * spatial_hw.1 as f64
                    * BATCHNORM_FLOPS_PER_ELEMENT
            }
        };
        Ok(flops)
    }

    /// Activation-memory footprint in bytes: output-tensor element count times
    /// the numeric precision `dtype_bytes`.
    pub fn activation_bytes(&self, dtype_bytes: usize) -> Result<f64> {
        self.validate()?;
        Ok(self.output_elements()? * dtype_bytes as f64)
    }

    /// Total bytes moved between memory and compute for one forward pass:
    /// `(input + weights/bias + output) · dtype_bytes`. This is the denominator
    /// of the roofline arithmetic intensity.
    pub fn bytes_moved(&self, dtype_bytes: usize) -> Result<f64> {
        self.validate()?;
        let elements = self.input_elements() + self.parameter_count() + self.output_elements()?;
        Ok(elements * dtype_bytes as f64)
    }
}

/// Representative ("nominal") hardware deployment target.
///
/// The constants in the [`HardwareProfile::edge_cpu`], [`HardwareProfile::mobile_gpu`]
/// and [`HardwareProfile::server_gpu`] presets are **representative order-of-magnitude
/// figures for a class of device**, hand-written into the source. They are
/// **not** queried, probed, or auto-detected from any physical hardware. Use
/// [`HardwareProfile::new`] to supply numbers measured on your own target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Human-readable label for the profile.
    pub name: String,
    /// Sustained peak arithmetic throughput in FLOPs per second.
    pub peak_flops_per_s: f64,
    /// Sustained memory bandwidth in bytes per second.
    pub memory_bandwidth_bytes_per_s: f64,
    /// Energy charged per floating-point operation, in joules.
    pub energy_per_flop_j: f64,
    /// Energy charged per byte moved, in joules.
    pub energy_per_byte_j: f64,
    /// Numeric precision of activations/weights, in bytes per element
    /// (e.g. `4` for fp32, `2` for fp16, `1` for int8).
    pub dtype_bytes: usize,
}

impl HardwareProfile {
    /// Construct a profile from explicit figures (e.g. measured on your target).
    pub fn new(
        name: impl Into<String>,
        peak_flops_per_s: f64,
        memory_bandwidth_bytes_per_s: f64,
        energy_per_flop_j: f64,
        energy_per_byte_j: f64,
        dtype_bytes: usize,
    ) -> Self {
        Self {
            name: name.into(),
            peak_flops_per_s,
            memory_bandwidth_bytes_per_s,
            energy_per_flop_j,
            energy_per_byte_j,
            dtype_bytes,
        }
    }

    /// Nominal embedded / mobile CPU class (e.g. an ARM Cortex-A application
    /// core with NEON, LPDDR4X memory). Representative figures only.
    pub fn edge_cpu() -> Self {
        Self::new("edge-cpu", 2.5e10, 1.5e10, 1.0e-10, 1.5e-9, 4)
    }

    /// Nominal mobile GPU class (integrated mobile accelerator running fp16).
    /// Representative figures only.
    pub fn mobile_gpu() -> Self {
        Self::new("mobile-gpu", 1.2e12, 6.0e10, 2.0e-11, 6.0e-10, 2)
    }

    /// Nominal data-center GPU class (HBM-backed accelerator running fp32).
    /// Representative figures only.
    pub fn server_gpu() -> Self {
        Self::new("server-gpu", 1.95e13, 1.55e12, 5.0e-12, 1.0e-10, 4)
    }

    /// Roofline ridge point in FLOPs per byte: a layer is compute-bound exactly
    /// when its arithmetic intensity exceeds this value. Assumes a validated
    /// profile (positive bandwidth).
    pub fn ridge_point(&self) -> f64 {
        self.peak_flops_per_s / self.memory_bandwidth_bytes_per_s
    }

    /// Validate that the profile figures are finite and physically sensible.
    pub fn validate(&self) -> Result<()> {
        if !self.peak_flops_per_s.is_finite() || self.peak_flops_per_s <= 0.0 {
            return Err(OptimError::InvalidParameter(
                "peak_flops_per_s must be finite and > 0".to_string(),
            ));
        }
        if !self.memory_bandwidth_bytes_per_s.is_finite()
            || self.memory_bandwidth_bytes_per_s <= 0.0
        {
            return Err(OptimError::InvalidParameter(
                "memory_bandwidth_bytes_per_s must be finite and > 0".to_string(),
            ));
        }
        if !self.energy_per_flop_j.is_finite() || self.energy_per_flop_j < 0.0 {
            return Err(OptimError::InvalidParameter(
                "energy_per_flop_j must be finite and >= 0".to_string(),
            ));
        }
        if !self.energy_per_byte_j.is_finite() || self.energy_per_byte_j < 0.0 {
            return Err(OptimError::InvalidParameter(
                "energy_per_byte_j must be finite and >= 0".to_string(),
            ));
        }
        if self.dtype_bytes == 0 {
            return Err(OptimError::InvalidParameter(
                "dtype_bytes must be >= 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Energy in joules for moving `bytes_moved` bytes and executing `flops`
    /// floating-point operations on this profile.
    pub fn energy_joules(&self, flops: f64, bytes_moved: f64) -> f64 {
        flops * self.energy_per_flop_j + bytes_moved * self.energy_per_byte_j
    }

    /// Evaluate the roofline model for a single layer on this profile.
    pub fn roofline(&self, layer: &LayerSpec) -> Result<RooflineResult> {
        self.validate()?;
        let flops = layer.flops()?;
        let bytes_moved = layer.bytes_moved(self.dtype_bytes)?;
        let arithmetic_intensity = if bytes_moved > 0.0 {
            flops / bytes_moved
        } else {
            0.0
        };
        let compute_bound_s = flops / self.peak_flops_per_s;
        let memory_bound_s = bytes_moved / self.memory_bandwidth_bytes_per_s;
        let (latency_s, bottleneck) = if compute_bound_s >= memory_bound_s {
            (compute_bound_s, Bottleneck::Compute)
        } else {
            (memory_bound_s, Bottleneck::Bandwidth)
        };
        Ok(RooflineResult {
            flops,
            bytes_moved,
            arithmetic_intensity,
            compute_bound_s,
            memory_bound_s,
            latency_s,
            bottleneck,
        })
    }
}

/// Outcome of the roofline analysis for one layer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RooflineResult {
    /// Floating-point operation count.
    pub flops: f64,
    /// Bytes moved (input + weights + output).
    pub bytes_moved: f64,
    /// Arithmetic intensity in FLOPs per byte.
    pub arithmetic_intensity: f64,
    /// Compute-bound latency term `flops / peak_flops` (seconds).
    pub compute_bound_s: f64,
    /// Memory-bound latency term `bytes / bandwidth` (seconds).
    pub memory_bound_s: f64,
    /// Predicted latency `max(compute_bound_s, memory_bound_s)` (seconds).
    pub latency_s: f64,
    /// Which term binds the latency.
    pub bottleneck: Bottleneck,
}

/// Hashable key identifying a layer measurement at a specific numeric precision.
///
/// A measured latency depends on both the layer shape and the precision it was
/// captured at, so the signature pairs a [`LayerSpec`] with `dtype_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerSignature {
    /// The layer shape.
    pub spec: LayerSpec,
    /// Numeric precision (bytes per element) of the measurement.
    pub dtype_bytes: usize,
}

impl LayerSignature {
    /// Build a signature from an owned spec and precision.
    pub fn new(spec: LayerSpec, dtype_bytes: usize) -> Self {
        Self { spec, dtype_bytes }
    }

    /// Build a signature from a borrowed layer and precision.
    pub fn of(layer: &LayerSpec, dtype_bytes: usize) -> Self {
        Self {
            spec: *layer,
            dtype_bytes,
        }
    }
}

/// Where a predicted latency came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LatencySource {
    /// Exact hit in the lookup table.
    TableExact,
    /// Interpolated / scaled from neighbouring same-kind table entries.
    TableInterpolated,
    /// Analytical roofline fallback (no usable table entry).
    Roofline,
}

/// A latency prediction together with its provenance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatencyPrediction {
    /// Predicted latency in seconds.
    pub latency_s: f64,
    /// How the prediction was produced.
    pub source: LatencySource,
}

/// Table of measured `(LayerSignature → latency_s)` entries with roofline
/// fallback.
///
/// A single table is implicitly tied to one hardware target; entries are keyed
/// by numeric precision so the same shape can hold fp16 and fp32 measurements.
#[derive(Debug, Clone, Default)]
pub struct LatencyLookupTable {
    entries: HashMap<LayerSignature, f64>,
}

impl LatencyLookupTable {
    /// Create an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored measurements.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table holds no measurements.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a raw `(signature → latency)` measurement.
    pub fn insert(&mut self, signature: LayerSignature, latency_s: f64) {
        self.entries.insert(signature, latency_s);
    }

    /// Record a measured latency for a layer at a given precision, validating
    /// the layer and the latency value.
    pub fn record(&mut self, layer: &LayerSpec, dtype_bytes: usize, latency_s: f64) -> Result<()> {
        if !latency_s.is_finite() || latency_s < 0.0 {
            return Err(OptimError::InvalidParameter(
                "measured latency must be finite and >= 0".to_string(),
            ));
        }
        if dtype_bytes == 0 {
            return Err(OptimError::InvalidParameter(
                "dtype_bytes must be >= 1".to_string(),
            ));
        }
        layer.validate()?;
        self.entries
            .insert(LayerSignature::of(layer, dtype_bytes), latency_s);
        Ok(())
    }

    /// Predict the latency of `layer` on `profile`.
    ///
    /// Resolution order: an exact table hit, then a FLOP-interpolated estimate
    /// from same-kind / same-precision neighbours (linear interpolation between
    /// the bracketing entries, or proportional scaling when the query lies
    /// outside the measured range), then the analytical roofline bound.
    pub fn predict(
        &self,
        layer: &LayerSpec,
        profile: &HardwareProfile,
    ) -> Result<LatencyPrediction> {
        profile.validate()?;
        let query_flops = layer.flops()?;
        let dtype_bytes = profile.dtype_bytes;
        let signature = LayerSignature::of(layer, dtype_bytes);

        if let Some(&latency_s) = self.entries.get(&signature) {
            return Ok(LatencyPrediction {
                latency_s,
                source: LatencySource::TableExact,
            });
        }

        // Gather the bracketing neighbours in FLOP space among same-kind,
        // same-precision entries: `lower` is the largest entry with
        // `flops <= query`, `upper` the smallest with `flops >= query`.
        let kind = layer.kind();
        let mut lower: Option<(f64, f64)> = None;
        let mut upper: Option<(f64, f64)> = None;
        for (entry_sig, &entry_latency) in &self.entries {
            if entry_sig.dtype_bytes != dtype_bytes || entry_sig.spec.kind() != kind {
                continue;
            }
            let entry_flops = match entry_sig.spec.flops() {
                Ok(value) => value,
                Err(_) => continue,
            };
            if entry_flops <= query_flops {
                let replace = match lower {
                    None => true,
                    Some((flops, _)) => entry_flops > flops,
                };
                if replace {
                    lower = Some((entry_flops, entry_latency));
                }
            }
            if entry_flops >= query_flops {
                let replace = match upper {
                    None => true,
                    Some((flops, _)) => entry_flops < flops,
                };
                if replace {
                    upper = Some((entry_flops, entry_latency));
                }
            }
        }

        let prediction = match (lower, upper) {
            (Some((lo_flops, lo_lat)), Some((hi_flops, hi_lat))) => {
                let span = hi_flops - lo_flops;
                let latency_s = if span.abs() < f64::EPSILON {
                    lo_lat
                } else {
                    let t = (query_flops - lo_flops) / span;
                    lo_lat + t * (hi_lat - lo_lat)
                };
                LatencyPrediction {
                    latency_s,
                    source: LatencySource::TableInterpolated,
                }
            }
            (Some((lo_flops, lo_lat)), None) => {
                let latency_s = if lo_flops > 0.0 {
                    lo_lat * (query_flops / lo_flops)
                } else {
                    lo_lat
                };
                LatencyPrediction {
                    latency_s,
                    source: LatencySource::TableInterpolated,
                }
            }
            (None, Some((hi_flops, hi_lat))) => {
                let latency_s = if hi_flops > 0.0 {
                    hi_lat * (query_flops / hi_flops)
                } else {
                    hi_lat
                };
                LatencyPrediction {
                    latency_s,
                    source: LatencySource::TableInterpolated,
                }
            }
            (None, None) => {
                let roofline = profile.roofline(layer)?;
                LatencyPrediction {
                    latency_s: roofline.latency_s,
                    source: LatencySource::Roofline,
                }
            }
        };
        Ok(prediction)
    }
}

/// Per-layer cost breakdown produced by [`HardwareCostModel::estimate`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PerLayerCost {
    /// Index of the layer within the model.
    pub layer_index: usize,
    /// Layer family.
    pub kind: LayerKind,
    /// Floating-point operation count.
    pub flops: f64,
    /// Trainable parameter count.
    pub parameters: f64,
    /// Activation-memory footprint in bytes.
    pub activation_bytes: f64,
    /// Bytes moved (input + weights + output).
    pub bytes_moved: f64,
    /// Arithmetic intensity in FLOPs per byte.
    pub arithmetic_intensity: f64,
    /// Compute-bound latency term (seconds).
    pub compute_bound_s: f64,
    /// Memory-bound latency term (seconds).
    pub memory_bound_s: f64,
    /// Predicted latency (seconds) — table-driven when available, else roofline.
    pub latency_s: f64,
    /// Provenance of `latency_s`.
    pub latency_source: LatencySource,
    /// Energy in joules.
    pub energy_j: f64,
    /// Binding roofline regime for this layer.
    pub bottleneck: Bottleneck,
}

/// Whole-model analytical cost report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostReport {
    /// Total floating-point operation count.
    pub total_flops: f64,
    /// Total trainable parameter count.
    pub total_params: f64,
    /// Total activation-memory footprint in bytes.
    pub total_activation_bytes: f64,
    /// Total predicted latency in seconds (sum of per-layer latencies).
    pub latency_s: f64,
    /// Total energy in joules.
    pub energy_j: f64,
    /// Per-layer breakdown in model order.
    pub per_layer: Vec<PerLayerCost>,
    /// Dominant bottleneck: whichever regime accounts for more total latency.
    pub bottleneck: Bottleneck,
}

/// Top-level analytical cost estimator tying the FLOP, memory, roofline,
/// energy, and lookup-table models together.
#[derive(Debug, Clone, Default)]
pub struct HardwareCostModel {
    lookup: LatencyLookupTable,
}

impl HardwareCostModel {
    /// Create an estimator that uses the pure roofline latency model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an estimator backed by a measured latency lookup table.
    pub fn with_lookup(lookup: LatencyLookupTable) -> Self {
        Self { lookup }
    }

    /// Shared access to the backing lookup table.
    pub fn lookup(&self) -> &LatencyLookupTable {
        &self.lookup
    }

    /// Mutable access to the backing lookup table (to record measurements).
    pub fn lookup_mut(&mut self) -> &mut LatencyLookupTable {
        &mut self.lookup
    }

    /// Estimate the whole-model cost of `layers` on `profile`.
    ///
    /// FLOPs, parameters, activation bytes, energy, and the roofline bottleneck
    /// are computed analytically; latency is taken from the lookup table when a
    /// matching measurement exists and falls back to the roofline bound
    /// otherwise. An empty model yields an all-zero report. Returns an error if
    /// the profile or any layer is malformed.
    pub fn estimate(&self, layers: &[LayerSpec], profile: &HardwareProfile) -> Result<CostReport> {
        profile.validate()?;

        let mut per_layer = Vec::with_capacity(layers.len());
        let mut total_flops = 0.0;
        let mut total_params = 0.0;
        let mut total_activation_bytes = 0.0;
        let mut latency_s = 0.0;
        let mut energy_j = 0.0;
        let mut compute_latency = 0.0;
        let mut bandwidth_latency = 0.0;

        for (layer_index, layer) in layers.iter().enumerate() {
            let roofline = profile.roofline(layer)?;
            let parameters = layer.parameter_count();
            let activation_bytes = layer.activation_bytes(profile.dtype_bytes)?;
            let energy = profile.energy_joules(roofline.flops, roofline.bytes_moved);
            let prediction = self.lookup.predict(layer, profile)?;

            let cost = PerLayerCost {
                layer_index,
                kind: layer.kind(),
                flops: roofline.flops,
                parameters,
                activation_bytes,
                bytes_moved: roofline.bytes_moved,
                arithmetic_intensity: roofline.arithmetic_intensity,
                compute_bound_s: roofline.compute_bound_s,
                memory_bound_s: roofline.memory_bound_s,
                latency_s: prediction.latency_s,
                latency_source: prediction.source,
                energy_j: energy,
                bottleneck: roofline.bottleneck,
            };

            total_flops += cost.flops;
            total_params += cost.parameters;
            total_activation_bytes += cost.activation_bytes;
            latency_s += cost.latency_s;
            energy_j += cost.energy_j;
            match cost.bottleneck {
                Bottleneck::Compute => compute_latency += cost.latency_s,
                Bottleneck::Bandwidth => bandwidth_latency += cost.latency_s,
            }

            per_layer.push(cost);
        }

        let bottleneck = if compute_latency >= bandwidth_latency {
            Bottleneck::Compute
        } else {
            Bottleneck::Bandwidth
        };

        Ok(CostReport {
            total_flops,
            total_params,
            total_activation_bytes,
            latency_s,
            energy_j,
            per_layer,
            bottleneck,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing exact integer-valued `f64` results.
    const EPS: f64 = 1e-6;

    fn small_conv() -> LayerSpec {
        // 3 -> 8 channels, 3x3 kernel, stride 1, no padding, 8x8 input.
        // out = (8 - 3)/1 + 1 = 6, so output is 8 x 6 x 6.
        LayerSpec::Conv2d {
            in_channels: 3,
            out_channels: 8,
            kernel: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            input_hw: (8, 8),
        }
    }

    #[test]
    fn test_conv2d_flops_hand_verified() {
        let conv = small_conv();
        // 2 * 6 * 6 * 8 * 3 * 3 * 3 = 15552.
        let flops = conv.flops().expect("conv flops");
        assert!((flops - 15552.0).abs() < EPS, "got {flops}");
    }

    #[test]
    fn test_conv2d_params_and_activation_bytes() {
        let conv = small_conv();
        // weights 8*3*3*3 = 216, bias 8 -> 224.
        let params = conv.parameter_count();
        assert!((params - 224.0).abs() < EPS, "params {params}");
        // output elements 8*6*6 = 288, fp32 -> 1152 bytes.
        let bytes = conv.activation_bytes(4).expect("conv activation bytes");
        assert!((bytes - 1152.0).abs() < EPS, "bytes {bytes}");
    }

    #[test]
    fn test_linear_flops_hand_verified() {
        let linear = LayerSpec::Linear {
            in_features: 128,
            out_features: 64,
        };
        // 2 * 128 * 64 = 16384.
        let flops = linear.flops().expect("linear flops");
        assert!((flops - 16384.0).abs() < EPS, "flops {flops}");
        // params 128*64 + 64 = 8256.
        let params = linear.parameter_count();
        assert!((params - 8256.0).abs() < EPS, "params {params}");
        // output 64 elements, fp32 -> 256 bytes.
        let bytes = linear.activation_bytes(4).expect("linear activation bytes");
        assert!((bytes - 256.0).abs() < EPS, "bytes {bytes}");
    }

    #[test]
    fn test_attention_flops_and_params_hand_verified() {
        // S = 2, D = 4, h = 2:
        // qkv+out = 8*S*D^2 = 8*2*16 = 256
        // scores+context = 4*S^2*D = 4*4*4 = 64
        // softmax = 5*h*S^2 = 5*2*4 = 40
        // total = 360.
        let attention = LayerSpec::Attention {
            seq_len: 2,
            embed_dim: 4,
            num_heads: 2,
        };
        let flops = attention.flops().expect("attention flops");
        assert!((flops - 360.0).abs() < EPS, "flops {flops}");
        // params 4*D^2 + 4*D = 4*16 + 4*4 = 80.
        let params = attention.parameter_count();
        assert!((params - 80.0).abs() < EPS, "params {params}");
    }

    #[test]
    fn test_conv_output_dim_with_stride_and_padding() {
        // (32 + 2*1 - 3)/2 + 1 = 31/2 + 1 = 15 + 1 = 16.
        let conv = LayerSpec::Conv2d {
            in_channels: 16,
            out_channels: 16,
            kernel: (3, 3),
            stride: (2, 2),
            padding: (1, 1),
            input_hw: (32, 32),
        };
        // output elements 16 * 16 * 16 = 4096.
        let elements = conv.output_elements().expect("output elements");
        assert!((elements - 4096.0).abs() < EPS, "elements {elements}");
    }

    #[test]
    fn test_roofline_compute_bound_for_high_intensity_layer() {
        // A fat convolution reuses each weight across a large spatial grid, so
        // its arithmetic intensity sits well above the ridge point.
        let conv = LayerSpec::Conv2d {
            in_channels: 64,
            out_channels: 64,
            kernel: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            input_hw: (56, 56),
        };
        let profile = HardwareProfile::server_gpu();
        let roofline = profile.roofline(&conv).expect("roofline");
        assert!(
            roofline.arithmetic_intensity > profile.ridge_point(),
            "intensity {} should exceed ridge {}",
            roofline.arithmetic_intensity,
            profile.ridge_point()
        );
        assert_eq!(roofline.bottleneck, Bottleneck::Compute);
    }

    #[test]
    fn test_roofline_bandwidth_bound_for_low_intensity_layer() {
        // Elementwise ReLU does one op per element but moves both the input and
        // output tensors: intensity is far below any ridge point.
        let relu = LayerSpec::Activation {
            kind: ActivationKind::Relu,
            num_elements: 1_000_000,
        };
        let profile = HardwareProfile::server_gpu();
        let roofline = profile.roofline(&relu).expect("roofline");
        assert!(
            roofline.arithmetic_intensity < profile.ridge_point(),
            "intensity {} should be below ridge {}",
            roofline.arithmetic_intensity,
            profile.ridge_point()
        );
        assert_eq!(roofline.bottleneck, Bottleneck::Bandwidth);
    }

    #[test]
    fn test_energy_is_monotonic_in_flops() {
        let profile = HardwareProfile::server_gpu();
        let small = LayerSpec::Activation {
            kind: ActivationKind::Relu,
            num_elements: 1_000_000,
        };
        let large = LayerSpec::Activation {
            kind: ActivationKind::Relu,
            num_elements: 2_000_000,
        };
        let energy_small = {
            let r = profile.roofline(&small).expect("roofline small");
            profile.energy_joules(r.flops, r.bytes_moved)
        };
        let energy_large = {
            let r = profile.roofline(&large).expect("roofline large");
            profile.energy_joules(r.flops, r.bytes_moved)
        };
        assert!(
            energy_large > energy_small,
            "energy should grow with FLOPs: {energy_small} vs {energy_large}"
        );
    }

    #[test]
    fn test_lookup_table_exact_hit() {
        let profile = HardwareProfile::server_gpu();
        let conv = small_conv();
        let mut table = LatencyLookupTable::new();
        table
            .record(&conv, profile.dtype_bytes, 0.005)
            .expect("record");
        let prediction = table.predict(&conv, &profile).expect("predict");
        assert_eq!(prediction.source, LatencySource::TableExact);
        assert!((prediction.latency_s - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_lookup_table_interpolation() {
        let profile = HardwareProfile::server_gpu();
        let make_conv = |channels: usize| LayerSpec::Conv2d {
            in_channels: channels,
            out_channels: channels,
            kernel: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            input_hw: (16, 16),
        };
        let low = make_conv(8);
        let mid = make_conv(16);
        let high = make_conv(32);

        let mut table = LatencyLookupTable::new();
        table.record(&low, profile.dtype_bytes, 0.001).expect("low");
        table
            .record(&high, profile.dtype_bytes, 0.010)
            .expect("high");

        let prediction = table.predict(&mid, &profile).expect("predict mid");
        assert_eq!(prediction.source, LatencySource::TableInterpolated);

        // FLOPs: low 225792, mid 903168, high 3612672.
        // t = (903168 - 225792)/(3612672 - 225792) = 0.2,
        // latency = 0.001 + 0.2*(0.010 - 0.001) = 0.0028.
        assert!(
            (prediction.latency_s - 0.0028).abs() < 1e-9,
            "interpolated latency {}",
            prediction.latency_s
        );
    }

    #[test]
    fn test_lookup_table_roofline_fallback() {
        let profile = HardwareProfile::server_gpu();
        // Table holds only a convolution; querying a Linear has no same-kind
        // neighbour and must fall back to the roofline bound.
        let conv = small_conv();
        let linear = LayerSpec::Linear {
            in_features: 512,
            out_features: 512,
        };
        let mut table = LatencyLookupTable::new();
        table
            .record(&conv, profile.dtype_bytes, 0.005)
            .expect("record");

        let prediction = table.predict(&linear, &profile).expect("predict");
        assert_eq!(prediction.source, LatencySource::Roofline);
        let roofline = profile.roofline(&linear).expect("roofline");
        assert!((prediction.latency_s - roofline.latency_s).abs() < 1e-15);
    }

    #[test]
    fn test_estimate_full_model_aggregates() {
        let profile = HardwareProfile::server_gpu();
        let model = HardwareCostModel::new();
        let layers = vec![
            LayerSpec::Conv2d {
                in_channels: 3,
                out_channels: 32,
                kernel: (3, 3),
                stride: (1, 1),
                padding: (1, 1),
                input_hw: (32, 32),
            },
            LayerSpec::BatchNorm {
                num_features: 32,
                spatial_hw: (32, 32),
            },
            LayerSpec::Activation {
                kind: ActivationKind::Relu,
                num_elements: 32 * 32 * 32,
            },
            LayerSpec::Pooling {
                kind: PoolKind::Max,
                channels: 32,
                kernel: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
                input_hw: (32, 32),
            },
            LayerSpec::Linear {
                in_features: 32 * 16 * 16,
                out_features: 10,
            },
        ];

        let report = model.estimate(&layers, &profile).expect("estimate");
        assert_eq!(report.per_layer.len(), 5);

        let summed_flops: f64 = report.per_layer.iter().map(|c| c.flops).sum();
        let summed_params: f64 = report.per_layer.iter().map(|c| c.parameters).sum();
        let summed_latency: f64 = report.per_layer.iter().map(|c| c.latency_s).sum();
        assert!((report.total_flops - summed_flops).abs() < 1e-3);
        assert!((report.total_params - summed_params).abs() < 1e-3);
        assert!((report.latency_s - summed_latency).abs() < 1e-15);

        assert!(report.total_flops > 0.0);
        assert!(report.total_params > 0.0);
        assert!(report.total_activation_bytes > 0.0);
        assert!(report.energy_j > 0.0);
        // Every layer's latency must come from the roofline (no table set).
        assert!(report
            .per_layer
            .iter()
            .all(|c| c.latency_source == LatencySource::Roofline));
    }

    #[test]
    fn test_invalid_layer_geometry_rejected() {
        // A 9x9 kernel cannot fit a 4x4 input with no padding.
        let conv = LayerSpec::Conv2d {
            in_channels: 8,
            out_channels: 8,
            kernel: (9, 9),
            stride: (1, 1),
            padding: (0, 0),
            input_hw: (4, 4),
        };
        assert!(matches!(conv.flops(), Err(OptimError::InvalidParameter(_))));
        let profile = HardwareProfile::server_gpu();
        let model = HardwareCostModel::new();
        assert!(matches!(
            model.estimate(&[conv], &profile),
            Err(OptimError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_attention_requires_divisible_heads() {
        let attention = LayerSpec::Attention {
            seq_len: 8,
            embed_dim: 10,
            num_heads: 3,
        };
        assert!(matches!(
            attention.validate(),
            Err(OptimError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_invalid_profile_rejected() {
        let profile = HardwareProfile::new("broken", 0.0, 1.0e9, 1.0e-12, 1.0e-12, 4);
        assert!(matches!(
            profile.validate(),
            Err(OptimError::InvalidParameter(_))
        ));
        let model = HardwareCostModel::new();
        let linear = LayerSpec::Linear {
            in_features: 8,
            out_features: 8,
        };
        assert!(matches!(
            model.estimate(&[linear], &profile),
            Err(OptimError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_presets_are_valid_and_have_positive_ridge() {
        for profile in [
            HardwareProfile::edge_cpu(),
            HardwareProfile::mobile_gpu(),
            HardwareProfile::server_gpu(),
        ] {
            assert!(profile.validate().is_ok(), "{} invalid", profile.name);
            assert!(profile.ridge_point() > 0.0);
        }
    }

    #[test]
    fn test_signature_hash_distinguishes_precision() {
        use std::collections::HashSet;
        let conv = small_conv();
        let mut set: HashSet<LayerSignature> = HashSet::new();
        set.insert(LayerSignature::of(&conv, 4));
        set.insert(LayerSignature::of(&conv, 4));
        set.insert(LayerSignature::of(&conv, 2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_layer_spec_serde_roundtrip() {
        let conv = small_conv();
        let json = serde_json::to_string(&conv).expect("serialize");
        let restored: LayerSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(conv, restored);
    }
}
