//! # Quantization-Aware Training (QAT) primitives
//!
//! This module is a **CPU reference implementation** of the numerical core used
//! for quantization-aware training. It contains no real GPU calls: every routine
//! operates on host [`Array1`] / [`Array2`] data and exactly simulates the
//! precision loss that the corresponding low-precision GPU kernels would
//! introduce. The intent is that the *bit-for-bit* rounding behaviour modelled
//! here matches what a fused int8 / fp8 kernel produces, so a network trained
//! with these "fake-quant" operators behaves like the eventually-deployed
//! quantized network.
//!
//! ## What "fake quantization" means
//!
//! A fake-quant operator maps a floating-point value through the
//! quantize/dequantize round trip **while staying in floating point**:
//!
//! ```text
//!   fake_quant(x) = dequant(quant(x))
//! ```
//!
//! The result is the value the network *would* see if `x` were stored in the
//! low-precision format, but it remains an `f64` so that the rest of the
//! forward/backward pass runs in full precision. The gradient of this
//! (piecewise-constant, hence a.e. zero-derivative) operator is supplied by the
//! straight-through estimator (see [`fake_quant_backward`]).
//!
//! ## Integer quantization (`int8` / `int4`)
//!
//! For an affine integer grid with step `scale`, integer zero-point
//! `zero_point` and clamp range `[qmin, qmax]`:
//!
//! ```text
//!   quant(x)   = clamp(round(x / scale) + zero_point, qmin, qmax)
//!   dequant(q) = (q - zero_point) * scale
//! ```
//!
//! Two schemes are supported ([`QuantScheme`]):
//!
//! * **Symmetric** -- `zero_point = 0`. The grid is symmetric about zero and the
//!   scale is derived from the absolute maximum. To keep the negative and
//!   positive arms equal in length we use the **restricted** signed range, i.e.
//!   `int8` uses `qmin = -127, qmax = 127` (the `-128` code is dropped) and
//!   `int4` uses `qmin = -7, qmax = 7`. `scale = absmax / qmax`.
//! * **Affine** (asymmetric) -- the scale comes from the real `[min, max]`
//!   interval (nudged to include the real value `0`) and `zero_point` is the
//!   integer code onto which the real value `0` maps. Affine uses the **full**
//!   signed range, `int8 = [-128, 127]`, `int4 = [-8, 7]`.
//!
//! Note that because `dequant(quant(x)) = round(x / scale) * scale` the
//! reconstructed grid is always a set of integer multiples of `scale` (the
//! `zero_point` cancels in the round trip and only affects the asymmetric clamp
//! window). Hence the per-element error is bounded by `scale / 2` for
//! round-to-nearest, which the test-suite checks.
//!
//! ## fp8 quantization (`E4M3` / `E5M2`)
//!
//! Two 8-bit floating formats are modelled, decomposing each value into
//! sign / exponent / mantissa and rounding the mantissa to the available bits,
//! correctly handling **normals**, **subnormals** and **saturation**:
//!
//! | format | sign | exp | mantissa | bias | max-normal | min-normal | min-subnormal |
//! |--------|------|-----|----------|------|-----------:|-----------:|--------------:|
//! | E4M3   | 1    | 4   | 3        | 7    | `448`      | `2^-6`     | `2^-9`        |
//! | E5M2   | 1    | 5   | 2        | 15   | `57344`    | `2^-14`    | `2^-16`       |
//!
//! * **E4M3** follows the OCP / deep-learning "E4M3" variant: there are **no
//!   infinities**, the only NaN encoding is `S.1111.111`, and the largest finite
//!   value is `S.1111.110 = 1.75 * 2^8 =` [`E4M3_MAX_NORMAL`] `= 448`. Its
//!   dynamic range runs from the smallest subnormal `2^-9` up to `448`.
//! * **E5M2** is IEEE-like (it *has* `Inf`/`NaN` at exponent field `11111`); the
//!   largest finite value is `S.11110.11 = 1.75 * 2^15 =` [`E5M2_MAX_NORMAL`]
//!   `= 57344`, with dynamic range from `2^-16` up to `57344`.
//!
//! The cast implemented here is **saturating**: magnitudes above the format max
//! (and any infinities) clamp to the format max rather than overflowing to
//! `Inf`. `NaN` inputs propagate to `NaN`.
//!
//! The rounding uses a single unified rule that is continuous across the
//! normal/subnormal boundary. For a magnitude `a` with binade exponent
//! `e = floor(log2(a))`, the unit-in-the-last-place is `2^(max(e, emin) - mbits)`
//! where `emin = 1 - bias` is the smallest normal exponent and `mbits` is the
//! mantissa width; `a` is rounded to the nearest multiple of that ULP. For
//! `e >= emin` this reproduces the normal-number grid; for `e < emin` it freezes
//! at the subnormal granularity `2^(emin - mbits)`.
//!
//! ## Rounding modes
//!
//! [`RoundingMode`] selects between:
//!
//! * **Nearest** -- round-to-nearest-**even** (ties to even), the deterministic
//!   default.
//! * **Stochastic** -- round up with probability equal to the fractional part
//!   and down otherwise, drawing a uniform `[0, 1)` variate from the supplied
//!   [`Rng`]. Stochastic rounding is **unbiased**: `E[round(r)] = r`, so the
//!   expected reconstruction equals the true value; the test-suite verifies this
//!   by Monte-Carlo averaging.
//!
//! ## Straight-through estimator (STE)
//!
//! Because `quant` is piecewise constant its true derivative is zero almost
//! everywhere, which would block training. The STE replaces it with the identity
//! on the representable interval: the incoming gradient passes through unchanged
//! where the (pre-quant) value lies inside `[qmin_real, qmax_real]` and is zeroed
//! outside (the clamp saturates, so no gradient flows). See
//! [`fake_quant_backward`].
//!
//! ## QAT master-weight scheme
//!
//! [`QatOptimizer`] implements the standard QAT bookkeeping: a full-precision
//! **master copy** of the weights is what the optimizer (here an Adam/AdamW
//! update) actually integrates, while the **fake-quantized view** of those
//! master weights is what the forward pass "uses". On every [`QatOptimizer::step`]
//! the FP32 master is updated and the quantized view is re-derived (re-calibrated
//! and re-rounded) from the new master. Keeping the master in full precision is
//! essential: the tiny gradient steps would otherwise vanish under the
//! quantization rounding and the network would never learn.

use crate::GpuOptimError;
use scirs2_core::ndarray::{Array1, Array2, Axis, Zip};
use scirs2_core::random::{Rng, RngExt};

/// Largest finite (max-normal) magnitude of the E4M3 format, `1.75 * 2^8`.
pub const E4M3_MAX_NORMAL: f64 = 448.0;

/// Largest finite (max-normal) magnitude of the E5M2 format, `1.75 * 2^15`.
pub const E5M2_MAX_NORMAL: f64 = 57344.0;

/// Quantization grid geometry: symmetric (zero-point pinned to `0`) or affine
/// (asymmetric, zero-point derived from the real minimum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Symmetric, zero-point `= 0`, restricted signed range.
    Symmetric,
    /// Affine / asymmetric, zero-point derived from the real `[min, max]`.
    Affine,
}

/// Supported signed-integer quantization widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntDtype {
    /// 8-bit signed integer quantization.
    Int8,
    /// 4-bit signed integer quantization.
    Int4,
}

impl IntDtype {
    /// Number of bits in the integer code.
    pub fn bits(self) -> u32 {
        match self {
            IntDtype::Int8 => 8,
            IntDtype::Int4 => 4,
        }
    }

    /// Construct an [`IntDtype`] from a bit-width, rejecting unsupported widths.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::UnsupportedOperation`] for any width other than
    /// `4` or `8`.
    pub fn from_bits(bits: u32) -> Result<Self, GpuOptimError> {
        match bits {
            8 => Ok(IntDtype::Int8),
            4 => Ok(IntDtype::Int4),
            other => Err(GpuOptimError::UnsupportedOperation(format!(
                "unsupported integer quantization width: {other} bits (expected 4 or 8)"
            ))),
        }
    }

    /// Integer clamp range `[qmin, qmax]` for this width under the given scheme.
    ///
    /// Symmetric uses the restricted range (drops the most-negative code) so the
    /// grid is symmetric about zero; affine uses the full two's-complement range.
    pub fn q_range(self, scheme: QuantScheme) -> (i32, i32) {
        match (self, scheme) {
            (IntDtype::Int8, QuantScheme::Symmetric) => (-127, 127),
            (IntDtype::Int8, QuantScheme::Affine) => (-128, 127),
            (IntDtype::Int4, QuantScheme::Symmetric) => (-7, 7),
            (IntDtype::Int4, QuantScheme::Affine) => (-8, 7),
        }
    }
}

/// 8-bit floating-point formats modelled by [`fake_quant_fp8`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// 4 exponent bits, 3 mantissa bits, bias 7, max-normal `448` (no infinities).
    E4M3,
    /// 5 exponent bits, 2 mantissa bits, bias 15, max-normal `57344` (IEEE-like).
    E5M2,
}

impl Fp8Format {
    /// Number of explicit mantissa bits.
    pub fn mantissa_bits(self) -> i32 {
        match self {
            Fp8Format::E4M3 => 3,
            Fp8Format::E5M2 => 2,
        }
    }

    /// Exponent bias.
    pub fn exponent_bias(self) -> i32 {
        match self {
            Fp8Format::E4M3 => 7,
            Fp8Format::E5M2 => 15,
        }
    }

    /// Largest finite representable magnitude (the documented max-normal).
    pub fn max_normal(self) -> f64 {
        match self {
            Fp8Format::E4M3 => E4M3_MAX_NORMAL,
            Fp8Format::E5M2 => E5M2_MAX_NORMAL,
        }
    }

    /// Smallest normal magnitude, `2^(1 - bias)`.
    pub fn min_normal(self) -> f64 {
        pow2(1 - self.exponent_bias())
    }

    /// Smallest (subnormal) positive magnitude, `2^(1 - bias - mantissa_bits)`.
    pub fn min_subnormal(self) -> f64 {
        pow2(1 - self.exponent_bias() - self.mantissa_bits())
    }
}

/// Rounding rule applied when collapsing a real value onto the quantization grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (deterministic).
    Nearest,
    /// Stochastic rounding: round up with probability equal to the fractional
    /// part. Unbiased in expectation.
    Stochastic,
}

impl RoundingMode {
    /// Round a real value `x` to an integer according to this mode.
    ///
    /// For [`RoundingMode::Stochastic`] a uniform `[0, 1)` variate is drawn from
    /// `rng`; the result is `floor(x) + 1` with probability `x - floor(x)` and
    /// `floor(x)` otherwise, which makes the expectation exactly `x`.
    fn round_to_int(self, x: f64, rng: &mut impl Rng) -> f64 {
        match self {
            RoundingMode::Nearest => x.round_ties_even(),
            RoundingMode::Stochastic => {
                let lower = x.floor();
                let frac = x - lower;
                let draw: f64 = rng.random();
                if draw < frac {
                    lower + 1.0
                } else {
                    lower
                }
            }
        }
    }
}

/// Exact power of two `2^exp` for the small exponent range used by the fp8 grid.
///
/// `powi` multiplies/divides by two, each step exact in IEEE-754, so the result
/// is the exact power of two.
fn pow2(exp: i32) -> f64 {
    2.0_f64.powi(exp)
}

/// Affine/symmetric integer quantization parameters.
///
/// Holds the grid step (`scale`), the integer `zero_point` (always `0` for the
/// symmetric scheme) and the integer clamp range `[qmin, qmax]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    /// Grid step: the real spacing between adjacent integer codes.
    pub scale: f64,
    /// Integer code onto which the real value `0` maps.
    pub zero_point: i32,
    /// Lowest representable integer code.
    pub qmin: i32,
    /// Highest representable integer code.
    pub qmax: i32,
}

impl QuantParams {
    /// Build parameters directly from a real `[min, max]` interval.
    ///
    /// For [`QuantScheme::Symmetric`] this reduces to [`Self::from_absmax`] over
    /// `max(|min|, |max|)`. For [`QuantScheme::Affine`] the interval is nudged to
    /// include the real `0`, `scale = (max - min) / (qmax - qmin)` and the
    /// `zero_point` is chosen so the real `0` maps onto an exact integer code.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::InvalidState`] if the inputs are non-finite,
    /// inverted (`max < min`), or describe a degenerate zero-width interval.
    pub fn from_minmax(
        min_v: f64,
        max_v: f64,
        dtype: IntDtype,
        scheme: QuantScheme,
    ) -> Result<Self, GpuOptimError> {
        if !min_v.is_finite() || !max_v.is_finite() || max_v < min_v {
            return Err(GpuOptimError::InvalidState(format!(
                "invalid calibration interval: [{min_v}, {max_v}]"
            )));
        }
        match scheme {
            QuantScheme::Symmetric => Self::from_absmax(min_v.abs().max(max_v.abs()), dtype),
            QuantScheme::Affine => {
                let (qmin, qmax) = dtype.q_range(QuantScheme::Affine);
                // Nudge the interval so the real value 0 is representable.
                let rmin = min_v.min(0.0);
                let rmax = max_v.max(0.0);
                let range = rmax - rmin;
                if range <= 0.0 {
                    return Err(GpuOptimError::InvalidState(
                        "degenerate (zero-range) calibration: real min == max == 0".to_string(),
                    ));
                }
                let scale = range / f64::from(qmax - qmin);
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(GpuOptimError::InvalidState(format!(
                        "degenerate affine scale derived from interval [{min_v}, {max_v}]"
                    )));
                }
                let zp = (f64::from(qmin) - rmin / scale)
                    .round_ties_even()
                    .clamp(f64::from(qmin), f64::from(qmax));
                Ok(Self {
                    scale,
                    zero_point: zp as i32,
                    qmin,
                    qmax,
                })
            }
        }
    }

    /// Build **symmetric** parameters from an absolute maximum.
    ///
    /// `scale = absmax / qmax`, `zero_point = 0`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::InvalidState`] when `absmax` is non-finite or not
    /// strictly positive (a degenerate, all-zero tensor cannot be calibrated).
    pub fn from_absmax(absmax: f64, dtype: IntDtype) -> Result<Self, GpuOptimError> {
        let (qmin, qmax) = dtype.q_range(QuantScheme::Symmetric);
        if !absmax.is_finite() || absmax <= 0.0 {
            return Err(GpuOptimError::InvalidState(
                "degenerate (zero-range) calibration: absmax must be finite and > 0".to_string(),
            ));
        }
        let scale = absmax / f64::from(qmax);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(GpuOptimError::InvalidState(
                "degenerate symmetric scale".to_string(),
            ));
        }
        Ok(Self {
            scale,
            zero_point: 0,
            qmin,
            qmax,
        })
    }

    /// Calibrate per-tensor parameters from the exact `[min, max]` of `x`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::InvalidState`] for an empty tensor or a
    /// degenerate (zero-range) calibration.
    pub fn per_tensor_minmax(
        x: &Array1<f64>,
        dtype: IntDtype,
        scheme: QuantScheme,
    ) -> Result<Self, GpuOptimError> {
        let (min_v, max_v) = finite_min_max(x.iter().copied())?;
        Self::from_minmax(min_v, max_v, dtype, scheme)
    }

    /// Calibrate symmetric parameters from `max(|x|)`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::InvalidState`] for an empty or all-zero tensor.
    pub fn symmetric_from_absmax(x: &Array1<f64>, dtype: IntDtype) -> Result<Self, GpuOptimError> {
        if x.is_empty() {
            return Err(GpuOptimError::InvalidState(
                "cannot calibrate an empty tensor".to_string(),
            ));
        }
        let mut absmax = 0.0_f64;
        for &v in x.iter() {
            if !v.is_finite() {
                return Err(GpuOptimError::InvalidState(
                    "non-finite value in calibration tensor".to_string(),
                ));
            }
            absmax = absmax.max(v.abs());
        }
        Self::from_absmax(absmax, dtype)
    }

    /// Calibrate per-tensor parameters after clipping the tails.
    ///
    /// The interval is taken between the `clip_fraction` and `1 - clip_fraction`
    /// quantiles (nearest-rank) of `x`, which suppresses outliers before the
    /// scale is derived.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::InvalidState`] for an empty tensor, a
    /// `clip_fraction` outside `[0, 0.5)`, or a degenerate calibration.
    pub fn per_tensor_percentile(
        x: &Array1<f64>,
        dtype: IntDtype,
        scheme: QuantScheme,
        clip_fraction: f64,
    ) -> Result<Self, GpuOptimError> {
        if !(0.0..0.5).contains(&clip_fraction) {
            return Err(GpuOptimError::InvalidState(format!(
                "clip_fraction must lie in [0, 0.5), got {clip_fraction}"
            )));
        }
        if x.is_empty() {
            return Err(GpuOptimError::InvalidState(
                "cannot calibrate an empty tensor".to_string(),
            ));
        }
        let mut sorted: Vec<f64> = Vec::with_capacity(x.len());
        for &v in x.iter() {
            if !v.is_finite() {
                return Err(GpuOptimError::InvalidState(
                    "non-finite value in calibration tensor".to_string(),
                ));
            }
            sorted.push(v);
        }
        sorted.sort_by(|a, b| a.total_cmp(b));
        let last = sorted.len() - 1;
        let lo_idx = (clip_fraction * last as f64).floor() as usize;
        let hi_idx = ((1.0 - clip_fraction) * last as f64).ceil() as usize;
        let min_v = sorted[lo_idx.min(last)];
        let max_v = sorted[hi_idx.min(last)];
        Self::from_minmax(min_v, max_v, dtype, scheme)
    }

    /// Quantize a single real value to an integer code (with clamping).
    pub fn quantize(&self, x: f64, mode: RoundingMode, rng: &mut impl Rng) -> i32 {
        let scaled = x / self.scale + f64::from(self.zero_point);
        let rounded = mode
            .round_to_int(scaled, rng)
            .clamp(f64::from(self.qmin), f64::from(self.qmax));
        rounded as i32
    }

    /// Dequantize an integer code back to a real value.
    pub fn dequantize(&self, q: i32) -> f64 {
        f64::from(q - self.zero_point) * self.scale
    }

    /// Full fake-quant round trip for a single value, `dequant(quant(x))`.
    pub fn fake_quant_scalar(&self, x: f64, mode: RoundingMode, rng: &mut impl Rng) -> f64 {
        self.dequantize(self.quantize(x, mode, rng))
    }

    /// Lowest real value representable before the clamp saturates, `qmin*scale`
    /// after removing the zero-point.
    pub fn real_min(&self) -> f64 {
        self.dequantize(self.qmin)
    }

    /// Highest real value representable before the clamp saturates.
    pub fn real_max(&self) -> f64 {
        self.dequantize(self.qmax)
    }
}

/// Compute the finite `(min, max)` of an iterator of values.
fn finite_min_max(values: impl Iterator<Item = f64>) -> Result<(f64, f64), GpuOptimError> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut count = 0_usize;
    for v in values {
        if !v.is_finite() {
            return Err(GpuOptimError::InvalidState(
                "non-finite value in calibration tensor".to_string(),
            ));
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        count += 1;
    }
    if count == 0 {
        return Err(GpuOptimError::InvalidState(
            "cannot calibrate an empty tensor".to_string(),
        ));
    }
    Ok((min_v, max_v))
}

/// Fake-quantize a 1-D tensor through the integer grid (per-tensor).
///
/// Returns `dequant(quant(x))` element-wise; the output stays in `f64` but only
/// takes values on the reconstructed grid.
pub fn fake_quant_int(
    x: &Array1<f64>,
    params: &QuantParams,
    mode: RoundingMode,
    rng: &mut impl Rng,
) -> Array1<f64> {
    let mut out = Vec::with_capacity(x.len());
    for &v in x.iter() {
        out.push(params.fake_quant_scalar(v, mode, rng));
    }
    Array1::from_vec(out)
}

/// Calibrate one [`QuantParams`] per channel along `axis` of a 2-D tensor.
///
/// `axis == 0` treats each **row** as a channel, `axis == 1` each **column**.
/// Each channel receives its own scale (and, for affine, its own zero-point)
/// computed from that channel's slice only.
///
/// # Errors
///
/// Returns [`GpuOptimError::InvalidState`] for an invalid axis and propagates any
/// per-channel calibration error (e.g. a degenerate all-zero channel).
pub fn per_channel_params(
    x: &Array2<f64>,
    axis: usize,
    dtype: IntDtype,
    scheme: QuantScheme,
) -> Result<Vec<QuantParams>, GpuOptimError> {
    if axis > 1 {
        return Err(GpuOptimError::InvalidState(format!(
            "per-channel axis must be 0 or 1, got {axis}"
        )));
    }
    let n_channels = x.shape()[axis];
    let mut params = Vec::with_capacity(n_channels);
    for channel in 0..n_channels {
        let lane = x.index_axis(Axis(axis), channel);
        let (min_v, max_v) = finite_min_max(lane.iter().copied())?;
        params.push(QuantParams::from_minmax(min_v, max_v, dtype, scheme)?);
    }
    Ok(params)
}

/// Fake-quantize a 2-D tensor with one [`QuantParams`] per channel.
///
/// `params` must contain exactly one entry per channel along `axis` (as produced
/// by [`per_channel_params`]).
///
/// # Errors
///
/// Returns [`GpuOptimError::InvalidState`] for an invalid axis and
/// [`GpuOptimError::DimensionMismatch`] if `params.len()` does not match the
/// number of channels.
pub fn fake_quant_int_per_channel(
    x: &Array2<f64>,
    params: &[QuantParams],
    axis: usize,
    mode: RoundingMode,
    rng: &mut impl Rng,
) -> Result<Array2<f64>, GpuOptimError> {
    if axis > 1 {
        return Err(GpuOptimError::InvalidState(format!(
            "per-channel axis must be 0 or 1, got {axis}"
        )));
    }
    let n_channels = x.shape()[axis];
    if params.len() != n_channels {
        return Err(GpuOptimError::DimensionMismatch {
            expected: vec![n_channels],
            actual: vec![params.len()],
        });
    }
    let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for r in 0..n_rows {
        for c in 0..n_cols {
            let channel = if axis == 0 { r } else { c };
            let p = &params[channel];
            out[[r, c]] = p.fake_quant_scalar(x[[r, c]], mode, rng);
        }
    }
    Ok(out)
}

/// Quantize a single magnitude onto an fp8 grid (sign handled by the caller).
///
/// Implements the unified normal/subnormal rounding described in the module
/// documentation and saturates to `max_normal`.
fn fp8_quantize_magnitude(
    a: f64,
    mantissa_bits: i32,
    exponent_bias: i32,
    max_normal: f64,
    mode: RoundingMode,
    rng: &mut impl Rng,
) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    // Smallest normal exponent for the format.
    let emin = 1 - exponent_bias;
    // Binade exponent of `a` read directly from the IEEE-754 f64 bit pattern:
    // for a normal f64 in [2^e, 2^(e+1)) the biased exponent field equals
    // e + 1023, so this is an exact floor(log2(a)). Subnormal f64 inputs (field
    // 0) are far below the fp8 range and collapse to 0 below.
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let step_exp = e.max(emin) - mantissa_bits;
    let step = pow2(step_exp);
    let ratio = a / step;
    let rounded = mode.round_to_int(ratio, rng);
    let magnitude = rounded * step;
    if magnitude > max_normal {
        max_normal
    } else {
        magnitude
    }
}

/// Fake-quantize a 1-D tensor onto an fp8 (`E4M3` / `E5M2`) grid.
///
/// The cast is saturating: magnitudes above the format max (and infinities)
/// clamp to `±max_normal`; `NaN` inputs propagate to `NaN`. The sign of the input
/// (including signed zero) is preserved.
pub fn fake_quant_fp8(
    x: &Array1<f64>,
    format: Fp8Format,
    mode: RoundingMode,
    rng: &mut impl Rng,
) -> Array1<f64> {
    let mantissa_bits = format.mantissa_bits();
    let exponent_bias = format.exponent_bias();
    let max_normal = format.max_normal();
    let mut out = Vec::with_capacity(x.len());
    for &v in x.iter() {
        let q = if v.is_nan() {
            f64::NAN
        } else if v.is_infinite() {
            max_normal.copysign(v)
        } else {
            let magnitude = fp8_quantize_magnitude(
                v.abs(),
                mantissa_bits,
                exponent_bias,
                max_normal,
                mode,
                rng,
            );
            magnitude.copysign(v)
        };
        out.push(q);
    }
    Array1::from_vec(out)
}

/// Straight-through estimator backward for a fake-quant op.
///
/// The gradient passes through unchanged where the (pre-quant) value `x` lies in
/// the representable interval `[qmin_real, qmax_real]` and is zeroed where the
/// clamp saturates (no gradient flows through a saturated value). The bounds may
/// be supplied in either order.
///
/// # Errors
///
/// Returns [`GpuOptimError::DimensionMismatch`] if `grad` and `x` differ in length.
pub fn fake_quant_backward(
    grad: &Array1<f64>,
    x: &Array1<f64>,
    qmin_real: f64,
    qmax_real: f64,
) -> Result<Array1<f64>, GpuOptimError> {
    if grad.len() != x.len() {
        return Err(GpuOptimError::DimensionMismatch {
            expected: vec![x.len()],
            actual: vec![grad.len()],
        });
    }
    let (lo, hi) = if qmin_real <= qmax_real {
        (qmin_real, qmax_real)
    } else {
        (qmax_real, qmin_real)
    };
    let mut out = Vec::with_capacity(grad.len());
    for (&g, &v) in grad.iter().zip(x.iter()) {
        if v >= lo && v <= hi {
            out.push(g);
        } else {
            out.push(0.0);
        }
    }
    Ok(Array1::from_vec(out))
}

/// What a [`QatOptimizer`] quantizes its weights to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTarget {
    /// Integer quantization with the given width.
    Int(IntDtype),
    /// fp8 quantization with the given format.
    Fp8(Fp8Format),
}

/// Configuration for a [`QatOptimizer`]: the quantization target plus the
/// (Adam/AdamW) master-update hyper-parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QatConfig {
    /// Quantization target for the fake-quant view.
    pub target: QuantTarget,
    /// Integer scheme (ignored for fp8 targets).
    pub scheme: QuantScheme,
    /// Rounding mode applied when deriving the quantized view.
    pub rounding: RoundingMode,
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay (`beta1`).
    pub beta1: f64,
    /// Second-moment decay (`beta2`).
    pub beta2: f64,
    /// Numerical-stability epsilon.
    pub eps: f64,
    /// Decoupled (AdamW) weight decay.
    pub weight_decay: f64,
}

impl QatConfig {
    /// Construct a config with standard Adam defaults
    /// (`beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`, `weight_decay = 0`).
    pub fn new(target: QuantTarget, scheme: QuantScheme, rounding: RoundingMode, lr: f64) -> Self {
        Self {
            target,
            scheme,
            rounding,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// QAT optimizer wrapper maintaining full-precision master weights.
///
/// The optimizer owns the Adam moment state and a cached **fake-quantized view**
/// of the master weights, but the FP32 master itself is owned by the caller and
/// passed into [`Self::step`]. Each step updates the master in place and
/// re-derives the quantized view from it.
#[derive(Debug, Clone)]
pub struct QatOptimizer {
    config: QatConfig,
    first_moment: Array1<f64>,
    second_moment: Array1<f64>,
    step_count: u64,
    quantized: Array1<f64>,
    int_params: Option<QuantParams>,
}

impl QatOptimizer {
    /// Create an optimizer for the given FP32 `master` weights.
    ///
    /// The initial quantized view is derived immediately from `master`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::InvalidState`] for empty master weights and
    /// propagates any calibration error from the initial quantization.
    pub fn new(
        master: &Array1<f64>,
        config: QatConfig,
        rng: &mut impl Rng,
    ) -> Result<Self, GpuOptimError> {
        if master.is_empty() {
            return Err(GpuOptimError::InvalidState(
                "cannot construct a QatOptimizer over empty master weights".to_string(),
            ));
        }
        let n = master.len();
        let mut optimizer = Self {
            config,
            first_moment: Array1::zeros(n),
            second_moment: Array1::zeros(n),
            step_count: 0,
            quantized: Array1::zeros(n),
            int_params: None,
        };
        optimizer.requantize(master, rng)?;
        Ok(optimizer)
    }

    /// The current fake-quantized view of the master weights (what the forward
    /// pass uses).
    pub fn quantized_weights(&self) -> &Array1<f64> {
        &self.quantized
    }

    /// The integer quantization parameters currently in effect, if the target is
    /// integer (always `None` for fp8 targets).
    pub fn quant_params(&self) -> Option<&QuantParams> {
        self.int_params.as_ref()
    }

    /// Number of [`Self::step`] calls performed so far.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Perform one optimizer step: an Adam/AdamW update of the FP32 `master`
    /// followed by re-derivation of the quantized view.
    ///
    /// The `master` weights stay in full precision; only the cached quantized
    /// view (see [`Self::quantized_weights`]) is rounded onto the grid.
    ///
    /// # Errors
    ///
    /// Returns [`GpuOptimError::DimensionMismatch`] if `master` and `grad` (or the
    /// optimizer's internal state) disagree on length, and propagates any
    /// re-calibration error.
    pub fn step(
        &mut self,
        master: &mut Array1<f64>,
        grad: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> Result<(), GpuOptimError> {
        if master.len() != grad.len() {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![master.len()],
                actual: vec![grad.len()],
            });
        }
        if master.len() != self.first_moment.len() {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![self.first_moment.len()],
                actual: vec![master.len()],
            });
        }

        self.step_count += 1;
        let t = self.step_count as i32;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.lr;
        let eps = self.config.eps;
        let weight_decay = self.config.weight_decay;
        let bias_correction1 = 1.0 - beta1.powi(t);
        let bias_correction2 = 1.0 - beta2.powi(t);

        Zip::from(&mut *master)
            .and(grad)
            .and(&mut self.first_moment)
            .and(&mut self.second_moment)
            .for_each(|weight, &g, m, v| {
                *m = beta1 * *m + (1.0 - beta1) * g;
                *v = beta2 * *v + (1.0 - beta2) * g * g;
                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;
                // Decoupled (AdamW) weight decay applied to the FP32 master.
                if weight_decay != 0.0 {
                    *weight -= lr * weight_decay * *weight;
                }
                *weight -= lr * m_hat / (v_hat.sqrt() + eps);
            });

        self.requantize(master, rng)?;
        Ok(())
    }

    /// Re-derive the cached quantized view from the current master weights.
    fn requantize(
        &mut self,
        master: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> Result<(), GpuOptimError> {
        match self.config.target {
            QuantTarget::Int(dtype) => {
                let params = QuantParams::per_tensor_minmax(master, dtype, self.config.scheme)?;
                self.quantized = fake_quant_int(master, &params, self.config.rounding, rng);
                self.int_params = Some(params);
            }
            QuantTarget::Fp8(format) => {
                self.quantized = fake_quant_fp8(master, format, self.config.rounding, rng);
                self.int_params = None;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::Random;

    const EPS: f64 = 1e-9;

    fn seeded(seed: u64) -> Random<scirs2_core::random::rngs::StdRng> {
        Random::seed(seed)
    }

    #[test]
    fn int8_grid_values_quantize_to_themselves() {
        // scale = 12.7 / 127 = 0.1, zero_point = 0.
        let params = QuantParams::from_absmax(12.7, IntDtype::Int8).expect("calibrate");
        assert!((params.scale - 0.1).abs() < EPS);
        assert_eq!(params.zero_point, 0);
        let mut rng = seeded(1);
        // Values exactly on the grid (multiples of scale, inside the range).
        for k in -120..=120 {
            let on_grid = k as f64 * params.scale;
            let round_trip = params.fake_quant_scalar(on_grid, RoundingMode::Nearest, &mut rng);
            assert!(
                (round_trip - on_grid).abs() < EPS,
                "grid value {on_grid} did not map to itself (got {round_trip})"
            );
        }
    }

    #[test]
    fn int8_affine_grid_values_quantize_to_themselves() {
        let data = Array1::from_vec(vec![-0.3, 1.7, 0.0, 0.9, -0.1]);
        let params = QuantParams::per_tensor_minmax(&data, IntDtype::Int8, QuantScheme::Affine)
            .expect("cal");
        let mut rng = seeded(7);
        // (q - zero_point) * scale is on the reconstructed grid for any code q.
        for q in params.qmin..=params.qmax {
            let on_grid = params.dequantize(q);
            let round_trip = params.fake_quant_scalar(on_grid, RoundingMode::Nearest, &mut rng);
            assert!(
                (round_trip - on_grid).abs() < 1e-9,
                "affine grid value {on_grid} (q={q}) -> {round_trip}"
            );
        }
    }

    #[test]
    fn int8_nearest_error_bounded_by_half_scale() {
        let params = QuantParams::from_absmax(2.0, IntDtype::Int8).expect("calibrate");
        let half = params.scale / 2.0;
        let mut rng = seeded(2);
        // Sweep values strictly inside the representable range.
        let mut x = -1.9;
        while x <= 1.9 {
            let fq = params.fake_quant_scalar(x, RoundingMode::Nearest, &mut rng);
            assert!(
                (fq - x).abs() <= half + EPS,
                "nearest error {} exceeded scale/2 = {half} at x = {x}",
                (fq - x).abs()
            );
            x += 0.013;
        }
    }

    #[test]
    fn int4_round_trip_on_grid() {
        let params = QuantParams::from_absmax(7.0, IntDtype::Int4).expect("calibrate");
        // scale = 7 / 7 = 1.0.
        assert!((params.scale - 1.0).abs() < EPS);
        let mut rng = seeded(3);
        for k in -7..=7 {
            let on_grid = k as f64;
            let fq = params.fake_quant_scalar(on_grid, RoundingMode::Nearest, &mut rng);
            assert!((fq - on_grid).abs() < EPS, "int4 grid {on_grid} -> {fq}");
        }
    }

    #[test]
    fn stochastic_rounding_is_unbiased() {
        let params = QuantParams::from_absmax(12.7, IntDtype::Int8).expect("calibrate");
        let scale = params.scale; // 0.1
                                  // A value sitting 70% of the way between two grid points.
        let value = 3.0 + 0.7 * scale;
        let n: usize = 400_000;
        let mut rng = seeded(12345);
        let mut sum = 0.0_f64;
        let mut saw_lower = false;
        let mut saw_upper = false;
        let lower = 3.0;
        let upper = 3.0 + scale;
        for _ in 0..n {
            let q = params.fake_quant_scalar(value, RoundingMode::Stochastic, &mut rng);
            // Stochastic rounding only ever produces the two bracketing levels.
            assert!(
                (q - lower).abs() < 1e-9 || (q - upper).abs() < 1e-9,
                "stochastic output {q} was not a bracketing grid level"
            );
            if (q - lower).abs() < 1e-9 {
                saw_lower = true;
            }
            if (q - upper).abs() < 1e-9 {
                saw_upper = true;
            }
            sum += q;
        }
        let mean = sum / n as f64;
        // Standard error of the mean is <= scale / (2 sqrt(n)); allow ~8 sigma.
        let tolerance = 8.0 * scale / (n as f64).sqrt();
        assert!(
            (mean - value).abs() < tolerance,
            "stochastic mean {mean} deviated from {value} by more than {tolerance}"
        );
        assert!(saw_lower && saw_upper, "expected both rounding directions");
    }

    #[test]
    fn stochastic_rounding_exact_grid_value_is_stable() {
        let params = QuantParams::from_absmax(12.7, IntDtype::Int8).expect("calibrate");
        let mut rng = seeded(99);
        let exact = 5.0 * params.scale; // exactly on the grid -> fractional part 0
        for _ in 0..1000 {
            let q = params.fake_quant_scalar(exact, RoundingMode::Stochastic, &mut rng);
            assert!((q - exact).abs() < EPS, "exact grid value drifted: {q}");
        }
    }

    #[test]
    fn fp8_constants_match_documentation() {
        assert_eq!(E4M3_MAX_NORMAL, 448.0);
        assert_eq!(E5M2_MAX_NORMAL, 57344.0);
        assert_eq!(Fp8Format::E4M3.max_normal(), 448.0);
        assert_eq!(Fp8Format::E5M2.max_normal(), 57344.0);
        assert!((Fp8Format::E4M3.min_normal() - 2.0_f64.powi(-6)).abs() < EPS);
        assert!((Fp8Format::E4M3.min_subnormal() - 2.0_f64.powi(-9)).abs() < EPS);
        assert!((Fp8Format::E5M2.min_normal() - 2.0_f64.powi(-14)).abs() < EPS);
        assert!((Fp8Format::E5M2.min_subnormal() - 2.0_f64.powi(-16)).abs() < EPS);
    }

    #[test]
    fn fp8_e4m3_representable_values_map_to_themselves() {
        let mut rng = seeded(4);
        let representable = [
            0.0,
            1.0,
            1.5,  // 1 + 4/8
            1.75, // 1 + 6/8
            2.0,
            -2.0,
            0.5,
            256.0,
            448.0, // max-normal
            -448.0,
            2.0_f64.powi(-6), // min-normal
            2.0_f64.powi(-9), // min-subnormal
        ];
        let input = Array1::from_vec(representable.to_vec());
        let out = fake_quant_fp8(&input, Fp8Format::E4M3, RoundingMode::Nearest, &mut rng);
        for (i, (&want, &got)) in representable.iter().zip(out.iter()).enumerate() {
            assert!(
                (want - got).abs() < EPS,
                "E4M3 representable[{i}] = {want} mapped to {got}"
            );
        }
    }

    #[test]
    fn fp8_e4m3_saturates_to_max_normal() {
        let mut rng = seeded(5);
        let input = Array1::from_vec(vec![449.0, 1000.0, 1.0e6, -1000.0, f64::INFINITY]);
        let out = fake_quant_fp8(&input, Fp8Format::E4M3, RoundingMode::Nearest, &mut rng);
        assert_eq!(out[0], 448.0);
        assert_eq!(out[1], 448.0);
        assert_eq!(out[2], 448.0);
        assert_eq!(out[3], -448.0);
        assert_eq!(out[4], 448.0);
    }

    #[test]
    fn fp8_e4m3_nan_propagates() {
        let mut rng = seeded(6);
        let input = Array1::from_vec(vec![f64::NAN]);
        let out = fake_quant_fp8(&input, Fp8Format::E4M3, RoundingMode::Nearest, &mut rng);
        assert!(out[0].is_nan());
    }

    #[test]
    fn fp8_e4m3_rounds_to_nearest_grid_within_half_ulp() {
        let mut rng = seeded(8);
        // Around 1.0 the E4M3 step is 2^(0-3) = 0.125, grid: 1.0, 1.125, 1.25, ...
        let input = Array1::from_vec(vec![1.1]);
        let out = fake_quant_fp8(&input, Fp8Format::E4M3, RoundingMode::Nearest, &mut rng);
        assert!((out[0] - 1.125).abs() < EPS, "1.1 -> {}", out[0]);
        assert!((out[0] - 1.1).abs() <= 0.125 / 2.0 + EPS);
    }

    #[test]
    fn fp8_e5m2_representable_values_map_to_themselves() {
        let mut rng = seeded(9);
        let representable = [
            0.0,
            1.0,
            1.5, // 1 + 2/4
            2.0,
            -4.0,
            57344.0, // max-normal
            -57344.0,
            2.0_f64.powi(-14), // min-normal
            2.0_f64.powi(-16), // min-subnormal
        ];
        let input = Array1::from_vec(representable.to_vec());
        let out = fake_quant_fp8(&input, Fp8Format::E5M2, RoundingMode::Nearest, &mut rng);
        for (i, (&want, &got)) in representable.iter().zip(out.iter()).enumerate() {
            assert!(
                (want - got).abs() < EPS,
                "E5M2 representable[{i}] = {want} mapped to {got}"
            );
        }
    }

    #[test]
    fn fp8_e5m2_saturates_to_max_normal() {
        let mut rng = seeded(10);
        let input = Array1::from_vec(vec![60000.0, 1.0e8, -70000.0, f64::INFINITY]);
        let out = fake_quant_fp8(&input, Fp8Format::E5M2, RoundingMode::Nearest, &mut rng);
        assert_eq!(out[0], 57344.0);
        assert_eq!(out[1], 57344.0);
        assert_eq!(out[2], -57344.0);
        assert_eq!(out[3], 57344.0);
    }

    #[test]
    fn per_channel_scales_differ_and_error_respects_each_scale() {
        // Row 0 is a small-magnitude channel, row 1 a large-magnitude channel.
        let x =
            Array2::from_shape_vec((2, 3), vec![0.0, 0.5, 1.0, 0.0, 50.0, 100.0]).expect("shape");
        let params =
            per_channel_params(&x, 0, IntDtype::Int8, QuantScheme::Symmetric).expect("cal");
        assert_eq!(params.len(), 2);
        // Channel scales must differ and track each channel's magnitude.
        assert!(params[1].scale > params[0].scale * 10.0);
        assert!((params[0].scale - 1.0 / 127.0).abs() < 1e-6);
        assert!((params[1].scale - 100.0 / 127.0).abs() < 1e-6);

        let mut rng = seeded(11);
        let out = fake_quant_int_per_channel(&x, &params, 0, RoundingMode::Nearest, &mut rng)
            .expect("quantize");
        for r in 0..2 {
            let half = params[r].scale / 2.0;
            for c in 0..3 {
                let err = (out[[r, c]] - x[[r, c]]).abs();
                assert!(
                    err <= half + EPS,
                    "per-channel error {err} exceeded scale/2={half} at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn per_channel_length_mismatch_errors() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
        let params = per_channel_params(&x, 0, IntDtype::Int8, QuantScheme::Symmetric).expect("c");
        let mut rng = seeded(13);
        // Only one param for a two-channel tensor -> dimension mismatch.
        let result =
            fake_quant_int_per_channel(&x, &params[..1], 0, RoundingMode::Nearest, &mut rng);
        assert!(matches!(
            result,
            Err(GpuOptimError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn ste_passes_gradient_in_range_and_zeros_out_of_range() {
        let x = Array1::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0]);
        let grad = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let out = fake_quant_backward(&grad, &x, -1.0, 1.0).expect("ste");
        let expected = [0.0, 1.0, 1.0, 1.0, 0.0];
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < EPS, "STE[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn ste_dimension_mismatch_errors() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grad = Array1::from_vec(vec![1.0, 1.0]);
        let result = fake_quant_backward(&grad, &x, -1.0, 1.0);
        assert!(matches!(
            result,
            Err(GpuOptimError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn degenerate_calibration_errors() {
        let zeros = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        assert!(QuantParams::symmetric_from_absmax(&zeros, IntDtype::Int8).is_err());
        assert!(
            QuantParams::per_tensor_minmax(&zeros, IntDtype::Int8, QuantScheme::Affine).is_err()
        );
    }

    #[test]
    fn invalid_bit_width_errors() {
        assert!(IntDtype::from_bits(8).is_ok());
        assert!(IntDtype::from_bits(4).is_ok());
        assert!(matches!(
            IntDtype::from_bits(3),
            Err(GpuOptimError::UnsupportedOperation(_))
        ));
        assert!(IntDtype::from_bits(16).is_err());
    }

    #[test]
    fn qat_master_update_keeps_fp32_master_and_quantized_view_tracks_it() {
        let mut master = Array1::from_vec(vec![0.12, -0.37, 0.88, -0.05, 0.51]);
        let master_before = master.clone();
        let config = QatConfig::new(
            QuantTarget::Int(IntDtype::Int8),
            QuantScheme::Symmetric,
            RoundingMode::Nearest,
            0.1,
        );
        let mut rng = seeded(2024);
        let mut optimizer = QatOptimizer::new(&master, config, &mut rng).expect("new");

        let grad = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.15]);
        optimizer.step(&mut master, &grad, &mut rng).expect("step");

        // (1) The FP32 master actually moved.
        let mut moved = false;
        for (&a, &b) in master.iter().zip(master_before.iter()) {
            if (a - b).abs() > EPS {
                moved = true;
            }
        }
        assert!(moved, "Adam update did not change the master weights");

        // (2) The quantized view equals an independent fake-quant of the master.
        let params = optimizer.quant_params().expect("int params").to_owned();
        let mut check_rng = seeded(2024);
        let reference = fake_quant_int(&master, &params, RoundingMode::Nearest, &mut check_rng);
        for (&q, &r) in optimizer.quantized_weights().iter().zip(reference.iter()) {
            assert!((q - r).abs() < EPS, "quantized view does not track master");
        }

        // (3) The quantized view lies exactly on the grid (integer multiples of
        //     scale) while the FP32 master is genuinely full-precision (at least
        //     one master weight is off-grid).
        let scale = params.scale;
        for &q in optimizer.quantized_weights().iter() {
            let codes = q / scale;
            assert!(
                (codes - codes.round()).abs() < 1e-6,
                "quantized weight {q} is not on the grid"
            );
        }
        let mut some_off_grid = false;
        for &w in master.iter() {
            let codes = w / scale;
            if (codes - codes.round()).abs() > 1e-6 {
                some_off_grid = true;
            }
        }
        assert!(
            some_off_grid,
            "master weights appear to be quantized (not full precision)"
        );

        assert_eq!(optimizer.step_count(), 1);
    }

    #[test]
    fn qat_fp8_target_tracks_master() {
        let mut master = Array1::from_vec(vec![0.3, -1.2, 4.0, -0.01, 2.5]);
        let config = QatConfig::new(
            QuantTarget::Fp8(Fp8Format::E4M3),
            QuantScheme::Symmetric,
            RoundingMode::Nearest,
            0.05,
        );
        let mut rng = seeded(77);
        let mut optimizer = QatOptimizer::new(&master, config, &mut rng).expect("new");
        assert!(optimizer.quant_params().is_none());

        let grad = Array1::from_vec(vec![0.2, 0.1, -0.3, 0.4, -0.05]);
        optimizer.step(&mut master, &grad, &mut rng).expect("step");

        let mut check_rng = seeded(77);
        // Re-derive the fp8 view from the (already updated) master and compare.
        let reference = fake_quant_fp8(
            &master,
            Fp8Format::E4M3,
            RoundingMode::Nearest,
            &mut check_rng,
        );
        for (&q, &r) in optimizer.quantized_weights().iter().zip(reference.iter()) {
            assert!(
                (q - r).abs() < EPS,
                "fp8 quantized view does not track master"
            );
        }
    }

    #[test]
    fn percentile_calibration_clips_outliers() {
        // One large outlier should be clipped away, yielding a smaller scale than
        // plain min/max calibration.
        let mut values = vec![0.0; 100];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f64) / 100.0; // 0.0 .. 0.99
        }
        values.push(1000.0); // outlier
        let data = Array1::from_vec(values);
        let plain = QuantParams::symmetric_from_absmax(&data, IntDtype::Int8).expect("plain");
        let clipped =
            QuantParams::per_tensor_percentile(&data, IntDtype::Int8, QuantScheme::Symmetric, 0.02)
                .expect("clipped");
        assert!(
            clipped.scale < plain.scale,
            "percentile clipping did not reduce the scale ({} vs {})",
            clipped.scale,
            plain.scale
        );
    }
}
