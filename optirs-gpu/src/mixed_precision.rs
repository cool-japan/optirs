//! Mixed-precision (AMP) building blocks: IEEE-754 binary16 conversion and
//! dynamic loss scaling.
//!
//! This module is pure Rust and has no device dependency, so it is usable from
//! CPU code paths and from the GPU optimizer path alike. The conversions are
//! full IEEE-754 `binary16` implementations — subnormals, infinities, NaN
//! payload preservation and round-half-to-even are all handled — not the
//! truncating placeholder they replace.

/// Configuration for dynamic loss scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedPrecisionConfig {
    /// Initial loss scale factor.
    pub init_scale: f32,
    /// Multiplier applied when growing the scale.
    pub growth_factor: f32,
    /// Multiplier applied when an overflow is observed.
    pub backoff_factor: f32,
    /// Number of consecutive overflow-free steps before growing.
    pub growth_interval: u32,
    /// Lower clamp for the scale.
    pub min_scale: f32,
    /// Upper clamp for the scale.
    pub max_scale: f32,
    /// Use `bfloat16` rather than `float16` for the reduced-precision copy.
    pub use_bfloat16: bool,
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
            use_bfloat16: false,
        }
    }
}

/// Aggregate overflow statistics over the scaler's rolling window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverflowStats {
    /// Steps recorded in the rolling window.
    pub total_steps: usize,
    /// Overflowing steps in the rolling window.
    pub overflow_count: usize,
    /// `overflow_count / total_steps`, or `0.0` for an empty window.
    pub overflow_rate: f32,
    /// Scale currently in effect.
    pub current_scale: f32,
}

/// Dynamic loss scaler with the standard grow/back-off schedule.
#[derive(Debug, Clone)]
pub struct DynamicLossScaler {
    scale: f32,
    config: MixedPrecisionConfig,
    growth_tracker: u32,
    window: Vec<bool>,
}

impl DynamicLossScaler {
    /// Rolling window length used for [`Self::overflow_stats`].
    pub const WINDOW: usize = 100;

    /// Create a scaler from a configuration.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            scale: config.init_scale.clamp(config.min_scale, config.max_scale),
            config,
            growth_tracker: 0,
            window: Vec::with_capacity(Self::WINDOW),
        }
    }

    /// Scale currently in effect.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Reciprocal of the current scale, for unscaling gradients.
    pub fn inv_scale(&self) -> f32 {
        1.0 / self.scale
    }

    /// Record the outcome of one step and update the scale.
    pub fn update(&mut self, has_overflow: bool) {
        if self.window.len() == Self::WINDOW {
            self.window.remove(0);
        }
        self.window.push(has_overflow);

        if has_overflow {
            self.scale = (self.scale * self.config.backoff_factor).max(self.config.min_scale);
            self.growth_tracker = 0;
        } else {
            self.growth_tracker = self.growth_tracker.saturating_add(1);
            if self.growth_tracker >= self.config.growth_interval {
                self.scale = (self.scale * self.config.growth_factor).min(self.config.max_scale);
                self.growth_tracker = 0;
            }
        }
    }

    /// Statistics over the rolling window.
    pub fn overflow_stats(&self) -> OverflowStats {
        let total = self.window.len();
        let overflows = self.window.iter().filter(|&&x| x).count();
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

    /// Divide `values` by the current scale, reporting whether the *scaled*
    /// input contained a non-finite entry.
    ///
    /// This is the standard AMP overflow test: an `inf`/`NaN` produced by the
    /// scaled backward pass means the step must be skipped and the scale cut.
    pub fn unscale_and_check(&mut self, values: &mut [f32]) -> bool {
        let inv = self.inv_scale();
        let mut overflow = false;
        for v in values.iter_mut() {
            if !v.is_finite() {
                overflow = true;
            }
            *v *= inv;
        }
        self.update(overflow);
        overflow
    }
}

/// Convert an `f32` to IEEE-754 `binary16` bits with round-half-to-even.
///
/// Overflow saturates to the signed infinity of `binary16` (the same behaviour
/// as hardware `f32 → f16` conversion); subnormal results are produced
/// correctly rather than flushed to zero; NaN stays NaN with a non-zero
/// mantissa.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        // Inf or NaN.
        return if mantissa == 0 {
            sign | 0x7c00
        } else {
            // Preserve NaN-ness; keep the top mantissa bits and force non-zero.
            sign | 0x7c00 | ((mantissa >> 13) as u16) | 0x0200
        };
    }

    // Unbiased exponent, then rebias for binary16.
    let unbiased = exponent - 127;
    let half_exp = unbiased + 15;

    if half_exp >= 0x1f {
        // Overflow → infinity.
        return sign | 0x7c00;
    }

    if half_exp <= 0 {
        // Subnormal (or underflow to zero). Reintroduce the implicit bit and
        // shift the significand into the subnormal range.
        if half_exp < -10 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - half_exp) as u32; // 14 = 23 - 10 + 1
        let result = significand >> shift;
        // Round half to even using the bits shifted out.
        let round_bit = 1u32 << (shift - 1);
        let remainder = significand & (round_bit.saturating_mul(2) - 1);
        let mut half = result as u16;
        if remainder > round_bit || (remainder == round_bit && (result & 1) == 1) {
            half = half.wrapping_add(1);
        }
        return sign | half;
    }

    // Normal range.
    let mut half = ((half_exp as u16) << 10) | ((mantissa >> 13) as u16);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1) {
        // Carrying into the exponent is handled naturally by the addition:
        // a mantissa of all ones rolls over into the exponent field, and an
        // exponent of 0x1e rolling over yields exactly the infinity pattern.
        half = half.wrapping_add(1);
    }
    sign | half
}

/// Convert IEEE-754 `binary16` bits to `f32`. Exact for every input.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = ((bits & 0x03ff) as u32) << 13;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: shift the significand left until the implicit bit lands
        // on bit 23. A binary16 subnormal is `m * 2^-24` with `m` in
        // `[1, 1023]`; after `k` normalising shifts the value is
        // `1.f * 2^(-14 - k)`, i.e. a biased f32 exponent of `113 - k`.
        let mut mant = mantissa;
        let mut shifts: i32 = 0;
        while mant & 0x0080_0000 == 0 {
            mant <<= 1;
            shifts += 1;
        }
        mant &= 0x007f_ffff;
        let f32_exp = ((113 - shifts) as u32) << 23;
        return f32::from_bits(sign | f32_exp | mant);
    }

    if exponent == 0x1f {
        // Inf / NaN.
        return f32::from_bits(sign | 0x7f80_0000 | mantissa);
    }

    let f32_exp = (exponent + (127 - 15)) << 23;
    f32::from_bits(sign | f32_exp | mantissa)
}

/// Convert a slice of `f32` to `binary16` bit patterns.
pub fn f32_slice_to_f16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}

/// Convert a slice of `binary16` bit patterns back to `f32`.
pub fn f16_bits_slice_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().copied().map(f16_bits_to_f32).collect()
}

/// Largest finite magnitude representable in `binary16`.
pub const F16_MAX: f32 = 65504.0;

/// Clamp to the `binary16` finite range before conversion.
pub fn saturate_to_f16_range(value: f32) -> f32 {
    if value.is_nan() {
        value
    } else {
        value.clamp(-F16_MAX, F16_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_exact_values() {
        for &v in &[
            0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, 65504.0, -65504.0, 0.125, 1024.0,
        ] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert_eq!(back.to_bits(), v.to_bits(), "value {v} did not round-trip");
        }
    }

    #[test]
    fn round_trips_every_finite_f16() {
        // Property: f16 -> f32 -> f16 is the identity on all 65536 patterns.
        for bits in 0u16..=u16::MAX {
            let exponent = (bits >> 10) & 0x1f;
            let value = f16_bits_to_f32(bits);
            if exponent == 0x1f {
                // Inf/NaN class: only require the class to be preserved.
                let back = f32_to_f16_bits(value);
                assert_eq!(
                    (back >> 10) & 0x1f,
                    0x1f,
                    "bits {bits:#06x} lost its inf/NaN class"
                );
                assert_eq!(back & 0x8000, bits & 0x8000, "bits {bits:#06x} lost sign");
                continue;
            }
            let back = f32_to_f16_bits(value);
            assert_eq!(
                back, bits,
                "bits {bits:#06x} did not round-trip (f32 {value})"
            );
        }
    }

    #[test]
    fn subnormals_are_not_flushed_to_zero() {
        // Smallest positive f16 subnormal is 2^-24.
        let smallest = f16_bits_to_f32(1);
        assert!(smallest > 0.0);
        assert!((smallest - 2f32.powi(-24)).abs() < f32::EPSILON * smallest);
        assert_eq!(f32_to_f16_bits(smallest), 1);
    }

    #[test]
    fn rounds_half_to_even() {
        // 1.0 + 2^-11 lies exactly halfway between 1.0 (even mantissa) and the
        // next f16; round-half-to-even must pick 1.0.
        let halfway = 1.0f32 + 2f32.powi(-11);
        assert_eq!(f32_to_f16_bits(halfway), f32_to_f16_bits(1.0));
        // 1.0 + 3 * 2^-11 lies halfway between the first and second f16 above
        // 1.0; the even neighbour is the second one.
        let halfway_up = 1.0f32 + 3.0 * 2f32.powi(-11);
        assert_eq!(f32_to_f16_bits(halfway_up), f32_to_f16_bits(1.0) + 2);
    }

    #[test]
    fn overflow_saturates_to_infinity() {
        assert_eq!(f32_to_f16_bits(1.0e30), 0x7c00);
        assert_eq!(f32_to_f16_bits(-1.0e30), 0xfc00);
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn nan_stays_nan() {
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn loss_scaler_backs_off_and_grows() {
        let config = MixedPrecisionConfig {
            growth_interval: 4,
            ..MixedPrecisionConfig::default()
        };
        let mut scaler = DynamicLossScaler::new(config);
        assert_eq!(scaler.scale(), 65536.0);

        scaler.update(true);
        assert_eq!(scaler.scale(), 32768.0);

        for _ in 0..4 {
            scaler.update(false);
        }
        assert_eq!(scaler.scale(), 65536.0);
    }

    #[test]
    fn loss_scaler_detects_overflow_while_unscaling() {
        let mut scaler = DynamicLossScaler::new(MixedPrecisionConfig {
            init_scale: 4.0,
            ..MixedPrecisionConfig::default()
        });
        let mut grads = [8.0f32, -4.0, 2.0];
        assert!(!scaler.unscale_and_check(&mut grads));
        assert_eq!(grads, [2.0, -1.0, 0.5]);
        assert_eq!(scaler.scale(), 4.0);

        let mut bad = [f32::INFINITY, 1.0];
        assert!(scaler.unscale_and_check(&mut bad));
        assert_eq!(scaler.scale(), 2.0);

        let stats = scaler.overflow_stats();
        assert_eq!(stats.total_steps, 2);
        assert_eq!(stats.overflow_count, 1);
    }

    #[test]
    fn saturation_keeps_values_finite() {
        assert_eq!(saturate_to_f16_range(1.0e30), F16_MAX);
        assert_eq!(saturate_to_f16_range(-1.0e30), -F16_MAX);
        assert!(saturate_to_f16_range(f32::NAN).is_nan());
    }
}
