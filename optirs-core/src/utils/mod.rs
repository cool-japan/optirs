// Utility functions for machine learning optimization
//
// This module provides utility functions and helpers for optimization
// tasks in machine learning.

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::{Float, ToPrimitive};
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Convert an `f64` value into the generic float type `A`, returning an honest
/// error when `A` cannot represent it.
///
/// Use this at call sites that already return [`Result`]: it replaces the
/// `A::from(x).expect("unwrap failed")` pattern with real error propagation, so
/// a value outside `A`'s range surfaces as an `Err` instead of a panic.
///
/// For the infallible counterpart — constructors, `Default` impls and struct
/// literals, which cannot propagate an error — use [`scalar_or`].
///
/// # Examples
///
/// ```
/// use optirs_core::utils::try_scalar;
///
/// let half: f32 = try_scalar(0.5).expect("0.5 is representable as f32");
/// assert_eq!(half, 0.5f32);
///
/// // Note what the error path does *not* cover: a float-to-float conversion
/// // saturates rather than failing, so an out-of-range `f64` becomes an
/// // infinity in `f32`, not an `Err`. The `Err` arm is a defensive guard for
/// // conversions that genuinely have no image, not an `f32` range check.
/// assert_eq!(
///     try_scalar::<f32, _>(f64::MAX).expect("f64 -> f32 saturates instead of failing"),
///     f32::INFINITY
/// );
/// ```
#[inline]
pub fn try_scalar<A: Float, V: ToPrimitive + Copy>(value: V) -> Result<A> {
    A::from(value).ok_or_else(|| OptimError::InvalidParameter(unrepresentable(value)))
}

/// Convert a generic float `A` down into `f64`, returning an honest error when
/// the value has no `f64` representation.
///
/// This is the mirror image of [`try_scalar`]: `try_scalar` widens a concrete
/// literal into the generic parameter type, `try_f64` narrows a generic value
/// back to `f64` for accumulators, metrics and reporting that are natively
/// `f64`. It replaces the `x.to_f64().expect("unwrap failed")` pattern.
///
/// # Examples
///
/// ```
/// use optirs_core::utils::try_f64;
///
/// assert_eq!(try_f64(0.5f32).expect("f32 always fits in f64"), 0.5);
/// assert_eq!(try_f64(f64::INFINITY).expect("infinity is an f64"), f64::INFINITY);
/// ```
#[inline]
pub fn try_f64<A: Float>(value: A) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        OptimError::InvalidParameter(
            "value is not representable as f64, so the metric cannot be computed".to_string(),
        )
    })
}

/// Error text for a value that the target float type cannot represent.
fn unrepresentable<V: ToPrimitive>(value: V) -> String {
    match value.to_f64() {
        Some(shown) => {
            format!("value {shown} is not representable in the target floating-point type")
        }
        None => "value is not representable in the target floating-point type".to_string(),
    }
}

/// [`try_scalar`] for the call sites whose error type is `String`.
///
/// Several streaming modules return `Result<T, String>` rather than
/// [`OptimError`]; this keeps their conversions honest without forcing a
/// `.map_err(..)` at every site.
///
/// # Examples
///
/// ```
/// use optirs_core::utils::try_scalar_str;
///
/// let half: f32 = try_scalar_str(0.5).expect("0.5 is representable as f32");
/// assert_eq!(half, 0.5f32);
/// // As with [`try_scalar`], `f64 -> f32` saturates rather than failing.
/// assert_eq!(
///     try_scalar_str::<f32, _>(f64::MAX).expect("saturates"),
///     f32::INFINITY
/// );
/// ```
#[inline]
pub fn try_scalar_str<A: Float, V: ToPrimitive + Copy>(value: V) -> std::result::Result<A, String> {
    A::from(value).ok_or_else(|| unrepresentable(value))
}

/// Convert an `f64` value into the generic float type `A`, falling back to
/// `fallback` when `A` cannot represent it.
///
/// This is the infallible counterpart to [`try_scalar`], for the call sites
/// that structurally cannot return an error: `Default` impls, constructors and
/// struct literals. For the `f32`/`f64` types this crate targets, conversion of
/// the numeric literals used in those positions always succeeds, so the
/// fallback is defensive rather than a papered-over failure — but pick a
/// fallback that is safe in context (for example `Float::one` for a
/// multiplicative factor or a divisor, so a failed conversion can never
/// introduce a division by zero).
///
/// # Examples
///
/// ```
/// use optirs_core::utils::scalar_or;
///
/// assert_eq!(scalar_or::<f64, _>(0.9, 1.0), 0.9);
/// // The fallback covers conversions with no image at all. A float-to-float
/// // conversion is not one of them: it saturates, so `f64::MAX` reaches `f32`
/// // as an infinity rather than falling back.
/// assert_eq!(scalar_or::<f32, _>(f64::MAX, 1.0), f32::INFINITY);
/// ```
#[inline]
pub fn scalar_or<A: Float, V: ToPrimitive>(value: V, fallback: A) -> A {
    A::from(value).unwrap_or(fallback)
}

/// Convert a value into the generic float type `A`, or `None` when `A` cannot
/// represent it.
///
/// For call sites that already work in `Option` — typically a configuration
/// lookup whose miss falls through to a default — an unrepresentable value is
/// naturally "absent", so this keeps the existing fallback path instead of
/// inventing an error or a magic number.
///
/// # Examples
///
/// ```
/// use optirs_core::utils::scalar_opt;
///
/// assert_eq!(scalar_opt::<f32, _>(0.25), Some(0.25f32));
/// // Float-to-float conversion saturates rather than returning `None`.
/// assert_eq!(scalar_opt::<f32, _>(f64::MAX), Some(f32::INFINITY));
/// ```
#[inline]
pub fn scalar_opt<A: Float, V: ToPrimitive>(value: V) -> Option<A> {
    A::from(value)
}

/// Total ordering for floating-point values that never panics.
///
/// `f64::total_cmp`/`f32::total_cmp` are inherent methods and therefore
/// unavailable behind a generic `A: Float` bound, so this reproduces the same
/// contract: a genuine total order in which `NaN` sorts after every real number
/// (and equals itself). Use it instead of
/// `partial_cmp(..).expect("unwrap failed")`, which panics the moment a `NaN`
/// reaches the comparator, in `sort_by`/`min_by`/`max_by`/`select_nth`.
///
/// # Examples
///
/// ```
/// use optirs_core::utils::total_order;
///
/// let mut values = vec![2.0, f64::NAN, 1.0];
/// values.sort_by(total_order);
/// assert_eq!(values[0], 1.0);
/// assert_eq!(values[1], 2.0);
/// assert!(values[2].is_nan(), "NaN sorts last");
/// ```
pub fn total_order<A: Float>(a: &A, b: &A) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match a.partial_cmp(b) {
        Some(ordering) => ordering,
        None => match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // `partial_cmp` only returns `None` when at least one side is
            // `NaN`, so this arm is genuinely unreachable; treating it as
            // `Equal` keeps the comparator total regardless.
            (false, false) => Ordering::Equal,
        },
    }
}

/// Clip gradient values to a specified range
///
/// # Arguments
///
/// * `gradients` - The gradients to clip
/// * `min_value` - Minimum allowed value
/// * `max_value` - Maximum allowed value
///
/// # Returns
///
/// The clipped gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::utils::clip_gradients;
///
/// let mut gradients = Array1::from_vec(vec![-10.0, 0.5, 8.0, -0.2]);
/// clip_gradients(&mut gradients, -5.0, 5.0);
/// assert_eq!(gradients, Array1::from_vec(vec![-5.0, 0.5, 5.0, -0.2]));
/// ```
pub fn clip_gradients<A, D>(
    gradients: &mut Array<A, D>,
    min_value: A,
    max_value: A,
) -> &mut Array<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    for grad in gradients.iter_mut() {
        *grad = if *grad < min_value {
            min_value
        } else if *grad > max_value {
            max_value
        } else {
            *grad
        };
    }
    gradients
}

/// Clip gradient norm (global gradient clipping)
///
/// # Arguments
///
/// * `gradients` - The gradients to clip
/// * `max_norm` - Maximum allowed L2 norm
///
/// # Returns
///
/// The clipped gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::utils::clip_gradient_norm;
///
/// let mut gradients = Array1::<f64>::from_vec(vec![3.0, 4.0]); // L2 norm = 5.0
/// clip_gradient_norm(&mut gradients, 1.0f64).expect("a finite max-norm is valid");
/// // After clipping, L2 norm = 1.0
/// let diff0 = (gradients[0] - 0.6f64).abs();
/// let diff1 = (gradients[1] - 0.8f64).abs();
/// assert!(diff0 < 1e-5);
/// assert!(diff1 < 1e-5);
/// ```
pub fn clip_gradient_norm<A, D>(
    gradients: &mut Array<A, D>,
    max_norm: A,
) -> Result<&mut Array<A, D>>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    if max_norm <= A::zero() {
        return Err(OptimError::InvalidConfig(
            "max_norm must be positive".to_string(),
        ));
    }

    // Calculate current L2 _norm
    let _norm = gradients
        .iter()
        .fold(A::zero(), |acc, &x| acc + x * x)
        .sqrt();

    // If _norm exceeds max_norm, scale gradients
    if _norm > max_norm {
        let scale = max_norm / _norm;
        for grad in gradients.iter_mut() {
            *grad = *grad * scale;
        }
    }

    Ok(gradients)
}

/// Compute gradient centralization
///
/// Gradient Centralization is a technique that improves training stability
/// by removing the mean from each gradient tensor.
///
/// # Arguments
///
/// * `gradients` - The gradients to centralize
///
/// # Returns
///
/// The centralized gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::utils::gradient_centralization;
///
/// let mut gradients = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0]);
/// gradient_centralization(&mut gradients);
/// assert_eq!(gradients, Array1::from_vec(vec![-1.0, 0.0, 1.0, 0.0]));
/// ```
pub fn gradient_centralization<A, D>(gradients: &mut Array<A, D>) -> &mut Array<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    // Calculate mean
    let sum = gradients.iter().fold(A::zero(), |acc, &x| acc + x);
    let mean = sum / A::from(gradients.len()).unwrap_or(A::one());

    // Subtract mean from each element
    for grad in gradients.iter_mut() {
        *grad = *grad - mean;
    }

    gradients
}

/// Zero out small gradient values
///
/// # Arguments
///
/// * `gradients` - The gradients to process
/// * `threshold` - Threshold below which gradients are set to zero
///
/// # Returns
///
/// The processed gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::utils::zero_small_gradients;
///
/// let mut gradients = Array1::from_vec(vec![0.001, 0.02, -0.005, 0.3]);
/// zero_small_gradients(&mut gradients, 0.01);
/// assert_eq!(gradients, Array1::from_vec(vec![0.0, 0.02, 0.0, 0.3]));
/// ```
pub fn zero_small_gradients<A, D>(gradients: &mut Array<A, D>, threshold: A) -> &mut Array<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    let abs_threshold = threshold.abs();

    for grad in gradients.iter_mut() {
        if grad.abs() < abs_threshold {
            *grad = A::zero();
        }
    }

    gradients
}
