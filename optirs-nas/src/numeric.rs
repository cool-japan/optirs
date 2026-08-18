//! Fallible numeric conversions shared across the crate.
//!
//! `T: Float` only promises `NumCast`, so `T::from(x)` returns an `Option`. The
//! usual shortcuts for that option are both wrong in production code:
//! `.expect("conversion failed")` aborts a whole architecture search over a
//! statistics divisor, and `.unwrap_or_else(T::zero)` silently substitutes a
//! divisor of zero (or a weight of zero, or a bound of zero) and lets the wrong
//! number propagate. These helpers return an honest [`OptimError`] instead, with
//! enough context to name the call site.
//!
//! # When the error arm actually fires
//!
//! For the element types this crate is instantiated with — `f32` and `f64` —
//! **it does not**. `NumCast::from::<f64>` saturates rather than failing:
//! `f64::MAX` becomes `f32::INFINITY`, `f64::NAN` stays NaN. The `Err` arm exists
//! because `T` is a type parameter: a caller may supply a `Float` implementation
//! with a bounded domain (a fixed-point or checked wrapper), and such a type is
//! entitled to reject a value. Callers therefore propagate the error rather than
//! substituting a plausible number, and the branch is simply never taken for the
//! built-in floats.
//!
//! Note what these helpers do *not* do: they do not reject NaN or infinity,
//! because those are representable in the target type. A caller that needs a
//! finite value must check for it — the guard belongs where the meaning is known.

use crate::error::{OptimError, Result};
use scirs2_core::numeric::{Float, NumCast};

/// Convert a count into the element type `T`.
///
/// `context` names what the count is, so the error identifies the call site
/// rather than only the failure mode.
pub(crate) fn count_as<T: Float>(count: usize, context: &str) -> Result<T> {
    NumCast::from(count as f64).ok_or_else(|| {
        OptimError::InvalidParameter(format!(
            "cannot represent {} ({}) in the optimizer's element type",
            context, count
        ))
    })
}

/// Convert an `f64` value into the element type `T`.
pub(crate) fn scalar_as<T: Float>(value: f64, context: &str) -> Result<T> {
    NumCast::from(value).ok_or_else(|| {
        OptimError::InvalidParameter(format!(
            "cannot represent {} ({}) in the optimizer's element type",
            context, value
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_scalars_convert() {
        assert_eq!(
            count_as::<f64>(7, "sample count").expect("convertible"),
            7.0
        );
        assert_eq!(
            count_as::<f32>(7, "sample count").expect("convertible"),
            7.0
        );
        assert_eq!(
            scalar_as::<f64>(1.96, "z score").expect("convertible"),
            1.96
        );
    }

    #[test]
    fn the_built_in_floats_saturate_rather_than_failing() {
        // This is the fact the module doc rests on: for `f32`/`f64` the `Err` arm
        // is unreachable, so a caller that propagates the error is not choosing a
        // fallback it will silently rely on. If a future `scirs2-core` made these
        // conversions reject out-of-range values instead, this test tells us.
        assert!(scalar_as::<f32>(f64::MAX, "huge")
            .expect("f64::MAX saturates to f32::INFINITY")
            .is_infinite());
        assert!(scalar_as::<f32>(f64::NAN, "nan")
            .expect("NaN is representable")
            .is_nan());
        assert!(scalar_as::<f32>(f64::INFINITY, "inf")
            .expect("infinity is representable")
            .is_infinite());
        assert!(count_as::<f32>(usize::MAX, "max count")
            .expect("usize::MAX is far inside f32's range")
            .is_finite());
    }

    #[test]
    fn a_failed_conversion_names_its_context() {
        // The message is what makes a propagated error actionable, so pin its
        // shape even though the built-in floats never take this branch.
        let message = OptimError::InvalidParameter(format!(
            "cannot represent {} ({}) in the optimizer's element type",
            "sample count", 3
        ))
        .to_string();
        assert!(message.contains("sample count"));
        assert!(message.contains("Invalid parameter"));
    }
}
