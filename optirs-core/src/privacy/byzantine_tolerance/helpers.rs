//! Numeric helpers, module-wide constants and the deterministic RNG used by
//! the isolation-forest outlier detector and FLAME's noise calibration.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use super::types::SplitMix64;

/// Type alias for validation rule function
pub(super) type RuleFn<T> = Box<dyn Fn(&Array1<T>) -> bool + Send + Sync>;

/// Maximum number of per-participant history samples retained.
pub(super) const HISTORY_CAPACITY: usize = 1000;

/// Maximum number of behaviour prototypes retained by [`PatternModel`].
pub(super) const MAX_PATTERNS: usize = 32;

/// Noise multiplier used by the FLAME aggregator (lambda in Nguyen et al., 2022).
pub(super) const FLAME_NOISE_LAMBDA: f64 = 0.001;

/// Number of trees built by the isolation-forest outlier detector.
pub(super) const ISOLATION_TREES: usize = 100;

/// Euler-Mascheroni constant, used by the isolation-forest path-length normaliser.
pub(super) const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Convert an `f64` constant into `T`, reporting an error when the target
/// floating point type cannot represent it.
pub(super) fn to_scalar<T: Float>(value: f64) -> Result<T> {
    T::from(value).ok_or_else(|| {
        OptimError::ComputationError(format!(
            "value {value} is not representable in the target floating point type"
        ))
    })
}

/// Convert a `T` value into `f64`, reporting an error when the conversion fails.
pub(super) fn from_scalar<T: Float>(value: T) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        OptimError::ComputationError("floating point value is not convertible to f64".to_string())
    })
}

/// L2 norm of a gradient.
pub(super) fn l2_norm<T: Float>(gradient: &Array1<T>) -> T {
    gradient
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt()
}

/// Euclidean distance between two gradients.
pub(super) fn euclidean_distance<T: Float>(a: &Array1<T>, b: &Array1<T>) -> Result<T> {
    if a.len() != b.len() {
        return Err(OptimError::DimensionMismatch(format!(
            "gradient dimensions don't match: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let mut sum = T::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }

    Ok(sum.sqrt())
}

/// Cosine similarity between two gradients. Returns `0` when either operand has a
/// zero norm (the angle is undefined in that case).
pub(super) fn cosine_similarity<T: Float>(a: &Array1<T>, b: &Array1<T>) -> Result<T> {
    if a.len() != b.len() {
        return Err(OptimError::DimensionMismatch(format!(
            "gradient dimensions don't match: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let mut dot_product = T::zero();
    let mut norm_a = T::zero();
    let mut norm_b = T::zero();

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product = dot_product + *x * *y;
        norm_a = norm_a + *x * *x;
        norm_b = norm_b + *y * *y;
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a > T::zero() && norm_b > T::zero() {
        Ok(dot_product / (norm_a * norm_b))
    } else {
        Ok(T::zero())
    }
}

/// Convert a gradient into an `f64` vector for the detectors that work in `f64`.
pub(super) fn to_f64_vec<T: Float>(gradient: &Array1<T>) -> Result<Vec<f64>> {
    gradient.iter().copied().map(from_scalar).collect()
}

/// Accumulate isolation-tree path lengths for every point in `idx`.
pub(super) fn isolation_path(
    points: &[Vec<f64>],
    idx: &[usize],
    depth: usize,
    limit: usize,
    rng: &mut SplitMix64,
    out: &mut [f64],
) {
    let terminate = |out: &mut [f64]| {
        let adjustment = average_path_length(idx.len());
        for &i in idx {
            out[i] += depth as f64 + adjustment;
        }
    };

    if idx.len() <= 1 || depth >= limit {
        terminate(out);
        return;
    }

    let dim = points[idx[0]].len();
    let mut candidates: Vec<(usize, f64, f64)> = Vec::new();
    // `d` indexes the coordinate of every point in `idx`, not a single slice, so
    // there is no iterator form of this scan.
    #[allow(clippy::needless_range_loop)]
    for d in 0..dim {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &i in idx {
            let value = points[i][d];
            if value < lo {
                lo = value;
            }
            if value > hi {
                hi = value;
            }
        }
        if hi > lo {
            candidates.push((d, lo, hi));
        }
    }

    if candidates.is_empty() {
        terminate(out);
        return;
    }

    let (dimension, lo, hi) = candidates[rng.next_range(candidates.len())];
    let split = lo + rng.next_f64() * (hi - lo);

    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in idx {
        if points[i][dimension] < split {
            left.push(i);
        } else {
            right.push(i);
        }
    }

    if left.is_empty() || right.is_empty() {
        terminate(out);
        return;
    }

    isolation_path(points, &left, depth + 1, limit, rng, out);
    isolation_path(points, &right, depth + 1, limit, rng, out);
}

/// Average path length of an unsuccessful search in a binary search tree of `n`
/// nodes, the isolation-forest normalisation constant `c(n)`.
pub(super) fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    2.0 * ((nf - 1.0).ln() + EULER_MASCHERONI) - 2.0 * (nf - 1.0) / nf
}
