//! Hypervolume indicator: exact and bounded-approximate computation (F17), plus
//! the pluggable [`HypervolumeCalculator`] / [`HypervolumeMethod`] front end.
//!
//! The calculator is used directly by [`super::nsga2`], [`super::nsga3`] and
//! [`super::moead`] to derive the `convergence` front metric. (It previously
//! also backed an `SmsEmoa` struct that had no constructor and no methods; that
//! type is gone, the calculator is not.)

use crate::error::{OptimError, Result};
use crate::nas_engine::OptimizationDirection;
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

/// Hypervolume calculator
#[derive(Debug)]
pub struct HypervolumeCalculator<T: Float + Debug + Send + Sync + 'static> {
    /// Calculation method
    pub(super) method: HypervolumeMethod,
    /// Reference point
    pub(super) reference_point: Vec<T>,
    /// Cached hypervolumes
    pub(super) cache: HashMap<String, T>,
    /// Number of samples drawn by [`HypervolumeMethod::MonteCarlo`].
    pub(super) monte_carlo_samples: usize,
    /// Seed used by [`HypervolumeMethod::MonteCarlo`] so estimates are
    /// reproducible for a given point set.
    pub(super) monte_carlo_seed: u64,
}
/// Hypervolume calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypervolumeMethod {
    /// Walking Fish Group algorithm.
    ///
    /// WFG computes the *exact* hypervolume, so this crate evaluates it with the
    /// same exact slicing recursion as [`HypervolumeMethod::HSO`]. The two
    /// therefore return bit-identical results; they differ only in the internal
    /// traversal order a dedicated implementation would use, never in the value.
    WFG,
    /// Quick hypervolume (QHV).
    ///
    /// Also an *exact* algorithm, evaluated here through the same exact
    /// recursion. Selecting it is honest — the returned number is the true
    /// hypervolume — it simply does not use QHV's own pivot decomposition.
    Quick,
    /// Hypervolume by slicing objectives (the recursion implemented here).
    HSO,
    /// Monte Carlo estimation by uniform sampling of the reference box.
    ///
    /// This is a genuine estimator (not a fallback to the exact routine): it
    /// draws `HypervolumeCalculator::monte_carlo_samples` uniform points from
    /// the box spanned by the front's ideal corner and the reference point and
    /// returns `box_volume * dominated_fraction`. Use it only when an
    /// approximation is acceptable; it is seeded, so repeated calls on the same
    /// input agree.
    MonteCarlo,
}
/// Maximum number of front points fed to the exact hypervolume recursion when
/// there are four or more objectives. Beyond this the front is truncated to
/// the points closest to the ideal, yielding a documented lower-bound
/// approximation that keeps the (super-cubic) recursion tractable. Two- and
/// three-objective fronts are always computed exactly.
pub(super) const HV_HIGH_D_POINT_CAP: usize = 64;

/// Remove points that are Pareto-dominated by another point under a pure
/// minimization convention. Dominated points contribute no additional
/// hypervolume, so discarding them is exact and both cheapens the recursion
/// and makes the metric invariant to adding dominated points.
fn retain_non_dominated_min<T: Float>(points: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut kept: Vec<Vec<T>> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let mut dominated = false;
        for (j, q) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            // `q` dominates `p` when q <= p in every objective and q < p in at
            // least one. Identical vectors do not dominate one another and are
            // both retained (duplicates add no volume to the union).
            let mut all_le = true;
            let mut any_lt = false;
            let len = p.len().min(q.len());
            for k in 0..len {
                if q[k] > p[k] {
                    all_le = false;
                    break;
                }
                if q[k] < p[k] {
                    any_lt = true;
                }
            }
            if all_le && any_lt {
                dominated = true;
                break;
            }
        }
        if !dominated {
            kept.push(p.clone());
        }
    }
    kept
}

/// Recursive Hypervolume by Slicing Objectives (HSO). `points` are oriented so
/// that smaller is better in every objective and `reference` is the (worse
/// than every point) upper corner. The algorithm sweeps the last objective,
/// accumulating the (dims-1)-dimensional cross-sectional volume of the active
/// point set times each slab depth. Exact for any dimension; complexity grows
/// with dimension and point count, so callers cap the point count for
/// high-dimensional fronts.
fn hv_recursive<T: Float>(points: &[Vec<T>], reference: &[T], dims: usize) -> T {
    if points.is_empty() || dims == 0 {
        return T::zero();
    }
    if dims == 1 {
        // Extent from the best (smallest) coordinate to the reference.
        let mut best = reference[0];
        for p in points {
            if p[0] < best {
                best = p[0];
            }
        }
        let extent = reference[0] - best;
        return if extent > T::zero() {
            extent
        } else {
            T::zero()
        };
    }

    let last = dims - 1;
    let mut sorted: Vec<&Vec<T>> = points.iter().collect();
    sorted.sort_by(|a, b| a[last].partial_cmp(&b[last]).unwrap_or(Ordering::Equal));

    let mut volume = T::zero();
    for i in 0..sorted.len() {
        let lower = sorted[i][last];
        let upper = if i + 1 < sorted.len() {
            sorted[i + 1][last]
        } else {
            reference[last]
        };
        let depth = upper - lower;
        if depth <= T::zero() {
            continue;
        }
        // Cross-section: the (dims-1)-dimensional hypervolume of the points
        // active in this slab (those with a smaller-or-equal last coordinate),
        // projected onto the leading objectives.
        let projections: Vec<Vec<T>> = sorted[..=i].iter().map(|p| p[..last].to_vec()).collect();
        let area = hv_recursive(&projections, &reference[..last], last);
        volume = volume + area * depth;
    }
    volume
}

/// Exact (two- and three-objective) / bounded-approximate (four or more
/// objective) hypervolume of `points` relative to `reference`, under a pure
/// minimization convention (F17). Points that do not strictly dominate the
/// reference contribute nothing and are discarded; Pareto-dominated points are
/// likewise removed (they add no volume), which makes the metric invariant to
/// adding dominated points and strictly increasing when a genuinely
/// non-dominated point is added.
pub fn hypervolume_minimization<T: Float>(points: &[Vec<T>], reference: &[T]) -> T {
    let dims = reference.len();
    if points.is_empty() || dims == 0 {
        return T::zero();
    }

    // Keep only points that strictly dominate the reference in every objective.
    let feasible: Vec<Vec<T>> = points
        .iter()
        .filter(|p| p.len() >= dims && (0..dims).all(|k| p[k] < reference[k]))
        .map(|p| p[..dims].to_vec())
        .collect();

    // Discard dominated points (exact, and it cheapens the recursion).
    let mut front = retain_non_dominated_min(&feasible);
    if front.is_empty() {
        return T::zero();
    }

    // In four or more objectives the recursion is super-cubic; cap the point
    // count to keep it tractable (documented lower-bound approximation). The
    // retained points are those closest (L1) to the front-derived ideal.
    if dims >= 4 && front.len() > HV_HIGH_D_POINT_CAP {
        let mut ideal = vec![T::infinity(); dims];
        for p in &front {
            for k in 0..dims {
                if p[k] < ideal[k] {
                    ideal[k] = p[k];
                }
            }
        }
        front.sort_by(|a, b| {
            let da = (0..dims).fold(T::zero(), |acc, k| acc + (a[k] - ideal[k]).abs());
            let db = (0..dims).fold(T::zero(), |acc, k| acc + (b[k] - ideal[k]).abs());
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        front.truncate(HV_HIGH_D_POINT_CAP);
    }

    hv_recursive(&front, reference, dims)
}

/// Negate every objective whose optimization direction is
/// [`OptimizationDirection::Maximize`], turning an arbitrary mixed-direction
/// objective vector into the pure-minimization convention that
/// [`hypervolume_minimization`] (and every other indicator in
/// [`super::metrics`]) requires.
///
/// The same transform must be applied to the reference point, otherwise a
/// maximized objective is compared against a bound on the wrong side and the
/// indicator silently collapses to zero. [`normalize_front_for_minimization`]
/// and [`normalize_reference_for_minimization`] exist so callers cannot forget.
fn negate_maximized<T: Float>(values: &[T], directions: &[OptimizationDirection]) -> Vec<T> {
    values
        .iter()
        .enumerate()
        .map(|(k, &v)| match directions.get(k) {
            Some(OptimizationDirection::Maximize) => -v,
            _ => v,
        })
        .collect()
}

/// Map a whole front into the pure-minimization convention (see
/// `negate_maximized`).
pub fn normalize_front_for_minimization<T: Float>(
    front: &[Vec<T>],
    directions: &[OptimizationDirection],
) -> Vec<Vec<T>> {
    front
        .iter()
        .map(|point| negate_maximized(point, directions))
        .collect()
}

/// Map a reference point into the pure-minimization convention (see
/// `negate_maximized`).
///
/// The transform is its own inverse (negating the same entries twice restores the
/// original), so the same call converts a minimization-space reference back into
/// raw objective space.
pub fn normalize_reference_for_minimization<T: Float>(
    reference: &[T],
    directions: &[OptimizationDirection],
) -> Vec<T> {
    negate_maximized(reference, directions)
}

/// Relative margin added beyond the observed nadir when a reference point has
/// to be derived from the first observed front.
const HV_REFERENCE_MARGIN_FRACTION: f64 = 0.1;

/// Absolute floor for the derived reference margin, so a degenerate (single
/// point, or zero-range) front still yields a strictly positive reference box
/// instead of a zero hypervolume.
const HV_REFERENCE_MARGIN_FLOOR: f64 = 1e-3;

/// Derive a reference point from an observed front, in the pure-minimization
/// convention.
///
/// For each objective the reference is placed past the observed nadir (the worst
/// value on the front) by `max(0.1 * range, 0.1 * |nadir|, 1e-3)`. Callers are
/// expected to derive this **once** and then hold it fixed: a reference that
/// tracks the current front makes the hypervolume non-comparable between
/// generations, since the measured box itself moves.
pub fn derive_reference_point<T: Float>(front: &[Vec<T>]) -> Option<Vec<T>> {
    let dims = front.first()?.len();
    if dims == 0 {
        return None;
    }
    let mut min_values = vec![T::infinity(); dims];
    let mut max_values = vec![T::neg_infinity(); dims];
    for point in front {
        if point.len() < dims {
            return None;
        }
        for k in 0..dims {
            if point[k] < min_values[k] {
                min_values[k] = point[k];
            }
            if point[k] > max_values[k] {
                max_values[k] = point[k];
            }
        }
    }

    let fraction: T =
        scirs2_core::numeric::NumCast::from(HV_REFERENCE_MARGIN_FRACTION).unwrap_or_else(T::zero);
    let floor: T =
        scirs2_core::numeric::NumCast::from(HV_REFERENCE_MARGIN_FLOOR).unwrap_or_else(T::zero);

    let mut reference = Vec::with_capacity(dims);
    for k in 0..dims {
        if !min_values[k].is_finite() || !max_values[k].is_finite() {
            return None;
        }
        let range = max_values[k] - min_values[k];
        let margin = (range * fraction)
            .max(max_values[k].abs() * fraction)
            .max(floor);
        reference.push(max_values[k] + margin);
    }
    Some(reference)
}

impl<T: Float + Debug + Send + Sync + 'static> HypervolumeCalculator<T> {
    /// Default number of Monte Carlo samples.
    pub const DEFAULT_MONTE_CARLO_SAMPLES: usize = 100_000;

    /// Create a calculator for `method` measuring against `reference_point`
    /// (expressed in the pure-minimization convention).
    pub fn new(method: HypervolumeMethod, reference_point: Vec<T>) -> Self {
        Self {
            method,
            reference_point,
            cache: HashMap::new(),
            monte_carlo_samples: Self::DEFAULT_MONTE_CARLO_SAMPLES,
            monte_carlo_seed: 0x5EED_1234_ABCD_0001,
        }
    }

    /// The configured calculation method.
    pub fn method(&self) -> HypervolumeMethod {
        self.method
    }

    /// The reference point currently in use.
    pub fn reference_point(&self) -> &[T] {
        &self.reference_point
    }

    /// Replace the reference point. The cache is invalidated because cached
    /// values are only meaningful for the reference they were computed against.
    pub fn set_reference_point(&mut self, reference_point: Vec<T>) {
        self.reference_point = reference_point;
        self.cache.clear();
    }

    /// Configure the Monte Carlo sample count (ignored by the exact methods).
    pub fn set_monte_carlo_samples(&mut self, samples: usize) {
        self.monte_carlo_samples = samples;
        self.cache.clear();
    }

    /// Drop every cached hypervolume.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Compute the hypervolume of `points` (pure-minimization convention).
    ///
    /// Returns an error rather than a fabricated number when the reference point
    /// is missing or a point has fewer objectives than the reference.
    pub fn calculate(&mut self, points: &[Vec<T>]) -> Result<T> {
        if self.reference_point.is_empty() {
            return Err(OptimError::InvalidConfig(
                "hypervolume requires a non-empty reference point".to_string(),
            ));
        }
        let dims = self.reference_point.len();
        if let Some(bad) = points.iter().find(|p| p.len() < dims) {
            return Err(OptimError::InvalidParameter(format!(
                "hypervolume point has {} objectives but the reference point has {}",
                bad.len(),
                dims
            )));
        }
        if points.is_empty() {
            return Ok(T::zero());
        }

        let key = self.cache_key(points);
        if let Some(cached) = self.cache.get(&key) {
            return Ok(*cached);
        }

        let value = match self.method {
            // All three exact methods return the true hypervolume; see the
            // variant docs for why they share one implementation.
            HypervolumeMethod::HSO | HypervolumeMethod::WFG | HypervolumeMethod::Quick => {
                hypervolume_minimization(points, &self.reference_point)
            }
            HypervolumeMethod::MonteCarlo => self.monte_carlo_estimate(points)?,
        };
        self.cache.insert(key, value);
        Ok(value)
    }

    /// Exclusive hypervolume contribution of `points[index]`: how much
    /// hypervolume is lost when that single point is removed. This is the
    /// quantity SMS-EMOA's environmental selection ranks on.
    pub fn contribution(&mut self, points: &[Vec<T>], index: usize) -> Result<T> {
        if index >= points.len() {
            return Err(OptimError::InvalidParameter(format!(
                "hypervolume contribution index {} is out of range for {} points",
                index,
                points.len()
            )));
        }
        let total = self.calculate(points)?;
        let without: Vec<Vec<T>> = points
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, p)| p.clone())
            .collect();
        let reduced = self.calculate(&without)?;
        Ok(total - reduced)
    }

    /// Monte Carlo estimate: sample the box spanned by the front's ideal corner
    /// and the reference point, and scale the box volume by the fraction of
    /// samples that are dominated by at least one point.
    fn monte_carlo_estimate(&self, points: &[Vec<T>]) -> Result<T> {
        let dims = self.reference_point.len();
        if self.monte_carlo_samples == 0 {
            return Err(OptimError::InvalidConfig(
                "Monte Carlo hypervolume requires at least one sample".to_string(),
            ));
        }

        // Only points that strictly dominate the reference bound any volume.
        let feasible: Vec<&Vec<T>> = points
            .iter()
            .filter(|p| (0..dims).all(|k| p[k] < self.reference_point[k]))
            .collect();
        if feasible.is_empty() {
            return Ok(T::zero());
        }

        let mut lower = vec![T::infinity(); dims];
        for point in &feasible {
            for (slot, value) in lower.iter_mut().zip(point.iter()) {
                if *value < *slot {
                    *slot = *value;
                }
            }
        }

        let mut box_volume = 1.0f64;
        let mut spans = Vec::with_capacity(dims);
        for (bound, floor) in self.reference_point.iter().zip(lower.iter()) {
            let span = (*bound - *floor).to_f64().unwrap_or(0.0);
            if span <= 0.0 {
                return Ok(T::zero());
            }
            spans.push(span);
            box_volume *= span;
        }

        let lower_f64: Vec<f64> = lower.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let feasible_f64: Vec<Vec<f64>> = feasible
            .iter()
            .map(|p| (0..dims).map(|k| p[k].to_f64().unwrap_or(0.0)).collect())
            .collect();

        let mut rng = Random::seed(self.monte_carlo_seed);
        let mut hits = 0usize;
        let mut sample = vec![0.0f64; dims];
        for _ in 0..self.monte_carlo_samples {
            for k in 0..dims {
                sample[k] = lower_f64[k] + rng.gen_range(0.0..1.0) * spans[k];
            }
            if feasible_f64
                .iter()
                .any(|p| (0..dims).all(|k| p[k] <= sample[k]))
            {
                hits += 1;
            }
        }

        let fraction = hits as f64 / self.monte_carlo_samples as f64;
        Ok(scirs2_core::numeric::NumCast::from(box_volume * fraction).unwrap_or_else(T::zero))
    }

    /// Deterministic cache key covering the method, the reference point and the
    /// (order-independent) point set, all by exact bit pattern.
    fn cache_key(&self, points: &[Vec<T>]) -> String {
        let mut rows: Vec<Vec<u64>> = points
            .iter()
            .map(|p| {
                p.iter()
                    .map(|v| v.to_f64().unwrap_or(f64::NAN).to_bits())
                    .collect()
            })
            .collect();
        rows.sort_unstable();
        let reference: Vec<u64> = self
            .reference_point
            .iter()
            .map(|v| v.to_f64().unwrap_or(f64::NAN).to_bits())
            .collect();
        format!(
            "{:?}|{:?}|{}|{:?}",
            self.method, reference, self.monte_carlo_samples, rows
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact 2-objective hypervolume, verifiable by hand: the union of
    /// `[1,2]x[0,2]` and `[0,2]x[1,2]` has area `2 + 2 - 1 = 3`.
    #[test]
    fn two_objective_hypervolume_matches_the_published_value() {
        let front = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let reference = vec![2.0, 2.0];
        let hv = hypervolume_minimization(&front, &reference);
        assert!(
            (hv - 3.0).abs() < 1e-12,
            "expected exactly 3.0, got {hv} (the pre-fix bounding-box heuristic returned 4.0 * 2 = 8.0)"
        );
    }

    /// Exact 3-objective hypervolume by inclusion-exclusion:
    /// `3 * 4 - 3 * 2 + 1 = 7`.
    #[test]
    fn three_objective_hypervolume_matches_the_published_value() {
        let front = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let reference = vec![2.0, 2.0, 2.0];
        let hv = hypervolume_minimization(&front, &reference);
        assert!(
            (hv - 7.0).abs() < 1e-12,
            "expected exactly 7.0 (12 - 6 + 1), got {hv}"
        );
    }

    /// A convex 2-objective front on the unit box: inclusion-exclusion gives
    /// `0.43 - 0.11 + 0.01 = 0.33`.
    #[test]
    fn three_point_convex_front_has_the_hand_computed_hypervolume() {
        let front = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let reference = vec![1.0, 1.0];
        let hv = hypervolume_minimization(&front, &reference);
        assert!((hv - 0.33).abs() < 1e-12, "expected 0.33, got {hv}");
    }

    /// A single point spans exactly its own box; the old heuristic multiplied a
    /// zero-range bounding box by the point count instead.
    #[test]
    fn single_point_hypervolume_is_its_own_box() {
        let hv = hypervolume_minimization(&[vec![0.5, 0.25]], &[1.0, 1.0]);
        assert!((hv - 0.375).abs() < 1e-12, "expected 0.5 * 0.75, got {hv}");
    }

    /// Adding a dominated point cannot change the indicator, and adding a
    /// genuinely non-dominated point must strictly increase it. The old
    /// bounding-box-times-count heuristic violated both.
    #[test]
    fn hypervolume_is_monotone_and_ignores_dominated_points() {
        let reference = vec![2.0, 2.0];
        let base = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let with_dominated = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.5, 1.5]];
        let with_new = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let hv_base = hypervolume_minimization(&base, &reference);
        let hv_dominated = hypervolume_minimization(&with_dominated, &reference);
        let hv_new = hypervolume_minimization(&with_new, &reference);

        assert!(
            (hv_base - hv_dominated).abs() < 1e-12,
            "a dominated point must not change the hypervolume: {hv_base} vs {hv_dominated}"
        );
        assert!(
            hv_new > hv_base + 1e-12,
            "a non-dominated point must strictly increase it: {hv_new} vs {hv_base}"
        );
    }

    /// Points on the far side of the reference point contribute nothing.
    #[test]
    fn points_not_dominating_the_reference_contribute_nothing() {
        let hv = hypervolume_minimization(&[vec![3.0, 3.0], vec![5.0, 0.5]], &[2.0, 2.0]);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn maximized_objectives_are_negated_on_both_front_and_reference() {
        let directions = vec![
            OptimizationDirection::Maximize,
            OptimizationDirection::Minimize,
        ];
        // Raw space: maximize accuracy (0.9 is good), minimize memory.
        let front = vec![vec![0.9, 1.0], vec![0.6, 0.5]];
        let reference = vec![0.0, 2.0];

        let min_front = normalize_front_for_minimization(&front, &directions);
        let min_reference = normalize_reference_for_minimization(&reference, &directions);
        assert_eq!(min_front, vec![vec![-0.9, 1.0], vec![-0.6, 0.5]]);
        assert_eq!(min_reference, vec![0.0, 2.0]);

        // Union of [-0.9,0]x[1,2] and [-0.6,0]x[0.5,2]: 0.9*1 + 0.6*1.5 - 0.6*1
        let hv = hypervolume_minimization(&min_front, &min_reference);
        assert!((hv - 1.2).abs() < 1e-12, "expected 1.2, got {hv}");

        // Forgetting the negation collapses the indicator to zero, which is the
        // silent failure this helper pair exists to prevent.
        assert_eq!(hypervolume_minimization(&front, &reference), 0.0);
    }

    #[test]
    fn every_exact_method_returns_the_same_value() {
        let front = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        for method in [
            HypervolumeMethod::HSO,
            HypervolumeMethod::WFG,
            HypervolumeMethod::Quick,
        ] {
            let mut calc = HypervolumeCalculator::new(method, vec![2.0, 2.0]);
            let hv = calc.calculate(&front).expect("exact hypervolume");
            assert!((hv - 3.0).abs() < 1e-12, "{method:?} returned {hv}");
        }
    }

    #[test]
    fn monte_carlo_estimate_is_close_to_the_exact_value_and_reproducible() {
        let front = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let mut calc = HypervolumeCalculator::new(HypervolumeMethod::MonteCarlo, vec![1.0, 1.0]);
        calc.set_monte_carlo_samples(200_000);

        let first = calc.calculate(&front).expect("monte carlo hypervolume");
        calc.clear_cache();
        let second = calc.calculate(&front).expect("monte carlo hypervolume");

        assert_eq!(first, second, "a seeded estimator must be reproducible");
        assert!(
            (first - 0.33).abs() < 5e-3,
            "estimate {first} is too far from the exact 0.33"
        );
        // It must be an estimate, not a silent hand-off to the exact routine.
        assert_ne!(first, 0.33);
    }

    #[test]
    fn calculate_rejects_a_missing_reference_and_short_points() {
        let mut no_reference: HypervolumeCalculator<f64> =
            HypervolumeCalculator::new(HypervolumeMethod::HSO, Vec::new());
        assert!(no_reference.calculate(&[vec![0.0, 0.0]]).is_err());

        let mut calc = HypervolumeCalculator::new(HypervolumeMethod::HSO, vec![1.0, 1.0]);
        assert!(calc.calculate(&[vec![0.5]]).is_err());
        assert_eq!(calc.calculate(&[]).expect("empty front"), 0.0);
    }

    #[test]
    fn exclusive_contribution_is_the_volume_lost_by_removing_a_point() {
        // Middle point of the convex front: 0.33 - (0.09 + 0.09 - 0.01) = 0.16
        let front = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let mut calc = HypervolumeCalculator::new(HypervolumeMethod::HSO, vec![1.0, 1.0]);
        let contribution = calc.contribution(&front, 1).expect("contribution");
        assert!(
            (contribution - 0.16).abs() < 1e-12,
            "expected 0.16, got {contribution}"
        );
        assert!(calc.contribution(&front, 9).is_err());
    }

    #[test]
    fn derived_reference_point_is_strictly_worse_than_the_whole_front() {
        let front = vec![vec![0.1, 0.9], vec![0.9, 0.1]];
        let reference = derive_reference_point(&front).expect("reference from a non-empty front");
        assert_eq!(reference.len(), 2);
        for point in &front {
            for k in 0..2 {
                assert!(
                    point[k] < reference[k],
                    "reference {:?} must dominate-bound {:?}",
                    reference,
                    point
                );
            }
        }
        assert!(
            hypervolume_minimization(&front, &reference) > 0.0,
            "a derived reference must yield a positive hypervolume"
        );

        // Degenerate single-point front still yields a usable box.
        let single = derive_reference_point(&[vec![0.0, 0.0]]).expect("single point reference");
        assert!(single.iter().all(|v| *v > 0.0), "got {single:?}");
        assert!(derive_reference_point::<f64>(&[]).is_none());
    }
}
