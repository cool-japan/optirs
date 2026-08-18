//! Real Pareto-front quality indicators (F17 cluster).
//!
//! Every function here computes an actual, defined quantity from the front it is
//! given. They replace the previously hardcoded `convergence = 0` and
//! `objective_space_coverage = 0.5` placeholders in [`super::core::FrontMetrics`].
//!
//! All of them assume the **pure-minimization convention**: smaller is better in
//! every objective. Use
//! [`super::hypervolume::normalize_front_for_minimization`] (and its reference
//! counterpart) to get there from a mixed-direction objective configuration.

use scirs2_core::numeric::Float;

use super::core::{CoverageMetrics, FrontMetrics};

/// Smallest denominator accepted before a ratio is reported as zero rather than
/// as a division by (near) zero.
const RATIO_EPSILON: f64 = 1e-12;

/// Per-objective minima and maxima of a front, or `None` for an empty front or
/// a front whose points carry no objectives.
pub fn front_bounds<T: Float>(front: &[Vec<T>]) -> Option<(Vec<T>, Vec<T>)> {
    let dims = front.first()?.len();
    if dims == 0 {
        return None;
    }
    let mut min_values = vec![T::infinity(); dims];
    let mut max_values = vec![T::neg_infinity(); dims];
    for point in front {
        for k in 0..dims.min(point.len()) {
            if point[k] < min_values[k] {
                min_values[k] = point[k];
            }
            if point[k] > max_values[k] {
                max_values[k] = point[k];
            }
        }
    }
    Some((min_values, max_values))
}

/// Relative change of the hypervolume indicator between two consecutive front
/// updates — the crate's `convergence` measure.
///
/// Defined as `|current - previous| / max(|current|, |previous|)`, clamped to
/// `[0, 1]`. A value near `0` means the indicator has stopped moving (the search
/// has converged in indicator terms); `1` means it changed by as much as its own
/// magnitude. When there is no earlier hypervolume (first update) the result is
/// `1`, because nothing has been shown to have converged yet.
///
/// This is a genuine measurement of the optimizer's own progress. It is
/// deliberately *not* a generational distance: that requires a known true Pareto
/// front, which a NAS run does not have.
pub fn hypervolume_convergence<T: Float>(current: T, previous: T) -> T {
    let scale = current.abs().max(previous.abs());
    let epsilon: T = scirs2_core::numeric::NumCast::from(RATIO_EPSILON).unwrap_or_else(T::zero);
    if scale <= epsilon {
        // Both hypervolumes are (numerically) zero: nothing has been measured,
        // so report "not converged" rather than a flattering 0.
        return T::one();
    }
    let ratio = (current - previous).abs() / scale;
    if ratio > T::one() {
        T::one()
    } else {
        ratio
    }
}

/// Fraction of the reachable objective box that the front's own bounding box
/// spans — the crate's `objective_space_coverage`.
///
/// The reachable box runs from the front's ideal corner (its per-objective
/// minima) to `reference`. The returned value is
/// `prod_k (max_k - min_k) / prod_k (reference_k - min_k)`, clamped to `[0, 1]`.
/// A single-point front therefore scores `0` (it spans no volume) and a front
/// whose extremes reach the reference in every objective scores `1`.
pub fn objective_space_coverage<T: Float>(front: &[Vec<T>], reference: &[T]) -> T {
    let Some((min_values, max_values)) = front_bounds(front) else {
        return T::zero();
    };
    let dims = min_values.len().min(reference.len());
    if dims == 0 {
        return T::zero();
    }
    let epsilon: T = scirs2_core::numeric::NumCast::from(RATIO_EPSILON).unwrap_or_else(T::zero);

    let mut spanned = T::one();
    let mut reachable = T::one();
    for k in 0..dims {
        let span = (max_values[k] - min_values[k]).max(T::zero());
        let reach = reference[k] - min_values[k];
        if reach <= epsilon {
            // The reference does not bound this objective, so no fraction of it
            // is meaningfully covered.
            return T::zero();
        }
        spanned = spanned * span;
        reachable = reachable * reach;
    }
    if reachable <= epsilon {
        return T::zero();
    }
    let coverage = spanned / reachable;
    if coverage > T::one() {
        T::one()
    } else {
        coverage
    }
}

/// Mean Euclidean distance from the front's solutions to `reference`.
///
/// Small values mean the front sits close to the (deliberately bad) reference
/// corner, i.e. the search has not pushed far into the objective space yet.
pub fn mean_reference_distance<T: Float>(front: &[Vec<T>], reference: &[T]) -> T {
    if front.is_empty() || reference.is_empty() {
        return T::zero();
    }
    let dims = reference.len();
    let mut total = T::zero();
    let mut counted = 0usize;
    for point in front {
        let mut sq_sum = T::zero();
        for k in 0..dims.min(point.len()) {
            let diff = reference[k] - point[k];
            sq_sum = sq_sum + diff * diff;
        }
        total = total + sq_sum.sqrt();
        counted += 1;
    }
    if counted == 0 {
        return T::zero();
    }
    total / scirs2_core::numeric::NumCast::from(counted as f64).unwrap_or_else(T::one)
}

/// Additive epsilon indicator `I_eps+(approximation, reference_set)`.
///
/// The smallest `eps >= 0` such that every member `r` of `reference_set` is
/// weakly dominated by some member `a` of `approximation` after `a` is shifted by
/// `eps` in every objective; formally
/// `max_r min_a max_k (a_k - r_k)`, floored at zero.
///
/// Feeding the whole population as `reference_set` and the non-dominated front as
/// `approximation` yields "how far, in objective units, the front is from
/// covering everything the optimizer has seen" — `0` when the front dominates
/// the entire population.
pub fn additive_epsilon_indicator<T: Float>(
    approximation: &[Vec<T>],
    reference_set: &[Vec<T>],
) -> T {
    if approximation.is_empty() || reference_set.is_empty() {
        return T::zero();
    }
    let mut worst = T::zero();
    for r in reference_set {
        let mut best_for_r = T::infinity();
        for a in approximation {
            let dims = a.len().min(r.len());
            if dims == 0 {
                continue;
            }
            let mut needed = T::neg_infinity();
            for k in 0..dims {
                let shift = a[k] - r[k];
                if shift > needed {
                    needed = shift;
                }
            }
            if needed < best_for_r {
                best_for_r = needed;
            }
        }
        if best_for_r.is_finite() && best_for_r > worst {
            worst = best_for_r;
        }
    }
    worst
}

/// Spread: the mean distance between objective-space neighbours along the front.
///
/// The front is first ordered by its leading objective so the measure depends
/// only on the *set* of solutions, never on the order they happen to be stored
/// in (the previous in-place version read storage order, which made the metric
/// non-deterministic under reordering).
pub fn spread<T: Float>(front: &[Vec<T>]) -> T {
    if front.len() < 2 {
        return T::zero();
    }
    let mut ordered: Vec<&Vec<T>> = front.iter().collect();
    ordered.sort_by(|a, b| {
        let ka = a.first().copied().unwrap_or_else(T::zero);
        let kb = b.first().copied().unwrap_or_else(T::zero);
        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut total = T::zero();
    for window in ordered.windows(2) {
        let mut sq_sum = T::zero();
        for (a, b) in window[0].iter().zip(window[1].iter()) {
            let diff = *b - *a;
            sq_sum = sq_sum + diff * diff;
        }
        total = total + sq_sum.sqrt();
    }
    total / scirs2_core::numeric::NumCast::from((ordered.len() - 1) as f64).unwrap_or_else(T::one)
}

/// Spacing: the standard deviation of each solution's L1 distance to its nearest
/// neighbour. `0` means a perfectly uniformly distributed front.
pub fn spacing<T: Float>(front: &[Vec<T>]) -> T {
    if front.len() < 2 {
        return T::zero();
    }
    let mut nearest = Vec::with_capacity(front.len());
    for (i, a) in front.iter().enumerate() {
        let mut best = T::infinity();
        for (j, b) in front.iter().enumerate() {
            if i == j {
                continue;
            }
            let dims = a.len().min(b.len());
            let mut distance = T::zero();
            for k in 0..dims {
                distance = distance + (a[k] - b[k]).abs();
            }
            if distance < best {
                best = distance;
            }
        }
        if best.is_finite() {
            nearest.push(best);
        }
    }
    if nearest.len() < 2 {
        return T::zero();
    }
    let count: T = scirs2_core::numeric::NumCast::from(nearest.len() as f64).unwrap_or_else(T::one);
    let mut sum = T::zero();
    for value in &nearest {
        sum = sum + *value;
    }
    let mean = sum / count;
    let mut variance = T::zero();
    for value in &nearest {
        let diff = *value - mean;
        variance = variance + diff * diff;
    }
    (variance / count).sqrt()
}

/// Mean pairwise Euclidean distance between objective vectors — the diversity
/// measurement the engine adapters report.
///
/// `0.0` for fewer than two vectors (a single point has no spread). Vectors of
/// differing length are compared over their common prefix, which is the same
/// convention the rest of this module uses.
pub fn mean_pairwise_distance<T: Float>(vectors: &[Vec<T>]) -> f64 {
    if vectors.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let a = &vectors[i];
            let b = &vectors[j];
            let dims = a.len().min(b.len());
            let mut sum_sq = T::zero();
            for k in 0..dims {
                let diff = a[k] - b[k];
                sum_sq = sum_sq + diff * diff;
            }
            total += sum_sq.sqrt().to_f64().unwrap_or(0.0);
            count += 1;
        }
    }
    if count > 0 {
        total / count as f64
    } else {
        0.0
    }
}

/// Assemble a complete [`FrontMetrics`] from an already-measured hypervolume plus
/// the front and population it was measured on.
///
/// Every algorithm in this module latches its hypervolume reference point
/// differently (NSGA-II derives one from its first front, MOEA/D and NSGA-III
/// derive one from the archive), so the indicator itself is computed by the
/// caller and passed in; everything derived *from* the front's geometry is
/// computed here so all three algorithms report the same quantities under the
/// same names instead of each re-deriving them.
///
/// * `front` — the non-dominated solutions, in minimization space.
/// * `population` — everything the optimizer currently holds, in minimization
///   space; used for the additive epsilon indicator ("how far is the front from
///   covering what we have seen").
/// * `reference` — the hypervolume reference point in minimization space, or an
///   empty slice when none has been established.
/// * `previous_hypervolume` — the value from the previous update, or `None` on the
///   first one (which yields `convergence == 1`: nothing has been shown to
///   converge yet).
pub fn front_metrics_in_minimization_space<T: Float + std::fmt::Debug + Send + Sync + 'static>(
    front: &[Vec<T>],
    population: &[Vec<T>],
    reference: &[T],
    hypervolume: T,
    previous_hypervolume: Option<T>,
) -> FrontMetrics<T> {
    let convergence = match previous_hypervolume {
        Some(previous) => hypervolume_convergence(hypervolume, previous),
        None => T::one(),
    };
    FrontMetrics {
        hypervolume,
        spread: spread(front),
        spacing: spacing(front),
        convergence,
        num_solutions: front.len(),
        coverage: CoverageMetrics {
            objective_space_coverage: objective_space_coverage(front, reference),
            reference_distance: mean_reference_distance(front, reference),
            epsilon_dominance: additive_epsilon_indicator(front, population),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convergence_reports_one_before_anything_has_been_measured() {
        // First update: previous hypervolume is zero, so nothing has converged.
        assert_eq!(hypervolume_convergence(0.0_f64, 0.0), 1.0);
        assert_eq!(hypervolume_convergence(3.0_f64, 0.0), 1.0);
    }

    #[test]
    fn convergence_falls_to_zero_as_the_hypervolume_stops_moving() {
        // A 1% improvement.
        let small = hypervolume_convergence(1.01_f64, 1.0);
        assert!((small - 0.01 / 1.01).abs() < 1e-12, "got {small}");
        // No change at all: fully converged.
        assert_eq!(hypervolume_convergence(2.5_f64, 2.5), 0.0);
        // A large regression is clamped, never negative.
        let big = hypervolume_convergence(0.1_f64, 5.0);
        assert!(big > 0.9 && big <= 1.0, "got {big}");
    }

    #[test]
    fn coverage_is_the_real_box_fraction_not_a_constant() {
        // Front spanning half of each side of the reference box.
        let front = vec![vec![0.0, 0.5], vec![0.5, 0.0]];
        let coverage = objective_space_coverage(&front, &[1.0, 1.0]);
        assert!(
            (coverage - 0.25).abs() < 1e-12,
            "expected 0.5 * 0.5 = 0.25, got {coverage}"
        );
        assert_ne!(coverage, 0.5, "the old hardcoded 0.5 must be gone");

        // A single point spans no volume.
        assert_eq!(
            objective_space_coverage(&[vec![0.2, 0.2]], &[1.0, 1.0]),
            0.0
        );

        // A front reaching the reference in both objectives covers everything.
        let full = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert!((objective_space_coverage(&full, &[1.0, 1.0]) - 1.0).abs() < 1e-12);

        assert_eq!(objective_space_coverage::<f64>(&[], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn mean_reference_distance_is_a_real_average() {
        // (0,0) and (2,0) against reference (2,2): sqrt(8) and 2 -> mean.
        let front = vec![vec![0.0, 0.0], vec![2.0, 0.0]];
        let d = mean_reference_distance(&front, &[2.0, 2.0]);
        let expected = (8.0_f64.sqrt() + 2.0) / 2.0;
        assert!((d - expected).abs() < 1e-12, "got {d}");
        assert_eq!(mean_reference_distance::<f64>(&[], &[1.0]), 0.0);
    }

    #[test]
    fn epsilon_indicator_is_zero_when_the_front_dominates_the_population() {
        let front = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let population = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 3.0], // dominated by (1,0)
        ];
        assert_eq!(additive_epsilon_indicator(&front, &population), 0.0);
    }

    #[test]
    fn epsilon_indicator_measures_the_shortfall() {
        // The front member (1,1) needs a +1 shift in the second objective to
        // weakly dominate (5,0), and +0 for the rest.
        let front = vec![vec![1.0, 1.0]];
        let population = vec![vec![5.0, 0.0]];
        let eps = additive_epsilon_indicator(&front, &population);
        assert!((eps - 1.0).abs() < 1e-12, "got {eps}");
        assert_eq!(additive_epsilon_indicator::<f64>(&[], &population), 0.0);
    }

    #[test]
    fn spread_is_independent_of_storage_order() {
        let front = vec![vec![0.0, 2.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let shuffled = vec![vec![2.0, 0.0], vec![0.0, 2.0], vec![1.0, 1.0]];
        let a = spread(&front);
        let b = spread(&shuffled);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        // Each consecutive step is sqrt(2).
        assert!((a - 2.0_f64.sqrt()).abs() < 1e-12, "got {a}");
        assert_eq!(spread(&[vec![1.0, 1.0]]), 0.0);
    }

    #[test]
    fn spacing_is_zero_for_a_uniform_front() {
        let uniform = vec![
            vec![0.0, 3.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 0.0],
        ];
        assert!(spacing(&uniform) < 1e-12, "got {}", spacing(&uniform));

        let clustered = vec![vec![0.0, 9.0], vec![0.1, 8.9], vec![9.0, 0.0]];
        assert!(
            spacing(&clustered) > 0.0,
            "a clustered front must have non-zero spacing"
        );
    }

    #[test]
    fn front_bounds_returns_per_objective_extremes() {
        let front = vec![vec![1.0, 5.0], vec![3.0, 2.0], vec![-1.0, 4.0]];
        let (min_values, max_values) = front_bounds(&front).expect("bounds");
        assert_eq!(min_values, vec![-1.0, 2.0]);
        assert_eq!(max_values, vec![3.0, 5.0]);
        assert!(front_bounds::<f64>(&[]).is_none());
    }
}
