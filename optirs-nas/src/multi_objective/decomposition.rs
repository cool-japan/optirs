//! Decomposition primitives shared by the reference-direction based optimizers
//! in this module: MOEA/D ([`super::moead`]) and NSGA-III ([`super::nsga3`]).
//!
//! Everything here works in the **pure-minimization convention**. Callers holding
//! raw objective vectors (which may mix `Minimize` and `Maximize` directions) map
//! them first with [`super::hypervolume::normalize_front_for_minimization`].
//!
//! References:
//! * Das & Dennis, "Normal-Boundary Intersection" (SIAM J. Optim. 1998) — the
//!   structured simplex-lattice weight vectors used by both algorithms.
//! * Zhang & Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
//!   Decomposition" (IEEE TEVC 2007) — weighted-sum / Tchebycheff / PBI
//!   scalarizations and the neighbourhood structure.
//! * Deb & Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
//!   Reference-Point-Based Nondominated Sorting Approach, Part I" (IEEE TEVC 2014)
//!   — the adaptive ideal-point / hyperplane-intercept normalization and the
//!   niche-preserving selection NSGA-III uses.

use scirs2_core::numeric::Float;

/// Smallest weight component a scalarization will divide by. Structured
/// Das-Dennis lattices contain vectors with exact zeros (the simplex corners);
/// Tchebycheff and the achievement scalarizing function divide by (or scale with)
/// each component, so an exact zero would either drop an objective entirely or
/// produce an infinity. Flooring at this value is the standard remedy and keeps
/// the corner subproblems meaningful.
pub const MIN_WEIGHT_COMPONENT: f64 = 1e-6;

/// Penalty factor `theta` of the penalty-based boundary intersection
/// scalarization. `5.0` is the value used throughout the MOEA/D literature.
pub const PBI_PENALTY: f64 = 5.0;

/// Upper bound on the number of simplex-lattice divisions
/// [`divisions_for_at_least`] will consider before giving up. Guards the search
/// against pathological targets; the resulting lattice already has well over a
/// million vectors for three objectives.
const MAX_LATTICE_DIVISIONS: usize = 2000;

/// Number of Das-Dennis weight vectors for `num_objectives` objectives and
/// `divisions` divisions per axis, i.e. `C(divisions + m - 1, m - 1)`.
///
/// Returns `None` on overflow instead of wrapping, so callers can report an
/// honest configuration error rather than sizing a population from a wrapped
/// count.
pub fn lattice_size(num_objectives: usize, divisions: usize) -> Option<usize> {
    if num_objectives == 0 {
        return Some(0);
    }
    if num_objectives == 1 {
        return Some(1);
    }
    // C(n, k) with n = divisions + m - 1, k = m - 1, computed multiplicatively so
    // intermediate values stay as small as possible.
    let n = divisions.checked_add(num_objectives - 1)?;
    let k = num_objectives - 1;
    let k = k.min(n.checked_sub(k)?);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.checked_mul(n - i)?;
        result /= i + 1;
    }
    Some(result)
}

/// Smallest number of divisions whose Das-Dennis lattice has at least `target`
/// weight vectors, together with that lattice's size.
///
/// MOEA/D's population size *is* the number of subproblems, and NSGA-III's
/// population is sized to the number of reference directions, so both need to go
/// from a requested population size to a lattice that can supply it.
pub fn divisions_for_at_least(num_objectives: usize, target: usize) -> Option<(usize, usize)> {
    if num_objectives == 0 {
        return None;
    }
    if num_objectives == 1 {
        return Some((1, 1));
    }
    for divisions in 1..=MAX_LATTICE_DIVISIONS {
        let size = lattice_size(num_objectives, divisions)?;
        if size >= target {
            return Some((divisions, size));
        }
    }
    None
}

/// Structured Das-Dennis weight vectors: every vector of non-negative multiples
/// of `1/divisions` whose components sum to exactly one.
///
/// The lattice is deterministic and covers the objective simplex uniformly, which
/// is what makes decomposition-based search reproducible and its subproblems
/// evenly spread. `divisions` is treated as at least one (a zero-division lattice
/// would be the all-zero vector, which is not a weight vector).
pub fn das_dennis_weights<T: Float>(num_objectives: usize, divisions: usize) -> Vec<Vec<T>> {
    if num_objectives == 0 {
        return Vec::new();
    }
    if num_objectives == 1 {
        return vec![vec![T::one()]];
    }
    let divisions = divisions.max(1);
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::with_capacity(num_objectives);
    lattice_recurse(num_objectives, divisions, &mut current, &mut out);

    let denominator = T::from(divisions).unwrap_or_else(T::one);
    out.into_iter()
        .map(|counts| {
            counts
                .into_iter()
                .map(|count| T::from(count).unwrap_or_else(T::zero) / denominator)
                .collect()
        })
        .collect()
}

/// Enumerate the integer compositions of `left` into `num_objectives` parts.
fn lattice_recurse(
    num_objectives: usize,
    left: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if current.len() + 1 == num_objectives {
        current.push(left);
        out.push(current.clone());
        current.pop();
        return;
    }
    for share in 0..=left {
        current.push(share);
        lattice_recurse(num_objectives, left - share, current, out);
        current.pop();
    }
}

/// Indices of the `size` weight vectors closest (Euclidean) to each weight
/// vector, itself included — the MOEA/D neighbourhood structure `B(i)`.
///
/// Ties are broken by index so the neighbourhoods are deterministic.
pub fn weight_neighborhoods<T: Float>(weights: &[Vec<T>], size: usize) -> Vec<Vec<usize>> {
    let size = size.max(1).min(weights.len().max(1));
    (0..weights.len())
        .map(|i| {
            let mut ordered: Vec<(usize, f64)> = (0..weights.len())
                .map(|j| (j, squared_distance(&weights[i], &weights[j])))
                .collect();
            ordered.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            ordered
                .into_iter()
                .take(size)
                .map(|(index, _)| index)
                .collect()
        })
        .collect()
}

fn squared_distance<T: Float>(a: &[T], b: &[T]) -> f64 {
    let len = a.len().min(b.len());
    let mut total = 0.0;
    for index in 0..len {
        let diff = (a[index] - b[index]).to_f64().unwrap_or(0.0);
        total += diff * diff;
    }
    total
}

/// Scalarization used to reduce a subproblem's objective vector to a single
/// comparable cost. Mirrors [`super::moead::DecompositionMethod`]; kept separate
/// so this module has no dependency on the optimizer that configures it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalarization {
    /// `sum_i w_i * (f_i - z*_i)`.
    WeightedSum,
    /// `max_i w_i * |f_i - z*_i|`.
    Tchebycheff,
    /// `d1 + theta * d2` along and perpendicular to the weight direction.
    PenaltyBoundaryIntersection,
    /// `max_i (f_i - z*_i) / w_i`.
    AchievementScalarizing,
}

/// Scalarized cost of `objectives` for the subproblem defined by `weight`,
/// relative to the ideal point `ideal`. **Lower is better** for every variant.
///
/// All inputs are in minimization space. A missing ideal-point entry is treated
/// as zero, and every weight component is floored at
/// [`MIN_WEIGHT_COMPONENT`] so the simplex corners remain usable.
pub fn scalarize<T: Float>(
    method: Scalarization,
    objectives: &[T],
    weight: &[T],
    ideal: &[T],
) -> T {
    let len = objectives.len().min(weight.len());
    if len == 0 {
        return T::zero();
    }
    let floor = T::from(MIN_WEIGHT_COMPONENT).unwrap_or_else(T::zero);
    let translated: Vec<T> = (0..len)
        .map(|i| objectives[i] - ideal.get(i).copied().unwrap_or_else(T::zero))
        .collect();

    match method {
        Scalarization::WeightedSum => {
            let mut total = T::zero();
            for i in 0..len {
                total = total + weight[i] * translated[i];
            }
            total
        }
        Scalarization::Tchebycheff => {
            let mut worst = T::neg_infinity();
            for i in 0..len {
                let scaled = weight[i].max(floor) * translated[i].abs();
                if scaled > worst {
                    worst = scaled;
                }
            }
            if worst.is_finite() {
                worst
            } else {
                T::zero()
            }
        }
        Scalarization::AchievementScalarizing => {
            let mut worst = T::neg_infinity();
            for i in 0..len {
                let scaled = translated[i] / weight[i].max(floor);
                if scaled > worst {
                    worst = scaled;
                }
            }
            if worst.is_finite() {
                worst
            } else {
                T::zero()
            }
        }
        Scalarization::PenaltyBoundaryIntersection => {
            let norm = {
                let mut sum = T::zero();
                for i in 0..len {
                    sum = sum + weight[i] * weight[i];
                }
                sum.sqrt()
            };
            if norm <= T::zero() {
                return T::zero();
            }
            let mut dot = T::zero();
            for i in 0..len {
                dot = dot + translated[i] * weight[i];
            }
            let d1 = dot / norm;
            let mut perpendicular_sq = T::zero();
            for i in 0..len {
                let along = weight[i] / norm * d1;
                let diff = translated[i] - along;
                perpendicular_sq = perpendicular_sq + diff * diff;
            }
            let d2 = perpendicular_sq.sqrt();
            let theta = T::from(PBI_PENALTY).unwrap_or_else(T::one);
            d1 + theta * d2
        }
    }
}

/// Fold `objectives` into a running ideal point, extending it when the point has
/// more objectives than have been seen so far. Returns `true` when the ideal
/// point moved.
pub fn update_ideal_point<T: Float>(ideal: &mut Vec<T>, objectives: &[T]) -> bool {
    let mut moved = false;
    if ideal.len() < objectives.len() {
        ideal.resize(objectives.len(), T::infinity());
        moved = true;
    }
    for (index, value) in objectives.iter().enumerate() {
        if *value < ideal[index] {
            ideal[index] = *value;
            moved = true;
        }
    }
    moved
}

/// Perpendicular (orthogonal) distance from `point` to the ray through the
/// origin with direction `direction`. This is the distance NSGA-III associates
/// solutions to reference directions by.
pub fn perpendicular_distance<T: Float>(point: &[T], direction: &[T]) -> T {
    let len = point.len().min(direction.len());
    if len == 0 {
        return T::zero();
    }
    let mut norm_sq = T::zero();
    for i in 0..len {
        norm_sq = norm_sq + direction[i] * direction[i];
    }
    if norm_sq <= T::zero() {
        // A zero direction defines no ray; the honest distance is the point's own
        // norm, which is what an all-zero reference direction would measure.
        let mut point_norm_sq = T::zero();
        for i in 0..len {
            point_norm_sq = point_norm_sq + point[i] * point[i];
        }
        return point_norm_sq.sqrt();
    }
    let norm = norm_sq.sqrt();
    let mut dot = T::zero();
    for i in 0..len {
        dot = dot + point[i] * direction[i];
    }
    let projection = dot / norm;
    let mut residual_sq = T::zero();
    for i in 0..len {
        let along = direction[i] / norm * projection;
        let diff = point[i] - along;
        residual_sq = residual_sq + diff * diff;
    }
    residual_sq.sqrt()
}

/// Per-objective minima of `points` — the ideal point of a set.
pub fn ideal_point_of<T: Float>(points: &[Vec<T>]) -> Option<Vec<T>> {
    let num_objectives = points.iter().map(|p| p.len()).max()?;
    if num_objectives == 0 {
        return None;
    }
    let mut ideal = vec![T::infinity(); num_objectives];
    for point in points {
        for (index, value) in point.iter().enumerate() {
            if *value < ideal[index] {
                ideal[index] = *value;
            }
        }
    }
    if ideal.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(ideal)
}

/// The `m` extreme points of a translated (ideal-point-subtracted) set: for each
/// axis `j`, the point minimizing the achievement scalarizing function with a
/// weight vector concentrated on `j`.
///
/// Returns the indices into `translated`, one per objective.
pub fn extreme_point_indices<T: Float>(translated: &[Vec<T>], num_objectives: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(num_objectives);
    for axis in 0..num_objectives {
        let mut weight =
            vec![T::from(MIN_WEIGHT_COMPONENT).unwrap_or_else(T::zero); num_objectives];
        if axis < weight.len() {
            weight[axis] = T::one();
        }
        let mut best_index = 0usize;
        let mut best_value = T::infinity();
        for (index, point) in translated.iter().enumerate() {
            let value = scalarize(Scalarization::AchievementScalarizing, point, &weight, &[]);
            if value < best_value {
                best_value = value;
                best_index = index;
            }
        }
        indices.push(best_index);
    }
    indices
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting.
///
/// Returns `None` when the matrix is singular to working precision, which is the
/// signal NSGA-III uses to fall back from hyperplane intercepts to the observed
/// per-objective maxima.
pub fn solve_linear_system<T: Float>(matrix: &[Vec<T>], rhs: &[T]) -> Option<Vec<T>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) || n == 0 {
        return None;
    }
    let mut augmented: Vec<Vec<T>> = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, value)| {
            let mut new_row = row.clone();
            new_row.push(*value);
            new_row
        })
        .collect();

    for column in 0..n {
        // Partial pivot.
        let mut pivot_row = column;
        let mut pivot_magnitude = augmented[column][column].abs();
        for row in (column + 1)..n {
            let magnitude = augmented[row][column].abs();
            if magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if pivot_magnitude <= T::from(1e-12).unwrap_or_else(T::zero) {
            return None;
        }
        if pivot_row != column {
            augmented.swap(pivot_row, column);
        }
        let pivot = augmented[column][column];
        for row in (column + 1)..n {
            let factor = augmented[row][column] / pivot;
            if factor == T::zero() {
                continue;
            }
            for col in column..=n {
                augmented[row][col] = augmented[row][col] - factor * augmented[column][col];
            }
        }
    }

    let mut solution = vec![T::zero(); n];
    for row in (0..n).rev() {
        let mut accumulated = augmented[row][n];
        for col in (row + 1)..n {
            accumulated = accumulated - augmented[row][col] * solution[col];
        }
        let pivot = augmented[row][row];
        if pivot == T::zero() {
            return None;
        }
        solution[row] = accumulated / pivot;
    }
    Some(solution)
}

/// NSGA-III's adaptive normalization: translate `points` by their ideal point and
/// divide each objective by the intercept of the hyperplane through the `m`
/// extreme points.
///
/// Returns the normalized points together with the intercepts actually used. When
/// the extreme points are degenerate (fewer than `m` distinct ones, or a singular
/// system, or a non-positive intercept) the per-objective maxima are used
/// instead — this is the documented fallback in the original paper, not a
/// fabricated value.
pub fn normalize_by_hyperplane<T: Float>(
    points: &[Vec<T>],
    num_objectives: usize,
) -> Option<(Vec<Vec<T>>, Vec<T>)> {
    if points.is_empty() || num_objectives == 0 {
        return None;
    }
    let ideal = ideal_point_of(points)?;
    let translated: Vec<Vec<T>> = points
        .iter()
        .map(|point| {
            (0..num_objectives)
                .map(|index| {
                    point.get(index).copied().unwrap_or_else(T::zero)
                        - ideal.get(index).copied().unwrap_or_else(T::zero)
                })
                .collect()
        })
        .collect();

    let mut intercepts = fallback_intercepts(&translated, num_objectives);
    let extreme = extreme_point_indices(&translated, num_objectives);
    let distinct: std::collections::BTreeSet<usize> = extreme.iter().copied().collect();
    if distinct.len() == num_objectives {
        let matrix: Vec<Vec<T>> = extreme
            .iter()
            .map(|index| translated[*index].clone())
            .collect();
        let rhs = vec![T::one(); num_objectives];
        if let Some(plane) = solve_linear_system(&matrix, &rhs) {
            let candidate: Vec<T> = plane
                .iter()
                .map(|coefficient| {
                    if *coefficient > T::zero() {
                        T::one() / *coefficient
                    } else {
                        T::zero()
                    }
                })
                .collect();
            let epsilon = T::from(1e-10).unwrap_or_else(T::zero);
            if candidate.iter().all(|value| *value > epsilon) {
                intercepts = candidate;
            }
        }
    }

    let normalized: Vec<Vec<T>> = translated
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(intercepts.iter())
                .map(|(value, intercept)| {
                    if *intercept > T::zero() {
                        *value / *intercept
                    } else {
                        *value
                    }
                })
                .collect()
        })
        .collect();
    Some((normalized, intercepts))
}

/// Per-objective maxima of a translated set, floored so a degenerate axis does
/// not divide by zero.
fn fallback_intercepts<T: Float>(translated: &[Vec<T>], num_objectives: usize) -> Vec<T> {
    let floor = T::from(1e-10).unwrap_or_else(T::zero);
    (0..num_objectives)
        .map(|axis| {
            let mut maximum = T::zero();
            for point in translated {
                if let Some(value) = point.get(axis) {
                    if *value > maximum {
                        maximum = *value;
                    }
                }
            }
            if maximum > floor {
                maximum
            } else {
                T::one()
            }
        })
        .collect()
}

/// Association of each point with its nearest reference direction:
/// `(direction index, perpendicular distance)`.
pub fn associate_with_references<T: Float>(
    normalized: &[Vec<T>],
    references: &[Vec<T>],
) -> Vec<(usize, T)> {
    normalized
        .iter()
        .map(|point| {
            let mut best_index = 0usize;
            let mut best_distance = T::infinity();
            for (index, reference) in references.iter().enumerate() {
                let distance = perpendicular_distance(point, reference);
                if distance < best_distance {
                    best_distance = distance;
                    best_index = index;
                }
            }
            (best_index, best_distance)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn das_dennis_lattice_has_the_published_size_and_sums_to_one() {
        // C(H + m - 1, m - 1): m = 3, H = 4 -> C(6, 2) = 15.
        let weights = das_dennis_weights::<f64>(3, 4);
        assert_eq!(weights.len(), 15);
        assert_eq!(lattice_size(3, 4), Some(15));
        for weight in &weights {
            assert_eq!(weight.len(), 3);
            let total: f64 = weight.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-12,
                "weight did not sum to 1: {weight:?}"
            );
            assert!(weight.iter().all(|value| *value >= 0.0));
        }
        // m = 2, H = 4 -> 5 evenly spaced vectors on the line w1 + w2 = 1.
        let two = das_dennis_weights::<f64>(2, 4);
        assert_eq!(two.len(), 5);
        let mut firsts: Vec<f64> = two.iter().map(|weight| weight[0]).collect();
        firsts.sort_by(|a, b| a.partial_cmp(b).expect("finite weights"));
        for (index, value) in firsts.iter().enumerate() {
            assert!((value - index as f64 * 0.25).abs() < 1e-12);
        }
        // The lattice is a set: no duplicates.
        let mut keys: Vec<String> = weights
            .iter()
            .map(|weight| format!("{:?}", weight))
            .collect();
        keys.sort();
        let before = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), before);
    }

    #[test]
    fn lattice_size_matches_enumeration_and_reports_overflow() {
        for objectives in 1..=4 {
            for divisions in 1..=6 {
                assert_eq!(
                    lattice_size(objectives, divisions),
                    Some(das_dennis_weights::<f64>(objectives, divisions).len()),
                    "m={objectives} H={divisions}"
                );
            }
        }
        assert_eq!(lattice_size(30, usize::MAX), None);
    }

    #[test]
    fn divisions_for_at_least_is_the_minimal_lattice() {
        let (divisions, size) = divisions_for_at_least(3, 12).expect("lattice exists");
        assert_eq!((divisions, size), (4, 15));
        let (smaller, smaller_size) =
            (divisions - 1, lattice_size(3, divisions - 1).expect("size"));
        assert!(smaller_size < 12, "H={smaller} should not already suffice");
        assert_eq!(divisions_for_at_least(1, 50), Some((1, 1)));
    }

    #[test]
    fn neighborhoods_contain_self_and_the_nearest_weights() {
        let weights = das_dennis_weights::<f64>(2, 4);
        let neighborhoods = weight_neighborhoods(&weights, 3);
        assert_eq!(neighborhoods.len(), weights.len());
        for (index, neighborhood) in neighborhoods.iter().enumerate() {
            assert_eq!(neighborhood.len(), 3);
            assert_eq!(neighborhood[0], index, "self must be the nearest weight");
            // Distances are non-decreasing along the neighbourhood.
            let mut previous = 0.0;
            for neighbor in neighborhood {
                let distance = squared_distance(&weights[index], &weights[*neighbor]);
                assert!(distance >= previous - 1e-12);
                previous = distance;
            }
        }
        // Requesting more neighbours than exist is clamped, not a panic.
        assert_eq!(weight_neighborhoods(&weights, 99)[0].len(), weights.len());
    }

    #[test]
    fn scalarizations_have_their_textbook_values() {
        let weight = vec![0.5f64, 0.5];
        let ideal = vec![0.0f64, 0.0];
        let point = vec![2.0f64, 4.0];

        assert!(
            (scalarize(Scalarization::WeightedSum, &point, &weight, &ideal) - 3.0).abs() < 1e-12
        );
        // max(0.5*2, 0.5*4) = 2
        assert!(
            (scalarize(Scalarization::Tchebycheff, &point, &weight, &ideal) - 2.0).abs() < 1e-12
        );
        // max(2/0.5, 4/0.5) = 8
        assert!(
            (scalarize(
                Scalarization::AchievementScalarizing,
                &point,
                &weight,
                &ideal
            ) - 8.0)
                .abs()
                < 1e-12
        );
        // w/|w| = (0.7071, 0.7071); d1 = (2+4)/sqrt(2) = 4.2426;
        // along = (3, 3); d2 = ||(-1, 1)|| = 1.4142; g = 4.2426 + 5*1.4142 = 11.3137
        let pbi = scalarize(
            Scalarization::PenaltyBoundaryIntersection,
            &point,
            &weight,
            &ideal,
        );
        assert!((pbi - 11.313708498984761).abs() < 1e-9, "pbi = {pbi}");
    }

    #[test]
    fn scalarizations_survive_a_simplex_corner_weight() {
        let weight = vec![1.0f64, 0.0];
        let ideal = vec![0.0f64, 0.0];
        let point = vec![1.0f64, 3.0];
        for method in [
            Scalarization::WeightedSum,
            Scalarization::Tchebycheff,
            Scalarization::AchievementScalarizing,
            Scalarization::PenaltyBoundaryIntersection,
        ] {
            let value = scalarize(method, &point, &weight, &ideal);
            assert!(value.is_finite(), "{method:?} produced {value}");
        }
        // The zero component is floored rather than dropped, so the ignored
        // objective still breaks ties between points that agree on the weighted
        // one. (The floor is deliberately tiny: it must not outrank the objective
        // the corner subproblem is actually about.)
        let tied = vec![0.0f64, 3.0];
        let tied_but_worse = vec![0.0f64, 300.0];
        assert!(
            scalarize(Scalarization::Tchebycheff, &tied_but_worse, &weight, &ideal)
                > scalarize(Scalarization::Tchebycheff, &tied, &weight, &ideal)
        );
        // ... and it does not: the corner subproblem still prefers the point that
        // is better on its own objective, however bad the other one is.
        assert!(
            scalarize(Scalarization::Tchebycheff, &tied_but_worse, &weight, &ideal)
                < scalarize(Scalarization::Tchebycheff, &point, &weight, &ideal)
        );
    }

    #[test]
    fn scalarization_respects_the_ideal_point_translation() {
        let weight = vec![0.5f64, 0.5];
        let point = vec![2.0f64, 4.0];
        let shifted_ideal = vec![1.0f64, 1.0];
        // Tchebycheff on (1, 3) -> max(0.5, 1.5) = 1.5
        assert!(
            (scalarize(Scalarization::Tchebycheff, &point, &weight, &shifted_ideal) - 1.5).abs()
                < 1e-12
        );
    }

    #[test]
    fn ideal_point_tracks_the_minimum_per_objective() {
        let mut ideal: Vec<f64> = Vec::new();
        assert!(update_ideal_point(&mut ideal, &[3.0, 5.0]));
        assert_eq!(ideal, vec![3.0, 5.0]);
        assert!(update_ideal_point(&mut ideal, &[4.0, 1.0]));
        assert_eq!(ideal, vec![3.0, 1.0]);
        assert!(!update_ideal_point(&mut ideal, &[9.0, 9.0]));
        assert_eq!(ideal, vec![3.0, 1.0]);
    }

    #[test]
    fn perpendicular_distance_is_zero_on_the_ray_and_the_height_otherwise() {
        let direction = vec![1.0f64, 1.0];
        assert!(perpendicular_distance(&[2.0, 2.0], &direction).abs() < 1e-12);
        // (1, 0) against the diagonal: height = 1/sqrt(2)
        let distance = perpendicular_distance(&[1.0f64, 0.0], &direction);
        assert!((distance - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        // A zero direction reports the point's own norm rather than NaN.
        assert!((perpendicular_distance(&[3.0f64, 4.0], &[0.0, 0.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn linear_solver_matches_a_hand_computed_solution_and_detects_singularity() {
        let matrix = vec![vec![2.0f64, 1.0], vec![1.0, 3.0]];
        let rhs = vec![5.0f64, 10.0];
        let solution = solve_linear_system(&matrix, &rhs).expect("non-singular");
        assert!((solution[0] - 1.0).abs() < 1e-12, "{solution:?}");
        assert!((solution[1] - 3.0).abs() < 1e-12, "{solution:?}");

        let singular = vec![vec![1.0f64, 2.0], vec![2.0, 4.0]];
        assert!(solve_linear_system(&singular, &vec![1.0, 2.0]).is_none());
    }

    #[test]
    fn hyperplane_normalization_maps_the_extreme_points_onto_the_unit_simplex() {
        // A linear front between (0, 4) and (2, 0): the hyperplane through the two
        // extreme points has intercepts (2, 4) after ideal-point translation.
        let front = vec![vec![0.0f64, 4.0], vec![1.0, 2.0], vec![2.0, 0.0]];
        let (normalized, intercepts) =
            normalize_by_hyperplane(&front, 2).expect("normalization succeeds");
        assert!((intercepts[0] - 2.0).abs() < 1e-9, "{intercepts:?}");
        assert!((intercepts[1] - 4.0).abs() < 1e-9, "{intercepts:?}");
        // Extremes land on the unit axes, the middle point on the unit simplex.
        assert!((normalized[0][0] - 0.0).abs() < 1e-9);
        assert!((normalized[0][1] - 1.0).abs() < 1e-9);
        assert!((normalized[2][0] - 1.0).abs() < 1e-9);
        assert!((normalized[2][1] - 0.0).abs() < 1e-9);
        let middle_sum: f64 = normalized[1].iter().sum();
        assert!((middle_sum - 1.0).abs() < 1e-9, "{:?}", normalized[1]);
    }

    #[test]
    fn hyperplane_normalization_falls_back_when_the_front_is_degenerate() {
        // A single point cannot define m distinct extreme points.
        let front = vec![vec![3.0f64, 7.0]];
        let (normalized, intercepts) = normalize_by_hyperplane(&front, 2).expect("fallback");
        assert!(intercepts.iter().all(|value| *value > 0.0));
        assert!(normalized[0].iter().all(|value| value.is_finite()));
        // Identical points: still finite, never NaN.
        let flat = vec![vec![1.0f64, 1.0], vec![1.0, 1.0]];
        let (normalized, _) = normalize_by_hyperplane(&flat, 2).expect("fallback");
        assert!(normalized
            .iter()
            .all(|point| point.iter().all(|value| value.is_finite())));
    }

    #[test]
    fn association_picks_the_nearest_reference_direction() {
        let references = das_dennis_weights::<f64>(2, 4);
        // Exactly on the (0.25, 0.75) direction.
        let points = vec![vec![0.25f64, 0.75], vec![1.0, 0.0]];
        let associations = associate_with_references(&points, &references);
        assert!(associations[0].1.abs() < 1e-12);
        assert!(
            (references[associations[0].0][0] - 0.25).abs() < 1e-12,
            "associated with {:?}",
            references[associations[0].0]
        );
        assert!((references[associations[1].0][0] - 1.0).abs() < 1e-12);
    }
}
