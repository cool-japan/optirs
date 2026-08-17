// Robust aggregation primitives for Byzantine-tolerant federated averaging.
//
// This module holds the actual estimators behind
// [`super::byzantine_aggregation::ByzantineRobustAggregator`]. They live in
// their own file so that neither file approaches the 2000-line limit, and so
// that each estimator can be unit-tested against a hand-computed value
// without going through the aggregator's configuration plumbing.
//
// References
// ----------
//   * Yin, D., Chen, Y., Ramchandran, K., Bartlett, P. "Byzantine-Robust
//     Distributed Learning: Towards Optimal Statistical Rates." ICML 2018.
//     (coordinate-wise median and trimmed mean)
//   * Blanchard, P., El Mhamdi, E. M., Guerraoui, R., Stainer, J. "Machine
//     Learning with Adversaries: Byzantine Tolerant Gradient Descent."
//     NeurIPS 2017. (Krum / Multi-Krum)
//   * El Mhamdi, E. M., Guerraoui, R., Rouault, S. "The Hidden Vulnerability
//     of Distributed Learning in Byzantium." ICML 2018. (Bulyan)
//   * Karimireddy, S. P., He, L., Jaggi, M. "Learning from History for
//     Byzantine Robust Optimization." ICML 2021. (centered clipping)
//
// Determinism
// -----------
// Every entry point takes an already-ordered cohort produced by
// [`ordered_cohort`], which sorts by client id. No result in this module
// depends on `HashMap` iteration order, and ties in any selection step are
// broken by cohort position (hence by client id).

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

/// One cohort member: a client id paired with its update vector.
pub type CohortMember<'a, T> = (&'a str, &'a Array1<T>);

/// Default number of centered-clipping refinement iterations.
///
/// Karimireddy et al. report that the fixed-point iteration is already
/// accurate after a handful of steps; three is the value used in the paper's
/// experiments.
pub const CENTERED_CLIPPING_ITERATIONS: usize = 3;

/// Order a map of client updates deterministically (by client id) and reject
/// malformed cohorts up front.
///
/// Validation performed here, once, so that every estimator below can assume
/// a well-formed input:
///   * the cohort is non-empty,
///   * the update dimension is non-zero and identical for every client,
///   * every coordinate is finite.
///
/// The finiteness check matters: a single NaN would otherwise silently
/// corrupt every comparison-based estimator (`partial_cmp` returns `None`,
/// and the usual `unwrap_or(Ordering::Equal)` fallback turns the sort into
/// garbage rather than an error).
pub fn ordered_cohort<T: Float + Debug + Send + Sync + 'static>(
    client_updates: &HashMap<String, Array1<T>>,
) -> Result<Vec<CohortMember<'_, T>>> {
    if client_updates.is_empty() {
        return Err(OptimError::InvalidConfig(
            "no client updates provided".to_string(),
        ));
    }

    let mut cohort: Vec<CohortMember<'_, T>> = client_updates
        .iter()
        .map(|(id, update)| (id.as_str(), update))
        .collect();
    cohort.sort_by(|a, b| a.0.cmp(b.0));

    let dim = cohort[0].1.len();
    if dim == 0 {
        return Err(OptimError::InvalidConfig(
            "client updates must have non-zero dimension".to_string(),
        ));
    }
    for (id, update) in cohort.iter() {
        if update.len() != dim {
            return Err(OptimError::DimensionMismatch(format!(
                "client {id} submitted {} values but the cohort dimension is {dim}",
                update.len()
            )));
        }
        for value in update.iter() {
            if !value.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "client {id} submitted a non-finite value ({value:?}); robust estimators \
                     cannot order non-finite updates"
                )));
            }
        }
    }
    Ok(cohort)
}

/// Dimension of a cohort validated by [`ordered_cohort`].
fn cohort_dim<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
) -> Result<usize> {
    cohort
        .first()
        .map(|(_, update)| update.len())
        .ok_or_else(|| OptimError::InvalidConfig("empty cohort".to_string()))
}

/// Total ordering over finite floats. Safe because [`ordered_cohort`] has
/// already rejected non-finite values.
fn cmp_finite<T: Float>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

fn as_f64<T: Float + Debug>(value: T) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        OptimError::ComputationError(format!("value {value:?} is not representable as f64"))
    })
}

fn from_f64<T: Float + Debug>(value: f64) -> Result<T> {
    T::from(value).ok_or_else(|| {
        OptimError::ComputationError(format!(
            "{value} is not representable in the target float type"
        ))
    })
}

/// Plain coordinate-wise arithmetic mean (FedAvg). Not Byzantine-robust; used
/// as the averaging step of methods that first *filter* the cohort.
pub fn mean<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
) -> Result<Array1<T>> {
    let dim = cohort_dim(cohort)?;
    let count = from_f64::<T>(cohort.len() as f64)?;
    let mut acc: Array1<T> = Array1::zeros(dim);
    for (_, update) in cohort.iter() {
        acc = acc + *update;
    }
    Ok(acc.mapv(|value| value / count))
}

/// Coordinate-wise weighted mean. `weights` must be positionally aligned with
/// `cohort`, non-negative, and not all zero.
pub fn weighted_mean<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    weights: &[f64],
) -> Result<Array1<T>> {
    let dim = cohort_dim(cohort)?;
    if weights.len() != cohort.len() {
        return Err(OptimError::DimensionMismatch(format!(
            "{} weights supplied for a cohort of {} clients",
            weights.len(),
            cohort.len()
        )));
    }
    let mut total = 0.0_f64;
    for &weight in weights.iter() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "aggregation weights must be finite and non-negative, got {weight}"
            )));
        }
        total += weight;
    }
    if total <= 0.0 {
        return Err(OptimError::InvalidParameter(
            "aggregation weights sum to zero; no client would contribute to the aggregate"
                .to_string(),
        ));
    }

    let mut acc: Array1<T> = Array1::zeros(dim);
    for ((_, update), &weight) in cohort.iter().zip(weights.iter()) {
        let scale = from_f64::<T>(weight / total)?;
        acc = acc + update.mapv(|value| value * scale);
    }
    Ok(acc)
}

/// Coordinate-wise median (Yin et al. 2018).
pub fn coordinate_wise_median<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
) -> Result<Array1<T>> {
    let dim = cohort_dim(cohort)?;
    let two = from_f64::<T>(2.0)?;
    let mut result = Array1::zeros(dim);
    let mut column: Vec<T> = Vec::with_capacity(cohort.len());
    for coordinate in 0..dim {
        column.clear();
        column.extend(cohort.iter().map(|(_, update)| update[coordinate]));
        column.sort_by(cmp_finite);
        let middle = column.len() / 2;
        result[coordinate] = if column.len().is_multiple_of(2) {
            (column[middle - 1] + column[middle]) / two
        } else {
            column[middle]
        };
    }
    Ok(result)
}

/// Number of values to remove from *each* tail for a trimmed mean of
/// `cohort_size` clients at ratio `trim_ratio`.
///
/// `trim_ratio` is the total fraction removed, split evenly between the two
/// tails, and is capped so that at least one value always survives. Returning
/// the count (rather than trimming inside the estimator) lets the caller
/// report exactly how much was trimmed.
pub fn trim_count_for_ratio(cohort_size: usize, trim_ratio: f64) -> Result<usize> {
    if !(0.0..1.0).contains(&trim_ratio) || !trim_ratio.is_finite() {
        return Err(OptimError::InvalidConfig(format!(
            "trim_ratio must be in [0, 1), got {trim_ratio}"
        )));
    }
    if cohort_size == 0 {
        return Err(OptimError::InvalidConfig(
            "cannot trim an empty cohort".to_string(),
        ));
    }
    let requested = ((cohort_size as f64 * trim_ratio) / 2.0).floor() as usize;
    // Keep at least one surviving value.
    let max_per_tail = (cohort_size - 1) / 2;
    Ok(requested.min(max_per_tail))
}

/// Coordinate-wise trimmed mean (Yin et al. 2018): drop `trim_count` values
/// from each tail of every coordinate, then average what remains.
pub fn coordinate_wise_trimmed_mean<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    trim_count: usize,
) -> Result<Array1<T>> {
    let dim = cohort_dim(cohort)?;
    let n = cohort.len();
    if 2 * trim_count >= n {
        return Err(OptimError::InvalidConfig(format!(
            "trimming {trim_count} values from each tail of a {n}-client cohort would leave \
             nothing to average"
        )));
    }
    let kept = n - 2 * trim_count;
    let denominator = from_f64::<T>(kept as f64)?;

    let mut result = Array1::zeros(dim);
    let mut column: Vec<T> = Vec::with_capacity(n);
    for coordinate in 0..dim {
        column.clear();
        column.extend(cohort.iter().map(|(_, update)| update[coordinate]));
        column.sort_by(cmp_finite);
        let mut sum = T::zero();
        for value in column[trim_count..n - trim_count].iter() {
            sum = sum + *value;
        }
        result[coordinate] = sum / denominator;
    }
    Ok(result)
}

/// Full matrix of squared Euclidean distances between cohort members.
///
/// `d[i][j] == d[j][i]` and `d[i][i] == 0`.
pub fn pairwise_squared_distances<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
) -> Result<Vec<Vec<f64>>> {
    let n = cohort.len();
    let mut distances = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sum = 0.0_f64;
            for (a, b) in cohort[i].1.iter().zip(cohort[j].1.iter()) {
                let diff = as_f64(*a)? - as_f64(*b)?;
                sum += diff * diff;
            }
            distances[i][j] = sum;
            distances[j][i] = sum;
        }
    }
    Ok(distances)
}

/// Krum scores (Blanchard et al. 2017).
///
/// `score(i)` is the sum of the `n - f - 2` smallest squared distances from
/// client `i` to the other clients. A low score means "close to many peers",
/// which is what Krum treats as evidence of honesty.
///
/// Requires `n >= f + 3` so that at least one neighbour is summed; anything
/// less makes the score meaningless and is reported as a configuration error
/// rather than silently degraded.
pub fn krum_scores<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    f: usize,
) -> Result<Vec<f64>> {
    let n = cohort.len();
    if n < f + 3 {
        return Err(OptimError::InvalidConfig(format!(
            "Krum with f = {f} Byzantine clients needs at least {} participants, got {n}",
            f + 3
        )));
    }
    let distances = pairwise_squared_distances(cohort)?;
    let neighbours = n - f - 2;
    Ok(krum_scores_from_distances(&distances, neighbours))
}

/// Krum scores for an arbitrary distance matrix and neighbour count.
///
/// Split out so that Bulyan's inner selection loop -- which shrinks the
/// candidate set and therefore needs a clamped neighbour count -- can reuse
/// the scoring without re-deriving distances.
fn krum_scores_from_distances(distances: &[Vec<f64>], neighbours: usize) -> Vec<f64> {
    let n = distances.len();
    let mut scores = Vec::with_capacity(n);
    for (i, distance_row) in distances.iter().enumerate() {
        let mut row: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .filter_map(|j| distance_row.get(j).copied())
            .collect();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let take = neighbours.min(row.len());
        scores.push(row[..take].iter().sum());
    }
    scores
}

/// Rank cohort positions by ascending Krum score, breaking ties by position.
fn rank_by_score(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Krum: the single cohort position with the lowest Krum score.
pub fn krum_select<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    f: usize,
) -> Result<usize> {
    let scores = krum_scores(cohort, f)?;
    rank_by_score(&scores)
        .first()
        .copied()
        .ok_or_else(|| OptimError::InvalidConfig("empty cohort".to_string()))
}

/// Multi-Krum: the `m` cohort positions with the lowest Krum scores.
pub fn multi_krum_indices<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    f: usize,
    m: usize,
) -> Result<Vec<usize>> {
    let n = cohort.len();
    if m == 0 {
        return Err(OptimError::InvalidConfig(
            "Multi-Krum must select at least one client (m > 0)".to_string(),
        ));
    }
    if m > n {
        return Err(OptimError::InvalidConfig(format!(
            "Multi-Krum cannot select m = {m} clients from a cohort of {n}"
        )));
    }
    let scores = krum_scores(cohort, f)?;
    let mut selected = rank_by_score(&scores);
    selected.truncate(m);
    selected.sort_unstable();
    Ok(selected)
}

/// Bulyan (El Mhamdi et al. 2018).
///
/// Stage 1 runs Krum `theta = n - 2f` times, removing the winner from the
/// candidate set each round. Stage 2 takes, per coordinate, the
/// `beta = theta - 2f` selected values closest to their median and averages
/// them, which bounds the influence any single coordinate of a Byzantine
/// update can have.
///
/// Requires `n >= 4f + 3`, the bound stated in the paper.
///
/// As the stage-1 candidate set shrinks, the ideal neighbour count
/// `|S| - f - 2` can drop to zero; like the reference implementations, this
/// one clamps it to at least one neighbour instead of aborting mid-selection.
pub fn bulyan<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    f: usize,
) -> Result<Array1<T>> {
    let n = cohort.len();
    if n < 4 * f + 3 {
        return Err(OptimError::InvalidConfig(format!(
            "Bulyan with f = {f} Byzantine clients needs at least {} participants, got {n}",
            4 * f + 3
        )));
    }
    let dim = cohort_dim(cohort)?;
    let theta = n - 2 * f;
    let beta = theta.saturating_sub(2 * f);
    if beta == 0 {
        return Err(OptimError::InvalidConfig(format!(
            "Bulyan's second stage would keep zero values per coordinate (theta = {theta}, \
             f = {f})"
        )));
    }

    let distances = pairwise_squared_distances(cohort)?;
    let mut candidates: Vec<usize> = (0..n).collect();
    let mut selected: Vec<usize> = Vec::with_capacity(theta);
    while selected.len() < theta {
        let live = candidates.len();
        if live == 0 {
            return Err(OptimError::InvalidState(
                "Bulyan exhausted its candidate set before selecting theta clients".to_string(),
            ));
        }
        let sub: Vec<Vec<f64>> = candidates
            .iter()
            .map(|&i| candidates.iter().map(|&j| distances[i][j]).collect())
            .collect();
        let neighbours = live.saturating_sub(f + 2).max(1);
        let scores = krum_scores_from_distances(&sub, neighbours);
        let best_local = rank_by_score(&scores).first().copied().ok_or_else(|| {
            OptimError::InvalidState("Bulyan scored an empty candidate set".to_string())
        })?;
        selected.push(candidates.remove(best_local));
    }

    let two = from_f64::<T>(2.0)?;
    let denominator = from_f64::<T>(beta as f64)?;
    let mut result = Array1::zeros(dim);
    let mut column: Vec<T> = Vec::with_capacity(theta);
    for coordinate in 0..dim {
        column.clear();
        column.extend(selected.iter().map(|&i| cohort[i].1[coordinate]));
        let mut sorted = column.clone();
        sorted.sort_by(cmp_finite);
        let middle = sorted.len() / 2;
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) / two
        } else {
            sorted[middle]
        };
        // Keep the beta values closest to the median; ties resolved by value
        // order so the result is deterministic.
        let mut by_distance: Vec<(T, T)> = column
            .iter()
            .map(|&value| ((value - median).abs(), value))
            .collect();
        by_distance.sort_by(|a, b| cmp_finite(&a.0, &b.0).then(cmp_finite(&a.1, &b.1)));
        let mut sum = T::zero();
        for (_, value) in by_distance[..beta].iter() {
            sum = sum + *value;
        }
        result[coordinate] = sum / denominator;
    }
    Ok(result)
}

/// Centered clipping (Karimireddy et al. 2021).
///
/// Starting from the coordinate-wise median -- a robust initial centre -- the
/// fixed-point iteration
///
/// ```text
/// v <- v + (1/n) * sum_i (x_i - v) * min(1, tau / ||x_i - v||)
/// ```
///
/// is applied `iterations` times. Each client can move the centre by at most
/// `tau / n`, which caps the damage a Byzantine update can do regardless of
/// its magnitude.
pub fn centered_clipping<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    tau: f64,
    iterations: usize,
) -> Result<Array1<T>> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(OptimError::InvalidConfig(format!(
            "centered clipping radius tau must be positive and finite, got {tau}"
        )));
    }
    if iterations == 0 {
        return Err(OptimError::InvalidConfig(
            "centered clipping needs at least one iteration".to_string(),
        ));
    }
    let dim = cohort_dim(cohort)?;
    let n = cohort.len();
    let inverse_n = from_f64::<T>(1.0 / n as f64)?;
    let tau_t = from_f64::<T>(tau)?;

    let mut centre = coordinate_wise_median(cohort)?;
    for _ in 0..iterations {
        let mut correction: Array1<T> = Array1::zeros(dim);
        for (_, update) in cohort.iter() {
            let difference = *update - &centre;
            let norm = difference
                .iter()
                .fold(T::zero(), |acc, &value| acc + value * value)
                .sqrt();
            if norm <= T::zero() {
                continue;
            }
            let scale = if norm > tau_t { tau_t / norm } else { T::one() };
            correction = correction + difference.mapv(|value| value * scale);
        }
        centre = centre + correction.mapv(|value| value * inverse_n);
    }
    Ok(centre)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cohort_from(pairs: &[(&str, Array1<f64>)]) -> HashMap<String, Array1<f64>> {
        pairs
            .iter()
            .map(|(id, update)| ((*id).to_string(), update.clone()))
            .collect()
    }

    fn borrow<'a>(map: &'a HashMap<String, Array1<f64>>) -> Vec<CohortMember<'a, f64>> {
        ordered_cohort(map).expect("well-formed cohort")
    }

    #[test]
    fn ordered_cohort_is_sorted_by_client_id() {
        let map = cohort_from(&[
            ("zulu", Array1::from(vec![1.0])),
            ("alpha", Array1::from(vec![2.0])),
            ("mike", Array1::from(vec![3.0])),
        ]);
        let cohort = borrow(&map);
        assert_eq!(
            cohort.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec!["alpha", "mike", "zulu"]
        );
    }

    #[test]
    fn ordered_cohort_rejects_ragged_and_non_finite_updates() {
        let ragged = cohort_from(&[
            ("a", Array1::from(vec![1.0, 2.0])),
            ("b", Array1::from(vec![1.0])),
        ]);
        assert!(ordered_cohort(&ragged).is_err());

        let nan = cohort_from(&[
            ("a", Array1::from(vec![1.0, 2.0])),
            ("b", Array1::from(vec![f64::NAN, 2.0])),
        ]);
        let err = ordered_cohort(&nan).expect_err("NaN must be rejected");
        assert!(format!("{err}").contains("non-finite"));

        let empty: HashMap<String, Array1<f64>> = HashMap::new();
        assert!(ordered_cohort(&empty).is_err());
    }

    #[test]
    fn median_matches_hand_computation() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![1.0, 4.0, 7.0])),
            ("b", Array1::from(vec![2.0, 5.0, 8.0])),
            ("c", Array1::from(vec![3.0, 6.0, 9.0])),
        ]);
        let median = coordinate_wise_median(&borrow(&map)).expect("median");
        assert_eq!(median.to_vec(), vec![2.0, 5.0, 8.0]);

        let even = cohort_from(&[
            ("a", Array1::from(vec![1.0])),
            ("b", Array1::from(vec![2.0])),
            ("c", Array1::from(vec![3.0])),
            ("d", Array1::from(vec![10.0])),
        ]);
        let median = coordinate_wise_median(&borrow(&even)).expect("median");
        assert_eq!(median.to_vec(), vec![2.5]);
    }

    #[test]
    fn trim_count_follows_the_configured_ratio() {
        // 20 clients, 40% trimmed in total => 4 from each tail.
        assert_eq!(trim_count_for_ratio(20, 0.4).expect("ratio"), 4);
        // 10 clients, 10% => 0.5 per tail, floored to 0.
        assert_eq!(trim_count_for_ratio(10, 0.1).expect("ratio"), 0);
        // 4 clients, 50% => 1 per tail.
        assert_eq!(trim_count_for_ratio(4, 0.5).expect("ratio"), 1);
        // Capped so at least one value survives.
        assert_eq!(trim_count_for_ratio(3, 0.99).expect("ratio"), 1);
        assert_eq!(trim_count_for_ratio(1, 0.9).expect("ratio"), 0);
        assert!(trim_count_for_ratio(10, 1.0).is_err());
        assert!(trim_count_for_ratio(10, -0.1).is_err());
    }

    #[test]
    fn trimmed_mean_discards_both_tails() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![-100.0])),
            ("b", Array1::from(vec![1.0])),
            ("c", Array1::from(vec![2.0])),
            ("d", Array1::from(vec![3.0])),
            ("e", Array1::from(vec![900.0])),
        ]);
        let cohort = borrow(&map);
        // Trim one from each tail: mean of {1, 2, 3} = 2.
        let trimmed = coordinate_wise_trimmed_mean(&cohort, 1).expect("trimmed mean");
        assert_eq!(trimmed.to_vec(), vec![2.0]);
        // Trim zero: the plain mean, which the outliers dominate.
        let untrimmed = coordinate_wise_trimmed_mean(&cohort, 0).expect("mean");
        assert!((untrimmed[0] - 161.2).abs() < 1e-9);
        // Over-trimming errors rather than returning zeros.
        assert!(coordinate_wise_trimmed_mean(&cohort, 3).is_err());
    }

    #[test]
    fn pairwise_distances_are_symmetric_with_zero_diagonal() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![0.0, 0.0])),
            ("b", Array1::from(vec![3.0, 4.0])),
        ]);
        let distances = pairwise_squared_distances(&borrow(&map)).expect("distances");
        assert_eq!(distances[0][0], 0.0);
        assert_eq!(distances[1][1], 0.0);
        assert_eq!(distances[0][1], 25.0);
        assert_eq!(distances[1][0], 25.0);
    }

    #[test]
    fn krum_scores_sum_the_closest_neighbours() {
        // Four one-dimensional clients at 0, 1, 2, 100 with f = 1 =>
        // neighbours = n - f - 2 = 1, so each score is the squared distance to
        // its single nearest peer.
        let map = cohort_from(&[
            ("a", Array1::from(vec![0.0])),
            ("b", Array1::from(vec![1.0])),
            ("c", Array1::from(vec![2.0])),
            ("d", Array1::from(vec![100.0])),
        ]);
        let scores = krum_scores(&borrow(&map), 1).expect("scores");
        assert_eq!(scores, vec![1.0, 1.0, 1.0, 9604.0]);
    }

    #[test]
    fn krum_selects_the_most_central_client_and_ignores_the_outlier() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![0.0, 0.0])),
            ("b", Array1::from(vec![0.1, 0.1])),
            ("c", Array1::from(vec![0.2, 0.2])),
            ("evil", Array1::from(vec![500.0, -500.0])),
        ]);
        let cohort = borrow(&map);
        let winner = krum_select(&cohort, 1).expect("krum");
        assert_ne!(cohort[winner].0, "evil");

        // Multi-Krum with m = 3 keeps the three honest clients.
        let selected = multi_krum_indices(&cohort, 1, 3).expect("multi-krum");
        let ids: Vec<&str> = selected.iter().map(|&i| cohort[i].0).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn krum_requires_enough_participants() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![0.0])),
            ("b", Array1::from(vec![1.0])),
            ("c", Array1::from(vec![2.0])),
        ]);
        let cohort = borrow(&map);
        // n = 3, f = 1 => needs 4.
        let err = krum_scores(&cohort, 1).expect_err("insufficient cohort");
        assert!(format!("{err}").contains("needs at least 4"));
        // f = 0 => needs 3, which is satisfied.
        assert!(krum_scores(&cohort, 0).is_ok());
    }

    #[test]
    fn bulyan_resists_a_coordinate_wise_attack() {
        // Seven honest clients near 1.0 and no attacker: f = 1 needs n >= 7.
        let mut pairs: Vec<(String, Array1<f64>)> = Vec::new();
        for i in 0..6 {
            pairs.push((
                format!("honest{i}"),
                Array1::from(vec![1.0 + i as f64 * 0.01, 2.0]),
            ));
        }
        pairs.push(("attacker".to_string(), Array1::from(vec![1.0e6, -1.0e6])));
        let map: HashMap<String, Array1<f64>> = pairs.into_iter().collect();
        let cohort = ordered_cohort(&map).expect("cohort");
        let result = bulyan(&cohort, 1).expect("bulyan");
        assert!(
            result[0] > 0.9 && result[0] < 1.2,
            "bulyan must stay near the honest cluster, got {}",
            result[0]
        );
        assert!((result[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn bulyan_requires_the_paper_bound() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![0.0])),
            ("b", Array1::from(vec![1.0])),
            ("c", Array1::from(vec![2.0])),
            ("d", Array1::from(vec![3.0])),
        ]);
        let err = bulyan(&borrow(&map), 1).expect_err("n = 4 < 4f + 3 = 7");
        assert!(format!("{err}").contains("needs at least 7"));
    }

    #[test]
    fn centered_clipping_bounds_a_single_clients_influence() {
        // Nine clients at exactly 0 and one at 1000. With tau = 1 the
        // attacker can move the centre by at most tau / n = 0.1 per
        // iteration, so after 3 iterations the centre is at most 0.3.
        let mut pairs: Vec<(String, Array1<f64>)> = (0..9)
            .map(|i| (format!("honest{i}"), Array1::from(vec![0.0])))
            .collect();
        pairs.push(("attacker".to_string(), Array1::from(vec![1000.0])));
        let map: HashMap<String, Array1<f64>> = pairs.into_iter().collect();
        let cohort = ordered_cohort(&map).expect("cohort");
        let result =
            centered_clipping(&cohort, 1.0, CENTERED_CLIPPING_ITERATIONS).expect("clipping");
        assert!(
            result[0] > 0.0 && result[0] <= 0.3 + 1e-9,
            "centre moved to {} which exceeds 3 * tau / n",
            result[0]
        );

        // The plain mean, by contrast, is dragged to 100.
        let plain = mean(&cohort).expect("mean");
        assert!((plain[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn centered_clipping_reproduces_the_centre_of_a_clean_cohort() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![1.0, 1.0])),
            ("b", Array1::from(vec![1.0, 1.0])),
            ("c", Array1::from(vec![1.0, 1.0])),
        ]);
        let result = centered_clipping(&borrow(&map), 10.0, 3).expect("clipping");
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn centered_clipping_validates_tau_and_iterations() {
        let map = cohort_from(&[("a", Array1::from(vec![1.0]))]);
        let cohort = borrow(&map);
        assert!(centered_clipping(&cohort, 0.0, 3).is_err());
        assert!(centered_clipping(&cohort, f64::NAN, 3).is_err());
        assert!(centered_clipping(&cohort, 1.0, 0).is_err());
    }

    #[test]
    fn weighted_mean_normalises_and_validates_weights() {
        let map = cohort_from(&[
            ("a", Array1::from(vec![0.0])),
            ("b", Array1::from(vec![10.0])),
        ]);
        let cohort = borrow(&map);
        let result = weighted_mean(&cohort, &[3.0, 1.0]).expect("weighted mean");
        assert!((result[0] - 2.5).abs() < 1e-12);

        assert!(weighted_mean(&cohort, &[1.0]).is_err());
        assert!(weighted_mean(&cohort, &[0.0, 0.0]).is_err());
        assert!(weighted_mean(&cohort, &[-1.0, 2.0]).is_err());
    }
}
