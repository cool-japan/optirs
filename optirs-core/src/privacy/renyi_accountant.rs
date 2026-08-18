// Renyi Differential Privacy (RDP) Accountant
//
// This module implements a Renyi Differential Privacy accountant for tight
// composition of Gaussian and subsampled-Gaussian mechanisms. The RDP
// formulation, introduced by Mironov (2017), composes linearly over iterations
// and converts to (epsilon, delta)-DP via a tight closed-form mapping.
//
// The implementation follows the bounds from:
//   * Mironov, "Renyi Differential Privacy", CSF 2017.
//   * Wang, Balle, Kasiviswanathan, "Subsampled Renyi Differential Privacy and
//     Analytical Moments Accountant", AISTATS 2019.
//   * Mironov, Talwar, Zhang, "Renyi Differential Privacy of the Sampled
//     Gaussian Mechanism", arXiv:1908.10530 (2019).
//
// The subsampled Gaussian bound used here is the standard tight bound for
// integer orders, computed in log space using the log-sum-exp trick for
// numerical stability. It is evaluated exactly for every sampling probability
// `q > 0` -- there is deliberately no "small q" analytical shortcut, because
// such shortcuts under-report epsilon (the one direction that silently voids a
// DP guarantee).
//
// Non-integer orders are handled by evaluating the kernel at `ceil(alpha)`.
// The Renyi divergence `D_alpha` is non-decreasing in `alpha`, so this is a
// valid *upper* bound on the RDP at the requested order: conservative, never
// optimistic. Interpolating (or extrapolating) between integer anchors, as
// earlier revisions did, can fall below the true value and is not used.
//
// This accountant is the reference privacy accountant of the crate;
// `moment_accountant::MomentsAccountant` composes the very same per-step kernel
// through a heterogeneous-composition ledger.

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};

/// Default Renyi orders tracked by the accountant.
///
/// These orders are the standard set used by reference implementations
/// (Opacus, TensorFlow Privacy). The range from 1.25 to 64.0 covers the
/// typical regime of practical DP-SGD configurations.
pub const DEFAULT_ALPHAS: &[f64] = &[
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    20.0, 24.0, 28.0, 32.0, 48.0, 64.0,
];

/// Threshold on the accumulated per-order RDP above which the accountant
/// reports *no* privacy at all.
///
/// The accountant never silently clamps a privacy loss: once a contribution
/// is not finite (or the accumulated spend crosses this threshold), the
/// accountant latches a saturation flag and every subsequent conversion
/// returns `epsilon = +infinity`. That is the fail-closed direction --
/// callers comparing against a budget will see the budget as exhausted
/// rather than believing a fabricated finite number.
const RDP_SATURATION_THRESHOLD: f64 = 1.0e12;

/// Snapshot of the current per-order RDP spend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdpSpend {
    /// Renyi orders (alpha values) tracked by the accountant.
    pub orders: Vec<f64>,

    /// Accumulated RDP epsilon at each corresponding order.
    pub epsilons: Vec<f64>,
}

/// Result of converting accumulated RDP into (epsilon, delta)-DP.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DpConversion {
    /// The minimum epsilon found across all tracked orders.
    pub epsilon: f64,

    /// The target delta used for the conversion.
    pub delta: f64,

    /// The Renyi order alpha that achieved the minimum epsilon.
    pub best_order: f64,
}

/// Renyi Differential Privacy accountant.
///
/// Tracks accumulated RDP epsilons across a set of Renyi orders. Each call
/// to [`add_gaussian`](Self::add_gaussian) or
/// [`add_subsampled_gaussian`](Self::add_subsampled_gaussian) composes the
/// new mechanism into the running budget. Conversion to standard
/// (epsilon, delta)-DP is performed lazily via
/// [`to_epsilon_delta`](Self::to_epsilon_delta).
#[derive(Debug, Clone)]
pub struct RenyiAccountant {
    /// Sorted ascending list of Renyi orders.
    orders: Vec<f64>,

    /// Accumulated RDP epsilon for each order in `orders`.
    rdp_epsilons: Vec<f64>,

    /// Total number of mechanism applications composed so far.
    total_steps: usize,

    /// Latched once any composed contribution overflowed the representable
    /// range (or crossed [`RDP_SATURATION_THRESHOLD`]). While set, the
    /// accountant reports an infinite epsilon.
    saturated: bool,
}

impl RenyiAccountant {
    /// Create a new accountant with a user-supplied set of Renyi orders.
    ///
    /// The orders are sorted ascending. All orders must be strictly greater
    /// than 1.0 (RDP is only defined for alpha > 1). The list must not be
    /// empty.
    pub fn new(orders: Vec<f64>) -> Result<Self> {
        if orders.is_empty() {
            return Err(OptimError::InvalidParameter(
                "RenyiAccountant requires at least one Renyi order".to_string(),
            ));
        }

        for &alpha in &orders {
            if !alpha.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "Renyi order must be finite, got {alpha}"
                )));
            }
            if alpha <= 1.0 {
                return Err(OptimError::InvalidParameter(format!(
                    "Renyi order must be strictly greater than 1.0, got {alpha}"
                )));
            }
        }

        let mut sorted = orders;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        Ok(Self {
            orders: sorted,
            rdp_epsilons: vec![0.0; len],
            total_steps: 0,
            saturated: false,
        })
    }

    /// Create an accountant using the canonical [`DEFAULT_ALPHAS`] list.
    pub fn with_default_orders() -> Self {
        // Safe to construct: DEFAULT_ALPHAS is a verified non-empty,
        // strictly-greater-than-1, sorted-ascending list.
        let orders = DEFAULT_ALPHAS.to_vec();
        let len = orders.len();
        Self {
            orders,
            rdp_epsilons: vec![0.0; len],
            total_steps: 0,
            saturated: false,
        }
    }

    /// Return the canonical default order list.
    pub fn default_orders() -> Vec<f64> {
        DEFAULT_ALPHAS.to_vec()
    }

    /// Compose a subsampled Gaussian mechanism into the running budget.
    ///
    /// Each step samples each record independently with probability
    /// `sampling_prob`, then adds Gaussian noise with standard deviation
    /// `noise_multiplier` to the sum of clipped per-example gradients.
    /// `steps` such applications are composed.
    ///
    /// The bound used is the tight Mironov/Wang/Balle bound for the sampled
    /// Gaussian mechanism (see module-level reference list).
    pub fn add_subsampled_gaussian(
        &mut self,
        noise_multiplier: f64,
        sampling_prob: f64,
        steps: usize,
    ) -> Result<()> {
        if !noise_multiplier.is_finite() || noise_multiplier <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "noise_multiplier must be a positive finite number, got {noise_multiplier}"
            )));
        }
        if !sampling_prob.is_finite() || !(0.0..=1.0).contains(&sampling_prob) {
            return Err(OptimError::InvalidParameter(format!(
                "sampling_prob must be in [0, 1], got {sampling_prob}"
            )));
        }

        if steps == 0 || sampling_prob == 0.0 {
            // No mechanism application contributes any privacy loss.
            self.total_steps = self.total_steps.saturating_add(steps);
            return Ok(());
        }

        let steps_f = steps as f64;
        for (i, &alpha) in self.orders.iter().enumerate() {
            let per_step = rdp_subsampled_gaussian_step(alpha, noise_multiplier, sampling_prob)?;
            let contribution = per_step * steps_f;
            let accumulated = self.rdp_epsilons[i] + contribution;
            if !accumulated.is_finite() || accumulated > RDP_SATURATION_THRESHOLD {
                self.saturated = true;
            }
            self.rdp_epsilons[i] = accumulated;
        }

        self.total_steps = self.total_steps.saturating_add(steps);
        Ok(())
    }

    /// Compose a pure Gaussian mechanism (no subsampling) into the budget.
    ///
    /// The RDP of a Gaussian mechanism with noise multiplier `sigma` at
    /// order alpha is the well-known closed form `alpha / (2 * sigma^2)`,
    /// for all alpha > 1 (Mironov 2017, Proposition 7).
    pub fn add_gaussian(&mut self, noise_multiplier: f64, steps: usize) -> Result<()> {
        if !noise_multiplier.is_finite() || noise_multiplier <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "noise_multiplier must be a positive finite number, got {noise_multiplier}"
            )));
        }

        if steps == 0 {
            return Ok(());
        }

        let steps_f = steps as f64;
        let variance = noise_multiplier * noise_multiplier;

        for (i, &alpha) in self.orders.iter().enumerate() {
            let per_step = alpha / (2.0 * variance);
            let accumulated = self.rdp_epsilons[i] + per_step * steps_f;
            if !accumulated.is_finite() || accumulated > RDP_SATURATION_THRESHOLD {
                self.saturated = true;
            }
            self.rdp_epsilons[i] = accumulated;
        }

        self.total_steps = self.total_steps.saturating_add(steps);
        Ok(())
    }

    /// Return a snapshot of the current per-order RDP spend.
    pub fn current_spend(&self) -> RdpSpend {
        RdpSpend {
            orders: self.orders.clone(),
            epsilons: self.rdp_epsilons.clone(),
        }
    }

    /// Convert the accumulated RDP into a tight (epsilon, delta)-DP bound.
    ///
    /// For each tracked order alpha, computes the improved RDP-to-DP
    /// conversion of Canonne, Kamath and Steinke (2020, Proposition 12), as
    /// used by Opacus:
    ///
    /// ```text
    /// eps(alpha) = rdp(alpha)
    ///            + ln((alpha - 1) / alpha)
    ///            - (ln(delta) + ln(alpha)) / (alpha - 1)
    /// ```
    ///
    /// This is uniformly tighter than the classic Mironov (2017,
    /// Proposition 3) conversion `rdp(alpha) + ln(1/delta) / (alpha - 1)`,
    /// because both correction terms `ln(1 - 1/alpha)` and
    /// `-ln(alpha)/(alpha - 1)` are negative. The minimum over the tracked
    /// orders is returned.
    ///
    /// If the accountant has saturated (see [`is_saturated`](Self::is_saturated)),
    /// `epsilon` is `+infinity`: the mechanism provides no usable guarantee
    /// and the caller must treat its budget as exhausted.
    pub fn to_epsilon_delta(&self, target_delta: f64) -> Result<DpConversion> {
        if !target_delta.is_finite() || target_delta <= 0.0 || target_delta > 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_delta must be in (0, 1], got {target_delta}"
            )));
        }

        if self.saturated {
            return Ok(DpConversion {
                epsilon: f64::INFINITY,
                delta: target_delta,
                best_order: self.orders[0],
            });
        }

        // Composing nothing costs nothing. Without this guard the conversion
        // slack (`ln(1/delta) / (alpha - 1)`) would report a positive epsilon
        // for an accountant that has never observed a mechanism.
        if self.rdp_epsilons.iter().all(|&e| e == 0.0) {
            return Ok(DpConversion {
                epsilon: 0.0,
                delta: target_delta,
                best_order: self.orders[self.orders.len() - 1],
            });
        }

        let log_delta = target_delta.ln();

        let mut best_epsilon = f64::INFINITY;
        let mut best_order = self.orders[0];

        for (i, &alpha) in self.orders.iter().enumerate() {
            let candidate = self.rdp_epsilons[i] + ((alpha - 1.0) / alpha).ln()
                - (log_delta + alpha.ln()) / (alpha - 1.0);
            if candidate.is_finite() && candidate < best_epsilon {
                best_epsilon = candidate;
                best_order = alpha;
            }
        }

        // Epsilon is non-negative by definition: clamp away tiny negatives
        // that could only arise from floating-point round-off.
        let epsilon = best_epsilon.max(0.0);

        Ok(DpConversion {
            epsilon,
            delta: target_delta,
            best_order,
        })
    }

    /// Reset the accumulated spend back to zero.
    pub fn reset(&mut self) {
        for value in self.rdp_epsilons.iter_mut() {
            *value = 0.0;
        }
        self.total_steps = 0;
        self.saturated = false;
    }

    /// Whether the accumulated privacy loss overflowed the representable
    /// range. Once true, [`to_epsilon_delta`](Self::to_epsilon_delta)
    /// reports an infinite epsilon until [`reset`](Self::reset) is called.
    pub fn is_saturated(&self) -> bool {
        self.saturated
    }

    /// Return the total number of composed mechanism applications.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Return the Renyi orders tracked by this accountant.
    pub fn orders(&self) -> &[f64] {
        &self.orders
    }
}

/// Compute the RDP of one application of the subsampled Gaussian mechanism
/// at a given Renyi order.
///
/// For integer orders `alpha >= 2` the exact binomial expansion of the
/// sampled-Gaussian moment generating function is evaluated in log space.
/// The expansion is used for **every** `q > 0`: there is no small-`q`
/// analytical shortcut, because such shortcuts under-report the privacy
/// loss, and `ln(q)` is perfectly well behaved down to the smallest
/// normal `f64`.
///
/// Non-integer orders are bounded by the value at `ceil(alpha)`. The Renyi
/// divergence is non-decreasing in its order, so `rdp(alpha) <=
/// rdp(ceil(alpha))`; the returned value is therefore a valid, conservative
/// bound. Orders in `(1, 2)` are bounded by the value at `alpha = 2` for the
/// same reason.
///
/// The result may be `+infinity` for pathologically small noise multipliers
/// (the exponent `k(k-1)/(2 sigma^2)` overflows). That is reported faithfully
/// rather than clamped: an infinite RDP means "no privacy".
pub(crate) fn rdp_subsampled_gaussian_step(
    alpha: f64,
    noise_multiplier: f64,
    q: f64,
) -> Result<f64> {
    if !alpha.is_finite() || alpha <= 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "Renyi order alpha must satisfy alpha > 1, got {alpha}"
        )));
    }
    if !noise_multiplier.is_finite() || noise_multiplier <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "noise_multiplier must be a positive finite number, got {noise_multiplier}"
        )));
    }
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(OptimError::InvalidParameter(format!(
            "sampling probability must be in [0, 1], got {q}"
        )));
    }

    if q == 0.0 {
        return Ok(0.0);
    }

    if q == 1.0 {
        // Sampling everything is equivalent to the pure Gaussian mechanism.
        let variance = noise_multiplier * noise_multiplier;
        return Ok(alpha / (2.0 * variance));
    }

    // Evaluate at an integer order that upper-bounds the requested one.
    let alpha_int = if (alpha - alpha.round()).abs() < 1.0e-12 {
        alpha.round() as usize
    } else {
        alpha.ceil() as usize
    };
    let alpha_int = alpha_int.max(2);

    Ok(rdp_subsampled_gaussian_step_integer(
        alpha_int,
        noise_multiplier,
        q,
    ))
}

/// Compute the RDP per step at an integer order alpha >= 2 using the
/// binomial expansion of the sampled-Gaussian moment generating function.
///
/// The bound is:
/// ```text
/// rdp(alpha) = (1 / (alpha - 1)) * ln( sum_{k=0..=alpha} C(alpha, k)
///                                       * (1 - q)^(alpha - k) * q^k
///                                       * exp( k * (k - 1) / (2 sigma^2) ) )
/// ```
/// Implemented in log space with the log-sum-exp trick.
pub(crate) fn rdp_subsampled_gaussian_step_integer(alpha: usize, sigma: f64, q: f64) -> f64 {
    if alpha < 2 {
        // Shouldn't happen given our call sites, but be defensive.
        return 0.0;
    }

    let alpha_f = alpha as f64;
    let variance = sigma * sigma;
    let log_q = q.ln();
    let log_one_minus_q = (1.0 - q).ln();

    let mut log_terms: Vec<f64> = Vec::with_capacity(alpha + 1);
    for k in 0..=alpha {
        let k_f = k as f64;
        let log_binom = log_binom_coefficient(alpha_f, k);
        let term = log_binom
            + (alpha_f - k_f) * log_one_minus_q
            + k_f * log_q
            + (k_f * (k_f - 1.0)) / (2.0 * variance);
        log_terms.push(term);
    }

    let log_sum = log_sum_exp(&log_terms);
    let rdp = log_sum / (alpha_f - 1.0);

    if rdp.is_nan() {
        // Pathological inputs: fail closed with "no privacy" rather than
        // propagating a NaN that compares false against every budget check.
        f64::INFINITY
    } else if rdp < 0.0 {
        // RDP is non-negative by definition; a small negative value can only
        // come from floating-point round-off in the log-sum-exp.
        0.0
    } else {
        // May legitimately be +infinity for a vanishing noise multiplier.
        rdp
    }
}

/// Numerically stable log of the sum of exponentials of the input slice.
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v > max {
            max = v;
        }
    }

    if !max.is_finite() {
        return max;
    }

    let mut sum = 0.0;
    for &v in values {
        sum += (v - max).exp();
    }

    max + sum.ln()
}

/// Log of the binomial coefficient C(n, k) for real-valued n and
/// non-negative integer k. Uses the recursion
/// `log_binom(n, k) = sum_{i=1..=k} (log(n - i + 1) - log(i))`.
///
/// This avoids dependency on `lgamma`/`tgamma` and is numerically robust
/// for the small values of k (up to ~64) used by this accountant.
fn log_binom_coefficient(n: f64, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }

    let mut accumulator = 0.0;
    for i in 1..=k {
        let i_f = i as f64;
        let numerator = n - i_f + 1.0;
        if numerator <= 0.0 {
            // C(n, k) = 0 in this case; return a large negative log.
            return f64::NEG_INFINITY;
        }
        accumulator += numerator.ln() - i_f.ln();
    }
    accumulator
}

#[cfg(test)]
mod tests {
    use super::*;

    const APPROX_TOL: f64 = 1.0e-9;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_default_orders_includes_typical_values() {
        let orders = RenyiAccountant::default_orders();
        for needle in [1.25_f64, 2.0, 4.0, 16.0, 64.0] {
            assert!(
                orders.iter().any(|o| (o - needle).abs() < 1.0e-12),
                "default orders must contain {needle}"
            );
        }
    }

    #[test]
    fn test_new_validates_orders_above_one() {
        let result = RenyiAccountant::new(vec![0.5_f64, 2.0]);
        match result {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_new_sorts_unsorted_input() {
        let accountant = RenyiAccountant::new(vec![4.0_f64, 2.0]).expect("should accept orders");
        let orders = accountant.orders();
        assert_eq!(orders.len(), 2);
        assert!(orders[0] < orders[1]);
        assert!(approx_eq(orders[0], 2.0, APPROX_TOL));
        assert!(approx_eq(orders[1], 4.0, APPROX_TOL));
    }

    #[test]
    fn test_zero_steps_zero_spend() {
        let accountant = RenyiAccountant::with_default_orders();
        let spend = accountant.current_spend();
        assert_eq!(spend.orders.len(), spend.epsilons.len());
        for eps in spend.epsilons {
            assert_eq!(eps, 0.0);
        }
        assert_eq!(accountant.total_steps(), 0);
    }

    #[test]
    fn test_spend_grows_monotonically_with_steps() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.0, 0.01, 100)
            .expect("first composition should succeed");
        let first = accountant.current_spend();
        for &eps in &first.epsilons {
            assert!(eps >= 0.0, "RDP must be non-negative, got {eps}");
        }

        accountant
            .add_subsampled_gaussian(1.0, 0.01, 100)
            .expect("second composition should succeed");
        let second = accountant.current_spend();

        for (a, b) in first.epsilons.iter().zip(second.epsilons.iter()) {
            assert!(b >= a, "RDP must grow monotonically, got {a} -> {b}");
            if *a > 0.0 {
                assert!(b > a, "RDP should strictly grow with more steps");
            }
        }

        assert_eq!(accountant.total_steps(), 200);
    }

    #[test]
    fn test_higher_noise_smaller_spend() {
        let mut low_noise = RenyiAccountant::with_default_orders();
        low_noise
            .add_subsampled_gaussian(1.0, 0.01, 500)
            .expect("low noise composition");
        let mut high_noise = RenyiAccountant::with_default_orders();
        high_noise
            .add_subsampled_gaussian(2.0, 0.01, 500)
            .expect("high noise composition");

        let low = low_noise.current_spend();
        let high = high_noise.current_spend();

        for (a, b) in low.epsilons.iter().zip(high.epsilons.iter()) {
            assert!(
                *b <= *a + APPROX_TOL,
                "higher noise should yield smaller RDP: low={a}, high={b}"
            );
        }
    }

    #[test]
    fn test_smaller_sampling_smaller_spend() {
        let mut sparse = RenyiAccountant::with_default_orders();
        sparse
            .add_subsampled_gaussian(1.0, 0.001, 500)
            .expect("sparse sampling composition");
        let mut dense = RenyiAccountant::with_default_orders();
        dense
            .add_subsampled_gaussian(1.0, 0.01, 500)
            .expect("dense sampling composition");

        let sparse_spend = sparse.current_spend();
        let dense_spend = dense.current_spend();

        for (s, d) in sparse_spend
            .epsilons
            .iter()
            .zip(dense_spend.epsilons.iter())
        {
            assert!(
                *s <= *d + APPROX_TOL,
                "smaller sampling probability should yield smaller RDP: sparse={s}, dense={d}"
            );
        }
    }

    #[test]
    fn test_pure_gaussian_matches_analytical_formula() {
        // For pure Gaussian sigma=1, the RDP at alpha=2 should be exactly
        // alpha / (2 sigma^2) = 1.0.
        let mut accountant = RenyiAccountant::new(vec![2.0_f64]).expect("alpha=2 is valid");
        accountant.add_gaussian(1.0, 1).expect("gaussian step");

        let spend = accountant.current_spend();
        assert_eq!(spend.orders.len(), 1);
        assert!(
            approx_eq(spend.epsilons[0], 1.0, 1.0e-12),
            "expected exactly 1.0, got {}",
            spend.epsilons[0]
        );

        // After 5 steps, RDP at alpha=2 should be 5.0.
        accountant
            .add_gaussian(1.0, 4)
            .expect("more gaussian steps");
        let spend = accountant.current_spend();
        assert!(
            approx_eq(spend.epsilons[0], 5.0, 1.0e-12),
            "expected 5.0, got {}",
            spend.epsilons[0]
        );
    }

    #[test]
    fn test_to_epsilon_delta_returns_finite_when_spend_nonzero() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.0, 0.01, 1000)
            .expect("composition");

        let result = accountant.to_epsilon_delta(1.0e-5).expect("conversion");
        assert!(result.epsilon.is_finite());
        assert!(result.epsilon > 0.0);
        assert!(approx_eq(result.delta, 1.0e-5, 1.0e-18));
        let orders = accountant.orders();
        assert!(orders
            .iter()
            .any(|o| approx_eq(*o, result.best_order, APPROX_TOL)));
    }

    #[test]
    fn test_to_epsilon_delta_invalid_target_delta_errors() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant.add_gaussian(1.0, 1).expect("step");

        match accountant.to_epsilon_delta(0.0) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for delta=0, got {other:?}"),
        }

        match accountant.to_epsilon_delta(2.0) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for delta>1, got {other:?}"),
        }

        match accountant.to_epsilon_delta(-0.1) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for negative delta, got {other:?}"),
        }
    }

    #[test]
    fn test_to_epsilon_delta_chooses_optimal_order() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.1, 0.005, 500)
            .expect("composition");
        let result = accountant.to_epsilon_delta(1.0e-5).expect("conversion");

        let orders = accountant.orders();
        assert!(
            orders
                .iter()
                .any(|o| approx_eq(*o, result.best_order, APPROX_TOL)),
            "best_order {} must come from configured order list",
            result.best_order
        );
    }

    #[test]
    fn test_composition_linear_in_steps() {
        // Doing 1000 steps in one call should produce the same per-order RDP
        // as ten calls with 100 steps each, up to floating point noise.
        let mut single = RenyiAccountant::with_default_orders();
        single
            .add_subsampled_gaussian(1.0, 0.01, 1000)
            .expect("single composition");

        let mut chunked = RenyiAccountant::with_default_orders();
        for _ in 0..10 {
            chunked
                .add_subsampled_gaussian(1.0, 0.01, 100)
                .expect("chunked composition");
        }

        let s = single.current_spend();
        let c = chunked.current_spend();
        assert_eq!(s.orders.len(), c.orders.len());
        for (a, b) in s.epsilons.iter().zip(c.epsilons.iter()) {
            assert!(
                approx_eq(*a, *b, 1.0e-9),
                "composition must be linear in steps: {a} vs {b}"
            );
        }

        assert_eq!(single.total_steps(), 1000);
        assert_eq!(chunked.total_steps(), 1000);
    }

    #[test]
    fn test_reset_clears_spend() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.0, 0.01, 500)
            .expect("composition");
        assert!(accountant.total_steps() > 0);

        accountant.reset();
        assert_eq!(accountant.total_steps(), 0);
        let spend = accountant.current_spend();
        for eps in spend.epsilons {
            assert_eq!(eps, 0.0);
        }
    }

    #[test]
    fn test_serde_roundtrip_rdpspend() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.2, 0.005, 200)
            .expect("composition");
        let spend = accountant.current_spend();

        let json = serde_json::to_string(&spend).expect("serialize");
        let parsed: RdpSpend = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.orders.len(), spend.orders.len());
        for (a, b) in parsed.orders.iter().zip(spend.orders.iter()) {
            assert!(approx_eq(*a, *b, APPROX_TOL));
        }
        for (a, b) in parsed.epsilons.iter().zip(spend.epsilons.iter()) {
            assert!(approx_eq(*a, *b, APPROX_TOL));
        }

        // DpConversion roundtrip as well, since it is also serde-derived.
        let conv = accountant.to_epsilon_delta(1.0e-5).expect("conversion");
        let conv_json = serde_json::to_string(&conv).expect("serialize conversion");
        let parsed_conv: DpConversion =
            serde_json::from_str(&conv_json).expect("deserialize conversion");
        assert!(approx_eq(parsed_conv.epsilon, conv.epsilon, APPROX_TOL));
        assert!(approx_eq(parsed_conv.delta, conv.delta, APPROX_TOL));
        assert!(approx_eq(
            parsed_conv.best_order,
            conv.best_order,
            APPROX_TOL
        ));
    }

    #[test]
    fn test_negative_noise_multiplier_errors() {
        let mut accountant = RenyiAccountant::with_default_orders();
        match accountant.add_subsampled_gaussian(-1.0, 0.01, 100) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for negative noise, got {other:?}"),
        }
        match accountant.add_gaussian(-1.0, 100) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for negative noise, got {other:?}"),
        }
        match accountant.add_subsampled_gaussian(0.0, 0.01, 100) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for zero noise, got {other:?}"),
        }
    }

    #[test]
    fn test_invalid_sampling_prob_errors() {
        let mut accountant = RenyiAccountant::with_default_orders();
        match accountant.add_subsampled_gaussian(1.0, -0.1, 100) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for negative q, got {other:?}"),
        }
        match accountant.add_subsampled_gaussian(1.0, 1.5, 100) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for q > 1, got {other:?}"),
        }
    }

    #[test]
    fn test_zero_sampling_prob_zero_spend() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.0, 0.0, 1000)
            .expect("zero-q composition should succeed");
        let spend = accountant.current_spend();
        for eps in spend.epsilons {
            assert_eq!(eps, 0.0, "zero sampling probability must yield zero RDP");
        }
        assert_eq!(accountant.total_steps(), 1000);
    }

    #[test]
    fn test_canonical_dp_sgd_setup() {
        // Standard DP-SGD config: sigma=1.0, q=0.01, 1000 steps, delta=1e-5.
        // The resulting epsilon should be a reasonable single-digit value.
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.0, 0.01, 1000)
            .expect("dp-sgd composition");

        let result = accountant.to_epsilon_delta(1.0e-5).expect("conversion");
        assert!(result.epsilon.is_finite());
        assert!(
            result.epsilon >= 0.5 && result.epsilon <= 10.0,
            "expected epsilon in [0.5, 10] for canonical setup, got {}",
            result.epsilon
        );
    }

    #[test]
    fn test_kernel_reproduces_the_published_tensorflow_privacy_reference() {
        // EXTERNAL validation of the sampled-Gaussian RDP kernel against a
        // number published by an independent implementation.
        //
        // The TensorFlow Privacy classification tutorial reports, for
        // `N = 60000, batch_size = 250, noise_multiplier = 1.3, epochs = 15`
        // (so `q = 250 / 60000` and `T = 15 * 60000 / 250 = 3600`) at
        // `delta = 1e-5`:
        //
        // > DP-SGD with sampling rate = 0.417% and noise_multiplier = 1.3
        // > iterated over 3600 steps satisfies differential privacy with
        // > eps = 1.18.
        //
        // TF Privacy's `get_privacy_spent` applies the *classic* Mironov
        // (2017) conversion `rdp + ln(1/delta) / (alpha - 1)`, so that
        // conversion is applied here explicitly instead of calling
        // `to_epsilon_delta` (which uses the strictly tighter Canonne-Kamath-
        // Steinke bound and would land at 0.9422, below the published value).
        //
        // Matching 1.18 to four significant figures is what makes every other
        // pinned constant in this module a real golden value rather than a
        // restatement of our own arithmetic.
        let orders: Vec<f64> = (2..=64).map(f64::from).collect();
        let mut accountant = RenyiAccountant::new(orders).expect("integer orders are valid");
        accountant
            .add_subsampled_gaussian(1.3, 250.0 / 60_000.0, 3600)
            .expect("composition");

        let spend = accountant.current_spend();
        let log_inv_delta = (1.0_f64 / 1.0e-5).ln();
        let mut best = f64::INFINITY;
        let mut best_order = 0.0;
        for (order, rdp) in spend.orders.iter().zip(spend.epsilons.iter()) {
            let candidate = rdp + log_inv_delta / (order - 1.0);
            if candidate < best {
                best = candidate;
                best_order = *order;
            }
        }

        assert!(
            approx_eq(best, 1.179_900_673_983, 1.0e-9),
            "classic-conversion epsilon must reproduce the published TF Privacy \
             value 1.18, got {best} at alpha={best_order}"
        );
        assert_eq!(best_order, 17.0, "TF Privacy also selects alpha = 17");

        // The crate's own (tighter) conversion must sit strictly below the
        // classic one and therefore remain a valid guarantee.
        let tight = accountant.to_epsilon_delta(1.0e-5).expect("conversion");
        assert!(
            tight.epsilon < best,
            "CKS conversion must be tighter than the classic one: {} vs {best}",
            tight.epsilon
        );
    }

    #[test]
    fn test_per_step_rdp_matches_quadrature_validated_golden_values() {
        // Golden per-step RDP values for the sampled Gaussian mechanism.
        //
        // Provenance: each value was cross-checked against a direct Simpson
        // quadrature of the Renyi divergence integral
        //
        //     exp((alpha - 1) * rdp)
        //       = E_{x ~ N(0, sigma^2)} [ ((1 - q) + q e^{(2x - 1)/(2 sigma^2)})^alpha ]
        //
        // which shares no code path with the binomial expansion implemented
        // here; agreement was better than 1e-12 relative in every case.
        let cases: [(f64, f64, f64, f64); 5] = [
            (2.0, 1.0, 0.01, 1.718_134_220_745_140_6e-4),
            (8.0, 1.0, 0.01, 8.936_439_076_060_275e-4),
            (16.0, 1.0, 0.01, 3.087_850_783_696_245),
            (12.0, 1.1, 256.0 / 60_000.0, 1.557_401_620_924_204_6e-4),
            (24.0, 2.0, 0.01, 3.663_592_275_686_629e-4),
        ];

        for (alpha, sigma, q, expected) in cases {
            let actual = rdp_subsampled_gaussian_step(alpha, sigma, q).expect("valid parameters");
            let relative = (actual - expected).abs() / expected;
            assert!(
                relative < 1.0e-12,
                "rdp(alpha={alpha}, sigma={sigma}, q={q}) = {actual}, expected {expected} \
                 (relative error {relative:e})"
            );
        }
    }

    #[test]
    fn test_kernel_converges_to_the_pure_gaussian_closed_form() {
        // As q -> 1 the sampled Gaussian *is* the Gaussian mechanism, whose
        // RDP has the closed form alpha / (2 sigma^2) (Mironov 2017,
        // Proposition 7). The binomial expansion must converge to it, which
        // pins the kernel against a formula it does not share any code with.
        for sigma in [0.5_f64, 1.0, 1.1, 2.0] {
            for alpha in [2.0_f64, 4.0, 8.0, 16.0, 32.0] {
                let closed_form = alpha / (2.0 * sigma * sigma);
                let expansion =
                    rdp_subsampled_gaussian_step(alpha, sigma, 1.0 - 1.0e-10).expect("valid");
                let relative = (expansion - closed_form).abs() / closed_form;
                assert!(
                    relative < 1.0e-8,
                    "expansion {expansion} must converge to {closed_form} \
                     (sigma={sigma}, alpha={alpha}, relative error {relative:e})"
                );

                // Exactly q = 1 takes the closed-form branch.
                let exact = rdp_subsampled_gaussian_step(alpha, sigma, 1.0).expect("valid");
                assert!(approx_eq(exact, closed_form, 1.0e-12));
            }
        }
    }

    #[test]
    fn test_golden_epsilon_for_the_canonical_dp_sgd_configuration() {
        // Pinned epsilon for sigma = 1.0, q = 0.01, delta = 1e-5 over the
        // canonical [`DEFAULT_ALPHAS`] grid with the CKS conversion.
        //
        // These constants pin *this crate's* configuration (integer-order
        // bound with `ceil` for fractional alphas, DEFAULT_ALPHAS grid, CKS
        // conversion). They are not published Opacus/TF-Privacy outputs -- the
        // external anchor is
        // `test_kernel_reproduces_the_published_tensorflow_privacy_reference`,
        // which validates the underlying kernel; these values then follow from
        // it by composition and conversion.
        let expected: [(usize, f64, f64); 4] = [
            (1, 0.956_281_055_679, 10.0),
            (10, 1.064_496_195_732, 9.0),
            (100, 1.224_845_779_636, 9.0),
            (1000, 2.107_753_075_452, 8.0),
        ];

        for (steps, epsilon, order) in expected {
            let mut accountant = RenyiAccountant::with_default_orders();
            accountant
                .add_subsampled_gaussian(1.0, 0.01, steps)
                .expect("composition");
            let result = accountant.to_epsilon_delta(1.0e-5).expect("conversion");
            assert!(
                approx_eq(result.epsilon, epsilon, 1.0e-9),
                "T={steps}: epsilon {} must equal the golden value {epsilon}",
                result.epsilon
            );
            assert_eq!(
                result.best_order, order,
                "T={steps}: optimal Renyi order changed"
            );
        }
    }

    #[test]
    fn test_small_q_uses_exact_expansion_not_a_shortcut() {
        // Regression for the deleted "small q" analytical shortcut, which
        // returned q^2 * alpha / (2 sigma^2) below q = 1e-6 and under-reported
        // the true RDP by orders of magnitude. The exact expansion must be
        // continuous across the old threshold and must dominate the shortcut.
        let sigma = 1.0_f64;
        let alpha = 8.0_f64;
        for &q in &[9.0e-7_f64, 1.0e-6, 1.1e-6] {
            let exact = rdp_subsampled_gaussian_step(alpha, sigma, q).expect("valid parameters");
            let old_shortcut = q * q * alpha / (2.0 * sigma * sigma);
            assert!(
                exact >= old_shortcut,
                "exact bound {exact} must not fall below the discarded shortcut {old_shortcut}"
            );
            assert!(exact.is_finite() && exact > 0.0);
        }

        // Continuity across the old threshold: relative change is tiny.
        let below = rdp_subsampled_gaussian_step(alpha, sigma, 9.99e-7).expect("valid");
        let above = rdp_subsampled_gaussian_step(alpha, sigma, 1.01e-6).expect("valid");
        assert!(
            (above - below).abs() / above < 0.05,
            "kernel must be continuous across the removed threshold: {below} vs {above}"
        );
    }

    #[test]
    fn test_fractional_orders_are_conservative_upper_bounds() {
        // RDP is non-decreasing in the order, so the value reported for a
        // fractional alpha must sit at or above the value at floor(alpha)
        // and equal the value at ceil(alpha).
        let sigma = 1.0_f64;
        let q = 0.01_f64;

        let at_two = rdp_subsampled_gaussian_step(2.0, sigma, q).expect("valid");
        let at_three = rdp_subsampled_gaussian_step(3.0, sigma, q).expect("valid");
        let at_two_five = rdp_subsampled_gaussian_step(2.5, sigma, q).expect("valid");
        assert!(at_two_five >= at_two);
        assert!(approx_eq(at_two_five, at_three, 1.0e-12));

        // Orders in (1, 2) previously extrapolated with a negative weight,
        // producing values below the true RDP. They must now be bounded by
        // the alpha = 2 value.
        for &alpha in &[1.25_f64, 1.5, 1.75] {
            let value = rdp_subsampled_gaussian_step(alpha, sigma, q).expect("valid");
            assert!(value > 0.0, "order {alpha} must have positive RDP");
            assert!(
                approx_eq(value, at_two, 1.0e-12),
                "order {alpha} must be bounded by the alpha=2 value"
            );
        }
    }

    #[test]
    fn test_tiny_noise_multiplier_reports_infinite_epsilon() {
        // Previously a sigma below 0.5 silently returned a capped constant.
        // The accountant must instead report saturation and an infinite
        // epsilon: fail closed, never a fabricated finite budget.
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.0e-8, 0.5, 1000)
            .expect("composition should be accepted");
        assert!(accountant.is_saturated());

        let conversion = accountant.to_epsilon_delta(1.0e-5).expect("conversion");
        assert!(
            conversion.epsilon.is_infinite(),
            "saturated accountant must report infinite epsilon, got {}",
            conversion.epsilon
        );

        accountant.reset();
        assert!(!accountant.is_saturated());
    }

    #[test]
    fn test_moderately_small_sigma_still_computed_exactly() {
        // sigma = 0.4 used to hit the MIN_SAFE_SIGMA shortcut; the log-space
        // routine handles it exactly and must produce a finite bound.
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(0.4, 0.01, 100)
            .expect("composition");
        assert!(!accountant.is_saturated());
        let conversion = accountant.to_epsilon_delta(1.0e-5).expect("conversion");
        assert!(conversion.epsilon.is_finite() && conversion.epsilon > 0.0);
    }

    #[test]
    fn test_cks_conversion_is_tighter_than_classic() {
        let mut accountant = RenyiAccountant::with_default_orders();
        accountant
            .add_subsampled_gaussian(1.0, 0.01, 1000)
            .expect("composition");
        let delta = 1.0e-5_f64;
        let converted = accountant.to_epsilon_delta(delta).expect("conversion");

        // Classic Mironov conversion over the same spend.
        let spend = accountant.current_spend();
        let log_inv_delta = (1.0 / delta).ln();
        let mut classic = f64::INFINITY;
        for (i, &alpha) in spend.orders.iter().enumerate() {
            let candidate = spend.epsilons[i] + log_inv_delta / (alpha - 1.0);
            if candidate < classic {
                classic = candidate;
            }
        }

        assert!(
            converted.epsilon <= classic + 1.0e-12,
            "CKS conversion {} must not exceed the classic bound {classic}",
            converted.epsilon
        );
        assert!(converted.epsilon > 0.0);
    }

    #[test]
    fn test_log_sum_exp_handles_extreme_inputs() {
        // Internal helper coverage to confirm numerical stability.
        let values = [1.0e6_f64, 1.0e6 + 1.0, 1.0e6 + 2.0];
        let result = log_sum_exp(&values);
        assert!(result.is_finite());
        // The result should be roughly max + ln(1 + e + e^2).
        let expected = 1.0e6 + (1.0_f64 + std::f64::consts::E + std::f64::consts::E.powi(2)).ln();
        assert!(approx_eq(result, expected, 1.0e-6));

        // Empty input is well-defined as -infinity.
        let empty: [f64; 0] = [];
        assert!(log_sum_exp(&empty).is_infinite());
    }

    #[test]
    fn test_log_binom_known_values() {
        // C(10, 0) = 1 -> log = 0
        assert!(approx_eq(log_binom_coefficient(10.0, 0), 0.0, 1.0e-12));
        // C(10, 1) = 10 -> log = ln(10)
        assert!(approx_eq(
            log_binom_coefficient(10.0, 1),
            10.0_f64.ln(),
            1.0e-12
        ));
        // C(5, 2) = 10 -> log = ln(10)
        assert!(approx_eq(
            log_binom_coefficient(5.0, 2),
            10.0_f64.ln(),
            1.0e-12
        ));
        // C(8, 4) = 70 -> log = ln(70)
        assert!(approx_eq(
            log_binom_coefficient(8.0, 4),
            70.0_f64.ln(),
            1.0e-12
        ));
    }
}
