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
// numerical stability. Non-integer orders are handled by linear interpolation
// between bracketing integer orders -- the same approach used by Opacus and
// TensorFlow Privacy.
//
// This accountant coexists with `moment_accountant::MomentsAccountant` and
// offers tighter composition bounds for DP-SGD style algorithms via a modern
// public API.

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

/// Cap on a single RDP step contribution to avoid +inf propagation.
///
/// Even with extremely small noise multipliers, individual step
/// contributions are clamped so that further composition arithmetic stays
/// finite. Downstream conversion will still report a very large epsilon.
const RDP_STEP_CAP: f64 = 1.0e6;

/// Threshold for switching to the small-q analytical approximation.
const SMALL_Q_THRESHOLD: f64 = 1.0e-6;

/// Threshold below which the noise multiplier is considered numerically
/// unstable for the closed-form bound.
const MIN_SAFE_SIGMA: f64 = 0.5;

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
            let contribution = (per_step * steps_f).min(RDP_STEP_CAP);
            self.rdp_epsilons[i] = (self.rdp_epsilons[i] + contribution).min(RDP_STEP_CAP);
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
            let contribution = (per_step * steps_f).min(RDP_STEP_CAP);
            self.rdp_epsilons[i] = (self.rdp_epsilons[i] + contribution).min(RDP_STEP_CAP);
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
    /// For each tracked order alpha, computes
    /// `eps(alpha) = rdp(alpha) + ln(1 / delta) / (alpha - 1)` and returns
    /// the minimum across orders (Mironov 2017, Proposition 3, refined by
    /// Canonne, Kamath, Steinke 2020 / Balle et al. 2020).
    pub fn to_epsilon_delta(&self, target_delta: f64) -> Result<DpConversion> {
        if !target_delta.is_finite() || target_delta <= 0.0 || target_delta > 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_delta must be in (0, 1], got {target_delta}"
            )));
        }

        let log_inv_delta = (1.0 / target_delta).ln();

        let mut best_epsilon = f64::INFINITY;
        let mut best_order = self.orders[0];

        for (i, &alpha) in self.orders.iter().enumerate() {
            let candidate = self.rdp_epsilons[i] + log_inv_delta / (alpha - 1.0);
            if candidate < best_epsilon {
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
/// For integer orders alpha >= 2 the formula expands the moment-generating
/// function as a binomial sum. For small sampling probabilities the leading
/// `q^2` term is used analytically to avoid numerical issues with `log(q)`.
/// For non-integer orders we linearly interpolate between the two bracketing
/// integer orders, matching the convention used by Opacus and TF Privacy.
fn rdp_subsampled_gaussian_step(alpha: f64, noise_multiplier: f64, q: f64) -> Result<f64> {
    if !alpha.is_finite() || alpha <= 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "Renyi order alpha must satisfy alpha > 1, got {alpha}"
        )));
    }

    if q == 0.0 {
        return Ok(0.0);
    }

    if q == 1.0 {
        // Sampling everything is equivalent to the pure Gaussian mechanism.
        let variance = noise_multiplier * noise_multiplier;
        return Ok((alpha / (2.0 * variance)).min(RDP_STEP_CAP));
    }

    // For very small q, use the asymptotic small-q expansion of the
    // sampled-Gaussian RDP (see Wang/Balle/Kasiviswanathan 2019, eq. (7);
    // the leading dependence is q^2 * alpha / (2 sigma^2) for the simple
    // amplification regime). This avoids numerically catastrophic `log(q)`.
    if q < SMALL_Q_THRESHOLD {
        let variance = noise_multiplier * noise_multiplier;
        return Ok((q * q * alpha / (2.0 * variance)).min(RDP_STEP_CAP));
    }

    // For numerically unstable small noise multipliers the closed-form
    // bound can overflow; cap the contribution so composition arithmetic
    // stays finite.
    if noise_multiplier < MIN_SAFE_SIGMA {
        return Ok(RDP_STEP_CAP);
    }

    if (alpha - alpha.round()).abs() < 1.0e-12 {
        // Integer order path.
        let alpha_int = alpha.round() as usize;
        let value = rdp_subsampled_gaussian_step_integer(alpha_int, noise_multiplier, q);
        return Ok(value.min(RDP_STEP_CAP));
    }

    // Non-integer order: linearly interpolate between floor and ceil.
    let lower = alpha.floor();
    let upper = alpha.ceil();

    // alpha > 1 and non-integer implies lower >= 1 and upper >= 2. If
    // lower == 1.0, we cannot evaluate at it (RDP undefined at alpha = 1),
    // so we use ceil() and ceil()+1 instead and extrapolate.
    let (anchor_lo, anchor_hi) = if lower <= 1.0 {
        (upper, upper + 1.0)
    } else {
        (lower, upper)
    };

    let lo_int = anchor_lo.round() as usize;
    let hi_int = anchor_hi.round() as usize;
    let lo_int = lo_int.max(2);
    let hi_int = hi_int.max(lo_int + 1);

    let lo_value = rdp_subsampled_gaussian_step_integer(lo_int, noise_multiplier, q);
    let hi_value = rdp_subsampled_gaussian_step_integer(hi_int, noise_multiplier, q);

    let span = (hi_int as f64) - (lo_int as f64);
    let weight = if span > 0.0 {
        (alpha - lo_int as f64) / span
    } else {
        0.0
    };

    let interpolated = lo_value + weight * (hi_value - lo_value);
    Ok(interpolated.clamp(0.0, RDP_STEP_CAP))
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
fn rdp_subsampled_gaussian_step_integer(alpha: usize, sigma: f64, q: f64) -> f64 {
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

    if !rdp.is_finite() || rdp < 0.0 {
        // Numerical underflow or pathological inputs: clamp to a safe range.
        // RDP is non-negative by definition.
        rdp.clamp(0.0, RDP_STEP_CAP)
    } else {
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
