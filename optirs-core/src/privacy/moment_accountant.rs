// Moment Accountant for Differential Privacy
//
// This module implements the moments accountant of Abadi et al., "Deep
// Learning with Differential Privacy" (CCS 2016) for the Gaussian mechanism
// with Poisson subsampling.
//
// # Relationship to the Renyi accountant
//
// The log moment of order `lambda` and the Renyi differential privacy of
// order `alpha = lambda + 1` are the same quantity up to a factor:
//
// ```text
// alpha_M(lambda) = lambda * rdp(lambda + 1)
// ```
//
// so a moments accountant and an RDP accountant differ only in bookkeeping
// and in the final conversion. This module therefore composes the *exact*
// per-step bound implemented (and unit-tested) in
// [`crate::privacy::renyi_accountant`] rather than a separate closed form:
// having two divergent implementations of the same bound is precisely how the
// earlier revision came to under-report epsilon by a factor of ~2 while a
// second copy over-reported it by ~165x.
//
// Conversion to `(epsilon, delta)`-DP uses the classic bound
// `epsilon = min_alpha [ rdp(alpha) + ln(1/delta) / (alpha - 1) ]`
// (Mironov 2017, Proposition 3), which is what the moments accountant
// literature states. `RenyiAccountant::to_epsilon_delta` applies the slightly
// tighter Canonne-Kamath-Steinke conversion; both are valid upper bounds.
//
// `delta` is a *reporting* parameter: the same composed mechanism can be
// described at any delta, trading it against epsilon. It is never consumed
// additively.

use crate::error::{OptimError, Result};
use std::collections::HashMap;

use super::accountant::AccountingSegment;
use super::renyi_accountant::rdp_subsampled_gaussian_step_integer;

/// Highest integer Renyi order evaluated by the accountant.
const DEFAULT_MAX_ORDER: usize = 64;

/// Accumulated RDP above which the accountant reports *no* privacy at all.
///
/// Matches [`crate::privacy::renyi_accountant`] so the two accountants agree
/// about when a configuration has stopped providing a usable guarantee.
/// Reporting `+infinity` is the fail-closed direction: every budget check
/// then treats the budget as exhausted.
const RDP_SATURATION_THRESHOLD: f64 = 1.0e12;

/// Moment accountant for tracking privacy loss.
#[derive(Debug, Clone)]
pub struct MomentsAccountant {
    /// Noise multiplier (sigma), relative to the clipping norm.
    noise_multiplier: f64,

    /// Delta at which epsilon is reported.
    target_delta: f64,

    /// Batch size.
    batch_size: usize,

    /// Total dataset size.
    dataset_size: usize,

    /// Sampling probability (q = batch_size / dataset_size).
    sampling_probability: f64,

    /// Maximum integer order evaluated.
    max_order: usize,
}

/// Privacy analysis result from the moments accountant.
#[derive(Debug, Clone)]
pub struct PrivacyAnalysis {
    /// Best epsilon for the given delta.
    pub epsilon: f64,

    /// Delta the epsilon is reported at.
    pub delta: f64,

    /// Number of steps analyzed.
    pub steps: usize,

    /// Renyi order that minimised epsilon.
    pub optimal_order: usize,

    /// Log moments `alpha_M(alpha - 1)` per order.
    pub log_moments: HashMap<usize, f64>,

    /// Measured amplification by subsampling: the ratio of the epsilon this
    /// mechanism would spend without subsampling (q = 1) to the epsilon it
    /// actually spends. Always >= 1; larger means more amplification.
    pub amplification_factor: f64,

    /// Ratio of the composed epsilon to the basic-composition epsilon
    /// (`steps` times the single-step epsilon). Values below 1 quantify how
    /// much tighter the moments accountant is than naive composition.
    pub bound_tightness: f64,
}

/// Advanced privacy composition analysis.
#[derive(Debug, Clone)]
pub struct CompositionAnalysis {
    /// Mechanism parameters for each composed segment.
    pub mechanisms: Vec<MechanismParameters>,

    /// Composed privacy guarantee.
    pub composed_epsilon: f64,

    /// Delta the epsilon is reported at.
    pub composed_delta: f64,

    /// Total number of mechanism applications.
    pub num_compositions: usize,

    /// Whether the composition mixes different parameters.
    pub is_heterogeneous: bool,
}

/// Parameters for a single mechanism application.
#[derive(Debug, Clone)]
pub struct MechanismParameters {
    /// Noise multiplier.
    pub noise_multiplier: f64,

    /// Sampling probability.
    pub sampling_probability: f64,

    /// Sensitivity of the query.
    pub sensitivity: f64,

    /// Number of applications.
    pub applications: usize,
}

/// Privacy budget status for real-time monitoring.
#[derive(Debug, Clone)]
pub struct PrivacyBudgetStatus {
    /// Remaining epsilon budget against the caller's target.
    pub epsilon_remaining: f64,

    /// Epsilon spent so far.
    pub epsilon_consumed: f64,

    /// Delta the epsilon is reported at.
    pub delta: f64,

    /// Utilization ratio (0.0 to 1.0, clamped).
    pub utilization_ratio: f64,

    /// Current budget status.
    pub status: BudgetStatus,

    /// Number of steps analyzed.
    pub steps_analyzed: usize,

    /// Largest number of steps that stays within the target epsilon.
    pub recommended_max_steps: usize,
}

/// Privacy budget status levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetStatus {
    /// Healthy: < 50% budget used.
    Healthy,

    /// Moderate: 50-80% budget used.
    Moderate,

    /// Critical: 80-95% budget used.
    Critical,

    /// Exhausted: > 95% budget used.
    Exhausted,
}

impl MomentsAccountant {
    /// Create a new moments accountant.
    ///
    /// The parameters are validated lazily: every method that reports a
    /// privacy number calls [`validate_configuration`](Self::validate_configuration)
    /// first and returns an error for an unusable configuration, so an
    /// invalid accountant can never produce a finite epsilon.
    pub fn new(
        noise_multiplier: f64,
        target_delta: f64,
        batch_size: usize,
        dataset_size: usize,
    ) -> Self {
        let sampling_probability = if dataset_size == 0 {
            0.0
        } else {
            (batch_size as f64 / dataset_size as f64).min(1.0)
        };

        Self {
            noise_multiplier,
            target_delta,
            batch_size,
            dataset_size,
            sampling_probability,
            max_order: DEFAULT_MAX_ORDER,
        }
    }

    /// Noise multiplier this accountant was built with.
    pub fn noise_multiplier(&self) -> f64 {
        self.noise_multiplier
    }

    /// Sampling probability `q = batch_size / dataset_size`.
    pub fn sampling_probability(&self) -> f64 {
        self.sampling_probability
    }

    /// Delta at which epsilon is reported.
    pub fn target_delta(&self) -> f64 {
        self.target_delta
    }

    /// Get privacy spent after a given number of steps.
    ///
    /// Returns `(epsilon, delta)`. Delta is the reporting parameter supplied
    /// at construction, not an additively consumed budget.
    pub fn get_privacy_spent(&self, steps: usize) -> Result<(f64, f64)> {
        if steps == 0 {
            return Ok((0.0, self.target_delta));
        }

        let analysis = self.analyze_privacy(steps)?;
        Ok((analysis.epsilon, self.target_delta))
    }

    /// Perform a comprehensive privacy analysis for `steps` applications.
    pub fn analyze_privacy(&self, steps: usize) -> Result<PrivacyAnalysis> {
        self.validate_configuration()?;

        let mut log_moments = HashMap::new();
        for order in 2..=self.max_order {
            log_moments.insert(order, self.compute_log_moment(order, steps)?);
        }

        let (epsilon, optimal_order) =
            self.epsilon_from_log_moments(&log_moments, self.target_delta);

        let amplification_factor = self.measure_amplification(steps, self.target_delta)?;
        let bound_tightness = self.measure_bound_tightness(steps, epsilon, self.target_delta)?;

        Ok(PrivacyAnalysis {
            epsilon,
            delta: self.target_delta,
            steps,
            optimal_order,
            log_moments,
            amplification_factor,
            bound_tightness,
        })
    }

    /// Compute the log moment `alpha_M(order - 1)` after `steps` applications.
    ///
    /// The log moment composes additively over independent applications, so
    /// this is `steps` times the single-step value.
    pub fn compute_log_moment(&self, order: usize, steps: usize) -> Result<f64> {
        if order < 2 {
            return Err(OptimError::InvalidConfig(
                "Moment order must be at least 2".to_string(),
            ));
        }

        let single =
            self.single_step_log_moment(order, self.noise_multiplier, self.sampling_probability);
        Ok(single * steps as f64)
    }

    /// Single-step log moment for an arbitrary mechanism description.
    fn single_step_log_moment(&self, order: usize, sigma: f64, q: f64) -> f64 {
        let rdp = rdp_subsampled_gaussian_step_integer(order, sigma, q);
        rdp * (order as f64 - 1.0)
    }

    /// Compose an append-only ledger of segments into a single epsilon.
    ///
    /// Returns `f64::INFINITY` if the composition saturates (for example a
    /// vanishing noise multiplier): reporting an infinite epsilon is the
    /// fail-closed direction and makes every budget check treat the budget as
    /// exhausted.
    pub fn compose_segments(
        &self,
        segments: &[AccountingSegment],
        target_delta: f64,
    ) -> Result<f64> {
        if !target_delta.is_finite() || target_delta <= 0.0 || target_delta >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_delta must be in (0, 1), got {target_delta}"
            )));
        }

        if segments.is_empty() {
            return Ok(0.0);
        }

        let mut best_epsilon = f64::INFINITY;
        for order in 2..=self.max_order {
            let alpha = order as f64;
            let mut rdp_total = 0.0;
            for segment in segments {
                if !segment.noise_multiplier.is_finite() || segment.noise_multiplier <= 0.0 {
                    return Err(OptimError::InvalidParameter(format!(
                        "segment noise_multiplier must be positive and finite, got {}",
                        segment.noise_multiplier
                    )));
                }
                let per_step = rdp_subsampled_gaussian_step_integer(
                    order,
                    segment.noise_multiplier,
                    segment.sampling_probability,
                );
                rdp_total += per_step * segment.steps as f64;
            }

            if !rdp_total.is_finite() || rdp_total > RDP_SATURATION_THRESHOLD {
                return Ok(f64::INFINITY);
            }

            let candidate = rdp_total + (1.0 / target_delta).ln() / (alpha - 1.0);
            if candidate.is_finite() && candidate < best_epsilon {
                best_epsilon = candidate;
            }
        }

        Ok(best_epsilon.max(0.0))
    }

    /// Convert per-order log moments into the tightest epsilon.
    fn epsilon_from_log_moments(
        &self,
        log_moments: &HashMap<usize, f64>,
        delta: f64,
    ) -> (f64, usize) {
        let log_inv_delta = (1.0 / delta).ln();
        let mut best_epsilon = f64::INFINITY;
        let mut best_order = 2;
        let mut saturated = false;

        for (&order, &log_moment) in log_moments {
            let alpha = order as f64;
            let rdp = log_moment / (alpha - 1.0);
            if !rdp.is_finite() || rdp > RDP_SATURATION_THRESHOLD {
                saturated = true;
                continue;
            }
            let epsilon = rdp + log_inv_delta / (alpha - 1.0);
            if epsilon.is_finite() && epsilon < best_epsilon {
                best_epsilon = epsilon;
                best_order = order;
            }
        }

        if best_epsilon.is_infinite() && saturated {
            return (f64::INFINITY, best_order);
        }

        (best_epsilon.max(0.0), best_order)
    }

    /// Amplification actually obtained from subsampling: the ratio between
    /// the epsilon of the same mechanism run without subsampling and the
    /// epsilon it spends with subsampling.
    fn measure_amplification(&self, steps: usize, delta: f64) -> Result<f64> {
        if self.sampling_probability >= 1.0 {
            return Ok(1.0);
        }

        let subsampled = self.epsilon_for(
            self.noise_multiplier,
            self.sampling_probability,
            steps,
            delta,
        );
        let full = self.epsilon_for(self.noise_multiplier, 1.0, steps, delta);

        if subsampled <= 0.0 || !subsampled.is_finite() || !full.is_finite() {
            return Ok(1.0);
        }

        Ok((full / subsampled).max(1.0))
    }

    /// How much tighter the composed bound is than basic composition.
    fn measure_bound_tightness(&self, steps: usize, epsilon: f64, delta: f64) -> Result<f64> {
        if steps == 0 || !epsilon.is_finite() {
            return Ok(1.0);
        }

        let single_step =
            self.epsilon_for(self.noise_multiplier, self.sampling_probability, 1, delta);
        let basic = single_step * steps as f64;

        if basic <= 0.0 || !basic.is_finite() {
            return Ok(1.0);
        }

        Ok(epsilon / basic)
    }

    /// Epsilon for a homogeneous run described directly by its parameters.
    fn epsilon_for(&self, sigma: f64, q: f64, steps: usize, delta: f64) -> f64 {
        let log_inv_delta = (1.0 / delta).ln();
        let mut best = f64::INFINITY;
        for order in 2..=self.max_order {
            let alpha = order as f64;
            let rdp = rdp_subsampled_gaussian_step_integer(order, sigma, q) * steps as f64;
            let candidate = rdp + log_inv_delta / (alpha - 1.0);
            if candidate.is_finite() && candidate < best {
                best = candidate;
            }
        }
        best.max(0.0)
    }

    /// Compose a heterogeneous sequence of mechanisms.
    pub fn analyze_heterogeneous_composition(
        &self,
        mechanisms: &[MechanismParameters],
        target_delta: f64,
    ) -> Result<CompositionAnalysis> {
        if !target_delta.is_finite() || target_delta <= 0.0 || target_delta >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_delta must be in (0, 1), got {target_delta}"
            )));
        }

        let mut segments = Vec::with_capacity(mechanisms.len());
        for mechanism in mechanisms {
            if !mechanism.sensitivity.is_finite() || mechanism.sensitivity <= 0.0 {
                return Err(OptimError::InvalidParameter(format!(
                    "mechanism sensitivity must be positive and finite, got {}",
                    mechanism.sensitivity
                )));
            }
            if !mechanism.sampling_probability.is_finite()
                || !(0.0..=1.0).contains(&mechanism.sampling_probability)
            {
                return Err(OptimError::InvalidParameter(format!(
                    "mechanism sampling_probability must be in [0, 1], got {}",
                    mechanism.sampling_probability
                )));
            }
            segments.push(AccountingSegment {
                // A sensitivity other than 1 rescales the effective noise
                // multiplier: sigma_effective = sigma / sensitivity.
                noise_multiplier: mechanism.noise_multiplier / mechanism.sensitivity,
                sampling_probability: mechanism.sampling_probability,
                steps: mechanism.applications,
            });
        }

        let composed_epsilon = self.compose_segments(&segments, target_delta)?;

        Ok(CompositionAnalysis {
            mechanisms: mechanisms.to_vec(),
            composed_epsilon,
            composed_delta: target_delta,
            num_compositions: mechanisms.iter().map(|m| m.applications).sum(),
            is_heterogeneous: mechanisms.len() > 1,
        })
    }

    /// Compose a heterogeneous sequence at the accountant's own delta.
    pub fn track_heterogeneous_composition(
        &mut self,
        mechanisms: &[MechanismParameters],
    ) -> Result<CompositionAnalysis> {
        self.analyze_heterogeneous_composition(mechanisms, self.target_delta)
    }

    /// Epsilon after `steps` applications, reported at a custom delta.
    pub fn compute_tight_epsilon_delta_bound(
        &self,
        steps: usize,
        target_delta: f64,
    ) -> Result<f64> {
        self.validate_configuration()?;
        if !target_delta.is_finite() || target_delta <= 0.0 || target_delta >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_delta must be in (0, 1), got {target_delta}"
            )));
        }
        if steps == 0 {
            return Ok(0.0);
        }

        Ok(self.epsilon_for(
            self.noise_multiplier,
            self.sampling_probability,
            steps,
            target_delta,
        ))
    }

    /// Real-time privacy budget monitoring against an explicit target.
    ///
    /// `target_epsilon` is a required parameter: an accountant has no way to
    /// know the caller's budget, and defaulting to 1.0 (as an earlier
    /// revision did) silently reports the wrong utilization for every other
    /// budget.
    pub fn get_privacy_budget_status(
        &self,
        steps: usize,
        target_epsilon: f64,
    ) -> Result<PrivacyBudgetStatus> {
        if !target_epsilon.is_finite() || target_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_epsilon must be a positive finite number, got {target_epsilon}"
            )));
        }

        let (epsilon_consumed, delta) = self.get_privacy_spent(steps)?;
        let utilization = if epsilon_consumed.is_finite() {
            (epsilon_consumed / target_epsilon).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let status = if utilization < 0.5 {
            BudgetStatus::Healthy
        } else if utilization < 0.8 {
            BudgetStatus::Moderate
        } else if utilization < 0.95 {
            BudgetStatus::Critical
        } else {
            BudgetStatus::Exhausted
        };

        Ok(PrivacyBudgetStatus {
            epsilon_remaining: (target_epsilon - epsilon_consumed).max(0.0),
            epsilon_consumed,
            delta,
            utilization_ratio: utilization,
            status,
            steps_analyzed: steps,
            recommended_max_steps: self.estimate_max_steps(target_epsilon)?,
        })
    }

    /// Largest number of steps whose composed epsilon stays within
    /// `target_epsilon`.
    pub fn estimate_max_steps(&self, target_epsilon: f64) -> Result<usize> {
        self.validate_configuration()?;
        if !target_epsilon.is_finite() || target_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_epsilon must be a positive finite number, got {target_epsilon}"
            )));
        }

        // Exponential search for an upper bracket, then binary search.
        let mut high = 1usize;
        let mut result = 0usize;
        while high <= 1 << 30 {
            let (epsilon, _) = self.get_privacy_spent(high)?;
            if epsilon > target_epsilon {
                break;
            }
            result = high;
            high = high.saturating_mul(2);
        }

        let mut low = result + 1;
        let mut hi = high.min(1 << 30);
        while low <= hi {
            let mid = low + (hi - low) / 2;
            let (epsilon, _) = self.get_privacy_spent(mid)?;
            if epsilon <= target_epsilon {
                result = mid;
                low = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
        }

        Ok(result)
    }

    /// Renyi orders evaluated by this accountant.
    pub fn get_computed_orders(&self) -> Vec<f64> {
        (2..=self.max_order).map(|x| x as f64).collect()
    }

    /// Validate the accountant configuration.
    pub fn validate_configuration(&self) -> Result<()> {
        if !self.noise_multiplier.is_finite() || self.noise_multiplier <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "Noise multiplier must be positive and finite".to_string(),
            ));
        }

        if !self.target_delta.is_finite() || self.target_delta <= 0.0 || self.target_delta >= 1.0 {
            return Err(OptimError::InvalidConfig(
                "Delta must be in (0, 1)".to_string(),
            ));
        }

        if self.batch_size == 0 || self.dataset_size == 0 {
            return Err(OptimError::InvalidConfig(
                "Batch size and dataset size must be positive".to_string(),
            ));
        }

        if self.batch_size > self.dataset_size {
            return Err(OptimError::InvalidConfig(
                "Batch size cannot exceed dataset size".to_string(),
            ));
        }

        Ok(())
    }

    /// Summary of the privacy analysis after `steps` applications.
    pub fn get_analysis_summary(&self, steps: usize) -> Result<PrivacyAnalysisSummary> {
        let analysis = self.analyze_privacy(steps)?;
        let privacy_per_step = if steps == 0 {
            0.0
        } else {
            analysis.epsilon / steps as f64
        };

        Ok(PrivacyAnalysisSummary {
            epsilon: analysis.epsilon,
            delta: analysis.delta,
            steps,
            noise_multiplier: self.noise_multiplier,
            sampling_probability: self.sampling_probability,
            optimal_order: analysis.optimal_order,
            amplification_factor: analysis.amplification_factor,
            bound_tightness: analysis.bound_tightness,
            privacy_per_step,
        })
    }
}

/// Privacy analysis summary.
#[derive(Debug, Clone)]
pub struct PrivacyAnalysisSummary {
    /// Composed epsilon.
    pub epsilon: f64,
    /// Delta the epsilon is reported at.
    pub delta: f64,
    /// Steps analysed.
    pub steps: usize,
    /// Noise multiplier used.
    pub noise_multiplier: f64,
    /// Sampling probability used.
    pub sampling_probability: f64,
    /// Renyi order that minimised epsilon.
    pub optimal_order: usize,
    /// Measured amplification by subsampling.
    pub amplification_factor: f64,
    /// Ratio to basic composition.
    pub bound_tightness: f64,
    /// Average epsilon per step (not a per-step guarantee).
    pub privacy_per_step: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moment_accountant_creation() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        assert_eq!(accountant.noise_multiplier(), 1.1);
        assert_eq!(accountant.target_delta(), 1e-5);
        assert!((accountant.sampling_probability() - 256.0 / 50000.0).abs() < 1e-15);
    }

    #[test]
    fn test_privacy_analysis() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        let analysis = accountant.analyze_privacy(100).expect("analysis");

        assert!(analysis.epsilon > 0.0 && analysis.epsilon.is_finite());
        assert_eq!(analysis.delta, 1e-5);
        assert_eq!(analysis.steps, 100);
        assert!(analysis.optimal_order >= 2);
        assert!(analysis.amplification_factor >= 1.0);
        assert!(analysis.bound_tightness > 0.0);
    }

    #[test]
    fn test_log_moment_matches_rdp_relation() {
        // alpha_M(alpha - 1) = (alpha - 1) * rdp(alpha)
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let order = 8usize;
        let log_moment = accountant.compute_log_moment(order, 1).expect("log moment");
        let rdp = rdp_subsampled_gaussian_step_integer(order, 1.0, 0.01);
        assert!((log_moment - rdp * (order as f64 - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_log_moment_is_linear_in_steps() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let one = accountant.compute_log_moment(4, 1).expect("one step");
        let ten = accountant.compute_log_moment(4, 10).expect("ten steps");
        assert!((ten - 10.0 * one).abs() < 1e-12);
    }

    #[test]
    fn test_privacy_spent_grows_with_steps() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        let (epsilon_50, delta) = accountant.get_privacy_spent(50).expect("50 steps");
        assert_eq!(delta, 1e-5);
        assert!(epsilon_50 > 0.0);

        let (epsilon_100, _) = accountant.get_privacy_spent(100).expect("100 steps");
        assert!(epsilon_100 > epsilon_50);
    }

    #[test]
    fn test_zero_steps_costs_nothing() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        let (epsilon, _) = accountant.get_privacy_spent(0).expect("zero steps");
        assert_eq!(epsilon, 0.0);
    }

    #[test]
    fn test_canonical_dp_sgd_reference_value() {
        // Golden value for the canonical DP-SGD configuration
        // (sigma = 1.0, q = 0.01, T = 1000, delta = 1e-5). Reference
        // implementations (Opacus / TF Privacy RDP accountant over the same
        // order grid) report epsilon on the order of a few units for this
        // configuration; the earlier closed form in this file reported 0.41
        // (a ~2x under-report) while a second copy reported 177.
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let (epsilon, _) = accountant.get_privacy_spent(1000).expect("analysis");
        assert!(
            (1.0..=6.0).contains(&epsilon),
            "canonical DP-SGD epsilon out of the expected band: {epsilon}"
        );
    }

    #[test]
    fn test_more_noise_means_less_epsilon() {
        let quiet = MomentsAccountant::new(2.0, 1e-5, 256, 50000);
        let loud = MomentsAccountant::new(1.0, 1e-5, 256, 50000);
        let (eps_quiet, _) = quiet.get_privacy_spent(500).expect("quiet");
        let (eps_loud, _) = loud.get_privacy_spent(500).expect("loud");
        assert!(eps_quiet < eps_loud);
    }

    #[test]
    fn test_smaller_sampling_rate_means_less_epsilon() {
        let sparse = MomentsAccountant::new(1.0, 1e-5, 10, 10_000);
        let dense = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let (eps_sparse, _) = sparse.get_privacy_spent(500).expect("sparse");
        let (eps_dense, _) = dense.get_privacy_spent(500).expect("dense");
        assert!(eps_sparse < eps_dense);
    }

    #[test]
    fn test_invalid_configuration_is_rejected_before_reporting_epsilon() {
        let invalid = MomentsAccountant::new(-1.0, 1e-5, 100, 1000);
        assert!(invalid.validate_configuration().is_err());
        assert!(invalid.get_privacy_spent(10).is_err());

        let oversized_batch = MomentsAccountant::new(1.0, 1e-5, 2000, 1000);
        assert!(oversized_batch.get_privacy_spent(10).is_err());

        let bad_delta = MomentsAccountant::new(1.0, 0.0, 100, 1000);
        assert!(bad_delta.get_privacy_spent(10).is_err());
    }

    #[test]
    fn test_delta_is_a_reporting_parameter() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let loose = accountant
            .compute_tight_epsilon_delta_bound(500, 1e-4)
            .expect("loose delta");
        let tight = accountant
            .compute_tight_epsilon_delta_bound(500, 1e-8)
            .expect("tight delta");
        assert!(tight > loose);
    }

    #[test]
    fn test_compose_segments_matches_homogeneous_analysis() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let segments = vec![AccountingSegment {
            noise_multiplier: 1.0,
            sampling_probability: 0.01,
            steps: 1000,
        }];
        let composed = accountant
            .compose_segments(&segments, 1e-5)
            .expect("composition");
        let (direct, _) = accountant.get_privacy_spent(1000).expect("direct");
        assert!((composed - direct).abs() < 1e-9, "{composed} vs {direct}");
    }

    #[test]
    fn test_compose_segments_is_monotone_in_added_segments() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let one = vec![AccountingSegment {
            noise_multiplier: 1.0,
            sampling_probability: 0.01,
            steps: 100,
        }];
        let two = vec![
            one[0],
            AccountingSegment {
                noise_multiplier: 2.0,
                sampling_probability: 0.02,
                steps: 100,
            },
        ];
        let eps_one = accountant.compose_segments(&one, 1e-5).expect("one");
        let eps_two = accountant.compose_segments(&two, 1e-5).expect("two");
        assert!(eps_two > eps_one);
    }

    #[test]
    fn test_empty_segments_cost_nothing() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        assert_eq!(accountant.compose_segments(&[], 1e-5).expect("empty"), 0.0);
    }

    #[test]
    fn test_vanishing_noise_reports_infinite_epsilon() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 10_000);
        let segments = vec![AccountingSegment {
            noise_multiplier: 1e-9,
            sampling_probability: 0.5,
            steps: 10,
        }];
        let epsilon = accountant
            .compose_segments(&segments, 1e-5)
            .expect("composition");
        assert!(
            epsilon.is_infinite(),
            "a vanishing noise multiplier must report no privacy, got {epsilon}"
        );
    }

    #[test]
    fn test_heterogeneous_composition() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 1000);

        let mechanisms = vec![
            MechanismParameters {
                noise_multiplier: 1.0,
                sampling_probability: 0.1,
                sensitivity: 1.0,
                applications: 50,
            },
            MechanismParameters {
                noise_multiplier: 1.5,
                sampling_probability: 0.1,
                sensitivity: 1.0,
                applications: 50,
            },
        ];

        let analysis = accountant
            .analyze_heterogeneous_composition(&mechanisms, 1e-5)
            .expect("heterogeneous composition");
        assert!(analysis.is_heterogeneous);
        assert_eq!(analysis.num_compositions, 100);
        assert!(analysis.composed_epsilon.is_finite() && analysis.composed_epsilon > 0.0);

        // Composing the noisier half alone must cost strictly less.
        let quieter_only = accountant
            .analyze_heterogeneous_composition(&mechanisms[1..], 1e-5)
            .expect("single mechanism");
        assert!(quieter_only.composed_epsilon < analysis.composed_epsilon);
    }

    #[test]
    fn test_heterogeneous_composition_rejects_bad_parameters() {
        let accountant = MomentsAccountant::new(1.0, 1e-5, 100, 1000);
        let bad = vec![MechanismParameters {
            noise_multiplier: 1.0,
            sampling_probability: 1.5,
            sensitivity: 1.0,
            applications: 10,
        }];
        assert!(accountant
            .analyze_heterogeneous_composition(&bad, 1e-5)
            .is_err());

        let zero_sensitivity = vec![MechanismParameters {
            noise_multiplier: 1.0,
            sampling_probability: 0.1,
            sensitivity: 0.0,
            applications: 10,
        }];
        assert!(accountant
            .analyze_heterogeneous_composition(&zero_sensitivity, 1e-5)
            .is_err());
    }

    #[test]
    fn test_budget_status_uses_caller_target_epsilon() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        let (epsilon, _) = accountant.get_privacy_spent(200).expect("spend");

        let generous = accountant
            .get_privacy_budget_status(200, epsilon * 4.0)
            .expect("generous target");
        let tight = accountant
            .get_privacy_budget_status(200, epsilon * 1.01)
            .expect("tight target");

        assert!(generous.utilization_ratio < tight.utilization_ratio);
        assert_eq!(generous.status, BudgetStatus::Healthy);
        assert_eq!(tight.status, BudgetStatus::Exhausted);
        assert!(generous.recommended_max_steps > 0);
        assert!(accountant.get_privacy_budget_status(200, -1.0).is_err());
    }

    #[test]
    fn test_estimate_max_steps_is_consistent_with_spend() {
        let accountant = MomentsAccountant::new(1.1, 1e-5, 256, 50000);
        let target = 2.0;
        let max_steps = accountant.estimate_max_steps(target).expect("max steps");
        assert!(max_steps > 0);

        let (at_limit, _) = accountant.get_privacy_spent(max_steps).expect("at limit");
        assert!(at_limit <= target);

        let (past_limit, _) = accountant
            .get_privacy_spent(max_steps + 1)
            .expect("past limit");
        assert!(past_limit > target);
    }
}
