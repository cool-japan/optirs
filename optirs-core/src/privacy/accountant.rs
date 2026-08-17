// Unified privacy accounting interface
//
// This module defines the [`PrivacyAccountant`] trait, the single interface
// through which differentially private optimizers track their privacy loss,
// together with an append-only ledger of mechanism *segments*.
//
// # Why an append-only ledger
//
// Privacy that has already been spent cannot be un-spent. Earlier revisions
// of this crate rebuilt the accountant whenever the batch size or the noise
// multiplier changed, which retroactively re-interpreted every past step as
// having been taken with the *new* parameters -- an accounting error that can
// only ever under-report the true loss. A ledger of immutable
// `(noise_multiplier, sampling_probability, steps)` segments makes that class
// of error unrepresentable: changing parameters opens a new segment, and the
// reported epsilon is the composition of all segments ever recorded.
//
// # Accounting methods
//
// Both implemented accountants compose the *same* exact per-step Renyi bound
// for the sampled Gaussian mechanism (see
// [`crate::privacy::renyi_accountant`]); they differ only in the order grid
// and in the RDP-to-DP conversion. Methods for which no correct
// implementation exists return an error from [`build_accountant`] rather than
// silently falling back to a different (and weaker) analysis.

use std::fmt::Debug;

use super::moment_accountant::MomentsAccountant;
use super::renyi_accountant::RenyiAccountant;
use super::AccountingMethod;
use crate::error::{OptimError, Result};

/// One homogeneous run of mechanism applications.
///
/// A segment is immutable once recorded. Consecutive segments with identical
/// parameters are merged by [`PrivacyLedger::record`] purely as a storage
/// optimisation; the composed privacy loss is unaffected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccountingSegment {
    /// Noise multiplier (sigma) relative to the clipping norm.
    pub noise_multiplier: f64,

    /// Per-record sampling probability q used during the segment.
    pub sampling_probability: f64,

    /// Number of mechanism applications in the segment.
    pub steps: usize,
}

/// Append-only ledger of accounting segments.
#[derive(Debug, Clone, Default)]
pub struct PrivacyLedger {
    segments: Vec<AccountingSegment>,
}

impl PrivacyLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Record `steps` applications of a mechanism.
    ///
    /// Returns an error for non-finite or out-of-range parameters so that an
    /// invalid configuration can never enter the ledger.
    pub fn record(
        &mut self,
        noise_multiplier: f64,
        sampling_probability: f64,
        steps: usize,
    ) -> Result<()> {
        if !noise_multiplier.is_finite() || noise_multiplier <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "noise_multiplier must be a positive finite number, got {noise_multiplier}"
            )));
        }
        if !sampling_probability.is_finite() || !(0.0..=1.0).contains(&sampling_probability) {
            return Err(OptimError::InvalidParameter(format!(
                "sampling_probability must be in [0, 1], got {sampling_probability}"
            )));
        }
        if steps == 0 {
            return Ok(());
        }

        if let Some(last) = self.segments.last_mut() {
            if last.noise_multiplier == noise_multiplier
                && last.sampling_probability == sampling_probability
            {
                last.steps = last.steps.saturating_add(steps);
                return Ok(());
            }
        }

        self.segments.push(AccountingSegment {
            noise_multiplier,
            sampling_probability,
            steps,
        });
        Ok(())
    }

    /// All recorded segments, oldest first.
    pub fn segments(&self) -> &[AccountingSegment] {
        &self.segments
    }

    /// Total number of mechanism applications recorded.
    pub fn total_steps(&self) -> usize {
        self.segments
            .iter()
            .fold(0usize, |acc, s| acc.saturating_add(s.steps))
    }

    /// Clear the ledger. Only meaningful when starting a genuinely new
    /// training run on fresh data.
    pub fn clear(&mut self) {
        self.segments.clear();
    }
}

/// Interface implemented by every privacy accountant in the crate.
pub trait PrivacyAccountant: Debug + Send {
    /// Compose `steps` applications of the subsampled Gaussian mechanism.
    fn compose_subsampled_gaussian(
        &mut self,
        noise_multiplier: f64,
        sampling_probability: f64,
        steps: usize,
    ) -> Result<()>;

    /// Total privacy spent so far, as `(epsilon, delta)` for the requested
    /// `target_delta`.
    ///
    /// `delta` is a *reporting parameter*, not an additive spend: the same
    /// composed mechanism can be reported at any delta, trading it against
    /// epsilon. Implementations therefore return the `target_delta` they were
    /// given together with the epsilon it implies.
    fn privacy_spent(&self, target_delta: f64) -> Result<(f64, f64)>;

    /// Privacy that *would* be spent if `steps` further applications of the
    /// described mechanism were composed on top of the current spend --
    /// without recording anything.
    ///
    /// This lets a caller refuse a step *before* releasing its noisy output,
    /// so a budget can be enforced rather than merely observed after the
    /// fact.
    fn projected_privacy_spent(
        &self,
        noise_multiplier: f64,
        sampling_probability: f64,
        steps: usize,
        target_delta: f64,
    ) -> Result<(f64, f64)>;

    /// Number of mechanism applications composed so far.
    fn total_steps(&self) -> usize;

    /// The immutable segment ledger backing this accountant.
    fn segments(&self) -> &[AccountingSegment];

    /// Which accounting method this instance implements.
    fn method(&self) -> AccountingMethod;

    /// Discard all accumulated spend (new training run on fresh data).
    fn reset(&mut self);
}

/// Renyi differential privacy accountant (the crate default).
#[derive(Debug, Clone)]
pub struct RenyiPrivacyAccountant {
    inner: RenyiAccountant,
    ledger: PrivacyLedger,
}

impl RenyiPrivacyAccountant {
    /// Create an accountant over the canonical default Renyi order grid.
    pub fn new() -> Self {
        Self {
            inner: RenyiAccountant::with_default_orders(),
            ledger: PrivacyLedger::new(),
        }
    }

    /// Create an accountant over a caller-supplied order grid.
    pub fn with_orders(orders: Vec<f64>) -> Result<Self> {
        Ok(Self {
            inner: RenyiAccountant::new(orders)?,
            ledger: PrivacyLedger::new(),
        })
    }

    /// Access the underlying RDP accountant (per-order spend, orders).
    pub fn inner(&self) -> &RenyiAccountant {
        &self.inner
    }
}

impl Default for RenyiPrivacyAccountant {
    fn default() -> Self {
        Self::new()
    }
}

impl PrivacyAccountant for RenyiPrivacyAccountant {
    fn compose_subsampled_gaussian(
        &mut self,
        noise_multiplier: f64,
        sampling_probability: f64,
        steps: usize,
    ) -> Result<()> {
        self.ledger
            .record(noise_multiplier, sampling_probability, steps)?;
        self.inner
            .add_subsampled_gaussian(noise_multiplier, sampling_probability, steps)
    }

    fn privacy_spent(&self, target_delta: f64) -> Result<(f64, f64)> {
        let conversion = self.inner.to_epsilon_delta(target_delta)?;
        Ok((conversion.epsilon, conversion.delta))
    }

    fn projected_privacy_spent(
        &self,
        noise_multiplier: f64,
        sampling_probability: f64,
        steps: usize,
        target_delta: f64,
    ) -> Result<(f64, f64)> {
        let mut probe = self.inner.clone();
        probe.add_subsampled_gaussian(noise_multiplier, sampling_probability, steps)?;
        let conversion = probe.to_epsilon_delta(target_delta)?;
        Ok((conversion.epsilon, conversion.delta))
    }

    fn total_steps(&self) -> usize {
        self.ledger.total_steps()
    }

    fn segments(&self) -> &[AccountingSegment] {
        self.ledger.segments()
    }

    fn method(&self) -> AccountingMethod {
        AccountingMethod::RenyiDP
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.ledger.clear();
    }
}

/// Moments accountant (Abadi et al. 2016) driven by the exact sampled
/// Gaussian RDP kernel.
#[derive(Debug, Clone)]
pub struct MomentsPrivacyAccountant {
    inner: MomentsAccountant,
    ledger: PrivacyLedger,
}

impl MomentsPrivacyAccountant {
    /// Create a moments accountant. The constructor parameters only seed the
    /// default mechanism description; every composed step is recorded in the
    /// ledger with its own parameters.
    pub fn new(
        noise_multiplier: f64,
        target_delta: f64,
        batch_size: usize,
        dataset_size: usize,
    ) -> Self {
        Self {
            inner: MomentsAccountant::new(noise_multiplier, target_delta, batch_size, dataset_size),
            ledger: PrivacyLedger::new(),
        }
    }

    /// Access the underlying calculator.
    pub fn inner(&self) -> &MomentsAccountant {
        &self.inner
    }
}

impl PrivacyAccountant for MomentsPrivacyAccountant {
    fn compose_subsampled_gaussian(
        &mut self,
        noise_multiplier: f64,
        sampling_probability: f64,
        steps: usize,
    ) -> Result<()> {
        self.ledger
            .record(noise_multiplier, sampling_probability, steps)
    }

    fn privacy_spent(&self, target_delta: f64) -> Result<(f64, f64)> {
        let epsilon = self
            .inner
            .compose_segments(self.ledger.segments(), target_delta)?;
        Ok((epsilon, target_delta))
    }

    fn projected_privacy_spent(
        &self,
        noise_multiplier: f64,
        sampling_probability: f64,
        steps: usize,
        target_delta: f64,
    ) -> Result<(f64, f64)> {
        let mut probe = self.ledger.clone();
        probe.record(noise_multiplier, sampling_probability, steps)?;
        let epsilon = self
            .inner
            .compose_segments(probe.segments(), target_delta)?;
        Ok((epsilon, target_delta))
    }

    fn total_steps(&self) -> usize {
        self.ledger.total_steps()
    }

    fn segments(&self) -> &[AccountingSegment] {
        self.ledger.segments()
    }

    fn method(&self) -> AccountingMethod {
        AccountingMethod::MomentsAccountant
    }

    fn reset(&mut self) {
        self.ledger.clear();
    }
}

/// Construct the accountant selected by `method`.
///
/// Only the two methods backed by a correct implementation can be built.
/// [`AccountingMethod::AdvancedComposition`] and [`AccountingMethod::ZCDP`]
/// return an error: silently substituting a different analysis would report a
/// privacy loss the caller did not ask for and cannot audit.
pub fn build_accountant(
    method: AccountingMethod,
    noise_multiplier: f64,
    target_delta: f64,
    batch_size: usize,
    dataset_size: usize,
) -> Result<Box<dyn PrivacyAccountant>> {
    match method {
        AccountingMethod::RenyiDP => Ok(Box::new(RenyiPrivacyAccountant::new())),
        AccountingMethod::MomentsAccountant => Ok(Box::new(MomentsPrivacyAccountant::new(
            noise_multiplier,
            target_delta,
            batch_size,
            dataset_size,
        ))),
        AccountingMethod::AdvancedComposition => Err(OptimError::InvalidConfig(
            "AccountingMethod::AdvancedComposition is not implemented for the subsampled \
             Gaussian mechanism; use AccountingMethod::RenyiDP or \
             AccountingMethod::MomentsAccountant"
                .to_string(),
        )),
        AccountingMethod::ZCDP => Err(OptimError::InvalidConfig(
            "AccountingMethod::ZCDP is not implemented for the subsampled Gaussian mechanism; \
             use AccountingMethod::RenyiDP or AccountingMethod::MomentsAccountant"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ledger_merges_identical_consecutive_segments() {
        let mut ledger = PrivacyLedger::new();
        ledger.record(1.1, 0.01, 10).expect("valid");
        ledger.record(1.1, 0.01, 5).expect("valid");
        assert_eq!(ledger.segments().len(), 1);
        assert_eq!(ledger.segments()[0].steps, 15);
        assert_eq!(ledger.total_steps(), 15);
    }

    #[test]
    fn test_ledger_opens_new_segment_on_parameter_change() {
        let mut ledger = PrivacyLedger::new();
        ledger.record(1.1, 0.01, 10).expect("valid");
        ledger.record(1.1, 0.02, 10).expect("valid");
        ledger.record(2.0, 0.02, 10).expect("valid");
        assert_eq!(ledger.segments().len(), 3);
        assert_eq!(ledger.total_steps(), 30);
    }

    #[test]
    fn test_ledger_rejects_invalid_parameters() {
        let mut ledger = PrivacyLedger::new();
        assert!(ledger.record(0.0, 0.01, 1).is_err());
        assert!(ledger.record(f64::NAN, 0.01, 1).is_err());
        assert!(ledger.record(1.0, 1.5, 1).is_err());
        assert!(ledger.record(1.0, -0.1, 1).is_err());
        assert!(ledger.segments().is_empty());
    }

    #[test]
    fn test_renyi_accountant_epsilon_grows_with_steps() {
        let mut accountant = RenyiPrivacyAccountant::new();
        accountant
            .compose_subsampled_gaussian(1.0, 0.01, 100)
            .expect("composition");
        let (eps_100, delta) = accountant.privacy_spent(1.0e-5).expect("conversion");
        assert_eq!(delta, 1.0e-5);

        accountant
            .compose_subsampled_gaussian(1.0, 0.01, 100)
            .expect("composition");
        let (eps_200, _) = accountant.privacy_spent(1.0e-5).expect("conversion");

        assert!(eps_200 > eps_100, "{eps_200} must exceed {eps_100}");
        assert_eq!(accountant.total_steps(), 200);
        assert_eq!(accountant.segments().len(), 1);
    }

    #[test]
    fn test_delta_is_a_reporting_parameter_not_a_spend() {
        // Reporting the same composed mechanism at a smaller delta must
        // *increase* epsilon, and must never exhaust a "delta budget".
        let mut accountant = RenyiPrivacyAccountant::new();
        accountant
            .compose_subsampled_gaussian(1.0, 0.01, 500)
            .expect("composition");

        let (eps_loose, _) = accountant.privacy_spent(1.0e-4).expect("conversion");
        let (eps_tight, _) = accountant.privacy_spent(1.0e-7).expect("conversion");
        assert!(
            eps_tight > eps_loose,
            "smaller delta must imply larger epsilon: {eps_tight} vs {eps_loose}"
        );
    }

    #[test]
    fn test_moments_and_renyi_agree_within_an_order_of_magnitude() {
        let mut renyi = RenyiPrivacyAccountant::new();
        let mut moments = MomentsPrivacyAccountant::new(1.0, 1.0e-5, 100, 10_000);
        renyi
            .compose_subsampled_gaussian(1.0, 0.01, 1000)
            .expect("composition");
        moments
            .compose_subsampled_gaussian(1.0, 0.01, 1000)
            .expect("composition");

        let (eps_renyi, _) = renyi.privacy_spent(1.0e-5).expect("conversion");
        let (eps_moments, _) = moments.privacy_spent(1.0e-5).expect("conversion");

        assert!(eps_renyi.is_finite() && eps_renyi > 0.0);
        assert!(eps_moments.is_finite() && eps_moments > 0.0);
        let ratio = eps_moments / eps_renyi;
        assert!(
            (0.5..=2.0).contains(&ratio),
            "the two accountants compose the same kernel and must agree closely: \
             renyi={eps_renyi}, moments={eps_moments}"
        );
    }

    #[test]
    fn test_golden_epsilon_for_both_accounting_methods() {
        // Canonical DP-SGD setting: sigma = 1.0, q = 0.01, delta = 1e-5.
        //
        // Both accountants compose the same externally-anchored per-step RDP
        // kernel (see
        // `renyi_accountant::tests::test_kernel_reproduces_the_published_tensorflow_privacy_reference`,
        // which reproduces the published TF Privacy tutorial value 1.18). They
        // differ only in the order grid and the RDP-to-DP conversion:
        //
        // * `RenyiPrivacyAccountant`  - DEFAULT_ALPHAS, Canonne-Kamath-Steinke
        // * `MomentsPrivacyAccountant` - integer orders 2..=64, classic Mironov
        //
        // The moments value is therefore expected to be the larger of the two.
        let cases: [(usize, f64, f64); 2] = [
            (100, 1.224_845_779_636, 1.617_281_887_460),
            (1000, 2.107_753_075_452, 2.538_347_545_459),
        ];

        for (steps, golden_renyi, golden_moments) in cases {
            let mut renyi = RenyiPrivacyAccountant::new();
            let mut moments = MomentsPrivacyAccountant::new(1.0, 1.0e-5, 100, 10_000);
            renyi
                .compose_subsampled_gaussian(1.0, 0.01, steps)
                .expect("composition");
            moments
                .compose_subsampled_gaussian(1.0, 0.01, steps)
                .expect("composition");

            let (eps_renyi, _) = renyi.privacy_spent(1.0e-5).expect("conversion");
            let (eps_moments, _) = moments.privacy_spent(1.0e-5).expect("conversion");

            assert!(
                (eps_renyi - golden_renyi).abs() < 1.0e-9,
                "T={steps}: Renyi epsilon {eps_renyi} != golden {golden_renyi}"
            );
            assert!(
                (eps_moments - golden_moments).abs() < 1.0e-9,
                "T={steps}: moments epsilon {eps_moments} != golden {golden_moments}"
            );
            assert!(
                eps_moments > eps_renyi,
                "T={steps}: the classic conversion cannot be tighter than CKS"
            );
        }
    }

    #[test]
    fn test_projected_spend_matches_the_spend_after_actually_composing() {
        // `projected_privacy_spent` is what enforces the budget *before* a
        // noisy value is released, so it must agree exactly with the spend the
        // step would actually incur -- and must leave the ledger untouched.
        for method in [
            AccountingMethod::RenyiDP,
            AccountingMethod::MomentsAccountant,
        ] {
            let mut accountant =
                build_accountant(method, 1.0, 1.0e-5, 100, 10_000).expect("implemented method");
            accountant
                .compose_subsampled_gaussian(1.0, 0.01, 99)
                .expect("composition");

            let (projected, _) = accountant
                .projected_privacy_spent(1.0, 0.01, 1, 1.0e-5)
                .expect("projection");
            assert_eq!(
                accountant.total_steps(),
                99,
                "{method:?}: a projection must not record anything"
            );

            accountant
                .compose_subsampled_gaussian(1.0, 0.01, 1)
                .expect("composition");
            let (actual, _) = accountant.privacy_spent(1.0e-5).expect("conversion");

            assert!(
                (projected - actual).abs() < 1.0e-12,
                "{method:?}: projected {projected} must equal realised {actual}"
            );
        }
    }

    #[test]
    fn test_heterogeneous_segments_compose() {
        let mut accountant = RenyiPrivacyAccountant::new();
        accountant
            .compose_subsampled_gaussian(1.0, 0.01, 100)
            .expect("segment 1");
        let (eps_after_first, _) = accountant.privacy_spent(1.0e-5).expect("conversion");
        accountant
            .compose_subsampled_gaussian(2.0, 0.005, 100)
            .expect("segment 2");
        let (eps_after_second, _) = accountant.privacy_spent(1.0e-5).expect("conversion");

        assert!(eps_after_second > eps_after_first);
        assert_eq!(accountant.segments().len(), 2);
        assert_eq!(accountant.total_steps(), 200);
    }

    #[test]
    fn test_build_accountant_selects_method() {
        let renyi = build_accountant(AccountingMethod::RenyiDP, 1.0, 1e-5, 100, 10_000)
            .expect("renyi is implemented");
        assert!(matches!(renyi.method(), AccountingMethod::RenyiDP));

        let moments = build_accountant(AccountingMethod::MomentsAccountant, 1.0, 1e-5, 100, 10_000)
            .expect("moments is implemented");
        assert!(matches!(
            moments.method(),
            AccountingMethod::MomentsAccountant
        ));
    }

    #[test]
    fn test_build_accountant_rejects_unimplemented_methods() {
        for method in [
            AccountingMethod::AdvancedComposition,
            AccountingMethod::ZCDP,
        ] {
            match build_accountant(method, 1.0, 1e-5, 100, 10_000) {
                Err(OptimError::InvalidConfig(_)) => {}
                other => panic!("expected InvalidConfig for {method:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_reset_clears_spend_and_ledger() {
        let mut accountant = RenyiPrivacyAccountant::new();
        accountant
            .compose_subsampled_gaussian(1.0, 0.01, 100)
            .expect("composition");
        accountant.reset();
        assert_eq!(accountant.total_steps(), 0);
        assert!(accountant.segments().is_empty());
        let (eps, _) = accountant.privacy_spent(1.0e-5).expect("conversion");
        assert_eq!(eps, 0.0);
    }
}
