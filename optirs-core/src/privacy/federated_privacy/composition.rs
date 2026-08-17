//! Privacy amplification and composition analysis for federated rounds.
//!
//! # The defects this replaces
//!
//! * `PrivacyAmplificationAnalyzer::compute_amplification_factor` was marked
//!   "Placeholder implementation" and returned `(1/q).sqrt()`, a formula with no
//!   basis in any amplification theorem, and recorded a *fabricated*
//!   `clients_sampled: (q * 1000.0) as usize, total_clients: 1000` in its
//!   history whatever the real federation size was.
//! * `FederatedCompositionAnalyzer` was a constructor only. It stored the
//!   configured `FederatedCompositionMethod` and never applied it, so
//!   `round_compositions` and `client_compositions` were dead and no composed
//!   epsilon was ever produced.
//!
//! # What is implemented
//!
//! The amplification factor is now `epsilon / epsilon'` for the **published**
//! subsampling bound `epsilon' = ln(1 + q (e^epsilon - 1))`
//! (Kasiviswanathan et al. 2011; Balle, Barthe & Gaboardi 2018), obtained by
//! delegating to the crate's audited
//! [`crate::privacy::differential_privacy::PrivacyAmplificationAnalyzer::amplify_by_subsampling`].
//! It is always `>= 1`, equals exactly `1` at `q = 1`, and is a *derived ratio*,
//! so dividing epsilon by it reproduces the amplified epsilon exactly rather
//! than treating amplification as unbounded free privacy.
//!
//! Composition supports three methods, all with real formulas:
//!
//! * `Basic`: epsilons and deltas add (Dwork & Roth Thm. 3.14).
//! * `AdvancedComposition`: `eps' = sqrt(2 k ln(1/delta')) eps + k eps (e^eps - 1)`
//!   (Dwork, Rothblum & Vadhan 2010 / Dwork & Roth Thm. 3.20).
//! * `FederatedMomentsAccountant`: delegates to the crate's
//!   [`MomentsAccountant`], which composes the subsampled Gaussian mechanism.
//!
//! `RenyiDP` and `ZCDP` return [`OptimError::UnsupportedOperation`], matching
//! how the rest of the crate treats unimplemented accountants.

use crate::error::{OptimError, Result};
use crate::privacy::differential_privacy::PrivacyAmplificationAnalyzer as SubsamplingAnalyzer;
use crate::privacy::moment_accountant::MomentsAccountant;

use super::components::{
    ClientComposition, FederatedCompositionAnalyzer, PrivacyAmplificationAnalyzer,
    RoundComposition, SubsamplingEvent,
};
use super::config::{AmplificationConfig, FederatedCompositionMethod};

/// Maximum number of subsampling events retained.
const MAX_SUBSAMPLING_HISTORY: usize = 1_000;

impl PrivacyAmplificationAnalyzer {
    /// The configured amplification settings.
    pub fn config(&self) -> &AmplificationConfig {
        &self.config
    }

    /// Recorded subsampling events, oldest first.
    pub fn subsampling_history(&self) -> impl Iterator<Item = &SubsamplingEvent> {
        self.subsampling_history.iter()
    }

    /// The amplification factor recorded for a round, if any.
    pub fn recorded_factor(&self, round: usize) -> Option<f64> {
        self.amplification_factors.get(&round.to_string()).copied()
    }

    /// The factor by which `base_epsilon` may be divided thanks to subsampling.
    ///
    /// `factor = base_epsilon / ln(1 + q (e^base_epsilon - 1))`, with
    /// `q = clients_sampled / total_clients`. The result is always in
    /// `[1, base_epsilon / ln(1 + q(e^eps - 1))]` and exactly `1.0` when
    /// `q = 1` or amplification is disabled, so a caller that writes
    /// `epsilon / factor` gets the published amplified epsilon and nothing more.
    ///
    /// # 0.3.2 signature change
    ///
    /// The previous signature was `(sampling_rate, round)` and the body returned
    /// `(1/sampling_rate).sqrt()`, which is unbounded as `q -> 0` and is not a
    /// bound from any theorem. The real client counts are now required because
    /// the recorded history used to fabricate them.
    pub fn compute_amplification_factor(
        &mut self,
        base_epsilon: f64,
        clients_sampled: usize,
        total_clients: usize,
        round: usize,
    ) -> Result<f64> {
        if !base_epsilon.is_finite() || base_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the base epsilon must be positive and finite, got {base_epsilon}"
            )));
        }
        if total_clients == 0 {
            return Err(OptimError::InvalidParameter(
                "total_clients must be positive".to_string(),
            ));
        }
        if clients_sampled == 0 || clients_sampled > total_clients {
            return Err(OptimError::InvalidParameter(format!(
                "clients_sampled must lie in 1..={total_clients}, got {clients_sampled}"
            )));
        }

        let sampling_rate = clients_sampled as f64 / total_clients as f64;
        let factor = if !self.config.enabled || sampling_rate >= 1.0 {
            1.0
        } else {
            // Delegate the bound itself to the audited implementation rather
            // than restating it here.
            let mut analyzer = SubsamplingAnalyzer::new((&self.config).into());
            let amplified = analyzer.amplify_by_subsampling(
                base_epsilon,
                clients_sampled,
                total_clients,
                round,
            )?;
            if !amplified.is_finite() || amplified <= 0.0 {
                return Err(OptimError::PrivacyAccountingError(format!(
                    "the subsampling bound produced an amplified epsilon of {amplified}"
                )));
            }
            let factor = base_epsilon / amplified;
            if !factor.is_finite() || factor < 1.0 {
                return Err(OptimError::PrivacyAccountingError(format!(
                    "the amplification factor came out as {factor}, which would *increase* the \
                     released epsilon"
                )));
            }
            factor
        };

        self.subsampling_history.push_back(SubsamplingEvent {
            round,
            sampling_rate,
            clients_sampled,
            total_clients,
            amplification_factor: factor,
        });
        while self.subsampling_history.len() > MAX_SUBSAMPLING_HISTORY {
            let _ = self.subsampling_history.pop_front();
        }
        self.amplification_factors.insert(round.to_string(), factor);
        Ok(factor)
    }
}

/// Composed privacy cost over a set of rounds.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposedPrivacyCost {
    /// Composed epsilon.
    pub epsilon: f64,
    /// Delta the composed epsilon is reported at.
    pub delta: f64,
    /// Number of rounds composed.
    pub rounds: usize,
    /// Method used to compose.
    pub method: FederatedCompositionMethod,
}

impl FederatedCompositionAnalyzer {
    /// The configured composition method.
    pub fn method(&self) -> FederatedCompositionMethod {
        self.method
    }

    /// Recorded round compositions, oldest first.
    pub fn round_compositions(&self) -> &[RoundComposition] {
        &self.round_compositions
    }

    /// Recorded per-client compositions.
    pub fn client_compositions(&self, client_id: &str) -> &[ClientComposition] {
        self.client_compositions
            .get(client_id)
            .map(|entries| entries.as_slice())
            .unwrap_or(&[])
    }

    /// Record what a round consumed.
    pub fn record_round(&mut self, composition: RoundComposition) -> Result<()> {
        if !composition.epsilon_consumed.is_finite() || composition.epsilon_consumed < 0.0 {
            return Err(OptimError::PrivacyAccountingError(format!(
                "round {} reported epsilon {}, which is not a spend",
                composition.round, composition.epsilon_consumed
            )));
        }
        if !composition.delta_consumed.is_finite()
            || !(0.0..1.0).contains(&composition.delta_consumed)
        {
            return Err(OptimError::PrivacyAccountingError(format!(
                "round {} reported delta {}, which must lie in [0, 1)",
                composition.round, composition.delta_consumed
            )));
        }
        self.round_compositions.push(composition);
        Ok(())
    }

    /// Record what a round cost one client.
    pub fn record_client(&mut self, composition: ClientComposition) -> Result<()> {
        if !composition.local_epsilon.is_finite() || composition.local_epsilon < 0.0 {
            return Err(OptimError::PrivacyAccountingError(format!(
                "client {} reported epsilon {}, which is not a spend",
                composition.client_id, composition.local_epsilon
            )));
        }
        self.client_compositions
            .entry(composition.client_id.clone())
            .or_default()
            .push(composition);
        Ok(())
    }

    /// Compose the recorded rounds under the configured method.
    ///
    /// `target_delta` is the delta the composed epsilon is reported at. It is
    /// only consulted by the methods whose bound depends on it.
    pub fn compose(&self, target_delta: f64) -> Result<ComposedPrivacyCost> {
        if self.round_compositions.is_empty() {
            return Err(OptimError::InvalidState(
                "no rounds have been recorded, so there is nothing to compose".to_string(),
            ));
        }
        if !target_delta.is_finite() || !(0.0..1.0).contains(&target_delta) || target_delta <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the reporting delta must lie in (0, 1), got {target_delta}"
            )));
        }

        let rounds = self.round_compositions.len();
        let epsilon = match self.method {
            FederatedCompositionMethod::Basic => self.compose_basic(),
            FederatedCompositionMethod::AdvancedComposition => {
                self.compose_advanced(target_delta)?
            }
            FederatedCompositionMethod::FederatedMomentsAccountant => {
                self.compose_with_moments_accountant(target_delta)?
            }
            FederatedCompositionMethod::RenyiDP => {
                return Err(OptimError::UnsupportedOperation(
                    "FederatedCompositionMethod::RenyiDP is not implemented at the federated \
                     level; use FederatedMomentsAccountant, or compose per client with \
                     privacy::renyi_accountant::RenyiAccountant"
                        .to_string(),
                ))
            }
            FederatedCompositionMethod::ZCDP => {
                return Err(OptimError::UnsupportedOperation(
                    "FederatedCompositionMethod::ZCDP is not implemented, matching \
                     AccountingMethod::ZCDP elsewhere in this crate"
                        .to_string(),
                ))
            }
        };

        Ok(ComposedPrivacyCost {
            epsilon,
            delta: target_delta,
            rounds,
            method: self.method,
        })
    }

    /// Basic composition: epsilons add (Dwork & Roth Thm. 3.14).
    fn compose_basic(&self) -> f64 {
        self.round_compositions
            .iter()
            .map(|round| round.epsilon_consumed)
            .sum()
    }

    /// Advanced composition (Dwork, Rothblum & Vadhan 2010).
    ///
    /// `eps' = sqrt(2 k ln(1/delta')) eps + k eps (e^eps - 1)` for `k`
    /// applications of an `eps`-DP mechanism. The rounds must share a common
    /// epsilon for the bound to apply, so the largest recorded epsilon is used --
    /// which is the conservative direction.
    fn compose_advanced(&self, target_delta: f64) -> Result<f64> {
        let rounds = self.round_compositions.len() as f64;
        let per_round = self
            .round_compositions
            .iter()
            .map(|round| round.epsilon_consumed)
            .fold(0.0f64, f64::max);
        if per_round <= 0.0 {
            return Ok(0.0);
        }
        let advanced = (2.0 * rounds * (1.0 / target_delta).ln()).sqrt() * per_round
            + rounds * per_round * (per_round.exp() - 1.0);
        if !advanced.is_finite() {
            return Err(OptimError::PrivacyAccountingError(format!(
                "advanced composition overflowed for {rounds} rounds at epsilon {per_round}"
            )));
        }
        // Advanced composition is only an improvement in the regime it was
        // derived for; never report a looser bound than basic composition.
        Ok(advanced.min(self.compose_basic()))
    }

    /// Compose through the crate's moments accountant.
    ///
    /// The subsampled-Gaussian composition is what the accountant implements, so
    /// the per-round noise multiplier and sampling rate are recovered from the
    /// recorded rounds. A round that recorded no client count cannot be composed
    /// this way and is reported as an error rather than silently dropped.
    fn compose_with_moments_accountant(&self, target_delta: f64) -> Result<f64> {
        let rounds = self.round_compositions.len();
        let noise_multiplier = self.round_compositions.iter().try_fold(
            f64::INFINITY,
            |smallest, round| -> Result<f64> {
                let multiplier = round.noise_multiplier.ok_or_else(|| {
                    OptimError::PrivacyAccountingError(format!(
                        "round {} recorded no noise multiplier, so the moments accountant cannot \
                         compose it; record one or use FederatedCompositionMethod::Basic",
                        round.round
                    ))
                })?;
                if !multiplier.is_finite() || multiplier <= 0.0 {
                    return Err(OptimError::PrivacyAccountingError(format!(
                        "round {} recorded a noise multiplier of {multiplier}",
                        round.round
                    )));
                }
                Ok(smallest.min(multiplier))
            },
        )?;

        let (sampled, total) = self.round_compositions.iter().try_fold(
            (0usize, 0usize),
            |(sampled, total), round| -> Result<(usize, usize)> {
                if round.participating_clients == 0 || round.total_clients == 0 {
                    return Err(OptimError::PrivacyAccountingError(format!(
                        "round {} recorded {} of {} clients, which is not a sampling rate",
                        round.round, round.participating_clients, round.total_clients
                    )));
                }
                Ok((
                    sampled.max(round.participating_clients),
                    total.max(round.total_clients),
                ))
            },
        )?;

        let accountant = MomentsAccountant::new(noise_multiplier, target_delta, sampled, total);
        let (epsilon, _delta) = accountant.get_privacy_spent(rounds)?;
        Ok(epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn amplification_analyzer(enabled: bool) -> PrivacyAmplificationAnalyzer {
        PrivacyAmplificationAnalyzer::new(AmplificationConfig {
            enabled,
            ..AmplificationConfig::default()
        })
    }

    fn round(index: usize, epsilon: f64) -> RoundComposition {
        RoundComposition {
            round: index,
            participating_clients: 100,
            total_clients: 1_000,
            epsilon_consumed: epsilon,
            delta_consumed: 1e-6,
            amplification_applied: true,
            composition_method: FederatedCompositionMethod::Basic,
            noise_multiplier: Some(1.1),
        }
    }

    #[test]
    fn the_amplification_factor_matches_the_published_bound() {
        // eps' = ln(1 + q (e^eps - 1)); factor = eps / eps'.
        let mut analyzer = amplification_analyzer(true);
        let base_epsilon = 1.0f64;
        let factor = match analyzer.compute_amplification_factor(base_epsilon, 100, 1_000, 1) {
            Ok(factor) => factor,
            Err(err) => panic!("amplification failed: {err}"),
        };
        let expected_amplified = (1.0 + 0.1 * (base_epsilon.exp() - 1.0)).ln();
        let expected_factor = base_epsilon / expected_amplified;
        assert!(
            (factor - expected_factor).abs() < 1e-12,
            "factor {factor} vs expected {expected_factor}"
        );
        // The old placeholder returned (1/q).sqrt() = 3.1623 here.
        assert!(
            (factor - (1.0f64 / 0.1).sqrt()).abs() > 0.5,
            "the factor still matches the discarded (1/q).sqrt() placeholder"
        );
        // Dividing by the factor reproduces the amplified epsilon exactly.
        assert!((base_epsilon / factor - expected_amplified).abs() < 1e-12);
    }

    #[test]
    fn the_factor_is_one_without_subsampling_or_when_disabled() {
        let mut analyzer = amplification_analyzer(true);
        match analyzer.compute_amplification_factor(1.0, 1_000, 1_000, 1) {
            Ok(factor) => assert!((factor - 1.0).abs() < 1e-12, "factor = {factor}"),
            Err(err) => panic!("amplification failed: {err}"),
        }

        let mut disabled = amplification_analyzer(false);
        match disabled.compute_amplification_factor(1.0, 1, 1_000, 1) {
            Ok(factor) => assert!((factor - 1.0).abs() < 1e-12, "factor = {factor}"),
            Err(err) => panic!("amplification failed: {err}"),
        }
    }

    #[test]
    fn the_factor_is_never_below_one_and_grows_as_sampling_shrinks() {
        let mut analyzer = amplification_analyzer(true);
        let mut previous = 1.0;
        for (round, sampled) in [(1usize, 500usize), (2, 100), (3, 10), (4, 1)] {
            let factor = match analyzer.compute_amplification_factor(1.0, sampled, 1_000, round) {
                Ok(factor) => factor,
                Err(err) => panic!("amplification failed: {err}"),
            };
            assert!(factor >= 1.0, "factor {factor} would increase epsilon");
            assert!(
                factor > previous,
                "factor at q={} ({factor}) must exceed the previous {previous}",
                sampled as f64 / 1000.0
            );
            previous = factor;
        }
    }

    #[test]
    fn the_recorded_history_carries_the_real_client_counts() {
        // Regression: the placeholder recorded `clients_sampled: (q * 1000) as
        // usize, total_clients: 1000` whatever the federation size was.
        let mut analyzer = amplification_analyzer(true);
        let ok = analyzer.compute_amplification_factor(1.0, 7, 42, 3);
        assert!(ok.is_ok());
        let event = match analyzer.subsampling_history().next() {
            Some(event) => event,
            None => panic!("the event must be recorded"),
        };
        assert_eq!(event.clients_sampled, 7);
        assert_eq!(event.total_clients, 42);
        assert_eq!(event.round, 3);
        assert!((event.sampling_rate - 7.0 / 42.0).abs() < 1e-12);
        assert!(event.amplification_factor >= 1.0);
        assert_eq!(
            analyzer.recorded_factor(3),
            Some(event.amplification_factor)
        );
    }

    #[test]
    fn degenerate_amplification_inputs_are_refused() {
        let mut analyzer = amplification_analyzer(true);
        assert!(analyzer
            .compute_amplification_factor(0.0, 10, 100, 1)
            .is_err());
        assert!(analyzer
            .compute_amplification_factor(-1.0, 10, 100, 1)
            .is_err());
        assert!(analyzer
            .compute_amplification_factor(f64::NAN, 10, 100, 1)
            .is_err());
        assert!(analyzer
            .compute_amplification_factor(1.0, 0, 100, 1)
            .is_err());
        assert!(analyzer
            .compute_amplification_factor(1.0, 10, 0, 1)
            .is_err());
        assert!(analyzer
            .compute_amplification_factor(1.0, 200, 100, 1)
            .is_err());
    }

    #[test]
    fn basic_composition_adds_the_recorded_epsilons() {
        let mut analyzer = FederatedCompositionAnalyzer::new(FederatedCompositionMethod::Basic);
        assert!(
            analyzer.compose(1e-5).is_err(),
            "composing nothing must be an error"
        );
        for index in 0..5usize {
            let ok = analyzer.record_round(round(index, 0.2));
            assert!(ok.is_ok());
        }
        assert_eq!(analyzer.round_compositions().len(), 5);
        let composed = match analyzer.compose(1e-5) {
            Ok(composed) => composed,
            Err(err) => panic!("composition failed: {err}"),
        };
        assert!(
            (composed.epsilon - 1.0).abs() < 1e-12,
            "5 x 0.2 = 1.0, got {}",
            composed.epsilon
        );
        assert_eq!(composed.rounds, 5);
        assert_eq!(composed.delta, 1e-5);
    }

    #[test]
    fn advanced_composition_beats_basic_for_many_small_rounds() {
        let mut basic = FederatedCompositionAnalyzer::new(FederatedCompositionMethod::Basic);
        let mut advanced =
            FederatedCompositionAnalyzer::new(FederatedCompositionMethod::AdvancedComposition);
        for index in 0..200usize {
            let ok = basic.record_round(round(index, 0.01));
            assert!(ok.is_ok());
            let ok = advanced.record_round(round(index, 0.01));
            assert!(ok.is_ok());
        }
        let basic_epsilon = match basic.compose(1e-5) {
            Ok(composed) => composed.epsilon,
            Err(err) => panic!("composition failed: {err}"),
        };
        let advanced_epsilon = match advanced.compose(1e-5) {
            Ok(composed) => composed.epsilon,
            Err(err) => panic!("composition failed: {err}"),
        };
        assert!(
            (basic_epsilon - 2.0).abs() < 1e-9,
            "basic = {basic_epsilon}"
        );
        assert!(
            advanced_epsilon < basic_epsilon,
            "advanced ({advanced_epsilon}) must beat basic ({basic_epsilon}) here"
        );
        // sqrt(2 * 200 * ln(1e5)) * 0.01 + 200 * 0.01 * (e^0.01 - 1)
        let expected = (2.0f64 * 200.0 * (1.0f64 / 1e-5).ln()).sqrt() * 0.01
            + 200.0 * 0.01 * (0.01f64.exp() - 1.0);
        assert!(
            (advanced_epsilon - expected).abs() < 1e-9,
            "advanced = {advanced_epsilon}, closed form = {expected}"
        );
    }

    #[test]
    fn advanced_composition_never_reports_more_than_basic() {
        let mut analyzer =
            FederatedCompositionAnalyzer::new(FederatedCompositionMethod::AdvancedComposition);
        // A single large-epsilon round is outside the regime where the advanced
        // bound helps; it must not be reported as worse than simply adding up.
        let ok = analyzer.record_round(round(0, 2.0));
        assert!(ok.is_ok());
        let composed = match analyzer.compose(1e-5) {
            Ok(composed) => composed,
            Err(err) => panic!("composition failed: {err}"),
        };
        assert!(
            composed.epsilon <= 2.0 + 1e-12,
            "composed {} exceeds basic composition",
            composed.epsilon
        );
    }

    #[test]
    fn the_moments_accountant_path_produces_a_real_epsilon() {
        let mut analyzer = FederatedCompositionAnalyzer::new(
            FederatedCompositionMethod::FederatedMomentsAccountant,
        );
        for index in 0..50usize {
            let ok = analyzer.record_round(round(index, 0.05));
            assert!(ok.is_ok());
        }
        let composed = match analyzer.compose(1e-5) {
            Ok(composed) => composed,
            Err(err) => panic!("composition failed: {err}"),
        };
        assert!(
            composed.epsilon > 0.0 && composed.epsilon.is_finite(),
            "epsilon = {}",
            composed.epsilon
        );
        // Independently: the same accountant queried directly must agree.
        let accountant = MomentsAccountant::new(1.1, 1e-5, 100, 1_000);
        let (direct, _) = match accountant.get_privacy_spent(50) {
            Ok(spent) => spent,
            Err(err) => panic!("accountant failed: {err}"),
        };
        assert!(
            (composed.epsilon - direct).abs() < 1e-12,
            "composed {} vs direct {direct}",
            composed.epsilon
        );
    }

    #[test]
    fn the_moments_accountant_path_refuses_rounds_it_cannot_compose() {
        let mut analyzer = FederatedCompositionAnalyzer::new(
            FederatedCompositionMethod::FederatedMomentsAccountant,
        );
        let mut without_multiplier = round(0, 0.05);
        without_multiplier.noise_multiplier = None;
        let ok = analyzer.record_round(without_multiplier);
        assert!(ok.is_ok());
        let message = match analyzer.compose(1e-5) {
            Err(err) => err.to_string(),
            Ok(composed) => panic!("composed {} from nothing", composed.epsilon),
        };
        assert!(message.contains("no noise multiplier"), "got: {message}");

        let mut analyzer = FederatedCompositionAnalyzer::new(
            FederatedCompositionMethod::FederatedMomentsAccountant,
        );
        let mut without_clients = round(0, 0.05);
        without_clients.participating_clients = 0;
        let ok = analyzer.record_round(without_clients);
        assert!(ok.is_ok());
        assert!(analyzer.compose(1e-5).is_err());
    }

    #[test]
    fn unimplemented_composition_methods_error_instead_of_returning_a_number() {
        for method in [
            FederatedCompositionMethod::RenyiDP,
            FederatedCompositionMethod::ZCDP,
        ] {
            let mut analyzer = FederatedCompositionAnalyzer::new(method);
            let ok = analyzer.record_round(round(0, 0.1));
            assert!(ok.is_ok());
            assert!(
                analyzer.compose(1e-5).is_err(),
                "{method:?} must not produce an epsilon"
            );
        }
    }

    #[test]
    fn an_invalid_round_record_is_refused() {
        let mut analyzer = FederatedCompositionAnalyzer::new(FederatedCompositionMethod::Basic);
        let mut bad = round(0, -1.0);
        assert!(analyzer.record_round(bad.clone()).is_err());
        bad.epsilon_consumed = f64::NAN;
        assert!(analyzer.record_round(bad.clone()).is_err());
        bad.epsilon_consumed = 0.1;
        bad.delta_consumed = 1.0;
        assert!(analyzer.record_round(bad).is_err());
        assert!(analyzer.compose(0.0).is_err());
        assert!(analyzer.compose(1.0).is_err());
    }

    #[test]
    fn per_client_compositions_are_recorded_and_retrievable() {
        let mut analyzer = FederatedCompositionAnalyzer::new(FederatedCompositionMethod::Basic);
        assert!(analyzer.client_compositions("alice").is_empty());
        for round_index in 0..3usize {
            let ok = analyzer.record_client(ClientComposition {
                client_id: "alice".to_string(),
                round: round_index,
                local_epsilon: 0.1,
                local_delta: 1e-6,
                contribution_weight: 1.0,
            });
            assert!(ok.is_ok());
        }
        assert_eq!(analyzer.client_compositions("alice").len(), 3);
        assert!(analyzer.client_compositions("bob").is_empty());

        assert!(analyzer
            .record_client(ClientComposition {
                client_id: "bob".to_string(),
                round: 0,
                local_epsilon: -1.0,
                local_delta: 1e-6,
                contribution_weight: 1.0,
            })
            .is_err());
    }
}
