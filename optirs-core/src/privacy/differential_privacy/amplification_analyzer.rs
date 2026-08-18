// Privacy Amplification Analyzer
//
// Privacy amplification is a *theorem*, not a discount factor: a mechanism
// that is `epsilon_0`-DP on a subsample of the data satisfies a strictly
// smaller epsilon on the full dataset, and the size of that reduction is
// determined by published bounds. This module implements those bounds
// directly:
//
// * **Amplification by subsampling** (Kasiviswanathan, Lee, Nissim,
//   Raskhodnikova, Smith 2011; tight form in Balle, Barthe, Gaboardi 2018):
//   an `epsilon_0`-DP mechanism applied to a Poisson subsample with rate `q`
//   is `ln(1 + q (e^{epsilon_0} - 1))`-DP.
// * **Amplification by shuffling** (Feldman, McMillan, Talwar, FOCS 2021,
//   Theorem 3.1): shuffling the outputs of `n` `epsilon_0`-DP local
//   randomizers yields `(epsilon, delta)`-DP with
//   `epsilon = ln(1 + (e^{eps0}-1)/(e^{eps0}+1) * (8 sqrt(e^{eps0} ln(4/delta))/sqrt(n) + 8 e^{eps0}/n))`,
//   valid when `n >= 16 e^{eps0} ln(4/delta)`.
//
// Amplification factors from different theorems do **not** multiply, and
// there is no "multi-round amplification": repeating a mechanism costs more
// privacy, never less. Both were invented by an earlier revision and are
// gone.

use crate::error::{OptimError, Result};
use std::collections::{HashMap, VecDeque};

/// Privacy amplification configuration
#[derive(Debug, Clone)]
pub struct AmplificationConfig {
    /// Enable privacy amplification analysis. When disabled the analyzer
    /// reports the unamplified epsilon, which is always a valid (if
    /// pessimistic) bound.
    pub enabled: bool,

    /// Whether the deployment shuffles client reports, making the
    /// Feldman-McMillan-Talwar bound applicable.
    pub shuffling_enabled: bool,
}

/// Privacy amplification analyzer
#[derive(Debug, Clone)]
pub struct PrivacyAmplificationAnalyzer {
    config: AmplificationConfig,
    subsampling_history: VecDeque<SubsamplingEvent>,
    client_epsilons: HashMap<String, f64>,
}

/// Subsampling event for amplification analysis
#[derive(Debug, Clone)]
pub struct SubsamplingEvent {
    /// Federated round the event belongs to.
    pub round: usize,
    /// Realised sampling rate `clients_sampled / total_clients`.
    pub sampling_rate: f64,
    /// Number of clients actually sampled.
    pub clients_sampled: usize,
    /// Size of the population sampled from.
    pub total_clients: usize,
    /// Epsilon of the mechanism before amplification.
    pub base_epsilon: f64,
    /// Epsilon after applying the subsampling bound.
    pub amplified_epsilon: f64,
}

/// Amplification statistics
#[derive(Debug, Clone)]
pub struct AmplificationStats {
    /// Number of recorded rounds.
    pub rounds_analyzed: usize,
    /// Mean reduction ratio `base_epsilon / amplified_epsilon` (>= 1).
    pub avg_amplification_factor: f64,
    /// Largest reduction ratio observed.
    pub max_amplification_factor: f64,
    /// Smallest reduction ratio observed.
    pub min_amplification_factor: f64,
    /// Total epsilon saved: `sum(base - amplified)`, never negative.
    pub total_privacy_saved: f64,
}

impl PrivacyAmplificationAnalyzer {
    /// Create a new analyzer.
    pub fn new(config: AmplificationConfig) -> Self {
        Self {
            config,
            subsampling_history: VecDeque::with_capacity(1000),
            client_epsilons: HashMap::new(),
        }
    }

    /// Amplification by subsampling.
    ///
    /// Returns the epsilon of the *composed* mechanism `M o sample`, given
    /// that `M` alone is `base_epsilon`-DP and each record is included with
    /// probability `q = clients_sampled / total_clients`:
    ///
    /// ```text
    /// epsilon' = ln(1 + q * (e^{epsilon} - 1))
    /// ```
    ///
    /// The result is always in `(0, base_epsilon]`, and equals
    /// `base_epsilon` exactly when `q = 1` (no subsampling, no amplification).
    /// Note this is a bound on epsilon, **not** a factor to divide epsilon
    /// by: dividing treats amplification as unbounded free privacy.
    pub fn amplify_by_subsampling(
        &mut self,
        base_epsilon: f64,
        clients_sampled: usize,
        total_clients: usize,
        round: usize,
    ) -> Result<f64> {
        if !base_epsilon.is_finite() || base_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "base_epsilon must be a positive finite number, got {base_epsilon}"
            )));
        }
        if total_clients == 0 {
            return Err(OptimError::InvalidParameter(
                "total_clients must be positive".to_string(),
            ));
        }
        if clients_sampled == 0 || clients_sampled > total_clients {
            return Err(OptimError::InvalidParameter(format!(
                "clients_sampled must be in 1..={total_clients}, got {clients_sampled}"
            )));
        }

        let q = clients_sampled as f64 / total_clients as f64;

        let amplified = if !self.config.enabled || q >= 1.0 {
            base_epsilon
        } else {
            subsampled_epsilon(base_epsilon, q)?
        };

        self.subsampling_history.push_back(SubsamplingEvent {
            round,
            sampling_rate: q,
            clients_sampled,
            total_clients,
            base_epsilon,
            amplified_epsilon: amplified,
        });
        if self.subsampling_history.len() > 1000 {
            self.subsampling_history.pop_front();
        }

        Ok(amplified)
    }

    /// Amplification by shuffling (Feldman, McMillan, Talwar 2021, Thm 3.1).
    ///
    /// Requires `n >= 16 e^{epsilon_0} ln(4/delta)`; outside that regime the
    /// bound does not hold and an error is returned rather than a number the
    /// caller would mistake for a guarantee.
    pub fn amplify_by_shuffling(
        &self,
        base_epsilon: f64,
        num_clients: usize,
        delta: f64,
    ) -> Result<f64> {
        if !self.config.shuffling_enabled {
            return Err(OptimError::InvalidConfig(
                "shuffling amplification requested but shuffling_enabled is false; the bound \
                 only holds if client reports are actually shuffled"
                    .to_string(),
            ));
        }
        if !base_epsilon.is_finite() || base_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "base_epsilon must be a positive finite number, got {base_epsilon}"
            )));
        }
        if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "delta must be in (0, 1), got {delta}"
            )));
        }
        if num_clients < 2 {
            return Err(OptimError::InvalidParameter(
                "shuffling amplification requires at least 2 clients".to_string(),
            ));
        }

        let n = num_clients as f64;
        let exp_eps = base_epsilon.exp();
        let requirement = 16.0 * exp_eps * (4.0 / delta).ln();
        if n < requirement {
            return Err(OptimError::InvalidConfig(format!(
                "the Feldman-McMillan-Talwar shuffle bound requires n >= 16 e^eps0 ln(4/delta) \
                 = {requirement:.1}, but only {num_clients} clients participate"
            )));
        }

        let term = (exp_eps - 1.0) / (exp_eps + 1.0)
            * (8.0 * (exp_eps * (4.0 / delta).ln()).sqrt() / n.sqrt() + 8.0 * exp_eps / n);
        let amplified = (1.0 + term).ln();

        if !amplified.is_finite() || amplified <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "shuffle amplification produced a non-finite epsilon".to_string(),
            ));
        }

        // The shuffled bound can never be worse than the local one.
        Ok(amplified.min(base_epsilon))
    }

    /// Statistics over the recorded subsampling events.
    pub fn get_amplification_stats(&self) -> AmplificationStats {
        if self.subsampling_history.is_empty() {
            return AmplificationStats::default();
        }

        let mut factors = Vec::with_capacity(self.subsampling_history.len());
        let mut saved = 0.0;
        for event in &self.subsampling_history {
            if event.amplified_epsilon > 0.0 {
                factors.push(event.base_epsilon / event.amplified_epsilon);
            } else {
                factors.push(1.0);
            }
            // Amplification never increases epsilon, so this is non-negative
            // by construction.
            saved += (event.base_epsilon - event.amplified_epsilon).max(0.0);
        }

        let count = factors.len() as f64;
        let avg = factors.iter().sum::<f64>() / count;
        let max = factors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = factors.iter().cloned().fold(f64::INFINITY, f64::min);

        AmplificationStats {
            rounds_analyzed: self.subsampling_history.len(),
            avg_amplification_factor: avg,
            max_amplification_factor: max,
            min_amplification_factor: min,
            total_privacy_saved: saved,
        }
    }

    /// Record a client-specific amplified epsilon.
    pub fn set_client_epsilon(&mut self, client_id: String, epsilon: f64) -> Result<()> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "client epsilon must be a positive finite number, got {epsilon}"
            )));
        }
        self.client_epsilons.insert(client_id, epsilon);
        Ok(())
    }

    /// Client-specific amplified epsilon, if recorded.
    pub fn get_client_epsilon(&self, client_id: &str) -> Option<f64> {
        self.client_epsilons.get(client_id).copied()
    }

    /// Current configuration.
    pub fn config(&self) -> &AmplificationConfig {
        &self.config
    }

    /// Number of recorded rounds.
    pub fn rounds_analyzed(&self) -> usize {
        self.subsampling_history.len()
    }

    /// Clear all recorded history.
    pub fn clear_history(&mut self) {
        self.subsampling_history.clear();
        self.client_epsilons.clear();
    }

    /// Whether amplification analysis is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Recorded subsampling events.
    pub fn get_subsampling_history(&self) -> &VecDeque<SubsamplingEvent> {
        &self.subsampling_history
    }

    /// Replace the configuration.
    pub fn update_config(&mut self, config: AmplificationConfig) {
        self.config = config;
    }
}

/// `epsilon' = ln(1 + q (e^epsilon - 1))`, computed stably for small
/// `epsilon` where `e^epsilon - 1` loses precision.
fn subsampled_epsilon(epsilon: f64, q: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&q) {
        return Err(OptimError::InvalidParameter(format!(
            "sampling probability must be in [0, 1], got {q}"
        )));
    }

    let amplified = (q * epsilon.exp_m1()).ln_1p();
    if !amplified.is_finite() || amplified < 0.0 {
        return Err(OptimError::InvalidConfig(format!(
            "subsampling bound produced an invalid epsilon for epsilon={epsilon}, q={q}"
        )));
    }

    Ok(amplified.min(epsilon))
}

impl Default for AmplificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            shuffling_enabled: false,
        }
    }
}

impl Default for AmplificationStats {
    fn default() -> Self {
        Self {
            rounds_analyzed: 0,
            avg_amplification_factor: 1.0,
            max_amplification_factor: 1.0,
            min_amplification_factor: 1.0,
            total_privacy_saved: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplification_analyzer_creation() {
        let analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        assert!(analyzer.is_enabled());
        assert_eq!(analyzer.rounds_analyzed(), 0);
    }

    #[test]
    fn test_subsampling_bound_matches_the_closed_form() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        let base = 1.0_f64;
        let amplified = match analyzer.amplify_by_subsampling(base, 100, 1000, 1) {
            Ok(value) => value,
            Err(err) => panic!("amplification failed: {err}"),
        };

        // ln(1 + 0.1 * (e - 1)) = ln(1.171828...) = 0.158552...
        let expected = (1.0 + 0.1 * (base.exp() - 1.0)).ln();
        assert!((amplified - expected).abs() < 1e-12);
        assert!(amplified < base, "subsampling must reduce epsilon");
    }

    #[test]
    fn test_subsampling_is_monotone_in_the_sampling_rate() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        let sparse = match analyzer.amplify_by_subsampling(1.0, 10, 10_000, 1) {
            Ok(value) => value,
            Err(err) => panic!("amplification failed: {err}"),
        };
        let dense = match analyzer.amplify_by_subsampling(1.0, 5_000, 10_000, 2) {
            Ok(value) => value,
            Err(err) => panic!("amplification failed: {err}"),
        };
        assert!(sparse < dense, "a smaller sampling rate must amplify more");
    }

    #[test]
    fn test_full_sampling_gives_no_amplification() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        let amplified = match analyzer.amplify_by_subsampling(2.0, 500, 500, 1) {
            Ok(value) => value,
            Err(err) => panic!("amplification failed: {err}"),
        };
        assert!((amplified - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_amplification_never_exceeds_the_base_epsilon() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        for (sampled, total, base) in [(1usize, 2usize, 0.1f64), (3, 4, 5.0), (7, 100, 0.001)] {
            let amplified = match analyzer.amplify_by_subsampling(base, sampled, total, 1) {
                Ok(value) => value,
                Err(err) => panic!("amplification failed: {err}"),
            };
            assert!(amplified > 0.0);
            assert!(
                amplified <= base + 1e-15,
                "amplified epsilon {amplified} exceeded the base {base}"
            );
        }
    }

    #[test]
    fn test_amplification_with_disabled_config() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig {
            enabled: false,
            ..Default::default()
        });
        let amplified = match analyzer.amplify_by_subsampling(0.7, 10, 1000, 1) {
            Ok(value) => value,
            Err(err) => panic!("amplification failed: {err}"),
        };
        assert_eq!(
            amplified, 0.7,
            "with analysis disabled the unamplified epsilon must be reported"
        );
    }

    #[test]
    fn test_subsampling_validates_its_inputs() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        assert!(analyzer.amplify_by_subsampling(0.0, 10, 100, 1).is_err());
        assert!(analyzer.amplify_by_subsampling(-1.0, 10, 100, 1).is_err());
        assert!(analyzer.amplify_by_subsampling(1.0, 0, 100, 1).is_err());
        assert!(analyzer.amplify_by_subsampling(1.0, 200, 100, 1).is_err());
        assert!(analyzer.amplify_by_subsampling(1.0, 10, 0, 1).is_err());
        assert!(analyzer
            .amplify_by_subsampling(f64::NAN, 10, 100, 1)
            .is_err());
    }

    #[test]
    fn test_amplification_stats_are_never_negative() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        for round in 1..=3 {
            assert!(analyzer
                .amplify_by_subsampling(1.0, 10 * round, 1000, round)
                .is_ok());
        }

        let stats = analyzer.get_amplification_stats();
        assert_eq!(stats.rounds_analyzed, 3);
        assert!(stats.avg_amplification_factor >= 1.0);
        assert!(stats.max_amplification_factor >= stats.min_amplification_factor);
        assert!(
            stats.total_privacy_saved >= 0.0,
            "privacy saved must never be negative, got {}",
            stats.total_privacy_saved
        );
    }

    #[test]
    fn test_shuffling_bound_requires_the_valid_regime() {
        let analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig {
            enabled: true,
            shuffling_enabled: true,
        });

        // n = 100 with eps0 = 1 violates n >= 16 e ln(4/delta).
        assert!(analyzer.amplify_by_shuffling(1.0, 100, 1e-6).is_err());

        // A large cohort is inside the regime and must amplify.
        let amplified = match analyzer.amplify_by_shuffling(1.0, 1_000_000, 1e-6) {
            Ok(value) => value,
            Err(err) => panic!("shuffle amplification failed: {err}"),
        };
        assert!(amplified > 0.0 && amplified < 1.0, "got {amplified}");
    }

    #[test]
    fn test_shuffling_requires_shuffling_to_be_enabled() {
        let analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig {
            enabled: true,
            shuffling_enabled: false,
        });
        assert!(analyzer.amplify_by_shuffling(1.0, 1_000_000, 1e-6).is_err());
    }

    #[test]
    fn test_shuffling_validates_delta_and_cohort_size() {
        let analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig {
            enabled: true,
            shuffling_enabled: true,
        });
        assert!(analyzer.amplify_by_shuffling(1.0, 1_000_000, 0.0).is_err());
        assert!(analyzer.amplify_by_shuffling(1.0, 1_000_000, 1.0).is_err());
        assert!(analyzer.amplify_by_shuffling(1.0, 1, 1e-6).is_err());
        assert!(analyzer
            .amplify_by_shuffling(-1.0, 1_000_000, 1e-6)
            .is_err());
    }

    #[test]
    fn test_shuffling_amplifies_more_with_more_clients() {
        let analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig {
            enabled: true,
            shuffling_enabled: true,
        });
        let small = match analyzer.amplify_by_shuffling(1.0, 1_000_000, 1e-6) {
            Ok(value) => value,
            Err(err) => panic!("shuffle amplification failed: {err}"),
        };
        let large = match analyzer.amplify_by_shuffling(1.0, 100_000_000, 1e-6) {
            Ok(value) => value,
            Err(err) => panic!("shuffle amplification failed: {err}"),
        };
        assert!(
            large < small,
            "more clients must amplify more: {small} -> {large}"
        );
    }

    #[test]
    fn test_client_specific_epsilons() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        assert!(analyzer
            .set_client_epsilon("client1".to_string(), 0.5)
            .is_ok());
        assert_eq!(analyzer.get_client_epsilon("client1"), Some(0.5));
        assert_eq!(analyzer.get_client_epsilon("client2"), None);
        assert!(analyzer
            .set_client_epsilon("bad".to_string(), -1.0)
            .is_err());
    }

    #[test]
    fn test_clear_history() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        assert!(analyzer.amplify_by_subsampling(1.0, 10, 1000, 1).is_ok());
        assert!(analyzer
            .set_client_epsilon("client1".to_string(), 1.5)
            .is_ok());
        assert_eq!(analyzer.rounds_analyzed(), 1);

        analyzer.clear_history();
        assert_eq!(analyzer.rounds_analyzed(), 0);
        assert!(analyzer.get_client_epsilon("client1").is_none());
    }

    #[test]
    fn test_recorded_events_use_the_real_cohort_size() {
        let mut analyzer = PrivacyAmplificationAnalyzer::new(AmplificationConfig::default());
        assert!(analyzer.amplify_by_subsampling(1.0, 7, 350, 4).is_ok());
        let event = match analyzer.get_subsampling_history().front() {
            Some(event) => event.clone(),
            None => panic!("event should have been recorded"),
        };
        assert_eq!(event.clients_sampled, 7);
        assert_eq!(event.total_clients, 350);
        assert_eq!(event.round, 4);
        assert!((event.sampling_rate - 0.02).abs() < 1e-12);
    }
}
