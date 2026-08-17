//! Validation of the federated privacy configuration.
//!
//! # The gap this closes
//!
//! `config.rs` was 1854 lines of struct and `Default` definitions with no
//! `validate` method, no checked constructor and no `InvalidParameter`
//! construction anywhere in the file, and
//! `FederatedPrivacyCoordinator::new` forwarded straight into
//! `MomentsAccountant::new`. So `target_epsilon: -1.0`, `target_delta: 2.0`,
//! `noise_multiplier: 0.0` (no noise at all) and
//! `clients_per_round > total_clients` were all silently accepted and produced
//! a meaningless epsilon.
//!
//! # Two kinds of rule
//!
//! 1. **Parameter ranges.** epsilon > 0, 0 < delta < 1, noise_multiplier > 0,
//!    l2_norm_clip > 0, `1 <= clients_per_round <= total_clients`,
//!    `max_dropouts < min_clients`, and so on.
//! 2. **Unimplemented switches.** A flag that no code path reads is not a
//!    feature; a configuration that sets one is asking for a guarantee this
//!    build does not deliver, so [`FederatedPrivacyConfig::validate`] refuses it
//!    with [`OptimError::UnsupportedOperation`] naming what is missing. Where
//!    such a flag *defaulted* to `true`, the default was corrected to `false` in
//!    0.3.2 rather than making the default configuration invalid -- see the
//!    module note in `config.rs`.

use crate::error::{OptimError, Result};

use super::config::{
    AmplificationConfig, ClientSamplingStrategy, CommunicationPrivacyConfig, CrossDeviceConfig,
    FederatedCompositionMethod, FederatedPrivacyConfig, SecureAggregationConfig, TrustModel,
};

/// Validate one `(epsilon, delta)` pair.
fn check_epsilon_delta(context: &str, epsilon: f64, delta: f64) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidPrivacyConfig(format!(
            "{context}: target_epsilon must be positive and finite, got {epsilon}"
        )));
    }
    if !delta.is_finite() || !(0.0..1.0).contains(&delta) || delta <= 0.0 {
        return Err(OptimError::InvalidPrivacyConfig(format!(
            "{context}: target_delta must lie in (0, 1), got {delta}"
        )));
    }
    Ok(())
}

impl FederatedPrivacyConfig {
    /// Validate the whole configuration.
    ///
    /// Called by `FederatedPrivacyCoordinator::new`, so an invalid federation
    /// cannot be constructed at all.
    pub fn validate(&self) -> Result<()> {
        self.validate_base()?;
        self.validate_federation()?;
        self.secure_aggregation
            .validate(self.total_clients, self.base_config.target_epsilon)?;
        self.amplification_config.validate()?;
        self.cross_device_config.validate()?;
        self.communication_privacy.validate()?;
        validate_sampling_strategy(self.sampling_strategy)?;
        validate_composition_method(self.composition_method)?;
        self.validate_trust_model()?;
        Ok(())
    }

    /// Range checks on the underlying differential-privacy configuration.
    fn validate_base(&self) -> Result<()> {
        let base = &self.base_config;
        check_epsilon_delta("base_config", base.target_epsilon, base.target_delta)?;
        if !base.noise_multiplier.is_finite() || base.noise_multiplier <= 0.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "base_config.noise_multiplier must be positive and finite, got {}; a zero \
                 multiplier adds no noise and provides no privacy",
                base.noise_multiplier
            )));
        }
        if !base.l2_norm_clip.is_finite() || base.l2_norm_clip <= 0.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "base_config.l2_norm_clip must be positive and finite, got {}; without a clipping \
                 bound the noise cannot be calibrated to any sensitivity",
                base.l2_norm_clip
            )));
        }
        if base.batch_size == 0 {
            return Err(OptimError::InvalidPrivacyConfig(
                "base_config.batch_size must be positive".to_string(),
            ));
        }
        if base.dataset_size == 0 {
            return Err(OptimError::InvalidPrivacyConfig(
                "base_config.dataset_size must be positive".to_string(),
            ));
        }
        if base.batch_size > base.dataset_size {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "base_config.batch_size ({}) exceeds dataset_size ({}), so the sampling \
                 probability would exceed 1",
                base.batch_size, base.dataset_size
            )));
        }
        if base.max_steps == 0 {
            return Err(OptimError::InvalidPrivacyConfig(
                "base_config.max_steps must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Range checks on the federation shape.
    fn validate_federation(&self) -> Result<()> {
        if self.total_clients == 0 {
            return Err(OptimError::InvalidConfig(
                "total_clients must be positive".to_string(),
            ));
        }
        if self.clients_per_round == 0 {
            return Err(OptimError::InvalidConfig(
                "clients_per_round must be positive".to_string(),
            ));
        }
        if self.clients_per_round > self.total_clients {
            return Err(OptimError::InvalidConfig(format!(
                "clients_per_round ({}) exceeds total_clients ({}), so the per-round sampling \
                 probability would exceed 1 and the amplification analysis would be invalid",
                self.clients_per_round, self.total_clients
            )));
        }
        Ok(())
    }

    /// The trust model must be backed by a mechanism that enforces it.
    fn validate_trust_model(&self) -> Result<()> {
        match self.trust_model {
            TrustModel::HonestButCurious | TrustModel::SemiHonest => Ok(()),
            TrustModel::Malicious | TrustModel::Byzantine => {
                if self.secure_aggregation.enabled {
                    Ok(())
                } else {
                    Err(OptimError::InvalidConfig(format!(
                        "TrustModel::{:?} was declared but secure_aggregation is disabled; the \
                         server would see every client update in the clear, so the declared trust \
                         model is not enforced by anything",
                        self.trust_model
                    )))
                }
            }
        }
    }

    /// The effective per-round sampling probability.
    pub fn sampling_probability(&self) -> f64 {
        if self.total_clients == 0 {
            0.0
        } else {
            self.clients_per_round as f64 / self.total_clients as f64
        }
    }
}

impl SecureAggregationConfig {
    /// Validate the secure-aggregation settings.
    pub fn validate(&self, total_clients: usize, target_epsilon: f64) -> Result<()> {
        if self.min_clients < 2 {
            return Err(OptimError::InvalidConfig(format!(
                "secure_aggregation.min_clients must be at least 2, got {}; the \"aggregate\" of a \
                 single client is that client's own update",
                self.min_clients
            )));
        }
        if self.min_clients > total_clients {
            return Err(OptimError::InvalidConfig(format!(
                "secure_aggregation.min_clients ({}) exceeds total_clients ({total_clients}), so \
                 no round could ever aggregate",
                self.min_clients
            )));
        }
        if self.max_dropouts >= self.min_clients {
            return Err(OptimError::InvalidConfig(format!(
                "secure_aggregation.max_dropouts ({}) must be below min_clients ({}), otherwise \
                 every client could drop out and the threshold would still be reported as met",
                self.max_dropouts, self.min_clients
            )));
        }
        if self.masking_dimension == 0 {
            return Err(OptimError::InvalidConfig(
                "secure_aggregation.masking_dimension must be positive".to_string(),
            ));
        }
        if let Some(bits) = self.quantization_bits {
            if !(1..=32).contains(&bits) {
                return Err(OptimError::InvalidConfig(format!(
                    "secure_aggregation.quantization_bits must lie in 1..=32, got {bits}"
                )));
            }
        }
        if self.aggregate_dp {
            return Err(OptimError::UnsupportedOperation(format!(
                "secure_aggregation.aggregate_dp was requested, but no code path adds noise to the \
                 aggregate: the coordinator's aggregation is a plain mean. Apply differential \
                 privacy on the client side (target_epsilon = {target_epsilon}) or use \
                 privacy::secure_aggregation::SecureAggregator, which implements the masked \
                 protocol"
            )));
        }
        Ok(())
    }
}

impl AmplificationConfig {
    /// Validate the amplification settings.
    pub fn validate(&self) -> Result<()> {
        if !self.subsampling_factor.is_finite() || !(0.0..=1.0).contains(&self.subsampling_factor) {
            return Err(OptimError::InvalidConfig(format!(
                "amplification_config.subsampling_factor must lie in (0, 1], got {}",
                self.subsampling_factor
            )));
        }
        if self.subsampling_factor <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "amplification_config.subsampling_factor must be positive".to_string(),
            ));
        }
        let mut unimplemented = Vec::new();
        if self.shuffling_enabled {
            unimplemented.push(
                "shuffling_enabled (the shuffle bound exists as \
                 privacy::differential_privacy::PrivacyAmplificationAnalyzer::amplify_by_shuffling \
                 but the federated round pipeline does not call it)",
            );
        }
        if self.multi_round_amplification {
            unimplemented.push(
                "multi_round_amplification (amplification is recomputed per round and never \
                 composed across rounds)",
            );
        }
        if self.heterogeneous_amplification {
            unimplemented
                .push("heterogeneous_amplification (per-client sampling rates are not tracked)");
        }
        if unimplemented.is_empty() {
            Ok(())
        } else {
            Err(OptimError::UnsupportedOperation(format!(
                "amplification_config requests behaviour no code path implements: {}",
                unimplemented.join("; ")
            )))
        }
    }
}

impl CrossDeviceConfig {
    /// Validate the cross-device settings.
    ///
    /// Every field of this struct is currently declared and never read, so a
    /// configuration that sets one is asking for a guarantee this build does not
    /// deliver. `user_level_privacy` in particular is a *stronger* guarantee than
    /// per-example DP and cannot be granted by accident.
    pub fn validate(&self) -> Result<()> {
        let mut unimplemented = Vec::new();
        if self.user_level_privacy {
            unimplemented.push(
                "user_level_privacy (needs a device-to-user mapping and per-user composition; see \
                 privacy::federated::cross_device_manager)",
            );
        }
        if self.device_clustering {
            unimplemented.push("device_clustering");
        }
        if self.temporal_privacy {
            unimplemented.push("temporal_privacy");
        }
        if self.geographic_privacy {
            unimplemented.push("geographic_privacy");
        }
        if self.demographic_privacy {
            unimplemented.push("demographic_privacy");
        }
        if unimplemented.is_empty() {
            Ok(())
        } else {
            Err(OptimError::UnsupportedOperation(format!(
                "cross_device_config requests behaviour the federated_privacy coordinator does not \
                 implement: {}",
                unimplemented.join("; ")
            )))
        }
    }
}

impl CommunicationPrivacyConfig {
    /// Validate the communication-privacy settings.
    pub fn validate(&self) -> Result<()> {
        let mut unimplemented = Vec::new();
        if self.encryption_enabled {
            unimplemented.push(
                "encryption_enabled (this crate performs no transport and carries no \
                 authenticated-encryption primitive)",
            );
        }
        if self.anonymous_channels {
            unimplemented.push("anonymous_channels (needs a mix network or onion routing)");
        }
        if self.communication_noise {
            unimplemented.push("communication_noise (no padding or cover traffic is generated)");
        }
        if self.traffic_analysis_protection {
            unimplemented.push("traffic_analysis_protection");
        }
        if self.cross_silo_config.is_some() {
            unimplemented.push(
                "cross_silo_config (the cross-silo trust, governance and marketplace engines do \
                 not exist)",
            );
        }
        if unimplemented.is_empty() {
            Ok(())
        } else {
            Err(OptimError::UnsupportedOperation(format!(
                "communication_privacy requests behaviour no code path implements: {}",
                unimplemented.join("; ")
            )))
        }
    }
}

/// Reject client-sampling strategies that silently fall back to uniform.
///
/// `FederatedPrivacyCoordinator::sample_clients` matches `UniformRandom` and has
/// a `_ =>` arm that performs uniform sampling for every other strategy. Poisson
/// sampling in particular is what the subsampling-amplification bound assumes,
/// so silently substituting uniform sampling invalidates the reported epsilon.
pub fn validate_sampling_strategy(strategy: ClientSamplingStrategy) -> Result<()> {
    match strategy {
        ClientSamplingStrategy::UniformRandom => Ok(()),
        ClientSamplingStrategy::PoissonSampling => Err(OptimError::UnsupportedOperation(
            "ClientSamplingStrategy::PoissonSampling is not implemented; the round sampler falls \
             back to uniform sampling without replacement, which does not satisfy the independent \
             per-client inclusion the amplification bound assumes"
                .to_string(),
        )),
        other => Err(OptimError::UnsupportedOperation(format!(
            "ClientSamplingStrategy::{other:?} is not implemented; the round sampler would fall \
             back to UniformRandom, so the requested strategy would not be applied"
        ))),
    }
}

/// Reject federated composition methods with no implementation.
pub fn validate_composition_method(method: FederatedCompositionMethod) -> Result<()> {
    match method {
        FederatedCompositionMethod::Basic
        | FederatedCompositionMethod::AdvancedComposition
        | FederatedCompositionMethod::FederatedMomentsAccountant => Ok(()),
        FederatedCompositionMethod::RenyiDP => Err(OptimError::UnsupportedOperation(
            "FederatedCompositionMethod::RenyiDP is not implemented at the federated level; use \
             FederatedMomentsAccountant, or compose per client with \
             privacy::renyi_accountant::RenyiAccountant"
                .to_string(),
        )),
        FederatedCompositionMethod::ZCDP => Err(OptimError::UnsupportedOperation(
            "FederatedCompositionMethod::ZCDP is not implemented, matching \
             AccountingMethod::ZCDP elsewhere in this crate"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::DifferentialPrivacyConfig;

    #[test]
    fn the_default_configuration_validates() {
        let config = FederatedPrivacyConfig::default();
        let outcome = config.validate();
        assert!(
            outcome.is_ok(),
            "the default configuration must be valid: {outcome:?}"
        );
    }

    #[test]
    fn enabling_secure_aggregation_still_validates() {
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.enabled = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn a_non_positive_epsilon_is_refused() {
        for epsilon in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
            let config = FederatedPrivacyConfig {
                base_config: DifferentialPrivacyConfig {
                    target_epsilon: epsilon,
                    ..DifferentialPrivacyConfig::default()
                },
                ..FederatedPrivacyConfig::default()
            };
            assert!(
                config.validate().is_err(),
                "target_epsilon = {epsilon} must be refused"
            );
        }
    }

    #[test]
    fn a_delta_outside_the_open_unit_interval_is_refused() {
        for delta in [0.0f64, 1.0, 2.0, -1e-5, f64::NAN] {
            let config = FederatedPrivacyConfig {
                base_config: DifferentialPrivacyConfig {
                    target_delta: delta,
                    ..DifferentialPrivacyConfig::default()
                },
                ..FederatedPrivacyConfig::default()
            };
            assert!(
                config.validate().is_err(),
                "target_delta = {delta} must be refused"
            );
        }
    }

    #[test]
    fn a_zero_noise_multiplier_is_refused() {
        for multiplier in [0.0f64, -1.0, f64::NAN] {
            let config = FederatedPrivacyConfig {
                base_config: DifferentialPrivacyConfig {
                    noise_multiplier: multiplier,
                    ..DifferentialPrivacyConfig::default()
                },
                ..FederatedPrivacyConfig::default()
            };
            let outcome = config.validate();
            assert!(
                outcome.is_err(),
                "noise_multiplier = {multiplier} adds no noise and must be refused"
            );
        }
    }

    #[test]
    fn a_non_positive_clipping_bound_is_refused() {
        for clip in [0.0f64, -1.0, f64::INFINITY] {
            let config = FederatedPrivacyConfig {
                base_config: DifferentialPrivacyConfig {
                    l2_norm_clip: clip,
                    ..DifferentialPrivacyConfig::default()
                },
                ..FederatedPrivacyConfig::default()
            };
            assert!(
                config.validate().is_err(),
                "l2_norm_clip = {clip} must be refused"
            );
        }
    }

    #[test]
    fn more_clients_per_round_than_exist_is_refused() {
        let config = FederatedPrivacyConfig {
            clients_per_round: 2000,
            total_clients: 1000,
            ..FederatedPrivacyConfig::default()
        };
        let message = match config.validate() {
            Err(err) => err.to_string(),
            Ok(()) => panic!("clients_per_round > total_clients must be refused"),
        };
        assert!(message.contains("exceeds total_clients"), "got: {message}");

        let config = FederatedPrivacyConfig {
            clients_per_round: 0,
            ..FederatedPrivacyConfig::default()
        };
        assert!(config.validate().is_err());
        let config = FederatedPrivacyConfig {
            total_clients: 0,
            clients_per_round: 0,
            ..FederatedPrivacyConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn a_dropout_tolerance_at_or_above_the_threshold_is_refused() {
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.max_dropouts = config.secure_aggregation.min_clients;
        let message = match config.validate() {
            Err(err) => err.to_string(),
            Ok(()) => panic!("max_dropouts >= min_clients must be refused"),
        };
        assert!(message.contains("max_dropouts"), "got: {message}");
    }

    #[test]
    fn a_degenerate_aggregation_threshold_is_refused() {
        for min_clients in [0usize, 1] {
            let mut config = FederatedPrivacyConfig::default();
            config.secure_aggregation.min_clients = min_clients;
            config.secure_aggregation.max_dropouts = 0;
            assert!(
                config.validate().is_err(),
                "min_clients = {min_clients} must be refused"
            );
        }
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.min_clients = 5000;
        assert!(config.validate().is_err(), "min_clients > total_clients");
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.masking_dimension = 0;
        assert!(config.validate().is_err());
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.quantization_bits = Some(0);
        assert!(config.validate().is_err());
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.quantization_bits = Some(64);
        assert!(config.validate().is_err());
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.quantization_bits = Some(16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn requesting_dp_on_the_aggregate_is_refused_because_nothing_adds_it() {
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.aggregate_dp = true;
        let message = match config.validate() {
            Err(err) => err.to_string(),
            Ok(()) => panic!("an unimplemented guarantee must not be granted"),
        };
        assert!(message.contains("plain mean"), "got: {message}");
    }

    #[test]
    fn every_unimplemented_cross_device_flag_is_refused() {
        type CrossDeviceFlag = (&'static str, fn(&mut CrossDeviceConfig));
        let flags: [CrossDeviceFlag; 5] = [
            ("user_level_privacy", |c| c.user_level_privacy = true),
            ("device_clustering", |c| c.device_clustering = true),
            ("temporal_privacy", |c| c.temporal_privacy = true),
            ("geographic_privacy", |c| c.geographic_privacy = true),
            ("demographic_privacy", |c| c.demographic_privacy = true),
        ];
        for (name, set) in flags {
            let mut config = FederatedPrivacyConfig::default();
            set(&mut config.cross_device_config);
            let message = match config.validate() {
                Err(err) => err.to_string(),
                Ok(()) => panic!("{name} must not be silently accepted"),
            };
            assert!(message.contains(name), "got: {message}");
        }
    }

    #[test]
    fn every_unimplemented_communication_flag_is_refused() {
        type CommunicationFlag = (&'static str, fn(&mut CommunicationPrivacyConfig));
        let flags: [CommunicationFlag; 4] = [
            ("encryption_enabled", |c| c.encryption_enabled = true),
            ("anonymous_channels", |c| c.anonymous_channels = true),
            ("communication_noise", |c| c.communication_noise = true),
            ("traffic_analysis_protection", |c| {
                c.traffic_analysis_protection = true
            }),
        ];
        for (name, set) in flags {
            let mut config = FederatedPrivacyConfig::default();
            set(&mut config.communication_privacy);
            let message = match config.validate() {
                Err(err) => err.to_string(),
                Ok(()) => panic!("{name} must not be silently accepted"),
            };
            assert!(message.contains(name), "got: {message}");
        }
    }

    #[test]
    fn every_unimplemented_amplification_flag_is_refused() {
        type AmplificationFlag = (&'static str, fn(&mut AmplificationConfig));
        let flags: [AmplificationFlag; 3] = [
            ("shuffling_enabled", |c| c.shuffling_enabled = true),
            ("multi_round_amplification", |c| {
                c.multi_round_amplification = true
            }),
            ("heterogeneous_amplification", |c| {
                c.heterogeneous_amplification = true
            }),
        ];
        for (name, set) in flags {
            let mut config = FederatedPrivacyConfig::default();
            set(&mut config.amplification_config);
            let message = match config.validate() {
                Err(err) => err.to_string(),
                Ok(()) => panic!("{name} must not be silently accepted"),
            };
            assert!(message.contains(name), "got: {message}");
        }
    }

    #[test]
    fn a_subsampling_factor_outside_the_unit_interval_is_refused() {
        for factor in [0.0f64, -0.5, 1.5, f64::NAN] {
            let mut config = FederatedPrivacyConfig::default();
            config.amplification_config.subsampling_factor = factor;
            assert!(
                config.validate().is_err(),
                "subsampling_factor = {factor} must be refused"
            );
        }
    }

    #[test]
    fn unimplemented_sampling_strategies_are_refused() {
        assert!(validate_sampling_strategy(ClientSamplingStrategy::UniformRandom).is_ok());
        for strategy in [
            ClientSamplingStrategy::Stratified,
            ClientSamplingStrategy::ImportanceSampling,
            ClientSamplingStrategy::PoissonSampling,
            ClientSamplingStrategy::FairSampling,
        ] {
            let config = FederatedPrivacyConfig {
                sampling_strategy: strategy,
                ..FederatedPrivacyConfig::default()
            };
            assert!(
                config.validate().is_err(),
                "{strategy:?} silently falls back to uniform and must be refused"
            );
        }
    }

    #[test]
    fn unimplemented_composition_methods_are_refused() {
        for method in [
            FederatedCompositionMethod::Basic,
            FederatedCompositionMethod::AdvancedComposition,
            FederatedCompositionMethod::FederatedMomentsAccountant,
        ] {
            assert!(
                validate_composition_method(method).is_ok(),
                "{method:?} is implemented"
            );
        }
        for method in [
            FederatedCompositionMethod::RenyiDP,
            FederatedCompositionMethod::ZCDP,
        ] {
            let config = FederatedPrivacyConfig {
                composition_method: method,
                ..FederatedPrivacyConfig::default()
            };
            assert!(
                config.validate().is_err(),
                "{method:?} must be refused rather than silently substituted"
            );
        }
    }

    #[test]
    fn a_malicious_trust_model_requires_secure_aggregation() {
        let mut config = FederatedPrivacyConfig {
            trust_model: TrustModel::Malicious,
            ..FederatedPrivacyConfig::default()
        };
        let message = match config.validate() {
            Err(err) => err.to_string(),
            Ok(()) => panic!("a declared trust model must be enforced by something"),
        };
        assert!(
            message.contains("secure_aggregation is disabled"),
            "got: {message}"
        );

        config.secure_aggregation.enabled = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn a_batch_larger_than_the_dataset_is_refused() {
        let config = FederatedPrivacyConfig {
            base_config: DifferentialPrivacyConfig {
                batch_size: 100_000,
                dataset_size: 1_000,
                ..DifferentialPrivacyConfig::default()
            },
            ..FederatedPrivacyConfig::default()
        };
        assert!(config.validate().is_err());

        for (batch, dataset) in [(0usize, 1000usize), (100, 0)] {
            let config = FederatedPrivacyConfig {
                base_config: DifferentialPrivacyConfig {
                    batch_size: batch,
                    dataset_size: dataset,
                    ..DifferentialPrivacyConfig::default()
                },
                ..FederatedPrivacyConfig::default()
            };
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn the_effective_sampling_probability_is_derived_from_the_client_counts() {
        let config = FederatedPrivacyConfig::default();
        assert!((config.sampling_probability() - 0.1).abs() < 1e-12);
        let config = FederatedPrivacyConfig {
            clients_per_round: 1,
            total_clients: 4,
            ..FederatedPrivacyConfig::default()
        };
        assert!((config.sampling_probability() - 0.25).abs() < 1e-12);
    }
}
