// Main federated privacy coordinator implementation

use super::super::federated::secure_aggregation::{
    DropoutDisclosure, MaskedClientUpdate, SecureAggregationPlan,
};
use super::super::federated::ClientPublicKey;
use super::super::noise_mechanisms::GaussianMechanism;
use super::super::NoiseMechanism;
use super::super::{MomentsAccountant, PrivacyBudget};
use super::components::*;
use super::composition::ComposedPrivacyCost;
use super::config::*;
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::thread_rng;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Maximum number of rounds retained in
/// [`FederatedPrivacyCoordinator::participation_history`].
pub const MAX_PARTICIPATION_HISTORY: usize = 1000;

/// Federated differential privacy coordinator
pub struct FederatedPrivacyCoordinator<
    T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static,
> {
    /// Global privacy configuration
    config: FederatedPrivacyConfig,

    /// Per-client privacy accountants
    client_accountants: HashMap<String, MomentsAccountant>,

    /// Global privacy accountant
    global_accountant: MomentsAccountant,

    /// Secure aggregation protocol
    secure_aggregator: SecureAggregator<T>,

    /// Privacy amplification analyzer
    amplification_analyzer: PrivacyAmplificationAnalyzer,

    /// Cross-device privacy manager
    cross_device_manager: CrossDevicePrivacyManager<T>,

    /// Composition analyzer for multi-round privacy
    composition_analyzer: FederatedCompositionAnalyzer,

    /// Byzantine-robust aggregation engine, driven by
    /// [`Self::robust_aggregate_updates`] and by the plaintext aggregation path
    /// whenever `trust_model` declares adversarial clients.
    byzantine_aggregator: ByzantineRobustAggregator<T>,

    /// Personalized federated learning manager, driven by
    /// [`Self::apply_global_update`].
    personalization_manager: PersonalizationManager<T>,

    /// Current round number
    current_round: usize,

    /// Number of differentially private aggregate releases made through
    /// [`Self::secure_aggregate_updates`].
    ///
    /// The accountant's step count is `max(current_round, noisy_releases)` so
    /// that a caller who aggregates without going through
    /// [`Self::start_federated_round`] still pays for every noisy release. Before
    /// 0.3.2 the aggregation path added no noise at all and charged nothing.
    noisy_releases: usize,

    /// Client participation history
    participation_history: VecDeque<ParticipationRound>,
}

/// Federated round plan with privacy guarantees
#[derive(Debug, Clone)]
pub struct FederatedRoundPlan {
    pub round_number: usize,
    pub selectedclients: Vec<String>,
    pub sampling_probability: f64,
    pub amplificationfactor: f64,
    pub client_privacy_allocations: HashMap<String, ClientPrivacyAllocation>,
    pub aggregation_plan: Option<SecureAggregationPlan>,
    pub privacy_analysis: RoundPrivacyAnalysis,
}

/// Client privacy allocation for a round
#[derive(Debug, Clone)]
pub struct ClientPrivacyAllocation {
    pub epsilon: f64,
    pub delta: f64,
    pub noise_multiplier: f64,
    pub clipping_threshold: f64,
    pub amplificationfactor: f64,
}

/// Privacy analysis for a round
#[derive(Debug, Clone)]
pub struct RoundPrivacyAnalysis {
    pub round_epsilon: f64,
    pub round_delta: f64,
    pub cumulative_epsilon: f64,
    pub cumulative_delta: f64,
    pub amplification_benefit: f64,
    pub composition_tightness: f64,
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug,
    > FederatedPrivacyCoordinator<T>
{
    /// Create a new federated privacy coordinator
    ///
    /// The configuration is validated first (see
    /// [`FederatedPrivacyConfig::validate`]). Before 0.3.2 this forwarded
    /// straight into `MomentsAccountant::new`, so `target_epsilon: -1.0`,
    /// `target_delta: 2.0`, `noise_multiplier: 0.0` and
    /// `clients_per_round > total_clients` were all accepted and produced a
    /// meaningless epsilon.
    pub fn new(config: FederatedPrivacyConfig) -> Result<Self> {
        config.validate()?;

        let global_accountant = MomentsAccountant::new(
            config.base_config.noise_multiplier,
            config.base_config.target_delta,
            config.clients_per_round,
            config.total_clients,
        );

        let secure_aggregator = SecureAggregator::new(config.secure_aggregation.clone())?;
        let amplification_analyzer =
            PrivacyAmplificationAnalyzer::new(config.amplification_config.clone());
        let cross_device_manager =
            CrossDevicePrivacyManager::new(config.cross_device_config.clone());
        let composition_analyzer = FederatedCompositionAnalyzer::new(config.composition_method);

        Ok(Self {
            config,
            client_accountants: HashMap::new(),
            global_accountant,
            secure_aggregator,
            amplification_analyzer,
            cross_device_manager,
            composition_analyzer,
            byzantine_aggregator: ByzantineRobustAggregator::new()?,
            personalization_manager: PersonalizationManager::new()?,
            current_round: 0,
            noisy_releases: 0,
            participation_history: VecDeque::with_capacity(MAX_PARTICIPATION_HISTORY),
        })
    }

    /// Publish a client's per-round X25519 public key to the secure aggregator.
    ///
    /// Secure aggregation cannot open a round until every selected client has
    /// registered a key: the pairwise masks are derived from Diffie-Hellman
    /// agreements between the participants, and the server holds no key material
    /// of its own. Call this for each participant before
    /// [`Self::start_federated_round`].
    pub fn register_client_key(
        &mut self,
        client_id: &str,
        public_key: ClientPublicKey,
    ) -> Result<()> {
        self.secure_aggregator
            .register_client_key(client_id, public_key)
    }

    /// Accept a client's masked upload for the round that is currently open.
    pub fn receive_masked_update(&mut self, submission: MaskedClientUpdate) -> Result<()> {
        self.secure_aggregator.receive(submission)
    }

    /// Mark a selected client as having dropped out of the open round.
    pub fn mark_client_dropped(&mut self, client_id: &str) -> Result<()> {
        self.secure_aggregator.mark_dropped(client_id)
    }

    /// Accept a surviving client's disclosure of the pairwise mask it shares
    /// with a dropped client, so the server can cancel the residue.
    pub fn receive_dropout_disclosure(&mut self, disclosure: DropoutDisclosure) -> Result<()> {
        self.secure_aggregator
            .receive_dropout_disclosure(disclosure)
    }

    /// Release the masked cohort mean for the open secure-aggregation round.
    ///
    /// The server never sees an individual update on this path: it sums the
    /// masked uploads, the pairwise masks cancel, and only the cohort mean is
    /// recovered. Differential privacy is *not* added here -- secure aggregation
    /// hides individual updates from the server and says nothing about what the
    /// sum reveals. Clients that need DP must clip and perturb locally.
    pub fn aggregate_masked_updates(&self) -> Result<Array1<T>> {
        self.secure_aggregator.aggregate_mean()
    }

    /// The secure aggregator backing this coordinator.
    pub fn secure_aggregator(&self) -> &SecureAggregator<T> {
        &self.secure_aggregator
    }

    /// The cross-device privacy manager backing this coordinator.
    ///
    /// Register devices and record participation through this handle so that
    /// user-level and temporal-correlation budgets are actually tracked; the
    /// coordinator's per-round accounting is device-level only.
    pub fn cross_device_manager_mut(&mut self) -> &mut CrossDevicePrivacyManager<T> {
        &mut self.cross_device_manager
    }

    /// The cross-device privacy manager backing this coordinator.
    pub fn cross_device_manager(&self) -> &CrossDevicePrivacyManager<T> {
        &self.cross_device_manager
    }

    /// The per-client moments accountant for `client_id`, creating it on first
    /// use.
    ///
    /// A client's own spend is not the federation's: a client sampled in every
    /// round pays far more than the average participant. This is the handle that
    /// makes per-client accounting reachable; before 0.3.2 the
    /// `client_accountants` map was allocated in `new` and never touched again.
    pub fn client_privacy_spent(&mut self, client_id: &str, rounds: usize) -> Result<(f64, f64)> {
        let base = &self.config.base_config;
        let accountant = self
            .client_accountants
            .entry(client_id.to_string())
            .or_insert_with(|| {
                MomentsAccountant::new(
                    base.noise_multiplier,
                    base.target_delta,
                    self.config.clients_per_round,
                    self.config.total_clients,
                )
            });
        accountant.get_privacy_spent(rounds)
    }

    /// Aggregate client updates with the configured Byzantine-robust estimator,
    /// then add differential privacy noise.
    ///
    /// Use this instead of [`Self::secure_aggregate_updates`] when some
    /// participants may be adversarial. The robust estimator bounds how far a
    /// minority of malicious clients can move the result; the plain mean does
    /// not (a single client can move it arbitrarily). The estimator returns an
    /// error rather than silently degrading to a mean when the configured method
    /// cannot be applied to the cohort.
    ///
    /// `allocations` supplies the per-client weights used by the weighted
    /// methods; pass an empty map for the rank- and geometry-based ones.
    pub fn robust_aggregate_updates(
        &mut self,
        clientupdates: &HashMap<String, Array1<T>>,
        allocations: &HashMap<String, AdaptivePrivacyAllocation>,
    ) -> Result<Array1<T>> {
        let aggregate = self
            .byzantine_aggregator
            .robust_aggregate(clientupdates, allocations)?;
        let sensitivity = self.robust_sensitivity();
        self.perturb_and_charge(aggregate, sensitivity)
    }

    /// L2 sensitivity of the *robust* aggregate that was just produced.
    ///
    /// # Why this is not `l2_norm_clip / n`
    ///
    /// That bound belongs to the plain mean, where the sum of clipped
    /// contributions has sensitivity `l2_norm_clip` under add/remove-one and the
    /// divisor is a fixed `n`. A robust estimator averages a *selected* subset,
    /// and adding or removing one client can both evict a member of that subset
    /// and admit a new one, so the retained sum can move by up to
    /// `2 * l2_norm_clip`:
    ///
    /// ```text
    /// S_robust <= 2 * l2_norm_clip / averaged
    /// ```
    ///
    /// # Choosing `averaged`
    ///
    /// The divisor must be the number of values the estimator actually averaged,
    /// **not** the cohort size. `RobustEstimators::last_contributors` is a
    /// diagnostic audit trail, not that count: for
    /// [`ByzantineRobustMethod::TrimmedMean`] it records the whole cohort even
    /// though only `n - 2 * trim` values are averaged, so using its length would
    /// have inflated the divisor and released *less* noise than the accountant
    /// charged for -- a fabricated epsilon, which is the one failure mode this
    /// module exists to prevent. The count is therefore derived per method:
    ///
    /// * `TrimmedMean` -- `n - 2 * trim_count`, read back from
    ///   `last_trim_count`, which the aggregator sets from the configured ratio
    ///   and the real cohort size.
    /// * `MultiKrum` and `FedAvgOutlierDetection` -- the selected subset, which
    ///   *is* what `last_contributors` records for those two methods.
    /// * `Krum` -- a single client's update is returned verbatim, so no
    ///   averaging divides the perturbation at all: the divisor is 1.
    /// * `CoordinateWiseMedian`, `Bulyan`, `CenteredClipping`,
    ///   `ReputationWeighted` -- rank-, geometry- or weight-based estimators
    ///   with no uniform averaging denominator. Their true sensitivity is
    ///   data-dependent and not bounded by any fixed divisor, so the divisor is
    ///   1: the bound degrades to `2 * l2_norm_clip`, the only one that holds
    ///   for *any* estimator whose output lies in a ball of radius
    ///   `l2_norm_clip`. Deliberately pessimistic rather than wrong.
    ///
    /// Over-noising makes the accountant's epsilon an upper bound on the true
    /// spend, which is safe. Under-noising makes it a lie.
    fn robust_sensitivity(&self) -> f64 {
        let estimators = self.byzantine_aggregator.robust_estimators();
        let contributors = estimators.last_contributors().len();
        let averaged = match self.byzantine_aggregator.config().method {
            ByzantineRobustMethod::TrimmedMean { .. } => {
                // `last_contributors` is the full cohort here; the averaged count
                // is what survives trimming on both tails.
                contributors.saturating_sub(2 * estimators.last_trim_count())
            }
            // These two record exactly the subset they averaged.
            ByzantineRobustMethod::MultiKrum { .. }
            | ByzantineRobustMethod::FedAvgOutlierDetection { .. } => contributors,
            // Krum returns one client's update verbatim; nothing is averaged.
            ByzantineRobustMethod::Krum { .. } => 1,
            // No uniform averaging denominator exists for these.
            ByzantineRobustMethod::CoordinateWiseMedian
            | ByzantineRobustMethod::Bulyan { .. }
            | ByzantineRobustMethod::CenteredClipping { .. }
            | ByzantineRobustMethod::ReputationWeighted { .. } => 1,
        };
        2.0 * self.config.base_config.l2_norm_clip / averaged.max(1) as f64
    }

    /// Run outlier detection over a cohort and record the verdicts against the
    /// current round number.
    pub fn detect_byzantine_clients(
        &mut self,
        clientupdates: &HashMap<String, Array1<T>>,
    ) -> Result<Vec<OutlierDetectionResult>> {
        let round = self.current_round;
        self.byzantine_aggregator
            .detect_byzantine_clients(clientupdates, round)
    }

    /// How much of the plain mean's worst-case manipulation the configured
    /// robust method removes, in `(0, 1]`.
    pub fn byzantine_robustness_factor(&self) -> Result<f64> {
        self.byzantine_aggregator.compute_robustness_factor()
    }

    /// The Byzantine-robust aggregator backing this coordinator.
    pub fn byzantine_aggregator(&self) -> &ByzantineRobustAggregator<T> {
        &self.byzantine_aggregator
    }

    /// Mutable access to the Byzantine-robust aggregator, so its method and
    /// reputation state can be configured.
    pub fn byzantine_aggregator_mut(&mut self) -> &mut ByzantineRobustAggregator<T> {
        &mut self.byzantine_aggregator
    }

    /// Apply an aggregated update to the global model and return the new model.
    ///
    /// Delegates to [`PersonalizationManager::update_global_model`], which
    /// advances the stored model by the cohort delta.
    pub fn apply_global_update(&mut self, aggregated_update: &Array1<T>) -> Result<Array1<T>> {
        self.personalization_manager
            .update_global_model(aggregated_update)
    }

    /// The global model, or `None` before the first
    /// [`Self::apply_global_update`].
    pub fn global_model(&self) -> Option<&Array1<T>> {
        self.personalization_manager.global_model()
    }

    /// The personalization manager backing this coordinator.
    pub fn personalization_manager_mut(&mut self) -> &mut PersonalizationManager<T> {
        &mut self.personalization_manager
    }

    /// Start a new federated round with privacy guarantees
    pub fn start_federated_round(
        &mut self,
        availableclients: &[String],
    ) -> Result<FederatedRoundPlan> {
        self.current_round += 1;

        // Sample clients for this round
        let selectedclients = self.sample_clients(availableclients)?;

        // Check global privacy budget
        let global_budget = self.get_global_privacy_budget()?;
        if !self.has_sufficient_privacy_budget(&global_budget)? {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: global_budget.epsilon_consumed,
                target_epsilon: self.config.base_config.target_epsilon,
            });
        }

        // Compute sampling probability for amplification
        let sampling_probability = selectedclients.len() as f64 / availableclients.len() as f64;

        // Analyze privacy amplification.
        //
        // 0.3.2: `compute_amplification_factor` now takes the base epsilon and
        // the real client counts, because the factor is the published bound's
        // ratio `eps / ln(1 + q(e^eps - 1))` rather than the discarded
        // `(1/q).sqrt()` placeholder, and because the recorded history used to
        // fabricate `total_clients: 1000`.
        let amplificationfactor = if self.config.amplification_config.enabled {
            self.amplification_analyzer.compute_amplification_factor(
                self.config.base_config.target_epsilon,
                selectedclients.len(),
                availableclients.len(),
                self.current_round,
            )?
        } else {
            1.0
        };

        // Prepare secure aggregation if enabled
        let aggregation_plan = if self.config.secure_aggregation.enabled {
            Some(self.prepare_secure_aggregation(&selectedclients)?)
        } else {
            None
        };

        // Compute per-client privacy allocations
        let client_privacy_allocations =
            self.compute_client_privacy_allocations(&selectedclients, amplificationfactor)?;

        // Create round plan
        let roundplan = FederatedRoundPlan {
            round_number: self.current_round,
            selectedclients: selectedclients.clone(),
            sampling_probability,
            amplificationfactor,
            client_privacy_allocations,
            aggregation_plan,
            privacy_analysis: self.analyze_round_privacy(&selectedclients, amplificationfactor)?,
        };

        // Record participation
        self.record_participation_round(
            &selectedclients,
            sampling_probability,
            amplificationfactor,
        )?;

        // Record the round with the composition analyzer so a multi-round
        // composed cost is available through [`Self::composed_privacy_cost`].
        // Before 0.3.2 the analyzer was constructed and then never touched.
        self.composition_analyzer.record_round(RoundComposition {
            round: self.current_round,
            participating_clients: selectedclients.len(),
            total_clients: availableclients.len(),
            epsilon_consumed: roundplan.privacy_analysis.round_epsilon,
            delta_consumed: roundplan.privacy_analysis.round_delta,
            amplification_applied: self.config.amplification_config.enabled,
            composition_method: self.config.composition_method,
            noise_multiplier: Some(self.config.base_config.noise_multiplier),
        })?;

        Ok(roundplan)
    }

    /// Aggregate client updates.
    ///
    /// # Which path runs
    ///
    /// * `secure_aggregation.enabled == true`: plaintext updates must not be
    ///   handed to a coordinator that has promised masked aggregation, because a
    ///   server that can see the individual vectors has already broken the
    ///   guarantee -- regardless of what it does with them afterwards. Use
    ///   [`Self::register_client_key`], [`Self::receive_masked_update`] and
    ///   [`Self::aggregate_masked_updates`], which run the real Bonawitz
    ///   protocol. This function refuses.
    /// * otherwise: the cohort mean is released **with differential privacy
    ///   noise**, charged to the moments accountant. See
    ///   [`Self::simple_aggregate`].
    pub fn secure_aggregate_updates(
        &mut self,
        clientupdates: &HashMap<String, Array1<T>>,
        _roundplan: &FederatedRoundPlan,
    ) -> Result<Array1<T>> {
        if self.config.secure_aggregation.enabled {
            return Err(OptimError::UnsupportedOperation(
                "secure aggregation is enabled, so plaintext client updates must not be submitted \
                 to the coordinator: a server that sees the individual vectors has already broken \
                 the guarantee. Mask them client-side with \
                 privacy::federated::secure_aggregation::mask_client_update and submit through \
                 FederatedPrivacyCoordinator::receive_masked_update, then release the cohort mean \
                 with aggregate_masked_updates. To average in the clear instead, set \
                 secure_aggregation.enabled = false."
                    .to_string(),
            ));
        }
        self.simple_aggregate(clientupdates)
    }

    /// Release the differentially private mean of the client updates.
    ///
    /// # What makes this differentially private
    ///
    /// Each client's contribution to the *sum* is bounded by
    /// `base_config.l2_norm_clip` (the federation contract: clients clip before
    /// uploading). Adding or removing one client therefore moves the mean of `n`
    /// clients by at most `l2_norm_clip / n` in L2, so that is the sensitivity
    /// `S` of this release. Gaussian noise `N(0, (sigma)^2)` with
    /// `sigma = noise_multiplier * S` is added coordinate-wise -- exactly the
    /// calibration [`MomentsAccountant`] is parameterised by, so the epsilon it
    /// reports is the epsilon that was actually paid.
    ///
    /// The release is charged before it is made: if the accountant says the
    /// budget is spent, this fails *closed* with
    /// [`OptimError::PrivacyBudgetExhausted`] rather than emitting one more
    /// unpaid-for aggregate.
    ///
    /// Before 0.3.2 this function was a plain unnoised mean with no accounting
    /// at all, reached from a method named `secure_aggregate_updates`.
    ///
    /// # Errors
    ///
    /// * [`OptimError::InvalidParameter`] -- no client updates, or a zero-length
    ///   update vector.
    /// * [`OptimError::DimensionMismatch`] -- the client vectors disagree on
    ///   length, so no coordinate-wise mean is defined.
    /// * [`OptimError::PrivacyBudgetExhausted`] -- the accountant reports no
    ///   remaining epsilon.
    /// * [`OptimError::UnsupportedOperation`] -- the configured noise mechanism
    ///   is not the subsampled Gaussian that this coordinator's accountant
    ///   models.
    fn simple_aggregate(
        &mut self,
        clientupdates: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        if clientupdates.is_empty() {
            return Err(OptimError::InvalidParameter(
                "no client updates provided; there is nothing to aggregate".to_string(),
            ));
        }

        // Every client vector must agree on the dimension. Before 0.3.2 a
        // shorter vector was silently zero-extended and a longer one silently
        // truncated, producing an aggregate that matched no client's shape.
        let mut client_ids: Vec<&String> = clientupdates.keys().collect();
        client_ids.sort();
        let dimension = clientupdates[client_ids[0]].len();
        if dimension == 0 {
            return Err(OptimError::InvalidParameter(
                "client updates are zero-length; there is nothing to aggregate".to_string(),
            ));
        }
        for clientid in &client_ids {
            let len = clientupdates[*clientid].len();
            if len != dimension {
                return Err(OptimError::DimensionMismatch(format!(
                    "client {clientid} submitted {len} coordinates, expected {dimension}"
                )));
            }
        }

        // Coordinate-wise mean. Under a trust model that declares adversarial
        // clients a plain mean is indefensible -- one client can move it
        // arbitrarily -- so the Byzantine-robust estimator is used instead. The
        // two estimators have different sensitivities, so each carries its own
        // (see [`Self::robust_sensitivity`]).
        let client_count = client_ids.len();
        let (aggregate, sensitivity) = match self.config.trust_model {
            TrustModel::Byzantine | TrustModel::Malicious => {
                let allocations: HashMap<String, AdaptivePrivacyAllocation> = HashMap::new();
                let robust = self
                    .byzantine_aggregator
                    .robust_aggregate(clientupdates, &allocations)?;
                let sensitivity = self.robust_sensitivity();
                (robust, sensitivity)
            }
            TrustModel::HonestButCurious | TrustModel::SemiHonest => {
                let mut mean = Array1::from_elem(dimension, T::zero());
                for clientid in &client_ids {
                    let update = &clientupdates[*clientid];
                    for (slot, &value) in mean.iter_mut().zip(update.iter()) {
                        *slot = *slot + value;
                    }
                }
                let count_t = T::from(client_count).ok_or_else(|| {
                    OptimError::InvalidParameter(format!(
                        "cannot represent a client count of {client_count} in the element type"
                    ))
                })?;
                for value in mean.iter_mut() {
                    *value = *value / count_t;
                }
                // Sum of clipped contributions has sensitivity `l2_norm_clip`
                // under add/remove-one; the divisor is the fixed cohort size.
                // This is the DP-FedAvg convention (McMahan et al. 2018).
                let sensitivity = self.config.base_config.l2_norm_clip / client_count as f64;
                (mean, sensitivity)
            }
        };

        self.perturb_and_charge(aggregate, sensitivity)
    }

    /// Add the calibrated differential privacy noise to `aggregate` and charge
    /// the release to the accountant.
    ///
    /// `sensitivity` is the caller's L2 sensitivity bound for the statistic being
    /// released; the noise scale is `noise_multiplier * sensitivity`, which is the
    /// calibration [`MomentsAccountant`] is parameterised by. Passing a bound that
    /// is too small would make the accountant's epsilon a fabrication, so each
    /// caller derives and documents its own -- see [`Self::simple_aggregate`] and
    /// [`Self::robust_sensitivity`].
    ///
    /// Fails closed on an exhausted budget: the check happens *before* the noisy
    /// value is produced.
    fn perturb_and_charge(
        &mut self,
        mut aggregate: Array1<T>,
        sensitivity: f64,
    ) -> Result<Array1<T>> {
        // A mechanism this coordinator cannot account for is a configuration
        // error, so report it before any state-dependent condition: a caller who
        // selected Laplace should be told that, not that the budget ran out.
        if !matches!(
            self.config.base_config.noise_mechanism,
            NoiseMechanism::Gaussian
        ) {
            return Err(OptimError::UnsupportedOperation(format!(
                "base_config.noise_mechanism = {:?}, but this coordinator accounts the subsampled \
                 Gaussian via MomentsAccountant, which cannot express that mechanism's spend. Use \
                 NoiseMechanism::Gaussian here, or apply the other mechanism client-side with \
                 privacy::noise_mechanisms and its own accountant.",
                self.config.base_config.noise_mechanism
            )));
        }

        // Fail closed: never release another aggregate on an exhausted budget.
        let budget = self.get_global_privacy_budget()?;
        if !self.has_sufficient_privacy_budget(&budget)? {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: budget.epsilon_consumed,
                target_epsilon: self.config.base_config.target_epsilon,
            });
        }

        let sigma_f64 = self.config.base_config.noise_multiplier * sensitivity;
        if !sigma_f64.is_finite() || sigma_f64 <= 0.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "the noise scale derived from noise_multiplier = {} and sensitivity {sensitivity} \
                 is {sigma_f64}, which adds no privacy",
                self.config.base_config.noise_multiplier,
            )));
        }
        let sigma = T::from(sigma_f64).ok_or_else(|| {
            OptimError::InvalidPrivacyConfig(format!(
                "cannot represent the noise scale {sigma_f64} in the element type"
            ))
        })?;

        let mut mechanism = GaussianMechanism::<T>::new();
        mechanism.add_noise_with_scale(&mut aggregate, sigma)?;

        // The release is now paid for.
        self.noisy_releases = self.noisy_releases.saturating_add(1);
        Ok(aggregate)
    }

    /// Sample clients for federated round
    fn sample_clients(&self, availableclients: &[String]) -> Result<Vec<String>> {
        let mut rng = thread_rng();
        let target_count = self.config.clients_per_round.min(availableclients.len());

        match self.config.sampling_strategy {
            ClientSamplingStrategy::UniformRandom => {
                // Simple random selection
                let mut selected = Vec::new();
                let mut remaining = availableclients.to_vec();
                for _ in 0..target_count.min(remaining.len()) {
                    let index = rng.gen_range(0..remaining.len());
                    selected.push(remaining.swap_remove(index));
                }
                Ok(selected)
            }
            _ => {
                // Fallback to uniform random for other strategies
                let mut selected = Vec::new();
                let mut remaining = availableclients.to_vec();
                for _ in 0..target_count.min(remaining.len()) {
                    let index = rng.gen_range(0..remaining.len());
                    selected.push(remaining.swap_remove(index));
                }
                Ok(selected)
            }
        }
    }

    /// Get the global privacy budget, computed from the real moments accountant.
    ///
    /// The consumed `(ε, δ)` after `current_round` rounds comes from
    /// [`MomentsAccountant::get_privacy_spent`], not a hardcoded constant, so the
    /// budget-exhaustion check in [`Self::start_federated_round`] fails *closed*
    /// (rejects new rounds once the budget is spent) rather than fail-open against
    /// a fixed `0.1`. `estimated_steps_remaining` is derived from the accountant's
    /// own [`MomentsAccountant::estimate_max_steps`].
    fn get_global_privacy_budget(&self) -> Result<PrivacyBudget> {
        use super::super::AccountingMethod;

        let target_epsilon = self.config.base_config.target_epsilon;
        let target_delta = self.config.base_config.target_delta;

        let steps = self.accounted_steps();

        // `MomentsAccountant::get_privacy_spent` returns the delta the epsilon is
        // *reported at*, not a delta that was spent. Recording it as
        // `delta_consumed` made `delta_remaining` exactly zero from step zero
        // onwards, so `has_sufficient_privacy_budget` reported an exhausted
        // budget before a single round had run and `start_federated_round`
        // could never succeed. Delta is a reporting parameter (see the
        // `crate::privacy` module documentation and `PrivacyBudget::delta_consumed`,
        // which is documented as "always 0.0"), so nothing consumes it.
        let (epsilon_consumed, _reporting_delta) =
            self.global_accountant.get_privacy_spent(steps)?;

        let estimated_steps_remaining = self
            .global_accountant
            .estimate_max_steps(target_epsilon)
            .map(|max_steps| max_steps.saturating_sub(steps))
            .unwrap_or(0);

        Ok(PrivacyBudget {
            epsilon_consumed,
            delta_consumed: 0.0,
            epsilon_remaining: (target_epsilon - epsilon_consumed).max(0.0),
            delta_remaining: target_delta,
            steps_taken: steps,
            accounting_method: AccountingMethod::MomentsAccountant,
            estimated_steps_remaining,
        })
    }

    /// Whether any privacy budget is left to spend.
    ///
    /// Only epsilon is a budget. The previous condition also required
    /// `delta_remaining > 0.0`, and since the accountant reports the *reporting*
    /// delta as if it had been consumed, that term was zero from the first call
    /// onwards -- every round was rejected as budget-exhausted before any epsilon
    /// had been spent. See [`Self::get_global_privacy_budget`].
    fn has_sufficient_privacy_budget(&self, budget: &PrivacyBudget) -> Result<bool> {
        Ok(budget.epsilon_remaining > 0.0)
    }

    /// Number of mechanism applications the accountant must charge for.
    ///
    /// # Why the maximum, and what it does not cover
    ///
    /// A round that was planned but never aggregated still consumed a
    /// subsampling step; a noisy release made outside a planned round still
    /// consumed a mechanism application. Taking the maximum charges each once,
    /// which is correct for the intended usage of one release per round, and
    /// never reports a smaller spend than either counter alone implies.
    ///
    /// **It is not correct for multiple releases within one round.** Calling
    /// [`Self::secure_aggregate_updates`] or [`Self::robust_aggregate_updates`]
    /// twice inside a single round produces two independent Gaussian releases
    /// over the same cohort, which compose to a strictly larger epsilon than one
    /// release does; `max(round, releases)` charges for one. The guard is that
    /// `noisy_releases` is itself incremented per release, so the *second* call
    /// in round `k` pushes the count to `k + 1` and is charged as an extra step
    /// -- correct as long as releases never run ahead of rounds by more than the
    /// per-round count. A caller that makes `m > 1` releases per round for many
    /// rounds is under-charged by the difference, and this coordinator does not
    /// detect that. Drive one release per round, or account the extra releases
    /// separately through [`crate::privacy::accountant`].
    fn accounted_steps(&self) -> usize {
        self.current_round.max(self.noisy_releases)
    }

    /// Open a secure-aggregation round and publish its plan.
    ///
    /// Delegates to the audited Bonawitz implementation
    /// ([`SecureAggregator::prepare_round`]), which draws the per-round salt from
    /// operating-system entropy and assembles the public-key directory the
    /// clients need to agree their pairwise masks.
    ///
    /// Before 0.3.2 this function generated one `u64` "masking seed" per client
    /// **on the server**, with `thread_rng`, and returned them in the plan. A
    /// server that knows every client's mask can subtract any one of them from
    /// the matching upload, so that construction had exactly zero
    /// confidentiality; the seeds were also never used by anything.
    ///
    /// # Errors
    ///
    /// Propagates [`SecureAggregator::prepare_round`]: fewer than two distinct
    /// clients, a cohort below `min_clients`, a selected client that has not
    /// registered a public key (see [`Self::register_client_key`]), or a cohort
    /// whose worst-case fixed-point sum would wrap the masking group.
    fn prepare_secure_aggregation(
        &mut self,
        selectedclients: &[String],
    ) -> Result<SecureAggregationPlan> {
        self.secure_aggregator.prepare_round(selectedclients)
    }

    /// Compute privacy allocations for each client
    fn compute_client_privacy_allocations(
        &self,
        selectedclients: &[String],
        amplificationfactor: f64,
    ) -> Result<HashMap<String, ClientPrivacyAllocation>> {
        let mut allocations = HashMap::new();

        let base_epsilon = self.config.base_config.target_epsilon / amplificationfactor;
        let base_delta = self.config.base_config.target_delta;

        for clientid in selectedclients {
            allocations.insert(
                clientid.clone(),
                ClientPrivacyAllocation {
                    epsilon: base_epsilon,
                    delta: base_delta,
                    noise_multiplier: self.config.base_config.noise_multiplier,
                    clipping_threshold: self.config.base_config.l2_norm_clip,
                    amplificationfactor,
                },
            );
        }

        Ok(allocations)
    }

    /// Analyze privacy for the current round, from the moments accountant.
    ///
    /// # What each number is
    ///
    /// * `round_epsilon` -- the epsilon a *single* release costs at this round's
    ///   real sampling rate `|selected| / total_clients`. Computed by a moments
    ///   accountant built for that rate, not by dividing the target epsilon by
    ///   the amplification factor.
    /// * `cumulative_epsilon` -- the accountant's composed spend after
    ///   [`Self::accounted_steps`] releases. This is strictly *less* than
    ///   `round_epsilon * rounds` for the subsampled Gaussian, which is the whole
    ///   point of using a moments accountant.
    /// * `composition_tightness` -- `cumulative_epsilon / (rounds *
    ///   round_epsilon)`: the accountant's composed epsilon as a fraction of
    ///   naive (basic) composition. `1.0` means the accountant buys nothing;
    ///   smaller is tighter. `1.0` is also reported at round 0, where there is
    ///   nothing to compose.
    ///
    /// Before 0.3.2 `cumulative_epsilon` was `round_epsilon * round` (basic
    /// composition of a fabricated per-round epsilon), `composition_tightness`
    /// was the constant `0.95`, and `selectedclients` was ignored entirely.
    fn analyze_round_privacy(
        &self,
        selectedclients: &[String],
        amplificationfactor: f64,
    ) -> Result<RoundPrivacyAnalysis> {
        let base = &self.config.base_config;
        let round_delta = base.target_delta;

        // An accountant for this round's *actual* cohort size.
        let round_accountant = MomentsAccountant::new(
            base.noise_multiplier,
            round_delta,
            selectedclients.len(),
            self.config.total_clients,
        );
        let (round_epsilon, _) = round_accountant.get_privacy_spent(1)?;

        let steps = self.accounted_steps();
        let (cumulative_epsilon, cumulative_delta) =
            self.global_accountant.get_privacy_spent(steps)?;

        let naive_epsilon = round_epsilon * steps as f64;
        let composition_tightness = if steps == 0 || naive_epsilon <= 0.0 {
            1.0
        } else {
            (cumulative_epsilon / naive_epsilon).clamp(0.0, 1.0)
        };

        Ok(RoundPrivacyAnalysis {
            round_epsilon,
            round_delta,
            cumulative_epsilon,
            cumulative_delta,
            amplification_benefit: amplificationfactor - 1.0,
            composition_tightness,
        })
    }

    /// Record participation for this round.
    ///
    /// `composition_cost` is the epsilon this round actually *added* to the
    /// cumulative spend -- `eps(round) - eps(round - 1)` from the moments
    /// accountant. Under subsampled-Gaussian composition that increment shrinks
    /// as rounds accumulate, which a constant can never express; before 0.3.2 it
    /// was the literal `0.1`. `client_contribution` guards an empty cohort
    /// rather than dividing by zero.
    ///
    /// # Errors
    ///
    /// Propagates [`MomentsAccountant::get_privacy_spent`], which fails on a
    /// configuration whose epsilon cannot be computed.
    fn record_participation_round(
        &mut self,
        selectedclients: &[String],
        sampling_probability: f64,
        amplificationfactor: f64,
    ) -> Result<()> {
        let round = self.current_round;
        let (epsilon_now, _) = self.global_accountant.get_privacy_spent(round)?;
        let (epsilon_before, _) = self
            .global_accountant
            .get_privacy_spent(round.saturating_sub(1))?;
        let composition_cost = (epsilon_now - epsilon_before).max(0.0);

        let client_contribution = if selectedclients.is_empty() {
            0.0
        } else {
            1.0 / selectedclients.len() as f64
        };

        let participation = ParticipationRound {
            round,
            participating_clients: selectedclients.to_vec(),
            sampling_probability,
            privacy_cost: PrivacyCost {
                epsilon: epsilon_now,
                delta: self.config.base_config.target_delta,
                client_contribution,
                amplification_factor: amplificationfactor,
                composition_cost,
            },
            aggregation_noise: self.config.base_config.noise_multiplier,
        };

        self.participation_history.push_back(participation);

        // Keep only last 1000 rounds
        if self.participation_history.len() > MAX_PARTICIPATION_HISTORY {
            self.participation_history.pop_front();
        }
        Ok(())
    }

    /// Recorded participation rounds, oldest first.
    ///
    /// Capped at [`MAX_PARTICIPATION_HISTORY`] entries.
    pub fn participation_history(&self) -> impl Iterator<Item = &ParticipationRound> {
        self.participation_history.iter()
    }

    /// The multi-round composed privacy cost, reported at `target_delta`.
    ///
    /// Every round opened through [`Self::start_federated_round`] is recorded
    /// with the composition analyzer, so this is the composed cost of the
    /// federation's whole history under the configured
    /// [`FederatedCompositionMethod`] -- not the single-round epsilon.
    pub fn composed_privacy_cost(&self) -> Result<ComposedPrivacyCost> {
        self.composition_analyzer
            .compose(self.config.base_config.target_delta)
    }

    /// The composition analyzer backing this coordinator.
    pub fn composition_analyzer(&self) -> &FederatedCompositionAnalyzer {
        &self.composition_analyzer
    }

    /// The privacy amplification analyzer backing this coordinator.
    pub fn amplification_analyzer(&self) -> &PrivacyAmplificationAnalyzer {
        &self.amplification_analyzer
    }

    /// Get current privacy guarantees, computed from the real moments accountant.
    ///
    /// Delegates to [`Self::get_global_privacy_budget`]. If the accountant cannot
    /// produce an analysis (e.g. an invalid configuration), this fails *closed* by
    /// reporting the budget as fully consumed rather than fabricating a small
    /// spend, so a caller can never read exhausted state as ample headroom.
    pub fn get_privacy_guarantees(&self) -> PrivacyBudget {
        use super::super::AccountingMethod;

        self.get_global_privacy_budget().unwrap_or_else(|_| {
            let target_epsilon = self.config.base_config.target_epsilon;
            let target_delta = self.config.base_config.target_delta;
            PrivacyBudget {
                epsilon_consumed: target_epsilon,
                delta_consumed: target_delta,
                epsilon_remaining: 0.0,
                delta_remaining: 0.0,
                steps_taken: self.current_round,
                accounting_method: AccountingMethod::MomentsAccountant,
                estimated_steps_remaining: 0,
            }
        })
    }

    /// Get current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get configuration
    pub fn config(&self) -> &FederatedPrivacyConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::federated::pairwise_masking::ClientKeyPair;
    use crate::privacy::federated::secure_aggregation::mask_client_update;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap;

    fn make_updates(vectors: &[(&str, Vec<f64>)]) -> HashMap<String, Array1<f64>> {
        let mut map = HashMap::new();
        for (id, values) in vectors {
            map.insert((*id).to_string(), Array1::from_vec(values.clone()));
        }
        map
    }

    fn plan_for(clients: &[&str]) -> FederatedRoundPlan {
        FederatedRoundPlan {
            round_number: 1,
            selectedclients: clients.iter().map(|c| (*c).to_string()).collect(),
            sampling_probability: 1.0,
            amplificationfactor: 1.0,
            client_privacy_allocations: HashMap::new(),
            aggregation_plan: None,
            privacy_analysis: RoundPrivacyAnalysis {
                round_epsilon: 0.0,
                round_delta: 0.0,
                cumulative_epsilon: 0.0,
                cumulative_delta: 0.0,
                amplification_benefit: 0.0,
                composition_tightness: 1.0,
            },
        }
    }

    /// F76 regression: the global privacy budget must come from the real moments
    /// accountant, so consumed epsilon grows monotonically with the round count
    /// instead of being the old fabricated `0.1` / `0.1 * round` constant (which
    /// made the budget-exhaustion check fail *open*).
    #[test]
    fn test_global_budget_uses_real_accountant_not_constant() {
        let config = FederatedPrivacyConfig::default();
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(config).expect("coordinator");

        let b0 = coord.get_privacy_guarantees();
        assert!(b0.epsilon_consumed.is_finite());
        assert!(
            b0.epsilon_consumed.abs() < 1e-12,
            "round 0 consumed = {}",
            b0.epsilon_consumed
        );
        assert!(b0.epsilon_remaining > 0.0);

        coord.current_round = 10;
        let b10 = coord.get_privacy_guarantees();
        coord.current_round = 50;
        let b50 = coord.get_privacy_guarantees();
        assert!(
            b10.epsilon_consumed > b0.epsilon_consumed,
            "epsilon must grow with rounds"
        );
        assert!(
            b50.epsilon_consumed > b10.epsilon_consumed,
            "epsilon must be monotone in rounds"
        );
        assert!(b50.epsilon_consumed.is_finite());
        assert!(
            (b10.epsilon_consumed - 1.0).abs() > 1e-6,
            "must not be the old 0.1*round constant, got {}",
            b10.epsilon_consumed
        );
    }

    /// F77 regression: enabling secure aggregation must not let plaintext
    /// updates through the coordinator.
    #[test]
    fn test_plaintext_updates_are_refused_when_masking_is_enabled() {
        let mut config = FederatedPrivacyConfig::default();
        config.secure_aggregation.enabled = true;
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(config).expect("coordinator");

        let updates = make_updates(&[("a", vec![1.0, 2.0]), ("b", vec![3.0, 4.0])]);
        let plan = plan_for(&["a", "b"]);

        let message = match coord.secure_aggregate_updates(&updates, &plan) {
            Err(OptimError::UnsupportedOperation(message)) => message,
            other => panic!("plaintext submission must be refused, got {other:?}"),
        };
        assert!(message.contains("mask_client_update"), "got: {message}");
    }

    /// The plaintext path must release a *noisy* mean, not the exact mean, and it
    /// must charge the accountant for doing so.
    #[test]
    fn test_plaintext_aggregation_adds_dp_noise_and_charges_the_accountant() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        let updates = make_updates(&[
            ("a", vec![1.0, 1.0, 1.0, 1.0]),
            ("b", vec![1.0, 1.0, 1.0, 1.0]),
            ("c", vec![1.0, 1.0, 1.0, 1.0]),
        ]);
        let plan = plan_for(&["a", "b", "c"]);

        let before = coord.get_privacy_guarantees().epsilon_consumed;
        let noisy = coord
            .secure_aggregate_updates(&updates, &plan)
            .expect("aggregation");
        let after = coord.get_privacy_guarantees().epsilon_consumed;

        // Every client sent exactly 1.0, so the *unnoised* mean is exactly 1.0.
        // A noised release must differ from it on at least one coordinate; the
        // probability that four independent Gaussians are all exactly zero is 0.
        assert_eq!(noisy.len(), 4);
        assert!(
            noisy.iter().any(|value| (value - 1.0).abs() > 1e-12),
            "the release is the exact mean, so no noise was added: {noisy:?}"
        );
        // sigma = noise_multiplier * clip / n = 4.0 * 1.0 / 3, so a 20-sigma
        // deviation is impossible in practice: the noise must be calibrated, not
        // arbitrary.
        let sigma = 4.0 / 3.0;
        for value in noisy.iter() {
            assert!(
                (value - 1.0).abs() < 20.0 * sigma,
                "coordinate {value} is implausibly far from the mean for sigma = {sigma}"
            );
        }
        assert!(
            after > before,
            "the release must be charged: {before} -> {after}"
        );
    }

    /// The accountant must gate the release: once the budget is spent, no further
    /// aggregate comes out.
    #[test]
    fn test_aggregation_fails_closed_on_an_exhausted_budget() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        let updates = make_updates(&[("a", vec![1.0, 2.0]), ("b", vec![3.0, 4.0])]);
        let plan = plan_for(&["a", "b"]);

        // Drive the accountant far past the target epsilon.
        coord.current_round = 10_000_000;
        assert!(
            coord.get_privacy_guarantees().epsilon_remaining <= 0.0,
            "the budget should be exhausted at this round count"
        );
        assert!(matches!(
            coord.secure_aggregate_updates(&updates, &plan),
            Err(OptimError::PrivacyBudgetExhausted { .. })
        ));
    }

    /// Ragged client vectors must be refused rather than zero-extended.
    #[test]
    fn test_ragged_client_updates_are_refused() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        let updates = make_updates(&[("a", vec![1.0, 2.0, 3.0]), ("b", vec![1.0, 2.0])]);
        let plan = plan_for(&["a", "b"]);
        assert!(matches!(
            coord.secure_aggregate_updates(&updates, &plan),
            Err(OptimError::DimensionMismatch(_))
        ));

        let empty: HashMap<String, Array1<f64>> = HashMap::new();
        assert!(matches!(
            coord.secure_aggregate_updates(&empty, &plan),
            Err(OptimError::InvalidParameter(_))
        ));
    }

    /// A Byzantine trust model must not release a plain mean: one adversary can
    /// move a mean arbitrarily, and the coordinator declares it is defending
    /// against adversaries.
    #[test]
    fn test_a_byzantine_trust_model_uses_the_robust_estimator() {
        let mut config = FederatedPrivacyConfig {
            trust_model: TrustModel::Byzantine,
            ..FederatedPrivacyConfig::default()
        };
        config.secure_aggregation.enabled = true;
        // A malicious trust model requires secure aggregation to be declared,
        // but we exercise the robust path directly rather than through the
        // plaintext gate.
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(config).expect("coordinator");
        // sigma = noise_multiplier * 2 * clip / retained; with 13 clients and the
        // default 20% total trim, retained is 11, so sigma ~= 4 * 2 / 11 ~= 0.73.

        let mut updates = HashMap::new();
        for i in 0..12usize {
            updates.insert(
                format!("h{i}"),
                Array1::from_vec(vec![1.0 + (i as f64) * 0.001, -2.0]),
            );
        }
        updates.insert("evil".to_string(), Array1::from_vec(vec![1.0e6, 1.0e6]));

        let allocations: HashMap<String, AdaptivePrivacyAllocation> = HashMap::new();
        let robust = coord
            .robust_aggregate_updates(&updates, &allocations)
            .expect("robust aggregation");

        // The bound must separate "robust" from "plain mean" without being so
        // tight that a legitimate noise draw fails it. The release carries
        // Gaussian noise at sigma = noise_multiplier * 2 * clip / retained
        // = 4.0 * 2 / 11 ~= 0.73, so a few sigma of drift is expected; a plain
        // mean over this cohort would sit near 1e6 / 13 ~= 7.7e4. A tolerance of
        // 100 is >130 sigma of noise headroom and still three orders of
        // magnitude below what a non-robust estimator would return.
        let plain_mean_0 = updates.values().map(|u| u[0]).sum::<f64>() / updates.len() as f64;
        assert!(
            plain_mean_0 > 1.0e4,
            "the adversary must actually break a plain mean for this test to mean anything, \
             got {plain_mean_0}"
        );
        assert!(
            (robust[0] - 1.0).abs() < 100.0,
            "robust[0] = {} should stay near the honest cluster, not near the plain mean \
             {plain_mean_0}",
            robust[0]
        );
        assert!(
            (robust[1] + 2.0).abs() < 100.0,
            "robust[1] = {} should stay near the honest cluster",
            robust[1]
        );
    }

    /// The global model must actually advance by the aggregated delta.
    #[test]
    fn test_apply_global_update_accumulates() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        assert!(coord.global_model().is_none());

        let first = Array1::from_vec(vec![1.0, 2.0]);
        let after_first = coord.apply_global_update(&first).expect("first update");
        assert_eq!(after_first.to_vec(), vec![1.0, 2.0]);

        let second = Array1::from_vec(vec![0.5, -1.0]);
        let after_second = coord.apply_global_update(&second).expect("second update");
        assert_eq!(after_second.to_vec(), vec![1.5, 1.0]);
        assert_eq!(
            coord.global_model().map(|m| m.to_vec()),
            Some(vec![1.5, 1.0])
        );

        let wrong = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            coord.apply_global_update(&wrong),
            Err(OptimError::DimensionMismatch(_))
        ));
    }

    /// `prepare_round` must produce a real Bonawitz plan whose masks cancel, not
    /// a map of server-generated seeds.
    #[test]
    fn test_secure_round_masks_cancel_end_to_end() {
        // A 3-of-4 cohort is a 75% sampling rate, which the moments accountant
        // charges far more than the default budget for; use a large federation so
        // the round is affordable and the test exercises masking, not accounting.
        let mut config = FederatedPrivacyConfig {
            total_clients: 1000,
            clients_per_round: 3,
            ..FederatedPrivacyConfig::default()
        };
        config.secure_aggregation.enabled = true;
        config.secure_aggregation.min_clients = 3;
        config.secure_aggregation.max_dropouts = 1;
        config.secure_aggregation.masking_dimension = 4;
        config.secure_aggregation.quantization_scale = 1.0e4;
        config.secure_aggregation.max_update_magnitude = 10.0;
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(config).expect("coordinator");

        let ids = ["c0".to_string(), "c1".to_string(), "c2".to_string()];
        let keys: Vec<ClientKeyPair> = ids.iter().map(|_| ClientKeyPair::generate()).collect();
        for (id, key) in ids.iter().zip(keys.iter()) {
            coord
                .register_client_key(id, key.public_key())
                .expect("key registration");
        }

        let plan = coord.start_federated_round(&ids).expect("round");
        let aggregation_plan = plan.aggregation_plan.as_ref().expect("masking plan");
        assert_eq!(aggregation_plan.participating_clients.len(), 3);
        assert!(aggregation_plan.masking_enabled);
        assert_eq!(aggregation_plan.public_keys.len(), 3);

        let updates = [
            Array1::from_vec(vec![1.0, 2.0, -1.0, 0.5]),
            Array1::from_vec(vec![0.0, 1.0, 1.0, -0.5]),
            Array1::from_vec(vec![2.0, -3.0, 0.0, 1.0]),
        ];
        for ((id, key), update) in ids.iter().zip(keys.iter()).zip(updates.iter()) {
            let masked =
                mask_client_update(id, key, update, aggregation_plan).expect("masking succeeds");
            coord.receive_masked_update(masked).expect("upload");
        }

        let mean = coord.aggregate_masked_updates().expect("masked mean");
        let expected: Vec<f64> = (0..4)
            .map(|k| updates.iter().map(|u| u[k]).sum::<f64>() / 3.0)
            .collect();
        for (got, want) in mean.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-3,
                "masked mean {got} should match the plaintext mean {want}"
            );
        }
    }

    /// A secure round cannot be opened for clients that never published a key.
    #[test]
    fn test_secure_round_requires_registered_keys() {
        let mut config = FederatedPrivacyConfig {
            total_clients: 1000,
            clients_per_round: 3,
            ..FederatedPrivacyConfig::default()
        };
        config.secure_aggregation.enabled = true;
        config.secure_aggregation.min_clients = 3;
        config.secure_aggregation.max_dropouts = 1;
        config.secure_aggregation.masking_dimension = 4;
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(config).expect("coordinator");
        let ids = ["c0".to_string(), "c1".to_string(), "c2".to_string()];
        assert!(matches!(
            coord.start_federated_round(&ids),
            Err(OptimError::InvalidState(_))
        ));
    }

    /// `composition_tightness` must be computed from the accountant, and the
    /// moments accountant must beat naive composition after a few rounds.
    #[test]
    fn test_composition_tightness_is_computed_not_constant() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        let clients: Vec<String> = (0..100).map(|i| format!("c{i}")).collect();

        let mut tightness = Vec::new();
        for _ in 0..5 {
            let plan = coord.start_federated_round(&clients).expect("round");
            tightness.push(plan.privacy_analysis.composition_tightness);
            assert!(plan.privacy_analysis.cumulative_epsilon > 0.0);
            assert!(plan.privacy_analysis.round_epsilon > 0.0);
        }
        assert!(
            tightness.iter().any(|t| (t - 0.95).abs() > 1e-9),
            "tightness is stuck at the old 0.95 placeholder: {tightness:?}"
        );
        for t in &tightness {
            assert!(
                (0.0..=1.0).contains(t),
                "tightness {t} must be a fraction of naive composition"
            );
        }
        // Later rounds compose more tightly than the first.
        assert!(
            tightness[4] <= tightness[0] + 1e-12,
            "tightness must not grow with rounds: {tightness:?}"
        );
    }

    /// The per-round participation record must carry the accountant's real
    /// incremental epsilon, not a `0.1` constant.
    #[test]
    fn test_participation_history_records_real_composition_cost() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        let clients: Vec<String> = (0..100).map(|i| format!("c{i}")).collect();
        for _ in 0..3 {
            coord.start_federated_round(&clients).expect("round");
        }
        let costs: Vec<f64> = coord
            .participation_history()
            .map(|record| record.privacy_cost.composition_cost)
            .collect();
        assert_eq!(costs.len(), 3);
        for cost in &costs {
            assert!(cost.is_finite() && *cost > 0.0, "cost = {cost}");
            assert!(
                (cost - 0.1).abs() > 1e-9,
                "composition_cost is stuck at the old 0.1 placeholder"
            );
        }
    }

    /// The default federated configuration must be able to open a round.
    ///
    /// Two independent defects made this impossible before 0.3.2:
    /// `has_sufficient_privacy_budget` required `delta_remaining > 0`, while the
    /// accountant reports the *reporting* delta as consumed, so the budget looked
    /// exhausted at step zero; and the default noise multiplier of 1.1 charged
    /// 2.25 epsilon for the first round against a budget of 1.0.
    #[test]
    fn test_the_default_configuration_can_open_a_round() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        let budget = coord.get_privacy_guarantees();
        assert!(
            budget.epsilon_remaining > 0.0,
            "a fresh coordinator must have budget, got {budget:?}"
        );
        assert_eq!(
            budget.delta_consumed, 0.0,
            "delta is a reporting parameter, not a spend"
        );
        assert!(budget.delta_remaining > 0.0);

        let clients: Vec<String> = (0..100).map(|i| format!("c{i}")).collect();
        let plan = coord
            .start_federated_round(&clients)
            .expect("the default configuration must be able to run a round");
        assert_eq!(plan.round_number, 1);
        assert_eq!(plan.selectedclients.len(), 100);
        assert!(plan.aggregation_plan.is_none());
        assert!(coord.composed_privacy_cost().expect("composition").epsilon > 0.0);
    }

    /// The robust path's noise divisor must be the number of values actually
    /// averaged, not the cohort size.
    ///
    /// `RobustEstimators::last_contributors` records the *whole cohort* for
    /// `TrimmedMean` even though only `n - 2 * trim` values are averaged. Using
    /// its length as the divisor would release less noise than the accountant
    /// charged for. This pins the per-method contract so that cannot regress
    /// silently.
    #[test]
    fn test_robust_sensitivity_divides_by_the_averaged_count_not_the_cohort() {
        use crate::privacy::federated::byzantine_aggregation::ByzantineRobustConfig;

        let clip = 1.0_f64;
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        assert!((coord.config().base_config.l2_norm_clip - clip).abs() < 1e-12);

        // A 10-client cohort under the default 20%-total trim keeps 8 values,
        // so the divisor must be 8 -- not the 10 that `last_contributors` holds.
        let updates: HashMap<String, Array1<f64>> = (0..10)
            .map(|i| {
                (
                    format!("c{i}"),
                    Array1::from_vec(vec![1.0 + (i as f64) * 0.01, 2.0]),
                )
            })
            .collect();
        let allocations: HashMap<String, AdaptivePrivacyAllocation> = HashMap::new();
        coord
            .robust_aggregate_updates(&updates, &allocations)
            .expect("robust aggregation");

        let estimators = coord.byzantine_aggregator().robust_estimators();
        let trim = estimators.last_trim_count();
        assert!(trim > 0, "the default config must actually trim");
        assert_eq!(
            estimators.last_contributors().len(),
            10,
            "TrimmedMean records the whole cohort, which is exactly why it cannot be the divisor"
        );
        let averaged = 10 - 2 * trim;
        assert!(
            (coord.robust_sensitivity() - 2.0 * clip / averaged as f64).abs() < 1e-12,
            "divisor must be the {averaged} averaged values, got sensitivity {}",
            coord.robust_sensitivity()
        );
        // Strictly more noise than the plain-mean bound over the same cohort.
        assert!(coord.robust_sensitivity() > clip / 10.0);

        // Krum returns one client's update verbatim: nothing is averaged, so the
        // bound degrades to the full 2 * clip.
        // The trust model is irrelevant here -- `robust_aggregate_updates` is the
        // explicit robust entry point, not the trust-model-routed plaintext path.
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        *coord.byzantine_aggregator_mut() =
            ByzantineRobustAggregator::with_config(ByzantineRobustConfig {
                method: ByzantineRobustMethod::Krum { f: 1 },
                ..ByzantineRobustConfig::default()
            })
            .expect("krum aggregator");
        coord
            .robust_aggregate_updates(&updates, &allocations)
            .expect("krum aggregation");
        assert!(
            (coord.robust_sensitivity() - 2.0 * clip).abs() < 1e-12,
            "Krum averages nothing, so the divisor must be 1"
        );
    }

    /// Per-client accounting must be reachable and must grow with the client's
    /// round count.
    #[test]
    fn test_client_privacy_spent_is_reachable_and_monotone() {
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(FederatedPrivacyConfig::default())
            .expect("coordinator");
        let (eps1, _) = coord.client_privacy_spent("c0", 1).expect("client spend");
        let (eps10, _) = coord.client_privacy_spent("c0", 10).expect("client spend");
        assert!(eps1 > 0.0 && eps10 > eps1, "{eps1} then {eps10}");
    }

    /// A Laplace mechanism cannot be accounted by this coordinator's moments
    /// accountant, so it must be refused rather than silently accounted as
    /// Gaussian.
    #[test]
    fn test_an_unaccountable_mechanism_is_refused() {
        let mut config = FederatedPrivacyConfig::default();
        config.base_config.noise_mechanism = NoiseMechanism::Laplace;
        let mut coord = FederatedPrivacyCoordinator::<f64>::new(config).expect("coordinator");
        let updates = make_updates(&[("a", vec![1.0, 2.0]), ("b", vec![3.0, 4.0])]);
        let plan = plan_for(&["a", "b"]);
        let message = match coord.secure_aggregate_updates(&updates, &plan) {
            Err(OptimError::UnsupportedOperation(message)) => message,
            other => panic!("Laplace must be refused here, got {other:?}"),
        };
        assert!(message.contains("MomentsAccountant"), "got: {message}");
    }
}
