// Secure Aggregation Module
//
// Bonawitz-style secure aggregation for federated learning: clients mask their
// updates with pairwise secrets agreed by X25519 key exchange, upload the
// masked values, and the server -- which holds no key material at all -- adds
// the uploads together. The masks cancel exactly, so the server learns the
// cohort's sum and nothing about any individual update.
//
// What was wrong before
// ---------------------
// The previous revision of this file was not a weakened protocol, it was the
// inverse of one:
//
//   * `prepare_round` generated *every client's mask on the server*
//     (`Random::default()` inside the server loop) and stored them in a
//     server-side map. A server that knows all the masks can subtract any one
//     of them from the matching upload, so confidentiality was exactly zero.
//   * The masks were drawn from `-1.0..1.0` and simply *added* to each update
//     during aggregation, then never removed. They therefore did not cancel:
//     the "aggregate" was the mean of the updates plus the mean of a pile of
//     random noise, i.e. neither private nor correct.
//   * `prepare_round` panicked on a poisoned lock (`.expect("lock poisoned")`)
//     over a mutex guarding nothing but a monotone counter.
//
// The current implementation moves all key material and all mask generation to
// the clients (see [`super::pairwise_masking`]), leaves the server with
// modular addition, and replaces the mutex-guarded counter with a per-round
// salt drawn from operating-system entropy -- so the poisoning failure mode no
// longer exists rather than being handled.
//
// Threat model, stated honestly
// -----------------------------
// What this provides: an honest-but-curious server that does not collude with
// any client learns only the sum of the received updates. Confidentiality of
// an individual update holds as long as at least two clients in the round keep
// their secret keys, since the mask of a client is the sum of its pairwise
// masks with all peers.
//
// What it does not provide:
//   * No self-mask. Bonawitz's full protocol adds a per-client mask `b_i`
//     shared out via Shamir secret sharing, which defends against a server
//     that declares a client dropped, collects the pairwise-mask disclosures,
//     and *then* processes that client's delayed upload. This implementation
//     instead closes the same hole procedurally: once a client is marked
//     dropped its upload is refused (see [`SecureAggregator::receive`]), and a
//     client that has submitted cannot be marked dropped.
//   * No malicious-server verification. A server that lies about the
//     public-key directory (a man-in-the-middle on the key distribution step)
//     can learn individual updates. Deployments must authenticate the
//     directory out of band.
//   * No differential privacy. Secure aggregation hides individual updates
//     from the server; it says nothing about what the *sum* reveals. Compose
//     it with the differential privacy machinery in [`crate::privacy`] for
//     that; `SecureAggregationConfig::aggregate_dp` is rejected here rather
//     than silently pretending to supply it.
//
// Reference
// ---------
//   * Bonawitz, K. et al. "Practical Secure Aggregation for Privacy-Preserving
//     Machine Learning." CCS 2017.

use super::super::secure_aggregation::{dequantize_gradient, quantize_gradient};
use super::pairwise_masking::{
    compute_client_mask, fresh_round_seed, signed_pairwise_mask, ClientKeyPair, ClientPublicKey,
};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;

/// Default width, in bits, of the additive group used for masking.
pub const DEFAULT_MODULUS_BITS: u8 = 31;

/// Narrowest and widest usable group widths. The lower bound keeps the group
/// large enough for any useful quantisation scale; the upper bound keeps
/// `modulus` and every intermediate sum inside `i64`.
pub const MIN_MODULUS_BITS: u8 = 8;
/// See [`MIN_MODULUS_BITS`].
pub const MAX_MODULUS_BITS: u8 = 62;

/// Secure aggregation configuration
#[derive(Debug, Clone)]
pub struct SecureAggregationConfig {
    /// Whether the protocol is active. Protocol operations error when this is
    /// false rather than quietly aggregating in the clear.
    pub enabled: bool,

    /// Minimum number of *received* submissions required to aggregate.
    pub min_clients: usize,

    /// Maximum number of dropouts a round may tolerate. A round with more
    /// dropouts than this is refused.
    pub max_dropouts: usize,

    /// Dimension of the update vectors being aggregated.
    pub masking_dimension: usize,

    /// How pairwise seeds are established.
    pub seed_sharing: SeedSharingMethod,

    /// Width, in bits, of the additive group `Z_(2^bits)` used for masking.
    /// `None` selects [`DEFAULT_MODULUS_BITS`].
    pub quantization_bits: Option<u8>,

    /// Fixed-point scale: an update coordinate `g` is encoded as
    /// `round(g * quantization_scale)`.
    pub quantization_scale: f64,

    /// Declared per-coordinate bound on client updates. Enforced when a client
    /// masks its update, and used to prove that the cohort's summed
    /// fixed-point value cannot wrap the group.
    pub max_update_magnitude: f64,

    /// Add differential privacy noise to the aggregate. Not implemented here;
    /// setting it is an error (see the module documentation).
    pub aggregate_dp: bool,
}

/// How the pairwise seeds behind the masks are established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedSharingMethod {
    /// Per-round X25519 elliptic-curve Diffie-Hellman between every pair of
    /// participants. The only method implemented here.
    EphemeralDiffieHellman,

    /// Shamir secret sharing of per-client seeds, as in the full Bonawitz
    /// protocol. Not implemented.
    ShamirSecretSharing,

    /// Threshold encryption of per-client seeds. Not implemented.
    ThresholdEncryption,

    /// Distributed key generation. Not implemented.
    DistributedKeyGeneration,
}

/// A client's masked upload.
///
/// `values[k] = (round(update[k] * scale) + mask[k]) mod modulus`, always in
/// `[0, modulus)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskedClientUpdate {
    /// Submitting client.
    pub client_id: String,
    /// Masked, quantised coordinates.
    pub values: Vec<i64>,
}

/// A surviving client's disclosure of the pairwise mask it holds with a client
/// that dropped out.
///
/// Revealing this leaks nothing about the dropped client's update -- it never
/// uploaded one -- but it does let the server cancel the residue the dropout
/// left behind. The corresponding upload from the dropped client is refused
/// afterwards, which is what keeps the disclosure safe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropoutDisclosure {
    /// Surviving client making the disclosure.
    pub from_client: String,
    /// Client that dropped out.
    pub dropped_client: String,
    /// The signed pairwise mask `from_client` applied on behalf of
    /// `dropped_client`, in `[0, modulus)`.
    pub signed_mask: Vec<i64>,
}

/// The public parameters of one aggregation round, published by the server.
///
/// Everything in here is public. The masks are protected by the clients'
/// X25519 secret keys, which never appear in a plan.
#[derive(Debug, Clone)]
pub struct SecureAggregationPlan {
    /// Public per-round salt mixed into every pairwise seed.
    pub round_seed: u64,

    /// Participating client identifiers, sorted.
    pub participating_clients: Vec<String>,

    /// Public-key directory for the round.
    pub public_keys: BTreeMap<String, ClientPublicKey>,

    /// Minimum number of submissions the server will aggregate.
    pub min_threshold: usize,

    /// Always true: a plan is only issued when masking is active.
    pub masking_enabled: bool,

    /// Update dimension.
    pub masking_dimension: usize,

    /// Modulus of the additive group.
    pub modulus: i64,

    /// Fixed-point scale.
    pub quantization_scale: f64,

    /// Per-coordinate bound each client must respect.
    pub max_update_magnitude: f64,
}

/// Server-side secure aggregation state.
///
/// Holds no secret key material -- only the published public keys, the
/// received masked uploads, the set of clients known to have dropped, and the
/// disclosures that cancel their residue. There is deliberately no field from
/// which an individual client's update could be recovered.
pub struct SecureAggregator<T: Float + Debug + Send + Sync + 'static> {
    config: SecureAggregationConfig,
    modulus: i64,
    registered_keys: BTreeMap<String, ClientPublicKey>,
    current_plan: Option<SecureAggregationPlan>,
    received: BTreeMap<String, MaskedClientUpdate>,
    dropped: HashSet<String>,
    disclosures: Vec<DropoutDisclosure>,
    rounds_prepared: u64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for SecureAggregator<T> {
    /// Summarises the round rather than dumping every masked coordinate.
    ///
    /// Every field of this type is server-side and public by construction --
    /// there is no key material to redact, which is the whole point.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SecureAggregator")
            .field("modulus", &self.modulus)
            .field("registered_clients", &self.registered_keys.len())
            .field("round_open", &self.current_plan.is_some())
            .field("received", &self.received.len())
            .field("dropped", &self.dropped.len())
            .field("disclosures", &self.disclosures.len())
            .field("rounds_prepared", &self.rounds_prepared)
            .finish()
    }
}

/// Mask a client's update for upload.
///
/// Performed entirely on the client, from its own key pair and the round's
/// public plan. Enforces the plan's `max_update_magnitude` so that the
/// server's no-wraparound argument actually holds instead of being assumed.
pub fn mask_client_update<T: Float + Debug + Send + Sync + 'static>(
    client_id: &str,
    keys: &ClientKeyPair,
    update: &Array1<T>,
    plan: &SecureAggregationPlan,
) -> Result<MaskedClientUpdate> {
    if update.len() != plan.masking_dimension {
        return Err(OptimError::DimensionMismatch(format!(
            "client {client_id} supplied {} coordinates but the round dimension is {}",
            update.len(),
            plan.masking_dimension
        )));
    }

    let mut as_f64 = Vec::with_capacity(update.len());
    for value in update.iter() {
        let converted = value.to_f64().ok_or_else(|| {
            OptimError::ComputationError(format!(
                "client {client_id} supplied a value that is not representable as f64: {value:?}"
            ))
        })?;
        if !converted.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "client {client_id} supplied a non-finite update coordinate ({converted})"
            )));
        }
        if converted.abs() > plan.max_update_magnitude {
            return Err(OptimError::InvalidParameter(format!(
                "client {client_id} supplied a coordinate of magnitude {} which exceeds the \
                 round's declared bound {}; clip the update before masking, otherwise the \
                 cohort's fixed-point sum could wrap the modulus and the aggregate would be \
                 silently wrong",
                converted.abs(),
                plan.max_update_magnitude
            )));
        }
        as_f64.push(converted);
    }

    let quantised =
        quantize_gradient(&Array1::from(as_f64), plan.quantization_scale, plan.modulus)?;
    let mask = compute_client_mask(
        client_id,
        keys,
        &plan.public_keys,
        plan.round_seed,
        plan.masking_dimension,
        plan.modulus,
    )?;

    if quantised.len() != mask.len() {
        return Err(OptimError::DimensionMismatch(format!(
            "quantised update has {} coordinates but the derived mask has {}; zipping them \
             would silently truncate the upload",
            quantised.len(),
            mask.len()
        )));
    }
    let values = quantised
        .iter()
        .zip(mask.iter())
        .map(|(quantum, mask_value)| (*quantum + *mask_value).rem_euclid(plan.modulus))
        .collect();
    Ok(MaskedClientUpdate {
        client_id: client_id.to_string(),
        values,
    })
}

/// Produce the disclosures a surviving client owes for a set of dropped
/// clients.
///
/// The client reveals only the signed pairwise masks it holds with the dropped
/// peers -- never its secret key, and never anything about its own update.
pub fn disclose_dropout_masks(
    client_id: &str,
    keys: &ClientKeyPair,
    dropped_clients: &[String],
    plan: &SecureAggregationPlan,
) -> Result<Vec<DropoutDisclosure>> {
    let mut disclosures = Vec::with_capacity(dropped_clients.len());
    for dropped in dropped_clients.iter() {
        if dropped == client_id {
            return Err(OptimError::InvalidParameter(format!(
                "client {client_id} cannot disclose a pairwise mask with itself"
            )));
        }
        let dropped_key = plan.public_keys.get(dropped).ok_or_else(|| {
            OptimError::InvalidParameter(format!(
                "dropped client {dropped} is not in the round's public-key directory"
            ))
        })?;
        let signed_mask = signed_pairwise_mask(
            client_id,
            keys,
            dropped,
            dropped_key,
            plan.round_seed,
            plan.masking_dimension,
            plan.modulus,
        )?;
        disclosures.push(DropoutDisclosure {
            from_client: client_id.to_string(),
            dropped_client: dropped.clone(),
            signed_mask,
        });
    }
    Ok(disclosures)
}

impl<T: Float + Debug + Send + Sync + 'static> SecureAggregator<T> {
    /// Build an aggregator, rejecting configurations whose guarantees cannot
    /// be delivered.
    pub fn new(config: SecureAggregationConfig) -> Result<Self> {
        let modulus = validate_config(&config)?;
        Ok(Self {
            config,
            modulus,
            registered_keys: BTreeMap::new(),
            current_plan: None,
            received: BTreeMap::new(),
            dropped: HashSet::new(),
            disclosures: Vec::new(),
            rounds_prepared: 0,
            _marker: std::marker::PhantomData,
        })
    }

    /// Publish a client's per-round public key.
    ///
    /// Re-registering the same client replaces its key, which is what happens
    /// when a client rejoins with a fresh key pair.
    pub fn register_client_key(
        &mut self,
        client_id: &str,
        public_key: ClientPublicKey,
    ) -> Result<()> {
        if client_id.is_empty() {
            return Err(OptimError::InvalidParameter(
                "client identifier must not be empty".to_string(),
            ));
        }
        if self
            .registered_keys
            .iter()
            .any(|(id, key)| id != client_id && *key == public_key)
        {
            return Err(OptimError::InvalidParameter(format!(
                "the public key offered by client {client_id} is already registered to another \
                 client; duplicate keys would make the pairwise masks collide"
            )));
        }
        self.registered_keys
            .insert(client_id.to_string(), public_key);
        Ok(())
    }

    /// Open a round for `selected_clients` and publish the resulting plan.
    ///
    /// Clears any state from a previous round. Every selected client must have
    /// registered a public key, and the cohort must be able to satisfy both
    /// `min_clients` and the no-wraparound bound.
    pub fn prepare_round(&mut self, selected_clients: &[String]) -> Result<SecureAggregationPlan> {
        if !self.config.enabled {
            return Err(OptimError::InvalidConfig(
                "secure aggregation is disabled in this configuration; enable it or aggregate \
                 updates directly instead of routing them through a protocol that is switched off"
                    .to_string(),
            ));
        }

        let mut sorted: Vec<String> = selected_clients.to_vec();
        sorted.sort();
        sorted.dedup();
        if sorted.len() < 2 {
            return Err(OptimError::InvalidConfig(format!(
                "pairwise masking needs at least 2 distinct clients, got {}",
                sorted.len()
            )));
        }
        if sorted.len() < self.config.min_clients {
            return Err(OptimError::InvalidConfig(format!(
                "{} clients were selected but min_clients is {}",
                sorted.len(),
                self.config.min_clients
            )));
        }

        let mut public_keys = BTreeMap::new();
        for client_id in sorted.iter() {
            let key = self.registered_keys.get(client_id).ok_or_else(|| {
                OptimError::InvalidState(format!(
                    "client {client_id} has not registered a public key for this round; without \
                     it no pairwise mask can be agreed"
                ))
            })?;
            public_keys.insert(client_id.clone(), *key);
        }

        // No-wraparound bound: the cohort's summed fixed-point magnitude must
        // stay inside half the group, otherwise a correct-looking but wrong
        // aggregate could come back.
        let worst_case =
            self.config.quantization_scale * self.config.max_update_magnitude * sorted.len() as f64;
        let half_modulus = (self.modulus / 2) as f64;
        if worst_case >= half_modulus {
            return Err(OptimError::InvalidConfig(format!(
                "a cohort of {} clients with quantization_scale {} and max_update_magnitude {} \
                 could sum to {worst_case}, which does not fit in half the modulus \
                 ({half_modulus}); widen quantization_bits, lower the scale, or shrink the cohort",
                sorted.len(),
                self.config.quantization_scale,
                self.config.max_update_magnitude
            )));
        }

        self.received.clear();
        self.dropped.clear();
        self.disclosures.clear();
        self.rounds_prepared = self.rounds_prepared.saturating_add(1);

        let plan = SecureAggregationPlan {
            round_seed: fresh_round_seed(),
            participating_clients: sorted,
            public_keys,
            min_threshold: self.config.min_clients,
            masking_enabled: true,
            masking_dimension: self.config.masking_dimension,
            modulus: self.modulus,
            quantization_scale: self.config.quantization_scale,
            max_update_magnitude: self.config.max_update_magnitude,
        };
        self.current_plan = Some(plan.clone());
        Ok(plan)
    }

    /// Accept a masked upload.
    ///
    /// Refuses uploads from clients already marked as dropped: their pairwise
    /// masks may already have been disclosed, so accepting a late upload would
    /// hand the server everything it needs to unmask that single client.
    pub fn receive(&mut self, submission: MaskedClientUpdate) -> Result<()> {
        let plan = self.plan()?;
        if !plan.participating_clients.contains(&submission.client_id) {
            return Err(OptimError::InvalidParameter(format!(
                "client {} is not part of the current round",
                submission.client_id
            )));
        }
        if submission.values.len() != plan.masking_dimension {
            return Err(OptimError::DimensionMismatch(format!(
                "submission from client {} has {} values, expected {}",
                submission.client_id,
                submission.values.len(),
                plan.masking_dimension
            )));
        }
        for &value in submission.values.iter() {
            if value < 0 || value >= self.modulus {
                return Err(OptimError::InvalidParameter(format!(
                    "submission value {value} from client {} lies outside [0, {})",
                    submission.client_id, self.modulus
                )));
            }
        }
        if self.dropped.contains(&submission.client_id) {
            return Err(OptimError::InvalidState(format!(
                "client {} was already marked as dropped and its pairwise masks may have been \
                 disclosed; accepting this upload would let the server unmask it individually",
                submission.client_id
            )));
        }

        self.received
            .insert(submission.client_id.clone(), submission);
        Ok(())
    }

    /// Mark a participating client as dropped.
    ///
    /// Refuses to mark a client that has already submitted, since its mask
    /// must stay secret for the aggregate to reveal only the sum.
    pub fn mark_dropped(&mut self, client_id: &str) -> Result<()> {
        let plan = self.plan()?;
        if !plan.participating_clients.iter().any(|id| id == client_id) {
            return Err(OptimError::InvalidParameter(format!(
                "client {client_id} is not part of the current round"
            )));
        }
        if self.received.contains_key(client_id) {
            return Err(OptimError::InvalidParameter(format!(
                "client {client_id} has already submitted; marking it dropped and collecting \
                 disclosures would expose its individual update"
            )));
        }
        self.dropped.insert(client_id.to_string());
        Ok(())
    }

    /// Accept a surviving client's disclosure for a dropped peer.
    pub fn receive_dropout_disclosure(&mut self, disclosure: DropoutDisclosure) -> Result<()> {
        let plan = self.plan()?;
        if !plan.participating_clients.contains(&disclosure.from_client) {
            return Err(OptimError::InvalidParameter(format!(
                "client {} is not part of the current round",
                disclosure.from_client
            )));
        }
        if !self.dropped.contains(&disclosure.dropped_client) {
            return Err(OptimError::InvalidParameter(format!(
                "client {} is not marked as dropped, so no disclosure is owed for it",
                disclosure.dropped_client
            )));
        }
        if disclosure.from_client == disclosure.dropped_client {
            return Err(OptimError::InvalidParameter(
                "a client cannot disclose a pairwise mask with itself".to_string(),
            ));
        }
        if disclosure.signed_mask.len() != plan.masking_dimension {
            return Err(OptimError::DimensionMismatch(format!(
                "disclosure from client {} has {} values, expected {}",
                disclosure.from_client,
                disclosure.signed_mask.len(),
                plan.masking_dimension
            )));
        }
        if self.disclosures.iter().any(|existing| {
            existing.from_client == disclosure.from_client
                && existing.dropped_client == disclosure.dropped_client
        }) {
            return Err(OptimError::InvalidState(format!(
                "client {} has already disclosed its mask with {}",
                disclosure.from_client, disclosure.dropped_client
            )));
        }
        self.disclosures.push(disclosure);
        Ok(())
    }

    /// Sum the received uploads modulo the group order, cancel the residue
    /// left by dropped clients, and return the dequantised **sum** of the
    /// received updates.
    ///
    /// The server never sees a summand. Errors -- rather than returning a
    /// wrong-but-plausible vector -- when the round is incomplete, when too
    /// many clients dropped, or when a dropout's disclosures are missing.
    pub fn aggregate(&self) -> Result<Array1<T>> {
        let plan = self.plan()?;
        if self.received.len() < self.config.min_clients {
            return Err(OptimError::InvalidConfig(format!(
                "only {} submissions received but min_clients is {}",
                self.received.len(),
                self.config.min_clients
            )));
        }
        if self.dropped.len() > self.config.max_dropouts {
            return Err(OptimError::InvalidConfig(format!(
                "{} clients dropped out but max_dropouts is {}",
                self.dropped.len(),
                self.config.max_dropouts
            )));
        }
        let accounted = self.received.len() + self.dropped.len();
        if accounted != plan.participating_clients.len() {
            let missing: Vec<&str> = plan
                .participating_clients
                .iter()
                .map(|id| id.as_str())
                .filter(|id| !self.received.contains_key(*id) && !self.dropped.contains(*id))
                .collect();
            return Err(OptimError::InvalidState(format!(
                "the round is incomplete: {} of {} clients neither submitted nor were marked \
                 dropped ({missing:?}). Their pairwise masks are still in the sum, so the \
                 aggregate would be meaningless",
                missing.len(),
                plan.participating_clients.len()
            )));
        }

        // Every surviving client owes one disclosure per dropped client.
        for dropped in self.dropped.iter() {
            for survivor in self.received.keys() {
                let present = self.disclosures.iter().any(|disclosure| {
                    disclosure.from_client == *survivor && disclosure.dropped_client == *dropped
                });
                if !present {
                    return Err(OptimError::InvalidState(format!(
                        "client {survivor} has not disclosed its pairwise mask with the dropped \
                         client {dropped}; without every disclosure the dropout residue cannot \
                         be cancelled"
                    )));
                }
            }
        }

        let dimension = plan.masking_dimension;
        let mut accumulator = vec![0_i64; dimension];
        for submission in self.received.values() {
            for (slot, value) in accumulator.iter_mut().zip(submission.values.iter()) {
                *slot = (*slot + *value).rem_euclid(self.modulus);
            }
        }
        // Each survivor's upload still carries its signed pairwise mask with
        // every dropped client; subtract exactly those disclosed terms.
        for disclosure in self.disclosures.iter() {
            if !self.received.contains_key(&disclosure.from_client) {
                continue;
            }
            for (slot, value) in accumulator.iter_mut().zip(disclosure.signed_mask.iter()) {
                *slot = (*slot - *value).rem_euclid(self.modulus);
            }
        }

        let real_sum = dequantize_gradient(&accumulator, plan.quantization_scale, self.modulus);
        if real_sum.len() != dimension {
            return Err(OptimError::ComputationError(format!(
                "dequantisation returned {} values for a {dimension}-dimensional round",
                real_sum.len()
            )));
        }
        let mut output = Array1::zeros(dimension);
        for (slot, value) in output.iter_mut().zip(real_sum.iter()) {
            *slot = T::from(*value).ok_or_else(|| {
                OptimError::ComputationError(format!(
                    "aggregated value {value} is not representable in the target float type"
                ))
            })?;
        }
        Ok(output)
    }

    /// The dequantised **mean** of the received updates.
    pub fn aggregate_mean(&self) -> Result<Array1<T>> {
        let sum = self.aggregate()?;
        let count = T::from(self.received.len() as f64).ok_or_else(|| {
            OptimError::ComputationError("submission count is not representable".to_string())
        })?;
        Ok(sum.mapv(|value| value / count))
    }

    /// Sum of the received uploads modulo the group order, before
    /// dequantisation. Exposed so that callers (and tests) can verify the
    /// exact integer identity the protocol guarantees.
    pub fn masked_sum(&self) -> Result<Vec<i64>> {
        let plan = self.plan()?;
        let mut accumulator = vec![0_i64; plan.masking_dimension];
        for submission in self.received.values() {
            for (slot, value) in accumulator.iter_mut().zip(submission.values.iter()) {
                *slot = (*slot + *value).rem_euclid(self.modulus);
            }
        }
        Ok(accumulator)
    }

    /// Discard the current round's state, keeping registered keys.
    pub fn reset_round(&mut self) {
        self.current_plan = None;
        self.received.clear();
        self.dropped.clear();
        self.disclosures.clear();
    }

    /// Get current configuration
    pub fn config(&self) -> &SecureAggregationConfig {
        &self.config
    }

    /// Modulus of the additive group in use.
    pub fn modulus(&self) -> i64 {
        self.modulus
    }

    /// Number of received submissions.
    pub fn received_count(&self) -> usize {
        self.received.len()
    }

    /// Number of clients marked as dropped.
    pub fn dropped_count(&self) -> usize {
        self.dropped.len()
    }

    /// Number of disclosures collected.
    pub fn disclosure_count(&self) -> usize {
        self.disclosures.len()
    }

    /// Rounds opened by this aggregator.
    pub fn rounds_prepared(&self) -> u64 {
        self.rounds_prepared
    }

    /// The minimum number of submissions required to aggregate.
    pub fn aggregation_threshold(&self) -> usize {
        self.config.min_clients
    }

    /// Check if secure aggregation is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// The current round's plan, if a round is open.
    pub fn current_plan(&self) -> Option<&SecureAggregationPlan> {
        self.current_plan.as_ref()
    }

    fn plan(&self) -> Result<&SecureAggregationPlan> {
        self.current_plan.as_ref().ok_or_else(|| {
            OptimError::InvalidState(
                "no aggregation round is open; call prepare_round first".to_string(),
            )
        })
    }
}

/// Validate a configuration and return the modulus it selects.
fn validate_config(config: &SecureAggregationConfig) -> Result<i64> {
    if config.seed_sharing != SeedSharingMethod::EphemeralDiffieHellman {
        return Err(OptimError::InvalidConfig(format!(
            "seed sharing method {:?} is not implemented; only \
             SeedSharingMethod::EphemeralDiffieHellman (per-round X25519 key agreement between \
             every pair of clients) is available here",
            config.seed_sharing
        )));
    }
    if config.aggregate_dp {
        return Err(OptimError::InvalidConfig(
            "aggregate_dp is not implemented by this module. Secure aggregation hides individual \
             updates from the server; it adds no differential privacy noise and claiming \
             otherwise would overstate the guarantee. Compose this with \
             crate::privacy::dp_sgd or crate::privacy::noise_mechanisms, which account the \
             epsilon they spend."
                .to_string(),
        ));
    }
    if config.masking_dimension == 0 {
        return Err(OptimError::InvalidConfig(
            "masking_dimension must be greater than zero".to_string(),
        ));
    }
    if config.min_clients < 2 {
        return Err(OptimError::InvalidConfig(format!(
            "min_clients must be at least 2 for pairwise masking to hide anything, got {}",
            config.min_clients
        )));
    }
    if !config.quantization_scale.is_finite() || config.quantization_scale <= 0.0 {
        return Err(OptimError::InvalidConfig(format!(
            "quantization_scale must be positive and finite, got {}",
            config.quantization_scale
        )));
    }
    if !config.max_update_magnitude.is_finite() || config.max_update_magnitude <= 0.0 {
        return Err(OptimError::InvalidConfig(format!(
            "max_update_magnitude must be positive and finite, got {}",
            config.max_update_magnitude
        )));
    }

    let bits = config.quantization_bits.unwrap_or(DEFAULT_MODULUS_BITS);
    if !(MIN_MODULUS_BITS..=MAX_MODULUS_BITS).contains(&bits) {
        return Err(OptimError::InvalidConfig(format!(
            "quantization_bits must be in [{MIN_MODULUS_BITS}, {MAX_MODULUS_BITS}], got {bits}"
        )));
    }
    let modulus = 1_i64 << bits;

    // A round of exactly `min_clients` must at least be representable.
    let minimum_capacity =
        config.quantization_scale * config.max_update_magnitude * config.min_clients as f64;
    if minimum_capacity >= (modulus / 2) as f64 {
        return Err(OptimError::InvalidConfig(format!(
            "a cohort of min_clients = {} at quantization_scale {} and max_update_magnitude {} \
             needs more than half of the {bits}-bit group; widen quantization_bits or lower the \
             scale",
            config.min_clients, config.quantization_scale, config.max_update_magnitude
        )));
    }
    Ok(modulus)
}

impl Default for SecureAggregationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_clients: 10,
            max_dropouts: 5,
            masking_dimension: 1000,
            seed_sharing: SeedSharingMethod::EphemeralDiffieHellman,
            quantization_bits: None,
            // 2^31 / 2 = 1.07e9, so a 10-client cohort of unit-bounded
            // updates uses at most 1e5 of the available 1.07e9 headroom.
            quantization_scale: 1.0e4,
            max_update_magnitude: 1.0,
            aggregate_dp: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 32;

    fn config(min_clients: usize) -> SecureAggregationConfig {
        SecureAggregationConfig {
            min_clients,
            max_dropouts: 2,
            masking_dimension: DIM,
            quantization_scale: 1.0e6,
            max_update_magnitude: 10.0,
            quantization_bits: Some(48),
            ..SecureAggregationConfig::default()
        }
    }

    /// A deterministic but non-trivial update for client `index`.
    fn update_for(index: usize) -> Array1<f64> {
        Array1::from_shape_fn(DIM, |k| {
            ((index + 1) as f64) * 0.125 - (k as f64) * 0.03125 + 0.5
        })
    }

    struct Cohort {
        aggregator: SecureAggregator<f64>,
        plan: SecureAggregationPlan,
        ids: Vec<String>,
        keys: Vec<ClientKeyPair>,
        updates: Vec<Array1<f64>>,
    }

    fn cohort(size: usize, min_clients: usize) -> Cohort {
        let mut aggregator =
            SecureAggregator::<f64>::new(config(min_clients)).expect("valid config");
        let ids: Vec<String> = (0..size).map(|i| format!("client{i:02}")).collect();
        let keys: Vec<ClientKeyPair> = (0..size).map(|_| ClientKeyPair::generate()).collect();
        for (id, key_pair) in ids.iter().zip(keys.iter()) {
            aggregator
                .register_client_key(id, key_pair.public_key())
                .expect("register");
        }
        let plan = aggregator.prepare_round(&ids).expect("plan");
        let updates: Vec<Array1<f64>> = (0..size).map(update_for).collect();
        Cohort {
            aggregator,
            plan,
            ids,
            keys,
            updates,
        }
    }

    fn submit_all(cohort: &mut Cohort) {
        for index in 0..cohort.ids.len() {
            let masked = mask_client_update(
                &cohort.ids[index],
                &cohort.keys[index],
                &cohort.updates[index],
                &cohort.plan,
            )
            .expect("mask");
            cohort.aggregator.receive(masked).expect("receive");
        }
    }

    /// The exact fixed-point sum the protocol must reproduce.
    fn expected_quantised_sum(cohort: &Cohort) -> Vec<i64> {
        let mut total = vec![0_i64; DIM];
        for update in cohort.updates.iter() {
            let quantised =
                quantize_gradient(update, cohort.plan.quantization_scale, cohort.plan.modulus)
                    .expect("quantise");
            for (slot, value) in total.iter_mut().zip(quantised.iter()) {
                *slot = (*slot + *value).rem_euclid(cohort.plan.modulus);
            }
        }
        total
    }

    // ---------------------------------------------------------------------
    // F25: the masks cancel EXACTLY.
    // ---------------------------------------------------------------------

    #[test]
    fn masked_sum_equals_the_sum_of_quantised_inputs_exactly() {
        let mut cohort = cohort(8, 4);
        submit_all(&mut cohort);

        let masked_sum = cohort.aggregator.masked_sum().expect("masked sum");
        // Exact integer identity: this is the property the protocol
        // guarantees, and the one the previous implementation violated.
        assert_eq!(masked_sum, expected_quantised_sum(&cohort));
    }

    #[test]
    fn dequantised_aggregate_matches_the_true_float_sum() {
        let mut cohort = cohort(8, 4);
        submit_all(&mut cohort);

        let aggregate = cohort.aggregator.aggregate().expect("aggregate");
        let mut truth = Array1::<f64>::zeros(DIM);
        for update in cohort.updates.iter() {
            truth += update;
        }
        assert_eq!(aggregate.len(), DIM);
        for (got, want) in aggregate.iter().zip(truth.iter()) {
            // The only error is fixed-point rounding: at most n/2 units of
            // 1/scale, i.e. 8 / (2 * 1e6).
            assert!(
                (got - want).abs() < 1.0e-5,
                "aggregate {got} differs from the true sum {want}"
            );
        }

        let mean = cohort.aggregator.aggregate_mean().expect("mean");
        for (got, want) in mean.iter().zip(truth.iter()) {
            assert!((got - want / 8.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn the_aggregate_is_exact_for_representable_inputs() {
        // Multiples of 1/1024 are exactly representable at scale 1e6 only up
        // to rounding, so use integers, which are exact.
        let mut cohort = cohort(4, 4);
        cohort.updates = (0..4)
            .map(|i| Array1::from_shape_fn(DIM, |k| ((i * DIM + k) % 7) as f64 - 3.0))
            .collect();
        submit_all(&mut cohort);

        let aggregate = cohort.aggregator.aggregate().expect("aggregate");
        let mut truth = Array1::<f64>::zeros(DIM);
        for update in cohort.updates.iter() {
            truth += update;
        }
        for (got, want) in aggregate.iter().zip(truth.iter()) {
            assert_eq!(got, want, "integers must round-trip exactly");
        }
    }

    // ---------------------------------------------------------------------
    // F24: an individual upload reveals nothing, and the server holds no key.
    // ---------------------------------------------------------------------

    #[test]
    fn a_single_masked_upload_is_not_its_true_input() {
        let cohort = cohort(6, 4);
        let masked = mask_client_update(
            &cohort.ids[0],
            &cohort.keys[0],
            &cohort.updates[0],
            &cohort.plan,
        )
        .expect("mask");
        let quantised = quantize_gradient(
            &cohort.updates[0],
            cohort.plan.quantization_scale,
            cohort.plan.modulus,
        )
        .expect("quantise");

        assert_eq!(masked.values.len(), DIM);
        let coinciding = masked
            .values
            .iter()
            .zip(quantised.iter())
            .filter(|(a, b)| a == b)
            .count();
        assert_eq!(
            coinciding, 0,
            "{coinciding} of {DIM} coordinates were uploaded unmasked"
        );

        // The upload is spread over the whole group, not jittered around the
        // input. The previous implementation added a mask in [-1, 1] to the
        // update, so the "masked" value stayed within 1.0 of the truth --
        // which is what this assertion rules out.
        let minimum = masked.values.iter().copied().min().expect("non-empty");
        let maximum = masked.values.iter().copied().max().expect("non-empty");
        let modulus = cohort.plan.modulus;
        assert!(
            maximum - minimum > modulus / 2,
            "masked values span [{minimum}, {maximum}], which is not spread over [0, {modulus})"
        );
        // And the server-side state that observes the upload carries no key
        // material at all, so there is no server-side path back to the input.
        let rendered = format!("{:?}", cohort.aggregator);
        assert!(rendered.starts_with("SecureAggregator"));
        assert!(
            !rendered.contains("secret"),
            "server state mentions a secret"
        );
        assert!(!rendered.contains("mask"), "server state mentions a mask");
    }

    #[test]
    fn the_server_cannot_reproduce_a_clients_mask_from_the_plan() {
        let cohort = cohort(4, 4);
        // Everything the server publishes and observes.
        let plan = cohort.plan.clone();
        assert!(plan.public_keys.len() == 4 && plan.round_seed != 0);

        // A server holding its own key pair and the whole public directory
        // cannot derive the mask client00 shares with client01.
        let server_keys = ClientKeyPair::generate();
        let truth = cohort.keys[0]
            .shared_seed_with(&cohort.keys[1].public_key(), plan.round_seed)
            .expect("real seed");
        let forged = server_keys
            .shared_seed_with(&cohort.keys[1].public_key(), plan.round_seed)
            .expect("server seed");
        assert_ne!(truth, forged);

        // And the aggregator itself refuses to produce anything from a single
        // upload: with one submission the round is incomplete.
        let mut aggregator = cohort.aggregator;
        let masked = mask_client_update(&cohort.ids[0], &cohort.keys[0], &cohort.updates[0], &plan)
            .expect("mask");
        aggregator.receive(masked).expect("receive");
        let err = aggregator
            .aggregate()
            .expect_err("one upload must not be aggregatable");
        assert!(format!("{err}").contains("min_clients"));
    }

    #[test]
    fn two_clients_learn_only_the_sum() {
        let mut aggregator = SecureAggregator::<f64>::new(SecureAggregationConfig {
            min_clients: 2,
            ..config(2)
        })
        .expect("config");
        let ids = vec!["a".to_string(), "b".to_string()];
        let keys = [ClientKeyPair::generate(), ClientKeyPair::generate()];
        for (id, key_pair) in ids.iter().zip(keys.iter()) {
            aggregator
                .register_client_key(id, key_pair.public_key())
                .expect("register");
        }
        let plan = aggregator.prepare_round(&ids).expect("plan");

        let first = Array1::from_shape_fn(DIM, |k| 1.0 + k as f64 * 0.001);
        let second = Array1::from_shape_fn(DIM, |k| -0.5 + k as f64 * 0.002);
        aggregator
            .receive(mask_client_update("a", &keys[0], &first, &plan).expect("mask"))
            .expect("receive");
        aggregator
            .receive(mask_client_update("b", &keys[1], &second, &plan).expect("mask"))
            .expect("receive");

        let aggregate = aggregator.aggregate().expect("aggregate");
        for k in 0..DIM {
            assert!((aggregate[k] - (first[k] + second[k])).abs() < 1e-5);
            // The sum is not either summand.
            assert!((aggregate[k] - first[k]).abs() > 1e-3);
            assert!((aggregate[k] - second[k]).abs() > 1e-3);
        }
    }

    // ---------------------------------------------------------------------
    // Dropout handling: exact after disclosure, honest error without it.
    // ---------------------------------------------------------------------

    #[test]
    fn a_dropout_without_disclosures_is_an_error_not_a_wrong_answer() {
        let mut cohort = cohort(6, 4);
        let dropped = cohort.ids[5].clone();
        cohort.aggregator.mark_dropped(&dropped).expect("drop");
        for index in 0..5 {
            let masked = mask_client_update(
                &cohort.ids[index],
                &cohort.keys[index],
                &cohort.updates[index],
                &cohort.plan,
            )
            .expect("mask");
            cohort.aggregator.receive(masked).expect("receive");
        }
        let err = cohort
            .aggregator
            .aggregate()
            .expect_err("residue cannot be cancelled without disclosures");
        assert!(format!("{err}").contains("has not disclosed its pairwise mask"));
    }

    #[test]
    fn a_dropout_is_recovered_exactly_once_every_survivor_discloses() {
        let mut cohort = cohort(6, 4);
        let dropped_index = 5;
        let dropped = cohort.ids[dropped_index].clone();
        cohort.aggregator.mark_dropped(&dropped).expect("drop");

        for index in 0..dropped_index {
            let masked = mask_client_update(
                &cohort.ids[index],
                &cohort.keys[index],
                &cohort.updates[index],
                &cohort.plan,
            )
            .expect("mask");
            cohort.aggregator.receive(masked).expect("receive");
            for disclosure in disclose_dropout_masks(
                &cohort.ids[index],
                &cohort.keys[index],
                std::slice::from_ref(&dropped),
                &cohort.plan,
            )
            .expect("disclose")
            {
                cohort
                    .aggregator
                    .receive_dropout_disclosure(disclosure)
                    .expect("accept disclosure");
            }
        }

        let aggregate = cohort.aggregator.aggregate().expect("aggregate");
        let mut truth = Array1::<f64>::zeros(DIM);
        for update in cohort.updates[..dropped_index].iter() {
            truth += update;
        }
        for (got, want) in aggregate.iter().zip(truth.iter()) {
            assert!(
                (got - want).abs() < 1.0e-5,
                "dropout recovery gave {got}, expected {want}"
            );
        }
        assert_eq!(cohort.aggregator.dropped_count(), 1);
        assert_eq!(cohort.aggregator.disclosure_count(), dropped_index);
    }

    #[test]
    fn a_dropped_client_cannot_upload_afterwards() {
        let mut cohort = cohort(6, 4);
        let dropped = cohort.ids[5].clone();
        cohort.aggregator.mark_dropped(&dropped).expect("drop");
        let late = mask_client_update(
            &cohort.ids[5],
            &cohort.keys[5],
            &cohort.updates[5],
            &cohort.plan,
        )
        .expect("mask");
        let err = cohort
            .aggregator
            .receive(late)
            .expect_err("late upload must be refused");
        assert!(format!("{err}").contains("already marked as dropped"));
    }

    #[test]
    fn a_client_that_submitted_cannot_be_marked_dropped() {
        let mut cohort = cohort(6, 4);
        let masked = mask_client_update(
            &cohort.ids[0],
            &cohort.keys[0],
            &cohort.updates[0],
            &cohort.plan,
        )
        .expect("mask");
        cohort.aggregator.receive(masked).expect("receive");
        let id = cohort.ids[0].clone();
        let err = cohort
            .aggregator
            .mark_dropped(&id)
            .expect_err("must be refused");
        assert!(format!("{err}").contains("has already submitted"));
    }

    #[test]
    fn too_many_dropouts_is_refused() {
        let mut cohort = cohort(8, 4);
        for index in 5..8 {
            let id = cohort.ids[index].clone();
            cohort.aggregator.mark_dropped(&id).expect("drop");
        }
        for index in 0..5 {
            let masked = mask_client_update(
                &cohort.ids[index],
                &cohort.keys[index],
                &cohort.updates[index],
                &cohort.plan,
            )
            .expect("mask");
            cohort.aggregator.receive(masked).expect("receive");
        }
        let err = cohort
            .aggregator
            .aggregate()
            .expect_err("3 dropouts exceeds max_dropouts = 2");
        assert!(format!("{err}").contains("max_dropouts"));
    }

    #[test]
    fn an_incomplete_round_is_refused() {
        let mut cohort = cohort(8, 4);
        for index in 0..5 {
            let masked = mask_client_update(
                &cohort.ids[index],
                &cohort.keys[index],
                &cohort.updates[index],
                &cohort.plan,
            )
            .expect("mask");
            cohort.aggregator.receive(masked).expect("receive");
        }
        let err = cohort
            .aggregator
            .aggregate()
            .expect_err("3 clients are unaccounted for");
        assert!(format!("{err}").contains("round is incomplete"));
    }

    // ---------------------------------------------------------------------
    // Configuration and plan validation.
    // ---------------------------------------------------------------------

    #[test]
    fn test_secure_aggregation_config() {
        let config = SecureAggregationConfig {
            enabled: true,
            min_clients: 5,
            max_dropouts: 2,
            masking_dimension: 100,
            seed_sharing: SeedSharingMethod::EphemeralDiffieHellman,
            quantization_bits: Some(40),
            quantization_scale: 1.0e6,
            max_update_magnitude: 1.0,
            aggregate_dp: false,
        };
        assert!(config.enabled);
        assert_eq!(config.min_clients, 5);
        assert_eq!(config.max_dropouts, 2);
        assert!(SecureAggregator::<f64>::new(config).is_ok());
    }

    #[test]
    fn test_secure_aggregator_creation() {
        let config = SecureAggregationConfig::default();
        let aggregator = SecureAggregator::<f64>::new(config.clone()).expect("default config");
        assert_eq!(aggregator.aggregation_threshold(), config.min_clients);
        assert!(aggregator.is_enabled());
        assert_eq!(aggregator.modulus(), 1_i64 << DEFAULT_MODULUS_BITS);
    }

    #[test]
    fn unimplemented_seed_sharing_methods_are_refused() {
        for method in [
            SeedSharingMethod::ShamirSecretSharing,
            SeedSharingMethod::ThresholdEncryption,
            SeedSharingMethod::DistributedKeyGeneration,
        ] {
            let err = SecureAggregator::<f64>::new(SecureAggregationConfig {
                seed_sharing: method,
                ..SecureAggregationConfig::default()
            })
            .expect_err("must be refused");
            assert!(format!("{err}").contains("is not implemented"));
        }
    }

    #[test]
    fn aggregate_dp_is_refused_rather_than_silently_ignored() {
        let err = SecureAggregator::<f64>::new(SecureAggregationConfig {
            aggregate_dp: true,
            ..SecureAggregationConfig::default()
        })
        .expect_err("must be refused");
        let message = format!("{err}");
        assert!(message.contains("aggregate_dp is not implemented"));
        assert!(message.contains("dp_sgd"));
    }

    #[test]
    fn wraparound_is_refused_at_configuration_and_at_round_time() {
        // 2^16 / 2 = 32768, far too small for scale 1e6.
        let err = SecureAggregator::<f64>::new(SecureAggregationConfig {
            quantization_bits: Some(16),
            quantization_scale: 1.0e6,
            max_update_magnitude: 1.0,
            min_clients: 2,
            masking_dimension: 4,
            ..SecureAggregationConfig::default()
        })
        .expect_err("group too small");
        assert!(format!("{err}").contains("needs more than half"));

        // A configuration that is fine for min_clients but not for the cohort
        // actually selected: 2^24 / 2 = 8_388_608 units, scale 1e5, bound 1.0
        // => at most 83 clients.
        let mut aggregator = SecureAggregator::<f64>::new(SecureAggregationConfig {
            quantization_bits: Some(24),
            quantization_scale: 1.0e5,
            max_update_magnitude: 1.0,
            min_clients: 2,
            masking_dimension: 4,
            ..SecureAggregationConfig::default()
        })
        .expect("valid for a small cohort");
        let ids: Vec<String> = (0..100).map(|i| format!("c{i:03}")).collect();
        for id in ids.iter() {
            aggregator
                .register_client_key(id, ClientKeyPair::generate().public_key())
                .expect("register");
        }
        let err = aggregator
            .prepare_round(&ids)
            .expect_err("100 clients overflow the group");
        assert!(format!("{err}").contains("does not fit in half the modulus"));
    }

    #[test]
    fn an_update_beyond_the_declared_bound_is_refused() {
        let cohort = cohort(4, 4);
        let mut oversized = cohort.updates[0].clone();
        oversized[0] = cohort.plan.max_update_magnitude * 2.0;
        let err = mask_client_update(&cohort.ids[0], &cohort.keys[0], &oversized, &cohort.plan)
            .expect_err("must be refused");
        assert!(format!("{err}").contains("exceeds the round's declared bound"));

        let mut infinite = cohort.updates[0].clone();
        infinite[1] = f64::INFINITY;
        assert!(
            mask_client_update(&cohort.ids[0], &cohort.keys[0], &infinite, &cohort.plan).is_err()
        );
    }

    #[test]
    fn disabled_secure_aggregation_refuses_to_open_a_round() {
        let mut aggregator = SecureAggregator::<f64>::new(SecureAggregationConfig {
            enabled: false,
            ..SecureAggregationConfig::default()
        })
        .expect("config");
        let ids = vec!["a".to_string(), "b".to_string()];
        for id in ids.iter() {
            aggregator
                .register_client_key(id, ClientKeyPair::generate().public_key())
                .expect("register");
        }
        let err = aggregator
            .prepare_round(&ids)
            .expect_err("disabled protocol");
        assert!(format!("{err}").contains("disabled"));
        assert!(!aggregator.is_enabled());
    }

    #[test]
    fn test_secure_aggregation_plan() {
        let cohort = cohort(4, 4);
        assert_eq!(cohort.plan.participating_clients.len(), 4);
        assert!(cohort.plan.masking_enabled);
        assert_eq!(cohort.plan.public_keys.len(), 4);
        assert_eq!(cohort.plan.masking_dimension, DIM);
        assert_eq!(cohort.aggregator.rounds_prepared(), 1);
    }

    #[test]
    fn round_seeds_are_fresh_and_not_a_counter() {
        let mut aggregator = SecureAggregator::<f64>::new(config(2)).expect("config");
        let ids = vec!["a".to_string(), "b".to_string()];
        for id in ids.iter() {
            aggregator
                .register_client_key(id, ClientKeyPair::generate().public_key())
                .expect("register");
        }
        let mut seeds = std::collections::HashSet::new();
        for round in 1..=5_u64 {
            let plan = aggregator.prepare_round(&ids).expect("plan");
            // The previous implementation returned 1, 2, 3, ... from a
            // mutex-guarded counter.
            assert_ne!(plan.round_seed, round);
            seeds.insert(plan.round_seed);
        }
        assert_eq!(seeds.len(), 5);
        assert_eq!(aggregator.rounds_prepared(), 5);
    }

    #[test]
    fn different_rounds_produce_different_masks_for_the_same_update() {
        let mut cohort = cohort(4, 4);
        let first = mask_client_update(
            &cohort.ids[0],
            &cohort.keys[0],
            &cohort.updates[0],
            &cohort.plan,
        )
        .expect("mask");
        let second_plan = cohort.aggregator.prepare_round(&cohort.ids).expect("plan");
        let second = mask_client_update(
            &cohort.ids[0],
            &cohort.keys[0],
            &cohort.updates[0],
            &second_plan,
        )
        .expect("mask");
        assert_ne!(first.values, second.values);
    }

    #[test]
    fn a_round_cannot_open_without_registered_keys() {
        let mut aggregator = SecureAggregator::<f64>::new(config(2)).expect("config");
        let ids = vec!["a".to_string(), "b".to_string()];
        aggregator
            .register_client_key("a", ClientKeyPair::generate().public_key())
            .expect("register");
        let err = aggregator.prepare_round(&ids).expect_err("b has no key");
        assert!(format!("{err}").contains("has not registered a public key"));
    }

    #[test]
    fn duplicate_public_keys_are_refused() {
        let mut aggregator = SecureAggregator::<f64>::new(config(2)).expect("config");
        let shared = ClientKeyPair::generate().public_key();
        aggregator
            .register_client_key("a", shared)
            .expect("register a");
        let err = aggregator
            .register_client_key("b", shared)
            .expect_err("duplicate key");
        assert!(format!("{err}").contains("already registered to another client"));
    }

    #[test]
    fn submissions_are_validated_against_the_open_round() {
        let mut cohort = cohort(4, 4);
        let stranger = MaskedClientUpdate {
            client_id: "nobody".to_string(),
            values: vec![0; DIM],
        };
        assert!(cohort.aggregator.receive(stranger).is_err());

        let wrong_length = MaskedClientUpdate {
            client_id: cohort.ids[0].clone(),
            values: vec![0; DIM + 1],
        };
        assert!(cohort.aggregator.receive(wrong_length).is_err());

        let out_of_range = MaskedClientUpdate {
            client_id: cohort.ids[0].clone(),
            values: vec![cohort.plan.modulus; DIM],
        };
        let err = cohort
            .aggregator
            .receive(out_of_range)
            .expect_err("out of range");
        assert!(format!("{err}").contains("lies outside"));
    }

    #[test]
    fn operations_before_prepare_round_are_refused() {
        let mut aggregator = SecureAggregator::<f64>::new(config(2)).expect("config");
        let submission = MaskedClientUpdate {
            client_id: "a".to_string(),
            values: vec![0; DIM],
        };
        assert!(aggregator.receive(submission).is_err());
        assert!(aggregator.mark_dropped("a").is_err());
        let err = aggregator.aggregate().expect_err("no round");
        assert!(format!("{err}").contains("no aggregation round is open"));
    }

    #[test]
    fn disclosures_are_validated() {
        let mut cohort = cohort(6, 4);
        let dropped = cohort.ids[5].clone();

        // No disclosure is owed before the client is marked dropped.
        let premature = DropoutDisclosure {
            from_client: cohort.ids[0].clone(),
            dropped_client: dropped.clone(),
            signed_mask: vec![0; DIM],
        };
        let err = cohort
            .aggregator
            .receive_dropout_disclosure(premature)
            .expect_err("not dropped yet");
        assert!(format!("{err}").contains("is not marked as dropped"));

        cohort.aggregator.mark_dropped(&dropped).expect("drop");
        let real = disclose_dropout_masks(
            &cohort.ids[0],
            &cohort.keys[0],
            std::slice::from_ref(&dropped),
            &cohort.plan,
        )
        .expect("disclose")
        .remove(0);
        cohort
            .aggregator
            .receive_dropout_disclosure(real.clone())
            .expect("first");
        let err = cohort
            .aggregator
            .receive_dropout_disclosure(real)
            .expect_err("duplicate");
        assert!(format!("{err}").contains("has already disclosed"));

        // A client cannot disclose a mask with itself.
        assert!(disclose_dropout_masks(
            &dropped,
            &cohort.keys[5],
            std::slice::from_ref(&dropped),
            &cohort.plan
        )
        .is_err());
    }

    #[test]
    fn reset_round_clears_state_but_keeps_keys() {
        let mut cohort = cohort(4, 4);
        submit_all(&mut cohort);
        assert_eq!(cohort.aggregator.received_count(), 4);
        cohort.aggregator.reset_round();
        assert_eq!(cohort.aggregator.received_count(), 0);
        assert!(cohort.aggregator.current_plan().is_none());
        // Keys survive, so a new round can open immediately.
        assert!(cohort.aggregator.prepare_round(&cohort.ids).is_ok());
    }

    #[test]
    fn aggregation_is_independent_of_submission_order() {
        let mut forward = cohort(6, 4);
        // Reuse the *same* plan so the masks match, then submit in reverse.
        submit_all(&mut forward);
        let expected = forward.aggregator.aggregate().expect("aggregate");

        forward.aggregator.reset_round();
        let plan = forward.plan.clone();
        let mut aggregator = forward.aggregator;
        aggregator.current_plan = Some(plan.clone());
        for index in (0..forward.ids.len()).rev() {
            let masked = mask_client_update(
                &forward.ids[index],
                &forward.keys[index],
                &forward.updates[index],
                &plan,
            )
            .expect("mask");
            aggregator.receive(masked).expect("receive");
        }
        let actual = aggregator.aggregate().expect("aggregate");
        assert_eq!(actual.to_vec(), expected.to_vec());
    }
}
