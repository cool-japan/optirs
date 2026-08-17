//! Core SMPC building blocks: Shamir secret sharing, the hash commitment
//! scheme, verification tags, protocol configuration and participant
//! bookkeeping, the [`CryptographicAggregator`], and the keyed-digest
//! "homomorphic" engine (not real homomorphic encryption; see the module-level
//! docs).

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;

use super::coordinator::{
    HomomorphicCiphertext, MaliciousTolerance, PrivacyLevel, SecureAggregationResult,
};
use super::helpers::{
    add_mod, ct_eq, evaluate_polynomial, field_to_value, hash_values, inv_mod, mul_mod, neg_mod,
    os_seeded_rng, random_bytes, require_supported_security, sub_mod, unimplemented_homomorphic,
    value_to_field, SecureRng, AGGREGATE_DOMAIN, COMMITMENT_DOMAIN, COMMITMENT_NONCE_LEN,
    SHAMIR_PRIME, VALUE_DIGEST_DOMAIN, VERIFICATION_DOMAIN,
};

/// Communication security models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationSecurity {
    /// Semi-honest adversaries.
    SemiHonest,
    /// Malicious adversaries with abort. **Not implemented.**
    MaliciousAbort,
    /// Malicious adversaries with guaranteed output. **Not implemented.**
    MaliciousGuaranteed,
}
/// SMPC protocol execution state
#[derive(Debug, Clone)]
pub enum SMPCProtocolState {
    /// Initialization phase
    Initialization,
    /// Key generation and setup
    Setup,
    /// Input sharing phase
    InputSharing,
    /// Computation phase
    Computation,
    /// Output reconstruction
    OutputReconstruction,
    /// Protocol completed
    Completed,
    /// Protocol aborted; the payload is the reason.
    Aborted(String),
}
/// Guarantees achieved by an SMPC run.
///
/// Every field is derived from what actually executed; nothing here is hardcoded to
/// an aspirational value.
#[derive(Debug, Clone)]
pub struct SMPCSecurityGuarantees {
    /// Protocol variant used
    pub protocol_variant: SMPCProtocol,
    /// Communication security model under which the run happened.
    pub communication_security: CommunicationSecurity,
    /// Number of malicious parties actually tolerated. Currently always `0`.
    pub malicious_tolerance: usize,
    /// Privacy level achieved.
    pub privacy_level: PrivacyLevel,
    /// Whether an honest run always reconstructs the intended value.
    pub completeness: bool,
    /// Whether a dishonest party is prevented from forging an accepted result.
    /// Currently always `false`.
    pub soundness: bool,
    /// Human-readable limitations of the executed protocol.
    pub limitations: Vec<String>,
}
/// Shamir `k`-of-`n` secret sharing over `F_p`, `p = 2^127 - 1`.
///
/// Values of type `T` are mapped into the field by fixed-point quantisation with
/// [`FIXED_POINT_BITS`] fractional bits, so reconstruction is exact in the field and
/// the only error is the documented quantisation error. Coefficients are sampled
/// uniformly over the *whole* field from an OS-seeded CSPRNG, which is what makes
/// fewer than `k` shares information-theoretically independent of the secret.
pub struct ShamirSecretSharing<T: Float + Debug + Send + Sync + 'static> {
    /// Threshold for reconstruction (`k`).
    pub(super) threshold: usize,
    /// Number of shares produced (`n`).
    pub(super) num_shares: usize,
    /// Prime field modulus used for all arithmetic.
    pub(super) prime_field: u128,
    /// Generator used for polynomial coefficients.
    pub(super) rng: SecureRng,
    /// Marker for the value type handled by this instance.
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> ShamirSecretSharing<T> {
    /// Create a new secret sharing instance seeded from OS entropy.
    ///
    /// Fails unless `1 <= threshold <= num_shares`.
    pub fn new(threshold: usize, num_shares: usize) -> Result<Self> {
        Self::validate_parameters(threshold, num_shares)?;
        Ok(Self {
            threshold,
            num_shares,
            prime_field: SHAMIR_PRIME,
            rng: os_seeded_rng(),
            _phantom: PhantomData,
        })
    }
    /// Create a deterministic instance from an explicit seed.
    ///
    /// **Test-only.** Shares produced by two instances with the same seed are
    /// identical, which destroys the secrecy of the polynomial coefficients.
    pub fn with_seed(threshold: usize, num_shares: usize, seed: u64) -> Result<Self> {
        Self::validate_parameters(threshold, num_shares)?;
        Ok(Self {
            threshold,
            num_shares,
            prime_field: SHAMIR_PRIME,
            rng: Random::seed(seed),
            _phantom: PhantomData,
        })
    }
    pub(super) fn validate_parameters(threshold: usize, num_shares: usize) -> Result<()> {
        if threshold == 0 {
            return Err(OptimError::InvalidConfig(
                "Shamir threshold must be at least 1".to_string(),
            ));
        }
        if num_shares == 0 {
            return Err(OptimError::InvalidConfig(
                "Shamir requires at least one share".to_string(),
            ));
        }
        if threshold > num_shares {
            return Err(OptimError::InvalidConfig(format!(
                "Shamir threshold {threshold} exceeds the number of shares {num_shares}"
            )));
        }
        Ok(())
    }
    /// Reconstruction threshold `k`.
    pub fn threshold(&self) -> usize {
        self.threshold
    }
    /// Number of shares `n`.
    pub fn num_shares(&self) -> usize {
        self.num_shares
    }
    /// Prime modulus of the field used for all share arithmetic.
    pub fn prime_field(&self) -> u128 {
        self.prime_field
    }
    /// Split `secret` into [`Self::num_shares`] shares.
    ///
    /// Fresh coefficients are drawn on every call, so sharing the same secret twice
    /// yields unrelated shares.
    pub fn share_secret(&mut self, secret: T) -> Result<Vec<Share>> {
        let element = value_to_field(secret)?;
        self.share_field_element(element)
    }
    /// Split a field element into shares.
    pub fn share_field_element(&mut self, secret: u128) -> Result<Vec<Share>> {
        let mut coefficients = Vec::with_capacity(self.threshold);
        coefficients.push(secret % self.prime_field);
        for _ in 1..self.threshold {
            coefficients.push(self.random_field_element());
        }
        let mut shares = Vec::with_capacity(self.num_shares);
        for index in 1..=self.num_shares {
            let x = index as u128;
            if x >= self.prime_field {
                return Err(OptimError::InvalidConfig(
                    "number of shares exceeds the size of the field".to_string(),
                ));
            }
            shares.push(Share {
                x: index,
                y: evaluate_polynomial(&coefficients, x),
            });
        }
        Ok(shares)
    }
    /// Reconstruct a secret from at least `threshold` shares.
    pub fn reconstruct_secret(&self, shares: &[Share]) -> Result<T> {
        let element = self.reconstruct_field_element(shares)?;
        field_to_value(element)
    }
    /// Reconstruct the underlying field element from at least `threshold` shares.
    pub fn reconstruct_field_element(&self, shares: &[Share]) -> Result<u128> {
        if shares.len() < self.threshold {
            return Err(OptimError::InvalidConfig(format!(
                "insufficient shares for reconstruction: got {}, need {}",
                shares.len(),
                self.threshold
            )));
        }
        let used = &shares[..self.threshold];
        for (i, share) in used.iter().enumerate() {
            if share.x == 0 {
                return Err(OptimError::InvalidConfig(
                    "share x-coordinate must not be zero".to_string(),
                ));
            }
            if used.iter().skip(i + 1).any(|other| other.x == share.x) {
                return Err(OptimError::InvalidConfig(format!(
                    "duplicate share x-coordinate {}",
                    share.x
                )));
            }
        }
        let mut result = 0u128;
        for (i, share) in used.iter().enumerate() {
            let xi = (share.x as u128) % self.prime_field;
            let mut numerator = 1u128;
            let mut denominator = 1u128;
            for (j, other) in used.iter().enumerate() {
                if i == j {
                    continue;
                }
                let xj = (other.x as u128) % self.prime_field;
                numerator = mul_mod(numerator, neg_mod(xj));
                denominator = mul_mod(denominator, sub_mod(xi, xj));
            }
            let lagrange = mul_mod(numerator, inv_mod(denominator)?);
            result = add_mod(result, mul_mod(share.y % self.prime_field, lagrange));
        }
        Ok(result)
    }
    /// Draw a uniformly random field element.
    pub(super) fn random_field_element(&mut self) -> u128 {
        loop {
            let high = self.rng.next_u64() as u128;
            let low = self.rng.next_u64() as u128;
            let candidate = ((high << 64) | low) & ((1u128 << 127) - 1);
            if candidate < self.prime_field {
                return candidate;
            }
        }
    }
}
/// Keyed digest engine kept for API compatibility.
///
/// # WARNING: This is not homomorphic encryption
///
/// [`HomomorphicEngine::encrypt`] returns one keyed SHA-256 digest per input value.
/// A digest is one-way: there is no key that recovers the plaintext, and digests are
/// not additively homomorphic. Consequently
/// [`decrypt`](HomomorphicEngine::decrypt) and
/// [`add_encrypted`](HomomorphicEngine::add_encrypted) return
/// [`OptimError::UnsupportedOperation`] instead of returning bytes that look like a
/// plaintext or a ciphertext sum.
///
/// **Do not use this type for confidentiality.** For additive aggregation use the
/// masking protocol in [`crate::privacy::secure_aggregation`].
pub struct HomomorphicEngine<T: Float + Debug + Send + Sync + 'static> {
    /// Informational parameters carried on every produced value.
    pub(super) params: HomomorphicParameters<T>,
    /// Per-instance random key mixed into every digest.
    pub(super) digest_key: Vec<u8>,
}
impl<T: Float + Debug + Send + Sync + 'static> HomomorphicEngine<T> {
    /// Create an engine with an OS-seeded digest key.
    pub fn new() -> Self {
        let mut rng = os_seeded_rng();
        Self {
            params: HomomorphicParameters::new(),
            digest_key: random_bytes(&mut rng, 32),
        }
    }
    /// Create a deterministic engine from an explicit seed.
    ///
    /// **Test-only.**
    pub fn with_seed(seed: u64) -> Self {
        let mut rng = Random::seed(seed);
        Self {
            params: HomomorphicParameters::new(),
            digest_key: random_bytes(&mut rng, 32),
        }
    }
    /// Informational parameters attached to produced values.
    pub fn params(&self) -> &HomomorphicParameters<T> {
        &self.params
    }
    /// Compute one keyed digest per value.
    ///
    /// This is **not** encryption; see the type-level documentation.
    pub fn digest_values(&self, data: &Array1<T>) -> Result<HomomorphicCiphertext<T>> {
        let mut digests = Vec::with_capacity(data.len());
        for &value in data.iter() {
            digests.push(self.digest_value(value)?);
        }
        Ok(HomomorphicCiphertext {
            data: digests,
            params: self.params.clone(),
        })
    }
    /// Alias of [`Self::digest_values`], kept for API compatibility.
    ///
    /// **This does not encrypt anything.** The returned value cannot be decrypted.
    pub fn encrypt(&self, data: &Array1<T>) -> Result<HomomorphicCiphertext<T>> {
        self.digest_values(data)
    }
    /// Always fails: digests cannot be inverted.
    ///
    /// Returns [`OptimError::UnsupportedOperation`]; the previous implementation
    /// reinterpreted digest bytes as an `f64` and returned garbage.
    pub fn decrypt(&self, ciphertext: &HomomorphicCiphertext<T>) -> Result<Array1<T>> {
        ciphertext.validate()?;
        Err(unimplemented_homomorphic("decryption"))
    }
    /// Always fails: digests are not additively homomorphic.
    pub fn add_encrypted(
        &self,
        a: &HomomorphicCiphertext<T>,
        b: &HomomorphicCiphertext<T>,
    ) -> Result<HomomorphicCiphertext<T>> {
        a.validate()?;
        b.validate()?;
        if a.data.len() != b.data.len() {
            return Err(OptimError::DimensionMismatch(
                "digest vectors have different lengths".to_string(),
            ));
        }
        Err(unimplemented_homomorphic("addition"))
    }
    /// Digest a single value under the instance key.
    pub(super) fn digest_value(&self, value: T) -> Result<Vec<u8>> {
        let as_f64 = value.to_f64().ok_or_else(|| {
            OptimError::InvalidConfig("value cannot be converted to f64 for digesting".to_string())
        })?;
        let mut hasher = Sha256::new();
        hasher.update(VALUE_DIGEST_DOMAIN);
        hasher.update((self.digest_key.len() as u64).to_le_bytes());
        hasher.update(&self.digest_key);
        hasher.update(as_f64.to_le_bytes());
        Ok(hasher.finalize().to_vec())
    }
}
/// Aggregator that opens participant commitments before averaging their inputs.
pub struct CryptographicAggregator<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration
    pub(super) config: SMPCConfig,
    /// Commitment scheme used for the aggregate digest and for opening participants.
    pub(super) commitment_scheme: CommitmentScheme<T>,
    /// Verification parameters
    pub(super) verification_params: VerificationParameters<T>,
    /// Aggregation proofs produced so far
    pub(super) aggregation_proofs: Vec<AggregationProof<T>>,
}
impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    CryptographicAggregator<T>
{
    /// Create a new cryptographic aggregator seeded from OS entropy.
    pub fn new(config: SMPCConfig) -> Self {
        Self {
            config,
            commitment_scheme: CommitmentScheme::new(),
            verification_params: VerificationParameters::new(),
            aggregation_proofs: Vec::new(),
        }
    }
    /// Create a deterministic aggregator from an explicit seed.
    ///
    /// **Test-only.**
    pub fn with_seed(config: SMPCConfig, seed: u64) -> Self {
        Self {
            config,
            commitment_scheme: CommitmentScheme::with_seed(seed),
            verification_params: VerificationParameters::with_seed(seed),
            aggregation_proofs: Vec::new(),
        }
    }
    /// Proofs generated by previous calls to [`Self::secure_aggregate`].
    pub fn aggregation_proofs(&self) -> &[AggregationProof<T>] {
        &self.aggregation_proofs
    }
    /// Verification parameters used to tag aggregates.
    pub fn verification_params(&self) -> &VerificationParameters<T> {
        &self.verification_params
    }
    /// Aggregate the inputs of every participant whose commitment opens correctly.
    ///
    /// The commitment check binds a participant to the value it published earlier; it
    /// does **not** authenticate the participant, so a corrupted party can still
    /// commit to an arbitrary input. Malicious-security models are rejected instead of
    /// being silently downgraded.
    pub fn secure_aggregate(
        &mut self,
        participant_inputs: &HashMap<String, Array1<T>>,
        participants: &HashMap<String, Participant>,
    ) -> Result<SecureAggregationResult<T>> {
        require_supported_security(self.config.communication_security)?;
        let honest_participants =
            self.select_honest_participants(participant_inputs, participants)?;
        let aggregate = self.aggregate_honest_inputs(participant_inputs, &honest_participants)?;
        let mut commitments = HashMap::new();
        for id in &honest_participants {
            if let Some(commitment) = participants.get(id).and_then(|p| p.commitment.clone()) {
                commitments.insert(id.clone(), commitment);
            }
        }
        let proof = self.generate_aggregation_proof(&aggregate, &commitments)?;
        Ok(SecureAggregationResult {
            aggregate,
            honest_participants,
            proof,
            security_level: self.config.communication_security,
        })
    }
    /// Select the participants whose submitted input opens their published commitment.
    pub(super) fn select_honest_participants(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        participants: &HashMap<String, Participant>,
    ) -> Result<Vec<String>> {
        let mut ids: Vec<&String> = participants.keys().collect();
        ids.sort();
        let mut honest_participants = Vec::new();
        for id in ids {
            let participant = match participants.get(id) {
                Some(participant) => participant,
                None => continue,
            };
            let input = match inputs.get(id) {
                Some(input) => input,
                None => continue,
            };
            if self.verify_participant_honesty(participant, input)? {
                honest_participants.push(id.clone());
            }
        }
        if honest_participants.len() < self.config.threshold {
            return Err(OptimError::InvalidConfig(format!(
                "insufficient honest participants for secure aggregation: {} verified, {} required",
                honest_participants.len(),
                self.config.threshold
            )));
        }
        Ok(honest_participants)
    }
    /// Average the inputs of the selected participants.
    pub(super) fn aggregate_honest_inputs(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        honest_participants: &[String],
    ) -> Result<Array1<T>> {
        let first_participant = honest_participants.first().ok_or_else(|| {
            OptimError::InvalidConfig("no honest participants for aggregation".to_string())
        })?;
        let first_input = inputs.get(first_participant).ok_or_else(|| {
            OptimError::InvalidConfig(format!("missing input for participant {first_participant}"))
        })?;
        let dimension = first_input.len();
        let mut aggregate = Array1::zeros(dimension);
        let mut count = 0usize;
        for participant_id in honest_participants {
            let input = inputs.get(participant_id).ok_or_else(|| {
                OptimError::InvalidConfig(format!("missing input for participant {participant_id}"))
            })?;
            if input.len() != dimension {
                return Err(OptimError::DimensionMismatch(format!(
                    "participant {} submitted {} values, expected {}",
                    participant_id,
                    input.len(),
                    dimension
                )));
            }
            aggregate = aggregate + input;
            count += 1;
        }
        let divisor = T::from(count).ok_or_else(|| {
            OptimError::InvalidConfig("participant count is not representable in T".to_string())
        })?;
        if divisor == T::zero() {
            return Err(OptimError::InvalidConfig(
                "no honest participants for aggregation".to_string(),
            ));
        }
        Ok(aggregate / divisor)
    }
    /// Generate the integrity record for an aggregation round.
    pub(super) fn generate_aggregation_proof(
        &mut self,
        aggregate: &Array1<T>,
        commitments: &HashMap<String, Vec<u8>>,
    ) -> Result<AggregationProof<T>> {
        let proof = AggregationProof {
            aggregate_digest: hash_values(AGGREGATE_DOMAIN, &[], aggregate)?,
            participant_commitments: commitments.clone(),
            verification_data: self
                .verification_params
                .generate_verification_data(aggregate)?,
            timestamp: std::time::SystemTime::now(),
            _phantom: PhantomData,
        };
        self.aggregation_proofs.push(proof.clone());
        Ok(proof)
    }
    /// Verify that a participant's submitted input opens its published commitment.
    ///
    /// Returns `false` — never an unearned `true` — when the participant is flagged,
    /// inactive, publishes no commitment or no opening nonce, submits a value that
    /// does not open its commitment, or declares a trust score below the configured
    /// threshold.
    pub fn verify_participant_honesty(
        &self,
        participant: &Participant,
        submitted_input: &Array1<T>,
    ) -> Result<bool> {
        if participant.status != ParticipantStatus::Active {
            return Ok(false);
        }
        let commitment = match &participant.commitment {
            Some(commitment) => commitment,
            None => return Ok(false),
        };
        let nonce = match &participant.commitment_nonce {
            Some(nonce) => nonce,
            None => return Ok(false),
        };
        if !self
            .commitment_scheme
            .open(commitment, submitted_input, nonce)?
        {
            return Ok(false);
        }
        Ok(participant.trust_score >= self.config.malicious_tolerance.verification_threshold)
    }
}
/// Configuration for SMPC protocols
#[derive(Debug, Clone)]
pub struct SMPCConfig {
    /// Number of participants
    pub num_participants: usize,
    /// Threshold for secret sharing (k in k-out-of-n). Must be in `1..=num_participants`.
    pub threshold: usize,
    /// Security parameter recorded for auditing. Informational only.
    pub security_parameter: usize,
    /// Request homomorphic encryption.
    ///
    /// No homomorphic backend exists; [`SMPCCoordinator::new`] rejects `true`.
    pub enable_homomorphic: bool,
    /// Request zero-knowledge proofs.
    ///
    /// No proof system exists; [`SMPCCoordinator::new`] rejects `true`.
    pub enable_zk_proofs: bool,
    /// SMPC protocol variant. Only [`SMPCProtocol::FederatedSMPC`] is implemented.
    pub protocol_variant: SMPCProtocol,
    /// Communication security level. Only [`CommunicationSecurity::SemiHonest`] is
    /// implemented.
    pub communication_security: CommunicationSecurity,
    /// Malicious adversary tolerance requested by the caller.
    ///
    /// No malicious-security mechanism is implemented, so the *achieved* tolerance
    /// reported in [`SMPCSecurityGuarantees`] is always zero.
    pub malicious_tolerance: MaliciousTolerance,
}
/// A single Shamir share: the evaluation point and the field element `P(x)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Share {
    /// Evaluation point. Always `>= 1`; `x = 0` is the secret itself.
    pub x: usize,
    /// Field element `P(x) mod SHAMIR_PRIME`.
    pub y: u128,
}
/// SMPC protocol variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SMPCProtocol {
    /// BGW protocol for arithmetic circuits. **Not implemented.**
    BGW,
    /// GMW protocol for boolean circuits. **Not implemented.**
    GMW,
    /// SPDZ protocol with preprocessing. **Not implemented.**
    SPDZ,
    /// ABY hybrid protocol. **Not implemented.**
    ABY,
    /// Single-coordinator Shamir sharing simulation used for federated learning.
    FederatedSMPC,
}
/// Hash commitment scheme with per-commitment blinding.
///
/// `commit` returns `(SHA-256(domain || nonce || len || value), nonce)`. It is
/// *binding* under the collision resistance of SHA-256 and *hiding* as long as the
/// 32-byte nonce, drawn from an OS-seeded CSPRNG, stays secret. Committing to the
/// same value twice yields two different commitments.
pub struct CommitmentScheme<T: Float + Debug + Send + Sync + 'static> {
    /// Generator used for commitment nonces.
    pub(super) rng: SecureRng,
    /// Marker for the committed value type.
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> CommitmentScheme<T> {
    /// Create a scheme whose nonces come from OS entropy.
    pub fn new() -> Self {
        Self {
            rng: os_seeded_rng(),
            _phantom: PhantomData,
        }
    }
    /// Create a deterministic scheme from an explicit seed.
    ///
    /// **Test-only.** Deterministic nonces make commitments non-hiding.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Random::seed(seed),
            _phantom: PhantomData,
        }
    }
    /// Commit to `value`, returning the commitment and its opening nonce.
    pub fn commit(&mut self, value: &Array1<T>) -> Result<(Vec<u8>, CommitmentNonce)> {
        let mut nonce_bytes = [0u8; COMMITMENT_NONCE_LEN];
        let random = random_bytes(&mut self.rng, COMMITMENT_NONCE_LEN);
        nonce_bytes.copy_from_slice(&random);
        let nonce = CommitmentNonce(nonce_bytes);
        let commitment = hash_values(COMMITMENT_DOMAIN, nonce.as_bytes(), value)?;
        Ok((commitment, nonce))
    }
    /// Check that `commitment` opens to `value` under `nonce`.
    pub fn open(
        &self,
        commitment: &[u8],
        value: &Array1<T>,
        nonce: &CommitmentNonce,
    ) -> Result<bool> {
        let expected = hash_values(COMMITMENT_DOMAIN, nonce.as_bytes(), value)?;
        Ok(ct_eq(commitment, &expected))
    }
}
/// Participant in an SMPC protocol.
#[derive(Debug, Clone)]
pub struct Participant {
    /// Unique participant identifier.
    pub id: String,
    /// Public key material for the participant. Not used for any signature check;
    /// no authenticated channel is implemented.
    pub public_key: Vec<u8>,
    /// Participation status.
    pub status: ParticipantStatus,
    /// Trust score supplied by the caller (see
    /// [`MaliciousTolerance::verification_threshold`]).
    pub trust_score: f64,
    /// Commitment the participant published for its input, produced with
    /// [`CommitmentScheme::commit`].
    pub commitment: Option<Vec<u8>>,
    /// Opening nonce for [`Participant::commitment`].
    pub commitment_nonce: Option<CommitmentNonce>,
}
impl Participant {
    /// Create an active participant with no commitment yet.
    pub fn new(id: impl Into<String>, public_key: Vec<u8>, trust_score: f64) -> Self {
        Self {
            id: id.into(),
            public_key,
            status: ParticipantStatus::Active,
            trust_score,
            commitment: None,
            commitment_nonce: None,
        }
    }
    /// Attach a commitment and its opening nonce.
    pub fn with_commitment(mut self, commitment: Vec<u8>, nonce: CommitmentNonce) -> Self {
        self.commitment = Some(commitment);
        self.commitment_nonce = Some(nonce);
        self
    }
}
/// Participant status in protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticipantStatus {
    /// Active and participating
    Active,
    /// Temporarily unavailable
    Unavailable,
    /// Suspected malicious behavior
    Suspicious,
    /// Confirmed malicious behavior
    Malicious,
}
/// Keyed integrity tag over an aggregation result.
///
/// The key is generated per instance from OS entropy and never leaves the process,
/// so the tag is only verifiable by the instance that produced it. It detects
/// accidental corruption of an aggregate held in memory or on disk; it is **not** a
/// proof of correct aggregation and cannot be checked by a third party.
pub struct VerificationParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Secret key mixed into every tag.
    pub(super) verification_key: Vec<u8>,
    /// Marker for the aggregate value type.
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> VerificationParameters<T> {
    /// Create parameters with an OS-seeded key.
    pub fn new() -> Self {
        let mut rng = os_seeded_rng();
        Self {
            verification_key: random_bytes(&mut rng, 64),
            _phantom: PhantomData,
        }
    }
    /// Create deterministic parameters from an explicit seed.
    ///
    /// **Test-only.**
    pub fn with_seed(seed: u64) -> Self {
        let mut rng = Random::seed(seed);
        Self {
            verification_key: random_bytes(&mut rng, 64),
            _phantom: PhantomData,
        }
    }
    /// Produce the keyed integrity tag for `aggregate`.
    pub fn generate_verification_data(&self, aggregate: &Array1<T>) -> Result<Vec<u8>> {
        hash_values(VERIFICATION_DOMAIN, &self.verification_key, aggregate)
    }
    /// Check a tag previously produced by this instance.
    pub fn verify_verification_data(&self, aggregate: &Array1<T>, tag: &[u8]) -> Result<bool> {
        let expected = self.generate_verification_data(aggregate)?;
        Ok(ct_eq(tag, &expected))
    }
}
/// Informational parameters attached to [`HomomorphicCiphertext`].
///
/// These describe what a real FHE backend *would* be configured with. Nothing in this
/// module consumes them.
#[derive(Debug, Clone)]
pub struct HomomorphicParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Security level in bits a real backend would target. Informational only.
    pub security_level: usize,
    /// Modulus a real backend would use. Informational only.
    pub modulus: u128,
    /// Marker for the value type.
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> HomomorphicParameters<T> {
    /// Create the default informational parameters.
    pub fn new() -> Self {
        Self {
            security_level: 128,
            modulus: u64::MAX as u128,
            _phantom: PhantomData,
        }
    }
}
/// Blinding factor of a single commitment.
///
/// The nonce must be kept secret until the commitment is opened: it is what makes the
/// commitment hiding. Its `Debug` representation is redacted for that reason.
#[derive(Clone, PartialEq, Eq)]
pub struct CommitmentNonce(pub(super) [u8; COMMITMENT_NONCE_LEN]);
impl CommitmentNonce {
    /// Wrap raw nonce bytes.
    pub fn from_bytes(bytes: [u8; COMMITMENT_NONCE_LEN]) -> Self {
        Self(bytes)
    }
    /// Access the raw nonce bytes.
    pub fn as_bytes(&self) -> &[u8; COMMITMENT_NONCE_LEN] {
        &self.0
    }
}
/// Record of one aggregation round.
#[derive(Debug, Clone)]
pub struct AggregationProof<T: Float + Debug + Send + Sync + 'static> {
    /// Unkeyed SHA-256 digest of the (public) aggregate.
    ///
    /// An integrity checksum, not a hiding commitment.
    pub aggregate_digest: Vec<u8>,
    /// Commitments published by the participants that were included.
    pub participant_commitments: HashMap<String, Vec<u8>>,
    /// Keyed tag from [`VerificationParameters`], verifiable only locally.
    pub verification_data: Vec<u8>,
    /// Timestamp of record generation.
    pub timestamp: std::time::SystemTime,
    /// Marker for the value type.
    pub(super) _phantom: PhantomData<T>,
}

impl Debug for CommitmentNonce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CommitmentNonce(<redacted>)")
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CommitmentScheme<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for HomomorphicEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for HomomorphicParameters<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for VerificationParameters<T> {
    fn default() -> Self {
        Self::new()
    }
}
