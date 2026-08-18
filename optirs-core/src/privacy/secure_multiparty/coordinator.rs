//! The [`SMPCCoordinator`]: single-process simulation of a full SMPC round
//! (share distribution, homomorphic-style aggregation, digest verification and
//! zero-knowledge-shaped hooks — see the module-level docs for what is and is
//! not a real cryptographic guarantee), plus its computation/result/security
//! summary types and the [`ComputationDigestSystem`].

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;

use super::helpers::{
    add_mod, ct_eq, hash_values, max_representable_magnitude, require_supported_security,
    unimplemented_zero_knowledge, COMPUTATION_DOMAIN, DIGEST_LEN, FIXED_POINT_BITS, SHAMIR_PRIME,
};
use super::primitives::{
    AggregationProof, CommunicationSecurity, CryptographicAggregator, HomomorphicParameters,
    Participant, ParticipantStatus, SMPCConfig, SMPCProtocol, SMPCProtocolState,
    SMPCSecurityGuarantees, ShamirSecretSharing, Share,
};

/// Secure Multi-Party Computation coordinator.
///
/// # WARNING: Single-process simulation
///
/// The coordinator generates every share and reconstructs every output itself, so it
/// sees all secret material. It is useful for exercising the protocol shape and for
/// deterministic tests, not for protecting inputs from the coordinator.
pub struct SMPCCoordinator<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for SMPC protocols
    pub(super) config: SMPCConfig,
    /// Shamir secret sharing engine
    pub(super) secret_sharing: ShamirSecretSharing<T>,
    /// Secure aggregation with commitment opening
    pub(super) secure_aggregator: CryptographicAggregator<T>,
    /// Computation digest system
    pub(super) digest_system: ComputationDigestSystem<T>,
    /// Participant management
    pub(super) participants: HashMap<String, Participant>,
    /// Current protocol state
    pub(super) protocol_state: SMPCProtocolState,
}
impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    SMPCCoordinator<T>
{
    /// Create a new SMPC coordinator.
    ///
    /// Configurations that request unimplemented functionality (homomorphic
    /// encryption, zero-knowledge proofs, malicious-adversary security, or a protocol
    /// variant other than [`SMPCProtocol::FederatedSMPC`]) are rejected here rather
    /// than silently ignored.
    pub fn new(config: SMPCConfig) -> Result<Self> {
        if config.num_participants == 0 {
            return Err(OptimError::InvalidConfig(
                "num_participants must be at least 1".to_string(),
            ));
        }
        if config.threshold == 0 || config.threshold > config.num_participants {
            return Err(OptimError::InvalidConfig(format!(
                "threshold {} must be in 1..={}",
                config.threshold, config.num_participants
            )));
        }
        if config.enable_homomorphic {
            return Err(OptimError::UnsupportedOperation(
                "enable_homomorphic requires a homomorphic encryption backend, which is not \
                 implemented — `HomomorphicEngine` only produces one-way digests"
                    .to_string(),
            ));
        }
        if config.enable_zk_proofs {
            return Err(OptimError::UnsupportedOperation(
                "enable_zk_proofs requires a zero-knowledge proof system, which is not \
                 implemented — only the non-hiding `ComputationDigest` is available"
                    .to_string(),
            ));
        }
        require_supported_security(config.communication_security)?;
        if config.protocol_variant != SMPCProtocol::FederatedSMPC {
            return Err(OptimError::UnsupportedOperation(format!(
                "protocol variant {:?} is not implemented; only SMPCProtocol::FederatedSMPC \
                 (single-coordinator Shamir sharing) is available",
                config.protocol_variant
            )));
        }
        let secret_sharing = ShamirSecretSharing::new(config.threshold, config.num_participants)?;
        let secure_aggregator = CryptographicAggregator::new(config.clone());
        Ok(Self {
            config,
            secret_sharing,
            secure_aggregator,
            digest_system: ComputationDigestSystem::new(),
            participants: HashMap::new(),
            protocol_state: SMPCProtocolState::Initialization,
        })
    }
    /// Current protocol state.
    pub fn protocol_state(&self) -> &SMPCProtocolState {
        &self.protocol_state
    }
    /// Registered participants.
    pub fn participants(&self) -> &HashMap<String, Participant> {
        &self.participants
    }
    /// Configuration this coordinator was built with.
    pub fn config(&self) -> &SMPCConfig {
        &self.config
    }
    /// Add a participant to the protocol.
    pub fn add_participant(&mut self, participant: Participant) -> Result<()> {
        if participant.id.is_empty() {
            return Err(OptimError::InvalidConfig(
                "participant id must not be empty".to_string(),
            ));
        }
        if self.participants.contains_key(&participant.id) {
            return Err(OptimError::InvalidConfig(format!(
                "participant {} is already registered",
                participant.id
            )));
        }
        if self.participants.len() >= self.config.num_participants {
            return Err(OptimError::InvalidConfig(
                "maximum number of participants reached".to_string(),
            ));
        }
        self.participants
            .insert(participant.id.clone(), participant);
        Ok(())
    }
    /// Aggregate participant inputs through the commitment-opening aggregator.
    pub fn secure_aggregate(
        &mut self,
        participant_inputs: &HashMap<String, Array1<T>>,
    ) -> Result<SecureAggregationResult<T>> {
        self.secure_aggregator
            .secure_aggregate(participant_inputs, &self.participants)
    }
    /// Execute a secure multi-party computation.
    ///
    /// The returned array always has the dimension of the participant inputs: shares
    /// are kept per coordinate and every coordinate is reconstructed independently.
    pub fn execute_smpc(
        &mut self,
        participant_inputs: HashMap<String, Array1<T>>,
        computation: SMPCComputation,
    ) -> Result<SMPCResult<T>> {
        match self.execute_smpc_inner(&participant_inputs, &computation) {
            Ok(result) => {
                self.protocol_state = SMPCProtocolState::Completed;
                Ok(result)
            }
            Err(error) => {
                self.protocol_state = SMPCProtocolState::Aborted(error.to_string());
                Err(error)
            }
        }
    }
    pub(super) fn execute_smpc_inner(
        &mut self,
        participant_inputs: &HashMap<String, Array1<T>>,
        computation: &SMPCComputation,
    ) -> Result<SMPCResult<T>> {
        self.protocol_state = SMPCProtocolState::Setup;
        self.verify_participants()?;
        let dimension = self.validate_inputs(participant_inputs)?;
        self.protocol_state = SMPCProtocolState::InputSharing;
        let shared_inputs = self.share_inputs(participant_inputs)?;
        self.protocol_state = SMPCProtocolState::Computation;
        let (combined_shares, divisor) =
            self.perform_secure_computation(&shared_inputs, computation, dimension)?;
        self.protocol_state = SMPCProtocolState::OutputReconstruction;
        let reconstructed = self.reconstruct_output(&combined_shares)?;
        let result = reconstructed / divisor;
        let combined_input = self.combine_inputs(participant_inputs);
        let digest = self.digest_system.digest_computation(
            &combined_input,
            &result,
            &computation.label(),
        )?;
        let mut participating_parties: Vec<String> = participant_inputs.keys().cloned().collect();
        participating_parties.sort();
        Ok(SMPCResult {
            result,
            digest,
            participating_parties,
            security_guarantees: self.get_security_guarantees(),
        })
    }
    /// Verify that enough participants are registered and active.
    pub(super) fn verify_participants(&self) -> Result<()> {
        if self.config.threshold > self.config.num_participants {
            return Err(OptimError::InvalidConfig(
                "threshold exceeds the number of participants".to_string(),
            ));
        }
        if self.participants.len() < self.config.threshold {
            return Err(OptimError::InvalidConfig(format!(
                "insufficient participants for protocol: {} registered, {} required",
                self.participants.len(),
                self.config.threshold
            )));
        }
        let active = self
            .participants
            .values()
            .filter(|p| p.status == ParticipantStatus::Active)
            .count();
        if active < self.config.threshold {
            return Err(OptimError::InvalidConfig(format!(
                "insufficient active participants: {} active, {} required",
                active, self.config.threshold
            )));
        }
        Ok(())
    }
    /// Validate the submitted inputs and return their common dimension.
    pub(super) fn validate_inputs(&self, inputs: &HashMap<String, Array1<T>>) -> Result<usize> {
        if inputs.is_empty() {
            return Err(OptimError::InvalidConfig(
                "no participant inputs provided".to_string(),
            ));
        }
        if inputs.len() > self.config.num_participants {
            return Err(OptimError::InvalidConfig(format!(
                "{} inputs submitted for {} configured participants",
                inputs.len(),
                self.config.num_participants
            )));
        }
        let mut dimension: Option<usize> = None;
        let mut ids: Vec<&String> = inputs.keys().collect();
        ids.sort();
        for id in ids {
            if !self.participants.contains_key(id) {
                return Err(OptimError::InvalidConfig(format!(
                    "input submitted by unregistered participant {id}"
                )));
            }
            let input = inputs.get(id).ok_or_else(|| {
                OptimError::InvalidConfig(format!("missing input for participant {id}"))
            })?;
            if input.is_empty() {
                return Err(OptimError::InvalidConfig(format!(
                    "participant {id} submitted an empty input"
                )));
            }
            match dimension {
                None => dimension = Some(input.len()),
                Some(expected) if expected != input.len() => {
                    return Err(OptimError::DimensionMismatch(format!(
                        "participant {} submitted {} values, expected {}",
                        id,
                        input.len(),
                        expected
                    )));
                }
                Some(_) => {}
            }
        }
        dimension
            .ok_or_else(|| OptimError::InvalidConfig("no participant inputs provided".to_string()))
    }
    /// Share every coordinate of every input separately.
    pub(super) fn share_inputs(
        &mut self,
        inputs: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, Vec<Vec<Share>>>> {
        let mut shared_inputs = HashMap::new();
        for (participant_id, input) in inputs {
            let mut per_coordinate = Vec::with_capacity(input.len());
            for &value in input.iter() {
                per_coordinate.push(self.secret_sharing.share_secret(value)?);
            }
            shared_inputs.insert(participant_id.clone(), per_coordinate);
        }
        Ok(shared_inputs)
    }
    /// Run the requested computation over the shares.
    ///
    /// Returns the resulting per-coordinate shares plus a public divisor applied after
    /// reconstruction (field division by a public constant would not correspond to
    /// fixed-point division).
    pub(super) fn perform_secure_computation(
        &self,
        shared_inputs: &HashMap<String, Vec<Vec<Share>>>,
        computation: &SMPCComputation,
        dimension: usize,
    ) -> Result<(Vec<Vec<Share>>, T)> {
        match computation {
            SMPCComputation::Sum => Ok((self.secure_sum(shared_inputs, dimension)?, T::one())),
            SMPCComputation::Average => {
                let count = T::from(shared_inputs.len()).ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "participant count is not representable in T".to_string(),
                    )
                })?;
                if count == T::zero() {
                    return Err(OptimError::InvalidConfig(
                        "no shared inputs provided".to_string(),
                    ));
                }
                Ok((self.secure_sum(shared_inputs, dimension)?, count))
            }
            SMPCComputation::WeightedSum(_) => Err(OptimError::UnsupportedOperation(
                "weighted sum over secret shares is not implemented: multiplying a share by a \
                 public fixed-point weight needs a truncation protocol to restore the scale"
                    .to_string(),
            )),
            SMPCComputation::Custom(name) => Err(OptimError::UnsupportedOperation(format!(
                "custom SMPC computation `{name}` is not implemented"
            ))),
        }
    }
    /// Add the shares of every participant coordinate-wise inside the field.
    ///
    /// Field addition is exact and commutative, so the result does not depend on the
    /// iteration order of the input map.
    pub(super) fn secure_sum(
        &self,
        shared_inputs: &HashMap<String, Vec<Vec<Share>>>,
        dimension: usize,
    ) -> Result<Vec<Vec<Share>>> {
        if shared_inputs.is_empty() {
            return Err(OptimError::InvalidConfig(
                "no shared inputs provided".to_string(),
            ));
        }
        let num_shares = self.secret_sharing.num_shares();
        let mut accumulator: Vec<Vec<Share>> = (0..dimension)
            .map(|_| (1..=num_shares).map(|x| Share { x, y: 0 }).collect())
            .collect();
        for (participant_id, coordinates) in shared_inputs {
            if coordinates.len() != dimension {
                return Err(OptimError::DimensionMismatch(format!(
                    "participant {} contributed {} coordinates, expected {}",
                    participant_id,
                    coordinates.len(),
                    dimension
                )));
            }
            for (coordinate, shares) in coordinates.iter().enumerate() {
                if shares.len() != num_shares {
                    return Err(OptimError::DimensionMismatch(format!(
                        "participant {} contributed {} shares for coordinate {}, expected {}",
                        participant_id,
                        shares.len(),
                        coordinate,
                        num_shares
                    )));
                }
                for (index, share) in shares.iter().enumerate() {
                    let slot = accumulator
                        .get_mut(coordinate)
                        .and_then(|column| column.get_mut(index))
                        .ok_or_else(|| {
                            OptimError::InvalidConfig("share accumulator overflow".to_string())
                        })?;
                    if slot.x != share.x {
                        return Err(OptimError::InvalidConfig(format!(
                            "share x-coordinate mismatch: expected {}, got {}",
                            slot.x, share.x
                        )));
                    }
                    slot.y = add_mod(slot.y, share.y % SHAMIR_PRIME);
                }
            }
        }
        Ok(accumulator)
    }
    /// Reconstruct one value per coordinate.
    pub(super) fn reconstruct_output(&self, columns: &[Vec<Share>]) -> Result<Array1<T>> {
        if columns.is_empty() {
            return Err(OptimError::InvalidConfig(
                "no shares to reconstruct".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(columns.len());
        for shares in columns {
            values.push(self.secret_sharing.reconstruct_secret(shares)?);
        }
        Ok(Array1::from(values))
    }
    /// Concatenate the inputs in a deterministic (participant-id sorted) order.
    pub(super) fn combine_inputs(&self, inputs: &HashMap<String, Array1<T>>) -> Array1<T> {
        let mut ids: Vec<&String> = inputs.keys().collect();
        ids.sort();
        let mut combined = Vec::new();
        for id in ids {
            if let Some(input) = inputs.get(id) {
                combined.extend(input.iter().copied());
            }
        }
        Array1::from(combined)
    }
    /// Report the guarantees that the executed protocol actually provides.
    pub(super) fn get_security_guarantees(&self) -> SMPCSecurityGuarantees {
        SMPCSecurityGuarantees {
            protocol_variant: self.config.protocol_variant,
            communication_security: self.config.communication_security,
            malicious_tolerance: 0,
            privacy_level: PrivacyLevel::Computational,
            completeness: true,
            soundness: false,
            limitations: vec![
                "the coordinator creates and reconstructs every share in-process, so no \
                 privacy is obtained against the coordinator"
                    .to_string(),
                "no authenticated channels, signatures or verifiable secret sharing are \
                 implemented: a corrupted participant can submit any input undetected"
                    .to_string(),
                "the result carries an integrity digest, not a zero-knowledge proof: it has \
                 no soundness against a malicious prover"
                    .to_string(),
                format!(
                    "values are quantised with {FIXED_POINT_BITS} fractional bits, so \
                     reconstruction is exact in the field but carries a fixed-point rounding \
                     error and rejects magnitudes above {:e}",
                    max_representable_magnitude()
                ),
            ],
        }
    }
}
/// SMPC computation types
#[derive(Debug, Clone)]
pub enum SMPCComputation {
    /// Sum of all inputs
    Sum,
    /// Average of all inputs
    Average,
    /// Weighted sum with given weights. **Not implemented.**
    WeightedSum(Vec<f64>),
    /// Custom computation function. **Not implemented.**
    Custom(String),
}
impl SMPCComputation {
    /// Short label used in computation digests.
    pub fn label(&self) -> String {
        match self {
            SMPCComputation::Sum => "sum".to_string(),
            SMPCComputation::Average => "average".to_string(),
            SMPCComputation::WeightedSum(_) => "weighted_sum".to_string(),
            SMPCComputation::Custom(name) => name.clone(),
        }
    }
}
/// SMPC computation result
#[derive(Debug, Clone)]
pub struct SMPCResult<T: Float + Debug + Send + Sync + 'static> {
    /// Computation result, with the same dimension as the participant inputs.
    pub result: Array1<T>,
    /// Integrity digest of the computation (**not** a zero-knowledge proof).
    pub digest: ComputationDigest<T>,
    /// Participating parties, sorted by id.
    pub participating_parties: Vec<String>,
    /// Guarantees the executed protocol actually provides.
    pub security_guarantees: SMPCSecurityGuarantees,
}
/// Vector of keyed value digests.
///
/// # WARNING: Not a ciphertext
///
/// Each entry is a 32-byte SHA-256 digest of one input value. No key recovers the
/// input and the entries are not additively homomorphic.
#[derive(Debug, Clone)]
pub struct HomomorphicCiphertext<T: Float + Debug + Send + Sync + 'static> {
    /// One [`DIGEST_LEN`]-byte digest per input value.
    pub data: Vec<Vec<u8>>,
    /// Informational parameters.
    pub params: HomomorphicParameters<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> HomomorphicCiphertext<T> {
    /// Number of digest blocks.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Whether the value carries no digest blocks.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Reject blocks that are not exactly [`DIGEST_LEN`] bytes.
    ///
    /// The fields of this type are public and it derives `Clone`/`Debug`, so a caller
    /// can construct or deserialize a short block. Every consumer validates instead of
    /// slice-indexing, so a short block yields an error rather than a panic.
    pub fn validate(&self) -> Result<()> {
        for (index, block) in self.data.iter().enumerate() {
            if block.get(0..DIGEST_LEN).is_none() {
                return Err(OptimError::InvalidConfig(format!(
                    "digest block {} is {} bytes, expected {}",
                    index,
                    block.len(),
                    DIGEST_LEN
                )));
            }
        }
        Ok(())
    }
}
/// Producer of [`ComputationDigest`] values.
///
/// # WARNING: Not a zero-knowledge proof system
///
/// [`prove_computation`](Self::prove_computation) and
/// [`verify_proof`](Self::verify_proof) exist only to fail loudly: no Sigma protocol,
/// SNARK or STARK is implemented here. Use
/// [`digest_computation`](Self::digest_computation) and
/// [`verify_digest`](Self::verify_digest) for the integrity digest that *is*
/// implemented.
pub struct ComputationDigestSystem<T: Float + Debug + Send + Sync + 'static> {
    /// Marker for the value type.
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> ComputationDigestSystem<T> {
    /// Create a digest system.
    ///
    /// There is no common reference string: the digest is unkeyed so that anybody
    /// holding the inputs and outputs can recompute it.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    /// Compute the integrity digest of a computation.
    pub fn digest_computation(
        &self,
        input: &Array1<T>,
        output: &Array1<T>,
        computation: &str,
    ) -> Result<ComputationDigest<T>> {
        let statement = format!("computed {computation} on {} inputs", input.len());
        let mut hasher = Sha256::new();
        hasher.update(COMPUTATION_DOMAIN);
        hasher.update((statement.len() as u64).to_le_bytes());
        hasher.update(statement.as_bytes());
        hasher.update(hash_values(COMPUTATION_DOMAIN, b"input", input)?);
        hasher.update(hash_values(COMPUTATION_DOMAIN, b"output", output)?);
        Ok(ComputationDigest {
            statement,
            digest: hasher.finalize().to_vec(),
            _phantom: PhantomData,
        })
    }
    /// Recompute the digest and compare it with `digest`.
    ///
    /// The verifier must know the inputs and outputs; this is an integrity check, not
    /// a proof.
    pub fn verify_digest(
        &self,
        digest: &ComputationDigest<T>,
        input: &Array1<T>,
        output: &Array1<T>,
        computation: &str,
    ) -> Result<bool> {
        let expected = self.digest_computation(input, output, computation)?;
        Ok(digest.statement == expected.statement && ct_eq(&digest.digest, &expected.digest))
    }
    /// Always fails: no zero-knowledge proof system is implemented.
    pub fn prove_computation(
        &self,
        _input: &Array1<T>,
        _output: &Array1<T>,
        _computation: &str,
    ) -> Result<ComputationDigest<T>> {
        Err(unimplemented_zero_knowledge("proving"))
    }
    /// Always fails: no zero-knowledge proof system is implemented.
    ///
    /// The previous implementation returned `true` whenever the proof bytes were
    /// non-empty and the verification key matched a public constant, so any caller
    /// could forge an accepting proof.
    pub fn verify_proof(&self, _digest: &ComputationDigest<T>) -> Result<bool> {
        Err(unimplemented_zero_knowledge("verification"))
    }
}
/// Privacy levels for SMPC
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyLevel {
    /// Computational privacy
    Computational,
    /// Information-theoretic privacy
    InformationTheoretic,
    /// Perfect privacy
    Perfect,
}
/// Secure aggregation result
#[derive(Debug, Clone)]
pub struct SecureAggregationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Aggregated result
    pub aggregate: Array1<T>,
    /// Participants whose commitment opened against their submitted input.
    pub honest_participants: Vec<String>,
    /// Integrity record for this round.
    pub proof: AggregationProof<T>,
    /// Security model under which the round ran.
    pub security_level: CommunicationSecurity,
}
/// Integrity digest over one SMPC computation.
///
/// # WARNING: Not a zero-knowledge proof
///
/// The digest binds the statement, the inputs and the outputs. Verifying it requires
/// knowing the inputs, so it is not zero-knowledge; and anyone can recompute it for
/// any inputs they choose, so it has no soundness against a malicious prover. It
/// detects accidental corruption of a recorded computation, nothing more.
///
/// Unlike the type it replaces, it carries **no witness**: nothing derived from the
/// plaintext inputs is published inside it.
#[derive(Debug, Clone)]
pub struct ComputationDigest<T: Float + Debug + Send + Sync + 'static> {
    /// Human-readable description of the computation.
    pub statement: String,
    /// `SHA-256(domain || statement || inputs || outputs)`.
    pub digest: Vec<u8>,
    /// Marker for the value type.
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> ComputationDigest<T> {
    /// The digest bytes.
    pub fn digest(&self) -> &[u8] {
        &self.digest
    }
}
/// Malicious adversary tolerance configuration
#[derive(Debug, Clone)]
pub struct MaliciousTolerance {
    /// Maximum number of corrupted participants the caller wants to tolerate.
    pub max_corrupted: usize,
    /// Enable Byzantine fault tolerance. Informational; no such mechanism runs here.
    pub byzantine_tolerance: bool,
    /// Minimum trust score a participant must declare to be included.
    ///
    /// The trust score is supplied by the caller, so this is an operator policy
    /// filter, not a cryptographic check.
    pub verification_threshold: f64,
    /// Enable commit-and-prove protocols. Informational.
    pub commit_and_prove: bool,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ComputationDigestSystem<T> {
    fn default() -> Self {
        Self::new()
    }
}
