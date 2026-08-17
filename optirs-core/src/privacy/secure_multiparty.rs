//! Secure Multi-Party Computation (SMPC) building blocks for privacy-preserving
//! optimization.
//!
//! # Security status — read this before using anything in this module
//!
//! Two constructions in this module are real:
//!
//! * [`ShamirSecretSharing`] — genuine `k`-of-`n` threshold sharing over the prime
//!   field `F_p` with `p = 2^127 - 1`. Coefficients are sampled uniformly from the
//!   whole field with an OS-seeded ChaCha12 generator and values are mapped into the
//!   field by fixed-point quantisation (see [`FIXED_POINT_BITS`]).
//! * [`CommitmentScheme`] — a hash commitment `SHA-256(domain || nonce || value)`
//!   with a fresh 32-byte nonce per commitment. It is binding under collision
//!   resistance and hiding as long as the nonce stays secret.
//!
//! Everything else is explicitly **not** a cryptographic guarantee:
//!
//! * [`HomomorphicEngine`] is **not** homomorphic encryption. It produces keyed
//!   SHA-256 digests; [`HomomorphicEngine::decrypt`] and
//!   [`HomomorphicEngine::add_encrypted`] therefore return an error instead of
//!   fabricating a plaintext.
//! * [`ComputationDigestSystem`] is **not** a zero-knowledge proof system. It
//!   produces a publicly recomputable integrity digest that reveals nothing only
//!   because it is never given a witness, and it has no soundness against a
//!   malicious prover. The zero-knowledge entry points return an error.
//! * [`SMPCCoordinator`] simulates every party inside a single process: it creates
//!   and reconstructs all shares itself, so it provides **no privacy against the
//!   coordinator**. Protocol variants other than [`SMPCProtocol::FederatedSMPC`],
//!   malicious-adversary security models, homomorphic encryption and zero-knowledge
//!   proofs are rejected at construction time rather than silently ignored.
//!
//! [`SMPCSecurityGuarantees`] reports what actually executed, including the list of
//! limitations above, so callers and auditors are never handed an unearned claim.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::{rngs::StdRng, thread_rng, Random, Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Randomness
// ---------------------------------------------------------------------------

/// Generator used for every piece of secret material in this module.
///
/// `StdRng` is ChaCha12-backed (a CSPRNG) and `Send`, unlike the thread-local
/// generator, so it can be stored inside the `Send + Sync` types below.
type SecureRng = Random<StdRng>;

/// Create a generator seeded from OS entropy.
///
/// Every instance of every type in this module gets its own seed; nothing in this
/// module uses a compile-time constant seed unless the caller explicitly asks for a
/// deterministic `*_with_seed` constructor (intended for tests only).
fn os_seeded_rng() -> SecureRng {
    SeedableRng::from_rng(&mut thread_rng())
}

/// Draw `len` uniformly random bytes.
fn random_bytes(rng: &mut SecureRng, len: usize) -> Vec<u8> {
    let mut buffer = vec![0u8; len];
    rng.fill_bytes(&mut buffer);
    buffer
}

/// Constant-time comparison of two byte strings.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut difference = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        difference |= x ^ y;
    }
    difference == 0
}

/// Hash a vector of values with a domain separator and a salt.
///
/// The element count and the salt length are hashed as explicit length prefixes so
/// that no two distinct `(salt, values)` pairs share a preimage. Values are hashed
/// through their IEEE-754 little-endian representation, so `0.0` and `-0.0` hash to
/// different digests.
fn hash_values<T: Float + Debug + Send + Sync + 'static>(
    domain: &[u8],
    salt: &[u8],
    values: &Array1<T>,
) -> Result<Vec<u8>> {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((salt.len() as u64).to_le_bytes());
    hasher.update(salt);
    hasher.update((values.len() as u64).to_le_bytes());
    for &value in values.iter() {
        let as_f64 = value.to_f64().ok_or_else(|| {
            OptimError::InvalidConfig("value cannot be converted to f64 for hashing".to_string())
        })?;
        hasher.update(as_f64.to_le_bytes());
    }
    Ok(hasher.finalize().to_vec())
}

// ---------------------------------------------------------------------------
// Prime field arithmetic (F_p with p = 2^127 - 1)
// ---------------------------------------------------------------------------

/// Prime modulus of the secret-sharing field: the Mersenne prime `2^127 - 1`.
pub const SHAMIR_PRIME: u128 = (1u128 << 127) - 1;

/// Number of fractional bits used when mapping a floating point value into the field.
///
/// A value `x` is represented by `round(x * 2^FIXED_POINT_BITS) mod p`. The absolute
/// quantisation error is therefore bounded by `2^-61`, and the round-trip error for
/// an `f64` input is bounded by the larger of `2^-61` and the `f64` rounding error of
/// `x * 2^60` (i.e. a relative error of about `2^-53`).
pub const FIXED_POINT_BITS: u32 = 60;

/// `2^FIXED_POINT_BITS` as an `f64`.
const FIXED_POINT_SCALE: f64 = 1_152_921_504_606_846_976.0;

/// Largest field element interpreted as a non-negative value.
///
/// Elements above this limit represent negative values (`element - p`).
const FIELD_POSITIVE_LIMIT: u128 = (SHAMIR_PRIME - 1) / 2;

/// Largest magnitude that [`FIXED_POINT_BITS`] quantisation can represent (`2^66`).
pub fn max_representable_magnitude() -> f64 {
    FIELD_POSITIVE_LIMIT as f64 / FIXED_POINT_SCALE
}

/// Modular addition in `F_p`.
#[inline]
fn add_mod(a: u128, b: u128) -> u128 {
    // a, b < p < 2^127 so the sum cannot overflow a u128.
    let sum = a + b;
    if sum >= SHAMIR_PRIME {
        sum - SHAMIR_PRIME
    } else {
        sum
    }
}

/// Modular subtraction in `F_p`.
#[inline]
fn sub_mod(a: u128, b: u128) -> u128 {
    if a >= b {
        a - b
    } else {
        SHAMIR_PRIME - (b - a)
    }
}

/// Modular negation in `F_p`.
#[inline]
fn neg_mod(a: u128) -> u128 {
    if a == 0 {
        0
    } else {
        SHAMIR_PRIME - a
    }
}

/// Modular multiplication in `F_p`.
///
/// `p` needs 127 bits, so a plain `u128` product overflows. Operands that both fit in
/// 63 bits take the direct path; otherwise a double-and-add ("Russian peasant")
/// multiplication is used, which is exact for any 127-bit operands at the cost of one
/// modular addition per bit.
fn mul_mod(a: u128, b: u128) -> u128 {
    let mut multiplicand = a % SHAMIR_PRIME;
    let mut multiplier = b % SHAMIR_PRIME;

    if multiplicand < (1u128 << 63) && multiplier < (1u128 << 63) {
        return (multiplicand * multiplier) % SHAMIR_PRIME;
    }

    let mut result = 0u128;
    while multiplier > 0 {
        if multiplier & 1 == 1 {
            result = add_mod(result, multiplicand);
        }
        multiplicand = add_mod(multiplicand, multiplicand);
        multiplier >>= 1;
    }
    result
}

/// Modular exponentiation in `F_p`.
fn pow_mod(base: u128, exponent: u128) -> u128 {
    let mut result = 1u128;
    let mut acc = base % SHAMIR_PRIME;
    let mut remaining = exponent;

    while remaining > 0 {
        if remaining & 1 == 1 {
            result = mul_mod(result, acc);
        }
        acc = mul_mod(acc, acc);
        remaining >>= 1;
    }
    result
}

/// Modular inverse in `F_p` via Fermat's little theorem (`a^(p-2)`).
///
/// Returns an error for `a = 0`, which is what a duplicated Lagrange x-coordinate
/// would produce.
fn inv_mod(a: u128) -> Result<u128> {
    if a % SHAMIR_PRIME == 0 {
        return Err(OptimError::InvalidConfig(
            "cannot invert zero in the Shamir field (duplicate share x-coordinate?)".to_string(),
        ));
    }
    Ok(pow_mod(a, SHAMIR_PRIME - 2))
}

/// Quantise a value into a field element.
fn value_to_field<T: Float + Debug + Send + Sync + 'static>(value: T) -> Result<u128> {
    let as_f64 = value.to_f64().ok_or_else(|| {
        OptimError::InvalidConfig("value cannot be converted to f64 for sharing".to_string())
    })?;
    if !as_f64.is_finite() {
        return Err(OptimError::InvalidConfig(
            "cannot secret-share a non-finite value".to_string(),
        ));
    }

    let scaled = (as_f64 * FIXED_POINT_SCALE).round();
    if scaled.abs() >= FIELD_POSITIVE_LIMIT as f64 {
        return Err(OptimError::InvalidConfig(format!(
            "value {} exceeds the representable magnitude {:e} of the Shamir field",
            as_f64,
            max_representable_magnitude()
        )));
    }

    let quantised = scaled as i128;
    Ok(if quantised < 0 {
        SHAMIR_PRIME - quantised.unsigned_abs()
    } else {
        quantised as u128
    })
}

/// De-quantise a field element back into a value.
fn field_to_value<T: Float + Debug + Send + Sync + 'static>(element: u128) -> Result<T> {
    let reduced = element % SHAMIR_PRIME;
    let signed = if reduced > FIELD_POSITIVE_LIMIT {
        -((SHAMIR_PRIME - reduced) as f64)
    } else {
        reduced as f64
    };
    T::from(signed / FIXED_POINT_SCALE).ok_or_else(|| {
        OptimError::InvalidConfig("reconstructed value is not representable in T".to_string())
    })
}

/// Evaluate a polynomial given by its coefficients (lowest degree first) at `x`.
fn evaluate_polynomial(coefficients: &[u128], x: u128) -> u128 {
    let mut accumulator = 0u128;
    for &coefficient in coefficients.iter().rev() {
        accumulator = add_mod(mul_mod(accumulator, x), coefficient % SHAMIR_PRIME);
    }
    accumulator
}

// ---------------------------------------------------------------------------
// Shamir secret sharing
// ---------------------------------------------------------------------------

/// A single Shamir share: the evaluation point and the field element `P(x)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Share {
    /// Evaluation point. Always `>= 1`; `x = 0` is the secret itself.
    pub x: usize,

    /// Field element `P(x) mod SHAMIR_PRIME`.
    pub y: u128,
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
    threshold: usize,

    /// Number of shares produced (`n`).
    num_shares: usize,

    /// Prime field modulus used for all arithmetic.
    prime_field: u128,

    /// Generator used for polynomial coefficients.
    rng: SecureRng,

    /// Marker for the value type handled by this instance.
    _phantom: PhantomData<T>,
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

    fn validate_parameters(threshold: usize, num_shares: usize) -> Result<()> {
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
    fn random_field_element(&mut self) -> u128 {
        loop {
            let high = self.rng.next_u64() as u128;
            let low = self.rng.next_u64() as u128;
            // Keep 127 bits, which covers [0, p]; the single out-of-range value p is
            // rejected so the result is exactly uniform over [0, p).
            let candidate = ((high << 64) | low) & ((1u128 << 127) - 1);
            if candidate < self.prime_field {
                return candidate;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Commitments
// ---------------------------------------------------------------------------

const COMMITMENT_DOMAIN: &[u8] = b"optirs.smpc.commitment.v1";
const VERIFICATION_DOMAIN: &[u8] = b"optirs.smpc.verification-tag.v1";
const AGGREGATE_DOMAIN: &[u8] = b"optirs.smpc.aggregate-digest.v1";
const VALUE_DIGEST_DOMAIN: &[u8] = b"optirs.smpc.value-digest.v1";
const COMPUTATION_DOMAIN: &[u8] = b"optirs.smpc.computation-digest.v1";

/// Length of a commitment nonce in bytes.
pub const COMMITMENT_NONCE_LEN: usize = 32;

/// Length of a SHA-256 digest in bytes.
pub const DIGEST_LEN: usize = 32;

/// Blinding factor of a single commitment.
///
/// The nonce must be kept secret until the commitment is opened: it is what makes the
/// commitment hiding. Its `Debug` representation is redacted for that reason.
#[derive(Clone, PartialEq, Eq)]
pub struct CommitmentNonce([u8; COMMITMENT_NONCE_LEN]);

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

impl Debug for CommitmentNonce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CommitmentNonce(<redacted>)")
    }
}

/// Hash commitment scheme with per-commitment blinding.
///
/// `commit` returns `(SHA-256(domain || nonce || len || value), nonce)`. It is
/// *binding* under the collision resistance of SHA-256 and *hiding* as long as the
/// 32-byte nonce, drawn from an OS-seeded CSPRNG, stays secret. Committing to the
/// same value twice yields two different commitments.
pub struct CommitmentScheme<T: Float + Debug + Send + Sync + 'static> {
    /// Generator used for commitment nonces.
    rng: SecureRng,

    /// Marker for the committed value type.
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CommitmentScheme<T> {
    fn default() -> Self {
        Self::new()
    }
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

// ---------------------------------------------------------------------------
// Verification tags
// ---------------------------------------------------------------------------

/// Keyed integrity tag over an aggregation result.
///
/// The key is generated per instance from OS entropy and never leaves the process,
/// so the tag is only verifiable by the instance that produced it. It detects
/// accidental corruption of an aggregate held in memory or on disk; it is **not** a
/// proof of correct aggregation and cannot be checked by a third party.
pub struct VerificationParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Secret key mixed into every tag.
    verification_key: Vec<u8>,

    /// Marker for the aggregate value type.
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for VerificationParameters<T> {
    fn default() -> Self {
        Self::new()
    }
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

// ---------------------------------------------------------------------------
// Configuration and participants
// ---------------------------------------------------------------------------

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

/// Reject configurations whose security model is not implemented.
fn require_supported_security(security: CommunicationSecurity) -> Result<()> {
    match security {
        CommunicationSecurity::SemiHonest => Ok(()),
        CommunicationSecurity::MaliciousAbort | CommunicationSecurity::MaliciousGuaranteed => {
            Err(OptimError::UnsupportedOperation(
                "malicious-adversary SMPC is not implemented: no authenticated channels, \
                 signatures or verifiable secret sharing exist in this module"
                    .to_string(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Cryptographic aggregator
// ---------------------------------------------------------------------------

/// Aggregator that opens participant commitments before averaging their inputs.
pub struct CryptographicAggregator<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration
    config: SMPCConfig,

    /// Commitment scheme used for the aggregate digest and for opening participants.
    commitment_scheme: CommitmentScheme<T>,

    /// Verification parameters
    verification_params: VerificationParameters<T>,

    /// Aggregation proofs produced so far
    aggregation_proofs: Vec<AggregationProof<T>>,
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
    fn select_honest_participants(
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
    fn aggregate_honest_inputs(
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
    fn generate_aggregation_proof(
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
            // A participant that publishes no commitment cannot be checked at all.
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

// ---------------------------------------------------------------------------
// Keyed value digests (NOT homomorphic encryption)
// ---------------------------------------------------------------------------

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
    params: HomomorphicParameters<T>,

    /// Per-instance random key mixed into every digest.
    digest_key: Vec<u8>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for HomomorphicEngine<T> {
    fn default() -> Self {
        Self::new()
    }
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
        // Validate first so a malformed value reports the malformed value rather than
        // the (also fatal) unimplemented-primitive error, and never panics.
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
    fn digest_value(&self, value: T) -> Result<Vec<u8>> {
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

fn unimplemented_homomorphic(operation: &str) -> OptimError {
    OptimError::UnsupportedOperation(format!(
        "HomomorphicEngine {operation} is not homomorphic encryption — unimplemented, \
         do not use for confidentiality; use privacy::secure_aggregation for additive \
         aggregation instead"
    ))
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
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for HomomorphicParameters<T> {
    fn default() -> Self {
        Self::new()
    }
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

// ---------------------------------------------------------------------------
// Computation digests (NOT zero-knowledge proofs)
// ---------------------------------------------------------------------------

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
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ComputationDigest<T> {
    /// The digest bytes.
    pub fn digest(&self) -> &[u8] {
        &self.digest
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
    _phantom: PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ComputationDigestSystem<T> {
    fn default() -> Self {
        Self::new()
    }
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

fn unimplemented_zero_knowledge(operation: &str) -> OptimError {
    OptimError::UnsupportedOperation(format!(
        "zero-knowledge {operation} is not cryptographically secure — unimplemented; \
         `ComputationDigestSystem` only provides a non-hiding integrity digest via \
         digest_computation/verify_digest"
    ))
}

// ---------------------------------------------------------------------------
// Aggregation records
// ---------------------------------------------------------------------------

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
    _phantom: PhantomData<T>,
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

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

/// Secure Multi-Party Computation coordinator.
///
/// # WARNING: Single-process simulation
///
/// The coordinator generates every share and reconstructs every output itself, so it
/// sees all secret material. It is useful for exercising the protocol shape and for
/// deterministic tests, not for protecting inputs from the coordinator.
pub struct SMPCCoordinator<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for SMPC protocols
    config: SMPCConfig,

    /// Shamir secret sharing engine
    secret_sharing: ShamirSecretSharing<T>,

    /// Secure aggregation with commitment opening
    secure_aggregator: CryptographicAggregator<T>,

    /// Computation digest system
    digest_system: ComputationDigestSystem<T>,

    /// Participant management
    participants: HashMap<String, Participant>,

    /// Current protocol state
    protocol_state: SMPCProtocolState,
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

    fn execute_smpc_inner(
        &mut self,
        participant_inputs: &HashMap<String, Array1<T>>,
        computation: &SMPCComputation,
    ) -> Result<SMPCResult<T>> {
        // Phase 1: setup and validation.
        self.protocol_state = SMPCProtocolState::Setup;
        self.verify_participants()?;
        let dimension = self.validate_inputs(participant_inputs)?;

        // Phase 2: input sharing, one share vector per coordinate.
        self.protocol_state = SMPCProtocolState::InputSharing;
        let shared_inputs = self.share_inputs(participant_inputs)?;

        // Phase 3: computation on the shares.
        self.protocol_state = SMPCProtocolState::Computation;
        let (combined_shares, divisor) =
            self.perform_secure_computation(&shared_inputs, computation, dimension)?;

        // Phase 4: reconstruct every coordinate.
        self.protocol_state = SMPCProtocolState::OutputReconstruction;
        let reconstructed = self.reconstruct_output(&combined_shares)?;
        let result = reconstructed / divisor;

        // Phase 5: integrity digest (not a proof - see `ComputationDigest`).
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
    fn verify_participants(&self) -> Result<()> {
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
    fn validate_inputs(&self, inputs: &HashMap<String, Array1<T>>) -> Result<usize> {
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
    fn share_inputs(
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
    fn perform_secure_computation(
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
    fn secure_sum(
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
    fn reconstruct_output(&self, columns: &[Vec<Share>]) -> Result<Array1<T>> {
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
    fn combine_inputs(&self, inputs: &HashMap<String, Array1<T>>) -> Array1<T> {
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
    fn get_security_guarantees(&self) -> SMPCSecurityGuarantees {
        SMPCSecurityGuarantees {
            protocol_variant: self.config.protocol_variant,
            communication_security: self.config.communication_security,
            // No malicious-security mechanism runs, so nothing is tolerated,
            // regardless of what the configuration asked for.
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn test_config(num_participants: usize, threshold: usize) -> SMPCConfig {
        SMPCConfig {
            num_participants,
            threshold,
            security_parameter: 128,
            enable_homomorphic: false,
            enable_zk_proofs: false,
            protocol_variant: SMPCProtocol::FederatedSMPC,
            communication_security: CommunicationSecurity::SemiHonest,
            malicious_tolerance: MaliciousTolerance {
                max_corrupted: 1,
                byzantine_tolerance: true,
                verification_threshold: 0.8,
                commit_and_prove: true,
            },
        }
    }

    fn coordinator_with_participants(
        num_participants: usize,
        threshold: usize,
        ids: &[&str],
    ) -> SMPCCoordinator<f64> {
        let mut coordinator = SMPCCoordinator::<f64>::new(test_config(num_participants, threshold))
            .expect("coordinator construction failed");
        for id in ids {
            coordinator
                .add_participant(Participant::new(*id, vec![0u8; 4], 1.0))
                .expect("participant registration failed");
        }
        coordinator
    }

    // ---------------------------------------------------------------- field

    #[test]
    fn test_fixed_point_scale_constant() {
        assert_eq!(FIXED_POINT_SCALE, 2f64.powi(FIXED_POINT_BITS as i32));
        assert_eq!(SHAMIR_PRIME, (1u128 << 127) - 1);
    }

    #[test]
    fn test_field_arithmetic() {
        let a = SHAMIR_PRIME - 5;
        let b = 12u128;

        assert_eq!(add_mod(a, b), 7);
        assert_eq!(sub_mod(b, a), 17);
        assert_eq!(add_mod(a, neg_mod(a)), 0);

        // Large operands exercise the double-and-add path.
        let big = SHAMIR_PRIME - 1; // == -1 mod p
        assert_eq!(mul_mod(big, big), 1);
        assert_eq!(mul_mod(big, 1), big);
        assert_eq!(mul_mod(0, big), 0);

        let inverse = inv_mod(big).expect("inverse of -1 exists");
        assert_eq!(mul_mod(big, inverse), 1);

        let inverse_small = inv_mod(3).expect("inverse of 3 exists");
        assert_eq!(mul_mod(3, inverse_small), 1);

        assert!(inv_mod(0).is_err());
    }

    #[test]
    fn test_quantisation_round_trip() {
        for value in [0.0f64, 1.0, -1.0, 42.0, -3.75, 0.1, 1234.5678, -1e6] {
            let element = value_to_field(value).expect("quantisation failed");
            let restored: f64 = field_to_value(element).expect("de-quantisation failed");
            assert!(
                (restored - value).abs() <= 1e-12 + value.abs() * 1e-15,
                "round trip failed for {value}: got {restored}"
            );
        }
    }

    #[test]
    fn test_quantisation_rejects_out_of_range_and_non_finite() {
        assert!(value_to_field(f64::NAN).is_err());
        assert!(value_to_field(f64::INFINITY).is_err());
        assert!(value_to_field(max_representable_magnitude() * 2.0).is_err());
    }

    // ------------------------------------------------------- secret sharing

    #[test]
    fn test_secret_sharing_round_trip_any_subset() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let secret = 42.0;

        let shares = secret_sharing.share_secret(secret).expect("sharing failed");
        assert_eq!(shares.len(), 5);

        // Prefix subset.
        let reconstructed = secret_sharing
            .reconstruct_secret(&shares[0..3])
            .expect("reconstruction failed");
        assert!((reconstructed - secret).abs() < 1e-10);

        // Non-prefix subset: only correct Lagrange interpolation over the field
        // recovers the secret here.
        let reconstructed_tail = secret_sharing
            .reconstruct_secret(&shares[2..5])
            .expect("reconstruction failed");
        assert!((reconstructed_tail - secret).abs() < 1e-10);

        // Arbitrary out-of-order subset.
        let mixed = vec![shares[4], shares[0], shares[3]];
        let reconstructed_mixed = secret_sharing
            .reconstruct_secret(&mixed)
            .expect("reconstruction failed");
        assert!((reconstructed_mixed - secret).abs() < 1e-10);
    }

    #[test]
    fn test_secret_sharing_handles_negative_and_fractional_secrets() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(2, 4).expect("valid sharing parameters");

        for secret in [-7.25f64, 0.0, 1e-6, 1024.5] {
            let shares = secret_sharing.share_secret(secret).expect("sharing failed");
            let reconstructed = secret_sharing
                .reconstruct_secret(&shares[1..3])
                .expect("reconstruction failed");
            assert!(
                (reconstructed - secret).abs() < 1e-10,
                "failed for {secret}: got {reconstructed}"
            );
        }
    }

    #[test]
    fn test_secret_sharing_requires_threshold_shares() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let shares = secret_sharing.share_secret(1.5).expect("sharing failed");
        assert!(secret_sharing.reconstruct_secret(&shares[0..2]).is_err());
    }

    #[test]
    fn test_secret_sharing_rejects_invalid_parameters() {
        assert!(ShamirSecretSharing::<f64>::new(3, 2).is_err());
        assert!(ShamirSecretSharing::<f64>::new(0, 5).is_err());
        assert!(ShamirSecretSharing::<f64>::new(1, 0).is_err());
    }

    #[test]
    fn test_secret_sharing_rejects_duplicate_x_coordinates() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(2, 3).expect("valid sharing parameters");
        let shares = secret_sharing.share_secret(3.0).expect("sharing failed");
        let duplicated = vec![shares[0], shares[0]];
        assert!(secret_sharing.reconstruct_secret(&duplicated).is_err());
    }

    #[test]
    fn test_secret_sharing_coefficients_are_fresh_per_call() {
        // Previously every call re-created Random::seed(42), making the coefficients
        // compile-time constants; sharing the same secret twice produced identical
        // shares and one share revealed the secret.
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let first = secret_sharing.share_secret(42.0).expect("sharing failed");
        let second = secret_sharing.share_secret(42.0).expect("sharing failed");

        assert!(
            first.iter().zip(second.iter()).any(|(a, b)| a.y != b.y),
            "shares of the same secret must not repeat across calls"
        );

        // Two independently constructed instances must differ as well.
        let mut other = ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let third = other.share_secret(42.0).expect("sharing failed");
        assert!(first.iter().zip(third.iter()).any(|(a, b)| a.y != b.y));
    }

    #[test]
    fn test_shares_do_not_reveal_the_secret_magnitude() {
        // Coefficients are uniform over the whole field, so an individual share is a
        // uniform field element rather than `secret + small noise`.
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(2, 3).expect("valid sharing parameters");
        let shares = secret_sharing.share_secret(42.0).expect("sharing failed");
        let secret_element = value_to_field(42.0).expect("quantisation failed");

        assert!(shares
            .iter()
            .all(|share| share.y.abs_diff(secret_element) > (1u128 << 100)));
    }

    #[test]
    fn test_secret_sharing_with_seed_is_deterministic() {
        let mut a = ShamirSecretSharing::<f64>::with_seed(3, 5, 7).expect("valid parameters");
        let mut b = ShamirSecretSharing::<f64>::with_seed(3, 5, 7).expect("valid parameters");
        assert_eq!(
            a.share_secret(2.5).expect("sharing failed"),
            b.share_secret(2.5).expect("sharing failed")
        );
    }

    // ----------------------------------------------------------- commitment

    #[test]
    fn test_commitment_scheme_is_hiding_and_binding() {
        let mut scheme = CommitmentScheme::<f64>::new();
        let data = Array1::from(vec![1.0, 2.0, 3.0]);

        let (commitment1, nonce1) = scheme.commit(&data).expect("commit failed");
        let (commitment2, nonce2) = scheme.commit(&data).expect("commit failed");

        // Hiding: two commitments to the same value must differ (the previous
        // implementation asserted the opposite).
        assert_ne!(commitment1, commitment2);
        assert_ne!(nonce1, nonce2);

        // Each commitment opens with its own nonce.
        assert!(scheme
            .open(&commitment1, &data, &nonce1)
            .expect("open failed"));
        assert!(scheme
            .open(&commitment2, &data, &nonce2)
            .expect("open failed"));

        // Binding: neither a different value nor a different nonce opens it.
        let different = Array1::from(vec![1.0, 2.0, 4.0]);
        assert!(!scheme
            .open(&commitment1, &different, &nonce1)
            .expect("open failed"));
        assert!(!scheme
            .open(&commitment1, &data, &nonce2)
            .expect("open failed"));
    }

    #[test]
    fn test_commitment_nonce_debug_is_redacted() {
        let nonce = CommitmentNonce::from_bytes([7u8; COMMITMENT_NONCE_LEN]);
        assert_eq!(format!("{nonce:?}"), "CommitmentNonce(<redacted>)");
        assert_eq!(nonce.as_bytes()[0], 7);
    }

    #[test]
    fn test_key_material_is_not_a_compile_time_constant() {
        // Every key used to live behind `Random::seed(42)`, so two instances in two
        // processes shared identical key material.
        let data = Array1::from(vec![1.0, 2.0, 3.0]);

        let first_engine = HomomorphicEngine::<f64>::new();
        let second_engine = HomomorphicEngine::<f64>::new();
        assert_ne!(
            first_engine.encrypt(&data).expect("digesting failed").data,
            second_engine.encrypt(&data).expect("digesting failed").data
        );

        let first_params = VerificationParameters::<f64>::new();
        let second_params = VerificationParameters::<f64>::new();
        assert_ne!(
            first_params
                .generate_verification_data(&data)
                .expect("tag generation failed"),
            second_params
                .generate_verification_data(&data)
                .expect("tag generation failed")
        );

        let mut first_scheme = CommitmentScheme::<f64>::with_seed(1);
        let mut second_scheme = CommitmentScheme::<f64>::with_seed(2);
        assert_ne!(
            first_scheme.commit(&data).expect("commit failed").0,
            second_scheme.commit(&data).expect("commit failed").0
        );
    }

    #[test]
    fn test_verification_tag_round_trip() {
        let params = VerificationParameters::<f64>::new();
        let aggregate = Array1::from(vec![1.0, 2.0]);
        let tag = params
            .generate_verification_data(&aggregate)
            .expect("tag generation failed");

        assert!(params
            .verify_verification_data(&aggregate, &tag)
            .expect("tag verification failed"));
        assert!(!params
            .verify_verification_data(&Array1::from(vec![1.0, 2.5]), &tag)
            .expect("tag verification failed"));
    }

    // --------------------------------------------------------- participants

    #[test]
    fn test_verify_participant_honesty_requires_a_valid_opening() {
        let config = test_config(3, 2);
        let aggregator = CryptographicAggregator::<f64>::new(config);
        let mut scheme = CommitmentScheme::<f64>::new();
        let input = Array1::from(vec![1.0, 2.0]);
        let (commitment, nonce) = scheme.commit(&input).expect("commit failed");

        // No commitment at all -> rejected (self-attestation is not verification).
        let bare = Participant::new("p0", vec![], 1.0);
        assert!(!aggregator
            .verify_participant_honesty(&bare, &input)
            .expect("verification failed"));

        // Commitment without opening nonce -> rejected.
        let mut no_nonce = Participant::new("p1", vec![], 1.0);
        no_nonce.commitment = Some(commitment.clone());
        assert!(!aggregator
            .verify_participant_honesty(&no_nonce, &input)
            .expect("verification failed"));

        // Correct opening -> accepted.
        let honest =
            Participant::new("p2", vec![], 1.0).with_commitment(commitment.clone(), nonce.clone());
        assert!(aggregator
            .verify_participant_honesty(&honest, &input)
            .expect("verification failed"));

        // Same commitment, different submitted input -> rejected.
        let tampered = Array1::from(vec![1.0, 9.0]);
        assert!(!aggregator
            .verify_participant_honesty(&honest, &tampered)
            .expect("verification failed"));

        // Flagged participants are rejected regardless of the opening.
        let mut suspicious = honest.clone();
        suspicious.status = ParticipantStatus::Suspicious;
        assert!(!aggregator
            .verify_participant_honesty(&suspicious, &input)
            .expect("verification failed"));

        // Trust score below the configured threshold -> rejected.
        let mut distrusted = honest;
        distrusted.trust_score = 0.1;
        assert!(!aggregator
            .verify_participant_honesty(&distrusted, &input)
            .expect("verification failed"));
    }

    #[test]
    fn test_secure_aggregate_opens_commitments() {
        let config = test_config(3, 2);
        let mut aggregator = CryptographicAggregator::<f64>::new(config);
        let mut scheme = CommitmentScheme::<f64>::new();

        let mut inputs = HashMap::new();
        let mut participants = HashMap::new();
        for (index, id) in ["a", "b", "c"].iter().enumerate() {
            let input = Array1::from(vec![index as f64, 1.0]);
            let (commitment, nonce) = scheme.commit(&input).expect("commit failed");
            participants.insert(
                (*id).to_string(),
                Participant::new(*id, vec![], 1.0).with_commitment(commitment, nonce),
            );
            inputs.insert((*id).to_string(), input);
        }

        let result = aggregator
            .secure_aggregate(&inputs, &participants)
            .expect("aggregation failed");
        assert_eq!(result.honest_participants.len(), 3);
        assert_eq!(result.aggregate.len(), 2);
        assert!((result.aggregate[0] - 1.0).abs() < 1e-12);
        assert!((result.aggregate[1] - 1.0).abs() < 1e-12);
        assert_eq!(aggregator.aggregation_proofs().len(), 1);
        assert!(aggregator
            .verification_params()
            .verify_verification_data(&result.aggregate, &result.proof.verification_data)
            .expect("tag verification failed"));

        // Dropping the opening of one participant drops them from the aggregate.
        if let Some(participant) = participants.get_mut("c") {
            participant.commitment = None;
            participant.commitment_nonce = None;
        }
        let filtered = aggregator
            .secure_aggregate(&inputs, &participants)
            .expect("aggregation failed");
        assert_eq!(filtered.honest_participants, vec!["a", "b"]);
    }

    #[test]
    fn test_secure_aggregate_rejects_malicious_security_models() {
        let mut config = test_config(3, 2);
        config.communication_security = CommunicationSecurity::MaliciousAbort;
        let mut aggregator = CryptographicAggregator::<f64>::new(config);
        assert!(aggregator
            .secure_aggregate(&HashMap::new(), &HashMap::new())
            .is_err());
    }

    #[test]
    fn test_secure_aggregate_rejects_dimension_mismatch() {
        let config = test_config(2, 1);
        let mut aggregator = CryptographicAggregator::<f64>::new(config);
        let mut scheme = CommitmentScheme::<f64>::new();

        let mut inputs = HashMap::new();
        let mut participants = HashMap::new();
        for (id, values) in [("a", vec![1.0, 2.0]), ("b", vec![1.0])] {
            let input = Array1::from(values);
            let (commitment, nonce) = scheme.commit(&input).expect("commit failed");
            participants.insert(
                id.to_string(),
                Participant::new(id, vec![], 1.0).with_commitment(commitment, nonce),
            );
            inputs.insert(id.to_string(), input);
        }

        assert!(aggregator.secure_aggregate(&inputs, &participants).is_err());
    }

    // ---------------------------------------------------- "homomorphic" API

    #[test]
    fn test_homomorphic_engine_is_not_encryption() {
        let engine = HomomorphicEngine::<f64>::new();
        let data1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let data2 = Array1::from(vec![4.0, 5.0, 6.0]);

        let digests1 = engine.encrypt(&data1).expect("digesting failed");
        let digests2 = engine.encrypt(&data2).expect("digesting failed");
        assert_eq!(digests1.len(), 3);
        assert!(digests1.data.iter().all(|block| block.len() == DIGEST_LEN));
        assert_ne!(digests1.data[0], digests2.data[0]);

        // Decryption and homomorphic addition are unimplemented and say so instead of
        // returning garbage (the old implementation reinterpreted hash bytes as f64).
        let decrypt_error = engine
            .decrypt(&digests1)
            .expect_err("decryption must not succeed");
        assert!(matches!(decrypt_error, OptimError::UnsupportedOperation(_)));
        assert!(format!("{decrypt_error}").contains("not homomorphic encryption"));

        let add_error = engine
            .add_encrypted(&digests1, &digests2)
            .expect_err("homomorphic addition must not succeed");
        assert!(matches!(add_error, OptimError::UnsupportedOperation(_)));
    }

    #[test]
    fn test_homomorphic_ciphertext_rejects_short_blocks_without_panicking() {
        let engine = HomomorphicEngine::<f64>::new();
        let malformed = HomomorphicCiphertext::<f64> {
            data: vec![vec![0u8; 4]],
            params: HomomorphicParameters::new(),
        };

        assert!(malformed.validate().is_err());
        let error = engine
            .decrypt(&malformed)
            .expect_err("short block must be rejected");
        assert!(format!("{error}").contains("digest block 0"));
    }

    #[test]
    fn test_homomorphic_engine_with_seed_is_deterministic() {
        let a = HomomorphicEngine::<f64>::with_seed(11);
        let b = HomomorphicEngine::<f64>::with_seed(11);
        let data = Array1::from(vec![1.0]);
        assert_eq!(
            a.encrypt(&data).expect("digesting failed").data,
            b.encrypt(&data).expect("digesting failed").data
        );
        assert_eq!(a.params().security_level, 128);
    }

    // ---------------------------------------------------- computation digest

    #[test]
    fn test_computation_digest_verifies_and_zk_entry_points_fail() {
        let system = ComputationDigestSystem::<f64>::new();
        let input = Array1::from(vec![1.0, 2.0]);
        let output = Array1::from(vec![3.0]);

        let digest = system
            .digest_computation(&input, &output, "sum")
            .expect("digest failed");
        assert!(system
            .verify_digest(&digest, &input, &output, "sum")
            .expect("verification failed"));

        // Any change to statement, input or output is detected.
        assert!(!system
            .verify_digest(&digest, &input, &Array1::from(vec![4.0]), "sum")
            .expect("verification failed"));
        assert!(!system
            .verify_digest(&digest, &Array1::from(vec![1.0, 2.5]), &output, "sum")
            .expect("verification failed"));

        // The digest never carries a witness derived from the plaintext inputs.
        assert_eq!(digest.digest().len(), DIGEST_LEN);

        // The zero-knowledge entry points refuse instead of accepting forgeries: the
        // old verify_proof returned true for any non-empty proof under a public CRS.
        assert!(system.prove_computation(&input, &output, "sum").is_err());
        assert!(system.verify_proof(&digest).is_err());
    }

    // ------------------------------------------------------------ coordinator

    #[test]
    fn test_smpc_config() {
        let config = test_config(5, 3);
        assert_eq!(config.num_participants, 5);
        assert_eq!(config.threshold, 3);
        assert!(!config.enable_homomorphic);
    }

    #[test]
    fn test_coordinator_rejects_unimplemented_features() {
        let mut homomorphic = test_config(3, 2);
        homomorphic.enable_homomorphic = true;
        assert!(SMPCCoordinator::<f64>::new(homomorphic).is_err());

        let mut zk = test_config(3, 2);
        zk.enable_zk_proofs = true;
        assert!(SMPCCoordinator::<f64>::new(zk).is_err());

        let mut malicious = test_config(3, 2);
        malicious.communication_security = CommunicationSecurity::MaliciousGuaranteed;
        assert!(SMPCCoordinator::<f64>::new(malicious).is_err());

        let mut bgw = test_config(3, 2);
        bgw.protocol_variant = SMPCProtocol::BGW;
        assert!(SMPCCoordinator::<f64>::new(bgw).is_err());
    }

    #[test]
    fn test_coordinator_rejects_threshold_above_participants() {
        assert!(SMPCCoordinator::<f64>::new(test_config(5, 6)).is_err());
        assert!(SMPCCoordinator::<f64>::new(test_config(5, 0)).is_err());
        assert!(SMPCCoordinator::<f64>::new(test_config(0, 0)).is_err());
    }

    #[test]
    fn test_add_participant_limits_and_duplicates() {
        let mut coordinator = coordinator_with_participants(2, 1, &["a", "b"]);
        assert!(coordinator
            .add_participant(Participant::new("a", vec![], 1.0))
            .is_err());
        assert!(coordinator
            .add_participant(Participant::new("c", vec![], 1.0))
            .is_err());
        assert!(coordinator
            .add_participant(Participant::new("", vec![], 1.0))
            .is_err());
        assert_eq!(coordinator.participants().len(), 2);
    }

    #[test]
    fn test_execute_smpc_preserves_input_dimension() {
        // Previously every call flattened all coordinates into one share vector and
        // returned a length-1 array whatever the input dimension was.
        let mut coordinator = coordinator_with_participants(3, 2, &["a", "b", "c"]);

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
        inputs.insert("b".to_string(), Array1::from(vec![0.5, 0.5, 0.5, 0.5]));
        inputs.insert("c".to_string(), Array1::from(vec![-1.0, 0.0, 1.0, 2.0]));

        let result = coordinator
            .execute_smpc(inputs.clone(), SMPCComputation::Sum)
            .expect("smpc execution failed");

        assert_eq!(result.result.len(), 4);
        let expected = [0.5, 2.5, 4.5, 6.5];
        for (index, expected_value) in expected.iter().enumerate() {
            assert!(
                (result.result[index] - expected_value).abs() < 1e-9,
                "coordinate {index}: got {}",
                result.result[index]
            );
        }
        assert_eq!(result.participating_parties, vec!["a", "b", "c"]);
        assert!(matches!(
            coordinator.protocol_state(),
            SMPCProtocolState::Completed
        ));
    }

    #[test]
    fn test_execute_smpc_average() {
        let mut coordinator = coordinator_with_participants(2, 2, &["a", "b"]);

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0, 3.0]));
        inputs.insert("b".to_string(), Array1::from(vec![3.0, 5.0]));

        let result = coordinator
            .execute_smpc(inputs, SMPCComputation::Average)
            .expect("smpc execution failed");

        assert_eq!(result.result.len(), 2);
        assert!((result.result[0] - 2.0).abs() < 1e-9);
        assert!((result.result[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_execute_smpc_validates_inputs_and_aborts() {
        let mut coordinator = coordinator_with_participants(3, 2, &["a", "b"]);

        let mut mismatched = HashMap::new();
        mismatched.insert("a".to_string(), Array1::from(vec![1.0, 2.0]));
        mismatched.insert("b".to_string(), Array1::from(vec![1.0]));
        assert!(coordinator
            .execute_smpc(mismatched, SMPCComputation::Sum)
            .is_err());
        assert!(matches!(
            coordinator.protocol_state(),
            SMPCProtocolState::Aborted(_)
        ));

        let mut unknown = HashMap::new();
        unknown.insert("zz".to_string(), Array1::from(vec![1.0]));
        assert!(coordinator
            .execute_smpc(unknown, SMPCComputation::Sum)
            .is_err());

        let mut empty_input = HashMap::new();
        empty_input.insert("a".to_string(), Array1::from(Vec::<f64>::new()));
        assert!(coordinator
            .execute_smpc(empty_input, SMPCComputation::Sum)
            .is_err());
    }

    #[test]
    fn test_execute_smpc_requires_enough_active_participants() {
        let mut coordinator = coordinator_with_participants(3, 3, &["a", "b", "c"]);
        if let Some(participant) = coordinator.participants.get_mut("c") {
            participant.status = ParticipantStatus::Malicious;
        }

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0]));
        inputs.insert("b".to_string(), Array1::from(vec![1.0]));
        assert!(coordinator
            .execute_smpc(inputs, SMPCComputation::Sum)
            .is_err());
    }

    #[test]
    fn test_unsupported_computations_report_errors() {
        let mut coordinator = coordinator_with_participants(2, 2, &["a", "b"]);
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0]));
        inputs.insert("b".to_string(), Array1::from(vec![2.0]));

        assert!(coordinator
            .execute_smpc(inputs.clone(), SMPCComputation::WeightedSum(vec![0.5, 0.5]))
            .is_err());
        assert!(coordinator
            .execute_smpc(inputs, SMPCComputation::Custom("median".to_string()))
            .is_err());
    }

    #[test]
    fn test_security_guarantees_are_derived_not_asserted() {
        let mut coordinator = coordinator_with_participants(2, 2, &["a", "b"]);
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0]));
        inputs.insert("b".to_string(), Array1::from(vec![2.0]));

        let result = coordinator
            .execute_smpc(inputs, SMPCComputation::Sum)
            .expect("smpc execution failed");
        let guarantees = &result.security_guarantees;

        // The old implementation hardcoded information-theoretic privacy with
        // soundness and completeness whatever ran.
        assert_eq!(guarantees.privacy_level, PrivacyLevel::Computational);
        assert!(!guarantees.soundness);
        assert!(guarantees.completeness);
        assert_eq!(guarantees.malicious_tolerance, 0);
        assert_eq!(
            guarantees.communication_security,
            CommunicationSecurity::SemiHonest
        );
        assert!(!guarantees.limitations.is_empty());
    }

    #[test]
    fn test_coordinator_secure_aggregate_path() {
        let mut coordinator = SMPCCoordinator::<f64>::new(test_config(2, 2))
            .expect("coordinator construction failed");
        let mut scheme = CommitmentScheme::<f64>::new();

        let mut inputs = HashMap::new();
        for id in ["a", "b"] {
            let input = Array1::from(vec![2.0, 4.0]);
            let (commitment, nonce) = scheme.commit(&input).expect("commit failed");
            coordinator
                .add_participant(
                    Participant::new(id, vec![], 1.0).with_commitment(commitment, nonce),
                )
                .expect("participant registration failed");
            inputs.insert(id.to_string(), input);
        }

        let aggregated = coordinator
            .secure_aggregate(&inputs)
            .expect("aggregation failed");
        assert_eq!(aggregated.aggregate.len(), 2);
        assert!((aggregated.aggregate[0] - 2.0).abs() < 1e-12);
        assert_eq!(aggregated.security_level, CommunicationSecurity::SemiHonest);
    }
}
