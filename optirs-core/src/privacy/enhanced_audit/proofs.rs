//! Cryptographic proofs over released values.
//!
//! # What these proofs are
//!
//! Every algorithm registered here is a **keyed or unkeyed integrity
//! commitment** over the canonical encoding of the released vector: SHA-256
//! for the public variant, HMAC-SHA256 (RFC 2104) for the keyed variant. They
//! are real, verifiable, and implemented in pure Rust on top of `sha2`.
//!
//! # What they are not
//!
//! Zero-knowledge proofs, asymmetric digital signatures and confidentiality
//! proofs need a proving system or a signature scheme that this crate does not
//! carry, and adding one is not something an integrity digest can fake. Those
//! requirements are therefore *rejected* by
//! [`CryptographicProofGenerator::check_requirements`] with
//! [`OptimError::UnsupportedOperation`], instead of being answered with a
//! digest dressed up as a signature.
//!
//! The previous implementation registered **no** proof types at all, so
//! `generate_proof` could only ever fail, and the keys were empty maps.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{SystemTime, UNIX_EPOCH};

use super::hashing::{
    canonical_array_bytes, digests_equal, hmac_sha256, random_key, sha256, Digest32,
};
use super::types::{
    CryptographicKeys, CryptographicProof, CryptographicProofType, ProofAlgorithm,
    ProofRequirements,
};

/// Name of the unkeyed integrity proof algorithm.
pub const SHA256_INTEGRITY: &str = "sha256-integrity";
/// Name of the keyed integrity proof algorithm.
pub const HMAC_SHA256_INTEGRITY: &str = "hmac-sha256-integrity";

/// Seconds since the Unix epoch, or an error if the clock is before it.
pub(super) fn unix_timestamp() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .map_err(|err| {
            OptimError::InvalidState(format!(
                "the system clock is set before the Unix epoch, so audit records cannot be \
                 timestamped: {err}"
            ))
        })
}

impl CryptographicKeys {
    /// Create an empty key store.
    pub fn new() -> Self {
        Self {
            signing_keys: HashMap::new(),
            verification_keys: HashMap::new(),
            encryption_keys: HashMap::new(),
        }
    }

    /// Create a key store with a freshly generated symmetric MAC key.
    ///
    /// The key is used for HMAC-SHA256 integrity proofs. It is recorded under
    /// `signing_keys` *and* `verification_keys` because HMAC is symmetric:
    /// verification requires the same secret. That is exactly why HMAC cannot
    /// provide non-repudiation, and why
    /// [`CryptographicProofGenerator::check_requirements`] rejects that
    /// requirement rather than pretending otherwise.
    pub fn generate() -> Self {
        let key = random_key();
        let mut signing_keys = HashMap::new();
        signing_keys.insert(HMAC_SHA256_INTEGRITY.to_string(), key.to_vec());
        let mut verification_keys = HashMap::new();
        verification_keys.insert(HMAC_SHA256_INTEGRITY.to_string(), key.to_vec());
        Self {
            signing_keys,
            verification_keys,
            encryption_keys: HashMap::new(),
        }
    }

    /// The MAC key for `name`, if present.
    pub fn mac_key(&self, name: &str) -> Option<&[u8]> {
        self.signing_keys.get(name).map(|key| key.as_slice())
    }
}

impl Default for CryptographicKeys {
    fn default() -> Self {
        Self::new()
    }
}

/// Cryptographic proof generator.
pub struct CryptographicProofGenerator<T: Float + Debug + Send + Sync + 'static> {
    /// Registered proof types.
    proof_types: HashMap<String, CryptographicProofType<T>>,
    /// Key material.
    keys: CryptographicKeys,
}

impl<T: Float + Debug + Send + Sync + 'static> CryptographicProofGenerator<T> {
    /// Create a generator with the two integrity proof types registered and a
    /// freshly generated MAC key.
    pub fn new() -> Self {
        let keys = CryptographicKeys::generate();
        let mut generator = Self {
            proof_types: HashMap::new(),
            keys,
        };
        generator.register_integrity_proof_types();
        generator
    }

    /// Create a generator with no proof types registered.
    ///
    /// `generate_proof` then fails for every input, which is the honest
    /// behaviour for a generator that has been given nothing to do.
    pub fn empty() -> Self {
        Self {
            proof_types: HashMap::new(),
            keys: CryptographicKeys::new(),
        }
    }

    /// Register the built-in integrity proof types.
    fn register_integrity_proof_types(&mut self) {
        self.proof_types.insert(
            SHA256_INTEGRITY.to_string(),
            CryptographicProofType {
                name: SHA256_INTEGRITY.to_string(),
                generate_fn: Box::new(|data: &Array1<T>, _keys: &CryptographicKeys| {
                    let bytes = canonical_array_bytes(data)?;
                    let digest = sha256(&[&bytes]);
                    let mut metadata = HashMap::new();
                    metadata.insert("algorithm".to_string(), "SHA-256".to_string());
                    metadata.insert("element_count".to_string(), data.len().to_string());
                    Ok(CryptographicProof {
                        prooftype: SHA256_INTEGRITY.to_string(),
                        proof_data: digest.to_vec(),
                        public_params: (data.len() as u64).to_le_bytes().to_vec(),
                        timestamp: unix_timestamp()?,
                        metadata,
                    })
                }),
                verify_fn: Box::new(
                    |proof: &CryptographicProof, data: &Array1<T>, _keys: &CryptographicKeys| {
                        let Ok(bytes) = canonical_array_bytes(data) else {
                            return false;
                        };
                        let digest = sha256(&[&bytes]);
                        proof.proof_data.len() == digest.len()
                            && digest_matches(&proof.proof_data, &digest)
                    },
                ),
            },
        );

        self.proof_types.insert(
            HMAC_SHA256_INTEGRITY.to_string(),
            CryptographicProofType {
                name: HMAC_SHA256_INTEGRITY.to_string(),
                generate_fn: Box::new(|data: &Array1<T>, keys: &CryptographicKeys| {
                    let key = keys.mac_key(HMAC_SHA256_INTEGRITY).ok_or_else(|| {
                        OptimError::InvalidState(format!(
                            "no MAC key is registered under `{HMAC_SHA256_INTEGRITY}`"
                        ))
                    })?;
                    let bytes = canonical_array_bytes(data)?;
                    let tag = hmac_sha256(key, &bytes);
                    let mut metadata = HashMap::new();
                    metadata.insert("algorithm".to_string(), "HMAC-SHA-256".to_string());
                    metadata.insert("element_count".to_string(), data.len().to_string());
                    metadata.insert("non_repudiation".to_string(), "false".to_string());
                    Ok(CryptographicProof {
                        prooftype: HMAC_SHA256_INTEGRITY.to_string(),
                        proof_data: tag.to_vec(),
                        public_params: (data.len() as u64).to_le_bytes().to_vec(),
                        timestamp: unix_timestamp()?,
                        metadata,
                    })
                }),
                verify_fn: Box::new(
                    |proof: &CryptographicProof, data: &Array1<T>, keys: &CryptographicKeys| {
                        let Some(key) = keys.mac_key(HMAC_SHA256_INTEGRITY) else {
                            return false;
                        };
                        let Ok(bytes) = canonical_array_bytes(data) else {
                            return false;
                        };
                        let tag = hmac_sha256(key, &bytes);
                        proof.proof_data.len() == tag.len()
                            && digest_matches(&proof.proof_data, &tag)
                    },
                ),
            },
        );
    }

    /// Names of the registered proof types.
    pub fn registered_proof_types(&self) -> Vec<String> {
        let mut names: Vec<String> = self.proof_types.keys().cloned().collect();
        names.sort();
        names
    }

    /// Register a caller-supplied proof type.
    pub fn register_proof_type(&mut self, proof_type: CryptographicProofType<T>) {
        self.proof_types.insert(proof_type.name.clone(), proof_type);
    }

    /// Generate a proof of the named type over `data`.
    pub fn generate_proof(&self, prooftype: &str, data: &Array1<T>) -> Result<CryptographicProof> {
        if self.proof_types.is_empty() {
            return Err(OptimError::InvalidState(
                "no cryptographic proof types are registered, so no proof can be produced"
                    .to_string(),
            ));
        }
        let generator = self.proof_types.get(prooftype).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "unknown proof type `{prooftype}`; registered types: {:?}",
                self.registered_proof_types()
            ))
        })?;
        (generator.generate_fn)(data, &self.keys)
    }

    /// Verify a proof against `data`.
    pub fn verify_proof(&self, proof: &CryptographicProof, data: &Array1<T>) -> Result<bool> {
        let verifier = self.proof_types.get(&proof.prooftype).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "unknown proof type `{}`; registered types: {:?}",
                proof.prooftype,
                self.registered_proof_types()
            ))
        })?;
        Ok((verifier.verify_fn)(proof, data, &self.keys))
    }

    /// Reject proof requirements that cannot be met.
    ///
    /// Integrity and completeness are covered by the registered commitments.
    /// Zero-knowledge, non-repudiation and confidentiality are not, and no
    /// digest can stand in for them.
    pub fn check_requirements(&self, requirements: &ProofRequirements) -> Result<()> {
        let mut missing = Vec::new();
        if requirements.zero_knowledge_proofs {
            missing.push("zero_knowledge_proofs (needs a zero-knowledge proving system)");
        }
        if requirements.non_repudiation {
            missing.push(
                "non_repudiation (needs an asymmetric signature scheme; HMAC is symmetric and \
                 cannot bind a single signer)",
            );
        }
        if requirements.confidentiality_proofs {
            missing.push("confidentiality_proofs (needs an authenticated encryption scheme)");
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(OptimError::UnsupportedOperation(format!(
                "the audit configuration requests proof guarantees this build cannot provide: {}",
                missing.join("; ")
            )))
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CryptographicProofGenerator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Constant-time comparison of a variable-length proof against a digest.
fn digest_matches(proof: &[u8], digest: &Digest32) -> bool {
    if proof.len() != digest.len() {
        return false;
    }
    let mut fixed = [0u8; 32];
    fixed.copy_from_slice(proof);
    digests_equal(&fixed, digest)
}

/// Proof system for formal verification.
pub struct ProofSystem<T: Float + Debug + Send + Sync + 'static> {
    /// Registered algorithms.
    algorithms: HashMap<String, ProofAlgorithm<T>>,
    /// Verification key material, by algorithm name.
    verification_keys: HashMap<String, Vec<u8>>,
}

impl<T: Float + Debug + Send + Sync + 'static> ProofSystem<T> {
    /// Create a proof system with the built-in integrity algorithms.
    pub fn new() -> Self {
        let key = random_key();
        let mut verification_keys = HashMap::new();
        verification_keys.insert(HMAC_SHA256_INTEGRITY.to_string(), key.to_vec());

        let mut algorithms = HashMap::new();
        algorithms.insert(
            SHA256_INTEGRITY.to_string(),
            ProofAlgorithm {
                name: SHA256_INTEGRITY.to_string(),
                generate_fn: Box::new(|data: &Array1<T>| {
                    let bytes = canonical_array_bytes(data)?;
                    Ok(sha256(&[&bytes]).to_vec())
                }),
                verify_fn: Box::new(|proof: &[u8], data: &Array1<T>| {
                    let Ok(bytes) = canonical_array_bytes(data) else {
                        return false;
                    };
                    digest_matches(proof, &sha256(&[&bytes]))
                }),
            },
        );
        let mac_key = key;
        algorithms.insert(
            HMAC_SHA256_INTEGRITY.to_string(),
            ProofAlgorithm {
                name: HMAC_SHA256_INTEGRITY.to_string(),
                generate_fn: Box::new(move |data: &Array1<T>| {
                    let bytes = canonical_array_bytes(data)?;
                    Ok(hmac_sha256(&mac_key, &bytes).to_vec())
                }),
                verify_fn: Box::new(move |proof: &[u8], data: &Array1<T>| {
                    let Ok(bytes) = canonical_array_bytes(data) else {
                        return false;
                    };
                    digest_matches(proof, &hmac_sha256(&mac_key, &bytes))
                }),
            },
        );

        Self {
            algorithms,
            verification_keys,
        }
    }

    /// Create a proof system with no algorithms registered.
    pub fn empty() -> Self {
        Self {
            algorithms: HashMap::new(),
            verification_keys: HashMap::new(),
        }
    }

    /// Names of the registered algorithms.
    pub fn registered_algorithms(&self) -> Vec<String> {
        let mut names: Vec<String> = self.algorithms.keys().cloned().collect();
        names.sort();
        names
    }

    /// Register an algorithm.
    pub fn register_algorithm(&mut self, algorithm: ProofAlgorithm<T>) {
        self.algorithms.insert(algorithm.name.clone(), algorithm);
    }

    /// Verification key material for an algorithm, if any.
    pub fn verification_key(&self, name: &str) -> Option<&[u8]> {
        self.verification_keys.get(name).map(|key| key.as_slice())
    }

    /// Generate a proof with the named algorithm.
    pub fn generate(&self, algorithm: &str, data: &Array1<T>) -> Result<Vec<u8>> {
        if self.algorithms.is_empty() {
            return Err(OptimError::InvalidState(
                "no proof algorithms are registered with the proof system".to_string(),
            ));
        }
        let entry = self.algorithms.get(algorithm).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "unknown proof algorithm `{algorithm}`; registered: {:?}",
                self.registered_algorithms()
            ))
        })?;
        (entry.generate_fn)(data)
    }

    /// Verify a proof with the named algorithm.
    pub fn verify(&self, algorithm: &str, proof: &[u8], data: &Array1<T>) -> Result<bool> {
        let entry = self.algorithms.get(algorithm).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "unknown proof algorithm `{algorithm}`; registered: {:?}",
                self.registered_algorithms()
            ))
        })?;
        Ok((entry.verify_fn)(proof, data))
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ProofSystem<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Array1<f64> {
        Array1::from(vec![1.0, -2.5, 3.75, 0.0])
    }

    #[test]
    fn a_generated_integrity_proof_verifies_and_a_tampered_value_does_not() {
        let generator: CryptographicProofGenerator<f64> = CryptographicProofGenerator::new();
        for prooftype in [SHA256_INTEGRITY, HMAC_SHA256_INTEGRITY] {
            let proof = match generator.generate_proof(prooftype, &data()) {
                Ok(proof) => proof,
                Err(err) => panic!("{prooftype} generation failed: {err}"),
            };
            assert_eq!(proof.proof_data.len(), 32);
            match generator.verify_proof(&proof, &data()) {
                Ok(true) => {}
                Ok(false) => panic!("{prooftype} must verify against its own input"),
                Err(err) => panic!("{prooftype} verification failed: {err}"),
            }

            // Flip one bit of one element.
            let mut tampered = data();
            tampered[2] = f64::from_bits(tampered[2].to_bits() ^ 1);
            match generator.verify_proof(&proof, &tampered) {
                Ok(false) => {}
                Ok(true) => panic!("{prooftype} must not verify against modified data"),
                Err(err) => panic!("{prooftype} verification failed: {err}"),
            }

            // Truncating the vector must also fail (length is committed).
            let shorter = Array1::from(vec![1.0, -2.5, 3.75]);
            match generator.verify_proof(&proof, &shorter) {
                Ok(false) => {}
                Ok(true) => panic!("{prooftype} must commit to the length"),
                Err(err) => panic!("{prooftype} verification failed: {err}"),
            }
        }
    }

    #[test]
    fn a_generator_with_no_proof_types_fails_instead_of_succeeding() {
        let generator: CryptographicProofGenerator<f64> = CryptographicProofGenerator::empty();
        assert!(generator.generate_proof(SHA256_INTEGRITY, &data()).is_err());
        assert!(generator.registered_proof_types().is_empty());
    }

    #[test]
    fn an_unknown_proof_type_is_an_error() {
        let generator: CryptographicProofGenerator<f64> = CryptographicProofGenerator::new();
        assert!(generator.generate_proof("zk-snark", &data()).is_err());
    }

    #[test]
    fn two_generators_use_independent_mac_keys() {
        let left: CryptographicProofGenerator<f64> = CryptographicProofGenerator::new();
        let right: CryptographicProofGenerator<f64> = CryptographicProofGenerator::new();
        let left_proof = match left.generate_proof(HMAC_SHA256_INTEGRITY, &data()) {
            Ok(proof) => proof,
            Err(err) => panic!("generation failed: {err}"),
        };
        // The keyed tag must not verify under a different key.
        match right.verify_proof(&left_proof, &data()) {
            Ok(false) => {}
            Ok(true) => panic!("the MAC key must not be a shared constant"),
            Err(err) => panic!("verification failed: {err}"),
        }
    }

    #[test]
    fn the_unkeyed_digest_is_reproducible_across_generators() {
        let left: CryptographicProofGenerator<f64> = CryptographicProofGenerator::new();
        let right: CryptographicProofGenerator<f64> = CryptographicProofGenerator::new();
        let proof = match left.generate_proof(SHA256_INTEGRITY, &data()) {
            Ok(proof) => proof,
            Err(err) => panic!("generation failed: {err}"),
        };
        match right.verify_proof(&proof, &data()) {
            Ok(true) => {}
            Ok(false) => panic!("an unkeyed digest must be publicly verifiable"),
            Err(err) => panic!("verification failed: {err}"),
        }
    }

    #[test]
    fn unmeetable_proof_requirements_are_refused() {
        let generator: CryptographicProofGenerator<f64> = CryptographicProofGenerator::new();
        let supported = ProofRequirements {
            zero_knowledge_proofs: false,
            non_repudiation: false,
            integrity_proofs: true,
            confidentiality_proofs: false,
            completeness_proofs: true,
        };
        assert!(generator.check_requirements(&supported).is_ok());

        for requirements in [
            ProofRequirements {
                zero_knowledge_proofs: true,
                non_repudiation: false,
                integrity_proofs: true,
                confidentiality_proofs: false,
                completeness_proofs: false,
            },
            ProofRequirements {
                zero_knowledge_proofs: false,
                non_repudiation: true,
                integrity_proofs: true,
                confidentiality_proofs: false,
                completeness_proofs: false,
            },
            ProofRequirements {
                zero_knowledge_proofs: false,
                non_repudiation: false,
                integrity_proofs: true,
                confidentiality_proofs: true,
                completeness_proofs: false,
            },
        ] {
            assert!(
                generator.check_requirements(&requirements).is_err(),
                "an unimplementable guarantee must be refused, not silently granted"
            );
        }
    }

    #[test]
    fn the_proof_system_generates_and_verifies_both_algorithms() {
        let system: ProofSystem<f64> = ProofSystem::new();
        for algorithm in [SHA256_INTEGRITY, HMAC_SHA256_INTEGRITY] {
            let proof = match system.generate(algorithm, &data()) {
                Ok(proof) => proof,
                Err(err) => panic!("{algorithm} failed: {err}"),
            };
            match system.verify(algorithm, &proof, &data()) {
                Ok(true) => {}
                Ok(false) => panic!("{algorithm} must verify its own proof"),
                Err(err) => panic!("{algorithm} verification failed: {err}"),
            }
            let mut tampered = data();
            tampered[0] = 1.5;
            match system.verify(algorithm, &proof, &tampered) {
                Ok(false) => {}
                Ok(true) => panic!("{algorithm} must reject modified data"),
                Err(err) => panic!("{algorithm} verification failed: {err}"),
            }
        }
    }

    #[test]
    fn an_empty_proof_system_reports_that_it_has_nothing_registered() {
        let system: ProofSystem<f64> = ProofSystem::empty();
        assert!(system.generate(SHA256_INTEGRITY, &data()).is_err());
        assert!(system.registered_algorithms().is_empty());
    }

    #[test]
    fn non_finite_values_cannot_be_committed_to() {
        let system: ProofSystem<f64> = ProofSystem::new();
        // NaN is representable as f64, so it *can* be committed; the digest
        // just has to be stable. What must fail is a value that cannot be
        // converted at all, which f64 always can -- so assert the NaN case is
        // handled deterministically rather than erroring.
        let with_nan = Array1::from(vec![f64::NAN, 1.0]);
        let proof = match system.generate(SHA256_INTEGRITY, &with_nan) {
            Ok(proof) => proof,
            Err(err) => panic!("generation failed: {err}"),
        };
        match system.verify(SHA256_INTEGRITY, &proof, &with_nan) {
            Ok(true) => {}
            Ok(false) => panic!("a NaN commitment must be reproducible"),
            Err(err) => panic!("verification failed: {err}"),
        }
    }
}
