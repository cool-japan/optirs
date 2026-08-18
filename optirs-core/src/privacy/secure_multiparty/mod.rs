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

mod coordinator;
mod helpers;
mod primitives;

pub use coordinator::*;
pub use helpers::{
    max_representable_magnitude, COMMITMENT_NONCE_LEN, DIGEST_LEN, FIXED_POINT_BITS, SHAMIR_PRIME,
};
pub use primitives::*;

#[cfg(test)]
mod tests;
