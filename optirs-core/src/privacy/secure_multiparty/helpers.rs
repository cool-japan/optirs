//! Shared low-level primitives: the module-wide CSPRNG, constant-time and
//! keyed-hash helpers, and the `F_p` (`p = 2^127 - 1`) modular-arithmetic and
//! fixed-point quantisation used by [`super::primitives::ShamirSecretSharing`].
//!
//! Domain-separator byte strings and length constants used by the commitment,
//! verification-tag, aggregate-digest, value-digest and computation-digest
//! hashes live here too, since every one of them is consumed from a different
//! sibling module.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::{rngs::StdRng, thread_rng, Random, Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::fmt::Debug;

use super::primitives::CommunicationSecurity;

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
pub(super) const FIXED_POINT_SCALE: f64 = 1_152_921_504_606_846_976.0;

/// Largest field element interpreted as a non-negative value.
///
/// Elements above this limit represent negative values (`element - p`).
pub(super) const FIELD_POSITIVE_LIMIT: u128 = (SHAMIR_PRIME - 1) / 2;

pub(super) const COMMITMENT_DOMAIN: &[u8] = b"optirs.smpc.commitment.v1";

pub(super) const VERIFICATION_DOMAIN: &[u8] = b"optirs.smpc.verification-tag.v1";

pub(super) const AGGREGATE_DOMAIN: &[u8] = b"optirs.smpc.aggregate-digest.v1";

pub(super) const VALUE_DIGEST_DOMAIN: &[u8] = b"optirs.smpc.value-digest.v1";

pub(super) const COMPUTATION_DOMAIN: &[u8] = b"optirs.smpc.computation-digest.v1";

/// Length of a commitment nonce in bytes.
pub const COMMITMENT_NONCE_LEN: usize = 32;

/// Length of a SHA-256 digest in bytes.
pub const DIGEST_LEN: usize = 32;

/// Generator used for every piece of secret material in this module.
///
/// `StdRng` is ChaCha12-backed (a CSPRNG) and `Send`, unlike the thread-local
/// generator, so it can be stored inside the `Send + Sync` types below.
pub(super) type SecureRng = Random<StdRng>;

/// Create a generator seeded from OS entropy.
///
/// Every instance of every type in this module gets its own seed; nothing in this
/// module uses a compile-time constant seed unless the caller explicitly asks for a
/// deterministic `*_with_seed` constructor (intended for tests only).
pub(super) fn os_seeded_rng() -> SecureRng {
    SeedableRng::from_rng(&mut thread_rng())
}

/// Draw `len` uniformly random bytes.
pub(super) fn random_bytes(rng: &mut SecureRng, len: usize) -> Vec<u8> {
    let mut buffer = vec![0u8; len];
    rng.fill_bytes(&mut buffer);
    buffer
}

/// Constant-time comparison of two byte strings.
pub(super) fn ct_eq(a: &[u8], b: &[u8]) -> bool {
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
pub(super) fn hash_values<T: Float + Debug + Send + Sync + 'static>(
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

/// Largest magnitude that [`FIXED_POINT_BITS`] quantisation can represent (`2^66`).
pub fn max_representable_magnitude() -> f64 {
    FIELD_POSITIVE_LIMIT as f64 / FIXED_POINT_SCALE
}

/// Modular addition in `F_p`.
#[inline]
pub(super) fn add_mod(a: u128, b: u128) -> u128 {
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
pub(super) fn sub_mod(a: u128, b: u128) -> u128 {
    if a >= b {
        a - b
    } else {
        SHAMIR_PRIME - (b - a)
    }
}

/// Modular negation in `F_p`.
#[inline]
pub(super) fn neg_mod(a: u128) -> u128 {
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
pub(super) fn mul_mod(a: u128, b: u128) -> u128 {
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
pub(super) fn pow_mod(base: u128, exponent: u128) -> u128 {
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
pub(super) fn inv_mod(a: u128) -> Result<u128> {
    if a.is_multiple_of(SHAMIR_PRIME) {
        return Err(OptimError::InvalidConfig(
            "cannot invert zero in the Shamir field (duplicate share x-coordinate?)".to_string(),
        ));
    }
    Ok(pow_mod(a, SHAMIR_PRIME - 2))
}

/// Quantise a value into a field element.
pub(super) fn value_to_field<T: Float + Debug + Send + Sync + 'static>(value: T) -> Result<u128> {
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
pub(super) fn field_to_value<T: Float + Debug + Send + Sync + 'static>(element: u128) -> Result<T> {
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
pub(super) fn evaluate_polynomial(coefficients: &[u128], x: u128) -> u128 {
    let mut accumulator = 0u128;
    for &coefficient in coefficients.iter().rev() {
        accumulator = add_mod(mul_mod(accumulator, x), coefficient % SHAMIR_PRIME);
    }
    accumulator
}

/// Reject configurations whose security model is not implemented.
pub(super) fn require_supported_security(security: CommunicationSecurity) -> Result<()> {
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

pub(super) fn unimplemented_homomorphic(operation: &str) -> OptimError {
    OptimError::UnsupportedOperation(format!(
        "HomomorphicEngine {operation} is not homomorphic encryption — unimplemented, \
         do not use for confidentiality; use privacy::secure_aggregation for additive \
         aggregation instead"
    ))
}

pub(super) fn unimplemented_zero_knowledge(operation: &str) -> OptimError {
    OptimError::UnsupportedOperation(format!(
        "zero-knowledge {operation} is not cryptographically secure — unimplemented; \
         `ComputationDigestSystem` only provides a non-hiding integrity digest via \
         digest_computation/verify_digest"
    ))
}
