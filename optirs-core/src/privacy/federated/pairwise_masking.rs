// Pairwise mask derivation for Bonawitz-style secure aggregation.
//
// This module supplies the cryptographic half of
// [`super::secure_aggregation`]: real X25519 elliptic-curve Diffie-Hellman key
// agreement between every pair of participating clients, a SHA-256
// counter-mode PRG that expands each agreed secret into a mask vector, and the
// signed accumulation rule that makes those masks cancel exactly when the
// server adds the uploads together.
//
// Why key agreement, and not a published seed
// -------------------------------------------
// The security of pairwise masking rests entirely on the server being unable
// to reproduce the masks. If the pairwise seed is a function of public data
// only -- client identifiers plus a round salt the server itself published --
// then the server can recompute every mask and subtract it from any single
// upload, recovering that client's gradient exactly. That is a masking
// *protocol* with zero confidentiality.
//
// Here each client generates a fresh X25519 key pair per round and publishes
// only the public half. The seed shared by clients `i` and `j` is
//
// ```text
// seed_ij = SHA-256( DOMAIN || round_seed || min(pk_i, pk_j) || max(pk_i, pk_j)
//                    || X25519(sk_i, pk_j) )
// ```
//
// which is symmetric (X25519 is, and the public keys are sorted) yet requires
// one of the two secret keys. The aggregation server holds neither, so it can
// verify nothing about, and reconstruct nothing from, an individual upload.
//
// Mask expansion
// --------------
// The 32-byte seed is stretched with SHA-256 in counter mode:
// `block_k = SHA-256(PRG_DOMAIN || seed || k)`, each block yielding four
// little-endian `u64` words. Each word is reduced into `[0, modulus)` by
// rejection sampling, so the mask is uniform over the additive group rather
// than modulo-biased.
//
// Sign rule
// ---------
// Client `i` adds `+mask_ij` for every peer `j` whose identifier sorts after
// its own and `-mask_ij` for every peer that sorts before it, all modulo the
// group order. Each unordered pair therefore contributes `+mask_ij` exactly
// once and `-mask_ij` exactly once to the server's sum, so the masks telescope
// to zero and the server recovers the exact sum of the quantised inputs -- and
// nothing else.
//
// Reference
// ---------
//   * Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B.,
//     Patel, S., Ramage, D., Segal, A., Seth, K. "Practical Secure Aggregation
//     for Privacy-Preserving Machine Learning." CCS 2017.
//
// Relation to `privacy::secure_aggregation`
// -----------------------------------------
// The sibling module `crate::privacy::secure_aggregation` implements the same
// aggregation arithmetic over `u64` client identifiers, but derives its
// pairwise seeds from a *public* formula, which it documents as a deliberate
// demo simplification. This module reuses that module's quantisation
// primitives verbatim (see `super::secure_aggregation`) and replaces exactly
// the part that cannot be left simplified: the seed derivation.

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use x25519_dalek::{PublicKey, StaticSecret};

use scirs2_core::random::thread_rng;

/// Domain separation tag for pairwise seed derivation.
const SEED_DOMAIN: &[u8] = b"OPTIRS-FED-SECAGG-PAIRWISE-SEED-v1";

/// Domain separation tag for the mask-expansion PRG.
const PRG_DOMAIN: &[u8] = b"OPTIRS-FED-SECAGG-MASK-PRG-v1";

/// Words produced per SHA-256 PRG block.
const WORDS_PER_BLOCK: usize = 4;

/// A participant's X25519 public key, as published to the server.
///
/// Byte-comparable so that the pairwise seed derivation can canonicalise the
/// unordered pair `{pk_i, pk_j}` without extra state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ClientPublicKey([u8; 32]);

impl ClientPublicKey {
    /// Wrap raw key bytes received from a peer.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw key bytes, for transport.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A participant's per-round X25519 key pair.
///
/// The secret half never leaves the client. There is intentionally no
/// accessor for it and no `Debug` output that could print it: the whole point
/// of this type is that the aggregation server cannot obtain it.
pub struct ClientKeyPair {
    secret: StaticSecret,
    public: ClientPublicKey,
}

impl std::fmt::Debug for ClientKeyPair {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClientKeyPair")
            .field("public", &self.public)
            .field("secret", &"<redacted>")
            .finish()
    }
}

impl ClientKeyPair {
    /// Generate a fresh key pair from operating-system entropy.
    ///
    /// A new pair per round is what makes the masks of different rounds
    /// independent; reusing one across rounds would let a server that
    /// observes two rounds cancel the shared structure.
    pub fn generate() -> Self {
        let mut bytes = [0_u8; 32];
        // `thread_rng` is rand's cryptographically secure thread-local
        // generator, seeded from the operating system.
        thread_rng().fill(&mut bytes[..]);
        Self::from_secret_bytes(bytes)
    }

    /// Build a key pair from explicit secret bytes.
    ///
    /// Intended for reproducible protocol transcripts in tests and for
    /// callers that already derive client secrets from their own key
    /// hierarchy. Production clients should prefer [`Self::generate`].
    pub fn from_secret_bytes(bytes: [u8; 32]) -> Self {
        let secret = StaticSecret::from(bytes);
        let public = ClientPublicKey(PublicKey::from(&secret).to_bytes());
        Self { secret, public }
    }

    /// The public half, to be published to the aggregation server.
    pub fn public_key(&self) -> ClientPublicKey {
        self.public
    }

    /// Derive the 32-byte seed shared with `peer` for `round_seed`.
    ///
    /// Symmetric: `a.shared_seed_with(b.public, r) == b.shared_seed_with(a.public, r)`.
    ///
    /// Errors when `peer` is this client's own key (a client has no pairwise
    /// mask with itself) or when the Diffie-Hellman exchange yields the
    /// all-zero shared secret, which is what a low-order (small-subgroup)
    /// public key produces and would make the mask predictable.
    pub fn shared_seed_with(&self, peer: &ClientPublicKey, round_seed: u64) -> Result<[u8; 32]> {
        if *peer == self.public {
            return Err(OptimError::InvalidParameter(
                "a client cannot derive a pairwise mask with its own public key".to_string(),
            ));
        }
        let shared = self.secret.diffie_hellman(&PublicKey::from(peer.0));
        let shared_bytes = shared.as_bytes();
        if shared_bytes.iter().all(|&byte| byte == 0) {
            return Err(OptimError::InvalidParameter(
                "X25519 key agreement produced the all-zero shared secret; the peer supplied a \
                 low-order public key and the resulting mask would be predictable"
                    .to_string(),
            ));
        }

        let (low, high) = if self.public.0 <= peer.0 {
            (&self.public.0, &peer.0)
        } else {
            (&peer.0, &self.public.0)
        };

        let mut hasher = Sha256::new();
        hasher.update(SEED_DOMAIN);
        hasher.update(round_seed.to_le_bytes());
        hasher.update(low);
        hasher.update(high);
        hasher.update(shared_bytes);
        let digest = hasher.finalize();

        let mut seed = [0_u8; 32];
        seed.copy_from_slice(&digest);
        Ok(seed)
    }
}

/// Expand a 32-byte seed into a `dim`-long mask over `[0, modulus)`.
///
/// SHA-256 counter mode with rejection sampling, so the output is uniform over
/// the additive group. Deterministic in `(seed, dim, modulus)` -- that
/// determinism is exactly what makes the two holders of the seed produce
/// identical masks and therefore what makes the masks cancel.
pub fn expand_mask(seed: &[u8; 32], dim: usize, modulus: i64) -> Result<Vec<i64>> {
    if modulus <= 1 {
        return Err(OptimError::InvalidParameter(format!(
            "mask modulus must be greater than 1, got {modulus}"
        )));
    }
    if dim == 0 {
        return Ok(Vec::new());
    }

    let modulus_u = modulus as u64;
    // Largest multiple of `modulus` that fits in a u64; words at or above it
    // are rejected so no residue is over-represented.
    let acceptance_bound = (u64::MAX / modulus_u) * modulus_u;

    let mut mask = Vec::with_capacity(dim);
    let mut counter = 0_u64;
    while mask.len() < dim {
        let mut hasher = Sha256::new();
        hasher.update(PRG_DOMAIN);
        hasher.update(seed);
        hasher.update(counter.to_le_bytes());
        let block = hasher.finalize();
        counter = counter.wrapping_add(1);

        for word_index in 0..WORDS_PER_BLOCK {
            if mask.len() == dim {
                break;
            }
            let start = word_index * 8;
            let mut word_bytes = [0_u8; 8];
            word_bytes.copy_from_slice(&block[start..start + 8]);
            let word = u64::from_le_bytes(word_bytes);
            if word < acceptance_bound {
                mask.push((word % modulus_u) as i64);
            }
        }
    }
    Ok(mask)
}

/// The signed pairwise mask that `own_id` contributes for peer `peer_id`.
///
/// `+mask` when `own_id` sorts before `peer_id`, `-mask` otherwise, reduced
/// into `[0, modulus)`.
pub fn signed_pairwise_mask(
    own_id: &str,
    own_keys: &ClientKeyPair,
    peer_id: &str,
    peer_key: &ClientPublicKey,
    round_seed: u64,
    dim: usize,
    modulus: i64,
) -> Result<Vec<i64>> {
    if own_id == peer_id {
        return Err(OptimError::InvalidParameter(format!(
            "client {own_id} cannot hold a pairwise mask with itself"
        )));
    }
    let seed = own_keys.shared_seed_with(peer_key, round_seed)?;
    let mask = expand_mask(&seed, dim, modulus)?;
    let positive = own_id < peer_id;
    Ok(mask
        .into_iter()
        .map(|value| {
            if positive {
                value.rem_euclid(modulus)
            } else {
                (-value).rem_euclid(modulus)
            }
        })
        .collect())
}

/// The complete additive mask client `own_id` applies to its quantised
/// update.
///
/// `peers` is the round's published public-key directory. `own_id`'s own
/// entry, if present, is skipped; every other entry contributes one signed
/// pairwise mask. Errors when the directory does not contain `own_id` (the
/// client is not part of this round) or when it contains no peers (a cohort of
/// one cannot be masked, and pretending otherwise would upload the raw
/// gradient).
pub fn compute_client_mask(
    own_id: &str,
    own_keys: &ClientKeyPair,
    peers: &BTreeMap<String, ClientPublicKey>,
    round_seed: u64,
    dim: usize,
    modulus: i64,
) -> Result<Vec<i64>> {
    if modulus <= 1 {
        return Err(OptimError::InvalidParameter(format!(
            "mask modulus must be greater than 1, got {modulus}"
        )));
    }
    match peers.get(own_id) {
        None => {
            return Err(OptimError::InvalidParameter(format!(
                "client {own_id} is not in the round's public-key directory"
            )));
        }
        Some(published) if *published != own_keys.public_key() => {
            return Err(OptimError::InvalidParameter(format!(
                "the public key published for client {own_id} does not match the supplied key \
                 pair; the derived masks would not cancel"
            )));
        }
        Some(_) => {}
    }
    if peers.len() < 2 {
        return Err(OptimError::InvalidConfig(format!(
            "client {own_id} has no peers in this round; a single-client cohort cannot be \
             masked, so the upload would be the raw update"
        )));
    }

    let mut total = vec![0_i64; dim];
    for (peer_id, peer_key) in peers.iter() {
        if peer_id == own_id {
            continue;
        }
        let signed = signed_pairwise_mask(
            own_id, own_keys, peer_id, peer_key, round_seed, dim, modulus,
        )?;
        for (accumulator, value) in total.iter_mut().zip(signed.iter()) {
            *accumulator = (*accumulator + *value).rem_euclid(modulus);
        }
    }
    Ok(total)
}

/// Draw a fresh public per-round salt from operating-system entropy.
///
/// Published to every client. It does not need to be secret -- the masks are
/// protected by the Diffie-Hellman secrets -- but it must be unpredictable
/// enough that two rounds never reuse a salt with the same key pairs.
pub fn fresh_round_seed() -> u64 {
    thread_rng().random::<u64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(tag: u8) -> ClientKeyPair {
        let mut bytes = [0_u8; 32];
        bytes[0] = tag;
        bytes[31] = tag.wrapping_mul(7).wrapping_add(1);
        ClientKeyPair::from_secret_bytes(bytes)
    }

    const MODULUS: i64 = 1 << 31;

    #[test]
    fn key_agreement_is_symmetric() {
        let alice = keys(1);
        let bob = keys(2);
        let seed_ab = alice
            .shared_seed_with(&bob.public_key(), 7)
            .expect("alice -> bob");
        let seed_ba = bob
            .shared_seed_with(&alice.public_key(), 7)
            .expect("bob -> alice");
        assert_eq!(seed_ab, seed_ba);
    }

    #[test]
    fn the_shared_seed_requires_a_secret_key_so_the_server_cannot_derive_it() {
        let alice = keys(1);
        let bob = keys(2);
        // The server: it sees both public keys and the round seed, and may of
        // course generate a key pair of its own.
        let server = keys(3);

        let truth = alice
            .shared_seed_with(&bob.public_key(), 7)
            .expect("alice -> bob");
        let server_attempt = server
            .shared_seed_with(&bob.public_key(), 7)
            .expect("server -> bob");
        assert_ne!(
            truth, server_attempt,
            "the pairwise seed must depend on a client secret, not only on public data"
        );

        // And the masks it expands to differ everywhere.
        let real = expand_mask(&truth, 64, MODULUS).expect("real mask");
        let forged = expand_mask(&server_attempt, 64, MODULUS).expect("forged mask");
        let matches = real
            .iter()
            .zip(forged.iter())
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            matches < 4,
            "a mask derived without the secret should not coincide with the real one \
             ({matches}/64 coordinates matched)"
        );
    }

    #[test]
    fn seeds_differ_across_rounds_and_across_pairs() {
        let alice = keys(1);
        let bob = keys(2);
        let carol = keys(3);

        let round_one = alice
            .shared_seed_with(&bob.public_key(), 1)
            .expect("round 1");
        let round_two = alice
            .shared_seed_with(&bob.public_key(), 2)
            .expect("round 2");
        assert_ne!(round_one, round_two);

        let with_carol = alice
            .shared_seed_with(&carol.public_key(), 1)
            .expect("alice -> carol");
        assert_ne!(round_one, with_carol);
    }

    #[test]
    fn a_client_cannot_pair_with_itself() {
        let alice = keys(1);
        let err = alice
            .shared_seed_with(&alice.public_key(), 1)
            .expect_err("self pairing must fail");
        assert!(format!("{err}").contains("own public key"));
    }

    #[test]
    fn low_order_public_keys_are_rejected() {
        let alice = keys(1);
        // The all-zero point is the canonical low-order X25519 public key; it
        // drives every shared secret to zero.
        let malicious = ClientPublicKey::from_bytes([0_u8; 32]);
        let err = alice
            .shared_seed_with(&malicious, 1)
            .expect_err("low-order key must be rejected");
        assert!(format!("{err}").contains("all-zero shared secret"));
    }

    #[test]
    fn expand_mask_is_deterministic_and_in_range() {
        let seed = [0x5A_u8; 32];
        let first = expand_mask(&seed, 1000, MODULUS).expect("mask");
        let second = expand_mask(&seed, 1000, MODULUS).expect("mask");
        assert_eq!(first, second);
        assert_eq!(first.len(), 1000);
        assert!(first.iter().all(|&value| (0..MODULUS).contains(&value)));

        assert!(expand_mask(&seed, 0, MODULUS).expect("empty").is_empty());
        assert!(expand_mask(&seed, 4, 1).is_err());
    }

    #[test]
    fn expand_mask_output_covers_the_whole_group() {
        // A real PRG mask is spread over [0, modulus); an implementation that
        // only jittered by a small amount would fail this.
        let seed = [0x11_u8; 32];
        let mask = expand_mask(&seed, 4096, MODULUS).expect("mask");
        let minimum = mask.iter().copied().min().expect("non-empty");
        let maximum = mask.iter().copied().max().expect("non-empty");
        assert!(
            minimum < MODULUS / 100,
            "minimum {minimum} is not near zero"
        );
        assert!(
            maximum > MODULUS - MODULUS / 100,
            "maximum {maximum} is not near the modulus"
        );

        // Rough uniformity: each of eight buckets should hold ~12.5%.
        let mut buckets = [0_usize; 8];
        for &value in mask.iter() {
            let bucket = ((value as i128 * 8) / MODULUS as i128) as usize;
            buckets[bucket.min(7)] += 1;
        }
        for (index, &count) in buckets.iter().enumerate() {
            assert!(
                count > 4096 / 16 && count < 4096 / 4,
                "bucket {index} holds {count} of 4096 samples, which is not roughly uniform"
            );
        }
    }

    #[test]
    fn expand_mask_changes_with_the_seed() {
        let a = expand_mask(&[1_u8; 32], 64, MODULUS).expect("mask");
        let b = expand_mask(&[2_u8; 32], 64, MODULUS).expect("mask");
        assert_ne!(a, b);
    }

    #[test]
    fn signed_masks_of_a_pair_are_additive_inverses() {
        let alice = keys(1);
        let bob = keys(2);
        let from_alice =
            signed_pairwise_mask("alice", &alice, "bob", &bob.public_key(), 42, 32, MODULUS)
                .expect("alice mask");
        let from_bob =
            signed_pairwise_mask("bob", &bob, "alice", &alice.public_key(), 42, 32, MODULUS)
                .expect("bob mask");

        assert_eq!(from_alice.len(), 32);
        for (a, b) in from_alice.iter().zip(from_bob.iter()) {
            assert_eq!(
                (a + b).rem_euclid(MODULUS),
                0,
                "pairwise masks must cancel: {a} + {b} != 0 mod {MODULUS}"
            );
        }
    }

    #[test]
    fn every_clients_mask_sums_to_zero_over_the_cohort() {
        let dim = 128;
        let round_seed = 0xDEAD_BEEF;
        let pairs: Vec<(String, ClientKeyPair)> = (1..=6_u8)
            .map(|tag| (format!("client{tag:02}"), keys(tag)))
            .collect();
        let directory: BTreeMap<String, ClientPublicKey> = pairs
            .iter()
            .map(|(id, keys)| (id.clone(), keys.public_key()))
            .collect();

        let mut total = vec![0_i64; dim];
        for (id, key_pair) in pairs.iter() {
            let mask = compute_client_mask(id, key_pair, &directory, round_seed, dim, MODULUS)
                .expect("client mask");
            assert_eq!(mask.len(), dim);
            for (accumulator, value) in total.iter_mut().zip(mask.iter()) {
                *accumulator = (*accumulator + *value).rem_euclid(MODULUS);
            }
        }
        assert!(
            total.iter().all(|&value| value == 0),
            "cohort masks did not telescope to zero"
        );
    }

    #[test]
    fn an_individual_mask_is_not_trivial() {
        let dim = 256;
        let pairs: Vec<(String, ClientKeyPair)> = (1..=4_u8)
            .map(|tag| (format!("client{tag}"), keys(tag)))
            .collect();
        let directory: BTreeMap<String, ClientPublicKey> = pairs
            .iter()
            .map(|(id, keys)| (id.clone(), keys.public_key()))
            .collect();
        let (id, key_pair) = &pairs[0];
        let mask = compute_client_mask(id, key_pair, &directory, 1, dim, MODULUS).expect("mask");
        let zeros = mask.iter().filter(|&&value| value == 0).count();
        assert!(zeros < 4, "{zeros} of {dim} mask coordinates were zero");
    }

    #[test]
    fn compute_client_mask_validates_the_directory() {
        let alice = keys(1);
        let bob = keys(2);
        let mut directory = BTreeMap::new();
        directory.insert("bob".to_string(), bob.public_key());

        // Alice is not in the directory.
        let err = compute_client_mask("alice", &alice, &directory, 1, 8, MODULUS)
            .expect_err("missing from directory");
        assert!(format!("{err}").contains("not in the round's public-key directory"));

        // Alice is in the directory under someone else's public key.
        directory.insert("alice".to_string(), keys(9).public_key());
        let err = compute_client_mask("alice", &alice, &directory, 1, 8, MODULUS)
            .expect_err("key mismatch");
        assert!(format!("{err}").contains("does not match the supplied key pair"));

        // A cohort of one cannot be masked.
        let mut solo = BTreeMap::new();
        solo.insert("alice".to_string(), alice.public_key());
        let err =
            compute_client_mask("alice", &alice, &solo, 1, 8, MODULUS).expect_err("solo cohort");
        assert!(format!("{err}").contains("no peers"));
    }

    #[test]
    fn generated_key_pairs_are_distinct() {
        let first = ClientKeyPair::generate();
        let second = ClientKeyPair::generate();
        assert_ne!(first.public_key(), second.public_key());
        // And the secret is never printed.
        assert!(format!("{first:?}").contains("<redacted>"));
    }

    #[test]
    fn fresh_round_seeds_are_not_a_counter() {
        let seeds: Vec<u64> = (0..8).map(|_| fresh_round_seed()).collect();
        let distinct: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(distinct.len(), seeds.len());
        // A wrapping counter starting at zero would produce 1, 2, 3, ...
        assert!(seeds.iter().any(|&seed| seed > u64::MAX / 1024));
    }
}
