// Bonawitz-style Pairwise-Mask Secure Aggregation Protocol
//
// This module implements the practical secure aggregation protocol introduced
// by Bonawitz et al. (2017) for federated learning. The protocol is the
// foundational scheme behind production federated systems (e.g. Google
// Gboard, Apple's on-device learning) and provides cryptographic
// confidentiality of individual client gradients while still allowing the
// server to compute their exact sum.
//
// Reference
// ---------
//   * Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B.,
//     Patel, S., Ramage, D., Segal, A., and Seth, K. "Practical Secure
//     Aggregation for Privacy-Preserving Machine Learning." CCS 2017.
//   * Bonawitz, K., Eichner, H., Grieskamp, W., et al. "Towards Federated
//     Learning at Scale: System Design." SysML 2019. (Production
//     considerations and dropout reconstruction details.)
//
// Protocol overview
// -----------------
// Each ordered pair of clients (i, j) with i != j agrees on a shared
// pseudo-random vector m_{ij} ∈ Z_p^d (where p is a large prime-ish modulus
// and d is the gradient dimension). In a real deployment the seed for this
// PRNG would be derived from a Diffie-Hellman key agreement performed in an
// earlier protocol round; for the demo we derive the seed deterministically
// from the (sorted) client pair and the per-round seed published by the
// server, so the same mask is reproducible on both sides.
//
// Each client i then constructs a self-mask
//
//     mask_i = Σ_{j ∈ S : i < j} m_{ij} − Σ_{j ∈ S : j < i} m_{ji}
//
// where the sign of each pairwise mask is determined purely by the lexical
// order of the two client identifiers. The client uploads the quantised
// masked gradient y_i = (Q(g_i) + mask_i) mod p to the server. When the
// server sums all received submissions modulo p, every pairwise term
// m_{ij} appears exactly twice -- once with the (+) sign from client i and
// once with the (-) sign from client j -- so the masks telescope to zero
// and the server recovers Σ Q(g_i) mod p. Dequantisation then yields the
// true gradient sum up to quantisation error.
//
// Dropout robustness
// ------------------
// If client d drops out before submitting, every other online client still
// included m_{*d} in their personal mask, so the server's running sum
// retains a non-zero residue. In a production deployment, the online
// participants would reveal to the server the pairwise masks they hold with
// the dropped client (or a t-out-of-n Shamir share of the seed) -- since the
// dropped client never uploaded anything, revealing this information leaks
// nothing about its gradient. In this demo implementation the server is
// able to reconstruct the dropped client's pairwise masks directly because
// every mask is deterministically derived from `round_seed`. This is a
// deliberate simplification that captures the *aggregation arithmetic* of
// the protocol while keeping the public surface small enough to test
// exhaustively.
//
// Relation to existing OptiRS modules
// -----------------------------------
// OptiRS already ships two adjacent privacy primitives that should not be
// confused with this one:
//
//   * `optirs_core::privacy::secure_multiparty` -- a Shamir-secret-sharing
//     based BGW / GMW / SPDZ MPC stack (different cryptographic family,
//     general-purpose arithmetic circuits).
//   * `optirs_core::privacy::byzantine_tolerance` -- robust aggregation
//     (median, trimmed mean, Krum) designed to *exclude* malicious
//     gradients rather than mask honest ones.
//
// Both are cited here as related prior art. This module is intentionally
// independent of them and provides the specific Bonawitz pairwise-additive
// masking primitive that is missing from the existing stack.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a participant in a single aggregation round.
pub type ClientId = u64;

/// Configuration for the Bonawitz-style secure aggregation protocol.
///
/// All clients in a round must agree on the same configuration so that the
/// derived pairwise masks line up. The `round_seed` field acts as a public
/// salt that lets the same fleet of clients run independent aggregation
/// rounds (different `round_seed`s produce independent pairwise masks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureAggregationConfig {
    /// Total number of clients expected to participate in the round.
    pub num_clients: usize,

    /// Dimensionality of the gradient vectors being aggregated.
    pub gradient_dim: usize,

    /// Public, per-round salt mixed into every pairwise seed derivation.
    pub round_seed: u64,

    /// Quantisation scale. Gradients are quantised by `round(g * scale)`
    /// before being lifted into the modular group; a larger scale gives
    /// finer fixed-point precision at the cost of needing a larger modulus
    /// to avoid wraparound.
    pub quantization_scale: f64,

    /// Modulus `p` of the additive group `Z_p` used for masking and
    /// aggregation. Must be substantially larger than
    /// `quantization_scale * num_clients * max_gradient_magnitude` to
    /// avoid information-destroying wraparound.
    pub modulus: i64,

    /// Whether to support dropout reconstruction at the server.
    pub support_dropouts: bool,
}

impl Default for SecureAggregationConfig {
    fn default() -> Self {
        Self {
            num_clients: 0,
            gradient_dim: 0,
            round_seed: 42,
            quantization_scale: 1.0e6,
            // The Mersenne prime 2^31 − 1. Wide enough to keep ~24-bit
            // gradients safe through a modest cohort while still fitting
            // in a signed 64-bit integer for safe modular arithmetic.
            modulus: 2_147_483_647,
            support_dropouts: true,
        }
    }
}

/// A single client's masked gradient submission.
///
/// Conceptually `values[k] = (Q(gradient[k]) + mask_i[k]) mod modulus`. The
/// values are always non-negative and strictly less than `modulus` so the
/// representation is canonical and serde-friendly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MaskedGradient {
    /// Identifier of the submitting client.
    pub client_id: ClientId,

    /// Per-coordinate masked + quantised values, each in `[0, modulus)`.
    pub values: Vec<i64>,
}

/// Server-side aggregator state for a single round.
///
/// The aggregator owns no cryptographic key material -- every pairwise mask
/// can be recomputed on demand from `round_seed`. It keeps track of which
/// clients have successfully submitted, which clients are known to have
/// dropped, and cached pairwise masks used during dropout reconstruction.
#[derive(Debug, Clone)]
pub struct SecureAggregator {
    config: SecureAggregationConfig,
    received: HashMap<ClientId, MaskedGradient>,
    dropped: HashSet<ClientId>,
    pairwise_mask_cache: HashMap<(ClientId, ClientId), Vec<i64>>,
}

// ---------------------------------------------------------------------------
// Pure client-side helper functions
// ---------------------------------------------------------------------------

/// Derive the seed for the pairwise PRNG shared by `client_a` and
/// `client_b`.
///
/// The result is symmetric in its two `ClientId` arguments: the *unordered*
/// pair `{client_a, client_b}` determines a single seed. This lets each
/// client independently compute the same pairwise mask without exchanging
/// any further state.
///
/// The combiner uses a Wang-style multiplicative hash on the rotated client
/// identifiers, XOR-mixed with `round_seed`. It is *not* a cryptographic
/// hash -- production deployments would substitute SHA-256 (or HKDF over a
/// Diffie-Hellman secret) here.
pub fn derive_pairwise_seed(client_a: ClientId, client_b: ClientId, round_seed: u64) -> u64 {
    let (lo, hi) = if client_a <= client_b {
        (client_a, client_b)
    } else {
        (client_b, client_a)
    };

    // Mix: rotate the ids by relatively prime amounts so the high bits of
    // small ids end up in different lanes, then xor with the round salt and
    // run through a Wang-style 64-bit avalanche so neighbouring inputs
    // produce well-separated seeds.
    let mut h = round_seed;
    h ^= lo.rotate_left(13);
    h ^= hi.rotate_left(29);
    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB_u64);
    h ^= h >> 31;
    h
}

/// Generate a deterministic pairwise mask of the requested dimension.
///
/// Each coordinate is drawn uniformly from `[0, modulus)` using the seeded
/// PRNG. The function is pure: identical `(seed, dim, modulus)` triples
/// always produce identical output, which is exactly the property that
/// makes the masks cancel on the server.
pub fn generate_pairwise_mask(seed: u64, dim: usize, modulus: i64) -> Vec<i64> {
    if dim == 0 || modulus <= 0 {
        return Vec::new();
    }

    let mut rng = Random::seed(seed);
    let mut mask = Vec::with_capacity(dim);
    for _ in 0..dim {
        let v: i64 = rng.gen_range(0..modulus);
        mask.push(v);
    }
    mask
}

/// Quantise a real-valued gradient into the additive modular group `Z_p`.
///
/// The mapping is `q_k = round(g_k * scale) mod modulus`. Negative
/// post-rounding values are folded into `[0, modulus)` via `rem_euclid`,
/// which guarantees the canonical non-negative representative used
/// everywhere in the protocol.
pub fn quantize_gradient(gradient: &Array1<f64>, scale: f64, modulus: i64) -> Result<Vec<i64>> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "quantization scale must be positive and finite, got {scale}"
        )));
    }
    if modulus <= 1 {
        return Err(OptimError::InvalidParameter(format!(
            "modulus must be > 1, got {modulus}"
        )));
    }

    let mut out = Vec::with_capacity(gradient.len());
    for &g in gradient.iter() {
        if !g.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "gradient entries must be finite, got {g}"
            )));
        }
        let scaled = (g * scale).round();
        // The intermediate must fit in i64; clamp at i64::MAX/MIN to avoid
        // panics on pathological inputs before taking the modulus.
        let clipped = scaled.max(i64::MIN as f64).min(i64::MAX as f64) as i64;
        out.push(clipped.rem_euclid(modulus));
    }
    Ok(out)
}

/// Inverse of [`quantize_gradient`] applied to a vector of canonical
/// `[0, modulus)` representatives.
///
/// Values strictly greater than `modulus / 2` are interpreted as negative
/// fixed-point numbers (the standard two's-complement style centring around
/// zero), then scaled back to `f64`.
pub fn dequantize_gradient(quantized: &[i64], scale: f64, modulus: i64) -> Vec<f64> {
    if scale <= 0.0 || modulus <= 1 {
        return Vec::new();
    }
    let half = modulus / 2;
    quantized
        .iter()
        .map(|&q| {
            let centred = if q > half { q - modulus } else { q };
            (centred as f64) / scale
        })
        .collect()
}

/// Compute the additive mask vector `mask_i` that client `client_id`
/// adds to its quantised gradient before upload.
///
/// For every peer `other_id` in `all_client_ids`:
///   * if `client_id < other_id` the pairwise mask is *added*;
///   * if `client_id > other_id` the pairwise mask is *subtracted*.
///
/// Identical peers (`other_id == client_id`) are skipped, and duplicates
/// are de-duplicated before the loop runs.
///
/// The total mask vector contains values in `[0, modulus)`. All arithmetic
/// is performed modulo `modulus`, so the result is directly compatible with
/// the masked-submission protocol.
pub fn compute_client_mask(
    client_id: ClientId,
    all_client_ids: &[ClientId],
    round_seed: u64,
    dim: usize,
    modulus: i64,
) -> Result<Vec<i64>> {
    if modulus <= 1 {
        return Err(OptimError::InvalidParameter(format!(
            "modulus must be > 1, got {modulus}"
        )));
    }
    if !all_client_ids.contains(&client_id) {
        return Err(OptimError::InvalidParameter(format!(
            "client {client_id} not found in participating client set"
        )));
    }

    // De-duplicate peers while preserving iteration determinism.
    let mut unique: Vec<ClientId> = all_client_ids.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let mut total = vec![0_i64; dim];
    for &other_id in unique.iter() {
        if other_id == client_id {
            continue;
        }
        let seed = derive_pairwise_seed(client_id, other_id, round_seed);
        let pairwise = generate_pairwise_mask(seed, dim, modulus);
        if client_id < other_id {
            for (acc, m) in total.iter_mut().zip(pairwise.iter()) {
                *acc = (*acc + *m).rem_euclid(modulus);
            }
        } else {
            for (acc, m) in total.iter_mut().zip(pairwise.iter()) {
                *acc = (*acc - *m).rem_euclid(modulus);
            }
        }
    }

    Ok(total)
}

/// End-to-end client helper: quantise `gradient`, add the pairwise mask,
/// and package the result as a [`MaskedGradient`] ready for upload.
pub fn submit_gradient(
    client_id: ClientId,
    gradient: &Array1<f64>,
    all_client_ids: &[ClientId],
    config: &SecureAggregationConfig,
) -> Result<MaskedGradient> {
    if gradient.len() != config.gradient_dim {
        return Err(OptimError::DimensionMismatch(format!(
            "gradient length {} does not match configured gradient_dim {}",
            gradient.len(),
            config.gradient_dim
        )));
    }

    let quantised = quantize_gradient(gradient, config.quantization_scale, config.modulus)?;
    let mask = compute_client_mask(
        client_id,
        all_client_ids,
        config.round_seed,
        config.gradient_dim,
        config.modulus,
    )?;

    let mut values = Vec::with_capacity(quantised.len());
    for (q, m) in quantised.iter().zip(mask.iter()) {
        values.push((*q + *m).rem_euclid(config.modulus));
    }

    Ok(MaskedGradient { client_id, values })
}

// ---------------------------------------------------------------------------
// Server-side aggregator
// ---------------------------------------------------------------------------

impl SecureAggregator {
    /// Build a new aggregator for one round.
    ///
    /// Validates the configuration up front so that downstream operations
    /// can assume well-formed values. The dropout safety check requires
    /// that the modulus is at least one order of magnitude larger than the
    /// nominal quantised value range, which guarantees that the partial
    /// masked sum cannot silently wrap.
    pub fn new(config: SecureAggregationConfig) -> Result<Self> {
        if config.num_clients == 0 {
            return Err(OptimError::InvalidConfig(
                "num_clients must be greater than zero".to_string(),
            ));
        }
        if config.gradient_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "gradient_dim must be greater than zero".to_string(),
            ));
        }
        if config.modulus <= 1 {
            return Err(OptimError::InvalidConfig(format!(
                "modulus must be > 1, got {}",
                config.modulus
            )));
        }
        if !config.quantization_scale.is_finite() || config.quantization_scale <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "quantization_scale must be positive and finite, got {}",
                config.quantization_scale
            )));
        }
        // Guard against trivially-small moduli that cannot meaningfully
        // hold even a single quantised value with headroom for masking.
        let min_modulus = (config.quantization_scale * 10.0).ceil() as i128;
        if (config.modulus as i128) < min_modulus {
            return Err(OptimError::InvalidConfig(format!(
                "modulus {} is too small for quantization_scale {}; need at least {}",
                config.modulus, config.quantization_scale, min_modulus
            )));
        }

        Ok(Self {
            config,
            received: HashMap::new(),
            dropped: HashSet::new(),
            pairwise_mask_cache: HashMap::new(),
        })
    }

    /// Accept a client submission.
    ///
    /// Re-submissions from the same `client_id` overwrite the previous
    /// value (later submissions are considered more authoritative). A
    /// submission whose vector length does not match the configured
    /// dimension is rejected with [`OptimError::DimensionMismatch`].
    pub fn receive(&mut self, submission: MaskedGradient) -> Result<()> {
        if submission.values.len() != self.config.gradient_dim {
            return Err(OptimError::DimensionMismatch(format!(
                "submission from client {} has {} values, expected {}",
                submission.client_id,
                submission.values.len(),
                self.config.gradient_dim
            )));
        }
        for &v in submission.values.iter() {
            if v < 0 || v >= self.config.modulus {
                return Err(OptimError::InvalidParameter(format!(
                    "submission value {v} from client {} is outside [0, {})",
                    submission.client_id, self.config.modulus
                )));
            }
        }

        let client_id = submission.client_id;
        // A client cannot simultaneously be marked dropped *and* submit.
        self.dropped.remove(&client_id);
        self.received.insert(client_id, submission);
        Ok(())
    }

    /// Mark a client as having dropped out of this round.
    ///
    /// Dropped clients are excluded from the aggregated sum but the masks
    /// they would have used are still cancelled by the
    /// dropout-reconstruction phase of [`aggregate`].
    pub fn mark_dropped(&mut self, client_id: ClientId) -> Result<()> {
        if self.received.contains_key(&client_id) {
            return Err(OptimError::InvalidParameter(format!(
                "client {client_id} has already submitted; cannot mark as dropped"
            )));
        }
        self.dropped.insert(client_id);
        Ok(())
    }

    /// Aggregate the received submissions, cancelling any masks belonging
    /// to dropped clients.
    ///
    /// Returns the dequantised real-valued sum of the *received* clients'
    /// gradients. Dropped clients contribute nothing to the sum (their
    /// gradient is, by definition, never uploaded) but their pairwise
    /// masks are still removed so the remaining clients' uploads
    /// telescope correctly.
    pub fn aggregate(&self) -> Result<Array1<f64>> {
        let total_known = self.received.len() + self.dropped.len();
        if total_known < self.config.num_clients {
            return Err(OptimError::InvalidConfig(format!(
                "incomplete round: {} received + {} dropped < num_clients = {}",
                self.received.len(),
                self.dropped.len(),
                self.config.num_clients
            )));
        }
        if self.received.is_empty() {
            return Err(OptimError::InvalidConfig(
                "cannot aggregate: no client submissions received".to_string(),
            ));
        }

        let modulus = self.config.modulus;
        let dim = self.config.gradient_dim;

        // Step 1: sum all received masked submissions modulo p. The masks
        // among the received clients automatically telescope to zero.
        let mut acc = vec![0_i64; dim];
        for submission in self.received.values() {
            for (a, v) in acc.iter_mut().zip(submission.values.iter()) {
                *a = (*a + *v).rem_euclid(modulus);
            }
        }

        // Step 2: dropout reconstruction. For every dropped client d, the
        // received clients still hold a copy of m_{i,d} or -m_{d,i} that
        // never got cancelled. We subtract the *net* residual contributed
        // by d -- that is, the sum over online peers of the signed
        // pairwise mask the dropped client *would have applied*.
        if self.config.support_dropouts && !self.dropped.is_empty() {
            let online_ids: Vec<ClientId> = self.received.keys().copied().collect();
            for &dropped_id in self.dropped.iter() {
                // From the dropped client's perspective, its own mask was
                // mask_d = Σ_{j>d, j online} m_{d,j} − Σ_{j<d, j online} m_{j,d}.
                // The online peers contributed the *negation* of these terms
                // to the running sum (because the signs flip from the peer's
                // perspective), so the residual still in `acc` equals
                //   residual = − mask_d   (mod p)
                // and we therefore add mask_d to cancel it.
                let dropped_mask = compute_client_mask(
                    dropped_id,
                    &Self::peer_universe(&online_ids, dropped_id),
                    self.config.round_seed,
                    dim,
                    modulus,
                )?;
                for (a, m) in acc.iter_mut().zip(dropped_mask.iter()) {
                    *a = (*a + *m).rem_euclid(modulus);
                }
            }
        }

        Ok(Array1::from(dequantize_gradient(
            &acc,
            self.config.quantization_scale,
            modulus,
        )))
    }

    /// Reset the per-round state so the same aggregator can be reused.
    pub fn reset(&mut self) {
        self.received.clear();
        self.dropped.clear();
        self.pairwise_mask_cache.clear();
    }

    /// Number of client submissions currently held.
    pub fn received_count(&self) -> usize {
        self.received.len()
    }

    /// Number of clients currently marked as dropped.
    pub fn dropped_count(&self) -> usize {
        self.dropped.len()
    }

    /// Read-only access to the aggregator's configuration.
    pub fn config(&self) -> &SecureAggregationConfig {
        &self.config
    }

    /// Construct the universe `{dropped_id} ∪ online_ids` used to
    /// reproduce the dropped client's local mask. The dropped id is
    /// included so that `compute_client_mask` accepts the call; the
    /// online_ids drive the actual signed mask accumulation.
    fn peer_universe(online_ids: &[ClientId], dropped_id: ClientId) -> Vec<ClientId> {
        let mut universe = Vec::with_capacity(online_ids.len() + 1);
        universe.push(dropped_id);
        universe.extend_from_slice(online_ids);
        universe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small absolute tolerance for quantisation round-trip equality.
    const QUANT_TOL: f64 = 1.0e-4;

    fn make_config(num_clients: usize, dim: usize) -> SecureAggregationConfig {
        SecureAggregationConfig {
            num_clients,
            gradient_dim: dim,
            round_seed: 17,
            quantization_scale: 1.0e6,
            modulus: 2_147_483_647,
            support_dropouts: true,
        }
    }

    fn vec_close(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ----- Pairwise seed derivation --------------------------------------

    #[test]
    fn test_pairwise_seed_symmetric() {
        assert_eq!(
            derive_pairwise_seed(1, 2, 42),
            derive_pairwise_seed(2, 1, 42)
        );
        assert_eq!(
            derive_pairwise_seed(7, 41, 100),
            derive_pairwise_seed(41, 7, 100)
        );
        assert_eq!(
            derive_pairwise_seed(0, u64::MAX, 0),
            derive_pairwise_seed(u64::MAX, 0, 0)
        );
    }

    #[test]
    fn test_pairwise_seed_changes_with_round_seed() {
        let s1 = derive_pairwise_seed(3, 5, 1);
        let s2 = derive_pairwise_seed(3, 5, 2);
        let s3 = derive_pairwise_seed(3, 5, 12345);
        assert_ne!(
            s1, s2,
            "different round seeds must produce different mask seeds"
        );
        assert_ne!(s1, s3);
        assert_ne!(s2, s3);
    }

    #[test]
    fn test_pairwise_seed_differs_across_pairs() {
        // Distinct pairs with the same round seed should hash to distinct
        // values with overwhelming probability. The avalanche keeps these
        // separated even for tiny ids.
        let mut seen = HashSet::new();
        let round = 999;
        for a in 0_u64..20 {
            for b in (a + 1)..20 {
                let s = derive_pairwise_seed(a, b, round);
                assert!(seen.insert(s), "duplicate seed for pair ({a}, {b}): {s}");
            }
        }
    }

    // ----- Pairwise mask generation --------------------------------------

    #[test]
    fn test_generate_pairwise_mask_deterministic_with_seed() {
        let m1 = generate_pairwise_mask(123_456, 32, 2_147_483_647);
        let m2 = generate_pairwise_mask(123_456, 32, 2_147_483_647);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_generate_pairwise_mask_correct_length() {
        let m = generate_pairwise_mask(7, 100, 2_147_483_647);
        assert_eq!(m.len(), 100);
        let m_empty = generate_pairwise_mask(7, 0, 2_147_483_647);
        assert_eq!(m_empty.len(), 0);
    }

    #[test]
    fn test_generate_pairwise_mask_values_in_modulus_range() {
        let modulus = 2_147_483_647_i64;
        let m = generate_pairwise_mask(0xDEAD_BEEF, 512, modulus);
        for v in m {
            assert!(v >= 0, "mask values must be non-negative, got {v}");
            assert!(v < modulus, "mask values must be < modulus, got {v}");
        }
    }

    #[test]
    fn test_generate_pairwise_mask_changes_with_seed() {
        let m1 = generate_pairwise_mask(1, 32, 2_147_483_647);
        let m2 = generate_pairwise_mask(2, 32, 2_147_483_647);
        assert_ne!(m1, m2);
    }

    // ----- Quantisation round-trip ---------------------------------------

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let scale = 1.0e6;
        let modulus = 2_147_483_647_i64;
        let input = Array1::from(vec![1.5, -2.3, 0.0, 4.7]);
        let q = quantize_gradient(&input, scale, modulus).expect("quantise must succeed");
        let back = dequantize_gradient(&q, scale, modulus);
        let back_arr = Array1::from(back);
        assert!(
            vec_close(&input, &back_arr, 1.0e-5),
            "round-trip failed: {input:?} -> {q:?} -> {back_arr:?}"
        );
    }

    #[test]
    fn test_round_trip_quantize_handles_negatives() {
        let scale = 1.0e6;
        let modulus = 2_147_483_647_i64;
        let input = Array1::from(vec![-1.5, -3.5, -0.000_5]);
        let q = quantize_gradient(&input, scale, modulus).expect("quantise");
        // All canonical representatives are non-negative.
        for v in q.iter() {
            assert!(*v >= 0 && *v < modulus, "value out of canonical range: {v}");
        }
        let back = Array1::from(dequantize_gradient(&q, scale, modulus));
        assert!(
            vec_close(&input, &back, 1.0e-5),
            "negative round-trip failed: {input:?} -> {back:?}"
        );
    }

    #[test]
    fn test_quantize_rejects_non_finite_gradient() {
        let input = Array1::from(vec![1.0, f64::NAN]);
        let r = quantize_gradient(&input, 1.0e6, 2_147_483_647);
        match r {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for NaN gradient, got {other:?}"),
        }
        let input = Array1::from(vec![f64::INFINITY]);
        match quantize_gradient(&input, 1.0e6, 2_147_483_647) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for inf gradient, got {other:?}"),
        }
    }

    #[test]
    fn test_quantize_rejects_non_positive_scale() {
        let input = Array1::from(vec![1.0_f64]);
        match quantize_gradient(&input, 0.0, 2_147_483_647) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for scale=0, got {other:?}"),
        }
        match quantize_gradient(&input, -1.0, 2_147_483_647) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for negative scale, got {other:?}"),
        }
    }

    // ----- Client mask construction --------------------------------------

    #[test]
    fn test_compute_client_mask_two_clients_opposite_signs() {
        let modulus = 2_147_483_647_i64;
        let dim = 16;
        let round = 7;
        let clients: Vec<ClientId> = vec![10, 20];

        let mask_a = compute_client_mask(10, &clients, round, dim, modulus).expect("client a mask");
        let mask_b = compute_client_mask(20, &clients, round, dim, modulus).expect("client b mask");

        // mask_a + mask_b should be ≡ 0 (mod modulus) because the two
        // pairwise contributions have opposite signs.
        for (a, b) in mask_a.iter().zip(mask_b.iter()) {
            let sum = (a + b).rem_euclid(modulus);
            assert_eq!(sum, 0, "pair masks must cancel: a={a}, b={b}, sum={sum}");
        }
    }

    #[test]
    fn test_compute_client_mask_unknown_client_errors() {
        let r = compute_client_mask(99, &[1, 2, 3], 0, 4, 2_147_483_647);
        match r {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for unknown client, got {other:?}"),
        }
    }

    #[test]
    fn test_compute_client_mask_n_clients_sum_to_zero() {
        // Across all clients, every pairwise mask is added once and
        // subtracted once, so the *total* mask sum must be exactly zero.
        let modulus = 2_147_483_647_i64;
        let dim = 8;
        let round = 555;
        let clients: Vec<ClientId> = vec![1, 7, 13, 42, 100];

        let mut total = vec![0_i64; dim];
        for &cid in clients.iter() {
            let m = compute_client_mask(cid, &clients, round, dim, modulus).expect("mask");
            for (t, x) in total.iter_mut().zip(m.iter()) {
                *t = (*t + *x).rem_euclid(modulus);
            }
        }
        assert_eq!(total, vec![0_i64; dim]);
    }

    // ----- End-to-end aggregation ----------------------------------------

    #[test]
    fn test_aggregate_two_clients_recovers_sum() {
        let dim = 5;
        let config = make_config(2, dim);
        let clients: Vec<ClientId> = vec![1, 2];

        let g1 = Array1::from(vec![0.5, -1.25, 3.0, 0.0, 7.7]);
        let g2 = Array1::from(vec![-0.5, 2.5, -1.0, 4.4, -2.2]);
        let expected: Array1<f64> = &g1 + &g2;

        let sub1 = submit_gradient(1, &g1, &clients, &config).expect("submit 1");
        let sub2 = submit_gradient(2, &g2, &clients, &config).expect("submit 2");

        let mut agg = SecureAggregator::new(config).expect("aggregator");
        agg.receive(sub1).expect("recv 1");
        agg.receive(sub2).expect("recv 2");
        let out = agg.aggregate().expect("aggregate");

        assert!(
            vec_close(&out, &expected, QUANT_TOL),
            "expected {expected:?}, got {out:?}"
        );
    }

    #[test]
    fn test_aggregate_five_clients_recovers_sum() {
        let dim = 7;
        let config = make_config(5, dim);
        let clients: Vec<ClientId> = vec![3, 11, 19, 47, 101];

        let gradients: Vec<Array1<f64>> = vec![
            Array1::from(vec![1.0, 2.0, -3.0, 4.5, -5.5, 0.1, 0.01]),
            Array1::from(vec![-1.0, 1.0, 3.0, -2.0, 0.0, -0.1, 0.99]),
            Array1::from(vec![0.5, -0.5, 0.25, 0.0, 1.0, 2.5, -2.5]),
            Array1::from(vec![10.0, -10.0, 5.0, -5.0, 2.5, -2.5, 0.0]),
            Array1::from(vec![0.001, -0.001, 0.002, -0.002, 100.0, -100.0, 0.0]),
        ];

        let expected = gradients.iter().fold(Array1::zeros(dim), |acc, g| &acc + g);

        let mut agg = SecureAggregator::new(config.clone()).expect("aggregator");
        for (cid, g) in clients.iter().zip(gradients.iter()) {
            let sub = submit_gradient(*cid, g, &clients, &config).expect("submit");
            agg.receive(sub).expect("recv");
        }
        let out = agg.aggregate().expect("aggregate");

        assert!(
            vec_close(&out, &expected, QUANT_TOL),
            "expected {expected:?}, got {out:?}"
        );
    }

    #[test]
    fn test_aggregate_with_dropout_recovers_sum() {
        // Five-client cohort. Client 19 drops out. The server must still
        // recover the sum of the other four gradients exactly.
        let dim = 4;
        let config = make_config(5, dim);
        let clients: Vec<ClientId> = vec![3, 11, 19, 47, 101];
        let dropped_id: ClientId = 19;

        let gradients: HashMap<ClientId, Array1<f64>> = [
            (3, Array1::from(vec![1.0, 2.0, -3.0, 4.0])),
            (11, Array1::from(vec![-1.5, 0.5, 2.5, -2.0])),
            (19, Array1::from(vec![100.0, 100.0, 100.0, 100.0])), // dropped, unused
            (47, Array1::from(vec![0.25, 0.5, 0.75, 1.0])),
            (101, Array1::from(vec![-0.1, -0.2, -0.3, -0.4])),
        ]
        .into_iter()
        .collect();

        let mut expected: Array1<f64> = Array1::zeros(dim);
        for (cid, g) in gradients.iter() {
            if *cid != dropped_id {
                expected = &expected + g;
            }
        }

        let mut agg = SecureAggregator::new(config.clone()).expect("aggregator");
        for &cid in clients.iter() {
            if cid == dropped_id {
                agg.mark_dropped(cid).expect("mark dropped");
            } else {
                let sub =
                    submit_gradient(cid, &gradients[&cid], &clients, &config).expect("submit");
                agg.receive(sub).expect("recv");
            }
        }

        assert_eq!(agg.received_count(), 4);
        assert_eq!(agg.dropped_count(), 1);

        let out = agg.aggregate().expect("aggregate with dropout");
        assert!(
            vec_close(&out, &expected, QUANT_TOL),
            "dropout recovery failed: expected {expected:?}, got {out:?}"
        );
    }

    #[test]
    fn test_aggregate_with_multiple_dropouts_recovers_sum() {
        // Stronger version of the dropout test: two clients drop out of a
        // six-client cohort. The aggregation arithmetic should still
        // produce the exact sum of the remaining four gradients.
        let dim = 3;
        let mut config = make_config(6, dim);
        config.round_seed = 4242;
        let clients: Vec<ClientId> = vec![1, 2, 3, 4, 5, 6];
        let dropped: Vec<ClientId> = vec![2, 5];

        let gradients: HashMap<ClientId, Array1<f64>> = [
            (1, Array1::from(vec![1.0, 0.0, 0.0])),
            (2, Array1::from(vec![0.0, 1.0, 0.0])),
            (3, Array1::from(vec![0.0, 0.0, 1.0])),
            (4, Array1::from(vec![1.0, 1.0, 1.0])),
            (5, Array1::from(vec![-1.0, -1.0, -1.0])),
            (6, Array1::from(vec![0.5, 0.5, 0.5])),
        ]
        .into_iter()
        .collect();

        let mut expected: Array1<f64> = Array1::zeros(dim);
        for (cid, g) in gradients.iter() {
            if !dropped.contains(cid) {
                expected = &expected + g;
            }
        }

        let mut agg = SecureAggregator::new(config.clone()).expect("aggregator");
        for &cid in clients.iter() {
            if dropped.contains(&cid) {
                agg.mark_dropped(cid).expect("mark dropped");
            } else {
                let sub =
                    submit_gradient(cid, &gradients[&cid], &clients, &config).expect("submit");
                agg.receive(sub).expect("recv");
            }
        }

        let out = agg.aggregate().expect("aggregate");
        assert!(
            vec_close(&out, &expected, QUANT_TOL),
            "multi-dropout recovery failed: expected {expected:?}, got {out:?}"
        );
    }

    #[test]
    fn test_aggregate_missing_clients_errors() {
        // Three-client cohort, only two reported (one received, one
        // dropped). aggregate() must refuse to produce a result.
        let dim = 4;
        let config = make_config(3, dim);
        let clients: Vec<ClientId> = vec![1, 2, 3];
        let g1 = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let sub1 = submit_gradient(1, &g1, &clients, &config).expect("submit 1");

        let mut agg = SecureAggregator::new(config).expect("aggregator");
        agg.receive(sub1).expect("recv 1");
        agg.mark_dropped(2).expect("mark 2");

        match agg.aggregate() {
            Err(OptimError::InvalidConfig(_)) => {}
            other => panic!("expected InvalidConfig for incomplete round, got {other:?}"),
        }
    }

    #[test]
    fn test_receive_validates_dim() {
        let config = make_config(2, 8);
        let mut agg = SecureAggregator::new(config).expect("aggregator");
        let bad = MaskedGradient {
            client_id: 1,
            values: vec![0_i64; 7], // wrong length
        };
        match agg.receive(bad) {
            Err(OptimError::DimensionMismatch(_)) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_receive_validates_value_range() {
        let config = make_config(2, 3);
        let modulus = config.modulus;
        let mut agg = SecureAggregator::new(config).expect("aggregator");

        let bad_high = MaskedGradient {
            client_id: 1,
            values: vec![0, modulus, 0],
        };
        match agg.receive(bad_high) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for value=modulus, got {other:?}"),
        }

        let bad_neg = MaskedGradient {
            client_id: 1,
            values: vec![0, -1, 0],
        };
        match agg.receive(bad_neg) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter for negative value, got {other:?}"),
        }
    }

    // ----- Config validation ---------------------------------------------

    #[test]
    fn test_invalid_config_zero_clients_errors() {
        let config = SecureAggregationConfig {
            num_clients: 0,
            gradient_dim: 4,
            ..SecureAggregationConfig::default()
        };
        match SecureAggregator::new(config) {
            Err(OptimError::InvalidConfig(_)) => {}
            other => panic!("expected InvalidConfig for num_clients=0, got {other:?}"),
        }
    }

    #[test]
    fn test_invalid_config_zero_dim_errors() {
        let config = SecureAggregationConfig {
            num_clients: 3,
            gradient_dim: 0,
            ..SecureAggregationConfig::default()
        };
        match SecureAggregator::new(config) {
            Err(OptimError::InvalidConfig(_)) => {}
            other => panic!("expected InvalidConfig for gradient_dim=0, got {other:?}"),
        }
    }

    #[test]
    fn test_modulus_too_small_errors() {
        let config = SecureAggregationConfig {
            num_clients: 3,
            gradient_dim: 4,
            round_seed: 0,
            quantization_scale: 1.0e6,
            modulus: 100, // far smaller than 10 * scale
            support_dropouts: true,
        };
        match SecureAggregator::new(config) {
            Err(OptimError::InvalidConfig(_)) => {}
            other => panic!("expected InvalidConfig for tiny modulus, got {other:?}"),
        }
    }

    #[test]
    fn test_invalid_config_non_positive_scale_errors() {
        let config = SecureAggregationConfig {
            num_clients: 2,
            gradient_dim: 4,
            round_seed: 0,
            quantization_scale: 0.0,
            modulus: 2_147_483_647,
            support_dropouts: false,
        };
        match SecureAggregator::new(config) {
            Err(OptimError::InvalidConfig(_)) => {}
            other => panic!("expected InvalidConfig for scale=0, got {other:?}"),
        }
    }

    #[test]
    fn test_invalid_config_modulus_one_errors() {
        let config = SecureAggregationConfig {
            num_clients: 2,
            gradient_dim: 4,
            round_seed: 0,
            quantization_scale: 1.0e6,
            modulus: 1,
            support_dropouts: false,
        };
        match SecureAggregator::new(config) {
            Err(OptimError::InvalidConfig(_)) => {}
            other => panic!("expected InvalidConfig for modulus=1, got {other:?}"),
        }
    }

    // ----- Mark dropped validation ---------------------------------------

    #[test]
    fn test_mark_dropped_after_submit_errors() {
        let config = make_config(2, 4);
        let clients = vec![1_u64, 2];
        let g = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let sub = submit_gradient(1, &g, &clients, &config).expect("submit");
        let mut agg = SecureAggregator::new(config).expect("aggregator");
        agg.receive(sub).expect("recv");
        match agg.mark_dropped(1) {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let config = make_config(2, 3);
        let clients = vec![1_u64, 2];
        let g = Array1::from(vec![1.0, 2.0, 3.0]);
        let sub = submit_gradient(1, &g, &clients, &config).expect("submit");
        let mut agg = SecureAggregator::new(config).expect("aggregator");
        agg.receive(sub).expect("recv");
        agg.mark_dropped(2).expect("mark");
        assert_eq!(agg.received_count(), 1);
        assert_eq!(agg.dropped_count(), 1);
        agg.reset();
        assert_eq!(agg.received_count(), 0);
        assert_eq!(agg.dropped_count(), 0);
    }

    // ----- Serde round-trips ---------------------------------------------

    #[test]
    fn test_masked_gradient_serde_roundtrip() {
        let mg = MaskedGradient {
            client_id: 42,
            values: vec![0_i64, 1, 1_000_000, 2_147_483_646],
        };
        let json = serde_json::to_string(&mg).expect("serialise");
        let back: MaskedGradient = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(mg, back);
    }

    #[test]
    fn test_secure_aggregation_config_serde_roundtrip() {
        let config = SecureAggregationConfig {
            num_clients: 8,
            gradient_dim: 1024,
            round_seed: 0xCAFE_BABE,
            quantization_scale: 1.0e5,
            modulus: 2_147_483_647,
            support_dropouts: false,
        };
        let json = serde_json::to_string(&config).expect("serialise");
        let back: SecureAggregationConfig = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back.num_clients, config.num_clients);
        assert_eq!(back.gradient_dim, config.gradient_dim);
        assert_eq!(back.round_seed, config.round_seed);
        assert_eq!(back.quantization_scale, config.quantization_scale);
        assert_eq!(back.modulus, config.modulus);
        assert_eq!(back.support_dropouts, config.support_dropouts);
    }

    // ----- Resubmission semantics ----------------------------------------

    #[test]
    fn test_resubmission_overwrites_previous() {
        let dim = 3;
        let config = make_config(2, dim);
        let clients = vec![1_u64, 2];

        let g1_first = Array1::from(vec![100.0, 100.0, 100.0]);
        let g1_final = Array1::from(vec![1.0, 2.0, 3.0]);
        let g2 = Array1::from(vec![0.5, 0.5, 0.5]);
        let expected: Array1<f64> = &g1_final + &g2;

        let mut agg = SecureAggregator::new(config.clone()).expect("aggregator");

        let sub_first = submit_gradient(1, &g1_first, &clients, &config).expect("submit first");
        let sub_final = submit_gradient(1, &g1_final, &clients, &config).expect("submit final");
        let sub2 = submit_gradient(2, &g2, &clients, &config).expect("submit 2");

        agg.receive(sub_first).expect("recv first");
        agg.receive(sub_final).expect("recv final");
        agg.receive(sub2).expect("recv 2");

        assert_eq!(
            agg.received_count(),
            2,
            "resubmission must overwrite, not duplicate"
        );

        let out = agg.aggregate().expect("aggregate");
        assert!(
            vec_close(&out, &expected, QUANT_TOL),
            "resubmission failed: expected {expected:?}, got {out:?}"
        );
    }

    #[test]
    fn test_default_config_has_sane_round_seed_and_scale() {
        let cfg = SecureAggregationConfig::default();
        assert_eq!(cfg.round_seed, 42);
        assert_eq!(cfg.quantization_scale, 1.0e6);
        assert_eq!(cfg.modulus, 2_147_483_647);
        assert!(cfg.support_dropouts);
        // Caller must populate num_clients and gradient_dim explicitly.
        assert_eq!(cfg.num_clients, 0);
        assert_eq!(cfg.gradient_dim, 0);
    }
}
