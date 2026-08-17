//! Hashing primitives for the tamper-evident audit trail.
//!
//! Everything in this module is deliberately *deterministic* and
//! *domain-separated*:
//!
//! * A digest is only ever computed over a **length-prefixed canonical
//!   encoding**, never over `serde_json` output. `AuditEvent` carries two
//!   `HashMap` fields, and `HashMap` iteration order is randomised per
//!   process, so a JSON digest is not reproducible across runs (or after a
//!   `clone`) and therefore cannot be used as a commitment.
//! * Leaves and interior Merkle nodes are hashed under different tags, so a
//!   published interior node cannot be replayed as a leaf (second-preimage
//!   resistance for Merkle trees; see Certificate Transparency, RFC 6962 §2.1
//!   for the same construction).
//!
//! The keyed digest is an HMAC-SHA256 (RFC 2104) *message authentication
//! code*. It is not an asymmetric digital signature: it authenticates the
//! chain to a holder of the chain key and gives no non-repudiation against
//! that holder. Callers requiring non-repudiation are rejected explicitly
//! rather than being handed a MAC that looks like a signature.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Debug;

use super::types::{AuditEvent, ComplianceFramework, ComplianceStatus};

/// Digest length in bytes (SHA-256).
pub const DIGEST_LEN: usize = 32;

/// A 32-byte SHA-256 digest.
pub type Digest32 = [u8; DIGEST_LEN];

/// Domain tag for a Merkle leaf.
const TAG_LEAF: u8 = 0x00;
/// Domain tag for an interior Merkle node.
const TAG_NODE: u8 = 0x01;
/// Domain tag for a hash-chain link.
const TAG_LINK: u8 = 0x02;
/// Domain tag for the canonical encoding of an audit event.
const TAG_EVENT: u8 = 0x03;

/// The first link of an empty chain (`H(TAG_LINK || "optirs-audit-genesis")`).
pub fn genesis_link() -> Digest32 {
    sha256(&[&[TAG_LINK], b"optirs-audit-genesis"])
}

/// Draw a fresh 32-byte key from OS entropy.
///
/// Used for the audit chain's authentication tags and for keyed integrity
/// proofs. A constant key would let anyone recompute every tag after
/// rewriting the chain, which is the whole point of having one.
pub fn random_key() -> Digest32 {
    use scirs2_core::random::{thread_rng, Rng};
    let mut rng = thread_rng();
    let mut key = [0u8; DIGEST_LEN];
    for chunk in key.chunks_mut(8) {
        let word: u64 = rng.gen_range(0u64..u64::MAX);
        let bytes = word.to_le_bytes();
        let len = chunk.len();
        chunk.copy_from_slice(&bytes[..len]);
    }
    key
}

/// SHA-256 over the concatenation of `parts`.
pub fn sha256(parts: &[&[u8]]) -> Digest32 {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    let out = hasher.finalize().to_vec();
    let mut digest = [0u8; DIGEST_LEN];
    // SHA-256 always produces exactly `DIGEST_LEN` bytes.
    let n = out.len().min(DIGEST_LEN);
    digest[..n].copy_from_slice(&out[..n]);
    digest
}

/// Hash a Merkle leaf: `H(TAG_LEAF || payload)`.
pub fn hash_leaf(payload: &[u8]) -> Digest32 {
    sha256(&[&[TAG_LEAF], payload])
}

/// Hash an interior Merkle node: `H(TAG_NODE || left || right)`.
pub fn hash_node(left: &Digest32, right: &Digest32) -> Digest32 {
    sha256(&[&[TAG_NODE], left, right])
}

/// Extend a hash chain: `H(TAG_LINK || previous || leaf)`.
pub fn hash_link(previous: &Digest32, leaf: &Digest32) -> Digest32 {
    sha256(&[&[TAG_LINK], previous, leaf])
}

/// HMAC-SHA256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> Digest32 {
    const BLOCK: usize = 64;
    let mut block_key = [0u8; BLOCK];
    if key.len() > BLOCK {
        let digest = sha256(&[key]);
        block_key[..DIGEST_LEN].copy_from_slice(&digest);
    } else {
        block_key[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for index in 0..BLOCK {
        ipad[index] ^= block_key[index];
        opad[index] ^= block_key[index];
    }

    let inner = sha256(&[&ipad, message]);
    sha256(&[&opad, &inner])
}

/// Constant-time equality for digests.
///
/// Verification results are observable, so a short-circuiting `==` would leak
/// the position of the first differing byte through timing.
pub fn digests_equal(left: &Digest32, right: &Digest32) -> bool {
    let mut difference = 0u8;
    for index in 0..DIGEST_LEN {
        difference |= left[index] ^ right[index];
    }
    difference == 0
}

/// Append a length-prefixed byte string.
fn push_bytes(buffer: &mut Vec<u8>, bytes: &[u8]) {
    buffer.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buffer.extend_from_slice(bytes);
}

/// Append a length-prefixed UTF-8 string.
fn push_str(buffer: &mut Vec<u8>, value: &str) {
    push_bytes(buffer, value.as_bytes());
}

/// Append a length-prefixed, lexicographically ordered list of strings.
fn push_str_list(buffer: &mut Vec<u8>, values: &[String]) {
    buffer.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        push_str(buffer, value);
    }
}

/// Append a `HashMap<String, String>` in key order.
fn push_str_map(buffer: &mut Vec<u8>, map: &std::collections::HashMap<String, String>) {
    let ordered: BTreeMap<&String, &String> = map.iter().collect();
    buffer.extend_from_slice(&(ordered.len() as u64).to_le_bytes());
    for (key, value) in ordered {
        push_str(buffer, key);
        push_str(buffer, value);
    }
}

/// A stable textual key for a compliance framework.
pub fn framework_key(framework: &ComplianceFramework) -> String {
    match framework {
        ComplianceFramework::GDPR => "GDPR".to_string(),
        ComplianceFramework::CCPA => "CCPA".to_string(),
        ComplianceFramework::HIPAA => "HIPAA".to_string(),
        ComplianceFramework::SOX => "SOX".to_string(),
        ComplianceFramework::FISMA => "FISMA".to_string(),
        ComplianceFramework::ISO27001 => "ISO27001".to_string(),
        ComplianceFramework::NISTPrivacy => "NISTPrivacy".to_string(),
        ComplianceFramework::Custom(name) => format!("Custom:{name}"),
    }
}

/// A stable textual key for a compliance status.
fn status_key(status: &ComplianceStatus) -> String {
    match status {
        ComplianceStatus::Compliant => "Compliant".to_string(),
        ComplianceStatus::NonCompliant(reason) => format!("NonCompliant:{reason}"),
        ComplianceStatus::RequiresReview(reason) => format!("RequiresReview:{reason}"),
        ComplianceStatus::NotApplicable => "NotApplicable".to_string(),
    }
}

/// A stable discriminant for an audit event type.
pub fn event_type_key(event_type: &super::types::AuditEventType) -> &'static str {
    use super::types::AuditEventType as E;
    match event_type {
        E::PrivacyBudgetAllocation => "PrivacyBudgetAllocation",
        E::PrivacyBudgetConsumption => "PrivacyBudgetConsumption",
        E::GradientComputation => "GradientComputation",
        E::ModelParameterUpdate => "ModelParameterUpdate",
        E::DataAccess => "DataAccess",
        E::UserConsent => "UserConsent",
        E::DataDeletion => "DataDeletion",
        E::AnonymizationProcess => "AnonymizationProcess",
        E::SecurityIncident => "SecurityIncident",
        E::ComplianceCheck => "ComplianceCheck",
        E::ConfigurationChange => "ConfigurationChange",
        E::SystemLifecycle => "SystemLifecycle",
    }
}

/// Deterministic canonical encoding of an audit event.
///
/// Every field is length-prefixed and every map is emitted in key order, so
/// the encoding is byte-identical in every process for a given event, and any
/// single-byte change to any field changes the encoding.
///
/// The `signature` field is deliberately **excluded**: the signature is
/// computed over this encoding, so including it would be circular.
pub fn canonical_event_bytes(event: &AuditEvent) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(512);
    buffer.push(TAG_EVENT);
    push_str(&mut buffer, &event.id);
    buffer.extend_from_slice(&event.timestamp.to_le_bytes());
    push_str(&mut buffer, event_type_key(&event.event_type));
    push_str(&mut buffer, &event.actor);

    push_str(&mut buffer, &event.data.description);
    push_str_list(&mut buffer, &event.data.affected_data_subjects);
    push_str_list(&mut buffer, &event.data.data_categories);
    push_str_list(&mut buffer, &event.data.processing_purposes);
    push_str_list(&mut buffer, &event.data.legal_basis);
    push_str_list(&mut buffer, &event.data.technical_measures);
    push_str_map(&mut buffer, &event.data.metadata);

    buffer.extend_from_slice(&event.privacy_context.epsilon_budget.to_bits().to_le_bytes());
    buffer.extend_from_slice(&event.privacy_context.delta_budget.to_bits().to_le_bytes());
    push_str(&mut buffer, &event.privacy_context.privacy_mechanism);
    buffer.push(u8::from(event.privacy_context.data_minimization));
    buffer.push(u8::from(event.privacy_context.purpose_limitation));
    buffer.push(u8::from(event.privacy_context.storage_limitation));

    let annotations: BTreeMap<String, String> = event
        .compliance_annotations
        .iter()
        .map(|(framework, status)| (framework_key(framework), status_key(status)))
        .collect();
    buffer.extend_from_slice(&(annotations.len() as u64).to_le_bytes());
    for (framework, status) in annotations {
        push_str(&mut buffer, &framework);
        push_str(&mut buffer, &status);
    }

    buffer
}

/// The Merkle leaf digest of an audit event.
pub fn event_leaf_digest(event: &AuditEvent) -> Digest32 {
    hash_leaf(&canonical_event_bytes(event))
}

/// Deterministic canonical encoding of a released value vector.
///
/// The length is prefixed and every element is emitted as its IEEE-754 bit
/// pattern, so the encoding is exact (no decimal rounding) and two vectors of
/// different length can never encode identically. An element that cannot be
/// represented as `f64` is an error rather than a silently substituted zero.
pub fn canonical_array_bytes<T: Float + Debug + Send + Sync + 'static>(
    data: &Array1<T>,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::with_capacity(8 + data.len() * 8);
    buffer.extend_from_slice(&(data.len() as u64).to_le_bytes());
    for (index, value) in data.iter().enumerate() {
        let as_f64 = value.to_f64().ok_or_else(|| {
            OptimError::InvalidParameter(format!(
                "element {index} of the released vector cannot be represented as f64, so it \
                 cannot be committed to"
            ))
        })?;
        buffer.extend_from_slice(&as_f64.to_bits().to_le_bytes());
    }
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::enhanced_audit::types::{AuditEventData, AuditEventType, PrivacyContext};
    use std::collections::HashMap;

    fn sample_event() -> AuditEvent {
        let mut metadata = HashMap::new();
        // Insert in an order that differs from the sorted order so a
        // non-canonical encoder would be order-sensitive.
        metadata.insert("zulu".to_string(), "26".to_string());
        metadata.insert("alpha".to_string(), "1".to_string());
        metadata.insert("mike".to_string(), "13".to_string());
        AuditEvent {
            id: "event-1".to_string(),
            timestamp: 1_700_000_000,
            event_type: AuditEventType::DataAccess,
            actor: "trainer".to_string(),
            data: AuditEventData {
                description: "read a training shard".to_string(),
                affected_data_subjects: vec!["subject-a".to_string()],
                data_categories: vec!["personal_data".to_string()],
                processing_purposes: vec!["ml_training".to_string()],
                legal_basis: vec!["consent".to_string()],
                technical_measures: vec!["differential_privacy".to_string()],
                metadata,
            },
            privacy_context: PrivacyContext {
                epsilon_budget: 0.5,
                delta_budget: 1e-6,
                privacy_mechanism: "dp_sgd".to_string(),
                data_minimization: true,
                purpose_limitation: true,
                storage_limitation: true,
            },
            signature: None,
            compliance_annotations: HashMap::new(),
        }
    }

    #[test]
    fn canonical_encoding_is_stable_under_map_reinsertion() {
        // The same logical event built with a different `HashMap` insertion
        // order must produce the same bytes. A `serde_json` digest does not
        // have this property, which is why the JSON encoding was replaced.
        let left = sample_event();
        let mut right = sample_event();
        let mut metadata = HashMap::new();
        metadata.insert("mike".to_string(), "13".to_string());
        metadata.insert("zulu".to_string(), "26".to_string());
        metadata.insert("alpha".to_string(), "1".to_string());
        right.data.metadata = metadata;

        assert_eq!(canonical_event_bytes(&left), canonical_event_bytes(&right));
    }

    #[test]
    fn a_single_changed_byte_changes_the_leaf_digest() {
        let event = sample_event();
        let baseline = event_leaf_digest(&event);

        let mut tampered = event.clone();
        tampered.privacy_context.epsilon_budget = 0.5 + f64::EPSILON;
        assert!(!digests_equal(&baseline, &event_leaf_digest(&tampered)));

        let mut tampered = event.clone();
        tampered.actor.push('x');
        assert!(!digests_equal(&baseline, &event_leaf_digest(&tampered)));

        let mut tampered = event.clone();
        tampered.timestamp += 1;
        assert!(!digests_equal(&baseline, &event_leaf_digest(&tampered)));
    }

    #[test]
    fn length_prefixing_prevents_field_boundary_collisions() {
        // Without length prefixes, moving a character from one field to the
        // next would leave the concatenation unchanged.
        let mut left = sample_event();
        left.id = "ab".to_string();
        left.actor = "cd".to_string();
        let mut right = sample_event();
        right.id = "a".to_string();
        right.actor = "bcd".to_string();

        assert!(!digests_equal(
            &event_leaf_digest(&left),
            &event_leaf_digest(&right)
        ));
    }

    #[test]
    fn leaf_and_node_hashes_are_domain_separated() {
        let payload = [7u8; DIGEST_LEN];
        let leaf = hash_leaf(&payload);
        let node = hash_node(&payload, &payload);
        assert!(!digests_equal(&leaf, &node));
    }

    #[test]
    fn hmac_sha256_matches_the_rfc4231_test_vectors() {
        // RFC 4231 test case 1: key = 20 x 0x0b, data = "Hi There".
        let mac = hmac_sha256(&[0x0bu8; 20], b"Hi There");
        let expected = "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7";
        assert_eq!(hex(&mac), expected);

        // RFC 4231 test case 2: key = "Jefe", data = "what do ya want for nothing?".
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
        assert_eq!(hex(&mac), expected);

        // RFC 4231 test case 3: key = 20 x 0xaa, data = 50 x 0xdd.
        let mac = hmac_sha256(&[0xaau8; 20], &[0xddu8; 50]);
        let expected = "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe";
        assert_eq!(hex(&mac), expected);

        // RFC 4231 test case 6: key longer than the block size (131 x 0xaa).
        let mac = hmac_sha256(
            &[0xaau8; 131],
            b"Test Using Larger Than Block-Size Key - Hash Key First",
        );
        let expected = "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54";
        assert_eq!(hex(&mac), expected);
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
