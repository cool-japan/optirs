//! Tamper-evident audit storage: a real Merkle tree, a real hash chain, and
//! an audit trail whose integrity check actually re-derives every commitment
//! from the stored events.
//!
//! # Why this module exists
//!
//! The previous implementation of `MerkleTree::verify_integrity` was
//!
//! ```text
//! !self.nodes.is_empty() && self.root_hash.is_some()
//! ```
//!
//! which is true for *any* tree that has ever had a leaf added, whatever the
//! leaves contain. The audit chain delegated to it, so the tamper evidence the
//! chain exists to provide did not exist: mutating any stored event left
//! verification reporting success.
//!
//! What is implemented here instead:
//!
//! * [`MerkleTree`] keeps only the leaf digests and re-derives the root, with
//!   domain-separated leaf/interior hashing (RFC 6962 style) and real
//!   inclusion proofs ([`MerkleTree::inclusion_proof`] /
//!   [`MerkleTree::verify_inclusion_proof`]).
//! * [`AuditChain`] maintains a genuine chain, `link[i] = H(link[i-1] ||
//!   leaf[i])`, plus an HMAC-SHA256 tag per link under a chain key generated
//!   from OS entropy. Both are re-derived on verification.
//! * [`AuditTrail::verify_integrity`] re-hashes every stored event and
//!   compares against the commitments recorded at insertion time, so any
//!   single-byte change to any field of any event is detected.

use crate::error::{OptimError, Result};
use std::collections::{HashMap, VecDeque};

use super::hashing::{
    digests_equal, event_leaf_digest, genesis_link, hash_leaf, hash_link, hash_node, hmac_sha256,
    random_key, Digest32,
};
use super::types::{AuditEvent, AuditEventType, AuditQueryCriteria};

/// One step of a Merkle inclusion proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerkleProofStep {
    /// Sibling digest to combine with the running digest.
    pub sibling: Digest32,
    /// Whether the sibling sits on the left of the running digest.
    pub sibling_is_left: bool,
}

/// A Merkle inclusion proof for a single leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleProof {
    /// Index of the proved leaf.
    pub leaf_index: usize,
    /// Number of leaves the tree held when the proof was produced.
    pub leaf_count: usize,
    /// Sibling path from the leaf up to the root.
    pub steps: Vec<MerkleProofStep>,
}

/// Merkle tree over audit-event leaf digests.
///
/// Only the leaves are stored; the root is a pure function of them. Leaves are
/// hashed under a different domain tag from interior nodes, so promoting the
/// lone node of an odd level (as done here, and in Certificate Transparency)
/// cannot be abused to present an interior node as a leaf.
pub struct MerkleTree {
    /// Leaf digests, in insertion order.
    leaves: Vec<Digest32>,
    /// Root committed by the most recent `push_leaf_digest`/`add_leaf`.
    committed_root: Option<Digest32>,
}

impl MerkleTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            leaves: Vec::new(),
            committed_root: None,
        }
    }

    /// Number of leaves.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Whether the tree holds no leaves.
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Height of the tree (0 for an empty tree, 1 for a single leaf).
    pub fn depth(&self) -> usize {
        let mut width = self.leaves.len();
        if width == 0 {
            return 0;
        }
        let mut depth = 1;
        while width > 1 {
            width = width.div_ceil(2);
            depth += 1;
        }
        depth
    }

    /// The committed root, if the tree is non-empty.
    pub fn root_hash(&self) -> Option<Vec<u8>> {
        self.committed_root.map(|root| root.to_vec())
    }

    /// The committed root as a fixed-size digest.
    pub fn root(&self) -> Option<Digest32> {
        self.committed_root
    }

    /// Append a leaf digest that has already been domain-tagged.
    pub fn push_leaf_digest(&mut self, digest: Digest32) {
        self.leaves.push(digest);
        self.committed_root = Self::compute_root(&self.leaves);
    }

    /// Hash `payload` as a leaf and append it.
    ///
    /// Note the change of meaning relative to the pre-0.3.2 signature: the
    /// argument is the leaf *payload*, and the domain-separated leaf hash is
    /// applied here. Pass an already-hashed digest through
    /// [`MerkleTree::push_leaf_digest`] instead.
    pub fn add_leaf(&mut self, payload: Vec<u8>) -> Result<()> {
        if payload.is_empty() {
            return Err(OptimError::InvalidParameter(
                "a Merkle leaf payload must not be empty".to_string(),
            ));
        }
        self.push_leaf_digest(hash_leaf(&payload));
        Ok(())
    }

    /// Fold a level of digests into the level above.
    fn fold_level(level: &[Digest32]) -> Vec<Digest32> {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                next.push(hash_node(&pair[0], &pair[1]));
            } else {
                // Lone node is promoted unchanged; safe because leaves and
                // interior nodes are hashed under distinct domain tags.
                next.push(pair[0]);
            }
        }
        next
    }

    /// Recompute the root from a leaf slice.
    pub fn compute_root(leaves: &[Digest32]) -> Option<Digest32> {
        if leaves.is_empty() {
            return None;
        }
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            level = Self::fold_level(&level);
        }
        level.into_iter().next()
    }

    /// Produce an inclusion proof for the leaf at `leaf_index`.
    pub fn inclusion_proof(&self, leaf_index: usize) -> Result<MerkleProof> {
        if leaf_index >= self.leaves.len() {
            return Err(OptimError::InvalidParameter(format!(
                "leaf index {leaf_index} is out of range for a tree with {} leaves",
                self.leaves.len()
            )));
        }

        let mut steps = Vec::new();
        let mut level = self.leaves.clone();
        let mut index = leaf_index;
        while level.len() > 1 {
            let sibling_index = if index.is_multiple_of(2) {
                index + 1
            } else {
                index - 1
            };
            if sibling_index < level.len() {
                steps.push(MerkleProofStep {
                    sibling: level[sibling_index],
                    sibling_is_left: sibling_index < index,
                });
            }
            index /= 2;
            level = Self::fold_level(&level);
        }

        Ok(MerkleProof {
            leaf_index,
            leaf_count: self.leaves.len(),
            steps,
        })
    }

    /// Verify an inclusion proof against a root.
    pub fn verify_inclusion_proof(root: &Digest32, leaf: &Digest32, proof: &MerkleProof) -> bool {
        let mut running = *leaf;
        for step in &proof.steps {
            running = if step.sibling_is_left {
                hash_node(&step.sibling, &running)
            } else {
                hash_node(&running, &step.sibling)
            };
        }
        digests_equal(root, &running)
    }

    /// Re-derive the root from the stored leaves and compare with the
    /// committed root.
    ///
    /// An empty tree is trivially consistent (and has no root), so this
    /// returns `true` for it. Use [`MerkleTree::verify_against_leaves`] when
    /// the leaves themselves must be re-derived from an external source --
    /// that is the check that detects a mutated audit event.
    pub fn verify_integrity(&self) -> bool {
        match (Self::compute_root(&self.leaves), self.committed_root) {
            (None, None) => true,
            (Some(derived), Some(committed)) => digests_equal(&derived, &committed),
            _ => false,
        }
    }

    /// Check that `leaves` reproduces the committed root, leaf for leaf.
    pub fn verify_against_leaves(&self, leaves: &[Digest32]) -> Result<()> {
        if leaves.len() != self.leaves.len() {
            return Err(OptimError::InvalidState(format!(
                "Merkle leaf count mismatch: the tree commits to {} leaves, {} were supplied",
                self.leaves.len(),
                leaves.len()
            )));
        }
        for (index, (committed, supplied)) in self.leaves.iter().zip(leaves.iter()).enumerate() {
            if !digests_equal(committed, supplied) {
                return Err(OptimError::InvalidState(format!(
                    "Merkle leaf {index} does not match the committed digest"
                )));
            }
        }
        let derived = Self::compute_root(leaves);
        match (derived, self.committed_root) {
            (None, None) => Ok(()),
            (Some(derived), Some(committed)) if digests_equal(&derived, &committed) => Ok(()),
            _ => Err(OptimError::InvalidState(
                "the Merkle root re-derived from the supplied leaves does not match the \
                 committed root"
                    .to_string(),
            )),
        }
    }
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Cryptographic audit chain.
///
/// Maintains three independent commitments per event: the leaf digest, the
/// hash-chain link binding it to every preceding event, and an HMAC-SHA256 tag
/// over the link. All three are re-derived by [`AuditChain::verify_events`].
pub struct AuditChain {
    /// Leaf digest of each event, in insertion order.
    leaf_digests: Vec<Digest32>,
    /// `hash_chain[i] = H(TAG_LINK || hash_chain[i-1] || leaf_digests[i])`,
    /// with `hash_chain[-1]` the genesis link.
    hash_chain: Vec<Digest32>,
    /// HMAC-SHA256 tag over each chain link.
    ///
    /// This is a *message authentication code*, not an asymmetric digital
    /// signature: it proves the chain was extended by a holder of
    /// `chain_key`, and provides no non-repudiation against that holder.
    link_macs: Vec<Digest32>,
    /// Merkle tree over the leaf digests.
    merkle_tree: MerkleTree,
    /// Chain authentication key, drawn from OS entropy at construction.
    chain_key: Digest32,
}

impl AuditChain {
    /// Create a new chain with a fresh key from OS entropy.
    pub fn new() -> Self {
        Self {
            leaf_digests: Vec::new(),
            hash_chain: Vec::new(),
            link_macs: Vec::new(),
            merkle_tree: MerkleTree::new(),
            chain_key: random_key(),
        }
    }

    /// Create a chain with a caller-supplied key (tests and key escrow).
    pub fn with_key(chain_key: Digest32) -> Self {
        Self {
            leaf_digests: Vec::new(),
            hash_chain: Vec::new(),
            link_macs: Vec::new(),
            merkle_tree: MerkleTree::new(),
            chain_key,
        }
    }

    /// Number of committed events.
    pub fn len(&self) -> usize {
        self.leaf_digests.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.leaf_digests.is_empty()
    }

    /// The current chain head, if any.
    pub fn head(&self) -> Option<Digest32> {
        self.hash_chain.last().copied()
    }

    /// The current Merkle root, if any.
    pub fn merkle_root(&self) -> Option<Digest32> {
        self.merkle_tree.root()
    }

    /// Commit an event to the chain.
    pub fn add_event(&mut self, event: &AuditEvent) -> Result<()> {
        let leaf = event_leaf_digest(event);
        let previous = self.head().unwrap_or_else(genesis_link);
        let link = hash_link(&previous, &leaf);
        let mac = hmac_sha256(&self.chain_key, &link);

        self.leaf_digests.push(leaf);
        self.hash_chain.push(link);
        self.link_macs.push(mac);
        self.merkle_tree.push_leaf_digest(leaf);
        Ok(())
    }

    /// Re-derive every commitment from `events` and compare.
    ///
    /// Returns the first inconsistency found. This is the load-bearing tamper
    /// check: the leaf digests are recomputed from the caller's events rather
    /// than read back from the chain's own storage.
    pub fn verify_events(&self, events: &[AuditEvent]) -> Result<()> {
        if events.len() != self.leaf_digests.len() {
            return Err(OptimError::InvalidState(format!(
                "audit chain commits to {} events, {} were supplied: events have been added or \
                 removed outside the chain",
                self.leaf_digests.len(),
                events.len()
            )));
        }

        let mut recomputed_leaves = Vec::with_capacity(events.len());
        let mut previous = genesis_link();
        for (index, event) in events.iter().enumerate() {
            let leaf = event_leaf_digest(event);
            if !digests_equal(&leaf, &self.leaf_digests[index]) {
                return Err(OptimError::InvalidState(format!(
                    "audit event {index} (id {:?}) has been modified since it was recorded",
                    event.id
                )));
            }
            let link = hash_link(&previous, &leaf);
            if !digests_equal(&link, &self.hash_chain[index]) {
                return Err(OptimError::InvalidState(format!(
                    "hash-chain link {index} does not match: the chain has been reordered or \
                     spliced"
                )));
            }
            let mac = hmac_sha256(&self.chain_key, &link);
            if !digests_equal(&mac, &self.link_macs[index]) {
                return Err(OptimError::InvalidState(format!(
                    "the authentication tag for link {index} does not verify"
                )));
            }
            previous = link;
            recomputed_leaves.push(leaf);
        }

        self.merkle_tree.verify_against_leaves(&recomputed_leaves)
    }

    /// Produce an inclusion proof for the event at `index`.
    pub fn inclusion_proof(&self, index: usize) -> Result<MerkleProof> {
        self.merkle_tree.inclusion_proof(index)
    }

    /// Verify an inclusion proof for `event` against the current Merkle root.
    pub fn verify_inclusion(&self, event: &AuditEvent, proof: &MerkleProof) -> Result<()> {
        let root = self.merkle_tree.root().ok_or_else(|| {
            OptimError::InvalidState("the audit chain has no Merkle root yet".to_string())
        })?;
        let leaf = event_leaf_digest(event);
        if MerkleTree::verify_inclusion_proof(&root, &leaf, proof) {
            Ok(())
        } else {
            Err(OptimError::InvalidState(
                "the Merkle inclusion proof does not verify against the committed root".to_string(),
            ))
        }
    }

    /// Self-consistency of the chain's own storage.
    ///
    /// Re-derives every link and MAC from the stored leaf digests and the
    /// Merkle root from the stored leaves. This detects tampering with the
    /// chain structures themselves; use [`AuditChain::verify_events`] (or
    /// [`AuditTrail::verify_integrity`]) to detect tampering with the events.
    pub fn verify_integrity(&self) -> bool {
        if self.hash_chain.len() != self.leaf_digests.len()
            || self.link_macs.len() != self.leaf_digests.len()
        {
            return false;
        }
        let mut previous = genesis_link();
        for index in 0..self.leaf_digests.len() {
            let link = hash_link(&previous, &self.leaf_digests[index]);
            if !digests_equal(&link, &self.hash_chain[index]) {
                return false;
            }
            if !digests_equal(&hmac_sha256(&self.chain_key, &link), &self.link_macs[index]) {
                return false;
            }
            previous = link;
        }
        self.merkle_tree.verify_integrity()
    }
}

impl Default for AuditChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Audit trail: append-only event storage with a tamper-evident chain.
pub struct AuditTrail {
    /// Audit events, in insertion order.
    pub(super) events: VecDeque<AuditEvent>,
    /// Actor -> event positions, for fast lookup.
    actor_index: HashMap<String, Vec<usize>>,
    /// Cryptographic chain committing to every event.
    pub(super) chain: AuditChain,
}

impl AuditTrail {
    /// Create an empty trail.
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            actor_index: HashMap::new(),
            chain: AuditChain::new(),
        }
    }

    /// Create an empty trail whose chain uses a caller-supplied key.
    pub fn with_chain_key(chain_key: Digest32) -> Self {
        Self {
            events: VecDeque::new(),
            actor_index: HashMap::new(),
            chain: AuditChain::with_key(chain_key),
        }
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the trail is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Read-only view of the recorded events.
    pub fn events(&self) -> impl Iterator<Item = &AuditEvent> {
        self.events.iter()
    }

    /// Read-only view of the cryptographic chain.
    pub fn chain(&self) -> &AuditChain {
        &self.chain
    }

    /// Append an event, committing it to the chain first.
    pub fn add_event(&mut self, event: AuditEvent) -> Result<()> {
        self.chain.add_event(&event)?;
        let position = self.events.len();
        self.actor_index
            .entry(event.actor.clone())
            .or_default()
            .push(position);
        self.events.push_back(event);
        Ok(())
    }

    /// Re-derive every commitment and compare with the stored events.
    ///
    /// Any modification, reordering, insertion or removal of a stored event is
    /// reported as an error naming the offending position.
    pub fn verify_integrity(&self) -> Result<()> {
        let events: Vec<AuditEvent> = self.events.iter().cloned().collect();
        self.chain.verify_events(&events)
    }

    /// Produce an inclusion proof for the event at `index`.
    pub fn inclusion_proof(&self, index: usize) -> Result<MerkleProof> {
        self.chain.inclusion_proof(index)
    }

    /// Query events by criteria.
    pub fn query_events(&self, criteria: &AuditQueryCriteria) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|event| Self::matches_criteria(event, criteria))
            .collect()
    }

    /// Positions recorded for a given actor.
    pub fn positions_for_actor(&self, actor: &str) -> &[usize] {
        self.actor_index
            .get(actor)
            .map(|positions| positions.as_slice())
            .unwrap_or(&[])
    }

    /// Whether an event matches the query criteria.
    ///
    /// The event-type comparison used to be
    /// `!matches!(&event.event_type, event_type)`, in which `event_type` is an
    /// irrefutable *binding* pattern rather than the value to compare
    /// against -- so it always matched and the filter silently did nothing.
    fn matches_criteria(event: &AuditEvent, criteria: &AuditQueryCriteria) -> bool {
        if let Some(actor) = criteria.actor.as_ref() {
            if event.actor != *actor {
                return false;
            }
        }
        if let Some(wanted) = criteria.event_type.as_ref() {
            if !Self::same_event_type(&event.event_type, wanted) {
                return false;
            }
        }
        if let Some(start_time) = criteria.start_time {
            if event.timestamp < start_time {
                return false;
            }
        }
        if let Some(end_time) = criteria.end_time {
            if event.timestamp > end_time {
                return false;
            }
        }
        if let Some(needle) = criteria.text_search.as_ref() {
            if !event.data.description.contains(needle.as_str()) {
                return false;
            }
        }
        true
    }

    /// Discriminant comparison for event types.
    fn same_event_type(left: &AuditEventType, right: &AuditEventType) -> bool {
        super::hashing::event_type_key(left) == super::hashing::event_type_key(right)
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::hashing::DIGEST_LEN;
    use super::*;
    use crate::privacy::enhanced_audit::types::{AuditEventData, PrivacyContext};

    fn event(id: &str, actor: &str, epsilon: f64) -> AuditEvent {
        AuditEvent {
            id: id.to_string(),
            timestamp: 1_700_000_000 + id.len() as u64,
            event_type: AuditEventType::PrivacyBudgetConsumption,
            actor: actor.to_string(),
            data: AuditEventData {
                description: format!("spent {epsilon} epsilon"),
                affected_data_subjects: Vec::new(),
                data_categories: Vec::new(),
                processing_purposes: vec!["ml_training".to_string()],
                legal_basis: vec!["consent".to_string()],
                technical_measures: vec!["differential_privacy".to_string()],
                metadata: HashMap::new(),
            },
            privacy_context: PrivacyContext {
                epsilon_budget: epsilon,
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

    fn trail_with(count: usize) -> AuditTrail {
        let mut trail = AuditTrail::new();
        for index in 0..count {
            let ok = trail.add_event(event(&format!("e{index}"), "trainer", 0.1 * index as f64));
            assert!(ok.is_ok(), "add_event failed");
        }
        trail
    }

    #[test]
    fn an_untampered_trail_verifies() {
        for count in [0usize, 1, 2, 3, 5, 8, 13] {
            let trail = trail_with(count);
            assert!(
                trail.verify_integrity().is_ok(),
                "a clean trail of {count} events must verify"
            );
        }
    }

    #[test]
    fn flipping_one_byte_of_one_event_fails_verification() {
        // This is the regression that the old `!nodes.is_empty() &&
        // root_hash.is_some()` check could never fail.
        for victim in 0..5usize {
            let mut trail = trail_with(5);
            assert!(trail.verify_integrity().is_ok());

            // Flip a single bit of a single field of a single event.
            let bits = trail.events[victim]
                .privacy_context
                .epsilon_budget
                .to_bits();
            trail.events[victim].privacy_context.epsilon_budget = f64::from_bits(bits ^ 1);

            let verdict = trail.verify_integrity();
            assert!(
                verdict.is_err(),
                "tampering with event {victim} must be detected"
            );
            let message = match verdict {
                Err(err) => err.to_string(),
                Ok(()) => String::new(),
            };
            assert!(
                message.contains(&format!("audit event {victim}")),
                "the error must name the tampered event, got: {message}"
            );
        }
    }

    #[test]
    fn flipping_one_byte_of_a_string_field_fails_verification() {
        let mut trail = trail_with(4);
        trail.events[2].actor.push('!');
        assert!(trail.verify_integrity().is_err());

        let mut trail = trail_with(4);
        trail.events[1].data.description = "spent nothing at all".to_string();
        assert!(trail.verify_integrity().is_err());

        let mut trail = trail_with(4);
        trail.events[0].data.legal_basis.clear();
        assert!(trail.verify_integrity().is_err());
    }

    #[test]
    fn reordering_events_fails_verification() {
        let mut trail = trail_with(4);
        trail.events.swap(1, 2);
        let verdict = trail.verify_integrity();
        assert!(verdict.is_err(), "a reordered trail must not verify");
    }

    #[test]
    fn removing_or_appending_an_event_out_of_band_fails_verification() {
        let mut trail = trail_with(4);
        let _ = trail.events.pop_back();
        assert!(trail.verify_integrity().is_err());

        let mut trail = trail_with(4);
        trail.events.push_back(event("smuggled", "attacker", 9.0));
        assert!(trail.verify_integrity().is_err());
    }

    #[test]
    fn the_merkle_root_changes_when_a_leaf_changes() {
        let clean = trail_with(5);
        let clean_root = clean.chain.merkle_root();
        assert!(clean_root.is_some());

        let mut altered = AuditTrail::new();
        for index in 0..5usize {
            let mut ev = event(&format!("e{index}"), "trainer", 0.1 * index as f64);
            if index == 3 {
                ev.privacy_context.epsilon_budget = 42.0;
            }
            let ok = altered.add_event(ev);
            assert!(ok.is_ok());
        }
        assert_ne!(clean_root, altered.chain.merkle_root());
    }

    #[test]
    fn inclusion_proofs_verify_for_every_leaf_and_fail_for_the_wrong_event() {
        for count in [1usize, 2, 3, 4, 7, 9] {
            let trail = trail_with(count);
            for index in 0..count {
                let proof = match trail.inclusion_proof(index) {
                    Ok(proof) => proof,
                    Err(err) => panic!("proof for {index}/{count} failed: {err}"),
                };
                let target = &trail.events[index];
                assert!(
                    trail.chain.verify_inclusion(target, &proof).is_ok(),
                    "leaf {index} of {count} must prove inclusion"
                );
                let outsider = event("not-in-the-tree", "attacker", 7.0);
                assert!(
                    trail.chain.verify_inclusion(&outsider, &proof).is_err(),
                    "an event outside the tree must not prove inclusion"
                );
            }
        }
    }

    #[test]
    fn an_inclusion_proof_out_of_range_is_an_error() {
        let trail = trail_with(3);
        assert!(trail.inclusion_proof(3).is_err());
        let empty = AuditTrail::new();
        assert!(empty.inclusion_proof(0).is_err());
    }

    #[test]
    fn a_tampered_committed_root_is_detected() {
        let mut tree = MerkleTree::new();
        for index in 0..6u8 {
            let ok = tree.add_leaf(vec![index; 8]);
            assert!(ok.is_ok());
        }
        assert!(tree.verify_integrity());
        // Overwrite the committed root without touching the leaves.
        tree.committed_root = Some([0xabu8; DIGEST_LEN]);
        assert!(
            !tree.verify_integrity(),
            "a root that does not follow from the leaves must not verify"
        );
    }

    #[test]
    fn an_empty_merkle_leaf_payload_is_rejected() {
        let mut tree = MerkleTree::new();
        assert!(tree.add_leaf(Vec::new()).is_err());
    }

    #[test]
    fn merkle_depth_grows_logarithmically() {
        let mut tree = MerkleTree::new();
        assert_eq!(tree.depth(), 0);
        let expected = [1usize, 2, 3, 3, 4, 4, 4, 4, 5];
        for (index, want) in expected.iter().enumerate() {
            let ok = tree.add_leaf(vec![index as u8 + 1; 4]);
            assert!(ok.is_ok());
            assert_eq!(tree.depth(), *want, "after {} leaves", index + 1);
        }
    }

    #[test]
    fn query_by_event_type_actually_filters() {
        // Regression: the event-type filter was an irrefutable binding
        // pattern, so every event matched every requested type.
        let mut trail = AuditTrail::new();
        let mut access = event("a", "reader", 0.0);
        access.event_type = AuditEventType::DataAccess;
        let mut consumption = event("b", "trainer", 0.5);
        consumption.event_type = AuditEventType::PrivacyBudgetConsumption;
        let ok = trail.add_event(access);
        assert!(ok.is_ok());
        let ok = trail.add_event(consumption);
        assert!(ok.is_ok());

        let hits = trail.query_events(&AuditQueryCriteria {
            actor: None,
            event_type: Some(AuditEventType::DataAccess),
            start_time: None,
            end_time: None,
            text_search: None,
        });
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "a");

        let hits = trail.query_events(&AuditQueryCriteria {
            actor: None,
            event_type: Some(AuditEventType::SecurityIncident),
            start_time: None,
            end_time: None,
            text_search: None,
        });
        assert!(hits.is_empty(), "no event has that type");
    }

    #[test]
    fn query_by_text_search_actually_filters() {
        let trail = trail_with(3);
        let hits = trail.query_events(&AuditQueryCriteria {
            actor: None,
            event_type: None,
            start_time: None,
            end_time: None,
            text_search: Some("spent 0.1".to_string()),
        });
        assert_eq!(hits.len(), 1);

        let hits = trail.query_events(&AuditQueryCriteria {
            actor: None,
            event_type: None,
            start_time: None,
            end_time: None,
            text_search: Some("no such text".to_string()),
        });
        assert!(hits.is_empty());
    }

    #[test]
    fn two_chains_use_independent_keys() {
        // A constant chain key would let anyone recompute the tags after
        // rewriting the chain.
        let left = AuditChain::new();
        let right = AuditChain::new();
        assert_ne!(left.chain_key, right.chain_key);
    }

    #[test]
    fn a_forged_mac_is_detected() {
        let mut trail = AuditTrail::with_chain_key([9u8; DIGEST_LEN]);
        for index in 0..3usize {
            let ok = trail.add_event(event(&format!("e{index}"), "trainer", 0.2));
            assert!(ok.is_ok());
        }
        assert!(trail.chain.verify_integrity());
        trail.chain.link_macs[1] = [0u8; DIGEST_LEN];
        assert!(!trail.chain.verify_integrity());
        assert!(trail.verify_integrity().is_err());
    }
}
