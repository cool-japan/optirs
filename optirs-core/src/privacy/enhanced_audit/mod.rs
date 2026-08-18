//! Enhanced audit system for privacy-preserving optimization.
//!
//! # Module layout
//!
//! | module | responsibility |
//! |---|---|
//! | [`hashing`] | deterministic canonical encodings, SHA-256, HMAC-SHA256 |
//! | [`integrity`] | Merkle tree with inclusion proofs, hash chain, audit trail |
//! | [`verification`] | formal verification rules and the theorem prover |
//! | [`model_checking`] | bounded invariant model checking |
//! | [`proofs`] | cryptographic integrity proofs |
//! | [`compliance`] | GDPR / HIPAA / CCPA rule sets and reporting |
//! | [`budget`] | privacy budget allocation, accounting and forecasting |
//! | [`dashboard`] | metrics and threshold alerts |
//! | [`types`] | data types and the [`EnhancedAuditSystem`] entry point |
//!
//! # 0.3.2 notes
//!
//! The 19 auto-generated `*_traits.rs` shells, each containing a single
//! `Default` forwarding impl, were removed. Every trait implementation they
//! carried now lives next to the type it belongs to, or in [`trait_impls`] for
//! the two that have no natural home. No trait implementation was dropped, so
//! the change is source-compatible; the module *paths*
//! `enhanced_audit::<type>_traits` no longer exist (they contained no items to
//! import).
//!
//! Behavioural changes in the same release:
//!
//! * [`EnhancedAuditSystem::new`] returns `Result`: a configuration requesting
//!   at-rest encryption, external audit submission, or a zero-knowledge /
//!   non-repudiation / confidentiality proof is now refused instead of being
//!   accepted and ignored.
//! * [`EnhancedAuditSystem::log_event`] returns the number of compliance
//!   violations the event triggered.
//! * `MerkleTree::add_leaf` takes a leaf *payload* and applies the
//!   domain-separated leaf hash; pass a precomputed digest to
//!   [`integrity::MerkleTree::push_leaf_digest`].
//! * `AuditChain::verify_integrity` still returns `bool` (self-consistency of
//!   the chain structures); the tamper check over the stored events is
//!   [`integrity::AuditTrail::verify_integrity`], which returns `Result`.

pub mod budget;
pub mod compliance;
pub mod dashboard;
pub mod functions;
pub mod hashing;
pub mod integrity;
pub mod model_checking;
pub mod proofs;
pub mod trait_impls;
pub mod types;
pub mod verification;

// Re-export all types so `privacy::enhanced_audit::<Type>` keeps working.
pub use budget::{BudgetForecastingModel, PredictionModel, PrivacyBudgetTracker};
pub use compliance::{
    ccpa_rules, gdpr_rules, hipaa_rules, rules_for, supported_frameworks, ComplianceMonitor,
    RegulationChecker, RegulatoryComplianceChecker,
};
pub use dashboard::MonitoringDashboard;
pub use functions::*;
pub use hashing::{
    canonical_array_bytes, canonical_event_bytes, event_leaf_digest, framework_key, hmac_sha256,
    sha256, Digest32, DIGEST_LEN,
};
pub use integrity::{AuditChain, AuditTrail, MerkleProof, MerkleProofStep, MerkleTree};
pub use model_checking::{
    CompareOp, ContextFlag, ModelCheckOutcome, ModelChecker, StatePredicate, SystemModel, Term,
};
pub use proofs::{
    CryptographicProofGenerator, ProofSystem, HMAC_SHA256_INTEGRITY, SHA256_INTEGRITY,
};
pub use types::*;
pub use verification::{
    all_axioms_strategy, default_axioms, default_privacy_rules, FormalVerificationEngine,
    TheoremProver,
};
