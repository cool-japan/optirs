//! Function-pointer type aliases used by the audit system's extension points.

use crate::error::Result;
use scirs2_core::ndarray::Array1;

use super::types::{
    AuditEvent, Axiom, ComplianceAssessment, CryptographicKeys, CryptographicProof, PrivacyContext,
    ProofResult, SystemState, VerificationResult,
};

/// Type alias for verification function
pub type VerifyFn<T> = Box<dyn Fn(&Array1<T>, &PrivacyContext) -> VerificationResult + Send + Sync>;
/// Type alias for proof generation function
pub type ProofGenerateFn<T> = Box<dyn Fn(&Array1<T>) -> Result<Vec<u8>> + Send + Sync>;
/// Type alias for proof verification function
pub type ProofVerifyFn<T> = Box<dyn Fn(&[u8], &Array1<T>) -> bool + Send + Sync>;
/// Type alias for transition logic function
pub type TransitionLogicFn<T> = Box<dyn Fn(&SystemState<T>) -> Vec<SystemState<T>> + Send + Sync>;
/// Type alias for axiom verification function
pub type AxiomVerifyFn<T> = Box<dyn Fn(&Array1<T>) -> bool + Send + Sync>;
/// Type alias for proof strategy application function
pub type ProofStrategyApplyFn<T> =
    Box<dyn Fn(&Array1<T>, &[Axiom<T>]) -> ProofResult + Send + Sync>;
/// Type alias for cryptographic proof generation function
pub type CryptoProofGenerateFn<T> =
    Box<dyn Fn(&Array1<T>, &CryptographicKeys) -> Result<CryptographicProof> + Send + Sync>;
/// Type alias for cryptographic proof verification function
pub type CryptoProofVerifyFn<T> =
    Box<dyn Fn(&CryptographicProof, &Array1<T>, &CryptographicKeys) -> bool + Send + Sync>;
/// Type alias for compliance assessment function
pub type ComplianceAssessmentFn = Box<dyn Fn(&AuditEvent) -> ComplianceAssessment + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::super::integrity::AuditTrail;
    use super::super::proofs::unix_timestamp;
    use super::super::types::{
        AuditConfig, AuditEvent, AuditEventData, AuditEventType, ComplianceFramework,
        PrivacyContext, ProofRequirements,
    };
    use std::collections::HashMap;

    fn now() -> u64 {
        unix_timestamp().unwrap_or_default()
    }

    #[test]
    fn test_audit_config() {
        let config = AuditConfig {
            comprehensive_logging: true,
            real_time_monitoring: true,
            formal_verification: true,
            retention_period_days: 365,
            compliance_frameworks: vec![ComplianceFramework::GDPR, ComplianceFramework::HIPAA],
            proof_requirements: ProofRequirements {
                zero_knowledge_proofs: false,
                non_repudiation: false,
                integrity_proofs: true,
                confidentiality_proofs: false,
                completeness_proofs: true,
            },
            encrypt_audit_trail: false,
            external_audit_integration: false,
        };
        assert!(config.comprehensive_logging);
        assert_eq!(config.compliance_frameworks.len(), 2);
        assert!(config.proof_requirements.integrity_proofs);
    }

    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent {
            id: "test_event_1".to_string(),
            timestamp: now(),
            event_type: AuditEventType::PrivacyBudgetConsumption,
            actor: "test_user".to_string(),
            data: AuditEventData {
                description: "Test privacy budget consumption".to_string(),
                affected_data_subjects: vec!["subject1".to_string()],
                data_categories: vec!["personal_data".to_string()],
                processing_purposes: vec!["ml_training".to_string()],
                legal_basis: vec!["consent".to_string()],
                technical_measures: vec!["differential_privacy".to_string()],
                metadata: HashMap::new(),
            },
            privacy_context: PrivacyContext {
                epsilon_budget: 1.0,
                delta_budget: 1e-5,
                privacy_mechanism: "dp_sgd".to_string(),
                data_minimization: true,
                purpose_limitation: true,
                storage_limitation: true,
            },
            signature: None,
            compliance_annotations: HashMap::new(),
        };
        assert_eq!(event.actor, "test_user");
        assert!(matches!(
            event.event_type,
            AuditEventType::PrivacyBudgetConsumption
        ));
        assert_eq!(event.privacy_context.epsilon_budget, 1.0);
    }

    #[test]
    fn test_audit_trail() {
        let mut trail = AuditTrail::new();
        let event = AuditEvent {
            id: "test_event".to_string(),
            timestamp: now(),
            event_type: AuditEventType::DataAccess,
            actor: "test_actor".to_string(),
            data: AuditEventData {
                description: "Test data access".to_string(),
                affected_data_subjects: vec![],
                data_categories: vec![],
                processing_purposes: vec![],
                legal_basis: vec![],
                technical_measures: vec![],
                metadata: HashMap::new(),
            },
            privacy_context: PrivacyContext {
                epsilon_budget: 0.5,
                delta_budget: 1e-6,
                privacy_mechanism: "test".to_string(),
                data_minimization: true,
                purpose_limitation: true,
                storage_limitation: true,
            },
            signature: None,
            compliance_annotations: HashMap::new(),
        };
        let outcome = trail.add_event(event);
        assert!(outcome.is_ok(), "add_event failed: {outcome:?}");
        assert_eq!(trail.len(), 1);
        // The structural self-check and the event re-derivation must both pass.
        assert!(trail.chain().verify_integrity());
        assert!(trail.verify_integrity().is_ok());
    }
}
