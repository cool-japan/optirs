//! Trait implementations that have no natural home next to their type.
//!
//! This file replaces the 19 auto-generated `*_traits.rs` shells listed in the
//! module documentation. Seventeen of them held a single
//! `Default::default() -> Self::new()` forwarding impl, which now sits next to
//! the constructor it forwards to. The two that did not -- `Debug for
//! ComplianceRule` and `Default for DashboardConfig` -- live here, together
//! with `Debug` impls for the remaining closure-carrying types so that a
//! configured audit system can be printed.

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{
    Axiom, ComplianceRule, CryptographicKeys, CryptographicProofType, DashboardConfig,
    DashboardLayout, ExternalComplianceAPI, FormalVerificationRule, ProofAlgorithm, ProofStrategy,
    TransitionFunction,
};

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            refresh_interval: 30,
            history_retention_hours: 24,
            alert_thresholds: std::collections::HashMap::new(),
            layout: DashboardLayout {
                columns: 3,
                widgets: Vec::new(),
            },
        }
    }
}

impl Debug for ComplianceRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplianceRule")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("evaluation_fn", &"<function>")
            .field("severity", &self.severity)
            .field("frameworks", &self.frameworks)
            .finish()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for FormalVerificationRule<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FormalVerificationRule")
            .field("name", &self.name)
            .field("specification", &self.specification)
            .field("verify_fn", &"<function>")
            .field("criticality", &self.criticality)
            .finish()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for Axiom<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Axiom")
            .field("name", &self.name)
            .field("statement", &self.statement)
            .field("verify_fn", &"<function>")
            .finish()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for ProofStrategy<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProofStrategy")
            .field("name", &self.name)
            .field("apply_fn", &"<function>")
            .finish()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for ProofAlgorithm<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProofAlgorithm")
            .field("name", &self.name)
            .field("generate_fn", &"<function>")
            .field("verify_fn", &"<function>")
            .finish()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for CryptographicProofType<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CryptographicProofType")
            .field("name", &self.name)
            .field("generate_fn", &"<function>")
            .field("verify_fn", &"<function>")
            .finish()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for TransitionFunction<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransitionFunction")
            .field("name", &self.name)
            .field("logic", &"<function>")
            .finish()
    }
}

impl Debug for ExternalComplianceAPI {
    /// The API key is redacted: audit records must not leak credentials.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalComplianceAPI")
            .field("name", &self.name)
            .field("endpoint", &self.endpoint)
            .field(
                "api_key",
                &self
                    .api_key
                    .as_ref()
                    .map(|_| "<redacted>")
                    .unwrap_or("None"),
            )
            .field("frameworks", &self.frameworks)
            .finish()
    }
}

impl Debug for CryptographicKeys {
    /// Key *names* are printed; key material is not.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut signing: Vec<&String> = self.signing_keys.keys().collect();
        signing.sort();
        let mut verification: Vec<&String> = self.verification_keys.keys().collect();
        verification.sort();
        let mut encryption: Vec<&String> = self.encryption_keys.keys().collect();
        encryption.sort();
        f.debug_struct("CryptographicKeys")
            .field("signing_keys", &signing)
            .field("verification_keys", &verification)
            .field("encryption_keys", &encryption)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::enhanced_audit::compliance::gdpr_rules;
    use crate::privacy::enhanced_audit::proofs::HMAC_SHA256_INTEGRITY;

    #[test]
    fn the_default_dashboard_config_is_unchanged() {
        let config = DashboardConfig::default();
        assert_eq!(config.refresh_interval, 30);
        assert_eq!(config.history_retention_hours, 24);
        assert_eq!(config.layout.columns, 3);
        assert!(config.alert_thresholds.is_empty());
    }

    #[test]
    fn compliance_rules_are_debug_printable_without_their_closure() {
        let rules = gdpr_rules();
        let rendered = format!("{:?}", rules[0]);
        assert!(rendered.contains("<function>"));
        assert!(rendered.contains("gdpr-art6-legal-basis"));
    }

    #[test]
    fn key_material_is_never_printed() {
        let keys = CryptographicKeys::generate();
        let rendered = format!("{keys:?}");
        assert!(rendered.contains(HMAC_SHA256_INTEGRITY));
        let secret = match keys.mac_key(HMAC_SHA256_INTEGRITY) {
            Some(key) => key,
            None => panic!("a MAC key must be generated"),
        };
        let hex: String = secret.iter().map(|byte| format!("{byte:02x}")).collect();
        assert!(
            !rendered.contains(&hex),
            "the Debug output must not contain the key bytes"
        );
    }

    #[test]
    fn an_api_key_is_redacted() {
        let api = ExternalComplianceAPI {
            name: "auditor".to_string(),
            endpoint: "https://example.invalid/audit".to_string(),
            api_key: Some("super-secret-token".to_string()),
            frameworks: Vec::new(),
        };
        let rendered = format!("{api:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("super-secret-token"));
    }
}
