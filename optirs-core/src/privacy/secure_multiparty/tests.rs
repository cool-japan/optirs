//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::error::OptimError;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

use super::helpers::{add_mod, field_to_value, inv_mod, mul_mod, neg_mod, sub_mod, value_to_field};
use super::helpers::{
    COMMITMENT_NONCE_LEN, DIGEST_LEN, FIXED_POINT_BITS, FIXED_POINT_SCALE, SHAMIR_PRIME,
};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn test_config(num_participants: usize, threshold: usize) -> SMPCConfig {
        SMPCConfig {
            num_participants,
            threshold,
            security_parameter: 128,
            enable_homomorphic: false,
            enable_zk_proofs: false,
            protocol_variant: SMPCProtocol::FederatedSMPC,
            communication_security: CommunicationSecurity::SemiHonest,
            malicious_tolerance: MaliciousTolerance {
                max_corrupted: 1,
                byzantine_tolerance: true,
                verification_threshold: 0.8,
                commit_and_prove: true,
            },
        }
    }

    fn coordinator_with_participants(
        num_participants: usize,
        threshold: usize,
        ids: &[&str],
    ) -> SMPCCoordinator<f64> {
        let mut coordinator = SMPCCoordinator::<f64>::new(test_config(num_participants, threshold))
            .expect("coordinator construction failed");
        for id in ids {
            coordinator
                .add_participant(Participant::new(*id, vec![0u8; 4], 1.0))
                .expect("participant registration failed");
        }
        coordinator
    }

    // ---------------------------------------------------------------- field

    #[test]
    fn test_fixed_point_scale_constant() {
        assert_eq!(FIXED_POINT_SCALE, 2f64.powi(FIXED_POINT_BITS as i32));
        assert_eq!(SHAMIR_PRIME, (1u128 << 127) - 1);
    }

    #[test]
    fn test_field_arithmetic() {
        let a = SHAMIR_PRIME - 5;
        let b = 12u128;

        assert_eq!(add_mod(a, b), 7);
        assert_eq!(sub_mod(b, a), 17);
        assert_eq!(add_mod(a, neg_mod(a)), 0);

        // Large operands exercise the double-and-add path.
        let big = SHAMIR_PRIME - 1; // == -1 mod p
        assert_eq!(mul_mod(big, big), 1);
        assert_eq!(mul_mod(big, 1), big);
        assert_eq!(mul_mod(0, big), 0);

        let inverse = inv_mod(big).expect("inverse of -1 exists");
        assert_eq!(mul_mod(big, inverse), 1);

        let inverse_small = inv_mod(3).expect("inverse of 3 exists");
        assert_eq!(mul_mod(3, inverse_small), 1);

        assert!(inv_mod(0).is_err());
    }

    #[test]
    fn test_quantisation_round_trip() {
        for value in [0.0f64, 1.0, -1.0, 42.0, -3.75, 0.1, 1234.5678, -1e6] {
            let element = value_to_field(value).expect("quantisation failed");
            let restored: f64 = field_to_value(element).expect("de-quantisation failed");
            assert!(
                (restored - value).abs() <= 1e-12 + value.abs() * 1e-15,
                "round trip failed for {value}: got {restored}"
            );
        }
    }

    #[test]
    fn test_quantisation_rejects_out_of_range_and_non_finite() {
        assert!(value_to_field(f64::NAN).is_err());
        assert!(value_to_field(f64::INFINITY).is_err());
        assert!(value_to_field(max_representable_magnitude() * 2.0).is_err());
    }

    // ------------------------------------------------------- secret sharing

    #[test]
    fn test_secret_sharing_round_trip_any_subset() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let secret = 42.0;

        let shares = secret_sharing.share_secret(secret).expect("sharing failed");
        assert_eq!(shares.len(), 5);

        // Prefix subset.
        let reconstructed = secret_sharing
            .reconstruct_secret(&shares[0..3])
            .expect("reconstruction failed");
        assert!((reconstructed - secret).abs() < 1e-10);

        // Non-prefix subset: only correct Lagrange interpolation over the field
        // recovers the secret here.
        let reconstructed_tail = secret_sharing
            .reconstruct_secret(&shares[2..5])
            .expect("reconstruction failed");
        assert!((reconstructed_tail - secret).abs() < 1e-10);

        // Arbitrary out-of-order subset.
        let mixed = vec![shares[4], shares[0], shares[3]];
        let reconstructed_mixed = secret_sharing
            .reconstruct_secret(&mixed)
            .expect("reconstruction failed");
        assert!((reconstructed_mixed - secret).abs() < 1e-10);
    }

    #[test]
    fn test_secret_sharing_handles_negative_and_fractional_secrets() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(2, 4).expect("valid sharing parameters");

        for secret in [-7.25f64, 0.0, 1e-6, 1024.5] {
            let shares = secret_sharing.share_secret(secret).expect("sharing failed");
            let reconstructed = secret_sharing
                .reconstruct_secret(&shares[1..3])
                .expect("reconstruction failed");
            assert!(
                (reconstructed - secret).abs() < 1e-10,
                "failed for {secret}: got {reconstructed}"
            );
        }
    }

    #[test]
    fn test_secret_sharing_requires_threshold_shares() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let shares = secret_sharing.share_secret(1.5).expect("sharing failed");
        assert!(secret_sharing.reconstruct_secret(&shares[0..2]).is_err());
    }

    #[test]
    fn test_secret_sharing_rejects_invalid_parameters() {
        assert!(ShamirSecretSharing::<f64>::new(3, 2).is_err());
        assert!(ShamirSecretSharing::<f64>::new(0, 5).is_err());
        assert!(ShamirSecretSharing::<f64>::new(1, 0).is_err());
    }

    #[test]
    fn test_secret_sharing_rejects_duplicate_x_coordinates() {
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(2, 3).expect("valid sharing parameters");
        let shares = secret_sharing.share_secret(3.0).expect("sharing failed");
        let duplicated = vec![shares[0], shares[0]];
        assert!(secret_sharing.reconstruct_secret(&duplicated).is_err());
    }

    #[test]
    fn test_secret_sharing_coefficients_are_fresh_per_call() {
        // Previously every call re-created Random::seed(42), making the coefficients
        // compile-time constants; sharing the same secret twice produced identical
        // shares and one share revealed the secret.
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let first = secret_sharing.share_secret(42.0).expect("sharing failed");
        let second = secret_sharing.share_secret(42.0).expect("sharing failed");

        assert!(
            first.iter().zip(second.iter()).any(|(a, b)| a.y != b.y),
            "shares of the same secret must not repeat across calls"
        );

        // Two independently constructed instances must differ as well.
        let mut other = ShamirSecretSharing::<f64>::new(3, 5).expect("valid sharing parameters");
        let third = other.share_secret(42.0).expect("sharing failed");
        assert!(first.iter().zip(third.iter()).any(|(a, b)| a.y != b.y));
    }

    #[test]
    fn test_shares_do_not_reveal_the_secret_magnitude() {
        // Coefficients are uniform over the whole field, so an individual share is a
        // uniform field element rather than `secret + small noise`.
        let mut secret_sharing =
            ShamirSecretSharing::<f64>::new(2, 3).expect("valid sharing parameters");
        let shares = secret_sharing.share_secret(42.0).expect("sharing failed");
        let secret_element = value_to_field(42.0).expect("quantisation failed");

        assert!(shares
            .iter()
            .all(|share| share.y.abs_diff(secret_element) > (1u128 << 100)));
    }

    #[test]
    fn test_secret_sharing_with_seed_is_deterministic() {
        let mut a = ShamirSecretSharing::<f64>::with_seed(3, 5, 7).expect("valid parameters");
        let mut b = ShamirSecretSharing::<f64>::with_seed(3, 5, 7).expect("valid parameters");
        assert_eq!(
            a.share_secret(2.5).expect("sharing failed"),
            b.share_secret(2.5).expect("sharing failed")
        );
    }

    // ----------------------------------------------------------- commitment

    #[test]
    fn test_commitment_scheme_is_hiding_and_binding() {
        let mut scheme = CommitmentScheme::<f64>::new();
        let data = Array1::from(vec![1.0, 2.0, 3.0]);

        let (commitment1, nonce1) = scheme.commit(&data).expect("commit failed");
        let (commitment2, nonce2) = scheme.commit(&data).expect("commit failed");

        // Hiding: two commitments to the same value must differ (the previous
        // implementation asserted the opposite).
        assert_ne!(commitment1, commitment2);
        assert_ne!(nonce1, nonce2);

        // Each commitment opens with its own nonce.
        assert!(scheme
            .open(&commitment1, &data, &nonce1)
            .expect("open failed"));
        assert!(scheme
            .open(&commitment2, &data, &nonce2)
            .expect("open failed"));

        // Binding: neither a different value nor a different nonce opens it.
        let different = Array1::from(vec![1.0, 2.0, 4.0]);
        assert!(!scheme
            .open(&commitment1, &different, &nonce1)
            .expect("open failed"));
        assert!(!scheme
            .open(&commitment1, &data, &nonce2)
            .expect("open failed"));
    }

    #[test]
    fn test_commitment_nonce_debug_is_redacted() {
        let nonce = CommitmentNonce::from_bytes([7u8; COMMITMENT_NONCE_LEN]);
        assert_eq!(format!("{nonce:?}"), "CommitmentNonce(<redacted>)");
        assert_eq!(nonce.as_bytes()[0], 7);
    }

    #[test]
    fn test_key_material_is_not_a_compile_time_constant() {
        // Every key used to live behind `Random::seed(42)`, so two instances in two
        // processes shared identical key material.
        let data = Array1::from(vec![1.0, 2.0, 3.0]);

        let first_engine = HomomorphicEngine::<f64>::new();
        let second_engine = HomomorphicEngine::<f64>::new();
        assert_ne!(
            first_engine.encrypt(&data).expect("digesting failed").data,
            second_engine.encrypt(&data).expect("digesting failed").data
        );

        let first_params = VerificationParameters::<f64>::new();
        let second_params = VerificationParameters::<f64>::new();
        assert_ne!(
            first_params
                .generate_verification_data(&data)
                .expect("tag generation failed"),
            second_params
                .generate_verification_data(&data)
                .expect("tag generation failed")
        );

        let mut first_scheme = CommitmentScheme::<f64>::with_seed(1);
        let mut second_scheme = CommitmentScheme::<f64>::with_seed(2);
        assert_ne!(
            first_scheme.commit(&data).expect("commit failed").0,
            second_scheme.commit(&data).expect("commit failed").0
        );
    }

    #[test]
    fn test_verification_tag_round_trip() {
        let params = VerificationParameters::<f64>::new();
        let aggregate = Array1::from(vec![1.0, 2.0]);
        let tag = params
            .generate_verification_data(&aggregate)
            .expect("tag generation failed");

        assert!(params
            .verify_verification_data(&aggregate, &tag)
            .expect("tag verification failed"));
        assert!(!params
            .verify_verification_data(&Array1::from(vec![1.0, 2.5]), &tag)
            .expect("tag verification failed"));
    }

    // --------------------------------------------------------- participants

    #[test]
    fn test_verify_participant_honesty_requires_a_valid_opening() {
        let config = test_config(3, 2);
        let aggregator = CryptographicAggregator::<f64>::new(config);
        let mut scheme = CommitmentScheme::<f64>::new();
        let input = Array1::from(vec![1.0, 2.0]);
        let (commitment, nonce) = scheme.commit(&input).expect("commit failed");

        // No commitment at all -> rejected (self-attestation is not verification).
        let bare = Participant::new("p0", vec![], 1.0);
        assert!(!aggregator
            .verify_participant_honesty(&bare, &input)
            .expect("verification failed"));

        // Commitment without opening nonce -> rejected.
        let mut no_nonce = Participant::new("p1", vec![], 1.0);
        no_nonce.commitment = Some(commitment.clone());
        assert!(!aggregator
            .verify_participant_honesty(&no_nonce, &input)
            .expect("verification failed"));

        // Correct opening -> accepted.
        let honest =
            Participant::new("p2", vec![], 1.0).with_commitment(commitment.clone(), nonce.clone());
        assert!(aggregator
            .verify_participant_honesty(&honest, &input)
            .expect("verification failed"));

        // Same commitment, different submitted input -> rejected.
        let tampered = Array1::from(vec![1.0, 9.0]);
        assert!(!aggregator
            .verify_participant_honesty(&honest, &tampered)
            .expect("verification failed"));

        // Flagged participants are rejected regardless of the opening.
        let mut suspicious = honest.clone();
        suspicious.status = ParticipantStatus::Suspicious;
        assert!(!aggregator
            .verify_participant_honesty(&suspicious, &input)
            .expect("verification failed"));

        // Trust score below the configured threshold -> rejected.
        let mut distrusted = honest;
        distrusted.trust_score = 0.1;
        assert!(!aggregator
            .verify_participant_honesty(&distrusted, &input)
            .expect("verification failed"));
    }

    #[test]
    fn test_secure_aggregate_opens_commitments() {
        let config = test_config(3, 2);
        let mut aggregator = CryptographicAggregator::<f64>::new(config);
        let mut scheme = CommitmentScheme::<f64>::new();

        let mut inputs = HashMap::new();
        let mut participants = HashMap::new();
        for (index, id) in ["a", "b", "c"].iter().enumerate() {
            let input = Array1::from(vec![index as f64, 1.0]);
            let (commitment, nonce) = scheme.commit(&input).expect("commit failed");
            participants.insert(
                (*id).to_string(),
                Participant::new(*id, vec![], 1.0).with_commitment(commitment, nonce),
            );
            inputs.insert((*id).to_string(), input);
        }

        let result = aggregator
            .secure_aggregate(&inputs, &participants)
            .expect("aggregation failed");
        assert_eq!(result.honest_participants.len(), 3);
        assert_eq!(result.aggregate.len(), 2);
        assert!((result.aggregate[0] - 1.0).abs() < 1e-12);
        assert!((result.aggregate[1] - 1.0).abs() < 1e-12);
        assert_eq!(aggregator.aggregation_proofs().len(), 1);
        assert!(aggregator
            .verification_params()
            .verify_verification_data(&result.aggregate, &result.proof.verification_data)
            .expect("tag verification failed"));

        // Dropping the opening of one participant drops them from the aggregate.
        if let Some(participant) = participants.get_mut("c") {
            participant.commitment = None;
            participant.commitment_nonce = None;
        }
        let filtered = aggregator
            .secure_aggregate(&inputs, &participants)
            .expect("aggregation failed");
        assert_eq!(filtered.honest_participants, vec!["a", "b"]);
    }

    #[test]
    fn test_secure_aggregate_rejects_malicious_security_models() {
        let mut config = test_config(3, 2);
        config.communication_security = CommunicationSecurity::MaliciousAbort;
        let mut aggregator = CryptographicAggregator::<f64>::new(config);
        assert!(aggregator
            .secure_aggregate(&HashMap::new(), &HashMap::new())
            .is_err());
    }

    #[test]
    fn test_secure_aggregate_rejects_dimension_mismatch() {
        let config = test_config(2, 1);
        let mut aggregator = CryptographicAggregator::<f64>::new(config);
        let mut scheme = CommitmentScheme::<f64>::new();

        let mut inputs = HashMap::new();
        let mut participants = HashMap::new();
        for (id, values) in [("a", vec![1.0, 2.0]), ("b", vec![1.0])] {
            let input = Array1::from(values);
            let (commitment, nonce) = scheme.commit(&input).expect("commit failed");
            participants.insert(
                id.to_string(),
                Participant::new(id, vec![], 1.0).with_commitment(commitment, nonce),
            );
            inputs.insert(id.to_string(), input);
        }

        assert!(aggregator.secure_aggregate(&inputs, &participants).is_err());
    }

    // ---------------------------------------------------- "homomorphic" API

    #[test]
    fn test_homomorphic_engine_is_not_encryption() {
        let engine = HomomorphicEngine::<f64>::new();
        let data1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let data2 = Array1::from(vec![4.0, 5.0, 6.0]);

        let digests1 = engine.encrypt(&data1).expect("digesting failed");
        let digests2 = engine.encrypt(&data2).expect("digesting failed");
        assert_eq!(digests1.len(), 3);
        assert!(digests1.data.iter().all(|block| block.len() == DIGEST_LEN));
        assert_ne!(digests1.data[0], digests2.data[0]);

        // Decryption and homomorphic addition are unimplemented and say so instead of
        // returning garbage (the old implementation reinterpreted hash bytes as f64).
        let decrypt_error = engine
            .decrypt(&digests1)
            .expect_err("decryption must not succeed");
        assert!(matches!(decrypt_error, OptimError::UnsupportedOperation(_)));
        assert!(format!("{decrypt_error}").contains("not homomorphic encryption"));

        let add_error = engine
            .add_encrypted(&digests1, &digests2)
            .expect_err("homomorphic addition must not succeed");
        assert!(matches!(add_error, OptimError::UnsupportedOperation(_)));
    }

    #[test]
    fn test_homomorphic_ciphertext_rejects_short_blocks_without_panicking() {
        let engine = HomomorphicEngine::<f64>::new();
        let malformed = HomomorphicCiphertext::<f64> {
            data: vec![vec![0u8; 4]],
            params: HomomorphicParameters::new(),
        };

        assert!(malformed.validate().is_err());
        let error = engine
            .decrypt(&malformed)
            .expect_err("short block must be rejected");
        assert!(format!("{error}").contains("digest block 0"));
    }

    #[test]
    fn test_homomorphic_engine_with_seed_is_deterministic() {
        let a = HomomorphicEngine::<f64>::with_seed(11);
        let b = HomomorphicEngine::<f64>::with_seed(11);
        let data = Array1::from(vec![1.0]);
        assert_eq!(
            a.encrypt(&data).expect("digesting failed").data,
            b.encrypt(&data).expect("digesting failed").data
        );
        assert_eq!(a.params().security_level, 128);
    }

    // ---------------------------------------------------- computation digest

    #[test]
    fn test_computation_digest_verifies_and_zk_entry_points_fail() {
        let system = ComputationDigestSystem::<f64>::new();
        let input = Array1::from(vec![1.0, 2.0]);
        let output = Array1::from(vec![3.0]);

        let digest = system
            .digest_computation(&input, &output, "sum")
            .expect("digest failed");
        assert!(system
            .verify_digest(&digest, &input, &output, "sum")
            .expect("verification failed"));

        // Any change to statement, input or output is detected.
        assert!(!system
            .verify_digest(&digest, &input, &Array1::from(vec![4.0]), "sum")
            .expect("verification failed"));
        assert!(!system
            .verify_digest(&digest, &Array1::from(vec![1.0, 2.5]), &output, "sum")
            .expect("verification failed"));

        // The digest never carries a witness derived from the plaintext inputs.
        assert_eq!(digest.digest().len(), DIGEST_LEN);

        // The zero-knowledge entry points refuse instead of accepting forgeries: the
        // old verify_proof returned true for any non-empty proof under a public CRS.
        assert!(system.prove_computation(&input, &output, "sum").is_err());
        assert!(system.verify_proof(&digest).is_err());
    }

    // ------------------------------------------------------------ coordinator

    #[test]
    fn test_smpc_config() {
        let config = test_config(5, 3);
        assert_eq!(config.num_participants, 5);
        assert_eq!(config.threshold, 3);
        assert!(!config.enable_homomorphic);
    }

    #[test]
    fn test_coordinator_rejects_unimplemented_features() {
        let mut homomorphic = test_config(3, 2);
        homomorphic.enable_homomorphic = true;
        assert!(SMPCCoordinator::<f64>::new(homomorphic).is_err());

        let mut zk = test_config(3, 2);
        zk.enable_zk_proofs = true;
        assert!(SMPCCoordinator::<f64>::new(zk).is_err());

        let mut malicious = test_config(3, 2);
        malicious.communication_security = CommunicationSecurity::MaliciousGuaranteed;
        assert!(SMPCCoordinator::<f64>::new(malicious).is_err());

        let mut bgw = test_config(3, 2);
        bgw.protocol_variant = SMPCProtocol::BGW;
        assert!(SMPCCoordinator::<f64>::new(bgw).is_err());
    }

    #[test]
    fn test_coordinator_rejects_threshold_above_participants() {
        assert!(SMPCCoordinator::<f64>::new(test_config(5, 6)).is_err());
        assert!(SMPCCoordinator::<f64>::new(test_config(5, 0)).is_err());
        assert!(SMPCCoordinator::<f64>::new(test_config(0, 0)).is_err());
    }

    #[test]
    fn test_add_participant_limits_and_duplicates() {
        let mut coordinator = coordinator_with_participants(2, 1, &["a", "b"]);
        assert!(coordinator
            .add_participant(Participant::new("a", vec![], 1.0))
            .is_err());
        assert!(coordinator
            .add_participant(Participant::new("c", vec![], 1.0))
            .is_err());
        assert!(coordinator
            .add_participant(Participant::new("", vec![], 1.0))
            .is_err());
        assert_eq!(coordinator.participants().len(), 2);
    }

    #[test]
    fn test_execute_smpc_preserves_input_dimension() {
        // Previously every call flattened all coordinates into one share vector and
        // returned a length-1 array whatever the input dimension was.
        let mut coordinator = coordinator_with_participants(3, 2, &["a", "b", "c"]);

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
        inputs.insert("b".to_string(), Array1::from(vec![0.5, 0.5, 0.5, 0.5]));
        inputs.insert("c".to_string(), Array1::from(vec![-1.0, 0.0, 1.0, 2.0]));

        let result = coordinator
            .execute_smpc(inputs.clone(), SMPCComputation::Sum)
            .expect("smpc execution failed");

        assert_eq!(result.result.len(), 4);
        let expected = [0.5, 2.5, 4.5, 6.5];
        for (index, expected_value) in expected.iter().enumerate() {
            assert!(
                (result.result[index] - expected_value).abs() < 1e-9,
                "coordinate {index}: got {}",
                result.result[index]
            );
        }
        assert_eq!(result.participating_parties, vec!["a", "b", "c"]);
        assert!(matches!(
            coordinator.protocol_state(),
            SMPCProtocolState::Completed
        ));
    }

    #[test]
    fn test_execute_smpc_average() {
        let mut coordinator = coordinator_with_participants(2, 2, &["a", "b"]);

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0, 3.0]));
        inputs.insert("b".to_string(), Array1::from(vec![3.0, 5.0]));

        let result = coordinator
            .execute_smpc(inputs, SMPCComputation::Average)
            .expect("smpc execution failed");

        assert_eq!(result.result.len(), 2);
        assert!((result.result[0] - 2.0).abs() < 1e-9);
        assert!((result.result[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_execute_smpc_validates_inputs_and_aborts() {
        let mut coordinator = coordinator_with_participants(3, 2, &["a", "b"]);

        let mut mismatched = HashMap::new();
        mismatched.insert("a".to_string(), Array1::from(vec![1.0, 2.0]));
        mismatched.insert("b".to_string(), Array1::from(vec![1.0]));
        assert!(coordinator
            .execute_smpc(mismatched, SMPCComputation::Sum)
            .is_err());
        assert!(matches!(
            coordinator.protocol_state(),
            SMPCProtocolState::Aborted(_)
        ));

        let mut unknown = HashMap::new();
        unknown.insert("zz".to_string(), Array1::from(vec![1.0]));
        assert!(coordinator
            .execute_smpc(unknown, SMPCComputation::Sum)
            .is_err());

        let mut empty_input = HashMap::new();
        empty_input.insert("a".to_string(), Array1::from(Vec::<f64>::new()));
        assert!(coordinator
            .execute_smpc(empty_input, SMPCComputation::Sum)
            .is_err());
    }

    #[test]
    fn test_execute_smpc_requires_enough_active_participants() {
        let mut coordinator = coordinator_with_participants(3, 3, &["a", "b", "c"]);
        if let Some(participant) = coordinator.participants.get_mut("c") {
            participant.status = ParticipantStatus::Malicious;
        }

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0]));
        inputs.insert("b".to_string(), Array1::from(vec![1.0]));
        assert!(coordinator
            .execute_smpc(inputs, SMPCComputation::Sum)
            .is_err());
    }

    #[test]
    fn test_unsupported_computations_report_errors() {
        let mut coordinator = coordinator_with_participants(2, 2, &["a", "b"]);
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0]));
        inputs.insert("b".to_string(), Array1::from(vec![2.0]));

        assert!(coordinator
            .execute_smpc(inputs.clone(), SMPCComputation::WeightedSum(vec![0.5, 0.5]))
            .is_err());
        assert!(coordinator
            .execute_smpc(inputs, SMPCComputation::Custom("median".to_string()))
            .is_err());
    }

    #[test]
    fn test_security_guarantees_are_derived_not_asserted() {
        let mut coordinator = coordinator_with_participants(2, 2, &["a", "b"]);
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array1::from(vec![1.0]));
        inputs.insert("b".to_string(), Array1::from(vec![2.0]));

        let result = coordinator
            .execute_smpc(inputs, SMPCComputation::Sum)
            .expect("smpc execution failed");
        let guarantees = &result.security_guarantees;

        // The old implementation hardcoded information-theoretic privacy with
        // soundness and completeness whatever ran.
        assert_eq!(guarantees.privacy_level, PrivacyLevel::Computational);
        assert!(!guarantees.soundness);
        assert!(guarantees.completeness);
        assert_eq!(guarantees.malicious_tolerance, 0);
        assert_eq!(
            guarantees.communication_security,
            CommunicationSecurity::SemiHonest
        );
        assert!(!guarantees.limitations.is_empty());
    }

    #[test]
    fn test_coordinator_secure_aggregate_path() {
        let mut coordinator = SMPCCoordinator::<f64>::new(test_config(2, 2))
            .expect("coordinator construction failed");
        let mut scheme = CommitmentScheme::<f64>::new();

        let mut inputs = HashMap::new();
        for id in ["a", "b"] {
            let input = Array1::from(vec![2.0, 4.0]);
            let (commitment, nonce) = scheme.commit(&input).expect("commit failed");
            coordinator
                .add_participant(
                    Participant::new(id, vec![], 1.0).with_commitment(commitment, nonce),
                )
                .expect("participant registration failed");
            inputs.insert(id.to_string(), input);
        }

        let aggregated = coordinator
            .secure_aggregate(&inputs)
            .expect("aggregation failed");
        assert_eq!(aggregated.aggregate.len(), 2);
        assert!((aggregated.aggregate[0] - 2.0).abs() < 1e-12);
        assert_eq!(aggregated.security_level, CommunicationSecurity::SemiHonest);
    }
}
