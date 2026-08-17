//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap;

    fn cohort(entries: &[(&str, &[f64])]) -> HashMap<String, Array1<f64>> {
        entries
            .iter()
            .map(|(id, values)| ((*id).to_string(), Array1::from(values.to_vec())))
            .collect()
    }

    fn aggregator(
        method: ByzantineAggregationMethod,
        max_byzantine: usize,
        min_participants: usize,
    ) -> ByzantineTolerantAggregator<f64> {
        let config = ByzantineConfig {
            max_byzantine,
            min_participants,
            aggregation_method: method,
            gradient_verification: false,
            ..ByzantineConfig::default()
        };
        ByzantineTolerantAggregator::new(config).expect("configuration must be valid")
    }

    // -- configuration ------------------------------------------------------

    #[test]
    fn test_byzantine_config_accepts_sound_settings() {
        let config = ByzantineConfig {
            max_byzantine: 2,
            min_participants: 7,
            aggregation_method: ByzantineAggregationMethod::Krum,
            anomaly_threshold: 0.5,
            reputation_decay: 0.9,
            gradient_verification: true,
            outlier_detection: OutlierDetectionMethod::ZScore,
            consensus_threshold: 0.7,
        };

        assert_eq!(config.max_byzantine, 2);
        assert_eq!(config.min_participants, 7);
        assert_eq!(config.required_participants(), 7);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_byzantine_config_rejects_undersized_krum_cohort() {
        // Krum needs n >= 2f + 3 = 7; the old code accepted this and then underflowed.
        let config = ByzantineConfig {
            max_byzantine: 2,
            min_participants: 5,
            aggregation_method: ByzantineAggregationMethod::Krum,
            ..ByzantineConfig::default()
        };
        assert!(config.validate().is_err());
        assert!(ByzantineTolerantAggregator::<f64>::new(config).is_err());
    }

    #[test]
    fn test_byzantine_config_rejects_zero_denominator_state() {
        // This is the config that used to make confidence_score NaN.
        let config = ByzantineConfig {
            max_byzantine: 0,
            min_participants: 0,
            aggregation_method: ByzantineAggregationMethod::Median,
            ..ByzantineConfig::default()
        };
        assert!(config.validate().is_err());

        for bad in [0.0, -0.1, 1.5, f64::NAN] {
            let config = ByzantineConfig {
                consensus_threshold: bad,
                ..ByzantineConfig::default()
            };
            assert!(config.validate().is_err(), "consensus_threshold {bad}");

            let config = ByzantineConfig {
                anomaly_threshold: bad,
                ..ByzantineConfig::default()
            };
            assert!(config.validate().is_err(), "anomaly_threshold {bad}");
        }

        for bad in [1.0, 1.5, -0.1, f64::NAN] {
            let config = ByzantineConfig {
                reputation_decay: bad,
                ..ByzantineConfig::default()
            };
            assert!(config.validate().is_err(), "reputation_decay {bad}");
        }
    }

    #[test]
    fn test_bulyan_config_requires_four_f_plus_three() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 6,
            aggregation_method: ByzantineAggregationMethod::Bulyan,
            ..ByzantineConfig::default()
        };
        assert_eq!(config.required_participants(), 7);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_reputation_score() {
        let mut reputation = ReputationScore::new();
        assert_eq!(reputation.score, 0.7);
        assert_eq!(reputation.successful_aggregations, 0);
        assert_eq!(reputation.trust_level, TrustLevel::Medium);

        reputation.successful_aggregations += 1;
        reputation.score = 0.9;
        assert_eq!(reputation.successful_aggregations, 1);
    }

    // -- input validation ---------------------------------------------------

    #[test]
    fn test_non_finite_gradients_are_rejected() {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 2.0]),
            ("b", &[1.0, f64::NAN]),
            ("c", &[1.0, 2.0]),
        ]);
        assert!(agg.byzantine_robust_aggregate(&gradients).is_err());

        let gradients = cohort(&[("a", &[1.0, 2.0]), ("b", &[1.0, f64::INFINITY])]);
        assert!(agg.byzantine_robust_aggregate(&gradients).is_err());
    }

    #[test]
    fn test_ragged_cohort_is_rejected() {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        let gradients = cohort(&[("a", &[1.0, 2.0]), ("b", &[1.0]), ("c", &[1.0, 2.0])]);
        assert!(agg.byzantine_robust_aggregate(&gradients).is_err());
    }

    // -- trimmed mean (F12, F13) --------------------------------------------

    #[test]
    fn test_trimmed_mean_rejects_cohort_too_small_to_trim() {
        // Regression: n = 2 with one trimmed value per tail used to slice an empty
        // range and silently return an all-zero update.
        let agg = aggregator(ByzantineAggregationMethod::TrimmedMean, 1, 3);
        let gradients = cohort(&[("a", &[1.0, 2.0]), ("b", &[3.0, 4.0])]);

        let result = agg.trimmed_mean_aggregation(&gradients, 1);
        assert!(
            result.is_err(),
            "n = 2 with trim 1 per tail must be an error"
        );

        // n = 3 with trim 1 keeps exactly the median and is accepted.
        let gradients = cohort(&[("a", &[1.0]), ("b", &[2.0]), ("c", &[100.0])]);
        let result = agg
            .trimmed_mean_aggregation(&gradients, 1)
            .expect("n = 3 with trim 1 must succeed");
        assert!((result[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_trims_max_byzantine_per_tail() {
        // Regression for the hardcoded 10% trim: with n = 5 and f = 2, Yin et al.
        // requires two values dropped from each tail (leaving the median alone).
        // The old 10% rule dropped one, letting a Byzantine value survive.
        let agg = aggregator(ByzantineAggregationMethod::TrimmedMean, 2, 5);
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[2.0]),
            ("c", &[3.0]),
            ("d", &[50.0]),
            ("e", &[100.0]),
        ]);

        let robust = agg
            .trimmed_mean_aggregation(&gradients, 2)
            .expect("trimming 2 per tail from 5 gradients must succeed");
        assert!((robust[0] - 3.0).abs() < 1e-12, "got {}", robust[0]);

        // What the old hardcoded 10% (one per tail) would have produced.
        let weak = agg
            .trimmed_mean_aggregation(&gradients, 1)
            .expect("trimming 1 per tail must succeed");
        assert!((weak[0] - 18.333_333_333).abs() < 1e-6, "got {}", weak[0]);
    }

    #[test]
    fn test_trimmed_mean_aggregation() {
        let agg = aggregator(ByzantineAggregationMethod::TrimmedMean, 1, 3);

        let gradients = cohort(&[
            ("client1", &[1.0, 2.0, 3.0]),
            ("client2", &[1.1, 2.1, 3.1]),
            ("client3", &[0.9, 1.9, 2.9]),
            ("client4", &[1.0, 2.0, 3.0]),
            ("client5", &[10.0, 20.0, 30.0]),
        ]);

        let result = agg
            .trimmed_mean_aggregation(&gradients, 1)
            .expect("trimmed mean must succeed");

        // Trimming one value per tail removes both the low and the high extreme.
        assert!(
            (result[0] - 1.033_333_333).abs() < 1e-6,
            "got {}",
            result[0]
        );
        assert!(
            (result[1] - 2.033_333_333).abs() < 1e-6,
            "got {}",
            result[1]
        );
        assert!(
            (result[2] - 3.033_333_333).abs() < 1e-6,
            "got {}",
            result[2]
        );
    }

    #[test]
    fn test_coordinate_median_aggregation() {
        let agg = aggregator(ByzantineAggregationMethod::CoordinateMedian, 1, 3);
        let gradients = cohort(&[
            ("client1", &[1.0, 2.0, 3.0]),
            ("client2", &[2.0, 3.0, 4.0]),
            ("client3", &[3.0, 4.0, 5.0]),
        ]);

        let result = agg
            .coordinate_median_aggregation(&gradients)
            .expect("median must succeed");

        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
    }

    // -- Krum family (F11, F14) ---------------------------------------------

    #[test]
    fn test_krum_rejects_undersized_cohort_without_panicking() {
        // Regression: `participants.len() - max_byzantine - 2` underflowed for
        // f = 2 and 3 survivors, panicking in debug and wrapping in release.
        let agg = aggregator(ByzantineAggregationMethod::Krum, 2, 7);
        let gradients = cohort(&[("a", &[1.0]), ("b", &[2.0]), ("c", &[3.0])]);

        assert!(agg.krum_aggregation(&gradients, 2).is_err());
        assert!(agg.multi_krum_aggregation(&gradients, 2).is_err());
        assert!(agg.bulyan_aggregation(&gradients, 2).is_err());
    }

    #[test]
    fn test_krum_selects_a_representative_gradient() {
        let agg = aggregator(ByzantineAggregationMethod::Krum, 0, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.1, 1.1]),
            ("c", &[100.0, 100.0]),
        ]);

        let result = agg.krum_aggregation(&gradients, 0).expect("krum");
        assert!(
            result[0] < 2.0,
            "Krum must not pick the outlier: {result:?}"
        );
    }

    #[test]
    fn test_multi_krum_averages_the_selected_gradients() {
        let agg = aggregator(ByzantineAggregationMethod::MultiKrum, 1, 5);
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[1.0]),
            ("c", &[1.0]),
            ("d", &[1.0]),
            ("e", &[100.0]),
        ]);

        let result = agg
            .multi_krum_aggregation(&gradients, 1)
            .expect("multi-krum");
        assert!((result[0] - 1.0).abs() < 1e-12, "got {}", result[0]);
    }

    #[test]
    fn test_bulyan_runs_iterative_krum_and_median_selection() {
        // n = 7, f = 1 satisfies n >= 4f + 3. theta = 5, beta = 3.
        let agg = aggregator(ByzantineAggregationMethod::Bulyan, 1, 7);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.0]),
            ("c", &[1.0, 1.0]),
            ("d", &[1.0, 1.0]),
            ("e", &[1.0, 1.0]),
            ("f", &[1.0, 1.0]),
            ("g", &[500.0, -500.0]),
        ]);

        let result = agg.bulyan_aggregation(&gradients, 1).expect("bulyan");
        assert!((result[0] - 1.0).abs() < 1e-12, "got {}", result[0]);
        assert!((result[1] - 1.0).abs() < 1e-12, "got {}", result[1]);
    }

    #[test]
    fn test_bulyan_no_longer_zeroes_the_update() {
        // Regression: the old implementation handed 2 gradients to a trimmed mean
        // that trimmed them both, returning Ok(zeros).
        let agg = aggregator(ByzantineAggregationMethod::Bulyan, 3, 15);
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[2.0]),
            ("c", &[3.0]),
            ("d", &[4.0]),
            ("e", &[5.0]),
        ]);

        // n = 5 < 4f + 3 = 15, so this is an explicit error rather than zeros.
        assert!(agg.bulyan_aggregation(&gradients, 3).is_err());
    }

    // -- outlier detectors (F10, F20) ---------------------------------------

    fn outlier_scores(
        method: OutlierDetectionMethod,
        entries: &[(&str, &[f64])],
    ) -> Vec<OutlierScore> {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        agg.config.outlier_detection = method;
        let gradients = cohort(entries);
        let ordered = ByzantineTolerantAggregator::<f64>::ordered_cohort(&gradients);
        let grads: Vec<&Array1<f64>> = ordered.iter().map(|(_, g)| *g).collect();
        let stats = agg
            .statistics_engine
            .compute_statistics(&grads)
            .expect("statistics");
        (0..grads.len())
            .map(|i| {
                agg.compute_outlier_score(i, &grads, &stats)
                    .expect("outlier score")
            })
            .collect()
    }

    #[test]
    fn test_every_outlier_method_produces_a_bounded_score() {
        // Regression: IsolationForest / LocalOutlierFactor / MahalanobisDistance
        // fell into a wildcard arm that recursed into itself forever.
        let entries: &[(&str, &[f64])] = &[
            ("a", &[1.0, 1.0, 1.0]),
            ("b", &[1.1, 0.9, 1.0]),
            ("c", &[0.9, 1.1, 1.0]),
            ("d", &[1.0, 1.0, 1.1]),
            ("e", &[50.0, -50.0, 50.0]),
        ];

        for method in [
            OutlierDetectionMethod::ZScore,
            OutlierDetectionMethod::IQR,
            OutlierDetectionMethod::IsolationForest,
            OutlierDetectionMethod::LocalOutlierFactor,
            OutlierDetectionMethod::MahalanobisDistance,
        ] {
            let scores = outlier_scores(method, entries);
            assert_eq!(scores.len(), entries.len());
            for score in &scores {
                assert!(score.score.is_finite(), "{method:?} produced {score:?}");
                assert!(
                    (0.0..=1.0).contains(&score.score),
                    "{method:?} produced {score:?}"
                );
                assert_eq!(score.method, method);
            }
            // The planted outlier is index 4 in id order.
            assert!(
                scores[4].score >= scores[0].score,
                "{method:?} did not rank the outlier at least as high as an inlier"
            );
        }
    }

    #[test]
    fn test_iqr_score_is_finite_when_the_iqr_is_zero() {
        // Regression: [1, 1, 1, 1, 10] gives q1 == q3 == 1, so the dissenter used to
        // divide by zero and score +inf.
        let entries: &[(&str, &[f64])] = &[
            ("a", &[1.0]),
            ("b", &[1.0]),
            ("c", &[1.0]),
            ("d", &[1.0]),
            ("e", &[10.0]),
        ];
        let scores = outlier_scores(OutlierDetectionMethod::IQR, entries);
        for score in &scores {
            assert!(score.score.is_finite(), "{score:?}");
        }
        assert!(scores[4].score > 0.0, "the dissenter must score above zero");
    }

    #[test]
    fn test_outlier_detectors_are_deterministic() {
        let entries: &[(&str, &[f64])] = &[
            ("a", &[1.0, 2.0]),
            ("b", &[1.2, 1.9]),
            ("c", &[0.8, 2.1]),
            ("d", &[30.0, -30.0]),
        ];
        for method in [
            OutlierDetectionMethod::IsolationForest,
            OutlierDetectionMethod::LocalOutlierFactor,
        ] {
            let first = outlier_scores(method, entries);
            let second = outlier_scores(method, entries);
            for (a, b) in first.iter().zip(second.iter()) {
                assert_eq!(a.score.to_bits(), b.score.to_bits(), "{method:?}");
            }
        }
    }

    // -- statistics (F23) ---------------------------------------------------

    #[test]
    fn test_skewness_and_kurtosis_are_computed() {
        let mut engine = StatisticalAnalysis::<f64>::new();
        let a = Array1::from(vec![-1.0]);
        let b = Array1::from(vec![0.0]);
        let c = Array1::from(vec![1.0]);
        let measures = engine
            .compute_statistics(&[&a, &b, &c])
            .expect("statistics");

        // Symmetric sample: skewness 0, standardised fourth moment 1.5.
        assert!(measures.skewness[0].abs() < 1e-12);
        assert!((measures.kurtosis[0] - 1.5).abs() < 1e-12);

        // A right-skewed sample must report a positive third moment.
        let a = Array1::from(vec![0.0]);
        let b = Array1::from(vec![0.0]);
        let c = Array1::from(vec![0.0]);
        let d = Array1::from(vec![10.0]);
        let measures = engine
            .compute_statistics(&[&a, &b, &c, &d])
            .expect("statistics");
        assert!(measures.skewness[0] > 0.5, "got {}", measures.skewness[0]);

        // A constant coordinate reports the normal reference values.
        let a = Array1::from(vec![2.0]);
        let b = Array1::from(vec![2.0]);
        let measures = engine.compute_statistics(&[&a, &b]).expect("statistics");
        assert_eq!(measures.skewness[0], 0.0);
        assert_eq!(measures.kurtosis[0], 3.0);
    }

    #[test]
    fn test_quartiles_are_reported() {
        let mut engine = StatisticalAnalysis::<f64>::new();
        let values: Vec<Array1<f64>> = (0..5).map(|i| Array1::from(vec![i as f64])).collect();
        let refs: Vec<&Array1<f64>> = values.iter().collect();
        let measures = engine.compute_statistics(&refs).expect("statistics");

        assert_eq!(measures.q1[0], 1.0);
        assert_eq!(measures.q3[0], 3.0);
        assert_eq!(measures.iqr[0], 2.0);
        assert_eq!(measures.median[0], 2.0);
    }

    // -- FoolsGold (F15) ----------------------------------------------------

    #[test]
    fn test_fools_gold_penalises_sybils() {
        let mut agg = aggregator(ByzantineAggregationMethod::FoolsGold, 1, 3);
        // Three colluding clients share a direction; two honest clients do not.
        let gradients = cohort(&[
            ("honest1", &[1.0, 0.0, 0.0, 0.0]),
            ("honest2", &[0.0, 1.0, 0.0, 0.0]),
            ("sybil1", &[0.0, 0.0, 1.0, 1.0]),
            ("sybil2", &[0.0, 0.0, 1.0, 1.0]),
            ("sybil3", &[0.0, 0.0, 1.0, 1.0]),
        ]);

        let result = agg.fools_gold_aggregation(&gradients).expect("fools gold");

        // The Sybil direction is (0, 0, 1, 1); it must be suppressed relative to the
        // honest directions even though it is the numerical majority.
        let honest_mass = result[0] + result[1];
        let sybil_mass = result[2] + result[3];
        assert!(
            honest_mass > sybil_mass * 10.0,
            "Sybils were not suppressed: {result:?}"
        );

        let ids: Vec<String> = vec![
            "honest1".to_string(),
            "honest2".to_string(),
            "sybil1".to_string(),
            "sybil2".to_string(),
            "sybil3".to_string(),
        ];
        let weights = agg.compute_fools_gold_weights(&ids).expect("weights");
        assert!(weights[0] > 0.9 && weights[1] > 0.9, "{weights:?}");
        assert!(
            weights[2] < 0.1 && weights[3] < 0.1 && weights[4] < 0.1,
            "{weights:?}"
        );
    }

    #[test]
    fn test_fools_gold_is_not_a_plain_mean() {
        // Regression: the old implementation returned reputation-weighted means, so
        // a fresh aggregator produced exactly mean(gradients).
        let mut agg = aggregator(ByzantineAggregationMethod::FoolsGold, 1, 3);
        let gradients = cohort(&[
            ("honest", &[1.0, 0.0]),
            ("sybil1", &[0.0, 1.0]),
            ("sybil2", &[0.0, 1.0]),
            ("sybil3", &[0.0, 1.0]),
        ]);

        let result = agg.fools_gold_aggregation(&gradients).expect("fools gold");
        let mean_second = 0.75;
        assert!(
            (result[1] - mean_second).abs() > 0.1,
            "FoolsGold degenerated into the mean: {result:?}"
        );
    }

    // -- FLAME (F16, F17) ---------------------------------------------------

    #[test]
    fn test_flame_clustering_is_deterministic() {
        // Regression: the cluster seed came from HashMap::iter().next(), so the
        // admitted set changed between processes.
        let agg = aggregator(ByzantineAggregationMethod::FLAME, 1, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.1]),
            ("c", &[0.9, 1.0]),
            ("d", &[-1.0, -1.0]),
            ("e", &[-1.0, -0.9]),
        ]);

        let first = agg.flame_admitted_ids(&gradients).expect("clustering");
        for _ in 0..25 {
            let again = agg.flame_admitted_ids(&gradients).expect("clustering");
            assert_eq!(first, again);
        }
        assert!(first.len() > gradients.len() / 2);
        assert!(first.contains(&"a".to_string()));
        assert!(!first.contains(&"d".to_string()));
    }

    #[test]
    fn test_flame_clips_scaling_attacks() {
        let agg = aggregator(ByzantineAggregationMethod::FLAME, 1, 3);
        // The attacker agrees in direction (so it survives clustering) but inflates
        // its norm by 1000x; norm-median clipping must neutralise that.
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.0]),
            ("c", &[1.0, 1.0]),
            ("d", &[1.0, 1.0]),
            ("attacker", &[1000.0, 1000.0]),
        ]);

        let result = agg.flame_aggregation(&gradients).expect("flame");
        assert!(
            result[0] < 2.0,
            "clipping did not bound the attack: {result:?}"
        );
        assert!(
            result[0] > 0.5,
            "the honest signal was destroyed: {result:?}"
        );
    }

    #[test]
    fn test_flame_noise_is_small_relative_to_the_update() {
        let agg = aggregator(ByzantineAggregationMethod::FLAME, 1, 3);
        let gradients = cohort(&[("a", &[1.0]), ("b", &[1.0]), ("c", &[1.0])]);
        let result = agg.flame_aggregation(&gradients).expect("flame");
        assert!((result[0] - 1.0).abs() < 0.05, "got {}", result[0]);
    }

    // -- anomaly detection (F18, F19) ---------------------------------------

    #[test]
    fn test_anomaly_history_is_per_participant_and_excludes_the_sample() {
        let mut detector = AnomalyDetector::<f64>::new(0.5);

        // Alice builds a history of comparable, slightly varying updates.
        for magnitude in [1.0, 1.1, 0.9, 1.05, 0.95] {
            let score = detector
                .detect_anomaly("alice", &Array1::from(vec![magnitude, 0.0]))
                .expect("anomaly");
            assert!(score.score.is_finite());
        }

        // A huge update from Alice deviates from her own baseline...
        let alice_attack = detector
            .detect_anomaly("alice", &Array1::from(vec![100.0, 0.0]))
            .expect("anomaly");
        assert!(alice_attack.score > 0.0, "{alice_attack:?}");
        assert!(alice_attack.is_anomalous, "{alice_attack:?}");

        // ...and Bob, who has no history of his own, is unaffected by it.
        let bob = detector
            .detect_anomaly("bob", &Array1::from(vec![1.0, 0.0]))
            .expect("anomaly");
        assert_eq!(bob.score, 0.0, "{bob:?}");

        let alice_history = detector
            .gradient_stats()
            .norm_history_of("alice")
            .expect("alice history");
        let bob_history = detector
            .gradient_stats()
            .norm_history_of("bob")
            .expect("bob history");
        assert_eq!(alice_history.len(), 6);
        assert_eq!(bob_history.len(), 1);
    }

    #[test]
    fn test_anomaly_score_has_no_constant_pattern_term() {
        // Regression: compute_pattern_deviation always returned 0.5, so the combined
        // score could never drop below 0.25.
        let mut detector = AnomalyDetector::<f64>::new(0.5);
        for _ in 0..5 {
            let score = detector
                .detect_anomaly("alice", &Array1::from(vec![1.0, 0.0]))
                .expect("anomaly");
            assert!(
                score.score < 0.25,
                "constant pattern term survives: {score:?}"
            );
        }
        assert_eq!(detector.pattern_counts(), (0, 0));
    }

    #[test]
    fn test_pattern_model_learns_and_scores() {
        let mut model = PatternModel::<f64>::new();
        assert_eq!(
            model
                .compute_pattern_deviation(&Array1::from(vec![1.0, 0.0]))
                .expect("deviation"),
            None
        );

        model
            .learn_normal(&Array1::from(vec![1.0, 0.0]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (1, 0));

        // A second, near-identical observation is merged rather than appended.
        model
            .learn_normal(&Array1::from(vec![2.0, 0.05]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (1, 0));

        // A different direction becomes a new prototype.
        model
            .learn_normal(&Array1::from(vec![0.0, 1.0]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (2, 0));

        let aligned = model
            .compute_pattern_deviation(&Array1::from(vec![3.0, 0.0]))
            .expect("deviation")
            .expect("some");
        let opposite = model
            .compute_pattern_deviation(&Array1::from(vec![-1.0, -1.0]))
            .expect("deviation")
            .expect("some");
        assert!(aligned < 0.1, "aligned update scored {aligned}");
        assert!(opposite > aligned, "opposite update scored {opposite}");

        // A known attack direction is flagged even when it is not novel.
        model
            .learn_attack(&Array1::from(vec![0.0, -1.0]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (2, 1));
        let attack = model
            .compute_pattern_deviation(&Array1::from(vec![0.0, -5.0]))
            .expect("deviation")
            .expect("some");
        assert!(attack > 0.9, "attack direction scored {attack}");
        assert!(model.matching_threshold() > 0.0);
    }

    // -- confidence and reputation (F21, F22) -------------------------------

    #[test]
    fn test_confidence_score_is_finite_and_meaningful() {
        let agg = aggregator(ByzantineAggregationMethod::Median, 0, 1);

        let tight = cohort(&[("a", &[1.0]), ("b", &[1.0]), ("c", &[1.0])]);
        let aggregate = Array1::from(vec![1.0]);
        let tight_score = agg
            .calculate_confidence_score(&tight, 3, &aggregate)
            .expect("confidence");
        assert!(tight_score.is_finite());
        assert!((tight_score - 1.0).abs() < 1e-9, "got {tight_score}");

        let spread = cohort(&[("a", &[1.0]), ("b", &[-3.0]), ("c", &[7.0])]);
        let spread_score = agg
            .calculate_confidence_score(&spread, 6, &aggregate)
            .expect("confidence");
        assert!(spread_score.is_finite());
        assert!(spread_score < tight_score, "got {spread_score}");
        assert!((0.0..=1.0).contains(&spread_score));
    }

    #[test]
    fn test_reputation_uses_the_configured_decay() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 3,
            aggregation_method: ByzantineAggregationMethod::Median,
            reputation_decay: 0.5,
            gradient_verification: false,
            ..ByzantineConfig::default()
        };
        let mut agg = ByzantineTolerantAggregator::<f64>::new(config).expect("config");

        let honest = cohort(&[("good", &[1.0])]);
        agg.update_reputations(&honest, &["bad".to_string()])
            .expect("reputations");

        // Honest: 0.5 * 0.7 + 0.5 = 0.85. Byzantine rate = min(0.5 * 5, 1) = 1.
        let good = agg.reputation("good").expect("good");
        assert!((good.score - 0.85).abs() < 1e-12, "got {}", good.score);
        assert_eq!(good.trust_level, TrustLevel::High);

        let bad = agg.reputation("bad").expect("bad");
        assert!(bad.score.abs() < 1e-12, "got {}", bad.score);
        assert_eq!(bad.trust_level, TrustLevel::Blacklisted);
    }

    #[test]
    fn test_gradient_verifier_enforces_expected_properties() {
        let verifier = GradientVerifier::<f64>::with_properties(GradientProperties {
            norm_range: (0.5, 10.0),
            sparsity_threshold: 0.5,
            direction_consistency: 0.0,
        });
        assert_eq!(verifier.expected_properties().sparsity_threshold, 0.5);

        let good = verifier
            .verify_gradient(&Array1::from(vec![1.0, 1.0]))
            .expect("verify");
        assert!(good.passed, "{good:?}");

        let too_large = verifier
            .verify_gradient(&Array1::from(vec![100.0, 100.0]))
            .expect("verify");
        assert!(too_large.score < good.score, "{too_large:?}");
        assert_eq!(too_large.rule_scores.get("Norm range"), Some(&0.0));

        let too_sparse = verifier
            .verify_gradient(&Array1::from(vec![1.0, 0.0, 0.0, 0.0]))
            .expect("verify");
        assert_eq!(too_sparse.rule_scores.get("Sparsity"), Some(&0.0));

        let mut verifier = verifier;
        verifier.set_reference_direction(Array1::from(vec![1.0, 1.0]));
        let opposed = verifier
            .verify_gradient(&Array1::from(vec![-1.0, -1.0]))
            .expect("verify");
        assert_eq!(opposed.rule_scores.get("Direction consistency"), Some(&0.0));
    }

    // -- full pipeline ------------------------------------------------------

    #[test]
    fn test_full_pipeline_records_behaviour_and_reputations() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 4,
            aggregation_method: ByzantineAggregationMethod::CoordinateMedian,
            outlier_detection: OutlierDetectionMethod::ZScore,
            gradient_verification: true,
            consensus_threshold: 0.5,
            ..ByzantineConfig::default()
        };
        let mut agg = ByzantineTolerantAggregator::<f64>::new(config).expect("config");

        let gradients = cohort(&[
            ("a", &[1.0, 2.0]),
            ("b", &[1.1, 2.1]),
            ("c", &[0.9, 1.9]),
            ("d", &[1.0, 2.0]),
            ("e", &[1.05, 1.95]),
        ]);

        let result = agg
            .byzantine_robust_aggregate(&gradients)
            .expect("aggregation");

        assert!(result.aggregate.iter().all(|x| x.is_finite()));
        assert!(result.confidence_score.is_finite());
        assert!((0.0..=1.0).contains(&result.confidence_score));
        assert!((0.0..=1.0).contains(&result.consensus_ratio));
        assert_eq!(agg.round(), 1);

        let mut expected: Vec<String> = gradients.keys().cloned().collect();
        expected.sort();
        assert_eq!(result.honest_participants, expected);

        for id in &expected {
            let history = agg.behavior_history(id).expect("behaviour history");
            assert_eq!(history.rounds_participated, 1);
            assert_eq!(history.gradient_norms.len(), 1);
            assert_eq!(history.gradient_similarities.len(), 1);
            assert_eq!(history.participation_pattern, vec![true]);

            let reputation = agg.reputation(id).expect("reputation");
            assert_eq!(reputation.successful_aggregations, 1);
            assert!(reputation.gradient_quality.is_finite());
            assert!(reputation.consistency_score > 0.5);
        }

        // The pattern model has now learned from the accepted updates.
        assert!(agg.anomaly_detector.pattern_counts().0 > 0);
    }

    #[test]
    fn test_full_pipeline_is_deterministic() {
        let gradients = cohort(&[
            ("a", &[1.0, 2.0]),
            ("b", &[1.1, 2.1]),
            ("c", &[0.9, 1.9]),
            ("d", &[1.0, 2.0]),
            ("e", &[3.0, -4.0]),
        ]);

        let run = || {
            let config = ByzantineConfig {
                max_byzantine: 1,
                min_participants: 3,
                aggregation_method: ByzantineAggregationMethod::CoordinateMedian,
                outlier_detection: OutlierDetectionMethod::LocalOutlierFactor,
                gradient_verification: true,
                ..ByzantineConfig::default()
            };
            let mut agg = ByzantineTolerantAggregator::<f64>::new(config).expect("config");
            agg.byzantine_robust_aggregate(&gradients)
                .expect("aggregation")
        };

        let first = run();
        let second = run();
        assert_eq!(first.honest_participants, second.honest_participants);
        assert_eq!(first.byzantine_participants, second.byzantine_participants);
        assert_eq!(
            first
                .aggregate
                .to_vec()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            second
                .aggregate
                .to_vec()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_consensus_threshold_gates_a_split_cohort() {
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[1.0]),
            ("c", &[1.0]),
            ("d", &[900.0]),
            ("e", &[-900.0]),
        ]);

        let build = |consensus_threshold: f64| {
            let config = ByzantineConfig {
                max_byzantine: 1,
                min_participants: 3,
                aggregation_method: ByzantineAggregationMethod::Median,
                anomaly_threshold: 0.1,
                consensus_threshold,
                gradient_verification: false,
                ..ByzantineConfig::default()
            };
            ByzantineTolerantAggregator::<f64>::new(config).expect("config")
        };

        // The two extremes are flagged, so 3 of 5 participants survive.
        let mut permissive = build(0.5);
        let accepted = permissive
            .byzantine_robust_aggregate(&gradients)
            .expect("60% consensus clears a 50% threshold");
        assert_eq!(accepted.byzantine_participants.len(), 2);
        assert!((accepted.consensus_ratio - 0.6).abs() < 1e-12);
        assert!((accepted.aggregate[0] - 1.0).abs() < 1e-12);

        // The same round is rejected when 90% consensus is demanded.
        let mut strict = build(0.9);
        let error = strict
            .byzantine_robust_aggregate(&gradients)
            .expect_err("60% consensus must not clear a 90% threshold");
        assert!(
            format!("{error}").contains("consensus"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_blacklisted_participants_are_filtered() {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        for _ in 0..5 {
            agg.update_reputations(&HashMap::new(), &["bad".to_string()])
                .expect("reputations");
        }
        assert_eq!(
            agg.reputation("bad").expect("bad").trust_level,
            TrustLevel::Blacklisted
        );

        let gradients = cohort(&[("good1", &[1.0]), ("good2", &[1.0]), ("bad", &[500.0])]);
        let filtered = agg.filter_by_reputation(&gradients).expect("filter");
        assert_eq!(filtered.len(), 2);
        assert!(!filtered.contains_key("bad"));
    }

    #[test]
    fn test_euclidean_distance() {
        let agg = aggregator(ByzantineAggregationMethod::Krum, 1, 5);

        let a = Array1::from(vec![1.0, 2.0, 3.0]);
        let b = Array1::from(vec![4.0, 5.0, 6.0]);

        let distance = agg
            .compute_euclidean_distance(&a, &b)
            .expect("distance must be computable");
        let expected = (3.0_f64.powi(2) + 3.0_f64.powi(2) + 3.0_f64.powi(2)).sqrt();

        assert!((distance - expected).abs() < 1e-10);
        assert!(agg
            .compute_euclidean_distance(&a, &Array1::from(vec![1.0]))
            .is_err());
    }

    #[test]
    fn test_cosine_similarity_handles_zero_norms() {
        let agg = aggregator(ByzantineAggregationMethod::Krum, 1, 5);
        let a = Array1::from(vec![1.0, 0.0]);
        let zero = Array1::from(vec![0.0, 0.0]);
        assert_eq!(
            agg.compute_cosine_similarity(&a, &zero)
                .expect("similarity"),
            0.0
        );
        assert!((agg.compute_cosine_similarity(&a, &a).expect("similarity") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_geometric_median_converges() {
        let agg = aggregator(ByzantineAggregationMethod::GeometricMedian, 1, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.0]),
            ("c", &[1.0, 1.0]),
            ("d", &[100.0, 100.0]),
        ]);
        let result = agg
            .geometric_median_aggregation(&gradients)
            .expect("geometric median");
        assert!(result.iter().all(|x| x.is_finite()));
        assert!((result[0] - 1.0).abs() < 1e-6, "got {}", result[0]);
    }
}
