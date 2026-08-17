//! Tests for the privacy-utility analyzer.
//!
//! The module intentionally exports no runtime items; it holds the test suite
//! that pins the behaviour of [`super::types::PrivacyUtilityAnalyzer`].

#[cfg(test)]
mod tests {
    use crate::error::OptimError;
    use crate::privacy::utility_analysis::risk::binary_inference_advantage_bound;
    use crate::privacy::utility_analysis::types::{
        CorrectionMethodApplied, PerturbationType, PrivacyUtilityAnalyzer, RiskCategory,
    };
    use crate::privacy::utility_analysis::types_3::{
        AnalysisConfig, ComplianceStatus, EpsilonSemantics, ParameterRange, PrivacyConfiguration,
        PrivacyParameterSpace, SamplingStrategy,
    };
    use crate::privacy::{NoiseMechanism, PrivacyBudget};
    use scirs2_core::ndarray::Array1;

    fn seeded_config() -> AnalysisConfig {
        AnalysisConfig {
            random_seed: Some(20_240_517),
            ..AnalysisConfig::default()
        }
    }

    fn make_analyzer() -> PrivacyUtilityAnalyzer<f64> {
        PrivacyUtilityAnalyzer::new(seeded_config()).expect("default configuration is valid")
    }

    fn analyzer_with(config: AnalysisConfig) -> PrivacyUtilityAnalyzer<f64> {
        PrivacyUtilityAnalyzer::new(config).expect("configuration is valid")
    }

    fn make_budget(eps_remaining: f64) -> PrivacyBudget {
        PrivacyBudget {
            epsilon_consumed: 0.0,
            delta_consumed: 0.0,
            epsilon_remaining: eps_remaining,
            delta_remaining: 1e-5,
            steps_taken: 0,
            accounting_method: crate::privacy::AccountingMethod::MomentsAccountant,
            estimated_steps_remaining: 1000,
        }
    }

    fn make_config(epsilon: f64) -> PrivacyConfiguration<f64> {
        PrivacyConfiguration {
            epsilon,
            delta: 1e-5,
            noise_multiplier: 1.1,
            clipping_threshold: 1.0,
            sampling_probability: 0.1,
            iterations: 1000,
            batch_size: 256,
            learning_rate: 0.01,
            noise_mechanism: NoiseMechanism::Gaussian,
        }
    }

    /// A non-degenerate oracle: utility grows with epsilon and shrinks with the
    /// noise multiplier, so the frontier has many distinct points.
    fn utility_oracle(
        _data: &Array1<f64>,
        config: &PrivacyConfiguration<f64>,
    ) -> crate::error::Result<f64> {
        let eps_term = config.epsilon / (1.0 + config.epsilon);
        let noise_term = 1.0 / (1.0 + config.noise_multiplier);
        Ok(eps_term * noise_term)
    }

    // ---------------------------------------------------------------- F110

    #[test]
    fn test_parameter_range_rejects_invalid_ranges() {
        assert!(ParameterRange::new(1.0, 0.5, 10, SamplingStrategy::Linear).is_err());
        assert!(ParameterRange::new(f64::NAN, 1.0, 10, SamplingStrategy::Linear).is_err());
        assert!(ParameterRange::new(0.0, 1.0, 10, SamplingStrategy::Logarithmic).is_err());
        assert!(ParameterRange::new(-1.0, 1.0, 10, SamplingStrategy::Logarithmic).is_err());
        assert!(ParameterRange::new(0.1, 1.0, 0, SamplingStrategy::Linear).is_err());
        assert!(ParameterRange::new(0.1, 1.0, 5, SamplingStrategy::Logarithmic).is_ok());
    }

    #[test]
    fn test_inverted_range_is_rejected_instead_of_panicking() {
        // The old sampler called gen_range(min..max) directly and panicked.
        let mut space = PrivacyParameterSpace::default();
        space.epsilon_range = ParameterRange {
            min: 10.0,
            max: 0.1,
            num_samples: 10,
            sampling_strategy: SamplingStrategy::Random,
        };
        let config = AnalysisConfig {
            privacy_parameters: space,
            ..seeded_config()
        };
        let error = PrivacyUtilityAnalyzer::<f64>::new(config)
            .err()
            .expect("an inverted range must be rejected");
        assert!(matches!(error, OptimError::InvalidParameter(_)));
    }

    #[test]
    fn test_non_positive_logarithmic_range_is_rejected() {
        let mut space = PrivacyParameterSpace::default();
        space.delta_range = ParameterRange {
            min: 0.0,
            max: 1e-3,
            num_samples: 10,
            sampling_strategy: SamplingStrategy::Logarithmic,
        };
        let config = AnalysisConfig {
            privacy_parameters: space,
            ..seeded_config()
        };
        assert!(PrivacyUtilityAnalyzer::<f64>::new(config).is_err());
    }

    #[test]
    fn test_unimplemented_sampling_strategy_reports_error() {
        let mut space = PrivacyParameterSpace::default();
        space.epsilon_range = ParameterRange {
            min: 0.1,
            max: 10.0,
            num_samples: 10,
            sampling_strategy: SamplingStrategy::Sobol,
        };
        let analyzer = analyzer_with(AnalysisConfig {
            privacy_parameters: space,
            ..seeded_config()
        });
        let error = analyzer
            .generate_privacy_configurations()
            .err()
            .expect("Sobol sampling is not implemented and must not fall back to linear");
        assert!(matches!(error, OptimError::UnsupportedOperation(_)));
    }

    // ----------------------------------------------------------------- F95

    #[test]
    fn test_grid_covers_the_full_parameter_ranges() {
        let analyzer = make_analyzer();
        let configs = analyzer
            .generate_privacy_configurations()
            .expect("default space yields configurations");
        assert!(!configs.is_empty());
        assert!(configs.len() <= analyzer.config().pareto_resolution);

        let space = &analyzer.config().privacy_parameters;
        let max_epsilon = configs
            .iter()
            .fold(f64::NEG_INFINITY, |a, c| a.max(c.epsilon));
        let min_epsilon = configs.iter().fold(f64::INFINITY, |a, c| a.min(c.epsilon));
        // The old implementation never went above 0.123 for the default range.
        assert!(
            max_epsilon >= 0.9 * space.epsilon_range.max,
            "epsilon grid tops out at {max_epsilon}, range max is {}",
            space.epsilon_range.max
        );
        assert!(min_epsilon <= 1.1 * space.epsilon_range.min);

        let max_noise = configs
            .iter()
            .fold(f64::NEG_INFINITY, |a, c| a.max(c.noise_multiplier));
        assert!(
            max_noise >= 0.9 * space.noise_multiplier_range.max,
            "noise grid tops out at {max_noise}"
        );
        let max_batch = configs.iter().map(|c| c.batch_size).max().unwrap_or(0);
        assert!(
            max_batch as f64 >= 0.9 * space.batch_size_range.max,
            "batch grid tops out at {max_batch}"
        );
        let max_iterations = configs.iter().map(|c| c.iterations).max().unwrap_or(0);
        assert!(
            max_iterations as f64 >= 0.9 * space.iterations_range.max,
            "iteration grid tops out at {max_iterations}"
        );
    }

    #[test]
    fn test_grid_respects_the_resolution_cap() {
        let analyzer = analyzer_with(AnalysisConfig {
            pareto_resolution: 20,
            ..seeded_config()
        });
        let configs = analyzer
            .generate_privacy_configurations()
            .expect("configurations");
        assert!(configs.len() <= 20, "got {} configurations", configs.len());
        let max_epsilon = configs
            .iter()
            .fold(f64::NEG_INFINITY, |a, c| a.max(c.epsilon));
        assert!(max_epsilon >= 9.0, "epsilon grid tops out at {max_epsilon}");
    }

    // ----------------------------------------------------------------- F94

    #[test]
    fn test_analyze_runs_every_sub_analysis() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let results = analyzer
            .analyze(&data, utility_oracle)
            .expect("analysis succeeds for a well-behaved oracle");

        assert!(results.pareto_frontier.len() > 1, "frontier collapsed");
        assert!(!results.optimal_configurations.is_empty());
        assert!(
            results.sensitivity_results.is_some(),
            "sensitivity analysis was not run"
        );
        assert!(
            results.robustness_results.is_some(),
            "robustness evaluation was not run"
        );
        assert!(
            results.budget_recommendations.is_some(),
            "budget optimization was not run"
        );
        assert!(
            results.privacy_risk_assessment.is_some(),
            "risk assessment was not run"
        );
        assert!(
            results.statistical_tests.is_some(),
            "statistical tests were not run"
        );
        assert!(
            !results.degradation_predictions.is_empty(),
            "degradation prediction was not run"
        );
    }

    #[test]
    fn test_analyze_results_depend_on_the_oracle() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let results = analyzer
            .analyze(&data, utility_oracle)
            .expect("analysis succeeds");
        let robustness = results
            .robustness_results
            .as_ref()
            .expect("robustness present");
        let risk = results
            .privacy_risk_assessment
            .as_ref()
            .expect("risk present");

        // The old entry point returned these literals no matter the input.
        assert!(
            (robustness.robustness_score - 0.8).abs() > 1e-9,
            "robustness score is still the hardcoded 0.8"
        );
        assert!(
            (risk.overall_risk_score - 0.25).abs() > 1e-9,
            "risk score is still the hardcoded 0.25"
        );
        assert_eq!(risk.compliance_status, ComplianceStatus::Unknown);

        let tests = results.statistical_tests.as_ref().expect("tests present");
        let welch = &tests.hypothesis_tests[0];
        assert!(
            (welch.p_value - 0.01).abs() > 1e-9,
            "p-value is still the hardcoded 0.01"
        );
        // The confidence score comes from that very p-value.
        let score = results.optimal_configurations[0]
            .confidence_score
            .expect("confidence score is derived when the test ran");
        assert!((score - (1.0 - welch.p_value)).abs() < 1e-12);
    }

    #[test]
    fn test_analyze_reports_not_assessed_when_the_frontier_is_degenerate() {
        // A constant oracle collapses the frontier to a single point, so the
        // frontier-halves test cannot run: the result must be `None`, never a
        // fabricated stand-in.
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let results = analyzer
            .analyze(&data, |_, _| Ok(0.5_f64))
            .expect("analysis succeeds");
        assert!(results.statistical_tests.is_none() || results.pareto_frontier.len() >= 4);
        if results.pareto_frontier.len() < 4 {
            assert!(results.optimal_configurations[0].confidence_score.is_none());
        }
    }

    #[test]
    fn test_analyze_uses_the_configuration_of_the_recommended_point() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let results = analyzer.analyze(&data, utility_oracle).expect("analysis");
        let best = &results.optimal_configurations[0];
        // Fields used to be hardcoded (noise 1.1, batch 256, dataset 50000).
        assert_eq!(best.privacy_config.dataset_size, data.len());
        let max_utility = results
            .pareto_frontier
            .iter()
            .map(|p| p.utility_value)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((best.expected_utility - max_utility).abs() < 1e-12);
        // Every reported field comes from one and the same frontier point.
        let sourced_from_frontier = results.pareto_frontier.iter().any(|point| {
            (point.utility_value - best.expected_utility).abs() < 1e-12
                && (point.configuration.noise_multiplier - best.privacy_config.noise_multiplier)
                    .abs()
                    < 1e-12
                && point.configuration.batch_size == best.privacy_config.batch_size
                && point.configuration.iterations == best.privacy_config.max_steps
                && (point.configuration.clipping_threshold - best.privacy_config.l2_norm_clip).abs()
                    < 1e-12
        });
        assert!(
            sourced_from_frontier,
            "the recommended configuration must mirror a real frontier point"
        );
    }

    // ------------------------------------------------------------ F96/F97/F98

    #[test]
    fn test_sensitivity_confidence_interval_comes_from_replicates() {
        // Exactly linear oracle: every replicate estimate of d(utility)/d(eps)
        // is the same, so the interval has zero width. The old code reported
        // +/-19.6% of the estimate regardless of the data.
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0]);
        let base = make_config(1.0);
        let results = analyzer
            .perform_sensitivity_analysis(&data, |_, c| Ok(2.0 * c.epsilon), &base)
            .expect("sensitivity analysis succeeds");
        let slope = results.parameter_sensitivities["epsilon"];
        assert!((slope - 2.0).abs() < 1e-6, "slope={slope}");
        let (lo, hi) = results.confidence_intervals["epsilon"];
        assert!(
            (hi - lo).abs() < 1e-6,
            "a deterministic linear oracle must give a zero-width interval, got [{lo}, {hi}]"
        );
        assert!(
            (hi - lo - 2.0 * 1.96 * 0.1 * slope).abs() > 1e-6,
            "interval still looks like the fabricated 10% standard error"
        );
    }

    #[test]
    fn test_sensitivity_interval_widens_for_a_nonlinear_oracle() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0]);
        let base = make_config(1.0);
        let results = analyzer
            .perform_sensitivity_analysis(&data, |_, c| Ok(c.epsilon * c.epsilon), &base)
            .expect("sensitivity analysis succeeds");
        let (lo, hi) = results.confidence_intervals["epsilon"];
        assert!(
            hi > lo,
            "a curved oracle must produce a non-degenerate interval"
        );
    }

    #[test]
    fn test_sensitivity_without_replicates_reports_no_interval() {
        let analyzer = analyzer_with(AnalysisConfig {
            monte_carlo_samples: 1,
            ..seeded_config()
        });
        let data = Array1::<f64>::from_vec(vec![1.0]);
        let base = make_config(1.0);
        let results = analyzer
            .perform_sensitivity_analysis(&data, |_, c| Ok(c.epsilon), &base)
            .expect("sensitivity analysis succeeds");
        assert!(
            results.confidence_intervals.is_empty(),
            "a single estimate carries no uncertainty and must not report an interval"
        );
    }

    #[test]
    fn test_sensitivity_handles_zero_delta_without_nan() {
        // delta = 0 is the normal case for pure epsilon-DP; the old code
        // divided by delta * 0.01 and propagated NaN into the rankings.
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0]);
        let mut base = make_config(1.0);
        base.delta = 0.0;
        let results = analyzer
            .perform_sensitivity_analysis(&data, |_, c| Ok(c.epsilon + c.delta), &base)
            .expect("zero delta must be analyzable");
        for (name, value) in &results.parameter_sensitivities {
            assert!(value.is_finite(), "sensitivity for {name} is {value}");
        }
        assert_eq!(results.most_sensitive_parameter, "epsilon");
        assert!(results.gradient_magnitudes.values().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sensitivity_rejects_invalid_base_configuration() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0]);
        let mut base = make_config(0.0);
        base.epsilon = 0.0;
        assert!(analyzer
            .perform_sensitivity_analysis(&data, |_, c| Ok(c.epsilon), &base)
            .is_err());
    }

    #[test]
    fn test_sensitivity_reports_rankings_and_second_derivatives() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0]);
        let base = make_config(1.0);
        let results = analyzer
            .perform_sensitivity_analysis(&data, |_, c| Ok(c.epsilon * c.epsilon), &base)
            .expect("sensitivity analysis succeeds");
        assert_eq!(results.sensitivity_rankings.len(), 4);
        assert_eq!(results.sensitivity_rankings[0].0, "epsilon");
        let epsilon_local = results
            .local_sensitivities
            .iter()
            .find(|l| l.parameter == "epsilon")
            .expect("epsilon is analyzed");
        let hessian = epsilon_local.hessian.expect("second derivative available");
        assert!((hessian - 2.0).abs() < 1e-3, "hessian={hessian}");
    }

    // ----------------------------------------------------------------- F99

    #[test]
    fn test_distributional_robustness_tracks_data_not_noise() {
        // Oracle that depends on the data only. Under the old implementation
        // the "data" branch perturbed the noise multiplier, so this model
        // looked perfectly robust distributionally.
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let config = make_config(1.0);
        let results = analyzer
            .evaluate_robustness(
                &data,
                |d: &Array1<f64>, _| {
                    let mean_square = d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64;
                    Ok(1.0 / (1.0 + mean_square))
                },
                &config,
            )
            .expect("robustness evaluation succeeds");
        assert!(
            results.distributional_robustness < 1.0,
            "data perturbation had no effect: {}",
            results.distributional_robustness
        );
        let data_effect = results
            .stability_analysis
            .perturbation_analysis
            .perturbation_effects
            .iter()
            .find(|e| e.perturbation_type == PerturbationType::Data)
            .expect("data effect reported");
        let noise_effect = results
            .stability_analysis
            .perturbation_analysis
            .perturbation_effects
            .iter()
            .find(|e| e.perturbation_type == PerturbationType::Noise)
            .expect("noise effect reported");
        assert!(data_effect.utility_drop > 0.0);
        assert!(
            noise_effect.utility_drop == 0.0,
            "the oracle ignores the noise multiplier, yet a noise effect was reported"
        );
    }

    #[test]
    fn test_noise_sensitive_model_is_distributionally_robust() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let config = make_config(1.0);
        let results = analyzer
            .evaluate_robustness(&data, |_, c| Ok(1.0 / (1.0 + c.noise_multiplier)), &config)
            .expect("robustness evaluation succeeds");
        assert!(
            (results.distributional_robustness - 1.0).abs() < 1e-12,
            "a data-independent oracle must be distributionally robust, got {}",
            results.distributional_robustness
        );
        assert!(
            results.worst_case_degradation > 0.0,
            "the noise branch should still register a drop"
        );
    }

    #[test]
    fn test_adversarial_search_is_at_least_as_bad_as_any_single_branch() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let config = make_config(1.0);
        let results = analyzer
            .evaluate_robustness(
                &data,
                |d: &Array1<f64>, c| {
                    let mean = d.iter().sum::<f64>() / d.len() as f64;
                    Ok(1.0 / (1.0 + c.noise_multiplier) + 0.01 * mean)
                },
                &config,
            )
            .expect("robustness evaluation succeeds");
        assert!(
            results.adversarial_robustness <= results.distributional_robustness + 1e-12,
            "the worst case must be no better than the data branch"
        );
        assert!(results.adversarial_robustness <= results.robustness_score + 1e-12);
    }

    #[test]
    fn test_robustness_error_propagates() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let config = make_config(1.0);
        assert!(analyzer
            .evaluate_robustness(
                &data,
                |_, _| Err(OptimError::ComputationError("test error".to_string())),
                &config,
            )
            .is_err());
    }

    #[test]
    fn test_degradation_threshold_drives_failure_modes() {
        // A 20% relative drop at the largest probed magnitude.
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let config = make_config(1.0);
        let oracle =
            |_: &Array1<f64>, c: &PrivacyConfiguration<f64>| Ok(1.0 - (c.epsilon - 1.0).abs());

        let strict = analyzer_with(AnalysisConfig {
            utility_degradation_threshold: 0.1,
            ..seeded_config()
        });
        let strict_results = strict
            .evaluate_robustness(&data, oracle, &config)
            .expect("robustness evaluation succeeds");
        assert!(
            !strict_results.failure_modes.is_empty(),
            "a 20% drop must breach a 10% threshold"
        );

        let lenient = analyzer_with(AnalysisConfig {
            utility_degradation_threshold: 0.5,
            ..seeded_config()
        });
        let lenient_results = lenient
            .evaluate_robustness(&data, oracle, &config)
            .expect("robustness evaluation succeeds");
        assert!(
            lenient_results.failure_modes.is_empty(),
            "a 20% drop must not breach a 50% threshold"
        );
        // The confidence level must have no influence on this threshold.
        let other_confidence = analyzer_with(AnalysisConfig {
            confidence_level: 0.99,
            utility_degradation_threshold: 0.5,
            ..seeded_config()
        });
        assert!(other_confidence
            .evaluate_robustness(&data, oracle, &config)
            .expect("robustness evaluation succeeds")
            .failure_modes
            .is_empty());
    }

    // ---------------------------------------------------------------- F100

    #[test]
    fn test_membership_bound_matches_its_derivation() {
        // eps = ln 3, delta = 0: (3 - 1) / (3 + 1) = 0.5
        let bound = binary_inference_advantage_bound(3.0_f64.ln(), 0.0);
        assert!((bound - 0.5).abs() < 1e-12, "bound={bound}");
        // delta enters as (e^eps - 1 + 2 delta) / (e^eps + 1)
        let with_delta = binary_inference_advantage_bound(3.0_f64.ln(), 0.1);
        assert!((with_delta - (2.0 + 0.2) / 4.0).abs() < 1e-12);
        // Saturates instead of producing NaN.
        assert!((binary_inference_advantage_bound(1e4, 0.0) - 1.0).abs() < 1e-12);
        assert!(binary_inference_advantage_bound(1e-9, 0.0) < 1e-6);
    }

    #[test]
    fn test_risk_composes_epsilon_over_iterations() {
        let data_free_config = make_config(0.01);
        let total = analyzer_with(AnalysisConfig {
            epsilon_semantics: EpsilonSemantics::Total,
            ..seeded_config()
        });
        let per_iteration = analyzer_with(AnalysisConfig {
            epsilon_semantics: EpsilonSemantics::PerIteration,
            ..seeded_config()
        });
        let total_risk = total
            .assess_privacy_risk(&data_free_config)
            .expect("risk assessment succeeds")
            .overall_risk_score;
        let composed_risk = per_iteration
            .assess_privacy_risk(&data_free_config)
            .expect("risk assessment succeeds")
            .overall_risk_score;
        assert!(
            composed_risk > total_risk,
            "composing 1000 steps must not score like a single query: {composed_risk} vs {total_risk}"
        );
        assert!(composed_risk > 0.99, "1000 x eps=0.01 is a large budget");
    }

    #[test]
    fn test_risk_does_not_overflow_for_huge_iteration_counts() {
        // (1 - delta).powi(iterations as i32) used to overflow to a negative
        // exponent and report a re-identification risk of 0.
        let analyzer = analyzer_with(AnalysisConfig {
            epsilon_semantics: EpsilonSemantics::PerIteration,
            ..seeded_config()
        });
        let mut config = make_config(1e-9);
        config.iterations = 3_000_000_000;
        config.delta = 1e-9;
        let assessment = analyzer
            .assess_privacy_risk(&config)
            .expect("risk assessment succeeds");
        let reidentification = assessment.risk_categories[&RiskCategory::ReIdentification];
        assert!(
            reidentification > 0.9,
            "composed delta over 3e9 steps must be close to 1, got {reidentification}"
        );
        assert!(assessment.overall_risk_score.is_finite());
    }

    #[test]
    fn test_risk_omits_categories_without_a_bound() {
        let analyzer = make_analyzer();
        let assessment = analyzer
            .assess_privacy_risk(&make_config(1.0))
            .expect("risk assessment succeeds");
        assert!(assessment
            .risk_categories
            .contains_key(&RiskCategory::MembershipInference));
        assert!(assessment
            .risk_categories
            .contains_key(&RiskCategory::AttributeInference));
        assert!(assessment
            .risk_categories
            .contains_key(&RiskCategory::ReIdentification));
        // These had invented coefficients and are now reported as not assessed.
        assert!(!assessment
            .risk_categories
            .contains_key(&RiskCategory::ModelInversion));
        assert!(!assessment
            .risk_categories
            .contains_key(&RiskCategory::PropertyInference));
        assert!(!assessment
            .risk_categories
            .contains_key(&RiskCategory::Reconstruction));
    }

    #[test]
    fn test_risk_never_claims_compliance() {
        let analyzer = make_analyzer();
        let mut tiny = make_config(1e-6);
        tiny.delta = 0.0;
        let assessment = analyzer
            .assess_privacy_risk(&tiny)
            .expect("risk assessment succeeds");
        assert!(assessment.overall_risk_score < 0.01);
        assert_eq!(
            assessment.compliance_status,
            ComplianceStatus::Unknown,
            "compliance cannot be decided from epsilon and delta alone"
        );
    }

    #[test]
    fn test_risk_rejects_invalid_configuration() {
        let analyzer = make_analyzer();
        assert!(analyzer.assess_privacy_risk(&make_config(0.0)).is_err());
        let mut negative_delta = make_config(1.0);
        negative_delta.delta = -1.0;
        assert!(analyzer.assess_privacy_risk(&negative_delta).is_err());
    }

    #[test]
    fn test_risk_evolution_only_exists_under_composition() {
        let total = make_analyzer();
        assert!(total
            .assess_privacy_risk(&make_config(1.0))
            .expect("risk assessment succeeds")
            .risk_evolution
            .is_empty());
        let composed = analyzer_with(AnalysisConfig {
            epsilon_semantics: EpsilonSemantics::PerIteration,
            ..seeded_config()
        });
        let mut config = make_config(1e-5);
        config.iterations = 10;
        let evolution = composed
            .assess_privacy_risk(&config)
            .expect("risk assessment succeeds")
            .risk_evolution;
        assert_eq!(evolution.len(), 5);
        assert_eq!(evolution[0].time_point, 10);
        assert!(evolution[4].risk_score >= evolution[0].risk_score);
    }

    // ---------------------------------------------------------------- F112

    #[test]
    fn test_statistical_tests_use_the_t_distribution() {
        let analyzer = make_analyzer();
        // Two samples of three observations each: the normal approximation
        // used to reject at 5% where the exact t test does not.
        let results: Vec<(f64, f64)> = vec![(0.1, 1.00), (0.2, 1.01), (0.3, 0.99)];
        let baseline: Vec<(f64, f64)> = vec![(0.1, 0.97), (0.2, 0.96), (0.3, 0.98)];
        let tests = analyzer
            .perform_statistical_tests(&results, &baseline)
            .expect("tests run");
        let welch = &tests.hypothesis_tests[0];
        // Welch: t = 3.674 on df = 4, so P(|T_4| >= 3.674) = 0.0213 (between
        // the tabulated t_{0.025,4} = 2.776 and t_{0.01,4} = 3.747).
        assert!(
            (welch.p_value - 0.0213).abs() < 0.001,
            "p_value={}",
            welch.p_value
        );
        // A normal approximation would have reported p = 0.00024 here, i.e.
        // significance at the 0.1% level for a three-point sample.
        assert!(
            welch.p_value > 0.01,
            "the exact t test must not reject at the 1% level"
        );
        assert!(welch.reject_null, "but it does reject at the 5% level");
    }

    #[test]
    fn test_statistical_tests_report_controlled_error_rates() {
        let analyzer = make_analyzer();
        let results: Vec<(f64, f64)> = (0..10)
            .map(|i| (i as f64 * 0.1, 1.0 + i as f64 * 0.01))
            .collect();
        let baseline: Vec<(f64, f64)> =
            (0..10).map(|i| (i as f64 * 0.1, i as f64 * 0.01)).collect();
        let tests = analyzer
            .perform_statistical_tests(&results, &baseline)
            .expect("tests run");
        let alpha = 1.0 - analyzer.config().confidence_level;
        let bonferroni = tests
            .multiple_comparison_corrections
            .iter()
            .find(|c| c.correction_method == CorrectionMethodApplied::Bonferroni)
            .expect("Bonferroni correction present");
        // The corrected bound is alpha, not the uncorrected 1 - 0.95^k.
        let fwer = bonferroni
            .controlled_family_wise_error_rate
            .expect("Bonferroni controls the FWER");
        assert!((fwer - alpha).abs() < 1e-12, "fwer={fwer}");
        assert!((fwer - (1.0 - 0.95_f64.powi(2))).abs() > 1e-6);

        let benjamini = tests
            .multiple_comparison_corrections
            .iter()
            .find(|c| c.correction_method == CorrectionMethodApplied::BenjaminiHochberg)
            .expect("Benjamini-Hochberg correction present");
        assert!(
            benjamini.controlled_family_wise_error_rate.is_none(),
            "Benjamini-Hochberg does not control the FWER"
        );
        let fdr = benjamini
            .controlled_false_discovery_rate
            .expect("Benjamini-Hochberg controls the FDR");
        assert!((fdr - alpha).abs() < 1e-12);
        // Adjusted p-values are ordered no larger than Bonferroni's.
        for (bh, bf) in benjamini
            .adjusted_p_values
            .iter()
            .zip(bonferroni.adjusted_p_values.iter())
        {
            assert!(bh <= &(bf + 1e-12));
        }
    }

    #[test]
    fn test_power_analysis_fields_are_mutually_consistent() {
        let analyzer = make_analyzer();
        let results: Vec<(f64, f64)> = (0..12)
            .map(|i| (i as f64 * 0.1, 1.0 + (i % 4) as f64 * 0.1))
            .collect();
        let baseline: Vec<(f64, f64)> = (0..12)
            .map(|i| (i as f64 * 0.1, 0.9 + (i % 4) as f64 * 0.1))
            .collect();
        let tests = analyzer
            .perform_statistical_tests(&results, &baseline)
            .expect("tests run");
        let power = &tests.power_analysis;
        let effect = tests.hypothesis_tests[0].effect_size.abs();
        let required = power
            .required_sample_size
            .expect("a non-zero effect has a required sample size");
        // The detectable effect at the required sample size is the observed
        // effect: the two fields assume the same power.
        let z_alpha = 1.959_963_984_540_054_f64;
        let z_beta = 0.841_621_233_572_914_3_f64;
        let mde_at_required = (z_alpha + z_beta) * (2.0 / required as f64).sqrt();
        assert!(
            (mde_at_required - effect).abs() < 0.05 * effect,
            "mde={mde_at_required}, effect={effect}, required={required}"
        );
        // The reported minimum detectable effect belongs to the *observed*
        // sample size and uses the same two critical values.
        let observed_n = 12.0_f64;
        assert!(
            (power.minimum_detectable_effect - (z_alpha + z_beta) * (2.0 / observed_n).sqrt())
                .abs()
                < 1e-9,
            "minimum_detectable_effect={}",
            power.minimum_detectable_effect
        );
        assert!(power.statistical_power >= 0.0 && power.statistical_power <= 1.0);
        assert_eq!(power.power_curve.len(), 20);
    }

    #[test]
    fn test_statistical_tests_reject_tiny_samples() {
        let analyzer = make_analyzer();
        let single = vec![(1.0_f64, 0.5_f64)];
        let pair = vec![(1.0_f64, 0.5_f64), (2.0, 0.6)];
        assert!(analyzer.perform_statistical_tests(&single, &pair).is_err());
        assert!(analyzer.perform_statistical_tests(&pair, &single).is_err());
        // With three observations per sample the correlation test is skipped
        // rather than run with an undefined standard error.
        let triple_a = vec![(1.0_f64, 0.5_f64), (2.0, 0.6), (3.0, 0.7)];
        let triple_b = vec![(1.0_f64, 0.4_f64), (2.0, 0.5), (3.0, 0.6)];
        let tests = analyzer
            .perform_statistical_tests(&triple_a, &triple_b)
            .expect("tests run");
        assert_eq!(tests.hypothesis_tests.len(), 1);
    }

    #[test]
    fn test_identical_samples_are_not_significant() {
        let analyzer = make_analyzer();
        let data: Vec<(f64, f64)> = (0..10).map(|i| (i as f64 * 0.1, 0.5)).collect();
        let tests = analyzer
            .perform_statistical_tests(&data, &data)
            .expect("tests run");
        assert!(tests.hypothesis_tests[0].p_value > 0.5);
        assert!(!tests.hypothesis_tests[0].reject_null);
    }

    // ----------------------------------------------------- budget allocation

    #[test]
    fn test_budget_allocation_sums_to_the_budget() {
        let analyzer = make_analyzer();
        let budget = make_budget(1.0);
        let concave = |eps: f64| (1.0 + eps).ln();
        let allocation = analyzer
            .optimize_budget_allocation(&budget, 10, 0.0, &concave)
            .expect("allocation succeeds");
        assert_eq!(allocation.per_iteration_allocation.len(), 10);
        let sum: f64 = allocation.per_iteration_allocation.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
        assert!(allocation.allocation_imbalance >= 0.0);
    }

    #[test]
    fn test_budget_allocation_ranks_by_the_callers_utility_model() {
        let analyzer = make_analyzer();
        let concave = |eps: f64| (1.0 + eps).ln();
        let convex = |eps: f64| eps * eps;
        let concave_best = analyzer
            .budget_allocation_candidates(1.0, 8, &concave)
            .expect("candidates")
            .remove(0);
        let convex_best = analyzer
            .budget_allocation_candidates(1.0, 8, &convex)
            .expect("candidates")
            .remove(0);
        // A concave model prefers the uniform split; a convex one prefers a
        // skewed split. The ranking therefore has to depend on the model.
        assert!(concave_best.allocation_imbalance < convex_best.allocation_imbalance);
    }

    #[test]
    fn test_budget_allocation_rejects_unreachable_threshold() {
        let analyzer = make_analyzer();
        let budget = make_budget(1.0);
        let concave = |eps: f64| (1.0 + eps).ln();
        let error = analyzer
            .optimize_budget_allocation(&budget, 10, 10.0, &concave)
            .err()
            .expect("an unreachable threshold must be an error");
        assert!(matches!(error, OptimError::OptimizationError(_)));
    }

    #[test]
    fn test_budget_allocation_validates_inputs() {
        let analyzer = make_analyzer();
        let concave = |eps: f64| (1.0 + eps).ln();
        assert!(analyzer
            .optimize_budget_allocation(&make_budget(1.0), 0, 0.0, &concave)
            .is_err());
        assert!(analyzer
            .optimize_budget_allocation(&make_budget(0.0), 5, 0.0, &concave)
            .is_err());
    }

    // --------------------------------------------------- degradation predictions

    #[test]
    fn test_predict_degradation_linear() {
        let analyzer = make_analyzer();
        let historical: Vec<(f64, f64)> = (1..=4).map(|i| (i as f64, 0.1 * i as f64)).collect();
        let predictions = analyzer
            .predict_utility_degradation(&[5.0_f64], &historical)
            .expect("prediction succeeds");
        assert_eq!(predictions.len(), 1);
        assert!((predictions[0].predicted_utility_loss - 0.5).abs() < 1e-6);
        assert!(predictions[0].model_accuracy > 0.99);
        let (lo, hi) = predictions[0].confidence_interval;
        assert!(lo <= predictions[0].predicted_utility_loss);
        assert!(hi >= predictions[0].predicted_utility_loss);
    }

    #[test]
    fn test_predict_degradation_validates_history() {
        let analyzer = make_analyzer();
        assert!(analyzer
            .predict_utility_degradation(&[1.0_f64], &[])
            .is_err());
        assert!(analyzer
            .predict_utility_degradation(&[], &[(1.0_f64, 0.5_f64), (2.0, 0.8)])
            .expect("empty query list is not an error")
            .is_empty());
        // All observations at the same epsilon cannot support a regression.
        let degenerate = vec![(1.0_f64, 0.2_f64), (1.0, 0.3), (1.0, 0.4)];
        assert!(analyzer
            .predict_utility_degradation(&[1.0_f64], &degenerate)
            .is_err());
    }

    // ---------------------------------------------------------------- F111

    #[test]
    fn test_metadata_is_measured_not_fabricated() {
        let analyzer = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let results = analyzer.analyze(&data, utility_oracle).expect("analysis");
        let metadata = &results.metadata;
        assert_eq!(metadata.analysis_version, env!("CARGO_PKG_VERSION"));
        assert_ne!(metadata.configuration_hash, "abc123");
        assert_eq!(metadata.configuration_hash.len(), 16);
        assert!(metadata.computational_resources.peak_memory_bytes.is_none());
        assert!(metadata.computational_resources.gpu_usage.is_none());
        assert!(metadata.computational_resources.wall_time.as_nanos() > 0);
        // The reported seed is the one that actually drove the run.
        assert_eq!(
            metadata.reproducibility_info.random_seed,
            analyzer.seed(),
            "metadata must report the seed the analysis used"
        );
        assert_eq!(metadata.reproducibility_info.random_seed, 20_240_517);
        assert!(metadata
            .reproducibility_info
            .software_versions
            .values()
            .any(|v| v == env!("CARGO_PKG_VERSION")));
        assert!(metadata
            .reproducibility_info
            .hardware_info
            .contains(std::env::consts::ARCH));
    }

    #[test]
    fn test_configuration_hash_changes_with_the_configuration() {
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let first = make_analyzer()
            .analyze(&data, utility_oracle)
            .expect("analysis");
        let second = analyzer_with(AnalysisConfig {
            pareto_resolution: 40,
            ..seeded_config()
        })
        .analyze(&data, utility_oracle)
        .expect("analysis");
        assert_ne!(
            first.metadata.configuration_hash,
            second.metadata.configuration_hash
        );
    }

    #[test]
    fn test_seeded_runs_are_reproducible() {
        let mut space = PrivacyParameterSpace::default();
        space.epsilon_range =
            ParameterRange::new(0.1, 10.0, 20, SamplingStrategy::Random).expect("valid range");
        let config = AnalysisConfig {
            privacy_parameters: space,
            random_seed: Some(7),
            pareto_resolution: 30,
            ..AnalysisConfig::default()
        };
        let first = analyzer_with(config.clone())
            .generate_privacy_configurations()
            .expect("configurations");
        let second = analyzer_with(config)
            .generate_privacy_configurations()
            .expect("configurations");
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a.epsilon - b.epsilon).abs() < 1e-15);
        }
    }

    #[test]
    fn test_unseeded_analyzer_reports_the_seed_it_drew() {
        let analyzer = PrivacyUtilityAnalyzer::<f64>::new(AnalysisConfig::default())
            .expect("default configuration is valid");
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let results = analyzer.analyze(&data, utility_oracle).expect("analysis");
        assert_eq!(
            results.metadata.reproducibility_info.random_seed,
            analyzer.seed()
        );
    }

    // ---------------------------------------------------------------- F101

    #[test]
    fn test_configuration_is_validated_on_construction() {
        assert!(PrivacyUtilityAnalyzer::<f64>::new(AnalysisConfig {
            confidence_level: 1.5,
            ..AnalysisConfig::default()
        })
        .is_err());
        assert!(PrivacyUtilityAnalyzer::<f64>::new(AnalysisConfig {
            target_power: 0.0,
            ..AnalysisConfig::default()
        })
        .is_err());
        assert!(PrivacyUtilityAnalyzer::<f64>::new(AnalysisConfig {
            pareto_resolution: 0,
            ..AnalysisConfig::default()
        })
        .is_err());
        assert!(PrivacyUtilityAnalyzer::<f64>::new(AnalysisConfig {
            utility_degradation_threshold: 0.0,
            ..AnalysisConfig::default()
        })
        .is_err());
        assert!(PrivacyUtilityAnalyzer::<f64>::new(AnalysisConfig {
            monte_carlo_samples: 0,
            ..AnalysisConfig::default()
        })
        .is_err());
    }

    #[test]
    fn test_privacy_cost_follows_the_configured_semantics() {
        let config = make_config(0.01);
        let total = make_analyzer()
            .compute_privacy_cost(&config)
            .expect("cost computed");
        let composed = analyzer_with(AnalysisConfig {
            epsilon_semantics: EpsilonSemantics::PerIteration,
            ..seeded_config()
        })
        .compute_privacy_cost(&config)
        .expect("cost computed");
        assert!((total - 0.01).abs() < 1e-12);
        assert!((composed - 10.0).abs() < 1e-9, "composed={composed}");
    }
}
