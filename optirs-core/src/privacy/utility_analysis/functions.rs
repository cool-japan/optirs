//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::privacy::{DifferentialPrivacyConfig, NoiseMechanism, PrivacyBudget};

use super::types::{PrivacyUtilityAnalyzer, RiskCategory};
use super::types_3::{AnalysisConfig, CorrectionMethod, PrivacyConfiguration};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    fn make_analyzer() -> PrivacyUtilityAnalyzer<f64> {
        PrivacyUtilityAnalyzer::new(AnalysisConfig::default())
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
            noise_mechanism: crate::privacy::NoiseMechanism::Gaussian,
        }
    }
    #[test]
    fn test_optimize_budget_allocation_uniform() {
        let a = make_analyzer();
        let budget = make_budget(1.0);
        let result = a.optimize_budget_allocation(&budget, 10, 0.0).unwrap();
        assert_eq!(result.per_iteration_allocation.len(), 10);
        let sum: f64 = result.per_iteration_allocation.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
    }
    #[test]
    fn test_optimize_budget_allocation_zero_iterations() {
        let a = make_analyzer();
        let budget = make_budget(1.0);
        assert!(a.optimize_budget_allocation(& budget, 0, 0.0).is_err());
    }
    #[test]
    fn test_optimize_budget_allocation_best_utility() {
        let a = make_analyzer();
        let budget = make_budget(2.0);
        let result = a.optimize_budget_allocation(&budget, 5, 0.0).unwrap();
        assert!(result.expected_utility > 0.0, "expected_utility should be positive");
    }
    #[test]
    fn test_evaluate_robustness_constant_model() {
        let a = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = make_config(1.0);
        let result = a.evaluate_robustness(&data, |_, c| Ok(c.epsilon), &cfg).unwrap();
        assert!(
            result.robustness_score > 0.5, "constant model should be robust, score={}",
            result.robustness_score
        );
    }
    #[test]
    fn test_evaluate_robustness_brittle_model() {
        let a = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = make_config(1.0);
        let result = a
            .evaluate_robustness(
                &data,
                |_, c| {
                    let eps = c.epsilon;
                    if eps > 1.0 { Ok(0.2) } else { Ok(1.0) }
                },
                &cfg,
            )
            .unwrap();
        assert!(
            result.robustness_score < 0.5,
            "brittle model should have low robustness, score={}", result.robustness_score
        );
    }
    #[test]
    fn test_evaluate_robustness_propagates_error() {
        let a = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = make_config(1.0);
        let result = a
            .evaluate_robustness(
                &data,
                |_, _| Err(
                    crate::error::OptimError::ComputationError("test error".to_string()),
                ),
                &cfg,
            );
        assert!(result.is_err());
    }
    #[test]
    fn test_predict_degradation_linear() {
        let a = make_analyzer();
        let historical: Vec<(f64, f64)> = (1..=4)
            .map(|i| (i as f64, 2.0 * i as f64))
            .collect();
        let params = vec![5.0_f64];
        let preds = a.predict_utility_degradation(&params, &historical).unwrap();
        assert_eq!(preds.len(), 1);
        let pred = preds[0].predicted_utility_loss;
        assert!((pred - 1.0_f64).abs() < 0.2, "predicted loss for x=5 in [0,1]: {pred}");
    }
    #[test]
    fn test_predict_degradation_quadratic() {
        let a = make_analyzer();
        let historical: Vec<(f64, f64)> = (1..=8)
            .map(|i| (i as f64 * 0.1, (i as f64 * 0.1).powi(2)))
            .collect();
        let params = vec![0.5_f64];
        let preds = a.predict_utility_degradation(&params, &historical).unwrap();
        assert_eq!(preds.len(), 1);
        let acc = preds[0].model_accuracy;
        assert!(acc > 0.9, "R^2 should be close to 1.0, got {acc}");
    }
    #[test]
    fn test_predict_degradation_empty_history() {
        let a = make_analyzer();
        let result = a.predict_utility_degradation(&[1.0_f64], &[]);
        assert!(result.is_err());
    }
    #[test]
    fn test_predict_degradation_empty_params() {
        let a = make_analyzer();
        let historical = vec![(1.0_f64, 0.5_f64), (2.0, 0.8)];
        let result = a.predict_utility_degradation(&[], &historical).unwrap();
        assert!(result.is_empty());
    }
    #[test]
    fn test_assess_risk_low_epsilon() {
        let a = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = make_config(0.1);
        let result = a.assess_privacy_risk(&data, &cfg).unwrap();
        let score = result.overall_risk_score;
        assert!(score < 0.5, "low epsilon should yield low risk, got {score}");
    }
    #[test]
    fn test_assess_risk_high_epsilon() {
        let a = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = make_config(10.0);
        let result = a.assess_privacy_risk(&data, &cfg).unwrap();
        let score = result.overall_risk_score;
        assert!(score > 0.5, "high epsilon should yield high risk, got {score}");
    }
    #[test]
    fn test_assess_risk_invalid_epsilon() {
        let a = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = make_config(0.0);
        assert!(a.assess_privacy_risk(& data, & cfg).is_err());
    }
    #[test]
    fn test_assess_risk_all_categories_present() {
        let a = make_analyzer();
        let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = make_config(1.0);
        let result = a.assess_privacy_risk(&data, &cfg).unwrap();
        assert!(
            result.risk_categories.contains_key(& RiskCategory::MembershipInference)
        );
        assert!(result.risk_categories.contains_key(& RiskCategory::AttributeInference));
        assert!(result.risk_categories.contains_key(& RiskCategory::ModelInversion));
        assert!(result.risk_categories.contains_key(& RiskCategory::PropertyInference));
        assert!(result.risk_categories.contains_key(& RiskCategory::Reconstruction));
        assert!(result.risk_categories.contains_key(& RiskCategory::ReIdentification));
    }
    #[test]
    fn test_stat_tests_significant_difference() {
        let a = make_analyzer();
        let results: Vec<(f64, f64)> = (0..10)
            .map(|i| (i as f64 * 0.1, 1.0 + i as f64 * 0.001))
            .collect();
        let baseline: Vec<(f64, f64)> = (0..10)
            .map(|i| (i as f64 * 0.1, i as f64 * 0.001))
            .collect();
        let st = a.perform_statistical_tests(&results, &baseline).unwrap();
        let welch = &st.hypothesis_tests[0];
        assert!(welch.reject_null, "should reject null for large difference");
    }
    #[test]
    fn test_stat_tests_same_data() {
        let a = make_analyzer();
        let data: Vec<(f64, f64)> = (0..10).map(|i| (i as f64 * 0.1, 0.5)).collect();
        let st = a.perform_statistical_tests(&data, &data).unwrap();
        let p = st.hypothesis_tests[0].p_value;
        assert!(p > 0.5, "p-value should be large for identical data, got {p}");
    }
    #[test]
    fn test_stat_tests_too_few_points() {
        let a = make_analyzer();
        let one_point = vec![(1.0_f64, 0.5_f64)];
        let good = vec![(1.0_f64, 0.5_f64), (2.0, 0.6)];
        assert!(a.perform_statistical_tests(& one_point, & good).is_err());
        assert!(a.perform_statistical_tests(& good, & one_point).is_err());
    }
    #[test]
    fn test_stat_tests_bonferroni_present() {
        let a = make_analyzer();
        let data: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, i as f64 * 0.1)).collect();
        let st = a.perform_statistical_tests(&data, &data).unwrap();
        let has_bonf = st
            .multiple_comparison_corrections
            .iter()
            .any(|c| matches!(c.correction_method, CorrectionMethod::Bonferroni));
        assert!(has_bonf, "should have Bonferroni correction");
    }
    #[test]
    fn test_normal_cdf_known() {
        let a = make_analyzer();
        let cdf0 = a.normal_cdf_f64(0.0);
        assert!((cdf0 - 0.5).abs() < 1e-6, "CDF(0)={cdf0}");
        let cdf196 = a.normal_cdf_f64(1.96);
        assert!((cdf196 - 0.975).abs() < 0.005, "CDF(1.96)={cdf196}");
    }
    #[test]
    fn test_polyfit_linear() {
        let a = make_analyzer();
        let xs = vec![0.0_f64, 1.0, 2.0];
        let ys = vec![1.0_f64, 3.0, 5.0];
        let coeffs = a.polyfit_f64(&xs, &ys, 1).unwrap();
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 1.0).abs() < 1e-8, "c0={}", coeffs[0]);
        assert!((coeffs[1] - 2.0).abs() < 1e-8, "c1={}", coeffs[1]);
    }
    #[test]
    fn test_polyfit_quadratic() {
        let a = make_analyzer();
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
        let ys = xs.iter().map(|&x| 1.0 + 2.0 * x + 3.0 * x * x).collect::<Vec<_>>();
        let coeffs = a.polyfit_f64(&xs, &ys, 2).unwrap();
        assert_eq!(coeffs.len(), 3);
        assert!((coeffs[0] - 1.0).abs() < 1e-6, "c0={}", coeffs[0]);
        assert!((coeffs[1] - 2.0).abs() < 1e-6, "c1={}", coeffs[1]);
        assert!((coeffs[2] - 3.0).abs() < 1e-6, "c2={}", coeffs[2]);
    }
}
