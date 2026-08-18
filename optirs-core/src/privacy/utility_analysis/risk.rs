//! Privacy risk bounds and statistical significance testing.

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::stats;
use super::types::{
    to_f64, to_t, CorrectionMethodApplied, MultipleComparisonCorrection, PowerAnalysis,
    PrivacyUtilityAnalyzer, RiskCategory, RiskEvolution,
};
use super::types_3::{
    ComplianceStatus, EpsilonSemantics, HypothesisTestResult, PrivacyConfiguration,
    PrivacyRiskAssessment, RiskTrend, StatisticalTestResults,
};

/// Maximum advantage `TPR - FPR` of any adversary that decides a binary
/// question about a single record from the output of an `(eps, delta)`-DP
/// mechanism.
///
/// # Derivation
/// For neighbouring datasets `D` (record present) and `D'` (record absent) and
/// any decision region `S`, differential privacy gives
///
/// * `P[M(D) in S] <= e^eps P[M(D') in S] + delta`, i.e. `TPR <= e^eps FPR + delta`
/// * applying the same inequality to the complement of `S`:
///   `1 - FPR <= e^eps (1 - TPR) + delta`
///
/// Maximizing `TPR - FPR` over the intersection of those two half planes is
/// attained where both hold with equality, at
/// `FPR = (1 - delta)/(e^eps + 1)`, giving
/// `advantage = (e^eps - 1 + 2 delta) / (e^eps + 1)`.
///
/// # Assumptions
/// Record-level adjacency, an adversary that only sees the mechanism output,
/// and a balanced prior over the two hypotheses. By post-processing the bound
/// carries over unchanged to *any binary* inference about a single record
/// (membership, or the value of one binary attribute).
///
/// The expression is evaluated in the `e^{-eps}` form so that large `eps`
/// saturates at `1.0` instead of producing `inf/inf`.
pub(super) fn binary_inference_advantage_bound(epsilon: f64, delta: f64) -> f64 {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return 1.0;
    }
    let exp_neg = (-epsilon).exp();
    ((1.0 - exp_neg + 2.0 * delta * exp_neg) / (1.0 + exp_neg)).clamp(0.0, 1.0)
}

impl<T: Float + Debug + Send + Sync + 'static> PrivacyUtilityAnalyzer<T> {
    /// Assess the privacy risk implied by a configuration.
    ///
    /// The `(epsilon, delta)` of the configuration are first **composed** into
    /// the guarantee of the whole run according to
    /// [`AnalysisConfig::epsilon_semantics`](super::types_3::AnalysisConfig),
    /// so a 1000-step run is not scored as a single query.
    ///
    /// Only risks that follow from the composed guarantee are reported:
    ///
    /// * [`RiskCategory::MembershipInference`] and
    ///   [`RiskCategory::AttributeInference`]:
    ///   `binary_inference_advantage_bound`, derived from the DP definition.
    /// * [`RiskCategory::ReIdentification`]: the composed `delta`, i.e. an
    ///   upper bound on the probability that the epsilon guarantee fails.
    ///
    /// Model inversion, dataset-level property inference and record
    /// reconstruction have no bound that follows from `(epsilon, delta)`
    /// alone, so they are **absent** from `risk_categories` rather than being
    /// reported with an invented coefficient.
    ///
    /// The compliance status is always [`ComplianceStatus::Unknown`]: whether
    /// a deployment complies with a given regulation cannot be decided from
    /// privacy parameters alone.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for an invalid configuration.
    pub fn assess_privacy_risk(
        &self,
        config: &PrivacyConfiguration<T>,
    ) -> Result<PrivacyRiskAssessment<T>> {
        let (eps_total, delta_total) = self.composed_budget(config)?;
        let membership = binary_inference_advantage_bound(eps_total, delta_total);
        let reidentification = delta_total.clamp(0.0, 1.0);

        let mut risk_categories: HashMap<RiskCategory, T> = HashMap::new();
        risk_categories.insert(RiskCategory::MembershipInference, to_t(membership)?);
        risk_categories.insert(RiskCategory::AttributeInference, to_t(membership)?);
        risk_categories.insert(RiskCategory::ReIdentification, to_t(reidentification)?);
        let overall = membership.max(reidentification);

        let mut mitigation_recommendations = Vec::new();
        if membership > 0.5 {
            mitigation_recommendations.push(format!(
                "Composed epsilon of {eps_total:.3} admits a membership-inference advantage of up to {:.2}; reduce epsilon or the number of composed iterations",
                membership
            ));
        }
        if reidentification > 1e-5 {
            mitigation_recommendations.push(format!(
                "Composed delta of {delta_total:.3e} exceeds the customary 1e-5 ceiling; reduce delta or compose fewer steps"
            ));
        }
        if matches!(
            self.config.epsilon_semantics,
            EpsilonSemantics::PerIteration
        ) {
            mitigation_recommendations.push(
                "Composition is basic (sequential); a Renyi or moments accountant would report a tighter total epsilon"
                    .to_string(),
            );
        }
        if mitigation_recommendations.is_empty() {
            mitigation_recommendations.push(format!(
                "Composed guarantee (eps={eps_total:.3}, delta={delta_total:.3e}) bounds any single-record binary inference advantage at {membership:.3}"
            ));
        }

        // A composition trajectory only exists when the configuration states a
        // per-iteration budget; a stated total budget does not grow with time.
        let risk_evolution = match self.config.epsilon_semantics {
            EpsilonSemantics::Total => Vec::new(),
            EpsilonSemantics::PerIteration => {
                let eps_step = to_f64(config.epsilon)?;
                let delta_step = to_f64(config.delta)?;
                let mut evolution = Vec::with_capacity(5);
                let mut previous = membership;
                for multiple in 1..=5_usize {
                    let iterations = config.iterations.saturating_mul(multiple);
                    let eps = eps_step * iterations as f64;
                    let delta = 1.0 - (1.0 - delta_step).powf(iterations as f64);
                    let bound = binary_inference_advantage_bound(eps, delta.clamp(0.0, 1.0));
                    let risk_trend = if bound > previous * 1.01 {
                        RiskTrend::Increasing
                    } else {
                        RiskTrend::Stable
                    };
                    previous = bound;
                    evolution.push(RiskEvolution {
                        time_point: iterations,
                        risk_score: to_t(bound)?,
                        contributing_factors: vec![
                            "sequential composition of epsilon".to_string(),
                            "accumulated delta".to_string(),
                        ],
                        risk_trend,
                    });
                }
                evolution
            }
        };

        Ok(PrivacyRiskAssessment {
            overall_risk_score: to_t(overall)?,
            risk_categories,
            mitigation_recommendations,
            compliance_status: ComplianceStatus::Unknown,
            risk_evolution,
        })
    }

    /// Statistical comparison of two sets of `(privacy, utility)` observations.
    ///
    /// Runs a Welch two-sample t-test on the utilities, whose p-value comes
    /// from the exact Student t distribution (regularized incomplete beta),
    /// not from a normal approximation. When both samples have at least four
    /// observations, a Fisher z-test on the privacy-utility correlations is
    /// added; with fewer observations that test's standard error is not
    /// defined and the test is omitted rather than fudged.
    ///
    /// The p-values are then corrected with Bonferroni, Holm-Bonferroni and
    /// Benjamini-Hochberg. Each correction reports the level at which it
    /// *controls* the family-wise error rate / false discovery rate - a
    /// guarantee, not an estimate.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] when either sample has fewer
    /// than two observations, and propagates numerical failures.
    pub fn perform_statistical_tests(
        &self,
        results: &[(T, T)],
        baseline: &[(T, T)],
    ) -> Result<StatisticalTestResults<T>> {
        if results.len() < 2 {
            return Err(OptimError::InvalidParameter(
                "results must have at least 2 elements".to_string(),
            ));
        }
        if baseline.len() < 2 {
            return Err(OptimError::InvalidParameter(
                "baseline must have at least 2 elements".to_string(),
            ));
        }
        let alpha = self.alpha();
        let z_alpha = self.z_alpha()?;
        let z_beta = self.z_beta()?;

        let mut results_privacy = Vec::with_capacity(results.len());
        let mut results_utility = Vec::with_capacity(results.len());
        for (privacy, utility) in results {
            results_privacy.push(to_f64(*privacy)?);
            results_utility.push(to_f64(*utility)?);
        }
        let mut baseline_privacy = Vec::with_capacity(baseline.len());
        let mut baseline_utility = Vec::with_capacity(baseline.len());
        for (privacy, utility) in baseline {
            baseline_privacy.push(to_f64(*privacy)?);
            baseline_utility.push(to_f64(*utility)?);
        }

        let (mean1, var1) = stats::mean_var(&results_utility);
        let (mean2, var2) = stats::mean_var(&baseline_utility);
        let n1 = results_utility.len() as f64;
        let n2 = baseline_utility.len() as f64;
        let standard_error = (var1 / n1 + var2 / n2).sqrt();
        let t_statistic = if standard_error > 1e-12 {
            (mean1 - mean2) / standard_error
        } else {
            0.0
        };
        // Welch-Satterthwaite degrees of freedom.
        let df_numerator = (var1 / n1 + var2 / n2).powi(2);
        let df_denominator =
            (var1 / n1).powi(2) / (n1 - 1.0).max(1.0) + (var2 / n2).powi(2) / (n2 - 1.0).max(1.0);
        let df = if df_denominator > 1e-12 {
            df_numerator / df_denominator
        } else {
            (n1 + n2 - 2.0).max(1.0)
        };
        let p_welch = stats::student_t_two_sided_p(t_statistic, df.max(1.0))?;
        let pooled_std = ((var1 * (n1 - 1.0) + var2 * (n2 - 1.0)) / (n1 + n2 - 2.0).max(1.0))
            .sqrt()
            .max(1e-12);
        let cohen_d = (mean1 - mean2) / pooled_std;

        let mut hypothesis_tests: Vec<HypothesisTestResult<T>> = Vec::new();
        let mut raw_p_values = Vec::new();
        let mut effect_sizes = Vec::new();
        hypothesis_tests.push(HypothesisTestResult {
            test_name: "Welch t-test (utility)".to_string(),
            test_statistic: to_t(t_statistic)?,
            p_value: to_t(p_welch)?,
            significance_level: to_t(alpha)?,
            reject_null: p_welch < alpha,
            effect_size: to_t(cohen_d)?,
        });
        raw_p_values.push(p_welch);
        effect_sizes.push(to_t(cohen_d)?);

        // Fisher z-test on the privacy-utility correlations. The standard
        // error of a Fisher-transformed correlation needs n > 3 per sample.
        if results.len() >= 4 && baseline.len() >= 4 {
            let r1 = stats::pearson_correlation(&results_privacy, &results_utility);
            let r2 = stats::pearson_correlation(&baseline_privacy, &baseline_utility);
            let z1 = r1.clamp(-0.9999, 0.9999).atanh();
            let z2 = r2.clamp(-0.9999, 0.9999).atanh();
            let se_z = (1.0 / (results.len() as f64 - 3.0) + 1.0 / (baseline.len() as f64 - 3.0))
                .sqrt()
                .max(1e-12);
            let z_statistic = (z1 - z2) / se_z;
            let p_correlation =
                (2.0 * (1.0 - stats::normal_cdf(z_statistic.abs()))).clamp(0.0, 1.0);
            hypothesis_tests.push(HypothesisTestResult {
                test_name: "Fisher z-test (privacy-utility correlation)".to_string(),
                test_statistic: to_t(z_statistic)?,
                p_value: to_t(p_correlation)?,
                significance_level: to_t(alpha)?,
                reject_null: p_correlation < alpha,
                effect_size: to_t((r1 - r2).abs())?,
            });
            raw_p_values.push(p_correlation);
            effect_sizes.push(to_t((r1 - r2).abs())?);
        }

        // Power analysis. The three reported quantities are mutually
        // consistent by construction: they all use the same `z_alpha` and
        // `z_beta`, so the effect detectable at `required_sample_size` is the
        // observed effect and the power there is `target_power`.
        let per_group_n = (n1 + n2) / 2.0;
        let power_analysis = PowerAnalysis {
            statistical_power: to_t(stats::two_sample_power(cohen_d, per_group_n, z_alpha))?,
            required_sample_size: stats::required_per_group_sample_size(cohen_d, z_alpha, z_beta),
            minimum_detectable_effect: to_t(stats::minimum_detectable_effect(
                per_group_n,
                z_alpha,
                z_beta,
            ))?,
            power_curve: (1..=20_usize)
                .map(|k| {
                    let effect = k as f64 * 0.05;
                    Ok((
                        to_t(effect)?,
                        to_t(stats::two_sample_power(effect, per_group_n, z_alpha))?,
                    ))
                })
                .collect::<Result<Vec<(T, T)>>>()?,
        };

        let alpha_t = to_t(alpha)?;
        let to_t_vec = |values: Vec<f64>| -> Result<Vec<T>> {
            values.into_iter().map(to_t).collect::<Result<Vec<T>>>()
        };
        let multiple_comparison_corrections = vec![
            MultipleComparisonCorrection {
                correction_method: CorrectionMethodApplied::Bonferroni,
                adjusted_p_values: to_t_vec(stats::bonferroni_adjust(&raw_p_values))?,
                controlled_family_wise_error_rate: Some(alpha_t),
                controlled_false_discovery_rate: Some(alpha_t),
            },
            MultipleComparisonCorrection {
                correction_method: CorrectionMethodApplied::HolmBonferroni,
                adjusted_p_values: to_t_vec(stats::holm_adjust(&raw_p_values))?,
                controlled_family_wise_error_rate: Some(alpha_t),
                controlled_false_discovery_rate: Some(alpha_t),
            },
            MultipleComparisonCorrection {
                correction_method: CorrectionMethodApplied::BenjaminiHochberg,
                adjusted_p_values: to_t_vec(stats::benjamini_hochberg_adjust(&raw_p_values))?,
                // Benjamini-Hochberg controls the FDR, not the FWER.
                controlled_family_wise_error_rate: None,
                controlled_false_discovery_rate: Some(alpha_t),
            },
        ];

        Ok(StatisticalTestResults {
            hypothesis_tests,
            significance_levels: vec![alpha_t],
            effect_sizes,
            power_analysis,
            multiple_comparison_corrections,
        })
    }
}
