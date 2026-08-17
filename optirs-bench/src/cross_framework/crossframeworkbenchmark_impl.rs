//! # `CrossFrameworkBenchmark` - calculate_cohens_d_group Methods
//!
//! This module contains method implementations for `CrossFrameworkBenchmark`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::regression_tester::distributions::{f_distribution_sf, sample_variance, t_quantile};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{AnovaResult, ConfidenceInterval};

use super::crossframeworkbenchmark_type::CrossFrameworkBenchmark;

impl<A: Float + Debug + Send + Sync> CrossFrameworkBenchmark<A> {
    /// Cohen's d effect size using the pooled standard deviation.
    ///
    /// Returns `0.0` when either sample has fewer than two observations or the
    /// pooled standard deviation is zero (no scale against which to measure an
    /// effect).
    pub(super) fn calculate_cohens_d(&self, sample1: &[f64], sample2: &[f64]) -> f64 {
        let (Some(var1), Some(var2)) = (sample_variance(sample1), sample_variance(sample2)) else {
            return 0.0;
        };
        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;
        if n1 < 2.0 || n2 < 2.0 {
            return 0.0;
        }

        let mean1 = sample1.iter().sum::<f64>() / n1;
        let mean2 = sample2.iter().sum::<f64>() / n2;

        // Pooled standard deviation weighted by degrees of freedom.
        let pooled_variance = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        let pooled_std = pooled_variance.sqrt();
        if !pooled_std.is_finite() || pooled_std <= 0.0 {
            return 0.0;
        }

        (mean1 - mean2) / pooled_std
    }

    /// Confidence interval for the mean, honouring the requested level.
    ///
    /// The critical value is the Student's t quantile for `n - 1` degrees of
    /// freedom at the requested level - not a hardcoded `1.96`, which is only
    /// correct for a 95% interval with infinitely many samples.
    pub(super) fn calculate_confidence_interval(
        &self,
        values: &[f64],
        confidence_level: f64,
    ) -> ConfidenceInterval<A> {
        let degenerate = |value: f64| ConfidenceInterval {
            lower: A::from(value).unwrap_or_else(A::zero),
            upper: A::from(value).unwrap_or_else(A::zero),
            confidence_level,
        };

        if values.is_empty() {
            return degenerate(0.0);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let Some(std_dev) = sample_variance(values).map(f64::sqrt) else {
            // A single observation supports no interval.
            return degenerate(mean);
        };
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return degenerate(mean);
        }

        let n = values.len() as f64;
        let standard_error = std_dev / n.sqrt();
        let alpha = 1.0 - confidence_level;
        let Some(critical_value) = t_quantile(1.0 - alpha / 2.0, n - 1.0) else {
            return degenerate(mean);
        };
        let margin_of_error = critical_value * standard_error;

        ConfidenceInterval {
            lower: A::from(mean - margin_of_error).unwrap_or_else(A::zero),
            upper: A::from(mean + margin_of_error).unwrap_or_else(A::zero),
            confidence_level,
        }
    }

    /// One-way ANOVA across the supplied groups.
    ///
    /// The p-value is the exact upper-tail probability of the F distribution,
    /// `P(F_{df_between, df_within} > F)`, evaluated through the regularized
    /// incomplete beta function. The previous two-valued lookup ("0.01 if
    /// F > 3 else 0.1") reported significance that had nothing to do with the
    /// degrees of freedom of the design.
    ///
    /// Groups with fewer than one observation are dropped; the analysis is
    /// undefined (and returns `p = 1`) when fewer than two groups survive or
    /// when there is no residual degrees of freedom.
    pub(super) fn perform_anova(&self, groups: &[Vec<f64>]) -> AnovaResult<A> {
        let undefined = AnovaResult {
            f_statistic: 0.0,
            p_value: 1.0,
            between_ss: A::zero(),
            within_ss: A::zero(),
            total_ss: A::zero(),
            df_between: 0,
            df_within: 0,
        };

        let usable: Vec<Vec<f64>> = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .collect::<Vec<f64>>()
            })
            .filter(|group| !group.is_empty())
            .collect();

        if usable.len() < 2 {
            return undefined;
        }

        let total_n: usize = usable.iter().map(|g| g.len()).sum();
        if total_n <= usable.len() {
            // No within-group degrees of freedom.
            return undefined;
        }
        let grand_mean = usable.iter().flat_map(|g| g.iter()).sum::<f64>() / total_n as f64;

        let between_ss = usable
            .iter()
            .map(|group| {
                let group_mean = group.iter().sum::<f64>() / group.len() as f64;
                group.len() as f64 * (group_mean - grand_mean).powi(2)
            })
            .sum::<f64>();

        let within_ss = usable
            .iter()
            .flat_map(|group| {
                let group_mean = group.iter().sum::<f64>() / group.len() as f64;
                group.iter().map(move |&x| (x - group_mean).powi(2))
            })
            .sum::<f64>();

        let total_ss = between_ss + within_ss;
        let df_between = usable.len() - 1;
        let df_within = total_n - usable.len();

        let ms_between = between_ss / df_between as f64;
        let ms_within = within_ss / df_within as f64;

        // Zero within-group variance: either the groups are identical (no
        // effect) or they differ deterministically (infinite F). Both are
        // reported without dividing by zero.
        let (f_statistic, p_value) = if ms_within > 0.0 && ms_within.is_finite() {
            let f = ms_between / ms_within;
            (f, f_distribution_sf(f, df_between as f64, df_within as f64))
        } else if between_ss > 0.0 {
            (f64::INFINITY, 0.0)
        } else {
            (0.0, 1.0)
        };

        AnovaResult {
            f_statistic,
            p_value,
            between_ss: A::from(between_ss).unwrap_or_else(A::zero),
            within_ss: A::from(within_ss).unwrap_or_else(A::zero),
            total_ss: A::from(total_ss).unwrap_or_else(A::zero),
            df_between,
            df_within,
        }
    }
}
