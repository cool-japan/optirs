//! Real optimization-landscape statistics derived from an optimization history.
//!
//! `OptimizationLandscapeAnalyzer::analyze` used to ignore both of its
//! arguments and return the constants `complexity = 0.5`, `difficulty = 0.3`,
//! `confidence = 0.9`. Every downstream consumer (sequence adaptation, attention
//! sparsity, architecture adaptation, performance prediction) therefore made the
//! same decision on every problem. This module computes those numbers from the
//! gradient and loss histories the analyzer is actually handed.
//!
//! Nothing here is a heuristic label attached to a constant: every quantity is a
//! statistic of the input, documented with the formula that produces it.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Statistics extracted from a gradient/loss history.
///
/// All fields are dimensionless and, except `loss_slope`, bounded to `[0, 1]` so
/// they can be combined without one term swamping the others.
#[derive(Debug, Clone, PartialEq)]
pub struct LandscapeStatistics {
    /// Mean gradient L2 norm across the history.
    pub mean_gradient_norm: f64,

    /// Coefficient of variation of the gradient norm — how unstable the
    /// gradient magnitude is. `0` for a perfectly steady norm.
    pub gradient_norm_cv: f64,

    /// Mean cosine similarity between consecutive gradients, mapped from
    /// `[-1, 1]` to `[0, 1]`. `1` = perfectly consistent descent direction,
    /// `0.5` = orthogonal (uninformative), `0` = perfectly reversing.
    pub direction_consistency: f64,

    /// Fraction of consecutive gradient pairs whose cosine similarity is
    /// negative, i.e. how often the descent direction reversed. A ravine or an
    /// oscillating step size shows up here.
    pub reversal_rate: f64,

    /// Relative loss improvement `(first - last) / |first|`, clamped to
    /// `[-1, 1]`. Positive means the loss went down.
    pub loss_slope: f64,

    /// Normalized second-difference energy of the loss curve — a curvature
    /// proxy. `0` for a straight line, approaching `1` for a jagged curve.
    pub loss_roughness: f64,

    /// Fraction of loss steps that went *up*. Pure noise sits near `0.5`.
    pub loss_increase_rate: f64,

    /// Number of history entries the statistics were computed from.
    pub samples: usize,
}

impl LandscapeStatistics {
    /// Statistics for an empty history: everything neutral, zero samples.
    ///
    /// Callers use `samples == 0` to know the numbers carry no evidence, which
    /// is what drives the low confidence reported by
    /// [`Self::analysis_confidence`].
    pub fn empty() -> Self {
        Self {
            mean_gradient_norm: 0.0,
            gradient_norm_cv: 0.0,
            direction_consistency: 0.5,
            reversal_rate: 0.0,
            loss_slope: 0.0,
            loss_roughness: 0.0,
            loss_increase_rate: 0.0,
            samples: 0,
        }
    }

    /// Compute the statistics of a gradient and loss history.
    ///
    /// The two histories are independent: either may be empty, and each block
    /// of statistics is computed only from the data that exists.
    pub fn from_history<T>(gradient_history: &[Array1<T>], loss_history: &[T]) -> Self
    where
        T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
    {
        let mut stats = Self::empty();
        stats.samples = gradient_history.len().max(loss_history.len());

        // ---- gradient block -------------------------------------------------
        let norms: Vec<f64> = gradient_history
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&x| {
                        let v = x.to_f64().unwrap_or(0.0);
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        if !norms.is_empty() {
            let n = norms.len() as f64;
            let mean = norms.iter().sum::<f64>() / n;
            stats.mean_gradient_norm = mean;
            if mean > f64::EPSILON {
                let var = norms.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
                // A CV above 1 already means "wildly unstable"; saturate there so
                // one outlier step cannot dominate the composite scores.
                stats.gradient_norm_cv = (var.sqrt() / mean).min(1.0);
            }
        }

        if gradient_history.len() >= 2 {
            let mut cos_sum = 0.0;
            let mut reversals = 0usize;
            let mut pairs = 0usize;
            for window in gradient_history.windows(2) {
                let (a, b) = (&window[0], &window[1]);
                let width = a.len().min(b.len());
                if width == 0 {
                    continue;
                }
                let mut dot = 0.0;
                let mut na = 0.0;
                let mut nb = 0.0;
                for i in 0..width {
                    let av = a[i].to_f64().unwrap_or(0.0);
                    let bv = b[i].to_f64().unwrap_or(0.0);
                    dot += av * bv;
                    na += av * av;
                    nb += bv * bv;
                }
                if na <= f64::EPSILON || nb <= f64::EPSILON {
                    continue;
                }
                let cos = (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0);
                cos_sum += cos;
                if cos < 0.0 {
                    reversals += 1;
                }
                pairs += 1;
            }
            if pairs > 0 {
                let mean_cos = cos_sum / pairs as f64;
                // Map [-1, 1] -> [0, 1].
                stats.direction_consistency = (mean_cos + 1.0) / 2.0;
                stats.reversal_rate = reversals as f64 / pairs as f64;
            }
        }

        // ---- loss block -----------------------------------------------------
        let losses: Vec<f64> = loss_history
            .iter()
            .map(|&l| l.to_f64().unwrap_or(0.0))
            .collect();

        if losses.len() >= 2 {
            let first = losses[0];
            let last = losses[losses.len() - 1];
            if first.abs() > f64::EPSILON {
                stats.loss_slope = ((first - last) / first.abs()).clamp(-1.0, 1.0);
            }

            let increases = losses.windows(2).filter(|w| w[1] > w[0]).count();
            stats.loss_increase_rate = increases as f64 / (losses.len() - 1) as f64;
        }

        if losses.len() >= 3 {
            // Second differences measure how far the curve is from a straight
            // line. Normalize by the mean |first difference| so the result is
            // scale-free, then squash into [0, 1).
            let first_diff_mag: f64 = losses.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>()
                / (losses.len() - 1) as f64;
            let second_diff_mag: f64 = losses
                .windows(3)
                .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
                .sum::<f64>()
                / (losses.len() - 2) as f64;
            if first_diff_mag > f64::EPSILON {
                let ratio = second_diff_mag / first_diff_mag;
                stats.loss_roughness = ratio / (1.0 + ratio);
            }
        }

        stats
    }

    /// Composite landscape **complexity** in `[0, 1]`.
    ///
    /// Complexity here means "how structurally hard is this surface to model":
    /// a jagged loss curve, an unstable gradient magnitude and frequent
    /// direction reversals all raise it. Weights sum to 1.
    pub fn complexity(&self) -> f64 {
        let terms = [
            (0.35, self.loss_roughness),
            (0.30, self.gradient_norm_cv),
            (0.20, self.reversal_rate),
            (0.15, 1.0 - self.direction_consistency),
        ];
        terms
            .iter()
            .map(|(w, v)| w * v)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    /// Composite optimization **difficulty** in `[0, 1]`.
    ///
    /// Difficulty means "how badly is optimization currently going": a loss that
    /// is not descending, steps that increase it, and reversing gradients all
    /// raise it. Distinct from `complexity`, which is about the surface rather
    /// than the progress.
    pub fn difficulty(&self) -> f64 {
        // `loss_slope` in [-1, 1] is the *fraction of the loss removed*: +1 means
        // the loss went to zero, 0 means it did not move, negative means it grew.
        // `stalled` is therefore the fraction of available progress that was
        // **not** made: 0 for a fully converged run, 1 for a flat one, and also 1
        // (saturated) for a diverging one.
        let stalled = (1.0 - self.loss_slope).clamp(0.0, 1.0);
        let terms = [
            (0.45, stalled),
            (0.30, self.loss_increase_rate),
            (0.25, self.reversal_rate),
        ];
        terms
            .iter()
            .map(|(w, v)| w * v)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    /// Confidence in the analysis, in `[0, 1]`.
    ///
    /// Grows with the number of samples (saturating at `horizon`) and shrinks
    /// when the signal is noisy. An empty history reports ~0, not the old
    /// hardcoded `0.9`.
    pub fn analysis_confidence(&self, horizon: usize) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        let target = horizon.max(2) as f64;
        let coverage = (self.samples as f64 / target).min(1.0);
        let noise = (self.gradient_norm_cv + self.loss_increase_rate) / 2.0;
        (coverage * (1.0 - 0.5 * noise)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn grads(vectors: &[[f64; 2]]) -> Vec<Array1<f64>> {
        vectors
            .iter()
            .map(|v| Array1::from_vec(v.to_vec()))
            .collect()
    }

    #[test]
    fn empty_history_has_zero_confidence() {
        let stats = LandscapeStatistics::from_history::<f64>(&[], &[]);
        assert_eq!(stats.samples, 0);
        assert_eq!(stats.analysis_confidence(10), 0.0);
    }

    #[test]
    fn consistent_descent_is_easy_and_simple() {
        // Same direction every step, geometrically shrinking loss.
        let g = grads(&[[1.0, 0.0], [0.9, 0.0], [0.81, 0.0], [0.73, 0.0]]);
        let losses = vec![1.0, 0.5, 0.25, 0.125];
        let stats = LandscapeStatistics::from_history(&g, &losses);

        assert!(
            stats.direction_consistency > 0.99,
            "consistency {}",
            stats.direction_consistency
        );
        assert_eq!(stats.reversal_rate, 0.0);
        assert!(stats.loss_slope > 0.8, "slope {}", stats.loss_slope);
        assert!(
            stats.difficulty() < 0.2,
            "difficulty {}",
            stats.difficulty()
        );
    }

    #[test]
    fn oscillating_history_is_hard_and_complex() {
        let g = grads(&[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]);
        let losses = vec![1.0, 2.0, 1.0, 2.0];
        let stats = LandscapeStatistics::from_history(&g, &losses);

        assert_eq!(stats.reversal_rate, 1.0);
        assert!(
            stats.direction_consistency < 0.01,
            "consistency {}",
            stats.direction_consistency
        );
        assert!(
            stats.difficulty() > 0.5,
            "difficulty {}",
            stats.difficulty()
        );
        assert!(
            stats.complexity() > 0.3,
            "complexity {}",
            stats.complexity()
        );
    }

    #[test]
    fn complexity_and_difficulty_separate_concerns() {
        // Smooth, steady gradients but the loss is stuck: low complexity,
        // high difficulty.
        let g = grads(&[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]);
        let flat = vec![1.0, 1.0, 1.0, 1.0];
        let stuck = LandscapeStatistics::from_history(&g, &flat);
        assert!(
            stuck.complexity() < 0.1,
            "complexity {}",
            stuck.complexity()
        );
        assert!(
            stuck.difficulty() > 0.4,
            "difficulty {}",
            stuck.difficulty()
        );
    }

    #[test]
    fn different_histories_give_different_scores() {
        let easy = LandscapeStatistics::from_history(
            &grads(&[[1.0, 0.0], [0.5, 0.0], [0.25, 0.0]]),
            &[1.0, 0.4, 0.1],
        );
        let hard = LandscapeStatistics::from_history(
            &grads(&[[1.0, 0.0], [-2.0, 0.5], [3.0, -1.0]]),
            &[1.0, 3.0, 0.5],
        );
        assert!(
            (easy.complexity() - hard.complexity()).abs() > 1e-6,
            "complexity collapsed to a constant"
        );
        assert!(
            (easy.difficulty() - hard.difficulty()).abs() > 1e-6,
            "difficulty collapsed to a constant"
        );
    }

    #[test]
    fn confidence_grows_with_samples() {
        let short =
            LandscapeStatistics::from_history(&grads(&[[1.0, 0.0], [1.0, 0.0]]), &[1.0, 0.5]);
        let long = LandscapeStatistics::from_history(&grads(&[[1.0, 0.0]; 20]), &vec![1.0; 20]);
        assert!(
            long.analysis_confidence(20) > short.analysis_confidence(20),
            "{} !> {}",
            long.analysis_confidence(20),
            short.analysis_confidence(20)
        );
    }
}
