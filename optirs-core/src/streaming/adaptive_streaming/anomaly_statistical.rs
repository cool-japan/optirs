// Statistical (non-ML) anomaly detectors for adaptive streaming.
//
// Two classic, real univariate outlier tests over the summed feature value of
// each data point:
//
// - [`ZScoreDetector`] — Welford running mean/variance, with the z-score taken
//   against the true variance `M2 / n` (A1: it used to divide by `sqrt(M2)`,
//   the raw accumulator, which under-scaled the score by `sqrt(n)` so the
//   detector grew steadily less sensitive the longer it ran).
// - [`IQRDetector`] — Tukey fences `[q1 - k*IQR, q3 + k*IQR]` over a sliding
//   window, sorted with a NaN-safe total order (A5: it used to sort with
//   `partial_cmp(..).expect(..)`, which panics on the first `NaN`).
//
// Split out of `anomaly_detection.rs` to keep that file under the 2000-line
// limit.

use super::anomaly_detection::{
    AnomalyDetectionResult, AnomalySeverity, AnomalyType, StatisticalAnomalyDetector,
};
use super::optimizer::StreamingDataPoint;

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};

/// Converts an `f64` constant into the generic element type, reporting an honest
/// error rather than panicking when the type cannot represent it.
fn from_f64<A: Float>(value: f64) -> Result<A, String> {
    A::from(value).ok_or_else(|| format!("{value} cannot be represented in the element type"))
}

/// Maps a score and its decision threshold onto a severity band and a
/// confidence, both real functions of how far past the boundary the score sits.
///
/// Both detectors previously reported a *constant* confidence (`0.8` when
/// firing, `0.2` otherwise, and `0.7`/`0.3` respectively), which carried no
/// information about the strength of the evidence.
///
/// The confidence is the normalised distance of the score from the decision
/// boundary, floored at [`MIN_REPORTED_CONFIDENCE`] so that a verdict is never
/// published with exactly zero confidence — a marginal anomaly is weak evidence,
/// not *no* evidence, and a zero would silently drag an ensemble's averaged
/// confidence to zero.
fn severity_and_confidence<A: Float>(
    score: A,
    threshold: A,
) -> Result<(AnomalySeverity, A), String> {
    if threshold <= A::zero() {
        return Ok((AnomalySeverity::Low, A::zero()));
    }
    let ratio = score / threshold;
    let floor = from_f64::<A>(MIN_REPORTED_CONFIDENCE)?;
    let confidence = (ratio - A::one()).abs().min(A::one()).max(floor);

    let critical_ratio = from_f64::<A>(2.0)?;
    let high_ratio = from_f64::<A>(1.5)?;
    let severity = if ratio >= critical_ratio {
        AnomalySeverity::Critical
    } else if ratio >= high_ratio {
        AnomalySeverity::High
    } else if ratio >= A::one() {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    };
    Ok((severity, confidence))
}

/// Floor on the confidence a detector reports alongside a verdict.
const MIN_REPORTED_CONFIDENCE: f64 = 0.05;

/// Z-Score based statistical detector
pub struct ZScoreDetector<A: Float + Send + Sync> {
    threshold: A,
    running_mean: A,
    running_variance: A,
    sample_count: usize,
}

impl<A: Float + Default + Clone + Send + Sync + Send + Sync> ZScoreDetector<A> {
    /// Creates the detector with the given decision threshold.
    pub fn new(threshold: f64) -> Result<Self, String> {
        Ok(Self {
            threshold: from_f64(threshold)?,
            running_mean: A::zero(),
            running_variance: A::zero(),
            sample_count: 0,
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalAnomalyDetector<A>
    for ZScoreDetector<A>
{
    fn detect_anomaly(
        &mut self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyDetectionResult<A>, String> {
        if self.sample_count < 10 {
            // Not enough samples for reliable detection
            return Ok(AnomalyDetectionResult {
                is_anomaly: false,
                anomaly_score: A::zero(),
                confidence: A::zero(),
                anomaly_type: None,
                severity: AnomalySeverity::Low,
                metadata: HashMap::new(),
            });
        }

        // Calculate Z-score for the data point.
        //
        // `running_variance` is Welford's M2 accumulator (running sum of
        // squared deviations from the mean), not the variance itself — the
        // real (population) variance is `M2 / sample_count` (A1 fix).
        // Dividing by `sqrt(M2)` directly under-scaled the z-score by a
        // factor of `sqrt(sample_count)`, so the detector grew steadily
        // *less* sensitive the longer it ran, eventually never firing.
        let feature_sum = data_point.features.iter().cloned().sum::<A>();
        let z_score = if self.running_variance > A::zero() {
            let count = A::from(self.sample_count).unwrap_or(A::one());
            let variance = self.running_variance / count;
            (feature_sum - self.running_mean) / variance.sqrt()
        } else {
            A::zero()
        };

        let is_anomaly = z_score.abs() > self.threshold;
        let anomaly_score = z_score.abs();
        let (severity, confidence) = severity_and_confidence(anomaly_score, self.threshold)?;

        let mut metadata = HashMap::new();
        metadata.insert("z_score".to_string(), z_score);
        metadata.insert("running_mean".to_string(), self.running_mean);
        metadata.insert(
            "sample_count".to_string(),
            from_f64(self.sample_count as f64)?,
        );

        Ok(AnomalyDetectionResult {
            is_anomaly,
            anomaly_score,
            confidence,
            anomaly_type: is_anomaly.then_some(AnomalyType::StatisticalOutlier),
            severity,
            metadata,
        })
    }

    fn update(&mut self, data_point: &StreamingDataPoint<A>) -> Result<(), String> {
        let feature_sum = data_point.features.iter().cloned().sum::<A>();

        // Update running statistics
        self.sample_count += 1;
        let count = from_f64::<A>(self.sample_count as f64)?;
        let delta = feature_sum - self.running_mean;
        self.running_mean = self.running_mean + delta / count;
        let delta2 = feature_sum - self.running_mean;
        self.running_variance = self.running_variance + delta * delta2;

        Ok(())
    }

    fn reset(&mut self) {
        self.running_mean = A::zero();
        self.running_variance = A::zero();
        self.sample_count = 0;
    }

    fn name(&self) -> String {
        "zscore".to_string()
    }

    fn get_threshold(&self) -> A {
        self.threshold
    }

    fn set_threshold(&mut self, threshold: A) {
        self.threshold = threshold;
    }
}

/// IQR-based statistical detector
pub struct IQRDetector<A: Float + Send + Sync> {
    threshold: A,
    recent_values: VecDeque<A>,
    window_size: usize,
}

impl<A: Float + Default + Clone + Send + Sync + Send + Sync> IQRDetector<A> {
    /// Creates the detector with the given decision threshold.
    pub fn new(threshold: f64) -> Result<Self, String> {
        Ok(Self {
            threshold: from_f64(threshold)?,
            recent_values: VecDeque::with_capacity(100),
            window_size: 100,
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalAnomalyDetector<A>
    for IQRDetector<A>
{
    fn detect_anomaly(
        &mut self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyDetectionResult<A>, String> {
        if self.recent_values.len() < 20 {
            return Ok(AnomalyDetectionResult {
                is_anomaly: false,
                anomaly_score: A::zero(),
                confidence: A::zero(),
                anomaly_type: None,
                severity: AnomalySeverity::Low,
                metadata: HashMap::new(),
            });
        }

        // Calculate IQR.
        //
        // A5: this used `partial_cmp(..).expect(..)`, which panics the instant
        // a single `NaN` reaches the comparator — a plausible occurrence for a
        // metric stream fed by a diverging optimizer. `total_order` is a
        // genuine total order in which `NaN` sorts last, so the sort is safe
        // and the quartiles are still taken over the real observations.
        let mut sorted_values: Vec<A> = self.recent_values.iter().cloned().collect();
        super::statistics::sort_ascending(&mut sorted_values);

        // Compute the quartiles over the finite observations only, so a single
        // non-finite value cannot drag `q3` (and therefore the whole decision
        // band) to `NaN` or infinity. Filtering rather than truncating at the
        // first non-finite value matters: `sort_ascending` places `NaN` last but
        // `-inf` sorts *first*, so a prefix-based guard would throw away the
        // entire window over one `-inf`.
        sorted_values.retain(|value| value.is_finite());
        if sorted_values.len() < 4 {
            return Err(
                "IQR detector: fewer than four finite observations in the window".to_string(),
            );
        }
        let finite_len = sorted_values.len();
        let q1_idx = finite_len / 4;
        let q3_idx = (3 * finite_len / 4).min(finite_len - 1);
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - self.threshold * iqr;
        let upper_bound = q3 + self.threshold * iqr;

        let feature_sum = data_point.features.iter().cloned().sum::<A>();
        let is_anomaly = feature_sum < lower_bound || feature_sum > upper_bound;

        let distance_from_bounds = if feature_sum < lower_bound {
            lower_bound - feature_sum
        } else if feature_sum > upper_bound {
            feature_sum - upper_bound
        } else {
            A::zero()
        };

        // The score is the overshoot expressed in IQR units, so it is comparable
        // across streams of different scale. A degenerate (zero) IQR means the
        // window is constant and there is no spread to measure against, so the
        // epsilon floor keeps the division well-defined instead of blowing up.
        let epsilon = from_f64::<A>(1e-8)?;
        let anomaly_score = distance_from_bounds / iqr.max(epsilon);
        // A point exactly on a Tukey fence is the decision boundary, so severity
        // and confidence are graded against one IQR-unit of overshoot beyond it.
        let (severity, confidence) = severity_and_confidence(
            anomaly_score + if is_anomaly { A::one() } else { A::zero() },
            A::one(),
        )?;

        let mut metadata = HashMap::new();
        metadata.insert("q1".to_string(), q1);
        metadata.insert("q3".to_string(), q3);
        metadata.insert("iqr".to_string(), iqr);
        metadata.insert("lower_bound".to_string(), lower_bound);
        metadata.insert("upper_bound".to_string(), upper_bound);

        Ok(AnomalyDetectionResult {
            is_anomaly,
            anomaly_score,
            confidence,
            anomaly_type: is_anomaly.then_some(AnomalyType::StatisticalOutlier),
            severity,
            metadata,
        })
    }

    fn update(&mut self, data_point: &StreamingDataPoint<A>) -> Result<(), String> {
        let feature_sum = data_point.features.iter().cloned().sum::<A>();

        if self.recent_values.len() >= self.window_size {
            self.recent_values.pop_front();
        }
        self.recent_values.push_back(feature_sum);

        Ok(())
    }

    fn reset(&mut self) {
        self.recent_values.clear();
    }

    fn name(&self) -> String {
        "iqr".to_string()
    }

    fn get_threshold(&self) -> A {
        self.threshold
    }

    fn set_threshold(&mut self, threshold: A) {
        self.threshold = threshold;
    }
}
