// Detection bookkeeping and quality measurement for the anomaly detectors.
//
// Holds the confusion-matrix counters every detector reports its quality from,
// plus the bounded score-retention buffer that makes a *full-curve* ROC area
// computable rather than a single operating point.
//
// A one-threshold confusion matrix pins down exactly one point of the ROC
// curve, so `(TPR + TNR) / 2` is the only "area" it can support — and that
// number is not the AUC anyone reading `auc_roc` expects. Tracing the real
// curve needs the *scores* kept against their labels so the threshold can be
// swept; that is what [`DetectionCounters::retained_scores`] is for.

use super::anomaly_detection::MLModelMetrics;

use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::time::Duration;

/// Labelled scores retained per detector for the ROC sweep.
///
/// Bounded so a long-running stream cannot grow this without limit; the oldest
/// labelled score is evicted first, which keeps the measured curve describing
/// the detector's recent behaviour rather than its whole history.
pub const MAX_RETAINED_SCORES: usize = 4096;

/// Real confusion-matrix counters backing every ML detector's quality metrics.
///
/// `predictions`/`flagged` are updated on every scored point (so the observed
/// flag rate is always available), while the four confusion cells only move
/// when ground truth is supplied via [`DetectionCounters::record_outcome`].
#[derive(Debug, Clone, Default)]
pub struct DetectionCounters {
    /// Points scored by the detector.
    pub predictions: usize,
    /// Points the detector flagged as anomalous.
    pub flagged: usize,
    /// Flagged and genuinely anomalous.
    pub true_positives: usize,
    /// Flagged but genuinely normal.
    pub false_positives: usize,
    /// Not flagged and genuinely normal.
    pub true_negatives: usize,
    /// Not flagged but genuinely anomalous.
    pub false_negatives: usize,
    /// Score of the most recent prediction, awaiting a label.
    ///
    /// `record_detection_outcome` supplies ground truth for the *most recent*
    /// detection, so the score to pair a label with is the last one produced.
    /// It is cleared on consumption, so one score can never be labelled twice.
    pending_score: Option<f64>,
    /// Bounded buffer of `(score, was_true_anomaly)` pairs, the raw material of
    /// the ROC sweep.
    labelled_scores: VecDeque<(f64, bool)>,
}

impl DetectionCounters {
    /// Records that a point was scored, whether it was flagged, and the score
    /// that produced the verdict.
    ///
    /// The score is retained (unlabelled) until [`Self::record_outcome`]
    /// supplies ground truth for it. A non-finite score is not retained: it
    /// cannot be placed on a threshold sweep.
    pub fn record_prediction<A: Float>(&mut self, flagged: bool, score: A) {
        self.predictions += 1;
        if flagged {
            self.flagged += 1;
        }
        self.pending_score = score.to_f64().filter(|value| value.is_finite());
    }

    /// Records ground truth for one prediction.
    ///
    /// When a score is awaiting a label (i.e. the detector scored a point since
    /// the last outcome), the pair is added to the ROC buffer.
    pub fn record_outcome(&mut self, predicted_anomaly: bool, was_true_anomaly: bool) {
        match (predicted_anomaly, was_true_anomaly) {
            (true, true) => self.true_positives += 1,
            (true, false) => self.false_positives += 1,
            (false, false) => self.true_negatives += 1,
            (false, true) => self.false_negatives += 1,
        }

        if let Some(score) = self.pending_score.take() {
            if self.labelled_scores.len() >= MAX_RETAINED_SCORES {
                self.labelled_scores.pop_front();
            }
            self.labelled_scores.push_back((score, was_true_anomaly));
        }
    }

    /// Total number of labelled outcomes recorded.
    pub fn labelled(&self) -> usize {
        self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
    }

    /// Labelled `(score, was_true_anomaly)` pairs currently retained.
    pub fn retained_scores(&self) -> usize {
        self.labelled_scores.len()
    }

    /// Fraction of scored points that were flagged, or `None` before any point
    /// has been scored. This is a real observation, available without labels.
    pub fn observed_flag_rate(&self) -> Option<f64> {
        if self.predictions == 0 {
            None
        } else {
            Some(self.flagged as f64 / self.predictions as f64)
        }
    }

    /// Area under the ROC curve, traced over every threshold the retained
    /// scores admit.
    ///
    /// The curve is swept from the highest score downwards; each group of
    /// **equal** scores advances the operating point once, and the area between
    /// two consecutive operating points is taken as a trapezoid. Grouping ties
    /// is what makes this correct for quantised scores: stepping through tied
    /// scores one at a time would trace a staircase through the middle of the
    /// diagonal segment the tie really produces, and would report a different
    /// area depending on the order the tied points happened to arrive in.
    ///
    /// Returns an honest error while fewer than two labelled scores of either
    /// class are retained: with one point of a class the curve has a single
    /// vertical (or horizontal) segment and the "area" would be an artefact of
    /// where that one point landed, not a measurement of ranking quality.
    pub fn auc_roc(&self) -> Result<f64, String> {
        let positives = self
            .labelled_scores
            .iter()
            .filter(|(_, label)| *label)
            .count();
        let negatives = self.labelled_scores.len() - positives;
        if positives < 2 || negatives < 2 {
            return Err(format!(
                "AUC-ROC needs at least two labelled scores of each class to trace a \
                 curve; {positives} anomalous and {negatives} normal scores are \
                 retained. Score points with the detector and feed ground truth back \
                 through `record_detection_outcome` first"
            ));
        }

        let mut ordered: Vec<(f64, bool)> = self.labelled_scores.iter().copied().collect();
        // Descending by score: the sweep starts with the strictest threshold,
        // where nothing is flagged and the operating point is the origin.
        ordered.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut true_positives = 0.0f64;
        let mut false_positives = 0.0f64;
        let mut previous_tp = 0.0f64;
        let mut previous_fp = 0.0f64;
        let mut area = 0.0f64;

        let mut index = 0usize;
        while index < ordered.len() {
            let threshold = ordered[index].0;
            while index < ordered.len() && ordered[index].0 == threshold {
                if ordered[index].1 {
                    true_positives += 1.0;
                } else {
                    false_positives += 1.0;
                }
                index += 1;
            }
            // Trapezoid between the previous operating point and this one. A
            // group containing both classes produces a genuinely sloped edge,
            // which is exactly the half-credit a tie deserves.
            area += (false_positives - previous_fp) * (true_positives + previous_tp) / 2.0;
            previous_tp = true_positives;
            previous_fp = false_positives;
        }

        Ok(area / (positives as f64 * negatives as f64))
    }

    /// Derives quality metrics from the recorded confusion matrix.
    ///
    /// `auc_roc` carries the full-curve area when the retained scores support
    /// it and `None` otherwise — the single-operating-point substitute
    /// `(TPR + TNR) / 2` that used to be reported here is *not* the AUC, and
    /// reporting it under that name overstated a one-threshold detector by
    /// however much of the curve it never saw. Use [`Self::auc_roc`] directly
    /// for the reason it is unavailable.
    pub fn to_metrics<A: Float + Send + Sync>(
        &self,
        detector_name: String,
        training_time: Duration,
        inference_time: Duration,
    ) -> Result<MLModelMetrics<A>, String> {
        let labelled = self.labelled();
        if labelled == 0 {
            return Err(format!(
                "{detector_name}: no labelled outcomes recorded, so accuracy, \
                 precision, recall, F1 and AUC are undefined — call \
                 `record_outcome` with ground truth first"
            ));
        }

        let true_positives = self.true_positives as f64;
        let false_positives = self.false_positives as f64;
        let true_negatives = self.true_negatives as f64;
        let false_negatives = self.false_negatives as f64;

        let precision = if true_positives + false_positives > 0.0 {
            true_positives / (true_positives + false_positives)
        } else {
            0.0
        };
        let recall = if true_positives + false_negatives > 0.0 {
            true_positives / (true_positives + false_negatives)
        } else {
            0.0
        };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let accuracy = (true_positives + true_negatives) / labelled as f64;
        let false_positive_rate = if false_positives + true_negatives > 0.0 {
            false_positives / (false_positives + true_negatives)
        } else {
            0.0
        };

        let convert = |value: f64| -> Result<A, String> {
            A::from(value)
                .ok_or_else(|| format!("{value} cannot be represented in the element type"))
        };

        let auc_roc = match self.auc_roc() {
            Ok(area) => Some(convert(area)?),
            Err(_) => None,
        };

        Ok(MLModelMetrics {
            accuracy: convert(accuracy)?,
            precision: convert(precision)?,
            recall: convert(recall)?,
            f1_score: convert(f1_score)?,
            auc_roc,
            false_positive_rate: convert(false_positive_rate)?,
            training_time,
            inference_time,
        })
    }

    /// Balanced accuracy `(TPR + TNR) / 2` measured from the confusion matrix,
    /// or `None` before any labelled outcome has been recorded.
    ///
    /// This is the quantity the adaptive ensemble weights its members by: it is
    /// insensitive to how rare anomalies are, which plain accuracy is not — a
    /// detector that flags nothing scores 0.99 accuracy on a 1%-anomaly stream
    /// and would dominate an accuracy-weighted vote while detecting nothing.
    pub fn balanced_accuracy(&self) -> Option<f64> {
        if self.labelled() == 0 {
            return None;
        }
        let true_positives = self.true_positives as f64;
        let false_negatives = self.false_negatives as f64;
        let true_negatives = self.true_negatives as f64;
        let false_positives = self.false_positives as f64;

        let positives = true_positives + false_negatives;
        let negatives = true_negatives + false_positives;
        // A class that has never been observed contributes nothing, so the
        // average is taken over the classes actually seen rather than crediting
        // an unobserved class with a perfect (or zero) rate.
        let mut total = 0.0;
        let mut terms = 0.0;
        if positives > 0.0 {
            total += true_positives / positives;
            terms += 1.0;
        }
        if negatives > 0.0 {
            total += true_negatives / negatives;
            terms += 1.0;
        }
        if terms == 0.0 {
            return None;
        }
        Some(total / terms)
    }
}

#[cfg(test)]
#[path = "anomaly_scoring_tests.rs"]
mod anomaly_scoring_tests;
