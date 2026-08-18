// Machine-learning anomaly detectors for adaptive streaming.
//
// Three genuinely different unsupervised novelty detectors, each implementing
// its published algorithm over a bounded sliding window of recent observations:
//
// - [`IsolationForestDetector`] — Liu, Ting & Zhou, "Isolation Forest",
//   ICDM 2008. Anomalies are *easy to isolate*, so they sit at a shallow
//   average depth in randomly-split trees.
// - [`OneClassSvmDetector`] — the NORMA online one-class SVM of Kivinen, Smola
//   & Williamson, "Online Learning with Kernels", IEEE TSP 2004, with an RBF
//   kernel expansion over a bounded support-vector set.
// - [`LofDetector`] — Breunig, Kriegel, Ng & Sander, "LOF: Identifying
//   Density-Based Local Outliers", SIGMOD 2000. Compares a point's local
//   density to that of its k nearest neighbours.
//
// None of them fabricates a score, and none of them reports a quality metric
// it has not measured: `get_performance_metrics` is driven by a real confusion
// matrix that only exists once labelled feedback has been supplied through
// `record_outcome`, and returns an honest error before that.

use super::anomaly_detection::{
    AnomalyDetectionResult, AnomalySeverity, AnomalyType, DetectionCounters, MLAnomalyDetector,
    MLModelMetrics,
};
use super::optimizer::StreamingDataPoint;
use super::statistics as stats;

use scirs2_core::numeric::Float;
use scirs2_core::random::{seeded_rng, CoreRandom};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Default number of recent observations each detector keeps.
const DEFAULT_WINDOW: usize = 512;

/// Extracts a finite `f64` feature vector, or `None` when the point carries no
/// usable features.
fn feature_vector<A: Float + Send + Sync>(data_point: &StreamingDataPoint<A>) -> Option<Vec<f64>> {
    let features: Vec<f64> = data_point
        .features
        .iter()
        .map(|v| v.to_f64().unwrap_or(f64::NAN))
        .collect();
    if features.is_empty() || features.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(features)
}

/// Converts an `f64` into the generic element type with an honest error.
fn from_f64<A: Float>(value: f64) -> Result<A, String> {
    A::from(value).ok_or_else(|| format!("{value} cannot be represented in the element type"))
}

/// Squared Euclidean distance between two equal-length vectors.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Maps a raw anomaly score and threshold onto a severity band.
///
/// The band is a real function of how far past the threshold the score sits,
/// not a constant.
fn severity_for(score: f64, threshold: f64) -> AnomalySeverity {
    if threshold <= 0.0 {
        return AnomalySeverity::Low;
    }
    let ratio = score / threshold;
    if ratio >= 2.0 {
        AnomalySeverity::Critical
    } else if ratio >= 1.5 {
        AnomalySeverity::High
    } else if ratio >= 1.0 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Builds a detection result from a real score.
fn result_from_score<A: Float + Send + Sync>(
    score: f64,
    threshold: f64,
    anomaly_type: AnomalyType,
    metadata: HashMap<String, A>,
) -> Result<AnomalyDetectionResult<A>, String> {
    let is_anomaly = score > threshold;
    // Confidence is the normalised margin between the score and the decision
    // boundary, so it moves continuously with the evidence.
    let confidence = if threshold > 0.0 {
        ((score - threshold).abs() / threshold).min(1.0)
    } else {
        0.0
    };
    Ok(AnomalyDetectionResult {
        is_anomaly,
        anomaly_score: from_f64(score)?,
        confidence: from_f64(confidence)?,
        anomaly_type: is_anomaly.then_some(anomaly_type),
        severity: severity_for(score, threshold),
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Isolation Forest
// ---------------------------------------------------------------------------

/// A single node of an isolation tree.
#[derive(Debug, Clone)]
enum ITreeNode {
    /// Internal split on `feature < threshold`.
    Split {
        feature: usize,
        threshold: f64,
        left: Box<ITreeNode>,
        right: Box<ITreeNode>,
    },
    /// External node holding the number of points that reached it.
    Leaf { size: usize },
}

/// Average path length of an unsuccessful BST search over `n` points —
/// `c(n) = 2 H(n-1) - 2(n-1)/n`, the normalisation constant of the isolation
/// forest score.
fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n = n as f64;
    let harmonic = (n - 1.0).ln() + std::f64::consts::EULER_GAMMA;
    2.0 * harmonic - 2.0 * (n - 1.0) / n
}

/// Builds one isolation tree from a subsample.
fn build_itree(
    sample: &[Vec<f64>],
    depth: usize,
    max_depth: usize,
    rng: &mut CoreRandom<scirs2_core::random::rngs::StdRng>,
) -> ITreeNode {
    if depth >= max_depth || sample.len() <= 1 {
        return ITreeNode::Leaf { size: sample.len() };
    }

    let dimensions = sample[0].len();
    if dimensions == 0 {
        return ITreeNode::Leaf { size: sample.len() };
    }

    // Pick a random attribute, then a random split point strictly inside its
    // observed range — the defining construction of an isolation tree.
    let feature = rng.gen_range(0..dimensions);
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for row in sample {
        let value = row.get(feature).copied().unwrap_or(0.0);
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    if max <= min || !(min.is_finite() && max.is_finite()) {
        // The attribute is constant (or unusable) here: nothing left to isolate.
        return ITreeNode::Leaf { size: sample.len() };
    }

    let threshold = rng.gen_range(min..max);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for row in sample {
        if row.get(feature).copied().unwrap_or(0.0) < threshold {
            left.push(row.clone());
        } else {
            right.push(row.clone());
        }
    }

    if left.is_empty() || right.is_empty() {
        return ITreeNode::Leaf { size: sample.len() };
    }

    ITreeNode::Split {
        feature,
        threshold,
        left: Box::new(build_itree(&left, depth + 1, max_depth, rng)),
        right: Box::new(build_itree(&right, depth + 1, max_depth, rng)),
    }
}

/// Path length of `point` in `tree`, with the standard `c(size)` correction
/// applied at external nodes that still hold more than one point.
fn path_length(tree: &ITreeNode, point: &[f64], depth: usize) -> f64 {
    match tree {
        ITreeNode::Leaf { size } => depth as f64 + average_path_length(*size),
        ITreeNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            let value = point.get(*feature).copied().unwrap_or(0.0);
            if value < *threshold {
                path_length(left, point, depth + 1)
            } else {
                path_length(right, point, depth + 1)
            }
        }
    }
}

/// Isolation Forest novelty detector.
///
/// Keeps a sliding window of recent observations and periodically rebuilds a
/// forest of isolation trees from a subsample of it. The anomaly score is the
/// published `s(x) = 2^(-E[h(x)] / c(psi))`, which lies in `(0, 1)` and rises
/// towards `1` for points that isolate quickly.
pub struct IsolationForestDetector<A: Float + Send + Sync> {
    trees: Vec<ITreeNode>,
    window: VecDeque<Vec<f64>>,
    window_capacity: usize,
    tree_count: usize,
    subsample_size: usize,
    max_depth: usize,
    /// Score above which a point is called an anomaly.
    threshold: f64,
    /// Observations seen since the forest was last rebuilt.
    since_refit: usize,
    /// Rebuild cadence.
    refit_interval: usize,
    rng: CoreRandom<scirs2_core::random::rngs::StdRng>,
    counters: DetectionCounters,
    training_time: std::time::Duration,
    inference_time_total: std::time::Duration,
    inference_samples: usize,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> IsolationForestDetector<A> {
    /// Creates a detector with the published defaults (100 trees, subsample
    /// 256, depth `ceil(log2(psi))`) and a deterministic RNG so repeated runs
    /// on the same stream are reproducible.
    pub fn new() -> Result<Self, String> {
        Self::with_parameters(100, 256, 0.6, 20250817)
    }

    /// Creates a detector with explicit parameters.
    pub fn with_parameters(
        tree_count: usize,
        subsample_size: usize,
        threshold: f64,
        seed: u64,
    ) -> Result<Self, String> {
        if tree_count == 0 {
            return Err("isolation forest needs at least one tree".to_string());
        }
        if subsample_size < 2 {
            return Err("isolation forest subsample size must be at least 2".to_string());
        }
        if !(threshold > 0.0 && threshold < 1.0) {
            return Err(format!(
                "isolation forest threshold must lie strictly in (0, 1), got {threshold}"
            ));
        }
        let max_depth = (subsample_size as f64).log2().ceil().max(1.0) as usize;
        Ok(Self {
            trees: Vec::new(),
            window: VecDeque::with_capacity(DEFAULT_WINDOW),
            window_capacity: DEFAULT_WINDOW.max(subsample_size),
            tree_count,
            subsample_size,
            max_depth,
            threshold,
            since_refit: 0,
            refit_interval: subsample_size,
            rng: seeded_rng(seed),
            counters: DetectionCounters::default(),
            training_time: std::time::Duration::ZERO,
            inference_time_total: std::time::Duration::ZERO,
            inference_samples: 0,
            _marker: std::marker::PhantomData,
        })
    }

    /// Whether the forest has been fitted and can score points.
    pub fn is_fitted(&self) -> bool {
        !self.trees.is_empty()
    }

    fn push_observation(&mut self, features: Vec<f64>) {
        if self.window.len() >= self.window_capacity {
            self.window.pop_front();
        }
        self.window.push_back(features);
        self.since_refit += 1;
    }

    fn fit_forest(&mut self) {
        let available = self.window.len();
        if available < 2 {
            return;
        }
        let started = Instant::now();
        let sample_size = self.subsample_size.min(available);
        let pool: Vec<Vec<f64>> = self.window.iter().cloned().collect();

        let mut trees = Vec::with_capacity(self.tree_count);
        for _ in 0..self.tree_count {
            // Sample without replacement by partial shuffle of an index list.
            let mut indices: Vec<usize> = (0..available).collect();
            for i in 0..sample_size {
                let j = self.rng.gen_range(i..available);
                indices.swap(i, j);
            }
            let subsample: Vec<Vec<f64>> = indices[..sample_size]
                .iter()
                .map(|&i| pool[i].clone())
                .collect();
            trees.push(build_itree(&subsample, 0, self.max_depth, &mut self.rng));
        }

        self.trees = trees;
        self.since_refit = 0;
        self.training_time += started.elapsed();
    }

    /// Raw isolation score in `(0, 1)`, or `None` before the forest is fitted.
    pub fn score(&self, features: &[f64]) -> Option<f64> {
        if self.trees.is_empty() {
            return None;
        }
        let mean_depth: f64 = self
            .trees
            .iter()
            .map(|tree| path_length(tree, features, 0))
            .sum::<f64>()
            / self.trees.len() as f64;
        let normaliser = average_path_length(self.subsample_size.min(self.window.len().max(2)));
        if normaliser <= 0.0 {
            return None;
        }
        Some(2.0_f64.powf(-mean_depth / normaliser))
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> MLAnomalyDetector<A>
    for IsolationForestDetector<A>
{
    fn detect_anomaly(
        &mut self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyDetectionResult<A>, String> {
        let features = feature_vector(data_point)
            .ok_or_else(|| "isolation forest: data point has no finite features".to_string())?;

        if self.trees.is_empty() || self.since_refit >= self.refit_interval {
            self.fit_forest();
        }

        let started = Instant::now();
        let Some(score) = self.score(&features) else {
            // Honest "not enough data yet" rather than a fabricated score.
            let mut metadata = HashMap::new();
            metadata.insert(
                "window_size".to_string(),
                from_f64::<A>(self.window.len() as f64)?,
            );
            return Ok(AnomalyDetectionResult {
                is_anomaly: false,
                anomaly_score: A::zero(),
                confidence: A::zero(),
                anomaly_type: None,
                severity: AnomalySeverity::Low,
                metadata,
            });
        };
        self.inference_time_total += started.elapsed();
        self.inference_samples += 1;

        let mut metadata = HashMap::new();
        metadata.insert(
            "tree_count".to_string(),
            from_f64::<A>(self.trees.len() as f64)?,
        );
        metadata.insert(
            "window_size".to_string(),
            from_f64::<A>(self.window.len() as f64)?,
        );

        let result =
            result_from_score(score, self.threshold, AnomalyType::SpatialAnomaly, metadata)?;
        self.counters
            .record_prediction(result.is_anomaly, result.anomaly_score);
        Ok(result)
    }

    fn train(&mut self, training_data: &[StreamingDataPoint<A>]) -> Result<(), String> {
        let mut accepted = 0usize;
        for data_point in training_data {
            if let Some(features) = feature_vector(data_point) {
                self.push_observation(features);
                accepted += 1;
            }
        }
        if accepted == 0 {
            return Err("isolation forest: no usable training points".to_string());
        }
        self.fit_forest();
        Ok(())
    }

    fn update_incremental(&mut self, data_point: &StreamingDataPoint<A>) -> Result<(), String> {
        let features = feature_vector(data_point)
            .ok_or_else(|| "isolation forest: data point has no finite features".to_string())?;
        self.push_observation(features);
        Ok(())
    }

    fn record_outcome(&mut self, predicted_anomaly: bool, was_true_anomaly: bool) {
        self.counters
            .record_outcome(predicted_anomaly, was_true_anomaly);
    }

    fn get_performance_metrics(&self) -> Result<MLModelMetrics<A>, String> {
        self.counters
            .to_metrics(self.name(), self.training_time, self.mean_inference_time())
    }

    fn name(&self) -> String {
        "isolation_forest".to_string()
    }
}

impl<A: Float + Send + Sync> IsolationForestDetector<A> {
    fn mean_inference_time(&self) -> std::time::Duration {
        if self.inference_samples == 0 {
            std::time::Duration::ZERO
        } else {
            self.inference_time_total / self.inference_samples as u32
        }
    }
}

// ---------------------------------------------------------------------------
// Online one-class SVM (NORMA)
// ---------------------------------------------------------------------------

/// Online one-class SVM with an RBF kernel expansion.
///
/// Implements the NORMA stochastic-gradient rule for novelty detection: the
/// decision function `f(x) = sum_i alpha_i k(x_i, x)` is compared against a
/// learned offset `rho`, all coefficients decay by `(1 - eta * lambda)` on
/// every step, and a new support vector is admitted whenever the hinge
/// `rho - f(x)` is active. `rho` itself is learned by the same gradient rule
/// with the `nu` quantile target, so the detector converges towards flagging
/// roughly a `nu` fraction of the stream.
pub struct OneClassSvmDetector<A: Float + Send + Sync> {
    support_vectors: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    max_support_vectors: usize,
    /// Offset of the decision boundary.
    rho: f64,
    /// Target outlier fraction.
    nu: f64,
    /// Learning rate.
    eta: f64,
    /// Regularisation (coefficient decay).
    lambda: f64,
    /// RBF kernel width; `None` until it is estimated from real data.
    gamma: Option<f64>,
    /// Recent observations, used to estimate `gamma` by the median heuristic.
    window: VecDeque<Vec<f64>>,
    window_capacity: usize,
    updates: usize,
    counters: DetectionCounters,
    training_time: std::time::Duration,
    inference_time_total: std::time::Duration,
    inference_samples: usize,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> OneClassSvmDetector<A> {
    /// Creates a detector with a `nu` of 0.05 (target 5% outliers).
    pub fn new() -> Result<Self, String> {
        Self::with_parameters(0.05, 0.1, 0.01, 128)
    }

    /// Creates a detector with explicit hyper-parameters.
    pub fn with_parameters(
        nu: f64,
        eta: f64,
        lambda: f64,
        max_support_vectors: usize,
    ) -> Result<Self, String> {
        if !(nu > 0.0 && nu < 1.0) {
            return Err(format!("one-class SVM nu must lie in (0, 1), got {nu}"));
        }
        if !(eta.is_finite() && eta > 0.0) {
            return Err(format!(
                "one-class SVM learning rate must be positive, got {eta}"
            ));
        }
        if max_support_vectors < 1 {
            return Err("one-class SVM needs at least one support vector slot".to_string());
        }
        Ok(Self {
            support_vectors: Vec::new(),
            coefficients: Vec::new(),
            max_support_vectors,
            rho: 0.0,
            nu,
            eta,
            lambda,
            gamma: None,
            window: VecDeque::with_capacity(DEFAULT_WINDOW),
            window_capacity: DEFAULT_WINDOW,
            updates: 0,
            counters: DetectionCounters::default(),
            training_time: std::time::Duration::ZERO,
            inference_time_total: std::time::Duration::ZERO,
            inference_samples: 0,
            _marker: std::marker::PhantomData,
        })
    }

    /// Number of support vectors currently retained.
    pub fn support_vector_count(&self) -> usize {
        self.support_vectors.len()
    }

    /// Learned decision offset.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Estimates the RBF width from the median pairwise distance of the
    /// window (the standard median heuristic). Returns `None` while there is
    /// not enough data to estimate anything.
    fn estimate_gamma(&self) -> Option<f64> {
        if self.window.len() < 4 {
            return None;
        }
        let points: Vec<&Vec<f64>> = self.window.iter().collect();
        let mut distances = Vec::with_capacity(points.len());
        for i in 1..points.len() {
            distances.push(squared_distance(points[i - 1], points[i]));
        }
        let median = stats::median(&distances)?;
        if median > 0.0 {
            Some(1.0 / median)
        } else {
            None
        }
    }

    fn kernel(&self, a: &[f64], b: &[f64], gamma: f64) -> f64 {
        (-gamma * squared_distance(a, b)).exp()
    }

    /// Decision-function value `f(x)`, or `None` before the kernel width has
    /// been estimated.
    pub fn decision_value(&self, features: &[f64]) -> Option<f64> {
        let gamma = self.gamma?;
        if self.support_vectors.is_empty() {
            return Some(0.0);
        }
        Some(
            self.support_vectors
                .iter()
                .zip(self.coefficients.iter())
                .map(|(sv, &alpha)| alpha * self.kernel(sv, features, gamma))
                .sum(),
        )
    }

    fn push_observation(&mut self, features: Vec<f64>) {
        if self.window.len() >= self.window_capacity {
            self.window.pop_front();
        }
        self.window.push_back(features);
        if self.gamma.is_none() {
            self.gamma = self.estimate_gamma();
        }
    }

    /// One NORMA gradient step.
    fn learn_one(&mut self, features: &[f64]) {
        let Some(gamma) = self.gamma else { return };
        let started = Instant::now();

        let f_x: f64 = self
            .support_vectors
            .iter()
            .zip(self.coefficients.iter())
            .map(|(sv, &alpha)| alpha * self.kernel(sv, features, gamma))
            .sum();

        // Coefficient decay from the regulariser.
        let decay = 1.0 - self.eta * self.lambda;
        for alpha in &mut self.coefficients {
            *alpha *= decay;
        }

        if f_x < self.rho {
            // Hinge is active: admit the point as a support vector and lower
            // the offset towards it.
            self.support_vectors.push(features.to_vec());
            self.coefficients.push(self.eta);
            self.rho -= self.eta * (1.0 - self.nu);
        } else {
            self.rho += self.eta * self.nu;
        }

        // Bound the expansion by dropping the least influential vector.
        while self.support_vectors.len() > self.max_support_vectors {
            let mut weakest = 0usize;
            let mut weakest_magnitude = f64::INFINITY;
            for (index, alpha) in self.coefficients.iter().enumerate() {
                if alpha.abs() < weakest_magnitude {
                    weakest_magnitude = alpha.abs();
                    weakest = index;
                }
            }
            self.support_vectors.remove(weakest);
            self.coefficients.remove(weakest);
        }

        self.updates += 1;
        self.training_time += started.elapsed();
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> MLAnomalyDetector<A>
    for OneClassSvmDetector<A>
{
    fn detect_anomaly(
        &mut self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyDetectionResult<A>, String> {
        let features = feature_vector(data_point)
            .ok_or_else(|| "one-class SVM: data point has no finite features".to_string())?;

        let started = Instant::now();
        let Some(f_x) = self.decision_value(&features) else {
            let mut metadata = HashMap::new();
            metadata.insert(
                "window_size".to_string(),
                from_f64::<A>(self.window.len() as f64)?,
            );
            return Ok(AnomalyDetectionResult {
                is_anomaly: false,
                anomaly_score: A::zero(),
                confidence: A::zero(),
                anomaly_type: None,
                severity: AnomalySeverity::Low,
                metadata,
            });
        };
        self.inference_time_total += started.elapsed();
        self.inference_samples += 1;

        // Novelty score: how far the point falls *below* the learned boundary.
        let score = (self.rho - f_x).max(0.0);
        let is_anomaly = self.updates > 0 && f_x < self.rho;
        // The natural scale for the margin is the offset magnitude itself.
        let scale = self.rho.abs().max(self.eta);
        let confidence = (score / scale).min(1.0);

        let mut metadata = HashMap::new();
        metadata.insert("decision_value".to_string(), from_f64::<A>(f_x)?);
        metadata.insert("rho".to_string(), from_f64::<A>(self.rho)?);
        metadata.insert(
            "support_vectors".to_string(),
            from_f64::<A>(self.support_vectors.len() as f64)?,
        );

        let result = AnomalyDetectionResult {
            is_anomaly,
            anomaly_score: from_f64(score)?,
            confidence: from_f64(confidence)?,
            anomaly_type: is_anomaly.then_some(AnomalyType::PointAnomaly),
            severity: severity_for(score, scale),
            metadata,
        };
        self.counters
            .record_prediction(result.is_anomaly, result.anomaly_score);
        Ok(result)
    }

    fn train(&mut self, training_data: &[StreamingDataPoint<A>]) -> Result<(), String> {
        let mut accepted = 0usize;
        for data_point in training_data {
            if let Some(features) = feature_vector(data_point) {
                self.push_observation(features.clone());
                self.learn_one(&features);
                accepted += 1;
            }
        }
        if accepted == 0 {
            return Err("one-class SVM: no usable training points".to_string());
        }
        Ok(())
    }

    fn update_incremental(&mut self, data_point: &StreamingDataPoint<A>) -> Result<(), String> {
        let features = feature_vector(data_point)
            .ok_or_else(|| "one-class SVM: data point has no finite features".to_string())?;
        self.push_observation(features.clone());
        self.learn_one(&features);
        Ok(())
    }

    fn record_outcome(&mut self, predicted_anomaly: bool, was_true_anomaly: bool) {
        self.counters
            .record_outcome(predicted_anomaly, was_true_anomaly);
    }

    fn get_performance_metrics(&self) -> Result<MLModelMetrics<A>, String> {
        let mean_inference = if self.inference_samples == 0 {
            std::time::Duration::ZERO
        } else {
            self.inference_time_total / self.inference_samples as u32
        };
        self.counters
            .to_metrics(self.name(), self.training_time, mean_inference)
    }

    fn name(&self) -> String {
        "one_class_svm".to_string()
    }
}

// ---------------------------------------------------------------------------
// Local Outlier Factor
// ---------------------------------------------------------------------------

/// Local Outlier Factor detector over a sliding window.
///
/// For a query point `p` the detector finds its `k` nearest window neighbours,
/// forms the reachability distance `reach_k(p, o) = max(k-dist(o), d(p, o))`,
/// derives the local reachability density
/// `lrd(p) = 1 / mean_o reach_k(p, o)`, and reports
/// `LOF(p) = mean_o lrd(o) / lrd(p)`. A value near `1` means `p` sits in a
/// region as dense as its neighbours'; substantially above `1` means it is a
/// local outlier. Unlike the isolation forest this is a *local*, density-ratio
/// criterion, so it flags points that are only anomalous relative to their own
/// neighbourhood.
pub struct LofDetector<A: Float + Send + Sync> {
    window: VecDeque<Vec<f64>>,
    window_capacity: usize,
    k: usize,
    threshold: f64,
    counters: DetectionCounters,
    inference_time_total: std::time::Duration,
    inference_samples: usize,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> LofDetector<A> {
    /// Creates a detector with `k = 20` and the conventional `LOF > 1.5`
    /// outlier threshold.
    pub fn new() -> Result<Self, String> {
        Self::with_parameters(20, 1.5, DEFAULT_WINDOW)
    }

    /// Creates a detector with explicit parameters.
    pub fn with_parameters(
        k: usize,
        threshold: f64,
        window_capacity: usize,
    ) -> Result<Self, String> {
        if k < 2 {
            return Err(format!("LOF needs k >= 2, got {k}"));
        }
        if !(threshold.is_finite() && threshold > 0.0) {
            return Err(format!("LOF threshold must be positive, got {threshold}"));
        }
        if window_capacity <= k {
            return Err("LOF window must be larger than k".to_string());
        }
        Ok(Self {
            window: VecDeque::with_capacity(window_capacity),
            window_capacity,
            k,
            threshold,
            counters: DetectionCounters::default(),
            inference_time_total: std::time::Duration::ZERO,
            inference_samples: 0,
            _marker: std::marker::PhantomData,
        })
    }

    fn push_observation(&mut self, features: Vec<f64>) {
        if self.window.len() >= self.window_capacity {
            self.window.pop_front();
        }
        self.window.push_back(features);
    }

    /// Distances from `point` to every window entry, excluding an optional
    /// index (used so a window member does not count itself as its own
    /// neighbour).
    fn neighbour_distances(&self, point: &[f64], exclude: Option<usize>) -> Vec<(usize, f64)> {
        let mut distances: Vec<(usize, f64)> = self
            .window
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != exclude)
            .map(|(index, other)| (index, squared_distance(point, other).sqrt()))
            .collect();
        distances.sort_by(|a, b| stats::total_order(&a.1, &b.1));
        distances
    }

    /// `k`-distance of the window member at `index`.
    fn k_distance(&self, index: usize) -> Option<f64> {
        let point = self.window.get(index)?;
        let distances = self.neighbour_distances(point, Some(index));
        distances.get(self.k - 1).map(|(_, d)| *d)
    }

    /// Local reachability density of the window member at `index`.
    fn lrd_of_member(&self, index: usize) -> Option<f64> {
        let point = self.window.get(index)?;
        let distances = self.neighbour_distances(point, Some(index));
        if distances.len() < self.k {
            return None;
        }
        let mut total = 0.0_f64;
        for &(neighbour_index, distance) in distances.iter().take(self.k) {
            let neighbour_k_distance = self.k_distance(neighbour_index)?;
            total += neighbour_k_distance.max(distance);
        }
        let mean_reachability = total / self.k as f64;
        if mean_reachability <= 0.0 {
            // Duplicated points: infinite density. Report it as a very large
            // but finite density so downstream ratios stay well-defined.
            return Some(f64::MAX.sqrt());
        }
        Some(1.0 / mean_reachability)
    }

    /// LOF score of an arbitrary query point against the current window, or
    /// `None` while the window holds fewer than `k + 1` points.
    pub fn score(&self, point: &[f64]) -> Option<f64> {
        if self.window.len() <= self.k {
            return None;
        }
        let distances = self.neighbour_distances(point, None);
        if distances.len() < self.k {
            return None;
        }

        let mut reachability_total = 0.0_f64;
        let mut neighbour_lrd_total = 0.0_f64;
        for &(neighbour_index, distance) in distances.iter().take(self.k) {
            let neighbour_k_distance = self.k_distance(neighbour_index)?;
            reachability_total += neighbour_k_distance.max(distance);
            neighbour_lrd_total += self.lrd_of_member(neighbour_index)?;
        }

        let mean_reachability = reachability_total / self.k as f64;
        let lrd_point = if mean_reachability <= 0.0 {
            f64::MAX.sqrt()
        } else {
            1.0 / mean_reachability
        };
        let mean_neighbour_lrd = neighbour_lrd_total / self.k as f64;

        if lrd_point <= 0.0 {
            return None;
        }
        Some(mean_neighbour_lrd / lrd_point)
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> MLAnomalyDetector<A>
    for LofDetector<A>
{
    fn detect_anomaly(
        &mut self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyDetectionResult<A>, String> {
        let features = feature_vector(data_point)
            .ok_or_else(|| "LOF: data point has no finite features".to_string())?;

        let started = Instant::now();
        let Some(score) = self.score(&features) else {
            let mut metadata = HashMap::new();
            metadata.insert(
                "window_size".to_string(),
                from_f64::<A>(self.window.len() as f64)?,
            );
            return Ok(AnomalyDetectionResult {
                is_anomaly: false,
                anomaly_score: A::zero(),
                confidence: A::zero(),
                anomaly_type: None,
                severity: AnomalySeverity::Low,
                metadata,
            });
        };
        self.inference_time_total += started.elapsed();
        self.inference_samples += 1;

        let mut metadata = HashMap::new();
        metadata.insert("k".to_string(), from_f64::<A>(self.k as f64)?);
        metadata.insert(
            "window_size".to_string(),
            from_f64::<A>(self.window.len() as f64)?,
        );

        let result = result_from_score(
            score,
            self.threshold,
            AnomalyType::ContextualAnomaly,
            metadata,
        )?;
        self.counters
            .record_prediction(result.is_anomaly, result.anomaly_score);
        Ok(result)
    }

    fn train(&mut self, training_data: &[StreamingDataPoint<A>]) -> Result<(), String> {
        let mut accepted = 0usize;
        for data_point in training_data {
            if let Some(features) = feature_vector(data_point) {
                self.push_observation(features);
                accepted += 1;
            }
        }
        if accepted == 0 {
            return Err("LOF: no usable training points".to_string());
        }
        Ok(())
    }

    fn update_incremental(&mut self, data_point: &StreamingDataPoint<A>) -> Result<(), String> {
        let features = feature_vector(data_point)
            .ok_or_else(|| "LOF: data point has no finite features".to_string())?;
        self.push_observation(features);
        Ok(())
    }

    fn record_outcome(&mut self, predicted_anomaly: bool, was_true_anomaly: bool) {
        self.counters
            .record_outcome(predicted_anomaly, was_true_anomaly);
    }

    fn get_performance_metrics(&self) -> Result<MLModelMetrics<A>, String> {
        let mean_inference = if self.inference_samples == 0 {
            std::time::Duration::ZERO
        } else {
            self.inference_time_total / self.inference_samples as u32
        };
        // LOF is a lazy learner: there is no separate training phase to time.
        self.counters
            .to_metrics(self.name(), std::time::Duration::ZERO, mean_inference)
    }

    fn name(&self) -> String {
        "lof".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap as StdHashMap;

    fn point(features: Vec<f64>) -> StreamingDataPoint<f64> {
        StreamingDataPoint {
            features: Array1::from_vec(features),
            target: None,
            timestamp: Instant::now(),
            source_id: None,
            quality_score: 1.0,
            metadata: StdHashMap::new(),
        }
    }

    fn wobble(index: usize) -> f64 {
        ((index as f64) * 0.7548776662).fract() - 0.5
    }

    /// A dense cluster of inliers plus one far-away point.
    fn cluster(n: usize) -> Vec<StreamingDataPoint<f64>> {
        (0..n)
            .map(|i| point(vec![wobble(i), wobble(i * 7 + 3)]))
            .collect()
    }

    /// A2: `IsolationForestDetector::detect_anomaly` returned the constant
    /// `0.3` for every input. A real isolation forest must score a far-away
    /// point strictly higher than a point inside the training cluster.
    #[test]
    fn isolation_forest_scores_outliers_above_inliers() {
        let mut detector = IsolationForestDetector::<f64>::new().expect("detector");
        detector.train(&cluster(400)).expect("train");
        assert!(detector.is_fitted());

        let inlier_score = detector.score(&[0.0, 0.0]).expect("inlier score");
        let outlier_score = detector.score(&[50.0, -50.0]).expect("outlier score");

        assert!(
            outlier_score > inlier_score,
            "A2 regression: outlier scored {outlier_score} but inlier scored \
             {inlier_score} (a constant score would make these equal)"
        );
        // A constant 0.3 would fail this too.
        assert!(
            (inlier_score - outlier_score).abs() > 1e-6,
            "scores are indistinguishable — the detector is not scoring the input"
        );

        let outlier = MLAnomalyDetector::detect_anomaly(&mut detector, &point(vec![50.0, -50.0]))
            .expect("detect");
        assert!(outlier.is_anomaly, "far-away point must be flagged");

        let inlier = MLAnomalyDetector::detect_anomaly(&mut detector, &point(vec![0.0, 0.0]))
            .expect("detect");
        assert!(!inlier.is_anomaly, "cluster centre must not be flagged");
    }

    /// A2: `OneClassSVMDetector` always returned `is_anomaly: false` with score
    /// `0.2`. A real online one-class SVM must learn a boundary and place a
    /// distant point outside it.
    #[test]
    fn one_class_svm_learns_a_boundary_and_flags_novelty() {
        let mut detector = OneClassSvmDetector::<f64>::new().expect("detector");
        detector.train(&cluster(400)).expect("train");

        assert!(
            detector.support_vector_count() > 0,
            "A2 regression: the SVM admitted no support vectors — it never trained"
        );

        let inlier_value = detector.decision_value(&[0.0, 0.0]).expect("inlier");
        let outlier_value = detector.decision_value(&[40.0, 40.0]).expect("outlier");
        assert!(
            inlier_value > outlier_value,
            "the decision function must rate a cluster point above a distant one \
             ({inlier_value} vs {outlier_value})"
        );

        let outlier = MLAnomalyDetector::detect_anomaly(&mut detector, &point(vec![40.0, 40.0]))
            .expect("detect");
        assert!(
            outlier.is_anomaly,
            "A2 regression: the SVM still never reports an anomaly (score={})",
            outlier.anomaly_score
        );
        assert!(
            (outlier.anomaly_score - 0.2).abs() > 1e-9,
            "score {} looks like the old hard-coded 0.2",
            outlier.anomaly_score
        );
    }

    /// A2: `LOFDetector` always returned `is_anomaly: false` with score `0.1`.
    /// A real LOF must return ~1 for an inlier and clearly more for a point in
    /// a sparse region.
    #[test]
    fn lof_reports_density_ratio_near_one_for_inliers() {
        let mut detector = LofDetector::<f64>::with_parameters(10, 1.5, 256).expect("detector");
        detector.train(&cluster(200)).expect("train");

        let inlier = detector.score(&[0.0, 0.0]).expect("inlier score");
        let outlier = detector.score(&[30.0, 30.0]).expect("outlier score");

        assert!(
            inlier < 2.0,
            "LOF of a point inside the cluster should be near 1, got {inlier}"
        );
        assert!(
            outlier > inlier * 2.0,
            "LOF of a sparse-region point ({outlier}) must clearly exceed an \
             inlier's ({inlier})"
        );

        let result = MLAnomalyDetector::detect_anomaly(&mut detector, &point(vec![30.0, 30.0]))
            .expect("detect");
        assert!(result.is_anomaly);
        assert!(
            (result.anomaly_score - 0.1).abs() > 1e-9,
            "score {} looks like the old hard-coded 0.1",
            result.anomaly_score
        );
    }

    /// The three detectors must be genuinely different models: on identical
    /// data they must not agree numerically.
    #[test]
    fn the_three_ml_detectors_produce_distinct_scores() {
        let training = cluster(300);
        let probe = point(vec![12.0, -8.0]);

        let mut forest = IsolationForestDetector::<f64>::new().expect("forest");
        forest.train(&training).expect("train");
        let mut svm = OneClassSvmDetector::<f64>::new().expect("svm");
        svm.train(&training).expect("train");
        let mut lof = LofDetector::<f64>::with_parameters(10, 1.5, 512).expect("lof");
        lof.train(&training).expect("train");

        let forest_score = MLAnomalyDetector::detect_anomaly(&mut forest, &probe)
            .expect("forest")
            .anomaly_score;
        let svm_score = MLAnomalyDetector::detect_anomaly(&mut svm, &probe)
            .expect("svm")
            .anomaly_score;
        let lof_score = MLAnomalyDetector::detect_anomaly(&mut lof, &probe)
            .expect("lof")
            .anomaly_score;

        assert!((forest_score - svm_score).abs() > 1e-9);
        assert!((svm_score - lof_score).abs() > 1e-9);
        assert!((forest_score - lof_score).abs() > 1e-9);
    }

    /// A2: `get_performance_metrics` used to return `accuracy: 0.85`,
    /// `auc_roc: 0.88`, ... regardless of what the detector had actually done.
    /// It must now be an honest error until labelled feedback exists, and then
    /// reflect the real confusion matrix exactly.
    #[test]
    fn performance_metrics_require_real_labelled_feedback() {
        let mut detector = IsolationForestDetector::<f64>::new().expect("detector");
        detector.train(&cluster(200)).expect("train");

        assert!(
            detector.get_performance_metrics().is_err(),
            "A2 regression: metrics were reported without any labelled outcome"
        );

        // 3 true positives, 1 false positive, 1 false negative, 5 true negatives.
        for _ in 0..3 {
            MLAnomalyDetector::record_outcome(&mut detector, true, true);
        }
        MLAnomalyDetector::record_outcome(&mut detector, true, false);
        MLAnomalyDetector::record_outcome(&mut detector, false, true);
        for _ in 0..5 {
            MLAnomalyDetector::record_outcome(&mut detector, false, false);
        }

        let metrics = detector.get_performance_metrics().expect("metrics");
        // precision = 3/4, recall = 3/4, accuracy = 8/10, fpr = 1/6.
        assert!(
            (metrics.precision - 0.75).abs() < 1e-12,
            "precision {}",
            metrics.precision
        );
        assert!(
            (metrics.recall - 0.75).abs() < 1e-12,
            "recall {}",
            metrics.recall
        );
        assert!(
            (metrics.accuracy - 0.8).abs() < 1e-12,
            "accuracy {}",
            metrics.accuracy
        );
        assert!(
            (metrics.false_positive_rate - 1.0 / 6.0).abs() < 1e-12,
            "fpr {}",
            metrics.false_positive_rate
        );
        assert!(
            (metrics.f1_score - 0.75).abs() < 1e-12,
            "f1 {}",
            metrics.f1_score
        );
        assert!(
            (metrics.accuracy - 0.85).abs() > 1e-9,
            "accuracy is still the hard-coded 0.85"
        );
    }

    /// A detector that has not seen enough data must say so honestly rather
    /// than emitting a made-up score.
    #[test]
    fn cold_detectors_report_no_score_instead_of_a_constant() {
        let mut forest = IsolationForestDetector::<f64>::new().expect("forest");
        let cold =
            MLAnomalyDetector::detect_anomaly(&mut forest, &point(vec![1.0, 2.0])).expect("detect");
        assert!(!cold.is_anomaly);
        assert_eq!(cold.anomaly_score, 0.0);
        assert_eq!(cold.confidence, 0.0);

        let mut lof = LofDetector::<f64>::new().expect("lof");
        assert!(lof.score(&[0.0, 0.0]).is_none());
        let cold_lof =
            MLAnomalyDetector::detect_anomaly(&mut lof, &point(vec![1.0, 2.0])).expect("detect");
        assert!(!cold_lof.is_anomaly);
        assert_eq!(cold_lof.anomaly_score, 0.0);
    }

    /// A point with no finite features cannot be scored; that must be an
    /// error, not a silently fabricated verdict.
    #[test]
    fn non_finite_features_are_rejected() {
        let mut forest = IsolationForestDetector::<f64>::new().expect("forest");
        forest.train(&cluster(100)).expect("train");
        let broken = point(vec![f64::NAN, 1.0]);
        assert!(MLAnomalyDetector::detect_anomaly(&mut forest, &broken).is_err());
        assert!(MLAnomalyDetector::update_incremental(&mut forest, &broken).is_err());
    }
}
