//! Performance predictor for NAS evaluation
//!
//! Provides models for predicting optimizer performance without full evaluation.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Mutex;
use std::time::Instant;

use super::types::*;
use crate::error::{OptimError, Result};
use crate::nas_engine::results::EvaluationResults;
use crate::{EvaluationConfig, EvaluationMetric, OptimizerArchitecture};

/// Fixed length of the architecture feature vector used by the linear predictor.
///
/// The feature layout is deterministic (see [`PerformancePredictor::extract_features`])
/// and includes a leading bias/intercept term so the model can fit a constant offset.
const FEATURE_DIM: usize = 12;

/// Performance predictor
///
/// Implements a genuinely-learned, feature-based online ridge-regression predictor.
/// An [`OptimizerArchitecture`] is mapped to a fixed-length numeric feature vector and
/// scored by a linear model squashed through a logistic function into `[0, 1]`. The
/// model is updated online from observed evaluation results.
///
/// # Removed scaffolding
///
/// Earlier revisions carried a large tree of neural-network / feature-pipeline /
/// prediction-cache types around this model: `ModelArchitecture`,
/// `RegularizationParameters`, `LearningRateSchedule`, `EarlyStoppingState`,
/// `FeatureExtractor`, `FeatureEngineeringPipeline`, `FeatureScaling`,
/// `FeatureInteractions`, `PolynomialFeatures`, `FeatureSelection`, `FeatureCache`,
/// `TrainingMetadata`, `ResourceUsageRecord`, `DataSplits`, `PredictionCache`,
/// `PredictionResult`, `CacheStatistics`, `CacheConfig`, `UncertaintyEstimator`,
/// `UncertaintyParameters`, `CalibrationData` and `CalibrationCurve`. Every one of
/// them was constructed once, stored, and never read again — not one influenced a
/// single prediction, and none could be built or inspected from outside the module.
/// They are gone rather than left as a public API that does nothing; what remains is
/// exactly the state the model uses, and all of it is readable through the accessors
/// below.
#[derive(Debug)]
pub struct PerformancePredictor<T: Float + Debug + Send + Sync + 'static> {
    /// Predictor model
    predictor_model: PredictorModel<T>,

    /// Training data
    training_data: PredictorTrainingData<T>,

    /// Confidence level used to scale the reported interval, in `(0, 1)`.
    confidence_level: T,

    /// Online learning rate for the ridge / SGD weight update.
    learning_rate: T,

    /// L2 (ridge) regularization strength applied during the online update.
    ridge_lambda: T,

    /// Pending feature vectors recorded during `predict_performance`.
    ///
    /// Because `predict_performance` takes `&self` and [`EvaluationResults`] does not
    /// carry the originating architecture, the features observed at prediction time are
    /// buffered here (FIFO). `update_with_results` then pairs each incoming result's
    /// `overall_score` target with the corresponding buffered feature vector to form a
    /// genuine `(features -> target)` training pair. A `Mutex` is used (rather than
    /// `RefCell`) so the predictor remains `Sync`. Assumption: results are supplied to
    /// `update_with_results` in the same order their architectures were predicted; any
    /// surplus results beyond the buffered features are skipped (no architecture to
    /// reconstruct features from).
    pending_features: Mutex<Vec<Array1<T>>>,
}

/// Predictor model
#[derive(Debug)]
pub struct PredictorModel<T: Float + Debug + Send + Sync + 'static> {
    /// Model type
    model_type: PredictorModelType,

    /// Model parameters
    parameters: ModelParameters<T>,

    /// Training state
    training_state: ModelTrainingState<T>,
}

/// Model parameters
///
/// The intercept is carried by feature `[0]` of the feature vector rather than by a
/// separate bias vector, so the weight matrix is the model's entire parameter set.
#[derive(Debug)]
pub struct ModelParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Linear weights: a single `[FEATURE_DIM, 1]` matrix.
    weights: Vec<Array2<T>>,
}

/// Model training state
#[derive(Debug)]
pub struct ModelTrainingState<T: Float + Debug + Send + Sync + 'static> {
    /// Number of online updates applied so far
    current_epoch: usize,

    /// Squared error of every online update, in application order
    loss_history: Vec<T>,
}

/// Predictor training data
#[derive(Debug)]
pub struct PredictorTrainingData<T: Float + Debug + Send + Sync + 'static> {
    /// Architecture features
    architecture_features: Vec<Array1<T>>,

    /// Performance targets
    performance_targets: Vec<T>,
}

// Implementations

impl<T: Float + Debug + Default + Send + Sync> PerformancePredictor<T> {
    /// Default confidence level used for the reported prediction interval.
    const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

    pub fn new(_config: &EvaluationConfig) -> Result<Self> {
        Ok(Self {
            predictor_model: PredictorModel::new(),
            training_data: PredictorTrainingData::new(),
            confidence_level: Self::scalar(Self::DEFAULT_CONFIDENCE_LEVEL),
            learning_rate: Self::scalar(0.05),
            ridge_lambda: Self::scalar(1e-4),
            pending_features: Mutex::new(Vec::new()),
        })
    }

    /// Model family this predictor implements.
    pub fn model_type(&self) -> &PredictorModelType {
        &self.predictor_model.model_type
    }

    /// Number of online updates applied so far.
    pub fn epochs_trained(&self) -> usize {
        self.predictor_model.training_state.current_epoch
    }

    /// Squared error of every online update, in application order.
    pub fn training_loss_history(&self) -> &[T] {
        &self.predictor_model.training_state.loss_history
    }

    /// Number of `(features -> target)` pairs the model has been trained on.
    pub fn observation_count(&self) -> usize {
        self.num_observations()
    }

    /// Root-mean-square error of the *current* weights over every training pair
    /// recorded so far, or `None` before any pair has been observed.
    ///
    /// This is a resubstitution error, not a held-out estimate: it says how well
    /// the model now fits what it has already seen. It is what makes the stored
    /// training features and targets observable rather than write-only.
    pub fn training_rmse(&self) -> Option<T> {
        let targets = &self.training_data.performance_targets;
        let features = &self.training_data.architecture_features;
        let n = targets.len().min(features.len());
        if n == 0 {
            return None;
        }
        let mut sum_sq = T::zero();
        for i in 0..n {
            let error = Self::sigmoid(self.raw_score(&features[i])) - targets[i];
            sum_sq = sum_sq + error * error;
        }
        let count: T = NumCast::from(n as f64)?;
        Some((sum_sq / count).sqrt())
    }

    /// Confidence level used to widen the reported prediction interval.
    pub fn confidence_level(&self) -> T {
        self.confidence_level
    }

    /// Set the confidence level used for the reported prediction interval.
    ///
    /// Rejects values outside the open interval `(0, 1)`: a level of `0` or `1` is
    /// not a confidence level, and silently clamping it would make the reported
    /// interval mean something other than what the caller asked for.
    pub fn set_confidence_level(&mut self, level: T) -> Result<()> {
        if !(level > T::zero() && level < T::one()) {
            return Err(OptimError::InvalidParameter(format!(
                "confidence level must lie strictly inside (0, 1), got {:?}",
                level
            )));
        }
        self.confidence_level = level;
        Ok(())
    }

    /// Convert a finite `f64` constant into `T`, falling back to zero on failure.
    fn scalar(value: f64) -> T {
        NumCast::from(value).unwrap_or_else(T::zero)
    }

    /// Numerically-stable logistic squashing function mapping a raw score to `(0, 1)`.
    fn sigmoid(raw: T) -> T {
        let one = T::one();
        if raw >= T::zero() {
            let z = (-raw).exp();
            one / (one + z)
        } else {
            let z = raw.exp();
            z / (one + z)
        }
    }

    /// Clamp a value into the closed unit interval `[0, 1]`.
    fn clamp_unit(value: T) -> T {
        value.max(T::zero()).min(T::one())
    }

    /// Deterministic, order-independent content hash of the architecture structure,
    /// normalized into `[0, 1]`. Provides a stable identity-derived feature without
    /// depending on `HashMap` iteration order.
    fn structure_signature(architecture: &OptimizerArchitecture<T>) -> T {
        // FNV-1a over a canonicalized (sorted) view of the structure entries so the
        // result is independent of vector ordering noise yet sensitive to content.
        let mut entries: Vec<&str> = architecture.components.iter().map(|s| s.as_str()).collect();
        entries.sort_unstable();

        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for entry in entries {
            for byte in entry.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            // Separator so ["ab","c"] and ["a","bc"] differ.
            hash ^= 0x1f;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }

        // Map the high 24 bits into [0, 1].
        let bucket = (hash >> 40) as f64;
        let max = ((1u64 << 24) - 1) as f64;
        Self::scalar(bucket / max)
    }

    /// Map an [`OptimizerArchitecture`] to a fixed-length numeric feature vector.
    ///
    /// The layout is deterministic and length `FEATURE_DIM`:
    /// * `[0]`  bias / intercept (always `1`)
    /// * `[1]`  squashed component count (`structure.len()`)
    /// * `[2]`  squashed distinct-component-type count
    /// * `[3]`  squashed parameter count (`parameters.len()`)
    /// * `[4]`  mean parameter value
    /// * `[5]`  squashed sum of parameter values
    /// * `[6]`  maximum parameter value
    /// * `[7]`  minimum parameter value
    /// * `[8]`  normalized structure content signature
    /// * `[9]`  squashed `id` length
    /// * `[10]` density of recognized optimizer keywords in the structure
    /// * `[11]` learning-rate hyperparameter (if present, else `0`)
    fn extract_features(&self, architecture: &OptimizerArchitecture<T>) -> Array1<T> {
        let mut features = Array1::zeros(FEATURE_DIM);

        // [0] bias term.
        features[0] = T::one();

        // [1] component count (compressed via a saturating transform to keep the
        // linear model well-conditioned for large architectures).
        let n_components = architecture.components.len();
        features[1] = Self::saturating_count(n_components);

        // [2] distinct component types.
        let mut distinct: Vec<&str> = architecture.components.iter().map(|s| s.as_str()).collect();
        distinct.sort_unstable();
        distinct.dedup();
        features[2] = Self::saturating_count(distinct.len());

        // Numeric attributes are the union of the architecture's `parameters`
        // and `hyperparameters` maps: search strategies populate one or the
        // other depending on how the candidate was produced, and the predictor
        // must see the same architecture either way.
        let numeric_values: Vec<f64> = architecture
            .parameters
            .values()
            .chain(architecture.hyperparameters.values())
            .map(|v| v.to_f64().unwrap_or(0.0))
            .filter(|v| v.is_finite())
            .collect();

        // [3] parameter count.
        features[3] = Self::saturating_count(numeric_values.len());

        // [4..8] parameter value statistics.
        if numeric_values.is_empty() {
            features[4] = T::zero();
            features[5] = T::zero();
            features[6] = T::zero();
            features[7] = T::zero();
        } else {
            let mut sum = 0.0_f64;
            let mut max = f64::NEG_INFINITY;
            let mut min = f64::INFINITY;
            for &v in &numeric_values {
                sum += v;
                if v > max {
                    max = v;
                }
                if v < min {
                    min = v;
                }
            }
            let count = numeric_values.len() as f64;
            features[4] = Self::scalar(sum / count);
            features[5] = Self::scalar((sum.abs()).tanh()); // squashed magnitude
            features[6] = Self::scalar(max);
            features[7] = Self::scalar(min);
        }

        // [8] structure content signature.
        features[8] = Self::structure_signature(architecture);

        // [9] identifier length (proxy for descriptive complexity).
        features[9] = Self::saturating_count(architecture.architecture_id.len());

        // [10] recognized optimizer keyword density.
        if n_components == 0 {
            features[10] = T::zero();
        } else {
            const KEYWORDS: [&str; 7] = [
                "adam", "sgd", "momentum", "rmsprop", "adagrad", "adamw", "lamb",
            ];
            let mut hits = 0usize;
            for entry in &architecture.components {
                let lower = entry.to_ascii_lowercase();
                if KEYWORDS.iter().any(|kw| lower.contains(kw)) {
                    hits += 1;
                }
            }
            features[10] = Self::scalar(hits as f64 / n_components as f64);
        }

        // [11] learning-rate hyperparameter if exposed under a common key.
        let lr = ["learning_rate", "lr"]
            .iter()
            .find_map(|key| {
                architecture
                    .hyperparameters
                    .get(*key)
                    .or_else(|| architecture.parameters.get(*key))
            })
            .and_then(|v| v.to_f64())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        features[11] = Self::scalar(lr);

        features
    }

    /// Saturating, monotonic transform of a non-negative count into `[0, 1)`:
    /// `n -> n / (n + 1)`. Keeps unbounded counts inside a bounded feature range.
    fn saturating_count(n: usize) -> T {
        let nf = n as f64;
        Self::scalar(nf / (nf + 1.0))
    }

    /// Compute the linear raw score `w . x` from the stored weight vector.
    fn raw_score(&self, features: &Array1<T>) -> T {
        let weights = &self.predictor_model.parameters.weights[0];
        let mut acc = T::zero();
        for i in 0..FEATURE_DIM {
            acc = acc + weights[[i, 0]] * features[i];
        }
        acc
    }

    /// Number of `(features -> target)` pairs the model has been trained on so far.
    fn num_observations(&self) -> usize {
        self.training_data.performance_targets.len()
    }

    /// Estimate the half-width of the prediction confidence interval.
    ///
    /// The interval shrinks as more observations accumulate (a variance proxy:
    /// `base / sqrt(1 + n)`) and is scaled by the configured confidence level so a
    /// higher requested confidence yields a wider interval. With no training data the
    /// interval is at its widest, reflecting high epistemic uncertainty.
    fn confidence_half_width(&self) -> T {
        let n = self.num_observations();
        let base = Self::scalar(0.25);
        let denom = Self::scalar((1.0 + n as f64).sqrt());
        let spread = base / denom;
        // Scale by the requested confidence level (e.g. 0.95 -> wider than 0.5).
        let scale = T::one() + self.confidence_level;
        spread * scale
    }

    pub fn predict_performance(
        &self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<EvaluationResults<T>> {
        let start_time = Instant::now();

        // 1. Deterministic feature extraction.
        let features = self.extract_features(architecture);

        // 2. Linear model -> logistic squashing -> clamp into [0, 1].
        let raw = self.raw_score(&features);
        let overall_score = Self::clamp_unit(Self::sigmoid(raw));

        // 3. Uncertainty-derived confidence interval (wider with little data).
        let half_width = self.confidence_half_width();
        let lower = Self::clamp_unit(overall_score - half_width);
        let upper = Self::clamp_unit(overall_score + half_width);

        // 4. Record the features observed at prediction time so a subsequent
        //    `update_with_results` call can form a genuine training pair. Failure to
        //    acquire the lock is non-fatal for prediction (we simply skip buffering).
        if let Ok(mut pending) = self.pending_features.lock() {
            pending.push(features);
        }

        // 5. Populate the evaluation results.
        let mut metric_scores: HashMap<EvaluationMetric, T> = HashMap::new();
        metric_scores.insert(EvaluationMetric::FinalPerformance, overall_score);

        let mut confidence_intervals: HashMap<EvaluationMetric, (T, T)> = HashMap::new();
        confidence_intervals.insert(EvaluationMetric::FinalPerformance, (lower, upper));

        Ok(EvaluationResults {
            metric_scores,
            overall_score,
            confidence_intervals,
            evaluation_time: start_time.elapsed(),
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: HashMap::new(),
            training_trajectory: Vec::new(),
        })
    }

    pub fn update_with_results(&mut self, results: &[EvaluationResults<T>]) -> Result<()> {
        // Build the (features -> target) training pairs first, releasing the feature
        // buffer lock before mutating the model. Each successful result's
        // `overall_score` target is paired (FIFO) with the features recorded at the
        // corresponding `predict_performance` call. Surplus results with no buffered
        // features are skipped (no architecture to reconstruct features from).
        let mut pairs: Vec<(Array1<T>, T)> = Vec::new();
        {
            let mut pending = self.pending_features.lock().map_err(|_| {
                OptimError::EvaluationError("predictor feature buffer poisoned".into())
            })?;
            for result in results {
                if !result.success {
                    continue;
                }
                if pending.is_empty() {
                    break;
                }
                let features = pending.remove(0);
                let target = Self::clamp_unit(result.overall_score);
                pairs.push((features, target));
            }
        }

        let lr = self.learning_rate;
        let lambda = self.ridge_lambda;

        for (features, target) in pairs {
            // Online ridge / logistic SGD step:
            //   w <- w - lr * ((sigmoid(w . x) - y) * x + lambda * w)
            let raw = self.raw_score(&features);
            let prediction = Self::sigmoid(raw);
            let error = prediction - target;

            {
                let weights = &mut self.predictor_model.parameters.weights[0];
                for i in 0..FEATURE_DIM {
                    let grad = error * features[i] + lambda * weights[[i, 0]];
                    weights[[i, 0]] = weights[[i, 0]] - lr * grad;
                }
            }

            // Track the loss for diagnostics / training state.
            let loss = error * error;
            self.predictor_model.training_state.loss_history.push(loss);
            self.predictor_model.training_state.current_epoch += 1;

            // Persist the training pair. It is what `training_rmse` and the
            // observation-count-driven confidence interval are computed from.
            self.training_data.architecture_features.push(features);
            self.training_data.performance_targets.push(target);
        }

        Ok(())
    }
}

impl<T: Float + Debug + Default + Send + Sync> PredictorModel<T> {
    /// Build the linear (logistic-squashed) model this predictor uses: one
    /// `[FEATURE_DIM, 1]` weight matrix, zero-initialised.
    ///
    /// The previous body claimed a `NeuralNetwork` model type and populated a
    /// 64-128-64-1 layer stack, ReLU activations, dropout rates, an exponential
    /// learning-rate schedule and an early-stopping state — none of which any code
    /// path ever read; the model was, and is, linear.
    fn new() -> Self {
        Self {
            model_type: PredictorModelType::LinearRegression,
            parameters: ModelParameters {
                weights: vec![Array2::zeros((FEATURE_DIM, 1))],
            },
            training_state: ModelTrainingState {
                current_epoch: 0,
                loss_history: Vec::new(),
            },
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> PredictorTrainingData<T> {
    fn new() -> Self {
        Self {
            architecture_features: Vec::new(),
            performance_targets: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal evaluation configuration for the predictor.
    fn make_config() -> EvaluationConfig {
        EvaluationConfig {
            epochs: 1,
            batch_size: 8,
            learning_rate: 0.01,
            performance_prediction: true,
        }
    }

    /// Build a deterministic optimizer architecture for testing.
    fn make_architecture(id: &str) -> OptimizerArchitecture<f64> {
        let mut parameters = HashMap::new();
        parameters.insert("learning_rate".to_string(), 0.01_f64);
        parameters.insert("momentum".to_string(), 0.9_f64);
        parameters.insert("weight_decay".to_string(), 0.0001_f64);

        OptimizerArchitecture {
            architecture_id: id.to_string(),
            parameters,
            components: vec![
                "AdamW".to_string(),
                "CosineSchedule".to_string(),
                "GradientClipping".to_string(),
            ],
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
        }
    }

    #[test]
    fn test_feature_vector_is_fixed_length_and_deterministic() {
        let predictor = PerformancePredictor::<f64>::new(&make_config())
            .expect("predictor construction should succeed");
        let architecture = make_architecture("arch-features");

        let f1 = predictor.extract_features(&architecture);
        let f2 = predictor.extract_features(&architecture);

        assert_eq!(f1.len(), FEATURE_DIM);
        // Determinism: identical architecture -> identical features.
        for i in 0..FEATURE_DIM {
            assert!((f1[i] - f2[i]).abs() < 1e-12);
        }
        // Bias/intercept feature is always 1.
        assert!((f1[0] - 1.0).abs() < 1e-12);
        // All features remain finite.
        for i in 0..FEATURE_DIM {
            assert!(f1[i].is_finite());
        }
    }

    #[test]
    fn test_predict_returns_valid_results() {
        let predictor = PerformancePredictor::<f64>::new(&make_config())
            .expect("predictor construction should succeed");
        let architecture = make_architecture("arch-predict");

        let results = predictor
            .predict_performance(&architecture)
            .expect("prediction should succeed");

        // Score must be a valid probability in [0, 1].
        assert!(results.overall_score >= 0.0 && results.overall_score <= 1.0);
        assert!(results.success);
        assert!(results.error_message.is_none());

        // At least one metric score must be populated.
        assert!(!results.metric_scores.is_empty());
        let predicted = results
            .metric_scores
            .get(&EvaluationMetric::FinalPerformance)
            .copied()
            .expect("FinalPerformance metric should be present");
        assert!((predicted - results.overall_score).abs() < 1e-12);

        // A confidence interval must be present and well-formed within [0, 1].
        let (lo, hi) = results
            .confidence_intervals
            .get(&EvaluationMetric::FinalPerformance)
            .copied()
            .expect("confidence interval should be present");
        assert!(lo >= 0.0 && hi <= 1.0);
        assert!(lo <= hi);

        // Empty optional collections are valid initial state.
        assert!(results.cv_results.is_none());
        assert!(results.benchmark_results.is_empty());
        assert!(results.training_trajectory.is_empty());
    }

    #[test]
    fn test_confidence_interval_narrows_with_more_data() {
        let mut predictor = PerformancePredictor::<f64>::new(&make_config())
            .expect("predictor construction should succeed");
        let architecture = make_architecture("arch-uncertainty");

        let wide = predictor.confidence_half_width();

        // Feed several observations to accumulate samples.
        for _ in 0..25 {
            let _ = predictor
                .predict_performance(&architecture)
                .expect("prediction should succeed");
            let result = predictor
                .predict_performance(&architecture)
                .expect("prediction should succeed");
            predictor
                .update_with_results(std::slice::from_ref(&result))
                .expect("update should succeed");
        }

        let narrow = predictor.confidence_half_width();
        // More observed data must reduce epistemic uncertainty.
        assert!(narrow < wide);
    }

    #[test]
    fn test_model_learns_toward_high_target() {
        let mut predictor = PerformancePredictor::<f64>::new(&make_config())
            .expect("predictor construction should succeed");
        let architecture = make_architecture("arch-learn");

        // Baseline prediction before any training.
        let baseline = predictor
            .predict_performance(&architecture)
            .expect("prediction should succeed")
            .overall_score;

        let high_target = 0.95_f64;

        // Repeatedly present the SAME architecture with a consistently high target.
        // Each iteration: predict (buffers the architecture features), then update
        // with a synthetic result carrying the high target so the online ridge step
        // forms a genuine (features -> target) pair.
        for _ in 0..200 {
            let mut result = predictor
                .predict_performance(&architecture)
                .expect("prediction should succeed");
            result.overall_score = high_target;
            predictor
                .update_with_results(std::slice::from_ref(&result))
                .expect("update should succeed");
        }

        let learned = predictor
            .predict_performance(&architecture)
            .expect("prediction should succeed")
            .overall_score;

        // Learning must move the prediction upward toward the target.
        assert!(
            learned > baseline + 0.05,
            "prediction did not increase: baseline={baseline}, learned={learned}"
        );
        // And it should approach (without necessarily reaching) the target, within
        // a generous tolerance for the squashed online learner.
        assert!(
            (high_target - learned).abs() < 0.25,
            "prediction did not converge toward target: learned={learned}, target={high_target}"
        );
        // The learner must have recorded the training pairs it consumed.
        assert!(predictor.num_observations() >= 200);
    }

    #[test]
    fn test_update_skips_failed_results() {
        let mut predictor = PerformancePredictor::<f64>::new(&make_config())
            .expect("predictor construction should succeed");
        let architecture = make_architecture("arch-failed");

        let mut result = predictor
            .predict_performance(&architecture)
            .expect("prediction should succeed");
        result.success = false;
        result.error_message = Some("synthetic failure".to_string());
        result.overall_score = 0.99;

        predictor
            .update_with_results(std::slice::from_ref(&result))
            .expect("update should succeed");

        // Failed evaluations must not contribute training pairs.
        assert_eq!(predictor.num_observations(), 0);
    }

    #[test]
    fn test_recorded_training_state_is_readable_and_consistent() {
        let mut predictor = PerformancePredictor::<f64>::new(&make_config())
            .expect("predictor construction should succeed");
        let architecture = make_architecture("arch-state");

        // A fresh predictor has trained on nothing, and says so instead of
        // reporting a fabricated error.
        assert_eq!(
            predictor.model_type(),
            &PredictorModelType::LinearRegression
        );
        assert_eq!(predictor.epochs_trained(), 0);
        assert!(predictor.training_loss_history().is_empty());
        assert_eq!(predictor.observation_count(), 0);
        assert!(predictor.training_rmse().is_none());

        for _ in 0..40 {
            let mut result = predictor
                .predict_performance(&architecture)
                .expect("prediction should succeed");
            result.overall_score = 0.9;
            predictor
                .update_with_results(std::slice::from_ref(&result))
                .expect("update should succeed");
        }

        // Every applied update is counted once and leaves exactly one loss entry.
        assert_eq!(predictor.epochs_trained(), 40);
        assert_eq!(predictor.training_loss_history().len(), 40);
        assert_eq!(predictor.observation_count(), 40);

        let rmse = predictor
            .training_rmse()
            .expect("training RMSE is defined once pairs exist");
        assert!(rmse.is_finite());
        assert!((0.0..=1.0).contains(&rmse));
        // Fitting one repeated target must leave the model closer to it than a
        // zero-weight model was (sigmoid(0) = 0.5, so the initial error is 0.4).
        assert!(rmse < 0.4, "training did not reduce the fit error: {rmse}");
    }

    #[test]
    fn test_confidence_level_is_validated_and_widens_the_interval() {
        let mut predictor = PerformancePredictor::<f64>::new(&make_config())
            .expect("predictor construction should succeed");

        assert!((predictor.confidence_level() - 0.95).abs() < 1e-12);

        let wide = predictor.confidence_half_width();
        predictor
            .set_confidence_level(0.5)
            .expect("0.5 is a valid confidence level");
        let narrow = predictor.confidence_half_width();
        assert!(
            narrow < wide,
            "a lower confidence level must not widen the interval: {narrow} vs {wide}"
        );

        // Degenerate levels are rejected rather than silently clamped.
        for bad in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
            assert!(
                predictor.set_confidence_level(bad).is_err(),
                "confidence level {bad} must be rejected"
            );
        }
        // A rejected value must not have been stored.
        assert!((predictor.confidence_level() - 0.5).abs() < 1e-12);
    }
}
