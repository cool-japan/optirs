//! Performance predictor for NAS evaluation
//!
//! Provides models for predicting optimizer performance without full evaluation.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Mutex;
use std::time::{Instant, SystemTime};

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
#[derive(Debug)]
pub struct PerformancePredictor<T: Float + Debug + Send + Sync + 'static> {
    /// Predictor model
    predictor_model: PredictorModel<T>,

    /// Feature extractor
    feature_extractor: FeatureExtractor<T>,

    /// Training data
    training_data: PredictorTrainingData<T>,

    /// Prediction cache
    prediction_cache: PredictionCache<T>,

    /// Uncertainty estimator
    uncertainty_estimator: UncertaintyEstimator<T>,

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

    /// Model architecture
    architecture: ModelArchitecture,

    /// Training state
    training_state: ModelTrainingState<T>,
}

/// Model parameters
#[derive(Debug)]
pub struct ModelParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Weights
    weights: Vec<Array2<T>>,

    /// Biases
    biases: Vec<Array1<T>>,

    /// Hyperparameters
    hyperparameters: HashMap<String, T>,

    /// Regularization parameters
    regularization: RegularizationParameters<T>,
}

/// Model architecture specification
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    /// Layer sizes
    layer_sizes: Vec<usize>,

    /// Activation functions
    activations: Vec<ActivationFunction>,

    /// Dropout rates
    dropout_rates: Vec<f64>,

    /// Skip connections
    skip_connections: Vec<(usize, usize)>,
}

/// Regularization parameters
#[derive(Debug)]
pub struct RegularizationParameters<T: Float + Debug + Send + Sync + 'static> {
    /// L1 regularization strength
    l1_strength: T,

    /// L2 regularization strength
    l2_strength: T,

    /// Dropout probability
    dropout_prob: T,

    /// Batch normalization flag
    batch_norm: bool,
}

/// Model training state
#[derive(Debug)]
pub struct ModelTrainingState<T: Float + Debug + Send + Sync + 'static> {
    /// Current epoch
    current_epoch: usize,

    /// Training loss history
    loss_history: Vec<T>,

    /// Validation loss history
    validation_loss_history: Vec<T>,

    /// Learning rate schedule
    learning_rate_schedule: LearningRateSchedule<T>,

    /// Early stopping state
    early_stopping_state: EarlyStoppingState<T>,
}

/// Learning rate schedule
#[derive(Debug)]
pub struct LearningRateSchedule<T: Float + Debug + Send + Sync + 'static> {
    /// Schedule type
    schedule_type: ScheduleType,

    /// Initial learning rate
    initial_lr: T,

    /// Current learning rate
    current_lr: T,

    /// Schedule parameters
    parameters: HashMap<String, T>,
}

/// Early stopping state
#[derive(Debug)]
pub struct EarlyStoppingState<T: Float + Debug + Send + Sync + 'static> {
    /// Best validation loss
    best_val_loss: T,

    /// Patience counter
    patience_counter: usize,

    /// Maximum patience
    max_patience: usize,

    /// Should stop flag
    should_stop: bool,
}

/// Feature extractor for performance prediction
#[derive(Debug)]
pub struct FeatureExtractor<T: Float + Debug + Send + Sync + 'static> {
    /// Feature extraction methods
    extraction_methods: Vec<FeatureExtractionMethod>,

    /// Feature engineering pipeline
    engineering_pipeline: FeatureEngineeringPipeline<T>,

    /// Feature selection
    feature_selection: FeatureSelection<T>,

    /// Feature cache
    feature_cache: FeatureCache<T>,
}

/// Feature engineering pipeline
#[derive(Debug)]
pub struct FeatureEngineeringPipeline<T: Float + Debug + Send + Sync + 'static> {
    /// Normalization method
    normalization: NormalizationMethod,

    /// Feature scaling
    scaling: FeatureScaling<T>,

    /// Feature interactions
    interactions: FeatureInteractions,

    /// Polynomial features
    polynomial_features: PolynomialFeatures,
}

/// Feature scaling
#[derive(Debug)]
pub struct FeatureScaling<T: Float + Debug + Send + Sync + 'static> {
    /// Scaling method
    method: ScalingMethod,

    /// Scale parameters
    scale_params: HashMap<String, T>,

    /// Feature ranges
    feature_ranges: HashMap<String, (T, T)>,
}

/// Feature interactions
#[derive(Debug, Clone)]
pub struct FeatureInteractions {
    /// Interaction order
    interaction_order: usize,

    /// Include bias term
    include_bias: bool,

    /// Selected interactions
    selected_interactions: Vec<Vec<usize>>,
}

/// Polynomial features
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    /// Polynomial degree
    degree: usize,

    /// Include bias term
    include_bias: bool,

    /// Interaction only flag
    interaction_only: bool,
}

/// Feature selection
#[derive(Debug)]
pub struct FeatureSelection<T: Float + Debug + Send + Sync + 'static> {
    /// Selection method
    selection_method: FeatureSelectionMethod,

    /// Selection parameters
    parameters: HashMap<String, T>,

    /// Selected features
    selected_features: Vec<usize>,

    /// Feature importance scores
    importance_scores: Vec<T>,
}

/// Feature cache
#[derive(Debug)]
pub struct FeatureCache<T: Float + Debug + Send + Sync + 'static> {
    /// Cached features
    cached_features: HashMap<String, Array1<T>>,

    /// Cache hit rate
    hit_rate: f64,

    /// Cache size limit
    size_limit: usize,

    /// Eviction policy
    eviction_policy: CacheEvictionPolicy,
}

/// Predictor training data
#[derive(Debug)]
pub struct PredictorTrainingData<T: Float + Debug + Send + Sync + 'static> {
    /// Architecture features
    architecture_features: Vec<Array1<T>>,

    /// Performance targets
    performance_targets: Vec<T>,

    /// Training metadata
    metadata: Vec<TrainingMetadata>,

    /// Data splits
    data_splits: DataSplits,
}

/// Training metadata
#[derive(Debug, Clone)]
pub struct TrainingMetadata {
    /// Architecture ID
    architecture_id: String,

    /// Benchmark name
    benchmark_name: String,

    /// Evaluation timestamp
    timestamp: SystemTime,

    /// Resource usage
    resource_usage: ResourceUsageRecord,
}

/// Resource usage record
#[derive(Debug, Clone)]
pub struct ResourceUsageRecord {
    /// Memory usage (MB)
    memory_mb: f64,

    /// CPU time (seconds)
    cpu_time_seconds: f64,

    /// GPU time (seconds)
    gpu_time_seconds: f64,

    /// Energy consumption (kWh)
    energy_kwh: f64,
}

/// Data splits for training
#[derive(Debug, Clone)]
pub struct DataSplits {
    /// Training indices
    train_indices: Vec<usize>,

    /// Validation indices
    validation_indices: Vec<usize>,

    /// Test indices
    test_indices: Vec<usize>,

    /// Split ratios
    split_ratios: (f64, f64, f64),
}

/// Prediction cache
#[derive(Debug)]
pub struct PredictionCache<T: Float + Debug + Send + Sync + 'static> {
    /// Cached predictions
    predictions: HashMap<String, PredictionResult<T>>,

    /// Cache statistics
    statistics: CacheStatistics,

    /// Cache configuration
    config: CacheConfig,
}

/// Prediction result
#[derive(Debug, Clone)]
pub struct PredictionResult<T: Float + Debug + Send + Sync + 'static> {
    /// Predicted performance
    predicted_performance: T,

    /// Confidence interval
    confidence_interval: (T, T),

    /// Prediction uncertainty
    uncertainty: T,

    /// Feature importance
    feature_importance: Vec<T>,

    /// Prediction timestamp
    timestamp: Instant,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Total requests
    total_requests: usize,

    /// Cache hits
    cache_hits: usize,

    /// Cache misses
    cache_misses: usize,

    /// Hit rate
    hit_rate: f64,

    /// Average prediction time
    avg_prediction_time_ms: f64,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    max_size: usize,

    /// TTL for entries (seconds)
    ttl_seconds: u64,

    /// Eviction policy
    eviction_policy: CacheEvictionPolicy,

    /// Enable persistence
    enable_persistence: bool,
}

/// Uncertainty estimator
#[derive(Debug)]
pub struct UncertaintyEstimator<T: Float + Debug + Send + Sync + 'static> {
    /// Estimation method
    estimation_method: UncertaintyEstimationMethod,

    /// Model ensemble (if using ensemble methods)
    model_ensemble: Vec<PredictorModel<T>>,

    /// Uncertainty parameters
    parameters: UncertaintyParameters<T>,

    /// Calibration data
    calibration_data: CalibrationData<T>,
}

/// Uncertainty parameters
#[derive(Debug)]
pub struct UncertaintyParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Number of samples for MC methods
    num_samples: usize,

    /// Confidence level
    confidence_level: T,

    /// Calibration alpha
    calibration_alpha: T,

    /// Method-specific parameters
    method_params: HashMap<String, T>,
}

/// Calibration data
#[derive(Debug)]
pub struct CalibrationData<T: Float + Debug + Send + Sync + 'static> {
    /// Calibration predictions
    predictions: Vec<T>,

    /// Calibration targets
    targets: Vec<T>,

    /// Calibration scores
    scores: Vec<T>,

    /// Calibration curve
    calibration_curve: CalibrationCurve<T>,
}

/// Calibration curve
#[derive(Debug)]
pub struct CalibrationCurve<T: Float + Debug + Send + Sync + 'static> {
    /// Bin edges
    bin_edges: Vec<T>,

    /// Bin accuracies
    bin_accuracies: Vec<T>,

    /// Bin confidences
    bin_confidences: Vec<T>,

    /// Bin counts
    bin_counts: Vec<usize>,
}

// Implementations

impl<T: Float + Debug + Default + Send + Sync> PerformancePredictor<T> {
    pub fn new(_config: &EvaluationConfig) -> Result<Self> {
        let mut predictor_model = PredictorModel::new()?;
        // This predictor is a linear (logistic-squashed) model over a fixed-length
        // feature vector, so override the placeholder neural-net model type and
        // initialize a single weight matrix of shape [FEATURE_DIM, 1]. The existing
        // `ModelParameters.weights` field is reused to hold the linear weights.
        predictor_model.model_type = PredictorModelType::LinearRegression;
        predictor_model.parameters.weights = vec![Array2::zeros((FEATURE_DIM, 1))];
        predictor_model.parameters.biases = vec![Array1::zeros(1)];

        let learning_rate = Self::scalar(0.05);
        let ridge_lambda = Self::scalar(1e-4);

        Ok(Self {
            predictor_model,
            feature_extractor: FeatureExtractor::new()?,
            training_data: PredictorTrainingData::new(),
            prediction_cache: PredictionCache::new(),
            uncertainty_estimator: UncertaintyEstimator::new()?,
            learning_rate,
            ridge_lambda,
            pending_features: Mutex::new(Vec::new()),
        })
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
        let confidence = self.uncertainty_estimator.parameters.confidence_level;
        let scale = T::one() + confidence;
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

            // Persist the training pair and bump the sample count used by the
            // uncertainty estimate.
            self.uncertainty_estimator
                .calibration_data
                .predictions
                .push(prediction);
            self.uncertainty_estimator
                .calibration_data
                .targets
                .push(target);
            self.training_data.architecture_features.push(features);
            self.training_data.performance_targets.push(target);
        }

        Ok(())
    }
}

impl<T: Float + Debug + Default + Send + Sync> PredictorModel<T> {
    fn new() -> Result<Self> {
        Ok(Self {
            model_type: PredictorModelType::NeuralNetwork,
            parameters: ModelParameters {
                weights: Vec::new(),
                biases: Vec::new(),
                hyperparameters: HashMap::new(),
                regularization: RegularizationParameters {
                    l1_strength: T::zero(),
                    l2_strength: scirs2_core::numeric::NumCast::from(0.01)
                        .unwrap_or_else(|| T::zero()),
                    dropout_prob: scirs2_core::numeric::NumCast::from(0.1)
                        .unwrap_or_else(|| T::zero()),
                    batch_norm: true,
                },
            },
            architecture: ModelArchitecture {
                layer_sizes: vec![64, 128, 64, 1],
                activations: vec![ActivationFunction::ReLU; 3],
                dropout_rates: vec![0.1, 0.2, 0.1],
                skip_connections: Vec::new(),
            },
            training_state: ModelTrainingState {
                current_epoch: 0,
                loss_history: Vec::new(),
                validation_loss_history: Vec::new(),
                learning_rate_schedule: LearningRateSchedule {
                    schedule_type: ScheduleType::Exponential,
                    initial_lr: scirs2_core::numeric::NumCast::from(0.001)
                        .unwrap_or_else(|| T::zero()),
                    current_lr: scirs2_core::numeric::NumCast::from(0.001)
                        .unwrap_or_else(|| T::zero()),
                    parameters: HashMap::new(),
                },
                early_stopping_state: EarlyStoppingState {
                    best_val_loss: T::infinity(),
                    patience_counter: 0,
                    max_patience: 10,
                    should_stop: false,
                },
            },
        })
    }
}

impl<T: Float + Debug + Default + Send + Sync> FeatureExtractor<T> {
    fn new() -> Result<Self> {
        Ok(Self {
            extraction_methods: vec![
                FeatureExtractionMethod::ArchitectureEmbedding,
                FeatureExtractionMethod::HyperparameterEncoding,
            ],
            engineering_pipeline: FeatureEngineeringPipeline {
                normalization: NormalizationMethod::ZScore,
                scaling: FeatureScaling {
                    method: ScalingMethod::Standard,
                    scale_params: HashMap::new(),
                    feature_ranges: HashMap::new(),
                },
                interactions: FeatureInteractions {
                    interaction_order: 2,
                    include_bias: true,
                    selected_interactions: Vec::new(),
                },
                polynomial_features: PolynomialFeatures {
                    degree: 2,
                    include_bias: true,
                    interaction_only: false,
                },
            },
            feature_selection: FeatureSelection {
                selection_method: FeatureSelectionMethod::VarianceThreshold,
                parameters: HashMap::new(),
                selected_features: Vec::new(),
                importance_scores: Vec::new(),
            },
            feature_cache: FeatureCache {
                cached_features: HashMap::new(),
                hit_rate: 0.0,
                size_limit: 1000,
                eviction_policy: CacheEvictionPolicy::LRU,
            },
        })
    }
}

impl<T: Float + Debug + Default + Send + Sync> PredictorTrainingData<T> {
    fn new() -> Self {
        Self {
            architecture_features: Vec::new(),
            performance_targets: Vec::new(),
            metadata: Vec::new(),
            data_splits: DataSplits {
                train_indices: Vec::new(),
                validation_indices: Vec::new(),
                test_indices: Vec::new(),
                split_ratios: (0.7, 0.15, 0.15),
            },
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> PredictionCache<T> {
    fn new() -> Self {
        Self {
            predictions: HashMap::new(),
            statistics: CacheStatistics {
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                hit_rate: 0.0,
                avg_prediction_time_ms: 0.0,
            },
            config: CacheConfig {
                max_size: 1000,
                ttl_seconds: 3600,
                eviction_policy: CacheEvictionPolicy::LRU,
                enable_persistence: false,
            },
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> UncertaintyEstimator<T> {
    fn new() -> Result<Self> {
        Ok(Self {
            estimation_method: UncertaintyEstimationMethod::MonteCarloDropout,
            model_ensemble: Vec::new(),
            parameters: UncertaintyParameters {
                num_samples: 100,
                confidence_level: scirs2_core::numeric::NumCast::from(0.95)
                    .unwrap_or_else(|| T::zero()),
                calibration_alpha: scirs2_core::numeric::NumCast::from(0.05)
                    .unwrap_or_else(|| T::zero()),
                method_params: HashMap::new(),
            },
            calibration_data: CalibrationData {
                predictions: Vec::new(),
                targets: Vec::new(),
                scores: Vec::new(),
                calibration_curve: CalibrationCurve {
                    bin_edges: Vec::new(),
                    bin_accuracies: Vec::new(),
                    bin_confidences: Vec::new(),
                    bin_counts: Vec::new(),
                },
            },
        })
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
}
