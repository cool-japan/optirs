//! End-to-end automated machine learning (AutoML) pipeline coordinator.
//!
//! `automl_pipeline` implements a *search-only* AutoML coordinator: it does
//! not ship the underlying ML models. Instead, it orchestrates the search
//! over the four classical AutoML axes — data preprocessing, feature
//! engineering, model selection, and ensembling — and delegates the actual
//! model training / scoring to a user-supplied callback.
//!
//! # Pipeline anatomy
//!
//! A complete pipeline is composed of:
//!
//! 1. **Preprocessing** — a sequence of [`PreprocessingStep`] transforms
//!    applied to the raw feature matrix (e.g. `StandardScaler`,
//!    `RobustScaler`, `ImputeMean`). Each transform is fitted on the
//!    training fold and applied to both the training and validation folds.
//! 2. **Feature engineering** — a sequence of [`FeatureEngineeringStep`]
//!    transforms that *append* derived features (polynomial expansions,
//!    log / sqrt transforms, pairwise products of the top-k correlated
//!    features, …) to the design matrix.
//! 3. **Model selection** — a list of [`CandidateModel`] types to
//!    evaluate. Each model is trained and scored on the (preprocessed,
//!    feature-engineered) data via the user-supplied `model_evaluator`
//!    callback.
//! 4. **Ensembling** — an [`EnsembleStrategy`] describing how to combine
//!    the candidate-model losses into a single pipeline score
//!    (`BestOnly`, `Average`, `Weighted`, `Median`, or `Stacking`).
//!
//! # High-level workflow
//!
//! ```ignore
//! use optirs_nas::automl_pipeline::{AutomlPipelineCoordinator, CandidateModel};
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_core::random::Random;
//!
//! let mut coord = AutomlPipelineCoordinator::new();
//! let mut rng = Random::seed(7);
//! let x = Array2::<f64>::zeros((100, 5));
//! let y = Array1::<f64>::zeros(100);
//!
//! // Propose a random pipeline from the configured search space.
//! let config = coord.propose_pipeline(&mut rng).expect("propose");
//!
//! // Evaluate the pipeline with a user-supplied model-training callback.
//! let eval = coord
//!     .evaluate_pipeline(&config, &x, &y, |_model, _x_train, _y_train, _x_val, _y_val| {
//!         Ok(0.42_f64)
//!     })
//!     .expect("evaluate");
//!
//! coord.record_pipeline(config, eval);
//! assert!(coord.best_pipeline().is_some());
//! ```
//!
//! # Numerical conventions
//!
//! All numeric computation happens in `f64`. The implementation is fully
//! deterministic given the configured `random_seed`: shuffling for the
//! train / validation split and the stochastic pipeline proposal both
//! consume a single seeded `Random<StdRng>`.
//!
//! # Error handling
//!
//! All errors funnel through [`crate::error::OptimError`]. The variants
//! `InvalidParameter`, `EvaluationError`, and `SearchSpaceError` are the
//! ones most commonly produced by this module.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod primitives;

use primitives::{
    aggregate_losses, append_per_feature, apply_feature_engineering_train_val,
    apply_preprocessing_train_val, compute_variance, impute_mean, impute_median, instant_to_ms,
    min_max_scaler, pairwise_top_k_products, pick_ensemble, polynomial_degree_2, robust_scaler,
    sample_distinct, standard_scaler, RECIPROCAL_EPSILON, VARIANCE_FLOOR,
};

// ============================================================================
// Enums describing the search space.
// ============================================================================

/// A single preprocessing transform applied to the raw feature matrix.
///
/// Each variant is *stateless* from the user's perspective: parameters
/// (column means, medians, IQR, …) are recomputed from the training fold
/// inside [`AutomlPipelineCoordinator::apply_preprocessing`]. This keeps
/// the coordinator side-effect-free and serialisable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreprocessingStep {
    /// Per-column standardisation to mean `0` and stddev `1`.
    StandardScaler,
    /// Per-column min-max scaling into `[0, 1]`.
    MinMaxScaler,
    /// Per-column robust scaling using the median and inter-quartile
    /// range (`q75 - q25`). Useful when outliers would corrupt
    /// `StandardScaler`'s mean / stddev.
    RobustScaler,
    /// Replace `NaN` / `Inf` entries with the column mean computed from
    /// the finite values.
    ImputeMean,
    /// Replace `NaN` / `Inf` entries with the column median computed from
    /// the finite values.
    ImputeMedian,
    /// Identity transform — the matrix is returned unchanged.
    Identity,
}

/// A single feature-engineering transform that *appends* derived features
/// to the design matrix.
///
/// Feature-engineering steps grow the column count but never shrink it.
/// The original features are always preserved at the head of the output
/// matrix to keep ordering predictable across pipeline runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureEngineeringStep {
    /// Append polynomial-degree-2 features: every product `x_i * x_j` for
    /// `i <= j` (including squares `x_i * x_i`).
    PolynomialDegree2,
    /// Append `log(|x| + 1)` per feature. Always well-defined.
    LogTransform,
    /// Append `sign(x) * sqrt(|x|)` per feature. Preserves sign and is
    /// continuous through zero.
    SqrtAbsTransform,
    /// Append pairwise products of the top-`k` features ranked by the
    /// absolute Pearson correlation with the target `y`. Requires the
    /// `y` argument to
    /// [`AutomlPipelineCoordinator::apply_feature_engineering`].
    PairwiseProducts {
        /// Number of features to consider for pairwise products.
        top_k: u32,
    },
    /// Append `1 / (|x| + epsilon)` per feature.
    Reciprocal,
    /// Identity (no new features).
    Identity,
}

/// A candidate ML model type considered during selection.
///
/// We do *not* train these models here. The coordinator records which
/// model types should be tried and forwards the choice to the user's
/// `model_evaluator` callback during pipeline evaluation. Hyperparameters
/// that would normally be real-valued are stored as integers (`u32`) so
/// that the enum can derive `Eq` / `Hash` — see the `_milli` suffix for
/// values scaled by `1000` (e.g. `reg_milli = 1500` means `reg = 1.5`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandidateModel {
    /// Ordinary least-squares linear regression.
    LinearRegression,
    /// Ridge (L2-regularised) linear regression.
    RidgeRegression {
        /// Regularisation strength scaled by 1000 (`reg = reg_milli * 1e-3`).
        reg_milli: u32,
    },
    /// Lasso (L1-regularised) linear regression.
    LassoRegression {
        /// Regularisation strength scaled by 1000.
        reg_milli: u32,
    },
    /// Single decision-tree regressor.
    DecisionTree {
        /// Maximum tree depth.
        max_depth: u32,
    },
    /// Random-forest regressor.
    RandomForest {
        /// Number of trees in the forest.
        n_trees: u32,
        /// Maximum depth per tree.
        max_depth: u32,
    },
    /// Gradient-boosted regressor.
    GradientBoosting {
        /// Number of weak learners.
        n_estimators: u32,
        /// Learning rate scaled by 1000 (`lr = lr_milli * 1e-3`).
        lr_milli: u32,
    },
    /// Support-vector regressor.
    SVM {
        /// Regularisation `C` scaled by 1000.
        c_milli: u32,
        /// Kernel type.
        kernel: SvmKernel,
    },
    /// Feedforward neural network regressor.
    NeuralNet {
        /// Hidden-layer width.
        hidden_dim: u32,
        /// Number of hidden layers.
        layers: u32,
    },
}

/// Kernel type for [`CandidateModel::SVM`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SvmKernel {
    /// Linear kernel `<x, x'>`.
    Linear,
    /// Radial basis function kernel `exp(-γ‖x - x'‖²)`.
    Rbf,
    /// Polynomial kernel `(γ<x, x'> + c)^d`.
    Polynomial,
}

/// Ensembling strategy combining the per-candidate-model losses into a
/// single pipeline score.
///
/// The pipeline score is *minimisation-oriented*: lower is better. For
/// strategies that fundamentally maximise (e.g. inverse-loss weighting),
/// the strategy returns the resulting loss in the same minimisation
/// convention so the coordinator can rank pipelines uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Arithmetic mean of the per-candidate losses.
    Average,
    /// Inverse-loss weighted average. Better models get more weight.
    Weighted,
    /// Stacking with a meta-learner. The simplified implementation
    /// approximates this with the mean — full stacking would require
    /// fitting a second-level model over per-candidate predictions, which
    /// is out of scope for the coordinator (the actual models live in the
    /// user's callback, not here).
    Stacking,
    /// Median of the per-candidate losses.
    Median,
    /// The best (lowest-loss) candidate only.
    BestOnly,
}

// ============================================================================
// Configuration and evaluation records.
// ============================================================================

/// A complete pipeline configuration.
///
/// A `AutomlPipelineConfig` specifies *what* to evaluate but not *how*
/// to evaluate it. The evaluation policy (cross-validation, holdout
/// fraction, …) is controlled by the coordinator. The default config
/// covers the most common preprocessing and model choices and is a
/// reasonable starting point for tabular regression problems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomlPipelineConfig {
    /// Preprocessing search-space pool.
    pub preprocessing: Vec<PreprocessingStep>,
    /// Feature-engineering search-space pool.
    pub feature_engineering: Vec<FeatureEngineeringStep>,
    /// Candidate model pool.
    pub candidate_models: Vec<CandidateModel>,
    /// Ensembling strategy applied across `candidate_models`.
    pub ensemble_strategy: EnsembleStrategy,
    /// Fraction of samples reserved for validation (`(0.0, 1.0)`).
    pub validation_fraction: f64,
    /// Maximum number of consecutive steps in the preprocessing /
    /// feature-engineering chain.
    pub max_pipeline_depth: usize,
    /// Deterministic seed for shuffling and proposal.
    pub random_seed: u64,
}

impl Default for AutomlPipelineConfig {
    fn default() -> Self {
        Self {
            preprocessing: vec![
                PreprocessingStep::Identity,
                PreprocessingStep::StandardScaler,
                PreprocessingStep::RobustScaler,
            ],
            feature_engineering: vec![
                FeatureEngineeringStep::Identity,
                FeatureEngineeringStep::LogTransform,
                FeatureEngineeringStep::PolynomialDegree2,
            ],
            candidate_models: vec![
                CandidateModel::LinearRegression,
                CandidateModel::RidgeRegression { reg_milli: 1000 },
                CandidateModel::RandomForest {
                    n_trees: 50,
                    max_depth: 8,
                },
            ],
            ensemble_strategy: EnsembleStrategy::Weighted,
            validation_fraction: 0.2,
            max_pipeline_depth: 3,
            random_seed: 42,
        }
    }
}

/// Outcome of evaluating one pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineEvaluation {
    /// Aggregated validation loss (smaller is better).
    pub validation_loss: f64,
    /// Validation `R²` derived from `validation_loss` and target variance,
    /// clamped to `[0, 1]`.
    pub validation_r_squared: f64,
    /// Unique pipeline identifier allocated by the coordinator.
    pub pipeline_id: String,
    /// Per-stage wall-clock timings in milliseconds.
    pub stage_timings_ms: HashMap<String, f64>,
    /// Number of features after the feature-engineering chain.
    pub feature_count_after_engineering: usize,
    /// Number of training samples (post-split).
    pub training_samples: usize,
}

/// A configuration paired with its evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPipeline {
    /// Pipeline configuration.
    pub config: AutomlPipelineConfig,
    /// Evaluation result.
    pub evaluation: PipelineEvaluation,
}

// ============================================================================
// Coordinator.
// ============================================================================

/// Train / validation split returned by
/// [`AutomlPipelineCoordinator::split_train_validation`].
/// Tuple ordering: `(x_train, y_train, x_val, y_val)`.
pub type TrainValSplit = (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>);

/// The AutoML coordinator.
///
/// `AutomlPipelineCoordinator` owns the search-space configuration, a
/// running history of evaluated pipelines, and the current best-so-far
/// record. It is intentionally orthogonal to the actual ML models: the
/// model-training callback is supplied per evaluation, so the coordinator
/// can be reused across different ML frameworks.
#[derive(Debug, Clone)]
pub struct AutomlPipelineCoordinator {
    /// Active configuration / search space.
    config: AutomlPipelineConfig,
    /// All recorded pipelines.
    history: Vec<ScoredPipeline>,
    /// Best-so-far pipeline (lowest `validation_loss`).
    best_pipeline: Option<ScoredPipeline>,
    /// Internal counter for unique pipeline identifiers.
    next_id: usize,
}

impl Default for AutomlPipelineCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl AutomlPipelineCoordinator {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Construct a coordinator using [`AutomlPipelineConfig::default`].
    pub fn new() -> Self {
        Self::with_config(AutomlPipelineConfig::default())
    }

    /// Construct a coordinator from an explicit configuration.
    pub fn with_config(config: AutomlPipelineConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            best_pipeline: None,
            next_id: 0,
        }
    }

    /// Builder: override the preprocessing pool.
    pub fn with_preprocessing(mut self, steps: Vec<PreprocessingStep>) -> Self {
        self.config.preprocessing = steps;
        self
    }

    /// Builder: override the feature-engineering pool.
    pub fn with_feature_engineering(mut self, steps: Vec<FeatureEngineeringStep>) -> Self {
        self.config.feature_engineering = steps;
        self
    }

    /// Builder: override the candidate-model pool.
    pub fn with_candidate_models(mut self, models: Vec<CandidateModel>) -> Self {
        self.config.candidate_models = models;
        self
    }

    /// Builder: override the ensembling strategy.
    pub fn with_ensemble(mut self, strategy: EnsembleStrategy) -> Self {
        self.config.ensemble_strategy = strategy;
        self
    }

    /// Builder: override the validation fraction. Must lie in `(0, 1)`.
    pub fn with_validation_fraction(mut self, fraction: f64) -> Self {
        self.config.validation_fraction = fraction;
        self
    }

    /// Builder: override the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = seed;
        self
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Borrow the active configuration.
    pub fn config(&self) -> &AutomlPipelineConfig {
        &self.config
    }

    /// Borrow the recorded pipeline history.
    pub fn history(&self) -> &[ScoredPipeline] {
        &self.history
    }

    /// Borrow the best-so-far pipeline, if any.
    pub fn best_pipeline(&self) -> Option<&ScoredPipeline> {
        self.best_pipeline.as_ref()
    }

    /// Return the top-`k` pipelines sorted by `validation_loss` ascending.
    pub fn top_k(&self, k: usize) -> Vec<&ScoredPipeline> {
        let mut sorted: Vec<&ScoredPipeline> = self.history.iter().collect();
        sorted.sort_by(|a, b| {
            a.evaluation
                .validation_loss
                .partial_cmp(&b.evaluation.validation_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k);
        sorted
    }

    /// Wipe the history, best-so-far record, and id counter.
    pub fn reset(&mut self) {
        self.history.clear();
        self.best_pipeline = None;
        self.next_id = 0;
    }

    // ------------------------------------------------------------------
    // Preprocessing
    // ------------------------------------------------------------------

    /// Apply a single preprocessing step to `x`.
    ///
    /// The matrix `x` is expected to be in `(n_samples, n_features)`
    /// layout. Statistics (mean, stddev, median, IQR, …) are computed
    /// from `x` itself; callers that need to fit on training and apply
    /// to validation should pass the *training* matrix into this method
    /// and reuse the produced transform offline. For the search-only
    /// pipeline evaluation in [`Self::evaluate_pipeline`] this is
    /// handled internally by training-fold-only fits — see the
    /// implementation comment on `apply_preprocessing_train_val`.
    pub fn apply_preprocessing(
        &self,
        x: &Array2<f64>,
        step: PreprocessingStep,
    ) -> Result<Array2<f64>> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(OptimError::InvalidParameter(
                "preprocessing input matrix must be non-empty".to_string(),
            ));
        }
        match step {
            PreprocessingStep::Identity => Ok(x.clone()),
            PreprocessingStep::StandardScaler => Ok(standard_scaler(x)),
            PreprocessingStep::MinMaxScaler => Ok(min_max_scaler(x)),
            PreprocessingStep::RobustScaler => Ok(robust_scaler(x)),
            PreprocessingStep::ImputeMean => Ok(impute_mean(x)),
            PreprocessingStep::ImputeMedian => Ok(impute_median(x)),
        }
    }

    /// Apply a single feature-engineering step to `x`.
    ///
    /// `y` is consumed only by [`FeatureEngineeringStep::PairwiseProducts`]
    /// to rank features by absolute Pearson correlation. The other
    /// variants ignore it.
    pub fn apply_feature_engineering(
        &self,
        x: &Array2<f64>,
        step: FeatureEngineeringStep,
        y: Option<&Array1<f64>>,
    ) -> Result<Array2<f64>> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(OptimError::InvalidParameter(
                "feature engineering input matrix must be non-empty".to_string(),
            ));
        }
        match step {
            FeatureEngineeringStep::Identity => Ok(x.clone()),
            FeatureEngineeringStep::PolynomialDegree2 => Ok(polynomial_degree_2(x)),
            FeatureEngineeringStep::LogTransform => {
                Ok(append_per_feature(x, |v| (v.abs() + 1.0).ln()))
            }
            FeatureEngineeringStep::SqrtAbsTransform => Ok(append_per_feature(x, |v| {
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                sign * v.abs().sqrt()
            })),
            FeatureEngineeringStep::Reciprocal => Ok(append_per_feature(x, |v| {
                1.0 / (v.abs() + RECIPROCAL_EPSILON)
            })),
            FeatureEngineeringStep::PairwiseProducts { top_k } => {
                let y = y.ok_or_else(|| {
                    OptimError::InvalidParameter(
                        "PairwiseProducts requires a target vector `y` for correlation ranking"
                            .to_string(),
                    )
                })?;
                if y.len() != x.nrows() {
                    return Err(OptimError::InvalidParameter(format!(
                        "y length ({}) does not match x rows ({})",
                        y.len(),
                        x.nrows()
                    )));
                }
                Ok(pairwise_top_k_products(x, y, top_k as usize))
            }
        }
    }

    // ------------------------------------------------------------------
    // Train / validation split
    // ------------------------------------------------------------------

    /// Random shuffle then split `(x, y)` by `validation_fraction`.
    ///
    /// Returns `(x_train, y_train, x_val, y_val)`. The split point is
    /// `n * (1.0 - validation_fraction)` rounded *down* so the validation
    /// set never exceeds `validation_fraction * n` samples.
    pub fn split_train_validation(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<TrainValSplit> {
        if x.nrows() != y.len() {
            return Err(OptimError::InvalidParameter(format!(
                "x rows ({}) and y length ({}) must match",
                x.nrows(),
                y.len()
            )));
        }
        if x.nrows() == 0 {
            return Err(OptimError::InvalidParameter(
                "cannot split an empty dataset".to_string(),
            ));
        }
        let fraction = self.config.validation_fraction;
        if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
            return Err(OptimError::InvalidParameter(format!(
                "validation_fraction must lie in [0, 1), got {}",
                fraction
            )));
        }

        let n = x.nrows();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = Random::seed(self.config.random_seed);
        // Fisher-Yates in-place shuffle.
        for i in (1..n).rev() {
            let j: usize = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let train_size = ((n as f64) * (1.0 - fraction)).floor() as usize;
        let train_size = train_size.min(n).max(1);
        let val_size = n - train_size;

        let n_features = x.ncols();
        let mut x_train = Array2::<f64>::zeros((train_size, n_features));
        let mut y_train = Array1::<f64>::zeros(train_size);
        let mut x_val = Array2::<f64>::zeros((val_size, n_features));
        let mut y_val = Array1::<f64>::zeros(val_size);

        for (k, &idx) in indices.iter().enumerate() {
            if k < train_size {
                for j in 0..n_features {
                    x_train[[k, j]] = x[[idx, j]];
                }
                y_train[k] = y[idx];
            } else {
                let r = k - train_size;
                for j in 0..n_features {
                    x_val[[r, j]] = x[[idx, j]];
                }
                y_val[r] = y[idx];
            }
        }
        Ok((x_train, y_train, x_val, y_val))
    }

    // ------------------------------------------------------------------
    // Proposal
    // ------------------------------------------------------------------

    /// Sample a random pipeline from the configured search space.
    ///
    /// The proposal:
    /// 1. Samples a random number of preprocessing steps
    ///    `n_pre ∈ [1, min(max_pipeline_depth, len(preprocessing))]` and
    ///    draws `n_pre` distinct steps without replacement.
    /// 2. Same for feature engineering.
    /// 3. Samples `n_models ∈ [1, min(3, len(candidate_models))]` and
    ///    draws `n_models` distinct models without replacement.
    /// 4. Picks one ensemble strategy from `{Average, Weighted, Median,
    ///    BestOnly, Stacking}`.
    pub fn propose_pipeline<R: Rng>(
        &mut self,
        rng: &mut Random<R>,
    ) -> Result<AutomlPipelineConfig> {
        self.ensure_search_space_valid()?;

        let preprocessing = sample_distinct(
            rng,
            &self.config.preprocessing,
            self.config.max_pipeline_depth,
        )?;
        let feature_engineering = sample_distinct(
            rng,
            &self.config.feature_engineering,
            self.config.max_pipeline_depth,
        )?;
        let model_budget = self.config.candidate_models.len().min(3);
        let candidate_models = sample_distinct(rng, &self.config.candidate_models, model_budget)?;

        let ensemble_strategy = pick_ensemble(rng);

        Ok(AutomlPipelineConfig {
            preprocessing,
            feature_engineering,
            candidate_models,
            ensemble_strategy,
            validation_fraction: self.config.validation_fraction,
            max_pipeline_depth: self.config.max_pipeline_depth,
            random_seed: self.config.random_seed,
        })
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// Evaluate one pipeline configuration on `(x, y)`.
    ///
    /// `model_evaluator` is invoked once per candidate model in
    /// `config.candidate_models` and must return the validation loss for
    /// that model (lower is better). The aggregated loss is then computed
    /// from `config.ensemble_strategy`.
    pub fn evaluate_pipeline<F>(
        &self,
        config: &AutomlPipelineConfig,
        x: &Array2<f64>,
        y: &Array1<f64>,
        mut model_evaluator: F,
    ) -> Result<PipelineEvaluation>
    where
        F: FnMut(
            &CandidateModel,
            &Array2<f64>,
            &Array1<f64>,
            &Array2<f64>,
            &Array1<f64>,
        ) -> Result<f64>,
    {
        if config.candidate_models.is_empty() {
            return Err(OptimError::EvaluationError(
                "cannot evaluate a pipeline with no candidate models".to_string(),
            ));
        }
        if !config.validation_fraction.is_finite()
            || !(0.0..1.0).contains(&config.validation_fraction)
        {
            return Err(OptimError::InvalidParameter(format!(
                "validation_fraction must lie in [0, 1), got {}",
                config.validation_fraction
            )));
        }

        let mut timings: HashMap<String, f64> = HashMap::new();

        // ---- 1. Split ---------------------------------------------------
        let t_split = std::time::Instant::now();
        // Build a temporary coordinator with the configured validation
        // fraction / seed so the split honours the pipeline configuration
        // (rather than `self.config`).
        let split_helper = AutomlPipelineCoordinator::with_config(config.clone());
        let (mut x_train, y_train, mut x_val, y_val) = split_helper.split_train_validation(x, y)?;
        timings.insert("split_ms".to_string(), instant_to_ms(t_split.elapsed()));

        // ---- 2. Preprocessing -------------------------------------------
        let t_pre = std::time::Instant::now();
        for step in &config.preprocessing {
            // For non-identity steps, we fit on the training matrix and
            // re-apply the same scalar transforms to the validation matrix
            // to avoid validation leakage.
            let (x_train_new, x_val_new) = apply_preprocessing_train_val(*step, &x_train, &x_val)?;
            x_train = x_train_new;
            x_val = x_val_new;
        }
        timings.insert(
            "preprocessing_ms".to_string(),
            instant_to_ms(t_pre.elapsed()),
        );

        // ---- 3. Feature engineering -------------------------------------
        let t_feat = std::time::Instant::now();
        for step in &config.feature_engineering {
            let (x_train_new, x_val_new) =
                apply_feature_engineering_train_val(*step, &x_train, &y_train, &x_val)?;
            x_train = x_train_new;
            x_val = x_val_new;
        }
        timings.insert(
            "feature_engineering_ms".to_string(),
            instant_to_ms(t_feat.elapsed()),
        );

        // ---- 4. Candidate evaluation ------------------------------------
        let t_eval = std::time::Instant::now();
        let mut losses: Vec<f64> = Vec::with_capacity(config.candidate_models.len());
        for model in &config.candidate_models {
            let loss = model_evaluator(model, &x_train, &y_train, &x_val, &y_val)?;
            if !loss.is_finite() {
                return Err(OptimError::EvaluationError(format!(
                    "model_evaluator returned non-finite loss ({}) for {:?}",
                    loss, model
                )));
            }
            losses.push(loss);
        }
        timings.insert("model_eval_ms".to_string(), instant_to_ms(t_eval.elapsed()));

        // ---- 5. Ensemble ------------------------------------------------
        let aggregated = aggregate_losses(&losses, config.ensemble_strategy);

        // ---- 6. R² ------------------------------------------------------
        let variance = compute_variance(&y_val);
        let r_squared = if variance > VARIANCE_FLOOR {
            (1.0 - aggregated / variance).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Pipeline id is *informational* — recorded into the evaluation so
        // the user can match evaluations back to entries in `history`.
        let pipeline_id = format!("pipeline_{}", self.next_id);

        Ok(PipelineEvaluation {
            validation_loss: aggregated,
            validation_r_squared: r_squared,
            pipeline_id,
            stage_timings_ms: timings,
            feature_count_after_engineering: x_train.ncols(),
            training_samples: x_train.nrows(),
        })
    }

    /// Record a pipeline and its evaluation. Updates `best_pipeline` if
    /// the new loss is strictly lower than the previous best.
    pub fn record_pipeline(&mut self, config: AutomlPipelineConfig, eval: PipelineEvaluation) {
        // Overwrite the pipeline_id so it matches the coordinator's
        // sequence — `evaluate_pipeline` only knows about `self.next_id`
        // at the time it was called, but the user might call
        // `record_pipeline` out of order.
        let mut eval = eval;
        eval.pipeline_id = format!("pipeline_{}", self.next_id);
        self.next_id += 1;

        let scored = ScoredPipeline {
            config,
            evaluation: eval,
        };

        let update_best = match &self.best_pipeline {
            None => true,
            Some(prev) => scored.evaluation.validation_loss < prev.evaluation.validation_loss,
        };
        if update_best {
            self.best_pipeline = Some(scored.clone());
        }
        self.history.push(scored);
    }

    /// Validate the search-space configuration.
    fn ensure_search_space_valid(&self) -> Result<()> {
        if self.config.preprocessing.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "preprocessing pool is empty".to_string(),
            ));
        }
        if self.config.feature_engineering.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "feature_engineering pool is empty".to_string(),
            ));
        }
        if self.config.candidate_models.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "candidate_models pool is empty".to_string(),
            ));
        }
        if self.config.max_pipeline_depth == 0 {
            return Err(OptimError::InvalidParameter(
                "max_pipeline_depth must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::aggregate_losses;

    // ---- helpers -----------------------------------------------------

    fn make_dataset(n_rows: usize, n_cols: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = Random::seed(seed);
        let mut x = Array2::<f64>::zeros((n_rows, n_cols));
        let mut y = Array1::<f64>::zeros(n_rows);
        for i in 0..n_rows {
            let mut sum = 0.0;
            for j in 0..n_cols {
                let v: f64 = rng.gen_range(-1.0_f64..1.0_f64);
                x[[i, j]] = v;
                sum += v * (j as f64 + 1.0);
            }
            y[i] = sum;
        }
        (x, y)
    }

    // ---- config / builder --------------------------------------------

    #[test]
    fn test_default_config_values_match_spec() {
        let cfg = AutomlPipelineConfig::default();
        assert_eq!(
            cfg.preprocessing,
            vec![
                PreprocessingStep::Identity,
                PreprocessingStep::StandardScaler,
                PreprocessingStep::RobustScaler,
            ]
        );
        assert_eq!(
            cfg.feature_engineering,
            vec![
                FeatureEngineeringStep::Identity,
                FeatureEngineeringStep::LogTransform,
                FeatureEngineeringStep::PolynomialDegree2,
            ]
        );
        assert_eq!(
            cfg.candidate_models,
            vec![
                CandidateModel::LinearRegression,
                CandidateModel::RidgeRegression { reg_milli: 1000 },
                CandidateModel::RandomForest {
                    n_trees: 50,
                    max_depth: 8
                },
            ]
        );
        assert_eq!(cfg.ensemble_strategy, EnsembleStrategy::Weighted);
        assert!((cfg.validation_fraction - 0.2).abs() < 1e-12);
        assert_eq!(cfg.max_pipeline_depth, 3);
        assert_eq!(cfg.random_seed, 42);
    }

    #[test]
    fn test_builder_pattern_chains() {
        let coord = AutomlPipelineCoordinator::new()
            .with_preprocessing(vec![PreprocessingStep::MinMaxScaler])
            .with_feature_engineering(vec![FeatureEngineeringStep::SqrtAbsTransform])
            .with_candidate_models(vec![CandidateModel::LinearRegression])
            .with_ensemble(EnsembleStrategy::BestOnly)
            .with_validation_fraction(0.3)
            .with_seed(7);
        assert_eq!(coord.config.preprocessing.len(), 1);
        assert_eq!(coord.config.feature_engineering.len(), 1);
        assert_eq!(coord.config.candidate_models.len(), 1);
        assert_eq!(coord.config.ensemble_strategy, EnsembleStrategy::BestOnly);
        assert!((coord.config.validation_fraction - 0.3).abs() < 1e-12);
        assert_eq!(coord.config.random_seed, 7);
    }

    // ---- preprocessing primitives -------------------------------------

    #[test]
    fn test_standard_scaler_zero_mean_unit_stddev() {
        // 10 rows, 1 column with the values 1..=10.
        let mut x = Array2::<f64>::zeros((10, 1));
        for i in 0..10 {
            x[[i, 0]] = (i + 1) as f64;
        }
        let coord = AutomlPipelineCoordinator::new();
        let out = coord
            .apply_preprocessing(&x, PreprocessingStep::StandardScaler)
            .expect("apply ok");
        let n = out.nrows() as f64;
        let mean: f64 = out.column(0).iter().sum::<f64>() / n;
        let var: f64 = out
            .column(0)
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        assert!(mean.abs() < 1e-9, "mean ≈ 0, got {}", mean);
        assert!((std - 1.0).abs() < 1e-9, "stddev ≈ 1, got {}", std);
    }

    #[test]
    fn test_minmax_scaler_range_zero_to_one() {
        let mut x = Array2::<f64>::zeros((5, 2));
        let raw = [(-3.0, 0.5), (1.0, 1.5), (2.0, 2.5), (4.0, 3.5), (-1.0, 4.5)];
        for (i, (a, b)) in raw.iter().enumerate() {
            x[[i, 0]] = *a;
            x[[i, 1]] = *b;
        }
        let coord = AutomlPipelineCoordinator::new();
        let out = coord
            .apply_preprocessing(&x, PreprocessingStep::MinMaxScaler)
            .expect("apply ok");
        for j in 0..out.ncols() {
            let col = out.column(j);
            let mn = col.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(mn.abs() < 1e-12, "col {} min ≈ 0, got {}", j, mn);
            assert!((mx - 1.0).abs() < 1e-12, "col {} max ≈ 1, got {}", j, mx);
        }
    }

    #[test]
    fn test_robust_scaler_resists_outliers() {
        // 99 values uniformly in [-1, 1] plus one extreme outlier at 1e6.
        let mut x = Array2::<f64>::zeros((100, 1));
        for i in 0..99 {
            x[[i, 0]] = -1.0 + 2.0 * (i as f64) / 98.0;
        }
        x[[99, 0]] = 1e6;
        let coord = AutomlPipelineCoordinator::new();
        let robust_out = coord
            .apply_preprocessing(&x, PreprocessingStep::RobustScaler)
            .expect("robust");
        // The 50th (median) row from x is at index 49 -> 0.0. RobustScaler
        // centres on the median (≈ 0) and divides by the IQR (~1.0), so
        // the 0.0 row stays close to 0.
        let robust_at_zero = robust_out[[49, 0]];
        assert!(
            robust_at_zero.abs() < 0.5,
            "robust scaler should keep zero-row near zero, got {}",
            robust_at_zero
        );
        // The boundary samples are still in a moderate range because the
        // IQR is ~1.0 and the boundary samples lie at ±1.0.
        let iqr_scaled_first = robust_out[[0, 0]];
        assert!(
            iqr_scaled_first.abs() < 5.0,
            "robust-scaled bulk sample should stay in a moderate range, got {}",
            iqr_scaled_first
        );
    }

    #[test]
    fn test_impute_mean_replaces_nan() {
        let mut x = Array2::<f64>::zeros((4, 2));
        x[[0, 0]] = 1.0;
        x[[1, 0]] = f64::NAN;
        x[[2, 0]] = 3.0;
        x[[3, 0]] = f64::INFINITY;
        x[[0, 1]] = 2.0;
        x[[1, 1]] = 4.0;
        x[[2, 1]] = 6.0;
        x[[3, 1]] = 8.0;

        let coord = AutomlPipelineCoordinator::new();
        let out = coord
            .apply_preprocessing(&x, PreprocessingStep::ImputeMean)
            .expect("impute");
        for j in 0..out.ncols() {
            for i in 0..out.nrows() {
                assert!(
                    out[[i, j]].is_finite(),
                    "ImputeMean must produce finite values, got {} at ({}, {})",
                    out[[i, j]],
                    i,
                    j
                );
            }
        }
        // Column 0 finite mean = (1 + 3) / 2 = 2.
        assert!((out[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((out[[3, 0]] - 2.0).abs() < 1e-12);
    }

    // ---- feature engineering primitives -------------------------------

    #[test]
    fn test_polynomial_features_count() {
        // 3 features → polynomial degree 2 → 3 + 6 = 9 columns.
        let mut x = Array2::<f64>::zeros((2, 3));
        x[[0, 0]] = 1.0;
        x[[0, 1]] = 2.0;
        x[[0, 2]] = 3.0;
        x[[1, 0]] = 4.0;
        x[[1, 1]] = 5.0;
        x[[1, 2]] = 6.0;
        let coord = AutomlPipelineCoordinator::new();
        let out = coord
            .apply_feature_engineering(&x, FeatureEngineeringStep::PolynomialDegree2, None)
            .expect("poly");
        assert_eq!(out.ncols(), 9);
        // Originals preserved.
        assert!((out[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((out[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((out[[0, 2]] - 3.0).abs() < 1e-12);
        // Lex pairs (a, b) for a <= b: (0,0)(0,1)(0,2)(1,1)(1,2)(2,2)
        //                              k=3   4    5    6    7    8
        assert!((out[[0, 3]] - 1.0).abs() < 1e-12); // 1*1
        assert!((out[[0, 4]] - 2.0).abs() < 1e-12); // 1*2
        assert!((out[[0, 5]] - 3.0).abs() < 1e-12); // 1*3
        assert!((out[[0, 6]] - 4.0).abs() < 1e-12); // 2*2
        assert!((out[[0, 7]] - 6.0).abs() < 1e-12); // 2*3
        assert!((out[[0, 8]] - 9.0).abs() < 1e-12); // 3*3
    }

    #[test]
    fn test_log_transform_handles_negative() {
        let mut x = Array2::<f64>::zeros((5, 1));
        x[[0, 0]] = -2.0;
        x[[1, 0]] = -1.0;
        x[[2, 0]] = 0.0;
        x[[3, 0]] = 1.0;
        x[[4, 0]] = 2.0;
        let coord = AutomlPipelineCoordinator::new();
        let out = coord
            .apply_feature_engineering(&x, FeatureEngineeringStep::LogTransform, None)
            .expect("log");
        assert_eq!(out.ncols(), 2);
        for j in 0..out.ncols() {
            for i in 0..out.nrows() {
                assert!(
                    out[[i, j]].is_finite(),
                    "LogTransform must produce finite outputs, got {} at ({}, {})",
                    out[[i, j]],
                    i,
                    j
                );
            }
        }
        // log(|0| + 1) = 0.
        assert!(out[[2, 1]].abs() < 1e-12);
        // log(|-2| + 1) = log(3).
        assert!((out[[0, 1]] - 3.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_pairwise_products_use_top_k_correlated() {
        // Construct: feature 0 perfectly correlated with y, feature 1
        // anti-correlated, feature 2 uncorrelated.
        let mut x = Array2::<f64>::zeros((6, 3));
        let mut y = Array1::<f64>::zeros(6);
        for i in 0..6 {
            let v = i as f64;
            x[[i, 0]] = v;
            x[[i, 1]] = -v;
            x[[i, 2]] = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
            y[i] = v;
        }
        let coord = AutomlPipelineCoordinator::new();
        let out = coord
            .apply_feature_engineering(
                &x,
                FeatureEngineeringStep::PairwiseProducts { top_k: 2 },
                Some(&y),
            )
            .expect("pairwise");
        // 3 originals + 1 pair (top_k=2 → 1 pair).
        assert_eq!(out.ncols(), 4);
    }

    // ---- split --------------------------------------------------------

    #[test]
    fn test_split_train_validation_proportions() {
        let (x, y) = make_dataset(100, 5, 1);
        let coord = AutomlPipelineCoordinator::new(); // fraction = 0.2 default.
        let (x_train, y_train, x_val, y_val) = coord.split_train_validation(&x, &y).expect("split");
        assert_eq!(x_train.nrows(), 80);
        assert_eq!(y_train.len(), 80);
        assert_eq!(x_val.nrows(), 20);
        assert_eq!(y_val.len(), 20);
        assert_eq!(x_train.ncols(), 5);
        assert_eq!(x_val.ncols(), 5);
    }

    #[test]
    fn test_split_train_validation_no_overlap() {
        // Encode the row index inside the data so we can detect any
        // duplication or loss after the split.
        let n_rows = 20_usize;
        let mut x = Array2::<f64>::zeros((n_rows, 1));
        let mut y = Array1::<f64>::zeros(n_rows);
        for i in 0..n_rows {
            x[[i, 0]] = i as f64;
            y[i] = i as f64;
        }
        let coord = AutomlPipelineCoordinator::new();
        let (x_train, _, x_val, _) = coord.split_train_validation(&x, &y).expect("split");

        let mut seen: Vec<f64> = Vec::with_capacity(n_rows);
        for i in 0..x_train.nrows() {
            seen.push(x_train[[i, 0]]);
        }
        for i in 0..x_val.nrows() {
            seen.push(x_val[[i, 0]]);
        }
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (i, value) in seen.iter().enumerate().take(n_rows) {
            assert!(
                (value - i as f64).abs() < 1e-12,
                "expected {} after sort, got {}",
                i,
                value
            );
        }
        // Disjointness follows from no duplicates among `n_rows` unique
        // values.
        assert_eq!(seen.len(), n_rows);
    }

    // ---- proposal -----------------------------------------------------

    #[test]
    fn test_propose_pipeline_within_search_space() {
        let mut coord = AutomlPipelineCoordinator::new();
        let mut rng = Random::seed(11);
        for _ in 0..20 {
            let proposal = coord.propose_pipeline(&mut rng).expect("propose");
            for step in &proposal.preprocessing {
                assert!(
                    coord.config.preprocessing.contains(step),
                    "preprocessing step {:?} not in pool",
                    step
                );
            }
            for step in &proposal.feature_engineering {
                assert!(
                    coord.config.feature_engineering.contains(step),
                    "feature_engineering step {:?} not in pool",
                    step
                );
            }
            for model in &proposal.candidate_models {
                assert!(
                    coord.config.candidate_models.contains(model),
                    "candidate_model {:?} not in pool",
                    model
                );
            }
            assert!(!proposal.preprocessing.is_empty());
            assert!(!proposal.feature_engineering.is_empty());
            assert!(!proposal.candidate_models.is_empty());
            // Distinctness — sample_distinct must not repeat.
            let mut pre_sorted = proposal.preprocessing.clone();
            pre_sorted.sort_by_key(|s| format!("{:?}", s));
            pre_sorted.dedup();
            assert_eq!(pre_sorted.len(), proposal.preprocessing.len());
        }
    }

    // ---- evaluation ---------------------------------------------------

    #[test]
    fn test_evaluate_pipeline_calls_model_evaluator_correct_number_of_times() {
        let (x, y) = make_dataset(40, 3, 2);
        let coord = AutomlPipelineCoordinator::new();
        let config = AutomlPipelineConfig {
            preprocessing: vec![PreprocessingStep::Identity],
            feature_engineering: vec![FeatureEngineeringStep::Identity],
            candidate_models: vec![
                CandidateModel::LinearRegression,
                CandidateModel::RidgeRegression { reg_milli: 500 },
                CandidateModel::DecisionTree { max_depth: 6 },
            ],
            ensemble_strategy: EnsembleStrategy::Average,
            validation_fraction: 0.25,
            max_pipeline_depth: 3,
            random_seed: 42,
        };

        // Counter closure to verify callback invocation count.
        use std::cell::Cell;
        let counter = Cell::new(0_usize);
        let evaluator = |_m: &CandidateModel,
                         _xt: &Array2<f64>,
                         _yt: &Array1<f64>,
                         _xv: &Array2<f64>,
                         _yv: &Array1<f64>| {
            counter.set(counter.get() + 1);
            Ok(0.5)
        };
        let eval = coord
            .evaluate_pipeline(&config, &x, &y, evaluator)
            .expect("evaluate ok");
        assert_eq!(counter.get(), 3, "evaluator called once per candidate");
        assert!((eval.validation_loss - 0.5).abs() < 1e-12);
        assert_eq!(eval.training_samples, 30); // 40 * (1 - 0.25) = 30
    }

    #[test]
    fn test_evaluate_pipeline_with_empty_models_errors() {
        let (x, y) = make_dataset(20, 2, 3);
        let coord = AutomlPipelineCoordinator::new();
        let config = AutomlPipelineConfig {
            preprocessing: vec![PreprocessingStep::Identity],
            feature_engineering: vec![FeatureEngineeringStep::Identity],
            candidate_models: Vec::new(),
            ensemble_strategy: EnsembleStrategy::Average,
            validation_fraction: 0.2,
            max_pipeline_depth: 3,
            random_seed: 0,
        };
        let result = coord.evaluate_pipeline(&config, &x, &y, |_, _, _, _, _| Ok(0.1));
        assert!(matches!(result, Err(OptimError::EvaluationError(_))));
    }

    #[test]
    fn test_invalid_validation_fraction_errors() {
        let (x, y) = make_dataset(10, 2, 5);
        let coord_neg = AutomlPipelineCoordinator::new().with_validation_fraction(-0.1);
        let coord_one = AutomlPipelineCoordinator::new().with_validation_fraction(1.0);
        let coord_nan = AutomlPipelineCoordinator::new().with_validation_fraction(f64::NAN);

        let r1 = coord_neg.split_train_validation(&x, &y);
        let r2 = coord_one.split_train_validation(&x, &y);
        let r3 = coord_nan.split_train_validation(&x, &y);
        assert!(matches!(r1, Err(OptimError::InvalidParameter(_))));
        assert!(matches!(r2, Err(OptimError::InvalidParameter(_))));
        assert!(matches!(r3, Err(OptimError::InvalidParameter(_))));
    }

    // ---- history ------------------------------------------------------

    #[test]
    fn test_record_pipeline_updates_best() {
        let mut coord = AutomlPipelineCoordinator::new();
        let cfg_a = AutomlPipelineConfig::default();
        let cfg_b = AutomlPipelineConfig::default();
        let eval_a = PipelineEvaluation {
            validation_loss: 0.5,
            validation_r_squared: 0.6,
            pipeline_id: "ignored_a".to_string(),
            stage_timings_ms: HashMap::new(),
            feature_count_after_engineering: 5,
            training_samples: 80,
        };
        let eval_b = PipelineEvaluation {
            validation_loss: 0.3,
            validation_r_squared: 0.7,
            pipeline_id: "ignored_b".to_string(),
            stage_timings_ms: HashMap::new(),
            feature_count_after_engineering: 5,
            training_samples: 80,
        };
        coord.record_pipeline(cfg_a, eval_a);
        coord.record_pipeline(cfg_b, eval_b);
        let best = coord.best_pipeline().expect("best set");
        assert!((best.evaluation.validation_loss - 0.3).abs() < 1e-12);
        assert_eq!(coord.history().len(), 2);
        // After recording, pipeline ids are coordinator-allocated.
        assert_eq!(coord.history()[0].evaluation.pipeline_id, "pipeline_0");
        assert_eq!(coord.history()[1].evaluation.pipeline_id, "pipeline_1");
    }

    #[test]
    fn test_top_k_sorted_ascending() {
        let mut coord = AutomlPipelineCoordinator::new();
        let losses = [0.7, 0.2, 0.5, 0.1, 0.9];
        for l in losses.iter() {
            let eval = PipelineEvaluation {
                validation_loss: *l,
                validation_r_squared: 0.0,
                pipeline_id: String::new(),
                stage_timings_ms: HashMap::new(),
                feature_count_after_engineering: 1,
                training_samples: 1,
            };
            coord.record_pipeline(AutomlPipelineConfig::default(), eval);
        }
        let top_3 = coord.top_k(3);
        assert_eq!(top_3.len(), 3);
        assert!((top_3[0].evaluation.validation_loss - 0.1).abs() < 1e-12);
        assert!((top_3[1].evaluation.validation_loss - 0.2).abs() < 1e-12);
        assert!((top_3[2].evaluation.validation_loss - 0.5).abs() < 1e-12);
        let top_all = coord.top_k(100);
        assert_eq!(top_all.len(), 5);
    }

    #[test]
    fn test_reset_clears_history() {
        let mut coord = AutomlPipelineCoordinator::new();
        for l in [0.5, 0.3] {
            let eval = PipelineEvaluation {
                validation_loss: l,
                validation_r_squared: 0.0,
                pipeline_id: String::new(),
                stage_timings_ms: HashMap::new(),
                feature_count_after_engineering: 0,
                training_samples: 0,
            };
            coord.record_pipeline(AutomlPipelineConfig::default(), eval);
        }
        assert!(coord.best_pipeline().is_some());
        assert_eq!(coord.history().len(), 2);
        coord.reset();
        assert!(coord.best_pipeline().is_none());
        assert_eq!(coord.history().len(), 0);
        // next_id must reset too — the next record uses pipeline_0.
        let eval = PipelineEvaluation {
            validation_loss: 1.0,
            validation_r_squared: 0.0,
            pipeline_id: String::new(),
            stage_timings_ms: HashMap::new(),
            feature_count_after_engineering: 0,
            training_samples: 0,
        };
        coord.record_pipeline(AutomlPipelineConfig::default(), eval);
        assert_eq!(coord.history()[0].evaluation.pipeline_id, "pipeline_0");
    }

    // ---- serde --------------------------------------------------------

    #[test]
    fn test_pipeline_serde_roundtrip() {
        let cfg = AutomlPipelineConfig {
            preprocessing: vec![
                PreprocessingStep::StandardScaler,
                PreprocessingStep::ImputeMean,
            ],
            feature_engineering: vec![
                FeatureEngineeringStep::PairwiseProducts { top_k: 4 },
                FeatureEngineeringStep::Reciprocal,
            ],
            candidate_models: vec![
                CandidateModel::SVM {
                    c_milli: 750,
                    kernel: SvmKernel::Rbf,
                },
                CandidateModel::NeuralNet {
                    hidden_dim: 32,
                    layers: 2,
                },
            ],
            ensemble_strategy: EnsembleStrategy::Median,
            validation_fraction: 0.15,
            max_pipeline_depth: 4,
            random_seed: 7,
        };
        let s = serde_json::to_string(&cfg).expect("to_string");
        let back: AutomlPipelineConfig = serde_json::from_str(&s).expect("from_str");
        assert_eq!(back.preprocessing, cfg.preprocessing);
        assert_eq!(back.feature_engineering, cfg.feature_engineering);
        assert_eq!(back.candidate_models, cfg.candidate_models);
        assert_eq!(back.ensemble_strategy, cfg.ensemble_strategy);
        assert!((back.validation_fraction - cfg.validation_fraction).abs() < 1e-12);
        assert_eq!(back.max_pipeline_depth, cfg.max_pipeline_depth);
        assert_eq!(back.random_seed, cfg.random_seed);
    }

    // ---- end-to-end ---------------------------------------------------

    #[test]
    fn test_end_to_end_pipeline_runs() {
        let (x, y) = make_dataset(60, 4, 9);
        let mut coord = AutomlPipelineCoordinator::new()
            .with_preprocessing(vec![
                PreprocessingStep::Identity,
                PreprocessingStep::StandardScaler,
            ])
            .with_feature_engineering(vec![
                FeatureEngineeringStep::Identity,
                FeatureEngineeringStep::LogTransform,
            ])
            .with_candidate_models(vec![
                CandidateModel::LinearRegression,
                CandidateModel::RidgeRegression { reg_milli: 1000 },
            ])
            .with_ensemble(EnsembleStrategy::BestOnly)
            .with_validation_fraction(0.2)
            .with_seed(123);
        let mut rng = Random::seed(99);
        let cfg = coord.propose_pipeline(&mut rng).expect("propose");
        let eval = coord
            .evaluate_pipeline(&cfg, &x, &y, |model, x_train, y_train, _x_val, _y_val| {
                // Mock evaluator: loss depends on the model identity
                // and the data shape so the test is deterministic.
                let base = match model {
                    CandidateModel::LinearRegression => 0.10,
                    CandidateModel::RidgeRegression { .. } => 0.15,
                    _ => 0.20,
                };
                let scale = (x_train.ncols() as f64 + 1.0) / (y_train.len() as f64 + 1.0);
                Ok(base + 0.01 * scale)
            })
            .expect("eval");
        assert!(eval.validation_loss.is_finite());
        assert!(eval.validation_r_squared >= 0.0 && eval.validation_r_squared <= 1.0);
        assert!(eval.feature_count_after_engineering >= 4);
        assert!(eval.stage_timings_ms.contains_key("split_ms"));
        assert!(eval.stage_timings_ms.contains_key("preprocessing_ms"));
        assert!(eval.stage_timings_ms.contains_key("feature_engineering_ms"));
        assert!(eval.stage_timings_ms.contains_key("model_eval_ms"));
        coord.record_pipeline(cfg, eval);
        assert!(coord.best_pipeline().is_some());
    }

    #[test]
    fn test_ensemble_strategies_distinct_outputs() {
        let losses = [0.1_f64, 0.3, 0.5];
        assert!((aggregate_losses(&losses, EnsembleStrategy::BestOnly) - 0.1).abs() < 1e-12);
        assert!((aggregate_losses(&losses, EnsembleStrategy::Average) - 0.3).abs() < 1e-12);
        assert!((aggregate_losses(&losses, EnsembleStrategy::Median) - 0.3).abs() < 1e-12);
        let weighted = aggregate_losses(&losses, EnsembleStrategy::Weighted);
        // Weighted (inverse-loss) should be < Average because low-loss
        // models get more weight.
        assert!(
            weighted < 0.3,
            "weighted ({}) should be < average",
            weighted
        );
        assert!(weighted >= 0.1, "weighted ({}) should be >= best", weighted);
    }

    #[test]
    fn test_evaluator_error_propagates() {
        let (x, y) = make_dataset(20, 2, 17);
        let coord = AutomlPipelineCoordinator::new();
        let cfg = AutomlPipelineConfig {
            preprocessing: vec![PreprocessingStep::Identity],
            feature_engineering: vec![FeatureEngineeringStep::Identity],
            candidate_models: vec![CandidateModel::LinearRegression],
            ensemble_strategy: EnsembleStrategy::Average,
            validation_fraction: 0.2,
            max_pipeline_depth: 3,
            random_seed: 0,
        };
        let result = coord.evaluate_pipeline(&cfg, &x, &y, |_, _, _, _, _| {
            Err(OptimError::EvaluationError("boom".to_string()))
        });
        assert!(matches!(result, Err(OptimError::EvaluationError(_))));
    }

    #[test]
    fn test_evaluator_returning_nan_errors() {
        let (x, y) = make_dataset(20, 2, 31);
        let coord = AutomlPipelineCoordinator::new();
        let cfg = AutomlPipelineConfig {
            preprocessing: vec![PreprocessingStep::Identity],
            feature_engineering: vec![FeatureEngineeringStep::Identity],
            candidate_models: vec![CandidateModel::LinearRegression],
            ensemble_strategy: EnsembleStrategy::Average,
            validation_fraction: 0.2,
            max_pipeline_depth: 3,
            random_seed: 0,
        };
        let result = coord.evaluate_pipeline(&cfg, &x, &y, |_, _, _, _, _| Ok(f64::NAN));
        assert!(matches!(result, Err(OptimError::EvaluationError(_))));
    }

    #[test]
    fn test_evaluate_pipeline_propagates_split_error() {
        let (x, y) = make_dataset(10, 2, 99);
        let coord = AutomlPipelineCoordinator::new();
        let cfg = AutomlPipelineConfig {
            preprocessing: vec![PreprocessingStep::Identity],
            feature_engineering: vec![FeatureEngineeringStep::Identity],
            candidate_models: vec![CandidateModel::LinearRegression],
            ensemble_strategy: EnsembleStrategy::Average,
            validation_fraction: 1.0, // invalid
            max_pipeline_depth: 3,
            random_seed: 0,
        };
        let r = coord.evaluate_pipeline(&cfg, &x, &y, |_, _, _, _, _| Ok(0.0));
        assert!(matches!(r, Err(OptimError::InvalidParameter(_))));
    }
}
