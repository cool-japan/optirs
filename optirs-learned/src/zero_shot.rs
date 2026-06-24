//! # Zero-Shot Optimizer Selection
//!
//! This module implements **zero-shot optimizer selection**: given a brand-new
//! optimization task, it recommends an optimizer class together with hyper-parameters
//! *without performing any per-task training*. The recommendation is produced by
//!
//! 1. **lightweight probing** of the task (querying a few gradients), which yields a
//!    real-valued **meta-feature** vector describing the local geometry of the problem,
//!    and
//! 2. a **meta-model** that was fit OFFLINE on a meta-dataset of previously-solved tasks
//!    and that maps a (standardized) meta-feature vector to an optimizer class and a
//!    learning rate.
//!
//! No gradient steps are taken on the new task itself — only a handful of gradient
//! *evaluations* are needed to extract the meta-features. This mirrors the "learning to
//! optimize / algorithm selection" literature, where a portfolio of optimizers is chosen
//! per problem from cheap descriptive features.
//!
//! ## Meta-features
//!
//! The extractor probes the task at the initial point and along a short
//! normalized-descent path, plus along sampled directions for curvature, and computes a
//! fixed-length vector of [`NUM_FEATURES`] features. Each feature is documented on the
//! corresponding accessor of [`MetaFeatures`]; in index order they are:
//!
//! - `log_dimension` (0): `ln(dim + 1)`, the (log) problem dimensionality.
//! - `grad_norm_mean` (1): mean gradient L2-norm across the probe path.
//! - `grad_norm_cv` (2): coefficient of variation of the gradient L2-norm — how strongly
//!   the gradient magnitude changes across the landscape.
//! - `grad_sparsity` (3): average fraction of gradient components that are effectively
//!   zero (below a relative threshold).
//! - `mean_log_curvature` (4): `ln(1 + mean directional curvature)`, an overall scale of
//!   the local Hessian.
//! - `log_condition_number` (5): `ln(max_dir_curvature / min_dir_curvature)`, a
//!   condition-number proxy estimated from finite differences of the gradient along
//!   sampled directions. This is the feature that distinguishes ill-conditioned from
//!   well-conditioned problems.
//! - `direction_stability` (6): mean cosine similarity between successive gradients along
//!   the descent path (gradient-direction autocorrelation).
//! - `noise_estimate` (7): mean per-component coefficient of variation of repeated
//!   (stochastic) gradient samples at the initial point; `0` for a deterministic oracle,
//!   `> 0` for a noisy one.
//! - `log_param_scale` (8): `ln(1 + RMS(params))`, the magnitude/scale of the parameters.
//!
//! The raw features are **standardized** with the offline mean/standard-deviation learned
//! by [`ZeroShotSelector::fit`]; a feature with zero variance over the meta-dataset is
//! divided by one (instead of zero) so standardization never produces `NaN`.
//!
//! ## Meta-model (learned offline)
//!
//! The meta-model is a genuine learned mapping, trained on the standardized meta-features:
//!
//! - **Optimizer class:** a **multinomial logistic-regression** classifier over the five
//!   [`OptimizerKind`] classes, trained by full-batch gradient descent on the softmax
//!   cross-entropy loss with L2 regularization. The winning class is the recommendation
//!   and the maximum softmax probability is reported as the **confidence**.
//! - **Learning rate:** a **linear regressor** predicting `log10(learning_rate)`, trained
//!   by gradient descent on the mean-squared error with L2 regularization. The prediction
//!   is exponentiated and clamped to a sane range.
//!
//! Both predictors are deterministic functions of the meta-dataset (zero-initialized,
//! full-batch gradient descent), so a fixed meta-dataset yields a fixed meta-model.
//!
//! ## Zero-shot inference
//!
//! [`ZeroShotSelector::recommend`] extracts meta-features from a [`TaskProbe`],
//! standardizes them with the fitted statistics, and runs the two predictors — without
//! ever touching the new task beyond the cheap probe.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Number of meta-features produced by the extractor.
pub const NUM_FEATURES: usize = 9;

/// Number of optimizer classes the meta-model discriminates between.
pub const NUM_CLASSES: usize = 5;

/// Human-readable names of the meta-features, in index order.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "log_dimension",
    "grad_norm_mean",
    "grad_norm_cv",
    "grad_sparsity",
    "mean_log_curvature",
    "log_condition_number",
    "direction_stability",
    "noise_estimate",
    "log_param_scale",
];

/// Small constant guarding divisions by (near-)zero quantities.
const EPS: f64 = 1e-12;

/// Lower bound on a feature's standard deviation before it is treated as zero-variance and
/// the standardization divisor is replaced by one.
const STD_FLOOR: f64 = 1e-8;

// ---------------------------------------------------------------------------------------
// Optimizer kinds and hyper-parameters
// ---------------------------------------------------------------------------------------

/// The optimizer families the zero-shot selector can recommend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizerKind {
    /// Plain stochastic gradient descent.
    Sgd,
    /// SGD with classical (heavy-ball) momentum.
    Momentum,
    /// Adam (adaptive moments with bias correction).
    Adam,
    /// RMSProp (root-mean-square propagation).
    RmsProp,
    /// Adagrad (per-parameter accumulated-gradient scaling).
    Adagrad,
}

impl OptimizerKind {
    /// All optimizer kinds in canonical (class-index) order.
    pub fn all() -> [OptimizerKind; NUM_CLASSES] {
        [
            OptimizerKind::Sgd,
            OptimizerKind::Momentum,
            OptimizerKind::Adam,
            OptimizerKind::RmsProp,
            OptimizerKind::Adagrad,
        ]
    }

    /// Class index in `0..NUM_CLASSES` used by the meta-model.
    pub fn index(self) -> usize {
        match self {
            OptimizerKind::Sgd => 0,
            OptimizerKind::Momentum => 1,
            OptimizerKind::Adam => 2,
            OptimizerKind::RmsProp => 3,
            OptimizerKind::Adagrad => 4,
        }
    }

    /// Inverse of [`OptimizerKind::index`]; errors on an out-of-range index.
    pub fn from_index(index: usize) -> Result<OptimizerKind> {
        match index {
            0 => Ok(OptimizerKind::Sgd),
            1 => Ok(OptimizerKind::Momentum),
            2 => Ok(OptimizerKind::Adam),
            3 => Ok(OptimizerKind::RmsProp),
            4 => Ok(OptimizerKind::Adagrad),
            other => Err(OptimError::ComputationError(format!(
                "optimizer class index {other} out of range 0..{NUM_CLASSES}"
            ))),
        }
    }

    /// Stable human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            OptimizerKind::Sgd => "Sgd",
            OptimizerKind::Momentum => "Momentum",
            OptimizerKind::Adam => "Adam",
            OptimizerKind::RmsProp => "RmsProp",
            OptimizerKind::Adagrad => "Adagrad",
        }
    }

    /// Canonical hyper-parameters for this optimizer family.
    ///
    /// The learning rate is predicted separately by the regressor; these momentum / beta /
    /// epsilon values are the well-established defaults for the selected family.
    pub fn default_hyperparameters(self) -> OptimizerHyperparameters {
        match self {
            OptimizerKind::Sgd => OptimizerHyperparameters {
                momentum: 0.0,
                beta1: 0.0,
                beta2: 0.0,
                epsilon: 0.0,
            },
            OptimizerKind::Momentum => OptimizerHyperparameters {
                momentum: 0.9,
                beta1: 0.0,
                beta2: 0.0,
                epsilon: 0.0,
            },
            OptimizerKind::Adam => OptimizerHyperparameters {
                momentum: 0.0,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            OptimizerKind::RmsProp => OptimizerHyperparameters {
                momentum: 0.0,
                beta1: 0.0,
                beta2: 0.9,
                epsilon: 1e-8,
            },
            OptimizerKind::Adagrad => OptimizerHyperparameters {
                momentum: 0.0,
                beta1: 0.0,
                beta2: 0.0,
                epsilon: 1e-10,
            },
        }
    }
}

/// Hyper-parameters accompanying an optimizer recommendation.
///
/// Not every field is meaningful for every optimizer family — e.g. `beta1`/`beta2` apply
/// to Adam, `momentum` to heavy-ball momentum, `beta2` to the RMSProp decay. Fields that
/// do not apply to the selected family are reported as `0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OptimizerHyperparameters {
    /// Heavy-ball momentum coefficient (Momentum family).
    pub momentum: f64,
    /// First-moment decay `β₁` (Adam family).
    pub beta1: f64,
    /// Second-moment / squared-gradient decay `β₂` (Adam, RMSProp families).
    pub beta2: f64,
    /// Numerical stabilizer added before division.
    pub epsilon: f64,
}

impl Default for OptimizerHyperparameters {
    fn default() -> Self {
        OptimizerHyperparameters {
            momentum: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

// ---------------------------------------------------------------------------------------
// Task probe abstraction
// ---------------------------------------------------------------------------------------

/// A lightweight handle on an optimization task used for meta-feature extraction.
///
/// A probe exposes the problem dimensionality, a starting point, and a way to query the
/// gradient at arbitrary points. Implementations should make [`TaskProbe::gradient_at`]
/// cheap, since the extractor calls it a handful of times.
pub trait TaskProbe {
    /// Dimensionality of the parameter vector.
    fn dimension(&self) -> usize;

    /// Gradient (or its expectation, for a stochastic oracle) at `params`.
    fn gradient_at(&self, params: &Array1<f64>) -> Array1<f64>;

    /// A representative initial parameter vector for the task.
    fn initial_params(&self) -> Array1<f64>;

    /// Draw the `sample_index`-th (possibly noisy) gradient sample at `params`.
    ///
    /// The default implementation is deterministic and equal to [`TaskProbe::gradient_at`]
    /// (i.e. a noise-free oracle). Stochastic tasks override this so that distinct
    /// `sample_index` values yield distinct — but *reproducible* — gradient samples,
    /// which lets the extractor estimate the gradient noise level deterministically.
    fn gradient_sample(&self, params: &Array1<f64>, sample_index: usize) -> Array1<f64> {
        let _ = sample_index;
        self.gradient_at(params)
    }
}

/// A separable quadratic probe `L(w) = ½ Σ c_i (w_i - center_i)²`.
///
/// Its gradient is `g_i = c_i (w_i - center_i)` and its Hessian is the diagonal matrix
/// `diag(c)`, so the per-coordinate `curvature` values are exactly the eigenvalues. This
/// makes it the natural fixture for exercising the condition-number meta-feature: an
/// anisotropic curvature vector produces a known, controllable condition number.
///
/// An optional gradient-noise level turns the probe into a *stochastic* oracle whose
/// per-sample noise is seeded by the sample index, so repeated samples differ yet the
/// whole extraction stays reproducible.
#[derive(Debug, Clone)]
pub struct QuadraticProbe {
    curvature: Array1<f64>,
    center: Array1<f64>,
    init: Array1<f64>,
    noise_std: f64,
    noise_seed: u64,
}

impl QuadraticProbe {
    /// Build a bowl with per-coordinate `curvature` (the Hessian diagonal), centered at
    /// the origin, started from `init`.
    pub fn new(curvature: Array1<f64>, init: Array1<f64>) -> Result<Self> {
        if curvature.is_empty() {
            return Err(OptimError::InvalidConfig(
                "QuadraticProbe curvature must be non-empty".to_string(),
            ));
        }
        if curvature.len() != init.len() {
            return Err(OptimError::InvalidConfig(format!(
                "curvature length {} != init length {}",
                curvature.len(),
                init.len()
            )));
        }
        let center = Array1::zeros(curvature.len());
        Ok(Self {
            curvature,
            center,
            init,
            noise_std: 0.0,
            noise_seed: 0,
        })
    }

    /// Build an isotropic `dim`-dimensional bowl with constant `curvature`, started from a
    /// constant `init_value` in every coordinate.
    pub fn isotropic(dim: usize, curvature: f64, init_value: f64) -> Result<Self> {
        if dim == 0 {
            return Err(OptimError::InvalidConfig(
                "QuadraticProbe dimension must be positive".to_string(),
            ));
        }
        Self::new(
            Array1::from_elem(dim, curvature),
            Array1::from_elem(dim, init_value),
        )
    }

    /// Override the bowl center (the location of the minimum).
    pub fn with_center(mut self, center: Array1<f64>) -> Result<Self> {
        if center.len() != self.curvature.len() {
            return Err(OptimError::InvalidConfig(
                "center dimension must match curvature dimension".to_string(),
            ));
        }
        self.center = center;
        Ok(self)
    }

    /// Turn the probe into a stochastic oracle whose gradient samples carry uniform noise
    /// of the given standard deviation, seeded by `seed`.
    pub fn with_noise(mut self, noise_std: f64, seed: u64) -> Self {
        self.noise_std = noise_std.abs();
        self.noise_seed = seed;
        self
    }

    fn mean_gradient(&self, params: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(self.curvature.len(), |i| {
            let xi = params.get(i).copied().unwrap_or(0.0);
            self.curvature[i] * (xi - self.center[i])
        })
    }
}

impl TaskProbe for QuadraticProbe {
    fn dimension(&self) -> usize {
        self.curvature.len()
    }

    fn gradient_at(&self, params: &Array1<f64>) -> Array1<f64> {
        self.mean_gradient(params)
    }

    fn initial_params(&self) -> Array1<f64> {
        self.init.clone()
    }

    fn gradient_sample(&self, params: &Array1<f64>, sample_index: usize) -> Array1<f64> {
        let mut grad = self.mean_gradient(params);
        if self.noise_std > 0.0 {
            let seed = mix_seed(self.noise_seed, sample_index as u64);
            let mut rng = Random::seed(seed);
            let std = self.noise_std;
            grad.mapv_inplace(|g| g + rng.random_range(-std..std));
        }
        grad
    }
}

/// The classical Rosenbrock valley probe
/// `L(w) = Σ_i [ b (w_{i+1} - w_i²)² + (a - w_i)² ]`.
///
/// A non-convex, strongly ill-conditioned task (the default `b = 100` produces a long
/// curved valley), useful for stress-testing the curvature meta-feature on a non-quadratic
/// landscape.
#[derive(Debug, Clone)]
pub struct RosenbrockProbe {
    dim: usize,
    a: f64,
    b: f64,
    init: Array1<f64>,
}

impl RosenbrockProbe {
    /// Build a `dim`-dimensional Rosenbrock probe (`a = 1`, `b = 100`) started from
    /// `init`. Requires `dim >= 2`.
    pub fn new(init: Array1<f64>) -> Result<Self> {
        let dim = init.len();
        if dim < 2 {
            return Err(OptimError::InvalidConfig(
                "RosenbrockProbe requires at least two dimensions".to_string(),
            ));
        }
        Ok(Self {
            dim,
            a: 1.0,
            b: 100.0,
            init,
        })
    }
}

impl TaskProbe for RosenbrockProbe {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn gradient_at(&self, params: &Array1<f64>) -> Array1<f64> {
        let mut grad = Array1::<f64>::zeros(self.dim);
        for i in 0..self.dim - 1 {
            let xi = params.get(i).copied().unwrap_or(0.0);
            let xi1 = params.get(i + 1).copied().unwrap_or(0.0);
            let residual = xi1 - xi * xi;
            let offset = self.a - xi;
            grad[i] += self.b * 2.0 * residual * (-2.0 * xi) - 2.0 * offset;
            grad[i + 1] += self.b * 2.0 * residual;
        }
        grad
    }

    fn initial_params(&self) -> Array1<f64> {
        self.init.clone()
    }
}

// ---------------------------------------------------------------------------------------
// Meta-features
// ---------------------------------------------------------------------------------------

/// A fixed-length, real-valued description of an optimization task.
///
/// The values are stored in canonical order (see the module-level documentation and
/// [`FEATURE_NAMES`]); named accessors document the meaning of each entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaFeatures {
    /// The raw (un-standardized) feature values, length [`NUM_FEATURES`].
    pub values: Array1<f64>,
}

impl MetaFeatures {
    /// Wrap a raw feature vector, validating its length.
    pub fn from_values(values: Array1<f64>) -> Result<Self> {
        if values.len() != NUM_FEATURES {
            return Err(OptimError::InvalidConfig(format!(
                "meta-feature vector has length {} but {NUM_FEATURES} were expected",
                values.len()
            )));
        }
        Ok(Self { values })
    }

    /// The raw feature values.
    pub fn as_array(&self) -> &Array1<f64> {
        &self.values
    }

    /// Feature names in index order.
    pub fn names() -> [&'static str; NUM_FEATURES] {
        FEATURE_NAMES
    }

    /// `ln(dim + 1)` — the (log) problem dimensionality.
    pub fn log_dimension(&self) -> f64 {
        self.values[0]
    }

    /// Mean gradient L2-norm across the probe path.
    pub fn grad_norm_mean(&self) -> f64 {
        self.values[1]
    }

    /// Coefficient of variation of the gradient L2-norm across the probe path.
    pub fn grad_norm_cv(&self) -> f64 {
        self.values[2]
    }

    /// Average fraction of gradient components that are effectively zero.
    pub fn grad_sparsity(&self) -> f64 {
        self.values[3]
    }

    /// `ln(1 + mean directional curvature)`.
    pub fn mean_log_curvature(&self) -> f64 {
        self.values[4]
    }

    /// `ln(max directional curvature / min directional curvature)` — condition-number
    /// proxy.
    pub fn log_condition_number(&self) -> f64 {
        self.values[5]
    }

    /// Mean cosine similarity between successive gradients along the descent path.
    pub fn direction_stability(&self) -> f64 {
        self.values[6]
    }

    /// Mean per-component coefficient of variation of repeated gradient samples.
    pub fn noise_estimate(&self) -> f64 {
        self.values[7]
    }

    /// `ln(1 + RMS(params))` — the magnitude/scale of the parameters.
    pub fn log_param_scale(&self) -> f64 {
        self.values[8]
    }
}

/// One row of the offline meta-dataset: the meta-features of a previously solved task,
/// together with the optimizer / learning rate that worked best on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaExample {
    /// Meta-features describing the task.
    pub features: MetaFeatures,
    /// The optimizer family that performed best on the task.
    pub best_optimizer: OptimizerKind,
    /// The learning rate that performed best on the task (must be positive).
    pub best_learning_rate: f64,
}

impl MetaExample {
    /// Construct a meta-dataset row, validating the learning rate.
    pub fn new(
        features: MetaFeatures,
        best_optimizer: OptimizerKind,
        best_learning_rate: f64,
    ) -> Result<Self> {
        if !(best_learning_rate.is_finite() && best_learning_rate > 0.0) {
            return Err(OptimError::InvalidConfig(format!(
                "best_learning_rate must be finite and positive, got {best_learning_rate}"
            )));
        }
        Ok(Self {
            features,
            best_optimizer,
            best_learning_rate,
        })
    }
}

// ---------------------------------------------------------------------------------------
// Recommendation
// ---------------------------------------------------------------------------------------

/// The output of zero-shot inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizerRecommendation {
    /// Recommended optimizer family.
    pub optimizer: OptimizerKind,
    /// Recommended learning rate.
    pub learning_rate: f64,
    /// Recommended hyper-parameters for the selected family.
    pub hyperparameters: OptimizerHyperparameters,
    /// Classifier confidence in `[0, 1]` (the maximum softmax probability).
    pub confidence: f64,
}

// ---------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------

/// Configuration for meta-feature extraction and meta-model training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZeroShotConfig {
    /// Number of points visited along the normalized-descent probe path (`>= 1`).
    pub num_probe_points: usize,
    /// Step size of the normalized-descent probe walk.
    pub probe_step: f64,
    /// Maximum number of coordinate directions used for curvature probing.
    pub num_curvature_directions: usize,
    /// Number of additional random directions used for curvature probing.
    pub num_random_curvature_dirs: usize,
    /// Finite-difference step (scaled by the parameter RMS) for curvature estimation.
    pub finite_diff_eps: f64,
    /// Relative threshold below which a gradient component counts as "zero" for sparsity.
    pub sparsity_rel_threshold: f64,
    /// Number of repeated gradient samples used to estimate gradient noise (`>= 0`).
    pub noise_repeats: usize,
    /// Number of full-batch gradient-descent epochs for the classifier.
    pub classifier_epochs: usize,
    /// Classifier learning rate.
    pub classifier_lr: f64,
    /// Classifier L2 regularization strength.
    pub classifier_l2: f64,
    /// Number of full-batch gradient-descent epochs for the learning-rate regressor.
    pub regressor_epochs: usize,
    /// Regressor learning rate.
    pub regressor_lr: f64,
    /// Regressor L2 regularization strength.
    pub regressor_l2: f64,
    /// Minimum learning rate the regressor may recommend.
    pub min_learning_rate: f64,
    /// Maximum learning rate the regressor may recommend.
    pub max_learning_rate: f64,
    /// Seed for the random curvature directions (keeps extraction deterministic).
    pub seed: u64,
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        Self {
            num_probe_points: 6,
            probe_step: 0.05,
            num_curvature_directions: 12,
            num_random_curvature_dirs: 4,
            finite_diff_eps: 1e-3,
            sparsity_rel_threshold: 1e-3,
            noise_repeats: 5,
            classifier_epochs: 800,
            classifier_lr: 0.2,
            classifier_l2: 1e-4,
            regressor_epochs: 800,
            regressor_lr: 0.1,
            regressor_l2: 1e-4,
            min_learning_rate: 1e-6,
            max_learning_rate: 1.0,
            seed: 0,
        }
    }
}

impl ZeroShotConfig {
    /// Validate the configuration, returning an error describing the first problem found.
    pub fn validate(&self) -> Result<()> {
        if self.num_probe_points == 0 {
            return Err(OptimError::InvalidConfig(
                "num_probe_points must be >= 1".to_string(),
            ));
        }
        if !(self.probe_step.is_finite() && self.probe_step > 0.0) {
            return Err(OptimError::InvalidConfig(
                "probe_step must be finite and positive".to_string(),
            ));
        }
        if !(self.finite_diff_eps.is_finite() && self.finite_diff_eps > 0.0) {
            return Err(OptimError::InvalidConfig(
                "finite_diff_eps must be finite and positive".to_string(),
            ));
        }
        if self.num_curvature_directions == 0 && self.num_random_curvature_dirs == 0 {
            return Err(OptimError::InvalidConfig(
                "at least one curvature direction is required".to_string(),
            ));
        }
        if self.classifier_epochs == 0 || self.regressor_epochs == 0 {
            return Err(OptimError::InvalidConfig(
                "classifier_epochs and regressor_epochs must be >= 1".to_string(),
            ));
        }
        if !(self.classifier_lr > 0.0 && self.regressor_lr > 0.0) {
            return Err(OptimError::InvalidConfig(
                "classifier_lr and regressor_lr must be positive".to_string(),
            ));
        }
        if self.classifier_l2 < 0.0 || self.regressor_l2 < 0.0 {
            return Err(OptimError::InvalidConfig(
                "L2 strengths must be non-negative".to_string(),
            ));
        }
        if !(self.min_learning_rate > 0.0
            && self.max_learning_rate > self.min_learning_rate
            && self.max_learning_rate.is_finite())
        {
            return Err(OptimError::InvalidConfig(
                "require 0 < min_learning_rate < max_learning_rate < inf".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------
// Zero-shot selector
// ---------------------------------------------------------------------------------------

/// Zero-shot optimizer selector: an offline-fitted meta-model plus the meta-feature
/// extractor it consumes.
///
/// The selector is fit once on a meta-dataset of [`MetaExample`]s (see
/// [`ZeroShotSelector::fit`]) and thereafter answers [`ZeroShotSelector::recommend`]
/// queries for brand-new tasks with no further training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotSelector {
    config: ZeroShotConfig,
    fitted: bool,
    /// Per-feature mean over the meta-dataset (standardization).
    feature_mean: Array1<f64>,
    /// Per-feature standard deviation over the meta-dataset (standardization).
    feature_std: Array1<f64>,
    /// Logistic-regression weights, shape `(NUM_CLASSES, NUM_FEATURES)`.
    clf_w: Array2<f64>,
    /// Logistic-regression biases, length `NUM_CLASSES`.
    clf_b: Array1<f64>,
    /// Learning-rate regressor weights, length `NUM_FEATURES`.
    reg_w: Array1<f64>,
    /// Learning-rate regressor bias.
    reg_b: f64,
}

impl ZeroShotSelector {
    /// Create an unfitted selector with the given configuration.
    pub fn new(config: ZeroShotConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            fitted: false,
            feature_mean: Array1::zeros(NUM_FEATURES),
            feature_std: Array1::ones(NUM_FEATURES),
            clf_w: Array2::zeros((NUM_CLASSES, NUM_FEATURES)),
            clf_b: Array1::zeros(NUM_CLASSES),
            reg_w: Array1::zeros(NUM_FEATURES),
            reg_b: 0.0,
        })
    }

    /// Create an unfitted selector with the default configuration.
    pub fn with_default_config() -> Result<Self> {
        Self::new(ZeroShotConfig::default())
    }

    /// Whether [`ZeroShotSelector::fit`] has been called successfully.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// The active configuration.
    pub fn config(&self) -> &ZeroShotConfig {
        &self.config
    }

    // -- meta-feature extraction --------------------------------------------------------

    /// Extract the meta-feature vector for a task by lightweight probing.
    ///
    /// This performs only gradient *evaluations* (no optimization steps on the task) and
    /// is deterministic: the random curvature directions are drawn from a generator seeded
    /// by [`ZeroShotConfig::seed`], and the noise estimate uses index-seeded gradient
    /// samples, so repeated calls return identical features.
    pub fn extract_meta_features<P>(&self, probe: &P) -> Result<MetaFeatures>
    where
        P: TaskProbe + ?Sized,
    {
        let dim = probe.dimension();
        if dim == 0 {
            return Err(OptimError::InvalidConfig(
                "probe dimension must be positive".to_string(),
            ));
        }
        let x0 = probe.initial_params();
        if x0.len() != dim {
            return Err(OptimError::InvalidConfig(format!(
                "initial_params length {} != probe dimension {dim}",
                x0.len()
            )));
        }

        let cfg = &self.config;

        // --- Probe-path statistics: walk a short normalized-descent path, collecting the
        //     gradient at each visited point plus the cosine between successive gradients.
        let mut path_norms: Vec<f64> = Vec::with_capacity(cfg.num_probe_points);
        let mut sparsity_acc = 0.0;
        let mut stability_acc = 0.0;
        let mut stability_count = 0usize;
        let mut prev_grad: Option<Array1<f64>> = None;
        let mut x = x0.clone();

        for _ in 0..cfg.num_probe_points {
            let grad = probe.gradient_at(&x);
            if grad.len() != dim {
                return Err(OptimError::ComputationError(format!(
                    "probe gradient length {} != dimension {dim}",
                    grad.len()
                )));
            }
            let norm = l2_norm(&grad);
            path_norms.push(norm);
            sparsity_acc += fraction_near_zero(&grad, cfg.sparsity_rel_threshold);

            if let Some(prev) = &prev_grad {
                stability_acc += cosine_similarity(prev, &grad);
                stability_count += 1;
            }

            // Normalized-descent step; if the gradient vanishes we stay put.
            if norm > EPS {
                let scale = cfg.probe_step / norm;
                x = &x - &grad.mapv(|g| g * scale);
            }
            prev_grad = Some(grad);
        }

        let grad_norm_mean = mean(&path_norms);
        let grad_norm_cv = coefficient_of_variation(&path_norms);
        let grad_sparsity = sparsity_acc / cfg.num_probe_points as f64;
        let direction_stability = if stability_count > 0 {
            stability_acc / stability_count as f64
        } else {
            0.0
        };

        // --- Curvature / condition-number proxy: finite-difference the gradient along
        //     coordinate directions (which recover the Hessian diagonal exactly for a
        //     separable quadratic) plus a few random directions.
        let g_base = probe.gradient_at(&x0);
        if g_base.len() != dim {
            return Err(OptimError::ComputationError(
                "probe gradient length mismatch at base point".to_string(),
            ));
        }
        let h = cfg.finite_diff_eps * (1.0 + rms(&x0));
        let mut curvatures: Vec<f64> = Vec::new();

        let n_coord = dim.min(cfg.num_curvature_directions);
        for i in 0..n_coord {
            let mut xp = x0.clone();
            xp[i] += h;
            let gp = probe.gradient_at(&xp);
            if gp.len() != dim {
                return Err(OptimError::ComputationError(
                    "probe gradient length mismatch during curvature probe".to_string(),
                ));
            }
            // e_iᵀ H e_i = ∂g_i / ∂x_i ≈ (g_i(x + h e_i) − g_i(x)) / h.
            curvatures.push(((gp[i] - g_base[i]) / h).abs());
        }

        let mut rng = Random::seed(cfg.seed);
        for _ in 0..cfg.num_random_curvature_dirs {
            let mut dir = Array1::from_shape_fn(dim, |_| rng.random_range(-1.0..1.0));
            let dir_norm = l2_norm(&dir);
            if dir_norm <= EPS {
                continue;
            }
            dir.mapv_inplace(|v| v / dir_norm);
            let xp = &x0 + &dir.mapv(|v| v * h);
            let gp = probe.gradient_at(&xp);
            if gp.len() != dim {
                return Err(OptimError::ComputationError(
                    "probe gradient length mismatch during random curvature probe".to_string(),
                ));
            }
            // dᵀ H d ≈ (g(x + h d) − g(x)) · d / h.
            let diff = &gp - &g_base;
            curvatures.push((dot(&diff, &dir) / h).abs());
        }

        let mean_curvature = mean(&curvatures);
        let mean_log_curvature = (1.0 + mean_curvature).ln();
        let log_condition_number = log_condition(&curvatures);

        // --- Noise estimate: per-component coefficient of variation across repeated,
        //     index-seeded gradient samples at the base point.
        let noise_estimate = if cfg.noise_repeats >= 2 {
            let samples: Vec<Array1<f64>> = (0..cfg.noise_repeats)
                .map(|k| probe.gradient_sample(&x0, k))
                .collect();
            mean_component_cv(&samples, dim)
        } else {
            0.0
        };

        let log_dimension = (dim as f64 + 1.0).ln();
        let log_param_scale = (1.0 + rms(&x0)).ln();

        let values = Array1::from_vec(vec![
            log_dimension,
            grad_norm_mean,
            grad_norm_cv,
            grad_sparsity,
            mean_log_curvature,
            log_condition_number,
            direction_stability,
            noise_estimate,
            log_param_scale,
        ]);

        // Guard against non-finite values leaking into the meta-model.
        if values.iter().any(|v| !v.is_finite()) {
            return Err(OptimError::ComputationError(
                "meta-feature extraction produced a non-finite value".to_string(),
            ));
        }

        MetaFeatures::from_values(values)
    }

    /// Standardize a raw feature vector with the fitted mean/standard-deviation.
    ///
    /// A feature whose standard deviation is below [`STD_FLOOR`] is treated as
    /// zero-variance and divided by one, so the result is never `NaN`.
    fn standardize(&self, raw: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(NUM_FEATURES, |i| {
            let std = self.feature_std[i];
            let denom = if std > STD_FLOOR { std } else { 1.0 };
            (raw[i] - self.feature_mean[i]) / denom
        })
    }

    // -- offline fitting ----------------------------------------------------------------

    /// Fit the meta-model on an offline meta-dataset.
    ///
    /// This learns the standardization statistics, the multinomial logistic-regression
    /// classifier (softmax cross-entropy, full-batch gradient descent with L2), and the
    /// linear `log10(learning_rate)` regressor (MSE, full-batch gradient descent with L2).
    pub fn fit(&mut self, examples: &[MetaExample]) -> Result<()> {
        if examples.is_empty() {
            return Err(OptimError::InsufficientData(
                "meta-dataset must contain at least one example".to_string(),
            ));
        }
        for ex in examples {
            if ex.features.values.len() != NUM_FEATURES {
                return Err(OptimError::InvalidConfig(
                    "a meta-example has the wrong feature length".to_string(),
                ));
            }
            if !(ex.best_learning_rate.is_finite() && ex.best_learning_rate > 0.0) {
                return Err(OptimError::InvalidConfig(
                    "a meta-example has a non-positive learning rate".to_string(),
                ));
            }
        }

        let n = examples.len();
        let n_f = n as f64;

        // Standardization statistics (population mean / standard deviation).
        let mut mean_vec = Array1::<f64>::zeros(NUM_FEATURES);
        for ex in examples {
            mean_vec = &mean_vec + &ex.features.values;
        }
        mean_vec.mapv_inplace(|v| v / n_f);

        let mut var_vec = Array1::<f64>::zeros(NUM_FEATURES);
        for ex in examples {
            let diff = &ex.features.values - &mean_vec;
            var_vec = &var_vec + &diff.mapv(|d| d * d);
        }
        var_vec.mapv_inplace(|v| v / n_f);
        let std_vec = var_vec.mapv(f64::sqrt);

        self.feature_mean = mean_vec;
        self.feature_std = std_vec;

        // Standardize all rows once.
        let xs: Vec<Array1<f64>> = examples
            .iter()
            .map(|ex| self.standardize(&ex.features.values))
            .collect();
        let y_class: Vec<usize> = examples
            .iter()
            .map(|ex| ex.best_optimizer.index())
            .collect();
        let y_lr: Vec<f64> = examples
            .iter()
            .map(|ex| ex.best_learning_rate.log10())
            .collect();

        self.train_classifier(&xs, &y_class);
        self.train_regressor(&xs, &y_lr);
        self.fitted = true;
        Ok(())
    }

    /// Train the softmax classifier by full-batch gradient descent on cross-entropy.
    fn train_classifier(&mut self, xs: &[Array1<f64>], y: &[usize]) {
        let n = xs.len().max(1) as f64;
        let lr = self.config.classifier_lr;
        let l2 = self.config.classifier_l2;
        let mut w = Array2::<f64>::zeros((NUM_CLASSES, NUM_FEATURES));
        let mut b = Array1::<f64>::zeros(NUM_CLASSES);

        for _ in 0..self.config.classifier_epochs {
            let mut grad_w = Array2::<f64>::zeros((NUM_CLASSES, NUM_FEATURES));
            let mut grad_b = Array1::<f64>::zeros(NUM_CLASSES);

            for (x, &yi) in xs.iter().zip(y.iter()) {
                let logits =
                    Array1::from_shape_fn(NUM_CLASSES, |c| linear_score(&w.row(c), x) + b[c]);
                let probs = softmax(&logits);
                for (c, mut row) in grad_w.outer_iter_mut().enumerate() {
                    let target = if c == yi { 1.0 } else { 0.0 };
                    let diff = probs[c] - target;
                    for (slot, &xj) in row.iter_mut().zip(x.iter()) {
                        *slot += diff * xj;
                    }
                    grad_b[c] += diff;
                }
            }

            grad_w.mapv_inplace(|g| g / n);
            grad_b.mapv_inplace(|g| g / n);
            grad_w = &grad_w + &w.mapv(|wv| wv * l2);

            w = &w - &grad_w.mapv(|g| g * lr);
            b = &b - &grad_b.mapv(|g| g * lr);
        }

        self.clf_w = w;
        self.clf_b = b;
    }

    /// Train the linear `log10(lr)` regressor by full-batch gradient descent on MSE.
    fn train_regressor(&mut self, xs: &[Array1<f64>], y: &[f64]) {
        let n = xs.len().max(1) as f64;
        let lr = self.config.regressor_lr;
        let l2 = self.config.regressor_l2;
        let mut w = Array1::<f64>::zeros(NUM_FEATURES);
        let mut b = 0.0f64;

        for _ in 0..self.config.regressor_epochs {
            let mut grad_w = Array1::<f64>::zeros(NUM_FEATURES);
            let mut grad_b = 0.0f64;

            for (x, &yi) in xs.iter().zip(y.iter()) {
                let pred = dot(&w, x) + b;
                let err = pred - yi;
                for (slot, &xj) in grad_w.iter_mut().zip(x.iter()) {
                    *slot += err * xj;
                }
                grad_b += err;
            }

            let scale = 2.0 / n;
            grad_w.mapv_inplace(|g| g * scale);
            grad_b *= scale;
            grad_w = &grad_w + &w.mapv(|wv| wv * (2.0 * l2));

            w = &w - &grad_w.mapv(|g| g * lr);
            b -= grad_b * lr;
        }

        self.reg_w = w;
        self.reg_b = b;
    }

    // -- zero-shot inference ------------------------------------------------------------

    /// Recommend an optimizer and hyper-parameters for a brand-new task.
    ///
    /// Extracts meta-features from the probe, standardizes them with the fitted statistics,
    /// and runs the classifier + regressor. Returns an error if the selector has not been
    /// fit yet.
    pub fn recommend<P>(&self, probe: &P) -> Result<OptimizerRecommendation>
    where
        P: TaskProbe + ?Sized,
    {
        if !self.fitted {
            return Err(OptimError::InvalidState(
                "ZeroShotSelector must be fit before calling recommend".to_string(),
            ));
        }
        let features = self.extract_meta_features(probe)?;
        self.predict(&features)
    }

    /// Predict directly from an already-extracted meta-feature vector.
    ///
    /// Returns an error if the selector has not been fit yet.
    pub fn predict(&self, features: &MetaFeatures) -> Result<OptimizerRecommendation> {
        if !self.fitted {
            return Err(OptimError::InvalidState(
                "ZeroShotSelector must be fit before calling predict".to_string(),
            ));
        }
        if features.values.len() != NUM_FEATURES {
            return Err(OptimError::InvalidConfig(
                "meta-feature vector has the wrong length".to_string(),
            ));
        }

        let xs = self.standardize(&features.values);

        let logits = Array1::from_shape_fn(NUM_CLASSES, |c| {
            linear_score(&self.clf_w.row(c), &xs) + self.clf_b[c]
        });
        let probs = softmax(&logits);
        let (class_index, confidence) = argmax(&probs);
        let optimizer = OptimizerKind::from_index(class_index)?;

        let log_lr = dot(&self.reg_w, &xs) + self.reg_b;
        let learning_rate = if log_lr.is_finite() {
            10f64
                .powf(log_lr)
                .clamp(self.config.min_learning_rate, self.config.max_learning_rate)
        } else {
            // Non-finite prediction: fall back to the geometric midpoint of the range.
            (self.config.min_learning_rate * self.config.max_learning_rate).sqrt()
        };

        Ok(OptimizerRecommendation {
            optimizer,
            learning_rate,
            hyperparameters: optimizer.default_hyperparameters(),
            confidence: confidence.clamp(0.0, 1.0),
        })
    }

    // -- persistence --------------------------------------------------------------------

    /// Serialize the fitted selector (including its learned parameters) to JSON on disk.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load a previously-saved selector from JSON on disk.
    pub fn load_json(path: &Path) -> Result<Self> {
        let serialized = std::fs::read_to_string(path)?;
        let selector: ZeroShotSelector = serde_json::from_str(&serialized)?;
        Ok(selector)
    }
}

// ---------------------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------------------

/// Deterministically mix a base seed with a sample index into a fresh seed.
fn mix_seed(base: u64, index: u64) -> u64 {
    let mut h = base ^ 0x9E37_79B9_7F4A_7C15;
    h = h.wrapping_add(index.wrapping_add(1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Euclidean (L2) norm.
fn l2_norm(arr: &Array1<f64>) -> f64 {
    arr.iter().fold(0.0, |acc, &x| acc + x * x).sqrt()
}

/// Inner product of two equally-sized vectors (extra elements of the longer are ignored).
fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |acc, (&x, &y)| acc + x * y)
}

/// Inner product of a matrix-row view with a vector.
fn linear_score(row: &scirs2_core::ndarray::ArrayView1<f64>, x: &Array1<f64>) -> f64 {
    row.iter()
        .zip(x.iter())
        .fold(0.0, |acc, (&w, &v)| acc + w * v)
}

/// Cosine similarity; returns `0` if either vector is (numerically) zero.
fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na <= EPS || nb <= EPS {
        0.0
    } else {
        (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Root-mean-square of a vector.
fn rms(arr: &Array1<f64>) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    (arr.iter().fold(0.0, |acc, &x| acc + x * x) / arr.len() as f64).sqrt()
}

/// Arithmetic mean of a slice (`0` if empty).
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation divided by the mean (`0` if the mean is ~0 or empty).
fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    if m.abs() <= EPS {
        return 0.0;
    }
    let var = values.iter().fold(0.0, |acc, &x| acc + (x - m) * (x - m)) / values.len() as f64;
    var.sqrt() / m.abs()
}

/// Fraction of components of `grad` whose magnitude is below `rel_threshold · max|grad|`.
fn fraction_near_zero(grad: &Array1<f64>, rel_threshold: f64) -> f64 {
    if grad.is_empty() {
        return 0.0;
    }
    let max_abs = grad.iter().fold(0.0_f64, |acc, &g| acc.max(g.abs()));
    if max_abs <= EPS {
        // An all-zero gradient is maximally sparse.
        return 1.0;
    }
    let threshold = rel_threshold * max_abs;
    let near = grad.iter().filter(|&&g| g.abs() < threshold).count();
    near as f64 / grad.len() as f64
}

/// `ln(max_positive / min_positive)` over the curvature magnitudes (`0` if degenerate).
fn log_condition(curvatures: &[f64]) -> f64 {
    let mut max_c = 0.0_f64;
    let mut min_c = f64::INFINITY;
    for &c in curvatures {
        if c > EPS {
            if c > max_c {
                max_c = c;
            }
            if c < min_c {
                min_c = c;
            }
        }
    }
    if max_c > EPS && min_c.is_finite() && min_c > EPS {
        (max_c / min_c).ln()
    } else {
        0.0
    }
}

/// Mean over components of the coefficient of variation across repeated gradient samples.
fn mean_component_cv(samples: &[Array1<f64>], dim: usize) -> f64 {
    if samples.len() < 2 || dim == 0 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mut acc = 0.0;
    for j in 0..dim {
        let mut sum = 0.0;
        for s in samples {
            sum += s.get(j).copied().unwrap_or(0.0);
        }
        let m = sum / n;
        let mut var = 0.0;
        for s in samples {
            let v = s.get(j).copied().unwrap_or(0.0);
            var += (v - m) * (v - m);
        }
        var /= n;
        acc += var.sqrt() / (m.abs() + EPS);
    }
    acc / dim as f64
}

/// Numerically-stable softmax. The result is strictly positive and sums to one.
fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    if logits.is_empty() {
        return Array1::zeros(0);
    }
    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let exps = logits.mapv(|x| (x - max_logit).exp());
    let sum = exps.iter().sum::<f64>();
    if sum <= EPS {
        return Array1::from_elem(logits.len(), 1.0 / logits.len() as f64);
    }
    exps.mapv(|x| x / sum)
}

/// Index of the maximum entry and its value (`(0, 0.0)` for an empty vector).
fn argmax(values: &Array1<f64>) -> (usize, f64) {
    let mut best_index = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_value {
            best_value = v;
            best_index = i;
        }
    }
    if best_value.is_finite() {
        (best_index, best_value)
    } else {
        (0, 0.0)
    }
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn selector() -> ZeroShotSelector {
        ZeroShotSelector::with_default_config().expect("default selector")
    }

    /// Build a meta-example by extracting features from a probe and labeling it.
    fn labeled_example(
        sel: &ZeroShotSelector,
        probe: &QuadraticProbe,
        kind: OptimizerKind,
        lr: f64,
    ) -> MetaExample {
        let feats = sel.extract_meta_features(probe).expect("extract");
        MetaExample::new(feats, kind, lr).expect("meta example")
    }

    #[test]
    fn test_extract_features_on_known_quadratic_condition_number() {
        let sel = selector();
        // Curvatures [1, 1, 1, 100] => Hessian eigenvalues {1, 100} => condition 100.
        let curvature = Array1::from_vec(vec![1.0, 1.0, 1.0, 100.0]);
        let init = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let probe = QuadraticProbe::new(curvature, init).expect("probe");

        let feats = sel.extract_meta_features(&probe).expect("extract");
        let expected = 100f64.ln();
        assert!(
            (feats.log_condition_number() - expected).abs() < 1e-3,
            "log condition number should reflect curvature 100, expected {expected}, got {}",
            feats.log_condition_number()
        );
        // Dimensionality feature.
        assert!((feats.log_dimension() - 5f64.ln()).abs() < 1e-9);
        // Deterministic, noise-free quadratic => zero noise estimate.
        assert!(feats.noise_estimate().abs() < 1e-12);
        // All features must be finite.
        assert!(feats.as_array().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_isotropic_quadratic_is_well_conditioned() {
        let sel = selector();
        let probe = QuadraticProbe::isotropic(5, 3.0, 1.0).expect("probe");
        let feats = sel.extract_meta_features(&probe).expect("extract");
        // Equal curvatures => condition number ~1 => log ~0.
        assert!(
            feats.log_condition_number().abs() < 1e-6,
            "isotropic bowl should have log-condition ~ 0, got {}",
            feats.log_condition_number()
        );
    }

    #[test]
    fn test_extraction_is_deterministic() {
        let sel = selector();
        let probe = QuadraticProbe::new(
            Array1::from_vec(vec![1.0, 7.0, 30.0, 2.0]),
            Array1::from_vec(vec![0.5, -1.0, 2.0, 1.5]),
        )
        .expect("probe");
        let a = sel.extract_meta_features(&probe).expect("extract a");
        let b = sel.extract_meta_features(&probe).expect("extract b");
        assert_eq!(a, b, "extraction must be deterministic");
    }

    #[test]
    fn test_noise_feature_detects_stochastic_probe() {
        let sel = selector();
        let curvature = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let init = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let det = QuadraticProbe::new(curvature.clone(), init.clone()).expect("det");
        let det_feats = sel.extract_meta_features(&det).expect("extract det");
        assert!(det_feats.noise_estimate().abs() < 1e-12);

        let noisy = QuadraticProbe::new(curvature, init)
            .expect("noisy")
            .with_noise(0.5, 123);
        let noisy_feats_a = sel.extract_meta_features(&noisy).expect("extract noisy a");
        let noisy_feats_b = sel.extract_meta_features(&noisy).expect("extract noisy b");
        assert!(
            noisy_feats_a.noise_estimate() > 1e-3,
            "stochastic probe should have a positive noise estimate, got {}",
            noisy_feats_a.noise_estimate()
        );
        // Index-seeded samples keep extraction reproducible even for a noisy oracle.
        assert_eq!(noisy_feats_a, noisy_feats_b);
    }

    #[test]
    fn test_grad_sparsity_detects_zero_components() {
        let sel = selector();
        // A coordinate already at its optimum contributes a zero gradient component.
        let curvature = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let init = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let probe = QuadraticProbe::new(curvature, init).expect("probe");
        let feats = sel.extract_meta_features(&probe).expect("extract");
        assert!(
            feats.grad_sparsity() >= 0.5 - 1e-9,
            "two of four components are zero, sparsity should be >= 0.5, got {}",
            feats.grad_sparsity()
        );
    }

    #[test]
    fn test_rosenbrock_probe_extracts_finite_features() {
        let sel = selector();
        let probe = RosenbrockProbe::new(Array1::from_vec(vec![-1.2, 1.0, -0.5])).expect("probe");
        let feats = sel.extract_meta_features(&probe).expect("extract");
        assert!(feats.as_array().iter().all(|v| v.is_finite()));
        // Rosenbrock is strongly ill-conditioned => sizeable log condition number.
        assert!(feats.log_condition_number() > 0.5);
    }

    #[test]
    fn test_recommend_before_fit_errors() {
        let sel = selector();
        let probe = QuadraticProbe::isotropic(4, 1.0, 1.0).expect("probe");
        let result = sel.recommend(&probe);
        assert!(result.is_err(), "recommend before fit must error");

        let feats = sel.extract_meta_features(&probe).expect("extract");
        assert!(
            sel.predict(&feats).is_err(),
            "predict before fit must error"
        );
    }

    #[test]
    fn test_fit_rejects_empty_dataset() {
        let mut sel = selector();
        assert!(sel.fit(&[]).is_err(), "fit on empty dataset must error");
    }

    /// The central behavioral test: after fitting on a meta-dataset following the rule
    /// "ill-conditioned => Adam, well-conditioned => SGD", held-out probes obeying the
    /// rule are classified accordingly.
    #[test]
    fn test_rule_illconditioned_adam_wellconditioned_sgd() {
        let sel = selector();
        let dim = 5usize;
        let mut examples: Vec<MetaExample> = Vec::new();

        // Well-conditioned (isotropic) training tasks => SGD with a larger learning rate.
        for k in 0..8 {
            let c = 1.0 + 0.1 * k as f64;
            let probe = QuadraticProbe::isotropic(dim, c, 1.0 + 0.05 * k as f64).expect("probe");
            examples.push(labeled_example(&sel, &probe, OptimizerKind::Sgd, 0.05));
        }
        // Ill-conditioned (anisotropic) training tasks => Adam with a smaller learning rate.
        for k in 0..8 {
            let big = 50.0 + 15.0 * k as f64;
            let mut curvature = Array1::from_elem(dim, 1.0);
            curvature[dim - 1] = big;
            let init = Array1::from_elem(dim, 1.0 + 0.05 * k as f64);
            let probe = QuadraticProbe::new(curvature, init).expect("probe");
            examples.push(labeled_example(&sel, &probe, OptimizerKind::Adam, 0.001));
        }

        let mut fitted = selector();
        fitted.fit(&examples).expect("fit");
        assert!(fitted.is_fitted());

        // Held-out ill-conditioned probe (condition 120, not present in training).
        let mut ill_curv = Array1::from_elem(dim, 1.0);
        ill_curv[dim - 1] = 120.0;
        let ill_probe = QuadraticProbe::new(ill_curv, Array1::from_elem(dim, 1.3)).expect("probe");
        let ill_rec = fitted.recommend(&ill_probe).expect("recommend ill");
        assert_eq!(
            ill_rec.optimizer,
            OptimizerKind::Adam,
            "ill-conditioned held-out task should map to Adam"
        );

        // Held-out well-conditioned probe (isotropic, not present in training).
        let well_probe = QuadraticProbe::isotropic(dim, 2.0, 0.8).expect("probe");
        let well_rec = fitted.recommend(&well_probe).expect("recommend well");
        assert_eq!(
            well_rec.optimizer,
            OptimizerKind::Sgd,
            "well-conditioned held-out task should map to SGD"
        );

        // Confidence is a probability.
        for rec in [&ill_rec, &well_rec] {
            assert!(
                (0.0..=1.0).contains(&rec.confidence),
                "confidence must be in [0, 1], got {}",
                rec.confidence
            );
            assert!(rec.confidence >= 0.5, "clear cases should be confident");
        }

        // The learning-rate regressor should order the rates as in the meta-dataset.
        assert!(
            ill_rec.learning_rate < well_rec.learning_rate,
            "ill-conditioned lr ({}) should be below well-conditioned lr ({})",
            ill_rec.learning_rate,
            well_rec.learning_rate
        );
        assert!(
            ill_rec.learning_rate >= fitted.config().min_learning_rate
                && well_rec.learning_rate <= fitted.config().max_learning_rate,
            "learning rates must stay within the configured range"
        );
    }

    #[test]
    fn test_logistic_regression_separates_synthetic_features() {
        // Construct meta-examples directly (bypassing extraction) with a single strongly
        // discriminative feature (the condition number, index 5) and verify the trained
        // classifier separates two classes on held-out feature vectors.
        let make = |cond: f64, kind: OptimizerKind| -> MetaExample {
            let mut v = vec![1.0; NUM_FEATURES];
            v[5] = cond;
            MetaExample::new(
                MetaFeatures::from_values(Array1::from_vec(v)).expect("feats"),
                kind,
                0.01,
            )
            .expect("example")
        };

        let mut examples = Vec::new();
        for k in 0..6 {
            examples.push(make(0.0 + 0.05 * k as f64, OptimizerKind::Sgd));
            examples.push(make(4.0 + 0.05 * k as f64, OptimizerKind::Adam));
        }

        let mut sel = selector();
        sel.fit(&examples).expect("fit");

        let mut low = vec![1.0; NUM_FEATURES];
        low[5] = 0.1;
        let low_rec = sel
            .predict(&MetaFeatures::from_values(Array1::from_vec(low)).expect("feats"))
            .expect("predict low");
        assert_eq!(low_rec.optimizer, OptimizerKind::Sgd);

        let mut high = vec![1.0; NUM_FEATURES];
        high[5] = 4.2;
        let high_rec = sel
            .predict(&MetaFeatures::from_values(Array1::from_vec(high)).expect("feats"))
            .expect("predict high");
        assert_eq!(high_rec.optimizer, OptimizerKind::Adam);
    }

    #[test]
    fn test_standardization_zero_variance_no_nan() {
        // Every example shares an identical feature vector => every feature has zero
        // variance. Standardization must not produce NaN/inf.
        let v = Array1::from_vec(vec![2.0; NUM_FEATURES]);
        let examples: Vec<MetaExample> = (0..4)
            .map(|_| {
                MetaExample::new(
                    MetaFeatures::from_values(v.clone()).expect("feats"),
                    OptimizerKind::RmsProp,
                    0.01,
                )
                .expect("example")
            })
            .collect();

        let mut sel = selector();
        sel.fit(&examples).expect("fit");

        let standardized = sel.standardize(&v);
        assert!(
            standardized.iter().all(|x| x.is_finite()),
            "zero-variance standardization must stay finite"
        );
        // On a training point each zero-variance feature standardizes exactly to zero.
        assert!(standardized.iter().all(|&x| x.abs() < 1e-12));

        let rec = sel
            .predict(&MetaFeatures::from_values(v).expect("feats"))
            .expect("predict");
        assert!(rec.learning_rate.is_finite() && rec.confidence.is_finite());
        // With a single class present it must be the recommendation.
        assert_eq!(rec.optimizer, OptimizerKind::RmsProp);
    }

    #[test]
    fn test_meta_features_length_validation() {
        let too_short = Array1::from_vec(vec![0.0; NUM_FEATURES - 1]);
        assert!(MetaFeatures::from_values(too_short).is_err());
        let ok = Array1::from_vec(vec![0.0; NUM_FEATURES]);
        assert!(MetaFeatures::from_values(ok).is_ok());
    }

    #[test]
    fn test_meta_example_rejects_bad_learning_rate() {
        let feats =
            MetaFeatures::from_values(Array1::from_vec(vec![0.0; NUM_FEATURES])).expect("feats");
        assert!(MetaExample::new(feats.clone(), OptimizerKind::Sgd, 0.0).is_err());
        assert!(MetaExample::new(feats.clone(), OptimizerKind::Sgd, -1.0).is_err());
        assert!(MetaExample::new(feats, OptimizerKind::Sgd, 0.01).is_ok());
    }

    #[test]
    fn test_optimizer_kind_index_roundtrip() {
        for kind in OptimizerKind::all() {
            let idx = kind.index();
            assert_eq!(OptimizerKind::from_index(idx).expect("roundtrip"), kind);
        }
        assert!(OptimizerKind::from_index(NUM_CLASSES).is_err());
    }

    #[test]
    fn test_config_validation_rejects_bad_inputs() {
        let bad_points = ZeroShotConfig {
            num_probe_points: 0,
            ..ZeroShotConfig::default()
        };
        assert!(bad_points.validate().is_err());

        let bad_range = ZeroShotConfig {
            min_learning_rate: 1.0,
            max_learning_rate: 0.1,
            ..ZeroShotConfig::default()
        };
        assert!(bad_range.validate().is_err());

        let bad_epochs = ZeroShotConfig {
            classifier_epochs: 0,
            ..ZeroShotConfig::default()
        };
        assert!(bad_epochs.validate().is_err());

        assert!(ZeroShotConfig::default().validate().is_ok());
    }

    #[test]
    fn test_softmax_is_distribution() {
        let logits = Array1::from_vec(vec![-1.0, 0.0, 2.0, 0.5, -3.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(probs.iter().all(|&p| p > 0.0));
        let (idx, _) = argmax(&probs);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let sel = selector();
        let probe = QuadraticProbe::isotropic(4, 2.0, 1.0).expect("probe");
        let example = labeled_example(&sel, &probe, OptimizerKind::Adam, 0.01);
        let mut fitted = selector();
        fitted.fit(std::slice::from_ref(&example)).expect("fit");

        let mut path = std::env::temp_dir();
        path.push(format!("optirs_zero_shot_{}.json", std::process::id()));
        fitted.save_json(&path).expect("save");
        let loaded = ZeroShotSelector::load_json(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        assert!(loaded.is_fitted());
        let original = fitted.recommend(&probe).expect("recommend original");
        let reloaded = loaded.recommend(&probe).expect("recommend reloaded");
        assert_eq!(original.optimizer, reloaded.optimizer);
        assert!((original.learning_rate - reloaded.learning_rate).abs() < 1e-9);
    }
}
