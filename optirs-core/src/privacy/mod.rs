// Differential Privacy support for optimizers
//
// This module provides differential privacy mechanisms for machine learning
// optimization, including DP-SGD with a Renyi/moments accountant for privacy
// budget tracking.
//
// # Reading the privacy numbers this module reports
//
// * **Epsilon is the budget.** It accumulates with every mechanism
//   application and is enforced *before* a noisy gradient is released.
// * **Delta is a reporting parameter, not a spend.** The same composed
//   mechanism can be described at any delta, trading it against epsilon.
//   Nothing in this crate consumes delta additively; an implementation that
//   did would exhaust itself on the first step.
// * **Per-example clipping is what makes DP-SGD private.** Use
//   [`DifferentiallyPrivateOptimizer::dp_step_per_example`] (or
//   [`DifferentiallyPrivateOptimizer::dp_step_presummed`] if you clip and sum
//   yourself). Clipping an already-aggregated batch gradient does *not* bound
//   any single example's influence and therefore does not deliver the
//   standard DP-SGD guarantee -- see
//   [`DifferentiallyPrivateOptimizer::dp_step`].

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use scirs2_core::random::thread_rng;
use std::collections::VecDeque;
use std::fmt::Debug;

pub mod accountant;
pub mod byzantine_tolerance;
pub mod differential_privacy; // New modular differential privacy
pub mod dp_sgd;
pub mod enhanced_audit;
pub mod federated; // New modular federated privacy
pub mod federated_privacy;
pub mod moment_accountant;
pub mod noise_mechanisms;
pub mod private_hyperparameter_optimization;
pub mod renyi_accountant;
pub mod secure_aggregation;
pub mod secure_multiparty;
pub mod utility_analysis;

use crate::optimizers::Optimizer;

// Re-export key utility analysis types
pub use utility_analysis::{
    AnalysisConfig, AnalysisMetadata, BudgetRecommendations, OptimalConfiguration, ParetoPoint,
    PrivacyConfiguration, PrivacyParameterSpace, PrivacyRiskAssessment, PrivacyUtilityAnalyzer,
    PrivacyUtilityResults, RobustnessResults, SensitivityResults, StatisticalTestResults,
    UtilityMetric,
};

// Re-export modular federated privacy types
pub use federated::{
    ByzantineRobustAggregator, ByzantineRobustConfig, ByzantineRobustMethod, ClientComposition,
    CompositionStats, CrossDeviceConfig, CrossDevicePrivacyManager, DeviceProfile, DeviceType,
    FederatedCompositionAnalyzer, FederatedCompositionMethod, OutlierDetectionResult,
    ReputationSystemConfig, RoundComposition, SecureAggregationConfig, SecureAggregationPlan,
    SecureAggregator, SeedSharingMethod, StatisticalTestConfig, StatisticalTestType, TemporalEvent,
    TemporalEventType,
};

// Re-export modular differential privacy types
pub use differential_privacy::{
    AmplificationConfig, AmplificationStats, PrivacyAmplificationAnalyzer, SubsamplingEvent,
};

// Re-export Renyi differential privacy accountant types
pub use renyi_accountant::{DpConversion, RdpSpend, RenyiAccountant};

// Re-export the unified accounting interface.
pub use accountant::{
    build_accountant, AccountingSegment, MomentsPrivacyAccountant, PrivacyAccountant,
    PrivacyLedger, RenyiPrivacyAccountant,
};

/// The moments accountant of Abadi et al. (2016).
///
/// There is exactly one implementation of this type in the crate; an earlier
/// revision shipped two divergent copies that disagreed by a factor of 436.
pub use moment_accountant::MomentsAccountant;

/// Differential privacy configuration
#[derive(Debug, Clone)]
pub struct DifferentialPrivacyConfig {
    /// Target privacy parameter epsilon (the budget that is enforced)
    pub target_epsilon: f64,

    /// Delta at which epsilon is reported (typically << 1/n). This is a
    /// reporting parameter, not an additively consumed budget.
    pub target_delta: f64,

    /// Noise multiplier for gradient perturbation (sigma, relative to the
    /// clipping norm)
    pub noise_multiplier: f64,

    /// L2 norm clipping threshold applied to each *per-example* gradient
    pub l2_norm_clip: f64,

    /// Expected batch size (used for the sampling probability)
    pub batch_size: usize,

    /// Dataset size for privacy accounting
    pub dataset_size: usize,

    /// Maximum number of training steps (enforced)
    pub max_steps: usize,

    /// Noise mechanism to use
    pub noise_mechanism: NoiseMechanism,

    /// Enable secure aggregation (for federated learning). When set, the
    /// optimizer refuses to run unless a secure aggregation backend has been
    /// wired in, rather than silently training without it.
    pub secure_aggregation: bool,

    /// Enable adaptive clipping (Andrew et al. 2021, differentially private
    /// quantile estimation)
    pub adaptive_clipping: bool,

    /// Initial clipping threshold for adaptive clipping
    pub adaptive_clip_init: f64,

    /// Learning rate (geometric update rate) for adaptive clipping
    pub adaptive_clip_lr: f64,

    /// Target fraction of per-example gradients that should fall *below* the
    /// clipping threshold (Andrew et al. 2021 use 0.5)
    pub adaptive_clip_target_quantile: f64,

    /// Noise multiplier applied to the privatized above-threshold count used
    /// by adaptive clipping. The count has sensitivity 1, so this is the
    /// standard deviation of the Gaussian added to it.
    pub adaptive_clip_noise_multiplier: f64,

    /// Privacy accounting method
    pub accounting_method: AccountingMethod,

    /// Explicit acknowledgement that the caller understands the semantics of
    /// [`DifferentiallyPrivateOptimizer::dp_step`], which clips an already
    /// aggregated gradient and therefore does **not** provide per-example
    /// differential privacy. Left `false`, that entry point returns an error.
    pub acknowledge_aggregate_clipping: bool,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            target_epsilon: 1.0,
            target_delta: 1e-5,
            noise_multiplier: 1.1,
            l2_norm_clip: 1.0,
            batch_size: 256,
            dataset_size: 50000,
            max_steps: 1000,
            noise_mechanism: NoiseMechanism::Gaussian,
            secure_aggregation: false,
            adaptive_clipping: false,
            adaptive_clip_init: 1.0,
            adaptive_clip_lr: 0.2,
            adaptive_clip_target_quantile: 0.5,
            adaptive_clip_noise_multiplier: 1.0,
            accounting_method: AccountingMethod::RenyiDP,
            acknowledge_aggregate_clipping: false,
        }
    }
}

impl DifferentialPrivacyConfig {
    /// Validate the configuration.
    ///
    /// Every parameter that can silently void a privacy guarantee is checked
    /// here: a zero noise multiplier, a non-positive clipping norm, a delta
    /// outside `(0, 1)`, a batch larger than the dataset, and so on.
    pub fn validate(&self) -> Result<()> {
        if !self.target_epsilon.is_finite() || self.target_epsilon <= 0.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "target_epsilon must be a positive finite number, got {}",
                self.target_epsilon
            )));
        }
        if !self.target_delta.is_finite() || self.target_delta <= 0.0 || self.target_delta >= 1.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "target_delta must be in (0, 1), got {}",
                self.target_delta
            )));
        }
        if !self.noise_multiplier.is_finite() || self.noise_multiplier <= 0.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "noise_multiplier must be a positive finite number, got {}",
                self.noise_multiplier
            )));
        }
        if !self.l2_norm_clip.is_finite() || self.l2_norm_clip <= 0.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "l2_norm_clip must be a positive finite number, got {}",
                self.l2_norm_clip
            )));
        }
        if self.batch_size == 0 || self.dataset_size == 0 {
            return Err(OptimError::InvalidPrivacyConfig(
                "batch_size and dataset_size must be positive".to_string(),
            ));
        }
        if self.batch_size > self.dataset_size {
            return Err(OptimError::InvalidPrivacyConfig(
                "batch_size cannot exceed dataset_size".to_string(),
            ));
        }
        if self.max_steps == 0 {
            return Err(OptimError::InvalidPrivacyConfig(
                "max_steps must be positive".to_string(),
            ));
        }
        if self.adaptive_clipping {
            if !self.adaptive_clip_init.is_finite() || self.adaptive_clip_init <= 0.0 {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "adaptive_clip_init must be a positive finite number, got {}",
                    self.adaptive_clip_init
                )));
            }
            if !self.adaptive_clip_lr.is_finite() || self.adaptive_clip_lr <= 0.0 {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "adaptive_clip_lr must be a positive finite number, got {}",
                    self.adaptive_clip_lr
                )));
            }
            if !(0.0..=1.0).contains(&self.adaptive_clip_target_quantile) {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "adaptive_clip_target_quantile must be in [0, 1], got {}",
                    self.adaptive_clip_target_quantile
                )));
            }
            if !self.adaptive_clip_noise_multiplier.is_finite()
                || self.adaptive_clip_noise_multiplier <= 0.0
            {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "adaptive_clip_noise_multiplier must be positive and finite, got {}",
                    self.adaptive_clip_noise_multiplier
                )));
            }
        }
        Ok(())
    }

    /// Per-record sampling probability `q = batch_size / dataset_size`.
    pub fn sampling_probability(&self) -> f64 {
        if self.dataset_size == 0 {
            0.0
        } else {
            (self.batch_size as f64 / self.dataset_size as f64).min(1.0)
        }
    }
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseMechanism {
    /// Gaussian noise mechanism ((epsilon, delta)-DP)
    Gaussian,
    /// Laplace noise mechanism (pure epsilon-DP, delta = 0)
    Laplace,
    /// Tree aggregation with Gaussian noise (not implemented for the
    /// optimizer path; selecting it returns an error rather than silently
    /// using a different mechanism)
    TreeAggregation,
    /// Improved composition with amplification (not implemented for the
    /// optimizer path; selecting it returns an error)
    ImprovedComposition,
}

/// Privacy budget tracking information
#[derive(Debug, Clone)]
pub struct PrivacyBudget {
    /// Current epsilon consumed
    pub epsilon_consumed: f64,

    /// Delta *consumed*. Always 0.0: delta is a reporting parameter, not an
    /// additive spend. Retained for API compatibility.
    pub delta_consumed: f64,

    /// Remaining epsilon budget
    pub epsilon_remaining: f64,

    /// The delta at which `epsilon_consumed` is reported. Since delta is not
    /// consumed, the full reporting delta always "remains".
    pub delta_remaining: f64,

    /// Number of steps taken
    pub steps_taken: usize,

    /// Privacy accounting method used
    pub accounting_method: AccountingMethod,

    /// Estimated steps until budget exhaustion
    pub estimated_steps_remaining: usize,
}

impl Default for PrivacyBudget {
    fn default() -> Self {
        Self {
            epsilon_consumed: 0.0,
            delta_consumed: 0.0,
            epsilon_remaining: 1.0,
            delta_remaining: 1e-5,
            steps_taken: 0,
            accounting_method: AccountingMethod::RenyiDP,
            estimated_steps_remaining: 1000,
        }
    }
}

/// Privacy accounting methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountingMethod {
    /// Moments accountant (Abadi et al. 2016)
    MomentsAccountant,
    /// Renyi differential privacy (default; tightest available here)
    RenyiDP,
    /// Advanced composition (not implemented for the subsampled Gaussian
    /// mechanism; selecting it returns an error)
    AdvancedComposition,
    /// Zero-concentrated differential privacy (not implemented; selecting it
    /// returns an error)
    ZCDP,
}

/// Differentially private optimizer wrapper
pub struct DifferentiallyPrivateOptimizer<O, A, D>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
    O: Optimizer<A, D>,
{
    /// Base optimizer
    base_optimizer: O,

    /// Privacy configuration
    config: DifferentialPrivacyConfig,

    /// Privacy accountant selected from `config.accounting_method`
    accountant: Box<dyn PrivacyAccountant>,

    /// Pure epsilon-DP ledger, used by the Laplace mechanism (which provides
    /// delta = 0 and therefore does not compose through the RDP accountant)
    pure_epsilon_spent: f64,

    /// Random number generator for noise (seeded from OS entropy)
    rng: scirs2_core::random::CoreRandom,

    /// Adaptive clipping state
    adaptive_clip_state: Option<AdaptiveClippingState>,

    /// Gradient history for analysis
    gradient_history: VecDeque<GradientNorms>,

    /// Privacy audit trail
    audit_trail: Vec<PrivacyEvent>,

    /// Current step count
    step_count: usize,

    /// Phantom data for unused type parameters
    _phantom: std::marker::PhantomData<(A, D)>,
}

/// Adaptive clipping state (Andrew et al. 2021).
#[derive(Debug, Clone)]
struct AdaptiveClippingState {
    /// Current clipping threshold C
    current_threshold: f64,

    /// Target fraction of gradients below the threshold
    target_quantile: f64,

    /// Geometric update rate
    learning_rate: f64,

    /// Most recent privatized below-threshold fraction
    last_fraction_estimate: f64,

    /// Number of privatized quantile updates performed
    updates: usize,
}

/// Gradient norm statistics
#[derive(Debug, Clone)]
struct GradientNorms {
    #[allow(dead_code)]
    step: usize,
    pre_clip_norm: f64,
    #[allow(dead_code)]
    post_clip_norm: f64,
    clipping_ratio: f64,
    clipped: bool,
}

/// Privacy event for audit trail
#[derive(Debug, Clone)]
pub struct PrivacyEvent {
    /// Step at which the event occurred.
    pub step: usize,
    /// Kind of release.
    pub event_type: PrivacyEventType,
    /// Cumulative epsilon after the event.
    pub epsilon_spent: f64,
    /// Delta the epsilon is reported at.
    pub reporting_delta: f64,
    /// Standard deviation of the noise added by this release.
    pub noise_scale: f64,
}

/// Kind of privacy-consuming release recorded in the audit trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyEventType {
    /// A privatized gradient was released.
    GradientRelease,
    /// A model update was released.
    ModelUpdate,
    /// A parameter query was answered.
    ParameterQuery,
    /// A privatized clipping-threshold update was released.
    AdaptiveClipUpdate,
}

impl<O, A, D> DifferentiallyPrivateOptimizer<O, A, D>
where
    A: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Send
        + Sync
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug,
    D: Dimension,
    O: Optimizer<A, D>,
{
    /// Create a new differentially private optimizer.
    ///
    /// The configuration is validated up front, and the accountant named by
    /// `config.accounting_method` is constructed (an unimplemented method is
    /// an error, never a silent substitution).
    pub fn new(baseoptimizer: O, config: DifferentialPrivacyConfig) -> Result<Self> {
        config.validate()?;

        if config.secure_aggregation {
            return Err(OptimError::InvalidPrivacyConfig(
                "secure_aggregation is not wired into DifferentiallyPrivateOptimizer; use \
                 privacy::secure_aggregation::SecureAggregator explicitly instead of enabling \
                 this flag, which would otherwise train with plaintext aggregation"
                    .to_string(),
            ));
        }

        let accountant = build_accountant(
            config.accounting_method,
            config.noise_multiplier,
            config.target_delta,
            config.batch_size,
            config.dataset_size,
        )?;

        let rng = thread_rng();

        let adaptive_clip_state = if config.adaptive_clipping {
            Some(AdaptiveClippingState {
                current_threshold: config.adaptive_clip_init,
                target_quantile: config.adaptive_clip_target_quantile,
                learning_rate: config.adaptive_clip_lr,
                last_fraction_estimate: config.adaptive_clip_target_quantile,
                updates: 0,
            })
        } else {
            None
        };

        Ok(Self {
            base_optimizer: baseoptimizer,
            config,
            accountant,
            pure_epsilon_spent: 0.0,
            rng,
            adaptive_clip_state,
            gradient_history: VecDeque::with_capacity(1000),
            audit_trail: Vec::new(),
            step_count: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Perform a differentially private step from **per-example** gradients.
    ///
    /// This is the entry point that delivers the standard DP-SGD guarantee:
    ///
    /// 1. every per-example gradient is clipped to the current L2 threshold
    ///    `C`, bounding one example's influence on the sum by `C`;
    /// 2. the clipped gradients are summed;
    /// 3. `N(0, sigma^2 C^2)` noise is added to the **sum** (adding it after
    ///    the division would silently reduce the effective sigma by the batch
    ///    size);
    /// 4. the noisy sum is divided by the batch size.
    ///
    /// Adjacency: add/remove-one-example, for which the L2 sensitivity of the
    /// clipped sum is exactly `C`.
    ///
    /// The privacy budget is enforced *before* the noisy gradient is
    /// released: if composing this step would push epsilon past
    /// `target_epsilon`, the call returns
    /// [`OptimError::PrivacyBudgetExhausted`] and nothing is released.
    pub fn dp_step_per_example(
        &mut self,
        params: &Array<A, D>,
        per_example_gradients: &[Array<A, D>],
    ) -> Result<Array<A, D>> {
        if per_example_gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "dp_step_per_example requires at least one per-example gradient".to_string(),
            ));
        }
        for gradient in per_example_gradients {
            if gradient.raw_dim() != params.raw_dim() {
                return Err(OptimError::DimensionMismatch(format!(
                    "per-example gradient shape {:?} does not match parameter shape {:?}",
                    gradient.shape(),
                    params.shape()
                )));
            }
        }

        let batch_size = per_example_gradients.len();
        self.begin_step(batch_size)?;

        let clip_threshold = self.get_clipping_threshold();
        let mut summed: Array<A, D> = Array::zeros(params.raw_dim());
        let mut pre_clip_norm_sum = 0.0;
        let mut clipped_count = 0usize;
        let mut below_threshold = 0usize;

        for gradient in per_example_gradients {
            let norm = self.compute_l2_norm(gradient);
            if !norm.is_finite() {
                return Err(OptimError::InvalidConfig(
                    "per-example gradient contains a non-finite value; clipping cannot bound \
                     its sensitivity"
                        .to_string(),
                ));
            }
            pre_clip_norm_sum += norm;

            let scale = if norm > clip_threshold && norm > 0.0 {
                clipped_count += 1;
                clip_threshold / norm
            } else {
                below_threshold += 1;
                1.0
            };

            let scale_a = A::from(scale).ok_or_else(|| {
                OptimError::InvalidConfig("failed to convert clipping scale".to_string())
            })?;
            Zip::from(&mut summed)
                .and(gradient)
                .for_each(|acc, &g| *acc += g * scale_a);
        }

        // Noise the *sum*: sensitivity of the clipped sum is exactly C.
        let noise_scale = self.config.noise_multiplier * clip_threshold;
        self.add_mechanism_noise(&mut summed, noise_scale, clip_threshold)?;

        let batch_a = A::from(batch_size as f64)
            .ok_or_else(|| OptimError::InvalidConfig("failed to convert batch size".to_string()))?;
        summed.mapv_inplace(|x| x / batch_a);

        self.commit_step(batch_size, noise_scale, PrivacyEventType::GradientRelease)?;

        let mean_pre_clip = pre_clip_norm_sum / batch_size as f64;
        self.record_gradient_stats(mean_pre_clip, clip_threshold, clipped_count, batch_size);

        if self.config.adaptive_clipping {
            self.update_adaptive_clipping(below_threshold, batch_size)?;
        }

        self.base_optimizer.step(params, &summed)
    }

    /// Perform a differentially private step from a **pre-clipped sum** of
    /// per-example gradients.
    ///
    /// The caller asserts that `summed_clipped_gradients` is the sum of
    /// `batch_size` per-example gradients, each already clipped to
    /// [`DifferentialPrivacyConfig::l2_norm_clip`]. Noise `N(0, sigma^2 C^2)`
    /// is added to the sum, which is then divided by `batch_size`.
    ///
    /// Use this when per-example gradients are produced by an external
    /// framework and materialising them all is impractical.
    pub fn dp_step_presummed(
        &mut self,
        params: &Array<A, D>,
        summed_clipped_gradients: &Array<A, D>,
        batch_size: usize,
    ) -> Result<Array<A, D>> {
        if batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "batch_size must be positive".to_string(),
            ));
        }
        if summed_clipped_gradients.raw_dim() != params.raw_dim() {
            return Err(OptimError::DimensionMismatch(format!(
                "gradient shape {:?} does not match parameter shape {:?}",
                summed_clipped_gradients.shape(),
                params.shape()
            )));
        }

        self.begin_step(batch_size)?;

        let clip_threshold = self.get_clipping_threshold();
        let mut summed = summed_clipped_gradients.clone();
        let noise_scale = self.config.noise_multiplier * clip_threshold;
        self.add_mechanism_noise(&mut summed, noise_scale, clip_threshold)?;

        let batch_a = A::from(batch_size as f64)
            .ok_or_else(|| OptimError::InvalidConfig("failed to convert batch size".to_string()))?;
        summed.mapv_inplace(|x| x / batch_a);

        self.commit_step(batch_size, noise_scale, PrivacyEventType::GradientRelease)?;

        let norm = self.compute_l2_norm(summed_clipped_gradients) / batch_size as f64;
        self.record_gradient_stats(norm, clip_threshold, 0, batch_size);

        self.base_optimizer.step(params, &summed)
    }

    /// Perform a step that clips the **already aggregated** gradient.
    ///
    /// # This does not provide per-example differential privacy
    ///
    /// Clipping a batch-mean (or batch-sum) gradient bounds the influence of
    /// the *whole batch*, not of any single example, so the standard DP-SGD
    /// analysis -- and the epsilon this optimizer reports -- does not apply
    /// under example-level adjacency. One outlier example can still move the
    /// aggregate arbitrarily far inside the clipping ball.
    ///
    /// The entry point is retained for batch-level adjacency (neighbouring
    /// datasets differing in an entire batch) and for reproducing legacy
    /// behaviour. Because subsampling amplification does not apply under that
    /// adjacency, the step is accounted with sampling probability `q = 1`,
    /// which is strictly more conservative than the per-example path.
    ///
    /// It returns an error unless
    /// [`DifferentialPrivacyConfig::acknowledge_aggregate_clipping`] is set,
    /// so nobody gets this behaviour by accident. Prefer
    /// [`Self::dp_step_per_example`].
    pub fn dp_step(
        &mut self,
        params: &Array<A, D>,
        gradients: &mut Array<A, D>,
    ) -> Result<Array<A, D>> {
        if !self.config.acknowledge_aggregate_clipping {
            return Err(OptimError::InvalidPrivacyConfig(
                "dp_step clips an already-aggregated gradient and therefore does NOT provide \
                 per-example differential privacy. Use dp_step_per_example (or \
                 dp_step_presummed), or set acknowledge_aggregate_clipping = true to accept \
                 batch-level adjacency semantics"
                    .to_string(),
            ));
        }

        // Batch-level adjacency: no subsampling amplification.
        self.begin_step(self.config.dataset_size)?;

        let pre_clip_norm = self.compute_l2_norm(gradients);
        if !pre_clip_norm.is_finite() {
            return Err(OptimError::InvalidConfig(
                "gradient contains a non-finite value; clipping cannot bound its sensitivity"
                    .to_string(),
            ));
        }

        let clip_threshold = self.get_clipping_threshold();
        let (clipping_ratio, clipped) = if pre_clip_norm > clip_threshold && pre_clip_norm > 0.0 {
            let scale = clip_threshold / pre_clip_norm;
            let scale_a = A::from(scale).ok_or_else(|| {
                OptimError::InvalidConfig("failed to convert clipping scale".to_string())
            })?;
            gradients.mapv_inplace(|g| g * scale_a);
            (scale, true)
        } else {
            (1.0, false)
        };

        let noise_scale = self.config.noise_multiplier * clip_threshold;
        self.add_mechanism_noise(gradients, noise_scale, clip_threshold)?;

        self.commit_step(
            self.config.dataset_size,
            noise_scale,
            PrivacyEventType::GradientRelease,
        )?;

        let post_clip_norm = self.compute_l2_norm(gradients);
        self.gradient_history.push_back(GradientNorms {
            step: self.step_count,
            pre_clip_norm,
            post_clip_norm,
            clipping_ratio,
            clipped,
        });
        if self.gradient_history.len() > 1000 {
            self.gradient_history.pop_front();
        }

        self.base_optimizer.step(params, gradients)
    }

    /// Enforce the step budget *before* any output is released.
    fn begin_step(&mut self, batch_size: usize) -> Result<()> {
        if self.step_count >= self.config.max_steps {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.consumed_epsilon()?,
                target_epsilon: self.config.target_epsilon,
            });
        }

        let projected = self.projected_epsilon(batch_size)?;
        if projected > self.config.target_epsilon {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.consumed_epsilon()?,
                target_epsilon: self.config.target_epsilon,
            });
        }

        Ok(())
    }

    /// Record a released step in the accountant and the audit trail.
    fn commit_step(
        &mut self,
        batch_size: usize,
        noise_scale: f64,
        event_type: PrivacyEventType,
    ) -> Result<()> {
        self.step_count += 1;

        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let q = self.sampling_probability(batch_size);
                self.accountant
                    .compose_subsampled_gaussian(self.config.noise_multiplier, q, 1)?;
            }
            NoiseMechanism::Laplace => {
                self.pure_epsilon_spent += self.laplace_epsilon_per_step();
            }
            other => {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "noise mechanism {other:?} is not implemented for this optimizer"
                )));
            }
        }

        let epsilon_spent = self.consumed_epsilon()?;
        self.audit_trail.push(PrivacyEvent {
            step: self.step_count,
            event_type,
            epsilon_spent,
            reporting_delta: self.reporting_delta(),
            noise_scale,
        });

        Ok(())
    }

    /// Record clipping statistics for the step.
    fn record_gradient_stats(
        &mut self,
        mean_pre_clip_norm: f64,
        clip_threshold: f64,
        clipped_count: usize,
        batch_size: usize,
    ) {
        let clipping_ratio = if mean_pre_clip_norm > clip_threshold && mean_pre_clip_norm > 0.0 {
            clip_threshold / mean_pre_clip_norm
        } else {
            1.0
        };

        self.gradient_history.push_back(GradientNorms {
            step: self.step_count,
            pre_clip_norm: mean_pre_clip_norm,
            post_clip_norm: mean_pre_clip_norm.min(clip_threshold),
            clipping_ratio,
            clipped: clipped_count * 2 > batch_size,
        });

        if self.gradient_history.len() > 1000 {
            self.gradient_history.pop_front();
        }
    }

    /// Differentially private clipping-threshold update (Andrew et al. 2021,
    /// "Differentially Private Learning with Adaptive Clipping").
    ///
    /// The fraction of per-example gradients whose norm falls below the
    /// current threshold is a counting query with sensitivity 1. It is
    /// released with Gaussian noise -- charged to the privacy budget like any
    /// other release -- and the threshold is updated geometrically:
    ///
    /// ```text
    /// C <- C * exp(-eta * (fraction_below - target_quantile))
    /// ```
    ///
    /// Using the *raw* norms (as an earlier revision did) leaks the gradient
    /// distribution and is not differentially private at any epsilon.
    fn update_adaptive_clipping(
        &mut self,
        below_threshold: usize,
        batch_size: usize,
    ) -> Result<()> {
        if batch_size == 0 {
            return Ok(());
        }

        let sigma_count = self.config.adaptive_clip_noise_multiplier;
        let noise = sigma_count * standard_normal(&mut self.rng);
        let noisy_below = below_threshold as f64 + noise;
        let fraction = (noisy_below / batch_size as f64).clamp(0.0, 1.0);

        let (target, learning_rate) = match self.adaptive_clip_state {
            Some(ref state) => (state.target_quantile, state.learning_rate),
            None => return Ok(()),
        };

        let factor = (-learning_rate * (fraction - target)).exp();
        let updated = if let Some(ref mut state) = self.adaptive_clip_state {
            state.last_fraction_estimate = fraction;
            state.updates += 1;
            state.current_threshold = (state.current_threshold * factor).clamp(1e-6, 1e6);
            state.current_threshold
        } else {
            return Ok(());
        };

        // The privatized count is an extra Gaussian release and must be paid
        // for. Sensitivity of the count is 1, so `sigma_count` is directly
        // the noise multiplier of that mechanism.
        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let q = self.sampling_probability(batch_size);
                self.accountant
                    .compose_subsampled_gaussian(sigma_count, q, 1)?;
            }
            NoiseMechanism::Laplace => {
                // Charged with the same per-step pure-epsilon allowance.
                self.pure_epsilon_spent += self.laplace_epsilon_per_step();
            }
            other => {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "noise mechanism {other:?} is not implemented for this optimizer"
                )));
            }
        }

        let epsilon_spent = self.consumed_epsilon()?;
        self.audit_trail.push(PrivacyEvent {
            step: self.step_count,
            event_type: PrivacyEventType::AdaptiveClipUpdate,
            epsilon_spent,
            reporting_delta: self.reporting_delta(),
            noise_scale: sigma_count,
        });

        debug_assert!(updated > 0.0);
        Ok(())
    }

    /// Sampling probability for a batch of the given size.
    fn sampling_probability(&self, batch_size: usize) -> f64 {
        if self.config.dataset_size == 0 {
            0.0
        } else {
            (batch_size as f64 / self.config.dataset_size as f64).min(1.0)
        }
    }

    /// Per-step epsilon allowance of the pure epsilon-DP (Laplace) path.
    ///
    /// Pure DP composes basically: the target budget is divided evenly over
    /// the configured maximum number of steps.
    fn laplace_epsilon_per_step(&self) -> f64 {
        self.config.target_epsilon / self.config.max_steps.max(1) as f64
    }

    /// Delta at which the reported epsilon holds (0 for pure epsilon-DP).
    fn reporting_delta(&self) -> f64 {
        match self.config.noise_mechanism {
            NoiseMechanism::Laplace => 0.0,
            _ => self.config.target_delta,
        }
    }

    /// Epsilon consumed so far.
    ///
    /// Errors from the accountant are propagated rather than swallowed: an
    /// accounting failure must never be reported as "zero spent", which would
    /// let training continue with no budget enforcement at all.
    pub fn consumed_epsilon(&self) -> Result<f64> {
        match self.config.noise_mechanism {
            NoiseMechanism::Laplace => Ok(self.pure_epsilon_spent),
            _ => {
                let (epsilon, _) = self.accountant.privacy_spent(self.config.target_delta)?;
                Ok(epsilon)
            }
        }
    }

    /// Epsilon that composing one more step with the given batch size would
    /// bring the total to.
    fn projected_epsilon(&self, batch_size: usize) -> Result<f64> {
        match self.config.noise_mechanism {
            NoiseMechanism::Laplace => {
                Ok(self.pure_epsilon_spent + self.laplace_epsilon_per_step())
            }
            NoiseMechanism::Gaussian => {
                let q = self.sampling_probability(batch_size);
                let (epsilon, _) = self.accountant.projected_privacy_spent(
                    self.config.noise_multiplier,
                    q,
                    1,
                    self.config.target_delta,
                )?;
                Ok(epsilon)
            }
            other => Err(OptimError::InvalidPrivacyConfig(format!(
                "noise mechanism {other:?} is not implemented for this optimizer"
            ))),
        }
    }

    /// Whether at least one more step fits inside the epsilon budget.
    pub fn has_privacy_budget(&self) -> Result<bool> {
        if self.step_count >= self.config.max_steps {
            return Ok(false);
        }
        let projected = self.projected_epsilon(self.config.batch_size)?;
        Ok(projected <= self.config.target_epsilon)
    }

    /// Current privacy budget status.
    ///
    /// Returns an error if the accountant cannot produce a number -- failing
    /// closed instead of reporting a fabricated zero spend.
    pub fn get_privacy_budget(&self) -> Result<PrivacyBudget> {
        let epsilon_consumed = self.consumed_epsilon()?;
        let epsilon_remaining = (self.config.target_epsilon - epsilon_consumed).max(0.0);

        let epsilon_per_step = if self.step_count > 0 && epsilon_consumed.is_finite() {
            epsilon_consumed / self.step_count as f64
        } else {
            0.0
        };

        let estimated_steps_remaining = if epsilon_per_step > 0.0 {
            let by_budget = (epsilon_remaining / epsilon_per_step) as usize;
            by_budget.min(self.config.max_steps.saturating_sub(self.step_count))
        } else {
            self.config.max_steps.saturating_sub(self.step_count)
        };

        Ok(PrivacyBudget {
            epsilon_consumed,
            delta_consumed: 0.0,
            epsilon_remaining,
            delta_remaining: self.reporting_delta(),
            steps_taken: self.step_count,
            accounting_method: self.config.accounting_method,
            estimated_steps_remaining,
        })
    }

    /// Immutable view of the accountant's segment ledger.
    pub fn accounting_segments(&self) -> &[AccountingSegment] {
        self.accountant.segments()
    }

    /// The configuration in force.
    pub fn config(&self) -> &DifferentialPrivacyConfig {
        &self.config
    }

    fn compute_l2_norm<S, DIM>(&self, array: &ArrayBase<S, DIM>) -> f64
    where
        S: Data<Elem = A>,
        DIM: Dimension,
    {
        array
            .iter()
            .map(|&x| {
                let val = x.to_f64().unwrap_or(f64::NAN);
                val * val
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Current clipping threshold (adaptive if enabled).
    pub fn get_clipping_threshold(&self) -> f64 {
        if let Some(ref state) = self.adaptive_clip_state {
            state.current_threshold
        } else {
            self.config.l2_norm_clip
        }
    }

    /// Add mechanism noise to a gradient container.
    ///
    /// * Gaussian: `N(0, (sigma * C)^2)` per coordinate.
    /// * Laplace: `Lap(b)` with `b = sensitivity / epsilon_step`, where the
    ///   L1 sensitivity is bounded by `sqrt(d) * C` for an L2-clipped
    ///   gradient of dimension `d`. This is what actually delivers pure
    ///   epsilon-DP; sampling a Gaussian and calling it Laplace does not.
    /// * Every other variant is an error rather than a silent fallback.
    fn add_mechanism_noise<S, DIM>(
        &mut self,
        gradients: &mut ArrayBase<S, DIM>,
        gaussian_noise_scale: f64,
        clip_threshold: f64,
    ) -> Result<()>
    where
        S: DataMut<Elem = A>,
        DIM: Dimension,
    {
        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let sigma = gaussian_noise_scale;
                let mut failed = false;
                gradients.mapv_inplace(|g| {
                    let sample = standard_normal(&mut self.rng) * sigma;
                    match A::from(sample) {
                        Some(noise) => g + noise,
                        None => {
                            failed = true;
                            g
                        }
                    }
                });
                if failed {
                    return Err(OptimError::InvalidConfig(
                        "failed to convert Gaussian noise sample into the gradient element type"
                            .to_string(),
                    ));
                }
            }
            NoiseMechanism::Laplace => {
                let dimension = gradients.len().max(1) as f64;
                let l1_sensitivity = clip_threshold * dimension.sqrt();
                let epsilon_step = self.laplace_epsilon_per_step();
                if epsilon_step <= 0.0 {
                    return Err(OptimError::InvalidPrivacyConfig(
                        "Laplace mechanism requires a positive per-step epsilon".to_string(),
                    ));
                }
                let scale = l1_sensitivity / epsilon_step;
                let mut failed = false;
                gradients.mapv_inplace(|g| {
                    let sample = standard_laplace(&mut self.rng) * scale;
                    match A::from(sample) {
                        Some(noise) => g + noise,
                        None => {
                            failed = true;
                            g
                        }
                    }
                });
                if failed {
                    return Err(OptimError::InvalidConfig(
                        "failed to convert Laplace noise sample into the gradient element type"
                            .to_string(),
                    ));
                }
            }
            other => {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "noise mechanism {other:?} is not implemented for this optimizer; use \
                     Gaussian or Laplace"
                )));
            }
        }

        Ok(())
    }

    /// Gradient clipping statistics.
    pub fn get_clipping_stats(&self) -> ClippingStats {
        if self.gradient_history.is_empty() {
            return ClippingStats {
                current_threshold: self.get_clipping_threshold(),
                ..ClippingStats::default()
            };
        }

        let total_steps = self.gradient_history.len();
        let clipped_steps = self
            .gradient_history
            .iter()
            .filter(|stats| stats.clipped)
            .count();

        let avg_clipping_ratio: f64 = self
            .gradient_history
            .iter()
            .map(|stats| stats.clipping_ratio)
            .sum::<f64>()
            / total_steps as f64;

        let avg_pre_clip_norm: f64 = self
            .gradient_history
            .iter()
            .map(|stats| stats.pre_clip_norm)
            .sum::<f64>()
            / total_steps as f64;

        ClippingStats {
            total_steps,
            clipped_steps,
            clipping_frequency: clipped_steps as f64 / total_steps as f64,
            avg_clipping_ratio,
            avg_pre_clip_norm,
            current_threshold: self.get_clipping_threshold(),
        }
    }

    /// Privacy audit trail.
    pub fn get_audit_trail(&self) -> &[PrivacyEvent] {
        &self.audit_trail
    }

    /// Validate privacy guarantees against the configured budget.
    pub fn validate_privacy(&self) -> Result<PrivacyValidation> {
        let budget = self.get_privacy_budget()?;
        let clipping_stats = self.get_clipping_stats();

        let mut warnings = Vec::new();
        let mut is_valid = true;

        if !budget.epsilon_consumed.is_finite() {
            warnings.push(
                "Privacy accounting saturated: the configured noise provides no usable guarantee"
                    .to_string(),
            );
            is_valid = false;
        } else if budget.epsilon_consumed > self.config.target_epsilon {
            warnings.push("Epsilon budget exceeded".to_string());
            is_valid = false;
        }

        if clipping_stats.total_steps > 0 {
            if clipping_stats.clipping_frequency < 0.1 {
                warnings.push(
                    "Low clipping frequency may indicate sub-optimal privacy-utility tradeoff"
                        .to_string(),
                );
            }
            if clipping_stats.clipping_frequency > 0.9 {
                warnings.push("High clipping frequency may severely impact utility".to_string());
            }
        }

        let recommendations = self.generate_recommendations(&budget, &clipping_stats);

        Ok(PrivacyValidation {
            is_valid,
            budget,
            clipping_stats,
            warnings,
            recommendations,
        })
    }

    fn generate_recommendations(
        &self,
        budget: &PrivacyBudget,
        clipping: &ClippingStats,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if clipping.total_steps > 0 {
            if clipping.clipping_frequency > 0.8 {
                recommendations.push("Consider increasing the clipping threshold".to_string());
            }
            if clipping.clipping_frequency < 0.2 {
                recommendations.push("Consider decreasing the clipping threshold".to_string());
            }
        }

        if budget.epsilon_remaining < budget.epsilon_consumed * 0.1 {
            recommendations.push(
                "Privacy budget nearly exhausted - increase the noise multiplier or stop training"
                    .to_string(),
            );
        }

        recommendations
    }
}

/// Draw a standard normal sample via the Box-Muller transform.
///
/// `gen_range(0.0..1.0)` *includes* 0.0, and `ln(0) = -inf` would poison the
/// gradient with an infinite noise value. Mapping the draw into `(0, 1]`
/// removes that failure mode entirely.
pub(crate) fn standard_normal(rng: &mut scirs2_core::random::CoreRandom) -> f64 {
    let u1: f64 = 1.0 - rng.gen_range(0.0..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Draw a `Laplace(0, 1)` sample by inverse transform sampling.
///
/// `u` is mapped strictly inside `(0, 1)` so neither tail evaluates `ln(0)`.
pub(crate) fn standard_laplace(rng: &mut scirs2_core::random::CoreRandom) -> f64 {
    let raw: f64 = rng.gen_range(0.0..1.0);
    let u = raw.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    if u < 0.5 {
        (2.0 * u).ln()
    } else {
        -(2.0 * (1.0 - u)).ln()
    }
}

/// Gradient clipping statistics
#[derive(Debug, Clone)]
pub struct ClippingStats {
    /// Number of recorded steps.
    pub total_steps: usize,
    /// Steps in which clipping was active.
    pub clipped_steps: usize,
    /// Fraction of steps in which clipping was active.
    pub clipping_frequency: f64,
    /// Mean scaling factor applied by clipping.
    pub avg_clipping_ratio: f64,
    /// Mean pre-clipping gradient norm.
    pub avg_pre_clip_norm: f64,
    /// Clipping threshold currently in force.
    pub current_threshold: f64,
}

impl Default for ClippingStats {
    fn default() -> Self {
        Self {
            total_steps: 0,
            clipped_steps: 0,
            clipping_frequency: 0.0,
            avg_clipping_ratio: 1.0,
            avg_pre_clip_norm: 0.0,
            current_threshold: 1.0,
        }
    }
}

/// Privacy validation results
#[derive(Debug, Clone)]
pub struct PrivacyValidation {
    /// Whether the run is still inside its budget.
    pub is_valid: bool,
    /// Budget snapshot.
    pub budget: PrivacyBudget,
    /// Clipping statistics.
    pub clipping_stats: ClippingStats,
    /// Warnings raised by validation.
    pub warnings: Vec<String>,
    /// Suggested adjustments.
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;
    use scirs2_core::ndarray::{Array1, Ix1};

    fn test_config() -> DifferentialPrivacyConfig {
        DifferentialPrivacyConfig {
            target_epsilon: 10.0,
            target_delta: 1e-5,
            noise_multiplier: 1.1,
            l2_norm_clip: 1.0,
            batch_size: 64,
            dataset_size: 50_000,
            max_steps: 1000,
            ..Default::default()
        }
    }

    fn build_optimizer(
        config: DifferentialPrivacyConfig,
    ) -> DifferentiallyPrivateOptimizer<SGD<f64>, f64, Ix1> {
        match DifferentiallyPrivateOptimizer::new(SGD::new(0.01), config) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("optimizer construction failed: {err}"),
        }
    }

    #[test]
    fn test_dp_config_default() {
        let config = DifferentialPrivacyConfig::default();
        assert_eq!(config.target_epsilon, 1.0);
        assert_eq!(config.noise_multiplier, 1.1);
        assert!(matches!(config.noise_mechanism, NoiseMechanism::Gaussian));
        assert!(matches!(
            config.accounting_method,
            AccountingMethod::RenyiDP
        ));
        assert!(!config.acknowledge_aggregate_clipping);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_rejects_unusable_parameters() {
        let mut config = DifferentialPrivacyConfig {
            noise_multiplier: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        config = DifferentialPrivacyConfig {
            target_delta: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        config = DifferentialPrivacyConfig {
            batch_size: 100,
            dataset_size: 10,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dp_optimizer_creation() {
        let optimizer = DifferentiallyPrivateOptimizer::<_, f64, Ix1>::new(
            SGD::new(0.01),
            DifferentialPrivacyConfig::default(),
        );
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_unimplemented_accounting_method_is_rejected() {
        for method in [
            AccountingMethod::AdvancedComposition,
            AccountingMethod::ZCDP,
        ] {
            let config = DifferentialPrivacyConfig {
                accounting_method: method,
                ..Default::default()
            };
            let result =
                DifferentiallyPrivateOptimizer::<SGD<f64>, f64, Ix1>::new(SGD::new(0.01), config);
            assert!(result.is_err(), "{method:?} must not be silently accepted");
        }
    }

    #[test]
    fn test_privacy_budget_tracking_starts_empty() {
        let optimizer = build_optimizer(test_config());
        let budget = match optimizer.get_privacy_budget() {
            Ok(budget) => budget,
            Err(err) => panic!("budget query failed: {err}"),
        };

        assert_eq!(budget.epsilon_consumed, 0.0);
        assert_eq!(budget.epsilon_remaining, 10.0);
        assert_eq!(budget.steps_taken, 0);
        // Delta is a reporting parameter: nothing is consumed, and the full
        // reporting delta remains available.
        assert_eq!(budget.delta_consumed, 0.0);
        assert_eq!(budget.delta_remaining, 1e-5);
        assert!(match optimizer.has_privacy_budget() {
            Ok(available) => available,
            Err(err) => panic!("budget check failed: {err}"),
        });
    }

    #[test]
    fn test_per_example_step_succeeds_and_spends_epsilon_monotonically() {
        let mut optimizer = build_optimizer(test_config());
        let params = Array1::<f64>::zeros(4);
        let batch: Vec<Array1<f64>> = (0..8)
            .map(|i| Array1::from_elem(4, 0.1 * (i as f64 + 1.0)))
            .collect();

        let mut previous = 0.0;
        for step in 1..=25 {
            let updated = optimizer.dp_step_per_example(&params, &batch);
            assert!(updated.is_ok(), "step {step} failed: {updated:?}");

            let epsilon = match optimizer.consumed_epsilon() {
                Ok(value) => value,
                Err(err) => panic!("accounting failed at step {step}: {err}"),
            };
            assert!(
                epsilon > previous,
                "epsilon must strictly increase: {previous} -> {epsilon} at step {step}"
            );
            previous = epsilon;
        }

        assert_eq!(optimizer.accounting_segments().len(), 1);
    }

    #[test]
    fn test_per_example_clipping_bounds_the_summed_contribution() {
        // A single huge gradient must not be able to move the released mean
        // by more than roughly C / batch_size before noise.
        let config = DifferentialPrivacyConfig {
            noise_multiplier: 0.001,
            l2_norm_clip: 1.0,
            target_epsilon: 1e9,
            ..test_config()
        };
        let mut optimizer = build_optimizer(config);
        let params = Array1::<f64>::zeros(3);
        let batch = vec![
            Array1::from_vec(vec![1e6, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
        ];

        let updated = match optimizer.dp_step_per_example(&params, &batch) {
            Ok(updated) => updated,
            Err(err) => panic!("step failed: {err}"),
        };

        // SGD with lr = 0.01 applied to a mean gradient bounded by C / 4.
        let bound = 0.01 * (1.0 / 4.0) * 1.05;
        assert!(
            updated[0].abs() <= bound,
            "clipping failed to bound the update: {} > {bound}",
            updated[0].abs()
        );
    }

    #[test]
    fn test_budget_exhaustion_is_enforced_before_release() {
        let config = DifferentialPrivacyConfig {
            target_epsilon: 3.0,
            noise_multiplier: 1.0,
            batch_size: 64,
            dataset_size: 640,
            max_steps: 100_000,
            ..test_config()
        };
        let mut optimizer = build_optimizer(config);
        let params = Array1::<f64>::zeros(2);
        let batch: Vec<Array1<f64>> = (0..64).map(|_| Array1::from_elem(2, 0.5)).collect();

        let mut steps = 0usize;
        loop {
            match optimizer.dp_step_per_example(&params, &batch) {
                Ok(_) => {
                    steps += 1;
                    assert!(steps < 100_000, "budget was never enforced");
                }
                Err(OptimError::PrivacyBudgetExhausted {
                    consumed_epsilon,
                    target_epsilon,
                }) => {
                    assert!(steps > 0, "the very first step must be allowed");
                    assert!(consumed_epsilon <= target_epsilon);
                    break;
                }
                Err(other) => panic!("unexpected error: {other}"),
            }
        }

        // Once exhausted it stays exhausted, and the reported spend never
        // exceeded the target.
        assert!(match optimizer.has_privacy_budget() {
            Ok(available) => !available,
            Err(err) => panic!("budget check failed: {err}"),
        });
        let budget = match optimizer.get_privacy_budget() {
            Ok(budget) => budget,
            Err(err) => panic!("budget query failed: {err}"),
        };
        assert!(budget.epsilon_consumed <= 3.0 + 1e-12);
        assert_eq!(budget.steps_taken, steps);
    }

    #[test]
    fn test_max_steps_is_enforced() {
        let config = DifferentialPrivacyConfig {
            target_epsilon: 1e9,
            max_steps: 3,
            ..test_config()
        };
        let mut optimizer = build_optimizer(config);
        let params = Array1::<f64>::zeros(2);
        let batch = vec![Array1::from_elem(2, 0.1)];

        for _ in 0..3 {
            assert!(optimizer.dp_step_per_example(&params, &batch).is_ok());
        }
        match optimizer.dp_step_per_example(&params, &batch) {
            Err(OptimError::PrivacyBudgetExhausted { .. }) => {}
            other => panic!("max_steps must be enforced, got {other:?}"),
        }
    }

    #[test]
    fn test_aggregate_step_requires_explicit_acknowledgement() {
        let mut optimizer = build_optimizer(test_config());
        let params = Array1::<f64>::zeros(2);
        let mut gradients = Array1::from_elem(2, 0.5);
        match optimizer.dp_step(&params, &mut gradients) {
            Err(OptimError::InvalidPrivacyConfig(message)) => {
                assert!(message.contains("per-example"));
            }
            other => panic!("aggregate clipping must be opt-in, got {other:?}"),
        }

        let config = DifferentialPrivacyConfig {
            acknowledge_aggregate_clipping: true,
            ..test_config()
        };
        let mut acknowledged = build_optimizer(config);
        assert!(acknowledged.dp_step(&params, &mut gradients).is_ok());
    }

    #[test]
    fn test_unimplemented_noise_mechanisms_error() {
        for mechanism in [
            NoiseMechanism::TreeAggregation,
            NoiseMechanism::ImprovedComposition,
        ] {
            let config = DifferentialPrivacyConfig {
                noise_mechanism: mechanism,
                ..test_config()
            };
            let mut optimizer = build_optimizer(config);
            let params = Array1::<f64>::zeros(2);
            let batch = vec![Array1::from_elem(2, 0.1)];
            match optimizer.dp_step_per_example(&params, &batch) {
                Err(OptimError::InvalidPrivacyConfig(_)) => {}
                other => panic!("{mechanism:?} must not silently fall back: {other:?}"),
            }
        }
    }

    #[test]
    fn test_laplace_path_uses_pure_epsilon_ledger() {
        let config = DifferentialPrivacyConfig {
            noise_mechanism: NoiseMechanism::Laplace,
            target_epsilon: 1.0,
            max_steps: 10,
            ..test_config()
        };
        let mut optimizer = build_optimizer(config);
        let params = Array1::<f64>::zeros(2);
        let batch = vec![Array1::from_elem(2, 0.1)];

        for step in 1..=10 {
            assert!(
                optimizer.dp_step_per_example(&params, &batch).is_ok(),
                "step {step} should fit in the pure-epsilon budget"
            );
            let epsilon = match optimizer.consumed_epsilon() {
                Ok(value) => value,
                Err(err) => panic!("accounting failed: {err}"),
            };
            assert!((epsilon - 0.1 * step as f64).abs() < 1e-12);
        }

        // Delta is exactly zero for pure epsilon-DP.
        let budget = match optimizer.get_privacy_budget() {
            Ok(budget) => budget,
            Err(err) => panic!("budget query failed: {err}"),
        };
        assert_eq!(budget.delta_remaining, 0.0);

        match optimizer.dp_step_per_example(&params, &batch) {
            Err(OptimError::PrivacyBudgetExhausted { .. }) => {}
            other => panic!("pure epsilon budget must be enforced, got {other:?}"),
        }
    }

    #[test]
    fn test_noise_is_actually_random_across_instances() {
        // Two independently constructed optimizers must not produce identical
        // noise: a hardcoded seed would make DP noise reproducible and thus
        // worthless.
        let params = Array1::<f64>::zeros(64);
        let batch = vec![Array1::<f64>::zeros(64)];

        let mut first = build_optimizer(test_config());
        let mut second = build_optimizer(test_config());

        let a = match first.dp_step_per_example(&params, &batch) {
            Ok(value) => value,
            Err(err) => panic!("step failed: {err}"),
        };
        let b = match second.dp_step_per_example(&params, &batch) {
            Ok(value) => value,
            Err(err) => panic!("step failed: {err}"),
        };

        let identical = a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() < f64::EPSILON);
        assert!(!identical, "two optimizers produced identical noise");
    }

    #[test]
    fn test_adaptive_clipping_updates_threshold_and_charges_budget() {
        let config = DifferentialPrivacyConfig {
            adaptive_clipping: true,
            adaptive_clip_init: 1.0,
            adaptive_clip_lr: 0.5,
            adaptive_clip_target_quantile: 0.5,
            target_epsilon: 1e6,
            ..test_config()
        };
        let mut optimizer = build_optimizer(config);
        let params = Array1::<f64>::zeros(2);
        // Every gradient is far above the threshold, so the privatized
        // below-threshold fraction is ~0 -- well under the 0.5 target -- and
        // Andrew et al.'s geometric update must *raise* the threshold.
        let batch: Vec<Array1<f64>> = (0..16).map(|_| Array1::from_elem(2, 100.0)).collect();

        let before = optimizer.get_clipping_threshold();
        let epsilon_before = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };
        for _ in 0..5 {
            assert!(optimizer.dp_step_per_example(&params, &batch).is_ok());
        }
        let after = optimizer.get_clipping_threshold();
        let epsilon_after = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };

        assert!(after > before, "threshold should grow: {before} -> {after}");
        assert!(epsilon_after > epsilon_before);
        // Both the gradient release and the privatized quantile release are
        // charged, so more mechanism applications than steps are recorded.
        assert!(optimizer
            .get_audit_trail()
            .iter()
            .any(|event| event.event_type == PrivacyEventType::AdaptiveClipUpdate));
    }

    #[test]
    fn test_secure_aggregation_flag_is_not_silently_ignored() {
        let config = DifferentialPrivacyConfig {
            secure_aggregation: true,
            ..test_config()
        };
        let result =
            DifferentiallyPrivateOptimizer::<SGD<f64>, f64, Ix1>::new(SGD::new(0.01), config);
        assert!(result.is_err(), "an unimplemented flag must not be ignored");
    }

    #[test]
    fn test_dimension_mismatch_is_rejected() {
        let mut optimizer = build_optimizer(test_config());
        let params = Array1::<f64>::zeros(4);
        let batch = vec![Array1::<f64>::zeros(3)];
        assert!(optimizer.dp_step_per_example(&params, &batch).is_err());
        assert!(optimizer.dp_step_per_example(&params, &[]).is_err());
    }

    #[test]
    fn test_non_finite_gradients_are_rejected() {
        let mut optimizer = build_optimizer(test_config());
        let params = Array1::<f64>::zeros(2);
        let batch = vec![Array1::from_vec(vec![f64::NAN, 0.0])];
        assert!(optimizer.dp_step_per_example(&params, &batch).is_err());
    }

    #[test]
    fn test_presummed_path_matches_per_example_accounting() {
        let mut optimizer = build_optimizer(test_config());
        let params = Array1::<f64>::zeros(2);
        let summed = Array1::from_elem(2, 0.5);
        assert!(optimizer.dp_step_presummed(&params, &summed, 8).is_ok());
        assert_eq!(optimizer.accounting_segments().len(), 1);
        assert_eq!(optimizer.accounting_segments()[0].steps, 1);
        assert!(match optimizer.consumed_epsilon() {
            Ok(value) => value > 0.0,
            Err(err) => panic!("accounting failed: {err}"),
        });
        assert!(optimizer.dp_step_presummed(&params, &summed, 0).is_err());
    }

    #[test]
    fn test_standard_normal_never_returns_non_finite() {
        let mut rng = thread_rng();
        for _ in 0..10_000 {
            let sample = standard_normal(&mut rng);
            assert!(sample.is_finite(), "Box-Muller produced {sample}");
            let laplace = standard_laplace(&mut rng);
            assert!(
                laplace.is_finite(),
                "inverse-CDF Laplace produced {laplace}"
            );
        }
    }
}
