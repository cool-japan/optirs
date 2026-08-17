// Differentially Private Stochastic Gradient Descent (DP-SGD)
//
// This module implements DP-SGD with per-example gradient clipping, adaptive
// clipping via a differentially private quantile estimate, calibrated noise
// and privacy budget tracking.
//
// # What makes this private
//
// DP-SGD bounds the influence of a single training example by clipping *each
// per-example gradient* to an L2 norm `C`, summing the clipped gradients, and
// adding `N(0, sigma^2 C^2)` to the **sum** before dividing by the batch
// size. Clipping an already-aggregated batch gradient bounds nothing about a
// single example, which is why [`DPSGDOptimizer::dp_step`] requires an
// explicit acknowledgement and why
// [`DPSGDOptimizer::dp_step_per_example`] is the entry point to prefer.
//
// # Accounting
//
// Privacy spend is recorded in an append-only ledger of
// `(noise_multiplier, sampling_probability, steps)` segments held by the
// configured [`PrivacyAccountant`]. Changing the batch size or the noise
// multiplier mid-training opens a new segment; it never rewrites the cost of
// steps already taken.

// `!(x > 0)` is used deliberately over `x <= 0` in privacy-parameter validation:
// it rejects NaN (whose comparisons are all `false`), which `x <= 0` would let
// through. A NaN epsilon silently voiding the DP guarantee is exactly the failure
// this spelling prevents.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use scirs2_core::ndarray::{Array, ArrayBase, Data, DataMut, Dimension, Zip};
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

use super::accountant::{build_accountant, AccountingSegment, PrivacyAccountant};
use super::{
    standard_laplace, standard_normal, DifferentialPrivacyConfig, NoiseMechanism, PrivacyBudget,
};
use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// DP-SGD optimizer with privacy guarantees
pub struct DPSGDOptimizer<O, A, D>
where
    A: Float
        + Send
        + Sync
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug
        + Default
        + Clone
        + std::iter::Sum,
    D: scirs2_core::ndarray::Dimension,
    O: Optimizer<A, D>,
{
    /// Base optimizer (SGD, Adam, etc.)
    baseoptimizer: O,

    /// Privacy configuration
    config: DifferentialPrivacyConfig,

    /// Privacy accountant selected from `config.accounting_method`
    accountant: Box<dyn PrivacyAccountant>,

    /// Pure epsilon-DP ledger used by the Laplace mechanism
    pure_epsilon_spent: f64,

    /// Random number generator for noise (seeded from OS entropy)
    rng: scirs2_core::random::CoreRandom,

    /// Adaptive clipping state
    adaptive_clipping: Option<AdaptiveClippingState>,

    /// Privacy consumption history
    privacy_budget: PrivacyBudgetTracker,

    /// Gradient statistics
    gradient_stats: GradientStatistics<A>,

    /// Noise calibration history
    noise_calibrator: NoiseCalibrator<A>,

    /// Current step count
    step_count: usize,

    /// Batch size used for the most recent step
    current_batch_size: usize,

    /// Phantom data for unused type parameter
    _phantom: std::marker::PhantomData<D>,
}

/// Adaptive clipping state for DP-SGD
#[derive(Debug, Clone)]
struct AdaptiveClippingState {
    /// Current clipping threshold
    current_threshold: f64,

    /// Target quantile (fraction of gradients that should stay unclipped)
    target_quantile: f64,

    /// Geometric adaptation rate
    adaptationlr: f64,

    /// History of (privatized) below-threshold fractions
    fraction_history: VecDeque<f64>,

    /// Non-private quantile estimator over observed norms, used only for
    /// diagnostics that never leave the process
    quantile_estimator: QuantileEstimator,

    /// Number of threshold updates performed
    updates: usize,
}

/// P-squared quantile estimator (Jain & Chlamtac, 1985).
#[derive(Debug, Clone)]
struct QuantileEstimator {
    /// Quantile being estimated, in (0, 1)
    quantile: f64,

    /// P-squared marker state
    p2_state: P2AlgorithmState,

    /// Exponential moving average, used before five samples are available
    ema: f64,

    /// EMA decay factor
    ema_decay: f64,
}

/// P-squared algorithm state.
///
/// `heights[i]` are the marker heights `q_i`, `positions[i]` the marker
/// positions `n_i`, `desired[i]` the desired positions `n'_i` and
/// `increments[i]` the per-observation increments `dn'_i`.
#[derive(Debug, Clone)]
struct P2AlgorithmState {
    heights: [f64; 5],
    positions: [f64; 5],
    desired: [f64; 5],
    increments: [f64; 5],
    count: usize,
    initial: Vec<f64>,
}

/// Privacy budget tracker
#[derive(Debug, Clone)]
struct PrivacyBudgetTracker {
    /// Total epsilon consumed (as reported by the accountant)
    epsilon_consumed: f64,

    /// Target epsilon
    target_epsilon: f64,

    /// Delta at which epsilon is reported
    reporting_delta: f64,

    /// Privacy consumption history
    consumption_history: Vec<PrivacyConsumption>,
}

/// Privacy consumption record
#[derive(Debug, Clone)]
pub struct PrivacyConsumption {
    /// Step at which the release happened.
    pub step: usize,
    /// Cumulative epsilon after the release.
    pub epsilon_spent: f64,
    /// Delta the epsilon is reported at.
    pub reporting_delta: f64,
    /// Batch size used for the release.
    pub batchsize: usize,
    /// Noise multiplier used for the release.
    pub noise_multiplier: f64,
}

/// Gradient statistics for DP-SGD
#[derive(Debug, Clone)]
struct GradientStatistics<A: Float + Default + Clone + std::iter::Sum> {
    /// Recent gradient norms
    norm_history: VecDeque<A>,

    /// Exponential moving average of the clipping indicator
    clipping_frequency: f64,

    /// EMA rate for the clipping frequency
    clipping_frequency_rate: f64,

    /// Average gradient norm
    avg_norm: A,

    /// Std deviation of gradient norms
    std_norm: A,

    /// Percentile statistics
    percentiles: HashMap<String, A>,

    /// Maximum history size
    max_history_size: usize,
}

/// Noise calibration for different mechanisms
#[derive(Debug, Clone)]
struct NoiseCalibrator<A: Float> {
    /// Current noise multiplier
    noise_multiplier: A,

    /// Base noise scale
    base_noise_scale: A,

    /// Noise mechanism
    mechanism: NoiseMechanism,

    /// Calibration history
    calibration_history: Vec<NoiseCalibration<A>>,
}

/// Noise calibration record
#[derive(Debug, Clone)]
pub struct NoiseCalibration<A: Float> {
    /// Step of the record.
    pub step: usize,
    /// Standard deviation of the noise added.
    pub noise_scale: A,
    /// Mean pre-clipping gradient norm.
    pub gradientnorm: A,
    /// Clipping threshold in force.
    pub clipping_threshold: A,
    /// Cumulative epsilon after the release.
    pub privacy_cost: A,
}

impl<O, A, D> DPSGDOptimizer<O, A, D>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug
        + std::iter::Sum
        + std::ops::AddAssign,
    D: scirs2_core::ndarray::Dimension,
    O: Optimizer<A, D> + Send + Sync,
{
    /// Create a new DP-SGD optimizer.
    ///
    /// The configuration is validated and the accountant named by
    /// `config.accounting_method` is constructed; an unimplemented accounting
    /// method or noise mechanism is an error rather than a silent fallback.
    pub fn new(baseoptimizer: O, config: DifferentialPrivacyConfig) -> Result<Self> {
        config.validate()?;

        if config.secure_aggregation {
            return Err(OptimError::InvalidPrivacyConfig(
                "secure_aggregation is not wired into DPSGDOptimizer; use \
                 privacy::secure_aggregation::SecureAggregator explicitly rather than enabling \
                 a flag that would otherwise be ignored"
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

        let rng = scirs2_core::random::thread_rng();

        let adaptive_clipping = if config.adaptive_clipping {
            Some(AdaptiveClippingState::new(
                config.adaptive_clip_init,
                config.adaptive_clip_lr,
                config.adaptive_clip_target_quantile,
            )?)
        } else {
            None
        };

        let privacy_budget = PrivacyBudgetTracker::new(&config);
        let gradient_stats = GradientStatistics::new();
        let noise_calibrator = NoiseCalibrator::new(&config)?;

        let batchsize = config.batch_size;
        Ok(Self {
            baseoptimizer,
            config,
            accountant,
            pure_epsilon_spent: 0.0,
            rng,
            adaptive_clipping,
            privacy_budget,
            gradient_stats,
            noise_calibrator,
            step_count: 0,
            current_batch_size: batchsize,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Perform a DP-SGD step from **per-example** gradients.
    ///
    /// Each gradient is clipped to the current threshold `C`, the clipped
    /// gradients are summed, `N(0, sigma^2 C^2)` is added to the sum, and the
    /// result is divided by the batch size. The privacy budget is enforced
    /// before anything is released.
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

        let batchsize = per_example_gradients.len();
        self.enforce_budget(batchsize)?;

        let clipping_threshold = self.get_clipping_threshold();
        let mut summed: Array<A, D> = Array::zeros(params.raw_dim());
        let mut norm_sum = 0.0;
        let mut below_threshold = 0usize;
        let mut clipped_count = 0usize;

        for gradient in per_example_gradients {
            let norm = self.compute_gradient_norm_f64(gradient)?;
            norm_sum += norm;

            let scale = if norm > clipping_threshold && norm > 0.0 {
                clipped_count += 1;
                clipping_threshold / norm
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

        self.add_noise(&mut summed, clipping_threshold)?;

        let batch_a = A::from(batchsize as f64)
            .ok_or_else(|| OptimError::InvalidConfig("failed to convert batch size".to_string()))?;
        summed.mapv_inplace(|x| x / batch_a);

        let mean_norm = norm_sum / batchsize as f64;
        self.finish_step(batchsize, mean_norm, clipping_threshold, clipped_count)?;

        if self.config.adaptive_clipping {
            self.update_adaptive_clipping(below_threshold, batchsize)?;
        }

        self.baseoptimizer.step(params, &summed)
    }

    /// Perform a DP-SGD step from a pre-clipped **sum** of per-example
    /// gradients.
    ///
    /// The caller asserts that each summed gradient was clipped to
    /// `config.l2_norm_clip`.
    pub fn dp_step_presummed(
        &mut self,
        params: &Array<A, D>,
        summed_clipped_gradients: &Array<A, D>,
        batchsize: usize,
    ) -> Result<Array<A, D>> {
        if batchsize == 0 {
            return Err(OptimError::InvalidConfig(
                "batchsize must be positive".to_string(),
            ));
        }
        if summed_clipped_gradients.raw_dim() != params.raw_dim() {
            return Err(OptimError::DimensionMismatch(format!(
                "gradient shape {:?} does not match parameter shape {:?}",
                summed_clipped_gradients.shape(),
                params.shape()
            )));
        }

        self.enforce_budget(batchsize)?;

        let clipping_threshold = self.get_clipping_threshold();
        let mut summed = summed_clipped_gradients.clone();
        self.add_noise(&mut summed, clipping_threshold)?;

        let batch_a = A::from(batchsize as f64)
            .ok_or_else(|| OptimError::InvalidConfig("failed to convert batch size".to_string()))?;
        summed.mapv_inplace(|x| x / batch_a);

        let mean_norm =
            self.compute_gradient_norm_f64(summed_clipped_gradients)? / batchsize as f64;
        self.finish_step(batchsize, mean_norm, clipping_threshold, 0)?;

        self.baseoptimizer.step(params, &summed)
    }

    /// Perform a DP-SGD step that clips an **already aggregated** gradient.
    ///
    /// # This does not provide per-example differential privacy
    ///
    /// Clipping the batch gradient bounds the influence of the whole batch,
    /// not of any single example. The step is therefore accounted with
    /// sampling probability `q = 1` (batch-level adjacency, no subsampling
    /// amplification) and requires
    /// [`DifferentialPrivacyConfig::acknowledge_aggregate_clipping`].
    /// Prefer [`Self::dp_step_per_example`].
    pub fn dp_step(
        &mut self,
        params: &Array<A, D>,
        gradients: &mut Array<A, D>,
        batchsize: usize,
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
        if batchsize == 0 {
            return Err(OptimError::InvalidConfig(
                "batchsize must be positive".to_string(),
            ));
        }

        // Batch-level adjacency: account without subsampling amplification.
        self.enforce_budget(self.config.dataset_size)?;

        let pre_clip_norm = self.compute_gradient_norm_f64(gradients)?;
        let clipping_threshold = self.get_clipping_threshold();
        let clipped = if pre_clip_norm > clipping_threshold && pre_clip_norm > 0.0 {
            let scale = A::from(clipping_threshold / pre_clip_norm).ok_or_else(|| {
                OptimError::InvalidConfig("failed to convert clipping scale".to_string())
            })?;
            gradients.mapv_inplace(|g| g * scale);
            1
        } else {
            0
        };

        self.add_noise(gradients, clipping_threshold)?;
        self.finish_step(
            self.config.dataset_size,
            pre_clip_norm,
            clipping_threshold,
            clipped,
        )?;

        self.baseoptimizer.step(params, gradients)
    }

    /// Refuse the step unless it fits inside the budget and the step cap.
    fn enforce_budget(&mut self, batchsize: usize) -> Result<()> {
        if self.step_count >= self.config.max_steps {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.consumed_epsilon()?,
                target_epsilon: self.config.target_epsilon,
            });
        }

        let projected = self.projected_epsilon(batchsize)?;
        if projected > self.config.target_epsilon {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.consumed_epsilon()?,
                target_epsilon: self.config.target_epsilon,
            });
        }
        Ok(())
    }

    /// Record a released step: accounting, statistics, calibration history.
    fn finish_step(
        &mut self,
        batchsize: usize,
        mean_pre_clip_norm: f64,
        clipping_threshold: f64,
        clipped_count: usize,
    ) -> Result<()> {
        self.step_count += 1;
        self.current_batch_size = batchsize;

        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let q = self.sampling_probability(batchsize);
                self.accountant
                    .compose_subsampled_gaussian(self.config.noise_multiplier, q, 1)?;
            }
            NoiseMechanism::Laplace => {
                self.pure_epsilon_spent += self.laplace_epsilon_per_step();
            }
            other => {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "noise mechanism {other:?} is not implemented for DP-SGD"
                )));
            }
        }

        let epsilon_spent = self.consumed_epsilon()?;
        self.privacy_budget.record(
            self.step_count,
            epsilon_spent,
            self.reporting_delta(),
            batchsize,
            self.config.noise_multiplier,
        );

        let norm_a = A::from(mean_pre_clip_norm).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert gradient norm".to_string())
        })?;
        self.gradient_stats.update_norm(norm_a)?;
        self.gradient_stats
            .update_clipping(clipped_count * 2 > batchsize.max(1));

        if let Some(ref mut state) = self.adaptive_clipping {
            state.observe_norm(mean_pre_clip_norm);
        }

        let threshold_a = A::from(clipping_threshold).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert clipping threshold".to_string())
        })?;
        let cost_a = A::from(if epsilon_spent.is_finite() {
            epsilon_spent
        } else {
            f64::MAX
        })
        .ok_or_else(|| OptimError::InvalidConfig("failed to convert privacy cost".to_string()))?;
        self.noise_calibrator
            .update_calibration(self.step_count, norm_a, threshold_a, cost_a);

        Ok(())
    }

    /// Differentially private clipping-threshold update
    /// (Andrew et al. 2021).
    ///
    /// The number of per-example gradients below the current threshold is a
    /// counting query with sensitivity 1; it is released with Gaussian noise
    /// and charged to the budget. The threshold is then updated
    /// geometrically towards the target quantile.
    fn update_adaptive_clipping(&mut self, below_threshold: usize, batchsize: usize) -> Result<()> {
        if batchsize == 0 {
            return Ok(());
        }

        let sigma_count = self.config.adaptive_clip_noise_multiplier;
        let noisy_below = below_threshold as f64 + sigma_count * standard_normal(&mut self.rng);
        let fraction = (noisy_below / batchsize as f64).clamp(0.0, 1.0);

        if let Some(ref mut state) = self.adaptive_clipping {
            state.update_threshold(fraction);
        } else {
            return Ok(());
        }

        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let q = self.sampling_probability(batchsize);
                self.accountant
                    .compose_subsampled_gaussian(sigma_count, q, 1)?;
            }
            NoiseMechanism::Laplace => {
                self.pure_epsilon_spent += self.laplace_epsilon_per_step();
            }
            other => {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "noise mechanism {other:?} is not implemented for DP-SGD"
                )));
            }
        }

        let epsilon_spent = self.consumed_epsilon()?;
        self.privacy_budget.record(
            self.step_count,
            epsilon_spent,
            self.reporting_delta(),
            batchsize,
            sigma_count,
        );

        Ok(())
    }

    /// Sampling probability for a batch of the given size.
    fn sampling_probability(&self, batchsize: usize) -> f64 {
        if self.config.dataset_size == 0 {
            0.0
        } else {
            (batchsize as f64 / self.config.dataset_size as f64).min(1.0)
        }
    }

    /// Per-step epsilon allowance of the pure epsilon-DP (Laplace) path.
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

    /// Epsilon consumed so far. Accounting errors are propagated, never
    /// swallowed into a zero spend.
    pub fn consumed_epsilon(&self) -> Result<f64> {
        match self.config.noise_mechanism {
            NoiseMechanism::Laplace => Ok(self.pure_epsilon_spent),
            _ => {
                let (epsilon, _) = self.accountant.privacy_spent(self.config.target_delta)?;
                Ok(epsilon)
            }
        }
    }

    /// Epsilon after one more step with the given batch size.
    fn projected_epsilon(&self, batchsize: usize) -> Result<f64> {
        match self.config.noise_mechanism {
            NoiseMechanism::Laplace => {
                Ok(self.pure_epsilon_spent + self.laplace_epsilon_per_step())
            }
            NoiseMechanism::Gaussian => {
                let q = self.sampling_probability(batchsize);
                let (epsilon, _) = self.accountant.projected_privacy_spent(
                    self.config.noise_multiplier,
                    q,
                    1,
                    self.config.target_delta,
                )?;
                Ok(epsilon)
            }
            other => Err(OptimError::InvalidPrivacyConfig(format!(
                "noise mechanism {other:?} is not implemented for DP-SGD"
            ))),
        }
    }

    /// Whether at least one more step fits inside the budget.
    pub fn has_privacy_budget(&self) -> Result<bool> {
        if self.step_count >= self.config.max_steps {
            return Ok(false);
        }
        Ok(self.projected_epsilon(self.current_batch_size)? <= self.config.target_epsilon)
    }

    /// Current privacy budget status.
    pub fn get_privacy_budget(&self) -> Result<PrivacyBudget> {
        let epsilon_consumed = self.consumed_epsilon()?;
        Ok(PrivacyBudget {
            epsilon_consumed,
            // Delta is a reporting parameter, never an additive spend.
            delta_consumed: 0.0,
            epsilon_remaining: (self.config.target_epsilon - epsilon_consumed).max(0.0),
            delta_remaining: self.reporting_delta(),
            steps_taken: self.step_count,
            accounting_method: self.config.accounting_method,
            estimated_steps_remaining: self.estimate_remaining_steps(epsilon_consumed),
        })
    }

    /// Immutable view of the append-only accounting ledger.
    pub fn accounting_segments(&self) -> &[AccountingSegment] {
        self.accountant.segments()
    }

    /// Get adaptive clipping statistics
    pub fn get_clipping_stats(&self) -> AdaptiveClippingStats {
        AdaptiveClippingStats {
            current_threshold: self.get_clipping_threshold(),
            target_quantile: self
                .adaptive_clipping
                .as_ref()
                .map(|ac| ac.target_quantile)
                .unwrap_or(self.config.adaptive_clip_target_quantile),
            clipping_frequency: self.gradient_stats.clipping_frequency,
            avg_gradient_norm: self.gradient_stats.avg_norm.to_f64().unwrap_or(0.0),
            std_gradient_norm: self.gradient_stats.std_norm.to_f64().unwrap_or(0.0),
            adaptation_rate: self
                .adaptive_clipping
                .as_ref()
                .map(|ac| ac.adaptationlr)
                .unwrap_or(0.0),
            quantile_estimate: self
                .adaptive_clipping
                .as_ref()
                .map(|ac| ac.quantile_estimator.estimate())
                .unwrap_or(0.0),
            threshold_updates: self
                .adaptive_clipping
                .as_ref()
                .map(|ac| ac.updates)
                .unwrap_or(0),
        }
    }

    /// Set the batch size used by subsequent steps.
    ///
    /// This opens a new accounting segment from the next step onwards; steps
    /// already taken keep the cost they were charged. Rebuilding the
    /// accountant here (as an earlier revision did) retroactively rewrote
    /// privacy that had already been spent.
    pub fn set_batch_size(&mut self, batchsize: usize) -> Result<()> {
        if batchsize == 0 || batchsize > self.config.dataset_size {
            return Err(OptimError::InvalidConfig(format!(
                "batch size must be in 1..={}, got {batchsize}",
                self.config.dataset_size
            )));
        }
        self.current_batch_size = batchsize;
        self.config.batch_size = batchsize;
        Ok(())
    }

    /// Update the privacy configuration mid-training.
    ///
    /// The privacy *target* (epsilon, delta), the dataset size, the noise
    /// mechanism and the accounting method are immutable once training has
    /// started: changing them would re-interpret privacy that has already
    /// been spent. Operational parameters (noise multiplier, clipping norm,
    /// batch size, step cap) may change and simply open a new ledger segment.
    pub fn update_privacy_config(&mut self, newconfig: DifferentialPrivacyConfig) -> Result<()> {
        newconfig.validate()?;

        if newconfig.target_epsilon != self.config.target_epsilon
            || newconfig.target_delta != self.config.target_delta
        {
            return Err(OptimError::InvalidConfig(
                "the privacy target (epsilon, delta) cannot be changed mid-training: already \
                 spent privacy would be re-interpreted under a different target"
                    .to_string(),
            ));
        }
        if newconfig.dataset_size != self.config.dataset_size {
            return Err(OptimError::InvalidConfig(
                "dataset_size cannot be changed mid-training: it defines the sampling \
                 probability of steps already accounted"
                    .to_string(),
            ));
        }
        if newconfig.noise_mechanism != self.config.noise_mechanism
            || newconfig.accounting_method != self.config.accounting_method
        {
            return Err(OptimError::InvalidConfig(
                "the noise mechanism and accounting method cannot be changed mid-training"
                    .to_string(),
            ));
        }

        self.privacy_budget.target_epsilon = newconfig.target_epsilon;
        self.privacy_budget.reporting_delta = newconfig.target_delta;
        self.noise_calibrator.noise_multiplier =
            A::from(newconfig.noise_multiplier).ok_or_else(|| {
                OptimError::InvalidConfig("failed to convert noise multiplier".to_string())
            })?;
        self.config = newconfig;
        Ok(())
    }

    /// Compute an L2 gradient norm, rejecting non-finite values.
    fn compute_gradient_norm_f64<S, DIM>(&self, gradients: &ArrayBase<S, DIM>) -> Result<f64>
    where
        S: Data<Elem = A>,
        DIM: Dimension,
    {
        let mut sum_squares = 0.0f64;
        for &value in gradients.iter() {
            let v = value.to_f64().unwrap_or(f64::NAN);
            if !v.is_finite() {
                return Err(OptimError::InvalidConfig(
                    "gradient contains a non-finite value; clipping cannot bound its sensitivity"
                        .to_string(),
                ));
            }
            sum_squares += v * v;
        }
        Ok(sum_squares.sqrt())
    }

    /// Current clipping threshold.
    pub fn get_clipping_threshold(&self) -> f64 {
        if let Some(ref adaptive_state) = self.adaptive_clipping {
            adaptive_state.current_threshold
        } else {
            self.config.l2_norm_clip
        }
    }

    /// Add calibrated noise to a (summed) gradient container.
    ///
    /// * Gaussian: `N(0, (sigma * C)^2)`.
    /// * Laplace: `Lap(b)` with `b = sensitivity / epsilon_step`, the L1
    ///   sensitivity bounded by `sqrt(d) * C`. Sampling a Gaussian and
    ///   labelling it Laplace (as an earlier revision did) provides none of
    ///   the pure epsilon-DP guarantee the mechanism is chosen for.
    /// * Any other mechanism is an error.
    fn add_noise<S, DIM>(
        &mut self,
        gradients: &mut ArrayBase<S, DIM>,
        clipping_threshold: f64,
    ) -> Result<()>
    where
        S: DataMut<Elem = A>,
        DIM: Dimension,
    {
        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let sigma = self.config.noise_multiplier * clipping_threshold;
                if !sigma.is_finite() || sigma <= 0.0 {
                    return Err(OptimError::InvalidConfig(format!(
                        "noise scale must be positive and finite, got {sigma}"
                    )));
                }
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
                        "failed to convert Gaussian noise sample".to_string(),
                    ));
                }
            }
            NoiseMechanism::Laplace => {
                let dimension = gradients.len().max(1) as f64;
                let l1_sensitivity = clipping_threshold * dimension.sqrt();
                let epsilon_step = self.laplace_epsilon_per_step();
                if !(epsilon_step > 0.0) {
                    return Err(OptimError::InvalidPrivacyConfig(
                        "Laplace mechanism requires a positive per-step epsilon".to_string(),
                    ));
                }
                let scale = l1_sensitivity / epsilon_step;
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(OptimError::InvalidConfig(format!(
                        "Laplace scale must be positive and finite, got {scale}"
                    )));
                }
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
                        "failed to convert Laplace noise sample".to_string(),
                    ));
                }
            }
            other => {
                return Err(OptimError::InvalidPrivacyConfig(format!(
                    "noise mechanism {other:?} is not implemented for DP-SGD; use Gaussian or \
                     Laplace"
                )));
            }
        }

        Ok(())
    }

    /// Estimate remaining steps before budget exhaustion.
    fn estimate_remaining_steps(&self, epsilon_consumed: f64) -> usize {
        let by_steps = self.config.max_steps.saturating_sub(self.step_count);
        if self.step_count == 0 || !epsilon_consumed.is_finite() || epsilon_consumed <= 0.0 {
            return by_steps;
        }

        let epsilon_per_step = epsilon_consumed / self.step_count as f64;
        let remaining_epsilon = (self.config.target_epsilon - epsilon_consumed).max(0.0);
        let by_budget = (remaining_epsilon / epsilon_per_step) as usize;
        by_budget.min(by_steps)
    }

    /// Get privacy accounting details
    pub fn get_privacy_accounting_details(&self) -> PrivacyAccountingDetails {
        PrivacyAccountingDetails {
            accounting_segments: self.accountant.segments().to_vec(),
            privacy_consumption_history: self.privacy_budget.consumption_history.clone(),
            gradient_statistics: GradientStatsSnapshot {
                avg_norm: self.gradient_stats.avg_norm.to_f64().unwrap_or(0.0),
                std_norm: self.gradient_stats.std_norm.to_f64().unwrap_or(0.0),
                clipping_frequency: self.gradient_stats.clipping_frequency,
                percentiles: self
                    .gradient_stats
                    .percentiles
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or(0.0)))
                    .collect(),
            },
            noise_calibration_history: self
                .noise_calibrator
                .calibration_history
                .iter()
                .map(|entry| NoiseCalibration {
                    step: entry.step,
                    noise_scale: entry.noise_scale.to_f64().unwrap_or(0.0),
                    gradientnorm: entry.gradientnorm.to_f64().unwrap_or(0.0),
                    clipping_threshold: entry.clipping_threshold.to_f64().unwrap_or(0.0),
                    privacy_cost: entry.privacy_cost.to_f64().unwrap_or(0.0),
                })
                .collect(),
        }
    }

    /// Validate the DP-SGD configuration and report advisory warnings.
    pub fn validate_configuration(&self) -> Result<ConfigurationValidation> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        if let Err(err) = self.config.validate() {
            errors.push(err.to_string());
        }

        if self.config.noise_multiplier < 0.5 {
            warnings.push(
                "Noise multipliers below 0.5 give very weak guarantees for DP-SGD".to_string(),
            );
        }
        if self.config.noise_multiplier > 10.0 {
            warnings.push("Very high noise multiplier may severely impact utility".to_string());
        }
        if self.config.l2_norm_clip < 0.01 {
            warnings.push("Very low clipping threshold may destroy gradient signal".to_string());
        }
        if self.config.l2_norm_clip > 100.0 {
            warnings.push(
                "Very high clipping threshold may not provide effective clipping".to_string(),
            );
        }
        if self.config.batch_size < 16 {
            warnings.push("Small batch size reduces privacy amplification benefits".to_string());
        }
        if self.config.dataset_size < 1000 {
            warnings
                .push("Small dataset limits the achievable privacy-utility tradeoff".to_string());
        }
        if self.config.target_epsilon > 10.0 {
            warnings.push("Large epsilon provides a weak privacy guarantee".to_string());
        }
        if self.config.target_delta > 1.0 / self.config.dataset_size as f64 {
            errors.push("Delta should be much smaller than 1/n".to_string());
        }

        Ok(ConfigurationValidation {
            is_valid: errors.is_empty(),
            warnings,
            errors,
            recommended_adjustments: self.generate_recommendations(),
        })
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.gradient_stats.clipping_frequency > 0.8 {
            recommendations.push(
                "Consider increasing the clipping threshold - high clipping frequency detected"
                    .to_string(),
            );
        }
        if self.gradient_stats.clipping_frequency < 0.1 && self.step_count > 0 {
            recommendations.push(
                "Consider decreasing the clipping threshold - low clipping frequency detected"
                    .to_string(),
            );
        }
        if let Ok(epsilon) = self.consumed_epsilon() {
            if epsilon / self.config.target_epsilon > 0.9 {
                recommendations.push(
                    "Privacy budget nearly exhausted - increase the noise multiplier or stop"
                        .to_string(),
                );
            }
        }

        recommendations
    }
}

// Implementation of helper structures

impl AdaptiveClippingState {
    fn new(initial_threshold: f64, adaptationlr: f64, target_quantile: f64) -> Result<Self> {
        if !initial_threshold.is_finite() || initial_threshold <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "initial clipping threshold must be positive and finite, got {initial_threshold}"
            )));
        }
        if !adaptationlr.is_finite() || adaptationlr <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "adaptation rate must be positive and finite, got {adaptationlr}"
            )));
        }
        if !(0.0..=1.0).contains(&target_quantile) {
            return Err(OptimError::InvalidConfig(format!(
                "target quantile must be in [0, 1], got {target_quantile}"
            )));
        }

        Ok(Self {
            current_threshold: initial_threshold,
            target_quantile,
            adaptationlr,
            fraction_history: VecDeque::with_capacity(1000),
            quantile_estimator: QuantileEstimator::new(target_quantile.clamp(0.01, 0.99)),
            updates: 0,
        })
    }

    /// Record a (non-private) observed norm for diagnostics only.
    fn observe_norm(&mut self, norm: f64) {
        if norm.is_finite() {
            self.quantile_estimator.update(norm);
        }
    }

    /// Geometric threshold update from a *privatized* below-threshold
    /// fraction (Andrew et al. 2021).
    fn update_threshold(&mut self, private_fraction_below: f64) {
        self.fraction_history.push_back(private_fraction_below);
        if self.fraction_history.len() > 1000 {
            self.fraction_history.pop_front();
        }

        let factor = (-self.adaptationlr * (private_fraction_below - self.target_quantile)).exp();
        self.current_threshold = (self.current_threshold * factor).clamp(1e-6, 1e6);
        self.updates += 1;
    }
}

impl QuantileEstimator {
    fn new(quantile: f64) -> Self {
        Self {
            quantile,
            p2_state: P2AlgorithmState::new(quantile),
            ema: 0.0,
            ema_decay: 0.99,
        }
    }

    fn update(&mut self, value: f64) {
        self.p2_state.update(value);

        if self.p2_state.count == 1 {
            self.ema = value;
        } else {
            self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * value;
        }
    }

    /// Estimate of the configured quantile.
    fn estimate(&self) -> f64 {
        match self.p2_state.estimate() {
            Some(value) => value,
            None => self.ema,
        }
    }
}

impl P2AlgorithmState {
    fn new(quantile: f64) -> Self {
        let p = quantile.clamp(0.0, 1.0);
        Self {
            heights: [0.0; 5],
            positions: [1.0, 2.0, 3.0, 4.0, 5.0],
            desired: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            increments: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            count: 0,
            initial: Vec::with_capacity(5),
        }
    }

    /// Full P-squared update (Jain & Chlamtac 1985): marker position
    /// increments followed by parabolic (falling back to linear) height
    /// adjustment. The earlier revision stopped updating after five samples
    /// and reported a frozen median.
    fn update(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        self.count += 1;

        if self.initial.len() < 5 {
            self.initial.push(value);
            if self.initial.len() == 5 {
                self.initial
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                for (i, &v) in self.initial.iter().enumerate() {
                    self.heights[i] = v;
                }
            }
            return;
        }

        // 1. Find the cell k containing the new observation.
        let k = if value < self.heights[0] {
            self.heights[0] = value;
            0usize
        } else if value >= self.heights[4] {
            self.heights[4] = value;
            3usize
        } else {
            let mut cell = 0usize;
            for i in 0..4 {
                if self.heights[i] <= value && value < self.heights[i + 1] {
                    cell = i;
                    break;
                }
            }
            cell
        };

        // 2. Increment positions above the cell and shift desired positions.
        for i in (k + 1)..5 {
            self.positions[i] += 1.0;
        }
        for i in 0..5 {
            self.desired[i] += self.increments[i];
        }

        // 3. Adjust the interior markers.
        for i in 1..4 {
            let d = self.desired[i] - self.positions[i];
            let gap_up = self.positions[i + 1] - self.positions[i];
            let gap_down = self.positions[i - 1] - self.positions[i];

            if (d >= 1.0 && gap_up > 1.0) || (d <= -1.0 && gap_down < -1.0) {
                let step = if d >= 0.0 { 1.0 } else { -1.0 };

                let parabolic = self.parabolic_prediction(i, step);
                let new_height =
                    if self.heights[i - 1] < parabolic && parabolic < self.heights[i + 1] {
                        parabolic
                    } else {
                        self.linear_prediction(i, step)
                    };

                self.heights[i] = new_height;
                self.positions[i] += step;
            }
        }
    }

    fn parabolic_prediction(&self, i: usize, d: f64) -> f64 {
        let n_prev = self.positions[i - 1];
        let n_cur = self.positions[i];
        let n_next = self.positions[i + 1];
        let q_prev = self.heights[i - 1];
        let q_cur = self.heights[i];
        let q_next = self.heights[i + 1];

        let denom = n_next - n_prev;
        if denom == 0.0 {
            return q_cur;
        }

        let left =
            (n_cur - n_prev + d) * (q_next - q_cur) / (n_next - n_cur).max(f64::MIN_POSITIVE);
        let right =
            (n_next - n_cur - d) * (q_cur - q_prev) / (n_cur - n_prev).max(f64::MIN_POSITIVE);

        q_cur + (d / denom) * (left + right)
    }

    fn linear_prediction(&self, i: usize, d: f64) -> f64 {
        let neighbour = if d > 0.0 { i + 1 } else { i - 1 };
        let gap = self.positions[neighbour] - self.positions[i];
        if gap == 0.0 {
            return self.heights[i];
        }
        self.heights[i] + d * (self.heights[neighbour] - self.heights[i]) / gap
    }

    /// Current estimate, or `None` before five observations are available.
    fn estimate(&self) -> Option<f64> {
        if self.initial.len() < 5 {
            None
        } else {
            Some(self.heights[2])
        }
    }
}

impl PrivacyBudgetTracker {
    fn new(config: &DifferentialPrivacyConfig) -> Self {
        Self {
            epsilon_consumed: 0.0,
            target_epsilon: config.target_epsilon,
            reporting_delta: config.target_delta,
            consumption_history: Vec::new(),
        }
    }

    fn record(
        &mut self,
        step: usize,
        epsilon_spent: f64,
        reporting_delta: f64,
        batchsize: usize,
        noise_multiplier: f64,
    ) {
        self.epsilon_consumed = epsilon_spent;
        self.consumption_history.push(PrivacyConsumption {
            step,
            epsilon_spent,
            reporting_delta,
            batchsize,
            noise_multiplier,
        });

        if self.consumption_history.len() > 10_000 {
            self.consumption_history.remove(0);
        }
    }
}

impl<A: Float + Default + Clone + std::iter::Sum + Send + Sync> GradientStatistics<A> {
    fn new() -> Self {
        Self {
            norm_history: VecDeque::with_capacity(1000),
            clipping_frequency: 0.0,
            clipping_frequency_rate: 0.01,
            avg_norm: A::zero(),
            std_norm: A::zero(),
            percentiles: HashMap::new(),
            max_history_size: 1000,
        }
    }

    fn update_norm(&mut self, norm: A) -> Result<()> {
        self.norm_history.push_back(norm);
        if self.norm_history.len() > self.max_history_size {
            self.norm_history.pop_front();
        }

        let n = A::from(self.norm_history.len()).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert history length".to_string())
        })?;
        self.avg_norm = self.norm_history.iter().cloned().sum::<A>() / n;

        let variance = self
            .norm_history
            .iter()
            .map(|&x| (x - self.avg_norm) * (x - self.avg_norm))
            .sum::<A>()
            / n;
        self.std_norm = variance.sqrt();

        self.update_percentiles();
        Ok(())
    }

    fn update_percentiles(&mut self) {
        if self.norm_history.is_empty() {
            return;
        }

        let mut sorted: Vec<A> = self.norm_history.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        for (label, q) in [("p50", 0.5), ("p90", 0.9), ("p99", 0.99)] {
            let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
            if let Some(&value) = sorted.get(idx.min(sorted.len() - 1)) {
                self.percentiles.insert(label.to_string(), value);
            }
        }
    }

    /// Exponential moving average of the clipping indicator.
    ///
    /// Both outcomes update the average; only ever moving it towards 1.0 (as
    /// an earlier revision did) drives the reported frequency to 1.0 no
    /// matter what happens.
    fn update_clipping(&mut self, was_clipped: bool) {
        let alpha = self.clipping_frequency_rate;
        let indicator = if was_clipped { 1.0 } else { 0.0 };
        self.clipping_frequency = (1.0 - alpha) * self.clipping_frequency + alpha * indicator;
    }
}

impl<A: Float + Default + Clone + Send + Sync> NoiseCalibrator<A> {
    fn new(config: &DifferentialPrivacyConfig) -> Result<Self> {
        let noise_multiplier = A::from(config.noise_multiplier).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert noise multiplier".to_string())
        })?;
        let base_noise_scale =
            A::from(config.noise_multiplier * config.l2_norm_clip).ok_or_else(|| {
                OptimError::InvalidConfig("failed to convert base noise scale".to_string())
            })?;

        Ok(Self {
            noise_multiplier,
            base_noise_scale,
            mechanism: config.noise_mechanism,
            calibration_history: Vec::new(),
        })
    }

    fn update_calibration(
        &mut self,
        step: usize,
        gradientnorm: A,
        clipping_threshold: A,
        privacy_cost: A,
    ) {
        let noise_scale = self.noise_multiplier * clipping_threshold;

        self.calibration_history.push(NoiseCalibration {
            step,
            noise_scale,
            gradientnorm,
            clipping_threshold,
            privacy_cost,
        });

        if self.calibration_history.len() > 1000 {
            self.calibration_history.remove(0);
        }
    }

    /// Mechanism this calibrator was built for.
    #[allow(dead_code)]
    fn mechanism(&self) -> NoiseMechanism {
        self.mechanism
    }

    /// Noise scale implied by the configured clipping norm.
    #[allow(dead_code)]
    fn base_noise_scale(&self) -> A {
        self.base_noise_scale
    }
}

/// Adaptive clipping statistics
#[derive(Debug, Clone)]
pub struct AdaptiveClippingStats {
    /// Threshold currently in force.
    pub current_threshold: f64,
    /// Target fraction of unclipped gradients.
    pub target_quantile: f64,
    /// EMA of the clipping indicator.
    pub clipping_frequency: f64,
    /// Mean observed gradient norm.
    pub avg_gradient_norm: f64,
    /// Std deviation of observed gradient norms.
    pub std_gradient_norm: f64,
    /// Geometric adaptation rate.
    pub adaptation_rate: f64,
    /// P-squared estimate of the target quantile of gradient norms.
    pub quantile_estimate: f64,
    /// Number of privatized threshold updates performed.
    pub threshold_updates: usize,
}

/// Privacy accounting details
#[derive(Debug, Clone)]
pub struct PrivacyAccountingDetails {
    /// Append-only ledger segments.
    pub accounting_segments: Vec<AccountingSegment>,
    /// Per-step consumption records.
    pub privacy_consumption_history: Vec<PrivacyConsumption>,
    /// Gradient statistics snapshot.
    pub gradient_statistics: GradientStatsSnapshot,
    /// Noise calibration history.
    pub noise_calibration_history: Vec<NoiseCalibration<f64>>,
}

/// Gradient statistics snapshot
#[derive(Debug, Clone)]
pub struct GradientStatsSnapshot {
    /// Mean gradient norm.
    pub avg_norm: f64,
    /// Std deviation of gradient norms.
    pub std_norm: f64,
    /// EMA of the clipping indicator.
    pub clipping_frequency: f64,
    /// Selected percentiles of observed gradient norms.
    pub percentiles: HashMap<String, f64>,
}

/// Configuration validation result
#[derive(Debug, Clone)]
pub struct ConfigurationValidation {
    /// Whether the configuration is usable.
    pub is_valid: bool,
    /// Advisory warnings.
    pub warnings: Vec<String>,
    /// Blocking errors.
    pub errors: Vec<String>,
    /// Suggested adjustments.
    pub recommended_adjustments: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;
    use crate::privacy::AccountingMethod;
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

    fn build(config: DifferentialPrivacyConfig) -> DPSGDOptimizer<SGD<f64>, f64, Ix1> {
        match DPSGDOptimizer::new(SGD::new(0.01), config) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        }
    }

    #[test]
    fn test_dp_sgd_creation() {
        let dp_sgd = DPSGDOptimizer::<_, f64, Ix1>::new(
            SGD::new(0.01),
            DifferentialPrivacyConfig::default(),
        );
        assert!(dp_sgd.is_ok());
    }

    #[test]
    fn test_per_example_step_runs_many_steps_with_monotone_epsilon() {
        let mut optimizer = build(test_config());
        let params = Array1::<f64>::zeros(4);
        let batch: Vec<Array1<f64>> = (0..8).map(|_| Array1::from_elem(4, 0.25)).collect();

        let mut previous = 0.0;
        for step in 1..=100 {
            let result = optimizer.dp_step_per_example(&params, &batch);
            assert!(result.is_ok(), "step {step} failed: {result:?}");
            let epsilon = match optimizer.consumed_epsilon() {
                Ok(value) => value,
                Err(err) => panic!("accounting failed: {err}"),
            };
            assert!(
                epsilon > previous,
                "epsilon must increase strictly: {previous} -> {epsilon}"
            );
            previous = epsilon;
        }

        let budget = match optimizer.get_privacy_budget() {
            Ok(budget) => budget,
            Err(err) => panic!("budget query failed: {err}"),
        };
        assert_eq!(budget.steps_taken, 100);
        assert!(budget.epsilon_consumed < budget.epsilon_consumed + budget.epsilon_remaining);
    }

    #[test]
    fn test_budget_exhaustion_terminates_training() {
        let config = DifferentialPrivacyConfig {
            target_epsilon: 3.0,
            noise_multiplier: 1.0,
            batch_size: 64,
            dataset_size: 640,
            max_steps: 100_000,
            ..test_config()
        };
        let mut optimizer = build(config);
        let params = Array1::<f64>::zeros(2);
        let batch: Vec<Array1<f64>> = (0..64).map(|_| Array1::from_elem(2, 0.5)).collect();

        let mut steps = 0;
        loop {
            match optimizer.dp_step_per_example(&params, &batch) {
                Ok(_) => {
                    steps += 1;
                    assert!(steps < 100_000, "budget was never enforced");
                }
                Err(OptimError::PrivacyBudgetExhausted { .. }) => break,
                Err(other) => panic!("unexpected error: {other}"),
            }
        }

        assert!(steps > 0);
        let epsilon = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };
        assert!(epsilon <= 3.0 + 1e-12);
    }

    #[test]
    fn test_aggregate_step_is_opt_in() {
        let mut optimizer = build(test_config());
        let params = Array1::<f64>::zeros(2);
        let mut gradients = Array1::from_elem(2, 0.5);
        assert!(optimizer.dp_step(&params, &mut gradients, 8).is_err());

        let mut acknowledged = build(DifferentialPrivacyConfig {
            acknowledge_aggregate_clipping: true,
            ..test_config()
        });
        assert!(acknowledged.dp_step(&params, &mut gradients, 8).is_ok());
    }

    #[test]
    fn test_batch_size_is_used_by_accounting() {
        // A larger batch means a larger sampling probability and therefore a
        // larger privacy cost per step. An implementation that ignored the
        // batch size would report identical epsilons.
        let params = Array1::<f64>::zeros(2);

        let mut small = build(test_config());
        let small_batch: Vec<Array1<f64>> = (0..4).map(|_| Array1::from_elem(2, 0.1)).collect();
        assert!(small.dp_step_per_example(&params, &small_batch).is_ok());

        let mut large = build(test_config());
        let large_batch: Vec<Array1<f64>> = (0..512).map(|_| Array1::from_elem(2, 0.1)).collect();
        assert!(large.dp_step_per_example(&params, &large_batch).is_ok());

        let eps_small = match small.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };
        let eps_large = match large.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };
        assert!(
            eps_large > eps_small,
            "batch size must affect accounting: {eps_small} vs {eps_large}"
        );
        assert_eq!(
            small.accounting_segments()[0].sampling_probability,
            4.0 / 50_000.0
        );
        assert_eq!(
            large.accounting_segments()[0].sampling_probability,
            512.0 / 50_000.0
        );
    }

    #[test]
    fn test_changing_batch_size_appends_a_segment_and_never_rewrites_history() {
        let mut optimizer = build(test_config());
        let params = Array1::<f64>::zeros(2);
        let batch_a: Vec<Array1<f64>> = (0..8).map(|_| Array1::from_elem(2, 0.1)).collect();
        for _ in 0..5 {
            assert!(optimizer.dp_step_per_example(&params, &batch_a).is_ok());
        }
        let epsilon_before = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };

        assert!(optimizer.set_batch_size(256).is_ok());
        let batch_b: Vec<Array1<f64>> = (0..256).map(|_| Array1::from_elem(2, 0.1)).collect();
        assert!(optimizer.dp_step_per_example(&params, &batch_b).is_ok());

        let segments = optimizer.accounting_segments();
        assert_eq!(segments.len(), 2, "a parameter change must open a segment");
        assert_eq!(segments[0].steps, 5);
        assert_eq!(segments[1].steps, 1);

        let epsilon_after = match optimizer.consumed_epsilon() {
            Ok(value) => value,
            Err(err) => panic!("accounting failed: {err}"),
        };
        assert!(
            epsilon_after > epsilon_before,
            "already-spent privacy must never be reduced by a configuration change"
        );
    }

    #[test]
    fn test_privacy_target_cannot_be_changed_mid_training() {
        let mut optimizer = build(test_config());
        let relaxed = DifferentialPrivacyConfig {
            target_epsilon: 100.0,
            ..test_config()
        };
        assert!(optimizer.update_privacy_config(relaxed).is_err());

        let tightened = DifferentialPrivacyConfig {
            target_delta: 1e-9,
            ..test_config()
        };
        assert!(optimizer.update_privacy_config(tightened).is_err());

        let operational = DifferentialPrivacyConfig {
            noise_multiplier: 2.0,
            ..test_config()
        };
        assert!(optimizer.update_privacy_config(operational).is_ok());
    }

    #[test]
    fn test_unimplemented_mechanisms_and_methods_error() {
        for mechanism in [
            NoiseMechanism::TreeAggregation,
            NoiseMechanism::ImprovedComposition,
        ] {
            let mut optimizer = build(DifferentialPrivacyConfig {
                noise_mechanism: mechanism,
                ..test_config()
            });
            let params = Array1::<f64>::zeros(2);
            let batch = vec![Array1::from_elem(2, 0.1)];
            assert!(optimizer.dp_step_per_example(&params, &batch).is_err());
        }

        for method in [
            AccountingMethod::AdvancedComposition,
            AccountingMethod::ZCDP,
        ] {
            let result = DPSGDOptimizer::<SGD<f64>, f64, Ix1>::new(
                SGD::new(0.01),
                DifferentialPrivacyConfig {
                    accounting_method: method,
                    ..test_config()
                },
            );
            assert!(result.is_err(), "{method:?} must not be silently accepted");
        }
    }

    #[test]
    fn test_non_finite_gradients_are_rejected() {
        let mut optimizer = build(test_config());
        let params = Array1::<f64>::zeros(2);
        let batch = vec![Array1::from_vec(vec![f64::NAN, 1.0])];
        assert!(optimizer.dp_step_per_example(&params, &batch).is_err());
    }

    #[test]
    fn test_adaptive_clipping_state() {
        let state = AdaptiveClippingState::new(1.0, 0.1, 0.5);
        assert!(state.is_ok());
        let state = match state {
            Ok(state) => state,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert_eq!(state.current_threshold, 1.0);
        assert_eq!(state.adaptationlr, 0.1);

        assert!(AdaptiveClippingState::new(0.0, 0.1, 0.5).is_err());
        assert!(AdaptiveClippingState::new(1.0, -1.0, 0.5).is_err());
        assert!(AdaptiveClippingState::new(1.0, 0.1, 2.0).is_err());
    }

    #[test]
    fn test_adaptive_threshold_moves_towards_the_target_quantile() {
        let mut state = match AdaptiveClippingState::new(1.0, 0.5, 0.5) {
            Ok(state) => state,
            Err(err) => panic!("construction failed: {err}"),
        };

        // Everything clipped (fraction below = 0 < target) -> raise C.
        state.update_threshold(0.0);
        assert!(state.current_threshold > 1.0);

        // Nothing clipped (fraction below = 1 > target) -> lower C.
        let raised = state.current_threshold;
        state.update_threshold(1.0);
        assert!(state.current_threshold < raised);
        assert_eq!(state.updates, 2);
    }

    #[test]
    fn test_privacy_budget_tracker() {
        let config = DifferentialPrivacyConfig::default();
        let tracker = PrivacyBudgetTracker::new(&config);
        assert_eq!(tracker.target_epsilon, config.target_epsilon);
        assert_eq!(tracker.epsilon_consumed, 0.0);
    }

    #[test]
    fn test_p2_quantile_estimator_tracks_the_median() {
        let mut estimator = QuantileEstimator::new(0.5);
        for i in 1..=1000 {
            estimator.update(i as f64);
        }
        let estimate = estimator.estimate();
        assert!(
            (estimate - 500.0).abs() < 60.0,
            "P-squared median estimate {estimate} is far from 500"
        );
    }

    #[test]
    fn test_p2_quantile_estimator_honours_the_requested_quantile() {
        let mut p90 = QuantileEstimator::new(0.9);
        let mut p10 = QuantileEstimator::new(0.1);
        for i in 1..=1000 {
            p90.update(i as f64);
            p10.update(i as f64);
        }
        let high = p90.estimate();
        let low = p10.estimate();
        assert!(
            high > low,
            "the 0.9 quantile ({high}) must exceed the 0.1 quantile ({low})"
        );
        assert!((high - 900.0).abs() < 120.0, "p90 estimate was {high}");
        assert!((low - 100.0).abs() < 120.0, "p10 estimate was {low}");
    }

    #[test]
    fn test_p2_estimator_does_not_freeze_after_five_samples() {
        let mut estimator = QuantileEstimator::new(0.5);
        for i in 1..=5 {
            estimator.update(i as f64);
        }
        let early = estimator.estimate();
        for i in 6..=500 {
            estimator.update(i as f64);
        }
        let late = estimator.estimate();
        assert!(
            late > early,
            "the estimator must keep updating past five samples: {early} -> {late}"
        );
    }

    #[test]
    fn test_p2_estimator_ignores_non_finite_input() {
        let mut estimator = QuantileEstimator::new(0.5);
        for i in 1..=10 {
            estimator.update(i as f64);
        }
        let before = estimator.estimate();
        estimator.update(f64::NAN);
        estimator.update(f64::INFINITY);
        assert_eq!(estimator.estimate(), before);
    }

    #[test]
    fn test_gradient_statistics() {
        let mut stats = GradientStatistics::<f64>::new();
        assert!(stats.update_norm(1.0).is_ok());
        assert!(stats.update_norm(2.0).is_ok());
        assert!(stats.update_norm(3.0).is_ok());

        assert_eq!(stats.avg_norm, 2.0);
        assert!(stats.std_norm > 0.0);
        assert!(stats.percentiles.contains_key("p50"));
    }

    #[test]
    fn test_clipping_frequency_ema_moves_in_both_directions() {
        let mut stats = GradientStatistics::<f64>::new();
        for _ in 0..500 {
            stats.update_clipping(true);
        }
        let after_clipping = stats.clipping_frequency;
        assert!(after_clipping > 0.9);

        for _ in 0..500 {
            stats.update_clipping(false);
        }
        assert!(
            stats.clipping_frequency < after_clipping,
            "the EMA must fall when clipping stops: {after_clipping} -> {}",
            stats.clipping_frequency
        );
    }
}
