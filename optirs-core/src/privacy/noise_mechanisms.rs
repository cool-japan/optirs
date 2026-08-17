// Noise mechanisms for differential privacy
//
// This module implements the noise mechanisms used in differential privacy:
// Gaussian, Laplace, exponential, truncated noise, tree aggregation and the
// sparse vector technique.
//
// # Randomness
//
// Every mechanism seeds its generator from OS entropy. A hardcoded seed makes
// the "noise" a deterministic function of the data, which provides no privacy
// whatsoever -- an attacker who knows the seed subtracts the noise exactly.
// `*_with_seed` constructors exist for reproducible tests and must never be
// used to release real data.

// Parameter validation uses `!(x > zero)` deliberately: unlike `x <= zero`, this
// form rejects NaN (a NaN comparison is `false`, so its negation is `true`). Under
// differential privacy a silently-accepted NaN scale voids the guarantee, so the
// NaN-rejecting spelling is a correctness requirement, not a style choice.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::fmt::Debug;

use scirs2_core::ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayViewMut, Data, DataMut, Dimension, IxDyn,
};
use scirs2_core::numeric::Float;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use std::marker::PhantomData;

use crate::error::{OptimError, Result};

/// Random generator used by the mechanisms (seedable, `Send`).
type MechanismRng = scirs2_core::random::Random<scirs2_core::random::rngs::StdRng>;

/// Create a generator seeded from OS entropy.
fn os_seeded_rng() -> MechanismRng {
    SeedableRng::from_rng(&mut thread_rng())
}

/// Create a deterministic generator (tests and reproducible benchmarks only).
fn seeded_rng(seed: u64) -> MechanismRng {
    scirs2_core::random::Random::seed(seed)
}

/// Draw a standard normal sample (Box-Muller), never returning a non-finite
/// value: `gen_range(0.0..1.0)` includes 0.0 and `ln(0) = -inf`.
fn standard_normal(rng: &mut MechanismRng) -> f64 {
    let u1: f64 = 1.0 - rng.gen_range(0.0..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Draw a `Laplace(0, 1)` sample by inverse transform sampling.
fn standard_laplace(rng: &mut MechanismRng) -> f64 {
    let raw: f64 = rng.gen_range(0.0..1.0);
    let u = raw.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    if u < 0.5 {
        (2.0 * u).ln()
    } else {
        -(2.0 * (1.0 - u)).ln()
    }
}

// Type alias for complex sensitivity function type
type SensitivityFn<T> = Box<dyn Fn(&[T]) -> T + Send + Sync>;

/// Trait for differential privacy noise mechanisms
pub trait NoiseMechanism<T: Float + Debug + Send + Sync + 'static> {
    /// Add noise to a dimension-erased view.
    ///
    /// This is the single required operation: the fixed-dimension helpers
    /// below are implemented in terms of it via `view_mut().into_dyn()`,
    /// which is a safe, runtime-checked reshape. An earlier revision
    /// `transmute`d `&mut ArrayBase<S, D>` into a concrete `Array<T, IxN>`,
    /// which is undefined behaviour for every storage type and dimension.
    fn add_noise_dyn(
        &mut self,
        data: &mut ArrayViewMut<'_, T, IxDyn>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()>;

    /// Add noise to maintain differential privacy for 1D arrays
    fn add_noise_1d(
        &mut self,
        data: &mut Array<T, scirs2_core::ndarray::Ix1>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        let mut view = data.view_mut().into_dyn();
        self.add_noise_dyn(&mut view, sensitivity, epsilon, delta)
    }

    /// Add noise to maintain differential privacy for 2D arrays
    fn add_noise_2d(
        &mut self,
        data: &mut Array<T, scirs2_core::ndarray::Ix2>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        let mut view = data.view_mut().into_dyn();
        self.add_noise_dyn(&mut view, sensitivity, epsilon, delta)
    }

    /// Add noise to maintain differential privacy for 3D arrays
    fn add_noise_3d(
        &mut self,
        data: &mut Array<T, scirs2_core::ndarray::Ix3>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        let mut view = data.view_mut().into_dyn();
        self.add_noise_dyn(&mut view, sensitivity, epsilon, delta)
    }

    /// Get the mechanism name
    fn name(&self) -> &'static str;

    /// Check if mechanism supports (ε, δ)-DP
    fn supports_delta(&self) -> bool;

    /// Parameters of the most recent noise application.
    ///
    /// Returns `None` before the mechanism has been used: reporting a scale
    /// of zero for a mechanism that has never run reads as "no noise needed",
    /// which is exactly backwards.
    fn get_parameters(&self) -> Option<NoiseParameters<T>>;
}

/// Add noise to an arbitrary owned/borrowed array through the dyn interface.
pub fn add_noise_to<T, M, S, D>(
    mechanism: &mut M,
    data: &mut ArrayBase<S, D>,
    sensitivity: T,
    epsilon: T,
    delta: Option<T>,
) -> Result<()>
where
    T: Float + Debug + Send + Sync + 'static,
    M: NoiseMechanism<T> + ?Sized,
    S: DataMut<Elem = T>,
    D: Dimension,
{
    let mut view = data.view_mut().into_dyn();
    mechanism.add_noise_dyn(&mut view, sensitivity, epsilon, delta)
}

/// Noise mechanism parameters
#[derive(Debug, Clone)]
pub struct NoiseParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Mechanism name.
    pub mechanism_type: String,
    /// Noise scale actually used (sigma for Gaussian, b for Laplace).
    pub scale: T,
    /// Sensitivity used.
    pub sensitivity: T,
    /// Epsilon used.
    pub epsilon: T,
    /// Delta used, if any.
    pub delta: Option<T>,
    /// Distribution shape parameter, if any.
    pub shape: Option<T>,
    /// Distribution rate parameter, if any.
    pub rate: Option<T>,
}

/// Gaussian noise mechanism for (ε, δ)-differential privacy
pub struct GaussianMechanism<T: Float + Debug + Send + Sync + 'static> {
    rng: MechanismRng,
    last_parameters: Option<NoiseParameters<T>>,
    _phantom: PhantomData<T>,
}

/// Laplace noise mechanism for ε-differential privacy
pub struct LaplaceMechanism<T: Float + Debug + Send + Sync + 'static> {
    rng: MechanismRng,
    last_parameters: Option<NoiseParameters<T>>,
    _phantom: PhantomData<T>,
}

/// Exponential mechanism for discrete selection
pub struct ExponentialMechanism<T: Float + Debug + Send + Sync + 'static> {
    rng: MechanismRng,
    qualityfunction: Box<dyn Fn(&T) -> T + Send + Sync>,
    _phantom: PhantomData<T>,
}

/// Truncated noise mechanism for bounded sensitivity
pub struct TruncatedNoiseMechanism<T: Float + Debug + Send + Sync + 'static> {
    basemechanism: Box<dyn NoiseMechanism<T> + Send>,
    truncationbound: T,
    _phantom: PhantomData<T>,
}

/// Tree aggregation mechanism for hierarchical (continual) release
pub struct TreeAggregationMechanism<T: Float + Debug + Send + Sync + 'static> {
    tree_height: usize,
    basemechanism: Box<dyn NoiseMechanism<T> + Send>,
    _phantom: PhantomData<T>,
}

/// Sparse Vector Technique mechanism.
///
/// Implements the standard, provably correct SVT (Lyu, Su and Li, "Understanding
/// the Sparse Vector Technique for Differential Privacy", VLDB 2017):
///
/// * the threshold is perturbed **once**, at construction, with
///   `Lap(sensitivity / epsilon_threshold)`;
/// * each query is compared against that fixed noisy threshold using fresh
///   per-query noise `Lap(2 c sensitivity / epsilon_query)`, where `c` is the
///   maximum number of above-threshold answers;
/// * queries **below** the threshold cost nothing and return `Ok(None)`;
/// * the mechanism halts after `c` above-threshold answers.
///
/// Re-noising the threshold on every query (as an earlier revision did) and
/// charging the full budget per query destroys both the privacy analysis and
/// the point of the technique.
pub struct SparseVectorMechanism<T: Float + Debug + Send + Sync + 'static> {
    noisy_threshold: f64,
    sensitivity: f64,
    epsilon_threshold: f64,
    epsilon_query: f64,
    epsilon_value: Option<f64>,
    queries_answered: usize,
    max_queries: usize,
    rng: MechanismRng,
    _phantom: PhantomData<T>,
}

/// Smooth sensitivity mechanism
pub struct SmoothSensitivityMechanism<T: Float + Debug + Send + Sync + 'static> {
    beta: T,
    sensitivity_function: SensitivityFn<T>,
    _phantom: PhantomData<T>,
}

/// Advanced noise calibration
pub struct NoiseCalibrator<T: Float + Debug + Send + Sync + 'static> {
    /// Target privacy parameters
    target_epsilon: T,
    target_delta: Option<T>,

    /// Sensitivity bounds
    l2_sensitivity: T,
    l1_sensitivity: T,
    linf_sensitivity: T,

    /// Mechanism selection strategy
    selection_strategy: MechanismSelectionStrategy,

    /// The single mechanism instance this calibrator drives. Constructing a
    /// fresh mechanism per call would reseed the generator on every use.
    mechanism: Box<dyn NoiseMechanism<T> + Send>,

    /// Adaptive noise scaling
    adaptive_scaling: bool,
    scaling_factor: T,
    _phantom: PhantomData<T>,
}

/// Strategy for selecting noise mechanism
#[derive(Debug, Clone, Copy)]
pub enum MechanismSelectionStrategy {
    /// Always use Gaussian mechanism
    AlwaysGaussian,

    /// Always use Laplace mechanism
    AlwaysLaplace,

    /// Choose based on privacy parameters
    PrivacyOptimal,

    /// Choose based on utility optimization
    UtilityOptimal,

    /// Adaptive selection based on data characteristics
    Adaptive,
}

impl<T> Default for GaussianMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GaussianMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    /// Create a new Gaussian mechanism seeded from OS entropy.
    pub fn new() -> Self {
        Self {
            rng: os_seeded_rng(),
            last_parameters: None,
            _phantom: PhantomData,
        }
    }

    /// Create a deterministic mechanism for tests.
    ///
    /// # Warning
    /// Deterministic noise provides **no** privacy. Never use this to release
    /// real data.
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            rng: seeded_rng(seed),
            last_parameters: None,
            _phantom: PhantomData,
        }
    }

    /// Compute the noise scale for the classic Gaussian mechanism.
    ///
    /// `sigma = sqrt(2 ln(1.25 / delta)) * sensitivity / epsilon`
    /// (Dwork & Roth, Theorem A.1).
    ///
    /// The bound is only valid for `epsilon <= 1`; larger epsilons require
    /// the analytic Gaussian mechanism (Balle & Wang 2018) and are rejected
    /// rather than silently returning an unsound scale.
    pub fn compute_noise_scale(sensitivity: T, epsilon: T, delta: T) -> Result<T> {
        if !(epsilon > T::zero()) || !(delta > T::zero()) || delta >= T::one() {
            return Err(OptimError::InvalidConfig(
                "Gaussian mechanism requires epsilon > 0 and delta in (0, 1)".to_string(),
            ));
        }
        if !(sensitivity > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "Gaussian mechanism requires a positive sensitivity".to_string(),
            ));
        }
        if epsilon > T::one() {
            return Err(OptimError::InvalidConfig(
                "the classic Gaussian mechanism bound sqrt(2 ln(1.25/delta)) * S / epsilon is \
                 only valid for epsilon <= 1; use the analytic Gaussian mechanism (Balle & Wang \
                 2018) or the RDP accountant for larger epsilon"
                    .to_string(),
            ));
        }

        let epsilon_f = epsilon.to_f64().unwrap_or(f64::NAN);
        let delta_f = delta.to_f64().unwrap_or(f64::NAN);
        let sensitivity_f = sensitivity.to_f64().unwrap_or(f64::NAN);
        if !epsilon_f.is_finite() || !delta_f.is_finite() || !sensitivity_f.is_finite() {
            return Err(OptimError::InvalidConfig(
                "Gaussian mechanism parameters must be finite".to_string(),
            ));
        }

        let sigma = (2.0 * (1.25 / delta_f).ln()).sqrt() * sensitivity_f / epsilon_f;
        T::from(sigma).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert the computed noise scale".to_string())
        })
    }
}

impl<T> NoiseMechanism<T> for GaussianMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    fn add_noise_dyn(
        &mut self,
        data: &mut ArrayViewMut<'_, T, IxDyn>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        let delta = delta.ok_or_else(|| {
            OptimError::InvalidConfig("Gaussian mechanism requires delta parameter".to_string())
        })?;

        let sigma = Self::compute_noise_scale(sensitivity, epsilon, delta)?;
        let sigma_f64 = sigma.to_f64().unwrap_or(f64::NAN);
        if !sigma_f64.is_finite() || sigma_f64 <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "computed Gaussian noise scale must be positive and finite, got {sigma_f64}"
            )));
        }

        let mut failed = false;
        data.mapv_inplace(|x| {
            let sample = standard_normal(&mut self.rng) * sigma_f64;
            match T::from(sample) {
                Some(noise) => x + noise,
                None => {
                    failed = true;
                    x
                }
            }
        });
        if failed {
            return Err(OptimError::InvalidConfig(
                "failed to convert a Gaussian noise sample into the array element type".to_string(),
            ));
        }

        self.last_parameters = Some(NoiseParameters {
            mechanism_type: "Gaussian".to_string(),
            scale: sigma,
            sensitivity,
            epsilon,
            delta: Some(delta),
            shape: None,
            rate: None,
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Gaussian"
    }

    fn supports_delta(&self) -> bool {
        true
    }

    fn get_parameters(&self) -> Option<NoiseParameters<T>> {
        self.last_parameters.clone()
    }
}

impl<T> Default for LaplaceMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LaplaceMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    /// Create a new Laplace mechanism seeded from OS entropy.
    pub fn new() -> Self {
        Self {
            rng: os_seeded_rng(),
            last_parameters: None,
            _phantom: PhantomData,
        }
    }

    /// Create a deterministic mechanism for tests.
    ///
    /// # Warning
    /// Deterministic noise provides **no** privacy.
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            rng: seeded_rng(seed),
            last_parameters: None,
            _phantom: PhantomData,
        }
    }

    /// Laplace scale `b = sensitivity / epsilon` (the L1 sensitivity).
    pub fn compute_noise_scale(sensitivity: T, epsilon: T) -> Result<T> {
        if !(epsilon > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "Epsilon must be positive for Laplace mechanism".to_string(),
            ));
        }
        if !(sensitivity > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "Laplace mechanism requires a positive sensitivity".to_string(),
            ));
        }

        Ok(sensitivity / epsilon)
    }
}

impl<T> NoiseMechanism<T> for LaplaceMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    fn add_noise_dyn(
        &mut self,
        data: &mut ArrayViewMut<'_, T, IxDyn>,
        sensitivity: T,
        epsilon: T,
        _delta: Option<T>,
    ) -> Result<()> {
        let scale = Self::compute_noise_scale(sensitivity, epsilon)?;
        let scale_f64 = scale.to_f64().unwrap_or(f64::NAN);
        if !scale_f64.is_finite() || scale_f64 <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "computed Laplace scale must be positive and finite, got {scale_f64}"
            )));
        }

        let mut failed = false;
        data.mapv_inplace(|x| {
            let sample = standard_laplace(&mut self.rng) * scale_f64;
            match T::from(sample) {
                Some(noise) => x + noise,
                None => {
                    failed = true;
                    x
                }
            }
        });
        if failed {
            return Err(OptimError::InvalidConfig(
                "failed to convert a Laplace noise sample into the array element type".to_string(),
            ));
        }

        self.last_parameters = Some(NoiseParameters {
            mechanism_type: "Laplace".to_string(),
            scale,
            sensitivity,
            epsilon,
            delta: None,
            shape: None,
            rate: Some(scale),
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Laplace"
    }

    fn supports_delta(&self) -> bool {
        false
    }

    fn get_parameters(&self) -> Option<NoiseParameters<T>> {
        self.last_parameters.clone()
    }
}

impl<T> ExponentialMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    /// Create a new exponential mechanism seeded from OS entropy.
    pub fn new(qualityfunction: Box<dyn Fn(&T) -> T + Send + Sync>) -> Self {
        Self {
            rng: os_seeded_rng(),
            qualityfunction,
            _phantom: PhantomData,
        }
    }

    /// Create a deterministic mechanism for tests.
    ///
    /// # Warning
    /// Deterministic selection provides **no** privacy.
    pub fn new_with_seed(qualityfunction: Box<dyn Fn(&T) -> T + Send + Sync>, seed: u64) -> Self {
        Self {
            rng: seeded_rng(seed),
            qualityfunction,
            _phantom: PhantomData,
        }
    }

    /// Select an output with probability proportional to
    /// `exp(epsilon * quality(c) / (2 * sensitivity))` (McSherry & Talwar 2007).
    pub fn select_output(&mut self, candidates: &[T], sensitivity: T, epsilon: T) -> Result<T> {
        if candidates.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No candidates provided".to_string(),
            ));
        }
        if !(sensitivity > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "exponential mechanism requires a positive sensitivity".to_string(),
            ));
        }
        if !(epsilon > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "exponential mechanism requires a positive epsilon".to_string(),
            ));
        }

        let epsilon_f = epsilon.to_f64().unwrap_or(f64::NAN);
        let sensitivity_f = sensitivity.to_f64().unwrap_or(f64::NAN);
        if !epsilon_f.is_finite() || !sensitivity_f.is_finite() {
            return Err(OptimError::InvalidConfig(
                "exponential mechanism parameters must be finite".to_string(),
            ));
        }

        // Quality scores, shifted by the maximum for numerical stability.
        let mut scores = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let score = (self.qualityfunction)(candidate)
                .to_f64()
                .unwrap_or(f64::NAN);
            if !score.is_finite() {
                return Err(OptimError::InvalidConfig(
                    "quality function produced a non-finite score".to_string(),
                ));
            }
            scores.push(score);
        }

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores
            .iter()
            .map(|score| (epsilon_f * (score - max_score) / (2.0 * sensitivity_f)).exp())
            .collect();

        let total_weight: f64 = weights.iter().sum();
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "exponential mechanism weights must sum to a positive finite value, got \
                 {total_weight}"
            )));
        }

        // Sampling on the unit interval keeps `gen_range` away from a NaN or
        // zero-width range, which would panic.
        let u: f64 = self.rng.gen_range(0.0..1.0);
        let target = u * total_weight;
        let mut cumulative = 0.0;
        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight;
            if target < cumulative {
                return Ok(candidates[i]);
            }
        }

        // Only reachable through floating-point round-off at the very end of
        // the cumulative sum.
        candidates
            .last()
            .copied()
            .ok_or_else(|| OptimError::InvalidConfig("No candidates provided".to_string()))
    }
}

impl<T> TruncatedNoiseMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    /// Create a new truncated noise mechanism.
    pub fn new(
        base_mechanism: Box<dyn NoiseMechanism<T> + Send>,
        truncationbound: T,
    ) -> Result<Self> {
        if !(truncationbound > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "truncation bound must be positive".to_string(),
            ));
        }
        Ok(Self {
            basemechanism: base_mechanism,
            truncationbound,
            _phantom: PhantomData,
        })
    }
}

impl<T> NoiseMechanism<T> for TruncatedNoiseMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    fn add_noise_dyn(
        &mut self,
        data: &mut ArrayViewMut<'_, T, IxDyn>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        self.basemechanism
            .add_noise_dyn(data, sensitivity, epsilon, delta)?;
        data.mapv_inplace(|x| x.max(-self.truncationbound).min(self.truncationbound));
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Truncated"
    }

    fn supports_delta(&self) -> bool {
        self.basemechanism.supports_delta()
    }

    fn get_parameters(&self) -> Option<NoiseParameters<T>> {
        self.basemechanism.get_parameters().map(|mut params| {
            params.mechanism_type = format!("Truncated_{}", params.mechanism_type);
            params
        })
    }
}

impl<T> TreeAggregationMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum,
{
    /// Create a new tree aggregation mechanism.
    ///
    /// `tree_height` must be at least 1: a height of zero skips the
    /// aggregation loop entirely and returns the **un-noised** sum, i.e. the
    /// raw data.
    pub fn new(
        tree_height: usize,
        basemechanism: Box<dyn NoiseMechanism<T> + Send>,
    ) -> Result<Self> {
        if tree_height == 0 {
            return Err(OptimError::InvalidConfig(
                "tree_height must be at least 1; a height of 0 would return the un-noised sum"
                    .to_string(),
            ));
        }
        Ok(Self {
            tree_height,
            basemechanism,
            _phantom: PhantomData,
        })
    }

    /// Height of the aggregation tree.
    pub fn tree_height(&self) -> usize {
        self.tree_height
    }

    /// Aggregate values with a binary tree, noising every internal node.
    ///
    /// The privacy budget is split evenly across the levels, so any prefix
    /// sum -- which touches at most one node per level -- is covered by the
    /// total `epsilon`.
    pub fn aggregate_with_tree(
        &mut self,
        values: &[T],
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<T> {
        if values.is_empty() {
            return Ok(T::zero());
        }
        if !(epsilon > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "tree aggregation requires a positive epsilon".to_string(),
            ));
        }

        let height = T::from(self.tree_height).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert tree height".to_string())
        })?;
        let level_epsilon = epsilon / height;

        let mut current_level = values.to_vec();
        for _level in 0..self.tree_height {
            if current_level.len() <= 1 {
                break;
            }

            let mut next_level = Vec::with_capacity(current_level.len().div_ceil(2));
            for chunk in current_level.chunks(2) {
                let mut sum = Array1::from_vec(vec![chunk.iter().cloned().sum()]);
                self.basemechanism
                    .add_noise_1d(&mut sum, sensitivity, level_epsilon, delta)?;
                next_level.push(sum[0]);
            }

            current_level = next_level;
        }

        Ok(current_level.into_iter().sum())
    }
}

impl<T> SparseVectorMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    /// Create a sparse vector mechanism.
    ///
    /// * `threshold` -- the comparison threshold, perturbed once here.
    /// * `sensitivity` -- L1 sensitivity of the queries.
    /// * `epsilon_threshold` / `epsilon_query` -- budget split between the
    ///   threshold perturbation and the per-query noise.
    /// * `epsilon_value` -- optional extra budget for releasing a noisy
    ///   answer; when `None`, [`Self::answer_query`] reports only that the
    ///   query was above threshold (returning the noisy threshold crossing
    ///   indicator through `above_threshold`).
    /// * `max_queries` -- number of above-threshold answers before the
    ///   mechanism halts.
    pub fn new(
        threshold: T,
        sensitivity: T,
        epsilon_threshold: f64,
        epsilon_query: f64,
        epsilon_value: Option<f64>,
        max_queries: usize,
    ) -> Result<Self> {
        Self::build(
            threshold,
            sensitivity,
            epsilon_threshold,
            epsilon_query,
            epsilon_value,
            max_queries,
            os_seeded_rng(),
        )
    }

    /// Deterministic constructor for tests. Provides **no** privacy.
    pub fn new_with_seed(
        threshold: T,
        sensitivity: T,
        epsilon_threshold: f64,
        epsilon_query: f64,
        epsilon_value: Option<f64>,
        max_queries: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::build(
            threshold,
            sensitivity,
            epsilon_threshold,
            epsilon_query,
            epsilon_value,
            max_queries,
            seeded_rng(seed),
        )
    }

    fn build(
        threshold: T,
        sensitivity: T,
        epsilon_threshold: f64,
        epsilon_query: f64,
        epsilon_value: Option<f64>,
        max_queries: usize,
        mut rng: MechanismRng,
    ) -> Result<Self> {
        let sensitivity_f = sensitivity.to_f64().unwrap_or(f64::NAN);
        let threshold_f = threshold.to_f64().unwrap_or(f64::NAN);

        if !sensitivity_f.is_finite() || sensitivity_f <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "sparse vector technique requires a positive finite sensitivity".to_string(),
            ));
        }
        if !threshold_f.is_finite() {
            return Err(OptimError::InvalidConfig(
                "sparse vector technique requires a finite threshold".to_string(),
            ));
        }
        if !epsilon_threshold.is_finite() || epsilon_threshold <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "epsilon_threshold must be positive and finite".to_string(),
            ));
        }
        if !epsilon_query.is_finite() || epsilon_query <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "epsilon_query must be positive and finite".to_string(),
            ));
        }
        if let Some(value_epsilon) = epsilon_value {
            if !value_epsilon.is_finite() || value_epsilon <= 0.0 {
                return Err(OptimError::InvalidConfig(
                    "epsilon_value must be positive and finite when provided".to_string(),
                ));
            }
        }
        if max_queries == 0 {
            return Err(OptimError::InvalidConfig(
                "max_queries must be at least 1".to_string(),
            ));
        }

        // The threshold is perturbed exactly once, for the lifetime of the
        // mechanism.
        let threshold_noise = standard_laplace(&mut rng) * (sensitivity_f / epsilon_threshold);

        Ok(Self {
            noisy_threshold: threshold_f + threshold_noise,
            sensitivity: sensitivity_f,
            epsilon_threshold,
            epsilon_query,
            epsilon_value,
            queries_answered: 0,
            max_queries,
            rng,
            _phantom: PhantomData,
        })
    }

    /// Total epsilon spent by the mechanism over its whole lifetime.
    pub fn total_epsilon(&self) -> f64 {
        self.epsilon_threshold + self.epsilon_query + self.epsilon_value.unwrap_or(0.0)
    }

    /// Number of above-threshold answers released so far.
    pub fn queries_answered(&self) -> usize {
        self.queries_answered
    }

    /// Whether the mechanism has released its budgeted number of
    /// above-threshold answers.
    pub fn is_halted(&self) -> bool {
        self.queries_answered >= self.max_queries
    }

    /// Test one query against the noisy threshold.
    ///
    /// Below-threshold queries cost nothing and do not advance the counter --
    /// that asymmetry is the entire point of the sparse vector technique.
    pub fn above_threshold(&mut self, query_result: T) -> Result<bool> {
        if self.is_halted() {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.total_epsilon(),
                target_epsilon: self.total_epsilon(),
            });
        }

        let value = query_result.to_f64().unwrap_or(f64::NAN);
        if !value.is_finite() {
            return Err(OptimError::InvalidConfig(
                "query result must be finite".to_string(),
            ));
        }

        let query_scale = 2.0 * self.max_queries as f64 * self.sensitivity / self.epsilon_query;
        let noisy_value = value + standard_laplace(&mut self.rng) * query_scale;

        if noisy_value >= self.noisy_threshold {
            self.queries_answered += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Answer a query if it is above the noisy threshold.
    ///
    /// Returns `Ok(None)` for below-threshold queries -- an earlier revision
    /// returned `Ok(Some(0))`, which a caller cannot distinguish from a
    /// genuine zero answer.
    pub fn answer_query(&mut self, query_result: T) -> Result<Option<T>> {
        if !self.above_threshold(query_result)? {
            return Ok(None);
        }

        let value_epsilon = match self.epsilon_value {
            Some(value_epsilon) => value_epsilon,
            None => {
                return Err(OptimError::InvalidConfig(
                    "this mechanism was constructed without a value-release budget; use \
                     above_threshold to obtain the indicator only"
                        .to_string(),
                ))
            }
        };

        let value = query_result.to_f64().unwrap_or(f64::NAN);
        let value_scale = self.max_queries as f64 * self.sensitivity / value_epsilon;
        let noisy = value + standard_laplace(&mut self.rng) * value_scale;

        let converted = T::from(noisy).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert the noisy answer".to_string())
        })?;
        Ok(Some(converted))
    }
}

impl<T> SmoothSensitivityMechanism<T>
where
    T: Float + Debug + Default + Clone + Send + Sync,
{
    /// Create a smooth sensitivity mechanism with smoothing parameter `beta`.
    pub fn new(beta: T, sensitivity_function: SensitivityFn<T>) -> Result<Self> {
        if !(beta > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "smooth sensitivity requires beta > 0".to_string(),
            ));
        }
        Ok(Self {
            beta,
            sensitivity_function,
            _phantom: PhantomData,
        })
    }

    /// Smoothing parameter.
    pub fn beta(&self) -> T {
        self.beta
    }

    /// Evaluate the configured sensitivity function on a dataset.
    pub fn local_sensitivity(&self, data: &[T]) -> T {
        (self.sensitivity_function)(data)
    }
}

impl<T> NoiseCalibrator<T>
where
    T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static,
{
    /// Create a new noise calibrator holding a single mechanism instance.
    pub fn new(
        target_epsilon: T,
        target_delta: Option<T>,
        l2_sensitivity: T,
        selection_strategy: MechanismSelectionStrategy,
    ) -> Self {
        let mechanism = Self::build_mechanism(selection_strategy, target_delta, l2_sensitivity);
        Self {
            target_epsilon,
            target_delta,
            l2_sensitivity,
            l1_sensitivity: l2_sensitivity, // Default assumption
            linf_sensitivity: l2_sensitivity,
            selection_strategy,
            mechanism,
            adaptive_scaling: false,
            scaling_factor: T::one(),
            _phantom: PhantomData,
        }
    }

    fn build_mechanism(
        strategy: MechanismSelectionStrategy,
        target_delta: Option<T>,
        l2_sensitivity: T,
    ) -> Box<dyn NoiseMechanism<T> + Send> {
        match strategy {
            MechanismSelectionStrategy::AlwaysGaussian => Box::new(GaussianMechanism::new()),
            MechanismSelectionStrategy::AlwaysLaplace => Box::new(LaplaceMechanism::new()),
            MechanismSelectionStrategy::PrivacyOptimal
            | MechanismSelectionStrategy::UtilityOptimal => {
                if target_delta.is_some() {
                    Box::new(GaussianMechanism::new())
                } else {
                    Box::new(LaplaceMechanism::new())
                }
            }
            MechanismSelectionStrategy::Adaptive => {
                // The Gaussian mechanism needs a delta; without one the only
                // sound choice is Laplace, whatever the sensitivity looks
                // like. (An earlier revision compared the L2 sensitivity
                // against an L1 bound that was initialised to the same
                // value, so the branch was never taken.)
                let _ = l2_sensitivity;
                if target_delta.is_some() {
                    Box::new(GaussianMechanism::new())
                } else {
                    Box::new(LaplaceMechanism::new())
                }
            }
        }
    }

    /// Name of the mechanism this calibrator drives.
    pub fn mechanism_name(&self) -> &'static str {
        self.mechanism.name()
    }

    /// Strategy the mechanism was selected with.
    pub fn selection_strategy(&self) -> MechanismSelectionStrategy {
        self.selection_strategy
    }

    /// Enable or disable adaptive scaling of the sensitivity estimate.
    pub fn set_adaptive_scaling(&mut self, enabled: bool) {
        self.adaptive_scaling = enabled;
        if !enabled {
            self.scaling_factor = T::one();
        }
    }

    /// Set the L1 and Linf sensitivity bounds.
    pub fn set_sensitivity_bounds(&mut self, l1: T, linf: T) {
        self.l1_sensitivity = l1;
        self.linf_sensitivity = linf;
    }

    /// L-infinity sensitivity bound in force.
    pub fn linf_sensitivity(&self) -> T {
        self.linf_sensitivity
    }

    /// L1 sensitivity bound in force.
    pub fn l1_sensitivity(&self) -> T {
        self.l1_sensitivity
    }

    /// Calibrate and add noise to an array of any dimensionality.
    ///
    /// The dimension is erased with a safe, runtime-checked view rather than
    /// a `transmute`, so arrays of rank 4 and above work as well.
    pub fn calibrate_noise<S, D>(
        &mut self,
        data: &mut ArrayBase<S, D>,
        actual_sensitivity: Option<T>,
    ) -> Result<NoiseCalibrationResult<T>>
    where
        S: DataMut<Elem = T>,
        D: Dimension,
    {
        let sensitivity = actual_sensitivity.unwrap_or(self.l2_sensitivity);
        if !(sensitivity > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "noise calibration requires a positive sensitivity".to_string(),
            ));
        }

        if self.adaptive_scaling {
            let data_scale = self.estimate_data_scale(data)?;
            if sensitivity > T::zero() {
                self.scaling_factor = data_scale / sensitivity;
            }
        }

        let adjusted_sensitivity = sensitivity * self.scaling_factor;
        if !(adjusted_sensitivity > T::zero()) {
            return Err(OptimError::InvalidConfig(
                "adjusted sensitivity must be positive".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();
        {
            let mut view = data.view_mut().into_dyn();
            self.mechanism.add_noise_dyn(
                &mut view,
                adjusted_sensitivity,
                self.target_epsilon,
                self.target_delta,
            )?;
        }
        let calibration_time = start_time.elapsed();

        let noise_scale = match self.mechanism.get_parameters() {
            Some(params) => params.scale,
            None => adjusted_sensitivity / self.target_epsilon,
        };

        Ok(NoiseCalibrationResult {
            mechanism_used: self.mechanism.name().to_string(),
            noise_scale,
            sensitivity_used: adjusted_sensitivity,
            scaling_factor: self.scaling_factor,
            calibration_time_us: calibration_time.as_micros() as u64,
            privacy_parameters: PrivacyParameters {
                epsilon: self.target_epsilon,
                delta: self.target_delta,
            },
        })
    }

    fn estimate_data_scale<S, D>(&self, data: &ArrayBase<S, D>) -> Result<T>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        if data.is_empty() {
            return Err(OptimError::InvalidConfig(
                "cannot estimate the scale of an empty array".to_string(),
            ));
        }
        let sum_squares = data.iter().map(|&x| x * x).sum::<T>();
        let n = T::from(data.len()).ok_or_else(|| {
            OptimError::InvalidConfig("failed to convert array length".to_string())
        })?;
        Ok((sum_squares / n).sqrt())
    }
}

/// Result of noise calibration
#[derive(Debug, Clone)]
pub struct NoiseCalibrationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Mechanism that produced the noise.
    pub mechanism_used: String,
    /// Noise scale actually applied.
    pub noise_scale: T,
    /// Sensitivity used.
    pub sensitivity_used: T,
    /// Scaling factor applied to the sensitivity.
    pub scaling_factor: T,
    /// Wall-clock time of the calibration.
    pub calibration_time_us: u64,
    /// Privacy parameters used.
    pub privacy_parameters: PrivacyParameters<T>,
}

/// Privacy parameters used
#[derive(Debug, Clone)]
pub struct PrivacyParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Epsilon.
    pub epsilon: T,
    /// Delta, when the mechanism supports it.
    pub delta: Option<T>,
}

/// Generate correlated Gaussian noise for matrix operations.
///
/// The correlation is applied with a Cholesky factor `L` of the supplied
/// covariance matrix (`noise = L z`, `z ~ N(0, I)`), which is what actually
/// produces the requested covariance. Multiplying independent noise by the
/// covariance matrix itself -- as an earlier revision did -- yields
/// covariance `C C^T`, not `C`.
pub fn generate_correlated_gaussian_noise<T>(
    shape: (usize, usize),
    correlation_matrix: &Array2<T>,
    scale: T,
    rng: &mut MechanismRng,
) -> Result<Array2<T>>
where
    T: Float + Default + Clone + 'static,
{
    let (rows, cols) = shape;

    if correlation_matrix.nrows() != cols || correlation_matrix.ncols() != cols {
        return Err(OptimError::InvalidConfig(
            "correlation matrix dimensions must match the number of columns".to_string(),
        ));
    }
    let scale_f64 = scale.to_f64().unwrap_or(f64::NAN);
    if !scale_f64.is_finite() || scale_f64 <= 0.0 {
        return Err(OptimError::InvalidConfig(
            "noise scale must be positive and finite".to_string(),
        ));
    }

    // Cholesky decomposition of the (symmetric, positive definite)
    // correlation matrix.
    let mut lower = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..=i {
            let a_ij = correlation_matrix[[i, j]].to_f64().unwrap_or(f64::NAN);
            let a_ji = correlation_matrix[[j, i]].to_f64().unwrap_or(f64::NAN);
            if !a_ij.is_finite() || (a_ij - a_ji).abs() > 1e-9 {
                return Err(OptimError::InvalidConfig(
                    "correlation matrix must be finite and symmetric".to_string(),
                ));
            }

            let mut sum = a_ij;
            for k in 0..j {
                sum -= lower[i * cols + k] * lower[j * cols + k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(OptimError::InvalidConfig(
                        "correlation matrix must be positive definite".to_string(),
                    ));
                }
                lower[i * cols + j] = sum.sqrt();
            } else {
                lower[i * cols + j] = sum / lower[j * cols + j];
            }
        }
    }

    let mut noise = Array2::zeros((rows, cols));
    for i in 0..rows {
        let independent: Vec<f64> = (0..cols).map(|_| standard_normal(rng)).collect();
        for j in 0..cols {
            let mut value = 0.0;
            for k in 0..=j {
                value += lower[j * cols + k] * independent[k];
            }
            noise[[i, j]] = T::from(value * scale_f64).unwrap_or_else(|| T::zero());
        }
    }

    Ok(noise)
}

/// Validate differential privacy parameters
pub fn validate_privacy_parameters<T: Float + Debug + Send + Sync + 'static>(
    epsilon: T,
    delta: Option<T>,
) -> Result<()> {
    if epsilon <= T::zero() {
        return Err(OptimError::InvalidConfig(
            "Epsilon must be positive".to_string(),
        ));
    }

    if let Some(d) = delta {
        if d < T::zero() || d >= T::one() {
            return Err(OptimError::InvalidConfig(
                "Delta must be in [0, 1)".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array3, Array4};

    #[test]
    fn test_gaussian_noise_scale_matches_the_published_formula() {
        // sigma = sqrt(2 ln(1.25 / 1e-5)) = sqrt(2 * 11.7361) = 4.844813...
        let sigma = match GaussianMechanism::<f64>::compute_noise_scale(1.0, 1.0, 1e-5) {
            Ok(value) => value,
            Err(err) => panic!("noise scale computation failed: {err}"),
        };
        assert!(
            (sigma - 4.8448).abs() < 1e-4,
            "expected sigma ~= 4.8448, got {sigma}"
        );

        // Linear in sensitivity, inverse in epsilon.
        let doubled = match GaussianMechanism::<f64>::compute_noise_scale(2.0, 1.0, 1e-5) {
            Ok(value) => value,
            Err(err) => panic!("noise scale computation failed: {err}"),
        };
        assert!((doubled - 2.0 * sigma).abs() < 1e-9);

        let halved_epsilon = match GaussianMechanism::<f64>::compute_noise_scale(1.0, 0.5, 1e-5) {
            Ok(value) => value,
            Err(err) => panic!("noise scale computation failed: {err}"),
        };
        assert!((halved_epsilon - 2.0 * sigma).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_noise_scale_rejects_epsilon_above_one() {
        // The classic bound is only valid for epsilon <= 1.
        assert!(GaussianMechanism::<f64>::compute_noise_scale(1.0, 1.5, 1e-5).is_err());
        assert!(GaussianMechanism::<f64>::compute_noise_scale(1.0, 1.0, 1e-5).is_ok());
        assert!(GaussianMechanism::<f64>::compute_noise_scale(1.0, 0.0, 1e-5).is_err());
        assert!(GaussianMechanism::<f64>::compute_noise_scale(1.0, 0.5, 0.0).is_err());
        assert!(GaussianMechanism::<f64>::compute_noise_scale(0.0, 0.5, 1e-5).is_err());
    }

    #[test]
    fn test_gaussian_mechanism() {
        let mut mechanism = GaussianMechanism::<f64>::new();
        let mut data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        assert!(mechanism
            .add_noise_1d(&mut data, 1.0, 1.0, Some(1e-5))
            .is_ok());
        assert_eq!(mechanism.name(), "Gaussian");
        assert!(mechanism.supports_delta());
        assert!(data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_gaussian_mechanism_requires_delta() {
        let mut mechanism = GaussianMechanism::<f64>::new();
        let mut data = Array1::from_vec(vec![1.0]);
        assert!(mechanism.add_noise_1d(&mut data, 1.0, 1.0, None).is_err());
    }

    #[test]
    fn test_mechanisms_are_not_deterministic_across_instances() {
        // Two independently constructed mechanisms must not produce the same
        // noise: a hardcoded seed makes DP noise exactly reproducible.
        let mut first = GaussianMechanism::<f64>::new();
        let mut second = GaussianMechanism::<f64>::new();
        let mut a = Array1::<f64>::zeros(64);
        let mut b = Array1::<f64>::zeros(64);

        assert!(first.add_noise_1d(&mut a, 1.0, 1.0, Some(1e-5)).is_ok());
        assert!(second.add_noise_1d(&mut b, 1.0, 1.0, Some(1e-5)).is_ok());
        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-12),
            "two Gaussian mechanisms produced identical noise"
        );

        let mut first_laplace = LaplaceMechanism::<f64>::new();
        let mut second_laplace = LaplaceMechanism::<f64>::new();
        let mut c = Array1::<f64>::zeros(64);
        let mut d = Array1::<f64>::zeros(64);
        assert!(first_laplace.add_noise_1d(&mut c, 1.0, 1.0, None).is_ok());
        assert!(second_laplace.add_noise_1d(&mut d, 1.0, 1.0, None).is_ok());
        assert!(
            c.iter().zip(d.iter()).any(|(x, y)| (x - y).abs() > 1e-12),
            "two Laplace mechanisms produced identical noise"
        );
    }

    #[test]
    fn test_seeded_constructors_are_reproducible() {
        let mut first = GaussianMechanism::<f64>::new_with_seed(7);
        let mut second = GaussianMechanism::<f64>::new_with_seed(7);
        let mut a = Array1::<f64>::zeros(16);
        let mut b = Array1::<f64>::zeros(16);
        assert!(first.add_noise_1d(&mut a, 1.0, 1.0, Some(1e-5)).is_ok());
        assert!(second.add_noise_1d(&mut b, 1.0, 1.0, Some(1e-5)).is_ok());
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
    }

    #[test]
    fn test_laplace_mechanism() {
        let mut mechanism = LaplaceMechanism::<f64>::new();
        let mut data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        assert!(mechanism.add_noise_1d(&mut data, 1.0, 1.0, None).is_ok());
        assert_eq!(mechanism.name(), "Laplace");
        assert!(!mechanism.supports_delta());
    }

    #[test]
    fn test_laplace_sampler_has_laplace_statistics() {
        // Mean ~ 0, variance ~ 2 b^2 for Lap(b).
        let mut rng = seeded_rng(11);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let sample = standard_laplace(&mut rng);
            sum += sample;
            sum_sq += sample * sample;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "Laplace mean was {mean}");
        assert!(
            (variance - 2.0).abs() < 0.15,
            "Laplace variance was {variance}, expected ~2"
        );
    }

    #[test]
    fn test_noise_scale_computation() {
        let laplace_scale = match LaplaceMechanism::<f64>::compute_noise_scale(1.0, 1.0) {
            Ok(value) => value,
            Err(err) => panic!("laplace scale failed: {err}"),
        };
        assert_eq!(laplace_scale, 1.0);
        assert!(LaplaceMechanism::<f64>::compute_noise_scale(1.0, 0.0).is_err());
    }

    #[test]
    fn test_get_parameters_reports_what_was_used() {
        let mut mechanism = GaussianMechanism::<f64>::new();
        assert!(
            mechanism.get_parameters().is_none(),
            "an unused mechanism must not report a noise scale"
        );

        let mut data = Array1::from_vec(vec![0.0; 4]);
        assert!(mechanism
            .add_noise_1d(&mut data, 2.0, 0.5, Some(1e-6))
            .is_ok());
        let params = match mechanism.get_parameters() {
            Some(params) => params,
            None => panic!("parameters must be reported after use"),
        };
        assert_eq!(params.mechanism_type, "Gaussian");
        assert_eq!(params.sensitivity, 2.0);
        assert_eq!(params.epsilon, 0.5);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_truncated_mechanism() {
        let base = Box::new(LaplaceMechanism::<f64>::new());
        let mut truncated = match TruncatedNoiseMechanism::new(base, 5.0) {
            Ok(mechanism) => mechanism,
            Err(err) => panic!("construction failed: {err}"),
        };
        let mut data = Array1::from_vec(vec![100.0]);

        assert!(truncated.add_noise_1d(&mut data, 1.0, 0.1, None).is_ok());
        assert!(data[0].abs() <= 5.0);
        assert!(
            TruncatedNoiseMechanism::new(Box::new(LaplaceMechanism::<f64>::new()), 0.0).is_err()
        );
    }

    #[test]
    fn test_noise_added_to_arrays_of_every_rank() {
        // The dimension-erased path replaces an unsound transmute and must
        // handle ranks beyond 3.
        let mut mechanism = GaussianMechanism::<f64>::new();

        let mut a1 = Array1::<f64>::zeros(3);
        assert!(mechanism
            .add_noise_1d(&mut a1, 1.0, 1.0, Some(1e-5))
            .is_ok());

        let mut a2 = Array2::<f64>::zeros((2, 3));
        assert!(mechanism
            .add_noise_2d(&mut a2, 1.0, 1.0, Some(1e-5))
            .is_ok());

        let mut a3 = Array3::<f64>::zeros((2, 2, 2));
        assert!(mechanism
            .add_noise_3d(&mut a3, 1.0, 1.0, Some(1e-5))
            .is_ok());

        let mut a4 = Array4::<f64>::zeros((2, 2, 2, 2));
        assert!(add_noise_to(&mut mechanism, &mut a4, 1.0, 1.0, Some(1e-5)).is_ok());
        assert!(a4.iter().all(|value| value.is_finite()));
        assert!(a4.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn test_exponential_mechanism() {
        let quality_fn = Box::new(|x: &f64| -*x); // Prefer smaller values
        let mut mechanism = ExponentialMechanism::new(quality_fn);

        let candidates = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let selected = match mechanism.select_output(&candidates, 1.0, 1.0) {
            Ok(value) => value,
            Err(err) => panic!("selection failed: {err}"),
        };
        assert!(candidates.contains(&selected));
    }

    #[test]
    fn test_exponential_mechanism_prefers_high_quality_candidates() {
        let quality_fn = Box::new(|x: &f64| -*x);
        let mut mechanism = ExponentialMechanism::new_with_seed(quality_fn, 3);
        let candidates = vec![1.0, 100.0];

        let mut best_selected = 0;
        for _ in 0..200 {
            match mechanism.select_output(&candidates, 1.0, 10.0) {
                Ok(value) if value == 1.0 => best_selected += 1,
                Ok(_) => {}
                Err(err) => panic!("selection failed: {err}"),
            }
        }
        assert!(
            best_selected > 150,
            "the high-quality candidate should dominate, selected {best_selected}/200"
        );
    }

    #[test]
    fn test_exponential_mechanism_rejects_degenerate_input() {
        let quality_fn = Box::new(|_: &f64| f64::NAN);
        let mut mechanism = ExponentialMechanism::new(quality_fn);
        assert!(mechanism.select_output(&[1.0, 2.0], 1.0, 1.0).is_err());

        let quality_fn = Box::new(|x: &f64| -*x);
        let mut mechanism = ExponentialMechanism::new(quality_fn);
        assert!(mechanism.select_output(&[], 1.0, 1.0).is_err());
        assert!(mechanism.select_output(&[1.0], 0.0, 1.0).is_err());
        assert!(mechanism.select_output(&[1.0], 1.0, 0.0).is_err());
    }

    #[test]
    fn test_noise_calibrator_holds_one_mechanism() {
        let calibrator = NoiseCalibrator::<f64>::new(
            1.0,
            Some(1e-5),
            1.0,
            MechanismSelectionStrategy::PrivacyOptimal,
        );
        assert_eq!(calibrator.mechanism_name(), "Gaussian");

        let laplace =
            NoiseCalibrator::<f64>::new(1.0, None, 1.0, MechanismSelectionStrategy::PrivacyOptimal);
        assert_eq!(laplace.mechanism_name(), "Laplace");
    }

    #[test]
    fn test_noise_calibrator_handles_high_rank_arrays() {
        let mut calibrator = NoiseCalibrator::<f64>::new(
            1.0,
            Some(1e-5),
            1.0,
            MechanismSelectionStrategy::AlwaysGaussian,
        );

        let mut data = Array4::<f64>::zeros((2, 2, 2, 2));
        let result = match calibrator.calibrate_noise(&mut data, None) {
            Ok(result) => result,
            Err(err) => panic!("calibration failed: {err}"),
        };
        assert_eq!(result.mechanism_used, "Gaussian");
        assert!(result.noise_scale > 0.0);
        assert!(data.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn test_tree_aggregation_requires_positive_height() {
        let base = Box::new(LaplaceMechanism::<f64>::new());
        assert!(TreeAggregationMechanism::new(0, base).is_err());

        let base = Box::new(LaplaceMechanism::<f64>::new());
        let mut tree = match TreeAggregationMechanism::new(3, base) {
            Ok(tree) => tree,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert_eq!(tree.tree_height(), 3);

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let aggregated = match tree.aggregate_with_tree(&values, 1.0, 1.0, None) {
            Ok(value) => value,
            Err(err) => panic!("aggregation failed: {err}"),
        };
        assert!(aggregated.is_finite());
        assert!(tree.aggregate_with_tree(&values, 1.0, 0.0, None).is_err());
    }

    #[test]
    fn test_sparse_vector_threshold_is_noised_once() {
        let mut svt =
            match SparseVectorMechanism::<f64>::new_with_seed(5.0, 1.0, 0.5, 0.5, Some(0.5), 3, 17)
            {
                Ok(svt) => svt,
                Err(err) => panic!("construction failed: {err}"),
            };

        let first_threshold = svt.noisy_threshold;
        assert!(svt.above_threshold(1000.0).is_ok());
        assert!(svt.above_threshold(1000.0).is_ok());
        assert_eq!(
            svt.noisy_threshold, first_threshold,
            "the threshold must be perturbed exactly once, at construction"
        );
    }

    #[test]
    fn test_sparse_vector_below_threshold_returns_none_and_costs_nothing() {
        let mut svt = match SparseVectorMechanism::<f64>::new_with_seed(
            1.0e6,
            1.0,
            1.0,
            1.0,
            Some(1.0),
            2,
            23,
        ) {
            Ok(svt) => svt,
            Err(err) => panic!("construction failed: {err}"),
        };

        for _ in 0..50 {
            match svt.answer_query(-1.0e6) {
                Ok(None) => {}
                other => panic!("below-threshold query must return Ok(None), got {other:?}"),
            }
        }
        assert_eq!(
            svt.queries_answered(),
            0,
            "below-threshold queries must not consume the answer budget"
        );
        assert!(!svt.is_halted());
    }

    #[test]
    fn test_sparse_vector_halts_after_max_answers() {
        let mut svt = match SparseVectorMechanism::<f64>::new_with_seed(
            -1.0e6,
            1.0,
            1.0,
            1.0,
            Some(1.0),
            2,
            29,
        ) {
            Ok(svt) => svt,
            Err(err) => panic!("construction failed: {err}"),
        };

        assert!(svt.answer_query(1.0e6).is_ok());
        assert!(svt.answer_query(1.0e6).is_ok());
        assert!(svt.is_halted());
        match svt.answer_query(1.0e6) {
            Err(OptimError::PrivacyBudgetExhausted { .. }) => {}
            other => panic!("a halted SVT must report exhaustion, got {other:?}"),
        }
    }

    #[test]
    fn test_sparse_vector_validates_its_parameters() {
        assert!(
            SparseVectorMechanism::<f64>::new(1.0, 0.0, 1.0, 1.0, None, 1).is_err(),
            "sensitivity must be positive"
        );
        assert!(SparseVectorMechanism::<f64>::new(1.0, 1.0, 0.0, 1.0, None, 1).is_err());
        assert!(SparseVectorMechanism::<f64>::new(1.0, 1.0, 1.0, 0.0, None, 1).is_err());
        assert!(SparseVectorMechanism::<f64>::new(1.0, 1.0, 1.0, 1.0, None, 0).is_err());

        let mut indicator_only =
            match SparseVectorMechanism::<f64>::new_with_seed(0.0, 1.0, 1.0, 1.0, None, 1, 5) {
                Ok(svt) => svt,
                Err(err) => panic!("construction failed: {err}"),
            };
        // Without a value budget, releasing a value must be refused.
        assert!(indicator_only.answer_query(1.0e6).is_err());
    }

    #[test]
    fn test_correlated_noise_uses_a_cholesky_factor() {
        let mut rng = seeded_rng(41);
        let correlation = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0])
            .unwrap_or_else(|_| Array2::eye(2));
        let noise = match generate_correlated_gaussian_noise((5000, 2), &correlation, 1.0, &mut rng)
        {
            Ok(noise) => noise,
            Err(err) => panic!("noise generation failed: {err}"),
        };
        assert_eq!(noise.dim(), (5000, 2));

        // Empirical correlation should be near 0.5.
        let col0: Vec<f64> = noise.column(0).to_vec();
        let col1: Vec<f64> = noise.column(1).to_vec();
        let n = col0.len() as f64;
        let mean0 = col0.iter().sum::<f64>() / n;
        let mean1 = col1.iter().sum::<f64>() / n;
        let cov = col0
            .iter()
            .zip(col1.iter())
            .map(|(a, b)| (a - mean0) * (b - mean1))
            .sum::<f64>()
            / n;
        let var0 = col0.iter().map(|a| (a - mean0).powi(2)).sum::<f64>() / n;
        let var1 = col1.iter().map(|b| (b - mean1).powi(2)).sum::<f64>() / n;
        let corr = cov / (var0.sqrt() * var1.sqrt());
        assert!(
            (corr - 0.5).abs() < 0.08,
            "empirical correlation was {corr}, expected ~0.5"
        );
    }

    #[test]
    fn test_correlated_noise_rejects_invalid_matrices() {
        let mut rng = seeded_rng(43);
        let asymmetric = Array2::from_shape_vec((2, 2), vec![1.0, 0.9, 0.1, 1.0])
            .unwrap_or_else(|_| Array2::eye(2));
        assert!(generate_correlated_gaussian_noise((4, 2), &asymmetric, 1.0, &mut rng).is_err());

        let indefinite = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0])
            .unwrap_or_else(|_| Array2::eye(2));
        assert!(generate_correlated_gaussian_noise((4, 2), &indefinite, 1.0, &mut rng).is_err());

        let identity = Array2::<f64>::eye(2);
        assert!(generate_correlated_gaussian_noise((4, 2), &identity, 0.0, &mut rng).is_err());
    }

    #[test]
    fn test_smooth_sensitivity_mechanism_validates_beta() {
        let sensitivity_fn: SensitivityFn<f64> =
            Box::new(|data: &[f64]| data.iter().cloned().fold(0.0f64, |acc, x| acc.max(x.abs())));
        assert!(SmoothSensitivityMechanism::new(0.0, sensitivity_fn).is_err());

        let sensitivity_fn: SensitivityFn<f64> =
            Box::new(|data: &[f64]| data.iter().cloned().fold(0.0f64, |acc, x| acc.max(x.abs())));
        let mechanism = match SmoothSensitivityMechanism::new(0.1, sensitivity_fn) {
            Ok(mechanism) => mechanism,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert_eq!(mechanism.beta(), 0.1);
        assert_eq!(mechanism.local_sensitivity(&[1.0, -3.0, 2.0]), 3.0);
    }

    #[test]
    fn test_privacy_parameter_validation() {
        assert!(validate_privacy_parameters(1.0, Some(1e-5)).is_ok());
        assert!(validate_privacy_parameters(-1.0, Some(1e-5)).is_err());
        assert!(validate_privacy_parameters(1.0, Some(1.5)).is_err());
    }
}
