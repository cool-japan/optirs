// Bayesian optimization for architecture search

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::RngExt;
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::random::RandomSearch;
use super::{SearchStrategy, SearchStrategyStatistics};

/// Number of random architectures drawn before the surrogate model is trusted.
const INITIAL_RANDOM_TRIALS: usize = 5;

/// Number of candidate architectures scored by the acquisition function on each
/// `generate_architecture` call.
const ACQUISITION_CANDIDATES: usize = 100;

/// Bayesian optimization for architecture search
pub struct BayesianOptimization<T: Float + Debug + Send + Sync + 'static + std::iter::Sum> {
    gaussian_process: GaussianProcess<T>,
    acquisition_function: AcquisitionFunction<T>,
    observed_architectures: Vec<OptimizerArchitecture<T>>,
    observed_performances: Vec<T>,
    statistics: SearchStrategyStatistics<T>,
    exploration_factor: T,
    /// Persistent candidate sampler.
    ///
    /// Previously a *fresh* `RandomSearch::new(Some(42))` was constructed on
    /// every call, so all 100 "candidates" were byte-identical and the
    /// acquisition function had nothing to choose between. One long-lived
    /// sampler keeps the candidate pool genuinely diverse.
    candidate_sampler: RandomSearch<T>,
}

/// Gaussian Process regressor.
///
/// Maintains the Cholesky factor `L` of `K(X, X) + sigma^2 I` together with
/// `alpha = (K + sigma^2 I)^-1 (y - mean(y))`, which is all that is needed for
/// the exact GP posterior:
///
/// * mean:     `mu(x)     = mean(y) + k(x, X) . alpha`
/// * variance: `sigma2(x) = k(x, x) - v . v`, where `L v = k(x, X)`
#[derive(Debug)]
pub struct GaussianProcess<T: Float + Debug + Send + Sync + 'static> {
    /// Kernel used to build the covariance matrix.
    kernel: GPKernel<T>,
    /// Observation noise variance added to the diagonal.
    noise_variance: T,
    /// Training inputs, retained to evaluate `k(x, X)` at prediction time.
    train_inputs: Vec<Array1<T>>,
    /// Lower-triangular Cholesky factor of `K + sigma^2 I`.
    cholesky: Array2<T>,
    /// Pre-solved weight vector `(K + sigma^2 I)^-1 (y - mean(y))`.
    alpha: Array1<T>,
    /// Mean of the training targets; the GP is fitted on the centred residual.
    target_mean: T,
    /// Whether `fit` has completed successfully at least once.
    fitted: bool,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug)]
pub struct AcquisitionFunction<T: Float + Debug + Send + Sync + 'static> {
    function_type: AcquisitionType,
    explorationweight: T,
    current_best: T,
    /// Persistent RNG used by Thompson sampling. A fresh `Random::default()`
    /// per evaluation (the old behaviour) made the "sample" a fixed offset.
    rng: Random<scirs2_core::random::rngs::StdRng>,
}

/// Types of acquisition functions
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionType {
    /// Expected Improvement
    EI,
    /// Upper Confidence Bound
    UCB,
    /// Probability of Improvement
    PI,
    /// Thompson Sampling
    Thompson,
    /// Information Gain
    InfoGain,
}

/// Gaussian Process kernel.
///
/// `hyperparameters` holds `[length_scale, signal_variance]`.
#[derive(Debug, Clone)]
pub struct GPKernel<T: Float + Debug + Send + Sync + 'static> {
    kerneltype: KernelType,
    hyperparameters: Array1<T>,
}

/// Kernel types for GP
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    RBF,
    Matern32,
    Matern52,
    Linear,
    Polynomial,
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > BayesianOptimization<T>
{
    /// Create a Bayesian optimizer seeded from OS entropy.
    pub fn new(
        kerneltype: KernelType,
        acquisition_type: AcquisitionType,
        exploration_factor: f64,
    ) -> Self {
        Self::build(kerneltype, acquisition_type, exploration_factor, None)
    }

    /// Create a fully reproducible Bayesian optimizer.
    ///
    /// Both the candidate sampler and the Thompson-sampling RNG are derived
    /// from `seed`, so an identical sequence of observations produces an
    /// identical sequence of suggestions.
    pub fn new_with_seed(
        kerneltype: KernelType,
        acquisition_type: AcquisitionType,
        exploration_factor: f64,
        seed: u64,
    ) -> Self {
        Self::build(kerneltype, acquisition_type, exploration_factor, Some(seed))
    }

    fn build(
        kerneltype: KernelType,
        acquisition_type: AcquisitionType,
        exploration_factor: f64,
        seed: Option<u64>,
    ) -> Self {
        let exploration: T =
            scirs2_core::numeric::NumCast::from(exploration_factor).unwrap_or_else(|| T::zero());

        Self {
            gaussian_process: GaussianProcess::new(kerneltype),
            acquisition_function: AcquisitionFunction::new(
                acquisition_type,
                exploration,
                // Offset the acquisition RNG so it does not replay the exact
                // stream the candidate sampler consumes.
                seed.map(|s| s.wrapping_add(0x9e37_79b9_7f4a_7c15)),
            ),
            observed_architectures: Vec::new(),
            observed_performances: Vec::new(),
            statistics: SearchStrategyStatistics::default(),
            exploration_factor: exploration,
            candidate_sampler: RandomSearch::<T>::new(seed),
        }
    }

    /// Kernel currently used by the surrogate model.
    pub fn kernel(&self) -> &GPKernel<T> {
        self.gaussian_process.kernel()
    }

    /// Immutable view of the surrogate Gaussian process.
    pub fn gaussian_process(&self) -> &GaussianProcess<T> {
        &self.gaussian_process
    }

    /// Best objective value observed so far (the acquisition incumbent).
    pub fn current_best(&self) -> T {
        self.acquisition_function.current_best
    }

    fn encode_architecture(&self, architecture: &OptimizerArchitecture<T>) -> Array1<T> {
        // Deterministic, vocabulary-based encoding of the architecture's
        // component types, followed by its hyperparameter values.  Each
        // component string (produced elsewhere via
        // `format!("{:?}", ComponentType)`) is mapped onto a fixed, ordered
        // vocabulary and accumulated into a multi-hot block.  Unlike a raw
        // string hash this preserves locality and gives the Gaussian-process
        // kernel a meaningful, stable feature, so similar architectures map to
        // nearby points in the input space.
        let mut encoding = encode_component_block::<T, _>(architecture.components.iter());

        // Encode hyperparameters from the architecture's hyperparameters map.
        // Sorted by key so the appended values are order-stable regardless of
        // the underlying hash-map iteration order.
        let mut hyperparameters: Vec<(&String, &T)> = architecture.hyperparameters.iter().collect();
        hyperparameters.sort_by(|a, b| a.0.cmp(b.0));
        for (_key, value) in hyperparameters {
            encoding.push(*value);
        }

        // Pad (or truncate) to the fixed GP input dimension.  This preserves
        // the original 64-length contract consumed by the kernel/GP.
        encoding.resize(64, T::zero());
        Array1::from_vec(encoding)
    }

    fn fit_gp(&mut self) -> Result<()> {
        if self.observed_architectures.len() < 2 {
            return Ok(());
        }

        // Encode all observed architectures
        let encoded_archs: Vec<Array1<T>> = self
            .observed_architectures
            .iter()
            .map(|arch| self.encode_architecture(arch))
            .collect();

        // Fit Gaussian Process
        self.gaussian_process
            .fit(&encoded_archs, &self.observed_performances)?;

        Ok(())
    }

    fn suggest_next_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        if searchspace.components.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "Bayesian optimization requires a non-empty SearchSpaceConfig::components"
                    .to_string(),
            ));
        }

        // Cold start: the surrogate has too little data to be informative, so
        // draw from the persistent sampler directly.
        if self.observed_architectures.len() < INITIAL_RANDOM_TRIALS
            || !self.gaussian_process.is_fitted()
        {
            return self
                .candidate_sampler
                .generate_architecture(searchspace, &VecDeque::new());
        }

        // Generate a diverse candidate pool from the persistent sampler.
        let mut candidates = Vec::with_capacity(ACQUISITION_CANDIDATES);
        for _ in 0..ACQUISITION_CANDIDATES {
            candidates.push(
                self.candidate_sampler
                    .generate_architecture(searchspace, &VecDeque::new())?,
            );
        }

        let mut best_architecture: Option<OptimizerArchitecture<T>> = None;
        let mut best_acquisition = T::neg_infinity();

        for candidate in candidates {
            let encoded = self.encode_architecture(&candidate);
            let (mean, variance) = self.gaussian_process.predict(&encoded)?;
            let acquisition_value = self.acquisition_function.evaluate(mean, variance);

            if best_architecture.is_none() || acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_architecture = Some(candidate);
            }
        }

        match best_architecture {
            Some(architecture) => Ok(architecture),
            // Unreachable while ACQUISITION_CANDIDATES > 0, but reported
            // honestly rather than indexed into an empty vector.
            None => Err(OptimError::OptimizationError(
                "acquisition maximisation produced no candidate".to_string(),
            )),
        }
    }
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > SearchStrategy<T> for BayesianOptimization<T>
{
    fn initialize(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        if searchspace.components.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "Bayesian optimization requires a non-empty SearchSpaceConfig::components"
                    .to_string(),
            ));
        }
        self.observed_architectures.clear();
        self.observed_performances.clear();
        self.candidate_sampler.initialize(searchspace)?;
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        let architecture = self.suggest_next_architecture(searchspace)?;
        self.statistics.total_architectures_generated += 1;
        Ok(architecture)
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        let mut observed_any = false;

        for result in results {
            if let Some(&performance) = result
                .evaluation_results
                .metric_scores
                .get(&EvaluationMetric::FinalPerformance)
            {
                self.observed_architectures
                    .push(result.architecture.clone());
                self.observed_performances.push(performance);
                observed_any = true;

                // Update statistics
                if performance > self.statistics.best_performance {
                    self.statistics.best_performance = performance;
                }

                // Track the acquisition incumbent. Without this, `current_best`
                // stayed at its initial value forever and every
                // improvement-based acquisition degenerated.
                self.acquisition_function.observe(performance);
            }
        }

        // Refit the surrogate once, after all new observations are recorded,
        // rather than once per result.
        if observed_any {
            self.fit_gp()?;
        }

        // Update average performance
        if !self.observed_performances.is_empty() {
            let sum: T = self.observed_performances.iter().cloned().sum();
            let count: T = scirs2_core::numeric::NumCast::from(self.observed_performances.len())
                .unwrap_or_else(|| T::one());
            self.statistics.average_performance = sum / count;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "BayesianOptimization"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate = self.exploration_factor;
        stats.exploitation_rate = T::one() - self.exploration_factor;
        stats
    }
}

// ---------------------------------------------------------------------------
// Numerical helpers
// ---------------------------------------------------------------------------

/// Convert an `f64` constant into `T`, falling back to zero.
fn scalar<T: Float>(value: f64) -> T {
    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(T::zero)
}

/// Gauss error function.
///
/// Abramowitz & Stegun formula 7.1.26 — a rational approximation with a
/// maximum absolute error of `1.5e-7` over the whole real line, which is well
/// below the resolution at which acquisition values are compared.
fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Standard normal cumulative distribution function.
pub fn norm_cdf<T: Float>(z: T) -> T {
    let z = z.to_f64().unwrap_or(0.0);
    scalar(0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2)))
}

/// Standard normal probability density function.
pub fn norm_pdf<T: Float>(z: T) -> T {
    let z = z.to_f64().unwrap_or(0.0);
    scalar((-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt())
}

/// Numerical floor below which a standard deviation is treated as zero.
fn std_dev_floor<T: Float>() -> T {
    scalar(1e-12)
}

/// In-place Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor `L` with `A = L L^T`, or `None` when a
/// non-positive pivot is encountered (i.e. `A` is not positive definite at the
/// supplied jitter level).
fn cholesky<T: Float + Debug + Send + Sync + 'static>(a: &Array2<T>) -> Option<Array2<T>> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n {
        return None;
    }

    let mut l: Array2<T> = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }

            if i == j {
                // Reject non-positive and non-finite pivots alike: the matrix is
                // not positive definite, so there is no Cholesky factor.
                if sum <= T::zero() || !sum.is_finite() {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                let pivot = l[[j, j]];
                if pivot <= T::zero() || pivot.is_nan() {
                    return None;
                }
                l[[i, j]] = sum / pivot;
            }
        }
    }

    Some(l)
}

/// Solve `L y = b` for a lower-triangular `L` (forward substitution).
fn forward_substitution<T: Float + Debug + Send + Sync + 'static>(
    l: &Array2<T>,
    b: &Array1<T>,
) -> Option<Array1<T>> {
    let n = l.nrows();
    if b.len() != n {
        return None;
    }

    let mut y: Array1<T> = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum = sum - l[[i, k]] * y[k];
        }
        let pivot = l[[i, i]];
        if pivot.abs() <= T::zero() || pivot.is_nan() {
            return None;
        }
        y[i] = sum / pivot;
    }
    Some(y)
}

/// Solve `L^T x = y` for a lower-triangular `L` (back substitution).
fn back_substitution<T: Float + Debug + Send + Sync + 'static>(
    l: &Array2<T>,
    y: &Array1<T>,
) -> Option<Array1<T>> {
    let n = l.nrows();
    if y.len() != n {
        return None;
    }

    let mut x: Array1<T> = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum = sum - l[[k, i]] * x[k];
        }
        let pivot = l[[i, i]];
        if pivot.abs() <= T::zero() || pivot.is_nan() {
            return None;
        }
        x[i] = sum / pivot;
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Gaussian process
// ---------------------------------------------------------------------------

/// Jitter levels tried, in order, when the covariance matrix is numerically
/// indefinite. Duplicate or near-duplicate architecture encodings make `K`
/// singular, which is common in a discrete search space.
const CHOLESKY_JITTER_LADDER: [f64; 5] = [0.0, 1e-10, 1e-8, 1e-6, 1e-4];

impl<T: Float + Debug + Default + Send + Sync + 'static> GaussianProcess<T> {
    fn new(kerneltype: KernelType) -> Self {
        Self {
            kernel: GPKernel::new(kerneltype),
            noise_variance: scalar(1e-6),
            train_inputs: Vec::new(),
            cholesky: Array2::zeros((0, 0)),
            alpha: Array1::zeros(0),
            target_mean: T::zero(),
            fitted: false,
        }
    }

    /// Kernel backing this process.
    pub fn kernel(&self) -> &GPKernel<T> {
        &self.kernel
    }

    /// Whether the process has been successfully fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Observation-noise variance added to the covariance diagonal.
    pub fn noise_variance(&self) -> T {
        self.noise_variance
    }

    /// Set the observation-noise variance (must be non-negative).
    pub fn set_noise_variance(&mut self, noise_variance: T) {
        if noise_variance >= T::zero() {
            self.noise_variance = noise_variance;
        }
    }

    /// Fit the exact GP posterior to `(x, y)`.
    ///
    /// Builds `K = k(X, X) + sigma^2 I`, Cholesky-factorises it (escalating the
    /// diagonal jitter if the matrix is numerically indefinite) and pre-solves
    /// `alpha = K^-1 (y - mean(y))`.
    pub fn fit(&mut self, x: &[Array1<T>], y: &[T]) -> Result<()> {
        if x.len() != y.len() {
            return Err(OptimError::InvalidParameter(format!(
                "GP fit requires matching input/target counts, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            self.fitted = false;
            return Ok(());
        }

        let n = x.len();
        let count: T = scalar(n as f64);
        let sum = y.iter().fold(T::zero(), |acc, &v| acc + v);
        let target_mean = sum / count;

        // Covariance matrix of the training inputs.
        let mut base: Array2<T> = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let value = self.kernel.evaluate(&x[i], &x[j]);
                base[[i, j]] = value;
                base[[j, i]] = value;
            }
        }

        let centred: Array1<T> = Array1::from_iter(y.iter().map(|&v| v - target_mean));

        for jitter in CHOLESKY_JITTER_LADDER {
            let mut k = base.clone();
            let diagonal = self.noise_variance + scalar::<T>(jitter);
            for i in 0..n {
                k[[i, i]] = k[[i, i]] + diagonal;
            }

            let Some(l) = cholesky(&k) else {
                continue;
            };
            let Some(intermediate) = forward_substitution(&l, &centred) else {
                continue;
            };
            let Some(alpha) = back_substitution(&l, &intermediate) else {
                continue;
            };

            self.cholesky = l;
            self.alpha = alpha;
            self.train_inputs = x.to_vec();
            self.target_mean = target_mean;
            self.fitted = true;
            return Ok(());
        }

        self.fitted = false;
        Err(OptimError::OptimizationError(
            "Gaussian-process covariance matrix is not positive definite even with \
             the maximum diagonal jitter; the observed architectures may be degenerate"
                .to_string(),
        ))
    }

    /// Posterior mean and variance at `x`.
    ///
    /// Before any successful `fit` the process reports its prior: mean zero and
    /// the kernel's own signal variance, which is an honest statement of total
    /// ignorance rather than a fabricated point estimate.
    pub fn predict(&self, x: &Array1<T>) -> Result<(T, T)> {
        if !self.fitted || self.train_inputs.is_empty() {
            return Ok((T::zero(), self.kernel.signal_variance()));
        }

        let n = self.train_inputs.len();
        let mut k_star: Array1<T> = Array1::zeros(n);
        for (i, input) in self.train_inputs.iter().enumerate() {
            k_star[i] = self.kernel.evaluate(x, input);
        }

        // Posterior mean.
        let mut mean = self.target_mean;
        for i in 0..n {
            mean = mean + k_star[i] * self.alpha[i];
        }

        // Posterior variance: k(x, x) - v.v with L v = k_*.
        let prior_variance = self.kernel.evaluate(x, x);
        let variance = match forward_substitution(&self.cholesky, &k_star) {
            Some(v) => {
                let reduction = v.iter().fold(T::zero(), |acc, &vi| acc + vi * vi);
                (prior_variance - reduction).max(T::zero())
            }
            // A failed solve means the stored factor is unusable; fall back to
            // the prior variance rather than reporting a fake certainty.
            None => prior_variance,
        };

        Ok((mean, variance))
    }
}

// ---------------------------------------------------------------------------
// Acquisition functions
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + Send + Sync + 'static> AcquisitionFunction<T> {
    fn new(function_type: AcquisitionType, explorationweight: T, seed: Option<u64>) -> Self {
        Self {
            function_type,
            explorationweight,
            current_best: T::neg_infinity(),
            rng: Random::seed(seed.unwrap_or_else(scirs2_core::random::random::<u64>)),
        }
    }

    /// Incumbent used by the improvement-based acquisitions.
    fn incumbent(&self) -> T {
        if self.current_best.is_finite() {
            self.current_best
        } else {
            T::zero()
        }
    }

    /// Score a candidate from its posterior mean and variance.
    ///
    /// All formulations assume **maximization** of the objective.
    fn evaluate(&mut self, mean: T, variance: T) -> T {
        let variance = variance.max(T::zero());
        let std_dev = variance.sqrt();
        let floor = std_dev_floor::<T>();
        let best = self.incumbent();

        match self.function_type {
            // Upper confidence bound: mu + kappa * sigma (standard deviation,
            // not variance — the previous code used the variance directly).
            AcquisitionType::UCB => mean + self.explorationweight * std_dev,

            // Expected improvement:
            //   EI = (mu - f*) Phi(z) + sigma phi(z),  z = (mu - f*)/sigma
            // With no uncertainty the improvement is deterministic.
            AcquisitionType::EI => {
                let improvement = mean - best;
                if std_dev <= floor {
                    improvement.max(T::zero())
                } else {
                    let z = improvement / std_dev;
                    improvement * norm_cdf(z) + std_dev * norm_pdf(z)
                }
            }

            // Probability of improvement: Phi(z).
            AcquisitionType::PI => {
                let improvement = mean - best;
                if std_dev <= floor {
                    if improvement > T::zero() {
                        T::one()
                    } else {
                        T::zero()
                    }
                } else {
                    norm_cdf(improvement / std_dev)
                }
            }

            // Thompson sampling: draw once from N(mu, sigma^2) via Box-Muller.
            AcquisitionType::Thompson => {
                if std_dev <= floor {
                    mean
                } else {
                    let u1 = self.rng.random::<f64>().clamp(1e-12, 1.0 - 1e-12);
                    let u2 = self.rng.random::<f64>();
                    let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    mean + std_dev * scalar::<T>(normal)
                }
            }

            // Differential entropy of the posterior Gaussian:
            //   H = 0.5 ln(2 pi e sigma^2)
            // Maximising it selects the least-known candidate.
            AcquisitionType::InfoGain => {
                if variance <= floor {
                    T::neg_infinity()
                } else {
                    let two_pi_e = scalar::<T>(2.0 * std::f64::consts::PI * std::f64::consts::E);
                    scalar::<T>(0.5) * (two_pi_e * variance).ln()
                }
            }
        }
    }

    /// Raise the incumbent to `value` when it is an improvement.
    fn observe(&mut self, value: T) {
        if !self.current_best.is_finite() || value > self.current_best {
            self.current_best = value;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Degree used by the polynomial kernel.
const POLYNOMIAL_DEGREE: i32 = 2;

impl<T: Float + Debug + Default + Send + Sync + 'static> GPKernel<T> {
    fn new(kerneltype: KernelType) -> Self {
        Self {
            kerneltype,
            // [length_scale, signal_variance]
            hyperparameters: Array1::ones(2),
        }
    }

    /// Kernel family.
    pub fn kernel_type(&self) -> KernelType {
        self.kerneltype
    }

    /// Characteristic length scale (strictly positive).
    pub fn length_scale(&self) -> T {
        let value = self.hyperparameters.first().copied().unwrap_or_else(T::one);
        if value > T::zero() {
            value
        } else {
            T::one()
        }
    }

    /// Signal variance (strictly positive).
    pub fn signal_variance(&self) -> T {
        let value = self.hyperparameters.get(1).copied().unwrap_or_else(T::one);
        if value > T::zero() {
            value
        } else {
            T::one()
        }
    }

    /// Set the kernel hyperparameters. Non-positive values are ignored.
    pub fn set_hyperparameters(&mut self, length_scale: T, signal_variance: T) {
        if length_scale > T::zero() {
            self.hyperparameters[0] = length_scale;
        }
        if signal_variance > T::zero() {
            self.hyperparameters[1] = signal_variance;
        }
    }

    /// Evaluate `k(a, b)`.
    ///
    /// Every [`KernelType`] variant has a real implementation:
    ///
    /// * `RBF`        `s^2 exp(-r^2 / (2 l^2))`
    /// * `Matern32`   `s^2 (1 + sqrt(3) r / l) exp(-sqrt(3) r / l)`
    /// * `Matern52`   `s^2 (1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r / l)`
    /// * `Linear`     `s^2 (a . b) / l^2`
    /// * `Polynomial` `s^2 ((a . b) / l^2 + 1)^d`, `d = 2`
    ///
    /// where `r = ||a - b||`.
    pub fn evaluate(&self, a: &Array1<T>, b: &Array1<T>) -> T {
        let length_scale = self.length_scale();
        let signal_variance = self.signal_variance();
        let n = a.len().min(b.len());

        match self.kerneltype {
            KernelType::RBF => {
                let mut sq_dist = T::zero();
                for i in 0..n {
                    let d = a[i] - b[i];
                    sq_dist = sq_dist + d * d;
                }
                let two = scalar::<T>(2.0);
                signal_variance * (-sq_dist / (two * length_scale * length_scale)).exp()
            }

            KernelType::Matern32 => {
                let r = euclidean_distance(a, b, n);
                let sqrt3 = scalar::<T>(3.0_f64.sqrt());
                let scaled = sqrt3 * r / length_scale;
                signal_variance * (T::one() + scaled) * (-scaled).exp()
            }

            KernelType::Matern52 => {
                let r = euclidean_distance(a, b, n);
                let sqrt5 = scalar::<T>(5.0_f64.sqrt());
                let scaled = sqrt5 * r / length_scale;
                let quadratic =
                    scalar::<T>(5.0) * r * r / (scalar::<T>(3.0) * length_scale * length_scale);
                signal_variance * (T::one() + scaled + quadratic) * (-scaled).exp()
            }

            KernelType::Linear => {
                let mut dot = T::zero();
                for i in 0..n {
                    dot = dot + a[i] * b[i];
                }
                signal_variance * dot / (length_scale * length_scale)
            }

            KernelType::Polynomial => {
                let mut dot = T::zero();
                for i in 0..n {
                    dot = dot + a[i] * b[i];
                }
                let base = dot / (length_scale * length_scale) + T::one();
                signal_variance * base.powi(POLYNOMIAL_DEGREE)
            }
        }
    }
}

/// Euclidean distance between the first `n` entries of two vectors.
fn euclidean_distance<T: Float + Debug + Send + Sync + 'static>(
    a: &Array1<T>,
    b: &Array1<T>,
    n: usize,
) -> T {
    let mut sq = T::zero();
    for i in 0..n {
        let d = a[i] - b[i];
        sq = sq + d * d;
    }
    sq.sqrt()
}

/// Ordered vocabulary of known optimizer-component type names.
///
/// These correspond exactly to the field-less variants of
/// [`crate::architecture::ComponentType`], whose `Debug` representation is the
/// variant name and is what populates `OptimizerArchitecture::components`
/// across every search strategy.  The order is fixed (matching the enum's
/// declaration order) so the produced encoding is deterministic and stable
/// across runs and builds.  A trailing out-of-vocabulary slot (added by
/// [`encode_component_block`]) absorbs any unrecognised name.
const COMPONENT_VOCABULARY: [&str; 41] = [
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "AdaGrad",
    "AdaDelta",
    "Momentum",
    "Nesterov",
    "LRScheduler",
    "GradientClipping",
    "BatchNorm",
    "Dropout",
    "LAMB",
    "LARS",
    "Lion",
    "RAdam",
    "Lookahead",
    "SAM",
    "LBFGS",
    "SparseAdam",
    "GroupedAdam",
    "MAML",
    "L1Regularizer",
    "L2Regularizer",
    "ElasticNetRegularizer",
    "DropoutRegularizer",
    "WeightDecay",
    "AdaptiveLR",
    "AdaptiveMomentum",
    "AdaptiveRegularization",
    "LSTMOptimizer",
    "TransformerOptimizer",
    "AttentionOptimizer",
    "MetaSGD",
    "ConstantLR",
    "ExponentialLR",
    "StepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "CyclicLR",
    "Reptile",
];

/// Number of continuous descriptors appended after the multi-hot block.
const COMPONENT_DESCRIPTOR_COUNT: usize = 3;

/// Total fixed length of the component feature block produced by
/// [`encode_component_block`]: one slot per known type, one out-of-vocabulary
/// slot, and the trailing continuous descriptors.
const COMPONENT_BLOCK_LEN: usize = COMPONENT_VOCABULARY.len() + 1 + COMPONENT_DESCRIPTOR_COUNT;

/// Resolve a component type name to its vocabulary index.
///
/// Returns the matching index for a known type, or the dedicated
/// out-of-vocabulary index (`COMPONENT_VOCABULARY.len()`) for any unrecognised
/// name.  The lookup is exact and deterministic.
fn component_vocab_index(name: &str) -> usize {
    COMPONENT_VOCABULARY
        .iter()
        .position(|known| *known == name)
        .unwrap_or(COMPONENT_VOCABULARY.len())
}

/// Build a deterministic, fixed-length feature block for a sequence of
/// component type names.
///
/// Layout (length [`COMPONENT_BLOCK_LEN`]):
/// * `[0, VOCAB_LEN)`  multi-hot counts: how many components of each known type
///   are present (occurrence counts, so repeated types accumulate);
/// * `[VOCAB_LEN]`     out-of-vocabulary count for unrecognised names;
/// * trailing descriptors: normalised component count, mean normalised name
///   length, and fraction of names containing the `"Adam"` substring (a cheap
///   family indicator).  These add continuous structure on top of the
///   discrete one-hot signal so the GP kernel can exploit gradients in
///   component count / family composition.
fn encode_component_block<'a, T, I>(components: I) -> Vec<T>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
    I: Iterator<Item = &'a String>,
{
    let mut block = vec![T::zero(); COMPONENT_BLOCK_LEN];

    let one: T = scirs2_core::numeric::NumCast::from(1.0).unwrap_or_else(|| T::zero());

    let mut total: usize = 0;
    let mut name_len_sum: usize = 0;
    let mut adam_family: usize = 0;

    for component in components {
        let idx = component_vocab_index(component);
        block[idx] = block[idx] + one;

        total += 1;
        name_len_sum += component.len();
        if component.contains("Adam") {
            adam_family += 1;
        }
    }

    // Continuous descriptors.  Normalisers are chosen to keep values in a
    // roughly unit range without depending on any RNG.
    let descriptor_base = COMPONENT_VOCABULARY.len() + 1;
    if total > 0 {
        let total_t: T =
            scirs2_core::numeric::NumCast::from(total as f64).unwrap_or_else(|| T::zero());

        // Normalised component count (relative to a nominal cap of 16).
        block[descriptor_base] =
            scirs2_core::numeric::NumCast::from(total as f64 / 16.0).unwrap_or_else(|| T::zero());

        // Mean name length, normalised by a nominal max name length of 24.
        let mean_len = (name_len_sum as f64 / total as f64) / 24.0;
        block[descriptor_base + 1] =
            scirs2_core::numeric::NumCast::from(mean_len).unwrap_or_else(|| T::zero());

        // Fraction of Adam-family components.
        let adam_frac: T =
            scirs2_core::numeric::NumCast::from(adam_family as f64).unwrap_or_else(|| T::zero());
        block[descriptor_base + 2] = adam_frac / total_t;
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_arch(components: &[&str], hyper: &[(&str, f64)]) -> OptimizerArchitecture<f64> {
        let mut hyperparameters = HashMap::new();
        for (k, v) in hyper {
            hyperparameters.insert(k.to_string(), *v);
        }
        OptimizerArchitecture {
            components: components.iter().map(|s| s.to_string()).collect(),
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters,
            architecture_id: "test".to_string(),
        }
    }

    fn make_bo() -> BayesianOptimization<f64> {
        BayesianOptimization::<f64>::new(KernelType::RBF, AcquisitionType::UCB, 0.1)
    }

    #[test]
    fn encode_is_deterministic_and_fixed_length() {
        let bo = make_bo();
        let arch = make_arch(&["Adam", "SGD"], &[("learning_rate", 0.01), ("beta1", 0.9)]);

        let first = bo.encode_architecture(&arch);
        let second = bo.encode_architecture(&arch);

        assert_eq!(first.len(), 64);
        assert_eq!(second.len(), 64);
        assert_eq!(first, second, "encoding must be deterministic");
    }

    #[test]
    fn different_known_types_differ_same_type_matches() {
        let bo = make_bo();

        let adam = bo.encode_architecture(&make_arch(&["Adam"], &[]));
        let adam_again = bo.encode_architecture(&make_arch(&["Adam"], &[]));
        let sgd = bo.encode_architecture(&make_arch(&["SGD"], &[]));

        assert_eq!(adam, adam_again, "same type must encode identically");
        assert_ne!(adam, sgd, "different known types must differ");

        let adam_idx = component_vocab_index("Adam");
        let sgd_idx = component_vocab_index("SGD");
        assert_ne!(adam_idx, sgd_idx);
        assert_eq!(adam[adam_idx], 1.0);
        assert_eq!(sgd[sgd_idx], 1.0);
    }

    #[test]
    fn unknown_type_maps_to_oov_slot() {
        let bo = make_bo();
        let oov_index = COMPONENT_VOCABULARY.len();

        let encoded = bo.encode_architecture(&make_arch(&["NoSuchOptimizer"], &[]));

        assert_eq!(encoded.len(), 64);
        assert_eq!(
            encoded[oov_index], 1.0,
            "unknown name must land in the out-of-vocabulary slot"
        );
        for (i, value) in encoded.iter().enumerate().take(oov_index) {
            assert_eq!(*value, 0.0, "known slot {} must stay zero", i);
        }
    }

    // -----------------------------------------------------------------
    // Gaussian process (F4)
    // -----------------------------------------------------------------

    /// 1-D training inputs padded into the GP's fixed 64-length input space.
    fn point(x: f64) -> Array1<f64> {
        let mut v = vec![0.0; 64];
        v[0] = x;
        Array1::from_vec(v)
    }

    fn toy_dataset() -> (Vec<Array1<f64>>, Vec<f64>) {
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let inputs: Vec<Array1<f64>> = xs.iter().map(|&x| point(x)).collect();
        // A smooth target so the GP has something to interpolate.
        let targets: Vec<f64> = xs.iter().map(|&x| (x).sin()).collect();
        (inputs, targets)
    }

    #[test]
    fn gp_interpolates_its_training_points() {
        for kernel in [KernelType::RBF, KernelType::Matern32, KernelType::Matern52] {
            let mut gp = GaussianProcess::<f64>::new(kernel);
            gp.set_noise_variance(1e-10);
            let (inputs, targets) = toy_dataset();
            gp.fit(&inputs, &targets).expect("fit");
            assert!(gp.is_fitted());

            for (input, &target) in inputs.iter().zip(targets.iter()) {
                let (mean, variance) = gp.predict(input).expect("predict");
                assert!(
                    (mean - target).abs() < 1e-4,
                    "{:?}: posterior mean {} should interpolate {}",
                    kernel,
                    mean,
                    target
                );
                assert!(
                    variance < 1e-6,
                    "{:?}: posterior variance {} should vanish at a training point",
                    kernel,
                    variance
                );
            }
        }
    }

    #[test]
    fn gp_variance_grows_away_from_the_data() {
        let mut gp = GaussianProcess::<f64>::new(KernelType::RBF);
        gp.set_noise_variance(1e-10);
        let (inputs, targets) = toy_dataset();
        gp.fit(&inputs, &targets).expect("fit");

        let (_, near) = gp.predict(&point(0.0)).expect("near");
        let (_, far) = gp.predict(&point(25.0)).expect("far");

        assert!(near < 1e-6);
        assert!(
            far > 0.5,
            "variance far from the data ({}) must approach the prior",
            far
        );
        assert!(far > near);
    }

    #[test]
    fn gp_prior_is_reported_before_fitting() {
        let gp = GaussianProcess::<f64>::new(KernelType::RBF);
        let (mean, variance) = gp.predict(&point(0.0)).expect("predict");
        assert_eq!(mean, 0.0);
        assert_eq!(variance, gp.kernel().signal_variance());
        assert!(!gp.is_fitted());
    }

    #[test]
    fn gp_handles_duplicate_inputs_via_jitter() {
        // Duplicate encodings make K singular; the jitter ladder must rescue it
        // rather than returning an error.
        let mut gp = GaussianProcess::<f64>::new(KernelType::RBF);
        gp.set_noise_variance(0.0);
        let inputs = vec![point(1.0), point(1.0), point(2.0)];
        let targets = vec![0.5, 0.5, 0.9];
        gp.fit(&inputs, &targets)
            .expect("degenerate fit must succeed");
        assert!(gp.is_fitted());
    }

    #[test]
    fn every_kernel_variant_is_implemented() {
        let a = point(1.0);
        let b = point(2.0);
        for kernel in [
            KernelType::RBF,
            KernelType::Matern32,
            KernelType::Matern52,
            KernelType::Linear,
            KernelType::Polynomial,
        ] {
            let k = GPKernel::<f64>::new(kernel);
            let same = k.evaluate(&a, &a);
            let cross = k.evaluate(&a, &b);
            assert!(same.is_finite(), "{:?} k(a,a) not finite", kernel);
            assert!(cross.is_finite(), "{:?} k(a,b) not finite", kernel);
            // Stationary kernels peak on the diagonal.
            if matches!(
                kernel,
                KernelType::RBF | KernelType::Matern32 | KernelType::Matern52
            ) {
                assert!(same > cross, "{:?} must decay with distance", kernel);
                assert!((same - k.signal_variance()).abs() < 1e-12);
            }
            // Symmetry holds for every kernel.
            assert!((cross - k.evaluate(&b, &a)).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_reconstructs_the_original_matrix() {
        let a = Array2::from_shape_vec((3, 3), vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0])
            .expect("matrix");
        let l = cholesky(&a).expect("positive definite");
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += l[[i, k]] * l[[j, k]];
                }
                assert!((acc - a[[i, j]]).abs() < 1e-9);
            }
        }
        // Not positive definite -> None, not a panic.
        let bad = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("matrix");
        assert!(cholesky(&bad).is_none());
    }

    // -----------------------------------------------------------------
    // Acquisition functions (F5)
    // -----------------------------------------------------------------

    #[test]
    fn norm_cdf_and_pdf_match_known_values() {
        assert!((norm_cdf::<f64>(0.0) - 0.5).abs() < 1e-6);
        assert!((norm_cdf::<f64>(1.96) - 0.975).abs() < 1e-4);
        assert!((norm_cdf::<f64>(-1.96) - 0.025).abs() < 1e-4);
        assert!((norm_pdf::<f64>(0.0) - 0.398_942_280).abs() < 1e-8);
        assert!((norm_pdf::<f64>(1.0) - 0.241_970_724).abs() < 1e-8);
    }

    #[test]
    fn ei_prefers_uncertainty_at_equal_means() {
        let mut ei = AcquisitionFunction::<f64>::new(AcquisitionType::EI, 0.0, Some(1));
        ei.observe(0.5);

        let certain = ei.evaluate(0.5, 1e-9);
        let uncertain = ei.evaluate(0.5, 0.25);

        assert!(
            uncertain > certain,
            "EI must reward uncertainty ({} vs {})",
            uncertain,
            certain
        );
        // At the incumbent, EI is exactly sigma * phi(0); with sigma = 0 there
        // is no expected gain at all, and it grows with sigma.
        assert_eq!(ei.evaluate(0.5, 0.0), 0.0);
        let expected = 0.25_f64.sqrt() * norm_pdf::<f64>(0.0);
        assert!((uncertain - expected).abs() < 1e-9);
    }

    #[test]
    fn ei_and_pi_are_monotone_in_the_mean() {
        let mut ei = AcquisitionFunction::<f64>::new(AcquisitionType::EI, 0.0, Some(2));
        let mut pi = AcquisitionFunction::<f64>::new(AcquisitionType::PI, 0.0, Some(3));
        ei.observe(0.5);
        pi.observe(0.5);

        let (low_ei, high_ei) = (ei.evaluate(0.4, 0.04), ei.evaluate(0.8, 0.04));
        let (low_pi, high_pi) = (pi.evaluate(0.4, 0.04), pi.evaluate(0.8, 0.04));

        assert!(high_ei > low_ei);
        assert!(high_pi > low_pi);
        for value in [low_pi, high_pi] {
            assert!((0.0..=1.0).contains(&value), "PI must be a probability");
        }
    }

    #[test]
    fn ucb_uses_the_standard_deviation() {
        let mut ucb = AcquisitionFunction::<f64>::new(AcquisitionType::UCB, 2.0, Some(4));
        // mu + kappa * sigma with sigma = sqrt(0.25) = 0.5 -> 1.0 + 2*0.5 = 2.0
        assert!((ucb.evaluate(1.0, 0.25) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn info_gain_is_monotone_in_variance() {
        let mut ig = AcquisitionFunction::<f64>::new(AcquisitionType::InfoGain, 0.0, Some(5));
        assert!(ig.evaluate(0.0, 1.0) > ig.evaluate(0.0, 0.1));
    }

    #[test]
    fn thompson_sampling_actually_varies() {
        let mut ts = AcquisitionFunction::<f64>::new(AcquisitionType::Thompson, 0.0, Some(6));
        let draws: Vec<f64> = (0..16).map(|_| ts.evaluate(0.0, 1.0)).collect();
        let distinct: std::collections::HashSet<String> =
            draws.iter().map(|v| format!("{:.9}", v)).collect();
        assert!(
            distinct.len() > 8,
            "Thompson sampling must draw a fresh sample each call"
        );
    }

    // -----------------------------------------------------------------
    // Strategy-level behaviour (F5 current_best, F6 persistent RNG)
    // -----------------------------------------------------------------

    fn search_space() -> SearchSpaceConfig {
        use crate::nas_engine::config::{
            ComponentType as ConfigComponentType, OptimizerComponentConfig, ParameterRange,
        };
        SearchSpaceConfig {
            components: vec![
                OptimizerComponentConfig {
                    component_type: ConfigComponentType::Adam,
                    hyperparameter_ranges: {
                        let mut r = HashMap::new();
                        r.insert(
                            "learning_rate".to_string(),
                            ParameterRange::LogUniform(1e-4, 1e-1),
                        );
                        r
                    },
                    complexity_score: 1.0,
                    memory_requirement: 1024,
                    computational_cost: 1.0,
                    compatibility_constraints: Vec::new(),
                },
                OptimizerComponentConfig {
                    component_type: ConfigComponentType::SGD,
                    hyperparameter_ranges: {
                        let mut r = HashMap::new();
                        r.insert(
                            "learning_rate".to_string(),
                            ParameterRange::Continuous(1e-3, 1e-1),
                        );
                        r
                    },
                    complexity_score: 0.5,
                    memory_requirement: 512,
                    computational_cost: 0.5,
                    compatibility_constraints: Vec::new(),
                },
            ],
            min_components: 1,
            max_components: 3,
            ..SearchSpaceConfig::default()
        }
    }

    #[test]
    fn candidate_pool_is_diverse() {
        // Regression test: a fresh seeded RandomSearch used to be built on every
        // call, so all candidates were byte-identical.
        let space = search_space();
        let mut bo = BayesianOptimization::<f64>::new_with_seed(
            KernelType::RBF,
            AcquisitionType::UCB,
            0.1,
            42,
        );
        bo.initialize(&space).expect("initialize");

        let mut distinct = std::collections::HashSet::new();
        for _ in 0..20 {
            let arch = bo
                .generate_architecture(&space, &VecDeque::new())
                .expect("generate");
            let mut hyper: Vec<String> = arch
                .hyperparameters
                .iter()
                .map(|(k, v)| format!("{}={:.9}", k, v))
                .collect();
            hyper.sort();
            distinct.insert(format!("{:?}|{:?}", arch.components, hyper));
        }
        assert!(
            distinct.len() > 1,
            "Bayesian optimization must produce more than one distinct candidate"
        );
    }

    #[test]
    fn empty_search_space_is_an_error() {
        let mut space = search_space();
        space.components.clear();
        let mut bo = BayesianOptimization::<f64>::new(KernelType::RBF, AcquisitionType::EI, 0.1);
        assert!(matches!(
            bo.initialize(&space),
            Err(OptimError::SearchSpaceError(_))
        ));
        assert!(matches!(
            bo.generate_architecture(&space, &VecDeque::new()),
            Err(OptimError::SearchSpaceError(_))
        ));
    }

    #[test]
    fn current_best_tracks_the_observations() {
        use crate::nas_engine::{
            ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
        };

        let mut bo = BayesianOptimization::<f64>::new_with_seed(
            KernelType::RBF,
            AcquisitionType::EI,
            0.1,
            9,
        );
        assert!(!bo.current_best().is_finite(), "no incumbent before data");

        let make_result = |score: f64| {
            let mut metric_scores = HashMap::new();
            metric_scores.insert(EvaluationMetric::FinalPerformance, score);
            SearchResult {
                architecture: make_arch(&["Adam"], &[("learning_rate", score)]),
                evaluation_results: EvaluationResults {
                    metric_scores,
                    overall_score: score,
                    confidence_intervals: HashMap::new(),
                    evaluation_time: std::time::Duration::from_secs(0),
                    success: true,
                    error_message: None,
                    cv_results: None,
                    benchmark_results: HashMap::new(),
                    training_trajectory: Vec::new(),
                },
                generation: 0,
                search_time: 0.0,
                resource_usage: ResourceUsage::default(),
                encoding: ArchitectureEncoding::default(),
                metadata: SearchResultMetadata::default(),
            }
        };

        bo.update_with_results(&[make_result(0.3), make_result(0.7)])
            .expect("update");
        assert!((bo.current_best() - 0.7).abs() < 1e-12);

        bo.update_with_results(&[make_result(0.5)]).expect("update");
        assert!(
            (bo.current_best() - 0.7).abs() < 1e-12,
            "incumbent must not regress"
        );

        bo.update_with_results(&[make_result(0.95)])
            .expect("update");
        assert!((bo.current_best() - 0.95).abs() < 1e-12);
    }

    #[test]
    fn hyperparameter_order_is_stable() {
        let bo = make_bo();
        // Same hyperparameters supplied in different insertion order must yield
        // identical encodings thanks to the key-sorted append.
        let a = bo.encode_architecture(&make_arch(
            &["Adam"],
            &[("alpha", 0.1), ("beta", 0.2), ("gamma", 0.3)],
        ));
        let b = bo.encode_architecture(&make_arch(
            &["Adam"],
            &[("gamma", 0.3), ("alpha", 0.1), ("beta", 0.2)],
        ));
        assert_eq!(a, b, "hyperparameter ordering must be stable");
    }
}
