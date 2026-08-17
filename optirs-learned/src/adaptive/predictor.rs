//! Real performance prediction for the adaptive transformer enhancement.
//!
//! `TransformerPerformancePredictor::predict_improvement` used to return the
//! constants `0.15 / 0.92 / 0.85 / 0.05` for every input, and its
//! `PredictorNetwork` was zero-initialized (so wiring it up as-is would have
//! output `0` forever). This module replaces both with a working predictor:
//!
//! 1. **Features** — [`PredictionFeatures`] turns a landscape analysis plus a
//!    proposed architecture adaptation into a fixed-length real feature vector.
//! 2. **Random feature map** — [`PredictorNetwork`] keeps Xavier-initialized
//!    hidden layers with `tanh` nonlinearities. They are *fixed* (a random
//!    feature expansion, as in random-feature / extreme-learning-machine
//!    regression), so the model is linear in its trainable parameters.
//! 3. **Learned heads** — the output layer is fitted by ridge regression with a
//!    closed-form normal-equation solve (Cholesky). Two heads are learned:
//!    convergence improvement and final performance.
//! 4. **Uncertainty** — the ridge posterior gives a genuine predictive standard
//!    deviation `σ·sqrt(1 + φᵗA⁻¹φ)`; confidence is derived from it. Before any
//!    training the predictor reports maximum uncertainty and zero confidence
//!    instead of a fabricated `0.85`.
//!
//! Everything is pure Rust over `scirs2_core::ndarray`.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Number of real features the predictor consumes.
pub const PREDICTION_FEATURE_COUNT: usize = 16;

/// Feature vector handed to [`PredictorNetwork`].
///
/// Field order is the vector order; see [`Self::to_vec`].
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionFeatures {
    /// Landscape complexity in `[0, 1]`.
    pub complexity: f64,
    /// Landscape difficulty in `[0, 1]`.
    pub difficulty: f64,
    /// Confidence attached to the landscape analysis, `[0, 1]`.
    pub landscape_confidence: f64,
    /// Improvement the architecture adapter expects.
    pub expected_improvement: f64,
    /// Confidence the architecture adapter reports.
    pub adaptation_confidence: f64,
    /// Number of architecture changes, normalized by a soft cap of 8.
    pub change_count: f64,
    /// Proposed layer count, normalized by a soft cap of 24.
    pub layer_count: f64,
    /// Proposed hidden width, normalized by a soft cap of 2048.
    pub hidden_size: f64,
    /// Proposed attention-head count, normalized by a soft cap of 32.
    pub head_count: f64,
    /// Proposed dropout rate.
    pub dropout: f64,
    /// `complexity · difficulty` interaction.
    pub interaction: f64,
    /// `complexity²`.
    pub complexity_sq: f64,
    /// `difficulty²`.
    pub difficulty_sq: f64,
    /// One-hot-ish encoding of the recommended strategy family (3 slots):
    /// conservative, aggressive, exploratory.
    pub strategy_conservative: f64,
    /// See [`Self::strategy_conservative`].
    pub strategy_aggressive: f64,
    /// See [`Self::strategy_conservative`].
    pub strategy_exploratory: f64,
}

impl PredictionFeatures {
    /// Flatten into the fixed-length vector the network expects.
    pub fn to_vec<T>(&self) -> Array1<T>
    where
        T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
    {
        let raw = [
            self.complexity,
            self.difficulty,
            self.landscape_confidence,
            self.expected_improvement,
            self.adaptation_confidence,
            self.change_count,
            self.layer_count,
            self.hidden_size,
            self.head_count,
            self.dropout,
            self.interaction,
            self.complexity_sq,
            self.difficulty_sq,
            self.strategy_conservative,
            self.strategy_aggressive,
            self.strategy_exploratory,
        ];
        debug_assert_eq!(raw.len(), PREDICTION_FEATURE_COUNT);
        Array1::from_shape_fn(PREDICTION_FEATURE_COUNT, |i| {
            scirs2_core::numeric::NumCast::from(raw[i]).unwrap_or_else(|| T::zero())
        })
    }

    /// Stable cache key. Quantized to 4 decimals so numerically identical
    /// requests hit the cache while genuinely different ones do not collide.
    pub fn cache_key(&self) -> String {
        let raw = [
            self.complexity,
            self.difficulty,
            self.landscape_confidence,
            self.expected_improvement,
            self.adaptation_confidence,
            self.change_count,
            self.layer_count,
            self.hidden_size,
            self.head_count,
            self.dropout,
            self.strategy_conservative,
            self.strategy_aggressive,
            self.strategy_exploratory,
        ];
        let mut key = String::with_capacity(raw.len() * 8);
        for v in raw {
            key.push_str(&format!("{:.4}|", v));
        }
        key
    }
}

/// One `(features, targets)` training pair.
#[derive(Debug, Clone)]
pub struct PredictorSample {
    /// Input features.
    pub features: PredictionFeatures,
    /// Observed convergence improvement.
    pub convergence_improvement: f64,
    /// Observed final performance.
    pub final_performance: f64,
}

/// Result of a prediction: two means plus a real uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictorOutput {
    /// Predicted convergence improvement.
    pub convergence_improvement: f64,
    /// Predicted final performance.
    pub final_performance: f64,
    /// Predictive standard deviation shared by both heads (the larger of the two).
    pub uncertainty: f64,
    /// `1 / (1 + uncertainty)` — monotonically decreasing in uncertainty, and
    /// exactly `0` while the predictor is untrained.
    pub confidence: f64,
}

/// Fitted-model diagnostics returned by [`PredictorNetwork::fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct PredictorFitReport {
    /// Root-mean-square error of the convergence head on the training set.
    pub convergence_rmse: f64,
    /// Root-mean-square error of the performance head on the training set.
    pub performance_rmse: f64,
    /// Number of samples the fit used.
    pub samples: usize,
}

/// Performance prediction network: fixed random feature map + ridge-fitted heads.
#[derive(Debug, Clone)]
pub struct PredictorNetwork<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Fixed hidden layers (Xavier-initialized, never zero).
    hidden_weights: Vec<Array2<T>>,
    /// Hidden biases.
    hidden_biases: Vec<Array1<T>>,
    /// Learned head for convergence improvement (length `feature_width + 1`).
    convergence_head: Array1<T>,
    /// Learned head for final performance.
    performance_head: Array1<T>,
    /// `(ΦᵗΦ + λI)⁻¹` in augmented feature space, kept for predictive variance.
    posterior_precision_inv: Option<Array2<T>>,
    /// Residual standard deviation of the fit.
    residual_std: T,
    /// Ridge coefficient.
    ridge_lambda: T,
    /// Layer widths, `[input, hidden..., feature_width]`.
    architecture: Vec<usize>,
    /// Whether [`Self::fit`] has succeeded at least once.
    trained: bool,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    PredictorNetwork<T>
{
    /// Build the network.
    ///
    /// `architecture` lists the widths of the input layer and every hidden
    /// layer; the last entry is the width of the feature map the linear heads
    /// read. `ridge_lambda` must be positive so the normal equations stay
    /// positive-definite (and hence Cholesky-solvable) even with fewer samples
    /// than features.
    ///
    /// # Errors
    /// Returns `Err` when the architecture has fewer than two layers, any width
    /// is zero, the input width disagrees with [`PREDICTION_FEATURE_COUNT`], or
    /// `ridge_lambda <= 0`.
    pub fn new(architecture: Vec<usize>, ridge_lambda: T) -> Result<Self> {
        if architecture.len() < 2 {
            return Err(OptimError::InvalidConfig(
                "predictor architecture needs an input and at least one hidden layer".to_string(),
            ));
        }
        if architecture.contains(&0) {
            return Err(OptimError::InvalidConfig(
                "predictor layer widths must be positive".to_string(),
            ));
        }
        if architecture[0] != PREDICTION_FEATURE_COUNT {
            return Err(OptimError::InvalidConfig(format!(
                "predictor input width {} must equal PREDICTION_FEATURE_COUNT ({})",
                architecture[0], PREDICTION_FEATURE_COUNT
            )));
        }
        if ridge_lambda <= T::zero() {
            return Err(OptimError::InvalidConfig(
                "ridge_lambda must be positive".to_string(),
            ));
        }

        let mut hidden_weights = Vec::with_capacity(architecture.len() - 1);
        let mut hidden_biases = Vec::with_capacity(architecture.len() - 1);
        let mut rng = scirs2_core::random::thread_rng();
        for i in 0..architecture.len() - 1 {
            let fan_in = architecture[i];
            let fan_out = architecture[i + 1];
            // Xavier/Glorot uniform: limit sqrt(6 / (fan_in + fan_out)).
            let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let weight = Array2::from_shape_fn((fan_out, fan_in), |_| {
                let sample = rng.random::<f64>() * 2.0 - 1.0;
                scirs2_core::numeric::NumCast::from(sample * bound).unwrap_or_else(|| T::zero())
            });
            hidden_weights.push(weight);
            hidden_biases.push(Array1::zeros(fan_out));
        }

        let feature_width = architecture[architecture.len() - 1];
        Ok(Self {
            hidden_weights,
            hidden_biases,
            // Untrained heads are zero on purpose: an untrained predictor must
            // say "I predict nothing, with maximum uncertainty", not invent a
            // number. `trained` gates that.
            convergence_head: Array1::zeros(feature_width + 1),
            performance_head: Array1::zeros(feature_width + 1),
            posterior_precision_inv: None,
            residual_std: T::zero(),
            ridge_lambda,
            architecture,
            trained: false,
        })
    }

    /// Whether the heads have been fitted.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Layer widths.
    pub fn architecture(&self) -> &[usize] {
        &self.architecture
    }

    /// Width of the feature map the heads read.
    pub fn feature_width(&self) -> usize {
        self.architecture[self.architecture.len() - 1]
    }

    /// Total parameter count (fixed map + learned heads).
    pub fn parameter_count(&self) -> usize {
        let hidden: usize = self.hidden_weights.iter().map(|w| w.len()).sum::<usize>()
            + self.hidden_biases.iter().map(|b| b.len()).sum::<usize>();
        hidden + self.convergence_head.len() + self.performance_head.len()
    }

    /// Random feature map `φ(x)`, augmented with a trailing 1 for the bias.
    fn feature_map(&self, input: &Array1<T>) -> Result<Array1<T>> {
        if input.len() != self.architecture[0] {
            return Err(OptimError::InvalidConfig(format!(
                "predictor expected {} features, got {}",
                self.architecture[0],
                input.len()
            )));
        }
        let mut activation = input.clone();
        for (weight, bias) in self.hidden_weights.iter().zip(self.hidden_biases.iter()) {
            let pre = weight.dot(&activation) + bias;
            // tanh keeps the map bounded, so the ridge solve is well-scaled
            // regardless of how large the raw features are.
            activation = pre.mapv(|x| x.tanh());
        }

        let mut augmented = Array1::zeros(activation.len() + 1);
        for (i, &v) in activation.iter().enumerate() {
            augmented[i] = v;
        }
        augmented[activation.len()] = T::one();
        Ok(augmented)
    }

    /// Fit both heads by ridge regression on `samples`.
    ///
    /// Solves `(ΦᵗΦ + λI) w = Φᵗ y` exactly via Cholesky. Returns the training
    /// RMSE of each head.
    ///
    /// # Errors
    /// Returns `Err` when `samples` is empty or the normal-equation matrix is
    /// not positive-definite (which cannot happen for `λ > 0`, but is checked
    /// rather than assumed).
    pub fn fit(&mut self, samples: &[PredictorSample]) -> Result<PredictorFitReport> {
        if samples.is_empty() {
            return Err(OptimError::InsufficientData(
                "predictor training needs at least one sample".to_string(),
            ));
        }

        let width = self.feature_width() + 1;
        let mut design: Vec<Array1<T>> = Vec::with_capacity(samples.len());
        for sample in samples {
            design.push(self.feature_map(&sample.features.to_vec::<T>())?);
        }

        // Normal equations.
        let mut gram = Array2::<T>::zeros((width, width));
        let mut rhs_conv = Array1::<T>::zeros(width);
        let mut rhs_perf = Array1::<T>::zeros(width);
        for (phi, sample) in design.iter().zip(samples.iter()) {
            let y_c: T = scirs2_core::numeric::NumCast::from(sample.convergence_improvement)
                .unwrap_or_else(|| T::zero());
            let y_p: T = scirs2_core::numeric::NumCast::from(sample.final_performance)
                .unwrap_or_else(|| T::zero());
            for i in 0..width {
                rhs_conv[i] = rhs_conv[i] + phi[i] * y_c;
                rhs_perf[i] = rhs_perf[i] + phi[i] * y_p;
                for j in 0..width {
                    gram[[i, j]] = gram[[i, j]] + phi[i] * phi[j];
                }
            }
        }
        for i in 0..width {
            gram[[i, i]] = gram[[i, i]] + self.ridge_lambda;
        }

        let chol = cholesky(&gram)?;
        self.convergence_head = cholesky_solve(&chol, &rhs_conv);
        self.performance_head = cholesky_solve(&chol, &rhs_perf);
        self.posterior_precision_inv = Some(cholesky_inverse(&chol));

        // Residual statistics (pooled across both heads).
        let mut sq_conv = T::zero();
        let mut sq_perf = T::zero();
        for (phi, sample) in design.iter().zip(samples.iter()) {
            let pred_c = dot(&self.convergence_head, phi);
            let pred_p = dot(&self.performance_head, phi);
            let y_c: T = scirs2_core::numeric::NumCast::from(sample.convergence_improvement)
                .unwrap_or_else(|| T::zero());
            let y_p: T = scirs2_core::numeric::NumCast::from(sample.final_performance)
                .unwrap_or_else(|| T::zero());
            sq_conv = sq_conv + (pred_c - y_c) * (pred_c - y_c);
            sq_perf = sq_perf + (pred_p - y_p) * (pred_p - y_p);
        }
        let n = T::from(samples.len()).unwrap_or_else(|| T::one());
        let rmse_conv = (sq_conv / n).sqrt();
        let rmse_perf = (sq_perf / n).sqrt();
        self.residual_std = if rmse_conv > rmse_perf {
            rmse_conv
        } else {
            rmse_perf
        };
        self.trained = true;

        Ok(PredictorFitReport {
            convergence_rmse: rmse_conv.to_f64().unwrap_or(0.0),
            performance_rmse: rmse_perf.to_f64().unwrap_or(0.0),
            samples: samples.len(),
        })
    }

    /// Predict from a feature vector.
    ///
    /// While untrained the means are `0` and the uncertainty is `1` with
    /// confidence `0` — an explicit "no information" answer rather than a
    /// fabricated constant.
    pub fn predict(&self, features: &PredictionFeatures) -> Result<PredictorOutput> {
        let phi = self.feature_map(&features.to_vec::<T>())?;
        if !self.trained {
            return Ok(PredictorOutput {
                convergence_improvement: 0.0,
                final_performance: 0.0,
                uncertainty: 1.0,
                confidence: 0.0,
            });
        }

        let mean_c = dot(&self.convergence_head, &phi).to_f64().unwrap_or(0.0);
        let mean_p = dot(&self.performance_head, &phi).to_f64().unwrap_or(0.0);

        // Ridge predictive variance: σ²·(1 + φᵗ A⁻¹ φ).
        let leverage = match &self.posterior_precision_inv {
            Some(a_inv) => {
                let av = a_inv.dot(&phi);
                dot(&av, &phi).to_f64().unwrap_or(0.0).max(0.0)
            }
            None => 0.0,
        };
        let sigma = self.residual_std.to_f64().unwrap_or(0.0).abs();
        let uncertainty = sigma * (1.0 + leverage).sqrt();

        Ok(PredictorOutput {
            convergence_improvement: mean_c,
            final_performance: mean_p,
            uncertainty,
            confidence: 1.0 / (1.0 + uncertainty),
        })
    }
}

/// Bounded prediction cache with real hit-rate accounting.
///
/// The previous version stored a `HashMap`, a `capacity` and a `hit_rate` and
/// never read or updated any of them.
#[derive(Debug, Clone)]
pub struct PredictionCache {
    entries: HashMap<String, PredictorOutput>,
    /// Insertion order, for FIFO eviction at capacity.
    order: Vec<String>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl PredictionCache {
    /// Create a cache holding at most `capacity` entries (`capacity == 0`
    /// disables caching entirely).
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a prediction, updating the hit/miss counters.
    pub fn get(&mut self, key: &str) -> Option<PredictorOutput> {
        match self.entries.get(key) {
            Some(v) => {
                self.hits += 1;
                Some(v.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert a prediction, evicting the oldest entry at capacity.
    pub fn insert(&mut self, key: String, value: PredictorOutput) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.insert(key.clone(), value).is_none() {
            self.order.push(key);
            while self.order.len() > self.capacity {
                let evicted = self.order.remove(0);
                self.entries.remove(&evicted);
            }
        }
    }

    /// Observed hit rate, or `0` before any lookup.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every entry (counters are preserved).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
}

// ---------------------------------------------------------------------------
// Small dense linear algebra (pure Rust, no BLAS)
// ---------------------------------------------------------------------------

fn dot<T: Float>(a: &Array1<T>, b: &Array1<T>) -> T {
    let n = a.len().min(b.len());
    let mut acc = T::zero();
    for i in 0..n {
        acc = acc + a[i] * b[i];
    }
    acc
}

/// Lower-triangular Cholesky factor `L` with `A = L Lᵗ`.
///
/// # Errors
/// Returns `Err` when `A` is not symmetric positive-definite.
fn cholesky<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>(
    a: &Array2<T>,
) -> Result<Array2<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OptimError::InvalidConfig(
            "Cholesky requires a square matrix".to_string(),
        ));
    }
    let mut l = Array2::<T>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= T::zero() {
                    return Err(OptimError::ComputationError(format!(
                        "matrix is not positive-definite at pivot {i}"
                    )));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Solve `L Lᵗ x = b` given the Cholesky factor `L`.
fn cholesky_solve<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
>(
    l: &Array2<T>,
    b: &Array1<T>,
) -> Array1<T> {
    let n = l.nrows();
    // Forward substitution: L y = b
    let mut y = Array1::<T>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum = sum - l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    // Back substitution: Lᵗ x = y
    let mut x = Array1::<T>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum = sum - l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// `(L Lᵗ)⁻¹`, computed column by column.
fn cholesky_inverse<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
>(
    l: &Array2<T>,
) -> Array2<T> {
    let n = l.nrows();
    let mut inv = Array2::<T>::zeros((n, n));
    let mut unit = Array1::<T>::zeros(n);
    for col in 0..n {
        unit.fill(T::zero());
        unit[col] = T::one();
        let solution = cholesky_solve(l, &unit);
        for row in 0..n {
            inv[[row, col]] = solution[row];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(complexity: f64, difficulty: f64) -> PredictionFeatures {
        PredictionFeatures {
            complexity,
            difficulty,
            landscape_confidence: 0.8,
            expected_improvement: 0.1,
            adaptation_confidence: 0.7,
            change_count: 0.125,
            layer_count: 6.0 / 24.0,
            hidden_size: 512.0 / 2048.0,
            head_count: 8.0 / 32.0,
            dropout: 0.1,
            interaction: complexity * difficulty,
            complexity_sq: complexity * complexity,
            difficulty_sq: difficulty * difficulty,
            strategy_conservative: 0.0,
            strategy_aggressive: 0.0,
            strategy_exploratory: 1.0,
        }
    }

    fn network() -> PredictorNetwork<f64> {
        PredictorNetwork::<f64>::new(vec![PREDICTION_FEATURE_COUNT, 32, 24], 1e-4)
            .expect("network construction")
    }

    /// Synthetic ground truth: improvement falls with difficulty and rises with
    /// (low) complexity; final performance is the complement.
    fn truth(f: &PredictionFeatures) -> (f64, f64) {
        let improvement = 0.30 - 0.20 * f.difficulty - 0.05 * f.complexity;
        let performance = 0.95 - 0.30 * f.difficulty - 0.10 * f.complexity;
        (improvement, performance)
    }

    fn training_set(n: usize) -> Vec<PredictorSample> {
        (0..n)
            .map(|i| {
                let c = (i as f64 * 0.037).fract();
                let d = (i as f64 * 0.091).fract();
                let f = features(c, d);
                let (ci, fp) = truth(&f);
                PredictorSample {
                    features: f,
                    convergence_improvement: ci,
                    final_performance: fp,
                }
            })
            .collect()
    }

    #[test]
    fn rejects_bad_architecture() {
        assert!(PredictorNetwork::<f64>::new(vec![PREDICTION_FEATURE_COUNT], 1e-4).is_err());
        assert!(PredictorNetwork::<f64>::new(vec![3, 8], 1e-4).is_err());
        assert!(PredictorNetwork::<f64>::new(vec![PREDICTION_FEATURE_COUNT, 0], 1e-4).is_err());
        assert!(PredictorNetwork::<f64>::new(vec![PREDICTION_FEATURE_COUNT, 8], 0.0).is_err());
    }

    /// The old `PredictorNetwork::new` used `Array2::zeros`, so the feature map
    /// output `tanh(0) = 0` for every input.
    #[test]
    fn hidden_layers_are_not_zero_initialized() {
        let net = network();
        let phi_a = net
            .feature_map(&features(0.1, 0.9).to_vec::<f64>())
            .expect("map");
        let phi_b = net
            .feature_map(&features(0.9, 0.1).to_vec::<f64>())
            .expect("map");
        let magnitude = phi_a.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            magnitude > 1e-6,
            "feature map collapsed to zero (max |phi| {magnitude})"
        );
        let delta = phi_a
            .iter()
            .zip(phi_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            delta > 1e-6,
            "feature map ignores its input (delta {delta})"
        );
    }

    #[test]
    fn untrained_predictor_reports_no_information() {
        let net = network();
        let out = net.predict(&features(0.5, 0.5)).expect("predict");
        assert_eq!(out.convergence_improvement, 0.0);
        assert_eq!(out.confidence, 0.0);
        assert_eq!(out.uncertainty, 1.0);
    }

    /// The headline requirement: different inputs must give different outputs.
    #[test]
    fn different_inputs_give_different_predictions() {
        let mut net = network();
        net.fit(&training_set(80)).expect("fit");

        let easy = net.predict(&features(0.05, 0.05)).expect("predict easy");
        let hard = net.predict(&features(0.95, 0.95)).expect("predict hard");

        assert!(
            (easy.convergence_improvement - hard.convergence_improvement).abs() > 1e-4,
            "convergence prediction is constant: {} vs {}",
            easy.convergence_improvement,
            hard.convergence_improvement
        );
        assert!(
            (easy.final_performance - hard.final_performance).abs() > 1e-4,
            "performance prediction is constant: {} vs {}",
            easy.final_performance,
            hard.final_performance
        );
        // And the direction has to be right: an easier landscape must predict a
        // larger improvement.
        assert!(
            easy.convergence_improvement > hard.convergence_improvement,
            "easy {} should beat hard {}",
            easy.convergence_improvement,
            hard.convergence_improvement
        );
    }

    /// The second headline requirement: training must reduce prediction error.
    #[test]
    fn training_reduces_held_out_prediction_error() {
        let mut net = network();
        let train = training_set(90);
        let held_out: Vec<PredictorSample> = (0..25)
            .map(|i| {
                let c = ((i as f64 + 0.5) * 0.061).fract();
                let d = ((i as f64 + 0.5) * 0.113).fract();
                let f = features(c, d);
                let (ci, fp) = truth(&f);
                PredictorSample {
                    features: f,
                    convergence_improvement: ci,
                    final_performance: fp,
                }
            })
            .collect();

        let error = |net: &PredictorNetwork<f64>| -> f64 {
            let mut acc = 0.0;
            for s in &held_out {
                let p = net.predict(&s.features).expect("predict");
                acc += (p.convergence_improvement - s.convergence_improvement).powi(2);
                acc += (p.final_performance - s.final_performance).powi(2);
            }
            (acc / (2.0 * held_out.len() as f64)).sqrt()
        };

        let before = error(&net);
        let report = net.fit(&train).expect("fit");
        let after = error(&net);

        assert_eq!(report.samples, train.len());
        assert!(
            after < before,
            "training did not reduce held-out RMSE: {before} -> {after}"
        );
        assert!(
            after < 0.02,
            "held-out RMSE {after} is too large for a learnable target"
        );
        assert!(
            report.convergence_rmse < 0.02,
            "train RMSE {}",
            report.convergence_rmse
        );
    }

    #[test]
    fn training_produces_real_uncertainty_and_confidence() {
        let mut net = network();
        net.fit(&training_set(80)).expect("fit");
        let out = net.predict(&features(0.4, 0.6)).expect("predict");
        assert!(out.uncertainty.is_finite() && out.uncertainty >= 0.0);
        assert!(out.confidence > 0.0 && out.confidence <= 1.0);
        // Confidence must be the documented function of uncertainty, not 0.85.
        assert!((out.confidence - 1.0 / (1.0 + out.uncertainty)).abs() < 1e-12);
    }

    #[test]
    fn fit_rejects_an_empty_training_set() {
        let mut net = network();
        assert!(net.fit(&[]).is_err());
    }

    #[test]
    fn feature_vector_has_the_declared_width() {
        assert_eq!(
            features(0.3, 0.4).to_vec::<f64>().len(),
            PREDICTION_FEATURE_COUNT
        );
    }

    #[test]
    fn cache_tracks_hits_and_evicts_at_capacity() {
        let mut cache = PredictionCache::new(2);
        let out = PredictorOutput {
            convergence_improvement: 0.1,
            final_performance: 0.9,
            uncertainty: 0.05,
            confidence: 0.95,
        };
        assert!(cache.get("a").is_none());
        cache.insert("a".to_string(), out.clone());
        cache.insert("b".to_string(), out.clone());
        assert_eq!(cache.len(), 2);
        cache.insert("c".to_string(), out);
        assert_eq!(cache.len(), 2, "capacity was not enforced");
        assert!(cache.get("a").is_none(), "oldest entry should be evicted");
        assert!(cache.get("c").is_some());
        assert!(cache.hit_rate() > 0.0 && cache.hit_rate() < 1.0);
    }

    #[test]
    fn zero_capacity_cache_stores_nothing() {
        let mut cache = PredictionCache::new(0);
        cache.insert(
            "a".to_string(),
            PredictorOutput {
                convergence_improvement: 0.0,
                final_performance: 0.0,
                uncertainty: 0.0,
                confidence: 1.0,
            },
        );
        assert!(cache.is_empty());
    }

    #[test]
    fn cholesky_solves_a_known_system() {
        // A = [[4, 2], [2, 3]], b = [2, 1] -> x = [0.25, 0.1666...]
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).expect("shape");
        let b = Array1::from_vec(vec![2.0, 1.0]);
        let l = cholesky(&a).expect("factor");
        let x = cholesky_solve(&l, &b);
        let residual = a.dot(&x) - &b;
        let err = residual.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(err < 1e-12, "residual {err}");

        let inv = cholesky_inverse(&l);
        let identity = a.dot(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((identity[[i, j]] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cholesky_rejects_indefinite_matrices() {
        let a = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("shape");
        assert!(cholesky(&a).is_err());
    }
}
