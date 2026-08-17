// Analytically differentiable linear policy / value / Q models.
//
// The optimizers in this module need real gradients. OptiRS is deliberately not
// an autodiff framework, so instead of fabricating gradients this file provides
// concrete, fully analytic models that close the loop end to end:
//
// * [`LinearSoftmaxPolicy`]  — discrete actions, `π(a|s) = softmax(W φ(s))`
// * [`LinearGaussianPolicy`] — continuous actions, `a ~ N(W φ(s), diag(σ²))`
// * [`LinearValueFunction`]  — `V(s) = w · φ(s)`
// * [`LinearQFunction`]      — `Q(s,a) = w · [s ; a ; vec(s aᵀ)]`
//
// Every gradient below is derived in closed form and unit-tested against finite
// differences, so REINFORCE / A2C / PPO / DDPG / TD3 / SAC genuinely learn when
// driven with these models. They are also the reference implementations of the
// gradient-oracle contract documented on [`PolicyNetwork`] / [`ValueNetwork`] /
// [`QNetwork`], which any user-supplied (e.g. autodiff-backed) network can follow.
//
// Feature engineering (bias terms, tile coding, polynomial features, …) is the
// caller's responsibility: `φ(s)` is simply the observation row that is handed in.

use super::{
    ActionDistribution, DistributionType, KroneckerBlock, PolicyEvaluation, PolicyNetwork,
    QNetwork, ValueNetwork,
};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Canonical key used for the weight matrix of every model in this file.
const WEIGHTS: &str = "weights";

/// Canonical key for the (log) standard deviation of [`LinearGaussianPolicy`].
const LOG_STD: &str = "log_std";

/// Smallest standard deviation a Gaussian policy is allowed to use.
///
/// Guards `1/σ`, `ln σ` and the reparameterization against a collapsed policy.
fn min_sigma<T: Float>() -> T {
    T::from(1e-6).unwrap_or_else(T::epsilon)
}

fn scalar<T: Float>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::zero)
}

/// Convert a flat row-major buffer into a named-parameter map entry.
fn flat_from_matrix<T: Float + Debug + Send + Sync + 'static>(matrix: &Array2<T>) -> Array1<T> {
    let (rows, cols) = matrix.dim();
    let mut flat = Array1::zeros(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            flat[r * cols + c] = matrix[[r, c]];
        }
    }
    flat
}

/// Add a flat row-major delta onto a matrix, validating its length.
fn add_flat_to_matrix<T: Float + Debug + Send + Sync + 'static>(
    matrix: &mut Array2<T>,
    delta: &Array1<T>,
    what: &str,
) -> Result<()> {
    let (rows, cols) = matrix.dim();
    if delta.len() != rows * cols {
        return Err(OptimError::DimensionMismatch(format!(
            "{what} delta length ({}) does not match parameter size ({}x{})",
            delta.len(),
            rows,
            cols
        )));
    }
    for r in 0..rows {
        for c in 0..cols {
            matrix[[r, c]] = matrix[[r, c]] + delta[r * cols + c];
        }
    }
    Ok(())
}

/// Validate that a batch of observations/actions/coefficients is self-consistent.
fn check_batch<T: Float + Debug + Send + Sync + 'static>(
    observations: &Array2<T>,
    n_features: usize,
    actions: Option<(&Array2<T>, usize)>,
    coefficients: Option<&Array1<T>>,
) -> Result<usize> {
    let n = observations.nrows();
    if observations.ncols() != n_features {
        return Err(OptimError::DimensionMismatch(format!(
            "observation dimension ({}) does not match model feature dimension ({})",
            observations.ncols(),
            n_features
        )));
    }
    if let Some((actions, expected)) = actions {
        if actions.nrows() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "action batch ({}) does not match observation batch ({})",
                actions.nrows(),
                n
            )));
        }
        if actions.ncols() != expected {
            return Err(OptimError::DimensionMismatch(format!(
                "action dimension ({}) does not match model action dimension ({})",
                actions.ncols(),
                expected
            )));
        }
    }
    if let Some(coefficients) = coefficients {
        if coefficients.len() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "coefficient count ({}) does not match batch size ({})",
                coefficients.len(),
                n
            )));
        }
    }
    Ok(n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear softmax policy (discrete actions)
// ─────────────────────────────────────────────────────────────────────────────

/// Categorical policy `π(a|s) = softmax(W φ(s))` with an analytic score function.
///
/// Actions are one-hot rows of width `n_actions`. The score function is the
/// textbook softmax result
///
/// ```text
/// ∇_W log π(a|s) = (e_a − p(s)) φ(s)ᵀ ,  p(s) = softmax(W φ(s))
/// ```
///
/// which is exactly rank one — so this policy is also an exact source of
/// Kronecker factors for K-FAC (`φ` is the input factor, `e_a − p` the output
/// factor).
#[derive(Debug, Clone)]
pub struct LinearSoftmaxPolicy<T: Float + Debug + Send + Sync + 'static> {
    /// Logit weights, shape `(n_actions, n_features)`.
    weights: Array2<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> LinearSoftmaxPolicy<T> {
    /// Create a zero-initialized policy (uniform over actions).
    pub fn new(n_actions: usize, n_features: usize) -> Result<Self> {
        if n_actions == 0 || n_features == 0 {
            return Err(OptimError::InvalidConfig(
                "LinearSoftmaxPolicy requires n_actions > 0 and n_features > 0".to_string(),
            ));
        }
        Ok(Self {
            weights: Array2::zeros((n_actions, n_features)),
        })
    }

    /// Create a policy from an explicit weight matrix `(n_actions, n_features)`.
    pub fn from_weights(weights: Array2<T>) -> Result<Self> {
        if weights.nrows() == 0 || weights.ncols() == 0 {
            return Err(OptimError::InvalidConfig(
                "LinearSoftmaxPolicy weights must be non-empty".to_string(),
            ));
        }
        Ok(Self { weights })
    }

    /// Number of discrete actions.
    pub fn n_actions(&self) -> usize {
        self.weights.nrows()
    }

    /// Observation/feature dimension.
    pub fn n_features(&self) -> usize {
        self.weights.ncols()
    }

    /// Current logit weights.
    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }

    /// Logits `W φ(s)` for a batch, shape `(n, n_actions)`.
    pub fn logits(&self, observations: &Array2<T>) -> Result<Array2<T>> {
        let n = check_batch(observations, self.n_features(), None, None)?;
        let mut logits = Array2::zeros((n, self.n_actions()));
        for i in 0..n {
            for a in 0..self.n_actions() {
                let mut acc = T::zero();
                for f in 0..self.n_features() {
                    acc = acc + self.weights[[a, f]] * observations[[i, f]];
                }
                logits[[i, a]] = acc;
            }
        }
        Ok(logits)
    }

    /// Action probabilities for a batch, shape `(n, n_actions)`.
    pub fn probabilities(&self, observations: &Array2<T>) -> Result<Array2<T>> {
        let logits = self.logits(observations)?;
        Ok(softmax_rows(&logits))
    }

    /// One-hot encode an action index.
    pub fn one_hot(&self, index: usize) -> Result<Array1<T>> {
        if index >= self.n_actions() {
            return Err(OptimError::InvalidParameter(format!(
                "action index {index} out of range for {} actions",
                self.n_actions()
            )));
        }
        let mut row = Array1::zeros(self.n_actions());
        row[index] = T::one();
        Ok(row)
    }

    /// Sample one action per observation using a caller-supplied uniform source.
    ///
    /// Returns the one-hot action matrix together with the log-probabilities of
    /// the sampled actions — exactly the two arrays a
    /// [`super::TrajectoryBatch`] needs. Taking the uniform source as a closure
    /// keeps rollouts fully reproducible in tests.
    pub fn sample_actions_with(
        &self,
        observations: &Array2<T>,
        mut uniform: impl FnMut() -> f64,
    ) -> Result<(Array2<T>, Array1<T>, Vec<usize>)> {
        let probs = self.probabilities(observations)?;
        let n = probs.nrows();
        let n_actions = self.n_actions();

        let mut actions = Array2::zeros((n, n_actions));
        let mut log_probs = Array1::zeros(n);
        let mut indices = Vec::with_capacity(n);

        for i in 0..n {
            let u = scalar::<T>(uniform());
            let mut cumulative = T::zero();
            // Default to the last action so floating point drift in the CDF can
            // never leave the index unset.
            let mut chosen = n_actions - 1;
            for a in 0..n_actions {
                cumulative = cumulative + probs[[i, a]];
                if u <= cumulative {
                    chosen = a;
                    break;
                }
            }
            actions[[i, chosen]] = T::one();
            log_probs[i] = safe_ln(probs[[i, chosen]]);
            indices.push(chosen);
        }

        Ok((actions, log_probs, indices))
    }

    /// Index of the (one-hot) action stored in row `i`.
    fn action_index(&self, actions: &Array2<T>, i: usize) -> usize {
        let mut best = 0usize;
        let mut best_value = actions[[i, 0]];
        for a in 1..actions.ncols() {
            if actions[[i, a]] > best_value {
                best_value = actions[[i, a]];
                best = a;
            }
        }
        best
    }

    /// Per-sample softmax "delta" `e_a − p`, shape `(n, n_actions)`.
    fn deltas(&self, observations: &Array2<T>, actions: &Array2<T>) -> Result<Array2<T>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((actions, self.n_actions())),
            None,
        )?;
        let probs = self.probabilities(observations)?;
        let mut deltas = Array2::zeros((n, self.n_actions()));
        for i in 0..n {
            let chosen = self.action_index(actions, i);
            for a in 0..self.n_actions() {
                let indicator = if a == chosen { T::one() } else { T::zero() };
                deltas[[i, a]] = indicator - probs[[i, a]];
            }
        }
        Ok(deltas)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> PolicyNetwork<T> for LinearSoftmaxPolicy<T> {
    fn evaluate_actions(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<PolicyEvaluation<T>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((actions, self.n_actions())),
            None,
        )?;
        let probs = self.probabilities(observations)?;

        let mut log_probs = Array1::zeros(n);
        let mut entropy = Array1::zeros(n);
        for i in 0..n {
            let chosen = self.action_index(actions, i);
            log_probs[i] = safe_ln(probs[[i, chosen]]);

            let mut h = T::zero();
            for a in 0..self.n_actions() {
                let p = probs[[i, a]];
                if p > T::zero() {
                    h = h - p * p.ln();
                }
            }
            entropy[i] = h;
        }

        Ok(PolicyEvaluation {
            log_probs,
            entropy,
            metrics: HashMap::new(),
        })
    }

    fn get_action_distribution(&self, observations: &Array2<T>) -> Result<ActionDistribution<T>> {
        Ok(ActionDistribution {
            mean: None,
            std: None,
            logits: Some(self.logits(observations)?),
            distribution_type: DistributionType::Categorical,
        })
    }

    fn update_parameters(&mut self, deltas: &HashMap<String, Array1<T>>) -> Result<()> {
        if let Some(delta) = deltas.get(WEIGHTS) {
            add_flat_to_matrix(&mut self.weights, delta, "LinearSoftmaxPolicy weights")?;
        }
        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, Array1<T>> {
        let mut map = HashMap::with_capacity(1);
        map.insert(WEIGHTS.to_string(), flat_from_matrix(&self.weights));
        map
    }

    fn log_prob_gradient(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
        coefficients: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((actions, self.n_actions())),
            Some(coefficients),
        )?;
        let deltas = self.deltas(observations, actions)?;

        let mut grad = Array2::zeros((self.n_actions(), self.n_features()));
        for i in 0..n {
            let c = coefficients[i];
            if c == T::zero() {
                continue;
            }
            for a in 0..self.n_actions() {
                let d = c * deltas[[i, a]];
                if d == T::zero() {
                    continue;
                }
                for f in 0..self.n_features() {
                    grad[[a, f]] = grad[[a, f]] + d * observations[[i, f]];
                }
            }
        }

        let mut map = HashMap::with_capacity(1);
        map.insert(WEIGHTS.to_string(), flat_from_matrix(&grad));
        Ok(map)
    }

    fn entropy_gradient(&self, observations: &Array2<T>) -> Result<HashMap<String, Array1<T>>> {
        let n = check_batch(observations, self.n_features(), None, None)?;
        let probs = self.probabilities(observations)?;

        let mut grad = Array2::zeros((self.n_actions(), self.n_features()));
        if n == 0 {
            let mut map = HashMap::with_capacity(1);
            map.insert(WEIGHTS.to_string(), flat_from_matrix(&grad));
            return Ok(map);
        }
        let inv_n = T::one() / scalar::<T>(n as f64);

        for i in 0..n {
            // H_i = −Σ_k p_k ln p_k  and  ∂H_i/∂z_j = −p_j (ln p_j + H_i)
            let mut h = T::zero();
            for a in 0..self.n_actions() {
                let p = probs[[i, a]];
                if p > T::zero() {
                    h = h - p * p.ln();
                }
            }
            for a in 0..self.n_actions() {
                let p = probs[[i, a]];
                if p <= T::zero() {
                    continue;
                }
                let d = -p * (p.ln() + h) * inv_n;
                for f in 0..self.n_features() {
                    grad[[a, f]] = grad[[a, f]] + d * observations[[i, f]];
                }
            }
        }

        let mut map = HashMap::with_capacity(1);
        map.insert(WEIGHTS.to_string(), flat_from_matrix(&grad));
        Ok(map)
    }

    fn score_matrix(&self, observations: &Array2<T>, actions: &Array2<T>) -> Result<Array2<T>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((actions, self.n_actions())),
            None,
        )?;
        let deltas = self.deltas(observations, actions)?;

        let dim = self.n_actions() * self.n_features();
        let mut scores = Array2::zeros((n, dim));
        for i in 0..n {
            for a in 0..self.n_actions() {
                let d = deltas[[i, a]];
                for f in 0..self.n_features() {
                    scores[[i, a * self.n_features() + f]] = d * observations[[i, f]];
                }
            }
        }
        Ok(scores)
    }

    fn kronecker_factors(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<Vec<KroneckerBlock<T>>> {
        let deltas = self.deltas(observations, actions)?;
        Ok(vec![KroneckerBlock {
            name: WEIGHTS.to_string(),
            inputs: observations.clone(),
            outputs: deltas,
        }])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear Gaussian policy (continuous actions)
// ─────────────────────────────────────────────────────────────────────────────

/// Diagonal Gaussian policy `a ~ N(W φ(s), diag(σ²))` with `σ = exp(log_std)`.
///
/// Both the mean weights and the (state-independent) log standard deviations are
/// trainable parameters, exposed as `"weights"` and `"log_std"`. All gradients are
/// closed form:
///
/// ```text
/// ∇_W      log π(a|s) = ((a − μ)/σ²) φ(s)ᵀ
/// ∇_logσ_j log π(a|s) = (a_j − μ_j)²/σ_j² − 1
/// ∇_logσ_j H[π]       = 1                      ∇_W H[π] = 0
/// ∇_W  Σ_ij w_ij μ_j(s_i) = Σ_i w_i φ(s_i)ᵀ    (deterministic-policy-gradient hook)
/// ```
#[derive(Debug, Clone)]
pub struct LinearGaussianPolicy<T: Float + Debug + Send + Sync + 'static> {
    /// Mean weights, shape `(action_dim, n_features)`.
    weights: Array2<T>,
    /// Log standard deviations, length `action_dim`.
    log_std: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> LinearGaussianPolicy<T> {
    /// Create a zero-mean policy with the given initial standard deviation.
    pub fn new(action_dim: usize, n_features: usize, init_std: T) -> Result<Self> {
        if action_dim == 0 || n_features == 0 {
            return Err(OptimError::InvalidConfig(
                "LinearGaussianPolicy requires action_dim > 0 and n_features > 0".to_string(),
            ));
        }
        if !matches!(
            init_std.partial_cmp(&T::zero()),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(OptimError::InvalidConfig(
                "LinearGaussianPolicy requires init_std > 0".to_string(),
            ));
        }
        Ok(Self {
            weights: Array2::zeros((action_dim, n_features)),
            log_std: Array1::from_elem(action_dim, init_std.ln()),
        })
    }

    /// Action dimension.
    pub fn action_dim(&self) -> usize {
        self.weights.nrows()
    }

    /// Observation/feature dimension.
    pub fn n_features(&self) -> usize {
        self.weights.ncols()
    }

    /// Current mean weights.
    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }

    /// Current standard deviations (clamped away from zero).
    pub fn std(&self) -> Array1<T> {
        self.log_std.mapv(|l| l.exp().max(min_sigma::<T>()))
    }

    /// Deterministic mean action `μ(s) = W φ(s)`, shape `(n, action_dim)`.
    pub fn mean_actions(&self, observations: &Array2<T>) -> Result<Array2<T>> {
        let n = check_batch(observations, self.n_features(), None, None)?;
        let mut mean = Array2::zeros((n, self.action_dim()));
        for i in 0..n {
            for a in 0..self.action_dim() {
                let mut acc = T::zero();
                for f in 0..self.n_features() {
                    acc = acc + self.weights[[a, f]] * observations[[i, f]];
                }
                mean[[i, a]] = acc;
            }
        }
        Ok(mean)
    }

    /// Sample actions with a caller-supplied standard-normal source.
    ///
    /// Returns the sampled actions and their log-probabilities.
    pub fn sample_actions_with(
        &self,
        observations: &Array2<T>,
        mut standard_normal: impl FnMut() -> f64,
    ) -> Result<(Array2<T>, Array1<T>)> {
        let mean = self.mean_actions(observations)?;
        let std = self.std();
        let n = mean.nrows();

        let mut actions = Array2::zeros((n, self.action_dim()));
        for i in 0..n {
            for a in 0..self.action_dim() {
                let z = scalar::<T>(standard_normal());
                actions[[i, a]] = mean[[i, a]] + std[a] * z;
            }
        }

        let evaluation = self.evaluate_actions(observations, &actions)?;
        Ok((actions, evaluation.log_probs))
    }

    /// `(a − μ)/σ²` per sample — the mean-direction score factor.
    fn standardized_residual(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<Array2<T>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((actions, self.action_dim())),
            None,
        )?;
        let mean = self.mean_actions(observations)?;
        let std = self.std();

        let mut residual = Array2::zeros((n, self.action_dim()));
        for i in 0..n {
            for a in 0..self.action_dim() {
                let sigma = std[a];
                residual[[i, a]] = (actions[[i, a]] - mean[[i, a]]) / (sigma * sigma);
            }
        }
        Ok(residual)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> PolicyNetwork<T> for LinearGaussianPolicy<T> {
    fn evaluate_actions(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<PolicyEvaluation<T>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((actions, self.action_dim())),
            None,
        )?;
        let mean = self.mean_actions(observations)?;
        let std = self.std();

        let half = scalar::<T>(0.5);
        let half_log_two_pi = scalar::<T>(0.5 * (2.0 * std::f64::consts::PI).ln());
        // H[N(μ, σ²)] = Σ_j (ln σ_j + ½ ln(2πe))
        let half_log_two_pi_e = scalar::<T>(0.5 * (2.0 * std::f64::consts::PI * 1.0f64.exp()).ln());

        let mut log_probs = Array1::zeros(n);
        let mut entropy_value = T::zero();
        for a in 0..self.action_dim() {
            entropy_value = entropy_value + std[a].ln() + half_log_two_pi_e;
        }

        for i in 0..n {
            let mut lp = T::zero();
            for a in 0..self.action_dim() {
                let sigma = std[a];
                let z = (actions[[i, a]] - mean[[i, a]]) / sigma;
                lp = lp - half * z * z - sigma.ln() - half_log_two_pi;
            }
            log_probs[i] = lp;
        }

        Ok(PolicyEvaluation {
            log_probs,
            entropy: Array1::from_elem(n, entropy_value),
            metrics: HashMap::new(),
        })
    }

    fn get_action_distribution(&self, observations: &Array2<T>) -> Result<ActionDistribution<T>> {
        let mean = self.mean_actions(observations)?;
        let std_vec = self.std();
        let mut std = Array2::zeros(mean.dim());
        for i in 0..mean.nrows() {
            for a in 0..self.action_dim() {
                std[[i, a]] = std_vec[a];
            }
        }
        Ok(ActionDistribution {
            mean: Some(mean),
            std: Some(std),
            logits: None,
            distribution_type: DistributionType::Gaussian,
        })
    }

    fn update_parameters(&mut self, deltas: &HashMap<String, Array1<T>>) -> Result<()> {
        if let Some(delta) = deltas.get(WEIGHTS) {
            add_flat_to_matrix(&mut self.weights, delta, "LinearGaussianPolicy weights")?;
        }
        if let Some(delta) = deltas.get(LOG_STD) {
            if delta.len() != self.log_std.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "LinearGaussianPolicy log_std delta length ({}) does not match action_dim ({})",
                    delta.len(),
                    self.log_std.len()
                )));
            }
            for a in 0..self.log_std.len() {
                self.log_std[a] = self.log_std[a] + delta[a];
            }
        }
        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, Array1<T>> {
        let mut map = HashMap::with_capacity(2);
        map.insert(WEIGHTS.to_string(), flat_from_matrix(&self.weights));
        map.insert(LOG_STD.to_string(), self.log_std.clone());
        map
    }

    fn log_prob_gradient(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
        coefficients: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((actions, self.action_dim())),
            Some(coefficients),
        )?;
        let mean = self.mean_actions(observations)?;
        let std = self.std();

        let mut grad_w = Array2::zeros((self.action_dim(), self.n_features()));
        let mut grad_log_std = Array1::zeros(self.action_dim());

        for i in 0..n {
            let c = coefficients[i];
            if c == T::zero() {
                continue;
            }
            for a in 0..self.action_dim() {
                let sigma = std[a];
                let diff = actions[[i, a]] - mean[[i, a]];
                let z = diff / sigma;

                // ∂ log π / ∂ μ_a = (a − μ)/σ²
                let dmu = c * diff / (sigma * sigma);
                for f in 0..self.n_features() {
                    grad_w[[a, f]] = grad_w[[a, f]] + dmu * observations[[i, f]];
                }

                // ∂ log π / ∂ log σ_a = z² − 1
                grad_log_std[a] = grad_log_std[a] + c * (z * z - T::one());
            }
        }

        let mut map = HashMap::with_capacity(2);
        map.insert(WEIGHTS.to_string(), flat_from_matrix(&grad_w));
        map.insert(LOG_STD.to_string(), grad_log_std);
        Ok(map)
    }

    fn entropy_gradient(&self, observations: &Array2<T>) -> Result<HashMap<String, Array1<T>>> {
        check_batch(observations, self.n_features(), None, None)?;
        // H = Σ_j (log σ_j + ½ ln 2πe) is state-independent: ∂H/∂log σ = 1, ∂H/∂W = 0.
        let mut map = HashMap::with_capacity(2);
        map.insert(
            WEIGHTS.to_string(),
            Array1::zeros(self.action_dim() * self.n_features()),
        );
        map.insert(
            LOG_STD.to_string(),
            Array1::from_elem(self.action_dim(), T::one()),
        );
        Ok(map)
    }

    fn mean_action_gradient(
        &self,
        observations: &Array2<T>,
        weights: &Array2<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let n = check_batch(
            observations,
            self.n_features(),
            Some((weights, self.action_dim())),
            None,
        )?;

        let mut grad_w = Array2::zeros((self.action_dim(), self.n_features()));
        for i in 0..n {
            for a in 0..self.action_dim() {
                let w = weights[[i, a]];
                if w == T::zero() {
                    continue;
                }
                for f in 0..self.n_features() {
                    grad_w[[a, f]] = grad_w[[a, f]] + w * observations[[i, f]];
                }
            }
        }

        let mut map = HashMap::with_capacity(2);
        map.insert(WEIGHTS.to_string(), flat_from_matrix(&grad_w));
        map.insert(LOG_STD.to_string(), Array1::zeros(self.action_dim()));
        Ok(map)
    }

    fn kronecker_factors(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<Vec<KroneckerBlock<T>>> {
        let residual = self.standardized_residual(observations, actions)?;
        let n = residual.nrows();
        let std = self.std();

        // log_std behaves as a linear layer with a constant unit input, so its
        // score reshaped to (action_dim, 1) is exactly δ · [1].
        let mut log_std_outputs = Array2::zeros((n, self.action_dim()));
        let mean = self.mean_actions(observations)?;
        for i in 0..n {
            for a in 0..self.action_dim() {
                let z = (actions[[i, a]] - mean[[i, a]]) / std[a];
                log_std_outputs[[i, a]] = z * z - T::one();
            }
        }

        Ok(vec![
            KroneckerBlock {
                name: WEIGHTS.to_string(),
                inputs: observations.clone(),
                outputs: residual,
            },
            KroneckerBlock {
                name: LOG_STD.to_string(),
                inputs: Array2::from_elem((n, 1), T::one()),
                outputs: log_std_outputs,
            },
        ])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear value function
// ─────────────────────────────────────────────────────────────────────────────

/// State-value function `V(s) = w · φ(s)` with the exact gradient `∇_w V = φ(s)`.
#[derive(Debug, Clone)]
pub struct LinearValueFunction<T: Float + Debug + Send + Sync + 'static> {
    weights: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> LinearValueFunction<T> {
    /// Create a zero-initialized value function over `n_features` features.
    pub fn new(n_features: usize) -> Result<Self> {
        if n_features == 0 {
            return Err(OptimError::InvalidConfig(
                "LinearValueFunction requires n_features > 0".to_string(),
            ));
        }
        Ok(Self {
            weights: Array1::zeros(n_features),
        })
    }

    /// Feature dimension.
    pub fn n_features(&self) -> usize {
        self.weights.len()
    }

    /// Current weights.
    pub fn weights(&self) -> &Array1<T> {
        &self.weights
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ValueNetwork<T> for LinearValueFunction<T> {
    fn evaluate_value(&self, observations: &Array2<T>) -> Result<Array1<T>> {
        let n = check_batch(observations, self.n_features(), None, None)?;
        let mut values = Array1::zeros(n);
        for i in 0..n {
            let mut acc = T::zero();
            for f in 0..self.n_features() {
                acc = acc + self.weights[f] * observations[[i, f]];
            }
            values[i] = acc;
        }
        Ok(values)
    }

    fn update_parameters(&mut self, deltas: &HashMap<String, Array1<T>>) -> Result<()> {
        if let Some(delta) = deltas.get(WEIGHTS) {
            if delta.len() != self.weights.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "LinearValueFunction delta length ({}) does not match n_features ({})",
                    delta.len(),
                    self.weights.len()
                )));
            }
            for f in 0..self.weights.len() {
                self.weights[f] = self.weights[f] + delta[f];
            }
        }
        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, Array1<T>> {
        let mut map = HashMap::with_capacity(1);
        map.insert(WEIGHTS.to_string(), self.weights.clone());
        map
    }

    fn value_gradient(
        &self,
        observations: &Array2<T>,
        residuals: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let n = check_batch(observations, self.n_features(), None, Some(residuals))?;
        let mut grad = Array1::zeros(self.n_features());
        for i in 0..n {
            let r = residuals[i];
            if r == T::zero() {
                continue;
            }
            for f in 0..self.n_features() {
                grad[f] = grad[f] + r * observations[[i, f]];
            }
        }
        let mut map = HashMap::with_capacity(1);
        map.insert(WEIGHTS.to_string(), grad);
        Ok(map)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear Q function
// ─────────────────────────────────────────────────────────────────────────────

/// Action-value function linear in the state-action features
///
/// ```text
/// φ(s,a) = [ s ; a ; vec(s aᵀ) ]        Q(s,a) = w · φ(s,a)
/// ```
///
/// The bilinear `s aᵀ` block is what makes the deterministic policy gradient
/// non-trivial: without it `∇_a Q` would be a constant vector and every DPG step
/// would push the actor to the same saturated action regardless of the state.
///
/// [`ValueNetwork::evaluate_value`] deliberately returns
/// [`OptimError::UnsupportedOperation`]: a Q function has no state value without a
/// policy to average over, and silently returning `Q(s, 0)` would be a trap.
#[derive(Debug, Clone)]
pub struct LinearQFunction<T: Float + Debug + Send + Sync + 'static> {
    weights: Array1<T>,
    state_dim: usize,
    action_dim: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> LinearQFunction<T> {
    /// Create a zero-initialized Q function.
    pub fn new(state_dim: usize, action_dim: usize) -> Result<Self> {
        if state_dim == 0 || action_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "LinearQFunction requires state_dim > 0 and action_dim > 0".to_string(),
            ));
        }
        Ok(Self {
            weights: Array1::zeros(state_dim + action_dim + state_dim * action_dim),
            state_dim,
            action_dim,
        })
    }

    /// State dimension.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Action dimension.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Current weights over the state-action feature vector.
    pub fn weights(&self) -> &Array1<T> {
        &self.weights
    }

    /// Offset of the bilinear `vec(s aᵀ)` block inside the feature vector.
    fn bilinear_offset(&self) -> usize {
        self.state_dim + self.action_dim
    }

    /// Build `φ(s, a)` for row `i`.
    fn features(&self, states: &Array2<T>, actions: &Array2<T>, i: usize) -> Array1<T> {
        let mut phi = Array1::zeros(self.weights.len());
        for s in 0..self.state_dim {
            phi[s] = states[[i, s]];
        }
        for a in 0..self.action_dim {
            phi[self.state_dim + a] = actions[[i, a]];
        }
        let base = self.bilinear_offset();
        for s in 0..self.state_dim {
            for a in 0..self.action_dim {
                phi[base + s * self.action_dim + a] = states[[i, s]] * actions[[i, a]];
            }
        }
        phi
    }

    fn check_pairs(&self, states: &Array2<T>, actions: &Array2<T>) -> Result<usize> {
        check_batch(
            states,
            self.state_dim,
            Some((actions, self.action_dim)),
            None,
        )
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ValueNetwork<T> for LinearQFunction<T> {
    fn evaluate_value(&self, observations: &Array2<T>) -> Result<Array1<T>> {
        let _ = observations;
        Err(OptimError::UnsupportedOperation(
            "LinearQFunction is an action-value critic: use QNetwork::evaluate_q(states, actions). \
             A state value V(s) is only defined relative to a policy."
                .to_string(),
        ))
    }

    fn update_parameters(&mut self, deltas: &HashMap<String, Array1<T>>) -> Result<()> {
        if let Some(delta) = deltas.get(WEIGHTS) {
            if delta.len() != self.weights.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "LinearQFunction delta length ({}) does not match feature count ({})",
                    delta.len(),
                    self.weights.len()
                )));
            }
            for f in 0..self.weights.len() {
                self.weights[f] = self.weights[f] + delta[f];
            }
        }
        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, Array1<T>> {
        let mut map = HashMap::with_capacity(1);
        map.insert(WEIGHTS.to_string(), self.weights.clone());
        map
    }

    fn value_gradient(
        &self,
        observations: &Array2<T>,
        residuals: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let _ = (observations, residuals);
        Err(OptimError::UnsupportedOperation(
            "LinearQFunction is an action-value critic: use QNetwork::q_gradient".to_string(),
        ))
    }
}

impl<T: Float + Debug + Send + Sync + 'static> QNetwork<T> for LinearQFunction<T> {
    fn evaluate_q(&self, states: &Array2<T>, actions: &Array2<T>) -> Result<Array1<T>> {
        let n = self.check_pairs(states, actions)?;
        let mut q = Array1::zeros(n);
        for i in 0..n {
            let phi = self.features(states, actions, i);
            let mut acc = T::zero();
            for f in 0..self.weights.len() {
                acc = acc + self.weights[f] * phi[f];
            }
            q[i] = acc;
        }
        Ok(q)
    }

    fn q_gradient(
        &self,
        states: &Array2<T>,
        actions: &Array2<T>,
        residuals: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let n = self.check_pairs(states, actions)?;
        if residuals.len() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "residual count ({}) does not match batch size ({n})",
                residuals.len()
            )));
        }

        let mut grad = Array1::zeros(self.weights.len());
        for i in 0..n {
            let r = residuals[i];
            if r == T::zero() {
                continue;
            }
            let phi = self.features(states, actions, i);
            for f in 0..self.weights.len() {
                grad[f] = grad[f] + r * phi[f];
            }
        }

        let mut map = HashMap::with_capacity(1);
        map.insert(WEIGHTS.to_string(), grad);
        Ok(map)
    }

    fn action_gradient(&self, states: &Array2<T>, actions: &Array2<T>) -> Result<Array2<T>> {
        let n = self.check_pairs(states, actions)?;
        let base = self.bilinear_offset();

        let mut grad = Array2::zeros((n, self.action_dim));
        for i in 0..n {
            for a in 0..self.action_dim {
                // ∂Q/∂a_j = w_a[j] + Σ_s w_sa[s, j] · s_s
                let mut acc = self.weights[self.state_dim + a];
                for s in 0..self.state_dim {
                    acc = acc + self.weights[base + s * self.action_dim + a] * states[[i, s]];
                }
                grad[[i, a]] = acc;
            }
        }
        Ok(grad)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Row-wise softmax with the standard max-subtraction for numerical stability.
fn softmax_rows<T: Float + Debug + Send + Sync + 'static>(logits: &Array2<T>) -> Array2<T> {
    let (n, k) = logits.dim();
    let mut probs = Array2::zeros((n, k));
    for i in 0..n {
        let mut max_logit = T::neg_infinity();
        for a in 0..k {
            if logits[[i, a]] > max_logit {
                max_logit = logits[[i, a]];
            }
        }
        let mut sum = T::zero();
        for a in 0..k {
            let e = (logits[[i, a]] - max_logit).exp();
            probs[[i, a]] = e;
            sum = sum + e;
        }
        if sum > T::zero() {
            for a in 0..k {
                probs[[i, a]] = probs[[i, a]] / sum;
            }
        } else {
            // Degenerate (all logits −inf): fall back to a uniform distribution
            // rather than producing NaNs.
            let uniform = T::one() / scalar::<T>(k as f64);
            for a in 0..k {
                probs[[i, a]] = uniform;
            }
        }
    }
    probs
}

/// `ln(x)` floored at a tiny probability so a saturated softmax cannot emit −inf.
fn safe_ln<T: Float>(x: T) -> T {
    let floor = T::from(1e-300).unwrap_or_else(T::min_positive_value);
    x.max(floor).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2};

    /// Central finite difference of a scalar function of the flat parameters.
    fn fd_gradient(len: usize, eps: f64, mut f: impl FnMut(&[f64]) -> f64, at: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; len];
        let mut probe = at.to_vec();
        for i in 0..len {
            probe[i] = at[i] + eps;
            let plus = f(&probe);
            probe[i] = at[i] - eps;
            let minus = f(&probe);
            probe[i] = at[i];
            out[i] = (plus - minus) / (2.0 * eps);
        }
        out
    }

    fn observations() -> Array2<f64> {
        arr2(&[[1.0, 0.5], [0.2, -1.3], [-0.7, 0.9]])
    }

    #[test]
    fn softmax_log_prob_gradient_matches_finite_differences() {
        let obs = observations();
        let actions = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]);
        let coeffs = arr1(&[0.7, -1.1, 0.3]);

        let init = vec![0.3, -0.2, 0.1, 0.4];
        let policy = LinearSoftmaxPolicy::from_weights(
            Array2::from_shape_vec((2, 2), init.clone()).expect("shape"),
        )
        .expect("policy");

        let analytic = policy
            .log_prob_gradient(&obs, &actions, &coeffs)
            .expect("grad");
        let analytic = analytic[WEIGHTS].to_vec();

        let numeric = fd_gradient(
            4,
            1e-6,
            |w| {
                let p = LinearSoftmaxPolicy::from_weights(
                    Array2::from_shape_vec((2, 2), w.to_vec()).expect("shape"),
                )
                .expect("policy");
                let eval = p.evaluate_actions(&obs, &actions).expect("eval");
                eval.log_probs
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(&lp, &c)| c * lp)
                    .sum::<f64>()
            },
            &init,
        );

        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!((a - n).abs() < 1e-6, "analytic {a} vs numeric {n}");
        }
    }

    #[test]
    fn softmax_entropy_gradient_matches_finite_differences() {
        let obs = observations();
        let init = vec![0.3, -0.2, 0.1, 0.4];
        let policy = LinearSoftmaxPolicy::from_weights(
            Array2::from_shape_vec((2, 2), init.clone()).expect("shape"),
        )
        .expect("policy");

        let analytic = policy.entropy_gradient(&obs).expect("grad")[WEIGHTS].to_vec();

        let numeric = fd_gradient(
            4,
            1e-6,
            |w| {
                let p = LinearSoftmaxPolicy::from_weights(
                    Array2::from_shape_vec((2, 2), w.to_vec()).expect("shape"),
                )
                .expect("policy");
                let probs = p.probabilities(&obs).expect("probs");
                let n = probs.nrows() as f64;
                let mut total = 0.0;
                for i in 0..probs.nrows() {
                    for a in 0..probs.ncols() {
                        let pv = probs[[i, a]];
                        if pv > 0.0 {
                            total -= pv * pv.ln();
                        }
                    }
                }
                total / n
            },
            &init,
        );

        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!((a - n).abs() < 1e-6, "analytic {a} vs numeric {n}");
        }
    }

    #[test]
    fn softmax_score_matrix_matches_kronecker_factors() {
        let obs = observations();
        let actions = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]);
        let policy = LinearSoftmaxPolicy::from_weights(
            Array2::from_shape_vec((2, 2), vec![0.3, -0.2, 0.1, 0.4]).expect("shape"),
        )
        .expect("policy");

        let scores = policy.score_matrix(&obs, &actions).expect("scores");
        let blocks = policy.kronecker_factors(&obs, &actions).expect("kfac");
        assert_eq!(blocks.len(), 1);
        let block = &blocks[0];

        // Contract: score row reshaped (n_out, n_in) row-major == δ φᵀ.
        for i in 0..obs.nrows() {
            for a in 0..2 {
                for f in 0..2 {
                    let expected = block.outputs[[i, a]] * block.inputs[[i, f]];
                    assert!((scores[[i, a * 2 + f]] - expected).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn gaussian_log_prob_gradient_matches_finite_differences() {
        let obs = observations();
        let actions = arr2(&[[0.4], [-0.9], [1.2]]);
        let coeffs = arr1(&[1.0, -0.5, 0.25]);

        // Flat layout: sorted keys => "log_std" (1) then "weights" (2).
        let init = vec![-0.3_f64, 0.6, -0.4];
        let build = |p: &[f64]| {
            let mut policy = LinearGaussianPolicy::<f64>::new(1, 2, 1.0).expect("policy");
            let mut deltas = HashMap::new();
            deltas.insert(LOG_STD.to_string(), arr1(&[p[0]]));
            deltas.insert(WEIGHTS.to_string(), arr1(&[p[1], p[2]]));
            policy.update_parameters(&deltas).expect("update");
            policy
        };

        let policy = build(&init);
        let grad = policy
            .log_prob_gradient(&obs, &actions, &coeffs)
            .expect("grad");
        let mut analytic = grad[LOG_STD].to_vec();
        analytic.extend(grad[WEIGHTS].to_vec());

        let numeric = fd_gradient(
            3,
            1e-6,
            |p| {
                let policy = build(p);
                let eval = policy.evaluate_actions(&obs, &actions).expect("eval");
                eval.log_probs
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(&lp, &c)| c * lp)
                    .sum::<f64>()
            },
            &init,
        );

        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!((a - n).abs() < 1e-5, "analytic {a} vs numeric {n}");
        }
    }

    #[test]
    fn gaussian_mean_action_gradient_matches_finite_differences() {
        let obs = observations();
        let weights = arr2(&[[0.5], [-1.0], [2.0]]);

        let init = vec![0.6_f64, -0.4];
        let build = |p: &[f64]| {
            let mut policy = LinearGaussianPolicy::<f64>::new(1, 2, 1.0).expect("policy");
            let mut deltas = HashMap::new();
            deltas.insert(WEIGHTS.to_string(), arr1(&[p[0], p[1]]));
            policy.update_parameters(&deltas).expect("update");
            policy
        };

        let policy = build(&init);
        let analytic = policy.mean_action_gradient(&obs, &weights).expect("grad")[WEIGHTS].to_vec();

        let numeric = fd_gradient(
            2,
            1e-6,
            |p| {
                let policy = build(p);
                let mean = policy.mean_actions(&obs).expect("mean");
                let mut total = 0.0;
                for i in 0..mean.nrows() {
                    total += weights[[i, 0]] * mean[[i, 0]];
                }
                total
            },
            &init,
        );

        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!((a - n).abs() < 1e-6, "analytic {a} vs numeric {n}");
        }
    }

    #[test]
    fn q_function_action_gradient_is_state_dependent() {
        let mut q = LinearQFunction::<f64>::new(2, 1).expect("q");
        // w = [ws(2) | wa(1) | wsa(2)] -> Q = s0 + 2 s1 + 0.5 a + (1.0 s0 - 3.0 s1) a
        let mut deltas = HashMap::new();
        deltas.insert(WEIGHTS.to_string(), arr1(&[1.0, 2.0, 0.5, 1.0, -3.0]));
        q.update_parameters(&deltas).expect("update");

        let states = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let actions = arr2(&[[0.3], [0.3]]);
        let grad = q.action_gradient(&states, &actions).expect("dq/da");

        assert!((grad[[0, 0]] - (0.5 + 1.0)).abs() < 1e-12);
        assert!((grad[[1, 0]] - (0.5 - 3.0)).abs() < 1e-12);
        assert!(
            (grad[[0, 0]] - grad[[1, 0]]).abs() > 1e-6,
            "bilinear features must make dQ/da state dependent"
        );
    }

    #[test]
    fn q_function_gradient_matches_finite_differences() {
        let states = arr2(&[[1.0, 0.5], [-0.3, 0.8]]);
        let actions = arr2(&[[0.4], [-0.6]]);
        let residuals = arr1(&[1.5, -0.5]);

        let init = vec![0.2_f64, -0.1, 0.7, 0.3, -0.4];
        let build = |p: &[f64]| {
            let mut q = LinearQFunction::<f64>::new(2, 1).expect("q");
            let mut deltas = HashMap::new();
            deltas.insert(WEIGHTS.to_string(), Array1::from_vec(p.to_vec()));
            q.update_parameters(&deltas).expect("update");
            q
        };

        let q = build(&init);
        let analytic = q.q_gradient(&states, &actions, &residuals).expect("grad")[WEIGHTS].to_vec();

        let numeric = fd_gradient(
            5,
            1e-6,
            |p| {
                let q = build(p);
                let values = q.evaluate_q(&states, &actions).expect("q");
                values
                    .iter()
                    .zip(residuals.iter())
                    .map(|(&v, &r)| r * v)
                    .sum::<f64>()
            },
            &init,
        );

        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!((a - n).abs() < 1e-6, "analytic {a} vs numeric {n}");
        }
    }

    #[test]
    fn q_function_rejects_state_value_queries() {
        let q = LinearQFunction::<f64>::new(2, 1).expect("q");
        assert!(q.evaluate_value(&arr2(&[[1.0, 2.0]])).is_err());
    }

    #[test]
    fn value_gradient_matches_finite_differences() {
        let obs = observations();
        let residuals = arr1(&[0.5, -1.0, 2.0]);

        let init = vec![0.3_f64, -0.6];
        let build = |p: &[f64]| {
            let mut v = LinearValueFunction::<f64>::new(2).expect("value");
            let mut deltas = HashMap::new();
            deltas.insert(WEIGHTS.to_string(), Array1::from_vec(p.to_vec()));
            v.update_parameters(&deltas).expect("update");
            v
        };

        let v = build(&init);
        let analytic = v.value_gradient(&obs, &residuals).expect("grad")[WEIGHTS].to_vec();

        let numeric = fd_gradient(
            2,
            1e-6,
            |p| {
                let v = build(p);
                let values = v.evaluate_value(&obs).expect("values");
                values
                    .iter()
                    .zip(residuals.iter())
                    .map(|(&val, &r)| r * val)
                    .sum::<f64>()
            },
            &init,
        );

        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!((a - n).abs() < 1e-8, "analytic {a} vs numeric {n}");
        }
    }

    #[test]
    fn softmax_sampling_follows_the_distribution() {
        // logit[0] large => class 0 almost surely.
        let policy = LinearSoftmaxPolicy::from_weights(
            Array2::from_shape_vec((2, 1), vec![10.0, 0.0]).expect("shape"),
        )
        .expect("policy");
        let obs = Array2::from_elem((50, 1), 1.0_f64);

        let mut counter = 0usize;
        let (actions, log_probs, indices) = policy
            .sample_actions_with(&obs, || {
                counter += 1;
                // Deterministic sweep over (0, 1).
                (counter as f64) / 51.0
            })
            .expect("sample");

        assert_eq!(actions.nrows(), 50);
        assert!(indices.iter().all(|&i| i == 0), "class 0 should dominate");
        assert!(log_probs.iter().all(|lp| lp.is_finite()));
    }

    #[test]
    fn dimension_mismatches_are_rejected() {
        let policy = LinearSoftmaxPolicy::<f64>::new(2, 2).expect("policy");
        // Wrong feature dimension.
        assert!(policy.logits(&arr2(&[[1.0, 2.0, 3.0]])).is_err());
        // Wrong action width.
        assert!(policy
            .evaluate_actions(&arr2(&[[1.0, 2.0]]), &arr2(&[[1.0, 0.0, 0.0]]))
            .is_err());
        // Coefficient count mismatch.
        assert!(policy
            .log_prob_gradient(
                &arr2(&[[1.0, 2.0]]),
                &arr2(&[[1.0, 0.0]]),
                &arr1(&[1.0, 2.0])
            )
            .is_err());
    }
}
