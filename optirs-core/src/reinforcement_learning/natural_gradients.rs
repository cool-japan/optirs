// Natural Policy Gradients
//
// This module implements natural policy gradient methods that use the Fisher information
// matrix to precondition policy gradients for more efficient optimization.

use super::{PolicyNetwork, RLOptimizationMetrics, RLOptimizerConfig, TrajectoryBatch};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Natural gradient configuration
#[derive(Debug, Clone)]
pub struct NaturalGradientConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Base RL configuration
    pub base_config: RLOptimizerConfig<T>,

    /// Fisher information matrix estimation method
    pub fisher_method: FisherEstimationMethod,

    /// Damping parameter for Fisher matrix regularization
    pub damping: T,

    /// Fisher matrix update frequency
    pub fisher_update_freq: usize,

    /// Use empirical Fisher information matrix
    pub use_empirical_fisher: bool,

    /// Conjugate gradient parameters
    pub cg_iters: usize,
    pub cg_tolerance: T,

    /// Natural gradient scaling factor
    pub natural_grad_scale: T,

    /// Enable Fisher matrix preconditioning
    pub enable_preconditioning: bool,

    /// Diagonal Fisher approximation
    pub diagonal_fisher: bool,

    /// Block diagonal Fisher approximation
    pub block_diagonal_fisher: bool,

    /// Kronecker factored approximation (K-FAC style)
    pub kronecker_factored: bool,
}

/// Fisher information matrix estimation methods
#[derive(Debug, Clone, Copy)]
pub enum FisherEstimationMethod {
    /// Empirical Fisher Information Matrix
    Empirical,

    /// True Fisher Information Matrix (using log-likelihood Hessian)
    True,

    /// Diagonal approximation
    Diagonal,

    /// Block diagonal approximation
    BlockDiagonal,

    /// Kronecker factored approximation
    KroneckerFactored,

    /// Gauss-Newton approximation
    GaussNewton,

    /// BFGS quasi-Newton approximation
    BFGS,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for NaturalGradientConfig<T> {
    fn default() -> Self {
        Self {
            base_config: RLOptimizerConfig::default(),
            fisher_method: FisherEstimationMethod::Empirical,
            damping: T::from(1e-4).unwrap_or_else(|| T::zero()),
            fisher_update_freq: 10,
            use_empirical_fisher: true,
            cg_iters: 10,
            cg_tolerance: T::from(1e-8).unwrap_or_else(|| T::zero()),
            natural_grad_scale: T::from(1.0).unwrap_or_else(|| T::zero()),
            enable_preconditioning: true,
            diagonal_fisher: false,
            block_diagonal_fisher: false,
            kronecker_factored: false,
        }
    }
}

/// Natural Policy Gradient optimizer
pub struct NaturalPolicyGradient<T: Float + Debug + Send + Sync + 'static, P: PolicyNetwork<T>> {
    /// Configuration
    _config: NaturalGradientConfig<T>,

    /// Policy network
    policy: P,

    /// Fisher Information Matrix
    fisher_matrix: Option<Array2<T>>,

    /// Diagonal Fisher approximation
    fisher_diagonal: Option<Array1<T>>,

    /// Kronecker factors (for K-FAC style approximation)
    kronecker_factors: Option<KroneckerFactors<T>>,

    /// Empirical Fisher accumulator
    empirical_fisher_accumulator: FisherAccumulator<T>,

    /// Natural gradient state
    natural_grad_state: NaturalGradientState<T>,

    /// Update counter
    update_count: usize,

    /// Parameter dimension
    paramdim: usize,
}

/// Kronecker factorization components
#[derive(Debug, Clone)]
pub struct KroneckerFactors<T: Float + Debug + Send + Sync + 'static> {
    /// Input statistics (activation covariances)
    pub input_factors: Vec<Array2<T>>,

    /// Output statistics (gradient covariances)
    pub output_factors: Vec<Array2<T>>,

    /// Layer indices for factor mapping
    pub layer_indices: Vec<usize>,
}

/// Fisher information accumulator for empirical estimation
#[derive(Debug, Clone)]
pub struct FisherAccumulator<T: Float + Debug + Send + Sync + 'static> {
    /// Accumulated Fisher matrix
    pub fisher_sum: Array2<T>,

    /// Number of samples accumulated
    pub sample_count: usize,

    /// Gradient history for empirical Fisher
    pub gradient_history: Vec<Array1<T>>,

    /// Maximum history size
    pub max_history_size: usize,
}

/// Natural gradient optimization state
#[derive(Debug, Clone)]
pub struct NaturalGradientState<T: Float + Debug + Send + Sync + 'static> {
    /// Previous natural gradients
    pub prev_natural_grad: Option<Array1<T>>,

    /// Momentum for natural gradients
    pub momentum: T,

    /// Adaptive scaling factors
    pub adaptive_scales: Option<Array1<T>>,

    /// Trust region radius
    pub trust_radius: T,

    /// KL divergence history
    pub kl_history: Vec<T>,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + ScalarOperand
            + std::ops::AddAssign
            + std::iter::Sum,
        P: PolicyNetwork<T>,
    > NaturalPolicyGradient<T, P>
{
    /// Create a new natural policy gradient optimizer
    pub fn new(_config: NaturalGradientConfig<T>, policy: P, paramdim: usize) -> Self {
        let fisher_accumulator = FisherAccumulator {
            fisher_sum: Array2::zeros((paramdim, paramdim)),
            sample_count: 0,
            gradient_history: Vec::new(),
            max_history_size: 1000,
        };

        let natural_grad_state = NaturalGradientState {
            prev_natural_grad: None,
            momentum: T::from(0.9).unwrap_or_else(|| T::zero()),
            adaptive_scales: None,
            trust_radius: T::from(1.0).unwrap_or_else(|| T::zero()),
            kl_history: Vec::new(),
        };

        Self {
            _config,
            policy,
            fisher_matrix: None,
            fisher_diagonal: None,
            kronecker_factors: None,
            empirical_fisher_accumulator: fisher_accumulator,
            natural_grad_state,
            update_count: 0,
            paramdim,
        }
    }

    /// Update using trajectory data
    pub fn update(
        &mut self,
        trajectory: TrajectoryBatch<T>,
        gradients: Array1<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        // Update Fisher information matrix
        if self
            .update_count
            .is_multiple_of(self._config.fisher_update_freq)
        {
            self.update_fisher_information(&trajectory)?;
        }

        // Compute natural gradients
        let naturalgradients = self.compute_natural_gradients(&gradients)?;

        // Apply natural gradient update
        self.apply_natural_gradient_update(&naturalgradients)?;

        // Update state
        self.natural_grad_state.prev_natural_grad = Some(naturalgradients);
        self.update_count += 1;

        // Compute metrics
        let metrics = RLOptimizationMetrics {
            policy_grad_norm: self.vector_norm(&gradients),
            ..Default::default()
        };

        Ok(metrics)
    }

    /// Update Fisher Information Matrix
    fn update_fisher_information(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        match self._config.fisher_method {
            FisherEstimationMethod::Empirical => self.update_empirical_fisher(trajectory)?,
            FisherEstimationMethod::True => self.update_true_fisher(trajectory)?,
            FisherEstimationMethod::Diagonal => self.update_diagonal_fisher(trajectory)?,
            FisherEstimationMethod::BlockDiagonal => {
                self.update_block_diagonal_fisher(trajectory)?
            }
            FisherEstimationMethod::KroneckerFactored => {
                self.update_kronecker_factors(trajectory)?
            }
            _ => {
                // Fallback to empirical Fisher
                self.update_empirical_fisher(trajectory)?;
            }
        }

        Ok(())
    }

    /// Update empirical Fisher Information Matrix
    fn update_empirical_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        let batch_size = trajectory.observations.nrows();

        // Collect gradients for each sample
        for i in 0..batch_size {
            let obs = trajectory.observations.row(i).to_owned();
            let action = trajectory.actions.row(i).to_owned();

            // Compute log probability gradients
            let log_prob_grad = self.compute_log_prob_gradients(&obs, &action)?;

            // Add to empirical Fisher accumulator
            self.add_to_empirical_fisher(&log_prob_grad)?;
        }

        // Compute final empirical Fisher matrix
        self.finalize_empirical_fisher()?;

        Ok(())
    }

    /// Update true Fisher Information Matrix
    fn update_true_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        // True Fisher requires computing the Hessian of the log-likelihood
        // This is computationally expensive and often approximated
        self.update_empirical_fisher(trajectory) // Fallback for now
    }

    /// Update diagonal Fisher approximation
    fn update_diagonal_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        let mut diagonal = Array1::zeros(self.paramdim);
        let batch_size = trajectory.observations.nrows();

        for i in 0..batch_size {
            let obs = trajectory.observations.row(i).to_owned();
            let action = trajectory.actions.row(i).to_owned();

            let log_prob_grad = self.compute_log_prob_gradients(&obs, &action)?;
            diagonal = diagonal + log_prob_grad.mapv(|x| x * x);
        }

        diagonal = diagonal / T::from(batch_size).unwrap_or_else(|| T::zero());
        diagonal += T::from(self._config.damping).unwrap_or_else(|| T::zero());

        self.fisher_diagonal = Some(diagonal);

        Ok(())
    }

    /// Update block diagonal Fisher approximation
    fn update_block_diagonal_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        // Block diagonal approximation groups parameters into blocks
        // and assumes independence between blocks
        Ok(())
    }

    /// Update Kronecker factorization
    fn update_kronecker_factors(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        // Kronecker factorization approximates the Fisher matrix as
        // a Kronecker product of smaller matrices (K-FAC style)
        Ok(())
    }

    /// Compute natural gradients
    fn compute_natural_gradients(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        if !self._config.enable_preconditioning {
            return Ok(gradients.clone());
        }

        let natural_grad = match self._config.fisher_method {
            FisherEstimationMethod::Diagonal => {
                if let Some(ref diag) = self.fisher_diagonal {
                    gradients / diag
                } else {
                    gradients.clone()
                }
            }
            _ => {
                if let Some(ref fisher) = self.fisher_matrix {
                    // Solve Fisher * natural_grad = gradients
                    self.solve_fisher_system(fisher, gradients)?
                } else {
                    gradients.clone()
                }
            }
        };

        // Apply scaling
        let scaled_natural_grad = natural_grad * self._config.natural_grad_scale;

        Ok(scaled_natural_grad)
    }

    /// Solve Fisher information system using conjugate gradient
    fn solve_fisher_system(&self, fisher: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>> {
        let n = rhs.len();
        let mut x = Array1::zeros(n);
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rsold = self.dot(&r, &r);

        for _i in 0..self._config.cg_iters {
            let ap = fisher.dot(&p);
            let alpha = rsold / self.dot(&p, &ap);

            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);

            let rsnew = self.dot(&r, &r);

            if rsnew.sqrt() < self._config.cg_tolerance {
                break;
            }

            let beta = rsnew / rsold;
            p = &r + &(&p * beta);
            rsold = rsnew;
        }

        Ok(x)
    }

    /// Apply natural gradient update
    fn apply_natural_gradient_update(&mut self, naturalgradients: &Array1<T>) -> Result<()> {
        // In practice, this would update the policy network parameters
        // using the natural _gradients

        // Apply momentum if previous natural _gradients exist
        let update = if let Some(ref prev_ng) = self.natural_grad_state.prev_natural_grad {
            naturalgradients + &(prev_ng * self.natural_grad_state.momentum)
        } else {
            naturalgradients.clone()
        };

        // Apply trust region constraint
        let clipped_update = self.apply_trust_region_constraint(&update)?;

        self.update_policy_parameters(&clipped_update)?;

        Ok(())
    }

    /// Apply trust region constraint to natural gradient update
    fn apply_trust_region_constraint(&self, update: &Array1<T>) -> Result<Array1<T>> {
        let update_norm = self.vector_norm(update);
        let trust_radius = self.natural_grad_state.trust_radius;

        if update_norm <= trust_radius {
            Ok(update.clone())
        } else {
            Ok(update * (trust_radius / update_norm))
        }
    }

    /// Update policy network parameters from a flat natural-gradient update.
    ///
    /// The flat `update` vector is mapped back onto the policy's named parameters.
    /// Parameter keys are visited in SORTED order for determinism, the flat update
    /// is sliced into contiguous chunks matching each parameter's length, and the
    /// resulting `HashMap<String, Array1<T>>` is forwarded to
    /// `policy.update_parameters`.
    ///
    /// Returns [`OptimError::DimensionMismatch`] if the flat update length does not
    /// equal the total parameter count across all named parameters.
    fn update_policy_parameters(&mut self, update: &Array1<T>) -> Result<()> {
        let params = self.policy.get_parameters();

        // Deterministic ordering of parameter names.
        let mut keys: Vec<String> = params.keys().cloned().collect();
        keys.sort();

        // Total parameter count must match the flat update length.
        let total: usize = keys.iter().map(|k| params[k].len()).sum();
        if total != update.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "Flat update length ({}) does not match total policy parameter count ({})",
                update.len(),
                total
            )));
        }

        // Slice the flat update into per-parameter chunks.
        let mut grads: HashMap<String, Array1<T>> = HashMap::with_capacity(keys.len());
        let mut offset = 0usize;
        for key in keys {
            let len = params[&key].len();
            let chunk = update
                .slice(scirs2_core::ndarray::s![offset..offset + len])
                .to_owned();
            grads.insert(key, chunk);
            offset += len;
        }

        self.policy.update_parameters(&grads)
    }

    /// Maximum parameter count for finite-difference score estimation.
    const FD_MAX_DIMS: usize = 500;

    /// Compute log probability gradients via central finite differences.
    ///
    /// For each scalar parameter θ_i, the score function component is:
    ///   s_i = [log π(a|s; θ+ε·eᵢ) − log π(a|s; θ−ε·eᵢ)] / (2ε)
    ///
    /// Uses three additive calls to `update_parameters` per dimension (+ε,
    /// −2ε, +ε) so the policy is exactly restored to its original state.
    /// Skipped (returns zeros) when `paramdim > FD_MAX_DIMS` — callers can
    /// supply gradients directly via `add_to_empirical_fisher` instead.
    fn compute_log_prob_gradients(
        &mut self,
        obs: &Array1<T>,
        action: &Array1<T>,
    ) -> Result<Array1<T>> {
        if self.paramdim == 0 || self.paramdim > Self::FD_MAX_DIMS {
            return Ok(Array1::zeros(self.paramdim));
        }

        let eps = T::from(1e-5_f64).unwrap_or_else(|| T::zero());
        let two_eps = eps + eps;

        let obs_dim = obs.len();
        let act_dim = action.len();
        let mut obs_2d = Array2::zeros((1, obs_dim));
        obs_2d.row_mut(0).assign(obs);
        let mut act_2d = Array2::zeros((1, act_dim));
        act_2d.row_mut(0).assign(action);

        let params = self.policy.get_parameters();
        let mut sorted_keys: Vec<String> = params.keys().cloned().collect();
        sorted_keys.sort();

        let total: usize = sorted_keys.iter().map(|k| params[k].len()).sum();
        if total != self.paramdim {
            return Ok(Array1::zeros(self.paramdim));
        }

        let mut score = Array1::zeros(self.paramdim);
        let mut flat_idx = 0usize;

        for key in &sorted_keys {
            let param_len = params[key].len();
            for i in 0..param_len {
                // +ε perturbation
                let mut delta = Array1::zeros(param_len);
                delta[i] = eps;
                let mut d_plus: HashMap<String, Array1<T>> = HashMap::with_capacity(1);
                d_plus.insert(key.clone(), delta.clone());
                self.policy.update_parameters(&d_plus)?;
                let lp_plus = self.policy.evaluate_actions(&obs_2d, &act_2d)?.log_probs[0];

                // −2ε (net −ε from original)
                let mut d_minus: HashMap<String, Array1<T>> = HashMap::with_capacity(1);
                let mut neg2 = Array1::zeros(param_len);
                neg2[i] = -two_eps;
                d_minus.insert(key.clone(), neg2);
                self.policy.update_parameters(&d_minus)?;
                let lp_minus = self.policy.evaluate_actions(&obs_2d, &act_2d)?.log_probs[0];

                // +ε restore
                let mut d_restore: HashMap<String, Array1<T>> = HashMap::with_capacity(1);
                d_restore.insert(key.clone(), delta);
                self.policy.update_parameters(&d_restore)?;

                score[flat_idx] = (lp_plus - lp_minus) / two_eps;
                flat_idx += 1;
            }
        }

        Ok(score)
    }

    /// Add gradient to empirical Fisher accumulator
    fn add_to_empirical_fisher(&mut self, gradient: &Array1<T>) -> Result<()> {
        // Add outer product of gradient to Fisher sum
        for i in 0..self.paramdim {
            for j in 0..self.paramdim {
                self.empirical_fisher_accumulator.fisher_sum[[i, j]] += gradient[i] * gradient[j];
            }
        }

        self.empirical_fisher_accumulator.sample_count += 1;

        // Store gradient in history
        if self.empirical_fisher_accumulator.gradient_history.len()
            >= self.empirical_fisher_accumulator.max_history_size
        {
            self.empirical_fisher_accumulator.gradient_history.remove(0);
        }
        self.empirical_fisher_accumulator
            .gradient_history
            .push(gradient.clone());

        Ok(())
    }

    /// Finalize empirical Fisher matrix computation
    fn finalize_empirical_fisher(&mut self) -> Result<()> {
        if self.empirical_fisher_accumulator.sample_count == 0 {
            return Ok(());
        }

        // Normalize by sample count
        let fisher = &self.empirical_fisher_accumulator.fisher_sum
            / T::from(self.empirical_fisher_accumulator.sample_count).unwrap_or_else(|| T::zero());

        // Add damping for numerical stability
        let mut damped_fisher = fisher;
        for i in 0..self.paramdim {
            damped_fisher[[i, i]] += self._config.damping;
        }

        self.fisher_matrix = Some(damped_fisher);

        // Reset accumulator
        self.empirical_fisher_accumulator.fisher_sum.fill(T::zero());
        self.empirical_fisher_accumulator.sample_count = 0;

        Ok(())
    }

    /// Compute dot product
    fn dot(&self, a: &Array1<T>, b: &Array1<T>) -> T {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Compute vector norm
    fn vector_norm(&self, v: &Array1<T>) -> T {
        self.dot(v, v).sqrt()
    }

    /// Get current Fisher matrix
    pub fn get_fisher_matrix(&self) -> Option<&Array2<T>> {
        self.fisher_matrix.as_ref()
    }

    /// Get current natural gradient state
    pub fn get_natural_grad_state(&self) -> &NaturalGradientState<T> {
        &self.natural_grad_state
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ActionDistribution, DistributionType, PolicyEvaluation};
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::arr1;
    use std::cell::RefCell;

    /// Minimal mock policy network with two named parameters (`"a"` len 2,
    /// `"b"` len 3) so the flat→named split is non-trivial. The last gradients
    /// passed to `update_parameters` are recorded so the test can assert the
    /// forwarded split.
    struct MockPolicy {
        params: HashMap<String, Array1<f64>>,
        last_gradients: RefCell<Option<HashMap<String, Array1<f64>>>>,
    }

    impl MockPolicy {
        fn new() -> Self {
            let mut params = HashMap::new();
            params.insert("a".to_string(), arr1(&[0.0, 0.0]));
            params.insert("b".to_string(), arr1(&[0.0, 0.0, 0.0]));
            Self {
                params,
                last_gradients: RefCell::new(None),
            }
        }
    }

    impl PolicyNetwork<f64> for MockPolicy {
        fn evaluate_actions(
            &self,
            _observations: &Array2<f64>,
            _actions: &Array2<f64>,
        ) -> Result<PolicyEvaluation<f64>> {
            Ok(PolicyEvaluation {
                log_probs: arr1(&[0.0]),
                entropy: arr1(&[0.0]),
                metrics: HashMap::new(),
            })
        }

        fn get_action_distribution(
            &self,
            _observations: &Array2<f64>,
        ) -> Result<ActionDistribution<f64>> {
            Ok(ActionDistribution {
                mean: None,
                std: None,
                logits: None,
                distribution_type: DistributionType::Gaussian,
            })
        }

        fn update_parameters(&mut self, gradients: &HashMap<String, Array1<f64>>) -> Result<()> {
            for (key, grad) in gradients {
                if let Some(p) = self.params.get_mut(key) {
                    *p = &*p + grad;
                }
            }
            *self.last_gradients.borrow_mut() = Some(gradients.clone());
            Ok(())
        }

        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            self.params.clone()
        }
    }

    fn make_optimizer() -> NaturalPolicyGradient<f64, MockPolicy> {
        // Total parameter count is 5 ("a": 2 + "b": 3).
        NaturalPolicyGradient::new(
            NaturalGradientConfig::<f64>::default(),
            MockPolicy::new(),
            5,
        )
    }

    #[test]
    fn test_update_policy_parameters_forwards_split() {
        let mut opt = make_optimizer();

        // Flat update of length 5. Keys are visited in SORTED order: "a" then "b",
        // so "a" gets [0.1, 0.2] and "b" gets [0.3, 0.4, 0.5].
        let update = arr1(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        opt.update_policy_parameters(&update).unwrap();

        let recorded = opt.policy.last_gradients.borrow();
        let map = recorded.as_ref().expect("update_parameters was not called");

        let a_grad = map.get("a").expect("missing 'a' gradient");
        assert_eq!(a_grad.len(), 2);
        assert_abs_diff_eq!(a_grad[0], 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grad[1], 0.2, epsilon = 1e-12);

        let b_grad = map.get("b").expect("missing 'b' gradient");
        assert_eq!(b_grad.len(), 3);
        assert_abs_diff_eq!(b_grad[0], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(b_grad[1], 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(b_grad[2], 0.5, epsilon = 1e-12);

        // The policy parameters were actually advanced by the update.
        let params = opt.policy.get_parameters();
        let a = params.get("a").unwrap();
        let b = params.get("b").unwrap();
        assert_abs_diff_eq!(a[0], 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(a[1], 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(b[0], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(b[1], 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(b[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_update_policy_parameters_length_mismatch_errors() {
        let mut opt = make_optimizer();
        // Wrong length (4 != 5) must return an error.
        let bad = arr1(&[0.1, 0.2, 0.3, 0.4]);
        assert!(opt.update_policy_parameters(&bad).is_err());
    }

    // ── Finite-difference score function test ─────────────────────────────

    /// A Gaussian policy π(a|s) = N(a; mean, 1).
    /// Parameters: {"mean": μ ∈ R²}.
    /// update_parameters is additive; evaluate_actions computes the real log-prob.
    struct GaussianPolicy {
        mean: Array1<f64>,
    }

    impl GaussianPolicy {
        fn new(m0: f64, m1: f64) -> Self {
            Self {
                mean: arr1(&[m0, m1]),
            }
        }
    }

    impl PolicyNetwork<f64> for GaussianPolicy {
        fn evaluate_actions(
            &self,
            _observations: &Array2<f64>,
            actions: &Array2<f64>,
        ) -> Result<PolicyEvaluation<f64>> {
            let batch_size = actions.nrows();
            let mut log_probs = Array1::zeros(batch_size);
            for i in 0..batch_size {
                let mut lp = 0.0_f64;
                for j in 0..self.mean.len().min(actions.ncols()) {
                    let diff = actions[[i, j]] - self.mean[j];
                    lp -= 0.5 * diff * diff;
                }
                log_probs[i] = lp;
            }
            Ok(PolicyEvaluation {
                log_probs,
                entropy: Array1::zeros(batch_size),
                metrics: HashMap::new(),
            })
        }

        fn get_action_distribution(&self, obs: &Array2<f64>) -> Result<ActionDistribution<f64>> {
            let n = obs.nrows();
            Ok(ActionDistribution {
                mean: Some(Array2::from_shape_fn((n, 2), |(_, j)| self.mean[j])),
                std: Some(Array2::ones((n, 2))),
                logits: None,
                distribution_type: DistributionType::Gaussian,
            })
        }

        fn update_parameters(&mut self, gradients: &HashMap<String, Array1<f64>>) -> Result<()> {
            if let Some(delta) = gradients.get("mean") {
                self.mean = &self.mean + delta;
            }
            Ok(())
        }

        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            let mut m = HashMap::new();
            m.insert("mean".to_string(), self.mean.clone());
            m
        }
    }

    #[test]
    fn test_fd_score_matches_analytical_for_gaussian_policy() {
        // For π(a|s) = N(a; mean, 1), the score ∇_mean log π = (a − mean).
        // With mean=[0,0] and action=[3, -2], analytical score = [3, -2].
        // FD (ε=1e-5) should match within 1e-4.
        let policy = GaussianPolicy::new(0.0, 0.0);
        let mut opt = NaturalPolicyGradient::new(
            NaturalGradientConfig::<f64>::default(),
            policy,
            2, // paramdim = len("mean")
        );

        let obs = arr1(&[1.0_f64]); // observation is irrelevant here
        let action = arr1(&[3.0_f64, -2.0]);

        let score = opt
            .compute_log_prob_gradients(&obs, &action)
            .expect("FD score should succeed for paramdim=2");

        assert_eq!(score.len(), 2);
        assert_abs_diff_eq!(score[0], 3.0, epsilon = 1e-4);
        assert_abs_diff_eq!(score[1], -2.0, epsilon = 1e-4);

        // Verify the policy parameters are restored (perturbation was undone).
        let restored = opt.policy.get_parameters();
        let mean = &restored["mean"];
        assert_abs_diff_eq!(mean[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean[1], 0.0, epsilon = 1e-10);
    }
}
