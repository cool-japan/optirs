// Reinforcement Learning Optimizers
//
// This module provides specialized optimizers for reinforcement learning,
// including policy gradient methods, actor-critic algorithms, and trust region methods.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

pub mod actor_critic;
pub mod linear_models;
pub mod natural_gradients;
pub mod policy_gradient;
pub mod trust_region;

// Re-export key types
pub use actor_critic::{ActorCriticConfig, ActorCriticMethod, ActorCriticOptimizer};
pub use linear_models::{
    LinearGaussianPolicy, LinearQFunction, LinearSoftmaxPolicy, LinearValueFunction,
};
pub use natural_gradients::{NaturalGradientConfig, NaturalPolicyGradient};
pub use policy_gradient::{PolicyGradientConfig, PolicyGradientMethod, PolicyGradientOptimizer};
pub use trust_region::{TrustRegionConfig, TrustRegionMethod, TrustRegionOptimizer};

/// Reinforcement Learning optimization configuration
#[derive(Debug, Clone)]
pub struct RLOptimizerConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Policy learning rate
    pub policy_lr: T,

    /// Value function learning rate  
    pub value_lr: T,

    /// Discount factor (gamma)
    pub discount_factor: T,

    /// GAE lambda parameter
    pub gae_lambda: T,

    /// Clipping parameter for PPO
    pub clip_epsilon: T,

    /// Entropy regularization coefficient
    pub entropy_coeff: T,

    /// Value function loss coefficient
    pub value_loss_coeff: T,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: T,

    /// Number of optimization epochs per update
    pub n_epochs: usize,

    /// Mini-batch size for optimization
    pub mini_batchsize: usize,

    /// Trust region methods configuration
    pub trust_region_config: Option<TrustRegionConfig<T>>,

    /// Enable natural policy gradients
    pub use_natural_gradients: bool,

    /// Fisher information matrix approximation method
    pub fisher_approximation: FisherApproximationMethod,
}

/// Methods for approximating the Fisher Information Matrix
#[derive(Debug, Clone, Copy)]
pub enum FisherApproximationMethod {
    /// Empirical Fisher Information Matrix
    Empirical,

    /// Kronecker-factored approximation
    KroneckerFactored,

    /// Diagonal approximation
    Diagonal,

    /// Block-diagonal approximation
    BlockDiagonal,

    /// Low-rank approximation
    LowRank,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for RLOptimizerConfig<T> {
    fn default() -> Self {
        Self {
            policy_lr: T::from(3e-4).unwrap_or_else(|| T::zero()),
            value_lr: T::from(1e-3).unwrap_or_else(|| T::zero()),
            discount_factor: T::from(0.99).unwrap_or_else(|| T::zero()),
            gae_lambda: T::from(0.95).unwrap_or_else(|| T::zero()),
            clip_epsilon: T::from(0.2).unwrap_or_else(|| T::zero()),
            entropy_coeff: T::from(0.01).unwrap_or_else(|| T::zero()),
            value_loss_coeff: T::from(0.5).unwrap_or_else(|| T::zero()),
            max_grad_norm: T::from(0.5).unwrap_or_else(|| T::zero()),
            n_epochs: 4,
            mini_batchsize: 64,
            trust_region_config: None,
            use_natural_gradients: false,
            fisher_approximation: FisherApproximationMethod::Diagonal,
        }
    }
}

/// Trajectory data for RL optimization
#[derive(Debug, Clone)]
pub struct TrajectoryBatch<T: Float + Debug + Send + Sync + 'static> {
    /// Observations
    pub observations: Array2<T>,

    /// Actions taken
    pub actions: Array2<T>,

    /// Log probabilities of actions
    pub log_probs: Array1<T>,

    /// Rewards received
    pub rewards: Array1<T>,

    /// Value function estimates
    pub values: Array1<T>,

    /// Done flags (episode termination)
    pub dones: Array1<bool>,

    /// Advantage estimates
    pub advantages: Array1<T>,

    /// Target returns
    pub returns: Array1<T>,

    /// Observation reached *after* the final transition of the batch (`s_T`).
    ///
    /// GAE and V-trace both need the value of the state that follows the last
    /// stored transition in order to bootstrap. That state is **not** part of the
    /// batch — `observations.row(len - 1)` is `s_{T-1}`, the state the last action
    /// was taken *from*. Bootstrapping on `s_{T-1}` is an off-by-one error that
    /// biases every advantage in the batch, so the successor state is carried
    /// explicitly here.
    ///
    /// `None` means "no successor available" (e.g. the batch ends on a terminal
    /// transition, or the caller did not record it); consumers then bootstrap with
    /// zero. When the final transition is terminal the bootstrap is masked out by
    /// `dones` regardless of this field.
    pub final_observation: Option<Array1<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::numeric::FromPrimitive>
    TrajectoryBatch<T>
{
    /// Create a new trajectory batch
    pub fn new(
        observations: Array2<T>,
        actions: Array2<T>,
        log_probs: Array1<T>,
        rewards: Array1<T>,
        values: Array1<T>,
        dones: Array1<bool>,
    ) -> Result<Self> {
        let batch_size = observations.nrows();

        // Validate dimensions
        if actions.nrows() != batch_size
            || log_probs.len() != batch_size
            || rewards.len() != batch_size
            || values.len() != batch_size
            || dones.len() != batch_size
        {
            return Err(OptimError::InvalidConfig(
                "Inconsistent batch dimensions".to_string(),
            ));
        }

        // Compute advantages and returns (will be updated by compute_advantages)
        let advantages = Array1::zeros(batch_size);
        let returns = Array1::zeros(batch_size);

        Ok(Self {
            observations,
            actions,
            log_probs,
            rewards,
            values,
            dones,
            advantages,
            returns,
            final_observation: None,
        })
    }

    /// Attach the successor observation `s_T` used to bootstrap the final step.
    ///
    /// See [`TrajectoryBatch::final_observation`]. Returns an error if the
    /// dimensionality does not match the batch's observation dimension.
    pub fn with_final_observation(mut self, final_observation: Array1<T>) -> Result<Self> {
        let expected = self.observations.ncols();
        if final_observation.len() != expected {
            return Err(OptimError::DimensionMismatch(format!(
                "final observation length ({}) does not match observation dimension ({})",
                final_observation.len(),
                expected
            )));
        }
        self.final_observation = Some(final_observation);
        Ok(self)
    }

    /// Compute Generalized Advantage Estimation (GAE) **without** normalizing.
    ///
    /// ```text
    /// δ_t = r_t + γ·(1 − done_t)·V(s_{t+1}) − V(s_t)
    /// A_t = δ_t + γ·λ·(1 − done_t)·A_{t+1}
    /// R_t = A_t + V(s_t)
    /// ```
    ///
    /// `done_t` is read from `self.dones[t]` for **every** `t`, including the last
    /// one: a batch whose final transition terminates the episode must not
    /// bootstrap. `nextvalue` is `V(s_T)`, the value of the successor of the final
    /// transition (see [`TrajectoryBatch::final_observation`]); it is ignored when
    /// the final transition is terminal.
    pub fn compute_gae(&mut self, gamma: T, lambda: T, nextvalue: T) -> Result<()> {
        let batch_size = self.rewards.len();
        if batch_size == 0 {
            return Ok(());
        }
        let mut gae = T::zero();

        for t in (0..batch_size).rev() {
            // The terminal flag of step `t` itself — never a hardcoded `false`.
            let nonterminal = if self.dones[t] { T::zero() } else { T::one() };

            let next_val = if t == batch_size - 1 {
                nextvalue
            } else {
                self.values[t + 1]
            };

            let delta = self.rewards[t] + gamma * next_val * nonterminal - self.values[t];
            gae = delta + gamma * lambda * nonterminal * gae;

            self.advantages[t] = gae;
            self.returns[t] = gae + self.values[t];
        }

        Ok(())
    }

    /// Compute GAE advantages/returns and normalize the advantages to zero mean
    /// and unit variance (the usual policy-gradient variance reduction).
    ///
    /// Normalization is skipped for batches of fewer than two samples (where the
    /// sample standard deviation is zero and normalizing would annihilate the
    /// signal) and whenever the spread is numerically negligible.
    pub fn compute_advantages(&mut self, gamma: T, lambda: T, nextvalue: T) -> Result<()> {
        self.compute_gae(gamma, lambda, nextvalue)?;

        if self.advantages.len() < 2 {
            return Ok(());
        }

        let mean = self.advantages.mean().unwrap_or(T::zero());
        let std = self
            .advantages
            .mapv(|x| (x - mean) * (x - mean))
            .mean()
            .unwrap_or(T::one())
            .sqrt();

        if std > T::from(1e-8).unwrap_or_else(|| T::zero()) {
            self.advantages.mapv_inplace(|x| (x - mean) / std);
        }

        Ok(())
    }

    /// Fill `returns` with plain discounted Monte-Carlo returns
    /// `G_t = r_t + γ·(1 − done_t)·G_{t+1}`, bootstrapping the final step with
    /// `nextvalue` when the final transition is non-terminal.
    ///
    /// Used by baseline-free REINFORCE, where the advantage *is* the return.
    /// `advantages` is set to `G_t − V(s_t)` so downstream code that reads
    /// advantages stays meaningful when a value baseline happens to be present.
    pub fn compute_discounted_returns(&mut self, gamma: T, nextvalue: T) -> Result<()> {
        let batch_size = self.rewards.len();
        if batch_size == 0 {
            return Ok(());
        }

        let mut running = nextvalue;
        for t in (0..batch_size).rev() {
            let nonterminal = if self.dones[t] { T::zero() } else { T::one() };
            running = self.rewards[t] + gamma * nonterminal * running;
            self.returns[t] = running;
            self.advantages[t] = running - self.values[t];
        }

        Ok(())
    }

    /// Get mini-batches for optimization
    pub fn get_mini_batches(&self, mini_batchsize: usize) -> Vec<TrajectoryBatch<T>> {
        let batch_size = self.observations.nrows();
        let n_mini_batches = batch_size.div_ceil(mini_batchsize);

        let mut mini_batches = Vec::new();

        for i in 0..n_mini_batches {
            let start = i * mini_batchsize;
            let end = ((i + 1) * mini_batchsize).min(batch_size);

            if start >= end {
                break;
            }

            let obs = self.observations.slice(s![start..end, ..]).to_owned();
            let acts = self.actions.slice(s![start..end, ..]).to_owned();
            let log_probs = self.log_probs.slice(s![start..end]).to_owned();
            let rewards = self.rewards.slice(s![start..end]).to_owned();
            let values = self.values.slice(s![start..end]).to_owned();
            let dones = self.dones.slice(s![start..end]).to_owned().to_vec();
            let advantages = self.advantages.slice(s![start..end]).to_owned();
            let returns = self.returns.slice(s![start..end]).to_owned();

            // Convert Vec<bool> back to Array1<bool>
            let dones_array = Array1::from_vec(dones);

            // The successor of this slice's last transition is the first
            // observation of the next slice, or the whole batch's successor for
            // the final slice.
            let final_observation = if end < batch_size {
                Some(self.observations.row(end).to_owned())
            } else {
                self.final_observation.clone()
            };

            let mini_batch = TrajectoryBatch {
                observations: obs,
                actions: acts,
                log_probs,
                rewards,
                values,
                dones: dones_array,
                advantages,
                returns,
                final_observation,
            };

            mini_batches.push(mini_batch);
        }

        mini_batches
    }
}

/// A Kronecker-factored block of the Fisher information matrix.
///
/// K-FAC approximates the Fisher block of a linear layer `W ∈ R^{n_out × n_in}` as
/// `F ≈ G ⊗ A` with `A = E[φ φᵀ]` (input/activation covariance) and
/// `G = E[δ δᵀ]` (output pre-activation gradient covariance). This struct carries
/// the *per-sample* factors so the covariances can be formed by the consumer:
/// row `i` of `inputs` is `φ_i`, row `i` of `outputs` is `δ_i`.
///
/// Contract: the per-sample score for the named parameter, reshaped **row-major**
/// to `(n_out, n_in)`, must equal `δ_i φ_iᵀ`.
#[derive(Debug, Clone)]
pub struct KroneckerBlock<T: Float + Debug + Send + Sync + 'static> {
    /// Name of the parameter this block factorizes (a key of `get_parameters`).
    pub name: String,

    /// Per-sample layer inputs, shape `(n_samples, n_in)`.
    pub inputs: Array2<T>,

    /// Per-sample pre-activation gradients, shape `(n_samples, n_out)`.
    pub outputs: Array2<T>,
}

/// Total number of scalars across a named-parameter map.
pub fn parameter_count<T: Float + Debug + Send + Sync + 'static>(
    params: &HashMap<String, Array1<T>>,
) -> usize {
    params.values().map(|p| p.len()).sum()
}

/// Parameter keys in deterministic (sorted) order — the canonical flat layout.
pub fn parameter_keys<T: Float + Debug + Send + Sync + 'static>(
    params: &HashMap<String, Array1<T>>,
) -> Vec<String> {
    let mut keys: Vec<String> = params.keys().cloned().collect();
    keys.sort();
    keys
}

/// Flatten a named-parameter map into a single vector using the canonical
/// (sorted-key, contiguous) layout shared by every flat/named conversion here.
pub fn flatten_named<T: Float + Debug + Send + Sync + 'static>(
    params: &HashMap<String, Array1<T>>,
) -> Array1<T> {
    let mut flat = Array1::zeros(parameter_count(params));
    let mut offset = 0usize;
    for key in parameter_keys(params) {
        let value = &params[&key];
        for (i, &v) in value.iter().enumerate() {
            flat[offset + i] = v;
        }
        offset += value.len();
    }
    flat
}

/// Split a flat vector back into a named-parameter map matching `template`'s
/// keys and lengths, using the canonical sorted-key layout.
///
/// Returns [`OptimError::DimensionMismatch`] when the flat length does not equal
/// the template's total parameter count.
pub fn unflatten_named<T: Float + Debug + Send + Sync + 'static>(
    template: &HashMap<String, Array1<T>>,
    flat: &Array1<T>,
) -> Result<HashMap<String, Array1<T>>> {
    let total = parameter_count(template);
    if total != flat.len() {
        return Err(OptimError::DimensionMismatch(format!(
            "Flat vector length ({}) does not match total parameter count ({})",
            flat.len(),
            total
        )));
    }

    let keys = parameter_keys(template);
    let mut out: HashMap<String, Array1<T>> = HashMap::with_capacity(keys.len());
    let mut offset = 0usize;
    for key in keys {
        let len = template[&key].len();
        let mut chunk = Array1::zeros(len);
        for i in 0..len {
            chunk[i] = flat[offset + i];
        }
        out.insert(key, chunk);
        offset += len;
    }
    Ok(out)
}

/// Global-norm clipping of a named-gradient map.
///
/// Returns the clipped gradients together with the **pre-clipping** global norm
/// (what the metrics should report). A non-positive `max_norm` disables clipping.
pub fn clip_named_gradients<T: Float + Debug + Send + Sync + 'static>(
    gradients: &HashMap<String, Array1<T>>,
    max_norm: T,
) -> (HashMap<String, Array1<T>>, T) {
    let mut total = T::zero();
    for grad in gradients.values() {
        for &g in grad.iter() {
            total = total + g * g;
        }
    }
    let norm = total.sqrt();

    let factor = if max_norm > T::zero() && norm > max_norm && norm > T::zero() {
        max_norm / norm
    } else {
        T::one()
    };

    let clipped = gradients
        .iter()
        .map(|(name, grad)| (name.clone(), grad.mapv(|g| g * factor)))
        .collect();

    (clipped, norm)
}

/// Multiply every entry of a named-gradient map by `factor`.
pub fn scale_named_gradients<T: Float + Debug + Send + Sync + 'static>(
    gradients: &HashMap<String, Array1<T>>,
    factor: T,
) -> HashMap<String, Array1<T>> {
    gradients
        .iter()
        .map(|(name, grad)| (name.clone(), grad.mapv(|g| g * factor)))
        .collect()
}

/// Accumulate `addend` into `base`, matching entries by name.
///
/// Returns [`OptimError::DimensionMismatch`] when a shared key has mismatched
/// lengths; keys present only in `addend` are inserted as-is.
pub fn add_named_gradients<T: Float + Debug + Send + Sync + 'static>(
    base: &mut HashMap<String, Array1<T>>,
    addend: HashMap<String, Array1<T>>,
) -> Result<()> {
    for (name, grad) in addend {
        match base.get_mut(&name) {
            Some(target) => {
                if target.len() != grad.len() {
                    return Err(OptimError::DimensionMismatch(format!(
                        "gradient '{name}' has length {} in one term and {} in the other",
                        target.len(),
                        grad.len()
                    )));
                }
                for i in 0..target.len() {
                    target[i] = target[i] + grad[i];
                }
            }
            None => {
                base.insert(name, grad);
            }
        }
    }
    Ok(())
}

/// Policy network interface for RL optimizers.
///
/// # Parameter update contract
///
/// [`PolicyNetwork::update_parameters`] receives a **parameter delta**, not a raw
/// gradient: the optimizer has already applied the learning rate, the gradient
/// clipping and the sign (descent on the loss). Implementations must therefore
/// *add* the supplied arrays to their parameters. Every optimizer in this module
/// (policy gradient, trust region, natural gradient, target-network soft updates)
/// relies on this additive semantics.
///
/// # Gradient oracle
///
/// The `*_gradient` methods form the differentiable path used by every learning
/// rule here. They have no meaningful default, so the default bodies return
/// [`OptimError::UnsupportedOperation`] — a policy that cannot differentiate
/// itself must fail loudly rather than be "trained" with a fabricated gradient.
/// [`linear_models`] provides ready-made analytic implementations.
pub trait PolicyNetwork<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluate actions for given observations
    fn evaluate_actions(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<PolicyEvaluation<T>>;

    /// Get action distribution for given observations
    fn get_action_distribution(&self, observations: &Array2<T>) -> Result<ActionDistribution<T>>;

    /// Add a parameter delta to the policy parameters (see the trait docs).
    fn update_parameters(&mut self, deltas: &HashMap<String, Array1<T>>) -> Result<()>;

    /// Get current policy parameters
    fn get_parameters(&self) -> HashMap<String, Array1<T>>;

    /// Gradient of a coefficient-weighted sum of log-probabilities:
    /// `∂/∂θ Σᵢ cᵢ · log π(aᵢ | sᵢ)`.
    ///
    /// Every surrogate loss implemented in this module — REINFORCE, A2C/A3C,
    /// PPO-clip, PPO adaptive-KL, V-trace/IMPALA — has a policy gradient of
    /// exactly this shape with `cᵢ = ∂L/∂ log π(aᵢ|sᵢ)`, so this single oracle is
    /// enough to train all of them end to end.
    ///
    /// The returned map must have the same keys and lengths as
    /// [`Self::get_parameters`].
    fn log_prob_gradient(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
        coefficients: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let _ = (observations, actions, coefficients);
        Err(OptimError::UnsupportedOperation(
            "PolicyNetwork::log_prob_gradient is not implemented for this policy; \
             policy-gradient updates require an analytic (or autodiff) score function"
                .to_string(),
        ))
    }

    /// Gradient of the batch-mean entropy `∂/∂θ (1/N) Σᵢ H[π(·|sᵢ)]`.
    ///
    /// Only consulted when the entropy coefficient is non-zero.
    fn entropy_gradient(&self, observations: &Array2<T>) -> Result<HashMap<String, Array1<T>>> {
        let _ = observations;
        Err(OptimError::UnsupportedOperation(
            "PolicyNetwork::entropy_gradient is not implemented for this policy; \
             set entropy_coeff = 0 or provide an analytic entropy gradient"
                .to_string(),
        ))
    }

    /// Gradient of a weighted sum of the distribution mean:
    /// `∂/∂θ Σᵢ Σⱼ w[i,j] · μⱼ(sᵢ)`.
    ///
    /// This is the chain-rule hook required by the *deterministic* policy gradient
    /// (DDPG/TD3) and by the reparameterized SAC actor update, where the loss
    /// depends on the parameters through the sampled action rather than through
    /// the log-probability.
    fn mean_action_gradient(
        &self,
        observations: &Array2<T>,
        weights: &Array2<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let _ = (observations, weights);
        Err(OptimError::UnsupportedOperation(
            "PolicyNetwork::mean_action_gradient is not implemented for this policy; \
             deterministic-policy-gradient updates (DDPG/TD3/SAC actor) require it"
                .to_string(),
        ))
    }

    /// Per-sample score vectors `g_i = ∇_θ log π(aᵢ|sᵢ)`, one per **row**, flattened
    /// with [`flatten_named`]'s canonical layout.
    ///
    /// Used to build empirical / block-diagonal Fisher estimates. The default
    /// implementation derives them from [`Self::log_prob_gradient`] one sample at a
    /// time, which is correct but costs `N` oracle calls; policies that can produce
    /// them in one pass should override it.
    fn score_matrix(&self, observations: &Array2<T>, actions: &Array2<T>) -> Result<Array2<T>> {
        let n = observations.nrows();
        let dim = parameter_count(&self.get_parameters());
        let mut scores = Array2::zeros((n, dim));

        let one = Array1::from_elem(1, T::one());
        for i in 0..n {
            let obs_i = observations.slice(s![i..i + 1, ..]).to_owned();
            let act_i = actions.slice(s![i..i + 1, ..]).to_owned();
            let grad = self.log_prob_gradient(&obs_i, &act_i, &one)?;
            let flat = flatten_named(&grad);
            if flat.len() != dim {
                return Err(OptimError::DimensionMismatch(format!(
                    "score vector length ({}) does not match parameter count ({})",
                    flat.len(),
                    dim
                )));
            }
            for j in 0..dim {
                scores[[i, j]] = flat[j];
            }
        }

        Ok(scores)
    }

    /// Per-sample Kronecker factors of the Fisher information matrix.
    ///
    /// See [`KroneckerBlock`] for the exact contract. Returning
    /// [`OptimError::UnsupportedOperation`] (the default) makes K-FAC estimation
    /// fail loudly instead of silently degrading to an identity Fisher.
    fn kronecker_factors(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<Vec<KroneckerBlock<T>>> {
        let _ = (observations, actions);
        Err(OptimError::UnsupportedOperation(
            "PolicyNetwork::kronecker_factors is not implemented for this policy; \
             Kronecker-factored Fisher estimation requires per-layer factors"
                .to_string(),
        ))
    }
}

/// Blanket forwarding so a `&mut P` can stand in for an owned policy.
///
/// This lets an optimizer that already owns a policy hand a *borrow* of it to
/// another optimizer (e.g. [`policy_gradient::PolicyGradientOptimizer`] routing
/// its TRPO update through [`trust_region::TrustRegionOptimizer`]) without
/// transferring ownership. Every method — including the gradient oracle — is
/// forwarded, so the borrow behaves exactly like the underlying policy.
impl<T: Float + Debug + Send + Sync + 'static, P: PolicyNetwork<T> + ?Sized> PolicyNetwork<T>
    for &mut P
{
    fn evaluate_actions(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<PolicyEvaluation<T>> {
        (**self).evaluate_actions(observations, actions)
    }

    fn get_action_distribution(&self, observations: &Array2<T>) -> Result<ActionDistribution<T>> {
        (**self).get_action_distribution(observations)
    }

    fn update_parameters(&mut self, deltas: &HashMap<String, Array1<T>>) -> Result<()> {
        (**self).update_parameters(deltas)
    }

    fn get_parameters(&self) -> HashMap<String, Array1<T>> {
        (**self).get_parameters()
    }

    fn log_prob_gradient(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
        coefficients: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        (**self).log_prob_gradient(observations, actions, coefficients)
    }

    fn entropy_gradient(&self, observations: &Array2<T>) -> Result<HashMap<String, Array1<T>>> {
        (**self).entropy_gradient(observations)
    }

    fn mean_action_gradient(
        &self,
        observations: &Array2<T>,
        weights: &Array2<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        (**self).mean_action_gradient(observations, weights)
    }

    fn score_matrix(&self, observations: &Array2<T>, actions: &Array2<T>) -> Result<Array2<T>> {
        (**self).score_matrix(observations, actions)
    }

    fn kronecker_factors(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<Vec<KroneckerBlock<T>>> {
        (**self).kronecker_factors(observations, actions)
    }
}

/// Value network interface for RL optimizers.
///
/// [`ValueNetwork::update_parameters`] follows the same additive **delta**
/// contract as [`PolicyNetwork::update_parameters`].
pub trait ValueNetwork<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluate value function for given observations
    fn evaluate_value(&self, observations: &Array2<T>) -> Result<Array1<T>>;

    /// Add a parameter delta to the value-function parameters.
    fn update_parameters(&mut self, deltas: &HashMap<String, Array1<T>>) -> Result<()>;

    /// Get current value function parameters
    fn get_parameters(&self) -> HashMap<String, Array1<T>>;

    /// Gradient of a residual-weighted sum of value predictions:
    /// `∂/∂θ Σᵢ rᵢ · V(sᵢ)`.
    ///
    /// Callers pass `rᵢ = ∂L/∂V(sᵢ)`; for the mean-squared value loss
    /// `L = (1/N) Σ (V(sᵢ) − yᵢ)²` that is `rᵢ = 2(V(sᵢ) − yᵢ)/N`.
    fn value_gradient(
        &self,
        observations: &Array2<T>,
        residuals: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let _ = (observations, residuals);
        Err(OptimError::UnsupportedOperation(
            "ValueNetwork::value_gradient is not implemented for this network; \
             value-function updates require an analytic (or autodiff) gradient"
                .to_string(),
        ))
    }
}

/// Action-value (Q) network interface.
///
/// The off-policy actor-critic methods (SAC, TD3, DDPG) are built on `Q(s, a)`,
/// not on a state value `V(s)`: without the action argument the deterministic
/// policy gradient `∇_a Q(s, a)` does not exist and the critic cannot distinguish
/// the actions it is supposed to rank.
///
/// A pure Q network has no intrinsic state value, so it is free to return
/// [`OptimError::UnsupportedOperation`] from
/// [`ValueNetwork::evaluate_value`] — see [`linear_models::LinearQFunction`].
pub trait QNetwork<T: Float + Debug + Send + Sync + 'static>: ValueNetwork<T> {
    /// Evaluate `Q(s, a)` for a batch of state-action pairs.
    fn evaluate_q(&self, states: &Array2<T>, actions: &Array2<T>) -> Result<Array1<T>>;

    /// Gradient of a residual-weighted sum of Q predictions:
    /// `∂/∂θ Σᵢ rᵢ · Q(sᵢ, aᵢ)`.
    fn q_gradient(
        &self,
        states: &Array2<T>,
        actions: &Array2<T>,
        residuals: &Array1<T>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let _ = (states, actions, residuals);
        Err(OptimError::UnsupportedOperation(
            "QNetwork::q_gradient is not implemented for this critic".to_string(),
        ))
    }

    /// `∇_a Q(s, a)` for each row, shape `(n_samples, action_dim)`.
    ///
    /// This is the term the deterministic policy gradient chains with
    /// [`PolicyNetwork::mean_action_gradient`].
    fn action_gradient(&self, states: &Array2<T>, actions: &Array2<T>) -> Result<Array2<T>> {
        let _ = (states, actions);
        Err(OptimError::UnsupportedOperation(
            "QNetwork::action_gradient is not implemented for this critic; \
             the deterministic policy gradient requires ∇_a Q(s, a)"
                .to_string(),
        ))
    }
}

/// Policy evaluation results
#[derive(Debug, Clone)]
pub struct PolicyEvaluation<T: Float + Debug + Send + Sync + 'static> {
    /// Log probabilities of actions
    pub log_probs: Array1<T>,

    /// Entropy of action distribution
    pub entropy: Array1<T>,

    /// Additional metrics
    pub metrics: HashMap<String, T>,
}

/// Action distribution representation
#[derive(Debug, Clone)]
pub struct ActionDistribution<T: Float + Debug + Send + Sync + 'static> {
    /// Mean of the distribution (for continuous actions)
    pub mean: Option<Array2<T>>,

    /// Standard deviation (for continuous actions)
    pub std: Option<Array2<T>>,

    /// Logits (for discrete actions)
    pub logits: Option<Array2<T>>,

    /// Distribution type
    pub distribution_type: DistributionType,
}

/// Types of action distributions
#[derive(Debug, Clone, Copy)]
pub enum DistributionType {
    /// Continuous Gaussian distribution
    Gaussian,

    /// Discrete categorical distribution
    Categorical,

    /// Beta distribution (for bounded continuous actions)
    Beta,

    /// Mixed discrete-continuous
    Mixed,
}

/// Learning rate scheduling for RL optimizers
#[derive(Debug, Clone)]
pub struct RLScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Initial learning rate
    pub initiallr: T,

    /// Current learning rate
    pub current_lr: T,

    /// Decay factor
    pub decay_factor: T,

    /// Decay schedule
    pub schedule: ScheduleType,

    /// Number of updates so far
    pub update_count: usize,

    /// Schedule parameters
    pub schedule_params: HashMap<String, T>,
}

/// Learning rate schedule types
#[derive(Debug, Clone, Copy)]
pub enum ScheduleType {
    /// Constant learning rate
    Constant,

    /// Linear decay
    Linear,

    /// Exponential decay
    Exponential,

    /// Cosine annealing
    Cosine,

    /// Step decay
    Step,

    /// Adaptive based on performance
    Adaptive,
}

impl<T: Float + Debug + Send + Sync + 'static> RLScheduler<T> {
    /// Create a new learning rate scheduler
    pub fn new(initiallr: T, schedule: ScheduleType) -> Self {
        Self {
            initiallr,
            current_lr: initiallr,
            decay_factor: T::from(0.99).unwrap_or_else(|| T::zero()),
            schedule,
            update_count: 0,
            schedule_params: HashMap::new(),
        }
    }

    /// Update learning rate based on schedule
    pub fn step(&mut self) -> T {
        self.update_count += 1;

        match self.schedule {
            ScheduleType::Constant => {
                // No change
            }
            ScheduleType::Linear => {
                let decay_steps = self
                    .schedule_params
                    .get("decay_steps")
                    .copied()
                    .unwrap_or(T::from(10000).unwrap_or_else(|| T::zero()));
                let progress =
                    T::from(self.update_count).unwrap_or_else(|| T::zero()) / decay_steps;
                self.current_lr = self.initiallr * (T::one() - progress).max(T::zero());
            }
            ScheduleType::Exponential => {
                self.current_lr = self.current_lr * self.decay_factor;
            }
            ScheduleType::Step => {
                let step_size = self
                    .schedule_params
                    .get("step_size")
                    .copied()
                    .unwrap_or(T::from(1000).unwrap_or_else(|| T::zero()));
                if T::from(self.update_count).unwrap_or_else(|| T::zero()) % step_size == T::zero()
                {
                    self.current_lr = self.current_lr * self.decay_factor;
                }
            }
            ScheduleType::Cosine => {
                let max_steps = self
                    .schedule_params
                    .get("max_steps")
                    .copied()
                    .unwrap_or(T::from(10000).unwrap_or_else(|| T::zero()));
                let progress = T::from(self.update_count).unwrap_or_else(|| T::zero()) / max_steps;
                let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::zero());
                self.current_lr = self.initiallr * (T::one() + (pi * progress).cos())
                    / T::from(2).unwrap_or_else(|| T::zero());
            }
            ScheduleType::Adaptive => {
                // Adaptive scheduling based on performance metrics
                // Implementation depends on specific performance indicators
            }
        }

        self.current_lr
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> T {
        self.current_lr
    }

    /// Set schedule parameter
    pub fn set_param(&mut self, key: &str, value: T) {
        self.schedule_params.insert(key.to_string(), value);
    }
}

/// RL optimization metrics
#[derive(Debug, Clone)]
pub struct RLOptimizationMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Policy loss
    pub policy_loss: T,

    /// Value function loss
    pub value_loss: T,

    /// Entropy loss
    pub entropy_loss: T,

    /// Total loss
    pub total_loss: T,

    /// KL divergence (for trust region methods)
    pub kl_divergence: Option<T>,

    /// Explained variance
    pub explained_variance: T,

    /// Clip fraction (for PPO)
    pub clip_fraction: Option<T>,

    /// Learning rates
    pub policy_lr: T,
    pub value_lr: T,

    /// Gradient norms
    pub policy_grad_norm: T,
    pub value_grad_norm: T,

    /// Additional metrics
    pub custom_metrics: HashMap<String, T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for RLOptimizationMetrics<T> {
    fn default() -> Self {
        Self {
            policy_loss: T::zero(),
            value_loss: T::zero(),
            entropy_loss: T::zero(),
            total_loss: T::zero(),
            kl_divergence: None,
            explained_variance: T::zero(),
            clip_fraction: None,
            policy_lr: T::from(3e-4).unwrap_or_else(|| T::zero()),
            value_lr: T::from(1e-3).unwrap_or_else(|| T::zero()),
            policy_grad_norm: T::zero(),
            value_grad_norm: T::zero(),
            custom_metrics: HashMap::new(),
        }
    }
}

// Import slice syntax
use scirs2_core::ndarray::s;
// use statrs::statistics::Statistics; // statrs not available
