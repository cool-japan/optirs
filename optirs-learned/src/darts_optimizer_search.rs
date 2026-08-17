//! # DARTS over Optimizer Update Rules
//!
//! This module applies **Differentiable Architecture Search (DARTS)** to the space of
//! *optimizer update rules* rather than to neural-network cells. Instead of searching
//! over convolutions and skip-connections, the search operates over a set of primitive
//! per-parameter update operations (plain gradient, momentum, RMSProp, sign, an
//! Adam-like rule and weight decay). A continuous relaxation over these primitives is
//! optimized with a bilevel-style alternating procedure, and the result is discretized
//! into a concrete optimizer that implements [`AdvancedOptimizer`].
//!
//! ## 1. Continuous relaxation
//!
//! Each primitive `i` is a real map `op_i(grad, params, state_i) -> direction_i`. We
//! attach one architecture logit `α_i` to every primitive and define the **mixed
//! update** as a softmax convex combination of the primitive directions:
//!
//! ```text
//! w_i      = softmax(α)_i                       (architecture weights, w_i > 0, Σ w_i = 1)
//! mixed    = Σ_i w_i · op_i(grad, params, state_i)
//! params' := params - lr · mixed
//! ```
//!
//! Because the architecture weights come from a softmax they are strictly positive and
//! sum to one, so `mixed` is a genuine convex blend of the candidate directions.
//!
//! ## 2. Bilevel-style alternating optimization
//!
//! The search alternates an **inner (training)** parameter step with an **outer
//! (architecture)** step that reduces a held-out **validation** loss. Within one inner
//! step we move the candidate parameters `w` with the mixed update derived from the
//! *training* gradient, then we update `α` to reduce the validation loss evaluated at
//! the *stepped* point.
//!
//! The architecture gradient is a genuine closed-form derivative (no random walk). For a
//! single inner step, define the look-ahead validation objective
//!
//! ```text
//! φ(α) = L_val( w - lr · mixed(α) ),   mixed(α) = Σ_i softmax(α)_i · u_i
//! ```
//!
//! where `u_i = op_i(grad, params, state_i)` do **not** depend on `α` (they only depend
//! on the training gradient and each op's own state). The exact total derivative is
//!
//! ```text
//! ∂φ/∂α_j = ∇L_val(w_new) · ( -lr · ∂mixed/∂α_j )
//! ∂mixed/∂α_j = Σ_i softmax_i (δ_ij - softmax_j) u_i = softmax_j ( u_j - mixed )
//! ⇒ ∂φ/∂α_j = -lr · softmax_j · ⟨ g_val , u_j - mixed ⟩ ,   g_val = ∇L_val(w_new).
//! ```
//!
//! Evaluated at `w_new = w - lr · mixed(α)` this is the *exact* gradient of `φ` (the
//! Jacobian of the softmax is folded in analytically). When more than one inner step is
//! taken per epoch we accumulate the per-step look-ahead gradients and average them — a
//! first-order meta-gradient (the validation-gradient readout `g_val` is treated as the
//! straight-through term, exactly as in first-order DARTS), which remains a sound descent
//! direction on the validation loss. The architecture logits are then updated by gradient
//! descent: `α ← α - arch_lr · ∂φ/∂α`.
//!
//! ## 3. Discretization
//!
//! After the search, the architecture is discretized by taking the top-`k` primitives by
//! softmax weight (`k = 1` recovers the usual `argmax`). Their weights are renormalized to
//! sum to one and packaged into a [`DiscoveredOptimizer`] that applies only the winning
//! op(s) and implements the crate's [`AdvancedOptimizer`] trait.
//!
//! ## Determinism
//!
//! Given a fixed [`DartsConfig::seed`], the only stochastic element — the initial
//! candidate parameters — is drawn from a seeded generator, and every other computation is
//! deterministic floating-point arithmetic. Two searches with identical configuration and
//! seed therefore produce identical results.

use crate::domain_optimizers::{l2_norm, AdvancedOptimizer, OptimizerStateInfo};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::fmt::Debug;
use std::sync::Arc;

/// Infallible numeric cast for internal magnitudes that always represent in the target
/// float type (counts, fixed constants). Falls back to zero only for exotic `T` that
/// cannot represent the value, which never occurs for `f32`/`f64`.
fn cast<T: Float>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::zero)
}

/// Checked numeric cast used at API boundaries (hyper-parameter conversion). Returns an
/// error instead of silently substituting a value if `T` cannot represent `value`.
fn to_scalar<T: Float>(value: f64) -> Result<T> {
    T::from(value).ok_or_else(|| {
        OptimError::ComputationError(format!(
            "cannot represent {value} in the target floating-point type"
        ))
    })
}

/// Numerically-stable softmax of an architecture-logit vector.
///
/// The maximum logit is subtracted before exponentiation for stability. The returned
/// vector is guaranteed to be strictly positive (every exponential is positive) and to
/// sum to one (the maximum element contributes `exp(0) = 1`, so the denominator is at
/// least one and never zero).
pub(crate) fn softmax<T: Float>(logits: &Array1<T>) -> Array1<T> {
    if logits.is_empty() {
        return Array1::zeros(0);
    }
    let max_logit = logits
        .iter()
        .fold(T::neg_infinity(), |acc, &x| if x > acc { x } else { acc });
    let exps = logits.mapv(|x| (x - max_logit).exp());
    let sum = exps.iter().fold(T::zero(), |acc, &x| acc + x);
    exps.mapv(|x| x / sum)
}

/// Convex combination `Σ_i weights_i · ops_i`.
fn mixed_update<T: Float>(weights: &Array1<T>, ops: &[Array1<T>]) -> Array1<T> {
    let dim = ops.first().map_or(0, Array1::len);
    let mut acc = Array1::<T>::zeros(dim);
    for (&weight, op) in weights.iter().zip(ops.iter()) {
        for (slot, &value) in acc.iter_mut().zip(op.iter()) {
            *slot = *slot + value * weight;
        }
    }
    acc
}

/// Closed-form architecture gradient `∂φ/∂α_j = -lr · softmax_j · ⟨g_val, u_j - mixed⟩`.
///
/// This is the analytic derivative of the one-step look-ahead validation objective with
/// respect to the architecture logits (see the module documentation).
fn architecture_gradient<T: Float>(
    weights: &Array1<T>,
    ops: &[Array1<T>],
    mixed: &Array1<T>,
    g_val: &Array1<T>,
    lr: T,
) -> Array1<T> {
    let grads: Vec<T> = ops
        .iter()
        .zip(weights.iter())
        .map(|(op, &weight)| {
            let directional = op
                .iter()
                .zip(mixed.iter())
                .zip(g_val.iter())
                .fold(T::zero(), |acc, ((&u, &m), &g)| acc + g * (u - m));
            -lr * weight * directional
        })
        .collect();
    Array1::from_vec(grads)
}

/// A candidate primitive update operation that the search mixes over.
///
/// Every variant is a real per-parameter map `(grad, params, state) -> direction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpdatePrimitive {
    /// Plain gradient descent direction: `direction = grad`.
    Grad,
    /// Exponential moving average of the gradient (classical momentum):
    /// `m ← β m + (1-β) grad`, `direction = m`.
    Momentum,
    /// Root-mean-square propagation: `v ← γ v + (1-γ) grad²`,
    /// `direction = grad / (sqrt(v) + ε)`.
    RmsProp,
    /// Sign of the gradient: `direction = sign(grad)`.
    Sign,
    /// Adam-like rule with bias correction: `direction = m̂ / (sqrt(v̂) + ε)`.
    AdamLike,
    /// Decoupled weight decay: `direction = λ · params` (shrinks parameters toward zero).
    WeightDecay,
}

impl UpdatePrimitive {
    /// Stable human-readable name of the primitive.
    pub fn name(&self) -> &'static str {
        match self {
            UpdatePrimitive::Grad => "Grad",
            UpdatePrimitive::Momentum => "Momentum",
            UpdatePrimitive::RmsProp => "RmsProp",
            UpdatePrimitive::Sign => "Sign",
            UpdatePrimitive::AdamLike => "AdamLike",
            UpdatePrimitive::WeightDecay => "WeightDecay",
        }
    }

    /// The full default search set, in canonical order.
    pub fn default_set() -> Vec<UpdatePrimitive> {
        vec![
            UpdatePrimitive::Grad,
            UpdatePrimitive::Momentum,
            UpdatePrimitive::RmsProp,
            UpdatePrimitive::Sign,
            UpdatePrimitive::AdamLike,
            UpdatePrimitive::WeightDecay,
        ]
    }

    /// Whether this primitive maintains internal state across steps.
    pub fn is_stateful(&self) -> bool {
        matches!(
            self,
            UpdatePrimitive::Momentum | UpdatePrimitive::RmsProp | UpdatePrimitive::AdamLike
        )
    }

    /// Apply the primitive, returning its proposed update direction.
    ///
    /// `state` is mutated in place for the stateful primitives (momentum / second-moment
    /// accumulators and the Adam step counter); stateless primitives leave it untouched.
    pub fn apply<T>(
        &self,
        grad: &Array1<T>,
        params: &Array1<T>,
        state: &mut PrimitiveState<T>,
        hp: &PrimitiveHyperparams<T>,
    ) -> Array1<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        match self {
            UpdatePrimitive::Grad => grad.clone(),
            UpdatePrimitive::Sign => grad.mapv(|g| {
                // True sign: a zero gradient yields no update (unlike `f64::signum`,
                // which returns +1 for +0.0).
                if g > T::zero() {
                    T::one()
                } else if g < T::zero() {
                    -T::one()
                } else {
                    T::zero()
                }
            }),
            UpdatePrimitive::Momentum => {
                let beta = hp.momentum_beta;
                let one_minus = T::one() - beta;
                state
                    .momentum
                    .zip_mut_with(grad, |m, &g| *m = beta * *m + one_minus * g);
                state.momentum.clone()
            }
            UpdatePrimitive::RmsProp => {
                let decay = hp.rms_decay;
                let one_minus = T::one() - decay;
                let eps = hp.epsilon;
                state
                    .second_moment
                    .zip_mut_with(grad, |v, &g| *v = decay * *v + one_minus * g * g);
                let mut out = grad.clone();
                out.zip_mut_with(&state.second_moment, |o, &v| *o = *o / (v.sqrt() + eps));
                out
            }
            UpdatePrimitive::AdamLike => {
                state.step += 1;
                let beta1 = hp.adam_beta1;
                let beta2 = hp.adam_beta2;
                let eps = hp.epsilon;
                let one_minus1 = T::one() - beta1;
                let one_minus2 = T::one() - beta2;
                state
                    .momentum
                    .zip_mut_with(grad, |m, &g| *m = beta1 * *m + one_minus1 * g);
                state
                    .second_moment
                    .zip_mut_with(grad, |v, &g| *v = beta2 * *v + one_minus2 * g * g);
                let step_exp = i32::try_from(state.step).unwrap_or(i32::MAX);
                let bias1 = T::one() - beta1.powi(step_exp);
                let bias2 = T::one() - beta2.powi(step_exp);
                let mut out = Array1::<T>::zeros(grad.len());
                out.zip_mut_with(&state.momentum, |o, &m| *o = m / bias1);
                out.zip_mut_with(&state.second_moment, |o, &v| {
                    let v_hat = (v / bias2).sqrt();
                    *o = *o / (v_hat + eps);
                });
                out
            }
            UpdatePrimitive::WeightDecay => params.mapv(|p| p * hp.weight_decay_strength),
        }
    }
}

/// Per-operation state shared by the stateful primitives.
///
/// A separate instance is held for every primitive in the search so that, for example,
/// the momentum accumulator of [`UpdatePrimitive::Momentum`] is independent of the
/// first-moment accumulator of [`UpdatePrimitive::AdamLike`].
#[derive(Debug, Clone)]
pub struct PrimitiveState<T: Float + Debug + Send + Sync + 'static> {
    /// First-moment (momentum) accumulator.
    momentum: Array1<T>,
    /// Second-moment (squared-gradient) accumulator.
    second_moment: Array1<T>,
    /// Step counter (used for Adam bias correction).
    step: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> PrimitiveState<T> {
    /// Create a fresh zero-initialized state for a `dim`-dimensional parameter vector.
    pub fn new(dim: usize) -> Self {
        Self {
            momentum: Array1::zeros(dim),
            second_moment: Array1::zeros(dim),
            step: 0,
        }
    }
}

/// Hyper-parameters shared by the primitive operations, converted once to the target
/// float type.
#[derive(Debug, Clone)]
pub struct PrimitiveHyperparams<T: Float + Debug + Send + Sync + 'static> {
    /// Momentum EMA coefficient `β` for [`UpdatePrimitive::Momentum`].
    pub momentum_beta: T,
    /// First-moment coefficient `β₁` for [`UpdatePrimitive::AdamLike`].
    pub adam_beta1: T,
    /// Second-moment coefficient `β₂` for [`UpdatePrimitive::AdamLike`].
    pub adam_beta2: T,
    /// Squared-gradient EMA coefficient `γ` for [`UpdatePrimitive::RmsProp`].
    pub rms_decay: T,
    /// Numerical stabilizer added before division.
    pub epsilon: T,
    /// Decoupled weight-decay strength `λ` for [`UpdatePrimitive::WeightDecay`].
    pub weight_decay_strength: T,
}

impl<T: Float + Debug + Send + Sync + 'static> PrimitiveHyperparams<T> {
    /// Build the hyper-parameter bundle from a configuration, converting each value to `T`.
    fn from_config(config: &DartsConfig) -> Result<Self> {
        Ok(Self {
            momentum_beta: to_scalar(config.momentum_beta)?,
            adam_beta1: to_scalar(config.adam_beta1)?,
            adam_beta2: to_scalar(config.adam_beta2)?,
            rms_decay: to_scalar(config.rms_decay)?,
            epsilon: to_scalar(config.epsilon)?,
            weight_decay_strength: to_scalar(config.weight_decay_strength)?,
        })
    }
}

/// Boxed loss-and-gradient closure used by [`ClosureObjective`].
type LossGradFn<T> = Arc<dyn Fn(&Array1<T>) -> Result<(T, Array1<T>)> + Send + Sync>;

/// A differentiable scalar objective over the candidate parameters.
///
/// The optimizer search needs both a *training* signal (used to take the inner parameter
/// step) and a *validation* signal (used to drive the architecture step). Implementors
/// return `(loss, gradient)` for each. The two may coincide.
pub trait DifferentiableObjective<T: Float + Debug + Send + Sync + 'static> {
    /// Training loss and its gradient at `params`.
    fn train_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)>;
    /// Validation loss and its gradient at `params`.
    fn validation_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)>;
    /// Dimensionality of the parameter space.
    fn dim(&self) -> usize;
}

/// A separable quadratic bowl `L(w) = ½ Σ c_i (w_i - center_i)²`.
///
/// Training and validation halves may use different curvatures / centers, which makes the
/// bilevel search non-trivial; by default they coincide and the bowl is centered at the
/// origin.
#[derive(Debug, Clone)]
pub struct QuadraticBowl<T: Float + Debug + Send + Sync + 'static> {
    train_curvature: Array1<T>,
    train_center: Array1<T>,
    val_curvature: Array1<T>,
    val_center: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> QuadraticBowl<T> {
    /// Build a bowl with the given per-coordinate `curvature`, centered at the origin,
    /// using the same function for training and validation.
    pub fn new(curvature: Array1<T>) -> Self {
        let center = Array1::zeros(curvature.len());
        Self {
            train_curvature: curvature.clone(),
            train_center: center.clone(),
            val_curvature: curvature,
            val_center: center,
        }
    }

    /// Build an isotropic `dim`-dimensional bowl with a constant `curvature`.
    pub fn isotropic(dim: usize, curvature: T) -> Result<Self> {
        if dim == 0 {
            return Err(OptimError::InvalidConfig(
                "QuadraticBowl dimension must be positive".to_string(),
            ));
        }
        Ok(Self::new(Array1::from_elem(dim, curvature)))
    }

    /// Override the validation half with a separate curvature and center.
    pub fn with_validation(mut self, curvature: Array1<T>, center: Array1<T>) -> Result<Self> {
        if curvature.len() != self.train_curvature.len() || center.len() != self.train_center.len()
        {
            return Err(OptimError::InvalidConfig(
                "validation curvature/center dimension must match the training half".to_string(),
            ));
        }
        self.val_curvature = curvature;
        self.val_center = center;
        Ok(self)
    }

    fn loss_grad(
        curvature: &Array1<T>,
        center: &Array1<T>,
        params: &Array1<T>,
    ) -> Result<(T, Array1<T>)> {
        if params.len() != curvature.len() {
            return Err(OptimError::InvalidConfig(format!(
                "parameter length {} != objective dimension {}",
                params.len(),
                curvature.len()
            )));
        }
        let half = cast::<T>(0.5);
        let mut loss = T::zero();
        let mut grad = Array1::<T>::zeros(params.len());
        for (slot, ((&c, &p), &ctr)) in grad
            .iter_mut()
            .zip(curvature.iter().zip(params.iter()).zip(center.iter()))
        {
            let diff = p - ctr;
            *slot = c * diff;
            loss = loss + half * c * diff * diff;
        }
        Ok((loss, grad))
    }
}

impl<T: Float + Debug + Send + Sync + 'static> DifferentiableObjective<T> for QuadraticBowl<T> {
    fn train_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        Self::loss_grad(&self.train_curvature, &self.train_center, params)
    }

    fn validation_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        Self::loss_grad(&self.val_curvature, &self.val_center, params)
    }

    fn dim(&self) -> usize {
        self.train_curvature.len()
    }
}

/// The classical Rosenbrock valley
/// `L(w) = Σ_i [ b (w_{i+1} - w_i²)² + (a - w_i)² ]`.
///
/// A non-convex, ill-conditioned stress test for optimizer search. Training and validation
/// use the same function.
#[derive(Debug, Clone)]
pub struct Rosenbrock<T: Float + Debug + Send + Sync + 'static> {
    dim: usize,
    a: T,
    b: T,
}

impl<T: Float + Debug + Send + Sync + 'static> Rosenbrock<T> {
    /// Build a `dim`-dimensional Rosenbrock function with the standard constants
    /// `a = 1`, `b = 100`. Requires `dim >= 2`.
    pub fn new(dim: usize) -> Result<Self> {
        if dim < 2 {
            return Err(OptimError::InvalidConfig(
                "Rosenbrock requires at least two dimensions".to_string(),
            ));
        }
        Ok(Self {
            dim,
            a: T::one(),
            b: cast::<T>(100.0),
        })
    }

    /// Override the `a` and `b` constants.
    pub fn with_constants(mut self, a: T, b: T) -> Self {
        self.a = a;
        self.b = b;
        self
    }

    fn loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        if params.len() != self.dim {
            return Err(OptimError::InvalidConfig(format!(
                "parameter length {} != objective dimension {}",
                params.len(),
                self.dim
            )));
        }
        let two = cast::<T>(2.0);
        let mut loss = T::zero();
        let mut grad = Array1::<T>::zeros(self.dim);
        for i in 0..self.dim - 1 {
            let xi = params[i];
            let xi1 = params[i + 1];
            let residual = xi1 - xi * xi;
            let offset = self.a - xi;
            loss = loss + self.b * residual * residual + offset * offset;
            grad[i] = grad[i] + self.b * two * residual * (-two * xi) - two * offset;
            grad[i + 1] = grad[i + 1] + self.b * two * residual;
        }
        Ok((loss, grad))
    }
}

impl<T: Float + Debug + Send + Sync + 'static> DifferentiableObjective<T> for Rosenbrock<T> {
    fn train_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        self.loss_grad(params)
    }

    fn validation_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        self.loss_grad(params)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// A pluggable objective backed by user-provided closures.
#[derive(Clone)]
pub struct ClosureObjective<T: Float + Debug + Send + Sync + 'static> {
    dim: usize,
    train: LossGradFn<T>,
    validation: LossGradFn<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ClosureObjective<T> {
    /// Build an objective from independent training and validation closures.
    pub fn new<F, G>(dim: usize, train: F, validation: G) -> Result<Self>
    where
        F: Fn(&Array1<T>) -> Result<(T, Array1<T>)> + Send + Sync + 'static,
        G: Fn(&Array1<T>) -> Result<(T, Array1<T>)> + Send + Sync + 'static,
    {
        if dim == 0 {
            return Err(OptimError::InvalidConfig(
                "ClosureObjective dimension must be positive".to_string(),
            ));
        }
        Ok(Self {
            dim,
            train: Arc::new(train),
            validation: Arc::new(validation),
        })
    }

    /// Build an objective where the same closure is used for training and validation.
    pub fn from_single<F>(dim: usize, objective: F) -> Result<Self>
    where
        F: Fn(&Array1<T>) -> Result<(T, Array1<T>)> + Send + Sync + 'static,
    {
        if dim == 0 {
            return Err(OptimError::InvalidConfig(
                "ClosureObjective dimension must be positive".to_string(),
            ));
        }
        let shared: LossGradFn<T> = Arc::new(objective);
        Ok(Self {
            dim,
            train: Arc::clone(&shared),
            validation: shared,
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static> DifferentiableObjective<T> for ClosureObjective<T> {
    fn train_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        (self.train)(params)
    }

    fn validation_loss_grad(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        (self.validation)(params)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Configuration for [`DartsOptimizerSearch`].
#[derive(Debug, Clone)]
pub struct DartsConfig {
    /// The candidate set of primitive update operations to search over.
    pub primitives: Vec<UpdatePrimitive>,
    /// Inner (parameter) learning rate `lr`.
    pub learning_rate: f64,
    /// Outer (architecture) learning rate used to update the logits `α`.
    pub arch_learning_rate: f64,
    /// Number of outer search epochs.
    pub epochs: usize,
    /// Number of inner parameter steps taken per epoch.
    pub inner_steps: usize,
    /// Number of primitives retained when discretizing (`1` = `argmax`).
    pub top_k: usize,
    /// If true, the candidate parameters and per-op state are reset to the initial point
    /// at the start of every epoch (fixed-budget search). If false, they persist across
    /// epochs (continuous descent).
    pub reset_each_epoch: bool,
    /// Half-width of the uniform range used to draw the initial candidate parameters.
    pub init_params_scale: f64,
    /// Momentum coefficient `β` for [`UpdatePrimitive::Momentum`].
    pub momentum_beta: f64,
    /// First-moment coefficient `β₁` for [`UpdatePrimitive::AdamLike`].
    pub adam_beta1: f64,
    /// Second-moment coefficient `β₂` for [`UpdatePrimitive::AdamLike`].
    pub adam_beta2: f64,
    /// Squared-gradient EMA coefficient `γ` for [`UpdatePrimitive::RmsProp`].
    pub rms_decay: f64,
    /// Numerical stabilizer added before division.
    pub epsilon: f64,
    /// Decoupled weight-decay strength `λ` for [`UpdatePrimitive::WeightDecay`].
    pub weight_decay_strength: f64,
    /// Seed for the deterministic initial-parameter generator.
    pub seed: u64,
}

impl Default for DartsConfig {
    fn default() -> Self {
        Self {
            primitives: UpdatePrimitive::default_set(),
            learning_rate: 0.1,
            arch_learning_rate: 0.3,
            epochs: 60,
            inner_steps: 5,
            top_k: 1,
            reset_each_epoch: true,
            init_params_scale: 2.0,
            momentum_beta: 0.9,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            rms_decay: 0.9,
            epsilon: 1e-8,
            weight_decay_strength: 0.01,
            seed: 0x0DA2_750F_C0FF_EE11,
        }
    }
}

impl DartsConfig {
    /// Validate the configuration, returning an error describing the first problem found.
    pub fn validate(&self) -> Result<()> {
        if self.primitives.is_empty() {
            return Err(OptimError::InvalidConfig(
                "the primitive search set must not be empty".to_string(),
            ));
        }
        if self.top_k == 0 || self.top_k > self.primitives.len() {
            return Err(OptimError::InvalidConfig(format!(
                "top_k must be in 1..={}, got {}",
                self.primitives.len(),
                self.top_k
            )));
        }
        if self.epochs == 0 {
            return Err(OptimError::InvalidConfig(
                "epochs must be at least 1".to_string(),
            ));
        }
        if self.inner_steps == 0 {
            return Err(OptimError::InvalidConfig(
                "inner_steps must be at least 1".to_string(),
            ));
        }
        Self::require_positive_finite("learning_rate", self.learning_rate)?;
        Self::require_positive_finite("arch_learning_rate", self.arch_learning_rate)?;
        Self::require_positive_finite("init_params_scale", self.init_params_scale)?;
        Self::require_positive_finite("epsilon", self.epsilon)?;
        Self::require_unit_open("momentum_beta", self.momentum_beta)?;
        Self::require_unit_open("adam_beta1", self.adam_beta1)?;
        Self::require_unit_open("adam_beta2", self.adam_beta2)?;
        Self::require_unit_open("rms_decay", self.rms_decay)?;
        if !self.weight_decay_strength.is_finite() || self.weight_decay_strength < 0.0 {
            return Err(OptimError::InvalidConfig(
                "weight_decay_strength must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }

    fn require_positive_finite(field: &str, value: f64) -> Result<()> {
        if !value.is_finite() || value <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "{field} must be finite and positive, got {value}"
            )));
        }
        Ok(())
    }

    fn require_unit_open(field: &str, value: f64) -> Result<()> {
        if !value.is_finite() || value <= 0.0 || value >= 1.0 {
            return Err(OptimError::InvalidConfig(format!(
                "{field} must lie in the open interval (0, 1), got {value}"
            )));
        }
        Ok(())
    }
}

/// The result of a completed architecture search.
#[derive(Debug, Clone)]
pub struct SearchOutcome<T: Float + Debug + Send + Sync + 'static> {
    /// Learned architecture logits `α`.
    pub alpha: Array1<T>,
    /// Architecture weights `softmax(α)` (strictly positive, sum to one).
    pub softmax_weights: Array1<T>,
    /// The primitive set, aligned with `alpha` / `softmax_weights`.
    pub primitives: Vec<UpdatePrimitive>,
    /// Validation loss recorded at the end of each epoch.
    pub validation_trajectory: Vec<T>,
    /// The discretized optimizer applying the winning primitive(s).
    pub discovered: DiscoveredOptimizer<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> SearchOutcome<T> {
    /// The primitive with the largest architecture weight.
    pub fn best_primitive(&self) -> UpdatePrimitive {
        let mut best_index = 0usize;
        let mut best_weight = T::neg_infinity();
        for (index, &weight) in self.softmax_weights.iter().enumerate() {
            if weight > best_weight {
                best_weight = weight;
                best_index = index;
            }
        }
        self.primitives[best_index]
    }

    /// The largest architecture weight.
    pub fn best_weight(&self) -> T {
        self.softmax_weights
            .iter()
            .fold(T::neg_infinity(), |acc, &w| if w > acc { w } else { acc })
    }
}

/// Differentiable architecture search over optimizer update rules.
pub struct DartsOptimizerSearch<T: Float + Debug + Send + Sync + 'static> {
    config: DartsConfig,
    primitives: Vec<UpdatePrimitive>,
    hyperparams: PrimitiveHyperparams<T>,
    learning_rate: T,
    arch_learning_rate: T,
    alpha: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> DartsOptimizerSearch<T> {
    /// Create a new search from a validated configuration.
    ///
    /// The architecture logits are initialized to zero, i.e. a uniform mixture over the
    /// primitive set.
    pub fn new(config: DartsConfig) -> Result<Self> {
        config.validate()?;
        let hyperparams = PrimitiveHyperparams::from_config(&config)?;
        let learning_rate = to_scalar(config.learning_rate)?;
        let arch_learning_rate = to_scalar(config.arch_learning_rate)?;
        let primitives = config.primitives.clone();
        let alpha = Array1::zeros(primitives.len());
        Ok(Self {
            config,
            primitives,
            hyperparams,
            learning_rate,
            arch_learning_rate,
            alpha,
        })
    }

    /// The current architecture logits.
    pub fn alpha(&self) -> &Array1<T> {
        &self.alpha
    }

    /// The current architecture weights `softmax(α)`.
    pub fn architecture_weights(&self) -> Array1<T> {
        softmax(&self.alpha)
    }

    /// The primitive search set.
    pub fn primitives(&self) -> &[UpdatePrimitive] {
        &self.primitives
    }

    /// Allocate one fresh per-primitive state of dimension `dim`.
    fn fresh_states(&self, dim: usize) -> Vec<PrimitiveState<T>> {
        self.primitives
            .iter()
            .map(|_| PrimitiveState::new(dim))
            .collect()
    }

    /// Run the full bilevel architecture search and discretize the result.
    pub fn search<O: DifferentiableObjective<T>>(
        &mut self,
        objective: &O,
    ) -> Result<SearchOutcome<T>> {
        let dim = objective.dim();
        if dim == 0 {
            return Err(OptimError::InsufficientData(
                "objective dimensionality must be positive".to_string(),
            ));
        }

        // Reset architecture logits so repeated searches are independent and deterministic.
        self.alpha = Array1::zeros(self.primitives.len());

        // Draw the (single) initial candidate parameter vector from a seeded generator.
        let mut rng = Random::seed(self.config.seed);
        let scale = self.config.init_params_scale;
        let initial_params = Array1::from_shape_fn(dim, |_| {
            let value: f64 = rng.random_range(-scale..scale);
            cast::<T>(value)
        });

        let lr = self.learning_rate;
        let arch_lr = self.arch_learning_rate;
        let num_primitives = self.primitives.len();
        let inner_steps = self.config.inner_steps;
        let inner_scale = cast::<T>(1.0 / inner_steps as f64);

        let mut params = initial_params.clone();
        let mut states = self.fresh_states(dim);
        let mut trajectory = Vec::with_capacity(self.config.epochs);

        for _epoch in 0..self.config.epochs {
            if self.config.reset_each_epoch {
                params = initial_params.clone();
                states = self.fresh_states(dim);
            }

            let mut alpha_grad = Array1::<T>::zeros(num_primitives);
            let mut epoch_val_loss = T::zero();

            for _inner in 0..inner_steps {
                let weights = softmax(&self.alpha);
                let (_, grad_train) = objective.train_loss_grad(&params)?;

                // Evaluate every primitive (advancing its own state) and form the mix.
                let ops: Vec<Array1<T>> = self
                    .primitives
                    .iter()
                    .zip(states.iter_mut())
                    .map(|(primitive, state)| {
                        primitive.apply(&grad_train, &params, state, &self.hyperparams)
                    })
                    .collect();
                let mixed = mixed_update(&weights, &ops);

                // Inner step: descend the parameters with the mixed update.
                let stepped = &params - &mixed.mapv(|m| m * lr);

                // Outer signal: exact look-ahead architecture gradient at the stepped point.
                let (val_loss, grad_val) = objective.validation_loss_grad(&stepped)?;
                let step_grad = architecture_gradient(&weights, &ops, &mixed, &grad_val, lr);
                alpha_grad = alpha_grad + &step_grad;

                params = stepped;
                epoch_val_loss = val_loss;
            }

            // Architecture (outer) step: gradient descent on the averaged meta-gradient.
            let scaled = alpha_grad.mapv(|g| g * inner_scale * arch_lr);
            self.alpha = &self.alpha - &scaled;

            trajectory.push(epoch_val_loss);
        }

        let discovered = self.discretize()?;
        Ok(SearchOutcome {
            alpha: self.alpha.clone(),
            softmax_weights: softmax(&self.alpha),
            primitives: self.primitives.clone(),
            validation_trajectory: trajectory,
            discovered,
        })
    }

    /// Discretize the relaxed architecture by selecting the top-`k` primitives.
    fn discretize(&self) -> Result<DiscoveredOptimizer<T>> {
        let weights = softmax(&self.alpha);
        let mut order: Vec<usize> = (0..weights.len()).collect();
        order.sort_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let k = self.config.top_k.clamp(1, weights.len());
        let selected_indices = &order[..k];

        let selected_primitives: Vec<UpdatePrimitive> = selected_indices
            .iter()
            .map(|&i| self.primitives[i])
            .collect();
        let raw_weights: Vec<T> = selected_indices.iter().map(|&i| weights[i]).collect();
        let weight_sum = raw_weights.iter().fold(T::zero(), |acc, &w| acc + w);

        // Renormalize the retained weights to sum to one. The maximum weight is positive,
        // so the sum is strictly positive and the division is well-defined.
        let normalized: Array1<T> = if weight_sum > T::zero() {
            Array1::from_vec(raw_weights).mapv(|w| w / weight_sum)
        } else {
            let uniform = cast::<T>(1.0 / k as f64);
            Array1::from_elem(k, uniform)
        };

        DiscoveredOptimizer::new(
            selected_primitives,
            normalized,
            self.hyperparams.clone(),
            lr_of(&self.config)?,
        )
    }

    /// Look-ahead validation loss `φ(α) = L_val(w - lr · mixed(softmax(α)))` for a fixed
    /// set of primitive directions.
    ///
    /// Test-only: the search itself never needs φ evaluated at a perturbed α —
    /// this exists so
    /// `test_analytic_alpha_gradient_matches_finite_difference` can central-
    /// difference it against [`Self::analytic_architecture_gradient`].
    #[cfg(test)]
    fn lookahead_validation_loss<O: DifferentiableObjective<T>>(
        &self,
        alpha: &Array1<T>,
        ops: &[Array1<T>],
        params: &Array1<T>,
        objective: &O,
    ) -> Result<T> {
        let weights = softmax(alpha);
        let mixed = mixed_update(&weights, ops);
        let stepped = params - &mixed.mapv(|m| m * self.learning_rate);
        let (loss, _) = objective.validation_loss_grad(&stepped)?;
        Ok(loss)
    }

    /// Analytic architecture gradient `∂φ/∂α` for a fixed set of primitive directions.
    ///
    /// Test-only: the search computes this inline inside its alternation loop; the
    /// standalone form exists purely so the finite-difference check has a
    /// side-effect-free entry point.
    #[cfg(test)]
    fn analytic_architecture_gradient<O: DifferentiableObjective<T>>(
        &self,
        alpha: &Array1<T>,
        ops: &[Array1<T>],
        params: &Array1<T>,
        objective: &O,
    ) -> Result<Array1<T>> {
        let weights = softmax(alpha);
        let mixed = mixed_update(&weights, ops);
        let stepped = params - &mixed.mapv(|m| m * self.learning_rate);
        let (_, grad_val) = objective.validation_loss_grad(&stepped)?;
        Ok(architecture_gradient(
            &weights,
            ops,
            &mixed,
            &grad_val,
            self.learning_rate,
        ))
    }
}

/// Convert the inner learning rate of a configuration into the target float type.
fn lr_of<T: Float>(config: &DartsConfig) -> Result<T> {
    to_scalar(config.learning_rate)
}

/// A concrete optimizer discovered by [`DartsOptimizerSearch`].
///
/// It applies the winning primitive(s) with weights renormalized to sum to one and
/// implements the crate's [`AdvancedOptimizer`] trait.
#[derive(Debug, Clone)]
pub struct DiscoveredOptimizer<T: Float + Debug + Send + Sync + 'static> {
    primitives: Vec<UpdatePrimitive>,
    weights: Array1<T>,
    hyperparams: PrimitiveHyperparams<T>,
    states: Vec<PrimitiveState<T>>,
    learning_rate: T,
    param_len: usize,
    step_count: usize,
    grad_norm_ema: T,
    norm_ema_decay: T,
}

impl<T: Float + Debug + Send + Sync + 'static> DiscoveredOptimizer<T> {
    /// Build a discovered optimizer from the selected primitives and (already
    /// renormalized) weights.
    fn new(
        primitives: Vec<UpdatePrimitive>,
        weights: Array1<T>,
        hyperparams: PrimitiveHyperparams<T>,
        learning_rate: T,
    ) -> Result<Self> {
        if primitives.is_empty() || primitives.len() != weights.len() {
            return Err(OptimError::InvalidConfig(
                "discovered optimizer requires a non-empty, weight-aligned primitive set"
                    .to_string(),
            ));
        }
        Ok(Self {
            primitives,
            weights,
            hyperparams,
            states: Vec::new(),
            learning_rate,
            param_len: 0,
            step_count: 0,
            grad_norm_ema: T::zero(),
            norm_ema_decay: cast::<T>(0.999),
        })
    }

    /// The primitive(s) retained by the search, ordered by descending weight.
    pub fn selected_primitives(&self) -> &[UpdatePrimitive] {
        &self.primitives
    }

    /// The renormalized weights aligned with [`Self::selected_primitives`].
    pub fn selected_weights(&self) -> &Array1<T> {
        &self.weights
    }

    /// Number of optimization steps performed so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdvancedOptimizer<T> for DiscoveredOptimizer<T> {
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {
        if params.len() != gradients.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Parameter length {} != gradient length {}",
                params.len(),
                gradients.len()
            )));
        }
        if params.is_empty() {
            return Err(OptimError::InsufficientData(
                "Empty parameter array".to_string(),
            ));
        }

        if self.param_len != params.len() {
            self.states = self
                .primitives
                .iter()
                .map(|_| PrimitiveState::new(params.len()))
                .collect();
            self.param_len = params.len();
        }

        let ops: Vec<Array1<T>> = self
            .primitives
            .iter()
            .zip(self.states.iter_mut())
            .map(|(primitive, state)| primitive.apply(gradients, params, state, &self.hyperparams))
            .collect();
        let mixed = mixed_update(&self.weights, &ops);
        let updated = params - &mixed.mapv(|m| m * self.learning_rate);

        let grad_norm = l2_norm(gradients);
        self.grad_norm_ema =
            self.norm_ema_decay * self.grad_norm_ema + (T::one() - self.norm_ema_decay) * grad_norm;
        self.step_count += 1;

        Ok(updated)
    }

    fn get_learning_rate(&self) -> T {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }

    fn name(&self) -> &str {
        "DartsDiscovered"
    }

    fn get_state(&self) -> OptimizerStateInfo<T> {
        OptimizerStateInfo {
            step_count: self.step_count,
            current_lr: self.learning_rate,
            grad_norm_ema: self.grad_norm_ema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(arr: &Array1<f64>) -> f64 {
        arr.iter().fold(0.0, |acc, &x| acc + x * x).sqrt()
    }

    #[test]
    fn test_softmax_positive_and_sums_to_one() {
        let logits = Array1::from_vec(vec![-2.0_f64, 0.5, 3.0, -0.1, 1.2]);
        let weights = softmax(&logits);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "softmax must sum to 1, got {sum}"
        );
        for &w in weights.iter() {
            assert!(
                w > 0.0,
                "softmax weights must be strictly positive, got {w}"
            );
        }
        // The largest logit must receive the largest weight.
        let argmax = weights
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, (i, &w)| {
                if w > best.1 {
                    (i, w)
                } else {
                    best
                }
            })
            .0;
        assert_eq!(argmax, 2, "largest logit should map to largest weight");
    }

    #[test]
    fn test_softmax_numerically_stable_large_logits() {
        let logits = Array1::from_vec(vec![1000.0_f64, 1001.0, 999.0]);
        let weights = softmax(&logits);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        for &w in weights.iter() {
            assert!(w.is_finite() && w > 0.0);
        }
    }

    #[test]
    fn test_update_primitive_apply_directions() {
        let hp = PrimitiveHyperparams::<f64> {
            momentum_beta: 0.9,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            rms_decay: 0.9,
            epsilon: 1e-8,
            weight_decay_strength: 0.5,
        };
        let grad = Array1::from_vec(vec![2.0, -4.0, 0.0]);
        let params = Array1::from_vec(vec![1.0, -1.0, 3.0]);

        let mut state = PrimitiveState::new(3);
        let grad_dir = UpdatePrimitive::Grad.apply(&grad, &params, &mut state, &hp);
        assert_eq!(grad_dir, grad, "Grad must return the gradient unchanged");

        let mut state = PrimitiveState::new(3);
        let sign_dir = UpdatePrimitive::Sign.apply(&grad, &params, &mut state, &hp);
        assert_eq!(sign_dir, Array1::from_vec(vec![1.0, -1.0, 0.0]));

        let mut state = PrimitiveState::new(3);
        let mom_dir = UpdatePrimitive::Momentum.apply(&grad, &params, &mut state, &hp);
        // First momentum step is (1-β)·grad.
        for (m, g) in mom_dir.iter().zip(grad.iter()) {
            assert!((m - 0.1 * g).abs() < 1e-12);
        }

        let mut state = PrimitiveState::new(3);
        let wd_dir = UpdatePrimitive::WeightDecay.apply(&grad, &params, &mut state, &hp);
        for (w, p) in wd_dir.iter().zip(params.iter()) {
            assert!((w - 0.5 * p).abs() < 1e-12);
        }

        // RmsProp and AdamLike must produce finite directions.
        let mut state = PrimitiveState::new(3);
        let rms_dir = UpdatePrimitive::RmsProp.apply(&grad, &params, &mut state, &hp);
        assert!(rms_dir.iter().all(|x| x.is_finite()));
        let mut state = PrimitiveState::new(3);
        let adam_dir = UpdatePrimitive::AdamLike.apply(&grad, &params, &mut state, &hp);
        assert!(adam_dir.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_config_validation_rejects_bad_inputs() {
        let bad_top_k = DartsConfig {
            top_k: 99,
            ..DartsConfig::default()
        };
        assert!(bad_top_k.validate().is_err(), "top_k > num primitives");

        let bad_epochs = DartsConfig {
            epochs: 0,
            ..DartsConfig::default()
        };
        assert!(bad_epochs.validate().is_err(), "zero epochs");

        let bad_lr = DartsConfig {
            learning_rate: -1.0,
            ..DartsConfig::default()
        };
        assert!(bad_lr.validate().is_err(), "negative lr");

        let empty_set = DartsConfig {
            primitives: Vec::new(),
            ..DartsConfig::default()
        };
        assert!(empty_set.validate().is_err(), "empty primitive set");

        let bad_beta = DartsConfig {
            adam_beta2: 1.0,
            ..DartsConfig::default()
        };
        assert!(bad_beta.validate().is_err(), "beta out of (0,1)");

        assert!(DartsConfig::default().validate().is_ok());
    }

    #[test]
    fn test_search_drives_validation_loss_down() {
        let objective = QuadraticBowl::<f64>::isotropic(6, 1.0).expect("bowl");
        let config = DartsConfig {
            epochs: 60,
            inner_steps: 5,
            learning_rate: 0.15,
            arch_learning_rate: 0.4,
            seed: 11,
            ..DartsConfig::default()
        };
        let mut search = DartsOptimizerSearch::<f64>::new(config).expect("search");
        let outcome = search.search(&objective).expect("search runs");

        assert_eq!(outcome.validation_trajectory.len(), 60);
        let initial = outcome.validation_trajectory[0];
        let final_loss = *outcome
            .validation_trajectory
            .last()
            .expect("non-empty trajectory");
        assert!(
            final_loss < initial,
            "validation loss should fall: initial={initial}, final={final_loss}"
        );
        assert!(final_loss.is_finite());
    }

    #[test]
    fn test_softmax_weights_in_outcome_valid() {
        let objective = QuadraticBowl::<f64>::isotropic(4, 1.0).expect("bowl");
        let config = DartsConfig {
            epochs: 20,
            seed: 5,
            ..DartsConfig::default()
        };
        let mut search = DartsOptimizerSearch::<f64>::new(config).expect("search");
        let outcome = search.search(&objective).expect("search runs");

        let sum: f64 = outcome.softmax_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(outcome.softmax_weights.iter().all(|&w| w > 0.0));
        assert_eq!(outcome.softmax_weights.len(), outcome.primitives.len());
    }

    #[test]
    fn test_grad_primitive_wins_on_well_conditioned_quadratic() {
        // On a well-conditioned (isotropic) but steep quadratic, the plain gradient is the
        // unambiguously best descent direction: its alignment with the validation gradient
        // scales with the curvature `c²`, whereas magnitude-normalizing primitives such as
        // sign/RMSProp scale only with `c`. The search should therefore up-weight Grad most.
        let objective = QuadraticBowl::<f64>::isotropic(5, 6.0).expect("bowl");
        let config = DartsConfig {
            primitives: UpdatePrimitive::default_set(),
            epochs: 120,
            inner_steps: 10,
            learning_rate: 0.06,
            arch_learning_rate: 0.6,
            init_params_scale: 2.0,
            seed: 7,
            ..DartsConfig::default()
        };
        let mut search = DartsOptimizerSearch::<f64>::new(config).expect("search");
        let outcome = search.search(&objective).expect("search runs");

        assert_eq!(
            outcome.best_primitive(),
            UpdatePrimitive::Grad,
            "Grad should win on a well-conditioned quadratic; weights = {:?}",
            outcome.softmax_weights
        );
        assert_eq!(
            outcome.discovered.selected_primitives()[0],
            UpdatePrimitive::Grad
        );
    }

    #[test]
    fn test_discovered_optimizer_converges_on_holdout() {
        // Search on one quadratic, then run the discovered optimizer on a held-out one.
        let train_objective = QuadraticBowl::<f64>::isotropic(5, 1.0).expect("bowl");
        let config = DartsConfig {
            epochs: 100,
            inner_steps: 12,
            learning_rate: 0.25,
            arch_learning_rate: 0.5,
            seed: 7,
            ..DartsConfig::default()
        };
        let mut search = DartsOptimizerSearch::<f64>::new(config).expect("search");
        let outcome = search.search(&train_objective).expect("search runs");

        let mut optimizer = outcome.discovered;
        // Held-out quadratic with different curvature and a non-zero starting point.
        let curvature = Array1::from_vec(vec![1.0, 0.8, 1.2, 0.9, 1.1]);
        let holdout = QuadraticBowl::<f64>::new(curvature);
        let mut params = Array1::from_vec(vec![3.0, -2.5, 4.0, -3.5, 2.0]);
        let initial_norm = l2(&params);

        for _ in 0..200 {
            let (_, grad) = holdout.train_loss_grad(&params).expect("grad");
            params = optimizer.step(&params, &grad).expect("step");
        }
        let final_norm = l2(&params);
        assert!(
            final_norm < initial_norm,
            "discovered optimizer should converge: initial={initial_norm}, final={final_norm}"
        );
        assert!(
            final_norm < 0.5 * initial_norm,
            "should make substantial progress"
        );
        assert_eq!(optimizer.step_count(), 200);
    }

    #[test]
    fn test_deterministic_given_seed() {
        let objective = QuadraticBowl::<f64>::isotropic(5, 1.0).expect("bowl");
        let config = DartsConfig {
            epochs: 30,
            inner_steps: 4,
            seed: 4242,
            ..DartsConfig::default()
        };

        let mut search_a =
            DartsOptimizerSearch::<f64>::new(config.clone()).expect("first search constructs");
        let outcome_a = search_a.search(&objective).expect("a runs");
        let mut search_b =
            DartsOptimizerSearch::<f64>::new(config).expect("second search constructs");
        let outcome_b = search_b.search(&objective).expect("b runs");

        for (a, b) in outcome_a.alpha.iter().zip(outcome_b.alpha.iter()) {
            assert!((a - b).abs() < 1e-15, "alpha must be identical: {a} vs {b}");
        }
        for (a, b) in outcome_a
            .validation_trajectory
            .iter()
            .zip(outcome_b.validation_trajectory.iter())
        {
            assert!((a - b).abs() < 1e-15, "trajectory must be identical");
        }
    }

    #[test]
    fn test_analytic_alpha_gradient_matches_finite_difference() {
        let objective = QuadraticBowl::<f64>::isotropic(4, 1.3).expect("bowl");
        let config = DartsConfig {
            learning_rate: 0.2,
            seed: 1,
            ..DartsConfig::default()
        };
        let search = DartsOptimizerSearch::<f64>::new(config).expect("search");

        // Fix a parameter point, an arbitrary alpha, and a fixed set of primitive
        // directions; then compare the analytic gradient of φ(α) against central FD.
        let params = Array1::from_vec(vec![0.7, -1.1, 0.3, 0.9]);
        let alpha = Array1::from_vec(vec![0.2, -0.5, 0.8, 0.1, -0.3, 0.4]);
        let (_, grad_train) = objective.train_loss_grad(&params).expect("grad");
        let mut states = search.fresh_states(4);
        let ops: Vec<Array1<f64>> = search
            .primitives
            .iter()
            .zip(states.iter_mut())
            .map(|(p, s)| p.apply(&grad_train, &params, s, &search.hyperparams))
            .collect();

        let analytic = search
            .analytic_architecture_gradient(&alpha, &ops, &params, &objective)
            .expect("analytic");

        let eps = 1e-6;
        for j in 0..alpha.len() {
            let mut plus = alpha.clone();
            let mut minus = alpha.clone();
            plus[j] += eps;
            minus[j] -= eps;
            let loss_plus = search
                .lookahead_validation_loss(&plus, &ops, &params, &objective)
                .expect("loss+");
            let loss_minus = search
                .lookahead_validation_loss(&minus, &ops, &params, &objective)
                .expect("loss-");
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (analytic[j] - numeric).abs() < 1e-5,
                "analytic vs FD mismatch at {j}: {} vs {numeric}",
                analytic[j]
            );
        }
    }

    #[test]
    fn test_discovered_optimizer_trait_surface() {
        let objective = QuadraticBowl::<f64>::isotropic(3, 1.0).expect("bowl");
        let mut search = DartsOptimizerSearch::<f64>::new(DartsConfig::default()).expect("search");
        let outcome = search.search(&objective).expect("search runs");
        let mut optimizer = outcome.discovered;

        assert_eq!(optimizer.name(), "DartsDiscovered");
        let lr0 = optimizer.get_learning_rate();
        assert!(lr0 > 0.0);
        optimizer.set_learning_rate(0.05);
        assert!((optimizer.get_learning_rate() - 0.05).abs() < 1e-12);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grad = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let _ = optimizer.step(&params, &grad).expect("step");
        let state = optimizer.get_state();
        assert_eq!(state.step_count, 1);
        assert!((state.current_lr - 0.05).abs() < 1e-12);
        assert!(state.grad_norm_ema > 0.0);

        // Selected weights sum to one.
        let weight_sum: f64 = optimizer.selected_weights().iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_top_k_discretization_keeps_two() {
        let objective = QuadraticBowl::<f64>::isotropic(4, 1.0).expect("bowl");
        let config = DartsConfig {
            top_k: 2,
            epochs: 25,
            seed: 99,
            ..DartsConfig::default()
        };
        let mut search = DartsOptimizerSearch::<f64>::new(config).expect("search");
        let outcome = search.search(&objective).expect("search runs");
        assert_eq!(outcome.discovered.selected_primitives().len(), 2);
        let weight_sum: f64 = outcome.discovered.selected_weights().iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_closure_objective_search_runs() {
        // A simple separable quadratic supplied as a closure.
        let objective = ClosureObjective::<f64>::from_single(3, |p| {
            let loss = 0.5 * p.iter().fold(0.0, |acc, &x| acc + x * x);
            let grad = p.clone();
            Ok((loss, grad))
        })
        .expect("closure objective");
        let config = DartsConfig {
            epochs: 30,
            inner_steps: 6,
            learning_rate: 0.2,
            seed: 3,
            ..DartsConfig::default()
        };
        let mut search = DartsOptimizerSearch::<f64>::new(config).expect("search");
        let outcome = search.search(&objective).expect("search runs");
        let initial = outcome.validation_trajectory[0];
        let final_loss = *outcome.validation_trajectory.last().expect("trajectory");
        assert!(final_loss < initial);
    }

    #[test]
    fn test_rosenbrock_objective_evaluates() {
        let objective = Rosenbrock::<f64>::new(3).expect("rosenbrock");
        // Gradient at the global minimum (1,1,1) must vanish.
        let minimum = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let (loss, grad) = objective.train_loss_grad(&minimum).expect("grad");
        assert!(loss.abs() < 1e-12);
        assert!(grad.iter().all(|&g| g.abs() < 1e-9));

        // A short search should run end to end on Rosenbrock without error.
        let config = DartsConfig {
            epochs: 15,
            inner_steps: 4,
            learning_rate: 0.001,
            init_params_scale: 0.5,
            seed: 2,
            ..DartsConfig::default()
        };
        let mut search = DartsOptimizerSearch::<f64>::new(config).expect("search");
        let outcome = search.search(&objective).expect("search runs");
        assert!(outcome.validation_trajectory.iter().all(|x| x.is_finite()));
    }
}
