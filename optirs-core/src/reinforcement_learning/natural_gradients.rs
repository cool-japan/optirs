// Natural Policy Gradients
//
// This module implements natural policy gradient methods that use the Fisher information
// matrix to precondition policy gradients for more efficient optimization.

use super::{
    parameter_keys, unflatten_named, PolicyNetwork, RLOptimizationMetrics, RLOptimizerConfig,
    TrajectoryBatch,
};
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

    /// Block-diagonal Fisher blocks, one per named parameter.
    fisher_blocks: Option<Vec<FisherBlock<T>>>,

    /// Empirical Fisher accumulator
    empirical_fisher_accumulator: FisherAccumulator<T>,

    /// Natural gradient state
    natural_grad_state: NaturalGradientState<T>,

    /// Update counter
    update_count: usize,

    /// Parameter dimension
    paramdim: usize,
}

/// Kronecker factorization components.
///
/// Block `l` approximates its Fisher sub-matrix as `G_l ⊗ A_l`, where
/// `A_l = E[φφᵀ]` is the input/activation covariance and `G_l = E[δδᵀ]` the
/// pre-activation gradient covariance. `offsets[l]` is where the block starts in
/// the flat parameter layout and `parameter_names[l]` names it.
#[derive(Debug, Clone)]
pub struct KroneckerFactors<T: Float + Debug + Send + Sync + 'static> {
    /// Input statistics (activation covariances) `A_l`
    pub input_factors: Vec<Array2<T>>,

    /// Output statistics (gradient covariances) `G_l`
    pub output_factors: Vec<Array2<T>>,

    /// Name of the parameter each factor pair belongs to
    pub parameter_names: Vec<String>,

    /// Flat-layout offset of each block
    pub offsets: Vec<usize>,
}

/// One block of a block-diagonal Fisher estimate.
#[derive(Debug, Clone)]
pub struct FisherBlock<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter this block corresponds to.
    pub name: String,

    /// Offset of the block in the flat parameter layout.
    pub offset: usize,

    /// Dense `len × len` (damped) Fisher block.
    pub matrix: Array2<T>,
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
            fisher_blocks: None,
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

    /// Update Fisher Information Matrix.
    ///
    /// Every estimator below consumes **real** per-sample score vectors. There is
    /// no silent fall-through: an estimator that cannot be built (because the
    /// policy provides no differentiable path) returns an error rather than
    /// leaving the Fisher empty, which used to degrade natural gradients to
    /// vanilla SGD without any indication.
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
            FisherEstimationMethod::GaussNewton => {
                // For a log-likelihood objective the Gauss-Newton matrix and the
                // (empirical) Fisher coincide, so this is an exact identity, not a
                // fallback.
                self.update_empirical_fisher(trajectory)?
            }
            FisherEstimationMethod::BFGS => {
                return Err(OptimError::UnsupportedOperation(
                    "BFGS Fisher approximation is not implemented; use Empirical, Diagonal, \
                     BlockDiagonal or KroneckerFactored"
                        .to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Per-sample score vectors `g_i = ∇_θ log π(aᵢ|sᵢ)`, one per row.
    ///
    /// Resolution order:
    /// 1. the policy's analytic score oracle ([`PolicyNetwork::score_matrix`]);
    /// 2. central finite differences, for policies without an oracle and at most
    ///    [`Self::FD_MAX_DIMS`] parameters;
    /// 3. an error.
    ///
    /// Step 3 replaces the previous behaviour of returning a matrix of **zeros**
    /// for large policies, which made every Fisher estimate the zero matrix and
    /// (after damping) amplified the raw gradient by `1/damping`.
    fn per_sample_scores(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<Array2<T>> {
        let batch_size = trajectory.observations.nrows();

        match self
            .policy
            .score_matrix(&trajectory.observations, &trajectory.actions)
        {
            Ok(scores) => {
                if scores.ncols() != self.paramdim {
                    return Err(OptimError::DimensionMismatch(format!(
                        "policy score matrix has {} columns but paramdim is {}",
                        scores.ncols(),
                        self.paramdim
                    )));
                }
                return Ok(scores);
            }
            Err(OptimError::UnsupportedOperation(_)) => {}
            Err(other) => return Err(other),
        }

        if self.paramdim == 0 {
            return Err(OptimError::InvalidConfig(
                "natural gradients require a non-zero parameter dimension".to_string(),
            ));
        }
        if self.paramdim > Self::FD_MAX_DIMS {
            return Err(OptimError::UnsupportedOperation(format!(
                "policy has {} parameters (> {} finite-difference limit) and provides no analytic \
                 score function: implement PolicyNetwork::log_prob_gradient / score_matrix",
                self.paramdim,
                Self::FD_MAX_DIMS
            )));
        }

        let mut scores = Array2::zeros((batch_size, self.paramdim));
        for i in 0..batch_size {
            let obs = trajectory.observations.row(i).to_owned();
            let action = trajectory.actions.row(i).to_owned();
            let score = self.finite_difference_score(&obs, &action)?;
            for j in 0..self.paramdim {
                scores[[i, j]] = score[j];
            }
        }
        Ok(scores)
    }

    /// Flat-layout `(name, offset, len)` triples of the policy's parameters.
    fn parameter_layout(&self) -> Vec<(String, usize, usize)> {
        let params = self.policy.get_parameters();
        let mut layout = Vec::with_capacity(params.len());
        let mut offset = 0usize;
        for key in parameter_keys(&params) {
            let len = params[&key].len();
            layout.push((key, offset, len));
            offset += len;
        }
        layout
    }

    /// Update empirical Fisher Information Matrix
    fn update_empirical_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        let scores = self.per_sample_scores(trajectory)?;

        for i in 0..scores.nrows() {
            let row = scores.row(i).to_owned();
            self.add_to_empirical_fisher(&row)?;
        }

        self.finalize_empirical_fisher()?;

        Ok(())
    }

    /// Update true Fisher Information Matrix.
    ///
    /// For an exponential-family policy the Fisher equals the expected outer
    /// product of scores, `E[∇log π ∇log πᵀ]` — the same quantity the empirical
    /// estimator accumulates, evaluated on the sampled actions.
    fn update_true_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        self.update_empirical_fisher(trajectory)
    }

    /// Update diagonal Fisher approximation
    fn update_diagonal_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        let scores = self.per_sample_scores(trajectory)?;
        let batch_size = scores.nrows();
        if batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "cannot estimate a Fisher matrix from an empty trajectory".to_string(),
            ));
        }
        let count = T::from(batch_size).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;

        let mut diagonal = Array1::zeros(self.paramdim);
        for i in 0..batch_size {
            for j in 0..self.paramdim {
                let g = scores[[i, j]];
                diagonal[j] += g * g;
            }
        }
        for j in 0..self.paramdim {
            diagonal[j] = diagonal[j] / count + self._config.damping;
        }

        self.fisher_diagonal = Some(diagonal);

        Ok(())
    }

    /// Update block diagonal Fisher approximation.
    ///
    /// One dense block per **named parameter**: parameters within a block are
    /// treated as correlated, parameters across blocks as independent. Each block
    /// is the empirical covariance of that block's score components,
    /// `F_b = (1/N) Σ_i g_{i,b} g_{i,b}ᵀ + λI`.
    fn update_block_diagonal_fisher(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        let scores = self.per_sample_scores(trajectory)?;
        let batch_size = scores.nrows();
        if batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "cannot estimate a Fisher matrix from an empty trajectory".to_string(),
            ));
        }
        let count = T::from(batch_size).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;

        let mut blocks = Vec::new();
        for (name, offset, len) in self.parameter_layout() {
            let mut matrix = Array2::zeros((len, len));
            for i in 0..batch_size {
                for a in 0..len {
                    let ga = scores[[i, offset + a]];
                    if ga == T::zero() {
                        continue;
                    }
                    for b in 0..len {
                        matrix[[a, b]] += ga * scores[[i, offset + b]];
                    }
                }
            }
            for a in 0..len {
                for b in 0..len {
                    matrix[[a, b]] = matrix[[a, b]] / count;
                }
                matrix[[a, a]] += self._config.damping;
            }
            blocks.push(FisherBlock {
                name,
                offset,
                matrix,
            });
        }

        self.fisher_blocks = Some(blocks);

        Ok(())
    }

    /// Update Kronecker factorization (K-FAC).
    ///
    /// The policy supplies per-sample layer inputs `φ_i` and pre-activation
    /// gradients `δ_i`; the factors are their second moments
    /// `A = (1/N) ΦᵀΦ` and `G = (1/N) ΔᵀΔ`, giving `F_block ≈ G ⊗ A`.
    fn update_kronecker_factors(&mut self, trajectory: &TrajectoryBatch<T>) -> Result<()> {
        let blocks = self
            .policy
            .kronecker_factors(&trajectory.observations, &trajectory.actions)?;

        if blocks.is_empty() {
            return Err(OptimError::InvalidState(
                "policy returned no Kronecker blocks".to_string(),
            ));
        }

        let layout = self.parameter_layout();
        let mut input_factors = Vec::with_capacity(blocks.len());
        let mut output_factors = Vec::with_capacity(blocks.len());
        let mut parameter_names = Vec::with_capacity(blocks.len());
        let mut offsets = Vec::with_capacity(blocks.len());

        for block in blocks {
            let n_samples = block.inputs.nrows();
            if n_samples == 0 || block.outputs.nrows() != n_samples {
                return Err(OptimError::DimensionMismatch(format!(
                    "Kronecker block '{}' has mismatched or empty factors",
                    block.name
                )));
            }
            let count = T::from(n_samples).ok_or_else(|| {
                OptimError::ComputationError("failed to convert sample count".to_string())
            })?;

            let (name, offset, len) = layout
                .iter()
                .find(|(name, _, _)| name == &block.name)
                .cloned()
                .ok_or_else(|| {
                    OptimError::InvalidState(format!(
                        "Kronecker block '{}' does not name a policy parameter",
                        block.name
                    ))
                })?;

            let n_in = block.inputs.ncols();
            let n_out = block.outputs.ncols();
            if n_in * n_out != len {
                return Err(OptimError::DimensionMismatch(format!(
                    "Kronecker block '{name}' factors imply {} parameters but '{name}' has {len}",
                    n_in * n_out
                )));
            }

            let mut a_factor = Array2::zeros((n_in, n_in));
            for i in 0..n_samples {
                for r in 0..n_in {
                    let value = block.inputs[[i, r]];
                    if value == T::zero() {
                        continue;
                    }
                    for c in 0..n_in {
                        a_factor[[r, c]] += value * block.inputs[[i, c]];
                    }
                }
            }
            let mut g_factor = Array2::zeros((n_out, n_out));
            for i in 0..n_samples {
                for r in 0..n_out {
                    let value = block.outputs[[i, r]];
                    if value == T::zero() {
                        continue;
                    }
                    for c in 0..n_out {
                        g_factor[[r, c]] += value * block.outputs[[i, c]];
                    }
                }
            }
            a_factor.mapv_inplace(|x| x / count);
            g_factor.mapv_inplace(|x| x / count);

            input_factors.push(a_factor);
            output_factors.push(g_factor);
            parameter_names.push(name);
            offsets.push(offset);
        }

        self.kronecker_factors = Some(KroneckerFactors {
            input_factors,
            output_factors,
            parameter_names,
            offsets,
        });

        Ok(())
    }

    /// Compute natural gradients `F⁻¹ g` for the configured estimator.
    fn compute_natural_gradients(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        if !self._config.enable_preconditioning {
            return Ok(gradients.clone());
        }

        let natural_grad = match self._config.fisher_method {
            FisherEstimationMethod::Diagonal => {
                let diagonal = self.fisher_diagonal.as_ref().ok_or_else(|| {
                    OptimError::InvalidState(
                        "diagonal Fisher has not been estimated yet".to_string(),
                    )
                })?;
                if diagonal.len() != gradients.len() {
                    return Err(OptimError::DimensionMismatch(
                        "diagonal Fisher length does not match the gradient".to_string(),
                    ));
                }
                let mut out = Array1::zeros(gradients.len());
                for i in 0..gradients.len() {
                    let d = diagonal[i];
                    out[i] = if d.abs() > T::epsilon() {
                        gradients[i] / d
                    } else {
                        gradients[i]
                    };
                }
                out
            }
            FisherEstimationMethod::BlockDiagonal => {
                let blocks = self.fisher_blocks.as_ref().ok_or_else(|| {
                    OptimError::InvalidState(
                        "block-diagonal Fisher has not been estimated yet".to_string(),
                    )
                })?;
                let mut out = Array1::zeros(gradients.len());
                for block in blocks {
                    let len = block.matrix.nrows();
                    let mut rhs = Array2::zeros((len, 1));
                    for i in 0..len {
                        rhs[[i, 0]] = gradients[block.offset + i];
                    }
                    let solution = gaussian_solve(&block.matrix, &rhs)?;
                    for i in 0..len {
                        out[block.offset + i] = solution[[i, 0]];
                    }
                }
                out
            }
            FisherEstimationMethod::KroneckerFactored => {
                let factors = self.kronecker_factors.as_ref().ok_or_else(|| {
                    OptimError::InvalidState(
                        "Kronecker factors have not been estimated yet".to_string(),
                    )
                })?;
                self.solve_kronecker_system(factors, gradients)?
            }
            _ => {
                let fisher = self.fisher_matrix.as_ref().ok_or_else(|| {
                    OptimError::InvalidState(
                        "empirical Fisher has not been estimated yet".to_string(),
                    )
                })?;
                self.solve_fisher_system(fisher, gradients)?
            }
        };

        // Apply scaling
        let scaled_natural_grad = natural_grad * self._config.natural_grad_scale;

        Ok(scaled_natural_grad)
    }

    /// Solve `(G ⊗ A + λI) x = g` block-wise using the K-FAC factored form.
    ///
    /// With **row-major** vectorization `vecr(M)` of the `(n_out, n_in)` parameter
    /// matrix, `(G ⊗ A) vecr(M) = vecr(G M Aᵀ)`, so the solve is
    /// `X = G_λ⁻¹ M A_λ⁻¹` with the standard factored damping
    /// `A_λ = A + √λ I`, `G_λ = G + √λ I` (whose Kronecker product has the same
    /// √λ·√λ = λ diagonal contribution as adding `λI` to the full matrix).
    fn solve_kronecker_system(
        &self,
        factors: &KroneckerFactors<T>,
        gradients: &Array1<T>,
    ) -> Result<Array1<T>> {
        let sqrt_damping = self._config.damping.max(T::zero()).sqrt();
        let mut out = gradients.clone();

        for index in 0..factors.input_factors.len() {
            let a_factor = &factors.input_factors[index];
            let g_factor = &factors.output_factors[index];
            let offset = factors.offsets[index];
            let n_in = a_factor.nrows();
            let n_out = g_factor.nrows();

            if offset + n_in * n_out > gradients.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "Kronecker block '{}' exceeds the gradient length",
                    factors.parameter_names[index]
                )));
            }

            // Reshape the block's gradient slice to (n_out, n_in), row-major.
            let mut m = Array2::zeros((n_out, n_in));
            for r in 0..n_out {
                for c in 0..n_in {
                    m[[r, c]] = gradients[offset + r * n_in + c];
                }
            }

            let mut a_damped = a_factor.clone();
            for i in 0..n_in {
                a_damped[[i, i]] += sqrt_damping;
            }
            let mut g_damped = g_factor.clone();
            for i in 0..n_out {
                g_damped[[i, i]] += sqrt_damping;
            }

            // Y = G_λ⁻¹ M
            let y = gaussian_solve(&g_damped, &m)?;
            // X = Y A_λ⁻¹  ⇔  A_λ Xᵀ = Yᵀ (A_λ symmetric)
            let mut y_t = Array2::zeros((n_in, n_out));
            for r in 0..n_out {
                for c in 0..n_in {
                    y_t[[c, r]] = y[[r, c]];
                }
            }
            let x_t = gaussian_solve(&a_damped, &y_t)?;

            for r in 0..n_out {
                for c in 0..n_in {
                    out[offset + r * n_in + c] = x_t[[c, r]];
                }
            }
        }

        Ok(out)
    }

    /// Solve Fisher information system using conjugate gradient.
    ///
    /// Guards every division of the recurrence: a zero right-hand side returns
    /// `x = 0` immediately, a vanishing or non-finite curvature `pᵀAp` breaks with
    /// the current iterate, and `β` is only formed while `rsold > 0`.
    fn solve_fisher_system(&self, fisher: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>> {
        let n = rhs.len();
        if fisher.nrows() != n || fisher.ncols() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "Fisher matrix is {}x{} but the gradient has length {n}",
                fisher.nrows(),
                fisher.ncols()
            )));
        }

        let tiny = T::from(1e-30).unwrap_or_else(T::epsilon);
        let mut x = Array1::zeros(n);
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rsold = self.dot(&r, &r);

        if !matches!(rsold.partial_cmp(&tiny), Some(std::cmp::Ordering::Greater)) {
            return Ok(x);
        }

        for _i in 0..self._config.cg_iters {
            let ap = fisher.dot(&p);
            let pap = self.dot(&p, &ap);
            if !matches!(
                pap.abs().partial_cmp(&tiny),
                Some(std::cmp::Ordering::Greater)
            ) || !pap.is_finite()
            {
                break;
            }

            let alpha = rsold / pap;

            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);

            let rsnew = self.dot(&r, &r);

            if rsnew.sqrt() < self._config.cg_tolerance {
                break;
            }
            if !matches!(rsnew.partial_cmp(&tiny), Some(std::cmp::Ordering::Greater)) {
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
        let deltas = unflatten_named(&params, update)?;
        self.policy.update_parameters(&deltas)
    }

    /// Maximum parameter count for finite-difference score estimation.
    const FD_MAX_DIMS: usize = 500;

    /// Score function `∇_θ log π(a|s)` for a single sample.
    ///
    /// Prefers the policy's analytic oracle and falls back to central finite
    /// differences for small policies. Returns an error when neither is available
    /// — never a vector of zeros, which would silently zero the Fisher.
    ///
    /// This is a public single-sample convenience wrapper; the batched Fisher
    /// estimation path (`Self::per_sample_scores`) computes the same
    /// quantity for a whole trajectory more efficiently (one analytic
    /// `score_matrix` call for the batch, rather than one per sample) and does
    /// not call through this method.
    pub fn compute_log_prob_gradients(
        &mut self,
        obs: &Array1<T>,
        action: &Array1<T>,
    ) -> Result<Array1<T>> {
        let mut obs_2d = Array2::zeros((1, obs.len()));
        obs_2d.row_mut(0).assign(obs);
        let mut act_2d = Array2::zeros((1, action.len()));
        act_2d.row_mut(0).assign(action);

        match self.policy.score_matrix(&obs_2d, &act_2d) {
            Ok(scores) => {
                let mut out = Array1::zeros(scores.ncols());
                for j in 0..scores.ncols() {
                    out[j] = scores[[0, j]];
                }
                return Ok(out);
            }
            Err(OptimError::UnsupportedOperation(_)) => {}
            Err(other) => return Err(other),
        }

        self.finite_difference_score(obs, action)
    }

    /// Compute log probability gradients via central finite differences.
    ///
    /// For each scalar parameter θ_i, the score function component is:
    ///   s_i = [log π(a|s; θ+ε·eᵢ) − log π(a|s; θ−ε·eᵢ)] / (2ε)
    ///
    /// Uses three additive calls to `update_parameters` per dimension (+ε,
    /// −2ε, +ε) so the policy is exactly restored to its original state.
    /// Refuses (with an error) to run above `FD_MAX_DIMS` parameters instead of
    /// quietly returning zeros.
    fn finite_difference_score(
        &mut self,
        obs: &Array1<T>,
        action: &Array1<T>,
    ) -> Result<Array1<T>> {
        if self.paramdim == 0 {
            return Err(OptimError::InvalidConfig(
                "finite-difference score requires a non-zero parameter dimension".to_string(),
            ));
        }
        if self.paramdim > Self::FD_MAX_DIMS {
            return Err(OptimError::UnsupportedOperation(format!(
                "finite-difference score is limited to {} parameters (policy has {}); implement \
                 PolicyNetwork::log_prob_gradient for an analytic score",
                Self::FD_MAX_DIMS,
                self.paramdim
            )));
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
            return Err(OptimError::DimensionMismatch(format!(
                "policy exposes {total} parameters but the optimizer was built for {}",
                self.paramdim
            )));
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

/// Solve the dense linear system `A X = B` by Gaussian elimination with partial
/// pivoting.
///
/// Used for the small per-parameter blocks of the block-diagonal Fisher and for
/// the Kronecker factors, both of which are far too small to justify an iterative
/// solver. Returns [`OptimError::ComputationError`] when the matrix is singular,
/// rather than emitting infinities.
fn gaussian_solve<T: Float + Debug + Send + Sync + 'static>(
    a: &Array2<T>,
    b: &Array2<T>,
) -> Result<Array2<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(OptimError::DimensionMismatch(
            "gaussian_solve requires a square matrix".to_string(),
        ));
    }
    if b.nrows() != n {
        return Err(OptimError::DimensionMismatch(format!(
            "right-hand side has {} rows but the matrix is {n}x{n}",
            b.nrows()
        )));
    }

    let m = b.ncols();
    let mut aug = a.clone();
    let mut rhs = b.clone();

    for column in 0..n {
        // Partial pivoting.
        let mut pivot_row = column;
        let mut pivot_value = aug[[column, column]].abs();
        for row in (column + 1)..n {
            let candidate = aug[[row, column]].abs();
            if candidate > pivot_value {
                pivot_value = candidate;
                pivot_row = row;
            }
        }

        if !matches!(
            pivot_value.partial_cmp(&T::epsilon()),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(OptimError::ComputationError(
                "singular matrix in Fisher system solve (increase the damping)".to_string(),
            ));
        }

        if pivot_row != column {
            for c in 0..n {
                let tmp = aug[[column, c]];
                aug[[column, c]] = aug[[pivot_row, c]];
                aug[[pivot_row, c]] = tmp;
            }
            for c in 0..m {
                let tmp = rhs[[column, c]];
                rhs[[column, c]] = rhs[[pivot_row, c]];
                rhs[[pivot_row, c]] = tmp;
            }
        }

        let pivot = aug[[column, column]];
        for row in (column + 1)..n {
            let factor = aug[[row, column]] / pivot;
            if factor == T::zero() {
                continue;
            }
            for c in column..n {
                aug[[row, c]] = aug[[row, c]] - factor * aug[[column, c]];
            }
            for c in 0..m {
                rhs[[row, c]] = rhs[[row, c]] - factor * rhs[[column, c]];
            }
        }
    }

    // Back substitution.
    let mut x = Array2::zeros((n, m));
    for c in 0..m {
        for row in (0..n).rev() {
            let mut acc = rhs[[row, c]];
            for col in (row + 1)..n {
                acc = acc - aug[[row, col]] * x[[col, c]];
            }
            x[[row, c]] = acc / aug[[row, row]];
        }
    }

    Ok(x)
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
