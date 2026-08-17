// Trust Region Methods for Policy Optimization
//
// This module implements trust region methods including TRPO (Trust Region Policy Optimization)
// and other constrained optimization techniques for policy learning.

#[allow(dead_code)]
use super::{unflatten_named, PolicyNetwork, RLOptimizationMetrics};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Smallest magnitude treated as non-zero by the iterative solvers.
///
/// Anything at or below this is treated as an exact zero so the conjugate
/// gradient never forms `0/0` (which would silently poison the whole step with
/// NaNs). Comparisons are written as `!(x > tiny())` so NaN inputs also stop.
fn tiny<T: Float>() -> T {
    T::from(1e-30).unwrap_or_else(T::epsilon)
}

/// Trust region methods
#[derive(Debug, Clone, Copy)]
pub enum TrustRegionMethod {
    /// Trust Region Policy Optimization (TRPO)
    TRPO,

    /// Constrained Policy Optimization (CPO)
    CPO,

    /// Projection-based trust region
    Projection,

    /// Natural gradient with trust region
    NaturalGradient,
}

/// Trust region configuration
#[derive(Debug, Clone)]
pub struct TrustRegionConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Trust region method
    pub method: TrustRegionMethod,

    /// Maximum KL divergence
    pub max_kl: T,

    /// Conjugate gradient parameters
    pub cg_iters: usize,
    pub cg_damping: T,
    pub cg_tolerance: T,

    /// Line search parameters
    pub max_backtracks: usize,
    pub backtrack_coeff: T,
    pub accept_ratio: T,

    /// Natural gradient Fisher information matrix estimation
    pub fisher_subsample_freq: usize,
    pub fisher_reg: T,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for TrustRegionConfig<T> {
    fn default() -> Self {
        Self {
            method: TrustRegionMethod::TRPO,
            max_kl: T::from(0.01).unwrap_or_else(|| T::zero()),
            cg_iters: 10,
            cg_damping: T::from(0.1).unwrap_or_else(|| T::zero()),
            cg_tolerance: T::from(1e-8).unwrap_or_else(|| T::zero()),
            max_backtracks: 10,
            backtrack_coeff: T::from(0.5).unwrap_or_else(|| T::zero()),
            accept_ratio: T::from(0.1).unwrap_or_else(|| T::zero()),
            fisher_subsample_freq: 1,
            fisher_reg: T::from(1e-5).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Trust region optimizer
pub struct TrustRegionOptimizer<T: Float + Debug + Send + Sync + 'static, P: PolicyNetwork<T>> {
    /// Configuration
    config: TrustRegionConfig<T>,

    /// Policy network
    policy: P,

    /// Fisher information matrix
    fisher_matrix: Option<Array2<T>>,

    /// Per-sample score vectors used for the empirical Fisher Information Matrix.
    ///
    /// Each ROW is a per-sample score vector `g_i = ∇_θ log π(a_i | s_i)` and the
    /// number of columns equals the policy parameter dimension `d`. When present,
    /// the empirical Fisher estimate `F̂ = (1/N) Σ_i g_i g_iᵀ` is used to compute
    /// Fisher-vector products. When `None`, the optimizer falls back to an identity
    /// Fisher (see [`TrustRegionOptimizer::fisher_vector_product`]).
    score_samples: Option<Array2<T>>,

    /// Linearized safety constraint used by [`TrustRegionMethod::CPO`].
    ///
    /// `Some((b, c))` where `b = ∇_θ J_C(π)` is the cost surrogate gradient and
    /// `c = J_C(π) − d` is the current constraint surplus (positive ⇒ violated).
    /// The CPO step enforces `c + bᵀx ≤ 0` alongside the KL trust region.
    cost_constraint: Option<(Array1<T>, T)>,

    /// Natural gradient state
    natural_grad_state: NaturalGradientState<T>,

    /// Update counter
    update_count: usize,
}

/// Outcome of a single trust-region step.
#[derive(Debug, Clone)]
pub struct TrustRegionStepReport<T: Float + Debug + Send + Sync + 'static> {
    /// Whether a step was accepted (a rejected line search applies **no** update).
    pub accepted: bool,

    /// Backtracking coefficient of the accepted step (`1` = full step).
    pub step_scale: T,

    /// Quadratic-model KL of the applied step (`0` when nothing was applied).
    pub kl: T,

    /// Measured surrogate improvement of the applied step.
    pub surrogate_improvement: T,

    /// Number of backtracking iterations performed.
    pub backtracks: usize,
}

/// Natural gradient computation state
#[derive(Debug, Clone)]
pub struct NaturalGradientState<T: Float + Debug + Send + Sync + 'static> {
    /// Previous gradients for momentum
    pub prev_gradients: Option<Array1<T>>,

    /// Momentum coefficient
    pub momentum: T,

    /// Adaptive learning rate state
    pub adaptive_lr_state: AdaptiveLRState<T>,
}

/// Adaptive learning rate state
#[derive(Debug, Clone)]
pub struct AdaptiveLRState<T: Float + Debug + Send + Sync + 'static> {
    /// Current learning rate
    pub learning_rate: T,

    /// Learning rate adaptation factor
    pub adapt_factor: T,

    /// Success counter for adaptation
    pub success_count: usize,

    /// Failure counter for adaptation
    pub failure_count: usize,
}

impl<
        T: Float + Debug + Send + Sync + std::iter::Sum + ScalarOperand + 'static,
        P: PolicyNetwork<T>,
    > TrustRegionOptimizer<T, P>
{
    /// Create a new trust region optimizer
    pub fn new(config: TrustRegionConfig<T>, policy: P) -> Self {
        Self {
            config,
            policy,
            fisher_matrix: None,
            score_samples: None,
            cost_constraint: None,
            natural_grad_state: NaturalGradientState {
                prev_gradients: None,
                momentum: T::from(0.9).unwrap_or_else(|| T::zero()),
                adaptive_lr_state: AdaptiveLRState {
                    learning_rate: T::from(0.01).unwrap_or_else(|| T::zero()),
                    adapt_factor: T::from(1.5).unwrap_or_else(|| T::zero()),
                    success_count: 0,
                    failure_count: 0,
                },
            },
            update_count: 0,
        }
    }

    /// Feed per-sample score vectors for the empirical Fisher Information Matrix.
    ///
    /// Each row of `samples` is a per-sample score vector
    /// `g_i = ∇_θ log π(a_i | s_i)` whose length must equal the policy parameter
    /// dimension. These are consumed by [`Self::fisher_vector_product`] to form the
    /// empirical estimate `F̂ = (1/N) Σ_i g_i g_iᵀ` without ever materializing the
    /// dense `d × d` matrix.
    pub fn set_score_samples(&mut self, samples: Array2<T>) {
        self.score_samples = Some(samples);
    }

    /// Clear any stored score samples, reverting the Fisher-vector product to the
    /// identity-Fisher fallback.
    pub fn clear_score_samples(&mut self) {
        self.score_samples = None;
    }

    /// Install the linearized safety constraint required by CPO.
    ///
    /// * `cost_gradient` — `b = ∇_θ J_C(π)`, the ascent direction of the expected
    ///   cost surrogate (same flat layout as the objective gradient).
    /// * `cost_surplus` — `c = J_C(π) − d`, i.e. how far the current policy is
    ///   *above* the cost limit. Positive means the constraint is violated.
    ///
    /// The CPO step then solves
    /// `max gᵀx s.t. c + bᵀx ≤ 0, ½ xᵀFx ≤ δ`.
    pub fn set_cost_constraint(&mut self, cost_gradient: Array1<T>, cost_surplus: T) {
        self.cost_constraint = Some((cost_gradient, cost_surplus));
    }

    /// Remove the CPO safety constraint.
    pub fn clear_cost_constraint(&mut self) {
        self.cost_constraint = None;
    }

    /// Perform trust region update.
    ///
    /// `gradients` is the **ascent** direction of the surrogate objective (e.g.
    /// `∇_θ E[log π(a|s)·A(s,a)]`); the step taken moves *along* it, subject to
    /// the KL trust region.
    pub fn update(&mut self, gradients: &Array1<T>) -> Result<RLOptimizationMetrics<T>> {
        match self.config.method {
            TrustRegionMethod::TRPO => self.update_trpo(gradients),
            TrustRegionMethod::CPO => self.update_cpo(gradients),
            TrustRegionMethod::Projection => self.update_projection(gradients),
            TrustRegionMethod::NaturalGradient => self.update_natural_gradient(gradients),
        }
    }

    /// TRPO update judged against the *quadratic model* of the surrogate.
    ///
    /// Without trajectory data the optimizer cannot re-evaluate the true
    /// surrogate, so the line search scores candidates with the second-order
    /// model `m(x) = gᵀx − ½ xᵀFx`. Use
    /// [`Self::update_trpo_with_surrogate`] to line-search against the real
    /// surrogate (that is what
    /// [`super::policy_gradient::PolicyGradientOptimizer`] does).
    fn update_trpo(&mut self, gradients: &Array1<T>) -> Result<RLOptimizationMetrics<T>> {
        let report = self.trpo_step(gradients, None::<&mut dyn FnMut(&P) -> Result<T>>)?;
        self.update_count += 1;
        Ok(Self::metrics_from_report(&report))
    }

    /// TRPO update whose backtracking line search evaluates a **real** surrogate.
    ///
    /// `surrogate` is called with the policy *after* a candidate step has been
    /// applied and must return the value of the objective being maximized
    /// (typically `E[ (π(a|s)/π_old(a|s)) · A(s,a) ]`). A candidate is accepted
    /// only when
    ///
    /// * the quadratic-model KL stays within `max_kl`,
    /// * the surrogate actually improved, and
    /// * the improvement ratio `actual / expected` exceeds `accept_ratio`.
    ///
    /// If every candidate fails, the policy is left **exactly** where it started —
    /// a rejected step is never applied.
    pub fn update_trpo_with_surrogate<F>(
        &mut self,
        gradients: &Array1<T>,
        mut surrogate: F,
    ) -> Result<RLOptimizationMetrics<T>>
    where
        F: FnMut(&P) -> Result<T>,
    {
        let report = self.trpo_step(gradients, Some(&mut surrogate))?;
        self.update_count += 1;
        Ok(Self::metrics_from_report(&report))
    }

    fn metrics_from_report(report: &TrustRegionStepReport<T>) -> RLOptimizationMetrics<T> {
        let mut metrics = RLOptimizationMetrics {
            kl_divergence: Some(report.kl),
            ..Default::default()
        };
        metrics.policy_loss = -report.surrogate_improvement;
        metrics
            .custom_metrics
            .insert("step_scale".to_string(), report.step_scale);
        metrics.custom_metrics.insert(
            "line_search_accepted".to_string(),
            if report.accepted { T::one() } else { T::zero() },
        );
        metrics
    }

    /// Shared TRPO machinery: natural gradient, `β = √(2δ / sᵀFs)` initial step,
    /// then backtracking line search with acceptance test.
    fn trpo_step(
        &mut self,
        gradients: &Array1<T>,
        surrogate: Option<&mut dyn FnMut(&P) -> Result<T>>,
    ) -> Result<TrustRegionStepReport<T>> {
        // s ≈ F⁻¹g.
        let natural_grad = self.compute_natural_gradient(gradients)?;
        let fvp = self.fisher_vector_product(&natural_grad)?;
        let shs = self.dot(&natural_grad, &fvp);

        // A non-positive curvature means the quadratic model is useless here; the
        // only safe action is to take no step at all.
        if !(shs > tiny::<T>()) {
            return Ok(TrustRegionStepReport {
                accepted: false,
                step_scale: T::zero(),
                kl: T::zero(),
                surrogate_improvement: T::zero(),
                backtracks: 0,
            });
        }

        // Full step: the largest multiple of s whose quadratic KL equals δ.
        let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());
        let beta = (two * self.config.max_kl / shs).sqrt();
        if !beta.is_finite() {
            return Err(OptimError::ComputationError(
                "TRPO step size sqrt(2δ/sᵀFs) is not finite".to_string(),
            ));
        }
        let full_step = &natural_grad * beta;

        self.line_search(gradients, &full_step, surrogate, None)
    }

    /// CPO (Constrained Policy Optimization) update.
    ///
    /// Solves the linearized safety problem of Achiam et al. (2017)
    ///
    /// ```text
    /// max_x gᵀx    s.t.   c + bᵀx ≤ 0 ,   ½ xᵀFx ≤ δ
    /// ```
    ///
    /// with `q = gᵀF⁻¹g`, `r = gᵀF⁻¹b`, `s = bᵀF⁻¹b`:
    ///
    /// * **Infeasible** (`c > 0` and `c²/s > 2δ`): no step inside the trust region
    ///   can restore feasibility, so CPO takes the pure recovery step
    ///   `x = −√(2δ/s)·F⁻¹b`, which reduces the cost as fast as the trust region
    ///   allows.
    /// * **Feasible**: both constraints active gives `λ = √(A/B)` with
    ///   `A = q − r²/s`, `B = 2δ − c²/s`, and `ν = (r + λc)/s`. If `ν < 0` the cost
    ///   constraint is inactive and the step degenerates to plain TRPO
    ///   (`ν = 0`, `λ = √(q/2δ)`). The step is `x = (F⁻¹g − ν F⁻¹b)/λ`.
    ///
    /// Requires [`Self::set_cost_constraint`]; without it there is no cost signal
    /// and the method returns [`OptimError::UnsupportedOperation`] rather than
    /// silently running unconstrained TRPO under a "CPO" label.
    fn update_cpo(&mut self, gradients: &Array1<T>) -> Result<RLOptimizationMetrics<T>> {
        let (cost_gradient, cost_surplus) =
            match self.cost_constraint.clone() {
                Some(pair) => pair,
                None => return Err(OptimError::UnsupportedOperation(
                    "CPO requires a safety constraint: call set_cost_constraint(cost_gradient, \
                     cost_surplus) before update(), or select TrustRegionMethod::TRPO for the \
                     unconstrained problem"
                        .to_string(),
                )),
            };

        if cost_gradient.len() != gradients.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "cost gradient length ({}) does not match objective gradient length ({})",
                cost_gradient.len(),
                gradients.len()
            )));
        }

        let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());
        let delta = self.config.max_kl;

        let hinv_g = self.conjugate_gradient(gradients)?;
        let hinv_b = self.conjugate_gradient(&cost_gradient)?;

        let q = self.dot(gradients, &hinv_g);
        let r = self.dot(gradients, &hinv_b);
        let s = self.dot(&cost_gradient, &hinv_b);

        // No usable cost curvature ⇒ the constraint carries no information here.
        if !(s > tiny::<T>()) {
            let report = self.trpo_step(gradients, None::<&mut dyn FnMut(&P) -> Result<T>>)?;
            self.update_count += 1;
            return Ok(Self::metrics_from_report(&report));
        }

        let c = cost_surplus;
        let b_coeff = two * delta - c * c / s;

        let step = if c > T::zero() && !(b_coeff > T::zero()) {
            // Infeasible: recovery step straight down the cost gradient.
            let scale = (two * delta / s).sqrt();
            &hinv_b * (-scale)
        } else {
            let a_coeff = q - r * r / s;
            let mut lambda = if a_coeff > T::zero() && b_coeff > T::zero() {
                (a_coeff / b_coeff).sqrt()
            } else {
                (q / (two * delta)).sqrt()
            };
            let mut nu = (r + lambda * c) / s;
            if nu < T::zero() {
                // Cost constraint inactive at the optimum ⇒ plain TRPO step.
                nu = T::zero();
                lambda = (q / (two * delta)).sqrt();
            }
            if !(lambda > tiny::<T>()) || !lambda.is_finite() {
                return Ok(Self::metrics_from_report(&TrustRegionStepReport {
                    accepted: false,
                    step_scale: T::zero(),
                    kl: T::zero(),
                    surrogate_improvement: T::zero(),
                    backtracks: 0,
                }));
            }
            (&hinv_g - &(&hinv_b * nu)) / lambda
        };

        let report = self.line_search(
            gradients,
            &step,
            None::<&mut dyn FnMut(&P) -> Result<T>>,
            Some((&cost_gradient, c)),
        )?;
        self.update_count += 1;

        let mut metrics = Self::metrics_from_report(&report);
        metrics.custom_metrics.insert("cost_surplus".to_string(), c);
        Ok(metrics)
    }

    /// Projection-based trust region update
    fn update_projection(&mut self, gradients: &Array1<T>) -> Result<RLOptimizationMetrics<T>> {
        // Project gradients onto trust region
        let projected_grad = self.project_to_trust_region(gradients)?;
        self.apply_parameter_update(&projected_grad)?;

        Ok(RLOptimizationMetrics::default())
    }

    /// Natural gradient update
    fn update_natural_gradient(
        &mut self,
        gradients: &Array1<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let natural_grad = self.compute_natural_gradient(gradients)?;
        let lr = self.natural_grad_state.adaptive_lr_state.learning_rate;
        let update_step = &natural_grad * lr;

        self.apply_parameter_update(&update_step)?;

        Ok(RLOptimizationMetrics::default())
    }

    /// Compute natural gradient using conjugate gradient method
    fn compute_natural_gradient(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        // Solve F * x = g for natural gradient x, where F is Fisher information matrix
        self.conjugate_gradient(gradients)
    }

    /// Conjugate gradient solver for the (damped) Fisher information system.
    ///
    /// Guards every division that the textbook recurrence performs:
    /// * a zero right-hand side returns the exact solution `x = 0` immediately
    ///   instead of computing `0/0` for `α`;
    /// * a vanishing (or non-finite) curvature `pᵀAp` breaks out with the best
    ///   iterate found so far rather than injecting `±inf`/NaN into `x`;
    /// * `β = rsnew/rsold` is only evaluated while `rsold` is strictly positive.
    fn conjugate_gradient(&self, b: &Array1<T>) -> Result<Array1<T>> {
        let n = b.len();
        let mut x = Array1::zeros(n);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rsold = self.dot(&r, &r);

        // ‖b‖ = 0 (or non-finite): x = 0 already solves the system.
        if !(rsold > tiny::<T>()) {
            return Ok(x);
        }

        for _i in 0..self.config.cg_iters {
            let ap = self.fisher_vector_product(&p)?;
            let pap = self.dot(&p, &ap);

            // Zero / negative / non-finite curvature: stop with the current iterate.
            if !(pap.abs() > tiny::<T>()) || !pap.is_finite() {
                break;
            }

            let alpha = rsold / pap;

            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);

            let rsnew = self.dot(&r, &r);

            if rsnew.sqrt() < self.config.cg_tolerance {
                break;
            }
            if !(rsnew > tiny::<T>()) {
                break;
            }

            let beta = rsnew / rsold;
            p = &r + &(&p * beta);
            rsold = rsnew;
        }

        Ok(x)
    }

    /// Empirical Fisher information matrix vector product.
    ///
    /// The Fisher Information Matrix is `F = E[ g gᵀ ]` where
    /// `g = ∇_θ log π(a | s)` is the score (gradient of the log-likelihood). Given
    /// `N` per-sample score rows `g_i`, the empirical estimate is
    /// `F̂ = (1/N) Σ_i g_i g_iᵀ`.
    ///
    /// The product `F̂·v` is computed WITHOUT ever forming the dense `d × d` matrix
    /// by exploiting `g_i g_iᵀ v = g_i (g_i · v)`, giving
    /// `F̂ v = (1/N) Σ_i g_i (g_i · v)` in `O(N · d)` time and `O(d)` memory.
    ///
    /// For conjugate-gradient stability the DAMPED product is returned:
    /// `F̂·v + cg_damping·v` (the standard TRPO/Hessian-free damping). An optional
    /// additional ridge `fisher_reg·v` is folded into the estimate so that the
    /// effective system is `(F̂ + fisher_reg·I + cg_damping·I) v`.
    ///
    /// Fallback: if no score samples are available (`None` or an empty matrix),
    /// the Fisher is treated as the identity and `v + cg_damping·v` is returned.
    /// This keeps the CG solver well-defined before any empirical data is fed in.
    fn fisher_vector_product(&self, v: &Array1<T>) -> Result<Array1<T>> {
        // CG damping is always applied (primary regularization for CG stability).
        let damping = self.config.cg_damping;

        match &self.score_samples {
            Some(samples) if samples.nrows() > 0 => {
                let n_samples = samples.nrows();
                let dim = samples.ncols();

                if dim != v.len() {
                    return Err(OptimError::DimensionMismatch(format!(
                        "Score sample dimension ({}) does not match vector dimension ({})",
                        dim,
                        v.len()
                    )));
                }

                // Accumulate F̂ v = (1/N) Σ_i g_i (g_i · v) without forming F̂.
                let mut accum: Array1<T> = Array1::zeros(dim);
                for row in samples.rows() {
                    // g_i · v
                    let proj: T = row.iter().zip(v.iter()).map(|(&g, &x)| g * x).sum();
                    // accum += g_i * (g_i · v)
                    for (acc, &g) in accum.iter_mut().zip(row.iter()) {
                        *acc = *acc + g * proj;
                    }
                }

                let inv_n = T::one()
                    / T::from(n_samples).ok_or_else(|| {
                        OptimError::ComputationError(
                            "Failed to convert sample count to scalar type".to_string(),
                        )
                    })?;
                accum.mapv_inplace(|x| x * inv_n);

                // (F̂ + fisher_reg·I + cg_damping·I) v
                let ridge = self.config.fisher_reg + damping;
                Ok(&accum + &(v * ridge))
            }
            // Identity-Fisher fallback: treat F̂ = I, return (I + cg_damping·I) v.
            _ => Ok(v + &(v * damping)),
        }
    }

    /// Backtracking line search over `full_step · backtrack_coeff^j`.
    ///
    /// Each candidate is *applied* to the policy, scored, and **reverted unless it
    /// is accepted** — the previous implementation applied the last candidate it
    /// examined even after rejecting it, which silently pushed the policy outside
    /// the trust region. Acceptance requires all of:
    ///
    /// * quadratic-model KL `½ xᵀFx ≤ max_kl`,
    /// * a strictly positive surrogate improvement,
    /// * improvement ratio `actual / expected > accept_ratio` (`expected = gᵀx`),
    /// * and, when a CPO cost constraint is supplied, `c + bᵀx ≤ 0`.
    ///
    /// If no candidate is accepted the policy is left untouched (zero step).
    fn line_search(
        &mut self,
        gradients: &Array1<T>,
        full_step: &Array1<T>,
        mut surrogate: Option<&mut dyn FnMut(&P) -> Result<T>>,
        cost_constraint: Option<(&Array1<T>, T)>,
    ) -> Result<TrustRegionStepReport<T>> {
        let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));

        // Baseline surrogate value at the current parameters.
        let base = match surrogate {
            Some(ref mut f) => f(&self.policy)?,
            None => T::zero(),
        };

        let mut scale = T::one();
        for attempt in 0..self.config.max_backtracks.max(1) {
            let step = full_step * scale;

            // Quadratic-model KL of this candidate.
            let fvp = self.fisher_vector_product(&step)?;
            let kl = half * self.dot(&step, &fvp);
            let expected = self.dot(gradients, &step);

            // Cost feasibility of the linearized safety constraint.
            let cost_ok = match cost_constraint {
                Some((cost_gradient, surplus)) => {
                    surplus + self.dot(cost_gradient, &step) <= T::zero()
                }
                None => true,
            };

            self.apply_parameter_update(&step)?;

            let value = match surrogate {
                Some(ref mut f) => f(&self.policy)?,
                // Model surrogate m(x) = gᵀx − ½ xᵀFx (base is 0).
                None => expected - kl,
            };
            let actual = value - base;

            let ratio = if expected > tiny::<T>() {
                actual / expected
            } else {
                T::neg_infinity()
            };

            let accept = kl <= self.config.max_kl
                && cost_ok
                && actual > T::zero()
                && ratio > self.config.accept_ratio;

            if accept {
                self.natural_grad_state.adaptive_lr_state.success_count += 1;
                return Ok(TrustRegionStepReport {
                    accepted: true,
                    step_scale: scale,
                    kl,
                    surrogate_improvement: actual,
                    backtracks: attempt,
                });
            }

            // Rejected: undo the candidate before trying a shorter one.
            self.apply_parameter_update(&(&step * -T::one()))?;
            scale = scale * self.config.backtrack_coeff;
        }

        // Nothing acceptable: take no step at all.
        self.natural_grad_state.adaptive_lr_state.failure_count += 1;
        Ok(TrustRegionStepReport {
            accepted: false,
            step_scale: T::zero(),
            kl: T::zero(),
            surrogate_improvement: T::zero(),
            backtracks: self.config.max_backtracks.max(1),
        })
    }

    /// Check if a step of `stepsize` along `direction` satisfies the KL trust region.
    fn check_trust_region_constraint(&self, direction: &Array1<T>, stepsize: T) -> Result<bool> {
        // Compute expected KL divergence after update
        let expected_kl = self.estimate_kl_divergence(direction, stepsize)?;
        Ok(expected_kl <= self.config.max_kl)
    }

    /// Estimate KL divergence for proposed update
    fn estimate_kl_divergence(&self, direction: &Array1<T>, stepsize: T) -> Result<T> {
        // Quadratic approximation: KL ≈ 0.5 * d^T * F * d * step_size^2
        let fvp = self.fisher_vector_product(direction)?;
        let kl_estimate = T::from(0.5).unwrap_or_else(|| T::zero())
            * self.dot(direction, &fvp)
            * stepsize
            * stepsize;
        Ok(kl_estimate)
    }

    /// Project gradients onto trust region
    fn project_to_trust_region(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let grad_norm = self.norm(gradients);
        let max_norm = (T::from(2.0).unwrap_or_else(|| T::zero()) * self.config.max_kl).sqrt();

        if grad_norm <= max_norm {
            Ok(gradients.clone())
        } else {
            Ok(gradients * (max_norm / grad_norm))
        }
    }

    /// Apply a flat parameter update onto the policy network.
    ///
    /// The flat `update` vector is mapped back onto the policy's named parameters.
    /// Keys are visited in SORTED order for determinism, the flat update is sliced
    /// into contiguous chunks matching each parameter's length, and the resulting
    /// `HashMap<String, Array1<T>>` is forwarded to `policy.update_parameters`.
    ///
    /// Returns an error if the flat update length does not equal the total
    /// parameter count across all named parameters.
    fn apply_parameter_update(&mut self, update: &Array1<T>) -> Result<()> {
        let params = self.policy.get_parameters();
        let deltas = unflatten_named(&params, update)?;
        self.policy.update_parameters(&deltas)
    }

    /// Dot product
    fn dot(&self, a: &Array1<T>, b: &Array1<T>) -> T {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Vector norm
    fn norm(&self, v: &Array1<T>) -> T {
        self.dot(v, v).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ActionDistribution, DistributionType, PolicyEvaluation};
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{arr1, arr2};
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Minimal mock policy network over a tiny parameter map (`"w"`, length 3).
    ///
    /// Only `get_parameters` / `update_parameters` carry real behavior; the
    /// distribution-related trait methods return trivially valid values. The last
    /// gradients passed to `update_parameters` are recorded via interior mutability
    /// so tests can assert the parameter update was actually applied.
    struct MockPolicy {
        params: HashMap<String, Array1<f64>>,
        last_gradients: RefCell<Option<HashMap<String, Array1<f64>>>>,
    }

    impl MockPolicy {
        fn new() -> Self {
            let mut params = HashMap::new();
            params.insert("w".to_string(), arr1(&[0.0, 0.0, 0.0]));
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
            // Apply and record the update for assertions.
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

    fn make_optimizer(cg_damping: f64) -> TrustRegionOptimizer<f64, MockPolicy> {
        // Isolate the cg_damping contribution from the additional ridge for tests.
        let config = TrustRegionConfig::<f64> {
            cg_damping,
            fisher_reg: 0.0,
            ..Default::default()
        };
        TrustRegionOptimizer::new(config, MockPolicy::new())
    }

    /// Reference dense computation of `(1/N) Σ_i g_i (g_i · v) + cg_damping · v`.
    fn reference_fvp(samples: &Array2<f64>, v: &Array1<f64>, damping: f64) -> Array1<f64> {
        let n = samples.nrows();
        let dim = samples.ncols();
        let mut out = Array1::<f64>::zeros(dim);
        for row in samples.rows() {
            let proj: f64 = row.iter().zip(v.iter()).map(|(&g, &x)| g * x).sum();
            for (o, &g) in out.iter_mut().zip(row.iter()) {
                *o += g * proj;
            }
        }
        out.mapv_inplace(|x| x / n as f64);
        &out + &(v * damping)
    }

    #[test]
    fn test_fisher_vector_product_matches_empirical_formula() {
        let damping = 0.1;
        let mut opt = make_optimizer(damping);

        // Two score samples over a 3-dim parameter space.
        let samples = arr2(&[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]);
        opt.set_score_samples(samples.clone());

        let v = arr1(&[0.3, -0.7, 1.1]);
        let got = opt
            .fisher_vector_product(&v)
            .expect("fisher-vector product");
        let expected = reference_fvp(&samples, &v, damping);

        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fisher_vector_product_identity_fallback() {
        let damping = 0.1;
        let opt = make_optimizer(damping);
        // No score samples set => identity Fisher: (I + cg_damping I) v.
        let v = arr1(&[1.0, -2.0, 4.0]);
        let got = opt
            .fisher_vector_product(&v)
            .expect("fisher-vector product");
        let expected = &v + &(&v * damping);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-12);
        }

        // Empty score matrix also triggers the fallback.
        let mut opt2 = make_optimizer(damping);
        opt2.set_score_samples(Array2::<f64>::zeros((0, 3)));
        let got2 = opt2
            .fisher_vector_product(&v)
            .expect("fisher-vector product");
        for (g, e) in got2.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_conjugate_gradient_solves_damped_system() {
        let damping = 0.5;
        let mut opt = make_optimizer(damping);
        let samples = arr2(&[[1.0, 0.5, -0.3], [0.2, 1.5, 0.7], [-0.5, 0.1, 1.2]]);
        opt.set_score_samples(samples.clone());

        let b = arr1(&[1.0, -2.0, 0.5]);
        let x = opt.conjugate_gradient(&b).expect("conjugate gradient");

        // Residual ||(F̂ + λI) x − b|| must be small: fisher_vector_product already
        // applies the damped operator (F̂ + cg_damping·I) since fisher_reg = 0.
        let ax = opt
            .fisher_vector_product(&x)
            .expect("fisher-vector product");
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(&a, &bv)| (a - bv) * (a - bv))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual < 1e-6,
            "CG residual too large: {residual} (x = {x:?})"
        );
    }

    #[test]
    fn test_apply_parameter_update_forwards_split_gradient() {
        let damping = 0.1;
        let mut opt = make_optimizer(damping);

        // Flat update of length 3 maps onto the single "w" parameter (len 3).
        let update = arr1(&[0.1, 0.2, 0.3]);
        opt.apply_parameter_update(&update).expect("apply update");

        // The mock recorded the forwarded gradient map.
        let recorded = opt.policy.last_gradients.borrow();
        let map = recorded.as_ref().expect("update_parameters was not called");
        let w_grad = map.get("w").expect("missing 'w' gradient");
        assert_eq!(w_grad.len(), 3);
        assert_abs_diff_eq!(w_grad[0], 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(w_grad[1], 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(w_grad[2], 0.3, epsilon = 1e-12);

        // And the policy parameters were actually advanced by the update.
        let params = opt.policy.get_parameters();
        let w = params.get("w").expect("w parameter");
        assert_abs_diff_eq!(w[0], 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(w[1], 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(w[2], 0.3, epsilon = 1e-12);
    }

    #[test]
    fn test_apply_parameter_update_length_mismatch_errors() {
        let mut opt = make_optimizer(0.1);
        // Wrong length (4 != 3) must return an error.
        let bad = arr1(&[0.1, 0.2, 0.3, 0.4]);
        assert!(opt.apply_parameter_update(&bad).is_err());
    }

    #[test]
    fn test_kl_estimate_uses_real_damped_fisher() {
        let damping = 0.2;
        let mut opt = make_optimizer(damping);
        let samples = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        opt.set_score_samples(samples.clone());

        let direction = arr1(&[1.0, 1.0, 1.0]);
        let step = 0.5_f64;

        // KL ≈ 0.5 * dᵀ (F̂ + λI) d * step².
        let fvp = reference_fvp(&samples, &direction, damping);
        let quad: f64 = direction.iter().zip(fvp.iter()).map(|(&d, &f)| d * f).sum();
        let expected = 0.5 * quad * step * step;

        let got = opt
            .estimate_kl_divergence(&direction, step)
            .expect("kl estimate");
        assert_abs_diff_eq!(got, expected, epsilon = 1e-10);
    }
}
