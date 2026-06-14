// Trust Region Methods for Policy Optimization
//
// This module implements trust region methods including TRPO (Trust Region Policy Optimization)
// and other constrained optimization techniques for policy learning.

#[allow(dead_code)]
use super::{PolicyNetwork, RLOptimizationMetrics};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

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

    /// Natural gradient state
    natural_grad_state: NaturalGradientState<T>,

    /// Update counter
    update_count: usize,
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

    /// Perform trust region update
    pub fn update(&mut self, gradients: &Array1<T>) -> Result<RLOptimizationMetrics<T>> {
        match self.config.method {
            TrustRegionMethod::TRPO => self.update_trpo(gradients),
            TrustRegionMethod::CPO => self.update_cpo(gradients),
            TrustRegionMethod::Projection => self.update_projection(gradients),
            TrustRegionMethod::NaturalGradient => self.update_natural_gradient(gradients),
        }
    }

    /// TRPO update with conjugate gradient and line search
    fn update_trpo(&mut self, gradients: &Array1<T>) -> Result<RLOptimizationMetrics<T>> {
        // 1. Compute natural gradient using conjugate gradient
        let natural_grad = self.compute_natural_gradient(gradients)?;

        // 2. Compute step size using line search
        let step_size = self.line_search(&natural_grad)?;

        // 3. Apply update
        let update_step = &natural_grad * step_size;
        self.apply_parameter_update(&update_step)?;

        self.update_count += 1;

        Ok(RLOptimizationMetrics::default())
    }

    /// CPO (Constrained Policy Optimization) update
    fn update_cpo(&mut self, gradients: &Array1<T>) -> Result<RLOptimizationMetrics<T>> {
        // CPO extends TRPO with additional safety constraints
        self.update_trpo(gradients) // Simplified
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

    /// Conjugate gradient solver for Fisher information system
    fn conjugate_gradient(&self, b: &Array1<T>) -> Result<Array1<T>> {
        let n = b.len();
        let mut x = Array1::zeros(n);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rsold = self.dot(&r, &r);

        for _i in 0..self.config.cg_iters {
            let ap = self.fisher_vector_product(&p)?;
            let alpha = rsold / self.dot(&p, &ap);

            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);

            let rsnew = self.dot(&r, &r);

            if rsnew.sqrt() < self.config.cg_tolerance {
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

    /// Line search for step size selection
    fn line_search(&self, direction: &Array1<T>) -> Result<T> {
        let mut step_size = T::from(1.0).unwrap_or_else(|| T::zero());

        for _i in 0..self.config.max_backtracks {
            // Check if step satisfies trust region constraint
            if self.check_trust_region_constraint(direction, step_size)? {
                return Ok(step_size);
            }

            step_size = step_size * self.config.backtrack_coeff;
        }

        // If no acceptable step found, use small step
        Ok(step_size)
    }

    /// Check if step satisfies trust region constraint
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
        let got = opt.fisher_vector_product(&v).unwrap();
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
        let got = opt.fisher_vector_product(&v).unwrap();
        let expected = &v + &(&v * damping);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-12);
        }

        // Empty score matrix also triggers the fallback.
        let mut opt2 = make_optimizer(damping);
        opt2.set_score_samples(Array2::<f64>::zeros((0, 3)));
        let got2 = opt2.fisher_vector_product(&v).unwrap();
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
        let x = opt.conjugate_gradient(&b).unwrap();

        // Residual ||(F̂ + λI) x − b|| must be small: fisher_vector_product already
        // applies the damped operator (F̂ + cg_damping·I) since fisher_reg = 0.
        let ax = opt.fisher_vector_product(&x).unwrap();
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
        opt.apply_parameter_update(&update).unwrap();

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
        let w = params.get("w").unwrap();
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

        let got = opt.estimate_kl_divergence(&direction, step).unwrap();
        assert_abs_diff_eq!(got, expected, epsilon = 1e-10);
    }
}
