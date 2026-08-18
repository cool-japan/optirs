use std::fmt::Debug;
// Higher-order automatic differentiation
//
// This module implements computation of higher-order derivatives including
// Hessians, third-order derivatives, and mixed partial derivatives for
// advanced optimization algorithms.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::Float;

use crate::error::{OptimError, Result};

/// Hessian-vector-product and mixed-partial back ends.
///
/// Kept in its own file so `higher_order.rs` stays under the 2000-line cap.
pub mod hvp;

/// Higher-order differentiation engine
pub struct HigherOrderEngine<
    T: Float
        + Debug
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand,
> {
    /// Maximum derivative order to compute
    _maxorder: usize,

    /// Use mixed-mode for Hessian computation
    mixed_mode: bool,

    /// Finite difference settings for numerical verification
    finite_diff_eps: T,

    /// Enable automatic parallelization
    parallel_computation: bool,

    /// Thread pool size
    thread_pool_size: usize,

    /// Use sparse computations when beneficial
    adaptive_sparsity: bool,

    /// Automatic differentiation mode selection
    auto_mode_selection: bool,

    /// Performance profiler
    profiler: ComputationProfiler<T>,
}

/// Hessian computation configuration
#[derive(Debug, Clone)]
pub struct HessianConfig {
    /// Use exact computation (vs approximation)
    pub exact: bool,

    /// Use sparse representation
    pub sparse: bool,

    /// Sparsity threshold
    pub sparsity_threshold: f64,

    /// Use diagonal approximation
    pub diagonal_only: bool,

    /// Use BFGS approximation
    pub bfgs_approximation: bool,

    /// Use finite differences for verification
    pub verify_with_finite_diff: bool,
}

impl Default for HessianConfig {
    fn default() -> Self {
        Self {
            exact: true,
            sparse: false,
            sparsity_threshold: 1e-8,
            diagonal_only: false,
            bfgs_approximation: false,
            verify_with_finite_diff: false,
        }
    }
}

/// Sparse Hessian representation
#[derive(Debug, Clone)]
pub struct SparseHessian<T: Float + Debug + Send + Sync + 'static> {
    /// Row indices
    pub rows: Vec<usize>,

    /// Column indices  
    pub cols: Vec<usize>,

    /// Values
    pub values: Vec<T>,

    /// Matrix dimensions
    pub shape: (usize, usize),

    /// Number of non-zero elements
    pub nnz: usize,
}

/// Third-order derivative tensor
#[derive(Debug, Clone)]
pub struct ThirdOrderTensor<T: Float + Debug + Send + Sync + 'static> {
    /// Dense tensor data
    pub data: Array3<T>,

    /// Tensor dimensions
    pub shape: (usize, usize, usize),
}

/// Mixed partial derivatives
#[derive(Debug, Clone)]
pub struct MixedPartials<T: Float + Debug + Send + Sync + 'static> {
    /// Variable indices for mixed partial
    pub variables: Vec<usize>,

    /// Derivative order for each variable
    pub orders: Vec<usize>,

    /// Computed value
    pub value: T,

    /// Computation method used
    pub method: MixedPartialMethod,
}

/// Methods for computing mixed partials.
///
/// This used to list `ForwardOverReverse`, `ReverseOverForward` and
/// `PureForward` alongside `FiniteDifference` — but all three delegated, in one
/// line each, to the finite-difference routine (finding F89). The variants below
/// are the ones that can genuinely differ behind this engine's black-box
/// objective type; see [`crate::higher_order::hvp`] for why nested automatic
/// differentiation cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedPartialMethod {
    /// Central finite differences: `O(h²)` accurate, evaluates on both sides of
    /// the base point. The default and the most accurate available method.
    FiniteDifference,

    /// One-sided *forward* finite differences: `O(h)` accurate, never evaluates
    /// below the base point in any differentiated coordinate. Use it when the
    /// objective is undefined or discontinuous on that side.
    ForwardFiniteDifference,

    /// Nested automatic differentiation. **Not available** behind this engine's
    /// `Fn(&Array1<T>) -> T` objective — [`Self::is_available`] returns `false`
    /// and [`HigherOrderEngine::mixed_partial`] returns a typed error naming the
    /// alternative, instead of silently running a finite difference under an
    /// autodiff name.
    NestedAutodiff,
}

/// Layer information for K-FAC computation
#[derive(Debug, Clone)]
pub struct LayerInfo<T: Float + Debug + Send + Sync + 'static> {
    pub layer_type: LayerType,
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array2<T>,
    pub bias: Option<Array1<T>>,
}

/// Types of neural network layers
#[derive(Debug, Clone, Copy)]
pub enum LayerType {
    Linear,
    Convolutional,
    LSTM,
    Attention,
}

/// Machine-precision-aware central-difference step for a derivative of the
/// given `order`.
///
/// A central stencil for the `order`-th derivative has truncation error
/// `O(h²)` and roundoff error `O(ε_mach / h^order)`; balancing the two gives
/// the optimum `h ≈ ε_mach^{1/(order+2)}`. Deriving the step from
/// [`Float::epsilon`] adapts it to `T`'s precision (e.g. ~6e-6 for an `f64`
/// first derivative, ~5e-3 for `f32`), unlike a single hard-coded constant
/// which is far too small for `f32` and for high-order stencils.
fn central_step<T: Float>(order: usize) -> T {
    let eps_mach = T::epsilon();
    let denom = T::from(order + 2).unwrap_or_else(|| T::from(3usize).unwrap_or_else(T::one));
    eps_mach.powf(T::one() / denom)
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + 'static
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > HigherOrderEngine<T>
{
    /// Create a new higher-order differentiation engine
    pub fn new(_maxorder: usize) -> Self {
        Self {
            _maxorder,
            mixed_mode: true,
            finite_diff_eps: scirs2_core::numeric::NumCast::from(1e-5).unwrap_or_else(|| T::zero()),
            parallel_computation: true,
            thread_pool_size: 4, // Conservative default
            adaptive_sparsity: true,
            auto_mode_selection: true,
            profiler: ComputationProfiler::new(),
        }
    }

    /// Create a new engine with advanced configuration
    pub fn with_config(config: HigherOrderConfig<T>) -> Self {
        Self {
            _maxorder: config._maxorder,
            mixed_mode: config.mixed_mode,
            finite_diff_eps: config.finite_diff_eps,
            parallel_computation: config.parallel_computation,
            thread_pool_size: config.thread_pool_size,
            adaptive_sparsity: config.adaptive_sparsity,
            auto_mode_selection: config.auto_mode_selection,
            profiler: ComputationProfiler::new(),
        }
    }

    /// Enable/disable mixed-mode computation
    pub fn set_mixed_mode(&mut self, enabled: bool) {
        self.mixed_mode = enabled;
    }

    /// Set finite difference epsilon for numerical verification
    pub fn set_finite_diff_eps(&mut self, eps: T) {
        self.finite_diff_eps = eps;
    }

    /// Central-difference step for a derivative of the given `order`.
    ///
    /// Returns the coarser of the machine-precision-optimal step
    /// ([`central_step`]) and any user-configured `finite_diff_eps`, so an
    /// explicit override can only *widen* the step, never sharpen it into the
    /// roundoff-dominated regime.
    fn fd_step(&self, order: usize) -> T {
        let machine = central_step::<T>(order);
        if self.finite_diff_eps > machine {
            self.finite_diff_eps
        } else {
            machine
        }
    }

    /// Compute Hessian matrix using forward-over-reverse mode.
    ///
    /// When `HessianConfig::sparse` is set and the problem is large enough that
    /// sparsification pays for itself, entries
    /// below `HessianConfig::sparsity_threshold` are zeroed. Previously
    /// `sparse`/`sparsity_threshold` were accepted and ignored: the sparsity
    /// decision and the thresholding routine both existed but nothing called
    /// them.
    ///
    /// The wall time is recorded on the engine's profiler, so
    /// [`Self::get_derivative_stats`]'s `performance_profile` reports measured
    /// numbers rather than the zeros it returned when nothing was ever recorded.
    pub fn hessian_forward_over_reverse(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        config: &HessianConfig,
    ) -> Result<Array2<T>> {
        let n = point.len();

        if config.diagonal_only {
            return self.hessian_diagonal(&function, point);
        }

        let started = std::time::Instant::now();
        let parallel = self.should_use_parallel(n);
        let sparse = self.should_use_sparse(n, config);
        let mut hessian = Array2::zeros((n, n));

        // Use forward-over-reverse: compute one row of Hessian at a time
        for i in 0..n {
            let grad_fn = |x: &Array1<T>| -> Array1<T> {
                self.gradient_at_point(&function, x)
                    .unwrap_or_else(|_| Array1::zeros(n))
            };

            // Compute directional derivative of gradient
            let mut direction = Array1::zeros(n);
            direction[i] = T::one();

            let hessian_row =
                self.directional_derivative_of_gradient(&grad_fn, point, &direction)?;

            for j in 0..n {
                hessian[[i, j]] = hessian_row[j];
            }
        }

        // Symmetrize if needed
        if config.exact {
            for i in 0..n {
                for j in i + 1..n {
                    let avg = (hessian[[i, j]] + hessian[[j, i]])
                        / scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero());
                    hessian[[i, j]] = avg;
                    hessian[[j, i]] = avg;
                }
            }
        }

        // Verify with finite differences if requested
        if config.verify_with_finite_diff {
            let fd_hessian = self.finite_difference_hessian(&function, point)?;
            self.verify_hessian_accuracy(&hessian, &fd_hessian)?;
        }

        if sparse {
            hessian = self.apply_adaptive_sparsity(hessian, config.sparsity_threshold)?;
        }
        self.profiler
            .record_hessian_computation(n, started.elapsed(), parallel, sparse);

        Ok(hessian)
    }

    /// Compute the Hessian **column by column**, by taking the gradient of each
    /// first partial derivative.
    ///
    /// The name is historical — it describes the reverse-over-forward autodiff
    /// scheme this emulates, not the implementation, which is a nested central
    /// finite difference (see [`hvp`] for why autodiff cannot be threaded through
    /// this engine's objective type). Distinct from
    /// [`Self::hessian_forward_over_reverse`], which works row-wise from a
    /// directional derivative of the gradient.
    ///
    /// Both nested steps use the *order-2* step. With the old raw
    /// `finite_diff_eps` (`1e-5`) on both levels the roundoff floor was
    /// `ε/h² ≈ 2e-6` in `f64` and `≈ 1.2e3` in `f32` — i.e. the `f32` result was
    /// pure noise.
    pub fn hessian_reverse_over_forward(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        _config: &HessianConfig,
    ) -> Result<Array2<T>> {
        let n = point.len();
        let mut hessian = Array2::zeros((n, n));
        let inner_step = self.fd_step(2);

        // Use reverse-over-forward: compute one column of Hessian at a time
        for j in 0..n {
            // Create a function that computes partial derivative w.r.t. x_j
            let partial_fn = |x: &Array1<T>| -> T {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();

                x_plus[j] = x_plus[j] + inner_step;
                x_minus[j] = x_minus[j] - inner_step;

                (function(&x_plus) - function(&x_minus))
                    / (scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero())
                        * inner_step)
            };

            // Compute gradient of partial derivative
            let hessian_col = self.gradient_at_point(&partial_fn, point)?;

            for i in 0..n {
                hessian[[i, j]] = hessian_col[i];
            }
        }

        Ok(hessian)
    }

    /// Compute diagonal Hessian elements only
    pub fn hessian_diagonal(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array2<T>> {
        let n = point.len();
        let mut hessian = Array2::zeros((n, n));
        let h = self.fd_step(2);

        // Compute second derivatives using finite differences
        for i in 0..n {
            let mut x_plus = point.clone();
            let mut x_minus = point.clone();

            x_plus[i] = x_plus[i] + h;
            x_minus[i] = x_minus[i] - h;

            let f_plus = function(&x_plus);
            let f_center = function(point);
            let f_minus = function(&x_minus);

            // Second derivative: f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2
            let second_deriv = (f_plus
                - scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero()) * f_center
                + f_minus)
                / (h * h);

            hessian[[i, i]] = second_deriv;
        }

        Ok(hessian)
    }

    /// Compute sparse Hessian representation
    pub fn sparse_hessian(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        config: &HessianConfig,
    ) -> Result<SparseHessian<T>> {
        let dense_hessian = self.hessian_forward_over_reverse(function, point, config)?;

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        let threshold = scirs2_core::numeric::NumCast::from(config.sparsity_threshold)
            .unwrap_or_else(|| T::zero());

        for i in 0..dense_hessian.nrows() {
            for j in 0..dense_hessian.ncols() {
                let val = dense_hessian[[i, j]];
                if val.abs() > threshold {
                    rows.push(i);
                    cols.push(j);
                    values.push(val);
                }
            }
        }

        let nnz = values.len();
        Ok(SparseHessian {
            rows,
            cols,
            values,
            shape: dense_hessian.dim(),
            nnz,
        })
    }

    /// Compute third-order derivatives (tensor)
    pub fn third_order_derivatives(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<ThirdOrderTensor<T>> {
        let n = point.len();
        let mut tensor = Array3::zeros((n, n, n));

        // Compute third derivatives using finite differences
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let third_deriv = self.compute_third_partial(&function, point, i, j, k)?;
                    tensor[[i, j, k]] = third_deriv;
                }
            }
        }

        Ok(ThirdOrderTensor {
            data: tensor.clone(),
            shape: tensor.dim(),
        })
    }

    /// Compute mixed partial derivatives
    pub fn mixed_partial(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
        method: MixedPartialMethod,
    ) -> Result<MixedPartials<T>> {
        if variables.len() != orders.len() {
            return Err(OptimError::InvalidConfig(
                "Variables and orders length mismatch".to_string(),
            ));
        }

        let value = match method {
            MixedPartialMethod::FiniteDifference => {
                self.mixed_partial_finite_difference(&function, point, variables, orders)?
            }
            MixedPartialMethod::ForwardFiniteDifference => {
                self.mixed_partial_forward_difference(&function, point, variables, orders)?
            }
            MixedPartialMethod::NestedAutodiff => {
                return Err(OptimError::InvalidConfig(
                    hvp::NESTED_AUTODIFF_UNAVAILABLE.to_string(),
                ))
            }
        };

        Ok(MixedPartials {
            variables: variables.to_vec(),
            orders: orders.to_vec(),
            value,
            method,
        })
    }

    /// Compute Hessian-vector product efficiently
    pub fn hessian_vector_product(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Use forward-over-reverse mode for efficient Hv computation
        let grad_fn = |x: &Array1<T>| -> Array1<T> {
            self.gradient_at_point(&function, x)
                .unwrap_or_else(|_| Array1::zeros(x.len()))
        };

        self.directional_derivative_of_gradient(&grad_fn, point, vector)
    }

    /// Compute vector-Hessian-vector product (quadratic form)
    pub fn vector_hessian_vector_product(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<T> {
        let hv = self.hessian_vector_product(function, point, vector)?;
        Ok(vector.dot(&hv))
    }

    /// Verify higher-order derivatives using finite differences
    pub fn verify_derivatives(
        &self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        computed_hessian: &Array2<T>,
    ) -> Result<DerivativeVerification<T>> {
        let fd_hessian = self.finite_difference_hessian(&function, point)?;

        let max_error = self.compute_max_error(computed_hessian, &fd_hessian);
        let avg_error = self.compute_avg_error(computed_hessian, &fd_hessian);
        let relative_error = self.compute_relative_error(computed_hessian, &fd_hessian);

        let is_accurate =
            max_error < scirs2_core::numeric::NumCast::from(1e-4).unwrap_or_else(|| T::zero());

        Ok(DerivativeVerification {
            max_absolute_error: max_error,
            avg_absolute_error: avg_error,
            max_relative_error: relative_error,
            is_accurate,
            finite_diff_hessian: fd_hessian,
        })
    }

    /// Get higher-order derivative statistics
    pub fn get_derivative_stats(&self) -> HigherOrderStats {
        HigherOrderStats {
            _maxorder: self._maxorder,
            mixed_mode_enabled: self.mixed_mode,
            memory_usage_estimate: self.estimate_memory_usage(),
            parallel_computation: self.parallel_computation,
            thread_pool_size: self.thread_pool_size,
            adaptive_sparsity: self.adaptive_sparsity,
            auto_mode_selection: self.auto_mode_selection,
            performance_profile: self.profiler.get_summary(),
        }
    }

    /// Advanced Hessian-vector product with automatic mode selection
    pub fn hessian_vector_product_advanced(
        &mut self,
        function: impl Fn(&Array1<T>) -> T + Send + Sync,
        point: &Array1<T>,
        vector: &Array1<T>,
        mode: Option<HvpMode>,
    ) -> Result<Array1<T>> {
        let n = point.len();
        let selected_mode = mode.unwrap_or_else(|| self.select_optimal_hvp_mode(n));

        let started = std::time::Instant::now();
        let product = match selected_mode {
            HvpMode::CentralDifference => self.hvp_central_difference(&function, point, vector),
            HvpMode::ForwardDifference => self.hvp_forward_difference(&function, point, vector),
            HvpMode::QuadraticSecant => self.hvp_quadratic_secant(&function, point, vector),
            HvpMode::MaterializedHessian => self.hvp_materialized_hessian(&function, point, vector),
            HvpMode::NestedAutodiff => Err(OptimError::InvalidConfig(
                hvp::NESTED_AUTODIFF_UNAVAILABLE.to_string(),
            )),
        }?;
        // Only successful products are timed: an unavailable mode returns
        // immediately and its duration says nothing about the cost of the work.
        self.profiler.record_hvp_computation(n, started.elapsed());
        Ok(product)
    }

    /// Compute vector-Hessian-vector efficiently
    pub fn vector_hessian_vector_efficient(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<T> {
        // Use forward-mode to compute directional derivative of gradient
        let grad_fn = |x: &Array1<T>| -> Array1<T> {
            self.gradient_at_point(&function, x)
                .unwrap_or_else(|_| Array1::zeros(x.len()))
        };

        let directional_grad = self.directional_derivative_of_gradient(&grad_fn, point, vector)?;
        Ok(vector.dot(&directional_grad))
    }

    /// Compute Jacobian efficiently with automatic mode selection
    pub fn jacobian_efficient<F>(
        &mut self,
        function: F,
        point: &Array1<T>,
        output_dim: usize,
    ) -> Result<Array2<T>>
    where
        F: Fn(&Array1<T>) -> Array1<T> + Send + Sync,
    {
        let input_dim = point.len();

        let started = std::time::Instant::now();
        // Select mode based on dimensions
        let jacobian = if input_dim <= output_dim {
            // Forward mode is more efficient
            self.jacobian_forward_mode(&function, point, output_dim)?
        } else {
            // Reverse mode is more efficient
            self.jacobian_reverse_mode(&function, point, output_dim)?
        };
        self.profiler
            .record_jacobian_computation(input_dim, started.elapsed());
        Ok(jacobian)
    }

    /// Compute K-FAC approximation to the Hessian
    pub fn kfac_hessian_approximation(
        &mut self,
        layers: &[LayerInfo<T>],
        activations: &[Array1<T>],
        gradients: &[Array1<T>],
    ) -> Result<Array2<T>> {
        // The three slices are parallel arrays indexed by layer. Indexing them
        // with the layer index without checking used to panic (out of bounds) on
        // any caller that passed a short activation or gradient list.
        if activations.len() != layers.len() || gradients.len() != layers.len() {
            return Err(OptimError::InvalidConfig(format!(
                "K-FAC needs one activation and one gradient per layer: {} layers, \
                 {} activations, {} gradients",
                layers.len(),
                activations.len(),
                gradients.len()
            )));
        }
        if layers.is_empty() {
            return Err(OptimError::InvalidConfig(
                "K-FAC needs at least one layer".to_string(),
            ));
        }

        let mut kfac_blocks = Vec::with_capacity(layers.len());

        for (i, layer) in layers.iter().enumerate() {
            let activation = &activations[i];
            let gradient = &gradients[i];

            // K-FAC factorizes the layer's Fisher block as `A ⊗ G` with
            // `A = E[a aᵀ]` over the layer *inputs* and `G = E[g gᵀ]` over the
            // layer *output* pre-activation gradients. So `activation` must be
            // `input_size` long and `gradient` `output_size` long. Without this
            // check a caller that swapped or mis-sized the two slices still got
            // a plausible-looking matrix back — a silently wrong curvature
            // estimate — because the factor builders just square whatever
            // length they are handed.
            if activation.len() != layer.input_size {
                return Err(OptimError::InvalidConfig(format!(
                    "K-FAC layer {i} ({:?}) declares input_size {} but its activation \
                     vector has {} entries",
                    layer.layer_type,
                    layer.input_size,
                    activation.len()
                )));
            }
            if gradient.len() != layer.output_size {
                return Err(OptimError::InvalidConfig(format!(
                    "K-FAC layer {i} ({:?}) declares output_size {} but its gradient \
                     vector has {} entries",
                    layer.layer_type,
                    layer.output_size,
                    gradient.len()
                )));
            }

            // Compute Kronecker factors
            let factor_a = self.compute_activation_factor(activation)?;
            let factor_g = self.compute_gradient_factor(gradient)?;

            // K-FAC block is Kronecker product approximation
            let block = self.kronecker_product_approximation(&factor_a, &factor_g)?;
            kfac_blocks.push(block);
        }

        // Combine blocks into full approximation
        self.combine_kfac_blocks(&kfac_blocks)
    }

    /// Compute Natural Gradient using Fisher Information Matrix
    pub fn natural_gradient(
        &mut self,
        log_likelihood: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        gradient: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Compute Fisher Information Matrix (FIM)
        let fim = self.compute_fisher_information_matrix(&log_likelihood, point)?;

        // Natural gradient = FIM^(-1) * gradient
        self.solve_linear_system(&fim, gradient)
    }

    /// Compute truncated Newton direction
    pub fn truncated_newton_direction(
        &mut self,
        function: impl Fn(&Array1<T>) -> T + Send + Sync,
        point: &Array1<T>,
        gradient: &Array1<T>,
        max_cg_iterations: usize,
        cg_tolerance: T,
    ) -> Result<Array1<T>> {
        // Use Conjugate Gradient to approximately solve Hx = -g
        let neg_gradient = gradient.mapv(|x| -x);

        // Create a copy of point and function to avoid borrow conflicts
        let point_copy = point.clone();
        let function_copy = function;

        // `Hv` by a central difference of the gradient *along `v`*: two gradient
        // evaluations, i.e. `O(n)` objective calls.
        //
        // This used to materialize the whole Hessian one column at a time inside
        // every CG iteration — `n` gradient pairs, so `O(n²)` objective calls per
        // `Hv` and `O(n³)` for the solve — to compute a quantity that a single
        // directional difference gives exactly as accurately. On a 1000-parameter
        // problem that is ~4·10⁶ objective evaluations per CG step.
        let hvp_fn = move |v: &Array1<T>| -> Result<Array1<T>> {
            if v.len() != point_copy.len() {
                return Err(OptimError::InvalidConfig(format!(
                    "CG direction has length {} but the point has {}",
                    v.len(),
                    point_copy.len()
                )));
            }
            // Keep the displacement `h·v` at the intended scale even when `v` is
            // large; CG residuals are not normalized.
            let scale = v.iter().fold(T::one(), |acc, value| {
                let magnitude = value.abs();
                if magnitude > acc {
                    magnitude
                } else {
                    acc
                }
            });
            let eps = central_step::<T>(2) / scale;
            let two: T = scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::one());

            let grad_plus =
                Self::finite_diff_gradient(&function_copy, &(&point_copy + &(v * eps)))?;
            let grad_minus =
                Self::finite_diff_gradient(&function_copy, &(&point_copy - &(v * eps)))?;

            Ok((grad_plus - grad_minus) / (two * eps))
        };

        self.conjugate_gradient_solve(hvp_fn, &neg_gradient, max_cg_iterations, cg_tolerance)
    }

    /// Helper method for finite difference gradient computation
    fn finite_diff_gradient<F>(function: &F, point: &Array1<T>) -> Result<Array1<T>>
    where
        F: Fn(&Array1<T>) -> T,
    {
        let eps = central_step::<T>(1);
        let mut gradient = Array1::zeros(point.len());

        for i in 0..point.len() {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();

            point_plus[i] = point_plus[i] + eps;
            point_minus[i] = point_minus[i] - eps;

            let loss_plus = function(&point_plus);
            let loss_minus = function(&point_minus);

            gradient[i] = (loss_plus - loss_minus)
                / (scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero()) * eps);
        }

        Ok(gradient)
    }

    // Helper methods

    fn gradient_at_point(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array1<T>> {
        let n = point.len();
        let mut gradient = Array1::zeros(n);
        let h = self.fd_step(1);

        for i in 0..n {
            let mut x_plus = point.clone();
            let mut x_minus = point.clone();

            x_plus[i] = x_plus[i] + h;
            x_minus[i] = x_minus[i] - h;

            gradient[i] = (function(&x_plus) - function(&x_minus))
                / (scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero()) * h);
        }

        Ok(gradient)
    }

    fn directional_derivative_of_gradient(
        &self,
        grad_fn: &impl Fn(&Array1<T>) -> Array1<T>,
        point: &Array1<T>,
        direction: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Differencing the gradient approximates a second-order quantity.
        let eps = self.fd_step(2);
        let point_plus = point + &(direction * eps);
        let point_minus = point - &(direction * eps);

        let grad_plus = grad_fn(&point_plus);
        let grad_minus = grad_fn(&point_minus);

        Ok((grad_plus - grad_minus)
            / (scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero()) * eps))
    }

    fn finite_difference_hessian(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array2<T>> {
        let n = point.len();
        let mut hessian = Array2::zeros((n, n));
        // Second-order stencils need the order-2 step. The raw `finite_diff_eps`
        // default of 1e-5 that used to be inlined here divides by `h² = 1e-10`,
        // putting the roundoff floor at `ε/h²` — ~2e-6 in `f64` and ~1.2e3 in
        // `f32`, where the "Hessian" was noise.
        let eps = self.fd_step(2);
        // `f(x)` does not depend on `i`; it used to be re-evaluated once per
        // diagonal entry.
        let f_center = function(point);

        for i in 0..n {
            for j in i..n {
                // Only compute upper triangle due to symmetry
                let second_deriv = if i == j {
                    // Diagonal element: f''_ii
                    let mut x_plus = point.clone();
                    let mut x_minus = point.clone();

                    x_plus[i] = x_plus[i] + eps;
                    x_minus[i] = x_minus[i] - eps;

                    let f_plus = function(&x_plus);
                    let f_minus = function(&x_minus);

                    (f_plus
                        - scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero())
                            * f_center
                        + f_minus)
                        / (eps * eps)
                } else {
                    // Off-diagonal element: f''_ij
                    let mut x_pp = point.clone();
                    x_pp[i] = x_pp[i] + eps;
                    x_pp[j] = x_pp[j] + eps;

                    let mut x_pm = point.clone();
                    x_pm[i] = x_pm[i] + eps;
                    x_pm[j] = x_pm[j] - eps;

                    let mut x_mp = point.clone();
                    x_mp[i] = x_mp[i] - eps;
                    x_mp[j] = x_mp[j] + eps;

                    let mut x_mm = point.clone();
                    x_mm[i] = x_mm[i] - eps;
                    x_mm[j] = x_mm[j] - eps;

                    (function(&x_pp) - function(&x_pm) - function(&x_mp) + function(&x_mm))
                        / (scirs2_core::numeric::NumCast::from(4.0).unwrap_or_else(|| T::zero())
                            * eps
                            * eps)
                };

                hessian[[i, j]] = second_deriv;
                hessian[[j, i]] = second_deriv; // Symmetry
            }
        }

        Ok(hessian)
    }

    /// Third partial derivative ∂³f/∂xᵢ∂xⱼ∂xₖ by central finite differences.
    ///
    /// The correct stencil depends on how many of `(i, j, k)` coincide, so
    /// three cases are handled separately (all previously collapsed into one
    /// wrong two-point formula that was off by `~1/h²`):
    ///
    /// * all equal — pure ∂³/∂xᵢ³:
    ///   `[f(x+2h) − 2f(x+h) + 2f(x−h) − f(x−2h)] / (2h³)`.
    /// * all distinct — fully mixed:
    ///   `Σ_{s∈{±1}³} sᵢsⱼsₖ · f(x + h(sᵢeᵢ+sⱼeⱼ+sₖeₖ)) / (8h³)`.
    /// * exactly two equal — semi-mixed ∂³/∂x_r²∂x_s:
    ///   a central second difference in `r` differenced once in `s`.
    fn compute_third_partial(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Result<T> {
        let n = point.len();
        if i >= n || j >= n || k >= n {
            return Err(OptimError::InvalidConfig(
                "third-derivative index out of range".to_string(),
            ));
        }
        let h = self.fd_step(3);
        let two: T = scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero());
        let eight: T = scirs2_core::numeric::NumCast::from(8.0).unwrap_or_else(|| T::zero());
        let h3 = h * h * h;

        // Evaluate `f` at `point` shifted by integer multiples of `h`.
        let shifted = |shifts: &[(usize, i32)]| -> T {
            let mut x = point.clone();
            for &(coord, mult) in shifts {
                let step: T =
                    scirs2_core::numeric::NumCast::from(mult).unwrap_or_else(|| T::zero());
                x[coord] = x[coord] + step * h;
            }
            function(&x)
        };

        let all_same = i == j && j == k;
        let all_distinct = i != j && j != k && i != k;

        let value = if all_same {
            let a = shifted(&[(i, 2)]);
            let b = shifted(&[(i, 1)]);
            let c = shifted(&[(i, -1)]);
            let d = shifted(&[(i, -2)]);
            (a - two * b + two * c - d) / (two * h3)
        } else if all_distinct {
            let mut acc = T::zero();
            for &si in &[1i32, -1] {
                for &sj in &[1i32, -1] {
                    for &sk in &[1i32, -1] {
                        let sign: T = scirs2_core::numeric::NumCast::from(si * sj * sk)
                            .unwrap_or_else(|| T::zero());
                        acc = acc + sign * shifted(&[(i, si), (j, sj), (k, sk)]);
                    }
                }
            }
            acc / (eight * h3)
        } else {
            // Exactly two coincide: `r` is the repeated index, `s` the distinct.
            let (r, s) = if i == j {
                (i, k)
            } else if i == k {
                (i, j)
            } else {
                (j, i)
            };
            let d2_plus =
                shifted(&[(r, 1), (s, 1)]) - two * shifted(&[(s, 1)]) + shifted(&[(r, -1), (s, 1)]);
            let d2_minus = shifted(&[(r, 1), (s, -1)]) - two * shifted(&[(s, -1)])
                + shifted(&[(r, -1), (s, -1)]);
            (d2_plus - d2_minus) / (two * h3)
        };

        Ok(value)
    }

    /// Mixed partial derivative by central finite differences, supporting every
    /// combination with `total_order ≤ 3` (previously only the two-variable
    /// second order was handled and all other cases silently returned `0`).
    ///
    /// `(variables, orders)` is expanded into a flat multi-index; order-1 and
    /// order-2 cases use their own central stencils and order-3 delegates to
    /// [`Self::compute_third_partial`], which picks the right stencil for the
    /// index-coincidence pattern.
    fn mixed_partial_finite_difference(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
    ) -> Result<T> {
        let total_order: usize = orders.iter().sum();
        if total_order > 3 {
            return Err(OptimError::InvalidConfig(
                "Mixed partial order too high (finite differences support order <= 3)".to_string(),
            ));
        }
        let n = point.len();
        for &v in variables {
            if v >= n {
                return Err(OptimError::InvalidConfig(
                    "mixed-partial variable index out of range".to_string(),
                ));
            }
        }

        // Expand into a flat list that repeats each variable `order` times.
        let mut flat: Vec<usize> = Vec::with_capacity(total_order);
        for (&v, &o) in variables.iter().zip(orders.iter()) {
            for _ in 0..o {
                flat.push(v);
            }
        }

        let two: T = scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero());

        match flat.as_slice() {
            [] => Ok(function(point)),
            &[i] => {
                let h = self.fd_step(1);
                let mut xp = point.clone();
                let mut xm = point.clone();
                xp[i] = xp[i] + h;
                xm[i] = xm[i] - h;
                Ok((function(&xp) - function(&xm)) / (two * h))
            }
            &[i, j] => {
                let h = self.fd_step(2);
                if i == j {
                    // ∂²f/∂xᵢ² central second difference.
                    let mut xp = point.clone();
                    let mut xm = point.clone();
                    xp[i] = xp[i] + h;
                    xm[i] = xm[i] - h;
                    Ok((function(&xp) - two * function(point) + function(&xm)) / (h * h))
                } else {
                    // ∂²f/∂xᵢ∂xⱼ four-point stencil.
                    let four: T =
                        scirs2_core::numeric::NumCast::from(4.0).unwrap_or_else(|| T::zero());
                    let mut x_pp = point.clone();
                    x_pp[i] = x_pp[i] + h;
                    x_pp[j] = x_pp[j] + h;
                    let mut x_pm = point.clone();
                    x_pm[i] = x_pm[i] + h;
                    x_pm[j] = x_pm[j] - h;
                    let mut x_mp = point.clone();
                    x_mp[i] = x_mp[i] - h;
                    x_mp[j] = x_mp[j] + h;
                    let mut x_mm = point.clone();
                    x_mm[i] = x_mm[i] - h;
                    x_mm[j] = x_mm[j] - h;
                    Ok(
                        (function(&x_pp) - function(&x_pm) - function(&x_mp) + function(&x_mm))
                            / (four * h * h),
                    )
                }
            }
            &[i, j, k] => self.compute_third_partial(function, point, i, j, k),
            _ => Err(OptimError::InvalidConfig(
                "unsupported mixed-partial structure".to_string(),
            )),
        }
    }

    fn verify_hessian_accuracy(&self, computed: &Array2<T>, reference: &Array2<T>) -> Result<()> {
        let max_error = self.compute_max_error(computed, reference);
        let threshold = scirs2_core::numeric::NumCast::from(1e-3).unwrap_or_else(|| T::zero());

        if max_error > threshold {
            return Err(OptimError::InvalidConfig(format!(
                "Hessian verification failed: max error {}",
                max_error.to_f64().unwrap_or(0.0)
            )));
        }

        Ok(())
    }

    fn compute_max_error(&self, a: &Array2<T>, b: &Array2<T>) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(T::zero(), |acc, x| if x > acc { x } else { acc })
    }

    fn compute_avg_error(&self, a: &Array2<T>, b: &Array2<T>) -> T {
        let sum = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum::<T>();

        let count = scirs2_core::numeric::NumCast::from(a.len()).unwrap_or_else(T::one);
        sum / count
    }

    fn compute_relative_error(&self, a: &Array2<T>, b: &Array2<T>) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                if y.abs() > scirs2_core::numeric::NumCast::from(1e-12).unwrap_or_else(|| T::zero())
                {
                    ((x - y) / y).abs()
                } else {
                    (x - y).abs()
                }
            })
            .fold(T::zero(), |acc, x| if x > acc { x } else { acc })
    }

    /// Rough resident-size estimate: the engine struct itself plus the timing
    /// records the profiler has accumulated.
    fn estimate_memory_usage(&self) -> usize {
        let engine_size = std::mem::size_of::<Self>();
        let profiler_size =
            self.profiler.recorded_timings() * std::mem::size_of::<ComputationTiming>();

        engine_size + profiler_size
    }

    /// Decide whether to use parallel computation
    fn should_use_parallel(&self, problemsize: usize) -> bool {
        self.parallel_computation && problemsize >= 50
    }

    /// Decide whether to use sparse computations
    fn should_use_sparse(&self, problemsize: usize, config: &HessianConfig) -> bool {
        self.adaptive_sparsity && config.sparse && problemsize >= 100
    }

    /// Select an HVP mode from the problem size.
    ///
    /// Only two of the available modes are ever auto-selected, because only two
    /// are unconditionally correct: the central difference (the accurate default)
    /// and, for a problem small enough that `O(n²)` is free, the materialized
    /// Hessian, whose result is reusable across directions. The forward
    /// difference is opt-in (it trades accuracy for one-sidedness) and the
    /// quadratic secant is opt-in (it is only sound on a quadratic model), so
    /// neither may be chosen on the caller's behalf.
    ///
    /// Previously this returned `ForwardOverReverse` / `ReverseOverForward` /
    /// `FiniteDifference` — three names for one identical finite difference — so
    /// the "selection" changed nothing at all.
    fn select_optimal_hvp_mode(&self, problemsize: usize) -> HvpMode {
        if !self.auto_mode_selection {
            return HvpMode::CentralDifference;
        }

        match problemsize {
            0..=8 => HvpMode::MaterializedHessian,
            _ => HvpMode::CentralDifference,
        }
    }

    /// Apply adaptive sparsity to dense matrix
    fn apply_adaptive_sparsity(&self, mut matrix: Array2<T>, threshold: f64) -> Result<Array2<T>> {
        let sparsity_threshold =
            scirs2_core::numeric::NumCast::from(threshold).unwrap_or_else(|| T::zero());

        for elem in matrix.iter_mut() {
            if elem.abs() < sparsity_threshold {
                *elem = T::zero();
            }
        }

        Ok(matrix)
    }

    /// Jacobian **column by column**, one central difference per input.
    ///
    /// Cheaper than [`Self::jacobian_reverse_mode`] when `input_dim <=
    /// output_dim`, which is exactly when [`Self::jacobian_efficient`] picks it.
    /// It used to differ from the row-wise routine in accuracy as well as cost —
    /// a *one-sided* step, so `jacobian_efficient` silently returned a
    /// first-order answer for tall problems and a second-order one for wide
    /// ones. Both are central now, so the choice is purely about cost.
    ///
    /// (It also re-evaluated `function(point)` once per input column; that base
    /// evaluation is gone entirely with the central stencil.)
    fn jacobian_forward_mode<F>(
        &mut self,
        function: &F,
        point: &Array1<T>,
        output_dim: usize,
    ) -> Result<Array2<T>>
    where
        F: Fn(&Array1<T>) -> Array1<T>,
    {
        let input_dim = point.len();
        let mut jacobian = Array2::zeros((output_dim, input_dim));
        let eps = self.fd_step(1);
        let two: T = scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::one());

        for j in 0..input_dim {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            point_plus[j] = point_plus[j] + eps;
            point_minus[j] = point_minus[j] - eps;

            let f_plus = function(&point_plus);
            let f_minus = function(&point_minus);
            if f_plus.len() < output_dim || f_minus.len() < output_dim {
                return Err(OptimError::InvalidConfig(format!(
                    "function returned {} outputs, expected at least {output_dim}",
                    f_plus.len().min(f_minus.len())
                )));
            }

            let column = (f_plus - f_minus) / (two * eps);

            for i in 0..output_dim {
                jacobian[[i, j]] = column[i];
            }
        }

        Ok(jacobian)
    }

    /// Reverse-mode Jacobian computation
    fn jacobian_reverse_mode<F>(
        &mut self,
        function: &F,
        point: &Array1<T>,
        output_dim: usize,
    ) -> Result<Array2<T>>
    where
        F: Fn(&Array1<T>) -> Array1<T>,
    {
        let input_dim = point.len();
        let mut jacobian = Array2::zeros((output_dim, input_dim));

        // Compute Jacobian row by row using reverse mode
        for i in 0..output_dim {
            // Create scalar function for i-th output
            let scalar_fn = |x: &Array1<T>| function(x)[i];

            // Compute gradient of scalar function
            let grad = self.gradient_at_point(&scalar_fn, point)?;

            for j in 0..input_dim {
                jacobian[[i, j]] = grad[j];
            }
        }

        Ok(jacobian)
    }

    /// K-FAC helper methods
    fn compute_activation_factor(&self, activation: &Array1<T>) -> Result<Array2<T>> {
        let n = activation.len();
        let mut factor = Array2::zeros((n, n));

        // A = E[a a^T] where a is the activation
        for i in 0..n {
            for j in 0..n {
                factor[[i, j]] = activation[i] * activation[j];
            }
        }

        Ok(factor)
    }

    fn compute_gradient_factor(&self, gradient: &Array1<T>) -> Result<Array2<T>> {
        let n = gradient.len();
        let mut factor = Array2::zeros((n, n));

        // G = E[g g^T] where g is the gradient
        for i in 0..n {
            for j in 0..n {
                factor[[i, j]] = gradient[i] * gradient[j];
            }
        }

        Ok(factor)
    }

    fn kronecker_product_approximation(&self, a: &Array2<T>, g: &Array2<T>) -> Result<Array2<T>> {
        // Simplified Kronecker product approximation
        // In practice, this would be more sophisticated
        let n_a = a.nrows();
        let n_g = g.nrows();
        let n_total = n_a * n_g;

        let mut result = Array2::zeros((n_total, n_total));

        for i in 0..n_a {
            for j in 0..n_a {
                for k in 0..n_g {
                    for l in 0..n_g {
                        let row = i * n_g + k;
                        let col = j * n_g + l;
                        result[[row, col]] = a[[i, j]] * g[[k, l]];
                    }
                }
            }
        }

        Ok(result)
    }

    fn combine_kfac_blocks(&self, blocks: &[Array2<T>]) -> Result<Array2<T>> {
        if blocks.is_empty() {
            return Err(OptimError::InvalidConfig("Empty K-FAC blocks".to_string()));
        }

        // Simplified block combination - would be more sophisticated in practice
        let total_size: usize = blocks.iter().map(|b| b.nrows()).sum();
        let mut combined = Array2::zeros((total_size, total_size));

        let mut row_offset = 0;
        let mut col_offset = 0;

        for block in blocks {
            let block_size = block.nrows();

            for i in 0..block_size {
                for j in 0..block_size {
                    combined[[row_offset + i, col_offset + j]] = block[[i, j]];
                }
            }

            row_offset += block_size;
            col_offset += block_size;
        }

        Ok(combined)
    }

    fn compute_fisher_information_matrix(
        &mut self,
        log_likelihood: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array2<T>> {
        // Fisher Information Matrix: F = E[∇log p(x) ∇log p(x)^T]
        // Approximated as F ≈ -H[log p(x)] (for exponential family)
        self.hessian_forward_over_reverse(log_likelihood, point, &HessianConfig::default())
    }

    /// Solve `A x = rhs` for a symmetric positive-(semi)definite `A` (e.g. a
    /// Fisher information matrix) using Conjugate Gradient.
    ///
    /// A small Tikhonov ridge proportional to the mean absolute diagonal keeps
    /// the operator strictly positive definite despite finite-difference noise,
    /// which replaces the previous diagonal-only approximation that solved the
    /// system correctly only when `A` was already diagonal.
    fn solve_linear_system(&self, matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>> {
        let n = matrix.nrows();
        if n != rhs.len() || n != matrix.ncols() {
            return Err(OptimError::InvalidConfig(
                "Matrix dimension mismatch".to_string(),
            ));
        }
        if n == 0 {
            return Ok(Array1::zeros(0));
        }

        let mut diag_sum = T::zero();
        for i in 0..n {
            diag_sum = diag_sum + matrix[[i, i]].abs();
        }
        let n_t: T = scirs2_core::numeric::NumCast::from(n).unwrap_or_else(|| T::one());
        let ridge_scale: T = scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero());
        let ridge_floor: T =
            scirs2_core::numeric::NumCast::from(1e-12).unwrap_or_else(|| T::zero());
        let ridge = (diag_sum / n_t) * ridge_scale + ridge_floor;

        let hvp = |v: &Array1<T>| -> Result<Array1<T>> { Ok(matrix.dot(v) + &(v * ridge)) };

        let tolerance: T = scirs2_core::numeric::NumCast::from(1e-10).unwrap_or_else(|| T::zero());
        self.conjugate_gradient_solve(hvp, rhs, 2 * n + 50, tolerance)
    }

    fn conjugate_gradient_solve<F>(
        &self,
        mut hvp_fn: F,
        rhs: &Array1<T>,
        max_iterations: usize,
        tolerance: T,
    ) -> Result<Array1<T>>
    where
        F: FnMut(&Array1<T>) -> Result<Array1<T>>,
    {
        let n = rhs.len();
        let mut x = Array1::zeros(n);
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for _iter in 0..max_iterations {
            let ap = hvp_fn(&p)?;
            let denom = p.dot(&ap);
            // Guard against a breakdown (curvature ~0) that would divide by zero.
            if denom.abs() <= T::epsilon() {
                break;
            }
            let alpha = rsold / denom;

            x = x + &p * alpha;
            r = r - &ap * alpha;

            let rsnew = r.dot(&r);

            if rsnew.sqrt() < tolerance {
                break;
            }

            let beta = rsnew / rsold;
            p = &r + &p * beta;
            rsold = rsnew;
        }

        Ok(x)
    }
}

/// Derivative verification results
#[derive(Debug, Clone)]
pub struct DerivativeVerification<T: Float + Debug + Send + Sync + 'static> {
    pub max_absolute_error: T,
    pub avg_absolute_error: T,
    pub max_relative_error: T,
    pub is_accurate: bool,
    pub finite_diff_hessian: Array2<T>,
}

/// Higher-order differentiation statistics
#[derive(Debug, Clone)]
pub struct HigherOrderStats {
    pub _maxorder: usize,
    pub mixed_mode_enabled: bool,
    pub memory_usage_estimate: usize,
    pub parallel_computation: bool,
    pub thread_pool_size: usize,
    pub adaptive_sparsity: bool,
    pub auto_mode_selection: bool,
    pub performance_profile: PerformanceProfile,
}

/// Advanced configuration for higher-order engine
#[derive(Debug, Clone)]
pub struct HigherOrderConfig<T: Float + Debug + Send + Sync + 'static> {
    pub _maxorder: usize,
    pub mixed_mode: bool,
    pub finite_diff_eps: T,
    pub parallel_computation: bool,
    pub thread_pool_size: usize,
    pub adaptive_sparsity: bool,
    pub auto_mode_selection: bool,
    pub cache_size_limit: usize,
}

impl<T: Float + Debug + Default + Send + Sync> Default for HigherOrderConfig<T> {
    fn default() -> Self {
        Self {
            _maxorder: 3,
            mixed_mode: true,
            finite_diff_eps: scirs2_core::numeric::NumCast::from(1e-5).unwrap_or_else(|| T::zero()),
            parallel_computation: true,
            thread_pool_size: 4, // Conservative default
            adaptive_sparsity: true,
            auto_mode_selection: true,
            cache_size_limit: 10000,
        }
    }
}

/// Hessian-vector product computation modes.
///
/// Every variant below names a genuinely different numerical algorithm with its
/// own cost and truncation error — see [`HvpMode::objective_evaluations`] and the
/// [`hvp`] module docs. The previous four variants
/// (`ForwardOverReverse`/`ReverseOverForward`/`FiniteDifference`/`PearLman`)
/// advertised four differentiation schemes over what were, in the source, two
/// byte-identical bodies plus a third copy behind a helper (finding F89).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvpMode {
    /// `Hv ≈ (∇f(x + h·v) − ∇f(x − h·v)) / 2h`. Second-order accurate; the
    /// accurate default.
    CentralDifference,

    /// `Hv ≈ (∇f(x + h·v) − ∇f(x)) / h`. First-order accurate, and never
    /// evaluates the objective at `x − h·v` — the mode to use at a domain edge.
    ForwardDifference,

    /// `Hv = ∇f(x + v) − ∇f(x)`, algebraically exact for a quadratic objective
    /// and badly biased for anything else. Needs no step-size choice.
    QuadraticSecant,

    /// Materialize the finite-difference Hessian and multiply. `O(n²)` objective
    /// evaluations; the reference path, and the right choice when one `H` will be
    /// applied to many vectors.
    MaterializedHessian,

    /// Nested automatic differentiation. **Not available** behind this engine's
    /// `Fn(&Array1<T>) -> T` objective; the call returns a typed error naming
    /// [`crate::forward_mode`]/[`crate::reverse_mode`] instead of quietly
    /// running a finite difference. [`HvpMode::is_available`] reports this.
    NestedAutodiff,
}

/// Computation profiler for performance optimization
#[derive(Debug, Clone)]
pub struct ComputationProfiler<T: Float + Debug + Send + Sync + 'static> {
    hessian_timings: Vec<ComputationTiming>,
    hvp_timings: Vec<ComputationTiming>,
    jacobian_timings: Vec<ComputationTiming>,
    total_computations: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// Method tag for a recorded [`ComputationTiming`].
const HESSIAN_METHOD: &str = "hessian";
/// Method tag for a recorded Hessian-vector-product timing.
const HVP_METHOD: &str = "hvp";
/// Method tag for a recorded Jacobian timing.
const JACOBIAN_METHOD: &str = "jacobian";

#[derive(Debug, Clone)]
struct ComputationTiming {
    problemsize: usize,
    duration_us: u64,
    parallel: bool,
    sparse: bool,
    method: &'static str,
}

/// Measured performance of a [`HigherOrderEngine`], from
/// [`HigherOrderEngine::get_derivative_stats`].
///
/// Every field is derived from timings recorded by the engine's own entry
/// points. Before those entry points recorded anything, `avg_hvp_time_us`,
/// `avg_jacobian_time_us` and `cache_hit_rate` were hard-coded zeros.
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Mean wall time of a full Hessian computation, in microseconds.
    pub avg_hessian_time_us: f64,
    /// Mean wall time of a Hessian-vector product, in microseconds.
    pub avg_hvp_time_us: f64,
    /// Mean wall time of a Jacobian computation, in microseconds.
    pub avg_jacobian_time_us: f64,
    /// Mean sequential time / mean parallel time over recorded Hessians; `1.0`
    /// when only one of the two regimes has been observed.
    pub parallel_efficiency: f64,
    /// Mean dense time / mean sparsified time over recorded Hessians; `1.0`
    /// when only one of the two regimes has been observed.
    pub sparsity_benefit: f64,
    /// Number of timings recorded across all three kinds.
    pub recorded_computations: usize,
    /// Largest problem dimension seen in a recorded Hessian computation.
    pub largest_hessian_problem: usize,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> ComputationProfiler<T> {
    fn new() -> Self {
        Self {
            hessian_timings: Vec::new(),
            hvp_timings: Vec::new(),
            jacobian_timings: Vec::new(),
            total_computations: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    fn record_hessian_computation(
        &mut self,
        size: usize,
        duration: std::time::Duration,
        parallel: bool,
        sparse: bool,
    ) {
        self.hessian_timings.push(ComputationTiming {
            problemsize: size,
            duration_us: duration.as_micros() as u64,
            parallel,
            sparse,
            method: HESSIAN_METHOD,
        });
        self.total_computations += 1;
    }

    fn record_hvp_computation(&mut self, size: usize, duration: std::time::Duration) {
        self.hvp_timings.push(ComputationTiming {
            problemsize: size,
            duration_us: duration.as_micros() as u64,
            parallel: false,
            sparse: false,
            method: HVP_METHOD,
        });
        self.total_computations += 1;
    }

    fn record_jacobian_computation(&mut self, size: usize, duration: std::time::Duration) {
        self.jacobian_timings.push(ComputationTiming {
            problemsize: size,
            duration_us: duration.as_micros() as u64,
            parallel: false,
            sparse: false,
            method: JACOBIAN_METHOD,
        });
        self.total_computations += 1;
    }

    /// Total recorded computations across all three kinds.
    fn total_computations(&self) -> usize {
        self.total_computations
    }

    /// Number of individual timing records held.
    fn recorded_timings(&self) -> usize {
        self.hessian_timings.len() + self.hvp_timings.len() + self.jacobian_timings.len()
    }

    /// Mean recorded duration, or `0.0` when nothing has been recorded.
    fn mean_duration_us(timings: &[ComputationTiming]) -> f64 {
        if timings.is_empty() {
            return 0.0;
        }
        timings.iter().map(|t| t.duration_us as f64).sum::<f64>() / timings.len() as f64
    }

    /// The largest problem size seen for `method`, or `0` when none.
    fn largest_problem(&self, method: &'static str) -> usize {
        self.hessian_timings
            .iter()
            .chain(self.hvp_timings.iter())
            .chain(self.jacobian_timings.iter())
            .filter(|t| t.method == method)
            .map(|t| t.problemsize)
            .max()
            .unwrap_or(0)
    }

    fn get_summary(&self) -> PerformanceProfile {
        PerformanceProfile {
            avg_hessian_time_us: Self::mean_duration_us(&self.hessian_timings),
            avg_hvp_time_us: Self::mean_duration_us(&self.hvp_timings),
            avg_jacobian_time_us: Self::mean_duration_us(&self.jacobian_timings),
            parallel_efficiency: self.compute_parallel_efficiency(),
            sparsity_benefit: self.compute_sparsity_benefit(),
            recorded_computations: self.total_computations(),
            largest_hessian_problem: self.largest_problem(HESSIAN_METHOD),
        }
    }

    fn compute_parallel_efficiency(&self) -> f64 {
        if self.hessian_timings.is_empty() {
            return 1.0;
        }

        let parallel_times: Vec<_> = self.hessian_timings.iter().filter(|t| t.parallel).collect();
        let sequential_times: Vec<_> = self
            .hessian_timings
            .iter()
            .filter(|t| !t.parallel)
            .collect();

        if parallel_times.is_empty() || sequential_times.is_empty() {
            return 1.0;
        }

        let avg_parallel = parallel_times
            .iter()
            .map(|t| t.duration_us as f64)
            .sum::<f64>()
            / parallel_times.len() as f64;
        let avg_sequential = sequential_times
            .iter()
            .map(|t| t.duration_us as f64)
            .sum::<f64>()
            / sequential_times.len() as f64;

        avg_sequential / avg_parallel
    }

    /// Mean dense time divided by mean sparsified time over recorded Hessians.
    ///
    /// Symmetric with [`Self::compute_parallel_efficiency`], and `1.0` until both
    /// regimes have actually been observed. This used to return a hard-coded
    /// `1.2` — a made-up speedup reported through the public
    /// [`PerformanceProfile`] as if it had been measured.
    fn compute_sparsity_benefit(&self) -> f64 {
        let sparse = Self::mean_duration_us(
            &self
                .hessian_timings
                .iter()
                .filter(|t| t.sparse)
                .cloned()
                .collect::<Vec<_>>(),
        );
        let dense = Self::mean_duration_us(
            &self
                .hessian_timings
                .iter()
                .filter(|t| !t.sparse)
                .cloned()
                .collect::<Vec<_>>(),
        );
        if sparse <= 0.0 || dense <= 0.0 {
            return 1.0;
        }
        dense / sparse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_higher_order_engine_creation() {
        let engine = HigherOrderEngine::<f64>::new(3);
        assert_eq!(engine._maxorder, 3);
        assert!(engine.mixed_mode);
    }

    #[test]
    fn test_hessian_diagonal() {
        let engine = HigherOrderEngine::<f64>::new(2);

        // Test function: f(x) = x₁² + 2x₂²
        let function = |x: &Array1<f64>| x[0] * x[0] + 2.0 * x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let hessian = engine
            .hessian_diagonal(&function, &point)
            .expect("hessian_diagonal should succeed");

        // Expected diagonal: [2, 4]
        assert!((hessian[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((hessian[[1, 1]] - 4.0).abs() < 1e-5);
        assert!((hessian[[0, 1]]).abs() < 1e-10); // Off-diagonal should be zero
    }

    #[test]
    fn test_finite_difference_hessian() {
        let engine = HigherOrderEngine::<f64>::new(2);

        // Test function: f(x,y) = x² + xy + y²
        let function = |x: &Array1<f64>| x[0] * x[0] + x[0] * x[1] + x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let hessian = engine
            .finite_difference_hessian(&function, &point)
            .expect("finite_difference_hessian should succeed");

        // Expected Hessian: [[2, 1], [1, 2]]
        assert!((hessian[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((hessian[[0, 1]] - 1.0).abs() < 1e-5);
        assert!((hessian[[1, 0]] - 1.0).abs() < 1e-5);
        assert!((hessian[[1, 1]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_mixed_partial() {
        let mut engine = HigherOrderEngine::<f64>::new(2);

        // Test function: f(x,y) = x²y + xy²
        let function = |x: &Array1<f64>| x[0] * x[0] * x[1] + x[0] * x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let mixed_partial = engine
            .mixed_partial(
                function,
                &point,
                &[0, 1],
                &[1, 1],
                MixedPartialMethod::FiniteDifference,
            )
            .expect("mixed_partial should succeed");

        // ∂²f/∂x∂y = 2x + 2y = 4 at (1,1)
        assert!((mixed_partial.value - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_hessian() {
        let mut engine = HigherOrderEngine::<f64>::new(2);

        // Diagonal function: f(x) = x₁² + x₂²
        let function = |x: &Array1<f64>| x[0] * x[0] + x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let config = HessianConfig {
            sparse: true,
            sparsity_threshold: 1e-4,
            ..Default::default()
        };

        let sparse_hessian = engine
            .sparse_hessian(function, &point, &config)
            .expect("sparse_hessian should succeed");

        // Should have 2 non-zero elements (diagonal)
        assert_eq!(sparse_hessian.nnz, 2);
        assert_eq!(sparse_hessian.shape, (2, 2));
    }

    #[test]
    fn test_hessian_config_default() {
        let config = HessianConfig::default();
        assert!(config.exact);
        assert!(!config.sparse);
        assert!(!config.diagonal_only);
    }

    /// Smoke test: higher-order AD computes the correct second derivative.
    ///
    /// For f(x) = x^2, f''(x) = 2 everywhere, so the (1x1) Hessian is [[2]].
    #[test]
    fn test_higher_order_hessian_of_square() {
        let mut engine = HigherOrderEngine::<f64>::new(2);

        let function = |x: &Array1<f64>| x[0] * x[0];
        let point = Array1::from_vec(vec![3.0]);

        let hessian = engine
            .hessian_forward_over_reverse(function, &point, &HessianConfig::default())
            .expect("hessian");

        approx::assert_abs_diff_eq!(hessian[[0, 0]], 2.0, epsilon = 1e-3);
    }

    /// Smoke test: Hessian-vector product on a separable quadratic.
    ///
    /// For f(x) = 0.5 * (x_0^2 + x_1^2) the Hessian is the identity, so the
    /// Hessian-vector product H * v must return v unchanged.
    #[test]
    fn test_higher_order_hvp_quadratic() {
        let mut engine = HigherOrderEngine::<f64>::new(2);

        let function = |x: &Array1<f64>| 0.5 * (x[0] * x[0] + x[1] * x[1]);
        let point = Array1::from_vec(vec![1.0, -2.0]);
        let vector = Array1::from_vec(vec![3.0, 4.0]);

        let hv = engine
            .hessian_vector_product(function, &point, &vector)
            .expect("hvp");

        approx::assert_abs_diff_eq!(hv[0], 3.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(hv[1], 4.0, epsilon = 1e-3);
    }

    /// F29 regression: third derivatives must be numerically correct (the old
    /// two-point formula was wrong by a factor of ~1/h², i.e. ~1e10 for f64).
    ///
    /// f(x) = x0³ + x0·x1·x2 + x0²·x1 exercises the pure, fully-mixed and
    /// semi-mixed stencils simultaneously:
    /// * ∂³/∂x0³        = 6
    /// * ∂³/∂x0∂x1∂x2   = 1
    /// * ∂³/∂x0²∂x1      = 2
    #[test]
    fn test_third_order_derivatives_are_accurate() {
        let mut engine = HigherOrderEngine::<f64>::new(3);
        let function =
            |x: &Array1<f64>| x[0] * x[0] * x[0] + x[0] * x[1] * x[2] + x[0] * x[0] * x[1];
        let point = Array1::from_vec(vec![0.7, -0.4, 1.3]);

        let tensor = engine
            .third_order_derivatives(function, &point)
            .expect("third order");

        approx::assert_abs_diff_eq!(tensor.data[[0, 0, 0]], 6.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(tensor.data[[0, 1, 2]], 1.0, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(tensor.data[[0, 0, 1]], 2.0, epsilon = 1e-3);
        // Symmetry of the mixed third derivative.
        approx::assert_abs_diff_eq!(tensor.data[[2, 1, 0]], 1.0, epsilon = 1e-3);
    }

    /// F90 regression: mixed partials of order 1 and 3 used to silently return
    /// 0 for everything but the two-variable second order.
    #[test]
    fn test_mixed_partial_supports_orders_one_and_three() {
        let mut engine = HigherOrderEngine::<f64>::new(3);
        let point = Array1::from_vec(vec![0.5, -0.3, 0.9]);

        // Order 1: ∂/∂x0 (3x0 + x1²) = 3.
        let g = |x: &Array1<f64>| 3.0 * x[0] + x[1] * x[1];
        let d1 = engine
            .mixed_partial(g, &point, &[0], &[1], MixedPartialMethod::FiniteDifference)
            .expect("order 1");
        approx::assert_abs_diff_eq!(d1.value, 3.0, epsilon = 1e-6);

        // Order 3 (pure): ∂³/∂x0³ x0³ = 6.
        let cube = |x: &Array1<f64>| x[0] * x[0] * x[0];
        let d3 = engine
            .mixed_partial(
                cube,
                &point,
                &[0],
                &[3],
                MixedPartialMethod::FiniteDifference,
            )
            .expect("order 3 pure");
        approx::assert_abs_diff_eq!(d3.value, 6.0, epsilon = 1e-3);

        // Order 3 (fully mixed): ∂³/∂x0∂x1∂x2 (x0·x1·x2) = 1.
        let prod = |x: &Array1<f64>| x[0] * x[1] * x[2];
        let d3m = engine
            .mixed_partial(
                prod,
                &point,
                &[0, 1, 2],
                &[1, 1, 1],
                MixedPartialMethod::FiniteDifference,
            )
            .expect("order 3 mixed");
        approx::assert_abs_diff_eq!(d3m.value, 1.0, epsilon = 1e-3);
    }

    /// F30 regression: `solve_linear_system` must solve a genuinely dense SPD
    /// system, not merely divide by the diagonal.
    #[test]
    fn test_solve_linear_system_dense_spd() {
        let engine = HigherOrderEngine::<f64>::new(2);
        // A = [[4, 1], [1, 3]] (SPD, non-diagonal); b = [1, 2].
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).expect("matrix");
        let b = Array1::from_vec(vec![1.0, 2.0]);

        let x = engine.solve_linear_system(&a, &b).expect("solve");

        // Exact solution A^{-1} b = [1/11, 7/11].
        approx::assert_abs_diff_eq!(x[0], 1.0 / 11.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(x[1], 7.0 / 11.0, epsilon = 1e-6);

        // Residual A x - b must be ~0 (a diagonal solve would leave it large).
        let residual = a.dot(&x) - &b;
        assert!(residual.dot(&residual).sqrt() < 1e-6);
    }
}
