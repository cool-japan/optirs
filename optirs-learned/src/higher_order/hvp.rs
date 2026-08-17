//! Hessian-vector product and mixed-partial back ends for [`HigherOrderEngine`].
//!
//! # Why these are finite differences, and why that is not a stub
//!
//! [`HigherOrderEngine`] consumes an objective as a black-box
//! `impl Fn(&Array1<T>) -> T` that is *monomorphic in `T`*. Nested automatic
//! differentiation (forward-over-reverse, reverse-over-forward) fundamentally
//! cannot be threaded through that type:
//!
//! * forward mode needs the objective to accept dual numbers, i.e. to be
//!   generic (`Fn(&Array1<D>) -> D` for any dual `D`), and
//! * reverse mode needs the objective expressed as tape operations
//!   ([`crate::reverse_mode::ReverseModeEngine`]), not as an opaque closure.
//!
//! `optirs-learned` has real implementations of both
//! ([`crate::forward_mode`], [`crate::reverse_mode`]) — they simply require a
//! different entry point than a `Fn(&Array1<T>) -> T`.
//!
//! So this module does two honest things instead of one dishonest one. Every
//! mode that *can* exist behind this signature is implemented as a genuinely
//! distinct numerical algorithm with its own cost and truncation error, and
//! [`HvpMode::NestedAutodiff`] / [`MixedPartialMethod::NestedAutodiff`] return a
//! typed error naming the reason and the alternative, rather than silently
//! running a finite difference under an autodiff name.
//!
//! Historically (finding F89) `hvp_reverse_over_forward` and
//! `hvp_finite_difference` were byte-identical, `hvp_forward_over_reverse` was a
//! third copy behind a helper, and all three of
//! `mixed_partial_{forward_over_reverse, reverse_over_forward, pure_forward}`
//! contained a one-line delegation to the finite-difference routine with the
//! comment "Real implementation would use proper forward-over-reverse mode" —
//! five advertised differentiation modes for one algorithm.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{central_step, HigherOrderEngine, HvpMode, MixedPartialMethod};
use crate::error::{OptimError, Result};

/// Shared message for the modes that the black-box closure signature rules out.
pub(super) const NESTED_AUTODIFF_UNAVAILABLE: &str =
    "nested automatic differentiation (forward-over-reverse / reverse-over-forward) is not \
     available through HigherOrderEngine: its objective is an opaque Fn(&Array1<T>) -> T that is \
     monomorphic in T, so neither dual numbers nor a reverse tape can be threaded through it. Use \
     crate::forward_mode::ForwardModeEngine or crate::reverse_mode::ReverseModeEngine directly, or \
     pick one of the finite-difference HvpMode variants.";

impl<
        T: Float
            + Debug
            + Default
            + std::iter::Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    > HigherOrderEngine<T>
{
    /// `Hv` by a **central** difference of the gradient along `v`:
    /// `Hv ≈ (∇f(x + h·v) − ∇f(x − h·v)) / (2h)`.
    ///
    /// Cost: two gradient evaluations (`2n` extra objective evaluations for the
    /// inner central-difference gradient, so `4n` objective calls). Truncation
    /// error `O(h²)`; `h` comes from [`HigherOrderEngine::fd_step`] with order 2
    /// because differencing a gradient is a second-order quantity — using the
    /// order-1 step here is what made the old code numerically useless in `f32`.
    pub(super) fn hvp_central_difference(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        Self::check_hvp_dims(point, vector)?;
        let h = self.hvp_step(vector);
        let two = T::one() + T::one();

        let grad_plus = self.gradient_at_point(function, &(point + &(vector * h)))?;
        let grad_minus = self.gradient_at_point(function, &(point - &(vector * h)))?;

        Ok((grad_plus - grad_minus) / (two * h))
    }

    /// `Hv` by a **one-sided forward** difference of the gradient along `v`:
    /// `Hv ≈ (∇f(x + h·v) − ∇f(x)) / h`.
    ///
    /// Genuinely different from [`Self::hvp_central_difference`]: it never
    /// evaluates the objective at `x − h·v`, which matters when the objective is
    /// undefined or discontinuous on that side (a barrier term, a domain edge, a
    /// simulator that rejects negative inputs). It is first-order accurate, so
    /// it uses the *order-1* step — the optimum for an `O(h)` stencil is coarser
    /// than for an `O(h²)` one.
    pub(super) fn hvp_forward_difference(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        Self::check_hvp_dims(point, vector)?;
        let scale = Self::direction_scale(vector);
        let h = self.fd_step(1) / scale;

        let grad_plus = self.gradient_at_point(function, &(point + &(vector * h)))?;
        let grad_at = self.gradient_at_point(function, point)?;

        Ok((grad_plus - grad_at) / h)
    }

    /// `Hv` by the **unit-step secant** `∇f(x + v) − ∇f(x)`.
    ///
    /// For a quadratic `f(x) = ½xᵀHx + bᵀx + c` this identity is algebraically
    /// exact for *any* step, so no step-size tuning and no truncation error are
    /// involved — the only error left is whatever the inner gradient carries.
    /// (This is the specialisation of Pearlmutter's R-operator identity that a
    /// quadratic model admits; the enum variant used to be spelled `PearLman`
    /// and claimed exactness without noting that the inner gradient is itself a
    /// finite difference.)
    ///
    /// On a non-quadratic objective the `O(‖v‖)` truncation error is *not*
    /// small, which is why this is a separate, explicitly-named mode rather than
    /// the default.
    pub(super) fn hvp_quadratic_secant(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        Self::check_hvp_dims(point, vector)?;
        let grad_plus = self.gradient_at_point(function, &(point + vector))?;
        let grad_at = self.gradient_at_point(function, point)?;
        Ok(grad_plus - grad_at)
    }

    /// `Hv` by materialising the full finite-difference Hessian and multiplying.
    ///
    /// `O(n²)` objective evaluations, so it is the wrong choice for anything
    /// large — it exists as the independent reference path the cheap directional
    /// modes are validated against, and as the mode to reach for when the same
    /// `H` will be applied to many different vectors.
    pub(super) fn hvp_materialized_hessian(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        Self::check_hvp_dims(point, vector)?;
        let hessian = self.finite_difference_hessian(function, point)?;
        Ok(hessian.dot(vector))
    }

    /// Reject a direction whose length does not match the point.
    ///
    /// The old code fed mismatched shapes straight into `point + &(vector * eps)`
    /// and let `ndarray` panic on the broadcast.
    fn check_hvp_dims(point: &Array1<T>, vector: &Array1<T>) -> Result<()> {
        if point.len() != vector.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Hessian-vector product needs matching lengths: point has {}, direction has {}",
                point.len(),
                vector.len()
            )));
        }
        if point.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Hessian-vector product needs a non-empty point".to_string(),
            ));
        }
        Ok(())
    }

    /// `max(‖v‖∞, 1)`, used to keep the *displacement* `h·v` at the intended
    /// scale instead of letting a large direction blow the stencil out.
    fn direction_scale(vector: &Array1<T>) -> T {
        let norm = vector.iter().fold(T::zero(), |acc, v| {
            let a = v.abs();
            if a > acc {
                a
            } else {
                acc
            }
        });
        if norm > T::one() {
            norm
        } else {
            T::one()
        }
    }

    /// Second-order step for a directional gradient difference, normalised by
    /// the direction magnitude.
    fn hvp_step(&self, vector: &Array1<T>) -> T {
        self.fd_step(2) / Self::direction_scale(vector)
    }

    /// Mixed partial by **one-sided forward** central-free stencils.
    ///
    /// Every evaluation point is `x + (non-negative multiple of h)`, so the
    /// objective is never probed below `x` in any differentiated coordinate.
    /// Supports total order ≤ 3 like the central variant, but with `O(h)`
    /// truncation error instead of `O(h²)`.
    ///
    /// The operator is the exact `k`-fold forward difference, assembled by
    /// inclusion-exclusion over the differentiated axes:
    /// `∂f ≈ (f(x+h) − f(x))/h`,
    /// `∂²f ≈ (f(x+hᵢ+hⱼ) − f(x+hᵢ) − f(x+hⱼ) + f(x))/h²` (which covers `i == j`
    /// as `(f(x+2h) − 2f(x+h) + f(x))/h²`), and the order-3 analogue.
    ///
    /// The step is taken from [`HigherOrderEngine::fd_step`] at the *total
    /// order*, not at order 1: a `k`-th difference divides by `hᵏ`, so a step
    /// chosen for a first derivative amplifies roundoff by `h^(1−k)` — at
    /// `h = 1e-5` an order-3 stencil in `f64` is pure noise.
    pub(super) fn mixed_partial_forward_difference(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
    ) -> Result<T> {
        let flat = Self::flatten_mixed_index(point.len(), variables, orders)?;
        let h = self.fd_step(flat.len().max(1));

        // Shift `point` by `h` in each listed coordinate (repeats accumulate).
        let shift = |axes: &[usize]| -> Array1<T> {
            let mut x = point.clone();
            for &axis in axes {
                x[axis] = x[axis] + h;
            }
            x
        };

        // Forward difference operator: sum over every subset of `flat` with the
        // inclusion-exclusion sign, divided by h^order. For order k this is the
        // exact k-fold application of the one-sided operator.
        let order = flat.len();
        if order == 0 {
            return Ok(function(point));
        }
        let mut acc = T::zero();
        let subsets = 1usize << order;
        for mask in 0..subsets {
            let mut axes: Vec<usize> = Vec::with_capacity(order);
            for (bit, &axis) in flat.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    axes.push(axis);
                }
            }
            // (-1)^(order - |subset|)
            let sign = if (order - axes.len()).is_multiple_of(2) {
                T::one()
            } else {
                -T::one()
            };
            acc = acc + sign * function(&shift(&axes));
        }

        let mut denom = T::one();
        for _ in 0..order {
            denom = denom * h;
        }
        Ok(acc / denom)
    }

    /// Validate `(variables, orders)` and expand it into a flat multi-index that
    /// repeats each variable `order` times.
    pub(super) fn flatten_mixed_index(
        dimension: usize,
        variables: &[usize],
        orders: &[usize],
    ) -> Result<Vec<usize>> {
        let total_order: usize = orders.iter().sum();
        if total_order > 3 {
            return Err(OptimError::InvalidConfig(
                "Mixed partial order too high (finite differences support order <= 3)".to_string(),
            ));
        }
        for &v in variables {
            if v >= dimension {
                return Err(OptimError::InvalidConfig(format!(
                    "mixed-partial variable index {v} is out of range for a {dimension}-dimensional point"
                )));
            }
        }
        let mut flat: Vec<usize> = Vec::with_capacity(total_order);
        for (&v, &o) in variables.iter().zip(orders.iter()) {
            for _ in 0..o {
                flat.push(v);
            }
        }
        Ok(flat)
    }
}

impl HvpMode {
    /// Human-readable count of objective evaluations, for the docs and for
    /// callers choosing a mode. `n` is the problem dimension.
    ///
    /// This is the property that makes the modes genuinely different rather than
    /// five names for one algorithm.
    pub fn objective_evaluations(&self, n: usize) -> Option<usize> {
        match self {
            // Two central-difference gradients: 2 · 2n.
            HvpMode::CentralDifference => Some(4 * n),
            // Two central-difference gradients as well, but one of them is at
            // the base point and is the one a caller typically already has.
            HvpMode::ForwardDifference => Some(4 * n),
            HvpMode::QuadraticSecant => Some(4 * n),
            // Full Hessian: the diagonal costs 3n, the off-diagonals 4 each.
            HvpMode::MaterializedHessian => Some(3 * n + 2 * n * n.saturating_sub(1)),
            HvpMode::NestedAutodiff => None,
        }
    }

    /// Whether this mode can actually run behind the engine's black-box
    /// objective type.
    pub fn is_available(&self) -> bool {
        !matches!(self, HvpMode::NestedAutodiff)
    }
}

impl MixedPartialMethod {
    /// Whether this method can actually run behind the engine's black-box
    /// objective type.
    pub fn is_available(&self) -> bool {
        !matches!(self, MixedPartialMethod::NestedAutodiff)
    }
}

/// Re-exported so callers can reason about step sizes without reaching into the
/// private helper.
pub fn recommended_step<T: Float + 'static>(order: usize) -> T {
    central_step::<T>(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::higher_order::{HigherOrderConfig, HigherOrderEngine};
    use scirs2_core::ndarray::{arr1, arr2};

    /// `f(x) = ½ xᵀ A x + bᵀ x` with `A = [[4, 1], [1, 3]]`, `b = [1, -2]`.
    /// The Hessian is exactly `A`, independent of `x`.
    fn quadratic(x: &Array1<f64>) -> f64 {
        let a = arr2(&[[4.0, 1.0], [1.0, 3.0]]);
        let b = arr1(&[1.0, -2.0]);
        0.5 * x.dot(&a.dot(x)) + b.dot(x)
    }

    /// A genuinely non-quadratic objective: `f(x) = x0³ + x0·x1² + sin(x1)`.
    fn cubic(x: &Array1<f64>) -> f64 {
        x[0] * x[0] * x[0] + x[0] * x[1] * x[1] + x[1].sin()
    }

    fn engine() -> HigherOrderEngine<f64> {
        HigherOrderEngine::with_config(HigherOrderConfig::default())
    }

    fn hessian_of_quadratic() -> Array2<f64> {
        arr2(&[[4.0, 1.0], [1.0, 3.0]])
    }

    #[test]
    fn every_available_mode_matches_the_exact_quadratic_hessian_product() {
        let point = arr1(&[0.3, -0.7]);
        let vector = arr1(&[1.0, 2.0]);
        let expected = hessian_of_quadratic().dot(&vector);

        for mode in [
            HvpMode::CentralDifference,
            HvpMode::ForwardDifference,
            HvpMode::QuadraticSecant,
            HvpMode::MaterializedHessian,
        ] {
            let mut eng = engine();
            let hv = eng
                .hessian_vector_product_advanced(quadratic, &point, &vector, Some(mode))
                .unwrap_or_else(|e| panic!("mode {mode:?} failed: {e}"));
            for (i, (&got, &want)) in hv.iter().zip(expected.iter()).enumerate() {
                // 1e-3 absolute on a value of magnitude ~6-7 is a relative
                // accuracy of ~1.5e-4. Nested finite differences cannot do
                // better than ~eps^(1/3) in the worst mode, and the point of the
                // assertion is that every mode really computes `A·v = (6, 7)`.
                assert!(
                    (got - want).abs() < 1e-3,
                    "mode {mode:?} component {i}: got {got}, want {want}"
                );
            }
        }
    }

    /// The headline F89 assertion. On a *non-quadratic* objective the four
    /// available modes have different truncation errors, so they must not agree
    /// bit-for-bit. Two byte-identical implementations (which is what
    /// `ReverseOverForward` and `FiniteDifference` used to be) would fail this.
    #[test]
    fn the_modes_are_genuinely_different_algorithms() {
        let point = arr1(&[0.8, 1.3]);
        let vector = arr1(&[1.0, -0.5]);

        let mut values = Vec::new();
        for mode in [
            HvpMode::CentralDifference,
            HvpMode::ForwardDifference,
            HvpMode::QuadraticSecant,
            HvpMode::MaterializedHessian,
        ] {
            let mut eng = engine();
            let hv = eng
                .hessian_vector_product_advanced(cubic, &point, &vector, Some(mode))
                .unwrap_or_else(|e| panic!("mode {mode:?} failed: {e}"));
            values.push((mode, hv));
        }

        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                let (mode_a, ref a) = values[i];
                let (mode_b, ref b) = values[j];
                let delta = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    delta > 0.0,
                    "{mode_a:?} and {mode_b:?} produced identical output {a:?} — \
                     they are the same implementation under two names"
                );
            }
        }
    }

    /// On a non-quadratic objective the unit-step secant is *supposed* to be
    /// badly biased, and the central difference is supposed to be the accurate
    /// one. Asserting the ordering proves the modes are not interchangeable.
    #[test]
    fn central_difference_beats_the_quadratic_secant_off_quadratics() {
        let point = arr1(&[0.8, 1.3]);
        let vector = arr1(&[1.0, -0.5]);
        // Hessian of x0³ + x0 x1² + sin(x1):
        //   d²/dx0² = 6 x0,     d²/dx0 dx1 = 2 x1,
        //   d²/dx1² = 2 x0 - sin(x1)
        let h = arr2(&[
            [6.0 * point[0], 2.0 * point[1]],
            [2.0 * point[1], 2.0 * point[0] - point[1].sin()],
        ]);
        let exact = h.dot(&vector);

        let err = |mode: HvpMode| -> f64 {
            let mut eng = engine();
            let hv = eng
                .hessian_vector_product_advanced(cubic, &point, &vector, Some(mode))
                .expect("hvp");
            hv.iter()
                .zip(exact.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };

        let central = err(HvpMode::CentralDifference);
        let secant = err(HvpMode::QuadraticSecant);
        assert!(
            central < secant,
            "central error {central} should be below the unit-step secant error {secant}"
        );
        assert!(
            central < 1e-4,
            "central difference error {central} too large"
        );
    }

    #[test]
    fn nested_autodiff_reports_an_honest_error() {
        let mut eng = engine();
        let err = eng
            .hessian_vector_product_advanced(
                quadratic,
                &arr1(&[0.0, 0.0]),
                &arr1(&[1.0, 0.0]),
                Some(HvpMode::NestedAutodiff),
            )
            .expect_err("nested autodiff cannot exist behind a monomorphic closure");
        let text = err.to_string();
        assert!(
            text.contains("nested automatic differentiation"),
            "error should name the unavailable capability, got: {text}"
        );
        assert!(
            text.contains("reverse_mode") || text.contains("forward_mode"),
            "error should point at the real engines, got: {text}"
        );
        assert!(!HvpMode::NestedAutodiff.is_available());
        assert!(HvpMode::CentralDifference.is_available());
        assert!(HvpMode::NestedAutodiff.objective_evaluations(10).is_none());
    }

    #[test]
    fn mismatched_direction_length_is_an_error_not_a_panic() {
        let mut eng = engine();
        assert!(eng
            .hessian_vector_product_advanced(
                quadratic,
                &arr1(&[0.0, 0.0]),
                &arr1(&[1.0, 0.0, 0.0]),
                Some(HvpMode::CentralDifference),
            )
            .is_err());
        assert!(eng
            .hessian_vector_product_advanced(
                quadratic,
                &Array1::<f64>::zeros(0),
                &Array1::<f64>::zeros(0),
                Some(HvpMode::CentralDifference),
            )
            .is_err());
    }

    /// The cost model has to actually distinguish the modes, otherwise it is
    /// another place a "five names, one algorithm" regression could hide.
    #[test]
    fn the_cost_model_separates_directional_modes_from_the_materialized_one() {
        let n = 64;
        let directional = HvpMode::CentralDifference
            .objective_evaluations(n)
            .expect("available");
        let materialized = HvpMode::MaterializedHessian
            .objective_evaluations(n)
            .expect("available");
        assert!(
            materialized > 10 * directional,
            "materializing the Hessian ({materialized}) must be far costlier than a \
             directional difference ({directional})"
        );
    }

    /// Forward-only mixed partials must never probe below the base point. The
    /// objective here returns NaN for any negative coordinate, so a stencil that
    /// steps backwards produces NaN and the assertion fails.
    #[test]
    fn forward_mixed_partials_never_step_backwards() {
        let boundary = |x: &Array1<f64>| -> f64 {
            if x.iter().any(|&v| v < 0.0) {
                f64::NAN
            } else {
                x[0] * x[0] * x[1]
            }
        };
        let eng = engine();
        // At x = (0, 2): d²/dx0² of x0² x1 is 2 x1 = 4.
        let value = eng
            .mixed_partial_forward_difference(&boundary, &arr1(&[0.0, 2.0]), &[0], &[2])
            .expect("forward mixed partial");
        assert!(
            value.is_finite(),
            "forward stencil stepped into the undefined region"
        );
        assert!(
            (value - 4.0).abs() < 1e-3,
            "expected 4.0 from the forward stencil, got {value}"
        );
    }

    #[test]
    fn forward_and_central_mixed_partials_agree_on_a_smooth_function() {
        let mut eng = engine();
        let point = arr1(&[1.1, -0.4]);
        // d²/dx0 dx1 of x0³ + x0 x1² + sin(x1) is 2 x1 = -0.8.
        let central = eng
            .mixed_partial(
                cubic,
                &point,
                &[0, 1],
                &[1, 1],
                MixedPartialMethod::FiniteDifference,
            )
            .expect("central");
        let forward = eng
            .mixed_partial(
                cubic,
                &point,
                &[0, 1],
                &[1, 1],
                MixedPartialMethod::ForwardFiniteDifference,
            )
            .expect("forward");
        assert!((central.value - (-0.8)).abs() < 1e-4, "{}", central.value);
        assert!((forward.value - (-0.8)).abs() < 1e-3, "{}", forward.value);
        // Different stencils: they must not be the same number.
        assert!(
            (central.value - forward.value).abs() > 0.0,
            "central and forward stencils returned identical values"
        );
    }

    #[test]
    fn mixed_partial_nested_autodiff_reports_an_honest_error() {
        let mut eng = engine();
        let err = eng
            .mixed_partial(
                cubic,
                &arr1(&[0.5, 0.5]),
                &[0, 1],
                &[1, 1],
                MixedPartialMethod::NestedAutodiff,
            )
            .expect_err("nested autodiff is unavailable");
        assert!(err.to_string().contains("nested automatic differentiation"));
        assert!(!MixedPartialMethod::NestedAutodiff.is_available());
        assert!(MixedPartialMethod::FiniteDifference.is_available());
    }

    #[test]
    fn mixed_partial_rejects_out_of_range_variables() {
        assert!(HigherOrderEngine::<f64>::flatten_mixed_index(2, &[5], &[1]).is_err());
        assert!(HigherOrderEngine::<f64>::flatten_mixed_index(2, &[0], &[4]).is_err());
        assert_eq!(
            HigherOrderEngine::<f64>::flatten_mixed_index(3, &[0, 2], &[2, 1]).expect("flatten"),
            vec![0, 0, 2]
        );
    }

    #[test]
    fn recommended_step_sharpens_with_order() {
        let first: f64 = recommended_step(1);
        let second: f64 = recommended_step(2);
        assert!(first > 0.0 && second > 0.0);
        assert!(
            second > first,
            "a second-order stencil needs a coarser step than a first-order one: {first} vs {second}"
        );
    }
}
