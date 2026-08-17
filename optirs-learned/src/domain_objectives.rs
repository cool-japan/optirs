//! Meta-training objectives: the tasks a learned optimizer is trained *on*.
//!
//! A learned optimizer's weights can only be trained against something that can
//! actually be optimized, so meta-training needs a task abstraction with a real
//! loss and a real gradient. [`MetaObjective`] is that abstraction, and
//! [`QuadraticObjective`] is the canonical instance: an anisotropic quadratic
//! bowl whose optimum, loss and gradient are all known in closed form, which is
//! what makes a "did meta-training help?" assertion checkable rather than a
//! matter of opinion.
//!
//! Introduced alongside the [`crate::gnn_optimizer::meta_training`] entry point
//! (finding F75: the GNN and NTM optimizers had genuinely-correct forward maths
//! and no way whatsoever to train their weights).

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// A differentiable task a learned optimizer can be meta-trained on.
///
/// Implementors must be cheap to evaluate: a meta-training run calls
/// [`Self::loss_and_gradient`] on the order of
/// `population × iterations × tasks × horizon` times.
pub trait MetaObjective<T: Float + Debug + Send + Sync + 'static> {
    /// Number of parameters the task optimizes over.
    fn dimension(&self) -> usize;

    /// The point a rollout starts from. Must have length [`Self::dimension`].
    fn initial_parameters(&self) -> Array1<T>;

    /// The loss and its gradient at `params`.
    ///
    /// # Errors
    /// Returns an error if `params` has the wrong length.
    fn loss_and_gradient(&self, params: &Array1<T>) -> Result<(T, Array1<T>)>;

    /// A short human-readable name, for diagnostics.
    fn name(&self) -> &str {
        "meta-objective"
    }
}

/// An anisotropic quadratic bowl: `f(x) = ½ Σᵢ cᵢ (xᵢ − x*ᵢ)²`.
///
/// The Hessian is `diag(c)`, the unique minimum is at `x*` with `f = 0`, and the
/// condition number is `max(c) / min(c)` — so a family of these with different
/// `c` vectors is a controllable benchmark for whether a learned optimizer has
/// picked up anything about curvature.
#[derive(Debug, Clone)]
pub struct QuadraticObjective<T: Float + Debug + Send + Sync + 'static> {
    curvature: Array1<T>,
    optimum: Array1<T>,
    start: Array1<T>,
    label: String,
}

impl<T: Float + Debug + Send + Sync + 'static> QuadraticObjective<T> {
    /// Build a bowl from an explicit per-coordinate curvature vector, with the
    /// optimum at the origin and every start coordinate at `start_value`.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] for an empty or non-positive
    /// curvature vector, or a non-finite `start_value`.
    pub fn from_curvature(curvature: Vec<f64>, start_value: f64) -> Result<Self> {
        if curvature.is_empty() {
            return Err(OptimError::InvalidConfig(
                "a quadratic objective needs at least one coordinate".to_string(),
            ));
        }
        if !start_value.is_finite() {
            return Err(OptimError::InvalidConfig(
                "start_value must be finite".to_string(),
            ));
        }
        for value in &curvature {
            if !(value.is_finite() && *value > 0.0) {
                return Err(OptimError::InvalidConfig(format!(
                    "curvature entries must be finite and positive, got {value}"
                )));
            }
        }
        let dimension = curvature.len();
        let cast =
            |v: f64| -> T { scirs2_core::numeric::NumCast::from(v).unwrap_or_else(|| T::zero()) };
        Ok(Self {
            curvature: Array1::from_iter(curvature.into_iter().map(cast)),
            optimum: Array1::zeros(dimension),
            start: Array1::from_elem(dimension, cast(start_value)),
            label: "quadratic".to_string(),
        })
    }

    /// A `dimension`-dimensional bowl with uniform curvature.
    ///
    /// # Errors
    /// As [`Self::from_curvature`].
    pub fn isotropic(dimension: usize, curvature: f64, start_value: f64) -> Result<Self> {
        Self::from_curvature(vec![curvature; dimension], start_value)
    }

    /// Move the optimum away from the origin.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] if `optimum` has the wrong length or
    /// a non-finite entry.
    pub fn with_optimum(mut self, optimum: Vec<f64>) -> Result<Self> {
        if optimum.len() != self.curvature.len() {
            return Err(OptimError::InvalidConfig(format!(
                "optimum has {} coordinates but the objective has {}",
                optimum.len(),
                self.curvature.len()
            )));
        }
        if optimum.iter().any(|v| !v.is_finite()) {
            return Err(OptimError::InvalidConfig(
                "optimum entries must be finite".to_string(),
            ));
        }
        self.optimum = Array1::from_iter(
            optimum
                .into_iter()
                .map(|v| scirs2_core::numeric::NumCast::from(v).unwrap_or_else(|| T::zero())),
        );
        Ok(self)
    }

    /// Attach a diagnostic label.
    pub fn labelled(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// `max(c) / min(c)`, the condition number of the Hessian.
    pub fn condition_number(&self) -> f64 {
        let mut low = f64::INFINITY;
        let mut high = 0.0_f64;
        for value in self.curvature.iter() {
            let v = value.to_f64().unwrap_or(0.0);
            low = low.min(v);
            high = high.max(v);
        }
        if low > 0.0 && low.is_finite() {
            high / low
        } else {
            f64::INFINITY
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> MetaObjective<T> for QuadraticObjective<T> {
    fn dimension(&self) -> usize {
        self.curvature.len()
    }

    fn initial_parameters(&self) -> Array1<T> {
        self.start.clone()
    }

    fn loss_and_gradient(&self, params: &Array1<T>) -> Result<(T, Array1<T>)> {
        if params.len() != self.curvature.len() {
            return Err(OptimError::InvalidConfig(format!(
                "quadratic objective has {} coordinates but received {}",
                self.curvature.len(),
                params.len()
            )));
        }
        let half = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::one());
        let mut loss = T::zero();
        let mut gradient = Array1::zeros(params.len());
        for i in 0..params.len() {
            let offset = params[i] - self.optimum[i];
            let scaled = self.curvature[i] * offset;
            loss = loss + half * scaled * offset;
            gradient[i] = scaled;
        }
        Ok((loss, gradient))
    }

    fn name(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_loss_and_gradient_match_the_closed_form() {
        let task = QuadraticObjective::<f64>::from_curvature(vec![2.0, 8.0], 0.0)
            .expect("task")
            .with_optimum(vec![1.0, -0.5])
            .expect("optimum")
            .labelled("test-bowl");
        assert_eq!(task.dimension(), 2);
        assert_eq!(task.name(), "test-bowl");
        assert!((task.condition_number() - 4.0).abs() < 1e-12);

        // At x = (3, 0.5): offsets (2, 1) -> loss = ½(2·4 + 8·1) = 8,
        // gradient = (2·2, 8·1) = (4, 8).
        let point = Array1::from_vec(vec![3.0, 0.5]);
        let (loss, gradient) = task.loss_and_gradient(&point).expect("evaluate");
        assert!((loss - 8.0).abs() < 1e-12, "loss {loss}");
        assert!((gradient[0] - 4.0).abs() < 1e-12);
        assert!((gradient[1] - 8.0).abs() < 1e-12);

        // At the optimum the loss and gradient are exactly zero.
        let (loss, gradient) = task
            .loss_and_gradient(&Array1::from_vec(vec![1.0, -0.5]))
            .expect("evaluate");
        assert_eq!(loss, 0.0);
        assert!(gradient.iter().all(|g| *g == 0.0));
    }

    /// The analytic gradient must match a central finite difference — the whole
    /// point of a meta-objective is that its gradient is trustworthy.
    #[test]
    fn quadratic_gradient_matches_finite_differences() {
        let task =
            QuadraticObjective::<f64>::from_curvature(vec![0.7, 3.1, 11.0], 0.0).expect("task");
        let point = Array1::from_vec(vec![0.4, -1.2, 2.5]);
        let (_, analytic) = task.loss_and_gradient(&point).expect("analytic");

        let h = 1e-6;
        for i in 0..point.len() {
            let mut plus = point.clone();
            let mut minus = point.clone();
            plus[i] += h;
            minus[i] -= h;
            let (loss_plus, _) = task.loss_and_gradient(&plus).expect("plus");
            let (loss_minus, _) = task.loss_and_gradient(&minus).expect("minus");
            let numeric = (loss_plus - loss_minus) / (2.0 * h);
            assert!(
                (analytic[i] - numeric).abs() < 1e-6,
                "coordinate {i}: analytic {} vs numeric {numeric}",
                analytic[i]
            );
        }
    }

    #[test]
    fn invalid_objectives_are_rejected() {
        assert!(QuadraticObjective::<f64>::from_curvature(vec![], 0.0).is_err());
        assert!(QuadraticObjective::<f64>::from_curvature(vec![0.0], 0.0).is_err());
        assert!(QuadraticObjective::<f64>::from_curvature(vec![-1.0], 0.0).is_err());
        assert!(QuadraticObjective::<f64>::from_curvature(vec![f64::NAN], 0.0).is_err());
        assert!(QuadraticObjective::<f64>::from_curvature(vec![1.0], f64::INFINITY).is_err());
        assert!(QuadraticObjective::<f64>::isotropic(3, 1.0, 0.5)
            .expect("task")
            .with_optimum(vec![0.0, 0.0])
            .is_err());
        let task = QuadraticObjective::<f64>::isotropic(2, 1.0, 0.5).expect("task");
        assert!(task
            .loss_and_gradient(&Array1::from_vec(vec![1.0]))
            .is_err());
    }
}
