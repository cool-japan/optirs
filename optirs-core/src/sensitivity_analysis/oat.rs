// One-At-A-Time (OAT) local sensitivity analysis.
//
// Computes both central- and forward-difference gradients of a model around a
// user-supplied baseline. OAT is the cheapest analysis technique offered in
// this module and is best suited to interrogating a known good operating
// point (for example, the optimum returned by a Sobol/Morris-guided search).
//
// References
// ----------
// - Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer.*
//   Chapter on local methods, sections 1.2-1.3.

use crate::error::{OptimError, Result};
use crate::sensitivity_analysis::{SensitivityAnalyzer, SensitivityIndices};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Outcome of a One-At-a-Time local sensitivity analysis.
#[derive(Debug, Clone)]
pub struct OatResult<F: Float> {
    /// Central-difference gradient `[ (f(x+ε e_i) - f(x-ε e_i)) / (2ε) ]_i`.
    pub central_gradient: Vec<F>,
    /// Forward-difference gradient `[ (f(x+ε e_i) - f(x)) / ε ]_i`.
    pub forward_gradient: Vec<F>,
    /// Value of the model at the baseline point.
    pub baseline_value: F,
    /// Actual (absolute) perturbation size used per parameter.
    pub perturbation_size: Vec<F>,
    /// Parameter labels.
    pub parameter_names: Vec<String>,
}

impl<F: Float> OatResult<F> {
    /// Number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.central_gradient.len()
    }
}

/// One-At-a-Time local sensitivity analyzer.
#[derive(Debug, Clone)]
pub struct OatAnalyzer<F: Float + Debug> {
    /// Perturbation size expressed as a fraction of the bound width
    /// (default: 1 %).
    perturbation_fraction: F,
    /// Optional baseline. If `None`, the midpoint of `bounds` is used.
    baseline: Option<Array1<F>>,
    /// Cached result from the last successful analysis.
    last_result: Option<OatResult<F>>,
}

impl<F: Float + Debug> OatAnalyzer<F> {
    /// Construct a new analyzer with the default 1 % perturbation.
    pub fn new() -> Self {
        let default_eps = F::from(0.01_f64).unwrap_or_else(F::one);
        Self {
            perturbation_fraction: default_eps,
            baseline: None,
            last_result: None,
        }
    }

    /// Override the relative perturbation size. Values are clamped to a
    /// strictly positive range smaller than 0.5 so that `x ± ε` stays inside
    /// the bounds when the baseline lies in the interior.
    pub fn with_perturbation(mut self, eps: F) -> Self {
        let half = F::from(0.5_f64).unwrap_or_else(F::one);
        let tiny = F::from(1e-12_f64).unwrap_or_else(F::epsilon);
        let mut clamped = eps;
        if clamped <= F::zero() {
            clamped = tiny;
        }
        if clamped >= half {
            clamped = half - tiny;
        }
        self.perturbation_fraction = clamped;
        self
    }

    /// Override the baseline point. If unset the midpoint of `bounds` is
    /// used.
    pub fn with_baseline(mut self, baseline: Array1<F>) -> Self {
        self.baseline = Some(baseline);
        self
    }

    /// Currently configured relative perturbation size.
    pub fn perturbation_fraction(&self) -> F {
        self.perturbation_fraction
    }

    /// Cached result, if any.
    pub fn last_result(&self) -> Option<&OatResult<F>> {
        self.last_result.as_ref()
    }

    /// Run the analysis around the configured baseline (or the midpoint).
    pub fn analyze_oat(
        &mut self,
        model: &dyn Fn(&Array1<F>) -> F,
        bounds: &[(F, F)],
    ) -> Result<OatResult<F>> {
        let k = bounds.len();
        if k == 0 {
            return Err(OptimError::InvalidConfig(
                "OAT analysis requires at least one parameter".into(),
            ));
        }
        for (idx, (low, high)) in bounds.iter().enumerate() {
            if *low >= *high {
                return Err(OptimError::InvalidConfig(format!(
                    "bounds[{idx}] must satisfy low < high"
                )));
            }
        }

        // Determine baseline: explicit override or midpoint.
        let baseline = match &self.baseline {
            Some(b) => {
                if b.len() != k {
                    return Err(OptimError::InvalidConfig(format!(
                        "baseline has length {} but bounds have length {}",
                        b.len(),
                        k
                    )));
                }
                b.clone()
            }
            None => {
                let mut mid = Array1::<F>::zeros(k);
                let two = F::from(2.0_f64).unwrap_or_else(F::one);
                for (j, slot) in mid.iter_mut().enumerate() {
                    *slot = (bounds[j].0 + bounds[j].1) / two;
                }
                mid
            }
        };

        // Per-parameter absolute perturbation, derived from the bound width.
        let mut perturbation_size = Vec::with_capacity(k);
        for &(low, high) in bounds.iter() {
            let width = high - low;
            perturbation_size.push(width * self.perturbation_fraction);
        }

        let baseline_value = model(&baseline);

        let mut central_gradient = Vec::with_capacity(k);
        let mut forward_gradient = Vec::with_capacity(k);

        for j in 0..k {
            let eps = perturbation_size[j];
            if eps <= F::zero() {
                central_gradient.push(F::zero());
                forward_gradient.push(F::zero());
                continue;
            }
            let mut x_plus = baseline.clone();
            let mut x_minus = baseline.clone();
            x_plus[j] = x_plus[j] + eps;
            x_minus[j] = x_minus[j] - eps;

            // Project back into bounds so that the user cannot inadvertently
            // sample outside the rectangular domain when the baseline lies on
            // a boundary.
            x_plus[j] = if x_plus[j] > bounds[j].1 {
                bounds[j].1
            } else {
                x_plus[j]
            };
            x_minus[j] = if x_minus[j] < bounds[j].0 {
                bounds[j].0
            } else {
                x_minus[j]
            };

            let f_plus = model(&x_plus);
            let f_minus = model(&x_minus);

            // Central difference uses the *actual* span after clamping.
            let span = x_plus[j] - x_minus[j];
            let central = if span > F::zero() {
                (f_plus - f_minus) / span
            } else {
                F::zero()
            };
            central_gradient.push(central);

            let forward_span = x_plus[j] - baseline[j];
            let forward = if forward_span > F::zero() {
                (f_plus - baseline_value) / forward_span
            } else {
                F::zero()
            };
            forward_gradient.push(forward);
        }

        let parameter_names = (0..k).map(|i| format!("x{i}")).collect::<Vec<_>>();
        let result = OatResult {
            central_gradient,
            forward_gradient,
            baseline_value,
            perturbation_size,
            parameter_names,
        };
        self.last_result = Some(result.clone());
        Ok(result)
    }
}

impl<F: Float + Debug> Default for OatAnalyzer<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> SensitivityAnalyzer<F> for OatAnalyzer<F> {
    fn analyze(
        &mut self,
        model: &dyn Fn(&Array1<F>) -> F,
        bounds: &[(F, F)],
    ) -> Result<SensitivityIndices<F>> {
        let result = self.analyze_oat(model, bounds)?;
        // Surface |central gradient| as the "first-order" importance proxy and
        // the forward-difference magnitude as the "total" component so that
        // OAT plugs into the common analyzer interface.
        let first_order: Vec<F> = result.central_gradient.iter().map(|g| g.abs()).collect();
        let total_order: Vec<F> = result.forward_gradient.iter().map(|g| g.abs()).collect();
        Ok(SensitivityIndices {
            first_order,
            total_order,
            second_order: None,
            parameter_names: result.parameter_names.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_function_gradient() {
        let mut oa = OatAnalyzer::<f64>::new();
        // f(x) = 2 x1 + 3 x2.
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| 2.0 * x[0] + 3.0 * x[1];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let res = oa.analyze_oat(model, &bounds).expect("analyze failed");
        // Linear functions: the central difference recovers the slope
        // *exactly* in floating point (subject to round-off).
        assert!((res.central_gradient[0] - 2.0).abs() < 1e-9);
        assert!((res.central_gradient[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_quadratic_function_gradient() {
        let baseline = Array1::from(vec![1.0, 0.0]);
        let mut oa = OatAnalyzer::<f64>::new()
            .with_perturbation(0.001)
            .with_baseline(baseline);
        // f(x) = x1^2.
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| x[0] * x[0];
        let bounds = vec![(0.0, 2.0), (-1.0, 1.0)];
        let res = oa.analyze_oat(model, &bounds).expect("analyze failed");
        assert!(
            (res.central_gradient[0] - 2.0).abs() < 1e-4,
            "∂f/∂x₁ ≈ 2, got {}",
            res.central_gradient[0]
        );
        assert!(
            res.central_gradient[1].abs() < 1e-9,
            "∂f/∂x₂ should be 0, got {}",
            res.central_gradient[1]
        );
    }

    #[test]
    fn test_forward_difference_matches_for_linear() {
        let mut oa = OatAnalyzer::<f64>::new();
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| 1.5 * x[0] - 4.0 * x[1];
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let res = oa.analyze_oat(model, &bounds).expect("analyze failed");
        for (c, f) in res.central_gradient.iter().zip(res.forward_gradient.iter()) {
            assert!(
                (c - f).abs() < 1e-9,
                "central {c} disagrees with forward {f}"
            );
        }
    }

    #[test]
    fn test_central_vs_forward_consistency_for_smooth() {
        let mut oa = OatAnalyzer::<f64>::new().with_perturbation(0.0005);
        // Smooth f(x) = sin(x1) + x2^2.
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| x[0].sin() + x[1] * x[1];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let res = oa.analyze_oat(model, &bounds).expect("analyze failed");
        for (c, f) in res.central_gradient.iter().zip(res.forward_gradient.iter()) {
            assert!(
                (c - f).abs() < 5e-3,
                "central {c} vs forward {f} differ by too much"
            );
        }
    }

    #[test]
    fn test_perturbation_size_changes_result() {
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| x[0].powi(3);
        let bounds = vec![(0.0, 1.0)];
        let mut a = OatAnalyzer::<f64>::new().with_perturbation(0.001);
        let mut b = OatAnalyzer::<f64>::new().with_perturbation(0.2);
        let res_a = a.analyze_oat(model, &bounds).expect("analyze failed");
        let res_b = b.analyze_oat(model, &bounds).expect("analyze failed");
        // Larger ε must change the cached perturbation size and, for a
        // non-linear function, the forward-difference estimate.
        assert!(
            (res_a.perturbation_size[0] - res_b.perturbation_size[0]).abs() > 1e-6,
            "perturbation sizes did not change"
        );
        assert!(
            (res_a.forward_gradient[0] - res_b.forward_gradient[0]).abs() > 1e-3,
            "forward gradient should differ for cubic with different ε"
        );
    }

    #[test]
    fn test_baseline_value_recorded() {
        let baseline = Array1::from(vec![0.5, 0.25]);
        let mut oa = OatAnalyzer::<f64>::new().with_baseline(baseline);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| 3.0 * x[0] + x[1] * x[1];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let res = oa.analyze_oat(model, &bounds).expect("analyze failed");
        // 3 * 0.5 + 0.25^2 = 1.5625.
        assert!((res.baseline_value - 1.5625).abs() < 1e-9);
    }

    #[test]
    fn test_invalid_bounds_error() {
        let mut oa = OatAnalyzer::<f64>::new();
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x| x[0];
        let err = oa.analyze_oat(model, &[]);
        assert!(matches!(err, Err(OptimError::InvalidConfig(_))));
        let err2 = oa.analyze_oat(model, &[(1.0, 1.0)]);
        assert!(matches!(err2, Err(OptimError::InvalidConfig(_))));
    }
}
