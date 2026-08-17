//! Truncated backpropagation-through-time meta-training for the LSTM optimizer.
//!
//! # What this implements
//!
//! The learned-optimizer setup of Andrychowicz et al., *"Learning to learn by
//! gradient descent by gradient descent"* (NeurIPS 2016). An *optimizee* is a
//! task with parameters `θ` and loss `f(θ)`. The *optimizer* is the LSTM
//! controller with weights `φ`. At step `t`:
//!
//! ```text
//! x_t  = features(∇f(θ_{t-1}), history)      // see `build_lstm_features`
//! y_t  = net_φ(x_t)                          // LSTM stack -> LayerNorm -> [attention] -> projection
//! u_t  = lr_t · τ(y_t)                       // τ = the configured OutputTransform
//! θ_t  = θ_{t-1} − u_t                       // note the minus: matches `LSTMOptimizer::lstm_step`
//! ```
//!
//! The meta-loss over an unrolled horizon `K` is `L(φ) = Σ_{t=1..K} f(θ_t)`, and
//! this module computes `dL/dφ` and applies an Adam step to `φ`.
//!
//! # The one approximation, stated plainly
//!
//! `x_t` depends on `∇f(θ_{t-1})`, which depends on `φ`. Differentiating through
//! that dependency would require second derivatives of the optimizee. Following
//! the original paper we **drop** `∂x_t/∂φ` (the paper's Figure 2 caption: "we
//! ... drop gradients along the dashed edges"). Everything else — the LSTM
//! recurrence, the layer norms, the attention projections, the output projection
//! and the output transform — is differentiated exactly.
//!
//! With that approximation the meta-gradient is
//!
//! ```text
//! dL/dφ = Σ_t (Σ_{r≥t} ∇f(θ_r))ᵗ · ∂θ_t/∂φ|_{u}   =   −Σ_t s_tᵗ · ∂u_t/∂φ,
//! s_t := Σ_{r=t..K} ∇f(θ_r)
//! ```
//!
//! i.e. the reverse-cumulative sum of the per-step task gradients is what gets
//! injected at each unrolled step, and BPTT carries the recurrent part.
//!
//! # Verification
//!
//! [`meta_gradient_frozen`] is the exact gradient of [`rollout_loss_frozen`] —
//! the *same* function, over a pre-recorded (hence `φ`-independent) input
//! sequence. `bptt_gradient_matches_finite_differences` in
//! `tests/lstm_meta_training.rs` checks every controller parameter against a
//! central finite difference of that function in `f64`. Comparing against a
//! *live* rollout instead would (correctly) disagree, because a live rollout
//! includes exactly the term the approximation drops.
//!
//! # Determinism
//!
//! Dropout is bypassed during meta-training (`LSTMNetwork` is in evaluation mode
//! by default; see `LSTMNetwork::set_training`), so a rollout is a deterministic
//! function of `φ` and the frozen inputs. Without that the finite-difference
//! check would be meaningless.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{LSTMNetwork, OutputTransform, TrajectoryPoint};
use crate::error::{OptimError, Result};

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

/// A differentiable optimizee the meta-trainer can unroll on.
pub trait MetaTrainingTask<T: Float + Debug + Send + Sync + 'static> {
    /// Dimension of the optimizee's parameter vector.
    fn dimension(&self) -> usize;

    /// Loss at `params`.
    fn loss(&self, params: &Array1<T>) -> T;

    /// Gradient of [`Self::loss`] at `params`.
    fn gradient(&self, params: &Array1<T>) -> Array1<T>;

    /// Where a rollout starts.
    fn initial_parameters(&self) -> Array1<T>;
}

/// Diagonal quadratic optimizee `f(θ) = ½ Σ_j a_j (θ_j − θ*_j)²`.
///
/// The canonical learned-optimizer benchmark: convex, exactly solvable, and with
/// a controllable condition number (the spread of `a`).
#[derive(Debug, Clone)]
pub struct DiagonalQuadraticTask<T: Float + Debug + Send + Sync + 'static> {
    curvature: Array1<T>,
    optimum: Array1<T>,
    start: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> DiagonalQuadraticTask<T> {
    /// Build a task from explicit curvature, optimum and starting point.
    ///
    /// # Errors
    /// Returns `Err` when the three vectors disagree on length, are empty, or
    /// any curvature entry is non-positive (the problem would not be strictly
    /// convex and the surrogate would be meaningless).
    pub fn new(curvature: Array1<T>, optimum: Array1<T>, start: Array1<T>) -> Result<Self> {
        if curvature.is_empty() {
            return Err(OptimError::InvalidConfig(
                "a quadratic task needs at least one dimension".to_string(),
            ));
        }
        if curvature.len() != optimum.len() || curvature.len() != start.len() {
            return Err(OptimError::InvalidConfig(format!(
                "quadratic task vectors disagree: curvature {}, optimum {}, start {}",
                curvature.len(),
                optimum.len(),
                start.len()
            )));
        }
        if curvature.iter().any(|&a| a <= T::zero()) {
            return Err(OptimError::InvalidConfig(
                "quadratic curvature entries must be strictly positive".to_string(),
            ));
        }
        Ok(Self {
            curvature,
            optimum,
            start,
        })
    }

    /// Identify a diagonal quadratic surrogate from a recorded trajectory.
    ///
    /// The surrogate's gradient is `g_j(θ) = a_j·(θ_j − θ*_j) = a_j·θ_j − a_j·θ*_j`,
    /// which is **linear in `θ_j`** for each coordinate independently. So for
    /// every coordinate `j` we regress the observed `gradient[j]` on the observed
    /// `parameters[j]` across the trajectory by ordinary least squares. The slope
    /// is `a_j` and the intercept is `−a_j·θ*_j`, giving
    /// `θ*_j = −intercept / a_j`.
    ///
    /// A coordinate only yields a usable curvature when the least-squares slope is
    /// strictly positive. Coordinates whose parameter never varies (zero
    /// regression variance) or whose slope is non-positive / noise-dominated fall
    /// back to `a_j = 1`, with `θ*_j` placed so the observed mean gradient is
    /// reproduced at the observed mean parameter — the only choice consistent with
    /// the information available. **Not** clamping the slope up to a tiny
    /// `min_curvature` is deliberate: `θ*_j = mean_x − mean_y/a_j` with
    /// `a_j = 1e-6` would place the optimum up to a million units away and
    /// manufacture an absurd task.
    ///
    /// `θ*_j` is additionally kept inside the parameter span the trajectory
    /// actually visited, widened by one span, since the quadratic model has no
    /// evidence outside that region.
    ///
    /// The starting point is the trajectory's first recorded parameter vector.
    ///
    /// # Errors
    /// Returns `Err` when the trajectory has fewer than two points, when the
    /// points disagree on dimension, or when the dimension is zero.
    pub fn from_trajectory(points: &[TrajectoryPoint<T>]) -> Result<Self> {
        if points.len() < 2 {
            return Err(OptimError::InsufficientData(format!(
                "identifying a quadratic surrogate needs at least 2 trajectory \
                 points, got {}",
                points.len()
            )));
        }
        let dim = points[0].parameters.len();
        if dim == 0 {
            return Err(OptimError::InsufficientData(
                "trajectory points have zero-dimensional parameters".to_string(),
            ));
        }
        for (idx, p) in points.iter().enumerate() {
            if p.parameters.len() != dim || p.gradient.len() != dim {
                return Err(OptimError::InvalidConfig(format!(
                    "trajectory point {idx} has parameters of length {} and \
                     gradient of length {}, expected {dim}",
                    p.parameters.len(),
                    p.gradient.len()
                )));
            }
        }

        let n: T = scirs2_core::numeric::NumCast::from(points.len()).unwrap_or_else(T::one);
        let min_curvature: T =
            scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::one());
        let mut curvature = Array1::zeros(dim);
        let mut optimum = Array1::zeros(dim);

        for j in 0..dim {
            let mut sum_x = T::zero();
            let mut sum_y = T::zero();
            for p in points {
                sum_x = sum_x + p.parameters[j];
                sum_y = sum_y + p.gradient[j];
            }
            let mean_x = sum_x / n;
            let mean_y = sum_y / n;

            let mut sxx = T::zero();
            let mut sxy = T::zero();
            for p in points {
                let dx = p.parameters[j] - mean_x;
                sxx = sxx + dx * dx;
                sxy = sxy + dx * (p.gradient[j] - mean_y);
            }

            let fitted_slope = if sxx > T::zero() {
                Some(sxy / sxx)
            } else {
                // Degenerate coordinate: no variation to regress against.
                None
            };

            // A coordinate only yields a usable curvature when the fit is
            // strictly convex. Otherwise (no variation, or a non-positive /
            // noise-dominated slope) fall back to unit curvature. Clamping a
            // near-zero slope up to `min_curvature` instead — as this used to do —
            // is far worse than it looks: `θ* = mean_x − mean_y/a` with
            // `a = 1e-6` places the optimum up to a *million* units away, which
            // manufactures an absurd task and silently poisons the whole
            // meta-training batch it is averaged into.
            let a = match fitted_slope {
                Some(slope) if slope > min_curvature => slope,
                _ => T::one(),
            };

            // intercept = mean_y − a·mean_x, and intercept = −a·θ*  =>  θ* = mean_x − mean_y/a
            let mut opt = mean_x - mean_y / a;

            // Even a legitimately small curvature can push the optimum far outside
            // the region the trajectory actually visited, where the quadratic model
            // has no evidence at all. Keep it within the observed span, widened by
            // one span (or a unit margin for a single-point span).
            let mut lo = points[0].parameters[j];
            let mut hi = lo;
            for p in points {
                let v = p.parameters[j];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let span = hi - lo;
            let margin = if span > T::zero() { span } else { T::one() };
            if opt < lo - margin {
                opt = lo - margin;
            } else if opt > hi + margin {
                opt = hi + margin;
            }

            curvature[j] = a;
            optimum[j] = opt;
        }

        Ok(Self {
            curvature,
            optimum,
            start: points[0].parameters.clone(),
        })
    }

    /// Identified (or supplied) curvature.
    pub fn curvature(&self) -> &Array1<T> {
        &self.curvature
    }

    /// Identified (or supplied) optimum.
    pub fn optimum(&self) -> &Array1<T> {
        &self.optimum
    }
}

impl<T: Float + Debug + Send + Sync + 'static> MetaTrainingTask<T> for DiagonalQuadraticTask<T> {
    fn dimension(&self) -> usize {
        self.curvature.len()
    }

    fn loss(&self, params: &Array1<T>) -> T {
        let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(T::one);
        let mut acc = T::zero();
        let width = params.len().min(self.curvature.len());
        for j in 0..width {
            let d = params[j] - self.optimum[j];
            acc = acc + self.curvature[j] * d * d;
        }
        half * acc
    }

    fn gradient(&self, params: &Array1<T>) -> Array1<T> {
        let width = params.len().min(self.curvature.len());
        let mut g = Array1::zeros(params.len());
        for j in 0..width {
            g[j] = self.curvature[j] * (params[j] - self.optimum[j]);
        }
        g
    }

    fn initial_parameters(&self) -> Array1<T> {
        self.start.clone()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Meta-training hyper-parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BpttConfig {
    /// Unroll horizon `K`: how many optimizer steps one meta-gradient sees.
    pub unroll_steps: usize,
    /// Adam step size for the controller weights.
    pub meta_learning_rate: f64,
    /// Global L2 norm the meta-gradient is clipped to (`<= 0` disables clipping).
    pub gradient_clip: f64,
}

impl Default for BpttConfig {
    fn default() -> Self {
        Self {
            unroll_steps: 20,
            meta_learning_rate: 1e-2,
            gradient_clip: 1.0,
        }
    }
}

/// A recorded (hence `φ`-independent) rollout: exactly what
/// [`meta_gradient_frozen`] and [`rollout_loss_frozen`] consume.
#[derive(Debug, Clone)]
pub struct FrozenRollout<T: Float + Debug + Send + Sync + 'static> {
    /// LSTM input features, one per unrolled step.
    pub inputs: Vec<Array1<T>>,
    /// Per-step learning rate applied to the transformed output.
    pub learning_rates: Vec<T>,
    /// Gradient-norm-derived scale used by `OutputTransform::AdaptiveScale`,
    /// one per step. Recorded so that transform is reproducible too.
    pub adaptive_scales: Vec<T>,
    /// Optimizee parameters the rollout starts from.
    pub initial_parameters: Array1<T>,
}

// ---------------------------------------------------------------------------
// Gradient containers
// ---------------------------------------------------------------------------

/// Accumulated gradients for one LSTM layer.
#[derive(Debug, Clone)]
struct LayerGrads<T: Float + Debug + Send + Sync + 'static> {
    weight_ih: Array2<T>,
    weight_hh: Array2<T>,
    bias_ih: Array1<T>,
    bias_hh: Array1<T>,
}

/// Accumulated gradients for one layer normalization.
#[derive(Debug, Clone)]
struct NormGrads<T: Float + Debug + Send + Sync + 'static> {
    gamma: Array1<T>,
    beta: Array1<T>,
}

/// Accumulated gradients for the attention projections.
///
/// At this call site the attention output is `output_proj · (value_proj · x)` —
/// the softmax is over a single key and is identically `1`, so `query_proj` and
/// `key_proj` have **exactly zero** influence on the output and therefore exactly
/// zero gradient. That is a property of the sequence-length-1 calling convention,
/// not an omission; the zero blocks are materialized so the flattened parameter
/// vector still covers every weight.
#[derive(Debug, Clone)]
struct AttentionGrads<T: Float + Debug + Send + Sync + 'static> {
    query_proj: Array2<T>,
    key_proj: Array2<T>,
    value_proj: Array2<T>,
    output_proj: Array2<T>,
}

/// Meta-gradient of the controller weights, in the same layout as
/// [`flatten_parameters`].
#[derive(Debug, Clone)]
pub struct NetworkGradients<T: Float + Debug + Send + Sync + 'static> {
    layers: Vec<LayerGrads<T>>,
    norms: Vec<NormGrads<T>>,
    attention: Option<AttentionGrads<T>>,
    projection_weights: Array2<T>,
    projection_bias: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> NetworkGradients<T> {
    /// Zero-initialized gradients shaped like `network`.
    fn zeros_like(network: &LSTMNetwork<T>) -> Self {
        let layers = network
            .layers
            .iter()
            .map(|l| LayerGrads {
                weight_ih: Array2::zeros(l.weight_ih.dim()),
                weight_hh: Array2::zeros(l.weight_hh.dim()),
                bias_ih: Array1::zeros(l.bias_ih.len()),
                bias_hh: Array1::zeros(l.bias_hh.len()),
            })
            .collect();
        let norms = network
            .layer_norms
            .iter()
            .map(|n| NormGrads {
                gamma: Array1::zeros(n.gamma.len()),
                beta: Array1::zeros(n.beta.len()),
            })
            .collect();
        let attention = network.attention.as_ref().map(|a| AttentionGrads {
            query_proj: Array2::zeros(a.query_proj.dim()),
            key_proj: Array2::zeros(a.key_proj.dim()),
            value_proj: Array2::zeros(a.value_proj.dim()),
            output_proj: Array2::zeros(a.output_proj.dim()),
        });
        Self {
            layers,
            norms,
            attention,
            projection_weights: Array2::zeros(network.output_projection.weights.dim()),
            projection_bias: Array1::zeros(network.output_projection.bias.len()),
        }
    }

    /// Flatten in the same order as [`flatten_parameters`].
    pub fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::new();
        for l in &self.layers {
            out.extend(l.weight_ih.iter().copied());
            out.extend(l.weight_hh.iter().copied());
            out.extend(l.bias_ih.iter().copied());
            out.extend(l.bias_hh.iter().copied());
        }
        for n in &self.norms {
            out.extend(n.gamma.iter().copied());
            out.extend(n.beta.iter().copied());
        }
        if let Some(a) = &self.attention {
            out.extend(a.query_proj.iter().copied());
            out.extend(a.key_proj.iter().copied());
            out.extend(a.value_proj.iter().copied());
            out.extend(a.output_proj.iter().copied());
        }
        out.extend(self.projection_weights.iter().copied());
        out.extend(self.projection_bias.iter().copied());
        out
    }

    /// Add `weight · other` into `self` (used to sum across tasks).
    pub(super) fn add_assign_weighted(&mut self, other: &Self, weight: T) {
        for (a, b) in self.layers.iter_mut().zip(other.layers.iter()) {
            a.weight_ih
                .zip_mut_with(&b.weight_ih, |x, &y| *x = *x + weight * y);
            a.weight_hh
                .zip_mut_with(&b.weight_hh, |x, &y| *x = *x + weight * y);
            a.bias_ih
                .zip_mut_with(&b.bias_ih, |x, &y| *x = *x + weight * y);
            a.bias_hh
                .zip_mut_with(&b.bias_hh, |x, &y| *x = *x + weight * y);
        }
        for (a, b) in self.norms.iter_mut().zip(other.norms.iter()) {
            a.gamma.zip_mut_with(&b.gamma, |x, &y| *x = *x + weight * y);
            a.beta.zip_mut_with(&b.beta, |x, &y| *x = *x + weight * y);
        }
        if let (Some(a), Some(b)) = (self.attention.as_mut(), other.attention.as_ref()) {
            a.query_proj
                .zip_mut_with(&b.query_proj, |x, &y| *x = *x + weight * y);
            a.key_proj
                .zip_mut_with(&b.key_proj, |x, &y| *x = *x + weight * y);
            a.value_proj
                .zip_mut_with(&b.value_proj, |x, &y| *x = *x + weight * y);
            a.output_proj
                .zip_mut_with(&b.output_proj, |x, &y| *x = *x + weight * y);
        }
        self.projection_weights
            .zip_mut_with(&other.projection_weights, |x, &y| *x = *x + weight * y);
        self.projection_bias
            .zip_mut_with(&other.projection_bias, |x, &y| *x = *x + weight * y);
    }

    /// Scale every entry.
    pub(super) fn scale_by(&mut self, factor: T) {
        for l in self.layers.iter_mut() {
            l.weight_ih.mapv_inplace(|x| x * factor);
            l.weight_hh.mapv_inplace(|x| x * factor);
            l.bias_ih.mapv_inplace(|x| x * factor);
            l.bias_hh.mapv_inplace(|x| x * factor);
        }
        for n in self.norms.iter_mut() {
            n.gamma.mapv_inplace(|x| x * factor);
            n.beta.mapv_inplace(|x| x * factor);
        }
        if let Some(a) = self.attention.as_mut() {
            a.query_proj.mapv_inplace(|x| x * factor);
            a.key_proj.mapv_inplace(|x| x * factor);
            a.value_proj.mapv_inplace(|x| x * factor);
            a.output_proj.mapv_inplace(|x| x * factor);
        }
        self.projection_weights.mapv_inplace(|x| x * factor);
        self.projection_bias.mapv_inplace(|x| x * factor);
    }

    /// Global L2 norm.
    pub fn l2_norm(&self) -> T {
        self.to_vec()
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |a, b| a + b)
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// Flat parameter access (used by the meta-optimizer and the FD test)
// ---------------------------------------------------------------------------

/// Flatten every learnable controller weight into one vector.
///
/// Layout: for each LSTM layer `weight_ih, weight_hh, bias_ih, bias_hh`; then for
/// each layer norm `gamma, beta`; then (if present) attention
/// `query_proj, key_proj, value_proj, output_proj`; then the output projection's
/// `weights, bias`. [`NetworkGradients::to_vec`] uses the identical order.
pub fn flatten_parameters<T: Float + Debug + Send + Sync + 'static>(
    network: &LSTMNetwork<T>,
) -> Vec<T> {
    let mut out = Vec::new();
    for l in &network.layers {
        out.extend(l.weight_ih.iter().copied());
        out.extend(l.weight_hh.iter().copied());
        out.extend(l.bias_ih.iter().copied());
        out.extend(l.bias_hh.iter().copied());
    }
    for n in &network.layer_norms {
        out.extend(n.gamma.iter().copied());
        out.extend(n.beta.iter().copied());
    }
    if let Some(a) = &network.attention {
        out.extend(a.query_proj.iter().copied());
        out.extend(a.key_proj.iter().copied());
        out.extend(a.value_proj.iter().copied());
        out.extend(a.output_proj.iter().copied());
    }
    out.extend(network.output_projection.weights.iter().copied());
    out.extend(network.output_projection.bias.iter().copied());
    out
}

/// Write a flat parameter vector back into `network`.
///
/// # Errors
/// Returns `Err` when `values` does not have exactly the length
/// [`flatten_parameters`] would produce.
pub fn set_parameters<T: Float + Debug + Send + Sync + 'static>(
    network: &mut LSTMNetwork<T>,
    values: &[T],
) -> Result<()> {
    let expected = flatten_parameters(network).len();
    if values.len() != expected {
        return Err(OptimError::InvalidConfig(format!(
            "controller has {expected} parameters, got {}",
            values.len()
        )));
    }
    let mut cursor = 0usize;
    let take_2d = |target: &mut Array2<T>, cursor: &mut usize| {
        for slot in target.iter_mut() {
            *slot = values[*cursor];
            *cursor += 1;
        }
    };
    let take_1d = |target: &mut Array1<T>, cursor: &mut usize| {
        for slot in target.iter_mut() {
            *slot = values[*cursor];
            *cursor += 1;
        }
    };

    for l in network.layers.iter_mut() {
        take_2d(&mut l.weight_ih, &mut cursor);
        take_2d(&mut l.weight_hh, &mut cursor);
        take_1d(&mut l.bias_ih, &mut cursor);
        take_1d(&mut l.bias_hh, &mut cursor);
    }
    for n in network.layer_norms.iter_mut() {
        take_1d(&mut n.gamma, &mut cursor);
        take_1d(&mut n.beta, &mut cursor);
    }
    if let Some(a) = network.attention.as_mut() {
        take_2d(&mut a.query_proj, &mut cursor);
        take_2d(&mut a.key_proj, &mut cursor);
        take_2d(&mut a.value_proj, &mut cursor);
        take_2d(&mut a.output_proj, &mut cursor);
    }
    take_2d(&mut network.output_projection.weights, &mut cursor);
    take_1d(&mut network.output_projection.bias, &mut cursor);
    Ok(())
}

// ---------------------------------------------------------------------------
// Forward tape
// ---------------------------------------------------------------------------

/// Everything one LSTM cell evaluation needs for its backward pass.
#[derive(Debug, Clone)]
struct CellTape<T: Float + Debug + Send + Sync + 'static> {
    input: Array1<T>,
    hidden_prev: Array1<T>,
    cell_prev: Array1<T>,
    input_gate: Array1<T>,
    forget_gate: Array1<T>,
    cell_gate: Array1<T>,
    output_gate: Array1<T>,
    tanh_cell: Array1<T>,
}

/// Everything one layer-norm evaluation needs for its backward pass.
#[derive(Debug, Clone)]
struct NormTape<T: Float + Debug + Send + Sync + 'static> {
    normalized: Array1<T>,
    inv_std: T,
}

/// One unrolled step.
#[derive(Debug, Clone)]
struct StepTape<T: Float + Debug + Send + Sync + 'static> {
    cells: Vec<CellTape<T>>,
    norms: Vec<NormTape<T>>,
    /// Input to the attention block, and `value_proj · input`, when attention is on.
    attention_input: Option<Array1<T>>,
    attention_value: Option<Array1<T>>,
    /// Input to the output projection.
    projection_input: Array1<T>,
    /// Raw output projection result `y_t`.
    projection_output: Array1<T>,
    /// `d u_t / d y_t`, diagonal, folded with the step's learning rate.
    update_jacobian: Array1<T>,
}

/// Result of a taped forward rollout.
struct RolloutTape<T: Float + Debug + Send + Sync + 'static> {
    steps: Vec<StepTape<T>>,
    /// Task gradient at `θ_t` for `t = 1..K`.
    task_gradients: Vec<Array1<T>>,
    /// Summed task loss over `t = 1..K`.
    meta_loss: T,
    /// Optimizee parameters after the last step.
    final_parameters: Array1<T>,
}

fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

/// `d τ(y) / d y` for the configured transform, elementwise.
///
/// Mirrors `LSTMOptimizer::generate_updates` exactly:
/// `Identity → 1`, `Tanh → 1 − tanh(y)²`,
/// `ScaledTanh{scale} → scale·(1 − tanh(y)²)`,
/// `AdaptiveScale → 1/(1+‖g‖)` (constant in `y`),
/// `LearnedNonlinear → 1 − tanh(y)²` (it computes `tanh` via `exp`).
fn transform_derivative<T: Float + Debug + Send + Sync + 'static>(
    transform: OutputTransform,
    y: &Array1<T>,
    adaptive_scale: T,
) -> Array1<T> {
    match transform {
        OutputTransform::Identity => Array1::from_elem(y.len(), T::one()),
        OutputTransform::Tanh | OutputTransform::LearnedNonlinear => {
            y.mapv(|v| T::one() - v.tanh() * v.tanh())
        }
        OutputTransform::ScaledTanh { scale } => {
            let s: T = scirs2_core::numeric::NumCast::from(scale).unwrap_or_else(T::one);
            y.mapv(|v| s * (T::one() - v.tanh() * v.tanh()))
        }
        OutputTransform::AdaptiveScale => Array1::from_elem(y.len(), adaptive_scale),
    }
}

/// `τ(y)` for the configured transform, elementwise. Mirrors
/// `LSTMOptimizer::generate_updates`.
fn apply_transform<T: Float + Debug + Send + Sync + 'static>(
    transform: OutputTransform,
    y: &Array1<T>,
    adaptive_scale: T,
) -> Array1<T> {
    match transform {
        OutputTransform::Identity => y.clone(),
        OutputTransform::Tanh | OutputTransform::LearnedNonlinear => y.mapv(|v| v.tanh()),
        OutputTransform::ScaledTanh { scale } => {
            let s: T = scirs2_core::numeric::NumCast::from(scale).unwrap_or_else(T::one);
            y.mapv(|v| v.tanh() * s)
        }
        OutputTransform::AdaptiveScale => y.mapv(|v| v * adaptive_scale),
    }
}

/// Taped forward rollout over a frozen input sequence.
///
/// Reproduces `LSTMNetwork::forward` step by step (LSTM cell → layer norm per
/// layer, then optional attention, then the output projection) while recording
/// what the backward pass needs. Dropout is not applied — see the module docs.
fn taped_rollout<T, K>(
    network: &mut LSTMNetwork<T>,
    task: &K,
    rollout: &FrozenRollout<T>,
) -> Result<RolloutTape<T>>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
    K: MetaTrainingTask<T> + ?Sized,
{
    if rollout.inputs.is_empty() {
        return Err(OptimError::InsufficientData(
            "a rollout needs at least one unrolled step".to_string(),
        ));
    }
    if rollout.learning_rates.len() != rollout.inputs.len()
        || rollout.adaptive_scales.len() != rollout.inputs.len()
    {
        return Err(OptimError::InvalidConfig(
            "frozen rollout inputs, learning rates and adaptive scales must agree in length"
                .to_string(),
        ));
    }
    let dim = task.dimension();
    if network.output_projection.output_size() != dim {
        return Err(OptimError::InvalidConfig(format!(
            "output projection produces {} values but the task has {dim} parameters; \
             call `MetaTrainer::prepare_network` first",
            network.output_projection.output_size()
        )));
    }

    network.reset_state();
    let transform = network.output_projection.output_transform;

    let mut params = rollout.initial_parameters.clone();
    if params.len() != dim {
        return Err(OptimError::InvalidConfig(format!(
            "rollout starts from {} parameters but the task has {dim}",
            params.len()
        )));
    }

    let mut steps = Vec::with_capacity(rollout.inputs.len());
    let mut task_gradients = Vec::with_capacity(rollout.inputs.len());
    let mut meta_loss = T::zero();

    for (t, features) in rollout.inputs.iter().enumerate() {
        let mut cells = Vec::with_capacity(network.layers.len());
        let mut norms = Vec::with_capacity(network.layers.len());
        let mut activation = features.clone();

        for layer_index in 0..network.layers.len() {
            let layer = &mut network.layers[layer_index];
            if activation.len() != layer.weight_ih.ncols() {
                return Err(OptimError::InvalidConfig(format!(
                    "layer {layer_index} expects {} inputs, got {}",
                    layer.weight_ih.ncols(),
                    activation.len()
                )));
            }
            let hidden_size = layer.hiddensize;
            let hidden_prev = layer.hidden_state.clone();
            let cell_prev = layer.cell_state.clone();

            let gates = layer.weight_ih.dot(&activation)
                + &layer.bias_ih
                + layer.weight_hh.dot(&hidden_prev)
                + &layer.bias_hh;

            let mut input_gate = Array1::zeros(hidden_size);
            let mut forget_gate = Array1::zeros(hidden_size);
            let mut cell_gate = Array1::zeros(hidden_size);
            let mut output_gate = Array1::zeros(hidden_size);
            for k in 0..hidden_size {
                input_gate[k] = sigmoid(gates[k]);
                forget_gate[k] = sigmoid(gates[hidden_size + k]);
                cell_gate[k] = gates[2 * hidden_size + k].tanh();
                output_gate[k] = sigmoid(gates[3 * hidden_size + k]);
            }

            let mut cell = Array1::zeros(hidden_size);
            let mut tanh_cell = Array1::zeros(hidden_size);
            let mut hidden = Array1::zeros(hidden_size);
            for k in 0..hidden_size {
                cell[k] = forget_gate[k] * cell_prev[k] + input_gate[k] * cell_gate[k];
                tanh_cell[k] = cell[k].tanh();
                hidden[k] = output_gate[k] * tanh_cell[k];
            }

            layer.cell_state = cell.clone();
            layer.hidden_state = hidden.clone();

            cells.push(CellTape {
                input: activation.clone(),
                hidden_prev,
                cell_prev,
                input_gate,
                forget_gate,
                cell_gate,
                output_gate,
                tanh_cell,
            });

            // Layer normalization.
            let norm = &network.layer_norms[layer_index];
            if hidden.len() != norm.gamma.len() {
                return Err(OptimError::InvalidConfig(format!(
                    "layer norm {layer_index} expects {} features, got {}",
                    norm.gamma.len(),
                    hidden.len()
                )));
            }
            let n: T = scirs2_core::numeric::NumCast::from(hidden.len()).unwrap_or_else(T::one);
            let mean = hidden.iter().copied().fold(T::zero(), |a, b| a + b) / n;
            let variance = hidden
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(T::zero(), |a, b| a + b)
                / n;
            let inv_std = T::one() / (variance + norm.epsilon).sqrt();
            let normalized = hidden.mapv(|x| (x - mean) * inv_std);
            let mut out = Array1::zeros(hidden.len());
            for k in 0..hidden.len() {
                out[k] = normalized[k] * norm.gamma[k] + norm.beta[k];
            }
            norms.push(NormTape {
                normalized,
                inv_std,
            });
            activation = out;
        }

        // Attention (a pure linear map at sequence length 1, see AttentionGrads).
        let (attention_input, attention_value) = match network.attention.as_mut() {
            Some(attention) => {
                if activation.len() != attention.query_proj.nrows() {
                    return Err(OptimError::InvalidConfig(format!(
                        "attention expects {} features, got {}",
                        attention.query_proj.nrows(),
                        activation.len()
                    )));
                }
                let value = attention.value_proj.dot(&activation);
                let out = attention.output_proj.dot(&value);
                let input = activation;
                activation = out;
                (Some(input), Some(value))
            }
            None => (None, None),
        };

        let projection_input = activation;
        let projection_output = network.output_projection.forward(&projection_input)?;

        let lr = rollout.learning_rates[t];
        let adaptive_scale = rollout.adaptive_scales[t];
        let update =
            apply_transform(transform, &projection_output, adaptive_scale).mapv(|v| v * lr);
        let mut update_jacobian =
            transform_derivative(transform, &projection_output, adaptive_scale);
        update_jacobian.mapv_inplace(|v| v * lr);

        // θ_t = θ_{t-1} − u_t  (matches LSTMOptimizer::lstm_step)
        for j in 0..dim {
            params[j] = params[j] - update[j];
        }

        meta_loss = meta_loss + task.loss(&params);
        task_gradients.push(task.gradient(&params));

        steps.push(StepTape {
            cells,
            norms,
            attention_input,
            attention_value,
            projection_input,
            projection_output,
            update_jacobian,
        });
    }

    Ok(RolloutTape {
        steps,
        task_gradients,
        meta_loss,
        final_parameters: params,
    })
}

/// Backward pass over a recorded tape.
fn backward_tape<T>(network: &LSTMNetwork<T>, tape: &RolloutTape<T>) -> NetworkGradients<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let mut grads = NetworkGradients::zeros_like(network);
    let num_layers = network.layers.len();
    let horizon = tape.steps.len();

    // Reverse-cumulative sum of the task gradients: s_t = Σ_{r≥t} ∇f(θ_r).
    let dim = tape.task_gradients[0].len();
    let mut suffix: Vec<Array1<T>> = vec![Array1::zeros(dim); horizon];
    let mut running: Array1<T> = Array1::zeros(dim);
    for t in (0..horizon).rev() {
        running = &running + &tape.task_gradients[t];
        suffix[t] = running.clone();
    }

    // Recurrent carries, per layer.
    let mut dh_next: Vec<Array1<T>> = network
        .layers
        .iter()
        .map(|l| Array1::zeros(l.hiddensize))
        .collect();
    let mut dc_next: Vec<Array1<T>> = dh_next.clone();

    for t in (0..horizon).rev() {
        let step = &tape.steps[t];

        // dL/dy_t = −s_t ⊙ (du_t/dy_t)
        let mut grad_y = Array1::zeros(step.projection_output.len());
        for j in 0..grad_y.len() {
            grad_y[j] = -suffix[t][j] * step.update_jacobian[j];
        }

        // Output projection: y = W·x + b
        for r in 0..grads.projection_weights.nrows() {
            let g = grad_y[r];
            if g == T::zero() {
                continue;
            }
            for c in 0..grads.projection_weights.ncols() {
                grads.projection_weights[[r, c]] =
                    grads.projection_weights[[r, c]] + g * step.projection_input[c];
            }
        }
        for r in 0..grads.projection_bias.len() {
            grads.projection_bias[r] = grads.projection_bias[r] + grad_y[r];
        }
        let mut grad_activation = network.output_projection.weights.t().dot(&grad_y);

        // Attention: out = output_proj · (value_proj · x)
        if let (Some(attention), Some(agrads), Some(att_in), Some(att_val)) = (
            network.attention.as_ref(),
            grads.attention.as_mut(),
            step.attention_input.as_ref(),
            step.attention_value.as_ref(),
        ) {
            for r in 0..agrads.output_proj.nrows() {
                let g = grad_activation[r];
                if g == T::zero() {
                    continue;
                }
                for c in 0..agrads.output_proj.ncols() {
                    agrads.output_proj[[r, c]] = agrads.output_proj[[r, c]] + g * att_val[c];
                }
            }
            let grad_value = attention.output_proj.t().dot(&grad_activation);
            for r in 0..agrads.value_proj.nrows() {
                let g = grad_value[r];
                if g == T::zero() {
                    continue;
                }
                for c in 0..agrads.value_proj.ncols() {
                    agrads.value_proj[[r, c]] = agrads.value_proj[[r, c]] + g * att_in[c];
                }
            }
            // query_proj / key_proj do not influence the output at sequence
            // length 1, so their gradients stay exactly zero.
            grad_activation = attention.value_proj.t().dot(&grad_value);
        }

        // Layers, top-down.
        for layer_index in (0..num_layers).rev() {
            let norm = &network.layer_norms[layer_index];
            let norm_tape = &step.norms[layer_index];
            let width = norm_tape.normalized.len();

            // Layer-norm backward.
            let mut grad_hidden = Array1::zeros(width);
            {
                let n: T = scirs2_core::numeric::NumCast::from(width).unwrap_or_else(T::one);
                let mut d = Array1::zeros(width);
                for k in 0..width {
                    let g = grad_activation[k];
                    grads.norms[layer_index].gamma[k] =
                        grads.norms[layer_index].gamma[k] + g * norm_tape.normalized[k];
                    grads.norms[layer_index].beta[k] = grads.norms[layer_index].beta[k] + g;
                    d[k] = g * norm.gamma[k];
                }
                let mean_d = d.iter().copied().fold(T::zero(), |a, b| a + b) / n;
                let mean_dx = d
                    .iter()
                    .zip(norm_tape.normalized.iter())
                    .fold(T::zero(), |acc, (&dk, &xk)| acc + dk * xk)
                    / n;
                for k in 0..width {
                    grad_hidden[k] =
                        norm_tape.inv_std * (d[k] - mean_d - norm_tape.normalized[k] * mean_dx);
                }
            }

            // Add the recurrent contribution from step t+1.
            for k in 0..width {
                grad_hidden[k] = grad_hidden[k] + dh_next[layer_index][k];
            }

            // LSTM cell backward.
            let cell_tape = &step.cells[layer_index];
            let layer = &network.layers[layer_index];
            let hidden_size = layer.hiddensize;

            let mut grad_gates = Array1::zeros(4 * hidden_size);
            let mut grad_cell_prev = Array1::zeros(hidden_size);
            for k in 0..hidden_size {
                let dh = grad_hidden[k];
                let d_output_gate = dh * cell_tape.tanh_cell[k];
                let dc = dh
                    * cell_tape.output_gate[k]
                    * (T::one() - cell_tape.tanh_cell[k] * cell_tape.tanh_cell[k])
                    + dc_next[layer_index][k];

                let d_input_gate = dc * cell_tape.cell_gate[k];
                let d_cell_gate = dc * cell_tape.input_gate[k];
                let d_forget_gate = dc * cell_tape.cell_prev[k];
                grad_cell_prev[k] = dc * cell_tape.forget_gate[k];

                let i = cell_tape.input_gate[k];
                let f = cell_tape.forget_gate[k];
                let g = cell_tape.cell_gate[k];
                let o = cell_tape.output_gate[k];
                grad_gates[k] = d_input_gate * i * (T::one() - i);
                grad_gates[hidden_size + k] = d_forget_gate * f * (T::one() - f);
                grad_gates[2 * hidden_size + k] = d_cell_gate * (T::one() - g * g);
                grad_gates[3 * hidden_size + k] = d_output_gate * o * (T::one() - o);
            }

            let lg = &mut grads.layers[layer_index];
            for r in 0..4 * hidden_size {
                let g = grad_gates[r];
                lg.bias_ih[r] = lg.bias_ih[r] + g;
                lg.bias_hh[r] = lg.bias_hh[r] + g;
                if g == T::zero() {
                    continue;
                }
                for c in 0..lg.weight_ih.ncols() {
                    lg.weight_ih[[r, c]] = lg.weight_ih[[r, c]] + g * cell_tape.input[c];
                }
                for c in 0..lg.weight_hh.ncols() {
                    lg.weight_hh[[r, c]] = lg.weight_hh[[r, c]] + g * cell_tape.hidden_prev[c];
                }
            }

            dh_next[layer_index] = layer.weight_hh.t().dot(&grad_gates);
            dc_next[layer_index] = grad_cell_prev;

            // Gradient flowing into this layer's input, which is the previous
            // layer's (normalized) output at the same step. For layer 0 the input
            // is the frozen feature vector, whose gradient is deliberately
            // discarded (see the module-level note on the approximation).
            grad_activation = layer.weight_ih.t().dot(&grad_gates);
        }
    }

    grads
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Loss of a frozen rollout — forward only. This is the function
/// [`meta_gradient_frozen`] differentiates exactly.
pub fn rollout_loss_frozen<T, K>(
    network: &mut LSTMNetwork<T>,
    task: &K,
    rollout: &FrozenRollout<T>,
) -> Result<T>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
    K: MetaTrainingTask<T> + ?Sized,
{
    Ok(taped_rollout(network, task, rollout)?.meta_loss)
}

/// Exact meta-gradient of [`rollout_loss_frozen`].
pub fn meta_gradient_frozen<T, K>(
    network: &mut LSTMNetwork<T>,
    task: &K,
    rollout: &FrozenRollout<T>,
) -> Result<(T, NetworkGradients<T>)>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
    K: MetaTrainingTask<T> + ?Sized,
{
    let tape = taped_rollout(network, task, rollout)?;
    let grads = backward_tape(network, &tape);
    Ok((tape.meta_loss, grads))
}

/// Per-step raw output-projection results `y_t` recorded by the taped forward.
///
/// Exposed so the taped forward can be pinned against the deployed
/// [`LSTMNetwork::forward`]. The two must agree to floating-point noise; if they
/// do not, meta-training is optimizing a different function than the one that
/// ships, and [`meta_gradient_frozen`]'s finite-difference check would not catch
/// it because that check is self-consistent by construction.
pub fn taped_projection_outputs<T, K>(
    network: &mut LSTMNetwork<T>,
    task: &K,
    rollout: &FrozenRollout<T>,
) -> Result<Vec<Array1<T>>>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
    K: MetaTrainingTask<T> + ?Sized,
{
    let tape = taped_rollout(network, task, rollout)?;
    Ok(tape
        .steps
        .iter()
        .map(|s| s.projection_output.clone())
        .collect())
}

/// Final optimizee parameters after a frozen rollout — used to measure how well
/// a controller actually optimizes.
pub fn rollout_final_parameters<T, K>(
    network: &mut LSTMNetwork<T>,
    task: &K,
    rollout: &FrozenRollout<T>,
) -> Result<Array1<T>>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
    K: MetaTrainingTask<T> + ?Sized,
{
    Ok(taped_rollout(network, task, rollout)?.final_parameters)
}
