//! Meta-training learned optimizers by evolution strategies.
//!
//! # What finding F75 was about
//!
//! [`GnnOptimizer`](crate::gnn_optimizer::GnnOptimizer) and
//! [`NtmOptimizer`](crate::ntm_optimizer::NtmOptimizer) both have real forward
//! maths — a genuine message-passing round with a GRU node update, and genuine
//! NTM content/location addressing with erase-add memory writes. What was missing
//! is that **nothing ever trained their weights**. Both drew their controller
//! weights from the configured seed at construction and kept that draw forever, so
//! "learned optimizer" meant "randomly initialised optimizer" in every use. The
//! module docs disclosed it honestly, but a disclosed gap is still a gap.
//!
//! # Why evolution strategies and not backpropagation
//!
//! Meta-training a learned optimizer by gradients means differentiating through
//! the unrolled inner optimization: `∂L_meta/∂w` has to flow back through
//! `horizon` applications of
//! [`AdvancedOptimizer::step`],
//! each of which contains a recurrent cell, a nonlinear addressing or aggregation
//! stage and a nonlinear readout. That is the truncated-BPTT machinery
//! `lstm::bptt` (behind the `lstm` feature) implements for the LSTM controller,
//! and doing it here
//! would mean hand-deriving and maintaining a separate adjoint for each of two
//! quite different architectures.
//!
//! Evolution strategies ([Salimans et al., 2017](https://arxiv.org/abs/1703.03864))
//! estimate the same `∂L_meta/∂w` from forward rollouts alone:
//!
//! ```text
//! ∇_w E[L(w + σ·ε)] = (1/σ) · E[ L(w + σ·ε) · ε ],   ε ~ N(0, I)
//! ```
//!
//! evaluated here with **antithetic (mirrored) pairs**, which cancels the
//! leading term of the estimator's variance:
//!
//! ```text
//! ĝ = (1 / (pairs · σ)) · Σₖ ½·(L(w + σ·εₖ) − L(w − σ·εₖ)) · εₖ
//! ```
//!
//! This is a real gradient estimator of a real objective, not a surrogate: every
//! loss fed into it comes from actually running the optimizer on an actual task.
//! It is far less sample-efficient than BPTT and is **not** presented as
//! equivalent — [`MetaTrainingReport`] reports the measured meta-loss before and
//! after, so a caller can see exactly what the training bought.
//!
//! # Meta-objective
//!
//! For each task the trainer rolls the optimizer out for `horizon` steps from the
//! task's own start point and accumulates the loss curve, normalised by the
//! initial loss:
//!
//! ```text
//! L_task = (1 / horizon) · Σ_{t=1..horizon} loss_t / loss_0
//! ```
//!
//! so `1.0` means "no progress", `0.0` means "solved immediately", and tasks of
//! very different scale contribute comparably. `L_meta` is the mean over tasks. A
//! non-finite rollout (a diverged optimizer) scores a large finite penalty rather
//! than poisoning the average with `NaN`.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng};
use std::fmt::Debug;

use crate::domain_objectives::MetaObjective;
use crate::domain_optimizers::AdvancedOptimizer;
use crate::error::{OptimError, Result};

/// Score assigned to a rollout that produced a non-finite loss.
///
/// Large enough to be strictly worse than any sane rollout (whose normalised
/// score is bounded by ~1 unless the optimizer actively increases the loss), and
/// finite so it cannot poison the population mean.
const DIVERGENCE_PENALTY: f64 = 1.0e3;

/// A learned optimizer whose weights an [`EsMetaTrainer`] can train.
///
/// The three requirements beyond [`AdvancedOptimizer`] are exactly what an ES
/// loop needs: read the weights out as a flat vector, write a candidate vector
/// back, and clear the persistent per-rollout state so tasks cannot contaminate
/// each other.
pub trait MetaTrainable<T>: AdvancedOptimizer<T> + Clone
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Every learned weight, flattened in an order that is stable across calls
    /// for a given architecture, so that
    /// `set_weight_vector(&weight_vector())` is the identity.
    fn weight_vector(&self) -> Vec<f64>;

    /// Overwrite every learned weight from a vector laid out like
    /// [`Self::weight_vector`]'s output.
    ///
    /// # Errors
    /// Must return an error — never partially apply — if `flat` is not exactly
    /// the right length for this architecture.
    fn set_weight_vector(&mut self, flat: &[f64]) -> Result<()>;

    /// Clear the persistent optimization state (momentum, hidden vectors,
    /// memory, step count) while keeping the learned weights.
    fn reset_state(&mut self);

    /// Number of learned weights.
    fn weight_count(&self) -> usize {
        self.weight_vector().len()
    }
}

/// Hyper-parameters for [`EsMetaTrainer`].
#[derive(Debug, Clone)]
pub struct MetaTrainingConfig {
    /// Number of weight perturbations evaluated per iteration. Rounded **up** to
    /// the next even number, because perturbations are drawn in mirrored pairs.
    pub population: usize,
    /// Number of ES iterations.
    pub iterations: usize,
    /// Perturbation standard deviation `σ`. Too small and the finite-difference
    /// signal drowns in rollout noise; too large and the estimate stops being the
    /// gradient of anything local.
    pub sigma: f64,
    /// Adam step size applied to the estimated meta-gradient.
    pub meta_learning_rate: f64,
    /// Inner-loop rollout length: how many optimizer steps each evaluation runs.
    pub horizon: usize,
    /// Seed for the perturbation draws, so a training run is reproducible.
    pub seed: u64,
}

impl Default for MetaTrainingConfig {
    fn default() -> Self {
        Self {
            population: 8,
            iterations: 20,
            sigma: 0.05,
            meta_learning_rate: 0.05,
            horizon: 20,
            seed: 0x11E7_A5EE,
        }
    }
}

impl MetaTrainingConfig {
    /// Validate the configuration.
    fn validate(&self) -> Result<()> {
        if self.population < 2 {
            return Err(OptimError::InvalidConfig(
                "population must be at least 2 (perturbations come in mirrored pairs)".to_string(),
            ));
        }
        if self.iterations == 0 {
            return Err(OptimError::InvalidConfig(
                "iterations must be at least 1".to_string(),
            ));
        }
        if !(self.sigma.is_finite() && self.sigma > 0.0) {
            return Err(OptimError::InvalidConfig(
                "sigma must be finite and positive".to_string(),
            ));
        }
        if !(self.meta_learning_rate.is_finite() && self.meta_learning_rate > 0.0) {
            return Err(OptimError::InvalidConfig(
                "meta_learning_rate must be finite and positive".to_string(),
            ));
        }
        if self.horizon == 0 {
            return Err(OptimError::InvalidConfig(
                "horizon must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Number of mirrored pairs actually evaluated.
    fn pairs(&self) -> usize {
        self.population.div_ceil(2)
    }
}

/// Which meta-loss the trainer used to decide which weights to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMetric {
    /// No validation tasks were supplied, so the training meta-loss chose the
    /// weights. This measures fit, not generalisation: an ES run can drive the
    /// training score down while making the optimizer *worse* on tasks outside
    /// the training family, and selecting on this metric cannot detect that.
    TrainingLoss,
    /// A held-out validation set chose the weights, so the kept weights are the
    /// ones that generalised best among those visited.
    ValidationLoss,
}

/// What a meta-training run measured.
///
/// Every field is an observation, not a target: `final_meta_loss` is the value the
/// kept weights actually achieve on the training tasks, and [`Self::improved`] is
/// the honest answer to "did this help", judged by whichever criterion
/// [`Self::selection_metric`] names.
#[derive(Debug, Clone)]
pub struct MetaTrainingReport {
    /// ES iterations completed.
    pub iterations: usize,
    /// Training meta-loss of the weights the trainer started from.
    pub initial_meta_loss: f64,
    /// Training meta-loss of the weights the trainer kept.
    pub final_meta_loss: f64,
    /// Training meta-loss after each iteration, in order.
    pub loss_history: Vec<f64>,
    /// Validation meta-loss of the starting weights, if validation tasks were
    /// supplied.
    pub initial_validation_loss: Option<f64>,
    /// Validation meta-loss of the weights the trainer kept.
    pub final_validation_loss: Option<f64>,
    /// Validation meta-loss after each iteration, if validation tasks were
    /// supplied.
    pub validation_history: Vec<f64>,
    /// Which loss chose the kept weights.
    pub selection_metric: SelectionMetric,
    /// Euclidean norm of the estimated meta-gradient at each iteration.
    pub gradient_norms: Vec<f64>,
    /// Number of learned weights being optimized.
    pub weight_count: usize,
}

impl MetaTrainingReport {
    /// The initial and final values of whichever loss chose the weights.
    fn selection_pair(&self) -> (f64, f64) {
        match self.selection_metric {
            SelectionMetric::TrainingLoss => (self.initial_meta_loss, self.final_meta_loss),
            SelectionMetric::ValidationLoss => (
                self.initial_validation_loss
                    .unwrap_or(self.initial_meta_loss),
                self.final_validation_loss.unwrap_or(self.final_meta_loss),
            ),
        }
    }

    /// Whether the run left the optimizer better than it found it, judged by the
    /// selection metric.
    ///
    /// # This is weaker than it sounds under [`SelectionMetric::TrainingLoss`]
    ///
    /// With no validation tasks there is nothing to compare the training loss
    /// against, so `improved() == true` means only "it fits the training tasks
    /// better" — which is exactly what an *overfitting* run also reports. The NTM
    /// controller was measured driving its training meta-loss down ~30% while
    /// making a held-out task 40x worse; that run returns `true` here, and under
    /// `TrainingLoss` [`Self::relative_improvement`] and
    /// [`Self::relative_training_improvement`] are the same number, so the report
    /// carries no signal that separates the two cases.
    ///
    /// To get a claim about generalisation, train through
    /// [`EsMetaTrainer::train_with_validation`]: the metric becomes
    /// [`SelectionMetric::ValidationLoss`], and the gap between
    /// `relative_improvement()` and `relative_training_improvement()` becomes the
    /// overfitting measurement. Check [`Self::selection_metric`] before trusting
    /// this method as a generalisation result.
    pub fn improved(&self) -> bool {
        let (initial, final_) = self.selection_pair();
        final_ < initial
    }

    /// Fractional reduction in the selection metric (negative if training made it
    /// worse).
    pub fn relative_improvement(&self) -> f64 {
        let (initial, final_) = self.selection_pair();
        if initial.abs() <= f64::EPSILON {
            return 0.0;
        }
        (initial - final_) / initial.abs()
    }

    /// Fractional reduction in the *training* meta-loss specifically.
    ///
    /// Comparing this against [`Self::relative_improvement`] under
    /// [`SelectionMetric::ValidationLoss`] is how a caller sees overfitting: a
    /// large training improvement next to a small validation one means the run
    /// specialised to the training family.
    pub fn relative_training_improvement(&self) -> f64 {
        if self.initial_meta_loss.abs() <= f64::EPSILON {
            return 0.0;
        }
        (self.initial_meta_loss - self.final_meta_loss) / self.initial_meta_loss.abs()
    }
}

/// Evolution-strategies meta-trainer for any [`MetaTrainable`] optimizer.
#[derive(Debug, Clone)]
pub struct EsMetaTrainer {
    config: MetaTrainingConfig,
}

impl EsMetaTrainer {
    /// Build a trainer.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] for an out-of-range hyper-parameter.
    pub fn new(config: MetaTrainingConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// The configuration in use.
    pub fn config(&self) -> &MetaTrainingConfig {
        &self.config
    }

    /// Roll `optimizer` out on every task and return the mean normalised loss
    /// curve — the meta-objective, measured, with no training involved.
    ///
    /// The optimizer's persistent state is reset before each task, so tasks do not
    /// contaminate each other and the score depends only on the weights. The
    /// optimizer passed in is never mutated.
    ///
    /// # Errors
    /// Returns an error if `tasks` is empty, if a task is internally inconsistent,
    /// or if the objective cannot be evaluated.
    pub fn meta_loss<T, O>(&self, optimizer: &O, tasks: &[&dyn MetaObjective<T>]) -> Result<f64>
    where
        T: Float + Debug + Send + Sync + 'static,
        O: MetaTrainable<T>,
    {
        if tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "meta_loss needs at least one task".to_string(),
            ));
        }
        let mut total = 0.0;
        for task in tasks {
            total += self.rollout_score(optimizer, *task)?;
        }
        Ok(total / tasks.len() as f64)
    }

    /// One task's normalised loss-curve score.
    fn rollout_score<T, O>(&self, optimizer: &O, task: &dyn MetaObjective<T>) -> Result<f64>
    where
        T: Float + Debug + Send + Sync + 'static,
        O: MetaTrainable<T>,
    {
        let mut runner = optimizer.clone();
        runner.reset_state();

        let mut params = task.initial_parameters();
        if params.len() != task.dimension() {
            return Err(OptimError::InvalidConfig(format!(
                "task '{}' reported dimension {} but produced {} initial parameters",
                task.name(),
                task.dimension(),
                params.len()
            )));
        }

        let (initial_loss, _) = task.loss_and_gradient(&params)?;
        let initial = initial_loss.to_f64().unwrap_or(f64::NAN);
        if !initial.is_finite() || initial.abs() <= f64::EPSILON {
            // Nothing to improve on (or an unusable task): score it neutrally
            // rather than dividing by ~0 and manufacturing an enormous gradient.
            return Ok(1.0);
        }

        let mut accumulated = 0.0;
        for _ in 0..self.config.horizon {
            let (_, gradient) = task.loss_and_gradient(&params)?;
            params = runner.step(&params, &gradient)?;
            let (loss, _) = task.loss_and_gradient(&params)?;
            let value = loss.to_f64().unwrap_or(f64::NAN);
            if !value.is_finite() {
                return Ok(DIVERGENCE_PENALTY);
            }
            accumulated += value / initial;
        }
        let score = accumulated / self.config.horizon as f64;
        if score.is_finite() {
            Ok(score)
        } else {
            Ok(DIVERGENCE_PENALTY)
        }
    }

    /// Meta-train `optimizer`'s weights in place, selecting on the training loss.
    ///
    /// Equivalent to [`Self::train_with_validation`] with no validation tasks.
    /// **Prefer the validation form**: selecting on the training loss measures fit,
    /// not generalisation, and an ES run can drive the training score down while
    /// making the optimizer strictly worse outside the training family. The NTM
    /// controller does exactly that on a family of quadratics — training loss down
    /// ~30%, held-out loss up by a factor of 40 — which is why the validation
    /// entry point exists.
    ///
    /// # Errors
    /// Returns an error if `tasks` is empty, if the optimizer reports no learned
    /// weights, if a task is inconsistent, or if the objective cannot be
    /// evaluated.
    pub fn train<T, O>(
        &self,
        optimizer: &mut O,
        tasks: &[&dyn MetaObjective<T>],
    ) -> Result<MetaTrainingReport>
    where
        T: Float + Debug + Send + Sync + 'static,
        O: MetaTrainable<T>,
    {
        self.train_with_validation(optimizer, tasks, &[])
    }

    /// Meta-train `optimizer`'s weights in place, selecting the kept weights by
    /// their score on `validation`.
    ///
    /// The ES gradient is always estimated from `tasks`; `validation` only decides
    /// *which* of the visited weight vectors is kept. That separation is what makes
    /// the result a generalisation claim: the trainer never optimizes against the
    /// validation tasks, it only measures against them.
    ///
    /// The best weights *seen* are kept, never simply the last ones visited — an ES
    /// run on a noisy objective can wander uphill, and silently returning weights
    /// worse than the caller handed in would be the same class of dishonesty this
    /// module exists to remove. [`MetaTrainingReport::improved`] reports which
    /// happened, and comparing it with
    /// [`MetaTrainingReport::relative_training_improvement`] exposes overfitting.
    ///
    /// Pass an empty `validation` slice to select on the training loss instead.
    ///
    /// # Errors
    /// Returns an error if `tasks` is empty, if the optimizer reports no learned
    /// weights, if a task is inconsistent, or if an objective cannot be evaluated.
    pub fn train_with_validation<T, O>(
        &self,
        optimizer: &mut O,
        tasks: &[&dyn MetaObjective<T>],
        validation: &[&dyn MetaObjective<T>],
    ) -> Result<MetaTrainingReport>
    where
        T: Float + Debug + Send + Sync + 'static,
        O: MetaTrainable<T>,
    {
        if tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "meta-training needs at least one task".to_string(),
            ));
        }

        let baseline_weights = optimizer.weight_vector();
        let dimension = baseline_weights.len();
        if dimension == 0 {
            return Err(OptimError::InvalidState(
                "the optimizer reported zero learned weights".to_string(),
            ));
        }

        let selection_metric = if validation.is_empty() {
            SelectionMetric::TrainingLoss
        } else {
            SelectionMetric::ValidationLoss
        };

        let initial_meta_loss = self.meta_loss(optimizer, tasks)?;
        let initial_validation_loss = if validation.is_empty() {
            None
        } else {
            Some(self.meta_loss(optimizer, validation)?)
        };

        let mut weights = baseline_weights.clone();
        let mut best_weights = baseline_weights;
        let mut best_training = initial_meta_loss;
        let mut best_validation = initial_validation_loss;
        // The value being minimised for selection purposes.
        let mut best_selection = match selection_metric {
            SelectionMetric::TrainingLoss => initial_meta_loss,
            SelectionMetric::ValidationLoss => initial_validation_loss.unwrap_or(initial_meta_loss),
        };
        let mut validation_history = Vec::with_capacity(self.config.iterations);

        // Adam moments over the ES gradient estimate.
        let mut moment1 = vec![0.0; dimension];
        let mut moment2 = vec![0.0; dimension];
        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const EPSILON: f64 = 1e-8;

        let mut rng = Random::seed(self.config.seed);
        let pairs = self.config.pairs();
        let mut loss_history = Vec::with_capacity(self.config.iterations);
        let mut gradient_norms = Vec::with_capacity(self.config.iterations);
        let mut probe = optimizer.clone();

        for iteration in 0..self.config.iterations {
            let mut perturbations: Vec<Vec<f64>> = Vec::with_capacity(pairs);
            let mut differences: Vec<f64> = Vec::with_capacity(pairs);
            let mut evaluations: Vec<f64> = Vec::with_capacity(2 * pairs);

            for _ in 0..pairs {
                let epsilon = standard_normal_vector(&mut rng, dimension);

                let mut plus = weights.clone();
                let mut minus = weights.clone();
                for i in 0..dimension {
                    plus[i] += self.config.sigma * epsilon[i];
                    minus[i] -= self.config.sigma * epsilon[i];
                }

                probe.set_weight_vector(&plus)?;
                let loss_plus = self.meta_loss(&probe, tasks)?;
                probe.set_weight_vector(&minus)?;
                let loss_minus = self.meta_loss(&probe, tasks)?;

                evaluations.push(loss_plus);
                evaluations.push(loss_minus);
                differences.push(0.5 * (loss_plus - loss_minus));
                perturbations.push(epsilon);
            }

            // Fitness normalisation (Salimans et al. §2): dividing the centred
            // differences by their spread makes the step size independent of the
            // objective's scale, which matters because the normalised loss curve
            // shrinks as training succeeds.
            let spread = population_std(&evaluations);
            let scale = if spread > 1e-12 { 1.0 / spread } else { 1.0 };

            let mut gradient = vec![0.0; dimension];
            for (epsilon, difference) in perturbations.iter().zip(differences.iter()) {
                let weight = difference * scale;
                for i in 0..dimension {
                    gradient[i] += weight * epsilon[i];
                }
            }
            let normaliser = 1.0 / (pairs as f64 * self.config.sigma);
            for value in gradient.iter_mut() {
                *value *= normaliser;
            }
            gradient_norms.push(gradient.iter().map(|g| g * g).sum::<f64>().sqrt());

            // Adam step. ES estimates ∇L and we are minimising, so we descend.
            let step = iteration as i32 + 1;
            let bias1 = 1.0 - BETA1.powi(step);
            let bias2 = 1.0 - BETA2.powi(step);
            for i in 0..dimension {
                moment1[i] = BETA1 * moment1[i] + (1.0 - BETA1) * gradient[i];
                moment2[i] = BETA2 * moment2[i] + (1.0 - BETA2) * gradient[i] * gradient[i];
                let m_hat = moment1[i] / bias1;
                let v_hat = moment2[i] / bias2;
                weights[i] -= self.config.meta_learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
            }

            probe.set_weight_vector(&weights)?;
            let training_loss = self.meta_loss(&probe, tasks)?;
            loss_history.push(training_loss);

            let validation_loss = if validation.is_empty() {
                None
            } else {
                let value = self.meta_loss(&probe, validation)?;
                validation_history.push(value);
                Some(value)
            };

            let selection = match selection_metric {
                SelectionMetric::TrainingLoss => training_loss,
                SelectionMetric::ValidationLoss => validation_loss.unwrap_or(training_loss),
            };
            if selection < best_selection {
                best_selection = selection;
                best_training = training_loss;
                best_validation = validation_loss;
                best_weights = weights.clone();
            }
        }

        optimizer.set_weight_vector(&best_weights)?;
        optimizer.reset_state();

        Ok(MetaTrainingReport {
            iterations: self.config.iterations,
            initial_meta_loss,
            final_meta_loss: best_training,
            loss_history,
            initial_validation_loss,
            final_validation_loss: best_validation,
            validation_history,
            selection_metric,
            gradient_norms,
            weight_count: dimension,
        })
    }
}

/// Population standard deviation of `values` (`0` for fewer than two samples).
fn population_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    variance.sqrt()
}

/// Draw `len` independent `N(0, 1)` samples by the Box–Muller transform.
///
/// `scirs2_core::random` gives uniforms, and Box–Muller turns a pair of them into
/// an exact pair of standard normals — no approximation, no extra dependency. The
/// lower bound is nudged off zero so `ln(u)` is always finite.
pub(crate) fn standard_normal_vector<R: Rng>(rng: &mut Random<R>, len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        out.push(radius * angle.cos());
        if out.len() < len {
            out.push(radius * angle.sin());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Flat weight-vector plumbing, shared by every `MetaTrainable` implementation.
// ---------------------------------------------------------------------------

/// Append every element of `matrix` to `out`, row-major.
pub(crate) fn push_matrix<T: Float>(matrix: &Array2<T>, out: &mut Vec<f64>) {
    out.extend(matrix.iter().map(|v| v.to_f64().unwrap_or(0.0)));
}

/// Append every element of `vector` to `out`.
pub(crate) fn push_vector<T: Float>(vector: &Array1<T>, out: &mut Vec<f64>) {
    out.extend(vector.iter().map(|v| v.to_f64().unwrap_or(0.0)));
}

/// Append a single scalar to `out`.
pub(crate) fn push_scalar<T: Float>(scalar: T, out: &mut Vec<f64>) {
    out.push(scalar.to_f64().unwrap_or(0.0));
}

/// Read `matrix.len()` values out of `flat` at `*cursor` into `matrix`, row-major.
pub(crate) fn pull_matrix<T: Float>(
    matrix: &mut Array2<T>,
    flat: &[f64],
    cursor: &mut usize,
) -> Result<()> {
    let end = *cursor + matrix.len();
    let slice = take(flat, *cursor, end)?;
    for (target, value) in matrix.iter_mut().zip(slice.iter()) {
        *target = scirs2_core::numeric::NumCast::from(*value).unwrap_or_else(|| T::zero());
    }
    *cursor = end;
    Ok(())
}

/// Read `vector.len()` values out of `flat` at `*cursor` into `vector`.
pub(crate) fn pull_vector<T: Float>(
    vector: &mut Array1<T>,
    flat: &[f64],
    cursor: &mut usize,
) -> Result<()> {
    let end = *cursor + vector.len();
    let slice = take(flat, *cursor, end)?;
    for (target, value) in vector.iter_mut().zip(slice.iter()) {
        *target = scirs2_core::numeric::NumCast::from(*value).unwrap_or_else(|| T::zero());
    }
    *cursor = end;
    Ok(())
}

/// Read one value out of `flat` at `*cursor` into `scalar`.
pub(crate) fn pull_scalar<T: Float>(
    scalar: &mut T,
    flat: &[f64],
    cursor: &mut usize,
) -> Result<()> {
    let slice = take(flat, *cursor, *cursor + 1)?;
    *scalar = scirs2_core::numeric::NumCast::from(slice[0]).unwrap_or_else(|| T::zero());
    *cursor += 1;
    Ok(())
}

/// Borrow `flat[start..end]`, or report exactly how short the vector is.
fn take(flat: &[f64], start: usize, end: usize) -> Result<&[f64]> {
    flat.get(start..end).ok_or_else(|| {
        OptimError::InvalidConfig(format!(
            "weight vector is too short: needed {end} values, got {}",
            flat.len()
        ))
    })
}

/// Fail unless the cursor consumed the whole vector.
///
/// A partial load would leave the network in a state no caller could reason
/// about, so an over-long vector is an error just like a short one.
pub(crate) fn expect_fully_consumed(flat: &[f64], cursor: usize) -> Result<()> {
    if cursor != flat.len() {
        return Err(OptimError::InvalidConfig(format!(
            "weight vector has {} values but this architecture needs exactly {cursor}",
            flat.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_muller_produces_standard_normals() {
        let mut rng = Random::seed(7);
        let samples = standard_normal_vector(&mut rng, 4000);
        assert_eq!(samples.len(), 4000);
        assert!(samples.iter().all(|v| v.is_finite()));
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.08, "mean {mean} is not ~0");
        assert!(
            (variance - 1.0).abs() < 0.12,
            "variance {variance} is not ~1"
        );
        // An odd length must still be exact.
        assert_eq!(standard_normal_vector(&mut rng, 7).len(), 7);
    }

    #[test]
    fn population_std_matches_the_definition() {
        assert_eq!(population_std(&[]), 0.0);
        assert_eq!(population_std(&[3.0]), 0.0);
        // Values 1,2,3,4: mean 2.5, variance 1.25, std sqrt(1.25).
        let observed = population_std(&[1.0, 2.0, 3.0, 4.0]);
        assert!((observed - 1.25_f64.sqrt()).abs() < 1e-12, "{observed}");
    }

    #[test]
    fn bad_configurations_are_rejected() {
        for config in [
            MetaTrainingConfig {
                population: 1,
                ..Default::default()
            },
            MetaTrainingConfig {
                iterations: 0,
                ..Default::default()
            },
            MetaTrainingConfig {
                sigma: 0.0,
                ..Default::default()
            },
            MetaTrainingConfig {
                sigma: f64::NAN,
                ..Default::default()
            },
            MetaTrainingConfig {
                meta_learning_rate: -1.0,
                ..Default::default()
            },
            MetaTrainingConfig {
                horizon: 0,
                ..Default::default()
            },
        ] {
            assert!(EsMetaTrainer::new(config).is_err());
        }
        let trainer = EsMetaTrainer::new(MetaTrainingConfig::default()).expect("default is valid");
        assert_eq!(trainer.config().population, 8);
    }

    #[test]
    fn odd_populations_round_up_to_whole_mirrored_pairs() {
        let config = MetaTrainingConfig {
            population: 7,
            ..Default::default()
        };
        assert_eq!(config.pairs(), 4, "7 perturbations means 4 mirrored pairs");
    }

    #[test]
    fn flat_plumbing_round_trips_and_rejects_bad_lengths() {
        use scirs2_core::ndarray::{Array1, Array2};

        let matrix = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f64);
        let vector = Array1::from_vec(vec![10.0, 11.0]);
        let scalar = 12.5_f64;

        let mut flat = Vec::new();
        push_matrix(&matrix, &mut flat);
        push_vector(&vector, &mut flat);
        push_scalar(scalar, &mut flat);
        assert_eq!(flat.len(), 6 + 2 + 1);

        let mut out_matrix = Array2::<f64>::zeros((2, 3));
        let mut out_vector = Array1::<f64>::zeros(2);
        let mut out_scalar = 0.0_f64;
        let mut cursor = 0usize;
        pull_matrix(&mut out_matrix, &flat, &mut cursor).expect("matrix");
        pull_vector(&mut out_vector, &flat, &mut cursor).expect("vector");
        pull_scalar(&mut out_scalar, &flat, &mut cursor).expect("scalar");
        expect_fully_consumed(&flat, cursor).expect("consumed");

        assert_eq!(out_matrix, matrix);
        assert_eq!(out_vector, vector);
        assert_eq!(out_scalar, scalar);

        // Short vector -> error, not a partial load.
        let mut cursor = 0usize;
        let mut target = Array2::<f64>::zeros((2, 3));
        assert!(pull_matrix(&mut target, &flat[..4], &mut cursor).is_err());
        // Over-long vector -> error too.
        assert!(expect_fully_consumed(&flat, 3).is_err());
    }
}
