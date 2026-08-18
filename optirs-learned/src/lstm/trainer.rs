//! The meta-training driver: unroll, backprop, Adam-update, repeat.
//!
//! [`MetaTrainer`] is what turns the exact meta-gradient computed by
//! [`super::bptt`] into actual learning. One [`MetaTrainer::meta_step`]:
//!
//! 1. For each task, runs a **live** rollout to record the input sequence the
//!    controller would really see (features derived from the optimizee's
//!    gradients via [`super::features::build_lstm_features`], the same function
//!    inference uses).
//! 2. Re-runs that rollout with a tape and backpropagates through time, giving
//!    the exact gradient of the recorded-input rollout loss.
//! 3. Averages the per-task gradients (weighted), clips the global L2 norm, and
//!    applies one Adam step to the flattened controller weights.
//!
//! Adam (Kingma & Ba 2015) with the standard `β₁ = 0.9`, `β₂ = 0.999`,
//! `ε = 1e-8` is used for the outer loop because the meta-gradient scale varies
//! by orders of magnitude across tasks and plain SGD needs a per-problem step
//! size to be stable.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

use super::bptt::{
    flatten_parameters, meta_gradient_frozen, rollout_loss_frozen, set_parameters, BpttConfig,
    FrozenRollout, MetaTrainingTask, NetworkGradients,
};
use super::features::build_lstm_features;
use super::{LSTMNetwork, OutputTransform};
use crate::error::{OptimError, Result};

/// How many previous gradients the recorded feature vector includes, matching
/// `LSTMOptimizer::prepare_lstm_input`'s `get_recent_gradients(5)`.
const FEATURE_HISTORY: usize = 5;

/// Adam state for the controller weights.
#[derive(Debug, Clone)]
struct AdamState<T: Float + Debug + Send + Sync + 'static> {
    first_moment: Vec<T>,
    second_moment: Vec<T>,
    step: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> AdamState<T> {
    fn new(width: usize) -> Self {
        Self {
            first_moment: vec![T::zero(); width],
            second_moment: vec![T::zero(); width],
            step: 0,
        }
    }
}

/// Truncated-BPTT meta-trainer for the LSTM controller.
#[derive(Debug, Clone)]
pub struct MetaTrainer<T: Float + Debug + Send + Sync + 'static> {
    config: BpttConfig,
    adam: Option<AdamState<T>>,
    last_gradient: Vec<T>,
    loss_history: VecDeque<T>,
}

impl<T> MetaTrainer<T>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
{
    /// Create a trainer with the given hyper-parameters.
    pub fn new(config: BpttConfig) -> Self {
        Self {
            config,
            adam: None,
            last_gradient: Vec::new(),
            loss_history: VecDeque::new(),
        }
    }

    /// Hyper-parameters in use.
    pub fn config(&self) -> &BpttConfig {
        &self.config
    }

    /// The meta-gradient applied by the most recent [`Self::meta_step`],
    /// flattened in [`flatten_parameters`] order (empty before the first step).
    pub fn last_gradient_vector(&self) -> Vec<T> {
        self.last_gradient.clone()
    }

    /// Meta-losses observed so far, oldest first.
    pub fn loss_history(&self) -> &VecDeque<T> {
        &self.loss_history
    }

    /// Resize the controller's output projection to the optimizee's dimension.
    ///
    /// The projection must emit exactly one update per optimizee parameter. This
    /// mirrors what `LSTMOptimizer::lstm_step` does at inference time, and is a
    /// no-op when the widths already agree (so it never silently discards trained
    /// weights).
    pub fn prepare_network(&self, network: &mut LSTMNetwork<T>, dimension: usize) -> Result<()> {
        if dimension == 0 {
            return Err(OptimError::InvalidConfig(
                "task dimension must be greater than 0".to_string(),
            ));
        }
        if network.output_projection.output_size() != dimension {
            let hidden = network.output_projection.weights.ncols();
            network.output_projection.reset(hidden, dimension);
        }
        Ok(())
    }

    /// Record the input sequence a live rollout would produce.
    ///
    /// The controller is advanced exactly as it would be at inference time —
    /// features from the optimizee's gradient plus the recent-gradient history,
    /// the network's own forward pass, the configured output transform — and the
    /// resulting inputs, per-step learning rates and adaptive scales are returned
    /// as a [`FrozenRollout`]. Freezing them is what makes the analytic
    /// meta-gradient exactly checkable by finite differences (see the
    /// `super::bptt` module docs).
    ///
    /// `learning_rate` is treated as a constant with respect to the controller
    /// weights, which it is: the adaptive learning-rate controller reads only
    /// gradients and losses.
    ///
    /// # Errors
    /// Propagates feature-construction and forward-pass errors, and rejects a
    /// zero unroll horizon.
    pub fn capture_rollout<K>(
        &self,
        network: &mut LSTMNetwork<T>,
        task: &K,
        input_features: usize,
        learning_rate: T,
    ) -> Result<FrozenRollout<T>>
    where
        K: MetaTrainingTask<T> + ?Sized,
    {
        if self.config.unroll_steps == 0 {
            return Err(OptimError::InvalidConfig(
                "unroll_steps must be greater than 0".to_string(),
            ));
        }
        self.prepare_network(network, task.dimension())?;
        network.reset_state();

        let transform = network.output_projection.output_transform;
        let mut params = task.initial_parameters();
        let mut history: VecDeque<Array1<T>> = VecDeque::new();
        let mut losses: VecDeque<T> = VecDeque::new();

        let mut inputs = Vec::with_capacity(self.config.unroll_steps);
        let mut learning_rates = Vec::with_capacity(self.config.unroll_steps);
        let mut adaptive_scales = Vec::with_capacity(self.config.unroll_steps);

        for _ in 0..self.config.unroll_steps {
            let gradient = task.gradient(&params);
            let loss = task.loss(&params);

            let recent: Vec<&Array1<T>> = if history.len() >= FEATURE_HISTORY {
                history.iter().rev().take(FEATURE_HISTORY).collect()
            } else {
                Vec::new()
            };
            let loss_features = if losses.len() >= 2 {
                let current = losses[losses.len() - 1];
                let previous = losses[losses.len() - 2];
                let eps: T = scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(T::zero);
                let ratio = if previous.abs() > eps {
                    current / previous
                } else {
                    T::one()
                };
                Some(vec![current - previous, ratio])
            } else {
                None
            };

            let features =
                build_lstm_features(&gradient, &recent, loss_features.as_deref(), input_features)?;

            let grad_norm = gradient
                .iter()
                .map(|&g| g * g)
                .fold(T::zero(), |a, b| a + b)
                .sqrt();
            let adaptive_scale = T::one() / (T::one() + grad_norm);

            let output = network.forward(&features)?;
            let update = match transform {
                OutputTransform::Identity => output.clone(),
                OutputTransform::Tanh | OutputTransform::LearnedNonlinear => {
                    output.mapv(|v| v.tanh())
                }
                OutputTransform::ScaledTanh { scale } => {
                    let s: T = scirs2_core::numeric::NumCast::from(scale).unwrap_or_else(T::one);
                    output.mapv(|v| v.tanh() * s)
                }
                OutputTransform::AdaptiveScale => output.mapv(|v| v * adaptive_scale),
            }
            .mapv(|v| v * learning_rate);

            for j in 0..params.len().min(update.len()) {
                params[j] = params[j] - update[j];
            }

            history.push_back(gradient);
            while history.len() > FEATURE_HISTORY {
                history.pop_front();
            }
            losses.push_back(loss);
            while losses.len() > FEATURE_HISTORY {
                losses.pop_front();
            }

            inputs.push(features);
            learning_rates.push(learning_rate);
            adaptive_scales.push(adaptive_scale);
        }

        Ok(FrozenRollout {
            inputs,
            learning_rates,
            adaptive_scales,
            initial_parameters: task.initial_parameters(),
        })
    }

    /// Mean weighted meta-loss of `tasks` without changing any weight.
    ///
    /// # Errors
    /// Returns `Err` when `tasks` is empty, `weights` is shorter than `tasks`, or
    /// any rollout fails.
    pub fn evaluate<K>(&self, network: &mut LSTMNetwork<T>, tasks: &[K], weights: &[T]) -> Result<T>
    where
        K: MetaTrainingTask<T>,
    {
        Self::validate_batch(tasks, weights)?;
        let input_features = Self::input_width(network)?;
        let learning_rate = self.rollout_learning_rate();

        let mut total = T::zero();
        let mut total_weight = T::zero();
        for (task, &weight) in tasks.iter().zip(weights.iter()) {
            let rollout = self.capture_rollout(network, task, input_features, learning_rate)?;
            let loss = rollout_loss_frozen(network, task, &rollout)?;
            total = total + weight * loss;
            total_weight = total_weight + weight;
        }
        if total_weight <= T::zero() {
            return Err(OptimError::InvalidConfig(
                "task weights must sum to a positive value".to_string(),
            ));
        }
        Ok(total / total_weight)
    }

    /// One meta-training step: unroll, backpropagate through time, Adam-update.
    ///
    /// Returns the mean weighted meta-loss measured *before* the update (so a
    /// decreasing sequence of return values means the controller is improving).
    ///
    /// # Errors
    /// Returns `Err` when `tasks` is empty, the tasks disagree on dimension,
    /// `weights` is shorter than `tasks`, or any rollout fails.
    pub fn meta_step<K>(
        &mut self,
        network: &mut LSTMNetwork<T>,
        tasks: &[K],
        weights: &[T],
    ) -> Result<T>
    where
        K: MetaTrainingTask<T>,
    {
        Self::validate_batch(tasks, weights)?;
        let dimension = tasks[0].dimension();
        if let Some(bad) = tasks.iter().position(|t| t.dimension() != dimension) {
            return Err(OptimError::InvalidConfig(format!(
                "task {bad} has dimension {} but task 0 has {dimension}; a single \
                 meta-step needs a shared optimizee dimension because the \
                 controller's output projection is sized to it",
                tasks[bad].dimension()
            )));
        }
        self.prepare_network(network, dimension)?;

        let input_features = Self::input_width(network)?;
        let learning_rate = self.rollout_learning_rate();

        let mut accumulated: Option<NetworkGradients<T>> = None;
        let mut total_loss = T::zero();
        let mut total_weight = T::zero();

        for (task, &weight) in tasks.iter().zip(weights.iter()) {
            let rollout = self.capture_rollout(network, task, input_features, learning_rate)?;
            let (loss, grads) = meta_gradient_frozen(network, task, &rollout)?;
            total_loss = total_loss + weight * loss;
            total_weight = total_weight + weight;
            match accumulated.as_mut() {
                Some(acc) => acc.add_assign_weighted(&grads, weight),
                None => {
                    let mut scaled = grads;
                    scaled.scale_by(weight);
                    accumulated = Some(scaled);
                }
            }
        }

        if total_weight <= T::zero() {
            return Err(OptimError::InvalidConfig(
                "task weights must sum to a positive value".to_string(),
            ));
        }
        let mut gradients = match accumulated {
            Some(g) => g,
            None => {
                return Err(OptimError::InsufficientData(
                    "no task produced a meta-gradient".to_string(),
                ))
            }
        };
        gradients.scale_by(T::one() / total_weight);

        // Global-norm clipping keeps an early, badly-scaled meta-gradient from
        // destroying the controller in one step.
        if self.config.gradient_clip > 0.0 {
            let clip: T = scirs2_core::numeric::NumCast::from(self.config.gradient_clip)
                .unwrap_or_else(T::one);
            let norm = gradients.l2_norm();
            if norm > clip && norm > T::zero() {
                gradients.scale_by(clip / norm);
            }
        }

        let flat_gradient = gradients.to_vec();
        self.apply_adam(network, &flat_gradient)?;
        self.last_gradient = flat_gradient;

        let mean_loss = total_loss / total_weight;
        self.loss_history.push_back(mean_loss);
        while self.loss_history.len() > 1024 {
            self.loss_history.pop_front();
        }
        Ok(mean_loss)
    }

    /// Apply one Adam step to the flattened controller weights.
    fn apply_adam(&mut self, network: &mut LSTMNetwork<T>, gradient: &[T]) -> Result<()> {
        let mut params = flatten_parameters(network);
        if params.len() != gradient.len() {
            return Err(OptimError::ComputationError(format!(
                "meta-gradient has {} entries but the controller has {}",
                gradient.len(),
                params.len()
            )));
        }
        let state = self
            .adam
            .get_or_insert_with(|| AdamState::new(params.len()));
        if state.first_moment.len() != params.len() {
            *state = AdamState::new(params.len());
        }
        state.step += 1;

        let lr: T = scirs2_core::numeric::NumCast::from(self.config.meta_learning_rate)
            .unwrap_or_else(T::zero);
        let beta1: T = scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(T::zero);
        let beta2: T = scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(T::zero);
        let eps: T = scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(T::zero);
        let bias1: T = scirs2_core::numeric::NumCast::from(1.0 - 0.9_f64.powi(state.step as i32))
            .unwrap_or_else(T::one);
        let bias2: T = scirs2_core::numeric::NumCast::from(1.0 - 0.999_f64.powi(state.step as i32))
            .unwrap_or_else(T::one);

        for i in 0..params.len() {
            let g = gradient[i];
            state.first_moment[i] = beta1 * state.first_moment[i] + (T::one() - beta1) * g;
            state.second_moment[i] = beta2 * state.second_moment[i] + (T::one() - beta2) * g * g;
            let m_hat = state.first_moment[i] / bias1;
            let v_hat = state.second_moment[i] / bias2;
            params[i] = params[i] - lr * m_hat / (v_hat.sqrt() + eps);
        }

        set_parameters(network, &params)
    }

    /// Learning rate used inside a rollout.
    ///
    /// A rollout must not depend on the shared adaptive learning-rate controller
    /// (that object carries per-run state and would make the recorded rollout
    /// non-reproducible), so meta-training uses a fixed inner step size of 1.
    /// The controller's own `OutputTransform` — `ScaledTanh { scale: 0.1 }` by
    /// default — already bounds each step, which is what makes 1 the right
    /// neutral choice rather than an arbitrary one.
    fn rollout_learning_rate(&self) -> T {
        T::one()
    }

    /// Input width the controller's first layer expects.
    fn input_width(network: &LSTMNetwork<T>) -> Result<usize> {
        match network.layers.first() {
            Some(layer) => Ok(layer.weight_ih.ncols()),
            None => Err(OptimError::InvalidConfig(
                "the controller has no LSTM layers".to_string(),
            )),
        }
    }

    fn validate_batch<K>(tasks: &[K], weights: &[T]) -> Result<()> {
        if tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "meta-training needs at least one task".to_string(),
            ));
        }
        if weights.len() < tasks.len() {
            return Err(OptimError::InvalidConfig(format!(
                "got {} tasks but only {} weights",
                tasks.len(),
                weights.len()
            )));
        }
        Ok(())
    }
}
