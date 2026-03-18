//! Online MAML for continuous task stream meta-learning.
//!
//! This module implements Online Model-Agnostic Meta-Learning (MAML) that
//! operates on a continuous stream of tasks, maintaining a task buffer with
//! staleness-weighted meta-updates. Unlike batch MAML which requires all tasks
//! upfront, Online MAML processes tasks as they arrive and applies exponential
//! decay to older tasks to prioritize recent experience.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Online MAML.
///
/// Controls the task buffer size, staleness decay, batch size for meta-updates,
/// and the inner/outer learning rates for bi-level optimization.
#[derive(Debug, Clone)]
pub struct OnlineMAMLConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum number of tasks kept in the rolling buffer.
    pub buffer_size: usize,
    /// Exponential decay factor applied to older tasks (0 < decay <= 1).
    pub staleness_decay: T,
    /// Number of tasks sampled per meta-update.
    pub online_batch_size: usize,
    /// Learning rate for the inner (task-level) adaptation loop.
    pub inner_lr: T,
    /// Learning rate for the outer (meta) parameter update.
    pub outer_lr: T,
    /// Number of gradient-descent steps in the inner loop.
    pub inner_steps: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for OnlineMAMLConfig<T> {
    fn default() -> Self {
        Self {
            buffer_size: 10,
            staleness_decay: T::from(0.95).unwrap_or_else(|| T::one()),
            online_batch_size: 5,
            inner_lr: T::from(0.01).unwrap_or_else(|| T::one()),
            outer_lr: T::from(0.001).unwrap_or_else(|| T::one()),
            inner_steps: 5,
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> OnlineMAMLConfig<T> {
    /// Set the buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the staleness decay factor.
    pub fn staleness_decay(mut self, decay: T) -> Self {
        self.staleness_decay = decay;
        self
    }

    /// Set the online batch size.
    pub fn online_batch_size(mut self, size: usize) -> Self {
        self.online_batch_size = size;
        self
    }

    /// Set the inner learning rate.
    pub fn inner_lr(mut self, lr: T) -> Self {
        self.inner_lr = lr;
        self
    }

    /// Set the outer learning rate.
    pub fn outer_lr(mut self, lr: T) -> Self {
        self.outer_lr = lr;
        self
    }

    /// Set the number of inner-loop steps.
    pub fn inner_steps(mut self, steps: usize) -> Self {
        self.inner_steps = steps;
        self
    }
}

// ---------------------------------------------------------------------------
// Task representation
// ---------------------------------------------------------------------------

/// A task received in the online stream.
///
/// Each task carries its own parameter snapshot, accumulated gradients, a
/// timestamp indicating when it was received, and the loss achieved.
#[derive(Debug, Clone)]
pub struct OnlineTask<T: Float + Debug + Send + Sync + 'static> {
    /// Unique identifier for this task.
    pub task_id: String,
    /// Parameter snapshot associated with the task.
    pub parameters: Array1<T>,
    /// Gradients collected during the task (one per inner step or evaluation).
    pub gradients: Vec<Array1<T>>,
    /// Logical timestamp (monotonically increasing counter).
    pub timestamp: usize,
    /// Loss value observed for this task.
    pub loss: T,
}

// ---------------------------------------------------------------------------
// Adaptation record
// ---------------------------------------------------------------------------

/// Record of a single task adaptation episode.
#[derive(Debug, Clone)]
pub struct AdaptationRecord<T: Float + Debug + Send + Sync + 'static> {
    /// Task that was adapted to.
    pub task_id: String,
    /// Loss before inner-loop adaptation.
    pub pre_adaptation_loss: T,
    /// Loss after inner-loop adaptation.
    pub post_adaptation_loss: T,
    /// Number of inner-loop steps executed.
    pub num_steps: usize,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing an [`OnlineMAML`] instance with a fluent API.
#[derive(Debug)]
pub struct OnlineMAMLBuilder<T: Float + Debug + Send + Sync + 'static> {
    config: OnlineMAMLConfig<T>,
    initial_params: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static + ScalarOperand> OnlineMAMLBuilder<T> {
    /// Create a new builder with the given initial meta-parameters.
    pub fn new(initial_params: Array1<T>) -> Self {
        Self {
            config: OnlineMAMLConfig::default(),
            initial_params,
        }
    }

    /// Set the buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set the staleness decay.
    pub fn staleness_decay(mut self, decay: T) -> Self {
        self.config.staleness_decay = decay;
        self
    }

    /// Set the online batch size.
    pub fn online_batch_size(mut self, size: usize) -> Self {
        self.config.online_batch_size = size;
        self
    }

    /// Set the inner learning rate.
    pub fn inner_lr(mut self, lr: T) -> Self {
        self.config.inner_lr = lr;
        self
    }

    /// Set the outer learning rate.
    pub fn outer_lr(mut self, lr: T) -> Self {
        self.config.outer_lr = lr;
        self
    }

    /// Set the number of inner steps.
    pub fn inner_steps(mut self, steps: usize) -> Self {
        self.config.inner_steps = steps;
        self
    }

    /// Build the [`OnlineMAML`] instance.
    pub fn build(self) -> OnlineMAML<T> {
        OnlineMAML {
            config: self.config,
            task_buffer: VecDeque::new(),
            meta_parameters: self.initial_params,
            step_count: 0,
            adaptation_history: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Core Online MAML
// ---------------------------------------------------------------------------

/// Online Model-Agnostic Meta-Learning.
///
/// Maintains a rolling buffer of recent tasks and performs staleness-weighted
/// meta-updates. As new tasks arrive they are added to the buffer; old tasks
/// are evicted when capacity is exceeded. Meta-updates sample a batch from
/// the buffer, run inner-loop adaptation for each task, then compute an
/// outer (meta) gradient weighted by how recent each task is.
#[derive(Debug)]
pub struct OnlineMAML<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration governing buffer size, learning rates, etc.
    config: OnlineMAMLConfig<T>,
    /// Rolling buffer of recently received tasks.
    task_buffer: VecDeque<OnlineTask<T>>,
    /// Current meta-parameters (shared initialisation).
    meta_parameters: Array1<T>,
    /// Number of meta-update steps performed so far.
    step_count: usize,
    /// History of adaptation episodes for analytics.
    adaptation_history: Vec<AdaptationRecord<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static + ScalarOperand> OnlineMAML<T> {
    /// Create a new `OnlineMAML` with the given config and initial parameters.
    pub fn new(config: OnlineMAMLConfig<T>, initial_params: Array1<T>) -> Self {
        Self {
            config,
            task_buffer: VecDeque::new(),
            meta_parameters: initial_params,
            step_count: 0,
            adaptation_history: Vec::new(),
        }
    }

    /// Return a builder initialised with the given meta-parameters.
    pub fn builder(initial_params: Array1<T>) -> OnlineMAMLBuilder<T> {
        OnlineMAMLBuilder::new(initial_params)
    }

    /// Get the current meta-parameters.
    pub fn meta_parameters(&self) -> &Array1<T> {
        &self.meta_parameters
    }

    /// Get the number of tasks currently in the buffer.
    pub fn buffer_len(&self) -> usize {
        self.task_buffer.len()
    }

    /// Get the number of meta-update steps executed so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the adaptation history.
    pub fn adaptation_history(&self) -> &[AdaptationRecord<T>] {
        &self.adaptation_history
    }

    // -----------------------------------------------------------------
    // Task management
    // -----------------------------------------------------------------

    /// Receive a new task into the buffer.
    ///
    /// If the buffer is at capacity the oldest task is evicted first.
    pub fn receive_task(&mut self, task: OnlineTask<T>) {
        if self.task_buffer.len() >= self.config.buffer_size {
            self.task_buffer.pop_front();
        }
        self.task_buffer.push_back(task);
    }

    // -----------------------------------------------------------------
    // Meta-update
    // -----------------------------------------------------------------

    /// Perform one online meta-update using the buffered tasks.
    ///
    /// # Algorithm
    /// 1. Select up to `online_batch_size` most-recent tasks from the buffer.
    /// 2. For each selected task run an inner-loop adaptation starting from the
    ///    current `meta_parameters`.
    /// 3. Compute the outer gradient as the staleness-weighted mean of the
    ///    difference between the meta-parameters and the adapted parameters.
    /// 4. Update `meta_parameters` with the outer gradient.
    ///
    /// # Errors
    /// Returns `OptimError::InsufficientData` if the buffer is empty.
    pub fn online_update(&mut self) -> Result<Array1<T>> {
        if self.task_buffer.is_empty() {
            return Err(OptimError::InsufficientData(
                "Task buffer is empty; cannot perform meta-update".into(),
            ));
        }

        let current_step = self.step_count;
        let batch_size = self.config.online_batch_size.min(self.task_buffer.len());

        // Take the most-recent `batch_size` tasks.
        let start_idx = self.task_buffer.len().saturating_sub(batch_size);
        let batch: Vec<&OnlineTask<T>> = self.task_buffer.iter().skip(start_idx).collect();

        let param_dim = self.meta_parameters.len();
        let mut meta_gradient = Array1::<T>::zeros(param_dim);
        let mut total_weight = T::zero();

        for task in &batch {
            // Staleness weight: decay ^ (current_step - task.timestamp)
            let age = current_step.saturating_sub(task.timestamp);
            let weight = self.config.staleness_decay.powi(age as i32);

            // Inner-loop adaptation
            let adapted = self.inner_adapt(&task.gradients)?;

            // Outer gradient contribution: (meta_params - adapted_params)
            // We want to move meta_parameters *towards* adapted, so gradient
            // direction is (adapted - meta_parameters).
            let mut diff = Array1::<T>::zeros(param_dim);
            Zip::from(&mut diff)
                .and(&adapted)
                .and(&self.meta_parameters)
                .for_each(|d, &a, &m| {
                    *d = a - m;
                });

            // Pre/post adaptation losses for the record
            let pre_loss = task.loss;
            let post_loss = self.approximate_post_loss(&task.gradients);

            self.adaptation_history.push(AdaptationRecord {
                task_id: task.task_id.clone(),
                pre_adaptation_loss: pre_loss,
                post_adaptation_loss: post_loss,
                num_steps: self.config.inner_steps.min(task.gradients.len()),
            });

            // Accumulate weighted gradient
            Zip::from(&mut meta_gradient).and(&diff).for_each(|mg, &d| {
                *mg = *mg + weight * d;
            });
            total_weight = total_weight + weight;
        }

        // Normalise by total weight
        if total_weight > T::zero() {
            meta_gradient.mapv_inplace(|v| v / total_weight);
        }

        // Outer update
        let lr = self.config.outer_lr;
        Zip::from(&mut self.meta_parameters)
            .and(&meta_gradient)
            .for_each(|p, &g| {
                *p = *p + lr * g;
            });

        self.step_count += 1;

        Ok(self.meta_parameters.clone())
    }

    /// Adapt meta-parameters to a specific task using its gradients.
    ///
    /// Runs the inner-loop gradient descent starting from the current
    /// `meta_parameters` and returns the adapted parameters.
    ///
    /// # Errors
    /// Returns `OptimError::InsufficientData` if `task_gradients` is empty.
    pub fn adapt_to_task(&self, task_gradients: &[Array1<T>]) -> Result<Array1<T>> {
        if task_gradients.is_empty() {
            return Err(OptimError::InsufficientData(
                "No gradients provided for task adaptation".into(),
            ));
        }
        self.inner_adapt(task_gradients)
    }

    /// Return the average adaptation efficiency (improvement ratio) over
    /// recorded history.
    ///
    /// Efficiency for a single record is defined as:
    ///   `(pre_loss - post_loss) / (pre_loss + epsilon)`
    ///
    /// The overall efficiency is the mean across all records.
    /// Returns `T::zero()` when there is no history.
    pub fn get_adaptation_efficiency(&self) -> T {
        if self.adaptation_history.is_empty() {
            return T::zero();
        }
        let epsilon = T::from(1e-10).unwrap_or_else(|| T::epsilon());
        let n = T::from(self.adaptation_history.len()).unwrap_or_else(|| T::one());
        let total: T = self.adaptation_history.iter().fold(T::zero(), |acc, rec| {
            let improvement = rec.pre_adaptation_loss - rec.post_adaptation_loss;
            let ratio = improvement / (rec.pre_adaptation_loss.abs() + epsilon);
            acc + ratio
        });
        total / n
    }

    // -----------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------

    /// Run inner-loop gradient descent using the supplied per-step gradients.
    fn inner_adapt(&self, gradients: &[Array1<T>]) -> Result<Array1<T>> {
        let steps = self.config.inner_steps.min(gradients.len());
        let mut params = self.meta_parameters.clone();
        let lr = self.config.inner_lr;

        for grad in gradients.iter().take(steps) {
            if grad.len() != params.len() {
                return Err(OptimError::ComputationError(format!(
                    "Gradient dimension {} does not match parameter dimension {}",
                    grad.len(),
                    params.len()
                )));
            }
            Zip::from(&mut params).and(grad).for_each(|p, &g| {
                *p = *p - lr * g;
            });
        }
        Ok(params)
    }

    /// Approximate the post-adaptation loss from gradient norms.
    ///
    /// Uses a simple heuristic: post_loss ~ mean gradient norm after the last
    /// inner step (smaller gradients indicate closer to optimum).
    fn approximate_post_loss(&self, gradients: &[Array1<T>]) -> T {
        let steps = self.config.inner_steps.min(gradients.len());
        if steps == 0 {
            return T::zero();
        }
        let last_grad = &gradients[steps - 1];
        let norm_sq = last_grad.iter().fold(T::zero(), |acc, &g| acc + g * g);
        norm_sq.sqrt()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_task(id: &str, dim: usize, timestamp: usize, loss: f64) -> OnlineTask<f64> {
        let params = Array1::from_elem(dim, 1.0);
        let gradients: Vec<Array1<f64>> = (0..5)
            .map(|i| Array1::from_elem(dim, 0.1 * (i as f64 + 1.0)))
            .collect();
        OnlineTask {
            task_id: id.to_string(),
            parameters: params,
            gradients,
            timestamp,
            loss,
        }
    }

    #[test]
    fn test_online_maml_receive_task() {
        let init = Array1::from_elem(4, 0.0_f64);
        let config = OnlineMAMLConfig::default().buffer_size(5);
        let mut maml = OnlineMAML::new(config, init);

        let task = make_task("t1", 4, 0, 1.0);
        maml.receive_task(task);

        assert_eq!(maml.buffer_len(), 1);
        assert_eq!(
            maml.task_buffer.back().map(|t| t.task_id.as_str()),
            Some("t1")
        );
    }

    #[test]
    fn test_online_maml_buffer_eviction() {
        let init = Array1::from_elem(4, 0.0_f64);
        let config = OnlineMAMLConfig::default().buffer_size(3);
        let mut maml = OnlineMAML::new(config, init);

        for i in 0..5 {
            let task = make_task(&format!("t{}", i), 4, i, 1.0);
            maml.receive_task(task);
        }

        // Buffer should contain only the 3 most-recent tasks.
        assert_eq!(maml.buffer_len(), 3);
        let ids: Vec<&str> = maml
            .task_buffer
            .iter()
            .map(|t| t.task_id.as_str())
            .collect();
        assert_eq!(ids, vec!["t2", "t3", "t4"]);
    }

    #[test]
    fn test_online_update_basic() {
        let dim = 4;
        let init = Array1::from_elem(dim, 0.0_f64);
        let config = OnlineMAMLConfig::default()
            .buffer_size(10)
            .online_batch_size(2)
            .inner_lr(0.01)
            .outer_lr(0.1)
            .inner_steps(3);
        let mut maml = OnlineMAML::new(config, init);

        // Add two tasks
        maml.receive_task(make_task("a", dim, 0, 2.0));
        maml.receive_task(make_task("b", dim, 1, 1.5));

        let updated = maml.online_update();
        assert!(updated.is_ok());
        let updated = updated.expect("online_update should succeed");

        // The meta-parameters should have moved from zero.
        let moved = updated.iter().any(|&v| v.abs() > 1e-12);
        assert!(moved, "Meta-parameters should change after an update");
        assert_eq!(maml.step_count(), 1);
    }

    #[test]
    fn test_adapt_to_task() {
        let dim = 4;
        let init = Array1::from_elem(dim, 1.0_f64);
        let config = OnlineMAMLConfig::default().inner_lr(0.1).inner_steps(3);
        let maml = OnlineMAML::new(config, init.clone());

        let grads: Vec<Array1<f64>> = (0..5).map(|_| Array1::from_elem(dim, 0.5)).collect();

        let adapted = maml.adapt_to_task(&grads);
        assert!(adapted.is_ok());
        let adapted = adapted.expect("adapt_to_task should succeed");

        // After 3 steps of lr=0.1 with gradient=0.5:
        // param = 1.0 - 3 * 0.1 * 0.5 = 0.85
        for &v in adapted.iter() {
            assert!((v - 0.85).abs() < 1e-10, "Expected 0.85 but got {}", v);
        }

        // Empty gradients should error.
        let empty_result = maml.adapt_to_task(&[]);
        assert!(empty_result.is_err());
    }

    #[test]
    fn test_adaptation_efficiency() {
        let dim = 4;
        let init = Array1::from_elem(dim, 0.0_f64);
        let config = OnlineMAMLConfig::default()
            .buffer_size(10)
            .online_batch_size(3)
            .inner_lr(0.01)
            .outer_lr(0.1)
            .inner_steps(3);
        let mut maml = OnlineMAML::new(config, init);

        // No history yet
        assert_eq!(maml.get_adaptation_efficiency(), 0.0);

        // Add tasks and run an update to generate history
        for i in 0..3 {
            maml.receive_task(make_task(&format!("t{}", i), dim, i, 2.0 + i as f64));
        }
        let _ = maml.online_update();

        assert!(!maml.adaptation_history().is_empty());
        // Efficiency should be a finite number
        let eff = maml.get_adaptation_efficiency();
        assert!(eff.is_finite(), "Efficiency should be finite, got {}", eff);
    }
}
