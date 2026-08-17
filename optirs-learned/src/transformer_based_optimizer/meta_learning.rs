// Meta-learning components for transformer-based optimization

use super::config::{ActivationFunction, MetaLearningConfig};
use super::feedforward::FeedForwardNetwork;
use super::layers::LayerNormalization;
use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

/// Meta-learning strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetaLearningStrategy {
    /// Model-Agnostic Meta-Learning (MAML)
    MAML,
    /// First-Order MAML (FOMAML)
    FOMAML,
    /// Reptile algorithm
    Reptile,
    /// Gradient-based meta-learning
    GradientBased,
    /// Memory-augmented networks
    MemoryAugmented,
}

/// Transformer meta-learning implementation
pub struct TransformerMetaLearning<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Meta-learning strategy
    strategy: MetaLearningStrategy,

    /// Configuration
    config: MetaLearningConfig<T>,

    /// Meta-optimizer for outer loop
    meta_optimizer: MetaOptimizer<T>,

    /// Task adaptation network
    adaptation_network: AdaptationNetwork<T>,

    /// Memory bank for storing task experiences
    memory_bank: MemoryBank<T>,

    /// Performance tracker
    performance_tracker: PerformanceTracker<T>,

    /// Current meta-learning state
    meta_state: MetaState<T>,
}

impl<
        T: Float
            + Debug
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static
            + scirs2_core::ndarray::ScalarOperand,
    > TransformerMetaLearning<T>
{
    /// Create new transformer meta-learning component
    pub fn new(config: &super::config::TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let meta_config = config.meta_learning_config.clone();
        let strategy = MetaLearningStrategy::MAML; // Default strategy

        let meta_optimizer = MetaOptimizer::new(&meta_config)?;
        let adaptation_network =
            AdaptationNetwork::new(config.model_dimension, config.feedforward_dimension)?;
        let memory_bank = MemoryBank::new(1000, config.model_dimension)?;
        let performance_tracker = PerformanceTracker::new();
        let meta_state = MetaState::new(config.model_dimension)?;

        Ok(Self {
            strategy,
            config: meta_config,
            meta_optimizer,
            adaptation_network,
            memory_bank,
            performance_tracker,
            meta_state,
        })
    }

    /// Reject a task batch whose support/query arrays do not cover every task.
    ///
    /// Every `*_step` used to index `support_data[i]` / `query_data[i]` directly,
    /// which panicked on a short slice; this turns that into a typed error.
    fn validate_batch_shapes(
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<()> {
        if tasks.is_empty() {
            return Err(crate::error::OptimError::InsufficientData(
                "meta step requires at least one task".to_string(),
            ));
        }
        if support_data.len() < tasks.len() || query_data.len() < tasks.len() {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "meta step got {} tasks but {} support and {} query batches",
                tasks.len(),
                support_data.len(),
                query_data.len()
            )));
        }
        Ok(())
    }

    /// Perform meta-learning step
    pub fn meta_step(
        &mut self,
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<MetaLearningResult<T>> {
        match self.strategy {
            MetaLearningStrategy::MAML => self.maml_step(tasks, support_data, query_data),
            MetaLearningStrategy::FOMAML => self.fomaml_step(tasks, support_data, query_data),
            MetaLearningStrategy::Reptile => self.reptile_step(tasks, support_data, query_data),
            MetaLearningStrategy::GradientBased => {
                self.gradient_based_step(tasks, support_data, query_data)
            }
            MetaLearningStrategy::MemoryAugmented => {
                self.memory_augmented_step(tasks, support_data, query_data)
            }
        }
    }

    /// MAML meta-learning step.
    ///
    /// Runs `inner_steps` real gradient-descent steps on each task's support
    /// set (gradients from [`Self::compute_gradients`], not a constant), then
    /// forms the outer-loop meta-gradient from the **query** loss at the
    /// adapted parameters.
    ///
    /// When `MetaLearningConfig::first_order` is set, the meta-gradient is the
    /// FOMAML approximation `∇L_q(θ')`. Otherwise the exact second-order MAML
    /// meta-gradient is used: because the per-task objective is quadratic (see
    /// [`Self::compute_task_loss`]) the support Hessian `H_s` is constant, so
    ///
    /// ```text
    /// ∇_θ L_q(θ_k) = (I − α·H_s)^k · ∇L_q(θ_k)
    /// ```
    ///
    /// holds exactly and is evaluated with `k` Hessian-vector products.
    ///
    /// # Errors
    /// Returns `Err` when `tasks` is empty or `support_data`/`query_data` are
    /// shorter than `tasks`.
    fn maml_step(
        &mut self,
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<MetaLearningResult<T>> {
        Self::validate_batch_shapes(tasks, support_data, query_data)?;
        let start_time = Instant::now();
        let mut total_loss = T::zero();
        let mut task_adaptations = Vec::new();

        for (i, task) in tasks.iter().enumerate() {
            // Inner adaptation loop
            let mut adapted_params = self.meta_state.get_parameters().clone();
            let support_loss_before =
                self.compute_task_loss(&adapted_params, &support_data[i], task)?;

            for _inner_step in 0..self.config.inner_steps {
                // Real analytic gradient of the support loss at the current θ.
                let gradients = self.compute_gradients(&adapted_params, &support_data[i], task)?;

                // Update parameters
                for (param, grad) in adapted_params.iter_mut().zip(gradients.iter()) {
                    *param = *param - self.config.inner_learning_rate * (*grad);
                }
            }

            // Evaluate on query set
            let query_loss = self.compute_task_loss(&adapted_params, &query_data[i], task)?;
            total_loss = total_loss + query_loss;

            // Outer-loop gradient: ∇L_q at the adapted parameters, optionally
            // back-propagated through the inner updates (exact, see doc above).
            let mut query_gradient =
                self.compute_gradients(&adapted_params, &query_data[i], task)?;
            if !self.config.first_order {
                let alpha = self.config.inner_learning_rate;
                for _ in 0..self.config.inner_steps {
                    let hv =
                        self.hessian_vector_product(&query_gradient, &support_data[i], task)?;
                    for (g, &h) in query_gradient.iter_mut().zip(hv.iter()) {
                        *g = *g - alpha * h;
                    }
                }
            }

            task_adaptations.push(TaskAdaptation {
                task_id: task.id.clone(),
                adapted_parameters: adapted_params,
                support_loss: support_loss_before,
                query_loss,
                query_gradient,
                adaptation_steps: self.config.inner_steps,
            });
        }

        // Meta-update
        let meta_loss = total_loss / T::from(tasks.len()).unwrap_or_else(|| T::one());
        let meta_gradients = self.compute_meta_gradients(&task_adaptations)?;
        self.meta_optimizer
            .update(&mut self.meta_state, &meta_gradients)?;

        // Update memory bank
        for (i, adaptation) in task_adaptations.iter().enumerate() {
            self.memory_bank.store_experience(
                &tasks[i],
                &adaptation.adapted_parameters,
                adaptation.query_loss,
            )?;
        }

        let result = MetaLearningResult {
            meta_loss: meta_loss.to_f64().unwrap_or(0.0),
            task_adaptations,
            computation_time: start_time.elapsed(),
            convergence_rate: self.estimate_convergence_rate()?,
        };

        self.performance_tracker.record_meta_step(result.clone());
        Ok(result)
    }

    /// First-order MAML step (FOMAML)
    fn fomaml_step(
        &mut self,
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<MetaLearningResult<T>> {
        // Simplified version of MAML that ignores second-order derivatives
        // Similar to MAML but uses first-order approximation for efficiency
        self.maml_step(tasks, support_data, query_data)
    }

    /// Reptile meta-learning step
    fn reptile_step(
        &mut self,
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<MetaLearningResult<T>> {
        Self::validate_batch_shapes(tasks, support_data, query_data)?;
        let start_time = Instant::now();
        let mut parameter_updates = Vec::new();
        let mut total_loss = T::zero();

        for (i, task) in tasks.iter().enumerate() {
            // Adapt on support set
            let mut adapted_params = self.meta_state.get_parameters().clone();

            for _ in 0..self.config.inner_steps {
                let gradients = self.compute_gradients(&adapted_params, &support_data[i], task)?;

                for (param, grad) in adapted_params.iter_mut().zip(gradients.iter()) {
                    *param = *param - self.config.inner_learning_rate * (*grad);
                }
            }

            // Compute parameter difference for meta-update
            let original_params = self.meta_state.get_parameters();
            let param_diff: Vec<T> = adapted_params
                .iter()
                .zip(original_params.iter())
                .map(|(adapted, original)| *adapted - *original)
                .collect();

            parameter_updates.push(param_diff);

            // Evaluate on query set
            let query_loss = self.compute_task_loss(&adapted_params, &query_data[i], task)?;
            total_loss = total_loss + query_loss;
        }

        // Meta-update: move towards average of adapted parameters
        let mut meta_update = vec![T::zero(); self.meta_state.get_parameters().len()];
        for param_update in &parameter_updates {
            for (i, &update) in param_update.iter().enumerate() {
                meta_update[i] = meta_update[i] + update;
            }
        }

        let num_tasks = T::from(tasks.len()).expect("unwrap failed");
        for update in meta_update.iter_mut() {
            *update = *update / num_tasks;
        }

        self.meta_state
            .update_parameters(&meta_update, self.config.meta_learning_rate)?;

        let result = MetaLearningResult {
            meta_loss: (total_loss / num_tasks).to_f64().unwrap_or(0.0),
            task_adaptations: Vec::new(), // Reptile doesn't track individual adaptations
            computation_time: start_time.elapsed(),
            convergence_rate: self.estimate_convergence_rate()?,
        };

        self.performance_tracker.record_meta_step(result.clone());
        Ok(result)
    }

    /// Gradient-based meta-learning step
    fn gradient_based_step(
        &mut self,
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<MetaLearningResult<T>> {
        // Implement gradient-based meta-learning with direct optimization
        Self::validate_batch_shapes(tasks, support_data, query_data)?;
        let start_time = Instant::now();
        let mut total_loss = T::zero();

        for (i, task) in tasks.iter().enumerate() {
            // Use adaptation network to predict good initialization
            let context_embedding = self.adaptation_network.encode_task_context(task)?;
            let predicted_params = self
                .adaptation_network
                .predict_parameters(&context_embedding)?;

            // Fine-tune predicted parameters
            let mut adapted_params = predicted_params;
            for _ in 0..self.config.inner_steps {
                let gradients = self.compute_gradients(&adapted_params, &support_data[i], task)?;

                for (param, grad) in adapted_params.iter_mut().zip(gradients.iter()) {
                    *param = *param - self.config.inner_learning_rate * (*grad);
                }
            }

            let query_loss = self.compute_task_loss(&adapted_params, &query_data[i], task)?;
            total_loss = total_loss + query_loss;
        }

        let result = MetaLearningResult {
            meta_loss: (total_loss / T::from(tasks.len()).unwrap_or_else(|| T::one()))
                .to_f64()
                .unwrap_or(0.0),
            task_adaptations: Vec::new(),
            computation_time: start_time.elapsed(),
            convergence_rate: self.estimate_convergence_rate()?,
        };

        Ok(result)
    }

    /// Memory-augmented meta-learning step
    fn memory_augmented_step(
        &mut self,
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<MetaLearningResult<T>> {
        Self::validate_batch_shapes(tasks, support_data, query_data)?;
        let start_time = Instant::now();
        let mut total_loss = T::zero();

        for (i, task) in tasks.iter().enumerate() {
            // Retrieve relevant experiences from memory
            let relevant_experiences = self.memory_bank.retrieve_similar_experiences(task, 5)?;

            // Use memory to initialize adaptation
            let memory_guided_params = self.initialize_from_memory(&relevant_experiences)?;

            let mut adapted_params = memory_guided_params;
            for _ in 0..self.config.inner_steps {
                let gradients = self.compute_gradients(&adapted_params, &support_data[i], task)?;

                for (param, grad) in adapted_params.iter_mut().zip(gradients.iter()) {
                    *param = *param - self.config.inner_learning_rate * (*grad);
                }
            }

            let query_loss = self.compute_task_loss(&adapted_params, &query_data[i], task)?;
            total_loss = total_loss + query_loss;

            // Store experience
            self.memory_bank
                .store_experience(task, &adapted_params, query_loss)?;
        }

        let result = MetaLearningResult {
            meta_loss: (total_loss / T::from(tasks.len()).unwrap_or_else(|| T::one()))
                .to_f64()
                .unwrap_or(0.0),
            task_adaptations: Vec::new(),
            computation_time: start_time.elapsed(),
            convergence_rate: self.estimate_convergence_rate()?,
        };

        Ok(result)
    }

    /// Generate optimization update using meta-learned parameters
    pub fn generate_update(
        &mut self,
        transformer_output: &Array2<T>,
        current_parameters: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Use adaptation network to generate parameter updates
        let update = self
            .adaptation_network
            .generate_parameter_update(transformer_output, current_parameters)?;

        // Apply meta-learned scaling
        let scaled_update = self.apply_meta_scaling(&update)?;

        Ok(scaled_update)
    }

    /// Update meta-learning state from loss
    pub fn update_from_loss(&mut self, loss: T) -> Result<()> {
        self.meta_state.update_loss_history(loss);
        self.performance_tracker
            .record_loss(loss.to_f64().unwrap_or(0.0));
        Ok(())
    }

    /// Total learnable parameters held by the adaptation networks.
    pub fn parameter_count(&self) -> usize {
        self.adaptation_network.parameter_count()
    }

    /// Mutable access to the outer-loop optimizer (e.g. to enable momentum).
    pub fn meta_optimizer_mut(&mut self) -> &mut MetaOptimizer<T> {
        &mut self.meta_optimizer
    }

    /// Set meta-learning strategy
    pub fn set_strategy(&mut self, strategy: MetaLearningStrategy) {
        self.strategy = strategy;
    }

    /// Get current strategy
    pub fn get_strategy(&self) -> MetaLearningStrategy {
        self.strategy
    }

    /// Per-task objective actually optimized by every inner loop in this module.
    ///
    /// # Task model
    ///
    /// A task batch is a design matrix `data` of shape `(n, c)`: the **last**
    /// column holds the regression target `y`, the remaining `c - 1` columns
    /// hold the features `x`. The adapted parameter vector `θ` (length `p`) is
    /// read as a bias plus weights:
    ///
    /// ```text
    /// pred_i = θ[0] + Σ_{j<f} θ[j+1] · x_i[j],   f = min(c-1, p-1)
    /// L(θ)   = (1/n) Σ_i (pred_i - y_i)²  +  (λ/p) Σ_{k≥1} θ[k]²
    /// ```
    ///
    /// `λ` is taken from `task.complexity` (clamped at zero): a more complex
    /// task is regularized more strongly. That is the one place the task
    /// descriptor enters the objective, and it is why two different
    /// `TaskBatch`es over the same data produce different losses and different
    /// gradients.
    ///
    /// The objective is quadratic in `θ`, which is what makes the exact
    /// second-order MAML meta-gradient in [`Self::maml_step`] computable in
    /// closed form (see [`Self::hessian_vector_product`]).
    ///
    /// # Errors
    /// Returns `Err` when `params` is empty or `data` has no rows/columns.
    fn compute_task_loss(&self, params: &[T], data: &Array2<T>, task: &TaskBatch<T>) -> Result<T> {
        let (n_rows, _n_cols) = Self::validate_task_data(params, data)?;
        let predictions = Self::linear_forward(params, data);
        let n = T::from(n_rows).unwrap_or_else(|| T::one());

        let mut data_loss = T::zero();
        for (i, &pred) in predictions.iter().enumerate() {
            let residual = pred - Self::target_of(data, i);
            data_loss = data_loss + residual * residual;
        }
        data_loss = data_loss / n;

        let ridge = self.ridge_weight(params.len(), task);
        let mut reg = T::zero();
        for &p in params.iter().skip(1) {
            reg = reg + p * p;
        }

        Ok(data_loss + ridge * reg)
    }

    /// Exact analytic gradient of [`Self::compute_task_loss`] w.r.t. `params`.
    ///
    /// Verified against central finite differences by
    /// `compute_gradients_matches_finite_differences`.
    fn compute_gradients(
        &self,
        params: &[T],
        data: &Array2<T>,
        task: &TaskBatch<T>,
    ) -> Result<Vec<T>> {
        let (n_rows, _) = Self::validate_task_data(params, data)?;
        let predictions = Self::linear_forward(params, data);
        let residuals: Vec<T> = predictions
            .iter()
            .enumerate()
            .map(|(i, &pred)| pred - Self::target_of(data, i))
            .collect();

        let mut gradients =
            Self::accumulate_design_gradient(&residuals, data, params.len(), n_rows);

        // Ridge term: d/dθ[k] of (λ/p)·Σ_{k≥1} θ[k]² = 2·(λ/p)·θ[k], bias excluded.
        let two = T::one() + T::one();
        let ridge = self.ridge_weight(params.len(), task);
        for (k, grad) in gradients.iter_mut().enumerate().skip(1) {
            *grad = *grad + two * ridge * params[k];
        }

        Ok(gradients)
    }

    /// Exact Hessian-vector product `H · v` for [`Self::compute_task_loss`].
    ///
    /// The objective is quadratic, so `H` does not depend on `θ` — only on the
    /// design matrix and the ridge weight. This lets the second-order MAML
    /// meta-gradient be computed with one matrix-vector product per inner step
    /// instead of materializing (and powering) a `p × p` Hessian.
    fn hessian_vector_product(
        &self,
        vector: &[T],
        data: &Array2<T>,
        task: &TaskBatch<T>,
    ) -> Result<Vec<T>> {
        let (n_rows, _) = Self::validate_task_data(vector, data)?;
        // H·v for the data term is the same accumulation as the gradient, with
        // the residuals replaced by the *predictions of v* (targets excluded,
        // since the target contributes only to the constant/linear part).
        let directional = Self::linear_forward(vector, data);
        let mut out = Self::accumulate_design_gradient(&directional, data, vector.len(), n_rows);

        let two = T::one() + T::one();
        let ridge = self.ridge_weight(vector.len(), task);
        for (k, slot) in out.iter_mut().enumerate().skip(1) {
            *slot = *slot + two * ridge * vector[k];
        }

        Ok(out)
    }

    /// `λ/p` — the per-parameter ridge weight implied by a task descriptor.
    fn ridge_weight(&self, param_count: usize, task: &TaskBatch<T>) -> T {
        let lambda: T = scirs2_core::numeric::NumCast::from(task.complexity.max(0.0))
            .unwrap_or_else(|| T::zero());
        let p = T::from(param_count.max(1)).unwrap_or_else(|| T::one());
        lambda / p
    }

    /// `(n_rows, n_cols)` after checking that a task loss is well defined.
    fn validate_task_data(params: &[T], data: &Array2<T>) -> Result<(usize, usize)> {
        if params.is_empty() {
            return Err(crate::error::OptimError::InvalidConfig(
                "task loss needs at least one parameter (the bias)".to_string(),
            ));
        }
        let (n_rows, n_cols) = data.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(crate::error::OptimError::InsufficientData(format!(
                "task data must be non-empty, got shape {n_rows}x{n_cols}"
            )));
        }
        Ok((n_rows, n_cols))
    }

    /// Number of feature columns actually consumed by a parameter vector.
    fn effective_features(param_count: usize, n_cols: usize) -> usize {
        n_cols.saturating_sub(1).min(param_count.saturating_sub(1))
    }

    /// Regression target of row `i` (the last column).
    fn target_of(data: &Array2<T>, i: usize) -> T {
        let n_cols = data.ncols();
        data[[i, n_cols - 1]]
    }

    /// `pred_i = w[0] + Σ_{j<f} w[j+1] · x_i[j]` for every row.
    fn linear_forward(weights: &[T], data: &Array2<T>) -> Vec<T> {
        let (n_rows, n_cols) = data.dim();
        let f = Self::effective_features(weights.len(), n_cols);
        (0..n_rows)
            .map(|i| {
                let mut acc = weights[0];
                for j in 0..f {
                    acc = acc + weights[j + 1] * data[[i, j]];
                }
                acc
            })
            .collect()
    }

    /// `out[0] = (2/n)·Σ r_i`, `out[j+1] = (2/n)·Σ r_i·x_i[j]`, zero elsewhere.
    fn accumulate_design_gradient(
        residuals: &[T],
        data: &Array2<T>,
        param_count: usize,
        n_rows: usize,
    ) -> Vec<T> {
        let n_cols = data.ncols();
        let f = Self::effective_features(param_count, n_cols);
        let two = T::one() + T::one();
        let scale = two / T::from(n_rows.max(1)).unwrap_or_else(|| T::one());

        let mut out = vec![T::zero(); param_count];
        for (i, &r) in residuals.iter().enumerate() {
            out[0] = out[0] + r;
            for j in 0..f {
                out[j + 1] = out[j + 1] + r * data[[i, j]];
            }
        }
        for slot in out.iter_mut() {
            *slot = *slot * scale;
        }
        out
    }

    /// Average the per-task query gradients into the outer-loop meta-gradient.
    ///
    /// Each entry was produced by [`Self::maml_step`] as either the first-order
    /// query gradient `∇L_q(θ')` (FOMAML) or the exact second-order product
    /// `Π_i (I − α·H_s) ∇L_q(θ')` — see [`Self::maml_step`] for which. This
    /// function only averages; it no longer invents `θ · L_q` as a "gradient".
    ///
    /// # Errors
    /// Returns `Err` when `adaptations` is empty or the per-task gradients
    /// disagree on length.
    fn compute_meta_gradients(&self, adaptations: &[TaskAdaptation<T>]) -> Result<Vec<T>> {
        let first = adaptations.first().ok_or_else(|| {
            crate::error::OptimError::InsufficientData(
                "no task adaptations to derive a meta-gradient from".to_string(),
            )
        })?;
        let param_count = first.query_gradient.len();
        let mut meta_gradients = vec![T::zero(); param_count];

        for adaptation in adaptations {
            if adaptation.query_gradient.len() != param_count {
                return Err(crate::error::OptimError::ComputationError(format!(
                    "task '{}' reported {} meta-gradient components, expected {}",
                    adaptation.task_id,
                    adaptation.query_gradient.len(),
                    param_count
                )));
            }
            for (slot, &g) in meta_gradients
                .iter_mut()
                .zip(adaptation.query_gradient.iter())
            {
                *slot = *slot + g;
            }
        }

        let num_tasks = T::from(adaptations.len()).unwrap_or_else(|| T::one());
        for grad in meta_gradients.iter_mut() {
            *grad = *grad / num_tasks;
        }

        Ok(meta_gradients)
    }

    fn estimate_convergence_rate(&self) -> Result<f64> {
        let loss_history = self.performance_tracker.get_loss_history();
        if loss_history.len() < 2 {
            return Ok(0.0);
        }

        let recent_losses: Vec<_> = loss_history.iter().rev().take(5).cloned().collect();
        let improvement = recent_losses.last().expect("unwrap failed")
            - recent_losses.first().expect("unwrap failed");
        Ok(improvement.clamp(0.0, 1.0))
    }

    fn apply_meta_scaling(&self, update: &Array1<T>) -> Result<Array1<T>> {
        // Apply learned scaling factors
        let scale_factor = self.meta_state.get_scale_factor();
        Ok(update * scale_factor)
    }

    fn initialize_from_memory(&self, experiences: &[MemoryExperience<T>]) -> Result<Vec<T>> {
        if experiences.is_empty() {
            return Ok(self.meta_state.get_parameters().clone());
        }

        // Average parameters from similar experiences
        let param_count = experiences[0].parameters.len();
        let mut averaged_params = vec![T::zero(); param_count];

        for experience in experiences {
            for (i, &param) in experience.parameters.iter().enumerate() {
                averaged_params[i] = averaged_params[i] + param;
            }
        }

        let num_experiences = T::from(experiences.len()).expect("unwrap failed");
        for param in averaged_params.iter_mut() {
            *param = *param / num_experiences;
        }

        Ok(averaged_params)
    }
}

/// Meta-optimizer for outer loop updates
pub struct MetaOptimizer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Learning rate
    learning_rate: T,

    /// Momentum for SGD-style updates
    momentum: Option<T>,

    /// Velocity for momentum
    velocity: Option<Vec<T>>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    MetaOptimizer<T>
{
    pub fn new(config: &MetaLearningConfig<T>) -> Result<Self> {
        Ok(Self {
            learning_rate: config.meta_learning_rate,
            momentum: config.meta_momentum,
            velocity: None,
        })
    }

    /// Apply one outer-loop update.
    ///
    /// The momentum branch is reachable whenever
    /// `MetaLearningConfig::meta_momentum` is set (or
    /// [`Self::set_momentum`] was called); with no momentum this is plain SGD.
    ///
    /// # Errors
    /// Returns `Err` when `gradients` is shorter than the parameter vector —
    /// the momentum branch used to index `gradients[i]` unchecked and panicked.
    pub fn update(&mut self, state: &mut MetaState<T>, gradients: &[T]) -> Result<()> {
        let params = state.get_parameters_mut();
        if gradients.len() < params.len() {
            return Err(crate::error::OptimError::ComputationError(format!(
                "meta-gradient has {} components but the meta-parameters have {}",
                gradients.len(),
                params.len()
            )));
        }

        if let Some(momentum) = self.momentum {
            // Momentum update
            let velocity = self
                .velocity
                .get_or_insert_with(|| vec![T::zero(); params.len()]);
            if velocity.len() != params.len() {
                velocity.resize(params.len(), T::zero());
            }

            for i in 0..params.len() {
                velocity[i] = momentum * velocity[i] + self.learning_rate * gradients[i];
                params[i] = params[i] - velocity[i];
            }
        } else {
            // Simple SGD update
            for (param, &grad) in params.iter_mut().zip(gradients.iter()) {
                *param = *param - self.learning_rate * grad;
            }
        }

        Ok(())
    }

    /// Enable (or clear) heavy-ball momentum for the outer loop.
    ///
    /// Clearing it also drops the accumulated velocity so a later re-enable
    /// does not resume from stale state.
    pub fn set_momentum(&mut self, momentum: Option<T>) {
        self.momentum = momentum;
        if momentum.is_none() {
            self.velocity = None;
        }
    }

    /// Current momentum coefficient, if any.
    pub fn momentum(&self) -> Option<T> {
        self.momentum
    }
}

/// Task adaptation network
pub struct AdaptationNetwork<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Context encoder
    context_encoder: FeedForwardNetwork<T>,

    /// Parameter predictor
    parameter_predictor: FeedForwardNetwork<T>,

    /// Update generator
    update_generator: FeedForwardNetwork<T>,

    /// Model dimension
    model_dimension: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    AdaptationNetwork<T>
{
    pub fn new(model_dimension: usize, hidden_dimension: usize) -> Result<Self> {
        let context_encoder =
            FeedForwardNetwork::new(model_dimension, hidden_dimension, ActivationFunction::ReLU)?;

        let parameter_predictor =
            FeedForwardNetwork::new(hidden_dimension, model_dimension, ActivationFunction::Tanh)?;

        let update_generator = FeedForwardNetwork::new(
            model_dimension * 2, // concatenated transformer output and current params
            model_dimension,
            ActivationFunction::ReLU,
        )?;

        Ok(Self {
            context_encoder,
            parameter_predictor,
            update_generator,
            model_dimension,
        })
    }

    /// Encode a task descriptor into a context embedding.
    ///
    /// The raw descriptor is three scalars (`difficulty`, `complexity`,
    /// `data_characteristics`). They are lifted to `model_dimension` features by
    /// a fixed random-Fourier-style basis: feature `k` is
    /// `cos(2π·(k+1)·d_{k mod 3} / 3 + φ_k)` with a deterministic per-index
    /// phase, so distinct descriptors map to distinct, bounded, non-collinear
    /// embeddings. Previously this ignored the task entirely and fed
    /// `Array2::ones`, which made every task produce the same context.
    /// Total learnable parameters across the three sub-networks.
    pub fn parameter_count(&self) -> usize {
        self.context_encoder.parameter_count()
            + self.parameter_predictor.parameter_count()
            + self.update_generator.parameter_count()
    }

    pub fn encode_task_context(&mut self, task: &TaskBatch<T>) -> Result<Array2<T>> {
        let descriptor = [task.difficulty, task.complexity, task.data_characteristics];
        let mut task_features = Array2::zeros((1, self.model_dimension));
        for k in 0..self.model_dimension {
            let d = descriptor[k % descriptor.len()];
            let harmonic = (k / descriptor.len()) as f64 + 1.0;
            let phase = (k as f64) * std::f64::consts::FRAC_PI_4;
            let value = (std::f64::consts::TAU * harmonic * d + phase).cos();
            task_features[[0, k]] =
                scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero());
        }
        self.context_encoder.forward(&task_features)
    }

    pub fn predict_parameters(&mut self, context: &Array2<T>) -> Result<Vec<T>> {
        let predicted = self.parameter_predictor.forward(context)?;
        Ok(predicted.row(0).to_vec())
    }

    pub fn generate_parameter_update(
        &mut self,
        transformer_output: &Array2<T>,
        current_parameters: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Concatenate transformer output and current parameters
        let batch_size = transformer_output.shape()[0];
        let mut input = Array2::zeros((batch_size, self.model_dimension * 2));

        for i in 0..batch_size {
            for j in 0..self.model_dimension {
                input[[i, j]] = transformer_output[[i, j]];
                if j < current_parameters.len() {
                    input[[i, j + self.model_dimension]] = current_parameters[j];
                }
            }
        }

        let update = self.update_generator.forward(&input)?;
        Ok(update.row(0).to_owned())
    }
}

/// Memory bank for storing task experiences
pub struct MemoryBank<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Stored experiences
    experiences: VecDeque<MemoryExperience<T>>,

    /// Maximum memory size
    max_size: usize,

    /// Dimension of stored parameters
    parameter_dimension: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> MemoryBank<T> {
    pub fn new(max_size: usize, parameter_dimension: usize) -> Result<Self> {
        Ok(Self {
            experiences: VecDeque::new(),
            max_size,
            parameter_dimension,
        })
    }

    pub fn store_experience(
        &mut self,
        task: &TaskBatch<T>,
        parameters: &[T],
        performance: T,
    ) -> Result<()> {
        let experience = MemoryExperience {
            task_signature: self.compute_task_signature(task),
            parameters: parameters.to_vec(),
            performance: performance.to_f64().unwrap_or(0.0),
            timestamp: Instant::now(),
        };

        self.experiences.push_back(experience);

        if self.experiences.len() > self.max_size {
            self.experiences.pop_front();
        }

        Ok(())
    }

    pub fn retrieve_similar_experiences(
        &self,
        task: &TaskBatch<T>,
        k: usize,
    ) -> Result<Vec<MemoryExperience<T>>> {
        let target_signature = self.compute_task_signature(task);

        let mut scored_experiences: Vec<_> = self
            .experiences
            .iter()
            .map(|exp| {
                let similarity = self.compute_similarity(&target_signature, &exp.task_signature);
                (similarity, exp.clone())
            })
            .collect();

        scored_experiences.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("unwrap failed"));

        Ok(scored_experiences
            .into_iter()
            .take(k)
            .map(|(_, exp)| exp)
            .collect())
    }

    fn compute_task_signature(&self, task: &TaskBatch<T>) -> Vec<f64> {
        // Simplified task signature
        vec![task.difficulty, task.complexity, task.data_characteristics]
    }

    fn compute_similarity(&self, sig1: &[f64], sig2: &[f64]) -> f64 {
        if sig1.len() != sig2.len() {
            return 0.0;
        }

        let dot_product: f64 = sig1.iter().zip(sig2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = sig1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = sig2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
}

/// Supporting data structures
#[derive(Debug, Clone)]
pub struct TaskBatch<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
{
    pub id: String,
    pub difficulty: f64,
    pub complexity: f64,
    pub data_characteristics: f64,
    pub _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct TaskAdaptation<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub task_id: String,
    pub adapted_parameters: Vec<T>,
    pub support_loss: T,
    pub query_loss: T,
    /// Outer-loop gradient contributed by this task — the query-loss gradient
    /// at the adapted parameters (FOMAML) or its exact second-order
    /// back-propagation through the inner steps. Averaged by
    /// `TransformerMetaLearning::compute_meta_gradients`.
    pub query_gradient: Vec<T>,
    pub adaptation_steps: usize,
}

#[derive(Debug, Clone)]
pub struct MetaLearningResult<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub meta_loss: f64,
    pub task_adaptations: Vec<TaskAdaptation<T>>,
    pub computation_time: std::time::Duration,
    pub convergence_rate: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryExperience<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub task_signature: Vec<f64>,
    pub parameters: Vec<T>,
    pub performance: f64,
    pub timestamp: Instant,
}

pub struct PerformanceTracker<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    loss_history: VecDeque<f64>,
    meta_results: VecDeque<MetaLearningResult<T>>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for PerformanceTracker<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    PerformanceTracker<T>
{
    pub fn new() -> Self {
        Self {
            loss_history: VecDeque::new(),
            meta_results: VecDeque::new(),
        }
    }

    pub fn record_loss(&mut self, loss: f64) {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 1000 {
            self.loss_history.pop_front();
        }
    }

    pub fn record_meta_step(&mut self, result: MetaLearningResult<T>) {
        self.meta_results.push_back(result);
        if self.meta_results.len() > 100 {
            self.meta_results.pop_front();
        }
    }

    pub fn get_loss_history(&self) -> &VecDeque<f64> {
        &self.loss_history
    }
}

#[derive(Debug, Clone)]
pub struct MetaState<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
{
    parameters: Vec<T>,
    loss_history: VecDeque<T>,
    scale_factor: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> MetaState<T> {
    pub fn new(parameter_count: usize) -> Result<Self> {
        Ok(Self {
            parameters: vec![T::zero(); parameter_count],
            loss_history: VecDeque::new(),
            scale_factor: T::one(),
        })
    }

    pub fn get_parameters(&self) -> &Vec<T> {
        &self.parameters
    }

    pub fn get_parameters_mut(&mut self) -> &mut Vec<T> {
        &mut self.parameters
    }

    pub fn update_parameters(&mut self, updates: &[T], learning_rate: T) -> Result<()> {
        for (param, &update) in self.parameters.iter_mut().zip(updates.iter()) {
            *param = *param + learning_rate * update;
        }
        Ok(())
    }

    pub fn update_loss_history(&mut self, loss: T) {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 100 {
            self.loss_history.pop_front();
        }
    }

    pub fn get_scale_factor(&self) -> T {
        self.scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small configuration so the meta-learning tests stay fast.
    fn small_config(
        param_dim: usize,
    ) -> super::super::config::TransformerBasedOptimizerConfig<f64> {
        let mut config = super::super::config::TransformerBasedOptimizerConfig::<f64> {
            model_dimension: param_dim,
            feedforward_dimension: 2 * param_dim,
            num_attention_heads: 2,
            attention_head_dimension: param_dim / 2,
            num_transformer_layers: 1,
            sequence_length: 8,
            ..Default::default()
        };
        config.meta_learning_config.inner_steps = 3;
        config.meta_learning_config.inner_learning_rate = 0.05;
        config.meta_learning_config.meta_learning_rate = 0.1;
        config
    }

    fn task(id: &str, difficulty: f64, complexity: f64, characteristics: f64) -> TaskBatch<f64> {
        TaskBatch {
            id: id.to_string(),
            difficulty,
            complexity,
            data_characteristics: characteristics,
            _phantom: std::marker::PhantomData,
        }
    }

    /// `(features..., target)` rows for `y = 2·x0 - x1 + 0.5`.
    fn linear_batch(rows: usize, offset: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, 3), |(i, j)| {
            let x0 = (i as f64 + offset) * 0.25;
            let x1 = (i as f64 - offset) * 0.5;
            match j {
                0 => x0,
                1 => x1,
                _ => 2.0 * x0 - x1 + 0.5,
            }
        })
    }

    #[test]
    fn test_meta_learning_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let meta_learning = TransformerMetaLearning::new(&config);
        assert!(meta_learning.is_ok());
    }

    /// F27: `compute_prediction_error` ignored `params` entirely, so every
    /// parameter vector produced the same loss.
    #[test]
    fn task_loss_actually_depends_on_the_parameters() {
        let ml = TransformerMetaLearning::new(&small_config(4)).expect("construction");
        let data = linear_batch(6, 1.0);
        let t = task("t", 0.5, 0.0, 0.3);

        // The exact solution of y = 2·x0 - x1 + 0.5 is [0.5, 2.0, -1.0].
        let exact = vec![0.5, 2.0, -1.0, 0.0];
        let wrong = vec![0.0, 0.0, 0.0, 0.0];

        let loss_exact = ml.compute_task_loss(&exact, &data, &t).expect("loss");
        let loss_wrong = ml.compute_task_loss(&wrong, &data, &t).expect("loss");

        assert!(
            loss_exact < 1e-18,
            "the exact solution should have ~zero loss, got {loss_exact}"
        );
        assert!(
            loss_wrong > 1.0,
            "the zero vector should have a large loss, got {loss_wrong}"
        );
    }

    /// F27: the task descriptor must reach the objective — two tasks over the
    /// same data used to be indistinguishable.
    #[test]
    fn task_descriptor_changes_the_objective() {
        let ml = TransformerMetaLearning::new(&small_config(4)).expect("construction");
        let data = linear_batch(6, 1.0);
        let params = vec![0.4, 1.8, -0.9, 0.2];

        let plain = ml
            .compute_task_loss(&params, &data, &task("a", 0.5, 0.0, 0.3))
            .expect("loss");
        let regularized = ml
            .compute_task_loss(&params, &data, &task("b", 0.5, 4.0, 0.3))
            .expect("loss");

        assert!(
            regularized > plain,
            "a higher-complexity task must be regularized more: {plain} vs {regularized}"
        );
    }

    /// F27: `compute_gradients` returned `loss/n` for every component. The real
    /// gradient must match central finite differences.
    #[test]
    fn compute_gradients_matches_finite_differences() {
        let ml = TransformerMetaLearning::new(&small_config(4)).expect("construction");
        let data = linear_batch(7, 0.5);
        let t = task("fd", 0.4, 1.5, 0.2);
        let params = vec![0.3, -0.7, 1.1, 0.45];

        let analytic = ml.compute_gradients(&params, &data, &t).expect("gradient");
        let h = 1e-6;
        for k in 0..params.len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[k] += h;
            minus[k] -= h;
            let numeric = (ml.compute_task_loss(&plus, &data, &t).expect("loss")
                - ml.compute_task_loss(&minus, &data, &t).expect("loss"))
                / (2.0 * h);
            let scale = numeric.abs().max(1.0);
            assert!(
                (analytic[k] - numeric).abs() / scale < 1e-6,
                "component {k}: analytic {} vs numeric {numeric}",
                analytic[k]
            );
        }
    }

    /// F27: the Hessian-vector product backing exact second-order MAML must
    /// match finite differences of the gradient.
    #[test]
    fn hessian_vector_product_matches_finite_differences() {
        let ml = TransformerMetaLearning::new(&small_config(4)).expect("construction");
        let data = linear_batch(7, 0.5);
        let t = task("hvp", 0.4, 1.5, 0.2);
        let params = vec![0.3, -0.7, 1.1, 0.45];
        let direction = vec![0.9, -0.2, 0.4, 1.3];

        let hv = ml
            .hessian_vector_product(&direction, &data, &t)
            .expect("hvp");

        let h = 1e-6;
        let mut plus = params.clone();
        let mut minus = params.clone();
        for k in 0..params.len() {
            plus[k] += h * direction[k];
            minus[k] -= h * direction[k];
        }
        let gp = ml.compute_gradients(&plus, &data, &t).expect("grad+");
        let gm = ml.compute_gradients(&minus, &data, &t).expect("grad-");

        for k in 0..params.len() {
            let numeric = (gp[k] - gm[k]) / (2.0 * h);
            let scale = numeric.abs().max(1.0);
            assert!(
                (hv[k] - numeric).abs() / scale < 1e-5,
                "component {k}: analytic {} vs numeric {numeric}",
                hv[k]
            );
        }
    }

    /// F27: a real MAML step must reduce the query loss over successive outer
    /// iterations. With fabricated gradients it could not.
    #[test]
    fn maml_step_reduces_the_query_loss() {
        let mut ml = TransformerMetaLearning::new(&small_config(4)).expect("construction");
        let tasks = vec![task("t0", 0.4, 0.0, 0.2), task("t1", 0.6, 0.0, 0.7)];
        let support = vec![linear_batch(8, 0.0), linear_batch(8, 1.0)];
        let query = vec![linear_batch(6, 0.5), linear_batch(6, 1.5)];

        let first = ml
            .maml_step(&tasks, &support, &query)
            .expect("first meta step")
            .meta_loss;
        let mut last = first;
        for _ in 0..40 {
            last = ml
                .maml_step(&tasks, &support, &query)
                .expect("meta step")
                .meta_loss;
        }

        assert!(
            last < first,
            "meta-training did not reduce the query loss: {first} -> {last}"
        );
    }

    /// F27: the exact second-order meta-gradient must differ from the
    /// first-order (FOMAML) one — if `first_order` were ignored they would be
    /// identical.
    #[test]
    fn second_order_meta_gradient_differs_from_first_order() {
        let tasks = vec![task("t0", 0.4, 0.0, 0.2)];
        let support = vec![linear_batch(8, 0.0)];
        let query = vec![linear_batch(6, 0.5)];

        let mut cfg_first = small_config(4);
        cfg_first.meta_learning_config.first_order = true;
        let mut fo = TransformerMetaLearning::new(&cfg_first).expect("construction");

        let mut cfg_second = small_config(4);
        cfg_second.meta_learning_config.first_order = false;
        let mut so = TransformerMetaLearning::new(&cfg_second).expect("construction");

        let fo_result = fo.maml_step(&tasks, &support, &query).expect("fomaml");
        let so_result = so.maml_step(&tasks, &support, &query).expect("maml");

        let fo_grad = &fo_result.task_adaptations[0].query_gradient;
        let so_grad = &so_result.task_adaptations[0].query_gradient;
        let delta = fo_grad
            .iter()
            .zip(so_grad.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            delta > 1e-9,
            "second-order correction had no effect (max delta {delta})"
        );
    }

    /// F27: two different task descriptors must produce different context
    /// embeddings; `encode_task_context` used to feed `Array2::ones` regardless.
    #[test]
    fn task_context_encoding_depends_on_the_task() {
        let mut net = AdaptationNetwork::<f64>::new(8, 16).expect("network");
        let a = net
            .encode_task_context(&task("a", 0.1, 0.2, 0.3))
            .expect("encode a");
        let b = net
            .encode_task_context(&task("b", 0.8, 0.4, 0.9))
            .expect("encode b");
        let delta = (&a - &b).iter().map(|d| d.abs()).fold(0.0, f64::max);
        assert!(
            delta > 1e-9,
            "distinct tasks produced identical context embeddings (delta {delta})"
        );
    }

    /// F86: the momentum branch of `MetaOptimizer::update` was unreachable
    /// because `momentum` was hardcoded to `None`. It must now be configurable
    /// and behave differently from plain SGD.
    #[test]
    fn meta_optimizer_momentum_branch_is_reachable() {
        let mut cfg = MetaLearningConfig::<f64> {
            meta_learning_rate: 0.1,
            ..Default::default()
        };
        let mut sgd = MetaOptimizer::new(&cfg).expect("sgd");
        assert!(sgd.momentum().is_none(), "default must be plain SGD");

        cfg.meta_momentum = Some(0.9);
        let mut momentum = MetaOptimizer::new(&cfg).expect("momentum");
        assert_eq!(momentum.momentum(), Some(0.9));

        let mut state_sgd = MetaState::<f64>::new(3).expect("state");
        let mut state_mom = MetaState::<f64>::new(3).expect("state");
        let grad = vec![1.0, -2.0, 0.5];
        for _ in 0..3 {
            sgd.update(&mut state_sgd, &grad).expect("sgd update");
            momentum
                .update(&mut state_mom, &grad)
                .expect("momentum update");
        }

        let delta = state_sgd
            .get_parameters()
            .iter()
            .zip(state_mom.get_parameters().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            delta > 1e-9,
            "momentum produced the same trajectory as SGD (delta {delta})"
        );
    }

    /// F86: the momentum branch indexed `gradients[i]` unchecked and panicked
    /// on a short gradient; it must return an error instead.
    #[test]
    fn meta_optimizer_rejects_short_gradients() {
        let cfg = MetaLearningConfig::<f64> {
            meta_momentum: Some(0.9),
            ..Default::default()
        };
        let mut opt = MetaOptimizer::new(&cfg).expect("optimizer");
        let mut state = MetaState::<f64>::new(4).expect("state");
        assert!(opt.update(&mut state, &[1.0, 2.0]).is_err());
    }

    /// A short support/query slice used to panic on `support_data[i]`.
    #[test]
    fn meta_step_rejects_mismatched_batches() {
        let mut ml = TransformerMetaLearning::new(&small_config(4)).expect("construction");
        let tasks = vec![task("t0", 0.4, 0.0, 0.2), task("t1", 0.6, 0.0, 0.7)];
        let support = vec![linear_batch(8, 0.0)];
        let query = vec![linear_batch(6, 0.5)];
        assert!(ml.meta_step(&tasks, &support, &query).is_err());
        assert!(ml.meta_step(&[], &support, &query).is_err());
    }

    #[test]
    fn test_memory_bank() {
        let memory = MemoryBank::<f32>::new(100, 64);
        assert!(memory.is_ok());

        let mut bank = memory.expect("unwrap failed");
        let task = TaskBatch {
            id: "test".to_string(),
            difficulty: 0.5,
            complexity: 0.7,
            data_characteristics: 0.3,
            _phantom: std::marker::PhantomData,
        };

        let params = vec![0.1f32; 64];
        assert!(bank.store_experience(&task, &params, 0.8).is_ok());
    }

    #[test]
    fn test_adaptation_network() {
        let network = AdaptationNetwork::<f32>::new(128, 256);
        assert!(network.is_ok());

        let mut net = network.expect("unwrap failed");
        let task = TaskBatch {
            id: "test".to_string(),
            difficulty: 0.5,
            complexity: 0.7,
            data_characteristics: 0.3,
            _phantom: std::marker::PhantomData,
        };

        let context = net.encode_task_context(&task);
        assert!(context.is_ok());
    }
}
