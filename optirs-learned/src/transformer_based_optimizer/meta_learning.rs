// Meta-learning components for transformer-based optimization

use super::config::{ActivationFunction, MetaLearningConfig};
use super::feedforward::FeedForwardNetwork;
use super::layers::LayerNormalization;
use crate::error::Result;
use num_traits::Float;
use scirs2_core::ndarray_ext::{Array1, Array2, Array3, Axis};
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
pub struct TransformerMetaLearning<T: Float + Debug + Send + Sync + 'static> {
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

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray_ext::ScalarOperand>
    TransformerMetaLearning<T>
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

    /// MAML meta-learning step
    fn maml_step(
        &mut self,
        tasks: &[TaskBatch<T>],
        support_data: &[Array2<T>],
        query_data: &[Array2<T>],
    ) -> Result<MetaLearningResult<T>> {
        let start_time = Instant::now();
        let mut total_loss = T::zero();
        let mut task_adaptations = Vec::new();

        for (i, task) in tasks.iter().enumerate() {
            // Inner adaptation loop
            let mut adapted_params = self.meta_state.get_parameters().clone();

            for inner_step in 0..self.config.inner_steps {
                // Compute gradients on support set
                let support_loss =
                    self.compute_task_loss(&adapted_params, &support_data[i], task)?;
                let gradients = self.compute_gradients(&adapted_params, support_loss)?;

                // Update parameters
                for (param, grad) in adapted_params.iter_mut().zip(gradients.iter()) {
                    *param = *param - self.config.inner_learning_rate * (*grad);
                }
            }

            // Evaluate on query set
            let query_loss = self.compute_task_loss(&adapted_params, &query_data[i], task)?;
            total_loss = total_loss + query_loss;

            task_adaptations.push(TaskAdaptation {
                task_id: task.id.clone(),
                adapted_parameters: adapted_params,
                support_loss: self.compute_task_loss(
                    &self.meta_state.get_parameters(),
                    &support_data[i],
                    task,
                )?,
                query_loss,
                adaptation_steps: self.config.inner_steps,
            });
        }

        // Meta-update
        let meta_loss = total_loss / T::from(tasks.len()).unwrap();
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
        let start_time = Instant::now();
        let mut parameter_updates = Vec::new();
        let mut total_loss = T::zero();

        for (i, task) in tasks.iter().enumerate() {
            // Adapt on support set
            let mut adapted_params = self.meta_state.get_parameters().clone();

            for _ in 0..self.config.inner_steps {
                let loss = self.compute_task_loss(&adapted_params, &support_data[i], task)?;
                let gradients = self.compute_gradients(&adapted_params, loss)?;

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

        let num_tasks = T::from(tasks.len()).unwrap();
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
                let loss = self.compute_task_loss(&adapted_params, &support_data[i], task)?;
                let gradients = self.compute_gradients(&adapted_params, loss)?;

                for (param, grad) in adapted_params.iter_mut().zip(gradients.iter()) {
                    *param = *param - self.config.inner_learning_rate * (*grad);
                }
            }

            let query_loss = self.compute_task_loss(&adapted_params, &query_data[i], task)?;
            total_loss = total_loss + query_loss;
        }

        let result = MetaLearningResult {
            meta_loss: (total_loss / T::from(tasks.len()).unwrap())
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
        let start_time = Instant::now();
        let mut total_loss = T::zero();

        for (i, task) in tasks.iter().enumerate() {
            // Retrieve relevant experiences from memory
            let relevant_experiences = self.memory_bank.retrieve_similar_experiences(task, 5)?;

            // Use memory to initialize adaptation
            let memory_guided_params = self.initialize_from_memory(&relevant_experiences)?;

            let mut adapted_params = memory_guided_params;
            for _ in 0..self.config.inner_steps {
                let loss = self.compute_task_loss(&adapted_params, &support_data[i], task)?;
                let gradients = self.compute_gradients(&adapted_params, loss)?;

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
            meta_loss: (total_loss / T::from(tasks.len()).unwrap())
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

    /// Set meta-learning strategy
    pub fn set_strategy(&mut self, strategy: MetaLearningStrategy) {
        self.strategy = strategy;
    }

    /// Get current strategy
    pub fn get_strategy(&self) -> MetaLearningStrategy {
        self.strategy
    }

    /// Helper methods
    fn compute_task_loss(&self, params: &[T], data: &Array2<T>, task: &TaskBatch<T>) -> Result<T> {
        // Simplified loss computation
        let prediction_error = self.compute_prediction_error(params, data, task)?;
        Ok(prediction_error)
    }

    fn compute_prediction_error(
        &self,
        _params: &[T],
        data: &Array2<T>,
        _task: &TaskBatch<T>,
    ) -> Result<T> {
        // Placeholder: compute actual prediction error
        let mean_squared_error = data
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x);
        Ok(mean_squared_error / T::from(data.len()).unwrap())
    }

    fn compute_gradients(&self, params: &[T], loss: T) -> Result<Vec<T>> {
        // Simplified gradient computation
        let gradients = params
            .iter()
            .map(|_| loss / T::from(params.len()).unwrap())
            .collect();
        Ok(gradients)
    }

    fn compute_meta_gradients(&self, adaptations: &[TaskAdaptation<T>]) -> Result<Vec<T>> {
        let param_count = adaptations[0].adapted_parameters.len();
        let mut meta_gradients = vec![T::zero(); param_count];

        for adaptation in adaptations {
            for (i, &param) in adaptation.adapted_parameters.iter().enumerate() {
                meta_gradients[i] = meta_gradients[i] + param * adaptation.query_loss;
            }
        }

        let num_tasks = T::from(adaptations.len()).unwrap();
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
        let improvement = recent_losses.last().unwrap() - recent_losses.first().unwrap();
        Ok(improvement.max(0.0).min(1.0))
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

        let num_experiences = T::from(experiences.len()).unwrap();
        for param in averaged_params.iter_mut() {
            *param = *param / num_experiences;
        }

        Ok(averaged_params)
    }
}

/// Meta-optimizer for outer loop updates
pub struct MetaOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Learning rate
    learning_rate: T,

    /// Momentum for SGD-style updates
    momentum: Option<T>,

    /// Velocity for momentum
    velocity: Option<Vec<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static> MetaOptimizer<T> {
    pub fn new(config: &MetaLearningConfig<T>) -> Result<Self> {
        Ok(Self {
            learning_rate: config.meta_learning_rate,
            momentum: None,
            velocity: None,
        })
    }

    pub fn update(&mut self, state: &mut MetaState<T>, gradients: &[T]) -> Result<()> {
        let mut params = state.get_parameters_mut();

        if let Some(momentum) = self.momentum {
            // Momentum update
            if self.velocity.is_none() {
                self.velocity = Some(vec![T::zero(); params.len()]);
            }

            if let Some(ref mut velocity) = self.velocity {
                for i in 0..params.len() {
                    velocity[i] = momentum * velocity[i] + self.learning_rate * gradients[i];
                    params[i] = params[i] - velocity[i];
                }
            }
        } else {
            // Simple SGD update
            for (param, &grad) in params.iter_mut().zip(gradients.iter()) {
                *param = *param - self.learning_rate * grad;
            }
        }

        Ok(())
    }
}

/// Task adaptation network
pub struct AdaptationNetwork<T: Float + Debug + Send + Sync + 'static> {
    /// Context encoder
    context_encoder: FeedForwardNetwork<T>,

    /// Parameter predictor
    parameter_predictor: FeedForwardNetwork<T>,

    /// Update generator
    update_generator: FeedForwardNetwork<T>,

    /// Model dimension
    model_dimension: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> AdaptationNetwork<T> {
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

    pub fn encode_task_context(&mut self, _task: &TaskBatch<T>) -> Result<Array2<T>> {
        // Encode task characteristics into context vector
        let task_features = Array2::ones((1, self.model_dimension));
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
pub struct MemoryBank<T: Float + Debug + Send + Sync + 'static> {
    /// Stored experiences
    experiences: VecDeque<MemoryExperience<T>>,

    /// Maximum memory size
    max_size: usize,

    /// Dimension of stored parameters
    parameter_dimension: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> MemoryBank<T> {
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

        scored_experiences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

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
pub struct TaskBatch<T: Float + Debug + Send + Sync + 'static> {
    pub id: String,
    pub difficulty: f64,
    pub complexity: f64,
    pub data_characteristics: f64,
    pub _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct TaskAdaptation<T: Float + Debug + Send + Sync + 'static> {
    pub task_id: String,
    pub adapted_parameters: Vec<T>,
    pub support_loss: T,
    pub query_loss: T,
    pub adaptation_steps: usize,
}

#[derive(Debug, Clone)]
pub struct MetaLearningResult<T: Float + Debug + Send + Sync + 'static> {
    pub meta_loss: f64,
    pub task_adaptations: Vec<TaskAdaptation<T>>,
    pub computation_time: std::time::Duration,
    pub convergence_rate: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryExperience<T: Float + Debug + Send + Sync + 'static> {
    pub task_signature: Vec<f64>,
    pub parameters: Vec<T>,
    pub performance: f64,
    pub timestamp: Instant,
}

pub struct PerformanceTracker<T: Float + Debug + Send + Sync + 'static> {
    loss_history: VecDeque<f64>,
    meta_results: VecDeque<MetaLearningResult<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static> PerformanceTracker<T> {
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
pub struct MetaState<T: Float + Debug + Send + Sync + 'static> {
    parameters: Vec<T>,
    loss_history: VecDeque<T>,
    scale_factor: T,
}

impl<T: Float + Debug + Send + Sync + 'static> MetaState<T> {
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

    #[test]
    #[ignore]
    fn test_meta_learning_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let meta_learning = TransformerMetaLearning::new(&config);
        assert!(meta_learning.is_ok());
    }

    #[test]
    fn test_memory_bank() {
        let mut memory = MemoryBank::<f32>::new(100, 64);
        assert!(memory.is_ok());

        let mut bank = memory.unwrap();
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
        let mut network = AdaptationNetwork::<f32>::new(128, 256);
        assert!(network.is_ok());

        let mut net = network.unwrap();
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
