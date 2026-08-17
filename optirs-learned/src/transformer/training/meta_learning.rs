// Meta-learning capabilities for transformer optimization
//
// This module implements meta-learning strategies that allow the transformer
// optimizer to quickly adapt to new tasks and optimization landscapes.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Meta-learning strategies
#[derive(Debug, Clone, Copy)]
pub enum MetaLearningStrategy {
    /// Model-Agnostic Meta-Learning (MAML)
    MAML,
    /// Reptile algorithm
    Reptile,
    /// Gradient-based meta-learning
    GradientBased,
    /// Memory-augmented meta-learning
    MemoryAugmented,
    /// Task-agnostic meta-learning
    TaskAgnostic,
    /// Few-shot meta-learning
    FewShot,
    /// Continual meta-learning
    Continual,
}

/// Meta-learner for transformer optimizer
#[derive(Debug, Clone)]
pub struct TransformerMetaLearner<
    T: Float
        + Debug
        + Default
        + Clone
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
> {
    /// Meta-learning strategy
    strategy: MetaLearningStrategy,

    /// Task embeddings
    task_embeddings: HashMap<String, Array1<T>>,

    /// Meta-training history
    meta_history: VecDeque<MetaTrainingEvent<T>>,

    /// Few-shot learning capabilities
    few_shot_learner: FewShotLearner<T>,

    /// Continual learning state
    continual_learning: ContinualLearningState<T>,

    /// Meta-learning parameters
    meta_params: MetaLearningParams<T>,

    /// Meta-initialization shared across tasks (learned)
    meta_parameters: Option<Array1<T>>,

    /// Task embedding recorded for the most recently adapted task, used by the
    /// continual-learning penalty
    previous_task_parameters: Option<Array1<T>>,
}

/// Meta-training event
#[derive(Debug, Clone)]
pub struct MetaTrainingEvent<T: Float + Debug + Send + Sync + 'static> {
    /// Event type
    event_type: MetaEventType,

    /// Task information
    task_info: TaskInfo<T>,

    /// Performance metrics
    performance: MetaPerformanceMetrics<T>,

    /// Adaptation steps
    adaptation_steps: usize,

    /// Timestamp
    timestamp: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> MetaTrainingEvent<T> {
    /// Which meta-learning phase produced this event.
    pub fn event_type(&self) -> MetaEventType {
        self.event_type
    }

    /// The task the event was recorded for.
    pub fn task_info(&self) -> &TaskInfo<T> {
        &self.task_info
    }

    /// Measured performance of the adaptation.
    pub fn performance(&self) -> &MetaPerformanceMetrics<T> {
        &self.performance
    }

    /// Number of inner-loop steps the adaptation ran.
    pub fn adaptation_steps(&self) -> usize {
        self.adaptation_steps
    }

    /// Position of this event in the meta-training history.
    pub fn timestamp(&self) -> usize {
        self.timestamp
    }
}

/// Meta-event types
#[derive(Debug, Clone, Copy)]
pub enum MetaEventType {
    /// Task adaptation
    TaskAdaptation,
    /// Domain transfer
    DomainTransfer,
    /// Few-shot learning
    FewShotLearning,
    /// Continual learning
    ContinualLearning,
    /// Meta-validation
    MetaValidation,
}

/// Task information
#[derive(Debug, Clone)]
pub struct TaskInfo<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    pub task_id: String,

    /// Task characteristics
    pub characteristics: TaskCharacteristics<T>,

    /// Domain information
    pub domain: DomainInfo,

    /// Difficulty level
    pub difficulty: T,

    /// Expected performance
    pub expected_performance: Option<T>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> TaskInfo<T> {
    /// Build a task descriptor with default characteristics.
    pub fn new(task_id: impl Into<String>, dimensionality: usize, difficulty: T) -> Self {
        Self {
            task_id: task_id.into(),
            characteristics: TaskCharacteristics {
                dimensionality,
                landscape_complexity: T::zero(),
                noise_level: T::zero(),
                conditioning: T::one(),
                sparsity: T::zero(),
                temporal_dependencies: T::zero(),
                feature_correlations: Array2::eye(dimensionality.max(1)),
            },
            domain: DomainInfo {
                name: "general".to_string(),
                domain_type: DomainType::General,
                related_domains: Vec::new(),
                features: HashMap::new(),
            },
            difficulty,
            expected_performance: None,
        }
    }
}

/// Task characteristics
#[derive(Debug, Clone)]
pub struct TaskCharacteristics<T: Float + Debug + Send + Sync + 'static> {
    /// Problem dimensionality
    pub dimensionality: usize,

    /// Landscape complexity
    pub landscape_complexity: T,

    /// Noise level
    pub noise_level: T,

    /// Conditioning number
    pub conditioning: T,

    /// Sparsity level
    pub sparsity: T,

    /// Temporal dependencies
    pub temporal_dependencies: T,

    /// Feature correlations
    pub feature_correlations: Array2<T>,
}

/// Domain information
#[derive(Debug, Clone)]
pub struct DomainInfo {
    /// Domain name
    pub name: String,

    /// Domain type
    pub domain_type: DomainType,

    /// Related domains
    pub related_domains: Vec<String>,

    /// Domain-specific features
    pub features: HashMap<String, f64>,
}

/// Domain types
#[derive(Debug, Clone, Copy)]
pub enum DomainType {
    /// Computer vision
    Vision,
    /// Natural language processing
    NLP,
    /// Reinforcement learning
    RL,
    /// Time series
    TimeSeries,
    /// Graph neural networks
    Graph,
    /// Scientific computing
    Scientific,
    /// General optimization
    General,
}

/// Meta-performance metrics
#[derive(Debug, Clone)]
pub struct MetaPerformanceMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Final performance
    final_performance: T,

    /// Convergence speed
    convergence_speed: T,

    /// Sample efficiency
    sample_efficiency: T,

    /// Generalization score
    generalization: T,

    /// Stability measure
    stability: T,

    /// Resource usage
    resource_usage: T,
}

impl<T: Float + Debug + Send + Sync + 'static> MetaPerformanceMetrics<T> {
    /// Query loss after adaptation (lower is better).
    pub fn final_performance(&self) -> T {
        self.final_performance
    }

    /// Relative loss reduction achieved by the inner loop.
    pub fn convergence_speed(&self) -> T {
        self.convergence_speed
    }

    /// Performance achieved per support sample.
    pub fn sample_efficiency(&self) -> T {
        self.sample_efficiency
    }

    /// `1 / (1 + query_loss)`: bounded generalization score.
    pub fn generalization(&self) -> T {
        self.generalization
    }

    /// `1 / (1 + |query_loss - support_loss|)`: bounded support/query agreement.
    pub fn stability(&self) -> T {
        self.stability
    }

    /// Inner-loop steps spent, as a resource proxy.
    pub fn resource_usage(&self) -> T {
        self.resource_usage
    }
}

/// Few-shot learner component
#[derive(Debug, Clone)]
pub struct FewShotLearner<T: Float + Debug + Send + Sync + 'static> {
    /// Support set memory
    support_memory: HashMap<String, Vec<Array1<T>>>,

    /// Prototype vectors
    prototypes: HashMap<String, Array1<T>>,
}

/// Continual learning state
#[derive(Debug, Clone)]
pub struct ContinualLearningState<T: Float + Debug + Send + Sync + 'static> {
    /// Elastic weight consolidation parameters
    ewc_params: HashMap<String, Array1<T>>,

    /// Fisher information matrix
    fisher_information: HashMap<String, Array2<T>>,

    /// Previous task importance scores
    task_importance: HashMap<String, T>,
}

/// Meta-learning parameters
#[derive(Debug, Clone)]
pub struct MetaLearningParams<T: Float + Debug + Send + Sync + 'static> {
    /// Learning rate for meta-updates
    pub meta_learning_rate: T,

    /// Learning rate used inside the task adaptation loop
    pub inner_learning_rate: T,

    /// Number of inner gradient steps
    pub inner_steps: usize,

    /// Meta-batch size
    pub meta_batch_size: usize,

    /// Task diversity weight
    pub diversity_weight: T,

    /// Transfer learning coefficient
    pub transfer_coefficient: T,

    /// Memory retention factor
    pub memory_retention: T,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > TransformerMetaLearner<T>
{
    /// Create new meta-learner
    pub fn new(strategy: MetaLearningStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            task_embeddings: HashMap::new(),
            meta_history: VecDeque::new(),
            few_shot_learner: FewShotLearner::new()?,
            continual_learning: ContinualLearningState::new()?,
            meta_params: MetaLearningParams::default(),
            meta_parameters: None,
            previous_task_parameters: None,
        })
    }

    /// Adapt to a new task and return the post-adaptation query loss.
    ///
    /// Every variant operates on the same concrete objective - fitting a single
    /// parameter vector `theta` to the task's samples under the mean squared
    /// distance `L(theta) = mean_i ||theta - x_i||^2`, whose gradient is
    /// `2 (theta - mean(x))`. That keeps the meta-learning algorithms honest
    /// (they really run inner gradient loops and really update the shared
    /// meta-initialization) while staying independent of any particular model.
    pub fn adapt_to_task(
        &mut self,
        task_info: &TaskInfo<T>,
        support_data: &[Array1<T>],
        query_data: &[Array1<T>],
    ) -> Result<T> {
        let dim = Self::validate_samples(support_data, query_data)?;
        let theta0 = match self.meta_parameters.as_ref() {
            Some(theta) if theta.len() == dim => theta.clone(),
            _ => {
                let theta = Array1::zeros(dim);
                self.meta_parameters = Some(theta.clone());
                theta
            }
        };

        let adapted = match self.strategy {
            MetaLearningStrategy::MAML => self.maml_adaptation(&theta0, support_data),
            MetaLearningStrategy::Reptile => self.reptile_adaptation(&theta0, support_data),
            MetaLearningStrategy::GradientBased => {
                self.gradient_based_adaptation(&theta0, support_data, query_data)
            }
            MetaLearningStrategy::MemoryAugmented => {
                self.memory_augmented_adaptation(task_info, &theta0, support_data)
            }
            MetaLearningStrategy::TaskAgnostic => self.task_agnostic_adaptation(&theta0, dim),
            MetaLearningStrategy::FewShot => {
                self.few_shot_adaptation(task_info, &theta0, support_data)
            }
            MetaLearningStrategy::Continual => {
                self.continual_adaptation(task_info, &theta0, support_data)?
            }
        };

        let query_loss = Self::sample_loss(&adapted, query_data);
        let support_loss = Self::sample_loss(&adapted, support_data);

        // Meta-update of the shared initialization.
        self.apply_meta_update(&theta0, &adapted);

        self.task_embeddings
            .insert(task_info.task_id.clone(), adapted.clone());
        self.previous_task_parameters = Some(adapted);

        let event = MetaTrainingEvent {
            event_type: Self::event_type(self.strategy),
            task_info: task_info.clone(),
            performance: MetaPerformanceMetrics {
                final_performance: query_loss,
                convergence_speed: Self::convergence_speed(
                    Self::sample_loss(&theta0, query_data),
                    query_loss,
                ),
                sample_efficiency: Self::sample_efficiency(support_data.len(), query_loss),
                generalization: T::one() / (T::one() + query_loss),
                stability: T::one() / (T::one() + (query_loss - support_loss).abs()),
                resource_usage: scirs2_core::numeric::NumCast::from(
                    self.meta_params.inner_steps as f64,
                )
                .unwrap_or_else(|| T::zero()),
            },
            adaptation_steps: self.meta_params.inner_steps,
            timestamp: self.meta_history.len(),
        };
        self.meta_history.push_back(event);
        if self.meta_history.len() > 1000 {
            self.meta_history.pop_front();
        }

        Ok(query_loss)
    }

    /// The meta-event category each strategy records.
    fn event_type(strategy: MetaLearningStrategy) -> MetaEventType {
        match strategy {
            MetaLearningStrategy::FewShot => MetaEventType::FewShotLearning,
            MetaLearningStrategy::Continual => MetaEventType::ContinualLearning,
            MetaLearningStrategy::TaskAgnostic => MetaEventType::DomainTransfer,
            _ => MetaEventType::TaskAdaptation,
        }
    }

    /// Validate that every sample shares one dimensionality and return it.
    fn validate_samples(support: &[Array1<T>], query: &[Array1<T>]) -> Result<usize> {
        let dim = support
            .first()
            .or_else(|| query.first())
            .map(|v| v.len())
            .ok_or_else(|| {
                OptimError::InvalidConfig(
                    "Adaptation needs at least one support or query sample".to_string(),
                )
            })?;
        if dim == 0 {
            return Err(OptimError::InvalidConfig(
                "Adaptation samples must be non-empty".to_string(),
            ));
        }
        for sample in support.iter().chain(query.iter()) {
            if sample.len() != dim {
                return Err(OptimError::InvalidConfig(format!(
                    "Inconsistent sample dimensions: expected {dim}, found {}",
                    sample.len()
                )));
            }
        }
        Ok(dim)
    }

    /// Mean of a sample set, or `None` when the set is empty.
    fn mean_vector(data: &[Array1<T>], dim: usize) -> Option<Array1<T>> {
        if data.is_empty() {
            return None;
        }
        let mut sum = Array1::zeros(dim);
        for sample in data {
            sum = sum + sample;
        }
        let count: T =
            scirs2_core::numeric::NumCast::from(data.len() as f64).unwrap_or_else(|| T::one());
        Some(sum / count)
    }

    /// `L(theta) = mean_i ||theta - x_i||^2`
    fn sample_loss(theta: &Array1<T>, data: &[Array1<T>]) -> T {
        if data.is_empty() {
            return T::zero();
        }
        let mut total = T::zero();
        for sample in data {
            for (a, b) in theta.iter().zip(sample.iter()) {
                let diff = *a - *b;
                total = total + diff * diff;
            }
        }
        let count: T =
            scirs2_core::numeric::NumCast::from(data.len() as f64).unwrap_or_else(|| T::one());
        total / count
    }

    /// `grad L(theta) = 2 (theta - mean(x))`
    fn sample_gradient(theta: &Array1<T>, data: &[Array1<T>]) -> Array1<T> {
        match Self::mean_vector(data, theta.len()) {
            Some(mean) => {
                let two: T = scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::one());
                (theta - &mean) * two
            }
            None => Array1::zeros(theta.len()),
        }
    }

    /// Run `steps` gradient-descent steps of the task objective.
    fn inner_loop(&self, theta: &Array1<T>, data: &[Array1<T>], steps: usize) -> Array1<T> {
        let lr = self.meta_params.inner_learning_rate;
        let mut current = theta.clone();
        for _ in 0..steps {
            let gradient = Self::sample_gradient(&current, data);
            current = current - gradient * lr;
        }
        current
    }

    /// MAML: adapt from the shared initialization with the full inner loop.
    fn maml_adaptation(&self, theta0: &Array1<T>, support: &[Array1<T>]) -> Array1<T> {
        self.inner_loop(theta0, support, self.meta_params.inner_steps)
    }

    /// Reptile: identical inner loop, but the meta-update (below) interpolates
    /// toward the adapted parameters instead of following the query gradient.
    fn reptile_adaptation(&self, theta0: &Array1<T>, support: &[Array1<T>]) -> Array1<T> {
        self.inner_loop(theta0, support, self.meta_params.inner_steps)
    }

    /// Gradient-based: descend the combined support and query objective.
    fn gradient_based_adaptation(
        &self,
        theta0: &Array1<T>,
        support: &[Array1<T>],
        query: &[Array1<T>],
    ) -> Array1<T> {
        let combined: Vec<Array1<T>> = support.iter().chain(query.iter()).cloned().collect();
        self.inner_loop(theta0, &combined, self.meta_params.inner_steps)
    }

    /// Memory-augmented: blend the inner-loop solution with the closest task
    /// embedding already stored in memory.
    fn memory_augmented_adaptation(
        &self,
        task_info: &TaskInfo<T>,
        theta0: &Array1<T>,
        support: &[Array1<T>],
    ) -> Array1<T> {
        let adapted = self.inner_loop(theta0, support, self.meta_params.inner_steps);

        let mut best: Option<(T, &Array1<T>)> = None;
        for (task_id, embedding) in &self.task_embeddings {
            if task_id == &task_info.task_id || embedding.len() != adapted.len() {
                continue;
            }
            let distance = adapted
                .iter()
                .zip(embedding.iter())
                .map(|(a, b)| (*a - *b) * (*a - *b))
                .fold(T::zero(), |x, y| x + y);
            if best.as_ref().is_none_or(|(d, _)| distance < *d) {
                best = Some((distance, embedding));
            }
        }

        match best {
            Some((_, neighbour)) => {
                let weight = self.meta_params.memory_retention;
                adapted * weight + neighbour * (T::one() - weight)
            }
            None => adapted,
        }
    }

    /// Task-agnostic: ignore the current task's samples and return the average
    /// of every task embedding seen so far (falling back to the shared
    /// initialization when nothing has been stored yet).
    fn task_agnostic_adaptation(&self, theta0: &Array1<T>, dim: usize) -> Array1<T> {
        let embeddings: Vec<Array1<T>> = self
            .task_embeddings
            .values()
            .filter(|v| v.len() == dim)
            .cloned()
            .collect();
        Self::mean_vector(&embeddings, dim).unwrap_or_else(|| theta0.clone())
    }

    /// Few-shot: use the support prototype directly, with no gradient steps.
    fn few_shot_adaptation(
        &mut self,
        task_info: &TaskInfo<T>,
        theta0: &Array1<T>,
        support: &[Array1<T>],
    ) -> Array1<T> {
        let prototype = Self::mean_vector(support, theta0.len()).unwrap_or_else(|| theta0.clone());
        self.few_shot_learner
            .record(&task_info.task_id, support, prototype.clone());
        prototype
    }

    /// Continual: inner loop with an elastic penalty pulling the solution back
    /// toward the previous task's parameters, damping catastrophic forgetting.
    fn continual_adaptation(
        &mut self,
        task_info: &TaskInfo<T>,
        theta0: &Array1<T>,
        support: &[Array1<T>],
    ) -> Result<Array1<T>> {
        self.continual_learning
            .update_for_task(task_info, support)?;

        let adapted = self.inner_loop(theta0, support, self.meta_params.inner_steps);
        let anchored = match self.previous_task_parameters.as_ref() {
            Some(previous) if previous.len() == adapted.len() => {
                let retention = self.meta_params.memory_retention;
                adapted * retention + previous * (T::one() - retention)
            }
            _ => adapted,
        };

        Ok(anchored)
    }

    /// Meta-update of the shared initialization.
    fn apply_meta_update(&mut self, theta0: &Array1<T>, adapted: &Array1<T>) {
        let meta_lr = self.meta_params.meta_learning_rate;
        let updated = match self.strategy {
            // Reptile moves the initialization toward the adapted parameters.
            MetaLearningStrategy::Reptile => {
                theta0 + &((adapted - theta0) * self.meta_params.transfer_coefficient)
            }
            // Everything else takes a (first-order) step along the adaptation
            // direction scaled by the meta learning rate.
            _ => theta0 + &((adapted - theta0) * meta_lr),
        };
        self.meta_parameters = Some(updated);
    }

    /// Relative improvement achieved by adaptation, clamped to [0, 1].
    fn convergence_speed(before: T, after: T) -> T {
        if before <= T::zero() {
            return T::zero();
        }
        ((before - after) / before).max(T::zero()).min(T::one())
    }

    /// Query performance per support sample.
    fn sample_efficiency(support_count: usize, query_loss: T) -> T {
        let count: T =
            scirs2_core::numeric::NumCast::from(support_count as f64).unwrap_or_else(|| T::one());
        if count <= T::zero() {
            return T::zero();
        }
        T::one() / ((T::one() + query_loss) * count)
    }

    /// The current meta-initialization, if any task has been adapted.
    pub fn meta_parameters(&self) -> Option<&Array1<T>> {
        self.meta_parameters.as_ref()
    }

    /// Embedding learned for a task.
    pub fn task_embedding(&self, task_id: &str) -> Option<&Array1<T>> {
        self.task_embeddings.get(task_id)
    }

    /// The active meta-learning strategy.
    pub fn strategy(&self) -> MetaLearningStrategy {
        self.strategy
    }

    /// Change the active meta-learning strategy.
    pub fn set_strategy(&mut self, strategy: MetaLearningStrategy) {
        self.strategy = strategy;
    }

    /// Get meta-learning statistics
    pub fn get_meta_statistics(&self) -> HashMap<String, T> {
        let mut stats = HashMap::new();

        stats.insert(
            "meta_events_count".to_string(),
            scirs2_core::numeric::NumCast::from(self.meta_history.len() as f64)
                .unwrap_or_else(|| T::zero()),
        );
        stats.insert(
            "task_embeddings_count".to_string(),
            scirs2_core::numeric::NumCast::from(self.task_embeddings.len() as f64)
                .unwrap_or_else(|| T::zero()),
        );

        // Compute average performance
        if !self.meta_history.is_empty() {
            let avg_performance = self
                .meta_history
                .iter()
                .map(|event| event.performance.final_performance)
                .fold(T::zero(), |a, b| a + b)
                / scirs2_core::numeric::NumCast::from(self.meta_history.len() as f64)
                    .unwrap_or_else(|| T::one());
            stats.insert("average_performance".to_string(), avg_performance);
        }

        stats
    }

    /// Update meta-parameters
    pub fn update_meta_parameters(&mut self, params: MetaLearningParams<T>) {
        self.meta_params = params;
    }

    /// Prototype learned for a task by the few-shot component.
    pub fn few_shot_prototype(&self, task_id: &str) -> Option<&Array1<T>> {
        self.few_shot_learner.prototype(task_id)
    }

    /// Recorded meta-training events, oldest first (capped at 1000).
    ///
    /// [`Self::adapt_to_task`] fills every field of each event with a measured
    /// quantity, but nothing could read them: the events went into a private
    /// deque whose only observable property was its length (via
    /// [`Self::get_meta_statistics`]'s `meta_events_count`).
    pub fn meta_history(&self) -> impl ExactSizeIterator<Item = &MetaTrainingEvent<T>> {
        self.meta_history.iter()
    }

    /// The most recently recorded meta-training event, if any.
    pub fn last_meta_event(&self) -> Option<&MetaTrainingEvent<T>> {
        self.meta_history.back()
    }

    /// Reset meta-learner state
    pub fn reset(&mut self) {
        self.meta_parameters = None;
        self.previous_task_parameters = None;
        self.task_embeddings.clear();
        self.meta_history.clear();
        self.few_shot_learner.reset();
        self.continual_learning.reset();
    }
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > FewShotLearner<T>
{
    fn new() -> Result<Self> {
        Ok(Self {
            support_memory: HashMap::new(),
            prototypes: HashMap::new(),
        })
    }

    /// Store the support set and its prototype for a task.
    fn record(&mut self, task_id: &str, support: &[Array1<T>], prototype: Array1<T>) {
        self.support_memory
            .insert(task_id.to_string(), support.to_vec());
        self.prototypes.insert(task_id.to_string(), prototype);
    }

    /// Prototype learned for a task, if one has been recorded.
    fn prototype(&self, task_id: &str) -> Option<&Array1<T>> {
        self.prototypes.get(task_id)
    }

    fn reset(&mut self) {
        self.support_memory.clear();
        self.prototypes.clear();
    }
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > ContinualLearningState<T>
{
    fn new() -> Result<Self> {
        Ok(Self {
            ewc_params: HashMap::new(),
            fisher_information: HashMap::new(),
            task_importance: HashMap::new(),
        })
    }

    fn update_for_task(
        &mut self,
        task_info: &TaskInfo<T>,
        _support_data: &[Array1<T>],
    ) -> Result<()> {
        self.task_importance
            .insert(task_info.task_id.clone(), task_info.difficulty);
        Ok(())
    }

    fn reset(&mut self) {
        self.ewc_params.clear();
        self.fisher_information.clear();
        self.task_importance.clear();
    }
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + Default
            + Clone
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > Default for MetaLearningParams<T>
{
    fn default() -> Self {
        Self {
            meta_learning_rate: scirs2_core::numeric::NumCast::from(0.05)
                .unwrap_or_else(|| T::zero()),
            inner_learning_rate: scirs2_core::numeric::NumCast::from(0.1)
                .unwrap_or_else(|| T::zero()),
            inner_steps: 5,
            meta_batch_size: 32,
            diversity_weight: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
            transfer_coefficient: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
            memory_retention: scirs2_core::numeric::NumCast::from(0.95)
                .unwrap_or_else(|| T::zero()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(values: &[[f64; 2]]) -> Vec<Array1<f64>> {
        values
            .iter()
            .map(|v| Array1::from_vec(v.to_vec()))
            .collect()
    }

    fn all_strategies() -> [MetaLearningStrategy; 7] {
        [
            MetaLearningStrategy::MAML,
            MetaLearningStrategy::Reptile,
            MetaLearningStrategy::GradientBased,
            MetaLearningStrategy::MemoryAugmented,
            MetaLearningStrategy::TaskAgnostic,
            MetaLearningStrategy::FewShot,
            MetaLearningStrategy::Continual,
        ]
    }

    #[test]
    fn every_strategy_adapts_end_to_end() {
        let support = samples(&[[1.0, 2.0], [1.2, 1.8], [0.8, 2.2]]);
        let query = samples(&[[1.1, 2.1], [0.9, 1.9]]);

        for strategy in all_strategies() {
            let mut learner =
                TransformerMetaLearner::<f64>::new(strategy).expect("meta-learner creation");
            let task = TaskInfo::new("task_a", 2, 0.5);

            let loss = learner
                .adapt_to_task(&task, &support, &query)
                .unwrap_or_else(|e| panic!("{strategy:?} failed: {e}"));

            assert!(loss.is_finite(), "{strategy:?} produced {loss}");
            assert!(loss >= 0.0, "{strategy:?} produced a negative loss");
            assert!(
                learner.meta_parameters().is_some(),
                "{strategy:?} never wrote meta-parameters"
            );
            assert!(
                learner.task_embedding("task_a").is_some(),
                "{strategy:?} never recorded a task embedding"
            );
            let stats = learner.get_meta_statistics();
            assert!(stats.contains_key("meta_events_count"));
        }
    }

    /// Every field of a `MetaTrainingEvent` is filled with a measured quantity,
    /// but they were all private with no accessor: the recorded history was
    /// write-only and only its *length* was observable.
    #[test]
    fn the_recorded_meta_history_is_readable_and_measured() {
        let support = samples(&[[1.0, 2.0], [1.2, 1.8], [0.8, 2.2]]);
        let query = samples(&[[1.1, 2.1], [0.9, 1.9]]);
        let mut learner =
            TransformerMetaLearner::<f64>::new(MetaLearningStrategy::MAML).expect("learner");

        assert_eq!(learner.meta_history().len(), 0);
        assert!(learner.last_meta_event().is_none());

        for i in 0..3 {
            let task = TaskInfo::new(format!("task_{i}"), 2, 0.5);
            learner
                .adapt_to_task(&task, &support, &query)
                .expect("adaptation");
        }

        assert_eq!(learner.meta_history().len(), 3);
        for (i, event) in learner.meta_history().enumerate() {
            assert_eq!(event.timestamp(), i, "events must be in recording order");
            assert_eq!(event.task_info().task_id, format!("task_{i}"));
            assert!(event.adaptation_steps() > 0);
            assert!(matches!(event.event_type(), MetaEventType::TaskAdaptation));

            let perf = event.performance();
            assert!(perf.final_performance().is_finite());
            assert!(perf.convergence_speed().is_finite());
            assert!(perf.sample_efficiency().is_finite());
            // Both scores are `1/(1+x)` for a non-negative `x`, so they live in
            // `(0, 1]` — a constant 0 would mean nothing was measured.
            assert!(perf.generalization() > 0.0 && perf.generalization() <= 1.0);
            assert!(perf.stability() > 0.0 && perf.stability() <= 1.0);
            assert!(perf.resource_usage() > 0.0);
        }

        let last = learner.last_meta_event().expect("a last event");
        assert_eq!(last.timestamp(), 2);

        learner.reset();
        assert_eq!(learner.meta_history().len(), 0);
    }

    #[test]
    fn maml_inner_loop_reduces_the_query_loss() {
        let support = samples(&[[3.0, -3.0], [3.2, -2.8], [2.8, -3.2]]);
        let query = samples(&[[3.1, -3.1], [2.9, -2.9]]);

        let mut learner = TransformerMetaLearner::<f64>::new(MetaLearningStrategy::MAML)
            .expect("meta-learner creation");
        let task = TaskInfo::new("task_a", 2, 0.5);

        let zero = Array1::<f64>::zeros(2);
        let baseline = TransformerMetaLearner::<f64>::sample_loss(&zero, &query);
        let adapted_loss = learner
            .adapt_to_task(&task, &support, &query)
            .expect("adaptation");

        assert!(
            adapted_loss < baseline,
            "adaptation did not help: {adapted_loss} vs {baseline}"
        );
    }

    #[test]
    fn repeated_meta_updates_move_the_initialization() {
        let support = samples(&[[5.0, 5.0], [5.0, 5.0]]);
        let query = samples(&[[5.0, 5.0]]);
        let mut learner = TransformerMetaLearner::<f64>::new(MetaLearningStrategy::Reptile)
            .expect("meta-learner creation");

        let mut losses = Vec::new();
        for i in 0..20 {
            let task = TaskInfo::new(format!("task_{i}"), 2, 0.5);
            losses.push(
                learner
                    .adapt_to_task(&task, &support, &query)
                    .expect("adaptation"),
            );
        }

        assert!(
            losses[19] < losses[0],
            "meta-training did not improve: {:?} -> {:?}",
            losses[0],
            losses[19]
        );
        let meta = learner.meta_parameters().expect("meta-parameters");
        assert!(
            meta.iter().any(|&v| v.abs() > 1e-6),
            "meta-init stayed zero"
        );
    }

    #[test]
    fn few_shot_uses_the_support_prototype() {
        let support = samples(&[[2.0, 4.0], [4.0, 8.0]]);
        let query = samples(&[[3.0, 6.0]]);
        let mut learner = TransformerMetaLearner::<f64>::new(MetaLearningStrategy::FewShot)
            .expect("meta-learner creation");
        let task = TaskInfo::new("proto", 2, 0.5);

        learner
            .adapt_to_task(&task, &support, &query)
            .expect("adaptation");

        let prototype = learner.few_shot_prototype("proto").expect("prototype");
        assert!((prototype[0] - 3.0).abs() < 1e-12);
        assert!((prototype[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn strategies_reach_different_solutions() {
        let support = samples(&[[1.0, 0.0], [3.0, 0.0]]);
        let query = samples(&[[10.0, 0.0]]);

        let run = |strategy| {
            let mut learner =
                TransformerMetaLearner::<f64>::new(strategy).expect("meta-learner creation");
            let task = TaskInfo::new("task", 2, 0.5);
            learner
                .adapt_to_task(&task, &support, &query)
                .expect("adaptation")
        };

        let maml = run(MetaLearningStrategy::MAML);
        let few_shot = run(MetaLearningStrategy::FewShot);
        let gradient_based = run(MetaLearningStrategy::GradientBased);
        let task_agnostic = run(MetaLearningStrategy::TaskAgnostic);

        let mut values = vec![maml, few_shot, gradient_based, task_agnostic];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert!(
            values.len() >= 3,
            "strategies collapsed to identical behaviour: {values:?}"
        );
    }

    #[test]
    fn inconsistent_sample_dimensions_are_rejected() {
        let mut learner = TransformerMetaLearner::<f64>::new(MetaLearningStrategy::MAML)
            .expect("meta-learner creation");
        let task = TaskInfo::new("task", 2, 0.5);
        let support = vec![Array1::from_vec(vec![1.0, 2.0])];
        let query = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        assert!(learner.adapt_to_task(&task, &support, &query).is_err());
    }

    #[test]
    fn empty_sample_sets_are_rejected() {
        let mut learner = TransformerMetaLearner::<f64>::new(MetaLearningStrategy::MAML)
            .expect("meta-learner creation");
        let task = TaskInfo::new("task", 2, 0.5);
        assert!(learner.adapt_to_task(&task, &[], &[]).is_err());
    }

    #[test]
    fn reset_clears_learned_state() {
        let support = samples(&[[1.0, 1.0]]);
        let query = samples(&[[1.0, 1.0]]);
        let mut learner = TransformerMetaLearner::<f64>::new(MetaLearningStrategy::MAML)
            .expect("meta-learner creation");
        let task = TaskInfo::new("task", 2, 0.5);
        learner
            .adapt_to_task(&task, &support, &query)
            .expect("adaptation");
        learner.reset();
        assert!(learner.meta_parameters().is_none());
        assert!(learner.task_embedding("task").is_none());
    }
}
