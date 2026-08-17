//! Meta-learning framework and its subsystems.
//!
//! [`MetaLearningFramework`] wires a concrete meta-learner to the task
//! sampler, the held-out validator, the adaptation engine and the transfer /
//! continual / multi-task / few-shot subsystems. Every subsystem here computes
//! its reported numbers from data it actually saw; see the individual doc
//! comments for what each figure means.

#[allow(unused_imports)]
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::MetaLearner;
use super::types::*;

/// Task distribution manager.
///
/// Samples meta-training batches from the caller's task pool honouring
/// [`TaskSamplingStrategy`]. Sampling is deterministic for a given seed and
/// call index, so repeated runs reproduce the same batches.
pub struct TaskDistributionManager<T: Float + Debug + Send + Sync + 'static> {
    config: MetaLearningConfig,
    /// Base RNG seed
    seed: u64,
    /// Number of batches drawn so far (mixed into the seed and used to advance
    /// the curriculum)
    draws: usize,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> TaskDistributionManager<T> {
    pub fn new(config: &MetaLearningConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            seed: 0x5EED_1234_ABCD_0001,
            draws: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Override the base sampling seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The sampling strategy in force.
    pub fn strategy(&self) -> TaskSamplingStrategy {
        self.config.task_sampling_strategy
    }

    /// Draw a batch of tasks from `tasks`.
    ///
    /// Returns `Err` when the pool is empty or `batch_size` is zero — an empty
    /// pool used to yield default (empty) tasks whose losses were `0/0`.
    pub fn sample_task_batch(
        &mut self,
        tasks: &[MetaTask<T>],
        batch_size: usize,
    ) -> Result<Vec<MetaTask<T>>> {
        if tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "cannot sample a task batch from an empty task pool".to_string(),
            ));
        }
        if batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "task batch size must be greater than zero".to_string(),
            ));
        }

        let draw = self.draws;
        self.draws += 1;
        let mut rng = scirs2_core::random::Random::seed(self.seed.wrapping_add(draw as u64));

        let indices: Vec<usize> = match self.config.task_sampling_strategy {
            TaskSamplingStrategy::Uniform => (0..batch_size)
                .map(|_| rng.random_range(0..tasks.len()))
                .collect(),
            TaskSamplingStrategy::Curriculum => {
                // Easiest first, advancing one batch per draw and wrapping.
                let order = Self::order_by_difficulty(tasks, false);
                (0..batch_size)
                    .map(|i| order[(draw * batch_size + i) % order.len()])
                    .collect()
            }
            TaskSamplingStrategy::DifficultyBased => {
                Self::sample_weighted_by_difficulty(tasks, batch_size, &mut rng)
            }
            TaskSamplingStrategy::DiversityBased => Self::greedy_diverse(tasks, batch_size),
            TaskSamplingStrategy::ActiveLearning | TaskSamplingStrategy::Adversarial => {
                // Hardest first: the tasks the current meta-parameters are most
                // likely to be wrong about.
                let order = Self::order_by_difficulty(tasks, true);
                (0..batch_size).map(|i| order[i % order.len()]).collect()
            }
        };

        let mut batch = Vec::with_capacity(indices.len());
        for idx in indices {
            let task = tasks.get(idx).ok_or_else(|| {
                OptimError::ComputationError(format!("sampled task index {idx} out of range"))
            })?;
            batch.push(task.clone());
        }
        Ok(batch)
    }

    /// Task indices ordered by difficulty (ascending, or descending when
    /// `hardest_first`). NaN difficulties sort as equal.
    fn order_by_difficulty(tasks: &[MetaTask<T>], hardest_first: bool) -> Vec<usize> {
        let mut order: Vec<usize> = (0..tasks.len()).collect();
        order.sort_by(|&a, &b| {
            let ord = tasks[a]
                .difficulty
                .partial_cmp(&tasks[b].difficulty)
                .unwrap_or(std::cmp::Ordering::Equal);
            if hardest_first {
                ord.reverse()
            } else {
                ord
            }
        });
        order
    }

    fn sample_weighted_by_difficulty<R: scirs2_core::random::Rng>(
        tasks: &[MetaTask<T>],
        batch_size: usize,
        rng: &mut scirs2_core::random::Random<R>,
    ) -> Vec<usize> {
        // Weights are difficulties shifted to be strictly positive.
        let mut weights: Vec<f64> = tasks
            .iter()
            .map(|t| t.difficulty.to_f64().unwrap_or(0.0))
            .collect();
        let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
        let shift = if min.is_finite() && min < 0.0 {
            -min
        } else {
            0.0
        };
        for w in weights.iter_mut() {
            if !w.is_finite() {
                *w = 0.0;
            }
            *w = *w + shift + 1e-9;
        }
        let total: f64 = weights.iter().sum();
        (0..batch_size)
            .map(|_| {
                let mut target = rng.random_range(0.0..total);
                for (i, w) in weights.iter().enumerate() {
                    target -= *w;
                    if target <= 0.0 {
                        return i;
                    }
                }
                weights.len() - 1
            })
            .collect()
    }

    /// Greedy farthest-point selection over per-task mean support features.
    fn greedy_diverse(tasks: &[MetaTask<T>], batch_size: usize) -> Vec<usize> {
        let signatures: Vec<Vec<f64>> = tasks.iter().map(Self::task_signature).collect();
        let mut chosen: Vec<usize> = Vec::with_capacity(batch_size);
        chosen.push(0);
        while chosen.len() < batch_size {
            let mut best = 0usize;
            let mut best_dist = -1.0f64;
            for (i, sig) in signatures.iter().enumerate() {
                let d = chosen
                    .iter()
                    .map(|&c| Self::signature_distance(sig, &signatures[c]))
                    .fold(f64::INFINITY, f64::min);
                if d > best_dist {
                    best_dist = d;
                    best = i;
                }
            }
            chosen.push(best);
            if chosen.len() >= batch_size {
                break;
            }
        }
        chosen.truncate(batch_size);
        chosen
    }

    fn task_signature(task: &MetaTask<T>) -> Vec<f64> {
        let mut acc: Vec<f64> = Vec::new();
        for feat in &task.support_set.features {
            if acc.len() < feat.len() {
                acc.resize(feat.len(), 0.0);
            }
            for (i, v) in feat.iter().enumerate() {
                acc[i] += v.to_f64().unwrap_or(0.0);
            }
        }
        let n = task.support_set.features.len().max(1) as f64;
        for a in acc.iter_mut() {
            *a /= n;
        }
        acc
    }

    fn signature_distance(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().max(b.len());
        let mut acc = 0.0;
        for i in 0..n {
            let d = a.get(i).copied().unwrap_or(0.0) - b.get(i).copied().unwrap_or(0.0);
            acc += d * d;
        }
        acc.sqrt()
    }
}
/// Continual learning system backed by
/// [`crate::continual_learning::ElasticWeightConsolidation`].
pub struct ContinualLearningSystem<T: Float + Debug + Send + Sync + 'static> {
    settings: ContinualLearningSettings,
    /// Gradient steps taken per task in the sequence
    steps_per_task: usize,
    /// Forgetting measured by the most recent sequence
    last_forgetting: T,
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
    > ContinualLearningSystem<T>
{
    pub fn new(settings: &ContinualLearningSettings) -> Result<Self> {
        Ok(Self {
            settings: settings.clone(),
            steps_per_task: 20,
            last_forgetting: T::zero(),
        })
    }

    /// Override the number of gradient steps taken per task.
    pub fn with_steps_per_task(mut self, steps: usize) -> Result<Self> {
        if steps == 0 {
            return Err(OptimError::InvalidConfig(
                "steps_per_task must be greater than zero".to_string(),
            ));
        }
        self.steps_per_task = steps;
        Ok(self)
    }
    /// Learn a sequence of tasks with elastic weight consolidation.
    ///
    /// Delegates to [`crate::continual_learning::ElasticWeightConsolidation`]:
    /// each task is trained with SGD on the shared linear model plus the EWC
    /// penalty gradient, then consolidated (Fisher diagonal + anchor).
    /// Forgetting is measured for real by re-evaluating every earlier task
    /// after the sequence finishes.
    pub fn learn_sequence(
        &mut self,
        sequence: &[MetaTask<T>],
        meta_parameters: &mut HashMap<String, Array1<T>>,
    ) -> Result<ContinualLearningResult<T>> {
        if sequence.is_empty() {
            return Err(OptimError::InsufficientData(
                "continual learning requires at least one task".to_string(),
            ));
        }

        let lambda: T = scirs2_core::numeric::NumCast::from(
            1.0 - self.settings.plasticity_stability_balance.clamp(0.0, 1.0),
        )
        .unwrap_or_else(T::zero);
        let uses_ewc = self
            .settings
            .anti_forgetting_strategies
            .iter()
            .any(|s| matches!(s, AntiForgettingStrategy::ElasticWeightConsolidation));

        let mut ewc = crate::continual_learning::ElasticWeightConsolidation::<T>::new(lambda)
            .with_num_samples(self.settings.memory_replay.replay_frequency.max(2));

        let lr: T = scirs2_core::numeric::NumCast::from(0.05).unwrap_or_else(T::zero);
        let mut sequence_results = Vec::with_capacity(sequence.len());
        let mut first_losses: Vec<T> = Vec::with_capacity(sequence.len());
        let mut total_steps = 0usize;

        for (task_idx, task) in sequence.iter().enumerate() {
            let before = crate::meta_learning::linear_model::mse_loss(
                &task.support_set.features,
                &task.support_set.targets,
                meta_parameters,
            )?;

            for _ in 0..self.steps_per_task {
                let mut grads = crate::meta_learning::linear_model::mse_gradients(
                    &task.support_set.features,
                    &task.support_set.targets,
                    meta_parameters,
                )?;
                if uses_ewc && task_idx > 0 {
                    let penalty = ewc.ewc_gradient(meta_parameters)?;
                    for (name, g) in grads.iter_mut() {
                        if let Some(p) = penalty.get(name) {
                            let n = g.len().min(p.len());
                            for i in 0..n {
                                g[i] = g[i] + p[i];
                            }
                        }
                    }
                }
                crate::meta_learning::linear_model::descend(meta_parameters, &grads, lr)?;
                total_steps += 1;
            }

            let after = crate::meta_learning::linear_model::mse_loss(
                &task.support_set.features,
                &task.support_set.targets,
                meta_parameters,
            )?;
            first_losses.push(after);

            if uses_ewc {
                // Fisher information from per-sample gradients of this task.
                let features = task.support_set.features.clone();
                let targets = task.support_set.targets.clone();
                ewc.compute_fisher_diagonal(meta_parameters, |params, sample_idx| {
                    let i = sample_idx % features.len().max(1);
                    let feats = features.get(i).cloned().ok_or_else(|| {
                        OptimError::InsufficientData("empty support set".to_string())
                    })?;
                    let target = targets.get(i).copied().ok_or_else(|| {
                        OptimError::InsufficientData("empty target set".to_string())
                    })?;
                    crate::meta_learning::linear_model::mse_gradients(&[feats], &[target], params)
                })?;
                ewc.consolidate(meta_parameters)?;
            }

            let mut metrics = HashMap::new();
            metrics.insert("loss_before".to_string(), before);
            metrics.insert("loss_after".to_string(), after);
            sequence_results.push(TaskResult {
                task_id: task.id.clone(),
                loss: after,
                metrics,
            });
        }

        // Forgetting: how much each earlier task's loss regressed once the
        // whole sequence has been learned.
        let mut regressions: Vec<T> = Vec::with_capacity(sequence.len());
        for (idx, task) in sequence.iter().enumerate() {
            let now = crate::meta_learning::linear_model::mse_loss(
                &task.support_set.features,
                &task.support_set.targets,
                meta_parameters,
            )?;
            if let Some(right_after) = first_losses.get(idx) {
                regressions.push((now - *right_after).max(T::zero()));
            }
        }
        let forgetting_measure =
            crate::meta_learning::metrics::mean_of(&regressions).unwrap_or_else(T::zero);
        self.last_forgetting = forgetting_measure;

        let improvement = sequence_results
            .iter()
            .filter_map(|r| r.metrics.get("loss_before").map(|b| *b - r.loss))
            .fold(T::zero(), |a, b| a + b);
        let steps_t = T::from(total_steps.max(1)).unwrap_or_else(T::one);
        let adaptation_efficiency = improvement / steps_t;

        Ok(ContinualLearningResult {
            sequence_results,
            forgetting_measure,
            adaptation_efficiency,
        })
    }

    /// Forgetting measured by the most recent [`Self::learn_sequence`] call.
    ///
    /// Zero before any sequence has been learned.
    pub fn forgetting_measure(&self) -> T {
        self.last_forgetting
    }
}
/// Few-shot learner over the shared linear task model.
///
/// Adapts the meta-parameters on the support set with analytic-gradient SGD,
/// then scores the query set. Every reported figure is measured.
pub struct FewShotLearner<T: Float + Debug + Send + Sync + 'static> {
    settings: FewShotSettings,
    /// Gradient steps taken on the support set
    adaptation_steps: usize,
    /// Support-set learning rate
    learning_rate: T,
    /// Accuracy of every episode learned so far
    episode_accuracies: Vec<T>,
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
    pub fn new(settings: &FewShotSettings) -> Result<Self> {
        Ok(Self {
            settings: settings.clone(),
            adaptation_steps: settings.num_shots.max(1) * 4,
            learning_rate: scirs2_core::numeric::NumCast::from(0.05).unwrap_or_else(T::zero),
            episode_accuracies: Vec::new(),
        })
    }

    /// The configured few-shot algorithm.
    pub fn algorithm(&self) -> FewShotAlgorithm {
        self.settings.algorithm
    }

    /// Adapt on `support_set` and score `query_set`.
    pub fn learn(
        &mut self,
        support_set: &TaskDataset<T>,
        query_set: &TaskDataset<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<FewShotResult<T>> {
        if support_set.features.is_empty() {
            return Err(OptimError::InsufficientData(
                "few-shot learning requires a non-empty support set".to_string(),
            ));
        }
        if query_set.features.is_empty() {
            return Err(OptimError::InsufficientData(
                "few-shot learning requires a non-empty query set".to_string(),
            ));
        }

        let mut adapted = meta_parameters.clone();
        for _ in 0..self.adaptation_steps {
            let grads = crate::meta_learning::linear_model::mse_gradients(
                &support_set.features,
                &support_set.targets,
                &adapted,
            )?;
            crate::meta_learning::linear_model::descend(&mut adapted, &grads, self.learning_rate)?;
        }

        let predictions =
            crate::meta_learning::linear_model::predict_all(&query_set.features, &adapted)?;
        // Few-shot benchmarks are classification: score by label match. When
        // the targets are not integral labels, fall back to clamped R^2.
        let accuracy =
            crate::meta_learning::linear_model::label_accuracy(&predictions, &query_set.targets)
                .or_else(|| {
                    crate::meta_learning::linear_model::r_squared(&predictions, &query_set.targets)
                        .map(|r| r.max(T::zero()).min(T::one()))
                })
                .unwrap_or_else(T::zero);

        // Per-sample uncertainty: the absolute residual, largest where the
        // adapted model is least sure.
        let uncertainty_estimates: Vec<T> = predictions
            .iter()
            .zip(query_set.targets.iter())
            .map(|(p, y)| (*p - *y).abs())
            .collect();
        let mean_uncertainty =
            crate::meta_learning::metrics::mean_of(&uncertainty_estimates).unwrap_or_else(T::zero);
        let confidence = T::one() / (T::one() + mean_uncertainty);

        self.episode_accuracies.push(accuracy);

        Ok(FewShotResult {
            accuracy,
            confidence,
            adaptation_steps: self.adaptation_steps,
            uncertainty_estimates,
        })
    }

    /// Mean accuracy across every episode learned so far.
    ///
    /// Zero before any episode has been learned.
    pub fn average_performance(&self) -> T {
        crate::meta_learning::metrics::mean_of(&self.episode_accuracies).unwrap_or_else(T::zero)
    }
}
/// Meta-optimization tracker.
///
/// Records what meta-training actually did: how many epochs ran, how many
/// tasks were consumed, the loss trajectory, and the best parameters seen.
pub struct MetaOptimizationTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Number of recorded epochs
    step_count: usize,
    /// Number of tasks actually consumed
    tasks_seen: usize,
    /// Training loss per recorded epoch
    training_losses: Vec<f64>,
    /// Validation loss per recorded epoch
    validation_losses: Vec<f64>,
    /// Best meta-parameters seen so far
    best_parameters: Option<MetaParameters<T>>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> MetaOptimizationTracker<T> {
    pub fn new() -> Self {
        Self {
            step_count: 0,
            tasks_seen: 0,
            training_losses: Vec::new(),
            validation_losses: Vec::new(),
            best_parameters: None,
        }
    }

    /// Record one meta-training epoch. `tasks_in_epoch` is the number of tasks
    /// the epoch actually consumed.
    pub fn record_epoch(
        &mut self,
        _epoch: usize,
        training_result: &TrainingResult,
        validation_result: &ValidationResult,
        tasks_in_epoch: usize,
    ) -> Result<()> {
        self.step_count += 1;
        self.tasks_seen += tasks_in_epoch;
        self.training_losses.push(training_result.training_loss);
        self.validation_losses
            .push(validation_result.validation_loss);
        Ok(())
    }

    pub fn update_best_parameters(&mut self, metaparameters: &MetaParameters<T>) -> Result<()> {
        self.best_parameters = Some(metaparameters.clone());
        Ok(())
    }

    /// The best meta-parameters recorded so far, if any.
    pub fn best_parameters(&self) -> Option<&MetaParameters<T>> {
        self.best_parameters.as_ref()
    }

    /// Number of epochs recorded.
    pub fn epochs_recorded(&self) -> usize {
        self.step_count
    }

    /// Number of tasks actually consumed across all recorded epochs.
    pub fn total_tasks_seen(&self) -> usize {
        self.tasks_seen
    }

    /// Mean training-loss improvement per epoch.
    ///
    /// Returns zero until at least two epochs have been recorded, because a
    /// single epoch provides no improvement to measure.
    pub fn adaptation_efficiency(&self) -> T {
        if self.training_losses.len() < 2 {
            return T::zero();
        }
        let first = self.training_losses.first().copied().unwrap_or(0.0);
        let last = self.training_losses.last().copied().unwrap_or(0.0);
        let epochs = (self.training_losses.len() - 1) as f64;
        T::from((first - last) / epochs).unwrap_or_else(T::zero)
    }

    /// Validation loss trajectory (one entry per recorded epoch).
    pub fn validation_losses(&self) -> &[f64] {
        &self.validation_losses
    }
}
/// Meta-Learning Framework for Learned Optimizers
pub struct MetaLearningFramework<T: Float + Debug + Send + Sync + 'static> {
    /// Meta-learning configuration
    config: MetaLearningConfig,
    /// Meta-learner implementation
    meta_learner: Box<dyn MetaLearner<T> + Send + Sync>,
    /// Task distribution manager
    task_manager: TaskDistributionManager<T>,
    /// Meta-validation system
    meta_validator: MetaValidator<T>,
    /// Adaptation engine
    adaptation_engine: AdaptationEngine<T>,
    /// Transfer learning manager
    transfer_manager: TransferLearningManager<T>,
    /// Continual learning system
    continual_learner: ContinualLearningSystem<T>,
    /// Multi-task coordinator
    multitask_coordinator: MultiTaskCoordinator<T>,
    /// Meta-optimization tracker
    meta_tracker: MetaOptimizationTracker<T>,
    /// Few-shot learning specialist
    few_shot_learner: FewShotLearner<T>,
}
impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + for<'a> std::iter::Sum<&'a T>
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug,
    > MetaLearningFramework<T>
{
    /// Create a new meta-learning framework
    pub fn new(config: MetaLearningConfig) -> Result<Self> {
        let meta_learner = Self::create_meta_learner(&config)?;
        let task_manager = TaskDistributionManager::new(&config)?;
        let meta_validator = MetaValidator::new(&config)?;
        let adaptation_engine = AdaptationEngine::new(&config)?;
        let transfer_manager = TransferLearningManager::new(&config.transfer_settings)?;
        let continual_learner = ContinualLearningSystem::new(&config.continual_settings)?;
        let multitask_coordinator = MultiTaskCoordinator::new(&config.multitask_settings)?;
        let meta_tracker = MetaOptimizationTracker::new();
        let few_shot_learner = FewShotLearner::new(&config.few_shot_settings)?;
        Ok(Self {
            config,
            meta_learner,
            task_manager,
            meta_validator,
            adaptation_engine,
            transfer_manager,
            continual_learner,
            multitask_coordinator,
            meta_tracker,
            few_shot_learner,
        })
    }
    /// Build the meta-learner named by `config.algorithm`.
    ///
    /// Only the algorithms that have a real implementation are constructed;
    /// every other variant returns [`OptimError::InvalidConfig`] rather than
    /// silently substituting a different algorithm.
    fn create_meta_learner(
        config: &MetaLearningConfig,
    ) -> Result<Box<dyn MetaLearner<T> + Send + Sync>> {
        let inner_lr: T = scirs2_core::numeric::NumCast::from(config.inner_learning_rate)
            .ok_or_else(|| {
                OptimError::InvalidConfig("inner learning rate is not representable".to_string())
            })?;
        let outer_lr: T = scirs2_core::numeric::NumCast::from(config.meta_learning_rate)
            .ok_or_else(|| {
                OptimError::InvalidConfig("meta learning rate is not representable".to_string())
            })?;

        match config.algorithm {
            MetaLearningAlgorithm::MAML | MetaLearningAlgorithm::FOMAML => {
                let maml_config = MAMLConfig {
                    // FOMAML is MAML with the second-order term dropped.
                    second_order: matches!(config.algorithm, MetaLearningAlgorithm::MAML)
                        && config.second_order,
                    inner_lr,
                    outer_lr,
                    inner_steps: config.inner_steps,
                    allow_unused: true,
                    gradient_clip: Some(config.gradient_clip),
                };
                Ok(Box::new(MAMLLearner::<T, scirs2_core::ndarray::Ix1>::new(
                    maml_config,
                )?))
            }
            MetaLearningAlgorithm::Reptile => Ok(Box::new(
                crate::meta_learning::reptile_learner::ReptileLearner::new(outer_lr, inner_lr)
                    .with_inner_steps(config.inner_steps),
            )),
            MetaLearningAlgorithm::MetaSGD => Ok(Box::new(
                crate::meta_learning::meta_sgd_learner::MetaSGDLearner::new(inner_lr)
                    .with_alpha_lr(outer_lr)
                    .with_outer_lr(outer_lr)
                    .with_inner_steps(config.inner_steps),
            )),
            other => Err(OptimError::InvalidConfig(format!(
                "meta-learning algorithm {other:?} is not implemented; \
                 available algorithms are MAML, FOMAML, Reptile and MetaSGD"
            ))),
        }
    }
    /// The meta-learning algorithm the dispatched learner actually implements.
    pub fn algorithm(&self) -> MetaLearningAlgorithm {
        self.meta_learner.get_algorithm()
    }

    /// The configuration this framework was built from.
    pub fn config(&self) -> &MetaLearningConfig {
        &self.config
    }

    /// Perform meta-training.
    ///
    /// **Update contract:** the meta-learner owns the meta-update. Its
    /// `meta_train_step` writes the new meta-parameters and reports the
    /// gradient it used; this loop does not apply that gradient a second time.
    pub async fn meta_train(
        &mut self,
        tasks: Vec<MetaTask<T>>,
        num_epochs: usize,
    ) -> Result<MetaTrainingResults<T>> {
        if tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "meta-training requires a non-empty task pool".to_string(),
            ));
        }
        let meta_params_raw = self.initialize_meta_parameters(&tasks)?;
        let mut meta_parameters = MetaParameters {
            parameters: meta_params_raw,
            metadata: HashMap::new(),
        };
        let mut training_history = Vec::new();
        let mut best_performance = T::neg_infinity();
        let batch_size = self.config.task_batch_size;
        for epoch in 0..num_epochs {
            let task_batch = self.task_manager.sample_task_batch(&tasks, batch_size)?;
            let batch_len = task_batch.len();
            // The learner applies the meta-update itself; see the contract above.
            let training_result = self
                .meta_learner
                .meta_train_step(&task_batch, &mut meta_parameters.parameters)?;
            let validation_result =
                self.meta_validator
                    .validate(&meta_parameters, &tasks, &mut *self.meta_learner)?;
            let mut training_metrics = HashMap::new();
            training_metrics.insert(
                "gradient_alignment".to_string(),
                training_result
                    .metrics
                    .gradient_alignment
                    .to_f64()
                    .unwrap_or(0.0),
            );
            training_metrics.insert(
                "adaptation_speed".to_string(),
                training_result
                    .metrics
                    .avg_adaptation_speed
                    .to_f64()
                    .unwrap_or(0.0),
            );
            let training_result_simple = TrainingResult {
                training_loss: training_result.meta_loss.to_f64().unwrap_or(f64::NAN),
                metrics: training_metrics,
                steps: epoch,
            };
            self.meta_tracker.record_epoch(
                epoch,
                &training_result_simple,
                &validation_result,
                batch_len,
            )?;
            let current_performance =
                T::from(-validation_result.validation_loss).unwrap_or_default();
            if current_performance > best_performance {
                best_performance = current_performance;
                self.meta_tracker.update_best_parameters(&meta_parameters)?;
            }
            let mut task_specific_metrics = HashMap::new();
            for (name, value) in &validation_result.metrics {
                if let Some(v) = T::from(*value) {
                    task_specific_metrics.insert(name.clone(), v);
                }
            }
            // Generalization gap: held-out loss minus meta-training loss.
            let generalization_gap = T::from(validation_result.validation_loss)
                .unwrap_or_else(T::zero)
                - training_result.meta_loss;
            let meta_validation_result = MetaValidationResult {
                performance: current_performance,
                adaptation_speed: training_result.metrics.avg_adaptation_speed,
                generalization_gap,
                task_specific_metrics,
            };
            training_history.push(MetaTrainingEpoch {
                epoch,
                training_result,
                validation_result: meta_validation_result,
                meta_parameters: meta_parameters.parameters.clone(),
            });
            if self.should_early_stop(&training_history) {
                break;
            }
        }
        let total_epochs = training_history.len();
        Ok(MetaTrainingResults {
            final_parameters: meta_parameters.parameters,
            training_history,
            best_performance,
            total_epochs,
        })
    }
    /// Adapt to new task
    pub fn adapt_to_task(
        &mut self,
        task: &MetaTask<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<TaskAdaptationResult<T>> {
        self.adaptation_engine.adapt(
            task,
            meta_parameters,
            &mut *self.meta_learner,
            self.config.inner_steps,
        )
    }
    /// Perform few-shot learning
    pub fn few_shot_learning(
        &mut self,
        support_set: &TaskDataset<T>,
        query_set: &TaskDataset<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<FewShotResult<T>> {
        self.few_shot_learner
            .learn(support_set, query_set, meta_parameters)
    }
    /// Transfer learning to new domain
    pub fn transfer_to_domain(
        &mut self,
        source_tasks: &[MetaTask<T>],
        target_tasks: &[MetaTask<T>],
        meta_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<TransferLearningResult<T>> {
        self.transfer_manager
            .transfer(source_tasks, target_tasks, meta_parameters)
    }
    /// Continual learning across task sequence
    pub fn continual_learning(
        &mut self,
        task_sequence: &[MetaTask<T>],
        meta_parameters: &mut HashMap<String, Array1<T>>,
    ) -> Result<ContinualLearningResult<T>> {
        self.continual_learner
            .learn_sequence(task_sequence, meta_parameters)
    }
    /// Multi-task learning
    pub fn multi_task_learning(
        &mut self,
        tasks: &[MetaTask<T>],
        meta_parameters: &mut HashMap<String, Array1<T>>,
    ) -> Result<MultiTaskResult<T>> {
        self.multitask_coordinator
            .learn_simultaneously(tasks, meta_parameters)
    }
    /// Initialise the meta-parameters of the shared linear task model.
    ///
    /// The weight vector is sized from the widest support feature vector in the
    /// pool and initialised with small deterministic pseudo-random values
    /// (Xavier-style scale `sqrt(1 / fan_in)`); a zero initialisation would put
    /// every task at the same stationary starting point.
    pub fn initialize_meta_parameters(
        &self,
        tasks: &[MetaTask<T>],
    ) -> Result<HashMap<String, Array1<T>>> {
        let feature_dim = tasks
            .iter()
            .flat_map(|t| {
                t.support_set
                    .features
                    .iter()
                    .chain(t.query_set.features.iter())
            })
            .map(|f| f.len())
            .max()
            .unwrap_or(0);
        if feature_dim == 0 {
            return Err(OptimError::InsufficientData(
                "cannot size meta-parameters: no task carries any features".to_string(),
            ));
        }

        let scale = 1.0 / (feature_dim as f64).sqrt();
        let mut rng = scirs2_core::random::Random::seed(0x1234_5678_9ABC_DEF0);
        let mut weights = Array1::zeros(feature_dim);
        for i in 0..feature_dim {
            let v: f64 = rng.random_range(-scale..scale);
            weights[i] = scirs2_core::numeric::NumCast::from(v).unwrap_or_else(T::zero);
        }

        let mut parameters = HashMap::new();
        parameters.insert(
            crate::meta_learning::linear_model::WEIGHTS_KEY.to_string(),
            weights,
        );
        parameters.insert(
            crate::meta_learning::linear_model::BIAS_KEY.to_string(),
            Array1::zeros(1),
        );
        Ok(parameters)
    }

    /// Stop when the held-out validation loss has plateaued.
    ///
    /// The decision uses the *real* validation losses recorded by the
    /// meta-validator; the old implementation compared a constant to itself and
    /// therefore stopped every run at epoch 10.
    fn should_early_stop(&self, history: &[MetaTrainingEpoch<T>]) -> bool {
        const WINDOW: usize = 5;
        if history.len() < 2 * WINDOW {
            return false;
        }
        let recent: Vec<T> = history
            .iter()
            .rev()
            .take(WINDOW)
            .map(|epoch| epoch.validation_result.performance)
            .collect();
        let max_recent = recent.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
        let min_recent = recent.iter().fold(T::infinity(), |a, &b| a.min(b));
        if !max_recent.is_finite() || !min_recent.is_finite() {
            return false;
        }
        let spread = max_recent - min_recent;
        let scale = max_recent.abs().max(T::one());
        let threshold: T = scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(T::zero);
        spread / scale < threshold
    }
    /// Get meta-learning statistics
    pub fn get_meta_learning_statistics(&self) -> MetaLearningStatistics<T> {
        MetaLearningStatistics {
            algorithm: self.config.algorithm,
            total_tasks_seen: self.meta_tracker.total_tasks_seen(),
            adaptation_efficiency: self.meta_tracker.adaptation_efficiency(),
            transfer_success_rate: self.transfer_manager.success_rate(),
            forgetting_measure: self.continual_learner.forgetting_measure(),
            multitask_interference: self.multitask_coordinator.interference_measure(),
            few_shot_performance: self.few_shot_learner.average_performance(),
        }
    }
}
/// Transfer learning manager backed by
/// [`crate::cross_domain_transfer::CrossDomainTransfer`].
pub struct TransferLearningManager<T: Float + Debug + Send + Sync + 'static> {
    settings: TransferLearningSettings,
    /// Number of transfers attempted so far
    attempts: usize,
    /// Number of transfers that improved the target-task loss
    successes: usize,
    _phantom: std::marker::PhantomData<T>,
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
    > TransferLearningManager<T>
{
    pub fn new(settings: &TransferLearningSettings) -> Result<Self> {
        Ok(Self {
            settings: settings.clone(),
            attempts: 0,
            successes: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Whether the configured settings request domain adaptation.
    pub fn domain_adaptation_enabled(&self) -> bool {
        self.settings.domain_adaptation
    }

    /// Transfer from a set of source tasks to a set of target tasks.
    ///
    /// Both task groups are summarised as domains (mean support features and
    /// the fitted linear parameters), registered with the real
    /// [`crate::cross_domain_transfer::CrossDomainTransfer`] engine, and the
    /// transferred parameters are evaluated on the target tasks. Every number
    /// reported here is measured, not assumed.
    pub fn transfer(
        &mut self,
        source_tasks: &[MetaTask<T>],
        target_tasks: &[MetaTask<T>],
        meta_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<TransferLearningResult<T>> {
        if source_tasks.is_empty() || target_tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "transfer requires at least one source and one target task".to_string(),
            ));
        }
        let weights = meta_parameters
            .get(crate::meta_learning::linear_model::WEIGHTS_KEY)
            .ok_or_else(|| {
                OptimError::InvalidConfig(
                    "transfer requires a 'weights' entry in the meta-parameters".to_string(),
                )
            })?;

        let source_repr = Self::group_signature(source_tasks, weights.len())?;
        let target_repr = Self::group_signature(target_tasks, weights.len())?;

        let mut engine = crate::cross_domain_transfer::CrossDomainTransfer::<T>::new(weights.len());
        engine.register_domain(crate::cross_domain_transfer::DomainKnowledge {
            domain_name: "source".to_string(),
            shared_representation: source_repr,
            domain_specific_params: weights.clone(),
            performance_history: Vec::new(),
        })?;
        engine.register_domain(crate::cross_domain_transfer::DomainKnowledge {
            domain_name: "target".to_string(),
            shared_representation: target_repr,
            domain_specific_params: weights.clone(),
            performance_history: Vec::new(),
        })?;

        let similarity = engine.compute_domain_similarity("source", "target")?;
        let result = engine.transfer("source", "target")?;

        let mut transferred = meta_parameters.clone();
        transferred.insert(
            crate::meta_learning::linear_model::WEIGHTS_KEY.to_string(),
            result.transferred_params.clone(),
        );

        let source_before = Self::group_loss(source_tasks, meta_parameters)?;
        let source_after = Self::group_loss(source_tasks, &transferred)?;
        let target_before = Self::group_loss(target_tasks, meta_parameters)?;
        let target_after = Self::group_loss(target_tasks, &transferred)?;

        self.attempts += 1;
        if target_after < target_before {
            self.successes += 1;
        }

        let relative = |before: T, after: T| -> T {
            if before > T::zero() {
                ((before - after) / before).max(T::zero()).min(T::one())
            } else {
                T::zero()
            }
        };

        Ok(TransferLearningResult {
            transfer_efficiency: relative(target_before, target_after),
            // Cosine similarity mapped from [-1, 1] to [0, 1].
            domain_adaptation_score: (similarity + T::one()) / (T::one() + T::one()),
            // Retention: how much of the source performance survived.
            source_task_retention: if source_after > T::zero() {
                (source_before / source_after).min(T::one())
            } else {
                T::one()
            },
            target_task_performance: T::one() / (T::one() + target_after),
        })
    }

    /// Fraction of transfers that improved the target-task loss.
    ///
    /// Returns zero before any transfer has been attempted.
    pub fn success_rate(&self) -> T {
        if self.attempts == 0 {
            return T::zero();
        }
        let a = T::from(self.attempts).unwrap_or_else(T::one);
        T::from(self.successes).unwrap_or_else(T::zero) / a
    }

    /// Mean support-feature vector across a group of tasks, resized to `dim`.
    fn group_signature(tasks: &[MetaTask<T>], dim: usize) -> Result<Array1<T>> {
        let mut acc: Array1<T> = Array1::zeros(dim);
        let mut count = 0usize;
        for task in tasks {
            for feat in &task.support_set.features {
                let n = dim.min(feat.len());
                for i in 0..n {
                    acc[i] = acc[i] + feat[i];
                }
                count += 1;
            }
        }
        if count == 0 {
            return Err(OptimError::InsufficientData(
                "task group has no support samples".to_string(),
            ));
        }
        let c = T::from(count).unwrap_or_else(T::one);
        Ok(acc.mapv(|v| v / c))
    }

    /// Mean query loss of a group of tasks at the given parameters.
    fn group_loss(tasks: &[MetaTask<T>], parameters: &HashMap<String, Array1<T>>) -> Result<T> {
        let mut total = T::zero();
        let mut count = 0usize;
        for task in tasks {
            if task.query_set.features.is_empty() {
                continue;
            }
            total = total
                + crate::meta_learning::linear_model::mse_loss(
                    &task.query_set.features,
                    &task.query_set.targets,
                    parameters,
                )?;
            count += 1;
        }
        if count == 0 {
            return Err(OptimError::InsufficientData(
                "task group has no query samples".to_string(),
            ));
        }
        let c = T::from(count).unwrap_or_else(T::one);
        Ok(total / c)
    }
}
/// Multi-task coordinator.
///
/// Trains one shared parameter vector on all tasks at once, combining the
/// per-task gradients according to [`TaskWeightingStrategy`] and
/// [`GradientBalancingMethod`], and measures the real gradient interference
/// between tasks.
pub struct MultiTaskCoordinator<T: Float + Debug + Send + Sync + 'static> {
    settings: MultiTaskSettings,
    /// Joint gradient steps per `learn_simultaneously` call
    steps: usize,
    /// Interference measured by the most recent call
    last_interference: T,
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
    > MultiTaskCoordinator<T>
{
    pub fn new(settings: &MultiTaskSettings) -> Result<Self> {
        Ok(Self {
            settings: settings.clone(),
            steps: 20,
            last_interference: T::zero(),
        })
    }

    /// Override the number of joint gradient steps.
    pub fn with_steps(mut self, steps: usize) -> Result<Self> {
        if steps == 0 {
            return Err(OptimError::InvalidConfig(
                "multi-task steps must be greater than zero".to_string(),
            ));
        }
        self.steps = steps;
        Ok(self)
    }

    pub fn learn_simultaneously(
        &mut self,
        tasks: &[MetaTask<T>],
        meta_parameters: &mut HashMap<String, Array1<T>>,
    ) -> Result<MultiTaskResult<T>> {
        if tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "multi-task learning requires at least one task".to_string(),
            ));
        }

        let lr: T = scirs2_core::numeric::NumCast::from(0.05).unwrap_or_else(T::zero);
        let start = std::time::Instant::now();
        let mut last_gradients: Vec<HashMap<String, Array1<T>>> = Vec::new();
        let mut initial_losses: Vec<T> = Vec::with_capacity(tasks.len());
        for task in tasks {
            initial_losses.push(crate::meta_learning::linear_model::mse_loss(
                &task.support_set.features,
                &task.support_set.targets,
                meta_parameters,
            )?);
        }

        for _ in 0..self.steps {
            let mut per_task: Vec<HashMap<String, Array1<T>>> = Vec::with_capacity(tasks.len());
            let mut losses: Vec<T> = Vec::with_capacity(tasks.len());
            for task in tasks {
                losses.push(crate::meta_learning::linear_model::mse_loss(
                    &task.support_set.features,
                    &task.support_set.targets,
                    meta_parameters,
                )?);
                per_task.push(crate::meta_learning::linear_model::mse_gradients(
                    &task.support_set.features,
                    &task.support_set.targets,
                    meta_parameters,
                )?);
            }

            let weights = self.task_weights(&losses, &per_task)?;
            let mut combined: HashMap<String, Array1<T>> = meta_parameters
                .iter()
                .map(|(name, p)| (name.clone(), Array1::zeros(p.len())))
                .collect();
            for (grad, weight) in per_task.iter().zip(weights.iter()) {
                for (name, g) in grad {
                    if let Some(acc) = combined.get_mut(name) {
                        let n = acc.len().min(g.len());
                        for i in 0..n {
                            acc[i] = acc[i] + *weight * g[i];
                        }
                    }
                }
            }
            crate::meta_learning::linear_model::descend(meta_parameters, &combined, lr)?;
            last_gradients = per_task;
        }

        // Interference: mean pairwise *conflict* between task gradients,
        // i.e. (1 - cosine)/2 in [0, 1]. Orthogonal tasks score 0.5, opposed
        // tasks score 1.
        let mut conflicts: Vec<T> = Vec::new();
        for i in 0..last_gradients.len() {
            for j in (i + 1)..last_gradients.len() {
                if let Some(c) = crate::meta_learning::linear_model::map_cosine(
                    &last_gradients[i],
                    &last_gradients[j],
                ) {
                    conflicts.push((T::one() - c) / (T::one() + T::one()));
                }
            }
        }
        self.last_interference =
            crate::meta_learning::metrics::mean_of(&conflicts).unwrap_or_else(T::zero);

        let mut task_results = Vec::with_capacity(tasks.len());
        let mut converged = true;
        for (idx, task) in tasks.iter().enumerate() {
            let loss = crate::meta_learning::linear_model::mse_loss(
                &task.support_set.features,
                &task.support_set.targets,
                meta_parameters,
            )?;
            if let Some(before) = initial_losses.get(idx) {
                if loss >= *before {
                    converged = false;
                }
            }
            let mut metrics = HashMap::new();
            if let Some(before) = initial_losses.get(idx) {
                metrics.insert("loss_before".to_string(), *before);
            }
            metrics.insert("loss_after".to_string(), loss);
            task_results.push(TaskResult {
                task_id: task.id.clone(),
                loss,
                metrics,
            });
        }

        let elapsed = T::from(start.elapsed().as_secs_f64()).unwrap_or_else(T::zero);
        Ok(MultiTaskResult {
            task_results,
            coordination_overhead: elapsed,
            convergence_status: if converged {
                "converged".to_string()
            } else {
                "not_converged".to_string()
            },
        })
    }

    /// Per-task gradient weights implied by the configured strategies.
    fn task_weights(
        &self,
        losses: &[T],
        gradients: &[HashMap<String, Array1<T>>],
    ) -> Result<Vec<T>> {
        let n = T::from(losses.len().max(1)).unwrap_or_else(T::one);
        let raw: Vec<T> = match self.settings.task_weighting {
            TaskWeightingStrategy::Uniform => vec![T::one(); losses.len()],
            TaskWeightingStrategy::PerformanceBased | TaskWeightingStrategy::Adaptive => {
                // Focus on the tasks that are currently doing worst.
                losses.iter().map(|l| l.max(T::zero())).collect()
            }
            TaskWeightingStrategy::UncertaintyBased => {
                // Homoscedastic-uncertainty weighting: 1 / (1 + loss).
                losses
                    .iter()
                    .map(|l| T::one() / (T::one() + l.max(T::zero())))
                    .collect()
            }
            TaskWeightingStrategy::GradientMagnitude | TaskWeightingStrategy::Learned => gradients
                .iter()
                .map(crate::meta_learning::linear_model::map_norm)
                .collect(),
        };

        let total = raw.iter().copied().fold(T::zero(), |a, b| a + b);
        let normalised: Vec<T> = if total > T::zero() {
            raw.iter().map(|w| *w / total).collect()
        } else {
            vec![T::one() / n; losses.len()]
        };

        Ok(match self.settings.gradient_balancing {
            // GradNorm-style rescaling keeps every task's contribution at a
            // comparable magnitude; the other methods use the weights directly.
            GradientBalancingMethod::GradNorm => {
                let norms: Vec<T> = gradients
                    .iter()
                    .map(crate::meta_learning::linear_model::map_norm)
                    .collect();
                let mean_norm =
                    crate::meta_learning::metrics::mean_of(&norms).unwrap_or_else(T::one);
                normalised
                    .iter()
                    .zip(norms.iter())
                    .map(|(w, nrm)| {
                        if *nrm > T::zero() {
                            *w * mean_norm / *nrm
                        } else {
                            *w
                        }
                    })
                    .collect()
            }
            _ => normalised,
        })
    }

    /// Gradient interference measured by the most recent call, in `[0, 1]`.
    ///
    /// Zero before any multi-task step has run.
    pub fn interference_measure(&self) -> T {
        self.last_interference
    }
}
/// Meta-validation system for meta-learning.
///
/// Holds out a deterministic slice of the task pool and measures the *real*
/// post-adaptation query loss of the current meta-parameters on it.
pub struct MetaValidator<T: Float + Debug + Send + Sync + 'static> {
    config: MetaLearningConfig,
    /// Fraction of the task pool reserved for validation
    holdout_fraction: f64,
    _phantom: std::marker::PhantomData<T>,
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
    > MetaValidator<T>
{
    pub fn new(config: &MetaLearningConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            holdout_fraction: 0.25,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Override the held-out fraction (must be in `(0, 1)`).
    pub fn with_holdout_fraction(mut self, fraction: f64) -> Result<Self> {
        if !(fraction > 0.0 && fraction < 1.0) {
            return Err(OptimError::InvalidConfig(
                "holdout fraction must lie strictly between 0 and 1".to_string(),
            ));
        }
        self.holdout_fraction = fraction;
        Ok(self)
    }

    /// Indices of the held-out tasks: the deterministic tail of the pool.
    fn holdout_range(&self, len: usize) -> std::ops::Range<usize> {
        if len == 0 {
            return 0..0;
        }
        let count = ((len as f64) * self.holdout_fraction).ceil() as usize;
        let count = count.clamp(1, len);
        (len - count)..len
    }

    /// Validate the current meta-parameters against held-out tasks.
    ///
    /// Each held-out task is adapted with the learner's own inner loop, then
    /// scored on its query set. The reported loss is the mean of those query
    /// losses — never a constant.
    pub fn validate(
        &self,
        meta_parameters: &MetaParameters<T>,
        tasks: &[MetaTask<T>],
        meta_learner: &mut dyn MetaLearner<T>,
    ) -> Result<ValidationResult> {
        let range = self.holdout_range(tasks.len());
        let holdout = tasks.get(range).unwrap_or(&[]);
        if holdout.is_empty() {
            return Err(OptimError::InsufficientData(
                "no held-out tasks available for meta-validation".to_string(),
            ));
        }

        let mut total = 0.0f64;
        let mut pre_total = 0.0f64;
        let mut counted = 0usize;
        for task in holdout {
            if task.query_set.features.is_empty() || task.support_set.features.is_empty() {
                continue;
            }
            let pre = crate::meta_learning::linear_model::mse_loss(
                &task.query_set.features,
                &task.query_set.targets,
                &meta_parameters.parameters,
            )?;
            let adaptation = meta_learner.adapt_to_task(
                task,
                &meta_parameters.parameters,
                self.config.inner_steps,
            )?;
            let query = meta_learner.evaluate_query_set(task, &adaptation.adapted_parameters)?;
            total += query.query_loss.to_f64().unwrap_or(f64::NAN);
            pre_total += pre.to_f64().unwrap_or(f64::NAN);
            counted += 1;
        }
        if counted == 0 {
            return Err(OptimError::InsufficientData(
                "every held-out task was empty".to_string(),
            ));
        }

        let validation_loss = total / counted as f64;
        let pre_loss = pre_total / counted as f64;
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("held_out_tasks".to_string(), counted as f64);
        metrics.insert("pre_adaptation_loss".to_string(), pre_loss);
        metrics.insert("post_adaptation_loss".to_string(), validation_loss);

        Ok(ValidationResult {
            is_valid: validation_loss.is_finite(),
            validation_loss,
            metrics,
        })
    }
}
/// Adaptation engine for meta-learning
pub struct AdaptationEngine<T: Float + Debug + Send + Sync + 'static> {
    config: MetaLearningConfig,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> AdaptationEngine<T> {
    pub fn new(config: &MetaLearningConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
    /// Adapt to a task by delegating to the meta-learner's own inner loop.
    ///
    /// `inner_steps` of zero falls back to the configured `inner_steps`.
    pub fn adapt(
        &mut self,
        task: &MetaTask<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
        meta_learner: &mut dyn MetaLearner<T>,
        inner_steps: usize,
    ) -> Result<TaskAdaptationResult<T>> {
        let steps = if inner_steps == 0 {
            self.config.inner_steps
        } else {
            inner_steps
        };
        if steps == 0 {
            return Err(OptimError::InvalidConfig(
                "adaptation requires at least one inner step".to_string(),
            ));
        }
        meta_learner.adapt_to_task(task, meta_parameters, steps)
    }

    /// The adaptation strategies this engine was configured with.
    pub fn strategies(&self) -> &[AdaptationStrategy] {
        &self.config.adaptation_strategies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_learning::linear_model;

    fn basis(x: f64) -> Array1<f64> {
        let u = x / 3.0;
        Array1::from_vec(vec![1.0, u, u * u, u * u * u])
    }

    fn linear_task(id: usize, slope: f64, intercept: f64) -> MetaTask<f64> {
        let mut support_features = Vec::new();
        let mut support_targets = Vec::new();
        let mut query_features = Vec::new();
        let mut query_targets = Vec::new();
        for k in 0..10 {
            let x = -3.0 + 6.0 * (k as f64) / 9.0;
            let y = slope * x + intercept;
            if k % 2 == 0 {
                support_features.push(basis(x));
                support_targets.push(y);
            } else {
                query_features.push(basis(x));
                query_targets.push(y);
            }
        }
        MetaTask {
            id: format!("task_{id}"),
            support_set: TaskDataset {
                features: support_features,
                targets: support_targets,
                weights: Vec::new(),
                metadata: DatasetMetadata::default(),
            },
            query_set: TaskDataset {
                features: query_features,
                targets: query_targets,
                weights: Vec::new(),
                metadata: DatasetMetadata::default(),
            },
            metadata: TaskMetadata::default(),
            difficulty: slope.abs(),
            domain: format!("domain_{}", id % 2),
            task_type: TaskType::Regression,
        }
    }

    fn task_pool(n: usize) -> Vec<MetaTask<f64>> {
        (0..n)
            .map(|i| linear_task(i, 0.2 + 0.3 * i as f64, -1.0 + 0.4 * i as f64))
            .collect()
    }

    fn params(dim: usize) -> HashMap<String, Array1<f64>> {
        let mut p = HashMap::new();
        p.insert(linear_model::WEIGHTS_KEY.to_string(), Array1::zeros(dim));
        p.insert(linear_model::BIAS_KEY.to_string(), Array1::zeros(1));
        p
    }

    /// F10: batches come from the caller's pool, honour the batch size, and an
    /// empty pool is an error rather than ten default (empty) tasks.
    #[test]
    fn test_sample_task_batch_uses_real_tasks() {
        let pool = task_pool(5);
        let config = MetaLearningConfig::default();
        let mut manager = TaskDistributionManager::<f64>::new(&config).expect("manager");

        // No `.min(10)` cap: asking for 20 yields 20.
        let batch = manager.sample_task_batch(&pool, 20).expect("batch");
        assert_eq!(batch.len(), 20);
        for task in &batch {
            assert!(pool.iter().any(|t| t.id == task.id), "unknown task sampled");
            assert!(!task.support_set.features.is_empty(), "empty support set");
        }

        assert!(manager.sample_task_batch(&[], 4).is_err());
        assert!(manager.sample_task_batch(&pool, 0).is_err());
    }

    /// F10: the sampling strategy actually changes which tasks are drawn.
    #[test]
    fn test_sampling_strategies_differ() {
        let pool = task_pool(6);

        let mut easiest = MetaLearningConfig::default();
        easiest.task_sampling_strategy = TaskSamplingStrategy::Curriculum;
        let mut curriculum =
            TaskDistributionManager::<f64>::new(&easiest).expect("curriculum manager");
        let first = curriculum.sample_task_batch(&pool, 2).expect("batch");

        let mut hardest = MetaLearningConfig::default();
        hardest.task_sampling_strategy = TaskSamplingStrategy::Adversarial;
        let mut adversarial =
            TaskDistributionManager::<f64>::new(&hardest).expect("adversarial manager");
        let hard = adversarial.sample_task_batch(&pool, 2).expect("batch");

        // Curriculum starts with the easiest task, adversarial with the hardest.
        assert_ne!(first[0].id, hard[0].id);
        assert!(first[0].difficulty < hard[0].difficulty);
    }

    /// F12: the requested algorithm is the one that gets built, and
    /// unimplemented algorithms are refused instead of silently substituted.
    #[test]
    fn test_meta_learner_dispatch() {
        let cases = [
            (MetaLearningAlgorithm::MAML, MetaLearningAlgorithm::FOMAML),
            (MetaLearningAlgorithm::FOMAML, MetaLearningAlgorithm::FOMAML),
            (
                MetaLearningAlgorithm::Reptile,
                MetaLearningAlgorithm::Reptile,
            ),
            (
                MetaLearningAlgorithm::MetaSGD,
                MetaLearningAlgorithm::MetaSGD,
            ),
        ];
        for (requested, expected) in cases {
            let mut config = MetaLearningConfig::default();
            config.algorithm = requested;
            let framework = MetaLearningFramework::<f64>::new(config).expect("framework");
            assert_eq!(
                format!("{:?}", framework.algorithm()),
                format!("{expected:?}"),
                "dispatch for {requested:?}"
            );
        }

        // Second-order MAML reports MAML.
        let mut config = MetaLearningConfig::default();
        config.second_order = true;
        let framework = MetaLearningFramework::<f64>::new(config).expect("framework");
        assert!(matches!(framework.algorithm(), MetaLearningAlgorithm::MAML));

        for unimplemented in [
            MetaLearningAlgorithm::ProtoNet,
            MetaLearningAlgorithm::MatchingNet,
            MetaLearningAlgorithm::IMaml,
            MetaLearningAlgorithm::L2L,
        ] {
            let mut config = MetaLearningConfig::default();
            config.algorithm = unimplemented;
            assert!(
                MetaLearningFramework::<f64>::new(config).is_err(),
                "{unimplemented:?} must be refused, not silently replaced by MAML"
            );
        }
    }

    /// F13: the adaptation engine runs the learner's inner loop.
    #[test]
    fn test_adaptation_engine_delegates() {
        let config = MetaLearningConfig::default();
        let mut engine = AdaptationEngine::<f64>::new(&config).expect("engine");
        let mut learner = crate::meta_learning::reptile_learner::ReptileLearner::new(0.1f64, 0.3);
        let task = linear_task(0, 0.7, -0.2);
        let start = params(4);

        let result = engine.adapt(&task, &start, &mut learner, 6).expect("adapt");
        assert_eq!(result.adaptation_trajectory.len(), 6);
        let moved = result.adapted_parameters[linear_model::WEIGHTS_KEY]
            .iter()
            .any(|w| w.abs() > 0.0);
        assert!(moved, "adaptation must move the parameters");
        let baseline = linear_model::mse_loss(
            &task.support_set.features,
            &task.support_set.targets,
            &start,
        )
        .expect("loss");
        assert!(result.final_loss < baseline);
    }

    /// F14: the validator measures a real held-out loss.
    #[test]
    fn test_validator_measures_real_holdout_loss() {
        let pool = task_pool(8);
        let config = MetaLearningConfig::default();
        let validator = MetaValidator::<f64>::new(&config).expect("validator");
        let mut learner = crate::meta_learning::reptile_learner::ReptileLearner::new(0.1f64, 0.3);

        let good = MetaParameters {
            parameters: params(4),
            metadata: HashMap::new(),
        };
        let mut bad_params = params(4);
        if let Some(w) = bad_params.get_mut(linear_model::WEIGHTS_KEY) {
            w.fill(50.0);
        }
        let bad = MetaParameters {
            parameters: bad_params,
            metadata: HashMap::new(),
        };

        let good_result = validator
            .validate(&good, &pool, &mut learner)
            .expect("validate good");
        let bad_result = validator
            .validate(&bad, &pool, &mut learner)
            .expect("validate bad");

        assert!(
            bad_result.validation_loss > good_result.validation_loss,
            "the validator must respond to the parameters: good={}, bad={}",
            good_result.validation_loss,
            bad_result.validation_loss
        );
        assert!(good_result.metrics.contains_key("held_out_tasks"));
        assert!(validator.validate(&good, &[], &mut learner).is_err());
    }

    /// F14: continual learning really trains and really measures forgetting.
    #[test]
    fn test_continual_learning_uses_ewc() {
        let config = MetaLearningConfig::default();
        let mut system =
            ContinualLearningSystem::<f64>::new(&config.continual_settings).expect("system");
        assert_eq!(system.forgetting_measure(), 0.0);

        let sequence = task_pool(3);
        let mut parameters = params(4);
        let result = system
            .learn_sequence(&sequence, &mut parameters)
            .expect("sequence");

        assert_eq!(result.sequence_results.len(), 3);
        for task_result in &result.sequence_results {
            let before = task_result.metrics["loss_before"];
            assert!(
                task_result.loss < before,
                "task {} did not improve: {} -> {}",
                task_result.task_id,
                before,
                task_result.loss
            );
        }
        assert!(result.forgetting_measure >= 0.0);
        assert!(system.forgetting_measure().is_finite());
        assert!(
            ContinualLearningSystem::<f64>::new(&config.continual_settings)
                .expect("system")
                .learn_sequence(&[], &mut parameters)
                .is_err()
        );
    }

    /// F14: transfer really runs through the cross-domain engine.
    #[test]
    fn test_transfer_manager_delegates() {
        let config = MetaLearningConfig::default();
        let mut manager =
            TransferLearningManager::<f64>::new(&config.transfer_settings).expect("manager");
        assert_eq!(manager.success_rate(), 0.0);

        let source = vec![linear_task(0, 1.0, 0.0), linear_task(1, 1.1, 0.1)];
        let target = vec![linear_task(2, -1.0, 2.0)];
        let mut parameters = params(4);
        if let Some(w) = parameters.get_mut(linear_model::WEIGHTS_KEY) {
            w.fill(0.3);
        }

        let result = manager
            .transfer(&source, &target, &parameters)
            .expect("transfer");
        assert!(result.domain_adaptation_score.is_finite());
        assert!((0.0..=1.0).contains(&result.domain_adaptation_score));
        assert!(manager.transfer(&[], &target, &parameters).is_err());
    }

    /// F14: the tracker counts what actually happened.
    #[test]
    fn test_tracker_counts_are_real() {
        let mut tracker = MetaOptimizationTracker::<f64>::new();
        assert_eq!(tracker.total_tasks_seen(), 0);
        assert_eq!(tracker.adaptation_efficiency(), 0.0);

        for (epoch, loss) in [(0usize, 1.0f64), (1, 0.5), (2, 0.25)] {
            let training = TrainingResult {
                training_loss: loss,
                metrics: HashMap::new(),
                steps: epoch,
            };
            let validation = ValidationResult {
                is_valid: true,
                validation_loss: loss,
                metrics: HashMap::new(),
            };
            tracker
                .record_epoch(epoch, &training, &validation, 4)
                .expect("record");
        }

        assert_eq!(tracker.epochs_recorded(), 3);
        // 3 epochs x 4 tasks -- not "epochs * 10".
        assert_eq!(tracker.total_tasks_seen(), 12);
        approx::assert_abs_diff_eq!(tracker.adaptation_efficiency(), 0.375, epsilon = 1e-12);
        assert_eq!(tracker.validation_losses().len(), 3);
    }

    /// F14: few-shot learning adapts and scores for real.
    #[test]
    fn test_few_shot_learner_is_real() {
        let config = MetaLearningConfig::default();
        let mut learner = FewShotLearner::<f64>::new(&config.few_shot_settings).expect("learner");
        assert_eq!(learner.average_performance(), 0.0);

        let task = linear_task(0, 0.5, 0.0);
        let result = learner
            .learn(&task.support_set, &task.query_set, &params(4))
            .expect("few-shot");
        assert_eq!(
            result.uncertainty_estimates.len(),
            task.query_set.features.len()
        );
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
        assert!(learner.average_performance().is_finite());

        let empty = TaskDataset::<f64>::default();
        assert!(learner.learn(&empty, &task.query_set, &params(4)).is_err());
    }

    /// F14: multi-task learning trains the shared parameters and measures real
    /// gradient interference.
    #[test]
    fn test_multi_task_coordinator_is_real() {
        let config = MetaLearningConfig::default();
        let mut coordinator =
            MultiTaskCoordinator::<f64>::new(&config.multitask_settings).expect("coordinator");
        assert_eq!(coordinator.interference_measure(), 0.0);

        let tasks = vec![linear_task(0, 1.0, 0.0), linear_task(1, -1.0, 0.0)];
        let mut parameters = params(4);
        let result = coordinator
            .learn_simultaneously(&tasks, &mut parameters)
            .expect("multi-task");
        assert_eq!(result.task_results.len(), 2);
        let interference = coordinator.interference_measure();
        assert!(
            (0.0..=1.0).contains(&interference),
            "interference out of range: {interference}"
        );
        // Opposed tasks must register as conflicting (cosine below zero).
        assert!(interference > 0.5, "opposed tasks should conflict");
        assert!(coordinator
            .learn_simultaneously(&[], &mut parameters)
            .is_err());
    }

    /// The framework's meta-parameter initialiser sizes itself from the tasks
    /// and is not all zeros.
    #[test]
    fn test_initialize_meta_parameters_is_sized_and_nonzero() {
        let pool = task_pool(3);
        let framework =
            MetaLearningFramework::<f64>::new(MetaLearningConfig::default()).expect("framework");
        let initial = framework
            .initialize_meta_parameters(&pool)
            .expect("initialise");
        assert_eq!(initial[linear_model::WEIGHTS_KEY].len(), 4);
        assert!(initial[linear_model::WEIGHTS_KEY]
            .iter()
            .any(|w| w.abs() > 0.0));
        assert!(framework.initialize_meta_parameters(&[]).is_err());
    }
}
