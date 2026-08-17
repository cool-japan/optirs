//! Reptile meta-learning algorithm
//!
//! Implements the Reptile algorithm by Nichol et al., which performs meta-learning
//! by interpolating between the current meta-parameters and task-adapted parameters.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::MetaLearner;
use super::linear_model;
use super::metrics::{mean_of, BatchObservations};
use super::types::{
    AdaptationStep, MetaLearningAlgorithm, MetaTask, MetaTrainingResult, QueryEvaluationMetrics,
    QueryEvaluationResult, TaskAdaptationMetrics, TaskAdaptationResult, TaskType,
};

/// Result type for inner loop adaptation: adapted parameters and adaptation trajectory
type InnerLoopResult<T> = (HashMap<String, Array1<T>>, Vec<AdaptationStep<T>>);

/// Reptile meta-learner
///
/// Reptile works by:
/// 1. For each task, running multiple steps of SGD from the meta-parameters
/// 2. Computing the difference between adapted and meta-parameters
/// 3. Updating meta-parameters toward the average adapted parameters
pub struct ReptileLearner<T: Float + Debug + Send + Sync + 'static> {
    /// Outer learning rate / interpolation factor
    epsilon: T,
    /// Inner loop learning rate
    inner_lr: T,
    /// Number of inner loop SGD steps
    inner_steps: usize,
    /// Step count for tracking
    step_count: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> ReptileLearner<T> {
    /// Create a new ReptileLearner with the given outer and inner learning rates
    pub fn new(epsilon: T, inner_lr: T) -> Self {
        Self {
            epsilon,
            inner_lr,
            inner_steps: 5,
            step_count: 0,
        }
    }

    /// Set the number of inner loop SGD steps (builder pattern)
    pub fn with_inner_steps(mut self, n: usize) -> Self {
        self.inner_steps = n;
        self
    }

    /// MSE loss of the shared linear task model on a dataset.
    ///
    /// See [`super::linear_model`] for the model contract shared with the other
    /// meta-learners.
    fn compute_loss(
        &self,
        features: &[Array1<T>],
        targets: &[T],
        parameters: &HashMap<String, Array1<T>>,
    ) -> Result<T> {
        linear_model::mse_loss(features, targets, parameters)
    }

    /// Single prediction from the shared linear task model.
    fn predict_single(
        &self,
        features: &Array1<T>,
        parameters: &HashMap<String, Array1<T>>,
    ) -> Result<T> {
        linear_model::predict(features, parameters)
    }

    /// Analytic gradients of the MSE loss.
    ///
    /// This used to be a central finite difference that re-evaluated the whole
    /// dataset twice per parameter (O(P * N) loss evaluations and O(P) clones of
    /// the parameter map). The closed-form gradient is exact and O(N).
    fn compute_gradients(
        &self,
        parameters: &HashMap<String, Array1<T>>,
        features: &[Array1<T>],
        targets: &[T],
    ) -> Result<HashMap<String, Array1<T>>> {
        linear_model::mse_gradients(features, targets, parameters)
    }

    /// Run inner loop SGD on a task's support set
    fn run_inner_loop(
        &self,
        task: &MetaTask<T>,
        initial_params: &HashMap<String, Array1<T>>,
        num_steps: usize,
    ) -> Result<InnerLoopResult<T>> {
        let mut params = initial_params.clone();
        let mut trajectory = Vec::new();

        for step in 0..num_steps {
            let loss = self.compute_loss(
                &task.support_set.features,
                &task.support_set.targets,
                &params,
            )?;
            let gradients = self.compute_gradients(
                &params,
                &task.support_set.features,
                &task.support_set.targets,
            )?;

            let gradient_norm = linear_model::map_norm(&gradients);
            let parameter_change_norm =
                linear_model::descend(&mut params, &gradients, self.inner_lr)?;

            trajectory.push(AdaptationStep {
                step,
                loss,
                gradient_norm,
                parameter_change_norm,
                learning_rate: self.inner_lr,
            });
        }

        Ok((params, trajectory))
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
    > MetaLearner<T> for ReptileLearner<T>
{
    fn meta_train_step(
        &mut self,
        task_batch: &[MetaTask<T>],
        meta_parameters: &mut HashMap<String, Array1<T>>,
    ) -> Result<MetaTrainingResult<T>> {
        if task_batch.is_empty() {
            return Err(OptimError::InsufficientData("Empty task batch".to_string()));
        }

        let batch_size = T::from(task_batch.len()).ok_or_else(|| {
            OptimError::ComputationError("Failed to convert batch size".to_string())
        })?;

        let mut total_meta_loss = T::zero();
        let mut task_losses = Vec::new();
        let mut observations = BatchObservations::<T>::new();
        let mut meta_gradients: HashMap<String, Array1<T>> = HashMap::new();

        // Initialize accumulated differences to zero
        let mut accumulated_diff: HashMap<String, Array1<T>> = HashMap::new();
        for (name, param) in meta_parameters.iter() {
            accumulated_diff.insert(name.clone(), Array1::zeros(param.len()));
        }

        for task in task_batch {
            let pre_loss = self.compute_loss(
                &task.query_set.features,
                &task.query_set.targets,
                meta_parameters,
            )?;

            // Run inner loop SGD from meta-parameters
            let (adapted_params, trajectory) =
                self.run_inner_loop(task, meta_parameters, self.inner_steps)?;

            // Compute task loss on query set with adapted parameters
            let task_loss = self.compute_loss(
                &task.query_set.features,
                &task.query_set.targets,
                &adapted_params,
            )?;
            task_losses.push(task_loss);
            total_meta_loss = total_meta_loss + task_loss;

            // Per-task "gradient" in descent convention: the direction Reptile
            // moves *away* from, i.e. -(adapted - meta).
            let mut task_gradient: HashMap<String, Array1<T>> = HashMap::new();

            // Accumulate difference: adapted_params - meta_parameters
            for (name, adapted_param) in &adapted_params {
                if let Some(meta_param) = meta_parameters.get(name) {
                    let diff = adapted_param - meta_param;
                    if let Some(acc) = accumulated_diff.get_mut(name) {
                        *acc = acc.clone() + &diff;
                    }
                    task_gradient.insert(name.clone(), diff.mapv(|v| -v));
                }
            }

            observations.pre_losses.push(pre_loss);
            observations.post_losses.push(task_loss);
            observations.convergence_steps.push(trajectory.len());
            observations.parameter_changes.push(
                trajectory
                    .iter()
                    .map(|s| s.parameter_change_norm)
                    .fold(T::zero(), |a, b| a + b),
            );
            observations
                .gradient_norms
                .push(linear_model::map_norm(&task_gradient));
            observations.gradients.push(task_gradient);
        }

        // Update meta-parameters: meta_params += epsilon * avg_difference.
        //
        // Update contract: the learner owns this update. `meta_gradients` is
        // reported in *descent* convention (the vector `g` for which the update
        // is `theta <- theta - epsilon * g`), so callers must not apply it again.
        for (name, param) in meta_parameters.iter_mut() {
            if let Some(acc) = accumulated_diff.get(name) {
                let avg_diff = acc / batch_size;
                for i in 0..param.len() {
                    param[i] = param[i] + self.epsilon * avg_diff[i];
                }
                meta_gradients.insert(name.clone(), avg_diff.mapv(|v| -v));
            }
        }

        let meta_loss = total_meta_loss / batch_size;
        self.step_count += 1;

        Ok(MetaTrainingResult {
            meta_loss,
            task_losses: task_losses.clone(),
            meta_gradients,
            metrics: observations.training_metrics(),
            adaptation_stats: observations.adaptation_statistics(),
        })
    }

    fn adapt_to_task(
        &mut self,
        task: &MetaTask<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
        adaptation_steps: usize,
    ) -> Result<TaskAdaptationResult<T>> {
        let (adapted_parameters, adaptation_trajectory) =
            self.run_inner_loop(task, meta_parameters, adaptation_steps)?;

        let initial_loss = adaptation_trajectory
            .first()
            .map(|s| s.loss)
            .unwrap_or_else(T::zero);
        let final_loss = self.compute_loss(
            &task.support_set.features,
            &task.support_set.targets,
            &adapted_parameters,
        )?;

        let steps_t = T::from(adaptation_steps.max(1)).ok_or_else(|| {
            OptimError::ComputationError("Failed to convert adaptation steps".to_string())
        })?;
        let improvement = initial_loss - final_loss;
        let travel = adaptation_trajectory
            .iter()
            .map(|s| s.parameter_change_norm)
            .fold(T::zero(), |a, b| a + b);
        let mut non_increasing = 0usize;
        for pair in adaptation_trajectory.windows(2) {
            if pair[1].loss <= pair[0].loss {
                non_increasing += 1;
            }
        }

        Ok(TaskAdaptationResult {
            adapted_parameters,
            metrics: TaskAdaptationMetrics {
                convergence_speed: improvement / steps_t,
                final_performance: if initial_loss > T::zero() {
                    (improvement / initial_loss).max(T::zero()).min(T::one())
                } else {
                    T::zero()
                },
                efficiency: if travel > T::zero() {
                    improvement / travel
                } else {
                    T::zero()
                },
                robustness: if adaptation_trajectory.len() > 1 {
                    let denom = T::from(adaptation_trajectory.len() - 1).unwrap_or_else(T::one);
                    T::from(non_increasing).unwrap_or_else(T::zero) / denom
                } else {
                    T::one()
                },
            },
            adaptation_trajectory,
            final_loss,
        })
    }

    fn evaluate_query_set(
        &self,
        task: &MetaTask<T>,
        adapted_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<QueryEvaluationResult<T>> {
        if task.query_set.features.is_empty() {
            return Err(OptimError::InsufficientData(format!(
                "task '{}' has an empty query set",
                task.id
            )));
        }

        let mut predictions = Vec::with_capacity(task.query_set.features.len());
        for features in task.query_set.features.iter() {
            predictions.push(self.predict_single(features, adapted_parameters)?);
        }

        let query_loss = self.compute_loss(
            &task.query_set.features,
            &task.query_set.targets,
            adapted_parameters,
        )?;

        // Confidence is the squashed per-sample residual: 1 for an exact hit.
        let confidence_scores: Vec<T> = predictions
            .iter()
            .zip(task.query_set.targets.iter())
            .map(|(p, y)| T::one() / (T::one() + (*p - *y).abs()))
            .collect();

        let classification_accuracy = match task.task_type {
            TaskType::Classification => {
                linear_model::label_accuracy(&predictions, &task.query_set.targets)
            }
            _ => None,
        };
        let accuracy = classification_accuracy
            .or_else(|| {
                linear_model::r_squared(&predictions, &task.query_set.targets)
                    .map(|r| r.max(T::zero()).min(T::one()))
            })
            .unwrap_or_else(T::zero);
        let uncertainty_quality = mean_of(&confidence_scores).unwrap_or_else(T::zero);

        // Full-curve AUC-ROC ranked on the retained predictions; `None` unless
        // the targets are genuinely binary with both classes represented.
        let auc = match task.task_type {
            TaskType::Classification => {
                linear_model::roc_auc(&predictions, &task.query_set.targets)
            }
            _ => None,
        };

        Ok(QueryEvaluationResult {
            query_loss,
            accuracy,
            predictions,
            confidence_scores,
            metrics: QueryEvaluationMetrics {
                mse: Some(query_loss),
                classification_accuracy,
                auc,
                uncertainty_quality,
            },
        })
    }

    fn get_algorithm(&self) -> MetaLearningAlgorithm {
        MetaLearningAlgorithm::Reptile
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{DatasetMetadata, TaskDataset, TaskMetadata, TaskType};
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_test_task() -> MetaTask<f64> {
        let support_features = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
            Array1::from_vec(vec![7.0, 8.0, 9.0]),
        ];
        let support_targets = vec![2.0, 5.0, 8.0];
        let query_features = vec![
            Array1::from_vec(vec![2.0, 3.0, 4.0]),
            Array1::from_vec(vec![5.0, 6.0, 7.0]),
        ];
        let query_targets = vec![3.0, 6.0];

        MetaTask {
            id: "test_task".to_string(),
            support_set: TaskDataset {
                features: support_features,
                targets: support_targets,
                weights: vec![1.0, 1.0, 1.0],
                metadata: DatasetMetadata::default(),
            },
            query_set: TaskDataset {
                features: query_features,
                targets: query_targets,
                weights: vec![1.0, 1.0],
                metadata: DatasetMetadata::default(),
            },
            metadata: TaskMetadata::default(),
            difficulty: 1.0,
            domain: "test".to_string(),
            task_type: TaskType::Regression,
        }
    }

    fn make_test_params() -> HashMap<String, Array1<f64>> {
        let mut params = HashMap::new();
        params.insert("weights".to_string(), Array1::from_vec(vec![0.5, 0.5, 0.5]));
        params
    }

    #[test]
    fn test_reptile_new() {
        let learner = ReptileLearner::new(0.1f64, 0.01);
        assert_eq!(learner.inner_steps, 5);
        assert_eq!(learner.step_count, 0);
    }

    #[test]
    fn test_reptile_with_inner_steps() {
        let learner = ReptileLearner::new(0.1f64, 0.01).with_inner_steps(10);
        assert_eq!(learner.inner_steps, 10);
    }

    #[test]
    fn test_reptile_adapt_to_task() {
        let mut learner = ReptileLearner::new(0.1f64, 0.01).with_inner_steps(3);
        let task = make_test_task();
        let params = make_test_params();

        let result = learner
            .adapt_to_task(&task, &params, 3)
            .expect("adapt_to_task should succeed");
        assert_eq!(result.adaptation_trajectory.len(), 3);
        assert!(result.adapted_parameters.contains_key("weights"));
    }

    #[test]
    fn test_reptile_meta_train_step() {
        let mut learner = ReptileLearner::new(0.1f64, 0.01).with_inner_steps(3);
        let task = make_test_task();
        let mut params = make_test_params();

        let original_weights = params.get("weights").expect("weights should exist").clone();
        let result = learner
            .meta_train_step(&[task], &mut params)
            .expect("meta_train_step should succeed");

        assert_eq!(result.task_losses.len(), 1);
        // Meta-parameters should have been updated
        let updated_weights = params.get("weights").expect("weights should exist");
        let changed = original_weights
            .iter()
            .zip(updated_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed, "Meta-parameters should change after training step");
    }

    #[test]
    fn test_reptile_evaluate_query_set() {
        let learner = ReptileLearner::new(0.1f64, 0.01);
        let task = make_test_task();
        let params = make_test_params();

        let result = learner
            .evaluate_query_set(&task, &params)
            .expect("evaluate_query_set should succeed");
        assert_eq!(result.predictions.len(), 2);
        assert_eq!(result.confidence_scores.len(), 2);
        assert!(result.query_loss >= 0.0);
    }

    #[test]
    fn test_reptile_get_algorithm() {
        let learner = ReptileLearner::new(0.1f64, 0.01);
        assert!(matches!(
            learner.get_algorithm(),
            MetaLearningAlgorithm::Reptile
        ));
    }

    #[test]
    fn test_reptile_empty_batch_error() {
        let mut learner = ReptileLearner::new(0.1f64, 0.01);
        let mut params = make_test_params();
        let result = learner.meta_train_step(&[], &mut params);
        assert!(result.is_err());
    }

    #[test]
    fn test_reptile_multiple_tasks() {
        let mut learner = ReptileLearner::new(0.1f64, 0.01).with_inner_steps(3);
        let task1 = make_test_task();
        let task2 = make_test_task();
        let mut params = make_test_params();

        let result = learner
            .meta_train_step(&[task1, task2], &mut params)
            .expect("meta_train_step with multiple tasks should succeed");
        assert_eq!(result.task_losses.len(), 2);
    }
}
