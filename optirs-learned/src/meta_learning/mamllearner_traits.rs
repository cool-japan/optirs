//! # MAMLLearner - Trait Implementations
//!
//! Model-Agnostic Meta-Learning over the shared linear task model defined in
//! [`super::linear_model`].
//!
//! * **Inner loop** — plain gradient descent on the task's *support-set* loss,
//!   using analytic gradients.
//! * **Outer loop** — FOMAML by default: the meta-gradient is the query-set
//!   gradient evaluated at the *adapted* parameters. When
//!   `MAMLConfig::second_order` is set, the exact MAML meta-gradient
//!   `prod_j (I - alpha H_support) grad L_query(theta')` is used instead. The model
//!   is linear in its parameters, so its MSE Hessian is constant and the
//!   Hessian-vector product is closed-form — the second-order path is exact,
//!   not an approximation.
//! * **Update contract** — the learner owns the meta-update: `meta_train_step`
//!   applies `theta <- theta - outer_lr * meta_gradient` to the map it is given
//!   and reports the gradient it used. Callers must not apply it a second time.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(unused_imports)]
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Dimension};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::MetaLearner;
use super::linear_model;
use super::metrics::{mean_of, BatchObservations};
use super::types::{
    AdaptationStep, MAMLLearner, MetaLearningAlgorithm, MetaTask, MetaTrainingResult,
    QueryEvaluationMetrics, QueryEvaluationResult, TaskAdaptationMetrics, TaskAdaptationResult,
    TaskType, MAML_HISTORY_CAPACITY,
};

impl<
        T: Float
            + Debug
            + 'static
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
        D: Dimension,
    > MAMLLearner<T, D>
{
    /// Meta-gradient for a single task.
    ///
    /// Returns `(meta_gradient, adaptation_result, query_result)`.
    #[allow(clippy::type_complexity)]
    fn task_meta_gradient(
        &mut self,
        task: &MetaTask<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<(
        HashMap<String, Array1<T>>,
        TaskAdaptationResult<T>,
        QueryEvaluationResult<T>,
    )> {
        let inner_steps = self.config.inner_steps;
        let adaptation = self.adapt_to_task(task, meta_parameters, inner_steps)?;
        let query = self.evaluate_query_set(task, &adaptation.adapted_parameters)?;

        // FOMAML: the meta-gradient is the query gradient at the adapted point.
        let mut meta_grad = self.compute_query_gradients(task, &adaptation.adapted_parameters)?;

        if self.config.second_order {
            // Exact MAML: back-propagate through the `inner_steps` gradient
            // descent updates. For theta' = theta - alpha * grad L_s(theta) the
            // Jacobian is (I - alpha * H_s), and H_s is constant here.
            let alpha = self.config.inner_lr;
            for _ in 0..inner_steps {
                let hvp = linear_model::mse_hessian_vector_product(
                    &task.support_set.features,
                    &adaptation.adapted_parameters,
                    &meta_grad,
                )?;
                for (name, g) in meta_grad.iter_mut() {
                    if let Some(h) = hvp.get(name) {
                        let n = g.len().min(h.len());
                        for i in 0..n {
                            g[i] = g[i] - alpha * h[i];
                        }
                    }
                }
            }
        }

        if let Some(clip) = self.config.gradient_clip {
            if let Some(max_norm) = scirs2_core::numeric::NumCast::from(clip) {
                linear_model::clip_gradients(&mut meta_grad, max_norm);
            }
        }

        Ok((meta_grad, adaptation, query))
    }
}

impl<
        T: Float
            + Debug
            + 'static
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
        D: Dimension,
    > MetaLearner<T> for MAMLLearner<T, D>
{
    /// One meta-training step over a batch of tasks.
    ///
    /// The meta-update is applied here (see the module docs): `meta_parameters`
    /// comes back already advanced by `-outer_lr * meta_gradient`.
    fn meta_train_step(
        &mut self,
        task_batch: &[MetaTask<T>],
        meta_parameters: &mut HashMap<String, Array1<T>>,
    ) -> Result<MetaTrainingResult<T>> {
        if task_batch.is_empty() {
            return Err(OptimError::InsufficientData(
                "meta_train_step requires at least one task".to_string(),
            ));
        }
        if meta_parameters.is_empty() {
            return Err(OptimError::InvalidConfig(
                "meta_train_step requires non-empty meta-parameters".to_string(),
            ));
        }

        let mut total_meta_loss = T::zero();
        let mut task_losses: Vec<T> = Vec::with_capacity(task_batch.len());
        let mut pre_adaptation_losses: Vec<T> = Vec::with_capacity(task_batch.len());
        let mut convergence_steps: Vec<usize> = Vec::with_capacity(task_batch.len());
        let mut per_task_gradients: Vec<HashMap<String, Array1<T>>> =
            Vec::with_capacity(task_batch.len());
        let mut parameter_changes: Vec<T> = Vec::with_capacity(task_batch.len());
        let mut gradient_norms: Vec<T> = Vec::with_capacity(task_batch.len());

        let mut meta_gradients: HashMap<String, Array1<T>> = meta_parameters
            .iter()
            .map(|(name, p)| (name.clone(), Array1::zeros(p.len())))
            .collect();

        for task in task_batch {
            let before = linear_model::mse_loss(
                &task.query_set.features,
                &task.query_set.targets,
                meta_parameters,
            )?;
            pre_adaptation_losses.push(before);

            let (task_grad, adaptation, query) = self.task_meta_gradient(task, meta_parameters)?;

            task_losses.push(query.query_loss);
            total_meta_loss = total_meta_loss + query.query_loss;
            convergence_steps.push(adaptation.adaptation_trajectory.len());
            parameter_changes.push(
                adaptation
                    .adaptation_trajectory
                    .last()
                    .map(|s| s.parameter_change_norm)
                    .unwrap_or_else(T::zero),
            );
            gradient_norms.push(linear_model::map_norm(&task_grad));

            for (name, grad) in &task_grad {
                if let Some(acc) = meta_gradients.get_mut(name) {
                    let n = acc.len().min(grad.len());
                    for i in 0..n {
                        acc[i] = acc[i] + grad[i];
                    }
                }
            }
            per_task_gradients.push(task_grad);

            if self.adaptation_history.len() == MAML_HISTORY_CAPACITY {
                self.adaptation_history.pop_front();
            }
            self.adaptation_history.push_back(adaptation);
        }

        let batch_size = T::from(task_batch.len()).ok_or_else(|| {
            OptimError::ComputationError("failed to convert task batch size".to_string())
        })?;
        let meta_loss = total_meta_loss / batch_size;
        for gradient in meta_gradients.values_mut() {
            for g in gradient.iter_mut() {
                *g = *g / batch_size;
            }
        }

        // The learner owns the update.
        linear_model::descend(meta_parameters, &meta_gradients, self.config.outer_lr)?;
        self.meta_steps += 1;

        // Real metrics, derived from what this batch actually observed.
        let observations = BatchObservations {
            pre_losses: pre_adaptation_losses,
            post_losses: task_losses.clone(),
            convergence_steps,
            parameter_changes,
            gradient_norms,
            gradients: per_task_gradients,
        };

        Ok(MetaTrainingResult {
            meta_loss,
            task_losses,
            meta_gradients,
            metrics: observations.training_metrics(),
            adaptation_stats: observations.adaptation_statistics(),
        })
    }

    /// Inner loop: descend the support-set loss from the meta-parameters.
    fn adapt_to_task(
        &mut self,
        task: &MetaTask<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
        adaptation_steps: usize,
    ) -> Result<TaskAdaptationResult<T>> {
        if meta_parameters.is_empty() {
            return Err(OptimError::InvalidConfig(
                "adaptation requires non-empty meta-parameters".to_string(),
            ));
        }
        if task.support_set.features.is_empty() {
            return Err(OptimError::InsufficientData(format!(
                "task '{}' has an empty support set",
                task.id
            )));
        }

        let mut adapted_parameters = meta_parameters.clone();
        let mut adaptation_trajectory = Vec::with_capacity(adaptation_steps);
        let learning_rate = self.config.inner_lr;
        let initial_loss = self.compute_support_loss(task, &adapted_parameters)?;

        for step in 0..adaptation_steps {
            let loss = self.compute_support_loss(task, &adapted_parameters)?;
            let gradients = self.compute_support_gradients(task, &adapted_parameters)?;
            let gradient_norm = linear_model::map_norm(&gradients);
            let parameter_change_norm =
                linear_model::descend(&mut adapted_parameters, &gradients, learning_rate)?;

            adaptation_trajectory.push(AdaptationStep {
                step,
                loss,
                gradient_norm,
                parameter_change_norm,
                learning_rate,
            });
        }

        let final_loss = self.compute_support_loss(task, &adapted_parameters)?;
        let steps_t = T::from(adaptation_steps.max(1)).ok_or_else(|| {
            OptimError::ComputationError("failed to convert adaptation steps".to_string())
        })?;
        let improvement = initial_loss - final_loss;
        let convergence_speed = improvement / steps_t;
        let final_performance = if initial_loss > T::zero() {
            (improvement / initial_loss).max(T::zero()).min(T::one())
        } else {
            T::zero()
        };
        let travel = adaptation_trajectory
            .iter()
            .map(|s| s.parameter_change_norm)
            .fold(T::zero(), |a, b| a + b);
        let efficiency = if travel > T::zero() {
            improvement / travel
        } else {
            T::zero()
        };
        // Robustness: how monotone the descent was (fraction of steps that did
        // not increase the loss).
        let mut non_increasing = 0usize;
        for pair in adaptation_trajectory.windows(2) {
            if pair[1].loss <= pair[0].loss {
                non_increasing += 1;
            }
        }
        let robustness = if adaptation_trajectory.len() > 1 {
            let denom = T::from(adaptation_trajectory.len() - 1).unwrap_or_else(T::one);
            T::from(non_increasing).unwrap_or_else(T::zero) / denom
        } else {
            T::one()
        };

        Ok(TaskAdaptationResult {
            adapted_parameters,
            adaptation_trajectory,
            final_loss,
            metrics: TaskAdaptationMetrics {
                convergence_speed,
                final_performance,
                efficiency,
                robustness,
            },
        })
    }

    /// Evaluate the query set **at the adapted parameters**.
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

        let predictions = linear_model::predict_all(&task.query_set.features, adapted_parameters)?;
        let query_loss = linear_model::mse_loss(
            &task.query_set.features,
            &task.query_set.targets,
            adapted_parameters,
        )?;

        // Confidence: squashed per-sample residual, 1 for an exact prediction.
        let confidence_scores: Vec<T> = predictions
            .iter()
            .zip(task.query_set.targets.iter())
            .map(|(p, y)| {
                let err = (*p - *y).abs();
                T::one() / (T::one() + err)
            })
            .collect();

        let classification_accuracy = match task.task_type {
            TaskType::Classification => {
                linear_model::label_accuracy(&predictions, &task.query_set.targets)
            }
            _ => None,
        };
        let regression_r2 = linear_model::r_squared(&predictions, &task.query_set.targets);

        // `accuracy` is the task-appropriate goodness score: label accuracy for
        // classification, R^2 clamped to [0, 1] for regression, 0 when neither
        // is defined.
        let accuracy = classification_accuracy
            .or_else(|| regression_r2.map(|r| r.max(T::zero()).min(T::one())))
            .unwrap_or_else(T::zero);

        // Uncertainty quality: mean confidence, i.e. how tight the residuals are.
        let uncertainty_quality = mean_of(&confidence_scores).unwrap_or_else(T::zero);

        Ok(QueryEvaluationResult {
            query_loss,
            accuracy,
            predictions,
            confidence_scores,
            metrics: QueryEvaluationMetrics {
                mse: Some(query_loss),
                classification_accuracy,
                // AUC needs ranked binary labels and score thresholds, which
                // this scalar-target model does not carry.
                auc: None,
                uncertainty_quality,
            },
        })
    }

    fn get_algorithm(&self) -> MetaLearningAlgorithm {
        if self.config.second_order {
            MetaLearningAlgorithm::MAML
        } else {
            MetaLearningAlgorithm::FOMAML
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_learning::types::{
        DatasetMetadata, MAMLConfig, MetaTask, TaskDataset, TaskMetadata,
    };
    use scirs2_core::ndarray::Ix1;

    /// Fixed polynomial basis. The model is linear in its parameters, so the
    /// non-linearity has to live in the features for a sine-regression task to
    /// be learnable at all.
    fn basis(x: f64) -> Array1<f64> {
        let u = x / 3.0;
        Array1::from_vec(vec![1.0, u, u * u, u * u * u, u * u * u * u])
    }

    /// One sine-regression task: `y = amplitude * sin(x + phase)`.
    fn sine_task(id: usize, amplitude: f64, phase: f64) -> MetaTask<f64> {
        let mut support_features = Vec::new();
        let mut support_targets = Vec::new();
        let mut query_features = Vec::new();
        let mut query_targets = Vec::new();
        for k in 0..12 {
            let x = -3.0 + 6.0 * (k as f64) / 11.0;
            let y = amplitude * (x + phase).sin();
            if k % 2 == 0 {
                support_features.push(basis(x));
                support_targets.push(y);
            } else {
                query_features.push(basis(x));
                query_targets.push(y);
            }
        }
        MetaTask {
            id: format!("sine_{id}"),
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
            difficulty: amplitude,
            domain: "sine".to_string(),
            task_type: crate::meta_learning::types::TaskType::Regression,
        }
    }

    fn sine_task_family(count: usize) -> Vec<MetaTask<f64>> {
        (0..count)
            .map(|i| {
                let amplitude = 0.5 + 1.5 * (i as f64) / (count as f64);
                let phase = std::f64::consts::PI * (i as f64) / (count as f64);
                sine_task(i, amplitude, phase)
            })
            .collect()
    }

    fn zero_params(dim: usize) -> HashMap<String, Array1<f64>> {
        let mut p = HashMap::new();
        p.insert(linear_model::WEIGHTS_KEY.to_string(), Array1::zeros(dim));
        p.insert(linear_model::BIAS_KEY.to_string(), Array1::zeros(1));
        p
    }

    fn maml(second_order: bool) -> MAMLLearner<f64, Ix1> {
        MAMLLearner::new(MAMLConfig {
            second_order,
            inner_lr: 0.4,
            outer_lr: 0.3,
            inner_steps: 5,
            allow_unused: true,
            gradient_clip: Some(10.0),
        })
        .expect("maml learner")
    }

    /// Mean query loss over `tasks` after running the inner loop from `params`.
    fn mean_adapted_query_loss(
        learner: &mut MAMLLearner<f64, Ix1>,
        tasks: &[MetaTask<f64>],
        params: &HashMap<String, Array1<f64>>,
    ) -> f64 {
        let mut total = 0.0;
        for task in tasks {
            let adapted = learner
                .adapt_to_task(task, params, 5)
                .expect("adapt")
                .adapted_parameters;
            total += learner
                .evaluate_query_set(task, &adapted)
                .expect("evaluate")
                .query_loss;
        }
        total / tasks.len() as f64
    }

    /// F8/F9: meta-training must genuinely improve post-adaptation performance.
    #[test]
    fn test_sine_regression_meta_learning_improves_adaptation() {
        let tasks = sine_task_family(16);
        let holdout: Vec<MetaTask<f64>> = sine_task_family(5)
            .into_iter()
            .map(|mut t| {
                t.id = format!("holdout_{}", t.id);
                t
            })
            .collect();

        let mut learner = maml(false);
        let mut params = zero_params(5);

        let before = mean_adapted_query_loss(&mut learner, &holdout, &params);

        for _ in 0..400 {
            learner
                .meta_train_step(&tasks, &mut params)
                .expect("meta train step");
        }

        let after = mean_adapted_query_loss(&mut learner, &holdout, &params);
        assert!(
            after < before,
            "meta-training must improve post-adaptation loss: before={before}, after={after}"
        );

        // At the meta-trained initialisation, adapting must beat not adapting.
        let mut pre_adaptation = 0.0;
        for task in &holdout {
            pre_adaptation +=
                linear_model::mse_loss(&task.query_set.features, &task.query_set.targets, &params)
                    .expect("loss");
        }
        pre_adaptation /= holdout.len() as f64;
        assert!(
            after < pre_adaptation,
            "adaptation must reduce query loss: pre={pre_adaptation}, post={after}"
        );
    }

    /// The inner loop must descend the *support* loss, not a weight-decay proxy.
    #[test]
    fn test_inner_loop_descends_the_support_loss() {
        let task = sine_task(0, 1.0, 0.0);
        let mut learner = maml(false);
        let params = zero_params(5);

        let before = linear_model::mse_loss(
            &task.support_set.features,
            &task.support_set.targets,
            &params,
        )
        .expect("loss");
        let result = learner.adapt_to_task(&task, &params, 5).expect("adapt");
        assert!(
            result.final_loss < before,
            "support loss must decrease: before={before}, after={}",
            result.final_loss
        );
        assert_eq!(result.adaptation_trajectory.len(), 5);
        // The adapted parameters must actually differ from the meta-parameters.
        let moved = result
            .adaptation_trajectory
            .iter()
            .any(|s| s.parameter_change_norm > 0.0);
        assert!(moved, "the inner loop never moved any parameter");
    }

    /// F9: the reported meta-gradient must be non-zero and must actually be the
    /// direction the meta-parameters moved along.
    #[test]
    fn test_meta_gradients_are_real_and_applied_once() {
        let tasks = sine_task_family(4);
        let mut learner = maml(false);
        let mut params = zero_params(5);
        let before = params[linear_model::WEIGHTS_KEY].clone();

        let result = learner
            .meta_train_step(&tasks, &mut params)
            .expect("meta train step");

        let grad = &result.meta_gradients[linear_model::WEIGHTS_KEY];
        assert!(
            linear_model::map_norm(&result.meta_gradients) > 0.0,
            "meta-gradients must not be zeros"
        );
        // theta_new = theta_old - outer_lr * grad, applied exactly once.
        let after = &params[linear_model::WEIGHTS_KEY];
        for i in 0..after.len() {
            approx::assert_abs_diff_eq!(after[i], before[i] - 0.3 * grad[i], epsilon = 1e-12);
        }
    }

    /// The second-order path must produce a different (and still descending)
    /// meta-gradient than FOMAML.
    #[test]
    fn test_second_order_differs_from_first_order() {
        let tasks = sine_task_family(4);

        let mut first = maml(false);
        let mut second = maml(true);
        let mut p1 = zero_params(5);
        let mut p2 = zero_params(5);

        let r1 = first.meta_train_step(&tasks, &mut p1).expect("fomaml");
        let r2 = second.meta_train_step(&tasks, &mut p2).expect("maml");

        let g1 = &r1.meta_gradients[linear_model::WEIGHTS_KEY];
        let g2 = &r2.meta_gradients[linear_model::WEIGHTS_KEY];
        let diff: f64 = g1
            .iter()
            .zip(g2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff > 1e-9,
            "the second-order meta-gradient must differ from the first-order one"
        );
        assert!(matches!(
            first.get_algorithm(),
            MetaLearningAlgorithm::FOMAML
        ));
        assert!(matches!(
            second.get_algorithm(),
            MetaLearningAlgorithm::MAML
        ));
    }

    /// Query evaluation must use the *adapted* parameters.
    #[test]
    fn test_query_evaluation_uses_adapted_parameters() {
        let task = sine_task(0, 1.5, 0.3);
        let mut learner = maml(false);
        let params = zero_params(5);

        let at_meta = learner
            .evaluate_query_set(&task, &params)
            .expect("meta eval");
        let adapted = learner
            .adapt_to_task(&task, &params, 5)
            .expect("adapt")
            .adapted_parameters;
        let at_adapted = learner
            .evaluate_query_set(&task, &adapted)
            .expect("adapted eval");

        assert!(
            at_adapted.query_loss < at_meta.query_loss,
            "adapted evaluation must differ from (and beat) the un-adapted one"
        );
        assert_eq!(at_adapted.predictions.len(), task.query_set.features.len());
        assert!(at_adapted.metrics.auc.is_none());
    }

    /// Empty support/query sets must be errors, not NaN losses.
    #[test]
    fn test_empty_task_is_an_error() {
        let mut empty = sine_task(0, 1.0, 0.0);
        empty.support_set.features.clear();
        empty.support_set.targets.clear();
        let mut learner = maml(false);
        let params = zero_params(5);
        assert!(learner.adapt_to_task(&empty, &params, 3).is_err());
        assert!(learner.meta_train_step(&[], &mut params.clone()).is_err());
    }
}
