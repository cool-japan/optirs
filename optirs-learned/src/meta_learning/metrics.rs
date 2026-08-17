//! Real meta-training metrics computed from observed quantities.
//!
//! Every meta-learner in this module reports [`MetaTrainingMetrics`] and
//! [`AdaptationStatistics`]. Those used to be filled with fixed constants
//! (`0.85`, `0.9`, ...). This module derives them from what the meta-training
//! step actually observed, and returns neutral values only where a quantity is
//! genuinely unobservable from a single batch (documented per field).

use std::collections::HashMap;
use std::fmt::Debug;

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use super::linear_model;
use super::types::{AdaptationStatistics, MetaTrainingMetrics, StabilityMetrics};

/// Mean of a slice, or `None` when empty.
pub fn mean_of<T: Float>(values: &[T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    let n = T::from(values.len())?;
    Some(values.iter().copied().fold(T::zero(), |a, b| a + b) / n)
}

/// Sample standard deviation, or `None` when fewer than two values.
pub fn std_of<T: Float>(values: &[T]) -> Option<T> {
    if values.len() < 2 {
        return None;
    }
    let mean = mean_of(values)?;
    let n = T::from(values.len() - 1)?;
    let var = values
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .fold(T::zero(), |a, b| a + b)
        / n;
    Some(var.sqrt())
}

/// A stability score in `(0, 1]`: 1 when a quantity is perfectly consistent
/// across the batch, decaying as its spread grows.
fn stability_from_spread<T: Float>(values: &[T]) -> T {
    match std_of(values) {
        Some(sd) => T::one() / (T::one() + sd),
        None => T::one(),
    }
}

/// Everything a meta-training step observed about one batch of tasks.
#[derive(Debug, Clone)]
pub struct BatchObservations<T: Float + Debug + Send + Sync + 'static> {
    /// Query loss of each task *before* adaptation
    pub pre_losses: Vec<T>,
    /// Query loss of each task *after* adaptation
    pub post_losses: Vec<T>,
    /// Number of inner steps actually run per task
    pub convergence_steps: Vec<usize>,
    /// L2 norm of the total parameter movement during each task's inner loop
    pub parameter_changes: Vec<T>,
    /// L2 norm of each task's meta-gradient
    pub gradient_norms: Vec<T>,
    /// Per-task meta-gradients (used for the alignment metric)
    pub gradients: Vec<HashMap<String, Array1<T>>>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for BatchObservations<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> BatchObservations<T> {
    /// An empty observation set.
    pub fn new() -> Self {
        Self {
            pre_losses: Vec::new(),
            post_losses: Vec::new(),
            convergence_steps: Vec::new(),
            parameter_changes: Vec::new(),
            gradient_norms: Vec::new(),
            gradients: Vec::new(),
        }
    }

    /// Mean loss improvement per inner step.
    fn adaptation_speed(&self) -> T {
        let mut speeds = Vec::with_capacity(self.post_losses.len());
        for (i, post) in self.post_losses.iter().enumerate() {
            let steps = self.convergence_steps.get(i).copied().unwrap_or(0);
            if steps == 0 {
                continue;
            }
            let Some(steps_t) = T::from(steps) else {
                continue;
            };
            if let Some(pre) = self.pre_losses.get(i) {
                speeds.push((*pre - *post) / steps_t);
            }
        }
        mean_of(&speeds).unwrap_or_else(T::zero)
    }

    /// Mean relative query-loss improvement, clamped to `[0, 1]`.
    fn generalization(&self) -> T {
        let mut improvements = Vec::with_capacity(self.post_losses.len());
        for (pre, post) in self.pre_losses.iter().zip(self.post_losses.iter()) {
            if *pre > T::zero() {
                let ratio = (*pre - *post) / *pre;
                improvements.push(ratio.max(T::zero()).min(T::one()));
            }
        }
        mean_of(&improvements).unwrap_or_else(T::zero)
    }

    /// Dispersion of the per-task query losses (coefficient of variation),
    /// clamped to `[0, 1]`.
    fn diversity(&self) -> T {
        match (std_of(&self.post_losses), mean_of(&self.post_losses)) {
            (Some(sd), Some(mean)) if mean > T::zero() => (sd / mean).min(T::one()),
            _ => T::zero(),
        }
    }

    /// Mean pairwise cosine similarity of the per-task meta-gradients.
    ///
    /// With fewer than two comparable gradients there is no alignment to
    /// measure and the neutral value 1 (perfectly aligned with itself) is used.
    fn alignment(&self) -> T {
        let mut cosines = Vec::new();
        for i in 0..self.gradients.len() {
            for j in (i + 1)..self.gradients.len() {
                if let Some(c) = linear_model::map_cosine(&self.gradients[i], &self.gradients[j]) {
                    cosines.push(c);
                }
            }
        }
        mean_of(&cosines).unwrap_or_else(T::one)
    }

    /// Total loss improvement per unit of parameter travel.
    fn efficiency(&self) -> T {
        let travel = self
            .parameter_changes
            .iter()
            .copied()
            .fold(T::zero(), |a, b| a + b);
        if travel <= T::zero() {
            return T::zero();
        }
        let improvement = self
            .pre_losses
            .iter()
            .zip(self.post_losses.iter())
            .map(|(pre, post)| *pre - *post)
            .fold(T::zero(), |a, b| a + b);
        improvement / travel
    }

    /// Build the training metrics for this batch.
    pub fn training_metrics(&self) -> MetaTrainingMetrics<T> {
        MetaTrainingMetrics {
            avg_adaptation_speed: self.adaptation_speed(),
            generalization_performance: self.generalization(),
            task_diversity: self.diversity(),
            gradient_alignment: self.alignment(),
        }
    }

    /// Build the adaptation statistics for this batch.
    ///
    /// `forgetting_measure` is reported as zero: catastrophic forgetting is a
    /// property of a task *sequence* and cannot be observed from one batch.
    /// [`super::types::ContinualLearningSystem`] measures it for real.
    pub fn adaptation_statistics(&self) -> AdaptationStatistics<T> {
        AdaptationStatistics {
            convergence_steps: self.convergence_steps.clone(),
            final_losses: self.post_losses.clone(),
            adaptation_efficiency: self.efficiency(),
            stability_metrics: StabilityMetrics {
                parameter_stability: stability_from_spread(&self.parameter_changes),
                performance_stability: stability_from_spread(&self.post_losses),
                gradient_stability: stability_from_spread(&self.gradient_norms),
                forgetting_measure: T::zero(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_reflect_observations() {
        let mut obs = BatchObservations::<f64>::new();
        obs.pre_losses = vec![1.0, 2.0];
        obs.post_losses = vec![0.5, 1.0];
        obs.convergence_steps = vec![5, 5];
        obs.parameter_changes = vec![0.25, 0.25];
        obs.gradient_norms = vec![0.1, 0.1];

        let m = obs.training_metrics();
        // Improvement 0.5 and 1.0 over 5 steps -> mean speed 0.15.
        approx::assert_abs_diff_eq!(m.avg_adaptation_speed, 0.15, epsilon = 1e-12);
        // Both tasks halved their loss -> generalisation 0.5.
        approx::assert_abs_diff_eq!(m.generalization_performance, 0.5, epsilon = 1e-12);

        let s = obs.adaptation_statistics();
        // Total improvement 1.5 over total travel 0.5.
        approx::assert_abs_diff_eq!(s.adaptation_efficiency, 3.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(
            s.stability_metrics.parameter_stability,
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_empty_observations_are_neutral_not_fabricated() {
        let obs = BatchObservations::<f64>::new();
        let m = obs.training_metrics();
        assert_eq!(m.avg_adaptation_speed, 0.0);
        assert_eq!(m.generalization_performance, 0.0);
        assert_eq!(m.task_diversity, 0.0);
    }
}
