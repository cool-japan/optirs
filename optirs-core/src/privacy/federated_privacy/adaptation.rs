//! Meta-learning and task-shift detection for federated rounds.
//!
//! # The defects this replaces
//!
//! ```text
//! pub fn compute_client_meta_gradients(...) -> Result<Array1<T>> {
//!     // Placeholder implementation
//!     Ok(Array1::default(0))
//! }
//!
//! pub fn detect_task_change(&mut self, updates: &[Array1<T>]) -> Result<bool> {
//!     // Placeholder implementation
//!     let _ = updates;
//!     Ok(false)
//! }
//! ```
//!
//! The first returned a length-0 array for every input -- it had no storage to
//! return, because `FederatedMetaLearner::new` discarded its `parameter_size`
//! argument -- and the second reported "no task change" unconditionally, so the
//! continual-learning machinery behind it could never fire.
//!
//! # What is implemented
//!
//! * [`FederatedMetaLearner::compute_client_meta_gradients`]: a first-order MAML
//!   / Reptile meta-gradient (Finn, Abbeel & Levine 2017; Nichol, Achiam &
//!   Schulman 2018). Each client's support gradient takes one inner step from
//!   the shared meta-parameters, the resulting adaptation is recorded, and the
//!   meta-gradient is the mean of the clients' query gradients. Every length is
//!   checked, so a mis-sized learner reports the mismatch instead of returning
//!   an empty array.
//! * [`TaskDetector::detect_task_change`]: gradient-based change detection. The
//!   round's mean update is compared with the running mean of the buffered
//!   history by relative L2 shift and cosine dissimilarity; a change is flagged
//!   when the larger of the two exceeds the configured threshold, and the
//!   [`ChangePoint`] records the magnitude that triggered it. Detection methods
//!   other than `GradientBased` return [`OptimError::UnsupportedOperation`]
//!   rather than silently behaving like the gradient-based one.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::components::{ChangePoint, FederatedMetaLearner, TaskDetector, TaskDistribution};
use super::config::TaskDetectionMethod;

/// Default inner-loop step size for the first-order meta-gradient.
pub const DEFAULT_INNER_LEARNING_RATE: f64 = 0.01;

/// Maximum number of rounds retained in the task detector's buffer.
const MAX_GRADIENT_BUFFER: usize = 100;

/// Convert an array to `f64`, rejecting anything non-finite.
fn as_finite_f64<T: Float + Debug + Send + Sync + 'static>(
    label: &str,
    values: &Array1<T>,
) -> Result<Vec<f64>> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let as_f64 = value.to_f64().ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "{label}: element {index} cannot be represented as f64"
                ))
            })?;
            if !as_f64.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "{label}: element {index} is {as_f64}"
                )));
            }
            Ok(as_f64)
        })
        .collect()
}

/// Cosine similarity of two equal-length vectors, or `None` if either is zero.
fn cosine_similarity(left: &[f64], right: &[f64]) -> Option<f64> {
    let dot: f64 = left.iter().zip(right.iter()).map(|(a, b)| a * b).sum();
    let left_norm: f64 = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm: f64 = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if left_norm <= 0.0 || right_norm <= 0.0 {
        None
    } else {
        Some((dot / (left_norm * right_norm)).clamp(-1.0, 1.0))
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
            + scirs2_core::ndarray::ScalarOperand,
    > FederatedMetaLearner<T>
{
    /// Compute the first-order meta-gradient from the clients' gradients.
    ///
    /// `client_gradients` supplies the gradient each client reported for the
    /// round; `support_data` and `query_data` supply that client's support and
    /// query gradients for the inner/outer split. Every client in
    /// `client_gradients` must appear in both, and every array must have exactly
    /// [`FederatedMetaLearner::parameter_size`] elements.
    pub fn compute_client_meta_gradients(
        &mut self,
        client_gradients: &HashMap<String, Array1<T>>,
        support_data: &HashMap<String, Array1<T>>,
        query_data: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        self.compute_client_meta_gradients_with_rate(
            client_gradients,
            support_data,
            query_data,
            DEFAULT_INNER_LEARNING_RATE,
        )
    }

    /// As [`FederatedMetaLearner::compute_client_meta_gradients`], with an
    /// explicit inner-loop step size.
    pub fn compute_client_meta_gradients_with_rate(
        &mut self,
        client_gradients: &HashMap<String, Array1<T>>,
        support_data: &HashMap<String, Array1<T>>,
        query_data: &HashMap<String, Array1<T>>,
        inner_learning_rate: f64,
    ) -> Result<Array1<T>> {
        let dimension = self.meta_parameters.len();
        if dimension == 0 {
            return Err(OptimError::InvalidState(
                "this FederatedMetaLearner was constructed with parameter_size = 0, so it has no \
                 storage for a meta-gradient; construct it with the model's parameter count"
                    .to_string(),
            ));
        }
        if client_gradients.is_empty() {
            return Err(OptimError::InvalidParameter(
                "no client gradients were supplied, so there is no meta-gradient to compute"
                    .to_string(),
            ));
        }
        if !inner_learning_rate.is_finite() || inner_learning_rate <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the inner learning rate must be positive and finite, got {inner_learning_rate}"
            )));
        }

        let meta = as_finite_f64("meta_parameters", &self.meta_parameters)?;
        let mut accumulated = vec![0.0f64; dimension];
        let mut adaptations: HashMap<String, Vec<f64>> = HashMap::new();
        let mut distributions: HashMap<String, (Vec<f64>, Vec<f64>, f64)> = HashMap::new();

        // Deterministic client order: `HashMap` iteration order is randomised
        // per process, and floating-point addition is not associative, so an
        // unordered sum would not be reproducible.
        let mut client_ids: Vec<&String> = client_gradients.keys().collect();
        client_ids.sort();

        for client_id in &client_ids {
            let reported = client_gradients.get(*client_id).ok_or_else(|| {
                OptimError::InvalidState(format!("client `{client_id}` vanished from the map"))
            })?;
            if reported.len() != dimension {
                return Err(OptimError::DimensionMismatch(format!(
                    "client `{client_id}` reported a {}-element gradient for a {dimension}-element \
                     model",
                    reported.len()
                )));
            }
            let support = support_data.get(*client_id).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "client `{client_id}` has a gradient but no support gradient, so no inner \
                     adaptation step can be taken"
                ))
            })?;
            let query = query_data.get(*client_id).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "client `{client_id}` has a gradient but no query gradient, so no outer step \
                     can be taken"
                ))
            })?;
            if support.len() != dimension || query.len() != dimension {
                return Err(OptimError::DimensionMismatch(format!(
                    "client `{client_id}` supplied a {}-element support gradient and a {}-element \
                     query gradient for a {dimension}-element model",
                    support.len(),
                    query.len()
                )));
            }

            let support_values = as_finite_f64(&format!("client `{client_id}` support"), support)?;
            let query_values = as_finite_f64(&format!("client `{client_id}` query"), query)?;

            // One inner step: theta_c = theta - alpha * g_support.
            let adapted: Vec<f64> = meta
                .iter()
                .zip(support_values.iter())
                .map(|(parameter, gradient)| parameter - inner_learning_rate * gradient)
                .collect();

            // First-order outer step: the meta-gradient is the mean of the
            // query gradients evaluated at the adapted parameters.
            for (slot, value) in accumulated.iter_mut().zip(query_values.iter()) {
                *slot += value;
            }

            let similarity = cosine_similarity(&support_values, &query_values).unwrap_or(0.0);
            adaptations.insert((*client_id).clone(), adapted);
            distributions.insert(
                (*client_id).clone(),
                (support_values, query_values, similarity),
            );
        }

        let count = client_ids.len() as f64;
        let mut meta_gradient = Array1::zeros(dimension);
        for (index, total) in accumulated.iter().enumerate() {
            let averaged = total / count;
            meta_gradient[index] = T::from(averaged).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "meta-gradient element {index} ({averaged}) cannot be represented"
                ))
            })?;
        }

        // Commit the recorded state only once every client has validated.
        for (client_id, adapted) in adaptations {
            let mut array = Array1::zeros(dimension);
            for (index, value) in adapted.iter().enumerate() {
                array[index] = T::from(*value).unwrap_or_else(T::zero);
            }
            self.client_adaptations.insert(client_id, array);
        }
        for (client_id, (support, query, similarity)) in distributions {
            let mut support_array = Array1::zeros(dimension);
            let mut query_array = Array1::zeros(dimension);
            for index in 0..dimension {
                support_array[index] = T::from(support[index]).unwrap_or_else(T::zero);
                query_array[index] = T::from(query[index]).unwrap_or_else(T::zero);
            }
            self.task_distributions.insert(
                client_id,
                TaskDistribution {
                    support_gradient: support_array,
                    query_gradient: query_array,
                    task_similarity: similarity,
                    adaptation_steps: 1,
                },
            );
        }
        self.meta_gradient_buffer = meta_gradient.clone();
        Ok(meta_gradient)
    }

    /// Apply the buffered meta-gradient to the meta-parameters.
    pub fn apply_meta_gradient(&mut self, outer_learning_rate: f64) -> Result<()> {
        if !outer_learning_rate.is_finite() || outer_learning_rate <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the outer learning rate must be positive and finite, got {outer_learning_rate}"
            )));
        }
        if self.meta_gradient_buffer.len() != self.meta_parameters.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "the meta-gradient has {} elements but the meta-parameters have {}",
                self.meta_gradient_buffer.len(),
                self.meta_parameters.len()
            )));
        }
        let rate = T::from(outer_learning_rate).ok_or_else(|| {
            OptimError::InvalidParameter(
                "the outer learning rate cannot be represented in the parameter type".to_string(),
            )
        })?;
        for index in 0..self.meta_parameters.len() {
            self.meta_parameters[index] =
                self.meta_parameters[index] - rate * self.meta_gradient_buffer[index];
        }
        Ok(())
    }
}

impl<T: Float + Debug + Send + Sync + 'static> TaskDetector<T> {
    /// Number of rounds currently buffered.
    pub fn buffered_rounds(&self) -> usize {
        self.gradient_buffer.len()
    }

    /// Detect whether this round's updates come from a different task.
    ///
    /// Returns `false` (and buffers the round) until there is history to compare
    /// against. The comparison is the larger of
    ///
    /// * the relative L2 shift `||mean_now - mean_history|| / max(||mean_history||, eps)`, and
    /// * the cosine dissimilarity `(1 - cos(mean_now, mean_history)) / 2`,
    ///
    /// so both a magnitude jump and a direction reversal are detected.
    pub fn detect_task_change(&mut self, updates: &[Array1<T>], round: usize) -> Result<bool> {
        match self.detection_method {
            TaskDetectionMethod::GradientBased => {}
            other => {
                return Err(OptimError::UnsupportedOperation(format!(
                    "TaskDetectionMethod::{other:?} is not implemented; only GradientBased \
                     detection exists, and reporting its verdict under another name would \
                     misdescribe what was measured"
                )))
            }
        }
        if updates.is_empty() {
            return Err(OptimError::InvalidParameter(
                "no client updates were supplied, so no task change can be detected".to_string(),
            ));
        }
        let dimension = updates[0].len();
        if dimension == 0 {
            return Err(OptimError::InvalidParameter(
                "the client updates are zero-dimensional".to_string(),
            ));
        }
        for (index, update) in updates.iter().enumerate() {
            if update.len() != dimension {
                return Err(OptimError::DimensionMismatch(format!(
                    "update {index} has {} elements, expected {dimension}",
                    update.len()
                )));
            }
        }

        // Mean update for this round.
        let mut mean_now = vec![0.0f64; dimension];
        for update in updates {
            let values = as_finite_f64("client update", update)?;
            for (slot, value) in mean_now.iter_mut().zip(values.iter()) {
                *slot += value;
            }
        }
        let count = updates.len() as f64;
        for slot in mean_now.iter_mut() {
            *slot /= count;
        }

        // Running mean of the buffered history.
        let history: Vec<Vec<f64>> = self
            .gradient_buffer
            .iter()
            .map(|entry| as_finite_f64("buffered update", entry))
            .collect::<Result<Vec<Vec<f64>>>>()?;
        let comparable: Vec<&Vec<f64>> = history
            .iter()
            .filter(|entry| entry.len() == dimension)
            .collect();

        let mut detected = false;
        if !comparable.is_empty() {
            let mut mean_history = vec![0.0f64; dimension];
            for entry in &comparable {
                for (slot, value) in mean_history.iter_mut().zip(entry.iter()) {
                    *slot += value;
                }
            }
            let history_count = comparable.len() as f64;
            for slot in mean_history.iter_mut() {
                *slot /= history_count;
            }

            let history_norm: f64 = mean_history
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let shift_norm: f64 = mean_now
                .iter()
                .zip(mean_history.iter())
                .map(|(now, past)| (now - past) * (now - past))
                .sum::<f64>()
                .sqrt();
            let relative_shift = shift_norm / history_norm.max(f64::EPSILON);
            let dissimilarity = cosine_similarity(&mean_now, &mean_history)
                .map(|similarity| (1.0 - similarity) / 2.0)
                .unwrap_or(0.0);
            let magnitude = relative_shift.max(dissimilarity);

            if magnitude > self.detection_threshold {
                detected = true;
                // Confidence rises with how far past the threshold the shift is,
                // saturating at 1; it is a monotone transform of the measured
                // magnitude, not an invented constant.
                let excess = (magnitude - self.detection_threshold)
                    / self.detection_threshold.max(f64::EPSILON);
                self.change_points.push(ChangePoint {
                    round,
                    confidence: (excess / (1.0 + excess)).clamp(0.0, 1.0),
                    change_magnitude: magnitude,
                });
            }
        }

        // Buffer this round's mean for the next comparison.
        let mut buffered = Array1::zeros(dimension);
        for (index, value) in mean_now.iter().enumerate() {
            buffered[index] = T::from(*value).unwrap_or_else(T::zero);
        }
        self.gradient_buffer.push_back(buffered);
        while self.gradient_buffer.len() > MAX_GRADIENT_BUFFER {
            let _ = self.gradient_buffer.pop_front();
        }

        // A detected change starts a new task, so the old history is no longer
        // the right baseline.
        if detected {
            self.gradient_buffer.clear();
            let mut restart = Array1::zeros(dimension);
            for (index, value) in mean_now.iter().enumerate() {
                restart[index] = T::from(*value).unwrap_or_else(T::zero);
            }
            self.gradient_buffer.push_back(restart);
        }

        Ok(detected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(values: &[f64]) -> Array1<f64> {
        Array1::from(values.to_vec())
    }

    /// `(client gradients, support gradients, query gradients)`.
    type GradientMaps = (
        HashMap<String, Array1<f64>>,
        HashMap<String, Array1<f64>>,
        HashMap<String, Array1<f64>>,
    );

    /// `(name, reported gradient, support gradient, query gradient)`.
    type ClientEntry<'a> = (&'a str, [f64; 3], [f64; 3], [f64; 3]);

    fn maps(entries: &[ClientEntry<'_>]) -> GradientMaps {
        let mut gradients = HashMap::new();
        let mut support = HashMap::new();
        let mut query = HashMap::new();
        for (name, gradient, support_gradient, query_gradient) in entries {
            gradients.insert((*name).to_string(), array(gradient));
            support.insert((*name).to_string(), array(support_gradient));
            query.insert((*name).to_string(), array(query_gradient));
        }
        (gradients, support, query)
    }

    #[test]
    fn a_meta_learner_sized_for_a_model_allocates_real_buffers() {
        // Regression for F109: `new` discarded its argument and allocated
        // `Array1::default(0)`, so a caller sizing for a real model got nothing.
        let learner = FederatedMetaLearner::<f64>::new(1_000);
        assert_eq!(learner.parameter_size(), 1_000);
        assert_eq!(learner.meta_parameters().len(), 1_000);
        assert_eq!(learner.meta_gradient_buffer().len(), 1_000);
    }

    #[test]
    fn an_unsized_meta_learner_reports_it_instead_of_returning_an_empty_array() {
        let mut learner = FederatedMetaLearner::<f64>::new(0);
        let (gradients, support, query) =
            maps(&[("a", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5])]);
        let message = match learner.compute_client_meta_gradients(&gradients, &support, &query) {
            Err(err) => err.to_string(),
            Ok(gradient) => panic!("returned a {}-element gradient", gradient.len()),
        };
        assert!(message.contains("parameter_size = 0"), "got: {message}");
    }

    #[test]
    fn the_meta_gradient_is_the_mean_of_the_query_gradients() {
        let mut learner = FederatedMetaLearner::<f64>::new(3);
        let (gradients, support, query) = maps(&[
            ("a", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 4.0, 0.0]),
        ]);
        let meta_gradient =
            match learner.compute_client_meta_gradients(&gradients, &support, &query) {
                Ok(gradient) => gradient,
                Err(err) => panic!("meta-gradient failed: {err}"),
            };
        assert_eq!(meta_gradient.len(), 3);
        assert!((meta_gradient[0] - 1.0).abs() < 1e-12, "{meta_gradient:?}");
        assert!((meta_gradient[1] - 2.0).abs() < 1e-12, "{meta_gradient:?}");
        assert!((meta_gradient[2] - 0.0).abs() < 1e-12, "{meta_gradient:?}");
        assert_eq!(learner.meta_gradient_buffer(), &meta_gradient);
    }

    #[test]
    fn the_inner_step_is_recorded_per_client() {
        let mut learner = FederatedMetaLearner::<f64>::new(3);
        let (gradients, support, query) =
            maps(&[("a", [1.0, 1.0, 1.0], [10.0, 20.0, 30.0], [1.0, 1.0, 1.0])]);
        let ok = learner.compute_client_meta_gradients_with_rate(&gradients, &support, &query, 0.1);
        assert!(ok.is_ok(), "meta-gradient failed");
        let adaptation = match learner.client_adaptation("a") {
            Some(adaptation) => adaptation,
            None => panic!("the per-client adaptation must be recorded"),
        };
        // theta = 0 initially, so theta_a = -0.1 * [10, 20, 30].
        assert!((adaptation[0] + 1.0).abs() < 1e-12, "{adaptation:?}");
        assert!((adaptation[1] + 2.0).abs() < 1e-12, "{adaptation:?}");
        assert!((adaptation[2] + 3.0).abs() < 1e-12, "{adaptation:?}");

        let distribution = match learner.task_distribution("a") {
            Some(distribution) => distribution,
            None => panic!("the task distribution must be recorded"),
        };
        assert_eq!(distribution.adaptation_steps, 1);
        // support = [10,20,30], query = [1,1,1]; the cosine is positive.
        assert!(distribution.task_similarity > 0.0);
        assert!(distribution.task_similarity <= 1.0);
    }

    #[test]
    fn the_meta_gradient_is_reproducible_regardless_of_map_insertion_order() {
        // `HashMap` iteration order is randomised per process, and float
        // addition is not associative, so the sum must be over a sorted order.
        let mut left = FederatedMetaLearner::<f64>::new(3);
        let (gradients, support, query) = maps(&[
            ("a", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.2, 0.3]),
            ("b", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.4, 0.5, 0.6]),
            ("c", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.7, 0.8, 0.9]),
        ]);
        let first = match left.compute_client_meta_gradients(&gradients, &support, &query) {
            Ok(gradient) => gradient,
            Err(err) => panic!("meta-gradient failed: {err}"),
        };

        let mut right = FederatedMetaLearner::<f64>::new(3);
        let mut reordered_gradients = HashMap::new();
        for name in ["c", "a", "b"] {
            if let Some(value) = gradients.get(name) {
                reordered_gradients.insert(name.to_string(), value.clone());
            }
        }
        let second =
            match right.compute_client_meta_gradients(&reordered_gradients, &support, &query) {
                Ok(gradient) => gradient,
                Err(err) => panic!("meta-gradient failed: {err}"),
            };
        assert_eq!(first, second);
    }

    #[test]
    fn a_mismatched_or_missing_client_gradient_is_refused() {
        let mut learner = FederatedMetaLearner::<f64>::new(3);
        let (gradients, support, query) =
            maps(&[("a", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5])]);

        // Wrong length.
        let mut wrong = gradients.clone();
        wrong.insert("a".to_string(), array(&[1.0, 2.0]));
        assert!(learner
            .compute_client_meta_gradients(&wrong, &support, &query)
            .is_err());

        // Missing support gradient.
        assert!(learner
            .compute_client_meta_gradients(&gradients, &HashMap::new(), &query)
            .is_err());
        // Missing query gradient.
        assert!(learner
            .compute_client_meta_gradients(&gradients, &support, &HashMap::new())
            .is_err());
        // No clients at all.
        assert!(learner
            .compute_client_meta_gradients(&HashMap::new(), &support, &query)
            .is_err());
        // Non-finite input.
        let mut poisoned = support.clone();
        poisoned.insert("a".to_string(), array(&[f64::NAN, 1.0, 1.0]));
        assert!(learner
            .compute_client_meta_gradients(&gradients, &poisoned, &query)
            .is_err());
        // Bad inner learning rate.
        assert!(learner
            .compute_client_meta_gradients_with_rate(&gradients, &support, &query, 0.0)
            .is_err());
    }

    #[test]
    fn applying_the_meta_gradient_moves_the_meta_parameters() {
        let mut learner = FederatedMetaLearner::<f64>::new(2);
        let mut gradients = HashMap::new();
        gradients.insert("a".to_string(), array(&[1.0, 1.0]));
        let mut support = HashMap::new();
        support.insert("a".to_string(), array(&[1.0, 1.0]));
        let mut query = HashMap::new();
        query.insert("a".to_string(), array(&[2.0, -4.0]));
        let ok = learner.compute_client_meta_gradients(&gradients, &support, &query);
        assert!(ok.is_ok());

        let ok = learner.apply_meta_gradient(0.5);
        assert!(ok.is_ok(), "apply failed");
        assert!((learner.meta_parameters()[0] + 1.0).abs() < 1e-12);
        assert!((learner.meta_parameters()[1] - 2.0).abs() < 1e-12);
        assert!(learner.apply_meta_gradient(0.0).is_err());
        assert!(learner.apply_meta_gradient(f64::NAN).is_err());
    }

    #[test]
    fn a_stable_gradient_stream_reports_no_task_change() {
        // Regression: the placeholder returned `Ok(false)` for everything, so a
        // test that only checks "no change on stable input" would have passed
        // against it. The next test is the one that could not.
        let mut detector = TaskDetector::<f64>::new();
        for round in 0..8usize {
            let updates = vec![array(&[1.0, 0.5, -0.25]), array(&[1.02, 0.48, -0.26])];
            match detector.detect_task_change(&updates, round) {
                Ok(false) => {}
                Ok(true) => panic!("a stable stream must not flag a change at round {round}"),
                Err(err) => panic!("detection failed: {err}"),
            }
        }
        assert!(detector.change_points().is_empty());
        assert!(detector.buffered_rounds() > 0);
    }

    #[test]
    fn a_direction_reversal_is_detected() {
        let mut detector = TaskDetector::<f64>::new();
        for round in 0..4usize {
            let updates = vec![array(&[1.0, 1.0, 1.0])];
            match detector.detect_task_change(&updates, round) {
                Ok(false) => {}
                Ok(true) => panic!("the warm-up rounds must not flag a change"),
                Err(err) => panic!("detection failed: {err}"),
            }
        }
        // Same magnitude, opposite direction.
        let flipped = vec![array(&[-1.0, -1.0, -1.0])];
        match detector.detect_task_change(&flipped, 4) {
            Ok(true) => {}
            Ok(false) => panic!("a sign flip must be detected"),
            Err(err) => panic!("detection failed: {err}"),
        }
        let change = match detector.change_points().first() {
            Some(change) => change,
            None => panic!("the change point must be recorded"),
        };
        assert_eq!(change.round, 4);
        assert!(change.change_magnitude > detector.detection_threshold());
        assert!((0.0..=1.0).contains(&change.confidence));
    }

    #[test]
    fn a_magnitude_jump_is_detected() {
        let mut detector = TaskDetector::<f64>::new();
        for round in 0..4usize {
            let updates = vec![array(&[0.01, 0.01, 0.01])];
            let ok = detector.detect_task_change(&updates, round);
            assert!(matches!(ok, Ok(false)), "{ok:?}");
        }
        let jump = vec![array(&[10.0, 10.0, 10.0])];
        match detector.detect_task_change(&jump, 4) {
            Ok(true) => {}
            Ok(false) => panic!("a 1000x magnitude jump must be detected"),
            Err(err) => panic!("detection failed: {err}"),
        }
    }

    #[test]
    fn the_threshold_governs_sensitivity() {
        let mut sensitive = TaskDetector::<f64>::new();
        let ok = sensitive.set_detection_threshold(0.001);
        assert!(ok.is_ok());
        let mut tolerant = TaskDetector::<f64>::new();
        let ok = tolerant.set_detection_threshold(5.0);
        assert!(ok.is_ok());

        for round in 0..3usize {
            let updates = vec![array(&[1.0, 1.0])];
            let _ = sensitive.detect_task_change(&updates, round);
            let _ = tolerant.detect_task_change(&updates, round);
        }
        let shifted = vec![array(&[1.2, 1.2])];
        assert!(
            matches!(sensitive.detect_task_change(&shifted, 3), Ok(true)),
            "a 0.001 threshold must flag a 20% shift"
        );
        assert!(
            matches!(tolerant.detect_task_change(&shifted, 3), Ok(false)),
            "a 5.0 threshold must not flag a 20% shift"
        );
        assert!(tolerant.set_detection_threshold(0.0).is_err());
        assert!(tolerant.set_detection_threshold(f64::NAN).is_err());
    }

    #[test]
    fn detection_restarts_its_baseline_after_a_change() {
        let mut detector = TaskDetector::<f64>::new();
        for round in 0..3usize {
            let _ = detector.detect_task_change(&[array(&[1.0, 1.0])], round);
        }
        assert!(matches!(
            detector.detect_task_change(&[array(&[-1.0, -1.0])], 3),
            Ok(true)
        ));
        assert_eq!(
            detector.buffered_rounds(),
            1,
            "the baseline must restart from the new task"
        );
        // The new regime is now normal, so it must not keep firing.
        assert!(matches!(
            detector.detect_task_change(&[array(&[-1.0, -1.0])], 4),
            Ok(false)
        ));
        assert_eq!(detector.change_points().len(), 1);
    }

    #[test]
    fn degenerate_detection_inputs_are_refused() {
        let mut detector = TaskDetector::<f64>::new();
        assert!(detector.detect_task_change(&[], 0).is_err());
        assert!(detector
            .detect_task_change(&[Array1::<f64>::zeros(0)], 0)
            .is_err());
        assert!(detector
            .detect_task_change(&[array(&[1.0, 2.0]), array(&[1.0])], 0)
            .is_err());
        assert!(detector
            .detect_task_change(&[array(&[f64::INFINITY, 1.0])], 0)
            .is_err());
    }

    #[test]
    fn unimplemented_detection_methods_are_refused() {
        for method in [
            TaskDetectionMethod::LossBased,
            TaskDetectionMethod::StatisticalTest,
            TaskDetectionMethod::ChangePointDetection,
            TaskDetectionMethod::EnsembleMethods,
        ] {
            let mut detector = TaskDetector::<f64>::new();
            detector.detection_method = method;
            let outcome = detector.detect_task_change(&[array(&[1.0, 1.0])], 0);
            assert!(
                outcome.is_err(),
                "{method:?} must not report a gradient-based verdict under another name"
            );
        }
    }
}
