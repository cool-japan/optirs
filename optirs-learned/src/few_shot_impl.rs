// Few-Shot Learning Implementation
//
// Implements core methods for PrototypicalNetwork, FastAdaptationEngine,
// and TaskSimilarityCalculator types defined in crate::few_shot.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::few_shot::{
    AdaptationStrategyType, FastAdaptationEngine, PrototypicalNetwork, TaskSimilarityCalculator,
};

// ---------------------------------------------------------------------------
// PrototypicalNetwork additional impl
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static> PrototypicalNetwork<T> {
    /// Encode a single feature vector through the encoder network.
    ///
    /// Performs a simple linear projection: out = features * W + b (truncating
    /// or zero-padding the input to match the weight matrix dimensions).
    /// A ReLU activation is applied element-wise.
    pub fn encode(&self, features: &Array1<T>) -> Result<Array1<T>> {
        let emb_dim = self.embedding_dim();
        let layers = self.encoder_layers();
        if layers.is_empty() {
            return Err(OptimError::InvalidState(
                "Encoder has no layers".to_string(),
            ));
        }
        let layer = &layers[0];
        let (input_rows, output_cols) = (layer.weights.nrows(), layer.weights.ncols());
        let actual_out = output_cols.min(emb_dim);

        // Build padded/truncated input
        let mut input = vec![T::zero(); input_rows];
        let copy_len = features.len().min(input_rows);
        for i in 0..copy_len {
            input[i] = features[i];
        }

        // Manual matmul: output[j] = sum_i input[i] * weights[i][j] + bias[j]
        let mut output = Array1::<T>::zeros(emb_dim);
        for j in 0..actual_out {
            let mut acc = layer.bias[j];
            for (i, &inp_val) in input.iter().enumerate().take(input_rows) {
                acc = acc + inp_val * layer.weights[[i, j]];
            }
            // ReLU activation
            output[j] = if acc > T::zero() { acc } else { T::zero() };
        }
        Ok(output)
    }

    /// Compute a prototype (class centroid) from a set of example embeddings.
    ///
    /// The prototype is the element-wise mean of the encoded examples.
    pub fn compute_prototype(&self, examples: &[Array1<T>]) -> Result<Array1<T>> {
        if examples.is_empty() {
            return Err(OptimError::InsufficientData(
                "Cannot compute prototype from empty example set".to_string(),
            ));
        }
        let dim = examples[0].len();
        let mut sum = Array1::<T>::zeros(dim);
        for ex in examples {
            let len = ex.len().min(dim);
            for i in 0..len {
                sum[i] = sum[i] + ex[i];
            }
        }
        let count_t: T =
            scirs2_core::numeric::NumCast::from(examples.len()).unwrap_or_else(|| T::one());
        for i in 0..dim {
            sum[i] = sum[i] / count_t;
        }
        Ok(sum)
    }

    /// Classify a query by finding the nearest prototype.
    ///
    /// Returns the index of the closest prototype according to squared
    /// Euclidean distance.
    pub fn classify(&self, query: &Array1<T>, prototypes: &[Array1<T>]) -> Result<usize> {
        if prototypes.is_empty() {
            return Err(OptimError::InsufficientData(
                "No prototypes to classify against".to_string(),
            ));
        }
        let (idx, _dist) = self.find_nearest_in_list(query, prototypes)?;
        Ok(idx)
    }

    /// Find the nearest prototype and return its index plus the distance.
    pub fn find_nearest_prototype(&self, query: &Array1<T>) -> Result<(usize, T)> {
        let stored = self.prototypes();
        if stored.is_empty() {
            return Err(OptimError::InsufficientData(
                "No stored prototypes".to_string(),
            ));
        }
        let vecs: Vec<Array1<T>> = stored.values().map(|p| p.vector.clone()).collect();
        self.find_nearest_in_list(query, &vecs)
    }

    // ---- private helpers ----

    fn find_nearest_in_list(
        &self,
        query: &Array1<T>,
        candidates: &[Array1<T>],
    ) -> Result<(usize, T)> {
        if candidates.is_empty() {
            return Err(OptimError::InsufficientData(
                "No candidates for nearest search".to_string(),
            ));
        }
        let mut best_idx = 0;
        let mut best_dist = T::infinity();
        for (i, proto) in candidates.iter().enumerate() {
            let dist = squared_euclidean(query, proto);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        Ok((best_idx, best_dist))
    }
}

// ---------------------------------------------------------------------------
// FastAdaptationEngine additional impl
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static> FastAdaptationEngine<T> {
    /// Perform multi-step gradient adaptation on parameters.
    ///
    /// For each gradient in `gradients`, applies one step:
    ///   params = params - inner_lr * gradient
    /// returning the final adapted parameters.
    pub fn adapt(
        &self,
        params: &Array1<T>,
        gradients: &[Array1<T>],
        inner_lr: T,
    ) -> Result<Array1<T>> {
        if params.is_empty() {
            return Err(OptimError::InvalidState(
                "Parameters must not be empty".to_string(),
            ));
        }
        let mut current = params.clone();
        for grad in gradients {
            let len = current.len().min(grad.len());
            for i in 0..len {
                current[i] = current[i] - inner_lr * grad[i];
            }
        }
        Ok(current)
    }

    /// Select an adaptation algorithm based on task complexity.
    ///
    /// Low complexity  -> FOMAML (fast, first-order)
    /// Medium          -> MAML (second-order)
    /// High            -> MemoryAugmented (richer capacity)
    pub fn select_algorithm(&self, task_complexity: T) -> AdaptationStrategyType {
        let low: T = scirs2_core::numeric::NumCast::from(0.3).unwrap_or_else(|| T::zero());
        let high: T = scirs2_core::numeric::NumCast::from(0.7).unwrap_or_else(|| T::one());

        if task_complexity < low {
            AdaptationStrategyType::FOMAML
        } else if task_complexity < high {
            AdaptationStrategyType::MAML
        } else {
            AdaptationStrategyType::MemoryAugmented
        }
    }

    /// Evaluate the quality of an adaptation step.
    ///
    /// Returns the relative improvement: ||before|| - ||after|| normalised by
    /// ||before|| (higher is better; negative means regression).
    pub fn evaluate_adaptation(&self, before: &Array1<T>, after: &Array1<T>) -> Result<T> {
        let norm_before = vec_norm(before);
        let norm_after = vec_norm(after);

        if norm_before == T::zero() {
            return Ok(T::zero());
        }
        Ok((norm_before - norm_after) / norm_before)
    }
}

// ---------------------------------------------------------------------------
// TaskSimilarityCalculator additional impl
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static> TaskSimilarityCalculator<T> {
    /// Compute cosine similarity between two task representation vectors.
    ///
    /// Returns a value in [-1, 1], with 1 meaning identical direction.
    pub fn compute_similarity(&self, task1_repr: &Array1<T>, task2_repr: &Array1<T>) -> Result<T> {
        if task1_repr.is_empty() || task2_repr.is_empty() {
            return Err(OptimError::InsufficientData(
                "Task representations must not be empty".to_string(),
            ));
        }
        let n1 = vec_norm(task1_repr);
        let n2 = vec_norm(task2_repr);
        if n1 == T::zero() || n2 == T::zero() {
            return Ok(T::zero());
        }
        let len = task1_repr.len().min(task2_repr.len());
        let mut dot = T::zero();
        for i in 0..len {
            dot = dot + task1_repr[i] * task2_repr[i];
        }
        Ok(dot / (n1 * n2))
    }

    /// Find the most similar candidate to the query.
    ///
    /// Returns `(index, similarity)` of the best match.
    pub fn find_most_similar(
        &self,
        query: &Array1<T>,
        candidates: &[Array1<T>],
    ) -> Result<(usize, T)> {
        if candidates.is_empty() {
            return Err(OptimError::InsufficientData(
                "No candidates for similarity search".to_string(),
            ));
        }
        let mut best_idx = 0;
        let mut best_sim = T::neg_infinity();
        for (i, cand) in candidates.iter().enumerate() {
            let sim = self.compute_similarity(query, cand)?;
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        Ok((best_idx, best_sim))
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two vectors (truncated to shorter length).
fn squared_euclidean<T: Float>(a: &Array1<T>, b: &Array1<T>) -> T {
    let len = a.len().min(b.len());
    let mut sum = T::zero();
    for i in 0..len {
        let d = a[i] - b[i];
        sum = sum + d * d;
    }
    sum
}

/// L2 norm of a vector.
fn vec_norm<T: Float>(v: &Array1<T>) -> T {
    let mut sum = T::zero();
    for &x in v.iter() {
        sum = sum + x * x;
    }
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_prototypical_network_encode() {
        let net = PrototypicalNetwork::<f64>::from_dims(4, 3)
            .expect("failed to create PrototypicalNetwork");
        // With zero-initialised weights the encode should return all zeros (ReLU of 0 = 0)
        let features = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let encoded = net.encode(&features).expect("encode failed");
        assert_eq!(encoded.len(), 4);
        // All outputs should be zero because weights are initialised to zero
        for &v in encoded.iter() {
            assert!((v - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_compute_prototype() {
        let net = PrototypicalNetwork::<f64>::from_dims(3, 2)
            .expect("failed to create PrototypicalNetwork");
        let examples = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![3.0, 4.0, 5.0]),
            Array1::from_vec(vec![5.0, 6.0, 7.0]),
        ];
        let proto = net
            .compute_prototype(&examples)
            .expect("compute_prototype failed");
        assert_eq!(proto.len(), 3);
        assert!((proto[0] - 3.0).abs() < 1e-12);
        assert!((proto[1] - 4.0).abs() < 1e-12);
        assert!((proto[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_classify_nearest() {
        let net = PrototypicalNetwork::<f64>::from_dims(3, 3)
            .expect("failed to create PrototypicalNetwork");
        let prototypes = vec![
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![10.0, 10.0, 10.0]),
            Array1::from_vec(vec![20.0, 20.0, 20.0]),
        ];
        // Query close to second prototype
        let query = Array1::from_vec(vec![9.0, 11.0, 10.0]);
        let class = net.classify(&query, &prototypes).expect("classify failed");
        assert_eq!(class, 1);

        // Query close to first prototype
        let query2 = Array1::from_vec(vec![0.1, -0.1, 0.2]);
        let class2 = net.classify(&query2, &prototypes).expect("classify failed");
        assert_eq!(class2, 0);
    }

    #[test]
    fn test_fast_adaptation() {
        let engine = FastAdaptationEngine::<f64>::from_params(0.1, 5)
            .expect("failed to create FastAdaptationEngine");
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = vec![
            Array1::from_vec(vec![0.5, 0.5, 0.5]),
            Array1::from_vec(vec![0.3, 0.3, 0.3]),
        ];
        let adapted = engine
            .adapt(&params, &gradients, 0.1)
            .expect("adapt failed");
        // After step 1: [1.0 - 0.05, 2.0 - 0.05, 3.0 - 0.05] = [0.95, 1.95, 2.95]
        // After step 2: [0.95 - 0.03, 1.95 - 0.03, 2.95 - 0.03] = [0.92, 1.92, 2.92]
        assert!((adapted[0] - 0.92).abs() < 1e-12);
        assert!((adapted[1] - 1.92).abs() < 1e-12);
        assert!((adapted[2] - 2.92).abs() < 1e-12);

        // Test algorithm selection
        let strat_low = engine.select_algorithm(0.1);
        assert!(matches!(strat_low, AdaptationStrategyType::FOMAML));
        let strat_mid = engine.select_algorithm(0.5);
        assert!(matches!(strat_mid, AdaptationStrategyType::MAML));
        let strat_high = engine.select_algorithm(0.9);
        assert!(matches!(
            strat_high,
            AdaptationStrategyType::MemoryAugmented
        ));
    }

    #[test]
    fn test_task_similarity() {
        let calc = TaskSimilarityCalculator::<f64>::default_new()
            .expect("failed to create TaskSimilarityCalculator");
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let sim = calc
            .compute_similarity(&a, &b)
            .expect("compute_similarity failed");
        assert!(
            (sim - 1.0).abs() < 1e-12,
            "identical vectors should have similarity 1.0"
        );

        // Orthogonal
        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let sim2 = calc
            .compute_similarity(&a, &c)
            .expect("compute_similarity failed");
        assert!(
            sim2.abs() < 1e-12,
            "orthogonal vectors should have similarity ~0"
        );

        // Opposite
        let d = Array1::from_vec(vec![-1.0, 0.0, 0.0]);
        let sim3 = calc
            .compute_similarity(&a, &d)
            .expect("compute_similarity failed");
        assert!(
            (sim3 - (-1.0)).abs() < 1e-12,
            "opposite vectors should have similarity -1.0"
        );
    }

    #[test]
    fn test_find_most_similar() {
        let calc = TaskSimilarityCalculator::<f64>::default_new()
            .expect("failed to create TaskSimilarityCalculator");
        let query = Array1::from_vec(vec![1.0, 1.0, 0.0]);
        let candidates = vec![
            Array1::from_vec(vec![0.0, 0.0, 1.0]),   // orthogonal to query
            Array1::from_vec(vec![1.0, 1.0, 0.01]),  // very similar to query
            Array1::from_vec(vec![-1.0, -1.0, 0.0]), // opposite to query
        ];
        let (idx, sim) = calc
            .find_most_similar(&query, &candidates)
            .expect("find_most_similar failed");
        assert_eq!(idx, 1);
        assert!(
            sim > 0.99,
            "best match should have high similarity, got {sim}"
        );
    }
}
