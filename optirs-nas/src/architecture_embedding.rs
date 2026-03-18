//! Architecture embedding for similarity search and performance prediction.
//!
//! Provides methods to embed optimizer architectures into dense vector spaces,
//! enabling similarity comparison, clustering, and gradient-free embedding updates
//! based on architecture performance.

use crate::architecture::{Architecture, ComponentType};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Method used to aggregate component embeddings into a single architecture embedding
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Simple mean of all component embeddings
    Mean,
    /// Weighted sum based on component counts
    WeightedSum,
    /// Softmax-weighted attention pooling
    AttentionPooling,
}

/// Embeds optimizer architectures into dense vector representations
///
/// Uses per-component-type embeddings that are aggregated to produce
/// architecture-level vectors suitable for similarity search and clustering.
#[derive(Debug, Clone)]
pub struct ArchitectureEmbedder<T: Float + Debug + Send + Sync + 'static + ScalarOperand> {
    /// Dimensionality of embedding vectors
    pub embedding_dim: usize,
    /// Learned embeddings for each component type
    pub component_embeddings: HashMap<ComponentType, Array1<T>>,
    /// Aggregation method for combining component embeddings
    pub aggregation_method: AggregationMethod,
    /// Performance-based weights for embedding updates
    pub performance_weights: HashMap<String, T>,
}

/// All component types for deterministic initialization
const ALL_COMPONENT_TYPES: &[ComponentType] = &[
    ComponentType::SGD,
    ComponentType::Adam,
    ComponentType::AdamW,
    ComponentType::RMSprop,
    ComponentType::AdaGrad,
    ComponentType::AdaDelta,
    ComponentType::Momentum,
    ComponentType::Nesterov,
    ComponentType::LRScheduler,
    ComponentType::GradientClipping,
    ComponentType::BatchNorm,
    ComponentType::Dropout,
    ComponentType::LAMB,
    ComponentType::LARS,
    ComponentType::Lion,
    ComponentType::RAdam,
    ComponentType::Lookahead,
    ComponentType::SAM,
    ComponentType::LBFGS,
    ComponentType::SparseAdam,
    ComponentType::GroupedAdam,
    ComponentType::MAML,
    ComponentType::L1Regularizer,
    ComponentType::L2Regularizer,
    ComponentType::ElasticNetRegularizer,
    ComponentType::DropoutRegularizer,
    ComponentType::WeightDecay,
    ComponentType::AdaptiveLR,
    ComponentType::AdaptiveMomentum,
    ComponentType::AdaptiveRegularization,
    ComponentType::LSTMOptimizer,
    ComponentType::TransformerOptimizer,
    ComponentType::AttentionOptimizer,
    ComponentType::MetaSGD,
    ComponentType::ConstantLR,
    ComponentType::ExponentialLR,
    ComponentType::StepLR,
    ComponentType::CosineAnnealingLR,
    ComponentType::OneCycleLR,
    ComponentType::CyclicLR,
    ComponentType::Reptile,
    ComponentType::Custom,
];

impl<T: Float + Debug + Send + Sync + 'static + ScalarOperand> ArchitectureEmbedder<T> {
    /// Create a new embedder with deterministic sin/cos initialization
    ///
    /// Each component type gets a unique embedding based on positional encoding.
    pub fn new(embedding_dim: usize, aggregation: AggregationMethod) -> Self {
        let mut component_embeddings = HashMap::new();

        for (idx, &comp_type) in ALL_COMPONENT_TYPES.iter().enumerate() {
            let embedding = Self::sincos_encoding(idx, embedding_dim);
            component_embeddings.insert(comp_type, embedding);
        }

        Self {
            embedding_dim,
            component_embeddings,
            aggregation_method: aggregation,
            performance_weights: HashMap::new(),
        }
    }

    /// Generate a deterministic sin/cos positional encoding for a given index
    fn sincos_encoding(index: usize, dim: usize) -> Array1<T> {
        let mut values = Vec::with_capacity(dim);
        let idx_f64 = index as f64;

        for d in 0..dim {
            let denom = (10000.0_f64).powf(2.0 * (d / 2) as f64 / dim as f64);
            let val = if d % 2 == 0 {
                (idx_f64 / denom).sin()
            } else {
                (idx_f64 / denom).cos()
            };
            let t_val: T = scirs2_core::numeric::NumCast::from(val).unwrap_or_else(|| T::zero());
            values.push(t_val);
        }

        Array1::from_vec(values)
    }

    /// Embed an architecture into a dense vector by aggregating component embeddings
    pub fn embed(&self, architecture: &Architecture) -> Result<Array1<T>> {
        if architecture.components.is_empty() {
            return Err(OptimError::ArchitectureError(
                "Cannot embed architecture with no components".to_string(),
            ));
        }

        // Collect embeddings for all enabled components
        let mut embeddings: Vec<Array1<T>> = Vec::new();
        for component in &architecture.components {
            if !component.enabled {
                continue;
            }
            let emb = self
                .component_embeddings
                .get(&component.component_type)
                .ok_or_else(|| {
                    OptimError::ArchitectureError(format!(
                        "No embedding found for component type {:?}",
                        component.component_type
                    ))
                })?;
            embeddings.push(emb.clone());
        }

        if embeddings.is_empty() {
            return Err(OptimError::ArchitectureError(
                "No enabled components to embed".to_string(),
            ));
        }

        match self.aggregation_method {
            AggregationMethod::Mean => self.aggregate_mean(&embeddings),
            AggregationMethod::WeightedSum => {
                self.aggregate_weighted_sum(&embeddings, architecture)
            }
            AggregationMethod::AttentionPooling => self.aggregate_attention(&embeddings),
        }
    }

    /// Aggregate embeddings by taking the mean
    fn aggregate_mean(&self, embeddings: &[Array1<T>]) -> Result<Array1<T>> {
        let n: T =
            scirs2_core::numeric::NumCast::from(embeddings.len()).unwrap_or_else(|| T::one());
        let mut result = Array1::zeros(self.embedding_dim);

        for emb in embeddings {
            Zip::from(&mut result).and(emb).for_each(|r, &e| {
                *r = *r + e;
            });
        }

        result.mapv_inplace(|v| v / n);
        Ok(result)
    }

    /// Aggregate embeddings by weighted sum (weight by occurrence count)
    fn aggregate_weighted_sum(
        &self,
        embeddings: &[Array1<T>],
        architecture: &Architecture,
    ) -> Result<Array1<T>> {
        // Count occurrences of each component type
        let mut type_counts: HashMap<ComponentType, usize> = HashMap::new();
        for comp in &architecture.components {
            if comp.enabled {
                *type_counts.entry(comp.component_type).or_insert(0) += 1;
            }
        }

        let total_count: usize = type_counts.values().sum();
        let total_t: T =
            scirs2_core::numeric::NumCast::from(total_count.max(1)).unwrap_or_else(|| T::one());

        let mut result = Array1::zeros(self.embedding_dim);
        let mut emb_idx = 0;

        for comp in &architecture.components {
            if !comp.enabled {
                continue;
            }
            if emb_idx >= embeddings.len() {
                break;
            }

            let count = type_counts.get(&comp.component_type).copied().unwrap_or(1);
            let weight: T = scirs2_core::numeric::NumCast::from(count).unwrap_or_else(|| T::one());

            let emb = &embeddings[emb_idx];
            Zip::from(&mut result).and(emb).for_each(|r, &e| {
                *r = *r + e * weight;
            });
            emb_idx += 1;
        }

        // Normalize by total count
        result.mapv_inplace(|v| v / total_t);
        Ok(result)
    }

    /// Aggregate embeddings using softmax-weighted attention pooling
    fn aggregate_attention(&self, embeddings: &[Array1<T>]) -> Result<Array1<T>> {
        // Compute attention scores as L2 norm of each embedding
        let mut scores: Vec<T> = Vec::with_capacity(embeddings.len());
        for emb in embeddings {
            let norm_sq = emb.iter().fold(T::zero(), |acc, &v| acc + v * v);
            scores.push(norm_sq.sqrt());
        }

        // Softmax over scores
        let max_score = scores
            .iter()
            .copied()
            .fold(T::neg_infinity(), |a, b| if b > a { b } else { a });

        let mut exp_scores: Vec<T> = scores.iter().map(|&s| (s - max_score).exp()).collect();

        let sum_exp: T = exp_scores.iter().copied().fold(T::zero(), |a, b| a + b);
        if sum_exp > T::zero() {
            for s in &mut exp_scores {
                *s = *s / sum_exp;
            }
        }

        // Weighted sum
        let mut result = Array1::zeros(self.embedding_dim);
        for (emb, &weight) in embeddings.iter().zip(exp_scores.iter()) {
            Zip::from(&mut result).and(emb).for_each(|r, &e| {
                *r = *r + e * weight;
            });
        }

        Ok(result)
    }

    /// Compute cosine similarity between two architecture embeddings
    pub fn compute_similarity(&self, arch1: &Architecture, arch2: &Architecture) -> Result<T> {
        let emb1 = self.embed(arch1)?;
        let emb2 = self.embed(arch2)?;

        let dot: T = Zip::from(&emb1)
            .and(&emb2)
            .fold(T::zero(), |acc, &a, &b| acc + a * b);

        let norm1 = emb1.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt();
        let norm2 = emb2.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt();

        let denom = norm1 * norm2;
        if denom <= T::zero() {
            return Ok(T::zero());
        }

        Ok(dot / denom)
    }

    /// Find the top-k most similar architectures to a query
    ///
    /// Returns vector of (index, similarity_score) pairs sorted by similarity descending.
    pub fn find_similar(
        &self,
        query: &Architecture,
        candidates: &[Architecture],
        k: usize,
    ) -> Result<Vec<(usize, T)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut similarities: Vec<(usize, T)> = Vec::with_capacity(candidates.len());
        for (idx, candidate) in candidates.iter().enumerate() {
            let sim = self.compute_similarity(query, candidate)?;
            similarities.push((idx, sim));
        }

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.truncate(k);
        Ok(similarities)
    }

    /// Update embeddings based on architecture performance (gradient-free)
    ///
    /// For high-performing architectures, slightly increase the magnitude of
    /// their component embeddings. For low-performing ones, slightly decrease.
    /// Embeddings are normalized after the update.
    pub fn update_embeddings(
        &mut self,
        architectures: &[Architecture],
        performances: &[T],
    ) -> Result<()> {
        if architectures.len() != performances.len() {
            return Err(OptimError::InvalidParameter(format!(
                "Architectures count ({}) must match performances count ({})",
                architectures.len(),
                performances.len()
            )));
        }

        if architectures.is_empty() {
            return Ok(());
        }

        // Compute mean performance for normalization
        let n: T =
            scirs2_core::numeric::NumCast::from(performances.len()).unwrap_or_else(|| T::one());
        let mean_perf = performances.iter().copied().fold(T::zero(), |a, b| a + b) / n;

        let learning_rate: T =
            scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero());

        // Update component embeddings based on relative performance
        for (arch, &perf) in architectures.iter().zip(performances.iter()) {
            let relative_perf = perf - mean_perf;

            for component in &arch.components {
                if !component.enabled {
                    continue;
                }
                if let Some(emb) = self.component_embeddings.get_mut(&component.component_type) {
                    // Scale embedding: positive relative perf -> increase, negative -> decrease
                    let scale = T::one() + learning_rate * relative_perf;
                    emb.mapv_inplace(|v| v * scale);
                }
            }
        }

        // Normalize all embeddings to unit length
        for emb in self.component_embeddings.values_mut() {
            let norm = emb.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt();
            if norm > T::zero() {
                emb.mapv_inplace(|v| v / norm);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{ComponentPosition, Connection, ConnectionType, OptimizerComponent};

    fn make_test_architecture(id: &str, component_types: &[ComponentType]) -> Architecture {
        let components: Vec<OptimizerComponent> = component_types
            .iter()
            .enumerate()
            .map(|(i, ct)| OptimizerComponent {
                id: format!("comp_{}", i),
                component_type: *ct,
                hyperparameters: HashMap::new(),
                position: ComponentPosition {
                    layer: i as u32,
                    index: 0,
                    x: i as f32 * 100.0,
                    y: 0.0,
                },
                enabled: true,
            })
            .collect();

        let mut connections = Vec::new();
        for i in 0..components.len().saturating_sub(1) {
            connections.push(Connection {
                from: components[i].id.clone(),
                to: components[i + 1].id.clone(),
                weight: 1.0,
                connection_type: ConnectionType::Gradient,
                enabled: true,
            });
        }

        Architecture {
            id: id.to_string(),
            components,
            connections,
            hyperparameters: HashMap::new(),
            performance: None,
            generation: 0,
        }
    }

    #[test]
    fn test_embed_architecture() {
        let embedder = ArchitectureEmbedder::<f64>::new(16, AggregationMethod::Mean);

        let arch = make_test_architecture(
            "test_arch",
            &[
                ComponentType::Adam,
                ComponentType::SGD,
                ComponentType::BatchNorm,
            ],
        );

        let embedding = embedder.embed(&arch);
        assert!(embedding.is_ok());
        let embedding = embedding.ok().unwrap_or_else(|| Array1::zeros(0));
        assert_eq!(embedding.len(), 16);

        // Embedding should not be all zeros (since we use sin/cos encoding)
        let sum: f64 = embedding.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Embedding should be non-zero");
    }

    #[test]
    fn test_compute_similarity() {
        let embedder = ArchitectureEmbedder::<f64>::new(32, AggregationMethod::Mean);

        // Same architecture should have similarity 1.0
        let arch1 = make_test_architecture("arch1", &[ComponentType::Adam, ComponentType::SGD]);
        let arch2 = make_test_architecture("arch2", &[ComponentType::Adam, ComponentType::SGD]);

        let sim = embedder.compute_similarity(&arch1, &arch2);
        assert!(sim.is_ok());
        let sim = sim.ok().unwrap_or(0.0);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same components should have similarity ~1.0, got {}",
            sim
        );

        // Different architectures should have lower similarity
        let arch3 = make_test_architecture(
            "arch3",
            &[
                ComponentType::LBFGS,
                ComponentType::Nesterov,
                ComponentType::BatchNorm,
            ],
        );
        let sim_diff = embedder.compute_similarity(&arch1, &arch3);
        assert!(sim_diff.is_ok());
        let sim_diff = sim_diff.ok().unwrap_or(1.0);
        assert!(
            sim_diff < sim,
            "Different architectures should be less similar: {} vs {}",
            sim_diff,
            sim
        );
    }

    #[test]
    fn test_find_similar_architectures() {
        let embedder = ArchitectureEmbedder::<f64>::new(16, AggregationMethod::Mean);

        let query = make_test_architecture("query", &[ComponentType::Adam, ComponentType::SGD]);

        let candidates = vec![
            make_test_architecture("c0", &[ComponentType::Adam, ComponentType::SGD]),
            make_test_architecture(
                "c1",
                &[
                    ComponentType::LBFGS,
                    ComponentType::Nesterov,
                    ComponentType::BatchNorm,
                ],
            ),
            make_test_architecture("c2", &[ComponentType::Adam]),
            make_test_architecture(
                "c3",
                &[
                    ComponentType::RMSprop,
                    ComponentType::AdaDelta,
                    ComponentType::Dropout,
                ],
            ),
        ];

        let result = embedder.find_similar(&query, &candidates, 2);
        assert!(result.is_ok());
        let top_k = result.ok().unwrap_or_default();
        assert_eq!(top_k.len(), 2);

        // The first result should be the most similar (same components)
        assert_eq!(
            top_k[0].0, 0,
            "Candidate 0 (identical components) should be most similar"
        );
        // Similarities should be in descending order
        assert!(
            top_k[0].1 >= top_k[1].1,
            "Results should be sorted by similarity descending"
        );
    }

    #[test]
    fn test_update_embeddings() {
        let mut embedder = ArchitectureEmbedder::<f64>::new(16, AggregationMethod::Mean);

        let arch_good =
            make_test_architecture("good", &[ComponentType::Adam, ComponentType::BatchNorm]);
        let arch_bad = make_test_architecture("bad", &[ComponentType::SGD, ComponentType::Dropout]);

        // Record pre-update embedding for Adam
        let adam_emb_before = embedder
            .component_embeddings
            .get(&ComponentType::Adam)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(16));

        let architectures = vec![arch_good, arch_bad];
        let performances = vec![0.95, 0.3];

        let result = embedder.update_embeddings(&architectures, &performances);
        assert!(result.is_ok());

        // After update, embeddings should be normalized (unit length)
        let adam_emb_after = embedder
            .component_embeddings
            .get(&ComponentType::Adam)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(16));

        let norm: f64 = adam_emb_after.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Embeddings should be normalized after update, norm = {}",
            norm
        );

        // Embedding should have changed
        let diff: f64 = Zip::from(&adam_emb_before)
            .and(&adam_emb_after)
            .fold(0.0, |acc, &a, &b| acc + (a - b).abs());
        assert!(diff > 1e-10, "Embeddings should change after update");
    }

    #[test]
    fn test_aggregation_methods() {
        let arch = make_test_architecture(
            "agg_test",
            &[
                ComponentType::Adam,
                ComponentType::SGD,
                ComponentType::BatchNorm,
                ComponentType::GradientClipping,
            ],
        );

        // Test all three aggregation methods produce valid embeddings
        for &method in &[
            AggregationMethod::Mean,
            AggregationMethod::WeightedSum,
            AggregationMethod::AttentionPooling,
        ] {
            let embedder = ArchitectureEmbedder::<f64>::new(8, method);
            let result = embedder.embed(&arch);
            assert!(result.is_ok(), "Aggregation {:?} should succeed", method);
            let emb = result.ok().unwrap_or_else(|| Array1::zeros(0));
            assert_eq!(emb.len(), 8, "Embedding dim should match for {:?}", method);

            let sum: f64 = emb.iter().map(|v| v.abs()).sum();
            assert!(sum > 0.0, "Embedding from {:?} should be non-zero", method);
        }

        // Different aggregation methods should produce different embeddings
        // Use architecture with repeated component types so WeightedSum differs from Mean
        let arch_repeated = make_test_architecture(
            "repeated",
            &[
                ComponentType::Adam,
                ComponentType::Adam,
                ComponentType::SGD,
                ComponentType::BatchNorm,
            ],
        );
        let mean_embedder = ArchitectureEmbedder::<f64>::new(8, AggregationMethod::Mean);
        let ws_embedder = ArchitectureEmbedder::<f64>::new(8, AggregationMethod::WeightedSum);

        let emb_mean = mean_embedder
            .embed(&arch_repeated)
            .ok()
            .unwrap_or_else(|| Array1::zeros(8));
        let emb_ws = ws_embedder
            .embed(&arch_repeated)
            .ok()
            .unwrap_or_else(|| Array1::zeros(8));

        let diff: f64 = Zip::from(&emb_mean)
            .and(&emb_ws)
            .fold(0.0, |acc, &a, &b| acc + (a - b).abs());
        assert!(
            diff > 1e-10,
            "Mean and WeightedSum should produce different embeddings for repeated components"
        );
    }
}
