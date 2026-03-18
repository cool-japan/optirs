//! Cross-domain knowledge transfer between domain optimizers.
//!
//! This module enables transferring learned optimization knowledge from one
//! domain (e.g., computer vision) to another (e.g., NLP) by maintaining
//! shared representations and computing domain similarity. The transfer
//! mechanism blends source-domain knowledge into the target domain weighted
//! by a transferability score derived from cosine similarity of shared
//! representations.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Domain knowledge
// ---------------------------------------------------------------------------

/// Knowledge captured for a single domain.
///
/// Separates the shared backbone representation (usable across domains) from
/// domain-specific parameters, and tracks performance history for analytics.
#[derive(Debug, Clone)]
pub struct DomainKnowledge<T: Float + Debug + Send + Sync + 'static> {
    /// Human-readable domain name.
    pub domain_name: String,
    /// Features that form the shared representation across domains.
    pub shared_representation: Array1<T>,
    /// Parameters specific to this domain.
    pub domain_specific_params: Array1<T>,
    /// Historical performance values recorded in this domain.
    pub performance_history: Vec<T>,
}

// ---------------------------------------------------------------------------
// Shared representation
// ---------------------------------------------------------------------------

/// Shared backbone features maintained across all registered domains.
#[derive(Debug, Clone)]
pub struct SharedRepresentation<T: Float + Debug + Send + Sync + 'static> {
    /// Shared feature vector.
    pub features: Array1<T>,
    /// Dimensionality of the shared features.
    pub dimension: usize,
    /// Version counter incremented on every update.
    pub version: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> SharedRepresentation<T> {
    /// Create a new zero-initialised shared representation of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            features: Array1::<T>::zeros(dimension),
            dimension,
            version: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer result
// ---------------------------------------------------------------------------

/// Outcome of a cross-domain transfer operation.
#[derive(Debug, Clone)]
pub struct TransferResult<T: Float + Debug + Send + Sync + 'static> {
    /// Parameters produced by the transfer.
    pub transferred_params: Array1<T>,
    /// Scalar in [0, 1] indicating how transferable the source is to the target.
    pub transferability_score: T,
    /// Name of the source domain.
    pub source_domain: String,
    /// Name of the target domain.
    pub target_domain: String,
}

// ---------------------------------------------------------------------------
// Cross-domain transfer engine
// ---------------------------------------------------------------------------

/// Engine for cross-domain knowledge transfer.
///
/// Maintains a registry of domain-specific knowledge and a global shared
/// representation. Supports computing pairwise domain similarities and
/// performing transfer of knowledge from one domain to another.
#[derive(Debug)]
pub struct CrossDomainTransfer<T: Float + Debug + Send + Sync + 'static> {
    /// Registered domains keyed by name.
    domains: HashMap<String, DomainKnowledge<T>>,
    /// Global shared representation.
    shared_repr: SharedRepresentation<T>,
    /// History of transfers performed.
    transfer_history: Vec<TransferResult<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static + ScalarOperand> CrossDomainTransfer<T> {
    /// Create a new engine with a shared representation of the given dimension.
    pub fn new(shared_dim: usize) -> Self {
        Self {
            domains: HashMap::new(),
            shared_repr: SharedRepresentation::new(shared_dim),
            transfer_history: Vec::new(),
        }
    }

    /// Get a reference to the transfer history.
    pub fn transfer_history(&self) -> &[TransferResult<T>] {
        &self.transfer_history
    }

    /// Get a reference to the shared representation.
    pub fn shared_representation(&self) -> &SharedRepresentation<T> {
        &self.shared_repr
    }

    // -----------------------------------------------------------------
    // Domain management
    // -----------------------------------------------------------------

    /// Register a new domain or update an existing one.
    ///
    /// # Errors
    /// Returns `OptimError::InvalidConfig` if the domain's shared
    /// representation dimension does not match the engine's shared dimension.
    pub fn register_domain(&mut self, knowledge: DomainKnowledge<T>) -> Result<()> {
        if knowledge.shared_representation.len() != self.shared_repr.dimension {
            return Err(OptimError::InvalidConfig(format!(
                "Shared representation dimension mismatch: expected {}, got {}",
                self.shared_repr.dimension,
                knowledge.shared_representation.len()
            )));
        }
        self.domains
            .insert(knowledge.domain_name.clone(), knowledge);
        Ok(())
    }

    /// Return the names of all registered domains (sorted for determinism).
    pub fn get_registered_domains(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.domains.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    // -----------------------------------------------------------------
    // Similarity
    // -----------------------------------------------------------------

    /// Compute the cosine similarity between two domains' shared representations.
    ///
    /// The result is in [-1, 1]. If both vectors are zero the similarity is
    /// defined as zero.
    ///
    /// # Errors
    /// Returns `OptimError::InvalidState` if either domain is not registered.
    pub fn compute_domain_similarity(&self, source: &str, target: &str) -> Result<T> {
        let src = self.get_domain(source)?;
        let tgt = self.get_domain(target)?;

        let dot = dot_product(&src.shared_representation, &tgt.shared_representation);
        let norm_src = l2_norm(&src.shared_representation);
        let norm_tgt = l2_norm(&tgt.shared_representation);

        let denom = norm_src * norm_tgt;
        if denom <= T::zero() {
            return Ok(T::zero());
        }
        Ok(dot / denom)
    }

    // -----------------------------------------------------------------
    // Transfer
    // -----------------------------------------------------------------

    /// Transfer knowledge from `source` domain to `target` domain.
    ///
    /// The adapted parameters are computed as:
    /// ```text
    /// transferred = target_specific
    ///             + transferability * (source_shared - target_shared)
    /// ```
    /// where `transferability` is `(cosine_similarity + 1) / 2` (mapped to
    /// [0, 1]).
    ///
    /// # Errors
    /// Returns `OptimError::InvalidState` if either domain is not registered,
    /// or `OptimError::ComputationError` if domain-specific parameter
    /// dimensions differ.
    pub fn transfer(&mut self, source: &str, target: &str) -> Result<TransferResult<T>> {
        // We need to clone to avoid double borrow.
        let src = self.get_domain(source)?.clone();
        let tgt = self.get_domain(target)?.clone();

        if src.domain_specific_params.len() != tgt.domain_specific_params.len() {
            return Err(OptimError::ComputationError(format!(
                "Domain-specific parameter dimension mismatch: source {} vs target {}",
                src.domain_specific_params.len(),
                tgt.domain_specific_params.len()
            )));
        }

        let similarity = self.compute_domain_similarity(source, target)?;
        // Map [-1, 1] -> [0, 1]
        let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());
        let transferability = (similarity + T::one()) / two;

        // Build transferred parameters
        let dim = tgt.domain_specific_params.len();
        let mut transferred = Array1::<T>::zeros(dim);

        // Shared-representation difference (potentially different length from
        // domain_specific_params). We only use it element-wise up to the
        // minimum shared dimension, padding the rest with zero.
        let shared_dim = src
            .shared_representation
            .len()
            .min(tgt.shared_representation.len());
        let mut shared_diff = Array1::<T>::zeros(dim);
        for i in 0..shared_dim.min(dim) {
            shared_diff[i] = src.shared_representation[i] - tgt.shared_representation[i];
        }

        Zip::from(&mut transferred)
            .and(&tgt.domain_specific_params)
            .and(&shared_diff)
            .for_each(|out, &tgt_p, &sd| {
                *out = tgt_p + transferability * sd;
            });

        let result = TransferResult {
            transferred_params: transferred,
            transferability_score: transferability,
            source_domain: source.to_string(),
            target_domain: target.to_string(),
        };

        self.transfer_history.push(result.clone());
        Ok(result)
    }

    // -----------------------------------------------------------------
    // Shared representation update
    // -----------------------------------------------------------------

    /// Update the shared representation using gradients from a specific domain.
    ///
    /// Applies a simple gradient-descent step:
    /// `shared_features -= lr * gradients`
    /// and also updates the domain's own shared-representation snapshot.
    ///
    /// # Errors
    /// Returns `OptimError::InvalidState` if the domain is not registered, or
    /// `OptimError::ComputationError` on dimension mismatch.
    pub fn update_shared_representation(
        &mut self,
        domain_name: &str,
        gradients: &Array1<T>,
        lr: T,
    ) -> Result<()> {
        if gradients.len() != self.shared_repr.dimension {
            return Err(OptimError::ComputationError(format!(
                "Gradient dimension {} does not match shared dimension {}",
                gradients.len(),
                self.shared_repr.dimension
            )));
        }

        // Check domain exists
        if !self.domains.contains_key(domain_name) {
            return Err(OptimError::InvalidState(format!(
                "Domain '{}' is not registered",
                domain_name
            )));
        }

        // Update global shared representation
        Zip::from(&mut self.shared_repr.features)
            .and(gradients)
            .for_each(|f, &g| {
                *f = *f - lr * g;
            });
        self.shared_repr.version += 1;

        // Mirror into the domain's snapshot
        if let Some(domain) = self.domains.get_mut(domain_name) {
            Zip::from(&mut domain.shared_representation)
                .and(&self.shared_repr.features)
                .for_each(|d, &s| {
                    *d = s;
                });
        }

        Ok(())
    }

    // -----------------------------------------------------------------
    // Transferability matrix
    // -----------------------------------------------------------------

    /// Compute an NxN pairwise transferability (cosine-similarity) matrix.
    ///
    /// Rows and columns are ordered by sorted domain name. The diagonal is 1.0
    /// (self-similarity).
    ///
    /// # Errors
    /// Returns `OptimError::InsufficientData` if fewer than two domains are
    /// registered.
    pub fn get_transferability_matrix(&self) -> Result<Array2<T>> {
        let names = self.get_registered_domains();
        let n = names.len();
        if n < 2 {
            return Err(OptimError::InsufficientData(
                "Need at least 2 registered domains to build a transferability matrix".into(),
            ));
        }

        let mut matrix = Array2::<T>::zeros((n, n));
        for (i, &src) in names.iter().enumerate() {
            for (j, &tgt) in names.iter().enumerate() {
                if i == j {
                    matrix[[i, j]] = T::one();
                } else {
                    matrix[[i, j]] = self.compute_domain_similarity(src, tgt)?;
                }
            }
        }
        Ok(matrix)
    }

    // -----------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------

    /// Retrieve a domain by name or return an error.
    fn get_domain(&self, name: &str) -> Result<&DomainKnowledge<T>> {
        self.domains
            .get(name)
            .ok_or_else(|| OptimError::InvalidState(format!("Domain '{}' is not registered", name)))
    }
}

// ---------------------------------------------------------------------------
// Free-standing math helpers
// ---------------------------------------------------------------------------

/// Dot product of two arrays.
fn dot_product<T: Float + Debug + Send + Sync + 'static>(a: &Array1<T>, b: &Array1<T>) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// L2 norm of an array.
fn l2_norm<T: Float + Debug + Send + Sync + 'static>(arr: &Array1<T>) -> T {
    arr.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_domain(name: &str, shared_dim: usize, specific_dim: usize) -> DomainKnowledge<f64> {
        let shared = Array1::from_vec(
            (0..shared_dim)
                .map(|i| (i as f64 + 1.0) * if name.contains("nlp") { -1.0 } else { 1.0 })
                .collect(),
        );
        let specific = Array1::from_elem(specific_dim, 0.5);
        DomainKnowledge {
            domain_name: name.to_string(),
            shared_representation: shared,
            domain_specific_params: specific,
            performance_history: vec![0.8, 0.85, 0.9],
        }
    }

    #[test]
    fn test_register_domain() {
        let mut engine = CrossDomainTransfer::<f64>::new(4);
        let domain = make_domain("cv", 4, 8);
        assert!(engine.register_domain(domain).is_ok());
        assert_eq!(engine.get_registered_domains(), vec!["cv"]);

        // Dimension mismatch should fail.
        let bad = make_domain("bad", 3, 8);
        assert!(engine.register_domain(bad).is_err());
    }

    #[test]
    fn test_compute_domain_similarity() {
        let mut engine = CrossDomainTransfer::<f64>::new(4);
        engine
            .register_domain(make_domain("cv", 4, 8))
            .expect("register cv");
        engine
            .register_domain(make_domain("nlp", 4, 8))
            .expect("register nlp");

        let sim = engine
            .compute_domain_similarity("cv", "nlp")
            .expect("similarity");
        // cv shared = [1,2,3,4], nlp shared = [-1,-2,-3,-4] => cosine = -1
        assert!(
            (sim - (-1.0)).abs() < 1e-10,
            "Expected -1.0 cosine similarity, got {}",
            sim
        );

        // Self-similarity should be 1.
        let self_sim = engine
            .compute_domain_similarity("cv", "cv")
            .expect("self similarity");
        assert!(
            (self_sim - 1.0).abs() < 1e-10,
            "Expected 1.0, got {}",
            self_sim
        );

        // Unknown domain should error.
        assert!(engine.compute_domain_similarity("cv", "rl").is_err());
    }

    #[test]
    fn test_transfer_knowledge() {
        let mut engine = CrossDomainTransfer::<f64>::new(4);

        // Two similar domains (same sign shared repr)
        let cv = DomainKnowledge {
            domain_name: "cv".to_string(),
            shared_representation: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            domain_specific_params: Array1::from_elem(4, 1.0),
            performance_history: vec![0.9],
        };
        let cv2 = DomainKnowledge {
            domain_name: "cv2".to_string(),
            shared_representation: Array1::from_vec(vec![1.1, 2.1, 3.1, 4.1]),
            domain_specific_params: Array1::from_elem(4, 0.5),
            performance_history: vec![0.7],
        };
        engine.register_domain(cv).expect("register cv");
        engine.register_domain(cv2).expect("register cv2");

        let result = engine.transfer("cv", "cv2").expect("transfer");
        assert_eq!(result.source_domain, "cv");
        assert_eq!(result.target_domain, "cv2");
        assert!(
            result.transferability_score > 0.9,
            "Similar domains should have high transferability, got {}",
            result.transferability_score
        );
        assert_eq!(result.transferred_params.len(), 4);
        assert_eq!(engine.transfer_history().len(), 1);
    }

    #[test]
    fn test_update_shared_representation() {
        let mut engine = CrossDomainTransfer::<f64>::new(4);
        let domain = DomainKnowledge {
            domain_name: "cv".to_string(),
            shared_representation: Array1::zeros(4),
            domain_specific_params: Array1::zeros(4),
            performance_history: vec![],
        };
        engine.register_domain(domain).expect("register");

        let grads = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        engine
            .update_shared_representation("cv", &grads, 0.1)
            .expect("update");

        // shared = 0 - 0.1 * [1,2,3,4] = [-0.1, -0.2, -0.3, -0.4]
        let shared = &engine.shared_representation().features;
        assert!((shared[0] - (-0.1)).abs() < 1e-10);
        assert!((shared[3] - (-0.4)).abs() < 1e-10);
        assert_eq!(engine.shared_representation().version, 1);

        // Unknown domain should error.
        assert!(engine
            .update_shared_representation("rl", &grads, 0.1)
            .is_err());

        // Dimension mismatch should error.
        let bad_grads = Array1::from_vec(vec![1.0, 2.0]);
        assert!(engine
            .update_shared_representation("cv", &bad_grads, 0.1)
            .is_err());
    }

    #[test]
    fn test_transferability_matrix() {
        let mut engine = CrossDomainTransfer::<f64>::new(3);

        // Need at least 2 domains.
        assert!(engine.get_transferability_matrix().is_err());

        let d1 = DomainKnowledge {
            domain_name: "a".to_string(),
            shared_representation: Array1::from_vec(vec![1.0, 0.0, 0.0]),
            domain_specific_params: Array1::zeros(2),
            performance_history: vec![],
        };
        let d2 = DomainKnowledge {
            domain_name: "b".to_string(),
            shared_representation: Array1::from_vec(vec![0.0, 1.0, 0.0]),
            domain_specific_params: Array1::zeros(2),
            performance_history: vec![],
        };
        let d3 = DomainKnowledge {
            domain_name: "c".to_string(),
            shared_representation: Array1::from_vec(vec![1.0, 1.0, 0.0]),
            domain_specific_params: Array1::zeros(2),
            performance_history: vec![],
        };
        engine.register_domain(d1).expect("register a");
        engine.register_domain(d2).expect("register b");
        engine.register_domain(d3).expect("register c");

        let matrix = engine.get_transferability_matrix().expect("matrix");
        assert_eq!(matrix.shape(), &[3, 3]);

        // Diagonal should be 1.
        for i in 0..3 {
            assert!(
                (matrix[[i, i]] - 1.0).abs() < 1e-10,
                "Diagonal [{},{}] should be 1.0, got {}",
                i,
                i,
                matrix[[i, i]]
            );
        }

        // a and b are orthogonal => similarity 0.
        assert!(
            matrix[[0, 1]].abs() < 1e-10,
            "Orthogonal domains should have 0 similarity, got {}",
            matrix[[0, 1]]
        );

        // Matrix should be symmetric.
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (matrix[[i, j]] - matrix[[j, i]]).abs() < 1e-10,
                    "Matrix should be symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }
}
