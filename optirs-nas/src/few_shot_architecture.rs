//! Few-shot architecture optimization for Neural Architecture Search.
//!
//! When a new target task is encountered, full retraining or thousands of
//! architecture evaluations are typically prohibitive. *Few-shot* methods
//! sidestep that cost by transferring knowledge from a small *support set*
//! of `(embedding, performance)` pairs — typically drawn from related tasks
//! through [`ArchitectureKnowledgeGraph`] — to a *query* architecture whose
//! performance we wish to predict.
//!
//! This module implements four widely-used few-shot algorithms specialised
//! for the architecture-search regression setting:
//!
//! 1. **Prototypical Networks** — partitions the support set into
//!    performance-quantile buckets, computes the mean embedding of each
//!    bucket as a *prototype*, and predicts the bucket whose prototype is
//!    closest to the query. The bucket's mean performance is returned as
//!    the prediction and the softmax over negative distances provides the
//!    confidence.
//! 2. **Matching Networks** — attention-weighted average of *every* support
//!    label, where the weights come from a temperature-scaled softmax over
//!    negative distances. Entropy of the weight distribution yields a
//!    normalised confidence in `[0, 1]`.
//! 3. **MAML-style linear adapter** — a one-layer linear regressor is
//!    fitted by a handful of inner-loop gradient descent steps on the
//!    support set. Predictions are simply `w · embedding + b`.
//! 4. **Distance-weighted KNN** — classical inverse-distance regression
//!    over the `k_shot` closest support points.
//!
//! All four share the same configuration struct, the same predict / batch
//! / `recommend_top_k` interface, and the same `(arch_id, distance)` style
//! diagnostic output, so callers can swap algorithms without touching the
//! surrounding code.
//!
//! # Examples
//!
//! ```
//! use optirs_nas::few_shot_architecture::{
//!     ArchitectureExample, FewShotAlgorithm, FewShotArchitectureOptimizer,
//! };
//!
//! let support = vec![
//!     ArchitectureExample {
//!         arch_id: "resnet".into(),
//!         embedding: vec![1.0, 0.0, 0.0],
//!         performance: 0.92,
//!         domain: "vision".into(),
//!     },
//!     ArchitectureExample {
//!         arch_id: "mobilenet".into(),
//!         embedding: vec![0.0, 1.0, 0.0],
//!         performance: 0.81,
//!         domain: "vision".into(),
//!     },
//! ];
//!
//! let mut optimizer = FewShotArchitectureOptimizer::new()
//!     .with_algorithm(FewShotAlgorithm::Matching)
//!     .with_k_shot(2);
//! optimizer.fit_from_examples(support).expect("fit");
//! let prediction = optimizer.predict(&[0.9, 0.1, 0.0]).expect("predict");
//! ```
//!
//! # Mathematical notes
//!
//! - **Euclidean**: `d(a, b) = sqrt(Σ_i (a_i - b_i)^2)`
//! - **Cosine**: `d(a, b) = 1 - (a · b) / (‖a‖‖b‖)` clipped to `[0, 2]`
//! - **Manhattan**: `d(a, b) = Σ_i |a_i - b_i|`
//! - **Softmax(temperature T)**: `a_i = exp(-d_i / T) / Σ_j exp(-d_j / T)`
//! - **Entropy**: `H(a) = -Σ_i a_i log(a_i)` with the convention `0 log 0 = 0`
//! - **Confidence from entropy**: `1 - H(a) / log(n)` so that a uniform
//!   distribution gives `0` (no information) and a one-hot gives `1`.

use crate::architecture_knowledge_graph::{ArchitectureKnowledgeGraph, NodeId, PerformanceRecord};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};

/// A single labelled architecture example used as part of the support set.
///
/// Each example pairs a dense embedding with a scalar performance score.
/// The `domain` field is informational — it lets the optimizer carry
/// provenance information into [`FewShotPrediction::nearest_support_arch`]
/// so callers can audit which domain a recommendation came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureExample {
    /// External / human-readable architecture identifier.
    pub arch_id: String,
    /// Dense embedding vector. Must have the same length across the entire
    /// support set; mismatched lengths cause [`FewShotArchitectureOptimizer::fit_from_examples`]
    /// to reject the input.
    pub embedding: Vec<f64>,
    /// Scalar performance target (typically accuracy in `[0, 1]`, but any
    /// finite real value is accepted).
    pub performance: f64,
    /// Free-form domain tag (e.g. `"vision"`, `"nlp"`).
    pub domain: String,
}

/// Few-shot learning algorithm to apply when predicting a query.
///
/// All four variants share the same `fit`/`predict` surface so callers can
/// swap algorithms without touching surrounding code. See the module-level
/// documentation for a brief overview of each algorithm's semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FewShotAlgorithm {
    /// Prototypical Networks — class prototypes via the mean embedding of
    /// each performance-quantile bucket.
    Prototypical,
    /// Matching Networks — attention-weighted nearest neighbours over the
    /// entire support set.
    Matching,
    /// MAML-style fast adaptation on a linear adapter (one weight per
    /// embedding dimension plus a bias).
    MamlAdaptation,
    /// Distance-weighted KNN regression over the `k_shot` closest support
    /// points.
    DistanceWeightedKnn,
}

/// Distance metric used when comparing embeddings.
///
/// All metrics return non-negative scalars: `Euclidean` and `Manhattan`
/// are unbounded, while `Cosine` is bounded above by `2` (1 minus a
/// correlation of `-1`). The optimizer uses negative-distance softmax
/// to convert these into attention weights, so any metric can be combined
/// with any algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// `sqrt(Σ_i (a_i - b_i)^2)`.
    Euclidean,
    /// `1 - cos(a, b)`, clipped to `[0, 2]`.
    Cosine,
    /// `Σ_i |a_i - b_i|`.
    Manhattan,
}

/// Configuration controlling every aspect of [`FewShotArchitectureOptimizer`].
///
/// The defaults — Prototypical Networks with `k_shot = 5`, `n_way = 2`,
/// temperature `1.0`, MAML inner-loop learning rate `0.01`, and Euclidean
/// distance — give sensible behaviour on most small support sets. The
/// `seed` field is honoured by the rare stochastic paths (currently only
/// the unit-test scaffolding) so that runs are reproducible.
#[derive(Debug, Clone)]
pub struct FewShotConfig {
    /// Few-shot algorithm to run when predicting.
    pub algorithm: FewShotAlgorithm,
    /// Number of support examples per query. Acts as the *k* in KNN and
    /// the minimum support-set size accepted by `fit_from_examples`.
    pub k_shot: usize,
    /// Number of performance buckets used by Prototypical Networks. Must
    /// be at least 1.
    pub n_way: usize,
    /// Softmax temperature for Matching Networks. Lower temperatures
    /// produce sharper distributions; higher temperatures produce
    /// closer-to-uniform distributions.
    pub temperature: f64,
    /// Inner-loop learning rate for the MAML linear adapter.
    pub adapter_lr: f64,
    /// Number of inner-loop gradient steps for the MAML adapter.
    pub adapter_steps: usize,
    /// Distance metric used by every algorithm that needs one.
    pub distance_metric: DistanceMetric,
    /// Deterministic seed for stochastic components.
    pub seed: u64,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            algorithm: FewShotAlgorithm::Prototypical,
            k_shot: 5,
            n_way: 2,
            temperature: 1.0,
            adapter_lr: 0.01,
            adapter_steps: 5,
            distance_metric: DistanceMetric::Euclidean,
            seed: 42,
        }
    }
}

/// Single-query prediction returned by [`FewShotArchitectureOptimizer::predict`].
///
/// The `confidence` field is in `[0, 1]` and is algorithm-specific — see
/// the algorithm-by-algorithm notes in the module-level documentation. The
/// `nearest_support_arch` field is always set unless the support set is
/// empty (which can never happen on a fitted optimizer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotPrediction {
    /// Predicted scalar performance.
    pub predicted_performance: f64,
    /// Algorithm-specific confidence in `[0, 1]`.
    pub confidence: f64,
    /// Identifier of the closest support example, if any.
    pub nearest_support_arch: Option<String>,
    /// Distance from the query to that closest support example, under the
    /// configured metric.
    pub nearest_distance: f64,
}

/// The few-shot architecture optimizer.
///
/// `FewShotArchitectureOptimizer` owns the support set, the configuration,
/// and (for the `MamlAdaptation` algorithm) the fitted adapter weights.
/// Calling [`reset`](Self::reset) wipes all of that state and returns the
/// optimizer to a freshly-constructed configuration.
#[derive(Debug, Clone)]
pub struct FewShotArchitectureOptimizer {
    config: FewShotConfig,
    support_set: Vec<ArchitectureExample>,
    is_fitted: bool,
    /// MAML linear adapter weights, one per embedding dimension. Populated
    /// only when `config.algorithm == FewShotAlgorithm::MamlAdaptation`.
    adapter_weights: Option<Vec<f64>>,
    /// MAML adapter scalar bias.
    adapter_bias: Option<f64>,
}

impl Default for FewShotArchitectureOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FewShotArchitectureOptimizer {
    /// Construct a new optimizer using [`FewShotConfig::default`].
    pub fn new() -> Self {
        Self::with_config(FewShotConfig::default())
    }

    /// Construct a new optimizer with an explicit configuration.
    pub fn with_config(config: FewShotConfig) -> Self {
        Self {
            config,
            support_set: Vec::new(),
            is_fitted: false,
            adapter_weights: None,
            adapter_bias: None,
        }
    }

    /// Replace the few-shot algorithm.
    pub fn with_algorithm(mut self, algorithm: FewShotAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Replace the `k_shot` (minimum support-set size).
    pub fn with_k_shot(mut self, k: usize) -> Self {
        self.config.k_shot = k;
        self
    }

    /// Replace the distance metric.
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Replace the softmax temperature used by Matching Networks.
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.config.temperature = t;
        self
    }

    /// Replace the deterministic seed used for stochastic components.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Borrow the current support set.
    pub fn support_set(&self) -> &[ArchitectureExample] {
        &self.support_set
    }

    /// Whether [`fit_from_examples`](Self::fit_from_examples) (or one of its
    /// shortcuts) has been called successfully.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Borrow the current configuration.
    pub fn config(&self) -> &FewShotConfig {
        &self.config
    }

    /// Clear the support set, adapter weights, and "fitted" flag.
    ///
    /// The configuration is preserved, so this is equivalent to constructing
    /// a fresh optimizer with the same configuration.
    pub fn reset(&mut self) {
        self.support_set.clear();
        self.is_fitted = false;
        self.adapter_weights = None;
        self.adapter_bias = None;
    }

    /// Fit the optimizer to a concrete support set.
    ///
    /// Validates that:
    /// - the support set contains at least `config.k_shot` examples, and
    /// - all embeddings share the same dimensionality, and
    /// - all performance and embedding values are finite.
    ///
    /// For `MamlAdaptation` the inner-loop training is run here so that
    /// subsequent `predict` calls are inexpensive.
    pub fn fit_from_examples(&mut self, examples: Vec<ArchitectureExample>) -> Result<()> {
        if examples.len() < self.config.k_shot {
            return Err(OptimError::InvalidParameter(format!(
                "fit_from_examples: need at least k_shot = {} examples, got {}",
                self.config.k_shot,
                examples.len()
            )));
        }
        if examples.is_empty() {
            return Err(OptimError::InvalidParameter(
                "fit_from_examples: support set is empty".to_string(),
            ));
        }

        // Check that every embedding has the same dimensionality and finite
        // values. We reject NaN/inf eagerly to prevent silent corruption of
        // downstream predictions.
        let embedding_dim = examples[0].embedding.len();
        if embedding_dim == 0 {
            return Err(OptimError::InvalidParameter(
                "fit_from_examples: embeddings must be non-empty".to_string(),
            ));
        }
        for (i, ex) in examples.iter().enumerate() {
            if ex.embedding.len() != embedding_dim {
                return Err(OptimError::InvalidParameter(format!(
                    "fit_from_examples: example {} has embedding dim {}, expected {}",
                    i,
                    ex.embedding.len(),
                    embedding_dim
                )));
            }
            if !ex.performance.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "fit_from_examples: example {} performance is not finite: {}",
                    i, ex.performance
                )));
            }
            if ex.embedding.iter().any(|v| !v.is_finite()) {
                return Err(OptimError::InvalidParameter(format!(
                    "fit_from_examples: example {} has non-finite embedding entries",
                    i
                )));
            }
        }

        if self.config.n_way == 0 {
            return Err(OptimError::InvalidParameter(
                "fit_from_examples: n_way must be at least 1".to_string(),
            ));
        }

        self.support_set = examples;

        // Train the MAML adapter eagerly. For every other algorithm the
        // adapter is left unset, so a stale adapter from a previous fit
        // does not bleed into the new predictions.
        if matches!(self.config.algorithm, FewShotAlgorithm::MamlAdaptation) {
            self.train_maml_adapter()?;
        } else {
            self.adapter_weights = None;
            self.adapter_bias = None;
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Fit from the top-`k` architectures of `source_domain` inside `graph`.
    ///
    /// Architectures without a [`PerformanceRecord`] are skipped. The
    /// remaining nodes are sorted descending by `accuracy` and the top
    /// `k` are used as the support set. If `k` is smaller than
    /// `config.k_shot`, fitting still fails because the underlying
    /// `fit_from_examples` call enforces the lower bound.
    pub fn fit_from_graph(
        &mut self,
        graph: &ArchitectureKnowledgeGraph,
        source_domain: &str,
        k: usize,
    ) -> Result<()> {
        if k == 0 {
            return Err(OptimError::InvalidParameter(
                "fit_from_graph: k must be greater than 0".to_string(),
            ));
        }

        // Gather all (node, performance) candidates from the source domain
        // that have a recorded performance. Cloning a `PerformanceRecord`
        // is cheap (small struct).
        let mut candidates: Vec<(NodeId, PerformanceRecord)> = graph
            .nodes()
            .iter()
            .filter(|n| n.domain == source_domain)
            .filter_map(|n| n.performance.as_ref().map(|p| (n.id, p.clone())))
            .collect();

        if candidates.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "fit_from_graph: no nodes with performance records in domain '{}'",
                source_domain
            )));
        }

        candidates.sort_by(|a, b| {
            b.1.accuracy
                .partial_cmp(&a.1.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);

        // Build `ArchitectureExample`s by copying the node embeddings out
        // of the graph. We use `node(...)` to look up by `NodeId` rather
        // than `nodes()[idx]` so that the API survives any future graph
        // re-indexing.
        let mut examples: Vec<ArchitectureExample> = Vec::with_capacity(candidates.len());
        for (node_id, record) in candidates {
            let node = graph.node(node_id).ok_or_else(|| {
                OptimError::ArchitectureError(format!(
                    "fit_from_graph: node {} disappeared while iterating",
                    node_id
                ))
            })?;
            examples.push(ArchitectureExample {
                arch_id: node.arch_id.clone(),
                embedding: node.embedding.clone(),
                performance: record.accuracy,
                domain: node.domain.clone(),
            });
        }

        self.fit_from_examples(examples)
    }

    /// Predict the scalar performance of `query_embedding`.
    ///
    /// Dispatches to the algorithm selected in [`FewShotConfig::algorithm`].
    pub fn predict(&self, query_embedding: &[f64]) -> Result<FewShotPrediction> {
        self.check_fitted()?;
        if query_embedding.is_empty() {
            return Err(OptimError::InvalidParameter(
                "predict: query embedding must be non-empty".to_string(),
            ));
        }
        let expected_dim = self.support_set[0].embedding.len();
        if query_embedding.len() != expected_dim {
            return Err(OptimError::InvalidParameter(format!(
                "predict: query has {} dims, support uses {}",
                query_embedding.len(),
                expected_dim
            )));
        }
        if query_embedding.iter().any(|v| !v.is_finite()) {
            return Err(OptimError::InvalidParameter(
                "predict: query embedding contains non-finite entries".to_string(),
            ));
        }

        match self.config.algorithm {
            FewShotAlgorithm::Prototypical => self.predict_prototypical(query_embedding),
            FewShotAlgorithm::Matching => self.predict_matching(query_embedding),
            FewShotAlgorithm::MamlAdaptation => self.predict_maml(query_embedding),
            FewShotAlgorithm::DistanceWeightedKnn => self.predict_knn(query_embedding),
        }
    }

    /// Batch wrapper that calls [`predict`](Self::predict) on each query.
    ///
    /// The errors propagated by individual queries are returned as soon as
    /// they occur — there is no partial-result mode.
    pub fn predict_batch(&self, queries: &[Vec<f64>]) -> Result<Vec<FewShotPrediction>> {
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            out.push(self.predict(q)?);
        }
        Ok(out)
    }

    /// Return the `top_k` candidates by predicted performance.
    ///
    /// Each candidate is supplied as a `(arch_id, embedding)` pair. The
    /// result is sorted descending by `predicted_performance`. Candidates
    /// whose predictions error out cause the whole call to fail.
    pub fn recommend_top_k(
        &self,
        candidates: &[(&str, Vec<f64>)],
        top_k: usize,
    ) -> Result<Vec<(String, FewShotPrediction)>> {
        self.check_fitted()?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let mut scored: Vec<(String, FewShotPrediction)> = Vec::with_capacity(candidates.len());
        for (arch_id, emb) in candidates {
            let prediction = self.predict(emb)?;
            scored.push(((*arch_id).to_string(), prediction));
        }
        scored.sort_by(|a, b| {
            b.1.predicted_performance
                .partial_cmp(&a.1.predicted_performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    // ---- internal helpers -------------------------------------------------

    /// Ensure the optimizer has been fitted; otherwise return
    /// [`OptimError::InvalidParameter`] (used in lieu of `InvalidState`,
    /// which optirs-nas does not have).
    fn check_fitted(&self) -> Result<()> {
        if !self.is_fitted || self.support_set.is_empty() {
            return Err(OptimError::InvalidParameter(
                "not fitted: call fit_from_examples or fit_from_graph first".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the configured distance between two equally-sized embeddings.
    ///
    /// Returns [`OptimError::InvalidParameter`] for mismatched lengths and
    /// non-finite distances. Cosine distance is clipped to `[0, 2]` to
    /// match the mathematical definition; zero-norm vectors return the
    /// neutral distance `1.0` (orthogonal).
    fn distance(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(OptimError::InvalidParameter(format!(
                "distance: length mismatch {} vs {}",
                a.len(),
                b.len()
            )));
        }
        let d = match self.config.distance_metric {
            DistanceMetric::Euclidean => {
                let mut s = 0.0;
                for (x, y) in a.iter().zip(b.iter()) {
                    let diff = x - y;
                    s += diff * diff;
                }
                s.sqrt()
            }
            DistanceMetric::Manhattan => {
                let mut s = 0.0;
                for (x, y) in a.iter().zip(b.iter()) {
                    s += (x - y).abs();
                }
                s
            }
            DistanceMetric::Cosine => {
                let mut dot = 0.0;
                let mut norm_a = 0.0;
                let mut norm_b = 0.0;
                for (x, y) in a.iter().zip(b.iter()) {
                    dot += x * y;
                    norm_a += x * x;
                    norm_b += y * y;
                }
                if norm_a <= 0.0 || norm_b <= 0.0 {
                    // Treat zero-norm as orthogonal to avoid NaN downstream.
                    1.0
                } else {
                    let cos = (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0);
                    (1.0 - cos).clamp(0.0, 2.0)
                }
            }
        };
        if !d.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "distance: produced non-finite result {}",
                d
            )));
        }
        Ok(d)
    }

    /// Find the (index, distance) of the support example closest to `query`.
    fn nearest_support(&self, query: &[f64]) -> Result<(usize, f64)> {
        let mut best_idx = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, ex) in self.support_set.iter().enumerate() {
            let d = self.distance(query, &ex.embedding)?;
            if d < best_d {
                best_d = d;
                best_idx = i;
            }
        }
        Ok((best_idx, best_d))
    }

    /// Numerically stable softmax over `-logits / temperature`.
    ///
    /// The standard `softmax = exp(x_i) / Σ exp(x_j)` formula overflows for
    /// large negative `-d / T`. Subtracting the maximum logit before the
    /// exponentiation gives the same answer without overflow, and the
    /// constant `T` is folded into the scaling step.
    fn softmax_neg(distances: &[f64], temperature: f64) -> Vec<f64> {
        if distances.is_empty() {
            return Vec::new();
        }
        let t = if temperature.is_finite() && temperature > 0.0 {
            temperature
        } else {
            1.0
        };
        // logits_i = -d_i / t — find max for stability.
        let mut max_logit = f64::NEG_INFINITY;
        for d in distances {
            let l = -d / t;
            if l > max_logit {
                max_logit = l;
            }
        }
        if !max_logit.is_finite() {
            // All logits were -inf (shouldn't happen with finite distances,
            // but defensively return a uniform distribution).
            let u = 1.0 / distances.len() as f64;
            return vec![u; distances.len()];
        }
        let exps: Vec<f64> = distances
            .iter()
            .map(|d| ((-d / t) - max_logit).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            let u = 1.0 / distances.len() as f64;
            return vec![u; distances.len()];
        }
        exps.into_iter().map(|e| e / sum).collect()
    }

    /// Entropy `H(p) = -Σ p_i log(p_i)` with the convention `0 log 0 = 0`.
    fn entropy(p: &[f64]) -> f64 {
        let mut h = 0.0;
        for &pi in p {
            if pi > 0.0 {
                h -= pi * pi.ln();
            }
        }
        h
    }

    // ---- Prototypical Networks ------------------------------------------

    /// Bucket the support set into `n_way` equal-width performance buckets
    /// and return, for each non-empty bucket, the mean embedding (as the
    /// prototype) and the mean performance (as the prediction). Buckets
    /// are derived by *equal-width binning of values*: we find `[min, max]`
    /// over performances and split the range into `n_way` chunks. The
    /// alternative (equal-frequency binning) often collapses repeated
    /// scores into the same bin, which is undesirable when the support set
    /// is small.
    fn prototype_buckets(&self) -> Vec<(Vec<f64>, f64)> {
        let n_way = self.config.n_way.max(1);
        let min_perf = self
            .support_set
            .iter()
            .map(|e| e.performance)
            .fold(f64::INFINITY, f64::min);
        let max_perf = self
            .support_set
            .iter()
            .map(|e| e.performance)
            .fold(f64::NEG_INFINITY, f64::max);
        let embedding_dim = self.support_set[0].embedding.len();

        // Degenerate case: all performances equal. Collapse to a single
        // bucket containing every example.
        if (max_perf - min_perf).abs() < 1e-12 || n_way == 1 {
            let mut proto = vec![0.0; embedding_dim];
            for ex in &self.support_set {
                for (slot, v) in proto.iter_mut().zip(ex.embedding.iter()) {
                    *slot += v;
                }
            }
            for slot in proto.iter_mut() {
                *slot /= self.support_set.len() as f64;
            }
            return vec![(proto, min_perf)];
        }

        // Bin width over the closed [min, max] range, with a tiny epsilon
        // pushed into the max bin so that `max_perf` itself lands in the
        // last bucket rather than `n_way` (which would be out of range).
        let width = (max_perf - min_perf) / n_way as f64;
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; embedding_dim]; n_way];
        let mut perf_sums: Vec<f64> = vec![0.0; n_way];
        let mut counts: Vec<usize> = vec![0; n_way];

        for ex in &self.support_set {
            let mut bin = ((ex.performance - min_perf) / width).floor() as isize;
            if bin < 0 {
                bin = 0;
            }
            if bin >= n_way as isize {
                bin = (n_way - 1) as isize;
            }
            let b = bin as usize;
            for (slot, v) in sums[b].iter_mut().zip(ex.embedding.iter()) {
                *slot += v;
            }
            perf_sums[b] += ex.performance;
            counts[b] += 1;
        }

        let mut buckets = Vec::new();
        for b in 0..n_way {
            if counts[b] == 0 {
                continue;
            }
            let c = counts[b] as f64;
            let proto: Vec<f64> = sums[b].iter().map(|v| v / c).collect();
            buckets.push((proto, perf_sums[b] / c));
        }
        buckets
    }

    fn predict_prototypical(&self, query: &[f64]) -> Result<FewShotPrediction> {
        let buckets = self.prototype_buckets();
        if buckets.is_empty() {
            return Err(OptimError::ConvergenceFailure(
                "predict_prototypical: no non-empty prototype buckets".to_string(),
            ));
        }
        let distances: Vec<f64> = buckets
            .iter()
            .map(|(proto, _)| self.distance(query, proto))
            .collect::<Result<Vec<_>>>()?;
        // Pick the nearest bucket as the prediction.
        let mut best_idx = 0;
        let mut best_d = f64::INFINITY;
        for (i, &d) in distances.iter().enumerate() {
            if d < best_d {
                best_d = d;
                best_idx = i;
            }
        }
        let weights = Self::softmax_neg(&distances, self.config.temperature);
        let confidence = weights
            .get(best_idx)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let (nearest_idx, nearest_d) = self.nearest_support(query)?;
        let nearest = &self.support_set[nearest_idx];

        Ok(FewShotPrediction {
            predicted_performance: buckets[best_idx].1,
            confidence,
            nearest_support_arch: Some(nearest.arch_id.clone()),
            nearest_distance: nearest_d,
        })
    }

    // ---- Matching Networks ----------------------------------------------

    fn predict_matching(&self, query: &[f64]) -> Result<FewShotPrediction> {
        let distances: Vec<f64> = self
            .support_set
            .iter()
            .map(|ex| self.distance(query, &ex.embedding))
            .collect::<Result<Vec<_>>>()?;
        let weights = Self::softmax_neg(&distances, self.config.temperature);
        let predicted: f64 = weights
            .iter()
            .zip(self.support_set.iter())
            .map(|(w, ex)| w * ex.performance)
            .sum();

        // Confidence: `1 - H(p) / log(n)` so a uniform distribution gives 0
        // (no information) and a one-hot distribution gives 1 (full
        // information). With n = 1 we are by definition fully confident.
        let n = weights.len();
        let confidence = if n <= 1 {
            1.0
        } else {
            let h = Self::entropy(&weights);
            let log_n = (n as f64).ln();
            if log_n > 0.0 {
                (1.0 - h / log_n).clamp(0.0, 1.0)
            } else {
                1.0
            }
        };

        let (nearest_idx, nearest_d) = self.nearest_support(query)?;
        let nearest = &self.support_set[nearest_idx];

        Ok(FewShotPrediction {
            predicted_performance: predicted,
            confidence,
            nearest_support_arch: Some(nearest.arch_id.clone()),
            nearest_distance: nearest_d,
        })
    }

    // ---- MAML linear adapter --------------------------------------------

    /// Fit a one-layer linear regressor `y_hat = w · x + b` on the support
    /// set using `config.adapter_steps` of full-batch gradient descent.
    ///
    /// The loss is the mean squared error
    /// `L = (1/n) Σ_i (w · x_i + b - y_i)^2` whose gradient with respect
    /// to `w` is `(2/n) Σ_i (w · x_i + b - y_i) · x_i` and with respect to
    /// `b` is `(2/n) Σ_i (w · x_i + b - y_i)`. We initialise to zero so
    /// the first iteration's prediction is purely a constant.
    fn train_maml_adapter(&mut self) -> Result<()> {
        let dim = self.support_set[0].embedding.len();
        let n = self.support_set.len() as f64;
        let lr = if self.config.adapter_lr.is_finite() && self.config.adapter_lr > 0.0 {
            self.config.adapter_lr
        } else {
            1e-2
        };
        let steps = self.config.adapter_steps.max(1);

        // Seed the random generator for any future stochastic extensions.
        // Currently the inner loop is deterministic; the `_rng` binding is
        // retained so that mini-batch variants can drop in later without
        // changing the signature.
        let _rng = Random::seed(self.config.seed);

        let mut w: Array1<f64> = Array1::zeros(dim);
        let mut b = 0.0_f64;

        for _ in 0..steps {
            let mut grad_w: Array1<f64> = Array1::zeros(dim);
            let mut grad_b = 0.0_f64;
            for ex in &self.support_set {
                let x = Array1::from(ex.embedding.clone());
                let y_hat = w.dot(&x) + b;
                let err = y_hat - ex.performance;
                // grad_w += 2 * err * x (averaged outside the loop).
                for i in 0..dim {
                    grad_w[i] += 2.0 * err * x[i];
                }
                grad_b += 2.0 * err;
            }
            for i in 0..dim {
                grad_w[i] /= n;
            }
            grad_b /= n;
            for i in 0..dim {
                w[i] -= lr * grad_w[i];
            }
            b -= lr * grad_b;
        }

        self.adapter_weights = Some(w.to_vec());
        self.adapter_bias = Some(b);
        Ok(())
    }

    fn predict_maml(&self, query: &[f64]) -> Result<FewShotPrediction> {
        let w = self.adapter_weights.as_ref().ok_or_else(|| {
            OptimError::InvalidParameter(
                "predict_maml: adapter not trained — refit with FewShotAlgorithm::MamlAdaptation"
                    .to_string(),
            )
        })?;
        let b = self.adapter_bias.unwrap_or(0.0);
        if w.len() != query.len() {
            return Err(OptimError::InvalidParameter(format!(
                "predict_maml: adapter dim {} vs query dim {}",
                w.len(),
                query.len()
            )));
        }
        let predicted: f64 = w
            .iter()
            .zip(query.iter())
            .map(|(wi, xi)| wi * xi)
            .sum::<f64>()
            + b;

        // Confidence: scale by distance to the nearest support example.
        // We map `d -> exp(-d / mean_d)` so a query that exactly coincides
        // with a support example gets confidence ≈ 1, and far-away queries
        // approach 0.
        let (nearest_idx, nearest_d) = self.nearest_support(query)?;
        let all_d: Vec<f64> = self
            .support_set
            .iter()
            .map(|ex| self.distance(query, &ex.embedding))
            .collect::<Result<Vec<_>>>()?;
        let mean_d = if all_d.is_empty() {
            1.0
        } else {
            let s: f64 = all_d.iter().sum();
            (s / all_d.len() as f64).max(1e-12)
        };
        let confidence = ((-nearest_d / mean_d).exp()).clamp(0.0, 1.0);

        let nearest = &self.support_set[nearest_idx];
        Ok(FewShotPrediction {
            predicted_performance: predicted,
            confidence,
            nearest_support_arch: Some(nearest.arch_id.clone()),
            nearest_distance: nearest_d,
        })
    }

    // ---- Distance-weighted KNN ------------------------------------------

    fn predict_knn(&self, query: &[f64]) -> Result<FewShotPrediction> {
        let mut distances: Vec<(usize, f64)> = Vec::with_capacity(self.support_set.len());
        for (i, ex) in self.support_set.iter().enumerate() {
            let d = self.distance(query, &ex.embedding)?;
            distances.push((i, d));
        }
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.config.k_shot.min(distances.len()).max(1);
        let neighbours = &distances[..k];

        // If the closest neighbour is essentially identical, just return
        // its performance directly. This avoids `1 / (d + eps)` numerical
        // weirdness from amplifying tiny-distance noise into the result.
        if neighbours[0].1 < 1e-12 {
            let nearest = &self.support_set[neighbours[0].0];
            return Ok(FewShotPrediction {
                predicted_performance: nearest.performance,
                confidence: 1.0,
                nearest_support_arch: Some(nearest.arch_id.clone()),
                nearest_distance: neighbours[0].1,
            });
        }

        let eps = 1e-9;
        let weights: Vec<f64> = neighbours.iter().map(|(_, d)| 1.0 / (d + eps)).collect();
        let weight_sum: f64 = weights.iter().sum();
        let predicted: f64 = if weight_sum > 0.0 {
            neighbours
                .iter()
                .zip(weights.iter())
                .map(|((idx, _), w)| w * self.support_set[*idx].performance)
                .sum::<f64>()
                / weight_sum
        } else {
            // Fall back to mean if all weights collapsed to zero.
            let mean = neighbours
                .iter()
                .map(|(idx, _)| self.support_set[*idx].performance)
                .sum::<f64>()
                / neighbours.len() as f64;
            mean
        };

        // Confidence: `1 - mean(neighbour_d) / max_d_across_support`. When
        // the support is tightly packed around the query we get a high
        // confidence; when neighbours are at the far end of the support
        // range we get a low confidence.
        let mean_neighbour_d: f64 =
            neighbours.iter().map(|(_, d)| d).sum::<f64>() / neighbours.len() as f64;
        let max_support_d = distances
            .iter()
            .map(|(_, d)| *d)
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        let confidence = (1.0 - mean_neighbour_d / max_support_d).clamp(0.0, 1.0);

        let nearest_idx = neighbours[0].0;
        let nearest_d = neighbours[0].1;
        let nearest = &self.support_set[nearest_idx];

        Ok(FewShotPrediction {
            predicted_performance: predicted,
            confidence,
            nearest_support_arch: Some(nearest.arch_id.clone()),
            nearest_distance: nearest_d,
        })
    }
}

// Public helper: build a similarity matrix for diagnostics. Not strictly
// required by any algorithm but useful for callers inspecting the support
// set's geometry. Kept here (rather than `architecture_knowledge_graph`)
// because it depends on the per-optimizer distance metric configuration.
impl FewShotArchitectureOptimizer {
    /// Pairwise distance matrix `D[i][j]` over the current support set
    /// under the configured metric.
    ///
    /// Returns [`OptimError::InvalidParameter`] when the optimizer has not
    /// been fitted yet, mirroring `predict`'s contract.
    pub fn support_distance_matrix(&self) -> Result<Array2<f64>> {
        self.check_fitted()?;
        let n = self.support_set.len();
        let mut m: Array2<f64> = Array2::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.distance(
                    &self.support_set[i].embedding,
                    &self.support_set[j].embedding,
                )?;
                m[[i, j]] = d;
                m[[j, i]] = d;
            }
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_knowledge_graph::{ArchitectureKnowledgeGraph, RelationType};

    fn example(arch_id: &str, embedding: Vec<f64>, performance: f64) -> ArchitectureExample {
        ArchitectureExample {
            arch_id: arch_id.to_string(),
            embedding,
            performance,
            domain: "vision".to_string(),
        }
    }

    fn make_support_set() -> Vec<ArchitectureExample> {
        vec![
            example("a", vec![1.0, 0.0, 0.0], 0.90),
            example("b", vec![0.9, 0.1, 0.0], 0.85),
            example("c", vec![0.0, 1.0, 0.0], 0.50),
            example("d", vec![0.0, 0.9, 0.1], 0.45),
            example("e", vec![0.1, 0.1, 0.9], 0.20),
        ]
    }

    #[test]
    fn test_default_config_values() {
        let c = FewShotConfig::default();
        assert_eq!(c.algorithm, FewShotAlgorithm::Prototypical);
        assert_eq!(c.k_shot, 5);
        assert_eq!(c.n_way, 2);
        assert!((c.temperature - 1.0).abs() < 1e-12);
        assert!((c.adapter_lr - 0.01).abs() < 1e-12);
        assert_eq!(c.adapter_steps, 5);
        assert_eq!(c.distance_metric, DistanceMetric::Euclidean);
        assert_eq!(c.seed, 42);
    }

    #[test]
    fn test_builder_pattern_chains() {
        let opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_k_shot(3)
            .with_distance_metric(DistanceMetric::Cosine)
            .with_temperature(0.5)
            .with_seed(7);
        assert_eq!(opt.config().algorithm, FewShotAlgorithm::Matching);
        assert_eq!(opt.config().k_shot, 3);
        assert_eq!(opt.config().distance_metric, DistanceMetric::Cosine);
        assert!((opt.config().temperature - 0.5).abs() < 1e-12);
        assert_eq!(opt.config().seed, 7);
        // Initial state.
        assert!(!opt.is_fitted());
        assert!(opt.support_set().is_empty());
    }

    #[test]
    fn test_fit_from_examples_validates_min_size() {
        let mut opt = FewShotArchitectureOptimizer::new().with_k_shot(5);
        let examples = vec![example("a", vec![1.0, 0.0], 0.5)];
        let err = opt.fit_from_examples(examples);
        assert!(err.is_err(), "expected InvalidParameter for too-small set");
        match err {
            Err(OptimError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
        assert!(!opt.is_fitted());
    }

    #[test]
    fn test_fit_from_examples_rejects_dim_mismatch() {
        let mut opt = FewShotArchitectureOptimizer::new().with_k_shot(2);
        let examples = vec![
            example("a", vec![1.0, 0.0], 0.5),
            example("b", vec![1.0, 0.0, 0.0], 0.6),
        ];
        assert!(opt.fit_from_examples(examples).is_err());
    }

    #[test]
    fn test_fit_from_examples_rejects_non_finite() {
        let mut opt = FewShotArchitectureOptimizer::new().with_k_shot(2);
        let examples = vec![
            example("a", vec![1.0, f64::NAN], 0.5),
            example("b", vec![1.0, 0.0], 0.6),
        ];
        assert!(opt.fit_from_examples(examples).is_err());
    }

    #[test]
    fn test_prototypical_predicts_high_for_similar_to_high_performers() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Prototypical)
            .with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");

        // Query closer to high-performance bucket ([1, 0, 0]) should give
        // a higher predicted performance than a query closer to the
        // low-performance bucket ([0.1, 0.1, 0.9]).
        let p_high = opt.predict(&[0.95, 0.05, 0.0]).expect("predict high");
        let p_low = opt.predict(&[0.1, 0.1, 0.9]).expect("predict low");
        assert!(
            p_high.predicted_performance > p_low.predicted_performance,
            "high query should outscore low: {} vs {}",
            p_high.predicted_performance,
            p_low.predicted_performance
        );
        assert!(p_high.confidence >= 0.0 && p_high.confidence <= 1.0);
        assert!(p_high.nearest_support_arch.is_some());
    }

    #[test]
    fn test_matching_networks_temperature_affects_sharpness() {
        // Lower temperature -> sharper distribution -> higher confidence.
        let support = make_support_set();
        let mut opt_low = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_temperature(0.1)
            .with_k_shot(5);
        opt_low.fit_from_examples(support.clone()).expect("fit low");
        let mut opt_high = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_temperature(10.0)
            .with_k_shot(5);
        opt_high.fit_from_examples(support).expect("fit high");

        let q = vec![1.0, 0.0, 0.0];
        let p_low = opt_low.predict(&q).expect("predict low T");
        let p_high = opt_high.predict(&q).expect("predict high T");
        assert!(
            p_low.confidence > p_high.confidence,
            "low temperature should be sharper: {} vs {}",
            p_low.confidence,
            p_high.confidence
        );
    }

    #[test]
    fn test_maml_adapter_decreases_training_loss() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::MamlAdaptation)
            .with_k_shot(5);
        let support = make_support_set();
        // MSE of the "always predict zero" baseline.
        let baseline_mse: f64 =
            support.iter().map(|ex| ex.performance.powi(2)).sum::<f64>() / support.len() as f64;

        opt.fit_from_examples(support.clone()).expect("fit");
        // MSE of MAML predictions on the training set.
        let trained_mse: f64 = support
            .iter()
            .map(|ex| {
                let pred = opt
                    .predict(&ex.embedding)
                    .expect("predict")
                    .predicted_performance;
                (pred - ex.performance).powi(2)
            })
            .sum::<f64>()
            / support.len() as f64;
        assert!(
            trained_mse < baseline_mse,
            "MAML should drop MSE below zero-baseline: {} vs {}",
            trained_mse,
            baseline_mse
        );
    }

    #[test]
    fn test_maml_adapter_state_present_after_fit() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::MamlAdaptation)
            .with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");
        assert!(opt.adapter_weights.is_some());
        assert!(opt.adapter_bias.is_some());
        assert_eq!(
            opt.adapter_weights.as_ref().expect("weights").len(),
            3,
            "adapter dim must match embedding dim"
        );
    }

    #[test]
    fn test_knn_predict_with_exact_match_returns_exact_performance() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::DistanceWeightedKnn)
            .with_k_shot(3);
        let support = make_support_set();
        opt.fit_from_examples(support.clone()).expect("fit");

        // Query exactly matches the "a" example whose performance is 0.90.
        let pred = opt.predict(&[1.0, 0.0, 0.0]).expect("predict");
        assert!(
            (pred.predicted_performance - 0.90).abs() < 1e-9,
            "exact match should return exact performance, got {}",
            pred.predicted_performance
        );
        assert!(
            (pred.confidence - 1.0).abs() < 1e-9,
            "confidence on exact match should be 1.0, got {}",
            pred.confidence
        );
        assert_eq!(pred.nearest_support_arch.as_deref(), Some("a"));
    }

    #[test]
    fn test_recommend_top_k_returns_sorted_descending() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");

        let candidates: Vec<(&str, Vec<f64>)> = vec![
            ("low-perf", vec![0.1, 0.1, 0.9]),
            ("mid-perf", vec![0.5, 0.5, 0.0]),
            ("high-perf", vec![1.0, 0.0, 0.0]),
            ("med-low", vec![0.0, 0.9, 0.1]),
        ];
        let ranked = opt.recommend_top_k(&candidates, 4).expect("top_k");
        assert_eq!(ranked.len(), 4);
        for window in ranked.windows(2) {
            assert!(
                window[0].1.predicted_performance >= window[1].1.predicted_performance,
                "ranking not descending: {} then {}",
                window[0].1.predicted_performance,
                window[1].1.predicted_performance
            );
        }
        // The highest-ranked candidate should be the one closest to the
        // high-performance bucket.
        assert_eq!(ranked[0].0, "high-perf");
    }

    #[test]
    fn test_recommend_top_k_truncates_to_top_k() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");

        let candidates: Vec<(&str, Vec<f64>)> = vec![
            ("a", vec![1.0, 0.0, 0.0]),
            ("b", vec![0.9, 0.1, 0.0]),
            ("c", vec![0.0, 1.0, 0.0]),
            ("d", vec![0.5, 0.5, 0.0]),
            ("e", vec![0.0, 0.0, 1.0]),
        ];
        let top2 = opt.recommend_top_k(&candidates, 2).expect("top_k=2");
        assert_eq!(top2.len(), 2);
        // top_k=0 returns empty.
        let empty = opt.recommend_top_k(&candidates, 0).expect("top_k=0");
        assert!(empty.is_empty());
        // top_k bigger than candidate count keeps all.
        let all = opt.recommend_top_k(&candidates, 100).expect("top_k=100");
        assert_eq!(all.len(), candidates.len());
    }

    #[test]
    fn test_fit_from_graph_selects_top_performers() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        // Insert 10 architectures with monotonically increasing accuracy.
        let mut ids = Vec::new();
        for i in 0..10 {
            let id = graph.add_architecture(
                format!("arch-{}", i),
                vec![i as f64 / 10.0, 0.0, 0.0],
                "vision",
            );
            graph
                .set_performance(
                    id,
                    PerformanceRecord {
                        task_id: "task".into(),
                        accuracy: 0.10 * (i + 1) as f64,
                        latency_ms: 1.0,
                        memory_mb: 1.0,
                        training_epochs: 1,
                    },
                )
                .expect("set_performance");
            ids.push(id);
        }
        // Also add an unrelated nlp node that must not leak in.
        graph.add_architecture("nlp-a", vec![1.0, 1.0, 1.0], "nlp");

        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::DistanceWeightedKnn)
            .with_k_shot(3);
        opt.fit_from_graph(&graph, "vision", 3)
            .expect("fit_from_graph");
        assert!(opt.is_fitted());
        let support = opt.support_set();
        assert_eq!(support.len(), 3);
        // Top 3 architectures by accuracy are arch-9, arch-8, arch-7 (in
        // descending order). Each must be present.
        let arch_ids: Vec<&str> = support.iter().map(|e| e.arch_id.as_str()).collect();
        assert!(arch_ids.contains(&"arch-9"));
        assert!(arch_ids.contains(&"arch-8"));
        assert!(arch_ids.contains(&"arch-7"));
        // No nlp architecture leaked in.
        assert!(arch_ids.iter().all(|id| !id.starts_with("nlp")));
    }

    #[test]
    fn test_fit_from_graph_errors_on_empty_domain() {
        let graph = ArchitectureKnowledgeGraph::new();
        let mut opt = FewShotArchitectureOptimizer::new().with_k_shot(1);
        assert!(opt.fit_from_graph(&graph, "vision", 5).is_err());
    }

    #[test]
    fn test_predict_before_fit_errors() {
        let opt = FewShotArchitectureOptimizer::new();
        let r = opt.predict(&[1.0, 0.0, 0.0]);
        assert!(r.is_err(), "predict before fit must error");
        match r {
            Err(OptimError::InvalidParameter(msg)) => assert!(msg.contains("not fitted")),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::MamlAdaptation)
            .with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");
        assert!(opt.is_fitted());
        assert!(!opt.support_set().is_empty());
        assert!(opt.adapter_weights.is_some());

        opt.reset();
        assert!(!opt.is_fitted());
        assert!(opt.support_set().is_empty());
        assert!(opt.adapter_weights.is_none());
        assert!(opt.adapter_bias.is_none());
    }

    #[test]
    fn test_distance_metrics_produce_different_predictions() {
        let support = make_support_set();
        let mut opt_euc = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_distance_metric(DistanceMetric::Euclidean)
            .with_k_shot(5);
        opt_euc.fit_from_examples(support.clone()).expect("fit euc");

        let mut opt_cos = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_distance_metric(DistanceMetric::Cosine)
            .with_k_shot(5);
        opt_cos.fit_from_examples(support.clone()).expect("fit cos");

        let mut opt_man = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_distance_metric(DistanceMetric::Manhattan)
            .with_k_shot(5);
        opt_man.fit_from_examples(support).expect("fit man");

        let q = vec![0.5, 0.5, 0.5];
        let p_euc = opt_euc.predict(&q).expect("euc predict");
        let p_cos = opt_cos.predict(&q).expect("cos predict");
        let p_man = opt_man.predict(&q).expect("man predict");

        // Euclidean and Cosine should yield different numbers on a non-
        // axis-aligned query because one is scale-sensitive and the other
        // is not.
        assert!(
            (p_euc.predicted_performance - p_cos.predicted_performance).abs() > 1e-9,
            "Euclidean ({}) and Cosine ({}) must differ",
            p_euc.predicted_performance,
            p_cos.predicted_performance
        );
        // Manhattan and Euclidean on a non-axis query also produce
        // different shapes, so the resulting weights diverge.
        assert!(
            (p_euc.predicted_performance - p_man.predicted_performance).abs() > 1e-12
                || (p_euc.confidence - p_man.confidence).abs() > 1e-12,
            "Euclidean and Manhattan should differ in at least one summary"
        );
    }

    #[test]
    fn test_fewshot_prediction_serde_roundtrip() {
        let pred = FewShotPrediction {
            predicted_performance: 0.87,
            confidence: 0.93,
            nearest_support_arch: Some("resnet50".to_string()),
            nearest_distance: 0.12,
        };
        let json = serde_json::to_string(&pred).expect("serialise");
        let back: FewShotPrediction = serde_json::from_str(&json).expect("deserialise");
        assert!((back.predicted_performance - pred.predicted_performance).abs() < 1e-12);
        assert!((back.confidence - pred.confidence).abs() < 1e-12);
        assert_eq!(back.nearest_support_arch, pred.nearest_support_arch);
        assert!((back.nearest_distance - pred.nearest_distance).abs() < 1e-12);
    }

    #[test]
    fn test_architecture_example_serde_roundtrip() {
        let ex = example("resnet", vec![1.0, 2.0, 3.0], 0.91);
        let json = serde_json::to_string(&ex).expect("serialise");
        let back: ArchitectureExample = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back.arch_id, ex.arch_id);
        assert_eq!(back.embedding, ex.embedding);
        assert!((back.performance - ex.performance).abs() < 1e-12);
        assert_eq!(back.domain, ex.domain);
    }

    #[test]
    fn test_support_distance_matrix_is_symmetric() {
        let mut opt = FewShotArchitectureOptimizer::new().with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");
        let m = opt.support_distance_matrix().expect("matrix");
        assert_eq!(m.shape(), &[5, 5]);
        for i in 0..5 {
            assert!(m[[i, i]].abs() < 1e-12, "diagonal {} must be 0", i);
            for j in 0..5 {
                assert!(
                    (m[[i, j]] - m[[j, i]]).abs() < 1e-12,
                    "matrix not symmetric at ({}, {}): {} vs {}",
                    i,
                    j,
                    m[[i, j]],
                    m[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_predict_batch_returns_one_per_query() {
        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::Matching)
            .with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");

        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let predictions = opt.predict_batch(&queries).expect("batch");
        assert_eq!(predictions.len(), 3);
        for p in &predictions {
            assert!(p.confidence >= 0.0 && p.confidence <= 1.0);
            assert!(p.predicted_performance.is_finite());
        }
    }

    #[test]
    fn test_softmax_neg_uniform_for_equal_distances() {
        let weights = FewShotArchitectureOptimizer::softmax_neg(&[1.0, 1.0, 1.0, 1.0], 1.0);
        for w in &weights {
            assert!((w - 0.25).abs() < 1e-12, "expected 0.25 got {}", w);
        }
    }

    #[test]
    fn test_softmax_neg_temperature_affects_sharpness() {
        let cold = FewShotArchitectureOptimizer::softmax_neg(&[0.0, 1.0, 2.0], 0.1);
        let warm = FewShotArchitectureOptimizer::softmax_neg(&[0.0, 1.0, 2.0], 10.0);
        // Cold should be near one-hot on the closest (lowest-d) point.
        assert!(cold[0] > 0.95, "expected near-1 max in cold: {:?}", cold);
        // Warm should be close to uniform.
        for w in &warm {
            assert!(
                (w - 1.0 / 3.0).abs() < 0.1,
                "warm should be near-uniform: {:?}",
                warm
            );
        }
    }

    #[test]
    fn test_entropy_handles_zero_probability() {
        // 0 log 0 = 0 by convention. The function must not panic.
        let h = FewShotArchitectureOptimizer::entropy(&[0.0, 1.0]);
        assert!(h.abs() < 1e-12);
        let h_uniform = FewShotArchitectureOptimizer::entropy(&[0.5, 0.5]);
        assert!((h_uniform - (2.0_f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_graph_with_transfer_edges_does_not_leak() {
        // Even when Transfer edges exist between domains, fit_from_graph
        // must respect the source_domain filter.
        let mut graph = ArchitectureKnowledgeGraph::new();
        let v1 = graph.add_architecture("v1", vec![1.0, 0.0, 0.0], "vision");
        let n1 = graph.add_architecture("n1", vec![1.0, 0.0, 0.0], "nlp");
        graph
            .set_performance(
                v1,
                PerformanceRecord {
                    task_id: "t".into(),
                    accuracy: 0.9,
                    latency_ms: 1.0,
                    memory_mb: 1.0,
                    training_epochs: 1,
                },
            )
            .expect("perf v1");
        graph
            .set_performance(
                n1,
                PerformanceRecord {
                    task_id: "t".into(),
                    accuracy: 0.95,
                    latency_ms: 1.0,
                    memory_mb: 1.0,
                    training_epochs: 1,
                },
            )
            .expect("perf n1");
        graph
            .add_edge(n1, v1, RelationType::Transfers, 0.7)
            .expect("transfers");

        let mut opt = FewShotArchitectureOptimizer::new()
            .with_algorithm(FewShotAlgorithm::DistanceWeightedKnn)
            .with_k_shot(1);
        opt.fit_from_graph(&graph, "vision", 1)
            .expect("fit_from_graph");
        let support = opt.support_set();
        assert_eq!(support.len(), 1);
        // Even though n1 has higher accuracy, the domain filter must
        // restrict the support set to vision-domain nodes.
        assert_eq!(support[0].arch_id, "v1");
        assert!((support[0].performance - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_predict_query_dim_mismatch_errors() {
        let mut opt = FewShotArchitectureOptimizer::new().with_k_shot(5);
        opt.fit_from_examples(make_support_set()).expect("fit");
        let bad = opt.predict(&[1.0, 0.0]);
        assert!(bad.is_err(), "expected dim mismatch error");
    }
}
