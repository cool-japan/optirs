// Neural predictor-based search strategy

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

#[allow(unused_imports)]
use crate::error::Result;
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::random::RandomSearch;
use super::{SearchStrategy, SearchStrategyStatistics};

/// Neural predictor-based search
pub struct NeuralPredictorSearch<T: Float + Debug + Send + Sync + 'static> {
    predictor_network: PredictorNetwork<T>,
    architecture_encoder: ArchitectureEncoder<T>,
    search_optimizer: SearchOptimizer<T>,
    confidence_threshold: T,
    statistics: SearchStrategyStatistics<T>,
    uncertainty_sampling: bool,
}

/// Predictor network for neural predictor search
#[derive(Debug)]
pub struct PredictorNetwork<T: Float + Debug + Send + Sync + 'static> {
    layers: Vec<PredictorLayer<T>>,
    dropout_rates: Vec<T>,
    architecture: Vec<usize>,
}

/// Predictor layer
#[derive(Debug)]
pub struct PredictorLayer<T: Float + Debug + Send + Sync + 'static> {
    weights: Array2<T>,
    bias: Array1<T>,
    activation: ActivationFunction,
}

/// Activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
}

/// Architecture encoder for neural predictor
#[derive(Debug)]
pub struct ArchitectureEncoder<T: Float + Debug + Send + Sync + 'static> {
    encoding_weights: Array2<T>,
    _embeddingdim: usize,
    max_components: usize,
}

/// Search optimizer for neural predictor
#[derive(Debug)]
pub struct SearchOptimizer<T: Float + Debug + Send + Sync + 'static> {
    optimizer_type: SearchOptimizerType,
    _learningrate: T,
    momentum: T,
    parameters: HashMap<String, Array1<T>>,
}

/// Search optimizer types
#[derive(Debug, Clone, Copy)]
pub enum SearchOptimizerType {
    Adam,
    SGD,
    RMSprop,
    AdamW,
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + 'static + std::iter::Sum,
    > NeuralPredictorSearch<T>
{
    pub fn new(
        predictor_architecture: Vec<usize>,
        embeddingdim: usize,
        confidence_threshold: f64,
    ) -> Self {
        Self {
            predictor_network: PredictorNetwork::new(predictor_architecture),
            architecture_encoder: ArchitectureEncoder::new(embeddingdim),
            search_optimizer: SearchOptimizer::new(
                SearchOptimizerType::Adam,
                scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero()),
            ),
            confidence_threshold: scirs2_core::numeric::NumCast::from(confidence_threshold)
                .unwrap_or_else(|| T::zero()),
            statistics: SearchStrategyStatistics::default(),
            uncertainty_sampling: true,
        }
    }

    fn predict_performance(&self, architecture: &OptimizerArchitecture<T>) -> Result<(T, T)> {
        // Encode architecture
        let encoded = self.architecture_encoder.encode(architecture)?;

        // Forward pass through predictor network
        let (prediction, uncertainty) =
            self.predictor_network.forward_with_uncertainty(&encoded)?;

        Ok((prediction, uncertainty))
    }

    fn train_predictor(
        &mut self,
        architectures: &[OptimizerArchitecture<T>],
        performances: &[T],
    ) -> Result<()> {
        if architectures.len() != performances.len() || architectures.is_empty() {
            return Ok(());
        }

        // Encode all architectures
        let encoded_archs: std::result::Result<Vec<_>, _> = architectures
            .iter()
            .map(|arch| self.architecture_encoder.encode(arch))
            .collect();
        let encoded_archs = encoded_archs?;

        // Train predictor network
        for (encoded_arch, &target_performance) in encoded_archs.iter().zip(performances.iter()) {
            let (_prediction, _) = self
                .predictor_network
                .forward_with_uncertainty(encoded_arch)?;

            // Simplified gradient update
            self.predictor_network.backward_update(
                encoded_arch,
                target_performance,
                &mut self.search_optimizer,
            )?;
        }

        Ok(())
    }

    fn generate_candidate_with_uncertainty(
        &mut self,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        // Generate multiple candidates and select based on uncertainty
        let num_candidates = 50;
        let mut candidates = Vec::new();
        let mut random_search = RandomSearch::<T>::new(None);
        random_search.initialize(searchspace)?;

        for _ in 0..num_candidates {
            candidates.push(random_search.generate_architecture(searchspace, &VecDeque::new())?);
        }

        // Select candidate with highest uncertainty (for exploration) or highest predicted
        // performance (for exploitation)
        let mut best_candidate = candidates[0].clone();
        let mut best_score = T::neg_infinity();

        for candidate in candidates {
            let (predicted_perf, uncertainty) = self.predict_performance(&candidate)?;

            // Combine prediction and uncertainty for selection
            let score = if self.uncertainty_sampling {
                predicted_perf + uncertainty // UCB-style selection
            } else {
                predicted_perf // Pure exploitation
            };

            if score > best_score {
                best_score = score;
                best_candidate = candidate;
            }
        }

        Ok(best_candidate)
    }
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + 'static + std::iter::Sum,
    > SearchStrategy<T> for NeuralPredictorSearch<T>
{
    fn initialize(&mut self, _searchspace: &SearchSpaceConfig) -> Result<()> {
        // Initialize predictor network with random weights
        self.predictor_network.initialize()?;
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        // Train predictor if enough data is available
        if history.len() > 10 {
            let architectures: Vec<_> = history.iter().map(|r| r.architecture.clone()).collect();
            let performances: Vec<_> = history
                .iter()
                .filter_map(|r| {
                    r.evaluation_results
                        .metric_scores
                        .get(&EvaluationMetric::FinalPerformance)
                })
                .cloned()
                .collect();

            if architectures.len() == performances.len() {
                self.train_predictor(&architectures, &performances)?;
            }
        }

        // Generate candidate based on predictor
        let architecture = if history.len() > 5 {
            self.generate_candidate_with_uncertainty(searchspace)?
        } else {
            // Use random search for initial exploration
            let mut random_search = RandomSearch::<T>::new(None);
            random_search.initialize(searchspace)?;
            random_search.generate_architecture(searchspace, history)?
        };

        self.statistics.total_architectures_generated += 1;
        Ok(architecture)
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        // Extract architectures and performances
        let architectures: Vec<_> = results.iter().map(|r| r.architecture.clone()).collect();
        let performances: Vec<_> = results
            .iter()
            .filter_map(|r| {
                r.evaluation_results
                    .metric_scores
                    .get(&EvaluationMetric::FinalPerformance)
            })
            .cloned()
            .collect();

        if architectures.len() == performances.len() && !performances.is_empty() {
            // Update statistics
            self.statistics.best_performance = performances
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(T::zero());

            let sum: T = performances.iter().cloned().sum();
            self.statistics.average_performance =
                sum / T::from(performances.len()).expect("conversion from usize to T failed");

            // Train predictor with new data
            self.train_predictor(&architectures, &performances)?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "NeuralPredictorSearch"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate = if self.uncertainty_sampling {
            scirs2_core::numeric::NumCast::from(0.7).unwrap_or_else(|| T::zero())
        } else {
            scirs2_core::numeric::NumCast::from(0.3).unwrap_or_else(|| T::zero())
        };
        stats.exploitation_rate = T::one() - stats.exploration_rate;
        stats
    }
}

// Implementation for supporting components
impl<T: Float + Debug + Default + Clone + 'static + std::iter::Sum + Send + Sync>
    PredictorNetwork<T>
{
    fn new(architecture: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..architecture.len() - 1 {
            layers.push(PredictorLayer::new(architecture[i], architecture[i + 1]));
        }

        Self {
            layers,
            dropout_rates: vec![
                scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::zero());
                architecture.len() - 1
            ],
            architecture,
        }
    }

    fn initialize(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.initialize()?;
        }
        Ok(())
    }

    fn forward_with_uncertainty(&self, input: &Array1<T>) -> Result<(T, T)> {
        let mut current = input.clone();

        // Forward pass through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current)?;

            // Apply dropout for uncertainty estimation (Monte Carlo dropout)
            if i < self.dropout_rates.len() {
                current = self.apply_dropout(&current, self.dropout_rates[i]);
            }
        }

        // For simplicity, return the first output as prediction and a simple uncertainty estimate
        let prediction = current[0];
        let uncertainty = current.iter().map(|&x| x * x).sum::<T>().sqrt()
            * scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());

        Ok((prediction, uncertainty))
    }

    fn backward_update(
        &mut self,
        _input: &Array1<T>,
        _target: T,
        _optimizer: &mut SearchOptimizer<T>,
    ) -> Result<()> {
        // Simplified backward pass - in practice would implement proper backpropagation
        Ok(())
    }

    fn apply_dropout(&self, input: &Array1<T>, dropoutrate: T) -> Array1<T> {
        input.mapv(|x| {
            if scirs2_core::random::Random::default().random::<f64>()
                < dropoutrate.to_f64().unwrap_or(0.0)
            {
                T::zero()
            } else {
                x / (T::one() - dropoutrate)
            }
        })
    }
}

impl<T: Float + Debug + Default + Clone + 'static + Send + Sync> PredictorLayer<T> {
    fn new(input_size: usize, outputsize: usize) -> Self {
        Self {
            weights: Array2::zeros((outputsize, input_size)),
            bias: Array1::zeros(outputsize),
            activation: ActivationFunction::ReLU,
        }
    }

    fn initialize(&mut self) -> Result<()> {
        // Xavier initialization
        let fan_in = self.weights.ncols() as f64;
        let fan_out = self.weights.nrows() as f64;
        let scale = (6.0 / (fan_in + fan_out)).sqrt();

        self.weights = Array2::from_shape_fn(self.weights.raw_dim(), |_| {
            T::from(scirs2_core::random::Random::default().random::<f64>() * scale * 2.0 - scale)
                .expect("xavier initialization conversion failed")
        });

        Ok(())
    }

    fn forward(&self, input: &Array1<T>) -> Result<Array1<T>> {
        let linear_output = self.weights.dot(input) + &self.bias;
        Ok(self.apply_activation(&linear_output))
    }

    fn apply_activation(&self, x: &Array1<T>) -> Array1<T> {
        match self.activation {
            ActivationFunction::ReLU => x.mapv(|xi| if xi > T::zero() { xi } else { T::zero() }),
            ActivationFunction::GELU => x.mapv(|xi| {
                let x_f64 = xi.to_f64().unwrap_or(0.0);
                let gelu_val = 0.5
                    * x_f64
                    * (1.0 + (x_f64 * 0.7978845608 * (1.0 + 0.044715 * x_f64 * x_f64)).tanh());
                scirs2_core::numeric::NumCast::from(gelu_val).unwrap_or_else(|| T::zero())
            }),
            ActivationFunction::Swish => x.mapv(|xi| {
                let sigmoid = T::one() / (T::one() + (-xi).exp());
                xi * sigmoid
            }),
            ActivationFunction::Tanh => x.mapv(|xi| xi.tanh()),
            ActivationFunction::Sigmoid => x.mapv(|xi| T::one() / (T::one() + (-xi).exp())),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> ArchitectureEncoder<T> {
    fn new(embeddingdim: usize) -> Self {
        Self {
            encoding_weights: Array2::zeros((embeddingdim, 64)), // Assume max 64 components
            _embeddingdim: embeddingdim,
            max_components: 64,
        }
    }

    fn encode(&self, architecture: &OptimizerArchitecture<T>) -> Result<Array1<T>> {
        // Deterministic, vocabulary-based encoding of the architecture's
        // component types.  Each component string (produced elsewhere via
        // `format!("{:?}", ComponentType)`) is mapped onto a fixed, ordered
        // vocabulary and accumulated into a multi-hot block, followed by a few
        // continuous descriptors.  Unlike a raw string hash this preserves
        // locality (identical types collide on the same slot) and yields a
        // meaningful, stable feature for the predictor network.
        let mut encoding = encode_component_block::<T, _>(
            architecture.components.iter().take(self.max_components),
        );

        // Pad (or truncate) to the embedding dimension expected by the
        // predictor network's input layer.  This preserves the original
        // fixed-length contract of `_embeddingdim`.
        encoding.resize(self._embeddingdim, T::zero());
        Ok(Array1::from_vec(encoding))
    }
}

/// Ordered vocabulary of known optimizer-component type names.
///
/// These correspond exactly to the field-less variants of
/// [`crate::architecture::ComponentType`], whose `Debug` representation is the
/// variant name and is what populates `OptimizerArchitecture::components`
/// across every search strategy.  The order is fixed (matching the enum's
/// declaration order) so the produced encoding is deterministic and stable
/// across runs and builds.  A trailing out-of-vocabulary slot (added by
/// [`encode_component_block`]) absorbs any unrecognised name.
const COMPONENT_VOCABULARY: [&str; 41] = [
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "AdaGrad",
    "AdaDelta",
    "Momentum",
    "Nesterov",
    "LRScheduler",
    "GradientClipping",
    "BatchNorm",
    "Dropout",
    "LAMB",
    "LARS",
    "Lion",
    "RAdam",
    "Lookahead",
    "SAM",
    "LBFGS",
    "SparseAdam",
    "GroupedAdam",
    "MAML",
    "L1Regularizer",
    "L2Regularizer",
    "ElasticNetRegularizer",
    "DropoutRegularizer",
    "WeightDecay",
    "AdaptiveLR",
    "AdaptiveMomentum",
    "AdaptiveRegularization",
    "LSTMOptimizer",
    "TransformerOptimizer",
    "AttentionOptimizer",
    "MetaSGD",
    "ConstantLR",
    "ExponentialLR",
    "StepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "CyclicLR",
    "Reptile",
];

/// Number of continuous descriptors appended after the multi-hot block.
const COMPONENT_DESCRIPTOR_COUNT: usize = 3;

/// Total fixed length of the component feature block produced by
/// [`encode_component_block`]: one slot per known type, one out-of-vocabulary
/// slot, and the trailing continuous descriptors.
const COMPONENT_BLOCK_LEN: usize = COMPONENT_VOCABULARY.len() + 1 + COMPONENT_DESCRIPTOR_COUNT;

/// Resolve a component type name to its vocabulary index.
///
/// Returns the matching index for a known type, or the dedicated
/// out-of-vocabulary index (`COMPONENT_VOCABULARY.len()`) for any unrecognised
/// name.  The lookup is exact and deterministic.
fn component_vocab_index(name: &str) -> usize {
    COMPONENT_VOCABULARY
        .iter()
        .position(|known| *known == name)
        .unwrap_or(COMPONENT_VOCABULARY.len())
}

/// Build a deterministic, fixed-length feature block for a sequence of
/// component type names.
///
/// Layout (length [`COMPONENT_BLOCK_LEN`]):
/// * `[0, VOCAB_LEN)`  multi-hot counts: how many components of each known type
///   are present (occurrence counts, so repeated types accumulate);
/// * `[VOCAB_LEN]`     out-of-vocabulary count for unrecognised names;
/// * trailing descriptors: normalised component count, mean normalised name
///   length, and fraction of names containing the `"Adam"` substring (a cheap
///   family indicator).  These add continuous structure on top of the
///   discrete one-hot signal.
fn encode_component_block<'a, T, I>(components: I) -> Vec<T>
where
    T: Float + Debug + Send + Sync + 'static + Default + Clone,
    I: Iterator<Item = &'a String>,
{
    let mut block = vec![T::zero(); COMPONENT_BLOCK_LEN];

    let one: T = scirs2_core::numeric::NumCast::from(1.0).unwrap_or_else(|| T::zero());

    let mut total: usize = 0;
    let mut name_len_sum: usize = 0;
    let mut adam_family: usize = 0;

    for component in components {
        // Unknown names resolve to the dedicated out-of-vocabulary index
        // (`COMPONENT_VOCABULARY.len()`) so they never collide with a known
        // type's slot.
        let idx = component_vocab_index(component);
        block[idx] = block[idx] + one;

        total += 1;
        name_len_sum += component.len();
        if component.contains("Adam") {
            adam_family += 1;
        }
    }

    // Continuous descriptors.  Normalisers are chosen to keep values in a
    // roughly unit range without depending on any RNG.
    let descriptor_base = COMPONENT_VOCABULARY.len() + 1;
    if total > 0 {
        let total_t: T =
            scirs2_core::numeric::NumCast::from(total as f64).unwrap_or_else(|| T::zero());

        // Normalised component count (relative to a nominal cap of 16).
        block[descriptor_base] =
            scirs2_core::numeric::NumCast::from(total as f64 / 16.0).unwrap_or_else(|| T::zero());

        // Mean name length, normalised by a nominal max name length of 24.
        let mean_len = (name_len_sum as f64 / total as f64) / 24.0;
        block[descriptor_base + 1] =
            scirs2_core::numeric::NumCast::from(mean_len).unwrap_or_else(|| T::zero());

        // Fraction of Adam-family components.
        let adam_frac: T =
            scirs2_core::numeric::NumCast::from(adam_family as f64).unwrap_or_else(|| T::zero());
        block[descriptor_base + 2] = adam_frac / total_t;
    }

    block
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> SearchOptimizer<T> {
    fn new(optimizer_type: SearchOptimizerType, learningrate: T) -> Self {
        Self {
            optimizer_type,
            _learningrate: learningrate,
            momentum: scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero()),
            parameters: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_arch(components: &[&str]) -> OptimizerArchitecture<f64> {
        OptimizerArchitecture {
            components: components.iter().map(|s| s.to_string()).collect(),
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: "test".to_string(),
        }
    }

    #[test]
    fn encode_is_deterministic_and_fixed_length() {
        let embedding_dim = 96;
        let encoder = ArchitectureEncoder::<f64>::new(embedding_dim);
        let arch = make_arch(&["Adam", "SGD", "RMSprop"]);

        let first = encoder.encode(&arch).expect("encode should succeed");
        let second = encoder.encode(&arch).expect("encode should succeed");

        assert_eq!(first.len(), embedding_dim);
        assert_eq!(second.len(), embedding_dim);
        assert_eq!(first, second, "encoding must be deterministic");
    }

    #[test]
    fn different_known_types_differ_same_type_matches() {
        let encoder = ArchitectureEncoder::<f64>::new(COMPONENT_BLOCK_LEN);

        let adam = encoder.encode(&make_arch(&["Adam"])).expect("encode");
        let adam_again = encoder.encode(&make_arch(&["Adam"])).expect("encode");
        let sgd = encoder.encode(&make_arch(&["SGD"])).expect("encode");

        assert_eq!(adam, adam_again, "same type must encode identically");
        assert_ne!(adam, sgd, "different known types must differ");

        // The multi-hot slots for Adam and SGD must be the distinct ones.
        let adam_idx = component_vocab_index("Adam");
        let sgd_idx = component_vocab_index("SGD");
        assert_ne!(adam_idx, sgd_idx);
        assert_eq!(adam[adam_idx], 1.0);
        assert_eq!(sgd[sgd_idx], 1.0);
    }

    #[test]
    fn unknown_type_maps_to_oov_slot() {
        let encoder = ArchitectureEncoder::<f64>::new(COMPONENT_BLOCK_LEN);
        let oov_index = COMPONENT_VOCABULARY.len();

        let encoded = encoder
            .encode(&make_arch(&["TotallyUnknownOptimizer"]))
            .expect("encode must not panic on unknown type");

        assert_eq!(encoded.len(), COMPONENT_BLOCK_LEN);
        assert_eq!(
            encoded[oov_index], 1.0,
            "unknown name must land in the out-of-vocabulary slot"
        );
        // No known-type slot should be set by an unknown name.
        for (i, value) in encoded.iter().enumerate().take(oov_index) {
            assert_eq!(*value, 0.0, "known slot {} must stay zero", i);
        }
    }

    #[test]
    fn repeated_types_accumulate_counts() {
        let block = encode_component_block::<f64, _>(
            ["Adam".to_string(), "Adam".to_string(), "SGD".to_string()].iter(),
        );
        let adam_idx = component_vocab_index("Adam");
        let sgd_idx = component_vocab_index("SGD");
        assert_eq!(block[adam_idx], 2.0);
        assert_eq!(block[sgd_idx], 1.0);
    }
}
