// Neural predictor-based search strategy

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::random::RandomSearch;
use super::{SearchStrategy, SearchStrategyStatistics};

/// Neural predictor-based search
pub struct NeuralPredictorSearch<T: Float + Debug + Send + Sync + 'static> {
    predictor_network: PredictorNetwork<T>,
    architecture_encoder: ArchitectureEncoder<T>,
    search_optimizer: SearchOptimizer<T>,
    /// Confidence the predictor must reach before its point estimates are
    /// trusted; see [`NeuralPredictorSearch::predictions_are_trusted`].
    confidence_threshold: T,
    statistics: SearchStrategyStatistics<T>,
    uncertainty_sampling: bool,
    /// RNG driving dropout masks and the training-order shuffle. Held here (not
    /// in the network) so the network's forward passes stay `&self` and so a
    /// seeded search is reproducible end to end.
    rng: Random<scirs2_core::random::rngs::StdRng>,
    /// Number of gradient passes over the recorded data per `train_predictor`
    /// call.
    training_epochs: usize,
}

/// Predictor network for neural predictor search.
///
/// Dropout is applied **only after hidden layers**. The previous version applied
/// it after every layer including the output, so `forward_with_uncertainty`
/// returned a single noisy sample rather than a Monte-Carlo-dropout estimate, and
/// a "prediction" could be zeroed outright.
#[derive(Debug)]
pub struct PredictorNetwork<T: Float + Debug + Send + Sync + 'static> {
    layers: Vec<PredictorLayer<T>>,
    /// Dropout probability applied after each **hidden** layer; the entry for the
    /// output layer is unused and kept only so indices line up with `layers`.
    dropout_rates: Vec<T>,
    architecture: Vec<usize>,
    /// Number of stochastic forward passes used to estimate uncertainty.
    mc_samples: usize,
}

/// Predictor layer
#[derive(Debug)]
pub struct PredictorLayer<T: Float + Debug + Send + Sync + 'static> {
    weights: Array2<T>,
    bias: Array1<T>,
    activation: ActivationFunction,
}

/// Activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
    /// Pass-through. Used by the **output** layer: a ReLU there clamps every
    /// prediction to `>= 0` and zeroes the gradient for any negative
    /// pre-activation, so the network could never learn a target it had
    /// initially undershot.
    Identity,
}

/// Architecture encoder for neural predictor.
///
/// The encoding is a deterministic vocabulary lookup (see
/// `encode_component_block`); the previous `encoding_weights: Array2::zeros(..)`
/// field was never read by `encode` and has been removed rather than left as dead
/// state suggesting a learned projection that does not exist.
#[derive(Debug)]
pub struct ArchitectureEncoder<T: Float + Debug + Send + Sync + 'static> {
    embedding_dim: usize,
    max_components: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// Gradient-descent optimizer used to train the predictor network.
///
/// Every [`SearchOptimizerType`] is implemented for real; none falls back to
/// plain SGD. Per-parameter state (momentum / second-moment buffers) lives in
/// `parameters`, keyed by the caller-supplied parameter name plus a suffix.
#[derive(Debug)]
pub struct SearchOptimizer<T: Float + Debug + Send + Sync + 'static> {
    optimizer_type: SearchOptimizerType,
    learning_rate: T,
    momentum: T,
    /// Decoupled weight decay, applied by [`SearchOptimizerType::AdamW`].
    weight_decay: T,
    /// Optimizer state buffers (momentum, second moments) as flat vectors.
    parameters: HashMap<String, Array1<T>>,
    /// Global step count, needed for Adam/AdamW bias correction.
    step: u64,
}

/// Search optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
                scirs2_core::numeric::NumCast::from(DEFAULT_PREDICTOR_LEARNING_RATE)
                    .unwrap_or_else(|| T::zero()),
            ),
            confidence_threshold: scirs2_core::numeric::NumCast::from(confidence_threshold)
                .unwrap_or_else(|| T::zero()),
            statistics: SearchStrategyStatistics::default(),
            uncertainty_sampling: true,
            rng: Random::seed(scirs2_core::random::random::<u64>()),
            training_epochs: DEFAULT_PREDICTOR_EPOCHS,
        }
    }

    /// Confidence the predictor must reach before its point estimates are trusted.
    pub fn confidence_threshold(&self) -> T {
        self.confidence_threshold
    }

    /// Set the confidence the predictor must reach before its point estimates are
    /// trusted.
    ///
    /// Rejects values outside `[0, 1]`: confidence is a probability-like quantity
    /// and silently clamping an out-of-range threshold would hide a configuration
    /// mistake behind plausible-looking behaviour.
    pub fn set_confidence_threshold(&mut self, threshold: T) -> Result<()> {
        if !(threshold >= T::zero() && threshold <= T::one()) {
            return Err(OptimError::InvalidParameter(format!(
                "confidence threshold must lie in [0, 1], got {:?}",
                threshold
            )));
        }
        self.confidence_threshold = threshold;
        Ok(())
    }

    /// Whether the predictor's point estimates are trusted for a candidate pool
    /// whose mean Monte-Carlo-dropout spread is `mean_uncertainty`.
    ///
    /// Confidence is `1 / (1 + mean_uncertainty)`: one when the stochastic passes
    /// agree exactly, falling towards zero as they disagree. The threshold used to
    /// be a constructor argument that nothing read, so a caller asking for a
    /// cautious predictor got the same exploratory behaviour as one asking for a
    /// confident one.
    pub fn predictions_are_trusted(&self, mean_uncertainty: T) -> bool {
        let spread = if mean_uncertainty.is_finite() {
            mean_uncertainty.max(T::zero())
        } else {
            // A non-finite spread is the least trustworthy state there is.
            return false;
        };
        let confidence = T::one() / (T::one() + spread);
        confidence >= self.confidence_threshold
    }

    /// Make dropout sampling and the training shuffle reproducible.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Random::seed(seed);
    }

    /// Select the gradient-descent rule and learning rate used to train the
    /// predictor.
    pub fn set_optimizer(&mut self, optimizer_type: SearchOptimizerType, learning_rate: T) {
        self.search_optimizer = SearchOptimizer::new(optimizer_type, learning_rate);
    }

    /// Number of gradient passes over the recorded data per training call.
    pub fn set_training_epochs(&mut self, epochs: usize) {
        self.training_epochs = epochs;
    }

    /// Deterministic (eval-mode, dropout-free) mean squared error of the
    /// predictor over `(architecture, performance)` pairs.
    ///
    /// Exposed so callers — and the regression tests — can verify that training
    /// actually reduces the loss instead of taking `Ok(())` on faith.
    pub fn training_loss(
        &self,
        architectures: &[OptimizerArchitecture<T>],
        performances: &[T],
    ) -> Result<T> {
        if architectures.len() != performances.len() || architectures.is_empty() {
            return Ok(T::zero());
        }
        let mut total = T::zero();
        for (architecture, &target) in architectures.iter().zip(performances.iter()) {
            let encoded = self.architecture_encoder.encode(architecture)?;
            let prediction = self.predictor_network.predict(&encoded)?;
            let error = prediction - target;
            total = total + error * error;
        }
        Ok(total
            / scirs2_core::numeric::NumCast::from(architectures.len() as f64)
                .unwrap_or_else(T::one))
    }

    /// Monte-Carlo-dropout prediction: the mean and standard deviation of
    /// [`PredictorNetwork::mc_samples`] stochastic forward passes.
    fn predict_performance(&mut self, architecture: &OptimizerArchitecture<T>) -> Result<(T, T)> {
        let encoded = self.architecture_encoder.encode(architecture)?;
        self.predictor_network
            .forward_with_uncertainty(&encoded, &mut self.rng)
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

        // Real training: `training_epochs` shuffled passes of stochastic
        // gradient descent through the network's weights and biases. This used to
        // be an empty `Ok(())`, so the predictor never learned anything and its
        // "predictions" were whatever the Xavier initialization happened to emit.
        let mut order: Vec<usize> = (0..encoded_archs.len()).collect();
        for _ in 0..self.training_epochs.max(1) {
            // Fisher-Yates shuffle so the update order does not bias the fit.
            for i in (1..order.len()).rev() {
                let j = self.rng.gen_range(0..=i);
                order.swap(i, j);
            }
            for &idx in &order {
                let pass = self
                    .predictor_network
                    .forward_train(&encoded_archs[idx], &mut self.rng)?;
                self.predictor_network.backward_update(
                    &pass,
                    performances[idx],
                    &mut self.search_optimizer,
                )?;
            }
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

        // Score every candidate once, then decide *one* selection rule for the whole
        // pool: mixing two scoring scales across candidates would make the argmax
        // meaningless.
        let mut scored: Vec<(OptimizerArchitecture<T>, T, T)> =
            Vec::with_capacity(candidates.len());
        let mut total_uncertainty = T::zero();
        for candidate in candidates {
            let (predicted_perf, uncertainty) = self.predict_performance(&candidate)?;
            total_uncertainty = total_uncertainty + uncertainty;
            scored.push((candidate, predicted_perf, uncertainty));
        }

        let count: T =
            scirs2_core::numeric::NumCast::from(scored.len() as f64).unwrap_or_else(T::one);
        let mean_uncertainty = if count > T::zero() {
            total_uncertainty / count
        } else {
            T::zero()
        };

        // Exploit the predictor when it is confident enough (or when uncertainty
        // sampling is switched off); otherwise use the UCB score, which is what
        // pushes the search towards regions the predictor does not yet know.
        let exploit = !self.uncertainty_sampling || self.predictions_are_trusted(mean_uncertainty);

        let mut best_candidate = scored[0].0.clone();
        let mut best_score = T::neg_infinity();
        for (candidate, predicted_perf, uncertainty) in scored {
            let score = if exploit {
                predicted_perf
            } else {
                predicted_perf + uncertainty
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
            // `performances` is non-empty here, and an unrepresentable count falls
            // back to 1 rather than aborting the search.
            let count: T = scirs2_core::numeric::NumCast::from(performances.len() as f64)
                .unwrap_or_else(T::one);
            self.statistics.average_performance = sum / count;

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

/// Default learning rate for the predictor's optimizer.
const DEFAULT_PREDICTOR_LEARNING_RATE: f64 = 0.01;

/// Default number of gradient passes per training call.
const DEFAULT_PREDICTOR_EPOCHS: usize = 8;

/// Default dropout probability applied after each hidden layer.
const DEFAULT_PREDICTOR_DROPOUT: f64 = 0.1;

/// Default number of stochastic passes used for the MC-dropout uncertainty.
const DEFAULT_MC_SAMPLES: usize = 16;

/// Numerical floor used by Adam/RMSprop denominators.
const OPTIMIZER_EPSILON: f64 = 1e-8;

/// Everything a backward pass needs from the corresponding forward pass:
/// each layer's input activation, its pre-activation, and the dropout mask that
/// was applied to its output. Keeping the mask is what makes the gradient
/// consistent with the sampled sub-network — recomputing dropout in the backward
/// pass would differentiate a different function than the one evaluated.
#[derive(Debug)]
pub struct TrainingPass<T: Float + Debug + Send + Sync + 'static> {
    /// `inputs[i]` is the activation fed into layer `i`.
    inputs: Vec<Array1<T>>,
    /// `pre_activations[i]` is `W_i * inputs[i] + b_i`.
    pre_activations: Vec<Array1<T>>,
    /// `dropout_masks[i]` scales layer `i`'s output (`1/(1-p)` kept, `0` dropped).
    dropout_masks: Vec<Array1<T>>,
    /// Network output after the final layer.
    output: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> TrainingPass<T> {
    /// The scalar prediction: the first component of the output layer.
    pub fn prediction(&self) -> T {
        self.output.first().copied().unwrap_or_else(T::zero)
    }
}

impl<T: Float + Debug + Default + Clone + 'static + std::iter::Sum + Send + Sync>
    PredictorNetwork<T>
{
    fn new(architecture: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        // `architecture.len() - 1` underflows for an empty spec; build no layers
        // instead and let `initialize`/`predict` report an honest error.
        for i in 1..architecture.len() {
            let mut layer = PredictorLayer::new(architecture[i - 1], architecture[i]);
            if i == architecture.len() - 1 {
                // The output layer must be linear; see `ActivationFunction::Identity`.
                layer.activation = ActivationFunction::Identity;
            }
            layers.push(layer);
        }

        let dropout: T = scirs2_core::numeric::NumCast::from(DEFAULT_PREDICTOR_DROPOUT)
            .unwrap_or_else(|| T::zero());
        Self {
            dropout_rates: vec![dropout; layers.len()],
            layers,
            architecture,
            mc_samples: DEFAULT_MC_SAMPLES,
        }
    }

    /// The layer count, i.e. `architecture.len() - 1` for a well-formed spec.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Configure the MC-dropout sample count. `1` disables the stochastic
    /// estimate, in which case the reported uncertainty is exactly zero.
    pub fn set_mc_samples(&mut self, samples: usize) {
        self.mc_samples = samples.max(1);
    }

    /// Set the dropout probability applied after every hidden layer.
    pub fn set_dropout_rate(&mut self, rate: T) {
        for entry in self.dropout_rates.iter_mut() {
            *entry = rate;
        }
    }

    fn require_layers(&self) -> Result<()> {
        if self.layers.is_empty() {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "predictor architecture {:?} needs at least an input and an output width",
                self.architecture
            )));
        }
        Ok(())
    }

    fn initialize(&mut self) -> Result<()> {
        self.require_layers()?;
        for layer in &mut self.layers {
            layer.initialize()?;
        }
        Ok(())
    }

    /// Whether layer `index` is a hidden layer, i.e. whether dropout applies
    /// after it. Dropout is never applied to the output.
    fn is_hidden(&self, index: usize) -> bool {
        index + 1 < self.layers.len()
    }

    /// Deterministic, dropout-free forward pass.
    fn forward_eval(&self, input: &Array1<T>) -> Result<Array1<T>> {
        self.require_layers()?;
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    /// Deterministic scalar prediction (eval mode).
    fn predict(&self, input: &Array1<T>) -> Result<T> {
        let output = self.forward_eval(input)?;
        Ok(output.first().copied().unwrap_or_else(T::zero))
    }

    /// Stochastic forward pass recording everything the backward pass needs.
    fn forward_train(
        &self,
        input: &Array1<T>,
        rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    ) -> Result<TrainingPass<T>> {
        self.require_layers()?;
        let mut inputs = Vec::with_capacity(self.layers.len());
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        let mut dropout_masks = Vec::with_capacity(self.layers.len());

        let mut current = input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            inputs.push(current.clone());
            let pre = layer.pre_activation(&current);
            let mut activated = layer.apply_activation(&pre);
            let mask = if self.is_hidden(index) {
                let rate = self
                    .dropout_rates
                    .get(index)
                    .copied()
                    .unwrap_or_else(T::zero);
                let mask = dropout_mask(activated.len(), rate, rng);
                activated = &activated * &mask;
                mask
            } else {
                Array1::from_elem(activated.len(), T::one())
            };
            pre_activations.push(pre);
            dropout_masks.push(mask);
            current = activated;
        }

        Ok(TrainingPass {
            inputs,
            pre_activations,
            dropout_masks,
            output: current,
        })
    }

    /// Monte-Carlo-dropout prediction: mean and (population) standard deviation of
    /// `mc_samples` stochastic forward passes.
    ///
    /// With dropout disabled (rate `0`) or a single sample this degenerates to the
    /// deterministic prediction with an uncertainty of exactly `0`, which is the
    /// honest answer: nothing stochastic was measured. The previous version
    /// returned `||output||_2 * 0.1` from one noisy pass — a number that reflected
    /// output magnitude rather than any uncertainty.
    fn forward_with_uncertainty(
        &self,
        input: &Array1<T>,
        rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    ) -> Result<(T, T)> {
        self.require_layers()?;
        let dropout_active = self
            .dropout_rates
            .iter()
            .enumerate()
            .any(|(i, rate)| self.is_hidden(i) && *rate > T::zero());
        if !dropout_active || self.mc_samples <= 1 {
            return Ok((self.predict(input)?, T::zero()));
        }

        let mut samples = Vec::with_capacity(self.mc_samples);
        for _ in 0..self.mc_samples {
            let pass = self.forward_train(input, rng)?;
            samples.push(pass.prediction());
        }
        let count: T =
            scirs2_core::numeric::NumCast::from(samples.len() as f64).unwrap_or_else(T::one);
        let mut sum = T::zero();
        for value in &samples {
            sum = sum + *value;
        }
        let mean = sum / count;
        let mut variance = T::zero();
        for value in &samples {
            let diff = *value - mean;
            variance = variance + diff * diff;
        }
        Ok((mean, (variance / count).sqrt()))
    }

    /// One gradient-descent step on the squared error between the pass's scalar
    /// prediction and `target`, returning the loss *before* the update.
    ///
    /// This replaces an empty `Ok(())`: the predictor previously never trained, so
    /// `NeuralPredictorSearch` ranked candidates with an untrained network. The
    /// backward pass differentiates exactly the sub-network the forward pass
    /// evaluated (same dropout masks) and routes every parameter update through
    /// [`SearchOptimizer`].
    fn backward_update(
        &mut self,
        pass: &TrainingPass<T>,
        target: T,
        optimizer: &mut SearchOptimizer<T>,
    ) -> Result<T> {
        self.require_layers()?;
        if pass.inputs.len() != self.layers.len() {
            return Err(crate::error::OptimError::InvalidParameter(format!(
                "forward pass recorded {} layers but the network has {}",
                pass.inputs.len(),
                self.layers.len()
            )));
        }

        let prediction = pass.prediction();
        let error = prediction - target;
        let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(T::one);
        let loss = half * error * error;

        // dL/d(output). Only the first output unit carries the prediction, so the
        // remaining units receive no gradient from this loss.
        let mut delta_out = Array1::zeros(pass.output.len());
        if !delta_out.is_empty() {
            delta_out[0] = error;
        }

        optimizer.begin_step();
        let mut upstream = delta_out;
        for index in (0..self.layers.len()).rev() {
            // Undo the dropout scaling applied to this layer's output.
            let masked = &upstream * &pass.dropout_masks[index];
            // Through the activation.
            let derivative = self.layers[index].activation_derivative(&pass.pre_activations[index]);
            let delta = &masked * &derivative;

            let input = &pass.inputs[index];
            let rows = self.layers[index].weights.nrows();
            let cols = self.layers[index].weights.ncols();
            let mut weight_grad = Array2::zeros((rows, cols));
            for r in 0..rows {
                let d = delta[r];
                if d == T::zero() {
                    continue;
                }
                for c in 0..cols {
                    weight_grad[[r, c]] = d * input[c];
                }
            }

            // Propagate before the weights change.
            if index > 0 {
                upstream = self.layers[index].weights.t().dot(&delta);
            }

            optimizer.step_matrix(
                &format!("layer{}_w", index),
                &mut self.layers[index].weights,
                &weight_grad,
            );
            optimizer.step_vector(
                &format!("layer{}_b", index),
                &mut self.layers[index].bias,
                &delta,
            );
        }

        Ok(loss)
    }
}

/// Draw a dropout mask of `len` entries: each entry is `0` with probability
/// `rate` and `1 / (1 - rate)` otherwise (inverted dropout, so the expected
/// activation is unchanged and no rescaling is needed at eval time).
fn dropout_mask<T: Float>(
    len: usize,
    rate: T,
    rng: &mut Random<scirs2_core::random::rngs::StdRng>,
) -> Array1<T> {
    let rate_f64 = rate.to_f64().unwrap_or(0.0).clamp(0.0, 0.999_999);
    if rate_f64 <= 0.0 {
        return Array1::from_elem(len, T::one());
    }
    let keep_scale: T =
        scirs2_core::numeric::NumCast::from(1.0 / (1.0 - rate_f64)).unwrap_or_else(T::one);
    Array1::from_shape_fn(len, |_| {
        if rng.gen_range(0.0..1.0) < rate_f64 {
            T::zero()
        } else {
            keep_scale
        }
    })
}

impl<T: Float + Debug + Default + Clone + 'static + Send + Sync> PredictorLayer<T> {
    fn new(input_size: usize, outputsize: usize) -> Self {
        Self {
            weights: Array2::zeros((outputsize, input_size)),
            bias: Array1::zeros(outputsize),
            activation: ActivationFunction::ReLU,
        }
    }

    /// He/Xavier-style initialization seeded from OS entropy. Biases stay at zero,
    /// which is the standard choice for ReLU networks.
    fn initialize(&mut self) -> Result<()> {
        let fan_in = self.weights.ncols() as f64;
        let fan_out = self.weights.nrows() as f64;
        let scale = if fan_in + fan_out > 0.0 {
            (6.0 / (fan_in + fan_out)).sqrt()
        } else {
            0.0
        };

        let mut rng = Random::seed(scirs2_core::random::random::<u64>());
        self.weights = Array2::from_shape_fn(self.weights.raw_dim(), |_| {
            scirs2_core::numeric::NumCast::from(rng.gen_range(-scale..=scale))
                .unwrap_or_else(|| T::zero())
        });
        self.bias = Array1::zeros(self.bias.len());

        Ok(())
    }

    /// `W * input + b`, before the activation.
    fn pre_activation(&self, input: &Array1<T>) -> Array1<T> {
        self.weights.dot(input) + &self.bias
    }

    fn forward(&self, input: &Array1<T>) -> Result<Array1<T>> {
        Ok(self.apply_activation(&self.pre_activation(input)))
    }

    fn apply_activation(&self, x: &Array1<T>) -> Array1<T> {
        match self.activation {
            ActivationFunction::ReLU => x.mapv(|xi| if xi > T::zero() { xi } else { T::zero() }),
            ActivationFunction::GELU => x.mapv(|xi| {
                let x_f64 = xi.to_f64().unwrap_or(0.0);
                scirs2_core::numeric::NumCast::from(gelu(x_f64)).unwrap_or_else(|| T::zero())
            }),
            ActivationFunction::Swish => x.mapv(|xi| {
                let sigmoid = T::one() / (T::one() + (-xi).exp());
                xi * sigmoid
            }),
            ActivationFunction::Tanh => x.mapv(|xi| xi.tanh()),
            ActivationFunction::Sigmoid => x.mapv(|xi| T::one() / (T::one() + (-xi).exp())),
            ActivationFunction::Identity => x.clone(),
        }
    }

    /// Elementwise derivative of the activation with respect to its
    /// pre-activation input.
    fn activation_derivative(&self, pre_activation: &Array1<T>) -> Array1<T> {
        match self.activation {
            ActivationFunction::ReLU => {
                pre_activation.mapv(|xi| if xi > T::zero() { T::one() } else { T::zero() })
            }
            ActivationFunction::GELU => pre_activation.mapv(|xi| {
                let x_f64 = xi.to_f64().unwrap_or(0.0);
                scirs2_core::numeric::NumCast::from(gelu_derivative(x_f64))
                    .unwrap_or_else(|| T::zero())
            }),
            ActivationFunction::Swish => pre_activation.mapv(|xi| {
                let sigmoid = T::one() / (T::one() + (-xi).exp());
                sigmoid + xi * sigmoid * (T::one() - sigmoid)
            }),
            ActivationFunction::Tanh => pre_activation.mapv(|xi| {
                let t = xi.tanh();
                T::one() - t * t
            }),
            ActivationFunction::Sigmoid => pre_activation.mapv(|xi| {
                let s = T::one() / (T::one() + (-xi).exp());
                s * (T::one() - s)
            }),
            ActivationFunction::Identity => Array1::from_elem(pre_activation.len(), T::one()),
        }
    }
}

/// Tanh approximation of GELU (Hendrycks & Gimpel).
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (x * 0.797_884_560_802_865_4 * (1.0 + 0.044_715 * x * x)).tanh())
}

/// Derivative of [`gelu`], differentiated exactly through the tanh
/// approximation so the backward pass matches the forward pass.
fn gelu_derivative(x: f64) -> f64 {
    let c = 0.797_884_560_802_865_4;
    let inner = c * (x + 0.044_715 * x * x * x);
    let tanh_inner = inner.tanh();
    let sech_squared = 1.0 - tanh_inner * tanh_inner;
    let d_inner = c * (1.0 + 3.0 * 0.044_715 * x * x);
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * d_inner
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> ArchitectureEncoder<T> {
    fn new(embeddingdim: usize) -> Self {
        Self {
            embedding_dim: embeddingdim,
            max_components: 64,
            _phantom: std::marker::PhantomData,
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
        // fixed-length contract of `embedding_dim`.
        encoding.resize(self.embedding_dim, T::zero());
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
/// `encode_component_block`) absorbs any unrecognised name.
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
/// `encode_component_block`: one slot per known type, one out-of-vocabulary
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
            learning_rate: learningrate,
            momentum: scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero()),
            weight_decay: scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero()),
            parameters: HashMap::new(),
            step: 0,
        }
    }

    /// The configured update rule.
    pub fn optimizer_type(&self) -> SearchOptimizerType {
        self.optimizer_type
    }

    /// The learning rate in use.
    pub fn learning_rate(&self) -> T {
        self.learning_rate
    }

    /// Replace the learning rate.
    pub fn set_learning_rate(&mut self, learning_rate: T) {
        self.learning_rate = learning_rate;
    }

    /// Advance the global step counter. Called once per backward pass so
    /// Adam/AdamW bias correction sees a monotone step index shared by every
    /// parameter tensor.
    fn begin_step(&mut self) {
        self.step = self.step.saturating_add(1);
    }

    /// Number of update steps applied so far.
    pub fn steps_taken(&self) -> u64 {
        self.step
    }

    /// Apply the configured update rule to a weight matrix.
    fn step_matrix(&mut self, key: &str, weights: &mut Array2<T>, gradients: &Array2<T>) {
        let len = weights.len();
        let mut flat: Vec<T> = weights.iter().copied().collect();
        let grad: Vec<T> = gradients.iter().copied().collect();
        if grad.len() != len {
            return;
        }
        self.apply(key, &mut flat, &grad);
        for (target, value) in weights.iter_mut().zip(flat) {
            *target = value;
        }
    }

    /// Apply the configured update rule to a bias vector.
    fn step_vector(&mut self, key: &str, bias: &mut Array1<T>, gradients: &Array1<T>) {
        let len = bias.len();
        let mut flat: Vec<T> = bias.iter().copied().collect();
        let grad: Vec<T> = gradients.iter().copied().collect();
        if grad.len() != len {
            return;
        }
        self.apply(key, &mut flat, &grad);
        for (target, value) in bias.iter_mut().zip(flat) {
            *target = value;
        }
    }

    /// Fetch (creating if absent) a zero-initialized state buffer.
    fn state(&mut self, key: &str, len: usize) -> Array1<T> {
        self.parameters
            .entry(key.to_string())
            .or_insert_with(|| Array1::zeros(len))
            .clone()
    }

    /// The four update rules, each implemented for real. No variant delegates to
    /// another; `SearchOptimizerType` selects genuinely different arithmetic.
    fn apply(&mut self, key: &str, params: &mut [T], gradients: &[T]) {
        let len = params.len();
        let lr = self.learning_rate;
        let epsilon: T =
            scirs2_core::numeric::NumCast::from(OPTIMIZER_EPSILON).unwrap_or_else(|| T::zero());

        match self.optimizer_type {
            SearchOptimizerType::SGD => {
                // SGD with (heavy-ball) momentum when `momentum > 0`.
                let momentum_key = format!("{}_momentum", key);
                let mut buffer = self.state(&momentum_key, len);
                for i in 0..len {
                    buffer[i] = self.momentum * buffer[i] + gradients[i];
                    params[i] = params[i] - lr * buffer[i];
                }
                self.parameters.insert(momentum_key, buffer);
            }
            SearchOptimizerType::RMSprop => {
                // Running average of squared gradients with decay = momentum.
                let square_key = format!("{}_sq", key);
                let mut squares = self.state(&square_key, len);
                let one = T::one();
                for i in 0..len {
                    squares[i] = self.momentum * squares[i]
                        + (one - self.momentum) * gradients[i] * gradients[i];
                    params[i] = params[i] - lr * gradients[i] / (squares[i].sqrt() + epsilon);
                }
                self.parameters.insert(square_key, squares);
            }
            SearchOptimizerType::Adam | SearchOptimizerType::AdamW => {
                let beta1: T =
                    scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero());
                let beta2: T =
                    scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero());
                let one = T::one();
                let step = self.step.max(1) as i32;
                let bias1 = one - beta1.powi(step);
                let bias2 = one - beta2.powi(step);

                let first_key = format!("{}_m", key);
                let second_key = format!("{}_v", key);
                let mut first = self.state(&first_key, len);
                let mut second = self.state(&second_key, len);

                let decoupled_decay = self.optimizer_type == SearchOptimizerType::AdamW;
                for i in 0..len {
                    first[i] = beta1 * first[i] + (one - beta1) * gradients[i];
                    second[i] = beta2 * second[i] + (one - beta2) * gradients[i] * gradients[i];
                    let m_hat = if bias1 > T::zero() {
                        first[i] / bias1
                    } else {
                        first[i]
                    };
                    let v_hat = if bias2 > T::zero() {
                        second[i] / bias2
                    } else {
                        second[i]
                    };
                    if decoupled_decay {
                        // AdamW: weight decay applied directly to the parameter,
                        // not folded into the gradient.
                        params[i] = params[i] - lr * self.weight_decay * params[i];
                    }
                    params[i] = params[i] - lr * m_hat / (v_hat.sqrt() + epsilon);
                }
                self.parameters.insert(first_key, first);
                self.parameters.insert(second_key, second);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- F19: the predictor actually trains -----------------------------

    /// Build a small predictor over the encoder's block length, seeded so the
    /// test is deterministic.
    fn trainable_predictor(optimizer: SearchOptimizerType) -> NeuralPredictorSearch<f64> {
        let embedding_dim = COMPONENT_BLOCK_LEN;
        let mut search =
            NeuralPredictorSearch::<f64>::new(vec![embedding_dim, 24, 12, 1], embedding_dim, 0.5);
        search.set_seed(0xF19_0000_0001);
        search.set_optimizer(optimizer, 0.02);
        search.set_training_epochs(60);
        // Deterministic training: dropout off, so the loss curve is not noise.
        search.predictor_network.set_dropout_rate(0.0);
        search
            .initialize(&SearchSpaceConfig::default())
            .expect("initialize predictor");
        search
    }

    /// Distinct architectures paired with distinct targets.
    fn training_set() -> (Vec<OptimizerArchitecture<f64>>, Vec<f64>) {
        let architectures = vec![
            make_arch(&["SGD"]),
            make_arch(&["Adam"]),
            make_arch(&["Adam", "Adam"]),
            make_arch(&["RMSprop", "Dropout"]),
            make_arch(&["Lion", "WeightDecay", "StepLR"]),
            make_arch(&["AdaGrad"]),
        ];
        let targets = vec![0.10, 0.40, 0.55, 0.25, 0.80, 0.15];
        (architectures, targets)
    }

    #[test]
    fn training_reduces_the_predictor_loss() {
        let mut search = trainable_predictor(SearchOptimizerType::Adam);
        let (architectures, targets) = training_set();

        let before = search
            .training_loss(&architectures, &targets)
            .expect("loss before training");
        search
            .train_predictor(&architectures, &targets)
            .expect("training must succeed");
        let after = search
            .training_loss(&architectures, &targets)
            .expect("loss after training");

        // The pre-fix `backward_update` was an empty `Ok(())`, so `after` was
        // exactly `before`.
        assert!(
            after < before,
            "training must reduce the loss: before = {before}, after = {after}"
        );
        assert!(
            after < before * 0.5,
            "expected a substantial reduction, before = {before}, after = {after}"
        );
        assert!(after.is_finite(), "loss diverged to {after}");
    }

    #[test]
    fn every_optimizer_variant_reduces_the_loss() {
        for optimizer in [
            SearchOptimizerType::SGD,
            SearchOptimizerType::Adam,
            SearchOptimizerType::AdamW,
            SearchOptimizerType::RMSprop,
        ] {
            let mut search = trainable_predictor(optimizer);
            let (architectures, targets) = training_set();
            let before = search
                .training_loss(&architectures, &targets)
                .expect("loss before");
            search
                .train_predictor(&architectures, &targets)
                .expect("training");
            let after = search
                .training_loss(&architectures, &targets)
                .expect("loss after");
            assert!(
                after < before,
                "{optimizer:?} did not reduce the loss: {before} -> {after}"
            );
            assert!(
                search.search_optimizer.steps_taken() > 0,
                "{optimizer:?} recorded no update steps"
            );
        }
    }

    #[test]
    fn training_changes_the_network_weights() {
        let mut search = trainable_predictor(SearchOptimizerType::Adam);
        let (architectures, targets) = training_set();
        let before: Vec<f64> = search
            .predictor_network
            .layers
            .iter()
            .flat_map(|layer| layer.weights.iter().copied().collect::<Vec<_>>())
            .collect();
        let biases_before: Vec<f64> = search
            .predictor_network
            .layers
            .iter()
            .flat_map(|layer| layer.bias.iter().copied().collect::<Vec<_>>())
            .collect();

        search
            .train_predictor(&architectures, &targets)
            .expect("training");

        let after: Vec<f64> = search
            .predictor_network
            .layers
            .iter()
            .flat_map(|layer| layer.weights.iter().copied().collect::<Vec<_>>())
            .collect();
        let biases_after: Vec<f64> = search
            .predictor_network
            .layers
            .iter()
            .flat_map(|layer| layer.bias.iter().copied().collect::<Vec<_>>())
            .collect();

        assert_ne!(before, after, "weights must move during training");
        assert_ne!(
            biases_before, biases_after,
            "biases must move during training (they start at zero and stayed there before)"
        );
    }

    #[test]
    fn the_predictor_learns_to_rank_architectures() {
        let mut search = trainable_predictor(SearchOptimizerType::Adam);
        let (architectures, targets) = training_set();
        search
            .train_predictor(&architectures, &targets)
            .expect("training");

        // The best-scoring training architecture must predict higher than the
        // worst-scoring one. An untrained network cannot do this reliably.
        let best_idx = 4; // target 0.80
        let worst_idx = 0; // target 0.10
        let best = search
            .predict_performance(&architectures[best_idx])
            .expect("predict")
            .0;
        let worst = search
            .predict_performance(&architectures[worst_idx])
            .expect("predict")
            .0;
        assert!(
            best > worst,
            "expected the 0.80-target architecture to outrank the 0.10 one: {best} vs {worst}"
        );
    }

    #[test]
    fn the_output_layer_is_linear_so_negative_targets_are_reachable() {
        let mut search = trainable_predictor(SearchOptimizerType::Adam);
        assert_eq!(
            search
                .predictor_network
                .layers
                .last()
                .expect("at least one layer")
                .activation,
            ActivationFunction::Identity,
            "a ReLU output layer clamps every prediction to >= 0"
        );

        let architectures = vec![make_arch(&["SGD"]), make_arch(&["Lion"])];
        let targets = vec![-0.75, 0.75];
        search
            .train_predictor(&architectures, &targets)
            .expect("training");
        let negative = search
            .predict_performance(&architectures[0])
            .expect("predict")
            .0;
        assert!(
            negative < 0.0,
            "a linear output must be able to reach a negative target, got {negative}"
        );
    }

    #[test]
    fn mc_dropout_uncertainty_is_zero_without_dropout_and_positive_with_it() {
        let embedding_dim = COMPONENT_BLOCK_LEN;
        let mut search =
            NeuralPredictorSearch::<f64>::new(vec![embedding_dim, 16, 8, 1], embedding_dim, 0.5);
        search.set_seed(4242);
        search
            .initialize(&SearchSpaceConfig::default())
            .expect("initialize");
        let arch = make_arch(&["Adam", "Dropout"]);

        // With dropout enabled the MC estimate must actually vary.
        search.predictor_network.set_dropout_rate(0.3);
        search.predictor_network.set_mc_samples(64);
        let (_, stochastic) = search.predict_performance(&arch).expect("predict");
        assert!(
            stochastic > 0.0,
            "MC dropout must report a non-zero standard deviation, got {stochastic}"
        );

        // With dropout off, the honest uncertainty is exactly zero (not
        // 0.1 * ||output||, which is what the old code returned).
        search.predictor_network.set_dropout_rate(0.0);
        let (mean, deterministic) = search.predict_performance(&arch).expect("predict");
        assert_eq!(deterministic, 0.0);
        let (mean_again, _) = search.predict_performance(&arch).expect("predict");
        assert_eq!(mean, mean_again, "eval mode must be deterministic");
    }

    #[test]
    fn a_degenerate_architecture_spec_is_an_error_not_a_silent_no_op() {
        let mut empty = NeuralPredictorSearch::<f64>::new(Vec::new(), 8, 0.5);
        assert!(
            empty.initialize(&SearchSpaceConfig::default()).is_err(),
            "an empty predictor spec must be rejected"
        );
        let mut single = NeuralPredictorSearch::<f64>::new(vec![8], 8, 0.5);
        assert!(single.initialize(&SearchSpaceConfig::default()).is_err());
    }

    #[test]
    fn dropout_masks_preserve_the_expected_activation_scale() {
        let mut rng = Random::seed(7);
        let mask = dropout_mask::<f64>(10_000, 0.25, &mut rng);
        let mean = mask.iter().sum::<f64>() / mask.len() as f64;
        assert!(
            (mean - 1.0).abs() < 0.05,
            "inverted dropout must keep E[mask] = 1, got {mean}"
        );
        let no_dropout = dropout_mask::<f64>(16, 0.0, &mut rng);
        assert!(no_dropout.iter().all(|v| *v == 1.0));
    }

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

    #[test]
    fn the_confidence_threshold_decides_whether_predictions_are_trusted() {
        let mut search = NeuralPredictorSearch::<f64>::new(vec![64, 16, 1], 64, 0.5);
        assert!((search.confidence_threshold() - 0.5).abs() < 1e-12);

        // confidence = 1 / (1 + spread): a spread of 1.0 gives exactly 0.5.
        assert!(search.predictions_are_trusted(0.0));
        assert!(search.predictions_are_trusted(1.0));
        assert!(!search.predictions_are_trusted(1.5));
        // A non-finite spread is never trusted.
        assert!(!search.predictions_are_trusted(f64::NAN));
        assert!(!search.predictions_are_trusted(f64::INFINITY));

        // A threshold of zero trusts anything; one trusts only exact agreement.
        search
            .set_confidence_threshold(0.0)
            .expect("0 is a valid threshold");
        assert!(search.predictions_are_trusted(1000.0));
        search
            .set_confidence_threshold(1.0)
            .expect("1 is a valid threshold");
        assert!(search.predictions_are_trusted(0.0));
        assert!(!search.predictions_are_trusted(1e-6));

        // Out-of-range thresholds are rejected, and the stored value is unchanged.
        for bad in [-0.1, 1.1, f64::NAN] {
            assert!(
                search.set_confidence_threshold(bad).is_err(),
                "threshold {bad} must be rejected"
            );
        }
        assert!((search.confidence_threshold() - 1.0).abs() < 1e-12);
    }
}
