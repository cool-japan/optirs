// Bayesian optimization for architecture search

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::RngExt;
use std::collections::VecDeque;
use std::fmt::Debug;

#[allow(unused_imports)]
use crate::error::Result;
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::random::RandomSearch;
use super::{SearchStrategy, SearchStrategyStatistics};

/// Bayesian optimization for architecture search
pub struct BayesianOptimization<T: Float + Debug + Send + Sync + 'static> {
    gaussian_process: GaussianProcess<T>,
    acquisition_function: AcquisitionFunction<T>,
    observed_architectures: Vec<OptimizerArchitecture<T>>,
    observed_performances: Vec<T>,
    kernel: GPKernel<T>,
    statistics: SearchStrategyStatistics<T>,
    exploration_factor: T,
}

/// Gaussian Process for Bayesian optimization
#[derive(Debug)]
pub struct GaussianProcess<T: Float + Debug + Send + Sync + 'static> {
    kernel_matrix: Array2<T>,
    inverse_kernel: Array2<T>,
    noise_variance: T,
    length_scales: Array1<T>,
    signal_variance: T,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug)]
pub struct AcquisitionFunction<T: Float + Debug + Send + Sync + 'static> {
    function_type: AcquisitionType,
    explorationweight: T,
    current_best: T,
}

/// Types of acquisition functions
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionType {
    /// Expected Improvement
    EI,
    /// Upper Confidence Bound
    UCB,
    /// Probability of Improvement
    PI,
    /// Thompson Sampling
    Thompson,
    /// Information Gain
    InfoGain,
}

/// Gaussian Process kernels
#[derive(Debug)]
pub struct GPKernel<T: Float + Debug + Send + Sync + 'static> {
    _kerneltype: KernelType,
    hyperparameters: Array1<T>,
}

/// Kernel types for GP
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    RBF,
    Matern32,
    Matern52,
    Linear,
    Polynomial,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + std::iter::Sum>
    BayesianOptimization<T>
{
    pub fn new(
        kerneltype: KernelType,
        acquisition_type: AcquisitionType,
        exploration_factor: f64,
    ) -> Self {
        Self {
            gaussian_process: GaussianProcess::new(kerneltype),
            acquisition_function: AcquisitionFunction::new(
                acquisition_type,
                scirs2_core::numeric::NumCast::from(exploration_factor)
                    .unwrap_or_else(|| T::zero()),
            ),
            observed_architectures: Vec::new(),
            observed_performances: Vec::new(),
            kernel: GPKernel::new(kerneltype),
            statistics: SearchStrategyStatistics::default(),
            exploration_factor: scirs2_core::numeric::NumCast::from(exploration_factor)
                .unwrap_or_else(|| T::zero()),
        }
    }

    fn encode_architecture(&self, architecture: &OptimizerArchitecture<T>) -> Array1<T> {
        // Deterministic, vocabulary-based encoding of the architecture's
        // component types, followed by its hyperparameter values.  Each
        // component string (produced elsewhere via
        // `format!("{:?}", ComponentType)`) is mapped onto a fixed, ordered
        // vocabulary and accumulated into a multi-hot block.  Unlike a raw
        // string hash this preserves locality and gives the Gaussian-process
        // kernel a meaningful, stable feature, so similar architectures map to
        // nearby points in the input space.
        let mut encoding = encode_component_block::<T, _>(architecture.components.iter());

        // Encode hyperparameters from the architecture's hyperparameters map.
        // Sorted by key so the appended values are order-stable regardless of
        // the underlying hash-map iteration order.
        let mut hyperparameters: Vec<(&String, &T)> = architecture.hyperparameters.iter().collect();
        hyperparameters.sort_by(|a, b| a.0.cmp(b.0));
        for (_key, value) in hyperparameters {
            encoding.push(*value);
        }

        // Pad (or truncate) to the fixed GP input dimension.  This preserves
        // the original 64-length contract consumed by the kernel/GP.
        encoding.resize(64, T::zero());
        Array1::from_vec(encoding)
    }

    fn fit_gp(&mut self) -> Result<()> {
        if self.observed_architectures.len() < 2 {
            return Ok(());
        }

        // Encode all observed architectures
        let encoded_archs: Vec<Array1<T>> = self
            .observed_architectures
            .iter()
            .map(|arch| self.encode_architecture(arch))
            .collect();

        // Fit Gaussian Process
        self.gaussian_process
            .fit(&encoded_archs, &self.observed_performances)?;

        Ok(())
    }

    fn suggest_next_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        if self.observed_architectures.len() < 5 {
            // Use random search for initial points
            let mut random_search = RandomSearch::<T>::new(Some(42));
            random_search.initialize(searchspace)?;
            return random_search.generate_architecture(searchspace, &VecDeque::new());
        }

        // Generate candidate architectures
        let num_candidates = 100;
        let mut candidates = Vec::new();
        let mut random_search = RandomSearch::<T>::new(Some(42));
        random_search.initialize(searchspace)?;

        for _ in 0..num_candidates {
            candidates.push(random_search.generate_architecture(searchspace, &VecDeque::new())?);
        }

        // Evaluate acquisition function for each candidate
        let mut best_architecture = candidates[0].clone();
        let mut best_acquisition = T::neg_infinity();

        for candidate in candidates {
            let encoded = self.encode_architecture(&candidate);
            let (mean, variance) = self.gaussian_process.predict(&encoded)?;
            let acquisition_value = self.acquisition_function.evaluate(mean, variance);

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_architecture = candidate;
            }
        }

        Ok(best_architecture)
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + std::iter::Sum>
    SearchStrategy<T> for BayesianOptimization<T>
{
    fn initialize(&mut self, _searchspace: &SearchSpaceConfig) -> Result<()> {
        self.observed_architectures.clear();
        self.observed_performances.clear();
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        let architecture = self.suggest_next_architecture(searchspace)?;
        self.statistics.total_architectures_generated += 1;
        Ok(architecture)
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        for result in results {
            if let Some(&performance) = result
                .evaluation_results
                .metric_scores
                .get(&EvaluationMetric::FinalPerformance)
            {
                self.observed_architectures
                    .push(result.architecture.clone());
                self.observed_performances.push(performance);

                // Update statistics
                if performance > self.statistics.best_performance {
                    self.statistics.best_performance = performance;
                }

                // Refit GP
                self.fit_gp()?;
            }
        }

        // Update average performance
        if !self.observed_performances.is_empty() {
            let sum: T = self.observed_performances.iter().cloned().sum();
            self.statistics.average_performance = sum
                / T::from(self.observed_performances.len())
                    .expect("conversion from usize to T failed");
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "BayesianOptimization"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate = self.exploration_factor;
        stats.exploitation_rate = T::one() - self.exploration_factor;
        stats
    }
}

// Implementation stubs for supporting components
impl<T: Float + Debug + Default + Send + Sync> GaussianProcess<T> {
    fn new(_kerneltype: KernelType) -> Self {
        Self {
            kernel_matrix: Array2::zeros((0, 0)),
            inverse_kernel: Array2::zeros((0, 0)),
            noise_variance: scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero()),
            length_scales: Array1::ones(1),
            signal_variance: T::one(),
        }
    }

    fn fit(&mut self, _x: &[Array1<T>], _y: &[T]) -> Result<()> {
        // Simplified GP fitting
        Ok(())
    }

    fn predict(&self, _x: &Array1<T>) -> Result<(T, T)> {
        // Simplified prediction - return mean and variance
        Ok((
            scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()),
            scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
        ))
    }
}

impl<T: Float + Debug + Default + Send + Sync> AcquisitionFunction<T> {
    fn new(function_type: AcquisitionType, explorationweight: T) -> Self {
        Self {
            function_type,
            explorationweight,
            current_best: T::zero(),
        }
    }

    fn evaluate(&self, mean: T, variance: T) -> T {
        match self.function_type {
            AcquisitionType::UCB => mean + self.explorationweight * variance.sqrt(),
            AcquisitionType::EI => {
                // Simplified Expected Improvement
                let std_dev = variance.sqrt();
                if std_dev > scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero())
                {
                    let z = (mean - self.current_best) / std_dev;
                    // Simplified calculation without proper CDF/PDF
                    z * std_dev
                } else {
                    T::zero()
                }
            }
            AcquisitionType::PI => {
                // Simplified Probability of Improvement
                if variance > scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero())
                {
                    let z = (mean - self.current_best) / variance.sqrt();
                    // Simplified - would need proper CDF
                    if z > T::zero() {
                        T::one()
                    } else {
                        T::zero()
                    }
                } else {
                    T::zero()
                }
            }
            AcquisitionType::Thompson => {
                // Thompson sampling - sample from posterior
                mean + variance.sqrt()
                    * T::from(scirs2_core::random::Random::default().random::<f64>())
                        .expect("thompson sampling conversion failed")
            }
            AcquisitionType::InfoGain => {
                // Information gain - simplified as entropy
                variance.ln()
            }
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> GPKernel<T> {
    fn new(kerneltype: KernelType) -> Self {
        Self {
            _kerneltype: kerneltype,
            hyperparameters: Array1::ones(2), // length_scale and signal_variance
        }
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
///   discrete one-hot signal so the GP kernel can exploit gradients in
///   component count / family composition.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_arch(components: &[&str], hyper: &[(&str, f64)]) -> OptimizerArchitecture<f64> {
        let mut hyperparameters = HashMap::new();
        for (k, v) in hyper {
            hyperparameters.insert(k.to_string(), *v);
        }
        OptimizerArchitecture {
            components: components.iter().map(|s| s.to_string()).collect(),
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters,
            architecture_id: "test".to_string(),
        }
    }

    fn make_bo() -> BayesianOptimization<f64> {
        BayesianOptimization::<f64>::new(KernelType::RBF, AcquisitionType::UCB, 0.1)
    }

    #[test]
    fn encode_is_deterministic_and_fixed_length() {
        let bo = make_bo();
        let arch = make_arch(&["Adam", "SGD"], &[("learning_rate", 0.01), ("beta1", 0.9)]);

        let first = bo.encode_architecture(&arch);
        let second = bo.encode_architecture(&arch);

        assert_eq!(first.len(), 64);
        assert_eq!(second.len(), 64);
        assert_eq!(first, second, "encoding must be deterministic");
    }

    #[test]
    fn different_known_types_differ_same_type_matches() {
        let bo = make_bo();

        let adam = bo.encode_architecture(&make_arch(&["Adam"], &[]));
        let adam_again = bo.encode_architecture(&make_arch(&["Adam"], &[]));
        let sgd = bo.encode_architecture(&make_arch(&["SGD"], &[]));

        assert_eq!(adam, adam_again, "same type must encode identically");
        assert_ne!(adam, sgd, "different known types must differ");

        let adam_idx = component_vocab_index("Adam");
        let sgd_idx = component_vocab_index("SGD");
        assert_ne!(adam_idx, sgd_idx);
        assert_eq!(adam[adam_idx], 1.0);
        assert_eq!(sgd[sgd_idx], 1.0);
    }

    #[test]
    fn unknown_type_maps_to_oov_slot() {
        let bo = make_bo();
        let oov_index = COMPONENT_VOCABULARY.len();

        let encoded = bo.encode_architecture(&make_arch(&["NoSuchOptimizer"], &[]));

        assert_eq!(encoded.len(), 64);
        assert_eq!(
            encoded[oov_index], 1.0,
            "unknown name must land in the out-of-vocabulary slot"
        );
        for (i, value) in encoded.iter().enumerate().take(oov_index) {
            assert_eq!(*value, 0.0, "known slot {} must stay zero", i);
        }
    }

    #[test]
    fn hyperparameter_order_is_stable() {
        let bo = make_bo();
        // Same hyperparameters supplied in different insertion order must yield
        // identical encodings thanks to the key-sorted append.
        let a = bo.encode_architecture(&make_arch(
            &["Adam"],
            &[("alpha", 0.1), ("beta", 0.2), ("gamma", 0.3)],
        ));
        let b = bo.encode_architecture(&make_arch(
            &["Adam"],
            &[("gamma", 0.3), ("alpha", 0.1), ("beta", 0.2)],
        ));
        assert_eq!(a, b, "hyperparameter ordering must be stable");
    }
}
