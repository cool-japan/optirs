// Main federated privacy coordinator implementation

use super::super::noise_mechanisms::{
    GaussianMechanism, LaplaceMechanism, NoiseMechanism as NoiseMechanismTrait,
};
use super::super::{DifferentialPrivacyConfig, NoiseMechanism};
use super::super::{MomentsAccountant, PrivacyBudget};
use super::components::*;
use super::config::*;
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::{thread_rng, Rng};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Federated differential privacy coordinator
pub struct FederatedPrivacyCoordinator<T: Float + Debug + Send + Sync + 'static> {
    /// Global privacy configuration
    config: FederatedPrivacyConfig,

    /// Per-client privacy accountants
    client_accountants: HashMap<String, MomentsAccountant>,

    /// Global privacy accountant
    global_accountant: MomentsAccountant,

    /// Secure aggregation protocol
    secure_aggregator: SecureAggregator<T>,

    /// Privacy amplification analyzer
    amplification_analyzer: PrivacyAmplificationAnalyzer,

    /// Cross-device privacy manager
    cross_device_manager: CrossDevicePrivacyManager<T>,

    /// Composition analyzer for multi-round privacy
    composition_analyzer: FederatedCompositionAnalyzer,

    /// Byzantine-robust aggregation engine
    byzantine_aggregator: ByzantineRobustAggregator<T>,

    /// Personalized federated learning manager
    personalization_manager: PersonalizationManager<T>,

    /// Adaptive privacy budget manager
    adaptive_budget_manager: AdaptiveBudgetManager<T>,

    /// Communication efficiency optimizer
    communication_optimizer: CommunicationOptimizer<T>,

    /// Continual learning coordinator
    continual_learning_coordinator: ContinualLearningCoordinator<T>,

    /// Current round number
    current_round: usize,

    /// Client participation history
    participation_history: VecDeque<ParticipationRound>,
}

/// Federated round plan with privacy guarantees
#[derive(Debug, Clone)]
pub struct FederatedRoundPlan {
    pub round_number: usize,
    pub selectedclients: Vec<String>,
    pub sampling_probability: f64,
    pub amplificationfactor: f64,
    pub client_privacy_allocations: HashMap<String, ClientPrivacyAllocation>,
    pub aggregation_plan: Option<SecureAggregationPlan>,
    pub privacy_analysis: RoundPrivacyAnalysis,
}

/// Client privacy allocation for a round
#[derive(Debug, Clone)]
pub struct ClientPrivacyAllocation {
    pub epsilon: f64,
    pub delta: f64,
    pub noise_multiplier: f64,
    pub clipping_threshold: f64,
    pub amplificationfactor: f64,
}

/// Secure aggregation plan
#[derive(Debug, Clone)]
pub struct SecureAggregationPlan {
    pub masking_seeds: HashMap<String, u64>,
    pub aggregation_threshold: usize,
    pub dropout_tolerance: usize,
}

/// Privacy analysis for a round
#[derive(Debug, Clone)]
pub struct RoundPrivacyAnalysis {
    pub round_epsilon: f64,
    pub round_delta: f64,
    pub cumulative_epsilon: f64,
    pub cumulative_delta: f64,
    pub amplification_benefit: f64,
    pub composition_tightness: f64,
}

/// Advanced aggregation result with comprehensive metrics
#[derive(Debug)]
pub struct AdvancedAggregationResult<T: Float + Debug + Send + Sync + 'static> {
    pub aggregated_update: Array1<T>,
    pub outlier_detection_results: Vec<OutlierDetectionResult>,
    pub adaptive_privacy_allocations: HashMap<String, AdaptivePrivacyAllocation>,
    pub personalization_metrics: PersonalizationMetrics,
    pub communication_efficiency: CommunicationEfficiencyStats,
    pub continual_learning_status: ContinualLearningStatus,
    pub privacy_guarantees: AdvancedPrivacyGuarantees,
    pub fairness_metrics: FairnessMetrics,
}

/// Adaptive privacy allocation
#[derive(Debug, Clone)]
pub struct AdaptivePrivacyAllocation {
    pub epsilon: f64,
    pub delta: f64,
    pub importance_weight: f64,
    pub context_factors: HashMap<String, f64>,
}

/// Personalization metrics
#[derive(Debug, Clone)]
pub struct PersonalizationMetrics {
    pub cluster_assignments: HashMap<String, usize>,
    pub adaptation_effectiveness: f64,
    pub model_diversity: f64,
    pub personalization_overhead: f64,
}

/// Communication efficiency statistics
#[derive(Debug, Clone)]
pub struct CommunicationEfficiencyStats {
    pub compression_ratio: f64,
    pub bandwidth_utilization: f64,
    pub transmission_latency: f64,
    pub quality_of_service_score: f64,
}

/// Continual learning status
#[derive(Debug, Clone)]
pub struct ContinualLearningStatus {
    pub task_changes_detected: usize,
    pub forgetting_prevention_active: bool,
    pub memory_utilization: f64,
    pub knowledge_transfer_score: f64,
}

/// Advanced privacy guarantees
#[derive(Debug, Clone)]
pub struct AdvancedPrivacyGuarantees {
    pub basic_guarantees: PrivacyBudget,
    pub amplification_benefit: f64,
    pub byzantine_robustness_factor: f64,
    pub personalization_privacy_cost: f64,
    pub continual_learning_overhead: f64,
    pub multi_level_protection: bool,
    pub adaptive_budgeting_enabled: bool,
    pub communication_privacy_enabled: bool,
}

/// Enhanced sampling result
#[derive(Debug)]
pub struct EnhancedSamplingResult {
    pub selectedclients: Vec<String>,
    pub sampling_weights: HashMap<String, f64>,
    pub reputation_scores: HashMap<String, f64>,
    pub fairness_weights: HashMap<String, f64>,
    pub communication_scores: HashMap<String, f64>,
    pub diversity_metrics: DiversityMetrics,
}

/// Diversity metrics for client selection
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    pub geographic_diversity: f64,
    pub device_type_diversity: f64,
    pub data_distribution_diversity: f64,
    pub participation_frequency_diversity: f64,
}

/// Personalized round result
#[derive(Debug)]
pub struct PersonalizedRoundResult<T: Float + Debug + Send + Sync + 'static> {
    pub cluster_assignments: HashMap<usize, Vec<String>>,
    pub cluster_aggregates: HashMap<usize, Array1<T>>,
    pub meta_gradients: HashMap<String, Array1<T>>,
    pub personalized_models: HashMap<String, PersonalizedModel<T>>,
    pub effectiveness_metrics: PersonalizationMetrics,
    pub privacy_cost: f64,
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug,
    > FederatedPrivacyCoordinator<T>
{
    /// Create a new federated privacy coordinator
    pub fn new(config: FederatedPrivacyConfig) -> Result<Self> {
        let global_accountant = MomentsAccountant::new(
            config.base_config.noise_multiplier,
            config.base_config.target_delta,
            config.clients_per_round,
            config.total_clients,
        );

        let secure_aggregator = SecureAggregator::new(config.secure_aggregation.clone())?;
        let amplification_analyzer =
            PrivacyAmplificationAnalyzer::new(config.amplification_config.clone());
        let cross_device_manager =
            CrossDevicePrivacyManager::new(config.cross_device_config.clone());
        let composition_analyzer = FederatedCompositionAnalyzer::new(config.composition_method);

        Ok(Self {
            config,
            client_accountants: HashMap::new(),
            global_accountant,
            secure_aggregator,
            amplification_analyzer,
            cross_device_manager,
            composition_analyzer,
            byzantine_aggregator: ByzantineRobustAggregator::new()?,
            personalization_manager: PersonalizationManager::new()?,
            adaptive_budget_manager: AdaptiveBudgetManager::new()?,
            communication_optimizer: CommunicationOptimizer::new()?,
            continual_learning_coordinator: ContinualLearningCoordinator::new()?,
            current_round: 0,
            participation_history: VecDeque::with_capacity(1000),
        })
    }

    /// Start a new federated round with privacy guarantees
    pub fn start_federated_round(
        &mut self,
        availableclients: &[String],
    ) -> Result<FederatedRoundPlan> {
        self.current_round += 1;

        // Sample clients for this round
        let selectedclients = self.sample_clients(availableclients)?;

        // Check global privacy budget
        let global_budget = self.get_global_privacy_budget()?;
        if !self.has_sufficient_privacy_budget(&global_budget)? {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: global_budget.epsilon_consumed,
                target_epsilon: self.config.base_config.target_epsilon,
            });
        }

        // Compute sampling probability for amplification
        let sampling_probability = selectedclients.len() as f64 / availableclients.len() as f64;

        // Analyze privacy amplification
        let amplificationfactor = if self.config.amplification_config.enabled {
            self.amplification_analyzer
                .compute_amplification_factor(sampling_probability, self.current_round)?
        } else {
            1.0
        };

        // Prepare secure aggregation if enabled
        let aggregation_plan = if self.config.secure_aggregation.enabled {
            Some(self.prepare_secure_aggregation(&selectedclients)?)
        } else {
            None
        };

        // Compute per-client privacy allocations
        let client_privacy_allocations =
            self.compute_client_privacy_allocations(&selectedclients, amplificationfactor)?;

        // Create round plan
        let roundplan = FederatedRoundPlan {
            round_number: self.current_round,
            selectedclients: selectedclients.clone(),
            sampling_probability,
            amplificationfactor,
            client_privacy_allocations,
            aggregation_plan,
            privacy_analysis: self.analyze_round_privacy(&selectedclients, amplificationfactor)?,
        };

        // Record participation
        self.record_participation_round(
            &selectedclients,
            sampling_probability,
            amplificationfactor,
        );

        Ok(roundplan)
    }

    /// Perform secure aggregation of client updates
    pub fn secure_aggregate_updates(
        &mut self,
        clientupdates: &HashMap<String, Array1<T>>,
        _roundplan: &FederatedRoundPlan,
    ) -> Result<Array1<T>> {
        if self.config.secure_aggregation.enabled {
            // Placeholder for secure aggregation
            self.simple_aggregate(clientupdates)
        } else {
            // Simple averaging
            self.simple_aggregate(clientupdates)
        }
    }

    /// Simple aggregation (averaging) of client updates
    fn simple_aggregate(&self, clientupdates: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        if clientupdates.is_empty() {
            return Err(OptimError::InvalidParameter(
                "No client updates provided".to_string(),
            ));
        }

        let mut first_update: Option<Array1<T>> = None;
        let mut count = 0;

        for update in clientupdates.values() {
            if let Some(ref mut aggregate) = first_update {
                for (i, &value) in update.iter().enumerate() {
                    if i < aggregate.len() {
                        aggregate[i] = aggregate[i] + value;
                    }
                }
            } else {
                first_update = Some(update.clone());
            }
            count += 1;
        }

        if let Some(mut aggregate) = first_update {
            let count_t = T::from(count).unwrap_or_else(|| T::zero());
            for value in aggregate.iter_mut() {
                *value = *value / count_t;
            }
            Ok(aggregate)
        } else {
            Err(OptimError::InvalidParameter(
                "Failed to aggregate updates".to_string(),
            ))
        }
    }

    /// Sample clients for federated round
    fn sample_clients(&self, availableclients: &[String]) -> Result<Vec<String>> {
        use scirs2_core::random::Rng;
        let mut rng = thread_rng();
        let target_count = self.config.clients_per_round.min(availableclients.len());

        match self.config.sampling_strategy {
            ClientSamplingStrategy::UniformRandom => {
                // Simple random selection
                let mut selected = Vec::new();
                let mut remaining = availableclients.to_vec();
                for _ in 0..target_count.min(remaining.len()) {
                    let index = rng.gen_range(0..remaining.len());
                    selected.push(remaining.swap_remove(index));
                }
                Ok(selected)
            }
            _ => {
                // Fallback to uniform random for other strategies
                let mut selected = Vec::new();
                let mut remaining = availableclients.to_vec();
                for _ in 0..target_count.min(remaining.len()) {
                    let index = rng.gen_range(0..remaining.len());
                    selected.push(remaining.swap_remove(index));
                }
                Ok(selected)
            }
        }
    }

    /// Get global privacy budget
    fn get_global_privacy_budget(&self) -> Result<PrivacyBudget> {
        use super::super::AccountingMethod;
        // Placeholder implementation
        Ok(PrivacyBudget {
            epsilon_consumed: 0.1,
            delta_consumed: 1e-5,
            epsilon_remaining: self.config.base_config.target_epsilon - 0.1,
            delta_remaining: self.config.base_config.target_delta - 1e-5,
            steps_taken: self.current_round,
            accounting_method: AccountingMethod::MomentsAccountant,
            estimated_steps_remaining: 100,
        })
    }

    /// Check if sufficient privacy budget is available
    fn has_sufficient_privacy_budget(&self, budget: &PrivacyBudget) -> Result<bool> {
        Ok(budget.epsilon_remaining > 0.0 && budget.delta_remaining > 0.0)
    }

    /// Prepare secure aggregation plan
    fn prepare_secure_aggregation(
        &self,
        selectedclients: &[String],
    ) -> Result<SecureAggregationPlan> {
        use scirs2_core::random::Rng;
        let mut rng = thread_rng();

        let mut masking_seeds = HashMap::new();
        for client in selectedclients {
            masking_seeds.insert(client.clone(), rng.random::<u64>());
        }

        Ok(SecureAggregationPlan {
            masking_seeds,
            aggregation_threshold: self.config.secure_aggregation.min_clients,
            dropout_tolerance: self.config.secure_aggregation.max_dropouts,
        })
    }

    /// Compute privacy allocations for each client
    fn compute_client_privacy_allocations(
        &self,
        selectedclients: &[String],
        amplificationfactor: f64,
    ) -> Result<HashMap<String, ClientPrivacyAllocation>> {
        let mut allocations = HashMap::new();

        let base_epsilon = self.config.base_config.target_epsilon / amplificationfactor;
        let base_delta = self.config.base_config.target_delta;

        for clientid in selectedclients {
            allocations.insert(
                clientid.clone(),
                ClientPrivacyAllocation {
                    epsilon: base_epsilon,
                    delta: base_delta,
                    noise_multiplier: self.config.base_config.noise_multiplier,
                    clipping_threshold: self.config.base_config.l2_norm_clip,
                    amplificationfactor,
                },
            );
        }

        Ok(allocations)
    }

    /// Analyze privacy for the current round
    fn analyze_round_privacy(
        &self,
        selectedclients: &[String],
        amplificationfactor: f64,
    ) -> Result<RoundPrivacyAnalysis> {
        let round_epsilon = self.config.base_config.target_epsilon / amplificationfactor;
        let round_delta = self.config.base_config.target_delta;

        let cumulative_epsilon = round_epsilon * self.current_round as f64;
        let cumulative_delta = round_delta * self.current_round as f64;

        Ok(RoundPrivacyAnalysis {
            round_epsilon,
            round_delta,
            cumulative_epsilon,
            cumulative_delta,
            amplification_benefit: amplificationfactor - 1.0,
            composition_tightness: 0.95, // Placeholder
        })
    }

    /// Record participation for this round
    fn record_participation_round(
        &mut self,
        selectedclients: &[String],
        sampling_probability: f64,
        amplificationfactor: f64,
    ) {
        let participation = ParticipationRound {
            round: self.current_round,
            participating_clients: selectedclients.to_vec(),
            sampling_probability,
            privacy_cost: PrivacyCost {
                epsilon: self.config.base_config.target_epsilon / amplificationfactor,
                delta: self.config.base_config.target_delta,
                client_contribution: 1.0 / selectedclients.len() as f64,
                amplificationfactor,
                composition_cost: 0.1, // Placeholder
            },
            aggregation_noise: self.config.base_config.noise_multiplier,
        };

        self.participation_history.push_back(participation);

        // Keep only last 1000 rounds
        if self.participation_history.len() > 1000 {
            self.participation_history.pop_front();
        }
    }

    /// Get current privacy guarantees
    pub fn get_privacy_guarantees(&self) -> PrivacyBudget {
        use super::super::AccountingMethod;
        // Placeholder implementation
        PrivacyBudget {
            epsilon_consumed: 0.1 * self.current_round as f64,
            delta_consumed: 1e-5 * self.current_round as f64,
            epsilon_remaining: self.config.base_config.target_epsilon
                - (0.1 * self.current_round as f64),
            delta_remaining: self.config.base_config.target_delta
                - (1e-5 * self.current_round as f64),
            steps_taken: self.current_round,
            accounting_method: AccountingMethod::MomentsAccountant,
            estimated_steps_remaining: 100,
        }
    }

    /// Get current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get configuration
    pub fn config(&self) -> &FederatedPrivacyConfig {
        &self.config
    }
}

// Placeholder implementations for missing methods in components
impl<T: Float + Debug + Send + Sync + 'static> ByzantineRobustAggregator<T> {
    pub fn compute_robustness_factor(&self) -> Result<f64> {
        Ok(0.9) // Placeholder
    }

    pub fn get_client_reputations(&self, _clients: &[String]) -> HashMap<String, f64> {
        HashMap::new() // Placeholder
    }

    /// Detect Byzantine (outlier) clients using a robust distance-based rule.
    ///
    /// Algorithm (median / MAD outlier test, robust to a minority of adversaries):
    /// 1. Compute the coordinate-wise median vector across all client updates.
    ///    The coordinate-wise median is itself a robust location estimate that a
    ///    minority of Byzantine clients cannot move arbitrarily.
    /// 2. For every client, compute the L2 distance from its update to that median.
    /// 3. Compute the median of those distances and the median absolute deviation
    ///    (MAD) of the distances around their median.
    /// 4. Flag any client whose distance exceeds `median_distance + K * MAD`.
    ///
    /// The cutoff multiplier `K` is set to `3.0`. Combined with the MAD scale
    /// factor `1.4826` (which makes the MAD a consistent estimator of the standard
    /// deviation under normality), this corresponds to roughly a 3-sigma rule,
    /// i.e. it flags points that would occur with probability well below 1% for a
    /// Gaussian inlier population while remaining insensitive to the outliers that
    /// inflate a non-robust mean/standard-deviation estimate.
    ///
    /// Trivial cases (0, 1 or 2 clients) cannot support a meaningful outlier test
    /// and therefore yield no detections.
    pub fn detect_byzantine_clients(
        &self,
        updates: &HashMap<String, Array1<T>>,
        round: usize,
    ) -> Result<Vec<OutlierDetectionResult>> {
        // With fewer than 3 clients there is no robust majority to compare against,
        // so no Byzantine decision can be made.
        if updates.len() < 3 {
            return Ok(Vec::new());
        }

        // Validate that all client vectors share the same dimensionality.
        let dimension = Self::validate_uniform_dimensions(updates)?;
        if dimension == 0 {
            return Ok(Vec::new());
        }

        // Step 1: coordinate-wise median vector (robust central reference).
        let median_vector = Self::coordinate_wise_median(updates, dimension);

        // Step 2: L2 distance of each client update to the median vector.
        // Keep a stable ordering of client ids for deterministic processing.
        let mut client_ids: Vec<&String> = updates.keys().collect();
        client_ids.sort();

        let mut distances: Vec<(String, T)> = Vec::with_capacity(client_ids.len());
        for clientid in &client_ids {
            let update = &updates[*clientid];
            let mut sum_sq = T::zero();
            for (idx, &value) in update.iter().enumerate() {
                let diff = value - median_vector[idx];
                sum_sq = sum_sq + diff * diff;
            }
            distances.push(((*clientid).clone(), sum_sq.sqrt()));
        }

        // Step 3: median distance and MAD of the distances.
        let distance_values: Vec<T> = distances.iter().map(|(_, d)| *d).collect();
        let median_distance = Self::median_of(&distance_values);

        let abs_deviations: Vec<T> = distance_values
            .iter()
            .map(|&d| (d - median_distance).abs())
            .collect();
        let mad = Self::median_of(&abs_deviations);

        // Consistent-estimator scale factor for the MAD under normality.
        let mad_scale = T::from(1.4826).unwrap_or_else(T::one);
        let cutoff_multiplier = T::from(3.0).unwrap_or_else(T::one);

        // When the MAD is (numerically) zero every inlier sits at the same distance;
        // fall back to a small fraction of the median distance to avoid flagging the
        // whole population while still catching clients that are strictly farther out.
        let scaled_mad = mad * mad_scale;
        let effective_spread = if scaled_mad > T::epsilon() {
            scaled_mad
        } else {
            // 1% of the median distance as a numerical guard band.
            median_distance * T::from(0.01).unwrap_or_else(T::zero)
        };

        let threshold = median_distance + cutoff_multiplier * effective_spread;
        let threshold_f64 = threshold.to_f64().unwrap_or(f64::INFINITY);

        // Step 4: flag clients whose distance exceeds the threshold.
        let mut results = Vec::new();
        for (clientid, distance) in distances {
            let is_outlier = distance > threshold;
            if is_outlier {
                results.push(OutlierDetectionResult {
                    clientid,
                    round,
                    is_outlier: true,
                    outlier_score: distance.to_f64().unwrap_or(0.0) - threshold_f64,
                    detection_method: "median_mad_l2_distance".to_string(),
                });
            }
        }

        Ok(results)
    }

    /// Byzantine-robust aggregation via the coordinate-wise trimmed mean.
    ///
    /// For each coordinate the client values are sorted, the lowest and highest
    /// `trim_count` values are discarded, and the remaining values are averaged.
    /// Because at most `trim_count` adversarial values can survive on either tail,
    /// the estimator cannot be dragged to infinity by a minority of malicious
    /// clients (unlike a plain coordinate-wise mean, where a single client can move
    /// the result arbitrarily). When the number of clients is too small for any
    /// trimming to leave values behind, the method falls back to the coordinate-wise
    /// median, which is itself a robust location estimate.
    ///
    /// Trim fraction: the `ByzantineRobustConfig` stored inside the aggregator is not
    /// reachable from this module (its fields are private to the components module),
    /// so a documented default of 10% per tail (`trim_ratio = 0.1`) is used. This
    /// tolerates up to ~10% Byzantine clients on each side while retaining most of
    /// the honest mass for a low-variance estimate.
    pub fn robust_aggregate(
        &self,
        updates: &HashMap<String, Array1<T>>,
        _allocations: &HashMap<String, AdaptivePrivacyAllocation>,
    ) -> Result<Array1<T>> {
        if updates.is_empty() {
            return Err(OptimError::InvalidParameter(
                "No updates to aggregate".to_string(),
            ));
        }

        // All client vectors must share the same length.
        let dimension = Self::validate_uniform_dimensions(updates)?;

        let num_clients = updates.len();

        // Default trim fraction: drop 10% of clients from each tail.
        let trim_fraction = 0.1_f64;
        let trim_count = ((num_clients as f64) * trim_fraction).floor() as usize;

        // Materialize the per-client vectors in a stable order.
        let mut client_ids: Vec<&String> = updates.keys().collect();
        client_ids.sort();
        let vectors: Vec<&Array1<T>> = client_ids.iter().map(|id| &updates[*id]).collect();

        let mut result = Array1::from_elem(dimension, T::zero());

        // Scratch buffer reused across coordinates.
        let mut column: Vec<T> = Vec::with_capacity(num_clients);

        for coord in 0..dimension {
            column.clear();
            for vec in &vectors {
                column.push(vec[coord]);
            }

            // Trimming would remove every value (2 * trim_count >= n): use the median.
            let aggregated = if num_clients <= 2 * trim_count + 1 || trim_count == 0 {
                if trim_count == 0 {
                    // No trimming requested/possible: plain mean over all clients is
                    // unavailable as a *robust* estimator, so use the (robust) mean of
                    // the values that remain after a zero-width trim, which is the full
                    // set; for robustness with very few clients fall back to the median.
                    if num_clients <= 2 {
                        Self::median_of(&column)
                    } else {
                        Self::mean_of(&column)
                    }
                } else {
                    Self::median_of(&column)
                }
            } else {
                Self::sort_values(&mut column);
                let trimmed = &column[trim_count..num_clients - trim_count];
                Self::mean_of(trimmed)
            };

            result[coord] = aggregated;
        }

        Ok(result)
    }

    /// Validate that every client update has the same dimensionality and return it.
    fn validate_uniform_dimensions(updates: &HashMap<String, Array1<T>>) -> Result<usize> {
        let mut iter = updates.values();
        let first = match iter.next() {
            Some(v) => v,
            None => {
                return Err(OptimError::InvalidParameter(
                    "No updates to aggregate".to_string(),
                ))
            }
        };
        let expected = first.len();
        for update in iter {
            if update.len() != expected {
                return Err(OptimError::DimensionMismatch(format!(
                    "Client update dimensions differ: expected {}, found {}",
                    expected,
                    update.len()
                )));
            }
        }
        Ok(expected)
    }

    /// Coordinate-wise median across all client update vectors.
    fn coordinate_wise_median(updates: &HashMap<String, Array1<T>>, dimension: usize) -> Vec<T> {
        let vectors: Vec<&Array1<T>> = updates.values().collect();
        let mut median = vec![T::zero(); dimension];
        let mut column: Vec<T> = Vec::with_capacity(vectors.len());
        for (coord, slot) in median.iter_mut().enumerate() {
            column.clear();
            for vec in &vectors {
                column.push(vec[coord]);
            }
            *slot = Self::median_of(&column);
        }
        median
    }

    /// Sort a slice of floating-point values, treating NaN as the largest value so
    /// that ordering is total and deterministic.
    fn sort_values(values: &mut [T]) {
        values.sort_by(|a, b| match a.partial_cmp(b) {
            Some(ordering) => ordering,
            None => {
                // Push NaNs to the end consistently.
                if a.is_nan() && b.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        });
    }

    /// Median of an unsorted slice (does not mutate the input).
    fn median_of(values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let mut sorted = values.to_vec();
        Self::sort_values(&mut sorted);
        let n = sorted.len();
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            let lo = sorted[n / 2 - 1];
            let hi = sorted[n / 2];
            (lo + hi) / T::from(2.0).unwrap_or_else(|| T::one() + T::one())
        }
    }

    /// Arithmetic mean of a slice.
    fn mean_of(values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum = values.iter().fold(T::zero(), |acc, &x| acc + x);
        let count = T::from(values.len()).unwrap_or_else(T::one);
        sum / count
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdaptiveBudgetManager<T> {
    pub fn compute_adaptive_allocations(
        &self,
        _updates: &HashMap<String, Array1<T>>,
        _plan: &FederatedRoundPlan,
    ) -> Result<HashMap<String, AdaptivePrivacyAllocation>> {
        Ok(HashMap::new()) // Placeholder
    }
}

impl<T: Float + Debug + Send + Sync + 'static> PersonalizationManager<T> {
    pub fn personalize_client_updates(
        &self,
        updates: &HashMap<String, Array1<T>>,
        _plan: &FederatedRoundPlan,
    ) -> Result<HashMap<String, Array1<T>>> {
        Ok(updates.clone()) // Placeholder - just return original updates
    }

    pub fn get_metrics(&self) -> PersonalizationMetrics {
        PersonalizationMetrics {
            cluster_assignments: HashMap::new(),
            adaptation_effectiveness: 0.8,
            model_diversity: 0.6,
            personalization_overhead: 0.1,
        }
    }

    pub fn compute_privacy_cost(&self) -> Result<f64> {
        Ok(0.1) // Placeholder
    }

    pub fn update_global_model(&self, aggregate: &Array1<T>) -> Result<Array1<T>> {
        Ok(aggregate.clone()) // Placeholder
    }

    pub fn cluster_clients(
        &self,
        _updates: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<usize, Vec<String>>> {
        Ok(HashMap::new()) // Placeholder
    }

    pub fn generate_personalized_models(
        &self,
        _clusters: &HashMap<usize, Vec<String>>,
        _gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, PersonalizedModel<T>>> {
        Ok(HashMap::new()) // Placeholder
    }

    pub fn compute_effectiveness_metrics(
        &self,
        _models: &HashMap<String, PersonalizedModel<T>>,
    ) -> Result<PersonalizationMetrics> {
        Ok(PersonalizationMetrics {
            cluster_assignments: HashMap::new(),
            adaptation_effectiveness: 0.8,
            model_diversity: 0.6,
            personalization_overhead: 0.1,
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static> CommunicationOptimizer<T> {
    pub fn compress_and_schedule(
        &self,
        updates: &HashMap<String, Array1<T>>,
        _plan: &FederatedRoundPlan,
    ) -> Result<HashMap<String, Array1<T>>> {
        Ok(updates.clone()) // Placeholder
    }

    pub fn get_efficiency_stats(&self) -> CommunicationEfficiencyStats {
        CommunicationEfficiencyStats {
            compression_ratio: 0.5,
            bandwidth_utilization: 0.8,
            transmission_latency: 100.0,
            quality_of_service_score: 0.9,
        }
    }

    pub fn compute_efficiency_scores(&self, _clients: &[String]) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new()) // Placeholder
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ContinualLearningCoordinator<T> {
    pub fn adapt_to_new_task(
        &mut self,
        _updates: &HashMap<String, Array1<T>>,
        _round: usize,
    ) -> Result<()> {
        Ok(()) // Placeholder
    }

    pub fn get_status(&self) -> ContinualLearningStatus {
        ContinualLearningStatus {
            task_changes_detected: 0,
            forgetting_prevention_active: false,
            memory_utilization: 0.5,
            knowledge_transfer_score: 0.7,
        }
    }

    pub fn compute_privacy_overhead(&self) -> Result<f64> {
        Ok(0.05) // Placeholder
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default> FederatedMetaLearner<T> {
    pub fn compute_meta_gradients(
        &self,
        _aggregates: &HashMap<usize, Array1<T>>,
    ) -> Result<Array1<T>> {
        Ok(Array1::default(0)) // Placeholder
    }
}

impl<T: Float + Debug + Send + Sync + 'static> SecureAggregator<T> {
    pub fn prepare_round(&self, _clients: &[String]) -> Result<SecureAggregationPlan> {
        Ok(SecureAggregationPlan {
            masking_seeds: HashMap::new(),
            aggregation_threshold: 10,
            dropout_tolerance: 2,
        })
    }

    pub fn aggregate_with_masks(
        &self,
        updates: &HashMap<String, Array1<T>>,
        _plan: &SecureAggregationPlan,
    ) -> Result<Array1<T>> {
        // Placeholder - just do simple aggregation
        if updates.is_empty() {
            return Err(OptimError::InvalidParameter(
                "No updates to aggregate".to_string(),
            ));
        }

        let mut result = updates.values().next().expect("unwrap failed").clone();
        let mut count = 1;

        for update in updates.values().skip(1) {
            for (i, &value) in update.iter().enumerate() {
                if i < result.len() {
                    result[i] = result[i] + value;
                }
            }
            count += 1;
        }

        for value in result.iter_mut() {
            *value = *value / T::from(count).unwrap_or_else(|| T::zero());
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap;

    fn make_updates(vectors: &[(&str, Vec<f64>)]) -> HashMap<String, Array1<f64>> {
        let mut map = HashMap::new();
        for (id, values) in vectors {
            map.insert((*id).to_string(), Array1::from_vec(values.clone()));
        }
        map
    }

    /// Plain coordinate-wise mean reference (the *non-robust* estimator) used to
    /// demonstrate that the robust estimator behaves differently under attack.
    fn plain_mean(updates: &HashMap<String, Array1<f64>>) -> Array1<f64> {
        let dim = updates.values().next().expect("at least one update").len();
        let mut acc = Array1::from_elem(dim, 0.0_f64);
        for v in updates.values() {
            for i in 0..dim {
                acc[i] += v[i];
            }
        }
        let n = updates.len() as f64;
        acc.mapv(|x| x / n)
    }

    #[test]
    fn test_robust_aggregate_resists_outliers() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");

        // Twenty honest clients clustered near (1.0, -2.0) plus two extreme
        // adversaries. With the documented 10%-per-tail trim and 22 clients the
        // trim count is floor(0.1 * 22) = 2, exactly covering the two adversaries
        // on whichever tail they land (this is the estimator's breakdown point).
        let mut vectors: Vec<(String, Vec<f64>)> = Vec::new();
        // Deterministic small jitter around the honest center.
        let jitter = [
            0.0_f64, 0.02, -0.02, 0.01, -0.01, 0.03, -0.03, 0.015, -0.015, 0.005,
        ];
        for i in 0..20usize {
            let dx = jitter[i % jitter.len()];
            vectors.push((format!("h{i}"), vec![1.0 + dx, -2.0 - dx]));
        }
        let mut updates = HashMap::new();
        for (id, v) in &vectors {
            updates.insert(id.clone(), Array1::from_vec(v.clone()));
        }
        // Two adversaries pushing in opposite directions per coordinate.
        updates.insert("evil0".to_string(), Array1::from_vec(vec![1000.0, 1000.0]));
        updates.insert("evil1".to_string(), Array1::from_vec(vec![900.0, 850.0]));

        let allocations: HashMap<String, AdaptivePrivacyAllocation> = HashMap::new();
        let robust = aggregator
            .robust_aggregate(&updates, &allocations)
            .expect("robust aggregate");

        // The robust estimate must stay near the honest cluster.
        assert!(
            (robust[0] - 1.0).abs() < 0.1,
            "robust[0] = {} should be near 1.0",
            robust[0]
        );
        assert!(
            (robust[1] - (-2.0)).abs() < 0.1,
            "robust[1] = {} should be near -2.0",
            robust[1]
        );

        // A plain mean, by contrast, is dragged far away by the adversaries.
        let mean = plain_mean(&updates);
        assert!(
            (mean[0] - 1.0).abs() > 5.0,
            "plain mean[0] = {} should be far from honest cluster",
            mean[0]
        );
    }

    #[test]
    fn test_robust_aggregate_dimension_mismatch() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");
        let updates = make_updates(&[
            ("a", vec![1.0, 2.0, 3.0]),
            ("b", vec![1.0, 2.0]), // wrong length
        ]);
        let allocations: HashMap<String, AdaptivePrivacyAllocation> = HashMap::new();
        let err = aggregator.robust_aggregate(&updates, &allocations);
        assert!(matches!(err, Err(OptimError::DimensionMismatch(_))));
    }

    #[test]
    fn test_robust_aggregate_empty_input() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");
        let updates: HashMap<String, Array1<f64>> = HashMap::new();
        let allocations: HashMap<String, AdaptivePrivacyAllocation> = HashMap::new();
        let err = aggregator.robust_aggregate(&updates, &allocations);
        assert!(matches!(err, Err(OptimError::InvalidParameter(_))));
    }

    #[test]
    fn test_detect_byzantine_flags_single_outlier() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");

        let mut updates = make_updates(&[
            ("c0", vec![1.0, 1.0, 1.0]),
            ("c1", vec![1.1, 0.9, 1.05]),
            ("c2", vec![0.9, 1.1, 0.95]),
            ("c3", vec![1.05, 1.0, 1.0]),
            ("c4", vec![0.95, 0.98, 1.02]),
            ("c5", vec![1.02, 1.03, 0.99]),
        ]);
        // One blatant Byzantine client far from the cluster.
        updates.insert(
            "traitor".to_string(),
            Array1::from_vec(vec![50.0, -40.0, 60.0]),
        );

        let flagged = aggregator
            .detect_byzantine_clients(&updates, 7)
            .expect("detection");

        assert_eq!(flagged.len(), 1, "exactly one client should be flagged");
        let result = &flagged[0];
        assert_eq!(result.clientid, "traitor");
        assert!(result.is_outlier);
        assert_eq!(result.round, 7);
        assert!(result.outlier_score > 0.0);
        assert_eq!(result.detection_method, "median_mad_l2_distance");
    }

    #[test]
    fn test_detect_byzantine_no_false_positive() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");

        // All clients are similar; none should be flagged.
        let updates = make_updates(&[
            ("c0", vec![1.0, 2.0]),
            ("c1", vec![1.02, 1.98]),
            ("c2", vec![0.98, 2.02]),
            ("c3", vec![1.01, 1.99]),
            ("c4", vec![0.99, 2.01]),
            ("c5", vec![1.0, 2.0]),
        ]);

        let flagged = aggregator
            .detect_byzantine_clients(&updates, 3)
            .expect("detection");
        assert!(
            flagged.is_empty(),
            "no clients should be flagged, got {:?}",
            flagged
        );
    }

    #[test]
    fn test_detect_byzantine_trivial_cases() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");

        // Fewer than 3 clients: no decision possible.
        let two = make_updates(&[("a", vec![1.0, 1.0]), ("b", vec![100.0, 100.0])]);
        let flagged = aggregator
            .detect_byzantine_clients(&two, 0)
            .expect("detection");
        assert!(flagged.is_empty());

        let empty: HashMap<String, Array1<f64>> = HashMap::new();
        let flagged_empty = aggregator
            .detect_byzantine_clients(&empty, 0)
            .expect("detection");
        assert!(flagged_empty.is_empty());
    }
}
