// Common search strategy functionality and base traits
//
// This module defines common traits, utilities, and base functionality
// shared across different search strategies.

use std::collections::HashMap;
use num_traits::Float;
use std::fmt::Debug;

use super::super::architecture::{ArchitectureSpec, ArchitectureCandidate};

/// Base trait for search strategies
pub trait SearchStrategyTrait<T: Float + Debug + Send + Sync + 'static> {
    /// Generate a new architecture candidate
    fn generate_candidate(&mut self) -> Result<ArchitectureCandidate, super::SearchError>;

    /// Update the strategy with evaluation results
    fn update(&mut self, candidate: &ArchitectureCandidate, performance: T);

    /// Check if the strategy should terminate
    fn should_terminate(&self) -> bool;

    /// Get the current best candidates
    fn get_best_candidates(&self) -> Vec<&ArchitectureCandidate>;

    /// Reset the strategy state
    fn reset(&mut self);

    /// Get strategy-specific statistics
    fn get_statistics(&self) -> HashMap<String, f64>;
}

/// Mutation operations for architecture modification
pub trait MutationOperator {
    /// Apply mutation to an architecture
    fn mutate(&self, architecture: &mut ArchitectureSpec, mutation_rate: f64) -> bool;

    /// Get mutation type identifier
    fn mutation_type(&self) -> String;

    /// Get estimated impact of mutation
    fn estimated_impact(&self) -> f64;
}

/// Crossover operations for architecture combination
pub trait CrossoverOperator {
    /// Perform crossover between two architectures
    fn crossover(
        &self,
        parent1: &ArchitectureSpec,
        parent2: &ArchitectureSpec,
    ) -> Result<(ArchitectureSpec, ArchitectureSpec), super::SearchError>;

    /// Get crossover type identifier
    fn crossover_type(&self) -> String;
}

/// Selection methods for choosing parents
pub trait SelectionMethod<T: Float + Debug + Send + Sync + 'static> {
    /// Select candidates for the next generation
    fn select(
        &self,
        population: &[ArchitectureCandidate],
        fitnesses: &[T],
        num_selected: usize,
    ) -> Vec<usize>;

    /// Get selection method identifier
    fn selection_type(&self) -> String;
}

/// Types of mutations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MutationType {
    LayerAddition,
    LayerRemoval,
    LayerModification,
    ConnectionAddition,
    ConnectionRemoval,
    ParameterMutation,
    StructuralChange,
    ActivationChange,
    NormalizationChange,
    SkipConnectionChange,
}

impl MutationType {
    /// Get all mutation types
    pub fn all_types() -> Vec<MutationType> {
        vec![
            MutationType::LayerAddition,
            MutationType::LayerRemoval,
            MutationType::LayerModification,
            MutationType::ConnectionAddition,
            MutationType::ConnectionRemoval,
            MutationType::ParameterMutation,
            MutationType::StructuralChange,
            MutationType::ActivationChange,
            MutationType::NormalizationChange,
            MutationType::SkipConnectionChange,
        ]
    }

    /// Get mutation severity (0.0 = minor, 1.0 = major)
    pub fn severity(&self) -> f64 {
        match self {
            MutationType::LayerAddition => 0.8,
            MutationType::LayerRemoval => 0.9,
            MutationType::LayerModification => 0.6,
            MutationType::ConnectionAddition => 0.5,
            MutationType::ConnectionRemoval => 0.6,
            MutationType::ParameterMutation => 0.3,
            MutationType::StructuralChange => 1.0,
            MutationType::ActivationChange => 0.4,
            MutationType::NormalizationChange => 0.3,
            MutationType::SkipConnectionChange => 0.7,
        }
    }
}

/// Basic layer addition mutation
pub struct LayerAdditionMutation {
    pub layer_types: Vec<super::super::architecture::LayerType>,
    pub activation_types: Vec<super::super::architecture::ActivationType>,
}

impl MutationOperator for LayerAdditionMutation {
    fn mutate(&self, architecture: &mut ArchitectureSpec, _mutation_rate: f64) -> bool {
        // Simplified implementation - add a random layer
        use super::super::architecture::{LayerSpec, LayerDimensions};
        
        if architecture.layers.is_empty() {
            return false;
        }

        let layer_type = self.layer_types.get(0).copied()
            .unwrap_or(super::super::architecture::LayerType::Linear);
        let activation = self.activation_types.get(0).copied()
            .unwrap_or(super::super::architecture::ActivationType::ReLU);

        let dimensions = LayerDimensions {
            input_dim: 128,
            output_dim: 128,
            hidden_dims: vec![],
        };

        let new_layer = LayerSpec::new(layer_type, dimensions, activation);
        architecture.layers.push(new_layer);

        true
    }

    fn mutation_type(&self) -> String {
        "layer_addition".to_string()
    }

    fn estimated_impact(&self) -> f64 {
        0.8
    }
}

impl Default for LayerAdditionMutation {
    fn default() -> Self {
        Self {
            layer_types: vec![
                super::super::architecture::LayerType::Linear,
                super::super::architecture::LayerType::LSTM,
                super::super::architecture::LayerType::Attention,
            ],
            activation_types: vec![
                super::super::architecture::ActivationType::ReLU,
                super::super::architecture::ActivationType::GELU,
                super::super::architecture::ActivationType::Tanh,
            ],
        }
    }
}

/// Basic layer removal mutation
pub struct LayerRemovalMutation {
    pub min_layers: usize,
}

impl MutationOperator for LayerRemovalMutation {
    fn mutate(&self, architecture: &mut ArchitectureSpec, _mutation_rate: f64) -> bool {
        if architecture.layers.len() <= self.min_layers {
            return false;
        }

        // Remove the last layer (simplified)
        architecture.layers.pop();
        true
    }

    fn mutation_type(&self) -> String {
        "layer_removal".to_string()
    }

    fn estimated_impact(&self) -> f64 {
        0.9
    }
}

impl Default for LayerRemovalMutation {
    fn default() -> Self {
        Self { min_layers: 1 }
    }
}

/// Parameter mutation operator
pub struct ParameterMutation {
    pub mutation_strength: f64,
    pub parameter_names: Vec<String>,
}

impl MutationOperator for ParameterMutation {
    fn mutate(&self, architecture: &mut ArchitectureSpec, mutation_rate: f64) -> bool {
        let mut mutated = false;

        for layer in &mut architecture.layers {
            for param_name in &self.parameter_names {
                if let Some(value) = layer.parameters.get_mut(param_name) {
                    if rand::random::<f64>() < mutation_rate {
                        *value += (rand::random::<f64>() - 0.5) * 2.0 * self.mutation_strength;
                        mutated = true;
                    }
                }
            }
        }

        mutated
    }

    fn mutation_type(&self) -> String {
        "parameter_mutation".to_string()
    }

    fn estimated_impact(&self) -> f64 {
        0.3
    }
}

impl Default for ParameterMutation {
    fn default() -> Self {
        Self {
            mutation_strength: 0.1,
            parameter_names: vec![
                "learning_rate".to_string(),
                "dropout_rate".to_string(),
                "weight_decay".to_string(),
            ],
        }
    }
}

/// Uniform crossover operator
pub struct UniformCrossover {
    pub crossover_probability: f64,
}

impl CrossoverOperator for UniformCrossover {
    fn crossover(
        &self,
        parent1: &ArchitectureSpec,
        parent2: &ArchitectureSpec,
    ) -> Result<(ArchitectureSpec, ArchitectureSpec), super::SearchError> {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        let min_layers = parent1.layers.len().min(parent2.layers.len());

        for i in 0..min_layers {
            if rand::random::<f64>() < self.crossover_probability {
                if i < child1.layers.len() && i < child2.layers.len() {
                    std::mem::swap(&mut child1.layers[i], &mut child2.layers[i]);
                }
            }
        }

        Ok((child1, child2))
    }

    fn crossover_type(&self) -> String {
        "uniform_crossover".to_string()
    }
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self {
            crossover_probability: 0.5,
        }
    }
}

/// Tournament selection method
pub struct TournamentSelection {
    pub tournament_size: usize,
    pub pressure: f64,
}

impl<T: Float + Debug + Send + Sync + 'static> SelectionMethod<T> for TournamentSelection {
    fn select(
        &self,
        population: &[ArchitectureCandidate],
        fitnesses: &[T],
        num_selected: usize,
    ) -> Vec<usize> {
        let mut selected = Vec::with_capacity(num_selected);

        for _ in 0..num_selected {
            let mut tournament = Vec::new();
            
            // Select tournament candidates
            for _ in 0..self.tournament_size {
                let idx = rand::random::<usize>() % population.len();
                tournament.push(idx);
            }

            // Find best in tournament
            let best_idx = tournament.iter()
                .max_by(|&&a, &&b| {
                    fitnesses[a].partial_cmp(&fitnesses[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(0);

            selected.push(best_idx);
        }

        selected
    }

    fn selection_type(&self) -> String {
        "tournament_selection".to_string()
    }
}

impl Default for TournamentSelection {
    fn default() -> Self {
        Self {
            tournament_size: 3,
            pressure: 1.0,
        }
    }
}

/// Roulette wheel selection method
pub struct RouletteWheelSelection {
    pub scaling_factor: f64,
}

impl<T: Float + Debug + Send + Sync + 'static> SelectionMethod<T> for RouletteWheelSelection {
    fn select(
        &self,
        _population: &[ArchitectureCandidate],
        fitnesses: &[T],
        num_selected: usize,
    ) -> Vec<usize> {
        let mut selected = Vec::with_capacity(num_selected);

        // Calculate fitness sum
        let fitness_sum = fitnesses.iter().fold(T::zero(), |acc, &x| acc + x);

        if fitness_sum == T::zero() {
            // If all fitnesses are zero, select randomly
            for _ in 0..num_selected {
                selected.push(rand::random::<usize>() % fitnesses.len());
            }
            return selected;
        }

        for _ in 0..num_selected {
            let mut roulette_value = T::from(rand::random::<f64>()).unwrap() * fitness_sum;
            let mut cumulative_fitness = T::zero();

            for (idx, &fitness) in fitnesses.iter().enumerate() {
                cumulative_fitness = cumulative_fitness + fitness;
                if roulette_value <= cumulative_fitness {
                    selected.push(idx);
                    break;
                }
            }
        }

        selected
    }

    fn selection_type(&self) -> String {
        "roulette_wheel_selection".to_string()
    }
}

impl Default for RouletteWheelSelection {
    fn default() -> Self {
        Self { scaling_factor: 1.0 }
    }
}

/// Rank selection method
pub struct RankSelection {
    pub selective_pressure: f64,
}

impl<T: Float + Debug + Send + Sync + 'static> SelectionMethod<T> for RankSelection {
    fn select(
        &self,
        population: &[ArchitectureCandidate],
        fitnesses: &[T],
        num_selected: usize,
    ) -> Vec<usize> {
        let mut indexed_fitnesses: Vec<(usize, T)> = fitnesses.iter()
            .enumerate()
            .map(|(i, &f)| (i, f))
            .collect();

        // Sort by fitness
        indexed_fitnesses.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected = Vec::with_capacity(num_selected);
        let pop_size = population.len();

        for _ in 0..num_selected {
            // Select based on rank
            let rank = (rand::random::<f64>() * pop_size as f64) as usize;
            let idx = indexed_fitnesses[rank.min(pop_size - 1)].0;
            selected.push(idx);
        }

        selected
    }

    fn selection_type(&self) -> String {
        "rank_selection".to_string()
    }
}

impl Default for RankSelection {
    fn default() -> Self {
        Self { selective_pressure: 2.0 }
    }
}

/// Strategy evaluation metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    pub convergence_rate: f64,
    pub diversity_maintained: f64,
    pub exploration_efficiency: f64,
    pub exploitation_efficiency: f64,
    pub computational_efficiency: f64,
}

impl Default for StrategyMetrics {
    fn default() -> Self {
        Self {
            convergence_rate: 0.0,
            diversity_maintained: 0.0,
            exploration_efficiency: 0.0,
            exploitation_efficiency: 0.0,
            computational_efficiency: 0.0,
        }
    }
}

/// Utility functions for strategies
pub mod utils {
    use super::*;

    /// Calculate population diversity
    pub fn calculate_diversity(population: &[ArchitectureCandidate]) -> f64 {
        if population.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut comparisons = 0;

        for i in 0..population.len() {
            for j in (i + 1)..population.len() {
                let distance = architecture_distance(&population[i].architecture, &population[j].architecture);
                total_distance += distance;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_distance / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate distance between two architectures
    pub fn architecture_distance(arch1: &ArchitectureSpec, arch2: &ArchitectureSpec) -> f64 {
        // Simplified distance metric based on layer count difference
        let layer_diff = (arch1.layers.len() as i32 - arch2.layers.len() as i32).abs() as f64;
        let param_diff = (arch1.parameter_count() as i64 - arch2.parameter_count() as i64).abs() as f64;
        
        layer_diff + param_diff / 1000.0 // Normalize parameter difference
    }

    /// Calculate convergence rate
    pub fn calculate_convergence_rate<T: Float + Debug + Send + Sync + 'static>(fitness_history: &[Vec<T>]) -> f64 {
        if fitness_history.len() < 2 {
            return 0.0;
        }

        let mut improvements = 0;
        let mut total_changes = 0;

        for i in 1..fitness_history.len() {
            if let (Some(prev_best), Some(curr_best)) = (
                fitness_history[i - 1].first(),
                fitness_history[i].first(),
            ) {
                if *curr_best > *prev_best {
                    improvements += 1;
                }
                total_changes += 1;
            }
        }

        if total_changes > 0 {
            improvements as f64 / total_changes as f64
        } else {
            0.0
        }
    }

    /// Validate architecture candidate
    pub fn validate_candidate(candidate: &ArchitectureCandidate) -> Result<(), super::SearchError> {
        // Basic validation
        if candidate.architecture.layers.is_empty() {
            return Err(super::SearchError::InvalidConfiguration(
                "Architecture must have at least one layer".to_string()
            ));
        }

        // Validate architecture specification
        candidate.architecture.validate().map_err(|e| {
            super::SearchError::InvalidConfiguration(format!("Architecture validation failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::architecture::*;

    #[test]
    fn test_layer_addition_mutation() {
        let mut arch = ArchitectureSpec::new(
            vec![LayerSpec::new(
                LayerType::Linear,
                LayerDimensions { input_dim: 64, output_dim: 32, hidden_dims: vec![] },
                ActivationType::ReLU,
            )],
            GlobalArchitectureConfig::default(),
        );

        let mutation = LayerAdditionMutation::default();
        let result = mutation.mutate(&mut arch, 1.0);

        assert!(result);
        assert_eq!(arch.layers.len(), 2);
    }

    #[test]
    fn test_layer_removal_mutation() {
        let mut arch = ArchitectureSpec::new(
            vec![
                LayerSpec::new(
                    LayerType::Linear,
                    LayerDimensions { input_dim: 64, output_dim: 32, hidden_dims: vec![] },
                    ActivationType::ReLU,
                ),
                LayerSpec::new(
                    LayerType::Linear,
                    LayerDimensions { input_dim: 32, output_dim: 16, hidden_dims: vec![] },
                    ActivationType::ReLU,
                ),
            ],
            GlobalArchitectureConfig::default(),
        );

        let mutation = LayerRemovalMutation::default();
        let result = mutation.mutate(&mut arch, 1.0);

        assert!(result);
        assert_eq!(arch.layers.len(), 1);
    }

    #[test]
    fn test_uniform_crossover() {
        let arch1 = ArchitectureSpec::new(
            vec![LayerSpec::new(
                LayerType::Linear,
                LayerDimensions { input_dim: 64, output_dim: 32, hidden_dims: vec![] },
                ActivationType::ReLU,
            )],
            GlobalArchitectureConfig::default(),
        );

        let arch2 = ArchitectureSpec::new(
            vec![LayerSpec::new(
                LayerType::LSTM,
                LayerDimensions { input_dim: 64, output_dim: 32, hidden_dims: vec![] },
                ActivationType::Tanh,
            )],
            GlobalArchitectureConfig::default(),
        );

        let crossover = UniformCrossover::default();
        let result = crossover.crossover(&arch1, &arch2);

        assert!(result.is_ok());
        let (child1, child2) = result.unwrap();
        assert_eq!(child1.layers.len(), 1);
        assert_eq!(child2.layers.len(), 1);
    }

    #[test]
    fn test_tournament_selection() {
        let population = vec![
            ArchitectureCandidate::new("arch1".to_string(), ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default())),
            ArchitectureCandidate::new("arch2".to_string(), ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default())),
            ArchitectureCandidate::new("arch3".to_string(), ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default())),
        ];
        let fitnesses = vec![0.5, 0.8, 0.3];

        let selection = TournamentSelection::default();
        let selected = selection.select(&population, &fitnesses, 2);

        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|&i| i < population.len()));
    }

    #[test]
    fn test_diversity_calculation() {
        let population = vec![
            ArchitectureCandidate::new("arch1".to_string(), ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default())),
            ArchitectureCandidate::new("arch2".to_string(), ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default())),
        ];

        let diversity = utils::calculate_diversity(&population);
        assert!(diversity >= 0.0);
    }
}