use std::fmt::Debug;
// Neural Architecture Search Engine Module
//
// This module provides a comprehensive Neural Architecture Search (NAS) implementation
// for automatically discovering optimal optimization algorithms. The module is organized
// into focused sub-modules for maintainability and modularity.
//
// ## Module Organization
//
// - **config**: Configuration types, enums, and parameter definitions
// - **results**: Search results, evaluation results, and statistics tracking
// - **resources**: Resource monitoring, constraint enforcement, and optimization
// - **engine**: Main NAS engine implementation and coordination logic
//
// ## Usage
//
// ```rust
// use optirs_core::neural_architecture_search::nas_engine::{
//     NeuralArchitectureSearch, NASConfig, SearchStrategyType,
//     SearchSpaceConfig, EvaluationConfig, MultiObjectiveConfig
// };
// use num_traits::Float;
//
// // Configure the NAS engine
// let config = NASConfig::<f64> {
//     search_strategy: SearchStrategyType::Evolutionary,
//     search_space: SearchSpaceConfig::default(),
//     evaluation_config: EvaluationConfig::default(),
//     multi_objective_config: MultiObjectiveConfig::default(),
//     search_budget: 1000,
//     // ... other configuration
//     ..Default::default()
// };
//
// // Create and run NAS engine
// let mut nas_engine = NeuralArchitectureSearch::new(config)?;
// let results = nas_engine.run_search()?;
//
// // Access best found architectures
// for architecture in &results.best_architectures {
//     println!("Found architecture: {:?}", architecture.architecture_id);
// }
// ```
//
// ## Key Features
//
// ### Search Strategies
// - Random search baseline
// - Evolutionary algorithms (genetic algorithms, differential evolution)
// - Bayesian optimization with Gaussian processes
// - Reinforcement learning-based search
// - Differentiable architecture search (DARTS)
// - Progressive search with increasing complexity
// - Hybrid strategies combining multiple approaches
//
// ### Multi-Objective Optimization
// - NSGA-II: Non-dominated Sorting Genetic Algorithm II
// - NSGA-III: Non-dominated Sorting Genetic Algorithm III
// - MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition
// - PAES: Pareto Archived Evolution Strategy
// - SPEA2: Strength Pareto Evolutionary Algorithm 2
//
// ### Resource Management
// - Memory usage monitoring and constraints
// - CPU and GPU time tracking
// - Energy consumption estimation
// - Financial cost tracking
// - Network bandwidth monitoring
// - Automatic resource constraint enforcement
//
// ### Performance Features
// - Performance prediction for faster evaluation
// - Progressive search for complex search spaces
// - Transfer learning between search sessions
// - Parallel evaluation of candidate architectures
// - Caching of evaluation results
// - Early stopping based on convergence criteria

use crate::EvaluationMetric;
use num_traits::Float;

pub mod config;
pub mod engine;
pub mod resources;
pub mod results;

// Re-export core types for convenience
pub use config::*;
pub use engine::*;
pub use resources::*;
pub use results::*;

// Additional convenience re-exports for commonly used combinations
pub use engine::{
    ArchitectureController, DiversityMetrics, MultiObjectiveOptimizer, NeuralArchitectureSearch,
    PerformanceEvaluator, PerformancePredictor, ProgressiveNAS, SearchStrategy,
};

// Re-export ParetoFront from multi_objective module
pub use crate::multi_objective::ParetoFront;

pub use config::{
    EarlyStoppingConfig, EvaluationConfig, MultiObjectiveConfig, NASConfig, ObjectiveConfig,
    SearchSpaceConfig, SearchStrategyType,
};

pub use results::{
    ArchitectureEncoding, ConvergenceData, EvaluationResults, OptimizerArchitecture,
    OptimizerComponent, SearchResult, SearchResults, SearchStatistics,
};

pub use config::{HardwareResources, ResourceConstraints};
pub use resources::{ResourceMonitor, ResourceOptimizer, ResourceSnapshot};
pub use results::ResourceUsage;

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Float;

    #[test]
    fn test_nas_config_creation() {
        let config = NASConfig::<f64>::default();
        assert_eq!(config.search_budget, 100);
        assert_eq!(config.population_size, 20);
        assert!(config.early_stopping.enabled);
    }

    #[test]
    fn test_resource_constraints_validation() {
        let constraints = ResourceConstraints::<f64> {
            max_memory_gb: 16.0,
            max_computation_hours: 24.0,
            max_energy_kwh: 100.0,
            max_cost_usd: 1000.0,
            hardware_resources: HardwareResources::default(),
            enable_monitoring: true,
            violation_handling: ResourceViolationHandling::Penalty,
            time_constraints: TimeConstraints::default(),
            energy_constraints: Some(100.0),
            cost_constraints: Some(1000.0),
        };

        let monitor = ResourceMonitor::new(constraints.clone());
        assert!(constraints.max_memory_gb > 0.0);
    }

    #[test]
    fn test_search_strategy_types() {
        let strategies = vec![
            SearchStrategyType::Random,
            SearchStrategyType::Evolutionary,
            SearchStrategyType::BayesianOptimization,
            SearchStrategyType::ReinforcementLearning,
            SearchStrategyType::Differentiable,
            SearchStrategyType::Progressive,
            SearchStrategyType::MultiObjectiveEvolutionary,
            SearchStrategyType::NeuralPredictorBased,
        ];

        assert_eq!(strategies.len(), 8);
    }

    #[test]
    fn test_multi_objective_algorithms() {
        let algorithms = vec![
            MultiObjectiveAlgorithm::NSGA2,
            MultiObjectiveAlgorithm::NSGA3,
            MultiObjectiveAlgorithm::MOEAD,
            MultiObjectiveAlgorithm::PAES,
            MultiObjectiveAlgorithm::SPEA2,
        ];

        assert_eq!(algorithms.len(), 5);
    }

    #[test]
    fn test_evaluation_metrics() {
        let metrics = vec![
            EvaluationMetric::FinalPerformance,
            EvaluationMetric::ConvergenceSpeed,
            EvaluationMetric::Stability,
            EvaluationMetric::Robustness,
            EvaluationMetric::Efficiency,
            EvaluationMetric::Generalization,
            EvaluationMetric::MemoryUsage,
            EvaluationMetric::ComputationTime,
        ];

        assert_eq!(metrics.len(), 8);
    }

    #[test]
    fn test_component_types() {
        let components = vec![
            ComponentType::SGD,
            ComponentType::Adam,
            ComponentType::AdamW,
            ComponentType::RMSprop,
            ComponentType::AdaGrad,
            ComponentType::AdaDelta,
            ComponentType::LBFGS,
            ComponentType::Momentum,
            ComponentType::Nesterov,
            ComponentType::Custom("custom".to_string()),
        ];

        assert_eq!(components.len(), 10);
    }

    #[test]
    fn test_resource_usage_calculation() {
        let usage = ResourceUsage::<f64> {
            memory_gb: 8.0,
            cpu_time_seconds: 3600.0,
            gpu_time_seconds: 1800.0,
            energy_kwh: 5.4,
            cost_usd: 0.648,
            network_gb: 0.1,
            network_io_gb: 0.1,
            disk_io_gb: 0.5,
            peak_memory_gb: 10.0,
            efficiency_score: 0.75,
        };

        assert!(usage.memory_gb > 0.0);
        assert!(usage.cpu_time_seconds > 0.0);
        assert!(usage.energy_kwh > 0.0);
    }

    #[test]
    fn test_architecture_encoding() {
        let encoding = ArchitectureEncoding {
            encoding_type: EncodingType::Binary,
            binary_encoding: Some(vec![1, 2, 3, 4]),
            string_encoding: None,
            graph_encoding: None,
            hash: 12345,
            metadata: EncodingMetadata::default(),
        };

        assert_eq!(encoding.binary_encoding.as_ref().unwrap().len(), 4);
        assert_eq!(encoding.hash, 12345);
    }

    #[test]
    fn test_search_statistics_initialization() {
        let stats = SearchStatistics::<f64>::default();
        assert_eq!(stats.total_evaluations, 0);
        assert_eq!(stats.current_generation, 0);
        assert!(stats.population_diversity >= 0.0);
    }

    #[test]
    fn test_pareto_front_creation() {
        use crate::multi_objective::{CoverageMetrics, FrontMetrics, ObjectiveBounds};

        let pareto_front = ParetoFront::<f64> {
            solutions: Vec::new(),
            objective_bounds: ObjectiveBounds {
                min_values: Vec::new(),
                max_values: Vec::new(),
                ideal_point: Vec::new(),
                nadir_point: Vec::new(),
            },
            metrics: FrontMetrics {
                hypervolume: 0.0,
                spread: 0.0,
                spacing: 0.0,
                convergence: 0.0,
                num_solutions: 0,
                coverage: CoverageMetrics {
                    objective_space_coverage: 0.0,
                    reference_distance: 0.0,
                    epsilon_dominance: 0.0,
                },
            },
            generation: 0,
            last_updated: std::time::SystemTime::now(),
        };

        assert_eq!(pareto_front.solutions.len(), 0);
        assert_eq!(pareto_front.generation, 0);
    }

    #[test]
    fn test_hardware_resources() {
        let hardware = HardwareResources {
            max_memory_gb: 128.0,
            max_cpu_cores: 32,
            max_gpu_devices: 8,
            max_storage_gb: 2000.0,
            max_network_bandwidth: 10000.0,
            enable_cloud_scaling: false,
            cloud_budget: None,
            cpu_cores: 16,
            ram_gb: 64,
            num_gpus: 4,
            gpu_memory_gb: 32,
            storage_gb: 1000,
            network_bandwidth_mbps: 1000.0,
        };

        assert!(hardware.cpu_cores > 0);
        assert!(hardware.ram_gb > 0);
        assert!(hardware.num_gpus > 0);
    }
}

/// Example function demonstrating basic NAS usage
pub fn create_example_nas_config<T: Float + Debug + Send + Sync + 'static>() -> NASConfig<T> {
    NASConfig {
        search_strategy: SearchStrategyType::Evolutionary,
        search_space: SearchSpaceConfig {
            components: vec![
                OptimizerComponentConfig {
                    component_type: ComponentType::Adam,
                    hyperparameter_ranges: {
                        let mut ranges = std::collections::HashMap::new();
                        ranges.insert(
                            "learning_rate".to_string(),
                            ParameterRange::Continuous(0.0001, 0.1),
                        );
                        ranges.insert("beta1".to_string(), ParameterRange::Continuous(0.8, 0.99));
                        ranges.insert("beta2".to_string(), ParameterRange::Continuous(0.9, 0.999));
                        ranges
                    },
                    complexity_score: 1.0,
                    memory_requirement: 2048,
                    computational_cost: 1.5,
                    compatibility_constraints: Vec::new(),
                },
                OptimizerComponentConfig {
                    component_type: ComponentType::SGD,
                    hyperparameter_ranges: {
                        let mut ranges = std::collections::HashMap::new();
                        ranges.insert(
                            "learning_rate".to_string(),
                            ParameterRange::Continuous(0.001, 0.1),
                        );
                        ranges.insert(
                            "momentum".to_string(),
                            ParameterRange::Continuous(0.0, 0.99),
                        );
                        ranges
                    },
                    complexity_score: 0.5,
                    memory_requirement: 1024,
                    computational_cost: 1.0,
                    compatibility_constraints: Vec::new(),
                },
                OptimizerComponentConfig {
                    component_type: ComponentType::RMSprop,
                    hyperparameter_ranges: {
                        let mut ranges = std::collections::HashMap::new();
                        ranges.insert(
                            "learning_rate".to_string(),
                            ParameterRange::Continuous(0.0001, 0.1),
                        );
                        ranges.insert("decay".to_string(), ParameterRange::Continuous(0.8, 0.99));
                        ranges
                    },
                    complexity_score: 0.8,
                    memory_requirement: 1536,
                    computational_cost: 1.2,
                    compatibility_constraints: Vec::new(),
                },
            ],
            connection_patterns: vec![ConnectionPatternType::Sequential],
            learning_rate_schedules: LearningRateScheduleSpace::default(),
            regularization_techniques: RegularizationSpace::default(),
            adaptive_mechanisms: AdaptiveMechanismSpace::default(),
            memory_constraints: MemoryConstraints::default(),
            computation_constraints: ComputationConstraints::default(),
            component_types: vec![
                ComponentType::Adam,
                ComponentType::SGD,
                ComponentType::RMSprop,
            ],
            max_components: 10,
            min_components: 2,
            max_connections: 20,
        },
        evaluation_config: EvaluationConfig::default(),
        multi_objective_config: MultiObjectiveConfig::default(),
        search_budget: 500,
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 25,
            min_improvement: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()),
            metric: EvaluationMetric::FinalPerformance,
            target_performance: None,
            convergence_detection: ConvergenceDetectionStrategy::NoImprovement,
            convergence_strategy: ConvergenceDetectionStrategy::BestScore,
            min_generations: 10,
        },
        progressive_search: false,
        population_size: 25,
        enable_transfer_learning: false,
        encoding_strategy: ArchitectureEncodingStrategy::Direct,
        enable_performance_prediction: true,
        parallelization_factor: 2,
        auto_hyperparameter_tuning: true,
        resource_constraints: ResourceConstraints {
            max_memory_gb: num_traits::cast::cast(8.0).unwrap_or_else(|| T::zero()),
            max_computation_hours: num_traits::cast::cast(12.0).unwrap_or_else(|| T::zero()),
            max_energy_kwh: num_traits::cast::cast(50.0).unwrap_or_else(|| T::zero()),
            max_cost_usd: num_traits::cast::cast(500.0).unwrap_or_else(|| T::zero()),
            energy_constraints: Some(num_traits::cast::cast(50.0).unwrap_or_else(|| T::zero())),
            cost_constraints: Some(num_traits::cast::cast(500.0).unwrap_or_else(|| T::zero())),
            time_constraints: TimeConstraints::default(),
            hardware_resources: HardwareResources {
                cpu_cores: 8,
                ram_gb: 32,
                num_gpus: 2,
                gpu_memory_gb: 16,
                storage_gb: 500,
                network_bandwidth_mbps: 500.0,
                max_memory_gb: 32.0,
                max_cpu_cores: 8,
                max_gpu_devices: 2,
                max_storage_gb: 500.0,
                max_network_bandwidth: 500.0,
                enable_cloud_scaling: false,
                cloud_budget: None,
            },
            enable_monitoring: true,
            violation_handling: ResourceViolationHandling::Penalty,
        },
    }
}

/// Create a minimal NAS configuration for quick testing
pub fn create_minimal_nas_config<T: Float + Debug + Send + Sync + 'static>() -> NASConfig<T> {
    NASConfig {
        search_strategy: SearchStrategyType::Random,
        search_space: SearchSpaceConfig {
            components: vec![
                OptimizerComponentConfig {
                    component_type: ComponentType::Adam,
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    complexity_score: 1.0,
                    memory_requirement: 1024,
                    computational_cost: 1.0,
                    compatibility_constraints: Vec::new(),
                },
                OptimizerComponentConfig {
                    component_type: ComponentType::SGD,
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    complexity_score: 0.5,
                    memory_requirement: 512,
                    computational_cost: 0.5,
                    compatibility_constraints: Vec::new(),
                },
            ],
            connection_patterns: vec![ConnectionPatternType::Sequential],
            learning_rate_schedules: LearningRateScheduleSpace::default(),
            regularization_techniques: RegularizationSpace::default(),
            adaptive_mechanisms: AdaptiveMechanismSpace::default(),
            memory_constraints: MemoryConstraints::default(),
            computation_constraints: ComputationConstraints::default(),
            component_types: vec![ComponentType::Adam, ComponentType::SGD],
            max_components: 5,
            min_components: 1,
            max_connections: 10,
        },
        evaluation_config: EvaluationConfig::default(),
        multi_objective_config: MultiObjectiveConfig::default(),
        search_budget: 100,
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 10,
            min_improvement: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            metric: EvaluationMetric::FinalPerformance,
            target_performance: None,
            convergence_detection: ConvergenceDetectionStrategy::NoImprovement,
            convergence_strategy: ConvergenceDetectionStrategy::BestScore,
            min_generations: 5,
        },
        progressive_search: false,
        population_size: 10,
        enable_transfer_learning: false,
        encoding_strategy: ArchitectureEncodingStrategy::Direct,
        enable_performance_prediction: false,
        parallelization_factor: 1,
        auto_hyperparameter_tuning: false,
        resource_constraints: ResourceConstraints {
            max_memory_gb: num_traits::cast::cast(4.0).unwrap_or_else(|| T::zero()),
            max_computation_hours: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            max_energy_kwh: num_traits::cast::cast(5.0).unwrap_or_else(|| T::zero()),
            max_cost_usd: num_traits::cast::cast(50.0).unwrap_or_else(|| T::zero()),
            hardware_resources: HardwareResources {
                max_cpu_cores: 4,
                max_memory_gb: 16.0,
                max_gpu_devices: 1,
                max_storage_gb: 100.0,
                max_network_bandwidth: 100.0,
                enable_cloud_scaling: false,
                cloud_budget: None,
                cpu_cores: 4,
                ram_gb: 16,
                num_gpus: 1,
                gpu_memory_gb: 8,
                storage_gb: 100,
                network_bandwidth_mbps: 100.0,
            },
            time_constraints: TimeConstraints::default(),
            energy_constraints: Some(num_traits::cast::cast(5.0).unwrap_or_else(|| T::zero())),
            cost_constraints: Some(num_traits::cast::cast(50.0).unwrap_or_else(|| T::zero())),
            violation_handling: ResourceViolationHandling::Penalty,
            enable_monitoring: false,
        },
    }
}
