# OptiRS NAS

Neural Architecture Search and hyperparameter optimization for the OptiRS machine learning optimization library.

## Overview

OptiRS-NAS provides comprehensive Neural Architecture Search (NAS) capabilities combined with hyperparameter optimization. This crate implements state-of-the-art NAS algorithms, Bayesian optimization, evolutionary search, and reinforcement learning-based approaches to automatically design optimal neural network architectures and hyperparameters.

## Features

- **Neural Architecture Search**: Automated neural network design
- **Bayesian Optimization**: Gaussian process-based hyperparameter tuning
- **Evolutionary Algorithms**: Population-based architecture evolution
- **Reinforcement Learning NAS**: RL agents for architecture design
- **Progressive Search**: Gradually growing architecture complexity
- **Multi-Objective Optimization**: Balancing accuracy, efficiency, and resource usage
- **Hardware-Aware NAS**: Architecture optimization for specific hardware
- **Transfer Learning**: Knowledge transfer across architecture searches

## Search Strategies

### Bayesian Optimization
- **Gaussian Process**: Probabilistic surrogate models
- **Acquisition Functions**: Expected improvement, upper confidence bound
- **Multi-Fidelity**: Early stopping and progressive evaluation
- **Constrained Optimization**: Resource-aware search
- **Parallel Evaluation**: Batch Bayesian optimization

### Evolutionary Algorithms
- **Genetic Algorithms**: Population-based architecture evolution
- **Differential Evolution**: Continuous hyperparameter optimization
- **NSGA-II**: Multi-objective Pareto frontier exploration
- **CMA-ES**: Covariance matrix adaptation evolution strategy
- **Population Diversity**: Maintaining search diversity

### Reinforcement Learning
- **Controller Networks**: RNN/Transformer controllers for architecture generation
- **REINFORCE**: Policy gradient-based architecture search
- **Progressive NAS**: Gradually increasing architecture complexity
- **Distributed Training**: Parallel architecture evaluation
- **Efficient NAS**: Resource-constrained architecture search

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-nas = "0.1.0-beta.3"
scirs2-core = "0.1.0-rc.2"  # Required foundation
```

### Feature Selection

Enable specific NAS approaches:

```toml
[dependencies]
optirs-nas = { version = "0.1.0-beta.3", features = ["bayesian", "evolutionary", "reinforcement"] }
```

Available features:
- `bayesian`: Bayesian optimization methods (enabled by default)
- `evolutionary`: Evolutionary algorithms
- `reinforcement`: Reinforcement learning approaches
- `progressive`: Progressive search strategies
- `multi_objective`: Multi-objective optimization
- `learned_integration`: Integration with OptiRS-Learned

## Usage

### Basic Bayesian Optimization

```rust
use optirs_nas::{BayesianOptimizer, SearchSpace, GaussianProcess};

// Define hyperparameter search space
let search_space = SearchSpace::new()
    .add_continuous("learning_rate", 1e-5, 1e-1)
    .add_discrete("batch_size", &[16, 32, 64, 128, 256])
    .add_categorical("optimizer", &["sgd", "adam", "adamw"])
    .add_discrete("hidden_layers", &[1, 2, 3, 4, 5])
    .add_continuous("dropout_rate", 0.0, 0.5)
    .build();

// Create Bayesian optimizer
let mut bayesian_opt = BayesianOptimizer::new()
    .with_search_space(search_space)
    .with_surrogate_model(GaussianProcess::default())
    .with_acquisition_function(ExpectedImprovement::new())
    .with_budget(100)  // Number of evaluations
    .build()?;

// Run optimization
for iteration in 0..100 {
    let candidate = bayesian_opt.suggest_next_candidate()?;

    // Evaluate the candidate configuration
    let performance = evaluate_configuration(&candidate).await?;

    // Update surrogate model
    bayesian_opt.update(&candidate, performance)?;

    println!("Iteration {}: Best performance = {:.4}",
             iteration, bayesian_opt.best_performance());
}

let best_config = bayesian_opt.best_configuration()?;
```

### Neural Architecture Search

```rust
use optirs_nas::{ArchitectureSearchSpace, NASController, ReinforcementLearningNAS};

// Define architecture search space
let arch_space = ArchitectureSearchSpace::new()
    .add_layer_types(&["conv2d", "depthwise_conv2d", "separable_conv2d"])
    .add_kernel_sizes(&[3, 5, 7])
    .add_channel_sizes(&[16, 32, 64, 128, 256])
    .add_activation_functions(&["relu", "gelu", "swish"])
    .add_normalization_types(&["batch_norm", "layer_norm", "group_norm"])
    .with_max_depth(20)
    .with_skip_connections(true)
    .build();

// Create RL-based NAS controller
let mut nas_controller = ReinforcementLearningNAS::new()
    .with_controller(LSTMController::new(256, 2))
    .with_search_space(arch_space)
    .with_reward_function(AccuracyEfficiencyReward::new())
    .with_training_episodes(1000)
    .build()?;

// Search for optimal architectures
let best_architectures = nas_controller.search().await?;

for (rank, arch) in best_architectures.iter().enumerate() {
    println!("Rank {}: Accuracy = {:.2}%, FLOPs = {:.2}M",
             rank + 1, arch.accuracy * 100.0, arch.flops / 1e6);
}
```

### Multi-Objective Optimization

```rust
use optirs_nas::{MultiObjectiveOptimizer, ParetoFrontier, Objective};

// Define multiple objectives
let objectives = vec![
    Objective::Maximize("accuracy"),
    Objective::Minimize("latency"),
    Objective::Minimize("memory_usage"),
    Objective::Minimize("power_consumption"),
];

let mut multi_obj = MultiObjectiveOptimizer::new()
    .with_objectives(objectives)
    .with_algorithm(NSGA2::default())
    .with_population_size(100)
    .with_generations(50)
    .build()?;

// Run multi-objective optimization
let pareto_frontier = multi_obj.optimize().await?;

// Analyze trade-offs
for solution in pareto_frontier.solutions() {
    println!("Accuracy: {:.2}%, Latency: {:.2}ms, Memory: {:.2}MB",
             solution.accuracy * 100.0,
             solution.latency,
             solution.memory_mb);
}
```

### Progressive NAS

```rust
use optirs_nas::{ProgressiveNAS, ArchitectureGrowthStrategy};

// Progressive architecture search with growing complexity
let mut progressive_nas = ProgressiveNAS::new()
    .with_initial_depth(3)
    .with_max_depth(15)
    .with_growth_strategy(ArchitectureGrowthStrategy::Gradual)
    .with_complexity_budget(1e9)  // FLOPs budget
    .build()?;

// Search with gradually increasing complexity
for stage in 0..5 {
    let stage_results = progressive_nas.search_stage(stage).await?;

    println!("Stage {}: Found {} architectures",
             stage, stage_results.len());

    // Progress to next complexity level
    progressive_nas.advance_complexity()?;
}
```

### Hardware-Aware Optimization

```rust
use optirs_nas::{HardwareAwareNAS, TargetHardware, LatencyPredictor};

// Define target hardware constraints
let target_hardware = TargetHardware::new()
    .with_device_type("mobile_gpu")
    .with_memory_limit(4_000_000_000)  // 4GB
    .with_power_limit(5.0)  // 5W
    .with_latency_constraint(50.0)  // 50ms
    .build();

let latency_predictor = LatencyPredictor::from_hardware_profile(&target_hardware)?;

let mut hw_aware_nas = HardwareAwareNAS::new()
    .with_target_hardware(target_hardware)
    .with_latency_predictor(latency_predictor)
    .with_accuracy_threshold(0.85)
    .build()?;

// Search for hardware-efficient architectures
let efficient_architectures = hw_aware_nas.search().await?;
```

### Transfer Learning for NAS

```rust
use optirs_nas::{TransferableNAS, ArchitectureKnowledge};

// Load pre-trained architecture knowledge
let knowledge_base = ArchitectureKnowledge::load_from_file("imagenet_nas_knowledge.json")?;

let mut transferable_nas = TransferableNAS::new()
    .with_source_knowledge(knowledge_base)
    .with_transfer_strategy(TransferStrategy::FineTuning)
    .with_adaptation_budget(20)  // Limited budget for new domain
    .build()?;

// Quickly adapt to new domain
let adapted_architectures = transferable_nas
    .adapt_to_new_domain(&target_dataset)
    .await?;
```

## Architecture Components

### Search Space Definition
- **Layer Types**: Convolution, attention, normalization, activation
- **Connection Patterns**: Skip connections, dense connections, routing
- **Hyperparameters**: Learning rates, regularization, optimization settings
- **Resource Constraints**: FLOPs, memory, latency, energy consumption

### Evaluation Framework
- **Performance Metrics**: Accuracy, F1-score, AUC, perplexity
- **Efficiency Metrics**: FLOPs, parameters, memory usage
- **Hardware Metrics**: Latency, throughput, power consumption
- **Multi-Fidelity**: Early stopping, progressive evaluation, proxy tasks

### Search Algorithms
- **Surrogate Models**: Gaussian processes, neural networks, tree-based models
- **Acquisition Functions**: Exploration-exploitation trade-offs
- **Population Methods**: Genetic algorithms, particle swarm optimization
- **Gradient-Based**: Differentiable architecture search (DARTS)

## Advanced Features

### Distributed Search
```rust
use optirs_nas::{DistributedNAS, WorkerPool};

// Distribute architecture evaluation across multiple workers
let worker_pool = WorkerPool::new()
    .with_workers(8)
    .with_gpu_per_worker(1)
    .build()?;

let distributed_nas = DistributedNAS::new()
    .with_worker_pool(worker_pool)
    .with_load_balancing(LoadBalancing::RoundRobin)
    .build()?;

let results = distributed_nas.parallel_search().await?;
```

### AutoML Pipeline
```rust
use optirs_nas::{AutoMLPipeline, DataPreprocessing, ModelSelection};

// Complete automated machine learning pipeline
let automl = AutoMLPipeline::new()
    .with_data_preprocessing(DataPreprocessing::default())
    .with_feature_engineering(FeatureEngineering::automated())
    .with_architecture_search(BayesianOptimizer::default())
    .with_hyperparameter_tuning(HyperparameterOptimizer::default())
    .with_model_selection(ModelSelection::cross_validation(5))
    .build()?;

let optimized_model = automl.fit(&training_data).await?;
```

## Performance Monitoring

```rust
use optirs_nas::{NASMonitor, SearchAnalytics};

let monitor = NASMonitor::new()
    .with_metrics_collection(true)
    .with_visualization(true)
    .with_early_stopping(EarlyStopping::patience(10))
    .build();

let analytics = SearchAnalytics::new()
    .track_convergence()
    .track_diversity()
    .track_efficiency()
    .build();

// Monitor search progress
monitor.track_search_progress(&nas_results).await?;
analytics.generate_search_report().await?;
```

## Integration with OptiRS Ecosystem

### Learned Optimizer Integration
```rust
use optirs_nas::learned_integration::LearnedNAS;
use optirs_learned::MetaLearner;

// Combine NAS with learned optimizers
let learned_nas = LearnedNAS::new()
    .with_meta_learner(MetaLearner::default())
    .with_architecture_adaptation(true)
    .build()?;

let co_optimized_result = learned_nas
    .co_optimize_architecture_and_optimizer(&task)
    .await?;
```

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS repository for contribution guidelines.

## Research Papers and References

This crate implements techniques from various research papers:
- "Neural Architecture Search with Reinforcement Learning" (Zoph & Le)
- "Efficient Neural Architecture Search via Parameter Sharing" (Pham et al.)
- "DARTS: Differentiable Architecture Search" (Liu et al.)
- "Progressive Neural Architecture Search" (Liu et al.)
- "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (Howard et al.)

## License

This project is licensed under either of:
- Apache License, Version 2.0
- MIT License

at your option.