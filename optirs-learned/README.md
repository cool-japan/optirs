# OptiRS Learned

Learned optimizers and meta-learning for adaptive optimization in the OptiRS machine learning optimization library.

## Overview

OptiRS-Learned implements state-of-the-art learned optimization techniques that use neural networks to learn better optimization strategies. This crate provides meta-learning algorithms, learned optimizers, and adaptive optimization techniques that can outperform traditional hand-designed optimizers on specific tasks and domains.

## Features

- **Learned Optimizers**: Neural network-based optimizers that learn from experience
- **Meta-Learning**: Algorithms that learn to learn optimization strategies
- **Transformer-Based Optimizers**: Attention-based optimization with sequence modeling
- **LSTM Optimizers**: Recurrent neural network optimizers for sequential optimization
- **Few-Shot Learning**: Quick adaptation to new optimization tasks
- **Domain-Specific Adaptation**: Optimizers that specialize for specific problem domains
- **Online Learning**: Continuous adaptation during training
- **Transfer Learning**: Knowledge transfer between optimization tasks

## Learned Optimizer Types

### Neural Optimizers
- **MLP Optimizers**: Multi-layer perceptron-based optimization rules
- **Transformer Optimizers**: Self-attention mechanisms for parameter updates
- **LSTM Optimizers**: Long short-term memory networks for optimization
- **Graph Neural Network Optimizers**: Exploiting computational graph structure
- **Hybrid Optimizers**: Combining learned and traditional components

### Meta-Learning Approaches
- **MAML (Model-Agnostic Meta-Learning)**: Generic meta-learning for optimizers
- **Learned Learning Rates**: Neural networks that predict optimal learning rates
- **Gradient-Based Meta-Learning**: Learning optimization rules through gradients
- **Memory-Augmented Optimizers**: External memory for optimization history
- **Few-Shot Optimizer Adaptation**: Quick specialization for new domains

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-learned = "0.1.0-beta.3"
scirs2-core = "0.1.0-rc.2"  # Required foundation
```

### Feature Selection

Enable specific learned optimizer types:

```toml
[dependencies]
optirs-learned = { version = "0.1.0-beta.3", features = ["transformer", "lstm", "meta_learning"] }
```

Available features:
- `transformer`: Transformer-based optimizers (enabled by default)
- `lstm`: LSTM-based optimizers
- `meta_learning`: Meta-learning algorithms
- `autograd_integration`: Automatic differentiation integration
- `nlp`: Natural language processing utilities for tokenization

## Usage

### Basic Learned Optimizer

```rust
use optirs_learned::{LearnedOptimizer, TransformerOptimizer};
use optirs_core::optimizers::OptimizerConfig;

// Create a transformer-based learned optimizer
let mut learned_optimizer = TransformerOptimizer::new()
    .with_hidden_size(256)
    .with_num_layers(4)
    .with_num_heads(8)
    .with_sequence_length(100)
    .build()?;

// Train the optimizer on a meta-training set
let meta_training_tasks = load_meta_training_tasks()?;
learned_optimizer.meta_train(&meta_training_tasks).await?;

// Use the learned optimizer for a new task
let mut params = create_model_parameters()?;
let grads = compute_gradients(&params, &training_data)?;

// The optimizer learns and adapts during training
learned_optimizer.step(&mut params, &grads).await?;
```

### LSTM-Based Optimizer

```rust
use optirs_learned::{LstmOptimizer, OptimizerMemory};

// Create LSTM optimizer with memory
let mut lstm_optimizer = LstmOptimizer::new()
    .with_hidden_size(128)
    .with_num_layers(2)
    .with_memory(OptimizerMemory::new(1000))
    .with_forget_gate_bias(1.0)
    .build()?;

// The LSTM maintains state across optimization steps
for epoch in 0..100 {
    for batch in training_data.batches() {
        let grads = compute_batch_gradients(&params, &batch)?;
        lstm_optimizer.step(&mut params, &grads).await?;

        // LSTM state carries forward optimization knowledge
    }
}
```

### Meta-Learning Example

```rust
use optirs_learned::{MetaLearner, MAMLOptimizer, TaskDistribution};

// Setup meta-learning for optimizer adaptation
let meta_learner = MAMLOptimizer::new()
    .with_inner_steps(5)
    .with_inner_lr(0.01)
    .with_outer_lr(0.001)
    .build()?;

// Define task distribution for meta-learning
let task_distribution = TaskDistribution::new()
    .add_domain("computer_vision", 0.4)
    .add_domain("nlp", 0.3)
    .add_domain("speech", 0.3)
    .build();

// Meta-train across multiple task domains
for meta_epoch in 0..1000 {
    let task_batch = task_distribution.sample_tasks(16)?;

    for task in task_batch {
        // Inner loop: adapt to specific task
        let adapted_optimizer = meta_learner.adapt_to_task(&task).await?;

        // Evaluate adapted optimizer
        let task_loss = adapted_optimizer.evaluate(&task.test_data).await?;

        // Outer loop: update meta-parameters
        meta_learner.meta_update(&task_loss).await?;
    }
}
```

### Domain-Specific Specialization

```rust
use optirs_learned::{DomainSpecificOptimizer, OptimizationDomain};

// Create optimizers specialized for different domains
let vision_optimizer = DomainSpecificOptimizer::new()
    .for_domain(OptimizationDomain::ComputerVision)
    .with_architecture("resnet_specialized")
    .with_data_augmentation_aware(true)
    .build()?;

let nlp_optimizer = DomainSpecificOptimizer::new()
    .for_domain(OptimizationDomain::NaturalLanguageProcessing)
    .with_architecture("transformer_specialized")
    .with_attention_pattern_aware(true)
    .with_tokenizer_integration(true)
    .build()?;

// Optimizers automatically adapt to domain characteristics
vision_optimizer.optimize_cnn_model(&mut cnn_params, &image_data).await?;
nlp_optimizer.optimize_transformer_model(&mut transformer_params, &text_data).await?;
```

### Online Learning and Adaptation

```rust
use optirs_learned::{OnlineLearner, AdaptiveOptimizer};

// Create optimizer that continuously learns during training
let mut adaptive_optimizer = OnlineLearner::new()
    .with_adaptation_rate(0.01)
    .with_memory_size(10000)
    .with_exploration_rate(0.1)
    .build()?;

// Optimizer adapts based on observed performance
for training_step in 0..100000 {
    let grads = compute_gradients(&params, &current_batch)?;

    // Optimizer learns from its own performance
    let step_result = adaptive_optimizer.step(&mut params, &grads).await?;

    // Provide feedback to improve future decisions
    let performance_feedback = evaluate_step_quality(&step_result)?;
    adaptive_optimizer.update_from_feedback(&performance_feedback).await?;
}
```

## Architecture

### Learned Optimizer Components
- **Neural Architecture**: Configurable neural network architectures for optimization
- **Memory Systems**: External memory for storing optimization history
- **Attention Mechanisms**: Self-attention for parameter importance weighting
- **Adaptation Layers**: Quick adaptation to new tasks and domains
- **Meta-Learning Framework**: Learning to learn optimization strategies

### Training Infrastructure
- **Meta-Training Pipeline**: Distributed training across multiple tasks
- **Task Sampling**: Intelligent sampling of training tasks
- **Gradient Computation**: Efficient second-order gradient computation
- **Checkpointing**: Save and restore learned optimizer states
- **Evaluation Framework**: Comprehensive evaluation on diverse tasks

## Advanced Features

### Automatic Hyperparameter Tuning
```rust
use optirs_learned::{AutoHyperparameterTuner, HyperparameterSpace};

let hyperparameter_space = HyperparameterSpace::new()
    .add_learning_rate_range(1e-5, 1e-1)
    .add_batch_size_options(&[16, 32, 64, 128])
    .add_architecture_options(&["small", "medium", "large"])
    .build();

let tuner = AutoHyperparameterTuner::new()
    .with_search_space(hyperparameter_space)
    .with_budget(100)  // Number of trials
    .with_parallel_evaluations(4)
    .build();

let optimal_config = tuner.find_optimal_hyperparameters(&task).await?;
```

### Neural Architecture Search for Optimizers
```rust
use optirs_learned::{OptimizerNAS, ArchitectureSearchSpace};

let search_space = ArchitectureSearchSpace::new()
    .add_layer_types(&["linear", "attention", "lstm", "gru"])
    .add_hidden_sizes(&[64, 128, 256, 512])
    .add_activation_functions(&["relu", "gelu", "swish"])
    .build();

let nas = OptimizerNAS::new()
    .with_search_space(search_space)
    .with_performance_predictor(PerformancePredictor::default())
    .build();

let optimal_architecture = nas.search_architecture(&meta_training_tasks).await?;
```

## Integration with Traditional Optimizers

### Hybrid Optimizers
```rust
use optirs_learned::{HybridOptimizer, TraditionalOptimizer};
use optirs_core::optimizers::Adam;

// Combine learned and traditional optimizers
let hybrid = HybridOptimizer::new()
    .with_learned_component(TransformerOptimizer::default())
    .with_traditional_component(Adam::new(0.001))
    .with_mixing_strategy(MixingStrategy::AdaptiveWeighting)
    .build()?;

// The optimizer learns when to use each component
hybrid.step(&mut params, &grads).await?;
```

## Performance Monitoring

### Learned Optimizer Analytics
```rust
use optirs_learned::{OptimizerAnalytics, LearningCurveTracker};

let analytics = OptimizerAnalytics::new()
    .with_metrics(&["convergence_speed", "final_performance", "stability"])
    .with_visualization(true)
    .build();

let tracker = LearningCurveTracker::new()
    .with_smoothing_window(100)
    .with_trend_detection(true)
    .build();

// Track optimizer learning progress
analytics.track_optimization_step(&step_result).await?;
tracker.update(&current_loss, &learning_metrics).await?;
```

## Research and Experimental Features

### Continual Learning Optimizers
- Optimizers that avoid catastrophic forgetting
- Elastic weight consolidation for optimization rules
- Progressive neural optimizer architectures
- Experience replay for optimization strategies

### Multi-Task Optimizers
- Shared optimization knowledge across tasks
- Task-specific adaptation layers
- Cross-domain knowledge transfer
- Meta-learning for multi-task scenarios

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS repository for contribution guidelines.

## Research Papers and References

This crate implements techniques from various research papers:
- "Learning to Learn by Gradient Descent by Gradient Descent" (Andrychowicz et al.)
- "Learned Optimizers that Scale and Generalize" (Metz et al.)
- "Tasks, stability, architecture, and compute: Training more effective learned optimizers" (Metz et al.)
- "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (Finn et al.)

## License

This project is licensed under either of:
- Apache License, Version 2.0
- MIT License

at your option.