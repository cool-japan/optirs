# OptiRS Learned

Learned optimizers and meta-learning for adaptive optimization in the OptiRS machine learning optimization library.

## Overview

OptiRS-Learned implements neural-network-based optimizers and gradient-based meta-learning
algorithms (MAML, Reptile, Meta-SGD) on top of [`optirs-core`](../optirs-core). It is labeled
**research-grade**: learned optimizers are sensitive to the distribution of tasks they were
meta-trained on, so benchmark against `optirs-core`'s hand-designed optimizers on your own
workload before depending on a learned one in production.

## Implementation Status (v0.3.3)

- Transformer-based optimizers (`transformer`, `transformer_based_optimizer` - self-/cross-attention over parameters)
- LSTM optimizer (`lstm` - recurrent per-parameter update rule)
- Meta-learning framework (`meta_learning` - MAML, Reptile, Meta-SGD)
- Graph-neural-network optimizer (`gnn_optimizer`) and Neural Turing Machine optimizer (`ntm_optimizer`)
- Differentiable optimizer search (`darts_optimizer_search` - DARTS-style)
- Forward/reverse-mode autodiff engines (`forward_mode`, `reverse_mode`)
- Continual learning (`continual_learning` - elastic weight consolidation, progressive networks), few-shot,
  zero-shot and realtime drift adaptation (`realtime_adaptation`)
- Domain-specific optimizers (`domain_optimizers` - vision, NLP, attention-pattern), cross-domain
  transfer (`cross_domain_transfer`), online continual meta-learning (`online_maml`)
- Quantum-inspired optimizers (`quantum_learned`, adapting `optirs-core`'s quantum annealing /
  variational quantum optimizer)

These are real, tested implementations - the research-grade label reflects the
distribution-sensitivity caveat above, not stub code.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-learned = "0.3.3"
scirs2-core = "0.6.5"  # Required foundation
```

### Feature Flags

| Feature | Gates | Default |
|---|---|---|
| `transformer` | `adaptive`, `transformer`, `transformer_based_optimizer` modules and the `TransformerOptimizer` / `TransformerBasedOptimizer` re-exports | yes |
| `lstm` | `lstm` module and the `LSTMOptimizer` re-export | yes |
| `meta_learning` | `meta_learning` module (MAML, Reptile, Meta-SGD) | yes |

All three are on by default, so a plain `optirs-learned = "0.3"` dependency is unchanged; each
feature genuinely gates its modules, so `--no-default-features --features <one family>` is a
real, smaller build. Everything else (`common`, `continual_learning`, `darts_optimizer_search`,
`forward_mode`, `reverse_mode`, `gnn_optimizer`, `higher_order`, `ntm_optimizer`, `online_maml`,
`quantum_learned`, `realtime_adaptation`, `zero_shot`, `few_shot`, `es_meta_training`,
`domain_objectives`, `domain_optimizers`, `episodic_memory_impl`, `cross_domain_transfer`) is
ungated.

```toml
[dependencies]
optirs-learned = { version = "0.3.3", default-features = false, features = ["lstm"] }
```

## Usage

### Transformer-Based Optimizer

Requires the `transformer` feature (on by default). Parameters and gradients are keyed by
name, one entry per tensor, matching how a real training loop hands over named layers rather
than a single flat array.

```rust
use optirs_learned::transformer::{TransformerOptimizer, TransformerOptimizerConfig};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a transformer-based optimizer (default architecture)
    let config = TransformerOptimizerConfig::default();
    let mut optimizer = TransformerOptimizer::<f64>::new(config)?;

    let mut params = HashMap::new();
    params.insert("layer1.weight".to_string(), Array2::from_elem((4, 4), 1.0));

    let mut grads = HashMap::new();
    grads.insert("layer1.weight".to_string(), Array2::from_elem((4, 4), 0.01));

    // `step` takes the current loss and hands it back, so callers can chain steps
    let loss = optimizer.step(&mut params, &mut grads, 0.5)?;
    println!("{loss}");
    Ok(())
}
```

For the LSTM optimizer (`lstm` feature) and the MAML / Reptile / Meta-SGD meta-learning
framework (`meta_learning` feature), see the crate documentation
(`cargo doc -p optirs-learned --open`) - their configuration types are more involved than fit
comfortably in a README snippet, and this file's examples are verified against the real API
rather than sketched from memory.

## Modules

### Neural Optimizers
- **Transformer** (`transformer`, `transformer_based_optimizer`) - self-/cross-attention over parameters
- **LSTM** (`lstm::LSTMOptimizer`) - recurrent per-parameter update rule
- **Graph Neural Network** (`gnn_optimizer::GnnOptimizer`) - message-passing over the parameter/layer graph
- **Neural Turing Machine** (`ntm_optimizer::NtmOptimizer`) - external memory with content/location/hybrid addressing

### Meta-Learning (`meta_learning` feature)
- **MAML** - Model-Agnostic Meta-Learning (SecondOrder/FirstOrder/Reptile variants)
- **Reptile** - First-order meta-learning
- **Meta-SGD** - Learn per-parameter learning rates

### Differentiable / Autodiff
- **DARTS-style optimizer search** (`darts_optimizer_search`) - softmax architecture weights over update primitives
- **Forward-mode autodiff** (`forward_mode::ForwardModeEngine`) - dual numbers
- **Reverse-mode autodiff** (`reverse_mode::ReverseModeEngine`)
- **Higher-order derivatives** (`higher_order`) - Hessian-vector products, mixed partials

### Domain-Specific and Transfer (`domain_optimizers`, `cross_domain_transfer`)
- **CVOptimizer**, **NLPOptimizer**, **AttentionOptimizer** - vision / NLP / attention-pattern specializations
- **CrossDomainTransfer** - inter-domain similarity and knowledge transfer

### Few-Shot and Online Adaptation (`few_shot`, `online_maml`, `realtime_adaptation`)
- **PrototypicalNetwork**, **TaskSimilarityCalculator**, **EpisodicMemoryBank**, **FastAdaptationEngine**
- **OnlineMAML** - continuous task-stream meta-learning with staleness decay
- **DriftDetector** / **RealtimeAdaptationController** - EWMA + CUSUM drift detection driving an AIMD LR/momentum controller

### Continual Learning (`continual_learning`)
- **ElasticWeightConsolidation**, **ProgressiveNetworks**

### Quantum-Inspired (`quantum_learned`)
- **QuantumLearnedOptimizer** - adapts `optirs-core`'s quantum annealing / hybrid quantum-classical optimizers
- `QuantumBackend::Variational` - wraps `optirs-core`'s variational quantum optimizer (SPSA)

### Zero-Shot Selection (`zero_shot`)
- **ZeroShotSelector** - gradient/landscape meta-features to an offline-fit optimizer classifier and learning-rate regressor

## Architecture

Built on [`optirs-core`](../optirs-core) and `scirs2-core` (`scirs2_core::ndarray`,
`scirs2_core::numeric`, `scirs2_core::random`) - no direct `ndarray`/`rand` dependency.

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS repository for contribution guidelines.

## Research Papers and References

This crate implements techniques from various research papers:
- "Learning to Learn by Gradient Descent by Gradient Descent" (Andrychowicz et al.)
- "Learned Optimizers that Scale and Generalize" (Metz et al.)
- "Tasks, stability, architecture, and compute: Training more effective learned optimizers" (Metz et al.)
- "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (Finn et al.)

## License

This project is licensed under the Apache License, Version 2.0.
