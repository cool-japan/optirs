//! Architecture representation and search space definition for Neural Architecture Search.
//!
//! This module provides the core architecture representation used in NAS,
//! including the search space definition, architecture encoding/decoding,
//! and mutation/crossover operations for evolutionary algorithms.

use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Represents an optimizer architecture in the search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    /// Unique identifier for this architecture
    pub id: String,

    /// List of optimizer components in the architecture
    pub components: Vec<OptimizerComponent>,

    /// Connections between components
    pub connections: Vec<Connection>,

    /// Architecture-level hyperparameters
    pub hyperparameters: HashMap<String, f64>,

    /// Performance metrics from evaluation
    pub performance: Option<PerformanceMetrics>,

    /// Generation number (for evolutionary algorithms)
    pub generation: u32,
}

/// Individual optimizer component within an architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerComponent {
    /// Unique ID within the architecture
    pub id: String,

    /// Type of optimizer component
    pub component_type: ComponentType,

    /// Component-specific hyperparameters
    pub hyperparameters: HashMap<String, f64>,

    /// Position in the architecture graph
    pub position: ComponentPosition,

    /// Whether this component is enabled
    pub enabled: bool,
}

/// Types of optimizer components that can be used in architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    /// Stochastic Gradient Descent
    SGD,
    /// Adam optimizer
    Adam,
    /// AdamW (Adam with weight decay)
    AdamW,
    /// RMSprop optimizer
    RMSprop,
    /// AdaGrad optimizer
    AdaGrad,
    /// AdaDelta optimizer
    AdaDelta,
    /// Momentum component
    Momentum,
    /// Nesterov momentum
    Nesterov,
    /// Learning rate scheduler
    LRScheduler,
    /// Gradient clipping
    GradientClipping,
    /// Batch normalization
    BatchNorm,
    /// Dropout regularization
    Dropout,
    /// LAMB optimizer
    LAMB,
    /// LARS optimizer
    LARS,
    /// Lion optimizer
    Lion,
    /// RAdam optimizer
    RAdam,
    /// Lookahead optimizer
    Lookahead,
    /// SAM optimizer
    SAM,
    /// L-BFGS optimizer
    LBFGS,
    /// Sparse Adam optimizer
    SparseAdam,
    /// Grouped Adam optimizer
    GroupedAdam,
    /// MAML optimizer
    MAML,
    /// L1 regularizer
    L1Regularizer,
    /// L2 regularizer
    L2Regularizer,
    /// Elastic Net regularizer
    ElasticNetRegularizer,
    /// Dropout regularizer
    DropoutRegularizer,
    /// Weight decay
    WeightDecay,
    /// Adaptive learning rate
    AdaptiveLR,
    /// Adaptive momentum
    AdaptiveMomentum,
    /// Adaptive regularization
    AdaptiveRegularization,
    /// LSTM optimizer
    LSTMOptimizer,
    /// Transformer optimizer
    TransformerOptimizer,
    /// Attention optimizer
    AttentionOptimizer,
    /// Meta SGD
    MetaSGD,
    /// Constant learning rate
    ConstantLR,
    /// Exponential learning rate
    ExponentialLR,
    /// Step learning rate
    StepLR,
    /// Cosine annealing learning rate
    CosineAnnealingLR,
    /// One cycle learning rate
    OneCycleLR,
    /// Cyclic learning rate
    CyclicLR,
    /// Reptile optimizer
    Reptile,
    /// Custom component
    Custom,
}

/// Position and layout information for a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPosition {
    /// Layer index in the architecture
    pub layer: u32,
    /// Position within the layer
    pub index: u32,
    /// Spatial coordinates (for visualization)
    pub x: f32,
    pub y: f32,
}

/// Connection between two components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Source component ID
    pub from: String,
    /// Target component ID
    pub to: String,
    /// Connection weight/strength
    pub weight: f64,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Whether this connection is enabled
    pub enabled: bool,
}

/// Types of connections between components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Direct gradient flow
    Gradient,
    /// Parameter update flow
    Parameter,
    /// Skip connection
    Skip,
    /// Residual connection
    Residual,
    /// Information flow (for meta-learning)
    Information,
}

/// Performance metrics for an evaluated architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Final optimization performance
    pub final_performance: f64,
    /// Convergence speed (iterations to convergence)
    pub convergence_speed: f64,
    /// Stability (variance in performance)
    pub stability: f64,
    /// Computational efficiency
    pub efficiency: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// Robustness across problem instances
    pub robustness: f64,
    /// Overall score (weighted combination)
    pub overall_score: f64,
}

/// Search space definition for architecture generation
#[derive(Debug, Clone)]
pub struct ArchitectureSpace {
    /// Available component types
    pub component_types: Vec<ComponentType>,

    /// Minimum number of components
    pub min_components: usize,

    /// Maximum number of components
    pub max_components: usize,

    /// Maximum number of connections
    pub max_connections: usize,

    /// Whether cycles are allowed in the architecture graph
    pub allow_cycles: bool,

    /// Maximum depth of the architecture
    pub max_depth: u32,

    /// Hyperparameter ranges for each component type
    pub hyperparameter_ranges: HashMap<ComponentType, HashMap<String, (f64, f64)>>,

    /// Probability of each component type being selected
    pub component_probabilities: HashMap<ComponentType, f64>,

    /// Connection probability between components
    pub connection_probability: f64,
}

impl ArchitectureSpace {
    /// Create a new architecture space with default settings
    pub fn new() -> Self {
        let component_types = vec![
            ComponentType::SGD,
            ComponentType::Adam,
            ComponentType::AdamW,
            ComponentType::RMSprop,
            ComponentType::Momentum,
        ];

        let mut component_probabilities = HashMap::new();
        component_probabilities.insert(ComponentType::Adam, 0.3);
        component_probabilities.insert(ComponentType::AdamW, 0.25);
        component_probabilities.insert(ComponentType::SGD, 0.2);
        component_probabilities.insert(ComponentType::RMSprop, 0.15);
        component_probabilities.insert(ComponentType::Momentum, 0.1);

        let mut hyperparameter_ranges = HashMap::new();

        // Adam hyperparameters
        let mut adam_ranges = HashMap::new();
        adam_ranges.insert("learning_rate".to_string(), (1e-5, 1e-1));
        adam_ranges.insert("beta1".to_string(), (0.8, 0.99));
        adam_ranges.insert("beta2".to_string(), (0.9, 0.9999));
        adam_ranges.insert("epsilon".to_string(), (1e-8, 1e-6));
        hyperparameter_ranges.insert(ComponentType::Adam, adam_ranges);

        // SGD hyperparameters
        let mut sgd_ranges = HashMap::new();
        sgd_ranges.insert("learning_rate".to_string(), (1e-4, 1e-1));
        sgd_ranges.insert("momentum".to_string(), (0.0, 0.99));
        hyperparameter_ranges.insert(ComponentType::SGD, sgd_ranges);

        // RMSprop hyperparameters
        let mut rmsprop_ranges = HashMap::new();
        rmsprop_ranges.insert("learning_rate".to_string(), (1e-5, 1e-1));
        rmsprop_ranges.insert("alpha".to_string(), (0.9, 0.999));
        rmsprop_ranges.insert("epsilon".to_string(), (1e-8, 1e-6));
        hyperparameter_ranges.insert(ComponentType::RMSprop, rmsprop_ranges);

        Self {
            component_types,
            min_components: 1,
            max_components: 10,
            max_connections: 20,
            allow_cycles: false,
            max_depth: 5,
            hyperparameter_ranges,
            component_probabilities,
            connection_probability: 0.3,
        }
    }

    /// Generate a random architecture within this search space
    pub fn generate_random_architecture(&self) -> Architecture {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::Random::default();

        let num_components = rng.gen_range(self.min_components..=self.max_components);
        let mut components = Vec::new();

        for i in 0..num_components {
            let component_type = self.component_types[rng.gen_range(0..self.component_types.len())];
            let mut hyperparameters = HashMap::new();

            if let Some(ranges) = self.hyperparameter_ranges.get(&component_type) {
                for (param_name, (min_val, max_val)) in ranges {
                    let value = rng.gen_range(*min_val..=*max_val);
                    hyperparameters.insert(param_name.clone(), value);
                }
            }

            components.push(OptimizerComponent {
                id: format!("comp_{}", i),
                component_type,
                hyperparameters,
                position: ComponentPosition {
                    layer: i as u32,
                    index: 0,
                    x: i as f32 * 100.0,
                    y: 0.0,
                },
                enabled: true,
            });
        }

        let mut connections = Vec::new();
        let max_connections = self
            .max_connections
            .min(num_components * (num_components - 1));
        let num_connections = rng.gen_range(0..=max_connections);

        for _ in 0..num_connections {
            if rng.random::<f64>() < self.connection_probability {
                let from_idx = rng.gen_range(0..num_components);
                let to_idx = rng.gen_range(0..num_components);

                // For acyclic graphs, only allow forward connections (from lower index to higher)
                if !self.allow_cycles {
                    if from_idx < to_idx {
                        connections.push(Connection {
                            from: components[from_idx].id.clone(),
                            to: components[to_idx].id.clone(),
                            weight: rng.gen_range(0.1..=1.0),
                            connection_type: ConnectionType::Gradient,
                            enabled: true,
                        });
                    }
                } else if from_idx != to_idx {
                    connections.push(Connection {
                        from: components[from_idx].id.clone(),
                        to: components[to_idx].id.clone(),
                        weight: rng.gen_range(0.1..=1.0),
                        connection_type: ConnectionType::Gradient,
                        enabled: true,
                    });
                }
            }
        }

        Architecture {
            id: format!("arch_{}", rng.random::<u64>()),
            components,
            connections,
            hyperparameters: HashMap::new(),
            performance: None,
            generation: 0,
        }
    }

    /// Mutate an architecture
    pub fn mutate_architecture(
        &self,
        architecture: &Architecture,
        mutation_rate: f64,
    ) -> Architecture {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::Random::default();
        let mut mutated = architecture.clone();
        mutated.id = format!("{}_{}_mut", architecture.id, rng.random::<u32>());
        mutated.generation = architecture.generation + 1;
        mutated.performance = None; // Reset performance after mutation

        // Mutate component hyperparameters
        for component in &mut mutated.components {
            if rng.random::<f64>() < mutation_rate {
                if let Some(ranges) = self.hyperparameter_ranges.get(&component.component_type) {
                    for (param_name, value) in &mut component.hyperparameters {
                        if let Some((min_val, max_val)) = ranges.get(param_name) {
                            // Gaussian mutation around current value
                            let std_dev = (max_val - min_val) * 0.1;
                            let mutation = rng.random::<f64>() * std_dev - std_dev / 2.0;
                            *value = (*value + mutation).clamp(*min_val, *max_val);
                        }
                    }
                }
            }
        }

        // Randomly add or remove components
        if rng.random::<f64>() < mutation_rate * 0.1 {
            if mutated.components.len() < self.max_components && rng.random::<bool>() {
                // Add component
                let component_type =
                    self.component_types[rng.gen_range(0..self.component_types.len())];
                let mut hyperparameters = HashMap::new();

                if let Some(ranges) = self.hyperparameter_ranges.get(&component_type) {
                    for (param_name, (min_val, max_val)) in ranges {
                        let value = rng.gen_range(*min_val..=*max_val);
                        hyperparameters.insert(param_name.clone(), value);
                    }
                }

                mutated.components.push(OptimizerComponent {
                    id: format!("comp_{}", mutated.components.len()),
                    component_type,
                    hyperparameters,
                    position: ComponentPosition {
                        layer: mutated.components.len() as u32,
                        index: 0,
                        x: mutated.components.len() as f32 * 100.0,
                        y: 0.0,
                    },
                    enabled: true,
                });
            } else if mutated.components.len() > self.min_components {
                // Remove component only if we'll still have at least min_components
                let remove_idx = rng.gen_range(0..mutated.components.len());
                let removed_id = mutated.components[remove_idx].id.clone();
                mutated.components.remove(remove_idx);

                // Remove connections involving the removed component
                mutated
                    .connections
                    .retain(|conn| conn.from != removed_id && conn.to != removed_id);
            }
        }

        mutated
    }

    /// Crossover two architectures to create offspring
    pub fn crossover_architectures(
        &self,
        parent1: &Architecture,
        parent2: &Architecture,
    ) -> (Architecture, Architecture) {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::Random::default();

        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        child1.id = format!("{}_{}_cross", parent1.id, rng.random::<u32>());
        child2.id = format!("{}_{}_cross", parent2.id, rng.random::<u32>());
        child1.generation = parent1.generation.max(parent2.generation) + 1;
        child2.generation = parent1.generation.max(parent2.generation) + 1;
        child1.performance = None;
        child2.performance = None;

        // Single-point crossover for components
        if !parent1.components.is_empty() && !parent2.components.is_empty() {
            let crossover_point1 = rng.gen_range(0..parent1.components.len());
            let crossover_point2 = rng.gen_range(0..parent2.components.len());

            let mut new_components1 = parent1.components[..crossover_point1].to_vec();
            new_components1.extend_from_slice(&parent2.components[crossover_point2..]);

            let mut new_components2 = parent2.components[..crossover_point2].to_vec();
            new_components2.extend_from_slice(&parent1.components[crossover_point1..]);

            // Ensure components respect min/max constraints
            if new_components1.len() > self.max_components {
                new_components1.truncate(self.max_components);
            } else if new_components1.len() < self.min_components {
                // Pad with random components from parents if too few
                while new_components1.len() < self.min_components {
                    let source = if rng.random_bool() {
                        &parent1.components
                    } else {
                        &parent2.components
                    };
                    if !source.is_empty() {
                        let idx = rng.gen_range(0..source.len());
                        new_components1.push(source[idx].clone());
                    } else {
                        break;
                    }
                }
            }

            if new_components2.len() > self.max_components {
                new_components2.truncate(self.max_components);
            } else if new_components2.len() < self.min_components {
                // Pad with random components from parents if too few
                while new_components2.len() < self.min_components {
                    let source = if rng.random_bool() {
                        &parent1.components
                    } else {
                        &parent2.components
                    };
                    if !source.is_empty() {
                        let idx = rng.gen_range(0..source.len());
                        new_components2.push(source[idx].clone());
                    } else {
                        break;
                    }
                }
            }

            child1.components = new_components1;
            child2.components = new_components2;

            // Clean up connections - remove connections to non-existent components
            let child1_ids: std::collections::HashSet<_> =
                child1.components.iter().map(|c| c.id.clone()).collect();
            let child2_ids: std::collections::HashSet<_> =
                child2.components.iter().map(|c| c.id.clone()).collect();

            child1
                .connections
                .retain(|conn| child1_ids.contains(&conn.from) && child1_ids.contains(&conn.to));

            child2
                .connections
                .retain(|conn| child2_ids.contains(&conn.from) && child2_ids.contains(&conn.to));
        }

        (child1, child2)
    }

    /// Validate that an architecture satisfies search space constraints
    pub fn validate_architecture(&self, architecture: &Architecture) -> Result<(), String> {
        if architecture.components.len() < self.min_components {
            return Err(format!(
                "Too few components: {} < {}",
                architecture.components.len(),
                self.min_components
            ));
        }

        if architecture.components.len() > self.max_components {
            return Err(format!(
                "Too many components: {} > {}",
                architecture.components.len(),
                self.max_components
            ));
        }

        if architecture.connections.len() > self.max_connections {
            return Err(format!(
                "Too many connections: {} > {}",
                architecture.connections.len(),
                self.max_connections
            ));
        }

        // Check for cycles if not allowed
        if !self.allow_cycles && self.has_cycles(architecture) {
            return Err("Architecture contains cycles, but cycles are not allowed".to_string());
        }

        // Validate hyperparameters are within ranges
        for component in &architecture.components {
            if let Some(ranges) = self.hyperparameter_ranges.get(&component.component_type) {
                for (param_name, value) in &component.hyperparameters {
                    if let Some((min_val, max_val)) = ranges.get(param_name) {
                        if *value < *min_val || *value > *max_val {
                            return Err(format!(
                                "Hyperparameter {} = {} is out of range [{}, {}]",
                                param_name, value, min_val, max_val
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if the architecture contains cycles
    fn has_cycles(&self, architecture: &Architecture) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for component in &architecture.components {
            if !visited.contains(&component.id)
                && Self::has_cycle_util(architecture, &component.id, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }

        false
    }

    fn has_cycle_util(
        architecture: &Architecture,
        node: &str,
        visited: &mut std::collections::HashSet<String>,
        rec_stack: &mut std::collections::HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        // Find all outgoing connections from this node
        for connection in &architecture.connections {
            if connection.from == node && connection.enabled {
                if !visited.contains(&connection.to) {
                    if Self::has_cycle_util(architecture, &connection.to, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&connection.to) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }
}

impl Default for ArchitectureSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Architecture: {}", self.id)?;
        writeln!(f, "  Components: {}", self.components.len())?;
        writeln!(f, "  Connections: {}", self.connections.len())?;
        writeln!(f, "  Generation: {}", self.generation)?;

        if let Some(ref perf) = self.performance {
            writeln!(f, "  Performance: {:.4}", perf.overall_score)?;
        }

        for component in &self.components {
            writeln!(
                f,
                "    - {:?}: {} params",
                component.component_type,
                component.hyperparameters.len()
            )?;
        }

        Ok(())
    }
}

impl fmt::Display for ComponentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentType::SGD => write!(f, "SGD"),
            ComponentType::Adam => write!(f, "Adam"),
            ComponentType::AdamW => write!(f, "AdamW"),
            ComponentType::RMSprop => write!(f, "RMSprop"),
            ComponentType::AdaGrad => write!(f, "AdaGrad"),
            ComponentType::AdaDelta => write!(f, "AdaDelta"),
            ComponentType::Momentum => write!(f, "Momentum"),
            ComponentType::Nesterov => write!(f, "Nesterov"),
            ComponentType::LRScheduler => write!(f, "LRScheduler"),
            ComponentType::GradientClipping => write!(f, "GradientClipping"),
            ComponentType::BatchNorm => write!(f, "BatchNorm"),
            ComponentType::Dropout => write!(f, "Dropout"),
            ComponentType::LAMB => write!(f, "LAMB"),
            ComponentType::LARS => write!(f, "LARS"),
            ComponentType::Lion => write!(f, "Lion"),
            ComponentType::RAdam => write!(f, "RAdam"),
            ComponentType::Lookahead => write!(f, "Lookahead"),
            ComponentType::SAM => write!(f, "SAM"),
            ComponentType::LBFGS => write!(f, "LBFGS"),
            ComponentType::SparseAdam => write!(f, "SparseAdam"),
            ComponentType::GroupedAdam => write!(f, "GroupedAdam"),
            ComponentType::MAML => write!(f, "MAML"),
            ComponentType::L1Regularizer => write!(f, "L1Regularizer"),
            ComponentType::L2Regularizer => write!(f, "L2Regularizer"),
            ComponentType::ElasticNetRegularizer => write!(f, "ElasticNetRegularizer"),
            ComponentType::DropoutRegularizer => write!(f, "DropoutRegularizer"),
            ComponentType::WeightDecay => write!(f, "WeightDecay"),
            ComponentType::AdaptiveLR => write!(f, "AdaptiveLR"),
            ComponentType::AdaptiveMomentum => write!(f, "AdaptiveMomentum"),
            ComponentType::AdaptiveRegularization => write!(f, "AdaptiveRegularization"),
            ComponentType::LSTMOptimizer => write!(f, "LSTMOptimizer"),
            ComponentType::TransformerOptimizer => write!(f, "TransformerOptimizer"),
            ComponentType::AttentionOptimizer => write!(f, "AttentionOptimizer"),
            ComponentType::MetaSGD => write!(f, "MetaSGD"),
            ComponentType::ConstantLR => write!(f, "ConstantLR"),
            ComponentType::ExponentialLR => write!(f, "ExponentialLR"),
            ComponentType::StepLR => write!(f, "StepLR"),
            ComponentType::CosineAnnealingLR => write!(f, "CosineAnnealingLR"),
            ComponentType::OneCycleLR => write!(f, "OneCycleLR"),
            ComponentType::CyclicLR => write!(f, "CyclicLR"),
            ComponentType::Reptile => write!(f, "Reptile"),
            ComponentType::Custom => write!(f, "Custom"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_space_creation() {
        let space = ArchitectureSpace::new();
        assert!(space.min_components <= space.max_components);
        assert!(!space.component_types.is_empty());
        assert!(space.connection_probability >= 0.0 && space.connection_probability <= 1.0);
    }

    #[test]
    fn test_random_architecture_generation() {
        let space = ArchitectureSpace::new();
        let arch = space.generate_random_architecture();

        assert!(arch.components.len() >= space.min_components);
        assert!(arch.components.len() <= space.max_components);
        assert!(space.validate_architecture(&arch).is_ok());
    }

    #[test]
    fn test_architecture_mutation() {
        let space = ArchitectureSpace::new();
        let original = space.generate_random_architecture();
        let mutated = space.mutate_architecture(&original, 0.5);

        assert_ne!(original.id, mutated.id);
        assert_eq!(mutated.generation, original.generation + 1);

        // Debug info for failures
        if let Err(e) = space.validate_architecture(&mutated) {
            eprintln!("Validation failed: {}", e);
            eprintln!("Original components: {}", original.components.len());
            eprintln!("Mutated components: {}", mutated.components.len());
            eprintln!("Min components: {}", space.min_components);
            eprintln!("Max components: {}", space.max_components);
        }

        assert!(space.validate_architecture(&mutated).is_ok());
    }

    #[test]
    fn test_architecture_crossover() {
        let space = ArchitectureSpace::new();
        let parent1 = space.generate_random_architecture();
        let parent2 = space.generate_random_architecture();

        let (child1, child2) = space.crossover_architectures(&parent1, &parent2);

        assert_ne!(child1.id, parent1.id);
        assert_ne!(child2.id, parent2.id);
        assert!(space.validate_architecture(&child1).is_ok());
        assert!(space.validate_architecture(&child2).is_ok());
    }

    #[test]
    fn test_architecture_validation() {
        let space = ArchitectureSpace::new();
        let valid_arch = space.generate_random_architecture();
        assert!(space.validate_architecture(&valid_arch).is_ok());

        // Test invalid architecture with too many components
        let mut invalid_arch = valid_arch.clone();
        for i in 0..space.max_components + 5 {
            invalid_arch.components.push(OptimizerComponent {
                id: format!("invalid_{}", i),
                component_type: ComponentType::Adam,
                hyperparameters: HashMap::new(),
                position: ComponentPosition {
                    layer: i as u32,
                    index: 0,
                    x: 0.0,
                    y: 0.0,
                },
                enabled: true,
            });
        }

        assert!(space.validate_architecture(&invalid_arch).is_err());
    }
}
