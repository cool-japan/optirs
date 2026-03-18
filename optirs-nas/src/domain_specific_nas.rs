//! Domain-specific Neural Architecture Search for optimizer discovery.
//!
//! Provides pre-configured search spaces and constraint validation
//! tailored to specific machine learning domains such as computer vision,
//! natural language processing, time series, reinforcement learning,
//! and scientific computing.

use crate::architecture::{
    Architecture, ComponentPosition, ComponentType, Connection, ConnectionType, OptimizerComponent,
    PerformanceMetrics,
};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use scirs2_core::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Constraint applied to NAS architectures within a domain
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NASConstraint {
    /// Maximum inference latency in milliseconds
    MaxLatencyMs(f64),
    /// Maximum memory usage in megabytes
    MaxMemoryMb(f64),
    /// Minimum accuracy threshold
    MinAccuracy(f64),
    /// Architecture must include a specific component type
    RequiresComponent(ComponentType),
    /// Maximum depth (number of layers) of the architecture
    MaxDepth(usize),
    /// Maximum width (components per layer) of the architecture
    MaxWidth(usize),
}

/// Domain types for specialized NAS configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainType {
    /// Computer vision tasks (image classification, detection, segmentation)
    ComputerVision,
    /// Natural language processing tasks (translation, summarization, QA)
    NaturalLanguageProcessing,
    /// Time series forecasting and analysis
    TimeSeries,
    /// Reinforcement learning tasks
    Reinforcement,
    /// Scientific computing and simulation optimization
    Scientific,
}

/// Domain-specific search space defining allowed components and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSearchSpace {
    /// The domain this search space is designed for
    pub domain: DomainType,
    /// Components allowed in architectures for this domain
    pub allowed_components: Vec<ComponentType>,
    /// Constraints that architectures must satisfy
    pub constraints: Vec<NASConstraint>,
    /// Minimum architecture depth
    pub min_depth: usize,
    /// Maximum architecture depth
    pub max_depth: usize,
    /// Recommended hyperparameter values for this domain
    pub recommended_hyperparameters: HashMap<String, f64>,
}

/// Engine for domain-specific neural architecture search
#[derive(Debug, Clone)]
pub struct DomainNASEngine<T: Float + Debug + Send + Sync + 'static + ScalarOperand> {
    /// Target domain
    pub domain: DomainType,
    /// Search space configuration for the domain
    pub search_space: DomainSearchSpace,
    /// Evaluated architectures with their scores
    pub evaluated_architectures: Vec<(Architecture, T)>,
    /// Best architecture found so far
    pub best_architecture: Option<Architecture>,
}

impl<T: Float + Debug + Send + Sync + 'static + ScalarOperand> DomainNASEngine<T> {
    /// Create a new domain-specific NAS engine with pre-configured search space
    pub fn new_for_domain(domain: DomainType) -> Self {
        let search_space = Self::get_default_search_space(domain);
        Self {
            domain,
            search_space,
            evaluated_architectures: Vec::new(),
            best_architecture: None,
        }
    }

    /// Get the default search space configuration for a given domain
    pub fn get_default_search_space(domain: DomainType) -> DomainSearchSpace {
        match domain {
            DomainType::ComputerVision => {
                let mut recommended = HashMap::new();
                recommended.insert("learning_rate".to_string(), 0.001);
                recommended.insert("weight_decay".to_string(), 0.0001);
                recommended.insert("momentum".to_string(), 0.9);
                recommended.insert("beta1".to_string(), 0.9);
                recommended.insert("beta2".to_string(), 0.999);
                DomainSearchSpace {
                    domain,
                    allowed_components: vec![
                        ComponentType::Adam,
                        ComponentType::AdamW,
                        ComponentType::SGD,
                        ComponentType::Momentum,
                        ComponentType::LARS,
                        ComponentType::BatchNorm,
                        ComponentType::GradientClipping,
                        ComponentType::CosineAnnealingLR,
                        ComponentType::L2Regularizer,
                    ],
                    constraints: vec![
                        NASConstraint::MaxDepth(50),
                        NASConstraint::MaxLatencyMs(100.0),
                        NASConstraint::MaxMemoryMb(4096.0),
                    ],
                    min_depth: 3,
                    max_depth: 50,
                    recommended_hyperparameters: recommended,
                }
            }
            DomainType::NaturalLanguageProcessing => {
                let mut recommended = HashMap::new();
                recommended.insert("learning_rate".to_string(), 0.0001);
                recommended.insert("warmup_steps".to_string(), 4000.0);
                recommended.insert("beta1".to_string(), 0.9);
                recommended.insert("beta2".to_string(), 0.98);
                recommended.insert("epsilon".to_string(), 1e-9);
                DomainSearchSpace {
                    domain,
                    allowed_components: vec![
                        ComponentType::Adam,
                        ComponentType::AdamW,
                        ComponentType::LAMB,
                        ComponentType::Lookahead,
                        ComponentType::GradientClipping,
                        ComponentType::CosineAnnealingLR,
                        ComponentType::OneCycleLR,
                        ComponentType::WeightDecay,
                    ],
                    constraints: vec![
                        NASConstraint::MaxDepth(24),
                        NASConstraint::RequiresComponent(ComponentType::GradientClipping),
                        NASConstraint::MaxMemoryMb(16384.0),
                    ],
                    min_depth: 6,
                    max_depth: 24,
                    recommended_hyperparameters: recommended,
                }
            }
            DomainType::TimeSeries => {
                let mut recommended = HashMap::new();
                recommended.insert("learning_rate".to_string(), 0.001);
                recommended.insert("alpha".to_string(), 0.99);
                recommended.insert("epsilon".to_string(), 1e-8);
                recommended.insert("clip_value".to_string(), 1.0);
                DomainSearchSpace {
                    domain,
                    allowed_components: vec![
                        ComponentType::Adam,
                        ComponentType::RMSprop,
                        ComponentType::AdaDelta,
                        ComponentType::GradientClipping,
                        ComponentType::StepLR,
                        ComponentType::ExponentialLR,
                        ComponentType::L1Regularizer,
                        ComponentType::L2Regularizer,
                    ],
                    constraints: vec![
                        NASConstraint::MaxDepth(10),
                        NASConstraint::MaxLatencyMs(50.0),
                        NASConstraint::MaxMemoryMb(2048.0),
                    ],
                    min_depth: 2,
                    max_depth: 10,
                    recommended_hyperparameters: recommended,
                }
            }
            DomainType::Reinforcement => {
                let mut recommended = HashMap::new();
                recommended.insert("learning_rate".to_string(), 0.0003);
                recommended.insert("gamma".to_string(), 0.99);
                recommended.insert("epsilon".to_string(), 1e-5);
                recommended.insert("beta1".to_string(), 0.9);
                recommended.insert("beta2".to_string(), 0.999);
                DomainSearchSpace {
                    domain,
                    allowed_components: vec![
                        ComponentType::Adam,
                        ComponentType::RMSprop,
                        ComponentType::SGD,
                        ComponentType::GradientClipping,
                        ComponentType::CyclicLR,
                        ComponentType::L2Regularizer,
                        ComponentType::DropoutRegularizer,
                    ],
                    constraints: vec![
                        NASConstraint::MaxDepth(8),
                        NASConstraint::MaxLatencyMs(10.0),
                        NASConstraint::MaxMemoryMb(1024.0),
                    ],
                    min_depth: 2,
                    max_depth: 8,
                    recommended_hyperparameters: recommended,
                }
            }
            DomainType::Scientific => {
                let mut recommended = HashMap::new();
                recommended.insert("learning_rate".to_string(), 0.01);
                recommended.insert("momentum".to_string(), 0.9);
                recommended.insert("tolerance".to_string(), 1e-12);
                recommended.insert("max_iterations".to_string(), 10000.0);
                DomainSearchSpace {
                    domain,
                    allowed_components: vec![
                        ComponentType::Adam,
                        ComponentType::LBFGS,
                        ComponentType::SGD,
                        ComponentType::AdaGrad,
                        ComponentType::Nesterov,
                        ComponentType::GradientClipping,
                        ComponentType::CosineAnnealingLR,
                        ComponentType::L1Regularizer,
                        ComponentType::L2Regularizer,
                        ComponentType::ElasticNetRegularizer,
                    ],
                    constraints: vec![
                        NASConstraint::MaxDepth(20),
                        NASConstraint::MinAccuracy(0.95),
                        NASConstraint::MaxMemoryMb(8192.0),
                    ],
                    min_depth: 3,
                    max_depth: 20,
                    recommended_hyperparameters: recommended,
                }
            }
        }
    }

    /// Generate and evaluate architectures within the given budget
    ///
    /// Returns a list of valid architectures sorted by score (descending).
    pub fn search(&mut self, budget: usize) -> Result<Vec<Architecture>> {
        use scirs2_core::random::Rng;

        if budget == 0 {
            return Err(OptimError::InvalidConfig(
                "Search budget must be greater than 0".to_string(),
            ));
        }

        let mut rng = scirs2_core::random::Random::default();
        let mut results = Vec::new();

        for i in 0..budget {
            let arch = self.generate_domain_architecture(&mut rng, i);

            // Validate against domain constraints
            let violations = self.validate_for_domain(&arch)?;
            if !violations.is_empty() {
                continue;
            }

            // Score the architecture based on domain heuristics
            let score = self.score_architecture(&arch);

            // Track if this is the best so far
            let dominated = match &self.best_architecture {
                Some(best) => {
                    let best_score = self
                        .evaluated_architectures
                        .iter()
                        .filter(|(a, _)| a.id == best.id)
                        .map(|(_, s)| *s)
                        .next();
                    match best_score {
                        Some(bs) => score <= bs,
                        None => false,
                    }
                }
                None => false,
            };

            if !dominated {
                self.best_architecture = Some(arch.clone());
            }

            self.evaluated_architectures.push((arch.clone(), score));
            results.push(arch);
        }

        // Sort results by score (descending)
        results.sort_by(|a, b| {
            let score_a = self
                .evaluated_architectures
                .iter()
                .filter(|(arch, _)| arch.id == a.id)
                .map(|(_, s)| *s)
                .next()
                .unwrap_or_else(T::zero);
            let score_b = self
                .evaluated_architectures
                .iter()
                .filter(|(arch, _)| arch.id == b.id)
                .map(|(_, s)| *s)
                .next()
                .unwrap_or_else(T::zero);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Validate an architecture against domain-specific constraints
    ///
    /// Returns a list of constraint violations. An empty list means the architecture is valid.
    pub fn validate_for_domain(&self, arch: &Architecture) -> Result<Vec<String>> {
        let mut violations = Vec::new();

        for constraint in &self.search_space.constraints {
            match constraint {
                NASConstraint::MaxLatencyMs(max_ms) => {
                    // Estimate latency based on component count and types
                    let estimated_latency = self.estimate_latency(arch);
                    if estimated_latency > *max_ms {
                        violations.push(format!(
                            "Estimated latency {:.2}ms exceeds maximum {:.2}ms",
                            estimated_latency, max_ms
                        ));
                    }
                }
                NASConstraint::MaxMemoryMb(max_mb) => {
                    let estimated_memory = self.estimate_memory(arch);
                    if estimated_memory > *max_mb {
                        violations.push(format!(
                            "Estimated memory {:.2}MB exceeds maximum {:.2}MB",
                            estimated_memory, max_mb
                        ));
                    }
                }
                NASConstraint::MinAccuracy(min_acc) => {
                    // Check performance if available
                    if let Some(ref perf) = arch.performance {
                        if perf.final_performance < *min_acc {
                            violations.push(format!(
                                "Performance {:.4} below minimum accuracy {:.4}",
                                perf.final_performance, min_acc
                            ));
                        }
                    }
                }
                NASConstraint::RequiresComponent(required_type) => {
                    let has_component = arch
                        .components
                        .iter()
                        .any(|c| c.component_type == *required_type && c.enabled);
                    if !has_component {
                        violations.push(format!("Missing required component: {:?}", required_type));
                    }
                }
                NASConstraint::MaxDepth(max_depth) => {
                    let depth = self.compute_depth(arch);
                    if depth > *max_depth {
                        violations.push(format!(
                            "Architecture depth {} exceeds maximum {}",
                            depth, max_depth
                        ));
                    }
                }
                NASConstraint::MaxWidth(max_width) => {
                    let width = self.compute_max_width(arch);
                    if width > *max_width {
                        violations.push(format!(
                            "Architecture width {} exceeds maximum {}",
                            width, max_width
                        ));
                    }
                }
            }
        }

        // Check that all components are from the allowed set
        for component in &arch.components {
            if !self
                .search_space
                .allowed_components
                .contains(&component.component_type)
            {
                violations.push(format!(
                    "Component type {:?} is not allowed in domain {:?}",
                    component.component_type, self.domain
                ));
            }
        }

        Ok(violations)
    }

    /// Get the best architecture found so far
    pub fn get_best_architecture(&self) -> Option<&Architecture> {
        self.best_architecture.as_ref()
    }

    /// Generate a random architecture conforming to the domain search space
    fn generate_domain_architecture(
        &self,
        rng: &mut scirs2_core::random::Random,
        index: usize,
    ) -> Architecture {
        use scirs2_core::random::Rng;

        let allowed = &self.search_space.allowed_components;
        let num_components = rng.gen_range(
            self.search_space.min_depth.max(1)..=self.search_space.max_depth.min(allowed.len() * 3),
        );

        let mut components = Vec::new();
        for i in 0..num_components {
            let comp_type = allowed[rng.gen_range(0..allowed.len())];
            let mut hyperparameters = HashMap::new();

            // Apply recommended hyperparameters with small random perturbation
            for (key, value) in &self.search_space.recommended_hyperparameters {
                let perturbation = (rng.random::<f64>() - 0.5) * 0.2 * (*value).abs();
                hyperparameters.insert(key.clone(), value + perturbation);
            }

            components.push(OptimizerComponent {
                id: format!("domain_comp_{}", i),
                component_type: comp_type,
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

        // Create sequential connections between adjacent components
        let mut connections = Vec::new();
        for i in 0..components.len().saturating_sub(1) {
            connections.push(Connection {
                from: components[i].id.clone(),
                to: components[i + 1].id.clone(),
                weight: 1.0,
                connection_type: ConnectionType::Gradient,
                enabled: true,
            });
        }

        // Add some skip connections with low probability
        for i in 0..components.len() {
            for j in (i + 2)..components.len() {
                if rng.random::<f64>() < 0.1 {
                    connections.push(Connection {
                        from: components[i].id.clone(),
                        to: components[j].id.clone(),
                        weight: rng.gen_range(0.1..=0.5),
                        connection_type: ConnectionType::Skip,
                        enabled: true,
                    });
                }
            }
        }

        Architecture {
            id: format!("domain_{:?}_arch_{}", self.domain, index),
            components,
            connections,
            hyperparameters: self.search_space.recommended_hyperparameters.clone(),
            performance: None,
            generation: 0,
        }
    }

    /// Score an architecture based on domain-specific heuristics
    fn score_architecture(&self, arch: &Architecture) -> T {
        let mut score = T::zero();
        let one = T::one();
        let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or(one);

        // Base score from component count (favor architectures near the middle of allowed range)
        let mid_depth = (self.search_space.min_depth + self.search_space.max_depth) / 2;
        let depth_diff = if arch.components.len() > mid_depth {
            arch.components.len() - mid_depth
        } else {
            mid_depth - arch.components.len()
        };
        let depth_penalty: T = scirs2_core::numeric::NumCast::from(depth_diff as f64 * 0.01)
            .unwrap_or_else(|| T::zero());
        score = score + one - depth_penalty;

        // Bonus for diversity of component types
        let unique_types: std::collections::HashSet<_> =
            arch.components.iter().map(|c| c.component_type).collect();
        let diversity_bonus: T =
            scirs2_core::numeric::NumCast::from(unique_types.len() as f64 * 0.05)
                .unwrap_or_else(|| T::zero());
        score = score + diversity_bonus;

        // Bonus for having connections (architecture with flow)
        let connection_bonus: T =
            scirs2_core::numeric::NumCast::from((arch.connections.len() as f64).min(10.0) * 0.02)
                .unwrap_or_else(|| T::zero());
        score = score + connection_bonus;

        // Domain-specific bonuses
        let domain_bonus = self.compute_domain_bonus(arch);
        score = score + domain_bonus;

        // Ensure score is positive
        if score < half {
            score = half;
        }

        score
    }

    /// Compute domain-specific bonus for an architecture
    fn compute_domain_bonus(&self, arch: &Architecture) -> T {
        let small_bonus: T = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());
        let zero = T::zero();
        let mut bonus = zero;

        let has_component =
            |ct: ComponentType| -> bool { arch.components.iter().any(|c| c.component_type == ct) };

        match self.domain {
            DomainType::ComputerVision => {
                if has_component(ComponentType::BatchNorm) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::Momentum) || has_component(ComponentType::SGD) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::CosineAnnealingLR) {
                    bonus = bonus + small_bonus;
                }
            }
            DomainType::NaturalLanguageProcessing => {
                if has_component(ComponentType::AdamW) || has_component(ComponentType::LAMB) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::GradientClipping) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::Lookahead) {
                    bonus = bonus + small_bonus;
                }
            }
            DomainType::TimeSeries => {
                if has_component(ComponentType::RMSprop) || has_component(ComponentType::AdaDelta) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::GradientClipping) {
                    bonus = bonus + small_bonus;
                }
            }
            DomainType::Reinforcement => {
                if has_component(ComponentType::Adam) || has_component(ComponentType::RMSprop) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::GradientClipping) {
                    bonus = bonus + small_bonus;
                }
            }
            DomainType::Scientific => {
                if has_component(ComponentType::LBFGS) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::Nesterov) || has_component(ComponentType::AdaGrad) {
                    bonus = bonus + small_bonus;
                }
                if has_component(ComponentType::ElasticNetRegularizer) {
                    bonus = bonus + small_bonus;
                }
            }
        }

        bonus
    }

    /// Estimate latency in milliseconds for an architecture
    fn estimate_latency(&self, arch: &Architecture) -> f64 {
        // Simple heuristic: each component adds some base latency
        let mut latency = 0.0;
        for component in &arch.components {
            if !component.enabled {
                continue;
            }
            latency += match component.component_type {
                ComponentType::LBFGS => 5.0,
                ComponentType::LAMB | ComponentType::LARS => 3.0,
                ComponentType::Adam | ComponentType::AdamW | ComponentType::RAdam => 2.0,
                ComponentType::SGD | ComponentType::Momentum | ComponentType::Nesterov => 1.0,
                ComponentType::BatchNorm => 1.5,
                ComponentType::GradientClipping => 0.5,
                _ => 1.0,
            };
        }
        // Connections add overhead
        latency += arch.connections.len() as f64 * 0.1;
        latency
    }

    /// Estimate memory usage in megabytes for an architecture
    fn estimate_memory(&self, arch: &Architecture) -> f64 {
        let mut memory = 0.0;
        for component in &arch.components {
            if !component.enabled {
                continue;
            }
            memory += match component.component_type {
                ComponentType::LBFGS => 200.0,
                ComponentType::Adam | ComponentType::AdamW | ComponentType::RAdam => 100.0,
                ComponentType::LAMB | ComponentType::LARS => 120.0,
                ComponentType::SGD | ComponentType::Momentum => 50.0,
                ComponentType::BatchNorm => 30.0,
                ComponentType::GradientClipping => 10.0,
                _ => 50.0,
            };
        }
        memory
    }

    /// Compute the depth of an architecture (max layer index + 1)
    fn compute_depth(&self, arch: &Architecture) -> usize {
        arch.components
            .iter()
            .filter(|c| c.enabled)
            .map(|c| c.position.layer as usize + 1)
            .max()
            .unwrap_or(0)
    }

    /// Compute the maximum width (components sharing the same layer)
    fn compute_max_width(&self, arch: &Architecture) -> usize {
        let mut layer_counts: HashMap<u32, usize> = HashMap::new();
        for component in &arch.components {
            if component.enabled {
                *layer_counts.entry(component.position.layer).or_insert(0) += 1;
            }
        }
        layer_counts.values().copied().max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_architecture(id: &str, component_types: &[ComponentType]) -> Architecture {
        let components: Vec<OptimizerComponent> = component_types
            .iter()
            .enumerate()
            .map(|(i, ct)| OptimizerComponent {
                id: format!("comp_{}", i),
                component_type: *ct,
                hyperparameters: HashMap::new(),
                position: ComponentPosition {
                    layer: i as u32,
                    index: 0,
                    x: i as f32 * 100.0,
                    y: 0.0,
                },
                enabled: true,
            })
            .collect();

        let mut connections = Vec::new();
        for i in 0..components.len().saturating_sub(1) {
            connections.push(Connection {
                from: components[i].id.clone(),
                to: components[i + 1].id.clone(),
                weight: 1.0,
                connection_type: ConnectionType::Gradient,
                enabled: true,
            });
        }

        Architecture {
            id: id.to_string(),
            components,
            connections,
            hyperparameters: HashMap::new(),
            performance: None,
            generation: 0,
        }
    }

    #[test]
    fn test_domain_search_space_cv() {
        let engine = DomainNASEngine::<f64>::new_for_domain(DomainType::ComputerVision);
        let space = &engine.search_space;

        assert_eq!(space.domain, DomainType::ComputerVision);
        assert_eq!(space.min_depth, 3);
        assert_eq!(space.max_depth, 50);
        assert!(space.allowed_components.contains(&ComponentType::Adam));
        assert!(space.allowed_components.contains(&ComponentType::AdamW));
        assert!(space.allowed_components.contains(&ComponentType::SGD));
        assert!(space.allowed_components.contains(&ComponentType::Momentum));
        assert!(space.allowed_components.contains(&ComponentType::LARS));
        assert!(space.allowed_components.contains(&ComponentType::BatchNorm));
        assert!(space
            .recommended_hyperparameters
            .contains_key("learning_rate"));
    }

    #[test]
    fn test_domain_search_space_nlp() {
        let engine = DomainNASEngine::<f64>::new_for_domain(DomainType::NaturalLanguageProcessing);
        let space = &engine.search_space;

        assert_eq!(space.domain, DomainType::NaturalLanguageProcessing);
        assert_eq!(space.min_depth, 6);
        assert_eq!(space.max_depth, 24);
        assert!(space.allowed_components.contains(&ComponentType::Adam));
        assert!(space.allowed_components.contains(&ComponentType::AdamW));
        assert!(space.allowed_components.contains(&ComponentType::LAMB));
        assert!(space.allowed_components.contains(&ComponentType::Lookahead));
        assert!(space
            .recommended_hyperparameters
            .contains_key("warmup_steps"));

        // NLP should require gradient clipping
        let has_requires_gc = space.constraints.iter().any(|c| {
            matches!(
                c,
                NASConstraint::RequiresComponent(ComponentType::GradientClipping)
            )
        });
        assert!(has_requires_gc);
    }

    #[test]
    fn test_validate_architecture() {
        let engine = DomainNASEngine::<f64>::new_for_domain(DomainType::ComputerVision);

        // Valid CV architecture
        let valid_arch = make_test_architecture(
            "valid_cv",
            &[
                ComponentType::Adam,
                ComponentType::BatchNorm,
                ComponentType::SGD,
                ComponentType::Momentum,
            ],
        );
        let violations = engine.validate_for_domain(&valid_arch);
        assert!(violations.is_ok());
        let violations = violations.ok().unwrap_or_default();
        assert!(
            violations.is_empty(),
            "Expected no violations, got: {:?}",
            violations
        );

        // Invalid: uses component not in allowed set for CV
        let invalid_arch = make_test_architecture(
            "invalid_cv",
            &[ComponentType::LAMB, ComponentType::Lookahead],
        );
        let violations = engine
            .validate_for_domain(&invalid_arch)
            .ok()
            .unwrap_or_default();
        assert!(
            !violations.is_empty(),
            "Expected violations for disallowed components"
        );
    }

    #[test]
    fn test_search_budget() {
        let mut engine = DomainNASEngine::<f64>::new_for_domain(DomainType::TimeSeries);
        let results = engine.search(10);
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_default();

        // We should get some results (may be fewer than budget if some fail validation)
        assert!(
            !results.is_empty(),
            "Search should produce at least some architectures"
        );
        assert!(results.len() <= 10, "Should not exceed budget of 10");

        // Best architecture should be set
        assert!(
            engine.get_best_architecture().is_some(),
            "Best architecture should be found"
        );
    }

    #[test]
    fn test_constraint_checking() {
        let engine = DomainNASEngine::<f64>::new_for_domain(DomainType::NaturalLanguageProcessing);

        // Architecture missing required GradientClipping component
        let arch_no_gc =
            make_test_architecture("no_gc", &[ComponentType::Adam, ComponentType::AdamW]);
        let violations = engine
            .validate_for_domain(&arch_no_gc)
            .ok()
            .unwrap_or_default();
        let has_missing_gc = violations
            .iter()
            .any(|v| v.contains("Missing required component"));
        assert!(
            has_missing_gc,
            "Should report missing GradientClipping. Violations: {:?}",
            violations
        );

        // Architecture with GradientClipping should pass that constraint
        let arch_with_gc = make_test_architecture(
            "with_gc",
            &[
                ComponentType::Adam,
                ComponentType::AdamW,
                ComponentType::GradientClipping,
                ComponentType::Lookahead,
                ComponentType::LAMB,
                ComponentType::CosineAnnealingLR,
            ],
        );
        let violations = engine
            .validate_for_domain(&arch_with_gc)
            .ok()
            .unwrap_or_default();
        let has_missing_gc = violations
            .iter()
            .any(|v| v.contains("Missing required component"));
        assert!(
            !has_missing_gc,
            "Should NOT report missing GradientClipping when present"
        );
    }
}
