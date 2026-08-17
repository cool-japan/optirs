//! The real ArchitectureController implementation: random generation, mutation and crossover against a configured search space.

use crate::error::Result;
use crate::nas_engine::config::*;
use crate::nas_engine::resources::*;
use crate::nas_engine::results::*;
use scirs2_core::numeric::Float;
use scirs2_core::RngExt;
use std::collections::HashMap;
use std::fmt::Debug;

use super::support::ArchitectureController;

/// Real architecture controller.
///
/// Generates, mutates and crosses over architectures against the configured
/// search space, and validates that a candidate conforms to the space's
/// component-count limits. This replaces the former placeholder whose
/// `generate_random` returned an empty architecture with a constant id,
/// `mutate`/`crossover` were identity clones, and `validate` always returned
/// `true`.
pub(super) struct DefaultArchitectureController<T: Float + Debug + Send + Sync + 'static> {
    pub(super) search_space: SearchSpaceConfig,
    pub(super) rng: scirs2_core::random::Random<scirs2_core::random::rngs::StdRng>,
    pub(super) counter: usize,
    pub(super) _phantom: std::marker::PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> DefaultArchitectureController<T> {
    pub fn new(config: &NASConfig<T>) -> Result<Self> {
        Ok(Self {
            search_space: config.search_space.clone(),
            rng: scirs2_core::random::Random::seed(42),
            counter: 0,
            _phantom: std::marker::PhantomData,
        })
    }
    /// The component vocabulary to sample from: the declared `component_types`,
    /// falling back to the detailed `components` list, and finally to a single
    /// generic optimizer so generation never fails on an under-specified space.
    pub(super) fn component_labels(&self) -> Vec<String> {
        if !self.search_space.component_types.is_empty() {
            self.search_space
                .component_types
                .iter()
                .map(|c| format!("{:?}", c))
                .collect()
        } else if !self.search_space.components.is_empty() {
            self.search_space
                .components
                .iter()
                .map(|c| format!("{:?}", c.component_type))
                .collect()
        } else {
            vec!["Adam".to_string()]
        }
    }
    /// Inclusive `[min, max]` component-count bounds, clamped so `min <= max` and
    /// both are at least 1.
    pub(super) fn component_count_bounds(&self) -> (usize, usize) {
        let max = self.search_space.max_components.max(1);
        let min = self.search_space.min_components.max(1).min(max);
        (min, max)
    }
    /// Sample a log-uniform learning rate in `[1e-4, 1e-1]`.
    pub(super) fn sample_learning_rate(&mut self) -> T {
        let log_min = (1e-4f64).ln();
        let log_max = (1e-1f64).ln();
        let log_val: f64 = self.rng.gen_range(log_min..log_max);
        scirs2_core::numeric::NumCast::from(log_val.exp()).unwrap_or_else(|| T::zero())
    }
    pub(super) fn next_id(&mut self, kind: &str) -> String {
        self.counter += 1;
        format!(
            "ctrl_{}_{}_{:08x}",
            kind,
            self.counter,
            self.rng.random::<u32>()
        )
    }
}
impl<T: Float + Debug + Send + Sync + 'static> ArchitectureController<T>
    for DefaultArchitectureController<T>
{
    fn generate_random(&mut self) -> Result<OptimizerArchitecture<T>> {
        let labels = self.component_labels();
        let (min_c, max_c) = self.component_count_bounds();
        let num = self.rng.gen_range(min_c..=max_c);
        let mut components = Vec::with_capacity(num);
        for _ in 0..num {
            let idx = self.rng.gen_range(0..labels.len());
            components.push(labels[idx].clone());
        }
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), self.sample_learning_rate());
        let architecture_id = self.next_id("random");
        Ok(OptimizerArchitecture {
            components,
            parameters: HashMap::new(),
            connections: Vec::new(),
            hyperparameters,
            architecture_id,
            metadata: HashMap::new(),
        })
    }
    fn mutate(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>> {
        let labels = self.component_labels();
        let mut child = architecture.clone();
        if child.components.is_empty() {
            let idx = self.rng.gen_range(0..labels.len());
            child.components.push(labels[idx].clone());
        } else {
            let pos = self.rng.gen_range(0..child.components.len());
            let idx = self.rng.gen_range(0..labels.len());
            child.components[pos] = labels[idx].clone();
        }
        let new_lr = self.sample_learning_rate();
        child
            .hyperparameters
            .insert("learning_rate".to_string(), new_lr);
        child.architecture_id = self.next_id("mutant");
        Ok(child)
    }
    fn crossover(
        &mut self,
        parent1: &OptimizerArchitecture<T>,
        parent2: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>> {
        let (min_c, max_c) = self.component_count_bounds();
        let len1 = parent1.components.len();
        let len2 = parent2.components.len();
        let mut components = Vec::new();
        if len1 == 0 && len2 == 0 {
            let labels = self.component_labels();
            let idx = self.rng.gen_range(0..labels.len());
            components.push(labels[idx].clone());
        } else {
            let cut1 = self.rng.gen_range(0..=len1);
            components.extend(parent1.components[..cut1].iter().cloned());
            let cut2 = self.rng.gen_range(0..=len2);
            components.extend(parent2.components[cut2..].iter().cloned());
            if components.is_empty() {
                let source = if len1 >= len2 {
                    &parent1.components
                } else {
                    &parent2.components
                };
                components.push(source[0].clone());
            }
        }
        if components.len() > max_c {
            components.truncate(max_c);
        }
        while components.len() < min_c {
            let labels = self.component_labels();
            let idx = self.rng.gen_range(0..labels.len());
            components.push(labels[idx].clone());
        }
        let lr1 = parent1.hyperparameters.get("learning_rate").copied();
        let lr2 = parent2.hyperparameters.get("learning_rate").copied();
        let lr = match (lr1, lr2) {
            (Some(a), Some(b)) => {
                let half: T = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
                (a + b) * half
            }
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => self.sample_learning_rate(),
        };
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), lr);
        let architecture_id = self.next_id("cross");
        Ok(OptimizerArchitecture {
            components,
            parameters: HashMap::new(),
            connections: Vec::new(),
            hyperparameters,
            architecture_id,
            metadata: HashMap::new(),
        })
    }
    fn validate(&self, architecture: &OptimizerArchitecture<T>) -> Result<bool> {
        if architecture.components.is_empty() {
            return Ok(false);
        }
        let max_c = self.search_space.max_components;
        if max_c > 0 && architecture.components.len() > max_c {
            return Ok(false);
        }
        Ok(true)
    }
}
