// Architecture generation utilities

use scirs2_core::random::Rng;
use crate::error::Result;
use super::architecture_space::ArchitectureSearchSpace;

/// Architecture generator
pub struct ArchitectureGenerator {
    generation_strategy: GenerationStrategy,
    search_space: Option<ArchitectureSearchSpace>,
}

impl ArchitectureGenerator {
    pub fn new(search_space: &ArchitectureSearchSpace) -> Result<Self> {
        Ok(Self {
            generation_strategy: GenerationStrategy::Random,
            search_space: Some(search_space.clone()),
        })
    }

    pub fn generate_random_population(&self, size: usize) -> Result<Vec<String>> {
        let mut architectures = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..size {
            if let Some(ref space) = self.search_space {
                architectures.push(space.sample_random_architecture(&mut rng)?);
            }
        }

        Ok(architectures)
    }
}

/// Generation strategy
#[derive(Debug, Clone, Copy)]
pub enum GenerationStrategy {
    Random,
    Guided,
    Template,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned_optimizers::neural_architecture_search::config::SearchConstraints;

    #[test]
    fn test_architecture_generator() {
        let constraints = SearchConstraints::default();
        let space = ArchitectureSearchSpace::new(&constraints).unwrap();
        let generator = ArchitectureGenerator::new(&space);
        assert!(generator.is_ok());
    }
}