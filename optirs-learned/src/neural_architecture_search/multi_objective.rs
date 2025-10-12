// Multi-objective optimization for neural architecture search

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::collections::HashMap;
use crate::error::Result;
use super::population::Individual;

/// Multi-objective optimizer
pub struct MultiObjectiveOptimizer<T: Float + Debug + Send + Sync + 'static> {
    objective_weights: Vec<f64>,
    pareto_front: Vec<Individual<T>>,
    optimization_objectives: Vec<OptimizationObjective>,
}

impl<T: Float + Debug + Send + Sync + 'static> MultiObjectiveOptimizer<T> {
    pub fn new(objective_weights: Vec<f64>) -> Result<Self> {
        let optimization_objectives = vec![
            OptimizationObjective::Accuracy,
            OptimizationObjective::Speed,
            OptimizationObjective::Memory,
        ];

        Ok(Self {
            objective_weights,
            pareto_front: Vec::new(),
            optimization_objectives,
        })
    }

    pub fn update_pareto_front(&mut self, population: &[Individual<T>]) -> Result<Vec<Individual<T>>> {
        self.pareto_front = self.compute_pareto_front(population)?;
        Ok(self.pareto_front.clone())
    }

    fn compute_pareto_front(&self, population: &[Individual<T>]) -> Result<Vec<Individual<T>>> {
        let mut front = Vec::new();

        for individual in population {
            let mut dominated = false;

            for existing in &front {
                if self.dominates(existing, individual)? {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                front.retain(|existing| !self.dominates(individual, existing).unwrap_or(false));
                front.push(individual.clone());
            }
        }

        Ok(front)
    }

    fn dominates(&self, a: &Individual<T>, b: &Individual<T>) -> Result<bool> {
        let objectives_a = self.extract_objectives(&a.metrics)?;
        let objectives_b = self.extract_objectives(&b.metrics)?;

        let mut at_least_one_better = false;
        for (obj_a, obj_b) in objectives_a.iter().zip(objectives_b.iter()) {
            if obj_a < obj_b {
                return Ok(false);
            }
            if obj_a > obj_b {
                at_least_one_better = true;
            }
        }

        Ok(at_least_one_better)
    }

    fn extract_objectives(&self, metrics: &super::evaluation::EvaluationMetrics) -> Result<Vec<f64>> {
        Ok(vec![
            metrics.accuracy,
            1.0 / (1.0 + metrics.training_time_seconds / 3600.0),
            1.0 / (1.0 + metrics.memory_usage_mb as f64 / 1024.0),
        ])
    }

    pub fn get_pareto_front(&self) -> &[Individual<T>] {
        &self.pareto_front
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationObjective {
    Accuracy,
    Speed,
    Memory,
    Energy,
    Complexity,
}

/// Pareto front representation
pub struct ParetoFront<T: Float + Debug + Send + Sync + 'static> {
    pub individuals: Vec<Individual<T>>,
    pub objective_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_objective_optimizer() {
        let weights = vec![0.5, 0.3, 0.2];
        let optimizer = MultiObjectiveOptimizer::<f32>::new(weights);
        assert!(optimizer.is_ok());
    }
}