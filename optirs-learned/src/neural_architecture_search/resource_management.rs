// Resource management for neural architecture search

use std::time::{Duration, Instant};
use crate::error::Result;

/// Resource manager for NAS
pub struct ResourceManager {
    budget: ComputeBudget,
    usage: ResourceUsage,
    constraints: ResourceConstraints,
    start_time: Instant,
}

impl ResourceManager {
    pub fn new(evaluation_budget: usize) -> Result<Self> {
        let budget = ComputeBudget {
            max_evaluations: evaluation_budget,
            max_time_hours: 24.0,
            max_memory_gb: 32.0,
            max_gpu_hours: 12.0,
        };

        let constraints = ResourceConstraints::default();
        let usage = ResourceUsage::new();

        Ok(Self {
            budget,
            usage,
            constraints,
            start_time: Instant::now(),
        })
    }

    pub fn has_budget_remaining(&self) -> bool {
        self.usage.evaluations_used < self.budget.max_evaluations &&
        self.get_elapsed_hours() < self.budget.max_time_hours &&
        self.usage.memory_used_gb < self.budget.max_memory_gb
    }

    pub fn consume_evaluation_budget(&mut self, count: usize) -> Result<()> {
        if self.usage.evaluations_used + count > self.budget.max_evaluations {
            return Err(crate::error::OptimError::Other("Evaluation budget exceeded".to_string()));
        }

        self.usage.evaluations_used += count;
        Ok(())
    }

    fn get_elapsed_hours(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() / 3600.0
    }

    pub fn get_usage_stats(&self) -> super::ResourceUsageStats {
        super::ResourceUsageStats {
            evaluations_used: self.usage.evaluations_used,
            evaluations_remaining: self.budget.max_evaluations.saturating_sub(self.usage.evaluations_used),
            total_compute_time: self.start_time.elapsed(),
        }
    }
}

/// Compute budget for NAS
#[derive(Debug, Clone)]
pub struct ComputeBudget {
    pub max_evaluations: usize,
    pub max_time_hours: f64,
    pub max_memory_gb: f64,
    pub max_gpu_hours: f64,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub evaluations_used: usize,
    pub time_used_hours: f64,
    pub memory_used_gb: f64,
    pub gpu_hours_used: f64,
}

impl ResourceUsage {
    pub fn new() -> Self {
        Self {
            evaluations_used: 0,
            time_used_hours: 0.0,
            memory_used_gb: 0.0,
            gpu_hours_used: 0.0,
        }
    }
}

/// Resource constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_concurrent_evaluations: usize,
    pub memory_per_evaluation_gb: f64,
    pub time_per_evaluation_minutes: f64,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_concurrent_evaluations: 4,
            memory_per_evaluation_gb: 2.0,
            time_per_evaluation_minutes: 30.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager() {
        let manager = ResourceManager::new(100);
        assert!(manager.is_ok());

        let rm = manager.unwrap();
        assert!(rm.has_budget_remaining());
        assert_eq!(rm.usage.evaluations_used, 0);
    }

    #[test]
    fn test_budget_consumption() {
        let mut manager = ResourceManager::new(10).unwrap();

        let result = manager.consume_evaluation_budget(5);
        assert!(result.is_ok());
        assert_eq!(manager.usage.evaluations_used, 5);

        let result = manager.consume_evaluation_budget(10);
        assert!(result.is_err()); // Should exceed budget
    }
}