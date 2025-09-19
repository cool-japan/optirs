// Path optimization algorithms

use serde::{Deserialize, Serialize};

/// Path optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathOptimization {
    /// Optimization enabled
    pub enabled: bool,
    /// Optimization algorithms
    pub algorithms: Vec<String>,
    /// Optimization criteria
    pub criteria: OptimizationCriteria,
}

impl Default for PathOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec!["shortest_path".to_string(), "least_cost".to_string()],
            criteria: OptimizationCriteria::default(),
        }
    }
}

/// Optimization criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCriteria {
    /// Minimize latency
    pub minimize_latency: bool,
    /// Minimize cost
    pub minimize_cost: bool,
    /// Maximize reliability
    pub maximize_reliability: bool,
}

impl Default for OptimizationCriteria {
    fn default() -> Self {
        Self {
            minimize_latency: true,
            minimize_cost: false,
            maximize_reliability: true,
        }
    }
}
