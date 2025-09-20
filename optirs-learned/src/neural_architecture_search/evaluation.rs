// Architecture evaluation and metrics

use scirs2_core::ndarray_ext::{Array1, Array2};
use num_traits::Float;
use std::fmt::Debug;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use crate::error::Result;
use super::config::{NASConfig, ArchitectureSpec};

/// Architecture evaluator
pub struct ArchitectureEvaluator<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluation strategy
    strategy: EvaluationStrategy,
    /// Evaluation configuration
    config: EvaluationConfig,
    /// Cached evaluations
    cache: HashMap<String, EvaluationMetrics>,
    /// Performance estimator
    estimator: PerformanceEstimator<T>,
    /// Resource tracker
    resource_tracker: ResourceTracker,
}

impl<T: Float + Debug + Send + Sync + 'static> ArchitectureEvaluator<T> {
    /// Create new evaluator
    pub fn new(config: &NASConfig) -> Result<Self> {
        let evaluation_config = EvaluationConfig::from_nas_config(config);
        let strategy = EvaluationStrategy::default();
        let estimator = PerformanceEstimator::new()?;
        let resource_tracker = ResourceTracker::new();

        Ok(Self {
            strategy,
            config: evaluation_config,
            cache: HashMap::new(),
            estimator,
            resource_tracker,
        })
    }

    /// Evaluate architecture
    pub fn evaluate(&mut self, architecture: &str) -> Result<EvaluationMetrics> {
        // Check cache first
        if let Some(cached_metrics) = self.cache.get(architecture) {
            return Ok(cached_metrics.clone());
        }

        let start_time = Instant::now();

        // Parse architecture
        let spec = self.parse_architecture(architecture)?;

        // Validate architecture
        if !self.is_valid_architecture(&spec)? {
            return Ok(EvaluationMetrics::invalid());
        }

        // Perform evaluation based on strategy
        let metrics = match self.strategy {
            EvaluationStrategy::FullTraining => self.evaluate_full_training(&spec)?,
            EvaluationStrategy::EarlyStoppingTraining => self.evaluate_early_stopping(&spec)?,
            EvaluationStrategy::ProxyMetrics => self.evaluate_proxy_metrics(&spec)?,
            EvaluationStrategy::WeightSharing => self.evaluate_weight_sharing(&spec)?,
            EvaluationStrategy::PerformanceEstimation => self.evaluate_performance_estimation(&spec)?,
        };

        // Update resource tracking
        let elapsed = start_time.elapsed();
        self.resource_tracker.record_evaluation(elapsed, &metrics);

        // Cache result
        self.cache.insert(architecture.to_string(), metrics.clone());

        Ok(metrics)
    }

    /// Parse architecture string to specification
    fn parse_architecture(&self, architecture: &str) -> Result<ArchitectureSpec> {
        serde_json::from_str(architecture)
            .map_err(|e| crate::error::OptimError::Other(format!("Parse error: {}", e)))
    }

    /// Validate architecture
    fn is_valid_architecture(&self, spec: &ArchitectureSpec) -> Result<bool> {
        // Check basic constraints
        if spec.layers.is_empty() {
            return Ok(false);
        }

        // Check memory constraints
        if spec.estimated_memory_mb > self.config.max_memory_mb {
            return Ok(false);
        }

        // Check parameter count
        if spec.estimated_params > self.config.max_parameters {
            return Ok(false);
        }

        // Check FLOP count
        if spec.estimated_flops > self.config.max_flops {
            return Ok(false);
        }

        Ok(true)
    }

    /// Full training evaluation
    fn evaluate_full_training(&mut self, spec: &ArchitectureSpec) -> Result<EvaluationMetrics> {
        // Simulate full training
        let training_time = self.estimate_training_time(spec)?;
        let accuracy = self.simulate_training_accuracy(spec)?;
        let memory_usage = spec.estimated_memory_mb;
        let flops = spec.estimated_flops;

        Ok(EvaluationMetrics {
            accuracy,
            training_time_seconds: training_time.as_secs_f64(),
            inference_time_ms: self.estimate_inference_time(spec)?,
            memory_usage_mb: memory_usage,
            flops,
            parameters: spec.estimated_params,
            energy_consumption: self.estimate_energy_consumption(spec)?,
            convergence_rate: self.estimate_convergence_rate(spec)?,
            robustness_score: self.estimate_robustness(spec)?,
            generalization_score: self.estimate_generalization(spec)?,
            efficiency_score: self.calculate_efficiency_score(accuracy, training_time, flops)?,
            valid: true,
        })
    }

    /// Early stopping evaluation
    fn evaluate_early_stopping(&mut self, spec: &ArchitectureSpec) -> Result<EvaluationMetrics> {
        let mut metrics = self.evaluate_full_training(spec)?;

        // Reduce training time due to early stopping
        metrics.training_time_seconds *= 0.7; // Assume 30% time savings
        metrics.accuracy *= 0.95; // Slight accuracy reduction

        Ok(metrics)
    }

    /// Proxy metrics evaluation
    fn evaluate_proxy_metrics(&mut self, spec: &ArchitectureSpec) -> Result<EvaluationMetrics> {
        // Use proxy metrics instead of full training
        let complexity_score = self.calculate_complexity_score(spec)?;
        let expressivity_score = self.calculate_expressivity_score(spec)?;
        let connectivity_score = self.calculate_connectivity_score(spec)?;

        // Estimate accuracy from proxy metrics
        let estimated_accuracy = (complexity_score + expressivity_score + connectivity_score) / 3.0;

        Ok(EvaluationMetrics {
            accuracy: estimated_accuracy,
            training_time_seconds: 100.0, // Minimal time for proxy evaluation
            inference_time_ms: self.estimate_inference_time(spec)?,
            memory_usage_mb: spec.estimated_memory_mb,
            flops: spec.estimated_flops,
            parameters: spec.estimated_params,
            energy_consumption: self.estimate_energy_consumption(spec)?,
            convergence_rate: 0.8, // Default estimate
            robustness_score: 0.7,
            generalization_score: estimated_accuracy * 0.9,
            efficiency_score: self.calculate_efficiency_score(estimated_accuracy, Duration::from_secs(100), spec.estimated_flops)?,
            valid: true,
        })
    }

    /// Weight sharing evaluation
    fn evaluate_weight_sharing(&mut self, spec: &ArchitectureSpec) -> Result<EvaluationMetrics> {
        // Simulate weight sharing evaluation (faster than full training)
        let mut metrics = self.evaluate_full_training(spec)?;
        metrics.training_time_seconds *= 0.1; // Much faster with weight sharing
        metrics.accuracy *= 0.85; // Lower accuracy due to shared weights
        Ok(metrics)
    }

    /// Performance estimation evaluation
    fn evaluate_performance_estimation(&mut self, spec: &ArchitectureSpec) -> Result<EvaluationMetrics> {
        let estimated_metrics = self.estimator.estimate_performance(spec)?;
        Ok(estimated_metrics)
    }

    /// Estimate training time
    fn estimate_training_time(&self, spec: &ArchitectureSpec) -> Result<Duration> {
        let base_time = 1000.0; // Base training time in seconds
        let complexity_factor = (spec.estimated_params as f64).log10() / 6.0; // Normalize by 1M params
        let time_seconds = base_time * (1.0 + complexity_factor);
        Ok(Duration::from_secs_f64(time_seconds))
    }

    /// Simulate training accuracy
    fn simulate_training_accuracy(&self, spec: &ArchitectureSpec) -> Result<f64> {
        // Simple heuristic based on architecture complexity
        let param_factor = (spec.estimated_params as f64 / 1_000_000.0).min(1.0);
        let layer_factor = (spec.layers.len() as f64 / 10.0).min(1.0);
        let base_accuracy = 0.6 + 0.3 * param_factor + 0.1 * layer_factor;

        // Add some randomness
        let noise = (scirs2_core::random::random::<f64>() - 0.5) * 0.1;
        Ok((base_accuracy + noise).clamp(0.0, 1.0))
    }

    /// Estimate inference time
    fn estimate_inference_time(&self, spec: &ArchitectureSpec) -> Result<f64> {
        let base_time = 1.0; // Base inference time in ms
        let flop_factor = (spec.estimated_flops as f64 / 1_000_000.0).log10();
        Ok(base_time * (1.0 + flop_factor * 0.1))
    }

    /// Estimate energy consumption
    fn estimate_energy_consumption(&self, spec: &ArchitectureSpec) -> Result<f64> {
        // Energy proportional to FLOPs and parameters
        let energy_per_flop = 1e-12; // Joules per FLOP
        let energy_per_param = 1e-9;  // Joules per parameter

        let flop_energy = spec.estimated_flops as f64 * energy_per_flop;
        let param_energy = spec.estimated_params as f64 * energy_per_param;

        Ok(flop_energy + param_energy)
    }

    /// Estimate convergence rate
    fn estimate_convergence_rate(&self, spec: &ArchitectureSpec) -> Result<f64> {
        // Deeper networks tend to converge slower
        let depth_penalty = (spec.layers.len() as f64 / 20.0).min(1.0);
        let base_rate = 0.8;
        Ok(base_rate * (1.0 - depth_penalty * 0.3))
    }

    /// Estimate robustness
    fn estimate_robustness(&self, spec: &ArchitectureSpec) -> Result<f64> {
        // More parameters generally increase robustness
        let param_factor = (spec.estimated_params as f64 / 10_000_000.0).min(1.0);
        Ok(0.5 + 0.4 * param_factor)
    }

    /// Estimate generalization
    fn estimate_generalization(&self, spec: &ArchitectureSpec) -> Result<f64> {
        // Balance between complexity and simplicity
        let complexity = spec.estimated_params as f64 / 1_000_000.0;
        let generalization = if complexity < 1.0 {
            0.6 + 0.3 * complexity
        } else {
            0.9 - 0.2 * (complexity - 1.0).min(1.0)
        };
        Ok(generalization.clamp(0.0, 1.0))
    }

    /// Calculate efficiency score
    fn calculate_efficiency_score(&self, accuracy: f64, training_time: Duration, flops: u64) -> Result<f64> {
        let time_factor = 1.0 / (1.0 + training_time.as_secs_f64() / 3600.0); // Normalize by hour
        let flop_factor = 1.0 / (1.0 + flops as f64 / 1e9); // Normalize by billion FLOPs
        let efficiency = accuracy * time_factor * flop_factor;
        Ok(efficiency)
    }

    /// Calculate complexity score
    fn calculate_complexity_score(&self, spec: &ArchitectureSpec) -> Result<f64> {
        let param_score = (spec.estimated_params as f64 / 1_000_000.0).min(1.0);
        let depth_score = (spec.layers.len() as f64 / 20.0).min(1.0);
        Ok((param_score + depth_score) / 2.0)
    }

    /// Calculate expressivity score
    fn calculate_expressivity_score(&self, spec: &ArchitectureSpec) -> Result<f64> {
        // Score based on layer diversity and connectivity
        let mut layer_types = std::collections::HashSet::new();
        for layer in &spec.layers {
            layer_types.insert(layer.layer_type);
        }

        let diversity_score = layer_types.len() as f64 / 5.0; // Assume max 5 layer types
        Ok(diversity_score.min(1.0))
    }

    /// Calculate connectivity score
    fn calculate_connectivity_score(&self, _spec: &ArchitectureSpec) -> Result<f64> {
        // Simplified connectivity score
        Ok(0.7) // Default score
    }

    /// Get evaluation statistics
    pub fn get_statistics(&self) -> EvaluationStatistics {
        EvaluationStatistics {
            total_evaluations: self.cache.len(),
            cache_size: self.cache.len(),
            total_evaluation_time: self.resource_tracker.total_time,
            average_evaluation_time: self.resource_tracker.average_time(),
            resource_usage: self.resource_tracker.get_usage(),
        }
    }

    /// Clear evaluation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Evaluation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationStrategy {
    /// Full training evaluation
    FullTraining,
    /// Early stopping training
    EarlyStoppingTraining,
    /// Proxy metrics without training
    ProxyMetrics,
    /// Weight sharing evaluation
    WeightSharing,
    /// Performance estimation
    PerformanceEstimation,
}

impl Default for EvaluationStrategy {
    fn default() -> Self {
        EvaluationStrategy::ProxyMetrics
    }
}

/// Evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub max_memory_mb: usize,
    pub max_parameters: usize,
    pub max_flops: u64,
    pub max_training_time_hours: f64,
    pub target_accuracy: f64,
    pub enable_caching: bool,
    pub parallel_evaluations: usize,
}

impl EvaluationConfig {
    pub fn from_nas_config(config: &NASConfig) -> Self {
        Self {
            max_memory_mb: config.constraints.max_memory_mb,
            max_parameters: config.constraints.max_params_per_layer * config.constraints.max_layers,
            max_flops: config.constraints.max_flops,
            max_training_time_hours: config.constraints.max_training_time_secs as f64 / 3600.0,
            target_accuracy: config.constraints.min_accuracy,
            enable_caching: true,
            parallel_evaluations: config.parallelization_level,
        }
    }
}

/// Evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Model accuracy
    pub accuracy: f64,
    /// Training time in seconds
    pub training_time_seconds: f64,
    /// Inference time in milliseconds
    pub inference_time_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// Floating point operations
    pub flops: u64,
    /// Number of parameters
    pub parameters: usize,
    /// Energy consumption in Joules
    pub energy_consumption: f64,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Robustness score
    pub robustness_score: f64,
    /// Generalization score
    pub generalization_score: f64,
    /// Overall efficiency score
    pub efficiency_score: f64,
    /// Whether evaluation is valid
    pub valid: bool,
}

impl EvaluationMetrics {
    /// Create invalid metrics
    pub fn invalid() -> Self {
        Self {
            accuracy: 0.0,
            training_time_seconds: f64::INFINITY,
            inference_time_ms: f64::INFINITY,
            memory_usage_mb: usize::MAX,
            flops: u64::MAX,
            parameters: usize::MAX,
            energy_consumption: f64::INFINITY,
            convergence_rate: 0.0,
            robustness_score: 0.0,
            generalization_score: 0.0,
            efficiency_score: 0.0,
            valid: false,
        }
    }

    /// Get overall score (weighted combination of metrics)
    pub fn overall_score(&self, weights: &[f64]) -> f64 {
        if !self.valid {
            return 0.0;
        }

        let metrics = vec![
            self.accuracy,
            1.0 / (1.0 + self.training_time_seconds / 3600.0), // Inverse time preference
            1.0 / (1.0 + self.inference_time_ms / 1000.0),
            1.0 / (1.0 + self.memory_usage_mb as f64 / 1024.0),
            self.convergence_rate,
            self.robustness_score,
            self.generalization_score,
            self.efficiency_score,
        ];

        let default_weights = vec![0.3, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05];
        let used_weights = if weights.len() == metrics.len() { weights } else { &default_weights };

        metrics.iter()
            .zip(used_weights.iter())
            .map(|(metric, weight)| metric * weight)
            .sum()
    }
}

/// Performance estimator using surrogate models
pub struct PerformanceEstimator<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> PerformanceEstimator<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn estimate_performance(&self, spec: &ArchitectureSpec) -> Result<EvaluationMetrics> {
        // Simplified performance estimation
        let accuracy = 0.7 + (spec.estimated_params as f64 / 10_000_000.0).min(0.2);

        Ok(EvaluationMetrics {
            accuracy,
            training_time_seconds: 500.0,
            inference_time_ms: 10.0,
            memory_usage_mb: spec.estimated_memory_mb,
            flops: spec.estimated_flops,
            parameters: spec.estimated_params,
            energy_consumption: 100.0,
            convergence_rate: 0.8,
            robustness_score: 0.7,
            generalization_score: accuracy * 0.9,
            efficiency_score: 0.6,
            valid: true,
        })
    }
}

/// Resource tracker for evaluations
#[derive(Debug)]
pub struct ResourceTracker {
    total_evaluations: usize,
    total_time: Duration,
    evaluation_times: Vec<Duration>,
}

impl ResourceTracker {
    pub fn new() -> Self {
        Self {
            total_evaluations: 0,
            total_time: Duration::new(0, 0),
            evaluation_times: Vec::new(),
        }
    }

    pub fn record_evaluation(&mut self, duration: Duration, _metrics: &EvaluationMetrics) {
        self.total_evaluations += 1;
        self.total_time += duration;
        self.evaluation_times.push(duration);

        // Keep only recent evaluations
        if self.evaluation_times.len() > 1000 {
            self.evaluation_times.remove(0);
        }
    }

    pub fn average_time(&self) -> Duration {
        if self.total_evaluations == 0 {
            Duration::new(0, 0)
        } else {
            self.total_time / self.total_evaluations as u32
        }
    }

    pub fn get_usage(&self) -> ResourceUsage {
        ResourceUsage {
            total_evaluations: self.total_evaluations,
            total_time: self.total_time,
            average_time: self.average_time(),
        }
    }
}

/// Evaluation statistics
#[derive(Debug, Clone)]
pub struct EvaluationStatistics {
    pub total_evaluations: usize,
    pub cache_size: usize,
    pub total_evaluation_time: Duration,
    pub average_evaluation_time: Duration,
    pub resource_usage: ResourceUsage,
}

/// Resource usage information
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub total_evaluations: usize,
    pub total_time: Duration,
    pub average_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned_optimizers::neural_architecture_search::config::NASConfig;

    #[test]
    fn test_evaluator_creation() {
        let config = NASConfig::new();
        let evaluator = ArchitectureEvaluator::<f32>::new(&config);
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_evaluation_metrics() {
        let metrics = EvaluationMetrics::invalid();
        assert!(!metrics.valid);
        assert_eq!(metrics.accuracy, 0.0);

        let weights = vec![1.0, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05];
        let score = metrics.overall_score(&weights);
        assert_eq!(score, 0.0); // Invalid metrics should give 0 score
    }

    #[test]
    fn test_resource_tracker() {
        let mut tracker = ResourceTracker::new();
        let duration = Duration::from_millis(100);
        let metrics = EvaluationMetrics::invalid();

        tracker.record_evaluation(duration, &metrics);
        assert_eq!(tracker.total_evaluations, 1);
        assert_eq!(tracker.average_time(), duration);
    }
}