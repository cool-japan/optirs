//! Performance evaluator for optimizer architectures
//!
//! Main evaluation entry point for NAS system.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Instant;

use super::benchmark::{BenchmarkSuite, TestResult};
use super::cache::EvaluationCache;
use super::predictor::PerformancePredictor;
use super::resource::ResourceMonitor;
use super::statistical::StatisticalAnalyzer;
use crate::error::Result;
use crate::nas_engine::results::EvaluationResults;
use crate::{EvaluationConfig, EvaluationMetric, OptimizerArchitecture};

/// Performance evaluator for optimizer architectures
pub struct PerformanceEvaluator<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluation configuration
    config: EvaluationConfig,

    /// Benchmark suite
    benchmark_suite: BenchmarkSuite<T>,

    /// Performance predictor
    predictor: Option<PerformancePredictor<T>>,

    /// Evaluation cache
    evaluation_cache: EvaluationCache<T>,

    /// Statistical analyzer
    statistical_analyzer: StatisticalAnalyzer<T>,

    /// Resource monitor
    resource_monitor: ResourceMonitor<T>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + std::iter::Sum>
    PerformanceEvaluator<T>
{
    /// Create new performance evaluator
    pub fn new(config: EvaluationConfig) -> Result<Self> {
        Ok(Self {
            benchmark_suite: BenchmarkSuite::new()?,
            predictor: None,
            evaluation_cache: EvaluationCache::new(),
            statistical_analyzer: StatisticalAnalyzer::new(),
            resource_monitor: ResourceMonitor::new(),
            config,
        })
    }

    /// Initialize the evaluator
    pub fn initialize(&mut self) -> Result<()> {
        // Initialize benchmark suite
        self.benchmark_suite.initialize(&self.config)?;

        // Initialize performance predictor if enabled
        if self.config.performance_prediction {
            self.predictor = Some(PerformancePredictor::new(&self.config)?);
        }

        // Start resource monitoring
        self.resource_monitor.start_monitoring()?;

        Ok(())
    }

    /// Evaluate an optimizer architecture
    pub fn evaluate_architecture(
        &mut self,
        architecture: &OptimizerArchitecture,
    ) -> Result<EvaluationResults<T>> {
        let start_time = Instant::now();

        // Check cache first
        let cache_key = self.generate_cache_key(architecture);
        if let Some(cached_result) = self.evaluation_cache.get(&cache_key) {
            return Ok(cached_result.results.clone());
        }

        // Run benchmarks
        let benchmark_results = self.benchmark_suite.run_benchmarks(architecture)?;

        // Compute overall metrics
        let mut metric_scores = HashMap::new();

        // Aggregate benchmark scores
        let overall_score = self.aggregate_benchmark_scores(&benchmark_results)?;
        metric_scores.insert(EvaluationMetric::FinalPerformance, overall_score);

        // Compute convergence speed
        let convergence_speed = self.compute_convergence_speed(&benchmark_results)?;
        metric_scores.insert(EvaluationMetric::ConvergenceSpeed, convergence_speed);

        // Compute stability metrics
        let stability = self.compute_stability(&benchmark_results)?;
        metric_scores.insert(EvaluationMetric::TrainingStability, stability);

        // Compute efficiency metrics
        let memory_efficiency = self.compute_memory_efficiency(&benchmark_results)?;
        let computational_efficiency = self.compute_computational_efficiency(&benchmark_results)?;
        metric_scores.insert(EvaluationMetric::MemoryEfficiency, memory_efficiency);
        metric_scores.insert(
            EvaluationMetric::ComputationalEfficiency,
            computational_efficiency,
        );

        // Statistical analysis
        let confidence_intervals = self
            .statistical_analyzer
            .compute_confidence_intervals(&benchmark_results)?;

        let evaluation_time = start_time.elapsed();

        let results = EvaluationResults {
            metric_scores,
            overall_score,
            confidence_intervals,
            evaluation_time,
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: std::collections::HashMap::new(),
            training_trajectory: Vec::new(),
        };

        // Cache results
        self.evaluation_cache.insert(cache_key, results.clone());

        Ok(results)
    }

    fn generate_cache_key(&self, architecture: &OptimizerArchitecture) -> String {
        // Generate a unique key for the architecture
        // This is simplified - in practice would use better hashing
        format!("arch_{}", architecture.structure.len())
    }

    fn aggregate_benchmark_scores(&self, results: &[TestResult<T>]) -> Result<T> {
        if results.is_empty() {
            return Ok(T::zero());
        }

        let sum: T = results.iter().map(|r| r.normalized_score).sum();
        Ok(sum / T::from(results.len()).expect("conversion failed"))
    }

    fn compute_convergence_speed(&self, results: &[TestResult<T>]) -> Result<T> {
        // Simplified convergence speed computation
        let avg_time: f64 = results
            .iter()
            .map(|r| r.execution_time.as_secs_f64())
            .sum::<f64>()
            / results.len() as f64;

        // Inverse of average time (higher is better)
        Ok(T::from(1.0 / (avg_time + 1e-6)).expect("conversion failed"))
    }

    fn compute_stability(&self, results: &[TestResult<T>]) -> Result<T> {
        if results.len() < 2 {
            return Ok(T::one());
        }

        let scores: Vec<T> = results.iter().map(|r| r.score).collect();
        let mean =
            scores.iter().cloned().sum::<T>() / T::from(scores.len()).expect("conversion failed");
        let variance = scores.iter().map(|&s| (s - mean) * (s - mean)).sum::<T>()
            / T::from(scores.len()).expect("conversion failed");
        let std_dev = variance.sqrt();

        // Stability as inverse of coefficient of variation
        let cv = std_dev
            / mean
                .abs()
                .max(scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero()));
        Ok(
            T::one()
                / (cv + scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero())),
        )
    }

    fn compute_memory_efficiency(&self, results: &[TestResult<T>]) -> Result<T> {
        let avg_memory = results
            .iter()
            .map(|r| r.resource_usage.memory_gb)
            .sum::<f64>()
            / results.len() as f64;

        // Efficiency as inverse of memory usage
        let efficiency = 1.0 / (avg_memory + 1e-6);
        Ok(scirs2_core::numeric::NumCast::from(efficiency).unwrap_or_else(|| T::zero()))
    }

    fn compute_computational_efficiency(&self, results: &[TestResult<T>]) -> Result<T> {
        let avg_cpu_time = results
            .iter()
            .map(|r| r.resource_usage.cpu_time_seconds)
            .sum::<f64>()
            / results.len() as f64;

        // Efficiency as inverse of CPU time
        let efficiency = 1.0 / (avg_cpu_time + 1e-6);
        Ok(scirs2_core::numeric::NumCast::from(efficiency).unwrap_or_else(|| T::zero()))
    }

    /// Get the evaluation cache
    pub fn cache(&self) -> &EvaluationCache<T> {
        &self.evaluation_cache
    }

    /// Get the statistical analyzer
    pub fn analyzer(&self) -> &StatisticalAnalyzer<T> {
        &self.statistical_analyzer
    }

    /// Get the resource monitor
    pub fn resource_monitor(&self) -> &ResourceMonitor<T> {
        &self.resource_monitor
    }

    /// Get the evaluation config
    pub fn config(&self) -> &EvaluationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_evaluator_creation() {
        // Skip this test for now - EvaluationConfig needs Default implementation
        // but some dependent types (EvaluationBudget, StatisticalTestingConfig) are not yet defined
    }
}
