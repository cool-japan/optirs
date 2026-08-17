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
#[derive(Debug)]
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
    ///
    /// Runs the full benchmark suite: every registered test function is
    /// actually minimized by the concrete optimizer the architecture describes,
    /// and the achieved objectives are aggregated into the returned metrics.
    /// Repeated evaluation of the same architecture is served from the cache.
    pub fn evaluate_architecture(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
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

    /// Build a cache key that identifies the *whole* architecture.
    ///
    /// An FNV-1a hash is taken over the component list (in order), every
    /// hyperparameter and parameter entry (key-sorted, so `HashMap` iteration
    /// order cannot perturb the result) and the connection list. Two
    /// architectures therefore share a cache entry only when they would produce
    /// an identical optimizer, and a change to any single hyperparameter yields
    /// a different key.
    fn generate_cache_key(&self, architecture: &OptimizerArchitecture<T>) -> String {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

        let mut hash: u64 = FNV_OFFSET;
        let mix_bytes = |bytes: &[u8], hash: &mut u64| {
            for byte in bytes {
                *hash ^= *byte as u64;
                *hash = hash.wrapping_mul(FNV_PRIME);
            }
            // Field separator so ["ab","c"] and ["a","bc"] differ.
            *hash ^= 0x1f;
            *hash = hash.wrapping_mul(FNV_PRIME);
        };

        for component in &architecture.components {
            mix_bytes(component.as_bytes(), &mut hash);
        }

        let mut hyperparameters: Vec<(&String, &T)> = architecture.hyperparameters.iter().collect();
        hyperparameters.sort_by(|a, b| a.0.cmp(b.0));
        for (key, value) in hyperparameters {
            mix_bytes(key.as_bytes(), &mut hash);
            mix_bytes(
                &value.to_f64().unwrap_or(0.0).to_bits().to_le_bytes(),
                &mut hash,
            );
        }

        let mut parameters: Vec<(&String, &T)> = architecture.parameters.iter().collect();
        parameters.sort_by(|a, b| a.0.cmp(b.0));
        for (key, value) in parameters {
            mix_bytes(key.as_bytes(), &mut hash);
            mix_bytes(
                &value.to_f64().unwrap_or(0.0).to_bits().to_le_bytes(),
                &mut hash,
            );
        }

        for (from, to) in &architecture.connections {
            mix_bytes(&(*from as u64).to_le_bytes(), &mut hash);
            mix_bytes(&(*to as u64).to_le_bytes(), &mut hash);
        }

        format!("arch_{:016x}", hash)
    }

    fn aggregate_benchmark_scores(&self, results: &[TestResult<T>]) -> Result<T> {
        if results.is_empty() {
            return Ok(T::zero());
        }

        let sum: T = results.iter().map(|r| r.normalized_score).sum();
        let count: T =
            scirs2_core::numeric::NumCast::from(results.len()).unwrap_or_else(|| T::one());
        Ok(sum / count)
    }

    fn compute_convergence_speed(&self, results: &[TestResult<T>]) -> Result<T> {
        if results.is_empty() {
            return Ok(T::zero());
        }

        let avg_time: f64 = results
            .iter()
            .map(|r| r.execution_time.as_secs_f64())
            .sum::<f64>()
            / results.len() as f64;

        // Inverse of average time (higher is better)
        Ok(scirs2_core::numeric::NumCast::from(1.0 / (avg_time + 1e-6))
            .unwrap_or_else(|| T::zero()))
    }

    fn compute_stability(&self, results: &[TestResult<T>]) -> Result<T> {
        if results.len() < 2 {
            return Ok(T::one());
        }

        let scores: Vec<T> = results.iter().map(|r| r.score).collect();
        let count: T =
            scirs2_core::numeric::NumCast::from(scores.len()).unwrap_or_else(|| T::one());
        let mean = scores.iter().cloned().sum::<T>() / count;
        let variance = scores.iter().map(|&s| (s - mean) * (s - mean)).sum::<T>() / count;
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
        if results.is_empty() {
            return Ok(T::zero());
        }

        let avg_memory = results
            .iter()
            .map(|r| r.resource_usage.memory_gb.to_f64().unwrap_or(0.0))
            .sum::<f64>()
            / results.len() as f64;

        // Efficiency as inverse of memory usage
        let efficiency = 1.0 / (avg_memory + 1e-6);
        Ok(scirs2_core::numeric::NumCast::from(efficiency).unwrap_or_else(|| T::zero()))
    }

    fn compute_computational_efficiency(&self, results: &[TestResult<T>]) -> Result<T> {
        if results.is_empty() {
            return Ok(T::zero());
        }

        let avg_cpu_time = results
            .iter()
            .map(|r| r.resource_usage.cpu_time_seconds.to_f64().unwrap_or(0.0))
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

    fn config() -> EvaluationConfig {
        EvaluationConfig {
            epochs: 40,
            ..Default::default()
        }
    }

    fn architecture(
        id: &str,
        component: &str,
        hyper: &[(&str, f64)],
    ) -> OptimizerArchitecture<f64> {
        let mut hyperparameters = HashMap::new();
        for (k, v) in hyper {
            hyperparameters.insert((*k).to_string(), *v);
        }
        OptimizerArchitecture {
            components: vec![component.to_string()],
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters,
            architecture_id: id.to_string(),
        }
    }

    fn evaluator() -> PerformanceEvaluator<f64> {
        let mut evaluator = PerformanceEvaluator::<f64>::new(config()).expect("construct");
        evaluator.initialize().expect("initialize");
        evaluator
    }

    #[test]
    fn test_performance_evaluator_creation() {
        let evaluator = PerformanceEvaluator::<f64>::new(config());
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_evaluation_produces_real_metrics() {
        let mut evaluator = evaluator();
        let results = evaluator
            .evaluate_architecture(&architecture(
                "adam_good",
                "Adam",
                &[("learning_rate", 0.1)],
            ))
            .expect("evaluate");

        assert!(results.success);
        // The score is an aggregate of real benchmark progress, not a constant.
        assert!(results.overall_score > 0.0 && results.overall_score <= 1.0);
        assert!(results
            .metric_scores
            .contains_key(&EvaluationMetric::FinalPerformance));
        assert!(results
            .metric_scores
            .contains_key(&EvaluationMetric::ConvergenceSpeed));
    }

    #[test]
    fn test_two_different_architectures_get_different_scores() {
        let mut evaluator = evaluator();

        let good = evaluator
            .evaluate_architecture(&architecture("good", "Adam", &[("learning_rate", 0.1)]))
            .expect("good")
            .overall_score;
        let bad = evaluator
            .evaluate_architecture(&architecture("bad", "SGD", &[("learning_rate", 1e-7)]))
            .expect("bad")
            .overall_score;

        assert!(
            (good - bad).abs() > 1e-9,
            "distinct architectures must score differently ({} vs {})",
            good,
            bad
        );
        assert!(good > bad, "the well-tuned candidate must score higher");
    }

    #[test]
    fn test_identical_architecture_reproduces_and_hits_cache() {
        let mut evaluator = evaluator();
        let arch = architecture("repeat", "AdamW", &[("learning_rate", 0.05)]);

        let first = evaluator.evaluate_architecture(&arch).expect("first");
        assert_eq!(evaluator.cache().len(), 1);

        let second = evaluator.evaluate_architecture(&arch).expect("second");
        assert_eq!(evaluator.cache().len(), 1, "second call must hit the cache");
        assert_eq!(first.overall_score, second.overall_score);
    }

    #[test]
    fn test_cache_key_covers_the_whole_architecture() {
        let evaluator = evaluator();

        let base = architecture("a", "Adam", &[("learning_rate", 0.01)]);
        let same_shape_other_lr = architecture("a", "Adam", &[("learning_rate", 0.02)]);
        let other_component = architecture("a", "SGD", &[("learning_rate", 0.01)]);
        let identical = architecture("different_id", "Adam", &[("learning_rate", 0.01)]);

        let key = evaluator.generate_cache_key(&base);
        assert_ne!(key, evaluator.generate_cache_key(&same_shape_other_lr));
        assert_ne!(key, evaluator.generate_cache_key(&other_component));
        // The key is content-derived, so the (cosmetic) identifier is irrelevant.
        assert_eq!(key, evaluator.generate_cache_key(&identical));
    }
}
