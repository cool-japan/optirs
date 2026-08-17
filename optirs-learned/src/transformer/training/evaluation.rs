use std::fmt::Debug;
// Evaluation metrics and methods for transformer optimization
//
// This module implements comprehensive evaluation strategies for assessing
// the performance of transformer-based learned optimizers across various metrics.

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};

use crate::error::{OptimError, Result};

/// Evaluation strategies for transformer optimizers
#[derive(Debug, Clone, Copy)]
pub enum EvaluationStrategy {
    /// Single-task evaluation
    SingleTask,
    /// Multi-task evaluation
    MultiTask,
    /// Cross-domain evaluation
    CrossDomain,
    /// Few-shot evaluation
    FewShot,
    /// Continual learning evaluation
    ContinualLearning,
    /// Robustness evaluation
    Robustness,
    /// Efficiency evaluation
    Efficiency,
    /// Comprehensive evaluation
    Comprehensive,
}

/// Performance evaluator for transformer optimizers
#[derive(Debug, Clone)]
pub struct TransformerEvaluator<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluation strategy
    strategy: EvaluationStrategy,

    /// Evaluation parameters
    eval_params: EvaluationParams<T>,

    /// Metric calculators
    metric_calculators: HashMap<String, MetricCalculator<T>>,

    /// Performance history
    performance_history: VecDeque<EvaluationResult<T>>,

    /// Baseline comparisons
    baseline_comparisons: HashMap<String, BaselineComparison<T>>,

    /// Robustness scores per task, keyed by task id then by score name.
    ///
    /// [`TransformerEvaluator::evaluate_robustness`] used to compute these and
    /// hand them straight back to the caller, discarding the `task_id` it was
    /// given, so the evaluator itself retained no robustness record and
    /// [`TransformerEvaluator::robustness_results`] had nothing to report.
    robustness_results: HashMap<String, HashMap<String, T>>,
}

/// Evaluation parameters
#[derive(Debug, Clone)]
pub struct EvaluationParams<T: Float + Debug + Send + Sync + 'static> {
    /// Number of evaluation episodes
    pub num_episodes: usize,

    /// Evaluation frequency
    pub eval_frequency: usize,

    /// Convergence tolerance
    pub convergence_tolerance: T,

    /// Maximum evaluation steps
    pub max_eval_steps: usize,

    /// Confidence level for statistical tests
    pub confidence_level: T,

    /// Number of bootstrap samples
    pub bootstrap_samples: usize,

    /// Cross-validation folds
    pub cv_folds: usize,

    /// Robustness test severity (relative noise amplitude)
    pub robustness_severity: T,

    /// Width of the (disjoint) moving-average windows used for convergence detection
    pub convergence_window: usize,

    /// Number of consecutive windows that must be stable before declaring convergence
    pub convergence_consecutive_hits: usize,

    /// Number of leading samples used by the few-shot evaluation variant
    pub few_shot_samples: usize,

    /// Number of noise-perturbed replicas used by the robustness variant
    pub robustness_replicas: usize,

    /// Seed for the deterministic robustness perturbations
    pub robustness_seed: u64,
}

/// Evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluation identifier
    eval_id: String,

    /// Task identifier
    task_id: String,

    /// Performance metrics
    metrics: HashMap<String, T>,

    /// Convergence information
    convergence_info: ConvergenceInfo<T>,

    /// Efficiency metrics
    efficiency_metrics: EfficiencyMetrics<T>,

    /// Statistical significance, present only when a baseline has been registered
    statistical_significance: Option<StatisticalSignificance<T>>,

    /// Evaluation timestamp
    timestamp: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> EvaluationResult<T> {
    /// Identifier of this evaluation
    pub fn eval_id(&self) -> &str {
        &self.eval_id
    }

    /// Task this evaluation belongs to
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Metrics computed for this evaluation
    pub fn metrics(&self) -> &HashMap<String, T> {
        &self.metrics
    }

    /// Convergence information
    pub fn convergence_info(&self) -> &ConvergenceInfo<T> {
        &self.convergence_info
    }

    /// Efficiency metrics
    pub fn efficiency_metrics(&self) -> &EfficiencyMetrics<T> {
        &self.efficiency_metrics
    }

    /// Statistical significance versus the registered baseline, if any
    pub fn statistical_significance(&self) -> Option<&StatisticalSignificance<T>> {
        self.statistical_significance.as_ref()
    }

    /// Evaluation timestamp
    pub fn timestamp(&self) -> usize {
        self.timestamp
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ConvergenceInfo<T> {
    /// Whether convergence was detected
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Step index at which convergence was detected
    pub fn steps_to_convergence(&self) -> Option<usize> {
        self.steps_to_convergence
    }

    /// Final loss value
    pub fn final_loss(&self) -> T {
        self.final_loss
    }

    /// Average relative improvement per step
    pub fn convergence_rate(&self) -> T {
        self.convergence_rate
    }

    /// Recorded loss trajectory
    pub fn loss_trajectory(&self) -> &[T] {
        &self.loss_trajectory
    }

    /// Recorded gradient norms
    pub fn gradient_norms(&self) -> &[T] {
        &self.gradient_norms
    }
}

impl<T: Float + Debug + Send + Sync + 'static> EfficiencyMetrics<T> {
    /// Wall-clock time
    pub fn wall_time(&self) -> T {
        self.wall_time
    }

    /// Estimated floating point operations
    pub fn flops(&self) -> u64 {
        self.flops
    }

    /// Peak memory usage in bytes
    pub fn peak_memory(&self) -> u64 {
        self.peak_memory
    }

    /// Parameter efficiency
    pub fn parameter_efficiency(&self) -> T {
        self.parameter_efficiency
    }

    /// Sample efficiency
    pub fn sample_efficiency(&self) -> T {
        self.sample_efficiency
    }

    /// Energy consumption estimate
    pub fn energy_consumption(&self) -> T {
        self.energy_consumption
    }
}

impl<T: Float + Debug + Send + Sync + 'static> StatisticalSignificance<T> {
    /// Two-sided p-value
    pub fn p_value(&self) -> T {
        self.p_value
    }

    /// Standardized effect size (Cohen's d)
    pub fn effect_size(&self) -> T {
        self.effect_size
    }

    /// Confidence interval of the mean difference
    pub fn confidence_interval(&self) -> (T, T) {
        self.confidence_interval
    }

    /// Statistical power estimate
    pub fn statistical_power(&self) -> T {
        self.statistical_power
    }

    /// Test statistic
    pub fn test_statistic(&self) -> T {
        self.test_statistic
    }
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo<T: Float + Debug + Send + Sync + 'static> {
    /// Whether convergence was achieved
    converged: bool,

    /// Number of steps to convergence
    steps_to_convergence: Option<usize>,

    /// Final loss value
    final_loss: T,

    /// Convergence rate
    convergence_rate: T,

    /// Loss trajectory
    loss_trajectory: Vec<T>,

    /// Gradient norms
    gradient_norms: Vec<T>,
}

/// Efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Wall-clock time
    wall_time: T,

    /// Computational FLOPs
    flops: u64,

    /// Memory usage peak
    peak_memory: u64,

    /// Parameter efficiency
    parameter_efficiency: T,

    /// Sample efficiency
    sample_efficiency: T,

    /// Energy consumption estimate
    energy_consumption: T,
}

/// Statistical significance analysis
#[derive(Debug, Clone)]
pub struct StatisticalSignificance<T: Float + Debug + Send + Sync + 'static> {
    /// P-value for performance comparison
    p_value: T,

    /// Effect size
    effect_size: T,

    /// Confidence interval
    confidence_interval: (T, T),

    /// Statistical power
    statistical_power: T,

    /// Test statistic
    test_statistic: T,
}

/// Metric calculator for specific metrics
#[derive(Debug, Clone)]
pub struct MetricCalculator<T: Float + Debug + Send + Sync + 'static> {
    /// Historical values for trend analysis
    historical_values: VecDeque<T>,

    /// Aggregation method
    aggregation_method: AggregationMethod,
}

/// Baseline comparison data
#[derive(Debug, Clone)]
pub struct BaselineComparison<T: Float + Debug + Send + Sync + 'static> {
    /// Baseline performance
    baseline_performance: HashMap<String, T>,
}

/// Robustness test suite
#[derive(Debug, Clone)]
pub struct RobustnessTestSuite<T: Float + Debug + Send + Sync + 'static> {
    /// Noise injection tests
    pub noise_tests: Vec<NoiseTest<T>>,

    /// Adversarial perturbation tests
    pub adversarial_tests: Vec<AdversarialTest<T>>,

    /// Hyperparameter sensitivity tests
    pub sensitivity_tests: Vec<SensitivityTest<T>>,

    /// Distribution shift tests
    pub distribution_tests: Vec<DistributionTest<T>>,
}

/// Individual robustness tests
#[derive(Debug, Clone)]
pub struct NoiseTest<T: Float + Debug + Send + Sync + 'static> {
    pub noise_type: NoiseType,
    pub noise_level: T,
    pub performance_degradation: T,
}

#[derive(Debug, Clone)]
pub struct AdversarialTest<T: Float + Debug + Send + Sync + 'static> {
    pub attack_type: AttackType,
    pub attack_strength: T,
    pub robustness_score: T,
}

#[derive(Debug, Clone)]
pub struct SensitivityTest<T: Float + Debug + Send + Sync + 'static> {
    pub parameter_name: String,
    pub parameter_range: (T, T),
    pub sensitivity_score: T,
}

#[derive(Debug, Clone)]
pub struct DistributionTest<T: Float + Debug + Send + Sync + 'static> {
    pub shift_type: DistributionShiftType,
    pub shift_magnitude: T,
    pub adaptation_score: T,
}

/// Aggregation methods for metrics
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    Mean,
    Median,
    Max,
    Min,
    WeightedAverage,
    ExponentialMovingAverage,
    Percentile(u8),
}

/// Noise types for robustness testing
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    Gaussian,
    Uniform,
    SaltPepper,
    Dropout,
}

/// Attack types for adversarial testing
#[derive(Debug, Clone, Copy)]
pub enum AttackType {
    FGSM,
    PGD,
    CarliniWagner,
    DeepFool,
}

/// Distribution shift types
#[derive(Debug, Clone, Copy)]
pub enum DistributionShiftType {
    CovariateShift,
    ConceptDrift,
    DatasetShift,
    TemporalShift,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> TransformerEvaluator<T> {
    /// Create new transformer evaluator
    pub fn new(strategy: EvaluationStrategy) -> Result<Self> {
        let mut metric_calculators = HashMap::new();

        // Initialize standard metric calculators
        metric_calculators.insert(
            "convergence_speed".to_string(),
            MetricCalculator::new(AggregationMethod::Mean)?,
        );
        metric_calculators.insert(
            "final_performance".to_string(),
            MetricCalculator::new(AggregationMethod::Mean)?,
        );
        metric_calculators.insert(
            "sample_efficiency".to_string(),
            MetricCalculator::new(AggregationMethod::Mean)?,
        );

        Ok(Self {
            strategy,
            eval_params: EvaluationParams::default(),
            metric_calculators,
            performance_history: VecDeque::new(),
            baseline_comparisons: HashMap::new(),
            robustness_results: HashMap::new(),
        })
    }

    /// Evaluate transformer optimizer performance
    pub fn evaluate(
        &mut self,
        task_id: &str,
        loss_trajectory: &[T],
        gradient_norms: &[T],
        wall_time: T,
        memory_usage: u64,
    ) -> Result<EvaluationResult<T>> {
        let eval_id = format!("eval_{}_{}", task_id, self.performance_history.len());

        // Compute convergence information
        let convergence_info = self.compute_convergence_info(loss_trajectory, gradient_norms)?;

        // Compute efficiency metrics
        let efficiency_metrics =
            self.compute_efficiency_metrics(wall_time, memory_usage, loss_trajectory.len())?;

        // Compute performance metrics
        let mut metrics = HashMap::new();
        metrics.insert("final_loss".to_string(), convergence_info.final_loss);
        metrics.insert(
            "convergence_rate".to_string(),
            convergence_info.convergence_rate,
        );
        metrics.insert(
            "sample_efficiency".to_string(),
            efficiency_metrics.sample_efficiency,
        );

        // Strategy-specific metrics. Each variant contributes a distinct
        // family of measurements; `Comprehensive` runs all of them.
        let strategy = self.strategy;
        let run_all = matches!(strategy, EvaluationStrategy::Comprehensive);

        if run_all || matches!(strategy, EvaluationStrategy::Robustness) {
            for (key, value) in self.robustness_metrics(task_id, loss_trajectory)? {
                metrics.insert(key, value);
            }
        }
        if run_all || matches!(strategy, EvaluationStrategy::FewShot) {
            for (key, value) in self.few_shot_metrics(loss_trajectory)? {
                metrics.insert(key, value);
            }
        }
        if run_all || matches!(strategy, EvaluationStrategy::MultiTask) {
            for (key, value) in self.multi_task_metrics() {
                metrics.insert(key, value);
            }
        }
        if run_all || matches!(strategy, EvaluationStrategy::CrossDomain) {
            for (key, value) in self.cross_domain_metrics(task_id, convergence_info.final_loss) {
                metrics.insert(key, value);
            }
        }
        if run_all || matches!(strategy, EvaluationStrategy::ContinualLearning) {
            for (key, value) in self.continual_learning_metrics(task_id, &convergence_info) {
                metrics.insert(key, value);
            }
        }
        if run_all || matches!(strategy, EvaluationStrategy::Efficiency) {
            for (key, value) in Self::efficiency_extra_metrics(&efficiency_metrics) {
                metrics.insert(key, value);
            }
        }

        // Update metric calculators
        for (metric_name, metric_value) in &metrics {
            if let Some(calculator) = self.metric_calculators.get_mut(metric_name) {
                calculator.update(*metric_value)?;
            }
        }

        // Compute statistical significance if baseline exists
        let statistical_significance = self.compute_statistical_significance(&metrics)?;

        let result = EvaluationResult {
            eval_id,
            task_id: task_id.to_string(),
            metrics,
            convergence_info,
            efficiency_metrics,
            statistical_significance,
            timestamp: self.performance_history.len(),
        };

        self.performance_history.push_back(result.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        Ok(result)
    }

    /// Compute convergence information from trajectories
    fn compute_convergence_info(
        &self,
        loss_trajectory: &[T],
        gradient_norms: &[T],
    ) -> Result<ConvergenceInfo<T>> {
        if loss_trajectory.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Empty loss trajectory".to_string(),
            ));
        }

        let Some(&final_loss) = loss_trajectory.last() else {
            return Err(OptimError::InvalidConfig(
                "Empty loss trajectory".to_string(),
            ));
        };
        let initial_loss = loss_trajectory[0];

        // Detect convergence
        let (converged, steps_to_convergence) = self.detect_convergence(loss_trajectory)?;

        // Compute convergence rate
        let convergence_rate = if loss_trajectory.len() > 1 {
            let improvement = (initial_loss - final_loss)
                / initial_loss
                    .max(scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero()));
            improvement
                / scirs2_core::numeric::NumCast::from(loss_trajectory.len() as f64)
                    .unwrap_or_else(|| T::one())
        } else {
            T::zero()
        };

        Ok(ConvergenceInfo {
            converged,
            steps_to_convergence,
            final_loss,
            convergence_rate,
            loss_trajectory: loss_trajectory.to_vec(),
            gradient_norms: gradient_norms.to_vec(),
        })
    }

    /// Detect convergence from a loss trajectory.
    ///
    /// Two *disjoint* windows of `convergence_window` samples are compared; the
    /// trajectory is declared converged once their relative difference stays
    /// below the tolerance for `convergence_consecutive_hits` consecutive
    /// positions. The scan runs to the end of the trajectory (inclusive), so
    /// the final sample is not ignored, and all index arithmetic is bounded
    /// below by the loop start, so no `usize` underflow is possible.
    fn detect_convergence(&self, loss_trajectory: &[T]) -> Result<(bool, Option<usize>)> {
        let window_size = self.eval_params.convergence_window.max(1);
        let required_hits = self.eval_params.convergence_consecutive_hits.max(1);
        let tolerance = self.eval_params.convergence_tolerance;

        // Two disjoint windows are needed before any comparison is possible.
        if loss_trajectory.len() < 2 * window_size {
            return Ok((false, None));
        }

        let window_len: T =
            scirs2_core::numeric::NumCast::from(window_size as f64).unwrap_or_else(|| T::one());
        let floor: T = scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero());
        let mut consecutive_hits = 0usize;

        for end in (2 * window_size)..=loss_trajectory.len() {
            let current_window = &loss_trajectory[end - window_size..end];
            let prev_window = &loss_trajectory[end - 2 * window_size..end - window_size];

            let current_avg =
                current_window.iter().cloned().fold(T::zero(), |a, b| a + b) / window_len;
            let prev_avg = prev_window.iter().cloned().fold(T::zero(), |a, b| a + b) / window_len;

            let change = (current_avg - prev_avg).abs() / prev_avg.abs().max(floor);

            if change < tolerance {
                consecutive_hits += 1;
                if consecutive_hits >= required_hits {
                    return Ok((true, Some(end)));
                }
            } else {
                consecutive_hits = 0;
            }
        }

        Ok((false, None))
    }

    /// Compute efficiency metrics
    fn compute_efficiency_metrics(
        &self,
        wall_time: T,
        memory_usage: u64,
        num_steps: usize,
    ) -> Result<EfficiencyMetrics<T>> {
        let flops = (num_steps as u64) * 1000; // Simplified FLOP estimation
        let parameter_efficiency = T::one()
            / (scirs2_core::numeric::NumCast::from(memory_usage as f64)
                .unwrap_or_else(|| T::zero())
                + T::one());
        let sample_efficiency = scirs2_core::numeric::NumCast::from(num_steps as f64)
            .unwrap_or_else(|| T::zero())
            / (wall_time + T::one());
        let energy_consumption = wall_time
            * scirs2_core::numeric::NumCast::from(memory_usage as f64).unwrap_or_else(|| T::zero())
            * scirs2_core::numeric::NumCast::from(1e-9).unwrap_or_else(|| T::zero());

        Ok(EfficiencyMetrics {
            wall_time,
            flops,
            peak_memory: memory_usage,
            parameter_efficiency,
            sample_efficiency,
            energy_consumption,
        })
    }

    /// Robustness metrics: re-score the trajectory under deterministic
    /// multiplicative noise and report how much the convergence rate degrades.
    fn robustness_metrics(&self, task_id: &str, loss_trajectory: &[T]) -> Result<Vec<(String, T)>> {
        if loss_trajectory.is_empty() {
            return Ok(Vec::new());
        }

        let severity = self.eval_params.robustness_severity.to_f64().unwrap_or(0.0);
        let replicas = self.eval_params.robustness_replicas.max(1);
        // Seed derived from the task so the perturbations are reproducible.
        let seed = self
            .eval_params
            .robustness_seed
            .wrapping_add(Self::task_seed(task_id));
        let mut rng = scirs2_core::random::Random::seed(seed);

        let clean = self.relative_improvement(loss_trajectory);
        let mut degraded_sum = T::zero();
        let mut converged_count = 0usize;

        for _ in 0..replicas {
            let perturbed: Vec<T> = loss_trajectory
                .iter()
                .map(|&x| {
                    let noise = 1.0 + severity * (rng.gen_range(0.0..1.0) * 2.0 - 1.0);
                    let factor: T =
                        scirs2_core::numeric::NumCast::from(noise).unwrap_or_else(|| T::one());
                    x * factor
                })
                .collect();
            degraded_sum = degraded_sum + self.relative_improvement(&perturbed);
            if self.detect_convergence(&perturbed)?.0 {
                converged_count += 1;
            }
        }

        let replicas_t: T =
            scirs2_core::numeric::NumCast::from(replicas as f64).unwrap_or_else(|| T::one());
        let degraded_mean = degraded_sum / replicas_t;
        // 1 means "noise did not hurt"; smaller means more degradation.
        let robustness_score = if clean.abs() > T::zero() {
            T::one() - (clean - degraded_mean).abs() / clean.abs()
        } else {
            T::one()
        };

        Ok(vec![
            ("robustness_score".to_string(), robustness_score),
            (
                "robustness_perturbed_improvement".to_string(),
                degraded_mean,
            ),
            (
                "robustness_converged_fraction".to_string(),
                scirs2_core::numeric::NumCast::from(converged_count as f64 / replicas as f64)
                    .unwrap_or_else(|| T::zero()),
            ),
        ])
    }

    /// Few-shot metrics: score the optimizer using only the first
    /// `few_shot_samples` steps of the trajectory.
    fn few_shot_metrics(&self, loss_trajectory: &[T]) -> Result<Vec<(String, T)>> {
        if loss_trajectory.is_empty() {
            return Ok(Vec::new());
        }
        let take = self
            .eval_params
            .few_shot_samples
            .max(1)
            .min(loss_trajectory.len());
        let prefix = &loss_trajectory[..take];
        let final_loss = prefix.last().copied().unwrap_or_else(T::zero);

        Ok(vec![
            ("few_shot_samples".to_string(), {
                let value: T =
                    scirs2_core::numeric::NumCast::from(take as f64).unwrap_or_else(|| T::zero());
                value
            }),
            ("few_shot_final_loss".to_string(), final_loss),
            (
                "few_shot_improvement".to_string(),
                self.relative_improvement(prefix),
            ),
        ])
    }

    /// Multi-task metrics: aggregate across every task seen so far.
    fn multi_task_metrics(&self) -> Vec<(String, T)> {
        let mut task_ids: Vec<&str> = self
            .performance_history
            .iter()
            .map(|r| r.task_id.as_str())
            .collect();
        task_ids.sort_unstable();
        task_ids.dedup();

        if self.performance_history.is_empty() {
            return Vec::new();
        }

        let count: T = scirs2_core::numeric::NumCast::from(self.performance_history.len() as f64)
            .unwrap_or_else(|| T::one());
        let mean_final_loss = self
            .performance_history
            .iter()
            .map(|r| r.convergence_info.final_loss)
            .fold(T::zero(), |a, b| a + b)
            / count;

        vec![
            ("multi_task_mean_final_loss".to_string(), mean_final_loss),
            ("multi_task_task_count".to_string(), {
                let value: T = scirs2_core::numeric::NumCast::from(task_ids.len() as f64)
                    .unwrap_or_else(|| T::zero());
                value
            }),
        ]
    }

    /// Cross-domain metrics: spread of final losses across distinct tasks,
    /// including the evaluation currently being computed.
    fn cross_domain_metrics(&self, task_id: &str, current_loss: T) -> Vec<(String, T)> {
        let mut per_task: HashMap<&str, (T, usize)> = HashMap::new();
        for result in &self.performance_history {
            let entry = per_task
                .entry(result.task_id.as_str())
                .or_insert((T::zero(), 0));
            entry.0 = entry.0 + result.convergence_info.final_loss;
            entry.1 += 1;
        }
        let entry = per_task.entry(task_id).or_insert((T::zero(), 0));
        entry.0 = entry.0 + current_loss;
        entry.1 += 1;

        if per_task.len() < 2 {
            return Vec::new();
        }

        let means: Vec<T> = per_task
            .values()
            .map(|(sum, n)| {
                let n_t: T =
                    scirs2_core::numeric::NumCast::from(*n as f64).unwrap_or_else(|| T::one());
                *sum / n_t
            })
            .collect();
        let n_t: T =
            scirs2_core::numeric::NumCast::from(means.len() as f64).unwrap_or_else(|| T::one());
        let mean = means.iter().cloned().fold(T::zero(), |a, b| a + b) / n_t;
        let variance = means
            .iter()
            .map(|&m| (m - mean) * (m - mean))
            .fold(T::zero(), |a, b| a + b)
            / n_t;

        vec![
            ("cross_domain_mean_final_loss".to_string(), mean),
            ("cross_domain_variance".to_string(), variance),
        ]
    }

    /// Continual-learning metrics: forgetting relative to this task's previous
    /// evaluation (positive means the loss got worse).
    fn continual_learning_metrics(
        &self,
        task_id: &str,
        convergence_info: &ConvergenceInfo<T>,
    ) -> Vec<(String, T)> {
        let previous = self
            .performance_history
            .iter()
            .rev()
            .find(|r| r.task_id == task_id);

        match previous {
            Some(previous) => vec![(
                "forgetting".to_string(),
                convergence_info.final_loss - previous.convergence_info.final_loss,
            )],
            None => vec![("forgetting".to_string(), T::zero())],
        }
    }

    /// Efficiency-focused derived metrics.
    fn efficiency_extra_metrics(efficiency: &EfficiencyMetrics<T>) -> Vec<(String, T)> {
        let flops: T = scirs2_core::numeric::NumCast::from(efficiency.flops as f64)
            .unwrap_or_else(|| T::zero());
        let memory: T = scirs2_core::numeric::NumCast::from(efficiency.peak_memory as f64)
            .unwrap_or_else(|| T::zero());
        vec![
            (
                "flops_per_second".to_string(),
                flops / (efficiency.wall_time + T::one()),
            ),
            ("peak_memory_bytes".to_string(), memory),
            (
                "energy_consumption".to_string(),
                efficiency.energy_consumption,
            ),
        ]
    }

    /// Relative improvement of a loss trajectory (first vs last sample).
    fn relative_improvement(&self, loss_trajectory: &[T]) -> T {
        let (Some(&first), Some(&last)) = (loss_trajectory.first(), loss_trajectory.last()) else {
            return T::zero();
        };
        let floor: T = scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero());
        (first - last) / first.abs().max(floor)
    }

    /// Deterministic seed contribution from a task identifier (FNV-1a).
    fn task_seed(task_id: &str) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in task_id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// Standard normal survival function via an Abramowitz-Stegun erfc approximation.
    fn normal_survival(z: f64) -> f64 {
        let x = z.abs() / std::f64::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        let erf = 1.0 - poly * (-x * x).exp();
        let tail = 0.5 * (1.0 - erf);
        if z >= 0.0 {
            tail
        } else {
            1.0 - tail
        }
    }

    /// Compute statistical significance against the registered baselines.
    ///
    /// Returns `None` when no baseline supplies the metric under test, when
    /// there are fewer than two observations, or when the observed sample has
    /// zero spread (in which case no variance can be estimated). No fabricated
    /// p-values are produced.
    fn compute_statistical_significance(
        &self,
        metrics: &HashMap<String, T>,
    ) -> Result<Option<StatisticalSignificance<T>>> {
        let Some(&current) = metrics.get("final_loss") else {
            return Ok(None);
        };

        let baseline_mean = self
            .baseline_comparisons
            .values()
            .filter_map(|b| b.baseline_performance.get("final_loss").copied())
            .fold(None::<(T, usize)>, |acc, value| match acc {
                Some((sum, n)) => Some((sum + value, n + 1)),
                None => Some((value, 1)),
            });
        let Some((baseline_sum, baseline_count)) = baseline_mean else {
            return Ok(None);
        };
        let baseline_count_t: T =
            scirs2_core::numeric::NumCast::from(baseline_count as f64).unwrap_or_else(|| T::one());
        let baseline = baseline_sum / baseline_count_t;

        // Sample statistics of the observed final losses.
        let mut samples: Vec<T> = self
            .performance_history
            .iter()
            .map(|r| r.convergence_info.final_loss)
            .collect();
        samples.push(current);
        if samples.len() < 2 {
            return Ok(None);
        }

        let n = samples.len();
        let n_t: T = scirs2_core::numeric::NumCast::from(n as f64).unwrap_or_else(|| T::one());
        let mean = samples.iter().cloned().fold(T::zero(), |a, b| a + b) / n_t;
        let denominator: T =
            scirs2_core::numeric::NumCast::from((n - 1) as f64).unwrap_or_else(|| T::one());
        let variance = samples
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |a, b| a + b)
            / denominator;
        let std_dev = variance.sqrt();

        if std_dev <= T::zero() {
            return Ok(None);
        }

        let effect_size = (baseline - mean) / std_dev;
        let test_statistic = effect_size * n_t.sqrt();
        let z = test_statistic.to_f64().unwrap_or(0.0);
        let p_value: T = scirs2_core::numeric::NumCast::from(
            (2.0 * Self::normal_survival(z.abs())).clamp(0.0, 1.0),
        )
        .unwrap_or_else(|| T::one());

        // Two-sided normal confidence interval for the mean at the configured level.
        let alpha = 1.0 - self.eval_params.confidence_level.to_f64().unwrap_or(0.95);
        let critical = Self::normal_quantile(1.0 - alpha / 2.0);
        let critical_t: T =
            scirs2_core::numeric::NumCast::from(critical).unwrap_or_else(|| T::one());
        let half_width = critical_t * std_dev / n_t.sqrt();

        // Power of the two-sided test at the observed effect size.
        let power: T = scirs2_core::numeric::NumCast::from(
            (Self::normal_survival(critical - z.abs())).clamp(0.0, 1.0),
        )
        .unwrap_or_else(|| T::zero());

        Ok(Some(StatisticalSignificance {
            p_value,
            effect_size,
            confidence_interval: (mean - half_width, mean + half_width),
            statistical_power: power,
            test_statistic,
        }))
    }

    /// Inverse standard normal CDF (Acklam's rational approximation).
    fn normal_quantile(p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        const A: [f64; 6] = [
            -3.969_683_028_665_376e1,
            2.209_460_984_245_205e2,
            -2.759_285_104_469_687e2,
            1.383_577_518_672_69e2,
            -3.066_479_806_614_716e1,
            2.506_628_277_459_239,
        ];
        const B: [f64; 5] = [
            -5.447_609_879_822_406e1,
            1.615_858_368_580_409e2,
            -1.556_989_798_598_866e2,
            6.680_131_188_771_972e1,
            -1.328_068_155_288_572e1,
        ];
        const C: [f64; 6] = [
            -7.784_894_002_430_293e-3,
            -3.223_964_580_411_365e-1,
            -2.400_758_277_161_838e0,
            -2.549_732_539_343_734e0,
            4.374_664_141_464_968e0,
            2.938_163_982_698_783e0,
        ];
        const D: [f64; 4] = [
            7.784_695_709_041_462e-3,
            3.224_671_290_700_398e-1,
            2.445_134_137_142_996e0,
            3.754_408_661_907_416e0,
        ];
        let p_low = 0.02425;
        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        } else if p <= 1.0 - p_low {
            let q = p - 0.5;
            let r = q * q;
            (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
                / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        }
    }

    /// Add baseline for comparison
    pub fn add_baseline(
        &mut self,
        baseline_name: String,
        baseline_performance: HashMap<String, T>,
    ) -> Result<()> {
        let comparison = BaselineComparison {
            baseline_performance: baseline_performance.clone(),
        };

        self.baseline_comparisons.insert(baseline_name, comparison);
        Ok(())
    }

    /// Run robustness evaluation
    pub fn evaluate_robustness(
        &mut self,
        task_id: &str,
        robustness_tests: &RobustnessTestSuite<T>,
    ) -> Result<HashMap<String, T>> {
        let mut robustness_scores = HashMap::new();

        // Evaluate noise robustness
        let mut noise_score = T::zero();
        for noise_test in &robustness_tests.noise_tests {
            noise_score = noise_score + (T::one() - noise_test.performance_degradation);
        }
        if !robustness_tests.noise_tests.is_empty() {
            noise_score = noise_score
                / scirs2_core::numeric::NumCast::from(robustness_tests.noise_tests.len() as f64)
                    .unwrap_or_else(|| T::one());
        }
        robustness_scores.insert("noise_robustness".to_string(), noise_score);

        // Evaluate adversarial robustness
        let mut adversarial_score = T::zero();
        for adv_test in &robustness_tests.adversarial_tests {
            adversarial_score = adversarial_score + adv_test.robustness_score;
        }
        if !robustness_tests.adversarial_tests.is_empty() {
            adversarial_score = adversarial_score
                / scirs2_core::numeric::NumCast::from(
                    robustness_tests.adversarial_tests.len() as f64
                )
                .unwrap_or_else(|| T::one());
        }
        robustness_scores.insert("adversarial_robustness".to_string(), adversarial_score);

        // Evaluate hyperparameter sensitivity
        let mut sensitivity_score = T::zero();
        for sens_test in &robustness_tests.sensitivity_tests {
            sensitivity_score =
                sensitivity_score + (T::one() / (T::one() + sens_test.sensitivity_score));
        }
        if !robustness_tests.sensitivity_tests.is_empty() {
            sensitivity_score = sensitivity_score
                / scirs2_core::numeric::NumCast::from(
                    robustness_tests.sensitivity_tests.len() as f64
                )
                .unwrap_or_else(|| T::one());
        }
        robustness_scores.insert("hyperparameter_robustness".to_string(), sensitivity_score);

        // Retain the scores under the task they were measured for; a later call
        // for the same task replaces its entry rather than accumulating stale
        // suites.
        self.robustness_results
            .insert(task_id.to_string(), robustness_scores.clone());

        Ok(robustness_scores)
    }

    /// Aggregated value of a tracked metric, under that calculator's
    /// [`AggregationMethod`].
    ///
    /// Every [`Self::evaluate`] call feeds the per-metric calculators, but
    /// nothing could read the aggregate back: `MetricCalculator` was written to
    /// and never queried.
    pub fn aggregated_metric(&self, metric_name: &str) -> Option<Result<T>> {
        self.metric_calculators
            .get(metric_name)
            .map(|c| c.get_aggregated_value())
    }

    /// Names of the metrics this evaluator aggregates.
    pub fn tracked_metrics(&self) -> impl Iterator<Item = &str> {
        self.metric_calculators.keys().map(String::as_str)
    }

    /// Robustness scores recorded for `task_id` by a previous
    /// [`Self::evaluate_robustness`] call, or `None` if that task has not been
    /// evaluated for robustness.
    pub fn robustness_results(&self, task_id: &str) -> Option<&HashMap<String, T>> {
        self.robustness_results.get(task_id)
    }

    /// Task ids that have a recorded robustness evaluation.
    pub fn robustness_evaluated_tasks(&self) -> impl Iterator<Item = &str> {
        self.robustness_results.keys().map(String::as_str)
    }

    /// Get comprehensive evaluation summary
    pub fn get_evaluation_summary(&self) -> HashMap<String, T> {
        let mut summary = HashMap::new();

        // Overall performance statistics
        if !self.performance_history.is_empty() {
            let history_len: T =
                scirs2_core::numeric::NumCast::from(self.performance_history.len() as f64)
                    .unwrap_or_else(|| T::one());
            // Average final performance
            let avg_final_loss = self
                .performance_history
                .iter()
                .map(|result| result.convergence_info.final_loss)
                .fold(T::zero(), |a, b| a + b)
                / history_len;
            summary.insert("average_final_loss".to_string(), avg_final_loss);

            // Average convergence rate
            let avg_convergence_rate = self
                .performance_history
                .iter()
                .map(|result| result.convergence_info.convergence_rate)
                .fold(T::zero(), |a, b| a + b)
                / history_len;
            summary.insert("average_convergence_rate".to_string(), avg_convergence_rate);

            // Success rate (convergence)
            let success_count = self
                .performance_history
                .iter()
                .filter(|result| result.convergence_info.converged)
                .count();
            let success_rate = scirs2_core::numeric::NumCast::from(success_count as f64)
                .unwrap_or_else(|| T::zero())
                / history_len;
            summary.insert("success_rate".to_string(), success_rate);
        }

        summary.insert(
            "total_evaluations".to_string(),
            scirs2_core::numeric::NumCast::from(self.performance_history.len() as f64)
                .unwrap_or_else(|| T::zero()),
        );
        summary
    }

    /// Reset evaluator state
    pub fn reset(&mut self) {
        self.performance_history.clear();
        self.baseline_comparisons.clear();
        self.robustness_results.clear();

        for calculator in self.metric_calculators.values_mut() {
            calculator.reset();
        }
    }

    /// Update evaluation parameters
    pub fn set_parameters(&mut self, params: EvaluationParams<T>) {
        self.eval_params = params;
    }

    /// Get the active evaluation strategy
    pub fn strategy(&self) -> EvaluationStrategy {
        self.strategy
    }

    /// Change the active evaluation strategy
    pub fn set_strategy(&mut self, strategy: EvaluationStrategy) {
        self.strategy = strategy;
    }

    /// Get the recorded evaluation history
    pub fn performance_history(&self) -> &VecDeque<EvaluationResult<T>> {
        &self.performance_history
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> MetricCalculator<T> {
    /// A calculator that aggregates its history with `aggregation_method`.
    ///
    /// The metric's name is the key it is stored under in
    /// `TransformerEvaluator::metric_calculators`; it used to be duplicated into
    /// a `metric_name` field that nothing ever read, so the two could disagree.
    fn new(aggregation_method: AggregationMethod) -> Result<Self> {
        Ok(Self {
            historical_values: VecDeque::new(),
            aggregation_method,
        })
    }

    fn update(&mut self, value: T) -> Result<()> {
        self.historical_values.push_back(value);
        if self.historical_values.len() > 1000 {
            self.historical_values.pop_front();
        }
        Ok(())
    }

    /// Aggregate the recorded history under this calculator's
    /// [`AggregationMethod`].
    pub fn get_aggregated_value(&self) -> Result<T> {
        if self.historical_values.is_empty() {
            return Ok(T::zero());
        }

        match self.aggregation_method {
            AggregationMethod::Mean => {
                let sum = self
                    .historical_values
                    .iter()
                    .cloned()
                    .fold(T::zero(), |a, b| a + b);
                let count: T =
                    scirs2_core::numeric::NumCast::from(self.historical_values.len() as f64)
                        .unwrap_or_else(|| T::one());
                Ok(sum / count)
            }
            AggregationMethod::Max => Ok(self
                .historical_values
                .iter()
                .cloned()
                .fold(T::zero(), |a, b| a.max(b))),
            AggregationMethod::Min => Ok(self.historical_values.iter().cloned().fold(
                scirs2_core::numeric::NumCast::from(f64::INFINITY).unwrap_or_else(|| T::zero()),
                |a, b| a.min(b),
            )),
            _ => Ok(self.historical_values.back().copied().unwrap_or(T::zero())),
        }
    }

    fn reset(&mut self) {
        self.historical_values.clear();
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> Default for EvaluationParams<T> {
    fn default() -> Self {
        Self {
            num_episodes: 10,
            eval_frequency: 100,
            convergence_tolerance: scirs2_core::numeric::NumCast::from(1e-6)
                .unwrap_or_else(|| T::zero()),
            max_eval_steps: 10000,
            confidence_level: scirs2_core::numeric::NumCast::from(0.95)
                .unwrap_or_else(|| T::zero()),
            bootstrap_samples: 1000,
            cv_folds: 5,
            robustness_severity: scirs2_core::numeric::NumCast::from(0.1)
                .unwrap_or_else(|| T::zero()),
            convergence_window: 10,
            convergence_consecutive_hits: 2,
            few_shot_samples: 5,
            robustness_replicas: 8,
            robustness_seed: 0x5EED_C0DE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descending(n: usize) -> Vec<f64> {
        (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect()
    }

    #[test]
    fn short_trajectories_do_not_underflow() {
        let evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::SingleTask)
            .expect("evaluator creation");
        for len in 0..25 {
            let trajectory: Vec<f64> = descending(len);
            let detected = evaluator
                .detect_convergence(&trajectory)
                .expect("convergence detection must not panic");
            if len < 20 {
                assert!(!detected.0, "len {len} cannot be converged yet");
            }
        }
    }

    #[test]
    fn descending_trajectory_is_not_converged() {
        let evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::SingleTask)
            .expect("evaluator creation");
        // Geometric decay by half each step: the moving averages keep changing.
        let trajectory: Vec<f64> = (0..64).map(|i| 0.5_f64.powi(i)).collect();
        let (converged, _) = evaluator
            .detect_convergence(&trajectory)
            .expect("convergence detection");
        assert!(
            !converged,
            "steadily descending trajectory is not converged"
        );
    }

    #[test]
    fn flat_trajectory_is_converged() {
        let evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::SingleTask)
            .expect("evaluator creation");
        let trajectory = vec![1.0_f64; 64];
        let (converged, step) = evaluator
            .detect_convergence(&trajectory)
            .expect("convergence detection");
        assert!(converged);
        assert!(step.is_some());
    }

    #[test]
    fn evaluation_strategies_produce_different_metric_sets() {
        let trajectory = descending(40);
        let grads = vec![0.1_f64; 40];

        let mut single = TransformerEvaluator::<f64>::new(EvaluationStrategy::SingleTask)
            .expect("evaluator creation");
        let single_result = single
            .evaluate("task", &trajectory, &grads, 1.0, 1024)
            .expect("single-task evaluation");

        let mut robust = TransformerEvaluator::<f64>::new(EvaluationStrategy::Robustness)
            .expect("evaluator creation");
        let robust_result = robust
            .evaluate("task", &trajectory, &grads, 1.0, 1024)
            .expect("robustness evaluation");

        let mut few_shot = TransformerEvaluator::<f64>::new(EvaluationStrategy::FewShot)
            .expect("evaluator creation");
        let few_shot_result = few_shot
            .evaluate("task", &trajectory, &grads, 1.0, 1024)
            .expect("few-shot evaluation");

        assert!(!single_result.metrics().contains_key("robustness_score"));
        assert!(robust_result.metrics().contains_key("robustness_score"));
        assert!(few_shot_result
            .metrics()
            .contains_key("few_shot_final_loss"));
        assert!(!few_shot_result.metrics().contains_key("robustness_score"));
    }

    #[test]
    fn comprehensive_strategy_covers_every_family() {
        let trajectory = descending(40);
        let grads = vec![0.1_f64; 40];
        let mut evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::Comprehensive)
            .expect("evaluator creation");
        let _ = evaluator
            .evaluate("task_a", &trajectory, &grads, 1.0, 1024)
            .expect("evaluation a");
        let result = evaluator
            .evaluate("task_b", &trajectory, &grads, 1.0, 1024)
            .expect("evaluation b");

        for key in [
            "robustness_score",
            "few_shot_final_loss",
            "multi_task_mean_final_loss",
            "cross_domain_variance",
            "forgetting",
            "flops_per_second",
        ] {
            assert!(result.metrics().contains_key(key), "missing metric {key}");
        }
    }

    #[test]
    fn robustness_metrics_are_deterministic() {
        let trajectory = descending(40);
        let grads = vec![0.1_f64; 40];
        let run = || {
            let mut evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::Robustness)
                .expect("evaluator creation");
            evaluator
                .evaluate("task", &trajectory, &grads, 1.0, 1024)
                .expect("evaluation")
                .metrics()["robustness_score"]
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn statistical_significance_requires_a_baseline() {
        let trajectory = descending(40);
        let grads = vec![0.1_f64; 40];
        let mut evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::SingleTask)
            .expect("evaluator creation");

        let result = evaluator
            .evaluate("task", &trajectory, &grads, 1.0, 1024)
            .expect("evaluation");
        assert!(
            result.statistical_significance().is_none(),
            "no baseline means no p-value"
        );

        let mut baseline = HashMap::new();
        baseline.insert("final_loss".to_string(), 0.5_f64);
        evaluator
            .add_baseline("adam".to_string(), baseline)
            .expect("baseline registration");

        // A second, different trajectory gives the sample a non-zero spread.
        let other: Vec<f64> = descending(40).iter().map(|v| v * 0.5).collect();
        let result = evaluator
            .evaluate("task", &other, &grads, 1.0, 1024)
            .expect("evaluation");
        let significance = result
            .statistical_significance()
            .expect("baseline registered and variance estimable");
        assert!((0.0..=1.0).contains(&significance.p_value()));
        assert!(significance.effect_size().is_finite());
        let (low, high) = significance.confidence_interval();
        assert!(low <= high);
    }

    #[test]
    fn degenerate_samples_yield_no_significance() {
        let trajectory = descending(40);
        let grads = vec![0.1_f64; 40];
        let mut evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::SingleTask)
            .expect("evaluator creation");
        let mut baseline = HashMap::new();
        baseline.insert("final_loss".to_string(), 0.5_f64);
        evaluator
            .add_baseline("adam".to_string(), baseline)
            .expect("baseline registration");

        let _ = evaluator
            .evaluate("task", &trajectory, &grads, 1.0, 1024)
            .expect("evaluation");
        // Identical repeat: zero spread, so no variance can be estimated.
        let result = evaluator
            .evaluate("task", &trajectory, &grads, 1.0, 1024)
            .expect("evaluation");
        assert!(result.statistical_significance().is_none());
    }

    #[test]
    fn empty_trajectory_is_rejected() {
        let mut evaluator = TransformerEvaluator::<f64>::new(EvaluationStrategy::SingleTask)
            .expect("evaluator creation");
        let result = evaluator.evaluate("task", &[], &[], 1.0, 0);
        assert!(result.is_err());
    }
}
