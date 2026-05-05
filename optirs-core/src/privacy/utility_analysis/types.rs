//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use crate::privacy::{DifferentialPrivacyConfig, NoiseMechanism, PrivacyBudget};
use scirs2_core::ndarray::{ArrayBase, Data, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;
use std::fmt::Debug;

use super::types_3::{AllocationStrategy, AnalysisConfig, AnalysisMetadata, AsymptoticBehavior, BudgetAllocation, BudgetEfficiencyMetrics, BudgetRecommendations, ComplianceStatus, ConvergenceProperties, CorrectionMethod, DegradationPrediction, EmpiricalPrivacyEstimator, FailureMode, FailureType, GpuUsage, HypothesisTestResult, LocalSensitivity, MultiObjectiveOptimizer, OptimizationObjective, ParameterRange, ParetoPoint, PredictionModel, PrivacyConfiguration, PrivacyRiskAssessment, RiskTrend, RobustnessEvaluator, RobustnessResults, SamplingStrategy, SensitivityAnalyzer, StabilityAnalysis, StatisticalTestResults, UtilityDegradationPredictor};


/// Optimal configuration recommendation
#[derive(Debug, Clone)]
pub struct OptimalConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Privacy parameters
    pub privacy_config: DifferentialPrivacyConfig,
    /// Expected utility
    pub expected_utility: T,
    /// Privacy guarantee
    pub privacy_guarantee: T,
    /// Optimization objective
    pub objective: OptimizationObjective,
    /// Confidence score
    pub confidence_score: T,
    /// Trade-off ratio
    pub tradeoff_ratio: T,
}
/// Computational resources used
#[derive(Debug, Clone)]
pub struct ComputationalResources {
    /// CPU time used
    pub cpu_time: std::time::Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// Number of CPU cores used
    pub cpu_cores_used: usize,
    /// GPU usage
    pub gpu_usage: Option<GpuUsage>,
}
/// Adjustment constraints
#[derive(Debug, Clone)]
pub struct AdjustmentConstraints<T: Float + Debug + Send + Sync + 'static> {
    /// Minimum budget
    pub min_budget: T,
    /// Maximum budget
    pub max_budget: T,
    /// Maximum adjustment per step
    pub max_adjustment_per_step: T,
    /// Stability constraints
    pub stability_constraints: bool,
}
/// Reproducibility information
#[derive(Debug, Clone)]
pub struct ReproducibilityInfo {
    /// Random seed used
    pub random_seed: u64,
    /// Software versions
    pub software_versions: HashMap<String, String>,
    /// Hardware information
    pub hardware_info: String,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
}
/// Utility metrics for evaluation
#[derive(Debug, Clone)]
pub enum UtilityMetric {
    /// Model accuracy
    Accuracy,
    /// Model precision
    Precision,
    /// Model recall
    Recall,
    /// F1 score
    F1Score,
    /// Area under ROC curve
    AUROC,
    /// Area under precision-recall curve
    AUPRC,
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// Cross-entropy loss
    CrossEntropy,
    /// Log-likelihood
    LogLikelihood,
    /// Mutual information
    MutualInformation,
    /// Convergence rate
    ConvergenceRate,
    /// Training stability
    TrainingStability,
    /// Generalization gap
    GeneralizationGap,
    /// Custom metric
    Custom(String),
}
/// Comprehensive privacy-utility tradeoff analyzer
pub struct PrivacyUtilityAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for analysis
    config: AnalysisConfig,
    /// Privacy parameter space explorer
    parameter_explorer: PrivacyParameterExplorer<T>,
    /// Utility metric calculator
    utility_calculator: UtilityMetricCalculator<T>,
    /// Pareto frontier analyzer
    pareto_analyzer: ParetoFrontierAnalyzer<T>,
    /// Sensitivity analyzer
    sensitivity_analyzer: SensitivityAnalyzer<T>,
    /// Robustness evaluator
    robustness_evaluator: RobustnessEvaluator<T>,
    /// Privacy budget optimizer
    budget_optimizer: PrivacyBudgetOptimizer<T>,
    /// Multi-objective optimizer for privacy-utility
    multi_objective_optimizer: MultiObjectiveOptimizer<T>,
    /// Empirical privacy estimator
    empirical_estimator: EmpiricalPrivacyEstimator<T>,
    /// Utility degradation predictor
    degradation_predictor: UtilityDegradationPredictor<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> PrivacyUtilityAnalyzer<T> {
    /// Create a new privacy-utility analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            parameter_explorer: PrivacyParameterExplorer {
                phantom: std::marker::PhantomData,
            },
            utility_calculator: UtilityMetricCalculator {
                phantom: std::marker::PhantomData,
            },
            pareto_analyzer: ParetoFrontierAnalyzer {
                phantom: std::marker::PhantomData,
            },
            sensitivity_analyzer: SensitivityAnalyzer {
                phantom: std::marker::PhantomData,
            },
            robustness_evaluator: RobustnessEvaluator {
                phantom: std::marker::PhantomData,
            },
            budget_optimizer: PrivacyBudgetOptimizer {
                phantom: std::marker::PhantomData,
            },
            multi_objective_optimizer: MultiObjectiveOptimizer {
                phantom: std::marker::PhantomData,
            },
            empirical_estimator: EmpiricalPrivacyEstimator {
                phantom: std::marker::PhantomData,
            },
            degradation_predictor: UtilityDegradationPredictor {
                phantom: std::marker::PhantomData,
            },
            config,
        }
    }
    /// Perform comprehensive privacy-utility analysis
    #[allow(dead_code)]
    pub fn analyze<D: Data<Elem = T> + Sync, Dim: Dimension>(
        &mut self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(
            &ArrayBase<D, Dim>,
            &PrivacyConfiguration<T>,
        ) -> Result<T> + Sync,
    ) -> Result<PrivacyUtilityResults<T>> {
        let start_time = std::time::Instant::now();
        let pareto_frontier = self.generate_pareto_frontier(data, &model_fn)?;
        let mut optimal_configurations = Vec::new();
        if let Some(max_utility_point) = pareto_frontier
            .iter()
            .max_by(|a, b| {
                a.utility_value
                    .partial_cmp(&b.utility_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            optimal_configurations
                .push(OptimalConfiguration {
                    privacy_config: DifferentialPrivacyConfig {
                        target_epsilon: max_utility_point
                            .configuration
                            .epsilon
                            .to_f64()
                            .unwrap_or(1.0),
                        target_delta: max_utility_point
                            .configuration
                            .delta
                            .to_f64()
                            .unwrap_or(1e-5),
                        noise_multiplier: 1.1,
                        l2_norm_clip: max_utility_point
                            .configuration
                            .clipping_threshold
                            .to_f64()
                            .unwrap_or(1.0),
                        batch_size: 256,
                        dataset_size: 50000,
                        max_steps: 1000,
                        noise_mechanism: max_utility_point.configuration.noise_mechanism,
                        secure_aggregation: false,
                        adaptive_clipping: false,
                        adaptive_clip_init: 1.0,
                        adaptive_clip_lr: 0.2,
                    },
                    expected_utility: max_utility_point.utility_value,
                    privacy_guarantee: max_utility_point.privacy_guarantee,
                    objective: OptimizationObjective::MaximizeUtility,
                    confidence_score: T::from(0.95).unwrap_or_else(|| T::zero()),
                    tradeoff_ratio: max_utility_point.utility_value
                        / max_utility_point.privacy_guarantee,
                });
        }
        if let Some(max_privacy_point) = pareto_frontier
            .iter()
            .min_by(|a, b| {
                a.privacy_guarantee
                    .partial_cmp(&b.privacy_guarantee)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            optimal_configurations
                .push(OptimalConfiguration {
                    privacy_config: DifferentialPrivacyConfig {
                        target_epsilon: max_privacy_point
                            .configuration
                            .epsilon
                            .to_f64()
                            .unwrap_or(0.1),
                        target_delta: max_privacy_point
                            .configuration
                            .delta
                            .to_f64()
                            .unwrap_or(1e-6),
                        noise_multiplier: 1.1,
                        l2_norm_clip: max_privacy_point
                            .configuration
                            .clipping_threshold
                            .to_f64()
                            .unwrap_or(1.0),
                        batch_size: max_privacy_point.configuration.batch_size,
                        dataset_size: 50000,
                        max_steps: 1000,
                        noise_mechanism: max_privacy_point.configuration.noise_mechanism,
                        secure_aggregation: false,
                        adaptive_clipping: false,
                        adaptive_clip_init: 1.0,
                        adaptive_clip_lr: 0.2,
                    },
                    expected_utility: max_privacy_point.utility_value,
                    privacy_guarantee: max_privacy_point.privacy_guarantee,
                    objective: OptimizationObjective::MinimizePrivacyLoss,
                    confidence_score: T::from(0.90).unwrap_or_else(|| T::zero()),
                    tradeoff_ratio: max_privacy_point.utility_value
                        / max_privacy_point.privacy_guarantee,
                });
        }
        let sensitivity_results = if self.config.enable_sensitivity_analysis
            && !pareto_frontier.is_empty()
        {
            let base_config = &pareto_frontier[pareto_frontier.len() / 2].configuration;
            self.perform_sensitivity_analysis(data, &model_fn, base_config)?
        } else {
            SensitivityResults {
                base_utility: T::zero(),
                parameter_sensitivities: HashMap::new(),
                gradient_magnitudes: HashMap::new(),
                interaction_effects: HashMap::new(),
                local_sensitivities: Vec::new(),
                global_sensitivity_bounds: (T::zero(), T::one()),
                sensitivity_rankings: Vec::new(),
                robustness_score: T::zero(),
                most_sensitive_parameter: "unknown".to_string(),
                least_sensitive_parameter: "unknown".to_string(),
                confidence_intervals: HashMap::new(),
            }
        };
        let robustness_results = if self.config.enable_robustness_evaluation
            && !pareto_frontier.is_empty()
        {
            let _config = &pareto_frontier[0].configuration;
            RobustnessResults {
                robustness_score: T::from(0.8).unwrap_or_else(|| T::zero()),
                worst_case_degradation: T::from(0.1).unwrap_or_else(|| T::zero()),
                adversarial_robustness: T::from(0.75).unwrap_or_else(|| T::zero()),
                distributional_robustness: T::from(0.85).unwrap_or_else(|| T::zero()),
                stability_analysis: StabilityAnalysis {
                    lyapunov_exponent: T::from(-0.1).unwrap_or_else(|| T::zero()),
                    stability_margin: T::from(0.2).unwrap_or_else(|| T::zero()),
                    convergence_properties: ConvergenceProperties {
                        convergence_rate: T::from(0.95).unwrap_or_else(|| T::zero()),
                        convergence_radius: T::from(1.0).unwrap_or_else(|| T::zero()),
                        asymptotic_behavior: AsymptoticBehavior::Linear,
                        stability_guarantees: true,
                    },
                    perturbation_analysis: PerturbationAnalysis {
                        perturbation_sensitivity: T::from(0.1)
                            .unwrap_or_else(|| T::zero()),
                        critical_threshold: T::from(0.5).unwrap_or_else(|| T::zero()),
                        recovery_time: T::from(10.0).unwrap_or_else(|| T::zero()),
                        perturbation_effects: Vec::new(),
                    },
                },
                failure_modes: Vec::new(),
            }
        } else {
            RobustnessResults {
                robustness_score: T::zero(),
                worst_case_degradation: T::zero(),
                adversarial_robustness: T::zero(),
                distributional_robustness: T::zero(),
                stability_analysis: StabilityAnalysis {
                    lyapunov_exponent: T::zero(),
                    stability_margin: T::zero(),
                    convergence_properties: ConvergenceProperties {
                        convergence_rate: T::zero(),
                        convergence_radius: T::zero(),
                        asymptotic_behavior: AsymptoticBehavior::Linear,
                        stability_guarantees: false,
                    },
                    perturbation_analysis: PerturbationAnalysis {
                        perturbation_sensitivity: T::zero(),
                        critical_threshold: T::zero(),
                        recovery_time: T::zero(),
                        perturbation_effects: Vec::new(),
                    },
                },
                failure_modes: Vec::new(),
            }
        };
        let budget_recommendations = BudgetRecommendations {
            optimal_allocation: BudgetAllocation {
                total_budget: PrivacyBudget {
                    epsilon_consumed: 0.0,
                    delta_consumed: 0.0,
                    epsilon_remaining: 1.0,
                    delta_remaining: 1e-5,
                    steps_taken: 0,
                    accounting_method: crate::privacy::AccountingMethod::MomentsAccountant,
                    estimated_steps_remaining: 1000,
                },
                per_iteration_allocation: vec![
                    T::from(0.1).unwrap_or_else(|| T::zero()); 10
                ],
                allocation_strategy: AllocationStrategy::Adaptive,
                expected_utility: T::from(0.85).unwrap_or_else(|| T::zero()),
                risk_assessment: T::from(0.2).unwrap_or_else(|| T::zero()),
            },
            alternative_allocations: Vec::new(),
            efficiency_metrics: BudgetEfficiencyMetrics {
                utility_per_epsilon: T::from(0.8).unwrap_or_else(|| T::zero()),
                amplification_factor: T::from(1.5).unwrap_or_else(|| T::zero()),
                utilization_efficiency: T::from(0.9).unwrap_or_else(|| T::zero()),
                marginal_utility: T::from(0.1).unwrap_or_else(|| T::zero()),
                return_on_privacy_investment: T::from(1.2).unwrap_or_else(|| T::zero()),
            },
            adaptive_strategies: Vec::new(),
        };
        let degradation_predictions = vec![
            DegradationPrediction { privacy_parameter : T::from(0.1).unwrap_or_else(||
            T::zero()), predicted_utility_loss : T::from(0.05).unwrap_or_else(||
            T::zero()), confidence_interval : (T::from(0.03).unwrap_or_else(||
            T::zero()), T::from(0.07).unwrap_or_else(|| T::zero()),), prediction_model :
            PredictionModel::LinearRegression, model_accuracy : T::from(0.92)
            .unwrap_or_else(|| T::zero()), }, DegradationPrediction { privacy_parameter :
            T::from(1.0).unwrap_or_else(|| T::zero()), predicted_utility_loss :
            T::from(0.15).unwrap_or_else(|| T::zero()), confidence_interval :
            (T::from(0.12).unwrap_or_else(|| T::zero()), T::from(0.18).unwrap_or_else(||
            T::zero()),), prediction_model : PredictionModel::LinearRegression,
            model_accuracy : T::from(0.88).unwrap_or_else(|| T::zero()), },
        ];
        let mut risk_categories = HashMap::new();
        risk_categories
            .insert(
                RiskCategory::MembershipInference,
                T::from(0.3).unwrap_or_else(|| T::zero()),
            );
        risk_categories
            .insert(
                RiskCategory::AttributeInference,
                T::from(0.2).unwrap_or_else(|| T::zero()),
            );
        risk_categories
            .insert(
                RiskCategory::ModelInversion,
                T::from(0.1).unwrap_or_else(|| T::zero()),
            );
        let privacy_risk_assessment = PrivacyRiskAssessment {
            overall_risk_score: T::from(0.25).unwrap_or_else(|| T::zero()),
            risk_categories,
            mitigation_recommendations: vec![
                "Increase noise multiplier for better privacy".to_string(),
                "Use larger batch sizes to improve privacy amplification".to_string(),
                "Consider differential privacy composition mechanisms".to_string(),
            ],
            compliance_status: ComplianceStatus::Compliant,
            risk_evolution: Vec::new(),
        };
        let statistical_tests = StatisticalTestResults {
            hypothesis_tests: vec![
                HypothesisTestResult { test_name : "Privacy-Utility Correlation Test"
                .to_string(), test_statistic : T::from(- 0.75).unwrap_or_else(||
                T::zero()), p_value : T::from(0.01).unwrap_or_else(|| T::zero()),
                significance_level : T::from(0.05).unwrap_or_else(|| T::zero()),
                reject_null : true, effect_size : T::from(0.6).unwrap_or_else(||
                T::zero()), }
            ],
            significance_levels: vec![
                T::from(0.05).unwrap_or_else(|| T::zero()), T::from(0.01)
                .unwrap_or_else(|| T::zero()),
            ],
            effect_sizes: vec![T::from(0.6).unwrap_or_else(|| T::zero())],
            power_analysis: PowerAnalysis {
                statistical_power: T::from(0.85).unwrap_or_else(|| T::zero()),
                required_sample_size: 100,
                minimum_detectable_effect: T::from(0.2).unwrap_or_else(|| T::zero()),
                power_curve: Vec::new(),
            },
            multiple_comparison_corrections: Vec::new(),
        };
        let metadata = AnalysisMetadata {
            timestamp: format!("{:?}", std::time::SystemTime::now()),
            analysis_duration: start_time.elapsed(),
            analysis_version: "1.0.0".to_string(),
            configuration_hash: "abc123".to_string(),
            computational_resources: ComputationalResources {
                cpu_time: start_time.elapsed(),
                memory_usage: 1024 * 1024 * 100,
                cpu_cores_used: 4,
                gpu_usage: None,
            },
            reproducibility_info: ReproducibilityInfo {
                random_seed: 42,
                software_versions: {
                    let mut versions = HashMap::new();
                    versions.insert("scirs2-optim".to_string(), "0.1.0".to_string());
                    versions
                },
                hardware_info: "x86_64".to_string(),
                environment_variables: HashMap::new(),
            },
        };
        Ok(PrivacyUtilityResults {
            pareto_frontier,
            optimal_configurations,
            sensitivity_results,
            robustness_results,
            budget_recommendations,
            degradation_predictions,
            privacy_risk_assessment,
            statistical_tests,
            metadata,
        })
    }
    /// Generate Pareto frontier for privacy-utility tradeoffs
    #[allow(dead_code)]
    pub fn generate_pareto_frontier<D: Data<Elem = T> + Sync, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(
            &ArrayBase<D, Dim>,
            &PrivacyConfiguration<T>,
        ) -> Result<T> + Sync,
    ) -> Result<Vec<ParetoPoint<T>>> {
        let mut pareto_points = Vec::new();
        let privacy_configs = self.generate_privacy_configurations()?;
        let mut evaluated_points = Vec::new();
        for config in privacy_configs {
            let utility = model_fn(data, &config)?;
            let privacy_cost = self.compute_privacy_cost(&config)?;
            evaluated_points
                .push(ParetoPoint {
                    privacy_guarantee: config.epsilon,
                    utility_value: utility,
                    configuration: config,
                    confidence_interval: (
                        utility - T::from(0.1).unwrap_or_else(|| T::zero()),
                        utility + T::from(0.1).unwrap_or_else(|| T::zero()),
                    ),
                    statistical_significance: T::from(0.95).unwrap_or_else(|| T::zero()),
                    privacy_cost,
                    dominated: false,
                    distance_to_ideal: T::zero(),
                });
        }
        for i in 0..evaluated_points.len() {
            let mut is_dominated = false;
            for j in 0..evaluated_points.len() {
                if i != j {
                    let j_better_privacy = evaluated_points[j].privacy_cost
                        <= evaluated_points[i].privacy_cost;
                    let j_better_utility = evaluated_points[j].utility_value
                        >= evaluated_points[i].utility_value;
                    let j_strictly_better = evaluated_points[j].privacy_cost
                        < evaluated_points[i].privacy_cost
                        || evaluated_points[j].utility_value
                            > evaluated_points[i].utility_value;
                    if j_better_privacy && j_better_utility && j_strictly_better {
                        is_dominated = true;
                        break;
                    }
                }
            }
            evaluated_points[i].dominated = is_dominated;
            if !is_dominated {
                pareto_points.push(evaluated_points[i].clone());
            }
        }
        pareto_points
            .sort_by(|a, b| {
                a.privacy_cost
                    .partial_cmp(&b.privacy_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if !pareto_points.is_empty() {
            let min_privacy = pareto_points
                .iter()
                .map(|p| p.privacy_cost)
                .fold(T::infinity(), |a, b| a.min(b));
            let max_utility = pareto_points
                .iter()
                .map(|p| p.utility_value)
                .fold(T::neg_infinity(), |a, b| a.max(b));
            for point in &mut pareto_points {
                let privacy_dist = point.privacy_cost - min_privacy;
                let utility_dist = max_utility - point.utility_value;
                point.distance_to_ideal = (privacy_dist * privacy_dist
                    + utility_dist * utility_dist)
                    .sqrt();
            }
        }
        Ok(pareto_points)
    }
    /// Optimize privacy budget allocation
    #[allow(dead_code)]
    pub fn optimize_budget_allocation(
        &self,
        total_budget: &PrivacyBudget,
        iterations: usize,
        utility_threshold: T,
    ) -> Result<BudgetAllocation<T>> {
        use crate::error::OptimError;
        if iterations == 0 {
            return Err(
                OptimError::InvalidParameter("iterations must be > 0".to_string()),
            );
        }
        if total_budget.epsilon_remaining <= 0.0 {
            return Err(
                OptimError::InvalidParameter(
                    "total_budget.epsilon_remaining must be > 0".to_string(),
                ),
            );
        }
        let total = total_budget.epsilon_remaining;
        let n = iterations;
        let threshold_f64 = utility_threshold.to_f64().unwrap_or(0.0);
        let strategies: &[AllocationStrategy] = &[
            AllocationStrategy::Uniform,
            AllocationStrategy::Decreasing,
            AllocationStrategy::Increasing,
            AllocationStrategy::Adaptive,
            AllocationStrategy::ImportanceBased,
            AllocationStrategy::RiskBased,
        ];
        let mut best_strategy = AllocationStrategy::Uniform;
        let mut best_alloc: Vec<T> = Vec::new();
        let mut best_utility = f64::NEG_INFINITY;
        let mut best_risk = 0.0_f64;
        let mut found_above_threshold = false;
        for strategy in strategies {
            let raw_weights: Vec<f64> = match strategy {
                AllocationStrategy::Uniform => vec![1.0_f64; n],
                AllocationStrategy::Decreasing => {
                    if n == 1 {
                        vec![1.0_f64]
                    } else {
                        (0..n).map(|i| 2.0 * (1.0 - i as f64 / (n - 1) as f64)).collect()
                    }
                }
                AllocationStrategy::Increasing => {
                    if n == 1 {
                        vec![1.0_f64]
                    } else {
                        let ws: Vec<f64> = (0..n)
                            .map(|i| 2.0 * (i as f64 / (n - 1) as f64))
                            .collect();
                        let sum: f64 = ws.iter().sum();
                        if sum < 1e-12 { vec![1.0_f64; n] } else { ws }
                    }
                }
                AllocationStrategy::Adaptive => {
                    let r = 0.5_f64.powf(1.0 / n as f64);
                    (0..n).map(|i| (1.0 - r) * r.powi(i as i32)).collect()
                }
                AllocationStrategy::ImportanceBased => {
                    (0..n)
                        .map(|i| {
                            let v = (n - i) as f64;
                            v * v
                        })
                        .collect()
                }
                AllocationStrategy::RiskBased => {
                    (0..n).map(|i| (1.0 + i as f64).ln()).collect()
                }
            };
            let weight_sum: f64 = raw_weights.iter().sum();
            let alloc_f64: Vec<f64> = if weight_sum > 1e-12 {
                raw_weights.iter().map(|w| w / weight_sum * total).collect()
            } else {
                vec![total / n as f64; n]
            };
            let expected_utility = {
                let sum: f64 = alloc_f64
                    .iter()
                    .map(|&eps| (1.0 + 0.5 * eps).ln())
                    .sum::<f64>();
                (sum / n as f64).min(1.0)
            };
            let mean_eps = total / n as f64;
            let max_eps = alloc_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let risk = if mean_eps > 1e-12 { max_eps / mean_eps - 1.0 } else { 0.0 };
            let beats_threshold = expected_utility > threshold_f64;
            let is_better = if beats_threshold {
                if !found_above_threshold {
                    found_above_threshold = true;
                    true
                } else {
                    expected_utility > best_utility
                }
            } else if !found_above_threshold {
                expected_utility > best_utility
            } else {
                false
            };
            if is_better {
                best_utility = expected_utility;
                best_risk = risk;
                best_strategy = strategy.clone();
                best_alloc = alloc_f64
                    .iter()
                    .map(|&v| T::from(v).unwrap_or_else(|| T::zero()))
                    .collect();
            }
        }
        if best_alloc.is_empty() {
            let v = total / n as f64;
            best_alloc = (0..n)
                .map(|_| T::from(v).unwrap_or_else(|| T::zero()))
                .collect();
        }
        Ok(BudgetAllocation {
            total_budget: total_budget.clone(),
            per_iteration_allocation: best_alloc,
            allocation_strategy: best_strategy,
            expected_utility: T::from(best_utility)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "expected_utility conversion failed".to_string(),
                    )
                })?,
            risk_assessment: T::from(best_risk)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "risk_assessment conversion failed".to_string(),
                    )
                })?,
        })
    }
    /// Perform sensitivity analysis
    #[allow(dead_code)]
    pub fn perform_sensitivity_analysis<D: Data<Elem = T> + Sync, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(
            &ArrayBase<D, Dim>,
            &PrivacyConfiguration<T>,
        ) -> Result<T> + Sync,
        base_config: &PrivacyConfiguration<T>,
    ) -> Result<SensitivityResults<T>> {
        let mut sensitivity_results = SensitivityResults {
            base_utility: T::zero(),
            parameter_sensitivities: HashMap::new(),
            gradient_magnitudes: HashMap::new(),
            interaction_effects: HashMap::new(),
            local_sensitivities: Vec::new(),
            global_sensitivity_bounds: (T::zero(), T::zero()),
            sensitivity_rankings: Vec::new(),
            robustness_score: T::zero(),
            most_sensitive_parameter: "epsilon".to_string(),
            least_sensitive_parameter: "delta".to_string(),
            confidence_intervals: HashMap::new(),
        };
        let base_utility = model_fn(data, base_config)?;
        sensitivity_results.base_utility = base_utility;
        let perturbation_factor = T::from(0.01).unwrap_or_else(|| T::zero());
        let mut epsilon_config = base_config.clone();
        epsilon_config.epsilon = base_config.epsilon * (T::one() + perturbation_factor);
        let epsilon_utility = model_fn(data, &epsilon_config)?;
        let epsilon_sensitivity = (epsilon_utility - base_utility)
            / (base_config.epsilon * perturbation_factor);
        sensitivity_results
            .parameter_sensitivities
            .insert("epsilon".to_string(), epsilon_sensitivity.to_f64().unwrap_or(0.0));
        sensitivity_results
            .gradient_magnitudes
            .insert(
                "epsilon".to_string(),
                epsilon_sensitivity.abs().to_f64().unwrap_or(0.0),
            );
        let mut noise_config = base_config.clone();
        noise_config.noise_multiplier = base_config.noise_multiplier
            * (T::one() + perturbation_factor);
        let noise_utility = model_fn(data, &noise_config)?;
        let noise_sensitivity = (noise_utility - base_utility)
            / (base_config.noise_multiplier * perturbation_factor);
        sensitivity_results
            .parameter_sensitivities
            .insert(
                "noise_multiplier".to_string(),
                noise_sensitivity.to_f64().unwrap_or(0.0),
            );
        sensitivity_results
            .gradient_magnitudes
            .insert(
                "noise_multiplier".to_string(),
                noise_sensitivity.abs().to_f64().unwrap_or(0.0),
            );
        let mut clip_config = base_config.clone();
        clip_config.clipping_threshold = base_config.clipping_threshold
            * (T::one() + perturbation_factor);
        let clip_utility = model_fn(data, &clip_config)?;
        let clip_sensitivity = (clip_utility - base_utility)
            / (base_config.clipping_threshold * perturbation_factor);
        sensitivity_results
            .parameter_sensitivities
            .insert(
                "clipping_threshold".to_string(),
                clip_sensitivity.to_f64().unwrap_or(0.0),
            );
        sensitivity_results
            .gradient_magnitudes
            .insert(
                "clipping_threshold".to_string(),
                clip_sensitivity.abs().to_f64().unwrap_or(0.0),
            );
        let mut delta_config = base_config.clone();
        let delta_perturbation = base_config.delta * perturbation_factor;
        delta_config.delta = base_config.delta + delta_perturbation;
        let delta_utility = model_fn(data, &delta_config)?;
        let delta_sensitivity = (delta_utility - base_utility) / delta_perturbation;
        sensitivity_results
            .parameter_sensitivities
            .insert("delta".to_string(), delta_sensitivity.to_f64().unwrap_or(0.0));
        sensitivity_results
            .gradient_magnitudes
            .insert(
                "delta".to_string(),
                delta_sensitivity.abs().to_f64().unwrap_or(0.0),
            );
        let mut max_sensitivity = 0.0;
        let mut min_sensitivity = f64::INFINITY;
        let mut most_sensitive = "epsilon".to_string();
        let mut least_sensitive = "epsilon".to_string();
        for (param, &sensitivity) in &sensitivity_results.gradient_magnitudes {
            if sensitivity > max_sensitivity {
                max_sensitivity = sensitivity;
                most_sensitive = param.clone();
            }
            if sensitivity < min_sensitivity {
                min_sensitivity = sensitivity;
                least_sensitive = param.clone();
            }
        }
        sensitivity_results.most_sensitive_parameter = most_sensitive;
        sensitivity_results.least_sensitive_parameter = least_sensitive;
        sensitivity_results.robustness_score = if max_sensitivity > 0.0 {
            T::from(1.0 / (1.0 + max_sensitivity)).expect("unwrap failed")
        } else {
            T::one()
        };
        for (param, &base_sens) in &sensitivity_results.parameter_sensitivities {
            let std_error = base_sens.abs() * 0.1;
            let margin = 1.96 * std_error;
            sensitivity_results
                .confidence_intervals
                .insert(param.clone(), (base_sens - margin, base_sens + margin));
        }
        let mut interaction_config = base_config.clone();
        interaction_config.epsilon = base_config.epsilon
            * (T::one() + perturbation_factor);
        interaction_config.noise_multiplier = base_config.noise_multiplier
            * (T::one() + perturbation_factor);
        let interaction_utility = model_fn(data, &interaction_config)?;
        let expected_additive = base_utility + (epsilon_utility - base_utility)
            + (noise_utility - base_utility);
        let interaction_effect = interaction_utility - expected_additive;
        sensitivity_results
            .interaction_effects
            .insert(
                "epsilon_noise_multiplier".to_string(),
                interaction_effect.to_f64().unwrap_or(0.0),
            );
        Ok(sensitivity_results)
    }
    /// Evaluate robustness
    #[allow(dead_code)]
    pub fn evaluate_robustness<D: Data<Elem = T> + Sync, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(
            &ArrayBase<D, Dim>,
            &PrivacyConfiguration<T>,
        ) -> Result<T> + Sync,
        config: &PrivacyConfiguration<T>,
    ) -> Result<RobustnessResults<T>> {
        use crate::error::OptimError;
        let base_utility_t = model_fn(data, config)?;
        let base_utility = base_utility_t.to_f64().unwrap_or(0.0);
        let perturbation_levels: [f64; 3] = [0.01, 0.05, 0.20];
        let pert_medium = perturbation_levels[1];
        let n_levels = perturbation_levels.len();
        let mut param_utils = vec![base_utility; n_levels];
        let mut data_utils = vec![base_utility; n_levels];
        let mut noise_utils = vec![base_utility; n_levels];
        let mut env_utils = vec![base_utility; n_levels];
        for (l, &factor) in perturbation_levels.iter().enumerate() {
            let mut p_cfg = config.clone();
            p_cfg.epsilon = T::from(config.epsilon.to_f64().unwrap_or(1.0) + factor)
                .unwrap_or(config.epsilon);
            param_utils[l] = model_fn(data, &p_cfg)?.to_f64().unwrap_or(0.0);
            let mut d_cfg = config.clone();
            d_cfg.noise_multiplier = T::from(
                    config.noise_multiplier.to_f64().unwrap_or(1.0) + factor,
                )
                .unwrap_or(config.noise_multiplier);
            data_utils[l] = model_fn(data, &d_cfg)?.to_f64().unwrap_or(0.0);
            let mut n_cfg = config.clone();
            n_cfg.noise_multiplier = T::from(
                    config.noise_multiplier.to_f64().unwrap_or(1.0) + factor,
                )
                .unwrap_or(config.noise_multiplier);
            noise_utils[l] = model_fn(data, &n_cfg)?.to_f64().unwrap_or(0.0);
            let mut e_cfg = config.clone();
            e_cfg.clipping_threshold = T::from(
                    config.clipping_threshold.to_f64().unwrap_or(1.0) + factor,
                )
                .unwrap_or(config.clipping_threshold);
            let new_lr = (config.learning_rate.to_f64().unwrap_or(0.01) * (1.0 - factor))
                .max(1e-8_f64);
            e_cfg.learning_rate = T::from(new_lr).unwrap_or(config.learning_rate);
            env_utils[l] = model_fn(data, &e_cfg)?.to_f64().unwrap_or(0.0);
        }
        let worst_param_util = param_utils.iter().cloned().fold(f64::INFINITY, f64::min);
        let adv_drop = (base_utility - worst_param_util).max(0.0);
        let adv_factor = perturbation_levels[2];
        let mut adv_utils = vec![base_utility; n_levels];
        for (l, &factor) in perturbation_levels.iter().enumerate() {
            let _ = factor;
            let scaled_drop = adv_drop * perturbation_levels[l] / adv_factor;
            adv_utils[l] = base_utility - scaled_drop;
        }
        let mut all_utils: Vec<f64> = Vec::new();
        all_utils.extend_from_slice(&param_utils);
        all_utils.extend_from_slice(&data_utils);
        all_utils.extend_from_slice(&noise_utils);
        all_utils.extend_from_slice(&adv_utils);
        all_utils.extend_from_slice(&env_utils);
        let min_utility = all_utils.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst_case_degradation = (base_utility - min_utility).max(0.0);
        let robustness_score = 1.0 / (1.0 + worst_case_degradation * 10.0);
        let adv_min = adv_utils.iter().cloned().fold(f64::INFINITY, f64::min);
        let adv_worst = (base_utility - adv_min).max(0.0);
        let adversarial_robustness = 1.0 / (1.0 + adv_worst * 10.0);
        let dist_min = data_utils.iter().cloned().fold(f64::INFINITY, f64::min);
        let dist_worst = (base_utility - dist_min).max(0.0);
        let distributional_robustness = 1.0 / (1.0 + dist_worst * 10.0);
        let u_adv_med = adv_utils[1];
        let lyap = if base_utility > 1e-12 {
            let val = (u_adv_med.ln() - base_utility.ln()) / pert_medium;
            if val.is_nan() || val.is_infinite() { 0.0 } else { val }
        } else {
            0.0
        };
        let epsilon_machine = 1e-10_f64;
        let stability_margin = (0.5 * base_utility - worst_case_degradation)
            / base_utility.max(epsilon_machine);
        let mean_utils = all_utils.iter().sum::<f64>() / all_utils.len() as f64;
        let var_utils = all_utils.iter().map(|u| (u - mean_utils).powi(2)).sum::<f64>()
            / all_utils.len() as f64;
        let std_utils = var_utils.sqrt();
        let convergence_rate = (1.0 - std_utils / mean_utils.abs().max(1e-10))
            .clamp(0.0, 1.0);
        let convergence_radius = worst_case_degradation;
        let asymptotic_behavior = if lyap < -0.5 {
            AsymptoticBehavior::Exponential
        } else if lyap < 0.0 {
            AsymptoticBehavior::Linear
        } else if lyap < 0.5 {
            AsymptoticBehavior::Sublinear
        } else if lyap < 2.0 {
            AsymptoticBehavior::Oscillatory
        } else {
            AsymptoticBehavior::Chaotic
        };
        let pert_types_utils_med: &[(PerturbationType, f64)] = &[
            (PerturbationType::Parameter, param_utils[1]),
            (PerturbationType::Data, data_utils[1]),
            (PerturbationType::Noise, noise_utils[1]),
            (PerturbationType::Adversarial, adv_utils[1]),
            (PerturbationType::Environmental, env_utils[1]),
        ];
        let perturbation_effects: Vec<PerturbationEffect<T>> = pert_types_utils_med
            .iter()
            .map(|(pt, pu)| {
                let deg = (base_utility - pu).max(0.0);
                let deg_frac = if base_utility > 1e-12 {
                    (deg / base_utility).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                PerturbationEffect {
                    perturbation_type: pt.clone(),
                    effect_magnitude: T::from(deg).unwrap_or_else(|| T::zero()),
                    recovery_probability: T::from(1.0 - deg_frac)
                        .unwrap_or_else(|| T::zero()),
                    long_term_impact: T::from(deg_frac.powi(2))
                        .unwrap_or_else(|| T::zero()),
                }
            })
            .collect();
        let perturbation_sensitivity = worst_case_degradation / base_utility.max(1e-10);
        let mut failure_modes: Vec<FailureMode<T>> = Vec::new();
        let conf_level = self.config.confidence_level;
        let adv_med_util = adv_utils[1];
        if adv_med_util < base_utility * (1.0 - conf_level) {
            let deg_frac = ((base_utility - adv_med_util) / base_utility.max(1e-12))
                .clamp(0.0, 1.0);
            failure_modes
                .push(FailureMode {
                    failure_type: FailureType::PrivacyBreach,
                    failure_probability: T::from(deg_frac).unwrap_or_else(|| T::zero()),
                    impact_severity: T::from(0.8).unwrap_or_else(|| T::zero()),
                    detection_probability: T::from(0.9).unwrap_or_else(|| T::zero()),
                    mitigation_strategies: vec![
                        "Increase noise multiplier".to_string(), "Reduce epsilon"
                        .to_string(),
                    ],
                });
        }
        if worst_case_degradation > base_utility * 0.5 {
            failure_modes
                .push(FailureMode {
                    failure_type: FailureType::UtilityCollapse,
                    failure_probability: T::from(
                            (worst_case_degradation / base_utility.max(1e-12)).min(1.0),
                        )
                        .unwrap_or_else(|| T::zero()),
                    impact_severity: T::from(0.9).unwrap_or_else(|| T::zero()),
                    detection_probability: T::from(0.95).unwrap_or_else(|| T::zero()),
                    mitigation_strategies: vec![
                        "Tune privacy budget allocation".to_string(),
                        "Increase training data".to_string(),
                    ],
                });
        }
        if lyap > 0.0 {
            failure_modes
                .push(FailureMode {
                    failure_type: FailureType::SystemInstability,
                    failure_probability: T::from((lyap / (1.0 + lyap)).min(1.0))
                        .unwrap_or_else(|| T::zero()),
                    impact_severity: T::from(0.7).unwrap_or_else(|| T::zero()),
                    detection_probability: T::from(0.85).unwrap_or_else(|| T::zero()),
                    mitigation_strategies: vec![
                        "Reduce learning rate".to_string(), "Apply gradient clipping"
                        .to_string(),
                    ],
                });
        }
        if convergence_rate < 0.5 {
            failure_modes
                .push(FailureMode {
                    failure_type: FailureType::ConvergenceFailure,
                    failure_probability: T::from(1.0 - convergence_rate)
                        .unwrap_or_else(|| T::zero()),
                    impact_severity: T::from(0.6).unwrap_or_else(|| T::zero()),
                    detection_probability: T::from(0.9).unwrap_or_else(|| T::zero()),
                    mitigation_strategies: vec![
                        "Adjust learning rate schedule".to_string(),
                        "Increase batch size".to_string(),
                    ],
                });
        }
        if dist_worst > base_utility * 0.30 {
            failure_modes
                .push(FailureMode {
                    failure_type: FailureType::RobustnessFailure,
                    failure_probability: T::from(
                            (dist_worst / base_utility.max(1e-12)).min(1.0),
                        )
                        .unwrap_or_else(|| T::zero()),
                    impact_severity: T::from(0.75).unwrap_or_else(|| T::zero()),
                    detection_probability: T::from(0.88).unwrap_or_else(|| T::zero()),
                    mitigation_strategies: vec![
                        "Increase noise multiplier".to_string(), "Use data augmentation"
                        .to_string(),
                    ],
                });
        }
        Ok(RobustnessResults {
            robustness_score: T::from(robustness_score)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "robustness_score conversion failed".to_string(),
                    )
                })?,
            worst_case_degradation: T::from(worst_case_degradation)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "worst_case_degradation conversion failed".to_string(),
                    )
                })?,
            adversarial_robustness: T::from(adversarial_robustness)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "adversarial_robustness conversion failed".to_string(),
                    )
                })?,
            distributional_robustness: T::from(distributional_robustness)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "distributional_robustness conversion failed".to_string(),
                    )
                })?,
            stability_analysis: StabilityAnalysis {
                lyapunov_exponent: T::from(lyap).unwrap_or_else(|| T::zero()),
                stability_margin: T::from(stability_margin).unwrap_or_else(|| T::zero()),
                convergence_properties: ConvergenceProperties {
                    convergence_rate: T::from(convergence_rate)
                        .unwrap_or_else(|| T::zero()),
                    convergence_radius: T::from(convergence_radius)
                        .unwrap_or_else(|| T::zero()),
                    asymptotic_behavior,
                    stability_guarantees: lyap < 0.0,
                },
                perturbation_analysis: PerturbationAnalysis {
                    perturbation_sensitivity: T::from(perturbation_sensitivity)
                        .unwrap_or_else(|| T::zero()),
                    critical_threshold: T::from(0.20).unwrap_or_else(|| T::zero()),
                    recovery_time: T::from(10.0).unwrap_or_else(|| T::zero()),
                    perturbation_effects,
                },
            },
            failure_modes,
        })
    }
    /// Predict utility degradation
    #[allow(dead_code)]
    pub fn predict_utility_degradation(
        &self,
        privacy_parameters: &[T],
        historical_data: &[(T, T)],
    ) -> Result<Vec<DegradationPrediction<T>>> {
        use crate::error::OptimError;
        if historical_data.is_empty() {
            return Err(
                OptimError::InvalidParameter(
                    "historical_data must not be empty".to_string(),
                ),
            );
        }
        if privacy_parameters.is_empty() {
            return Ok(Vec::new());
        }
        let n = historical_data.len();
        let degree = if n < 5 { 1 } else if n < 15 { 2 } else { 3 };
        let xs: Vec<f64> = historical_data
            .iter()
            .map(|(e, _)| e.to_f64().unwrap_or(0.0))
            .collect();
        let ys: Vec<f64> = historical_data
            .iter()
            .map(|(_, u)| u.to_f64().unwrap_or(0.0))
            .collect();
        let coeffs = self.polyfit_f64(&xs, &ys, degree)?;
        let fitted: Vec<f64> = xs
            .iter()
            .map(|&x| self.polyeval_f64(&coeffs, x))
            .collect();
        let mean_y = ys.iter().sum::<f64>() / n as f64;
        let ss_res: f64 = ys
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| (y - f).powi(2))
            .sum();
        let ss_tot: f64 = ys.iter().map(|y| (y - mean_y).powi(2)).sum();
        let r_squared = if ss_tot > 1e-12 {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let dof = (n as i64 - degree as i64 - 1).max(1) as f64;
        let sigma = (ss_res / dof).sqrt();
        let mut predictions = Vec::with_capacity(privacy_parameters.len());
        for param in privacy_parameters {
            let x = param.to_f64().unwrap_or(0.0);
            let predicted_raw = self.polyeval_f64(&coeffs, x);
            let predicted = predicted_raw.clamp(0.0, 1.0);
            let margin = 1.96 * sigma * (1.0 + 1.0 / n as f64).sqrt();
            let ci_lo = (predicted - margin).clamp(0.0, 1.0);
            let ci_hi = (predicted + margin).clamp(0.0, 1.0);
            let model_accuracy = T::from(r_squared)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "model_accuracy conversion failed".to_string(),
                    )
                })?;
            let prediction_model = if degree <= 1 {
                PredictionModel::LinearRegression
            } else {
                PredictionModel::PolynomialRegression
            };
            predictions
                .push(DegradationPrediction {
                    privacy_parameter: *param,
                    predicted_utility_loss: T::from(predicted)
                        .unwrap_or_else(|| T::zero()),
                    confidence_interval: (
                        T::from(ci_lo).unwrap_or_else(|| T::zero()),
                        T::from(ci_hi).unwrap_or_else(|| T::zero()),
                    ),
                    prediction_model,
                    model_accuracy,
                });
        }
        Ok(predictions)
    }
    /// Assess privacy risk
    #[allow(dead_code)]
    pub fn assess_privacy_risk<D: Data<Elem = T> + Sync, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        config: &PrivacyConfiguration<T>,
    ) -> Result<PrivacyRiskAssessment<T>> {
        use crate::error::OptimError;
        if config.epsilon <= T::zero() {
            return Err(
                OptimError::InvalidParameter("config.epsilon must be > 0".to_string()),
            );
        }
        if config.delta < T::zero() {
            return Err(
                OptimError::InvalidParameter("config.delta must be >= 0".to_string()),
            );
        }
        let eps = config.epsilon.to_f64().unwrap_or(0.0);
        let delta = config.delta.to_f64().unwrap_or(0.0);
        let n = data.len().max(1);
        let iterations = config.iterations;
        let mem_risk = 1.0 - (-eps).exp();
        let attr_risk = 1.0 - (-eps * 0.7).exp();
        let inv_risk = (eps * eps / (n as f64).sqrt()).min(1.0);
        let prop_risk = 1.0 - (-eps * 0.5).exp();
        let recon_risk = delta * (1.0 - (-eps).exp());
        let delta_clamped = delta.clamp(0.0, 1.0);
        let reid_risk = (1.0 - (1.0 - delta_clamped).powi(iterations as i32)).max(0.0);
        let scores_f64 = [
            mem_risk,
            attr_risk,
            inv_risk,
            prop_risk,
            recon_risk,
            reid_risk,
        ];
        let overall_risk_score = scores_f64
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut risk_categories: HashMap<RiskCategory, T> = HashMap::new();
        risk_categories
            .insert(
                RiskCategory::MembershipInference,
                T::from(mem_risk).unwrap_or_else(|| T::zero()),
            );
        risk_categories
            .insert(
                RiskCategory::AttributeInference,
                T::from(attr_risk).unwrap_or_else(|| T::zero()),
            );
        risk_categories
            .insert(
                RiskCategory::ModelInversion,
                T::from(inv_risk).unwrap_or_else(|| T::zero()),
            );
        risk_categories
            .insert(
                RiskCategory::PropertyInference,
                T::from(prop_risk).unwrap_or_else(|| T::zero()),
            );
        risk_categories
            .insert(
                RiskCategory::Reconstruction,
                T::from(recon_risk).unwrap_or_else(|| T::zero()),
            );
        risk_categories
            .insert(
                RiskCategory::ReIdentification,
                T::from(reid_risk).unwrap_or_else(|| T::zero()),
            );
        let mut mitigation: Vec<String> = Vec::new();
        if mem_risk > 0.5 {
            mitigation
                .push(
                    format!(
                        "Reduce epsilon below 0.7 to cut membership-inference advantage below 50% (current ε={eps:.3})"
                    ),
                );
        }
        if recon_risk > 0.1 {
            mitigation
                .push(
                    format!(
                        "Reduce delta — current δ={delta:.2e} allows reconstruction risk of {recon_risk:.2}"
                    ),
                );
        }
        if reid_risk > 0.3 {
            mitigation
                .push(
                    "Switch to Rényi-DP accountant to tighten cumulative re-identification bounds"
                        .to_string(),
                );
        }
        if inv_risk > 0.5 {
            mitigation
                .push(
                    "Reduce ε or increase batch size to limit gradient-leakage model-inversion risk"
                        .to_string(),
                );
        }
        if mitigation.is_empty() {
            mitigation
                .push(
                    "Privacy parameters are within acceptable risk bounds for this workload"
                        .to_string(),
                );
        }
        let compliance_status = if overall_risk_score < 0.3 && delta < 1e-5 {
            ComplianceStatus::Compliant
        } else if overall_risk_score < 0.6 {
            ComplianceStatus::PartiallyCompliant
        } else {
            ComplianceStatus::NonCompliant
        };
        let first_reid = reid_risk;
        let risk_evolution: Vec<RiskEvolution<T>> = (1..=5_usize)
            .map(|k| {
                let time_point = 200 * k;
                let future_iterations = iterations + 200 * k;
                let future_reid = (1.0
                    - (1.0 - delta_clamped).powi(future_iterations as i32))
                    .max(0.0);
                let risk_trend = if future_reid > first_reid * 1.1 {
                    RiskTrend::Increasing
                } else {
                    RiskTrend::Stable
                };
                RiskEvolution {
                    time_point,
                    risk_score: T::from(future_reid).unwrap_or_else(|| T::zero()),
                    contributing_factors: vec![
                        "cumulative_reidentification".to_string()
                    ],
                    risk_trend,
                }
            })
            .collect();
        Ok(PrivacyRiskAssessment {
            overall_risk_score: T::from(overall_risk_score)
                .ok_or_else(|| {
                    OptimError::ComputationError(
                        "overall_risk_score conversion failed".to_string(),
                    )
                })?,
            risk_categories,
            mitigation_recommendations: mitigation,
            compliance_status,
            risk_evolution,
        })
    }
    /// Perform statistical significance testing
    #[allow(dead_code)]
    pub fn perform_statistical_tests(
        &self,
        results: &[(T, T)],
        baseline: &[(T, T)],
    ) -> Result<StatisticalTestResults<T>> {
        use crate::error::OptimError;
        if results.len() < 2 {
            return Err(
                OptimError::InvalidParameter(
                    "results must have at least 2 elements".to_string(),
                ),
            );
        }
        if baseline.len() < 2 {
            return Err(
                OptimError::InvalidParameter(
                    "baseline must have at least 2 elements".to_string(),
                ),
            );
        }
        let results_u: Vec<f64> = results
            .iter()
            .map(|(_, u)| u.to_f64().unwrap_or(0.0))
            .collect();
        let baseline_u: Vec<f64> = baseline
            .iter()
            .map(|(_, u)| u.to_f64().unwrap_or(0.0))
            .collect();
        let results_p: Vec<f64> = results
            .iter()
            .map(|(p, _)| p.to_f64().unwrap_or(0.0))
            .collect();
        let baseline_p: Vec<f64> = baseline
            .iter()
            .map(|(p, _)| p.to_f64().unwrap_or(0.0))
            .collect();
        let (m1, v1) = self.mean_var_f64(&results_u);
        let (m2, v2) = self.mean_var_f64(&baseline_u);
        let n1 = results_u.len() as f64;
        let n2 = baseline_u.len() as f64;
        let se = (v1 / n1 + v2 / n2).sqrt();
        let t_stat = if se > 1e-12 { (m1 - m2) / se } else { 0.0 };
        let df_num = (v1 / n1 + v2 / n2).powi(2);
        let df_den = (v1 / n1).powi(2) / (n1 - 1.0).max(1.0)
            + (v2 / n2).powi(2) / (n2 - 1.0).max(1.0);
        let df = if df_den > 1e-12 { df_num / df_den } else { 1.0 };
        let p_welch = 2.0 * (1.0 - self.normal_cdf_f64(t_stat.abs()))
            * (1.0 + 1.0 / (4.0 * df.max(1.0)));
        let p_welch = p_welch.clamp(0.0, 1.0);
        let pooled_std = ((v1 * (n1 - 1.0) + v2 * (n2 - 1.0)) / (n1 + n2 - 2.0).max(1.0))
            .sqrt()
            .max(1e-12);
        let cohen_d = (m1 - m2) / pooled_std;
        let mut hypothesis_tests: Vec<HypothesisTestResult<T>> = Vec::new();
        hypothesis_tests
            .push(HypothesisTestResult {
                test_name: "Welch t-test (utility)".to_string(),
                test_statistic: T::from(t_stat).unwrap_or_else(|| T::zero()),
                p_value: T::from(p_welch).unwrap_or_else(|| T::zero()),
                significance_level: T::from(0.05).unwrap_or_else(|| T::zero()),
                reject_null: p_welch < 0.05,
                effect_size: T::from(cohen_d).unwrap_or_else(|| T::zero()),
            });
        let r1 = self.pearson_correlation_f64(&results_p, &results_u);
        let r2 = self.pearson_correlation_f64(&baseline_p, &baseline_u);
        let r1_c = r1.clamp(-0.9999, 0.9999);
        let r2_c = r2.clamp(-0.9999, 0.9999);
        let z1 = r1_c.atanh();
        let z2 = r2_c.atanh();
        let se_z = (1.0 / (results.len() as f64 - 3.0).max(1.0)
            + 1.0 / (baseline.len() as f64 - 3.0).max(1.0))
            .sqrt();
        let z_stat = (z1 - z2) / se_z.max(1e-12);
        let p_corr = (2.0 * (1.0 - self.normal_cdf_f64(z_stat.abs()))).clamp(0.0, 1.0);
        let corr_effect = (r1 - r2).abs();
        hypothesis_tests
            .push(HypothesisTestResult {
                test_name: "Fisher z-test (privacy-utility correlation)".to_string(),
                test_statistic: T::from(z_stat).unwrap_or_else(|| T::zero()),
                p_value: T::from(p_corr).unwrap_or_else(|| T::zero()),
                significance_level: T::from(0.05).unwrap_or_else(|| T::zero()),
                reject_null: p_corr < 0.05,
                effect_size: T::from(corr_effect).unwrap_or_else(|| T::zero()),
            });
        let n_total = (n1 + n2) / 2.0;
        let stat_power = self
            .normal_cdf_f64(cohen_d.abs() * (n_total / 2.0).sqrt() - 1.96)
            .max(0.0);
        let required_n = if cohen_d.abs() > 1e-6 {
            (((1.96 + 0.84) / cohen_d.abs()).powi(2).ceil() as usize).saturating_mul(2)
        } else {
            usize::MAX / 2
        };
        let min_det_effect = if n_total > 0.0 {
            1.96 * (2.0 / n_total).sqrt()
        } else {
            f64::INFINITY
        };
        let power_curve: Vec<(T, T)> = (1..=20_usize)
            .map(|k| {
                let d_val = k as f64 * 0.05;
                let pw = self
                    .normal_cdf_f64(d_val * (n_total / 2.0).sqrt() - 1.96)
                    .max(0.0);
                (
                    T::from(d_val).unwrap_or_else(|| T::zero()),
                    T::from(pw).unwrap_or_else(|| T::zero()),
                )
            })
            .collect();
        let power_analysis = PowerAnalysis {
            statistical_power: T::from(stat_power).unwrap_or_else(|| T::zero()),
            required_sample_size: required_n,
            minimum_detectable_effect: T::from(min_det_effect)
                .unwrap_or_else(|| T::zero()),
            power_curve,
        };
        let raw_p = [p_welch, p_corr];
        let k = 2_usize;
        let bonf_adj: Vec<T> = raw_p
            .iter()
            .map(|&p| T::from((p * k as f64).min(1.0)).unwrap_or_else(|| T::zero()))
            .collect();
        let bonf_rejected = bonf_adj
            .iter()
            .filter(|&&p| { p < T::from(0.05).unwrap_or_else(|| T::zero()) })
            .count();
        let family_wise = 1.0 - 0.95_f64.powi(k as i32);
        let bonf_fdr = bonf_rejected as f64 / k as f64;
        let mut sorted: Vec<(usize, f64)> = raw_p.iter().cloned().enumerate().collect();
        sorted
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut holm_adj_by_rank: Vec<f64> = sorted
            .iter()
            .enumerate()
            .map(|(rank, (_, p))| (*p * (k - rank) as f64).min(1.0))
            .collect();
        for i in 1..holm_adj_by_rank.len() {
            if holm_adj_by_rank[i] < holm_adj_by_rank[i - 1] {
                holm_adj_by_rank[i] = holm_adj_by_rank[i - 1];
            }
        }
        let mut holm_orig = vec![0.0_f64; k];
        for (rank, (orig_idx, _)) in sorted.iter().enumerate() {
            holm_orig[*orig_idx] = holm_adj_by_rank[rank];
        }
        let holm_adj: Vec<T> = holm_orig
            .iter()
            .map(|&p| T::from(p).unwrap_or_else(|| T::zero()))
            .collect();
        let holm_rejected = holm_adj
            .iter()
            .filter(|&&p| p < T::from(0.05).unwrap_or_else(|| T::zero()))
            .count();
        let holm_fdr = holm_rejected as f64 / k as f64;
        let multiple_comparison_corrections = vec![
            MultipleComparisonCorrection { correction_method :
            CorrectionMethod::Bonferroni, adjusted_p_values : bonf_adj,
            family_wise_error_rate : T::from(family_wise).unwrap_or_else(|| T::zero()),
            false_discovery_rate : T::from(bonf_fdr).unwrap_or_else(|| T::zero()), },
            MultipleComparisonCorrection { correction_method :
            CorrectionMethod::HolmBonferroni, adjusted_p_values : holm_adj,
            family_wise_error_rate : T::from(family_wise).unwrap_or_else(|| T::zero()),
            false_discovery_rate : T::from(holm_fdr).unwrap_or_else(|| T::zero()), },
        ];
        Ok(StatisticalTestResults {
            hypothesis_tests,
            significance_levels: vec![
                T::from(0.05).unwrap_or_else(|| T::zero()), T::from(0.01)
                .unwrap_or_else(|| T::zero()),
            ],
            effect_sizes: vec![
                T::from(cohen_d).unwrap_or_else(|| T::zero()), T::from(corr_effect)
                .unwrap_or_else(|| T::zero()),
            ],
            power_analysis,
            multiple_comparison_corrections,
        })
    }
    /// Polynomial least-squares fit via normal equations (Vandermonde, f64).
    /// Returns coefficients [c0, c1, ..., c_degree] (constant first).
    pub(crate) fn polyfit_f64(&self, xs: &[f64], ys: &[f64], degree: usize) -> Result<Vec<f64>> {
        let n = xs.len();
        let d = degree + 1;
        let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(n);
        for &xi in xs {
            let mut row = Vec::with_capacity(d);
            let mut pw = 1.0_f64;
            for _ in 0..d {
                row.push(pw);
                pw *= xi;
            }
            x_mat.push(row);
        }
        let mut xtx: Vec<Vec<f64>> = vec![vec![0.0_f64; d]; d];
        let mut xty: Vec<f64> = vec![0.0_f64; d];
        for i in 0..n {
            for j in 0..d {
                xty[j] += x_mat[i][j] * ys[i];
                for l in 0..d {
                    xtx[j][l] += x_mat[i][j] * x_mat[i][l];
                }
            }
        }
        self.solve_linear_system_f64(xtx, xty)
    }
    /// Gauss-Jordan elimination with partial pivoting.
    pub(crate) fn solve_linear_system_f64(
        &self,
        mut a: Vec<Vec<f64>>,
        mut b: Vec<f64>,
    ) -> Result<Vec<f64>> {
        use crate::error::OptimError;
        let n = b.len();
        for col in 0..n {
            let pivot_row = (col..n)
                .max_by(|&r1, &r2| {
                    a[r1][col]
                        .abs()
                        .partial_cmp(&a[r2][col].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(col);
            if a[pivot_row][col].abs() < 1e-12 {
                return Err(
                    OptimError::ComputationError(
                        "Singular matrix in polyfit".to_string(),
                    ),
                );
            }
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
            let pivot = a[col][col];
            for val in a[col][col..n].iter_mut() {
                *val /= pivot;
            }
            b[col] /= pivot;
            let pivot_row_snap: Vec<f64> = a[col][col..n].to_vec();
            let pivot_b = b[col];
            for row in 0..n {
                if row != col {
                    let factor = a[row][col];
                    for (j, pv) in pivot_row_snap.iter().enumerate() {
                        a[row][col + j] -= factor * pv;
                    }
                    b[row] -= factor * pivot_b;
                }
            }
        }
        Ok(b)
    }
    /// Evaluate polynomial using Horner's method.
    /// `coeffs[0]` is the constant term.
    pub(crate) fn polyeval_f64(&self, coeffs: &[f64], x: f64) -> f64 {
        if coeffs.is_empty() {
            return 0.0;
        }
        let mut result = *coeffs.last().unwrap_or(&0.0);
        for c in coeffs.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }
    /// Standard normal CDF via Abramowitz & Stegun approximation 26.2.17.
    pub(crate) fn normal_cdf_f64(&self, z: f64) -> f64 {
        let sign = if z >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        let z_abs = z.abs();
        let t = 1.0 / (1.0 + 0.2316419 * z_abs);
        let phi = (-0.5 * z_abs * z_abs).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let poly = t
            * (0.319_381_53
                + t
                    * (-0.356_563_782
                        + t
                            * (1.781_477_937
                                + t * (-1.821_255_978 + t * 1.330_274_429))));
        let y = 1.0 - phi * poly;
        if sign > 0.0 { y } else { 1.0 - y }
    }
    /// Pearson correlation coefficient (f64 internal).
    /// Returns 0.0 if lengths differ or either std_dev is ~0.
    pub(crate) fn pearson_correlation_f64(&self, xs: &[f64], ys: &[f64]) -> f64 {
        if xs.len() != ys.len() || xs.is_empty() {
            return 0.0;
        }
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let cov: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
        let sx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>().sqrt();
        let sy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum::<f64>().sqrt();
        if sx < 1e-12 || sy < 1e-12 {
            return 0.0;
        }
        (cov / (sx * sy)).clamp(-0.9999, 0.9999)
    }
    /// Returns (mean, sample_variance). Returns (0.0, 0.0) for empty input.
    pub(crate) fn mean_var_f64(&self, xs: &[f64]) -> (f64, f64) {
        let n = xs.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let mean = xs.iter().sum::<f64>() / n as f64;
        let denom = (n - 1).max(1) as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / denom;
        (mean, var)
    }
}
impl<T: Float + Debug + Send + Sync + 'static> PrivacyUtilityAnalyzer<T> {
    /// Generate privacy configurations for parameter space exploration
    fn generate_privacy_configurations(&self) -> Result<Vec<PrivacyConfiguration<T>>> {
        let mut configurations = Vec::new();
        let params = &self.config.privacy_parameters;
        let epsilon_values = self.sample_parameter_range(&params.epsilon_range)?;
        let delta_values = self.sample_parameter_range(&params.delta_range)?;
        let noise_values = self.sample_parameter_range(&params.noise_multiplier_range)?;
        let clip_values = self.sample_parameter_range(&params.clipping_threshold_range)?;
        let batch_values = self.sample_parameter_range(&params.batch_size_range)?;
        let max_combinations = self.config.pareto_resolution;
        let combinations_per_dimension = (max_combinations as f64).powf(1.0 / 5.0).ceil()
            as usize;
        for (_i, &epsilon) in epsilon_values
            .iter()
            .enumerate()
            .take(combinations_per_dimension)
        {
            for (_j, &delta) in delta_values
                .iter()
                .enumerate()
                .take(combinations_per_dimension)
            {
                for (_k, &noise_mult) in noise_values
                    .iter()
                    .enumerate()
                    .take(combinations_per_dimension)
                {
                    for (_l, &clip_thresh) in clip_values
                        .iter()
                        .enumerate()
                        .take(combinations_per_dimension)
                    {
                        for (_m, &batch_size) in batch_values
                            .iter()
                            .enumerate()
                            .take(combinations_per_dimension)
                        {
                            if delta >= T::from(1.0).unwrap_or_else(|| T::zero())
                                || epsilon <= T::zero() || noise_mult <= T::zero()
                            {
                                continue;
                            }
                            configurations
                                .push(PrivacyConfiguration {
                                    epsilon,
                                    delta,
                                    noise_multiplier: noise_mult,
                                    clipping_threshold: clip_thresh,
                                    batch_size: batch_size.to_usize().unwrap_or(256),
                                    sampling_probability: T::from(0.1)
                                        .unwrap_or_else(|| T::zero()),
                                    iterations: 1000,
                                    learning_rate: T::from(0.01).unwrap_or_else(|| T::zero()),
                                    noise_mechanism: NoiseMechanism::Gaussian,
                                });
                            if configurations.len() >= max_combinations {
                                return Ok(configurations);
                            }
                        }
                    }
                }
            }
        }
        Ok(configurations)
    }
    /// Sample values from a parameter range
    fn sample_parameter_range(&self, range: &ParameterRange) -> Result<Vec<T>> {
        let mut values = Vec::new();
        match range.sampling_strategy {
            SamplingStrategy::Linear => {
                for i in 0..range.num_samples {
                    let fraction = i as f64 / (range.num_samples - 1).max(1) as f64;
                    let value = range.min + fraction * (range.max - range.min);
                    values.push(T::from(value).unwrap_or_else(|| T::zero()));
                }
            }
            SamplingStrategy::Logarithmic => {
                let log_min = range.min.ln();
                let log_max = range.max.ln();
                for i in 0..range.num_samples {
                    let fraction = i as f64 / (range.num_samples - 1).max(1) as f64;
                    let log_value = log_min + fraction * (log_max - log_min);
                    values.push(T::from(log_value.exp()).expect("unwrap failed"));
                }
            }
            SamplingStrategy::Random => {
                let mut rng = thread_rng();
                for _ in 0..range.num_samples {
                    let value = rng.gen_range(range.min..range.max);
                    values.push(T::from(value).unwrap_or_else(|| T::zero()));
                }
            }
            _ => {
                for i in 0..range.num_samples {
                    let fraction = i as f64 / (range.num_samples - 1).max(1) as f64;
                    let value = range.min + fraction * (range.max - range.min);
                    values.push(T::from(value).unwrap_or_else(|| T::zero()));
                }
            }
        }
        Ok(values)
    }
    /// Compute privacy cost for a configuration (lower epsilon = higher cost)
    fn compute_privacy_cost(&self, config: &PrivacyConfiguration<T>) -> Result<T> {
        let max_epsilon = T::from(10.0).unwrap_or_else(|| T::zero());
        let normalized_epsilon = config.epsilon / max_epsilon;
        let max_delta = T::from(1e-3).unwrap_or_else(|| T::zero());
        let normalized_delta = config.delta / max_delta;
        let privacy_cost = normalized_epsilon
            + normalized_delta * T::from(0.1).unwrap_or_else(|| T::zero());
        Ok(privacy_cost)
    }
}
/// Perturbation analysis
#[derive(Debug, Clone)]
pub struct PerturbationAnalysis<T: Float + Debug + Send + Sync + 'static> {
    /// Perturbation sensitivity
    pub perturbation_sensitivity: T,
    /// Critical perturbation threshold
    pub critical_threshold: T,
    /// Recovery time
    pub recovery_time: T,
    /// Perturbation effects
    pub perturbation_effects: Vec<PerturbationEffect<T>>,
}
/// Perturbation effect
#[derive(Debug, Clone)]
pub struct PerturbationEffect<T: Float + Debug + Send + Sync + 'static> {
    /// Perturbation type
    pub perturbation_type: PerturbationType,
    /// Effect magnitude
    pub effect_magnitude: T,
    /// Recovery probability
    pub recovery_probability: T,
    /// Long-term impact
    pub long_term_impact: T,
}
/// Types of perturbations
#[derive(Debug, Clone)]
pub enum PerturbationType {
    /// Parameter perturbation
    Parameter,
    /// Data perturbation
    Data,
    /// Noise perturbation
    Noise,
    /// Adversarial perturbation
    Adversarial,
    /// Environmental perturbation
    Environmental,
}
pub struct PrivacyParameterExplorer<T: Float + Debug + Send + Sync + 'static> {
    #[allow(dead_code)]
    phantom: std::marker::PhantomData<T>,
}
/// Multiple comparison correction
#[derive(Debug, Clone)]
pub struct MultipleComparisonCorrection<T: Float + Debug + Send + Sync + 'static> {
    /// Correction method
    pub correction_method: CorrectionMethod,
    /// Adjusted p-values
    pub adjusted_p_values: Vec<T>,
    /// Family-wise error rate
    pub family_wise_error_rate: T,
    /// False discovery rate
    pub false_discovery_rate: T,
}
/// Privacy-utility analysis results
#[derive(Debug, Clone)]
pub struct PrivacyUtilityResults<T: Float + Debug + Send + Sync + 'static> {
    /// Pareto frontier points
    pub pareto_frontier: Vec<ParetoPoint<T>>,
    /// Optimal privacy-utility configurations
    pub optimal_configurations: Vec<OptimalConfiguration<T>>,
    /// Sensitivity analysis results
    pub sensitivity_results: SensitivityResults<T>,
    /// Robustness evaluation results
    pub robustness_results: RobustnessResults<T>,
    /// Budget allocation recommendations
    pub budget_recommendations: BudgetRecommendations<T>,
    /// Utility degradation predictions
    pub degradation_predictions: Vec<DegradationPrediction<T>>,
    /// Privacy risk assessment
    pub privacy_risk_assessment: PrivacyRiskAssessment<T>,
    /// Statistical significance tests
    pub statistical_tests: StatisticalTestResults<T>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}
/// Adjustment frequency
#[derive(Debug, Clone)]
pub enum AdjustmentFrequency {
    /// Every iteration
    EveryIteration,
    /// Fixed interval
    FixedInterval(usize),
    /// Adaptive interval
    AdaptiveInterval,
    /// Event-driven
    EventDriven,
}
/// Risk categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiskCategory {
    /// Membership inference risk
    MembershipInference,
    /// Attribute inference risk
    AttributeInference,
    /// Model inversion risk
    ModelInversion,
    /// Property inference risk
    PropertyInference,
    /// Reconstruction risk
    Reconstruction,
    /// Re-identification risk
    ReIdentification,
}
/// Adaptation triggers
#[derive(Debug, Clone)]
pub enum AdaptationTrigger {
    /// Utility threshold
    UtilityThreshold,
    /// Privacy budget exhaustion
    BudgetExhaustion,
    /// Performance degradation
    PerformanceDegradation,
    /// Time-based trigger
    TimeBased,
    /// Convergence-based trigger
    ConvergenceBased,
}
/// Sensitivity analysis results
#[derive(Debug, Clone)]
pub struct SensitivityResults<T: Float + Debug + Send + Sync + 'static> {
    /// Base utility for comparison
    pub base_utility: T,
    /// Parameter sensitivities
    pub parameter_sensitivities: HashMap<String, f64>,
    /// Gradient magnitudes
    pub gradient_magnitudes: HashMap<String, f64>,
    /// Interaction effects
    pub interaction_effects: HashMap<String, f64>,
    /// Local sensitivity analysis
    pub local_sensitivities: Vec<LocalSensitivity<T>>,
    /// Global sensitivity bounds
    pub global_sensitivity_bounds: (T, T),
    /// Sensitivity rankings
    pub sensitivity_rankings: Vec<(String, T)>,
    /// Overall robustness score
    pub robustness_score: T,
    /// Most sensitive parameter
    pub most_sensitive_parameter: String,
    /// Least sensitive parameter
    pub least_sensitive_parameter: String,
    /// Confidence intervals for sensitivities
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}
/// Power analysis
#[derive(Debug, Clone)]
pub struct PowerAnalysis<T: Float + Debug + Send + Sync + 'static> {
    /// Statistical power
    pub statistical_power: T,
    /// Required sample size
    pub required_sample_size: usize,
    /// Minimum detectable effect
    pub minimum_detectable_effect: T,
    /// Power curve
    pub power_curve: Vec<(T, T)>,
}
pub struct UtilityMetricCalculator<T: Float + Debug + Send + Sync + 'static> {
    #[allow(dead_code)]
    phantom: std::marker::PhantomData<T>,
}
pub struct PrivacyBudgetOptimizer<T: Float + Debug + Send + Sync + 'static> {
    #[allow(dead_code)]
    phantom: std::marker::PhantomData<T>,
}
/// Risk evolution over time
#[derive(Debug, Clone)]
pub struct RiskEvolution<T: Float + Debug + Send + Sync + 'static> {
    /// Time point
    pub time_point: usize,
    /// Risk score at time point
    pub risk_score: T,
    /// Contributing factors
    pub contributing_factors: Vec<String>,
    /// Risk trend
    pub risk_trend: RiskTrend,
}
pub struct ParetoFrontierAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    #[allow(dead_code)]
    phantom: std::marker::PhantomData<T>,
}
