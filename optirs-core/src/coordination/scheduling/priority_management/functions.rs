//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::SystemTime;

use super::types::{
    AdjustmentContext, AdjustmentLearningDataPoint, AdjustmentModel, AlgorithmComplexity,
    AnalysisResult, AnalyticsResult, FeedbackRecord, IntegrationResult, LearnedPattern,
    LearningDataPoint, PatternTemplate, PriorityAnalyticsData, PriorityItem, PriorityLevel,
    PriorityTrend, PriorityUpdateContext, PriorityWeights, RecognizedPattern, TaskContext,
};
use super::types_14::PatternMatch;

/// Priority update strategy trait
pub trait PriorityUpdateStrategy<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Update priorities based on current state
    fn update_priorities(
        &mut self,
        items: &mut [PriorityItem<T>],
        context: &PriorityUpdateContext<T>,
    ) -> Result<()>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy performance metrics
    fn get_metrics(&self) -> HashMap<String, T>;

    /// Configure strategy parameters
    fn configure(&mut self, config: &HashMap<String, T>) -> Result<()>;
}

/// Priority calculation algorithm trait
pub trait PriorityCalculationAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Calculate priority for a task
    fn calculate_priority(
        &self,
        task_context: &TaskContext<T>,
        weights: &PriorityWeights<T>,
    ) -> Result<PriorityLevel<T>>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm complexity
    fn complexity(&self) -> AlgorithmComplexity;
}

/// Trend detection algorithm trait
pub trait TrendDetectionAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Detect trends in priority data
    fn detect_trend(&self, data: &[T], timestamps: &[SystemTime]) -> Result<PriorityTrend<T>>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get detection sensitivity
    fn sensitivity(&self) -> T;
}

/// Priority change detector trait
pub trait PriorityChangeDetector<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Detect significant priority changes
    fn detect_change(
        &self,
        old_priority: &PriorityLevel<T>,
        new_priority: &PriorityLevel<T>,
    ) -> Result<bool>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Get detection threshold
    fn threshold(&self) -> T;
}

/// Dynamic adjustment algorithm trait
pub trait DynamicAdjustmentAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Calculate dynamic adjustments
    fn calculate_adjustment(
        &self,
        current_priority: &PriorityLevel<T>,
        context: &AdjustmentContext<T>,
    ) -> Result<PriorityLevel<T>>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm responsiveness
    fn responsiveness(&self) -> T;
}

/// Feedback collection strategy trait
pub trait FeedbackCollectionStrategy<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Collect feedback from source
    fn collect_feedback(&mut self) -> Result<Vec<FeedbackRecord<T>>>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get collection reliability
    fn reliability(&self) -> T;
}

/// Feedback filter trait
pub trait FeedbackFilter<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Filter feedback records
    fn filter(&self, feedback: &FeedbackRecord<T>) -> bool;

    /// Get filter name
    fn name(&self) -> &str;
}

/// Feedback analysis algorithm trait
pub trait FeedbackAnalysisAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Analyze feedback data
    fn analyze(&self, feedback: &[FeedbackRecord<T>]) -> Result<AnalysisResult<T>>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Pattern detector trait
pub trait PatternDetector<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Detect patterns in feedback
    fn detect_patterns(&self, feedback: &[FeedbackRecord<T>]) -> Result<Vec<RecognizedPattern<T>>>;

    /// Get detector name
    fn name(&self) -> &str;
}

/// Pattern matcher trait
pub trait PatternMatcher<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Match patterns against templates
    fn match_pattern(
        &self,
        pattern: &RecognizedPattern<T>,
        templates: &[PatternTemplate<T>],
    ) -> Result<Vec<PatternMatch<T>>>;

    /// Get matcher name
    fn name(&self) -> &str;
}

/// Pattern learning algorithm trait
pub trait PatternLearningAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Learn new patterns from data
    fn learn_patterns(&mut self, data: &[LearningDataPoint<T>]) -> Result<Vec<LearnedPattern<T>>>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Feedback integration strategy trait
pub trait FeedbackIntegrationStrategy<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Integrate feedback into priority system
    fn integrate_feedback(
        &mut self,
        feedback: &[FeedbackRecord<T>],
        analysis: &[AnalysisResult<T>],
    ) -> Result<IntegrationResult<T>>;

    /// Get strategy name
    fn name(&self) -> &str;
}

/// Adjustment learning algorithm trait
pub trait AdjustmentLearningAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Learn adjustment strategies
    fn learn_adjustments(
        &mut self,
        data: &[AdjustmentLearningDataPoint<T>],
    ) -> Result<AdjustmentModel<T>>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Priority analytics algorithm trait
pub trait PriorityAnalyticsAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Perform analytics on priority data
    fn analyze(&self, data: &PriorityAnalyticsData<T>) -> Result<AnalyticsResult<T>>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::types::{
        AnalyticsConfig, CompressionStrategy, HistoryRetentionPolicy, PriorityManagerConfig,
        StaticPriorityStrategy, StrategySelection, UpdateTrigger, WeightedSumPriorityCalculator,
    };
    use super::super::types_14::PriorityManager;
    use super::*;
    use std::time::Duration;

    fn make_config() -> PriorityManagerConfig<f64> {
        PriorityManagerConfig {
            default_weights: PriorityWeights::default(),
            strategy_selection: StrategySelection::Fixed,
            analytics_config: AnalyticsConfig {
                enable_real_time: false,
                analytics_frequency: Duration::from_secs(60),
                algorithms: Vec::new(),
                thresholds: HashMap::new(),
            },
            history_retention: HistoryRetentionPolicy {
                max_history_size: 100,
                retention_duration: Duration::from_secs(3600),
                compression_strategy: CompressionStrategy::None,
            },
            performance_thresholds: HashMap::new(),
        }
    }

    fn make_task_context(parameters: HashMap<String, f64>) -> TaskContext<f64> {
        TaskContext {
            task_id: "task-1".to_string(),
            task_type: "test".to_string(),
            parameters,
            resource_requirements: HashMap::new(),
            estimated_duration: Duration::from_secs(1),
            deadline: None,
            historical_performance: None,
            dependencies: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    // Regression tests for F64: `update_strategies` and
    // `calculation_algorithms` were always empty -- there wasn't even a
    // `register_*` method to populate them -- and `current_update_strategy`
    // was hardcoded to `"default"`, a key nothing ever registered. Every
    // call to `update_priorities` therefore silently updated nothing while
    // still reporting success via `stats.total_updates`.

    #[test]
    fn test_new_manager_has_a_working_default_strategy_and_algorithm() {
        let mut manager = PriorityManager::<f64>::new(make_config()).expect("construction");

        // Must not error: a real strategy is registered under the key
        // `current_update_strategy` actually points at.
        let context = PriorityUpdateContext {
            system_load: 0.5,
            resource_availability: HashMap::new(),
            performance_metrics: HashMap::new(),
            time_since_update: Duration::from_secs(1),
            update_trigger: UpdateTrigger::Periodic,
            environmental_factors: HashMap::new(),
        };
        assert!(manager.update_priorities(context).is_ok());
        assert_eq!(manager.get_statistics().total_updates, 1);

        let mut params = HashMap::new();
        params.insert("urgency".to_string(), 0.8);
        let task_context = make_task_context(params);
        let weights = PriorityWeights::default();
        assert!(manager
            .calculate_priority(&task_context, "weighted_sum", &weights)
            .is_ok());
    }

    #[test]
    fn test_set_update_strategy_rejects_unregistered_name() {
        let mut manager = PriorityManager::<f64>::new(make_config()).expect("construction");
        assert!(manager.set_update_strategy("does-not-exist").is_err());
    }

    #[test]
    fn test_calculate_priority_rejects_unregistered_algorithm() {
        let manager = PriorityManager::<f64>::new(make_config()).expect("construction");
        let task_context = make_task_context(HashMap::new());
        let weights = PriorityWeights::default();
        let result = manager.calculate_priority(&task_context, "does-not-exist", &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_register_update_strategy_makes_it_selectable() {
        let mut manager = PriorityManager::<f64>::new(make_config()).expect("construction");
        manager.register_update_strategy("static-2", Box::new(StaticPriorityStrategy));
        assert!(manager.set_update_strategy("static-2").is_ok());
    }

    #[test]
    fn test_weighted_sum_calculator_computes_real_composite_score() {
        let calculator = WeightedSumPriorityCalculator;

        let mut params = HashMap::new();
        params.insert("urgency".to_string(), 1.0);
        params.insert("importance".to_string(), 0.0);
        params.insert("efficiency".to_string(), 0.0);
        params.insert("cost".to_string(), 0.0);
        params.insert("quality".to_string(), 0.0);
        params.insert("deadline_factor".to_string(), 0.0);
        params.insert("base_priority".to_string(), 0.0);
        let task_context = make_task_context(params);

        let weights = PriorityWeights {
            base_weight: 0.0,
            urgency_weight: 1.0,
            importance_weight: 0.0,
            efficiency_weight: 0.0,
            cost_weight: 0.0,
            quality_weight: 0.0,
            deadline_weight: 0.0,
            dynamic_weights: HashMap::new(),
        };

        let level = calculator
            .calculate_priority(&task_context, &weights)
            .expect("calculation should succeed");

        assert_eq!(level.urgency, 1.0);
        // Only `urgency_weight` is nonzero, so the composite score must
        // equal exactly `urgency_weight * urgency` = 1.0, not a placeholder
        // constant independent of the inputs.
        assert_eq!(level.composite_score, 1.0);
    }
}
