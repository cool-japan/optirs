//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::SystemTime;

use super::functions::{
    PriorityCalculationAlgorithm, PriorityChangeDetector, PriorityUpdateStrategy,
};
use super::types::{
    DynamicAdjustmentController, HistoryConfig, PriorityAnalytics, PriorityChangeRecord,
    PriorityItem, PriorityLevel, PriorityManagerConfig, PriorityManagerStatistics, PriorityQueue,
    PriorityTrendAnalyzer, PriorityUpdateContext, PriorityWeights, StaticPriorityStrategy,
    TaskContext, WeightedSumPriorityCalculator,
};

/// Priority manager for optimization tasks
#[derive(Debug)]
pub struct PriorityManager<T: Float + Debug + Send + Sync + 'static> {
    /// Priority queues by category
    pub(super) priority_queues: HashMap<String, PriorityQueue<T>>,
    /// Priority update strategies
    pub(super) update_strategies: HashMap<String, Box<dyn PriorityUpdateStrategy<T>>>,
    /// Priority calculation algorithms
    pub(super) calculation_algorithms: HashMap<String, Box<dyn PriorityCalculationAlgorithm<T>>>,
    /// Current update strategy
    pub(super) current_update_strategy: String,
    /// Priority history tracking
    pub(super) priority_history: PriorityHistoryTracker<T>,
    /// Dynamic adjustment controller
    pub(super) adjustment_controller: DynamicAdjustmentController<T>,
    /// Priority analytics
    pub(super) analytics: PriorityAnalytics<T>,
    /// Manager configuration
    pub(super) config: PriorityManagerConfig<T>,
    /// Manager statistics
    pub(super) stats: PriorityManagerStatistics<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> PriorityManager<T> {
    /// Create new priority manager.
    ///
    /// Registers the built-in [`StaticPriorityStrategy`] (as `"static"`) and
    /// [`WeightedSumPriorityCalculator`] (as `"weighted_sum"`), and selects
    /// `"static"` as the active update strategy. Previously
    /// `update_strategies`/`calculation_algorithms` were left empty with no
    /// way to populate them (no `register_*` method existed), and
    /// `current_update_strategy` was set to `"default"`, a key nothing ever
    /// registered -- so [`Self::update_priorities`] silently updated
    /// nothing on every call while still reporting success.
    pub fn new(config: PriorityManagerConfig<T>) -> Result<Self> {
        let mut update_strategies: HashMap<String, Box<dyn PriorityUpdateStrategy<T>>> =
            HashMap::new();
        update_strategies.insert("static".to_string(), Box::new(StaticPriorityStrategy));
        let mut calculation_algorithms: HashMap<String, Box<dyn PriorityCalculationAlgorithm<T>>> =
            HashMap::new();
        calculation_algorithms.insert(
            "weighted_sum".to_string(),
            Box::new(WeightedSumPriorityCalculator),
        );
        Ok(Self {
            priority_queues: HashMap::new(),
            update_strategies,
            calculation_algorithms,
            current_update_strategy: "static".to_string(),
            priority_history: PriorityHistoryTracker::new()?,
            adjustment_controller: DynamicAdjustmentController::new()?,
            analytics: PriorityAnalytics::new()?,
            config,
            stats: PriorityManagerStatistics::default(),
        })
    }
    /// Register (or replace) a priority update strategy under `name`.
    pub fn register_update_strategy(
        &mut self,
        name: impl Into<String>,
        strategy: Box<dyn PriorityUpdateStrategy<T>>,
    ) {
        self.update_strategies.insert(name.into(), strategy);
    }
    /// Register (or replace) a priority calculation algorithm under `name`.
    pub fn register_calculation_algorithm(
        &mut self,
        name: impl Into<String>,
        algorithm: Box<dyn PriorityCalculationAlgorithm<T>>,
    ) {
        self.calculation_algorithms.insert(name.into(), algorithm);
    }
    /// Select the active update strategy by name. Errors (rather than
    /// silently leaving the previous, possibly-still-invalid selection) if
    /// `name` was never registered.
    pub fn set_update_strategy(&mut self, name: &str) -> Result<()> {
        if !self.update_strategies.contains_key(name) {
            return Err(OptimError::InvalidConfig(format!(
                "priority update strategy '{name}' is not registered"
            )));
        }
        self.current_update_strategy = name.to_string();
        Ok(())
    }
    /// Calculate a task's priority using the registered algorithm `name`.
    pub fn calculate_priority(
        &self,
        task_context: &TaskContext<T>,
        algorithm_name: &str,
        weights: &PriorityWeights<T>,
    ) -> Result<PriorityLevel<T>> {
        let algorithm = self
            .calculation_algorithms
            .get(algorithm_name)
            .ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "priority calculation algorithm '{algorithm_name}' is not registered"
                ))
            })?;
        algorithm.calculate_priority(task_context, weights)
    }
    /// Add task to priority queue
    pub fn add_task(
        &mut self,
        task_id: String,
        priority: PriorityLevel<T>,
        queue_name: &str,
    ) -> Result<()> {
        let queue = self
            .priority_queues
            .entry(queue_name.to_string())
            .or_insert_with(|| PriorityQueue::new());
        let item = PriorityItem {
            task_id,
            priority,
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            version: 1,
        };
        queue.push(item)?;
        Ok(())
    }
    /// Get next highest priority task
    pub fn get_next_task(&mut self, queue_name: &str) -> Option<PriorityItem<T>> {
        self.priority_queues.get_mut(queue_name)?.pop()
    }
    /// Update priorities for all tasks using the active update strategy.
    ///
    /// Errors if `current_update_strategy` does not name a registered
    /// strategy, rather than silently doing nothing while still reporting
    /// success via `stats.total_updates` (the previous behavior for every
    /// call, since nothing was ever registered under the old default key).
    pub fn update_priorities(&mut self, context: PriorityUpdateContext<T>) -> Result<()> {
        let strategy = self
            .update_strategies
            .get_mut(&self.current_update_strategy)
            .ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "priority update strategy '{}' is not registered",
                    self.current_update_strategy
                ))
            })?;
        for queue in self.priority_queues.values_mut() {
            let mut items = queue.drain();
            strategy.update_priorities(&mut items, &context)?;
            for item in items {
                queue.push(item)?;
            }
        }
        self.stats.total_updates += 1;
        Ok(())
    }
    /// Get manager statistics
    pub fn get_statistics(&self) -> &PriorityManagerStatistics<T> {
        &self.stats
    }
}
/// Priority statistics
#[derive(Debug, Clone)]
pub struct PriorityStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Average priority
    pub average_priority: T,
    /// Priority variance
    pub priority_variance: T,
    /// Priority distribution
    pub distribution: HashMap<String, usize>,
    /// Change frequency
    pub change_frequency: T,
    /// Stability metric
    pub stability: T,
    /// Effectiveness metric
    pub effectiveness: T,
}
/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch<T: Float + Debug + Send + Sync + 'static> {
    /// Matched template
    pub template_id: String,
    /// Match confidence
    pub confidence: T,
    /// Match details
    pub match_details: HashMap<String, T>,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}
/// Trend direction
#[derive(Debug, Clone, Copy)]
pub enum TrendDirection {
    /// Increasing priority
    Increasing,
    /// Decreasing priority
    Decreasing,
    /// Stable priority
    Stable,
    /// Oscillating priority
    Oscillating,
    /// Irregular changes
    Irregular,
}
/// Priority history tracker
#[derive(Debug)]
pub struct PriorityHistoryTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Priority change history
    pub(super) change_history: VecDeque<PriorityChangeRecord<T>>,
    /// Statistical summaries
    pub(super) statistical_summaries: HashMap<String, PriorityStatistics<T>>,
    /// Trend analysis
    pub(super) trend_analyzer: PriorityTrendAnalyzer<T>,
    /// Change detection algorithms
    pub(super) change_detectors: Vec<Box<dyn PriorityChangeDetector<T>>>,
    /// History configuration
    pub(super) config: HistoryConfig,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> PriorityHistoryTracker<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            change_history: VecDeque::new(),
            statistical_summaries: HashMap::new(),
            trend_analyzer: PriorityTrendAnalyzer::new()?,
            change_detectors: Vec::new(),
            config: HistoryConfig::default(),
        })
    }
}
