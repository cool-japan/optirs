// State management for transformer-based optimizer

use super::config::TransformerBasedOptimizerConfig;
use super::meta_learning::MetaState;
use crate::common::cast_scalar;
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant, SystemTime};

/// Transformer optimizer state
pub struct TransformerOptimizerState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Current model parameters
    pub current_parameters: Array1<T>,

    /// Parameter history
    parameter_history: ParameterHistory<T>,

    /// Optimization state
    optimization_state: OptimizationState<T>,

    /// Learning state
    learning_state: LearningState<T>,

    /// Memory state
    memory_state: MemoryState<T>,

    /// Checkpoint manager
    checkpoint_manager: CheckpointManager<T>,

    /// State configuration
    config: StateConfig,

    /// State statistics
    statistics: StateStatistics<T>,

    /// State version for tracking changes
    version: usize,

    /// Creation timestamp
    created_at: std::time::Instant,

    /// Last update timestamp
    last_updated: std::time::Instant,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    TransformerOptimizerState<T>
{
    /// Create new optimizer state
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let parameter_count = config.model_dimension * config.num_transformer_layers;
        let current_parameters = Array1::zeros(parameter_count);

        let parameter_history = ParameterHistory::new(1000, parameter_count)?;
        let optimization_state = OptimizationState::new(config)?;
        let learning_state = LearningState::new(config)?;
        let memory_state = MemoryState::new()?;
        let checkpoint_manager = CheckpointManager::new(config)?;
        let state_config = StateConfig::from_optimizer_config(config);
        let statistics = StateStatistics::new();

        let now = std::time::Instant::now();

        Ok(Self {
            current_parameters,
            parameter_history,
            optimization_state,
            learning_state,
            memory_state,
            checkpoint_manager,
            config: state_config,
            statistics,
            version: 0,
            created_at: now,
            last_updated: now,
        })
    }

    /// Update state with optimization step.
    ///
    /// `update` need not already match `current_parameters`'s length: the
    /// meta-learned update network produces one "layer's worth" of update
    /// (`model_dimension` elements), which is applied identically to every
    /// transformer-layer segment of the (`model_dimension *
    /// num_transformer_layers`)-sized parameter vector. See
    /// `Self::project_update`.
    pub fn update_with_step(&mut self, update: &Array1<T>, loss: Option<T>) -> Result<()> {
        let projected = Self::project_update(update, self.current_parameters.len())?;
        let update = &projected;

        // Apply parameter update
        self.current_parameters = &self.current_parameters + update;

        // Record parameter history
        self.parameter_history
            .record_parameters(&self.current_parameters)?;

        // Update optimization state
        self.optimization_state.update_with_step(update, loss)?;

        // Update learning state
        if let Some(loss_val) = loss {
            self.learning_state.update_with_loss(loss_val)?;
        }

        // Update statistics
        self.statistics.record_update(update, loss);

        // Increment version and update timestamp
        self.version += 1;
        self.last_updated = std::time::Instant::now();

        Ok(())
    }

    /// Reconcile an update vector's length with the parameter vector it must
    /// be added to.
    ///
    /// * If the lengths already match, the update is used as-is.
    /// * If `target_len` is an exact multiple of `update.len()` (the normal
    ///   case: a per-layer update tiled across every transformer layer), the
    ///   update is repeated to fill `target_len`.
    /// * Any other mismatch is a genuine configuration inconsistency and is
    ///   reported as an error instead of panicking inside the `ndarray`
    ///   addition (`current_parameters + update` panics on shape mismatch).
    fn project_update(update: &Array1<T>, target_len: usize) -> Result<Array1<T>> {
        let update_len = update.len();
        if update_len == target_len {
            return Ok(update.clone());
        }
        if update_len == 0 || !target_len.is_multiple_of(update_len) {
            return Err(OptimError::InvalidConfig(format!(
                "optimizer update has {update_len} elements, which cannot be \
                 tiled to fill the {target_len}-element parameter vector \
                 (expected {update_len} to evenly divide {target_len})"
            )));
        }
        let repeats = target_len / update_len;
        let mut tiled = Vec::with_capacity(target_len);
        for _ in 0..repeats {
            tiled.extend(update.iter().copied());
        }
        Ok(Array1::from_vec(tiled))
    }

    /// Create state snapshot
    pub fn create_snapshot(&self) -> Result<OptimizerStateSnapshot<T>> {
        Ok(OptimizerStateSnapshot {
            parameters: self.current_parameters.clone(),
            optimization_state: self.optimization_state.clone(),
            learning_state: self.learning_state.clone(),
            memory_state: self.memory_state.clone(),
            version: self.version,
            timestamp: self.last_updated,
            metadata: SnapshotMetadata {
                parameter_count: self.current_parameters.len(),
                total_updates: self.statistics.total_updates,
                session_duration: self.last_updated.duration_since(self.created_at),
            },
        })
    }

    /// Restore from snapshot
    pub fn restore_from_snapshot(&mut self, snapshot: &OptimizerStateSnapshot<T>) -> Result<()> {
        self.current_parameters = snapshot.parameters.clone();
        self.optimization_state = snapshot.optimization_state.clone();
        self.learning_state = snapshot.learning_state.clone();
        self.memory_state = snapshot.memory_state.clone();
        self.version = snapshot.version;
        self.last_updated = snapshot.timestamp;

        Ok(())
    }

    /// Save checkpoint
    pub fn save_checkpoint(&mut self, name: String) -> Result<String> {
        let snapshot = self.create_snapshot()?;
        let checkpoint_id = self.checkpoint_manager.save_checkpoint(name, snapshot)?;
        Ok(checkpoint_id)
    }

    /// Load checkpoint
    pub fn load_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        let snapshot = self.checkpoint_manager.load_checkpoint(checkpoint_id)?;
        self.restore_from_snapshot(&snapshot)?;
        Ok(())
    }

    /// Get parameter statistics
    pub fn get_parameter_stats(&self) -> ParameterStatistics<T> {
        self.parameter_history.get_statistics()
    }

    /// Get optimization progress
    pub fn get_optimization_progress(&self) -> OptimizationProgress<T> {
        self.optimization_state.get_progress()
    }

    /// Get learning statistics
    pub fn get_learning_stats(&self) -> LearningStatistics<T> {
        self.learning_state.get_statistics()
    }

    /// Reset state
    pub fn reset(&mut self) -> Result<()> {
        self.current_parameters.fill(T::zero());
        self.parameter_history.clear();
        self.optimization_state.reset()?;
        self.learning_state.reset()?;
        self.memory_state.reset()?;
        self.statistics.reset();
        self.version = 0;
        self.last_updated = std::time::Instant::now();
        Ok(())
    }

    /// Validate state consistency
    pub fn validate_state(&self) -> Result<StateValidationReport> {
        let mut issues = Vec::new();

        // Check parameter validity
        if self.current_parameters.iter().any(|&x| !x.is_finite()) {
            issues.push("Invalid parameters detected (NaN or infinity)".to_string());
        }

        // Check version consistency
        if self.version == 0 && self.statistics.total_updates > 0 {
            issues.push("Version mismatch with update count".to_string());
        }

        // Check timestamp consistency
        if self.last_updated < self.created_at {
            issues.push("Invalid timestamp ordering".to_string());
        }

        // Validate optimization state
        let opt_validation = self.optimization_state.validate()?;
        issues.extend(opt_validation.issues);

        // Validate learning state
        let learning_validation = self.learning_state.validate()?;
        issues.extend(learning_validation.issues);

        Ok(StateValidationReport {
            is_valid: issues.is_empty(),
            issues,
            validation_timestamp: std::time::Instant::now(),
        })
    }

    /// Get state summary
    pub fn get_state_summary(&self) -> StateSummary<T> {
        StateSummary {
            version: self.version,
            parameter_count: self.current_parameters.len(),
            parameter_norm: self.compute_parameter_norm(),
            total_updates: self.statistics.total_updates,
            session_duration: self.last_updated.duration_since(self.created_at),
            last_update_magnitude: self.statistics.last_update_magnitude,
            average_loss: self.learning_state.get_average_loss(),
            convergence_rate: self.learning_state.get_convergence_rate(),
            memory_usage: self.memory_state.get_total_usage(),
            checkpoint_count: self.checkpoint_manager.get_checkpoint_count(),
        }
    }

    /// Compute parameter norm
    fn compute_parameter_norm(&self) -> T {
        self.current_parameters
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }

    /// Get state metadata
    pub fn get_metadata(&self) -> StateMetadata {
        StateMetadata {
            version: self.version,
            created_at: SystemTime::now(), // Convert from Instant for serialization
            last_updated: SystemTime::now(), // Convert from Instant for serialization
            total_updates: self.statistics.total_updates,
            configuration: self.config.clone(),
        }
    }

    /// Export state to serializable format
    pub fn export_state(&self) -> Result<SerializableState<T>> {
        Ok(SerializableState {
            parameters: self.current_parameters.to_vec(),
            parameter_shape: self.current_parameters.shape().to_vec(),
            optimization_state: self.optimization_state.to_serializable()?,
            learning_state: self.learning_state.to_serializable()?,
            metadata: self.get_metadata(),
            statistics: self.statistics.clone(),
        })
    }

    /// Import state from serializable format
    pub fn import_state(&mut self, state: SerializableState<T>) -> Result<()> {
        // Reconstruct parameters
        if state.parameter_shape.len() != 1 {
            return Err(crate::error::OptimError::Other(
                "Invalid parameter shape for 1D array".to_string(),
            ));
        }

        self.current_parameters = Array1::from_vec(state.parameters);

        // Restore other state components
        self.optimization_state
            .from_serializable(state.optimization_state)?;
        self.learning_state
            .from_serializable(state.learning_state)?;
        self.statistics = state.statistics;
        self.version = state.metadata.version;
        self.last_updated = Instant::now(); // Convert from SystemTime for internal use

        Ok(())
    }
}

/// Parameter history management
pub struct ParameterHistory<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Parameter snapshots
    snapshots: VecDeque<ParameterSnapshot<T>>,

    /// Maximum history size
    max_size: usize,

    /// Parameter dimension
    parameter_dimension: usize,

    /// Statistics
    statistics: ParameterStatistics<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    ParameterHistory<T>
{
    pub fn new(max_size: usize, parameter_dimension: usize) -> Result<Self> {
        Ok(Self {
            snapshots: VecDeque::new(),
            max_size,
            parameter_dimension,
            statistics: ParameterStatistics::new(),
        })
    }

    /// Record a parameter snapshot.
    ///
    /// # Errors
    /// Returns `Err` when `parameters` is not `parameter_dimension` long. The
    /// history is later averaged and differenced across snapshots, so a
    /// mixed-width history silently produces meaningless statistics; the
    /// declared dimension used to be stored and never checked against anything.
    pub fn record_parameters(&mut self, parameters: &Array1<T>) -> Result<()> {
        if parameters.len() != self.parameter_dimension {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "ParameterHistory holds {}-dimensional snapshots but was given {}",
                self.parameter_dimension,
                parameters.len()
            )));
        }
        let snapshot = ParameterSnapshot {
            parameters: parameters.clone(),
            timestamp: std::time::Instant::now(),
            norm: parameters
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |acc, x| acc + x)
                .sqrt(),
        };

        self.snapshots.push_back(snapshot.clone());
        if self.snapshots.len() > self.max_size {
            self.snapshots.pop_front();
        }

        self.statistics.update_with_snapshot(&snapshot);
        Ok(())
    }

    pub fn get_recent_parameters(&self, count: usize) -> Vec<Array1<T>> {
        self.snapshots
            .iter()
            .rev()
            .take(count)
            .map(|snapshot| snapshot.parameters.clone())
            .collect()
    }

    pub fn get_statistics(&self) -> ParameterStatistics<T> {
        self.statistics.clone()
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.statistics = ParameterStatistics::new();
    }
}

/// Optimization state tracking
#[derive(Debug, Clone)]
pub struct OptimizationState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Current learning rate
    pub learning_rate: T,

    /// Momentum state
    pub momentum: Option<Array1<T>>,

    /// Adaptive learning rate state (e.g., Adam)
    pub adaptive_state: Option<AdaptiveState<T>>,

    /// Gradient accumulation
    pub gradient_accumulator: GradientAccumulator<T>,

    /// Step count
    pub step_count: usize,

    /// Last update magnitude
    pub last_update_magnitude: T,

    /// Convergence tracking
    pub convergence_tracker: ConvergenceTracker<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    OptimizationState<T>
{
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let parameter_count = config.model_dimension * config.num_transformer_layers;

        Ok(Self {
            learning_rate: config.learning_rate,
            momentum: None,
            adaptive_state: Some(AdaptiveState::new(parameter_count)?),
            gradient_accumulator: GradientAccumulator::new(parameter_count)?,
            step_count: 0,
            last_update_magnitude: T::zero(),
            convergence_tracker: ConvergenceTracker::new(),
        })
    }

    pub fn update_with_step(&mut self, update: &Array1<T>, loss: Option<T>) -> Result<()> {
        self.step_count += 1;
        self.last_update_magnitude = update
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt();

        if let Some(loss_val) = loss {
            self.convergence_tracker.record_loss(loss_val);
        }

        // Update adaptive state
        if let Some(ref mut adaptive) = self.adaptive_state {
            adaptive.update_with_step(update)?;
        }

        Ok(())
    }

    pub fn get_progress(&self) -> OptimizationProgress<T> {
        OptimizationProgress {
            step_count: self.step_count,
            current_learning_rate: self.learning_rate,
            last_update_magnitude: self.last_update_magnitude,
            convergence_rate: self.convergence_tracker.get_convergence_rate(),
            stability_score: self.convergence_tracker.get_stability_score(),
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.step_count = 0;
        self.last_update_magnitude = T::zero();
        self.convergence_tracker.reset();

        if let Some(ref mut adaptive) = self.adaptive_state {
            adaptive.reset()?;
        }

        self.gradient_accumulator.reset()?;
        Ok(())
    }

    pub fn validate(&self) -> Result<ValidationResult> {
        let mut issues = Vec::new();

        if self.learning_rate <= T::zero() {
            issues.push("Invalid learning rate".to_string());
        }

        if !self.last_update_magnitude.is_finite() {
            issues.push("Invalid update magnitude".to_string());
        }

        Ok(ValidationResult { issues })
    }

    pub fn to_serializable(&self) -> Result<SerializableOptimizationState<T>> {
        Ok(SerializableOptimizationState {
            learning_rate: self.learning_rate,
            step_count: self.step_count,
            last_update_magnitude: self.last_update_magnitude,
            momentum: self.momentum.as_ref().map(|m| m.to_vec()),
            convergence_metrics: self.convergence_tracker.to_serializable(),
        })
    }

    pub fn from_serializable(&mut self, state: SerializableOptimizationState<T>) -> Result<()> {
        self.learning_rate = state.learning_rate;
        self.step_count = state.step_count;
        self.last_update_magnitude = state.last_update_magnitude;

        if let Some(momentum_vec) = state.momentum {
            self.momentum = Some(Array1::from_vec(momentum_vec));
        }

        self.convergence_tracker
            .from_serializable(state.convergence_metrics)?;
        Ok(())
    }
}

/// Learning state tracking
#[derive(Debug, Clone)]
pub struct LearningState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Loss history
    loss_history: VecDeque<T>,

    /// Meta-learning state
    meta_state: Option<MetaState<T>>,

    /// Task adaptation history
    adaptation_history: VecDeque<TaskAdaptationRecord<T>>,

    /// Learning rate schedule
    learning_schedule: LearningSchedule<T>,

    /// Performance metrics
    performance_metrics: LearningPerformanceMetrics<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    LearningState<T>
{
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let meta_state = Some(MetaState::new(config.model_dimension)?);
        let learning_schedule = LearningSchedule::new(config.learning_rate, config.warmup_steps);

        Ok(Self {
            loss_history: VecDeque::new(),
            meta_state,
            adaptation_history: VecDeque::new(),
            learning_schedule,
            performance_metrics: LearningPerformanceMetrics::new(),
        })
    }

    pub fn update_with_loss(&mut self, loss: T) -> Result<()> {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 1000 {
            self.loss_history.pop_front();
        }

        self.performance_metrics.record_loss(loss);
        self.learning_schedule.step();

        if let Some(ref mut meta) = self.meta_state {
            meta.update_loss_history(loss);
        }

        Ok(())
    }

    /// Learning rate the schedule is currently at.
    pub fn current_learning_rate(&self) -> T {
        self.learning_schedule.current_rate
    }

    /// The learning-rate schedule this state advances on every recorded loss.
    pub fn learning_schedule(&self) -> &LearningSchedule<T> {
        &self.learning_schedule
    }

    pub fn get_statistics(&self) -> LearningStatistics<T> {
        LearningStatistics {
            total_episodes: self.loss_history.len(),
            average_loss: self.get_average_loss(),
            best_loss: self.get_best_loss(),
            convergence_rate: self.get_convergence_rate(),
            learning_stability: self.performance_metrics.get_stability_score(),
        }
    }

    pub fn get_average_loss(&self) -> T {
        let Some(count) = cast_scalar::<T, _>(self.loss_history.len())
            .ok()
            .filter(|c| *c > T::zero())
        else {
            return T::zero();
        };
        self.loss_history
            .iter()
            .fold(T::zero(), |acc, &loss| acc + loss)
            / count
    }

    pub fn get_best_loss(&self) -> T {
        self.loss_history
            .iter()
            .fold(T::infinity(), |min, &loss| min.min(loss))
    }

    pub fn get_convergence_rate(&self) -> T {
        if self.loss_history.len() < 2 {
            return T::zero();
        }

        let recent_losses: Vec<_> = self.loss_history.iter().rev().take(10).cloned().collect();
        if recent_losses.len() < 2 {
            return T::zero();
        }

        // `recent_losses.len() >= 2` was checked above; the fallible reads keep
        // that reasoning next to the accesses it justifies.
        let (Some(&initial), Some(&final_loss)) = (recent_losses.last(), recent_losses.first())
        else {
            return T::zero();
        };

        if initial > T::zero() {
            (initial - final_loss) / initial
        } else {
            T::zero()
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.loss_history.clear();
        self.adaptation_history.clear();
        self.performance_metrics.reset();

        if let Some(ref mut meta) = self.meta_state {
            *meta = MetaState::new(meta.get_parameters().len())?;
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<ValidationResult> {
        let mut issues = Vec::new();

        if self.loss_history.iter().any(|&loss| !loss.is_finite()) {
            issues.push("Invalid loss values detected".to_string());
        }

        Ok(ValidationResult { issues })
    }

    pub fn to_serializable(&self) -> Result<SerializableLearningState<T>> {
        Ok(SerializableLearningState {
            loss_history: self.loss_history.iter().cloned().collect(),
            average_loss: self.get_average_loss(),
            best_loss: self.get_best_loss(),
            convergence_rate: self.get_convergence_rate(),
        })
    }

    pub fn from_serializable(&mut self, state: SerializableLearningState<T>) -> Result<()> {
        self.loss_history = VecDeque::from(state.loss_history);
        Ok(())
    }
}

/// Memory state management
#[derive(Debug, Clone)]
pub struct MemoryState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Attention caches
    attention_caches: HashMap<String, AttentionCache<T>>,

    /// Memory usage tracking
    memory_usage: MemoryUsageTracker,

    /// Cache statistics
    cache_statistics: CacheStatistics,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    MemoryState<T>
{
    pub fn new() -> Result<Self> {
        Ok(Self {
            attention_caches: HashMap::new(),
            memory_usage: MemoryUsageTracker::new(),
            cache_statistics: CacheStatistics::new(),
        })
    }

    pub fn get_total_usage(&self) -> usize {
        self.memory_usage.total_usage
    }

    pub fn reset(&mut self) -> Result<()> {
        self.attention_caches.clear();
        self.memory_usage.reset();
        self.cache_statistics.reset();
        Ok(())
    }
}

/// Explicitly-saved checkpoints retained before the oldest is evicted. The
/// optimizer configuration carries no checkpoint-retention setting, so this is a
/// documented default rather than something derived from it.
const DEFAULT_MAX_CHECKPOINTS: usize = 10;

/// Name prefix that marks a checkpoint as produced by
/// [`CheckpointManager::maybe_auto_save`] rather than an explicit save, so the
/// two are retained under separate limits.
const AUTO_SAVE_PREFIX: &str = "auto_";

/// Checkpoint management
pub struct CheckpointManager<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Stored checkpoints
    checkpoints: HashMap<String, OptimizerStateSnapshot<T>>,

    /// Checkpoint metadata
    metadata: HashMap<String, CheckpointMetadata>,

    /// Maximum checkpoints to keep
    max_checkpoints: usize,

    /// Auto-save configuration
    auto_save_config: AutoSaveConfig,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    CheckpointManager<T>
{
    /// Build a checkpoint manager whose auto-save cadence follows `config`.
    ///
    /// The auto-save frequency used to be a fixed 100 steps regardless of the
    /// optimizer's configuration; it now tracks
    /// [`StateConfig::from_optimizer_config`], which derives it from
    /// `performance_config.metrics_interval`.
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let state_config = StateConfig::from_optimizer_config(config);
        Ok(Self {
            checkpoints: HashMap::new(),
            metadata: HashMap::new(),
            max_checkpoints: DEFAULT_MAX_CHECKPOINTS,
            auto_save_config: AutoSaveConfig {
                enabled: state_config.auto_save_enabled,
                frequency: state_config.checkpoint_frequency,
                ..AutoSaveConfig::default()
            },
        })
    }

    /// The auto-save policy this manager was built with.
    pub fn auto_save_config(&self) -> &AutoSaveConfig {
        &self.auto_save_config
    }

    /// Save `snapshot` if `step` falls on the configured auto-save cadence.
    ///
    /// Returns the new checkpoint id, or `None` when auto-save is disabled or
    /// `step` is not a save point. Auto-saves are named `auto_<step>` and are
    /// capped at [`AutoSaveConfig::max_auto_saves`] independently of the
    /// explicit-checkpoint limit, oldest evicted first, so a long run cannot
    /// grow the store without bound. `step == 0` saves the initial state.
    ///
    /// Without this, `auto_save_config` was populated at construction and never
    /// read by anything: the auto-save policy existed only as a value.
    pub fn maybe_auto_save(
        &mut self,
        step: usize,
        snapshot: OptimizerStateSnapshot<T>,
    ) -> Result<Option<String>> {
        if !self.auto_save_config.enabled
            || self.auto_save_config.frequency == 0
            || !step.is_multiple_of(self.auto_save_config.frequency)
        {
            return Ok(None);
        }
        let max_auto_saves = self.auto_save_config.max_auto_saves;
        let id = self.save_checkpoint(format!("{AUTO_SAVE_PREFIX}{step}"), snapshot)?;
        self.evict_surplus_auto_saves(max_auto_saves);
        Ok(Some(id))
    }

    /// Number of auto-saved checkpoints currently retained.
    pub fn auto_save_count(&self) -> usize {
        self.metadata
            .values()
            .filter(|m| m.name.starts_with(AUTO_SAVE_PREFIX))
            .count()
    }

    /// Drop the oldest auto-saves until at most `max_auto_saves` remain.
    /// Explicit checkpoints are untouched.
    fn evict_surplus_auto_saves(&mut self, max_auto_saves: usize) {
        loop {
            let mut autos: Vec<(String, std::time::Instant)> = self
                .metadata
                .iter()
                .filter(|(_, m)| m.name.starts_with(AUTO_SAVE_PREFIX))
                .map(|(id, m)| (id.clone(), m.created_at))
                .collect();
            if autos.len() <= max_auto_saves {
                return;
            }
            autos.sort_by_key(|(_, created)| *created);
            match autos.first() {
                Some((oldest, _)) => {
                    let oldest = oldest.clone();
                    self.checkpoints.remove(&oldest);
                    self.metadata.remove(&oldest);
                }
                None => return,
            }
        }
    }

    pub fn save_checkpoint(
        &mut self,
        name: String,
        snapshot: OptimizerStateSnapshot<T>,
    ) -> Result<String> {
        let checkpoint_id = format!("{}_{}", name, snapshot.version);

        // Add metadata
        let metadata = CheckpointMetadata {
            id: checkpoint_id.clone(),
            name: name.clone(),
            created_at: std::time::Instant::now(),
            size_estimate: snapshot.parameters.len() * std::mem::size_of::<T>(),
            description: format!("Checkpoint at version {}", snapshot.version),
        };

        self.checkpoints.insert(checkpoint_id.clone(), snapshot);
        self.metadata.insert(checkpoint_id.clone(), metadata);

        // Cleanup old checkpoints if necessary
        if self.checkpoints.len() > self.max_checkpoints {
            self.cleanup_old_checkpoints()?;
        }

        Ok(checkpoint_id)
    }

    pub fn load_checkpoint(&self, checkpoint_id: &str) -> Result<OptimizerStateSnapshot<T>> {
        self.checkpoints.get(checkpoint_id).cloned().ok_or_else(|| {
            crate::error::OptimError::Other(format!("Checkpoint {} not found", checkpoint_id))
        })
    }

    pub fn list_checkpoints(&self) -> Vec<CheckpointMetadata> {
        self.metadata.values().cloned().collect()
    }

    pub fn delete_checkpoint(&mut self, checkpoint_id: &str) -> Result<bool> {
        let removed_checkpoint = self.checkpoints.remove(checkpoint_id).is_some();
        let removed_metadata = self.metadata.remove(checkpoint_id).is_some();
        Ok(removed_checkpoint && removed_metadata)
    }

    pub fn get_checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        // Remove oldest checkpoints if over limit
        while self.checkpoints.len() > self.max_checkpoints {
            if let Some((oldest_id, _)) = self
                .metadata
                .iter()
                .min_by_key(|(_, metadata)| metadata.created_at)
                .map(|(id, metadata)| (id.clone(), metadata.clone()))
            {
                self.checkpoints.remove(&oldest_id);
                self.metadata.remove(&oldest_id);
            } else {
                break;
            }
        }
        Ok(())
    }
}

/// Supporting data structures and types

#[derive(Debug, Clone)]
pub struct ParameterSnapshot<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub parameters: Array1<T>,
    pub timestamp: std::time::Instant,
    pub norm: T,
}

#[derive(Debug, Clone)]
pub struct ParameterStatistics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub total_snapshots: usize,
    pub average_norm: T,
    pub max_norm: T,
    pub min_norm: T,
    pub norm_trend: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for ParameterStatistics<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    ParameterStatistics<T>
{
    pub fn new() -> Self {
        Self {
            total_snapshots: 0,
            average_norm: T::zero(),
            max_norm: T::zero(),
            min_norm: T::infinity(),
            norm_trend: T::zero(),
        }
    }

    pub fn update_with_snapshot(&mut self, snapshot: &ParameterSnapshot<T>) {
        self.total_snapshots += 1;
        self.average_norm = (self.average_norm
            * scirs2_core::numeric::NumCast::from(self.total_snapshots - 1)
                .unwrap_or_else(|| T::zero())
            + snapshot.norm)
            / scirs2_core::numeric::NumCast::from(self.total_snapshots)
                .unwrap_or_else(|| T::zero());
        self.max_norm = self.max_norm.max(snapshot.norm);
        self.min_norm = self.min_norm.min(snapshot.norm);
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// First moment estimates
    pub m: Array1<T>,
    /// Second moment estimates
    pub v: Array1<T>,
    /// Step count for bias correction
    pub step_count: usize,
    /// Beta parameters
    pub beta1: T,
    pub beta2: T,
    /// Epsilon for numerical stability
    pub epsilon: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    AdaptiveState<T>
{
    pub fn new(parameter_count: usize) -> Result<Self> {
        Ok(Self {
            m: Array1::zeros(parameter_count),
            v: Array1::zeros(parameter_count),
            step_count: 0,
            beta1: scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero()),
            beta2: scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero()),
            epsilon: scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero()),
        })
    }

    pub fn update_with_step(&mut self, _update: &Array1<T>) -> Result<()> {
        self.step_count += 1;
        // Adam-style update logic would go here
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.m.fill(T::zero());
        self.v.fill(T::zero());
        self.step_count = 0;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GradientAccumulator<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Accumulated gradients
    pub accumulated_gradients: Array1<T>,
    /// Accumulation count
    pub accumulation_count: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    GradientAccumulator<T>
{
    pub fn new(parameter_count: usize) -> Result<Self> {
        Ok(Self {
            accumulated_gradients: Array1::zeros(parameter_count),
            accumulation_count: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.accumulated_gradients.fill(T::zero());
        self.accumulation_count = 0;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ConvergenceTracker<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Recent loss values
    recent_losses: VecDeque<T>,
    /// Convergence threshold
    convergence_threshold: T,
    /// Stability window size
    stability_window: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for ConvergenceTracker<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    ConvergenceTracker<T>
{
    pub fn new() -> Self {
        Self {
            recent_losses: VecDeque::new(),
            convergence_threshold: scirs2_core::numeric::NumCast::from(1e-6)
                .unwrap_or_else(|| T::zero()),
            stability_window: 10,
        }
    }

    pub fn record_loss(&mut self, loss: T) {
        self.recent_losses.push_back(loss);
        if self.recent_losses.len() > self.stability_window {
            self.recent_losses.pop_front();
        }
    }

    /// Whether the tracked stream has converged: the mean absolute step-to-step
    /// loss change over the stability window is at or below
    /// `convergence_threshold`.
    ///
    /// Returns `false` until the window has at least two observations. The
    /// threshold used to be stored at construction and consulted by nothing, so
    /// the tracker could report a rate and a stability score but never a verdict.
    pub fn has_converged(&self) -> bool {
        if self.recent_losses.len() < 2 {
            return false;
        }
        let mut total = T::zero();
        for pair in self.recent_losses.iter().collect::<Vec<_>>().windows(2) {
            total = total + (*pair[1] - *pair[0]).abs();
        }
        let steps: T = scirs2_core::numeric::NumCast::from(self.recent_losses.len() - 1)
            .unwrap_or_else(|| T::one());
        (total / steps) <= self.convergence_threshold
    }

    /// The configured convergence threshold.
    pub fn convergence_threshold(&self) -> T {
        self.convergence_threshold
    }

    pub fn get_convergence_rate(&self) -> T {
        if self.recent_losses.len() < 2 {
            return T::zero();
        }

        let (Some(&first), Some(&last)) = (self.recent_losses.front(), self.recent_losses.back())
        else {
            return T::zero();
        };

        if first > T::zero() {
            (first - last) / first
        } else {
            T::zero()
        }
    }

    pub fn get_stability_score(&self) -> T {
        if self.recent_losses.len() < 2 {
            return T::zero();
        }

        let Some(count) = cast_scalar::<T, _>(self.recent_losses.len())
            .ok()
            .filter(|c| *c > T::zero())
        else {
            return T::zero();
        };
        let mean = self.recent_losses.iter().fold(T::zero(), |acc, &x| acc + x) / count;
        let variance = self
            .recent_losses
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / count;

        T::one() / (T::one() + variance.sqrt())
    }

    pub fn reset(&mut self) {
        self.recent_losses.clear();
    }

    pub fn to_serializable(&self) -> SerializableConvergenceState<T> {
        SerializableConvergenceState {
            recent_losses: self.recent_losses.iter().cloned().collect(),
            convergence_rate: self.get_convergence_rate(),
            stability_score: self.get_stability_score(),
        }
    }

    pub fn from_serializable(&mut self, state: SerializableConvergenceState<T>) -> Result<()> {
        self.recent_losses = VecDeque::from(state.recent_losses);
        Ok(())
    }
}

/// State snapshots and serialization
#[derive(Debug, Clone)]
pub struct OptimizerStateSnapshot<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub parameters: Array1<T>,
    pub optimization_state: OptimizationState<T>,
    pub learning_state: LearningState<T>,
    pub memory_state: MemoryState<T>,
    pub version: usize,
    pub timestamp: std::time::Instant,
    pub metadata: SnapshotMetadata,
}

#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    pub parameter_count: usize,
    pub total_updates: usize,
    pub session_duration: Duration,
}

/// Configuration and metadata structures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateConfig {
    pub max_history_size: usize,
    pub checkpoint_frequency: usize,
    pub auto_save_enabled: bool,
    pub validation_enabled: bool,
}

impl StateConfig {
    /// Derive the state-tracking configuration from the optimizer's own
    /// performance-tracking settings.
    ///
    /// Every field used to be a hard-coded literal, so a caller who shrank
    /// `performance_config.max_history_size` to bound memory still got a
    /// 1000-entry state history. `max_history_size` and `checkpoint_frequency`
    /// now follow the optimizer's configured history bound and metric cadence.
    /// `auto_save_enabled` and `validation_enabled` stay `true`: the optimizer
    /// config carries no corresponding switch, so there is nothing honest to
    /// derive them from.
    pub fn from_optimizer_config<
        T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
    >(
        config: &TransformerBasedOptimizerConfig<T>,
    ) -> Self {
        Self {
            max_history_size: config.performance_config.max_history_size,
            checkpoint_frequency: config.performance_config.metrics_interval.max(1),
            auto_save_enabled: true,
            validation_enabled: true,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateMetadata {
    pub version: usize,
    pub created_at: std::time::SystemTime,
    pub last_updated: std::time::SystemTime,
    pub total_updates: usize,
    pub configuration: StateConfig,
}

/// Statistics and tracking structures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateStatistics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub total_updates: usize,
    pub last_update_magnitude: T,
    pub average_update_magnitude: T,
    pub parameter_change_rate: T,
    pub update_frequency: f64,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for StateStatistics<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    StateStatistics<T>
{
    pub fn new() -> Self {
        Self {
            total_updates: 0,
            last_update_magnitude: T::zero(),
            average_update_magnitude: T::zero(),
            parameter_change_rate: T::zero(),
            update_frequency: 0.0,
        }
    }

    pub fn record_update(&mut self, update: &Array1<T>, _loss: Option<T>) {
        self.total_updates += 1;
        let magnitude = update
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt();
        self.last_update_magnitude = magnitude;
        self.average_update_magnitude = (self.average_update_magnitude
            * scirs2_core::numeric::NumCast::from(self.total_updates - 1)
                .unwrap_or_else(|| T::zero())
            + magnitude)
            / scirs2_core::numeric::NumCast::from(self.total_updates).unwrap_or_else(|| T::zero());
    }

    pub fn reset(&mut self) {
        self.total_updates = 0;
        self.last_update_magnitude = T::zero();
        self.average_update_magnitude = T::zero();
        self.parameter_change_rate = T::zero();
        self.update_frequency = 0.0;
    }
}

/// Validation and reporting structures
#[derive(Debug, Clone)]
pub struct StateValidationReport {
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub validation_timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub issues: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StateSummary<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub version: usize,
    pub parameter_count: usize,
    pub parameter_norm: T,
    pub total_updates: usize,
    pub session_duration: Duration,
    pub last_update_magnitude: T,
    pub average_loss: T,
    pub convergence_rate: T,
    pub memory_usage: usize,
    pub checkpoint_count: usize,
}

/// Serializable state structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub parameters: Vec<T>,
    pub parameter_shape: Vec<usize>,
    pub optimization_state: SerializableOptimizationState<T>,
    pub learning_state: SerializableLearningState<T>,
    pub metadata: StateMetadata,
    pub statistics: StateStatistics<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableOptimizationState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub learning_rate: T,
    pub step_count: usize,
    pub last_update_magnitude: T,
    pub momentum: Option<Vec<T>>,
    pub convergence_metrics: SerializableConvergenceState<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLearningState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub loss_history: Vec<T>,
    pub average_loss: T,
    pub best_loss: T,
    pub convergence_rate: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConvergenceState<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub recent_losses: Vec<T>,
    pub convergence_rate: T,
    pub stability_score: T,
}

/// Additional supporting structures
#[derive(Debug, Clone)]
pub struct TaskAdaptationRecord<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub task_id: String,
    pub adaptation_steps: usize,
    pub final_loss: T,
    pub adaptation_time: Duration,
}

#[derive(Debug, Clone)]
/// Linear-warmup-then-exponential-decay learning-rate schedule.
///
/// `initial_rate`, `warmup_steps` and `decay_factor` used to be stored and never
/// consulted: the schedule had no way to advance, so `current_rate` stayed at
/// `initial_rate` forever and [`LearningState`] held a schedule it never used.
pub struct LearningSchedule<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub initial_rate: T,
    pub current_rate: T,
    pub warmup_steps: usize,
    pub decay_factor: T,
    /// Steps taken so far, advanced by [`LearningSchedule::step`].
    steps_taken: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    LearningSchedule<T>
{
    pub fn new(initial_rate: T, warmup_steps: usize) -> Self {
        Self {
            initial_rate,
            current_rate: initial_rate,
            warmup_steps,
            decay_factor: scirs2_core::numeric::NumCast::from(0.95).unwrap_or_else(|| T::zero()),
            steps_taken: 0,
        }
    }

    /// Rate at `step`: `initial_rate · step / warmup_steps` while warming up,
    /// then `initial_rate · decay_factor^(step - warmup_steps)`.
    ///
    /// With `warmup_steps == 0` the warmup phase is skipped entirely.
    pub fn rate_at(&self, step: usize) -> T {
        if step < self.warmup_steps {
            let progress: T =
                scirs2_core::numeric::NumCast::from((step + 1) as f64 / self.warmup_steps as f64)
                    .unwrap_or_else(T::one);
            return self.initial_rate * progress;
        }
        let decayed: T = scirs2_core::numeric::NumCast::from((step - self.warmup_steps) as f64)
            .unwrap_or_else(T::zero);
        self.initial_rate * self.decay_factor.powf(decayed)
    }

    /// Advance one step and return the new rate.
    pub fn step(&mut self) -> T {
        self.current_rate = self.rate_at(self.steps_taken);
        self.steps_taken += 1;
        self.current_rate
    }

    /// Steps taken so far.
    pub fn steps_taken(&self) -> usize {
        self.steps_taken
    }
}

#[derive(Debug, Clone)]
pub struct LearningPerformanceMetrics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub loss_trend: T,
    pub convergence_stability: T,
    pub adaptation_efficiency: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for LearningPerformanceMetrics<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    LearningPerformanceMetrics<T>
{
    pub fn new() -> Self {
        Self {
            loss_trend: T::zero(),
            convergence_stability: T::zero(),
            adaptation_efficiency: T::zero(),
        }
    }

    pub fn record_loss(&mut self, _loss: T) {
        // Update performance metrics logic
    }

    pub fn get_stability_score(&self) -> T {
        self.convergence_stability
    }

    pub fn reset(&mut self) {
        self.loss_trend = T::zero();
        self.convergence_stability = T::zero();
        self.adaptation_efficiency = T::zero();
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationProgress<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub step_count: usize,
    pub current_learning_rate: T,
    pub last_update_magnitude: T,
    pub convergence_rate: T,
    pub stability_score: T,
}

#[derive(Debug, Clone)]
pub struct LearningStatistics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub total_episodes: usize,
    pub average_loss: T,
    pub best_loss: T,
    pub convergence_rate: T,
    pub learning_stability: T,
}

#[derive(Debug, Clone)]
pub struct AttentionCache<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub cached_keys: Array2<T>,
    pub cached_values: Array2<T>,
    pub cache_size: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryUsageTracker {
    pub total_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
}

impl Default for MemoryUsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryUsageTracker {
    pub fn new() -> Self {
        Self {
            total_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.total_usage = 0;
        self.peak_usage = 0;
        self.allocation_count = 0;
    }
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hit_count: usize,
    pub miss_count: usize,
    pub eviction_count: usize,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStatistics {
    pub fn new() -> Self {
        Self {
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.hit_count = 0;
        self.miss_count = 0;
        self.eviction_count = 0;
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    pub id: String,
    pub name: String,
    pub created_at: Instant,
    pub size_estimate: usize,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct AutoSaveConfig {
    pub enabled: bool,
    pub frequency: usize,
    pub max_auto_saves: usize,
}

impl Default for AutoSaveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: 100,
            max_auto_saves: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_state_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let state = TransformerOptimizerState::new(&config);
        assert!(state.is_ok());

        let s = state.expect("TransformerOptimizerState::new should succeed");
        assert_eq!(s.version, 0);
        assert!(!s.current_parameters.is_empty());
    }

    #[test]
    fn test_state_update() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let mut state = TransformerOptimizerState::new(&config)
            .expect("TransformerOptimizerState::new should succeed");

        let update = Array1::<f32>::ones(state.current_parameters.len());
        let result = state.update_with_step(&update, Some(1.5));
        assert!(result.is_ok());
        assert_eq!(state.version, 1);
    }

    /// F3 regression: on any config where `num_transformer_layers != 1` (the
    /// default is 6), `current_parameters.len() == model_dimension *
    /// num_transformer_layers` while the update network only ever produces
    /// `model_dimension` elements. Adding the two mismatched-length arrays
    /// directly used to panic inside `ndarray`'s `Add`; the update must
    /// instead be tiled across every layer segment without panicking.
    #[test]
    fn test_update_with_step_tiles_per_layer_update_on_default_config() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        assert_ne!(
            config.num_transformer_layers, 1,
            "test is only meaningful when layers > 1"
        );
        let mut state = TransformerOptimizerState::new(&config)
            .expect("TransformerOptimizerState::new should succeed");
        let total_len = state.current_parameters.len();
        assert_eq!(
            total_len,
            config.model_dimension * config.num_transformer_layers
        );

        // Exactly what the adaptation network actually produces: one layer's
        // worth of update, not the full parameter length.
        let per_layer_update = Array1::<f32>::from_elem(config.model_dimension, 0.5);
        let result = state.update_with_step(&per_layer_update, Some(1.0));
        assert!(
            result.is_ok(),
            "a per-layer update must not panic or error on the default config: {result:?}"
        );

        // Every layer segment must have received the same per-layer update.
        for layer in 0..config.num_transformer_layers {
            let start = layer * config.model_dimension;
            let segment = &state.current_parameters.as_slice().expect("contiguous")
                [start..start + config.model_dimension];
            for &v in segment {
                approx::assert_abs_diff_eq!(v, 0.5, epsilon = 1e-6);
            }
        }
    }

    /// F3 regression: a genuinely incompatible update length (one that cannot
    /// be tiled onto the parameter vector) must be a typed error, not a panic.
    #[test]
    fn test_update_with_step_rejects_untileable_length() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let mut state = TransformerOptimizerState::new(&config)
            .expect("TransformerOptimizerState::new should succeed");
        let bad_update = Array1::<f32>::ones(state.current_parameters.len() + 3);
        let result = state.update_with_step(&bad_update, None);
        assert!(result.is_err(), "an untileable update length must error");
    }

    #[test]
    fn test_snapshot_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let state = TransformerOptimizerState::new(&config)
            .expect("TransformerOptimizerState::new should succeed");

        let snapshot = state.create_snapshot();
        assert!(snapshot.is_ok());

        let snap = snapshot.expect("create_snapshot should succeed");
        assert_eq!(snap.version, 0);
        assert_eq!(snap.parameters.len(), state.current_parameters.len());
    }

    #[test]
    fn test_checkpoint_management() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let mut state = TransformerOptimizerState::new(&config)
            .expect("TransformerOptimizerState::new should succeed");

        let checkpoint_id = state.save_checkpoint("test_checkpoint".to_string());
        assert!(checkpoint_id.is_ok());

        let id = checkpoint_id.expect("save_checkpoint should succeed");
        let load_result = state.load_checkpoint(&id);
        assert!(load_result.is_ok());
    }

    #[test]
    fn test_parameter_history() {
        let history = ParameterHistory::<f32>::new(10, 5);
        assert!(history.is_ok());

        let mut h = history.expect("ParameterHistory::new should succeed");
        let params = Array1::<f32>::ones(5);
        assert!(h.record_parameters(&params).is_ok());

        let recent = h.get_recent_parameters(1);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_convergence_tracker() {
        let mut tracker = ConvergenceTracker::<f32>::new();

        tracker.record_loss(2.0);
        tracker.record_loss(1.5);
        tracker.record_loss(1.0);

        let convergence = tracker.get_convergence_rate();
        assert!(convergence > 0.0);

        let stability = tracker.get_stability_score();
        assert!(stability > 0.0 && stability <= 1.0);
    }

    #[test]
    fn test_state_validation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let state = TransformerOptimizerState::new(&config)
            .expect("TransformerOptimizerState::new should succeed");

        let validation = state.validate_state();
        assert!(validation.is_ok());

        let report = validation.expect("validate_state should succeed");
        assert!(report.is_valid);
    }

    /// `StateConfig::from_optimizer_config` used to ignore its argument entirely
    /// and return four hard-coded literals, so shrinking the optimizer's history
    /// bound had no effect on the state tracker.
    #[test]
    fn state_config_follows_the_optimizer_performance_config() {
        let mut config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        config.performance_config.max_history_size = 42;
        config.performance_config.metrics_interval = 7;

        let derived = StateConfig::from_optimizer_config(&config);
        assert_eq!(derived.max_history_size, 42);
        assert_eq!(derived.checkpoint_frequency, 7);

        // A zero interval would make the auto-save modulus undefined, so it is
        // floored at 1 rather than propagated.
        config.performance_config.metrics_interval = 0;
        assert_eq!(
            StateConfig::from_optimizer_config(&config).checkpoint_frequency,
            1
        );
    }

    /// `CheckpointManager::auto_save_config` was written at construction and
    /// never read: there was no auto-save path at all, and the cadence was a
    /// fixed 100 steps regardless of configuration.
    #[test]
    fn checkpoint_manager_auto_saves_on_the_configured_cadence() {
        let mut config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        config.performance_config.metrics_interval = 3;
        let state = TransformerOptimizerState::new(&config).expect("state");

        let mut manager = CheckpointManager::<f32>::new(&config).expect("manager");
        assert_eq!(manager.auto_save_config().frequency, 3);

        // Off-cadence steps must not save.
        for step in [1_usize, 2, 4, 5] {
            assert_eq!(
                manager
                    .maybe_auto_save(step, state.create_snapshot().expect("snapshot"))
                    .expect("auto save"),
                None,
                "step {step} is not a multiple of 3"
            );
        }
        assert_eq!(manager.auto_save_count(), 0);

        // On-cadence steps must save, and be capped at `max_auto_saves`.
        let cap = manager.auto_save_config().max_auto_saves;
        assert!(cap > 0, "default policy should retain some auto-saves");
        for step in (0..).step_by(3).take(cap + 3) {
            let id = manager
                .maybe_auto_save(step, state.create_snapshot().expect("snapshot"))
                .expect("auto save")
                .expect("on-cadence step must produce a checkpoint");
            assert!(id.starts_with("auto_"), "unexpected id {id}");
        }
        assert_eq!(
            manager.auto_save_count(),
            cap,
            "surplus auto-saves must be evicted oldest-first"
        );

        // An explicit checkpoint is retained under its own limit and is not
        // counted as (or evicted by) an auto-save.
        manager
            .save_checkpoint("manual".to_string(), state.create_snapshot().expect("snap"))
            .expect("manual save");
        manager
            .maybe_auto_save(3_000, state.create_snapshot().expect("snap"))
            .expect("auto save");
        assert_eq!(manager.auto_save_count(), cap);
        assert!(manager
            .list_checkpoints()
            .iter()
            .any(|m| m.name == "manual"));
    }

    /// Auto-save must be a no-op when the policy is disabled.
    #[test]
    fn checkpoint_manager_auto_save_respects_a_disabled_policy() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let state = TransformerOptimizerState::new(&config).expect("state");
        let mut manager = CheckpointManager::<f32>::new(&config).expect("manager");
        manager.auto_save_config.enabled = false;

        assert_eq!(
            manager
                .maybe_auto_save(0, state.create_snapshot().expect("snapshot"))
                .expect("auto save"),
            None
        );
        assert_eq!(manager.get_checkpoint_count(), 0);
    }
}
