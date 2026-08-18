//! Pipeline execution for streaming batches (T12).
//!
//! `PipelineExecutionManager::execute_pipeline` used to be `Ok(data)` — it
//! accepted a batch, ignored every configured stage, and handed the batch
//! straight back, while `StageMetrics` and `StageCoordinator` were pure
//! decoration. This module makes the manager actually run its stages, record
//! what they really cost, and reject stage definitions it cannot honour.

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::time::Instant;

use super::functions::to_a_or;
use super::primitives::{
    CoordinationStrategy, PipelineStage, StageCoordinator, StageMetrics, StreamPriority,
};
use super::types_2::StreamingDataPoint;
use super::types_3::SyncBarrier;

/// Per-degree-of-parallelism stage buffer capacity used to turn
/// `buffer_utilization` into a real occupancy figure.
const STAGE_CAPACITY_PER_DEGREE: usize = 64;

/// Maximum number of synchronization barriers retained for inspection.
const MAX_RETAINED_BARRIERS: usize = 100;

/// A transform a pipeline stage can perform.
///
/// Stages are declared by name in [`PipelineStage::processing_function`];
/// parsing is strict, because silently ignoring a stage the caller asked for
/// is exactly the failure mode this module exists to remove.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StageFunction {
    /// Scale every sample's feature vector to unit L2 norm. Zero vectors are
    /// passed through unchanged (there is no direction to normalise).
    Normalize,
    /// Clamp every feature to `[-max_abs, max_abs]`.
    Clip { max_abs: f64 },
    /// Multiply every feature by a constant.
    Scale { factor: f64 },
    /// Exponentially smooth features along the batch:
    /// `x[i] <- alpha * x[i] + (1 - alpha) * smoothed[i - 1]`.
    Smooth { alpha: f64 },
    /// Drop samples whose weight is below `min_weight`.
    FilterByWeight { min_weight: f64 },
}

impl StageFunction {
    /// Parse a stage function declaration of the form `name` or
    /// `name:parameter`.
    pub fn parse(declaration: &str) -> Result<Self> {
        let trimmed = declaration.trim();
        let (name, parameter) = match trimmed.split_once(':') {
            Some((name, parameter)) => (name.trim(), Some(parameter.trim())),
            None => (trimmed, None),
        };
        let numeric = |what: &str| -> Result<f64> {
            let raw = parameter.ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "pipeline stage '{name}' requires a {what} parameter, e.g. \"{name}:1.0\""
                ))
            })?;
            raw.parse::<f64>().map_err(|_| {
                OptimError::InvalidConfig(format!(
                    "pipeline stage '{name}' has an unparseable {what} parameter: '{raw}'"
                ))
            })
        };
        match name {
            "normalize" => Ok(Self::Normalize),
            "clip" => {
                let max_abs = numeric("magnitude")?;
                if !(max_abs.is_finite() && max_abs > 0.0) {
                    return Err(OptimError::InvalidConfig(
                        "pipeline stage 'clip' needs a positive, finite magnitude".to_string(),
                    ));
                }
                Ok(Self::Clip { max_abs })
            }
            "scale" => {
                let factor = numeric("factor")?;
                if !factor.is_finite() {
                    return Err(OptimError::InvalidConfig(
                        "pipeline stage 'scale' needs a finite factor".to_string(),
                    ));
                }
                Ok(Self::Scale { factor })
            }
            "smooth" => {
                let alpha = numeric("alpha")?;
                if !(alpha.is_finite() && (0.0..=1.0).contains(&alpha)) {
                    return Err(OptimError::InvalidConfig(
                        "pipeline stage 'smooth' needs an alpha in [0, 1]".to_string(),
                    ));
                }
                Ok(Self::Smooth { alpha })
            }
            "filter_low_weight" => {
                let min_weight = numeric("weight")?;
                if !min_weight.is_finite() {
                    return Err(OptimError::InvalidConfig(
                        "pipeline stage 'filter_low_weight' needs a finite weight".to_string(),
                    ));
                }
                Ok(Self::FilterByWeight { min_weight })
            }
            other => Err(OptimError::InvalidConfig(format!(
                "unknown pipeline stage function '{other}' (known: normalize, \
                 clip:<magnitude>, scale:<factor>, smooth:<alpha>, \
                 filter_low_weight:<weight>)"
            ))),
        }
    }

    /// Apply the transform to a batch, consuming it.
    pub fn apply<A: Float + Send + Sync>(
        &self,
        mut batch: Vec<StreamingDataPoint<A>>,
    ) -> Vec<StreamingDataPoint<A>> {
        match *self {
            Self::Normalize => {
                for point in &mut batch {
                    let norm = point
                        .features
                        .iter()
                        .fold(A::zero(), |acc, &value| acc + value * value)
                        .sqrt();
                    if norm > A::zero() && norm.is_finite() {
                        point.features.mapv_inplace(|value| value / norm);
                    }
                }
                batch
            }
            Self::Clip { max_abs } => {
                let limit = to_a_or(max_abs, A::one());
                for point in &mut batch {
                    point
                        .features
                        .mapv_inplace(|value| value.max(-limit).min(limit));
                }
                batch
            }
            Self::Scale { factor } => {
                let scale = to_a_or(factor, A::one());
                for point in &mut batch {
                    point.features.mapv_inplace(|value| value * scale);
                }
                batch
            }
            Self::Smooth { alpha } => {
                let a = to_a_or(alpha, A::one());
                let one_minus = A::one() - a;
                let mut previous: Option<Vec<A>> = None;
                for point in &mut batch {
                    if let Some(prior) = previous.as_ref() {
                        if prior.len() == point.features.len() {
                            for (index, value) in point.features.iter_mut().enumerate() {
                                *value = a * *value + one_minus * prior[index];
                            }
                        }
                    }
                    previous = Some(point.features.iter().copied().collect());
                }
                batch
            }
            Self::FilterByWeight { min_weight } => {
                let threshold = to_a_or(min_weight, A::zero());
                batch.retain(|point| point.weight >= threshold);
                batch
            }
        }
    }
}

/// Pipeline execution manager for parallel stream processing.
pub struct PipelineExecutionManager<A: Float + Send + Sync> {
    /// Pipeline stages, executed in order.
    pub(super) pipeline_stages: Vec<PipelineStage<A>>,
    /// Parallelism degree.
    pub(super) parallelismdegree: usize,
    /// Processing priority.
    pub(super) processingpriority: StreamPriority,
    /// Stage coordination.
    pub(super) stage_coordinator: StageCoordinator,
    /// Capacity a single stage buffer is considered full at; the basis for
    /// the real `buffer_utilization` figure in [`StageMetrics`].
    pub(super) stage_capacity: usize,
    /// Number of pipeline executions performed.
    pub(super) executions: usize,
}

impl<A: Float + Send + Sync> PipelineExecutionManager<A> {
    pub fn new(parallelismdegree: usize, processingpriority: StreamPriority) -> Self {
        Self {
            pipeline_stages: Vec::new(),
            parallelismdegree,
            processingpriority,
            stage_coordinator: StageCoordinator::new(parallelismdegree),
            stage_capacity: parallelismdegree.max(1) * STAGE_CAPACITY_PER_DEGREE,
            executions: 0,
        }
    }

    /// Append a stage running the named transform.
    ///
    /// The name is validated here, so a mistyped stage is rejected at
    /// configuration time rather than silently doing nothing forever.
    pub fn add_stage(
        &mut self,
        stage_id: impl Into<String>,
        processing_function: &str,
    ) -> Result<()> {
        StageFunction::parse(processing_function)?;
        self.pipeline_stages.push(PipelineStage {
            stage_id: stage_id.into(),
            processing_function: processing_function.to_string(),
            input_buffer: VecDeque::new(),
            output_buffer: VecDeque::new(),
            stage_metrics: StageMetrics::default(),
        });
        Ok(())
    }

    /// Number of configured stages.
    pub fn stage_count(&self) -> usize {
        self.pipeline_stages.len()
    }

    /// Measured metrics of every configured stage, in execution order.
    pub fn stage_metrics(&self) -> Vec<(&str, &StageMetrics)> {
        self.pipeline_stages
            .iter()
            .map(|stage| (stage.stage_id.as_str(), &stage.stage_metrics))
            .collect()
    }

    /// Synchronization barriers recorded by completed executions.
    pub fn synchronization_barriers(&self) -> &[SyncBarrier] {
        &self.stage_coordinator.synchronization_barriers
    }

    /// The priority this pipeline processes its work at.
    pub fn processing_priority(&self) -> StreamPriority {
        self.processingpriority
    }

    /// The configured degree of parallelism.
    pub fn parallelism_degree(&self) -> usize {
        self.parallelismdegree
    }

    /// Adopt a new degree of parallelism (and the stage capacity implied by
    /// it), e.g. after a resource re-allocation.
    pub fn set_parallelism_degree(&mut self, degree: usize) {
        self.parallelismdegree = degree.max(1);
        self.stage_coordinator.parallelismdegree = self.parallelismdegree;
        self.stage_capacity = self.parallelismdegree * STAGE_CAPACITY_PER_DEGREE;
    }

    /// Execute every configured stage, in order, over `data`.
    ///
    /// Each stage's own buffers are the transfer medium — the batch is moved
    /// into `input_buffer`, transformed, and moved on through `output_buffer`
    /// — and each stage records the time it really took, the throughput it
    /// really achieved and how full its buffer really was. With no stages
    /// configured this is a genuine no-op (there is nothing to run), not a
    /// silently-skipped pipeline.
    pub fn execute_pipeline(
        &mut self,
        data: Vec<StreamingDataPoint<A>>,
    ) -> Result<Vec<StreamingDataPoint<A>>> {
        if self.pipeline_stages.is_empty() {
            return Ok(data);
        }
        let capacity = self.stage_capacity.max(1) as f64;
        let mut current = data;
        for stage in &mut self.pipeline_stages {
            let function = match StageFunction::parse(&stage.processing_function) {
                Ok(function) => function,
                Err(error) => {
                    stage.stage_metrics.error_count += 1;
                    return Err(error);
                }
            };
            let input_len = current.len();
            stage.input_buffer = current.into_iter().collect();
            let started = Instant::now();
            let produced = function.apply(stage.input_buffer.drain(..).collect());
            let elapsed = started.elapsed();
            let output_len = produced.len();
            stage.output_buffer = produced.into_iter().collect();

            let seconds = elapsed.as_secs_f64();
            stage.stage_metrics.processing_time_ms = seconds * 1000.0;
            stage.stage_metrics.throughput_samples_per_sec = if seconds > 0.0 {
                output_len as f64 / seconds
            } else {
                0.0
            };
            stage.stage_metrics.buffer_utilization = (input_len as f64 / capacity).min(1.0);

            current = stage.output_buffer.drain(..).collect();
        }

        self.executions += 1;
        self.stage_coordinator
            .synchronization_barriers
            .push(SyncBarrier {
                barrier_id: format!("pipeline-execution-{}", self.executions),
                wait_count: self.pipeline_stages.len(),
                timestamp: Instant::now(),
            });
        if self.stage_coordinator.synchronization_barriers.len() > MAX_RETAINED_BARRIERS {
            self.stage_coordinator.synchronization_barriers.remove(0);
        }
        self.stage_coordinator.coordination_strategy = if self.parallelismdegree > 1 {
            CoordinationStrategy::PipelineParallel
        } else {
            CoordinationStrategy::DataParallel
        };
        Ok(current)
    }
}

#[cfg(test)]
mod pipeline_execution_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap;

    fn point(features: Vec<f64>, weight: f64) -> StreamingDataPoint<f64> {
        StreamingDataPoint {
            features: Array1::from_vec(features),
            target: Some(1.0),
            timestamp: Instant::now(),
            weight,
            metadata: HashMap::new(),
        }
    }

    fn manager() -> PipelineExecutionManager<f64> {
        PipelineExecutionManager::new(2, StreamPriority::Normal)
    }

    /// T12: a configured stage must actually transform the batch.
    #[test]
    fn configured_stages_actually_transform_the_batch() {
        let mut pipeline = manager();
        pipeline.add_stage("clip", "clip:1.0").expect("add clip");
        pipeline.add_stage("scale", "scale:2.0").expect("add scale");

        let output = pipeline
            .execute_pipeline(vec![
                point(vec![5.0, -3.0], 1.0),
                point(vec![0.25, 0.5], 1.0),
            ])
            .expect("execute");

        assert_eq!(output.len(), 2);
        // 5.0 clipped to 1.0 then doubled; -3.0 clipped to -1.0 then doubled.
        assert!((output[0].features[0] - 2.0).abs() < 1e-12);
        assert!((output[0].features[1] + 2.0).abs() < 1e-12);
        // Within the clip range, only the scaling applies.
        assert!((output[1].features[0] - 0.5).abs() < 1e-12);
        assert!((output[1].features[1] - 1.0).abs() < 1e-12);
    }

    /// T12: stages that drop samples really change the batch size, and the
    /// metrics recorded are real measurements, not defaults.
    #[test]
    fn filter_stage_drops_samples_and_records_real_metrics() {
        let mut pipeline = manager();
        pipeline
            .add_stage("filter", "filter_low_weight:0.5")
            .expect("add filter");

        let output = pipeline
            .execute_pipeline(vec![
                point(vec![1.0], 0.9),
                point(vec![2.0], 0.1),
                point(vec![3.0], 0.5),
            ])
            .expect("execute");

        assert_eq!(
            output.len(),
            2,
            "T12 regression: the filter stage did not run (batch came back unchanged)"
        );
        let metrics = pipeline.stage_metrics();
        assert_eq!(metrics.len(), 1);
        let (name, stage) = metrics[0];
        assert_eq!(name, "filter");
        assert!(
            stage.buffer_utilization > 0.0,
            "buffer utilization was never measured"
        );
        assert_eq!(stage.error_count, 0);
        assert_eq!(
            pipeline.synchronization_barriers().len(),
            1,
            "the coordinator recorded no barrier for the execution"
        );
        assert_eq!(pipeline.synchronization_barriers()[0].wait_count, 1);
    }

    /// A pipeline with no stages is genuinely a no-op, which is a different
    /// thing from ignoring the stages that were configured.
    #[test]
    fn empty_pipeline_passes_the_batch_through_untouched() {
        let mut pipeline = manager();
        let input = vec![point(vec![1.0, 2.0], 1.0)];
        let output = pipeline.execute_pipeline(input.clone()).expect("execute");
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].features, input[0].features);
        assert_eq!(pipeline.stage_count(), 0);
    }

    /// An unknown or malformed stage declaration must be an honest error.
    #[test]
    fn unknown_or_malformed_stage_declarations_are_errors() {
        let mut pipeline = manager();
        assert!(pipeline.add_stage("bogus", "teleport").is_err());
        assert!(pipeline.add_stage("clip", "clip").is_err());
        assert!(pipeline.add_stage("clip", "clip:not-a-number").is_err());
        assert!(pipeline.add_stage("clip", "clip:-1.0").is_err());
        assert!(pipeline.add_stage("smooth", "smooth:2.0").is_err());
        assert_eq!(
            pipeline.stage_count(),
            0,
            "invalid stages must not register"
        );

        // A stage that was somehow constructed with a bad name still fails
        // loudly at execution time rather than being skipped.
        pipeline.pipeline_stages.push(PipelineStage {
            stage_id: "sneaky".to_string(),
            processing_function: "teleport".to_string(),
            input_buffer: VecDeque::new(),
            output_buffer: VecDeque::new(),
            stage_metrics: StageMetrics::default(),
        });
        assert!(pipeline
            .execute_pipeline(vec![point(vec![1.0], 1.0)])
            .is_err());
        assert_eq!(pipeline.pipeline_stages[0].stage_metrics.error_count, 1);
    }

    #[test]
    fn normalize_stage_produces_unit_norm_features() {
        let mut pipeline = manager();
        pipeline.add_stage("norm", "normalize").expect("add");
        let output = pipeline
            .execute_pipeline(vec![point(vec![3.0, 4.0], 1.0), point(vec![0.0, 0.0], 1.0)])
            .expect("execute");
        assert!((output[0].features[0] - 0.6).abs() < 1e-12);
        assert!((output[0].features[1] - 0.8).abs() < 1e-12);
        // A zero vector has no direction and must survive unchanged.
        assert_eq!(output[1].features[0], 0.0);
        assert_eq!(output[1].features[1], 0.0);
    }
}
