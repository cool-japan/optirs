// Sequence processing for optimization trajectories

use super::config::TransformerBasedOptimizerConfig;
use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Type alias for sequence data tuple (gradients, parameters, losses)
type SequenceDataTuple<T> = (Vec<Array2<T>>, Vec<Array2<T>>, Vec<Array1<T>>);

/// Sequence processing strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SequenceProcessingStrategy {
    /// Sliding window approach
    SlidingWindow,
    /// Hierarchical chunking
    Hierarchical,
    /// Attention-based selection
    AttentionBased,
    /// Adaptive segmentation
    Adaptive,
    /// Truncated backpropagation
    TruncatedBPTT,
}

/// Optimization sequence processor
pub struct OptimizationSequenceProcessor<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Processing strategy
    strategy: SequenceProcessingStrategy,

    /// Maximum sequence length
    max_sequence_length: usize,

    /// Window size for sliding window
    window_size: usize,

    /// Overlap between windows
    window_overlap: usize,

    /// Sequence history buffer
    sequence_buffer: SequenceBuffer<T>,

    /// Sequence statistics
    statistics: SequenceStatistics<T>,

    /// Preprocessing pipeline
    preprocessor: SequencePreprocessor<T>,
}

impl<
        T: Float
            + Debug
            + scirs2_core::numeric::FromPrimitive
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync,
    > OptimizationSequenceProcessor<T>
{
    /// Create new sequence processor
    ///
    /// # Errors
    /// Returns `Err` when `config.sequence_length` is zero. The derived window
    /// size (`sequence_length / 2`) and stride (`window_size - overlap`) must
    /// both be at least one: a zero stride makes `Iterator::step_by` panic, and
    /// `window_size - window_overlap` would underflow in release builds. The
    /// window is therefore floored at 1 and the overlap is kept strictly below
    /// it.
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let strategy = SequenceProcessingStrategy::SlidingWindow;
        let max_sequence_length = config.sequence_length;
        if max_sequence_length == 0 {
            return Err(crate::error::OptimError::InvalidConfig(
                "sequence_length must be greater than 0".to_string(),
            ));
        }
        let window_size = (max_sequence_length / 2).max(1);
        // Overlap must stay strictly below the window so the stride is >= 1.
        let window_overlap = (window_size / 4).min(window_size - 1);
        let model_dimension = config.model_dimension;

        let sequence_buffer = SequenceBuffer::new(1000)?;
        let statistics = SequenceStatistics::new();
        let preprocessor = SequencePreprocessor::new(model_dimension)?;

        Ok(Self {
            strategy,
            max_sequence_length,
            window_size,
            window_overlap,
            sequence_buffer,
            statistics,
            preprocessor,
        })
    }

    /// Process optimization sequence
    pub fn process_optimization_sequence(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array2<T>> {
        // Update statistics
        self.statistics
            .update(gradient_history, parameter_history, loss_history)?;

        // Store in buffer
        self.sequence_buffer
            .add_sequence(gradient_history, parameter_history, loss_history)?;

        match self.strategy {
            SequenceProcessingStrategy::SlidingWindow => {
                self.process_sliding_window(gradient_history, parameter_history, loss_history)
            }
            SequenceProcessingStrategy::Hierarchical => {
                self.process_hierarchical(gradient_history, parameter_history, loss_history)
            }
            SequenceProcessingStrategy::AttentionBased => {
                self.process_attention_based(gradient_history, parameter_history, loss_history)
            }
            SequenceProcessingStrategy::Adaptive => {
                self.process_adaptive(gradient_history, parameter_history, loss_history)
            }
            SequenceProcessingStrategy::TruncatedBPTT => {
                self.process_truncated_bptt(gradient_history, parameter_history, loss_history)
            }
        }
    }

    /// Process using sliding window strategy
    fn process_sliding_window(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array2<T>> {
        let sequence_length = gradient_history.shape()[0];

        if sequence_length <= self.max_sequence_length {
            // Sequence fits in one window
            return self.combine_sequences(gradient_history, parameter_history, loss_history);
        }

        // Process in overlapping windows. `saturating_sub` plus `.max(1)` keeps
        // the stride positive even if the window/overlap pair is ever mutated
        // into an inconsistent state: `step_by(0)` panics.
        let mut processed_chunks = Vec::new();
        let step_size = self.window_size.saturating_sub(self.window_overlap).max(1);

        for start in (0..sequence_length).step_by(step_size) {
            let end = (start + self.window_size).min(sequence_length);

            let grad_chunk = gradient_history.slice(s![start..end, ..]);
            let param_chunk = parameter_history.slice(s![start..end, ..]);
            let loss_chunk = loss_history.slice(s![start..end]);

            let processed_chunk = self.combine_sequences(
                &grad_chunk.to_owned(),
                &param_chunk.to_owned(),
                &loss_chunk.to_owned(),
            )?;

            processed_chunks.push(processed_chunk);
        }

        // Combine processed chunks
        self.combine_chunks(&processed_chunks)
    }

    /// Process using hierarchical strategy
    fn process_hierarchical(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array2<T>> {
        let sequence_length = gradient_history.shape()[0];

        // Create hierarchical representation
        let mut levels = Vec::new();
        let mut current_level =
            self.combine_sequences(gradient_history, parameter_history, loss_history)?;
        levels.push(current_level.clone());

        // Build hierarchy by downsampling
        let mut current_length = sequence_length;
        while current_length > self.max_sequence_length {
            current_length /= 2;
            current_level = self.downsample_sequence(&current_level, current_length)?;
            levels.push(current_level.clone());
        }

        // Return the highest level that fits. `levels` is seeded with the
        // combined sequence before the loop, so it is never empty; taking the
        // fallible read keeps that local rather than asserting it from afar.
        levels.pop().ok_or_else(|| {
            crate::error::OptimError::ComputationError(
                "hierarchical processing produced no levels".to_string(),
            )
        })
    }

    /// Process using attention-based selection
    fn process_attention_based(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array2<T>> {
        let sequence_length = gradient_history.shape()[0];

        if sequence_length <= self.max_sequence_length {
            return self.combine_sequences(gradient_history, parameter_history, loss_history);
        }

        // Compute importance scores for each time step
        let importance_scores = self.compute_importance_scores(gradient_history, loss_history)?;

        // Select most important steps
        let selected_indices =
            self.select_top_k_indices(&importance_scores, self.max_sequence_length)?;

        // Extract selected sequences
        let mut selected_gradients =
            Array2::zeros((self.max_sequence_length, gradient_history.shape()[1]));
        let mut selected_parameters =
            Array2::zeros((self.max_sequence_length, parameter_history.shape()[1]));
        let mut selected_losses = Array1::zeros(self.max_sequence_length);

        for (i, &idx) in selected_indices.iter().enumerate() {
            selected_gradients
                .row_mut(i)
                .assign(&gradient_history.row(idx));
            selected_parameters
                .row_mut(i)
                .assign(&parameter_history.row(idx));
            selected_losses[i] = loss_history[idx];
        }

        self.combine_sequences(&selected_gradients, &selected_parameters, &selected_losses)
    }

    /// Process using adaptive segmentation
    fn process_adaptive(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array2<T>> {
        // Detect change points in the optimization trajectory
        let change_points = self.detect_change_points(loss_history)?;

        // Segment sequence based on change points
        let segments = self.segment_by_change_points(
            gradient_history,
            parameter_history,
            loss_history,
            &change_points,
        )?;

        // Process each segment and combine
        let mut processed_segments = Vec::new();
        for segment in segments {
            let processed =
                self.combine_sequences(&segment.gradients, &segment.parameters, &segment.losses)?;
            processed_segments.push(processed);
        }

        self.combine_chunks(&processed_segments)
    }

    /// Process using truncated BPTT
    fn process_truncated_bptt(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array2<T>> {
        let sequence_length = gradient_history.shape()[0];

        if sequence_length <= self.max_sequence_length {
            return self.combine_sequences(gradient_history, parameter_history, loss_history);
        }

        // Take the most recent subsequence
        let start_idx = sequence_length - self.max_sequence_length;
        let grad_chunk = gradient_history.slice(s![start_idx.., ..]);
        let param_chunk = parameter_history.slice(s![start_idx.., ..]);
        let loss_chunk = loss_history.slice(s![start_idx..]);

        self.combine_sequences(
            &grad_chunk.to_owned(),
            &param_chunk.to_owned(),
            &loss_chunk.to_owned(),
        )
    }

    /// Convert optimization trajectory to training sequences
    pub fn trajectory_to_sequences(
        &mut self,
        trajectory: &super::OptimizationTrajectory<T>,
    ) -> Result<Vec<super::TrainingSequence<T>>> {
        let sequence_length = trajectory.gradient_sequence.shape()[0];
        let mut sequences = Vec::new();

        // Create overlapping sequences for training. The stride is floored at 1
        // because `step_by(0)` panics, which it did for any window size < 2.
        let stride = (self.window_size / 2).max(1);
        for start in (0..sequence_length).step_by(stride) {
            let end = (start + self.window_size).min(sequence_length);

            if end - start < stride {
                break; // Skip sequences that are too short
            }

            let input_end = end - 1;
            let target_start = start + 1;

            let input_gradients = trajectory.gradient_sequence.slice(s![start..input_end, ..]);
            let input_parameters = trajectory
                .parameter_sequence
                .slice(s![start..input_end, ..]);
            let input_losses = trajectory.loss_sequence.slice(s![start..input_end]);

            let target_gradients = trajectory
                .gradient_sequence
                .slice(s![target_start..end, ..]);
            let target_parameters = trajectory
                .parameter_sequence
                .slice(s![target_start..end, ..]);
            let target_losses = trajectory.loss_sequence.slice(s![target_start..end]);

            let input = self.combine_sequences(
                &input_gradients.to_owned(),
                &input_parameters.to_owned(),
                &input_losses.to_owned(),
            )?;

            let target = self.combine_sequences(
                &target_gradients.to_owned(),
                &target_parameters.to_owned(),
                &target_losses.to_owned(),
            )?;

            sequences.push(super::TrainingSequence {
                input,
                target,
                sequence_length: input_end - start,
            });
        }

        Ok(sequences)
    }

    /// Combine gradients, parameters, and losses into unified sequence
    fn combine_sequences(
        &self,
        gradients: &Array2<T>,
        parameters: &Array2<T>,
        losses: &Array1<T>,
    ) -> Result<Array2<T>> {
        self.preprocessor
            .combine_sequences(gradients, parameters, losses)
    }

    /// Combine multiple processed chunks
    fn combine_chunks(&self, chunks: &[Array2<T>]) -> Result<Array2<T>> {
        if chunks.is_empty() {
            return Err(crate::error::OptimError::Other(
                "No chunks to combine".to_string(),
            ));
        }

        if chunks.len() == 1 {
            return Ok(chunks[0].clone());
        }

        // Simple concatenation strategy
        let total_length: usize = chunks.iter().map(|chunk| chunk.shape()[0]).sum();
        let feature_dim = chunks[0].shape()[1];

        let mut combined = Array2::zeros((total_length.min(self.max_sequence_length), feature_dim));
        let mut current_pos = 0;

        for chunk in chunks {
            let chunk_len = chunk.shape()[0];
            let copy_len = (chunk_len).min(self.max_sequence_length - current_pos);

            if copy_len == 0 {
                break;
            }

            combined
                .slice_mut(s![current_pos..current_pos + copy_len, ..])
                .assign(&chunk.slice(s![..copy_len, ..]));

            current_pos += copy_len;

            if current_pos >= self.max_sequence_length {
                break;
            }
        }

        Ok(combined)
    }

    /// Downsample sequence to target length
    fn downsample_sequence(&self, sequence: &Array2<T>, target_length: usize) -> Result<Array2<T>> {
        let current_length = sequence.shape()[0];
        let feature_dim = sequence.shape()[1];

        if current_length <= target_length {
            return Ok(sequence.clone());
        }

        let mut downsampled = Array2::zeros((target_length, feature_dim));
        let step = current_length as f64 / target_length as f64;

        for i in 0..target_length {
            let source_idx = (i as f64 * step) as usize;
            downsampled
                .row_mut(i)
                .assign(&sequence.row(source_idx.min(current_length - 1)));
        }

        Ok(downsampled)
    }

    /// Compute importance scores for sequence steps
    fn compute_importance_scores(
        &self,
        gradients: &Array2<T>,
        losses: &Array1<T>,
    ) -> Result<Array1<T>> {
        let sequence_length = gradients.shape()[0];
        let mut scores = Array1::zeros(sequence_length);

        for i in 0..sequence_length {
            // Combine gradient magnitude and loss change
            let grad_norm = gradients
                .row(i)
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |acc, x| acc + x)
                .sqrt();

            let loss_change = if i > 0 {
                (losses[i] - losses[i - 1]).abs()
            } else {
                T::zero()
            };

            scores[i] = grad_norm + loss_change;
        }

        Ok(scores)
    }

    /// Select top-k indices based on importance scores
    fn select_top_k_indices(&self, scores: &Array1<T>, k: usize) -> Result<Vec<usize>> {
        let mut indexed_scores: Vec<(usize, T)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // `total_cmp` is a total order, so a NaN importance score sorts
        // deterministically rather than panicking inside `sort_by`.
        indexed_scores.sort_by(|a, b| {
            b.1.to_f64()
                .unwrap_or(f64::NAN)
                .total_cmp(&a.1.to_f64().unwrap_or(f64::NAN))
        });

        let mut selected_indices: Vec<usize> = indexed_scores
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect();

        selected_indices.sort();
        Ok(selected_indices)
    }

    /// Detect change points in loss trajectory
    fn detect_change_points(&self, losses: &Array1<T>) -> Result<Vec<usize>> {
        let mut change_points = vec![0]; // Always include start
        let window_size = 5;
        let threshold = scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero());

        // `losses.len() - window_size` underflowed (usize) for any trajectory
        // shorter than 2*window_size, producing a huge upper bound and an
        // out-of-bounds slice panic. Bail out instead when there is not enough
        // history on both sides of a candidate change point.
        let last_candidate = losses.len().saturating_sub(window_size);
        for i in window_size..last_candidate {
            let before_mean = match losses.slice(s![i - window_size..i]).mean() {
                Some(m) => m,
                None => continue,
            };
            let after_mean = match losses.slice(s![i..i + window_size]).mean() {
                Some(m) => m,
                None => continue,
            };

            if (before_mean - after_mean).abs() > threshold {
                change_points.push(i);
            }
        }

        change_points.push(losses.len() - 1); // Always include end
        Ok(change_points)
    }

    /// Segment sequences by change points
    fn segment_by_change_points(
        &self,
        gradients: &Array2<T>,
        parameters: &Array2<T>,
        losses: &Array1<T>,
        change_points: &[usize],
    ) -> Result<Vec<SequenceSegment<T>>> {
        let mut segments = Vec::new();

        for i in 0..change_points.len() - 1 {
            let start = change_points[i];
            let end = change_points[i + 1];

            let segment = SequenceSegment {
                gradients: gradients.slice(s![start..end, ..]).to_owned(),
                parameters: parameters.slice(s![start..end, ..]).to_owned(),
                losses: losses.slice(s![start..end]).to_owned(),
                start_index: start,
                end_index: end,
            };

            segments.push(segment);
        }

        Ok(segments)
    }

    /// Set processing strategy
    pub fn set_strategy(&mut self, strategy: SequenceProcessingStrategy) {
        self.strategy = strategy;
    }

    /// Get current strategy
    pub fn get_strategy(&self) -> SequenceProcessingStrategy {
        self.strategy
    }

    /// Get sequence statistics
    pub fn get_statistics(&self) -> &SequenceStatistics<T> {
        &self.statistics
    }

    /// Reset processor state
    pub fn reset(&mut self) -> Result<()> {
        self.sequence_buffer.clear();
        self.statistics.reset();
        Ok(())
    }
}

/// Sequence buffer for storing optimization history
pub struct SequenceBuffer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Gradient history
    gradient_buffer: VecDeque<Array2<T>>,

    /// Parameter history
    parameter_buffer: VecDeque<Array2<T>>,

    /// Loss history
    loss_buffer: VecDeque<Array1<T>>,

    /// Maximum buffer size
    max_size: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    SequenceBuffer<T>
{
    /// A buffer holding at most `max_size` recorded sequences.
    ///
    /// The feature width is not a buffer concern: it belongs to
    /// [`SequencePreprocessor`], which is the component that actually truncates
    /// to it. The duplicate `model_dimension` this used to store was never read
    /// and could silently disagree with the preprocessor's.
    pub fn new(max_size: usize) -> Result<Self> {
        Ok(Self {
            gradient_buffer: VecDeque::new(),
            parameter_buffer: VecDeque::new(),
            loss_buffer: VecDeque::new(),
            max_size,
        })
    }

    pub fn add_sequence(
        &mut self,
        gradients: &Array2<T>,
        parameters: &Array2<T>,
        losses: &Array1<T>,
    ) -> Result<()> {
        self.gradient_buffer.push_back(gradients.clone());
        self.parameter_buffer.push_back(parameters.clone());
        self.loss_buffer.push_back(losses.clone());

        while self.gradient_buffer.len() > self.max_size {
            self.gradient_buffer.pop_front();
            self.parameter_buffer.pop_front();
            self.loss_buffer.pop_front();
        }

        Ok(())
    }

    pub fn clear(&mut self) {
        self.gradient_buffer.clear();
        self.parameter_buffer.clear();
        self.loss_buffer.clear();
    }

    pub fn get_recent_sequences(&self, count: usize) -> SequenceDataTuple<T> {
        let actual_count = count.min(self.gradient_buffer.len());

        let gradients = self
            .gradient_buffer
            .iter()
            .rev()
            .take(actual_count)
            .cloned()
            .collect();
        let parameters = self
            .parameter_buffer
            .iter()
            .rev()
            .take(actual_count)
            .cloned()
            .collect();
        let losses = self
            .loss_buffer
            .iter()
            .rev()
            .take(actual_count)
            .cloned()
            .collect();

        (gradients, parameters, losses)
    }
}

/// Sequence statistics tracker
pub struct SequenceStatistics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Gradient statistics
    gradient_stats: StatisticsAccumulator<T>,

    /// Parameter statistics
    parameter_stats: StatisticsAccumulator<T>,

    /// Loss statistics
    loss_stats: StatisticsAccumulator<T>,

    /// Sequence length statistics
    length_stats: StatisticsAccumulator<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for SequenceStatistics<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    SequenceStatistics<T>
{
    pub fn new() -> Self {
        Self {
            gradient_stats: StatisticsAccumulator::new(),
            parameter_stats: StatisticsAccumulator::new(),
            loss_stats: StatisticsAccumulator::new(),
            length_stats: StatisticsAccumulator::new(),
        }
    }

    pub fn update(
        &mut self,
        gradients: &Array2<T>,
        parameters: &Array2<T>,
        losses: &Array1<T>,
    ) -> Result<()> {
        self.gradient_stats.update_from_array2(gradients);
        self.parameter_stats.update_from_array2(parameters);
        self.loss_stats.update_from_array1(losses);
        self.length_stats
            .update(crate::common::cast_scalar(gradients.shape()[0])?);

        Ok(())
    }

    pub fn reset(&mut self) {
        self.gradient_stats.reset();
        self.parameter_stats.reset();
        self.loss_stats.reset();
        self.length_stats.reset();
    }

    pub fn get_gradient_stats(&self) -> &StatisticsAccumulator<T> {
        &self.gradient_stats
    }

    pub fn get_parameter_stats(&self) -> &StatisticsAccumulator<T> {
        &self.parameter_stats
    }

    pub fn get_loss_stats(&self) -> &StatisticsAccumulator<T> {
        &self.loss_stats
    }
}

/// Sequence preprocessor.
///
/// The per-feature normalization statistics this used to declare were never
/// populated or consulted — `combine_sequences` normalizes a loss against the
/// first observed loss instead — so the empty map has been removed rather than
/// left as a permanently-empty cache.
pub struct SequencePreprocessor<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Model dimension
    model_dimension: usize,

    _element: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    SequencePreprocessor<T>
{
    pub fn new(model_dimension: usize) -> Result<Self> {
        Ok(Self {
            model_dimension,
            _element: std::marker::PhantomData,
        })
    }

    pub fn combine_sequences(
        &self,
        gradients: &Array2<T>,
        parameters: &Array2<T>,
        losses: &Array1<T>,
    ) -> Result<Array2<T>> {
        let sequence_length = gradients.shape()[0];
        let grad_dim = gradients.shape()[1];
        let param_dim = parameters.shape()[1];

        // Combined features: [gradients, parameters, loss, normalized_loss]
        let feature_dim = grad_dim + param_dim + 2;
        let mut combined = Array2::zeros((sequence_length, feature_dim.min(self.model_dimension)));

        for i in 0..sequence_length {
            let mut feature_idx = 0;

            // Add gradient features
            for j in 0..grad_dim.min(self.model_dimension / 3) {
                if feature_idx < combined.shape()[1] {
                    combined[[i, feature_idx]] = gradients[[i, j]];
                    feature_idx += 1;
                }
            }

            // Add parameter features
            for j in 0..param_dim.min(self.model_dimension / 3) {
                if feature_idx < combined.shape()[1] {
                    combined[[i, feature_idx]] = parameters[[i, j]];
                    feature_idx += 1;
                }
            }

            // Add loss features
            if feature_idx < combined.shape()[1] {
                combined[[i, feature_idx]] = losses[i];
                feature_idx += 1;
            }

            // Add normalized loss (loss relative to first loss)
            if feature_idx < combined.shape()[1] && i > 0 {
                let normalized_loss = if losses[0] != T::zero() {
                    losses[i] / losses[0]
                } else {
                    T::one()
                };
                combined[[i, feature_idx]] = normalized_loss;
            }
        }

        Ok(combined)
    }
}

/// Chunking strategy
pub struct ChunkingStrategy<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Maximum chunk size
    max_chunk_size: usize,

    /// Overlap between chunks
    overlap_size: usize,

    /// Chunk statistics
    chunk_stats: StatisticsAccumulator<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    ChunkingStrategy<T>
{
    /// # Errors
    /// Returns `Err` when `max_chunk_size` is zero or `overlap_size` is not
    /// strictly smaller than it — either would make the chunking stride
    /// (`max_chunk_size - overlap_size`) zero or underflow, and a zero stride
    /// makes `Iterator::step_by` panic.
    pub fn new(max_chunk_size: usize, overlap_size: usize) -> Result<Self> {
        if max_chunk_size == 0 {
            return Err(crate::error::OptimError::InvalidConfig(
                "max_chunk_size must be greater than 0".to_string(),
            ));
        }
        if overlap_size >= max_chunk_size {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "overlap_size ({overlap_size}) must be smaller than max_chunk_size ({max_chunk_size})"
            )));
        }
        Ok(Self {
            max_chunk_size,
            overlap_size,
            chunk_stats: StatisticsAccumulator::new(),
        })
    }

    /// Accumulated statistics over every element of every chunk this strategy
    /// has produced.
    ///
    /// `create_chunks` used to build the chunks without ever feeding the
    /// accumulator, so these statistics were permanently empty.
    pub fn chunk_stats(&self) -> &StatisticsAccumulator<T> {
        &self.chunk_stats
    }

    pub fn create_chunks(&mut self, sequence: &Array2<T>) -> Result<Vec<Array2<T>>> {
        let sequence_length = sequence.shape()[0];
        let mut chunks = Vec::new();

        if sequence_length <= self.max_chunk_size {
            self.chunk_stats.update_from_array2(sequence);
            chunks.push(sequence.clone());
            return Ok(chunks);
        }

        let step_size = self.max_chunk_size.saturating_sub(self.overlap_size).max(1);

        for start in (0..sequence_length).step_by(step_size) {
            let end = (start + self.max_chunk_size).min(sequence_length);
            let chunk = sequence.slice(s![start..end, ..]).to_owned();
            self.chunk_stats.update_from_array2(&chunk);
            chunks.push(chunk);

            if end >= sequence_length {
                break;
            }
        }

        Ok(chunks)
    }
}

/// Supporting data structures
#[derive(Debug, Clone)]
pub struct SequenceSegment<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub gradients: Array2<T>,
    pub parameters: Array2<T>,
    pub losses: Array1<T>,
    pub start_index: usize,
    pub end_index: usize,
}

pub struct StatisticsAccumulator<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    count: usize,
    sum: T,
    sum_sq: T,
    min: T,
    max: T,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> Default
    for StatisticsAccumulator<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    StatisticsAccumulator<T>
{
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: T::zero(),
            sum_sq: T::zero(),
            min: T::infinity(),
            max: T::neg_infinity(),
        }
    }

    pub fn update(&mut self, value: T) {
        self.count += 1;
        self.sum = self.sum + value;
        self.sum_sq = self.sum_sq + value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    pub fn update_from_array1(&mut self, array: &Array1<T>) {
        for &value in array.iter() {
            self.update(value);
        }
    }

    pub fn update_from_array2(&mut self, array: &Array2<T>) {
        for &value in array.iter() {
            self.update(value);
        }
    }

    pub fn mean(&self) -> T {
        if self.count > 0 {
            self.sum / scirs2_core::numeric::NumCast::from(self.count).unwrap_or_else(|| T::zero())
        } else {
            T::zero()
        }
    }

    pub fn variance(&self) -> T {
        if self.count > 1 {
            let mean = self.mean();
            (self.sum_sq
                / scirs2_core::numeric::NumCast::from(self.count).unwrap_or_else(|| T::zero()))
                - (mean * mean)
        } else {
            T::zero()
        }
    }

    pub fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    pub fn min(&self) -> T {
        self.min
    }

    pub fn max(&self) -> T {
        self.max
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.sum = T::zero();
        self.sum_sq = T::zero();
        self.min = T::infinity();
        self.max = T::neg_infinity();
    }
}

// Import for slice macro
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_processor_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let processor = OptimizationSequenceProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_sequence_buffer() {
        let buffer = SequenceBuffer::<f32>::new(10);
        assert!(buffer.is_ok());

        let mut buf = buffer.expect("SequenceBuffer::new should succeed");
        let gradients = Array2::<f32>::ones((5, 64));
        let parameters = Array2::<f32>::ones((5, 64));
        let losses = Array1::<f32>::ones(5);

        assert!(buf.add_sequence(&gradients, &parameters, &losses).is_ok());
    }

    #[test]
    fn test_sequence_statistics() {
        let mut stats = SequenceStatistics::<f32>::new();

        let gradients = Array2::<f32>::ones((10, 5));
        let parameters = Array2::<f32>::ones((10, 5));
        let losses = Array1::<f32>::ones(10);

        assert!(stats.update(&gradients, &parameters, &losses).is_ok());
        assert!(stats.get_gradient_stats().mean() > 0.0);
    }

    #[test]
    fn test_statistics_accumulator() {
        let mut acc = StatisticsAccumulator::<f32>::new();

        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);

        assert_eq!(acc.mean(), 2.0);
        assert!(acc.std_dev() > 0.0);
        assert_eq!(acc.min(), 1.0);
        assert_eq!(acc.max(), 3.0);
    }

    #[test]
    fn test_chunking_strategy() {
        let chunking = ChunkingStrategy::<f32>::new(10, 2);
        assert!(chunking.is_ok());

        let mut strategy = chunking.expect("ChunkingStrategy::new should succeed");
        let sequence = Array2::<f32>::ones((25, 5));

        let chunks = strategy.create_chunks(&sequence);
        assert!(chunks.is_ok());

        let chunk_vec = chunks.expect("create_chunks should succeed");
        assert!(chunk_vec.len() > 1);
    }
}
