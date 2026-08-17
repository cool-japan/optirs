// Low-latency optimization for real-time streaming applications
//
// This module provides specialized optimizers and techniques for applications
// that require extremely low latency updates, such as high-frequency trading,
// real-time control systems, and interactive machine learning.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard, PoisonError,
};
use std::time::{Duration, Instant};

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

#[cfg(test)]
mod regression_tests;

/// Recovers a mutex guard even if the lock was poisoned by a panicking thread.
///
/// The data protected by every mutex in this module is a plain value with no
/// cross-field invariant that a panic could leave half-updated, so continuing
/// with the recovered value is strictly better than panicking a real-time
/// update path.
fn lock_recovered<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Low-latency optimization configuration
#[derive(Debug, Clone)]
pub struct LowLatencyConfig {
    /// Target latency budget (microseconds)
    pub target_latency_us: u64,

    /// Maximum acceptable latency (microseconds)
    pub max_latency_us: u64,

    /// Enable pre-computation of updates
    pub enable_precomputation: bool,

    /// Buffer size for pre-computed updates
    pub precomputation_buffer_size: usize,

    /// Enable lock-free algorithms
    pub enable_lock_free: bool,

    /// Use approximate algorithms for speed
    pub use_approximations: bool,

    /// Approximation tolerance
    pub approximation_tolerance: f64,

    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Batch processing threshold
    pub batch_threshold: usize,

    /// Enable zero-copy operations
    pub enable_zero_copy: bool,

    /// Memory pool size for allocations
    pub memory_pool_size: usize,

    /// Enable gradient quantization
    pub enable_quantization: bool,

    /// Quantization bits
    pub quantization_bits: u8,
}

impl Default for LowLatencyConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 100, // 100 microseconds
            max_latency_us: 1000,   // 1 millisecond
            enable_precomputation: true,
            precomputation_buffer_size: 64,
            enable_lock_free: true,
            use_approximations: true,
            approximation_tolerance: 0.01,
            enable_simd: true,
            batch_threshold: 8,
            enable_zero_copy: true,
            memory_pool_size: 1024 * 1024, // 1MB
            enable_quantization: false,
            quantization_bits: 8,
        }
    }
}

/// Low-latency streaming optimizer
pub struct LowLatencyOptimizer<O, A>
where
    A: Float + Send + Sync + scirs2_core::ndarray::ScalarOperand + std::fmt::Debug,
    O: Optimizer<A, scirs2_core::ndarray::Ix1> + Send + Sync,
{
    /// Base optimizer
    base_optimizer: Arc<Mutex<O>>,

    /// Configuration
    config: LowLatencyConfig,

    /// Live parameter vector (L1).
    ///
    /// The optimizer keeps the parameters it is optimizing so that every step
    /// is applied to the *result of the previous step*. Previously each step
    /// created a fresh `Array1::zeros(..)` as "current parameters", which meant
    /// the returned vector was always a single step away from the origin: the
    /// optimizer silently discarded all accumulated progress and, for any
    /// caller that stored the return value, effectively zeroed the parameters
    /// on every update. Seed real initial weights with [`Self::set_parameters`];
    /// if nothing is seeded the vector starts at the origin on the first step.
    parameters: Option<Array1<A>>,

    /// Pre-computation engine
    precomputation_engine: Option<PrecomputationEngine<A>>,

    /// Bounded staging ring for produced updates
    update_buffer: LockFreeBuffer<A>,

    /// Memory pool for fast allocations
    memory_pool: FastMemoryPool<A>,

    /// Chunked vector processor
    simd_processor: SIMDProcessor<A>,

    /// Quantization engine
    quantizer: Option<GradientQuantizer<A>>,

    /// Performance monitor
    perf_monitor: LatencyMonitor,

    /// Approximation controller
    approximation_controller: ApproximationController<A>,

    /// Step counter (atomic for thread safety)
    step_counter: AtomicUsize,
}

/// Pre-computation engine for preparing updates in advance
struct PrecomputationEngine<A: Float + Send + Sync> {
    /// Buffer of pre-computed updates
    precomputed_updates: VecDeque<PrecomputedUpdate<A>>,

    /// Prediction model for future gradients
    gradient_predictor: GradientPredictor<A>,

    /// Maximum buffer size
    max_buffer_size: usize,

    /// Number of steps that were served from a pre-computed update
    hits: usize,

    /// Number of steps that had to fall back to a full update
    misses: usize,

    /// Minimum recorded prediction confidence a pre-computed update must carry
    /// to be served.
    ///
    /// The predictor's measured confidence was stored on every entry and then
    /// never consulted, so a wild guess was served as readily as a well
    /// -supported prediction. Zero (the default) preserves that behaviour;
    /// raising it makes the engine fall back to a full update when the
    /// predictor is unsure.
    min_confidence: A,
}

/// Pre-computed update entry
#[derive(Debug, Clone)]
struct PrecomputedUpdate<A: Float + Send + Sync> {
    /// Predicted gradient
    gradient: Array1<A>,

    /// Pre-computed parameter update
    update: Array1<A>,

    /// Validity timestamp
    valid_until: Instant,

    /// Confidence score
    confidence: A,
}

/// Bounded staging ring for produced updates.
///
/// Accessed exclusively through `&mut self` from the owning optimizer, so it
/// needs no locking at all — hence "lock free". It is deliberately *not* a
/// concurrent MPMC queue: claiming that would require `unsafe` interior
/// mutability this module does not want in a real-time path.
struct LockFreeBuffer<A: Float + Send + Sync> {
    /// Buffer storage
    buffer: Vec<Option<Array1<A>>>,

    /// Write index (atomic)
    write_index: AtomicUsize,

    /// Read index (atomic)
    read_index: AtomicUsize,

    /// Buffer capacity
    capacity: usize,
}

/// Fast memory pool for low-latency allocations.
///
/// L5: the previous implementation held `Vec<*mut u8>` filled by
/// `std::alloc::alloc` with no `Drop`, so every dropped optimizer leaked its
/// whole pool (1 MB by default). Blocks are now owned `Vec<A>` buffers, which
/// release themselves when the pool is dropped — the leak is fixed by
/// ownership rather than by a hand-written `Drop`, and the module no longer
/// contains any `unsafe` code.
struct FastMemoryPool<A> {
    /// Currently free blocks, each pre-allocated to `elements_per_block`
    free_blocks: Mutex<Vec<Vec<A>>>,

    /// Elements per block
    elements_per_block: usize,

    /// Total blocks the pool was created with
    total_blocks: usize,

    /// Blocks currently checked out
    checked_out: AtomicUsize,

    /// High-water mark of simultaneously checked-out blocks
    peak_checked_out: AtomicUsize,

    /// Requests the pool could not satisfy (caller had to allocate)
    misses: AtomicUsize,
}

/// Chunked vector processor for the fast update path
struct SIMDProcessor<A: Float + Send + Sync> {
    /// Enable chunked processing
    enabled: bool,

    /// Chunk width used when walking the contiguous parameter slice
    vector_width: usize,

    /// Marker so the processor stays tied to the element type
    _element: std::marker::PhantomData<A>,
}

/// Gradient quantization for reduced precision
struct GradientQuantizer<A: Float + Send + Sync> {
    /// Quantization bits
    bits: u8,

    /// Quantization scale
    scale: A,

    /// Zero point
    zero_point: A,

    /// Quantization error accumulator (error feedback)
    error_accumulator: Option<Array1<A>>,
}

/// Latency monitoring and profiling
#[derive(Debug)]
struct LatencyMonitor {
    /// Recent latency samples
    latency_samples: VecDeque<Duration>,

    /// Maximum samples to keep
    maxsamples: usize,

    /// Current percentiles
    p50_latency: Duration,
    p95_latency: Duration,
    p99_latency: Duration,

    /// Violation count
    violations: usize,

    /// Total operations
    total_operations: usize,
}

/// Maximum age of a retained latency/accuracy measurement.
const PERFORMANCE_WINDOW_AGE: Duration = Duration::from_secs(30);

/// Maximum number of retained latency/accuracy measurements.
const PERFORMANCE_WINDOW_LEN: usize = 100;

/// Approximation controller for trading accuracy for speed
struct ApproximationController<A: Float + Send + Sync> {
    /// Current approximation level (0.0 = exact, 1.0 = maximum approximation)
    approximation_level: A,

    /// Performance history
    performance_history: VecDeque<PerformancePoint<A>>,

    /// Adaptation rate
    adaptation_rate: A,

    /// Target latency
    targetlatency: Duration,
}

/// Performance measurement point
#[derive(Debug, Clone)]
struct PerformancePoint<A: Float + Send + Sync> {
    /// Latency measurement
    latency: Duration,

    /// Accuracy achieved
    accuracy: A,

    /// Timestamp
    timestamp: Instant,
}

/// Gradient predictor for pre-computation
struct GradientPredictor<A: Float + Send + Sync> {
    /// Recent gradient history
    gradient_history: VecDeque<Array1<A>>,

    /// Per-coordinate least-squares slope of the observed history
    trend_weights: Option<Array1<A>>,

    /// History window size
    windowsize: usize,

    /// Measured prediction confidence (EWMA of cosine similarity between the
    /// last prediction and the gradient that actually arrived). `None` until
    /// at least one prediction has been scored against real data — the
    /// confidence is never seeded with an invented number.
    confidence: Option<A>,

    /// The prediction currently awaiting a real observation
    pending_prediction: Option<Array1<A>>,
}

impl<O, A> LowLatencyOptimizer<O, A>
where
    A: Float
        + Send
        + Sync
        + Default
        + Clone
        + std::fmt::Debug
        + scirs2_core::ndarray::ScalarOperand
        + 'static
        + std::iter::Sum,
    O: Optimizer<A, scirs2_core::ndarray::Ix1> + Send + Sync + 'static,
{
    /// Create a new low-latency optimizer
    pub fn new(_baseoptimizer: O, config: LowLatencyConfig) -> Result<Self> {
        let base_optimizer = Arc::new(Mutex::new(_baseoptimizer));

        let precomputation_engine = if config.enable_precomputation {
            Some(PrecomputationEngine::new(config.precomputation_buffer_size))
        } else {
            None
        };

        let update_buffer = LockFreeBuffer::new(config.precomputation_buffer_size);
        let memory_pool = FastMemoryPool::new(config.memory_pool_size, 4096)?; // 4KB blocks
        let simd_processor = SIMDProcessor::new(config.enable_simd, config.batch_threshold);

        let quantizer = if config.enable_quantization {
            Some(GradientQuantizer::new(config.quantization_bits))
        } else {
            None
        };

        let perf_monitor = LatencyMonitor::new(1000); // Keep 1000 samples
        let approximation_controller =
            ApproximationController::new(Duration::from_micros(config.target_latency_us));

        Ok(Self {
            base_optimizer,
            config,
            parameters: None,
            precomputation_engine,
            update_buffer,
            memory_pool,
            simd_processor,
            quantizer,
            perf_monitor,
            approximation_controller,
            step_counter: AtomicUsize::new(0),
        })
    }

    /// Seed the parameter vector the optimizer will keep updating.
    pub fn set_parameters(&mut self, parameters: Array1<A>) {
        self.parameters = Some(parameters);
    }

    /// Current parameter vector, if any step has been taken or seeded.
    /// Require a minimum predictor confidence before a pre-computed update is
    /// served, falling back to a full update below it.
    ///
    /// No-op when pre-computation is disabled. Defaults to zero, which accepts
    /// any prediction that matches the arriving gradient.
    pub fn set_precomputation_min_confidence(&mut self, min_confidence: A) {
        if let Some(precomp) = self.precomputation_engine.as_mut() {
            precomp.set_min_confidence(min_confidence);
        }
    }

    pub fn parameters(&self) -> Option<&Array1<A>> {
        self.parameters.as_ref()
    }

    /// Perform a low-latency update
    pub fn low_latency_step(&mut self, gradient: &Array1<A>) -> Result<Array1<A>> {
        let start_time = Instant::now();

        if gradient.is_empty() {
            return Err(OptimError::DimensionMismatch(
                "low_latency_step received an empty gradient".to_string(),
            ));
        }

        let previous_params = self.parameters.clone();
        let learning_rate = self.base_learning_rate();
        let tolerance = self.config.approximation_tolerance.max(0.0);

        // Speculative fast path: a pre-computed update is only served when the
        // gradient it was computed for actually matches the gradient that
        // arrived, within `approximation_tolerance`. That check is what makes
        // the reported hit rate a real measurement instead of a constant.
        let served = self
            .precomputation_engine
            .as_mut()
            .and_then(|precomp| precomp.try_get_precomputed(gradient, tolerance));
        if let Some(precomputed) = served {
            let update = precomputed.update;
            self.parameters = Some(update.clone());
            let latency = start_time.elapsed();
            self.perf_monitor.record_latency(latency);
            if self.config.enable_lock_free {
                self.update_buffer.push(update.clone());
            }
            let validity = Duration::from_micros(self.config.max_latency_us.max(1));
            if let Some(ref mut precomp) = self.precomputation_engine {
                precomp.start_precomputation(gradient, &update, learning_rate, validity);
            }
            self.step_counter.fetch_add(1, Ordering::Relaxed);
            return Ok(update);
        }

        // Quantize gradient if enabled. With zero-copy enabled and no
        // quantizer configured the original gradient is used in place, so the
        // hot path performs no defensive clone at all.
        let quantized = match self.quantizer.as_mut() {
            Some(quantizer) => Some(quantizer.quantize(gradient)?),
            None if self.config.enable_zero_copy => None,
            None => Some(gradient.clone()),
        };
        let processed_gradient: &Array1<A> = quantized.as_ref().unwrap_or(gradient);

        // Use approximation if necessary to meet latency budget
        let approximation_level = self.approximation_controller.get_approximation_level();
        let use_approximation = self.config.use_approximations && approximation_level > A::zero();
        let update = if use_approximation {
            let simplified = self.simplify_gradient(processed_gradient, approximation_level)?;
            self.fast_path_update(&simplified, learning_rate)?
        } else {
            self.exact_update(processed_gradient)?
        };

        let latency = start_time.elapsed();

        // Record performance and adapt approximation level
        let accuracy = Self::estimate_accuracy(previous_params.as_ref(), &update, gradient);
        self.approximation_controller
            .record_performance(latency, approximation_level, accuracy);
        self.perf_monitor.record_latency(latency);

        // Check for latency violations
        if latency.as_micros() as u64 > self.config.max_latency_us {
            self.handle_latency_violation(latency)?;
        }

        // Stage the produced update for asynchronous consumers.
        if self.config.enable_lock_free {
            self.update_buffer.push(update.clone());
        }

        // Prepare the next step's speculative update while the caller is busy
        // fetching its next sample.
        let validity = Duration::from_micros(self.config.max_latency_us.max(1));
        if let Some(ref mut precomp) = self.precomputation_engine {
            precomp.start_precomputation(gradient, &update, learning_rate, validity);
        }

        self.parameters = Some(update.clone());
        self.step_counter.fetch_add(1, Ordering::Relaxed);
        Ok(update)
    }

    /// Learning rate currently configured on the wrapped optimizer.
    fn base_learning_rate(&self) -> A {
        lock_recovered(&self.base_optimizer).get_learning_rate()
    }

    /// Parameter vector to step from, allocated at the origin on first use.
    fn current_parameters(&self, len: usize) -> Result<Array1<A>> {
        match self.parameters.as_ref() {
            Some(params) if params.len() == len => Ok(params.clone()),
            Some(params) => Err(OptimError::DimensionMismatch(format!(
                "gradient has {} elements but the tracked parameters have {}",
                len,
                params.len()
            ))),
            None => Ok(Array1::zeros(len)),
        }
    }

    /// Perform exact update using base optimizer
    fn exact_update(&mut self, gradient: &Array1<A>) -> Result<Array1<A>> {
        let current_params = self.current_parameters(gradient.len())?;
        let mut optimizer = lock_recovered(&self.base_optimizer);
        optimizer.step(&current_params, gradient)
    }

    /// Chunked first-order update used by the approximate / pre-computation
    /// paths.
    ///
    /// L3: this used to hand the gradient to a "SIMD processor" that returned
    /// `gradient.clone()`, so the approximate path returned the *gradient*
    /// where the caller expected *new parameters* and applied no update at
    /// all. It now performs a real chunked `params -= lr * gradient` walk over
    /// the contiguous parameter slice.
    fn fast_path_update(&mut self, gradient: &Array1<A>, learning_rate: A) -> Result<Array1<A>> {
        if !self.simd_processor.is_active(gradient.len()) {
            return self.exact_update(gradient);
        }
        let mut params = self.current_parameters(gradient.len())?;
        self.simd_processor
            .apply_scaled_subtract(&mut params, gradient, learning_rate);
        Ok(params)
    }

    /// Simplify gradient for approximation by keeping the largest magnitudes.
    fn simplify_gradient(&self, gradient: &Array1<A>, level: A) -> Result<Array1<A>> {
        let n = gradient.len();
        if n == 0 {
            return Ok(gradient.clone());
        }

        let sparsity_ratio = level.to_f64().unwrap_or(0.0).clamp(0.0, 1.0);
        let keep_ratio = 1.0 - sparsity_ratio * 0.8; // Keep 20% to 100% of gradients
        let keep_count = (((n as f64) * keep_ratio).round() as usize).clamp(1, n);
        if keep_count == n {
            return Ok(gradient.clone());
        }

        // Magnitudes go into a pooled scratch buffer so the hot path does not
        // allocate, and the k-th largest magnitude is found in linear time
        // instead of by fully sorting.
        let mut magnitudes = self
            .memory_pool
            .acquire(n)
            .unwrap_or_else(|| Vec::with_capacity(n));
        magnitudes.clear();
        magnitudes.extend(gradient.iter().map(|g| g.abs()));
        let kth = keep_count - 1;
        magnitudes.select_nth_unstable_by(kth, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = magnitudes[kth];
        self.memory_pool.release(magnitudes);

        let mut simplified = Array1::zeros(n);
        let mut kept = 0usize;
        for (i, &g) in gradient.iter().enumerate() {
            if kept < keep_count && g.abs() >= threshold {
                simplified[i] = g;
                kept += 1;
            }
        }

        Ok(simplified)
    }

    /// Estimate how well the applied step follows the true descent direction.
    ///
    /// The previous version compared the *new parameter vector* with the
    /// *gradient*, two quantities with no meaningful angle between them. The
    /// meaningful comparison is between the applied delta and `-gradient`.
    fn estimate_accuracy(
        previous_params: Option<&Array1<A>>,
        new_params: &Array1<A>,
        gradient: &Array1<A>,
    ) -> A {
        if new_params.len() != gradient.len() {
            return A::zero();
        }

        let zeros = Array1::zeros(new_params.len());
        let previous = match previous_params {
            Some(previous) if previous.len() == new_params.len() => previous,
            _ => &zeros,
        };

        let mut dot = A::zero();
        let mut norm_delta = A::zero();
        let mut norm_grad = A::zero();
        for ((&p_new, &p_old), &g) in new_params.iter().zip(previous.iter()).zip(gradient.iter()) {
            let delta = p_new - p_old;
            dot = dot + delta * (-g);
            norm_delta = norm_delta + delta * delta;
            norm_grad = norm_grad + g * g;
        }

        let norm_delta = norm_delta.sqrt();
        let norm_grad = norm_grad.sqrt();
        if norm_delta == A::zero() || norm_grad == A::zero() {
            A::zero()
        } else {
            dot / (norm_delta * norm_grad)
        }
    }

    /// Handle latency violations
    fn handle_latency_violation(&mut self, latency: Duration) -> Result<()> {
        // Record the violation so `LowLatencyMetrics::latency_violations` is a
        // real count rather than a permanent zero.
        self.perf_monitor.violations += 1;

        // Increase approximation level to reduce future latency
        self.approximation_controller.increase_approximation();

        // Escalate to gradient quantization when the budget is being missed by
        // a wide margin and quantization has not been enabled yet.
        if !self.config.enable_quantization
            && latency.as_micros() as u64 > self.config.max_latency_us.saturating_mul(2)
        {
            self.config.enable_quantization = true;
            self.quantizer = Some(GradientQuantizer::new(self.config.quantization_bits));
        }

        Ok(())
    }

    /// Take the oldest staged update, if any.
    pub fn try_pop_staged_update(&mut self) -> Option<Array1<A>> {
        self.update_buffer.pop()
    }

    /// Number of updates currently staged.
    pub fn staged_update_count(&self) -> usize {
        self.update_buffer.len()
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> LowLatencyMetrics {
        LowLatencyMetrics {
            avg_latency_us: self.perf_monitor.get_average_latency().as_micros() as u64,
            p50_latency_us: self.perf_monitor.p50_latency.as_micros() as u64,
            p95_latency_us: self.perf_monitor.p95_latency.as_micros() as u64,
            p99_latency_us: self.perf_monitor.p99_latency.as_micros() as u64,
            latency_violations: self.perf_monitor.violations,
            total_operations: self.perf_monitor.total_operations,
            current_approximation_level: self
                .approximation_controller
                .approximation_level
                .to_f64()
                .unwrap_or(0.0),
            approximation_accuracy: self
                .approximation_controller
                .mean_accuracy()
                .and_then(|value| value.to_f64()),
            precomputation_hit_rate: self
                .precomputation_engine
                .as_ref()
                .and_then(|pe| pe.hit_rate()),
            precomputation_attempts: self
                .precomputation_engine
                .as_ref()
                .map(|pe| pe.attempts())
                .unwrap_or(0),
            memory_efficiency: self.memory_pool.get_efficiency(),
            memory_pool_misses: self.memory_pool.misses(),
        }
    }

    /// Check if optimizer is meeting latency requirements
    pub fn is_meeting_latency_requirements(&self) -> bool {
        let avg_latency = self.perf_monitor.get_average_latency().as_micros() as u64;
        avg_latency <= self.config.target_latency_us
    }
}

// Implementation of helper structs
impl<A: Float + Send + Sync + std::iter::Sum> PrecomputationEngine<A> {
    fn new(_buffersize: usize) -> Self {
        let capacity = _buffersize.max(1);
        Self {
            precomputed_updates: VecDeque::with_capacity(capacity),
            gradient_predictor: GradientPredictor::new(10), // 10-step history
            max_buffer_size: capacity,
            hits: 0,
            misses: 0,
            min_confidence: A::zero(),
        }
    }

    /// Require at least `min_confidence` before a pre-computed update is used.
    fn set_min_confidence(&mut self, min_confidence: A) {
        self.min_confidence = min_confidence;
    }

    /// Serve a pre-computed update only when it was computed for a gradient
    /// that matches the one that actually arrived.
    fn try_get_precomputed(
        &mut self,
        actual_gradient: &Array1<A>,
        tolerance: f64,
    ) -> Option<PrecomputedUpdate<A>> {
        // Remove expired updates
        let now = Instant::now();
        while let Some(update) = self.precomputed_updates.front() {
            if update.valid_until <= now {
                self.precomputed_updates.pop_front();
            } else {
                break;
            }
        }

        let candidate = self.precomputed_updates.pop_front();
        self.gradient_predictor.observe(actual_gradient);

        match candidate {
            Some(candidate)
                if candidate.confidence >= self.min_confidence
                    && gradient_matches(&candidate.gradient, actual_gradient, tolerance) =>
            {
                self.hits += 1;
                Some(candidate)
            }
            _ => {
                self.misses += 1;
                None
            }
        }
    }

    /// Predict the next gradient and pre-compute the corresponding first-order
    /// update.
    ///
    /// The stored update is a first-order (`params - lr * predicted_gradient`)
    /// approximation of the wrapped optimizer's step, which is why it is only
    /// ever served when the predicted gradient turns out to match the real one
    /// within the configured tolerance.
    fn start_precomputation(
        &mut self,
        _observed_gradient: &Array1<A>,
        current_params: &Array1<A>,
        learning_rate: A,
        validity: Duration,
    ) {
        let Some((predicted, confidence)) = self.gradient_predictor.predict() else {
            return;
        };
        if predicted.len() != current_params.len() {
            return;
        }

        let mut update = current_params.clone();
        for (p, &g) in update.iter_mut().zip(predicted.iter()) {
            *p = *p - learning_rate * g;
        }

        if self.precomputed_updates.len() >= self.max_buffer_size {
            self.precomputed_updates.pop_front();
        }
        self.precomputed_updates.push_back(PrecomputedUpdate {
            gradient: predicted,
            update,
            valid_until: Instant::now() + validity,
            confidence,
        });
    }

    fn attempts(&self) -> usize {
        self.hits + self.misses
    }

    /// Measured hit rate, or `None` when no step has consulted the engine yet.
    fn hit_rate(&self) -> Option<f64> {
        let attempts = self.attempts();
        if attempts == 0 {
            None
        } else {
            Some(self.hits as f64 / attempts as f64)
        }
    }
}

/// Relative agreement test used to decide whether a speculative update is
/// still valid for the gradient that arrived.
fn gradient_matches<A: Float>(predicted: &Array1<A>, actual: &Array1<A>, tolerance: f64) -> bool {
    if predicted.len() != actual.len() || predicted.is_empty() {
        return false;
    }
    let mut diff_sq = A::zero();
    let mut actual_sq = A::zero();
    for (&p, &a) in predicted.iter().zip(actual.iter()) {
        let d = p - a;
        diff_sq = diff_sq + d * d;
        actual_sq = actual_sq + a * a;
    }
    let diff = diff_sq.sqrt().to_f64().unwrap_or(f64::INFINITY);
    let scale = actual_sq.sqrt().to_f64().unwrap_or(0.0);
    if !diff.is_finite() {
        return false;
    }
    if scale <= f64::EPSILON {
        diff <= tolerance
    } else {
        diff / scale <= tolerance
    }
}

impl<A: Float + Send + Sync> LockFreeBuffer<A> {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            buffer: vec![None; capacity],
            write_index: AtomicUsize::new(0),
            read_index: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Stage an update, dropping the oldest entry when the ring is full.
    fn push(&mut self, value: Array1<A>) {
        let write = self.write_index.load(Ordering::Relaxed);
        let read = self.read_index.load(Ordering::Relaxed);
        if write - read >= self.capacity {
            // Ring is full: advance the reader, discarding the oldest entry.
            let slot = read % self.capacity;
            self.buffer[slot] = None;
            self.read_index.store(read + 1, Ordering::Relaxed);
        }
        let slot = write % self.capacity;
        self.buffer[slot] = Some(value);
        self.write_index.store(write + 1, Ordering::Relaxed);
    }

    fn pop(&mut self) -> Option<Array1<A>> {
        let read = self.read_index.load(Ordering::Relaxed);
        if read == self.write_index.load(Ordering::Relaxed) {
            return None;
        }
        let slot = read % self.capacity;
        let value = self.buffer[slot].take();
        self.read_index.store(read + 1, Ordering::Relaxed);
        value
    }

    fn len(&self) -> usize {
        self.write_index.load(Ordering::Relaxed) - self.read_index.load(Ordering::Relaxed)
    }
}

impl<A: Float> FastMemoryPool<A> {
    fn new(_total_size: usize, block_size_bytes: usize) -> Result<Self> {
        let element_size = std::mem::size_of::<A>().max(1);
        let elements_per_block = (block_size_bytes / element_size).max(1);
        let total_blocks = _total_size / block_size_bytes.max(1);

        let mut free_blocks = Vec::with_capacity(total_blocks);
        for _ in 0..total_blocks {
            free_blocks.push(Vec::with_capacity(elements_per_block));
        }

        Ok(Self {
            free_blocks: Mutex::new(free_blocks),
            elements_per_block,
            total_blocks,
            checked_out: AtomicUsize::new(0),
            peak_checked_out: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        })
    }

    /// Check out a pre-allocated scratch buffer able to hold `len` elements.
    fn acquire(&self, len: usize) -> Option<Vec<A>> {
        if len > self.elements_per_block {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let block = lock_recovered(&self.free_blocks).pop();
        match block {
            Some(mut block) => {
                block.clear();
                let in_use = self.checked_out.fetch_add(1, Ordering::Relaxed) + 1;
                self.peak_checked_out.fetch_max(in_use, Ordering::Relaxed);
                Some(block)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Return a buffer previously obtained from [`Self::acquire`].
    fn release(&self, mut block: Vec<A>) {
        if block.capacity() < self.elements_per_block {
            // Not one of ours (the caller allocated it) — just drop it.
            return;
        }
        block.clear();
        let mut free = lock_recovered(&self.free_blocks);
        if free.len() < self.total_blocks {
            free.push(block);
            drop(free);
            let previous = self.checked_out.load(Ordering::Relaxed);
            if previous > 0 {
                self.checked_out.store(previous - 1, Ordering::Relaxed);
            }
        }
    }

    /// Fraction of the pool that has actually been exercised (high-water mark
    /// of simultaneously checked-out blocks). Returns `0.0` for an empty pool
    /// instead of dividing by zero.
    fn get_efficiency(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.peak_checked_out.load(Ordering::Relaxed) as f64 / self.total_blocks as f64
    }

    fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }
}

impl<A: Float + Send + Sync> SIMDProcessor<A> {
    fn new(enabled: bool, batch_threshold: usize) -> Self {
        Self {
            enabled,
            vector_width: batch_threshold.max(1),
            _element: std::marker::PhantomData,
        }
    }

    /// The chunked path only pays for itself once there is at least one full
    /// chunk of work, which is exactly what `batch_threshold` configures.
    fn is_active(&self, len: usize) -> bool {
        self.enabled && len >= self.vector_width
    }

    /// `params -= learning_rate * gradient`, walked in contiguous chunks so
    /// the inner loop is a fixed-width, auto-vectorizable kernel.
    fn apply_scaled_subtract(
        &self,
        params: &mut Array1<A>,
        gradient: &Array1<A>,
        learning_rate: A,
    ) {
        let width = self.vector_width.max(1);
        match (params.as_slice_mut(), gradient.as_slice()) {
            (Some(p), Some(g)) => {
                for (p_chunk, g_chunk) in p.chunks_mut(width).zip(g.chunks(width)) {
                    for (p_value, g_value) in p_chunk.iter_mut().zip(g_chunk.iter()) {
                        *p_value = *p_value - learning_rate * *g_value;
                    }
                }
            }
            _ => {
                for (p_value, g_value) in params.iter_mut().zip(gradient.iter()) {
                    *p_value = *p_value - learning_rate * *g_value;
                }
            }
        }
    }
}

impl<A: Float + Send + Sync> GradientQuantizer<A> {
    fn new(bits: u8) -> Self {
        Self {
            // 1..=24 keeps `1 << (bits - 1)` well inside `u32` and keeps the
            // level count non-zero, so the scale can never become 0.
            bits: bits.clamp(1, 24),
            scale: A::one(),
            zero_point: A::zero(),
            error_accumulator: None,
        }
    }

    /// Symmetric linear quantization with error feedback.
    ///
    /// L4: the previous version computed `scale = max_abs / levels` and then
    /// divided by it unconditionally. For an all-zero gradient (a normal
    /// occurrence once a stream converges, and the default state of a freshly
    /// initialised model) `max_abs` is 0, so every element became `0/0 = NaN`
    /// and the NaN propagated into the parameters. `bits = 0` produced the
    /// same division by zero via `2^0 - 1 = 0`.
    fn quantize(&mut self, gradient: &Array1<A>) -> Result<Array1<A>> {
        let n = gradient.len();
        if n == 0 {
            return Ok(gradient.clone());
        }

        // Error feedback: carry the previous step's rounding residual forward
        // so quantization does not introduce a systematic bias.
        let compensated = match self.error_accumulator.as_ref() {
            Some(error) if error.len() == n => gradient + error,
            _ => gradient.clone(),
        };

        // NaN loses every `>` comparison, so a fold-based maximum silently
        // ignores it; the window has to be scanned for finiteness explicitly.
        if compensated.iter().any(|value| !value.is_finite()) {
            return Err(OptimError::InvalidParameter(
                "cannot quantize a gradient containing non-finite values".to_string(),
            ));
        }
        let max_abs = compensated.iter().fold(
            A::zero(),
            |acc, x| if x.abs() > acc { x.abs() } else { acc },
        );

        self.zero_point = A::zero(); // symmetric quantization
        if max_abs == A::zero() {
            // Nothing to quantize; the representation is exact.
            self.scale = A::one();
            self.error_accumulator = Some(Array1::zeros(n));
            return Ok(compensated);
        }

        let level_count = (1u32 << (self.bits.max(1) as u32 - 1))
            .saturating_sub(1)
            .max(1);
        let levels = A::from(level_count).unwrap_or(A::one());
        self.scale = max_abs / levels;
        let scale = self.scale;
        let zero_point = self.zero_point;

        let quantized = compensated.mapv(|x| {
            let mut q = (x / scale).round();
            if q > levels {
                q = levels;
            } else if q < -levels {
                q = -levels;
            }
            q * scale + zero_point
        });

        self.error_accumulator = Some(&compensated - &quantized);
        Ok(quantized)
    }
}

impl LatencyMonitor {
    fn new(maxsamples: usize) -> Self {
        Self {
            latency_samples: VecDeque::with_capacity(maxsamples),
            maxsamples: maxsamples.max(1),
            p50_latency: Duration::from_micros(0),
            p95_latency: Duration::from_micros(0),
            p99_latency: Duration::from_micros(0),
            violations: 0,
            total_operations: 0,
        }
    }

    fn record_latency(&mut self, latency: Duration) {
        self.latency_samples.push_back(latency);
        if self.latency_samples.len() > self.maxsamples {
            self.latency_samples.pop_front();
        }

        self.total_operations += 1;
        self.update_percentiles();
    }

    fn update_percentiles(&mut self) {
        if self.latency_samples.is_empty() {
            return;
        }

        let mut sorted: Vec<_> = self.latency_samples.iter().cloned().collect();
        sorted.sort();

        let last = sorted.len() - 1;
        let index_for = |q: f64| ((sorted.len() as f64 * q) as usize).min(last);
        self.p50_latency = sorted[index_for(0.50)];
        self.p95_latency = sorted[index_for(0.95)];
        self.p99_latency = sorted[index_for(0.99)];
    }

    fn get_average_latency(&self) -> Duration {
        if self.latency_samples.is_empty() {
            Duration::from_micros(0)
        } else {
            let total: Duration = self.latency_samples.iter().sum();
            total / self.latency_samples.len() as u32
        }
    }
}

impl<A: Float + Send + Sync> ApproximationController<A> {
    fn new(targetlatency: Duration) -> Self {
        Self {
            approximation_level: A::zero(),
            performance_history: VecDeque::with_capacity(100),
            adaptation_rate: A::from(0.1).unwrap_or_else(A::one),
            targetlatency,
        }
    }

    fn get_approximation_level(&self) -> A {
        self.approximation_level
    }

    fn record_performance(&mut self, latency: Duration, _approximation_level: A, accuracy: A) {
        let now = Instant::now();
        let point = PerformancePoint {
            latency,
            accuracy,
            timestamp: now,
        };

        self.performance_history.push_back(point);
        // Bound the window by age as well as by count: a controller that reacts
        // to latencies measured minutes ago is chasing a workload that no
        // longer exists. `timestamp` was recorded for exactly this and never
        // read.
        while self
            .performance_history
            .front()
            .is_some_and(|p| now.duration_since(p.timestamp) > PERFORMANCE_WINDOW_AGE)
        {
            self.performance_history.pop_front();
        }
        if self.performance_history.len() > PERFORMANCE_WINDOW_LEN {
            self.performance_history.pop_front();
        }

        self.adapt_approximation_level();
    }

    /// Mean latency over the retained window, or `None` when it is empty.
    fn mean_latency(&self) -> Option<Duration> {
        let count = self.performance_history.len();
        if count == 0 {
            return None;
        }
        let total: Duration = self.performance_history.iter().map(|p| p.latency).sum();
        Some(total / count as u32)
    }

    /// Move the approximation level towards the latency target.
    ///
    /// Driven by the *mean* latency of the retained window rather than the
    /// single latest sample: every latency was already being recorded but only
    /// the newest one was ever looked at, so one unlucky slow step swung the
    /// approximation level as hard as a sustained regression.
    fn adapt_approximation_level(&mut self) {
        let Some(latency) = self.mean_latency() else {
            return;
        };
        let target = self.targetlatency.as_micros().max(1) as f64;
        let latency_ratio = latency.as_micros() as f64 / target;

        if latency_ratio > 1.1 {
            // Latency too high, increase approximation
            self.approximation_level =
                (self.approximation_level + self.adaptation_rate).min(A::one());
        } else if latency_ratio < 0.8 {
            // Latency low, can reduce approximation
            self.approximation_level =
                (self.approximation_level - self.adaptation_rate).max(A::zero());
        }
    }

    fn increase_approximation(&mut self) {
        let double = A::from(2.0).unwrap_or_else(A::one);
        self.approximation_level =
            (self.approximation_level + self.adaptation_rate * double).min(A::one());
    }

    /// Mean accuracy observed over the retained performance window.
    fn mean_accuracy(&self) -> Option<A> {
        if self.performance_history.is_empty() {
            return None;
        }
        let count = A::from(self.performance_history.len())?;
        let sum = self
            .performance_history
            .iter()
            .fold(A::zero(), |acc, point| acc + point.accuracy);
        Some(sum / count)
    }
}

impl<A: Float + Send + Sync + std::iter::Sum> GradientPredictor<A> {
    fn new(windowsize: usize) -> Self {
        Self {
            gradient_history: VecDeque::with_capacity(windowsize.max(2)),
            trend_weights: None,
            windowsize: windowsize.max(2),
            confidence: None,
            pending_prediction: None,
        }
    }

    /// Record the gradient that actually arrived and score the outstanding
    /// prediction against it.
    fn observe(&mut self, gradient: &Array1<A>) {
        if let Some(prediction) = self.pending_prediction.take() {
            if prediction.len() == gradient.len() {
                let similarity = cosine_similarity(&prediction, gradient);
                let alpha = A::from(0.2).unwrap_or_else(A::one);
                self.confidence = Some(match self.confidence {
                    Some(previous) => previous * (A::one() - alpha) + similarity * alpha,
                    None => similarity,
                });
            }
        }

        self.gradient_history.push_back(gradient.clone());
        while self.gradient_history.len() > self.windowsize {
            self.gradient_history.pop_front();
        }
        self.recompute_trend();
    }

    /// Per-coordinate ordinary-least-squares slope over the retained window.
    fn recompute_trend(&mut self) {
        let n = self.gradient_history.len();
        if n < 2 {
            self.trend_weights = None;
            return;
        }
        let dim = match self.gradient_history.back() {
            Some(last) => last.len(),
            None => return,
        };
        if self.gradient_history.iter().any(|g| g.len() != dim) {
            self.trend_weights = None;
            return;
        }

        // x = 0..n-1, so sum(x) and sum((x - x_mean)^2) are closed forms.
        let n_f = A::from(n).unwrap_or_else(A::one);
        let x_mean = A::from((n - 1) as f64 / 2.0).unwrap_or_else(A::zero);
        let mut denominator = A::zero();
        for i in 0..n {
            let dx = A::from(i).unwrap_or_else(A::zero) - x_mean;
            denominator = denominator + dx * dx;
        }
        if denominator == A::zero() {
            self.trend_weights = None;
            return;
        }

        let mut slopes = Array1::zeros(dim);
        for coordinate in 0..dim {
            let mut y_sum = A::zero();
            for gradient in &self.gradient_history {
                y_sum = y_sum + gradient[coordinate];
            }
            let y_mean = y_sum / n_f;
            let mut numerator = A::zero();
            for (i, gradient) in self.gradient_history.iter().enumerate() {
                let dx = A::from(i).unwrap_or_else(A::zero) - x_mean;
                numerator = numerator + dx * (gradient[coordinate] - y_mean);
            }
            slopes[coordinate] = numerator / denominator;
        }
        self.trend_weights = Some(slopes);
    }

    /// Linear extrapolation of the next gradient, with the measured
    /// confidence of the previous prediction.
    fn predict(&mut self) -> Option<(Array1<A>, A)> {
        let last = self.gradient_history.back()?.clone();
        let slopes = self.trend_weights.as_ref()?;
        if slopes.len() != last.len() {
            return None;
        }
        let mut predicted = last;
        for (value, &slope) in predicted.iter_mut().zip(slopes.iter()) {
            *value = *value + slope;
        }
        self.pending_prediction = Some(predicted.clone());
        // Until a prediction has been scored there is no measured confidence;
        // report zero rather than inventing one.
        let confidence = self.confidence.unwrap_or_else(A::zero);
        Some((predicted, confidence))
    }
}

/// Cosine similarity between two equally sized vectors, `0` when either is
/// degenerate.
fn cosine_similarity<A: Float>(a: &Array1<A>, b: &Array1<A>) -> A {
    if a.len() != b.len() {
        return A::zero();
    }
    let mut dot = A::zero();
    let mut norm_a = A::zero();
    let mut norm_b = A::zero();
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot = dot + x * y;
        norm_a = norm_a + x * x;
        norm_b = norm_b + y * y;
    }
    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();
    if norm_a == A::zero() || norm_b == A::zero() {
        A::zero()
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Performance metrics for low-latency optimization
#[derive(Debug, Clone)]
pub struct LowLatencyMetrics {
    /// Average latency (microseconds)
    pub avg_latency_us: u64,
    /// Median latency (microseconds)
    pub p50_latency_us: u64,
    /// 95th percentile latency (microseconds)
    pub p95_latency_us: u64,
    /// 99th percentile latency (microseconds)
    pub p99_latency_us: u64,
    /// Number of latency violations
    pub latency_violations: usize,
    /// Total operations performed
    pub total_operations: usize,
    /// Current approximation level (0.0 to 1.0)
    pub current_approximation_level: f64,
    /// Mean cosine agreement between the applied step and the descent
    /// direction over the retained window, or `None` before the first step.
    pub approximation_accuracy: Option<f64>,
    /// Measured pre-computation hit rate, or `None` when pre-computation is
    /// disabled or has not been consulted yet.
    pub precomputation_hit_rate: Option<f64>,
    /// Number of steps that consulted the pre-computation engine
    pub precomputation_attempts: usize,
    /// Fraction of the memory pool that has been exercised
    pub memory_efficiency: f64,
    /// Scratch requests the memory pool could not satisfy
    pub memory_pool_misses: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;

    #[test]
    fn test_low_latency_config() {
        let config = LowLatencyConfig::default();
        assert_eq!(config.target_latency_us, 100);
        assert!(config.enable_precomputation);
        assert!(config.enable_lock_free);
    }

    #[test]
    fn test_low_latency_optimizer_creation() {
        let sgd = SGD::new(0.01f64);
        let config = LowLatencyConfig::default();
        let result = LowLatencyOptimizer::new(sgd, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_latency_monitor() {
        let mut monitor = LatencyMonitor::new(10);

        for i in 1..=5 {
            monitor.record_latency(Duration::from_micros(i * 100));
        }

        assert_eq!(monitor.total_operations, 5);
        assert!(monitor.get_average_latency().as_micros() > 0);
    }

    #[test]
    fn test_gradient_quantizer() {
        let mut quantizer = GradientQuantizer::new(8);
        let gradient = Array1::from_vec(vec![0.1f64, 0.5, -0.3, 0.8]);

        let result = quantizer.quantize(&gradient);
        assert!(result.is_ok());

        let quantized = result.expect("quantization of a finite gradient must succeed");
        assert_eq!(quantized.len(), gradient.len());
    }

    #[test]
    fn test_approximation_controller() {
        let mut controller = ApproximationController::new(Duration::from_micros(100));

        // Record high latency - should increase approximation
        controller.record_performance(Duration::from_micros(200), 0.0f64, 0.9f64);

        assert!(controller.get_approximation_level() > 0.0);
    }

    #[test]
    fn test_lock_free_buffer() {
        let buffer = LockFreeBuffer::<f64>::new(4);
        assert_eq!(buffer.capacity, 4);
        assert_eq!(buffer.write_index.load(Ordering::Relaxed), 0);
        assert_eq!(buffer.read_index.load(Ordering::Relaxed), 0);
    }
}
