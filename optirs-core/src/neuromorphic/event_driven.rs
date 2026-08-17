// Event-Driven Optimization Algorithms
//
// This module implements event-driven optimization algorithms that process
// updates asynchronously based on neuromorphic events, designed for
// neuromorphic computing platforms with event-based architectures.

use super::{
    to_generic_or, EventPriority, MembraneDynamicsConfig, NeuromorphicEvent, NeuromorphicMetrics,
    PlasticityModel, STDPConfig, Spike, SpikeTrain,
};
use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use scirs2_core::numeric::Float;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// --- Pure-Rust varint helpers for event (de)serialization (F57) --------
//
// LEB128 unsigned varints, with zigzag encoding for signed values. Used by
// `EventCompressionEngine` to produce real, decodable bytes for events
// instead of fixed-size placeholder buffers.

fn write_uvarint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn read_uvarint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = *buf.get(*pos)?;
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    Some(result)
}

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn write_ivarint(buf: &mut Vec<u8>, value: i64) {
    write_uvarint(buf, zigzag_encode(value));
}

fn read_ivarint(buf: &[u8], pos: &mut usize) -> Option<i64> {
    read_uvarint(buf, pos).map(zigzag_decode)
}

/// Fixed-point scale used to convert floating-point event fields
/// (timestamps, neuron ids treated as integers, values, energy costs) to
/// integers before varint encoding. Millisecond timestamps are kept to
/// microsecond resolution.
const EVENT_FIXED_POINT_SCALE: f64 = 1000.0;

fn decode_event_type(byte: u8) -> Result<EventType> {
    match byte {
        0 => Ok(EventType::Spike),
        1 => Ok(EventType::WeightUpdate),
        2 => Ok(EventType::ThresholdCrossing),
        3 => Ok(EventType::PlasticityEvent),
        4 => Ok(EventType::ExternalStimulus),
        5 => Ok(EventType::TimerEvent),
        6 => Ok(EventType::ErrorEvent),
        7 => Ok(EventType::HomeostaticEvent),
        8 => Ok(EventType::SynchronizationEvent),
        9 => Ok(EventType::EnergyEvent),
        other => Err(OptimError::InvalidConfig(format!(
            "unknown encoded EventType discriminant: {other}"
        ))),
    }
}

fn decode_priority(byte: u8) -> Result<EventPriority> {
    match byte {
        0 => Ok(EventPriority::Low),
        1 => Ok(EventPriority::Normal),
        2 => Ok(EventPriority::High),
        3 => Ok(EventPriority::Critical),
        4 => Ok(EventPriority::RealTime),
        other => Err(OptimError::InvalidConfig(format!(
            "unknown encoded EventPriority discriminant: {other}"
        ))),
    }
}

fn truncated_bytes_err() -> crate::error::OptimError {
    OptimError::InvalidConfig("truncated compressed event bytes".to_string())
}

/// Event types for neuromorphic computing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Spike event from a neuron
    Spike,

    /// Synaptic weight update event
    WeightUpdate,

    /// Membrane potential threshold crossing
    ThresholdCrossing,

    /// Plasticity-triggered event
    PlasticityEvent,

    /// External stimulus event
    ExternalStimulus,

    /// Timer-based event
    TimerEvent,

    /// Error backpropagation event
    ErrorEvent,

    /// Homeostatic adaptation event
    HomeostaticEvent,

    /// Population synchronization event
    SynchronizationEvent,

    /// Energy budget event
    EnergyEvent,
}

/// Event-driven optimization configuration
#[derive(Debug, Clone)]
pub struct EventDrivenConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum event queue size
    pub max_queue_size: usize,

    /// Event processing timeout (ms)
    pub processing_timeout: T,

    /// Enable event priority scheduling
    pub priority_scheduling: bool,

    /// Event filtering threshold
    pub event_threshold: T,

    /// Enable event batching
    pub event_batching: bool,

    /// Batch size for event processing
    pub batch_size: usize,

    /// Enable temporal event correlation
    pub temporal_correlation: bool,

    /// Temporal correlation window (ms)
    pub correlation_window: T,

    /// Enable adaptive event handling
    pub adaptive_handling: bool,

    /// Event rate limits (events/second)
    pub rate_limits: HashMap<EventType, T>,

    /// Enable event compression
    pub event_compression: bool,

    /// Compression algorithm
    pub compression_algorithm: EventCompressionAlgorithm,

    /// Enable distributed event processing
    pub distributed_processing: bool,

    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Event compression algorithms
#[derive(Debug, Clone, Copy)]
pub enum EventCompressionAlgorithm {
    /// No compression
    None,

    /// Delta encoding
    DeltaEncoding,

    /// Huffman encoding
    HuffmanEncoding,

    /// Run-length encoding
    RunLengthEncoding,

    /// Sparse encoding
    SparseEncoding,

    /// Predictive encoding
    PredictiveEncoding,
}

/// Load balancing strategies for distributed event processing
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,

    /// Event type-based partitioning
    TypeBased,

    /// Load-aware distribution
    LoadAware,

    /// Locality-aware distribution
    LocalityAware,

    /// Dynamic load balancing
    Dynamic,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for EventDrivenConfig<T> {
    fn default() -> Self {
        let mut rate_limits = HashMap::new();
        rate_limits.insert(
            EventType::Spike,
            T::from(1000.0).unwrap_or_else(|| T::zero()),
        );
        rate_limits.insert(
            EventType::WeightUpdate,
            T::from(100.0).unwrap_or_else(|| T::zero()),
        );
        rate_limits.insert(
            EventType::PlasticityEvent,
            T::from(50.0).unwrap_or_else(|| T::zero()),
        );

        Self {
            max_queue_size: 10000,
            processing_timeout: T::from(1.0).unwrap_or_else(|| T::zero()),
            priority_scheduling: true,
            event_threshold: T::from(0.001).unwrap_or_else(|| T::zero()),
            event_batching: true,
            batch_size: 32,
            temporal_correlation: true,
            correlation_window: T::from(10.0).unwrap_or_else(|| T::zero()),
            adaptive_handling: true,
            rate_limits,
            event_compression: false,
            compression_algorithm: EventCompressionAlgorithm::None,
            distributed_processing: false,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Priority queue entry for event scheduling
#[derive(Debug, Clone)]
struct PriorityEventEntry<T: Float + Debug + Send + Sync + 'static> {
    event: NeuromorphicEvent<T>,
    insertion_time: Instant,
}

impl<T: Float + Debug + Send + Sync + 'static> PartialEq for PriorityEventEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.event.priority == other.event.priority
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Eq for PriorityEventEntry<T> {}

impl<T: Float + Debug + Send + Sync + 'static> PartialOrd for PriorityEventEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Ord for PriorityEventEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` is a max-heap: `pop()` returns the "greatest"
        // element. We want higher `EventPriority` to pop first, so
        // compare priorities directly rather than reversed (F55: the
        // previous `other.priority.cmp(&self.priority)` inverted this,
        // so pop() returned the LOWEST-priority event first). Ties break
        // FIFO: the earlier `insertion_time` must compare as "greater" so
        // it pops first, which is exactly what wrapping both sides in
        // `Reverse` gives us.
        self.event
            .priority
            .cmp(&other.event.priority)
            .then_with(|| Reverse(self.insertion_time).cmp(&Reverse(other.insertion_time)))
    }
}

/// Event-driven optimizer
pub struct EventDrivenOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration
    config: EventDrivenConfig<T>,

    /// STDP configuration
    stdp_config: STDPConfig<T>,

    /// Membrane dynamics configuration
    membrane_config: MembraneDynamicsConfig<T>,

    /// Event queue with priority scheduling
    event_queue: BinaryHeap<PriorityEventEntry<T>>,

    /// Event processing statistics
    event_stats: HashMap<EventType, EventStatistics<T>>,

    /// Current system state
    system_state: SystemState<T>,

    /// Event handlers
    event_handlers: HashMap<EventType, Box<dyn EventHandler<T>>>,

    /// Temporal correlation tracker
    correlation_tracker: TemporalCorrelationTracker<T>,

    /// Event rate limiter
    rate_limiter: EventRateLimiter<T>,

    /// Performance metrics
    metrics: NeuromorphicMetrics<T>,

    /// Distributed processing coordinator
    distributed_coordinator: Option<DistributedEventCoordinator<T>>,

    /// Reference (uncompressed) codec, used to measure how many bytes each
    /// event would occupy without compression so the achieved compression
    /// ratio reported by [`EventDrivenOptimizer::compression_ratio`] is a
    /// real measurement rather than an estimate.
    compression_engine: EventCompressionEngine<T>,

    /// Compressed event storage, one FIFO chain per [`EventPriority`] (F57).
    /// Used instead of `event_queue` while `config.event_compression` is
    /// enabled: enqueued events are compressed to bytes immediately and only
    /// reconstructed when they are actually processed.
    compressed_chains: BTreeMap<EventPriority, CompressedEventChain<T>>,

    /// Cumulative uncompressed size of every event that entered the
    /// compressed queue.
    compression_raw_bytes: usize,

    /// Cumulative compressed size of every event that entered the compressed
    /// queue.
    compression_compressed_bytes: usize,

    /// Adaptive handler
    adaptive_handler: AdaptiveEventHandler<T>,
}

/// Event processing statistics
#[derive(Debug, Clone)]
pub struct EventStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total events processed
    pub total_processed: usize,

    /// Average processing time (ms)
    pub avg_processing_time: T,

    /// Event rate (events/second)
    pub event_rate: T,

    /// Queue wait time (ms)
    pub avg_queue_wait_time: T,

    /// Error count
    pub error_count: usize,

    /// Last update time
    pub last_update: Instant,
}

/// System state for event-driven optimization
#[derive(Debug, Clone)]
pub struct SystemState<T: Float + Debug + Send + Sync + 'static> {
    /// Current membrane potentials
    pub membrane_potentials: Array1<T>,

    /// Synaptic weights
    pub synaptic_weights: Array2<T>,

    /// Last spike times
    pub last_spike_times: Array1<T>,

    /// Refractory states
    pub refractory_until: Array1<T>,

    /// Current simulation time
    pub current_time: T,

    /// Active neurons
    pub active_neurons: HashSet<usize>,

    /// Pending weight updates
    pub pending_updates: HashMap<(usize, usize), T>,
}

/// Event handler trait
trait EventHandler<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    fn handle_event(
        &mut self,
        event: &NeuromorphicEvent<T>,
        state: &mut SystemState<T>,
    ) -> Result<()>;
    fn can_handle(&self, eventtype: EventType) -> bool;
}

/// Spike event handler
struct SpikeEventHandler<T: Float + Debug + Send + Sync + 'static> {
    stdp_config: STDPConfig<T>,
    membrane_config: MembraneDynamicsConfig<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> EventHandler<T> for SpikeEventHandler<T> {
    fn handle_event(
        &mut self,
        event: &NeuromorphicEvent<T>,
        state: &mut SystemState<T>,
    ) -> Result<()> {
        let neuron_id = event.source_neuron;

        // Generate spike
        if neuron_id < state.membrane_potentials.len() {
            // Reset membrane potential
            state.membrane_potentials[neuron_id] = self.membrane_config.reset_potential;

            // Set refractory period
            state.refractory_until[neuron_id] =
                state.current_time + self.membrane_config.refractory_period;

            // Update last spike time
            state.last_spike_times[neuron_id] = state.current_time;

            // Add to active neurons
            state.active_neurons.insert(neuron_id);

            // Trigger STDP updates for connected synapses
            self.trigger_stdp_updates(neuron_id, state)?;
        }

        Ok(())
    }

    fn can_handle(&self, event_type: EventType) -> bool {
        event_type == EventType::Spike
    }
}

impl<T: Float + Debug + Send + Sync + 'static> SpikeEventHandler<T> {
    fn trigger_stdp_updates(&self, post_neuron: usize, state: &mut SystemState<T>) -> Result<()> {
        let long_ago = to_generic_or(-1000.0, T::zero());
        let now = state.current_time;

        for other_neuron in 0..state.last_spike_times.len() {
            if other_neuron == post_neuron {
                continue;
            }
            let other_spike_time = state.last_spike_times[other_neuron];
            if other_spike_time <= long_ago {
                continue; // no valid spike history for `other_neuron` yet
            }

            // `other_neuron` fired before `post_neuron` (now): it is PRE,
            // dt = t_post - t_pre > 0 => potentiation (LTP) on
            // other_neuron -> post_neuron.
            let dt_ltp = now - other_spike_time;
            let ltp = self.compute_stdp_weight_change(dt_ltp);
            Self::accumulate_pending(state, (other_neuron, post_neuron), ltp);

            // `post_neuron` is firing NOW, arriving after
            // `other_neuron`'s last spike: from `other_neuron`'s
            // perspective as POST, this is a PRE spike arriving late,
            // dt = t_post - t_pre = other_spike_time - now < 0 =>
            // depression (LTD) on post_neuron -> other_neuron. This is
            // the presynaptic-trace side of STDP that was previously
            // unreachable (F50): `dt` computed only from "post's own time
            // minus pre's last (necessarily past) spike time" is always
            // >= 0, so LTD never fired.
            let dt_ltd = other_spike_time - now;
            let ltd = self.compute_stdp_weight_change(dt_ltd);
            Self::accumulate_pending(state, (post_neuron, other_neuron), ltd);
        }

        Ok(())
    }

    /// Accumulate a weight delta into `pending_updates` (F56): the
    /// previous `insert(...)` overwrote any existing pending update for
    /// the same `(pre, post)` pair instead of summing contributions from
    /// multiple presynaptic partners within the same batch.
    fn accumulate_pending(state: &mut SystemState<T>, key: (usize, usize), delta: T) {
        let entry = state.pending_updates.entry(key).or_insert_with(T::zero);
        *entry = *entry + delta;
    }

    fn compute_stdp_weight_change(&self, dt: T) -> T {
        if dt > T::zero() {
            // Post-before-pre: LTP
            let exp_arg = -dt / self.stdp_config.tau_pot;
            self.stdp_config.learning_rate_pot * exp_arg.exp()
        } else {
            // Pre-before-post: LTD
            let exp_arg = dt / self.stdp_config.tau_dep;
            -self.stdp_config.learning_rate_dep * exp_arg.exp()
        }
    }
}

/// Weight update event handler
struct WeightUpdateEventHandler<T: Float + Debug + Send + Sync + 'static> {
    stdp_config: STDPConfig<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> EventHandler<T> for WeightUpdateEventHandler<T> {
    fn handle_event(
        &mut self,
        event: &NeuromorphicEvent<T>,
        state: &mut SystemState<T>,
    ) -> Result<()> {
        let source = event.source_neuron;

        if let Some(target) = event.target_neuron {
            if source < state.synaptic_weights.nrows() && target < state.synaptic_weights.ncols() {
                // Apply weight update
                let current_weight = state.synaptic_weights[[source, target]];
                let new_weight = (current_weight + event.value)
                    .max(self.stdp_config.weight_min)
                    .min(self.stdp_config.weight_max);

                state.synaptic_weights[[source, target]] = new_weight;
            }
        }

        Ok(())
    }

    fn can_handle(&self, event_type: EventType) -> bool {
        event_type == EventType::WeightUpdate
    }
}

/// Temporal correlation tracker
struct TemporalCorrelationTracker<T: Float + Debug + Send + Sync + 'static> {
    correlation_window: T,
    event_history: VecDeque<(T, EventType, usize)>,
    correlation_patterns: HashMap<(EventType, EventType), T>,
}

impl<T: Float + Debug + Send + Sync + 'static + std::ops::AddAssign> TemporalCorrelationTracker<T> {
    fn new(correlation_window: T) -> Self {
        Self {
            correlation_window,
            event_history: VecDeque::new(),
            correlation_patterns: HashMap::new(),
        }
    }

    fn add_event(&mut self, time: T, event_type: EventType, neuron_id: usize) {
        // Add new event
        self.event_history.push_back((time, event_type, neuron_id));

        // Remove old events outside correlation window
        while let Some(&(old_time, _, _)) = self.event_history.front() {
            if time - old_time > self.correlation_window {
                self.event_history.pop_front();
            } else {
                break;
            }
        }

        // Update correlation patterns
        self.update_correlations(time, event_type);
    }

    fn update_correlations(&mut self, current_time: T, current_event: EventType) {
        for &(event_time, event_type_, _) in &self.event_history {
            if current_time - event_time <= self.correlation_window {
                let correlation_key = (event_type_, current_event);
                let time_diff = current_time - event_time;
                let correlation_strength = (-time_diff / self.correlation_window).exp();

                *self
                    .correlation_patterns
                    .entry(correlation_key)
                    .or_insert(T::zero()) += correlation_strength;
            }
        }
    }

    fn get_correlation(&self, event1: EventType, event2: EventType) -> T {
        self.correlation_patterns
            .get(&(event1, event2))
            .copied()
            .unwrap_or(T::zero())
    }
}

/// Event rate limiter
struct EventRateLimiter<T: Float + Debug + Send + Sync + 'static> {
    rate_limits: HashMap<EventType, T>,
    event_counts: HashMap<EventType, usize>,
    last_reset: Instant,
    reset_interval: Duration,
}

impl<T: Float + Debug + Send + Sync + 'static> EventRateLimiter<T> {
    fn new(rate_limits: HashMap<EventType, T>) -> Self {
        Self {
            rate_limits,
            event_counts: HashMap::new(),
            last_reset: Instant::now(),
            reset_interval: Duration::from_secs(1),
        }
    }

    fn can_process(&mut self, event_type: EventType) -> bool {
        // Reset counters if interval elapsed
        if self.last_reset.elapsed() >= self.reset_interval {
            self.event_counts.clear();
            self.last_reset = Instant::now();
        }

        if let Some(&limit) = self.rate_limits.get(&event_type) {
            let current_count = self.event_counts.get(&event_type).copied().unwrap_or(0);
            if T::from(current_count).unwrap_or_else(|| T::zero()) < limit {
                *self.event_counts.entry(event_type).or_insert(0) += 1;
                true
            } else {
                false
            }
        } else {
            true
        }
    }
}

/// Event compression engine
struct EventCompressionEngine<T: Float + Debug + Send + Sync + 'static> {
    algorithm: EventCompressionAlgorithm,
    compression_buffer: Vec<u8>,
    decompression_buffer: Vec<u8>,
    /// Last event's (fixed-point-scaled) fields, used as the delta baseline
    /// by [`Self::delta_encode_event`]/[`Self::delta_decode_event`] (F57).
    /// `None` until the first event has been compressed.
    last_event_fields: Option<(i64, i64, Option<i64>, i64, i64)>,
    _phantom: std::marker::PhantomData<T>,
}

/// Convert a float event field to a fixed-point integer at
/// [`EVENT_FIXED_POINT_SCALE`] resolution, saturating instead of panicking
/// on out-of-range or non-finite values.
fn field_to_fixed<T: Float>(value: T) -> i64 {
    let scaled = value.to_f64().unwrap_or(0.0) * EVENT_FIXED_POINT_SCALE;
    if !scaled.is_finite() {
        0
    } else {
        scaled.clamp(i64::MIN as f64, i64::MAX as f64) as i64
    }
}

fn fixed_to_field<T: Float>(fixed: i64) -> T {
    to_generic_or(fixed as f64 / EVENT_FIXED_POINT_SCALE, T::zero())
}

impl<T: Float + Debug + Send + Sync + 'static> EventCompressionEngine<T> {
    fn new(algorithm: EventCompressionAlgorithm) -> Self {
        Self {
            algorithm,
            compression_buffer: Vec::new(),
            decompression_buffer: Vec::new(),
            last_event_fields: None,
            _phantom: std::marker::PhantomData,
        }
    }

    fn compress_event(&mut self, event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        let compressed = match self.algorithm {
            EventCompressionAlgorithm::None => {
                // No compression, serialize directly
                self.serialize_event(event)
            }
            EventCompressionAlgorithm::DeltaEncoding => self.delta_encode_event(event),
            EventCompressionAlgorithm::SparseEncoding => self.sparse_encode_event(event),
            _ => {
                // Fallback to no compression
                self.serialize_event(event)
            }
        }?;
        self.compression_buffer.clear();
        self.compression_buffer.extend_from_slice(&compressed);
        Ok(compressed)
    }

    /// Reconstruct a [`NeuromorphicEvent`] from bytes produced by
    /// [`Self::compress_event`], using the same algorithm and delta-baseline
    /// state. Must be called in the same order events were compressed when
    /// `DeltaEncoding` is in use, since each event's baseline is the
    /// previously *decoded* one.
    fn decompress_event(&mut self, bytes: &[u8]) -> Result<NeuromorphicEvent<T>> {
        self.decompression_buffer.clear();
        self.decompression_buffer.extend_from_slice(bytes);
        match self.algorithm {
            EventCompressionAlgorithm::DeltaEncoding => self.delta_decode_event(bytes),
            EventCompressionAlgorithm::SparseEncoding => self.sparse_decode_event(bytes),
            _ => self.deserialize_event(bytes),
        }
    }

    /// Full-fidelity, self-contained encoding of every event field as
    /// LEB128 varints (unsigned for the always-non-negative fields,
    /// zigzag for `value`/`energy_cost` which may be negative). This is
    /// both `EventCompressionAlgorithm::None`'s wire format and the
    /// decoding target every other algorithm falls back to.
    fn serialize_event(&self, event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        let mut data = Vec::new();
        data.push(event.event_type as u8);
        data.push(event.priority as u8);
        write_uvarint(&mut data, event.source_neuron as u64);
        match event.target_neuron {
            Some(target) => {
                data.push(1);
                write_uvarint(&mut data, target as u64);
            }
            None => data.push(0),
        }
        write_ivarint(&mut data, field_to_fixed(event.timestamp));
        write_ivarint(&mut data, field_to_fixed(event.value));
        write_ivarint(&mut data, field_to_fixed(event.energy_cost));
        Ok(data)
    }

    fn deserialize_event(&self, bytes: &[u8]) -> Result<NeuromorphicEvent<T>> {
        let mut pos = 0usize;
        let event_type = decode_event_type(*bytes.first().ok_or_else(truncated_bytes_err)?)?;
        pos += 1;
        let priority = decode_priority(*bytes.get(pos).ok_or_else(truncated_bytes_err)?)?;
        pos += 1;
        let source_neuron = read_uvarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)? as usize;
        let has_target = *bytes.get(pos).ok_or_else(truncated_bytes_err)?;
        pos += 1;
        let target_neuron = if has_target == 1 {
            Some(read_uvarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)? as usize)
        } else {
            None
        };
        let timestamp =
            fixed_to_field(read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?);
        let value = fixed_to_field(read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?);
        let energy_cost =
            fixed_to_field(read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?);
        Ok(NeuromorphicEvent {
            event_type,
            timestamp,
            source_neuron,
            target_neuron,
            value,
            energy_cost,
            priority,
        })
    }

    /// Delta encoding (F57): every numeric field is written as its
    /// varint-encoded difference from the previous compressed event's
    /// corresponding field, rather than its absolute value. For a stream of
    /// similar consecutive events (the common neuromorphic case — spikes
    /// from nearby neurons close together in time) small deltas take far
    /// fewer varint bytes than the absolute fixed-point values. The first
    /// event in a stream has no baseline and is encoded as full deltas from
    /// zero, which is exactly `serialize_event`'s absolute encoding.
    fn delta_encode_event(&mut self, event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        let ts = field_to_fixed(event.timestamp);
        let src = event.source_neuron as i64;
        let tgt = event.target_neuron.map(|t| t as i64);
        let val = field_to_fixed(event.value);
        let energy = field_to_fixed(event.energy_cost);

        let (base_ts, base_src, base_tgt, base_val, base_energy) =
            self.last_event_fields.unwrap_or((0, 0, None, 0, 0));

        let mut data = Vec::new();
        data.push(event.event_type as u8);
        data.push(event.priority as u8);
        write_ivarint(&mut data, src - base_src);
        match (tgt, base_tgt) {
            (Some(t), Some(b)) => {
                data.push(1);
                write_ivarint(&mut data, t - b);
            }
            (Some(t), None) => {
                data.push(2); // "present, no prior baseline": encode absolute
                write_ivarint(&mut data, t);
            }
            (None, _) => data.push(0),
        }
        write_ivarint(&mut data, ts - base_ts);
        write_ivarint(&mut data, val - base_val);
        write_ivarint(&mut data, energy - base_energy);

        self.last_event_fields = Some((ts, src, tgt, val, energy));
        Ok(data)
    }

    fn delta_decode_event(&mut self, bytes: &[u8]) -> Result<NeuromorphicEvent<T>> {
        let mut pos = 0usize;
        let event_type = decode_event_type(*bytes.first().ok_or_else(truncated_bytes_err)?)?;
        pos += 1;
        let priority = decode_priority(*bytes.get(pos).ok_or_else(truncated_bytes_err)?)?;
        pos += 1;

        let (base_ts, base_src, base_tgt, base_val, base_energy) =
            self.last_event_fields.unwrap_or((0, 0, None, 0, 0));

        let src = base_src + read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?;
        let target_tag = *bytes.get(pos).ok_or_else(truncated_bytes_err)?;
        pos += 1;
        let tgt = match target_tag {
            0 => None,
            1 => {
                let base = base_tgt.ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "delta-encoded target references a missing baseline".to_string(),
                    )
                })?;
                Some(base + read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?)
            }
            2 => Some(read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?),
            other => {
                return Err(OptimError::InvalidConfig(format!(
                    "invalid delta-encoded target tag: {other}"
                )))
            }
        };
        let ts = base_ts + read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?;
        let val = base_val + read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?;
        let energy = base_energy + read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?;

        self.last_event_fields = Some((ts, src, tgt, val, energy));

        if src < 0 {
            return Err(OptimError::InvalidConfig(
                "delta-decoded source_neuron underflowed".to_string(),
            ));
        }
        Ok(NeuromorphicEvent {
            event_type,
            timestamp: fixed_to_field(ts),
            source_neuron: src as usize,
            target_neuron: match tgt {
                Some(t) if t >= 0 => Some(t as usize),
                Some(_) => {
                    return Err(OptimError::InvalidConfig(
                        "delta-decoded target_neuron underflowed".to_string(),
                    ))
                }
                None => None,
            },
            value: fixed_to_field(val),
            energy_cost: fixed_to_field(energy),
            priority,
        })
    }

    /// Sparse encoding (F57): most events carry a zero/default `value` and
    /// `energy_cost`, and most events have no `target_neuron` (broadcast
    /// events). A leading bitmask flags which optional fields are present,
    /// so all-zero/absent fields cost a single bit each instead of a full
    /// varint.
    fn sparse_encode_event(&mut self, event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        let has_target = event.target_neuron.is_some();
        let has_value = event.value != T::zero();
        let has_energy = event.energy_cost != T::zero();

        let mut mask = 0u8;
        if has_target {
            mask |= 0b001;
        }
        if has_value {
            mask |= 0b010;
        }
        if has_energy {
            mask |= 0b100;
        }

        let mut data = Vec::new();
        data.push(event.event_type as u8);
        data.push(event.priority as u8);
        data.push(mask);
        write_uvarint(&mut data, event.source_neuron as u64);
        write_ivarint(&mut data, field_to_fixed(event.timestamp));
        if let Some(target) = event.target_neuron {
            write_uvarint(&mut data, target as u64);
        }
        if has_value {
            write_ivarint(&mut data, field_to_fixed(event.value));
        }
        if has_energy {
            write_ivarint(&mut data, field_to_fixed(event.energy_cost));
        }
        Ok(data)
    }

    fn sparse_decode_event(&mut self, bytes: &[u8]) -> Result<NeuromorphicEvent<T>> {
        let mut pos = 0usize;
        let event_type = decode_event_type(*bytes.first().ok_or_else(truncated_bytes_err)?)?;
        pos += 1;
        let priority = decode_priority(*bytes.get(pos).ok_or_else(truncated_bytes_err)?)?;
        pos += 1;
        let mask = *bytes.get(pos).ok_or_else(truncated_bytes_err)?;
        pos += 1;
        let source_neuron = read_uvarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)? as usize;
        let timestamp =
            fixed_to_field(read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?);
        let target_neuron = if mask & 0b001 != 0 {
            Some(read_uvarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)? as usize)
        } else {
            None
        };
        let value = if mask & 0b010 != 0 {
            fixed_to_field(read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?)
        } else {
            T::zero()
        };
        let energy_cost = if mask & 0b100 != 0 {
            fixed_to_field(read_ivarint(bytes, &mut pos).ok_or_else(truncated_bytes_err)?)
        } else {
            T::zero()
        };
        Ok(NeuromorphicEvent {
            event_type,
            timestamp,
            source_neuron,
            target_neuron,
            value,
            energy_cost,
            priority,
        })
    }
}

/// A FIFO chain of *compressed* event frames for a single priority level.
///
/// F57 (wiring): compressed storage has to be decoded in the same order it
/// was encoded, because the delta/predictive codecs carry a running baseline
/// (`EventCompressionEngine::last_event_fields`) from one frame to the next.
/// The event scheduler, on the other hand, must still honour
/// [`EventPriority`]. Keeping one independent chain per priority level
/// satisfies both constraints at once: inside a chain the order is strictly
/// FIFO (so the delta baseline chain stays intact, matching the FIFO
/// tie-break the uncompressed [`BinaryHeap`] path uses), and the scheduler
/// just picks the highest-priority non-empty chain.
///
/// Two engines are held because the encoder necessarily runs ahead of the
/// decoder by `frames.len()` events, so they cannot share one baseline.
struct CompressedEventChain<T: Float + Debug + Send + Sync + 'static> {
    /// Encoder state; advanced by [`Self::push`].
    encoder: EventCompressionEngine<T>,
    /// Decoder state; advanced by [`Self::pop`], trailing the encoder.
    decoder: EventCompressionEngine<T>,
    /// Compressed frames, oldest first.
    frames: VecDeque<Vec<u8>>,
    /// Bytes currently held by `frames`.
    stored_bytes: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> CompressedEventChain<T> {
    fn new(algorithm: EventCompressionAlgorithm) -> Self {
        Self {
            encoder: EventCompressionEngine::new(algorithm),
            decoder: EventCompressionEngine::new(algorithm),
            frames: VecDeque::new(),
            stored_bytes: 0,
        }
    }

    /// Compress `event` and append the resulting frame. Returns the number
    /// of bytes the frame occupies.
    fn push(&mut self, event: &NeuromorphicEvent<T>) -> Result<usize> {
        let frame = self.encoder.compress_event(event)?;
        let frame_len = frame.len();
        self.stored_bytes += frame_len;
        self.frames.push_back(frame);
        Ok(frame_len)
    }

    /// Decode and remove the oldest frame, reconstructing the event.
    fn pop(&mut self) -> Result<Option<NeuromorphicEvent<T>>> {
        match self.frames.pop_front() {
            Some(frame) => {
                self.stored_bytes = self.stored_bytes.saturating_sub(frame.len());
                Ok(Some(self.decoder.decompress_event(&frame)?))
            }
            None => Ok(None),
        }
    }

    fn len(&self) -> usize {
        self.frames.len()
    }

    fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Drop every pending frame. The codec baselines are reset as well:
    /// discarding frames breaks the delta chain, so the encoder and decoder
    /// must both restart from "no baseline" to stay consistent.
    fn clear(&mut self) {
        self.frames.clear();
        self.stored_bytes = 0;
        self.encoder.last_event_fields = None;
        self.decoder.last_event_fields = None;
    }
}

/// Adaptive event handler
struct AdaptiveEventHandler<T: Float + Debug + Send + Sync + 'static> {
    adaptation_rate: T,
    performance_history: VecDeque<T>,
    current_strategy: AdaptationStrategy,
}

#[derive(Debug, Clone, Copy)]
enum AdaptationStrategy {
    Conservative,
    Balanced,
    Aggressive,
}

impl<T: Float + Debug + Send + Sync + 'static + std::iter::Sum> AdaptiveEventHandler<T> {
    fn new() -> Self {
        Self {
            adaptation_rate: T::from(0.1).unwrap_or_else(|| T::zero()),
            performance_history: VecDeque::new(),
            current_strategy: AdaptationStrategy::Balanced,
        }
    }

    fn adapt_processing(&mut self, current_performance: T) {
        self.performance_history.push_back(current_performance);

        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }

        if self.performance_history.len() >= 10 {
            let recent_avg = self
                .performance_history
                .iter()
                .rev()
                .take(10)
                .cloned()
                .sum::<T>()
                / T::from(10).unwrap_or_else(|| T::zero());
            let older_avg = if self.performance_history.len() >= 20 {
                self.performance_history
                    .iter()
                    .rev()
                    .skip(10)
                    .take(10)
                    .cloned()
                    .sum::<T>()
                    / T::from(10).unwrap_or_else(|| T::zero())
            } else {
                recent_avg
            };

            let performance_change = recent_avg - older_avg;

            self.current_strategy =
                if performance_change > T::from(0.1).unwrap_or_else(|| T::zero()) {
                    AdaptationStrategy::Aggressive
                } else if performance_change < T::from(-0.1).unwrap_or_else(|| T::zero()) {
                    AdaptationStrategy::Conservative
                } else {
                    AdaptationStrategy::Balanced
                };
        }
    }

    fn get_adaptation_factor(&self) -> T {
        match self.current_strategy {
            AdaptationStrategy::Conservative => T::from(0.5).unwrap_or_else(|| T::zero()),
            AdaptationStrategy::Balanced => T::one(),
            AdaptationStrategy::Aggressive => T::from(1.5).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Distributed event coordinator
struct DistributedEventCoordinator<T: Float + Debug + Send + Sync + 'static> {
    load_balancing: LoadBalancingStrategy,
    worker_loads: HashMap<usize, T>,
    current_worker: usize,
    total_workers: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> DistributedEventCoordinator<T> {
    fn new(strategy: LoadBalancingStrategy, num_workers: usize) -> Self {
        Self {
            load_balancing: strategy,
            worker_loads: HashMap::new(),
            current_worker: 0,
            total_workers: num_workers,
        }
    }

    fn assign_worker(&mut self, event: &NeuromorphicEvent<T>) -> usize {
        match self.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                let worker = self.current_worker;
                self.current_worker = (self.current_worker + 1) % self.total_workers;
                worker
            }
            LoadBalancingStrategy::TypeBased => {
                // Hash event type to worker
                (event.event_type as usize) % self.total_workers
            }
            LoadBalancingStrategy::LoadAware => {
                // Find worker with minimum load
                self.worker_loads
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(&worker_id, _)| worker_id)
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    fn update_worker_load(&mut self, worker_id: usize, load: T) {
        self.worker_loads.insert(worker_id, load);
    }
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand
            + std::ops::AddAssign,
    > EventDrivenOptimizer<T>
{
    /// Create a new event-driven optimizer
    pub fn new(
        config: EventDrivenConfig<T>,
        stdp_config: STDPConfig<T>,
        membrane_config: MembraneDynamicsConfig<T>,
        num_neurons: usize,
    ) -> Self {
        let mut optimizer = Self {
            config: config.clone(),
            stdp_config: stdp_config.clone(),
            membrane_config: membrane_config.clone(),
            event_queue: BinaryHeap::new(),
            event_stats: HashMap::new(),
            system_state: SystemState {
                membrane_potentials: Array1::from_elem(
                    num_neurons,
                    membrane_config.resting_potential,
                ),
                synaptic_weights: Array2::ones((num_neurons, num_neurons))
                    * T::from(0.1).unwrap_or_else(|| T::zero()),
                last_spike_times: Array1::from_elem(
                    num_neurons,
                    T::from(-1000.0).unwrap_or_else(|| T::zero()),
                ),
                refractory_until: Array1::zeros(num_neurons),
                current_time: T::zero(),
                active_neurons: HashSet::new(),
                pending_updates: HashMap::new(),
            },
            event_handlers: HashMap::new(),
            correlation_tracker: TemporalCorrelationTracker::new(config.correlation_window),
            rate_limiter: EventRateLimiter::new(config.rate_limits.clone()),
            metrics: NeuromorphicMetrics::default(),
            distributed_coordinator: if config.distributed_processing {
                Some(DistributedEventCoordinator::new(config.load_balancing, 4))
            } else {
                None
            },
            compression_engine: EventCompressionEngine::new(config.compression_algorithm),
            compressed_chains: BTreeMap::new(),
            compression_raw_bytes: 0,
            compression_compressed_bytes: 0,
            adaptive_handler: AdaptiveEventHandler::new(),
        };

        // Register default event handlers
        optimizer.register_default_handlers();

        optimizer
    }

    /// Register default event handlers
    fn register_default_handlers(&mut self) {
        let spike_handler = Box::new(SpikeEventHandler {
            stdp_config: self.stdp_config.clone(),
            membrane_config: self.membrane_config.clone(),
        });

        let weight_handler = Box::new(WeightUpdateEventHandler {
            stdp_config: self.stdp_config.clone(),
        });

        self.event_handlers.insert(EventType::Spike, spike_handler);
        self.event_handlers
            .insert(EventType::WeightUpdate, weight_handler);
    }

    /// Add event to the processing queue.
    ///
    /// With `config.event_compression` enabled the event is compressed to
    /// bytes *here* and only those bytes are retained (F57): the live queue
    /// really does store compressed frames instead of whole events, and the
    /// event is reconstructed lazily in [`Self::pop_next_event`] when it is
    /// about to be processed.
    pub fn enqueue_event(&mut self, event: NeuromorphicEvent<T>) -> Result<()> {
        // Check rate limits
        if !self.rate_limiter.can_process(event.event_type) {
            return Err(OptimError::InvalidConfig("Rate limit exceeded".to_string()));
        }

        // Check queue capacity (both stores count towards the budget, so the
        // capacity check keeps working when compression is enabled).
        if self.get_queue_size() >= self.config.max_queue_size {
            return Err(OptimError::InvalidConfig("Event queue full".to_string()));
        }

        // Extract event fields before moving
        let timestamp = event.timestamp;
        let event_type = event.event_type;
        let source_neuron = event.source_neuron;

        if self.config.event_compression {
            let raw_bytes = self.compression_engine.serialize_event(&event)?.len();
            let algorithm = self.config.compression_algorithm;
            let chain = self
                .compressed_chains
                .entry(event.priority)
                .or_insert_with(|| CompressedEventChain::new(algorithm));
            let compressed_bytes = chain.push(&event)?;
            self.compression_raw_bytes += raw_bytes;
            self.compression_compressed_bytes += compressed_bytes;
        } else {
            self.event_queue.push(PriorityEventEntry {
                event,
                insertion_time: Instant::now(),
            });
        }

        // Update correlation tracking
        if self.config.temporal_correlation {
            self.correlation_tracker
                .add_event(timestamp, event_type, source_neuron);
        }

        Ok(())
    }

    /// Take the next event to process, highest priority first.
    ///
    /// Compressed frames are drained before the in-memory heap. The two
    /// stores only ever hold events simultaneously if
    /// `config.event_compression` was toggled while events were queued; in
    /// that case the compressed frames (which were necessarily enqueued
    /// while compression was on) are the older ones, so draining them first
    /// preserves arrival order across the switch.
    fn pop_next_event(&mut self) -> Result<Option<NeuromorphicEvent<T>>> {
        // `BTreeMap` iterates in ascending key order and `EventPriority`
        // derives `Ord` lowest-first, so scan in reverse for the
        // highest-priority non-empty chain.
        let highest = self
            .compressed_chains
            .iter()
            .rev()
            .find(|(_, chain)| !chain.is_empty())
            .map(|(&priority, _)| priority);
        if let Some(priority) = highest {
            if let Some(chain) = self.compressed_chains.get_mut(&priority) {
                return chain.pop();
            }
        }
        Ok(self.event_queue.pop().map(|entry| entry.event))
    }

    /// Whether any event (compressed or not) is still waiting.
    fn has_pending_events(&self) -> bool {
        !self.event_queue.is_empty()
            || self
                .compressed_chains
                .values()
                .any(|chain| !chain.is_empty())
    }

    /// Process events from the queue
    pub fn process_events(&mut self) -> Result<usize> {
        let mut processed_count = 0;
        let start_time = Instant::now();
        let timeout =
            Duration::from_millis(self.config.processing_timeout.to_u64().unwrap_or(1000));

        while self.has_pending_events() && start_time.elapsed() < timeout {
            if self.config.event_batching {
                let batch_size = self.adaptive_batch_size().min(self.get_queue_size());
                let batch_processed = self.process_event_batch(batch_size)?;
                if batch_processed == 0 {
                    break;
                }
                processed_count += batch_processed;
            } else if let Some(event) = self.pop_next_event()? {
                self.process_single_event(&event)?;
                processed_count += 1;
            } else {
                break;
            }
        }

        // Apply pending weight updates
        self.apply_pending_updates()?;

        // Update adaptive processing. An interval the monotonic clock reports
        // as exactly zero carries no rate information at all, so no sample is
        // recorded for it: the previous code divided by that zero (and
        // `expect`ed the conversion), poisoning `performance_history` with
        // `inf` whenever a batch completed inside one clock tick.
        let elapsed_secs = start_time.elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            let processing_rate = to_generic_or(processed_count as f64 / elapsed_secs, T::zero());
            self.adaptive_handler.adapt_processing(processing_rate);
        }

        Ok(processed_count)
    }

    /// Process a batch of events
    fn process_event_batch(&mut self, batch_size: usize) -> Result<usize> {
        let mut batch_events = Vec::with_capacity(batch_size);

        // Collect batch events
        for _ in 0..batch_size {
            match self.pop_next_event()? {
                Some(event) => batch_events.push(event),
                None => break,
            }
        }

        // Process batch
        for event in &batch_events {
            self.process_single_event(event)?;
        }

        Ok(batch_events.len())
    }

    /// Process a single event
    fn process_single_event(&mut self, event: &NeuromorphicEvent<T>) -> Result<()> {
        let start_time = Instant::now();

        // Find appropriate handler
        if let Some(handler) = self.event_handlers.get_mut(&event.event_type) {
            handler.handle_event(event, &mut self.system_state)?;
        } else {
            // Default handling
            self.default_event_handling(event)?;
        }

        // Update statistics
        let processing_time = start_time.elapsed().as_nanos() as f64 / 1_000_000.0;
        self.update_event_statistics(
            event.event_type,
            T::from(processing_time).unwrap_or_else(|| T::zero()),
        );

        // Update energy consumption
        self.metrics.energy_consumption += event.energy_cost;

        Ok(())
    }

    /// Default event handling
    fn default_event_handling(&mut self, event: &NeuromorphicEvent<T>) -> Result<()> {
        match event.event_type {
            EventType::ExternalStimulus
                // Apply external stimulus to neuron
                if event.source_neuron < self.system_state.membrane_potentials.len() => {
                    self.system_state.membrane_potentials[event.source_neuron] += event.value;
                }
            EventType::TimerEvent => {
                // Update system time
                self.system_state.current_time = event.timestamp;
            }
            _ => {
                // Ignore unknown events
            }
        }

        Ok(())
    }

    /// Apply pending weight updates
    fn apply_pending_updates(&mut self) -> Result<()> {
        for ((pre, post), weight_change) in self.system_state.pending_updates.drain() {
            if pre < self.system_state.synaptic_weights.nrows()
                && post < self.system_state.synaptic_weights.ncols()
            {
                let current_weight = self.system_state.synaptic_weights[[pre, post]];
                let new_weight = (current_weight + weight_change)
                    .max(self.stdp_config.weight_min)
                    .min(self.stdp_config.weight_max);

                self.system_state.synaptic_weights[[pre, post]] = new_weight;
            }
        }

        Ok(())
    }

    /// Update event processing statistics
    fn update_event_statistics(&mut self, event_type: EventType, processing_time: T) {
        let stats = self
            .event_stats
            .entry(event_type)
            .or_insert_with(|| EventStatistics {
                total_processed: 0,
                avg_processing_time: T::zero(),
                event_rate: T::zero(),
                avg_queue_wait_time: T::zero(),
                error_count: 0,
                last_update: Instant::now(),
            });

        stats.total_processed += 1;

        // Update average processing _time using exponential moving average
        let alpha = T::from(0.1).unwrap_or_else(|| T::zero());
        stats.avg_processing_time =
            stats.avg_processing_time * (T::one() - alpha) + processing_time * alpha;

        // Update event rate
        let time_since_last = stats.last_update.elapsed().as_secs_f64();
        if time_since_last > 0.0 {
            let current_rate = T::one() / T::from(time_since_last).unwrap_or_else(|| T::zero());
            stats.event_rate = stats.event_rate * (T::one() - alpha) + current_rate * alpha;
        }

        stats.last_update = Instant::now();
    }

    /// Get event processing statistics
    pub fn get_event_statistics(&self) -> &HashMap<EventType, EventStatistics<T>> {
        &self.event_stats
    }

    /// Get current system state
    pub fn get_system_state(&self) -> &SystemState<T> {
        &self.system_state
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &NeuromorphicMetrics<T> {
        &self.metrics
    }

    /// Clear event queue, including any compressed frames still pending.
    pub fn clear_event_queue(&mut self) {
        self.event_queue.clear();
        for chain in self.compressed_chains.values_mut() {
            chain.clear();
        }
    }

    /// Get queue size: events waiting in the in-memory priority queue plus
    /// events waiting as compressed frames.
    pub fn get_queue_size(&self) -> usize {
        self.event_queue.len()
            + self
                .compressed_chains
                .values()
                .map(|chain| chain.len())
                .sum::<usize>()
    }

    /// Bytes currently occupied by the compressed event queue (F57). Zero
    /// while `config.event_compression` is disabled, since nothing is stored
    /// in compressed form then.
    pub fn compressed_queue_bytes(&self) -> usize {
        self.compressed_chains
            .values()
            .map(|chain| chain.stored_bytes)
            .sum()
    }

    /// Cumulative `(uncompressed_bytes, compressed_bytes)` measured across
    /// every event that has entered the compressed queue since construction.
    /// The uncompressed figure is the size the same event occupies under the
    /// reference (`EventCompressionAlgorithm::None`) codec, so the pair is a
    /// direct measurement of what compression actually achieved.
    pub fn compression_statistics(&self) -> (usize, usize) {
        (
            self.compression_raw_bytes,
            self.compression_compressed_bytes,
        )
    }

    /// Achieved compression ratio (`compressed / uncompressed`); `None` until
    /// at least one event has been compressed.
    pub fn compression_ratio(&self) -> Option<f64> {
        if self.compression_raw_bytes == 0 {
            None
        } else {
            Some(self.compression_compressed_bytes as f64 / self.compression_raw_bytes as f64)
        }
    }

    /// The most recent *measured* event-processing rate (events per second)
    /// recorded by [`Self::process_events`], or `None` if no interval long
    /// enough to be measurable has been observed yet. Always finite: the
    /// previous implementation divided the processed count by an integer
    /// millisecond count, which is zero for any batch finishing inside one
    /// clock tick, feeding `inf`/`NaN` into the adaptive handler.
    pub fn last_measured_event_rate(&self) -> Option<T> {
        self.adaptive_handler.performance_history.back().copied()
    }

    /// The adaptation factor currently chosen by the adaptive event handler
    /// from the measured event-processing rate history (0.5 while throughput
    /// is degrading, 1.0 while it is stable, 1.5 while it is improving).
    pub fn current_adaptation_factor(&self) -> T {
        self.adaptive_handler.get_adaptation_factor()
    }

    /// Effective per-iteration batch size: the configured `batch_size` scaled
    /// by [`Self::current_adaptation_factor`], so a system whose measured
    /// throughput is improving takes larger bites and one that is degrading
    /// takes smaller ones. Never zero.
    pub fn adaptive_batch_size(&self) -> usize {
        let factor = self.current_adaptation_factor().to_f64().unwrap_or(1.0);
        let scaled = (self.config.batch_size as f64 * factor).round();
        if scaled.is_finite() && scaled >= 1.0 {
            scaled as usize
        } else {
            self.config.batch_size.max(1)
        }
    }

    /// Enable distributed processing
    pub fn enable_distributed_processing(&mut self, num_workers: usize) {
        self.distributed_coordinator = Some(DistributedEventCoordinator::new(
            self.config.load_balancing,
            num_workers,
        ));
        self.config.distributed_processing = true;
    }

    /// Disable distributed processing
    pub fn disable_distributed_processing(&mut self) {
        self.distributed_coordinator = None;
        self.config.distributed_processing = false;
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for EventStatistics<T> {
    fn default() -> Self {
        Self {
            total_processed: 0,
            avg_processing_time: T::zero(),
            event_rate: T::zero(),
            avg_queue_wait_time: T::zero(),
            error_count: 0,
            last_update: Instant::now(),
        }
    }
}

/// Regression tests for the F57 *wiring* (compressed live queue), kept in
/// their own file so `event_driven.rs` stays under the 2000-line policy.
#[cfg(test)]
#[path = "event_driven_compression_tests.rs"]
mod compression_wiring_tests;

#[cfg(test)]
mod f57_compression_tests {
    use super::*;

    fn sample_events() -> Vec<NeuromorphicEvent<f64>> {
        vec![
            NeuromorphicEvent {
                event_type: EventType::Spike,
                timestamp: 12.5,
                source_neuron: 3,
                target_neuron: Some(7),
                value: 1.0,
                energy_cost: 0.05,
                priority: EventPriority::High,
            },
            NeuromorphicEvent {
                event_type: EventType::WeightUpdate,
                timestamp: 12.6,
                source_neuron: 4,
                target_neuron: Some(8),
                value: -0.25,
                energy_cost: 0.02,
                priority: EventPriority::Normal,
            },
            NeuromorphicEvent {
                event_type: EventType::TimerEvent,
                timestamp: 100.0,
                source_neuron: 0,
                target_neuron: None,
                value: 0.0,
                energy_cost: 0.0,
                priority: EventPriority::Low,
            },
            NeuromorphicEvent {
                event_type: EventType::EnergyEvent,
                timestamp: 0.0,
                source_neuron: 42,
                target_neuron: Some(1),
                value: 999.75,
                energy_cost: -3.5,
                priority: EventPriority::RealTime,
            },
        ]
    }

    fn assert_events_close(a: &NeuromorphicEvent<f64>, b: &NeuromorphicEvent<f64>) {
        assert_eq!(a.event_type, b.event_type);
        assert_eq!(a.priority, b.priority);
        assert_eq!(a.source_neuron, b.source_neuron);
        assert_eq!(a.target_neuron, b.target_neuron);
        assert!(
            (a.timestamp - b.timestamp).abs() < 1e-3,
            "timestamp mismatch: {} vs {}",
            a.timestamp,
            b.timestamp
        );
        assert!(
            (a.value - b.value).abs() < 1e-3,
            "value mismatch: {} vs {}",
            a.value,
            b.value
        );
        assert!(
            (a.energy_cost - b.energy_cost).abs() < 1e-3,
            "energy_cost mismatch: {} vs {}",
            a.energy_cost,
            b.energy_cost
        );
    }

    /// F57: `EventCompressionAlgorithm::None` must round-trip every field,
    /// and must not be the fixed-size all-zero buffer the previous
    /// delta/sparse stubs returned regardless of algorithm choice.
    #[test]
    fn none_algorithm_round_trips_all_fields() {
        let mut engine = EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::None);
        for event in sample_events() {
            let bytes = engine.compress_event(&event).expect("compress");
            let decoded = engine.decompress_event(&bytes).expect("decompress");
            assert_events_close(&event, &decoded);
        }
    }

    /// F57: delta encoding must round-trip a whole stream of events in
    /// order (each event's baseline is the previously *decoded* event),
    /// including the target-neuron presence/absence transitions.
    #[test]
    fn delta_encoding_round_trips_event_stream() {
        let mut encoder =
            EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::DeltaEncoding);
        let mut decoder =
            EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::DeltaEncoding);

        for event in sample_events() {
            let bytes = encoder.compress_event(&event).expect("compress");
            let decoded = decoder.decompress_event(&bytes).expect("decompress");
            assert_events_close(&event, &decoded);
        }
    }

    /// F57: delta-encoded bytes for a stream of *similar* consecutive
    /// events (the case delta encoding exists to exploit) must be smaller
    /// than the same events' absolute (`None`) encoding — otherwise the
    /// "compression" is not actually compressing anything.
    #[test]
    fn delta_encoding_is_smaller_for_similar_events() {
        let mut none_engine = EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::None);
        let mut delta_engine =
            EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::DeltaEncoding);

        let mut none_total = 0usize;
        let mut delta_total = 0usize;
        for i in 0..20 {
            let event = NeuromorphicEvent {
                event_type: EventType::Spike,
                timestamp: 1000.0 + i as f64 * 0.1,
                source_neuron: 50 + (i % 3),
                target_neuron: Some(100 + (i % 2)),
                value: 0.5,
                energy_cost: 0.01,
                priority: EventPriority::Normal,
            };
            none_total += none_engine.compress_event(&event).expect("compress").len();
            delta_total += delta_engine.compress_event(&event).expect("compress").len();
        }
        assert!(
            delta_total < none_total,
            "delta encoding was not smaller for a similar-event stream: \
             delta={delta_total} bytes, none={none_total} bytes"
        );
    }

    /// F57: sparse encoding must round-trip events with all-default
    /// optional fields (no target, zero value/energy) as well as events
    /// that set every optional field.
    #[test]
    fn sparse_encoding_round_trips_default_and_full_events() {
        let mut engine =
            EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::SparseEncoding);
        for event in sample_events() {
            let bytes = engine.compress_event(&event).expect("compress");
            let decoded = engine.decompress_event(&bytes).expect("decompress");
            assert_events_close(&event, &decoded);
        }
    }

    /// F57: decoding truncated/corrupt bytes must return an honest `Err`,
    /// never panic (the varint readers use `?`/`ok_or_else` throughout,
    /// never indexing or unwrapping directly).
    #[test]
    fn decoding_truncated_bytes_is_an_error_not_a_panic() {
        let mut engine = EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::None);
        let event = sample_events().remove(0);
        let bytes = engine.compress_event(&event).expect("compress");

        for len in 0..bytes.len() {
            let mut fresh_engine =
                EventCompressionEngine::<f64>::new(EventCompressionAlgorithm::None);
            let result = fresh_engine.decompress_event(&bytes[..len]);
            assert!(
                result.is_err(),
                "truncating to {len} bytes should be a decode error, got {result:?}"
            );
        }
    }
}
