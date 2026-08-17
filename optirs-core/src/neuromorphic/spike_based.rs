// Spike-Based Optimization Algorithms
//
// This module implements optimization algorithms that operate on spike trains
// and temporal spike patterns, designed for neuromorphic computing platforms.

use super::{
    to_generic_or, EventPriority, MembraneDynamicsConfig, NeuromorphicEvent, NeuromorphicMetrics,
    PlasticityModel, STDPConfig, Spike, SpikeTrain,
};

// SciRS2 Integration - CRITICAL for neuromorphic computing
use scirs2_neural::activations_minimal::Activation;
use scirs2_neural::layers::Layer;
use scirs2_stats::distributions;

use crate::error::Result;
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::random::{thread_rng, Rng};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

/// Spike-based optimization configuration
#[derive(Debug, Clone)]
pub struct SpikingConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Simulation time step (ms)
    pub time_step: T,

    /// Total simulation time (ms)
    pub simulation_time: T,

    /// Encoding method for input data
    pub encoding_method: SpikeEncodingMethod,

    /// Decoding method for output spikes
    pub decoding_method: SpikeDecodingMethod,

    /// Spike train learning rate
    pub spike_learning_rate: T,

    /// Temporal window for spike correlation (ms)
    pub temporal_window: T,

    /// Enable lateral inhibition
    pub lateral_inhibition: bool,

    /// Homeostatic scaling parameters
    pub homeostatic_config: HomeostaticConfig<T>,

    /// Noise parameters for spike generation
    pub noise_config: SpikeNoiseConfig<T>,
}

/// Spike encoding methods for converting continuous values to spike trains
#[derive(Debug, Clone, Copy)]
pub enum SpikeEncodingMethod {
    /// Rate coding (firing rate proportional to value)
    RateCoding,

    /// Temporal coding (spike time proportional to value)
    TemporalCoding,

    /// Population vector coding
    PopulationVectorCoding,

    /// Sparse coding
    SparseCoding,

    /// Phase coding
    PhaseCoding,

    /// Burst coding
    BurstCoding,

    /// Rank order coding
    RankOrderCoding,
}

/// Spike decoding methods for converting spike trains to continuous values
#[derive(Debug, Clone, Copy)]
pub enum SpikeDecodingMethod {
    /// Rate decoding (spike count in time window)
    RateDecoding,

    /// Temporal decoding (first spike time)
    TemporalDecoding,

    /// Population vector decoding
    PopulationVectorDecoding,

    /// Weighted spike count
    WeightedSpikeCount,

    /// Moving average filter
    MovingAverageFilter,

    /// Exponential decay filter
    ExponentialDecayFilter,
}

/// Homeostatic plasticity configuration
#[derive(Debug, Clone)]
pub struct HomeostaticConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable homeostatic scaling
    pub enable_homeostatic_scaling: bool,

    /// Target firing rate (Hz)
    pub target_firing_rate: T,

    /// Scaling time constant (ms)
    pub scaling_time_constant: T,

    /// Scaling factor
    pub scaling_factor: T,

    /// Enable intrinsic plasticity
    pub enable_intrinsic_plasticity: bool,

    /// Threshold adaptation rate
    pub threshold_adaptation_rate: T,
}

/// Spike noise configuration
#[derive(Debug, Clone)]
pub struct SpikeNoiseConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Background firing rate (Hz)
    pub background_rate: T,

    /// Jitter standard deviation (ms)
    pub jitter_std: T,

    /// Enable Poisson noise
    pub poisson_noise: bool,

    /// Noise amplitude
    pub noise_amplitude: T,

    /// Correlation noise
    pub correlation_noise: T,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for SpikingConfig<T> {
    fn default() -> Self {
        Self {
            time_step: T::from(0.1).unwrap_or_else(|| T::zero()),
            simulation_time: T::from(1000.0).unwrap_or_else(|| T::zero()),
            encoding_method: SpikeEncodingMethod::RateCoding,
            decoding_method: SpikeDecodingMethod::RateDecoding,
            spike_learning_rate: T::from(0.01).unwrap_or_else(|| T::zero()),
            temporal_window: T::from(20.0).unwrap_or_else(|| T::zero()),
            lateral_inhibition: false,
            homeostatic_config: HomeostaticConfig::default(),
            noise_config: SpikeNoiseConfig::default(),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for HomeostaticConfig<T> {
    fn default() -> Self {
        Self {
            enable_homeostatic_scaling: false,
            target_firing_rate: T::from(10.0).unwrap_or_else(|| T::zero()),
            scaling_time_constant: T::from(1000.0).unwrap_or_else(|| T::zero()),
            scaling_factor: T::from(0.01).unwrap_or_else(|| T::zero()),
            enable_intrinsic_plasticity: false,
            threshold_adaptation_rate: T::from(0.001).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Shared rate-coding parameters (F54): `rate_encode` and `rate_decode`
/// must agree on both the encoding window and the max firing rate, or
/// decoding introduces a systematic gain error. Both now derive their
/// window from [`SpikingOptimizer::rate_coding_window`] and their max
/// rate from this single constant.
const RATE_CODING_MAX_RATE_HZ: f64 = 100.0;

impl<T: Float + Debug + Send + Sync + 'static> Default for SpikeNoiseConfig<T> {
    fn default() -> Self {
        Self {
            background_rate: T::from(1.0).unwrap_or_else(|| T::zero()),
            jitter_std: T::from(0.5).unwrap_or_else(|| T::zero()),
            poisson_noise: false,
            noise_amplitude: T::from(0.1).unwrap_or_else(|| T::zero()),
            correlation_noise: T::zero(),
        }
    }
}

/// Spike-based optimizer
pub struct SpikingOptimizer<
    T: Float + Debug + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
> {
    /// Configuration
    config: SpikingConfig<T>,

    /// STDP configuration
    stdp_config: STDPConfig<T>,

    /// Membrane dynamics configuration
    membrane_config: MembraneDynamicsConfig<T>,

    /// Current simulation time
    current_time: T,

    /// Spike trains for each neuron
    spike_trains: HashMap<usize, SpikeTrain<T>>,

    /// Current membrane potentials
    membrane_potentials: Array1<T>,

    /// Synaptic weights
    synaptic_weights: Array2<T>,

    /// Last spike times for each neuron
    last_spike_times: Array1<T>,

    /// Refractory state
    refractory_until: Array1<T>,

    /// Per-neuron synaptic current `I_syn`, accumulated from external
    /// input spikes and from internal spikes propagated through
    /// `synaptic_weights` (F52), then consumed each step by
    /// `update_membrane_potential`'s `R * I_syn` term.
    synaptic_current: Array1<T>,

    /// Homeostatic scaling factors
    homeostatic_scales: Array1<T>,

    /// Spike buffer for temporal processing
    spike_buffer: VecDeque<Spike<T>>,

    /// Performance metrics
    metrics: NeuromorphicMetrics<T>,

    /// Plasticity model
    plasticity_model: PlasticityModel,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static
            + std::iter::Sum,
    > SpikingOptimizer<T>
{
    /// Create a new spiking optimizer
    pub fn new(
        config: SpikingConfig<T>,
        stdp_config: STDPConfig<T>,
        membrane_config: MembraneDynamicsConfig<T>,
        num_neurons: usize,
    ) -> Self {
        let resting_potential = membrane_config.resting_potential;
        Self {
            config,
            stdp_config,
            membrane_config,
            current_time: T::zero(),
            spike_trains: HashMap::new(),
            membrane_potentials: Array1::from_elem(num_neurons, resting_potential),
            synaptic_weights: Array2::ones((num_neurons, num_neurons))
                * T::from(0.1).unwrap_or_else(|| T::zero()),
            last_spike_times: Array1::from_elem(
                num_neurons,
                T::from(-1000.0).unwrap_or_else(|| T::zero()),
            ),
            refractory_until: Array1::zeros(num_neurons),
            synaptic_current: Array1::zeros(num_neurons),
            homeostatic_scales: Array1::ones(num_neurons),
            spike_buffer: VecDeque::new(),
            metrics: NeuromorphicMetrics::default(),
            plasticity_model: PlasticityModel::STDP,
        }
    }

    /// Encode continuous input as spike trains
    pub fn encode_input(&self, input: &Array1<T>) -> Result<Vec<SpikeTrain<T>>> {
        let mut spike_trains = Vec::new();

        for (neuron_id, &value) in input.iter().enumerate() {
            let spike_train = match self.config.encoding_method {
                SpikeEncodingMethod::RateCoding => self.rate_encode(neuron_id, value)?,
                SpikeEncodingMethod::TemporalCoding => self.temporal_encode(neuron_id, value)?,
                SpikeEncodingMethod::PopulationVectorCoding => {
                    self.population_vector_encode(neuron_id, value)?
                }
                SpikeEncodingMethod::SparseCoding => self.sparse_encode(neuron_id, value)?,
                _ => {
                    // Fallback to rate coding
                    self.rate_encode(neuron_id, value)?
                }
            };

            spike_trains.push(spike_train);
        }

        Ok(spike_trains)
    }

    /// The time window (ms) that rate coding integrates spikes over.
    /// Shared by [`Self::rate_encode`] and [`Self::rate_decode`] (F54):
    /// using two different windows (e.g. encoding over `simulation_time`
    /// but decoding over the much shorter `temporal_window`) introduces a
    /// systematic gain error between the two.
    fn rate_coding_window(&self) -> T {
        self.config.simulation_time
    }

    /// Rate encoding: firing rate proportional to input value
    fn rate_encode(&self, neuron_id: usize, value: T) -> Result<SpikeTrain<T>> {
        let max_rate = to_generic_or(RATE_CODING_MAX_RATE_HZ, T::one()); // Hz
        let firing_rate = value.abs() * max_rate;

        let mut spike_times = Vec::new();
        let dt = self.config.time_step;
        let total_time = self.rate_coding_window();

        let mut time = T::zero();
        while time < total_time {
            // Poisson process: probability of spike in dt
            let spike_prob = firing_rate * dt / to_generic_or(1000.0, T::one());

            if thread_rng().random::<f64>() < spike_prob.to_f64().unwrap_or(0.0) {
                spike_times.push(time);
            }

            time = time + dt;
        }

        Ok(SpikeTrain::new(neuron_id, spike_times))
    }

    /// Temporal encoding: spike time inversely proportional to input value
    fn temporal_encode(&self, neuron_id: usize, value: T) -> Result<SpikeTrain<T>> {
        let max_delay = T::from(20.0).unwrap_or_else(|| T::zero()); // 20 ms max delay
        let spike_time = if value > T::zero() {
            max_delay * (T::one() - value.min(T::one()))
        } else {
            max_delay // No spike for negative values
        };

        let spike_times = if spike_time < max_delay {
            vec![spike_time]
        } else {
            Vec::new()
        };

        Ok(SpikeTrain::new(neuron_id, spike_times))
    }

    /// Population vector encoding
    fn population_vector_encode(&self, neuron_id: usize, value: T) -> Result<SpikeTrain<T>> {
        // Simplified population vector encoding
        self.rate_encode(neuron_id, value)
    }

    /// Sparse encoding: only strong inputs generate spikes
    fn sparse_encode(&self, neuron_id: usize, value: T) -> Result<SpikeTrain<T>> {
        let threshold = T::from(0.5).unwrap_or_else(|| T::zero());

        if value.abs() > threshold {
            self.rate_encode(neuron_id, value)
        } else {
            Ok(SpikeTrain::new(neuron_id, Vec::new()))
        }
    }

    /// Decode spike trains to continuous output
    pub fn decode_output(&self, spike_trains: &[SpikeTrain<T>]) -> Result<Array1<T>> {
        let mut output = Array1::zeros(spike_trains.len());

        for (i, spike_train) in spike_trains.iter().enumerate() {
            output[i] = match self.config.decoding_method {
                SpikeDecodingMethod::RateDecoding => self.rate_decode(spike_train)?,
                SpikeDecodingMethod::TemporalDecoding => self.temporal_decode(spike_train)?,
                SpikeDecodingMethod::WeightedSpikeCount => {
                    self.weighted_spike_count_decode(spike_train)?
                }
                _ => {
                    // Fallback to rate decoding
                    self.rate_decode(spike_train)?
                }
            };
        }

        Ok(output)
    }

    /// Rate decoding: spike count normalized by time window. Uses the
    /// *same* window and max rate as [`Self::rate_encode`] (F54) — this
    /// used to normalize by `temporal_window` (20ms default) while encode
    /// spiked over `simulation_time` (1000ms default), a 50x mismatch.
    fn rate_decode(&self, spike_train: &SpikeTrain<T>) -> Result<T> {
        let window_duration = self.rate_coding_window();
        let spike_count = to_generic_or(spike_train.spike_count as f64, T::zero());
        let window_seconds = window_duration / to_generic_or(1000.0, T::one());
        if window_seconds <= T::zero() {
            return Ok(T::zero());
        }
        let rate = spike_count / window_seconds;
        let max_rate = to_generic_or(RATE_CODING_MAX_RATE_HZ, T::one());
        Ok(rate / max_rate) // Normalize by the same max rate used to encode
    }

    /// Temporal decoding: use first spike time
    fn temporal_decode(&self, spike_train: &SpikeTrain<T>) -> Result<T> {
        if spike_train.spike_times.is_empty() {
            Ok(T::zero())
        } else {
            let first_spike = spike_train.spike_times[0];
            let max_delay = T::from(20.0).unwrap_or_else(|| T::zero());
            Ok(T::one() - (first_spike / max_delay).min(T::one()))
        }
    }

    /// Weighted spike count decoding
    fn weighted_spike_count_decode(&self, spike_train: &SpikeTrain<T>) -> Result<T> {
        if spike_train.spike_times.is_empty() {
            return Ok(T::zero());
        }

        let mut weighted_sum = T::zero();
        let current_time = self.current_time;

        for &spike_time in &spike_train.spike_times {
            let time_diff = current_time - spike_time;
            let weight = (-time_diff / T::from(10.0).unwrap_or_else(|| T::zero())).exp(); // Exponential decay
            weighted_sum = weighted_sum + weight;
        }

        Ok(weighted_sum)
    }

    /// Simulate membrane dynamics for one time step
    pub fn simulate_step(&mut self, input_spikes: &[Spike<T>]) -> Result<Vec<Spike<T>>> {
        let mut output_spikes = Vec::new();
        let dt = self.config.time_step;

        // Process input _spikes
        for spike in input_spikes {
            self.process_input_spike(spike)?;
        }

        // Update membrane potentials
        for neuron_id in 0..self.membrane_potentials.len() {
            if self.current_time >= self.refractory_until[neuron_id] {
                self.update_membrane_potential(neuron_id, dt)?;

                // Check for spike threshold
                if self.membrane_potentials[neuron_id] >= self.membrane_config.threshold_potential {
                    let spike = self.generate_spike(neuron_id)?;
                    output_spikes.push(spike);
                }
            }
        }

        // Apply plasticity updates
        self.update_plasticity(&output_spikes)?;

        // Update homeostatic mechanisms
        if self.config.homeostatic_config.enable_homeostatic_scaling {
            self.update_homeostatic_scaling()?;
        }

        self.current_time = self.current_time + dt;

        Ok(output_spikes)
    }

    /// Process an input spike (F52): external input is accumulated as
    /// synaptic current rather than jumping the membrane potential
    /// directly, so it flows through the same `R * I_syn` leaky-integrator
    /// term as internally-propagated spikes.
    fn process_input_spike(&mut self, spike: &Spike<T>) -> Result<()> {
        let target_neuron = spike.postsynaptic_id.unwrap_or(spike.neuron_id);

        if target_neuron < self.synaptic_current.len() {
            let synaptic_current = spike.weight * spike.amplitude;
            self.synaptic_current[target_neuron] =
                self.synaptic_current[target_neuron] + synaptic_current;
        }

        Ok(())
    }

    /// Update membrane potential using a leaky integrate-and-fire model
    /// with a synaptic drive term (F52):
    /// `tau * dV/dt = (V_rest - V) + R * I_syn`, where `R = 1 /
    /// leak_conductance`. Previously this dropped `I_syn` entirely, so
    /// `synaptic_weights` (built up by STDP/Hebbian learning) never
    /// actually influenced the dynamics it was supposed to shape.
    fn update_membrane_potential(&mut self, neuron_id: usize, dt: T) -> Result<()> {
        let v = self.membrane_potentials[neuron_id];
        let v_rest = self.membrane_config.resting_potential;
        let tau = self.membrane_config.tau_membrane;
        let leak_conductance = self.membrane_config.leak_conductance;
        let membrane_resistance = if leak_conductance > T::zero() {
            T::one() / leak_conductance
        } else {
            T::zero()
        };
        let i_syn = self.synaptic_current[neuron_id];

        let dv_dt = if tau > T::zero() {
            ((v_rest - v) + membrane_resistance * i_syn) / tau
        } else {
            T::zero()
        };
        let new_v = v + dv_dt * dt;

        self.membrane_potentials[neuron_id] = new_v;

        // The injected current is consumed by this integration step (a
        // simple pulse model); new input/network spikes re-inject it.
        self.synaptic_current[neuron_id] = T::zero();

        Ok(())
    }

    /// Generate a spike when threshold is reached
    fn generate_spike(&mut self, neuron_id: usize) -> Result<Spike<T>> {
        // Reset membrane potential
        self.membrane_potentials[neuron_id] = self.membrane_config.reset_potential;

        // Set refractory period
        self.refractory_until[neuron_id] =
            self.current_time + self.membrane_config.refractory_period;

        // Update last spike time
        self.last_spike_times[neuron_id] = self.current_time;

        // Create spike
        let spike = Spike {
            neuron_id,
            time: self.current_time,
            amplitude: to_generic_or(1.0, T::one()),
            width: Some(to_generic_or(1.0, T::one())),
            weight: T::one(),
            presynaptic_id: None,
            postsynaptic_id: None,
        };

        // Propagate this spike to every postsynaptic target through the
        // real synaptic weight matrix (F52): this is what makes
        // `synaptic_weights` (shaped by STDP/Hebbian plasticity) actually
        // affect network dynamics instead of being a write-only matrix.
        for target_id in 0..self.synaptic_weights.ncols() {
            if target_id != neuron_id {
                let w = self.synaptic_weights[[neuron_id, target_id]];
                self.synaptic_current[target_id] = self.synaptic_current[target_id] + w;
            }
        }

        // Update spike train, recomputing firing_rate/duration from the
        // updated history (F51) rather than leaving them permanently
        // stale.
        self.spike_trains
            .entry(neuron_id)
            .or_insert_with(|| SpikeTrain::new(neuron_id, Vec::new()))
            .record_spike(self.current_time);

        // Update metrics
        self.metrics.total_spikes += 1;

        Ok(spike)
    }

    /// Update synaptic plasticity
    fn update_plasticity(&mut self, output_spikes: &[Spike<T>]) -> Result<()> {
        match self.plasticity_model {
            PlasticityModel::STDP => {
                self.update_stdp(output_spikes)?;
            }
            PlasticityModel::Hebbian => {
                self.update_hebbian(output_spikes)?;
            }
            _ => {
                // Default to STDP
                self.update_stdp(output_spikes)?;
            }
        }

        Ok(())
    }

    /// Update STDP (Spike Timing Dependent Plasticity)
    fn update_stdp(&mut self, output_spikes: &[Spike<T>]) -> Result<()> {
        let long_ago = to_generic_or(-1000.0, T::zero());

        for spike in output_spikes {
            let fired_id = spike.neuron_id;
            let fired_time = spike.time;

            for other_id in 0..self.last_spike_times.len() {
                if other_id == fired_id {
                    continue;
                }
                let other_time = self.last_spike_times[other_id];
                if other_time <= long_ago {
                    continue; // no valid spike history for `other_id` yet
                }

                // `other_id` fired before `fired_id` (now): it is
                // PRE, `fired_id` is POST, dt = t_post - t_pre > 0
                // => potentiation (LTP) on other_id -> fired_id.
                let dt_ltp = fired_time - other_time;
                let ltp = self.compute_stdp_update(dt_ltp);
                self.synaptic_weights[[other_id, fired_id]] =
                    (self.synaptic_weights[[other_id, fired_id]] + ltp)
                        .max(self.stdp_config.weight_min)
                        .min(self.stdp_config.weight_max);

                // `fired_id` is firing NOW, arriving after `other_id`'s
                // last spike: from `other_id`'s perspective as POST, this
                // is a PRE spike arriving late, dt = t_post - t_pre =
                // other_time - fired_time < 0 => depression (LTD) on
                // fired_id -> other_id. This is the presynaptic-trace
                // side of STDP that was previously unreachable (F50):
                // without it, `dt` computed from "post's own time minus
                // pre's last (necessarily past) spike time" was always
                // >= 0, so LTD never fired.
                let dt_ltd = other_time - fired_time;
                let ltd = self.compute_stdp_update(dt_ltd);
                self.synaptic_weights[[fired_id, other_id]] =
                    (self.synaptic_weights[[fired_id, other_id]] + ltd)
                        .max(self.stdp_config.weight_min)
                        .min(self.stdp_config.weight_max);
            }
        }

        Ok(())
    }

    /// Compute STDP weight update
    fn compute_stdp_update(&self, dt: T) -> T {
        if dt > T::zero() {
            // Post-before-pre: LTP (potentiation)
            let exp_arg = -dt / self.stdp_config.tau_pot;
            self.stdp_config.learning_rate_pot * exp_arg.exp()
        } else {
            // Pre-before-post: LTD (depression)
            let exp_arg = dt / self.stdp_config.tau_dep;
            -self.stdp_config.learning_rate_dep * exp_arg.exp()
        }
    }

    /// Update Hebbian plasticity. Presynaptic activity is the normalized
    /// depolarization fraction `((v - v_rest) / (v_thresh - v_rest))
    /// .max(0)` — 0 at rest, 1 at threshold (F53). The previous `v /
    /// v_threshold` ratio of two negative mV values was inverted: a
    /// neuron sitting at rest (no activity) produced a *larger* ratio
    /// than one nearly at threshold (maximal activity).
    fn update_hebbian(&mut self, output_spikes: &[Spike<T>]) -> Result<()> {
        let v_rest = self.membrane_config.resting_potential;
        let v_thresh = self.membrane_config.threshold_potential;
        let range = v_thresh - v_rest;

        for spike in output_spikes {
            let post_id = spike.neuron_id;

            for pre_id in 0..self.membrane_potentials.len() {
                if pre_id != post_id {
                    let pre_activity = if range != T::zero() {
                        ((self.membrane_potentials[pre_id] - v_rest) / range).max(T::zero())
                    } else {
                        T::zero()
                    };

                    let weight_change = self.stdp_config.learning_rate_pot * pre_activity;

                    self.synaptic_weights[[pre_id, post_id]] =
                        (self.synaptic_weights[[pre_id, post_id]] + weight_change)
                            .max(self.stdp_config.weight_min)
                            .min(self.stdp_config.weight_max);
                }
            }
        }

        Ok(())
    }

    /// Update homeostatic scaling (F51).
    ///
    /// Two bugs made this diverge geometrically: `firing_rate` was read
    /// from the spike train but never recomputed as spikes accumulated
    /// (fixed by [`SpikeTrain::record_spike`] in `generate_spike`), and
    /// the *cumulative, unbounded* `homeostatic_scales` value was
    /// multiplied into every weight on *every* call — so corrections
    /// compounded on top of corrections indefinitely. This now applies a
    /// single, clamped per-step multiplier each call, and separately
    /// clamps the cumulative scale record so it cannot drift without
    /// bound even over very long runs.
    fn update_homeostatic_scaling(&mut self) -> Result<()> {
        let target_rate = self.config.homeostatic_config.target_firing_rate;
        let time_constant = self.config.homeostatic_config.scaling_time_constant;
        let dt = self.config.time_step;
        if time_constant <= T::zero() {
            return Ok(());
        }

        let min_step = to_generic_or(0.9, T::one());
        let max_step = to_generic_or(1.1, T::one());
        let min_cumulative = to_generic_or(0.1, T::zero());
        let max_cumulative = to_generic_or(10.0, T::one());

        for neuron_id in 0..self.homeostatic_scales.len() {
            if let Some(spike_train) = self.spike_trains.get(&neuron_id) {
                let current_rate = spike_train.firing_rate;
                let rate_error = target_rate - current_rate;

                // Bounded per-step multiplicative correction toward the
                // target rate.
                let raw_step_scale = T::one() + rate_error * dt / time_constant;
                let step_multiplier = raw_step_scale.max(min_step).min(max_step);

                // Track the cumulative scale purely for observability,
                // clamped so it cannot grow or collapse without bound.
                self.homeostatic_scales[neuron_id] = (self.homeostatic_scales[neuron_id]
                    * step_multiplier)
                    .max(min_cumulative)
                    .min(max_cumulative);

                // Apply only the bounded per-step multiplier to weights,
                // not the (potentially very different) cumulative value.
                for pre_id in 0..self.synaptic_weights.nrows() {
                    self.synaptic_weights[[pre_id, neuron_id]] =
                        (self.synaptic_weights[[pre_id, neuron_id]] * step_multiplier)
                            .max(self.stdp_config.weight_min)
                            .min(self.stdp_config.weight_max);
                }
            }
        }

        Ok(())
    }

    /// Get current neuromorphic metrics
    pub fn get_metrics(&self) -> &NeuromorphicMetrics<T> {
        &self.metrics
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.current_time = T::zero();
        self.membrane_potentials
            .fill(self.membrane_config.resting_potential);
        self.last_spike_times
            .fill(T::from(-1000.0).unwrap_or_else(|| T::zero()));
        self.refractory_until.fill(T::zero());
        self.synaptic_current.fill(T::zero());
        self.spike_trains.clear();
        self.spike_buffer.clear();
        self.metrics = NeuromorphicMetrics::default();
    }
}

/// Spike train optimizer for temporal pattern learning
pub struct SpikeTrainOptimizer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + std::fmt::Debug + Send + Sync,
> {
    /// Configuration
    config: SpikingConfig<T>,

    /// Spike pattern templates
    pattern_templates: Vec<SpikePattern<T>>,

    /// Pattern matching threshold
    matching_threshold: T,

    /// Learning rate for pattern adaptation
    pattern_learning_rate: T,

    /// Temporal kernel for pattern comparison
    temporal_kernel: TemporalKernel<T>,
}

/// Spike pattern template
#[derive(Debug, Clone)]
pub struct SpikePattern<T: Float + Debug + Send + Sync + 'static> {
    /// Pattern ID
    pub pattern_id: usize,

    /// Spike times relative to pattern start
    pub relative_spike_times: Vec<T>,

    /// Pattern duration
    pub duration: T,

    /// Pattern weight/importance
    pub weight: T,

    /// Number of times pattern was observed
    pub observation_count: usize,
}

/// Temporal kernel for pattern matching
#[derive(Debug, Clone)]
pub struct TemporalKernel<T: Float + Debug + Send + Sync + 'static> {
    /// Kernel type
    pub kernel_type: TemporalKernelType,

    /// Kernel width (ms)
    pub width: T,

    /// Kernel parameters
    pub parameters: Vec<T>,
}

/// Types of temporal kernels
#[derive(Debug, Clone, Copy)]
pub enum TemporalKernelType {
    /// Gaussian kernel
    Gaussian,

    /// Exponential kernel
    Exponential,

    /// Alpha function kernel
    Alpha,

    /// Rectangular kernel
    Rectangular,
}

impl<T: Float + Debug + Send + Sync + scirs2_core::ndarray::ScalarOperand + std::fmt::Debug>
    SpikeTrainOptimizer<T>
{
    /// Create a new spike train optimizer
    pub fn new(config: SpikingConfig<T>) -> Self {
        Self {
            config,
            pattern_templates: Vec::new(),
            matching_threshold: T::from(0.8).unwrap_or_else(|| T::zero()),
            pattern_learning_rate: T::from(0.1).unwrap_or_else(|| T::zero()),
            temporal_kernel: TemporalKernel {
                kernel_type: TemporalKernelType::Gaussian,
                width: T::from(5.0).unwrap_or_else(|| T::zero()),
                parameters: vec![T::one()],
            },
        }
    }

    /// Learn spike patterns from training data
    pub fn learn_patterns(&mut self, spike_trains: &[SpikeTrain<T>]) -> Result<()> {
        for spike_train in spike_trains {
            self.extract_and_learn_patterns(spike_train)?;
        }

        Ok(())
    }

    /// Extract patterns from a spike train
    fn extract_and_learn_patterns(&mut self, spike_train: &SpikeTrain<T>) -> Result<()> {
        let window_size = T::from(50.0).unwrap_or_else(|| T::zero()); // 50 ms windows
        let step_size = T::from(10.0).unwrap_or_else(|| T::zero()); // 10 ms steps

        let mut window_start = T::zero();

        while window_start < spike_train.duration {
            let window_end = window_start + window_size;

            // Extract spikes in current window
            let window_spikes: Vec<T> = spike_train
                .spike_times
                .iter()
                .filter(|&&t| t >= window_start && t < window_end)
                .map(|&t| t - window_start) // Make relative to window start
                .collect();

            if !window_spikes.is_empty() {
                let pattern = SpikePattern {
                    pattern_id: self.pattern_templates.len(),
                    relative_spike_times: window_spikes,
                    duration: window_size,
                    weight: T::one(),
                    observation_count: 1,
                };

                // Check if similar pattern exists
                if let Some(similar_pattern_id) = self.find_similar_pattern(&pattern) {
                    self.update_pattern(similar_pattern_id, &pattern)?;
                } else {
                    self.pattern_templates.push(pattern);
                }
            }

            window_start = window_start + step_size;
        }

        Ok(())
    }

    /// Find similar existing pattern
    fn find_similar_pattern(&self, new_pattern: &SpikePattern<T>) -> Option<usize> {
        for (i, existing_pattern) in self.pattern_templates.iter().enumerate() {
            let similarity = self.compute_pattern_similarity(new_pattern, existing_pattern);
            if similarity > self.matching_threshold {
                return Some(i);
            }
        }

        None
    }

    /// Compute similarity between two spike patterns
    fn compute_pattern_similarity(
        &self,
        pattern1: &SpikePattern<T>,
        pattern2: &SpikePattern<T>,
    ) -> T {
        // Use Victor-Purpura distance or similar metric
        let max_spikes = pattern1
            .relative_spike_times
            .len()
            .max(pattern2.relative_spike_times.len());
        if max_spikes == 0 {
            return T::one();
        }

        // Simplified similarity based on spike count and timing
        let count_diff = (pattern1.relative_spike_times.len() as i32
            - pattern2.relative_spike_times.len() as i32)
            .abs() as f64;
        let count_similarity =
            T::one() - T::from(count_diff / max_spikes as f64).unwrap_or_else(|| T::zero());

        // Add temporal similarity if both patterns have spikes
        if !pattern1.relative_spike_times.is_empty() && !pattern2.relative_spike_times.is_empty() {
            let temporal_similarity = self.compute_temporal_similarity(
                &pattern1.relative_spike_times,
                &pattern2.relative_spike_times,
            );
            (count_similarity + temporal_similarity) / T::from(2.0).unwrap_or_else(|| T::zero())
        } else {
            count_similarity
        }
    }

    /// Compute temporal similarity between spike time sequences
    fn compute_temporal_similarity(&self, spikes1: &[T], spikes2: &[T]) -> T {
        // Use cross-correlation or DTW-like measure
        let mut max_correlation = T::zero();
        let max_shift = T::from(10.0).unwrap_or_else(|| T::zero()); // 10 ms max shift
        let shift_step = T::from(1.0).unwrap_or_else(|| T::zero());

        let mut shift = -max_shift;
        while shift <= max_shift {
            let correlation = self.compute_spike_correlation(spikes1, spikes2, shift);
            max_correlation = max_correlation.max(correlation);
            shift = shift + shift_step;
        }

        max_correlation
    }

    /// Compute spike correlation with time shift
    fn compute_spike_correlation(&self, spikes1: &[T], spikes2: &[T], shift: T) -> T {
        let mut correlation = T::zero();
        let kernel_width = self.temporal_kernel.width;

        for &t1 in spikes1 {
            for &t2 in spikes2 {
                let dt = (t1 - (t2 + shift)).abs();
                let kernel_value = (-dt * dt
                    / (T::from(2.0).unwrap_or_else(|| T::zero()) * kernel_width * kernel_width))
                    .exp();
                correlation = correlation + kernel_value;
            }
        }

        // Normalize by number of spike pairs
        if !spikes1.is_empty() && !spikes2.is_empty() {
            correlation / to_generic_or((spikes1.len() * spikes2.len()) as f64, T::one())
        } else {
            T::zero()
        }
    }

    /// Update existing pattern with new observation
    fn update_pattern(&mut self, pattern_id: usize, new_pattern: &SpikePattern<T>) -> Result<()> {
        if let Some(existing_pattern) = self.pattern_templates.get_mut(pattern_id) {
            // Update _pattern using exponential moving average
            let alpha = self.pattern_learning_rate;

            // Update spike times (simplified)
            if existing_pattern.relative_spike_times.len() == new_pattern.relative_spike_times.len()
            {
                for (existing_time, &new_time) in existing_pattern
                    .relative_spike_times
                    .iter_mut()
                    .zip(new_pattern.relative_spike_times.iter())
                {
                    *existing_time = *existing_time * (T::one() - alpha) + new_time * alpha;
                }
            }

            existing_pattern.observation_count += 1;
            existing_pattern.weight =
                existing_pattern.weight * (T::one() - alpha) + new_pattern.weight * alpha;
        }

        Ok(())
    }

    /// Recognize patterns in new spike train
    pub fn recognize_patterns(&self, spike_train: &SpikeTrain<T>) -> Result<Vec<(usize, T, T)>> {
        let mut recognized_patterns = Vec::new();
        let window_size = T::from(50.0).unwrap_or_else(|| T::zero());
        let step_size = T::from(5.0).unwrap_or_else(|| T::zero());

        let mut window_start = T::zero();

        while window_start < spike_train.duration {
            let window_end = window_start + window_size;

            let window_spikes: Vec<T> = spike_train
                .spike_times
                .iter()
                .filter(|&&t| t >= window_start && t < window_end)
                .map(|&t| t - window_start)
                .collect();

            if !window_spikes.is_empty() {
                let test_pattern = SpikePattern {
                    pattern_id: 0,
                    relative_spike_times: window_spikes,
                    duration: window_size,
                    weight: T::one(),
                    observation_count: 1,
                };

                // Find best matching pattern
                let mut best_match = (0, T::zero());
                for (i, template) in self.pattern_templates.iter().enumerate() {
                    let similarity = self.compute_pattern_similarity(&test_pattern, template);
                    if similarity > best_match.1 {
                        best_match = (i, similarity);
                    }
                }

                if best_match.1 > self.matching_threshold {
                    recognized_patterns.push((best_match.0, window_start, best_match.1));
                }
            }

            window_start = window_start + step_size;
        }

        Ok(recognized_patterns)
    }

    /// Get learned patterns
    pub fn get_patterns(&self) -> &[SpikePattern<T>] {
        &self.pattern_templates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_optimizer(num_neurons: usize) -> SpikingOptimizer<f64> {
        SpikingOptimizer::new(
            SpikingConfig::default(),
            STDPConfig::default(),
            MembraneDynamicsConfig::default(),
            num_neurons,
        )
    }

    fn dummy_spike(neuron_id: usize, time: f64) -> Spike<f64> {
        Spike {
            neuron_id,
            time,
            amplitude: 1.0,
            width: None,
            weight: 1.0,
            presynaptic_id: None,
            postsynaptic_id: None,
        }
    }

    /// F50: STDP must produce both potentiation (LTP) and depression
    /// (LTD), not just LTP.
    #[test]
    fn stdp_produces_both_potentiation_and_depression() {
        let mut optimizer = make_optimizer(2);
        optimizer.last_spike_times[0] = 5.0;

        optimizer
            .update_stdp(&[dummy_spike(1, 15.0)])
            .expect("update_stdp failed");

        let initial = 0.1;
        assert!(
            optimizer.synaptic_weights[[0, 1]] > initial,
            "LTP (0->1) did not fire: {}",
            optimizer.synaptic_weights[[0, 1]]
        );
        assert!(
            optimizer.synaptic_weights[[1, 0]] < initial,
            "LTD (1->0) did not fire (F50 regression): {}",
            optimizer.synaptic_weights[[1, 0]]
        );
    }

    /// F51: homeostatic scaling must not diverge — weights and scale
    /// factors stay bounded over many steps.
    #[test]
    fn homeostatic_scaling_does_not_blow_up() {
        let mut optimizer = make_optimizer(3);
        optimizer
            .config
            .homeostatic_config
            .enable_homeostatic_scaling = true;

        for step in 0..500 {
            optimizer.current_time = step as f64 * 0.1;
            let train = optimizer
                .spike_trains
                .entry(0)
                .or_insert_with(|| SpikeTrain::new(0, Vec::new()));
            if step % 5 == 0 {
                let t = optimizer.current_time;
                train.record_spike(t);
            }
            optimizer
                .update_homeostatic_scaling()
                .expect("update_homeostatic_scaling failed");
        }

        for &w in optimizer.synaptic_weights.iter() {
            assert!(w.is_finite(), "weight diverged: {w}");
            assert!(
                (0.0..=1.0).contains(&w),
                "weight left [weight_min, weight_max]: {w}"
            );
        }
        for &s in optimizer.homeostatic_scales.iter() {
            assert!(
                s.is_finite() && (0.1..=10.0).contains(&s),
                "homeostatic scale diverged (F51 regression): {s}"
            );
        }
    }

    /// F52: a spike propagated through a strong synaptic weight must
    /// actually move the postsynaptic membrane potential.
    #[test]
    fn synaptic_weights_propagate_into_membrane_dynamics() {
        let mut optimizer = make_optimizer(2);
        optimizer.synaptic_weights[[0, 1]] = 50.0;
        optimizer.membrane_potentials[1] = optimizer.membrane_config.resting_potential;
        optimizer.membrane_potentials[0] = optimizer.membrane_config.threshold_potential;

        optimizer.generate_spike(0).expect("generate_spike failed");
        let dt = optimizer.config.time_step;
        optimizer
            .update_membrane_potential(1, dt)
            .expect("update_membrane_potential failed");

        assert!(
            optimizer.membrane_potentials[1] > optimizer.membrane_config.resting_potential,
            "postsynaptic potential did not respond to the propagated synaptic weight (F52 regression)"
        );
    }

    /// F53: Hebbian presynaptic activity must increase monotonically with
    /// depolarization (0 at rest, up to 1 near threshold), not the
    /// inverted `v / v_threshold` ratio.
    #[test]
    fn hebbian_activity_increases_with_depolarization() {
        let run = |pre_potential: f64| -> f64 {
            let mut optimizer = make_optimizer(2);
            optimizer.plasticity_model = PlasticityModel::Hebbian;
            optimizer.membrane_potentials[0] = pre_potential;
            optimizer
                .update_hebbian(&[dummy_spike(1, 1.0)])
                .expect("update_hebbian failed");
            optimizer.synaptic_weights[[0, 1]]
        };

        let membrane_config = MembraneDynamicsConfig::<f64>::default();
        let weight_at_rest = run(membrane_config.resting_potential);
        let weight_near_threshold = run(membrane_config.threshold_potential);

        assert!(
            (weight_at_rest - 0.1).abs() < 1e-9,
            "resting potential should contribute zero Hebbian activity: {weight_at_rest}"
        );
        assert!(
            weight_near_threshold > weight_at_rest,
            "activity did not increase with depolarization (F53 regression): \
             rest={weight_at_rest}, near_threshold={weight_near_threshold}"
        );
    }

    /// F54: `decode(encode(v))` must recover `v` (averaged over trials to
    /// cancel Poisson spiking noise), not be off by the previous 50x
    /// window mismatch between `rate_encode` and `rate_decode`.
    #[test]
    fn rate_encode_decode_round_trip_within_noise_tolerance() {
        let optimizer = make_optimizer(1);
        let true_value = 0.5_f64;
        let trials = 20;

        let mut sum = 0.0;
        for _ in 0..trials {
            let train = optimizer
                .rate_encode(0, true_value)
                .expect("rate_encode failed");
            sum += optimizer.rate_decode(&train).expect("rate_decode failed");
        }
        let avg_decoded = sum / trials as f64;

        assert!(
            (avg_decoded - true_value).abs() < 0.08,
            "decode(encode(v)) did not recover v within noise tolerance (F54 regression): \
             v={true_value}, avg_decoded={avg_decoded}"
        );
    }
}
