// Energy-Efficient Optimization for Neuromorphic Computing
//
// This module implements energy-efficient optimization algorithms specifically designed
// for neuromorphic computing platforms, focusing on minimizing power consumption while
// maintaining performance and accuracy.

use super::{
    EventPriority, MembraneDynamicsConfig, NeuromorphicEvent, NeuromorphicMetrics, PlasticityModel,
    STDPConfig, SleepModeConfig, Spike, SpikeTrain, ThermalManagementConfig,
    ThermalThrottlingStrategy,
};
use crate::error::Result;
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use scirs2_core::numeric::Float;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Convert an `f64` literal/derived value to the generic float type `T`,
/// falling back to `fallback` when the numeric type cannot represent it.
/// Defensive only: always succeeds for the `f32`/`f64` types this crate
/// targets.
#[inline]
fn to_t_or<T: Float>(value: f64, fallback: T) -> T {
    T::from(value).unwrap_or_else(|| fallback)
}

// --- Documented device power model -----------------------------------
//
// These constants define a small but real CMOS-style power model used to
// derive `current_power` and every optimization strategy's savings from
// the actual `WorkloadSample`/system state, instead of fixed percentages.
// The model has two components:
//
//   P_total = P_static(active_neurons) + P_dynamic * (V/V_nom)^2 * (f/f_nom)
//
// where `P_dynamic` is itself an activity-weighted sum of spike, synaptic
// and communication contributions. `(V/V_nom)^2 * (f/f_nom)` is the
// standard CMOS dynamic-power scaling law (`P_dyn ∝ C·V²·f`).

/// Nominal (reference) operating point: matches `DVFSController`'s
/// mid-range voltage/frequency level, used as the DVFS scaling baseline.
const NOMINAL_VOLTAGE: f64 = 1.0;
const NOMINAL_FREQUENCY_MHZ: f64 = 1000.0;

/// Static leakage power per provisioned neuron (nW), independent of
/// activity — this is what power-gating and clock-gating recover.
const STATIC_POWER_PER_NEURON_NW: f64 = 0.05;
/// Dynamic energy per spike (nJ) at the nominal operating point.
const DYNAMIC_ENERGY_PER_SPIKE_NJ: f64 = 0.02;
/// Dynamic power per unit of synaptic activity (nW) at nominal V/f.
const DYNAMIC_POWER_PER_SYNAPTIC_ACTIVITY_NW: f64 = 5.0;
/// Dynamic power per unit of communication overhead (nW) at nominal V/f.
const DYNAMIC_POWER_PER_COMM_OVERHEAD_NW: f64 = 2.0;

/// Number of independently power/clock-gateable neuron domains (a coarse,
/// fixed partition of the provisioned neuron population).
const GATING_DOMAINS: usize = 8;
/// Minimum idle fraction before any domain is considered worth gating (the
/// wake-up overhead makes gating unprofitable below this).
const GATING_IDLE_THRESHOLD: f64 = 0.15;
/// Fraction of a fully idle domain's dynamic power that clock gating can
/// actually recover (gating logic itself has overhead).
const CLOCK_GATING_EFFICIENCY: f64 = 0.9;
/// Fraction of dynamic power recovered in light vs. deep sleep, before
/// scaling by idle fraction.
const LIGHT_SLEEP_SAVINGS: f64 = 0.4;
const DEEP_SLEEP_SAVINGS: f64 = 0.85;
/// Idle fraction above which sleep mode escalates from light to deep sleep.
const DEEP_SLEEP_IDLE_THRESHOLD: f64 = 0.7;
/// Thermal throttling proportional-controller range (°C): no throttling
/// below the safe temperature, `THERMAL_MAX_REDUCTION` reached at or above
/// the critical temperature, linear in between.
const THERMAL_SAFE_TEMP_C: f64 = 60.0;
const THERMAL_CRITICAL_TEMP_C: f64 = 90.0;
const THERMAL_MIN_REDUCTION: f64 = 0.05;
const THERMAL_MAX_REDUCTION: f64 = 0.6;

/// Energy optimization strategies for neuromorphic computing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnergyOptimizationStrategy {
    /// Dynamic voltage and frequency scaling
    DynamicVoltageScaling,

    /// Power gating for unused neurons
    PowerGating,

    /// Clock gating for inactive regions
    ClockGating,

    /// Adaptive precision reduction
    AdaptivePrecision,

    /// Sparse computation optimization
    SparseComputation,

    /// Event-driven processing
    EventDrivenProcessing,

    /// Sleep mode management
    SleepModeOptimization,

    /// Thermal-aware optimization
    ThermalAwareOptimization,

    /// Multi-level optimization
    MultiLevel,
}

/// Energy budget configuration
#[derive(Debug, Clone)]
pub struct EnergyBudget<T: Float + Debug + Send + Sync + 'static> {
    /// Total energy budget (nJ)
    pub total_budget: T,

    /// Current energy consumption (nJ)
    pub current_consumption: T,

    /// Energy budget per operation (nJ/op)
    pub per_operation_budget: T,

    /// Energy allocation per component
    pub component_allocation: HashMap<EnergyComponent, T>,

    /// Energy efficiency targets
    pub efficiency_targets: EnergyEfficiencyTargets<T>,

    /// Emergency energy reserves
    pub emergency_reserves: T,

    /// Energy monitoring frequency
    pub monitoring_frequency: Duration,
}

/// Energy components for budget allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnergyComponent {
    /// Synaptic operations
    SynapticOps,

    /// Membrane dynamics
    MembraneDynamics,

    /// Spike generation
    SpikeGeneration,

    /// Plasticity updates
    PlasticityUpdates,

    /// Memory access
    MemoryAccess,

    /// Communication
    Communication,

    /// Control logic
    ControlLogic,

    /// Thermal management
    ThermalManagement,
}

/// Energy efficiency targets
#[derive(Debug, Clone)]
pub struct EnergyEfficiencyTargets<T: Float + Debug + Send + Sync + 'static> {
    /// Operations per joule target
    pub ops_per_joule: T,

    /// Spikes per joule target
    pub spikes_per_joule: T,

    /// Synaptic updates per joule target
    pub synaptic_updates_per_joule: T,

    /// Memory bandwidth efficiency (ops/J/bandwidth)
    pub memory_bandwidth_efficiency: T,

    /// Thermal efficiency (performance/Watt/°C)
    pub thermal_efficiency: T,
}

/// Energy-efficient optimizer configuration
#[derive(Debug, Clone)]
pub struct EnergyEfficientConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Primary optimization strategy
    pub primary_strategy: EnergyOptimizationStrategy,

    /// Fallback strategies
    pub fallback_strategies: Vec<EnergyOptimizationStrategy>,

    /// Energy budget configuration
    pub energy_budget: EnergyBudget<T>,

    /// Enable adaptive strategy switching
    pub adaptive_strategy_switching: bool,

    /// Strategy switching threshold (efficiency drop %)
    pub strategy_switching_threshold: T,

    /// Enable predictive energy management
    pub predictive_energy_management: bool,

    /// Prediction horizon (ms)
    pub prediction_horizon: T,

    /// Enable energy harvesting support
    pub energy_harvesting: bool,

    /// Harvesting efficiency
    pub harvesting_efficiency: T,

    /// Enable distributed energy management
    pub distributed_energy_management: bool,

    /// Enable real-time energy monitoring
    pub real_time_monitoring: bool,

    /// Monitoring resolution (μs)
    pub monitoring_resolution: T,

    /// Enable energy-aware workload balancing
    pub energy_aware_load_balancing: bool,

    /// Energy optimization aggressiveness (0.0 to 1.0)
    pub optimization_aggressiveness: T,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for EnergyEfficientConfig<T> {
    fn default() -> Self {
        let mut component_allocation = HashMap::new();
        component_allocation.insert(
            EnergyComponent::SynapticOps,
            T::from(0.4).unwrap_or_else(|| T::zero()),
        );
        component_allocation.insert(
            EnergyComponent::MembraneDynamics,
            T::from(0.2).unwrap_or_else(|| T::zero()),
        );
        component_allocation.insert(
            EnergyComponent::SpikeGeneration,
            T::from(0.15).unwrap_or_else(|| T::zero()),
        );
        component_allocation.insert(
            EnergyComponent::PlasticityUpdates,
            T::from(0.1).unwrap_or_else(|| T::zero()),
        );
        component_allocation.insert(
            EnergyComponent::MemoryAccess,
            T::from(0.1).unwrap_or_else(|| T::zero()),
        );
        component_allocation.insert(
            EnergyComponent::Communication,
            T::from(0.05).unwrap_or_else(|| T::zero()),
        );

        Self {
            primary_strategy: EnergyOptimizationStrategy::DynamicVoltageScaling,
            fallback_strategies: vec![
                EnergyOptimizationStrategy::PowerGating,
                EnergyOptimizationStrategy::ClockGating,
                EnergyOptimizationStrategy::SparseComputation,
            ],
            energy_budget: EnergyBudget {
                total_budget: T::from(1000.0).unwrap_or_else(|| T::zero()), // 1 μJ
                current_consumption: T::zero(),
                per_operation_budget: T::from(10.0).unwrap_or_else(|| T::zero()), // 10 nJ per op
                component_allocation,
                efficiency_targets: EnergyEfficiencyTargets {
                    ops_per_joule: T::from(1e12).unwrap_or_else(|| T::zero()), // 1 TOP/J
                    spikes_per_joule: T::from(1e9).unwrap_or_else(|| T::zero()), // 1 GSp/J
                    synaptic_updates_per_joule: T::from(1e10).unwrap_or_else(|| T::zero()), // 10 GSyOp/J
                    memory_bandwidth_efficiency: T::from(1e6).unwrap_or_else(|| T::zero()),
                    thermal_efficiency: T::from(1e9).unwrap_or_else(|| T::zero()),
                },
                emergency_reserves: T::from(100.0).unwrap_or_else(|| T::zero()), // 100 nJ reserve
                monitoring_frequency: Duration::from_micros(100),
            },
            adaptive_strategy_switching: true,
            strategy_switching_threshold: T::from(0.1).unwrap_or_else(|| T::zero()), // 10% efficiency drop
            predictive_energy_management: true,
            prediction_horizon: T::from(10.0).unwrap_or_else(|| T::zero()), // 10 ms
            energy_harvesting: false,
            harvesting_efficiency: T::from(0.1).unwrap_or_else(|| T::zero()),
            distributed_energy_management: false,
            real_time_monitoring: true,
            monitoring_resolution: T::from(1.0).unwrap_or_else(|| T::zero()), // 1 μs
            energy_aware_load_balancing: true,
            optimization_aggressiveness: T::from(0.7).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Energy monitoring and tracking
#[derive(Debug, Clone)]
struct EnergyMonitor<
    T: Float
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug
        + std::iter::Sum
        + Send
        + Sync,
> {
    /// Energy consumption history
    consumption_history: VecDeque<(Instant, T)>,

    /// Power consumption history
    power_history: VecDeque<(Instant, T)>,

    /// Current power draw (nW)
    current_power: T,

    /// Peak power observed (nW)
    peak_power: T,

    /// Average power over window (nW)
    average_power: T,

    /// Energy per component
    component_energy: HashMap<EnergyComponent, T>,

    /// Efficiency metrics
    efficiency_metrics: EfficiencyMetrics<T>,

    /// Last monitoring update
    last_update: Instant,

    /// Monitoring window size
    window_size: Duration,
}

/// Efficiency metrics tracking
#[derive(Debug, Clone)]
struct EfficiencyMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Operations per joule (current)
    current_ops_per_joule: T,

    /// Spikes per joule (current)
    current_spikes_per_joule: T,

    /// Synaptic updates per joule (current)
    current_synaptic_updates_per_joule: T,

    /// Memory efficiency
    memory_efficiency: T,

    /// Thermal efficiency
    thermal_efficiency: T,

    /// Overall efficiency score
    overall_efficiency: T,
}

/// Dynamic voltage and frequency scaling controller
#[derive(Debug, Clone)]
struct DVFSController<T: Float + Debug + Send + Sync + 'static> {
    /// Available voltage levels (V)
    voltage_levels: Vec<T>,

    /// Available frequency levels (MHz)
    frequency_levels: Vec<T>,

    /// Current voltage index
    current_voltage_idx: usize,

    /// Current frequency index
    current_frequency_idx: usize,

    /// Performance requirements
    performance_requirements: PerformanceRequirements<T>,

    /// Voltage scaling factor
    voltage_scaling_factor: T,

    /// Frequency scaling factor
    frequency_scaling_factor: T,

    /// Adaptation rate
    adaptation_rate: T,
}

impl<T: Float + Debug + Send + Sync + 'static> DVFSController<T> {
    fn new() -> Self {
        Self {
            voltage_levels: vec![
                to_t_or(0.7, T::one()),
                to_t_or(0.9, T::one()),
                to_t_or(1.0, T::one()),
                to_t_or(1.2, T::one()),
            ],
            frequency_levels: vec![
                to_t_or(500.0, T::one()),
                to_t_or(1000.0, T::one()),
                to_t_or(1500.0, T::one()),
                to_t_or(2000.0, T::one()),
            ],
            current_voltage_idx: 2,
            current_frequency_idx: 2,
            performance_requirements: PerformanceRequirements {
                min_frequency: to_t_or(500.0, T::one()),
                max_frequency: to_t_or(2000.0, T::one()),
                min_voltage: to_t_or(0.7, T::one()),
                max_voltage: to_t_or(1.2, T::one()),
                performance_headroom: to_t_or(0.2, T::zero()),
                qos_requirements: QoSRequirements {
                    max_latency: to_t_or(10.0, T::one()),
                    max_jitter: to_t_or(1.0, T::one()),
                    min_throughput: to_t_or(1000.0, T::one()),
                    max_error_rate: to_t_or(0.001, T::zero()),
                },
            },
            voltage_scaling_factor: T::one(),
            frequency_scaling_factor: T::one(),
            adaptation_rate: to_t_or(0.1, T::zero()),
        }
    }

    /// Choose a voltage/frequency level proportional to how utilized the
    /// device is for this workload sample: a workload using few of the
    /// provisioned neurons is scaled down towards the lowest V/f level, a
    /// fully-active workload runs at the highest.
    fn compute_optimal_levels(
        &mut self,
        workload: &WorkloadSample<T>,
        total_neurons: usize,
    ) -> Result<(T, T)> {
        let total = to_t_or(total_neurons.max(1) as f64, T::one());
        let utilization = (to_t_or(workload.active_neurons as f64, T::zero()) / total)
            .max(T::zero())
            .min(T::one());
        let idx = (utilization * to_t_or((self.voltage_levels.len() - 1) as f64, T::zero()))
            .to_usize()
            .unwrap_or(2);

        self.current_voltage_idx = idx.min(self.voltage_levels.len() - 1);
        self.current_frequency_idx = idx.min(self.frequency_levels.len() - 1);

        Ok((
            self.voltage_levels[self.current_voltage_idx],
            self.frequency_levels[self.current_frequency_idx],
        ))
    }

    fn apply_scaling(&mut self, voltage: T, frequency: T) -> Result<()> {
        // Apply the voltage and frequency scaling
        self.voltage_scaling_factor = voltage / self.voltage_levels[2]; // Relative to nominal
        self.frequency_scaling_factor = frequency / self.frequency_levels[2];
        Ok(())
    }
}

/// Performance requirements for DVFS
#[derive(Debug, Clone)]
struct PerformanceRequirements<T: Float + Debug + Send + Sync + 'static> {
    /// Minimum required frequency (MHz)
    min_frequency: T,

    /// Maximum allowed frequency (MHz)
    max_frequency: T,

    /// Minimum required voltage (V)
    min_voltage: T,

    /// Maximum allowed voltage (V)
    max_voltage: T,

    /// Performance headroom (%)
    performance_headroom: T,

    /// Quality of service requirements
    qos_requirements: QoSRequirements<T>,
}

/// Quality of service requirements
#[derive(Debug, Clone)]
struct QoSRequirements<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum allowed latency (ms)
    max_latency: T,

    /// Maximum allowed jitter (ms)
    max_jitter: T,

    /// Minimum throughput (ops/s)
    min_throughput: T,

    /// Maximum error rate
    max_error_rate: T,
}

/// Power gating controller
#[derive(Debug, Clone)]
struct PowerGatingController<T: Float + Debug + Send + Sync + 'static> {
    /// Gated neuron groups
    gated_groups: HashMap<usize, GatedGroup>,

    /// Gate control policy
    gating_policy: GatingPolicy,

    /// Power gate overhead
    gate_overhead_time: Duration,

    /// Power gate overhead energy
    gate_overhead_energy: f64,

    /// Total provisioned neurons, used to size each gateable domain.
    total_neurons: usize,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> PowerGatingController<T> {
    fn new(total_neurons: usize) -> Self {
        Self {
            gated_groups: HashMap::new(),
            gating_policy: GatingPolicy::Adaptive,
            gate_overhead_time: Duration::from_micros(10),
            gate_overhead_energy: 0.001,
            total_neurons: total_neurons.max(1),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Power saved (nW) by gating `region_id`, proportional to the static
    /// leakage power of the neurons in that domain scaled by how idle the
    /// current workload is (a domain fully idle recovers its whole static
    /// power budget; the recovery is proportional otherwise).
    fn gate_region(&mut self, region_id: usize, idle_fraction: T) -> Result<T> {
        let neurons_per_domain = (self.total_neurons / GATING_DOMAINS.max(1)).max(1);
        let domain_static_power = to_t_or(
            neurons_per_domain as f64 * STATIC_POWER_PER_NEURON_NW,
            T::zero(),
        );
        let saved = domain_static_power * idle_fraction;

        let group = self
            .gated_groups
            .entry(region_id)
            .or_insert_with(|| GatedGroup {
                group_id: region_id,
                neuron_ids: Vec::new(),
                is_gated: false,
                last_activity: Instant::now(),
                inactivity_threshold: Duration::from_millis(10),
                wakeup_latency: Duration::from_micros(10),
            });
        group.is_gated = true;

        Ok(saved)
    }

    /// Identify which fixed neuron domains are idle enough (given this
    /// workload's idle fraction) to be worth power-gating. The number of
    /// domains gated scales with how idle the device is, rather than a
    /// single fixed region.
    fn identify_gatable_regions(&self, idle_fraction: T) -> Vec<usize> {
        if idle_fraction < to_t_or(GATING_IDLE_THRESHOLD, T::zero()) {
            return Vec::new();
        }
        let fraction = idle_fraction.to_f64().unwrap_or(0.0).clamp(0.0, 1.0);
        let gatable_domains =
            ((fraction * GATING_DOMAINS as f64).round() as usize).clamp(1, GATING_DOMAINS);
        (0..gatable_domains).collect()
    }
}

/// Gated neuron group
#[derive(Debug, Clone)]
struct GatedGroup {
    /// Group ID
    group_id: usize,

    /// Neuron IDs in group
    neuron_ids: Vec<usize>,

    /// Is currently gated
    is_gated: bool,

    /// Last activity time
    last_activity: Instant,

    /// Inactivity threshold
    inactivity_threshold: Duration,

    /// Wake-up latency
    wakeup_latency: Duration,
}

/// Power gating policies
#[derive(Debug, Clone, Copy)]
enum GatingPolicy {
    /// Gate based on inactivity time
    InactivityBased,

    /// Gate based on predicted usage
    PredictiveBased,

    /// Gate based on energy budget
    EnergyBudgetBased,

    /// Adaptive gating
    Adaptive,
}

/// Sparse computation optimizer
#[derive(Debug, Clone)]
struct SparseComputationOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Sparsity threshold
    sparsity_threshold: T,

    /// Sparse matrix representations
    sparse_matrices: HashMap<String, SparseMatrix<T>>,

    /// Sparsity patterns
    sparsity_patterns: Vec<SparsityPattern>,

    /// Compression algorithms
    compression_algorithms: Vec<CompressionAlgorithm>,

    /// Dynamic sparsity adaptation
    dynamic_adaptation: bool,

    /// Total provisioned neurons, used as the activity-estimate denominator
    /// when no real weight/activation matrix is available.
    total_neurons: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> SparseComputationOptimizer<T> {
    fn new(total_neurons: usize) -> Self {
        Self {
            sparsity_threshold: to_t_or(0.01, T::zero()),
            sparse_matrices: HashMap::new(),
            sparsity_patterns: vec![SparsityPattern::MagnitudeBased],
            compression_algorithms: vec![CompressionAlgorithm::Csr],
            dynamic_adaptation: true,
            total_neurons: total_neurons.max(1),
        }
    }

    /// Estimate the sparsity relevant to this optimization step.
    ///
    /// When `matrix` is provided, sparsity is the *real* fraction of
    /// near-zero entries in it (elements with `|v| <= sparsity_threshold`).
    /// Otherwise it falls back to an activity-derived estimate using the
    /// device's actual provisioned neuron count: `1 - active/total`.
    fn analyze_sparsity<S, Dm>(
        &mut self,
        workload: &WorkloadSample<T>,
        matrix: Option<&ArrayBase<S, Dm>>,
    ) -> Result<SparsityAnalysis<T>>
    where
        S: Data<Elem = T>,
        Dm: Dimension,
    {
        let sparsity_ratio = if let Some(m) = matrix {
            let total = m.len().max(1);
            let zero_count = m
                .iter()
                .filter(|&&v| v.abs() <= self.sparsity_threshold)
                .count();
            to_t_or(zero_count as f64 / total as f64, T::zero())
        } else {
            let total = to_t_or(self.total_neurons as f64, T::one());
            let active = to_t_or(workload.active_neurons as f64, T::zero());
            (T::one() - (active / total)).max(T::zero()).min(T::one())
        };

        Ok(SparsityAnalysis {
            sparsity_ratio,
            pattern: SparsityPattern::MagnitudeBased,
            potential_savings: sparsity_ratio * to_t_or(0.8, T::zero()),
        })
    }

    fn apply_compression(&mut self, analysis: &SparsityAnalysis<T>) -> Result<T> {
        // Apply compression based on sparsity analysis
        Ok(analysis.potential_savings)
    }

    fn apply_sparse_optimizations(&mut self, analysis: &SparsityAnalysis<T>) -> Result<T> {
        // Apply sparse optimizations based on analysis
        let compression_savings = self.apply_compression(analysis)?;
        Ok(compression_savings)
    }
}

#[derive(Debug, Clone)]
struct SparsityAnalysis<T: Float + Debug + Send + Sync + 'static> {
    sparsity_ratio: T,
    pattern: SparsityPattern,
    potential_savings: T,
}

/// Sparse matrix representation
#[derive(Debug, Clone)]
struct SparseMatrix<T: Float + Debug + Send + Sync + 'static> {
    /// Non-zero values
    values: Vec<T>,

    /// Row indices
    row_indices: Vec<usize>,

    /// Column pointers
    col_pointers: Vec<usize>,

    /// Matrix dimensions
    dimensions: (usize, usize),

    /// Sparsity ratio
    sparsity_ratio: T,
}

/// Sparsity patterns
#[derive(Debug, Clone, Copy)]
enum SparsityPattern {
    /// Random sparsity
    Random,

    /// Structured sparsity
    Structured,

    /// Block sparsity
    Block,

    /// Channel sparsity
    Channel,

    /// Magnitude-based pruning
    MagnitudeBased,
}

/// Compression algorithms for sparse data
#[derive(Debug, Clone, Copy)]
enum CompressionAlgorithm {
    /// Compressed Sparse Row (CSR)
    Csr,

    /// Compressed Sparse Column (CSC)
    Csc,

    /// Block Sparse Row (BSR)
    Bsr,

    /// Coordinate format (COO)
    Coo,

    /// Dictionary of Keys (DOK)
    Dok,
}

/// Energy-efficient optimizer
pub struct EnergyEfficientOptimizer<
    T: Float
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug
        + std::iter::Sum
        + Send
        + Sync,
> {
    /// Configuration
    config: EnergyEfficientConfig<T>,

    /// Energy monitor
    energy_monitor: EnergyMonitor<T>,

    /// DVFS controller
    dvfs_controller: DVFSController<T>,

    /// Power gating controller
    power_gating_controller: PowerGatingController<T>,

    /// Sparse computation optimizer
    sparse_optimizer: SparseComputationOptimizer<T>,

    /// Thermal management
    thermal_manager: ThermalManager<T>,

    /// Sleep mode controller
    sleep_controller: SleepModeController<T>,

    /// Predictive energy manager
    predictive_manager: PredictiveEnergyManager<T>,

    /// Current optimization strategy
    current_strategy: EnergyOptimizationStrategy,

    /// Strategy effectiveness history
    strategy_effectiveness: HashMap<EnergyOptimizationStrategy, T>,

    /// System state
    system_state: EnergySystemState<T>,

    /// Performance metrics
    metrics: NeuromorphicMetrics<T>,
}

/// Energy system state
#[derive(Debug, Clone)]
pub struct EnergySystemState<T: Float + Debug + Send + Sync + 'static> {
    /// Current energy consumption (nJ)
    pub current_energy: T,

    /// Current power consumption (nW)
    pub current_power: T,

    /// Temperature (°C)
    pub temperature: T,

    /// Active neuron count
    pub active_neurons: usize,

    /// Active synapses count
    pub active_synapses: usize,

    /// Current voltage (V)
    pub current_voltage: T,

    /// Current frequency (MHz)
    pub current_frequency: T,

    /// Gated regions
    pub gated_regions: Vec<usize>,

    /// Sleep mode status
    pub sleep_status: SleepStatus,
}

#[derive(Debug, Clone, Copy)]
pub enum SleepStatus {
    Active,
    LightSleep,
    DeepSleep,
    Hibernation,
}

/// Thermal manager for energy efficiency
#[derive(Debug, Clone)]
struct ThermalManager<T: Float + Debug + Send + Sync + 'static> {
    /// Thermal management configuration
    config: ThermalManagementConfig<T>,

    /// Current temperature reading (°C)
    current_temperature: T,

    /// Temperature history
    temperature_history: VecDeque<(Instant, T)>,

    /// Thermal model parameters
    thermal_model: ThermalModel<T>,

    /// Cooling strategies
    cooling_strategies: Vec<CoolingStrategy>,

    /// Active thermal throttling
    active_throttling: Option<ThermalThrottlingStrategy>,

    /// Timestamp of the previous `update()` call, used to integrate the
    /// thermal RC model over the actual elapsed time (F59).
    last_update: Instant,
}

impl<T: Float + Debug + Send + Sync + 'static> ThermalManager<T> {
    fn new(config: ThermalManagementConfig<T>) -> Self {
        Self {
            config,
            current_temperature: to_t_or(25.0, T::zero()),
            temperature_history: VecDeque::new(),
            thermal_model: ThermalModel {
                time_constant: to_t_or(10.0, T::one()),
                thermal_resistance: to_t_or(0.5, T::zero()),
                thermal_capacitance: to_t_or(1000.0, T::one()),
                ambient_temperature: to_t_or(25.0, T::zero()),
            },
            cooling_strategies: vec![CoolingStrategy::Passive],
            active_throttling: None,
            last_update: Instant::now(),
        }
    }

    /// Integrate the thermal RC model one step forward (F59):
    /// `dT/dt = (P·R + T_amb - T) / τ`, discretized over the actual
    /// elapsed time since the previous call as
    /// `T += dt/τ · (P·R + T_amb - T)`. Unlike the previous instantaneous
    /// `T = P·R + T_amb` formula, this respects thermal inertia (a sudden
    /// power spike heats the die gradually, not instantly).
    fn update(&mut self, system_state: &EnergySystemState<T>) -> Result<()> {
        let now = Instant::now();
        let dt_seconds = to_t_or(
            now.duration_since(self.last_update).as_secs_f64().max(1e-6),
            to_t_or(1e-3, T::one()),
        );
        self.last_update = now;

        let steady_state = system_state.current_power * self.thermal_model.thermal_resistance
            + self.thermal_model.ambient_temperature;
        let tau = if self.thermal_model.time_constant > T::zero() {
            self.thermal_model.time_constant
        } else {
            T::one()
        };
        self.current_temperature = self.current_temperature
            + (dt_seconds / tau) * (steady_state - self.current_temperature);

        self.temperature_history
            .push_back((now, self.current_temperature));
        // Bound the history so it cannot grow without limit (F59).
        while self.temperature_history.len() > 100 {
            self.temperature_history.pop_front();
        }
        Ok(())
    }
}

/// Thermal model for prediction
#[derive(Debug, Clone)]
struct ThermalModel<T: Float + Debug + Send + Sync + 'static> {
    /// Thermal time constant (s)
    time_constant: T,

    /// Thermal resistance (°C/W)
    thermal_resistance: T,

    /// Thermal capacitance (J/°C)
    thermal_capacitance: T,

    /// Ambient temperature (°C)
    ambient_temperature: T,
}

/// Cooling strategies
#[derive(Debug, Clone, Copy)]
enum CoolingStrategy {
    /// Passive cooling
    Passive,

    /// Active air cooling
    ActiveAir,

    /// Liquid cooling
    Liquid,

    /// Thermoelectric cooling
    Thermoelectric,

    /// Phase change cooling
    PhaseChange,
}

/// Sleep mode controller
#[derive(Debug, Clone)]
struct SleepModeController<T: Float + Debug + Send + Sync + 'static> {
    /// Sleep mode configuration
    config: SleepModeConfig<T>,

    /// Current sleep status
    current_status: SleepStatus,

    /// Inactivity timer
    inactivity_timer: Instant,

    /// Sleep transition history
    transition_history: VecDeque<(Instant, SleepStatus, SleepStatus)>,

    /// Wake-up triggers
    wakeup_triggers: Vec<WakeupTrigger>,
}

impl<T: Float + Debug + Send + Sync + 'static> SleepModeController<T> {
    fn new(monitoring_frequency: Duration) -> Self {
        Self {
            config: SleepModeConfig::default(),
            current_status: SleepStatus::Active,
            inactivity_timer: Instant::now(),
            transition_history: VecDeque::new(),
            wakeup_triggers: vec![WakeupTrigger::ExternalStimulus, WakeupTrigger::Timer],
        }
    }
}

/// Wake-up triggers for sleep mode
#[derive(Debug, Clone, Copy)]
enum WakeupTrigger {
    /// External stimulus
    ExternalStimulus,

    /// Timer expiration
    Timer,

    /// Energy threshold
    EnergyThreshold,

    /// Temperature threshold
    TemperatureThreshold,

    /// Performance requirement
    PerformanceRequirement,
}

/// Predictive energy manager
#[derive(Debug, Clone)]
struct PredictiveEnergyManager<T: Float + Debug + Send + Sync + 'static> {
    /// Prediction models
    prediction_models: HashMap<String, PredictionModel<T>>,

    /// Workload history
    workload_history: VecDeque<WorkloadSample<T>>,

    /// Energy consumption predictions
    energy_predictions: VecDeque<EnergyPrediction<T>>,

    /// Observed `(timestamp, power_nw)` samples used to fit the linear
    /// trend that [`Self::predict_energy`] extrapolates from (F18: this
    /// history used to never be populated, so predictions always fell back
    /// to a fabricated constant regardless of `horizon`).
    power_history: VecDeque<(Instant, T)>,

    /// Prediction accuracy metrics
    prediction_accuracy: T,

    /// Model update frequency
    model_update_frequency: Duration,
}

impl<T: Float + Debug + Send + Sync + 'static> PredictiveEnergyManager<T> {
    fn new() -> Self {
        Self {
            prediction_models: HashMap::new(),
            workload_history: VecDeque::new(),
            energy_predictions: VecDeque::new(),
            power_history: VecDeque::new(),
            prediction_accuracy: to_t_or(0.9, T::zero()),
            model_update_frequency: Duration::from_secs(60),
        }
    }

    /// Record an observed workload/power sample so future
    /// [`Self::predict_energy`] calls have real data to extrapolate from.
    fn record_sample(&mut self, workload: WorkloadSample<T>, power: T) {
        self.power_history.push_back((Instant::now(), power));
        while self.power_history.len() > 100 {
            self.power_history.pop_front();
        }
        self.workload_history.push_back(workload);
        while self.workload_history.len() > 100 {
            self.workload_history.pop_front();
        }
    }

    /// Predict total energy (nJ) expected over the next `horizon`, by
    /// extrapolating the recent average power draw (nW) across that
    /// window: `E = P_avg * horizon` (F18: `horizon` was previously
    /// accepted but never used, and with no history ever recorded this
    /// always returned a hardcoded `1.0`).
    fn predict_energy(&self, horizon: Duration) -> Result<T> {
        if self.power_history.is_empty() {
            return Ok(T::zero());
        }
        let sum: T = self
            .power_history
            .iter()
            .map(|(_, power)| *power)
            .fold(T::zero(), |acc, x| acc + x);
        let avg_power = sum / to_t_or(self.power_history.len() as f64, T::one());

        // Mirrors the file's `energy(nJ) = power(nW) * time(ms) / 1000`
        // convention used throughout the strategy implementations above.
        let horizon_ms = to_t_or(horizon.as_secs_f64() * 1000.0, T::zero());
        let predicted = avg_power * horizon_ms / to_t_or(1000.0, T::one());

        Ok(predicted)
    }
}

/// Prediction model for energy consumption
#[derive(Debug, Clone)]
struct PredictionModel<T: Float + Debug + Send + Sync + 'static> {
    /// Model type
    model_type: ModelType,

    /// Model parameters
    parameters: Vec<T>,

    /// Model accuracy
    accuracy: T,

    /// Training data size
    training_data_size: usize,

    /// Last update time
    last_update: Instant,
}

#[derive(Debug, Clone, Copy)]
enum ModelType {
    Linear,
    Polynomial,
    Exponential,
    NeuralNetwork,
    AutoRegressive,
}

/// Workload sample for prediction
#[derive(Debug, Clone)]
pub struct WorkloadSample<T: Float + Debug + Send + Sync + 'static> {
    /// Timestamp
    pub timestamp: Instant,

    /// Number of active neurons
    pub active_neurons: usize,

    /// Spike rate (Hz)
    pub spike_rate: T,

    /// Synaptic activity
    pub synaptic_activity: T,

    /// Memory access pattern
    pub memory_access_pattern: MemoryAccessPattern,

    /// Communication overhead
    pub communication_overhead: T,
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Sparse,
    Burst,
    Mixed,
}

/// Energy prediction
#[derive(Debug, Clone)]
struct EnergyPrediction<T: Float + Debug + Send + Sync + 'static> {
    /// Prediction timestamp
    timestamp: Instant,

    /// Predicted energy consumption (nJ)
    predicted_energy: T,

    /// Confidence interval
    confidence_interval: (T, T),

    /// Prediction horizon (ms)
    horizon: T,

    /// Model used
    model_type: ModelType,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug
            + std::iter::Sum,
    > EnergyEfficientOptimizer<T>
{
    /// Create a new energy-efficient optimizer
    pub fn new(_config: EnergyEfficientConfig<T>, numneurons: usize) -> Self {
        Self {
            config: _config.clone(),
            energy_monitor: EnergyMonitor::new(_config.energy_budget.monitoring_frequency),
            dvfs_controller: DVFSController::new(),
            power_gating_controller: PowerGatingController::new(numneurons),
            sparse_optimizer: SparseComputationOptimizer::new(numneurons),
            thermal_manager: ThermalManager::new(ThermalManagementConfig::default()),
            sleep_controller: SleepModeController::new(_config.energy_budget.monitoring_frequency),
            predictive_manager: PredictiveEnergyManager::new(),
            current_strategy: _config.primary_strategy,
            strategy_effectiveness: HashMap::new(),
            system_state: EnergySystemState {
                current_energy: T::zero(),
                current_power: T::zero(),
                temperature: T::from(25.0).unwrap_or_else(|| T::zero()), // 25°C ambient
                active_neurons: numneurons,
                active_synapses: numneurons * numneurons,
                current_voltage: T::from(1.0).unwrap_or_else(|| T::zero()), // 1V
                current_frequency: T::from(100.0).unwrap_or_else(|| T::zero()), // 100 MHz
                gated_regions: Vec::new(),
                sleep_status: SleepStatus::Active,
            },
            metrics: NeuromorphicMetrics::default(),
        }
    }

    /// Optimize energy consumption
    pub fn optimize_energy(
        &mut self,
        workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        self.optimize_energy_impl(workload, None::<&Array2<T>>)
    }

    /// Like [`Self::optimize_energy`], but lets the caller supply a real
    /// weight/activation matrix so that, when the active strategy is
    /// [`EnergyOptimizationStrategy::SparseComputation`], sparsity is
    /// measured from the matrix's actual zero fraction instead of an
    /// activity-derived estimate (F18).
    pub fn optimize_energy_with_matrix<S, Dm>(
        &mut self,
        workload: &WorkloadSample<T>,
        matrix: Option<&ArrayBase<S, Dm>>,
    ) -> Result<EnergyOptimizationResult<T>>
    where
        S: Data<Elem = T>,
        Dm: Dimension,
    {
        self.optimize_energy_impl(workload, matrix)
    }

    fn optimize_energy_impl<S, Dm>(
        &mut self,
        workload: &WorkloadSample<T>,
        matrix: Option<&ArrayBase<S, Dm>>,
    ) -> Result<EnergyOptimizationResult<T>>
    where
        S: Data<Elem = T>,
        Dm: Dimension,
    {
        // Update energy monitoring
        self.energy_monitor.update(&self.system_state)?;

        // Get energy predictions (horizon-aware forecast for the next
        // minute; exercised here so predictive strategy switching has a
        // real signal to react to in future extensions).
        let _prediction = if self.config.predictive_energy_management {
            self.predictive_manager
                .predict_energy(Duration::from_secs(60))?
        } else {
            T::zero()
        };

        // Apply current optimization strategy
        let optimization_result = match self.current_strategy {
            EnergyOptimizationStrategy::DynamicVoltageScaling => {
                self.apply_dvfs_optimization(workload)?
            }
            EnergyOptimizationStrategy::PowerGating => {
                self.apply_power_gating_optimization(workload)?
            }
            EnergyOptimizationStrategy::ClockGating => {
                self.apply_clock_gating_optimization(workload)?
            }
            EnergyOptimizationStrategy::SparseComputation => {
                self.apply_sparse_computation_optimization(workload, matrix)?
            }
            EnergyOptimizationStrategy::SleepModeOptimization => {
                self.apply_sleep_mode_optimization(workload)?
            }
            EnergyOptimizationStrategy::ThermalAwareOptimization => {
                self.apply_thermal_aware_optimization(workload)?
            }
            EnergyOptimizationStrategy::MultiLevel => {
                self.apply_multi_level_optimization(workload)?
            }
            _ => {
                // Default optimization
                self.apply_default_optimization(workload)?
            }
        };

        // Record this (workload, power) sample for future predictions
        // (F18: `predict_energy` previously always saw an empty history and
        // returned a fabricated constant).
        self.predictive_manager
            .record_sample(workload.clone(), self.system_state.current_power);

        // Evaluate strategy effectiveness
        self.evaluate_strategy_effectiveness(&optimization_result);

        // Adaptive strategy switching
        if self.config.adaptive_strategy_switching {
            self.consider_strategy_switch()?;
        }

        // Update thermal management (RC-integrated, F59) and sync the
        // authoritative temperature back into `system_state` so the
        // thermal-aware strategy above reacts to the real modeled
        // temperature rather than a disconnected, manually decayed value.
        self.thermal_manager.update(&self.system_state)?;
        self.system_state.temperature = self.thermal_manager.current_temperature;

        // Update metrics
        self.update_metrics(&optimization_result);

        Ok(optimization_result)
    }

    /// Fraction of the device's provisioned neurons that are idle for this
    /// workload sample, clamped to `[0, 1]`. This is the common activity
    /// signal driving clock-gating, power-gating and sleep-mode savings
    /// below (F18): a workload using few of the provisioned neurons frees
    /// up a correspondingly large fraction of gateable/sleepable capacity.
    fn idle_fraction(&self, workload: &WorkloadSample<T>) -> T {
        let total = to_t_or(self.system_state.active_neurons.max(1) as f64, T::one());
        let active = to_t_or(workload.active_neurons as f64, T::zero());
        (T::one() - (active / total)).max(T::zero()).min(T::one())
    }

    /// Estimate the instantaneous power draw (nW) implied by a workload
    /// sample at the device's current voltage/frequency operating point:
    /// `P = P_static(active_neurons) + P_dynamic(spikes, synapses, comm) *
    /// (V/V_nom)² * (f/f_nom)`. This replaces reading back a stored
    /// `current_power` field that starts at (and can get stuck at) zero;
    /// every strategy below derives its baseline from the actual workload.
    fn estimate_workload_power(&self, workload: &WorkloadSample<T>) -> T {
        // Static leakage is incurred by the whole provisioned chip
        // (`system_state.active_neurons`, fixed at construction), not just
        // by however many neurons this particular sample activates — an
        // idle neuron still leaks, which is exactly what clock/power
        // gating and sleep mode recover below via `idle_fraction`.
        let static_power = to_t_or(
            self.system_state.active_neurons as f64 * STATIC_POWER_PER_NEURON_NW,
            T::zero(),
        );
        let spike_power = workload.spike_rate * to_t_or(DYNAMIC_ENERGY_PER_SPIKE_NJ, T::zero());
        let synaptic_power =
            workload.synaptic_activity * to_t_or(DYNAMIC_POWER_PER_SYNAPTIC_ACTIVITY_NW, T::zero());
        let comm_power = workload.communication_overhead
            * to_t_or(DYNAMIC_POWER_PER_COMM_OVERHEAD_NW, T::zero());
        let dynamic_baseline = spike_power + synaptic_power + comm_power;

        let v_nom = to_t_or(NOMINAL_VOLTAGE, T::one());
        let f_nom = to_t_or(NOMINAL_FREQUENCY_MHZ, T::one());
        let voltage_ratio = if v_nom > T::zero() {
            self.system_state.current_voltage / v_nom
        } else {
            T::one()
        };
        let freq_ratio = if f_nom > T::zero() {
            self.system_state.current_frequency / f_nom
        } else {
            T::one()
        };

        static_power + dynamic_baseline * voltage_ratio * voltage_ratio * freq_ratio
    }

    /// Apply DVFS optimization
    fn apply_dvfs_optimization(
        &mut self,
        workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        // Baseline power under the workload at the CURRENT operating point.
        let initial_power = self.estimate_workload_power(workload);

        // Capture the pre-transition operating point BEFORE mutating state
        // (F17: previously voltage/frequency were overwritten first and
        // the reduction ratio was computed against those *new* values,
        // forcing numerator == denominator == 1.0 on every call).
        let v_old = self.system_state.current_voltage;
        let f_old = self.system_state.current_frequency;

        // Determine optimal voltage and frequency for this workload.
        let (optimal_voltage, optimal_frequency) = self
            .dvfs_controller
            .compute_optimal_levels(workload, self.system_state.active_neurons)?;

        let power_reduction =
            self.calculate_power_reduction(v_old, f_old, optimal_voltage, optimal_frequency);
        let performance_impact = self.calculate_performance_impact(f_old, optimal_frequency);
        let new_power = initial_power * power_reduction;
        let thermal_impact = self.calculate_thermal_impact(initial_power, new_power);

        // Commit the new operating point and derived power.
        self.system_state.current_voltage = optimal_voltage;
        self.system_state.current_frequency = optimal_frequency;
        self.system_state.current_power = new_power;

        // Update accumulated energy consumption (nJ) over a 1 ms step.
        let time_delta = to_t_or(1.0, T::one());
        let energy_delta = new_power * time_delta / to_t_or(1000.0, T::one());
        self.system_state.current_energy = self.system_state.current_energy + energy_delta;

        Ok(EnergyOptimizationResult {
            strategy_used: EnergyOptimizationStrategy::DynamicVoltageScaling,
            energy_saved: (initial_power - new_power).max(T::zero()) * time_delta
                / to_t_or(1000.0, T::one()),
            power_reduction: initial_power - new_power,
            performance_impact,
            thermal_impact,
            optimization_overhead: to_t_or(0.1, T::zero()), // 0.1 nJ overhead
        })
    }

    /// Apply power gating optimization
    fn apply_power_gating_optimization(
        &mut self,
        workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        let initial_power = self.estimate_workload_power(workload);
        let idle_fraction = self.idle_fraction(workload);

        // Identify idle domains worth gating, sized to how idle we are.
        let gatable_regions = self
            .power_gating_controller
            .identify_gatable_regions(idle_fraction);

        let mut total_power_saved = T::zero();
        for region_id in gatable_regions {
            let power_saved = self
                .power_gating_controller
                .gate_region(region_id, idle_fraction)?;
            total_power_saved = total_power_saved + power_saved;
            if !self.system_state.gated_regions.contains(&region_id) {
                self.system_state.gated_regions.push(region_id);
            }
        }

        let new_power = (initial_power - total_power_saved).max(T::zero());
        self.system_state.current_power = new_power;

        let time_delta = to_t_or(1.0, T::one());
        let energy_saved = total_power_saved * time_delta / to_t_or(1000.0, T::one());
        let overhead = to_t_or(self.power_gating_controller.gate_overhead_energy, T::zero())
            * to_t_or(self.system_state.gated_regions.len() as f64, T::zero());

        Ok(EnergyOptimizationResult {
            strategy_used: EnergyOptimizationStrategy::PowerGating,
            energy_saved,
            power_reduction: total_power_saved,
            performance_impact: T::zero(), // Gated domains resume on demand
            thermal_impact: total_power_saved * to_t_or(0.8, T::zero()),
            optimization_overhead: overhead,
        })
    }

    /// Apply sparse computation optimization. `matrix`, when present, gives
    /// the *real* sparsity of the current weight/activation tensor (F18);
    /// otherwise sparsity is estimated from workload activity.
    fn apply_sparse_computation_optimization<S, Dm>(
        &mut self,
        workload: &WorkloadSample<T>,
        matrix: Option<&ArrayBase<S, Dm>>,
    ) -> Result<EnergyOptimizationResult<T>>
    where
        S: Data<Elem = T>,
        Dm: Dimension,
    {
        let initial_power = self.estimate_workload_power(workload);

        // Analyze sparsity patterns
        let sparsity_analysis = self.sparse_optimizer.analyze_sparsity(workload, matrix)?;

        // Apply sparse optimizations
        let energy_savings = self
            .sparse_optimizer
            .apply_sparse_optimizations(&sparsity_analysis)?;

        // Update system state
        let new_power = initial_power * (T::one() - energy_savings);
        self.system_state.current_power = new_power;

        Ok(EnergyOptimizationResult {
            strategy_used: EnergyOptimizationStrategy::SparseComputation,
            energy_saved: initial_power * energy_savings,
            power_reduction: initial_power - new_power,
            performance_impact: energy_savings * to_t_or(0.1, T::zero()), // Small performance impact
            thermal_impact: (initial_power - new_power) * to_t_or(0.9, T::zero()),
            optimization_overhead: to_t_or(0.2, T::zero()), // Moderate overhead
        })
    }

    /// Apply multi-level optimization
    fn apply_multi_level_optimization(
        &mut self,
        workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        let mut total_result = EnergyOptimizationResult {
            strategy_used: EnergyOptimizationStrategy::MultiLevel,
            energy_saved: T::zero(),
            power_reduction: T::zero(),
            performance_impact: T::zero(),
            thermal_impact: T::zero(),
            optimization_overhead: T::zero(),
        };

        // Apply multiple strategies in sequence
        let strategies = [
            EnergyOptimizationStrategy::SparseComputation,
            EnergyOptimizationStrategy::DynamicVoltageScaling,
            EnergyOptimizationStrategy::PowerGating,
        ];

        for strategy in &strategies {
            let prev_strategy = self.current_strategy;
            self.current_strategy = *strategy;

            let result = match strategy {
                EnergyOptimizationStrategy::SparseComputation => {
                    self.apply_sparse_computation_optimization(workload, None::<&Array2<T>>)?
                }
                EnergyOptimizationStrategy::DynamicVoltageScaling => {
                    self.apply_dvfs_optimization(workload)?
                }
                EnergyOptimizationStrategy::PowerGating => {
                    self.apply_power_gating_optimization(workload)?
                }
                _ => continue,
            };

            // Accumulate results
            total_result.energy_saved = total_result.energy_saved + result.energy_saved;
            total_result.power_reduction = total_result.power_reduction + result.power_reduction;
            total_result.performance_impact =
                total_result.performance_impact + result.performance_impact;
            total_result.thermal_impact = total_result.thermal_impact + result.thermal_impact;
            total_result.optimization_overhead =
                total_result.optimization_overhead + result.optimization_overhead;

            self.current_strategy = prev_strategy;
        }

        Ok(total_result)
    }

    /// Apply default optimization
    fn apply_default_optimization(
        &mut self,
        _workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        // Minimal optimization - just monitoring
        Ok(EnergyOptimizationResult {
            strategy_used: self.current_strategy,
            energy_saved: T::zero(),
            power_reduction: T::zero(),
            performance_impact: T::zero(),
            thermal_impact: T::zero(),
            optimization_overhead: to_t_or(0.01, T::zero()),
        })
    }

    /// Apply clock gating optimization. The fraction of dynamic power
    /// recoverable is the workload's idle fraction times the gating
    /// circuit's own efficiency (it cannot recover 100% due to gating
    /// overhead) — derived from the workload, not a fixed percentage.
    fn apply_clock_gating_optimization(
        &mut self,
        workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        let initial_power = self.estimate_workload_power(workload);
        let idle_fraction = self.idle_fraction(workload);
        let reduction_factor = idle_fraction * to_t_or(CLOCK_GATING_EFFICIENCY, T::zero());
        let new_power = initial_power * (T::one() - reduction_factor);

        self.system_state.current_power = new_power;

        Ok(EnergyOptimizationResult {
            strategy_used: EnergyOptimizationStrategy::ClockGating,
            energy_saved: initial_power * reduction_factor,
            power_reduction: initial_power - new_power,
            performance_impact: T::zero(), // Gated clocks resume with no latency
            thermal_impact: (initial_power - new_power) * to_t_or(0.8, T::zero()),
            optimization_overhead: to_t_or(0.05, T::zero()),
        })
    }

    /// Apply sleep mode optimization. Sleep depth (light vs. deep) and the
    /// resulting savings scale with the workload's idle fraction, rather
    /// than a fixed 50% constant.
    fn apply_sleep_mode_optimization(
        &mut self,
        workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        let initial_power = self.estimate_workload_power(workload);
        let idle_fraction = self.idle_fraction(workload);

        let deep_sleep_threshold = to_t_or(DEEP_SLEEP_IDLE_THRESHOLD, T::one());
        let (status, base_savings, wakeup_latency) = if idle_fraction >= deep_sleep_threshold {
            (SleepStatus::DeepSleep, DEEP_SLEEP_SAVINGS, 0.2)
        } else {
            (SleepStatus::LightSleep, LIGHT_SLEEP_SAVINGS, 0.1)
        };
        self.system_state.sleep_status = status;

        let reduction_factor = idle_fraction * to_t_or(base_savings, T::zero());
        let new_power = initial_power * (T::one() - reduction_factor);
        self.system_state.current_power = new_power;

        Ok(EnergyOptimizationResult {
            strategy_used: EnergyOptimizationStrategy::SleepModeOptimization,
            energy_saved: initial_power * reduction_factor,
            power_reduction: initial_power - new_power,
            performance_impact: to_t_or(wakeup_latency, T::zero()),
            thermal_impact: (initial_power - new_power) * to_t_or(0.95, T::zero()),
            optimization_overhead: to_t_or(0.1, T::zero()),
        })
    }

    /// Apply thermal-aware optimization: a proportional controller that
    /// scales power reduction linearly with how far the (RC-modeled)
    /// temperature has overshot a safe operating range, rather than a
    /// hardcoded two-step reduction factor.
    fn apply_thermal_aware_optimization(
        &mut self,
        workload: &WorkloadSample<T>,
    ) -> Result<EnergyOptimizationResult<T>> {
        let initial_power = self.estimate_workload_power(workload);

        let safe_temp = to_t_or(THERMAL_SAFE_TEMP_C, T::zero());
        let critical_temp = to_t_or(THERMAL_CRITICAL_TEMP_C, T::one());
        let min_reduction = to_t_or(THERMAL_MIN_REDUCTION, T::zero());
        let max_reduction = to_t_or(THERMAL_MAX_REDUCTION, T::zero());

        let span = (critical_temp - safe_temp).max(to_t_or(1e-6, T::one()));
        let overshoot = ((self.system_state.temperature - safe_temp) / span)
            .max(T::zero())
            .min(T::one());
        let reduction_factor = min_reduction + (max_reduction - min_reduction) * overshoot;

        let new_power = initial_power * (T::one() - reduction_factor);
        self.system_state.current_power = new_power;

        Ok(EnergyOptimizationResult {
            strategy_used: EnergyOptimizationStrategy::ThermalAwareOptimization,
            energy_saved: initial_power * reduction_factor,
            power_reduction: initial_power - new_power,
            performance_impact: reduction_factor * to_t_or(0.5, T::zero()),
            thermal_impact: initial_power - new_power,
            optimization_overhead: to_t_or(0.15, T::zero()),
        })
    }

    /// Ratio of new to old power (`P_new / P_old`) under a simplified CMOS
    /// dynamic power model `P ∝ V²·f`. Equals 1.0 only when voltage and
    /// frequency are genuinely unchanged; any real DVFS transition yields a
    /// ratio strictly different from 1.0 (F17).
    fn calculate_power_reduction(&self, v_old: T, f_old: T, v_new: T, f_new: T) -> T {
        let denom = v_old * v_old * f_old;
        if denom <= T::zero() {
            return T::one();
        }
        (v_new * v_new * f_new) / denom
    }

    /// Calculate performance impact of a frequency change:
    /// `(old_freq - new_freq) / old_freq`.
    fn calculate_performance_impact(&self, old_frequency: T, new_frequency: T) -> T {
        if old_frequency <= T::zero() {
            return T::zero();
        }
        (old_frequency - new_frequency) / old_frequency
    }

    /// Calculate thermal impact (°C-equivalent reduction) of a power
    /// change, via the thermal model's resistance: `ΔP · R_thermal`.
    fn calculate_thermal_impact(&self, old_power: T, newpower: T) -> T {
        let power_reduction = old_power - newpower;
        power_reduction * self.thermal_manager.thermal_model.thermal_resistance
    }

    /// Evaluate strategy effectiveness
    fn evaluate_strategy_effectiveness(&mut self, result: &EnergyOptimizationResult<T>) {
        // Calculate effectiveness score
        let effectiveness =
            result.energy_saved / (result.optimization_overhead + to_t_or(1e-6, T::zero()));

        // Update strategy effectiveness history
        *self
            .strategy_effectiveness
            .entry(result.strategy_used)
            .or_insert(T::zero()) = effectiveness;
    }

    /// Consider switching optimization strategy
    fn consider_strategy_switch(&mut self) -> Result<()> {
        if let Some(&current_effectiveness) =
            self.strategy_effectiveness.get(&self.current_strategy)
        {
            // Find best alternative strategy
            if let Some((&best_strategy, &best_effectiveness)) = self
                .strategy_effectiveness
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                // Switch if improvement exceeds threshold (guarded against
                // division by a zero/negative baseline effectiveness).
                if current_effectiveness > T::zero() {
                    let improvement =
                        (best_effectiveness - current_effectiveness) / current_effectiveness;
                    if improvement > self.config.strategy_switching_threshold {
                        self.current_strategy = best_strategy;
                    }
                } else if best_effectiveness > T::zero() {
                    self.current_strategy = best_strategy;
                }
            }
        }

        Ok(())
    }

    /// Update optimization metrics
    fn update_metrics(&mut self, _result: &EnergyOptimizationResult<T>) {
        self.metrics.energy_consumption = self.system_state.current_energy;
        self.metrics.power_consumption = self.system_state.current_power;
        let ambient = to_t_or(25.0, T::one());
        self.metrics.thermal_efficiency = if self.system_state.temperature > T::zero() {
            ambient / self.system_state.temperature
        } else {
            T::one()
        };
    }

    /// Get current energy budget status
    pub fn get_energy_budget_status(&self) -> EnergyBudgetStatus<T> {
        let remaining_budget =
            self.config.energy_budget.total_budget - self.system_state.current_energy;
        let budget_utilization =
            self.system_state.current_energy / self.config.energy_budget.total_budget;

        EnergyBudgetStatus {
            total_budget: self.config.energy_budget.total_budget,
            current_consumption: self.system_state.current_energy,
            remaining_budget,
            budget_utilization,
            emergency_reserve_available: remaining_budget
                > self.config.energy_budget.emergency_reserves,
        }
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &NeuromorphicMetrics<T> {
        &self.metrics
    }

    /// Get current system state
    pub fn get_system_state(&self) -> &EnergySystemState<T> {
        &self.system_state
    }
}

/// Energy optimization result
#[derive(Debug, Clone)]
pub struct EnergyOptimizationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Strategy that was used
    pub strategy_used: EnergyOptimizationStrategy,

    /// Energy saved (nJ)
    pub energy_saved: T,

    /// Power reduction (nW)
    pub power_reduction: T,

    /// Performance impact (ratio)
    pub performance_impact: T,

    /// Thermal impact (°C reduction)
    pub thermal_impact: T,

    /// Optimization overhead (nJ)
    pub optimization_overhead: T,
}

/// Energy budget status
#[derive(Debug, Clone)]
pub struct EnergyBudgetStatus<T: Float + Debug + Send + Sync + 'static> {
    /// Total energy budget (nJ)
    pub total_budget: T,

    /// Current energy consumption (nJ)
    pub current_consumption: T,

    /// Remaining budget (nJ)
    pub remaining_budget: T,

    /// Budget utilization (0.0 to 1.0)
    pub budget_utilization: T,

    /// Emergency reserve available
    pub emergency_reserve_available: bool,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug
            + std::iter::Sum,
    > EnergyMonitor<T>
{
    fn new(_monitoringfrequency: Duration) -> Self {
        Self {
            consumption_history: VecDeque::new(),
            power_history: VecDeque::new(),
            current_power: T::zero(),
            peak_power: T::zero(),
            average_power: T::zero(),
            component_energy: HashMap::new(),
            efficiency_metrics: EfficiencyMetrics::default(),
            last_update: Instant::now(),
            window_size: Duration::from_secs(1),
        }
    }

    fn update(&mut self, systemstate: &EnergySystemState<T>) -> Result<()> {
        let now = Instant::now();
        self.consumption_history
            .push_back((now, systemstate.current_energy));
        self.power_history
            .push_back((now, systemstate.current_power));

        // Clean old entries out of both histories (F59: `power_history` was
        // previously never trimmed and grew without bound).
        while let Some(&(time_, _)) = self.consumption_history.front() {
            if now.duration_since(time_) > self.window_size {
                self.consumption_history.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(time_, _)) = self.power_history.front() {
            if now.duration_since(time_) > self.window_size {
                self.power_history.pop_front();
            } else {
                break;
            }
        }

        // Update current metrics
        self.current_power = systemstate.current_power;
        self.peak_power = self.peak_power.max(systemstate.current_power);

        // Update average power
        if !self.power_history.is_empty() {
            let sum: T = self.power_history.iter().map(|(_, power)| *power).sum();
            self.average_power = sum / to_t_or(self.power_history.len() as f64, T::one());
        }

        self.last_update = now;
        Ok(())
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for EfficiencyMetrics<T> {
    fn default() -> Self {
        Self {
            current_ops_per_joule: T::zero(),
            current_spikes_per_joule: T::zero(),
            current_synaptic_updates_per_joule: T::zero(),
            memory_efficiency: T::zero(),
            thermal_efficiency: T::zero(),
            overall_efficiency: T::zero(),
        }
    }
}

#[cfg(test)]
#[path = "energy_efficient_tests.rs"]
mod tests;
