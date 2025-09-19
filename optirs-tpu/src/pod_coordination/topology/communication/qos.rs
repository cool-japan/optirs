// Quality of Service and Traffic Management
//
// This module handles QoS settings, traffic management, bandwidth allocation,
// priority queuing, and flow control for TPU pod communication.

use scirs2_core::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::config::{DeviceId, TopologyId};

/// Network Quality of Service settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkQoSSettings {
    /// Traffic classes
    pub traffic_classes: Vec<TrafficClass>,
    /// Bandwidth allocation strategy
    pub bandwidth_allocation: BandwidthAllocation,
    /// Priority queuing settings
    pub priority_queuing: PriorityQueuingSettings,
    /// Flow control settings
    pub flow_control: FlowControlSettings,
    /// Traffic shaping settings
    pub traffic_shaping: TrafficShapingSettings,
    /// Congestion control settings
    pub congestion_control: CongestionControlSettings,
}

impl Default for NetworkQoSSettings {
    fn default() -> Self {
        Self {
            traffic_classes: vec![
                TrafficClass::realtime(),
                TrafficClass::high_priority(),
                TrafficClass::normal(),
                TrafficClass::best_effort(),
            ],
            bandwidth_allocation: BandwidthAllocation::default(),
            priority_queuing: PriorityQueuingSettings::default(),
            flow_control: FlowControlSettings::default(),
            traffic_shaping: TrafficShapingSettings::default(),
            congestion_control: CongestionControlSettings::default(),
        }
    }
}

/// Traffic class definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficClass {
    /// Class identifier
    pub class_id: String,
    /// Class name
    pub name: String,
    /// Priority level (0 = highest)
    pub priority: u8,
    /// Minimum guaranteed bandwidth (bps)
    pub min_bandwidth: f64,
    /// Maximum allowed bandwidth (bps)
    pub max_bandwidth: f64,
    /// Latency requirements
    pub latency_requirements: LatencyRequirements,
    /// Jitter tolerance
    pub jitter_tolerance: Duration,
    /// Packet loss tolerance (0.0-1.0)
    pub loss_tolerance: f64,
    /// Traffic characteristics
    pub characteristics: TrafficCharacteristics,
}

impl TrafficClass {
    /// Create real-time traffic class
    pub fn realtime() -> Self {
        Self {
            class_id: "realtime".to_string(),
            name: "Real-time Traffic".to_string(),
            priority: 0,
            min_bandwidth: 1_000_000_000.0,  // 1 Gbps
            max_bandwidth: 10_000_000_000.0, // 10 Gbps
            latency_requirements: LatencyRequirements {
                max_latency: Duration::from_millis(1),
                target_latency: Duration::from_micros(100),
            },
            jitter_tolerance: Duration::from_micros(50),
            loss_tolerance: 0.0001, // 0.01%
            characteristics: TrafficCharacteristics::realtime(),
        }
    }

    /// Create high-priority traffic class
    pub fn high_priority() -> Self {
        Self {
            class_id: "high_priority".to_string(),
            name: "High Priority Traffic".to_string(),
            priority: 1,
            min_bandwidth: 500_000_000.0,   // 500 Mbps
            max_bandwidth: 5_000_000_000.0, // 5 Gbps
            latency_requirements: LatencyRequirements {
                max_latency: Duration::from_millis(10),
                target_latency: Duration::from_millis(1),
            },
            jitter_tolerance: Duration::from_millis(1),
            loss_tolerance: 0.001, // 0.1%
            characteristics: TrafficCharacteristics::high_priority(),
        }
    }

    /// Create normal traffic class
    pub fn normal() -> Self {
        Self {
            class_id: "normal".to_string(),
            name: "Normal Traffic".to_string(),
            priority: 2,
            min_bandwidth: 100_000_000.0,   // 100 Mbps
            max_bandwidth: 1_000_000_000.0, // 1 Gbps
            latency_requirements: LatencyRequirements {
                max_latency: Duration::from_millis(100),
                target_latency: Duration::from_millis(10),
            },
            jitter_tolerance: Duration::from_millis(10),
            loss_tolerance: 0.01, // 1%
            characteristics: TrafficCharacteristics::normal(),
        }
    }

    /// Create best-effort traffic class
    pub fn best_effort() -> Self {
        Self {
            class_id: "best_effort".to_string(),
            name: "Best Effort Traffic".to_string(),
            priority: 3,
            min_bandwidth: 0.0,           // No guarantee
            max_bandwidth: f64::INFINITY, // No limit
            latency_requirements: LatencyRequirements {
                max_latency: Duration::from_secs(1),
                target_latency: Duration::from_millis(100),
            },
            jitter_tolerance: Duration::from_millis(100),
            loss_tolerance: 0.05, // 5%
            characteristics: TrafficCharacteristics::best_effort(),
        }
    }
}

/// Latency requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Target latency for optimization
    pub target_latency: Duration,
}

/// Traffic characteristics for different types of traffic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficCharacteristics {
    /// Traffic pattern type
    pub pattern: TrafficPattern,
    /// Burst characteristics
    pub burst_characteristics: BurstCharacteristics,
    /// Predictability of traffic
    pub predictability: TrafficPredictability,
    /// Sensitivity to delay
    pub delay_sensitivity: DelaySensitivity,
    /// Bandwidth elasticity
    pub bandwidth_elasticity: BandwidthElasticity,
}

impl TrafficCharacteristics {
    /// Real-time traffic characteristics
    pub fn realtime() -> Self {
        Self {
            pattern: TrafficPattern::ConstantBitRate,
            burst_characteristics: BurstCharacteristics {
                burstiness: 0.1,
                burst_duration: Duration::from_micros(100),
                burst_frequency: 1000.0,
            },
            predictability: TrafficPredictability::HighlyPredictable,
            delay_sensitivity: DelaySensitivity::VerySensitive,
            bandwidth_elasticity: BandwidthElasticity::Inelastic,
        }
    }

    /// High-priority traffic characteristics
    pub fn high_priority() -> Self {
        Self {
            pattern: TrafficPattern::VariableBitRate,
            burst_characteristics: BurstCharacteristics {
                burstiness: 0.3,
                burst_duration: Duration::from_millis(1),
                burst_frequency: 100.0,
            },
            predictability: TrafficPredictability::ModeratelyPredictable,
            delay_sensitivity: DelaySensitivity::Sensitive,
            bandwidth_elasticity: BandwidthElasticity::SlightlyElastic,
        }
    }

    /// Normal traffic characteristics
    pub fn normal() -> Self {
        Self {
            pattern: TrafficPattern::Bursty,
            burst_characteristics: BurstCharacteristics {
                burstiness: 0.5,
                burst_duration: Duration::from_millis(10),
                burst_frequency: 10.0,
            },
            predictability: TrafficPredictability::SomewhatPredictable,
            delay_sensitivity: DelaySensitivity::ModeratelySensitive,
            bandwidth_elasticity: BandwidthElasticity::Elastic,
        }
    }

    /// Best-effort traffic characteristics
    pub fn best_effort() -> Self {
        Self {
            pattern: TrafficPattern::Irregular,
            burst_characteristics: BurstCharacteristics {
                burstiness: 0.8,
                burst_duration: Duration::from_millis(100),
                burst_frequency: 1.0,
            },
            predictability: TrafficPredictability::Unpredictable,
            delay_sensitivity: DelaySensitivity::Insensitive,
            bandwidth_elasticity: BandwidthElasticity::HighlyElastic,
        }
    }
}

/// Traffic pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficPattern {
    /// Constant bit rate
    ConstantBitRate,
    /// Variable bit rate
    VariableBitRate,
    /// Bursty traffic
    Bursty,
    /// Periodic traffic
    Periodic { period: Duration },
    /// Self-similar traffic
    SelfSimilar { hurst_parameter: f64 },
    /// Irregular traffic
    Irregular,
}

/// Burst characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstCharacteristics {
    /// Burstiness factor (0.0-1.0)
    pub burstiness: f64,
    /// Average burst duration
    pub burst_duration: Duration,
    /// Burst frequency (bursts per second)
    pub burst_frequency: f64,
}

impl Default for BurstCharacteristics {
    fn default() -> Self {
        Self {
            burstiness: 0.3,
            burst_duration: Duration::from_millis(10),
            burst_frequency: 10.0,
        }
    }
}

/// Traffic predictability levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficPredictability {
    HighlyPredictable,
    ModeratelyPredictable,
    SomewhatPredictable,
    Unpredictable,
}

/// Delay sensitivity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelaySensitivity {
    VerySensitive,
    Sensitive,
    ModeratelySensitive,
    Insensitive,
}

/// Bandwidth elasticity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthElasticity {
    Inelastic,
    SlightlyElastic,
    Elastic,
    HighlyElastic,
}

/// Bandwidth allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAllocation {
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Per-class allocations
    pub class_allocations: HashMap<String, ClassAllocation>,
    /// Dynamic allocation settings
    pub dynamic_allocation: DynamicAllocationSettings,
    /// Fair queuing settings
    pub fair_queuing: FairQueuingSettings,
}

impl Default for BandwidthAllocation {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::ClassBasedWeightedFair,
            class_allocations: HashMap::new(),
            dynamic_allocation: DynamicAllocationSettings::default(),
            fair_queuing: FairQueuingSettings::default(),
        }
    }
}

/// Bandwidth allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Fixed allocation per class
    FixedAllocation,
    /// Priority-based allocation
    PriorityBased,
    /// Weighted fair queuing
    WeightedFairQueuing,
    /// Class-based weighted fair queuing
    ClassBasedWeightedFair,
    /// Deficit round robin
    DeficitRoundRobin,
    /// Hierarchical token bucket
    HierarchicalTokenBucket,
}

/// Per-class bandwidth allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassAllocation {
    /// Minimum guaranteed bandwidth (bps)
    pub min_bandwidth: f64,
    /// Maximum allowed bandwidth (bps)
    pub max_bandwidth: f64,
    /// Weight for fair queuing
    pub weight: f64,
    /// Burst size allowance (bytes)
    pub burst_size: usize,
    /// Token bucket parameters
    pub token_bucket: TokenBucketParameters,
}

/// Token bucket parameters for traffic shaping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBucketParameters {
    /// Token generation rate (tokens per second)
    pub rate: f64,
    /// Bucket size (maximum tokens)
    pub bucket_size: usize,
    /// Initial token count
    pub initial_tokens: usize,
}

/// Dynamic allocation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAllocationSettings {
    /// Enable dynamic allocation
    pub enabled: bool,
    /// Adaptation interval
    pub adaptation_interval: Duration,
    /// Load threshold for reallocation
    pub load_threshold: f64,
    /// Minimum allocation change percentage
    pub min_change_percentage: f64,
}

impl Default for DynamicAllocationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            adaptation_interval: Duration::from_secs(10),
            load_threshold: 0.8,
            min_change_percentage: 0.05,
        }
    }
}

/// Fair queuing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairQueuingSettings {
    /// Fair queuing algorithm
    pub algorithm: FairQueuingAlgorithm,
    /// Quantum size for round-robin algorithms
    pub quantum_size: usize,
    /// Virtual time update interval
    pub virtual_time_interval: Duration,
}

impl Default for FairQueuingSettings {
    fn default() -> Self {
        Self {
            algorithm: FairQueuingAlgorithm::WeightedFairQueuing,
            quantum_size: 1500, // Typical MTU size
            virtual_time_interval: Duration::from_millis(1),
        }
    }
}

/// Fair queuing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairQueuingAlgorithm {
    /// Weighted Fair Queuing
    WeightedFairQueuing,
    /// Deficit Round Robin
    DeficitRoundRobin,
    /// Stochastic Fair Queuing
    StochasticFairQueuing,
    /// Class-Based Queuing
    ClassBasedQueuing,
}

/// Priority queuing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityQueuingSettings {
    /// Number of priority levels
    pub priority_levels: usize,
    /// Queue sizes per priority level
    pub queue_sizes: Vec<usize>,
    /// Preemption settings
    pub preemption: PreemptionSettings,
    /// Anti-starvation mechanisms
    pub anti_starvation: AntiStarvationSettings,
}

impl Default for PriorityQueuingSettings {
    fn default() -> Self {
        Self {
            priority_levels: 4,
            queue_sizes: vec![1000, 2000, 4000, 8000], // Larger queues for lower priority
            preemption: PreemptionSettings::default(),
            anti_starvation: AntiStarvationSettings::default(),
        }
    }
}

/// Preemption settings for priority queuing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionSettings {
    /// Enable preemption
    pub enabled: bool,
    /// Preemption threshold
    pub threshold: usize,
    /// Preemption policy
    pub policy: PreemptionPolicy,
}

impl Default for PreemptionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 100,
            policy: PreemptionPolicy::HighestPriorityFirst,
        }
    }
}

/// Preemption policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    /// Highest priority first
    HighestPriorityFirst,
    /// Shortest job first
    ShortestJobFirst,
    /// Earliest deadline first
    EarliestDeadlineFirst,
    /// Least laxity first
    LeastLaxityFirst,
}

/// Anti-starvation mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiStarvationSettings {
    /// Enable anti-starvation
    pub enabled: bool,
    /// Maximum wait time before priority boost
    pub max_wait_time: Duration,
    /// Priority boost amount
    pub priority_boost: u8,
    /// Aging algorithm
    pub aging_algorithm: AgingAlgorithm,
}

impl Default for AntiStarvationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_wait_time: Duration::from_secs(1),
            priority_boost: 1,
            aging_algorithm: AgingAlgorithm::LinearAging,
        }
    }
}

/// Aging algorithms for preventing starvation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgingAlgorithm {
    /// Linear aging (constant priority boost)
    LinearAging,
    /// Exponential aging (exponentially increasing boost)
    ExponentialAging,
    /// Step aging (discrete priority level increases)
    StepAging,
    /// Dynamic aging (adaptive based on system load)
    DynamicAging,
}

/// Flow control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControlSettings {
    /// Enable flow control
    pub enabled: bool,
    /// Flow control algorithm
    pub algorithm: FlowControlAlgorithm,
    /// Window-based settings
    pub window_settings: WindowSettings,
    /// Rate-based settings
    pub rate_settings: RateSettings,
    /// Congestion avoidance settings
    pub congestion_avoidance: CongestionAvoidanceSettings,
}

impl Default for FlowControlSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: FlowControlAlgorithm::SlidingWindow,
            window_settings: WindowSettings::default(),
            rate_settings: RateSettings::default(),
            congestion_avoidance: CongestionAvoidanceSettings::default(),
        }
    }
}

/// Flow control algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowControlAlgorithm {
    /// Sliding window protocol
    SlidingWindow,
    /// Stop-and-wait
    StopAndWait,
    /// Go-back-N
    GoBackN,
    /// Selective repeat
    SelectiveRepeat,
    /// Credit-based flow control
    CreditBased,
    /// Rate-based flow control
    RateBased,
}

/// Window-based flow control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSettings {
    /// Initial window size
    pub initial_window_size: usize,
    /// Maximum window size
    pub max_window_size: usize,
    /// Window scaling factor
    pub scaling_factor: f64,
    /// Window adaptation algorithm
    pub adaptation_algorithm: WindowAdaptationAlgorithm,
}

impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            initial_window_size: 16,
            max_window_size: 256,
            scaling_factor: 2.0,
            adaptation_algorithm: WindowAdaptationAlgorithm::AIMD,
        }
    }
}

/// Window adaptation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAdaptationAlgorithm {
    /// Additive Increase Multiplicative Decrease
    AIMD,
    /// Binary Increase Congestion Avoidance
    BICA,
    /// TCP-like congestion control
    TCPLike,
    /// Custom adaptation
    Custom(String),
}

/// Rate-based flow control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateSettings {
    /// Initial transmission rate (bps)
    pub initial_rate: f64,
    /// Maximum transmission rate (bps)
    pub max_rate: f64,
    /// Rate adaptation algorithm
    pub adaptation_algorithm: RateAdaptationAlgorithm,
    /// Rate control parameters
    pub control_parameters: RateControlParameters,
}

impl Default for RateSettings {
    fn default() -> Self {
        Self {
            initial_rate: 1_000_000.0, // 1 Mbps
            max_rate: 1_000_000_000.0, // 1 Gbps
            adaptation_algorithm: RateAdaptationAlgorithm::PIDController,
            control_parameters: RateControlParameters::default(),
        }
    }
}

/// Rate adaptation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateAdaptationAlgorithm {
    /// PID controller
    PIDController,
    /// Kalman filter-based
    KalmanFilter,
    /// Fuzzy logic controller
    FuzzyLogic,
    /// Neural network-based
    NeuralNetwork,
}

/// Rate control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateControlParameters {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Control update interval
    pub update_interval: Duration,
}

impl Default for RateControlParameters {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.01,
            update_interval: Duration::from_millis(10),
        }
    }
}

/// Congestion avoidance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionAvoidanceSettings {
    /// Enable congestion avoidance
    pub enabled: bool,
    /// Congestion detection algorithm
    pub detection_algorithm: CongestionDetectionAlgorithm,
    /// Avoidance strategy
    pub avoidance_strategy: CongestionAvoidanceStrategy,
    /// Detection thresholds
    pub detection_thresholds: CongestionDetectionThresholds,
}

impl Default for CongestionAvoidanceSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_algorithm: CongestionDetectionAlgorithm::QueueLength,
            avoidance_strategy: CongestionAvoidanceStrategy::BackoffAndRetry,
            detection_thresholds: CongestionDetectionThresholds::default(),
        }
    }
}

/// Congestion detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionDetectionAlgorithm {
    /// Queue length monitoring
    QueueLength,
    /// Packet loss detection
    PacketLoss,
    /// Round-trip time monitoring
    RTTMonitoring,
    /// Explicit congestion notification
    ECN,
    /// Combined metrics
    Combined,
}

/// Congestion avoidance strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionAvoidanceStrategy {
    /// Backoff and retry
    BackoffAndRetry,
    /// Rate reduction
    RateReduction,
    /// Load balancing
    LoadBalancing,
    /// Alternative path selection
    AlternativePathSelection,
    /// Traffic prioritization
    TrafficPrioritization,
}

/// Congestion detection thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionDetectionThresholds {
    /// Queue length threshold (0.0-1.0)
    pub queue_length_threshold: f64,
    /// Packet loss threshold (0.0-1.0)
    pub packet_loss_threshold: f64,
    /// RTT increase threshold (multiple of baseline)
    pub rtt_increase_threshold: f64,
    /// Throughput decrease threshold (0.0-1.0)
    pub throughput_decrease_threshold: f64,
}

impl Default for CongestionDetectionThresholds {
    fn default() -> Self {
        Self {
            queue_length_threshold: 0.8,
            packet_loss_threshold: 0.01,
            rtt_increase_threshold: 2.0,
            throughput_decrease_threshold: 0.5,
        }
    }
}

/// Traffic shaping settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficShapingSettings {
    /// Enable traffic shaping
    pub enabled: bool,
    /// Shaping algorithm
    pub algorithm: TrafficShapingAlgorithm,
    /// Per-class shaping parameters
    pub class_parameters: HashMap<String, ShapingParameters>,
    /// Burst handling
    pub burst_handling: BurstHandlingSettings,
}

impl Default for TrafficShapingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: TrafficShapingAlgorithm::TokenBucket,
            class_parameters: HashMap::new(),
            burst_handling: BurstHandlingSettings::default(),
        }
    }
}

/// Traffic shaping algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficShapingAlgorithm {
    /// Token bucket
    TokenBucket,
    /// Leaky bucket
    LeakyBucket,
    /// Hierarchical token bucket
    HierarchicalTokenBucket,
    /// Generic cell rate algorithm
    GCRA,
}

/// Shaping parameters for traffic classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapingParameters {
    /// Peak rate (bps)
    pub peak_rate: f64,
    /// Sustained rate (bps)
    pub sustained_rate: f64,
    /// Burst size (bytes)
    pub burst_size: usize,
    /// Excess burst size (bytes)
    pub excess_burst_size: usize,
}

/// Burst handling settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstHandlingSettings {
    /// Burst tolerance strategy
    pub tolerance_strategy: BurstToleranceStrategy,
    /// Maximum burst duration
    pub max_burst_duration: Duration,
    /// Burst smoothing enabled
    pub smoothing_enabled: bool,
    /// Burst penalty settings
    pub penalty_settings: BurstPenaltySettings,
}

impl Default for BurstHandlingSettings {
    fn default() -> Self {
        Self {
            tolerance_strategy: BurstToleranceStrategy::LimitedTolerance,
            max_burst_duration: Duration::from_millis(100),
            smoothing_enabled: true,
            penalty_settings: BurstPenaltySettings::default(),
        }
    }
}

/// Burst tolerance strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BurstToleranceStrategy {
    /// No burst tolerance
    NoTolerance,
    /// Limited burst tolerance
    LimitedTolerance,
    /// Adaptive burst tolerance
    AdaptiveTolerance,
    /// Full burst tolerance
    FullTolerance,
}

/// Burst penalty settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstPenaltySettings {
    /// Apply penalty for excessive bursts
    pub apply_penalty: bool,
    /// Penalty type
    pub penalty_type: BurstPenaltyType,
    /// Penalty duration
    pub penalty_duration: Duration,
    /// Penalty severity
    pub penalty_severity: f64,
}

impl Default for BurstPenaltySettings {
    fn default() -> Self {
        Self {
            apply_penalty: true,
            penalty_type: BurstPenaltyType::RateReduction,
            penalty_duration: Duration::from_secs(1),
            penalty_severity: 0.5,
        }
    }
}

/// Types of burst penalties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BurstPenaltyType {
    /// Reduce transmission rate
    RateReduction,
    /// Drop excess packets
    PacketDrop,
    /// Delay transmission
    TransmissionDelay,
    /// Priority reduction
    PriorityReduction,
}

/// Congestion control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionControlSettings {
    /// Enable congestion control
    pub enabled: bool,
    /// Congestion control algorithm
    pub algorithm: CongestionControlAlgorithm,
    /// Algorithm parameters
    pub parameters: CongestionControlParameters,
    /// Fairness settings
    pub fairness: FairnessSettings,
}

impl Default for CongestionControlSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CongestionControlAlgorithm::CUBIC,
            parameters: CongestionControlParameters::default(),
            fairness: FairnessSettings::default(),
        }
    }
}

/// Congestion control algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionControlAlgorithm {
    /// TCP Reno
    Reno,
    /// TCP CUBIC
    CUBIC,
    /// TCP BBR
    BBR,
    /// Vegas
    Vegas,
    /// NewReno
    NewReno,
    /// Westwood
    Westwood,
}

/// Congestion control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionControlParameters {
    /// Slow start threshold
    pub slow_start_threshold: usize,
    /// Congestion window size
    pub congestion_window: usize,
    /// RTT measurement parameters
    pub rtt_parameters: RTTParameters,
    /// Loss detection parameters
    pub loss_detection: LossDetectionParameters,
}

impl Default for CongestionControlParameters {
    fn default() -> Self {
        Self {
            slow_start_threshold: 65536,
            congestion_window: 16,
            rtt_parameters: RTTParameters::default(),
            loss_detection: LossDetectionParameters::default(),
        }
    }
}

/// RTT measurement parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTTParameters {
    /// RTT smoothing factor
    pub smoothing_factor: f64,
    /// RTT variance smoothing factor
    pub variance_smoothing_factor: f64,
    /// Minimum RTT
    pub min_rtt: Duration,
    /// Maximum RTT
    pub max_rtt: Duration,
}

impl Default for RTTParameters {
    fn default() -> Self {
        Self {
            smoothing_factor: 0.125,
            variance_smoothing_factor: 0.25,
            min_rtt: Duration::from_micros(100),
            max_rtt: Duration::from_secs(1),
        }
    }
}

/// Loss detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossDetectionParameters {
    /// Duplicate ACK threshold
    pub duplicate_ack_threshold: usize,
    /// Timeout multiplier
    pub timeout_multiplier: f64,
    /// Maximum retransmission timeout
    pub max_retransmission_timeout: Duration,
    /// Fast retransmit enabled
    pub fast_retransmit_enabled: bool,
}

impl Default for LossDetectionParameters {
    fn default() -> Self {
        Self {
            duplicate_ack_threshold: 3,
            timeout_multiplier: 2.0,
            max_retransmission_timeout: Duration::from_secs(60),
            fast_retransmit_enabled: true,
        }
    }
}

/// Fairness settings for congestion control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessSettings {
    /// Fairness algorithm
    pub algorithm: FairnessAlgorithm,
    /// Fairness index target
    pub fairness_index_target: f64,
    /// Enable fairness monitoring
    pub monitoring_enabled: bool,
    /// Fairness adjustment interval
    pub adjustment_interval: Duration,
}

impl Default for FairnessSettings {
    fn default() -> Self {
        Self {
            algorithm: FairnessAlgorithm::ProportionalFair,
            fairness_index_target: 0.9,
            monitoring_enabled: true,
            adjustment_interval: Duration::from_secs(5),
        }
    }
}

/// Fairness algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessAlgorithm {
    /// Proportional fairness
    ProportionalFair,
    /// Max-min fairness
    MaxMinFair,
    /// Weighted fairness
    WeightedFair,
    /// Alpha fairness
    AlphaFair { alpha: f64 },
}

/// Traffic management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficManagementSettings {
    /// Load balancing settings
    pub load_balancing: LoadBalancingSettings,
    /// Traffic engineering settings
    pub traffic_engineering: TrafficEngineeringSettings,
    /// Admission control settings
    pub admission_control: AdmissionControlSettings,
    /// Resource reservation settings
    pub resource_reservation: ResourceReservationSettings,
}

impl Default for TrafficManagementSettings {
    fn default() -> Self {
        Self {
            load_balancing: LoadBalancingSettings::default(),
            traffic_engineering: TrafficEngineeringSettings::default(),
            admission_control: AdmissionControlSettings::default(),
            resource_reservation: ResourceReservationSettings::default(),
        }
    }
}

/// Load balancing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingSettings {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Health checking settings
    pub health_checking: HealthCheckingSettings,
    /// Failover settings
    pub failover: FailoverSettings,
}

impl Default for LoadBalancingSettings {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::WeightedRoundRobin,
            health_checking: HealthCheckingSettings::default(),
            failover: FailoverSettings::default(),
        }
    }
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    LeastResponseTime,
    HashBased,
    PowerOfTwoChoices,
}

/// Health checking settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckingSettings {
    /// Health check interval
    pub check_interval: Duration,
    /// Health check timeout
    pub check_timeout: Duration,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Recovery threshold
    pub recovery_threshold: usize,
}

impl Default for HealthCheckingSettings {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(5),
            check_timeout: Duration::from_secs(1),
            failure_threshold: 3,
            recovery_threshold: 2,
        }
    }
}

/// Failover settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverSettings {
    /// Enable automatic failover
    pub automatic_failover: bool,
    /// Failover timeout
    pub failover_timeout: Duration,
    /// Failback settings
    pub failback: FailbackSettings,
}

impl Default for FailoverSettings {
    fn default() -> Self {
        Self {
            automatic_failover: true,
            failover_timeout: Duration::from_secs(5),
            failback: FailbackSettings::default(),
        }
    }
}

/// Failback settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailbackSettings {
    /// Enable automatic failback
    pub automatic_failback: bool,
    /// Failback delay
    pub failback_delay: Duration,
    /// Preemptive failback
    pub preemptive: bool,
}

impl Default for FailbackSettings {
    fn default() -> Self {
        Self {
            automatic_failback: true,
            failback_delay: Duration::from_secs(30),
            preemptive: false,
        }
    }
}

/// Traffic engineering settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficEngineeringSettings {
    /// Path selection algorithm
    pub path_selection: PathSelectionAlgorithm,
    /// Traffic matrix estimation
    pub traffic_matrix: TrafficMatrixSettings,
    /// Optimization objectives
    pub optimization_objectives: Vec<OptimizationObjective>,
}

impl Default for TrafficEngineeringSettings {
    fn default() -> Self {
        Self {
            path_selection: PathSelectionAlgorithm::ShortestPath,
            traffic_matrix: TrafficMatrixSettings::default(),
            optimization_objectives: vec![
                OptimizationObjective::MinimizeLatency,
                OptimizationObjective::MaximizeThroughput,
                OptimizationObjective::BalanceLoad,
            ],
        }
    }
}

/// Path selection algorithms for traffic engineering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathSelectionAlgorithm {
    ShortestPath,
    WidestPath,
    LeastCongestedPath,
    LoadBalancedPath,
    MultiObjectiveOptimal,
}

/// Traffic matrix estimation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficMatrixSettings {
    /// Estimation method
    pub estimation_method: TrafficMatrixEstimationMethod,
    /// Measurement interval
    pub measurement_interval: Duration,
    /// History length
    pub history_length: Duration,
}

impl Default for TrafficMatrixSettings {
    fn default() -> Self {
        Self {
            estimation_method: TrafficMatrixEstimationMethod::DirectMeasurement,
            measurement_interval: Duration::from_secs(60),
            history_length: Duration::from_hours(24),
        }
    }
}

/// Traffic matrix estimation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficMatrixEstimationMethod {
    DirectMeasurement,
    StatisticalInference,
    MachineLearning,
    HybridApproach,
}

/// Optimization objectives for traffic engineering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MinimizeLatency,
    MaximizeThroughput,
    BalanceLoad,
    MinimizeHops,
    MaximizeReliability,
    MinimizeCost,
}

/// Admission control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionControlSettings {
    /// Enable admission control
    pub enabled: bool,
    /// Admission control algorithm
    pub algorithm: AdmissionControlAlgorithm,
    /// Resource thresholds
    pub resource_thresholds: ResourceThresholds,
    /// Rejection policies
    pub rejection_policies: RejectionPolicies,
}

impl Default for AdmissionControlSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: AdmissionControlAlgorithm::MeasurementBased,
            resource_thresholds: ResourceThresholds::default(),
            rejection_policies: RejectionPolicies::default(),
        }
    }
}

/// Admission control algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdmissionControlAlgorithm {
    ParameterBased,
    MeasurementBased,
    PredictionBased,
    HybridApproach,
}

/// Resource thresholds for admission control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// Bandwidth utilization threshold (0.0-1.0)
    pub bandwidth_threshold: f64,
    /// Latency threshold
    pub latency_threshold: Duration,
    /// Queue occupancy threshold (0.0-1.0)
    pub queue_threshold: f64,
    /// Packet loss threshold (0.0-1.0)
    pub loss_threshold: f64,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            bandwidth_threshold: 0.9,
            latency_threshold: Duration::from_millis(100),
            queue_threshold: 0.8,
            loss_threshold: 0.01,
        }
    }
}

/// Rejection policies for admission control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionPolicies {
    /// Primary rejection policy
    pub primary_policy: RejectionPolicy,
    /// Fallback policies
    pub fallback_policies: Vec<RejectionPolicy>,
    /// Grace period for retries
    pub retry_grace_period: Duration,
}

impl Default for RejectionPolicies {
    fn default() -> Self {
        Self {
            primary_policy: RejectionPolicy::PriorityBased,
            fallback_policies: vec![RejectionPolicy::RandomDrop, RejectionPolicy::OldestFirst],
            retry_grace_period: Duration::from_secs(5),
        }
    }
}

/// Rejection policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RejectionPolicy {
    /// Reject based on priority
    PriorityBased,
    /// Random rejection
    RandomDrop,
    /// First-in-first-out rejection
    FIFO,
    /// Last-in-first-out rejection
    LIFO,
    /// Reject oldest requests first
    OldestFirst,
    /// Reject newest requests first
    NewestFirst,
}

/// Resource reservation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReservationSettings {
    /// Enable resource reservation
    pub enabled: bool,
    /// Reservation protocol
    pub protocol: ReservationProtocol,
    /// Reservation policies
    pub policies: ReservationPolicies,
    /// Cleanup settings
    pub cleanup: ReservationCleanupSettings,
}

impl Default for ResourceReservationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            protocol: ReservationProtocol::RSVP,
            policies: ReservationPolicies::default(),
            cleanup: ReservationCleanupSettings::default(),
        }
    }
}

/// Resource reservation protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReservationProtocol {
    /// Resource Reservation Protocol
    RSVP,
    /// Constraint-based LSP setup using LDP
    CRLDP,
    /// MPLS Traffic Engineering
    MPLSTE,
    /// Custom protocol
    Custom(String),
}

/// Resource reservation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservationPolicies {
    /// Maximum reservation duration
    pub max_duration: Duration,
    /// Overbooking ratio
    pub overbooking_ratio: f64,
    /// Preemption settings
    pub preemption: ReservationPreemptionSettings,
}

impl Default for ReservationPolicies {
    fn default() -> Self {
        Self {
            max_duration: Duration::from_hours(24),
            overbooking_ratio: 1.2,
            preemption: ReservationPreemptionSettings::default(),
        }
    }
}

/// Reservation preemption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservationPreemptionSettings {
    /// Enable preemption
    pub enabled: bool,
    /// Preemption priority levels
    pub priority_levels: usize,
    /// Preemption policies
    pub policies: Vec<PreemptionPolicy>,
}

impl Default for ReservationPreemptionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            priority_levels: 8,
            policies: vec![PreemptionPolicy::HighestPriorityFirst],
        }
    }
}

/// Reservation cleanup settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservationCleanupSettings {
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Grace period before cleanup
    pub grace_period: Duration,
    /// Automatic cleanup enabled
    pub automatic_cleanup: bool,
}

impl Default for ReservationCleanupSettings {
    fn default() -> Self {
        Self {
            cleanup_interval: Duration::from_minutes(10),
            grace_period: Duration::from_minutes(5),
            automatic_cleanup: true,
        }
    }
}

/// Traffic manager for coordinating QoS and traffic management
#[derive(Debug)]
pub struct TrafficManager {
    /// QoS settings
    pub qos_settings: NetworkQoSSettings,
    /// Traffic management settings
    pub traffic_management: TrafficManagementSettings,
    /// Active traffic flows
    pub active_flows: HashMap<String, TrafficFlow>,
    /// Resource allocations
    pub resource_allocations: HashMap<DeviceId, ResourceAllocation>,
    /// Performance metrics
    pub performance_metrics: TrafficPerformanceMetrics,
}

impl TrafficManager {
    /// Create new traffic manager
    pub fn new(traffic_management: TrafficManagementSettings) -> Result<Self> {
        Ok(Self {
            qos_settings: NetworkQoSSettings::default(),
            traffic_management,
            active_flows: HashMap::new(),
            resource_allocations: HashMap::new(),
            performance_metrics: TrafficPerformanceMetrics::new(),
        })
    }

    /// Initialize traffic management
    pub fn initialize(&mut self, topology: &super::NetworkTopology) -> Result<()> {
        self.allocate_initial_resources(topology)?;
        self.setup_traffic_classes()?;
        Ok(())
    }

    /// Update configuration
    pub fn update_configuration(&mut self, settings: &TrafficManagementSettings) -> Result<()> {
        self.traffic_management = settings.clone();
        self.reconfigure_traffic_management()
    }

    /// Optimize traffic flows
    pub fn optimize_traffic(&mut self, metrics: &super::NetworkMetrics) -> Result<()> {
        self.analyze_traffic_patterns(metrics)?;
        self.adjust_allocations(metrics)?;
        self.update_qos_policies(metrics)?;
        Ok(())
    }

    /// Redistribute traffic after device failure
    pub fn redistribute_traffic(&mut self, failed_device: DeviceId) -> Result<()> {
        self.reroute_affected_flows(failed_device)?;
        self.rebalance_load()?;
        Ok(())
    }

    /// Rebalance traffic load
    pub fn rebalance_traffic(&mut self) -> Result<()> {
        self.calculate_optimal_distribution()?;
        self.apply_new_allocations()?;
        Ok(())
    }

    /// Drain traffic from device
    pub fn drain_traffic_from_device(&mut self, device_id: DeviceId) -> Result<()> {
        self.redirect_incoming_flows(device_id)?;
        self.complete_outgoing_flows(device_id)?;
        Ok(())
    }

    fn allocate_initial_resources(&mut self, topology: &super::NetworkTopology) -> Result<()> {
        // Implementation would allocate initial resources based on topology
        Ok(())
    }

    fn setup_traffic_classes(&mut self) -> Result<()> {
        // Implementation would set up traffic classes and QoS policies
        Ok(())
    }

    fn reconfigure_traffic_management(&mut self) -> Result<()> {
        // Implementation would reconfigure traffic management with new settings
        Ok(())
    }

    fn analyze_traffic_patterns(&mut self, metrics: &super::NetworkMetrics) -> Result<()> {
        // Implementation would analyze current traffic patterns
        Ok(())
    }

    fn adjust_allocations(&mut self, metrics: &super::NetworkMetrics) -> Result<()> {
        // Implementation would adjust resource allocations based on metrics
        Ok(())
    }

    fn update_qos_policies(&mut self, metrics: &super::NetworkMetrics) -> Result<()> {
        // Implementation would update QoS policies based on performance
        Ok(())
    }

    fn reroute_affected_flows(&mut self, failed_device: DeviceId) -> Result<()> {
        // Implementation would reroute flows affected by device failure
        Ok(())
    }

    fn rebalance_load(&mut self) -> Result<()> {
        // Implementation would rebalance load across available devices
        Ok(())
    }

    fn calculate_optimal_distribution(&mut self) -> Result<()> {
        // Implementation would calculate optimal traffic distribution
        Ok(())
    }

    fn apply_new_allocations(&mut self) -> Result<()> {
        // Implementation would apply new resource allocations
        Ok(())
    }

    fn redirect_incoming_flows(&mut self, device_id: DeviceId) -> Result<()> {
        // Implementation would redirect incoming flows away from device
        Ok(())
    }

    fn complete_outgoing_flows(&mut self, device_id: DeviceId) -> Result<()> {
        // Implementation would complete outgoing flows from device
        Ok(())
    }
}

/// Traffic flow representation
#[derive(Debug, Clone)]
pub struct TrafficFlow {
    /// Flow identifier
    pub flow_id: String,
    /// Source device
    pub source: DeviceId,
    /// Destination device
    pub destination: DeviceId,
    /// Traffic class
    pub traffic_class: String,
    /// Flow characteristics
    pub characteristics: TrafficCharacteristics,
    /// Current metrics
    pub metrics: FlowMetrics,
    /// Flow status
    pub status: FlowStatus,
}

/// Flow metrics
#[derive(Debug, Clone)]
pub struct FlowMetrics {
    /// Current throughput (bps)
    pub throughput: f64,
    /// Current latency
    pub latency: Duration,
    /// Packet loss rate (0.0-1.0)
    pub packet_loss: f64,
    /// Jitter
    pub jitter: Duration,
    /// Queue occupancy (0.0-1.0)
    pub queue_occupancy: f64,
}

/// Flow status
#[derive(Debug, Clone, PartialEq)]
pub enum FlowStatus {
    Active,
    Congested,
    Throttled,
    Suspended,
    Terminated,
}

/// Resource allocation per device
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Device identifier
    pub device_id: DeviceId,
    /// Allocated bandwidth (bps)
    pub bandwidth: f64,
    /// Buffer allocation (bytes)
    pub buffer_size: usize,
    /// Processing allocation (operations per second)
    pub processing_capacity: f64,
    /// Allocation status
    pub status: AllocationStatus,
}

/// Resource allocation status
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationStatus {
    Active,
    Overallocated,
    Underutilized,
    Failed,
}

/// Traffic performance metrics
#[derive(Debug, Clone)]
pub struct TrafficPerformanceMetrics {
    /// Overall network utilization (0.0-1.0)
    pub network_utilization: f64,
    /// Average network latency
    pub average_latency: Duration,
    /// Total network throughput (bps)
    pub total_throughput: f64,
    /// Packet loss rate (0.0-1.0)
    pub packet_loss_rate: f64,
    /// QoS compliance score (0.0-1.0)
    pub qos_compliance: f64,
    /// Load balancing effectiveness (0.0-1.0)
    pub load_balancing_score: f64,
}

impl TrafficPerformanceMetrics {
    /// Create new traffic performance metrics
    pub fn new() -> Self {
        Self {
            network_utilization: 0.0,
            average_latency: Duration::from_millis(0),
            total_throughput: 0.0,
            packet_loss_rate: 0.0,
            qos_compliance: 1.0,
            load_balancing_score: 1.0,
        }
    }
}
