// Coordination Strategies and Consensus Algorithms
//
// This module implements various coordination strategies, consensus algorithms,
// and leader election mechanisms for TPU pod coordination.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{DeviceId, PodId, CoordinationSessionId};

/// Coordination strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// Centralized coordination with a single coordinator
    Centralized {
        leader_election: LeaderElectionConfig,
    },
    /// Distributed coordination using consensus
    Distributed {
        consensus: ConsensusAlgorithm,
    },
    /// Hierarchical coordination with multiple levels
    Hierarchical {
        hierarchy: HierarchyConfig,
    },
    /// Hybrid approach combining multiple strategies
    Hybrid {
        strategies: Vec<CoordinationStrategy>,
        selection_criteria: SelectionCriteria,
    },
    /// Adaptive strategy that changes based on conditions
    Adaptive {
        criteria: AdaptiveCriteria,
        fallback_strategy: Box<CoordinationStrategy>,
    },
}

impl Default for CoordinationStrategy {
    fn default() -> Self {
        Self::Distributed {
            consensus: ConsensusAlgorithm::Raft {
                election_timeout: Duration::from_millis(150),
                heartbeat_interval: Duration::from_millis(50),
                max_log_entries_per_request: 100,
            },
        }
    }
}

/// Consensus algorithms for distributed coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    /// Raft consensus algorithm
    Raft {
        election_timeout: Duration,
        heartbeat_interval: Duration,
        max_log_entries_per_request: usize,
    },
    /// PBFT (Practical Byzantine Fault Tolerance)
    PBFT {
        view_timeout: Duration,
        checkpoint_interval: usize,
        max_requests_in_batch: usize,
    },
    /// PAXOS consensus
    PAXOS {
        prepare_timeout: Duration,
        accept_timeout: Duration,
        max_proposals_per_round: usize,
    },
    /// Simplified Byzantine consensus
    Byzantine {
        round_timeout: Duration,
        max_byzantine_nodes: usize,
        signature_verification: bool,
    },
    /// Fast consensus for high-performance scenarios
    Fast {
        fast_round_timeout: Duration,
        slow_round_timeout: Duration,
        optimization_threshold: f64,
    },
    /// Custom consensus algorithm
    Custom {
        name: String,
        parameters: HashMap<String, String>,
    },
}

/// Leader election configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderElectionConfig {
    /// Election algorithm
    pub algorithm: LeaderElectionAlgorithm,
    /// Election timeout
    pub election_timeout: Duration,
    /// Term duration
    pub term_duration: Duration,
    /// Re-election triggers
    pub reelection_triggers: Vec<ReElectionTrigger>,
    /// Leader health check interval
    pub health_check_interval: Duration,
}

impl Default for LeaderElectionConfig {
    fn default() -> Self {
        Self {
            algorithm: LeaderElectionAlgorithm::Raft,
            election_timeout: Duration::from_millis(150),
            term_duration: Duration::from_minutes(5),
            reelection_triggers: vec![
                ReElectionTrigger::LeaderFailure,
                ReElectionTrigger::PerformanceDegradation { threshold: 0.7 },
                ReElectionTrigger::Timeout,
            ],
            health_check_interval: Duration::from_secs(1),
        }
    }
}

/// Leader election algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeaderElectionAlgorithm {
    /// Raft leader election
    Raft,
    /// Bully algorithm
    Bully,
    /// Ring algorithm
    Ring,
    /// Chang-Roberts algorithm
    ChangRoberts,
    /// Priority-based election
    Priority {
        priority_function: PriorityFunction,
    },
    /// Performance-based election
    Performance {
        metrics: Vec<String>,
        weights: HashMap<String, f64>,
    },
}

/// Priority functions for leader election
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityFunction {
    /// Highest device ID
    HighestId,
    /// Lowest device ID
    LowestId,
    /// Best performance metrics
    BestPerformance,
    /// Longest uptime
    LongestUptime,
    /// Custom priority calculation
    Custom(String),
}

/// Re-election triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReElectionTrigger {
    /// Leader failure detected
    LeaderFailure,
    /// Performance degradation below threshold
    PerformanceDegradation { threshold: f64 },
    /// Election timeout reached
    Timeout,
    /// Manual trigger
    Manual,
    /// Network partition recovery
    PartitionRecovery,
    /// Load balancing requirement
    LoadBalancing,
}

/// Hierarchical coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyConfig {
    /// Tree structure type
    pub structure: TreeStructure,
    /// Coordination flow direction
    pub flow: CoordinationFlow,
    /// Level-specific settings
    pub level_settings: HashMap<usize, LevelSettings>,
    /// Escalation policies
    pub escalation_policies: Vec<EscalationPolicy>,
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            structure: TreeStructure::BalancedTree { branching_factor: 4 },
            flow: CoordinationFlow::TopDown,
            level_settings: HashMap::new(),
            escalation_policies: vec![
                EscalationPolicy::TimeoutEscalation {
                    timeout: Duration::from_secs(30),
                    target_level: 1,
                },
                EscalationPolicy::FailureEscalation {
                    failure_threshold: 3,
                    target_level: 0,
                },
            ],
        }
    }
}

/// Tree structure types for hierarchical coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeStructure {
    /// Balanced tree with fixed branching factor
    BalancedTree { branching_factor: usize },
    /// Binary tree structure
    BinaryTree,
    /// Star topology (single root with all leaves)
    Star,
    /// Custom tree structure
    Custom {
        parent_child_mapping: HashMap<DeviceId, Vec<DeviceId>>,
    },
}

/// Coordination flow directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationFlow {
    /// Top-down coordination (from root to leaves)
    TopDown,
    /// Bottom-up coordination (from leaves to root)
    BottomUp,
    /// Bidirectional coordination
    Bidirectional,
    /// Peer-to-peer at each level
    PeerToPeer,
}

/// Level-specific coordination settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelSettings {
    /// Coordination timeout for this level
    pub timeout: Duration,
    /// Synchronization requirements
    pub sync_requirements: SyncRequirements,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceSettings,
}

impl Default for LevelSettings {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            sync_requirements: SyncRequirements::default(),
            performance_thresholds: PerformanceThresholds::default(),
            fault_tolerance: FaultToleranceSettings::default(),
        }
    }
}

/// Synchronization requirements for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequirements {
    /// Minimum number of nodes that must participate
    pub min_participants: usize,
    /// Percentage of nodes that must agree
    pub agreement_threshold: f64,
    /// Maximum time to wait for synchronization
    pub max_sync_time: Duration,
    /// Required consistency level
    pub consistency_level: ConsistencyLevel,
}

impl Default for SyncRequirements {
    fn default() -> Self {
        Self {
            min_participants: 1,
            agreement_threshold: 0.51, // Simple majority
            max_sync_time: Duration::from_secs(30),
            consistency_level: ConsistencyLevel::StrongConsistency,
        }
    }
}

/// Consistency levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency (all nodes see the same data)
    StrongConsistency,
    /// Eventual consistency (nodes will eventually converge)
    EventualConsistency,
    /// Weak consistency (no guarantees about convergence time)
    WeakConsistency,
    /// Causal consistency (causally related operations are seen in order)
    CausalConsistency,
}

/// Performance thresholds for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Minimum required throughput
    pub min_throughput: f64,
    /// Maximum CPU utilization
    pub max_cpu_utilization: f64,
    /// Maximum memory utilization
    pub max_memory_utilization: f64,
    /// Network performance requirements
    pub network_requirements: NetworkPerformanceRequirements,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_millis(100),
            min_throughput: 1000.0,
            max_cpu_utilization: 0.8,
            max_memory_utilization: 0.8,
            network_requirements: NetworkPerformanceRequirements::default(),
        }
    }
}

/// Network performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformanceRequirements {
    /// Minimum bandwidth (bps)
    pub min_bandwidth: f64,
    /// Maximum acceptable packet loss (0.0-1.0)
    pub max_packet_loss: f64,
    /// Maximum acceptable jitter
    pub max_jitter: Duration,
    /// Required network reliability (0.0-1.0)
    pub min_reliability: f64,
}

impl Default for NetworkPerformanceRequirements {
    fn default() -> Self {
        Self {
            min_bandwidth: 1_000_000.0, // 1 Mbps
            max_packet_loss: 0.01,       // 1%
            max_jitter: Duration::from_millis(10),
            min_reliability: 0.99,       // 99%
        }
    }
}

/// Fault tolerance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceSettings {
    /// Maximum tolerable node failures
    pub max_node_failures: usize,
    /// Failure detection timeout
    pub failure_detection_timeout: Duration,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Redundancy configuration
    pub redundancy: RedundancyConfig,
}

impl Default for FaultToleranceSettings {
    fn default() -> Self {
        Self {
            max_node_failures: 1,
            failure_detection_timeout: Duration::from_secs(5),
            recovery_strategy: RecoveryStrategy::AutomaticRecovery,
            redundancy: RedundancyConfig::default(),
        }
    }
}

/// Recovery strategies for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Automatic recovery without intervention
    AutomaticRecovery,
    /// Manual recovery requiring intervention
    ManualRecovery,
    /// Graceful degradation with reduced functionality
    GracefulDegradation,
    /// Failover to backup systems
    Failover,
    /// Restart coordination from checkpoint
    CheckpointRestart,
}

/// Redundancy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyConfig {
    /// Replication factor
    pub replication_factor: usize,
    /// Backup node selection strategy
    pub backup_selection: BackupSelectionStrategy,
    /// Synchronization mode for backups
    pub sync_mode: BackupSyncMode,
}

impl Default for RedundancyConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            backup_selection: BackupSelectionStrategy::PerformanceBased,
            sync_mode: BackupSyncMode::Synchronous,
        }
    }
}

/// Backup node selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupSelectionStrategy {
    /// Random selection
    Random,
    /// Performance-based selection
    PerformanceBased,
    /// Geographical distribution
    GeographicalDistribution,
    /// Load-based selection
    LoadBased,
    /// Custom selection algorithm
    Custom(String),
}

/// Backup synchronization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupSyncMode {
    /// Synchronous replication
    Synchronous,
    /// Asynchronous replication
    Asynchronous,
    /// Semi-synchronous replication
    SemiSynchronous,
    /// Lazy replication
    Lazy,
}

/// Escalation policies for hierarchical coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationPolicy {
    /// Escalate on timeout
    TimeoutEscalation {
        timeout: Duration,
        target_level: usize,
    },
    /// Escalate on failure count
    FailureEscalation {
        failure_threshold: usize,
        target_level: usize,
    },
    /// Escalate on performance degradation
    PerformanceEscalation {
        performance_threshold: f64,
        target_level: usize,
    },
    /// Manual escalation trigger
    ManualEscalation {
        trigger_condition: String,
        target_level: usize,
    },
}

/// Adaptive coordination criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCriteria {
    /// Workload-based criteria
    pub workload_criteria: WorkloadCriteria,
    /// Network condition criteria
    pub network_criteria: NetworkCriteria,
    /// Performance criteria
    pub performance_criteria: PerformanceCriteria,
    /// Adaptation interval
    pub adaptation_interval: Duration,
    /// Stability threshold
    pub stability_threshold: f64,
}

impl Default for AdaptiveCriteria {
    fn default() -> Self {
        Self {
            workload_criteria: WorkloadCriteria::default(),
            network_criteria: NetworkCriteria::default(),
            performance_criteria: PerformanceCriteria::default(),
            adaptation_interval: Duration::from_secs(60),
            stability_threshold: 0.8,
        }
    }
}

/// Workload-based criteria for adaptive coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadCriteria {
    /// High load threshold (triggers distributed strategy)
    pub high_load_threshold: f64,
    /// Low load threshold (triggers centralized strategy)
    pub low_load_threshold: f64,
    /// Load measurement window
    pub measurement_window: Duration,
    /// Load metrics to consider
    pub load_metrics: Vec<String>,
}

impl Default for WorkloadCriteria {
    fn default() -> Self {
        Self {
            high_load_threshold: 0.8,
            low_load_threshold: 0.3,
            measurement_window: Duration::from_minutes(5),
            load_metrics: vec![
                "cpu_utilization".to_string(),
                "memory_utilization".to_string(),
                "network_utilization".to_string(),
            ],
        }
    }
}

/// Network condition criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCriteria {
    /// High latency threshold
    pub high_latency_threshold: Duration,
    /// Low bandwidth threshold
    pub low_bandwidth_threshold: f64,
    /// Packet loss threshold
    pub packet_loss_threshold: f64,
    /// Network stability measurement window
    pub stability_window: Duration,
}

impl Default for NetworkCriteria {
    fn default() -> Self {
        Self {
            high_latency_threshold: Duration::from_millis(50),
            low_bandwidth_threshold: 10_000_000.0, // 10 Mbps
            packet_loss_threshold: 0.05,           // 5%
            stability_window: Duration::from_minutes(2),
        }
    }
}

/// Performance criteria for adaptive coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCriteria {
    /// Coordination efficiency threshold
    pub efficiency_threshold: f64,
    /// Response time threshold
    pub response_time_threshold: Duration,
    /// Success rate threshold
    pub success_rate_threshold: f64,
    /// Resource utilization threshold
    pub resource_utilization_threshold: f64,
}

impl Default for PerformanceCriteria {
    fn default() -> Self {
        Self {
            efficiency_threshold: 0.7,
            response_time_threshold: Duration::from_secs(5),
            success_rate_threshold: 0.95,
            resource_utilization_threshold: 0.8,
        }
    }
}

/// Selection criteria for hybrid coordination strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Strategy selection algorithm
    pub algorithm: StrategySelectionAlgorithm,
    /// Selection parameters
    pub parameters: HashMap<String, f64>,
    /// Evaluation metrics
    pub evaluation_metrics: Vec<String>,
    /// Selection interval
    pub selection_interval: Duration,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            algorithm: StrategySelectionAlgorithm::PerformanceBased,
            parameters: HashMap::new(),
            evaluation_metrics: vec![
                "latency".to_string(),
                "throughput".to_string(),
                "success_rate".to_string(),
                "resource_efficiency".to_string(),
            ],
            selection_interval: Duration::from_minutes(1),
        }
    }
}

/// Strategy selection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategySelectionAlgorithm {
    /// Performance-based selection
    PerformanceBased,
    /// Load-based selection
    LoadBased,
    /// Round-robin selection
    RoundRobin,
    /// Random selection
    Random,
    /// Machine learning-based selection
    MachineLearning(String),
    /// Rule-based selection
    RuleBased(Vec<SelectionRule>),
}

/// Selection rules for rule-based strategy selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRule {
    /// Rule condition
    pub condition: RuleCondition,
    /// Strategy to select if condition is met
    pub strategy_index: usize,
    /// Rule priority
    pub priority: u32,
}

/// Rule conditions for strategy selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    /// Load threshold condition
    LoadThreshold {
        metric: String,
        threshold: f64,
        operator: ComparisonOperator,
    },
    /// Time-based condition
    TimeBased {
        time_window: Duration,
        condition_type: TimeConditionType,
    },
    /// Event-based condition
    EventBased {
        event_type: String,
        event_count: usize,
        time_window: Duration,
    },
    /// Custom condition
    Custom(String),
}

/// Comparison operators for rule conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotEqual,
}

/// Time-based condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeConditionType {
    /// Peak hours
    PeakHours,
    /// Off-peak hours
    OffPeakHours,
    /// Scheduled maintenance window
    MaintenanceWindow,
    /// Custom time condition
    Custom(String),
}

/// Strategy executor for managing coordination strategies
#[derive(Debug)]
pub struct StrategyExecutor {
    /// Current active strategy
    pub current_strategy: CoordinationStrategy,
    /// Strategy performance metrics
    pub strategy_metrics: HashMap<String, StrategyMetrics>,
    /// Strategy adaptation settings
    pub adaptation_settings: StrategyAdaptationSettings,
    /// Last strategy evaluation
    pub last_evaluation: Instant,
}

impl StrategyExecutor {
    /// Create new strategy executor
    pub fn new(initial_strategy: CoordinationStrategy) -> Self {
        Self {
            current_strategy: initial_strategy,
            strategy_metrics: HashMap::new(),
            adaptation_settings: StrategyAdaptationSettings::default(),
            last_evaluation: Instant::now(),
        }
    }

    /// Execute coordination with current strategy
    pub fn execute_coordination(&mut self, session_id: CoordinationSessionId, participants: Vec<DeviceId>) -> Result<CoordinationResult> {
        let start_time = Instant::now();
        let result = self.execute_strategy(&self.current_strategy.clone(), session_id, participants)?;
        let execution_time = start_time.elapsed();

        self.update_strategy_metrics(execution_time, &result)?;
        self.check_adaptation_criteria()?;

        Ok(result)
    }

    /// Switch to a different coordination strategy
    pub fn switch_strategy(&mut self, new_strategy: CoordinationStrategy) -> Result<()> {
        self.current_strategy = new_strategy;
        self.last_evaluation = Instant::now();
        Ok(())
    }

    /// Evaluate strategy performance
    pub fn evaluate_strategy_performance(&self) -> StrategyPerformanceEvaluation {
        let metrics = self.strategy_metrics.get(&self.strategy_name()).cloned()
            .unwrap_or_else(StrategyMetrics::default);

        StrategyPerformanceEvaluation {
            strategy_name: self.strategy_name(),
            performance_score: self.calculate_performance_score(&metrics),
            metrics,
            evaluation_timestamp: Instant::now(),
        }
    }

    fn execute_strategy(&self, strategy: &CoordinationStrategy, session_id: CoordinationSessionId, participants: Vec<DeviceId>) -> Result<CoordinationResult> {
        match strategy {
            CoordinationStrategy::Centralized { leader_election } => {
                self.execute_centralized_coordination(session_id, participants, leader_election)
            }
            CoordinationStrategy::Distributed { consensus } => {
                self.execute_distributed_coordination(session_id, participants, consensus)
            }
            CoordinationStrategy::Hierarchical { hierarchy } => {
                self.execute_hierarchical_coordination(session_id, participants, hierarchy)
            }
            CoordinationStrategy::Hybrid { strategies, selection_criteria } => {
                self.execute_hybrid_coordination(session_id, participants, strategies, selection_criteria)
            }
            CoordinationStrategy::Adaptive { criteria, fallback_strategy } => {
                self.execute_adaptive_coordination(session_id, participants, criteria, fallback_strategy)
            }
        }
    }

    fn execute_centralized_coordination(&self, session_id: CoordinationSessionId, participants: Vec<DeviceId>, config: &LeaderElectionConfig) -> Result<CoordinationResult> {
        // Implementation would execute centralized coordination
        Ok(CoordinationResult::success(session_id, participants.len()))
    }

    fn execute_distributed_coordination(&self, session_id: CoordinationSessionId, participants: Vec<DeviceId>, consensus: &ConsensusAlgorithm) -> Result<CoordinationResult> {
        // Implementation would execute distributed coordination using consensus
        Ok(CoordinationResult::success(session_id, participants.len()))
    }

    fn execute_hierarchical_coordination(&self, session_id: CoordinationSessionId, participants: Vec<DeviceId>, hierarchy: &HierarchyConfig) -> Result<CoordinationResult> {
        // Implementation would execute hierarchical coordination
        Ok(CoordinationResult::success(session_id, participants.len()))
    }

    fn execute_hybrid_coordination(&self, session_id: CoordinationSessionId, participants: Vec<DeviceId>, strategies: &[CoordinationStrategy], criteria: &SelectionCriteria) -> Result<CoordinationResult> {
        // Implementation would select and execute appropriate strategy
        Ok(CoordinationResult::success(session_id, participants.len()))
    }

    fn execute_adaptive_coordination(&self, session_id: CoordinationSessionId, participants: Vec<DeviceId>, criteria: &AdaptiveCriteria, fallback: &CoordinationStrategy) -> Result<CoordinationResult> {
        // Implementation would adaptively choose coordination approach
        Ok(CoordinationResult::success(session_id, participants.len()))
    }

    fn strategy_name(&self) -> String {
        match &self.current_strategy {
            CoordinationStrategy::Centralized { .. } => "Centralized".to_string(),
            CoordinationStrategy::Distributed { .. } => "Distributed".to_string(),
            CoordinationStrategy::Hierarchical { .. } => "Hierarchical".to_string(),
            CoordinationStrategy::Hybrid { .. } => "Hybrid".to_string(),
            CoordinationStrategy::Adaptive { .. } => "Adaptive".to_string(),
        }
    }

    fn update_strategy_metrics(&mut self, execution_time: Duration, result: &CoordinationResult) -> Result<()> {
        let strategy_name = self.strategy_name();
        let metrics = self.strategy_metrics.entry(strategy_name).or_default();

        metrics.update(execution_time, result.success);
        Ok(())
    }

    fn check_adaptation_criteria(&mut self) -> Result<()> {
        if self.last_evaluation.elapsed() >= self.adaptation_settings.evaluation_interval {
            self.evaluate_and_adapt()?;
            self.last_evaluation = Instant::now();
        }
        Ok(())
    }

    fn evaluate_and_adapt(&mut self) -> Result<()> {
        // Implementation would evaluate current performance and potentially adapt strategy
        Ok(())
    }

    fn calculate_performance_score(&self, metrics: &StrategyMetrics) -> f64 {
        // Simple performance score calculation based on success rate and average execution time
        let success_weight = 0.7;
        let time_weight = 0.3;

        let success_score = metrics.success_rate;
        let time_score = if metrics.average_execution_time.as_millis() == 0 {
            1.0
        } else {
            1.0 / (metrics.average_execution_time.as_millis() as f64 / 1000.0)
        };

        success_weight * success_score + time_weight * time_score.min(1.0)
    }
}

/// Strategy metrics for performance tracking
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Total executions
    pub total_executions: usize,
    /// Successful executions
    pub successful_executions: usize,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Last execution timestamp
    pub last_execution: Instant,
}

impl Default for StrategyMetrics {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            success_rate: 1.0,
            average_execution_time: Duration::from_millis(0),
            min_execution_time: Duration::from_millis(0),
            max_execution_time: Duration::from_millis(0),
            last_execution: Instant::now(),
        }
    }
}

impl StrategyMetrics {
    /// Update metrics with new execution data
    pub fn update(&mut self, execution_time: Duration, success: bool) {
        self.total_executions += 1;
        if success {
            self.successful_executions += 1;
        }

        self.success_rate = self.successful_executions as f64 / self.total_executions as f64;

        if self.total_executions == 1 {
            self.average_execution_time = execution_time;
            self.min_execution_time = execution_time;
            self.max_execution_time = execution_time;
        } else {
            // Update running average
            let total_time = self.average_execution_time * (self.total_executions - 1) as u32 + execution_time;
            self.average_execution_time = total_time / self.total_executions as u32;

            if execution_time < self.min_execution_time {
                self.min_execution_time = execution_time;
            }
            if execution_time > self.max_execution_time {
                self.max_execution_time = execution_time;
            }
        }

        self.last_execution = Instant::now();
    }
}

/// Strategy adaptation settings
#[derive(Debug, Clone)]
pub struct StrategyAdaptationSettings {
    /// Enable automatic adaptation
    pub auto_adaptation: bool,
    /// Evaluation interval
    pub evaluation_interval: Duration,
    /// Performance threshold for switching
    pub performance_threshold: f64,
    /// Minimum adaptation interval
    pub min_adaptation_interval: Duration,
}

impl Default for StrategyAdaptationSettings {
    fn default() -> Self {
        Self {
            auto_adaptation: true,
            evaluation_interval: Duration::from_minutes(5),
            performance_threshold: 0.8,
            min_adaptation_interval: Duration::from_minutes(1),
        }
    }
}

/// Coordination result
#[derive(Debug, Clone)]
pub struct CoordinationResult {
    /// Session ID
    pub session_id: CoordinationSessionId,
    /// Success status
    pub success: bool,
    /// Number of participants
    pub participants: usize,
    /// Execution time
    pub execution_time: Duration,
    /// Result data
    pub data: HashMap<String, String>,
    /// Error message if failed
    pub error_message: Option<String>,
}

impl CoordinationResult {
    /// Create successful coordination result
    pub fn success(session_id: CoordinationSessionId, participants: usize) -> Self {
        Self {
            session_id,
            success: true,
            participants,
            execution_time: Duration::from_millis(0),
            data: HashMap::new(),
            error_message: None,
        }
    }

    /// Create failed coordination result
    pub fn failure(session_id: CoordinationSessionId, participants: usize, error: String) -> Self {
        Self {
            session_id,
            success: false,
            participants,
            execution_time: Duration::from_millis(0),
            data: HashMap::new(),
            error_message: Some(error),
        }
    }
}

/// Strategy performance evaluation
#[derive(Debug, Clone)]
pub struct StrategyPerformanceEvaluation {
    /// Strategy name
    pub strategy_name: String,
    /// Performance score (0.0-1.0)
    pub performance_score: f64,
    /// Detailed metrics
    pub metrics: StrategyMetrics,
    /// Evaluation timestamp
    pub evaluation_timestamp: Instant,
}