// Deadlock Detection and Prevention Module
//
// This module provides comprehensive deadlock detection, prevention, and recovery
// mechanisms for distributed TPU synchronization. It includes graph-based algorithms,
// machine learning approaches, performance monitoring, and recovery strategies.
//
// # Components
//
// - **Types**: Core data structures and configuration types
// - **Algorithms**: Detection algorithms and optimization strategies
// - **Prevention**: Prevention strategies and policies
// - **Graph**: Dependency graph management and analysis
// - **Performance**: Performance monitoring and statistics
// - **Recovery**: Recovery coordination and execution
// - **ML**: Machine learning components for prediction

pub mod algorithms;
pub mod graph;
pub mod ml;
pub mod performance;
pub mod prevention;
pub mod recovery;
pub mod types;

// Type aliases for resource and transaction IDs
pub type ResourceId = u64;
pub type TransactionId = u64;

// Re-export core types
pub use types::{
    AdaptiveSensitivity, AdvancedDeadlockConfig, AdvancedDiagnostics, DeadlockDetectionConfig,
    DeadlockDetector, DeadlockPerformanceConfig, DeadlockSensitivity, DeliveryMethod,
    DetectionEvent, DetectionEventType, DetectionState, DetectionStatus, ExportFormat,
    IntegrationSettings, NotificationConfig, NotificationType, PerformanceOptimization,
    ResourceLimits, SensitivityMetric,
};

// Re-export algorithm types
pub use algorithms::{
    BackoffStrategy, CacheOptimization, CachePolicy, CombinationStrategy, ConflictResolution,
    CycleDetectionMethod, DeadlockCriteria, DeadlockDetectionAlgorithm, ErrorHandling,
    GraphOptimization, GraphReductionMethod, ParallelProcessing, PrefetchingStrategy,
    PropagationStrategy, ResourceAllocationMethod, ResponseHandling, RetryPolicy, SafeStateMethod,
    SynchronizationMethod, TimestampOrdering, WorkDistribution,
};

// Re-export prevention types
pub use prevention::{
    AllocationPolicy, AvoidanceStrategy, BankersAlgorithmConfig, CircularWaitPrevention,
    ConservativeStrategy, DeadlockPrevention, DeadlockPreventionSystem, HoldAndWaitPrevention,
    MutualExclusionPrevention, NoPreemptionPrevention, OptimisticStrategy, OrderingStrategy,
    PreemptionPolicy, PreemptionStrategy, PreventionPolicy, PreventionStatistics, ResourceOrdering,
    TimeoutStrategy, ValidationStatistics, WoundWaitStrategy,
};

// Re-export machine learning types
pub use ml::{
    CombinationStrategy as MLCombinationStrategy, EnsembleMethod, FeatureExtraction, GraphFeature,
    MLModelType, ResourceFeature, TemporalFeature,
};

// Re-export graph types
pub use graph::{
    ChangeType, DependencyEdge, DependencyGraph, EdgeMetadata, EdgeType, GraphChange, GraphHistory,
    GraphMetadata, GraphNode, GraphOptimizationState, GraphProperties, GraphSnapshot,
    GraphStatistics, NodeMetadata, NodeState, NodeType, OptimizationOperation, OptimizationRecord,
    OptimizationStatistics, PerformanceImpact,
};

// Re-export performance types
pub use performance::{
    AllocationStrategy, AutoTuning, CachingStrategies, ComputationCaching, CpuLimits,
    CpuManagement, CpuMonitoring, DeadlockPerformanceConfig as PerformanceConfig,
    DeadlockStatistics, DetectionTimePercentiles, DetectionTimeStatistics, EvictionPolicy,
    GarbageCollection, GcStrategy, HealthChecking, HorizontalScaling, IoLimits, IoManagement,
    IoMonitoring, IoOptimization, IoScheduling, LoadBalancing, LoadBalancingMetric,
    LoadBalancingMonitoring, LoadBalancingStrategy, LoadBalancingThresholds, MemoryLimits,
    MemoryManagement, MemoryMonitoring, PerformanceMetric, PerformanceMonitoring,
    PerformanceOptimization as PerfOptimization, PerformanceTargets, PerformanceThresholds,
    ResultCaching, ScalabilityConfiguration, ScalingTrigger, SpawningStrategy,
    SystemImpactStatistics, ThreadManagement, UpdateStrategy, VerticalLimits, VerticalMetric,
    VerticalPerformanceMonitoring, VerticalScaling,
};

// Re-export recovery types
pub use recovery::{
    ActiveRecovery, CoordinatorSelection, CoordinatorState, DeadlockRecovery,
    DeadlockRecoverySystem, DeadlockSeverity, DetectedDeadlock, DistributedRecovery,
    DistributedRecoveryStrategy, ExecutionContext, ExecutionRecord, ExecutorCapabilities,
    ExecutorPerformance, PhaseRecord, PhaseResult, RecoveryAction, RecoveryConstraint,
    RecoveryConstraintType, RecoveryCoordination, RecoveryCoordinator, RecoveryExecutor,
    RecoveryExecutorStrategy, RecoveryObjective, RecoveryOptimization,
    RecoveryOptimizationAlgorithm, RecoveryPhase, RecoveryProgress, RecoveryRequest,
    RecoveryResult, RecoveryStatistics, RecoveryStrategy, RecoveryVerification,
    RecoveryVerificationMethod, RollbackMechanism, SelectionCriterion, StateSynchronization,
    StrategyStatistics, SynchronizationProtocol, SystemHealth, SystemState,
    VerificationSuccessCriteria, VictimSelection, VictimSelectionAlgorithm,
};

// Convenience type aliases for backward compatibility
pub type DeadlockConfig = DeadlockDetectionConfig;
pub type Statistics = DeadlockStatistics;
// PerformanceConfig already imported above as alias
pub type RecoveryConfig = DeadlockRecovery;

/// Create a new deadlock detector with default configuration
pub fn create_detector() -> crate::error::Result<DeadlockDetector> {
    DeadlockDetector::new()
}

/// Create a new deadlock detector with custom configuration
pub fn create_detector_with_config(
    config: DeadlockDetectionConfig,
) -> crate::error::Result<DeadlockDetector> {
    let mut detector = DeadlockDetector::new()?;
    detector.config = config;
    Ok(detector)
}

/// Create a new dependency graph
pub fn create_dependency_graph() -> crate::error::Result<DependencyGraph> {
    Ok(DependencyGraph::new())
}

/// Create a new recovery system
pub fn create_recovery_system(
    config: DeadlockRecovery,
) -> crate::error::Result<DeadlockRecoverySystem> {
    let mut system = DeadlockRecoverySystem::new();
    system.recovery = config;
    Ok(system)
}

// Implementation from the original file that needs to be preserved
impl DeadlockDetector {
    /// Create a new deadlock detector
    pub fn new() -> crate::error::Result<Self> {
        Ok(Self {
            config: DeadlockDetectionConfig::default(),
            dependency_graph: DependencyGraph::new(),
            detection_state: DetectionState {
                status: DetectionStatus::Idle,
                last_detection: std::time::Instant::now(),
                active_deadlocks: 0,
                history: Vec::new(),
            },
            statistics: DeadlockStatistics::default(),
            prevention_system: prevention::DeadlockPreventionSystem::new(),
            recovery_system: recovery::DeadlockRecoverySystem::new(),
        })
    }

    /// Detect deadlocks in the system
    pub fn detect_deadlocks(&mut self) -> crate::error::Result<Vec<String>> {
        self.detection_state.status = DetectionStatus::Running;
        self.detection_state.last_detection = std::time::Instant::now();

        // Use the graph's cycle detection
        if self.dependency_graph.has_cycle() {
            self.detection_state.status = DetectionStatus::DeadlockDetected;
            self.detection_state.active_deadlocks += 1;
            self.statistics.detection_count += 1;

            // Return detected deadlock IDs (simplified)
            Ok(vec!["deadlock_1".to_string()])
        } else {
            self.detection_state.status = DetectionStatus::Idle;
            Ok(Vec::new())
        }
    }

    /// Add a resource dependency
    pub fn add_dependency(&mut self, source: String, target: String) -> crate::error::Result<()> {
        use graph::{DependencyEdge, EdgeMetadata, EdgeType};
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Convert strings to u64 IDs using hash
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let source_id = hasher.finish();

        let mut hasher = DefaultHasher::new();
        target.hash(&mut hasher);
        let target_id = hasher.finish();

        // Add nodes if they don't exist
        if !self
            .dependency_graph
            .nodes
            .iter()
            .any(|n| n.id == source_id)
        {
            let node = graph::GraphNode {
                id: source_id,
                node_type: graph::NodeType::Process,
                state: graph::NodeState::Active,
                metadata: graph::GraphMetadata::default(),
                timestamp: std::time::Instant::now(),
            };
            self.dependency_graph.add_node(node);
        }

        if !self
            .dependency_graph
            .nodes
            .iter()
            .any(|n| n.id == target_id)
        {
            let node = graph::GraphNode {
                id: target_id,
                node_type: graph::NodeType::Resource,
                state: graph::NodeState::Active,
                metadata: graph::GraphMetadata::default(),
                timestamp: std::time::Instant::now(),
            };
            self.dependency_graph.add_node(node);
        }

        // Add the edge
        let edge = DependencyEdge {
            from: source_id,
            to: target_id,
            source: source_id,
            target: target_id,
            edge_type: EdgeType::WaitsFor,
            weight: 1.0,
            timestamp: std::time::Instant::now(),
            metadata: EdgeMetadata::default(),
        };

        self.dependency_graph.add_edge(edge);
        Ok(())
    }

    /// Remove a resource dependency
    pub fn remove_dependency(&mut self, source: &str, target: &str) -> crate::error::Result<()> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Convert strings to u64 IDs using hash
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let source_id = hasher.finish();

        let mut hasher = DefaultHasher::new();
        target.hash(&mut hasher);
        let target_id = hasher.finish();

        // Remove edges that match the source and target
        self.dependency_graph
            .edges
            .retain(|edge| !(edge.source == source_id && edge.target == target_id));

        Ok(())
    }

    /// Get current detection statistics
    pub fn get_statistics(&self) -> &DeadlockStatistics {
        &self.statistics
    }

    /// Update detector configuration
    pub fn update_config(&mut self, config: DeadlockDetectionConfig) {
        self.config = config;
    }
}
