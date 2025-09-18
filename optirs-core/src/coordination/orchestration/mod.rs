// Orchestration for optimization coordination
//
// This module provides orchestration capabilities for complex optimization
// workflows, including pipeline management, experiment lifecycle coordination,
// and checkpoint/recovery systems.

#[allow(dead_code)]
pub mod checkpoint_manager;
pub mod experiment_manager;
pub mod pipeline_orchestrator;

// Re-export key types
pub use pipeline_orchestrator::{
    OptimizationPipeline, PipelineConfiguration, PipelineExecution, PipelineOrchestrator,
    PipelineStage, StageResult, ResourceLimits, TimeoutSettings, MonitoringConfiguration,
    OrchestratorConfiguration,
};

pub use crate::coordination::monitoring::performance_tracking::{
    StorageConfiguration, AlertConfiguration,
};

pub use experiment_manager::{
    Experiment, ExperimentConfiguration, ExperimentExecution, ExperimentManager, ExperimentResult,
    ExperimentStatus,
};

pub use checkpoint_manager::{
    Checkpoint, CheckpointConfiguration, CheckpointManager, CheckpointMetadata, RecoveryManager,
    RecoveryStrategy,
};
