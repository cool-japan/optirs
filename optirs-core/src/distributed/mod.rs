// Distributed optimization support
//
// This module provides support for distributed training including parameter averaging,
// gradient compression, and communication optimization for multi-node/multi-GPU training.

pub mod fedprox;
pub use fedprox::*;

pub mod ring_allreduce;
pub use ring_allreduce::{CollectiveTransport, LocalTransport, ReduceOp, RingAllReduce};

pub mod pipeline_parallel;
pub use pipeline_parallel::{
    OpKind, PipelineConfig, PipelineExecution, PipelineMetrics, PipelineOp, PipelineSchedule,
    PipelineScheduler, StageCost, StagePartitioner, StageRange,
};

pub mod elastic;
pub use elastic::{
    ElasticConfig, ElasticCoordinator, EpochSnapshot, MembershipEvent, RendezvousState, ShardRange,
    WorkerShards, WorkerState,
};

mod averaging;
pub use averaging::{AveragingStrategy, ParameterAverager};

mod parameter_server;
pub use parameter_server::ParameterServer;

mod coordinator;
pub use coordinator::{CommunicationResult, DistributedCoordinator, TrainingStats};

mod compression;
pub use compression::{
    CompressedGradient, CompressionMetadata, CompressionStats, CompressionStrategy,
    GradientCompressor,
};

#[cfg(test)]
mod tests;
