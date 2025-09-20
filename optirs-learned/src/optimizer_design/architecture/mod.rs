// Architecture components and specifications for optimizer design
//
// This module contains the core architectural building blocks used in
// neural architecture search for learned optimizers.

pub mod components;
pub mod specifications;
pub mod constraints;
pub mod patterns;

pub use components::{
    LayerType, ActivationType, ConnectionPattern, AttentionType,
    NormalizationType, OptimizerComponent, MemoryType, SkipConnectionType,
    MemoryManagementStrategy, SparsityPattern, MemoryAccessPattern,
};

pub use specifications::{
    ArchitectureSpec, LayerSpec, LayerDimensions, GlobalArchitectureConfig,
    AttentionPattern, SparseAttentionConfig, MemoryManagement,
    SpecializedComponent, ArchitectureCandidate,
};

pub use constraints::{
    SearchConstraints, ComplexityConstraints, HardwareConstraints,
    TargetHardware, ComputeCapability, SpecializedUnit,
};

pub use patterns::{
    ConnectionTopology, ResidualPattern, AttentionConnectivity,
    MemoryAccessStrategy, OptimizationPath,
};