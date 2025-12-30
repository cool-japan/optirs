// Performance Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Deadlock performance types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AllocationStrategy {
    FirstFit,
    #[default]
    BestFit,
    WorstFit,
}

#[derive(Debug, Clone, Default)]
pub struct AutoTuning {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CachingStrategies {
    pub computation: ComputationCaching,
    pub result: ResultCaching,
}

#[derive(Debug, Clone, Default)]
pub struct ComputationCaching {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CpuLimits {
    pub max_cores: u32,
}

#[derive(Debug, Clone, Default)]
pub struct CpuManagement {
    pub limits: CpuLimits,
}

#[derive(Debug, Clone, Default)]
pub struct CpuMonitoring {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DeadlockPerformanceConfig {
    pub optimization_enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DeadlockStatistics {
    pub detection_count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DetectionTimePercentiles {
    pub p50_ms: f64,
    pub p99_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DetectionTimeStatistics {
    pub percentiles: DetectionTimePercentiles,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EvictionPolicy {
    #[default]
    LRU,
    LFU,
    FIFO,
}

#[derive(Debug, Clone, Default)]
pub struct GarbageCollection {
    pub strategy: GcStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum GcStrategy {
    #[default]
    Concurrent,
    Stop,
    Incremental,
}

#[derive(Debug, Clone, Default)]
pub struct HealthChecking {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct HorizontalScaling {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct IoLimits {
    pub max_iops: u64,
}

#[derive(Debug, Clone, Default)]
pub struct IoManagement {
    pub limits: IoLimits,
}

#[derive(Debug, Clone, Default)]
pub struct IoMonitoring {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct IoOptimization {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct IoScheduling {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct LoadBalancing {
    pub strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LoadBalancingMetric {
    #[default]
    CPU,
    Memory,
    IO,
}

#[derive(Debug, Clone, Default)]
pub struct LoadBalancingMonitoring {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    LeastConnections,
    Random,
}

#[derive(Debug, Clone, Default)]
pub struct LoadBalancingThresholds {
    pub high: f64,
    pub low: f64,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryLimits {
    pub max_gb: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryManagement {
    pub limits: MemoryLimits,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryMonitoring {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PerformanceMetric {
    #[default]
    Throughput,
    Latency,
    Utilization,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceMonitoring {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceOptimization {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceTargets {
    pub target_throughput: f64,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceThresholds {
    pub warning: f64,
    pub critical: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ResultCaching {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ScalabilityConfiguration {
    pub horizontal: HorizontalScaling,
    pub vertical: VerticalScaling,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ScalingTrigger {
    #[default]
    Load,
    Time,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SpawningStrategy {
    Eager,
    Lazy,
    #[default]
    OnDemand,
}

#[derive(Debug, Clone, Default)]
pub struct SystemImpactStatistics {
    pub cpu_impact: f64,
    pub memory_impact: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ThreadManagement {
    pub spawning: SpawningStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum UpdateStrategy {
    #[default]
    RollingUpdate,
    BlueGreen,
    Canary,
}

#[derive(Debug, Clone, Default)]
pub struct VerticalLimits {
    pub max_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum VerticalMetric {
    #[default]
    CPU,
    Memory,
}

#[derive(Debug, Clone, Default)]
pub struct VerticalPerformanceMonitoring {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct VerticalScaling {
    pub enabled: bool,
}

// Performance analysis types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AlertSeverity {
    #[default]
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AlertType {
    #[default]
    Performance,
    Resource,
    Error,
}

#[derive(Debug, Clone, Default)]
pub struct DevicePerformanceMetrics {
    pub device_id: DeviceId,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EffortLevel {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Debug, Clone, Default)]
pub struct OptimizationRecommendation {
    pub priority: RecommendationPriority,
    pub type_: RecommendationType,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceAlert {
    pub severity: AlertSeverity,
    pub type_: AlertType,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceBenchmark {
    pub baseline: f64,
    pub target: f64,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceTrend {
    pub direction: TrendDirection,
}

#[derive(Debug, Clone, Default)]
pub struct PodPerformanceAnalyzer {
    pub metrics: PodPerformanceMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct PodPerformanceMetrics {
    pub aggregated_tflops: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecommendationPriority {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecommendationType {
    #[default]
    Configuration,
    Scaling,
    Optimization,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TrendDirection {
    Improving,
    #[default]
    Stable,
    Declining,
}
