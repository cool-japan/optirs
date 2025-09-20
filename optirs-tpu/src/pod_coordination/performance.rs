// Performance Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Deadlock performance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::BestFit
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

#[derive(Debug, Clone, Default)]
pub struct GarbageCollection {
    pub strategy: GcStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GcStrategy {
    Concurrent,
    Stop,
    Incremental,
}

impl Default for GcStrategy {
    fn default() -> Self {
        Self::Concurrent
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingMetric {
    CPU,
    Memory,
    IO,
}

impl Default for LoadBalancingMetric {
    fn default() -> Self {
        Self::CPU
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadBalancingMonitoring {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    Random,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Throughput,
    Latency,
    Utilization,
}

impl Default for PerformanceMetric {
    fn default() -> Self {
        Self::Throughput
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingTrigger {
    Load,
    Time,
    Manual,
}

impl Default for ScalingTrigger {
    fn default() -> Self {
        Self::Load
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpawningStrategy {
    Eager,
    Lazy,
    OnDemand,
}

impl Default for SpawningStrategy {
    fn default() -> Self {
        Self::OnDemand
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateStrategy {
    RollingUpdate,
    BlueGreen,
    Canary,
}

impl Default for UpdateStrategy {
    fn default() -> Self {
        Self::RollingUpdate
    }
}

#[derive(Debug, Clone, Default)]
pub struct VerticalLimits {
    pub max_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerticalMetric {
    CPU,
    Memory,
}

impl Default for VerticalMetric {
    fn default() -> Self {
        Self::CPU
    }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl Default for AlertSeverity {
    fn default() -> Self {
        Self::Info
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    Performance,
    Resource,
    Error,
}

impl Default for AlertType {
    fn default() -> Self {
        Self::Performance
    }
}

#[derive(Debug, Clone, Default)]
pub struct DevicePerformanceMetrics {
    pub device_id: DeviceId,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
}

impl Default for EffortLevel {
    fn default() -> Self {
        Self::Medium
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for RecommendationPriority {
    fn default() -> Self {
        Self::Medium
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    Configuration,
    Scaling,
    Optimization,
}

impl Default for RecommendationType {
    fn default() -> Self {
        Self::Configuration
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
}

impl Default for TrendDirection {
    fn default() -> Self {
        Self::Stable
    }
}
