// Gradient Aggregation Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct AggregationConfig {
    pub method: crate::pod_coordination::coordination::config::GradientAggregationMethod,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum AggregationState {
    #[default]
    Idle,
    Collecting,
    Aggregating,
    Broadcasting,
}

#[derive(Debug, Clone, Default)]
pub struct AggregationStatistics {
    pub total_aggregations: u64,
    pub avg_time_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct BufferMetadata {
    pub buffer_id: u64,
    pub size_bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationOptimization {
    pub compression: CompressionSettings,
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CompressionParameters {
    pub ratio: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CompressionSettings {
    pub enabled: bool,
    pub method: QuantizationMethod,
}

#[derive(Debug, Clone, Default)]
pub struct FederatedParams {
    pub num_rounds: u32,
}

#[derive(Debug, Clone, Default)]
pub struct GradientAggregator {
    pub config: AggregationConfig,
    pub state: AggregationState,
}

#[derive(Debug, Clone, Default)]
pub struct GradientBuffer {
    pub metadata: BufferMetadata,
    pub status: GradientBufferStatus,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum GradientBufferStatus {
    #[default]
    Empty,
    Partial,
    Full,
}

#[derive(Debug, Clone, Default)]
pub struct LocalSGDParams {
    pub local_steps: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum QuantizationMethod {
    #[default]
    None,
    Int8,
    Int16,
    Dynamic,
}

#[derive(Debug, Clone, Default)]
pub struct QuantizationSettings {
    pub method: QuantizationMethod,
}

#[derive(Debug, Clone, Default)]
pub struct SCAFFOLDParams {
    pub control_variates: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SparsificationMethod {
    #[default]
    TopK,
    Threshold,
    Random,
}

#[derive(Debug, Clone, Default)]
pub struct GradientAggregationStatistics {
    pub total_gradients: u64,
    pub avg_aggregation_time_ms: f64,
}
