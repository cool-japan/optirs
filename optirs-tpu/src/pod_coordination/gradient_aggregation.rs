// Gradient Aggregation Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct AggregationConfig {
    pub method: crate::pod_coordination::coordination::config::GradientAggregationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationState {
    Idle,
    Collecting,
    Aggregating,
    Broadcasting,
}

impl Default for AggregationState {
    fn default() -> Self {
        Self::Idle
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientBufferStatus {
    Empty,
    Partial,
    Full,
}

impl Default for GradientBufferStatus {
    fn default() -> Self {
        Self::Empty
    }
}

#[derive(Debug, Clone, Default)]
pub struct LocalSGDParams {
    pub local_steps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMethod {
    None,
    Int8,
    Int16,
    Dynamic,
}

impl Default for QuantizationMethod {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Default)]
pub struct QuantizationSettings {
    pub method: QuantizationMethod,
}

#[derive(Debug, Clone, Default)]
pub struct SCAFFOLDParams {
    pub control_variates: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparsificationMethod {
    TopK,
    Threshold,
    Random,
}

impl Default for SparsificationMethod {
    fn default() -> Self {
        Self::TopK
    }
}

#[derive(Debug, Clone, Default)]
pub struct GradientAggregationStatistics {
    pub total_gradients: u64,
    pub avg_aggregation_time_ms: f64,
}
