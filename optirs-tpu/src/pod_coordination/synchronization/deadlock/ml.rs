// Deadlock ML Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

pub use super::algorithms::CombinationStrategy;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EnsembleMethod {
    #[default]
    Bagging,
    Boosting,
    Stacking,
}

#[derive(Debug, Clone, Default)]
pub struct FeatureExtraction {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GraphFeature {
    pub node_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MLModelType {
    #[default]
    DecisionTree,
    NeuralNetwork,
    SVM,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceFeature {
    pub resource_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TemporalFeature {
    pub timestamp_ms: u64,
}
