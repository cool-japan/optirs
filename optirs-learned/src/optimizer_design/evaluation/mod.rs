// Architecture evaluation and performance assessment
//
// This module provides functionality for evaluating neural architectures,
// including performance prediction, early stopping, and multi-objective assessment.

use std::collections::HashMap;

use super::architecture::{ArchitectureCandidate, PerformanceMetrics, ResourceUsage};

/// Architecture evaluator
pub struct ArchitectureEvaluator {
    /// Evaluation configuration
    pub config: EvaluationConfig,
    
    /// Performance predictor
    pub predictor: Option<PerformancePredictor>,
    
    /// Evaluation history
    pub evaluation_history: Vec<EvaluationRecord>,
}

/// Evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Evaluation budget (number of full evaluations)
    pub budget: usize,
    
    /// Use performance prediction
    pub use_prediction: bool,
    
    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,
    
    /// Validation strategy
    pub validation: ValidationConfig,
}

/// Performance predictor for fast architecture evaluation
pub struct PerformancePredictor {
    /// Predictor type
    pub predictor_type: PredictorType,
    
    /// Training data
    pub training_data: Vec<(ArchitectureCandidate, PerformanceMetrics)>,
    
    /// Prediction accuracy
    pub accuracy: f64,
}

/// Types of performance predictors
#[derive(Debug, Clone, Copy)]
pub enum PredictorType {
    NeuralNetwork,
    GradientBoosting,
    RandomForest,
    GaussianProcess,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,
    
    /// Patience (epochs without improvement)
    pub patience: usize,
    
    /// Minimum improvement threshold
    pub min_delta: f64,
    
    /// Monitor metric
    pub monitor: String,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Validation strategy
    pub strategy: ValidationStrategy,
    
    /// Cross-validation folds
    pub cv_folds: usize,
    
    /// Validation split ratio
    pub val_split: f64,
}

/// Validation strategies
#[derive(Debug, Clone, Copy)]
pub enum ValidationStrategy {
    HoldOut,
    CrossValidation,
    TimeSeriesSplit,
    StratifiedKFold,
}

/// Evaluation record
#[derive(Debug, Clone)]
pub struct EvaluationRecord {
    /// Architecture ID
    pub architecture_id: String,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
    
    /// Resource usage
    pub resource_usage: ResourceUsage,
    
    /// Evaluation time
    pub evaluation_time: std::time::Duration,
    
    /// Evaluation type (full or predicted)
    pub evaluation_type: EvaluationType,
}

/// Types of evaluations
#[derive(Debug, Clone, Copy)]
pub enum EvaluationType {
    Full,
    Predicted,
    EarlyStopped,
    Cached,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            budget: 1000,
            use_prediction: true,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 10,
                min_delta: 0.001,
                monitor: "loss".to_string(),
            },
            validation: ValidationConfig {
                strategy: ValidationStrategy::HoldOut,
                cv_folds: 5,
                val_split: 0.2,
            },
        }
    }
}