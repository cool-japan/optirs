//! Error handling for OptiRS Learned

use thiserror::Error;

/// Result type alias for OptiRS Learned operations
pub type Result<T> = std::result::Result<T, OptimError>;

/// Error types for OptiRS Learned operations
#[derive(Error, Debug)]
pub enum OptimError {
    /// Insufficient data for the operation
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Analytics operation error
    #[error("Analytics error: {0}")]
    AnalyticsError(String),

    /// Invalid state error
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Invalid configuration error
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Network architecture error
    #[error("Network architecture error: {0}")]
    NetworkError(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Generic error for other cases
    #[error("Error: {0}")]
    Other(String),
}

/// Convert from other error types
impl From<serde_json::Error> for OptimError {
    fn from(error: serde_json::Error) -> Self {
        OptimError::SerializationError(error.to_string())
    }
}

impl From<std::io::Error> for OptimError {
    fn from(error: std::io::Error) -> Self {
        OptimError::Other(error.to_string())
    }
}
