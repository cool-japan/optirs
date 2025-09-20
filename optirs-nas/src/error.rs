//! Error types for OptiRS NAS module

use std::error::Error;
use std::fmt;

/// Error type for NAS operations
#[derive(Debug)]
pub enum OptimError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Search space error
    SearchSpaceError(String),
    /// Evaluation error
    EvaluationError(String),
    /// Architecture encoding error
    ArchitectureError(String),
    /// Optimization error
    OptimizationError(String),
    /// Resource constraint violation
    ResourceConstraintViolation(String),
    /// Convergence failure
    ConvergenceFailure(String),
    /// Invalid parameter
    InvalidParameter(String),
    /// Configuration error
    ConfigurationError(String),
    /// Resource limit exceeded
    ResourceLimitExceeded(String),
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            OptimError::SearchSpaceError(msg) => write!(f, "Search space error: {}", msg),
            OptimError::EvaluationError(msg) => write!(f, "Evaluation error: {}", msg),
            OptimError::ArchitectureError(msg) => write!(f, "Architecture error: {}", msg),
            OptimError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            OptimError::ResourceConstraintViolation(msg) => {
                write!(f, "Resource constraint violation: {}", msg)
            }
            OptimError::ConvergenceFailure(msg) => write!(f, "Convergence failure: {}", msg),
            OptimError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            OptimError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            OptimError::ResourceLimitExceeded(msg) => write!(f, "Resource limit exceeded: {}", msg),
        }
    }
}

impl Error for OptimError {}

/// Result type for NAS operations
pub type Result<T> = std::result::Result<T, OptimError>;
