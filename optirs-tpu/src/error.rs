// Error handling for OptiRS TPU module
//
// This module re-exports error types from scirs2_core to maintain
// SciRS2 integration policy compliance.

// Re-export from scirs2_core for SciRS2 compliance
pub use scirs2_core::error::{CoreError as OptimError, Result};

// TPU-specific error types
#[derive(Debug, thiserror::Error)]
pub enum TpuError {
    #[error("TPU coordination error: {0}")]
    Coordination(String),

    #[error("XLA compilation error: {0}")]
    XlaCompilation(String),

    #[error("Pod synchronization error: {0}")]
    PodSynchronization(String),

    #[error("Core error: {0}")]
    Core(#[from] OptimError),
}

pub type TpuResult<T> = std::result::Result<T, TpuError>;