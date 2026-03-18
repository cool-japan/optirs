//! Error types for OptiRS WASM bindings.

use optirs_core::error::OptimError;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Error type for WASM bindings
#[derive(Debug, thiserror::Error)]
pub enum WasmError {
    #[error("Optimizer error: {0}")]
    OptimizerError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("GPU error: {0}")]
    GpuError(String),
}

impl From<OptimError> for WasmError {
    fn from(err: OptimError) -> Self {
        match err {
            OptimError::DimensionMismatch(msg) => WasmError::DimensionMismatch(msg),
            OptimError::InvalidParameter(msg) => WasmError::InvalidParameter(msg),
            OptimError::InvalidConfig(msg) => WasmError::ConfigError(msg),
            OptimError::ConfigurationError(msg) => WasmError::ConfigError(msg),
            other => WasmError::OptimizerError(other.to_string()),
        }
    }
}

#[cfg(feature = "wasm")]
impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

// Note: From<WasmError> for JsError is automatically provided by wasm-bindgen
// via blanket impl `From<E> for JsError where E: StdError`

/// Result type alias for WASM operations
pub type WasmResult<T> = std::result::Result<T, WasmError>;
