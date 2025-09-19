//! # OptiRS - Advanced ML Optimization and Hardware Acceleration
//!
//! OptiRS is a comprehensive optimization library for machine learning,
//! providing advanced algorithms, hardware acceleration, and learned optimizers.
//!
//! ## Features
//!
//! - **Core Optimizers**: SGD, Adam, AdamW, RMSprop, and more
//! - **GPU Acceleration**: CUDA, Metal, OpenCL, WebGPU backends
//! - **TPU Support**: Pod coordination and XLA integration
//! - **Learned Optimizers**: Transformer-based and LSTM optimizers
//! - **Neural Architecture Search**: Automated architecture discovery
//! - **Benchmarking**: Performance analysis and regression detection
//!
//! ## Quick Start
//!
//! ```rust
//! use optirs::prelude::*;
//! use scirs2_core::ndarray_ext::Array2;
//!
//! // Create an optimizer
//! let optimizer = Adam::new(0.001)?;
//!
//! // Use with your gradients
//! let gradients = Array2::zeros((10, 10));
//! let updated_params = optimizer.step(&gradients)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub use optirs_core as core;

#[cfg(feature = "gpu")]
pub use optirs_gpu as gpu;

#[cfg(feature = "tpu")]
pub use optirs_tpu as tpu;

#[cfg(feature = "learned")]
pub use optirs_learned as learned;

#[cfg(feature = "nas")]
pub use optirs_nas as nas;

#[cfg(feature = "bench")]
pub use optirs_bench as bench;

/// Common imports for ease of use
pub mod prelude {
    pub use crate::core::optimizers::*;
    pub use crate::core::regularizers::*;
    pub use crate::core::schedulers::*;

    #[cfg(feature = "gpu")]
    pub use crate::gpu::*;

    #[cfg(feature = "learned")]
    pub use crate::learned::*;

    #[cfg(feature = "nas")]
    pub use crate::nas::*;
}

// Re-export core functionality at the top level
pub use crate::core::error::{OptimError, Result};
pub use crate::core::optimizers;
pub use crate::core::regularizers;
pub use crate::core::schedulers;
