//! # `CrossFrameworkConfig` - Trait Implementations
//!
//! This module contains trait implementations for `CrossFrameworkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CrossFrameworkConfig, Precision};

impl Default for CrossFrameworkConfig {
    fn default() -> Self {
        Self {
            enable_pytorch: true,
            enable_tensorflow: true,
            python_path: "python3".to_string(),
            temp_dir: "/tmp/scirs2_benchmark".to_string(),
            precision: Precision::F64,
            max_iterations: 1000,
            tolerance: 1e-6,
            random_seed: 42,
            batch_sizes: vec![1, 32, 128, 512],
            problem_dimensions: vec![10, 100, 1000],
            num_runs: 5,
            confidence_level: 0.95,
            learning_rate: 0.01,
            pytorch_optimizers: vec!["Adam".to_string(), "SGD".to_string(), "RMSprop".to_string()],
            tensorflow_optimizers: vec![
                "Adam".to_string(),
                "SGD".to_string(),
                "RMSprop".to_string(),
            ],
        }
    }
}
