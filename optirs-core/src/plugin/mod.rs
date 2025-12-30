// Plugin architecture for custom optimizer development
//
// This module provides a comprehensive plugin system that allows developers to create,
// register, and use custom optimizers seamlessly with the existing optimizer ecosystem.
//
// # Plugin Architecture
//
// The plugin system consists of several key components:
// - Plugin trait definitions for optimizers and extensions
// - Plugin registry for discovery and management
// - Plugin loader for dynamic loading (when supported)
// - Plugin SDK with utilities and helpers
// - Plugin validation and testing framework
//
// # Examples
//
// ## Creating a Custom Optimizer Plugin
//
// ```no_run
// use optirs_core::plugin::{OptimizerPlugin, PluginCapabilities};
// use optirs_core::plugin::core::{PluginInfo, OptimizerConfig, OptimizerState};
// use optirs_core::error::{OptimError, Result};
// use scirs2_core::ndarray::Array1;
//
// #[derive(Debug, Clone)]
// struct MyCustomOptimizer {
//     learning_rate: f64,
// }
//
// impl OptimizerPlugin<f64> for MyCustomOptimizer {
//     fn step(&mut self, params: &Array1<f64>, gradients: &Array1<f64>) -> Result<Array1<f64>> {
//         // Custom optimization logic
//         Ok(params - &(gradients * self.learning_rate))
//     }
//
//     fn name(&self) -> &str { "MyCustomOptimizer" }
//     fn version(&self) -> &str { "1.0.0" }
//
//     fn plugin_info(&self) -> PluginInfo {
//         PluginInfo {
//             name: self.name().to_string(),
//             version: self.version().to_string(),
//             description: "A simple custom optimizer".to_string(),
//             author: "Example Author".to_string(),
//             ..PluginInfo::default()
//         }
//     }
//
//     fn capabilities(&self) -> PluginCapabilities {
//         PluginCapabilities::default()
//     }
//
//     fn initialize(&mut self, paramshapes: &[usize]) -> Result<()> { Ok(()) }
//     fn reset(&mut self) -> Result<()> { Ok(()) }
//     fn get_config(&self) -> OptimizerConfig { OptimizerConfig::default() }
//     fn set_config(&mut self,
//         config: OptimizerConfig) -> Result<()> { Ok(()) }
//     fn get_state(&self) -> Result<OptimizerState> { Ok(OptimizerState::default()) }
//     fn set_state(&mut self,
//         state: OptimizerState) -> Result<()> { Ok(()) }
//     fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<f64>> {
//         Box::new(self.clone())
//     }
// }
// ```

pub mod core;
pub mod loader;
pub mod registry;
pub mod sdk;
pub mod validation;
// Examples are in the examples/ directory

pub use core::{OptimizerPlugin, PluginCapabilities, PluginMetadata as CorePluginMetadata};
pub use loader::{
    LoaderConfig, PluginLoadResult, PluginLoader, PluginMetadata as LoaderPluginMetadata,
    ValidationRule as LoaderValidationRule,
};
pub use registry::{PluginRegistry, RegistryConfig};
pub use sdk::{PluginSDK, ValidationRule as SDKValidationRule};
pub use validation::{PluginValidationFramework, ValidationConfig};
