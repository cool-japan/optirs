// Checkpoint management for optimization coordination
//
// This module provides comprehensive checkpoint and recovery management for
// optimization workflows, including state persistence, incremental checkpointing,
// and robust recovery mechanisms.

//! Auto-generated module structure

pub mod accesspermissions_traits;
pub mod checkpointstatistics_traits;
pub mod compressioninfo_traits;
pub mod compressionstatistics_traits;
pub mod defaultrecoverystrategy_traits;
pub mod file_storage;
pub mod functions;
pub mod indexingstatistics_traits;
pub mod inmemorycheckpointstorage_traits;
pub mod nonemptydatarule_traits;
pub mod noselfdependencyrule_traits;
pub mod recoverystatistics_traits;
pub mod requiredidentifiersrule_traits;
pub mod schedulerstatistics_traits;
pub mod types;
pub mod types_15;
pub mod validationstatistics_traits;

// Re-export all types
pub use file_storage::{storage_backend_from_config, FileCheckpointStorage};
pub use functions::*;
pub use types::*;
pub use types_15::*;
