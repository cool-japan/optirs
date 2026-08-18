//! # `InMemoryCheckpointStorage` - Trait Implementations
//!
//! This module contains trait implementations for `InMemoryCheckpointStorage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `CheckpointStorage`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::functions::CheckpointStorage;
use super::types::{Checkpoint, InMemoryCheckpointStorage, StorageStatistics};
use super::types_15::CheckpointMetadata;

impl<T: Float + Debug + Send + Sync + 'static> Default for InMemoryCheckpointStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Float + Debug + Send + Sync + 'static + Clone> CheckpointStorage<T>
    for InMemoryCheckpointStorage<T>
{
    fn store(&mut self, checkpoint: &Checkpoint<T>) -> Result<String> {
        let location = format!("memory://{}", checkpoint.checkpoint_id);
        self.checkpoints
            .insert(checkpoint.checkpoint_id.clone(), checkpoint.clone());
        Ok(location)
    }

    fn retrieve(&self, checkpoint_id: &str) -> Result<Checkpoint<T>> {
        self.checkpoints
            .get(checkpoint_id)
            .cloned()
            .ok_or_else(|| Self::not_found(checkpoint_id))
    }

    fn delete(&mut self, checkpoint_id: &str) -> Result<()> {
        self.checkpoints
            .remove(checkpoint_id)
            .map(|_| ())
            .ok_or_else(|| Self::not_found(checkpoint_id))
    }

    fn list(&self, workflow_id: Option<&str>) -> Result<Vec<String>> {
        Ok(self
            .checkpoints
            .values()
            .filter(|checkpoint| {
                workflow_id
                    .map(|w| checkpoint.workflow_id == w)
                    .unwrap_or(true)
            })
            .map(|checkpoint| checkpoint.checkpoint_id.clone())
            .collect())
    }

    fn exists(&self, checkpoint_id: &str) -> Result<bool> {
        Ok(self.checkpoints.contains_key(checkpoint_id))
    }

    fn get_metadata(&self, checkpoint_id: &str) -> Result<CheckpointMetadata<T>> {
        self.checkpoints
            .get(checkpoint_id)
            .map(|checkpoint| checkpoint.metadata.clone())
            .ok_or_else(|| Self::not_found(checkpoint_id))
    }

    fn get_statistics(&self) -> Result<StorageStatistics> {
        let total_checkpoints = self.checkpoints.len();
        let total_storage_bytes: usize = self.checkpoints.values().map(|c| c.size_bytes).sum();
        let average_checkpoint_size = total_storage_bytes
            .checked_div(total_checkpoints)
            .unwrap_or(0);
        Ok(StorageStatistics {
            total_checkpoints,
            total_storage_bytes,
            average_checkpoint_size,
            // In-memory storage has no fixed capacity to report a
            // meaningful utilization/available-space figure against;
            // reporting a fabricated fraction would be worse than
            // reporting "unbounded" (0% utilized, nothing to run out of).
            utilization_percentage: 0.0,
            available_storage_bytes: usize::MAX,
        })
    }
}
