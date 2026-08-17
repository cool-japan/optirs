//! [`TPUBuffer`]: the typed data buffer that flows into and out of
//! [`super::backend::TPUBackend::execute_computation`].

use std::fmt::Debug;
use std::time::Instant;

use scirs2_core::numeric::Float;

use crate::error::Result;

use super::types::{BufferFlags, BufferMetadata, DataType, MemoryLayout};
use super::DeviceId;

/// TPU buffer for data
#[derive(Debug)]
pub struct TPUBuffer<T: Float + Debug + Send + Sync + 'static> {
    /// Buffer data
    ///
    /// `pub(super)`: read directly (bypassing the public API) by
    /// `super::serialization::serialize_tpu_buffers` and by the `tpu_backend`
    /// test module, both of which live in sibling submodules.
    pub(super) data: Vec<T>,

    /// Buffer shape
    pub(super) shape: Vec<usize>,

    /// Memory layout
    layout: MemoryLayout,

    /// Device location
    device: Option<DeviceId>,

    /// Buffer metadata
    metadata: BufferMetadata,
}

impl<T: Float + Debug + Send + Sync + 'static> TPUBuffer<T> {
    /// Create a new TPU buffer
    pub fn new(data: Vec<T>, shape: Vec<usize>, layout: MemoryLayout) -> Self {
        Self {
            data,
            shape,
            layout,
            device: None,
            metadata: BufferMetadata {
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                access_count: 0,
                data_type: DataType::F32, // Simplified
                flags: BufferFlags {
                    read_only: false,
                    persistent: false,
                    prefetch: false,
                    pinned: false,
                },
            },
        }
    }

    /// The memory layout the buffer was created with.
    ///
    /// Note that the reference tensor codec used by this module's private
    /// `serialization` submodule carries shape and data only, so a buffer that
    /// has been round-tripped through a computation comes back row-major
    /// regardless of the layout the original was created with.
    pub fn layout(&self) -> MemoryLayout {
        self.layout
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }

    /// Transfer buffer to device
    pub fn transfer_to_device(&mut self, device: DeviceId) -> Result<()> {
        self.device = Some(device);
        self.metadata.last_accessed = Instant::now();
        self.metadata.access_count += 1;
        Ok(())
    }
}
