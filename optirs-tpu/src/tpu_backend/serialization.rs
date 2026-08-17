//! CPU reference executor support: a small tensor codec used to give
//! [`super::execution::ExecutionEngine::execute_task`] real (de)serialized
//! argument and result data.
//!
//! There is no program-binary codec here any more. The compiled binary now
//! comes from the real code generator behind [`crate::xla::XLACompiler`]; the
//! descriptor encoding that used to stand in for it (a header plus the
//! computation id and target codes) has been deleted along with the
//! `TPUBackend` compile path that produced it.

use std::fmt::Debug;

use scirs2_core::numeric::{Float, NumCast};

use scirs2_core::error::ErrorContext;

use crate::error::{OptimError, Result};

use super::buffer::TPUBuffer;
use super::types::MemoryLayout;

/// Deterministic per-byte energy estimate (nanojoules) used by the CPU
/// reference executor to derive `energy_consumed` from real byte counts.
///
/// `pub(super)`: used by [`super::execution::ExecutionEngine::execute_task`]
/// and by the `tpu_backend` test module, both sibling submodules.
pub(super) const ENERGY_PER_BYTE_NANOJOULE: f64 = 0.05;

/// Intermediate CPU-side tensor used by the reference executor and the buffer
/// (de)serialization codec.
///
/// `pub(super)`: constructed directly by the `tpu_backend` test module (a
/// sibling submodule) to exercise the codec.
#[derive(Debug, Clone)]
pub(super) struct RefTensor {
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<f64>,
}

/// Encode reference tensors into a self-describing little-endian byte stream.
/// Layout: `[u32 tensor_count]` then per tensor
/// `[u32 rank][rank x u64 dim][u64 len][len x f64 value]`.
///
/// `pub(super)`: called from [`super::execution::ExecutionEngine::execute_task`]
/// and the `tpu_backend` test module, both sibling submodules.
pub(super) fn encode_ref_tensors(tensors: &[RefTensor]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for tensor in tensors {
        out.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for &dim in &tensor.shape {
            out.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        out.extend_from_slice(&(tensor.data.len() as u64).to_le_bytes());
        for &value in &tensor.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// Panic-free cursor over an untrusted byte payload. Every read is
/// bounds-checked and reports a structured error instead of indexing out of
/// range.
struct ByteCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            OptimError::InvalidInput(ErrorContext::new(
                "reference tensor payload length overflow".to_string(),
            ))
        })?;
        if end > self.bytes.len() {
            return Err(OptimError::InvalidInput(ErrorContext::new(format!(
                "truncated reference tensor payload: need {} bytes, have {}",
                end,
                self.bytes.len()
            ))));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let arr: [u8; 4] = self.take(4)?.try_into().map_err(|_| {
            OptimError::InvalidInput(ErrorContext::new(
                "invalid u32 in reference tensor payload".to_string(),
            ))
        })?;
        Ok(u32::from_le_bytes(arr))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let arr: [u8; 8] = self.take(8)?.try_into().map_err(|_| {
            OptimError::InvalidInput(ErrorContext::new(
                "invalid u64 in reference tensor payload".to_string(),
            ))
        })?;
        Ok(u64::from_le_bytes(arr))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let arr: [u8; 8] = self.take(8)?.try_into().map_err(|_| {
            OptimError::InvalidInput(ErrorContext::new(
                "invalid f64 in reference tensor payload".to_string(),
            ))
        })?;
        Ok(f64::from_le_bytes(arr))
    }
}

/// Decode a byte stream produced by [`encode_ref_tensors`]. Bounded and
/// panic-free even on malformed input.
///
/// `pub(super)`: called from [`super::execution::ExecutionEngine::execute_task`]
/// and the `tpu_backend` test module, both sibling submodules.
pub(super) fn decode_ref_tensors(bytes: &[u8]) -> Result<Vec<RefTensor>> {
    let mut cursor = ByteCursor::new(bytes);
    let tensor_count = cursor.read_u32()? as usize;
    // Cap the pre-allocation by the real payload size so a hostile header cannot
    // trigger a huge allocation; the reads below fail fast if data runs out.
    let mut tensors = Vec::with_capacity(tensor_count.min(bytes.len()));
    for _ in 0..tensor_count {
        let rank = cursor.read_u32()? as usize;
        let mut shape = Vec::with_capacity(rank.min(bytes.len()));
        for _ in 0..rank {
            shape.push(cursor.read_u64()? as usize);
        }
        let len = cursor.read_u64()? as usize;
        let mut data = Vec::with_capacity(len.min(bytes.len()));
        for _ in 0..len {
            data.push(cursor.read_f64()?);
        }
        tensors.push(RefTensor { shape, data });
    }
    Ok(tensors)
}

/// Serialize typed TPU buffers into the reference payload consumed by
/// [`super::execution::ExecutionEngine::execute_task`].
///
/// `pub(super)`: called from [`super::backend::TPUBackend::execute_computation`],
/// a sibling submodule.
pub(super) fn serialize_tpu_buffers<T: Float + Debug + Send + Sync + 'static>(
    buffers: &[TPUBuffer<T>],
) -> Result<Vec<u8>> {
    let mut tensors = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        let mut data = Vec::with_capacity(buffer.data.len());
        for &value in &buffer.data {
            let as_f64 = <f64 as NumCast>::from(value).ok_or_else(|| {
                OptimError::TypeError(ErrorContext::new(
                    "failed to convert TPU buffer element to f64".to_string(),
                ))
            })?;
            data.push(as_f64);
        }
        tensors.push(RefTensor {
            shape: buffer.shape.clone(),
            data,
        });
    }
    Ok(encode_ref_tensors(&tensors))
}

/// Deserialize a reference payload back into typed TPU buffers.
///
/// `pub(super)`: called from [`super::backend::TPUBackend::execute_computation`],
/// a sibling submodule.
pub(super) fn deserialize_tpu_buffers<T: Float + Debug + Send + Sync + 'static>(
    bytes: &[u8],
) -> Result<Vec<TPUBuffer<T>>> {
    let tensors = decode_ref_tensors(bytes)?;
    let mut buffers = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let mut data = Vec::with_capacity(tensor.data.len());
        for value in tensor.data {
            let typed = <T as NumCast>::from(value).ok_or_else(|| {
                OptimError::TypeError(ErrorContext::new(
                    "failed to convert reference tensor element to target type".to_string(),
                ))
            })?;
            data.push(typed);
        }
        buffers.push(TPUBuffer::new(data, tensor.shape, MemoryLayout::RowMajor));
    }
    Ok(buffers)
}
