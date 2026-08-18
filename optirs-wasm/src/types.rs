//! Type conversion utilities for WASM bindings.

use crate::error::{WasmError, WasmResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Convert a slice of f64 to Array1
pub fn slice_to_array1(data: &[f64]) -> Array1<f64> {
    Array1::from_vec(data.to_vec())
}

/// Convert Array1 to Vec
pub fn array1_to_vec(arr: Array1<f64>) -> Vec<f64> {
    arr.to_vec()
}

/// Convert ArrayView1 to Vec
pub fn array_view_to_vec(arr: ArrayView1<f64>) -> Vec<f64> {
    arr.to_vec()
}

/// Reshape a flat slice into chunks for multi-parameter step operations.
///
/// # Errors
/// Returns [`WasmError::InvalidParameter`] for `dim == 0` rather than
/// panicking: `<[f64]>::chunks` panics on a zero chunk size, and `dim` here
/// is caller-controlled (ultimately from JS via the WASM boundary), so it
/// must be validated instead of trusted.
pub fn reshape_params(flat: &[f64], dim: usize) -> WasmResult<Vec<Array1<f64>>> {
    if dim == 0 {
        return Err(WasmError::InvalidParameter(
            "reshape_params: dim must be non-zero".to_string(),
        ));
    }
    Ok(flat
        .chunks(dim)
        .map(|chunk| Array1::from_vec(chunk.to_vec()))
        .collect())
}

/// Flatten multiple Array1 results into a single Vec
pub fn flatten_results(arrays: Vec<Array1<f64>>) -> Vec<f64> {
    arrays.into_iter().flat_map(|a| a.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_params_splits_into_even_chunks() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let chunks = reshape_params(&flat, 2).expect("dim=2 is valid");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].to_vec(), vec![1.0, 2.0]);
        assert_eq!(chunks[2].to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn reshape_params_keeps_a_short_trailing_chunk() {
        let flat = vec![1.0, 2.0, 3.0];
        let chunks = reshape_params(&flat, 2).expect("dim=2 is valid");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].to_vec(), vec![3.0]);
    }

    #[test]
    fn reshape_params_rejects_zero_dim_instead_of_panicking() {
        let flat = vec![1.0, 2.0, 3.0];
        let err = reshape_params(&flat, 0).expect_err("dim=0 must be rejected");
        assert!(matches!(err, WasmError::InvalidParameter(_)));
    }

    #[test]
    fn flatten_results_concatenates_in_order() {
        let arrays = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0]),
        ];
        assert_eq!(flatten_results(arrays), vec![1.0, 2.0, 3.0]);
    }
}
