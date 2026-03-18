//! Type conversion utilities for WASM bindings.

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

/// Reshape a flat slice into chunks for multi-parameter step operations
pub fn reshape_params(flat: &[f64], dim: usize) -> Vec<Array1<f64>> {
    flat.chunks(dim)
        .map(|chunk| Array1::from_vec(chunk.to_vec()))
        .collect()
}

/// Flatten multiple Array1 results into a single Vec
pub fn flatten_results(arrays: Vec<Array1<f64>>) -> Vec<f64> {
    arrays.into_iter().flat_map(|a| a.to_vec()).collect()
}
