//! # OptiRS WASM
//!
//! WebAssembly bindings for OptiRS - High-performance deep learning optimizers
//! and learning rate schedulers for the browser and Node.js.

pub mod config;
pub mod error;
pub mod metrics;
pub mod optimizers;
pub mod schedulers;
pub mod types;

#[cfg(feature = "wasm")]
pub mod wasm_api;

#[cfg(feature = "webgpu")]
pub mod webgpu;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Initialize the WASM module with panic hook for better error messages
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Get the version of optirs-wasm
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get the version (non-wasm)
#[cfg(not(feature = "wasm"))]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
