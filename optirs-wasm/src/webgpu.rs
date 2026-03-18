//! WebGPU integration for GPU-accelerated optimization in the browser.
//!
//! This module is gated behind the `webgpu` feature flag.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WebGPU-accelerated optimizer wrapper (placeholder for future GPU integration)
///
/// This will wrap optirs-gpu's WgpuBackend for browser-based GPU computation.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmGpuOptimizer {
    initialized: bool,
    device_name: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmGpuOptimizer {
    /// Check if WebGPU is available
    #[wasm_bindgen]
    pub fn is_available() -> bool {
        // WebGPU availability check would go here
        // For now, return false as full integration is pending
        false
    }

    /// Get device capabilities description
    pub fn device_info(&self) -> String {
        if self.initialized {
            format!("WebGPU Device: {}", self.device_name)
        } else {
            "WebGPU not initialized".to_string()
        }
    }

    /// Check if the optimizer is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Non-wasm placeholder
#[cfg(not(feature = "wasm"))]
pub struct WasmGpuOptimizer {
    initialized: bool,
    device_name: String,
}

#[cfg(not(feature = "wasm"))]
impl WasmGpuOptimizer {
    /// Check if WebGPU is available (always false outside WASM)
    pub fn is_available() -> bool {
        false
    }

    /// Get device info
    pub fn device_info(&self) -> String {
        "WebGPU not available (non-WASM target)".to_string()
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}
