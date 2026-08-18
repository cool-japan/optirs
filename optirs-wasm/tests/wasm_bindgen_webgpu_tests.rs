//! Real wasm32 + JS-host tests for the `webgpu` feature, run with:
//!
//! ```sh
//! cd optirs-wasm
//! wasm-pack test --node --features webgpu
//! ```
//!
//! Node.js has no `window`/`navigator`/WebGPU implementation, so
//! `WasmGpuOptimizer::is_available()` must honestly report `false` here --
//! this test exists specifically to prove that on the real wasm32 target
//! (where the browser-detection code path actually compiles and runs) it
//! does not fabricate availability.

#![cfg(all(feature = "webgpu", target_arch = "wasm32"))]

use wasm_bindgen_test::*;

use optirs_wasm::webgpu::WasmGpuOptimizer;

#[wasm_bindgen_test]
fn is_available_is_honest_in_a_headless_node_host() {
    // No `navigator.gpu` in Node: must not fabricate `true`.
    assert!(!WasmGpuOptimizer::is_available());
}

#[wasm_bindgen_test]
fn new_optimizer_starts_uninitialized() {
    let opt = WasmGpuOptimizer::new();
    assert!(!opt.is_initialized());
    assert_eq!(opt.device_info(), "WebGPU not initialized");
}

#[wasm_bindgen_test]
async fn initialize_honestly_fails_without_a_webgpu_host() {
    let mut opt = WasmGpuOptimizer::new();
    let result = opt.initialize().await;
    assert!(
        result.is_err(),
        "must not fabricate a successful GPU device acquisition in a host with no WebGPU"
    );
    assert!(!opt.is_initialized());
}
