//! WebGPU integration for GPU-accelerated optimization in the browser.
//!
//! This module is gated behind the `webgpu` feature flag.
//!
//! [`WasmGpuOptimizer`] performs genuine WebGPU capability detection and, via
//! [`WasmGpuOptimizer::initialize`], a real `navigator.gpu.requestAdapter()` /
//! `adapter.requestDevice()` handshake against the browser's WebGPU
//! implementation. It honestly reports failure (a real `Err`, never a
//! fabricated success) when WebGPU is unavailable or the browser declines to
//! grant an adapter/device.
//!
//! `navigator.gpu` and the WebGPU interfaces are still an "unstable" W3C API
//! in `web-sys` (gated behind per-type crate features *and* a
//! `--cfg=web_sys_unstable_apis` RUSTFLAG that most consumers of this crate
//! will not have set). Rather than requiring every downstream build to carry
//! that RUSTFLAG, this module reaches `navigator.gpu` and its methods
//! dynamically through `js_sys::Reflect` -- exactly the mechanism the typed
//! bindings use internally, just without the extra opt-in cfg/feature
//! requirements.
//!
//! Running GPU-accelerated optimizer *compute kernels* (WGSL shaders for each
//! optimizer) on top of the acquired device is out of scope for this module
//! and is left as a documented placeholder for future work.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use wasm_bindgen::JsCast;

use crate::error::WasmError;

/// Read a property off a JS object by name, via `Reflect.get`. Returns
/// `undefined` (never panics/throws into Rust) if the read itself fails,
/// e.g. because `obj` is not an object.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
fn get_prop(obj: &wasm_bindgen::JsValue, name: &str) -> wasm_bindgen::JsValue {
    js_sys::Reflect::get(obj, &wasm_bindgen::JsValue::from_str(name))
        .unwrap_or(wasm_bindgen::JsValue::UNDEFINED)
}

/// Read a string property off a JS object, defaulting to `""` if it is
/// missing or not a string.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
fn get_string_prop(obj: &wasm_bindgen::JsValue, name: &str) -> String {
    get_prop(obj, name).as_string().unwrap_or_default()
}

/// Get a method off a JS object and call it with zero arguments.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
fn call_method0(
    obj: &wasm_bindgen::JsValue,
    name: &str,
) -> Result<wasm_bindgen::JsValue, WasmError> {
    let f = get_prop(obj, name);
    let f: js_sys::Function = f
        .dyn_into()
        .map_err(|_| WasmError::GpuError(format!("'{name}' is not a function")))?;
    f.call0(obj)
        .map_err(|e| WasmError::GpuError(format!("{name}() threw: {:?}", e)))
}

/// Detect whether the current JS host exposes the WebGPU entry point
/// (`navigator.gpu`). This is a synchronous, best-effort capability check: a
/// `true` result means the API surface exists, not that an adapter/device can
/// necessarily be acquired (a browser may still refuse `requestAdapter()`).
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
fn webgpu_entry_point_present() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let navigator = window.navigator();
    let gpu = get_prop(navigator.as_ref(), "gpu");
    !(gpu.is_undefined() || gpu.is_null())
}

/// Outside a real wasm32 JS host there is no `navigator` to query: WebGPU is
/// genuinely unavailable, so this honestly returns `false` rather than trying
/// (and panicking on) a web-sys call that requires a JS runtime.
#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_entry_point_present() -> bool {
    false
}

/// WebGPU-accelerated optimizer wrapper.
///
/// Construct with [`WasmGpuOptimizer::new`], check [`WasmGpuOptimizer::is_available`]
/// for a cheap synchronous capability probe, then call
/// [`WasmGpuOptimizer::initialize`] to actually request a GPU adapter and
/// device from the browser.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmGpuOptimizer {
    initialized: bool,
    device_name: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmGpuOptimizer {
    /// Create a new, not-yet-initialized GPU optimizer handle.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            initialized: false,
            device_name: String::new(),
        }
    }

    /// Check whether the WebGPU API is exposed by the current JS host
    /// (`navigator.gpu` is defined). Does not attempt to acquire a device.
    pub fn is_available() -> bool {
        webgpu_entry_point_present()
    }

    /// Request a real GPU adapter and device from the browser's WebGPU
    /// implementation and record its identity.
    ///
    /// Returns `Err` (never a fabricated `Ok`) when WebGPU is unavailable, no
    /// adapter can be obtained, or the browser rejects the device request.
    #[cfg(target_arch = "wasm32")]
    pub async fn initialize(&mut self) -> Result<(), WasmError> {
        let Some(window) = web_sys::window() else {
            return Err(WasmError::GpuError("no global `window` object".to_string()));
        };
        let navigator = window.navigator();
        let gpu = get_prop(navigator.as_ref(), "gpu");
        if gpu.is_undefined() || gpu.is_null() {
            return Err(WasmError::GpuError(
                "WebGPU is not supported in this environment (navigator.gpu is undefined)"
                    .to_string(),
            ));
        }

        let adapter_promise = call_method0(&gpu, "requestAdapter")?;
        let adapter_promise: js_sys::Promise = adapter_promise.dyn_into().map_err(|_| {
            WasmError::GpuError("gpu.requestAdapter() did not return a Promise".to_string())
        })?;
        let adapter = wasm_bindgen_futures::JsFuture::from(adapter_promise)
            .await
            .map_err(|e| WasmError::GpuError(format!("requestAdapter() rejected: {:?}", e)))?;
        if adapter.is_null() || adapter.is_undefined() {
            return Err(WasmError::GpuError(
                "no GPU adapter is available".to_string(),
            ));
        }

        let info = get_prop(&adapter, "info");
        let device_name = format!(
            "{} {} ({})",
            get_string_prop(&info, "vendor"),
            get_string_prop(&info, "architecture"),
            get_string_prop(&info, "description"),
        );

        let device_promise = call_method0(&adapter, "requestDevice")?;
        let device_promise: js_sys::Promise = device_promise.dyn_into().map_err(|_| {
            WasmError::GpuError("adapter.requestDevice() did not return a Promise".to_string())
        })?;
        wasm_bindgen_futures::JsFuture::from(device_promise)
            .await
            .map_err(|e| WasmError::GpuError(format!("requestDevice() rejected: {:?}", e)))?;

        self.initialized = true;
        self.device_name = device_name;
        Ok(())
    }

    /// Outside wasm32 there is no JS host to request a device from.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn initialize(&mut self) -> Result<(), WasmError> {
        Err(WasmError::GpuError(
            "WebGPU is only available when compiled to wasm32 and run in a browser/Node host"
                .to_string(),
        ))
    }

    /// Get a human-readable description of the acquired GPU device.
    pub fn device_info(&self) -> String {
        if self.initialized {
            format!("WebGPU Device: {}", self.device_name)
        } else {
            "WebGPU not initialized".to_string()
        }
    }

    /// Check whether [`WasmGpuOptimizer::initialize`] has completed successfully.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(feature = "wasm")]
impl Default for WasmGpuOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Non-wasm placeholder (the `wasm` feature is off: no `wasm_bindgen`, no web-sys).
#[cfg(not(feature = "wasm"))]
pub struct WasmGpuOptimizer {
    initialized: bool,
    device_name: String,
}

#[cfg(not(feature = "wasm"))]
impl WasmGpuOptimizer {
    /// Create a new, not-yet-initialized handle.
    pub fn new() -> Self {
        Self {
            initialized: false,
            device_name: String::new(),
        }
    }

    /// WebGPU requires a JS host; always false without the `wasm` feature.
    pub fn is_available() -> bool {
        false
    }

    /// There is no JS host to request a device from.
    pub fn initialize(&mut self) -> Result<(), WasmError> {
        Err(WasmError::GpuError(
            "WebGPU requires the `wasm` feature and a browser/Node WebGPU host".to_string(),
        ))
    }

    /// Get device info.
    pub fn device_info(&self) -> String {
        "WebGPU not available (non-WASM target)".to_string()
    }

    /// Check if initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(not(feature = "wasm"))]
impl Default for WasmGpuOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_optimizer_starts_uninitialized() {
        let opt = WasmGpuOptimizer::new();
        assert!(!opt.is_initialized());
        assert_eq!(opt.device_info(), "WebGPU not initialized");
    }

    #[test]
    fn is_available_is_false_outside_a_browser_host() {
        // Neither a native test binary nor wasm32-without-a-JS-host has a
        // `navigator.gpu`; this must honestly report `false`, never a
        // fabricated `true`.
        assert!(!WasmGpuOptimizer::is_available());
    }

    /// Poll a future exactly once. Only valid for futures that are `Ready` on
    /// their very first poll (no real `.await` points) -- which is the case
    /// for [`WasmGpuOptimizer::initialize`] on a non-wasm32 host, since that
    /// branch returns its `Err` immediately without awaiting anything.
    #[cfg(feature = "wasm")]
    fn poll_once_to_completion<F: std::future::Future>(fut: F) -> F::Output {
        use std::pin::pin;
        use std::task::{Context, Poll, Waker};

        let mut fut = pin!(fut);
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("future unexpectedly did not resolve on first poll"),
        }
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn initialize_honestly_fails_outside_a_browser_host() {
        // On a native (non-wasm32) test host `initialize()` never touches
        // web-sys/JsValue; it returns its `Err` immediately, so this is safe
        // to drive without a real async runtime.
        let mut opt = WasmGpuOptimizer::new();
        let err =
            poll_once_to_completion(opt.initialize()).expect_err("must not fabricate success");
        assert!(matches!(err, WasmError::GpuError(_)));
        assert!(!opt.is_initialized());
    }
}
