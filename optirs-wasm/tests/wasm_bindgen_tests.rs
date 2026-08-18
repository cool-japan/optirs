//! Real wasm32 + JS-host integration tests, run with:
//!
//! ```sh
//! cd optirs-wasm
//! wasm-pack test --node --features wasm
//! ```
//!
//! Unlike `tests/wasm_tests.rs` (which exercises the plain Rust structs
//! natively), this file specifically covers the `#[wasm_bindgen]`-exported
//! surface that only exists/behaves correctly on the real wasm32 target with
//! a JS host present: `create_optimizer`/`create_scheduler` returning a
//! `JsValue`-wrapped object (this panics on a native host, see
//! `src/wasm_api.rs`). The returned value is driven purely through
//! `js_sys::Reflect`, exactly as a real JavaScript caller would use it --
//! this crate's `#[wasm_bindgen]`-exported structs do not implement
//! `wasm_bindgen::JsCast`/`From<JsValue>` on this wasm-bindgen version, so a
//! Rust-side downcast is not available/needed for this check.

#![cfg(all(feature = "wasm", target_arch = "wasm32"))]

use js_sys::{Function, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

use optirs_wasm::wasm_api::{
    available_optimizers, available_schedulers, create_optimizer, create_scheduler,
};

/// Read a property (or invoke a zero-argument getter/method) on a JS object.
fn get_property(obj: &JsValue, name: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(name))
        .unwrap_or_else(|e| panic!("property/method '{name}' should exist: {:?}", e))
}

/// Call a zero-argument method on a JS object.
fn call_method0(obj: &JsValue, name: &str) -> JsValue {
    let f: Function = get_property(obj, name)
        .dyn_into()
        .unwrap_or_else(|_| panic!("'{name}' should be callable"));
    f.call0(obj)
        .unwrap_or_else(|e| panic!("calling '{name}' should not throw: {:?}", e))
}

#[wasm_bindgen_test]
fn version_is_non_empty() {
    assert!(!optirs_wasm::version().is_empty());
}

#[wasm_bindgen_test]
fn create_optimizer_covers_every_advertised_type() {
    for opt_type in available_optimizers() {
        let config = format!(r#"{{"type": "{}", "lr": 0.01}}"#, opt_type);
        create_optimizer(&config)
            .unwrap_or_else(|e| panic!("create_optimizer({opt_type}) failed: {:?}", e));
    }
}

#[wasm_bindgen_test]
fn create_scheduler_covers_every_advertised_type() {
    for sched_type in available_schedulers() {
        // `curriculum` needs a `stages` array; every other scheduler works
        // from just its `type` field plus internal defaults.
        let config = if sched_type == "curriculum" {
            r#"{"type": "curriculum", "stages": [{"learning_rate": 0.01, "duration": 10}], "final_lr": 0.001}"#.to_string()
        } else {
            format!(r#"{{"type": "{}"}}"#, sched_type)
        };
        create_scheduler(&config)
            .unwrap_or_else(|e| panic!("create_scheduler({sched_type}) failed: {:?}", e));
    }
}

#[wasm_bindgen_test]
fn create_optimizer_returns_a_usable_adam_not_a_string() {
    let value = create_optimizer(r#"{"type": "adam", "lr": 0.05}"#)
        .expect("create_optimizer(adam) should succeed");

    // The whole point of F87: this must be the real, live optimizer object
    // (with real internal state reachable via its properties/methods), not a
    // JS string describing what would have been constructed.
    assert!(
        value.as_string().is_none(),
        "create_optimizer must not return a stringified description"
    );

    let lr = get_property(&value, "learning_rate");
    assert_eq!(
        lr.as_f64(),
        Some(0.05),
        "lr from the config must be threaded through"
    );

    let name = call_method0(&value, "name");
    assert_eq!(name.as_string().as_deref(), Some("Adam"));
}

#[wasm_bindgen_test]
fn create_scheduler_returns_a_usable_cosine_annealing() {
    let value = create_scheduler(
        r#"{"type": "cosine_annealing", "initial_lr": 0.01, "min_lr": 0.0, "t_max": 10}"#,
    )
    .expect("create_scheduler(cosine_annealing) should succeed");

    assert!(value.as_string().is_none());
    let name = call_method0(&value, "name");
    assert_eq!(name.as_string().as_deref(), Some("CosineAnnealing"));

    // `step()` takes no args and returns a plain f64 -- safe to invoke
    // through raw JS reflection without any array-marshalling machinery.
    let lr_after_step = call_method0(&value, "step");
    let lr = lr_after_step
        .as_f64()
        .expect("scheduler.step() should return a JS number");
    assert!(lr > 0.0 && lr <= 0.01);
}

#[wasm_bindgen_test]
fn create_optimizer_rejects_unknown_type() {
    create_optimizer(r#"{"type": "not_a_real_optimizer", "lr": 0.01}"#)
        .expect_err("an unknown optimizer type must be rejected, not silently accepted");
}

#[wasm_bindgen_test]
fn create_scheduler_rejects_unknown_type() {
    create_scheduler(r#"{"type": "not_a_real_scheduler"}"#)
        .expect_err("an unknown scheduler type must be rejected, not silently accepted");
}
