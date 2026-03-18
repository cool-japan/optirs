//! Integration tests for optirs-wasm
//!
//! These tests run natively (not in wasm32) to verify the wrapper logic.

use optirs_wasm::config::WasmOptimizerConfig;
use optirs_wasm::metrics::WasmMetricsCollector;
use optirs_wasm::optimizers::*;
use optirs_wasm::schedulers::*;

// ===== Optimizer Tests =====

#[test]
fn test_adam_step() {
    let mut adam = WasmAdam::new(0.001);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = adam
        .step(&params, &grads)
        .expect("Adam step should succeed");
    assert_eq!(result.len(), 3);
    // Parameters should have been updated
    assert!((result[0] - 1.0).abs() > 1e-10);
}

#[test]
fn test_adam_with_config() {
    let adam = WasmAdam::new_with_config(0.01, 0.9, 0.999, 1e-8, 0.01);
    assert!((adam.learning_rate() - 0.01).abs() < 1e-10);
}

#[test]
fn test_adamw_step() {
    let mut adamw = WasmAdamW::new(0.001);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = adamw
        .step(&params, &grads)
        .expect("AdamW step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_sgd_step() {
    let mut sgd = WasmSGD::new(0.01);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = sgd.step(&params, &grads).expect("SGD step should succeed");
    assert_eq!(result.len(), 3);
    // SGD: param = param - lr * grad
    assert!((result[0] - 0.999).abs() < 1e-6);
}

#[test]
fn test_rmsprop_step() {
    let mut rmsprop = WasmRMSprop::new(0.01);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = rmsprop
        .step(&params, &grads)
        .expect("RMSprop step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_lion_step() {
    let mut lion = WasmLion::new(0.001);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = lion
        .step(&params, &grads)
        .expect("Lion step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_lamb_step() {
    let mut lamb = WasmLAMB::new(0.001);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = lamb
        .step(&params, &grads)
        .expect("LAMB step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_radam_step() {
    let mut radam = WasmRAdam::new(0.001);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = radam
        .step(&params, &grads)
        .expect("RAdam step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_adagrad_step() {
    let mut adagrad = WasmAdagrad::new(0.01);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = adagrad
        .step(&params, &grads)
        .expect("Adagrad step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_lars_step() {
    let mut lars = WasmLARS::new(0.01);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = lars
        .step(&params, &grads)
        .expect("LARS step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_sparse_adam_step() {
    let mut sparse = WasmSparseAdam::new(0.001);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.0, 0.3]; // Sparse: some zeros
    let result = sparse
        .step(&params, &grads)
        .expect("SparseAdam step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_adadelta_step() {
    let mut adadelta = WasmAdaDelta::new(0.95, 1e-6).expect("AdaDelta creation should succeed");
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = adadelta
        .step(&params, &grads)
        .expect("AdaDelta step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_adabound_step() {
    let mut adabound = WasmAdaBound::new(0.001, 0.1, 0.9, 0.999, 1e-8, 1e-3, 0.0, false)
        .expect("AdaBound creation should succeed");
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = adabound
        .step(&params, &grads)
        .expect("AdaBound step should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_ranger_step() {
    let mut ranger = WasmRanger::new(0.001, 0.9, 0.999, 1e-8, 0.0, 5, 0.5)
        .expect("Ranger creation should succeed");
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];
    let result = ranger
        .step(&params, &grads)
        .expect("Ranger step should succeed");
    assert_eq!(result.len(), 3);
}

// ===== Step List Tests =====

#[test]
fn test_adam_step_list() {
    let mut adam = WasmAdam::new(0.001);
    // Two parameter groups of dim=3
    let params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let grads = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let result = adam
        .step_list(&params, &grads, 3)
        .expect("step_list should succeed");
    assert_eq!(result.len(), 6);
}

// ===== Scheduler Tests =====

#[test]
fn test_cosine_annealing() {
    let mut sched = WasmCosineAnnealing::new(0.001, 0.0001, 100);
    let lr = sched.step();
    assert!(lr > 0.0);
    assert!(lr <= 0.001);
}

#[test]
fn test_cosine_annealing_warm_restarts() {
    let mut sched = WasmCosineAnnealingWarmRestarts::new(0.001, 0.0001, 10, 2.0);
    for _ in 0..20 {
        let lr = sched.step();
        assert!(lr >= 0.0);
    }
}

#[test]
fn test_step_decay() {
    let mut sched = WasmStepDecay::new(0.1, 10, 0.1);
    let initial_lr = sched.learning_rate();
    assert!((initial_lr - 0.1).abs() < 1e-10);
    for _ in 0..10 {
        sched.step();
    }
    let lr_after = sched.learning_rate();
    assert!(lr_after < initial_lr);
}

#[test]
fn test_exponential_decay() {
    let mut sched = WasmExponentialDecay::new(0.1, 0.95, 1);
    let lr1 = sched.step();
    let lr2 = sched.step();
    assert!(lr2 < lr1);
}

#[test]
fn test_linear_decay() {
    let mut sched = WasmLinearDecay::new(0.1, 0.01, 100);
    let lr = sched.step();
    assert!(lr < 0.1);
    assert!(lr > 0.01);
}

#[test]
fn test_constant_scheduler() {
    let mut sched = WasmConstantScheduler::new(0.001);
    assert!((sched.step() - 0.001).abs() < 1e-10);
    assert!((sched.step() - 0.001).abs() < 1e-10);
}

#[test]
fn test_cyclic_lr() {
    let mut sched = WasmCyclicLR::new(0.001, 0.01, 10);
    for _ in 0..20 {
        let lr = sched.step();
        assert!(lr >= 0.001 - 1e-10);
        assert!(lr <= 0.01 + 1e-10);
    }
}

#[test]
fn test_reduce_on_plateau() {
    let mut sched = WasmReduceOnPlateau::new(0.1, 0.1, 3);
    let initial_lr = sched.learning_rate();
    // Simulate no improvement
    for _ in 0..5 {
        sched.step_with_metric(1.0);
    }
    // After patience exceeded, LR should decrease
    let lr_after = sched.learning_rate();
    assert!(lr_after <= initial_lr);
}

#[test]
fn test_vit_layer_decay() {
    let mut sched = WasmViTLayerDecay::new(0.001, 0.75, 12);
    // Must step at least once to get a non-zero base LR
    sched.step();
    let rates = sched.get_all_layer_rates();
    assert_eq!(rates.len(), 12);
    // Earlier layers (lower index) should have lower LR due to decay
    assert!(
        rates[0] < rates[11],
        "Layer 0 ({}) should have lower LR than layer 11 ({})",
        rates[0],
        rates[11]
    );
}

// ===== Metrics Tests =====

#[test]
fn test_metrics_collector() {
    let mut collector = WasmMetricsCollector::new();
    collector.register_optimizer("adam");

    let params_before = vec![1.0, 2.0, 3.0];
    let params_after = vec![0.999, 1.998, 2.997];
    let grads = vec![0.1, 0.2, 0.3];

    collector.update("adam", 0.001, &grads, &params_before, &params_after);

    assert_eq!(collector.optimizer_count(), 1);
    let report = collector.summary_report();
    assert!(report.contains("adam"));
}

// ===== Config Tests =====

#[test]
fn test_config_json_roundtrip() {
    let config = WasmOptimizerConfig::new(0.001);
    let json = config.to_json().expect("Serialization should succeed");
    let restored = WasmOptimizerConfig::from_json(&json).expect("Deserialization should succeed");
    assert!((restored.lr() - 0.001).abs() < 1e-10);
}

// ===== Integration Tests =====

#[test]
fn test_optimizer_with_scheduler() {
    let mut adam = WasmAdam::new(0.001);
    let mut sched = WasmCosineAnnealing::new(0.001, 0.0001, 100);

    let mut params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];

    for _ in 0..10 {
        let lr = sched.step();
        adam.set_learning_rate(lr);
        params = adam.step(&params, &grads).expect("Step should succeed");
    }

    // Params should have changed
    assert!((params[0] - 1.0).abs() > 1e-6);
}

#[test]
fn test_error_dimension_mismatch() {
    let mut adam = WasmAdam::new(0.001);
    let params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2]; // Mismatched dimensions
    let result = adam.step(&params, &grads);
    assert!(result.is_err());
}

#[test]
fn test_version() {
    let v = optirs_wasm::version();
    assert!(!v.is_empty());
}
