//! Regression tests for F17 (DVFS), F18 (energy-strategy device model) and
//! F59 (thermal RC integration) in `energy_efficient.rs`.
//!
//! Split into its own file (rather than an inline `#[cfg(test)] mod tests`)
//! to keep `energy_efficient.rs` itself under the workspace's 2000-line
//! file-size policy.

use super::*;

fn workload(active_neurons: usize, spike_rate: f64, synaptic_activity: f64) -> WorkloadSample<f64> {
    WorkloadSample {
        timestamp: Instant::now(),
        active_neurons,
        spike_rate,
        synaptic_activity,
        memory_access_pattern: MemoryAccessPattern::Mixed,
        communication_overhead: 1.0,
    }
}

/// Config with a chosen primary strategy, built in one initializer.
fn config_with(strategy: EnergyOptimizationStrategy) -> EnergyEfficientConfig<f64> {
    EnergyEfficientConfig::<f64> {
        primary_strategy: strategy,
        ..EnergyEfficientConfig::<f64>::default()
    }
}

/// F17: the DVFS power-reduction ratio must not be arithmetically
/// forced to 1.0 by reading back state that was already overwritten.
/// A low-utilization workload should trigger a real voltage/frequency
/// transition away from the 1.0V/100MHz startup point, and the
/// resulting `power_reduction` in the result must be nonzero.
#[test]
fn dvfs_transition_yields_nonzero_power_reduction() {
    let config = config_with(EnergyOptimizationStrategy::DynamicVoltageScaling);
    let mut optimizer = EnergyEfficientOptimizer::new(config, 1000);

    let low_util = workload(10, 5.0, 2.0);
    let result = optimizer
        .optimize_energy(&low_util)
        .expect("optimize_energy failed");

    let state = optimizer.get_system_state();
    assert!(
        state.current_voltage != 1.0 || state.current_frequency != 100.0,
        "DVFS controller never transitioned away from the startup operating point"
    );
    assert_ne!(
        result.power_reduction, 0.0,
        "power_reduction was zero: the DVFS ratio is still being forced to 1.0"
    );
}

/// F18: clock gating savings must scale with the workload's idle
/// fraction rather than being a fixed percentage.
#[test]
fn clock_gating_scales_with_idle_fraction() {
    let make = |active: usize| {
        let config = config_with(EnergyOptimizationStrategy::ClockGating);
        let mut optimizer = EnergyEfficientOptimizer::new(config, 1000);
        optimizer
            .optimize_energy(&workload(active, 5.0, 2.0))
            .expect("optimize_energy failed")
            .power_reduction
    };
    let mostly_idle = make(10); // ~99% idle
    let mostly_active = make(900); // ~10% idle
    assert!(
        mostly_idle > mostly_active,
        "clock gating did not scale down with a less-idle workload: idle={mostly_idle}, active={mostly_active}"
    );
}

/// F18: power gating must gate zero domains (zero savings) when the
/// workload is not idle enough to clear the gating threshold, and a
/// positive number of domains when it is.
#[test]
fn power_gating_respects_idle_threshold() {
    let config = config_with(EnergyOptimizationStrategy::PowerGating);

    let mut busy_optimizer = EnergyEfficientOptimizer::new(config.clone(), 1000);
    let busy_result = busy_optimizer
        .optimize_energy(&workload(950, 5.0, 2.0)) // 5% idle: below threshold
        .expect("optimize_energy failed");
    assert_eq!(busy_result.power_reduction, 0.0);

    let mut idle_optimizer = EnergyEfficientOptimizer::new(config, 1000);
    let idle_result = idle_optimizer
        .optimize_energy(&workload(50, 5.0, 2.0)) // 95% idle: above threshold
        .expect("optimize_energy failed");
    assert!(idle_result.power_reduction > 0.0);
}

/// F18: sleep mode must escalate from light to deep sleep as the
/// workload becomes more idle.
#[test]
fn sleep_mode_escalates_with_idleness() {
    let config = config_with(EnergyOptimizationStrategy::SleepModeOptimization);

    let mut light = EnergyEfficientOptimizer::new(config.clone(), 1000);
    light
        .optimize_energy(&workload(600, 5.0, 2.0)) // 40% idle: light sleep
        .expect("optimize_energy failed");
    assert!(matches!(
        light.get_system_state().sleep_status,
        SleepStatus::LightSleep
    ));

    let mut deep = EnergyEfficientOptimizer::new(config, 1000);
    deep.optimize_energy(&workload(50, 5.0, 2.0)) // 95% idle: deep sleep
        .expect("optimize_energy failed");
    assert!(matches!(
        deep.get_system_state().sleep_status,
        SleepStatus::DeepSleep
    ));
}

/// F18: thermal-aware optimization is a proportional controller —
/// reduction should grow monotonically with temperature, not jump
/// between two hardcoded steps.
#[test]
fn thermal_aware_reduction_is_proportional_to_temperature() {
    let config = config_with(EnergyOptimizationStrategy::ThermalAwareOptimization);

    let mut cool = EnergyEfficientOptimizer::new(config.clone(), 1000);
    cool.system_state.temperature = 60.0; // at the safe boundary
    let cool_result = cool
        .optimize_energy(&workload(500, 5.0, 2.0))
        .expect("optimize_energy failed");

    let mut warm = EnergyEfficientOptimizer::new(config.clone(), 1000);
    warm.system_state.temperature = 75.0; // halfway to critical
    let warm_result = warm
        .optimize_energy(&workload(500, 5.0, 2.0))
        .expect("optimize_energy failed");

    let mut hot = EnergyEfficientOptimizer::new(config, 1000);
    hot.system_state.temperature = 95.0; // at/above critical
    let hot_result = hot
        .optimize_energy(&workload(500, 5.0, 2.0))
        .expect("optimize_energy failed");

    assert!(cool_result.power_reduction < warm_result.power_reduction);
    assert!(warm_result.power_reduction < hot_result.power_reduction);
}

/// F18: when a real weight/activation matrix is supplied, sparsity
/// must be measured from its actual zero fraction, not the
/// activity-derived estimate.
#[test]
fn sparse_strategy_uses_real_matrix_zero_fraction() {
    let config = config_with(EnergyOptimizationStrategy::SparseComputation);
    let mut optimizer = EnergyEfficientOptimizer::new(config, 1000);

    // 80% zero entries: a much higher sparsity than the workload
    // (900/1000 active => ~10% idle-derived estimate) would suggest.
    let matrix = Array2::from_shape_vec(
        (2, 10),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
        ],
    )
    .expect("failed to build test matrix");

    let with_matrix = optimizer
        .optimize_energy_with_matrix(&workload(900, 5.0, 2.0), Some(&matrix))
        .expect("optimize_energy_with_matrix failed");

    let mut baseline_optimizer = EnergyEfficientOptimizer::new(
        config_with(EnergyOptimizationStrategy::SparseComputation),
        1000,
    );
    let without_matrix = baseline_optimizer
        .optimize_energy(&workload(900, 5.0, 2.0))
        .expect("optimize_energy failed");

    assert!(
        with_matrix.power_reduction > without_matrix.power_reduction,
        "real 80%-sparse matrix should yield more savings than the ~10%-idle estimate: \
         with_matrix={}, without_matrix={}",
        with_matrix.power_reduction,
        without_matrix.power_reduction
    );
}

/// F18: `predict_energy` must honor `horizon` (roughly linear in it)
/// and must not return a fabricated constant when no data has been
/// observed yet.
#[test]
fn predict_energy_honors_horizon_and_history() {
    let config = EnergyEfficientConfig::<f64>::default();
    let empty_optimizer = EnergyEfficientOptimizer::new(config.clone(), 1000);
    let empty_prediction = empty_optimizer
        .predictive_manager
        .predict_energy(Duration::from_secs(60))
        .expect("predict_energy failed");
    assert_eq!(empty_prediction, 0.0);

    let mut optimizer = EnergyEfficientOptimizer::new(config, 1000);
    for _ in 0..10 {
        optimizer
            .optimize_energy(&workload(500, 5.0, 2.0))
            .expect("optimize_energy failed");
    }

    let short = optimizer
        .predictive_manager
        .predict_energy(Duration::from_millis(10))
        .expect("predict_energy failed");
    let long = optimizer
        .predictive_manager
        .predict_energy(Duration::from_millis(100))
        .expect("predict_energy failed");

    assert!(
        short > 0.0,
        "prediction should be positive once power has been observed"
    );
    // 10x the horizon should yield ~10x the predicted energy (same
    // average power extrapolated further out).
    assert!(
        (long / short - 10.0).abs() < 1e-6,
        "predict_energy is not linear in horizon: short={short}, long={long}"
    );
}

/// F59: the thermal model integrates via an RC lag rather than
/// snapping instantly to the steady-state temperature, and its power
/// history is bounded rather than growing without limit.
#[test]
fn thermal_model_integrates_gradually_and_bounds_history() {
    let config = EnergyEfficientConfig::<f64>::default();
    let mut optimizer = EnergyEfficientOptimizer::new(config, 1000);
    optimizer.system_state.current_power = 1000.0; // steady_state ≈ 1000*0.5+25 = 525°C
    let initial_temp = optimizer.thermal_manager.current_temperature;
    assert_eq!(initial_temp, 25.0);

    let state_snapshot = optimizer.system_state.clone();
    optimizer
        .thermal_manager
        .update(&state_snapshot)
        .expect("thermal update failed");
    let after_first = optimizer.thermal_manager.current_temperature;
    assert!(
        after_first > initial_temp && after_first < 525.0,
        "temperature should move toward, but not jump to, steady state: {after_first}"
    );

    std::thread::sleep(Duration::from_millis(20));
    let state_snapshot = optimizer.system_state.clone();
    optimizer
        .thermal_manager
        .update(&state_snapshot)
        .expect("thermal update failed");
    let after_second = optimizer.thermal_manager.current_temperature;
    assert!(
        after_second > after_first,
        "temperature should keep approaching steady state over time"
    );

    // Bound power_history growth (F59): with a tiny window, repeated
    // updates must not accumulate unboundedly.
    optimizer.energy_monitor.window_size = Duration::from_millis(1);
    for _ in 0..20 {
        let state_snapshot = optimizer.system_state.clone();
        optimizer
            .energy_monitor
            .update(&state_snapshot)
            .expect("energy monitor update failed");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        optimizer.energy_monitor.power_history.len() <= 3,
        "power_history grew without bound: len={}",
        optimizer.energy_monitor.power_history.len()
    );
}
