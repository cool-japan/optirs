// Regression tests for the low-latency streaming optimizer (findings L1-L6).
//
// Every assertion below names a value the pre-fix code provably could not
// produce; see the comment above each test for the exact old behaviour.

use super::*;
use crate::optimizers::SGD;

fn base_config() -> LowLatencyConfig {
    LowLatencyConfig {
        enable_precomputation: false,
        enable_quantization: false,
        use_approximations: false,
        enable_simd: false,
        enable_lock_free: false,
        ..LowLatencyConfig::default()
    }
}

fn optimizer_with(
    learning_rate: f64,
    config: LowLatencyConfig,
) -> LowLatencyOptimizer<SGD<f64>, f64> {
    LowLatencyOptimizer::new(SGD::new(learning_rate), config)
        .expect("constructing a low-latency optimizer with a valid config must succeed")
}

// ---------------------------------------------------------------------------
// L1: exact_update used to build `Array1::zeros(len)` as "current parameters"
// on every call, so the returned vector was always exactly one step from the
// origin and every step discarded all prior progress.
// ---------------------------------------------------------------------------

#[test]
fn exact_update_accumulates_parameters_across_steps() {
    let mut optimizer = optimizer_with(0.1, base_config());
    let gradient = Array1::from_vec(vec![1.0f64, 1.0, 1.0]);

    let first = optimizer
        .low_latency_step(&gradient)
        .expect("first step must succeed");
    let second = optimizer
        .low_latency_step(&gradient)
        .expect("second step must succeed");
    let third = optimizer
        .low_latency_step(&gradient)
        .expect("third step must succeed");

    for value in first.iter() {
        assert!((value - (-0.1)).abs() < 1e-12, "first step must be -lr*g");
    }
    for value in second.iter() {
        assert!(
            (value - (-0.2)).abs() < 1e-12,
            "L1 regression: the second step must accumulate on top of the first \
             (expected -0.2, got {value})"
        );
    }
    for value in third.iter() {
        assert!(
            (value - (-0.3)).abs() < 1e-12,
            "L1 regression: the third step must accumulate (expected -0.3, got {value})"
        );
    }
    assert_ne!(
        first.to_vec(),
        second.to_vec(),
        "L1 regression: repeated steps with the same gradient returned an identical vector, \
         which means the parameters were being rebuilt from zeros every step"
    );
}

#[test]
fn seeded_parameters_are_not_zeroed_by_the_first_step() {
    let mut optimizer = optimizer_with(0.1, base_config());
    optimizer.set_parameters(Array1::from_vec(vec![1.0f64, 2.0, 3.0]));
    let gradient = Array1::from_vec(vec![1.0f64, 1.0, 1.0]);

    let updated = optimizer
        .low_latency_step(&gradient)
        .expect("step must succeed");

    assert!((updated[0] - 0.9).abs() < 1e-12);
    assert!((updated[1] - 1.9).abs() < 1e-12);
    assert!(
        (updated[2] - 2.9).abs() < 1e-12,
        "L1 regression: seeded parameters were discarded (got {:?})",
        updated.to_vec()
    );
}

#[test]
fn changing_gradient_dimension_is_an_error_not_a_silent_reset() {
    let mut optimizer = optimizer_with(0.1, base_config());
    optimizer.set_parameters(Array1::from_vec(vec![1.0f64, 2.0, 3.0]));

    let mismatched = Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
    assert!(
        optimizer.low_latency_step(&mismatched).is_err(),
        "a gradient whose dimension does not match the tracked parameters must be rejected \
         instead of silently reallocating the parameters at the origin"
    );
}

#[test]
fn empty_gradient_is_rejected() {
    let mut optimizer = optimizer_with(0.1, base_config());
    let empty: Array1<f64> = Array1::from_vec(vec![]);
    assert!(optimizer.low_latency_step(&empty).is_err());
}

// ---------------------------------------------------------------------------
// L2: `get_hit_rate` returned the literal 0.8 and `start_precomputation` was
// an empty function body, so no update was ever pre-computed at all.
// ---------------------------------------------------------------------------

#[test]
fn precomputation_hit_rate_is_none_before_any_step() {
    let config = LowLatencyConfig {
        enable_precomputation: true,
        ..base_config()
    };
    let optimizer = optimizer_with(0.1, config);
    assert_eq!(
        optimizer.get_performance_metrics().precomputation_hit_rate,
        None,
        "L2 regression: an engine that has never been consulted reported a hit rate \
         (the old code always returned 0.8)"
    );
}

#[test]
fn precomputation_hits_a_predictable_gradient_stream() {
    let config = LowLatencyConfig {
        enable_precomputation: true,
        approximation_tolerance: 0.01,
        ..base_config()
    };
    let mut optimizer = optimizer_with(0.1, config);

    // A constant gradient is perfectly predictable by linear extrapolation.
    let gradient = Array1::from_vec(vec![0.5f64, -0.25, 0.75]);
    for _ in 0..10 {
        optimizer
            .low_latency_step(&gradient)
            .expect("step must succeed");
    }

    let metrics = optimizer.get_performance_metrics();
    assert_eq!(metrics.precomputation_attempts, 10);
    let hit_rate = metrics
        .precomputation_hit_rate
        .expect("ten steps consulted the engine, so a rate must be available");
    assert!(
        hit_rate > 0.5,
        "L2 regression: a perfectly predictable gradient stream produced a hit rate of \
         {hit_rate}, so no real pre-computation is happening"
    );
}

#[test]
fn precomputation_never_serves_an_unpredictable_gradient_stream() {
    let config = LowLatencyConfig {
        enable_precomputation: true,
        approximation_tolerance: 0.001,
        ..base_config()
    };
    let mut optimizer = optimizer_with(0.1, config);

    // Sign-flipping gradients of growing magnitude: linear extrapolation
    // cannot predict these, so nothing may be served from the cache.
    for step in 0..10 {
        let sign = if step % 2 == 0 { 1.0 } else { -1.0 };
        let magnitude = 1.0 + step as f64;
        let gradient = Array1::from_vec(vec![sign * magnitude, -sign * magnitude, sign]);
        optimizer
            .low_latency_step(&gradient)
            .expect("step must succeed");
    }

    let hit_rate = optimizer
        .get_performance_metrics()
        .precomputation_hit_rate
        .expect("ten steps consulted the engine");
    assert_eq!(
        hit_rate, 0.0,
        "L2 regression: an unpredictable stream reported a non-zero hit rate ({hit_rate}), \
         so the rate is not being measured against the gradient that actually arrived"
    );
}

// ---------------------------------------------------------------------------
// L3: the "SIMD path" returned `gradient.clone()`, so the approximate path
// returned the gradient itself and applied no update whatsoever.
// ---------------------------------------------------------------------------

#[test]
fn chunked_fast_path_applies_a_real_update_instead_of_returning_the_gradient() {
    let config = LowLatencyConfig {
        enable_precomputation: false,
        enable_quantization: false,
        use_approximations: true,
        enable_simd: true,
        enable_lock_free: false,
        batch_threshold: 4,
        ..LowLatencyConfig::default()
    };
    let mut optimizer = optimizer_with(0.1, config);
    // Force the approximate path: level 0.5 keeps round(8 * 0.6) = 5 entries.
    optimizer.approximation_controller.approximation_level = 0.5;

    let gradient = Array1::from_vec(vec![1.0f64; 8]);
    let update = optimizer
        .low_latency_step(&gradient)
        .expect("step must succeed");

    assert_ne!(
        update.to_vec(),
        gradient.to_vec(),
        "L3 regression: the fast path returned the gradient unchanged"
    );
    for index in 0..5 {
        assert!(
            (update[index] - (-0.1)).abs() < 1e-12,
            "kept coordinate {index} must receive -lr*g, got {}",
            update[index]
        );
    }
    for index in 5..8 {
        assert!(
            update[index].abs() < 1e-12,
            "sparsified coordinate {index} must be untouched, got {}",
            update[index]
        );
    }
}

#[test]
fn approximation_accuracy_measures_the_applied_step_not_the_parameter_vector() {
    let mut optimizer = optimizer_with(0.1, base_config());
    let gradient = Array1::from_vec(vec![1.0f64, -2.0, 3.0]);
    optimizer
        .low_latency_step(&gradient)
        .expect("step must succeed");

    let accuracy = optimizer
        .get_performance_metrics()
        .approximation_accuracy
        .expect("one step was recorded");
    // A plain SGD step moves exactly along -gradient, so agreement is 1.
    // The old code compared the new parameter vector with the gradient
    // itself, which for a first step from the origin yields exactly -1.
    assert!(
        accuracy > 0.99,
        "expected the applied step to align with the descent direction, got {accuracy}"
    );
}

// ---------------------------------------------------------------------------
// L4: quantization divided by a zero scale.
// ---------------------------------------------------------------------------

#[test]
fn quantizing_an_all_zero_gradient_does_not_produce_nan() {
    let mut quantizer = GradientQuantizer::<f64>::new(8);
    let gradient = Array1::from_vec(vec![0.0f64; 4]);

    let quantized = quantizer
        .quantize(&gradient)
        .expect("an all-zero gradient must quantize successfully");
    for value in quantized.iter() {
        assert!(
            value.is_finite(),
            "L4 regression: zero-magnitude gradient quantized to {value} (scale was 0)"
        );
        assert_eq!(*value, 0.0);
    }
}

#[test]
fn quantizing_with_degenerate_bit_width_stays_finite() {
    // bits = 0 gave `2^0 - 1 = 0` levels, i.e. a zero scale and NaN output.
    let mut quantizer = GradientQuantizer::<f64>::new(0);
    let gradient = Array1::from_vec(vec![0.25f64, -0.5, 1.0]);

    let quantized = quantizer
        .quantize(&gradient)
        .expect("a degenerate bit width must be clamped, not divide by zero");
    for value in quantized.iter() {
        assert!(
            value.is_finite(),
            "L4 regression: bits=0 produced {value} instead of a finite value"
        );
    }
}

#[test]
fn quantization_carries_the_rounding_error_forward() {
    let mut quantizer = GradientQuantizer::<f64>::new(2);
    let gradient = Array1::from_vec(vec![1.0f64, 0.3, -0.7, 0.05]);

    let first = quantizer
        .quantize(&gradient)
        .expect("quantization must succeed");
    let residual = quantizer
        .error_accumulator
        .as_ref()
        .expect("error feedback must be recorded")
        .clone();
    assert!(
        residual.iter().any(|value| value.abs() > 1e-9),
        "a coarse 2-bit quantization of {:?} must leave a non-zero residual",
        gradient.to_vec()
    );

    let second = quantizer
        .quantize(&gradient)
        .expect("quantization must succeed");
    assert_ne!(
        first.to_vec(),
        second.to_vec(),
        "error feedback must change the next quantization of the same input"
    );
}

#[test]
fn quantizing_a_non_finite_gradient_is_an_error() {
    let mut quantizer = GradientQuantizer::<f64>::new(8);
    let gradient = Array1::from_vec(vec![1.0f64, f64::NAN]);
    assert!(
        quantizer.quantize(&gradient).is_err(),
        "a non-finite gradient must be reported instead of silently poisoning the scale"
    );
}

// ---------------------------------------------------------------------------
// L5: the pool allocated raw blocks with `std::alloc::alloc` and had no
// `Drop`, and `get_efficiency` divided by a `total_blocks` that can be zero.
// ---------------------------------------------------------------------------

#[test]
fn memory_pool_hands_out_and_reclaims_real_blocks() {
    let pool = FastMemoryPool::<f64>::new(4096 * 4, 4096).expect("pool creation must succeed");
    assert_eq!(pool.total_blocks, 4);

    let block = pool
        .acquire(16)
        .expect("a fresh pool must satisfy a request");
    assert!(block.capacity() >= pool.elements_per_block);
    assert_eq!(pool.checked_out.load(Ordering::Relaxed), 1);

    pool.release(block);
    assert_eq!(pool.checked_out.load(Ordering::Relaxed), 0);
    assert!(
        pool.get_efficiency() > 0.0,
        "the pool must report the high-water mark it actually reached"
    );
}

#[test]
fn empty_memory_pool_reports_zero_efficiency_not_nan() {
    // A pool smaller than one block yields zero blocks; the old
    // `1.0 - available/total` divided by zero here.
    let pool = FastMemoryPool::<f64>::new(128, 4096).expect("pool creation must succeed");
    assert_eq!(pool.total_blocks, 0);
    let efficiency = pool.get_efficiency();
    assert!(
        efficiency.is_finite() && efficiency == 0.0,
        "L5 regression: an empty pool reported {efficiency}"
    );
    assert!(pool.acquire(1).is_none());
    assert_eq!(pool.misses(), 1);
}

#[test]
fn oversized_requests_miss_the_pool_instead_of_panicking() {
    let pool = FastMemoryPool::<f64>::new(4096 * 2, 4096).expect("pool creation must succeed");
    assert!(pool.acquire(pool.elements_per_block + 1).is_none());
    assert_eq!(pool.misses(), 1);
}

// ---------------------------------------------------------------------------
// L6: latency violations were never counted, and the staging ring was
// constructed but never written to.
// ---------------------------------------------------------------------------

#[test]
fn latency_violations_are_counted_and_escalate_to_quantization() {
    let config = LowLatencyConfig {
        max_latency_us: 10,
        enable_quantization: false,
        ..base_config()
    };
    let mut optimizer = optimizer_with(0.1, config);
    assert_eq!(optimizer.get_performance_metrics().latency_violations, 0);

    optimizer
        .handle_latency_violation(Duration::from_micros(100))
        .expect("handling a violation must succeed");

    assert_eq!(
        optimizer.get_performance_metrics().latency_violations,
        1,
        "L6 regression: violations were never recorded"
    );
    assert!(
        optimizer.config.enable_quantization,
        "a 10x budget overrun must escalate to gradient quantization"
    );
    assert!(optimizer.quantizer.is_some());
    assert!(
        optimizer
            .get_performance_metrics()
            .current_approximation_level
            > 0.0
    );
}

#[test]
fn produced_updates_are_staged_in_fifo_order() {
    let config = LowLatencyConfig {
        enable_lock_free: true,
        ..base_config()
    };
    let mut optimizer = optimizer_with(0.1, config);
    let gradient = Array1::from_vec(vec![1.0f64, 1.0]);

    optimizer
        .low_latency_step(&gradient)
        .expect("step must succeed");
    optimizer
        .low_latency_step(&gradient)
        .expect("step must succeed");

    assert_eq!(
        optimizer.staged_update_count(),
        2,
        "L6 regression: the staging ring was never written to"
    );
    let first = optimizer
        .try_pop_staged_update()
        .expect("first staged update");
    let second = optimizer
        .try_pop_staged_update()
        .expect("second staged update");
    assert!((first[0] - (-0.1)).abs() < 1e-12);
    assert!((second[0] - (-0.2)).abs() < 1e-12);
    assert!(optimizer.try_pop_staged_update().is_none());
}

#[test]
fn staging_ring_drops_the_oldest_entry_when_full() {
    let mut ring = LockFreeBuffer::<f64>::new(2);
    ring.push(Array1::from_vec(vec![1.0]));
    ring.push(Array1::from_vec(vec![2.0]));
    ring.push(Array1::from_vec(vec![3.0]));

    assert_eq!(ring.len(), 2);
    assert_eq!(ring.pop().map(|v| v[0]), Some(2.0));
    assert_eq!(ring.pop().map(|v| v[0]), Some(3.0));
    assert!(ring.pop().is_none());
}

#[test]
fn percentiles_stay_in_bounds_for_a_single_sample() {
    let mut monitor = LatencyMonitor::new(4);
    monitor.record_latency(Duration::from_micros(7));
    assert_eq!(monitor.p50_latency, Duration::from_micros(7));
    assert_eq!(monitor.p95_latency, Duration::from_micros(7));
    assert_eq!(monitor.p99_latency, Duration::from_micros(7));
}
