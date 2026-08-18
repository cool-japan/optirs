// Regression tests for learning-rate scheduler defects.
//
// Every test in this file pins behaviour that was broken before the
// corresponding fix landed. The finding id from the production-readiness
// audit is quoted in each test's doc comment.

use optirs_core::schedulers::{
    AnnealStrategy, ConstantScheduler, CosineAnnealing, CyclicLR, CyclicMode,
    LearningRateScheduler, NoiseDistribution, NoiseInjectionScheduler, OneCycle, ReduceOnPlateau,
    ViTLayerDecay,
};

// ---------------------------------------------------------------------------
// F33 - CosineAnnealing
// ---------------------------------------------------------------------------

/// F33: with `warm_restart == false` the schedule must anneal monotonically
/// from `initial_lr` down to `min_lr` over `t_max` steps and then stay pinned
/// at `min_lr`. Before the fix the schedule silently restarted every `t_max`
/// steps regardless of the flag.
#[test]
fn f33_cosine_annealing_without_warm_restart_does_not_cycle() {
    let initial_lr = 0.1f64;
    let min_lr = 0.001f64;
    let t_max = 10usize;
    let mut scheduler = CosineAnnealing::new(initial_lr, min_lr, t_max, false);

    assert_eq!(
        scheduler.get_learning_rate(),
        initial_lr,
        "initial learning rate must be the configured value"
    );

    let mut previous = initial_lr;
    for step in 1..=(t_max * 2) {
        let lr = scheduler.step();
        assert!(lr.is_finite(), "lr must stay finite at step {step}");
        assert!(
            lr <= previous + 1e-12,
            "lr must be non-increasing without warm restarts (step {step}: {lr} > {previous})"
        );
        assert!(
            lr >= min_lr - 1e-12,
            "lr must never fall below min_lr (step {step}: {lr})"
        );
        previous = lr;
    }

    // Once the cycle is exhausted the schedule must remain at min_lr.
    for step in 0..5 {
        let lr = scheduler.step();
        assert!(
            (lr - min_lr).abs() < 1e-12,
            "lr must stay at min_lr after t_max (extra step {step}: {lr})"
        );
    }
}

/// F33: with `warm_restart == true` the schedule must actually restart.
#[test]
fn f33_cosine_annealing_with_warm_restart_restarts() {
    let initial_lr = 0.1f64;
    let min_lr = 0.001f64;
    let t_max = 10usize;
    let mut scheduler = CosineAnnealing::new(initial_lr, min_lr, t_max, true);

    let mut lrs = Vec::new();
    for _ in 0..(t_max * 2) {
        lrs.push(scheduler.step());
    }

    // Late in the first cycle the LR is low; right after the restart it must
    // jump back up.
    let before_restart = lrs[t_max - 2];
    let after_restart = lrs[t_max - 1];
    assert!(
        after_restart > before_restart,
        "warm restart must raise the learning rate ({after_restart} <= {before_restart})"
    );
    assert!(
        (after_restart - initial_lr).abs() < 1e-12,
        "warm restart must return to initial_lr, got {after_restart}"
    );
}

/// F33: `t_max == 0` used to produce a 0/0 division and therefore NaN.
#[test]
fn f33_cosine_annealing_zero_t_max_stays_finite() {
    let mut scheduler = CosineAnnealing::new(0.1f64, 0.001f64, 0, false);
    for step in 0..5 {
        let lr = scheduler.step();
        assert!(
            lr.is_finite(),
            "lr must be finite with t_max == 0 (step {step}): {lr}"
        );
    }
    assert!(scheduler.get_learning_rate().is_finite());
}

// ---------------------------------------------------------------------------
// F34 - OneCycle
// ---------------------------------------------------------------------------

/// F34: stepping past `total_steps` used to yield ever more negative learning
/// rates with the linear annealing strategy.
#[test]
fn f34_one_cycle_never_returns_negative_lr_past_total_steps() {
    let total_steps = 100usize;
    let final_lr = 1e-5f64;
    let mut scheduler = OneCycle::new(1e-4f64, 1e-3f64, total_steps, 0.25)
        .with_anneal_strategy(AnnealStrategy::Linear)
        .with_final_lr(final_lr);

    for step in 1..=(total_steps * 2) {
        let lr = scheduler.step();
        assert!(lr.is_finite(), "lr must be finite at step {step}: {lr}");
        assert!(
            lr >= final_lr - 1e-15,
            "lr must never drop below final_lr (step {step}: {lr})"
        );
        if step >= total_steps {
            assert!(
                (lr - final_lr).abs() < 1e-12,
                "lr must saturate at final_lr past total_steps (step {step}: {lr})"
            );
        }
    }
}

/// F34: the cosine strategy used to bounce back up past `total_steps`.
#[test]
fn f34_one_cycle_cosine_saturates_past_total_steps() {
    let total_steps = 50usize;
    let final_lr = 1e-6f64;
    let mut scheduler = OneCycle::new(1e-4f64, 1e-3f64, total_steps, 0.3).with_final_lr(final_lr);

    for _ in 0..total_steps {
        scheduler.step();
    }
    let at_end = scheduler.get_learning_rate();
    for step in 0..total_steps {
        let lr = scheduler.step();
        assert!(
            (lr - at_end).abs() < 1e-12,
            "cosine one-cycle must saturate past total_steps (extra step {step}: {lr} vs {at_end})"
        );
    }
}

/// F34: `warmup_frac > 1.0` used to underflow `total_steps - warmup_steps`.
#[test]
fn f34_one_cycle_out_of_range_warmup_frac_is_safe() {
    let mut scheduler = OneCycle::new(1e-4f64, 1e-3f64, 100, 1.5);
    for step in 0..200 {
        let lr = scheduler.step();
        assert!(lr.is_finite(), "lr must be finite at step {step}: {lr}");
        assert!(lr >= 0.0, "lr must be non-negative at step {step}: {lr}");
    }
}

/// F34: `total_steps == 0` used to produce a 0/0 division.
#[test]
fn f34_one_cycle_zero_total_steps_is_safe() {
    let mut scheduler = OneCycle::new(1e-4f64, 1e-3f64, 0, 0.25);
    for step in 0..10 {
        let lr = scheduler.step();
        assert!(lr.is_finite(), "lr must be finite at step {step}: {lr}");
        assert!(lr >= 0.0, "lr must be non-negative at step {step}: {lr}");
    }
    assert!(scheduler.get_percentage_complete().is_finite());
}

// ---------------------------------------------------------------------------
// F35 - CyclicLR
// ---------------------------------------------------------------------------

/// F35: the `ExpRange` gamma decay must be driven by the *global* step count.
/// Before the fix it used the within-cycle index, so every cycle peaked at the
/// exact same amplitude and the exponential decay never took effect.
#[test]
fn f35_cyclic_lr_exp_range_decays_across_cycles() {
    let base_lr = 0.001f64;
    let max_lr = 0.01f64;
    let step_size = 5usize;
    let gamma = 0.9f64;

    let mut scheduler = CyclicLR::exp_range(base_lr, max_lr, step_size, gamma);

    // Peak of the first cycle is at global step `step_size`.
    for _ in 0..step_size {
        scheduler.step();
    }
    let first_peak = scheduler.get_learning_rate();

    // Peak of the second cycle is at global step `3 * step_size`.
    for _ in 0..(2 * step_size) {
        scheduler.step();
    }
    let second_peak = scheduler.get_learning_rate();

    assert!(
        second_peak < first_peak,
        "ExpRange peaks must decay across cycles ({second_peak} >= {first_peak})"
    );

    let expected_first = base_lr + (max_lr - base_lr) * gamma.powi(step_size as i32);
    let expected_second = base_lr + (max_lr - base_lr) * gamma.powi(3 * step_size as i32);
    assert!(
        (first_peak - expected_first).abs() < 1e-12,
        "first peak {first_peak} != expected {expected_first}"
    );
    assert!(
        (second_peak - expected_second).abs() < 1e-12,
        "second peak {second_peak} != expected {expected_second}"
    );
}

/// F35: `step_size == 0` used to panic with a divide-by-zero.
#[test]
fn f35_cyclic_lr_zero_step_size_does_not_panic() {
    let scheduler = CyclicLR::triangular(0.001f64, 0.01f64, 0);
    let lr = scheduler.get_learning_rate();
    assert!(
        lr.is_finite(),
        "lr must be finite with step_size == 0: {lr}"
    );

    let scheduler2 = CyclicLR::triangular2(0.001f64, 0.01f64, 0);
    assert!(scheduler2.get_learning_rate().is_finite());

    let scheduler3 = CyclicLR::exp_range(0.001f64, 0.01f64, 0, 0.99);
    assert!(scheduler3.get_learning_rate().is_finite());
}

// ---------------------------------------------------------------------------
// F36 - NoiseInjectionScheduler
// ---------------------------------------------------------------------------

/// F36: `get_learning_rate()` must be idempotent between `step()` calls.
/// Before the fix it re-sampled fresh noise from a brand new thread RNG on
/// every read, so two consecutive reads disagreed.
#[test]
fn f36_noise_injection_get_learning_rate_is_idempotent() {
    let mut scheduler = NoiseInjectionScheduler::new(
        ConstantScheduler::new(0.1f64),
        NoiseDistribution::Uniform {
            min: -0.02,
            max: 0.02,
        },
        0.001,
    );

    let a = scheduler.get_learning_rate();
    let b = scheduler.get_learning_rate();
    assert_eq!(
        a, b,
        "two consecutive get_learning_rate() calls must agree ({a} vs {b})"
    );

    for step in 0..20 {
        let stepped = scheduler.step();
        let read_back = scheduler.get_learning_rate();
        assert_eq!(
            stepped, read_back,
            "step() and get_learning_rate() must agree at step {step}"
        );
        assert_eq!(
            read_back,
            scheduler.get_learning_rate(),
            "get_learning_rate() must be idempotent at step {step}"
        );
    }
}

/// F36: Box-Muller must never see `u1 == 0.0` (ln(0) == -inf).
#[test]
fn f36_noise_injection_gaussian_is_always_finite() {
    let mut scheduler = NoiseInjectionScheduler::new(
        ConstantScheduler::new(0.1f64),
        NoiseDistribution::Gaussian {
            mean: 0.0,
            std_dev: 0.01,
        },
        0.001,
    );

    for step in 0..2000 {
        let lr = scheduler.step();
        assert!(
            lr.is_finite(),
            "gaussian noise produced non-finite lr at step {step}: {lr}"
        );
    }
}

// ---------------------------------------------------------------------------
// F37 - ViTLayerDecay
// ---------------------------------------------------------------------------

/// F37: with no warmup configured the scheduler must report the configured
/// base learning rate before the first `step()`. Before the fix it reported 0.
#[test]
fn f37_vit_layer_decay_initial_lr_without_warmup() {
    let scheduler = ViTLayerDecay::<f64>::new(0.001, 0.75, 12, 0, 1000);
    assert!(
        (scheduler.get_learning_rate() - 0.001).abs() < 1e-12,
        "initial lr must be base_lr when warmup_steps == 0, got {}",
        scheduler.get_learning_rate()
    );

    let built = ViTLayerDecay::<f64>::builder()
        .base_lr(0.005)
        .num_layers(4)
        .total_steps(100)
        .build();
    assert!(
        (built.get_learning_rate() - 0.005).abs() < 1e-12,
        "builder default (no warmup) must start at base_lr, got {}",
        built.get_learning_rate()
    );

    // Per-layer rates must be live before the first step as well.
    let rates = built.get_all_layer_rates();
    assert_eq!(rates.len(), 4);
    assert!(rates[3] > rates[0]);
    assert!((rates[3] - 0.005).abs() < 1e-12);
}

/// F37 companion: a configured warmup phase must still ramp from zero -
/// the fix must not break linear-warmup semantics.
#[test]
fn f37_vit_layer_decay_with_warmup_still_ramps_from_zero() {
    let mut scheduler = ViTLayerDecay::<f64>::new(0.001, 0.75, 12, 100, 1000);
    assert!(
        scheduler.get_learning_rate().abs() < 1e-15,
        "with warmup configured the schedule must start at 0"
    );

    let mut previous = 0.0;
    for i in 0..100 {
        let lr = scheduler.step();
        assert!(
            lr > previous,
            "warmup must be monotonically increasing at step {i}"
        );
        previous = lr;
    }
    assert!((scheduler.get_learning_rate() - 0.001).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// F87 - ReduceOnPlateau through the trait
// ---------------------------------------------------------------------------

/// F87: `ReduceOnPlateau` used to be completely inert when driven through the
/// `LearningRateScheduler` trait because the trait had no metric-aware entry
/// point.
#[test]
fn f87_reduce_on_plateau_drives_through_trait_object() {
    let mut scheduler: Box<dyn LearningRateScheduler<f64>> =
        Box::new(ReduceOnPlateau::new(0.1f64, 0.1, 2, 1e-6));

    let initial = scheduler.get_learning_rate();
    assert!((initial - 0.1).abs() < 1e-12);

    // Improve once, then plateau.
    scheduler.step_with_metric(1.0);
    for _ in 0..6 {
        scheduler.step_with_metric(1.0);
    }

    assert!(
        scheduler.get_learning_rate() < initial,
        "plateau must reduce the learning rate through the trait, got {}",
        scheduler.get_learning_rate()
    );
}

/// F87: the trait default must be a no-op forwarding to `step()` so existing
/// implementors keep working unchanged.
#[test]
fn f87_trait_default_step_with_metric_forwards_to_step() {
    let mut scheduler: Box<dyn LearningRateScheduler<f64>> =
        Box::new(ConstantScheduler::new(0.05f64));
    assert!((scheduler.step_with_metric(123.0) - 0.05).abs() < 1e-12);
    assert!((scheduler.get_learning_rate() - 0.05).abs() < 1e-12);
}

/// F87: plain `step()` on `ReduceOnPlateau` must not corrupt state.
#[test]
fn f87_reduce_on_plateau_plain_step_is_inert() {
    let mut scheduler = ReduceOnPlateau::new(0.1f64, 0.1, 2, 1e-6);
    for _ in 0..100 {
        assert!((scheduler.step() - 0.1).abs() < 1e-12);
    }
    assert_eq!(scheduler.stagnation_count(), 0);
    assert!(scheduler.best_metric().is_none());
}

/// F87: a cooldown must suppress back-to-back reductions.
#[test]
fn f87_reduce_on_plateau_cooldown_blocks_consecutive_reductions() {
    // patience = 1 so any non-improving metric would reduce immediately.
    let mut without = ReduceOnPlateau::new(1.0f64, 0.5, 1, 1e-9);
    without.step_with_metric(1.0);
    let without_lrs: Vec<f64> = (0..4).map(|_| without.step_with_metric(1.0)).collect();
    assert_eq!(without_lrs, vec![0.5, 0.25, 0.125, 0.0625]);

    let mut with = ReduceOnPlateau::new(1.0f64, 0.5, 1, 1e-9).with_cooldown(2);
    with.step_with_metric(1.0);
    let with_lrs: Vec<f64> = (0..4).map(|_| with.step_with_metric(1.0)).collect();
    assert_eq!(with_lrs, vec![0.5, 0.5, 0.5, 0.25]);
}

// ---------------------------------------------------------------------------
// Validating constructors (`try_new`)
// ---------------------------------------------------------------------------

/// F33/F34/F35: degenerate configurations are rejected up front by the validating
/// constructors, while the infallible constructors clamp them into a safe range.
#[test]
fn validating_constructors_reject_degenerate_configs() {
    // F33
    assert!(CosineAnnealing::try_new(0.1f64, 0.001, 0, false).is_err());
    assert!(CosineAnnealing::try_new(0.1f64, 0.5, 10, false).is_err());
    assert!(CosineAnnealing::try_new(0.1f64, 0.001, 10, false).is_ok());

    // F34
    assert!(OneCycle::try_new(1e-4f64, 1e-3, 0, 0.25).is_err());
    assert!(OneCycle::try_new(1e-4f64, 1e-3, 100, 0.0).is_err());
    assert!(OneCycle::try_new(1e-4f64, 1e-3, 100, 1.0).is_err());
    assert!(OneCycle::try_new(1e-4f64, 1e-3, 100, 1.5).is_err());
    assert!(OneCycle::try_new(1e-4f64, 1e-3, 100, 0.25).is_ok());

    // F35
    assert!(CyclicLR::try_new(0.001f64, 0.01, 0, CyclicMode::Triangular).is_err());
    assert!(CyclicLR::try_new(0.001f64, 0.01, 10, CyclicMode::ExpRange(0.0)).is_err());
    assert!(CyclicLR::try_new(0.001f64, 0.01, 10, CyclicMode::ExpRange(0.99)).is_ok());
}

/// F36: the configured seed must actually drive the RNG, so two schedulers built with the
/// same seed produce identical learning-rate sequences (and a different seed does not).
#[test]
fn f36_noise_injection_seed_is_honoured() {
    let dist = NoiseDistribution::Gaussian {
        mean: 0.0f64,
        std_dev: 0.01,
    };

    let mut a = NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 4242);
    let mut b =
        NoiseInjectionScheduler::new(ConstantScheduler::new(0.1), dist, 0.001).with_seed(4242);

    assert_eq!(
        a.get_learning_rate(),
        b.get_learning_rate(),
        "the cached step-0 noise must already be seed-derived"
    );

    let seq_a: Vec<f64> = (0..64).map(|_| a.step()).collect();
    let seq_b: Vec<f64> = (0..64).map(|_| b.step()).collect();
    assert_eq!(seq_a, seq_b, "same seed must give the same sequence");

    let mut c = NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 99);
    let seq_c: Vec<f64> = (0..64).map(|_| c.step()).collect();
    assert_ne!(
        seq_a, seq_c,
        "a different seed must give a different sequence"
    );
}

/// F36: `reset()` must put the scheduler back on the same deterministic stream.
#[test]
fn f36_noise_injection_reset_is_reproducible() {
    let dist = NoiseDistribution::Uniform {
        min: -0.02f64,
        max: 0.02,
    };
    let mut scheduler =
        NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 31337);

    let first: Vec<f64> = (0..24).map(|_| scheduler.step()).collect();
    scheduler.reset();
    let second: Vec<f64> = (0..24).map(|_| scheduler.step()).collect();
    assert_eq!(first, second);
}

/// The trait must stay object safe: `Box<dyn LearningRateScheduler<A>>` is used by
/// `optirs_core::unified_api`, so the new default method must not add generics.
#[test]
fn learning_rate_scheduler_stays_object_safe() {
    let schedulers: Vec<Box<dyn LearningRateScheduler<f64>>> = vec![
        Box::new(ConstantScheduler::new(0.01f64)),
        Box::new(CosineAnnealing::new(0.1f64, 0.001, 10, false)),
        Box::new(OneCycle::new(1e-4f64, 1e-3, 100, 0.25)),
        Box::new(CyclicLR::triangular(0.001f64, 0.01, 10)),
        Box::new(ReduceOnPlateau::new(0.1f64, 0.1, 2, 1e-6)),
        Box::new(ViTLayerDecay::<f64>::new(0.001, 0.75, 12, 0, 100)),
    ];

    for mut scheduler in schedulers {
        assert!(scheduler.get_learning_rate().is_finite());
        assert!(scheduler.step().is_finite());
        assert!(scheduler.step_with_metric(1.0).is_finite());
        scheduler.reset();
    }
}
