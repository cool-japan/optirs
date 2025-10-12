//! Comprehensive optimizer benchmarks
//!
//! This benchmark suite measures the performance of all OptiRS optimizers
//! with various parameter sizes to demonstrate:
//! 1. Throughput (operations per second)
//! 2. Latency (time per operation)
//! 3. Scalability (performance vs parameter count)
//! 4. Memory efficiency

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use optirs_core::optimizers::{Adam, AdamW, Optimizer, SGD};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

/// Benchmark SGD optimizer with various parameter sizes
fn bench_sgd(c: &mut Criterion) {
    let mut group = c.benchmark_group("SGD");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let params = Array1::zeros(size);
            let gradients = Array1::from_elem(size, 0.1);
            let mut optimizer = SGD::new(0.01);

            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark SGD with momentum
fn bench_sgd_momentum(c: &mut Criterion) {
    let mut group = c.benchmark_group("SGD_Momentum");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let params = Array1::zeros(size);
            let gradients = Array1::from_elem(size, 0.1);
            let mut optimizer = SGD::new_with_config(0.01, 0.9, 0.0001);

            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark Adam optimizer
fn bench_adam(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adam");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let params = Array1::zeros(size);
            let gradients = Array1::from_elem(size, 0.1);
            let mut optimizer = Adam::new(0.001);

            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark AdamW optimizer
fn bench_adamw(c: &mut Criterion) {
    let mut group = c.benchmark_group("AdamW");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let params = Array1::zeros(size);
            let gradients = Array1::from_elem(size, 0.1);
            let mut optimizer = AdamW::new(0.001);

            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Compare all optimizers on the same problem
fn bench_optimizer_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimizer_Comparison");
    let size = 10_000;

    group.throughput(Throughput::Elements(size as u64));

    let params = Array1::zeros(size);
    let gradients = Array1::from_elem(size, 0.1);

    group.bench_function("SGD", |b| {
        let mut optimizer = SGD::new(0.01);
        b.iter(|| {
            let result = optimizer.step(black_box(&params), black_box(&gradients));
            black_box(result)
        });
    });

    group.bench_function("SGD_Momentum", |b| {
        let mut optimizer = SGD::new_with_config(0.01, 0.9, 0.0001);
        b.iter(|| {
            let result = optimizer.step(black_box(&params), black_box(&gradients));
            black_box(result)
        });
    });

    group.bench_function("Adam", |b| {
        let mut optimizer = Adam::new(0.001);
        b.iter(|| {
            let result = optimizer.step(black_box(&params), black_box(&gradients));
            black_box(result)
        });
    });

    group.bench_function("AdamW", |b| {
        let mut optimizer = AdamW::new(0.001);
        b.iter(|| {
            let result = optimizer.step(black_box(&params), black_box(&gradients));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark memory allocation overhead
fn bench_cold_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cold_Start");
    let size = 10_000;

    let params = Array1::zeros(size);
    let gradients = Array1::from_elem(size, 0.1);

    group.bench_function("Adam_First_Step", |b| {
        b.iter(|| {
            let mut optimizer = Adam::new(0.001);
            let result = optimizer.step(black_box(&params), black_box(&gradients));
            black_box(result)
        });
    });

    group.bench_function("Adam_Subsequent_Steps", |b| {
        let mut optimizer = Adam::new(0.001);
        // Warm up - allocate state
        let _ = optimizer.step(&params, &gradients);

        b.iter(|| {
            let result = optimizer.step(black_box(&params), black_box(&gradients));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark convergence speed (multiple steps)
fn bench_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("Convergence");
    let size = 1000;
    let steps = 100;

    group.bench_function("SGD_100_Steps", |b| {
        b.iter(|| {
            let mut params = Array1::zeros(size);
            let gradients = Array1::from_elem(size, 0.1);
            let mut optimizer = SGD::new(0.01);

            for _ in 0..steps {
                params = optimizer.step(&params, &gradients).unwrap();
            }
            black_box(params)
        });
    });

    group.bench_function("Adam_100_Steps", |b| {
        b.iter(|| {
            let mut params = Array1::zeros(size);
            let gradients = Array1::from_elem(size, 0.1);
            let mut optimizer = Adam::new(0.001);

            for _ in 0..steps {
                params = optimizer.step(&params, &gradients).unwrap();
            }
            black_box(params)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sgd,
    bench_sgd_momentum,
    bench_adam,
    bench_adamw,
    bench_optimizer_comparison,
    bench_cold_start,
    bench_convergence,
);

criterion_main!(benches);
