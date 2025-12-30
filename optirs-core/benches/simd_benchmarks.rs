//! SIMD acceleration benchmarks
//!
//! This benchmark suite compares SIMD-accelerated vs scalar implementations
//! of optimizer operations to demonstrate performance improvements.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use optirs_core::optimizers::{Optimizer, SimdSGD, SGD};
use optirs_core::simd_optimizer::SimdOptimizer;
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

/// Benchmark SIMD vs scalar SGD updates
fn bench_sgd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("SGD_Comparison");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_vec((0..*size).map(|i| i as f32).collect());
        let gradients = Array1::from_elem(*size, 0.1f32);

        // Scalar SGD (general optimizer)
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |b, &_size| {
            let mut optimizer = SGD::new(0.01);
            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });

        // SIMD-accelerated SGD
        group.bench_with_input(BenchmarkId::new("SIMD", size), size, |b, &_size| {
            let mut optimizer = SimdSGD::new(0.01);
            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark SIMD vs scalar momentum updates
fn bench_momentum_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Momentum_Comparison");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_vec((0..*size).map(|i| i as f32).collect());
        let gradients = Array1::from_elem(*size, 0.1f32);

        // Scalar SGD with momentum
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |b, &_size| {
            let mut optimizer = SGD::new_with_config(0.01, 0.9, 0.0);
            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });

        // SIMD SGD with momentum
        group.bench_with_input(BenchmarkId::new("SIMD", size), size, |b, &_size| {
            let mut optimizer = SimdSGD::new_with_config(0.01, 0.9, 0.0);
            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark raw SIMD operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD_Operations");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_vec((0..*size).map(|i| i as f32).collect());
        let gradients = Array1::from_elem(*size, 0.1f32);

        // SIMD SGD update
        group.bench_with_input(BenchmarkId::new("SGD_Update", size), size, |b, &_size| {
            b.iter(|| {
                let result = f32::simd_sgd_update(
                    black_box(&params.view()),
                    black_box(&gradients.view()),
                    black_box(0.01),
                );
                black_box(result)
            });
        });

        // SIMD momentum update
        let velocity = Array1::zeros(*size);
        group.bench_with_input(
            BenchmarkId::new("Momentum_Update", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    let result = f32::simd_momentum_update(
                        black_box(&params.view()),
                        black_box(&gradients.view()),
                        black_box(&velocity.view()),
                        black_box(0.01),
                        black_box(0.9),
                    );
                    black_box(result)
                });
            },
        );

        // SIMD weight decay
        group.bench_with_input(BenchmarkId::new("Weight_Decay", size), size, |b, &_size| {
            b.iter(|| {
                let result = f32::simd_weight_decay(
                    black_box(&gradients.view()),
                    black_box(&params.view()),
                    black_box(0.01),
                );
                black_box(result)
            });
        });

        // SIMD gradient norm
        group.bench_with_input(
            BenchmarkId::new("Gradient_Norm", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    let result = f32::simd_gradient_norm(black_box(&gradients.view()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD Adam operations
fn bench_adam_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adam_SIMD_Operations");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let m = Array1::from_elem(*size, 0.01f32);
        let v = Array1::from_elem(*size, 0.001f32);
        let gradients = Array1::from_elem(*size, 0.1f32);

        // First moment update
        group.bench_with_input(BenchmarkId::new("First_Moment", size), size, |b, &_size| {
            b.iter(|| {
                let result = f32::simd_adam_first_moment(
                    black_box(&m.view()),
                    black_box(&gradients.view()),
                    black_box(0.9),
                );
                black_box(result)
            });
        });

        // Second moment update
        group.bench_with_input(
            BenchmarkId::new("Second_Moment", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    let result = f32::simd_adam_second_moment(
                        black_box(&v.view()),
                        black_box(&gradients.view()),
                        black_box(0.999),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD f32 vs f64 performance
fn bench_dtype_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dtype_Comparison");
    let size = 10_000;

    group.throughput(Throughput::Elements(size as u64));

    // f32 SIMD
    let params_f32 = Array1::from_vec((0..size).map(|i| i as f32).collect());
    let gradients_f32 = Array1::from_elem(size, 0.1f32);

    group.bench_function("f32_SGD", |b| {
        b.iter(|| {
            let result = f32::simd_sgd_update(
                black_box(&params_f32.view()),
                black_box(&gradients_f32.view()),
                black_box(0.01),
            );
            black_box(result)
        });
    });

    // f64 SIMD
    let params_f64 = Array1::from_vec((0..size).map(|i| i as f64).collect());
    let gradients_f64 = Array1::from_elem(size, 0.1f64);

    group.bench_function("f64_SGD", |b| {
        b.iter(|| {
            let result = f64::simd_sgd_update(
                black_box(&params_f64.view()),
                black_box(&gradients_f64.view()),
                black_box(0.01),
            );
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark scaling behavior
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD_Scaling");

    // Test with various sizes to see scaling behavior
    for size in [
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ]
    .iter()
    {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_vec((0..*size).map(|i| i as f32).collect());
        let gradients = Array1::from_elem(*size, 0.1f32);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let mut optimizer = SimdSGD::new(0.01);
            b.iter(|| {
                let result = optimizer.step(black_box(&params), black_box(&gradients));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sgd_comparison,
    bench_momentum_comparison,
    bench_simd_operations,
    bench_adam_simd_operations,
    bench_dtype_comparison,
    bench_scaling,
);

criterion_main!(benches);
