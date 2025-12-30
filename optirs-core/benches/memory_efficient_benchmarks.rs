//! Memory-efficient optimization benchmarks
//!
//! This benchmark suite demonstrates memory efficiency improvements
//! through gradient accumulation and chunked processing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use optirs_core::memory_efficient_optimizer::{
    ChunkedOptimizer, GradientAccumulator as MemoryEfficientGradientAccumulator,
    MemoryUsageEstimator,
};
use optirs_core::optimizers::{Optimizer, SGD};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

/// Benchmark gradient accumulation performance
fn bench_gradient_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gradient_Accumulation");

    for num_accumulations in [4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*num_accumulations as u64));

        let size = 10000;
        let gradients: Vec<Array1<f32>> = (0..*num_accumulations)
            .map(|_| Array1::from_elem(size, 0.1))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_accumulations),
            num_accumulations,
            |b, &_num_accumulations| {
                b.iter(|| {
                    let mut accumulator = MemoryEfficientGradientAccumulator::<f32>::new(size);
                    for grad in &gradients {
                        accumulator.accumulate(&grad.view()).unwrap();
                    }
                    let avg = accumulator.average().unwrap();
                    black_box(avg)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark chunked vs full optimization
fn bench_chunked_vs_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("Chunked_vs_Full");

    for size in [10_000, 50_000, 100_000, 500_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_elem(*size, 1.0);
        let gradients = Array1::from_elem(*size, 0.1);

        // Full optimization (baseline)
        group.bench_with_input(BenchmarkId::new("Full", size), size, |b, &_size| {
            let mut optimizer = SGD::new(0.01);
            b.iter(|| {
                let result = optimizer
                    .step(black_box(&params), black_box(&gradients))
                    .unwrap();
                black_box(result)
            });
        });

        // Chunked optimization (10K chunks)
        group.bench_with_input(BenchmarkId::new("Chunked_10K", size), size, |b, &_size| {
            let optimizer = SGD::new(0.01);
            let mut chunked_opt = ChunkedOptimizer::new(optimizer, Some(10_000));
            b.iter(|| {
                let result = chunked_opt
                    .step_chunked(black_box(&params), black_box(&gradients))
                    .unwrap();
                black_box(result)
            });
        });

        // Chunked optimization (50K chunks)
        group.bench_with_input(BenchmarkId::new("Chunked_50K", size), size, |b, &_size| {
            let optimizer = SGD::new(0.01);
            let mut chunked_opt = ChunkedOptimizer::new(optimizer, Some(50_000));
            b.iter(|| {
                let result = chunked_opt
                    .step_chunked(black_box(&params), black_box(&gradients))
                    .unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark accumulation step scaling
fn bench_accumulation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Accumulation_Scaling");

    let size = 10000;

    for num_steps in [2, 4, 8, 16, 32, 64].iter() {
        group.throughput(Throughput::Elements(*num_steps as u64));

        let gradients: Vec<Array1<f32>> = (0..*num_steps)
            .map(|_| Array1::from_elem(size, 0.1))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_steps),
            num_steps,
            |b, &_num_steps| {
                b.iter(|| {
                    let mut accumulator = MemoryEfficientGradientAccumulator::<f32>::new(size);
                    for grad in &gradients {
                        accumulator.accumulate(&grad.view()).unwrap();
                    }
                    accumulator.is_ready(*num_steps);
                    let avg = accumulator.average().unwrap();
                    black_box(avg)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different gradient sizes
fn bench_gradient_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gradient_Sizes");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let gradients: Vec<Array1<f32>> = (0..8).map(|_| Array1::from_elem(*size, 0.1)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let mut accumulator = MemoryEfficientGradientAccumulator::<f32>::new(*size);
                for grad in &gradients {
                    accumulator.accumulate(&grad.view()).unwrap();
                }
                let avg = accumulator.average().unwrap();
                black_box(avg)
            });
        });
    }

    group.finish();
}

/// Benchmark chunk size impact
fn bench_chunk_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("Chunk_Size_Impact");

    let size = 100_000;
    let params = Array1::from_elem(size, 1.0);
    let gradients = Array1::from_elem(size, 0.1);

    for chunk_size in [1_000, 5_000, 10_000, 25_000, 50_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*chunk_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            chunk_size,
            |b, &_chunk_size| {
                let optimizer = SGD::new(0.01);
                let mut chunked_opt = ChunkedOptimizer::new(optimizer, Some(*chunk_size));
                b.iter(|| {
                    let result = chunked_opt
                        .step_chunked(black_box(&params), black_box(&gradients))
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory estimation overhead
fn bench_memory_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory_Estimation");

    group.bench_function("SGD_Estimation", |b| {
        b.iter(|| {
            for num_params in [1_000, 10_000, 100_000, 1_000_000].iter() {
                let mem = MemoryUsageEstimator::sgd(*num_params, 4);
                black_box(mem);
            }
        });
    });

    group.bench_function("Adam_Estimation", |b| {
        b.iter(|| {
            for num_params in [1_000, 10_000, 100_000, 1_000_000].iter() {
                let mem = MemoryUsageEstimator::adam(*num_params, 4);
                black_box(mem);
            }
        });
    });

    group.bench_function("Chunk_Size_Recommendation", |b| {
        b.iter(|| {
            for total_params in [1_000_000, 10_000_000, 100_000_000].iter() {
                let chunk_size = MemoryUsageEstimator::recommend_chunk_size(
                    *total_params,
                    1_000_000_000, // 1GB
                    4,             // f32
                    4,             // Adam multiplier
                );
                black_box(chunk_size);
            }
        });
    });

    group.bench_function("Peak_Memory_Estimation", |b| {
        b.iter(|| {
            for num_params in [1_000_000, 10_000_000, 100_000_000].iter() {
                let peak = MemoryUsageEstimator::estimate_peak_memory(
                    *num_params,
                    32,  // batch size
                    512, // sequence length
                    4,   // f32
                    "adam",
                );
                black_box(peak);
            }
        });
    });

    group.finish();
}

/// Benchmark accumulator reset performance
fn bench_accumulator_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("Accumulator_Reset");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let mut accumulator = MemoryEfficientGradientAccumulator::<f32>::new(*size);
            let grad = Array1::from_elem(*size, 0.1);

            b.iter(|| {
                // Accumulate
                accumulator.accumulate(&grad.view()).unwrap();
                accumulator.accumulate(&grad.view()).unwrap();
                accumulator.accumulate(&grad.view()).unwrap();
                accumulator.accumulate(&grad.view()).unwrap();

                // Average (includes reset)
                let avg = accumulator.average().unwrap();
                black_box(avg)
            });
        });
    }

    group.finish();
}

/// Benchmark optimizer comparison for chunked processing
fn bench_optimizer_types_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimizer_Types_Chunked");

    let size = 100_000;
    let params = Array1::from_elem(size, 1.0);
    let gradients = Array1::from_elem(size, 0.1);
    let chunk_size = 10_000;

    group.throughput(Throughput::Elements(size as u64));

    // SGD
    group.bench_function("SGD_Chunked", |b| {
        let optimizer = SGD::new(0.01);
        let mut chunked_opt = ChunkedOptimizer::new(optimizer, Some(chunk_size));
        b.iter(|| {
            let result = chunked_opt
                .step_chunked(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    // SGD with momentum
    group.bench_function("SGD_Momentum_Chunked", |b| {
        let optimizer = SGD::new_with_config(0.01, 0.9, 0.0);
        let mut chunked_opt = ChunkedOptimizer::new(optimizer, Some(chunk_size));
        b.iter(|| {
            let result = chunked_opt
                .step_chunked(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gradient_accumulation,
    bench_chunked_vs_full,
    bench_accumulation_scaling,
    bench_gradient_sizes,
    bench_chunk_size_impact,
    bench_memory_estimation,
    bench_accumulator_reset,
    bench_optimizer_types_chunked,
);

criterion_main!(benches);
