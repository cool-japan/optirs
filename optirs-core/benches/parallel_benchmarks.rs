//! Parallel processing benchmarks
//!
//! This benchmark suite compares sequential vs parallel processing
//! for multiple parameter groups to demonstrate multi-core speedup.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use optirs_core::optimizers::{Optimizer, SGD};
use optirs_core::parallel_optimizer::{parallel_step_array1, ParallelOptimizer};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

/// Benchmark sequential vs parallel parameter group processing
fn bench_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sequential_vs_Parallel");

    for num_groups in [2, 4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*num_groups as u64));

        let params_list: Vec<Array1<f32>> = (0..*num_groups)
            .map(|_| Array1::from_elem(1000, 1.0))
            .collect();
        let grads_list: Vec<Array1<f32>> = (0..*num_groups)
            .map(|_| Array1::from_elem(1000, 0.1))
            .collect();

        // Sequential processing
        group.bench_with_input(
            BenchmarkId::new("Sequential", num_groups),
            num_groups,
            |b, &_num_groups| {
                let mut optimizer = SGD::new(0.01);
                b.iter(|| {
                    let mut results = Vec::with_capacity(params_list.len());
                    for (params, grads) in params_list.iter().zip(grads_list.iter()) {
                        let result = optimizer.step(black_box(params), black_box(grads)).unwrap();
                        results.push(result);
                    }
                    black_box(results)
                });
            },
        );

        // Parallel processing
        group.bench_with_input(
            BenchmarkId::new("Parallel", num_groups),
            num_groups,
            |b, &_num_groups| {
                let mut optimizer = SGD::new(0.01);
                b.iter(|| {
                    let result = parallel_step_array1(
                        black_box(&mut optimizer),
                        black_box(&params_list),
                        black_box(&grads_list),
                    )
                    .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ParallelOptimizer wrapper
fn bench_parallel_optimizer_wrapper(c: &mut Criterion) {
    let mut group = c.benchmark_group("ParallelOptimizer_Wrapper");

    for num_groups in [2, 4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*num_groups as u64));

        let params_list: Vec<Array1<f32>> = (0..*num_groups)
            .map(|_| Array1::from_elem(1000, 1.0))
            .collect();
        let grads_list: Vec<Array1<f32>> = (0..*num_groups)
            .map(|_| Array1::from_elem(1000, 0.1))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_groups),
            num_groups,
            |b, &_num_groups| {
                let optimizer = SGD::new(0.01);
                let mut parallel_opt: ParallelOptimizer<_, f32, _> =
                    ParallelOptimizer::new(optimizer);

                b.iter(|| {
                    let result = parallel_opt
                        .step_parallel_groups(black_box(&params_list), black_box(&grads_list))
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scaling with parameter group size
fn bench_parameter_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parameter_Size_Scaling");
    let num_groups = 8;

    for size in [100, 500, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements((num_groups * size) as u64));

        let params_list: Vec<Array1<f32>> = (0..num_groups)
            .map(|_| Array1::from_elem(*size, 1.0))
            .collect();
        let grads_list: Vec<Array1<f32>> = (0..num_groups)
            .map(|_| Array1::from_elem(*size, 0.1))
            .collect();

        // Sequential
        group.bench_with_input(BenchmarkId::new("Sequential", size), size, |b, &_size| {
            let mut optimizer = SGD::new(0.01);
            b.iter(|| {
                let mut results = Vec::with_capacity(params_list.len());
                for (params, grads) in params_list.iter().zip(grads_list.iter()) {
                    let result = optimizer.step(black_box(params), black_box(grads)).unwrap();
                    results.push(result);
                }
                black_box(results)
            });
        });

        // Parallel
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, &_size| {
            let mut optimizer = SGD::new(0.01);
            b.iter(|| {
                let result = parallel_step_array1(
                    black_box(&mut optimizer),
                    black_box(&params_list),
                    black_box(&grads_list),
                )
                .unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark different optimizer types in parallel
fn bench_optimizer_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimizer_Types_Parallel");
    let num_groups = 8;
    let size = 1000;

    let params_list: Vec<Array1<f32>> = (0..num_groups)
        .map(|_| Array1::from_elem(size, 1.0))
        .collect();
    let grads_list: Vec<Array1<f32>> = (0..num_groups)
        .map(|_| Array1::from_elem(size, 0.1))
        .collect();

    group.throughput(Throughput::Elements((num_groups * size) as u64));

    // SGD
    group.bench_function("SGD", |b| {
        let mut optimizer = SGD::new(0.01);
        b.iter(|| {
            let result = parallel_step_array1(
                black_box(&mut optimizer),
                black_box(&params_list),
                black_box(&grads_list),
            )
            .unwrap();
            black_box(result)
        });
    });

    // SGD with momentum
    group.bench_function("SGD_Momentum", |b| {
        let mut optimizer = SGD::new_with_config(0.01, 0.9, 0.0);
        b.iter(|| {
            let result = parallel_step_array1(
                black_box(&mut optimizer),
                black_box(&params_list),
                black_box(&grads_list),
            )
            .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark memory overhead of parallel processing
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory_Overhead");
    let num_groups = 16;

    for size in [100, 1000, 10000].iter() {
        let params_list: Vec<Array1<f32>> = (0..num_groups)
            .map(|_| Array1::from_elem(*size, 1.0))
            .collect();
        let grads_list: Vec<Array1<f32>> = (0..num_groups)
            .map(|_| Array1::from_elem(*size, 0.1))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("Parallel_with_cloning", size),
            size,
            |b, &_size| {
                let mut optimizer = SGD::new(0.01);
                b.iter(|| {
                    let result = parallel_step_array1(
                        black_box(&mut optimizer),
                        black_box(&params_list),
                        black_box(&grads_list),
                    )
                    .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel efficiency
fn bench_parallel_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel_Efficiency");
    let size = 1000;

    // Vary number of groups to see parallel efficiency
    for num_groups in [1, 2, 4, 8, 16, 32, 64].iter() {
        let params_list: Vec<Array1<f32>> = (0..*num_groups)
            .map(|_| Array1::from_elem(size, 1.0))
            .collect();
        let grads_list: Vec<Array1<f32>> = (0..*num_groups)
            .map(|_| Array1::from_elem(size, 0.1))
            .collect();

        group.throughput(Throughput::Elements(*num_groups as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_groups),
            num_groups,
            |b, &_num_groups| {
                let mut optimizer = SGD::new(0.01);
                b.iter(|| {
                    let result = parallel_step_array1(
                        black_box(&mut optimizer),
                        black_box(&params_list),
                        black_box(&grads_list),
                    )
                    .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_vs_parallel,
    bench_parallel_optimizer_wrapper,
    bench_parameter_size_scaling,
    bench_optimizer_types,
    bench_memory_overhead,
    bench_parallel_efficiency,
);

criterion_main!(benches);
