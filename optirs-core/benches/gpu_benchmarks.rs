//! GPU acceleration benchmarks
//!
//! This benchmark suite demonstrates GPU acceleration potential
//! for optimization operations on large models.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use optirs_core::gpu_optimizer::{GpuConfig, GpuOptimizer};
use optirs_core::optimizers::{Adam, Optimizer, SGD};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;

/// Benchmark GPU vs CPU optimization
fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_vs_CPU");

    for size in [10_000, 50_000, 100_000, 500_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_elem(*size, 1.0);
        let gradients = Array1::from_elem(*size, 0.1);

        // CPU baseline
        group.bench_with_input(BenchmarkId::new("CPU", size), size, |b, &_size| {
            let mut optimizer = SGD::new(0.01);
            b.iter(|| {
                let result = optimizer
                    .step(black_box(&params), black_box(&gradients))
                    .unwrap();
                black_box(result)
            });
        });

        // GPU accelerated
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |b, &_size| {
            let optimizer = SGD::new(0.01);
            let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();
            b.iter(|| {
                let result = gpu_opt
                    .step(black_box(&params), black_box(&gradients))
                    .unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark GPU initialization overhead
fn bench_gpu_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Initialization");

    group.bench_function("Default_Config", |b| {
        b.iter(|| {
            let optimizer = SGD::new(0.01);
            let gpu_opt = GpuOptimizer::with_default_config(optimizer);
            black_box(gpu_opt)
        });
    });

    group.bench_function("Custom_Config", |b| {
        b.iter(|| {
            let optimizer = SGD::new(0.01);
            let config = GpuConfig {
                use_tensor_cores: true,
                use_mixed_precision: true,
                preferred_backend: Some("cuda".to_string()),
                max_gpu_memory: Some(1_000_000_000),
                track_memory: true,
            };
            let gpu_opt = GpuOptimizer::new(optimizer, config);
            black_box(gpu_opt)
        });
    });

    group.finish();
}

/// Benchmark different optimizer types on GPU
fn bench_gpu_optimizer_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Optimizer_Types");

    let size = 100_000;
    let params = Array1::from_elem(size, 1.0);
    let gradients = Array1::from_elem(size, 0.1);

    group.throughput(Throughput::Elements(size as u64));

    // GPU SGD
    group.bench_function("GPU_SGD", |b| {
        let optimizer = SGD::new(0.01);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();
        b.iter(|| {
            let result = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    // GPU SGD with momentum
    group.bench_function("GPU_SGD_Momentum", |b| {
        let optimizer = SGD::new_with_config(0.01, 0.9, 0.0);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();
        b.iter(|| {
            let result = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    // GPU Adam
    group.bench_function("GPU_Adam", |b| {
        let optimizer = Adam::new(0.001);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();
        b.iter(|| {
            let result = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark GPU configuration options
fn bench_gpu_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Configurations");

    let size = 100_000;
    let params = Array1::from_elem(size, 1.0);
    let gradients = Array1::from_elem(size, 0.1);

    group.throughput(Throughput::Elements(size as u64));

    // Default configuration
    group.bench_function("Default", |b| {
        let optimizer = SGD::new(0.01);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();
        b.iter(|| {
            let result = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    // Tensor cores enabled
    group.bench_function("Tensor_Cores", |b| {
        let optimizer = SGD::new(0.01);
        let config = GpuConfig {
            use_tensor_cores: true,
            ..Default::default()
        };
        let mut gpu_opt = GpuOptimizer::new(optimizer, config).unwrap();
        b.iter(|| {
            let result = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    // Mixed precision
    group.bench_function("Mixed_Precision", |b| {
        let optimizer = SGD::new(0.01);
        let config = GpuConfig {
            use_mixed_precision: true,
            ..Default::default()
        };
        let mut gpu_opt = GpuOptimizer::new(optimizer, config).unwrap();
        b.iter(|| {
            let result = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    // Tensor cores + mixed precision
    group.bench_function("Tensor_Cores_Mixed", |b| {
        let optimizer = SGD::new(0.01);
        let config = GpuConfig {
            use_tensor_cores: true,
            use_mixed_precision: true,
            ..Default::default()
        };
        let mut gpu_opt = GpuOptimizer::new(optimizer, config).unwrap();
        b.iter(|| {
            let result = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark GPU memory estimation
fn bench_gpu_memory_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Memory_Estimation");

    group.bench_function("Estimate_Memory", |b| {
        b.iter(|| {
            for num_params in [1_000, 10_000, 100_000, 1_000_000, 10_000_000].iter() {
                let mem = GpuOptimizer::<SGD<f32>, f32>::estimate_gpu_memory(*num_params, 4, 1);
                black_box(mem);
            }
        });
    });

    group.finish();
}

/// Benchmark GPU parameter size scaling
fn bench_gpu_parameter_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Parameter_Scaling");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_elem(*size, 1.0);
        let gradients = Array1::from_elem(*size, 0.1);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let optimizer = SGD::new(0.01);
            let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();
            b.iter(|| {
                let result = gpu_opt
                    .step(black_box(&params), black_box(&gradients))
                    .unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark GPU batch processing
fn bench_gpu_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Batch_Processing");

    let size = 10_000;

    for batch_count in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements((size * batch_count) as u64));

        let params = Array1::from_elem(size, 1.0);
        let gradients = Array1::from_elem(size, 0.1);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_count),
            batch_count,
            |b, &count| {
                let optimizer = SGD::new(0.01);
                let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();
                b.iter(|| {
                    for _ in 0..count {
                        let result = gpu_opt
                            .step(black_box(&params), black_box(&gradients))
                            .unwrap();
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU configuration switching
fn bench_gpu_config_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU_Config_Switching");

    let size = 100_000;
    let params = Array1::from_elem(size, 1.0);
    let gradients = Array1::from_elem(size, 0.1);

    group.bench_function("Toggle_Tensor_Cores", |b| {
        let optimizer = SGD::new(0.01);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();

        b.iter(|| {
            gpu_opt.set_use_tensor_cores(true);
            let result1 = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();

            gpu_opt.set_use_tensor_cores(false);
            let result2 = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();

            black_box((result1, result2))
        });
    });

    group.bench_function("Toggle_Mixed_Precision", |b| {
        let optimizer = SGD::new(0.01);
        let mut gpu_opt = GpuOptimizer::with_default_config(optimizer).unwrap();

        b.iter(|| {
            gpu_opt.set_use_mixed_precision(true);
            let result1 = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();

            gpu_opt.set_use_mixed_precision(false);
            let result2 = gpu_opt
                .step(black_box(&params), black_box(&gradients))
                .unwrap();

            black_box((result1, result2))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gpu_vs_cpu,
    bench_gpu_initialization,
    bench_gpu_optimizer_types,
    bench_gpu_configurations,
    bench_gpu_memory_estimation,
    bench_gpu_parameter_scaling,
    bench_gpu_batch_processing,
    bench_gpu_config_switching,
);

criterion_main!(benches);
