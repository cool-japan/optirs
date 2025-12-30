//! Metrics collection overhead benchmarks
//!
//! This benchmark suite measures the performance overhead of metrics collection
//! to ensure monitoring doesn't significantly impact optimization performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use optirs_core::optimizer_metrics::{MetricsCollector, OptimizerMetrics};
use optirs_core::optimizers::{Optimizer, SGD};
use scirs2_core::ndarray::Array1;
use std::hint::black_box;
use std::time::Duration;

/// Benchmark optimizer without metrics (baseline)
fn bench_optimizer_without_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimizer_Without_Metrics");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_elem(*size, 1.0);
        let gradients = Array1::from_elem(*size, 0.1);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let mut optimizer = SGD::new(0.01);
            b.iter(|| {
                let result = optimizer
                    .step(black_box(&params), black_box(&gradients))
                    .unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark optimizer with metrics collection
fn bench_optimizer_with_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimizer_With_Metrics");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let params = Array1::from_elem(*size, 1.0);
        let gradients = Array1::from_elem(*size, 0.1);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let mut optimizer = SGD::new(0.01);
            let mut collector = MetricsCollector::new();
            collector.register_optimizer("sgd");

            b.iter(|| {
                let params_before = params.clone();
                let result = optimizer
                    .step(black_box(&params), black_box(&gradients))
                    .unwrap();

                collector
                    .update(
                        "sgd",
                        Duration::from_micros(10),
                        0.01,
                        &gradients.view(),
                        &params_before.view(),
                        &result.view(),
                    )
                    .unwrap();

                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark metrics computation overhead
fn bench_metrics_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Metrics_Computation");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let gradients = Array1::from_elem(*size, 0.1);
        let params_before = Array1::from_elem(*size, 1.0);
        let params_after = Array1::from_elem(*size, 0.99);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let mut metrics = OptimizerMetrics::new("test");

            b.iter(|| {
                metrics.update_step(
                    Duration::from_micros(10),
                    0.01,
                    black_box(&gradients.view()),
                    black_box(&params_before.view()),
                    black_box(&params_after.view()),
                );
                black_box(metrics.step_count)
            });
        });
    }

    group.finish();
}

/// Benchmark metrics collector registration
fn bench_metrics_registration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Metrics_Registration");

    group.bench_function("Register_1_Optimizer", |b| {
        b.iter(|| {
            let mut collector = MetricsCollector::new();
            collector.register_optimizer(black_box("sgd"));
            black_box(collector)
        });
    });

    group.bench_function("Register_10_Optimizers", |b| {
        b.iter(|| {
            let mut collector = MetricsCollector::new();
            for i in 0..10 {
                collector.register_optimizer(black_box(format!("opt_{}", i)));
            }
            black_box(collector)
        });
    });

    group.finish();
}

/// Benchmark metrics reporting
fn bench_metrics_reporting(c: &mut Criterion) {
    let mut group = c.benchmark_group("Metrics_Reporting");

    let metrics = OptimizerMetrics::new("test");

    group.bench_function("Generate_Summary", |b| {
        let mut collector = MetricsCollector::new();
        collector.register_optimizer("sgd");
        b.iter(|| {
            let report = collector.summary_report();
            black_box(report)
        });
    });

    group.bench_function("Export_JSON", |b| {
        b.iter(|| {
            let json =
                optirs_core::optimizer_metrics::MetricsReporter::to_json(black_box(&metrics));
            black_box(json)
        });
    });

    group.bench_function("Export_CSV", |b| {
        b.iter(|| {
            let csv = optirs_core::optimizer_metrics::MetricsReporter::to_csv(black_box(&metrics));
            black_box(csv)
        });
    });

    group.finish();
}

/// Benchmark metrics collection for multiple optimizers
fn bench_multi_optimizer_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multi_Optimizer_Metrics");

    for num_optimizers in [1, 5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*num_optimizers as u64));

        let size = 1000;
        let gradients = Array1::from_elem(size, 0.1);
        let params_before = Array1::from_elem(size, 1.0);
        let params_after = Array1::from_elem(size, 0.99);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_optimizers),
            num_optimizers,
            |b, &num| {
                let mut collector = MetricsCollector::new();
                for i in 0..num {
                    collector.register_optimizer(format!("opt_{}", i));
                }

                b.iter(|| {
                    for i in 0..num {
                        collector
                            .update(
                                &format!("opt_{}", i),
                                Duration::from_micros(10),
                                0.01,
                                black_box(&gradients.view()),
                                black_box(&params_before.view()),
                                black_box(&params_after.view()),
                            )
                            .unwrap();
                    }
                    black_box(collector.elapsed())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark metrics overhead percentage
fn bench_metrics_overhead_percentage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Metrics_Overhead_Percentage");

    let size = 10_000;
    let params = Array1::from_elem(size, 1.0);
    let gradients = Array1::from_elem(size, 0.1);

    // Baseline: just optimizer
    group.bench_function("Baseline_Optimizer_Only", |b| {
        let mut optimizer = SGD::new(0.01);
        b.iter(|| {
            let result = optimizer
                .step(black_box(&params), black_box(&gradients))
                .unwrap();
            black_box(result)
        });
    });

    // With metrics
    group.bench_function("With_Full_Metrics", |b| {
        let mut optimizer = SGD::new(0.01);
        let mut collector = MetricsCollector::new();
        collector.register_optimizer("sgd");

        b.iter(|| {
            let params_before = params.clone();
            let result = optimizer
                .step(black_box(&params), black_box(&gradients))
                .unwrap();

            collector
                .update(
                    "sgd",
                    Duration::from_micros(10),
                    0.01,
                    &gradients.view(),
                    &params_before.view(),
                    &result.view(),
                )
                .unwrap();

            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_optimizer_without_metrics,
    bench_optimizer_with_metrics,
    bench_metrics_computation,
    bench_metrics_registration,
    bench_metrics_reporting,
    bench_multi_optimizer_metrics,
    bench_metrics_overhead_percentage,
);

criterion_main!(benches);
