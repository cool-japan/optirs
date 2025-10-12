//! Production Monitoring and Metrics
//!
//! This example demonstrates production-ready monitoring features:
//! - Real-time metrics collection
//! - Gradient and parameter statistics
//! - Convergence detection
//! - Performance profiling
//! - Metrics export (JSON, CSV)
//!
//! Run with: cargo run --example production_monitoring

use optirs_core::optimizer_metrics::{
    ConvergenceMetrics, GradientStatistics, MetricsCollector, MetricsReporter, OptimizerMetrics,
    ParameterStatistics,
};
use optirs_core::optimizers::{Adam, Optimizer, SGD};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{thread_rng, Distribution, Normal};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Production Monitoring and Metrics ===\n");

    // Example 1: Basic Metrics Collection
    println!("1. Basic Metrics Collection");
    println!("---------------------------");
    basic_metrics()?;

    // Example 2: Multi-Optimizer Monitoring
    println!("\n2. Multi-Optimizer Monitoring");
    println!("-----------------------------");
    multi_optimizer_monitoring()?;

    // Example 3: Convergence Detection
    println!("\n3. Convergence Detection");
    println!("------------------------");
    convergence_detection()?;

    // Example 4: Gradient Analysis
    println!("\n4. Gradient Analysis");
    println!("--------------------");
    gradient_analysis()?;

    // Example 5: Metrics Export
    println!("\n5. Metrics Export (JSON/CSV)");
    println!("----------------------------");
    metrics_export()?;

    // Example 6: Performance Profiling
    println!("\n6. Performance Profiling");
    println!("------------------------");
    performance_profiling()?;

    Ok(())
}

/// Demonstrates basic metrics collection
fn basic_metrics() -> Result<(), Box<dyn std::error::Error>> {
    let mut metrics = OptimizerMetrics::new("sgd");
    let mut optimizer = SGD::new(0.01);

    let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.08, 0.12]);

    println!("Running 10 optimization steps...");

    for step in 0..10 {
        let params_before = params.clone();
        let start = Instant::now();

        let params = optimizer.step(&params, &gradients)?;
        let step_duration = start.elapsed();

        // Update metrics
        metrics.update_step(
            step_duration,
            0.01,
            &gradients.view(),
            &params_before.view(),
            &params.view(),
        );

        if step % 3 == 0 {
            println!(
                "  Step {}: time={:?}, throughput={:.2} steps/sec",
                step,
                metrics.avg_step_time,
                metrics.throughput()
            );
        }
    }

    println!("\nFinal metrics:");
    println!("  Total steps: {}", metrics.step_count);
    println!("  Avg step time: {:?}", metrics.avg_step_time);
    println!("  Throughput: {:.2} steps/sec", metrics.throughput());
    println!("  Gradient norm: {:.6}", metrics.gradient_stats.norm);
    println!(
        "  Update magnitude: {:.6}",
        metrics.parameter_stats.update_magnitude
    );

    Ok(())
}

/// Demonstrates monitoring multiple optimizers simultaneously
fn multi_optimizer_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let mut collector = MetricsCollector::new();

    // Register multiple optimizers
    collector.register_optimizer("sgd");
    collector.register_optimizer("adam");
    collector.register_optimizer("adamw");

    let params = Array1::from_elem(1000, 1.0);
    let gradients = Array1::from_elem(1000, 0.01);

    // Create optimizers
    let mut sgd = SGD::new(0.01);
    let mut adam = Adam::new(0.001);
    let mut adamw = Adam::new(0.001);

    println!("Training with 3 optimizers for 20 steps...\n");

    for step in 0..20 {
        // SGD step
        let params_before = params.clone();
        let start = Instant::now();
        let params_sgd = sgd.step(&params, &gradients)?;
        let sgd_duration = start.elapsed();

        collector.update(
            "sgd",
            sgd_duration,
            0.01,
            &gradients.view(),
            &params_before.view(),
            &params_sgd.view(),
        )?;

        // Adam step
        let start = Instant::now();
        let params_adam = adam.step(&params, &gradients)?;
        let adam_duration = start.elapsed();

        collector.update(
            "adam",
            adam_duration,
            0.001,
            &gradients.view(),
            &params_before.view(),
            &params_adam.view(),
        )?;

        // AdamW step
        let start = Instant::now();
        let params_adamw = adamw.step(&params, &gradients)?;
        let adamw_duration = start.elapsed();

        collector.update(
            "adamw",
            adamw_duration,
            0.001,
            &gradients.view(),
            &params_before.view(),
            &params_adamw.view(),
        )?;

        if step % 5 == 0 {
            println!("Step {}:", step);
            for name in &["sgd", "adam", "adamw"] {
                if let Some(metrics) = collector.get_metrics(name) {
                    println!(
                        "  {}: throughput={:.2} steps/sec, converging={}",
                        name,
                        metrics.throughput(),
                        metrics.convergence.is_converging
                    );
                }
            }
        }
    }

    // Print summary report
    println!("\n{}", collector.summary_report());

    Ok(())
}

/// Demonstrates convergence detection
fn convergence_detection() -> Result<(), Box<dyn std::error::Error>> {
    let mut convergence = ConvergenceMetrics::default();
    let mut param_stats = ParameterStatistics::default();

    println!("Simulating optimization trajectory:");
    println!("(Update magnitudes gradually decrease)\n");

    // Simulate decreasing update magnitudes (convergence)
    let update_magnitudes = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25];

    for (step, &magnitude) in update_magnitudes.iter().enumerate() {
        param_stats.update_magnitude = magnitude;
        convergence.update(&param_stats);

        println!(
            "Step {}: update_mag={:.2}, moving_avg={:.2}, converging={}",
            step, magnitude, convergence.update_moving_avg, convergence.is_converging
        );
    }

    if convergence.is_converging {
        println!("\n✓ Convergence detected!");
        println!(
            "  Convergence rate: {:.2}%",
            convergence.convergence_rate * 100.0
        );
    }

    Ok(())
}

/// Demonstrates gradient analysis for debugging
fn gradient_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate different gradient scenarios
    let scenarios = vec![
        ("Normal gradients", vec![0.1, 0.2, 0.15, 0.18, 0.12]),
        (
            "Vanishing gradients",
            vec![0.001, 0.0005, 0.0008, 0.0006, 0.0007],
        ),
        ("Exploding gradients", vec![10.0, 50.0, 100.0, 80.0, 120.0]),
        ("Dead gradients", vec![0.0, 0.0, 0.0, 0.0, 0.0]),
    ];

    for (name, values) in scenarios {
        let gradients = Array1::from_vec(values);
        let mut stats = GradientStatistics::default();
        stats.update(&gradients.view());

        println!("\n{}:", name);
        println!("  Mean: {:.6}", stats.mean);
        println!("  Std dev: {:.6}", stats.std_dev);
        println!("  Min: {:.6}", stats.min);
        println!("  Max: {:.6}", stats.max);
        println!("  Norm: {:.6}", stats.norm);
        println!("  Zero gradients: {}", stats.num_zeros);

        // Diagnose issues
        if stats.norm < 1e-4 {
            println!("  ⚠️  WARNING: Vanishing gradients detected!");
        } else if stats.norm > 10.0 {
            println!("  ⚠️  WARNING: Exploding gradients detected!");
        } else if stats.num_zeros > gradients.len() / 2 {
            println!("  ⚠️  WARNING: Many dead gradients!");
        } else {
            println!("  ✓ Gradients look healthy");
        }
    }

    Ok(())
}

/// Demonstrates metrics export to JSON and CSV
fn metrics_export() -> Result<(), Box<dyn std::error::Error>> {
    let mut metrics = OptimizerMetrics::new("adam");

    // Simulate some training
    let params = Array1::from_elem(100, 1.0);
    let gradients = Array1::from_elem(100, 0.01);

    for _ in 0..10 {
        metrics.update_step(
            Duration::from_micros(100),
            0.001,
            &gradients.view(),
            &params.view(),
            &params.view(),
        );
    }

    // Export to JSON
    println!("JSON Export:");
    println!("{}\n", MetricsReporter::to_json(&metrics));

    // Export to CSV
    println!("CSV Export:");
    println!("{}", MetricsReporter::to_csv_header());
    println!("{}", MetricsReporter::to_csv(&metrics));

    Ok(())
}

/// Demonstrates performance profiling
fn performance_profiling() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![100, 1_000, 10_000, 100_000];

    println!("Profiling optimizer performance across different sizes:\n");
    println!("{:<12} {:<15} {:<15}", "Size", "Avg Time", "Throughput");
    println!("{:-<45}", "");

    for &size in &sizes {
        let params = Array1::from_elem(size, 1.0);
        let gradients = Array1::from_elem(size, 0.01);
        let mut optimizer = Adam::new(0.001);

        // Warmup
        for _ in 0..5 {
            let _ = optimizer.step(&params, &gradients)?;
        }

        // Measure
        let num_iterations = 100;
        let start = Instant::now();

        for _ in 0..num_iterations {
            let _ = optimizer.step(&params, &gradients)?;
        }

        let total_time = start.elapsed();
        let avg_time = total_time / num_iterations;
        let throughput = 1.0 / avg_time.as_secs_f64();

        println!(
            "{:<12} {:>10.2}µs {:>10.2} steps/s",
            format!("{}", size),
            avg_time.as_micros(),
            throughput
        );
    }

    println!("\n✓ Profile complete - identify bottlenecks and optimize accordingly");

    Ok(())
}
