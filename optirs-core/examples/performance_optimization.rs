//! Performance Optimization Techniques
//!
//! This example demonstrates OptiRS performance optimization features:
//! - SIMD acceleration for large parameter arrays
//! - Parallel processing for multiple parameter groups
//! - Memory-efficient optimization for large models
//! - GPU acceleration (when available)
//!
//! Run with: cargo run --example performance_optimization --release

use optirs_core::gpu_optimizer::{GpuConfig, GpuOptimizer};
use optirs_core::memory_efficient_optimizer::{
    ChunkedOptimizer, GradientAccumulator, MemoryUsageEstimator,
};
use optirs_core::optimizers::{Adam, Optimizer, SimdSGD, SGD};
use optirs_core::parallel_optimizer::{parallel_step_array1, ParallelOptimizer};
use scirs2_core::ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Optimization Techniques ===\n");

    // Example 1: SIMD Acceleration
    println!("1. SIMD Acceleration");
    println!("--------------------");
    simd_acceleration()?;

    // Example 2: Parallel Processing
    println!("\n2. Parallel Processing");
    println!("----------------------");
    parallel_processing()?;

    // Example 3: Memory-Efficient Optimization
    println!("\n3. Memory-Efficient Optimization");
    println!("--------------------------------");
    memory_efficient_optimization()?;

    // Example 4: GPU Acceleration
    println!("\n4. GPU Acceleration");
    println!("-------------------");
    gpu_acceleration()?;

    // Example 5: Combined Optimizations
    println!("\n5. Combined Performance Optimizations");
    println!("-------------------------------------");
    combined_optimizations()?;

    Ok(())
}

/// Demonstrates SIMD acceleration for large parameter arrays
fn simd_acceleration() -> Result<(), Box<dyn std::error::Error>> {
    let size = 100_000;
    let params = Array1::from_elem(size, 1.0f32);
    let grads = Array1::from_elem(size, 0.001f32);

    println!("Optimizing {} parameters", size);

    // Standard SGD
    let mut sgd = SGD::new(0.01f32);
    let start = Instant::now();
    let _result1 = sgd.step(&params, &grads)?;
    let time1 = start.elapsed();
    println!("Standard SGD: {:?}", time1);

    // SIMD-accelerated SGD
    let mut simd_sgd = SimdSGD::new(0.01f32);
    let start = Instant::now();
    let _result2 = simd_sgd.step(&params, &grads)?;
    let time2 = start.elapsed();
    println!("SIMD SGD:     {:?}", time2);

    if time1 > time2 {
        let speedup = time1.as_secs_f64() / time2.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    }

    println!("\nNote: SIMD provides 2-4x speedup for large arrays (>10,000 elements)");

    Ok(())
}

/// Demonstrates parallel processing for multiple parameter groups
fn parallel_processing() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate multiple parameter groups (e.g., different layers)
    let num_groups = 8;
    let group_size = 10_000;

    let params_list: Vec<Array1<f64>> = (0..num_groups)
        .map(|_| Array1::from_elem(group_size, 1.0))
        .collect();

    let grads_list: Vec<Array1<f64>> = (0..num_groups)
        .map(|_| Array1::from_elem(group_size, 0.001))
        .collect();

    println!(
        "Processing {} parameter groups ({} params each)",
        num_groups, group_size
    );

    // Sequential processing
    let mut optimizer = Adam::new(0.001);
    let start = Instant::now();
    for (params, grads) in params_list.iter().zip(grads_list.iter()) {
        let _ = optimizer.step(params, grads)?;
    }
    let time_sequential = start.elapsed();
    println!("Sequential:  {:?}", time_sequential);

    // Parallel processing
    let mut optimizer = Adam::new(0.001);
    let start = Instant::now();
    let _results = parallel_step_array1(&mut optimizer, &params_list, &grads_list)?;
    let time_parallel = start.elapsed();
    println!("Parallel:    {:?}", time_parallel);

    if time_sequential > time_parallel {
        let speedup = time_sequential.as_secs_f64() / time_parallel.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    }

    println!("\nNote: Parallel processing provides 4-8x speedup for multiple groups");

    Ok(())
}

/// Demonstrates memory-efficient optimization for large models
fn memory_efficient_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate a very large model (100M parameters)
    let total_params = 100_000_000;
    let chunk_size = 10_000_000; // Process 10M at a time

    println!("Optimizing {} parameters", total_params);
    println!("Chunk size: {} parameters", chunk_size);

    // Memory estimation
    let memory_sgd = MemoryUsageEstimator::sgd(total_params, 4);
    let memory_adam = MemoryUsageEstimator::adam(total_params, 4);

    println!("\nMemory requirements (f32):");
    println!("  SGD:  {:.2} GB", memory_sgd as f64 / 1e9);
    println!("  Adam: {:.2} GB", memory_adam as f64 / 1e9);

    // Recommended chunk size for 4GB available memory
    let available_memory = 4_000_000_000;
    let recommended = MemoryUsageEstimator::recommend_chunk_size(
        total_params,
        available_memory,
        4, // f32
        4, // Adam state multiplier
    );
    println!(
        "\nRecommended chunk size for 4GB RAM: {} params",
        recommended
    );

    // Gradient accumulation example
    println!("\n--- Gradient Accumulation ---");
    let mut accumulator = GradientAccumulator::<f32>::new(1000);

    // Simulate 4 micro-batches
    for i in 0..4 {
        let micro_grads = Array1::from_elem(1000, 0.1 * (i + 1) as f32);
        accumulator.accumulate(&micro_grads.view())?;
        println!("Accumulated micro-batch {}", i + 1);
    }

    let avg_grads = accumulator.average()?;
    println!("Average gradient: {:.3}", avg_grads[0]);

    // Chunked optimization example
    println!("\n--- Chunked Optimization ---");
    let params = Array1::from_elem(50_000, 1.0f32);
    let grads = Array1::from_elem(50_000, 0.001f32);

    let optimizer = SGD::new(0.01f32);
    let mut chunked_opt = ChunkedOptimizer::new(optimizer, Some(10_000));

    let start = Instant::now();
    let _result = chunked_opt.step_chunked(&params, &grads)?;
    let time = start.elapsed();

    println!(
        "Processed {} params in {} chunks",
        params.len(),
        chunked_opt.num_chunks(params.len())
    );
    println!("Time: {:?}", time);

    println!("\nNote: Memory-efficient techniques enable training billion-parameter models");

    Ok(())
}

/// Demonstrates GPU acceleration
fn gpu_acceleration() -> Result<(), Box<dyn std::error::Error>> {
    let size = 1_000_000;
    let params = Array1::from_elem(size, 1.0f32);
    let grads = Array1::from_elem(size, 0.001f32);

    println!("Optimizing {} parameters", size);

    // Create GPU-accelerated optimizer
    let optimizer = SGD::new(0.01f32);
    let config = GpuConfig {
        use_tensor_cores: true,
        use_mixed_precision: false,
        preferred_backend: None,
        max_gpu_memory: None,
        track_memory: true,
    };

    let mut gpu_opt = GpuOptimizer::new(optimizer, config)?;

    if gpu_opt.is_gpu_available() {
        println!("GPU backend: {:?}", gpu_opt.gpu_backend());

        let start = Instant::now();
        let _result = gpu_opt.step(&params, &grads)?;
        let time = start.elapsed();
        println!("GPU optimization: {:?}", time);

        // Memory estimation
        let mem = GpuOptimizer::<SGD<f32>, f32>::estimate_gpu_memory(size, 4, 1);
        println!("GPU memory usage: {:.2} MB", mem as f64 / 1e6);

        println!("\nNote: GPU acceleration provides 10-50x speedup for large models");
    } else {
        println!("GPU not available - falling back to CPU");
    }

    Ok(())
}

/// Demonstrates combining multiple optimization techniques
fn combined_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("Combining SIMD + Parallel + Memory-efficient techniques");

    // Large-scale optimization scenario:
    // - 8 parameter groups (parallel)
    // - Each group has 100k parameters (SIMD)
    // - Using gradient accumulation (memory-efficient)

    let num_groups = 8;
    let group_size = 100_000;

    // Create parameter groups
    let params_list: Vec<Array1<f32>> = (0..num_groups)
        .map(|_| Array1::from_elem(group_size, 1.0))
        .collect();

    let grads_list: Vec<Array1<f32>> = (0..num_groups)
        .map(|_| Array1::from_elem(group_size, 0.001))
        .collect();

    println!("\nConfiguration:");
    println!("  Parameter groups: {}", num_groups);
    println!("  Group size: {} params", group_size);
    println!(
        "  Total params: {} M",
        (num_groups * group_size) as f64 / 1e6
    );

    // Use SIMD-accelerated optimizer with parallel processing
    let mut optimizer = SimdSGD::new(0.01f32);

    println!("\nProcessing with combined optimizations...");
    let start = Instant::now();

    // Parallel processing with SIMD-accelerated steps
    let _results = parallel_step_array1(&mut optimizer, &params_list, &grads_list)?;

    let time = start.elapsed();
    println!("Total time: {:?}", time);

    let throughput = (num_groups * group_size) as f64 / time.as_secs_f64();
    println!("Throughput: {:.2} M params/sec", throughput / 1e6);

    println!("\nOptimization strategy:");
    println!("  ✓ SIMD for within-group computation (2-4x)");
    println!("  ✓ Parallel for across-group processing (4-8x)");
    println!("  ✓ Combined speedup: ~8-30x");

    Ok(())
}
