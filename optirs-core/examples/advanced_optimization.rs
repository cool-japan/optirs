//! Advanced Optimization Techniques
//!
//! This example demonstrates advanced optimization features including:
//! - Learning rate schedulers
//! - Parameter groups with different learning rates
//! - L1 and L2 regularization
//! - Gradient clipping
//! - Custom loss functions
//!
//! Run with: cargo run --example advanced_optimization

use optirs_core::optimizers::{Adam, Optimizer, SGD};
use optirs_core::schedulers::{ExponentialDecay, LearningRateScheduler, StepDecay};
use scirs2_core::ndarray::{array, Array1, Ix1};
use scirs2_core::random::{thread_rng, Distribution, Normal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Optimization Techniques ===\n");

    // Example 1: Learning Rate Scheduling
    println!("1. Learning Rate Scheduling");
    println!("----------------------------");
    learning_rate_scheduling()?;

    // Example 2: Parameter Groups
    println!("\n2. Parameter Groups with Different Learning Rates");
    println!("-------------------------------------------------");
    parameter_groups()?;

    // Example 3: L1 and L2 Regularization
    println!("\n3. L1 and L2 Regularization");
    println!("---------------------------");
    regularization_example()?;

    // Example 4: Gradient Clipping
    println!("\n4. Gradient Clipping");
    println!("--------------------");
    gradient_clipping()?;

    // Example 5: Multi-layer Network Optimization
    println!("\n5. Multi-layer Network Optimization");
    println!("------------------------------------");
    multi_layer_optimization()?;

    Ok(())
}

/// Demonstrates learning rate scheduling strategies
fn learning_rate_scheduling() -> Result<(), Box<dyn std::error::Error>> {
    // Initial parameters and gradients for demonstration
    let mut params = Array1::from_vec(vec![5.0, 3.0, 8.0, 2.0]);
    let gradients = Array1::from_elem(4, 0.1);

    // Exponential Decay Scheduler
    println!("Exponential Decay (decay_rate=0.9, decay_steps=1):");
    let mut optimizer = SGD::new(1.0);
    let mut scheduler = ExponentialDecay::new(1.0, 0.9, 1);

    for epoch in 0..5 {
        // Advance scheduler and get new learning rate
        let lr = scheduler.step();
        // Perform optimization step with current learning rate
        params = optimizer.step(&params, &gradients)?;
        // Update optimizer's learning rate for next iteration
        Optimizer::<f64, Ix1>::set_learning_rate(&mut optimizer, lr);
        println!("  Epoch {}: LR = {:.4}", epoch, lr);
    }

    // Step Decay Scheduler
    println!("\nStep Decay (drop by 0.5 every 2 epochs):");
    let mut params = Array1::from_vec(vec![5.0, 3.0, 8.0, 2.0]);
    let mut optimizer = SGD::new(1.0);
    let mut scheduler = StepDecay::new(1.0, 0.5, 2);

    for epoch in 0..6 {
        // Advance scheduler and get new learning rate
        let lr = scheduler.step();
        // Perform optimization step with current learning rate
        params = optimizer.step(&params, &gradients)?;
        // Update optimizer's learning rate for next iteration
        Optimizer::<f64, Ix1>::set_learning_rate(&mut optimizer, lr);
        println!("  Epoch {}: LR = {:.4}", epoch, lr);
    }

    Ok(())
}

/// Demonstrates parameter groups with different learning rates
fn parameter_groups() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate a neural network with two layers
    // Layer 1 (feature extraction) - slower learning rate
    let layer1_params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let layer1_grads = Array1::from_elem(4, 0.1);

    // Layer 2 (classifier) - faster learning rate
    let layer2_params = Array1::from_vec(vec![5.0, 6.0]);
    let layer2_grads = Array1::from_elem(2, 0.1);

    // Create separate optimizers for each layer
    let mut layer1_optimizer = Adam::new(0.0001); // Slow learning
    let mut layer2_optimizer = Adam::new(0.01); // Fast learning

    println!("Before optimization:");
    println!("  Layer 1 params: {:?}", layer1_params);
    println!("  Layer 2 params: {:?}", layer2_params);

    // Perform optimization steps
    let layer1_updated = layer1_optimizer.step(&layer1_params, &layer1_grads)?;
    let layer2_updated = layer2_optimizer.step(&layer2_params, &layer2_grads)?;

    println!("\nAfter 1 step:");
    println!("  Layer 1 params: {:?}", layer1_updated);
    println!("  Layer 2 params: {:?}", layer2_updated);

    println!("\nObservation: Layer 2 (faster LR) changed more than Layer 1 (slower LR)");

    Ok(())
}

/// Demonstrates L1 and L2 regularization
fn regularization_example() -> Result<(), Box<dyn std::error::Error>> {
    let params = Array1::from_vec(vec![2.0, -3.0, 1.5, -0.5]);
    let gradients = Array1::from_elem(4, 0.1);

    // L2 Regularization (Weight Decay)
    println!("L2 Regularization (Weight Decay = 0.01):");
    let weight_decay = 0.01;
    let l2_penalty: Array1<f64> = params.mapv(|p| weight_decay * p);
    let regularized_grads = &gradients + &l2_penalty;

    let mut optimizer = SGD::new(0.1);
    let updated = optimizer.step(&params, &regularized_grads)?;

    println!("  Original params: {:?}", params);
    println!("  L2 penalty:      {:?}", l2_penalty);
    println!("  Updated params:  {:?}", updated);

    // L1 Regularization (Lasso)
    println!("\nL1 Regularization (alpha = 0.01):");
    let alpha = 0.01;
    let l1_penalty: Array1<f64> = params.mapv(|p| alpha * p.signum());
    let l1_regularized_grads = &gradients + &l1_penalty;

    let mut optimizer = SGD::new(0.1);
    let updated_l1 = optimizer.step(&params, &l1_regularized_grads)?;

    println!("  Original params: {:?}", params);
    println!("  L1 penalty:      {:?}", l1_penalty);
    println!("  Updated params:  {:?}", updated_l1);

    println!("\nNote: L1 encourages sparsity, L2 encourages small weights");

    Ok(())
}

/// Demonstrates gradient clipping techniques
fn gradient_clipping() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate exploding gradients
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let large_gradients = Array1::from_vec(vec![100.0, -150.0, 200.0, -80.0]);

    println!("Original gradients: {:?}", large_gradients);
    println!("Gradient norm: {:.2}", gradient_norm(&large_gradients));

    // Clip by norm
    let max_norm = 10.0;
    let clipped = clip_gradient_norm(&large_gradients, max_norm);
    println!("\nAfter clipping to norm {:.1}:", max_norm);
    println!("  Clipped gradients: {:?}", clipped);
    println!("  New norm: {:.2}", gradient_norm(&clipped));

    // Apply optimization with clipped gradients
    let mut optimizer = Adam::new(0.01);
    let updated = optimizer.step(&params, &clipped)?;
    println!("\nUpdated params: {:?}", updated);

    Ok(())
}

/// Demonstrates optimization of a multi-layer network
fn multi_layer_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate a 2-layer network: input -> hidden -> output
    // Input: 4 dimensions, Hidden: 8 dimensions, Output: 2 dimensions

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 0.1)?;

    // Initialize parameters
    let mut w1 = Array1::from_iter((0..32).map(|_| normal.sample(&mut rng))); // 4x8 flattened
    let mut w2 = Array1::from_iter((0..16).map(|_| normal.sample(&mut rng))); // 8x2 flattened

    println!("Network architecture: 4 -> 8 -> 2");
    println!("Layer 1 params: {} weights", w1.len());
    println!("Layer 2 params: {} weights", w2.len());

    // Create optimizers with different strategies
    let mut opt1 = Adam::new(0.001); // Adam for layer 1
    let mut opt2 = SGD::new(0.01); // SGD with momentum for layer 2

    // Simulate 10 training iterations
    println!("\nTraining for 10 iterations...");

    for iter in 0..10 {
        // Simulate forward pass and gradient computation
        let grad1 = Array1::from_iter((0..32).map(|_| normal.sample(&mut rng)));
        let grad2 = Array1::from_iter((0..16).map(|_| normal.sample(&mut rng)));

        // Clip gradients
        let grad1_clipped = clip_gradient_norm(&grad1, 5.0);
        let grad2_clipped = clip_gradient_norm(&grad2, 5.0);

        // Optimization steps
        w1 = opt1.step(&w1, &grad1_clipped)?;
        w2 = opt2.step(&w2, &grad2_clipped)?;

        if iter % 2 == 0 {
            let loss = simulate_loss(&w1, &w2);
            println!("  Iteration {}: Loss = {:.4}", iter, loss);
        }
    }

    println!("\nOptimization complete!");
    println!("Final layer 1 mean: {:.4}", w1.mean().unwrap());
    println!("Final layer 2 mean: {:.4}", w2.mean().unwrap());

    Ok(())
}

// Helper functions

fn gradient_norm(grads: &Array1<f64>) -> f64 {
    grads.iter().map(|&g| g * g).sum::<f64>().sqrt()
}

fn clip_gradient_norm(grads: &Array1<f64>, max_norm: f64) -> Array1<f64> {
    let norm = gradient_norm(grads);
    if norm > max_norm {
        grads.mapv(|g| g * max_norm / norm)
    } else {
        grads.clone()
    }
}

fn simulate_loss(w1: &Array1<f64>, w2: &Array1<f64>) -> f64 {
    // Simulate a loss that decreases over time
    let w1_norm = w1.iter().map(|&w| w * w).sum::<f64>();
    let w2_norm = w2.iter().map(|&w| w * w).sum::<f64>();
    (w1_norm + w2_norm).sqrt() * 0.1
}
