//! Basic optimization example demonstrating SciRS2 integration
//!
//! This example shows how to use OptiRS optimizers with SciRS2-Core
//! for a simple linear regression problem.
//!
//! Run with: cargo run --example basic_optimization --features full

use optirs_core::optimizers::{Adam, AdamW, Optimizer, SGD};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::{thread_rng, Rng};

/// Simple quadratic function: f(x) = (x - 5)^2
fn quadratic_function(x: f64) -> f64 {
    (x - 5.0).powi(2)
}

/// Gradient of quadratic function: f'(x) = 2(x - 5)
fn quadratic_gradient(x: f64) -> f64 {
    2.0 * (x - 5.0)
}

/// Demonstrate SGD optimization
fn demonstrate_sgd() {
    println!("=== SGD Optimization ===");

    let mut params = Array1::from_vec(vec![0.0]); // Start at x=0
    let mut optimizer = SGD::new(0.1); // Learning rate = 0.1

    println!(
        "Initial: x = {:.4}, f(x) = {:.4}",
        params[0],
        quadratic_function(params[0])
    );

    for step in 0..20 {
        let gradient = Array1::from_vec(vec![quadratic_gradient(params[0])]);
        params = optimizer.step(&params, &gradient).unwrap();

        if step % 5 == 4 {
            println!(
                "Step {}: x = {:.4}, f(x) = {:.4}",
                step + 1,
                params[0],
                quadratic_function(params[0])
            );
        }
    }

    println!(
        "Final: x = {:.4}, f(x) = {:.4}",
        params[0],
        quadratic_function(params[0])
    );
    println!("Target: x = 5.0, f(x) = 0.0\n");
}

/// Demonstrate Adam optimization
fn demonstrate_adam() {
    println!("=== Adam Optimization ===");

    let mut params = Array1::from_vec(vec![0.0]); // Start at x=0
    let mut optimizer = Adam::new(0.5); // Higher learning rate for faster convergence

    println!(
        "Initial: x = {:.4}, f(x) = {:.4}",
        params[0],
        quadratic_function(params[0])
    );

    for step in 0..20 {
        let gradient = Array1::from_vec(vec![quadratic_gradient(params[0])]);
        params = optimizer.step(&params, &gradient).unwrap();

        if step % 5 == 4 {
            println!(
                "Step {}: x = {:.4}, f(x) = {:.4}",
                step + 1,
                params[0],
                quadratic_function(params[0])
            );
        }
    }

    println!(
        "Final: x = {:.4}, f(x) = {:.4}",
        params[0],
        quadratic_function(params[0])
    );
    println!("Target: x = 5.0, f(x) = 0.0\n");
}

/// Demonstrate AdamW optimization with weight decay
fn demonstrate_adamw() {
    println!("=== AdamW Optimization (with weight decay) ===");

    let mut params = Array1::from_vec(vec![0.0]); // Start at x=0
    let mut optimizer = AdamW::new(0.5); // Default weight decay is 0.01

    println!(
        "Initial: x = {:.4}, f(x) = {:.4}",
        params[0],
        quadratic_function(params[0])
    );

    for step in 0..20 {
        let gradient = Array1::from_vec(vec![quadratic_gradient(params[0])]);
        params = optimizer.step(&params, &gradient).unwrap();

        if step % 5 == 4 {
            println!(
                "Step {}: x = {:.4}, f(x) = {:.4}",
                step + 1,
                params[0],
                quadratic_function(params[0])
            );
        }
    }

    println!(
        "Final: x = {:.4}, f(x) = {:.4}",
        params[0],
        quadratic_function(params[0])
    );
    println!("Target: x = 5.0, f(x) = 0.0\n");
}

/// Demonstrate multi-dimensional optimization
fn demonstrate_multidimensional() {
    println!("=== Multi-dimensional Optimization ===");
    println!("Function: f(x, y) = (x - 3)^2 + (y + 2)^2");
    println!("Target: x = 3.0, y = -2.0\n");

    // Initialize parameters at origin
    let mut params = Array1::from_vec(vec![0.0, 0.0]);
    let mut optimizer = Adam::new(0.3);

    println!("Initial: x = {:.4}, y = {:.4}", params[0], params[1]);

    for step in 0..30 {
        // Compute gradients
        let grad_x = 2.0 * (params[0] - 3.0);
        let grad_y = 2.0 * (params[1] + 2.0);
        let gradients = Array1::from_vec(vec![grad_x, grad_y]);

        // Update parameters
        params = optimizer.step(&params, &gradients).unwrap();

        if step % 10 == 9 {
            let loss = (params[0] - 3.0).powi(2) + (params[1] + 2.0).powi(2);
            println!(
                "Step {}: x = {:.4}, y = {:.4}, loss = {:.6}",
                step + 1,
                params[0],
                params[1],
                loss
            );
        }
    }

    let final_loss = (params[0] - 3.0).powi(2) + (params[1] + 2.0).powi(2);
    println!(
        "\nFinal: x = {:.4}, y = {:.4}, loss = {:.6}",
        params[0], params[1], final_loss
    );
}

/// Demonstrate SciRS2 random number generation integration
fn demonstrate_scirs2_random() {
    println!("=== SciRS2 Random Integration ===");

    let mut rng = thread_rng();

    // Generate random starting point
    let start_x: f64 = rng.gen_range(-10.0..10.0);
    let start_y: f64 = rng.gen_range(-10.0..10.0);

    println!(
        "Random starting point: x = {:.4}, y = {:.4}",
        start_x, start_y
    );

    let mut params = Array1::from_vec(vec![start_x, start_y]);
    let mut optimizer = Adam::new(0.3);

    for step in 0..50 {
        let grad_x = 2.0 * (params[0] - 3.0);
        let grad_y = 2.0 * (params[1] + 2.0);
        let gradients = Array1::from_vec(vec![grad_x, grad_y]);

        params = optimizer.step(&params, &gradients).unwrap();

        if step == 49 {
            let loss = (params[0] - 3.0).powi(2) + (params[1] + 2.0).powi(2);
            println!(
                "After {} steps: x = {:.4}, y = {:.4}, loss = {:.6}",
                step + 1,
                params[0],
                params[1],
                loss
            );
        }
    }
    println!();
}

fn main() {
    println!("OptiRS - Basic Optimization Examples");
    println!("Demonstrating SciRS2-Core Integration");
    println!("=====================================\n");

    demonstrate_sgd();
    demonstrate_adam();
    demonstrate_adamw();
    demonstrate_multidimensional();
    demonstrate_scirs2_random();

    println!("All examples completed successfully!");
    println!("\nKey Features Demonstrated:");
    println!("✓ SciRS2-Core ndarray integration");
    println!("✓ SciRS2-Core random number generation");
    println!("✓ Multiple optimizer types (SGD, Adam, AdamW)");
    println!("✓ Single and multi-dimensional optimization");
    println!("✓ Convergence to optimal solutions");
}
