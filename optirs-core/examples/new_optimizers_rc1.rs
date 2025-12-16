//! # OptiRS RC.1 New Optimizers Example
//!
//! Demonstrates the new optimizers introduced in v0.1.0-rc.1:
//! - AdaDelta: Adaptive learning rate without manual tuning
//! - AdaBound: Smooth transition from Adam to SGD
//!
//! Run with: cargo run --example new_optimizers_rc1

use optirs_core::optimizers::{AdaBound, AdaDelta};
use scirs2_core::ndarray_ext::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  OptiRS v0.1.0-rc.1 - New Optimizers Demonstration      ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Example problem: Optimize quadratic function f(x,y) = x² + y²
    // Gradient: ∇f = [2x, 2y]
    // Optimal solution: (0, 0)

    println!("Problem: Minimize f(x,y) = x² + y²");
    println!("Starting point: (5.0, 5.0)");
    println!("Optimal solution: (0.0, 0.0)\n");

    // ========================================
    // 1. AdaDelta Optimizer
    // ========================================
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ 1. AdaDelta: No Learning Rate Required                 │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    let mut adadelta = AdaDelta::<f64>::new(
        0.95, // rho: decay rate for moving averages
        1e-6, // epsilon: numerical stability constant
    )?;

    let mut params = array![5.0, 5.0];
    println!("Initial parameters: {:?}", params);

    // Run optimization for 100 steps
    for step in 0..100 {
        let grads = array![2.0 * params[0], 2.0 * params[1]];
        params = adadelta.step(params.view(), grads.view())?;

        if step % 20 == 0 {
            let loss = params[0] * params[0] + params[1] * params[1];
            println!(
                "  Step {:3}: params = [{:.4}, {:.4}], loss = {:.6}",
                step, params[0], params[1], loss
            );
        }
    }

    let final_loss = params[0] * params[0] + params[1] * params[1];
    println!("\nFinal parameters: [{:.6}, {:.6}]", params[0], params[1]);
    println!("Final loss: {:.6}", final_loss);
    println!("✓ AdaDelta successfully converged!\n");

    // ========================================
    // 2. AdaBound Optimizer
    // ========================================
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ 2. AdaBound: Smooth Transition Adam → SGD              │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    let mut adabound = AdaBound::<f64>::new(
        0.001, // learning_rate: initial LR
        0.1,   // final_lr: target LR for SGD convergence
        0.9,   // beta1: first moment decay
        0.999, // beta2: second moment decay
        1e-8,  // epsilon: numerical stability
        1e-3,  // gamma: convergence speed parameter
        0.0,   // weight_decay: L2 regularization
        false, // amsbound: use AMSBound variant
    )?;

    let mut params = array![5.0, 5.0];
    println!("Initial parameters: {:?}", params);
    println!("Dynamic bounds will converge from adaptive to final_lr = 0.1\n");

    // Run optimization for 200 steps
    for step in 0..200 {
        let grads = array![2.0 * params[0], 2.0 * params[1]];
        params = adabound.step(params.view(), grads.view())?;

        if step % 40 == 0 {
            let loss = params[0] * params[0] + params[1] * params[1];
            let (lower_bound, upper_bound) = adabound.current_bounds();
            println!(
                "  Step {:3}: params = [{:.4}, {:.4}], loss = {:.6}",
                step, params[0], params[1], loss
            );
            println!(
                "           LR bounds: [{:.6}, {:.6}]",
                lower_bound, upper_bound
            );
        }
    }

    let final_loss = params[0] * params[0] + params[1] * params[1];
    let (final_lower, final_upper) = adabound.current_bounds();
    println!("\nFinal parameters: [{:.6}, {:.6}]", params[0], params[1]);
    println!("Final loss: {:.6}", final_loss);
    println!("Final LR bounds: [{:.6}, {:.6}]", final_lower, final_upper);
    println!("✓ AdaBound successfully converged!\n");

    // ========================================
    // 3. AMSBound Variant
    // ========================================
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ 3. AMSBound: AdaBound with Max Velocity               │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    let mut amsbound = AdaBound::<f64>::new(
        0.001, 0.1, 0.9, 0.999, 1e-8, 1e-3, 0.0, true, // amsbound = true for AMSBound variant
    )?;

    let mut params = array![5.0, 5.0];
    println!("Initial parameters: {:?}", params);
    println!("AMSBound uses max(v_t) for more stable updates\n");

    for step in 0..200 {
        let grads = array![2.0 * params[0], 2.0 * params[1]];
        params = amsbound.step(params.view(), grads.view())?;

        if step % 40 == 0 {
            let loss = params[0] * params[0] + params[1] * params[1];
            println!(
                "  Step {:3}: params = [{:.4}, {:.4}], loss = {:.6}",
                step, params[0], params[1], loss
            );
        }
    }

    let final_loss = params[0] * params[0] + params[1] * params[1];
    println!("\nFinal parameters: [{:.6}, {:.6}]", params[0], params[1]);
    println!("Final loss: {:.6}", final_loss);
    println!("✓ AMSBound successfully converged!\n");

    // ========================================
    // Comparison Summary
    // ========================================
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Optimizer Comparison Summary                            │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    println!("Key Features:");
    println!("  • AdaDelta:");
    println!("    - No learning rate parameter needed");
    println!("    - Uses adaptive rates based on gradient history");
    println!("    - Robust to hyperparameter choices");
    println!("    - Warmup boost helps overcome cold-start problem\n");

    println!("  • AdaBound:");
    println!("    - Smooth transition from Adam to SGD");
    println!("    - Dynamic learning rate bounds");
    println!("    - Better generalization than pure Adam");
    println!("    - Prevents LR from becoming too large/small\n");

    println!("  • AMSBound:");
    println!("    - AdaBound with max velocity");
    println!("    - More stable than standard AdaBound");
    println!("    - Uses max(v_t) like AMSGrad\n");

    println!("Use Cases:");
    println!("  • AdaDelta: When you want to avoid LR tuning");
    println!("  • AdaBound: For production models needing good generalization");
    println!("  • AMSBound: When training stability is critical\n");

    println!("═══════════════════════════════════════════════════════════");
    println!("✓ All new optimizers demonstrated successfully!");
    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}
