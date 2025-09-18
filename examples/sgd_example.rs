//! Example of using SGD optimizer for a simple optimization problem

use scirs2_core::ndarray_ext::Array1;
use scirs2_optim::optimizers::{Optimizer, SGD};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SGD Optimizer Example");
    println!("=====================");

    // Define a simple quadratic function: f(x) = x^2
    // Gradient of f(x) = 2x
    let gradient_fn = |x: &Array1<f64>| -> Array1<f64> {
        let mut grad = x.clone();
        grad.mapv_inplace(|xi| 2.0 * xi);
        grad
    };

    // Initial parameters
    let mut params = Array1::from_vec(vec![5.0]); // Start at x = 5

    // Create SGD optimizer with learning rate 0.1
    let mut optimizer = SGD::new_with_config(0.1, 0.0, 0.0);

    println!("Starting optimization at x = {}", params[0]);
    println!("Learning rate = {}", optimizer.learning_rate());
    println!("\nIteration | Parameter | Gradient | Function Value");
    println!("--------------------------------------------------");

    // Run optimization for 20 iterations
    for i in 0..20 {
        // Compute gradient
        let gradient = gradient_fn(&params);

        // Compute function value for logging
        let function_value = params[0] * params[0];

        // Print current state
        println!(
            "{:9} | {:9.4} | {:8.4} | {:13.4}",
            i, params[0], gradient[0], function_value
        );

        // Update parameters
        params = optimizer.step(&params, &gradient)?;
    }

    // Final function value
    let final_value = params[0] * params[0];

    println!("\nFinal parameter: x = {:.6}", params[0]);
    println!("Final function value: f(x) = {:.6}", final_value);

    Ok(())
}
