//! Example of using Adam optimizer with learning rate scheduler

use scirs2_core::ndarray_ext::Array1;
use scirs2_optim::optimizers::{Adam, Optimizer};
use scirs2_optim::schedulers::{ExponentialDecay, LearningRateScheduler};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Adam Optimizer with Scheduler Example");
    println!("====================================");

    // Define a Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // This function has a global minimum at (1, 1)
    let f = |x: &Array1<f64>| -> f64 {
        let a = 1.0;
        let b = 100.0;
        let x_val = x[0];
        let y_val = x[1];

        (a - x_val).powi(2) + b * (y_val - x_val.powi(2)).powi(2)
    };

    // Gradient of the Rosenbrock function
    let gradient_fn = |x: &Array1<f64>| -> Array1<f64> {
        let a = 1.0;
        let b = 100.0;
        let x_val = x[0];
        let y_val = x[1];

        let dx = -2.0 * (a - x_val) - 4.0 * b * x_val * (y_val - x_val.powi(2));
        let dy = 2.0 * b * (y_val - x_val.powi(2));

        Array1::from_vec(vec![dx, dy])
    };

    // Initial parameters: far from the minimum
    let mut params = Array1::from_vec(vec![-1.5, 1.5]);

    // Create Adam optimizer with default parameters
    let mut optimizer = Adam::new(0.1);

    // Create an exponential decay scheduler
    let mut scheduler = ExponentialDecay::new(0.1, 0.95, 100);

    println!(
        "Starting optimization at x = [{}, {}]",
        params[0], params[1]
    );
    println!("Target minimum is at x = [1.0, 1.0]");
    println!("Initial learning rate = {}", optimizer.learning_rate());
    println!("\nIteration | Learning Rate |     x     |     y     | Function Value");
    println!("----------------------------------------------------------------");

    // Run optimization for 500 iterations
    for i in 0..500 {
        // Compute gradient
        let gradient = gradient_fn(&params);

        // Compute function value for logging
        let function_value = f(&params);

        // Print current state (every 50 iterations)
        if i % 50 == 0 {
            println!(
                "{:9} | {:13.6} | {:9.4} | {:9.4} | {:13.6}",
                i,
                optimizer.learning_rate(),
                params[0],
                params[1],
                function_value
            );
        }

        // Update parameters
        params = optimizer.step(&params, &gradient)?;

        // Update learning rate using scheduler
        let new_lr = scheduler.step();
        optimizer.set_lr(new_lr);
    }

    // Final function value
    let final_value = f(&params);

    println!(
        "\nFinal parameters: x = [{:.6}, {:.6}]",
        params[0], params[1]
    );
    println!("Final function value: f(x) = {:.6}", final_value);
    println!(
        "Distance to minimum: {:.6}",
        f64::sqrt((params[0] - 1.0).powi(2) + (params[1] - 1.0).powi(2))
    );

    Ok(())
}
