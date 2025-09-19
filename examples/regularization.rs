//! Example of using regularization techniques

use scirs2_core::ndarray_ext::Array1;
use optirs_core::optimizers::{Optimizer, SGD};
use optirs_core::regularizers::{ElasticNet, Regularizer, L1, L2};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Regularization Example");
    println!("=====================");

    // Create a parameter vector
    let params = Array1::from_vec(vec![0.5, -0.3, 0.0, 0.2, -0.8, 1.5]);

    // Create gradients for the parameters
    let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.15, -0.05]);

    println!("Parameters: {:?}", params);
    println!("Original Gradients: {:?}", gradients);
    println!();

    // Create L1 regularizer with strength 0.01
    let l1_reg = L1::new(0.01);

    // Apply L1 regularization
    let mut l1_gradients = gradients.clone();
    let l1_penalty = l1_reg.apply(&params, &mut l1_gradients)?;

    println!("L1 Regularization (alpha = 0.01)");
    println!("--------------------------------");
    println!("Modified Gradients: {:?}", l1_gradients);
    println!("L1 Penalty Term: {:.6}", l1_penalty);
    println!();

    // Create L2 regularizer with strength 0.01
    let l2_reg = L2::new(0.01);

    // Apply L2 regularization
    let mut l2_gradients = gradients.clone();
    let l2_penalty = l2_reg.apply(&params, &mut l2_gradients)?;

    println!("L2 Regularization (alpha = 0.01)");
    println!("--------------------------------");
    println!("Modified Gradients: {:?}", l2_gradients);
    println!("L2 Penalty Term: {:.6}", l2_penalty);
    println!();

    // Create ElasticNet regularizer with strength 0.01 and l1_ratio 0.5
    let elastic_net_reg = ElasticNet::new(0.01, 0.5);

    // Apply ElasticNet regularization
    let mut elastic_net_gradients = gradients.clone();
    let elastic_net_penalty = elastic_net_reg.apply(&params, &mut elastic_net_gradients)?;

    println!("ElasticNet Regularization (alpha = 0.01, l1_ratio = 0.5)");
    println!("-------------------------------------------------------");
    println!("Modified Gradients: {:?}", elastic_net_gradients);
    println!("ElasticNet Penalty Term: {:.6}", elastic_net_penalty);
    println!();

    // Demonstrate effect of regularization on optimization
    println!("Effect of Regularization on Optimization");
    println!("---------------------------------------");

    // Create a parameter vector with some large values
    let params_large = Array1::from_vec(vec![5.0, -3.0, 0.0, 2.0, -8.0, 15.0]);

    // Create gradients
    let gradients_base = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.15, -0.05]);

    // Create optimizers: one without regularization, one with L2
    let mut optimizer_no_reg = SGD::new(0.1);

    // Create a clone of the parameters for each optimization
    let mut params_no_reg = params_large.clone();
    let mut params_with_l2 = params_large.clone();

    // Perform 10 optimization steps
    println!("\nIteration | No Regularization | L2 Regularization");
    println!("------------------------------------------------");

    for i in 0..10 {
        // No regularization
        let gradients_no_reg = gradients_base.clone();
        params_no_reg = optimizer_no_reg.step(&params_no_reg, &gradients_no_reg)?;

        // With L2 regularization
        let mut gradients_with_l2 = gradients_base.clone();
        l2_reg.apply(&params_with_l2, &mut gradients_with_l2)?;
        params_with_l2 = optimizer_no_reg.step(&params_with_l2, &gradients_with_l2)?;

        // Print L2 norm of parameters
        let l2_norm_no_reg = f64::sqrt(params_no_reg.iter().fold(0.0, |acc, &x| acc + x * x));
        let l2_norm_with_l2 = f64::sqrt(params_with_l2.iter().fold(0.0, |acc, &x| acc + x * x));

        println!(
            "{:9} | {:16.6} | {:16.6}",
            i, l2_norm_no_reg, l2_norm_with_l2
        );
    }

    println!("\nFinal Parameters:");
    println!("No Regularization: {:?}", params_no_reg);
    println!("L2 Regularization: {:?}", params_with_l2);

    Ok(())
}
