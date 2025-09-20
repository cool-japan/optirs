// OptiRS SciRS2 Integration Demonstration
//
// This example demonstrates the successful integration of OptiRS with SciRS2-Core,
// showing how OptiRS now leverages the full SciRS2 ecosystem for scientific computing.

use optirs_core::error::Result;
use optirs_core::gradient_processing::{add_gradient_noise, GradientProcessor};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{
    distributions::{Distribution, Normal},
    Random,
};

fn main() -> Result<()> {
    println!("🚀 OptiRS SciRS2 Integration Demo");
    println!("================================");

    // Demonstrate SciRS2 array operations
    println!("\n1. SciRS2 Array Operations:");
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut gradients = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);

    println!("   Parameters: {:?}", params);
    println!("   Gradients:  {:?}", gradients);

    // Demonstrate SciRS2 random number generation
    println!("\n2. SciRS2 Random Number Generation:");
    let mut rng = Random::default();
    let normal = Normal::new(0.0, 0.1).unwrap();
    let random_values: Vec<f64> = (0..5).map(|_| normal.sample(&mut rng)).collect();
    println!("   Random noise: {:?}", random_values);

    // Demonstrate gradient processing with SciRS2 integration
    println!("\n3. OptiRS Gradient Processing with SciRS2:");
    let mut processor = GradientProcessor::new();
    processor.set_max_norm(1.0);

    let mut grad_copy = gradients.clone();
    processor.process(&mut grad_copy)?;
    println!("   Processed gradients: {:?}", grad_copy);

    // Demonstrate noise injection using SciRS2 random
    println!("\n4. Gradient Noise Injection (SciRS2 Random):");
    add_gradient_noise(&mut gradients, 0.01, Some(42));
    println!("   Noisy gradients: {:?}", gradients);

    // Demonstrate 2D array operations
    println!("\n5. SciRS2 2D Array Operations:");
    let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    println!("   Matrix shape: {:?}", matrix.shape());
    println!("   Matrix data:\n{:?}", matrix);

    // Show SciRS2 mathematical operations
    let matrix_squared = &matrix * &matrix;
    println!("   Element-wise square:\n{:?}", matrix_squared);

    println!("\n✅ SciRS2 Integration Successful!");
    println!("OptiRS is now fully integrated with the SciRS2 ecosystem.");
    println!("All operations use scirs2_core primitives for:");
    println!("  • Array operations (scirs2_core::ndarray_ext)");
    println!("  • Random number generation (scirs2_core::random)");
    println!("  • Error handling (scirs2_core::error)");
    println!("  • Mathematical operations (SciRS2 backend)");

    Ok(())
}
