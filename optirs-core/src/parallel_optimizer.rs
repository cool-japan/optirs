//! Parallel optimizer operations using scirs2_core
//!
//! This module provides parallel processing capabilities for optimizers,
//! enabling efficient multi-core utilization for large-scale optimization.
//!
//! # Features
//!
//! - Parallel parameter group processing
//! - Parallel batch updates
//! - Automatic work distribution across CPU cores
//! - Zero-copy parameter handling
//!
//! # Performance
//!
//! Expected speedup: 4-8x on multi-core systems for multiple parameter groups

use scirs2_core::ndarray::{Array, Array1, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Parallel optimizer wrapper for processing multiple parameter groups
///
/// This wrapper enables parallel processing of multiple parameter groups,
/// providing significant speedup on multi-core systems.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{SGD, Optimizer};
/// use optirs_core::parallel_optimizer::ParallelOptimizer;
///
/// // Create base optimizer
/// let optimizer = SGD::new(0.01);
///
/// // Wrap in parallel optimizer
/// let mut parallel_opt = ParallelOptimizer::new(optimizer);
///
/// // Process multiple parameter groups in parallel
/// let params_list = vec![
///     Array1::zeros(1000),
///     Array1::zeros(2000),
///     Array1::zeros(1500),
/// ];
/// let grads_list = vec![
///     Array1::from_elem(1000, 0.1),
///     Array1::from_elem(2000, 0.1),
///     Array1::from_elem(1500, 0.1),
/// ];
///
/// let updated = parallel_opt.step_parallel_groups(&params_list, &grads_list).expect("parallel_opt.step_parallel_groups succeeds");
/// ```
///
/// # State handling
///
/// Stateful optimizers (Adam, AdamW, LAMB, RAdam, Lion, ...) keep per-parameter
/// moment estimates. A dedicated optimizer instance is therefore materialized and
/// **retained** for every parameter group, and each group's instance is mutated in
/// place across calls. Without this, every call would optimize with a freshly reset
/// clone and, for example, Adam would degenerate into sign-SGD.
#[derive(Debug)]
pub struct ParallelOptimizer<O, A, D>
where
    O: Optimizer<A, D> + Clone + Send + Sync,
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    base_optimizer: O,
    /// Persistent per-group optimizer instances (index == parameter-group index)
    group_optimizers: Vec<O>,
    _phantom_a: std::marker::PhantomData<A>,
    _phantom_d: std::marker::PhantomData<D>,
}

impl<O, A, D> ParallelOptimizer<O, A, D>
where
    O: Optimizer<A, D> + Clone + Send + Sync,
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    /// Creates a new parallel optimizer wrapper
    ///
    /// # Arguments
    ///
    /// * `base_optimizer` - The base optimizer to parallelize
    pub fn new(base_optimizer: O) -> Self {
        Self {
            base_optimizer,
            group_optimizers: Vec::new(),
            _phantom_a: std::marker::PhantomData,
            _phantom_d: std::marker::PhantomData,
        }
    }

    /// Number of parameter groups for which persistent state is currently held
    pub fn num_groups(&self) -> usize {
        self.group_optimizers.len()
    }

    /// Access the persistent optimizer instance of a parameter group
    pub fn group_optimizer(&self, group: usize) -> Option<&O> {
        self.group_optimizers.get(group)
    }

    /// Access the persistent optimizer instance of a parameter group mutably
    pub fn group_optimizer_mut(&mut self, group: usize) -> Option<&mut O> {
        self.group_optimizers.get_mut(group)
    }

    /// Drop all per-group state, restarting every group from the base optimizer
    pub fn reset_group_state(&mut self) {
        self.group_optimizers.clear();
    }

    /// Grow the per-group optimizer pool so that `count` groups have persistent state
    fn ensure_group_optimizers(&mut self, count: usize) {
        while self.group_optimizers.len() < count {
            let fresh = self.base_optimizer.clone();
            self.group_optimizers.push(fresh);
        }
    }

    /// Process multiple parameter groups in parallel
    ///
    /// This method distributes parameter groups across available CPU cores
    /// for parallel processing.
    ///
    /// # Arguments
    ///
    /// * `params_list` - List of parameter arrays
    /// * `grads_list` - List of gradient arrays
    ///
    /// # Returns
    ///
    /// Updated parameter arrays processed in parallel
    pub fn step_parallel_groups(
        &mut self,
        params_list: &[Array<A, D>],
        grads_list: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>>
    where
        Array<A, D>: Clone + Send + Sync,
    {
        if params_list.len() != grads_list.len() {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "Parameter groups ({}) and gradient groups ({}) must have same length",
                params_list.len(),
                grads_list.len()
            )));
        }

        // Materialize (once) and then reuse a persistent optimizer instance per group so
        // that momentum / second-moment state survives across calls.
        let num_groups = params_list.len();
        self.ensure_group_optimizers(num_groups);

        // Use parallel iterator from scirs2_core. Each group mutates its own optimizer,
        // so the borrows are disjoint and no state is discarded.
        let results: Vec<Result<Array<A, D>>> = self.group_optimizers[..num_groups]
            .par_iter_mut()
            .zip(params_list.par_iter())
            .zip(grads_list.par_iter())
            .map(|((optimizer, params), grads)| optimizer.step(params, grads))
            .collect();

        // Collect results and handle errors
        let mut updated_params = Vec::with_capacity(results.len());
        for result in results {
            updated_params.push(result?);
        }

        Ok(updated_params)
    }

    /// Get the underlying optimizer
    pub fn inner(&self) -> &O {
        &self.base_optimizer
    }

    /// Get mutable reference to underlying optimizer
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.base_optimizer
    }

    /// Get the current learning rate from the base optimizer
    pub fn get_learning_rate(&self) -> A {
        self.base_optimizer.get_learning_rate()
    }

    /// Set the learning rate on the base optimizer and on every per-group instance
    pub fn set_learning_rate(&mut self, learning_rate: A) {
        self.base_optimizer.set_learning_rate(learning_rate);
        for optimizer in self.group_optimizers.iter_mut() {
            optimizer.set_learning_rate(learning_rate);
        }
    }
}

/// Parallel batch processor for large parameter arrays
///
/// This processor splits large parameter arrays into chunks and processes
/// them in parallel, providing speedup for very large models.
pub struct ParallelBatchProcessor {
    /// Minimum chunk size for parallel processing
    min_chunk_size: usize,
    /// Number of threads to use (None = automatic)
    num_threads: Option<usize>,
}

impl ParallelBatchProcessor {
    /// Creates a new parallel batch processor
    ///
    /// # Arguments
    ///
    /// * `min_chunk_size` - Minimum size of each chunk (default: 1024)
    pub fn new(min_chunk_size: usize) -> Self {
        Self {
            min_chunk_size,
            num_threads: None,
        }
    }

    /// Set the number of threads to use
    ///
    /// # Arguments
    ///
    /// * `num_threads` - Number of threads (None for automatic)
    pub fn with_threads(mut self, num_threads: Option<usize>) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Determine if parallel processing should be used
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the parameter array
    ///
    /// # Returns
    ///
    /// True if parallel processing would be beneficial
    pub fn should_use_parallel(&self, size: usize) -> bool {
        let num_cores = num_cpus::get();
        size >= self.min_chunk_size * num_cores
    }

    /// Get optimal chunk size for parallel processing
    ///
    /// # Arguments
    ///
    /// * `total_size` - Total size of the array
    ///
    /// # Returns
    ///
    /// Optimal chunk size for parallel processing
    pub fn optimal_chunk_size(&self, total_size: usize) -> usize {
        let num_cores = self.num_threads.unwrap_or_else(num_cpus::get);
        let chunk_size = total_size / num_cores;
        chunk_size.max(self.min_chunk_size)
    }
}

impl Default for ParallelBatchProcessor {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Helper function to update several parameter groups with a single optimizer
///
/// This is a convenience function for one-off multi-group processing without
/// creating a [`ParallelOptimizer`] instance.
///
/// The update is delegated to [`Optimizer::step_list`], which keeps an independent
/// state slot per parameter-group index. Consequently the optimizer state is
/// preserved across calls and groups never share moments.
///
/// # Note on parallelism
///
/// A single `&mut O` cannot be mutated from several threads at once, so this helper
/// walks the groups sequentially. Use [`ParallelOptimizer::step_parallel_groups`]
/// when you want the groups themselves processed in parallel: it keeps one
/// optimizer instance per group and therefore both parallelizes *and* preserves
/// state.
///
/// # Arguments
///
/// * `optimizer` - The optimizer to use (its per-index state is updated in place)
/// * `params_list` - List of parameter arrays
/// * `grads_list` - List of gradient arrays
///
/// # Returns
///
/// Updated parameter arrays
pub fn parallel_step<O, A, D>(
    optimizer: &mut O,
    params_list: &[Array<A, D>],
    grads_list: &[Array<A, D>],
) -> Result<Vec<Array<A, D>>>
where
    O: Optimizer<A, D> + Clone + Send + Sync,
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
    Array<A, D>: Clone + Send + Sync,
{
    if params_list.len() != grads_list.len() {
        return Err(crate::error::OptimError::InvalidConfig(format!(
            "Parameter groups ({}) and gradient groups ({}) must have same length",
            params_list.len(),
            grads_list.len()
        )));
    }

    let params_refs: Vec<&Array<A, D>> = params_list.iter().collect();
    let grads_refs: Vec<&Array<A, D>> = grads_list.iter().collect();
    optimizer.step_list(&params_refs, &grads_refs)
}

/// Multi-group processing for `Array1` specifically (optimized path)
///
/// See [`parallel_step`] for the state and parallelism semantics.
pub fn parallel_step_array1<O, A>(
    optimizer: &mut O,
    params_list: &[Array1<A>],
    grads_list: &[Array1<A>],
) -> Result<Vec<Array1<A>>>
where
    O: Optimizer<A, scirs2_core::ndarray::Ix1> + Clone + Send + Sync,
    A: Float + ScalarOperand + Debug + Send + Sync,
{
    if params_list.len() != grads_list.len() {
        return Err(crate::error::OptimError::InvalidConfig(format!(
            "Parameter groups ({}) and gradient groups ({}) must have same length",
            params_list.len(),
            grads_list.len()
        )));
    }

    let params_refs: Vec<&Array1<A>> = params_list.iter().collect();
    let grads_refs: Vec<&Array1<A>> = grads_list.iter().collect();
    optimizer.step_list(&params_refs, &grads_refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{Adam, SGD};
    use approx::assert_relative_eq;

    #[test]
    fn test_parallel_optimizer_basic() {
        let optimizer = SGD::new(0.1);
        let mut parallel_opt = ParallelOptimizer::new(optimizer);

        let params_list = vec![
            Array1::from_vec(vec![1.0f32, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
        ];
        let grads_list = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3]),
            Array1::from_vec(vec![0.1, 0.2, 0.3]),
        ];

        let results = parallel_opt
            .step_parallel_groups(&params_list, &grads_list)
            .expect("step_parallel_groups succeeds in test_parallel_optimizer_basic");

        assert_eq!(results.len(), 2);
        assert_relative_eq!(results[0][0], 0.99, epsilon = 1e-6);
        assert_relative_eq!(results[1][0], 3.99, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_optimizer_multiple_groups() {
        let optimizer = SGD::new(0.01);
        let mut parallel_opt = ParallelOptimizer::new(optimizer);

        // Create 10 parameter groups
        let params_list: Vec<Array1<f32>> =
            (0..10).map(|i| Array1::from_elem(100, i as f32)).collect();
        let grads_list: Vec<Array1<f32>> = (0..10).map(|_| Array1::from_elem(100, 0.1)).collect();

        let results = parallel_opt
            .step_parallel_groups(&params_list, &grads_list)
            .expect("step_parallel_groups succeeds in test_parallel_optimizer_multiple_groups");

        assert_eq!(results.len(), 10);
        // Verify first group was updated correctly
        assert_relative_eq!(results[0][0], 0.0 - 0.01 * 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_step_function() {
        let mut optimizer = SGD::new(0.1);

        let params_list = vec![
            Array1::from_vec(vec![1.0f32, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        ];
        let grads_list = vec![
            Array1::from_vec(vec![0.1, 0.2]),
            Array1::from_vec(vec![0.3, 0.4]),
        ];

        let results = parallel_step(&mut optimizer, &params_list, &grads_list)
            .expect("parallel_step succeeds in test_parallel_step_function");

        assert_eq!(results.len(), 2);
        assert_relative_eq!(results[0][0], 0.99, epsilon = 1e-6);
        assert_relative_eq!(results[1][0], 2.97, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_batch_processor() {
        let processor = ParallelBatchProcessor::new(1024);

        // Small array - should not use parallel
        assert!(!processor.should_use_parallel(100));

        // Large array - should use parallel
        let num_cores = num_cpus::get();
        assert!(processor.should_use_parallel(1024 * num_cores * 2));

        // Test optimal chunk size calculation
        let chunk_size = processor.optimal_chunk_size(10000);
        assert!(chunk_size >= 1024);
    }

    #[test]
    fn test_parallel_batch_processor_threads() {
        let processor = ParallelBatchProcessor::new(1024).with_threads(Some(4));

        let chunk_size = processor.optimal_chunk_size(10000);
        // With 4 threads, chunk size should be around 10000/4 = 2500
        assert!(chunk_size >= 1024);
        assert!(chunk_size <= 10000);
    }

    #[test]
    fn test_parallel_optimizer_learning_rate() {
        let optimizer = SGD::new(0.1);
        let mut parallel_opt: ParallelOptimizer<_, f64, scirs2_core::ndarray::Ix1> =
            ParallelOptimizer::new(optimizer);

        assert_relative_eq!(parallel_opt.get_learning_rate(), 0.1, epsilon = 1e-6);

        parallel_opt.set_learning_rate(0.2);
        assert_relative_eq!(parallel_opt.get_learning_rate(), 0.2, epsilon = 1e-6);
    }

    /// Regression test for the "parallel wrapper discards optimizer state" bug.
    ///
    /// The wrapper used to clone the base optimizer on every call and throw the clone
    /// away, so Adam always ran at t=1 with zero moments, i.e. it degenerated into
    /// sign-SGD (`lr * sign(g)`), and a zero gradient produced no movement at all.
    #[test]
    fn test_parallel_optimizer_preserves_adam_state() {
        let optimizer = Adam::new(0.1f64);
        let mut parallel_opt = ParallelOptimizer::new(optimizer);

        let params = vec![Array1::from_vec(vec![0.0f64])];
        let grads_first = vec![Array1::from_vec(vec![1.0f64])];
        let grads_second = vec![Array1::from_vec(vec![0.0f64])];

        let after_first = parallel_opt
            .step_parallel_groups(&params, &grads_first)
            .expect("first parallel step failed");
        // t = 1 with a unit gradient is exactly -lr for Adam.
        assert_relative_eq!(after_first[0][0], -0.1, epsilon = 1e-9);

        let after_second = parallel_opt
            .step_parallel_groups(&after_first, &grads_second)
            .expect("second parallel step failed");

        // A stateless (bugged) optimizer sees m = v = 0 and does not move at all.
        assert!(
            (after_second[0][0] - after_first[0][0]).abs() > 1e-3,
            "optimizer state was discarded between calls: {} vs {}",
            after_second[0][0],
            after_first[0][0]
        );

        // With retained state: m = 0.09, v = 0.000999 => step ~= 0.067014
        assert_relative_eq!(after_second[0][0], -0.167014, epsilon = 1e-5);
        assert_eq!(parallel_opt.num_groups(), 1);
    }

    /// Two groups must never share optimizer state.
    #[test]
    fn test_parallel_optimizer_groups_have_independent_state() {
        let optimizer = Adam::new(0.1f64);
        let mut parallel_opt = ParallelOptimizer::new(optimizer);

        let params = vec![
            Array1::from_vec(vec![0.0f64]),
            Array1::from_vec(vec![0.0f64, 0.0]),
        ];
        let grads = vec![
            Array1::from_vec(vec![1.0f64]),
            Array1::from_vec(vec![1.0f64, 1.0]),
        ];

        let first = parallel_opt
            .step_parallel_groups(&params, &grads)
            .expect("first parallel step failed");
        assert_eq!(parallel_opt.num_groups(), 2);
        assert_eq!(first[0].len(), 1);
        assert_eq!(first[1].len(), 2);

        let second = parallel_opt
            .step_parallel_groups(&first, &grads)
            .expect("second parallel step failed");

        // Both groups are at t = 2 with identical gradients, so both must agree.
        assert_relative_eq!(second[0][0], second[1][0], epsilon = 1e-12);
        assert_relative_eq!(second[1][0], second[1][1], epsilon = 1e-12);

        // And the second step must be smaller than the first (bias correction at t=2).
        let first_delta = (first[0][0] - 0.0).abs();
        let second_delta = (second[0][0] - first[0][0]).abs();
        assert!(second_delta < first_delta);
    }

    /// `parallel_step_array1` keeps per-group state in the optimizer it is given.
    #[test]
    fn test_parallel_step_array1_preserves_state() {
        let mut optimizer = Adam::new(0.1f64);

        let params = vec![Array1::from_vec(vec![0.0f64])];
        let grads_first = vec![Array1::from_vec(vec![1.0f64])];
        let grads_second = vec![Array1::from_vec(vec![0.0f64])];

        let first =
            parallel_step_array1(&mut optimizer, &params, &grads_first).expect("first step failed");
        let second = parallel_step_array1(&mut optimizer, &first, &grads_second)
            .expect("second step failed");

        assert_relative_eq!(first[0][0], -0.1, epsilon = 1e-9);
        assert_relative_eq!(second[0][0], -0.167014, epsilon = 1e-5);
    }

    #[test]
    fn test_parallel_optimizer_learning_rate_propagates_to_groups() {
        let optimizer = SGD::new(0.1f64);
        let mut parallel_opt = ParallelOptimizer::new(optimizer);

        let params = vec![Array1::from_vec(vec![1.0f64])];
        let grads = vec![Array1::from_vec(vec![1.0f64])];

        let _ = parallel_opt
            .step_parallel_groups(&params, &grads)
            .expect("step failed");
        parallel_opt.set_learning_rate(0.5);

        let updated = parallel_opt
            .step_parallel_groups(&params, &grads)
            .expect("step failed");
        assert_relative_eq!(updated[0][0], 0.5, epsilon = 1e-9);
    }

    #[test]
    fn test_parallel_step_array1() {
        let mut optimizer = SGD::new(0.1);

        let params_list = vec![
            Array1::from_vec(vec![1.0f32, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
        ];
        let grads_list = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3]),
            Array1::from_vec(vec![0.1, 0.2, 0.3]),
        ];

        let results = parallel_step_array1(&mut optimizer, &params_list, &grads_list)
            .expect("parallel_step_array1 succeeds in test_parallel_step_array1");

        assert_eq!(results.len(), 2);
        assert_relative_eq!(results[0][0], 0.99, epsilon = 1e-6);
        assert_relative_eq!(results[1][0], 3.99, epsilon = 1e-6);
    }
}
