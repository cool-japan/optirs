//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::Array1;

/// Type alias for optimizer functions taking parameters and gradients
pub(super) type OptimizerFn<A> = Box<dyn Fn(&Array1<A>, &Array1<A>) -> Array1<A>>;

/// Type alias for optimizer function references (trait objects)
pub(super) type OptimizerFnRef<'a, A> = &'a dyn Fn(&Array1<A>, &Array1<A>) -> Array1<A>;
