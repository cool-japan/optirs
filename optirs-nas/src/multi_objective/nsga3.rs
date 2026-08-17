//! NSGA-III scaffolding (reference-direction based selection; not yet wired to a real algorithm).

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::nsga2::NSGA2;

/// NSGA-III implementation
pub struct NSGA3<T: Float + Debug + Send + Sync + 'static> {
    /// Base NSGA-II functionality
    pub(super) base: NSGA2<T>,
    /// Reference directions
    pub(super) reference_directions: Vec<Array1<T>>,
    /// Association count for each reference direction
    pub(super) association_count: Vec<usize>,
    /// Niche count for each reference direction
    pub(super) niche_count: Vec<usize>,
}
