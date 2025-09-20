// Progressive neural architecture search
//
// This module implements progressive search strategies that gradually
// refine the search space and increase complexity over time.

use num_traits::Float;
use std::collections::HashMap;

pub use super::differentiable::{ProgressiveSearchState, SearchPhase, MultiObjectiveState, MultiObjectiveAlgorithm};