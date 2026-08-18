//! # `StaticPriorityStrategy` - Trait Implementations
//!
//! This module contains trait implementations for `StaticPriorityStrategy`.
//!
//! ## Implemented Traits
//!
//! - `PriorityUpdateStrategy`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::PriorityUpdateStrategy;
use super::types::{PriorityItem, PriorityUpdateContext, StaticPriorityStrategy};

impl<T: Float + Debug + Send + Sync + 'static> PriorityUpdateStrategy<T>
    for StaticPriorityStrategy
{
    fn update_priorities(
        &mut self,
        _items: &mut [PriorityItem<T>],
        _context: &PriorityUpdateContext<T>,
    ) -> Result<()> {
        // Static strategy doesn't update priorities
        Ok(())
    }

    fn name(&self) -> &str {
        "Static"
    }

    fn get_metrics(&self) -> HashMap<String, T> {
        HashMap::new()
    }

    fn configure(&mut self, _config: &HashMap<String, T>) -> Result<()> {
        Ok(())
    }
}
