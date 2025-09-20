// Source Selection Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SelectionCriterion {
    pub priority: u32,
    pub weight: f64,
}

pub type SourceSelectionCriteria = Vec<SelectionCriterion>;

#[derive(Debug, Clone, Default)]
pub struct SourceSelector {
    pub criteria: SourceSelectionCriteria,
}
