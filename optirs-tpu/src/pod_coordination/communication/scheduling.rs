// Communication Scheduling Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SchedulingError;

pub type SchedulingResult<T> = std::result::Result<T, SchedulingError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    FIFO,
    Priority,
    RoundRobin,
    Weighted,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self::Priority
    }
}

#[derive(Debug, Clone, Default)]
pub struct Scheduler {
    pub policy: SchedulingPolicy,
}
