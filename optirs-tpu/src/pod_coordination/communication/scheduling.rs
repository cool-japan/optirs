// Communication Scheduling Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SchedulingError;

pub type SchedulingResult<T> = std::result::Result<T, SchedulingError>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SchedulingPolicy {
    FIFO,
    #[default]
    Priority,
    RoundRobin,
    Weighted,
}

#[derive(Debug, Clone, Default)]
pub struct Scheduler {
    pub policy: SchedulingPolicy,
}
