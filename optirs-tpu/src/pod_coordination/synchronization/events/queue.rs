// Queue Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Default)]
pub struct EventQueue {
    pub events: VecDeque<Event>,
}

#[derive(Debug, Clone)]
pub struct Event;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueuePolicy {
    FIFO,
    LIFO,
    Priority,
}

impl Default for QueuePolicy {
    fn default() -> Self {
        Self::FIFO
    }
}
