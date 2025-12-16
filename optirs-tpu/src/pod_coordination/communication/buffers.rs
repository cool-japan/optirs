// Communication Buffers Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct BufferError;

pub type BufferResult<T> = std::result::Result<T, BufferError>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Buffer {
    pub capacity: usize,
    pub used: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BufferManager {
    pub buffers: Vec<Buffer>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum BufferStrategy {
    RingBuffer,
    #[default]
    DoubleBuffer,
    TripleBuffer,
}
