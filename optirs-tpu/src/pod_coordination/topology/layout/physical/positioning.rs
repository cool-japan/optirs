// Positioning Module

pub type NodeId = u64;

#[derive(Debug, Clone, Default)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
