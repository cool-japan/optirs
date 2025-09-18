// XLA frontend components
//
// This module contains the frontend components for XLA compilation,
// including graph capture, operation lowering, and shape inference.

pub mod graph_capture;
pub mod operation_lowering;
pub mod shape_inference;

pub use graph_capture::*;
pub use operation_lowering::*;
pub use shape_inference::*;
