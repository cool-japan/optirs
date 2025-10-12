// XLA frontend components
//
// This module contains the frontend components for XLA compilation,
// including graph capture, operation lowering, and shape inference.

pub mod graph_capture;
pub mod operation_lowering;
pub mod shape_inference;

// Re-export main types selectively to avoid ambiguous glob re-exports
// (ShapePattern exists in both operation_lowering and shape_inference)
pub use graph_capture::*;
pub use operation_lowering::OperationLowering;
pub use shape_inference::ShapeInference;
