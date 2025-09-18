use std::fmt::Debug;
// Shape inference and validation for XLA operations
//
// This module provides comprehensive shape inference capabilities for XLA computations,
// including static and dynamic shape analysis, constraint validation, and shape optimization.

use scirs2_core::ndarray_ext::{Array1, Array2, Dimension};
use num_traits::Float;
use std::collections::{HashMap, HashSet};

use super::graph_capture::{
    ConvolutionConfig, DataType, OperandId, OperationType, ReduceOperation, TensorShape,
    XLAComputation, XLAOperation,
};
use crate::error::{OptimError, Result};

/// Shape inference engine for XLA operations
pub struct ShapeInference {
    /// Shape inference rules registry
    inference_rules: HashMap<String, ShapeInferenceRule>,

    /// Broadcasting rules
    broadcasting_rules: Vec<BroadcastingRule>,

    /// Shape constraints
    constraints: Vec<ShapeConstraint>,

    /// Dynamic shape tracker
    dynamic_shapes: HashMap<OperandId, DynamicShapeInfo>,
}

/// Shape inference rule for operations
#[derive(Debug, Clone)]
pub struct ShapeInferenceRule {
    /// Rule name
    pub name: String,

    /// Operation type this rule applies to
    pub operation_type: String,

    /// Input shape requirements
    pub input_requirements: Vec<ShapeRequirement>,

    /// Output shape computation
    pub output_shape_fn: String,

    /// Constraints that must be satisfied
    pub constraints: Vec<String>,
}

/// Shape requirement for operation inputs
#[derive(Debug, Clone)]
pub struct ShapeRequirement {
    /// Input index
    pub input_index: usize,

    /// Required rank (None means any rank)
    pub required_rank: Option<usize>,

    /// Required dimensions (None means any size)
    pub required_dimensions: Vec<Option<usize>>,

    /// Data type requirements
    pub data_type_requirements: Vec<DataType>,

    /// Additional constraints
    pub additional_constraints: Vec<String>,
}

/// Broadcasting rule for shape compatibility
#[derive(Debug, Clone)]
pub struct BroadcastingRule {
    /// Rule name
    pub name: String,

    /// Compatible shape patterns
    pub compatible_patterns: Vec<ShapePattern>,

    /// Result shape computation
    pub result_shape_fn: String,
}

/// Shape pattern for matching
#[derive(Debug, Clone)]
pub enum ShapePattern {
    /// Exact shape match
    Exact(Vec<usize>),

    /// Broadcast compatible (trailing dimensions)
    BroadcastCompatible,

    /// Prefix shape (leading dimensions match)
    Prefix(Vec<usize>),

    /// Suffix shape (trailing dimensions match)
    Suffix(Vec<usize>),

    /// Any shape with specific rank
    AnyRank(usize),

    /// Any shape
    Any,
}

/// Shape constraint for validation
#[derive(Debug, Clone)]
pub struct ShapeConstraint {
    /// Constraint name
    pub name: String,

    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Operands this constraint applies to
    pub operands: Vec<OperandId>,

    /// Constraint parameters
    pub parameters: HashMap<String, String>,

    /// Error message if constraint fails
    pub error_message: String,
}

/// Types of shape constraints
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Shapes must be identical
    IdenticalShapes,

    /// Shapes must be broadcast compatible
    BroadcastCompatible,

    /// Ranks must match
    MatchingRanks,

    /// Specific dimensions must match
    DimensionMatch { dims: Vec<usize> },

    /// Element count must match
    ElementCountMatch,

    /// Shape must satisfy custom predicate
    CustomPredicate(String),
}

/// Dynamic shape information
#[derive(Debug, Clone)]
pub struct DynamicShapeInfo {
    /// Known static dimensions
    pub static_dimensions: HashMap<usize, usize>,

    /// Dynamic dimension constraints
    pub dynamic_constraints: Vec<DynamicConstraint>,

    /// Upper bounds for dynamic dimensions
    pub upper_bounds: HashMap<usize, usize>,

    /// Lower bounds for dynamic dimensions
    pub lower_bounds: HashMap<usize, usize>,
}

/// Dynamic shape constraints
#[derive(Debug, Clone)]
pub struct DynamicConstraint {
    /// Constraint type
    pub constraint_type: DynamicConstraintType,

    /// Dimensions involved
    pub dimensions: Vec<usize>,

    /// Constraint parameters
    pub parameters: HashMap<String, i64>,
}

/// Types of dynamic constraints
#[derive(Debug, Clone)]
pub enum DynamicConstraintType {
    /// Dimension sizes must be equal
    Equal,

    /// Dimension must be divisible by value
    Divisible,

    /// Dimension must be in range
    Range,

    /// Dimension must be multiple of another
    Multiple,
}

/// Shape inference context
#[derive(Debug)]
pub struct ShapeInferenceContext<T: Float + Debug + Send + Sync + 'static> {
    /// Current computation being analyzed
    pub computation: XLAComputation<T>,

    /// Inferred shapes for operands
    pub inferred_shapes: HashMap<OperandId, InferredShape>,

    /// Shape constraints discovered
    pub discovered_constraints: Vec<ShapeConstraint>,

    /// Dynamic shape information
    pub dynamic_info: HashMap<OperandId, DynamicShapeInfo>,

    /// Inference options
    pub options: ShapeInferenceOptions,
}

/// Inferred shape information
#[derive(Debug, Clone)]
pub struct InferredShape {
    /// Static shape (if fully known)
    pub static_shape: Option<TensorShape>,

    /// Dynamic shape template
    pub dynamic_shape: Option<DynamicShapeTemplate>,

    /// Confidence level (0.0-1.0)
    pub confidence: f64,

    /// Inference method used
    pub inference_method: String,

    /// Alternative shapes (for ambiguous cases)
    pub alternatives: Vec<TensorShape>,
}

/// Dynamic shape template
#[derive(Debug, Clone)]
pub struct DynamicShapeTemplate {
    /// Template dimensions (None = dynamic)
    pub template_dims: Vec<Option<usize>>,

    /// Dynamic dimension symbols
    pub symbols: HashMap<usize, String>,

    /// Symbolic expressions for dimensions
    pub expressions: HashMap<usize, String>,
}

/// Shape inference options
#[derive(Debug, Clone, Default)]
pub struct ShapeInferenceOptions {
    /// Allow dynamic shapes
    pub allow_dynamic_shapes: bool,

    /// Strict mode (fail on any ambiguity)
    pub strict_mode: bool,

    /// Maximum inference iterations
    pub max_iterations: usize,

    /// Enable shape optimization
    pub enable_optimization: bool,

    /// Propagate constraints backwards
    pub backward_propagation: bool,
}

impl ShapeInference {
    /// Create new shape inference engine
    pub fn new() -> Self {
        let mut inference = Self {
            inference_rules: HashMap::new(),
            broadcasting_rules: Vec::new(),
            constraints: Vec::new(),
            dynamic_shapes: HashMap::new(),
        };

        inference.initialize_builtin_rules();
        inference
    }

    /// Initialize built-in shape inference rules
    fn initialize_builtin_rules(&mut self) {
        // Add elementwise operation rules
        self.add_inference_rule(ShapeInferenceRule {
            name: "elementwise_binary".to_string(),
            operation_type: "Add".to_string(),
            input_requirements: vec![
                ShapeRequirement {
                    input_index: 0,
                    required_rank: None,
                    required_dimensions: vec![],
                    data_type_requirements: vec![],
                    additional_constraints: vec![],
                },
                ShapeRequirement {
                    input_index: 1,
                    required_rank: None,
                    required_dimensions: vec![],
                    data_type_requirements: vec![],
                    additional_constraints: vec!["broadcast_compatible_with_input_0".to_string()],
                },
            ],
            output_shape_fn: "broadcast_result".to_string(),
            constraints: vec!["inputs_broadcast_compatible".to_string()],
        });

        // Add dot product rule
        self.add_inference_rule(ShapeInferenceRule {
            name: "dot_product".to_string(),
            operation_type: "Dot".to_string(),
            input_requirements: vec![
                ShapeRequirement {
                    input_index: 0,
                    required_rank: Some(2),
                    required_dimensions: vec![None, None],
                    data_type_requirements: vec![],
                    additional_constraints: vec![],
                },
                ShapeRequirement {
                    input_index: 1,
                    required_rank: Some(2),
                    required_dimensions: vec![None, None],
                    data_type_requirements: vec![],
                    additional_constraints: vec!["inner_dimension_matches".to_string()],
                },
            ],
            output_shape_fn: "matrix_multiply_result".to_string(),
            constraints: vec!["inner_dimensions_match".to_string()],
        });

        // Add broadcasting rules
        self.add_broadcasting_rule(BroadcastingRule {
            name: "standard_broadcast".to_string(),
            compatible_patterns: vec![ShapePattern::BroadcastCompatible],
            result_shape_fn: "broadcast_shapes".to_string(),
        });
    }

    /// Infer shapes for entire computation
    pub fn infer_shapes<T: Float + Default + std::fmt::Debug + Clone>(
        computation: XLAComputation<T>,
    ) -> Result<XLAComputation<T>> {
        let mut inference = Self::new();
        let options = ShapeInferenceOptions::default();

        let mut context = ShapeInferenceContext {
            computation,
            inferred_shapes: HashMap::new(),
            discovered_constraints: Vec::new(),
            dynamic_info: HashMap::new(),
            options,
        };

        inference.run_inference(&mut context)
    }

    /// Run shape inference on computation
    fn run_inference<T: Float + Default + std::fmt::Debug + Clone>(
        &mut self,
        context: &mut ShapeInferenceContext<T>,
    ) -> Result<XLAComputation<T>> {
        // Initialize with input shapes
        self.initialize_input_shapes(context)?;

        // Iterative shape inference
        for iteration in 0..context.options.max_iterations {
            let mut changed = false;

            for operation in &context.computation.operations {
                if self.infer_operation_shape(operation, context)? {
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Validate constraints
            self.validate_constraints(context)?;
        }

        // Finalize shapes
        self.finalize_shapes(context)
    }

    /// Initialize shapes from computation inputs
    fn initialize_input_shapes<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        context: &mut ShapeInferenceContext<T>,
    ) -> Result<()> {
        for input_spec in &context.computation.inputs {
            let inferred = InferredShape {
                static_shape: Some(input_spec.shape.clone()),
                dynamic_shape: None,
                confidence: 1.0,
                inference_method: "input_specification".to_string(),
                alternatives: vec![],
            };

            // Find corresponding operand
            for (operand_id, operand) in &context.computation.operands {
                if operand.shape == input_spec.shape {
                    context
                        .inferred_shapes
                        .insert(*operand_id, inferred.clone());
                    break;
                }
            }
        }

        Ok(())
    }

    /// Infer shape for single operation
    fn infer_operation_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        operation: &XLAOperation<T>,
        context: &mut ShapeInferenceContext<T>,
    ) -> Result<bool> {
        // Check if output shape already inferred
        if context.inferred_shapes.contains_key(&operation.output) {
            return Ok(false);
        }

        // Check if all input shapes are available
        let input_shapes: Vec<Option<&InferredShape>> = operation
            .inputs
            .iter()
            .map(|&operand_id| context.inferred_shapes.get(&operand_id))
            .collect();

        if input_shapes.iter().any(|shape| shape.is_none()) {
            return Ok(false); // Not ready yet
        }

        // Infer shape based on operation type
        let output_shape = match &operation.op_type {
            OperationType::Add
            | OperationType::Multiply
            | OperationType::Subtract
            | OperationType::Divide => self.infer_elementwise_shape(&input_shapes, context)?,
            OperationType::Dot => self.infer_dot_shape(&input_shapes, context)?,
            OperationType::Reshape => {
                self.infer_reshape_shape(operation, &input_shapes, context)?
            }
            OperationType::Transpose => {
                self.infer_transpose_shape(operation, &input_shapes, context)?
            }
            OperationType::Reduce(reduce_op) => {
                self.infer_reduce_shape(reduce_op, &input_shapes, context)?
            }
            OperationType::Convolution(conv_config) => {
                self.infer_convolution_shape(conv_config, &input_shapes, context)?
            }
            OperationType::Constant(value) => self.infer_constant_shape(value, context)?,
            _ => {
                // Default inference - use first input shape
                if let Some(Some(first_input)) = input_shapes.first() {
                    first_input.clone()
                } else {
                    return Ok(false);
                }
            }
        };

        context
            .inferred_shapes
            .insert(operation.output, output_shape);
        Ok(true)
    }

    /// Infer shape for elementwise operations
    fn infer_elementwise_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        input_shapes: &[Option<&InferredShape>],
        _context: &ShapeInferenceContext<T>,
    ) -> Result<InferredShape> {
        if input_shapes.len() != 2 {
            return Err(OptimError::InvalidShape(
                "Elementwise operations require exactly 2 inputs".to_string(),
            ));
        }

        let shape1 = input_shapes[0].unwrap();
        let shape2 = input_shapes[1].unwrap();

        // Broadcast shapes
        if let (Some(static1), Some(static2)) = (&shape1.static_shape, &shape2.static_shape) {
            let result_shape = self.broadcast_shapes(static1, static2)?;

            Ok(InferredShape {
                static_shape: Some(result_shape),
                dynamic_shape: None,
                confidence: (shape1.confidence * shape2.confidence).min(1.0),
                inference_method: "elementwise_broadcast".to_string(),
                alternatives: vec![],
            })
        } else {
            // Handle dynamic shapes
            Ok(InferredShape {
                static_shape: None,
                dynamic_shape: None,
                confidence: 0.5,
                inference_method: "elementwise_dynamic".to_string(),
                alternatives: vec![],
            })
        }
    }

    /// Infer shape for dot product operations
    fn infer_dot_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        input_shapes: &[Option<&InferredShape>],
        _context: &ShapeInferenceContext<T>,
    ) -> Result<InferredShape> {
        if input_shapes.len() != 2 {
            return Err(OptimError::InvalidShape(
                "Dot operations require exactly 2 inputs".to_string(),
            ));
        }

        let shape1 = input_shapes[0].unwrap();
        let shape2 = input_shapes[1].unwrap();

        if let (Some(static1), Some(static2)) = (&shape1.static_shape, &shape2.static_shape) {
            // Matrix multiplication: [M, K] x [K, N] -> [M, N]
            if static1.dimensions.len() == 2 && static2.dimensions.len() == 2 {
                if static1.dimensions[1] != static2.dimensions[0] {
                    return Err(OptimError::InvalidShape(format!(
                        "Incompatible dimensions for dot product: {} vs {}",
                        static1.dimensions[1], static2.dimensions[0]
                    )));
                }

                let result_shape = TensorShape {
                    dimensions: vec![static1.dimensions[0], static2.dimensions[1]],
                    dynamic_dimensions: vec![false, false],
                    element_count: static1.dimensions[0] * static2.dimensions[1],
                    tuple_shapes: vec![],
                };

                Ok(InferredShape {
                    static_shape: Some(result_shape),
                    dynamic_shape: None,
                    confidence: (shape1.confidence * shape2.confidence).min(1.0),
                    inference_method: "matrix_multiply".to_string(),
                    alternatives: vec![],
                })
            } else {
                Err(OptimError::InvalidShape(
                    "Dot product requires 2D tensors".to_string(),
                ))
            }
        } else {
            // Handle dynamic shapes
            Ok(InferredShape {
                static_shape: None,
                dynamic_shape: None,
                confidence: 0.5,
                inference_method: "dot_dynamic".to_string(),
                alternatives: vec![],
            })
        }
    }

    /// Infer shape for reshape operations
    fn infer_reshape_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        _operation: &XLAOperation<T>,
        input_shapes: &[Option<&InferredShape>],
        _context: &ShapeInferenceContext<T>,
    ) -> Result<InferredShape> {
        if let Some(Some(input_shape)) = input_shapes.first() {
            // For now, return the input shape (reshape target would be in attributes)
            Ok(input_shape.clone())
        } else {
            Err(OptimError::InvalidShape(
                "Reshape requires input shape".to_string(),
            ))
        }
    }

    /// Infer shape for transpose operations
    fn infer_transpose_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        _operation: &XLAOperation<T>,
        input_shapes: &[Option<&InferredShape>],
        _context: &ShapeInferenceContext<T>,
    ) -> Result<InferredShape> {
        if let Some(Some(input_shape)) = input_shapes.first() {
            if let Some(static_shape) = &input_shape.static_shape {
                // Default transpose - reverse dimensions
                let mut new_dims = static_shape.dimensions.clone();
                new_dims.reverse();

                let result_shape = TensorShape {
                    dimensions: new_dims,
                    dynamic_dimensions: static_shape
                        .dynamic_dimensions
                        .iter()
                        .rev()
                        .cloned()
                        .collect(),
                    element_count: static_shape.element_count,
                    tuple_shapes: vec![],
                };

                Ok(InferredShape {
                    static_shape: Some(result_shape),
                    dynamic_shape: None,
                    confidence: input_shape.confidence,
                    inference_method: "transpose".to_string(),
                    alternatives: vec![],
                })
            } else {
                Ok(input_shape.clone())
            }
        } else {
            Err(OptimError::InvalidShape(
                "Transpose requires input shape".to_string(),
            ))
        }
    }

    /// Infer shape for reduction operations
    fn infer_reduce_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        reduce_op: &ReduceOperation,
        input_shapes: &[Option<&InferredShape>],
        _context: &ShapeInferenceContext<T>,
    ) -> Result<InferredShape> {
        if let Some(Some(input_shape)) = input_shapes.first() {
            if let Some(static_shape) = &input_shape.static_shape {
                // Reduce specified dimensions
                let mut result_dims = static_shape.dimensions.clone();
                let mut result_dynamic = static_shape.dynamic_dimensions.clone();

                // Remove reduced dimensions (in reverse order to maintain indices)
                let mut sorted_dims = reduce_op.dimensions.clone();
                sorted_dims.sort_by(|a, b| b.cmp(a));

                for &dim in &sorted_dims {
                    if dim < result_dims.len() {
                        result_dims.remove(dim);
                        result_dynamic.remove(dim);
                    }
                }

                let element_count = result_dims.iter().product();

                let result_shape = TensorShape {
                    dimensions: result_dims,
                    dynamic_dimensions: result_dynamic,
                    element_count,
                    tuple_shapes: vec![],
                };

                Ok(InferredShape {
                    static_shape: Some(result_shape),
                    dynamic_shape: None,
                    confidence: input_shape.confidence,
                    inference_method: "reduce".to_string(),
                    alternatives: vec![],
                })
            } else {
                Ok(input_shape.clone())
            }
        } else {
            Err(OptimError::InvalidShape(
                "Reduce requires input shape".to_string(),
            ))
        }
    }

    /// Infer shape for convolution operations  
    fn infer_convolution_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        _conv_config: &ConvolutionConfig,
        input_shapes: &[Option<&InferredShape>],
        _context: &ShapeInferenceContext<T>,
    ) -> Result<InferredShape> {
        // Simplified convolution shape inference
        if let Some(Some(input_shape)) = input_shapes.first() {
            Ok(input_shape.clone())
        } else {
            Err(OptimError::InvalidShape(
                "Convolution requires input shape".to_string(),
            ))
        }
    }

    /// Infer shape for constant operations
    fn infer_constant_shape<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        _value: &T,
        _context: &ShapeInferenceContext<T>,
    ) -> Result<InferredShape> {
        // Constants are scalars
        let result_shape = TensorShape {
            dimensions: vec![],
            dynamic_dimensions: vec![],
            element_count: 1,
            tuple_shapes: vec![],
        };

        Ok(InferredShape {
            static_shape: Some(result_shape),
            dynamic_shape: None,
            confidence: 1.0,
            inference_method: "constant".to_string(),
            alternatives: vec![],
        })
    }

    /// Broadcast two shapes
    fn broadcast_shapes(&self, shape1: &TensorShape, shape2: &TensorShape) -> Result<TensorShape> {
        let dims1 = &shape1.dimensions;
        let dims2 = &shape2.dimensions;

        let max_rank = dims1.len().max(dims2.len());
        let mut result_dims = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let dim1 = if i < dims1.len() {
                dims1[dims1.len() - 1 - i]
            } else {
                1
            };
            let dim2 = if i < dims2.len() {
                dims2[dims2.len() - 1 - i]
            } else {
                1
            };

            if dim1 == dim2 {
                result_dims.push(dim1);
            } else if dim1 == 1 {
                result_dims.push(dim2);
            } else if dim2 == 1 {
                result_dims.push(dim1);
            } else {
                return Err(OptimError::InvalidShape(format!(
                    "Incompatible dimensions for broadcasting: {} vs {}",
                    dim1, dim2
                )));
            }
        }

        result_dims.reverse();
        let element_count = result_dims.iter().product();

        Ok(TensorShape {
            dimensions: result_dims,
            dynamic_dimensions: vec![false; max_rank],
            element_count,
            tuple_shapes: vec![],
        })
    }

    /// Validate all constraints
    fn validate_constraints<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        _context: &ShapeInferenceContext<T>,
    ) -> Result<()> {
        // Constraint validation logic would go here
        Ok(())
    }

    /// Finalize shapes and update computation
    fn finalize_shapes<T: Float + Default + std::fmt::Debug + Clone>(
        &self,
        context: &mut ShapeInferenceContext<T>,
    ) -> Result<XLAComputation<T>> {
        let mut computation = context.computation.clone();

        // Update operand shapes with inferred shapes
        for (operand_id, inferred) in &context.inferred_shapes {
            if let Some(operand) = computation.operands.get_mut(operand_id) {
                if let Some(static_shape) = &inferred.static_shape {
                    operand.shape = static_shape.clone();
                }
            }
        }

        Ok(computation)
    }

    /// Add shape inference rule
    fn add_inference_rule(&mut self, rule: ShapeInferenceRule) {
        self.inference_rules.insert(rule.name.clone(), rule);
    }

    /// Add broadcasting rule
    fn add_broadcasting_rule(&mut self, rule: BroadcastingRule) {
        self.broadcasting_rules.push(rule);
    }
}

impl Default for ShapeInferenceOptions {
    fn default() -> Self {
        Self {
            allow_dynamic_shapes: true,
            strict_mode: false,
            max_iterations: 10,
            enable_optimization: true,
            backward_propagation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_inference_creation() {
        let inference = ShapeInference::new();
        assert!(!inference.inference_rules.is_empty());
        assert!(!inference.broadcasting_rules.is_empty());
    }

    #[test]
    fn test_broadcast_shapes() {
        let inference = ShapeInference::new();

        let shape1 = TensorShape {
            dimensions: vec![3, 1],
            dynamic_dimensions: vec![false, false],
            element_count: 3,
            tuple_shapes: vec![],
        };

        let shape2 = TensorShape {
            dimensions: vec![1, 4],
            dynamic_dimensions: vec![false, false],
            element_count: 4,
            tuple_shapes: vec![],
        };

        let result = inference.broadcast_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result.dimensions, vec![3, 4]);
        assert_eq!(result.element_count, 12);
    }

    #[test]
    fn test_dot_shape_inference() {
        let inference = ShapeInference::new();

        let shape1 = InferredShape {
            static_shape: Some(TensorShape {
                dimensions: vec![2, 3],
                dynamic_dimensions: vec![false, false],
                element_count: 6,
                tuple_shapes: vec![],
            }),
            dynamic_shape: None,
            confidence: 1.0,
            inference_method: "test".to_string(),
            alternatives: vec![],
        };

        let shape2 = InferredShape {
            static_shape: Some(TensorShape {
                dimensions: vec![3, 4],
                dynamic_dimensions: vec![false, false],
                element_count: 12,
                tuple_shapes: vec![],
            }),
            dynamic_shape: None,
            confidence: 1.0,
            inference_method: "test".to_string(),
            alternatives: vec![],
        };

        let input_shapes = vec![Some(&shape1), Some(&shape2)];
        let context = ShapeInferenceContext {
            computation: XLAComputation {
                id: super::super::graph_capture::ComputationId(0),
                operations: vec![],
                inputs: vec![],
                outputs: vec![],
                metadata: super::super::graph_capture::ComputationMetadata::default(),
                operands: std::collections::HashMap::new(),
                dependencies: std::collections::HashMap::new(),
            },
            inferred_shapes: std::collections::HashMap::new(),
            discovered_constraints: vec![],
            dynamic_info: std::collections::HashMap::new(),
            options: ShapeInferenceOptions::default(),
        };

        let result = inference.infer_dot_shape(&input_shapes, &context).unwrap();
        let static_shape = result.static_shape.unwrap();
        assert_eq!(static_shape.dimensions, vec![2, 4]);
        assert_eq!(static_shape.element_count, 8);
    }
}
