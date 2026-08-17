//! CPU reference executor for the XLA IR.
//!
//! This module evaluates an [`XLAComputation`] directly on the CPU using
//! `ndarray`. It exists so that computations built with
//! [`ComputationGraphBuilder`](super::frontend::ComputationGraphBuilder) and
//! rewritten by the optimization pipeline can actually be *run* and checked,
//! rather than merely being constructed.
//!
//! # What this is and is not
//!
//! There is no TPU in this environment. This executor is a **reference
//! implementation**: it defines the semantics that a TPU backend would have to
//! match, and it makes the optimization passes testable end-to-end (a pass is
//! correct exactly when it preserves this evaluator's results). It makes no
//! claim to TPU-like performance and performs no device execution.
//!
//! Operations that are not implemented return
//! [`NotImplementedError`](scirs2_core::error::CoreError::NotImplementedError)
//! naming the operation, never a fabricated result.
//!
//! # Example
//!
//! ```
//! use optirs_tpu::xla::execution::ReferenceExecutor;
//! use optirs_tpu::xla::frontend::{
//!     ComputationGraphBuilder, ConstantValue, OperationType, TensorShape,
//! };
//!
//! # fn main() -> optirs_tpu::error::Result<()> {
//! let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
//! let mut computation = builder.create_computation("two_plus_three");
//!
//! let scalar = TensorShape {
//!     dimensions: Vec::new(),
//!     dynamic_dimensions: Vec::new(),
//!     element_count: 1,
//!     tuple_shapes: Vec::new(),
//! };
//!
//! let two = builder.add_operation(
//!     &mut computation,
//!     OperationType::Constant(ConstantValue::scalar(2.0)),
//!     vec![],
//!     scalar.clone(),
//! )?;
//! let two = computation
//!     .operation_output(two)
//!     .ok_or_else(|| optirs_tpu::error::OptimError::from("missing operand".to_string()))?;
//!
//! let three = builder.add_operation(
//!     &mut computation,
//!     OperationType::Constant(ConstantValue::scalar(3.0)),
//!     vec![],
//!     scalar.clone(),
//! )?;
//! let three = computation
//!     .operation_output(three)
//!     .ok_or_else(|| optirs_tpu::error::OptimError::from("missing operand".to_string()))?;
//!
//! builder.add_operation(
//!     &mut computation,
//!     OperationType::Add,
//!     vec![two, three],
//!     scalar,
//! )?;
//! builder.mark_terminal_operands_as_outputs(&mut computation)?;
//!
//! let outputs = ReferenceExecutor::new().execute(&computation, Default::default())?;
//! assert_eq!(outputs[0].iter().copied().collect::<Vec<f64>>(), vec![5.0]);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt::Debug;

use scirs2_core::error::ErrorContext;
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};
use scirs2_core::numeric::Float;

use super::frontend::{
    ConstantValue, OperandId, OperationId, OperationType, ReduceOperation, ReductionFunction,
    TensorShape, XLAComputation, XLAOperation,
};
use crate::error::{OptimError, Result};

/// Values bound to operands during evaluation, keyed by operand.
///
/// All arithmetic is performed in `f64` regardless of the graph's element type,
/// matching [`ConstantValue`]'s storage.
pub type ValueMap = HashMap<OperandId, ArrayD<f64>>;

/// CPU reference evaluator for [`XLAComputation`] graphs.
#[derive(Debug, Default, Clone)]
pub struct ReferenceExecutor {
    /// Number of operations evaluated by the most recent run.
    evaluated_operations: usize,
}

impl ReferenceExecutor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of operations evaluated by the most recent [`Self::execute`] call.
    pub fn evaluated_operations(&self) -> usize {
        self.evaluated_operations
    }

    /// Evaluate `computation` and return one array per declared output.
    ///
    /// `inputs` supplies values for `Parameter` operations, keyed by the
    /// operand each parameter defines. A missing parameter is an error rather
    /// than an implicit zero.
    pub fn execute<T>(
        &mut self,
        computation: &XLAComputation<T>,
        inputs: ValueMap,
    ) -> Result<Vec<ArrayD<f64>>>
    where
        T: Float + Debug + Default + Clone + Send + Sync + 'static,
    {
        let mut values = self.evaluate_all(computation, inputs)?;

        if computation.outputs.is_empty() {
            return Err(OptimError::ValidationError(ErrorContext::new(format!(
                "computation '{}' declares no outputs; nothing to return",
                computation.metadata.name
            ))));
        }

        let mut results = Vec::with_capacity(computation.outputs.len());
        for output in &computation.outputs {
            let value = values.remove(&output.operand).ok_or_else(|| {
                OptimError::ComputationError(ErrorContext::new(format!(
                    "output {} of computation '{}' references operand {:?}, which was never \
                     produced",
                    output.index, computation.metadata.name, output.operand
                )))
            })?;
            results.push(value);
        }

        Ok(results)
    }

    /// Evaluate every operation and return the full operand environment.
    ///
    /// Useful for inspecting intermediates in tests.
    pub fn evaluate_all<T>(
        &mut self,
        computation: &XLAComputation<T>,
        inputs: ValueMap,
    ) -> Result<ValueMap>
    where
        T: Float + Debug + Default + Clone + Send + Sync + 'static,
    {
        self.evaluated_operations = 0;
        let mut values: ValueMap = inputs;

        for operation in Self::execution_order(computation)? {
            // A `Parameter` is defined by the caller, not computed. Anything
            // else whose operand is already bound was supplied as a pre-set
            // value and is likewise left alone.
            if values.contains_key(&operation.output) {
                continue;
            }
            let result = self.evaluate_operation(computation, operation, &values)?;
            values.insert(operation.output, result);
            self.evaluated_operations += 1;
        }

        Ok(values)
    }

    /// Operations in an order where every operand is defined before it is read.
    fn execution_order<T>(computation: &XLAComputation<T>) -> Result<Vec<&XLAOperation<T>>>
    where
        T: Float + Debug + Default + Clone + Send + Sync + 'static,
    {
        let by_id: HashMap<OperationId, &XLAOperation<T>> = computation
            .operations
            .iter()
            .map(|op| (op.id, op))
            .collect();

        // Kahn's algorithm over the operand dependencies, which is independent
        // of the (possibly stale) `dependencies` map.
        let producers: HashMap<OperandId, OperationId> = computation
            .operations
            .iter()
            .map(|op| (op.output, op.id))
            .collect();

        let mut in_degree: HashMap<OperationId, usize> = HashMap::new();
        let mut dependents: HashMap<OperationId, Vec<OperationId>> = HashMap::new();

        for operation in &computation.operations {
            in_degree.entry(operation.id).or_insert(0);
            for input in &operation.inputs {
                if let Some(&producer) = producers.get(input) {
                    dependents.entry(producer).or_default().push(operation.id);
                    *in_degree.entry(operation.id).or_insert(0) += 1;
                }
            }
        }

        let mut ready: Vec<OperationId> = computation
            .operations
            .iter()
            .filter(|op| in_degree.get(&op.id) == Some(&0))
            .map(|op| op.id)
            .collect();

        let mut order: Vec<&XLAOperation<T>> = Vec::with_capacity(computation.operations.len());
        while let Some(op_id) = ready.pop() {
            if let Some(&operation) = by_id.get(&op_id) {
                order.push(operation);
            }
            for next in dependents.get(&op_id).cloned().unwrap_or_default() {
                if let Some(degree) = in_degree.get_mut(&next) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.push(next);
                    }
                }
            }
        }

        if order.len() != computation.operations.len() {
            return Err(OptimError::ValidationError(ErrorContext::new(format!(
                "computation '{}' contains a dependency cycle and cannot be executed",
                computation.metadata.name
            ))));
        }

        Ok(order)
    }

    /// Evaluate a single operation against the current environment.
    fn evaluate_operation<T>(
        &self,
        computation: &XLAComputation<T>,
        operation: &XLAOperation<T>,
        values: &ValueMap,
    ) -> Result<ArrayD<f64>>
    where
        T: Float + Debug + Default + Clone + Send + Sync + 'static,
    {
        let output_shape = computation
            .operands
            .get(&operation.output)
            .map(|operand| operand.shape.clone())
            .unwrap_or_default();

        // Resolve operand values up front so each arm can pattern-match.
        let mut operands: Vec<&ArrayD<f64>> = Vec::with_capacity(operation.inputs.len());
        for input in &operation.inputs {
            let value = values.get(input).ok_or_else(|| {
                OptimError::ComputationError(ErrorContext::new(format!(
                    "operation {:?} ({:?}) reads operand {:?} before it is defined",
                    operation.id, operation.op_type, input
                )))
            })?;
            operands.push(value);
        }

        match (&operation.op_type, operands.as_slice()) {
            (OperationType::Constant(value), _) => Ok(constant_to_array(value)),

            (OperationType::Parameter, _) => {
                Err(OptimError::InvalidInput(ErrorContext::new(format!(
                    "no input value was supplied for parameter operand {:?} of computation '{}'",
                    operation.output, computation.metadata.name
                ))))
            }

            // Binary elementwise operations, with rank-0 scalar broadcasting.
            (OperationType::Add, [a, b]) => binary(a, b, |x, y| x + y),
            (OperationType::Subtract, [a, b]) => binary(a, b, |x, y| x - y),
            (OperationType::Multiply, [a, b]) => binary(a, b, |x, y| x * y),
            (OperationType::Divide, [a, b]) => binary(a, b, |x, y| x / y),
            (OperationType::Maximum, [a, b]) => binary(a, b, f64::max),
            (OperationType::Minimum, [a, b]) => binary(a, b, f64::min),
            (OperationType::And, [a, b]) => binary(a, b, |x, y| bool_to_f64(x != 0.0 && y != 0.0)),
            (OperationType::Or, [a, b]) => binary(a, b, |x, y| bool_to_f64(x != 0.0 || y != 0.0)),
            (OperationType::Xor, [a, b]) => {
                binary(a, b, |x, y| bool_to_f64((x != 0.0) != (y != 0.0)))
            }
            (OperationType::Equal, [a, b]) => binary(a, b, |x, y| bool_to_f64(x == y)),
            (OperationType::NotEqual, [a, b]) => binary(a, b, |x, y| bool_to_f64(x != y)),
            (OperationType::Less, [a, b]) => binary(a, b, |x, y| bool_to_f64(x < y)),
            (OperationType::LessEqual, [a, b]) => binary(a, b, |x, y| bool_to_f64(x <= y)),
            (OperationType::Greater, [a, b]) => binary(a, b, |x, y| bool_to_f64(x > y)),
            (OperationType::GreaterEqual, [a, b]) => binary(a, b, |x, y| bool_to_f64(x >= y)),

            // Unary elementwise operations.
            (OperationType::Negate, [a]) => Ok(a.mapv(|x| -x)),
            (OperationType::Abs, [a]) => Ok(a.mapv(f64::abs)),
            (OperationType::Square, [a]) => Ok(a.mapv(|x| x * x)),
            (OperationType::Sqrt, [a]) => Ok(a.mapv(f64::sqrt)),
            (OperationType::Rsqrt, [a]) => Ok(a.mapv(|x| 1.0 / x.sqrt())),
            (OperationType::Exp, [a]) => Ok(a.mapv(f64::exp)),
            (OperationType::Log, [a]) => Ok(a.mapv(f64::ln)),
            (OperationType::Sin, [a]) => Ok(a.mapv(f64::sin)),
            (OperationType::Cos, [a]) => Ok(a.mapv(f64::cos)),
            (OperationType::Tanh, [a]) => Ok(a.mapv(f64::tanh)),
            (OperationType::Ceil, [a]) => Ok(a.mapv(f64::ceil)),
            (OperationType::Floor, [a]) => Ok(a.mapv(f64::floor)),
            (OperationType::Round, [a]) => Ok(a.mapv(f64::round)),
            (OperationType::Sign, [a]) => Ok(a.mapv(|x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })),
            (OperationType::Not, [a]) => Ok(a.mapv(|x| bool_to_f64(x == 0.0))),
            (OperationType::Copy, [a]) => Ok((*a).clone()),

            // Matrix products.
            (OperationType::Dot, [a, b])
            | (OperationType::DotGeneral, [a, b])
            | (OperationType::MatMul, [a, b]) => matmul(a, b),

            // Shape manipulation.
            (OperationType::Reshape, [a]) => reshape(a, &output_shape),
            (OperationType::Transpose, [a]) => Ok(a.t().to_owned()),
            (OperationType::Broadcast, [a]) => broadcast_to(a, &output_shape),

            (OperationType::Reduce(reduce_op), [a]) => reduce(a, reduce_op),

            (OperationType::Concatenate, values) if !values.is_empty() => {
                concatenate(values, &operation.attributes)
            }

            // A tuple has no single array representation in this value model.
            (OperationType::Tuple, _) | (OperationType::GetTupleElement, _) => {
                Err(OptimError::NotImplementedError(ErrorContext::new(format!(
                    "the CPU reference executor does not model tuple values, so {:?} \
                     (operation {:?}) cannot be evaluated",
                    operation.op_type, operation.id
                ))))
            }

            // Everything else: say what is missing rather than invent a value.
            (op_type, operands) => {
                Err(OptimError::NotImplementedError(ErrorContext::new(format!(
                    "the CPU reference executor does not implement {:?} with {} operand(s) \
                     (operation {:?} in computation '{}')",
                    op_type,
                    operands.len(),
                    operation.id,
                    computation.metadata.name
                ))))
            }
        }
    }
}

/// Convert a literal into an array.
fn constant_to_array(value: &ConstantValue) -> ArrayD<f64> {
    let shape = IxDyn(&value.dims);
    ArrayD::from_shape_vec(shape, value.data.clone()).unwrap_or_else(|_| {
        // `ConstantValue::new` enforces the invariant, but a hand-built literal
        // could violate it; fall back to a flat view rather than panicking.
        ArrayD::from_shape_vec(IxDyn(&[value.data.len()]), value.data.clone())
            .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&[0])))
    })
}

fn bool_to_f64(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

/// Element-wise binary application with rank-0 scalar broadcasting.
fn binary(
    lhs: &ArrayD<f64>,
    rhs: &ArrayD<f64>,
    op: impl Fn(f64, f64) -> f64,
) -> Result<ArrayD<f64>> {
    if lhs.shape() == rhs.shape() {
        let mut out = lhs.clone();
        for (slot, &value) in out.iter_mut().zip(rhs.iter()) {
            *slot = op(*slot, value);
        }
        return Ok(out);
    }

    if lhs.len() == 1 {
        let scalar = lhs.iter().next().copied().unwrap_or(0.0);
        return Ok(rhs.mapv(|value| op(scalar, value)));
    }

    if rhs.len() == 1 {
        let scalar = rhs.iter().next().copied().unwrap_or(0.0);
        return Ok(lhs.mapv(|value| op(value, scalar)));
    }

    Err(OptimError::ShapeError(ErrorContext::new(format!(
        "cannot apply an element-wise operation to shapes {:?} and {:?}; only rank-0 \
         broadcasting is modelled",
        lhs.shape(),
        rhs.shape()
    ))))
}

/// Two-dimensional matrix product `[M, K] x [K, N] -> [M, N]`.
fn matmul(lhs: &ArrayD<f64>, rhs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
    let lhs_2d: Array2<f64> = lhs.clone().into_dimensionality().map_err(|_| {
        OptimError::ShapeError(ErrorContext::new(format!(
            "matrix product requires a rank-2 left operand, got shape {:?}",
            lhs.shape()
        )))
    })?;
    let rhs_2d: Array2<f64> = rhs.clone().into_dimensionality().map_err(|_| {
        OptimError::ShapeError(ErrorContext::new(format!(
            "matrix product requires a rank-2 right operand, got shape {:?}",
            rhs.shape()
        )))
    })?;

    if lhs_2d.ncols() != rhs_2d.nrows() {
        return Err(OptimError::ShapeError(ErrorContext::new(format!(
            "matrix product contraction mismatch: {:?} x {:?}",
            lhs_2d.shape(),
            rhs_2d.shape()
        ))));
    }

    Ok(lhs_2d.dot(&rhs_2d).into_dyn())
}

/// Reinterpret an array under a new shape with the same element count.
fn reshape(value: &ArrayD<f64>, target: &TensorShape) -> Result<ArrayD<f64>> {
    let expected: usize = if target.dimensions.is_empty() {
        1
    } else {
        target.dimensions.iter().product()
    };

    if expected != value.len() {
        return Err(OptimError::ShapeError(ErrorContext::new(format!(
            "reshape changes the element count: {} elements cannot become shape {:?}",
            value.len(),
            target.dimensions
        ))));
    }

    let flat: Vec<f64> = value.iter().copied().collect();
    ArrayD::from_shape_vec(IxDyn(&target.dimensions), flat).map_err(|error| {
        OptimError::ShapeError(ErrorContext::new(format!("reshape failed: {error}")))
    })
}

/// Broadcast a scalar (or already-matching array) to the target shape.
fn broadcast_to(value: &ArrayD<f64>, target: &TensorShape) -> Result<ArrayD<f64>> {
    let count: usize = if target.dimensions.is_empty() {
        1
    } else {
        target.dimensions.iter().product()
    };

    if value.len() == count {
        return reshape(value, target);
    }

    if value.len() == 1 {
        let scalar = value.iter().next().copied().unwrap_or(0.0);
        return Ok(ArrayD::from_elem(IxDyn(&target.dimensions), scalar));
    }

    Err(OptimError::NotImplementedError(ErrorContext::new(format!(
        "the CPU reference executor only broadcasts rank-0 scalars; cannot broadcast shape \
         {:?} to {:?}",
        value.shape(),
        target.dimensions
    ))))
}

/// Reduce over the requested dimensions.
fn reduce(value: &ArrayD<f64>, reduce_op: &ReduceOperation) -> Result<ArrayD<f64>> {
    let (init, combine): (f64, fn(f64, f64) -> f64) = match reduce_op.function {
        ReductionFunction::Add => (0.0, |a, b| a + b),
        ReductionFunction::Multiply => (1.0, |a, b| a * b),
        ReductionFunction::Max => (f64::NEG_INFINITY, f64::max),
        ReductionFunction::Min => (f64::INFINITY, f64::min),
        ReductionFunction::And => (1.0, |a, b| bool_to_f64(a != 0.0 && b != 0.0)),
        ReductionFunction::Or => (0.0, |a, b| bool_to_f64(a != 0.0 || b != 0.0)),
        ReductionFunction::Xor => (0.0, |a, b| bool_to_f64((a != 0.0) != (b != 0.0))),
    };

    // Reducing every dimension (or an unspecified set) collapses to a scalar.
    let reduces_all = reduce_op.dimensions.is_empty() || reduce_op.dimensions.len() == value.ndim();

    if reduces_all {
        let total = value.iter().copied().fold(init, combine);
        return ArrayD::from_shape_vec(IxDyn(&[]), vec![total]).map_err(|error| {
            OptimError::ShapeError(ErrorContext::new(format!("reduce failed: {error}")))
        });
    }

    let mut current = value.clone();
    // Remove axes from the highest index down so earlier indices stay valid.
    let mut axes = reduce_op.dimensions.clone();
    axes.sort_unstable();
    axes.dedup();

    for &axis in axes.iter().rev() {
        if axis >= current.ndim() {
            return Err(OptimError::ShapeError(ErrorContext::new(format!(
                "reduce dimension {axis} is out of range for a rank-{} operand",
                current.ndim()
            ))));
        }
        current = current.fold_axis(scirs2_core::ndarray::Axis(axis), init, |&a, &b| {
            combine(a, b)
        });
    }

    Ok(current)
}

/// Concatenate operands along the axis named by the `dimension` attribute.
fn concatenate(
    values: &[&ArrayD<f64>],
    attributes: &super::frontend::OperationAttributes,
) -> Result<ArrayD<f64>> {
    use super::frontend::AttributeValue;

    let axis = match attributes.attributes.get("dimension") {
        Some(AttributeValue::Int(value)) if *value >= 0 => *value as usize,
        Some(other) => {
            return Err(OptimError::InvalidInput(ErrorContext::new(format!(
                "concatenate `dimension` attribute must be a non-negative Int, got {other:?}"
            ))))
        }
        None => 0,
    };

    let views: Vec<_> = values.iter().map(|value| value.view()).collect();
    scirs2_core::ndarray::concatenate(scirs2_core::ndarray::Axis(axis), &views).map_err(|error| {
        OptimError::ShapeError(ErrorContext::new(format!("concatenate failed: {error}")))
    })
}

#[cfg(test)]
mod tests {
    use super::super::frontend::graph_capture::test_support::{add_op, scalar_shape, shape};
    use super::super::frontend::ComputationGraphBuilder;
    use super::*;

    /// The headline capability: build a graph, execute it, get a number.
    #[test]
    fn executes_a_three_operation_graph() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("add");

        let two = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(2.0)),
            vec![],
            scalar_shape(),
        );
        let three = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(3.0)),
            vec![],
            scalar_shape(),
        );
        let _sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![two, three],
            scalar_shape(),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        let outputs = ReferenceExecutor::new()
            .execute(&comp, ValueMap::new())
            .expect("execution must succeed");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].iter().copied().collect::<Vec<_>>(), vec![5.0]);
    }

    /// Parameters are bound from the caller's input map.
    #[test]
    fn executes_with_parameter_inputs() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("scale");

        let x = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            shape(&[3]),
        );
        let factor = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(10.0)),
            vec![],
            scalar_shape(),
        );
        let _scaled = add_op(
            &mut builder,
            &mut comp,
            OperationType::Multiply,
            vec![x, factor],
            shape(&[3]),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        let mut inputs = ValueMap::new();
        inputs.insert(
            x,
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0])
                .expect("input array must build"),
        );

        let outputs = ReferenceExecutor::new()
            .execute(&comp, inputs)
            .expect("execution must succeed");

        assert_eq!(
            outputs[0].iter().copied().collect::<Vec<_>>(),
            vec![10.0, 20.0, 30.0]
        );
    }

    /// A missing parameter binding is an error, not an implicit zero.
    #[test]
    fn missing_parameter_input_is_an_error() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("unbound");

        let x = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            shape(&[2]),
        );
        let _out = add_op(
            &mut builder,
            &mut comp,
            OperationType::Abs,
            vec![x],
            shape(&[2]),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        assert!(ReferenceExecutor::new()
            .execute(&comp, ValueMap::new())
            .is_err());
    }

    /// Matrix multiplication produces the mathematically correct product.
    #[test]
    fn executes_matrix_multiplication() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("matmul");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(
                ConstantValue::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
                    .expect("constant must be well formed"),
            ),
            vec![],
            shape(&[2, 2]),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(
                ConstantValue::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])
                    .expect("constant must be well formed"),
            ),
            vec![],
            shape(&[2, 2]),
        );
        let _product = add_op(
            &mut builder,
            &mut comp,
            OperationType::MatMul,
            vec![a, b],
            shape(&[2, 2]),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        let outputs = ReferenceExecutor::new()
            .execute(&comp, ValueMap::new())
            .expect("execution must succeed");

        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(
            outputs[0].iter().copied().collect::<Vec<_>>(),
            vec![19.0, 22.0, 43.0, 50.0]
        );
    }

    /// Reshape preserves the element count and reorders nothing.
    #[test]
    fn executes_reshape() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("reshape");

        let source = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(
                ConstantValue::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
                    .expect("constant must be well formed"),
            ),
            vec![],
            shape(&[2, 3]),
        );
        let _reshaped = add_op(
            &mut builder,
            &mut comp,
            OperationType::Reshape,
            vec![source],
            shape(&[3, 2]),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        let outputs = ReferenceExecutor::new()
            .execute(&comp, ValueMap::new())
            .expect("execution must succeed");

        assert_eq!(outputs[0].shape(), &[3, 2]);
        assert_eq!(
            outputs[0].iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    /// Reductions collapse the requested axes.
    #[test]
    fn executes_sum_reduction() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("reduce");

        let source = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(
                ConstantValue::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
                    .expect("constant must be well formed"),
            ),
            vec![],
            shape(&[2, 2]),
        );
        let _total = add_op(
            &mut builder,
            &mut comp,
            OperationType::Reduce(ReduceOperation {
                function: ReductionFunction::Add,
                dimensions: vec![0, 1],
                init_value: None,
            }),
            vec![source],
            scalar_shape(),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        let outputs = ReferenceExecutor::new()
            .execute(&comp, ValueMap::new())
            .expect("execution must succeed");

        assert_eq!(outputs[0].iter().copied().collect::<Vec<_>>(), vec![10.0]);
    }

    /// An unimplemented operation reports itself rather than returning a value.
    #[test]
    fn unimplemented_operation_reports_itself() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("unsupported");

        let source = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(1.0)),
            vec![],
            scalar_shape(),
        );
        let _gathered = add_op(
            &mut builder,
            &mut comp,
            OperationType::Gather,
            vec![source],
            scalar_shape(),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        let error = ReferenceExecutor::new()
            .execute(&comp, ValueMap::new())
            .expect_err("Gather is not implemented");
        let message = format!("{error}");
        assert!(
            message.contains("Gather"),
            "the error must name the operation: {message}"
        );
    }

    /// The optimization pipeline must preserve the executor's results.
    ///
    /// This is the property that makes the passes trustworthy: whatever
    /// constant folding, algebraic simplification, CSE and DCE do, the numbers
    /// coming out must not change.
    #[test]
    fn optimization_preserves_execution_results() {
        use super::super::optimization::OptimizationPipeline;
        use super::super::XLACompilerConfig;

        let build = || {
            let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
            let mut comp = builder.create_computation("preserved");

            let x = add_op(
                &mut builder,
                &mut comp,
                OperationType::Constant(ConstantValue::scalar(4.0)),
                vec![],
                scalar_shape(),
            );
            let one = add_op(
                &mut builder,
                &mut comp,
                OperationType::Constant(ConstantValue::scalar(1.0)),
                vec![],
                scalar_shape(),
            );
            let zero = add_op(
                &mut builder,
                &mut comp,
                OperationType::Constant(ConstantValue::scalar(0.0)),
                vec![],
                scalar_shape(),
            );
            // (x * 1) + 0, plus a duplicated subexpression for CSE to find.
            let scaled = add_op(
                &mut builder,
                &mut comp,
                OperationType::Multiply,
                vec![x, one],
                scalar_shape(),
            );
            let shifted = add_op(
                &mut builder,
                &mut comp,
                OperationType::Add,
                vec![scaled, zero],
                scalar_shape(),
            );
            let squared_a = add_op(
                &mut builder,
                &mut comp,
                OperationType::Square,
                vec![shifted],
                scalar_shape(),
            );
            let squared_b = add_op(
                &mut builder,
                &mut comp,
                OperationType::Square,
                vec![shifted],
                scalar_shape(),
            );
            let _total = add_op(
                &mut builder,
                &mut comp,
                OperationType::Add,
                vec![squared_a, squared_b],
                scalar_shape(),
            );
            builder
                .mark_terminal_operands_as_outputs(&mut comp)
                .expect("outputs must be inferable");
            comp
        };

        let before = ReferenceExecutor::new()
            .execute(&build(), ValueMap::new())
            .expect("the unoptimized graph must execute");

        let config = XLACompilerConfig::default();
        let mut pipeline: OptimizationPipeline<f32> = OptimizationPipeline::new(&config);
        let optimized = pipeline
            .optimize(build())
            .expect("optimization must succeed");

        let after = ReferenceExecutor::new()
            .execute(&optimized, ValueMap::new())
            .expect("the optimized graph must execute");

        // (4*1 + 0)^2 * 2 = 32
        assert_eq!(before[0].iter().copied().collect::<Vec<_>>(), vec![32.0]);
        assert_eq!(
            before[0].iter().copied().collect::<Vec<_>>(),
            after[0].iter().copied().collect::<Vec<_>>(),
            "optimization must not change the computed result"
        );
    }
}
