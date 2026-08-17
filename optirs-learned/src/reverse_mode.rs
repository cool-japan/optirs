//! Reverse-mode automatic differentiation (backpropagation).
//!
//! This module implements a tape-based reverse-mode automatic differentiation
//! engine used for gradient computation in the learned-optimizer and
//! meta-learning code paths of this crate.
//!
//! # Design
//!
//! Every operation appends exactly one node to the tape and *immediately*
//! evaluates its forward value, which is stored in [`ReverseModeEngine`]'s
//! parallel `values` vector. Because a node's index always equals `tape.len()`
//! at the moment it is created, every node's index is strictly greater than the
//! indices of all of its inputs. That invariant is what makes a simple reverse
//! iteration over the tape a valid topological order for backpropagation, and
//! it is why an operation is recorded even when recording is disabled (in that
//! case the node is recorded as a *constant*, which stops gradient flow without
//! breaking the index invariant).
//!
//! Values are flat [`Array1`] buffers plus an explicit logical shape, so 2-D
//! matrices are stored row-major and matrix multiplication is expressed through
//! the recorded shapes.

use std::fmt::Debug;

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// Trait bundle required by the reverse-mode engine.
///
/// This exists purely to keep the (long) bound lists readable.
pub trait ReverseScalar:
    Float
    + Debug
    + Default
    + Clone
    + Send
    + Sync
    + 'static
    + std::iter::Sum
    + scirs2_core::ndarray::ScalarOperand
{
}

impl<T> ReverseScalar for T where
    T: Float
        + Debug
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
{
}

/// Reverse-mode AD engine (gradient tape)
pub struct ReverseModeEngine<T: ReverseScalar> {
    /// Computation tape for the reverse pass
    tape: Vec<ReverseOperation<T>>,

    /// Forward value of every tape node (parallel to `tape`)
    values: Vec<Array1<T>>,

    /// Logical shape of every tape node (parallel to `tape`)
    shapes: Vec<Vec<usize>>,

    /// Whether each node is a differentiable leaf
    requiresgrad: Vec<bool>,

    /// Variable registry (name -> tape index)
    variables: HashMap<String, usize>,

    /// Gradient storage
    gradients: Vec<Option<Array1<T>>>,

    /// Current recording state
    recording: bool,

    /// Higher-order gradient tracking
    higher_order: bool,

    /// Set when a leaf value changed and the tape needs re-evaluation
    dirty: bool,

    /// Gradient computation cache
    cache: HashMap<usize, Array1<T>>,
}

/// Reverse-mode operation on the tape
#[derive(Debug, Clone)]
struct ReverseOperation<T: Float + Debug + Send + Sync + 'static> {
    /// Operation type
    op_type: ReverseOpType,

    /// Input variable indices
    inputs: Vec<usize>,

    /// Backward function for gradient computation
    backward_fn: BackwardFunction<T>,

    /// Detached nodes are evaluated in the forward pass but block gradient flow
    detached: bool,
}

/// Reverse operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReverseOpType {
    /// Differentiable leaf
    Variable,
    /// Non-differentiable leaf (also used for detached nodes)
    Constant,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU,
    MatMul,
    Dot,
    Sum,
    Mean,
    Norm,
    Reshape,
}

/// Backward function for computing gradients
#[derive(Debug, Clone)]
enum BackwardFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Leaf node: no inputs, nothing to propagate
    Leaf,

    /// Addition backward: gradient flows through unchanged
    AddBackward,

    /// Subtraction backward: negate gradient for the second operand
    SubtractBackward,

    /// Element-wise multiplication backward
    MultiplyBackward,

    /// Element-wise division backward
    DivideBackward,

    /// Dot-product backward
    DotBackward,

    /// Power backward: multiply by the derivative
    PowerBackward { exponent: T },

    /// Exponential backward
    ExpBackward,

    /// Logarithm backward
    LogBackward,

    /// Trigonometric function backward
    TrigBackward { function: TrigFunction },

    /// Activation function backward
    ActivationBackward { function: ActivationFunction },

    /// Matrix multiplication backward
    MatMulBackward,

    /// Reduction backward
    ReductionBackward { reduction_type: ReductionType },

    /// Reshape backward (the target shape is needed to re-evaluate the node)
    ReshapeBackward { newshape: Vec<usize> },
}

/// Trigonometric functions
#[derive(Debug, Clone, Copy)]
enum TrigFunction {
    Sin,
    Cos,
}

/// Activation functions
#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU { alpha: f64 },
}

/// Reduction types
#[derive(Debug, Clone, Copy)]
enum ReductionType {
    Sum,
    Mean,
    Norm,
}

/// Gradient computation context
#[derive(Debug, Clone)]
pub struct GradientContext<T: Float + Debug + Send + Sync + 'static> {
    /// Variables requiring gradients
    pub requiresgrad: HashMap<usize, bool>,

    /// Gradient accumulation mode
    pub accumulate: bool,

    /// Retain computation graph
    pub retain_graph: bool,

    /// Create computation graph
    pub create_graph: bool,

    /// Type parameter marker
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> Default for GradientContext<T> {
    fn default() -> Self {
        Self {
            requiresgrad: HashMap::new(),
            accumulate: true,
            retain_graph: false,
            create_graph: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: ReverseScalar> Default for ReverseModeEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ReverseScalar> ReverseModeEngine<T> {
    /// Create a new reverse-mode AD engine
    pub fn new() -> Self {
        Self {
            tape: Vec::new(),
            values: Vec::new(),
            shapes: Vec::new(),
            requiresgrad: Vec::new(),
            variables: HashMap::new(),
            gradients: Vec::new(),
            recording: true,
            higher_order: false,
            dirty: false,
            cache: HashMap::new(),
        }
    }

    /// Enable/disable gradient recording.
    ///
    /// While recording is disabled, subsequent operations are still evaluated
    /// and appended to the tape, but they are recorded as *constants*: they
    /// keep the tape's index invariant intact while blocking gradient flow.
    pub fn set_recording(&mut self, recording: bool) {
        self.recording = recording;
    }

    /// Check if recording is enabled
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Enable higher-order gradients
    pub fn enable_higher_order(&mut self, enabled: bool) {
        self.higher_order = enabled;
    }

    /// Whether higher-order gradient tracking is enabled
    pub fn higher_order_enabled(&self) -> bool {
        self.higher_order
    }

    /// Clear the computation tape
    pub fn clear_tape(&mut self) {
        self.tape.clear();
        self.values.clear();
        self.shapes.clear();
        self.requiresgrad.clear();
        self.variables.clear();
        self.gradients.clear();
        self.cache.clear();
        self.dirty = false;
    }

    /// Number of nodes currently on the tape
    pub fn tape_len(&self) -> usize {
        self.tape.len()
    }

    // ---------------------------------------------------------------- leaves

    /// Create a variable with gradient tracking (1-D, shape `[value.len()]`)
    pub fn create_variable(&mut self, name: &str, value: Array1<T>, requiresgrad: bool) -> usize {
        let shape = vec![value.len()];
        self.push_leaf(Some(name), value, shape, requiresgrad)
    }

    /// Create a variable with an explicit logical shape.
    ///
    /// `value` is the row-major flattening of a tensor whose logical shape is
    /// `shape`; the product of `shape` must equal `value.len()`.
    pub fn create_variable_with_shape(
        &mut self,
        name: &str,
        value: Array1<T>,
        shape: &[usize],
        requiresgrad: bool,
    ) -> Result<usize> {
        let expected: usize = shape.iter().product();
        if shape.is_empty() || expected != value.len() {
            return Err(OptimError::InvalidConfig(format!(
                "shape {:?} does not match value length {}",
                shape,
                value.len()
            )));
        }
        Ok(self.push_leaf(Some(name), value, shape.to_vec(), requiresgrad))
    }

    /// Create a constant (no gradient tracking)
    pub fn create_constant(&mut self, value: Array1<T>) -> usize {
        let shape = vec![value.len()];
        self.push_leaf(None, value, shape, false)
    }

    fn push_leaf(
        &mut self,
        name: Option<&str>,
        value: Array1<T>,
        shape: Vec<usize>,
        requiresgrad: bool,
    ) -> usize {
        let varid = self.tape.len();
        let op_type = if requiresgrad {
            ReverseOpType::Variable
        } else {
            ReverseOpType::Constant
        };

        self.tape.push(ReverseOperation {
            op_type,
            inputs: Vec::new(),
            backward_fn: BackwardFunction::Leaf,
            detached: !requiresgrad,
        });
        let len = value.len();
        self.values.push(value);
        self.shapes.push(shape);
        self.requiresgrad.push(requiresgrad);
        self.gradients.push(if requiresgrad {
            Some(Array1::zeros(len))
        } else {
            None
        });

        if let Some(name) = name {
            self.variables.insert(name.to_string(), varid);
        }

        varid
    }

    /// Replace the value of an existing leaf variable.
    ///
    /// The tape is marked dirty; [`ReverseModeEngine::forward`] (called
    /// automatically by [`ReverseModeEngine::backward`]) re-evaluates every
    /// dependent node.
    pub fn set_variable_value(&mut self, varid: usize, value: Array1<T>) -> Result<()> {
        let op = self
            .tape
            .get(varid)
            .ok_or_else(|| OptimError::InvalidConfig(format!("invalid variable id {varid}")))?;
        if !op.inputs.is_empty() {
            return Err(OptimError::InvalidConfig(format!(
                "node {varid} is not a leaf and cannot be assigned"
            )));
        }
        let shape = self.shapes.get(varid).ok_or_else(|| {
            OptimError::ComputationError(format!("missing shape for node {varid}"))
        })?;
        let expected: usize = shape.iter().product();
        if expected != value.len() {
            return Err(OptimError::InvalidConfig(format!(
                "assigned value length {} does not match node shape {:?}",
                value.len(),
                shape
            )));
        }
        self.values[varid] = value;
        self.dirty = true;
        Ok(())
    }

    /// Replace the value of an existing leaf variable, by name
    pub fn set_variable_value_by_name(&mut self, name: &str, value: Array1<T>) -> Result<()> {
        let varid = *self
            .variables
            .get(name)
            .ok_or_else(|| OptimError::InvalidConfig(format!("unknown variable '{name}'")))?;
        self.set_variable_value(varid, value)
    }

    // ------------------------------------------------------------ operations

    /// Addition operation
    pub fn add(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ReverseOpType::Add, lhs, rhs, BackwardFunction::AddBackward)
    }

    /// Subtraction operation
    pub fn subtract(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(
            ReverseOpType::Subtract,
            lhs,
            rhs,
            BackwardFunction::SubtractBackward,
        )
    }

    /// Element-wise multiplication operation
    pub fn multiply(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(
            ReverseOpType::Multiply,
            lhs,
            rhs,
            BackwardFunction::MultiplyBackward,
        )
    }

    /// Element-wise division operation
    pub fn divide(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(
            ReverseOpType::Divide,
            lhs,
            rhs,
            BackwardFunction::DivideBackward,
        )
    }

    /// Power operation
    pub fn power(&mut self, base: usize, exponent: T) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Power,
            base,
            BackwardFunction::PowerBackward { exponent },
        )
    }

    /// Exponential function
    pub fn exp(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ReverseOpType::Exp, input, BackwardFunction::ExpBackward)
    }

    /// Natural logarithm
    pub fn log(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ReverseOpType::Log, input, BackwardFunction::LogBackward)
    }

    /// Sine function
    pub fn sin(&mut self, input: usize) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Sin,
            input,
            BackwardFunction::TrigBackward {
                function: TrigFunction::Sin,
            },
        )
    }

    /// Cosine function
    pub fn cos(&mut self, input: usize) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Cos,
            input,
            BackwardFunction::TrigBackward {
                function: TrigFunction::Cos,
            },
        )
    }

    /// Hyperbolic tangent
    pub fn tanh(&mut self, input: usize) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Tanh,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::Tanh,
            },
        )
    }

    /// Sigmoid function
    pub fn sigmoid(&mut self, input: usize) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Sigmoid,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::Sigmoid,
            },
        )
    }

    /// ReLU function
    pub fn relu(&mut self, input: usize) -> Result<usize> {
        self.unary_op(
            ReverseOpType::ReLU,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::ReLU,
            },
        )
    }

    /// Leaky ReLU function
    pub fn leaky_relu(&mut self, input: usize, alpha: f64) -> Result<usize> {
        self.unary_op(
            ReverseOpType::LeakyReLU,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::LeakyReLU { alpha },
            },
        )
    }

    /// Matrix multiplication.
    ///
    /// `lhs` must have logical shape `[m, k]`; `rhs` must have shape `[k, n]`
    /// or `[k]` (interpreted as a `[k, 1]` column). The result has shape
    /// `[m, n]` (or `[m]` when `rhs` was 1-D).
    pub fn matmul(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(
            ReverseOpType::MatMul,
            lhs,
            rhs,
            BackwardFunction::MatMulBackward,
        )
    }

    /// Dot product of two vectors of equal length; produces a scalar node.
    pub fn dot(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ReverseOpType::Dot, lhs, rhs, BackwardFunction::DotBackward)
    }

    /// Sum reduction over all elements
    pub fn sum(&mut self, input: usize, _axis: Option<usize>) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Sum,
            input,
            BackwardFunction::ReductionBackward {
                reduction_type: ReductionType::Sum,
            },
        )
    }

    /// Mean reduction over all elements
    pub fn mean(&mut self, input: usize, _axis: Option<usize>) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Mean,
            input,
            BackwardFunction::ReductionBackward {
                reduction_type: ReductionType::Mean,
            },
        )
    }

    /// L2 norm over all elements
    pub fn norm(&mut self, input: usize) -> Result<usize> {
        self.unary_op(
            ReverseOpType::Norm,
            input,
            BackwardFunction::ReductionBackward {
                reduction_type: ReductionType::Norm,
            },
        )
    }

    /// Reshape operation (element count must be preserved)
    pub fn reshape(&mut self, input: usize, newshape: &[usize]) -> Result<usize> {
        let inputlen = self.value(input)?.len();
        let expected: usize = newshape.iter().product();
        if newshape.is_empty() || expected != inputlen {
            return Err(OptimError::InvalidConfig(format!(
                "reshape to {newshape:?} is incompatible with {inputlen} elements"
            )));
        }
        self.push_op(
            ReverseOpType::Reshape,
            vec![input],
            BackwardFunction::ReshapeBackward {
                newshape: newshape.to_vec(),
            },
        )
    }

    // ----------------------------------------------------------- forward pass

    /// Re-evaluate every non-leaf node of the tape from the current leaf values.
    ///
    /// Nodes are visited in tape order, which is a valid topological order
    /// because every node's index exceeds all of its inputs' indices.
    pub fn forward(&mut self) -> Result<()> {
        for idx in 0..self.tape.len() {
            if self.tape[idx].inputs.is_empty() {
                continue;
            }
            let (value, shape) = self.evaluate_node(idx)?;
            self.values[idx] = value;
            self.shapes[idx] = shape;
        }
        self.dirty = false;
        Ok(())
    }

    /// Forward value of a node
    pub fn value(&self, varid: usize) -> Result<&Array1<T>> {
        self.values
            .get(varid)
            .ok_or_else(|| OptimError::InvalidConfig(format!("invalid variable id {varid}")))
    }

    /// Forward value of a node, by variable name
    pub fn value_by_name(&self, name: &str) -> Result<&Array1<T>> {
        let varid = *self
            .variables
            .get(name)
            .ok_or_else(|| OptimError::InvalidConfig(format!("unknown variable '{name}'")))?;
        self.value(varid)
    }

    /// Logical shape of a node
    pub fn shape(&self, varid: usize) -> Result<&[usize]> {
        self.shapes
            .get(varid)
            .map(|s| s.as_slice())
            .ok_or_else(|| OptimError::InvalidConfig(format!("invalid variable id {varid}")))
    }

    /// Scalar value of a node that holds exactly one element
    pub fn scalar_value(&self, varid: usize) -> Result<T> {
        let value = self.value(varid)?;
        if value.len() != 1 {
            return Err(OptimError::ComputationError(format!(
                "node {varid} holds {} elements, expected a scalar",
                value.len()
            )));
        }
        Ok(value[0])
    }

    // ---------------------------------------------------------- backward pass

    /// Backward pass: compute gradients of `outputid` w.r.t. every leaf.
    ///
    /// If `gradient` is `None`, the output is seeded with ones matching the
    /// output's length.
    pub fn backward(&mut self, outputid: usize, gradient: Option<Array1<T>>) -> Result<()> {
        if outputid >= self.tape.len() {
            return Err(OptimError::InvalidConfig(format!(
                "invalid output id {outputid} (tape has {} nodes)",
                self.tape.len()
            )));
        }

        if self.dirty {
            self.forward()?;
        }

        // Gradient storage must cover the whole tape: a node created *after*
        // `outputid` is still visited by the reverse iteration below.
        self.gradients.resize(self.tape.len(), None);
        for (idx, slot) in self.gradients.iter_mut().enumerate() {
            let is_leaf_with_grad = self.requiresgrad.get(idx).copied().unwrap_or(false);
            if is_leaf_with_grad {
                let len = self.values[idx].len();
                *slot = Some(Array1::zeros(len));
            } else {
                *slot = None;
            }
        }

        let output_len = self.values[outputid].len();
        let output_grad = match gradient {
            Some(g) => {
                if g.len() != output_len {
                    return Err(OptimError::InvalidConfig(format!(
                        "seed gradient length {} does not match output length {output_len}",
                        g.len()
                    )));
                }
                g
            }
            None => Array1::ones(output_len),
        };
        self.gradients[outputid] = Some(output_grad);

        // Reverse pass through the tape.
        for idx in (0..self.tape.len()).rev() {
            let output_gradient = match self.gradients.get(idx).and_then(|g| g.clone()) {
                Some(g) => g,
                None => continue,
            };
            if self.tape[idx].inputs.is_empty() || self.tape[idx].detached {
                continue;
            }

            let input_gradients = self.compute_backward_pass(idx, &output_gradient)?;
            let inputs = self.tape[idx].inputs.clone();
            if input_gradients.len() != inputs.len() {
                return Err(OptimError::ComputationError(format!(
                    "backward for node {idx} produced {} gradients for {} inputs",
                    input_gradients.len(),
                    inputs.len()
                )));
            }

            for (grad, input_id) in input_gradients.into_iter().zip(inputs) {
                let expected = self.values.get(input_id).map(|v| v.len()).ok_or_else(|| {
                    OptimError::ComputationError(format!("missing value for node {input_id}"))
                })?;
                if grad.len() != expected {
                    return Err(OptimError::ComputationError(format!(
                        "gradient length {} does not match input {input_id} length {expected}",
                        grad.len()
                    )));
                }
                match self.gradients.get_mut(input_id) {
                    Some(Some(existing)) => *existing = &*existing + &grad,
                    Some(slot @ None) => *slot = Some(grad),
                    None => {
                        return Err(OptimError::ComputationError(format!(
                            "gradient slot {input_id} out of range"
                        )))
                    }
                }
            }
        }

        Ok(())
    }

    /// Get gradient for a variable
    pub fn get_gradient(&self, varid: usize) -> Option<&Array1<T>> {
        self.gradients.get(varid)?.as_ref()
    }

    /// Get gradient by variable name
    pub fn get_gradient_by_name(&self, name: &str) -> Option<&Array1<T>> {
        let varid = *self.variables.get(name)?;
        self.get_gradient(varid)
    }

    /// Zero all gradients
    pub fn zero_gradients(&mut self) {
        for grad in self.gradients.iter_mut().flatten() {
            grad.fill(T::zero());
        }
        self.cache.clear();
    }

    /// Get all named variable gradients
    pub fn get_all_gradients(&self) -> HashMap<String, Array1<T>> {
        let mut result = HashMap::new();

        for (name, &varid) in &self.variables {
            if let Some(grad) = self.get_gradient(varid) {
                result.insert(name.clone(), grad.clone());
            }
        }

        result
    }

    // -------------------------------------------------------------- internals

    fn binary_op(
        &mut self,
        op_type: ReverseOpType,
        lhs: usize,
        rhs: usize,
        backward_fn: BackwardFunction<T>,
    ) -> Result<usize> {
        self.push_op(op_type, vec![lhs, rhs], backward_fn)
    }

    fn unary_op(
        &mut self,
        op_type: ReverseOpType,
        input: usize,
        backward_fn: BackwardFunction<T>,
    ) -> Result<usize> {
        self.push_op(op_type, vec![input], backward_fn)
    }

    fn push_op(
        &mut self,
        op_type: ReverseOpType,
        inputs: Vec<usize>,
        backward_fn: BackwardFunction<T>,
    ) -> Result<usize> {
        for &input in &inputs {
            if input >= self.tape.len() {
                return Err(OptimError::InvalidConfig(format!(
                    "operand {input} is not a valid tape node"
                )));
            }
        }

        let outputid = self.tape.len();
        // When recording is off the node is *detached*: it is still evaluated
        // and still records its inputs (so the forward pass stays correct and
        // the index invariant holds), but the reverse sweep skips it, so no
        // gradient flows through it.
        self.tape.push(ReverseOperation {
            op_type,
            inputs: inputs.clone(),
            backward_fn: backward_fn.clone(),
            detached: !self.recording,
        });
        // Placeholder slots; overwritten by `evaluate_op` below.
        self.values.push(Array1::zeros(0));
        self.shapes.push(vec![0]);
        self.requiresgrad.push(false);
        self.gradients.push(None);

        let (value, shape) = match self.evaluate_op(op_type, &inputs, &backward_fn) {
            Ok(v) => v,
            Err(e) => {
                // Roll the partially recorded node back so the tape stays consistent.
                self.tape.pop();
                self.values.pop();
                self.shapes.pop();
                self.requiresgrad.pop();
                self.gradients.pop();
                return Err(e);
            }
        };
        self.values[outputid] = value;
        self.shapes[outputid] = shape;

        Ok(outputid)
    }

    /// Re-evaluate node `idx` from its recorded inputs.
    fn evaluate_node(&self, idx: usize) -> Result<(Array1<T>, Vec<usize>)> {
        let op = &self.tape[idx];
        self.evaluate_op(op.op_type, &op.inputs, &op.backward_fn)
    }

    fn evaluate_op(
        &self,
        op_type: ReverseOpType,
        inputs: &[usize],
        backward_fn: &BackwardFunction<T>,
    ) -> Result<(Array1<T>, Vec<usize>)> {
        match op_type {
            ReverseOpType::Variable | ReverseOpType::Constant => Err(OptimError::ComputationError(
                "leaf nodes carry their value directly and cannot be evaluated".to_string(),
            )),

            ReverseOpType::Add
            | ReverseOpType::Subtract
            | ReverseOpType::Multiply
            | ReverseOpType::Divide => {
                let (lhs, rhs) = self.binary_operands(inputs)?;
                self.require_same_len(lhs, rhs, op_type)?;
                let out = match op_type {
                    ReverseOpType::Add => lhs + rhs,
                    ReverseOpType::Subtract => lhs - rhs,
                    ReverseOpType::Multiply => lhs * rhs,
                    _ => {
                        for &d in rhs.iter() {
                            if d == T::zero() {
                                return Err(OptimError::ComputationError(
                                    "division by zero in reverse-mode tape".to_string(),
                                ));
                            }
                        }
                        lhs / rhs
                    }
                };
                let shape = self.shape_of(inputs[0])?.to_vec();
                Ok((out, shape))
            }

            ReverseOpType::Dot => {
                let (lhs, rhs) = self.binary_operands(inputs)?;
                self.require_same_len(lhs, rhs, op_type)?;
                let acc = lhs
                    .iter()
                    .zip(rhs.iter())
                    .fold(T::zero(), |a, (&x, &y)| a + x * y);
                Ok((Array1::from_elem(1, acc), vec![1]))
            }

            ReverseOpType::MatMul => {
                let (m, k, n, rhs_is_vector) = self.matmul_dims(inputs)?;
                let (lhs, rhs) = self.binary_operands(inputs)?;
                let out = matmul_flat(lhs, m, k, rhs, n);
                let shape = if rhs_is_vector { vec![m] } else { vec![m, n] };
                Ok((out, shape))
            }

            ReverseOpType::Power => {
                let x = self.unary_operand(inputs)?;
                let exponent = match backward_fn {
                    BackwardFunction::PowerBackward { exponent } => *exponent,
                    _ => {
                        return Err(OptimError::ComputationError(
                            "power node is missing its exponent".to_string(),
                        ))
                    }
                };
                Ok((
                    x.mapv(|v| v.powf(exponent)),
                    self.shape_of(inputs[0])?.to_vec(),
                ))
            }

            ReverseOpType::Exp => {
                let x = self.unary_operand(inputs)?;
                Ok((x.mapv(|v| v.exp()), self.shape_of(inputs[0])?.to_vec()))
            }

            ReverseOpType::Log => {
                let x = self.unary_operand(inputs)?;
                for &v in x.iter() {
                    if v <= T::zero() {
                        return Err(OptimError::ComputationError(
                            "log of a non-positive value in reverse-mode tape".to_string(),
                        ));
                    }
                }
                Ok((x.mapv(|v| v.ln()), self.shape_of(inputs[0])?.to_vec()))
            }

            ReverseOpType::Sin => {
                let x = self.unary_operand(inputs)?;
                Ok((x.mapv(|v| v.sin()), self.shape_of(inputs[0])?.to_vec()))
            }

            ReverseOpType::Cos => {
                let x = self.unary_operand(inputs)?;
                Ok((x.mapv(|v| v.cos()), self.shape_of(inputs[0])?.to_vec()))
            }

            ReverseOpType::Tanh => {
                let x = self.unary_operand(inputs)?;
                Ok((x.mapv(|v| v.tanh()), self.shape_of(inputs[0])?.to_vec()))
            }

            ReverseOpType::Sigmoid => {
                let x = self.unary_operand(inputs)?;
                Ok((
                    x.mapv(|v| T::one() / (T::one() + (-v).exp())),
                    self.shape_of(inputs[0])?.to_vec(),
                ))
            }

            ReverseOpType::ReLU => {
                let x = self.unary_operand(inputs)?;
                Ok((
                    x.mapv(|v| if v > T::zero() { v } else { T::zero() }),
                    self.shape_of(inputs[0])?.to_vec(),
                ))
            }

            ReverseOpType::LeakyReLU => {
                let x = self.unary_operand(inputs)?;
                let alpha = match backward_fn {
                    BackwardFunction::ActivationBackward {
                        function: ActivationFunction::LeakyReLU { alpha },
                    } => *alpha,
                    _ => {
                        return Err(OptimError::ComputationError(
                            "leaky relu node is missing its slope".to_string(),
                        ))
                    }
                };
                let alpha_t: T = scirs2_core::numeric::NumCast::from(alpha).ok_or_else(|| {
                    OptimError::ComputationError("failed to convert leaky-relu slope".to_string())
                })?;
                Ok((
                    x.mapv(|v| if v > T::zero() { v } else { alpha_t * v }),
                    self.shape_of(inputs[0])?.to_vec(),
                ))
            }

            ReverseOpType::Sum => {
                let x = self.unary_operand(inputs)?;
                let acc = x.iter().copied().fold(T::zero(), |a, b| a + b);
                Ok((Array1::from_elem(1, acc), vec![1]))
            }

            ReverseOpType::Mean => {
                let x = self.unary_operand(inputs)?;
                if x.is_empty() {
                    return Err(OptimError::ComputationError(
                        "mean of an empty tensor".to_string(),
                    ));
                }
                let n = T::from(x.len()).ok_or_else(|| {
                    OptimError::ComputationError("failed to convert length".to_string())
                })?;
                let acc = x.iter().copied().fold(T::zero(), |a, b| a + b) / n;
                Ok((Array1::from_elem(1, acc), vec![1]))
            }

            ReverseOpType::Norm => {
                let x = self.unary_operand(inputs)?;
                let acc = x.iter().fold(T::zero(), |a, &b| a + b * b).sqrt();
                Ok((Array1::from_elem(1, acc), vec![1]))
            }

            ReverseOpType::Reshape => {
                let x = self.unary_operand(inputs)?;
                let newshape = match backward_fn {
                    BackwardFunction::ReshapeBackward { newshape } => newshape.clone(),
                    _ => {
                        return Err(OptimError::ComputationError(
                            "reshape node is missing its target shape".to_string(),
                        ))
                    }
                };
                let expected: usize = newshape.iter().product();
                if expected != x.len() {
                    return Err(OptimError::ComputationError(format!(
                        "reshape to {newshape:?} is incompatible with {} elements",
                        x.len()
                    )));
                }
                Ok((x.clone(), newshape))
            }
        }
    }

    fn binary_operands(&self, inputs: &[usize]) -> Result<(&Array1<T>, &Array1<T>)> {
        if inputs.len() != 2 {
            return Err(OptimError::ComputationError(format!(
                "binary operation expects 2 operands, got {}",
                inputs.len()
            )));
        }
        Ok((self.value(inputs[0])?, self.value(inputs[1])?))
    }

    fn unary_operand(&self, inputs: &[usize]) -> Result<&Array1<T>> {
        if inputs.len() != 1 {
            return Err(OptimError::ComputationError(format!(
                "unary operation expects 1 operand, got {}",
                inputs.len()
            )));
        }
        self.value(inputs[0])
    }

    fn shape_of(&self, varid: usize) -> Result<&[usize]> {
        self.shapes
            .get(varid)
            .map(|s| s.as_slice())
            .ok_or_else(|| OptimError::InvalidConfig(format!("invalid variable id {varid}")))
    }

    fn require_same_len(
        &self,
        lhs: &Array1<T>,
        rhs: &Array1<T>,
        op_type: ReverseOpType,
    ) -> Result<()> {
        if lhs.len() != rhs.len() {
            return Err(OptimError::InvalidConfig(format!(
                "{:?} requires equal operand lengths, got {} and {}",
                op_type,
                lhs.len(),
                rhs.len()
            )));
        }
        Ok(())
    }

    /// Resolve `(m, k, n, rhs_is_vector)` for a matmul node.
    fn matmul_dims(&self, inputs: &[usize]) -> Result<(usize, usize, usize, bool)> {
        if inputs.len() != 2 {
            return Err(OptimError::ComputationError(
                "matmul expects 2 operands".to_string(),
            ));
        }
        let lhs_shape = self.shape_of(inputs[0])?;
        let rhs_shape = self.shape_of(inputs[1])?;
        if lhs_shape.len() != 2 {
            return Err(OptimError::InvalidConfig(format!(
                "matmul left operand must be 2-D, got shape {lhs_shape:?}"
            )));
        }
        let (m, k) = (lhs_shape[0], lhs_shape[1]);
        let (n, rhs_is_vector) = match rhs_shape.len() {
            1 => (1usize, true),
            2 => (rhs_shape[1], false),
            _ => {
                return Err(OptimError::InvalidConfig(format!(
                    "matmul right operand must be 1-D or 2-D, got shape {rhs_shape:?}"
                )))
            }
        };
        let rhs_rows = rhs_shape[0];
        if rhs_rows != k {
            return Err(OptimError::InvalidConfig(format!(
                "matmul inner dimensions disagree: {lhs_shape:?} x {rhs_shape:?}"
            )));
        }
        Ok((m, k, n, rhs_is_vector))
    }

    fn compute_backward_pass(&self, idx: usize, output_grad: &Array1<T>) -> Result<Vec<Array1<T>>> {
        let op = &self.tape[idx];
        let inputs = &op.inputs;

        match &op.backward_fn {
            BackwardFunction::Leaf => Ok(Vec::new()),

            BackwardFunction::AddBackward => Ok(vec![output_grad.clone(), output_grad.clone()]),

            BackwardFunction::SubtractBackward => {
                Ok(vec![output_grad.clone(), output_grad.mapv(|x| -x)])
            }

            BackwardFunction::MultiplyBackward => {
                let (lhs, rhs) = self.binary_operands(inputs)?;
                Ok(vec![output_grad * rhs, output_grad * lhs])
            }

            BackwardFunction::DivideBackward => {
                let (lhs, rhs) = self.binary_operands(inputs)?;
                let lhs_grad = output_grad / rhs;
                let rhs_grad = Array1::from_shape_fn(rhs.len(), |i| {
                    -output_grad[i] * lhs[i] / (rhs[i] * rhs[i])
                });
                Ok(vec![lhs_grad, rhs_grad])
            }

            BackwardFunction::DotBackward => {
                let (lhs, rhs) = self.binary_operands(inputs)?;
                if output_grad.len() != 1 {
                    return Err(OptimError::ComputationError(
                        "dot backward expects a scalar upstream gradient".to_string(),
                    ));
                }
                let g = output_grad[0];
                Ok(vec![rhs.mapv(|v| v * g), lhs.mapv(|v| v * g)])
            }

            BackwardFunction::PowerBackward { exponent } => {
                let x = self.unary_operand(inputs)?;
                let derivative = x.mapv(|v| *exponent * v.powf(*exponent - T::one()));
                Ok(vec![output_grad * &derivative])
            }

            BackwardFunction::ExpBackward => {
                let x = self.unary_operand(inputs)?;
                let derivative = x.mapv(|v| v.exp());
                Ok(vec![output_grad * &derivative])
            }

            BackwardFunction::LogBackward => {
                let x = self.unary_operand(inputs)?;
                Ok(vec![output_grad / x])
            }

            BackwardFunction::TrigBackward { function } => {
                let x = self.unary_operand(inputs)?;
                let derivative = match function {
                    TrigFunction::Sin => x.mapv(|v| v.cos()),
                    TrigFunction::Cos => x.mapv(|v| -v.sin()),
                };
                Ok(vec![output_grad * &derivative])
            }

            BackwardFunction::ActivationBackward { function } => {
                let x = self.unary_operand(inputs)?;
                let derivative = match function {
                    ActivationFunction::Tanh => {
                        let tanh_val = x.mapv(|v| v.tanh());
                        tanh_val.mapv(|y| T::one() - y * y)
                    }
                    ActivationFunction::Sigmoid => {
                        let sigmoid_val = x.mapv(|v| T::one() / (T::one() + (-v).exp()));
                        sigmoid_val.mapv(|y| y * (T::one() - y))
                    }
                    ActivationFunction::ReLU => {
                        x.mapv(|v| if v > T::zero() { T::one() } else { T::zero() })
                    }
                    ActivationFunction::LeakyReLU { alpha } => {
                        let alpha_t: T =
                            scirs2_core::numeric::NumCast::from(*alpha).ok_or_else(|| {
                                OptimError::ComputationError(
                                    "failed to convert leaky-relu slope".to_string(),
                                )
                            })?;
                        x.mapv(|v| if v > T::zero() { T::one() } else { alpha_t })
                    }
                };
                Ok(vec![output_grad * &derivative])
            }

            BackwardFunction::MatMulBackward => {
                let (m, k, n, _) = self.matmul_dims(inputs)?;
                let (lhs, rhs) = self.binary_operands(inputs)?;
                if output_grad.len() != m * n {
                    return Err(OptimError::ComputationError(format!(
                        "matmul upstream gradient has {} elements, expected {}",
                        output_grad.len(),
                        m * n
                    )));
                }
                // dL/dA = G · Bᵀ  (m×n · n×k -> m×k)
                let mut lhs_grad = Array1::zeros(m * k);
                for i in 0..m {
                    for p in 0..k {
                        let mut acc = T::zero();
                        for j in 0..n {
                            acc = acc + output_grad[i * n + j] * rhs[p * n + j];
                        }
                        lhs_grad[i * k + p] = acc;
                    }
                }
                // dL/dB = Aᵀ · G  (k×m · m×n -> k×n)
                let mut rhs_grad = Array1::zeros(k * n);
                for p in 0..k {
                    for j in 0..n {
                        let mut acc = T::zero();
                        for i in 0..m {
                            acc = acc + lhs[i * k + p] * output_grad[i * n + j];
                        }
                        rhs_grad[p * n + j] = acc;
                    }
                }
                Ok(vec![lhs_grad, rhs_grad])
            }

            BackwardFunction::ReductionBackward { reduction_type } => {
                let x = self.unary_operand(inputs)?;
                if output_grad.len() != 1 {
                    return Err(OptimError::ComputationError(
                        "reduction backward expects a scalar upstream gradient".to_string(),
                    ));
                }
                let g = output_grad[0];
                match reduction_type {
                    ReductionType::Sum => Ok(vec![Array1::from_elem(x.len(), g)]),
                    ReductionType::Mean => {
                        if x.is_empty() {
                            return Err(OptimError::ComputationError(
                                "mean backward on an empty tensor".to_string(),
                            ));
                        }
                        let n = T::from(x.len()).ok_or_else(|| {
                            OptimError::ComputationError("failed to convert length".to_string())
                        })?;
                        Ok(vec![Array1::from_elem(x.len(), g / n)])
                    }
                    ReductionType::Norm => {
                        let norm = x.iter().fold(T::zero(), |a, &v| a + v * v).sqrt();
                        if norm <= T::zero() {
                            // d‖x‖/dx is undefined at the origin; the
                            // sub-gradient 0 is the conventional choice.
                            Ok(vec![Array1::zeros(x.len())])
                        } else {
                            Ok(vec![x.mapv(|v| v * g / norm)])
                        }
                    }
                }
            }

            BackwardFunction::ReshapeBackward { .. } => {
                let x = self.unary_operand(inputs)?;
                if output_grad.len() != x.len() {
                    return Err(OptimError::ComputationError(format!(
                        "reshape backward received {} elements, expected {}",
                        output_grad.len(),
                        x.len()
                    )));
                }
                Ok(vec![output_grad.clone()])
            }
        }
    }

    /// Get tape statistics
    pub fn get_tape_stats(&self) -> ReverseModeStats {
        ReverseModeStats {
            tape_length: self.tape.len(),
            num_variables: self.variables.len(),
            num_gradients: self.gradients.iter().filter(|g| g.is_some()).count(),
            memory_usage_estimate: self.estimate_memory_usage(),
            cache_size: self.cache.len(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let tape_size = self.tape.len() * std::mem::size_of::<ReverseOperation<T>>();
        let value_size = self
            .values
            .iter()
            .map(|v| v.len() * std::mem::size_of::<T>())
            .sum::<usize>();
        let gradient_size = self
            .gradients
            .iter()
            .filter_map(|g| g.as_ref())
            .map(|g| g.len() * std::mem::size_of::<T>())
            .sum::<usize>();

        tape_size + value_size + gradient_size
    }
}

/// Row-major matrix product of a flat `m x k` buffer with a flat `k x n` buffer.
fn matmul_flat<T: Float>(
    lhs: &Array1<T>,
    m: usize,
    k: usize,
    rhs: &Array1<T>,
    n: usize,
) -> Array1<T>
where
    T: Clone,
{
    let mut out = Array1::zeros(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = T::zero();
            for p in 0..k {
                acc = acc + lhs[i * k + p] * rhs[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Reverse-mode AD statistics
#[derive(Debug, Clone)]
pub struct ReverseModeStats {
    pub tape_length: usize,
    pub num_variables: usize,
    pub num_gradients: usize,
    pub memory_usage_estimate: usize,
    pub cache_size: usize,
}

/// Gradient accumulation utilities
pub struct GradientAccumulator<T: Float + Debug + Send + Sync + 'static> {
    /// Accumulated gradients
    gradients: HashMap<String, Array1<T>>,

    /// Accumulation count
    count: usize,
}

impl<T: Float + Debug + Default + Clone + scirs2_core::ndarray::ScalarOperand + Send + Sync> Default
    for GradientAccumulator<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + Clone + scirs2_core::ndarray::ScalarOperand + Send + Sync>
    GradientAccumulator<T>
{
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
            count: 0,
        }
    }

    /// Accumulate gradients
    pub fn accumulate(&mut self, gradients: HashMap<String, Array1<T>>) {
        for (name, grad) in gradients {
            if let Some(existing) = self.gradients.get_mut(&name) {
                *existing = &*existing + &grad;
            } else {
                self.gradients.insert(name, grad);
            }
        }
        self.count += 1;
    }

    /// Get averaged gradients
    pub fn get_averaged_gradients(&self) -> HashMap<String, Array1<T>> {
        if self.count == 0 {
            return HashMap::new();
        }

        let count_t: T =
            scirs2_core::numeric::NumCast::from(self.count).unwrap_or_else(|| T::one());
        self.gradients
            .iter()
            .map(|(name, grad)| (name.clone(), grad / count_t))
            .collect()
    }

    /// Clear accumulated gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_mode_engine_creation() {
        let engine = ReverseModeEngine::<f64>::new();
        assert!(engine.is_recording());
        assert_eq!(engine.tape_len(), 0);
    }

    #[test]
    fn test_variable_creation() {
        let mut engine = ReverseModeEngine::new();
        let x_val = Array1::from_vec(vec![2.0, 3.0]);
        let x_id = engine.create_variable("x", x_val, true);

        assert_eq!(x_id, 0);
        assert!(engine.value_by_name("x").is_ok());
        assert!(engine.get_gradient(x_id).is_some());
    }

    #[test]
    fn test_simple_backward_pass() {
        let mut engine = ReverseModeEngine::new();

        let x_id = engine.create_variable("x", Array1::from_vec(vec![2.0]), true);
        let y_id = engine.create_variable("y", Array1::from_vec(vec![3.0]), true);

        let sum_id = engine.add(x_id, y_id).expect("add op");
        assert!((engine.scalar_value(sum_id).expect("value") - 5.0).abs() < 1e-12);

        engine
            .backward(sum_id, Some(Array1::from_vec(vec![1.0])))
            .expect("backward");

        let x_grad = engine.get_gradient(x_id).expect("x gradient");
        let y_grad = engine.get_gradient(y_id).expect("y gradient");

        assert_eq!(x_grad[0], 1.0);
        assert_eq!(y_grad[0], 1.0);
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = GradientAccumulator::<f64>::new();

        let mut grad1 = HashMap::new();
        grad1.insert("x".to_string(), Array1::from_vec(vec![1.0, 2.0]));

        let mut grad2 = HashMap::new();
        grad2.insert("x".to_string(), Array1::from_vec(vec![3.0, 4.0]));

        accumulator.accumulate(grad1);
        accumulator.accumulate(grad2);

        let averaged = accumulator.get_averaged_gradients();
        let x_avg = &averaged["x"];

        assert_eq!(x_avg[0], 2.0);
        assert_eq!(x_avg[1], 3.0);
    }

    #[test]
    fn test_tape_statistics() {
        let mut engine = ReverseModeEngine::new();

        let x_id = engine.create_variable("x", Array1::from_vec(vec![1.0]), true);
        let _exp_id = engine.exp(x_id).expect("exp op");

        let stats = engine.get_tape_stats();
        assert_eq!(stats.tape_length, 2);
        assert_eq!(stats.num_variables, 1);
    }

    #[test]
    fn test_reverse_mode_gradient_of_product() {
        let mut engine = ReverseModeEngine::<f64>::new();

        let x_id = engine.create_variable("x", Array1::from_vec(vec![5.0]), true);
        let y_id = engine.create_variable("y", Array1::from_vec(vec![7.0]), true);

        let prod_id = engine.multiply(x_id, y_id).expect("multiply op");

        engine
            .backward(prod_id, Some(Array1::from_vec(vec![1.0])))
            .expect("backward");

        let x_grad = engine.get_gradient(x_id).expect("x gradient");
        let y_grad = engine.get_gradient(y_id).expect("y gradient");

        approx::assert_abs_diff_eq!(x_grad[0], 7.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(y_grad[0], 5.0, epsilon = 1e-10);
    }

    /// F17 regression: a composite expression needs every intermediate node's
    /// forward value, not just depth-1 leaves.
    ///
    /// f(x, y) = exp(x + y); df/dx = df/dy = exp(x + y).
    #[test]
    fn test_forward_values_materialised_for_composites() {
        let mut engine = ReverseModeEngine::<f64>::new();

        let x_id = engine.create_variable("x", Array1::from_vec(vec![0.5]), true);
        let y_id = engine.create_variable("y", Array1::from_vec(vec![0.25]), true);

        let sum_id = engine.add(x_id, y_id).expect("add");
        let exp_id = engine.exp(sum_id).expect("exp");

        let expected = 0.75f64.exp();
        approx::assert_abs_diff_eq!(
            engine.scalar_value(exp_id).expect("value"),
            expected,
            epsilon = 1e-12
        );

        engine.backward(exp_id, None).expect("backward");

        approx::assert_abs_diff_eq!(
            engine.get_gradient(x_id).expect("x grad")[0],
            expected,
            epsilon = 1e-10
        );
        approx::assert_abs_diff_eq!(
            engine.get_gradient(y_id).expect("y grad")[0],
            expected,
            epsilon = 1e-10
        );
    }

    /// F19 regression: creating nodes *after* the node being differentiated
    /// used to index `gradients` out of bounds during the reverse sweep.
    #[test]
    fn test_backward_from_earlier_node_does_not_panic() {
        let mut engine = ReverseModeEngine::<f64>::new();

        let x_id = engine.create_variable("x", Array1::from_vec(vec![1.0]), true);
        // Extends the tape past `x_id` without extending gradient storage.
        let _exp_id = engine.exp(x_id).expect("exp");

        engine
            .backward(x_id, None)
            .expect("backward must not panic");

        let x_grad = engine.get_gradient(x_id).expect("x grad");
        approx::assert_abs_diff_eq!(x_grad[0], 1.0, epsilon = 1e-12);
    }

    /// F18 regression: matmul gradients checked against central finite
    /// differences of the scalar loss `L = sum(A · B)`.
    #[test]
    fn test_matmul_gradient_matches_finite_difference() {
        let a = vec![0.5, -1.5, 2.0, 0.25, 1.0, -0.75]; // 2x3
        let b = vec![1.5, -0.5, 0.25, 2.0, -1.0, 0.75]; // 3x2

        let build = |a: &[f64], b: &[f64]| -> (ReverseModeEngine<f64>, usize, usize, usize) {
            let mut engine = ReverseModeEngine::<f64>::new();
            let a_id = engine
                .create_variable_with_shape("a", Array1::from_vec(a.to_vec()), &[2, 3], true)
                .expect("a");
            let b_id = engine
                .create_variable_with_shape("b", Array1::from_vec(b.to_vec()), &[3, 2], true)
                .expect("b");
            let mm = engine.matmul(a_id, b_id).expect("matmul");
            let loss = engine.sum(mm, None).expect("sum");
            (engine, a_id, b_id, loss)
        };

        let (mut engine, a_id, b_id, loss_id) = build(&a, &b);
        // Reference product, checked by hand for the first entry.
        let product = engine
            .value(engine.tape_len() - 2)
            .expect("product")
            .clone();
        approx::assert_abs_diff_eq!(
            product[0],
            0.5 * 1.5 + (-1.5) * 0.25 + 2.0 * (-1.0),
            epsilon = 1e-12
        );

        engine.backward(loss_id, None).expect("backward");
        let grad_a = engine.get_gradient(a_id).expect("grad a").clone();
        let grad_b = engine.get_gradient(b_id).expect("grad b").clone();

        let loss_at = |a: &[f64], b: &[f64]| -> f64 {
            let (engine, _, _, loss_id) = build(a, b);
            engine.scalar_value(loss_id).expect("loss")
        };

        let h = 1e-6;
        for i in 0..a.len() {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[i] += h;
            am[i] -= h;
            let fd = (loss_at(&ap, &b) - loss_at(&am, &b)) / (2.0 * h);
            approx::assert_abs_diff_eq!(grad_a[i], fd, epsilon = 1e-6);
        }
        for i in 0..b.len() {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[i] += h;
            bm[i] -= h;
            let fd = (loss_at(&a, &bp) - loss_at(&a, &bm)) / (2.0 * h);
            approx::assert_abs_diff_eq!(grad_b[i], fd, epsilon = 1e-6);
        }
    }

    /// Matrix times column vector, the shape used by the linear meta-models.
    #[test]
    fn test_matmul_matrix_vector() {
        let mut engine = ReverseModeEngine::<f64>::new();
        let w = engine
            .create_variable_with_shape(
                "w",
                Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                &[2, 3],
                true,
            )
            .expect("w");
        let x = engine.create_variable("x", Array1::from_vec(vec![1.0, 0.0, -1.0]), false);
        let y = engine.matmul(w, x).expect("matmul");

        let value = engine.value(y).expect("value");
        assert_eq!(value.len(), 2);
        approx::assert_abs_diff_eq!(value[0], 1.0 - 3.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(value[1], 4.0 - 6.0, epsilon = 1e-12);

        engine
            .backward(y, Some(Array1::from_vec(vec![1.0, 1.0])))
            .expect("backward");
        let gw = engine.get_gradient(w).expect("grad w");
        // dL/dW = g xᵀ with g = [1, 1] and x = [1, 0, -1].
        for row in 0..2 {
            approx::assert_abs_diff_eq!(gw[row * 3], 1.0, epsilon = 1e-12);
            approx::assert_abs_diff_eq!(gw[row * 3 + 1], 0.0, epsilon = 1e-12);
            approx::assert_abs_diff_eq!(gw[row * 3 + 2], -1.0, epsilon = 1e-12);
        }
    }

    /// Mismatched operand lengths must be a hard error, not a panic and not a
    /// silently wrong gradient.
    #[test]
    fn test_mismatched_lengths_error() {
        let mut engine = ReverseModeEngine::<f64>::new();
        let a = engine.create_variable("a", Array1::from_vec(vec![1.0, 2.0]), true);
        let b = engine.create_variable("b", Array1::from_vec(vec![1.0]), true);
        assert!(engine.add(a, b).is_err());
        // The failed op must not have been left on the tape.
        assert_eq!(engine.tape_len(), 2);
    }

    /// Re-running `forward` after assigning new leaf values must refresh every
    /// intermediate value and give fresh gradients.
    #[test]
    fn test_set_variable_value_and_reforward() {
        let mut engine = ReverseModeEngine::<f64>::new();
        let x = engine.create_variable("x", Array1::from_vec(vec![2.0]), true);
        let sq = engine.multiply(x, x).expect("square");

        engine.backward(sq, None).expect("backward");
        approx::assert_abs_diff_eq!(
            engine.get_gradient(x).expect("grad")[0],
            4.0,
            epsilon = 1e-12
        );

        engine
            .set_variable_value_by_name("x", Array1::from_vec(vec![5.0]))
            .expect("assign");
        engine.backward(sq, None).expect("backward again");
        approx::assert_abs_diff_eq!(
            engine.scalar_value(sq).expect("value"),
            25.0,
            epsilon = 1e-12
        );
        approx::assert_abs_diff_eq!(
            engine.get_gradient(x).expect("grad")[0],
            10.0,
            epsilon = 1e-12
        );
    }
}
