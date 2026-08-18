use std::fmt::Debug;
// Forward-mode automatic differentiation
//
// This module implements forward-mode automatic differentiation for computing
// directional derivatives and Jacobian-vector products efficiently.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// Dual number for forward-mode automatic differentiation
#[derive(Debug, Clone)]
pub struct DualNumber<T: Float + Debug + Send + Sync + 'static> {
    /// Primal value
    pub value: T,

    /// Tangent (derivative) value
    pub tangent: T,
}

/// Multi-dimensional dual number for vector-valued functions
#[derive(Debug, Clone)]
pub struct VectorDual<T: Float + Debug + Send + Sync + 'static> {
    /// Primal value (vector)
    pub value: Array1<T>,

    /// Tangent matrix (Jacobian-vector product)
    pub tangent: Array1<T>,
}

/// Forward-mode AD engine
pub struct ForwardModeEngine<
    T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static,
> {
    /// Computation graph
    tape: Vec<ForwardOperation<T>>,

    /// Variable registry (name -> tape index)
    variables: HashMap<String, usize>,

    /// Variable names in creation order.
    ///
    /// The seed/tangent space is the concatenation of the variables' values in
    /// **this** order, so a variable's seed ordinal is its position here — it
    /// is stable and independent of how many intermediate operations were
    /// recorded between two variable creations (which is what the tape index
    /// would have measured).
    variable_order: Vec<String>,

    /// Current seed vectors for directional derivatives
    seed_vectors: Vec<Array1<T>>,

    /// Enable higher-order derivatives
    higher_order: bool,
}

/// Forward-mode operation
#[derive(Debug, Clone)]
struct ForwardOperation<T: Float + Debug + Send + Sync + 'static> {
    /// Operation type
    optype: ForwardOpType,

    /// Input variable indices
    inputs: Vec<usize>,

    /// Operation metadata
    metadata: ForwardOpMetadata<T>,
}

/// Forward operation types
#[derive(Debug, Clone)]
enum ForwardOpType {
    Variable,
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
    MatMul,
    Dot,
    Sum,
    Mean,
    Norm,
}

/// Operation metadata for forward mode
#[derive(Debug, Clone)]
struct ForwardOpMetadata<T: Float + Debug + Send + Sync + 'static> {
    /// Logical shape of the operation's output
    shape: Vec<usize>,

    /// Operation-specific data
    data: ForwardOpData<T>,
}

/// Operation-specific data
#[derive(Debug, Clone)]
enum ForwardOpData<T: Float + Debug + Send + Sync + 'static> {
    None,
    /// The full constant tensor (flat, row-major)
    ConstantVector(Array1<T>),
    PowerExponent(T),
    /// `lhs` is `m x k`, `rhs` is `k x n`, output is `m x n`
    MatMulDims {
        m: usize,
        k: usize,
        n: usize,
    },
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> DualNumber<T> {
    /// Create a new dual number
    pub fn new(value: T, tangent: T) -> Self {
        Self { value, tangent }
    }

    /// Create a constant (zero tangent)
    pub fn constant(value: T) -> Self {
        Self::new(value, T::zero())
    }

    /// Create a variable (unit tangent)
    pub fn variable(value: T) -> Self {
        Self::new(value, T::one())
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> VectorDual<T> {
    /// Create a new vector dual number
    pub fn new(value: Array1<T>, tangent: Array1<T>) -> Self {
        Self { value, tangent }
    }

    /// Create a constant vector
    pub fn constant(value: Array1<T>) -> Self {
        let tangent = Array1::zeros(value.len());
        Self::new(value, tangent)
    }

    /// Create a variable vector with unit tangent in direction i
    pub fn variable(value: Array1<T>, direction: usize) -> Self {
        let mut tangent = Array1::zeros(value.len());
        if direction < tangent.len() {
            tangent[direction] = T::one();
        }
        Self::new(value, tangent)
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static> Default
    for ForwardModeEngine<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static>
    ForwardModeEngine<T>
{
    /// Create a new forward-mode AD engine
    pub fn new() -> Self {
        Self {
            tape: Vec::new(),
            variables: HashMap::new(),
            variable_order: Vec::new(),
            seed_vectors: Vec::new(),
            higher_order: false,
        }
    }

    /// Enable higher-order derivatives
    pub fn enable_higher_order(&mut self, enabled: bool) {
        self.higher_order = enabled;
    }

    /// Whether higher-order derivative tracking is enabled
    pub fn higher_order_enabled(&self) -> bool {
        self.higher_order
    }

    /// Set seed vectors for computing directional derivatives
    pub fn set_seed_vectors(&mut self, seeds: Vec<Array1<T>>) {
        self.seed_vectors = seeds;
    }

    /// Seed vectors previously registered with [`Self::set_seed_vectors`]
    pub fn seed_vectors(&self) -> &[Array1<T>] {
        &self.seed_vectors
    }

    /// Variable names in creation order (the seed-space layout)
    pub fn variable_order(&self) -> &[String] {
        &self.variable_order
    }

    /// Declared logical shape of a tape node, when the builder recorded one.
    ///
    /// Leaves and matrix products carry an explicit shape; element-wise
    /// operations infer theirs from their operands during the forward pass and
    /// therefore report `None`.
    pub fn declared_shape(&self, node_id: usize) -> Result<Option<&[usize]>> {
        let op = self
            .tape
            .get(node_id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("invalid node id {node_id}")))?;
        if op.metadata.shape.is_empty() {
            Ok(None)
        } else {
            Ok(Some(op.metadata.shape.as_slice()))
        }
    }

    /// Create a variable.
    ///
    /// Re-creating a variable under an existing name rebinds the name to the
    /// new tape node but keeps its original seed ordinal.
    pub fn create_variable(&mut self, name: &str, value: Array1<T>) -> usize {
        let var_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Variable,
            inputs: Vec::new(),
            metadata: ForwardOpMetadata {
                shape: value.shape().to_vec(),
                data: ForwardOpData::None,
            },
        };

        self.tape.push(op);
        if self.variables.insert(name.to_string(), var_id).is_none() {
            self.variable_order.push(name.to_string());
        }
        var_id
    }

    /// Create a constant
    pub fn create_constant(&mut self, value: Array1<T>) -> usize {
        let const_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Constant,
            inputs: Vec::new(),
            metadata: ForwardOpMetadata {
                shape: value.shape().to_vec(),
                data: ForwardOpData::ConstantVector(value),
            },
        };

        self.tape.push(op);
        const_id
    }

    /// Total dimension of the seed / tangent space for the given inputs.
    ///
    /// This is the sum of the input lengths taken in variable-creation order.
    pub fn input_dimension(&self, inputs: &HashMap<String, Array1<T>>) -> Result<usize> {
        let mut total = 0usize;
        for name in &self.variable_order {
            let value = inputs.get(name).ok_or_else(|| {
                OptimError::InvalidConfig(format!("input value for '{name}' not provided"))
            })?;
            total += value.len();
        }
        Ok(total)
    }

    /// Offset of each variable inside the seed / tangent space.
    fn seed_offsets(&self, inputs: &HashMap<String, Array1<T>>) -> Result<HashMap<String, usize>> {
        let mut offsets = HashMap::new();
        let mut cursor = 0usize;
        for name in &self.variable_order {
            let value = inputs.get(name).ok_or_else(|| {
                OptimError::InvalidConfig(format!("input value for '{name}' not provided"))
            })?;
            offsets.insert(name.clone(), cursor);
            cursor += value.len();
        }
        Ok(offsets)
    }

    /// Add two variables
    pub fn add(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Add, lhs, rhs)
    }

    /// Subtract two variables
    pub fn subtract(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Subtract, lhs, rhs)
    }

    /// Multiply two variables
    pub fn multiply(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Multiply, lhs, rhs)
    }

    /// Divide two variables
    pub fn divide(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Divide, lhs, rhs)
    }

    /// Raise to power
    pub fn power(&mut self, base: usize, exponent: T) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::Power,
            inputs: vec![base],
            metadata: ForwardOpMetadata {
                shape: Vec::new(), // Inferred from the operand during the forward pass
                data: ForwardOpData::PowerExponent(exponent),
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    /// Exponential function
    pub fn exp(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Exp, input)
    }

    /// Natural logarithm
    pub fn log(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Log, input)
    }

    /// Sine function
    pub fn sin(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Sin, input)
    }

    /// Cosine function
    pub fn cos(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Cos, input)
    }

    /// Hyperbolic tangent
    pub fn tanh(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Tanh, input)
    }

    /// Sigmoid function
    pub fn sigmoid(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Sigmoid, input)
    }

    /// ReLU function
    pub fn relu(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::ReLU, input)
    }

    /// Matrix multiplication.
    ///
    /// `dims` is `(m, k, n)`: the left operand is the row-major flattening of
    /// an `m x k` matrix, the right operand of a `k x n` matrix, and the
    /// output is `m x n`.
    pub fn matmul(&mut self, lhs: usize, rhs: usize, dims: (usize, usize, usize)) -> Result<usize> {
        let (m, k, n) = dims;
        if m == 0 || k == 0 || n == 0 {
            return Err(OptimError::InvalidConfig(format!(
                "matmul dimensions must be non-zero, got {dims:?}"
            )));
        }
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype: ForwardOpType::MatMul,
            inputs: vec![lhs, rhs],
            metadata: ForwardOpMetadata {
                shape: vec![m, n],
                data: ForwardOpData::MatMulDims { m, k, n },
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    /// Dot product
    pub fn dot(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
        self.binary_op(ForwardOpType::Dot, lhs, rhs)
    }

    /// Sum reduction over all elements
    pub fn sum(&mut self, input: usize, _axis: Option<usize>) -> Result<usize> {
        self.unary_op(ForwardOpType::Sum, input)
    }

    /// Mean reduction over all elements
    pub fn mean(&mut self, input: usize, _axis: Option<usize>) -> Result<usize> {
        self.unary_op(ForwardOpType::Mean, input)
    }

    /// L2 norm
    pub fn norm(&mut self, input: usize) -> Result<usize> {
        self.unary_op(ForwardOpType::Norm, input)
    }

    /// Compute the forward pass with dual numbers.
    ///
    /// `seed_direction` is a vector in the concatenated input space: its length
    /// must equal [`Self::input_dimension`], and the slice belonging to a
    /// variable seeds that variable's tangent component-wise.
    pub fn forward_pass(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        seed_direction: &Array1<T>,
    ) -> Result<Vec<VectorDual<T>>> {
        let input_dim = self.input_dimension(inputs)?;
        if seed_direction.len() != input_dim {
            return Err(OptimError::InvalidConfig(format!(
                "seed direction has {} elements but the input space has {input_dim}",
                seed_direction.len()
            )));
        }
        let offsets = self.seed_offsets(inputs)?;
        let mut values: Vec<VectorDual<T>> = Vec::with_capacity(self.tape.len());

        for (idx, op) in self.tape.iter().enumerate() {
            match op.optype {
                ForwardOpType::Variable => {
                    let var_name = self
                        .variables
                        .iter()
                        .find(|(_, &id)| id == idx)
                        .map(|(name_, _)| name_.clone())
                        .ok_or_else(|| {
                            OptimError::InvalidConfig(format!(
                                "tape node {idx} is a variable with no registered name"
                            ))
                        })?;

                    let value = inputs
                        .get(&var_name)
                        .ok_or_else(|| {
                            OptimError::InvalidConfig(format!(
                                "input value for '{var_name}' not provided"
                            ))
                        })?
                        .clone();

                    let offset = *offsets.get(&var_name).ok_or_else(|| {
                        OptimError::ComputationError(format!(
                            "no seed offset registered for '{var_name}'"
                        ))
                    })?;
                    let tangent =
                        Array1::from_shape_fn(value.len(), |i| seed_direction[offset + i]);

                    values.push(VectorDual::new(value, tangent));
                }
                ForwardOpType::Constant => {
                    if let ForwardOpData::ConstantVector(ref val) = op.metadata.data {
                        let tangent = Array1::zeros(val.len());
                        values.push(VectorDual::new(val.clone(), tangent));
                    } else {
                        return Err(OptimError::InvalidConfig(
                            "constant node is missing its value".to_string(),
                        ));
                    }
                }
                _ => {
                    let result = self.compute_forward_operation(op, &values)?;
                    values.push(result);
                }
            }
        }

        Ok(values)
    }

    /// Compute Jacobian-vector product
    pub fn jacobian_vector_product(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        output_id: usize,
        direction: &Array1<T>,
    ) -> Result<Array1<T>> {
        let results = self.forward_pass(inputs, direction)?;

        if output_id >= results.len() {
            return Err(OptimError::InvalidConfig("Invalid output ID".to_string()));
        }

        Ok(results[output_id].tangent.clone())
    }

    /// Compute the full Jacobian of `output_id` with respect to the inputs.
    ///
    /// The result has shape `(output_dim, input_dim)` where `output_dim` is the
    /// number of elements of the selected output node and `input_dim` is
    /// [`Self::input_dimension`]. `input_size` must agree with `input_dim`.
    pub fn jacobian_matrix(
        &self,
        inputs: &HashMap<String, Array1<T>>,
        output_id: usize,
        input_size: usize,
    ) -> Result<Array2<T>> {
        let input_dim = self.input_dimension(inputs)?;
        if input_size != input_dim {
            return Err(OptimError::InvalidConfig(format!(
                "requested input size {input_size} does not match the input space dimension {input_dim}"
            )));
        }
        if input_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "cannot build a Jacobian for an empty input space".to_string(),
            ));
        }

        // One JVP per input coordinate; the first also fixes the output size.
        let mut columns: Vec<Array1<T>> = Vec::with_capacity(input_dim);
        for i in 0..input_dim {
            let mut direction = Array1::zeros(input_dim);
            direction[i] = T::one();
            columns.push(self.jacobian_vector_product(inputs, output_id, &direction)?);
        }

        let output_dim = columns
            .first()
            .map(|c| c.len())
            .ok_or_else(|| OptimError::ComputationError("no Jacobian columns".to_string()))?;
        let mut jacobian = Array2::zeros((output_dim, input_dim));
        for (i, column) in columns.iter().enumerate() {
            if column.len() != output_dim {
                return Err(OptimError::ComputationError(
                    "Jacobian columns have inconsistent lengths".to_string(),
                ));
            }
            for (j, &val) in column.iter().enumerate() {
                jacobian[[j, i]] = val;
            }
        }

        Ok(jacobian)
    }

    fn binary_op(&mut self, optype: ForwardOpType, lhs: usize, rhs: usize) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype,
            inputs: vec![lhs, rhs],
            metadata: ForwardOpMetadata {
                shape: Vec::new(), // Inferred from the operands during the forward pass
                data: ForwardOpData::None,
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    fn unary_op(&mut self, optype: ForwardOpType, input: usize) -> Result<usize> {
        let output_id = self.tape.len();

        let op = ForwardOperation {
            optype,
            inputs: vec![input],
            metadata: ForwardOpMetadata {
                shape: Vec::new(), // Inferred from the operand during the forward pass
                data: ForwardOpData::None,
            },
        };

        self.tape.push(op);
        Ok(output_id)
    }

    fn compute_forward_operation(
        &self,
        op: &ForwardOperation<T>,
        values: &[VectorDual<T>],
    ) -> Result<VectorDual<T>> {
        // Checked operand access: an operand index must exist on the tape and
        // must already have been evaluated (guaranteed by tape ordering).
        let fetch = |slot: usize| -> Result<&VectorDual<T>> {
            let idx = *op.inputs.get(slot).ok_or_else(|| {
                OptimError::ComputationError(format!("{:?} is missing operand {slot}", op.optype))
            })?;
            values.get(idx).ok_or_else(|| {
                OptimError::ComputationError(format!("operand {idx} has not been evaluated"))
            })
        };
        let require_same_len = |lhs: &VectorDual<T>, rhs: &VectorDual<T>| -> Result<()> {
            if lhs.value.len() != rhs.value.len() {
                return Err(OptimError::InvalidConfig(format!(
                    "{:?} requires equal operand lengths, got {} and {}",
                    op.optype,
                    lhs.value.len(),
                    rhs.value.len()
                )));
            }
            Ok(())
        };

        match op.optype {
            ForwardOpType::Add => {
                let lhs = fetch(0)?;
                let rhs = fetch(1)?;
                require_same_len(lhs, rhs)?;
                let value = &lhs.value + &rhs.value;
                let tangent = &lhs.tangent + &rhs.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Subtract => {
                let lhs = fetch(0)?;
                let rhs = fetch(1)?;
                require_same_len(lhs, rhs)?;
                let value = &lhs.value - &rhs.value;
                let tangent = &lhs.tangent - &rhs.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Multiply => {
                let lhs = fetch(0)?;
                let rhs = fetch(1)?;
                require_same_len(lhs, rhs)?;

                // Element-wise multiplication: (u*v)' = u'*v + u*v'
                let value = &lhs.value * &rhs.value;
                let tangent = &lhs.tangent * &rhs.value + &lhs.value * &rhs.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Divide => {
                let lhs = fetch(0)?;
                let rhs = fetch(1)?;
                require_same_len(lhs, rhs)?;

                // Division rule: (u/v)' = (u'*v - u*v') / v^2
                let value = &lhs.value / &rhs.value;
                let numerator = &lhs.tangent * &rhs.value - &lhs.value * &rhs.tangent;
                let denominator = &rhs.value * &rhs.value;
                let tangent = numerator / denominator;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Power => {
                let base = fetch(0)?;
                if let ForwardOpData::PowerExponent(exp) = op.metadata.data {
                    // Power rule: (u^n)' = n * u^(n-1) * u'
                    let value = base.value.mapv(|x| x.powf(exp));
                    let derivative = base.value.mapv(|x| exp * x.powf(exp - T::one()));
                    let tangent = derivative * &base.tangent;
                    Ok(VectorDual::new(value, tangent))
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid power operation".to_string(),
                    ))
                }
            }
            ForwardOpType::Exp => {
                let input = fetch(0)?;
                // (e^u)' = e^u * u'
                let value = input.value.mapv(|x| x.exp());
                let tangent = &value * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Log => {
                let input = fetch(0)?;
                // (ln(u))' = u' / u
                let value = input.value.mapv(|x| x.ln());
                let tangent = &input.tangent / &input.value;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Sin => {
                let input = fetch(0)?;
                // (sin(u))' = cos(u) * u'
                let value = input.value.mapv(|x| x.sin());
                let derivative = input.value.mapv(|x| x.cos());
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Cos => {
                let input = fetch(0)?;
                // (cos(u))' = -sin(u) * u'
                let value = input.value.mapv(|x| x.cos());
                let derivative = input.value.mapv(|x| -x.sin());
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Tanh => {
                let input = fetch(0)?;
                // (tanh(u))' = sech^2(u) * u' = (1 - tanh^2(u)) * u'
                let value = input.value.mapv(|x| x.tanh());
                let derivative = value.mapv(|y| T::one() - y * y);
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Sigmoid => {
                let input = fetch(0)?;
                // (sigmoid(u))' = sigmoid(u) * (1 - sigmoid(u)) * u'
                let value = input.value.mapv(|x| T::one() / (T::one() + (-x).exp()));
                let derivative = value.mapv(|y| y * (T::one() - y));
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::ReLU => {
                let input = fetch(0)?;
                // (ReLU(u))' = u' if u > 0, else 0
                let value = input
                    .value
                    .mapv(|x| if x > T::zero() { x } else { T::zero() });
                let derivative = input
                    .value
                    .mapv(|x| if x > T::zero() { T::one() } else { T::zero() });
                let tangent = derivative * &input.tangent;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Dot => {
                let lhs = fetch(0)?;
                let rhs = fetch(1)?;
                require_same_len(lhs, rhs)?;
                // (u·v)' = u'·v + u·v'
                let value = Array1::from_elem(1, lhs.value.dot(&rhs.value));
                let tangent =
                    Array1::from_elem(1, lhs.tangent.dot(&rhs.value) + lhs.value.dot(&rhs.tangent));
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Sum => {
                let input = fetch(0)?;
                // Sum derivative is sum of input derivatives
                let value = Array1::from_elem(1, input.value.sum());
                let tangent = Array1::from_elem(1, input.tangent.sum());
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Mean => {
                let input = fetch(0)?;
                // Mean derivative is mean of input derivatives
                let n =
                    scirs2_core::numeric::NumCast::from(input.value.len()).unwrap_or_else(T::one);
                let value = Array1::from_elem(1, input.value.sum() / n);
                let tangent = Array1::from_elem(1, input.tangent.sum() / n);
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Norm => {
                let input = fetch(0)?;
                // ||u||' = (u·u')/ ||u||
                let norm = input.value.iter().map(|&x| x * x).sum::<T>().sqrt();
                let value = Array1::from_elem(1, norm);
                let tangent = if norm > T::zero() {
                    let dot_product = input.value.dot(&input.tangent);
                    Array1::from_elem(1, dot_product / norm)
                } else {
                    Array1::zeros(1)
                };
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::MatMul => {
                let lhs = fetch(0)?;
                let rhs = fetch(1)?;
                let (m, k, n) = match op.metadata.data {
                    ForwardOpData::MatMulDims { m, k, n } => (m, k, n),
                    _ => {
                        return Err(OptimError::InvalidConfig(
                            "matmul node is missing its dimensions".to_string(),
                        ))
                    }
                };
                if lhs.value.len() != m * k {
                    return Err(OptimError::InvalidConfig(format!(
                        "matmul left operand has {} elements, expected {}",
                        lhs.value.len(),
                        m * k
                    )));
                }
                if rhs.value.len() != k * n {
                    return Err(OptimError::InvalidConfig(format!(
                        "matmul right operand has {} elements, expected {}",
                        rhs.value.len(),
                        k * n
                    )));
                }
                // (A · B)' = A' · B + A · B'
                let value = matmul_flat(&lhs.value, m, k, &rhs.value, n);
                let left_term = matmul_flat(&lhs.tangent, m, k, &rhs.value, n);
                let right_term = matmul_flat(&lhs.value, m, k, &rhs.tangent, n);
                let tangent = left_term + right_term;
                Ok(VectorDual::new(value, tangent))
            }
            ForwardOpType::Variable | ForwardOpType::Constant => Err(OptimError::InvalidConfig(
                "leaf nodes are seeded directly and must not be recomputed".to_string(),
            )),
        }
    }

    /// Get computation graph statistics
    pub fn get_graph_stats(&self) -> ForwardModeStats {
        ForwardModeStats {
            num_operations: self.tape.len(),
            num_variables: self.variables.len(),
            memory_usage_estimate: self.estimate_memory_usage(),
            max_depth: self.compute_max_depth(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        self.tape.len() * std::mem::size_of::<ForwardOperation<T>>()
    }

    fn compute_max_depth(&self) -> usize {
        // Simplified depth computation
        self.tape.len()
    }
}

/// Row-major matrix product of a flat `m x k` buffer with a flat `k x n` buffer.
fn matmul_flat<T: Float + Clone>(
    lhs: &Array1<T>,
    m: usize,
    k: usize,
    rhs: &Array1<T>,
    n: usize,
) -> Array1<T> {
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

/// Forward-mode AD statistics
#[derive(Debug, Clone)]
pub struct ForwardModeStats {
    pub num_operations: usize,
    pub num_variables: usize,
    pub memory_usage_estimate: usize,
    pub max_depth: usize,
}

// Implement arithmetic operations for dual numbers
impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> std::ops::Add for DualNumber<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> std::ops::Sub for DualNumber<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> std::ops::Mul for DualNumber<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> std::ops::Div for DualNumber<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self {
            value: self.value / rhs.value,
            tangent: (self.tangent * rhs.value - self.value * rhs.tangent)
                / (rhs.value * rhs.value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_number_creation() {
        let dual = DualNumber::new(2.0, 1.0);
        assert_eq!(dual.value, 2.0);
        assert_eq!(dual.tangent, 1.0);
    }

    #[test]
    fn test_dual_number_arithmetic() {
        let x = DualNumber::new(3.0, 1.0);
        let y = DualNumber::new(2.0, 0.0);

        let sum = x.clone() + y.clone();
        assert_eq!(sum.value, 5.0);
        assert_eq!(sum.tangent, 1.0);

        let product = x * y;
        assert_eq!(product.value, 6.0);
        assert_eq!(product.tangent, 2.0);
    }

    #[test]
    fn test_forward_mode_engine() {
        let mut engine = ForwardModeEngine::<f64>::new();

        let x_val = Array1::from_vec(vec![2.0]);
        let x_id = engine.create_variable("x", x_val.clone());

        let y_val = Array1::from_vec(vec![3.0]);
        let y_id = engine.create_variable("y", y_val.clone());

        let sum_id = engine.add(x_id, y_id).expect("add should succeed");

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x_val);
        inputs.insert("y".to_string(), y_val);

        let direction = Array1::from_vec(vec![1.0, 0.0]);
        let results = engine
            .forward_pass(&inputs, &direction)
            .expect("forward_pass should succeed");

        assert!(results.len() > sum_id);
        assert_eq!(results[sum_id].value[0], 5.0);
    }

    #[test]
    fn test_vector_dual_operations() {
        let value1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let tangent1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let dual1 = VectorDual::new(value1, tangent1);

        assert_eq!(dual1.value.len(), 3);
        assert_eq!(dual1.tangent.len(), 3);
        assert_eq!(dual1.tangent[0], 1.0);
    }

    /// Smoke test: forward-mode AD computes the correct derivative.
    ///
    /// For f(x) = x^2, f'(x) = 2x. Propagating a unit tangent through the
    /// `power` operation at x = 3 must yield the tangent 2 * 3 = 6.
    #[test]
    fn test_forward_mode_derivative_of_square() {
        // Direct DualNumber check: f(x) = x * x at x = 3 -> tangent 6.
        let x = DualNumber::variable(3.0_f64);
        let y = x.clone() * x;
        approx::assert_abs_diff_eq!(y.value, 9.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(y.tangent, 6.0, epsilon = 1e-10);

        // Engine-level JVP check via the `power` op builder.
        let mut engine = ForwardModeEngine::<f64>::new();
        let x_val = Array1::from_vec(vec![3.0]);
        let x_id = engine.create_variable("x", x_val.clone());
        let sq_id = engine.power(x_id, 2.0).expect("power op");

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x_val);

        // Seed direction selects variable 0 (x) with unit tangent.
        let direction = Array1::from_vec(vec![1.0]);
        let jvp = engine
            .jacobian_vector_product(&inputs, sq_id, &direction)
            .expect("jvp");
        approx::assert_abs_diff_eq!(jvp[0], 6.0, epsilon = 1e-10);
    }

    /// F41: matrix multiplication used to fall through to "unsupported
    /// operation". The JVP obeys the product rule (A·B)' = A'·B + A·B'.
    #[test]
    fn test_forward_mode_matmul_jvp() {
        let mut engine = ForwardModeEngine::<f64>::new();
        // A is 2x3, B is 3x2 -> C is 2x2.
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Array1::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let a_id = engine.create_variable("a", a.clone());
        let b_id = engine.create_variable("b", b.clone());
        let c_id = engine.matmul(a_id, b_id, (2, 3, 2)).expect("matmul");

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), a.clone());
        inputs.insert("b".to_string(), b.clone());

        assert_eq!(engine.input_dimension(&inputs).expect("dim"), 12);
        assert_eq!(
            engine.declared_shape(c_id).expect("shape"),
            Some(&[2, 2][..])
        );

        // Seed only a[0] (the first coordinate of the concatenated space).
        let mut direction = Array1::zeros(12);
        direction[0] = 1.0;
        let results = engine.forward_pass(&inputs, &direction).expect("forward");
        let c = &results[c_id];

        // C = A·B computed by hand.
        approx::assert_abs_diff_eq!(
            c.value[0],
            1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,
            epsilon = 1e-10
        );
        approx::assert_abs_diff_eq!(
            c.value[3],
            4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0,
            epsilon = 1e-10
        );
        // dC/da[0,0] = row 0 of B: [7, 8] into C[0,0], C[0,1]; zero elsewhere.
        approx::assert_abs_diff_eq!(c.tangent[0], 7.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(c.tangent[1], 8.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(c.tangent[2], 0.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(c.tangent[3], 0.0, epsilon = 1e-10);
    }

    /// F41: the Jacobian must be `(output_dim, input_dim)`.
    #[test]
    fn test_jacobian_matrix_dimensions_and_values() {
        let mut engine = ForwardModeEngine::<f64>::new();
        let x = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let x_id = engine.create_variable("x", x.clone());
        let y_id = engine.exp(x_id).expect("exp");

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x.clone());

        let jac = engine.jacobian_matrix(&inputs, y_id, 3).expect("jacobian");
        assert_eq!(jac.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { x[i].exp() } else { 0.0 };
                approx::assert_abs_diff_eq!(jac[[i, j]], expected, epsilon = 1e-9);
            }
        }

        // A mismatched requested size is an error, not a silently wrong matrix.
        assert!(engine.jacobian_matrix(&inputs, y_id, 5).is_err());
    }

    /// F41: seed ordinals follow variable-creation order, not tape indices, so
    /// intermediate operations between two variables do not shift the seeds.
    #[test]
    fn test_seed_ordinals_are_per_variable() {
        let mut engine = ForwardModeEngine::<f64>::new();
        let x = Array1::from_vec(vec![1.0]);
        let x_id = engine.create_variable("x", x.clone());
        // Several intermediate nodes push the tape index of `y` far past 1.
        let t1 = engine.exp(x_id).expect("exp");
        let _t2 = engine.sin(t1).expect("sin");
        let y = Array1::from_vec(vec![5.0]);
        let y_id = engine.create_variable("y", y.clone());
        let prod = engine.multiply(x_id, y_id).expect("multiply");

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);
        inputs.insert("y".to_string(), y);

        assert_eq!(engine.variable_order(), &["x".to_string(), "y".to_string()]);

        // Seed the second variable (ordinal 1): d(xy)/dy = x = 1.
        let direction = Array1::from_vec(vec![0.0, 1.0]);
        let jvp = engine
            .jacobian_vector_product(&inputs, prod, &direction)
            .expect("jvp");
        approx::assert_abs_diff_eq!(jvp[0], 1.0, epsilon = 1e-12);

        // Seed the first variable (ordinal 0): d(xy)/dx = y = 5.
        let direction = Array1::from_vec(vec![1.0, 0.0]);
        let jvp = engine
            .jacobian_vector_product(&inputs, prod, &direction)
            .expect("jvp");
        approx::assert_abs_diff_eq!(jvp[0], 5.0, epsilon = 1e-12);
    }

    /// A seed direction whose length disagrees with the input space is an error.
    #[test]
    fn test_seed_length_validation() {
        let mut engine = ForwardModeEngine::<f64>::new();
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let x_id = engine.create_variable("x", x.clone());
        let y_id = engine.tanh(x_id).expect("tanh");

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);

        let bad = Array1::from_vec(vec![1.0]);
        assert!(engine.jacobian_vector_product(&inputs, y_id, &bad).is_err());
    }
}
