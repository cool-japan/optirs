use std::fmt::Debug;
// Computation graph capture for XLA compilation
//
// This module handles the capture and construction of computation graphs
// from high-level operations, including graph validation and optimization
// preparation.

use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use super::super::{TPUConfig, XLAOptimizationLevel};
use crate::error::{OptimError, Result};

/// Computation graph builder
#[derive(Debug)]
pub struct ComputationGraphBuilder<T: Float + Debug + Send + Sync + 'static> {
    /// Next operation ID
    next_op_id: usize,

    /// Next computation ID
    next_computation_id: u64,

    /// Operation registry
    operation_registry: HashMap<String, OperationDefinition>,

    /// Graph validation rules
    validation_rules: Vec<ValidationRule>,

    /// Performance hints
    performance_hints: HashMap<String, PerformanceHint>,

    /// Phantom data for type parameter
    pub _phantom: std::marker::PhantomData<T>,
}

/// XLA computation representation
#[derive(Debug, Clone)]
pub struct XLAComputation<T: Float + Debug + Send + Sync + 'static> {
    /// Computation identifier
    pub id: ComputationId,

    /// Operations in topological order
    pub operations: Vec<XLAOperation<T>>,

    /// Input specifications
    pub inputs: Vec<InputSpecification<T>>,

    /// Output specifications  
    pub outputs: Vec<OutputSpecification<T>>,

    /// Computation metadata
    pub metadata: ComputationMetadata,

    /// Operand graph
    pub operands: HashMap<OperandId, Operand<T>>,

    /// Operation dependencies
    pub dependencies: HashMap<OperationId, Vec<OperationId>>,

    /// Monotonic allocator for [`OperandId`]s.
    ///
    /// This must never be derived from `operands.len()`: optimization passes
    /// such as dead-code elimination remove operands, and reusing a freed index
    /// would silently alias a live operand.
    pub next_operand_id: usize,
}

/// Computation identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputationId(pub u64);

/// Operation identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub usize);

/// XLA operation
#[derive(Debug, Clone)]
pub struct XLAOperation<T: Float + Debug + Send + Sync + 'static> {
    /// Operation ID
    pub id: OperationId,

    /// Operation type
    pub op_type: OperationType,

    /// Input operands
    pub inputs: Vec<OperandId>,

    /// Output operand
    pub output: OperandId,

    /// Operation attributes
    pub attributes: OperationAttributes,

    /// Performance characteristics
    pub performance: OperationPerformanceCharacteristics,

    /// Memory requirements
    pub memory_requirements: OperationMemoryRequirements,

    /// Source location (for debugging)
    pub source_location: Option<SourceLocation>,

    /// Phantom data for type parameter
    pub _phantom: std::marker::PhantomData<T>,
}

/// Materialized payload of an [`OperationType::Constant`] node.
///
/// The payload is stored in `f64` regardless of the graph's element type so
/// that `OperationType` stays non-generic (it is embedded in `XLAOperation<T>`
/// but carries no `T` of its own). Conversion to/from the graph element type
/// happens at the boundary via `NumCast`.
///
/// Unlike the previous `Box<dyn Any>` representation this survives `Clone`,
/// compares by value, and hashes consistently with `PartialEq`, which is what
/// makes constant folding and common-subexpression elimination work at all.
#[derive(Debug, Clone, Default)]
pub struct ConstantValue {
    /// Flattened row-major element data.
    pub data: Vec<f64>,

    /// Logical dimensions; empty means a rank-0 scalar.
    pub dims: Vec<usize>,
}

impl ConstantValue {
    /// Create a rank-0 scalar constant.
    pub fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            dims: Vec::new(),
        }
    }

    /// Create a constant from flattened data and dimensions.
    ///
    /// Returns `None` when `data.len()` does not match the product of `dims`.
    pub fn new(data: Vec<f64>, dims: Vec<usize>) -> Option<Self> {
        let expected: usize = dims.iter().product();
        if data.len() == expected {
            Some(Self { data, dims })
        } else {
            None
        }
    }

    /// Number of elements held by this constant.
    pub fn element_count(&self) -> usize {
        self.data.len()
    }

    /// Interpret the constant as a scalar, if it holds exactly one element.
    pub fn as_scalar(&self) -> Option<f64> {
        if self.data.len() == 1 {
            self.data.first().copied()
        } else {
            None
        }
    }

    /// Tensor shape described by this constant.
    pub fn tensor_shape(&self) -> TensorShape {
        TensorShape {
            dimensions: self.dims.clone(),
            dynamic_dimensions: vec![false; self.dims.len()],
            element_count: self.data.len(),
            tuple_shapes: Vec::new(),
        }
    }

    /// True when every element equals `value` bit-for-bit after conversion.
    pub fn is_uniform(&self, value: f64) -> bool {
        !self.data.is_empty() && self.data.iter().all(|&v| v == value)
    }
}

// `Vec<f64>` has no `Eq`/`Hash`, so both are implemented over the raw bit
// patterns. Doing it this way keeps `eq` and `hash` in agreement, which the
// CSE pass relies on when it keys operations by their expression hash.
impl PartialEq for ConstantValue {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims
            && self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for ConstantValue {}

impl std::hash::Hash for ConstantValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dims.hash(state);
        for value in &self.data {
            value.to_bits().hash(state);
        }
    }
}

/// Types of XLA operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    // Elementwise operations
    Add,
    Multiply,
    Subtract,
    Divide,
    Maximum,
    Minimum,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Square,
    Sign,
    Negate,
    Sin,
    Cos,
    Tanh,
    Ceil,
    Floor,
    Round,

    // Logical operations
    Not,
    And,
    Or,
    Xor,

    // Comparison operations
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // Array operations
    Reshape,
    Transpose,
    Slice,
    DynamicSlice,
    Pad,
    Reverse,
    Broadcast,
    Concatenate,
    Gather,
    Scatter,

    // Reduction operations
    Reduce(ReduceOperation),
    ReduceWindow,
    AllReduce(AllReduceOperation),

    // Linear algebra
    Dot,
    DotGeneral,
    MatMul,
    Convolution(ConvolutionConfig),

    // Control flow
    Conditional,
    While,
    Call,

    // Communication operations
    AllGather,
    AllToAll,
    CollectivePermute,
    ReduceScatter,

    // Deep learning operations
    BatchNorm,
    Dropout,

    // Memory operations
    Copy,
    Tuple,
    GetTupleElement,

    // Special operations
    Constant(ConstantValue),
    Parameter,
    Iota,

    // Custom operations
    Custom(CustomOperation),
}

/// Reduce operation configuration
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct ReduceOperation {
    /// Reduction function
    pub function: ReductionFunction,

    /// Dimensions to reduce over
    pub dimensions: Vec<usize>,

    /// Initial value
    pub init_value: Option<String>,
}

/// Reduction functions
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum ReductionFunction {
    Add,
    Multiply,
    Max,
    Min,
    And,
    Or,
    Xor,
}

/// All-reduce operation configuration
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct AllReduceOperation {
    /// Reduction function
    pub function: ReductionFunction,

    /// Replica groups
    pub replica_groups: Vec<Vec<usize>>,
}

/// Convolution configuration
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct ConvolutionConfig {
    /// Window strides
    pub strides: Vec<usize>,

    /// Padding configuration
    pub padding: PaddingConfig,

    /// Dilation factors
    pub dilation: Vec<usize>,

    /// Feature group count
    pub feature_group_count: usize,

    /// Batch group count
    pub batch_group_count: usize,
}

/// Padding configuration
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum PaddingConfig {
    Valid,
    Same,
    Explicit(Vec<(usize, usize)>),
}

/// Custom operation definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomOperation {
    /// Operation name
    pub name: String,

    /// Custom attributes
    pub custom_attributes: HashMap<String, String>,

    /// Backend configuration
    pub backend_config: Option<String>,
}

impl std::hash::Hash for CustomOperation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        // Hash the HashMap as a sorted list of pairs
        let mut attrs: Vec<_> = self.custom_attributes.iter().collect();
        attrs.sort_by_key(|&(k, _)| k);
        for (k, v) in attrs {
            k.hash(state);
            v.hash(state);
        }
        self.backend_config.hash(state);
    }
}

/// Operand in the computation
#[derive(Debug, Clone)]
pub struct Operand<T: Float + Debug + Send + Sync + 'static> {
    /// Operand ID
    pub id: OperandId,

    /// Tensor shape
    pub shape: TensorShape,

    /// Data layout
    pub layout: Layout,

    /// Data type
    pub dtype: DataType,

    /// Operand metadata
    pub metadata: OperandMetadata,

    /// Phantom data for type parameter
    pub _phantom: std::marker::PhantomData<T>,
}

/// Operand identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperandId(pub usize);

/// Tensor shape information
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorShape {
    /// Dimensions
    pub dimensions: Vec<usize>,

    /// Dynamic dimension flags
    pub dynamic_dimensions: Vec<bool>,

    /// Element count
    pub element_count: usize,

    /// Tuple shape (for nested structures)
    pub tuple_shapes: Vec<TensorShape>,
}

/// Data layout specification
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    /// Dimension order (minor to major)
    pub minor_to_major: Vec<usize>,

    /// Tiling information
    pub tiles: Vec<Tile>,

    /// Memory space
    pub memory_space: MemorySpace,
}

/// Tiling specification
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    /// Tile dimensions
    pub dimensions: Vec<usize>,

    /// Tile stride
    pub stride: Vec<usize>,
}

/// Memory space types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemorySpace {
    Default,
    Host,
    Device,
    Pinned,
}

/// Data types supported by XLA
#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub enum DataType {
    F16,
    F32,
    F64,
    BF16,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    Pred,
    C64,
    C128,
}

/// Operation attributes
#[derive(Debug, Clone, Default)]
pub struct OperationAttributes {
    /// Generic attributes
    pub attributes: HashMap<String, AttributeValue>,

    /// Sharding specification
    pub sharding: Option<ShardingSpec>,

    /// Fusion hint
    pub fusion_hint: Option<String>,

    /// Performance hint
    pub performance_hint: Option<PerformanceHint>,
}

/// Attribute value types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
}

/// Sharding specification
#[derive(Debug, Clone)]
pub struct ShardingSpec {
    /// Tile assignment
    pub tile_assignment: Vec<Vec<usize>>,

    /// Replicated dimensions
    pub replicated_dims: Vec<usize>,

    /// Manual sharding
    pub manual: bool,
}

/// Performance hint for operations
#[derive(Debug, Clone)]
pub struct PerformanceHint {
    /// Estimated cost
    pub estimated_cost: f64,

    /// Memory intensity
    pub memory_intensity: f64,

    /// Compute intensity
    pub compute_intensity: f64,

    /// Parallelization hint
    pub parallelization: ParallelizationHint,
}

/// Parallelization hints
#[derive(Debug, Clone)]
pub enum ParallelizationHint {
    Sequential,
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Custom(String),
}

/// Source location for debugging
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File name
    pub file: String,

    /// Line number
    pub line: u32,

    /// Column number
    pub column: u32,

    /// Function name
    pub function: String,
}

/// Operation performance characteristics
#[derive(Debug, Clone, Default)]
pub struct OperationPerformanceCharacteristics {
    /// Estimated execution time (microseconds)
    pub execution_time_us: u64,

    /// FLOP count
    pub flop_count: u64,

    /// Memory accesses
    pub memory_accesses: u64,

    /// Communication volume (bytes)
    pub communication_volume: u64,

    /// Compute utilization
    pub compute_utilization: f64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
}

/// Operation memory requirements
#[derive(Debug, Clone, Default)]
pub struct OperationMemoryRequirements {
    /// Input memory (bytes)
    pub input_memory: usize,

    /// Output memory (bytes)
    pub output_memory: usize,

    /// Temporary memory (bytes)
    pub temp_memory: usize,

    /// Peak memory (bytes)
    pub peak_memory: usize,

    /// Memory alignment requirements
    pub alignment_requirements: Vec<usize>,
}

/// Input specification
#[derive(Debug, Clone)]
pub struct InputSpecification<T: Float + Debug + Send + Sync + 'static> {
    /// Input index
    pub index: usize,

    /// Parameter name
    pub name: String,

    /// Shape specification
    pub shape: TensorShape,

    /// Data type
    pub dtype: DataType,

    /// Layout hint
    pub layout_hint: Option<Layout>,

    /// Phantom data for type parameter
    pub _phantom: std::marker::PhantomData<T>,
}

/// Output specification
#[derive(Debug, Clone)]
pub struct OutputSpecification<T: Float + Debug + Send + Sync + 'static> {
    /// Output index
    pub index: usize,

    /// Operand that carries this output value.
    ///
    /// Dead-code elimination roots its liveness walk here; without an explicit
    /// operand link there is no way to tell which operation produces an output.
    pub operand: OperandId,

    /// Shape specification
    pub shape: TensorShape,

    /// Data type
    pub dtype: DataType,

    /// Layout requirement
    pub layout: Layout,

    /// Phantom data for type parameter
    pub _phantom: std::marker::PhantomData<T>,
}

/// Computation metadata
#[derive(Debug, Clone, Default)]
pub struct ComputationMetadata {
    /// Computation name
    pub name: String,

    /// Creation timestamp
    pub created_at: Option<Instant>,

    /// Source information
    pub source_info: HashMap<String, String>,

    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,

    /// Performance hints
    pub performance_hints: Vec<PerformanceHint>,

    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Opportunity type
    pub opportunity_type: OpportunityType,

    /// Affected operations
    pub affected_operations: Vec<OperationId>,

    /// Estimated benefit
    pub estimated_benefit: f64,

    /// Implementation complexity
    pub complexity: ComplexityLevel,

    /// Description
    pub description: String,
}

/// Types of optimization opportunities
#[derive(Debug, Clone)]
pub enum OpportunityType {
    Fusion,
    MemoryLayout,
    Parallelization,
    Sparsity,
    Quantization,
    Scheduling,
    Custom(String),
}

/// Complexity levels
#[derive(Debug, Clone, Copy)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Resource requirements
#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    /// Compute requirements (FLOPS)
    pub compute_flops: u64,

    /// Memory requirements (bytes)
    pub memory_bytes: usize,

    /// Communication requirements (bytes)
    pub communication_bytes: usize,

    /// Execution time estimate (microseconds)
    pub execution_time_us: u64,
}

/// Operand metadata
#[derive(Debug, Clone, Default)]
pub struct OperandMetadata {
    /// Producer operation
    pub producer: Option<OperationId>,

    /// Consumer operations
    pub consumers: Vec<OperationId>,

    /// Usage hints
    pub usage_hint: UsageHint,

    /// Layout hints
    pub layout_hints: Vec<LayoutHint>,
}

/// Usage hints for operands
#[derive(Debug, Clone)]
pub struct UsageHint {
    /// Access pattern
    pub access_pattern: AccessPattern,

    /// Reuse factor
    pub reuse_factor: f64,

    /// Lifetime
    pub lifetime: OperandLifetime,
}

/// Access patterns
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided,
    Broadcast,
    Reduction,
}

/// Operand lifetime
#[derive(Debug, Clone)]
pub enum OperandLifetime {
    Temporary,
    Persistent,
    Parameter,
    Output,
}

/// Layout hints
#[derive(Debug, Clone)]
pub struct LayoutHint {
    /// Preferred layout
    pub preferred_layout: Layout,

    /// Priority
    pub priority: f64,

    /// Reason
    pub reason: String,
}

/// Operation definition for registration
#[derive(Debug, Clone)]
pub struct OperationDefinition {
    /// Operation name
    pub name: String,

    /// Input types
    pub input_types: Vec<DataType>,

    /// Output type
    pub output_type: DataType,

    /// Shape function
    pub shape_function: String,

    /// Performance model
    pub performance_model: String,
}

/// Graph validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Validation function
    pub validator: String,
}

impl<T: Float + Debug + Send + Sync + 'static> XLAComputation<T> {
    /// Allocate a fresh, never-reused [`OperandId`].
    pub fn allocate_operand_id(&mut self) -> OperandId {
        let id = OperandId(self.next_operand_id);
        self.next_operand_id += 1;
        id
    }

    /// Smallest [`OperationId`] not currently in use.
    ///
    /// Optimization passes that synthesize operations must use this rather than
    /// `operations.len()`, which collides with surviving ids after a removal and
    /// silently clobbers the dependency map.
    pub fn next_free_operation_id(&self) -> OperationId {
        let max = self
            .operations
            .iter()
            .map(|op| op.id.0)
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(0);
        OperationId(max)
    }

    /// The operation producing `operand`, if any.
    pub fn producer_of(&self, operand: OperandId) -> Option<&XLAOperation<T>> {
        self.operations.iter().find(|op| op.output == operand)
    }

    /// The operand an operation writes its result to.
    pub fn operation_output(&self, op_id: OperationId) -> Option<OperandId> {
        self.operations
            .iter()
            .find(|op| op.id == op_id)
            .map(|op| op.output)
    }

    /// Rewrite every use of operand `from` to instead read operand `to`.
    ///
    /// Covers operation inputs and declared computation outputs. Callers are
    /// expected to follow up with [`Self::rebuild_dependencies`].
    pub fn replace_operand_uses(&mut self, from: OperandId, to: OperandId) {
        if from == to {
            return;
        }

        for operation in &mut self.operations {
            for input in &mut operation.inputs {
                if *input == from {
                    *input = to;
                }
            }
        }

        for output in &mut self.outputs {
            if output.operand == from {
                output.operand = to;
            }
        }
    }

    /// Recompute producer/consumer metadata and the dependency map from the
    /// current operation list.
    ///
    /// Passes that add or remove operations invalidate this derived state;
    /// recomputing wholesale is cheaper to reason about than patching it
    /// incrementally and cannot drift out of sync.
    pub fn rebuild_dependencies(&mut self) {
        for operand in self.operands.values_mut() {
            operand.metadata.producer = None;
            operand.metadata.consumers.clear();
        }

        for operation in &self.operations {
            if let Some(operand) = self.operands.get_mut(&operation.output) {
                operand.metadata.producer = Some(operation.id);
            }
        }

        for operation in &self.operations {
            for input_id in &operation.inputs {
                if let Some(operand) = self.operands.get_mut(input_id) {
                    if !operand.metadata.consumers.contains(&operation.id) {
                        operand.metadata.consumers.push(operation.id);
                    }
                }
            }
        }

        let mut dependencies: HashMap<OperationId, Vec<OperationId>> = HashMap::new();
        for operation in &self.operations {
            let mut deps: Vec<OperationId> = Vec::new();
            for input_id in &operation.inputs {
                if let Some(producer) = self
                    .operands
                    .get(input_id)
                    .and_then(|operand| operand.metadata.producer)
                {
                    if !deps.contains(&producer) {
                        deps.push(producer);
                    }
                }
            }
            dependencies.insert(operation.id, deps);
        }

        self.dependencies = dependencies;
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> Default
    for ComputationGraphBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync>
    ComputationGraphBuilder<T>
{
    /// Create new computation graph builder
    pub fn new() -> Self {
        Self {
            next_op_id: 0,
            next_computation_id: 0,
            operation_registry: HashMap::new(),
            validation_rules: Vec::new(),
            performance_hints: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create new computation
    pub fn create_computation(&mut self, name: &str) -> XLAComputation<T> {
        let id = ComputationId(self.next_computation_id);
        self.next_computation_id += 1;

        XLAComputation {
            id,
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: ComputationMetadata {
                name: name.to_string(),
                created_at: Some(Instant::now()),
                ..Default::default()
            },
            operands: HashMap::new(),
            dependencies: HashMap::new(),
            next_operand_id: 0,
        }
    }

    /// Add operation to computation
    ///
    /// Besides appending the operation this wires up the operand graph in both
    /// directions: the freshly created output operand records this operation as
    /// its `producer`, and every input operand gains this operation as a
    /// `consumer`. Topological sorting, scheduling, cycle detection and
    /// dead-code elimination all read that metadata, so skipping it leaves the
    /// dependency map permanently empty.
    pub fn add_operation(
        &mut self,
        computation: &mut XLAComputation<T>,
        op_type: OperationType,
        inputs: Vec<OperandId>,
        output_shape: TensorShape,
    ) -> Result<OperationId> {
        // Reject dangling operands up front: silently producing an operation
        // with unknown inputs is what made dependency tracking unverifiable.
        for input_id in &inputs {
            if !computation.operands.contains_key(input_id) {
                return Err(OptimError::from(format!(
                    "Operation input operand {:?} does not exist in computation '{}'",
                    input_id, computation.metadata.name
                )));
            }
        }

        let op_id = OperationId(self.next_op_id);
        self.next_op_id += 1;

        // Create output operand using a monotonic id (never `operands.len()`,
        // which aliases after DCE removes entries).
        let output_operand_id = OperandId(computation.next_operand_id);
        computation.next_operand_id += 1;

        let output_operand = Operand {
            id: output_operand_id,
            shape: output_shape,
            layout: Layout::default(),
            dtype: DataType::F32, // Default type
            metadata: OperandMetadata {
                producer: Some(op_id),
                ..Default::default()
            },
            _phantom: std::marker::PhantomData,
        };

        computation
            .operands
            .insert(output_operand_id, output_operand);

        // Record this operation as a consumer of each input operand.
        for &input_id in &inputs {
            if let Some(operand) = computation.operands.get_mut(&input_id) {
                if !operand.metadata.consumers.contains(&op_id) {
                    operand.metadata.consumers.push(op_id);
                }
            }
        }

        // Update dependencies from the (now populated) producer metadata.
        let mut input_ops: Vec<OperationId> = Vec::new();
        for &operand_id in &inputs {
            if let Some(producer) = computation
                .operands
                .get(&operand_id)
                .and_then(|operand| operand.metadata.producer)
            {
                if !input_ops.contains(&producer) {
                    input_ops.push(producer);
                }
            }
        }

        // Create operation
        let operation = XLAOperation {
            id: op_id,
            op_type,
            inputs,
            output: output_operand_id,
            attributes: OperationAttributes::default(),
            performance: OperationPerformanceCharacteristics::default(),
            memory_requirements: OperationMemoryRequirements::default(),
            source_location: None,
            _phantom: std::marker::PhantomData,
        };

        computation.operations.push(operation);
        computation.dependencies.insert(op_id, input_ops);

        Ok(op_id)
    }

    /// Declare the operands that constitute the computation's results.
    ///
    /// Every downstream pass treats the output set as the liveness root, so a
    /// computation whose outputs were never declared looks entirely dead. Call
    /// this after the graph has been built, or use
    /// [`Self::mark_terminal_operands_as_outputs`] to infer it.
    pub fn set_outputs(
        &self,
        computation: &mut XLAComputation<T>,
        outputs: &[OperandId],
    ) -> Result<()> {
        let mut specs = Vec::with_capacity(outputs.len());

        for (index, &operand_id) in outputs.iter().enumerate() {
            let operand = computation.operands.get(&operand_id).ok_or_else(|| {
                OptimError::from(format!(
                    "Cannot mark unknown operand {:?} as output of computation '{}'",
                    operand_id, computation.metadata.name
                ))
            })?;

            specs.push(OutputSpecification {
                index,
                operand: operand_id,
                shape: operand.shape.clone(),
                dtype: operand.dtype,
                layout: operand.layout.clone(),
                _phantom: std::marker::PhantomData,
            });
        }

        // Outputs live for the whole computation; record that on the operand so
        // lifetime-driven passes (memory planning) do not recycle their buffers.
        for &operand_id in outputs {
            if let Some(operand) = computation.operands.get_mut(&operand_id) {
                operand.metadata.usage_hint.lifetime = OperandLifetime::Output;
            }
        }

        computation.outputs = specs;
        Ok(())
    }

    /// Infer the output set as every operand that no operation consumes.
    ///
    /// Returns the number of outputs discovered. A graph in which every operand
    /// feeds another operation has no terminal operand and yields zero, which
    /// [`Self::validate_computation`] then rejects.
    pub fn mark_terminal_operands_as_outputs(
        &self,
        computation: &mut XLAComputation<T>,
    ) -> Result<usize> {
        let consumed: HashSet<OperandId> = computation
            .operations
            .iter()
            .flat_map(|op| op.inputs.iter().copied())
            .collect();

        // Iterate operations rather than the operand map so the output order is
        // deterministic (HashMap iteration order is not).
        let terminals: Vec<OperandId> = computation
            .operations
            .iter()
            .map(|op| op.output)
            .filter(|operand_id| !consumed.contains(operand_id))
            .collect();

        self.set_outputs(computation, &terminals)?;
        Ok(terminals.len())
    }

    /// Validate computation graph
    pub fn validate_computation(&self, computation: &XLAComputation<T>) -> Result<()> {
        // Check for cycles
        self.check_for_cycles(computation)?;

        // A computation with operations but no declared outputs is malformed:
        // its entire body is unreachable and dead-code elimination would be
        // within its rights to delete everything.
        if !computation.operations.is_empty() && computation.outputs.is_empty() {
            return Err(OptimError::from(format!(
                "Computation '{}' declares no outputs; call set_outputs or \
                 mark_terminal_operands_as_outputs before optimization",
                computation.metadata.name
            )));
        }

        // Every declared output must resolve to an operand that exists.
        for output in &computation.outputs {
            if !computation.operands.contains_key(&output.operand) {
                return Err(OptimError::from(format!(
                    "Computation '{}' output {} references unknown operand {:?}",
                    computation.metadata.name, output.index, output.operand
                )));
            }
        }

        // Check shape compatibility
        self.check_shape_compatibility(computation)?;

        // Check resource requirements
        self.check_resource_requirements(computation)?;

        Ok(())
    }

    /// Check for cycles in computation graph
    fn check_for_cycles(&self, computation: &XLAComputation<T>) -> Result<()> {
        for operation in &computation.operations {
            if let Some(cycle_at) = Self::find_cycle_from(computation, operation.id) {
                return Err(OptimError::from(format!(
                    "Cycle detected in computation graph '{}' involving operation {:?}",
                    computation.metadata.name, cycle_at
                )));
            }
        }

        Ok(())
    }

    /// Iterative depth-first cycle search rooted at `start`.
    ///
    /// Uses an explicit stack rather than recursion so that deep computation
    /// graphs cannot overflow the native stack. Nodes are three-coloured: absent
    /// from `visited` (white), present in `on_stack` (grey), and visited but
    /// popped (black). An edge into a grey node is a back edge, i.e. a cycle.
    fn find_cycle_from(computation: &XLAComputation<T>, start: OperationId) -> Option<OperationId> {
        enum Step {
            Enter(OperationId),
            Leave(OperationId),
        }

        let mut visited: HashSet<OperationId> = HashSet::new();
        let mut on_stack: HashSet<OperationId> = HashSet::new();
        let mut stack: Vec<Step> = vec![Step::Enter(start)];

        while let Some(step) = stack.pop() {
            match step {
                Step::Leave(op_id) => {
                    on_stack.remove(&op_id);
                }
                Step::Enter(op_id) => {
                    if on_stack.contains(&op_id) {
                        return Some(op_id);
                    }
                    if !visited.insert(op_id) {
                        continue;
                    }

                    on_stack.insert(op_id);
                    stack.push(Step::Leave(op_id));

                    if let Some(dependencies) = computation.dependencies.get(&op_id) {
                        for &dep_id in dependencies {
                            if on_stack.contains(&dep_id) {
                                return Some(dep_id);
                            }
                            if !visited.contains(&dep_id) {
                                stack.push(Step::Enter(dep_id));
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Check shape compatibility
    fn check_shape_compatibility(&self, _computation: &XLAComputation<T>) -> Result<()> {
        // Shape compatibility checking logic would go here
        Ok(())
    }

    /// Check resource requirements
    fn check_resource_requirements(&self, _computation: &XLAComputation<T>) -> Result<()> {
        // Resource requirement checking logic would go here
        Ok(())
    }

    /// Get topological ordering of operations
    pub fn get_topological_order(
        &self,
        computation: &XLAComputation<T>,
    ) -> Result<Vec<OperationId>> {
        let mut in_degree = HashMap::new();
        let mut adj_list = HashMap::new();

        // Build adjacency list and compute in-degrees
        for operation in &computation.operations {
            in_degree.insert(operation.id, 0);
            adj_list.insert(operation.id, Vec::new());
        }

        for (op_id, dependencies) in &computation.dependencies {
            // An operation may have been removed by an earlier pass while its
            // dependency entry lingers; skip such stale edges instead of
            // panicking on the missing key.
            if !in_degree.contains_key(op_id) {
                continue;
            }
            for &dep_id in dependencies {
                let Some(neighbors) = adj_list.get_mut(&dep_id) else {
                    continue;
                };
                neighbors.push(*op_id);
                if let Some(degree) = in_degree.get_mut(op_id) {
                    *degree += 1;
                }
            }
        }

        // Topological sort using Kahn's algorithm. Seed the queue in operation
        // order so the result is deterministic (HashMap iteration is not).
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        for operation in &computation.operations {
            if in_degree.get(&operation.id) == Some(&0usize) {
                queue.push_back(operation.id);
            }
        }

        while let Some(op_id) = queue.pop_front() {
            result.push(op_id);

            if let Some(neighbors) = adj_list.get(&op_id) {
                for &neighbor in neighbors.iter() {
                    let Some(degree) = in_degree.get_mut(&neighbor) else {
                        continue;
                    };
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if result.len() != computation.operations.len() {
            return Err(OptimError::from("Graph contains cycles".to_string()));
        }

        Ok(result)
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            minor_to_major: vec![0, 1], // Default 2D layout
            tiles: Vec::new(),
            memory_space: MemorySpace::Default,
        }
    }
}

impl Default for UsageHint {
    fn default() -> Self {
        Self {
            access_pattern: AccessPattern::Sequential,
            reuse_factor: 1.0,
            lifetime: OperandLifetime::Temporary,
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Rank-0 scalar shape.
    pub fn scalar_shape() -> TensorShape {
        TensorShape {
            dimensions: Vec::new(),
            dynamic_dimensions: Vec::new(),
            element_count: 1,
            tuple_shapes: Vec::new(),
        }
    }

    /// Shape with the given dimensions.
    pub fn shape(dims: &[usize]) -> TensorShape {
        TensorShape {
            dimensions: dims.to_vec(),
            dynamic_dimensions: vec![false; dims.len()],
            element_count: dims.iter().product::<usize>().max(1),
            tuple_shapes: Vec::new(),
        }
    }

    /// Add an operation and return the operand carrying its result.
    pub fn add_op<T>(
        builder: &mut ComputationGraphBuilder<T>,
        computation: &mut XLAComputation<T>,
        op_type: OperationType,
        inputs: Vec<OperandId>,
        out_shape: TensorShape,
    ) -> OperandId
    where
        T: Float + Debug + Default + Clone + Send + Sync,
    {
        let op_id = builder
            .add_operation(computation, op_type, inputs, out_shape)
            .expect("test graph construction must succeed");
        computation
            .operation_output(op_id)
            .expect("freshly added operation must have an output operand")
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn test_computation_creation() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let computation = builder.create_computation("test_computation");
        assert_eq!(computation.metadata.name, "test_computation");
    }

    #[test]
    fn test_operation_addition() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut computation = builder.create_computation("test");

        let result = builder.add_operation(
            &mut computation,
            OperationType::Parameter,
            vec![],
            shape(&[10, 10]),
        );

        assert!(result.is_ok());
        assert_eq!(computation.operations.len(), 1);
    }

    /// F3: an operand must know which operation produced it and which
    /// operations consume it. Both were previously left permanently empty.
    #[test]
    fn producer_and_consumers_are_recorded() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("deps");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            scalar_shape(),
        );

        let sum_op = comp
            .producer_of(sum)
            .expect("sum operand must have a producer");
        assert_eq!(
            comp.operands
                .get(&a)
                .map(|operand| operand.metadata.consumers.clone()),
            Some(vec![sum_op.id])
        );
        assert_eq!(
            comp.operands
                .get(&b)
                .map(|operand| operand.metadata.consumers.clone()),
            Some(vec![sum_op.id])
        );

        // The Add depends on both parameter operations.
        let deps = comp
            .dependencies
            .get(&sum_op.id)
            .cloned()
            .unwrap_or_default();
        assert_eq!(deps.len(), 2, "add must depend on both parameters");
    }

    /// Operand ids must never be recycled, even after operands are removed.
    #[test]
    fn operand_ids_are_never_reused() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("ids");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );

        // Simulate an optimization pass dropping an operand.
        comp.operands.remove(&b);

        let c = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );

        assert_ne!(c, a);
        assert_ne!(c, b, "a freed operand id must not be handed out again");
    }

    /// F1: outputs can be declared explicitly or inferred from terminal operands.
    #[test]
    fn terminal_operands_become_outputs() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("outputs");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            scalar_shape(),
        );

        let count = builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("marking terminal operands must succeed");

        assert_eq!(count, 1);
        assert_eq!(comp.outputs.len(), 1);
        assert_eq!(comp.outputs[0].operand, sum);
    }

    /// F1: a graph with operations but no declared outputs is invalid.
    #[test]
    fn validation_rejects_output_less_graph() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("no_outputs");

        let _ = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );

        let err = builder
            .validate_computation(&comp)
            .expect_err("a graph with no declared outputs must be rejected");
        assert!(
            format!("{err}").contains("no outputs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_computation_validates() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let computation = builder.create_computation("test");
        assert!(builder.validate_computation(&computation).is_ok());
    }

    /// Operations must come out in dependency order.
    #[test]
    fn topological_order_respects_dependencies() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("topo");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            scalar_shape(),
        );
        let squared = add_op(
            &mut builder,
            &mut comp,
            OperationType::Square,
            vec![sum],
            scalar_shape(),
        );

        let order = builder
            .get_topological_order(&comp)
            .expect("acyclic graph must sort");
        assert_eq!(order.len(), 4);

        let position = |operand: OperandId| {
            let op_id = comp
                .producer_of(operand)
                .map(|op| op.id)
                .expect("operand must have a producer");
            order
                .iter()
                .position(|&id| id == op_id)
                .expect("every operation appears in the order")
        };

        assert!(position(a) < position(sum));
        assert!(position(b) < position(sum));
        assert!(position(sum) < position(squared));
    }

    /// A planted cycle must be detected rather than silently accepted.
    #[test]
    fn cycle_detection_catches_planted_cycle() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("cyclic");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Square,
            vec![a],
            scalar_shape(),
        );

        let first = comp
            .producer_of(a)
            .map(|op| op.id)
            .expect("operand a has a producer");
        let second = comp
            .producer_of(b)
            .map(|op| op.id)
            .expect("operand b has a producer");

        // Plant the back edge: `first` now also depends on `second`.
        comp.dependencies.insert(first, vec![second]);

        let err = builder
            .validate_computation(&comp)
            .expect_err("a cyclic dependency graph must be rejected");
        assert!(
            format!("{err}").contains("Cycle"),
            "unexpected error: {err}"
        );

        assert!(
            builder.get_topological_order(&comp).is_err(),
            "topological sort must fail on a cyclic graph"
        );
    }

    /// F2: constants must survive `Clone` and compare by value.
    #[test]
    fn constant_payload_survives_clone_and_compares_by_value() {
        let original = OperationType::Constant(ConstantValue::scalar(2.5));
        let cloned = original.clone();

        assert_eq!(original, cloned, "cloning must preserve the payload");

        match cloned {
            OperationType::Constant(value) => {
                assert_eq!(value.as_scalar(), Some(2.5));
            }
            other => panic!("clone changed the variant: {other:?}"),
        }

        assert_ne!(
            OperationType::Constant(ConstantValue::scalar(1.0)),
            OperationType::Constant(ConstantValue::scalar(2.0)),
            "different constants must not compare equal"
        );
        assert_eq!(
            OperationType::Constant(ConstantValue::scalar(1.0)),
            OperationType::Constant(ConstantValue::scalar(1.0)),
            "identical constants must compare equal"
        );
    }

    /// `Hash` and `PartialEq` must agree, or hash-map based passes break.
    #[test]
    fn constant_hash_agrees_with_equality() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |op: &OperationType| {
            let mut hasher = DefaultHasher::new();
            op.hash(&mut hasher);
            hasher.finish()
        };

        let a = OperationType::Constant(ConstantValue::scalar(3.25));
        let b = OperationType::Constant(ConstantValue::scalar(3.25));
        let c = OperationType::Constant(ConstantValue::scalar(4.0));

        assert_eq!(hash_of(&a), hash_of(&b));
        assert_ne!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn add_operation_rejects_unknown_input_operand() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("dangling");

        let result = builder.add_operation(
            &mut comp,
            OperationType::Square,
            vec![OperandId(999)],
            scalar_shape(),
        );

        assert!(result.is_err(), "dangling operand inputs must be rejected");
    }
}
