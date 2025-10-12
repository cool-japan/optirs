use std::any::Any;
use std::fmt::Debug;
// Computation graph capture for XLA compilation
//
// This module handles the capture and construction of computation graphs
// from high-level operations, including graph validation and optimization
// preparation.

use scirs2_core::ndarray::{Array1, Array2};
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

/// Types of XLA operations
#[derive(Debug)]
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
    Constant(Box<dyn Any>),
    Parameter,
    Iota,

    // Custom operations
    Custom(CustomOperation),
}

impl std::hash::Hash for OperationType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use OperationType::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Constant(_) => {
                // Don't hash the Any content
            }
            Reduce(r) => r.function.hash(state),
            AllReduce(a) => a.function.hash(state),
            Convolution(c) => {
                c.strides.hash(state);
                // Hash other fields if they implement Hash
            }
            Custom(c) => c.name.hash(state),
            _ => {}
        }
    }
}

impl PartialEq for OperationType {
    fn eq(&self, other: &Self) -> bool {
        use OperationType::*;
        match (self, other) {
            (Add, Add) | (Multiply, Multiply) | (Subtract, Subtract) | (Divide, Divide) => true,
            (Maximum, Maximum) | (Minimum, Minimum) | (Abs, Abs) | (Exp, Exp) => true,
            (Log, Log) | (Sqrt, Sqrt) | (Rsqrt, Rsqrt) | (Square, Square) => true,
            (Sign, Sign) | (Negate, Negate) | (Sin, Sin) | (Cos, Cos) => true,
            (Tanh, Tanh) | (Ceil, Ceil) | (Floor, Floor) | (Round, Round) => true,
            (Not, Not) | (And, And) | (Or, Or) | (Xor, Xor) => true,
            (Equal, Equal) | (NotEqual, NotEqual) | (Less, Less) | (LessEqual, LessEqual) => true,
            (Greater, Greater) | (GreaterEqual, GreaterEqual) => true,
            (Reshape, Reshape) | (Transpose, Transpose) | (Slice, Slice) => true,
            (DynamicSlice, DynamicSlice) | (Pad, Pad) | (Reverse, Reverse) => true,
            (Broadcast, Broadcast) | (Concatenate, Concatenate) => true,
            (Gather, Gather) | (Scatter, Scatter) => true,
            (Dot, Dot) | (DotGeneral, DotGeneral) | (MatMul, MatMul) => true,
            (Conditional, Conditional) | (While, While) | (Call, Call) => true,
            (AllGather, AllGather)
            | (AllToAll, AllToAll)
            | (CollectivePermute, CollectivePermute) => true,
            (ReduceScatter, ReduceScatter) | (BatchNorm, BatchNorm) | (Dropout, Dropout) => true,
            (Copy, Copy) | (Tuple, Tuple) | (GetTupleElement, GetTupleElement) => true,
            (Parameter, Parameter) | (Iota, Iota) | (ReduceWindow, ReduceWindow) => true,
            (Reduce(a), Reduce(b)) => a == b,
            (AllReduce(a), AllReduce(b)) => a == b,
            (Convolution(a), Convolution(b)) => a == b,
            (Custom(a), Custom(b)) => a == b,
            (Constant(_), Constant(_)) => false, // Can't compare Box<dyn Any>
            _ => false,
        }
    }
}

impl Eq for OperationType {}

impl Clone for OperationType {
    fn clone(&self) -> Self {
        match self {
            OperationType::Add => OperationType::Add,
            OperationType::Multiply => OperationType::Multiply,
            OperationType::Subtract => OperationType::Subtract,
            OperationType::Divide => OperationType::Divide,
            OperationType::Maximum => OperationType::Maximum,
            OperationType::Minimum => OperationType::Minimum,
            OperationType::Abs => OperationType::Abs,
            OperationType::Exp => OperationType::Exp,
            OperationType::Log => OperationType::Log,
            OperationType::Sqrt => OperationType::Sqrt,
            OperationType::Rsqrt => OperationType::Rsqrt,
            OperationType::Square => OperationType::Square,
            OperationType::Sign => OperationType::Sign,
            OperationType::Negate => OperationType::Negate,
            OperationType::Sin => OperationType::Sin,
            OperationType::Cos => OperationType::Cos,
            OperationType::Tanh => OperationType::Tanh,
            OperationType::Ceil => OperationType::Ceil,
            OperationType::Floor => OperationType::Floor,
            OperationType::Round => OperationType::Round,
            OperationType::Not => OperationType::Not,
            OperationType::And => OperationType::And,
            OperationType::Or => OperationType::Or,
            OperationType::Xor => OperationType::Xor,
            OperationType::Equal => OperationType::Equal,
            OperationType::NotEqual => OperationType::NotEqual,
            OperationType::Less => OperationType::Less,
            OperationType::LessEqual => OperationType::LessEqual,
            OperationType::Greater => OperationType::Greater,
            OperationType::GreaterEqual => OperationType::GreaterEqual,
            OperationType::MatMul => OperationType::MatMul,
            OperationType::Dot => OperationType::Dot,
            OperationType::Transpose => OperationType::Transpose,
            OperationType::Reshape => OperationType::Reshape,
            OperationType::Broadcast => OperationType::Broadcast,
            OperationType::Slice => OperationType::Slice,
            OperationType::Concatenate => OperationType::Concatenate,
            OperationType::Gather => OperationType::Gather,
            OperationType::Scatter => OperationType::Scatter,
            OperationType::Reduce(r) => OperationType::Reduce(r.clone()),
            OperationType::AllReduce(a) => OperationType::AllReduce(a.clone()),
            OperationType::AllGather => OperationType::AllGather,
            OperationType::AllToAll => OperationType::AllToAll,
            OperationType::CollectivePermute => OperationType::CollectivePermute,
            OperationType::ReduceScatter => OperationType::ReduceScatter,
            OperationType::Convolution(c) => OperationType::Convolution(c.clone()),
            OperationType::BatchNorm => OperationType::BatchNorm,
            OperationType::Dropout => OperationType::Dropout,
            OperationType::Constant(_) => OperationType::Constant(Box::new(())),
            OperationType::Parameter => OperationType::Parameter,
            OperationType::Iota => OperationType::Iota,
            OperationType::Custom(c) => OperationType::Custom(c.clone()),
            OperationType::DynamicSlice => OperationType::DynamicSlice,
            OperationType::Pad => OperationType::Pad,
            OperationType::Reverse => OperationType::Reverse,
            OperationType::ReduceWindow => OperationType::ReduceWindow,
            OperationType::DotGeneral => OperationType::DotGeneral,
            OperationType::Conditional => OperationType::Conditional,
            OperationType::While => OperationType::While,
            OperationType::Call => OperationType::Call,
            OperationType::Copy => OperationType::Copy,
            OperationType::Tuple => OperationType::Tuple,
            OperationType::GetTupleElement => OperationType::GetTupleElement,
        }
    }
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
        }
    }

    /// Add operation to computation
    pub fn add_operation(
        &mut self,
        computation: &mut XLAComputation<T>,
        op_type: OperationType,
        inputs: Vec<OperandId>,
        output_shape: TensorShape,
    ) -> Result<OperationId> {
        let op_id = OperationId(self.next_op_id);
        self.next_op_id += 1;

        // Create output operand
        let output_operand_id = OperandId(computation.operands.len());
        let output_operand = Operand {
            id: output_operand_id,
            shape: output_shape,
            layout: Layout::default(),
            dtype: DataType::F32, // Default type
            metadata: OperandMetadata::default(),
            _phantom: std::marker::PhantomData,
        };

        computation
            .operands
            .insert(output_operand_id, output_operand);

        // Create operation
        let operation = XLAOperation {
            id: op_id,
            op_type,
            inputs: inputs.clone(),
            output: output_operand_id,
            attributes: OperationAttributes::default(),
            performance: OperationPerformanceCharacteristics::default(),
            memory_requirements: OperationMemoryRequirements::default(),
            source_location: None,
            _phantom: std::marker::PhantomData,
        };

        computation.operations.push(operation);

        // Update dependencies
        let input_ops: Vec<OperationId> = inputs
            .iter()
            .filter_map(|&operand_id| {
                computation
                    .operands
                    .get(&operand_id)
                    .and_then(|operand| operand.metadata.producer)
            })
            .collect();

        computation.dependencies.insert(op_id, input_ops);

        Ok(op_id)
    }

    /// Validate computation graph
    pub fn validate_computation(&self, computation: &XLAComputation<T>) -> Result<()> {
        // Check for cycles
        self.check_for_cycles(computation)?;

        // Check shape compatibility
        self.check_shape_compatibility(computation)?;

        // Check resource requirements
        self.check_resource_requirements(computation)?;

        Ok(())
    }

    /// Check for cycles in computation graph
    fn check_for_cycles(&self, computation: &XLAComputation<T>) -> Result<()> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for operation in &computation.operations {
            if !visited.contains(&operation.id)
                && Self::has_cycle_util(computation, operation.id, &mut visited, &mut rec_stack)?
            {
                return Err(OptimError::from(
                    "Cycle detected in computation graph".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Utility function for cycle detection
    fn has_cycle_util(
        computation: &XLAComputation<T>,
        op_id: OperationId,
        visited: &mut HashSet<OperationId>,
        rec_stack: &mut HashSet<OperationId>,
    ) -> Result<bool> {
        visited.insert(op_id);
        rec_stack.insert(op_id);

        if let Some(dependencies) = computation.dependencies.get(&op_id) {
            for &dep_id in dependencies {
                if !visited.contains(&dep_id) {
                    if Self::has_cycle_util(computation, dep_id, visited, rec_stack)? {
                        return Ok(true);
                    }
                } else if rec_stack.contains(&dep_id) {
                    return Ok(true);
                }
            }
        }

        rec_stack.remove(&op_id);
        Ok(false)
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
            for &dep_id in dependencies {
                adj_list.get_mut(&dep_id).unwrap().push(*op_id);
                *in_degree.get_mut(op_id).unwrap() += 1;
            }
        }

        // Topological sort using Kahn's algorithm
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        for (&op_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(op_id);
            }
        }

        while let Some(op_id) = queue.pop_front() {
            result.push(op_id);

            if let Some(neighbors) = adj_list.get(&op_id) {
                for &neighbor in neighbors {
                    let degree = in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
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
mod tests {
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

        let shape = TensorShape {
            dimensions: vec![10, 10],
            dynamic_dimensions: vec![false, false],
            element_count: 100,
            tuple_shapes: Vec::new(),
        };

        let result = builder.add_operation(&mut computation, OperationType::Add, vec![], shape);

        assert!(result.is_ok());
        assert_eq!(computation.operations.len(), 1);
    }

    #[test]
    fn test_graph_validation() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let computation = builder.create_computation("test");

        let result = builder.validate_computation(&computation);
        assert!(result.is_ok());
    }
}
