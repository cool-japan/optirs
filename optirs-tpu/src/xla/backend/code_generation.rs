use std::fmt::Debug;
// TPU code generation for XLA computations
//
// This module implements code generation for TPU hardware, including
// kernel generation, instruction scheduling, register allocation,
// and hardware-specific optimizations.

use scirs2_core::numeric::Float;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

use super::super::frontend::{
    DataType, Layout, OperandId, OperationType, TensorShape, XLAComputation, XLAOperation,
};
use super::super::optimization::MemoryPlan;
use super::super::{GeneratedCode, TPUConfig};
use crate::error::{OptimError, Result};

/// TPU code generator
pub struct TPUCodeGenerator<T: Float + Debug + Send + Sync + 'static> {
    /// Target TPU configuration
    target_config: TPUConfig,

    /// Instruction generator
    instruction_generator: InstructionGenerator<T>,

    /// Kernel generator
    kernel_generator: KernelGenerator<T>,

    /// Register allocator
    register_allocator: RegisterAllocator,

    /// Instruction scheduler
    instruction_scheduler: InstructionScheduler<T>,

    /// Code optimizer
    code_optimizer: CodeOptimizer<T>,

    /// Generation statistics
    generation_stats: CodeGenerationStats,
}

/// Code generation statistics
#[derive(Debug, Default)]
pub struct CodeGenerationStats {
    /// Total instructions generated
    pub instructions_generated: usize,

    /// Number of kernels generated
    pub kernels_generated: usize,

    /// Register pressure peak
    pub max_register_pressure: usize,

    /// Code size (bytes)
    pub code_size: usize,

    /// Generation time (microseconds)
    pub generation_time_us: u64,

    /// Optimization passes applied
    pub optimization_passes: usize,
}

/// Instruction generator for TPU operations
pub struct InstructionGenerator<T: Float + Debug + Send + Sync + 'static> {
    /// Instruction templates
    instruction_templates: HashMap<OperationType, InstructionTemplate>,

    /// Generated instructions
    generated_instructions: Vec<TPUInstruction>,

    /// Instruction counter
    instruction_counter: usize,

    _phantom: std::marker::PhantomData<T>,
}

/// TPU instruction representation
#[derive(Debug, Clone)]
pub struct TPUInstruction {
    /// Instruction ID
    pub id: usize,

    /// Instruction opcode
    pub opcode: TPUOpcode,

    /// Operands
    pub operands: Vec<TPUOperand>,

    /// Result register
    pub result: Option<TPURegister>,

    /// Instruction attributes
    pub attributes: InstructionAttributes,

    /// Scheduling information
    pub scheduling_info: SchedulingInfo,
}

/// TPU opcodes
#[derive(Debug, Clone, PartialEq)]
pub enum TPUOpcode {
    // Matrix operations
    MatMul,
    MatMulAccumulate,

    // Vector operations
    VectorAdd,
    VectorMultiply,
    VectorDot,

    // Scalar operations
    ScalarAdd,
    ScalarMultiply,

    // Memory operations
    Load,
    Store,
    Move,

    // Control flow
    Branch,
    Call,
    Return,

    // Special operations
    Reduce,
    Transpose,
    Reshape,

    // Communication
    AllReduce,
    AllGather,

    // Custom operations
    Custom(String),
}

/// TPU operand
#[derive(Debug, Clone)]
pub enum TPUOperand {
    /// Register operand
    Register(TPURegister),

    /// Immediate value
    Immediate(i64),

    /// Memory address
    Memory(MemoryAddress),

    /// Label reference
    Label(String),
}

/// TPU register
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct TPURegister {
    /// Register type
    pub reg_type: RegisterType,

    /// Register index
    pub index: usize,

    /// Data type stored in register
    pub data_type: DataType,

    /// Register size (bytes)
    pub size: usize,
}

/// Types of TPU registers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RegisterType {
    /// Matrix registers (for matrix operations)
    Matrix,

    /// Vector registers (for vector operations)
    Vector,

    /// Scalar registers (for scalar operations)
    Scalar,

    /// Address registers (for memory operations)
    Address,

    /// Predicate registers (for control flow)
    Predicate,
}

/// Memory address representation
#[derive(Debug, Clone)]
pub struct MemoryAddress {
    /// Base address
    pub base: Option<TPURegister>,

    /// Offset
    pub offset: i64,

    /// Index register
    pub index: Option<TPURegister>,

    /// Scale factor
    pub scale: usize,

    /// Memory space
    pub memory_space: MemorySpace,
}

/// Memory spaces for TPU
#[derive(Debug, Clone)]
pub enum MemorySpace {
    /// Local memory (L1)
    Local,

    /// Shared memory (L2)
    Shared,

    /// Global memory (HBM)
    Global,

    /// Host memory
    Host,
}

/// Instruction attributes
#[derive(Debug, Clone, Default)]
pub struct InstructionAttributes {
    /// Instruction latency
    pub latency: u32,

    /// Throughput (instructions per cycle)
    pub throughput: f64,

    /// Resource requirements
    pub resources: Vec<String>,

    /// Memory bandwidth requirement
    pub memory_bandwidth: f64,

    /// Can be predicated
    pub predicable: bool,
}

/// Scheduling information
#[derive(Debug, Clone, Default)]
pub struct SchedulingInfo {
    /// Earliest scheduling cycle
    pub earliest_cycle: u64,

    /// Latest scheduling cycle
    pub latest_cycle: u64,

    /// Actual scheduled cycle
    pub scheduled_cycle: Option<u64>,

    /// Dependencies
    pub dependencies: Vec<usize>,

    /// Resource conflicts
    pub resource_conflicts: Vec<usize>,
}

/// Instruction template for code generation
#[derive(Debug, Clone)]
pub struct InstructionTemplate {
    /// Template name
    pub name: String,

    /// Operation type this template applies to
    pub operation_type: OperationType,

    /// Instruction pattern
    pub pattern: Vec<TPUOpcode>,

    /// Operand mapping
    pub operand_mapping: Vec<OperandMapping>,

    /// Resource requirements
    pub resource_requirements: Vec<String>,
}

/// Operand mapping for templates
#[derive(Debug, Clone)]
pub enum OperandMapping {
    /// Input operand
    Input(usize),

    /// Output operand
    Output(usize),

    /// Constant value
    Constant(i64),

    /// Register allocation
    Register(RegisterType),
}

/// Kernel generator for TPU kernels
pub struct KernelGenerator<T: Float + Debug + Send + Sync + 'static> {
    /// Generated kernels
    kernels: Vec<TPUKernel>,

    /// Kernel optimization passes, applied by [`Self::generate_kernels`].
    ///
    /// There is no `templates` map any more: kernels are emitted directly from
    /// the scheduled instruction stream (see `generate_kernels`), nothing ever
    /// looked a [`KernelTemplate`] up, and a permanently empty template registry
    /// only advertised a substitution mechanism that does not exist.
    optimization_passes: Vec<Box<dyn KernelOptimizationPass>>,

    _phantom: std::marker::PhantomData<T>,
}

/// TPU kernel representation
#[derive(Debug, Clone)]
pub struct TPUKernel {
    /// Kernel name
    pub name: String,

    /// Kernel instructions
    pub instructions: Vec<TPUInstruction>,

    /// Kernel parameters
    pub parameters: Vec<KernelParameter>,

    /// Local memory requirements
    pub local_memory: usize,

    /// Register requirements
    pub register_requirements: RegisterRequirements,

    /// Performance characteristics
    pub performance: KernelPerformance,
}

/// Kernel parameter
#[derive(Debug, Clone)]
pub struct KernelParameter {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub param_type: ParameterType,

    /// Memory layout
    pub layout: Layout,

    /// Access pattern
    pub access_pattern: AccessPattern,
}

/// Parameter types
#[derive(Debug, Clone)]
pub enum ParameterType {
    /// Input tensor
    InputTensor(TensorShape, DataType),

    /// Output tensor
    OutputTensor(TensorShape, DataType),

    /// Scalar parameter
    Scalar(DataType),

    /// Buffer parameter
    Buffer(usize),
}

/// Access patterns for parameters
#[derive(Debug, Clone)]
pub enum AccessPattern {
    /// Read-only access
    ReadOnly,

    /// Write-only access
    WriteOnly,

    /// Read-write access
    ReadWrite,

    /// Reduction access
    Reduction,
}

/// Register requirements for kernel
#[derive(Debug, Default, Clone)]
pub struct RegisterRequirements {
    /// Matrix registers needed
    pub matrix_registers: usize,

    /// Vector registers needed
    pub vector_registers: usize,

    /// Scalar registers needed
    pub scalar_registers: usize,

    /// Address registers needed
    pub address_registers: usize,
}

/// Kernel performance characteristics
#[derive(Debug, Default, Clone)]
pub struct KernelPerformance {
    /// Estimated cycles
    pub estimated_cycles: u64,

    /// Arithmetic intensity
    pub arithmetic_intensity: f64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_util: f64,

    /// Compute utilization
    pub compute_utilization: f64,
}

/// Kernel template for code generation
#[derive(Debug)]
pub struct KernelTemplate {
    /// Template name
    pub name: String,

    /// Supported operations
    pub supported_operations: Vec<OperationType>,

    /// Template code
    pub template_code: String,

    /// Parameter substitutions
    pub substitutions: HashMap<String, String>,
}

/// Kernel optimization pass
pub trait KernelOptimizationPass {
    /// Pass name
    fn name(&self) -> &str;

    /// Apply optimization to kernel
    fn optimize(&self, kernel: &mut TPUKernel) -> Result<bool>;

    /// Check if pass is applicable
    fn is_applicable(&self, kernel: &TPUKernel) -> bool;
}

/// Every register an operand reads, including the base/index registers of a
/// memory operand.
fn operand_registers(operand: &TPUOperand) -> Vec<TPURegister> {
    match operand {
        TPUOperand::Register(register) => vec![register.clone()],
        TPUOperand::Memory(address) => address
            .base
            .iter()
            .chain(address.index.iter())
            .cloned()
            .collect(),
        TPUOperand::Immediate(_) | TPUOperand::Label(_) => Vec::new(),
    }
}

/// Register allocator for TPU
pub struct RegisterAllocator {
    /// Available registers by type
    available_registers: HashMap<RegisterType, HashSet<usize>>,

    /// Register assignments
    assignments: HashMap<OperandId, TPURegister>,

    /// Register pressure tracking
    pressure_tracking: BTreeMap<u64, RegisterPressure>,

    /// Spill decisions
    spill_decisions: Vec<SpillDecision>,
}

/// Register pressure at a point in time
#[derive(Debug, Default)]
pub struct RegisterPressure {
    /// Pressure by register type
    pub pressure_by_type: HashMap<RegisterType, usize>,

    /// Total pressure
    pub total_pressure: usize,

    /// Spill cost at this point
    pub spill_cost: f64,
}

/// Spill decision
#[derive(Debug)]
pub struct SpillDecision {
    /// Operand to spill
    pub operand: OperandId,

    /// Register being spilled
    pub register: TPURegister,

    /// Spill location
    pub spill_location: MemoryAddress,

    /// Spill cost
    pub cost: f64,
}

/// Instruction scheduler for TPU
pub struct InstructionScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Scheduling strategy
    strategy: SchedulingStrategy,

    /// Resource model
    resource_model: ResourceModel,

    /// Dependency graph
    dependency_graph: InstructionDependencyGraph,

    _phantom: std::marker::PhantomData<T>,
}

/// Scheduling strategies for instructions
#[derive(Debug)]
pub enum SchedulingStrategy {
    /// List scheduling
    List,

    /// Critical path scheduling
    CriticalPath,

    /// Software pipelining
    SoftwarePipelining,

    /// Trace scheduling
    Trace,
}

/// Resource model for TPU.
///
/// Only the execution units are modelled. A pipeline-stage list and a
/// unit-conflict map used to be declared here and were never consulted: the
/// scheduler below issues one instruction per step, so it has no co-issue
/// decision to make and no structural hazard to resolve. Declaring a hazard
/// model that nothing enforces would overstate what the scheduler does.
#[derive(Debug)]
pub struct ResourceModel {
    /// Available execution units
    execution_units: Vec<ExecutionUnit>,
}

/// Execution unit model
#[derive(Debug)]
pub struct ExecutionUnit {
    /// Unit name
    pub name: String,

    /// Supported operations
    pub supported_ops: Vec<TPUOpcode>,

    /// Latency
    pub latency: u32,

    /// Throughput
    pub throughput: f64,
}

/// Pipeline stage model
#[derive(Debug)]
pub struct PipelineStage {
    /// Stage name
    pub name: String,

    /// Stage latency
    pub latency: u32,

    /// Resources used
    pub resources: Vec<String>,
}

/// Instruction dependency graph
#[derive(Debug)]
pub struct InstructionDependencyGraph {
    /// Dependencies between instructions
    pub dependencies: HashMap<usize, Vec<usize>>,

    /// Dependency types
    pub dependency_types: HashMap<(usize, usize), DependencyType>,

    /// Critical path
    pub critical_path: Vec<usize>,
}

/// Types of instruction dependencies
#[derive(Debug)]
pub enum DependencyType {
    /// True dependency (read after write)
    True,

    /// Anti dependency (write after read)
    Anti,

    /// Output dependency (write after write)
    Output,

    /// Control dependency
    Control,

    /// Resource dependency
    Resource,
}

/// Code optimizer for generated TPU code
pub struct CodeOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Optimization passes
    passes: Vec<Box<dyn CodeOptimizationPass<T>>>,

    /// Pass statistics
    pass_stats: HashMap<String, OptimizationStats>,
}

/// Code optimization pass trait
pub trait CodeOptimizationPass<T: Float + Debug + Send + Sync + 'static> {
    /// Pass name
    fn name(&self) -> &str;

    /// Apply optimization
    fn optimize(&self, code: &mut GeneratedCode) -> Result<bool>;

    /// Check if applicable
    fn is_applicable(&self, code: &GeneratedCode) -> bool;
}

/// Optimization statistics
#[derive(Debug, Default)]
pub struct OptimizationStats {
    /// Instructions eliminated
    pub instructions_eliminated: usize,

    /// Cycles saved
    pub cycles_saved: u64,

    /// Memory accesses eliminated
    pub memory_accesses_eliminated: usize,
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> TPUCodeGenerator<T> {
    /// Create new TPU code generator
    pub fn new(target_config: TPUConfig) -> Self {
        Self {
            instruction_generator: InstructionGenerator::new(&target_config),
            kernel_generator: KernelGenerator::new(&target_config),
            register_allocator: RegisterAllocator::new(&target_config),
            instruction_scheduler: InstructionScheduler::new(&target_config),
            code_optimizer: CodeOptimizer::new(),
            target_config,
            generation_stats: CodeGenerationStats::default(),
        }
    }

    /// Generate code for XLA computation
    pub fn generate_code(
        &mut self,
        computation: &XLAComputation<T>,
        memory_plan: &MemoryPlan<T>,
    ) -> Result<GeneratedCode> {
        let start_time = std::time::Instant::now();

        // Generate instructions for each operation
        let mut all_instructions = Vec::new();
        for operation in &computation.operations {
            let instructions = self
                .instruction_generator
                .generate_instructions(operation)?;
            all_instructions.extend(instructions);
        }

        // Allocate registers
        self.register_allocator
            .allocate_registers(&all_instructions, memory_plan)?;

        // Schedule instructions
        let scheduled_instructions = self
            .instruction_scheduler
            .schedule_instructions(&all_instructions)?;

        // Generate kernels
        let kernels = self
            .kernel_generator
            .generate_kernels(&scheduled_instructions, memory_plan)?;

        // Generate final code
        let mut generated_code = self.generate_final_code(&kernels)?;

        // Apply optimizations
        self.code_optimizer.optimize(&mut generated_code)?;

        self.generation_stats.generation_time_us = start_time.elapsed().as_micros() as u64;
        self.generation_stats.instructions_generated = all_instructions.len();
        self.generation_stats.kernels_generated = kernels.len();
        self.generation_stats.code_size = generated_code.kernel_code.len();

        Ok(generated_code)
    }

    /// Generate final code from kernels
    fn generate_final_code(&self, kernels: &[TPUKernel]) -> Result<GeneratedCode> {
        let mut kernel_code = String::new();
        let mut init_code = String::new();
        let mut cleanup_code = String::new();
        let mut memory_code = String::new();

        // Generate kernel code
        for kernel in kernels {
            writeln!(kernel_code, "// Kernel: {}", kernel.name)
                .map_err(|e| OptimError::from(e.to_string()))?;
            writeln!(kernel_code, "kernel {} {{", kernel.name)
                .map_err(|e| OptimError::from(e.to_string()))?;

            for instruction in &kernel.instructions {
                let asm_code = self.generate_assembly(instruction)?;
                writeln!(kernel_code, "  {}", asm_code)
                    .map_err(|e| OptimError::from(e.to_string()))?;
            }

            writeln!(kernel_code, "}}").map_err(|e| OptimError::from(e.to_string()))?;
            writeln!(kernel_code).map_err(|e| OptimError::from(e.to_string()))?;
        }

        // Generate initialization code. The target the generator was built for
        // is emitted here rather than being carried around unused: the runtime
        // needs to know which architecture and how many cores the code assumes.
        writeln!(init_code, "// Initialization").map_err(|e| OptimError::from(e.to_string()))?;
        writeln!(
            init_code,
            "// target: {:?}, cores: {}, opt: {:?}",
            self.target_config.tpu_version,
            self.target_config.num_cores,
            self.target_config.xla_optimization_level
        )
        .map_err(|e| OptimError::from(e.to_string()))?;
        writeln!(
            init_code,
            "init_tpu({:?}, {});",
            self.target_config.tpu_version, self.target_config.num_cores
        )
        .map_err(|e| OptimError::from(e.to_string()))?;

        // Generate cleanup code
        writeln!(cleanup_code, "// Cleanup").map_err(|e| OptimError::from(e.to_string()))?;
        writeln!(cleanup_code, "cleanup_tpu();").map_err(|e| OptimError::from(e.to_string()))?;

        // Generate memory management code
        writeln!(memory_code, "// Memory management")
            .map_err(|e| OptimError::from(e.to_string()))?;
        writeln!(memory_code, "allocate_buffers();")
            .map_err(|e| OptimError::from(e.to_string()))?;

        Ok(GeneratedCode {
            kernel_code,
            init_code,
            cleanup_code,
            memory_code,
        })
    }

    /// Generate assembly code for instruction
    fn generate_assembly(&self, instruction: &TPUInstruction) -> Result<String> {
        let mut asm = String::new();

        match &instruction.opcode {
            TPUOpcode::MatMul => {
                write!(asm, "matmul").map_err(|e| OptimError::from(e.to_string()))?;
            }
            TPUOpcode::VectorAdd => {
                write!(asm, "vadd").map_err(|e| OptimError::from(e.to_string()))?;
            }
            TPUOpcode::Load => {
                write!(asm, "load").map_err(|e| OptimError::from(e.to_string()))?;
            }
            TPUOpcode::Store => {
                write!(asm, "store").map_err(|e| OptimError::from(e.to_string()))?;
            }
            _ => {
                write!(asm, "{:?}", instruction.opcode)
                    .map_err(|e| OptimError::from(e.to_string()))?;
            }
        }

        // Add operands
        for (i, operand) in instruction.operands.iter().enumerate() {
            if i > 0 {
                write!(asm, ",").map_err(|e| OptimError::from(e.to_string()))?;
            }
            write!(asm, " {}", self.format_operand(operand)?)
                .map_err(|e| OptimError::from(e.to_string()))?;
        }

        // Add result
        if let Some(result) = &instruction.result {
            write!(asm, " -> {}", self.format_register(result)?)
                .map_err(|e| OptimError::from(e.to_string()))?;
        }

        Ok(asm)
    }

    /// Format operand for assembly
    fn format_operand(&self, operand: &TPUOperand) -> Result<String> {
        match operand {
            TPUOperand::Register(reg) => self.format_register(reg),
            TPUOperand::Immediate(val) => Ok(format!("#{}", val)),
            TPUOperand::Memory(addr) => Ok(format!("[{}]", self.format_memory_address(addr)?)),
            TPUOperand::Label(label) => Ok(label.clone()),
        }
    }

    /// Format register for assembly
    fn format_register(&self, register: &TPURegister) -> Result<String> {
        let prefix = match register.reg_type {
            RegisterType::Matrix => "m",
            RegisterType::Vector => "v",
            RegisterType::Scalar => "s",
            RegisterType::Address => "a",
            RegisterType::Predicate => "p",
        };
        Ok(format!("{}{}", prefix, register.index))
    }

    /// Format memory address for assembly
    fn format_memory_address(&self, address: &MemoryAddress) -> Result<String> {
        let mut addr_str = String::new();

        if let Some(base) = &address.base {
            write!(addr_str, "{}", self.format_register(base)?)
                .map_err(|e| OptimError::from(e.to_string()))?;
        }

        if address.offset != 0 {
            if !addr_str.is_empty() {
                write!(addr_str, "+").map_err(|e| OptimError::from(e.to_string()))?;
            }
            write!(addr_str, "{}", address.offset).map_err(|e| OptimError::from(e.to_string()))?;
        }

        if let Some(index) = &address.index {
            if !addr_str.is_empty() {
                write!(addr_str, "+").map_err(|e| OptimError::from(e.to_string()))?;
            }
            write!(
                addr_str,
                "{}*{}",
                self.format_register(index)?,
                address.scale
            )
            .map_err(|e| OptimError::from(e.to_string()))?;
        }

        Ok(addr_str)
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        self.generation_stats = CodeGenerationStats::default();
        self.instruction_generator.reset();
        self.kernel_generator.reset();
        self.register_allocator.reset();
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> InstructionGenerator<T> {
    /// Create new instruction generator
    pub fn new(target_config: &TPUConfig) -> Self {
        let mut generator = Self {
            instruction_templates: HashMap::new(),
            generated_instructions: Vec::new(),
            instruction_counter: 0,
            _phantom: std::marker::PhantomData,
        };

        generator.initialize_templates(target_config);
        generator
    }

    /// Initialize instruction templates
    fn initialize_templates(&mut self, _target_config: &TPUConfig) {
        // Add matrix multiplication template
        self.instruction_templates.insert(
            OperationType::Dot,
            InstructionTemplate {
                name: "dot_product".to_string(),
                operation_type: OperationType::Dot,
                pattern: vec![TPUOpcode::MatMul],
                operand_mapping: vec![
                    OperandMapping::Input(0),
                    OperandMapping::Input(1),
                    OperandMapping::Output(0),
                ],
                resource_requirements: vec!["matrix_unit".to_string()],
            },
        );

        // Add vector addition template
        self.instruction_templates.insert(
            OperationType::Add,
            InstructionTemplate {
                name: "vector_add".to_string(),
                operation_type: OperationType::Add,
                pattern: vec![TPUOpcode::VectorAdd],
                operand_mapping: vec![
                    OperandMapping::Input(0),
                    OperandMapping::Input(1),
                    OperandMapping::Output(0),
                ],
                resource_requirements: vec!["vector_unit".to_string()],
            },
        );
    }

    /// Generate instructions for operation
    pub fn generate_instructions(
        &mut self,
        operation: &XLAOperation<T>,
    ) -> Result<Vec<TPUInstruction>> {
        if let Some(template) = self.instruction_templates.get(&operation.op_type) {
            let mut instructions = Vec::new();

            for opcode in &template.pattern {
                let instruction = TPUInstruction {
                    id: self.instruction_counter,
                    opcode: opcode.clone(),
                    operands: self.map_operands(&template.operand_mapping, operation)?,
                    result: Some(TPURegister {
                        reg_type: RegisterType::Vector, // Default
                        index: operation.output.0,
                        data_type: DataType::F32,
                        size: 4,
                    }),
                    attributes: InstructionAttributes {
                        latency: self.get_operation_latency(&operation.op_type),
                        throughput: 1.0,
                        resources: template.resource_requirements.clone(),
                        memory_bandwidth: 0.0,
                        predicable: false,
                    },
                    scheduling_info: SchedulingInfo::default(),
                };

                instructions.push(instruction);
                self.instruction_counter += 1;
            }

            self.generated_instructions.extend(instructions.clone());
            Ok(instructions)
        } else {
            // Default instruction generation
            Ok(vec![TPUInstruction {
                id: self.instruction_counter,
                opcode: TPUOpcode::Custom(format!("{:?}", operation.op_type)),
                operands: vec![],
                result: Some(TPURegister {
                    reg_type: RegisterType::Vector,
                    index: operation.output.0,
                    data_type: DataType::F32,
                    size: 4,
                }),
                attributes: InstructionAttributes::default(),
                scheduling_info: SchedulingInfo::default(),
            }])
        }
    }

    /// Map operands according to template
    fn map_operands(
        &self,
        mapping: &[OperandMapping],
        operation: &XLAOperation<T>,
    ) -> Result<Vec<TPUOperand>> {
        let mut operands = Vec::new();

        for map in mapping {
            match map {
                OperandMapping::Input(idx) => {
                    if *idx < operation.inputs.len() {
                        operands.push(TPUOperand::Register(TPURegister {
                            reg_type: RegisterType::Vector,
                            index: operation.inputs[*idx].0,
                            data_type: DataType::F32,
                            size: 4,
                        }));
                    }
                }
                OperandMapping::Output(idx) => {
                    if *idx == 0 {
                        operands.push(TPUOperand::Register(TPURegister {
                            reg_type: RegisterType::Vector,
                            index: operation.output.0,
                            data_type: DataType::F32,
                            size: 4,
                        }));
                    }
                }
                OperandMapping::Constant(val) => {
                    operands.push(TPUOperand::Immediate(*val));
                }
                OperandMapping::Register(reg_type) => {
                    operands.push(TPUOperand::Register(TPURegister {
                        reg_type: reg_type.clone(),
                        index: 0,
                        data_type: DataType::F32,
                        size: 4,
                    }));
                }
            }
        }

        Ok(operands)
    }

    /// Get operation latency
    fn get_operation_latency(&self, op_type: &OperationType) -> u32 {
        match op_type {
            OperationType::Add | OperationType::Multiply | OperationType::Subtract => 1,
            OperationType::Dot => 10,
            OperationType::Convolution(_) => 50,
            _ => 5,
        }
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        self.generated_instructions.clear();
        self.instruction_counter = 0;
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> KernelGenerator<T> {
    /// Create new kernel generator
    pub fn new(_target_config: &TPUConfig) -> Self {
        Self {
            kernels: Vec::new(),
            optimization_passes: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Emit the kernels for a scheduled instruction stream and run every
    /// registered optimization pass over them.
    pub fn generate_kernels(
        &mut self,
        instructions: &[TPUInstruction],
        memory_plan: &MemoryPlan<T>,
    ) -> Result<Vec<TPUKernel>> {
        let mut kernel = TPUKernel {
            name: "main_kernel".to_string(),
            instructions: instructions.to_vec(),
            parameters: vec![],
            // The kernel's scratch requirement is the memory plan's own total,
            // not zero.
            local_memory: memory_plan.total_memory,
            register_requirements: RegisterRequirements::default(),
            performance: KernelPerformance::default(),
        };

        // Apply the registered passes. The registry is empty unless a caller
        // adds one via `register_optimization_pass`, so this is an extension
        // point rather than a claim that optimizations happen.
        for pass in &self.optimization_passes {
            if pass.is_applicable(&kernel) {
                pass.optimize(&mut kernel)?;
            }
        }

        self.kernels.push(kernel.clone());
        Ok(vec![kernel])
    }

    /// Register a kernel-level optimization pass.
    pub fn register_optimization_pass(&mut self, pass: Box<dyn KernelOptimizationPass>) {
        self.optimization_passes.push(pass);
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        self.kernels.clear();
    }
}

impl RegisterAllocator {
    /// Create new register allocator
    pub fn new(_target_config: &TPUConfig) -> Self {
        let mut available_registers = HashMap::new();

        // Initialize available registers for each type
        let mut matrix_regs = HashSet::new();
        for i in 0..32 {
            matrix_regs.insert(i);
        }
        available_registers.insert(RegisterType::Matrix, matrix_regs);

        let mut vector_regs = HashSet::new();
        for i in 0..64 {
            vector_regs.insert(i);
        }
        available_registers.insert(RegisterType::Vector, vector_regs);

        Self {
            available_registers,
            assignments: HashMap::new(),
            pressure_tracking: BTreeMap::new(),
            spill_decisions: Vec::new(),
        }
    }

    /// Allocate registers for instructions
    /// Linear-scan register allocation over the instruction stream.
    ///
    /// The instruction generator emits *virtual* registers (a `(type, index)`
    /// pair per value). This computes each virtual register's live interval from
    /// its definition and last use, sweeps the instructions in order freeing
    /// physical registers as intervals end, and assigns a physical register from
    /// the pool declared in `available_registers`. When a type's pool is
    /// exhausted the live value whose next use is furthest away is spilled --
    /// the standard linear-scan choice -- and the decision is recorded with a
    /// real cost derived from the memory plan's own bandwidth measurement rather
    /// than a placeholder.
    ///
    /// This replaces a body that did nothing at all, which meant `assignments`,
    /// `pressure_tracking` and `spill_decisions` were permanently empty and the
    /// register pools were never consulted.
    pub fn allocate_registers<T: Float + Debug + Send + Sync + 'static>(
        &mut self,
        instructions: &[TPUInstruction],
        memory_plan: &MemoryPlan<T>,
    ) -> Result<()> {
        self.assignments.clear();
        self.pressure_tracking.clear();
        self.spill_decisions.clear();

        // Live intervals for every virtual register, keyed by (type, index).
        let mut definition: HashMap<(RegisterType, usize), usize> = HashMap::new();
        let mut last_use: HashMap<(RegisterType, usize), usize> = HashMap::new();
        for (position, instruction) in instructions.iter().enumerate() {
            if let Some(result) = &instruction.result {
                let key = (result.reg_type.clone(), result.index);
                definition.entry(key.clone()).or_insert(position);
                last_use.insert(key, position);
            }
            for operand in &instruction.operands {
                for register in operand_registers(operand) {
                    let key = (register.reg_type.clone(), register.index);
                    last_use.insert(key, position);
                }
            }
        }

        // Free physical register pools, taken from the declared availability.
        let mut free: HashMap<RegisterType, Vec<usize>> = self
            .available_registers
            .iter()
            .map(|(reg_type, indices)| {
                let mut pool: Vec<usize> = indices.iter().copied().collect();
                // Descending so `pop` hands out the lowest index first.
                pool.sort_unstable_by_key(|index| std::cmp::Reverse(*index));
                (reg_type.clone(), pool)
            })
            .collect();

        // Virtual registers currently holding a physical one, with the interval
        // end that lets them be released (and the spill victim be chosen).
        let mut active: Vec<((RegisterType, usize), usize, usize)> = Vec::new();

        // Spilling a value costs a store plus a reload; scale it by the plan's
        // measured bandwidth utilization so a bandwidth-bound program reports a
        // higher spill cost than a compute-bound one.
        let bandwidth_pressure = memory_plan
            .performance_info
            .bandwidth_utilization
            .clamp(0.0, 1.0);

        for (position, instruction) in instructions.iter().enumerate() {
            // Release everything whose last use is behind us.
            active.retain(|(key, physical, end)| {
                if *end < position {
                    if let Some(pool) = free.get_mut(&key.0) {
                        pool.push(*physical);
                        pool.sort_unstable_by_key(|index| std::cmp::Reverse(*index));
                    }
                    false
                } else {
                    true
                }
            });

            if let Some(result) = &instruction.result {
                let key = (result.reg_type.clone(), result.index);
                let end = last_use.get(&key).copied().unwrap_or(position);
                let operand = OperandId(instruction.id);

                match free.get_mut(&result.reg_type).and_then(|pool| pool.pop()) {
                    Some(physical) => {
                        active.push((key, physical, end));
                        self.assignments.insert(
                            operand,
                            TPURegister {
                                reg_type: result.reg_type.clone(),
                                index: physical,
                                data_type: result.data_type,
                                size: result.size,
                            },
                        );
                    }
                    None => {
                        // Spill the active value of this type whose use is
                        // furthest in the future; if this definition is itself
                        // the furthest, spill it instead.
                        let victim = active
                            .iter()
                            .enumerate()
                            .filter(|(_, (candidate, _, _))| candidate.0 == result.reg_type)
                            .max_by_key(|(_, (_, _, candidate_end))| *candidate_end)
                            .map(|(slot, (candidate, physical, candidate_end))| {
                                (slot, candidate.clone(), *physical, *candidate_end)
                            });

                        match victim {
                            Some((slot, victim_key, physical, victim_end)) if victim_end > end => {
                                active.remove(slot);
                                // The decision names the value that was evicted
                                // (the victim), not the definition that took its
                                // register.
                                self.spill_decisions.push(SpillDecision {
                                    operand: OperandId(victim_key.1),
                                    register: TPURegister {
                                        reg_type: result.reg_type.clone(),
                                        index: physical,
                                        data_type: result.data_type,
                                        size: result.size,
                                    },
                                    spill_location: MemoryAddress {
                                        base: None,
                                        offset: (self.spill_decisions.len() * result.size.max(1))
                                            as i64,
                                        index: None,
                                        scale: 1,
                                        memory_space: MemorySpace::Local,
                                    },
                                    cost: result.size as f64 * (1.0 + bandwidth_pressure),
                                });
                                active.push((key, physical, end));
                                self.assignments.insert(
                                    operand,
                                    TPURegister {
                                        reg_type: result.reg_type.clone(),
                                        index: physical,
                                        data_type: result.data_type,
                                        size: result.size,
                                    },
                                );
                            }
                            _ => {
                                // This value itself is the cheapest to spill.
                                self.spill_decisions.push(SpillDecision {
                                    operand,
                                    register: result.clone(),
                                    spill_location: MemoryAddress {
                                        base: None,
                                        offset: (self.spill_decisions.len() * result.size.max(1))
                                            as i64,
                                        index: None,
                                        scale: 1,
                                        memory_space: MemorySpace::Local,
                                    },
                                    cost: result.size as f64 * (1.0 + bandwidth_pressure),
                                });
                            }
                        }
                    }
                }
            }

            // Record the pressure actually observed at this point.
            let mut pressure_by_type: HashMap<RegisterType, usize> = HashMap::new();
            for (key, _, _) in &active {
                *pressure_by_type.entry(key.0.clone()).or_insert(0) += 1;
            }
            let total_pressure = pressure_by_type.values().sum();
            let spill_cost = self
                .spill_decisions
                .iter()
                .map(|decision| decision.cost)
                .sum();
            self.pressure_tracking.insert(
                position as u64,
                RegisterPressure {
                    pressure_by_type,
                    total_pressure,
                    spill_cost,
                },
            );
        }

        Ok(())
    }

    /// Physical register chosen for the value produced by an instruction.
    pub fn assignment(&self, operand: OperandId) -> Option<&TPURegister> {
        self.assignments.get(&operand)
    }

    /// Spill decisions made during the last allocation.
    pub fn spills(&self) -> &[SpillDecision] {
        &self.spill_decisions
    }

    /// Highest register pressure observed during the last allocation.
    pub fn peak_pressure(&self) -> usize {
        self.pressure_tracking
            .values()
            .map(|pressure| pressure.total_pressure)
            .max()
            .unwrap_or(0)
    }

    /// Reset allocator state
    pub fn reset(&mut self) {
        self.assignments.clear();
        self.pressure_tracking.clear();
        self.spill_decisions.clear();
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> InstructionScheduler<T> {
    /// Create new instruction scheduler
    pub fn new(_target_config: &TPUConfig) -> Self {
        Self {
            strategy: SchedulingStrategy::CriticalPath,
            resource_model: ResourceModel {
                // A real (if simplified) machine model: the units a TPU core
                // actually has, with the opcodes each can retire and their
                // latencies. An empty model gave the scheduler nothing to
                // schedule against.
                execution_units: vec![
                    ExecutionUnit {
                        name: "matrix".to_string(),
                        supported_ops: vec![TPUOpcode::MatMul, TPUOpcode::MatMulAccumulate],
                        latency: 8,
                        throughput: 1.0,
                    },
                    ExecutionUnit {
                        name: "vector".to_string(),
                        supported_ops: vec![
                            TPUOpcode::VectorAdd,
                            TPUOpcode::VectorMultiply,
                            TPUOpcode::VectorDot,
                        ],
                        latency: 2,
                        throughput: 2.0,
                    },
                    ExecutionUnit {
                        name: "scalar".to_string(),
                        supported_ops: vec![TPUOpcode::ScalarAdd, TPUOpcode::ScalarMultiply],
                        latency: 1,
                        throughput: 4.0,
                    },
                ],
            },
            dependency_graph: InstructionDependencyGraph {
                dependencies: HashMap::new(),
                dependency_types: HashMap::new(),
                critical_path: vec![],
            },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Schedule instructions into dependency-respecting order.
    ///
    /// This builds the real dependency graph (read-after-write, write-after-read
    /// and write-after-write over the virtual registers the instruction
    /// generator emitted), derives the critical path from it using the resource
    /// model's per-opcode latencies, and then list-schedules the ready set.
    /// It replaces a body that returned the input untouched, which left
    /// `strategy`, `resource_model` and `dependency_graph` permanently unread.
    ///
    /// The produced instructions carry their real `scheduling_info`
    /// (`earliest_cycle` from the dependency height, `scheduled_cycle` from the
    /// chosen order, and the dependency list), so a later pass can see the
    /// schedule rather than having to recompute it.
    pub fn schedule_instructions(
        &mut self,
        instructions: &[TPUInstruction],
    ) -> Result<Vec<TPUInstruction>> {
        self.build_dependency_graph(instructions);

        let order: Vec<usize> = match &self.strategy {
            SchedulingStrategy::List => self.list_schedule(instructions, false),
            SchedulingStrategy::CriticalPath => self.list_schedule(instructions, true),
            // These need loop structure / trace profiling that the instruction
            // stream alone does not carry. Saying so is better than silently
            // falling back to a different algorithm than the caller asked for.
            SchedulingStrategy::SoftwarePipelining | SchedulingStrategy::Trace => {
                return Err(OptimError::from(format!(
                    "instruction scheduling strategy {:?} is not implemented",
                    self.strategy
                )))
            }
        };

        let by_position: HashMap<usize, &TPUInstruction> = instructions
            .iter()
            .map(|instruction| (instruction.id, instruction))
            .collect();

        let mut scheduled = Vec::with_capacity(order.len());
        let mut cycle: u64 = 0;
        for id in &order {
            let Some(source) = by_position.get(id) else {
                continue;
            };
            let mut instruction = (*source).clone();
            let dependencies = self
                .dependency_graph
                .dependencies
                .get(id)
                .cloned()
                .unwrap_or_default();
            instruction.scheduling_info.dependencies = dependencies;
            instruction.scheduling_info.earliest_cycle = cycle;
            instruction.scheduling_info.scheduled_cycle = Some(cycle);
            cycle += self.opcode_latency(&instruction.opcode) as u64;
            instruction.scheduling_info.latest_cycle = cycle;
            scheduled.push(instruction);
        }

        Ok(scheduled)
    }

    /// The dependency graph produced by the last [`Self::schedule_instructions`]
    /// call.
    pub fn dependency_graph(&self) -> &InstructionDependencyGraph {
        &self.dependency_graph
    }

    /// Select the scheduling strategy.
    pub fn set_strategy(&mut self, strategy: SchedulingStrategy) {
        self.strategy = strategy;
    }

    /// Latency of an opcode according to the resource model, defaulting to one
    /// cycle for opcodes no modelled unit claims.
    fn opcode_latency(&self, opcode: &TPUOpcode) -> u32 {
        self.resource_model
            .execution_units
            .iter()
            .find(|unit| unit.supported_ops.contains(opcode))
            .map(|unit| unit.latency)
            .unwrap_or(1)
    }

    /// Build RAW/WAR/WAW dependencies over the virtual registers.
    fn build_dependency_graph(&mut self, instructions: &[TPUInstruction]) {
        self.dependency_graph.dependencies.clear();
        self.dependency_graph.dependency_types.clear();
        self.dependency_graph.critical_path.clear();

        let mut last_writer: HashMap<(RegisterType, usize), usize> = HashMap::new();
        let mut readers_since_write: HashMap<(RegisterType, usize), Vec<usize>> = HashMap::new();

        for instruction in instructions {
            let mut predecessors: Vec<usize> = Vec::new();

            // Read-after-write on every register this instruction reads.
            for operand in &instruction.operands {
                for register in operand_registers(operand) {
                    let key = (register.reg_type.clone(), register.index);
                    if let Some(writer) = last_writer.get(&key) {
                        if *writer != instruction.id {
                            predecessors.push(*writer);
                            self.dependency_graph
                                .dependency_types
                                .insert((*writer, instruction.id), DependencyType::True);
                        }
                    }
                    readers_since_write
                        .entry(key)
                        .or_default()
                        .push(instruction.id);
                }
            }

            if let Some(result) = &instruction.result {
                let key = (result.reg_type.clone(), result.index);
                // Write-after-write against the previous definition.
                if let Some(writer) = last_writer.get(&key) {
                    if *writer != instruction.id {
                        predecessors.push(*writer);
                        self.dependency_graph
                            .dependency_types
                            .insert((*writer, instruction.id), DependencyType::Output);
                    }
                }
                // Write-after-read against everything that read the old value.
                if let Some(readers) = readers_since_write.get(&key) {
                    for reader in readers {
                        if *reader != instruction.id {
                            predecessors.push(*reader);
                            self.dependency_graph
                                .dependency_types
                                .insert((*reader, instruction.id), DependencyType::Anti);
                        }
                    }
                }
                last_writer.insert(key.clone(), instruction.id);
                readers_since_write.remove(&key);
            }

            predecessors.sort_unstable();
            predecessors.dedup();
            self.dependency_graph
                .dependencies
                .insert(instruction.id, predecessors);
        }

        self.dependency_graph.critical_path = self.compute_critical_path(instructions);
    }

    /// Longest latency-weighted chain through the dependency graph.
    fn compute_critical_path(&self, instructions: &[TPUInstruction]) -> Vec<usize> {
        // Heights over the DAG, computed in the instruction stream's order,
        // which is already a topological order of the dependencies built above.
        let mut height: HashMap<usize, u32> = HashMap::new();
        let mut best_predecessor: HashMap<usize, usize> = HashMap::new();
        let mut deepest: Option<(usize, u32)> = None;

        for instruction in instructions {
            let latency = self.opcode_latency(&instruction.opcode);
            let mut base = 0u32;
            if let Some(predecessors) = self.dependency_graph.dependencies.get(&instruction.id) {
                for predecessor in predecessors {
                    if let Some(candidate) = height.get(predecessor) {
                        if *candidate > base {
                            base = *candidate;
                            best_predecessor.insert(instruction.id, *predecessor);
                        }
                    }
                }
            }
            let total = base + latency;
            height.insert(instruction.id, total);
            match deepest {
                Some((_, best)) if best >= total => {}
                _ => deepest = Some((instruction.id, total)),
            }
        }

        let mut path = Vec::new();
        let mut cursor = deepest.map(|(id, _)| id);
        while let Some(id) = cursor {
            path.push(id);
            cursor = best_predecessor.get(&id).copied();
        }
        path.reverse();
        path
    }

    /// List-schedule the ready set. With `prefer_critical_path` the ready
    /// instruction on the critical path is issued first; otherwise ties are
    /// broken by instruction id, which reproduces program order.
    fn list_schedule(
        &self,
        instructions: &[TPUInstruction],
        prefer_critical_path: bool,
    ) -> Vec<usize> {
        let critical: HashMap<usize, usize> = self
            .dependency_graph
            .critical_path
            .iter()
            .enumerate()
            .map(|(position, id)| (*id, position))
            .collect();

        let mut remaining: Vec<usize> = instructions
            .iter()
            .map(|instruction| instruction.id)
            .collect();
        let mut issued: HashSet<usize> = HashSet::new();
        let mut order = Vec::with_capacity(remaining.len());

        while !remaining.is_empty() {
            let mut ready: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|id| {
                    self.dependency_graph
                        .dependencies
                        .get(id)
                        .map(|predecessors| {
                            predecessors
                                .iter()
                                .all(|predecessor| issued.contains(predecessor))
                        })
                        .unwrap_or(true)
                })
                .collect();

            if ready.is_empty() {
                // A dependency cycle cannot happen for a straight-line stream,
                // but if one ever did, issuing the remainder in program order is
                // still a valid (if unscheduled) answer -- better than looping.
                ready = remaining.clone();
            }

            if prefer_critical_path {
                ready.sort_by_key(|id| (critical.get(id).copied().unwrap_or(usize::MAX), *id));
            } else {
                ready.sort_unstable();
            }

            let Some(next) = ready.first().copied() else {
                break;
            };
            issued.insert(next);
            order.push(next);
            remaining.retain(|id| *id != next);
        }

        order
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> Default
    for CodeOptimizer<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> CodeOptimizer<T> {
    /// Create new code optimizer
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            pass_stats: HashMap::new(),
        }
    }

    /// Run every registered pass over the generated code, recording per-pass
    /// statistics.
    ///
    /// The registry is empty unless a caller adds a pass, so this is a real
    /// extension point: it applies what is registered and reports what each pass
    /// changed, instead of the previous body that ignored `passes` and
    /// `pass_stats` entirely.
    pub fn optimize(&mut self, code: &mut GeneratedCode) -> Result<()> {
        for pass in &self.passes {
            if !pass.is_applicable(code) {
                continue;
            }
            let lines_before = code.kernel_code.lines().count();
            let changed = pass.optimize(code)?;
            let lines_after = code.kernel_code.lines().count();
            let entry = self.pass_stats.entry(pass.name().to_string()).or_default();
            if changed {
                entry.instructions_eliminated += lines_before.saturating_sub(lines_after);
                entry.cycles_saved += lines_before.saturating_sub(lines_after) as u64;
            }
        }
        Ok(())
    }

    /// Register a code-level optimization pass.
    pub fn register_pass(&mut self, pass: Box<dyn CodeOptimizationPass<T>>) {
        self.passes.push(pass);
    }

    /// Per-pass statistics from the last [`Self::optimize`] call onwards.
    pub fn pass_statistics(&self) -> &HashMap<String, OptimizationStats> {
        &self.pass_stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpu_code_generator_creation() {
        use super::super::super::{super::PodTopology, TPUConfig, TPUVersion};

        let tpu_config = TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology: PodTopology::Pod2x2,
            memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        };

        let generator: TPUCodeGenerator<f32> = TPUCodeGenerator::new(tpu_config);
        assert_eq!(generator.generation_stats.instructions_generated, 0);
        assert_eq!(generator.generation_stats.kernels_generated, 0);
    }

    #[test]
    fn test_tpu_instruction_creation() {
        let instruction = TPUInstruction {
            id: 0,
            opcode: TPUOpcode::VectorAdd,
            operands: vec![
                TPUOperand::Register(TPURegister {
                    reg_type: RegisterType::Vector,
                    index: 0,
                    data_type: DataType::F32,
                    size: 4,
                }),
                TPUOperand::Register(TPURegister {
                    reg_type: RegisterType::Vector,
                    index: 1,
                    data_type: DataType::F32,
                    size: 4,
                }),
            ],
            result: Some(TPURegister {
                reg_type: RegisterType::Vector,
                index: 2,
                data_type: DataType::F32,
                size: 4,
            }),
            attributes: InstructionAttributes::default(),
            scheduling_info: SchedulingInfo::default(),
        };

        assert_eq!(instruction.opcode, TPUOpcode::VectorAdd);
        assert_eq!(instruction.operands.len(), 2);
        assert!(instruction.result.is_some());
    }

    // -----------------------------------------------------------------------
    // Register allocation
    // -----------------------------------------------------------------------

    fn vector_reg(index: usize) -> TPURegister {
        TPURegister {
            reg_type: RegisterType::Vector,
            index,
            data_type: DataType::F32,
            size: 4,
        }
    }

    /// `id`-th instruction: `dest = src_a + src_b` on vector registers.
    fn vector_add(id: usize, src_a: usize, src_b: usize, dest: usize) -> TPUInstruction {
        TPUInstruction {
            id,
            opcode: TPUOpcode::VectorAdd,
            operands: vec![
                TPUOperand::Register(vector_reg(src_a)),
                TPUOperand::Register(vector_reg(src_b)),
            ],
            result: Some(vector_reg(dest)),
            attributes: InstructionAttributes::default(),
            scheduling_info: SchedulingInfo::default(),
        }
    }

    fn empty_plan() -> MemoryPlan<f32> {
        MemoryPlan::empty()
    }

    /// A short program fits in the pool: every definition gets a physical
    /// register, nothing spills, and the recorded pressure is real.
    #[test]
    fn register_allocation_assigns_and_tracks_pressure() {
        let config = test_tpu_config();
        let mut allocator = RegisterAllocator::new(&config);
        let plan = empty_plan();

        // v2 = v0 + v1 ; v3 = v2 + v2 ; v4 = v3 + v3
        let instructions = vec![
            vector_add(0, 0, 1, 2),
            vector_add(1, 2, 2, 3),
            vector_add(2, 3, 3, 4),
        ];
        allocator
            .allocate_registers(&instructions, &plan)
            .expect("allocation must succeed");

        assert!(
            allocator.spills().is_empty(),
            "64 vector registers is plenty"
        );
        for id in 0..3usize {
            assert!(
                allocator.assignment(OperandId(id)).is_some(),
                "instruction {id} produced a value and must own a register"
            );
        }
        assert!(allocator.peak_pressure() >= 1);
        // v2's interval ends at instruction 1, so its register is reusable
        // afterwards: pressure must not simply equal the definition count.
        assert!(allocator.peak_pressure() <= 3);
    }

    /// More simultaneously-live values than the pool has registers forces real
    /// spill decisions rather than silently over-allocating.
    #[test]
    fn register_allocation_spills_when_the_pool_runs_out() {
        let config = test_tpu_config();
        let mut allocator = RegisterAllocator::new(&config);
        let plan = empty_plan();

        // 80 definitions (v100..v179), then 80 consumers reading them back in
        // reverse order. Every definition therefore stays live until its
        // consumer, so all 80 are simultaneously live at the last definition --
        // against a 64-entry vector pool.
        let mut instructions: Vec<TPUInstruction> =
            (0..80).map(|id| vector_add(id, 0, 0, id + 100)).collect();
        for j in 0..80usize {
            let source = 179 - j;
            instructions.push(vector_add(80 + j, source, source, 300 + j));
        }

        allocator
            .allocate_registers(&instructions, &plan)
            .expect("allocation must succeed");

        assert!(
            !allocator.spills().is_empty(),
            "80 simultaneously live values cannot fit in 64 vector registers"
        );
        assert!(
            allocator.peak_pressure() <= 64,
            "pressure must never exceed the pool size, saw {}",
            allocator.peak_pressure()
        );
        for spill in allocator.spills() {
            assert!(spill.cost > 0.0, "a spill must carry a real cost");
        }
    }

    // -----------------------------------------------------------------------
    // Instruction scheduling
    // -----------------------------------------------------------------------

    /// The scheduler must respect read-after-write order and stamp real
    /// scheduling information onto every instruction.
    #[test]
    fn scheduling_respects_dependencies_and_stamps_cycles() {
        let config = test_tpu_config();
        let mut scheduler: InstructionScheduler<f32> = InstructionScheduler::new(&config);

        // v2 = v0 + v1 ; v3 = v2 + v2  (instruction 1 depends on instruction 0)
        let instructions = vec![vector_add(0, 0, 1, 2), vector_add(1, 2, 2, 3)];
        let scheduled = scheduler
            .schedule_instructions(&instructions)
            .expect("scheduling must succeed");

        assert_eq!(scheduled.len(), 2);
        assert_eq!(scheduled[0].id, 0, "the producer must issue first");
        assert_eq!(scheduled[1].id, 1);
        assert_eq!(scheduled[1].scheduling_info.dependencies, vec![0]);
        assert_eq!(scheduled[0].scheduling_info.scheduled_cycle, Some(0));
        assert!(
            scheduled[1].scheduling_info.earliest_cycle
                > scheduled[0].scheduling_info.earliest_cycle,
            "a dependent instruction cannot issue in the producer's cycle"
        );

        // The dependency graph is real, and so is the critical path.
        let graph = scheduler.dependency_graph();
        assert_eq!(graph.dependencies.get(&1), Some(&vec![0]));
        assert!(matches!(
            graph.dependency_types.get(&(0, 1)),
            Some(DependencyType::True)
        ));
        assert_eq!(graph.critical_path, vec![0, 1]);
    }

    /// Independent instructions keep program order and carry no dependencies.
    #[test]
    fn scheduling_leaves_independent_instructions_alone() {
        let config = test_tpu_config();
        let mut scheduler: InstructionScheduler<f32> = InstructionScheduler::new(&config);

        let instructions = vec![
            vector_add(0, 0, 1, 10),
            vector_add(1, 2, 3, 11),
            vector_add(2, 4, 5, 12),
        ];
        let scheduled = scheduler
            .schedule_instructions(&instructions)
            .expect("scheduling must succeed");

        let ids: Vec<usize> = scheduled.iter().map(|i| i.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        for instruction in &scheduled {
            assert!(instruction.scheduling_info.dependencies.is_empty());
        }
    }

    /// Strategies with no implementation say so instead of quietly doing
    /// something else.
    #[test]
    fn unimplemented_scheduling_strategies_report_an_error() {
        let config = test_tpu_config();
        let mut scheduler: InstructionScheduler<f32> = InstructionScheduler::new(&config);
        scheduler.set_strategy(SchedulingStrategy::SoftwarePipelining);

        let instructions = vec![vector_add(0, 0, 1, 2)];
        assert!(scheduler.schedule_instructions(&instructions).is_err());
    }

    fn test_tpu_config() -> super::super::super::TPUConfig {
        use super::super::super::{super::PodTopology, TPUConfig, TPUVersion};
        TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology: PodTopology::Pod2x2,
            memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        }
    }
}
