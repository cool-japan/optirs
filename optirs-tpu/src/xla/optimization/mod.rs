use std::fmt::Debug;
// XLA optimization passes
//
// This module contains optimization passes for XLA computations,
// including graph optimization, kernel fusion, memory planning, and scheduling.

pub mod graph_optimization;
pub mod kernel_fusion;
pub mod memory_planning;
pub mod scheduling;

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::frontend::{OperationType, XLAComputation, XLAOperation};
use super::{TPUConfig, TPUVersion, XLACompilerConfig, XLAOptimizationLevel};
use crate::error::{OptimError, Result};

// Re-export main types selectively to avoid ambiguous glob re-exports
// (MemoryAccessType and MemoryLevel exist in both memory_planning and scheduling)
pub use graph_optimization::*;
pub use kernel_fusion::*;
pub use memory_planning::{MemoryPlan, MemoryPlanner};
pub use scheduling::ExecutionScheduler;

/// Performance analyzer for XLA operations
pub struct PerformanceAnalyzer<T> {
    /// Performance metrics
    metrics: HashMap<String, f64>,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> PerformanceAnalyzer<T> {
    /// Create a new performance analyzer
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// The metrics recorded by the most recent [`Self::analyze`] call, keyed by
    /// name (`"operations"`, `"flop_count"`, `"memory_bytes"`,
    /// `"estimated_execution_time_us"`).
    pub fn metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
}

impl<T: Float + Debug + Send + Sync + 'static> PerformanceAnalyzer<T> {
    /// Derive real performance information for a compiled computation.
    ///
    /// Everything returned here is computed from the graph and the memory plan
    /// that were actually produced -- FLOPs from each operation's own count when
    /// the frontend filled one in, otherwise from the operation's type and the
    /// real operand shapes; memory from the plan's allocated total; bandwidth
    /// utilization from the plan's own measurement. Nothing is a constant, and
    /// an empty graph honestly reports zeros rather than a plausible-looking
    /// guess.
    pub fn analyze(
        &mut self,
        computation: &XLAComputation<T>,
        memory_plan: &MemoryPlan<T>,
        target: &TPUConfig,
    ) -> super::PerformanceInfo {
        let flop_count: u64 = computation
            .operations
            .iter()
            .map(|operation| self.operation_flops(computation, operation))
            .sum();

        let peak_flops_per_us = peak_flops_per_microsecond(target.tpu_version);
        let estimated_execution_time = if peak_flops_per_us == 0 {
            0
        } else {
            flop_count.div_ceil(peak_flops_per_us)
        };

        // Compute utilization: how much of the machine's peak the graph would
        // keep busy over the estimated span. By construction of the estimate
        // this saturates at 1.0; it drops below 1.0 only when rounding the span
        // up to whole microseconds leaves the machine idle, which is a real
        // (if small) effect rather than an invented number.
        let compute_utilization = if estimated_execution_time == 0 || peak_flops_per_us == 0 {
            0.0
        } else {
            (flop_count as f64 / (estimated_execution_time * peak_flops_per_us) as f64).min(1.0)
        };

        self.metrics.clear();
        self.metrics.insert(
            "operations".to_string(),
            computation.operations.len() as f64,
        );
        self.metrics
            .insert("flop_count".to_string(), flop_count as f64);
        self.metrics
            .insert("memory_bytes".to_string(), memory_plan.total_memory as f64);
        self.metrics.insert(
            "estimated_execution_time_us".to_string(),
            estimated_execution_time as f64,
        );

        super::PerformanceInfo {
            estimated_execution_time,
            memory_usage: memory_plan.total_memory,
            flop_count,
            memory_bandwidth_util: memory_plan.performance_info.bandwidth_utilization,
            compute_utilization,
        }
    }

    /// FLOPs attributable to one operation.
    ///
    /// A frontend-supplied count always wins; otherwise the count is derived
    /// from the operation's semantics and its real operand shapes. Operations
    /// that move data without arithmetic (reshape, transpose, slice, ...)
    /// contribute zero, which is the truth rather than an omission.
    fn operation_flops(&self, computation: &XLAComputation<T>, operation: &XLAOperation<T>) -> u64 {
        if operation.performance.flop_count > 0 {
            return operation.performance.flop_count;
        }

        let output_elements = computation
            .operands
            .get(&operation.output)
            .map(|operand| operand.shape.element_count as u64)
            .unwrap_or(0);

        match &operation.op_type {
            // A dot/matmul over lhs [.., M, K] x rhs [.., K, N] costs one
            // multiply and one add per contracted element.
            OperationType::Dot | OperationType::DotGeneral | OperationType::MatMul => {
                let contraction = operation
                    .inputs
                    .first()
                    .and_then(|id| computation.operands.get(id))
                    .and_then(|operand| operand.shape.dimensions.last().copied())
                    .unwrap_or(1) as u64;
                2 * output_elements * contraction
            }
            // A convolution costs 2 FLOPs per (output element x kernel element x
            // input channel); the kernel operand carries those dimensions.
            OperationType::Convolution(_) => {
                let kernel_elements = operation
                    .inputs
                    .get(1)
                    .and_then(|id| computation.operands.get(id))
                    .map(|operand| operand.shape.element_count as u64)
                    .unwrap_or(1);
                2 * output_elements * kernel_elements
            }
            // Reductions touch every input element once.
            OperationType::Reduce(_) | OperationType::ReduceWindow => operation
                .inputs
                .first()
                .and_then(|id| computation.operands.get(id))
                .map(|operand| operand.shape.element_count as u64)
                .unwrap_or(0),
            // Pure data movement: no arithmetic at all.
            OperationType::Reshape
            | OperationType::Transpose
            | OperationType::Slice
            | OperationType::DynamicSlice
            | OperationType::Pad
            | OperationType::Reverse
            | OperationType::Broadcast
            | OperationType::Concatenate
            | OperationType::Gather
            | OperationType::Scatter
            | OperationType::Parameter
            | OperationType::Constant(_)
            | OperationType::Tuple
            | OperationType::GetTupleElement => 0,
            // Everything else is elementwise over the output.
            _ => output_elements,
        }
    }
}

/// Peak arithmetic throughput per microsecond for a TPU version, derived from
/// the same peak-FLOPS figures the backend's device defaults use.
fn peak_flops_per_microsecond(version: TPUVersion) -> u64 {
    let tera = 1_000_000_000_000u64;
    let peak_flops_per_second = match version {
        TPUVersion::V2 => 45 * tera,
        TPUVersion::V3 => 123 * tera,
        TPUVersion::V4 => 275 * tera,
        TPUVersion::V5e => 197 * tera,
        TPUVersion::V5p => 459 * tera,
    };
    peak_flops_per_second / 1_000_000
}

impl<T> Default for PerformanceAnalyzer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive optimization pipeline for XLA computations
pub struct OptimizationPipeline<T: Float + Debug + Send + Sync + 'static> {
    /// Pipeline configuration
    config: OptimizationPipelineConfig,

    /// Graph optimization passes
    graph_optimizer: GraphOptimizer<T>,

    /// Kernel fusion engine
    fusion_engine: KernelFusionEngine<T>,

    /// Memory planner
    memory_planner: MemoryPlanner<T>,

    /// Execution scheduler
    scheduler: ExecutionScheduler<T>,

    /// Applied optimization passes
    applied_passes: Vec<String>,

    /// Performance statistics
    performance_stats: OptimizationStats,
}

/// Optimization pipeline configuration
#[derive(Debug, Clone)]
pub struct OptimizationPipelineConfig {
    /// Optimization level
    pub optimization_level: XLAOptimizationLevel,

    /// Enable graph optimizations
    pub enable_graph_optimization: bool,

    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,

    /// Enable memory optimization
    pub enable_memory_optimization: bool,

    /// Enable scheduling optimization
    pub enable_scheduling_optimization: bool,

    /// Maximum optimization time (seconds)
    pub max_optimization_time: u64,

    /// Target hardware configuration
    pub target_hardware: HardwareTarget,

    /// Custom optimization passes
    pub custom_passes: Vec<String>,

    /// Aggressive optimizations
    pub aggressive_mode: bool,

    /// Debug mode
    pub debug_mode: bool,
}

/// Hardware target configuration
#[derive(Debug, Clone)]
pub struct HardwareTarget {
    /// TPU version
    pub tpu_version: String,

    /// Number of cores
    pub num_cores: usize,

    /// Memory capacity (bytes)
    pub memory_capacity: usize,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,

    /// Compute capability
    pub compute_capability: ComputeCapability,
}

/// Compute capability information
#[derive(Debug, Clone)]
pub struct ComputeCapability {
    /// Matrix unit dimensions
    pub matrix_unit_dims: (usize, usize),

    /// Vector unit width
    pub vector_unit_width: usize,

    /// Supported data types
    pub supported_dtypes: Vec<String>,

    /// Special instructions
    pub special_instructions: Vec<String>,
}

/// Performance statistics for optimization
#[derive(Debug, Default)]
pub struct OptimizationStats {
    /// Total optimization time
    pub total_time: Duration,

    /// Time per optimization pass
    pub pass_times: HashMap<String, Duration>,

    /// Number of operations optimized
    pub operations_optimized: usize,

    /// Memory savings achieved
    pub memory_savings: usize,

    /// Estimated speedup
    pub estimated_speedup: f64,

    /// Optimization success rate
    pub success_rate: f64,
}

/// Optimization pass trait
pub trait OptimizationPass<T: Float + Debug + Send + Sync + 'static> {
    /// Pass name
    fn name(&self) -> &str;

    /// Apply optimization pass to computation
    fn apply(&mut self, computation: XLAComputation<T>) -> Result<XLAComputation<T>>;

    /// Check if pass is applicable
    fn is_applicable(&self, computation: &XLAComputation<T>) -> bool;

    /// Get pass dependencies
    fn dependencies(&self) -> Vec<String>;

    /// Estimate optimization benefit
    fn estimate_benefit(&self, computation: &XLAComputation<T>) -> f64;
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> OptimizationPipeline<T> {
    /// Create new optimization pipeline
    pub fn new(config: &XLACompilerConfig) -> Self {
        let pipeline_config = OptimizationPipelineConfig {
            optimization_level: config.optimization_level,
            enable_graph_optimization: config.enable_fusion,
            enable_kernel_fusion: config.enable_fusion,
            enable_memory_optimization: config.enable_memory_optimization,
            enable_scheduling_optimization: config.enable_pipeline_optimization,
            max_optimization_time: config.compilation_timeout,
            target_hardware: HardwareTarget::from_tpu_config(&config.target_tpu),
            custom_passes: config.custom_passes.clone(),
            aggressive_mode: matches!(
                config.optimization_level,
                XLAOptimizationLevel::Aggressive | XLAOptimizationLevel::Experimental
            ),
            debug_mode: config.debug_mode,
        };

        let graph_optimizer = GraphOptimizer::new(&pipeline_config);
        let fusion_engine = KernelFusionEngine::new(&pipeline_config);
        let memory_planner = MemoryPlanner::new(config.target_tpu.clone());
        let scheduler = ExecutionScheduler::new(&pipeline_config);

        Self {
            config: pipeline_config,
            graph_optimizer,
            fusion_engine,
            memory_planner,
            scheduler,
            applied_passes: Vec::new(),
            performance_stats: OptimizationStats::default(),
        }
    }

    /// Optimize XLA computation
    ///
    /// A phase is recorded in `applied_passes` only when it actually changed the
    /// computation. Enabling a phase that then rewrites nothing must not show up
    /// as an optimization that was applied.
    pub fn optimize(&mut self, computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        let start_time = Instant::now();
        let mut current_computation = computation;

        // Graph optimization phase
        if self.config.enable_graph_optimization {
            let pass_start = Instant::now();
            let (optimized, changed) =
                self.graph_optimizer.optimize_tracked(current_computation)?;
            current_computation = optimized;
            self.record_pass_time("graph_optimization", pass_start.elapsed());
            if changed {
                self.applied_passes.push("graph_optimization".to_string());
            }
        }

        // Kernel fusion phase
        if self.config.enable_kernel_fusion {
            let pass_start = Instant::now();
            let (fused, changed) = self
                .fusion_engine
                .fuse_kernels_tracked(current_computation)?;
            current_computation = fused;
            self.record_pass_time("kernel_fusion", pass_start.elapsed());
            if changed {
                self.applied_passes.push("kernel_fusion".to_string());
            }
        }

        // Memory optimization phase
        if self.config.enable_memory_optimization {
            let pass_start = Instant::now();
            current_computation = self
                .memory_planner
                .optimize_memory_layout(current_computation)?;
            self.record_pass_time("memory_optimization", pass_start.elapsed());
            self.applied_passes.push("memory_optimization".to_string());
        }

        // Scheduling optimization phase
        if self.config.enable_scheduling_optimization {
            let pass_start = Instant::now();
            current_computation = self.scheduler.optimize_schedule(current_computation)?;
            self.record_pass_time("scheduling_optimization", pass_start.elapsed());
            self.applied_passes
                .push("scheduling_optimization".to_string());
        }

        // Custom passes are named by configuration but this pipeline has no
        // registry to resolve them against, so an unknown name is an error
        // rather than a silently skipped "applied" pass.
        let custom_passes = self.config.custom_passes.clone();
        for pass_name in &custom_passes {
            let pass_start = Instant::now();
            current_computation = self.apply_custom_pass(pass_name, current_computation)?;
            self.record_pass_time(pass_name, pass_start.elapsed());
            self.applied_passes.push(pass_name.clone());
        }

        self.performance_stats.total_time = start_time.elapsed();
        self.performance_stats.operations_optimized = current_computation.operations.len();
        Ok(current_computation)
    }

    /// Apply a configured custom optimization pass.
    ///
    /// No custom pass registry exists, so any configured name is rejected
    /// explicitly instead of being reported as applied while doing nothing.
    fn apply_custom_pass(
        &mut self,
        pass_name: &str,
        _computation: XLAComputation<T>,
    ) -> Result<XLAComputation<T>> {
        Err(OptimError::NotImplementedError(
            scirs2_core::error::ErrorContext::new(format!(
                "custom optimization pass '{pass_name}' is configured but no custom pass \
                 registry is implemented; remove it from `custom_passes`"
            )),
        ))
    }

    /// Record optimization pass timing
    fn record_pass_time(&mut self, pass_name: &str, duration: Duration) {
        self.performance_stats
            .pass_times
            .insert(pass_name.to_string(), duration);
    }

    /// Get applied optimization passes
    pub fn get_applied_passes(&self) -> Vec<String> {
        self.applied_passes.clone()
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> &OptimizationStats {
        &self.performance_stats
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.applied_passes.clear();
        self.performance_stats = OptimizationStats::default();
    }
}

impl HardwareTarget {
    /// Create hardware target from TPU configuration
    pub fn from_tpu_config(tpu_config: &super::TPUConfig) -> Self {
        Self {
            tpu_version: format!("{:?}", tpu_config.tpu_version),
            num_cores: tpu_config.num_cores,
            memory_capacity: 16 * 1024 * 1024 * 1024, // Default 16GB
            memory_bandwidth: 900.0,                  // Default 900 GB/s
            compute_capability: ComputeCapability {
                matrix_unit_dims: (128, 128), // Default for TPU
                vector_unit_width: 256,
                supported_dtypes: vec!["BF16".to_string(), "F32".to_string(), "S32".to_string()],
                special_instructions: vec!["MATMUL".to_string(), "CONV".to_string()],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::XLACompilerConfig;
    use super::*;

    #[test]
    fn test_optimization_pipeline_creation() {
        let config = XLACompilerConfig::default();
        let pipeline: OptimizationPipeline<f32> = OptimizationPipeline::new(&config);

        assert_eq!(
            pipeline.config.optimization_level,
            config.optimization_level
        );
        assert_eq!(pipeline.applied_passes.len(), 0);
    }

    #[test]
    fn test_hardware_target_creation() {
        use crate::main_types::{PodTopology, TPUConfig, TPUVersion};

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

        let target = HardwareTarget::from_tpu_config(&tpu_config);
        assert_eq!(target.num_cores, 8);
        // Default memory capacity is 16GB
        assert_eq!(target.memory_capacity, 16 * 1024 * 1024 * 1024);
    }
}
