// Architecture search constraints and hardware specifications
//
// This module defines the constraints and hardware requirements for
// neural architecture search, including complexity limits, hardware
// constraints, and performance requirements.

/// Search constraints
#[derive(Debug, Clone, Default)]
pub struct SearchConstraints {
    /// Maximum parameters
    pub max_parameters: usize,

    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,

    /// Maximum inference time (ms)
    pub max_inference_time_ms: u64,

    /// Minimum accuracy threshold
    pub min_accuracy: f64,

    /// Architecture complexity constraints
    pub complexity_constraints: ComplexityConstraints,

    /// Hardware constraints
    pub hardware_constraints: HardwareConstraints,
}

/// Architecture complexity constraints
#[derive(Debug, Clone, Default)]
pub struct ComplexityConstraints {
    /// Maximum depth
    pub max_depth: usize,

    /// Maximum width
    pub max_width: usize,

    /// Maximum connections
    pub max_connections: usize,

    /// Minimum efficiency ratio
    pub min_efficiency: f64,
}

/// Hardware-specific constraints
#[derive(Debug, Clone, Default)]
pub struct HardwareConstraints {
    /// Target hardware type
    pub target_hardware: TargetHardware,

    /// Memory bandwidth requirements
    pub memory_bandwidth_gb_s: f64,

    /// Compute capability requirements
    pub compute_capability: ComputeCapability,

    /// Power consumption limits
    pub max_power_watts: f64,
}

/// Target hardware types
#[derive(Debug, Clone, Copy, Default)]
pub enum TargetHardware {
    #[default]
    CPU,
    GPU,
    TPU,
    Mobile,
    Edge,
    Custom,
}

/// Compute capability requirements
#[derive(Debug, Clone, Default)]
pub struct ComputeCapability {
    /// FLOPS requirement
    pub flops: u64,

    /// Specialized units needed
    pub specialized_units: Vec<SpecializedUnit>,

    /// Parallelization level
    pub parallelization_level: usize,
}

/// Specialized computing units
#[derive(Debug, Clone, Copy, Default)]
pub enum SpecializedUnit {
    #[default]
    MatrixMultiplication,
    TensorCores,
    VectorProcessing,
    CustomAccelerator,
}

// Implementation methods
impl SearchConstraints {
    /// Create constraints for mobile deployment
    pub fn mobile_constraints() -> Self {
        Self {
            max_parameters: 1_000_000,
            max_memory_mb: 100,
            max_inference_time_ms: 50,
            min_accuracy: 0.7,
            complexity_constraints: ComplexityConstraints {
                max_depth: 10,
                max_width: 256,
                max_connections: 50,
                min_efficiency: 0.8,
            },
            hardware_constraints: HardwareConstraints {
                target_hardware: TargetHardware::Mobile,
                memory_bandwidth_gb_s: 10.0,
                compute_capability: ComputeCapability {
                    flops: 1_000_000_000,
                    specialized_units: vec![SpecializedUnit::VectorProcessing],
                    parallelization_level: 4,
                },
                max_power_watts: 2.0,
            },
        }
    }

    /// Create constraints for edge deployment
    pub fn edge_constraints() -> Self {
        Self {
            max_parameters: 500_000,
            max_memory_mb: 50,
            max_inference_time_ms: 20,
            min_accuracy: 0.6,
            complexity_constraints: ComplexityConstraints {
                max_depth: 8,
                max_width: 128,
                max_connections: 30,
                min_efficiency: 0.9,
            },
            hardware_constraints: HardwareConstraints {
                target_hardware: TargetHardware::Edge,
                memory_bandwidth_gb_s: 5.0,
                compute_capability: ComputeCapability {
                    flops: 500_000_000,
                    specialized_units: vec![SpecializedUnit::VectorProcessing],
                    parallelization_level: 2,
                },
                max_power_watts: 1.0,
            },
        }
    }

    /// Create constraints for GPU deployment
    pub fn gpu_constraints() -> Self {
        Self {
            max_parameters: 100_000_000,
            max_memory_mb: 8000,
            max_inference_time_ms: 100,
            min_accuracy: 0.95,
            complexity_constraints: ComplexityConstraints {
                max_depth: 50,
                max_width: 2048,
                max_connections: 1000,
                min_efficiency: 0.7,
            },
            hardware_constraints: HardwareConstraints {
                target_hardware: TargetHardware::GPU,
                memory_bandwidth_gb_s: 500.0,
                compute_capability: ComputeCapability {
                    flops: 100_000_000_000,
                    specialized_units: vec![
                        SpecializedUnit::MatrixMultiplication,
                        SpecializedUnit::TensorCores,
                    ],
                    parallelization_level: 1000,
                },
                max_power_watts: 300.0,
            },
        }
    }

    /// Create constraints for TPU deployment
    pub fn tpu_constraints() -> Self {
        Self {
            max_parameters: 500_000_000,
            max_memory_mb: 32000,
            max_inference_time_ms: 200,
            min_accuracy: 0.98,
            complexity_constraints: ComplexityConstraints {
                max_depth: 100,
                max_width: 4096,
                max_connections: 5000,
                min_efficiency: 0.6,
            },
            hardware_constraints: HardwareConstraints {
                target_hardware: TargetHardware::TPU,
                memory_bandwidth_gb_s: 1000.0,
                compute_capability: ComputeCapability {
                    flops: 1_000_000_000_000,
                    specialized_units: vec![
                        SpecializedUnit::MatrixMultiplication,
                        SpecializedUnit::TensorCores,
                    ],
                    parallelization_level: 2048,
                },
                max_power_watts: 250.0,
            },
        }
    }

    /// Validate architecture against constraints
    pub fn validate_architecture(&self, spec: &super::specifications::ArchitectureSpec) -> Result<(), ConstraintViolation> {
        // Check parameter count
        let param_count = spec.parameter_count();
        if param_count > self.max_parameters {
            return Err(ConstraintViolation::ParameterLimitExceeded(
                param_count,
                self.max_parameters,
            ));
        }

        // Check memory usage
        let memory_usage_mb = spec.memory_usage_estimate() / (1024 * 1024);
        if memory_usage_mb > self.max_memory_mb {
            return Err(ConstraintViolation::MemoryLimitExceeded(
                memory_usage_mb,
                self.max_memory_mb,
            ));
        }

        // Check complexity constraints
        self.complexity_constraints.validate_architecture(spec)?;

        Ok(())
    }

    /// Check if constraints are compatible with target hardware
    pub fn is_compatible_with_hardware(&self, hardware: TargetHardware) -> bool {
        match (self.hardware_constraints.target_hardware, hardware) {
            (TargetHardware::Custom, _) | (_, TargetHardware::Custom) => true,
            (a, b) => a == b,
        }
    }

    /// Get resource scaling factor for target hardware
    pub fn get_resource_scaling(&self) -> f64 {
        match self.hardware_constraints.target_hardware {
            TargetHardware::CPU => 1.0,
            TargetHardware::GPU => 0.3, // GPUs are more efficient for parallel workloads
            TargetHardware::TPU => 0.1, // TPUs are highly optimized
            TargetHardware::Mobile => 2.0, // Mobile has resource constraints
            TargetHardware::Edge => 3.0,   // Edge devices have severe constraints
            TargetHardware::Custom => 1.0,
        }
    }
}

impl ComplexityConstraints {
    /// Validate architecture complexity
    pub fn validate_architecture(&self, spec: &super::specifications::ArchitectureSpec) -> Result<(), ConstraintViolation> {
        // Check depth
        if spec.layers.len() > self.max_depth {
            return Err(ConstraintViolation::DepthLimitExceeded(
                spec.layers.len(),
                self.max_depth,
            ));
        }

        // Check width
        let max_layer_width = spec
            .layers
            .iter()
            .map(|layer| layer.dimensions.output_dim.max(layer.dimensions.input_dim))
            .max()
            .unwrap_or(0);

        if max_layer_width > self.max_width {
            return Err(ConstraintViolation::WidthLimitExceeded(
                max_layer_width,
                self.max_width,
            ));
        }

        // Check connections
        let connection_count = spec.connections.iter().filter(|&&x| x).count();
        if connection_count > self.max_connections {
            return Err(ConstraintViolation::ConnectionLimitExceeded(
                connection_count,
                self.max_connections,
            ));
        }

        Ok(())
    }

    /// Calculate efficiency ratio for architecture
    pub fn calculate_efficiency(&self, spec: &super::specifications::ArchitectureSpec) -> f64 {
        let param_efficiency = 1.0 / (spec.parameter_count() as f64).log10().max(1.0);
        let memory_efficiency = 1.0 / (spec.memory_usage_estimate() as f64).log10().max(1.0);
        let connection_efficiency = 1.0 / ((spec.connections.iter().filter(|&&x| x).count() as f64) + 1.0).log10();

        (param_efficiency + memory_efficiency + connection_efficiency) / 3.0
    }
}

impl HardwareConstraints {
    /// Check if hardware supports required capabilities
    pub fn supports_capabilities(&self) -> bool {
        match self.target_hardware {
            TargetHardware::CPU => {
                self.compute_capability.flops <= 100_000_000_000
                    && self.max_power_watts <= 100.0
            }
            TargetHardware::GPU => {
                self.compute_capability.flops <= 1_000_000_000_000
                    && self.max_power_watts <= 400.0
            }
            TargetHardware::TPU => {
                self.compute_capability.flops <= 10_000_000_000_000
                    && self.max_power_watts <= 300.0
            }
            TargetHardware::Mobile => {
                self.compute_capability.flops <= 10_000_000_000
                    && self.max_power_watts <= 5.0
            }
            TargetHardware::Edge => {
                self.compute_capability.flops <= 1_000_000_000
                    && self.max_power_watts <= 2.0
            }
            TargetHardware::Custom => true,
        }
    }

    /// Get optimal batch size for hardware
    pub fn optimal_batch_size(&self) -> usize {
        match self.target_hardware {
            TargetHardware::CPU => 32,
            TargetHardware::GPU => 256,
            TargetHardware::TPU => 1024,
            TargetHardware::Mobile => 8,
            TargetHardware::Edge => 4,
            TargetHardware::Custom => 64,
        }
    }

    /// Get memory hierarchy characteristics
    pub fn memory_hierarchy(&self) -> MemoryHierarchy {
        match self.target_hardware {
            TargetHardware::CPU => MemoryHierarchy {
                l1_cache_kb: 64,
                l2_cache_kb: 512,
                l3_cache_kb: 8192,
                main_memory_gb: 32,
            },
            TargetHardware::GPU => MemoryHierarchy {
                l1_cache_kb: 128,
                l2_cache_kb: 4096,
                l3_cache_kb: 0,
                main_memory_gb: 16,
            },
            TargetHardware::TPU => MemoryHierarchy {
                l1_cache_kb: 256,
                l2_cache_kb: 8192,
                l3_cache_kb: 0,
                main_memory_gb: 32,
            },
            TargetHardware::Mobile => MemoryHierarchy {
                l1_cache_kb: 32,
                l2_cache_kb: 256,
                l3_cache_kb: 0,
                main_memory_gb: 8,
            },
            TargetHardware::Edge => MemoryHierarchy {
                l1_cache_kb: 16,
                l2_cache_kb: 128,
                l3_cache_kb: 0,
                main_memory_gb: 2,
            },
            TargetHardware::Custom => MemoryHierarchy {
                l1_cache_kb: 64,
                l2_cache_kb: 512,
                l3_cache_kb: 4096,
                main_memory_gb: 16,
            },
        }
    }
}

impl TargetHardware {
    /// Get typical FLOPS for hardware type
    pub fn typical_flops(&self) -> u64 {
        match self {
            TargetHardware::CPU => 100_000_000_000,      // 100 GFLOPS
            TargetHardware::GPU => 10_000_000_000_000,   // 10 TFLOPS
            TargetHardware::TPU => 100_000_000_000_000,  // 100 TFLOPS
            TargetHardware::Mobile => 10_000_000_000,    // 10 GFLOPS
            TargetHardware::Edge => 1_000_000_000,       // 1 GFLOPS
            TargetHardware::Custom => 50_000_000_000,    // 50 GFLOPS
        }
    }

    /// Get typical power consumption
    pub fn typical_power_watts(&self) -> f64 {
        match self {
            TargetHardware::CPU => 65.0,
            TargetHardware::GPU => 250.0,
            TargetHardware::TPU => 200.0,
            TargetHardware::Mobile => 2.0,
            TargetHardware::Edge => 0.5,
            TargetHardware::Custom => 100.0,
        }
    }
}

impl SpecializedUnit {
    /// Get performance multiplier for unit type
    pub fn performance_multiplier(&self) -> f64 {
        match self {
            SpecializedUnit::MatrixMultiplication => 10.0,
            SpecializedUnit::TensorCores => 20.0,
            SpecializedUnit::VectorProcessing => 5.0,
            SpecializedUnit::CustomAccelerator => 15.0,
        }
    }

    /// Check if unit is available on hardware
    pub fn available_on_hardware(&self, hardware: TargetHardware) -> bool {
        match (self, hardware) {
            (SpecializedUnit::MatrixMultiplication, _) => true,
            (SpecializedUnit::TensorCores, TargetHardware::GPU | TargetHardware::TPU) => true,
            (SpecializedUnit::VectorProcessing, _) => true,
            (SpecializedUnit::CustomAccelerator, TargetHardware::Custom) => true,
            _ => false,
        }
    }
}

/// Memory hierarchy characteristics
#[derive(Debug, Clone)]
pub struct MemoryHierarchy {
    pub l1_cache_kb: usize,
    pub l2_cache_kb: usize,
    pub l3_cache_kb: usize,
    pub main_memory_gb: usize,
}

/// Constraint violation errors
#[derive(Debug, Clone)]
pub enum ConstraintViolation {
    ParameterLimitExceeded(usize, usize),
    MemoryLimitExceeded(usize, usize),
    InferenceTimeLimitExceeded(u64, u64),
    AccuracyThresholdNotMet(f64, f64),
    DepthLimitExceeded(usize, usize),
    WidthLimitExceeded(usize, usize),
    ConnectionLimitExceeded(usize, usize),
    EfficiencyThresholdNotMet(f64, f64),
    HardwareNotSupported(TargetHardware),
    IncompatibleSpecializedUnits(Vec<SpecializedUnit>),
}

impl std::fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintViolation::ParameterLimitExceeded(actual, limit) => {
                write!(f, "Parameter limit exceeded: {} > {}", actual, limit)
            }
            ConstraintViolation::MemoryLimitExceeded(actual, limit) => {
                write!(f, "Memory limit exceeded: {} MB > {} MB", actual, limit)
            }
            ConstraintViolation::InferenceTimeLimitExceeded(actual, limit) => {
                write!(f, "Inference time limit exceeded: {} ms > {} ms", actual, limit)
            }
            ConstraintViolation::AccuracyThresholdNotMet(actual, threshold) => {
                write!(f, "Accuracy threshold not met: {} < {}", actual, threshold)
            }
            ConstraintViolation::DepthLimitExceeded(actual, limit) => {
                write!(f, "Depth limit exceeded: {} > {}", actual, limit)
            }
            ConstraintViolation::WidthLimitExceeded(actual, limit) => {
                write!(f, "Width limit exceeded: {} > {}", actual, limit)
            }
            ConstraintViolation::ConnectionLimitExceeded(actual, limit) => {
                write!(f, "Connection limit exceeded: {} > {}", actual, limit)
            }
            ConstraintViolation::EfficiencyThresholdNotMet(actual, threshold) => {
                write!(f, "Efficiency threshold not met: {} < {}", actual, threshold)
            }
            ConstraintViolation::HardwareNotSupported(hardware) => {
                write!(f, "Hardware not supported: {:?}", hardware)
            }
            ConstraintViolation::IncompatibleSpecializedUnits(units) => {
                write!(f, "Incompatible specialized units: {:?}", units)
            }
        }
    }
}

impl std::error::Error for ConstraintViolation {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_constraints() {
        let constraints = SearchConstraints::mobile_constraints();
        assert_eq!(constraints.hardware_constraints.target_hardware, TargetHardware::Mobile);
        assert!(constraints.max_parameters <= 1_000_000);
    }

    #[test]
    fn test_hardware_capabilities() {
        let gpu_constraints = SearchConstraints::gpu_constraints();
        assert!(gpu_constraints.hardware_constraints.supports_capabilities());
        
        let edge_constraints = SearchConstraints::edge_constraints();
        assert!(edge_constraints.hardware_constraints.supports_capabilities());
    }

    #[test]
    fn test_specialized_unit_availability() {
        assert!(SpecializedUnit::MatrixMultiplication.available_on_hardware(TargetHardware::CPU));
        assert!(SpecializedUnit::TensorCores.available_on_hardware(TargetHardware::GPU));
        assert!(!SpecializedUnit::TensorCores.available_on_hardware(TargetHardware::CPU));
    }

    #[test]
    fn test_target_hardware_properties() {
        assert!(TargetHardware::GPU.typical_flops() > TargetHardware::CPU.typical_flops());
        assert!(TargetHardware::TPU.typical_flops() > TargetHardware::GPU.typical_flops());
    }

    #[test]
    fn test_constraint_compatibility() {
        let mobile_constraints = SearchConstraints::mobile_constraints();
        assert!(mobile_constraints.is_compatible_with_hardware(TargetHardware::Mobile));
        assert!(!mobile_constraints.is_compatible_with_hardware(TargetHardware::GPU));
    }
}