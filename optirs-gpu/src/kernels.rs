//! GPU Kernel Management and Compilation
//!
//! This module provides kernel management, compilation, and execution for GPU-accelerated
//! optimization algorithms. It supports CUDA, ROCm, Metal, and WebGPU backends.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

use crate::backends::{Backend, CompiledKernel, DeviceMemory, GpuBackend, KernelArg};

/// Kernel manager for compiling and executing GPU kernels
#[derive(Debug)]
pub struct KernelManager {
    /// Backend-specific kernel compiler
    compiler: Box<dyn KernelCompiler>,

    /// Compiled kernel cache
    kernel_cache: HashMap<String, Arc<CompiledKernel>>,

    /// Kernel templates
    templates: HashMap<String, KernelTemplate>,

    /// Backend type
    backend: GpuBackend,

    /// Compilation options
    compilation_options: CompilationOptions,
}

/// Kernel compilation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationOptions {
    /// Optimization level (0-3)
    pub optimization_level: u32,

    /// Enable fast math optimizations
    pub fast_math: bool,

    /// Enable debug information
    pub debug_info: bool,

    /// Target architecture
    pub target_arch: String,

    /// Additional compiler flags
    pub extra_flags: Vec<String>,

    /// Maximum register count per thread
    pub max_registers: Option<u32>,

    /// Shared memory size hint
    pub shared_memory_hint: Option<usize>,
}

/// Template for generating GPU kernels
#[derive(Debug, Clone)]
pub struct KernelTemplate {
    /// Template name
    pub name: String,

    /// Kernel source code template
    pub source_template: String,

    /// Parameter placeholders
    pub parameters: Vec<TemplateParameter>,

    /// Supported backends
    pub supported_backends: Vec<GpuBackend>,

    /// Default block and grid sizes
    pub default_launch_config: LaunchConfig,
}

/// Template parameter for kernel generation
#[derive(Debug, Clone)]
pub struct TemplateParameter {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub param_type: ParameterType,

    /// Default value (if any)
    pub default_value: Option<String>,

    /// Whether this parameter is required
    pub required: bool,
}

/// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid_size: (u32, u32, u32),

    /// Block dimensions (x, y, z)
    pub block_size: (u32, u32, u32),

    /// Shared memory size in bytes
    pub shared_memory_size: usize,

    /// CUDA stream (if applicable)
    pub stream: Option<u64>,
}

/// Parameter types for kernel templates
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterType {
    /// Integer type
    Integer,
    /// Floating point type
    Float,
    /// String/text type
    String,
    /// Data type specifier
    DataType,
    /// Boolean type
    Boolean,
}

/// Kernel types for different optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// SGD update kernel
    SGDUpdate,
    /// Adam update kernel
    AdamUpdate,
    /// AdamW update kernel
    AdamWUpdate,
    /// RMSprop update kernel
    RMSpropUpdate,
    /// AdaGrad update kernel
    AdaGradUpdate,
    /// Momentum update kernel
    MomentumUpdate,
    /// Gradient clipping kernel
    GradientClipping,
    /// Vector operations
    VectorOps,
    /// Matrix operations
    MatrixOps,
    /// Reduction operations
    Reduction,
}

/// Errors that can occur during kernel operations
#[derive(Debug, Error)]
pub enum KernelError {
    #[error("Compilation failed: {reason}")]
    CompilationFailed { reason: String },

    #[error("Kernel not found: {name}")]
    KernelNotFound { name: String },

    #[error("Invalid template parameter: {param}")]
    InvalidTemplateParameter { param: String },

    #[error("Backend not supported: {backend:?}")]
    BackendNotSupported { backend: GpuBackend },

    #[error("Launch configuration invalid: {reason}")]
    InvalidLaunchConfig { reason: String },

    #[error("Kernel execution failed: {reason}")]
    ExecutionFailed { reason: String },

    #[error("Template generation failed: {reason}")]
    TemplateGenerationFailed { reason: String },
}

/// Trait for backend-specific kernel compilation
pub trait KernelCompiler: Send + Sync {
    /// Compile kernel source code for the target backend
    fn compile_kernel(
        &self,
        source: &str,
        options: &CompilationOptions,
    ) -> Result<CompiledKernel, KernelError>;

    /// Get backend type
    fn backend_type(&self) -> GpuBackend;

    /// Check if a feature is supported
    fn supports_feature(&self, feature: KernelFeature) -> bool;

    /// Get compilation capabilities
    fn get_capabilities(&self) -> CompilerCapabilities;
}

/// Kernel features that may or may not be supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFeature {
    /// Half precision (f16) support
    HalfPrecision,
    /// Double precision (f64) support
    DoublePrecision,
    /// Tensor core operations
    TensorCores,
    /// Cooperative groups
    CooperativeGroups,
    /// Dynamic parallelism
    DynamicParallelism,
    /// Unified memory
    UnifiedMemory,
}

/// Compiler capabilities
#[derive(Debug, Clone)]
pub struct CompilerCapabilities {
    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Maximum blocks per grid
    pub max_blocks_per_grid: (u32, u32, u32),

    /// Maximum shared memory per block
    pub max_shared_memory: usize,

    /// Supported data types
    pub supported_data_types: Vec<String>,

    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
}

impl KernelManager {
    /// Create a new kernel manager for the specified backend
    pub fn new(backend: GpuBackend, options: CompilationOptions) -> Result<Self, KernelError> {
        let compiler = Self::create_compiler(backend)?;
        let templates = Self::load_builtin_templates();

        Ok(Self {
            compiler,
            kernel_cache: HashMap::new(),
            templates,
            backend,
            compilation_options: options,
        })
    }

    /// Create a backend-specific compiler
    fn create_compiler(backend: GpuBackend) -> Result<Box<dyn KernelCompiler>, KernelError> {
        match backend {
            GpuBackend::Cuda => Ok(Box::new(CudaCompiler::new())),
            GpuBackend::Rocm => Ok(Box::new(RocmCompiler::new())),
            GpuBackend::Metal => Ok(Box::new(MetalCompiler::new())),
            GpuBackend::Wgpu => Ok(Box::new(WgpuCompiler::new())),
            GpuBackend::Cpu => Err(KernelError::BackendNotSupported { backend }),
        }
    }

    /// Load built-in kernel templates
    fn load_builtin_templates() -> HashMap<String, KernelTemplate> {
        let mut templates = HashMap::new();

        // SGD kernel template
        templates.insert(
            "sgd_update".to_string(),
            KernelTemplate {
                name: "sgd_update".to_string(),
                source_template: sgd_kernel_template().to_string(),
                parameters: vec![
                    TemplateParameter {
                        name: "dtype".to_string(),
                        param_type: ParameterType::DataType,
                        default_value: Some("float".to_string()),
                        required: true,
                    },
                    TemplateParameter {
                        name: "learning_rate".to_string(),
                        param_type: ParameterType::Float,
                        default_value: Some("0.01".to_string()),
                        required: true,
                    },
                ],
                supported_backends: vec![GpuBackend::Cuda, GpuBackend::Rocm, GpuBackend::Metal],
                default_launch_config: LaunchConfig {
                    grid_size: (1, 1, 1),
                    block_size: (256, 1, 1),
                    shared_memory_size: 0,
                    stream: None,
                },
            },
        );

        // Adam kernel template
        templates.insert(
            "adam_update".to_string(),
            KernelTemplate {
                name: "adam_update".to_string(),
                source_template: adam_kernel_template().to_string(),
                parameters: vec![
                    TemplateParameter {
                        name: "dtype".to_string(),
                        param_type: ParameterType::DataType,
                        default_value: Some("float".to_string()),
                        required: true,
                    },
                    TemplateParameter {
                        name: "learning_rate".to_string(),
                        param_type: ParameterType::Float,
                        default_value: Some("0.001".to_string()),
                        required: true,
                    },
                    TemplateParameter {
                        name: "beta1".to_string(),
                        param_type: ParameterType::Float,
                        default_value: Some("0.9".to_string()),
                        required: true,
                    },
                    TemplateParameter {
                        name: "beta2".to_string(),
                        param_type: ParameterType::Float,
                        default_value: Some("0.999".to_string()),
                        required: true,
                    },
                ],
                supported_backends: vec![GpuBackend::Cuda, GpuBackend::Rocm, GpuBackend::Metal],
                default_launch_config: LaunchConfig {
                    grid_size: (1, 1, 1),
                    block_size: (256, 1, 1),
                    shared_memory_size: 0,
                    stream: None,
                },
            },
        );

        templates
    }

    /// Generate kernel from template
    pub fn generate_kernel_from_template(
        &self,
        template_name: &str,
        parameters: &HashMap<String, String>,
    ) -> Result<String, KernelError> {
        let template =
            self.templates
                .get(template_name)
                .ok_or_else(|| KernelError::KernelNotFound {
                    name: template_name.to_string(),
                })?;

        if !template.supported_backends.contains(&self.backend) {
            return Err(KernelError::BackendNotSupported {
                backend: self.backend,
            });
        }

        let mut source = template.source_template.clone();

        // Replace template parameters
        for param in &template.parameters {
            let value = if let Some(v) = parameters.get(&param.name) {
                v.clone()
            } else if let Some(default) = &param.default_value {
                default.clone()
            } else if param.required {
                return Err(KernelError::InvalidTemplateParameter {
                    param: param.name.clone(),
                });
            } else {
                continue;
            };

            let placeholder = format!("{{{{{}}}}}", param.name);
            source = source.replace(&placeholder, &value);
        }

        Ok(source)
    }

    /// Compile and cache a kernel
    pub fn compile_kernel(
        &mut self,
        name: String,
        source: String,
    ) -> Result<Arc<CompiledKernel>, KernelError> {
        // Check cache first
        if let Some(cached) = self.kernel_cache.get(&name) {
            return Ok(cached.clone());
        }

        // Compile kernel
        let compiled = self
            .compiler
            .compile_kernel(&source, &self.compilation_options)?;
        let kernel_arc = Arc::new(compiled);

        // Cache the compiled kernel
        self.kernel_cache.insert(name, kernel_arc.clone());

        Ok(kernel_arc)
    }

    /// Get cached kernel
    pub fn get_kernel(&self, name: &str) -> Option<Arc<CompiledKernel>> {
        self.kernel_cache.get(name).cloned()
    }

    /// Clear kernel cache
    pub fn clear_cache(&mut self) {
        self.kernel_cache.clear();
    }

    /// Get compiler capabilities
    pub fn get_capabilities(&self) -> CompilerCapabilities {
        self.compiler.get_capabilities()
    }

    /// Calculate optimal launch configuration
    pub fn calculate_launch_config(
        &self,
        kernel_type: KernelType,
        data_size: usize,
    ) -> LaunchConfig {
        let capabilities = self.get_capabilities();

        match kernel_type {
            KernelType::VectorOps | KernelType::SGDUpdate | KernelType::AdamUpdate => {
                let threads_per_block = 256.min(capabilities.max_threads_per_block);
                let blocks =
                    (data_size + threads_per_block as usize - 1) / threads_per_block as usize;

                LaunchConfig {
                    grid_size: (blocks as u32, 1, 1),
                    block_size: (threads_per_block, 1, 1),
                    shared_memory_size: 0,
                    stream: None,
                }
            }
            KernelType::MatrixOps => {
                let block_dim = 16; // 16x16 thread blocks for matrix operations
                let grid_dim =
                    ((data_size as f64).sqrt().ceil() as u32 + block_dim - 1) / block_dim;

                LaunchConfig {
                    grid_size: (grid_dim, grid_dim, 1),
                    block_size: (block_dim, block_dim, 1),
                    shared_memory_size: block_dim as usize * block_dim as usize * 4, // 4 bytes per float
                    stream: None,
                }
            }
            KernelType::Reduction => {
                let threads_per_block = 512.min(capabilities.max_threads_per_block);
                let blocks =
                    (data_size + threads_per_block as usize - 1) / threads_per_block as usize;

                LaunchConfig {
                    grid_size: (blocks as u32, 1, 1),
                    block_size: (threads_per_block, 1, 1),
                    shared_memory_size: threads_per_block as usize * 4, // Shared memory for reduction
                    stream: None,
                }
            }
            _ => {
                // Default configuration
                LaunchConfig {
                    grid_size: (1, 1, 1),
                    block_size: (256, 1, 1),
                    shared_memory_size: 0,
                    stream: None,
                }
            }
        }
    }
}

// Backend-specific compilers

/// CUDA kernel compiler
#[derive(Debug)]
pub struct CudaCompiler {
    nvcc_path: String,
}

impl CudaCompiler {
    pub fn new() -> Self {
        Self {
            nvcc_path: "nvcc".to_string(),
        }
    }
}

impl KernelCompiler for CudaCompiler {
    fn compile_kernel(
        &self,
        source: &str,
        options: &CompilationOptions,
    ) -> Result<CompiledKernel, KernelError> {
        // In a real implementation, this would call nvcc or use NVRTC
        Ok(CompiledKernel {
            name: "cuda_kernel".to_string(),
            backend: GpuBackend::Cuda,
            binary: source.as_bytes().to_vec(),
        })
    }

    fn backend_type(&self) -> GpuBackend {
        GpuBackend::Cuda
    }

    fn supports_feature(&self, feature: KernelFeature) -> bool {
        match feature {
            KernelFeature::HalfPrecision => true,
            KernelFeature::DoublePrecision => true,
            KernelFeature::TensorCores => true,
            KernelFeature::CooperativeGroups => true,
            KernelFeature::DynamicParallelism => true,
            KernelFeature::UnifiedMemory => true,
        }
    }

    fn get_capabilities(&self) -> CompilerCapabilities {
        CompilerCapabilities {
            max_threads_per_block: 1024,
            max_blocks_per_grid: (65535, 65535, 65535),
            max_shared_memory: 49152,
            supported_data_types: vec![
                "float".to_string(),
                "double".to_string(),
                "half".to_string(),
                "int".to_string(),
            ],
            compute_capability: Some((8, 6)),
        }
    }
}

/// ROCm kernel compiler
#[derive(Debug)]
pub struct RocmCompiler;

impl RocmCompiler {
    pub fn new() -> Self {
        Self
    }
}

impl KernelCompiler for RocmCompiler {
    fn compile_kernel(
        &self,
        source: &str,
        _options: &CompilationOptions,
    ) -> Result<CompiledKernel, KernelError> {
        Ok(CompiledKernel {
            name: "rocm_kernel".to_string(),
            backend: GpuBackend::Rocm,
            binary: source.as_bytes().to_vec(),
        })
    }

    fn backend_type(&self) -> GpuBackend {
        GpuBackend::Rocm
    }

    fn supports_feature(&self, feature: KernelFeature) -> bool {
        match feature {
            KernelFeature::HalfPrecision => true,
            KernelFeature::DoublePrecision => true,
            KernelFeature::TensorCores => false,
            KernelFeature::CooperativeGroups => true,
            KernelFeature::DynamicParallelism => false,
            KernelFeature::UnifiedMemory => false,
        }
    }

    fn get_capabilities(&self) -> CompilerCapabilities {
        CompilerCapabilities {
            max_threads_per_block: 1024,
            max_blocks_per_grid: (65535, 65535, 65535),
            max_shared_memory: 65536,
            supported_data_types: vec![
                "float".to_string(),
                "double".to_string(),
                "half".to_string(),
                "int".to_string(),
            ],
            compute_capability: None,
        }
    }
}

/// Metal kernel compiler
#[derive(Debug)]
pub struct MetalCompiler;

impl MetalCompiler {
    pub fn new() -> Self {
        Self
    }
}

impl KernelCompiler for MetalCompiler {
    fn compile_kernel(
        &self,
        source: &str,
        _options: &CompilationOptions,
    ) -> Result<CompiledKernel, KernelError> {
        Ok(CompiledKernel {
            name: "metal_kernel".to_string(),
            backend: GpuBackend::Metal,
            binary: source.as_bytes().to_vec(),
        })
    }

    fn backend_type(&self) -> GpuBackend {
        GpuBackend::Metal
    }

    fn supports_feature(&self, feature: KernelFeature) -> bool {
        match feature {
            KernelFeature::HalfPrecision => true,
            KernelFeature::DoublePrecision => false,
            KernelFeature::TensorCores => false,
            KernelFeature::CooperativeGroups => false,
            KernelFeature::DynamicParallelism => false,
            KernelFeature::UnifiedMemory => true,
        }
    }

    fn get_capabilities(&self) -> CompilerCapabilities {
        CompilerCapabilities {
            max_threads_per_block: 1024,
            max_blocks_per_grid: (65535, 65535, 65535),
            max_shared_memory: 32768,
            supported_data_types: vec!["float".to_string(), "half".to_string(), "int".to_string()],
            compute_capability: None,
        }
    }
}

/// WebGPU kernel compiler
#[derive(Debug)]
pub struct WgpuCompiler;

impl WgpuCompiler {
    pub fn new() -> Self {
        Self
    }
}

impl KernelCompiler for WgpuCompiler {
    fn compile_kernel(
        &self,
        source: &str,
        _options: &CompilationOptions,
    ) -> Result<CompiledKernel, KernelError> {
        Ok(CompiledKernel {
            name: "wgpu_kernel".to_string(),
            backend: GpuBackend::Wgpu,
            binary: source.as_bytes().to_vec(),
        })
    }

    fn backend_type(&self) -> GpuBackend {
        GpuBackend::Wgpu
    }

    fn supports_feature(&self, feature: KernelFeature) -> bool {
        match feature {
            KernelFeature::HalfPrecision => false,
            KernelFeature::DoublePrecision => false,
            KernelFeature::TensorCores => false,
            KernelFeature::CooperativeGroups => false,
            KernelFeature::DynamicParallelism => false,
            KernelFeature::UnifiedMemory => false,
        }
    }

    fn get_capabilities(&self) -> CompilerCapabilities {
        CompilerCapabilities {
            max_threads_per_block: 256,
            max_blocks_per_grid: (65535, 65535, 65535),
            max_shared_memory: 16384,
            supported_data_types: vec!["float".to_string(), "int".to_string()],
            compute_capability: None,
        }
    }
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            fast_math: true,
            debug_info: false,
            target_arch: "compute_70".to_string(),
            extra_flags: Vec::new(),
            max_registers: None,
            shared_memory_hint: None,
        }
    }
}

// Placeholder templates (in a real implementation, these would be separate files)
fn sgd_kernel_template() -> &'static str {
    r#"
__global__ void sgd_update_kernel(
    {{dtype}}* params,
    const {{dtype}}* gradients,
    const {{dtype}} learning_rate,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * gradients[idx];
    }
}
"#
}

fn adam_kernel_template() -> &'static str {
    r#"
__global__ void adam_update_kernel(
    {{dtype}}* params,
    {{dtype}}* m,
    {{dtype}}* v,
    const {{dtype}}* gradients,
    const {{dtype}} learning_rate,
    const {{dtype}} beta1,
    const {{dtype}} beta2,
    const {{dtype}} epsilon,
    const int step,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        {{dtype}} grad = gradients[idx];
        {{dtype}} m_val = beta1 * m[idx] + (1.0 - beta1) * grad;
        {{dtype}} v_val = beta2 * v[idx] + (1.0 - beta2) * grad * grad;

        {{dtype}} m_hat = m_val / (1.0 - pow(beta1, step));
        {{dtype}} v_hat = v_val / (1.0 - beta2, step));

        params[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);

        m[idx] = m_val;
        v[idx] = v_val;
    }
}
"#
}

// Template constants are provided by the functions above

// Hack to provide template content when files don't exist
fn include_str(path: &str) -> &str {
    match path {
        "templates/sgd_kernel.template" => sgd_kernel_template(),
        "templates/adam_kernel.template" => adam_kernel_template(),
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_manager_creation() {
        let options = CompilationOptions::default();
        let manager = KernelManager::new(GpuBackend::Cuda, options);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_template_generation() {
        let options = CompilationOptions::default();
        let manager = KernelManager::new(GpuBackend::Cuda, options).unwrap();

        let mut params = HashMap::new();
        params.insert("dtype".to_string(), "float".to_string());
        params.insert("learning_rate".to_string(), "0.01".to_string());

        let source = manager.generate_kernel_from_template("sgd_update", &params);
        assert!(source.is_ok());
    }

    #[test]
    fn test_launch_config_calculation() {
        let options = CompilationOptions::default();
        let manager = KernelManager::new(GpuBackend::Cuda, options).unwrap();

        let config = manager.calculate_launch_config(KernelType::SGDUpdate, 10000);
        assert!(config.grid_size.0 > 0);
        assert!(config.block_size.0 > 0);
    }

    #[test]
    fn test_compiler_capabilities() {
        let compiler = CudaCompiler::new();
        let capabilities = compiler.get_capabilities();

        assert!(capabilities.max_threads_per_block > 0);
        assert!(!capabilities.supported_data_types.is_empty());
    }
}
