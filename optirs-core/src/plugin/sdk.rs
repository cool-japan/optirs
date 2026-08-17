// Plugin SDK utilities and helpers for optimizer development
//
// This module provides a comprehensive SDK for developing custom optimizer plugins,
// including base classes, utilities, testing frameworks, and development tools.

#[allow(dead_code)]
use super::core::*;
#[cfg(feature = "cross-platform-testing")]
use crate::benchmarking::cross_platform_tester::{PerformanceBaseline, PlatformTarget};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Base optimizer plugin implementation with common functionality
pub struct BaseOptimizerPlugin<A: Float + std::fmt::Debug> {
    /// Plugin information
    info: PluginInfo,
    /// Plugin capabilities
    capabilities: PluginCapabilities,
    /// Optimizer configuration
    config: OptimizerConfig,
    /// Internal state
    state: BaseOptimizerState<A>,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Memory usage tracking
    memory_usage: MemoryUsage,
    /// Event handlers
    event_handlers: Vec<Box<dyn PluginEventHandler>>,
}

impl<A: Float + std::fmt::Debug + Send + Sync> std::fmt::Debug for BaseOptimizerPlugin<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseOptimizerPlugin")
            .field("info", &self.info)
            .field("capabilities", &self.capabilities)
            .field("config", &self.config)
            .field("state", &self.state)
            .field("metrics", &self.metrics)
            .field("memory_usage", &self.memory_usage)
            .field(
                "event_handlers",
                &format!("{} handlers", self.event_handlers.len()),
            )
            .finish()
    }
}

/// Base optimizer state
#[derive(Debug, Clone)]
pub struct BaseOptimizerState<A: Float + std::fmt::Debug> {
    /// Step count
    pub step_count: usize,
    /// Parameter count
    pub param_count: usize,
    /// Learning rate history
    pub lr_history: Vec<A>,
    /// Gradient norms history
    pub grad_norm_history: Vec<A>,
    /// Parameter change norms history
    pub param_change_history: Vec<A>,
    /// Momentum buffer (SGD-with-momentum velocity), one entry per
    /// parameter, lazily sized to the incoming gradient on the first step.
    pub momentum_buffer: Vec<A>,
    /// Custom state data
    pub custom_state: HashMap<String, StateValue>,
}

/// Plugin development utilities
pub struct PluginSDK;

/// Plugin testing framework
pub struct PluginTester<A: Float> {
    /// Test configuration
    config: TestConfig,
    /// Test suite
    test_suite: TestSuite<A>,
    /// Benchmark suite
    benchmark_suite: BenchmarkSuite<A>,
    /// Validation framework
    validator: PluginValidator<A>,
}

/// Test configuration
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Number of test iterations
    pub iterations: usize,
    /// Tolerance for numerical tests
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Enable performance testing
    pub enable_performance_tests: bool,
    /// Enable memory testing
    pub enable_memory_tests: bool,
    /// Enable convergence testing
    pub enable_convergence_tests: bool,
}

/// Test suite for plugin validation
#[derive(Debug)]
pub struct TestSuite<A: Float> {
    /// Functionality tests
    pub functionality_tests: Vec<Box<dyn PluginTest<A>>>,
    /// Performance tests
    pub performance_tests: Vec<Box<dyn PerformanceTest<A>>>,
    /// Convergence tests
    pub convergence_tests: Vec<Box<dyn ConvergenceTest<A>>>,
    /// Memory tests
    pub memory_tests: Vec<Box<dyn MemoryTest<A>>>,
}

/// Individual plugin test trait
pub trait PluginTest<A: Float>: Debug {
    /// Run the test
    fn run_test(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult;

    /// Get test name
    fn name(&self) -> &str;

    /// Get test description
    fn description(&self) -> &str;
}

/// Performance test trait
pub trait PerformanceTest<A: Float>: Debug {
    /// Run performance test
    fn run_performance_test(&self, plugin: &mut dyn OptimizerPlugin<A>) -> PerformanceTestResult;

    /// Get test name
    fn name(&self) -> &str;

    /// Get performance baseline
    #[cfg(feature = "cross-platform-testing")]
    fn baseline(&self) -> PerformanceBaseline;
}

/// Convergence test trait
pub trait ConvergenceTest<A: Float>: Debug {
    /// Run convergence test
    fn run_convergence_test(&self, plugin: &mut dyn OptimizerPlugin<A>)
        -> ConvergenceTestResult<A>;

    /// Get test name
    fn name(&self) -> &str;

    /// Get convergence criteria
    fn convergence_criteria(&self) -> ConvergenceCriteria<A>;
}

/// Memory test trait
pub trait MemoryTest<A: Float>: Debug {
    /// Run memory test
    fn run_memory_test(&self, plugin: &mut dyn OptimizerPlugin<A>) -> MemoryTestResult;

    /// Get test name
    fn name(&self) -> &str;

    /// Get memory constraints
    fn memory_constraints(&self) -> MemoryConstraints;
}

/// Test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test passed
    pub passed: bool,
    /// Test message
    pub message: String,
    /// Execution time
    pub execution_time: std::time::Duration,
    /// Additional data
    pub data: HashMap<String, serde_json::Value>,
}

/// Performance test result
#[derive(Debug, Clone)]
pub struct PerformanceTestResult {
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Comparison with baseline
    pub baseline_comparison: BaselineComparison,
    /// Performance score (0.0 to 1.0)
    pub performance_score: f64,
}

/// Convergence test result
#[derive(Debug, Clone)]
pub struct ConvergenceTestResult<A: Float> {
    /// Converged successfully
    pub converged: bool,
    /// Number of iterations to convergence
    pub iterations_to_convergence: Option<usize>,
    /// Final objective value
    pub final_objective: A,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Convergence metrics
    pub metrics: ConvergenceMetrics,
}

/// Memory test result
#[derive(Debug, Clone)]
pub struct MemoryTestResult {
    /// Memory usage metrics
    pub memory_metrics: MemoryUsage,
    /// Memory leak detected
    pub memory_leak_detected: bool,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Baseline comparison
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Relative performance (baseline = 1.0)
    pub relative_performance: f64,
    /// Performance difference (absolute)
    pub absolute_difference: f64,
    /// Performance improvement (percentage)
    pub improvement_percent: f64,
}

/// Convergence criteria
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria<A: Float> {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Gradient norm tolerance
    pub gradient_tolerance: A,
    /// Function value tolerance
    pub function_tolerance: A,
    /// Parameter change tolerance
    pub parameter_tolerance: A,
}

/// Memory constraints for testing
#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    /// Maximum memory usage (bytes)
    pub max_memory_usage: usize,
    /// Maximum allocation count
    pub max_allocations: usize,
    /// Memory leak tolerance (bytes)
    pub leak_tolerance: usize,
}

/// Plugin validator for comprehensive validation
#[derive(Debug)]
pub struct PluginValidator<A: Float> {
    /// Validation rules
    rules: Vec<Box<dyn ValidationRule<A>>>,
    /// Compatibility checker
    compatibility_checker: CompatibilityChecker,
}

/// Validation rule trait
pub trait ValidationRule<A: Float>: Debug {
    /// Validate plugin
    fn validate(&self, plugin: &dyn OptimizerPlugin<A>) -> ValidationResult;

    /// Get rule name
    fn name(&self) -> &str;

    /// Get rule severity
    fn severity(&self) -> ValidationSeverity;
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation passed
    pub passed: bool,
    /// Validation message
    pub message: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

/// Validation severity levels
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Compatibility checker
#[derive(Debug)]
pub struct CompatibilityChecker {
    /// Target platforms
    target_platforms: Vec<String>,
    /// Rust version requirements
    rust_versions: Vec<String>,
    /// Dependency compatibility
    dependency_compatibility: HashMap<String, String>,
}

/// Benchmark suite for performance evaluation
#[derive(Debug)]
pub struct BenchmarkSuite<A: Float> {
    /// Standard benchmarks
    standard_benchmarks: Vec<Box<dyn Benchmark<A>>>,
    /// Custom benchmarks
    custom_benchmarks: Vec<Box<dyn Benchmark<A>>>,
    /// Benchmark configuration
    config: BenchmarkConfig,
}

/// Benchmark trait
pub trait Benchmark<A: Float>: Debug {
    /// Run benchmark
    fn run_benchmark(&self, plugin: &mut dyn OptimizerPlugin<A>) -> BenchmarkResult<A>;

    /// Get benchmark name
    fn name(&self) -> &str;

    /// Get benchmark description
    fn description(&self) -> &str;

    /// Get benchmark category
    fn category(&self) -> BenchmarkCategory;
}

/// Benchmark categories
#[derive(Debug, Clone)]
pub enum BenchmarkCategory {
    /// Speed benchmarks
    Speed,
    /// Memory benchmarks
    Memory,
    /// Accuracy benchmarks
    Accuracy,
    /// Scalability benchmarks
    Scalability,
    /// Robustness benchmarks
    Robustness,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult<A: Float> {
    /// Benchmark name
    pub name: String,
    /// Score, normalized into `[0, 1]` against the benchmark's
    /// `expected_baseline` (higher is better)
    pub score: f64,
    /// Metrics
    pub metrics: HashMap<String, f64>,
    /// Execution time
    pub execution_time: std::time::Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// Additional data
    pub data: HashMap<String, A>,
    /// Whether this benchmark actually measured the plugin (e.g. `false`
    /// when `initialize()` failed before any step could run). Unverified
    /// results are excluded from the overall score rather than counted.
    pub verified: bool,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of benchmark runs
    pub runs: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Problem sizes to test
    pub problem_sizes: Vec<usize>,
    /// Random seeds
    pub random_seeds: Vec<u64>,
}

/// Plugin development helper macros and utilities
impl PluginSDK {
    /// Create a plugin template with common functionality
    pub fn create_plugin_template(name: &str) -> PluginTemplate {
        PluginTemplate::new(name)
    }

    /// Validate plugin configuration schema
    pub fn validate_config_schema(schema: &ConfigSchema) -> Result<()> {
        for (field_name, field_schema) in &schema.fields {
            if field_name.is_empty() {
                return Err(OptimError::InvalidConfig(
                    "Field name cannot be empty".to_string(),
                ));
            }

            if field_schema.description.is_empty() {
                return Err(OptimError::InvalidConfig(format!(
                    "Field '{}' must have a description",
                    field_name
                )));
            }
        }
        Ok(())
    }

    /// Generate plugin manifest template
    pub fn generate_plugin_manifest(info: &PluginInfo) -> String {
        format!(
            r#"[plugin]
name = "{}"
version = "{}"
description = "{}"
author = "{}"
license = "{}"
entry_point = "plugin_main"

[build]
rust_version = "1.70.0"
target = "*"
profile = "release"

[runtime]
min_rust_version = "1.70.0"
"#,
            info.name, info.version, info.description, info.author, info.license
        )
    }

    /// Create default test configuration
    pub fn default_test_config() -> TestConfig {
        TestConfig {
            iterations: 100,
            tolerance: 1e-6,
            random_seed: 42,
            enable_performance_tests: true,
            enable_memory_tests: true,
            enable_convergence_tests: true,
        }
    }

    /// Create performance baseline from existing optimizer
    #[cfg(feature = "cross-platform-testing")]
    pub fn create_performance_baseline<A>(
        optimizer: &mut dyn OptimizerPlugin<A>,
        test_data: &[(Array1<A>, Array1<A>)],
    ) -> PerformanceBaseline
    where
        A: Float + Debug + Send + Sync + 'static,
    {
        let start_time = std::time::Instant::now();
        let mut total_memory = 0;

        for (params, gradients) in test_data {
            let _result = optimizer.step(params, gradients);
            total_memory += optimizer.memory_usage().current_usage;
        }

        let execution_time = start_time.elapsed();
        let _avg_memory = total_memory / test_data.len();

        PerformanceBaseline {
            target: PlatformTarget::CPU,
            throughput_ops_per_sec: test_data.len() as f64 / execution_time.as_secs_f64(),
            latency_ms: execution_time.as_secs_f64() * 1000.0 / test_data.len() as f64,
            memory_usage_mb: total_memory as f64 / (1024.0 * 1024.0),
            energy_consumption_joules: None,
            accuracy_metrics: HashMap::new(),
        }
    }
}

/// Plugin template for rapid development
#[derive(Debug)]
pub struct PluginTemplate {
    /// Template name
    name: String,
    /// Template structure
    structure: TemplateStructure,
}

/// Template structure definition
#[derive(Debug)]
pub struct TemplateStructure {
    /// Source files
    pub source_files: Vec<TemplateFile>,
    /// Configuration files
    pub config_files: Vec<TemplateFile>,
    /// Test files
    pub test_files: Vec<TemplateFile>,
    /// Documentation files
    pub doc_files: Vec<TemplateFile>,
}

/// Template file
#[derive(Debug)]
pub struct TemplateFile {
    /// File path
    pub path: String,
    /// File content
    pub content: String,
    /// File type
    pub file_type: TemplateFileType,
}

/// Template file types
#[derive(Debug)]
pub enum TemplateFileType {
    /// Rust source file
    RustSource,
    /// TOML configuration
    TomlConfig,
    /// Markdown documentation
    Markdown,
    /// Test file
    Test,
}

impl PluginTemplate {
    /// Create a new plugin template
    pub fn new(name: &str) -> Self {
        let structure = Self::create_default_structure(name);
        Self {
            name: name.to_string(),
            structure,
        }
    }

    /// Generate template files to directory
    pub fn generate_to_directory(&self, outputdir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(outputdir)?;

        for file in &self.structure.source_files {
            let file_path = outputdir.join(&file.path);
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file_path, &file.content)?;
        }

        for file in &self.structure.config_files {
            let file_path = outputdir.join(&file.path);
            std::fs::write(&file_path, &file.content)?;
        }

        for file in &self.structure.test_files {
            let file_path = outputdir.join(&file.path);
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file_path, &file.content)?;
        }

        Ok(())
    }

    fn create_default_structure(name: &str) -> TemplateStructure {
        let lib_rs_content = format!(
            r#"//! {} optimizer plugin
//
// This is an auto-generated plugin template.

use optirs_core::plugin::*;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

#[derive(Debug)]
pub struct {}Optimizer<A: Float> {{
    learning_rate: A,
    // Add your optimizer state here
}}

impl<A: Float + Send + Sync> {}Optimizer<A> {{
    pub fn new(_learningrate: A) -> Self {{
        Self {{
            learning_rate,
        }}
    }}
}}

impl<A: Float + std::fmt::Debug + Send + Sync + 'static> OptimizerPlugin<A> for {}Optimizer<A> {{
    fn step(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>> {{
        // Implement your optimization step here
        Ok(params - &(gradients * self.learning_rate))
    }}
    
    fn name(&self) -> &str {{
        "{}"
    }}
    
    fn version(&self) -> &str {{
        "0.1.0"
    }}
    
    fn plugin_info(&self) -> PluginInfo {{
        create_plugin_info("{}", "0.1.0", "Plugin Developer")
    }}
    
    fn capabilities(&self) -> PluginCapabilities {{
        create_basic_capabilities()
    }}
    
    fn initialize(&mut self, paramshape: &[usize]) -> Result<()> {{
        Ok(())
    }}
    
    fn reset(&mut self) -> Result<()> {{
        Ok(())
    }}
    
    fn get_config(&self) -> OptimizerConfig {{
        OptimizerConfig::default()
    }}
    
    fn set_config(&mut self, config: OptimizerConfig) -> Result<()> {{
        Ok(())
    }}
    
    fn get_state(&self) -> Result<OptimizerState> {{
        Ok(OptimizerState::default())
    }}
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {{
        Ok(())
    }}
    
    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<A>> {{
        Box::new(Self::new(self.learning_rate))
    }}
}}

// Plugin factory implementation
#[derive(Debug)]
pub struct {}Factory;

impl<A: Float + std::fmt::Debug + Send + Sync + 'static> OptimizerPluginFactory<A> for {}Factory {{
    fn create_optimizer(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<A>>> {{
        let learning_rate = A::from(config.learning_rate).expect("unwrap failed");
        Ok(Box::new({}Optimizer::new(learning_rate)))
    }}
    
    fn factory_info(&self) -> PluginInfo {{
        create_plugin_info("{}", "0.1.0", "Plugin Developer")
    }}
    
    fn validate_config(&self, config: &OptimizerConfig) -> Result<()> {{
        if config.learning_rate <= 0.0 {{
            return Err(OptimError::InvalidConfig(
                "Learning rate must be positive".to_string(),
            ));
        }}
        Ok(())
    }}
    
    fn default_config(&self) -> OptimizerConfig {{
        OptimizerConfig {{
            learning_rate: 0.001,
            ..Default::default()
        }}
    }}
    
    fn config_schema(&self) -> ConfigSchema {{
        let mut schema = ConfigSchema {{
            fields: std::collections::HashMap::new(),
            required_fields: vec!["learning_rate".to_string()],
            version: "1.0".to_string(),
        }};
        
        schema.fields.insert(
            "learning_rate".to_string(),
            FieldSchema {{
                field_type: FieldType::Float {{ min: Some(0.0), max: None }},
                description: "Learning rate for optimization".to_string(),
                default_value: Some(ConfigValue::Float(0.001)),
                constraints: vec![ValidationConstraint::Positive],
                required: true,
            }},
        );
        
        schema
    }}
}}
"#,
            name, name, name, name, name, name, name, name, name, name
        );

        let plugin_toml_content = format!(
            r#"[plugin]
name = "{}"
version = "0.1.0"
description = "Custom optimizer plugin"
author = "Plugin Developer"
license = "MIT"
entry_point = "plugin_main"

[build]
rust_version = "1.70.0"
target = "*"
profile = "release"

[runtime]
min_rust_version = "1.70.0"
"#,
            name
        );

        let test_content = format!(
            r#"//! Tests for {} optimizer plugin

use super::*;
use scirs2_core::ndarray::Array1;

#[test]
#[allow(dead_code)]
fn test_{}_basic_functionality() {{
    let mut optimizer = {}Optimizer::new(0.01);
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    
    let result = optimizer.step(&params, &gradients).expect("unwrap failed");
    
    // Verify the result
    assert!((result[0] - 0.999).abs() < 1e-6);
    assert!((result[1] - 1.998).abs() < 1e-6);
    assert!((result[2] - 2.997).abs() < 1e-6);
}}

#[test]
#[allow(dead_code)]
fn test_{}_convergence() {{
    let mut optimizer = {}Optimizer::new(0.1);
    let mut params = Array1::from_vec(vec![1.0, 1.0]);
    
    // Optimize towards zero
    for _ in 0..100 {{
        let gradients = &params * 2.0; // Gradient of x^2
        params = optimizer.step(&params, &gradients).expect("unwrap failed");
    }}
    
    // Should converge close to zero
    assert!(params.iter().all(|&x| x.abs() < 0.1));
}}
"#,
            name,
            name.to_lowercase(),
            name,
            name.to_lowercase(),
            name
        );

        TemplateStructure {
            source_files: vec![TemplateFile {
                path: "src/lib.rs".to_string(),
                content: lib_rs_content,
                file_type: TemplateFileType::RustSource,
            }],
            config_files: vec![TemplateFile {
                path: "plugin.toml".to_string(),
                content: plugin_toml_content,
                file_type: TemplateFileType::TomlConfig,
            }],
            test_files: vec![TemplateFile {
                path: "tests/integration_tests.rs".to_string(),
                content: test_content,
                file_type: TemplateFileType::Test,
            }],
            doc_files: vec![],
        }
    }
}

// Implementation for base optimizer plugin

impl<A: Float + Debug + Send + Sync + 'static> BaseOptimizerPlugin<A> {
    /// Create a new base optimizer plugin
    pub fn new(info: PluginInfo, capabilities: PluginCapabilities) -> Self {
        Self {
            info,
            capabilities,
            config: OptimizerConfig::default(),
            state: BaseOptimizerState::new(),
            metrics: PerformanceMetrics::default(),
            memory_usage: MemoryUsage::default(),
            event_handlers: Vec::new(),
        }
    }

    /// Add event handler
    pub fn add_event_handler(&mut self, handler: Box<dyn PluginEventHandler>) {
        self.event_handlers.push(handler);
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self, steptime: std::time::Duration) {
        self.metrics.total_steps += 1;
        self.metrics.avg_step_time = (self.metrics.avg_step_time
            * (self.metrics.total_steps - 1) as f64
            + steptime.as_secs_f64())
            / self.metrics.total_steps as f64;
        self.metrics.throughput = 1.0 / self.metrics.avg_step_time;
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync> BaseOptimizerState<A> {
    fn new() -> Self {
        Self {
            step_count: 0,
            param_count: 0,
            lr_history: Vec::new(),
            grad_norm_history: Vec::new(),
            param_change_history: Vec::new(),
            momentum_buffer: Vec::new(),
            custom_state: HashMap::new(),
        }
    }
}

impl<A: Float + Debug + Send + Sync + 'static> OptimizerPlugin<A> for BaseOptimizerPlugin<A> {
    /// SGD with momentum, weight decay, and L2 gradient-norm clipping,
    /// driven entirely by `self.config` (`OptimizerConfig`). This is the
    /// SDK's documented base class for plugin authors: it must itself be a
    /// working, registrable optimizer -- not merely a field-storage struct
    /// -- since `BaseOptimizerPlugin` is re-exported at the crate root and
    /// used directly wherever a caller needs a plain baseline.
    fn step(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>> {
        if params.len() != gradients.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "parameter vector has {} elements but gradient vector has {}",
                params.len(),
                gradients.len()
            )));
        }
        if params.is_empty() {
            return Err(OptimError::InvalidConfig(
                "cannot step an optimizer over zero parameters".to_string(),
            ));
        }

        let lr = A::from(self.config.learning_rate).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "learning rate {} could not be represented in the parameter type",
                self.config.learning_rate
            ))
        })?;
        let weight_decay = A::from(self.config.weight_decay).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "weight decay {} could not be represented in the parameter type",
                self.config.weight_decay
            ))
        })?;
        let momentum = A::from(self.config.momentum).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "momentum {} could not be represented in the parameter type",
                self.config.momentum
            ))
        })?;

        // Decoupled weight decay (params contribute to the effective
        // gradient before clipping/momentum, matching standard SGD-with-
        // weight-decay semantics).
        let mut effective_grad: Vec<A> = gradients
            .iter()
            .zip(params.iter())
            .map(|(&g, &p)| g + weight_decay * p)
            .collect();

        // Optional global-norm gradient clipping.
        let grad_norm = effective_grad
            .iter()
            .fold(A::zero(), |acc, &g| acc + g * g)
            .sqrt();
        if let Some(clip) = self.config.gradient_clip {
            let clip_a = A::from(clip).ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "gradient clip {clip} could not be represented in the parameter type"
                ))
            })?;
            if clip_a > A::zero() && grad_norm > clip_a {
                let scale = clip_a / grad_norm;
                for g in &mut effective_grad {
                    *g = *g * scale;
                }
            }
        }

        // Lazily (re)size the momentum buffer to match the incoming
        // parameter count -- `initialize()` may have been called with a
        // different `paramshape`, or never called at all.
        if self.state.momentum_buffer.len() != params.len() {
            self.state.momentum_buffer = vec![A::zero(); params.len()];
        }

        let mut new_params = Array1::<A>::zeros(params.len());
        let mut change_sq = A::zero();
        for i in 0..params.len() {
            let velocity = momentum * self.state.momentum_buffer[i] + effective_grad[i];
            self.state.momentum_buffer[i] = velocity;
            let delta = lr * velocity;
            let updated = params[i] - delta;
            new_params[i] = updated;
            change_sq = change_sq + delta * delta;
        }

        self.state.step_count += 1;
        self.state.param_count = params.len();
        self.state.lr_history.push(lr);
        self.state.grad_norm_history.push(grad_norm);
        self.state.param_change_history.push(change_sq.sqrt());

        for handler in &mut self.event_handlers {
            let params_f64 = Array1::from_iter(params.iter().map(|p| p.to_f64().unwrap_or(0.0)));
            let grad_f64 = Array1::from_iter(gradients.iter().map(|g| g.to_f64().unwrap_or(0.0)));
            handler.on_step(self.state.step_count, &params_f64, &grad_f64);
        }

        Ok(new_params)
    }

    fn name(&self) -> &str {
        &self.info.name
    }

    fn version(&self) -> &str {
        &self.info.version
    }

    fn plugin_info(&self) -> PluginInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> PluginCapabilities {
        self.capabilities.clone()
    }

    fn initialize(&mut self, paramshape: &[usize]) -> Result<()> {
        if paramshape.is_empty() {
            return Err(OptimError::InvalidConfig(
                "cannot initialize with zero parameter groups".to_string(),
            ));
        }
        let total: usize = paramshape.iter().product();
        self.state = BaseOptimizerState::new();
        self.state.param_count = total;
        self.state.momentum_buffer = vec![A::zero(); total];
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        let param_count = self.state.param_count;
        self.state = BaseOptimizerState::new();
        self.state.param_count = param_count;
        self.state.momentum_buffer = vec![A::zero(); param_count];
        Ok(())
    }

    fn get_config(&self) -> OptimizerConfig {
        self.config.clone()
    }

    fn set_config(&mut self, config: OptimizerConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    fn get_state(&self) -> Result<OptimizerState> {
        let mut state_vectors = HashMap::new();
        state_vectors.insert(
            "momentum_buffer".to_string(),
            self.state
                .momentum_buffer
                .iter()
                .map(|v| v.to_f64().unwrap_or(0.0))
                .collect(),
        );
        Ok(OptimizerState {
            state_vectors,
            step_count: self.state.step_count,
            custom_state: self.state.custom_state.clone(),
        })
    }

    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.state.step_count = state.step_count;
        self.state.custom_state = state.custom_state;
        if let Some(buf) = state.state_vectors.get("momentum_buffer") {
            self.state.momentum_buffer = buf
                .iter()
                .map(|&v| A::from(v).unwrap_or_else(A::zero))
                .collect();
            self.state.param_count = self.state.momentum_buffer.len();
        }
        Ok(())
    }

    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<A>> {
        Box::new(BaseOptimizerPlugin {
            info: self.info.clone(),
            capabilities: self.capabilities.clone(),
            config: self.config.clone(),
            state: self.state.clone(),
            metrics: self.metrics.clone(),
            memory_usage: self.memory_usage.clone(),
            // Event handlers are per-instance observers, not cloneable
            // state; a clone starts with none attached, matching the
            // semantics of every other plugin's `clone_plugin`.
            event_handlers: Vec::new(),
        })
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_usage.clone()
    }

    fn performance_metrics(&self) -> PerformanceMetrics {
        self.metrics.clone()
    }
}

// Default implementations

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            tolerance: 1e-6,
            random_seed: 42,
            enable_performance_tests: true,
            enable_memory_tests: true,
            enable_convergence_tests: true,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            runs: 10,
            warmup_iterations: 5,
            problem_sizes: vec![10, 100, 1000],
            random_seeds: vec![42, 123, 456],
        }
    }
}

/// Macro for creating a simple optimizer plugin
#[macro_export]
macro_rules! create_optimizer_plugin {
    ($name:ident, $step_fn:expr) => {
        #[derive(Debug)]
        pub struct $name<A: Float> {
            config: OptimizerConfig,
            state: OptimizerState,
            _phantom: std::marker::PhantomData<A>,
        }

        impl<A: Float + Send + Sync> $name<A> {
            pub fn new() -> Self {
                Self {
                    config: OptimizerConfig::default(),
                    state: OptimizerState::default(),
                    _phantom: std::marker::PhantomData,
                }
            }
        }

        impl<A: Float + Send + Sync> Default for $name<A> {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<A: Float + std::fmt::Debug + Send + Sync + 'static> OptimizerPlugin<A> for $name<A> {
            fn step(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>> {
                $step_fn(self, params, gradients)
            }

            fn name(&self) -> &str {
                stringify!($name)
            }

            fn version(&self) -> &str {
                "0.1.0"
            }

            fn plugin_info(&self) -> PluginInfo {
                create_plugin_info(stringify!($name), "0.1.0", "Auto-generated")
            }

            fn capabilities(&self) -> PluginCapabilities {
                create_basic_capabilities()
            }

            fn initialize(&mut self, paramshape: &[usize]) -> Result<()> {
                if paramshape.is_empty() {
                    return Err($crate::error::OptimError::InvalidConfig(
                        "cannot initialize a plugin optimizer with zero parameter groups"
                            .to_string(),
                    ));
                }
                // A fresh initialization starts a fresh optimization run:
                // reset the step counter so re-initializing an existing
                // instance for a new run does not inherit a stale count.
                self.state.step_count = 0;
                Ok(())
            }

            fn reset(&mut self) -> Result<()> {
                self.state = OptimizerState::default();
                Ok(())
            }

            fn get_config(&self) -> OptimizerConfig {
                self.config.clone()
            }

            fn set_config(&mut self, config: OptimizerConfig) -> Result<()> {
                self.config = config;
                Ok(())
            }

            fn get_state(&self) -> Result<OptimizerState> {
                Ok(self.state.clone())
            }

            fn set_state(&mut self, state: OptimizerState) -> Result<()> {
                self.state = state;
                Ok(())
            }

            fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<A>> {
                Box::new(Self {
                    config: self.config.clone(),
                    state: self.state.clone(),
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time exercise of `create_optimizer_plugin!`, kept inside the
    // test module (not at module scope) so the generated `pub struct` never
    // becomes part of the shipped public API: the macro was invoked nowhere
    // in the workspace, so a field-name typo (`phantom` in the struct vs
    // `_phantom` in the constructor) went uncaught until any real user
    // tried to expand it. `use super::*` above still brings in the parent
    // module's `use super::core::*`, so the expansion resolves normally.
    //
    // The closure intentionally avoids naming the impl's generic parameter
    // `A` (macro hygiene keeps a bare `A` written at this call site in a
    // distinct context from the `A` the macro binds in `impl<A: ...>`); its
    // parameter and return types are inferred entirely from the immediate
    // call `$step_fn(self, params, gradients)` inside a `-> Result<Array1<A>>`
    // body.
    create_optimizer_plugin!(MacroGeneratedSgd, |_this, params, gradients| {
        Ok(params - gradients)
    });

    #[test]
    fn test_macro_generated_plugin_compiles_and_steps() {
        // Exercises the `create_optimizer_plugin!` expansion end-to-end:
        // construction, `initialize`, `step`, and `clone_plugin` all need
        // the struct's field names (`_phantom`, not `phantom`) to agree
        // between the struct definition and the constructor for this to
        // compile at all.
        let mut plugin: MacroGeneratedSgd<f64> = MacroGeneratedSgd::new();
        plugin.initialize(&[3]).expect("initialize should succeed");

        let params = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let grads = Array1::from(vec![0.1_f64, 0.2, 0.3]);
        let updated = plugin.step(&params, &grads).expect("step should succeed");
        assert_eq!(updated.len(), 3);

        // clone_plugin must preserve config/state, not silently reset to a
        // fresh instance.
        plugin
            .set_config(OptimizerConfig {
                learning_rate: 0.5,
                ..OptimizerConfig::default()
            })
            .expect("set_config should succeed");
        let cloned = plugin.clone_plugin();
        assert_eq!(cloned.get_config().learning_rate, 0.5);
    }

    #[test]
    fn test_macro_generated_plugin_rejects_empty_paramshape() {
        let mut plugin: MacroGeneratedSgd<f64> = MacroGeneratedSgd::new();
        assert!(plugin.initialize(&[]).is_err());
    }

    // F73 regression: `BaseOptimizerPlugin` previously implemented only
    // `Debug` -- there was no `impl OptimizerPlugin<A> for
    // BaseOptimizerPlugin<A>` at all, so the SDK's documented base class
    // (re-exported at the crate root) could never be registered or used
    // anywhere an `OptimizerPlugin` was expected. These exercise the trait
    // impl through the trait itself (`&mut dyn OptimizerPlugin<f64>`), not
    // just inherent methods, so a future regression back to "no impl" is a
    // compile error here, not a silently-vanished capability.
    fn base_plugin() -> BaseOptimizerPlugin<f64> {
        BaseOptimizerPlugin::new(
            create_plugin_info("TestBase", "0.1.0", "test"),
            create_basic_capabilities(),
        )
    }

    #[test]
    fn test_base_optimizer_plugin_implements_the_trait() {
        let mut owned = base_plugin();
        let plugin: &mut dyn OptimizerPlugin<f64> = &mut owned;
        plugin.initialize(&[3]).expect("initialize should succeed");

        let params = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let grads = Array1::from(vec![0.1_f64, 0.2, 0.3]);
        let updated = plugin.step(&params, &grads).expect("step should succeed");
        assert_eq!(updated.len(), 3);
        // Plain gradient descent moves parameters opposite the gradient.
        for (u, p) in updated.iter().zip(params.iter()) {
            assert!(u < p, "expected descent step to reduce each parameter");
        }
    }

    #[test]
    fn test_base_optimizer_plugin_rejects_dimension_mismatch() {
        let mut plugin = base_plugin();
        plugin.initialize(&[3]).expect("initialize should succeed");
        let params = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let grads = Array1::from(vec![0.1_f64, 0.2]);
        assert!(plugin.step(&params, &grads).is_err());
    }

    #[test]
    fn test_base_optimizer_plugin_gradient_clipping_bounds_update_norm() {
        let mut plugin = base_plugin();
        plugin
            .set_config(OptimizerConfig {
                learning_rate: 1.0,
                weight_decay: 0.0,
                momentum: 0.0,
                gradient_clip: Some(1.0),
                ..OptimizerConfig::default()
            })
            .expect("set_config should succeed");
        plugin.initialize(&[2]).expect("initialize should succeed");

        let params = Array1::from(vec![0.0_f64, 0.0]);
        // Unclipped gradient norm is 100 * sqrt(2) >> clip of 1.0.
        let grads = Array1::from(vec![100.0_f64, 100.0]);
        let updated = plugin.step(&params, &grads).expect("step should succeed");
        let step_norm =
            ((updated[0] - params[0]).powi(2) + (updated[1] - params[1]).powi(2)).sqrt();
        assert!(
            step_norm <= 1.0 + 1e-9,
            "clipped step norm should not exceed the configured clip value, got {step_norm}"
        );
    }

    #[test]
    fn test_base_optimizer_plugin_state_round_trips() {
        let mut plugin = base_plugin();
        plugin.initialize(&[2]).expect("initialize should succeed");
        let params = Array1::from(vec![1.0_f64, 1.0]);
        let grads = Array1::from(vec![0.5_f64, 0.5]);
        plugin.step(&params, &grads).expect("step should succeed");

        let saved = plugin.get_state().expect("get_state should succeed");
        assert_eq!(saved.step_count, 1);

        let mut restored = base_plugin();
        restored.set_state(saved).expect("set_state should succeed");
        assert_eq!(restored.get_state().unwrap().step_count, 1);
    }

    #[test]
    fn test_base_optimizer_plugin_clone_preserves_config_and_state() {
        let mut plugin = base_plugin();
        plugin
            .set_config(OptimizerConfig {
                learning_rate: 0.25,
                ..OptimizerConfig::default()
            })
            .expect("set_config should succeed");
        plugin.initialize(&[2]).expect("initialize should succeed");
        let params = Array1::from(vec![1.0_f64, 1.0]);
        let grads = Array1::from(vec![0.5_f64, 0.5]);
        plugin.step(&params, &grads).expect("step should succeed");

        let cloned = plugin.clone_plugin();
        assert_eq!(cloned.get_config().learning_rate, 0.25);
        assert_eq!(cloned.get_state().unwrap().step_count, 1);
    }

    #[test]
    fn test_plugin_template_creation() {
        let template = PluginTemplate::new("TestOptimizer");
        assert_eq!(template.name, "TestOptimizer");
        assert!(!template.structure.source_files.is_empty());
    }

    #[test]
    fn test_test_config_default() {
        let config = TestConfig::default();
        assert_eq!(config.iterations, 100);
        assert!(config.enable_performance_tests);
    }

    #[test]
    fn test_sdk_config_validation() {
        let mut schema = ConfigSchema {
            fields: HashMap::new(),
            required_fields: vec!["test_field".to_string()],
            version: "1.0".to_string(),
        };

        schema.fields.insert(
            "test_field".to_string(),
            FieldSchema {
                field_type: FieldType::Float {
                    min: None,
                    max: None,
                },
                description: "Test field".to_string(),
                default_value: None,
                constraints: Vec::new(),
                required: true,
            },
        );

        assert!(PluginSDK::validate_config_schema(&schema).is_ok());

        // Test with empty field name
        schema.fields.insert(
            "".to_string(),
            FieldSchema {
                field_type: FieldType::Float {
                    min: None,
                    max: None,
                },
                description: "Test".to_string(),
                default_value: None,
                constraints: Vec::new(),
                required: false,
            },
        );

        assert!(PluginSDK::validate_config_schema(&schema).is_err());
    }
}
