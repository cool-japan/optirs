//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::TestFunction;
use scirs2_core::numeric::Float;

use super::types::{CrossFrameworkBenchmarkResult, CrossFrameworkConfig, PythonScriptTemplates};

/// Cross-framework benchmark suite
pub struct CrossFrameworkBenchmark<A: Float> {
    /// Configuration
    pub(super) config: CrossFrameworkConfig,
    /// Test functions
    pub(super) test_functions: Vec<TestFunction<A>>,
    /// Python script templates
    pub(super) python_scripts: PythonScriptTemplates,
    /// Results storage
    pub(super) results: Vec<CrossFrameworkBenchmarkResult<A>>,
}
