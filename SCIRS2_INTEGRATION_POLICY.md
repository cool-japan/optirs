# SciRS2 Integration Policy for OptiRS

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**OptiRS MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, minimal, and effective integration of SciRS2 crates into OptiRS, following the [SciRS2 Ecosystem Policy](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md).

## Core Integration Principles

### 1. **Strict Dependency Abstraction (MANDATORY)**
- **OptiRS is a non-core SciRS2 ecosystem crate** - it extends but does not replace SciRS2
- **NO DIRECT EXTERNAL DEPENDENCIES** allowed in OptiRS (only scirs2-* crates)
- **ALL scientific computing MUST go through scirs2-core abstractions**
- This is not optional - it is the foundational architecture of the SciRS2 ecosystem

### 2. **Prohibited Direct Dependencies**
```toml
# ❌ FORBIDDEN in all OptiRS crates (workspace and individual Cargo.toml)
[dependencies]
rand = "*"              # ❌ Use scirs2-core::random instead
rand_distr = "*"        # ❌ Use scirs2-core::random instead
rand_core = "*"         # ❌ Use scirs2-core::random instead
ndarray = "*"           # ❌ Use scirs2-core::ndarray instead
ndarray-rand = "*"      # ❌ Use scirs2-core::ndarray instead
ndarray-stats = "*"     # ❌ Use scirs2-core::ndarray instead
num-traits = "*"        # ❌ Use scirs2-core::numeric instead
num-complex = "*"       # ❌ Use scirs2-core::numeric instead
nalgebra = "*"          # ❌ Use scirs2-core::linalg instead
```

### 3. **Required Import Patterns**
```rust
// ❌ FORBIDDEN - Direct external imports
use rand::*;
use rand::Rng;
use rand_distr::{Beta, Normal};
use ndarray::{Array, Array1, Array2, array, s};
use num_complex::Complex;
use num_traits::Float;

// ✅ REQUIRED - SciRS2-Core abstractions
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
use scirs2_core::ndarray::*;          // Complete ndarray ecosystem with macros
use scirs2_core::numeric::*;          // num-traits, num-complex, num-integer
```

### 4. **Architectural Hierarchy**
```
OptiRS (ML Optimization Specialization)
    ↓ uses abstractions from
SciRS2-Core (Unified Scientific Computing Layer)
    ↓ manages and abstracts
External Libraries (rand, ndarray, num-traits, etc.)
```

**Key Point**: OptiRS does NOT directly depend on external libraries. Only scirs2-core can use them.

## Complete Dependency Mapping (from SciRS2 Policy)

**ALL external dependencies MUST go through scirs2-core abstractions:**

| External Crate | SciRS2-Core Module | OptiRS Usage |
|----------------|-------------------|--------------|
| `rand` | `scirs2_core::random` | Random initialization, stochastic optimization |
| `rand_distr` | `scirs2_core::random` | Probability distributions for sampling |
| `ndarray` | `scirs2_core::ndarray` | Array operations, gradients, parameters |
| `ndarray-rand` | `scirs2_core::ndarray` | Array initialization |
| `ndarray-stats` | `scirs2_core::ndarray` | Statistical operations on arrays |
| `num-traits` | `scirs2_core::numeric` | Numerical traits (Float, Zero, One) |
| `num-complex` | `scirs2_core::numeric` | Complex number support |
| `num-integer` | `scirs2_core::numeric` | Integer traits |
| `nalgebra` | `scirs2_core::linalg` | Linear algebra (if needed) |

### Additional Required Abstractions for OptiRS

#### SIMD Operations (Performance-Critical)
```rust
// ✅ REQUIRED - Use unified SIMD operations
use scirs2_core::simd_ops::SimdUnifiedOps;

// Automatic SIMD optimization for gradient operations
let result = f32::simd_add(&gradients.view(), &updates.view());
let dot = f64::simd_dot(&params.view(), &grads.view());
```

#### Parallel Processing (Multi-threaded Optimization)
```rust
// ✅ REQUIRED - Use core parallel abstractions
use scirs2_core::parallel_ops::*;

// Parallel parameter updates
params.par_chunks_mut(chunk_size)
    .for_each(|chunk| update_chunk(chunk));
```

#### GPU Acceleration (optirs-gpu module)
```rust
// ✅ REQUIRED - Use core GPU abstractions
use scirs2_core::gpu::{GpuDevice, GpuKernel};

// Unified GPU interface for CUDA, Metal, OpenCL, WebGPU
let device = GpuDevice::default()?;
```

#### Error Handling (All Modules)
```rust
// ✅ REQUIRED - Use core error types
use scirs2_core::error::{CoreError, Result};
use scirs2_core::validation::{check_positive, check_finite};

// Consistent error handling across OptiRS
pub type OptiRsResult<T> = Result<T, OptiRsError>;

#[derive(Debug, thiserror::Error)]
pub enum OptiRsError {
    #[error(transparent)]
    Core(#[from] CoreError),
    // OptiRS-specific errors...
}
```

#### Memory Management (Large-Scale Optimization)
```rust
// ✅ REQUIRED - Use core memory-efficient operations
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray};
use scirs2_core::memory::{BufferPool, GlobalBufferPool};

// Memory-efficient gradient accumulation
let buffer_pool = GlobalBufferPool::get();
```

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION (CRITICAL)
- **Use Cases**:
  - Core scientific primitives, random number generation (replaces rand/rand_distr)
  - Array operations (replaces ndarray)
  - Numerical traits (replaces num-traits/num-complex)
  - SIMD operations, GPU abstractions, parallel processing
  - Error handling, memory management
- **OptiRS Modules**: **ALL MODULES** - Foundation for everything
- **Status**: ✅ REQUIRED - Absolutely mandatory foundation crate
- **Features**: Enable `["array", "random", "simd", "parallel", "gpu"]` as needed

#### `scirs2-optimize` - OPTIMIZATION BASE
- **Use Cases**: Base optimization interfaces, optimizer trait definitions
- **OptiRS Modules**: `optimizers/`, `second_order/`, `adaptive_selection/`
- **Status**: ✅ REQUIRED - Core optimization functionality

### **HIGHLY LIKELY REQUIRED**

#### `scirs2-linalg` - LINEAR ALGEBRA
- **Use Cases**: Matrix/vector operations for gradients, parameters, second-order methods
- **OptiRS Modules**: `optimizers/`, `gradient_processing/`, `second_order/`
- **Status**: 🔶 INVESTIGATE - Check if OptiRS does matrix ops beyond ndarray

#### `scirs2-autograd` - AUTOMATIC DIFFERENTIATION
- **Use Cases**: Gradient computation, backpropagation integration, **array! macro access**
- **OptiRS Modules**: `neural_integration/`, `gradient_accumulation/`, **all test modules**
- **Status**: ✅ REQUIRED - Provides array! macro for tests
- **Special Note**: The `array!` macro is accessed via `scirs2_autograd::ndarray::array` since scirs2-core doesn't re-export it

#### `scirs2-neural` - NEURAL NETWORKS
- **Use Cases**: Neural network layer integration, activation functions
- **OptiRS Modules**: `neural_integration/`, `neuromorphic/`
- **Status**: 🔶 INVESTIGATE - If OptiRS has NN-specific optimizers

#### `scirs2-metrics` - PERFORMANCE MONITORING
- **Use Cases**: Benchmark suites, performance profiling, optimization metrics
- **OptiRS Modules**: `benchmarking/`, `metrics/`, `coordination/monitoring/`
- **Status**: 🔶 INVESTIGATE - If OptiRS provides benchmarking tools

#### `scirs2-stats` - STATISTICAL ANALYSIS
- **Use Cases**: Convergence analysis, statistical tests, distribution sampling
- **OptiRS Modules**: `research/`, `privacy/`, convergence detection
- **Status**: 🔶 INVESTIGATE - For optimization analysis features

### **CONDITIONALLY REQUIRED**

#### `scirs2-cluster` - CLUSTERING
- **Use Cases**: Parameter clustering, optimization landscape analysis
- **OptiRS Modules**: `research/`, `adaptive_selection/`
- **Status**: ⚠️ CONDITIONAL - Only if OptiRS does clustering analysis

#### `scirs2-datasets` - DATA HANDLING
- **Use Cases**: Synthetic data generation for benchmarking
- **OptiRS Modules**: `benchmarking/`, `research/`
- **Status**: ⚠️ CONDITIONAL - Only if OptiRS generates synthetic datasets

#### `scirs2-signal` - SIGNAL PROCESSING
- **Use Cases**: Gradient smoothing, signal filtering in optimization
- **OptiRS Modules**: `neuromorphic/`, `gradient_processing/`
- **Status**: ⚠️ CONDITIONAL - Only if OptiRS does signal processing

#### `scirs2-series` - TIME SERIES
- **Use Cases**: Time-series analysis of optimization trajectories
- **OptiRS Modules**: `coordination/monitoring/`, `research/`
- **Status**: ⚠️ CONDITIONAL - Only if OptiRS analyzes optimization over time

### **LIKELY NOT REQUIRED**

#### `scirs2-vision` - COMPUTER VISION
- **Status**: ❌ UNLIKELY - OptiRS is optimization-focused, not vision-focused

#### `scirs2-text` - TEXT PROCESSING
- **Status**: ❌ UNLIKELY - OptiRS doesn't process text data

#### `scirs2-fft` - FAST FOURIER TRANSFORM
- **Status**: ❌ UNLIKELY - No obvious FFT use in optimization

#### `scirs2-graph` - GRAPH ALGORITHMS
- **Status**: ❌ UNLIKELY - Unless OptiRS does graph-based optimization

#### `scirs2-spatial` - SPATIAL DATA
- **Status**: ❌ UNLIKELY - No spatial data processing in optimization

#### `scirs2-ndimage` - IMAGE PROCESSING
- **Status**: ❌ UNLIKELY - No image processing in optimization core

#### `scirs2-transform` - MATHEMATICAL TRANSFORMS
- **Status**: ❌ UNLIKELY - Unless specific transforms needed

#### `scirs2-interpolate` - INTERPOLATION
- **Status**: ❌ UNLIKELY - Unless learning rate scheduling uses interpolation

#### `scirs2-integrate` - NUMERICAL INTEGRATION
- **Status**: ❌ UNLIKELY - Unless optimization uses integration

#### `scirs2-sparse` - SPARSE MATRICES
- **Status**: ❌ UNLIKELY - Unless OptiRS specializes in sparse optimization

#### `scirs2-special` - SPECIAL FUNCTIONS
- **Status**: ❌ UNLIKELY - Unless specialized mathematical functions needed

#### `scirs2-io` - INPUT/OUTPUT
- **Status**: ❌ UNLIKELY - Basic I/O likely sufficient

## Integration Guidelines

### **Adding New SciRS2 Dependencies**

1. **Document Justification**
   ```markdown
   ## SciRS2 Crate Addition Request

   **Crate**: scirs2-[name]
   **Requestor**: [Developer Name]
   **Date**: [Date]

   **Justification**:
   - Specific OptiRS feature requiring this crate
   - Code modules that will use it
   - Alternatives considered and why SciRS2 is preferred

   **Impact Assessment**:
   - Compilation time impact
   - Binary size impact
   - Maintenance burden
   ```

2. **Code Review Requirements**
   - Demonstrate actual usage in OptiRS code
   - Show integration examples
   - Verify no equivalent functionality exists in already-included crates

3. **Documentation Requirements**
   - Update this policy document
   - Document usage patterns in relevant module docs
   - Add examples to integration tests

### **Removing SciRS2 Dependencies**

1. **Regular Audits** (quarterly)
   - Review all SciRS2 dependencies for actual usage
   - Remove unused imports and dependencies
   - Update documentation

2. **Deprecation Process**
   - Mark as deprecated with removal timeline
   - Provide migration guide if functionality moves
   - Remove after deprecation period

### **Best Practices**

1. **Import Granularity**
   ```rust
   // ✅ GOOD - Specific imports
   use scirs2_core::random::Random;
   use scirs2_optimize::optimizers::OptimizerTrait;

   // ❌ BAD - Broad imports
   use scirs2_core::*;
   use scirs2_optimize::*;
   ```

2. **Array Macro Import Pattern**
   ```rust
   // ✅ CORRECT - Use scirs2_autograd for array! macro
   use scirs2_autograd::ndarray::array;

   // ❌ WRONG - This won't work
   use scirs2_core::ndarray_ext::array;  // array! macro not re-exported here

   // ❌ WRONG - Don't use ndarray directly
   use ndarray::array;  // Violates SciRS2 integration policy

   // Example usage in tests:
   #[cfg(test)]
   mod tests {
       use super::*;
       use scirs2_autograd::ndarray::array;

       #[test]
       fn test_example() {
           let data = array![1.0, 2.0, 3.0];
           // test implementation
       }
   }
   ```

3. **Feature Gates**
   ```rust
   // ✅ GOOD - Optional features
   #[cfg(feature = "neural-integration")]
   use scirs2_neural::networks::NeuralNetwork;
   ```

4. **Error Handling**
   ```rust
   // ✅ GOOD - Proper error context
   use scirs2_core::ScientificNumber;
   // Document why SciRS2 types are used over alternatives
   ```

## Enforcement

### **Automated Checks**
- CI pipeline checks for unused SciRS2 dependencies
- Documentation tests verify integration examples work
- Dependency graph analysis in builds

### **Manual Reviews**
- All SciRS2 integration changes require team review
- Quarterly dependency audits
- Annual architecture review

### **Violation Response**
1. **Warning**: Document why integration is needed
2. **Correction**: Remove unjustified dependencies
3. **Training**: Educate team on integration policy

## Future Considerations

### **SciRS2 Version Management**
- Track SciRS2 release cycle
- Test OptiRS against SciRS2 beta releases
- Coordinate breaking change migrations

### **Performance Monitoring**
- Benchmark impact of SciRS2 integration
- Monitor compilation times
- Track binary size impact

### **Community Alignment**
- Coordinate with SciRS2 team on roadmap
- Contribute improvements back to SciRS2
- Maintain architectural consistency

## Conclusion

This policy ensures OptiRS properly leverages SciRS2's scientific computing foundation while maintaining a clean, minimal, and justified dependency graph. **OptiRS must use SciRS2, but intelligently and purposefully.**

---

**Document Version**: 1.1
**Last Updated**: 2025-09-20
**Next Review**: Q1 2026
**Owner**: OptiRS Architecture Team

## Quick Reference

### Current Recommended Integration (Minimal Start)
```toml
# Essential SciRS2 dependencies for OptiRS
scirs2-core = { path = "../scirs/scirs2-core" }      # Always required - foundation
scirs2-optimize = { path = "../scirs/scirs2-optimize" }  # Core optimization interfaces
scirs2-autograd = { path = "../scirs/scirs2-autograd" } # Required for array! macro in tests

# Add these only when needed:
# scirs2-linalg = { path = "../scirs/scirs2-linalg" }     # If doing matrix operations
# scirs2-neural = { path = "../scirs/scirs2-neural" }     # If NN-specific features
# scirs2-metrics = { path = "../scirs/scirs2-metrics" }   # If benchmarking tools
# scirs2-stats = { path = "../scirs/scirs2-stats" }       # If statistical analysis
```

**Remember**: Start minimal, add based on evidence, document everything!