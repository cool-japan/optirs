# SciRS2 Integration Policy for OptiRS

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**OptiRS MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, minimal, and effective integration of SciRS2 crates into OptiRS.

## Core Integration Principles

### 1. **Foundation, Not Dependency Bloat**
- OptiRS extends SciRS2's capabilities with ML optimization specialization
- Use SciRS2 crates **only when actually needed** by OptiRS functionality
- **DO NOT** add SciRS2 crates "just in case" - add them when code requires them

### 2. **Evidence-Based Integration**
- Each SciRS2 crate must have **clear justification** based on OptiRS features
- Document **specific use cases** for each integrated SciRS2 crate
- Remove unused SciRS2 dependencies during code reviews

### 3. **Architectural Hierarchy**
```
OptiRS (ML Optimization Specialization)
    ↓ builds upon
SciRS2 (Scientific Computing Foundation)
    ↓ builds upon
ndarray, num-traits, etc. (Core Rust Scientific Stack)
```

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION
- **Use Cases**: Core scientific primitives, ScientificNumber trait, random number generation
- **OptiRS Modules**: All modules use core utilities
- **Status**: ✅ REQUIRED - Foundation crate

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