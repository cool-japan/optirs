# OptiRS TODO - Full SciRS2 Integration

## 🚨 CRITICAL: Full SciRS2-Core Integration Tasks

### Phase 0: Immediate SciRS2 Migration (HIGHEST PRIORITY)
- [ ] **Remove ALL direct ndarray imports**
  - [ ] Audit all `use ndarray::` statements
  - [ ] Replace with `scirs2_core::ndarray_ext::`
  - [ ] Update all Array, ArrayView, ArrayViewMut imports
  - [ ] Verify compilation after ndarray removal

- [ ] **Remove ALL direct rand imports**
  - [ ] Audit all `use rand::` and `use rand_distr::` statements
  - [ ] Replace with `scirs2_core::random::`
  - [ ] Update all RNG and distribution usage
  - [ ] Verify random number generation still works

- [ ] **Migrate to SciRS2 error handling**
  - [ ] Replace custom error types with `scirs2_core::error::CoreError`
  - [ ] Use `scirs2_core::Result` throughout
  - [ ] Update error propagation patterns

### Phase 1: Core Module SciRS2 Integration (v0.1.1)

#### optirs-core
- [ ] **Array Operations Migration**
  - [ ] Use `scirs2_core::ndarray_ext::stats` for statistical operations
  - [ ] Use `scirs2_core::ndarray_ext::matrix` for matrix operations
  - [ ] Use `scirs2_core::ndarray_ext::manipulation` for array manipulation
  - [ ] Use `scirs2_core::ndarray_ext::indexing` for NumPy-like indexing

- [ ] **Performance Optimization**
  - [ ] Implement SIMD optimization using `scirs2_core::simd_ops`
  - [ ] Add parallel processing with `scirs2_core::parallel_ops`
  - [ ] Use `scirs2_core::memory::BufferPool` for memory management
  - [ ] Integrate `scirs2_core::memory_efficient` for large gradients

- [ ] **Production Features**
  - [ ] Add profiling with `scirs2_core::profiling::Profiler`
  - [ ] Implement metrics with `scirs2_core::metrics`
  - [ ] Add benchmarking with `scirs2_core::benchmarking`
  - [ ] Use `scirs2_core::validation` for input validation

### Phase 2: GPU Module SciRS2 Integration (v0.1.2)

#### optirs-gpu
- [ ] **GPU Abstraction Migration**
  - [ ] Build on top of `scirs2_core::gpu::GpuContext`
  - [ ] Use `scirs2_core::gpu::GpuBuffer` for memory management
  - [ ] Implement kernels using `scirs2_core::gpu::GpuKernel`
  - [ ] Leverage `scirs2_core::tensor_cores` for mixed precision

- [ ] **Array Protocol Integration**
  - [ ] Implement `scirs2_core::array_protocol::GPUArray`
  - [ ] Support `scirs2_core::array_protocol::AsyncArray`
  - [ ] Use `scirs2_core::memory::TrackedGpuBuffer` for tracking

- [ ] **Multi-Backend Support**
  - [ ] Use `scirs2_core::gpu::CudaBackend`
  - [ ] Use `scirs2_core::gpu::MetalBackend`
  - [ ] Abstract backends with SciRS2 interfaces

### Phase 3: TPU Module SciRS2 Integration (v0.1.3)

#### optirs-tpu
- [ ] **Distributed Computing**
  - [ ] Use `scirs2_core::distributed::ClusterManager`
  - [ ] Implement with `scirs2_core::distributed::JobScheduler`
  - [ ] Use `scirs2_core::advanced_distributed_computing::AllReduce`
  - [ ] Support `scirs2_core::array_protocol::DistributedArray`

- [ ] **XLA Integration**
  - [ ] Use `scirs2_core::jit::JitCompiler` for XLA
  - [ ] Leverage `scirs2_core::jit::OptimizationLevel`
  - [ ] Apply `scirs2_core::advanced_jit_compilation`

### Phase 4: Learned Optimizers SciRS2 Integration (v0.1.4)

#### optirs-learned
- [ ] **ML Pipeline Integration**
  - [ ] Use `scirs2_core::ml_pipeline::MLPipeline`
  - [ ] Implement `scirs2_core::ml_pipeline::ModelPredictor`
  - [ ] Use `scirs2_core::ml_pipeline::FeatureTransformer`
  - [ ] Track with `scirs2_core::ml_pipeline::PipelineMetrics`

- [ ] **Neural Architecture Search Foundation**
  - [ ] Build on `scirs2_core::neural_architecture_search`
  - [ ] Use `scirs2_core::neural_architecture_search::SearchSpace`
  - [ ] Leverage `scirs2_core::quantum_optimization` for search

- [ ] **Memory Optimization**
  - [ ] Use `scirs2_core::memory_efficient::LazyArray` for history
  - [ ] Apply `scirs2_core::memory_efficient::AdaptiveChunking`
  - [ ] Implement gradient compression with SciRS2 tools

### Phase 5: NAS Module SciRS2 Integration (v0.1.5)

#### optirs-nas
- [ ] **Search Infrastructure**
  - [ ] Use `scirs2_core::neural_architecture_search::NeuralArchitectureSearch`
  - [ ] Implement with `scirs2_core::neural_architecture_search::SearchSpace`
  - [ ] Apply `scirs2_core::neural_architecture_search::ArchitecturePerformance`

- [ ] **Optimization Strategies**
  - [ ] Use `scirs2_core::quantum_optimization::QuantumOptimizer`
  - [ ] Apply `scirs2_core::quantum_optimization::QuantumStrategy`
  - [ ] Leverage `scirs2_core::parallel::LoadBalancer` for parallel search

### Phase 6: Benchmarking SciRS2 Integration (v0.1.6)

#### optirs-bench
- [ ] **Benchmarking Framework**
  - [ ] Replace custom benchmarks with `scirs2_core::benchmarking::BenchmarkSuite`
  - [ ] Use `scirs2_core::benchmarking::BenchmarkRunner`
  - [ ] Track with `scirs2_core::benchmarking::BenchmarkStatistics`

- [ ] **Performance Analysis**
  - [ ] Use `scirs2_core::profiling::Profiler` exclusively
  - [ ] Track with `scirs2_core::metrics::MetricRegistry`
  - [ ] Monitor stability with `scirs2_core::stability`

## 🔧 Code Quality & Best Practices

### Mandatory SciRS2 Usage Rules
- [ ] **NO direct ndarray imports** - Audit weekly
- [ ] **NO direct rand imports** - Audit weekly
- [ ] **NO custom SIMD** - Use scirs2_core::simd
- [ ] **NO custom parallel** - Use scirs2_core::parallel
- [ ] **NO custom profiling** - Use scirs2_core::profiling
- [ ] **NO custom benchmarking** - Use scirs2_core::benchmarking

### Performance Requirements
- [ ] All hot paths use `scirs2_core::simd_ops`
- [ ] Large operations use `scirs2_core::parallel_ops`
- [ ] Memory-intensive ops use `scirs2_core::memory_efficient`
- [ ] GPU operations use `scirs2_core::gpu` abstractions

### Testing & Validation
- [ ] Use `scirs2_core::testing::TestSuite`
- [ ] Validate with `scirs2_core::validation`
- [ ] Benchmark with `scirs2_core::benchmarking`
- [ ] Profile with `scirs2_core::profiling`

## 📊 Success Metrics

### SciRS2 Integration Completeness
- [ ] 0% direct ndarray usage (target: 100% scirs2_core)
- [ ] 0% direct rand usage (target: 100% scirs2_core)
- [ ] 100% error handling through scirs2_core
- [ ] 100% SIMD operations through scirs2_core
- [ ] 100% parallel operations through scirs2_core
- [ ] 100% benchmarking through scirs2_core

### Performance Improvements (via SciRS2)
- [ ] 2-5x speedup from SIMD operations
- [ ] 4-8x speedup from parallel processing
- [ ] 10-50x speedup from GPU acceleration
- [ ] 50% memory reduction from efficient ops

### Code Quality
- [ ] Zero clippy warnings
- [ ] Zero unused dependencies
- [ ] 100% public API documentation
- [ ] All examples use scirs2_core

## 🚀 Long-term Vision

### Advanced SciRS2 Features to Leverage
- [ ] Cloud storage with `scirs2_core::cloud`
- [ ] JIT compilation with `scirs2_core::jit`
- [ ] Quantum optimization with `scirs2_core::quantum_optimization`
- [ ] Advanced ecosystem integration
- [ ] Production observability with `scirs2_core::observability`

### Research Integration
- [ ] Use SciRS2's neural architecture search
- [ ] Leverage SciRS2's meta-learning capabilities
- [ ] Apply SciRS2's advanced optimization research

## 🎯 Immediate Actions (DO NOW!)

1. **Audit all imports** - Find and replace ndarray/rand
2. **Update Cargo.toml** - Remove direct ndarray/rand deps
3. **Fix compilation errors** - Update to scirs2_core types
4. **Add profiling** - Use scirs2_core::profiling everywhere
5. **Update documentation** - Show scirs2_core usage

---

**Status**: 🔴 **MIGRATION REQUIRED** - Must fully integrate SciRS2-Core
**Priority**: **CRITICAL** - This blocks all other development
**Deadline**: Before any new features or releases
**Success Criteria**: Zero direct ndarray/rand usage, full SciRS2 integration