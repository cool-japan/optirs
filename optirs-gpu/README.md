# OptiRS GPU

GPU acceleration and multi-GPU optimization for the OptiRS machine learning optimization library.

## Overview

OptiRS-GPU provides hardware acceleration for machine learning optimization workloads across multiple GPU backends. This crate enables high-performance, parallel optimization on CUDA, Metal, OpenCL, and WebGPU platforms, with automatic device selection and memory management.

## Features

- **Multi-Backend Support**: CUDA, Metal, OpenCL, and WebGPU compatibility
- **Automatic Device Detection**: Intelligent GPU selection and fallback mechanisms
- **Multi-GPU Training**: Distributed optimization across multiple GPUs
- **Memory Management**: Efficient GPU memory allocation and transfer
- **Async Operations**: Non-blocking GPU kernel execution
- **Cross-Platform**: Unified API across different GPU ecosystems
- **Performance Monitoring**: GPU utilization and memory usage tracking

## Supported Backends

### CUDA (NVIDIA)
- Full CUDA runtime integration via cudarc
- Support for CUDA 12.0+ features
- Optimized kernels for tensor operations
- Multi-GPU scaling with NCCL integration

### Metal (Apple Silicon)
- Native Metal Performance Shaders integration
- Optimized for Apple M1/M2/M3 architecture
- Unified memory architecture support
- Metal Performance Shaders Graph integration

### OpenCL (Cross-platform)
- OpenCL 3.0 support for maximum compatibility
- Vendor-agnostic GPU acceleration
- Support for Intel, AMD, and other OpenCL devices
- Custom kernel compilation and caching

### WebGPU (Web/Native)
- Cross-platform GPU acceleration via wgpu
- WebAssembly compatibility for web deployment
- Vulkan, DirectX, and Metal backend support
- Portable compute shaders

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-gpu = "0.1.0-rc.1"
scirs2-core = "0.1.0-rc.2"  # Required foundation
```

### Feature Selection

Enable specific GPU backends:

```toml
[dependencies]
optirs-gpu = { version = "0.1.0-rc.1", features = ["cuda", "metal"] }
```

Available features:
- `cuda`: NVIDIA CUDA support
- `metal`: Apple Metal support
- `opencl`: OpenCL support
- `wgpu`: WebGPU support (enabled by default)

## Usage

### Basic GPU Optimization

```rust
use optirs_gpu::{GpuOptimizer, DeviceManager};
use optirs_core::optimizers::Adam;

// Initialize GPU device manager
let device_manager = DeviceManager::new().await?;
let device = device_manager.select_best_device()?;

// Create GPU-accelerated optimizer
let mut optimizer = GpuOptimizer::new(device)
    .with_optimizer(Adam::new(0.001))
    .build()?;

// Your model parameters (automatically transferred to GPU)
let mut params = optimizer.create_tensor(&[1024, 512])?;
let grads = optimizer.create_tensor_from_slice(&gradient_data)?;

// Perform optimization step on GPU
optimizer.step(&mut params, &grads).await?;
```

### Multi-GPU Training

```rust
use optirs_gpu::{MultiGpuOptimizer, DataParallelStrategy};

// Setup multi-GPU training
let mut multi_gpu = MultiGpuOptimizer::new()
    .with_strategy(DataParallelStrategy::AllReduce)
    .with_devices(&device_manager.available_devices())
    .build()?;

// Distribute model across GPUs
multi_gpu.distribute_model(&model_parameters).await?;

// Synchronized optimization across all GPUs
multi_gpu.step_synchronized(&gradients).await?;
```

### Backend-Specific Features

#### CUDA-Specific Operations
```rust
use optirs_gpu::cuda::{CudaContext, CudaStream};

let cuda_ctx = CudaContext::new(device_id)?;
let stream = CudaStream::new(&cuda_ctx)?;

// Custom CUDA kernels
let result = stream.launch_kernel("custom_optimizer", &params, &config).await?;
```

#### Metal-Specific Operations
```rust
use optirs_gpu::metal::{MetalDevice, MetalCommandQueue};

let metal_device = MetalDevice::system_default()?;
let command_queue = metal_device.new_command_queue();

// Metal Performance Shaders integration
let mps_optimizer = command_queue.create_mps_optimizer(&config)?;
```

## Architecture

### Device Management
- Automatic GPU detection and selection
- Device capability querying
- Memory and compute capability assessment
- Fallback mechanism for unsupported operations

### Memory Management
- Efficient GPU memory allocation
- Automatic CPU-GPU data transfer
- Memory pool management
- Garbage collection for GPU resources

### Kernel Management
- Optimized compute kernels for common operations
- JIT compilation and caching
- Kernel fusion for performance optimization
- Platform-specific optimizations

## Performance Considerations

### Memory Transfer Optimization
- Asynchronous CPU-GPU transfers
- Memory pinning for faster transfers
- Batch operations to reduce transfer overhead
- Smart caching of frequently accessed data

### Compute Optimization
- Kernel fusion for reduced memory bandwidth
- Optimal thread block sizing
- Memory coalescing patterns
- Shared memory utilization

### Multi-GPU Scaling
- Efficient gradient synchronization
- Load balancing across devices
- Topology-aware communication
- Overlap communication and computation

## Error Handling

OptiRS-GPU provides comprehensive error handling for GPU operations:

```rust
use optirs_gpu::error::{GpuError, GpuResult};

match optimizer.step(&mut params, &grads).await {
    Ok(()) => println!("Optimization successful"),
    Err(GpuError::OutOfMemory) => {
        // Handle GPU memory exhaustion
        optimizer.clear_cache()?;
    }
    Err(GpuError::DeviceNotAvailable) => {
        // Fallback to CPU optimization
        fallback_to_cpu_optimizer()?;
    }
    Err(e) => return Err(e.into()),
}
```

## Benchmarking

OptiRS-GPU includes built-in benchmarking tools:

```rust
use optirs_gpu::benchmarks::{GpuBenchmark, BenchmarkConfig};

let benchmark = GpuBenchmark::new()
    .with_config(BenchmarkConfig::default())
    .with_optimizer(optimizer)
    .build()?;

let results = benchmark.run_performance_suite().await?;
println!("GPU throughput: {:.2} GFLOPS", results.throughput);
```

## Platform Support

| Platform | CUDA | Metal | OpenCL | WebGPU |
|----------|------|-------|--------|--------|
| Linux    | ✅   | ❌    | ✅     | ✅     |
| macOS    | ❌   | ✅    | ✅     | ✅     |
| Windows  | ✅   | ❌    | ✅     | ✅     |
| Web      | ❌   | ❌    | ❌     | ✅     |

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS repository for contribution guidelines.

## License

This project is licensed under either of:
- Apache License, Version 2.0
- MIT License

at your option.