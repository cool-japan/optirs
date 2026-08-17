//! Every shipped shader must be accepted by a real compiler.
//!
//! The WGSL half is checked with `scirs2_core::gpu::backends::try_compile_wgsl`,
//! which builds a `WebGPUContext` directly and runs the full `naga` validation
//! plus `wgpu` compute-pipeline creation. That entry point matters because
//! `GpuContext::new(GpuBackend::Wgpu)` is currently unreachable in scirs2-core
//! 0.6.5 (its runtime device probe never enumerates wgpu adapters), so without
//! it the WGSL sources would ship having never been seen by a compiler.
//!
//! The MSL half is checked by compiling through a real Metal `GpuContext`.

use optirs_gpu::shaders::OptimizerKernel;
use scirs2_core::gpu::{GpuBackend, GpuContext};

const ALL: [OptimizerKernel; 6] = [
    OptimizerKernel::Adam,
    OptimizerKernel::AdamW,
    OptimizerKernel::Sgd,
    OptimizerKernel::Rmsprop,
    OptimizerKernel::Adagrad,
    OptimizerKernel::Lamb,
];

/// Compile every WGSL kernel with `naga` + `wgpu`.
#[cfg(feature = "wgpu")]
#[test]
fn wgsl_kernels_compile() {
    use scirs2_core::gpu::backends::try_compile_wgsl;

    // A trivially valid shader tells us whether an adapter exists at all, so a
    // missing adapter is reported as a skip rather than as a shader failure.
    const PROBE: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    x[gid.x] = x[gid.x] + 1.0;
}
"#;
    if let Err(e) = try_compile_wgsl(PROBE) {
        eprintln!("SKIP: wgsl_kernels_compile — no WebGPU adapter available ({e})");
        return;
    }
    eprintln!("WGSL: compiling {} kernels through naga + wgpu", ALL.len());

    for kernel in ALL {
        let source = kernel
            .source_for(GpuBackend::Wgpu)
            .unwrap_or_else(|| panic!("{} has no WGSL source", kernel.id()));
        let pipeline = try_compile_wgsl(source)
            .unwrap_or_else(|e| panic!("{} WGSL failed to compile: {e}", kernel.id()));
        assert_eq!(
            pipeline.workgroup_size,
            [optirs_gpu::shaders::WORKGROUP_SIZE as u32, 1, 1],
            "{} declares an unexpected workgroup size",
            kernel.id()
        );
        eprintln!("WGSL: {} compiled", kernel.id());
    }
}

/// Compile every MSL kernel with the real Metal shader compiler.
#[cfg(feature = "metal")]
#[test]
fn msl_kernels_compile() {
    let context = match GpuContext::new(GpuBackend::Metal) {
        Ok(context) => context,
        Err(e) => {
            eprintln!("SKIP: msl_kernels_compile — no Metal device available ({e})");
            return;
        }
    };
    eprintln!("MSL: compiling {} kernels through MTLLibrary", ALL.len());

    for kernel in ALL {
        let source = kernel
            .source_for(GpuBackend::Metal)
            .unwrap_or_else(|| panic!("{} has no MSL source", kernel.id()));
        context
            .execute(|compiler| compiler.compile(source))
            .unwrap_or_else(|e| panic!("{} MSL failed to compile: {e}", kernel.id()));
        eprintln!("MSL: {} compiled", kernel.id());
    }
}

/// Invalid source must be rejected, so a passing compile test means something.
#[cfg(feature = "metal")]
#[test]
fn broken_msl_is_rejected() {
    let context = match GpuContext::new(GpuBackend::Metal) {
        Ok(context) => context,
        Err(e) => {
            eprintln!("SKIP: broken_msl_is_rejected — no Metal device available ({e})");
            return;
        }
    };
    let broken =
        "#include <metal_stdlib>\nkernel void bad(device float* x [[buffer(0)]]) { x[0] = ; }\n";
    assert!(
        context
            .execute(|compiler| compiler.compile(broken))
            .is_err(),
        "the Metal compiler accepted syntactically invalid MSL"
    );
}
