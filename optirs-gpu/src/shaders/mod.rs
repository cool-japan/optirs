//! Compute-shader sources for the GPU optimizer steps, in WGSL and MSL.
//!
//! # Why these shaders exist instead of `scirs2_core`'s registry kernels
//!
//! `scirs2-core` 0.6.5 registers `adam_optimizer`, `sgd_optimizer`,
//! `rmsprop_optimizer`, `adagrad_optimizer` and `lamb_optimizer` with real WGSL
//! bodies, but they are not drivable-as-correct through the public API:
//!
//! * every hyper-parameter lives in a multi-field `var<uniform>` block, and the
//!   wgpu backend packs those scalars by iterating a
//!   `HashMap<String, KernelParam>` (`gpu/backends/wgpu.rs`,
//!   `create_bind_group_from_params`). The byte order of the resulting uniform
//!   buffer is therefore the map's iteration order — effectively random per
//!   process — and there is no public `set_bytes` to bypass it;
//! * their `metal_source` is empty, so a Metal context resolves them to an
//!   empty shader.
//!
//! The shaders here carry every scalar in a **storage buffer** bound by name,
//! which is deterministic on both backends, and are compiled through
//! [`scirs2_core::gpu::GpuCompiler::compile`] (real `naga` validation on wgpu,
//! a real `MTLLibrary` on Metal).
//!
//! # The buffer naming convention
//!
//! The Metal backend binds buffers to argument-table indices by looking their
//! names up in the fixed list `["x", "y", "a", "b", "result", "output"]` and
//! only then falls back to a non-deterministic hash-map order. Every kernel
//! below therefore uses **only** those six names, in that order, so the binding
//! indices are fully determined. The wgpu backend binds by name against the
//! WGSL declarations, so the same names work there unchanged.
//!
//! The per-kernel meaning of each name is documented on each source constant.
//!
//! # Scalar packing convention
//!
//! Every kernel takes a hyper-parameter buffer of `f32`. Integer fields are
//! carried through it bit-for-bit (`bitcast<u32>` in WGSL, `as_type<uint>` in
//! MSL) so element counts above 2^24 stay exact.

pub mod msl;
pub mod wgsl;

use scirs2_core::gpu::GpuBackend;

/// Threads per workgroup / threadgroup used by every optimizer kernel.
///
/// The Metal backend in scirs2-core hard-codes a 256-wide threadgroup, so this
/// value is not free to change.
pub const WORKGROUP_SIZE: usize = 256;

/// The optimizer kernels this crate ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizerKernel {
    /// Adam with coupled L2 weight decay.
    Adam,
    /// AdamW with decoupled weight decay.
    AdamW,
    /// SGD with optional momentum / Nesterov.
    Sgd,
    /// RMSprop with optional centering and momentum.
    Rmsprop,
    /// Adagrad with learning-rate decay.
    Adagrad,
    /// LAMB (two-phase, one pipeline).
    Lamb,
}

impl OptimizerKernel {
    /// Stable identifier, used to key the compiled-pipeline cache.
    pub fn id(self) -> &'static str {
        match self {
            Self::Adam => "adam",
            Self::AdamW => "adamw",
            Self::Sgd => "sgd",
            Self::Rmsprop => "rmsprop",
            Self::Adagrad => "adagrad",
            Self::Lamb => "lamb",
        }
    }

    /// Shader source for `backend`, or `None` when that backend has no source.
    pub fn source_for(self, backend: GpuBackend) -> Option<&'static str> {
        match backend {
            GpuBackend::Wgpu => Some(match self {
                Self::Adam => wgsl::ADAM,
                Self::AdamW => wgsl::ADAMW,
                Self::Sgd => wgsl::SGD,
                Self::Rmsprop => wgsl::RMSPROP,
                Self::Adagrad => wgsl::ADAGRAD,
                Self::Lamb => wgsl::LAMB,
            }),
            GpuBackend::Metal => Some(match self {
                Self::Adam => msl::ADAM,
                Self::AdamW => msl::ADAMW,
                Self::Sgd => msl::SGD,
                Self::Rmsprop => msl::RMSPROP,
                Self::Adagrad => msl::ADAGRAD,
                Self::Lamb => msl::LAMB,
            }),
            _ => None,
        }
    }

    /// Cache key combining the kernel and the backend it was compiled for.
    pub fn cache_key(self, backend: GpuBackend) -> &'static str {
        // Backends never mix within one context, so the kernel id alone is a
        // sufficient key; this indirection exists so the invariant is explicit.
        let _ = backend;
        self.id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [OptimizerKernel; 6] = [
        OptimizerKernel::Adam,
        OptimizerKernel::AdamW,
        OptimizerKernel::Sgd,
        OptimizerKernel::Rmsprop,
        OptimizerKernel::Adagrad,
        OptimizerKernel::Lamb,
    ];

    /// The Metal backend extracts the entry point with
    /// `source.find("kernel void ")` up to the next `(`, so the name and the
    /// opening paren must sit on one line.
    #[test]
    fn msl_entry_points_are_extractable() {
        for kernel in ALL {
            let source = kernel
                .source_for(GpuBackend::Metal)
                .expect("every kernel has MSL");
            let start = source
                .find("kernel void ")
                .expect("MSL source declares a kernel");
            let rest = &source[start + "kernel void ".len()..];
            let paren = rest.find('(').expect("entry point is followed by '('");
            let name = &rest[..paren];
            assert!(
                !name.contains('\n'),
                "{}: entry point spans lines",
                kernel.id()
            );
            assert!(
                !name.trim().is_empty(),
                "{}: empty entry point",
                kernel.id()
            );
        }
    }

    /// scirs2-core's WGSL reflection parser is line-oriented: `@group`,
    /// `@binding` and the `var<...>` declaration must share a line, read-only
    /// storage must be spelled exactly `var<storage, read>`, and `@compute`
    /// must sit on the same line as `fn main(`.
    #[test]
    fn wgsl_matches_the_reflection_parser() {
        for kernel in ALL {
            let source = kernel
                .source_for(GpuBackend::Wgpu)
                .expect("every kernel has WGSL");
            let id = kernel.id();

            assert!(
                !source.contains("var<uniform>"),
                "{id}: uniform blocks are packed in non-deterministic order"
            );
            assert!(
                !source.contains("var<storage,read"),
                "{id}: `var<storage,read>` is not recognised; use `var<storage, read>`"
            );

            let mut entry_lines = 0;
            for line in source.lines() {
                let trimmed = line.trim();
                if trimmed.contains("@compute") {
                    entry_lines += 1;
                    assert!(
                        trimmed.contains("fn main("),
                        "{id}: @compute must share its line with `fn main(`"
                    );
                }
                if trimmed.contains("@binding(") {
                    assert!(
                        trimmed.contains("@group(0)"),
                        "{id}: every binding must be in @group(0)"
                    );
                    assert!(
                        trimmed.contains("var<"),
                        "{id}: binding attributes must share the declaration line"
                    );
                }
            }
            assert_eq!(entry_lines, 1, "{id}: expected exactly one entry point");
        }
    }

    /// Only the six names the Metal argument-table mapping understands may be
    /// used, otherwise binding indices become hash-map dependent.
    #[test]
    fn only_deterministic_buffer_names_are_used() {
        const ALLOWED: [&str; 6] = ["x", "y", "a", "b", "result", "output"];
        for kernel in ALL {
            let source = kernel
                .source_for(GpuBackend::Wgpu)
                .expect("every kernel has WGSL");
            for line in source.lines() {
                let trimmed = line.trim();
                if !trimmed.contains("@binding(") {
                    continue;
                }
                let after = trimmed
                    .split_once('>')
                    .map(|(_, rest)| rest)
                    .unwrap_or_default();
                let name = after
                    .split_once(':')
                    .map(|(n, _)| n.trim())
                    .unwrap_or_default();
                assert!(
                    ALLOWED.contains(&name),
                    "{}: buffer name {name:?} is not in the deterministic set {ALLOWED:?}",
                    kernel.id()
                );
            }
        }
    }

    #[test]
    fn unsupported_backends_have_no_source() {
        assert!(OptimizerKernel::Adam.source_for(GpuBackend::Cpu).is_none());
        assert!(OptimizerKernel::Adam.source_for(GpuBackend::Cuda).is_none());
        assert!(OptimizerKernel::Adam
            .source_for(GpuBackend::OpenCL)
            .is_none());
    }
}
