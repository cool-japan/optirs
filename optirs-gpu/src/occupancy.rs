//! CUDA Occupancy Calculator (CPU-side, analytical).
//!
//! This module implements the standard NVIDIA *occupancy* model: given a
//! kernel's per-thread/per-block resource usage and the architectural limits of
//! a streaming multiprocessor (SM), it computes how many thread blocks and warps
//! can be co-resident on a single SM, the resulting *occupancy* (the ratio of
//! resident warps to the hardware maximum), and which resource is the binding
//! constraint. It also provides an [`optimal_block_size`] search analogous to
//! `cudaOccupancyMaxPotentialBlockSize`.
//!
//! Nothing here queries a GPU — every value is derived from documented
//! architectural constants and integer arithmetic, so the calculator is
//! deterministic and works on any host.
//!
//! # Model
//!
//! For a block of `threads_per_block` threads on an SM with `warp_size` threads
//! per warp:
//!
//! ```text
//! warps_per_block      = ceil(threads_per_block / warp_size)
//! blocks_by_warps      = max_warps_per_sm / warps_per_block
//! blocks_by_registers  = registers_per_sm
//!                        / round_up(registers_per_thread * threads_per_block,
//!                                   register_alloc_granularity)
//! blocks_by_shared_mem = shared_mem_per_sm / shared_mem_per_block
//! blocks_by_cap        = max_blocks_per_sm
//!
//! active_blocks = min(blocks_by_warps, blocks_by_registers,
//!                     blocks_by_shared_mem, blocks_by_cap)
//! active_warps  = active_blocks * warps_per_block
//! occupancy     = active_warps / max_warps_per_sm
//! ```
//!
//! The register term uses a *per-block* allocation rounded up to
//! `register_alloc_granularity` (256 32-bit registers on every architecture
//! modelled here). This is a deliberately simple approximation of the hardware's
//! per-warp register allocation; for block sizes that are whole multiples of the
//! warp size (the common case) it coincides with the per-warp model.
//!
//! A `registers_per_thread` of `0` is treated as "no register pressure"
//! (unlimited), and a `shared_mem_per_block` of `0` is treated as "no shared
//! memory pressure" (unlimited), so those resources never bound occupancy.
//!
//! # Example
//!
//! ```
//! use optirs_gpu::{calculate_occupancy, KernelResourceUsage, SmResourceLimits};
//!
//! let limits = SmResourceLimits::sm_80();
//! let usage = KernelResourceUsage::new(32, 0, 256);
//! let result = calculate_occupancy(&usage, &limits).expect("valid configuration");
//! assert_eq!(result.active_blocks_per_sm, 8);
//! assert!((result.occupancy - 1.0).abs() < 1e-9);
//! ```

use crate::GpuOptimError;

/// Architectural per-SM resource limits for a streaming multiprocessor.
///
/// Instances are normally created with one of the named compute-capability
/// constructors ([`SmResourceLimits::sm_70`] … [`SmResourceLimits::sm_90`]) or
/// derived from a device report via
/// [`SmResourceLimits::from_compute_capability`] /
/// [`SmResourceLimits::from_device_capabilities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmResourceLimits {
    /// CUDA compute capability `(major, minor)` that this model represents.
    pub compute_capability: (u32, u32),

    /// Threads per warp (32 on every NVIDIA architecture to date).
    pub warp_size: u32,

    /// Maximum threads that may be launched in a single thread block.
    pub max_threads_per_block: u32,

    /// Maximum number of resident threads per SM (`max_warps_per_sm * warp_size`).
    pub max_threads_per_sm: u32,

    /// Maximum number of resident warps per SM.
    pub max_warps_per_sm: u32,

    /// Maximum number of resident thread blocks per SM (hard architectural cap).
    pub max_blocks_per_sm: u32,

    /// Number of 32-bit registers in the SM register file.
    pub registers_per_sm: u32,

    /// Granularity (in 32-bit registers) at which a block's register usage is
    /// rounded up.
    pub register_alloc_granularity: u32,

    /// Bytes of shared memory available per SM.
    pub shared_mem_per_sm_bytes: usize,
}

impl SmResourceLimits {
    /// Volta — compute capability 7.0 (e.g. Tesla V100).
    ///
    /// Documented per-SM limits (CUDA C Programming Guide, "Technical
    /// Specifications per Compute Capability"):
    /// - 64 resident warps / 2048 resident threads per SM
    /// - 32 resident thread blocks per SM
    /// - 65536 32-bit registers per SM, allocated with 256-register granularity
    /// - 96 KiB (98304 bytes) of shared memory per SM
    /// - 1024 threads per block, 32 threads per warp
    #[must_use]
    pub const fn sm_70() -> Self {
        Self {
            compute_capability: (7, 0),
            warp_size: 32,
            max_threads_per_block: 1024,
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_mem_per_sm_bytes: 98_304,
        }
    }

    /// Turing — compute capability 7.5 (e.g. RTX 2080, T4).
    ///
    /// Turing halves the resident warp/thread budget relative to Volta:
    /// - 32 resident warps / 1024 resident threads per SM
    /// - 16 resident thread blocks per SM
    /// - 65536 32-bit registers per SM, 256-register granularity
    /// - 64 KiB (65536 bytes) of shared memory per SM
    /// - 1024 threads per block, 32 threads per warp
    #[must_use]
    pub const fn sm_75() -> Self {
        Self {
            compute_capability: (7, 5),
            warp_size: 32,
            max_threads_per_block: 1024,
            max_threads_per_sm: 1024,
            max_warps_per_sm: 32,
            max_blocks_per_sm: 16,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_mem_per_sm_bytes: 65_536,
        }
    }

    /// Ampere A100 — compute capability 8.0 (datacenter GA100).
    ///
    /// - 64 resident warps / 2048 resident threads per SM
    /// - 32 resident thread blocks per SM
    /// - 65536 32-bit registers per SM, 256-register granularity
    /// - 164 KiB (167936 bytes) of shared memory per SM (opt-in maximum)
    /// - 1024 threads per block, 32 threads per warp
    #[must_use]
    pub const fn sm_80() -> Self {
        Self {
            compute_capability: (8, 0),
            warp_size: 32,
            max_threads_per_block: 1024,
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_mem_per_sm_bytes: 167_936,
        }
    }

    /// Ampere GA10x — compute capability 8.6 (consumer Ampere, e.g. RTX 3080).
    ///
    /// GA10x lowers the resident block/warp budget relative to A100:
    /// - 48 resident warps / 1536 resident threads per SM
    /// - 16 resident thread blocks per SM
    /// - 65536 32-bit registers per SM, 256-register granularity
    /// - 100 KiB (102400 bytes) of shared memory per SM (opt-in maximum)
    /// - 1024 threads per block, 32 threads per warp
    #[must_use]
    pub const fn sm_86() -> Self {
        Self {
            compute_capability: (8, 6),
            warp_size: 32,
            max_threads_per_block: 1024,
            max_threads_per_sm: 1536,
            max_warps_per_sm: 48,
            max_blocks_per_sm: 16,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_mem_per_sm_bytes: 102_400,
        }
    }

    /// Hopper — compute capability 9.0 (datacenter H100/GH100).
    ///
    /// - 64 resident warps / 2048 resident threads per SM
    /// - 32 resident thread blocks per SM
    /// - 65536 32-bit registers per SM, 256-register granularity
    /// - 228 KiB (233472 bytes) of shared memory per SM (opt-in maximum)
    /// - 1024 threads per block, 32 threads per warp
    #[must_use]
    pub const fn sm_90() -> Self {
        Self {
            compute_capability: (9, 0),
            warp_size: 32,
            max_threads_per_block: 1024,
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            registers_per_sm: 65536,
            register_alloc_granularity: 256,
            shared_mem_per_sm_bytes: 233_472,
        }
    }

    /// Build limits from a CUDA compute capability `(major, minor)`.
    ///
    /// Exact matches map to the corresponding constructor. Capabilities that
    /// belong to a known architecture family but are not modelled individually
    /// are mapped to the *nearest* modelled architecture (documented below). Any
    /// other capability — including the `(0, 0)` reported by non-CUDA devices —
    /// yields an honest [`GpuOptimError::UnsupportedOperation`] rather than a
    /// fabricated guess.
    ///
    /// Nearest-architecture mappings:
    /// - `7.2` (Volta Xavier) → [`Self::sm_70`]
    /// - `8.7` (Ampere Orin)  → [`Self::sm_86`]
    /// - `8.9` (Ada Lovelace) → [`Self::sm_86`] (closest documented per-SM budget)
    pub fn from_compute_capability(compute_capability: (u32, u32)) -> Result<Self, GpuOptimError> {
        let (major, minor) = compute_capability;
        match (major, minor) {
            (7, 0) | (7, 2) => Ok(Self::sm_70()),
            (7, 5) => Ok(Self::sm_75()),
            (8, 0) => Ok(Self::sm_80()),
            (8, 6) | (8, 7) | (8, 9) => Ok(Self::sm_86()),
            (9, 0) => Ok(Self::sm_90()),
            _ => Err(GpuOptimError::UnsupportedOperation(format!(
                "no CUDA occupancy model for compute capability {major}.{minor}"
            ))),
        }
    }

    /// Derive limits from a [`crate::backends::DeviceCapabilities`] report.
    ///
    /// All per-SM architectural constants are taken from the model keyed on the
    /// device's `compute_capability`; the per-block thread cap is overridden with
    /// the device-reported `max_threads_per_block` when that value is non-zero.
    /// Non-CUDA devices (which report a `(0, 0)` capability) produce an error,
    /// because the SM occupancy model does not apply to them.
    pub fn from_device_capabilities(
        capabilities: &crate::backends::DeviceCapabilities,
    ) -> Result<Self, GpuOptimError> {
        let mut limits = Self::from_compute_capability(capabilities.compute_capability)?;
        if capabilities.max_threads_per_block > 0 {
            limits.max_threads_per_block = capabilities.max_threads_per_block;
        }
        Ok(limits)
    }
}

/// Per-kernel resource usage that drives the occupancy calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelResourceUsage {
    /// 32-bit registers consumed by each thread (`0` means "not register-bound").
    pub registers_per_thread: u32,

    /// Static shared memory consumed by each block, in bytes
    /// (`0` means "not shared-memory-bound").
    pub shared_mem_per_block_bytes: usize,

    /// Threads launched per block.
    pub threads_per_block: u32,
}

impl KernelResourceUsage {
    /// Create a new [`KernelResourceUsage`].
    #[must_use]
    pub const fn new(
        registers_per_thread: u32,
        shared_mem_per_block_bytes: usize,
        threads_per_block: u32,
    ) -> Self {
        Self {
            registers_per_thread,
            shared_mem_per_block_bytes,
            threads_per_block,
        }
    }
}

/// The resource that bounds occupancy for a given kernel/SM combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLimiter {
    /// Limited by the maximum number of resident warps per SM.
    Warps,
    /// Limited by the SM register file.
    Registers,
    /// Limited by per-SM shared memory.
    SharedMemory,
    /// Limited by the hard cap on resident blocks per SM.
    BlocksPerSm,
    /// The block requests more threads than the hardware permits; it cannot
    /// launch, so occupancy is zero.
    ThreadsPerBlock,
}

/// Result of an occupancy calculation for a single SM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OccupancyResult {
    /// Resident warps per SM achieved by this configuration.
    pub active_warps_per_sm: u32,

    /// Resident blocks per SM achieved by this configuration.
    pub active_blocks_per_sm: u32,

    /// Hardware maximum resident warps per SM (the occupancy denominator).
    pub max_warps_per_sm: u32,

    /// Achieved occupancy in `0.0..=1.0` (`active_warps_per_sm / max_warps_per_sm`).
    pub occupancy: f64,

    /// Resource that bounds occupancy for this configuration.
    pub limiter: OccupancyLimiter,
}

/// Round `value` up to the nearest multiple of `granularity`.
///
/// `granularity == 0` disables rounding (returns `value` unchanged). The result
/// saturates to [`u64::MAX`] instead of overflowing for absurd inputs, which
/// downstream produces a zero block count (an unlaunchable kernel) rather than a
/// panic.
const fn round_up_to_multiple(value: u64, granularity: u64) -> u64 {
    if granularity == 0 {
        return value;
    }
    match value.div_ceil(granularity).checked_mul(granularity) {
        Some(rounded) => rounded,
        None => u64::MAX,
    }
}

/// Saturating narrowing of a `u64` block count to `u32`.
const fn clamp_u64_to_u32(value: u64) -> u32 {
    if value > u32::MAX as u64 {
        u32::MAX
    } else {
        value as u32
    }
}

/// Saturating narrowing of a `usize` block count to `u32`.
const fn clamp_usize_to_u32(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}

/// Compute SM occupancy for a kernel with the given resource usage.
///
/// See the [module documentation](crate::occupancy) for the full model. Returns
/// an error only for degenerate input (`threads_per_block == 0` or
/// `warp_size == 0`). A block that requests more threads than
/// `limits.max_threads_per_block` cannot launch and is reported as
/// [`OccupancyLimiter::ThreadsPerBlock`] with zero active blocks and zero
/// occupancy. When two resources tie for the binding constraint the priority
/// order is warps → registers → shared memory → block cap, so a fully occupied
/// SM is reported as [`OccupancyLimiter::Warps`].
pub fn calculate_occupancy(
    usage: &KernelResourceUsage,
    limits: &SmResourceLimits,
) -> Result<OccupancyResult, GpuOptimError> {
    let warp_size = limits.warp_size;
    let threads_per_block = usage.threads_per_block;

    if warp_size == 0 {
        return Err(GpuOptimError::InvalidState(
            "warp_size must be greater than zero".to_string(),
        ));
    }
    if threads_per_block == 0 {
        return Err(GpuOptimError::InvalidState(
            "threads_per_block must be greater than zero".to_string(),
        ));
    }

    // A block larger than the hardware permits can never launch: report zero
    // occupancy attributed to the block-size limit rather than silently clamping.
    if threads_per_block > limits.max_threads_per_block {
        return Ok(OccupancyResult {
            active_warps_per_sm: 0,
            active_blocks_per_sm: 0,
            max_warps_per_sm: limits.max_warps_per_sm,
            occupancy: 0.0,
            limiter: OccupancyLimiter::ThreadsPerBlock,
        });
    }

    let warps_per_block = threads_per_block.div_ceil(warp_size);

    // Blocks bounded by the resident-warp budget.
    let warps_limit = limits.max_warps_per_sm / warps_per_block;

    // Blocks bounded by the register file. Zero registers per thread means the
    // kernel exerts no register pressure and is never register-limited.
    let register_limit = if usage.registers_per_thread == 0 {
        u32::MAX
    } else {
        let raw_registers =
            u64::from(usage.registers_per_thread).saturating_mul(u64::from(threads_per_block));
        let registers_per_block =
            round_up_to_multiple(raw_registers, u64::from(limits.register_alloc_granularity))
                .max(1);
        clamp_u64_to_u32(u64::from(limits.registers_per_sm) / registers_per_block)
    };

    // Blocks bounded by shared memory. Zero shared memory per block means the
    // kernel exerts no shared-memory pressure and is never shared-memory-limited
    // (a zero divisor yields `None`, i.e. the unlimited sentinel).
    let shared_mem_limit = match limits
        .shared_mem_per_sm_bytes
        .checked_div(usage.shared_mem_per_block_bytes)
    {
        Some(blocks) => clamp_usize_to_u32(blocks),
        None => u32::MAX,
    };

    // Hard architectural cap on resident blocks.
    let block_cap_limit = limits.max_blocks_per_sm;

    // Pick the binding constraint. Earlier entries win ties, so a fully occupied
    // SM is attributed to `Warps`.
    let candidates = [
        (warps_limit, OccupancyLimiter::Warps),
        (register_limit, OccupancyLimiter::Registers),
        (shared_mem_limit, OccupancyLimiter::SharedMemory),
        (block_cap_limit, OccupancyLimiter::BlocksPerSm),
    ];

    let mut active_blocks = candidates[0].0;
    let mut limiter = candidates[0].1;
    for &(value, candidate_limiter) in &candidates[1..] {
        if value < active_blocks {
            active_blocks = value;
            limiter = candidate_limiter;
        }
    }

    let active_warps = active_blocks.saturating_mul(warps_per_block);
    let occupancy = if limits.max_warps_per_sm == 0 {
        0.0
    } else {
        f64::from(active_warps) / f64::from(limits.max_warps_per_sm)
    };

    Ok(OccupancyResult {
        active_warps_per_sm: active_warps,
        active_blocks_per_sm: active_blocks,
        max_warps_per_sm: limits.max_warps_per_sm,
        occupancy,
        limiter,
    })
}

/// Search for the block size that maximises occupancy (à la
/// `cudaOccupancyMaxPotentialBlockSize`).
///
/// Candidate block sizes are every multiple of `limits.warp_size` from
/// `warp_size` up to and including `limits.max_threads_per_block`. The usage
/// model is intentionally simple and documented:
/// - `registers_per_thread` is a constant independent of block size (the usual
///   assumption — register usage is a property of the compiled kernel).
/// - `shared_mem_per_block` is a closure `block_size -> bytes`, which covers both
///   a constant footprint (`|_| BYTES`) and block-size-dependent allocations such
///   as a reduction kernel's `|threads| threads as usize * size_of::<f32>()`.
///
/// Returns the `(block_size, occupancy_result)` with the highest occupancy.
/// Ties are broken toward the *larger* block size (i.e. fewer resident blocks),
/// matching the CUDA runtime's preference.
pub fn optimal_block_size<F>(
    registers_per_thread: u32,
    shared_mem_per_block: F,
    limits: &SmResourceLimits,
) -> Result<(u32, OccupancyResult), GpuOptimError>
where
    F: Fn(u32) -> usize,
{
    if limits.warp_size == 0 {
        return Err(GpuOptimError::InvalidState(
            "warp_size must be greater than zero".to_string(),
        ));
    }

    let mut best: Option<(u32, OccupancyResult)> = None;
    let mut block_size = limits.warp_size;
    while block_size <= limits.max_threads_per_block {
        let usage = KernelResourceUsage {
            registers_per_thread,
            shared_mem_per_block_bytes: shared_mem_per_block(block_size),
            threads_per_block: block_size,
        };
        let result = calculate_occupancy(&usage, limits)?;

        // Occupancy is monotonic in `active_warps_per_sm` for a fixed SM (same
        // denominator), so we compare the integer warp count to avoid any
        // floating-point comparison while keeping exact tie detection.
        let replace = match &best {
            None => true,
            Some((best_block_size, best_result)) => {
                result.active_warps_per_sm > best_result.active_warps_per_sm
                    || (result.active_warps_per_sm == best_result.active_warps_per_sm
                        && block_size > *best_block_size)
            }
        };
        if replace {
            best = Some((block_size, result));
        }

        block_size += limits.warp_size;
    }

    best.ok_or_else(|| {
        GpuOptimError::InvalidState(
            "no block size that is a multiple of warp_size fits within max_threads_per_block"
                .to_string(),
        )
    })
}

/// Convenience wrapper computing occupancy for a [`crate::kernels::LaunchConfig`].
///
/// The total threads per block is the product of the configured block
/// dimensions, and the per-block shared memory is taken from the launch config's
/// `shared_memory_size`. `registers_per_thread` must be supplied by the caller
/// (it is a property of the compiled kernel, not of the launch geometry).
pub fn occupancy_for_launch(
    config: &crate::kernels::LaunchConfig,
    registers_per_thread: u32,
    limits: &SmResourceLimits,
) -> Result<OccupancyResult, GpuOptimError> {
    let (block_x, block_y, block_z) = config.block_size;
    let threads_per_block = block_x
        .checked_mul(block_y)
        .and_then(|partial| partial.checked_mul(block_z))
        .ok_or_else(|| {
            GpuOptimError::InvalidState("block_size dimension product overflows u32".to_string())
        })?;

    let usage = KernelResourceUsage {
        registers_per_thread,
        shared_mem_per_block_bytes: config.shared_memory_size,
        threads_per_block,
    };
    calculate_occupancy(&usage, limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-tolerance float comparison (keeps clippy::float_cmp quiet).
    fn approx(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    #[test]
    fn textbook_sm80_256_threads_32_registers() {
        // 256 threads/block, 32 regs/thread, no shared memory on A100 (sm_80).
        // warps/block = 8; warps limit = 64/8 = 8; reg limit = 65536/8192 = 8;
        // block cap = 32 -> active_blocks = 8, active_warps = 64, occupancy = 1.0.
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(32, 0, 256);
        let result = calculate_occupancy(&usage, &limits).expect("valid configuration");

        assert_eq!(result.active_blocks_per_sm, 8);
        assert_eq!(result.active_warps_per_sm, 64);
        assert_eq!(result.max_warps_per_sm, 64);
        assert!(approx(result.occupancy, 1.0));
        // Warps and registers tie at 8; the documented tie-break reports Warps.
        assert_eq!(result.limiter, OccupancyLimiter::Warps);
    }

    #[test]
    fn register_bound_sm80() {
        // 64 regs/thread doubles the register footprint: reg limit = 65536/16384 = 4.
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(64, 0, 256);
        let result = calculate_occupancy(&usage, &limits).expect("valid configuration");

        assert_eq!(result.active_blocks_per_sm, 4);
        assert_eq!(result.active_warps_per_sm, 32);
        assert!(approx(result.occupancy, 0.5));
        assert_eq!(result.limiter, OccupancyLimiter::Registers);
    }

    #[test]
    fn shared_memory_bound_sm80() {
        // 128 threads/block, 16 regs/thread, 48 KiB shared memory per block.
        // smem limit = 167936 / 49152 = 3 (the binding constraint).
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(16, 48 * 1024, 128);
        let result = calculate_occupancy(&usage, &limits).expect("valid configuration");

        assert_eq!(result.active_blocks_per_sm, 3);
        assert_eq!(result.active_warps_per_sm, 12);
        assert!(approx(result.occupancy, 12.0 / 64.0));
        assert_eq!(result.limiter, OccupancyLimiter::SharedMemory);
    }

    #[test]
    fn block_cap_dominates_with_tiny_blocks_sm80() {
        // 32-thread (single-warp) blocks with no register/shared pressure: the
        // warp budget would allow 64 blocks, but the hard cap is 32 blocks.
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(0, 0, 32);
        let result = calculate_occupancy(&usage, &limits).expect("valid configuration");

        assert_eq!(result.active_blocks_per_sm, 32);
        assert_eq!(result.active_warps_per_sm, 32);
        assert!(approx(result.occupancy, 0.5));
        assert_eq!(result.limiter, OccupancyLimiter::BlocksPerSm);
    }

    #[test]
    fn register_allocation_granularity_rounds_up() {
        // 96 threads (3 warps), 33 regs/thread -> raw = 3168 registers, rounded up
        // to the 256-register granularity = 3328, so reg limit = 65536/3328 = 19.
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(33, 0, 96);
        let result = calculate_occupancy(&usage, &limits).expect("valid configuration");

        assert_eq!(result.active_blocks_per_sm, 19);
        assert_eq!(result.active_warps_per_sm, 57);
        assert!(approx(result.occupancy, 57.0 / 64.0));
        assert_eq!(result.limiter, OccupancyLimiter::Registers);
    }

    #[test]
    fn warps_per_block_uses_ceiling() {
        // 100 threads -> ceil(100/32) = 4 warps per block.
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(0, 0, 100);
        let result = calculate_occupancy(&usage, &limits).expect("valid configuration");
        // No register/shared pressure -> block cap (32) binds; 32 * 4 = 128 warps,
        // but warp budget caps to 16 blocks (64/4) which is below the cap.
        assert_eq!(result.active_blocks_per_sm, 16);
        assert_eq!(result.active_warps_per_sm, 64);
        assert_eq!(result.limiter, OccupancyLimiter::Warps);
    }

    #[test]
    fn threads_exceeding_hardware_limit_report_threadsperblock() {
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(32, 0, 2048); // > 1024 max
        let result = calculate_occupancy(&usage, &limits).expect("returns zero-occupancy result");

        assert_eq!(result.active_blocks_per_sm, 0);
        assert_eq!(result.active_warps_per_sm, 0);
        assert!(approx(result.occupancy, 0.0));
        assert_eq!(result.limiter, OccupancyLimiter::ThreadsPerBlock);
    }

    #[test]
    fn zero_threads_is_an_error() {
        let limits = SmResourceLimits::sm_80();
        let usage = KernelResourceUsage::new(32, 0, 0);
        assert!(calculate_occupancy(&usage, &limits).is_err());
    }

    #[test]
    fn optimal_block_size_prefers_full_occupancy_and_largest_block() {
        // 32 regs/thread, no shared memory: block sizes 64..=1024 all reach 100%
        // occupancy, so the tie-break selects the largest (1024).
        let limits = SmResourceLimits::sm_80();
        let (block_size, result) =
            optimal_block_size(32, |_| 0, &limits).expect("a candidate exists");

        assert_eq!(block_size % limits.warp_size, 0);
        assert!(block_size <= limits.max_threads_per_block);
        assert_eq!(block_size, 1024);
        assert!(approx(result.occupancy, 1.0));

        // Occupancy of the chosen block size is at least that of any spot-checked
        // candidate (256 threads also reaches full occupancy here).
        let spot = calculate_occupancy(&KernelResourceUsage::new(32, 0, 256), &limits)
            .expect("valid configuration");
        assert!(result.occupancy >= spot.occupancy);
    }

    #[test]
    fn optimal_block_size_finds_register_heavy_sweet_spot() {
        // 96 regs/thread caps the SM at 21 resident warps (65536 / (96*32) = 21).
        // Several block sizes tie at 21 warps; the documented tie-break selects the
        // largest, which is 672 threads (a single 21-warp block).
        let limits = SmResourceLimits::sm_80();
        let (block_size, result) =
            optimal_block_size(96, |_| 0, &limits).expect("a candidate exists");

        assert_eq!(result.active_warps_per_sm, 21);
        assert!(approx(result.occupancy, 21.0 / 64.0));
        assert_eq!(result.limiter, OccupancyLimiter::Registers);

        // The chosen block size is the maximum across the candidate set, and no
        // larger candidate reaches the same warp count (confirming the tie-break).
        let mut probe = limits.warp_size;
        while probe <= limits.max_threads_per_block {
            let candidate = calculate_occupancy(&KernelResourceUsage::new(96, 0, probe), &limits)
                .expect("valid configuration");
            assert!(result.active_warps_per_sm >= candidate.active_warps_per_sm);
            if candidate.active_warps_per_sm == result.active_warps_per_sm {
                assert!(probe <= block_size);
            }
            probe += limits.warp_size;
        }
        assert_eq!(block_size, 672);
    }

    #[test]
    fn optimal_block_size_supports_block_dependent_shared_memory() {
        // Reduction-style kernel: 4 bytes of shared memory per thread.
        let limits = SmResourceLimits::sm_80();
        let (block_size, result) = optimal_block_size(16, |threads| threads as usize * 4, &limits)
            .expect("a candidate exists");

        assert_eq!(block_size % limits.warp_size, 0);
        assert!(block_size <= limits.max_threads_per_block);
        assert!(result.occupancy > 0.0);
    }

    #[test]
    fn from_compute_capability_maps_known_architectures() {
        assert_eq!(
            SmResourceLimits::from_compute_capability((7, 0)).expect("known"),
            SmResourceLimits::sm_70()
        );
        assert_eq!(
            SmResourceLimits::from_compute_capability((7, 5)).expect("known"),
            SmResourceLimits::sm_75()
        );
        assert_eq!(
            SmResourceLimits::from_compute_capability((8, 0)).expect("known"),
            SmResourceLimits::sm_80()
        );
        assert_eq!(
            SmResourceLimits::from_compute_capability((8, 6)).expect("known"),
            SmResourceLimits::sm_86()
        );
        assert_eq!(
            SmResourceLimits::from_compute_capability((9, 0)).expect("known"),
            SmResourceLimits::sm_90()
        );
        // Nearest-architecture aliases.
        assert_eq!(
            SmResourceLimits::from_compute_capability((7, 2)).expect("nearest"),
            SmResourceLimits::sm_70()
        );
        assert_eq!(
            SmResourceLimits::from_compute_capability((8, 9)).expect("nearest"),
            SmResourceLimits::sm_86()
        );
    }

    #[test]
    fn unknown_compute_capability_is_an_error() {
        assert!(SmResourceLimits::from_compute_capability((5, 0)).is_err());
        assert!(SmResourceLimits::from_compute_capability((10, 0)).is_err());
        assert!(SmResourceLimits::from_compute_capability((0, 0)).is_err());
    }

    #[test]
    fn architecture_constants_are_consistent() {
        for limits in [
            SmResourceLimits::sm_70(),
            SmResourceLimits::sm_75(),
            SmResourceLimits::sm_80(),
            SmResourceLimits::sm_86(),
            SmResourceLimits::sm_90(),
        ] {
            assert_eq!(limits.warp_size, 32);
            assert_eq!(limits.max_threads_per_block, 1024);
            assert_eq!(limits.register_alloc_granularity, 256);
            assert_eq!(limits.registers_per_sm, 65536);
            assert!(limits.max_warps_per_sm > 0);
            assert!(limits.max_blocks_per_sm > 0);
            // max_threads_per_sm must equal the resident-warp budget in threads.
            assert_eq!(
                limits.max_threads_per_sm,
                limits.max_warps_per_sm * limits.warp_size
            );
        }
        // Spot-check the architecture-specific budgets that differ.
        assert_eq!(SmResourceLimits::sm_75().max_warps_per_sm, 32);
        assert_eq!(SmResourceLimits::sm_75().max_blocks_per_sm, 16);
        assert_eq!(SmResourceLimits::sm_86().max_warps_per_sm, 48);
        assert_eq!(SmResourceLimits::sm_90().shared_mem_per_sm_bytes, 233_472);
    }

    #[test]
    fn from_device_capabilities_uses_compute_capability() {
        let caps = crate::backends::DeviceCapabilities {
            name: "A100 (test)".to_string(),
            total_memory: 0,
            available_memory: 0,
            supports_f16: true,
            supports_bf16: true,
            supports_tensor_cores: true,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            multiprocessor_count: 108,
            compute_capability: (8, 0),
        };
        let limits = SmResourceLimits::from_device_capabilities(&caps).expect("cuda device");
        assert_eq!(limits.max_warps_per_sm, 64);
        assert_eq!(limits.max_blocks_per_sm, 32);
        assert_eq!(limits.max_threads_per_block, 1024);
    }

    #[test]
    fn from_device_capabilities_rejects_non_cuda_devices() {
        let caps = crate::backends::DeviceCapabilities {
            name: "CPU (test)".to_string(),
            total_memory: 0,
            available_memory: 0,
            supports_f16: false,
            supports_bf16: false,
            supports_tensor_cores: false,
            max_threads_per_block: 1,
            max_shared_memory_per_block: 0,
            multiprocessor_count: 1,
            compute_capability: (0, 0),
        };
        assert!(SmResourceLimits::from_device_capabilities(&caps).is_err());
    }

    #[test]
    fn occupancy_for_launch_matches_direct_calculation() {
        let limits = SmResourceLimits::sm_80();
        let config = crate::kernels::LaunchConfig {
            grid_size: (128, 1, 1),
            block_size: (256, 1, 1),
            shared_memory_size: 0,
            stream: None,
        };
        let result = occupancy_for_launch(&config, 32, &limits).expect("valid launch");
        assert_eq!(result.active_blocks_per_sm, 8);
        assert!(approx(result.occupancy, 1.0));
    }
}
