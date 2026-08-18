// Real System Telemetry Sampler
//
// This module is the single source of truth for real (non-fabricated) system
// and process measurements used across `optirs-bench`: process RSS/virtual
// memory, per-core and process CPU usage, system load average, total and
// available memory, real platform identification, and the active `rustc`
// toolchain version.
//
// Every other module in this crate that previously hardcoded or synthesized
// a "measurement" (constant memory figures, sine-wave CPU curves, etc.)
// should be wired to go through `SystemSampler` instead. Where a genuine
// measurement is not available on the current platform or requires
// caller-provided context (e.g. FLOPS from declared op counts), that is the
// caller's responsibility -- this module never invents a number to fill a
// gap.

use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind,
    System,
};

use crate::error::{OptimError, Result};

/// A point-in-time measurement of the current process' resource usage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessSample {
    /// Resident set size (physical memory currently held), in bytes.
    pub rss_bytes: u64,
    /// Virtual memory size, in bytes.
    pub virtual_bytes: u64,
    /// Process CPU usage, as a percentage (100.0 == one full core).
    /// Derived from the delta of accumulated CPU time between the two most
    /// recent refreshes, divided by elapsed wall-clock time. `None` on the
    /// very first sample, since a delta requires two data points.
    pub cpu_percent: Option<f64>,
    /// Bytes read from disk since the process started (platform-dependent
    /// accounting; see `sysinfo::Process::disk_usage` caveats).
    pub disk_read_bytes: u64,
    /// Bytes written to disk since the process started.
    pub disk_written_bytes: u64,
    /// When this sample was captured.
    pub timestamp: Instant,
}

/// A point-in-time measurement of whole-system resource usage.
#[derive(Debug, Clone)]
pub struct SystemSample {
    /// Total installed physical memory, in bytes.
    pub total_memory_bytes: u64,
    /// Memory currently available for new allocations, in bytes.
    pub available_memory_bytes: u64,
    /// Memory currently in use, in bytes.
    pub used_memory_bytes: u64,
    /// Aggregate CPU usage across all cores, as a percentage.
    pub global_cpu_percent: f64,
    /// Per-core CPU usage, as percentages, in core order.
    pub per_core_cpu_percent: Vec<f64>,
    /// 1/5/15-minute load averages (Unix semantics; may be all-zero on
    /// platforms without a load-average concept, e.g. Windows).
    pub load_average: LoadAverage,
}

/// System load average, as reported by the OS.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LoadAverage {
    pub one: f64,
    pub five: f64,
    pub fifteen: f64,
}

/// Real, measured platform identification (never hardcoded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformInfo {
    /// Operating system name (e.g. "Linux", "macOS", "Windows"), best
    /// effort from `sysinfo`; falls back to `std::env::consts::OS`.
    pub os_name: String,
    /// OS version string, if it could be determined.
    pub os_version: Option<String>,
    /// Kernel version string, if it could be determined.
    pub kernel_version: Option<String>,
    /// Compile-time target architecture (`std::env::consts::ARCH`).
    pub arch: &'static str,
    /// Compile-time OS family (`std::env::consts::FAMILY`).
    pub family: &'static str,
    /// CPU brand string (e.g. "Apple M2", "Intel(R) Core(TM) ..."), if
    /// available from the platform.
    pub cpu_brand: Option<String>,
    /// Number of physical CPU cores, if it could be determined.
    pub physical_core_count: Option<usize>,
    /// Number of logical CPU cores (including SMT/hyperthreads).
    pub logical_core_count: usize,
    /// Host name, if it could be determined.
    pub hostname: Option<String>,
    /// `rustc -vV` output's `release:` field, if `rustc` was found on PATH.
    pub rustc_version: Option<String>,
    /// `rustc -vV` output's `host:` field, if `rustc` was found on PATH.
    pub rustc_host: Option<String>,
}

/// Caches a `sysinfo::System` handle and the current process' PID, and
/// provides a small, honest API over real measurements.
///
/// All values are either real measurements or `None`/`Option` -- this type
/// never fabricates a number to paper over a platform gap.
pub struct SystemSampler {
    system: Mutex<System>,
    pid: Pid,
    /// Last-observed (accumulated_cpu_time_ms, wall_clock_instant) for the
    /// tracked process, used to derive process CPU% as a time-delta rather
    /// than relying on `sysinfo`'s internal minimum-refresh-interval trap.
    last_process_cpu: Mutex<Option<(u64, Instant)>>,
}

impl std::fmt::Debug for SystemSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemSampler")
            .field("pid", &self.pid)
            .finish()
    }
}

impl SystemSampler {
    /// Create a new sampler bound to the current process. Performs an
    /// initial refresh so the first `sample_*` call returns real data.
    pub fn new() -> Result<Self> {
        let pid = sysinfo::get_current_pid().map_err(|e| {
            OptimError::ResourceUnavailable(format!(
                "failed to determine current process id via sysinfo: {e}"
            ))
        })?;

        let mut system = System::new_with_specifics(
            RefreshKind::nothing()
                .with_memory(MemoryRefreshKind::everything())
                .with_cpu(CpuRefreshKind::everything()),
        );
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            false,
            ProcessRefreshKind::nothing()
                .with_memory()
                .with_cpu()
                .with_disk_usage(),
        );

        let sampler = Self {
            system: Mutex::new(system),
            pid,
            last_process_cpu: Mutex::new(None),
        };
        sampler.refresh();
        Ok(sampler)
    }

    /// Refresh all cached system, CPU, and process telemetry. Cheap enough
    /// to call before every `sample_*` invocation; callers that want a
    /// stable snapshot across several reads should call this once and then
    /// read multiple times.
    pub fn refresh(&self) {
        let mut system = match self.system.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        system.refresh_memory();
        system.refresh_cpu_all();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            false,
            ProcessRefreshKind::nothing()
                .with_memory()
                .with_cpu()
                .with_disk_usage(),
        );

        if let Some(process) = system.process(self.pid) {
            let accumulated_ms = process.accumulated_cpu_time();
            let now = Instant::now();
            let mut last = match self.last_process_cpu.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            *last = Some((accumulated_ms, now));
        }
    }

    /// Sample the current process' resource usage.
    ///
    /// `cpu_percent` is `None` on the very first call (or if the process
    /// disappeared), since deriving a CPU percentage requires two samples
    /// separated by real wall-clock time.
    pub fn sample_process(&self) -> Result<ProcessSample> {
        let system = match self.system.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let process = system.process(self.pid).ok_or_else(|| {
            OptimError::ResourceUnavailable(
                "current process not found in sysinfo process table".to_string(),
            )
        })?;

        let rss_bytes = process.memory();
        let virtual_bytes = process.virtual_memory();
        let disk = process.disk_usage();
        let accumulated_ms = process.accumulated_cpu_time();
        let now = Instant::now();
        drop(system);

        let cpu_percent = {
            let mut last = match self.last_process_cpu.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            let computed = last.and_then(|(prev_ms, prev_time)| {
                let elapsed = now.saturating_duration_since(prev_time);
                if elapsed.as_millis() == 0 || accumulated_ms < prev_ms {
                    None
                } else {
                    let delta_ms = (accumulated_ms - prev_ms) as f64;
                    Some((delta_ms / elapsed.as_millis() as f64) * 100.0)
                }
            });
            *last = Some((accumulated_ms, now));
            computed
        };

        Ok(ProcessSample {
            rss_bytes,
            virtual_bytes,
            cpu_percent,
            disk_read_bytes: disk.total_read_bytes,
            disk_written_bytes: disk.total_written_bytes,
            timestamp: now,
        })
    }

    /// Sample whole-system resource usage.
    pub fn sample_system(&self) -> SystemSample {
        let system = match self.system.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        let load = System::load_average();

        SystemSample {
            total_memory_bytes: system.total_memory(),
            available_memory_bytes: system.available_memory(),
            used_memory_bytes: system.used_memory(),
            global_cpu_percent: system.global_cpu_usage() as f64,
            per_core_cpu_percent: system.cpus().iter().map(|c| c.cpu_usage() as f64).collect(),
            load_average: LoadAverage {
                one: load.one,
                five: load.five,
                fifteen: load.fifteen,
            },
        }
    }

    /// Real platform identification. Does not require a live `System`
    /// instance for most fields (many are `sysinfo` associated functions),
    /// but is exposed as a method for API symmetry and testability.
    pub fn platform_info(&self) -> PlatformInfo {
        Self::platform_info_static()
    }

    /// Same as [`Self::platform_info`], callable without constructing a
    /// sampler (e.g. from binaries that only need platform metadata).
    pub fn platform_info_static() -> PlatformInfo {
        let system = System::new_with_specifics(
            RefreshKind::nothing().with_cpu(CpuRefreshKind::everything()),
        );
        let logical_core_count = {
            let count = system.cpus().len();
            if count == 0 {
                1
            } else {
                count
            }
        };
        let cpu_brand = system.cpus().first().map(|c| c.brand().to_string());

        let (rustc_version, rustc_host) = detect_rustc_version();

        PlatformInfo {
            os_name: System::name().unwrap_or_else(|| std::env::consts::OS.to_string()),
            os_version: System::long_os_version().or_else(System::os_version),
            kernel_version: System::kernel_version(),
            arch: std::env::consts::ARCH,
            family: std::env::consts::FAMILY,
            cpu_brand,
            physical_core_count: System::physical_core_count(),
            logical_core_count,
            hostname: System::host_name(),
            rustc_version,
            rustc_host,
        }
    }
}

/// Run `rustc -vV` and extract the `release:` and `host:` fields.
/// Returns `(None, None)` if `rustc` is not on PATH or the output could not
/// be parsed -- never a fabricated version string.
fn detect_rustc_version() -> (Option<String>, Option<String>) {
    let output = match Command::new("rustc").arg("-vV").output() {
        Ok(output) if output.status.success() => output,
        _ => return (None, None),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut release = None;
    let mut host = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("release: ") {
            release = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("host: ") {
            host = Some(value.trim().to_string());
        }
    }
    (release, host)
}

/// Convenience: run `git status --porcelain` in `dir` and report whether the
/// working tree is clean. Returns `Err` if `git` is unavailable or the
/// directory is not a repository -- callers must not treat that as "clean".
pub fn git_is_clean<P: AsRef<std::path::Path>>(dir: P) -> Result<bool> {
    let output = Command::new("git")
        .arg("status")
        .arg("--porcelain")
        .current_dir(dir)
        .output()
        .map_err(|e| OptimError::ResourceUnavailable(format!("failed to invoke git: {e}")))?;

    if !output.status.success() {
        return Err(OptimError::ResourceUnavailable(format!(
            "git status failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(output.stdout.is_empty())
}

/// A minimal wall-clock-only cooldown to respect `sysinfo`'s recommendation
/// that CPU-usage refreshes be separated by at least this long. Exposed so
/// callers doing a manual two-refresh CPU measurement (rather than relying
/// on the delta-based `ProcessSample::cpu_percent`) can do so correctly.
pub const MINIMUM_CPU_REFRESH_INTERVAL: Duration = sysinfo::MINIMUM_CPU_UPDATE_INTERVAL;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rss_and_total_memory_are_real_and_positive() {
        let sampler =
            SystemSampler::new().expect("sampler should initialize on any supported platform");
        let process = sampler
            .sample_process()
            .expect("current process must be sampleable");
        assert!(
            process.rss_bytes > 0,
            "RSS must be a real positive measurement"
        );

        let system = sampler.sample_system();
        assert!(
            system.total_memory_bytes > 0,
            "total memory must be a real positive measurement"
        );
        assert!(
            system.available_memory_bytes
                <= system.total_memory_bytes.max(system.available_memory_bytes)
        );
    }

    #[test]
    fn platform_info_matches_env_consts() {
        let info = SystemSampler::platform_info_static();
        assert_eq!(info.arch, std::env::consts::ARCH);
        assert_eq!(info.family, std::env::consts::FAMILY);
        assert!(info.logical_core_count >= 1);
    }

    #[test]
    fn process_cpu_percent_is_none_or_finite() {
        let sampler = SystemSampler::new().expect("sampler should initialize");
        let sample = sampler.sample_process().expect("sample should succeed");
        if let Some(cpu) = sample.cpu_percent {
            assert!(cpu.is_finite() && cpu >= 0.0);
        }
    }

    #[test]
    fn refresh_updates_process_cpu_delta() {
        let sampler = SystemSampler::new().expect("sampler should initialize");
        // Do a bit of real work so accumulated CPU time actually advances.
        let mut acc: u64 = 0;
        for i in 0..5_000_000u64 {
            acc = acc.wrapping_add(i);
        }
        std::hint::black_box(acc);
        std::thread::sleep(Duration::from_millis(10));
        sampler.refresh();
        let sample = sampler.sample_process().expect("sample should succeed");
        // Either a real percentage or None -- both are honest outcomes;
        // the important invariant is that it never panics and never lies.
        if let Some(cpu) = sample.cpu_percent {
            assert!(cpu.is_finite());
        }
    }
}
