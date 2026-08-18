//! Injectable system telemetry (F12).
//!
//! `SystemResourceTracker` used to fabricate its readings: 32 GB total memory,
//! 16 GB used, 60% CPU, 4 GPUs at 80%, 1 TB disk half full, 1 GB/s network, 65 C,
//! 250 W — all hard-coded literals behind
//! `// In a real implementation, this would query system APIs`. Those numbers were
//! then compared against the user's
//! [`crate::nas_engine::config::ResourceConstraints`], so a search
//! configured with (say) an 8 GB memory budget aborted immediately on the strength
//! of an invented 16 GB reading.
//!
//! The rule here is: **report only what can actually be measured, and `None`
//! otherwise.** Consumers must treat `None` as "unconstrained", never as zero and
//! never as a violation.
//!
//! What [`StdTelemetry`] can honestly measure, in pure Rust with no FFI:
//!
//! | field | source | portable? |
//! |---|---|---|
//! | `logical_cpus` | [`std::thread::available_parallelism`] | yes |
//! | `process_memory_gb` | `/proc/self/status` `VmRSS` | Linux only |
//! | `total_memory_gb`, `available_memory_gb` | `/proc/meminfo` | Linux only |
//! | `load_average_per_cpu` | `/proc/loadavg` divided by the CPU count | Linux only |
//! | `process_count` | entries in `/proc` | Linux only |
//!
//! Everything else — GPU count and utilization, disk capacity, network bandwidth,
//! die temperature, package power — needs a platform API or a vendor library. None
//! of those can be reached without FFI, so [`StdTelemetry`] reports `None` for
//! them rather than inventing a plausible-looking constant. Supply a
//! [`TelemetrySource`] of your own (e.g. backed by a monitoring agent you already
//! run) to fill them in.
//!
//! The `/proc` parsers are pure functions over `&str`
//! ([`parse_vm_rss_kib`], [`parse_meminfo`], [`parse_loadavg`]) so the Linux path
//! is unit-tested from fixture strings on every platform, including the macOS and
//! Windows hosts where the files do not exist.

use std::fmt::Debug;
use std::path::Path;

/// One telemetry reading. Every field is `Option`: `None` means *not measurable
/// here*, which consumers must treat as unconstrained.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TelemetrySample {
    /// Resident set size of *this process*, in GiB.
    ///
    /// This — not system-wide usage — is what belongs in
    /// [`crate::nas_engine::results::ResourceUsage::memory_gb`], because that field
    /// is compared against a budget for the search itself. Charging the search for
    /// every other process on the machine would abort it whenever the host is busy.
    pub process_memory_gb: Option<f64>,

    /// Total physical memory, in GiB (system capacity, not this search's usage).
    pub total_memory_gb: Option<f64>,

    /// Memory the OS reports as available for new allocations, in GiB.
    pub available_memory_gb: Option<f64>,

    /// Logical CPUs usable by this process.
    pub logical_cpus: Option<usize>,

    /// 1-minute load average divided by the logical CPU count, so `1.0` means
    /// "fully loaded" independent of core count.
    pub load_average_per_cpu: Option<f64>,

    /// Number of processes visible on the system.
    pub process_count: Option<usize>,

    /// GPU devices visible to this process.
    pub gpu_devices: Option<usize>,

    /// GPU utilization in `[0, 1]`.
    pub gpu_utilization: Option<f64>,

    /// Total capacity of the filesystem holding the working directory, in GiB.
    pub total_disk_gb: Option<f64>,

    /// Free space on that filesystem, in GiB.
    pub available_disk_gb: Option<f64>,

    /// Network bandwidth, in MB/s.
    pub network_bandwidth_mbps: Option<f64>,

    /// Package/die temperature, in degrees Celsius.
    pub temperature_celsius: Option<f64>,

    /// Instantaneous power draw, in watts.
    pub power_watts: Option<f64>,
}

impl TelemetrySample {
    /// A sample in which nothing is known. Equivalent to
    /// `TelemetrySample::default()`, spelled out so the intent is obvious at call
    /// sites.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Whether every field is `None`.
    pub fn is_fully_unknown(&self) -> bool {
        *self == Self::default()
    }
}

/// A source of system telemetry.
///
/// Injecting one is how a caller upgrades the honest-but-sparse default: wire in a
/// source backed by whatever monitoring you already have, and the resource monitor
/// starts enforcing the corresponding constraints. Constraints whose measurement is
/// still `None` stay unenforced.
pub trait TelemetrySource: Send + Sync + Debug {
    /// Human-readable name of the source, used in log messages.
    fn name(&self) -> &str;

    /// Take a reading.
    fn sample(&self) -> TelemetrySample;
}

/// The default source: standard library plus the Linux `/proc` files, and `None`
/// for everything that cannot be measured without FFI.
#[derive(Debug, Clone, Default)]
pub struct StdTelemetry;

impl StdTelemetry {
    /// Create the default telemetry source.
    pub fn new() -> Self {
        Self
    }
}

impl TelemetrySource for StdTelemetry {
    fn name(&self) -> &str {
        "std"
    }

    fn sample(&self) -> TelemetrySample {
        let logical_cpus = std::thread::available_parallelism()
            .ok()
            .map(|count| count.get());

        // `/proc` is absent on macOS and Windows; every read below therefore
        // degrades to `None` rather than to a guess.
        let process_memory_gb = read_to_string_if_present("/proc/self/status")
            .and_then(|status| parse_vm_rss_kib(&status))
            .map(|kib| kib as f64 / (1024.0 * 1024.0));

        let (total_memory_gb, available_memory_gb) = match read_to_string_if_present(
            "/proc/meminfo",
        )
        .and_then(|info| parse_meminfo(&info))
        {
            Some((total_kib, available_kib)) => (
                Some(total_kib as f64 / (1024.0 * 1024.0)),
                available_kib.map(|kib| kib as f64 / (1024.0 * 1024.0)),
            ),
            None => (None, None),
        };

        let load_average_per_cpu = read_to_string_if_present("/proc/loadavg")
            .and_then(|loadavg| parse_loadavg(&loadavg))
            .and_then(|load| logical_cpus.map(|cpus| load / cpus.max(1) as f64));

        let process_count = count_proc_processes();

        TelemetrySample {
            process_memory_gb,
            total_memory_gb,
            available_memory_gb,
            logical_cpus,
            load_average_per_cpu,
            process_count,
            // Not measurable without a platform API / vendor library.
            gpu_devices: None,
            gpu_utilization: None,
            total_disk_gb: None,
            available_disk_gb: None,
            network_bandwidth_mbps: None,
            temperature_celsius: None,
            power_watts: None,
        }
    }
}

/// Read a file, returning `None` if it does not exist or cannot be read. Used so
/// the absence of `/proc` is an ordinary "unknown", not an error.
fn read_to_string_if_present(path: &str) -> Option<String> {
    let path = Path::new(path);
    if !path.exists() {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

/// Extract `VmRSS` (in KiB) from the contents of `/proc/self/status`.
///
/// The file states its own units (`VmRSS:    123456 kB`), so no page-size lookup —
/// and therefore no `sysconf` FFI — is needed.
pub fn parse_vm_rss_kib(status: &str) -> Option<u64> {
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let value: u64 = fields.next()?.parse().ok()?;
        // Only kB is ever emitted by the kernel; anything else is unrecognised and
        // is reported as unknown rather than misinterpreted.
        return match fields.next() {
            Some(unit) if unit.eq_ignore_ascii_case("kb") => Some(value),
            None => Some(value),
            Some(_) => None,
        };
    }
    None
}

/// Extract `(MemTotal_kib, MemAvailable_kib)` from the contents of
/// `/proc/meminfo`. `MemAvailable` is absent on kernels older than 3.14, hence the
/// inner `Option`.
pub fn parse_meminfo(meminfo: &str) -> Option<(u64, Option<u64>)> {
    let mut total = None;
    let mut available = None;
    for line in meminfo.lines() {
        let mut parts = line.split(':');
        let key = parts.next()?.trim();
        let Some(rest) = parts.next() else {
            continue;
        };
        let Some(value) = rest.split_whitespace().next() else {
            continue;
        };
        let Ok(parsed) = value.parse::<u64>() else {
            continue;
        };
        match key {
            "MemTotal" => total = Some(parsed),
            "MemAvailable" => available = Some(parsed),
            _ => {}
        }
    }
    total.map(|total| (total, available))
}

/// Extract the 1-minute load average from the contents of `/proc/loadavg`.
pub fn parse_loadavg(loadavg: &str) -> Option<f64> {
    loadavg
        .split_whitespace()
        .next()
        .and_then(|field| field.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

/// Count numeric entries in `/proc`, which is the process count on Linux.
fn count_proc_processes() -> Option<usize> {
    let entries = std::fs::read_dir("/proc").ok()?;
    let count = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit()))
                .unwrap_or(false)
        })
        .count();
    Some(count)
}

/// A telemetry source returning a fixed sample. Intended for tests and for callers
/// that already have measurements from elsewhere.
#[derive(Debug, Clone)]
pub struct FixedTelemetry {
    name: String,
    sample: TelemetrySample,
}

impl FixedTelemetry {
    /// Wrap a pre-computed sample.
    pub fn new(name: impl Into<String>, sample: TelemetrySample) -> Self {
        Self {
            name: name.into(),
            sample,
        }
    }
}

impl TelemetrySource for FixedTelemetry {
    fn name(&self) -> &str {
        &self.name
    }

    fn sample(&self) -> TelemetrySample {
        self.sample.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATUS_FIXTURE: &str = "Name:\tcargo\nUmask:\t0022\nState:\tR (running)\n\
                                  VmPeak:\t 2097152 kB\nVmSize:\t 1048576 kB\n\
                                  VmRSS:\t  524288 kB\nThreads:\t8\n";

    const MEMINFO_FIXTURE: &str = "MemTotal:       32899072 kB\nMemFree:         1234567 kB\n\
                                   MemAvailable:   20971520 kB\nBuffers:          123456 kB\n";

    #[test]
    fn vm_rss_is_parsed_from_the_status_fixture() {
        assert_eq!(parse_vm_rss_kib(STATUS_FIXTURE), Some(524_288));
        // 512 MiB expressed in GiB.
        let gib: f64 = 524_288.0 / (1024.0 * 1024.0);
        assert!((gib - 0.5).abs() < 1e-12);
    }

    #[test]
    fn vm_rss_reports_unknown_rather_than_guessing() {
        assert_eq!(parse_vm_rss_kib("Name:\tcargo\nThreads:\t8\n"), None);
        assert_eq!(parse_vm_rss_kib(""), None);
        // An unrecognised unit must not be silently treated as kB.
        assert_eq!(parse_vm_rss_kib("VmRSS:\t 512 pages\n"), None);
        // Missing unit (defensive): the kernel always writes kB.
        assert_eq!(parse_vm_rss_kib("VmRSS:\t 4096\n"), Some(4096));
        assert_eq!(parse_vm_rss_kib("VmRSS:\tnot-a-number kB\n"), None);
    }

    #[test]
    fn meminfo_is_parsed_from_the_fixture() {
        let (total, available) = parse_meminfo(MEMINFO_FIXTURE).expect("MemTotal present");
        assert_eq!(total, 32_899_072);
        assert_eq!(available, Some(20_971_520));
    }

    #[test]
    fn meminfo_tolerates_a_kernel_without_memavailable() {
        let (total, available) =
            parse_meminfo("MemTotal:  1048576 kB\nMemFree:  524288 kB\n").expect("MemTotal");
        assert_eq!(total, 1_048_576);
        assert_eq!(available, None);
        assert_eq!(parse_meminfo("MemFree: 1 kB\n"), None);
        assert_eq!(parse_meminfo(""), None);
    }

    #[test]
    fn loadavg_is_parsed_and_validated() {
        assert_eq!(parse_loadavg("0.52 0.58 0.59 2/1234 56789\n"), Some(0.52));
        assert_eq!(parse_loadavg(""), None);
        assert_eq!(parse_loadavg("garbage 0.1 0.2\n"), None);
        assert_eq!(parse_loadavg("-1.0 0.1 0.2\n"), None);
    }

    /// The whole point of F12: the default source must never invent a number.
    #[test]
    fn the_default_source_reports_only_measurable_values() {
        let sample = StdTelemetry::new().sample();

        // Available parallelism is portable, so this one must always be present.
        assert!(
            sample.logical_cpus.is_some_and(|cpus| cpus >= 1),
            "available_parallelism must be reported"
        );

        // These are the fabricated readings the tracker used to emit. None of them
        // is reachable without FFI, so all must be `None` on every platform.
        assert_eq!(sample.gpu_devices, None, "the invented 4 GPUs must be gone");
        assert_eq!(
            sample.gpu_utilization, None,
            "the invented 80% must be gone"
        );
        assert_eq!(sample.total_disk_gb, None, "the invented 1 TB must be gone");
        assert_eq!(sample.available_disk_gb, None);
        assert_eq!(
            sample.network_bandwidth_mbps, None,
            "the invented 1 GB/s must be gone"
        );
        assert_eq!(
            sample.temperature_celsius, None,
            "the invented 65 C must be gone"
        );
        assert_eq!(sample.power_watts, None, "the invented 250 W must be gone");

        // Memory is only knowable where /proc exists. Whatever the platform, the
        // value must be either absent or genuinely measured — never the old 32/16.
        if let Some(total) = sample.total_memory_gb {
            assert!(total > 0.0 && total.is_finite(), "got {total}");
            assert_ne!(total, 32.0, "32.0 GB was the fabricated constant");
        }
        if let Some(process) = sample.process_memory_gb {
            assert!(process > 0.0 && process.is_finite(), "got {process}");
            assert_ne!(process, 16.0, "16.0 GB was the fabricated constant");
        }
        if let Some(load) = sample.load_average_per_cpu {
            assert!(load >= 0.0 && load.is_finite());
        }
    }

    #[test]
    fn a_fixed_source_reports_exactly_what_it_was_given() {
        let sample = TelemetrySample {
            process_memory_gb: Some(3.5),
            total_memory_gb: Some(64.0),
            logical_cpus: Some(16),
            temperature_celsius: Some(71.0),
            ..TelemetrySample::unknown()
        };
        let source = FixedTelemetry::new("injected", sample.clone());
        assert_eq!(source.name(), "injected");
        assert_eq!(source.sample(), sample);
        assert!(!sample.is_fully_unknown());
        assert!(TelemetrySample::unknown().is_fully_unknown());
    }
}
