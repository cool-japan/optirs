// Persistent OS resource probe (finding R1).
//
// `collect_resource_usage` used to build a throwaway `sysinfo::System` per
// call and then fill CPU / network / disk with the literals 50.0 / 1.0 / 5.0.
// Those three quantities are *rates*: they only exist as a delta between two
// refreshes of the same handle, which is exactly why the previous
// throwaway-handle design could not produce them. This probe therefore keeps
// its `System` and `Networks` handles across ticks and reports `None` (never a
// fabricated number) until it has two refreshes far enough apart.

use super::ResourceUsage;
use std::time::{Duration, Instant};

/// Bytes per megabyte used for every rate conversion here.
const BYTES_PER_MB: f64 = 1024.0 * 1024.0;

/// Long-lived handle onto the operating system's resource counters.
pub struct SystemProbe {
    system: sysinfo::System,
    networks: sysinfo::Networks,
    pid: Option<sysinfo::Pid>,
    last_cpu_refresh: Option<Instant>,
    last_network_refresh: Option<Instant>,
    last_disk_refresh: Option<Instant>,
    cpu_ready: bool,
}

impl std::fmt::Debug for SystemProbe {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SystemProbe")
            .field("cpu_ready", &self.cpu_ready)
            .field("has_pid", &self.pid.is_some())
            .finish()
    }
}

impl Default for SystemProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemProbe {
    /// Create a probe and take the baseline refresh that later deltas are
    /// measured against.
    pub fn new() -> Self {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        system.refresh_cpu_usage();
        let networks = sysinfo::Networks::new_with_refreshed_list();
        let pid = sysinfo::get_current_pid().ok();
        let now = Instant::now();
        let mut probe = Self {
            system,
            networks,
            pid,
            last_cpu_refresh: Some(now),
            last_network_refresh: Some(now),
            last_disk_refresh: None,
            cpu_ready: false,
        };
        probe.refresh_process();
        probe.last_disk_refresh = Some(now);
        probe
    }

    fn refresh_process(&mut self) {
        if let Some(pid) = self.pid {
            self.system.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                false,
                sysinfo::ProcessRefreshKind::nothing()
                    .with_disk_usage()
                    .with_memory(),
            );
        }
    }

    /// Take a resource sample. Rates that do not yet have a usable delta are
    /// reported as `None`.
    pub fn sample(&mut self) -> ResourceUsage {
        let now = Instant::now();
        let mut usage = ResourceUsage {
            timestamp: now,
            ..ResourceUsage::default()
        };

        // ---- memory (always available) ----------------------------------
        //
        // Two genuinely different quantities, and conflating them is what the
        // budget path used to do: `used_memory`/`total_memory` are system-wide
        // and only meaningful as a *ratio* (which is what the alert thresholds
        // compare), while the allocation budget is a per-process figure. Both
        // are captured separately so nothing has to divide one by the other.
        self.system.refresh_memory();
        usage.memory_usage_mb = (self.system.used_memory() / 1024 / 1024) as usize;
        usage.total_memory_mb = (self.system.total_memory() / 1024 / 1024) as usize;

        // ---- CPU (delta between refreshes) ------------------------------
        let cpu_elapsed = self
            .last_cpu_refresh
            .map(|last| now.saturating_duration_since(last))
            .unwrap_or(Duration::ZERO);
        if cpu_elapsed >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            self.system.refresh_cpu_usage();
            self.last_cpu_refresh = Some(now);
            self.cpu_ready = true;
        }
        if self.cpu_ready {
            let percent = self.system.global_cpu_usage() as f64;
            if percent.is_finite() {
                usage.cpu_usage_percent = percent.clamp(0.0, 100.0);
                usage.cpu_usage_percent_valid = true;
            }
        }

        // ---- network (byte delta over elapsed time) ---------------------
        if let Some(last) = self.last_network_refresh {
            let elapsed = now.saturating_duration_since(last).as_secs_f64();
            self.networks.refresh(true);
            self.last_network_refresh = Some(now);
            if elapsed > 0.0 {
                let bytes: u64 = self
                    .networks
                    .list()
                    .values()
                    .map(|interface| interface.received().saturating_add(interface.transmitted()))
                    .sum();
                usage.network_io_mbps = Some(bytes as f64 / BYTES_PER_MB / elapsed);
            }
        }

        // ---- this process: own memory, plus disk I/O byte deltas --------
        if let Some(pid) = self.pid {
            let elapsed = self
                .last_disk_refresh
                .map(|last| now.saturating_duration_since(last).as_secs_f64())
                .unwrap_or(0.0);
            self.refresh_process();
            self.last_disk_refresh = Some(now);
            if let Some(process) = self.system.process(pid) {
                usage.process_memory_mb = Some((process.memory() / 1024 / 1024) as usize);
                if elapsed > 0.0 {
                    let disk = process.disk_usage();
                    let bytes = disk.read_bytes.saturating_add(disk.written_bytes);
                    usage.disk_io_mbps = Some(bytes as f64 / BYTES_PER_MB / elapsed);
                }
            }
        }

        // ---- thread parallelism -----------------------------------------
        usage.active_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        usage
    }
}
