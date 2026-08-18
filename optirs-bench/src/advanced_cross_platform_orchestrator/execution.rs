// Real process execution and output-metric parsing for the cross-platform
// orchestrator.
//
// Split out of `orchestrator.rs` (which was approaching the workspace's
// 2000-line-per-file limit) per the prior wave's own suggestion: this module
// holds `CommandOutcome`, `run_process`, `parse_metrics_from_output`, and the
// smaller helpers they depend on. `CrossPlatformOrchestrator` itself, and the
// higher-level target-resolution/scenario-execution logic that calls into
// this module, remain in `orchestrator.rs`.

use crate::error::{OptimError, Result};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command as AsyncCommand;

use super::types::*;

/// Outcome of one real process invocation performed by the orchestrator.
///
/// Every field is measured from the spawned child: nothing here is simulated.
#[derive(Debug, Clone)]
pub(super) struct CommandOutcome {
    /// Full command line that was executed (program plus arguments).
    pub(super) command_line: String,
    /// Process exit code, when the platform reported one.
    pub(super) exit_code: Option<i32>,
    /// Whether the process exited successfully (status code 0).
    pub(super) success: bool,
    /// Captured standard output.
    pub(super) stdout: String,
    /// Captured standard error.
    pub(super) stderr: String,
    /// Wall-clock duration of the invocation.
    pub(super) duration: Duration,
    /// Whether the invocation was aborted because it exceeded its timeout.
    pub(super) timed_out: bool,
}

/// Where a matrix entry's scenario commands are executed.
#[derive(Debug)]
pub(super) enum ExecutionTarget<'a> {
    /// Directly on the host running the orchestrator.
    Local,
    /// Inside an already provisioned container, through the container runtime.
    Container(&'a ContainerInfo),
    /// On a provisioned cloud instance, over its configured SSH access.
    Cloud(&'a CloudInstance),
}

/// Resolved SSH access details for a provisioned cloud instance.
#[derive(Debug, Clone)]
pub(super) struct SshAccess {
    pub(super) host: String,
    pub(super) user: String,
    pub(super) port: Option<u16>,
    pub(super) identity_file: Option<String>,
}

/// A target that has passed its reachability pre-flight and can actually run
/// commands. Constructing one is the only way to reach [`run_process`], so a
/// scenario can never be "executed" against an unreachable target.
#[derive(Debug)]
pub(super) enum ResolvedTarget {
    Local,
    Container {
        runtime: String,
        container_id: String,
        windows: bool,
    },
    Cloud(SshAccess),
}

/// Identify the platform this process is currently running on.
///
/// Returns `None` for OS/architecture combinations that have no
/// [`PlatformTarget`] mapping. `None` is deliberate: silently falling back to a
/// default platform would reintroduce exactly the "guessed metadata" class of
/// bug that the declared-platform metadata was introduced to remove.
pub(super) fn host_platform() -> Option<PlatformTarget> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some(PlatformTarget::LinuxX86_64),
        ("linux", "aarch64") => Some(PlatformTarget::LinuxAarch64),
        ("linux", "mips64") => Some(PlatformTarget::LinuxMips64),
        ("linux", "powerpc64") => Some(PlatformTarget::LinuxPowerPC64),
        ("linux", "s390x") => Some(PlatformTarget::LinuxS390X),
        ("windows", "x86_64") => Some(PlatformTarget::WindowsX86_64),
        ("macos", "x86_64") => Some(PlatformTarget::MacOSX86_64),
        ("macos", "aarch64") => Some(PlatformTarget::MacOSAarch64),
        ("freebsd", "x86_64") => Some(PlatformTarget::FreeBSDX86_64),
        ("openbsd", "x86_64") => Some(PlatformTarget::OpenBSDX86_64),
        ("netbsd", "x86_64") => Some(PlatformTarget::NetBSDX86_64),
        ("solaris", "x86_64") | ("illumos", "x86_64") => Some(PlatformTarget::SolarisX86_64),
        _ => None,
    }
}

/// Whether a platform target uses the Windows command interpreter.
pub(super) fn is_windows_platform(platform: &PlatformTarget) -> bool {
    matches!(platform, PlatformTarget::WindowsX86_64)
}

/// Shell wrapper used to execute a scenario command string.
pub(super) fn shell_invocation(windows: bool, command: &str) -> (String, Vec<String>) {
    if windows {
        (
            "cmd".to_string(),
            vec!["/C".to_string(), command.to_string()],
        )
    } else {
        (
            "sh".to_string(),
            vec!["-c".to_string(), command.to_string()],
        )
    }
}

/// Render an optimization level as a stable lowercase identifier for the
/// environment handed to executed commands.
pub(super) fn optimization_level_to_string(level: &OptimizationLevel) -> String {
    match level {
        OptimizationLevel::Debug => "debug".to_string(),
        OptimizationLevel::Release => "release".to_string(),
        OptimizationLevel::ReleaseLTO => "release-lto".to_string(),
        OptimizationLevel::MinSize => "min-size".to_string(),
        OptimizationLevel::Custom(name) => name.clone(),
    }
}

/// Environment variables handed to every command the orchestrator runs, so the
/// executed scenario can adapt to the matrix entry it belongs to. All values are
/// taken verbatim from the execution context; none are invented.
pub(super) fn scenario_environment(context: &TestExecutionContext) -> Vec<(String, String)> {
    vec![
        (
            "OPTIRS_EXECUTION_ID".to_string(),
            context.execution_id.clone(),
        ),
        (
            "OPTIRS_TARGET_PLATFORM".to_string(),
            context.platform.to_string(),
        ),
        (
            "OPTIRS_RUST_VERSION".to_string(),
            context.rust_version.clone(),
        ),
        (
            "OPTIRS_BUILD_PROFILE".to_string(),
            context.build_profile.clone(),
        ),
        ("OPTIRS_FEATURES".to_string(), context.features.join(",")),
        (
            "OPTIRS_OPTIMIZATION".to_string(),
            optimization_level_to_string(&context.optimization),
        ),
    ]
}

/// Spawn `program` with `args` and wait for it, capturing the real exit status,
/// stdout and stderr.
///
/// Failing to spawn the program (for example because a container runtime or an
/// `ssh` client is not installed) is reported as
/// [`OptimError::ResourceUnavailable`] — the command genuinely could not be run,
/// which is a different thing from a test that ran and failed. Exceeding
/// `timeout` kills the child (via `kill_on_drop`) and is reported through
/// [`CommandOutcome::timed_out`].
pub(super) async fn run_process(
    program: &str,
    args: &[String],
    envs: &[(String, String)],
    timeout: Duration,
) -> Result<CommandOutcome> {
    let command_line = if args.is_empty() {
        program.to_string()
    } else {
        format!("{} {}", program, args.join(" "))
    };

    let mut command = AsyncCommand::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (key, value) in envs {
        command.env(key, value);
    }

    let started = Instant::now();
    let child = command.spawn().map_err(|e| {
        OptimError::ResourceUnavailable(format!(
            "cannot execute '{}': failed to spawn '{}' ({})",
            command_line, program, e
        ))
    })?;

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(CommandOutcome {
            command_line,
            exit_code: output.status.code(),
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            duration: started.elapsed(),
            timed_out: false,
        }),
        Ok(Err(e)) => Err(OptimError::ExecutionError(format!(
            "failed while waiting for '{}': {}",
            command_line, e
        ))),
        Err(_) => Ok(CommandOutcome {
            command_line,
            exit_code: None,
            success: false,
            stdout: String::new(),
            stderr: String::new(),
            duration: started.elapsed(),
            timed_out: true,
        }),
    }
}

/// Parse the leading numeric value (and its unit token) out of the text that
/// follows a metric label.
pub(super) fn parse_value_after(rest: &str) -> Option<(f64, String)> {
    let trimmed =
        rest.trim_start_matches(|c: char| matches!(c, ':' | '=' | '[' | '(') || c.is_whitespace());

    let mut number_end = 0usize;
    for (idx, ch) in trimmed.char_indices() {
        let acceptable = ch.is_ascii_digit() || ch == '.' || ((ch == '-' || ch == '+') && idx == 0);
        if acceptable {
            number_end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if number_end == 0 {
        return None;
    }

    let value: f64 = trimmed[..number_end].parse().ok()?;
    let unit: String = trimmed[number_end..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphabetic() || *c == '%' || *c == '/' || *c == 'µ')
        .collect();
    Some((value, unit))
}

/// Find the value attached to the first of `labels` that occurs in `line` at a
/// word boundary. `line` must already be lowercased.
pub(super) fn extract_labelled_value(line: &str, labels: &[&str]) -> Option<(f64, String)> {
    for label in labels {
        let mut search_from = 0usize;
        while let Some(offset) = line[search_from..].find(label) {
            let start = search_from + offset;
            let after = start + label.len();
            let boundary_ok = start == 0
                || !matches!(line.as_bytes()[start - 1], b'a'..=b'z' | b'0'..=b'9' | b'_');
            if boundary_ok {
                if let Some(parsed) = parse_value_after(&line[after..]) {
                    return Some(parsed);
                }
            }
            search_from = after;
        }
    }
    None
}

/// Convert a duration value expressed in `unit` to seconds.
pub(super) fn scale_seconds(value: f64, unit: &str) -> f64 {
    match unit {
        "ns" | "nsec" | "nanos" | "nanoseconds" => value * 1e-9,
        "us" | "µs" | "usec" | "micros" | "microseconds" => value * 1e-6,
        "ms" | "msec" | "millis" | "milliseconds" => value * 1e-3,
        "m" | "min" | "mins" | "minutes" => value * 60.0,
        _ => value,
    }
}

/// Convert a memory value expressed in `unit` to bytes.
pub(super) fn scale_bytes(value: f64, unit: &str) -> usize {
    let scaled = match unit {
        "kb" => value * 1e3,
        "kib" | "k" => value * 1024.0,
        "mb" => value * 1e6,
        "mib" | "m" => value * 1024.0 * 1024.0,
        "gb" => value * 1e9,
        "gib" | "g" => value * 1024.0 * 1024.0 * 1024.0,
        _ => value,
    };
    if scaled.is_finite() && scaled > 0.0 {
        scaled as usize
    } else {
        0
    }
}

/// Convert a throughput value expressed in `unit` to operations per second.
pub(super) fn scale_throughput(value: f64, unit: &str) -> f64 {
    match unit {
        "kops/s" | "kops" | "k/s" => value * 1e3,
        "mops/s" | "mops" | "m/s" => value * 1e6,
        "gops/s" | "gops" => value * 1e9,
        _ => value,
    }
}

/// Parse real performance metrics out of a command's standard output.
///
/// Only values that are actually present in the output are recorded; every other
/// field stays at zero, which by convention means "not measured" — never a
/// fabricated default. Recognised forms are `label: value unit` and
/// `label=value unit` (case-insensitive), with the value optionally wrapped in
/// `[`/`(` as criterion prints it. When a label appears on several lines the
/// last matching line wins, because summary lines are conventionally printed
/// last. Labels fused with their unit (`execution_time_ms: 12`) are not
/// recognised and are left unmeasured rather than guessed.
pub(super) fn parse_metrics_from_output(output: &str) -> PerformanceMetrics {
    let mut metrics = PerformanceMetrics::default();

    for line in output.lines() {
        let lower = line.to_ascii_lowercase();

        if let Some((value, unit)) = extract_labelled_value(
            &lower,
            &["throughput", "ops_per_sec", "operations_per_second"],
        ) {
            metrics.throughput = scale_throughput(value, &unit);
        }
        if let Some((value, unit)) =
            extract_labelled_value(&lower, &["latency", "duration", "time"])
        {
            metrics.latency = scale_seconds(value, &unit);
        }
        if let Some((value, unit)) =
            extract_labelled_value(&lower, &["peak_memory", "memory_usage", "memory", "rss"])
        {
            metrics.memory_usage = scale_bytes(value, &unit);
        }
        if let Some((value, _unit)) = extract_labelled_value(&lower, &["cpu_usage", "cpu"]) {
            metrics.cpu_usage = value;
        }
        if let Some((value, _unit)) =
            extract_labelled_value(&lower, &["energy_consumption", "energy"])
        {
            metrics.energy_consumption = Some(value);
        }
    }

    metrics
}
