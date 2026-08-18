// Container management for cross-platform testing
//
// This module provides container runtime management including Docker and Podman
// support for isolated, reproducible testing environments.

use crate::error::{OptimError, Result};
use std::process::{Command, Stdio};
use std::time::SystemTime;

/// Run a container-runtime command (docker/podman), returning an honest error if
/// the runtime cannot be spawned (e.g. not installed) or the command exits with a
/// non-zero status. This never fabricates success.
fn run_runtime_command(runtime: &str, args: &[&str]) -> Result<()> {
    let output = Command::new(runtime)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            OptimError::ResourceUnavailable(format!(
                "container runtime '{}' is not available (failed to spawn: {})",
                runtime, e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(OptimError::ExecutionError(format!(
            "'{} {}' failed with status {}: {}",
            runtime,
            args.join(" "),
            output.status,
            stderr.trim()
        )));
    }

    Ok(())
}

use super::config::*;
use super::types::platform_target_to_string;
use super::types::*;

/// Argv tail appended after the image reference in `docker create` /
/// `podman create`, so the container has a long-running foreground process
/// and does not exit the instant it starts.
///
/// Without this, `docker create --name X ubuntu:22.04` (no command) uses the
/// image's default `CMD` (an interactive shell), which exits immediately
/// once started with no attached tty/stdin -- `docker start` then succeeds,
/// but the container is already `Exited`, so every later `docker exec`
/// against it fails. `ensure_container_running`'s liveness check catches
/// that honestly (`ResourceUnavailable`), but the goal here is to make
/// container-based execution actually reachable, not merely to report its
/// absence correctly.
///
/// Linux-based images (all `PlatformTarget`s in this build except Windows
/// share the `ubuntu:22.04` base -- see `get_image_for_platform`) always
/// carry a POSIX `sleep`, so `sleep infinity` is used unconditionally there.
/// The Windows Server Core image has no `sleep`; `ping -t localhost` is the
/// standard keep-alive idiom for Windows containers. That path is untestable
/// in this environment (no Windows container runtime available here) and is
/// provided on a best-effort basis rather than left unhandled.
fn keep_alive_command(platform: &PlatformTarget) -> &'static [&'static str] {
    match platform {
        PlatformTarget::WindowsX86_64 => &["ping", "-t", "localhost"],
        _ => &["sleep", "infinity"],
    }
}

/// Build the full `create` argv (runtime-agnostic: used for both `docker`
/// and `podman`) for `container_id` running `image` on `platform`. Factored
/// out as a pure function so the keep-alive command placement is unit
/// testable without a container runtime installed.
fn create_args(
    container_id: &str,
    image: &str,
    platform: &PlatformTarget,
    config: &ContainerConfig,
) -> Vec<String> {
    let mut args = vec![
        "create".to_string(),
        "--name".to_string(),
        container_id.to_string(),
    ];

    if let Some(cpu_limit) = config.resource_limits.cpu_limit {
        args.push("--cpus".to_string());
        args.push(cpu_limit.to_string());
    }
    if let Some(memory_limit) = config.resource_limits.memory_limit {
        args.push("--memory".to_string());
        args.push(format!("{memory_limit}m"));
    }
    if let Some(process_limit) = config.resource_limits.process_limit {
        args.push("--pids-limit".to_string());
        args.push(process_limit.to_string());
    }

    // `bridge` is docker/podman's own implicit default, so it is only spelled out
    // explicitly when something other than the default was actually configured;
    // this keeps the common case's argv identical to before this config was wired
    // in at all.
    let network_flag = match &config.network.mode {
        NetworkMode::Bridge => None,
        NetworkMode::Host => Some("host".to_string()),
        NetworkMode::None => Some("none".to_string()),
        NetworkMode::Overlay => Some("overlay".to_string()),
        NetworkMode::Custom(name) => Some(name.clone()),
    };
    if let Some(network_flag) = network_flag {
        args.push("--network".to_string());
        args.push(network_flag);
    }
    for (host_port, container_port) in &config.network.port_mappings {
        args.push("-p".to_string());
        args.push(format!("{host_port}:{container_port}"));
    }
    for dns in &config.network.dns_servers {
        args.push("--dns".to_string());
        args.push(dns.clone());
    }
    for (host, ip) in &config.network.extra_hosts {
        args.push("--add-host".to_string());
        args.push(format!("{host}:{ip}"));
    }

    for volume in &config.volumes {
        args.push("-v".to_string());
        args.push(volume.clone());
    }
    for (key, value) in &config.env_vars {
        args.push("-e".to_string());
        args.push(format!("{key}={value}"));
    }

    args.push(image.to_string());
    args.extend(
        keep_alive_command(platform)
            .iter()
            .map(|part| part.to_string()),
    );
    args
}

/// Container manager for cross-platform testing
#[derive(Debug)]
pub struct ContainerManager {
    config: ContainerConfig,
    runtime: Box<dyn ContainerRuntimeTrait>,
}

/// Container runtime trait
pub trait ContainerRuntimeTrait: Send + Sync + std::fmt::Debug {
    /// Name of the executable this runtime shells out to (for example `docker`
    /// or `podman`). Callers that need to run additional runtime sub-commands —
    /// `exec` for in-container test execution, `inspect` for liveness checks —
    /// use this instead of hard-coding a binary name.
    fn runtime_binary(&self) -> &str;
    fn create_container(&self, platform: &PlatformTarget, image: &str) -> Result<ContainerInfo>;
    fn start_container(&self, container_id: &str) -> Result<()>;
    fn stop_container(&self, container_id: &str) -> Result<()>;
    fn remove_container(&self, container_id: &str) -> Result<()>;
    fn get_container_stats(&self, container_id: &str) -> Result<ContainerStats>;
}

/// Docker runtime implementation
#[derive(Debug)]
pub struct DockerRuntime {
    config: ContainerConfig,
}

/// Podman runtime implementation
#[derive(Debug)]
pub struct PodmanRuntime {
    config: ContainerConfig,
}

impl ContainerManager {
    /// Create new container manager
    pub fn new(config: ContainerConfig) -> Result<Self> {
        let runtime: Box<dyn ContainerRuntimeTrait> = match config.runtime {
            ContainerRuntime::Docker => Box::new(DockerRuntime::new(config.clone())?),
            ContainerRuntime::Podman => Box::new(PodmanRuntime::new(config.clone())?),
            ContainerRuntime::Containerd => Box::new(DockerRuntime::new(config.clone())?), // Use Docker interface
            ContainerRuntime::Custom(_) => Box::new(DockerRuntime::new(config.clone())?), // Fallback
        };

        Ok(Self { config, runtime })
    }

    /// Create container for specific platform
    pub async fn create_container_for_platform(
        &self,
        platform: &PlatformTarget,
    ) -> Result<ContainerInfo> {
        let image = self.get_image_for_platform(platform)?;
        let container = self.runtime.create_container(platform, &image)?;
        // If starting fails, best-effort remove the just-created container so we do
        // not leak it, then propagate the real error.
        if let Err(e) = self.runtime.start_container(&container.container_id) {
            let _ = self.runtime.remove_container(&container.container_id);
            return Err(e);
        }
        Ok(container)
    }

    /// Name of the container runtime executable in use (`docker`, `podman`, …).
    ///
    /// Exposed so that callers which need to run further runtime sub-commands —
    /// notably executing a test inside a provisioned container — invoke the same
    /// runtime that created it, instead of assuming `docker`.
    pub fn runtime_binary(&self) -> &str {
        self.runtime.runtime_binary()
    }

    /// Get base image for platform
    fn get_image_for_platform(&self, platform: &PlatformTarget) -> Result<String> {
        let base_image = match platform {
            PlatformTarget::LinuxX86_64 => "ubuntu:22.04",
            PlatformTarget::LinuxAarch64 => "ubuntu:22.04",
            PlatformTarget::WindowsX86_64 => "mcr.microsoft.com/windows/servercore:ltsc2022",
            PlatformTarget::MacOSX86_64 => "ubuntu:22.04", // macOS containers run on Linux base
            PlatformTarget::MacOSAarch64 => "ubuntu:22.04",
            _ => "ubuntu:22.04",
        };

        // Compose a valid image reference. Joining the prefix with ':' would produce
        // an invalid reference like "test:ubuntu:22.04" (tags cannot contain colons);
        // treat the prefix as a registry/namespace and join with '/', or use the base
        // image directly when no prefix is configured.
        let prefix = self.config.registry.image_prefix.trim_end_matches('/');
        if prefix.is_empty() {
            Ok(base_image.to_string())
        } else {
            Ok(format!("{}/{}", prefix, base_image))
        }
    }

    /// Stop and remove container
    pub async fn cleanup_container(&self, container_id: &str) -> Result<()> {
        self.runtime.stop_container(container_id)?;
        self.runtime.remove_container(container_id)?;
        Ok(())
    }
}

impl DockerRuntime {
    fn new(config: ContainerConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

impl ContainerRuntimeTrait for DockerRuntime {
    fn runtime_binary(&self) -> &str {
        "docker"
    }

    fn create_container(&self, platform: &PlatformTarget, image: &str) -> Result<ContainerInfo> {
        let container_id = format!(
            "test_{}_{}",
            platform_target_to_string(platform),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        // Actually create the container. If docker is missing or the command fails
        // (e.g. an invalid/nonexistent image), propagate the real error rather than
        // fabricating a "sim_" container. The trailing keep-alive command keeps the
        // container's main process running past `start`, so `docker exec` (used by
        // the orchestrator to run tests inside it) has something to attach to.
        let args = create_args(&container_id, image, platform, &self.config);
        run_runtime_command(
            "docker",
            &args.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;

        Ok(ContainerInfo {
            container_id: container_id.clone(),
            name: container_id,
            image: image.to_string(),
            platform: platform.clone(),
            status: ContainerStatus::Created,
            ports: vec![],
            resource_usage: ContainerStats::default(),
            created_at: SystemTime::now(),
            started_at: None,
        })
    }

    fn start_container(&self, container_id: &str) -> Result<()> {
        run_runtime_command("docker", &["start", container_id])
    }

    fn stop_container(&self, container_id: &str) -> Result<()> {
        run_runtime_command("docker", &["stop", container_id])
    }

    fn remove_container(&self, container_id: &str) -> Result<()> {
        run_runtime_command("docker", &["rm", container_id])
    }

    fn get_container_stats(&self, container_id: &str) -> Result<ContainerStats> {
        // Live per-container resource statistics are not implemented for this runtime;
        // returning all-zero stats would fabricate a real measurement. Be honest.
        Err(OptimError::UnsupportedOperation(format!(
            "live container statistics for '{}' are not implemented for the docker \
             runtime in this build",
            container_id
        )))
    }
}

impl PodmanRuntime {
    fn new(config: ContainerConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

impl ContainerRuntimeTrait for PodmanRuntime {
    fn runtime_binary(&self) -> &str {
        "podman"
    }

    fn create_container(&self, platform: &PlatformTarget, image: &str) -> Result<ContainerInfo> {
        let container_id = format!(
            "test_{}_{}",
            platform_target_to_string(platform),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        // Actually invoke podman (mirror of the docker path, including the
        // keep-alive command -- see `keep_alive_command`). If podman is missing or
        // the command fails, propagate the real error instead of fabricating an id.
        let args = create_args(&container_id, image, platform, &self.config);
        run_runtime_command(
            "podman",
            &args.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;

        Ok(ContainerInfo {
            container_id: container_id.clone(),
            name: container_id,
            image: image.to_string(),
            platform: platform.clone(),
            status: ContainerStatus::Created,
            ports: vec![],
            resource_usage: ContainerStats::default(),
            created_at: SystemTime::now(),
            started_at: None,
        })
    }

    fn start_container(&self, container_id: &str) -> Result<()> {
        run_runtime_command("podman", &["start", container_id])
    }

    fn stop_container(&self, container_id: &str) -> Result<()> {
        run_runtime_command("podman", &["stop", container_id])
    }

    fn remove_container(&self, container_id: &str) -> Result<()> {
        run_runtime_command("podman", &["rm", container_id])
    }

    fn get_container_stats(&self, container_id: &str) -> Result<ContainerStats> {
        // See DockerRuntime::get_container_stats — no real measurement is available,
        // so we do not fabricate zeroed statistics.
        Err(OptimError::UnsupportedOperation(format!(
            "live container statistics for '{}' are not implemented for the podman \
             runtime in this build",
            container_id
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `ContainerConfig` with every optional/collection field empty, so
    /// `create_args` emits no flags beyond the bare `create --name ID IMAGE`
    /// baseline. `ContainerConfig::default()` is deliberately *not* this --
    /// its resource limits are non-trivial (see `ContainerResourceLimits::default`)
    /// precisely so a fresh `ContainerManager` runs with sane real limits.
    fn empty_container_config() -> ContainerConfig {
        ContainerConfig {
            runtime: ContainerRuntime::Docker,
            registry: RegistryConfig::default(),
            image_tag_strategy: ImageTagStrategy::GitHash,
            resource_limits: ContainerResourceLimits {
                cpu_limit: None,
                memory_limit: None,
                network_limit: None,
                io_limit: None,
                process_limit: None,
            },
            network: ContainerNetworkConfig::default(),
            volumes: vec![],
            env_vars: Default::default(),
        }
    }

    #[test]
    fn test_container_manager_creation() {
        let config = ContainerConfig::default();
        let manager = ContainerManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_image_selection() {
        let config = ContainerConfig::default();
        let manager = ContainerManager::new(config).expect("unwrap failed");

        let linux_image = manager
            .get_image_for_platform(&PlatformTarget::LinuxX86_64)
            .expect("unwrap failed");
        assert!(linux_image.contains("ubuntu"));

        let windows_image = manager
            .get_image_for_platform(&PlatformTarget::WindowsX86_64)
            .expect("unwrap failed");
        assert!(windows_image.contains("windows"));
    }

    #[test]
    fn test_docker_create_bogus_image_is_error() {
        // Regression (F77): a bogus image (or an unavailable docker runtime) must
        // return Err, NOT a fabricated "sim_" container. "Invalid_Image_Name" fails
        // client-side with an "invalid reference format" error when docker is present
        // (uppercase is not a valid image reference), and fails to spawn when docker
        // is absent — Err in either case, with no side effects.
        let runtime =
            DockerRuntime::new(ContainerConfig::default()).expect("runtime construction succeeds");
        let result = runtime.create_container(&PlatformTarget::LinuxX86_64, "Invalid_Image_Name");
        assert!(
            result.is_err(),
            "a bogus image / unavailable docker runtime must return Err, not a simulated container"
        );
    }

    #[test]
    fn test_podman_create_bogus_image_is_error() {
        // Regression (F77): the podman path previously never invoked podman and just
        // fabricated an id. It must now actually run podman and return Err on failure
        // or when podman is unavailable.
        let runtime =
            PodmanRuntime::new(ContainerConfig::default()).expect("runtime construction succeeds");
        let result = runtime.create_container(&PlatformTarget::LinuxX86_64, "Invalid_Image_Name");
        assert!(
            result.is_err(),
            "a bogus image / unavailable podman runtime must return Err, not a fabricated id"
        );
    }

    #[test]
    fn test_create_args_keeps_linux_container_alive() {
        // Regression: `docker create --name X ubuntu:22.04` with no command uses the
        // image's default CMD, which exits immediately once started headless --
        // `ensure_container_running`'s later inspect check then always reports
        // "not running", so no in-container test could ever execute. The argv must
        // carry a long-running foreground command after the image.
        let args = create_args(
            "test_container",
            "ubuntu:22.04",
            &PlatformTarget::LinuxX86_64,
            &empty_container_config(),
        );
        assert_eq!(
            args,
            vec![
                "create",
                "--name",
                "test_container",
                "ubuntu:22.04",
                "sleep",
                "infinity"
            ]
        );
    }

    #[test]
    fn test_create_args_keep_alive_platform_specific() {
        // Every non-Windows platform in this build shares the ubuntu:22.04 base
        // image (see `ContainerManager::get_image_for_platform`), so `sleep
        // infinity` applies uniformly; Windows Server Core has no `sleep`.
        for platform in [
            PlatformTarget::LinuxX86_64,
            PlatformTarget::LinuxAarch64,
            PlatformTarget::MacOSX86_64,
            PlatformTarget::MacOSAarch64,
        ] {
            let args = create_args("c", "ubuntu:22.04", &platform, &empty_container_config());
            assert_eq!(&args[4..], &["sleep", "infinity"], "platform: {platform:?}");
        }

        let windows_args = create_args(
            "c",
            "mcr.microsoft.com/windows/servercore:ltsc2022",
            &PlatformTarget::WindowsX86_64,
            &empty_container_config(),
        );
        assert_eq!(&windows_args[4..], &["ping", "-t", "localhost"]);
    }

    #[test]
    fn test_create_args_wires_resource_limits_volumes_and_env_vars() {
        // The `config` field on `DockerRuntime`/`PodmanRuntime` used to be stashed
        // but never actually consulted when building the container's argv --
        // cpu/memory/process limits, non-default network mode, volumes, and
        // environment variables were silently dropped. They must now show up.
        let mut config = empty_container_config();
        config.resource_limits.cpu_limit = Some(1.5);
        config.resource_limits.memory_limit = Some(2048);
        config.resource_limits.process_limit = Some(256);
        config.network.mode = NetworkMode::Host;
        config.volumes = vec!["/host/data:/data".to_string()];
        config
            .env_vars
            .insert("OPTIRS_ENV".to_string(), "test".to_string());

        let args = create_args("c", "img:latest", &PlatformTarget::LinuxX86_64, &config);

        assert!(args.windows(2).any(|w| w == ["--cpus", "1.5"]));
        assert!(args.windows(2).any(|w| w == ["--memory", "2048m"]));
        assert!(args.windows(2).any(|w| w == ["--pids-limit", "256"]));
        assert!(args.windows(2).any(|w| w == ["--network", "host"]));
        assert!(args.windows(2).any(|w| w == ["-v", "/host/data:/data"]));
        assert!(args.windows(2).any(|w| w == ["-e", "OPTIRS_ENV=test"]));
        // The image and keep-alive tail must still come last, after every flag.
        assert_eq!(
            &args[args.len() - 3..],
            &["img:latest", "sleep", "infinity"]
        );
    }

    #[test]
    fn test_container_stats_is_honest_error() {
        // Live stats are not implemented; the method must not fabricate zeroed stats.
        let runtime =
            DockerRuntime::new(ContainerConfig::default()).expect("runtime construction succeeds");
        assert!(runtime.get_container_stats("nonexistent").is_err());
    }
}
