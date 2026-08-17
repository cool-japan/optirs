// Container management for cross-platform testing
//
// This module provides container runtime management including Docker and Podman
// support for isolated, reproducible testing environments.

use crate::error::{OptimError, Result};
use std::collections::HashMap;
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
fn create_args(container_id: &str, image: &str, platform: &PlatformTarget) -> Vec<String> {
    let mut args = vec![
        "create".to_string(),
        "--name".to_string(),
        container_id.to_string(),
        image.to_string(),
    ];
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
        let args = create_args(&container_id, image, platform);
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
        let args = create_args(&container_id, image, platform);
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
            let args = create_args("c", "ubuntu:22.04", &platform);
            assert_eq!(&args[4..], &["sleep", "infinity"], "platform: {platform:?}");
        }

        let windows_args = create_args(
            "c",
            "mcr.microsoft.com/windows/servercore:ltsc2022",
            &PlatformTarget::WindowsX86_64,
        );
        assert_eq!(&windows_args[4..], &["ping", "-t", "localhost"]);
    }

    #[test]
    fn test_container_stats_is_honest_error() {
        // Live stats are not implemented; the method must not fabricate zeroed stats.
        let runtime =
            DockerRuntime::new(ContainerConfig::default()).expect("runtime construction succeeds");
        assert!(runtime.get_container_stats("nonexistent").is_err());
    }
}
