//! Notification transport abstraction for external delivery.
//!
//! This crate has **no network client dependency** (no `reqwest`, no `hyper`,
//! ...) and none may be added per COOLJAPAN policy (pure-Rust default
//! features, no new deps for this concern). External delivery of alerts and
//! CI/CD notifications (GitHub, Slack, Email, generic webhooks) therefore
//! goes through one of three [`NotificationTransport`] implementations:
//!
//! - [`CommandTransport`]: shells out to the system `curl` binary via
//!   [`std::process::Command`]. `curl` is close to universally available on
//!   CI runners and developer machines; its absence is surfaced as an
//!   explicit `Err` (never silently skipped).
//! - [`FileTransport`]: writes the JSON payload to a configured outbox
//!   directory, one file per delivery. Useful for offline/air-gapped CI or
//!   for tests that want to assert on what *would* have been sent.
//! - [`LogTransport`]: routes the payload through the `log` crate only.
//!
//! ## Non-negotiable contract
//!
//! An unconfigured or disabled destination MUST return
//! [`DeliveryStatus::Disabled`] wrapped in an `Err` from the caller (the
//! integration/alert layer), and callers MUST derive their delivery
//! statistics (sent/failed counters, "sent successfully" log lines, etc.)
//! from the returned [`DeliveryOutcome`] -- never assume success because a
//! function returned `Ok(())`.

use crate::error::{OptimError, Result};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

/// HTTP-ish method used by [`CommandTransport`] requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl TransportMethod {
    fn as_curl_verb(self) -> &'static str {
        match self {
            TransportMethod::Get => "GET",
            TransportMethod::Post => "POST",
            TransportMethod::Put => "PUT",
            TransportMethod::Patch => "PATCH",
            TransportMethod::Delete => "DELETE",
        }
    }
}

/// Destination for a single delivery attempt.
#[derive(Debug, Clone)]
pub struct DeliveryTarget {
    /// Destination URL. Used directly by [`CommandTransport`]; informational
    /// (logged / embedded in the outbox filename) for the other transports.
    pub url: String,
    /// HTTP method for [`CommandTransport`].
    pub method: TransportMethod,
    /// Extra request headers (e.g. `Authorization`, `Content-Type`).
    pub headers: HashMap<String, String>,
    /// Request timeout.
    pub timeout: Duration,
    /// Logical channel/recipient name (Slack channel, GitHub repo, email
    /// address list, ...) -- used for log lines and outbox file names, never
    /// parsed by the transport.
    pub channel: String,
}

impl DeliveryTarget {
    /// Convenience constructor for a JSON POST target.
    pub fn json_post(url: impl Into<String>, channel: impl Into<String>) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        Self {
            url: url.into(),
            method: TransportMethod::Post,
            headers,
            timeout: Duration::from_secs(30),
            channel: channel.into(),
        }
    }

    /// Attach a bearer token as an `Authorization` header.
    pub fn with_bearer_token(mut self, token: &str) -> Self {
        self.headers
            .insert("Authorization".to_string(), format!("Bearer {token}"));
        self
    }
}

/// Outcome of one delivery attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryStatus {
    /// The transport confirms the payload was accepted (HTTP 2xx, file
    /// written, log line emitted).
    Sent { detail: String },
    /// The transport ran but the sink reported failure (HTTP 4xx/5xx,
    /// non-zero exit, I/O error writing the outbox file).
    Failed { detail: String },
    /// Nothing was sent because this destination is not configured/enabled.
    Disabled { reason: String },
}

/// Full result of a delivery attempt, including the raw transport-level
/// status code where one exists. Statistics MUST be derived from this.
#[derive(Debug, Clone)]
pub struct DeliveryOutcome {
    pub status: DeliveryStatus,
    /// Raw HTTP status code, when the transport is HTTP-shaped
    /// ([`CommandTransport`] only; `None` for file/log transports).
    pub http_status: Option<u16>,
    /// Name of the transport that produced this outcome.
    pub transport: &'static str,
}

impl DeliveryOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self.status, DeliveryStatus::Sent { .. })
    }

    pub fn detail(&self) -> &str {
        match &self.status {
            DeliveryStatus::Sent { detail } => detail,
            DeliveryStatus::Failed { detail } => detail,
            DeliveryStatus::Disabled { reason } => reason,
        }
    }

    fn disabled(transport: &'static str, reason: impl Into<String>) -> Self {
        Self {
            status: DeliveryStatus::Disabled {
                reason: reason.into(),
            },
            http_status: None,
            transport,
        }
    }
}

/// A pluggable delivery mechanism for notification payloads.
pub trait NotificationTransport: std::fmt::Debug + Send + Sync {
    /// Short transport name, for logs and statistics.
    fn name(&self) -> &'static str;

    /// Attempt to deliver `payload` to `target`.
    ///
    /// Returns `Err` only for transport-level failures the caller cannot
    /// recover information from (process could not be spawned at all,
    /// filesystem unwritable, ...). A reachable-but-rejecting destination
    /// (HTTP 4xx/5xx) is a normal `Ok(DeliveryOutcome { status: Failed, .. })`,
    /// not an `Err`, so callers can inspect `http_status` for retry logic.
    fn send(&self, target: &DeliveryTarget, payload: &str) -> Result<DeliveryOutcome>;
}

/// Delivers over the network by shelling out to the system `curl` binary.
///
/// This is the only network-capable transport in the crate and it
/// deliberately avoids linking any HTTP client library: `curl` is invoked
/// as a subprocess, with the payload piped over stdin (never placed on the
/// command line, since argv is visible to other local users via `ps` and
/// payloads may contain secrets/newlines).
#[derive(Debug, Clone)]
pub struct CommandTransport {
    /// Path/name of the curl executable (resolved via `PATH` by default).
    pub curl_path: String,
}

impl Default for CommandTransport {
    fn default() -> Self {
        Self {
            curl_path: "curl".to_string(),
        }
    }
}

impl NotificationTransport for CommandTransport {
    fn name(&self) -> &'static str {
        "curl"
    }

    fn send(&self, target: &DeliveryTarget, payload: &str) -> Result<DeliveryOutcome> {
        let mut cmd = Command::new(&self.curl_path);
        cmd.arg("-sS")
            .arg("-o")
            .arg("/dev/null")
            .arg("-w")
            .arg("%{http_code}")
            .arg("--max-time")
            .arg(target.timeout.as_secs().max(1).to_string())
            .arg("-X")
            .arg(target.method.as_curl_verb())
            .arg("--data-binary")
            .arg("@-")
            .arg(&target.url);

        for (key, value) in &target.headers {
            cmd.arg("-H").arg(format!("{key}: {value}"));
        }

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                OptimError::MissingDependency(format!(
                    "system `curl` binary not found on PATH; cannot deliver to '{}'. \
                     Install curl, or configure a FileTransport/LogTransport instead.",
                    target.channel
                ))
            } else {
                OptimError::InvalidConfig(format!(
                    "failed to spawn curl for '{}': {err}",
                    target.channel
                ))
            }
        })?;

        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(payload.as_bytes()).map_err(|err| {
                OptimError::InvalidConfig(format!(
                    "failed to write payload to curl stdin for '{}': {err}",
                    target.channel
                ))
            })?;
        }

        let output = child.wait_with_output().map_err(|err| {
            OptimError::InvalidConfig(format!(
                "curl process failed for '{}': {err}",
                target.channel
            ))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Ok(DeliveryOutcome {
                status: DeliveryStatus::Failed {
                    detail: format!(
                        "curl exited with {:?}: {}",
                        output.status.code(),
                        stderr.trim()
                    ),
                },
                http_status: None,
                transport: "curl",
            });
        }

        let code_text = String::from_utf8_lossy(&output.stdout);
        let http_status: Option<u16> = code_text.trim().parse().ok();

        match http_status {
            Some(code) if (200..300).contains(&code) => Ok(DeliveryOutcome {
                status: DeliveryStatus::Sent {
                    detail: format!("HTTP {code}"),
                },
                http_status,
                transport: "curl",
            }),
            Some(code) => Ok(DeliveryOutcome {
                status: DeliveryStatus::Failed {
                    detail: format!("HTTP {code}"),
                },
                http_status,
                transport: "curl",
            }),
            None => Ok(DeliveryOutcome {
                status: DeliveryStatus::Failed {
                    detail: format!("curl produced no parseable status code: {:?}", code_text),
                },
                http_status: None,
                transport: "curl",
            }),
        }
    }
}

/// Destination for a single SMTP submission via [`CommandTransport::send_email`].
#[derive(Debug, Clone)]
pub struct SmtpTarget {
    pub host: String,
    pub port: u16,
    /// `true` submits over implicit TLS (`smtps://`); `false` submits over
    /// plain `smtp://` with no STARTTLS negotiation.
    pub use_tls: bool,
    pub username: Option<String>,
    pub password: Option<String>,
    pub from: String,
    pub to: Vec<String>,
    pub timeout: Duration,
}

impl CommandTransport {
    /// Submit a pre-formatted RFC 822 message (headers + blank line + body)
    /// over SMTP by shelling out to `curl`, which has built-in SMTP submission
    /// support (`--mail-from`/`--mail-rcpt`/`--upload-file`). This avoids
    /// pulling in an SMTP/email client dependency for a single feature.
    pub fn send_email(&self, target: &SmtpTarget, message: &str) -> Result<DeliveryOutcome> {
        if target.to.is_empty() {
            return Ok(DeliveryOutcome {
                status: DeliveryStatus::Failed {
                    detail: "no recipients configured".to_string(),
                },
                http_status: None,
                transport: "curl-smtp",
            });
        }

        let scheme = if target.use_tls { "smtps" } else { "smtp" };
        let url = format!("{scheme}://{}:{}", target.host, target.port);

        let mut cmd = Command::new(&self.curl_path);
        cmd.arg("-sS")
            .arg("--max-time")
            .arg(target.timeout.as_secs().max(1).to_string())
            .arg("--url")
            .arg(&url)
            .arg("--mail-from")
            .arg(format!("<{}>", target.from));

        for recipient in &target.to {
            cmd.arg("--mail-rcpt").arg(format!("<{recipient}>"));
        }

        if let (Some(user), Some(pass)) = (&target.username, &target.password) {
            cmd.arg("--user").arg(format!("{user}:{pass}"));
        }

        cmd.arg("--upload-file").arg("-");
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                OptimError::MissingDependency(
                    "system `curl` binary not found on PATH; cannot deliver email. Install \
                     curl, or configure a FileTransport/LogTransport instead."
                        .to_string(),
                )
            } else {
                OptimError::InvalidConfig(format!(
                    "failed to spawn curl for SMTP submission: {err}"
                ))
            }
        })?;

        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(message.as_bytes()).map_err(|err| {
                OptimError::InvalidConfig(format!(
                    "failed to write email body to curl stdin: {err}"
                ))
            })?;
        }

        let output = child.wait_with_output().map_err(|err| {
            OptimError::InvalidConfig(format!("curl SMTP submission process failed: {err}"))
        })?;

        if output.status.success() {
            Ok(DeliveryOutcome {
                status: DeliveryStatus::Sent {
                    detail: format!("submitted to {} recipient(s)", target.to.len()),
                },
                http_status: None,
                transport: "curl-smtp",
            })
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Ok(DeliveryOutcome {
                status: DeliveryStatus::Failed {
                    detail: format!(
                        "curl exited with {:?}: {}",
                        output.status.code(),
                        stderr.trim()
                    ),
                },
                http_status: None,
                transport: "curl-smtp",
            })
        }
    }
}

/// Deliver an email through `kind`, mirroring [`deliver`] but for the
/// SMTP-shaped [`CommandTransport::send_email`] rather than the generic HTTP
/// `send`. `File`/`Log` transports still work (the message is written to the
/// outbox / logged verbatim) so tests and offline CI can exercise the same
/// code path without a real mail server.
pub fn deliver_email(
    kind: &TransportKind,
    target: &SmtpTarget,
    message: &str,
) -> Result<DeliveryOutcome> {
    match kind {
        TransportKind::Command => CommandTransport::default().send_email(target, message),
        TransportKind::File(dir) => {
            let file_target = DeliveryTarget::json_post(
                format!("smtp://{}:{}", target.host, target.port),
                format!("email-{}", target.to.join("_")),
            );
            FileTransport::new(dir.clone()).send(&file_target, message)
        }
        TransportKind::Log => {
            let log_target = DeliveryTarget::json_post(
                format!("smtp://{}:{}", target.host, target.port),
                format!("email-{}", target.to.join("_")),
            );
            LogTransport.send(&log_target, message)
        }
        TransportKind::Disabled => Err(OptimError::InvalidConfig(format!(
            "no transport configured for email to {:?}",
            target.to
        ))),
    }
}

/// Delivers by writing the JSON payload to a file in a configured outbox
/// directory. Never touches the network. Suitable for air-gapped CI,
/// dry-runs, and tests that assert on emitted payloads.
#[derive(Debug, Clone)]
pub struct FileTransport {
    pub outbox_dir: PathBuf,
}

impl FileTransport {
    pub fn new(outbox_dir: impl Into<PathBuf>) -> Self {
        Self {
            outbox_dir: outbox_dir.into(),
        }
    }
}

impl NotificationTransport for FileTransport {
    fn name(&self) -> &'static str {
        "file"
    }

    fn send(&self, target: &DeliveryTarget, payload: &str) -> Result<DeliveryOutcome> {
        std::fs::create_dir_all(&self.outbox_dir).map_err(|err| {
            OptimError::InvalidConfig(format!(
                "cannot create outbox directory {:?}: {err}",
                self.outbox_dir
            ))
        })?;

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let safe_channel: String = target
            .channel
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let file_name = format!("{timestamp_ns}_{safe_channel}.json");
        let path = self.outbox_dir.join(file_name);

        std::fs::write(&path, payload).map_err(|err| {
            OptimError::InvalidConfig(format!("failed writing outbox file {path:?}: {err}"))
        })?;

        Ok(DeliveryOutcome {
            status: DeliveryStatus::Sent {
                detail: format!("wrote {}", path.display()),
            },
            http_status: None,
            transport: "file",
        })
    }
}

/// Delivers by routing the payload through the `log` crate at `info` level.
/// Always "succeeds" in the sense that the log call itself cannot fail, but
/// this is clearly not external delivery, so callers that require guaranteed
/// external delivery (e.g. a `Critical` alert) should not treat `LogTransport`
/// as satisfying that requirement.
#[derive(Debug, Clone, Default)]
pub struct LogTransport;

impl NotificationTransport for LogTransport {
    fn name(&self) -> &'static str {
        "log"
    }

    fn send(&self, target: &DeliveryTarget, payload: &str) -> Result<DeliveryOutcome> {
        log::info!(target: "optirs_bench::notification", "[{}] {}", target.channel, payload);
        Ok(DeliveryOutcome {
            status: DeliveryStatus::Sent {
                detail: "logged".to_string(),
            },
            http_status: None,
            transport: "log",
        })
    }
}

/// Selects which [`NotificationTransport`] a destination should use.
#[derive(Debug, Clone, Default)]
pub enum TransportKind {
    /// Real network delivery via the system `curl`.
    #[default]
    Command,
    /// Write payloads to a local outbox directory instead of the network.
    File(PathBuf),
    /// Route payloads through `log` only.
    Log,
    /// This destination has no delivery mechanism configured at all; every
    /// [`build_transport`] caller must still be able to report `Disabled`.
    Disabled,
}

/// Build the concrete transport for a [`TransportKind`]. Returns `None` for
/// [`TransportKind::Disabled`] -- callers must translate that into an
/// explicit `Disabled` [`DeliveryOutcome`]/`Err`, not a silent no-op.
pub fn build_transport(kind: &TransportKind) -> Option<Box<dyn NotificationTransport>> {
    match kind {
        TransportKind::Command => Some(Box::new(CommandTransport::default())),
        TransportKind::File(dir) => Some(Box::new(FileTransport::new(dir.clone()))),
        TransportKind::Log => Some(Box::new(LogTransport)),
        TransportKind::Disabled => None,
    }
}

/// Deliver a payload through `kind`, honoring the "unconfigured => Disabled
/// + Err" contract in one place so callers cannot accidentally skip it.
pub fn deliver(
    kind: &TransportKind,
    target: &DeliveryTarget,
    payload: &str,
) -> Result<DeliveryOutcome> {
    match build_transport(kind) {
        Some(transport) => transport.send(target, payload),
        None => {
            let outcome = DeliveryOutcome::disabled(
                "none",
                format!("no transport configured for '{}'", target.channel),
            );
            Err(OptimError::InvalidConfig(outcome.detail().to_string()))
        }
    }
}

/// Resolve the transport to use from the `OPTIRS_NOTIFICATION_TRANSPORT`
/// environment variable, falling back to [`TransportKind::Command`] (real
/// `curl` delivery) when the variable is unset.
///
/// Recognized values (case-insensitive): `command`/`curl` (default), `log`,
/// `file:<outbox-dir>`, `disabled`/`none`.
pub fn transport_kind_from_env() -> TransportKind {
    match std::env::var("OPTIRS_NOTIFICATION_TRANSPORT") {
        Ok(value) => parse_transport_kind(&value),
        Err(_) => TransportKind::Command,
    }
}

fn parse_transport_kind(value: &str) -> TransportKind {
    let trimmed = value.trim();
    if let Some(dir) = trimmed.strip_prefix("file:") {
        return TransportKind::File(PathBuf::from(dir));
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "log" => TransportKind::Log,
        "disabled" | "none" => TransportKind::Disabled,
        _ => TransportKind::Command,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_transport_writes_payload_and_reports_sent() {
        let dir = std::env::temp_dir().join(format!(
            "optirs_bench_notification_transport_test_{}",
            std::process::id()
        ));
        let transport = FileTransport::new(&dir);
        let target = DeliveryTarget::json_post("https://example.invalid/hook", "unit-test");

        let outcome = transport
            .send(&target, r#"{"hello":"world"}"#)
            .expect("file transport should not fail");

        assert!(outcome.is_success());
        assert_eq!(outcome.transport, "file");

        let mut found = false;
        for entry in std::fs::read_dir(&dir).expect("outbox dir should exist") {
            let entry = entry.expect("dir entry should be readable");
            let contents = std::fs::read_to_string(entry.path()).expect("file should be readable");
            if contents.contains("hello") {
                found = true;
            }
        }
        assert!(found, "expected outbox file containing the payload");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_transport_always_reports_sent() {
        let transport = LogTransport;
        let target = DeliveryTarget::json_post("https://example.invalid/hook", "unit-test");
        let outcome = transport
            .send(&target, "payload")
            .expect("log transport cannot fail");
        assert!(outcome.is_success());
        assert_eq!(outcome.transport, "log");
    }

    #[test]
    fn disabled_transport_kind_is_explicit_err() {
        let target = DeliveryTarget::json_post("https://example.invalid/hook", "unit-test");
        let result = deliver(&TransportKind::Disabled, &target, "payload");
        assert!(result.is_err(), "disabled transport must never be Ok(())");
    }

    #[test]
    fn parse_transport_kind_recognizes_all_variants() {
        assert!(matches!(parse_transport_kind("log"), TransportKind::Log));
        assert!(matches!(parse_transport_kind("LOG"), TransportKind::Log));
        assert!(matches!(
            parse_transport_kind("disabled"),
            TransportKind::Disabled
        ));
        assert!(matches!(
            parse_transport_kind("none"),
            TransportKind::Disabled
        ));
        assert!(matches!(
            parse_transport_kind("curl"),
            TransportKind::Command
        ));
        match parse_transport_kind("file:/tmp/outbox") {
            TransportKind::File(dir) => assert_eq!(dir, PathBuf::from("/tmp/outbox")),
            other => panic!("expected File variant, got {other:?}"),
        }
    }

    #[test]
    fn command_transport_missing_binary_is_explicit_err() {
        let transport = CommandTransport {
            curl_path: "optirs-bench-definitely-not-a-real-binary".to_string(),
        };
        let target = DeliveryTarget::json_post("https://example.invalid/hook", "unit-test");
        let result = transport.send(&target, "payload");
        assert!(
            result.is_err(),
            "missing curl must surface as Err, never Ok(())"
        );
    }
}
