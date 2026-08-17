//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
#[allow(dead_code)]
use crate::plugin::core::*;
use crate::plugin::registry::*;
#[cfg(feature = "crypto")]
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::types::{
    CryptographicValidator, DependencyGraph, LoaderConfig, PluginConfig, SecurityScanResult,
    SignatureVerificationConfig, TrustedCA,
};
use super::types_7::{KeyUsage, PluginLoader, PluginMetadata, PluginSourceConfig};

/// Compute the SHA-256 digest of a file's full contents. `crypto`-gated
/// alongside the signature verification that consumes it.
#[cfg(feature = "crypto")]
pub(super) fn sha256_file(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut hasher = Sha256::new();
    hasher.update(&buffer);
    Ok(hasher.finalize().to_vec())
}

/// Decode a hex string (as produced by `encode_hex`) into raw bytes,
/// returning an honest error on malformed input rather than panicking.
#[cfg(feature = "crypto")]
pub(super) fn decode_hex(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) {
        return Err(OptimError::InvalidConfig(
            "hex string has an odd length".to_string(),
        ));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|_| OptimError::InvalidConfig(format!("invalid hex byte at offset {i}")))
        })
        .collect()
}

/// Strip a `#`-introduced trailing comment from one TOML line, respecting
/// quoted strings so a `#` inside a value (e.g. `description = "a # b"`) is
/// not mistaken for a comment marker.
pub(super) fn strip_toml_comment(line: &str) -> &str {
    let mut in_string = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Strip one layer of surrounding double quotes from a scalar TOML value,
/// or return the trimmed raw text unchanged if it is not quoted (covers
/// bare integers/booleans and malformed input alike -- this is a "best
/// effort on a simplified reader", not a validating parser).
pub(super) fn unquote_toml_string(raw_value: &str) -> String {
    let v = raw_value.trim();
    v.strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(v)
        .to_string()
}

/// Parse a single-line inline TOML string array, e.g. `["linux", "macos"]`.
/// Returns `None` for anything not shaped like `[ ... ]` (multi-line arrays
/// are not supported by this reader) or containing a non-string element.
pub(super) fn parse_toml_string_array(raw_value: &str) -> Option<Vec<String>> {
    let v = raw_value.trim();
    let inner = v.strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| {
            let item = item.trim();
            item.strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .map(|s| s.to_string())
        })
        .collect()
}

/// Best-effort check for whether a named system shared library is present
/// on disk, by looking for the conventional filename
/// (`lib{name}.so`/`.dylib` or `{name}.dll`) under the platform's usual
/// library search directories. This is not equivalent to what a real
/// dynamic linker does (it does not consult `LD_LIBRARY_PATH`,
/// `ldconfig`'s cache, rpath entries, or the Windows DLL search order) and
/// can both false-negative (a library findable by the linker through a
/// path this does not check) and false-positive (a same-named file that
/// is not actually loadable) -- but it is a genuine filesystem check
/// rather than a hardcoded `true`, which is the property `is_dependency_satisfied`
/// needs from it.
pub(super) fn system_library_exists(name: &str) -> bool {
    if name.trim().is_empty() {
        return false;
    }

    let candidates: Vec<PathBuf> = if cfg!(target_os = "windows") {
        let filename = format!("{name}.dll");
        ["C:\\Windows\\System32", "C:\\Windows\\SysWOW64"]
            .iter()
            .map(|dir| PathBuf::from(dir).join(&filename))
            .collect()
    } else if cfg!(target_os = "macos") {
        let filename = format!("lib{name}.dylib");
        [
            "/usr/lib",
            "/usr/local/lib",
            "/opt/homebrew/lib",
            "/opt/local/lib",
        ]
        .iter()
        .map(|dir| PathBuf::from(dir).join(&filename))
        .collect()
    } else {
        let filename = format!("lib{name}.so");
        [
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/lib",
            "/lib64",
        ]
        .iter()
        .map(|dir| PathBuf::from(dir).join(&filename))
        .collect()
    };

    candidates.iter().any(|path| path.is_file())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn test_plugin_loader_creation() {
        let config = LoaderConfig::default();
        let loader = PluginLoader::new(config);
        assert_eq!(loader.loaded_plugins.len(), 0);
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();
        graph.add_plugin("plugin_a", &["dep1".to_string(), "dep2".to_string()]);

        let dependents = graph.get_dependents("dep1");
        assert!(dependents.is_some());
        assert_eq!(dependents.expect("unwrap failed").len(), 1);
        assert_eq!(dependents.expect("unwrap failed")[0], "plugin_a");
    }

    // F65 regression: `is_dependency_satisfied` previously returned a bare
    // `true` for `SystemLibrary`, `Crate`, and `Runtime` dependency kinds
    // regardless of whether anything was actually checked, so a mandatory
    // dependency of any of these kinds could never block a plugin load.
    #[test]
    fn mandatory_crate_dependency_is_never_satisfied() {
        let loader = PluginLoader::new(LoaderConfig::default());
        let dep = PluginDependency {
            name: "some-crate".to_string(),
            version: "1.0".to_string(),
            optional: false,
            dependency_type: DependencyType::Crate,
        };
        assert!(
            loader.check_dependencies(&[dep]).is_err(),
            "an unverifiable mandatory Crate dependency must not silently pass"
        );
    }

    #[test]
    fn mandatory_runtime_dependency_is_never_satisfied() {
        let loader = PluginLoader::new(LoaderConfig::default());
        let dep = PluginDependency {
            name: "gpu".to_string(),
            version: "*".to_string(),
            optional: false,
            dependency_type: DependencyType::Runtime,
        };
        assert!(loader.check_dependencies(&[dep]).is_err());
    }

    #[test]
    fn optional_unverifiable_dependency_does_not_block_loading() {
        let loader = PluginLoader::new(LoaderConfig::default());
        let dep = PluginDependency {
            name: "nice-to-have".to_string(),
            version: "*".to_string(),
            optional: true,
            dependency_type: DependencyType::Crate,
        };
        assert!(
            loader.check_dependencies(&[dep]).is_ok(),
            "optional dependencies must not block loading even when unverifiable"
        );
    }

    #[test]
    fn system_library_probe_rejects_empty_name() {
        assert!(!system_library_exists(""));
        assert!(!system_library_exists("   "));
    }

    #[test]
    fn system_library_probe_rejects_a_name_that_does_not_exist_on_disk() {
        // A name with no plausible real-world library file: proves the
        // probe does a real filesystem check rather than defaulting to
        // `true` for anything non-empty.
        assert!(!system_library_exists(
            "definitely-not-a-real-system-library-xyz123"
        ));
    }

    #[test]
    fn test_security_scan_result() {
        let result = SecurityScanResult::default();
        assert!(!result.scan_successful);
        assert_eq!(result.security_score, 0.0);
    }

    // F62 regression: the original `parse_plugin_toml` matched bare keys
    // with no section tracking, so `[build] version = "..."` silently
    // overwrote `[plugin] version` -- both lines match the same bare key
    // `"version"`. This manifest puts a *different* version under `[build]`
    // than under `[plugin]`; a section-blind parser reports whichever one
    // it read last (here, "9.9.9-build" from `[build]`), which is wrong
    // either way but demonstrates the corruption. The fix must keep
    // `[plugin] version` as the plugin version regardless of what other
    // sections declare under the same key name.
    #[test]
    fn parse_plugin_toml_does_not_leak_across_sections() {
        let loader = PluginLoader::new(LoaderConfig::default());
        let toml = r#"
[plugin]
name = "real-plugin"
version = "1.2.3"
description = "the real plugin"
author = "someone"
license = "MIT"
entry_point = "plugin_main"

[build]
rust_version = "1.75.0"
version = "9.9.9-build"
target = "x86_64-unknown-linux-gnu"

[runtime]
min_rust_version = "1.70.0"
"#;
        let metadata = loader
            .parse_plugin_toml(toml, Path::new("plugin.toml"))
            .expect("parse should succeed");
        assert_eq!(metadata.plugin.name, "real-plugin");
        assert_eq!(
            metadata.plugin.version, "1.2.3",
            "the [build] section's `version` key must not overwrite [plugin] version"
        );
        assert_eq!(metadata.build.rust_version, "1.75.0");
        assert_eq!(metadata.build.target, "x86_64-unknown-linux-gnu");
        assert_eq!(metadata.runtime.min_rust_version, "1.70.0");
    }

    #[test]
    fn parse_plugin_toml_reads_inline_platform_array() {
        let loader = PluginLoader::new(LoaderConfig::default());
        let toml = r#"
[plugin]
name = "cross-platform-plugin"
version = "1.0.0"
platforms = ["linux", "macos", "windows"]
"#;
        let metadata = loader
            .parse_plugin_toml(toml, Path::new("plugin.toml"))
            .expect("parse should succeed");
        assert_eq!(
            metadata.plugin.platforms,
            vec![
                "linux".to_string(),
                "macos".to_string(),
                "windows".to_string()
            ]
        );
    }

    #[test]
    fn parse_plugin_toml_ignores_hash_inside_quoted_strings() {
        let loader = PluginLoader::new(LoaderConfig::default());
        let toml = r#"
[plugin]
name = "hash-plugin"
version = "1.0.0"
description = "handles the # symbol correctly"
"#;
        let metadata = loader
            .parse_plugin_toml(toml, Path::new("plugin.toml"))
            .expect("parse should succeed");
        assert_eq!(
            metadata.plugin.description,
            "handles the # symbol correctly"
        );
    }

    // F60/F63 regression: `load_plugin_from_registry` and
    // `load_plugin_from_http` previously fabricated an entire pipeline
    // (a synthetic registry response, a file written with literal
    // "dummy plugin package content", an unconditional `Ok(true)`
    // signature check, a file written with literal "dummy plugin binary")
    // before finally reporting failure -- multiple simulated steps that
    // could never actually fail, masking the real gap: this crate has no
    // dynamic-loading backend, so no source it downloads from can ever
    // become a callable `OptimizerPlugin`. Both paths now return an
    // immediate, honest `Err` (mirroring `load_plugin_from_git`), and
    // neither touches the filesystem.
    #[test]
    fn load_plugin_from_config_registry_source_fails_honestly() {
        let mut loader = PluginLoader::new(LoaderConfig::default());
        let config = PluginConfig {
            source: PluginSourceConfig::Registry {
                name: "some-plugin".to_string(),
                version: None,
            },
            name: "some-plugin".to_string(),
            version: None,
            config: HashMap::new(),
            auto_update: false,
        };
        let result = loader.load_plugin_from_config(config);
        assert!(
            result.is_err(),
            "registry loading must fail immediately, not fabricate a download"
        );
    }

    #[test]
    fn load_plugin_from_config_http_source_fails_honestly() {
        let mut loader = PluginLoader::new(LoaderConfig::default());
        let config = PluginConfig {
            source: PluginSourceConfig::Http("https://example.com/plugin.so".to_string()),
            name: "http-plugin".to_string(),
            version: None,
            config: HashMap::new(),
            auto_update: false,
        };
        let result = loader.load_plugin_from_config(config);
        assert!(
            result.is_err(),
            "HTTP loading must fail immediately, not fabricate a download"
        );
    }

    #[test]
    fn load_plugin_from_config_git_source_fails_honestly() {
        let mut loader = PluginLoader::new(LoaderConfig::default());
        let config = PluginConfig {
            source: PluginSourceConfig::Git {
                url: "https://example.com/plugin.git".to_string(),
                branch: None,
            },
            name: "git-plugin".to_string(),
            version: None,
            config: HashMap::new(),
            auto_update: false,
        };
        let result = loader.load_plugin_from_config(config);
        assert!(
            result.is_err(),
            "git loading must fail immediately, not fabricate a clone"
        );
    }
}

#[cfg(all(test, feature = "crypto"))]
pub(super) mod crypto_signature_tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    /// Exact bytes that were signed offline. Must not change without
    /// regenerating `SIGNATURE_HEX`.
    const PLUGIN_PAYLOAD: &[u8] = b"optirs-core plugin signature regression test payload v1";

    /// Public key whose private half produced `SIGNATURE_HEX`.
    const TRUSTED_PUBKEY_PEM: &str = "-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAs1G0JsOyE/zBHKlTDJb7
UqLpAyeAj9KZABgttEdzYdhWpV4hKJ+thAUr784lxq8cPg+1JzDfoI2U7uHI7JP5
jJKQv5Bc2VxL07KrKAdR0cGwNmXagn0FMzlndHpYqrnTFJmNsYbgErlzO7MJdrZ1
rBpZITzM9lcmpbXL+HfZJIuKje5hq0mB98Pew53k3Nu79XDJwreQ+xjgcT3lCShN
QoXbNkOiCjauZWPf5jpJxxRzcYq6BygQDywEapvtMrF8yE2zMhQhF/AJpbiKwS6y
NJO3nt+xjSJ72cKsFjYurOa/zkLtjChoEJ+hqqTE7yVhWTAGaiBxS9jXQyFUKmnb
bQIDAQAB
-----END PUBLIC KEY-----";

    /// A different, unrelated public key. The valid signature must NOT verify
    /// against it.
    const WRONG_PUBKEY_PEM: &str = "-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqqsEZZUULbT7eKqfsaG2
bAYN+RKDqxUxxzwq05TqPAbjByX4pljj0wABQk4jqj5FG7OjoTQeS8/E7SFkPd9s
BeJ/SMsuwQwL92YjLvnnN6jeBg8x56oV3aN+5oPJTyZR3999IVChckohRO5dqIY8
ZjwcSZXlpSXRB4T11I2TzkSo/8HgEbf5cJQixkfRiVRSOLLV5bYJyxceFCoYr394
JxO9leUMq+BNU5V8x61F18nEdf+3vcFNLxhKMWe2XAcF3wAeWqSPo+cPO8d/A9lk
zPrb8BbrUD9GYrvFUhg8/6htcvFDkdUeegiKNEU+QxHC0ZwUzTBObRxQbaaq4wsL
gQIDAQAB
-----END PUBLIC KEY-----";

    /// Hex-encoded PKCS#1 v1.5 signature over `SHA256(PLUGIN_PAYLOAD)`.
    const SIGNATURE_HEX: &str = "87feb12ff7e97618aa8babf37c0bacf374923feb9110b1eda0fce8cf7498bbe30b65e7266d2d79f54925f04df2bfaff0c0044d5b617e2e3b85d45fef33cbdd525f4cfa2f19486291b953a96707e66447ce30b7aca81958fa99dd5dcf68a7d9d6e3ed77cb06a4465b50d03fc89719251860212819755eeda918a2f5111b3d5fdc01cc88c6ef93fc05b165589a2497c57b481a1b9102e544e98eeb543723db599b39e7e72fc7544d8f1d967b1909cce2f94b871587089810877aa7d8a5580a7cd77327c166c2b38a321381d6dad67b4b3a0cd3088713e8cb388fdcd6c927728ac54aa5b899b305bd550145d5b4dcd4d4249f9747dc008adce3b5abd454a4d5c6f2";

    fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("optirs_sig_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn validator(pubkey_pem: Option<&str>) -> CryptographicValidator {
        let config = SignatureVerificationConfig {
            enabled: true,
            ..SignatureVerificationConfig::default()
        };
        let cas = match pubkey_pem {
            Some(pem) => vec![TrustedCA {
                name: "test-ca".to_string(),
                public_key: pem.to_string(),
                certificate: String::new(),
                key_usage: vec![KeyUsage::CodeSigning],
                valid_from: SystemTime::UNIX_EPOCH,
                valid_until: SystemTime::now() + Duration::from_secs(3600),
            }],
            None => Vec::new(),
        };
        CryptographicValidator::new(cas, config)
    }

    fn stage(dir: &std::path::Path, payload: &[u8], sig_hex: Option<&str>) -> std::path::PathBuf {
        let plugin_path = dir.join("libmyplugin.so");
        std::fs::write(&plugin_path, payload).expect("write plugin file");
        if let Some(sig) = sig_hex {
            std::fs::write(dir.join("plugin.sig"), sig).expect("write signature file");
        }
        plugin_path
    }

    #[test]
    fn accepts_valid_signature_from_trusted_key() {
        let dir = unique_temp_dir("accept");
        let plugin_path = stage(&dir, PLUGIN_PAYLOAD, Some(SIGNATURE_HEX));
        let meta = PluginMetadata::default_for_path(&plugin_path);
        let result = validator(Some(TRUSTED_PUBKEY_PEM))
            .verify_plugin_signature(&plugin_path, &meta)
            .expect("verification should not error");
        assert!(
            result.valid,
            "a valid signature from a trusted key must verify; errors: {:?}",
            result.errors
        );
        assert!(result.chain_valid);
        assert!(result.errors.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_tampered_payload() {
        let dir = unique_temp_dir("tamper");
        let mut tampered = PLUGIN_PAYLOAD.to_vec();
        tampered.extend_from_slice(b"!! injected malicious bytes");
        let plugin_path = stage(&dir, &tampered, Some(SIGNATURE_HEX));
        let meta = PluginMetadata::default_for_path(&plugin_path);
        let result = validator(Some(TRUSTED_PUBKEY_PEM))
            .verify_plugin_signature(&plugin_path, &meta)
            .expect("verification should not error");
        assert!(!result.valid, "a tampered payload must not verify");
        assert!(!result.chain_valid);
        assert!(!result.errors.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_signature_from_untrusted_key() {
        let dir = unique_temp_dir("wrongkey");
        let plugin_path = stage(&dir, PLUGIN_PAYLOAD, Some(SIGNATURE_HEX));
        let meta = PluginMetadata::default_for_path(&plugin_path);
        let result = validator(Some(WRONG_PUBKEY_PEM))
            .verify_plugin_signature(&plugin_path, &meta)
            .expect("verification should not error");
        assert!(
            !result.valid,
            "an otherwise-valid signature must not verify against an untrusted key"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_when_no_trusted_keys_configured() {
        let dir = unique_temp_dir("nocas");
        let plugin_path = stage(&dir, PLUGIN_PAYLOAD, Some(SIGNATURE_HEX));
        let meta = PluginMetadata::default_for_path(&plugin_path);
        let result = validator(None)
            .verify_plugin_signature(&plugin_path, &meta)
            .expect("verification should not error");
        assert!(
            !result.valid,
            "with no trusted keys nothing can be verified"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_missing_signature_file() {
        let dir = unique_temp_dir("nosig");
        let plugin_path = stage(&dir, PLUGIN_PAYLOAD, None);
        let meta = PluginMetadata::default_for_path(&plugin_path);
        let result = validator(Some(TRUSTED_PUBKEY_PEM))
            .verify_plugin_signature(&plugin_path, &meta)
            .expect("verification should not error");
        assert!(
            !result.valid,
            "a plugin with no signature file must not verify"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(all(test, not(feature = "crypto")))]
pub(super) mod no_crypto_signature_tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn verification_fails_closed_without_crypto_feature() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("optirs_sig_nocrypto_{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let plugin_path = dir.join("libmyplugin.so");
        std::fs::write(&plugin_path, b"payload").expect("write plugin");
        std::fs::write(dir.join("plugin.sig"), "00").expect("write sig");

        let config = SignatureVerificationConfig {
            enabled: true,
            ..SignatureVerificationConfig::default()
        };
        let validator = CryptographicValidator::new(Vec::new(), config);
        let meta = PluginMetadata::default_for_path(&plugin_path);
        let result = validator
            .verify_plugin_signature(&plugin_path, &meta)
            .expect("verification should not error");
        assert!(
            !result.valid,
            "without the crypto feature, signature verification must fail closed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
