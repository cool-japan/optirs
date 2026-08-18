//! Real TOML parsing for `plugin.toml` manifests, backed by the `toml` +
//! `serde` crates. Replaces the previous hand-rolled, section-blind
//! line-by-line reader (see the F62 finding this closes).
//!
//! # Schema
//!
//! ```toml
//! manifest_version = "1.0"
//!
//! [plugin]
//! name = "my-plugin"
//! version = "1.2.3"
//! description = "..."
//! author = "..."
//! license = "MIT"
//! homepage = "https://example.com"
//! entry_point = "my_plugin_entry"
//! platforms = ["linux", "macos", "windows"]
//!
//! # Array-of-tables: any number of dependency entries.
//! [[plugin.dependencies]]
//! name = "some-other-plugin"
//! version = ">=1.0.0"   # optional, defaults to "*" (any version)
//! optional = false       # optional, defaults to false
//! dependency_type = "Plugin"  # REQUIRED: one of Plugin | SystemLibrary | Crate | Runtime
//!
//! # Array-of-tables: any number of permission entries, adjacently tagged.
//! [[plugin.permissions]]
//! type = "FileSystem"   # REQUIRED tag: one of the `Permission` variant names
//! value = "cache/data"  # REQUIRED for FileSystem/Network/Hardware/Custom, absent for the two unit variants
//!
//! [[plugin.permissions]]
//! type = "ProcessExecution"  # unit variant: no `value` key
//!
//! [build]
//! rust_version = "1.82.0"
//! target = "x86_64-unknown-linux-gnu"
//! profile = "release"
//! timestamp = "..."
//! compiler_flags = ["-C", "target-cpu=native"]
//!
//! [runtime]
//! min_rust_version = "1.75.0"
//! system_libraries = ["libfoo"]
//! environment_variables = ["MY_PLUGIN_HOME"]
//! memory_mb = 512
//!
//! [runtime.cpu_requirements]
//! min_cores = 4
//! instruction_sets = ["avx2"]
//! architectures = ["x86_64"]
//! ```
//!
//! # Defaults
//!
//! Every key is optional unless stated otherwise. A key absent from the
//! document leaves the corresponding [`PluginMetadata`] field at whatever
//! [`PluginMetadata::default_for_path`] already set it to -- this parser
//! only *overlays* onto that baseline, it never starts from a blank struct.
//! That means a manifest containing nothing but `[plugin] name = "x"` (or
//! even an entirely empty file) is valid input and produces a fully
//! populated, sensibly-defaulted [`PluginMetadata`].
//!
//! The two hard requirements are inside array-of-tables entries, because
//! defaulting them would silently misclassify data: every
//! `[[plugin.dependencies]]` entry must name a `dependency_type` (there is
//! no safe default discriminant -- see [`RawDependency`]), and every
//! `[[plugin.permissions]]` entry must carry its `type` tag (inherent to
//! adjacently-tagged deserialization).
//!
//! # Leniency
//!
//! Unrecognised keys anywhere in the document are ignored rather than
//! rejected (this parser never sets `#[serde(deny_unknown_fields)]`). A
//! stray key under the wrong section -- e.g. `[build] version = "..."`,
//! which is not a `BuildInfo` field -- is therefore silently dropped
//! instead of either corrupting `[plugin] version` (the bug this parser
//! replaces) or hard-failing the whole parse. See
//! `parse_plugin_toml_does_not_leak_across_sections` in `functions.rs` for
//! the regression this leniency is required to keep passing.

use crate::error::{OptimError, Result};
use crate::plugin::core::{DependencyType, PluginDependency};
use serde::Deserialize;
use std::path::Path;

use super::types::{BuildInfo, Permission, RuntimeRequirements};
use super::types_7::{CpuRequirements, PluginManifest, PluginMetadata};

/// Parse `content` as a `plugin.toml` manifest, overlaying every field the
/// document specifies onto [`PluginMetadata::default_for_path`]. `path` is
/// used only for the default-derived plugin name and for error context --
/// it need not exist on disk.
pub(super) fn parse_manifest_toml(content: &str, path: &Path) -> Result<PluginMetadata> {
    let raw: RawManifest = toml::from_str(content).map_err(|e| {
        OptimError::PluginLoadError(format!(
            "failed to parse plugin manifest '{}': {e}",
            path.display()
        ))
    })?;
    let mut metadata = PluginMetadata::default_for_path(path);
    raw.merge_into(&mut metadata);
    Ok(metadata)
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    manifest_version: Option<String>,
    plugin: Option<RawPluginSection>,
    build: Option<RawBuildSection>,
    runtime: Option<RawRuntimeSection>,
}

impl RawManifest {
    fn merge_into(self, metadata: &mut PluginMetadata) {
        if let Some(v) = self.manifest_version {
            metadata.manifest_version = v;
        }
        if let Some(plugin) = self.plugin {
            plugin.merge_into(&mut metadata.plugin);
        }
        if let Some(build) = self.build {
            build.merge_into(&mut metadata.build);
        }
        if let Some(runtime) = self.runtime {
            runtime.merge_into(&mut metadata.runtime);
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawPluginSection {
    name: Option<String>,
    version: Option<String>,
    description: Option<String>,
    author: Option<String>,
    license: Option<String>,
    homepage: Option<String>,
    entry_point: Option<String>,
    platforms: Option<Vec<String>>,
    #[serde(default)]
    dependencies: Vec<RawDependency>,
    #[serde(default)]
    permissions: Vec<RawPermission>,
}

impl RawPluginSection {
    fn merge_into(self, manifest: &mut PluginManifest) {
        if let Some(v) = self.name {
            manifest.name = v;
        }
        if let Some(v) = self.version {
            manifest.version = v;
        }
        if let Some(v) = self.description {
            manifest.description = v;
        }
        if let Some(v) = self.author {
            manifest.author = v;
        }
        if let Some(v) = self.license {
            manifest.license = v;
        }
        if let Some(v) = self.homepage {
            manifest.homepage = Some(v);
        }
        if let Some(v) = self.entry_point {
            manifest.entry_point = v;
        }
        if let Some(v) = self.platforms {
            manifest.platforms = v;
        }
        if !self.dependencies.is_empty() {
            manifest.dependencies = self
                .dependencies
                .into_iter()
                .map(RawDependency::into_dependency)
                .collect();
        }
        if !self.permissions.is_empty() {
            manifest.permissions = self
                .permissions
                .into_iter()
                .map(RawPermission::into_permission)
                .collect();
        }
    }
}

/// One `[[plugin.dependencies]]` array-of-tables entry.
///
/// `dependency_type` is deliberately required, not defaulted:
/// `PluginLoader::is_dependency_satisfied` (F65) checks each
/// `dependency_type` through a completely different code path (a loaded-
/// plugin lookup, a filesystem probe, or an unconditional "unverifiable"),
/// so silently guessing a default here could misclassify a dependency into
/// whichever check happens to be the most lenient -- exactly the kind of
/// fabricated-pass this crate's stub-removal pass exists to prevent.
#[derive(Debug, Deserialize)]
struct RawDependency {
    name: String,
    #[serde(default = "default_dependency_version")]
    version: String,
    #[serde(default)]
    optional: bool,
    dependency_type: DependencyType,
}

fn default_dependency_version() -> String {
    "*".to_string()
}

impl RawDependency {
    fn into_dependency(self) -> PluginDependency {
        PluginDependency {
            name: self.name,
            version: self.version,
            optional: self.optional,
            dependency_type: self.dependency_type,
        }
    }
}

/// One `[[plugin.permissions]]` array-of-tables entry, adjacently tagged by
/// a `type` key naming the [`Permission`] variant and an optional `value`
/// key carrying that variant's payload (absent for the two unit variants).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "value")]
enum RawPermission {
    FileSystem(String),
    Network(String),
    ProcessExecution,
    SystemInfo,
    Hardware(String),
    Custom(String),
}

impl RawPermission {
    fn into_permission(self) -> Permission {
        match self {
            RawPermission::FileSystem(s) => Permission::FileSystem(s),
            RawPermission::Network(s) => Permission::Network(s),
            RawPermission::ProcessExecution => Permission::ProcessExecution,
            RawPermission::SystemInfo => Permission::SystemInfo,
            RawPermission::Hardware(s) => Permission::Hardware(s),
            RawPermission::Custom(s) => Permission::Custom(s),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawBuildSection {
    rust_version: Option<String>,
    target: Option<String>,
    profile: Option<String>,
    timestamp: Option<String>,
    #[serde(default)]
    compiler_flags: Vec<String>,
}

impl RawBuildSection {
    fn merge_into(self, build: &mut BuildInfo) {
        if let Some(v) = self.rust_version {
            build.rust_version = v;
        }
        if let Some(v) = self.target {
            build.target = v;
        }
        if let Some(v) = self.profile {
            build.profile = v;
        }
        if let Some(v) = self.timestamp {
            build.timestamp = v;
        }
        if !self.compiler_flags.is_empty() {
            build.compiler_flags = self.compiler_flags;
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawRuntimeSection {
    min_rust_version: Option<String>,
    #[serde(default)]
    system_libraries: Vec<String>,
    #[serde(default)]
    environment_variables: Vec<String>,
    memory_mb: Option<usize>,
    cpu_requirements: Option<RawCpuRequirements>,
}

impl RawRuntimeSection {
    fn merge_into(self, runtime: &mut RuntimeRequirements) {
        if let Some(v) = self.min_rust_version {
            runtime.min_rust_version = v;
        }
        if !self.system_libraries.is_empty() {
            runtime.system_libraries = self.system_libraries;
        }
        if !self.environment_variables.is_empty() {
            runtime.environment_variables = self.environment_variables;
        }
        if self.memory_mb.is_some() {
            runtime.memory_mb = self.memory_mb;
        }
        if let Some(cpu) = self.cpu_requirements {
            cpu.merge_into(&mut runtime.cpu_requirements);
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawCpuRequirements {
    min_cores: Option<usize>,
    #[serde(default)]
    instruction_sets: Vec<String>,
    #[serde(default)]
    architectures: Vec<String>,
}

impl RawCpuRequirements {
    fn merge_into(self, cpu: &mut CpuRequirements) {
        if self.min_cores.is_some() {
            cpu.min_cores = self.min_cores;
        }
        if !self.instruction_sets.is_empty() {
            cpu.instruction_sets = self.instruction_sets;
        }
        if !self.architectures.is_empty() {
            cpu.architectures = self.architectures;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(content: &str) -> Result<PluginMetadata> {
        parse_manifest_toml(content, Path::new("/plugins/example/plugin.toml"))
    }

    // --- Full-featured manifest -------------------------------------------------

    #[test]
    fn parses_full_featured_manifest() {
        let toml = r#"
manifest_version = "1.1"

[plugin]
name = "full-plugin"
version = "2.3.4"
description = "exercises every documented manifest field"
author = "Test Author"
license = "Apache-2.0"
homepage = "https://example.com/full-plugin"
entry_point = "full_plugin_entry"
platforms = ["linux", "macos", "windows"]

[[plugin.dependencies]]
name = "base-plugin"
version = ">=1.0.0"
dependency_type = "Plugin"

[[plugin.dependencies]]
name = "libfoo"
version = "*"
optional = true
dependency_type = "SystemLibrary"

[[plugin.dependencies]]
name = "some-crate"
version = "1.2"
optional = true
dependency_type = "Crate"

[[plugin.permissions]]
type = "ProcessExecution"

[[plugin.permissions]]
type = "SystemInfo"

[[plugin.permissions]]
type = "FileSystem"
value = "data/cache"

[[plugin.permissions]]
type = "Network"
value = "api.example.com:443"

[[plugin.permissions]]
type = "Hardware"
value = "gpu0"

[[plugin.permissions]]
type = "Custom"
value = "telemetry"

[build]
rust_version = "1.82.0"
target = "x86_64-unknown-linux-gnu"
profile = "release"
timestamp = "2026-08-17T00:00:00Z"
compiler_flags = ["-C", "target-cpu=native"]

[runtime]
min_rust_version = "1.75.0"
system_libraries = ["libfoo"]
environment_variables = ["OPTIRS_FULL_PLUGIN_HOME"]
memory_mb = 512

[runtime.cpu_requirements]
min_cores = 4
instruction_sets = ["avx2", "sse4.2"]
architectures = ["x86_64"]
"#;
        let metadata = parse(toml).expect("full-featured manifest should parse");

        assert_eq!(metadata.manifest_version, "1.1");

        let plugin = &metadata.plugin;
        assert_eq!(plugin.name, "full-plugin");
        assert_eq!(plugin.version, "2.3.4");
        assert_eq!(
            plugin.description,
            "exercises every documented manifest field"
        );
        assert_eq!(plugin.author, "Test Author");
        assert_eq!(plugin.license, "Apache-2.0");
        assert_eq!(
            plugin.homepage,
            Some("https://example.com/full-plugin".to_string())
        );
        assert_eq!(plugin.entry_point, "full_plugin_entry");
        assert_eq!(
            plugin.platforms,
            vec![
                "linux".to_string(),
                "macos".to_string(),
                "windows".to_string()
            ]
        );

        assert_eq!(
            plugin.dependencies,
            vec![
                PluginDependency {
                    name: "base-plugin".to_string(),
                    version: ">=1.0.0".to_string(),
                    optional: false,
                    dependency_type: DependencyType::Plugin,
                },
                PluginDependency {
                    name: "libfoo".to_string(),
                    version: "*".to_string(),
                    optional: true,
                    dependency_type: DependencyType::SystemLibrary,
                },
                PluginDependency {
                    name: "some-crate".to_string(),
                    version: "1.2".to_string(),
                    optional: true,
                    dependency_type: DependencyType::Crate,
                },
            ],
            "array-of-tables dependencies must parse with the exact fields declared, in document order"
        );

        assert_eq!(
            plugin.permissions,
            vec![
                Permission::ProcessExecution,
                Permission::SystemInfo,
                Permission::FileSystem("data/cache".to_string()),
                Permission::Network("api.example.com:443".to_string()),
                Permission::Hardware("gpu0".to_string()),
                Permission::Custom("telemetry".to_string()),
            ],
            "array-of-tables permissions must parse every variant (unit and payload-carrying)"
        );

        let build = &metadata.build;
        assert_eq!(build.rust_version, "1.82.0");
        assert_eq!(build.target, "x86_64-unknown-linux-gnu");
        assert_eq!(build.profile, "release");
        assert_eq!(build.timestamp, "2026-08-17T00:00:00Z");
        assert_eq!(
            build.compiler_flags,
            vec!["-C".to_string(), "target-cpu=native".to_string()]
        );

        let runtime = &metadata.runtime;
        assert_eq!(runtime.min_rust_version, "1.75.0");
        assert_eq!(runtime.system_libraries, vec!["libfoo".to_string()]);
        assert_eq!(
            runtime.environment_variables,
            vec!["OPTIRS_FULL_PLUGIN_HOME".to_string()]
        );
        assert_eq!(runtime.memory_mb, Some(512));
        assert_eq!(runtime.cpu_requirements.min_cores, Some(4));
        assert_eq!(
            runtime.cpu_requirements.instruction_sets,
            vec!["avx2".to_string(), "sse4.2".to_string()]
        );
        assert_eq!(
            runtime.cpu_requirements.architectures,
            vec!["x86_64".to_string()]
        );
    }

    // --- Minimal manifest ---------------------------------------------------

    #[test]
    fn parses_minimal_manifest_filling_in_defaults() {
        let path = Path::new("/plugins/minimal-plugin/plugin.toml");
        let expected_defaults = PluginMetadata::default_for_path(path);

        let metadata = parse_manifest_toml("[plugin]\nname = \"minimal-plugin\"\n", path)
            .expect("minimal manifest should parse");

        assert_eq!(metadata.plugin.name, "minimal-plugin");
        // Everything else must fall back to the same defaults
        // `default_for_path` would have produced on its own -- a minimal
        // manifest is not a *different* default set, it is the same one
        // with exactly one field overridden.
        assert_eq!(metadata.plugin.version, expected_defaults.plugin.version);
        assert_eq!(
            metadata.plugin.description,
            expected_defaults.plugin.description
        );
        assert_eq!(metadata.plugin.author, expected_defaults.plugin.author);
        assert_eq!(metadata.plugin.license, expected_defaults.plugin.license);
        assert_eq!(metadata.plugin.homepage, None);
        assert_eq!(
            metadata.plugin.entry_point,
            expected_defaults.plugin.entry_point
        );
        assert_eq!(
            metadata.plugin.platforms,
            expected_defaults.plugin.platforms
        );
        assert!(metadata.plugin.dependencies.is_empty());
        assert!(metadata.plugin.permissions.is_empty());
        assert_eq!(
            metadata.build.rust_version,
            expected_defaults.build.rust_version
        );
        assert_eq!(
            metadata.runtime.min_rust_version,
            expected_defaults.runtime.min_rust_version
        );
    }

    #[test]
    fn parses_completely_empty_document_as_all_defaults() {
        // An empty file is syntactically valid TOML (the empty table) --
        // this must not be confused with a malformed manifest. The plugin
        // name falls all the way back to the path's file stem.
        let path = Path::new("/plugins/stem-name/plugin.toml");
        let metadata = parse_manifest_toml("", path).expect("empty document should parse");
        assert_eq!(metadata.plugin.name, "plugin");
        assert_eq!(metadata.plugin.version, "0.1.0");
        assert!(metadata.plugin.dependencies.is_empty());
        assert!(metadata.plugin.permissions.is_empty());
    }

    // --- Malformed input: each must fail with a specific, non-panicking error --

    #[test]
    fn rejects_syntactically_invalid_toml() {
        // Unterminated string literal.
        let err = parse("[plugin]\nname = \"unterminated\n").unwrap_err();
        match err {
            OptimError::PluginLoadError(msg) => {
                assert!(
                    msg.contains("/plugins/example/plugin.toml"),
                    "error must name which manifest failed to parse: {msg}"
                );
            }
            other => panic!("expected PluginLoadError, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_type_for_a_typed_field() {
        // `memory_mb` is a number; a string must be rejected, not silently
        // coerced or dropped.
        let result = parse("[runtime]\nmemory_mb = \"not-a-number\"\n");
        assert!(
            result.is_err(),
            "a string where a number is expected must not parse"
        );
        assert!(matches!(
            result.unwrap_err(),
            OptimError::PluginLoadError(_)
        ));
    }

    #[test]
    fn rejects_dependency_missing_required_dependency_type() {
        // `dependency_type` has no default (see `RawDependency`'s doc
        // comment) -- omitting it must fail loudly rather than silently
        // classify the dependency as some guessed type.
        let result = parse(
            r#"
[[plugin.dependencies]]
name = "foo"
version = "1.0"
"#,
        );
        let err = result.expect_err("missing dependency_type must be rejected");
        let OptimError::PluginLoadError(msg) = err else {
            panic!("expected PluginLoadError");
        };
        assert!(
            msg.contains("dependency_type"),
            "error should name the missing field, got: {msg}"
        );
    }

    #[test]
    fn rejects_dependency_with_unknown_dependency_type_variant() {
        let result = parse(
            r#"
[[plugin.dependencies]]
name = "foo"
dependency_type = "NotARealVariant"
"#,
        );
        assert!(
            result.is_err(),
            "an unknown dependency_type variant must not silently become one of the real ones"
        );
        assert!(matches!(
            result.unwrap_err(),
            OptimError::PluginLoadError(_)
        ));
    }

    #[test]
    fn rejects_permission_missing_type_tag() {
        // Adjacently-tagged: every entry must carry the `type` discriminant.
        let result = parse(
            r#"
[[plugin.permissions]]
value = "data"
"#,
        );
        assert!(
            result.is_err(),
            "a permission entry with no `type` tag must not silently parse as some default variant"
        );
        assert!(matches!(
            result.unwrap_err(),
            OptimError::PluginLoadError(_)
        ));
    }

    #[test]
    fn rejects_permission_with_unknown_type_tag() {
        let result = parse(
            r#"
[[plugin.permissions]]
type = "SuperUserAccess"
"#,
        );
        assert!(
            result.is_err(),
            "an unrecognised permission type tag must not silently parse"
        );
        assert!(matches!(
            result.unwrap_err(),
            OptimError::PluginLoadError(_)
        ));
    }

    // --- Existing leniency guarantee (documented above) ---------------------

    #[test]
    fn ignores_unrecognized_keys_instead_of_rejecting_them() {
        // `[build]` has no `bogus_field` -- must be silently ignored, not a
        // hard error (see the "Leniency" doc section for why).
        let metadata = parse(
            r#"
[plugin]
name = "lenient-plugin"

[build]
bogus_field = "whatever"
rust_version = "1.80.0"
"#,
        )
        .expect("unrecognized keys must not fail the parse");
        assert_eq!(metadata.plugin.name, "lenient-plugin");
        assert_eq!(metadata.build.rust_version, "1.80.0");
    }
}
