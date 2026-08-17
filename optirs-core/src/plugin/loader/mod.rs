//! Plugin loader: dynamic loading, security scanning, and signature
//! verification for OptiRS plugins.
//!
//! Split via SplitRS from a single `loader.rs` (COOLJAPAN 2000-line policy).
//! This root stays a thin re-export layer: all real code lives in the
//! private submodules below and every item that was previously public is
//! re-exported here unchanged, so `plugin::loader::*` keeps the exact same
//! public surface as before the split.

mod functions;
mod loaderconfig_traits;
mod sandboxconfig_traits;
mod securitypolicy_traits;
mod securityscanresult_traits;
mod signatureverificationconfig_traits;
mod types;
mod types_7;

pub use types::{
    BuildInfo, CryptographicValidator, DependencyGraph, LoaderConfig, Permission,
    PermissionValidator, PluginConfig, PluginHandle, PluginLoadResult, RuntimeRequirements,
    ScanSeverity, ScanningRule, SecurityPolicy, SecurityScanResult, SecurityThreat,
    SignatureVerificationConfig, SignatureVerificationResult, SignerInfo, ThreatType, TrustedCA,
    ValidationRule,
};

pub use types_7::{
    CodeScanner, CpuRequirements, KeyUsage, LoadedPlugin, MalwareSignature, PluginLoader,
    PluginManifest, PluginMetadata, PluginSignature, PluginSourceConfig, SandboxConfig,
    SecurityFileScanResult, SecurityManager, SignatureAlgorithm,
};
