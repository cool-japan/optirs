// Plugin registry for managing and discovering optimizer plugins
//
// This module provides a centralized registry system for managing optimizer plugins,
// including registration, discovery, loading, and version management.

use super::core::*;
use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Take a read lock, recovering from poisoning instead of propagating the
/// panic. A panic inside one plugin call (which runs while these locks are
/// held) must never permanently brick the process-wide registry: the data
/// behind these locks stays structurally consistent even if one accessor
/// panicked partway through a call, since every mutation here is a single
/// insert/remove/assign with no multi-step invariant spanning the guard.
fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Take a write lock, recovering from poisoning. See [`read_lock`].
fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Take a mutex lock, recovering from poisoning. See [`read_lock`].
fn mutex_lock<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Parse a `major.minor.patch` prefix (ignoring any `-pre`/`+build` suffix,
/// per semver's separator rules) into a numeric triplet. `None` when the
/// string does not start with a dotted numeric version.
fn parse_version_triplet(version: &str) -> Option<(u64, u64, u64)> {
    let core = version.split(['-', '+']).next().unwrap_or(version);
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

/// Compare two version strings numerically by `(major, minor, patch)` when
/// both parse as dotted numeric versions (this crate has no `semver`
/// dependency, so pre-release/build metadata ordering is not modelled).
/// Falls back to a byte-lexicographic comparison for non-numeric version
/// strings so callers still get a total order rather than a panic.
///
/// Byte-lexicographic comparison alone is wrong for numeric versions --
/// `"0.10.0" < "0.9.0"` and `"1.10.0" < "1.9.0"` under `str`'s `Ord`, so an
/// ecosystem that ever reaches a double-digit minor or patch would silently
/// mis-resolve plugin version requirements.
fn version_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    match (parse_version_triplet(a), parse_version_triplet(b)) {
        (Some(va), Some(vb)) => va.cmp(&vb),
        _ => a.cmp(b),
    }
}

/// Central plugin registry for managing all optimizer plugins
#[derive(Debug)]
pub struct PluginRegistry {
    /// Registered plugin factories
    factories: RwLock<HashMap<String, PluginRegistration>>,
    /// Plugin search paths
    search_paths: RwLock<Vec<PathBuf>>,
    /// Registry configuration
    config: RegistryConfig,
    /// Plugin cache
    cache: Mutex<PluginCache>,
    /// Event listeners
    event_listeners: RwLock<Vec<Box<dyn RegistryEventListener>>>,
}

/// Plugin registration entry
#[derive(Debug)]
pub struct PluginRegistration {
    /// Plugin factory
    pub factory: Box<dyn PluginFactoryWrapper>,
    /// Plugin metadata
    pub info: PluginInfo,
    /// Registration timestamp
    pub registered_at: std::time::SystemTime,
    /// Plugin status
    pub status: PluginStatus,
    /// Load count
    pub load_count: usize,
    /// Last used timestamp
    pub last_used: Option<std::time::SystemTime>,
}

/// Wrapper trait for type-erased plugin factories
pub trait PluginFactoryWrapper: Debug + Send + Sync {
    /// Create optimizer with f32 precision
    fn create_f32(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>>;

    /// Create optimizer with f64 precision
    fn create_f64(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>>;

    /// Get factory information
    fn info(&self) -> PluginInfo;

    /// Validate configuration
    fn validate_config(&self, config: &OptimizerConfig) -> Result<()>;

    /// Get default configuration
    fn default_config(&self) -> OptimizerConfig;

    /// Get configuration schema
    fn config_schema(&self) -> ConfigSchema;

    /// Check if factory supports the given data type
    fn supports_type(&self, datatype: &DataType) -> bool;
}

/// Plugin status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginStatus {
    /// Plugin is active and available
    Active,
    /// Plugin is disabled
    Disabled,
    /// Plugin failed to load
    Failed(String),
    /// Plugin is deprecated
    Deprecated,
    /// Plugin is in maintenance mode
    Maintenance,
}

/// Registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Enable automatic plugin discovery
    pub auto_discovery: bool,
    /// Enable plugin validation on registration
    pub validate_on_registration: bool,
    /// Enable plugin caching
    pub enable_caching: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Plugin load timeout
    pub load_timeout: std::time::Duration,
    /// Enable plugin sandboxing (future feature)
    pub enable_sandboxing: bool,
    /// Allowed plugin sources
    pub allowed_sources: Vec<PluginSource>,
}

/// Plugin source types
#[derive(Debug, Clone)]
pub enum PluginSource {
    /// Built-in plugins
    BuiltIn,
    /// Local filesystem
    Local(PathBuf),
    /// Remote repository
    Remote(String),
    /// Package manager
    Package(String),
}

/// Plugin cache for performance optimization
#[derive(Debug)]
pub struct PluginCache {
    /// Cached plugin instances
    instances: HashMap<String, CachedPlugin>,
    /// Cache statistics
    stats: CacheStats,
}

/// Cached plugin instance
#[derive(Debug)]
pub struct CachedPlugin {
    /// Plugin instance
    pub plugin: Box<dyn OptimizerPlugin<f64>>,
    /// Cache timestamp
    pub cached_at: std::time::SystemTime,
    /// Access count
    pub access_count: usize,
    /// Last accessed
    pub last_accessed: std::time::SystemTime,
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Total evictions
    pub evictions: usize,
    /// Total memory used (bytes)
    pub memory_used: usize,
}

/// Registry event listener trait
pub trait RegistryEventListener: Debug + Send + Sync {
    /// Called when a plugin is registered
    fn on_plugin_registered(&mut self, info: &PluginInfo) {}

    /// Called when a plugin is unregistered
    fn on_plugin_unregistered(&mut self, name: &str) {}

    /// Called when a plugin is loaded
    fn on_plugin_loaded(&mut self, name: &str) {}

    /// Called when a plugin fails to load
    fn on_plugin_load_failed(&mut self, _name: &str, error: &str) {}

    /// Called when a plugin is enabled/disabled
    fn on_plugin_status_changed(&mut self, _name: &str, status: &PluginStatus) {}
}

/// Plugin search query
#[derive(Debug, Clone, Default)]
pub struct PluginQuery {
    /// Plugin name pattern
    pub name_pattern: Option<String>,
    /// Plugin category filter
    pub category: Option<PluginCategory>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Supported data types
    pub data_types: Vec<DataType>,
    /// Version requirements
    pub version_requirements: Option<VersionRequirement>,
    /// Tags filter
    pub tags: Vec<String>,
    /// Maximum results
    pub limit: Option<usize>,
}

/// Version requirement specification
#[derive(Debug, Clone)]
pub struct VersionRequirement {
    /// Minimum version (inclusive)
    pub min_version: Option<String>,
    /// Maximum version (exclusive)
    pub max_version: Option<String>,
    /// Exact version match
    pub exact_version: Option<String>,
}

/// Plugin search result
#[derive(Debug, Clone)]
pub struct PluginSearchResult {
    /// Matching plugins
    pub plugins: Vec<PluginInfo>,
    /// Total count (before limit)
    pub total_count: usize,
    /// Search query used
    pub query: PluginQuery,
    /// Search execution time
    pub search_time: std::time::Duration,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            factories: RwLock::new(HashMap::new()),
            search_paths: RwLock::new(Vec::new()),
            config,
            cache: Mutex::new(PluginCache::new()),
            event_listeners: RwLock::new(Vec::new()),
        }
    }

    /// Get the global plugin registry instance
    pub fn global() -> &'static Self {
        static INSTANCE: std::sync::OnceLock<PluginRegistry> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| {
            let config = RegistryConfig::default();
            let mut registry = PluginRegistry::new(config);
            registry.register_builtin_plugins();
            registry
        })
    }

    /// Register a plugin factory
    pub fn register_plugin<F>(&self, factory: F) -> Result<()>
    where
        F: PluginFactoryWrapper + 'static,
    {
        let info = factory.info();
        let name = info.name.clone();

        // Validate plugin if enabled
        if self.config.validate_on_registration {
            self.validate_plugin(&factory)?;
        }

        let registration = PluginRegistration {
            factory: Box::new(factory),
            info: info.clone(),
            registered_at: std::time::SystemTime::now(),
            status: PluginStatus::Active,
            load_count: 0,
            last_used: None,
        };

        {
            let mut factories = write_lock(&self.factories);
            factories.insert(name.clone(), registration);
        }

        // Notify event listeners
        {
            let mut listeners = write_lock(&self.event_listeners);
            for listener in listeners.iter_mut() {
                listener.on_plugin_registered(&info);
            }
        }

        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister_plugin(&self, name: &str) -> Result<()> {
        let mut factories = write_lock(&self.factories);
        if factories.remove(name).is_some() {
            // Notify event listeners
            drop(factories);
            let mut listeners = write_lock(&self.event_listeners);
            for listener in listeners.iter_mut() {
                listener.on_plugin_unregistered(name);
            }
            Ok(())
        } else {
            Err(OptimError::PluginNotFound(name.to_string()))
        }
    }

    /// Create optimizer instance from plugin
    pub fn create_optimizer<A>(
        &self,
        name: &str,
        config: OptimizerConfig,
    ) -> Result<Box<dyn OptimizerPlugin<A>>>
    where
        A: Float + Debug + Send + Sync + 'static,
    {
        let factories = read_lock(&self.factories);
        let registration = factories
            .get(name)
            .ok_or_else(|| OptimError::PluginNotFound(name.to_string()))?;

        // Check plugin status
        match registration.status {
            PluginStatus::Active => {}
            PluginStatus::Disabled => {
                return Err(OptimError::PluginDisabled(name.to_string()));
            }
            PluginStatus::Failed(ref error) => {
                return Err(OptimError::PluginLoadError(error.clone()));
            }
            PluginStatus::Deprecated => {
                // Log warning but continue
                log::warn!("Plugin '{}' is deprecated", name);
            }
            PluginStatus::Maintenance => {
                return Err(OptimError::PluginInMaintenance(name.to_string()));
            }
        }

        // Validate configuration
        registration.factory.validate_config(&config)?;

        // Create optimizer based on type. Third-party factory code runs here
        // while `factories` is still held read-locked below, so a panic is
        // caught rather than allowed to poison the registry-wide lock.
        let optimizer = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
            let opt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                registration.factory.create_f32(config)
            }))
            .map_err(|_| {
                OptimError::PluginLoadError(format!(
                    "plugin '{name}' panicked while creating an f32 optimizer"
                ))
            })??;
            // Safe downcast: A and f32 are the same type here (proven by the
            // TypeId check above), so boxing `opt` into `dyn Any` and
            // downcasting to `Box<dyn OptimizerPlugin<A>>` succeeds via
            // ordinary `Any` machinery -- no `transmute` of a trait object
            // (whose fat-pointer/vtable layout across distinct generic
            // instantiations is not guaranteed) is required.
            let boxed_any: Box<dyn Any> = Box::new(opt);
            *boxed_any
                .downcast::<Box<dyn OptimizerPlugin<A>>>()
                .map_err(|_| {
                    OptimError::UnsupportedDataType(
                        "internal error: f32 downcast failed".to_string(),
                    )
                })?
        } else if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f64>() {
            let opt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                registration.factory.create_f64(config)
            }))
            .map_err(|_| {
                OptimError::PluginLoadError(format!(
                    "plugin '{name}' panicked while creating an f64 optimizer"
                ))
            })??;
            let boxed_any: Box<dyn Any> = Box::new(opt);
            *boxed_any
                .downcast::<Box<dyn OptimizerPlugin<A>>>()
                .map_err(|_| {
                    OptimError::UnsupportedDataType(
                        "internal error: f64 downcast failed".to_string(),
                    )
                })?
        } else {
            return Err(OptimError::UnsupportedDataType(format!(
                "Type {} not supported",
                std::any::type_name::<A>()
            )));
        };

        // Update usage statistics
        drop(factories);
        let mut factories = write_lock(&self.factories);
        if let Some(registration) = factories.get_mut(name) {
            registration.load_count += 1;
            registration.last_used = Some(std::time::SystemTime::now());
        }

        // Notify event listeners
        drop(factories);
        let mut listeners = write_lock(&self.event_listeners);
        for listener in listeners.iter_mut() {
            listener.on_plugin_loaded(name);
        }

        Ok(optimizer)
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        let factories = read_lock(&self.factories);
        factories.values().map(|reg| reg.info.clone()).collect()
    }

    /// Search for plugins matching criteria
    pub fn search_plugins(&self, query: PluginQuery) -> PluginSearchResult {
        let start_time = std::time::Instant::now();
        let factories = read_lock(&self.factories);

        let mut matching_plugins = Vec::new();

        for registration in factories.values() {
            if self.matches_query(&registration.info, &query) {
                matching_plugins.push(registration.info.clone());
            }
        }

        let total_count = matching_plugins.len();

        // Apply limit if specified
        if let Some(limit) = query.limit {
            matching_plugins.truncate(limit);
        }

        let search_time = start_time.elapsed();

        PluginSearchResult {
            plugins: matching_plugins,
            total_count,
            query,
            search_time,
        }
    }

    /// Get plugin information
    pub fn get_plugin_info(&self, name: &str) -> Option<PluginInfo> {
        let factories = read_lock(&self.factories);
        factories.get(name).map(|reg| reg.info.clone())
    }

    /// Get plugin status
    pub fn get_plugin_status(&self, name: &str) -> Option<PluginStatus> {
        let factories = read_lock(&self.factories);
        factories.get(name).map(|reg| reg.status.clone())
    }

    /// Enable/disable plugin
    pub fn set_plugin_status(&self, name: &str, status: PluginStatus) -> Result<()> {
        let mut factories = write_lock(&self.factories);
        let registration = factories
            .get_mut(name)
            .ok_or_else(|| OptimError::PluginNotFound(name.to_string()))?;

        let old_status = registration.status.clone();
        registration.status = status.clone();

        // Notify event listeners if status changed
        if old_status != status {
            drop(factories);
            let mut listeners = write_lock(&self.event_listeners);
            for listener in listeners.iter_mut() {
                listener.on_plugin_status_changed(name, &status);
            }
        }

        Ok(())
    }

    /// Add plugin search path
    pub fn add_search_path<P: AsRef<Path>>(&self, path: P) {
        let mut search_paths = write_lock(&self.search_paths);
        search_paths.push(path.as_ref().to_path_buf());
    }

    /// Discover plugins in search paths
    pub fn discover_plugins(&self) -> Result<usize> {
        if !self.config.auto_discovery {
            return Ok(0);
        }

        let search_paths = read_lock(&self.search_paths);
        let mut discovered_count = 0;

        for path in search_paths.iter() {
            if path.exists() && path.is_dir() {
                discovered_count += self.discover_plugins_in_directory(path)?;
            }
        }

        Ok(discovered_count)
    }

    /// Add event listener
    pub fn add_event_listener(&self, listener: Box<dyn RegistryEventListener>) {
        let mut listeners = write_lock(&self.event_listeners);
        listeners.push(listener);
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        let cache = mutex_lock(&self.cache);
        cache.stats.clone()
    }

    /// Clear plugin cache
    pub fn clear_cache(&self) {
        let mut cache = mutex_lock(&self.cache);
        cache.instances.clear();
        cache.stats = CacheStats::default();
    }

    // Private helper methods

    fn validate_plugin(&self, factory: &dyn PluginFactoryWrapper) -> Result<()> {
        // Basic validation - check if plugin can be created
        let config = factory.default_config();
        let _optimizer = factory.create_f64(config)?;
        Ok(())
    }

    fn matches_query(&self, info: &PluginInfo, query: &PluginQuery) -> bool {
        // Check name pattern
        if let Some(ref pattern) = query.name_pattern {
            if !info.name.contains(pattern) {
                return false;
            }
        }

        // Check category
        if let Some(ref category) = query.category {
            if info.category != *category {
                return false;
            }
        }

        // Check data types
        if !query.data_types.is_empty() {
            let has_common_type = query
                .data_types
                .iter()
                .any(|dt| info.supported_types.contains(dt));
            if !has_common_type {
                return false;
            }
        }

        // Check tags
        if !query.tags.is_empty() {
            let has_common_tag = query.tags.iter().any(|tag| info.tags.contains(tag));
            if !has_common_tag {
                return false;
            }
        }

        // Check version requirements
        if let Some(ref version_req) = query.version_requirements {
            if !self.version_matches(&info.version, version_req) {
                return false;
            }
        }

        true
    }

    fn version_matches(&self, version: &str, requirement: &VersionRequirement) -> bool {
        if let Some(ref exact) = requirement.exact_version {
            return version == exact;
        }

        if let Some(ref min) = requirement.min_version {
            if version_cmp(version, min) == std::cmp::Ordering::Less {
                return false;
            }
        }

        if let Some(ref max) = requirement.max_version {
            if version_cmp(version, max) != std::cmp::Ordering::Less {
                return false;
            }
        }

        true
    }

    fn discover_plugins_in_directory(&self, path: &Path) -> Result<usize> {
        // In a real implementation, this would scan for plugin files
        // and attempt to load them dynamically
        Ok(0)
    }

    fn register_builtin_plugins(&mut self) {
        // Register built-in plugins would go here
        // For now, this is a placeholder
    }
}

impl PluginCache {
    fn new() -> Self {
        Self {
            instances: HashMap::new(),
            stats: CacheStats::default(),
        }
    }
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            validate_on_registration: true,
            enable_caching: true,
            max_cache_size: 100,
            load_timeout: std::time::Duration::from_secs(30),
            enable_sandboxing: false,
            allowed_sources: vec![
                PluginSource::BuiltIn,
                PluginSource::Local(PathBuf::from("./plugins")),
            ],
        }
    }
}

// Helper macro for registering plugins
#[macro_export]
macro_rules! register_optimizer_plugin {
    ($factory:expr) => {
        $crate::plugin::PluginRegistry::global().register_plugin($factory)?
    };
}

// Builder pattern for plugin queries
pub struct PluginQueryBuilder {
    query: PluginQuery,
}

impl Default for PluginQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginQueryBuilder {
    pub fn new() -> Self {
        Self {
            query: PluginQuery::default(),
        }
    }

    pub fn name_pattern(mut self, pattern: &str) -> Self {
        self.query.name_pattern = Some(pattern.to_string());
        self
    }

    pub fn category(mut self, category: PluginCategory) -> Self {
        self.query.category = Some(category);
        self
    }

    pub fn data_type(mut self, datatype: DataType) -> Self {
        self.query.data_types.push(datatype);
        self
    }

    pub fn tag(mut self, tag: &str) -> Self {
        self.query.tags.push(tag.to_string());
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.query.limit = Some(limit);
        self
    }

    pub fn build(self) -> PluginQuery {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_registry_creation() {
        let config = RegistryConfig::default();
        let registry = PluginRegistry::new(config);
        assert_eq!(registry.list_plugins().len(), 0);
    }

    #[test]
    fn test_plugin_query_builder() {
        let query = PluginQueryBuilder::new()
            .name_pattern("adam")
            .category(PluginCategory::FirstOrder)
            .data_type(DataType::F32)
            .limit(10)
            .build();

        assert_eq!(query.name_pattern, Some("adam".to_string()));
        assert_eq!(query.category, Some(PluginCategory::FirstOrder));
        assert_eq!(query.limit, Some(10));
    }
}
