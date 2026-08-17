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
    /// Capabilities declared by the factory at registration time, used to
    /// enforce `PluginQuery::required_capabilities` in `matches_query`.
    pub capabilities: PluginCapabilities,
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

    /// Get the capabilities the produced optimizer declares. Backed by a
    /// default so existing `PluginFactoryWrapper` implementors outside this
    /// crate keep compiling; the default reports every capability absent
    /// (`PluginCapabilities::default()` is all-`false`), which is the safe
    /// direction to fail in for `PluginQuery::required_capabilities`
    /// filtering -- an unimplemented override under-promises rather than
    /// over-promising what the plugin can do.
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::default()
    }

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
    /// Enable plugin caching. When `false`, [`PluginRegistry::create_optimizer`]
    /// never reads or writes [`PluginCache`] -- every call reaches the
    /// plugin factory, and `get_cache_stats()` stays at all-zero. See
    /// [`PluginCache`] for the (f64-only) scope of what caching covers.
    pub enable_caching: bool,
    /// Maximum number of distinct plugin names [`PluginCache`] holds at
    /// once. Once reached, inserting a new entry evicts the
    /// least-recently-used one first (see [`PluginCache`]). `0` means
    /// "cache nothing": caching stays logically enabled (bypass is still
    /// controlled solely by `enable_caching`) but every lookup misses and
    /// nothing is ever retained.
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

/// Plugin cache for performance optimization.
///
/// # Scope: `f64` only
///
/// `CachedPlugin::plugin` is monomorphized to `Box<dyn OptimizerPlugin<f64>>`
/// -- there is no generic `PluginCache<A>`. [`PluginRegistry::create_optimizer::<A>`]
/// only consults this cache when `A = f64`; a call with `A = f32` (or any
/// other `Float` impl) always goes straight to the factory and never
/// touches `instances`, `stats.hits`, or `stats.misses`. This is a
/// deliberate scope limitation, not an oversight: caching f32 instances
/// too would need either a second, separately-bounded `HashMap` or an
/// `Any`-erased value type, and nothing in this crate currently creates
/// enough f32 optimizers through the registry to justify that complexity.
///
/// # Cache key and correctness
///
/// Entries are keyed by plugin *name*, but a lookup is only a hit when the
/// caller's [`OptimizerConfig`] also equals the config the cached instance
/// was built with (`CachedPlugin::config`). A name-only key would let a
/// caller requesting e.g. a different `learning_rate` silently receive an
/// instance built with someone else's config -- that would be exactly the
/// kind of fabricated-success this crate's stub-removal pass exists to
/// eliminate, so a config mismatch is treated as a miss (the stale entry is
/// replaced by a freshly created one) rather than returned.
///
/// # Eviction
///
/// Bounded by [`RegistryConfig::max_cache_size`]: inserting past the limit
/// evicts the least-recently-used entry first. "Recently used" is tracked
/// with a monotonically increasing `u64` sequence number bumped on every
/// insert and every hit, not a wall-clock timestamp -- two cache
/// operations completing within the same clock tick (common on fast
/// hardware or under `#[test]`) would otherwise tie under
/// `SystemTime`-based LRU and evict a nondeterministically-chosen entry.
#[derive(Debug)]
pub struct PluginCache {
    /// Cached plugin instances, keyed by plugin name.
    instances: HashMap<String, CachedPlugin>,
    /// Cache statistics
    stats: CacheStats,
    /// Source of the next `CachedPlugin::sequence` value; incremented on
    /// every insert and every hit so eviction has a real, deterministic
    /// "least recently used" ordering (see the struct-level doc comment).
    next_sequence: u64,
}

/// Cached plugin instance
#[derive(Debug)]
pub struct CachedPlugin {
    /// Plugin instance
    pub plugin: Box<dyn OptimizerPlugin<f64>>,
    /// The exact [`OptimizerConfig`] this instance was created with. A
    /// lookup with a different config is treated as a miss -- see
    /// [`PluginCache`]'s "Cache key and correctness" section.
    pub config: OptimizerConfig,
    /// Cache timestamp
    pub cached_at: std::time::SystemTime,
    /// Access count
    pub access_count: usize,
    /// Last accessed
    pub last_accessed: std::time::SystemTime,
    /// Recency ordinal used for LRU eviction; see [`PluginCache::next_sequence`].
    pub(super) sequence: u64,
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses (an f64 `create_optimizer` call that reached the
    /// factory: no matching cached entry existed, or its config differed)
    pub misses: usize,
    /// Total evictions
    pub evictions: usize,
    /// Approximate memory used by currently cached instances, in bytes.
    /// Computed as `sum(size_of_val(&*cached.plugin))` -- the real,
    /// runtime size of each cached optimizer's own concrete struct
    /// (resolved through its vtable, not guessed). This deliberately does
    /// **not** account for any heap allocations *inside* that struct
    /// (e.g. a `Vec<f64>` momentum buffer): this crate has no allocator
    /// instrumentation to attribute those bytes, and reporting only the
    /// immediate struct size is an honest undercount rather than a
    /// fabricated total.
    pub memory_used: usize,
}

/// Registry event listener trait
pub trait RegistryEventListener: Debug + Send + Sync {
    /// Called when a plugin is registered
    fn on_plugin_registered(&mut self, _info: &PluginInfo) {}

    /// Called when a plugin is unregistered
    fn on_plugin_unregistered(&mut self, _name: &str) {}

    /// Called when a plugin is loaded
    fn on_plugin_loaded(&mut self, _name: &str) {}

    /// Called when a plugin fails to load
    fn on_plugin_load_failed(&mut self, _name: &str, _error: &str) {}

    /// Called when a plugin is enabled/disabled
    fn on_plugin_status_changed(&mut self, _name: &str, _status: &PluginStatus) {}
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

        let capabilities = factory.capabilities();
        let registration = PluginRegistration {
            factory: Box::new(factory),
            info: info.clone(),
            capabilities,
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
        // A single write guard covers status check, validation, creation,
        // and the load_count/last_used update -- there is no read-then-
        // reacquire-as-write gap for another thread to unregister the
        // plugin (or race a concurrent `create_optimizer` call) in between.
        // The previous version dropped its read lock and reacquired a write
        // lock purely to bump the usage counters, so `factories.get_mut(name)`
        // could silently find nothing if the plugin was unregistered in
        // that window -- the statistics update for an otherwise-successful
        // creation would vanish with no error. Third-party factory code
        // still runs under `catch_unwind` (as before), so a panicking
        // plugin cannot poison this exclusive lock either.
        let mut factories = write_lock(&self.factories);
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
        // while `factories` is held write-locked, so a panic is caught
        // rather than allowed to poison the registry-wide lock.
        let optimizer = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
            // `PluginCache` is monomorphized to `Box<dyn OptimizerPlugin<f64>>`
            // (see its doc comment) and so cannot represent an f32 instance
            // at all -- this branch always reaches the factory, regardless
            // of `enable_caching`.
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
            let use_cache = self.config.enable_caching;

            // A cache hit must match on *both* plugin name and config (see
            // `PluginCache`'s "Cache key and correctness" doc section), so
            // the lookup runs before the factory call and can skip it
            // entirely on a hit -- `get_or_record_miss` also folds in the
            // `stats.misses` bump for every other outcome.
            let cached_hit = if use_cache {
                let mut cache = mutex_lock(&self.cache);
                cache.get_or_record_miss(name, &config)?
            } else {
                None
            };

            let opt_f64: Box<dyn OptimizerPlugin<f64>> = if let Some(hit) = cached_hit {
                hit
            } else {
                // Only clone `config` when it will actually be stored --
                // avoids the clone entirely when caching is disabled.
                let config_for_cache = use_cache.then(|| config.clone());
                let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    registration.factory.create_f64(config)
                }))
                .map_err(|_| {
                    OptimError::PluginLoadError(format!(
                        "plugin '{name}' panicked while creating an f64 optimizer"
                    ))
                })??;
                if let Some(cache_config) = config_for_cache {
                    // Same rationale as `get_or_record_miss`: `clone_plugin`
                    // is third-party code, called here under `factories`'s
                    // write lock, so a panic must be caught rather than
                    // allowed to unwind through it.
                    let created_ref = &created;
                    let cloned_for_cache =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            created_ref.clone_plugin()
                        }))
                        .map_err(|_| {
                            OptimError::PluginLoadError(format!(
                                "plugin '{name}' panicked while cloning a newly created f64 \
                                 optimizer for the cache"
                            ))
                        })?;
                    let mut cache = mutex_lock(&self.cache);
                    cache.insert(
                        name.to_string(),
                        cloned_for_cache,
                        cache_config,
                        self.config.max_cache_size,
                    );
                }
                created
            };

            let boxed_any: Box<dyn Any> = Box::new(opt_f64);
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

        // Update usage statistics under the same write guard used to read
        // and create -- no reacquisition, so this entry cannot have been
        // removed since the lookup above.
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
            if self.matches_query(&registration.info, &registration.capabilities, &query) {
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

    fn matches_query(
        &self,
        info: &PluginInfo,
        capabilities: &PluginCapabilities,
        query: &PluginQuery,
    ) -> bool {
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

        // Check required capabilities: every named capability must be
        // declared `true` by the plugin, or it is excluded from the
        // results. Previously this field was declared on `PluginQuery` and
        // never consulted at all, so a caller searching for e.g.
        // `["gpu_support"]` got back plugins that do not support GPUs.
        if !query
            .required_capabilities
            .iter()
            .all(|cap| capabilities.has_capability(cap))
        {
            return false;
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

    /// Recursively count candidate plugin files under `path` (same
    /// extension/name convention as `PluginLoader::is_plugin_file`: shared
    /// libraries, or a `plugin.toml` manifest).
    ///
    /// This crate has no dynamic-loading backend (see the module-level note
    /// in `plugin::loader` on why `dlopen`/`libloading` is not wired up),
    /// so a discovered file cannot actually be turned into a registered
    /// `PluginRegistration` here -- previously this returned a hardcoded
    /// `Ok(0)` regardless of what was on disk, which reads identically to
    /// "no plugins present" and "discovery is unimplemented". Returning the
    /// real count at least tells a caller the truth about what discovery
    /// *found*, even though loading them still requires
    /// `PluginRegistry::register_plugin` with a statically compiled
    /// factory.
    fn discover_plugins_in_directory(&self, path: &Path) -> Result<usize> {
        let mut count = 0;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                count += self.discover_plugins_in_directory(&entry_path)?;
                continue;
            }
            let is_candidate = match entry_path.extension().and_then(|e| e.to_str()) {
                Some("so") | Some("dylib") | Some("dll") => true,
                _ => entry_path.file_name().and_then(|n| n.to_str()) == Some("plugin.toml"),
            };
            if is_candidate {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Register any statically-compiled built-in plugins. There are
    /// currently none shipped with this crate -- optimizers ship as their
    /// own `OptimizerPlugin` implementations registered directly by the
    /// caller via `register_plugin`, not as a fixed built-in set -- so this
    /// legitimately has nothing to do. Kept as an explicit extension point
    /// (and call site in `global()`) rather than removed, so adding a
    /// future built-in plugin is a one-line change here.
    fn register_builtin_plugins(&mut self) {}
}

impl PluginCache {
    fn new() -> Self {
        Self {
            instances: HashMap::new(),
            stats: CacheStats::default(),
            next_sequence: 0,
        }
    }

    /// Look up a cached instance for `name`. Only a hit when `config`
    /// equals the config the cached instance was built with (see
    /// [`PluginCache`]'s "Cache key and correctness" doc section); any
    /// other outcome -- no entry, or a config mismatch -- is a miss and
    /// bumps `stats.misses`. On a hit, returns an independent clone (via
    /// [`OptimizerPlugin::clone_plugin`]) so the caller gets an owned
    /// instance while the cache keeps its own, and refreshes the entry's
    /// recency so it is not the next eviction candidate.
    ///
    /// `clone_plugin` is third-party trait code -- the registered plugin
    /// author's own impl, not anything this crate controls -- called here
    /// while both the caller's `factories` write lock and this cache's own
    /// mutex are held. A panic inside it is caught the same way the
    /// adjacent factory-call sites in `create_optimizer` already catch
    /// `create_f32`/`create_f64` panics, so it surfaces as `Err` instead of
    /// unwinding through two held locks. The poisoned entry is evicted
    /// (bumping `stats.evictions`) rather than left cached: recency was
    /// already refreshed above, so leaving it in place would make this
    /// plugin name return `Err` on every subsequent call forever instead
    /// of just this one, once a broken `clone_plugin` demonstrated it
    /// cannot be trusted.
    fn get_or_record_miss(
        &mut self,
        name: &str,
        config: &OptimizerConfig,
    ) -> Result<Option<Box<dyn OptimizerPlugin<f64>>>> {
        if let Some(entry) = self.instances.get_mut(name) {
            if &entry.config == config {
                self.next_sequence += 1;
                entry.access_count += 1;
                entry.last_accessed = std::time::SystemTime::now();
                entry.sequence = self.next_sequence;

                let plugin = &entry.plugin;
                let clone_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    plugin.clone_plugin()
                }));

                return match clone_result {
                    Ok(cloned) => {
                        self.stats.hits += 1;
                        Ok(Some(cloned))
                    }
                    Err(_) => {
                        self.instances.remove(name);
                        self.stats.evictions += 1;
                        Err(OptimError::PluginLoadError(format!(
                            "plugin '{name}' panicked while cloning a cached f64 optimizer; \
                             the poisoned cache entry has been evicted"
                        )))
                    }
                };
            }
        }
        self.stats.misses += 1;
        Ok(None)
    }

    /// Insert a freshly created instance, evicting the least-recently-used
    /// entry first if this insertion would exceed `max_size`.
    /// `max_size == 0` means "cache nothing" -- the entry is not inserted
    /// (and, since there is nothing to make room for, nothing is evicted
    /// either).
    fn insert(
        &mut self,
        name: String,
        plugin: Box<dyn OptimizerPlugin<f64>>,
        config: OptimizerConfig,
        max_size: usize,
    ) {
        if max_size == 0 {
            return;
        }
        if !self.instances.contains_key(&name) && self.instances.len() >= max_size {
            self.evict_lru();
        }
        self.next_sequence += 1;
        let now = std::time::SystemTime::now();
        self.instances.insert(
            name,
            CachedPlugin {
                plugin,
                config,
                cached_at: now,
                access_count: 1,
                last_accessed: now,
                sequence: self.next_sequence,
            },
        );
        self.recompute_memory_used();
    }

    /// Evict the entry with the smallest `sequence` (the one least
    /// recently inserted or hit). A no-op on an empty cache.
    fn evict_lru(&mut self) {
        let lru_name = self
            .instances
            .iter()
            .min_by_key(|(_, cached)| cached.sequence)
            .map(|(name, _)| name.clone());
        if let Some(lru_name) = lru_name {
            self.instances.remove(&lru_name);
            self.stats.evictions += 1;
            self.recompute_memory_used();
        }
    }

    /// Recompute `stats.memory_used` from the currently cached instances;
    /// see [`CacheStats::memory_used`] for exactly what this does and does
    /// not account for.
    fn recompute_memory_used(&mut self) {
        self.stats.memory_used = self
            .instances
            .values()
            .map(|cached| std::mem::size_of_val(&*cached.plugin))
            .sum();
    }

    /// Number of distinct plugin names currently cached. Exposed (crate-
    /// visible only) for tests asserting eviction actually bounds cache
    /// size rather than merely incrementing a counter.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.instances.len()
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

    #[test]
    fn discover_plugins_counts_real_files_on_disk() {
        // F69 regression: `discover_plugins_in_directory` previously
        // returned a hardcoded `Ok(0)` regardless of directory contents,
        // making a directory full of plugin files indistinguishable from
        // an empty one.
        let root = std::env::temp_dir().join(format!(
            "optirs_registry_discover_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).expect("create temp dir tree");

        std::fs::write(root.join("plugin.toml"), "[plugin]\nname = \"x\"").expect("write");
        std::fs::write(root.join("libfoo.so"), b"not a real library").expect("write");
        std::fs::write(root.join("readme.txt"), b"not a plugin").expect("write");
        std::fs::write(nested.join("bar.dylib"), b"not a real library").expect("write");

        let config = RegistryConfig {
            auto_discovery: true,
            ..RegistryConfig::default()
        };
        let registry = PluginRegistry::new(config);
        registry.add_search_path(&root);

        let discovered = registry
            .discover_plugins()
            .expect("discovery should succeed");
        assert_eq!(
            discovered, 3,
            "expected plugin.toml + libfoo.so + nested/bar.dylib, not readme.txt"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discover_plugins_is_a_noop_when_auto_discovery_disabled() {
        let root = std::env::temp_dir().join(format!(
            "optirs_registry_discover_disabled_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).expect("create temp dir");
        std::fs::write(root.join("plugin.toml"), "[plugin]\nname = \"x\"").expect("write");

        let config = RegistryConfig {
            auto_discovery: false,
            ..RegistryConfig::default()
        };
        let registry = PluginRegistry::new(config);
        registry.add_search_path(&root);

        assert_eq!(registry.discover_plugins().expect("should succeed"), 0);

        let _ = std::fs::remove_dir_all(&root);
    }
}

// Regression tests for the static plugin path: register -> create -> step,
// the panic-at-the-trust-boundary guard (a plugin panic must surface as an
// error, never poison the process-wide registry), and the lock-poisoning
// recovery helpers themselves.
#[cfg(test)]
mod regression_tests;
