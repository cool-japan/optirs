//! Regression tests for the static plugin path: register -> create -> step,
//! `PluginCache` (hit/miss/eviction/panic-safety), the panic-at-the-trust-
//! boundary guard (a plugin panic must surface as an error, never poison
//! the process-wide registry), and the lock-poisoning recovery helpers
//! themselves.
//!
//! Split out of `registry.rs` into this file (COOLJAPAN 2000-line policy)
//! -- `registry.rs` declares `mod regression_tests;` under `#[cfg(test)]`,
//! which Rust resolves to this file with no `mod.rs` needed since
//! `registry.rs` is a plain sibling file, not a directory module.

use super::*;
use scirs2_core::ndarray::Array1;

/// Minimal SGD-style optimizer used as a well-behaved test plugin.
#[derive(Debug, Clone)]
struct TestOptimizer<A: Float> {
    lr: A,
    initialized: bool,
}

impl<A: Float + Debug + Send + Sync + 'static> OptimizerPlugin<A> for TestOptimizer<A> {
    fn step(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>> {
        // Element-wise to avoid a `ScalarOperand` bound on `A`.
        let out: Array1<A> = params
            .iter()
            .zip(gradients.iter())
            .map(|(&p, &g)| p - g * self.lr)
            .collect();
        Ok(out)
    }
    fn name(&self) -> &str {
        "test-sgd"
    }
    fn version(&self) -> &str {
        "1.0.0"
    }
    fn plugin_info(&self) -> PluginInfo {
        PluginInfo {
            name: "test-sgd".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::default()
    }
    fn initialize(&mut self, _paramshape: &[usize]) -> Result<()> {
        self.initialized = true;
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }
    fn get_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn set_config(&mut self, _config: OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn get_state(&self) -> Result<OptimizerState> {
        Ok(OptimizerState::default())
    }
    fn set_state(&mut self, _state: OptimizerState) -> Result<()> {
        Ok(())
    }
    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<A>> {
        Box::new(self.clone())
    }
}

#[derive(Debug)]
struct TestFactory;

impl PluginFactoryWrapper for TestFactory {
    fn create_f32(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        Ok(Box::new(TestOptimizer::<f32> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn create_f64(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        Ok(Box::new(TestOptimizer::<f64> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "test-sgd".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn validate_config(&self, _config: &OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn config_schema(&self) -> ConfigSchema {
        ConfigSchema {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            version: "1.0.0".to_string(),
        }
    }
    fn supports_type(&self, _datatype: &DataType) -> bool {
        true
    }
}

/// A factory whose creation always panics -- stands in for third-party
/// plugin code that blows up while a registry lock is held.
#[derive(Debug)]
struct PanicFactory;

impl PluginFactoryWrapper for PanicFactory {
    fn create_f32(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        panic!("intentional panic inside factory create_f32");
    }
    fn create_f64(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        panic!("intentional panic inside factory create_f64");
    }
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "panic-plugin".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn validate_config(&self, _config: &OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn config_schema(&self) -> ConfigSchema {
        ConfigSchema {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            version: "1.0.0".to_string(),
        }
    }
    fn supports_type(&self, _datatype: &DataType) -> bool {
        true
    }
}

/// An optimizer whose `clone_plugin` panics -- stands in for a
/// third-party plugin author's buggy `Clone` logic, to prove a panic
/// there is caught rather than allowed to unwind through
/// `create_optimizer`'s held locks (`factories`'s write lock and
/// `PluginCache`'s own mutex).
#[derive(Debug, Clone)]
struct PanicOnCloneOptimizer;

impl OptimizerPlugin<f64> for PanicOnCloneOptimizer {
    fn step(&mut self, params: &Array1<f64>, _gradients: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(params.clone())
    }
    fn name(&self) -> &str {
        "panic-on-clone"
    }
    fn version(&self) -> &str {
        "1.0.0"
    }
    fn plugin_info(&self) -> PluginInfo {
        PluginInfo {
            name: "panic-on-clone".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::default()
    }
    fn initialize(&mut self, _paramshape: &[usize]) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn get_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn set_config(&mut self, _config: OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn get_state(&self) -> Result<OptimizerState> {
        Ok(OptimizerState::default())
    }
    fn set_state(&mut self, _state: OptimizerState) -> Result<()> {
        Ok(())
    }
    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<f64>> {
        panic!("intentional panic inside clone_plugin");
    }
}

#[derive(Debug)]
struct PanicOnCloneFactory;

impl PluginFactoryWrapper for PanicOnCloneFactory {
    fn create_f32(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        Ok(Box::new(TestOptimizer::<f32> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn create_f64(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        Ok(Box::new(PanicOnCloneOptimizer))
    }
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "panic-on-clone".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn validate_config(&self, _config: &OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn config_schema(&self) -> ConfigSchema {
        ConfigSchema {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            version: "1.0.0".to_string(),
        }
    }
    fn supports_type(&self, _datatype: &DataType) -> bool {
        true
    }
}

#[test]
fn f64_cache_insert_survives_a_panicking_clone_plugin() {
    // The first `create_optimizer` call for a name is always a cache
    // miss, so `PluginCache::insert` calls `clone_plugin` on the
    // freshly created instance purely to populate the cache. Before
    // this fix, that call ran completely unguarded while both
    // `factories`'s write lock and the cache's own mutex were held --
    // a panic there would unwind straight through both instead of
    // surfacing as `Err` the way every other third-party call in
    // `create_optimizer` already does.
    let registry = PluginRegistry::new(RegistryConfig::default());
    registry
        .register_plugin(PanicOnCloneFactory)
        .expect("registration should succeed");

    // Silence the default panic hook for the duration of the caught
    // panic so the test output stays clean (same pattern as
    // `panicking_factory_is_caught_and_registry_survives`).
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = registry.create_optimizer::<f64>("panic-on-clone", OptimizerConfig::default());
    std::panic::set_hook(prev_hook);

    assert!(
        result.is_err(),
        "a panicking clone_plugin during cache population must surface as Err, not unwind"
    );

    // The registry must remain fully usable: neither lock was left in
    // a state the poison-recovering helpers can't heal.
    registry
        .register_plugin(TestFactory)
        .expect("registry must still accept registrations after a caught clone panic");
    let opt = registry
        .create_optimizer::<f64>("test-sgd", OptimizerConfig::default())
        .expect("registry must still create optimizers after a caught clone panic");
    assert_eq!(opt.name(), "test-sgd");
}

/// Optimizer whose `clone_plugin` succeeds exactly once and panics on
/// every call after that. Isolates `PluginCache::get_or_record_miss`'s
/// own panic-catching (a cache *hit* whose clone panics) from
/// `f64_cache_insert_survives_a_panicking_clone_plugin`, which only
/// ever exercises the insert path's clone.
#[derive(Debug)]
struct PanicOnSecondCloneOptimizer {
    remaining_successful_clones: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl OptimizerPlugin<f64> for PanicOnSecondCloneOptimizer {
    fn step(&mut self, params: &Array1<f64>, _gradients: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(params.clone())
    }
    fn name(&self) -> &str {
        "panic-on-second-clone"
    }
    fn version(&self) -> &str {
        "1.0.0"
    }
    fn plugin_info(&self) -> PluginInfo {
        PluginInfo {
            name: "panic-on-second-clone".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::default()
    }
    fn initialize(&mut self, _paramshape: &[usize]) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn get_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn set_config(&mut self, _config: OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn get_state(&self) -> Result<OptimizerState> {
        Ok(OptimizerState::default())
    }
    fn set_state(&mut self, _state: OptimizerState) -> Result<()> {
        Ok(())
    }
    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<f64>> {
        let previous = self
            .remaining_successful_clones
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |n| n.checked_sub(1),
            )
            .unwrap_or(0);
        if previous == 0 {
            panic!("intentional panic: no successful clones remaining");
        }
        Box::new(PanicOnSecondCloneOptimizer {
            remaining_successful_clones: self.remaining_successful_clones.clone(),
        })
    }
}

#[derive(Debug)]
struct PanicOnSecondCloneFactory {
    remaining_successful_clones: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl PluginFactoryWrapper for PanicOnSecondCloneFactory {
    fn create_f32(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        Ok(Box::new(TestOptimizer::<f32> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn create_f64(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        Ok(Box::new(PanicOnSecondCloneOptimizer {
            remaining_successful_clones: self.remaining_successful_clones.clone(),
        }))
    }
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "panic-on-second-clone".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn validate_config(&self, _config: &OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn config_schema(&self) -> ConfigSchema {
        ConfigSchema {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            version: "1.0.0".to_string(),
        }
    }
    fn supports_type(&self, _datatype: &DataType) -> bool {
        true
    }
}

#[test]
fn f64_cache_hit_panic_evicts_the_poisoned_entry_instead_of_trapping_the_name() {
    // The first `create_optimizer` call is a miss: `PluginCache::insert`
    // clones the freshly created instance once to populate the cache --
    // that first clone succeeds (`remaining_successful_clones` starts
    // at 1). The *second* call for the same name+config is a cache
    // hit, so `get_or_record_miss` clones the cached entry -- that
    // second clone panics. Without eviction on a hit-path panic, this
    // entry's recency would already have been bumped (see
    // `get_or_record_miss`'s doc comment) and it would stay cached,
    // making every future call for this name `Err` forever.
    let registry = PluginRegistry::new(RegistryConfig {
        validate_on_registration: false,
        ..RegistryConfig::default()
    });
    let remaining = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1));
    registry
        .register_plugin(PanicOnSecondCloneFactory {
            remaining_successful_clones: remaining.clone(),
        })
        .expect("registration should succeed");

    let cfg = OptimizerConfig::default();
    let _first = registry
        .create_optimizer::<f64>("panic-on-second-clone", cfg.clone())
        .expect(
            "first (miss) call must succeed: only its cache-population clone runs, \
             and that one is the single successful clone",
        );
    assert_eq!(
        mutex_lock(&registry.cache).len(),
        1,
        "sanity: the instance was cached after the first call"
    );

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let second = registry.create_optimizer::<f64>("panic-on-second-clone", cfg);
    std::panic::set_hook(prev_hook);
    assert!(
        second.is_err(),
        "a panicking clone_plugin on a cache hit must surface as Err, not unwind"
    );
    assert_eq!(
        mutex_lock(&registry.cache).len(),
        0,
        "the poisoned entry must be evicted, not left permanently cached"
    );

    // The registry survives and is not stuck: register a well-behaved
    // factory under a different name and confirm normal operation
    // continues. (This deliberately checks recovery via a *different*
    // plugin, not by retrying "panic-on-second-clone" -- that name's
    // own factory would panic again on any further create_f64 call,
    // which is a property of the test double, not of the registry.)
    registry
        .register_plugin(TestFactory)
        .expect("registry must still accept registrations after the caught hit-path panic");
    let opt = registry
        .create_optimizer::<f64>("test-sgd", OptimizerConfig::default())
        .expect("registry must still create optimizers after the caught hit-path panic");
    assert_eq!(opt.name(), "test-sgd");
}

#[test]
fn static_plugin_roundtrip_creates_and_steps() {
    let registry = PluginRegistry::new(RegistryConfig::default());
    registry
        .register_plugin(TestFactory)
        .expect("registration should succeed");

    // f64 path.
    let mut opt = registry
        .create_optimizer::<f64>("test-sgd", OptimizerConfig::default())
        .expect("f64 optimizer should be creatable");
    opt.initialize(&[3]).expect("initialize");
    let params = Array1::from(vec![1.0_f64, 2.0, 3.0]);
    let grads = Array1::from(vec![0.5_f64, 0.5, 0.5]);
    let updated = opt.step(&params, &grads).expect("step should succeed");
    // lr = 0.1, so each coordinate moves by 0.05.
    assert!((updated[0] - 0.95).abs() < 1e-9);
    assert!((updated[1] - 1.95).abs() < 1e-9);
    assert!((updated[2] - 2.95).abs() < 1e-9);

    // f32 path exercises the other `Any` downcast branch.
    let opt32 = registry
        .create_optimizer::<f32>("test-sgd", OptimizerConfig::default())
        .expect("f32 optimizer should be creatable");
    assert_eq!(opt32.name(), "test-sgd");
}

#[test]
fn create_optimizer_reliably_updates_usage_statistics() {
    // F71 regression: `create_optimizer` previously dropped its read
    // lock and reacquired a *new* write lock purely to bump
    // `load_count`/`last_used`, leaving a window in which another
    // thread could unregister the plugin -- the statistics update
    // would then silently vanish (`factories.get_mut` finds nothing)
    // even though the just-created optimizer was handed back as `Ok`.
    // The fix folds the update into the single write guard already
    // held for status-check/validate/create, so this can no longer
    // race. This test cannot easily force the race itself, but it
    // pins the now-guaranteed behaviour: `load_count` increments
    // exactly once per successful `create_optimizer` call, every time.
    let registry = PluginRegistry::new(RegistryConfig::default());
    registry
        .register_plugin(TestFactory)
        .expect("registration should succeed");

    for expected_count in 1..=5usize {
        let _ = registry
            .create_optimizer::<f64>("test-sgd", OptimizerConfig::default())
            .expect("optimizer should be creatable");
        let factories = read_lock(&registry.factories);
        let registration = factories.get("test-sgd").expect("plugin must still exist");
        assert_eq!(
            registration.load_count, expected_count,
            "load_count must increment exactly once per successful create_optimizer call"
        );
        assert!(
            registration.last_used.is_some(),
            "last_used must be set after a successful create_optimizer call"
        );
    }
}

/// A second factory identical to `TestFactory` except it declares
/// `gpu_support: true`, used to prove `required_capabilities`
/// filtering actually discriminates between plugins rather than
/// silently matching everything (F72).
#[derive(Debug)]
struct GpuTestFactory;

impl PluginFactoryWrapper for GpuTestFactory {
    fn create_f32(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        Ok(Box::new(TestOptimizer::<f32> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn create_f64(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        Ok(Box::new(TestOptimizer::<f64> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "gpu-sgd".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            gpu_support: true,
            ..PluginCapabilities::default()
        }
    }
    fn validate_config(&self, _config: &OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn config_schema(&self) -> ConfigSchema {
        ConfigSchema {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            version: "1.0.0".to_string(),
        }
    }
    fn supports_type(&self, _datatype: &DataType) -> bool {
        true
    }
}

#[test]
fn search_plugins_filters_by_required_capabilities() {
    // F72 regression: `PluginQuery.required_capabilities` was declared
    // and never consulted by `matches_query`, so a search filtering on
    // it silently returned every plugin regardless of what it
    // supported.
    let registry = PluginRegistry::new(RegistryConfig::default());
    registry
        .register_plugin(TestFactory) // gpu_support: false (default)
        .expect("registration should succeed");
    registry
        .register_plugin(GpuTestFactory) // gpu_support: true
        .expect("registration should succeed");

    let all = registry.search_plugins(PluginQuery::default());
    assert_eq!(all.total_count, 2, "sanity: both plugins are registered");

    let gpu_only = registry.search_plugins(PluginQuery {
        required_capabilities: vec!["gpu_support".to_string()],
        ..PluginQuery::default()
    });
    assert_eq!(
        gpu_only.total_count, 1,
        "only the GPU-capable plugin should match"
    );
    assert_eq!(gpu_only.plugins[0].name, "gpu-sgd");

    // An unknown capability name must exclude everything rather than
    // matching everything -- `has_capability` returns `false` for
    // names it does not recognise.
    let unknown = registry.search_plugins(PluginQuery {
        required_capabilities: vec!["quantum_teleportation".to_string()],
        ..PluginQuery::default()
    });
    assert_eq!(unknown.total_count, 0);
}

#[test]
fn panicking_factory_is_caught_and_registry_survives() {
    // Disable creation-based validation so registration itself does not
    // trip the panic; we want the panic to happen inside `create_optimizer`
    // while the registry read lock is held.
    let config = RegistryConfig {
        validate_on_registration: false,
        ..RegistryConfig::default()
    };
    let registry = PluginRegistry::new(config);
    registry
        .register_plugin(PanicFactory)
        .expect("registration should succeed");

    // Silence the default panic hook for the duration of the caught panic
    // so the test output stays clean.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = registry.create_optimizer::<f64>("panic-plugin", OptimizerConfig::default());
    std::panic::set_hook(prev_hook);

    assert!(
        result.is_err(),
        "a panicking factory must surface as Err, not unwind out of create_optimizer"
    );

    // The registry must remain fully usable: the lock was not poisoned
    // (catch_unwind) or, if it were, the recovery helpers heal it.
    registry
        .register_plugin(TestFactory)
        .expect("registry must still accept registrations after a caught panic");
    let opt = registry
        .create_optimizer::<f64>("test-sgd", OptimizerConfig::default())
        .expect("registry must still create optimizers after a caught panic");
    assert_eq!(opt.name(), "test-sgd");
}

#[test]
fn lock_helpers_recover_from_poisoning() {
    use std::sync::{Arc, Mutex, RwLock};

    // Poison an RwLock by panicking while holding its write guard.
    let rw = Arc::new(RwLock::new(5_i32));
    let rw_clone = Arc::clone(&rw);
    let _ = std::thread::spawn(move || {
        let _guard = rw_clone.write().unwrap_or_else(|e| e.into_inner());
        panic!("poison the rwlock");
    })
    .join();
    assert!(rw.read().is_err(), "precondition: the RwLock is poisoned");
    // Helpers hand back a usable guard rather than panicking.
    assert_eq!(*read_lock(&rw), 5);
    *write_lock(&rw) = 7;
    assert_eq!(*read_lock(&rw), 7);

    // Poison a Mutex the same way.
    let mx = Arc::new(Mutex::new(1_i32));
    let mx_clone = Arc::clone(&mx);
    let _ = std::thread::spawn(move || {
        let _guard = mx_clone.lock().unwrap_or_else(|e| e.into_inner());
        panic!("poison the mutex");
    })
    .join();
    assert!(mx.lock().is_err(), "precondition: the Mutex is poisoned");
    *mutex_lock(&mx) += 10;
    assert_eq!(*mutex_lock(&mx), 11);
}

/// Identical to `TestFactory` except the registered plugin name is
/// configurable, so eviction tests can register several distinct cache
/// keys without writing a separate factory type per name.
#[derive(Debug)]
struct NamedTestFactory {
    plugin_name: &'static str,
}

impl PluginFactoryWrapper for NamedTestFactory {
    fn create_f32(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        Ok(Box::new(TestOptimizer::<f32> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn create_f64(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        Ok(Box::new(TestOptimizer::<f64> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: self.plugin_name.to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn validate_config(&self, _config: &OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn config_schema(&self) -> ConfigSchema {
        ConfigSchema {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            version: "1.0.0".to_string(),
        }
    }
    fn supports_type(&self, _datatype: &DataType) -> bool {
        true
    }
}

// F70 regression: `PluginCache` was fully dead code -- `instances` was
// never inserted into outside `clear_cache`, and `enable_caching` /
// `max_cache_size` were declared on `RegistryConfig` but never read
// anywhere. The following tests pin the real behavior now wired into
// `create_optimizer`'s f64 branch.

/// A factory that counts how many times `create_f64` actually runs, so
/// a cache-hit test can prove the factory was genuinely not
/// re-invoked -- rather than only asserting on the cache's own
/// hit/miss counters, which are set by the same code under test.
#[derive(Debug)]
struct CountingFactory {
    create_f64_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl PluginFactoryWrapper for CountingFactory {
    fn create_f32(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        Ok(Box::new(TestOptimizer::<f32> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn create_f64(&self, _config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        self.create_f64_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Box::new(TestOptimizer::<f64> {
            lr: 0.1,
            initialized: false,
        }))
    }
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "counting-sgd".to_string(),
            version: "1.0.0".to_string(),
            ..PluginInfo::default()
        }
    }
    fn validate_config(&self, _config: &OptimizerConfig) -> Result<()> {
        Ok(())
    }
    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig::default()
    }
    fn config_schema(&self) -> ConfigSchema {
        ConfigSchema {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            version: "1.0.0".to_string(),
        }
    }
    fn supports_type(&self, _datatype: &DataType) -> bool {
        true
    }
}

#[test]
fn f64_cache_hit_avoids_recreation_and_bumps_stats() {
    let create_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // `validate_on_registration` (the default) itself calls
    // `create_f64` once during `register_plugin` to sanity-check the
    // factory -- disabled here so the counter only reflects calls made
    // by `create_optimizer` itself, which is what this test is about.
    let registry = PluginRegistry::new(RegistryConfig {
        validate_on_registration: false,
        ..RegistryConfig::default()
    });
    registry
        .register_plugin(CountingFactory {
            create_f64_calls: create_calls.clone(),
        })
        .expect("registration should succeed");

    let cfg = OptimizerConfig::default();
    let _first = registry
        .create_optimizer::<f64>("counting-sgd", cfg.clone())
        .expect("first create should succeed");
    assert_eq!(
        create_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the first call for a name must reach the factory"
    );
    let after_first = registry.get_cache_stats();
    assert_eq!(
        after_first.misses, 1,
        "first call for a name is always a miss"
    );
    assert_eq!(after_first.hits, 0);

    let _second = registry
        .create_optimizer::<f64>("counting-sgd", cfg)
        .expect("second create should succeed");
    assert_eq!(
        create_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "a cache hit must NOT re-invoke the factory's create_f64 -- this is the \
         actual 'avoids recreation' claim, checked independently of the cache's \
         own hit/miss counters"
    );
    let after_second = registry.get_cache_stats();
    assert_eq!(
        after_second.hits, 1,
        "same name + identical config must be a cache hit"
    );
    assert_eq!(after_second.misses, 1, "miss count must not grow on a hit");
}

#[test]
fn f64_cache_treats_a_config_change_as_a_miss_not_a_stale_hit() {
    // A cache keyed on plugin name alone would silently hand back an
    // instance built from a *different* config than the one just
    // requested -- exactly the kind of fabricated-correctness this
    // crate's stub-removal pass exists to eliminate. A config change
    // for the same name must be a miss (rebuild), never a hit.
    let registry = PluginRegistry::new(RegistryConfig::default());
    registry
        .register_plugin(TestFactory)
        .expect("registration should succeed");

    let cfg_a = OptimizerConfig {
        learning_rate: 0.01,
        ..OptimizerConfig::default()
    };
    let cfg_b = OptimizerConfig {
        learning_rate: 0.02,
        ..OptimizerConfig::default()
    };

    let _ = registry
        .create_optimizer::<f64>("test-sgd", cfg_a)
        .expect("first create should succeed");
    let _ = registry
        .create_optimizer::<f64>("test-sgd", cfg_b)
        .expect("second create with a different config should succeed");

    let stats = registry.get_cache_stats();
    assert_eq!(
        stats.misses, 2,
        "a config change for the same name must never be served from a stale cache entry"
    );
    assert_eq!(stats.hits, 0);
}

#[test]
fn f64_caching_disabled_bypasses_the_cache_entirely() {
    let registry = PluginRegistry::new(RegistryConfig {
        enable_caching: false,
        ..RegistryConfig::default()
    });
    registry
        .register_plugin(TestFactory)
        .expect("registration should succeed");

    let cfg = OptimizerConfig::default();
    let _ = registry
        .create_optimizer::<f64>("test-sgd", cfg.clone())
        .expect("first create should succeed");
    let _ = registry
        .create_optimizer::<f64>("test-sgd", cfg)
        .expect("second create should succeed");

    let stats = registry.get_cache_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(
        stats.misses, 0,
        "PluginCache must not be touched at all when enable_caching is false"
    );
    assert_eq!(mutex_lock(&registry.cache).len(), 0);
}

#[test]
fn f32_creation_never_touches_the_f64_only_cache() {
    // `PluginCache` is monomorphized to `Box<dyn OptimizerPlugin<f64>>`
    // (see its doc comment) -- an f32 request must bypass it entirely,
    // not silently miss-and-skip-insert every time.
    let registry = PluginRegistry::new(RegistryConfig::default());
    registry
        .register_plugin(TestFactory)
        .expect("registration should succeed");

    let cfg = OptimizerConfig::default();
    let _ = registry
        .create_optimizer::<f32>("test-sgd", cfg.clone())
        .expect("first create should succeed");
    let _ = registry
        .create_optimizer::<f32>("test-sgd", cfg)
        .expect("second create should succeed");

    let stats = registry.get_cache_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(mutex_lock(&registry.cache).len(), 0);
}

#[test]
fn f64_cache_evicts_lru_entry_once_max_size_is_exceeded() {
    let registry = PluginRegistry::new(RegistryConfig {
        max_cache_size: 2,
        ..RegistryConfig::default()
    });
    registry
        .register_plugin(NamedTestFactory {
            plugin_name: "cache-a",
        })
        .expect("register a");
    registry
        .register_plugin(NamedTestFactory {
            plugin_name: "cache-b",
        })
        .expect("register b");
    registry
        .register_plugin(NamedTestFactory {
            plugin_name: "cache-c",
        })
        .expect("register c");

    let cfg = OptimizerConfig::default();

    // Fill the cache to its limit: "cache-a" then "cache-b", both
    // fresh misses.
    let _ = registry
        .create_optimizer::<f64>("cache-a", cfg.clone())
        .expect("create a");
    let _ = registry
        .create_optimizer::<f64>("cache-b", cfg.clone())
        .expect("create b");
    assert_eq!(mutex_lock(&registry.cache).len(), 2);

    // A third distinct name must evict the least-recently-used entry
    // ("cache-a", never touched again since its own insert) to stay at
    // max_cache_size, not grow unbounded.
    let _ = registry
        .create_optimizer::<f64>("cache-c", cfg.clone())
        .expect("create c");
    assert_eq!(
        mutex_lock(&registry.cache).len(),
        2,
        "cache must never exceed max_cache_size"
    );
    assert_eq!(registry.get_cache_stats().evictions, 1);

    // The behavior that actually matters is that "cache-a" was
    // REMOVED, not merely that an `evictions` counter ticked (the old
    // dead `PluginCache` could never have made even that much true).
    // Re-requesting it now must be a genuine miss -- a surviving stale
    // entry would instead register as a hit.
    let misses_before = registry.get_cache_stats().misses;
    let _ = registry
        .create_optimizer::<f64>("cache-a", cfg)
        .expect("recreate a after eviction");
    assert_eq!(
        registry.get_cache_stats().misses,
        misses_before + 1,
        "cache-a must have actually been evicted from `instances`, not just counted as evicted"
    );
}
