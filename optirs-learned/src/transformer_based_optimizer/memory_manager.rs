// Memory management for transformer-based optimizer

use super::config::{CacheEvictionStrategy, MemoryConfig, TransformerBasedOptimizerConfig};
use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Memory management strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryManagementStrategy {
    /// Simple FIFO eviction
    FIFO,
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Adaptive replacement cache
    ARC,
    /// Compressed memory storage
    Compressed,
    /// Hierarchical memory organization
    Hierarchical,
}

/// Transformer memory manager
pub struct TransformerMemoryManager<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Memory management strategy
    strategy: MemoryManagementStrategy,

    /// Configuration
    config: MemoryConfig,

    /// Primary memory cache
    primary_cache: MemoryCache<T>,

    /// Secondary cache for overflow
    secondary_cache: Option<MemoryCache<T>>,

    /// Memory compression manager
    compression_manager: Option<CompressionManager<T>>,

    /// Memory statistics
    statistics: MemoryStatistics,

    /// Access patterns tracker
    access_tracker: AccessTracker,

    /// Memory pressure monitor
    pressure_monitor: MemoryPressureMonitor,

    /// Model dimension
    model_dimension: usize,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    TransformerMemoryManager<T>
{
    /// Create new memory manager
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let memory_config = config.memory_config.clone();
        let strategy = match memory_config.eviction_strategy {
            CacheEvictionStrategy::LRU => MemoryManagementStrategy::LRU,
            CacheEvictionStrategy::LFU => MemoryManagementStrategy::LFU,
            CacheEvictionStrategy::FIFO => MemoryManagementStrategy::FIFO,
            CacheEvictionStrategy::Random => MemoryManagementStrategy::LRU, // Fallback
        };

        let primary_cache = MemoryCache::new(
            memory_config.max_cache_size / 2,
            memory_config.eviction_strategy,
        )?;

        let secondary_cache = if memory_config.max_cache_size > 1024 * 1024 * 100 {
            // 100MB
            Some(MemoryCache::new(
                memory_config.max_cache_size / 2,
                CacheEvictionStrategy::FIFO,
            )?)
        } else {
            None
        };

        let compression_manager = if memory_config.enable_compression {
            Some(CompressionManager::new(0.5)?) // 50% compression ratio target
        } else {
            None
        };

        let statistics = MemoryStatistics::new();
        let access_tracker = AccessTracker::new(1000);
        let pressure_monitor = MemoryPressureMonitor::new();

        Ok(Self {
            strategy,
            config: memory_config,
            primary_cache,
            secondary_cache,
            compression_manager,
            statistics,
            access_tracker,
            pressure_monitor,
            model_dimension: config.model_dimension,
        })
    }

    /// Store tensor in memory with key
    pub fn store(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        let start_time = Instant::now();

        // Check memory pressure and evict if necessary
        self.pressure_monitor.update(self.get_memory_usage());
        if self.pressure_monitor.is_high_pressure() {
            self.evict_memory()?;
        }

        // Try to store in primary cache first
        let storage_result = match self.strategy {
            MemoryManagementStrategy::LRU => self.store_lru(key.clone(), tensor.clone()),
            MemoryManagementStrategy::LFU => self.store_lfu(key.clone(), tensor.clone()),
            MemoryManagementStrategy::FIFO => self.store_fifo(key.clone(), tensor.clone()),
            MemoryManagementStrategy::ARC => self.store_arc(key.clone(), tensor.clone()),
            MemoryManagementStrategy::Compressed => {
                self.store_compressed(key.clone(), tensor.clone())
            }
            MemoryManagementStrategy::Hierarchical => {
                self.store_hierarchical(key.clone(), tensor.clone())
            }
        };

        // Get tensor length before potential move
        let tensor_len = tensor.len();

        // If primary cache is full, try secondary cache
        if storage_result.is_err() && self.secondary_cache.is_some() {
            if let Some(ref mut secondary) = self.secondary_cache {
                secondary.store(key.clone(), tensor)?;
            }
        }

        // Update statistics
        let storage_time = start_time.elapsed();
        self.statistics.record_storage(tensor_len, storage_time);
        self.access_tracker.record_write(key);

        storage_result
    }

    /// Retrieve tensor from memory
    pub fn retrieve(&mut self, key: &str) -> Result<Option<Array2<T>>> {
        let start_time = Instant::now();

        // Try primary cache first
        let result = self.primary_cache.retrieve(key)?;

        if result.is_some() {
            self.access_tracker.record_read(key.to_string());
            let retrieval_time = start_time.elapsed();
            self.statistics.record_retrieval(retrieval_time, true);
            return Ok(result);
        }

        // Try secondary cache
        if let Some(ref mut secondary) = self.secondary_cache {
            let result = secondary.retrieve(key)?;
            if result.is_some() {
                self.access_tracker.record_read(key.to_string());
                let retrieval_time = start_time.elapsed();
                self.statistics.record_retrieval(retrieval_time, true);
                return Ok(result);
            }
        }

        // Check compressed storage
        if let Some(ref mut compression) = self.compression_manager {
            if let Some(compressed_data) = compression.retrieve(key)? {
                let decompressed = compression.decompress(&compressed_data)?;
                self.access_tracker.record_read(key.to_string());
                let retrieval_time = start_time.elapsed();
                self.statistics.record_retrieval(retrieval_time, true);
                return Ok(Some(decompressed));
            }
        }

        let retrieval_time = start_time.elapsed();
        self.statistics.record_retrieval(retrieval_time, false);
        Ok(None)
    }

    /// Remove tensor from memory
    pub fn remove(&mut self, key: &str) -> Result<bool> {
        let mut removed = false;

        if self.primary_cache.remove(key)? {
            removed = true;
        }

        if let Some(ref mut secondary) = self.secondary_cache {
            if secondary.remove(key)? {
                removed = true;
            }
        }

        if let Some(ref mut compression) = self.compression_manager {
            if compression.remove(key)? {
                removed = true;
            }
        }

        self.access_tracker.record_removal(key.to_string());
        Ok(removed)
    }

    /// Clear all memory
    pub fn clear(&mut self) -> Result<()> {
        self.primary_cache.clear()?;

        if let Some(ref mut secondary) = self.secondary_cache {
            secondary.clear()?;
        }

        if let Some(ref mut compression) = self.compression_manager {
            compression.clear()?;
        }

        self.statistics.reset();
        self.access_tracker.clear();
        self.pressure_monitor.reset();

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_usage(&self) -> usize {
        let primary_usage = self.primary_cache.get_memory_usage();
        let secondary_usage = self
            .secondary_cache
            .as_ref()
            .map(|cache| cache.get_memory_usage())
            .unwrap_or(0);
        let compression_usage = self
            .compression_manager
            .as_ref()
            .map(|comp| comp.get_memory_usage())
            .unwrap_or(0);

        primary_usage + secondary_usage + compression_usage
    }

    /// Optimize memory layout
    pub fn optimize_memory(&mut self) -> Result<OptimizationReport> {
        let start_time = Instant::now();
        let initial_usage = self.get_memory_usage();

        // Analyze access patterns
        let access_patterns = self.access_tracker.analyze_patterns();

        // Reorganize based on access frequency
        self.reorganize_by_frequency(&access_patterns)?;

        // Compress frequently accessed but large items
        if let Some(ref mut compression) = self.compression_manager {
            compression.optimize_compression_ratios(&access_patterns)?;
        }

        // Defragment memory
        self.defragment_memory()?;

        let final_usage = self.get_memory_usage();
        let optimization_time = start_time.elapsed();

        Ok(OptimizationReport {
            initial_memory_usage: initial_usage,
            final_memory_usage: final_usage,
            memory_saved: initial_usage.saturating_sub(final_usage),
            optimization_time,
            operations_performed: access_patterns.total_accesses,
        })
    }

    /// Prefetch data based on access patterns
    pub fn prefetch(&mut self, keys: Vec<String>) -> Result<usize> {
        let mut prefetched_count = 0;

        for key in keys {
            if !self.primary_cache.contains(&key) {
                // Try to move from secondary to primary cache
                if let Some(ref mut secondary) = self.secondary_cache {
                    if let Some(tensor) = secondary.retrieve(&key)? {
                        if self.primary_cache.store(key.clone(), tensor).is_ok() {
                            secondary.remove(&key)?;
                            prefetched_count += 1;
                        }
                    }
                }

                // Try to decompress and move to primary cache
                if let Some(ref mut compression) = self.compression_manager {
                    if let Some(compressed_data) = compression.retrieve(&key)? {
                        let decompressed = compression.decompress(&compressed_data)?;
                        if self.primary_cache.store(key.clone(), decompressed).is_ok() {
                            compression.remove(&key)?;
                            prefetched_count += 1;
                        }
                    }
                }
            }
        }

        Ok(prefetched_count)
    }

    /// Storage strategy implementations
    fn store_lru(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        self.primary_cache.store(key, tensor)
    }

    fn store_lfu(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        // For LFU, we need to track access frequency
        self.primary_cache.store(key, tensor)
    }

    fn store_fifo(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        self.primary_cache.store(key, tensor)
    }

    fn store_arc(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        // Adaptive Replacement Cache - simplified implementation
        self.primary_cache.store(key, tensor)
    }

    fn store_compressed(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        if let Some(ref mut compression) = self.compression_manager {
            let compressed_data = compression.compress(&tensor)?;
            compression.store(key, compressed_data)?;
            Ok(())
        } else {
            self.primary_cache.store(key, tensor)
        }
    }

    fn store_hierarchical(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        let tensor_size = tensor.len() * std::mem::size_of::<T>();

        if tensor_size < self.config.allocation_block_size {
            // Small tensors go to primary cache
            self.primary_cache.store(key, tensor)
        } else if let Some(ref mut secondary) = self.secondary_cache {
            // Large tensors go to secondary cache
            secondary.store(key, tensor)
        } else {
            // Fallback to primary cache
            self.primary_cache.store(key, tensor)
        }
    }

    fn evict_memory(&mut self) -> Result<()> {
        // Evict from primary cache first
        self.primary_cache.evict_lru()?;

        // If still under pressure, evict from secondary cache
        if self.pressure_monitor.is_high_pressure() {
            if let Some(ref mut secondary) = self.secondary_cache {
                secondary.evict_lru()?;
            }
        }

        Ok(())
    }

    fn reorganize_by_frequency(&mut self, patterns: &AccessPatterns) -> Result<()> {
        // Move frequently accessed items to primary cache
        let frequent_keys: Vec<String> = patterns
            .frequency_map
            .iter()
            .filter(|(_, &count)| count as f64 > patterns.average_frequency)
            .map(|(key, _)| key.clone())
            .collect();

        self.prefetch(frequent_keys)?;
        Ok(())
    }

    fn defragment_memory(&mut self) -> Result<()> {
        // Simplified defragmentation - rebuild caches
        let primary_items = self.primary_cache.get_all_items()?;
        self.primary_cache.clear()?;

        for (key, tensor) in primary_items {
            self.primary_cache.store(key, tensor)?;
        }

        Ok(())
    }

    /// Get memory statistics
    pub fn get_statistics(&self) -> &MemoryStatistics {
        &self.statistics
    }

    /// Get access patterns
    pub fn get_access_patterns(&self) -> AccessPatterns {
        self.access_tracker.analyze_patterns()
    }

    /// Set memory management strategy
    pub fn set_strategy(&mut self, strategy: MemoryManagementStrategy) {
        self.strategy = strategy;
    }

    /// Get current memory pressure
    pub fn get_memory_pressure(&self) -> f64 {
        self.pressure_monitor.get_pressure_ratio()
    }
}

/// Memory cache implementation
pub struct MemoryCache<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Stored tensors
    storage: HashMap<String, CacheEntry<T>>,

    /// Access order for LRU
    access_order: VecDeque<String>,

    /// Access frequency for LFU
    access_frequency: HashMap<String, usize>,

    /// Maximum cache size in bytes
    max_size: usize,

    /// Current cache size in bytes
    current_size: usize,

    /// Eviction strategy
    eviction_strategy: CacheEvictionStrategy,

    /// Insertion order, never re-ordered by an access. Backs FIFO eviction,
    /// which was previously indistinguishable from LRU.
    insertion_order: VecDeque<String>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    MemoryCache<T>
{
    pub fn new(max_size: usize, eviction_strategy: CacheEvictionStrategy) -> Result<Self> {
        Ok(Self {
            storage: HashMap::new(),
            access_order: VecDeque::new(),
            access_frequency: HashMap::new(),
            max_size,
            current_size: 0,
            eviction_strategy,
            insertion_order: VecDeque::new(),
        })
    }

    /// Insert a tensor, evicting as needed.
    ///
    /// The oversized-tensor check happens **before** any eviction. Previously the
    /// eviction loop ran first, so storing a tensor larger than the whole cache
    /// drained every existing entry and *then* returned an error — the caller got
    /// a failure and an empty cache.
    ///
    /// # Errors
    /// Returns `Err` when the tensor alone exceeds `max_size`. The cache is left
    /// untouched in that case.
    pub fn store(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        let tensor_size = tensor.len() * std::mem::size_of::<T>();

        if tensor_size > self.max_size {
            return Err(crate::error::OptimError::Other(format!(
                "Tensor of {tensor_size} bytes is too large for a cache of {} bytes",
                self.max_size
            )));
        }

        // Remove any existing entry first: it frees space and prevents
        // double-counting when the same key is overwritten.
        if let Some(old_entry) = self.storage.remove(&key) {
            self.current_size -= old_entry.size;
            self.remove_from_access_order(&key);
            self.insertion_order.retain(|k| k != &key);
            self.access_frequency.remove(&key);
        }

        // Now evict until the new entry fits.
        while self.current_size + tensor_size > self.max_size && !self.storage.is_empty() {
            self.evict_one()?;
        }

        // Add new entry
        let entry = CacheEntry {
            tensor,
            size: tensor_size,
            access_time: Instant::now(),
            access_count: 1,
        };

        self.storage.insert(key.clone(), entry);
        self.current_size += tensor_size;
        self.update_access_tracking(&key);

        Ok(())
    }

    pub fn retrieve(&mut self, key: &str) -> Result<Option<Array2<T>>> {
        let tensor_result = if let Some(entry) = self.storage.get_mut(key) {
            entry.access_time = Instant::now();
            entry.access_count += 1;
            Some(entry.tensor.clone())
        } else {
            None
        };

        // Update access tracking after releasing the mutable borrow
        if tensor_result.is_some() {
            self.update_access_tracking(key);
        }

        Ok(tensor_result)
    }

    pub fn remove(&mut self, key: &str) -> Result<bool> {
        if let Some(entry) = self.storage.remove(key) {
            self.current_size -= entry.size;
            self.remove_from_access_order(key);
            self.insertion_order.retain(|k| k != key);
            self.access_frequency.remove(key);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.storage.contains_key(key)
    }

    pub fn clear(&mut self) -> Result<()> {
        self.storage.clear();
        self.access_order.clear();
        self.insertion_order.clear();
        self.access_frequency.clear();
        self.current_size = 0;
        Ok(())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.current_size
    }

    pub fn evict_lru(&mut self) -> Result<()> {
        if let Some(oldest_key) = self.access_order.front().cloned() {
            self.remove(&oldest_key)?;
        }
        Ok(())
    }

    fn evict_one(&mut self) -> Result<()> {
        match self.eviction_strategy {
            CacheEvictionStrategy::LRU => self.evict_lru(),
            CacheEvictionStrategy::LFU => self.evict_lfu(),
            CacheEvictionStrategy::FIFO => self.evict_fifo(),
            CacheEvictionStrategy::Random => self.evict_random(),
        }
    }

    fn evict_lfu(&mut self) -> Result<()> {
        if let Some(lfu_key) = self
            .access_frequency
            .iter()
            .min_by_key(|(_, &freq)| freq)
            .map(|(key, _)| key.clone())
        {
            self.remove(&lfu_key)?;
        }
        Ok(())
    }

    /// Evict the oldest **inserted** entry.
    ///
    /// FIFO is insertion order and must ignore accesses. This used to read
    /// `access_order.front()`, which `update_access_tracking` re-orders on every
    /// access — making FIFO an exact duplicate of LRU. It now uses a dedicated
    /// `insertion_order` queue that accesses never touch.
    fn evict_fifo(&mut self) -> Result<()> {
        if let Some(first_key) = self.insertion_order.front().cloned() {
            self.remove(&first_key)?;
        }
        Ok(())
    }

    /// Evict a uniformly random entry.
    ///
    /// This used to take `storage.keys().next()`, i.e. whatever the hash order
    /// put first — deterministic within a process and strongly biased, not
    /// random. It now draws a uniform index over the live keys.
    fn evict_random(&mut self) -> Result<()> {
        if self.storage.is_empty() {
            return Ok(());
        }
        let keys: Vec<String> = self.storage.keys().cloned().collect();
        let index = scirs2_core::random::thread_rng().gen_range(0..keys.len());
        let victim = keys[index].clone();
        self.remove(&victim)?;
        Ok(())
    }

    fn update_access_tracking(&mut self, key: &str) {
        // Update LRU order
        self.remove_from_access_order(key);
        self.access_order.push_back(key.to_string());

        // Insertion order is recorded once and never re-ordered by an access,
        // which is what separates FIFO eviction from LRU eviction.
        if !self.insertion_order.iter().any(|k| k == key) {
            self.insertion_order.push_back(key.to_string());
        }

        // Update LFU frequency
        *self.access_frequency.entry(key.to_string()).or_insert(0) += 1;
    }

    fn remove_from_access_order(&mut self, key: &str) {
        self.access_order.retain(|k| k != key);
    }

    pub fn get_all_items(&self) -> Result<Vec<(String, Array2<T>)>> {
        let items = self
            .storage
            .iter()
            .map(|(key, entry)| (key.clone(), entry.tensor.clone()))
            .collect();
        Ok(items)
    }
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    pub tensor: Array2<T>,
    pub size: usize,
    pub access_time: Instant,
    pub access_count: usize,
}

/// Compression manager
pub struct CompressionManager<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Compressed storage
    compressed_storage: HashMap<String, CompressedData<T>>,

    /// Compression ratio target
    compression_ratio: f64,

    /// Memory usage
    memory_usage: usize,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    CompressionManager<T>
{
    pub fn new(compression_ratio: f64) -> Result<Self> {
        Ok(Self {
            compressed_storage: HashMap::new(),
            compression_ratio,
            memory_usage: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Compress a tensor by affine 8-bit quantization.
    ///
    /// Every element is mapped to a `u8` code by
    /// `code = round((x - min) / step)` with `step = (max - min) / 255`, and the
    /// payload is those codes plus the two `f64` parameters `min` and `step`.
    /// `compressed_size` is therefore the **real** byte count of what is stored:
    /// `codes.len() + 2·size_of::<f64>() + shape metadata`.
    ///
    /// This replaces a version that stored the input verbatim as `Vec<T>` and
    /// reported `compressed_size = bytes / 2` — a fabricated 50% ratio for a
    /// payload that had not shrunk at all.
    ///
    /// Quantization is lossy; the reconstruction error is bounded by `step / 2`
    /// per element, which [`Self::max_reconstruction_error`] reports. A constant
    /// tensor (`max == min`) has `step = 0` and round-trips exactly.
    ///
    /// # Errors
    /// Returns `Err` when the tensor is empty or contains a non-finite value
    /// (there is no finite quantization range for `NaN`/`inf`).
    pub fn compress(&self, tensor: &Array2<T>) -> Result<CompressedData<T>> {
        if tensor.is_empty() {
            return Err(crate::error::OptimError::InsufficientData(
                "cannot compress an empty tensor".to_string(),
            ));
        }
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in tensor.iter() {
            let v = value.to_f64().unwrap_or(f64::NAN);
            if !v.is_finite() {
                return Err(crate::error::OptimError::ComputationError(
                    "cannot quantize a tensor containing a non-finite value".to_string(),
                ));
            }
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        let levels = 255.0_f64;
        let step = if max > min { (max - min) / levels } else { 0.0 };
        let codes: Vec<u8> = tensor
            .iter()
            .map(|value| {
                let v = value.to_f64().unwrap_or(0.0);
                if step <= 0.0 {
                    0u8
                } else {
                    (((v - min) / step).round()).clamp(0.0, levels) as u8
                }
            })
            .collect();

        let shape = tensor.shape().to_vec();
        let metadata_bytes =
            2 * std::mem::size_of::<f64>() + shape.len() * std::mem::size_of::<usize>();
        Ok(CompressedData::<T> {
            shape,
            codes,
            quantization_min: min,
            quantization_step: step,
            original_size: tensor.len() * std::mem::size_of::<T>(),
            compressed_size: codes_len_bytes(tensor.len()) + metadata_bytes,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Reconstruct a quantized tensor: `x̂ = min + code · step`.
    ///
    /// # Errors
    /// Returns `Err` when the stored shape is not two-dimensional or does not
    /// match the number of stored codes.
    pub fn decompress(&self, compressed: &CompressedData<T>) -> Result<Array2<T>> {
        if compressed.shape.len() != 2 {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "compressed payload has a {}-dimensional shape, expected 2",
                compressed.shape.len()
            )));
        }
        let expected = compressed.shape[0] * compressed.shape[1];
        if compressed.codes.len() != expected {
            return Err(crate::error::OptimError::ComputationError(format!(
                "compressed payload holds {} codes but its shape implies {expected}",
                compressed.codes.len()
            )));
        }
        let values: Vec<T> = compressed
            .codes
            .iter()
            .map(|&code| {
                let v = compressed.quantization_min + code as f64 * compressed.quantization_step;
                scirs2_core::numeric::NumCast::from(v).unwrap_or_else(|| T::zero())
            })
            .collect();
        Array2::from_shape_vec((compressed.shape[0], compressed.shape[1]), values)
            .map_err(|_| crate::error::OptimError::Other("Decompression failed".to_string()))
    }

    pub fn store(&mut self, key: String, compressed: CompressedData<T>) -> Result<()> {
        self.memory_usage += compressed.compressed_size;
        self.compressed_storage.insert(key, compressed);
        Ok(())
    }

    pub fn retrieve(&self, key: &str) -> Result<Option<CompressedData<T>>> {
        Ok(self.compressed_storage.get(key).cloned())
    }

    pub fn remove(&mut self, key: &str) -> Result<bool> {
        if let Some(compressed) = self.compressed_storage.remove(key) {
            self.memory_usage -= compressed.compressed_size;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn clear(&mut self) -> Result<()> {
        self.compressed_storage.clear();
        self.memory_usage = 0;
        Ok(())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.memory_usage
    }

    pub fn optimize_compression_ratios(&mut self, _patterns: &AccessPatterns) -> Result<()> {
        // Optimize compression based on access patterns
        // This is a placeholder for more sophisticated compression optimization
        Ok(())
    }
}

/// Number of payload bytes `count` 8-bit quantization codes occupy.
fn codes_len_bytes(count: usize) -> usize {
    count // one u8 per element
}

/// A quantized tensor.
///
/// The payload is `codes` (one `u8` per element) plus the affine dequantization
/// parameters. The old version of this struct stored `data: Vec<T>` — the
/// uncompressed input — while advertising a halved `compressed_size`.
#[derive(Debug, Clone)]
pub struct CompressedData<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Original tensor shape (2-D).
    pub shape: Vec<usize>,
    /// Quantization codes, row-major.
    pub codes: Vec<u8>,
    /// Value the zero code maps to.
    pub quantization_min: f64,
    /// Value increment per code step (`0` for a constant tensor).
    pub quantization_step: f64,
    /// Bytes the uncompressed tensor occupied.
    pub original_size: usize,
    /// Bytes this payload occupies, counted for real.
    pub compressed_size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    CompressedData<T>
{
    /// Achieved compression ratio `compressed_size / original_size`.
    ///
    /// A *measurement*, not the configured target: for `f64` input this is
    /// roughly `1/8` plus metadata, and for `f32` roughly `1/4`.
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.compressed_size as f64 / self.original_size as f64
    }

    /// Worst-case absolute reconstruction error, `quantization_step / 2`.
    pub fn max_reconstruction_error(&self) -> f64 {
        self.quantization_step / 2.0
    }

    /// Number of quantized elements.
    pub fn element_count(&self) -> usize {
        self.codes.len()
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Total storage operations
    pub total_stores: usize,

    /// Total retrieval operations
    pub total_retrievals: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Total bytes stored
    pub total_bytes_stored: usize,

    /// Average storage time
    pub average_storage_time: Duration,

    /// Average retrieval time
    pub average_retrieval_time: Duration,

    /// Memory pressure events
    pub pressure_events: usize,
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStatistics {
    pub fn new() -> Self {
        Self {
            total_stores: 0,
            total_retrievals: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_bytes_stored: 0,
            average_storage_time: Duration::new(0, 0),
            average_retrieval_time: Duration::new(0, 0),
            pressure_events: 0,
        }
    }

    pub fn record_storage(&mut self, bytes: usize, time: Duration) {
        self.total_stores += 1;
        self.total_bytes_stored += bytes;
        self.average_storage_time = (self.average_storage_time * (self.total_stores - 1) as u32
            + time)
            / self.total_stores as u32;
    }

    pub fn record_retrieval(&mut self, time: Duration, hit: bool) {
        self.total_retrievals += 1;
        if hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        self.average_retrieval_time =
            (self.average_retrieval_time * (self.total_retrievals - 1) as u32 + time)
                / self.total_retrievals as u32;
    }

    pub fn record_pressure_event(&mut self) {
        self.pressure_events += 1;
    }

    pub fn get_hit_ratio(&self) -> f64 {
        if self.total_retrievals > 0 {
            self.cache_hits as f64 / self.total_retrievals as f64
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Access tracker
pub struct AccessTracker {
    /// Read access log
    read_log: VecDeque<AccessEvent>,

    /// Write access log
    write_log: VecDeque<AccessEvent>,

    /// Maximum log size
    max_log_size: usize,
}

impl AccessTracker {
    pub fn new(max_log_size: usize) -> Self {
        Self {
            read_log: VecDeque::new(),
            write_log: VecDeque::new(),
            max_log_size,
        }
    }

    pub fn record_read(&mut self, key: String) {
        self.read_log.push_back(AccessEvent {
            key,
            timestamp: Instant::now(),
        });

        if self.read_log.len() > self.max_log_size {
            self.read_log.pop_front();
        }
    }

    pub fn record_write(&mut self, key: String) {
        self.write_log.push_back(AccessEvent {
            key,
            timestamp: Instant::now(),
        });

        if self.write_log.len() > self.max_log_size {
            self.write_log.pop_front();
        }
    }

    pub fn record_removal(&mut self, _key: String) {
        // Record removal operation
    }

    pub fn analyze_patterns(&self) -> AccessPatterns {
        let mut frequency_map = HashMap::new();

        // Count access frequencies
        for event in self.read_log.iter().chain(self.write_log.iter()) {
            *frequency_map.entry(event.key.clone()).or_insert(0) += 1;
        }

        let total_accesses: usize = frequency_map.values().sum();
        let average_frequency = if frequency_map.is_empty() {
            0.0
        } else {
            total_accesses as f64 / frequency_map.len() as f64
        };

        AccessPatterns {
            frequency_map,
            average_frequency,
            total_accesses,
        }
    }

    pub fn clear(&mut self) {
        self.read_log.clear();
        self.write_log.clear();
    }
}

/// Access event
#[derive(Debug, Clone)]
pub struct AccessEvent {
    pub key: String,
    pub timestamp: Instant,
}

/// Access patterns analysis
#[derive(Debug, Clone)]
pub struct AccessPatterns {
    pub frequency_map: HashMap<String, usize>,
    pub average_frequency: f64,
    pub total_accesses: usize,
}

/// Memory pressure monitor
pub struct MemoryPressureMonitor {
    /// Current memory usage
    current_usage: usize,

    /// Maximum allowed memory
    max_memory: usize,

    /// Pressure thresholds
    warning_threshold: f64,
    critical_threshold: f64,

    /// Pressure history
    pressure_history: VecDeque<f64>,
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPressureMonitor {
    pub fn new() -> Self {
        Self {
            current_usage: 0,
            max_memory: 1024 * 1024 * 1024, // 1GB default
            warning_threshold: 0.7,
            critical_threshold: 0.9,
            pressure_history: VecDeque::new(),
        }
    }

    pub fn update(&mut self, current_usage: usize) {
        self.current_usage = current_usage;
        let pressure_ratio = self.get_pressure_ratio();

        self.pressure_history.push_back(pressure_ratio);
        if self.pressure_history.len() > 100 {
            self.pressure_history.pop_front();
        }
    }

    pub fn get_pressure_ratio(&self) -> f64 {
        if self.max_memory > 0 {
            self.current_usage as f64 / self.max_memory as f64
        } else {
            0.0
        }
    }

    pub fn is_high_pressure(&self) -> bool {
        self.get_pressure_ratio() > self.critical_threshold
    }

    pub fn is_warning_pressure(&self) -> bool {
        self.get_pressure_ratio() > self.warning_threshold
    }

    pub fn reset(&mut self) {
        self.current_usage = 0;
        self.pressure_history.clear();
    }
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub initial_memory_usage: usize,
    pub final_memory_usage: usize,
    pub memory_saved: usize,
    pub optimization_time: Duration,
    pub operations_performed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let manager = TransformerMemoryManager::new(&config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_memory_cache() {
        let cache = MemoryCache::<f32>::new(1024 * 1024, CacheEvictionStrategy::LRU);
        assert!(cache.is_ok());

        let mut c = cache.expect("unwrap failed");
        let tensor = Array2::<f32>::ones((10, 10));
        assert!(c.store("test".to_string(), tensor).is_ok());
        assert!(c.contains("test"));
    }

    /// F59: storing a tensor larger than the whole cache used to drain every
    /// existing entry and *then* report the error.
    #[test]
    fn an_oversized_tensor_is_rejected_without_evicting_anything() {
        let mut cache = MemoryCache::<f64>::new(4096, CacheEvictionStrategy::LRU).expect("cache");
        cache
            .store("keep_a".to_string(), Array2::<f64>::ones((4, 4)))
            .expect("store a");
        cache
            .store("keep_b".to_string(), Array2::<f64>::ones((4, 4)))
            .expect("store b");
        let before = cache.get_memory_usage();
        assert!(before > 0);

        // 100 x 100 f64 = 80_000 bytes, far larger than the 4096-byte cache.
        let huge = Array2::<f64>::ones((100, 100));
        assert!(cache.store("huge".to_string(), huge).is_err());

        assert!(cache.contains("keep_a"), "keep_a was evicted for nothing");
        assert!(cache.contains("keep_b"), "keep_b was evicted for nothing");
        assert_eq!(
            cache.get_memory_usage(),
            before,
            "the failed store changed the cache size"
        );
    }

    /// F67: FIFO eviction must be *insertion* order. It used to read the same
    /// access-ordered queue LRU uses, making the two strategies identical.
    #[test]
    fn fifo_eviction_ignores_accesses_unlike_lru() {
        // Each 4x4 f64 tensor is 128 bytes; a 300-byte cache holds two.
        let build = |strategy: CacheEvictionStrategy| {
            let mut cache = MemoryCache::<f64>::new(300, strategy).expect("cache");
            cache
                .store("first".to_string(), Array2::<f64>::ones((4, 4)))
                .expect("store first");
            cache
                .store("second".to_string(), Array2::<f64>::ones((4, 4)))
                .expect("store second");
            // Touch "first" so it becomes the most recently used.
            assert!(cache.retrieve("first").expect("retrieve").is_some());
            cache
                .store("third".to_string(), Array2::<f64>::ones((4, 4)))
                .expect("store third");
            cache
        };

        let lru = build(CacheEvictionStrategy::LRU);
        assert!(
            lru.contains("first"),
            "LRU must keep the recently accessed entry"
        );
        assert!(!lru.contains("second"), "LRU should drop 'second'");

        let fifo = build(CacheEvictionStrategy::FIFO);
        assert!(
            !fifo.contains("first"),
            "FIFO must drop the first-inserted entry regardless of access"
        );
        assert!(fifo.contains("second"), "FIFO should keep 'second'");
        assert!(fifo.contains("third"));
    }

    /// F67: `Random` eviction used to take `storage.keys().next()`, which is hash
    /// order — not random. Over many trials every candidate must get picked.
    #[test]
    fn random_eviction_actually_varies() {
        let mut victims = std::collections::HashSet::new();
        for _ in 0..200 {
            let mut cache =
                MemoryCache::<f64>::new(300, CacheEvictionStrategy::Random).expect("cache");
            for key in ["a", "b"] {
                cache
                    .store(key.to_string(), Array2::<f64>::ones((4, 4)))
                    .expect("store");
            }
            cache
                .store("c".to_string(), Array2::<f64>::ones((4, 4)))
                .expect("store c");
            for key in ["a", "b"] {
                if !cache.contains(key) {
                    victims.insert(key.to_string());
                }
            }
        }
        assert_eq!(
            victims.len(),
            2,
            "random eviction only ever chose {victims:?}"
        );
    }

    /// F60: compression reported `bytes / 2` while storing the input verbatim.
    /// The ratio must now be a measurement of a payload that really did shrink,
    /// and the round trip must reconstruct within the quantization step.
    #[test]
    fn compression_reports_a_real_measured_ratio() {
        let comp = CompressionManager::<f64>::new(0.5).expect("manager");
        let tensor = Array2::from_shape_fn((8, 8), |(i, j)| (i as f64) - 0.5 * (j as f64));
        let compressed = comp.compress(&tensor).expect("compress");

        assert_eq!(compressed.element_count(), 64);
        assert_eq!(compressed.original_size, 64 * std::mem::size_of::<f64>());
        // 64 code bytes + 2 f64 params + 2 usize shape entries.
        let expected = 64 + 2 * std::mem::size_of::<f64>() + 2 * std::mem::size_of::<usize>();
        assert_eq!(
            compressed.compressed_size, expected,
            "compressed_size is not the real payload size"
        );
        assert!(
            compressed.compressed_size < compressed.original_size,
            "the payload did not shrink: {} vs {}",
            compressed.compressed_size,
            compressed.original_size
        );
        // The old fabricated value was exactly half.
        let half = compressed.original_size / 2;
        assert_ne!(
            compressed.compressed_size, half,
            "compressed_size is still the fabricated original/2"
        );

        let restored = comp.decompress(&compressed).expect("decompress");
        assert_eq!(restored.dim(), tensor.dim());
        let tolerance = compressed.max_reconstruction_error() + 1e-12;
        for (a, b) in tensor.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() <= tolerance,
                "reconstruction error {} exceeds the quantization bound {tolerance}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn a_constant_tensor_round_trips_exactly() {
        let comp = CompressionManager::<f64>::new(0.5).expect("manager");
        let tensor = Array2::<f64>::from_elem((3, 4), 2.5);
        let compressed = comp.compress(&tensor).expect("compress");
        assert_eq!(compressed.quantization_step, 0.0);
        assert_eq!(compressed.max_reconstruction_error(), 0.0);
        let restored = comp.decompress(&compressed).expect("decompress");
        assert_eq!(restored, tensor);
    }

    #[test]
    fn compression_rejects_degenerate_input() {
        let comp = CompressionManager::<f64>::new(0.5).expect("manager");
        assert!(comp.compress(&Array2::<f64>::zeros((0, 3))).is_err());
        let mut nan = Array2::<f64>::zeros((2, 2));
        nan[[0, 0]] = f64::NAN;
        assert!(comp.compress(&nan).is_err());
    }

    #[test]
    fn test_compression_manager() {
        let compression = CompressionManager::<f32>::new(0.5);
        assert!(compression.is_ok());

        let comp = compression.expect("unwrap failed");
        let tensor = Array2::<f32>::ones((5, 5));
        let compressed = comp.compress(&tensor);
        assert!(compressed.is_ok());

        let decompressed = comp.decompress(&compressed.expect("unwrap failed"));
        assert!(decompressed.is_ok());
    }

    #[test]
    fn test_access_tracker() {
        let mut tracker = AccessTracker::new(100);

        tracker.record_read("key1".to_string());
        tracker.record_write("key2".to_string());

        let patterns = tracker.analyze_patterns();
        assert!(patterns.total_accesses > 0);
    }

    #[test]
    fn test_memory_pressure_monitor() {
        let mut monitor = MemoryPressureMonitor::new();

        monitor.update(500 * 1024 * 1024); // 500MB
        assert!(!monitor.is_high_pressure());

        monitor.update(950 * 1024 * 1024); // 950MB
        assert!(monitor.is_high_pressure());
    }
}
