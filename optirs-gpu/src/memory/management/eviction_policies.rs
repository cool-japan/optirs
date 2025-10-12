// Memory eviction policies for GPU memory management
//
// This module provides sophisticated eviction strategies to manage limited
// GPU memory efficiently by determining which data should be removed when
// memory pressure occurs.

#[allow(dead_code)]
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Main eviction engine that manages multiple eviction policies
pub struct EvictionEngine {
    /// Configuration
    config: EvictionConfig,
    /// Statistics
    stats: EvictionStats,
    /// Available eviction policies
    policies: HashMap<String, Box<dyn EvictionPolicy>>,
    /// Currently active policy
    active_policy: String,
    /// Memory regions under management
    memory_regions: HashMap<usize, MemoryRegion>,
    /// Performance monitor
    performance_monitor: EvictionPerformanceMonitor,
    /// Policy selection history
    policy_history: VecDeque<PolicySelection>,
}

/// Eviction configuration
#[derive(Debug, Clone)]
pub struct EvictionConfig {
    /// Enable automatic eviction
    pub auto_eviction: bool,
    /// Memory pressure threshold to trigger eviction
    pub pressure_threshold: f64,
    /// Enable adaptive policy selection
    pub enable_adaptive: bool,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Default eviction policy
    pub default_policy: String,
    /// Policy switching threshold
    pub policy_switch_threshold: f64,
    /// Minimum eviction batch size
    pub min_batch_size: usize,
    /// Maximum eviction batch size
    pub max_batch_size: usize,
    /// Enable workload-aware eviction
    pub workload_aware: bool,
    /// GPU kernel context consideration
    pub kernel_context_weight: f64,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            auto_eviction: true,
            pressure_threshold: 0.85,
            enable_adaptive: true,
            enable_monitoring: true,
            default_policy: "LRU".to_string(),
            policy_switch_threshold: 0.1,
            min_batch_size: 1,
            max_batch_size: 64,
            workload_aware: true,
            kernel_context_weight: 0.3,
        }
    }
}

/// Eviction statistics
#[derive(Debug, Clone, Default)]
pub struct EvictionStats {
    /// Total evictions performed
    pub total_evictions: u64,
    /// Total bytes evicted
    pub total_bytes_evicted: u64,
    /// Total objects evicted
    pub total_objects_evicted: u64,
    /// Average eviction time
    pub average_eviction_time: Duration,
    /// Eviction accuracy (correctly evicted items)
    pub eviction_accuracy: f64,
    /// Policy performance scores
    pub policy_scores: HashMap<String, f64>,
    /// Memory pressure events
    pub pressure_events: u64,
    /// Adaptive policy switches
    pub policy_switches: u64,
}

/// Memory region for eviction management
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Base address
    pub base_addr: usize,
    /// Region size
    pub size: usize,
    /// Cached objects in this region
    pub objects: HashMap<usize, CacheObject>,
    /// Region type (cache, buffer, etc.)
    pub region_type: RegionType,
    /// Current memory pressure
    pub pressure: f64,
    /// Last eviction time
    pub last_eviction: Option<Instant>,
}

/// Types of memory regions
#[derive(Debug, Clone, PartialEq)]
pub enum RegionType {
    Cache,
    Buffer,
    Texture,
    Constant,
    Shared,
    Global,
}

/// Cached object representation
#[derive(Debug, Clone)]
pub struct CacheObject {
    /// Object address
    pub address: usize,
    /// Object size
    pub size: usize,
    /// Creation time
    pub created_at: Instant,
    /// Last access time
    pub last_access: Instant,
    /// Access count
    pub access_count: u32,
    /// Access frequency (accesses per second)
    pub access_frequency: f64,
    /// Object priority
    pub priority: ObjectPriority,
    /// GPU kernel context
    pub kernel_context: Option<u32>,
    /// Object type
    pub object_type: ObjectType,
    /// Eviction cost (higher = more expensive to evict)
    pub eviction_cost: f64,
    /// Replacement cost (higher = more expensive to reload)
    pub replacement_cost: f64,
}

/// Object priority levels
#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum ObjectPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Object type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ObjectType {
    Data,
    Texture,
    Constant,
    Instruction,
    Temporary,
    Persistent,
    Critical,
}

impl CacheObject {
    /// Update access information
    pub fn update_access(&mut self) {
        self.access_count += 1;
        let now = Instant::now();
        let time_since_creation = now.duration_since(self.created_at).as_secs_f64();

        if time_since_creation > 0.0 {
            self.access_frequency = self.access_count as f64 / time_since_creation;
        }

        self.last_access = now;
    }

    /// Calculate object utility score for eviction decisions
    pub fn calculate_utility(&self) -> f64 {
        let age_factor = self.last_access.elapsed().as_secs_f64();
        let frequency_factor = self.access_frequency;
        let priority_factor = match self.priority {
            ObjectPriority::Critical => 10.0,
            ObjectPriority::High => 5.0,
            ObjectPriority::Normal => 1.0,
            ObjectPriority::Low => 0.5,
        };

        let size_factor = 1.0 / (self.size as f64).sqrt();

        // Higher utility = less likely to be evicted
        (frequency_factor * priority_factor * size_factor) / (age_factor + 1.0)
    }
}

/// Eviction policy trait
pub trait EvictionPolicy: Send + Sync {
    fn name(&self) -> &str;
    fn select_victims(&mut self, region: &MemoryRegion, target_bytes: usize) -> Vec<usize>;
    fn update_access(&mut self, address: usize, object: &CacheObject);
    fn add_object(&mut self, address: usize, object: &CacheObject);
    fn remove_object(&mut self, address: usize);
    fn get_statistics(&self) -> PolicyStats;
    fn configure(&mut self, config: &EvictionConfig);
    fn reset(&mut self);
}

/// Policy statistics
#[derive(Debug, Clone, Default)]
pub struct PolicyStats {
    pub evictions: u64,
    pub bytes_evicted: u64,
    pub average_latency: Duration,
    pub accuracy_score: f64,
    pub hit_rate: f64,
}

/// LRU (Least Recently Used) eviction policy
pub struct LRUPolicy {
    /// LRU order tracking
    lru_order: VecDeque<usize>,
    /// Address to position mapping
    address_map: HashMap<usize, usize>,
    /// Statistics
    stats: PolicyStats,
}

impl Default for LRUPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LRUPolicy {
    pub fn new() -> Self {
        Self {
            lru_order: VecDeque::new(),
            address_map: HashMap::new(),
            stats: PolicyStats::default(),
        }
    }

    fn move_to_end(&mut self, address: usize) {
        if let Some(&pos) = self.address_map.get(&address) {
            if pos < self.lru_order.len() {
                self.lru_order.remove(pos);
                self.lru_order.push_back(address);
                self.update_positions();
            }
        }
    }

    fn update_positions(&mut self) {
        self.address_map.clear();
        for (pos, &addr) in self.lru_order.iter().enumerate() {
            self.address_map.insert(addr, pos);
        }
    }
}

impl EvictionPolicy for LRUPolicy {
    fn name(&self) -> &str {
        "LRU"
    }

    fn select_victims(&mut self, region: &MemoryRegion, target_bytes: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        let mut bytes_selected = 0;

        // Start from least recently used
        for &address in &self.lru_order {
            if let Some(object) = region.objects.get(&address) {
                victims.push(address);
                bytes_selected += object.size;

                if bytes_selected >= target_bytes {
                    break;
                }
            }
        }

        self.stats.evictions += victims.len() as u64;
        self.stats.bytes_evicted += bytes_selected as u64;

        victims
    }

    fn update_access(&mut self, address: usize, _object: &CacheObject) {
        self.move_to_end(address);
    }

    fn add_object(&mut self, address: usize, _object: &CacheObject) {
        if !self.address_map.contains_key(&address) {
            self.lru_order.push_back(address);
            self.address_map.insert(address, self.lru_order.len() - 1);
        }
    }

    fn remove_object(&mut self, address: usize) {
        if let Some(&pos) = self.address_map.get(&address) {
            if pos < self.lru_order.len() {
                self.lru_order.remove(pos);
                self.address_map.remove(&address);
                self.update_positions();
            }
        }
    }

    fn get_statistics(&self) -> PolicyStats {
        self.stats.clone()
    }

    fn configure(&mut self, _config: &EvictionConfig) {
        // LRU typically doesn't need configuration
    }

    fn reset(&mut self) {
        self.lru_order.clear();
        self.address_map.clear();
        self.stats = PolicyStats::default();
    }
}

/// LFU (Least Frequently Used) eviction policy
pub struct LFUPolicy {
    /// Frequency tracking
    frequency_map: HashMap<usize, u32>,
    /// Frequency buckets for efficient selection
    frequency_buckets: BTreeMap<u32, HashSet<usize>>,
    /// Statistics
    stats: PolicyStats,
}

impl Default for LFUPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LFUPolicy {
    pub fn new() -> Self {
        Self {
            frequency_map: HashMap::new(),
            frequency_buckets: BTreeMap::new(),
            stats: PolicyStats::default(),
        }
    }

    fn update_frequency(&mut self, address: usize) {
        let old_freq = self.frequency_map.get(&address).copied().unwrap_or(0);
        let new_freq = old_freq + 1;

        // Remove from old bucket
        if old_freq > 0 {
            if let Some(bucket) = self.frequency_buckets.get_mut(&old_freq) {
                bucket.remove(&address);
                if bucket.is_empty() {
                    self.frequency_buckets.remove(&old_freq);
                }
            }
        }

        // Add to new bucket
        self.frequency_buckets
            .entry(new_freq)
            .or_default()
            .insert(address);
        self.frequency_map.insert(address, new_freq);
    }
}

impl EvictionPolicy for LFUPolicy {
    fn name(&self) -> &str {
        "LFU"
    }

    fn select_victims(&mut self, region: &MemoryRegion, target_bytes: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        let mut bytes_selected = 0;

        // Select from lowest frequency buckets first
        for addresses in self.frequency_buckets.values() {
            for &address in addresses {
                if let Some(object) = region.objects.get(&address) {
                    victims.push(address);
                    bytes_selected += object.size;

                    if bytes_selected >= target_bytes {
                        break;
                    }
                }
            }

            if bytes_selected >= target_bytes {
                break;
            }
        }

        self.stats.evictions += victims.len() as u64;
        self.stats.bytes_evicted += bytes_selected as u64;

        victims
    }

    fn update_access(&mut self, address: usize, _object: &CacheObject) {
        self.update_frequency(address);
    }

    fn add_object(&mut self, address: usize, _object: &CacheObject) {
        self.update_frequency(address);
    }

    fn remove_object(&mut self, address: usize) {
        if let Some(freq) = self.frequency_map.remove(&address) {
            if let Some(bucket) = self.frequency_buckets.get_mut(&freq) {
                bucket.remove(&address);
                if bucket.is_empty() {
                    self.frequency_buckets.remove(&freq);
                }
            }
        }
    }

    fn get_statistics(&self) -> PolicyStats {
        self.stats.clone()
    }

    fn configure(&mut self, _config: &EvictionConfig) {
        // LFU typically doesn't need configuration
    }

    fn reset(&mut self) {
        self.frequency_map.clear();
        self.frequency_buckets.clear();
        self.stats = PolicyStats::default();
    }
}

/// FIFO (First In, First Out) eviction policy
pub struct FIFOPolicy {
    /// Insertion order tracking
    insertion_order: VecDeque<usize>,
    /// Statistics
    stats: PolicyStats,
}

impl Default for FIFOPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl FIFOPolicy {
    pub fn new() -> Self {
        Self {
            insertion_order: VecDeque::new(),
            stats: PolicyStats::default(),
        }
    }
}

impl EvictionPolicy for FIFOPolicy {
    fn name(&self) -> &str {
        "FIFO"
    }

    fn select_victims(&mut self, region: &MemoryRegion, target_bytes: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        let mut bytes_selected = 0;

        // Select oldest insertions first
        for &address in &self.insertion_order {
            if let Some(object) = region.objects.get(&address) {
                victims.push(address);
                bytes_selected += object.size;

                if bytes_selected >= target_bytes {
                    break;
                }
            }
        }

        self.stats.evictions += victims.len() as u64;
        self.stats.bytes_evicted += bytes_selected as u64;

        victims
    }

    fn update_access(&mut self, _address: usize, _object: &CacheObject) {
        // FIFO doesn't consider access patterns
    }

    fn add_object(&mut self, address: usize, _object: &CacheObject) {
        self.insertion_order.push_back(address);
    }

    fn remove_object(&mut self, address: usize) {
        if let Some(pos) = self
            .insertion_order
            .iter()
            .position(|&addr| addr == address)
        {
            self.insertion_order.remove(pos);
        }
    }

    fn get_statistics(&self) -> PolicyStats {
        self.stats.clone()
    }

    fn configure(&mut self, _config: &EvictionConfig) {
        // FIFO typically doesn't need configuration
    }

    fn reset(&mut self) {
        self.insertion_order.clear();
        self.stats = PolicyStats::default();
    }
}

/// Clock (Second Chance) eviction policy
pub struct ClockPolicy {
    /// Circular list of objects
    clock_list: Vec<ClockEntry>,
    /// Address to index mapping
    address_map: HashMap<usize, usize>,
    /// Clock hand position
    clock_hand: usize,
    /// Statistics
    stats: PolicyStats,
}

/// Clock entry
#[derive(Debug, Clone)]
struct ClockEntry {
    address: usize,
    reference_bit: bool,
}

impl Default for ClockPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockPolicy {
    pub fn new() -> Self {
        Self {
            clock_list: Vec::new(),
            address_map: HashMap::new(),
            clock_hand: 0,
            stats: PolicyStats::default(),
        }
    }

    fn advance_clock(&mut self) -> Option<usize> {
        if self.clock_list.is_empty() {
            return None;
        }

        let start_pos = self.clock_hand;

        loop {
            let entry = &mut self.clock_list[self.clock_hand];

            if entry.reference_bit {
                // Give second chance
                entry.reference_bit = false;
            } else {
                // Victim found
                let victim = entry.address;
                self.clock_hand = (self.clock_hand + 1) % self.clock_list.len();
                return Some(victim);
            }

            self.clock_hand = (self.clock_hand + 1) % self.clock_list.len();

            if self.clock_hand == start_pos {
                // Full cycle completed, all had reference bits set
                break;
            }
        }

        // If all had reference bits, just return first one
        if !self.clock_list.is_empty() {
            Some(self.clock_list[0].address)
        } else {
            None
        }
    }
}

impl EvictionPolicy for ClockPolicy {
    fn name(&self) -> &str {
        "Clock"
    }

    fn select_victims(&mut self, region: &MemoryRegion, target_bytes: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        let mut bytes_selected = 0;

        while bytes_selected < target_bytes {
            if let Some(victim_addr) = self.advance_clock() {
                if let Some(object) = region.objects.get(&victim_addr) {
                    victims.push(victim_addr);
                    bytes_selected += object.size;
                }
            } else {
                break;
            }
        }

        self.stats.evictions += victims.len() as u64;
        self.stats.bytes_evicted += bytes_selected as u64;

        victims
    }

    fn update_access(&mut self, address: usize, _object: &CacheObject) {
        if let Some(&index) = self.address_map.get(&address) {
            if index < self.clock_list.len() {
                self.clock_list[index].reference_bit = true;
            }
        }
    }

    fn add_object(&mut self, address: usize, _object: &CacheObject) {
        let entry = ClockEntry {
            address,
            reference_bit: true,
        };

        self.clock_list.push(entry);
        self.address_map.insert(address, self.clock_list.len() - 1);
    }

    fn remove_object(&mut self, address: usize) {
        if let Some(&index) = self.address_map.get(&address) {
            if index < self.clock_list.len() {
                self.clock_list.remove(index);
                self.address_map.remove(&address);

                // Update all subsequent indices
                for i in index..self.clock_list.len() {
                    let addr = self.clock_list[i].address;
                    self.address_map.insert(addr, i);
                }

                // Adjust clock hand
                if self.clock_hand > index {
                    self.clock_hand -= 1;
                } else if self.clock_hand >= self.clock_list.len() && !self.clock_list.is_empty() {
                    self.clock_hand = 0;
                }
            }
        }
    }

    fn get_statistics(&self) -> PolicyStats {
        self.stats.clone()
    }

    fn configure(&mut self, _config: &EvictionConfig) {
        // Clock typically doesn't need configuration
    }

    fn reset(&mut self) {
        self.clock_list.clear();
        self.address_map.clear();
        self.clock_hand = 0;
        self.stats = PolicyStats::default();
    }
}

/// Adaptive Replacement Cache (ARC) policy
pub struct ARCPolicy {
    /// T1: Recent cache misses
    t1: VecDeque<usize>,
    /// T2: Recent cache hits
    t2: VecDeque<usize>,
    /// B1: Ghost entries for T1
    b1: VecDeque<usize>,
    /// B2: Ghost entries for T2
    b2: VecDeque<usize>,
    /// Adaptation parameter
    p: usize,
    /// Cache capacity
    capacity: usize,
    /// Statistics
    stats: PolicyStats,
}

impl ARCPolicy {
    pub fn new(capacity: usize) -> Self {
        Self {
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            p: 0,
            capacity,
            stats: PolicyStats::default(),
        }
    }

    fn replace(&mut self, address: usize) -> Option<usize> {
        let t1_len = self.t1.len();

        if t1_len > 0 && (t1_len > self.p || (self.b2.contains(&address) && t1_len == self.p)) {
            // Remove from T1
            if let Some(victim) = self.t1.pop_front() {
                self.b1.push_back(victim);
                if self.b1.len() > self.capacity {
                    self.b1.pop_front();
                }
                return Some(victim);
            }
        } else {
            // Remove from T2
            if let Some(victim) = self.t2.pop_front() {
                self.b2.push_back(victim);
                if self.b2.len() > self.capacity {
                    self.b2.pop_front();
                }
                return Some(victim);
            }
        }

        None
    }

    fn adapt(&mut self, address: usize) {
        let delta = if self.b1.len() >= self.b2.len() {
            1
        } else {
            self.b2.len() / self.b1.len().max(1)
        };

        if self.b1.contains(&address) {
            self.p = (self.p + delta).min(self.capacity);
        } else if self.b2.contains(&address) {
            self.p = self.p.saturating_sub(delta);
        }
    }
}

impl EvictionPolicy for ARCPolicy {
    fn name(&self) -> &str {
        "ARC"
    }

    fn select_victims(&mut self, region: &MemoryRegion, target_bytes: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        let mut bytes_selected = 0;

        while bytes_selected < target_bytes {
            if let Some(victim_addr) = self.replace(0) {
                // Simplified
                if let Some(object) = region.objects.get(&victim_addr) {
                    victims.push(victim_addr);
                    bytes_selected += object.size;
                }
            } else {
                break;
            }
        }

        self.stats.evictions += victims.len() as u64;
        self.stats.bytes_evicted += bytes_selected as u64;

        victims
    }

    fn update_access(&mut self, address: usize, _object: &CacheObject) {
        // Simplified ARC access handling
        if self.t1.contains(&address) {
            // Move from T1 to T2
            if let Some(pos) = self.t1.iter().position(|&addr| addr == address) {
                self.t1.remove(pos);
                self.t2.push_back(address);
            }
        } else if self.t2.contains(&address) {
            // Move to end of T2
            if let Some(pos) = self.t2.iter().position(|&addr| addr == address) {
                self.t2.remove(pos);
                self.t2.push_back(address);
            }
        }
    }

    fn add_object(&mut self, address: usize, _object: &CacheObject) {
        if self.b1.contains(&address) {
            self.adapt(address);
            self.b1.retain(|&addr| addr != address);
            self.t2.push_back(address);
        } else if self.b2.contains(&address) {
            self.adapt(address);
            self.b2.retain(|&addr| addr != address);
            self.t2.push_back(address);
        } else {
            self.t1.push_back(address);
        }
    }

    fn remove_object(&mut self, address: usize) {
        self.t1.retain(|&addr| addr != address);
        self.t2.retain(|&addr| addr != address);
        self.b1.retain(|&addr| addr != address);
        self.b2.retain(|&addr| addr != address);
    }

    fn get_statistics(&self) -> PolicyStats {
        self.stats.clone()
    }

    fn configure(&mut self, _config: &EvictionConfig) {
        // ARC adapts automatically
    }

    fn reset(&mut self) {
        self.t1.clear();
        self.t2.clear();
        self.b1.clear();
        self.b2.clear();
        self.p = 0;
        self.stats = PolicyStats::default();
    }
}

/// Workload-aware eviction policy
pub struct WorkloadAwarePolicy {
    /// Base policy to extend
    base_policy: Box<dyn EvictionPolicy>,
    /// Kernel context weights
    kernel_weights: HashMap<u32, f64>,
    /// Object type priorities
    type_priorities: HashMap<ObjectType, f64>,
    /// Statistics
    stats: PolicyStats,
}

impl WorkloadAwarePolicy {
    pub fn new(base_policy: Box<dyn EvictionPolicy>) -> Self {
        let mut type_priorities = HashMap::new();
        type_priorities.insert(ObjectType::Critical, 10.0);
        type_priorities.insert(ObjectType::Persistent, 5.0);
        type_priorities.insert(ObjectType::Data, 2.0);
        type_priorities.insert(ObjectType::Texture, 1.5);
        type_priorities.insert(ObjectType::Constant, 1.0);
        type_priorities.insert(ObjectType::Temporary, 0.5);

        Self {
            base_policy,
            kernel_weights: HashMap::new(),
            type_priorities,
            stats: PolicyStats::default(),
        }
    }

    fn calculate_eviction_priority(&self, object: &CacheObject) -> f64 {
        let mut priority = object.calculate_utility();

        // Apply object type priority
        if let Some(&type_priority) = self.type_priorities.get(&object.object_type) {
            priority *= type_priority;
        }

        // Apply kernel context weight
        if let Some(kernel_id) = object.kernel_context {
            if let Some(&weight) = self.kernel_weights.get(&kernel_id) {
                priority *= weight;
            }
        }

        // Apply object priority
        let priority_multiplier = match object.priority {
            ObjectPriority::Critical => 100.0,
            ObjectPriority::High => 10.0,
            ObjectPriority::Normal => 1.0,
            ObjectPriority::Low => 0.1,
        };

        priority * priority_multiplier
    }
}

impl EvictionPolicy for WorkloadAwarePolicy {
    fn name(&self) -> &str {
        "WorkloadAware"
    }

    fn select_victims(&mut self, region: &MemoryRegion, target_bytes: usize) -> Vec<usize> {
        // Calculate priorities for all objects
        let mut object_priorities: Vec<(usize, f64)> = region
            .objects
            .iter()
            .map(|(&addr, obj)| (addr, self.calculate_eviction_priority(obj)))
            .collect();

        // Sort by priority (lowest first = best eviction candidates)
        object_priorities
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut victims = Vec::new();
        let mut bytes_selected = 0;

        for (address, _priority) in object_priorities {
            if let Some(object) = region.objects.get(&address) {
                victims.push(address);
                bytes_selected += object.size;

                if bytes_selected >= target_bytes {
                    break;
                }
            }
        }

        self.stats.evictions += victims.len() as u64;
        self.stats.bytes_evicted += bytes_selected as u64;

        victims
    }

    fn update_access(&mut self, address: usize, object: &CacheObject) {
        self.base_policy.update_access(address, object);

        // Update kernel weights based on access patterns
        if let Some(kernel_id) = object.kernel_context {
            let weight = self.kernel_weights.entry(kernel_id).or_insert(1.0);
            *weight = (*weight * 0.9 + 1.1).min(10.0); // Increase weight for active kernels
        }
    }

    fn add_object(&mut self, address: usize, object: &CacheObject) {
        self.base_policy.add_object(address, object);
    }

    fn remove_object(&mut self, address: usize) {
        self.base_policy.remove_object(address);
    }

    fn get_statistics(&self) -> PolicyStats {
        let mut stats = self.stats.clone();
        let base_stats = self.base_policy.get_statistics();

        // Combine statistics
        stats.evictions += base_stats.evictions;
        stats.bytes_evicted += base_stats.bytes_evicted;

        stats
    }

    fn configure(&mut self, config: &EvictionConfig) {
        self.base_policy.configure(config);
    }

    fn reset(&mut self) {
        self.base_policy.reset();
        self.kernel_weights.clear();
        self.stats = PolicyStats::default();
    }
}

/// Performance monitoring for eviction policies
pub struct EvictionPerformanceMonitor {
    /// Performance history
    history: VecDeque<EvictionPerformance>,
    /// Policy performance tracking
    policy_performance: HashMap<String, Vec<f64>>,
    /// Configuration
    config: MonitorConfig,
}

/// Eviction performance sample
#[derive(Debug, Clone)]
pub struct EvictionPerformance {
    pub timestamp: Instant,
    pub policy_name: String,
    pub eviction_time: Duration,
    pub bytes_evicted: usize,
    pub objects_evicted: usize,
    pub accuracy_score: f64,
}

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub history_size: usize,
    pub performance_window: usize,
    pub enable_adaptive: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            performance_window: 100,
            enable_adaptive: true,
        }
    }
}

impl EvictionPerformanceMonitor {
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.history_size),
            policy_performance: HashMap::new(),
            config,
        }
    }

    /// Record eviction performance
    pub fn record_performance(&mut self, performance: EvictionPerformance) {
        self.history.push_back(performance.clone());
        if self.history.len() > self.config.history_size {
            self.history.pop_front();
        }

        // Update policy performance tracking
        let scores = self
            .policy_performance
            .entry(performance.policy_name.clone())
            .or_default();

        scores.push(performance.accuracy_score);
        if scores.len() > self.config.performance_window {
            scores.remove(0);
        }
    }

    /// Get best performing policy
    pub fn get_best_policy(&self) -> Option<String> {
        if !self.config.enable_adaptive {
            return None;
        }

        let mut best_policy = None;
        let mut best_score = 0.0;

        for (policy_name, scores) in &self.policy_performance {
            if scores.len() >= 5 {
                // Minimum samples required
                let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
                if avg_score > best_score {
                    best_score = avg_score;
                    best_policy = Some(policy_name.clone());
                }
            }
        }

        best_policy
    }
}

/// Policy selection record
#[derive(Debug, Clone)]
pub struct PolicySelection {
    pub timestamp: Instant,
    pub policy_name: String,
    pub reason: String,
    pub performance_score: f64,
}

impl EvictionEngine {
    pub fn new(config: EvictionConfig) -> Self {
        let mut policies: HashMap<String, Box<dyn EvictionPolicy>> = HashMap::new();

        // Add built-in policies
        policies.insert("LRU".to_string(), Box::new(LRUPolicy::new()));
        policies.insert("LFU".to_string(), Box::new(LFUPolicy::new()));
        policies.insert("FIFO".to_string(), Box::new(FIFOPolicy::new()));
        policies.insert("Clock".to_string(), Box::new(ClockPolicy::new()));
        policies.insert("ARC".to_string(), Box::new(ARCPolicy::new(1000)));

        if config.workload_aware {
            let base_policy = Box::new(LRUPolicy::new());
            policies.insert(
                "WorkloadAware".to_string(),
                Box::new(WorkloadAwarePolicy::new(base_policy)),
            );
        }

        let active_policy = config.default_policy.clone();
        let performance_monitor = EvictionPerformanceMonitor::new(MonitorConfig::default());

        Self {
            config,
            stats: EvictionStats::default(),
            policies,
            active_policy,
            memory_regions: HashMap::new(),
            performance_monitor,
            policy_history: VecDeque::with_capacity(100),
        }
    }

    /// Register a memory region
    pub fn register_region(&mut self, base_addr: usize, size: usize, region_type: RegionType) {
        let region = MemoryRegion {
            base_addr,
            size,
            objects: HashMap::new(),
            region_type,
            pressure: 0.0,
            last_eviction: None,
        };

        self.memory_regions.insert(base_addr, region);
    }

    /// Add object to tracking
    pub fn add_object(
        &mut self,
        region_addr: usize,
        object: CacheObject,
    ) -> Result<(), EvictionError> {
        let region = self
            .memory_regions
            .get_mut(&region_addr)
            .ok_or_else(|| EvictionError::RegionNotFound("Region not registered".to_string()))?;

        // Add to all policies
        for policy in self.policies.values_mut() {
            policy.add_object(object.address, &object);
        }

        region.objects.insert(object.address, object);
        Ok(())
    }

    /// Update object access
    pub fn update_access(
        &mut self,
        region_addr: usize,
        object_addr: usize,
    ) -> Result<(), EvictionError> {
        let region = self
            .memory_regions
            .get_mut(&region_addr)
            .ok_or_else(|| EvictionError::RegionNotFound("Region not found".to_string()))?;

        if let Some(object) = region.objects.get_mut(&object_addr) {
            object.update_access();

            // Update all policies
            for policy in self.policies.values_mut() {
                policy.update_access(object_addr, object);
            }
        }

        Ok(())
    }

    /// Check if eviction is needed
    pub fn should_evict(&self, region_addr: usize) -> bool {
        if let Some(region) = self.memory_regions.get(&region_addr) {
            region.pressure > self.config.pressure_threshold
        } else {
            false
        }
    }

    /// Perform eviction
    pub fn evict(
        &mut self,
        region_addr: usize,
        target_bytes: usize,
    ) -> Result<Vec<usize>, EvictionError> {
        let region = self
            .memory_regions
            .get(&region_addr)
            .ok_or_else(|| EvictionError::RegionNotFound("Region not found".to_string()))?;

        let start_time = Instant::now();

        // Select policy (adaptive if enabled)
        let policy_name = if self.config.enable_adaptive {
            self.performance_monitor
                .get_best_policy()
                .unwrap_or_else(|| self.active_policy.clone())
        } else {
            self.active_policy.clone()
        };

        let victims = if let Some(policy) = self.policies.get_mut(&policy_name) {
            policy.select_victims(region, target_bytes)
        } else {
            return Err(EvictionError::PolicyNotFound(
                "Policy not available".to_string(),
            ));
        };

        let eviction_time = start_time.elapsed();

        // Remove evicted objects
        if let Some(region) = self.memory_regions.get_mut(&region_addr) {
            for &victim_addr in &victims {
                region.objects.remove(&victim_addr);

                // Remove from all policies
                for policy in self.policies.values_mut() {
                    policy.remove_object(victim_addr);
                }
            }

            region.last_eviction = Some(Instant::now());
        }

        // Update statistics
        self.stats.total_evictions += 1;
        self.stats.total_objects_evicted += victims.len() as u64;

        let total_eviction_time = self.stats.average_eviction_time.as_nanos() as u64
            * (self.stats.total_evictions - 1)
            + eviction_time.as_nanos() as u64;
        self.stats.average_eviction_time =
            Duration::from_nanos(total_eviction_time / self.stats.total_evictions);

        // Record performance
        let performance = EvictionPerformance {
            timestamp: start_time,
            policy_name: policy_name.clone(),
            eviction_time,
            bytes_evicted: target_bytes,
            objects_evicted: victims.len(),
            accuracy_score: 0.8, // Would be calculated based on future access patterns
        };

        self.performance_monitor.record_performance(performance);

        Ok(victims)
    }

    /// Switch active policy
    pub fn switch_policy(&mut self, policy_name: String) -> Result<(), EvictionError> {
        if !self.policies.contains_key(&policy_name) {
            return Err(EvictionError::PolicyNotFound(
                "Policy not available".to_string(),
            ));
        }

        let selection = PolicySelection {
            timestamp: Instant::now(),
            policy_name: policy_name.clone(),
            reason: "Manual switch".to_string(),
            performance_score: 0.0,
        };

        self.policy_history.push_back(selection);
        if self.policy_history.len() > 100 {
            self.policy_history.pop_front();
        }

        self.active_policy = policy_name;
        self.stats.policy_switches += 1;

        Ok(())
    }

    /// Get statistics
    pub fn get_stats(&self) -> &EvictionStats {
        &self.stats
    }

    /// Get policy statistics
    pub fn get_policy_stats(&self) -> HashMap<String, PolicyStats> {
        self.policies
            .iter()
            .map(|(name, policy)| (name.clone(), policy.get_statistics()))
            .collect()
    }
}

/// Eviction errors
#[derive(Debug, Clone)]
pub enum EvictionError {
    RegionNotFound(String),
    PolicyNotFound(String),
    EvictionFailed(String),
    InvalidConfiguration(String),
}

impl std::fmt::Display for EvictionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvictionError::RegionNotFound(msg) => write!(f, "Region not found: {}", msg),
            EvictionError::PolicyNotFound(msg) => write!(f, "Policy not found: {}", msg),
            EvictionError::EvictionFailed(msg) => write!(f, "Eviction failed: {}", msg),
            EvictionError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
        }
    }
}

impl std::error::Error for EvictionError {}

/// Thread-safe eviction engine wrapper
pub struct ThreadSafeEvictionEngine {
    engine: Arc<RwLock<EvictionEngine>>,
}

impl ThreadSafeEvictionEngine {
    pub fn new(config: EvictionConfig) -> Self {
        Self {
            engine: Arc::new(RwLock::new(EvictionEngine::new(config))),
        }
    }

    pub fn should_evict(&self, region_addr: usize) -> bool {
        let engine = self.engine.read().unwrap();
        engine.should_evict(region_addr)
    }

    pub fn evict(
        &self,
        region_addr: usize,
        target_bytes: usize,
    ) -> Result<Vec<usize>, EvictionError> {
        let mut engine = self.engine.write().unwrap();
        engine.evict(region_addr, target_bytes)
    }

    pub fn add_object(&self, region_addr: usize, object: CacheObject) -> Result<(), EvictionError> {
        let mut engine = self.engine.write().unwrap();
        engine.add_object(region_addr, object)
    }

    pub fn get_stats(&self) -> EvictionStats {
        let engine = self.engine.read().unwrap();
        engine.get_stats().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eviction_engine_creation() {
        let config = EvictionConfig::default();
        let engine = EvictionEngine::new(config);
        assert!(!engine.policies.is_empty());
    }

    #[test]
    fn test_lru_policy() {
        let mut policy = LRUPolicy::new();
        assert_eq!(policy.name(), "LRU");

        let object = CacheObject {
            address: 0x1000,
            size: 64,
            created_at: Instant::now(),
            last_access: Instant::now(),
            access_count: 1,
            access_frequency: 1.0,
            priority: ObjectPriority::Normal,
            kernel_context: None,
            object_type: ObjectType::Data,
            eviction_cost: 1.0,
            replacement_cost: 1.0,
        };

        policy.add_object(0x1000, &object);
        assert_eq!(policy.lru_order.len(), 1);
    }

    #[test]
    fn test_cache_object_utility() {
        let object = CacheObject {
            address: 0x1000,
            size: 64,
            created_at: Instant::now() - Duration::from_secs(10),
            last_access: Instant::now() - Duration::from_secs(1),
            access_count: 5,
            access_frequency: 0.5,
            priority: ObjectPriority::High,
            kernel_context: Some(100),
            object_type: ObjectType::Data,
            eviction_cost: 1.0,
            replacement_cost: 2.0,
        };

        let utility = object.calculate_utility();
        assert!(utility > 0.0);
    }

    #[test]
    fn test_thread_safe_engine() {
        let config = EvictionConfig::default();
        let engine = ThreadSafeEvictionEngine::new(config);

        let stats = engine.get_stats();
        assert_eq!(stats.total_evictions, 0);
    }
}
