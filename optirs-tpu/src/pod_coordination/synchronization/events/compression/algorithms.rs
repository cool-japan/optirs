// Compression Algorithm Types and Configuration
//
// This module defines all supported compression algorithms and their configurations
// for event compression in TPU synchronization systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compression algorithms configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAlgorithms {
    /// Available algorithms
    pub available: Vec<Algorithm>,
    /// Default algorithm
    pub default: Algorithm,
    /// Algorithm preferences
    pub preferences: AlgorithmPreferences,
}

impl Default for CompressionAlgorithms {
    fn default() -> Self {
        Self {
            available: vec![
                Algorithm::Zstd(ZstdConfig::default()),
                Algorithm::Gzip(GzipConfig::default()),
                Algorithm::Lz4(Lz4Config::default()),
                Algorithm::Snappy(SnappyConfig::default()),
            ],
            default: Algorithm::Zstd(ZstdConfig::default()),
            preferences: AlgorithmPreferences::default(),
        }
    }
}

impl CompressionAlgorithms {
    /// High-performance algorithm configuration
    pub fn high_performance() -> Self {
        Self {
            available: vec![
                Algorithm::Lz4(Lz4Config {
                    compression_level: 1,
                    block_size: Lz4BlockSize::Block64KB,
                    block_independence: true,
                    checksum: false,
                    dictionary_enabled: false,
                    dictionary_size: 0,
                    auto_flush: true,
                }),
                Algorithm::Snappy(SnappyConfig::default()),
            ],
            default: Algorithm::Lz4(Lz4Config {
                compression_level: 1,
                block_size: Lz4BlockSize::Block64KB,
                block_independence: true,
                checksum: false,
                dictionary_enabled: false,
                dictionary_size: 0,
                auto_flush: true,
            }),
            preferences: AlgorithmPreferences {
                priority_order: vec![
                    "lz4".to_string(),
                    "snappy".to_string(),
                ],
                fallback_algorithm: "snappy".to_string(),
                selection_criteria: SelectionCriteria::Speed,
            },
        }
    }

    /// High-compression ratio algorithm configuration
    pub fn high_compression() -> Self {
        Self {
            available: vec![
                Algorithm::Zstd(ZstdConfig {
                    compression_level: 15,
                    dictionary: Some(ZstdDictionary {
                        data: Vec::new(),
                        size: 32768,
                        training_samples: 1000,
                    }),
                    window_log: 27,
                    hash_log: 20,
                    chain_log: 24,
                    search_log: 3,
                    min_match: 4,
                    target_length: 128,
                    strategy: ZstdStrategy::Btultra2,
                }),
                Algorithm::Brotli(BrotliConfig {
                    quality: 11,
                    window_size: 24,
                    mode: BrotliMode::Text,
                    size_hint: None,
                }),
            ],
            default: Algorithm::Zstd(ZstdConfig {
                compression_level: 15,
                dictionary: None,
                window_log: 27,
                hash_log: 20,
                chain_log: 24,
                search_log: 3,
                min_match: 4,
                target_length: 128,
                strategy: ZstdStrategy::Btultra2,
            }),
            preferences: AlgorithmPreferences {
                priority_order: vec![
                    "zstd".to_string(),
                    "brotli".to_string(),
                ],
                fallback_algorithm: "gzip".to_string(),
                selection_criteria: SelectionCriteria::CompressionRatio,
            },
        }
    }

    /// Balanced algorithm configuration
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Low-latency algorithm configuration
    pub fn low_latency() -> Self {
        Self {
            available: vec![
                Algorithm::Snappy(SnappyConfig::default()),
                Algorithm::Lz4(Lz4Config {
                    compression_level: 1,
                    block_size: Lz4BlockSize::Block4KB,
                    block_independence: true,
                    checksum: false,
                    dictionary_enabled: false,
                    dictionary_size: 0,
                    auto_flush: true,
                }),
            ],
            default: Algorithm::Snappy(SnappyConfig::default()),
            preferences: AlgorithmPreferences {
                priority_order: vec![
                    "snappy".to_string(),
                    "lz4".to_string(),
                ],
                fallback_algorithm: "lz4".to_string(),
                selection_criteria: SelectionCriteria::Latency,
            },
        }
    }
}

/// Algorithm preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmPreferences {
    /// Priority order for algorithm selection
    pub priority_order: Vec<String>,
    /// Fallback algorithm
    pub fallback_algorithm: String,
    /// Selection criteria
    pub selection_criteria: SelectionCriteria,
}

impl Default for AlgorithmPreferences {
    fn default() -> Self {
        Self {
            priority_order: vec![
                "zstd".to_string(),
                "lz4".to_string(),
                "snappy".to_string(),
                "gzip".to_string(),
            ],
            fallback_algorithm: "snappy".to_string(),
            selection_criteria: SelectionCriteria::Balanced,
        }
    }
}

/// Algorithm selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionCriteria {
    Speed,
    CompressionRatio,
    Balanced,
    Latency,
    MemoryUsage,
}

/// Compression algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Algorithm {
    /// Zstandard compression
    Zstd(ZstdConfig),
    /// Gzip compression
    Gzip(GzipConfig),
    /// LZ4 compression
    Lz4(Lz4Config),
    /// Brotli compression
    Brotli(BrotliConfig),
    /// Snappy compression
    Snappy(SnappyConfig),
    /// DEFLATE compression
    Deflate(DeflateConfig),
    /// LZO compression
    Lzo(LzoConfig),
    /// LZ77 compression
    Lz77(Lz77Config),
    /// LZ78 compression
    Lz78(Lz78Config),
    /// Burrows-Wheeler Transform
    BurrowsWheeler(BurrowsWheelerConfig),
    /// Huffman coding
    Huffman(HuffmanConfig),
    /// Arithmetic coding
    Arithmetic(ArithmeticConfig),
    /// Prediction by partial matching
    Ppm(PpmConfig),
    /// Lempel-Ziv-Welch 2
    Lzw2(Lzw2Config),
}

/// Zstandard compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZstdConfig {
    /// Compression level (1-22)
    pub compression_level: i32,
    /// Dictionary configuration
    pub dictionary: Option<ZstdDictionary>,
    /// Window log size
    pub window_log: u32,
    /// Hash log size
    pub hash_log: u32,
    /// Chain log size
    pub chain_log: u32,
    /// Search log size
    pub search_log: u32,
    /// Minimum match length
    pub min_match: u32,
    /// Target length
    pub target_length: u32,
    /// Compression strategy
    pub strategy: ZstdStrategy,
}

impl Default for ZstdConfig {
    fn default() -> Self {
        Self {
            compression_level: 3,
            dictionary: None,
            window_log: 23,
            hash_log: 17,
            chain_log: 17,
            search_log: 1,
            min_match: 4,
            target_length: 32,
            strategy: ZstdStrategy::Fast,
        }
    }
}

/// Zstandard compression strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZstdStrategy {
    Fast,
    Dfast,
    Greedy,
    Lazy,
    Lazy2,
    Btlazy2,
    Btopt,
    Btultra,
    Btultra2,
}

/// Zstandard dictionary configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZstdDictionary {
    /// Dictionary data
    pub data: Vec<u8>,
    /// Dictionary size
    pub size: usize,
    /// Number of training samples
    pub training_samples: usize,
}

/// Gzip compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GzipConfig {
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Window size
    pub window_size: u8,
    /// Memory level
    pub memory_level: u8,
    /// Compression strategy
    pub strategy: GzipStrategy,
    /// Enable header
    pub header: bool,
}

impl Default for GzipConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            window_size: 15,
            memory_level: 8,
            strategy: GzipStrategy::Default,
            header: true,
        }
    }
}

/// Gzip compression strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GzipStrategy {
    Default,
    Filtered,
    HuffmanOnly,
    Rle,
    Fixed,
}

/// LZ4 compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lz4Config {
    /// Compression level
    pub compression_level: i32,
    /// Block size
    pub block_size: Lz4BlockSize,
    /// Block independence
    pub block_independence: bool,
    /// Enable checksum
    pub checksum: bool,
    /// Enable dictionary
    pub dictionary_enabled: bool,
    /// Dictionary size
    pub dictionary_size: usize,
    /// Auto flush
    pub auto_flush: bool,
}

impl Default for Lz4Config {
    fn default() -> Self {
        Self {
            compression_level: 1,
            block_size: Lz4BlockSize::Block64KB,
            block_independence: true,
            checksum: true,
            dictionary_enabled: false,
            dictionary_size: 0,
            auto_flush: false,
        }
    }
}

/// LZ4 block sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Lz4BlockSize {
    Block4KB,
    Block16KB,
    Block64KB,
    Block256KB,
    Block1MB,
    Block4MB,
}

/// Brotli compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrotliConfig {
    /// Quality level (0-11)
    pub quality: u32,
    /// Window size
    pub window_size: u8,
    /// Compression mode
    pub mode: BrotliMode,
    /// Size hint
    pub size_hint: Option<usize>,
}

impl Default for BrotliConfig {
    fn default() -> Self {
        Self {
            quality: 6,
            window_size: 22,
            mode: BrotliMode::Generic,
            size_hint: None,
        }
    }
}

/// Brotli compression modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrotliMode {
    Generic,
    Text,
    Font,
}

/// Snappy compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnappyConfig {
    /// Enable checksum
    pub checksum: bool,
}

impl Default for SnappyConfig {
    fn default() -> Self {
        Self {
            checksum: false,
        }
    }
}

/// DEFLATE compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeflateConfig {
    /// Compression level
    pub compression_level: u8,
    /// Window bits
    pub window_bits: u8,
    /// Memory level
    pub memory_level: u8,
    /// Strategy
    pub strategy: DeflateStrategy,
}

impl Default for DeflateConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            window_bits: 15,
            memory_level: 8,
            strategy: DeflateStrategy::Default,
        }
    }
}

/// DEFLATE compression strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeflateStrategy {
    Default,
    Filtered,
    HuffmanOnly,
    Rle,
    Fixed,
}

/// LZO compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LzoConfig {
    /// Algorithm variant
    pub algorithm: LzoAlgorithm,
    /// Compression level
    pub compression_level: u8,
    /// Optimization level
    pub optimization_level: u8,
}

impl Default for LzoConfig {
    fn default() -> Self {
        Self {
            algorithm: LzoAlgorithm::Lzo1x,
            compression_level: 1,
            optimization_level: 1,
        }
    }
}

/// LZO algorithm variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LzoAlgorithm {
    Lzo1x,
    Lzo1y,
    Lzo1z,
    Lzo2a,
}

/// LZ77 compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lz77Config {
    /// Window size
    pub window_size: usize,
    /// Lookahead buffer size
    pub lookahead_size: usize,
    /// Minimum match length
    pub min_match_length: usize,
    /// Maximum match length
    pub max_match_length: usize,
}

impl Default for Lz77Config {
    fn default() -> Self {
        Self {
            window_size: 32768,
            lookahead_size: 258,
            min_match_length: 3,
            max_match_length: 258,
        }
    }
}

/// LZ78 compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lz78Config {
    /// Dictionary size
    pub dictionary_size: usize,
    /// Dictionary growth strategy
    pub growth_strategy: DictionaryGrowthStrategy,
}

impl Default for Lz78Config {
    fn default() -> Self {
        Self {
            dictionary_size: 4096,
            growth_strategy: DictionaryGrowthStrategy::Dynamic,
        }
    }
}

/// Dictionary growth strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DictionaryGrowthStrategy {
    Fixed,
    Dynamic,
    Adaptive,
}

/// Burrows-Wheeler Transform configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurrowsWheelerConfig {
    /// Block size
    pub block_size: usize,
    /// Use move-to-front transform
    pub move_to_front: bool,
    /// Use run-length encoding
    pub run_length_encoding: bool,
}

impl Default for BurrowsWheelerConfig {
    fn default() -> Self {
        Self {
            block_size: 900000,
            move_to_front: true,
            run_length_encoding: true,
        }
    }
}

/// Huffman coding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuffmanConfig {
    /// Maximum code length
    pub max_code_length: u8,
    /// Symbol frequency table
    pub frequency_table: HashMap<u8, u32>,
    /// Use canonical codes
    pub canonical: bool,
}

impl Default for HuffmanConfig {
    fn default() -> Self {
        Self {
            max_code_length: 15,
            frequency_table: HashMap::new(),
            canonical: true,
        }
    }
}

/// Arithmetic coding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmeticConfig {
    /// Precision bits
    pub precision_bits: u8,
    /// Probability model
    pub probability_model: ProbabilityModel,
    /// Use adaptive model
    pub adaptive: bool,
}

impl Default for ArithmeticConfig {
    fn default() -> Self {
        Self {
            precision_bits: 32,
            probability_model: ProbabilityModel::Uniform,
            adaptive: true,
        }
    }
}

/// Probability models for arithmetic coding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbabilityModel {
    Uniform,
    Adaptive,
    Static(HashMap<u8, f64>),
}

/// PPM (Prediction by Partial Matching) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpmConfig {
    /// Maximum context order
    pub max_order: u8,
    /// Escape method
    pub escape_method: EscapeMethod,
    /// Update exclusions
    pub update_exclusions: bool,
}

impl Default for PpmConfig {
    fn default() -> Self {
        Self {
            max_order: 4,
            escape_method: EscapeMethod::MethodC,
            update_exclusions: true,
        }
    }
}

/// Escape methods for PPM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscapeMethod {
    MethodA,
    MethodB,
    MethodC,
    MethodD,
}

/// LZW2 compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lzw2Config {
    /// Initial dictionary size
    pub initial_dict_size: usize,
    /// Maximum dictionary size
    pub max_dict_size: usize,
    /// Reset strategy
    pub reset_strategy: ResetStrategy,
}

impl Default for Lzw2Config {
    fn default() -> Self {
        Self {
            initial_dict_size: 256,
            max_dict_size: 4096,
            reset_strategy: ResetStrategy::OnFull,
        }
    }
}

/// Dictionary reset strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResetStrategy {
    Never,
    OnFull,
    Periodic(usize),
    Adaptive,
}