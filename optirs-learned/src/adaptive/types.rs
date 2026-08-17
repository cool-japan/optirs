//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(unused_imports)]
use crate::error::Result;
use crate::transformer_based_optimizer::{TransformerOptimizer, TransformerOptimizerConfig};

use super::landscape::LandscapeStatistics;
use super::performance_predictor::TransformerPerformancePredictor;
use super::predictor::{PredictorFitReport, PredictorSample};
use crate::LearnedOptimizerConfig;
#[allow(dead_code)]
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

/// Architecture adaptation result
#[derive(Debug)]
pub struct ArchitectureAdaptation<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Adapted configuration
    pub adapted_config: TransformerOptimizerConfig<T>,
    /// Architecture changes
    pub changes: Vec<ArchitectureChange>,
    /// Expected improvement
    pub expected_improvement: T,
    /// Adaptation confidence
    pub confidence: T,
}
/// Memory-efficient attention manager
#[derive(Debug)]
pub struct MemoryEfficientAttentionManager<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Attention pattern cache
    pattern_cache: AttentionPatternCache<T>,
    /// Sparse attention mask
    sparse_mask: Array2<bool>,
    /// Local attention windows
    local_windows: Vec<AttentionWindow>,
    /// Global attention heads
    global_heads: Vec<usize>,
    /// Memory usage tracker
    memory_tracker: MemoryUsageTracker,
    /// Minimum sparsity level, from `AdaptiveConfig::attention_sparsity_threshold`
    sparsity_floor: T,
    /// Whether the head count may be reduced, from
    /// `AdaptiveConfig::dynamic_head_pruning`
    allow_head_pruning: bool,
    /// Lower bound on the attention span, from
    /// `AdaptiveConfig::min_sequence_length`
    min_span: usize,
    /// Upper bound on the attention span, from
    /// `AdaptiveConfig::max_sequence_length`
    max_span: usize,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    MemoryEfficientAttentionManager<T>
{
    /// Build the attention manager from the adaptive configuration.
    ///
    /// The configuration is genuinely consumed: `attention_sparsity_threshold`
    /// becomes the sparsity floor used by [`Self::optimize_attention`],
    /// `memory_budget` becomes the tracker's budget, `dynamic_head_pruning`
    /// decides whether the head count is allowed to shrink, and
    /// `max_sequence_length` / `min_sequence_length` bound the attention span.
    /// Previously every field was ignored.
    fn new(config: &AdaptiveConfig<T>) -> Result<Self> {
        if config.min_sequence_length == 0
            || config.min_sequence_length > config.max_sequence_length
        {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "sequence length bounds must satisfy 0 < min ({}) <= max ({})",
                config.min_sequence_length, config.max_sequence_length
            )));
        }
        Ok(Self {
            pattern_cache: AttentionPatternCache::new(),
            sparse_mask: Array2::default((0, 0)),
            local_windows: Vec::new(),
            global_heads: Vec::new(),
            memory_tracker: MemoryUsageTracker::with_budget(config.memory_budget),
            sparsity_floor: config.attention_sparsity_threshold,
            allow_head_pruning: config.dynamic_head_pruning,
            min_span: config.min_sequence_length,
            max_span: config.max_sequence_length,
        })
    }
    fn optimize_attention(
        &mut self,
        analysis: &LandscapeAnalysis<T>,
    ) -> Result<AttentionOptimization<T>> {
        let complexity = analysis.complexity.to_f64().unwrap_or(0.5);
        let difficulty = analysis.difficulty.to_f64().unwrap_or(0.3);
        let (num_heads, seq_len) = self.determine_attention_dimensions(complexity, difficulty)?;
        let mut attention_patterns = Array3::zeros((num_heads, seq_len, seq_len));
        self.generate_attention_patterns(&mut attention_patterns, analysis)?;
        // A more complex landscape needs a denser (less pruned) pattern, but the
        // configured `attention_sparsity_threshold` is always the floor.
        let requested: T =
            scirs2_core::numeric::NumCast::from(if complexity > 0.7 { 0.05 } else { 0.15 })
                .unwrap_or_else(|| T::zero());
        let sparsitylevel = if requested > self.sparsity_floor {
            requested
        } else {
            self.sparsity_floor
        };
        self.apply_sparsity_mask(&mut attention_patterns, sparsitylevel)?;
        let pattern_key = format!("pattern_{}_{}", num_heads, seq_len);
        self.pattern_cache
            .patterns
            .insert(pattern_key.clone(), attention_patterns.clone());
        *self
            .pattern_cache
            .usage_frequency
            .entry(pattern_key)
            .or_insert(0) += 1;
        let original_size = 8 * 512 * 512 * std::mem::size_of::<f32>();
        let optimized_size = num_heads * seq_len * seq_len * std::mem::size_of::<f32>();
        let memory_savings = original_size.saturating_sub(optimized_size);
        let speedup_from_sparsity = T::one() / sparsitylevel;
        let speedup_from_dimensions =
            T::from(512.0 * 512.0 / (seq_len * seq_len) as f64).expect("unwrap failed");
        let computational_speedup = (speedup_from_sparsity + speedup_from_dimensions)
            / scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero());
        self.memory_tracker.current_usage += optimized_size;
        if self.memory_tracker.current_usage > self.memory_tracker.peak_usage {
            self.memory_tracker.peak_usage = self.memory_tracker.current_usage;
        }
        Ok(AttentionOptimization {
            attention_patterns,
            sparsitylevel,
            memory_savings,
            computational_speedup,
        })
    }
    /// Choose (head count, attention span) for the measured landscape.
    ///
    /// Head count grows with complexity; it is only allowed to *shrink* below
    /// the base when `AdaptiveConfig::dynamic_head_pruning` is enabled. The span
    /// grows with difficulty and is clamped to the configured
    /// `[min_sequence_length, max_sequence_length]` window instead of a
    /// hardcoded `[256, 1024]`.
    fn determine_attention_dimensions(
        &self,
        complexity: f64,
        difficulty: f64,
    ) -> Result<(usize, usize)> {
        let base_heads = 8;
        let scaled_heads = if complexity > 0.8 {
            (base_heads as f64 * 1.5) as usize
        } else if complexity < 0.3 && self.allow_head_pruning {
            (base_heads as f64 * 0.75) as usize
        } else {
            base_heads
        };
        let heads = scaled_heads.clamp(4, 16);

        // Base span is the midpoint of the configured window, so the whole
        // decision lives inside the caller's budget.
        let base_span = (self.min_span + self.max_span) / 2;
        let scaled_span = if difficulty > 0.7 {
            (base_span as f64 * 1.2) as usize
        } else if difficulty < 0.3 {
            (base_span as f64 * 0.8) as usize
        } else {
            base_span
        };
        let seq_len = scaled_span.clamp(self.min_span, self.max_span).max(1);
        Ok((heads, seq_len))
    }
    fn generate_attention_patterns(
        &self,
        patterns: &mut Array3<T>,
        analysis: &LandscapeAnalysis<T>,
    ) -> Result<()> {
        let (num_heads, seq_len, _) = patterns.dim();
        for head in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let distance = ((i as i32 - j as i32).abs() as f64).sqrt();
                    let base_attention = (-scirs2_core::numeric::NumCast::from(distance)
                        .unwrap_or_else(|| T::zero())
                        / (scirs2_core::numeric::NumCast::from(seq_len)
                            .unwrap_or_else(|| T::zero())
                            * scirs2_core::numeric::NumCast::from(0.1)
                                .unwrap_or_else(|| T::zero())))
                    .exp();
                    let complexity_factor = analysis.complexity.to_f64().unwrap_or(0.5);
                    let modulated_attention = base_attention
                        * (T::one()
                            + scirs2_core::numeric::NumCast::from(complexity_factor)
                                .unwrap_or_else(|| T::zero())
                                * scirs2_core::numeric::NumCast::from(0.3)
                                    .unwrap_or_else(|| T::zero()));
                    patterns[[head, i, j]] =
                        scirs2_core::numeric::NumCast::from(modulated_attention)
                            .unwrap_or_else(|| T::zero());
                }
            }
        }
        Ok(())
    }
    fn apply_sparsity_mask(&self, patterns: &mut Array3<T>, sparsitylevel: T) -> Result<()> {
        let sparsity_threshold = sparsitylevel.to_f64().unwrap_or(0.1);
        patterns.map_inplace(|x| {
            if x.to_f64().unwrap_or(0.0) < sparsity_threshold {
                *x = T::zero();
            }
        });
        Ok(())
    }
}
/// Adaptive sequence processor for variable-length optimization histories
#[derive(Debug)]
pub struct AdaptiveSequenceProcessor<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Current sequence length
    current_length: usize,
    /// Sequence importance scores
    importance_scores: VecDeque<T>,
    /// Sequence compression ratio
    compression_ratio: T,
    /// Information-preserving compressor
    compressor: SequenceCompressor<T>,
    /// Adaptive windowing strategy
    windowing_strategy: WindowingStrategy,
    /// Lower bound, from `AdaptiveConfig::min_sequence_length`
    min_length: usize,
    /// Upper bound, from `AdaptiveConfig::max_sequence_length`
    max_length: usize,
    /// Whether the length may change, from `AdaptiveConfig::adaptive_sequence_length`
    length_adaptation_enabled: bool,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    AdaptiveSequenceProcessor<T>
{
    /// Build the sequence processor from the adaptive configuration.
    ///
    /// `min_sequence_length` / `max_sequence_length` bound every adaptation (the
    /// starting length is the midpoint), and `adaptive_sequence_length` decides
    /// whether the length is allowed to move at all. Previously the starting
    /// length was the constant 512 and the bounds were hardcoded to `[64, 2048]`
    /// inside `adapt_to_landscape`.
    fn new(config: &AdaptiveConfig<T>) -> Result<Self> {
        if config.min_sequence_length == 0
            || config.min_sequence_length > config.max_sequence_length
        {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "sequence length bounds must satisfy 0 < min ({}) <= max ({})",
                config.min_sequence_length, config.max_sequence_length
            )));
        }
        Ok(Self {
            current_length: (config.min_sequence_length + config.max_sequence_length) / 2,
            importance_scores: VecDeque::new(),
            compression_ratio: scirs2_core::numeric::NumCast::from(0.8)
                .unwrap_or_else(|| T::zero()),
            compressor: SequenceCompressor::new()?,
            windowing_strategy: WindowingStrategy::ImportanceBased,
            min_length: config.min_sequence_length,
            max_length: config.max_sequence_length,
            length_adaptation_enabled: config.adaptive_sequence_length,
        })
    }
    fn adapt_to_landscape(
        &mut self,
        analysis: &LandscapeAnalysis<T>,
    ) -> Result<SequenceAdaptation<T>> {
        let complexity_factor = analysis.complexity.to_f64().unwrap_or(0.5);
        let difficulty_factor = analysis.difficulty.to_f64().unwrap_or(0.3);
        let new_length = if !self.length_adaptation_enabled {
            self.current_length
        } else if complexity_factor > 0.7 {
            ((self.current_length as f64 * 1.2) as usize).min(self.max_length)
        } else if complexity_factor < 0.3 {
            ((self.current_length as f64 * 0.8) as usize).max(self.min_length)
        } else {
            self.current_length
        };
        let new_compression_ratio = if difficulty_factor > 0.6 {
            self.compression_ratio
                * scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero())
        } else {
            self.compression_ratio
                * scirs2_core::numeric::NumCast::from(1.1).unwrap_or_else(|| T::zero())
        }
        .min(scirs2_core::numeric::NumCast::from(0.95).unwrap_or_else(|| T::zero()))
        .max(scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()));
        self.current_length = new_length;
        self.compression_ratio = new_compression_ratio;
        let information_preservation = T::one()
            - (T::one() - new_compression_ratio)
                * scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
        let length_efficiency =
            scirs2_core::numeric::NumCast::from(self.current_length as f64 / new_length as f64)
                .unwrap_or_else(|| T::zero());
        let compression_efficiency = T::one() / new_compression_ratio;
        let efficiency_gain = (length_efficiency + compression_efficiency)
            / scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::zero());
        self.update_importance_scores(analysis)?;
        Ok(SequenceAdaptation {
            new_length,
            compression_ratio: new_compression_ratio,
            information_preservation,
            efficiency_gain,
        })
    }
    fn update_importance_scores(&mut self, analysis: &LandscapeAnalysis<T>) -> Result<()> {
        let base_importance = scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
        let complexity_boost = analysis.complexity
            * scirs2_core::numeric::NumCast::from(0.3).unwrap_or_else(|| T::zero());
        let difficulty_boost = analysis.difficulty
            * scirs2_core::numeric::NumCast::from(0.2).unwrap_or_else(|| T::zero());
        let new_importance = base_importance + complexity_boost + difficulty_boost;
        self.importance_scores.push_back(new_importance);
        if self.importance_scores.len() > 100 {
            self.importance_scores.pop_front();
        }
        Ok(())
    }
}
/// Windowing strategies for adaptive sequences
#[derive(Debug, Clone, Copy)]
pub enum WindowingStrategy {
    /// Fixed size window
    Fixed,
    /// Sliding window
    Sliding,
    /// Importance-based window
    ImportanceBased,
    /// Hierarchical windowing
    Hierarchical,
    /// Attention-guided windowing
    AttentionGuided,
}
/// Landscape features for optimization analysis
#[derive(Debug, Clone)]
pub struct LandscapeFeatures<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Smoothness measure
    pub(super) smoothness: T,
    /// Multimodality indicator
    pub(super) multimodality: T,
    /// Noise level
    pub(super) noise_level: T,
    /// Curvature information
    pub(super) curvature: CurvatureInfo<T>,
    /// Gradient characteristics
    pub(super) gradient_characteristics: GradientCharacteristics<T>,
}
/// Symmetry types
#[derive(Debug, Clone, Copy)]
pub enum SymmetryType {
    Rotational,
    Reflectional,
    Translational,
    Scale,
    Discrete,
}
/// Pattern applicability
#[derive(Debug, Clone)]
pub struct PatternApplicability {
    /// Applicable regions
    regions: Vec<Array1<f64>>,
    /// Applicability score
    score: f64,
    /// Confidence level
    confidence: f64,
}
/// Sequence adaptation result
#[derive(Debug)]
pub struct SequenceAdaptation<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// New sequence length
    pub new_length: usize,
    /// Compression ratio
    pub compression_ratio: T,
    /// Information preservation score
    pub information_preservation: T,
    /// Processing efficiency gain
    pub efficiency_gain: T,
}
/// Landscape analysis result
#[derive(Debug)]
pub struct LandscapeAnalysis<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Landscape complexity
    pub complexity: T,
    /// Optimization difficulty
    pub difficulty: T,
    /// Recommended strategies
    pub recommended_strategies: Vec<OptimizationStrategy>,
    /// Analysis confidence
    pub confidence: T,
}
#[derive(Debug, Clone, Copy)]
pub enum PositionalEncodingType {
    Sinusoidal,
    Learned,
    Rotary,
    Relative,
}
/// Complexity estimator
#[derive(Debug)]
pub struct ComplexityEstimator<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Computational complexity
    computational_complexity: T,
    /// Sample complexity
    sample_complexity: T,
    /// Model complexity
    model_complexity: T,
    /// Generalization complexity
    generalization_complexity: T,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    ComplexityEstimator<T>
{
    fn new() -> Self {
        Self {
            computational_complexity: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
            sample_complexity: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
            model_complexity: scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()),
            generalization_complexity: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
        }
    }
}
/// Curvature information
#[derive(Debug, Clone)]
pub struct CurvatureInfo<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Mean curvature
    pub(super) mean_curvature: T,
    /// Gaussian curvature
    pub(super) gaussian_curvature: T,
    /// Principal curvatures
    pub(super) principal_curvatures: Vec<T>,
    /// Condition number
    pub(super) condition_number: T,
}
#[derive(Debug, Clone, Copy)]
pub enum SaddleDetectionAlgorithm {
    EigenvalueBased,
    NewtonBased,
    PerturbationBased,
    FlowBased,
}
/// Architecture performance metrics
#[derive(Debug, Clone)]
pub struct ArchitecturePerformance<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Convergence speed
    convergence_speed: T,
    /// Final performance
    final_performance: T,
    /// Memory efficiency
    memory_efficiency: T,
    /// Computational cost
    computational_cost: T,
    /// Adaptation time
    adaptation_time: T,
}
/// Enhancement result
#[derive(Debug)]
pub struct EnhancementResult<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Sequence processing adaptations
    pub sequence_adaptation: SequenceAdaptation<T>,
    /// Attention optimizations
    pub attention_optimization: AttentionOptimization<T>,
    /// Architecture adaptations
    pub architecture_adaptation: ArchitectureAdaptation<T>,
    /// Performance predictions
    pub performance_prediction: PerformancePrediction<T>,
    /// Landscape analysis
    pub landscape_analysis: LandscapeAnalysis<T>,
    /// Convergence metrics
    pub convergence_metrics: ConvergenceMetrics<T>,
}
/// Enhancement statistics for tracking performance
#[derive(Debug, Clone)]
pub struct EnhancementStatistics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Total number of enhancements performed
    pub total_enhancements: usize,
    /// Average complexity of analyzed landscapes
    pub average_complexity: T,
    /// Average performance achieved
    pub average_performance: T,
    /// Memory efficiency measure
    pub memory_efficiency: T,
    /// Success rate of adaptations
    pub adaptation_success_rate: T,
}
/// Sequence compressor for information-preserving compression
#[derive(Debug)]
pub struct SequenceCompressor<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Compression algorithm
    algorithm: CompressionAlgorithm,
    /// Compression parameters
    params: CompressionParams<T>,
    /// Quality metrics
    quality_metrics: CompressionQualityMetrics<T>,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    SequenceCompressor<T>
{
    fn new() -> Result<Self> {
        Ok(Self {
            algorithm: CompressionAlgorithm::PCA,
            params: CompressionParams::default(),
            quality_metrics: CompressionQualityMetrics::default(),
        })
    }
}
/// Prediction result
#[derive(Debug, Clone)]
pub struct PredictionResult<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Predicted performance
    predicted_performance: T,
    /// Confidence interval
    confidence_interval: (T, T),
    /// Prediction timestamp
    timestamp: Instant,
    /// Prediction features
    features: Array1<T>,
}
/// Cache eviction policies
#[derive(Debug, Clone, Copy)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
    /// Importance-based eviction
    ImportanceBased,
}
/// Symmetry representation
#[derive(Debug, Clone)]
pub struct Symmetry<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
{
    /// Symmetry type
    symmetry_type: SymmetryType,
    /// Symmetry parameters
    parameters: Array1<T>,
    /// Symmetry strength
    strength: T,
}
// NOTE: `PerformanceFeatureExtractor` used to be declared here holding a
// dimension, an empty cache and an all-ones importance vector, with no method
// that extracted anything. Real feature extraction is
// `TransformerPerformancePredictor::extract_features` above, producing the
// documented [`PredictionFeatures`] vector.
/// Compression algorithms
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    /// Principal Component Analysis
    PCA,
    /// Autoencoder compression
    Autoencoder,
    /// Singular Value Decomposition
    SVD,
    /// Random projection
    RandomProjection,
    /// Learned compression
    Learned,
}
/// Connectivity analyzer
#[derive(Debug)]
pub struct ConnectivityAnalyzer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Connectivity graph
    connectivity_graph: Array2<T>,
    /// Path analysis results
    path_analysis: PathAnalysisResults<T>,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    ConnectivityAnalyzer<T>
{
    fn new() -> Self {
        Self {
            connectivity_graph: Array2::zeros((0, 0)),
            path_analysis: PathAnalysisResults {
                shortest_paths: Vec::new(),
                path_difficulties: Vec::new(),
                connectivity_measure: T::zero(),
            },
        }
    }
}
/// Optimization path
#[derive(Debug, Clone)]
pub struct OptimizationPath<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Path points
    points: Vec<Array1<T>>,
    /// Path values
    values: Vec<T>,
    /// Path length
    length: T,
    /// Path difficulty
    difficulty: T,
}
/// Pattern library
#[derive(Debug)]
pub struct PatternLibrary<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Pattern database
    patterns: HashMap<String, OptimizationPattern<T>>,
    /// Pattern index
    pattern_index: HashMap<PatternType, Vec<String>>,
    /// Usage statistics
    usage_stats: HashMap<String, usize>,
}
/// Architecture change types
#[derive(Debug, Clone)]
pub enum ArchitectureChange {
    LayerCountChange(usize),
    HiddenSizeChange(usize),
    AttentionHeadChange(usize),
    ActivationChange(ActivationType),
    DropoutChange(f64),
}
/// Attention window for local attention
#[derive(Debug, Clone)]
pub struct AttentionWindow {
    /// Window start position
    start: usize,
    /// Window size
    size: usize,
    /// Window importance
    importance: f64,
    /// Window type
    window_type: WindowType,
}
/// Window types for attention
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Local neighborhood
    Local,
    /// Strided window
    Strided,
    /// Dilated window
    Dilated,
    /// Hierarchical window
    Hierarchical,
}
/// Resource constraints for adaptation
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum memory usage (MB)
    pub(super) max_memory: usize,
    /// Maximum computation time (ms)
    pub(super) max_computation_time: u64,
    /// Maximum model parameters
    pub(super) max_parameters: usize,
    /// Energy budget (if applicable)
    pub(super) energy_budget: Option<f64>,
}
#[derive(Debug, Clone, Copy)]
pub enum BasinAnalysisMethod {
    FloodFill,
    GradientFlow,
    MonteCarloSampling,
    TopologicalAnalysis,
}
/// Global structure detector
#[derive(Debug)]
pub struct GlobalStructureDetector<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Connectivity analyzer
    connectivity_analyzer: ConnectivityAnalyzer<T>,
    /// Symmetry detector
    symmetry_detector: SymmetryDetector<T>,
    /// Pattern recognizer
    pattern_recognizer: PatternRecognizer<T>,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    GlobalStructureDetector<T>
{
    fn new() -> Self {
        Self {
            connectivity_analyzer: ConnectivityAnalyzer::new(),
            symmetry_detector: SymmetryDetector::new(),
            pattern_recognizer: PatternRecognizer::new(),
        }
    }
}
/// Basin representation
#[derive(Debug, Clone)]
pub struct Basin<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static> {
    /// Basin boundary
    boundary: Vec<Array1<T>>,
    /// Volume
    volume: T,
    /// Depth
    depth: T,
    /// Shape characteristics
    shape: BasinShape,
}
/// Basin analyzer
#[derive(Debug)]
pub struct BasinAnalyzer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Basin characteristics
    basin_characteristics: Vec<Basin<T>>,
    /// Analysis method
    analysis_method: BasinAnalysisMethod,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    BasinAnalyzer<T>
{
    fn new() -> Self {
        Self {
            basin_characteristics: Vec::new(),
            analysis_method: BasinAnalysisMethod::GradientFlow,
        }
    }
}
/// Attention pattern cache for efficiency
#[derive(Debug)]
pub struct AttentionPatternCache<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Cached patterns
    patterns: HashMap<String, Array3<T>>,
    /// Pattern usage frequency
    usage_frequency: HashMap<String, usize>,
    /// Cache capacity
    capacity: usize,
    /// Eviction policy
    eviction_policy: CacheEvictionPolicy,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    AttentionPatternCache<T>
{
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            usage_frequency: HashMap::new(),
            capacity: 1000,
            eviction_policy: CacheEvictionPolicy::LRU,
        }
    }
}
/// Optimization landscape analyzer
#[derive(Debug)]
pub struct OptimizationLandscapeAnalyzer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Landscape features
    landscape_features: LandscapeFeatures<T>,
    /// Complexity estimator
    complexity_estimator: ComplexityEstimator<T>,
    /// Local geometry analyzer
    local_geometry: LocalGeometryAnalyzer<T>,
    /// Global structure detector
    global_structure: GlobalStructureDetector<T>,
    /// Analysis cache
    analysis_cache: HashMap<String, AnalysisResult<T>>,
    /// Horizon (in history samples) at which the analysis confidence saturates.
    /// Taken from `AdaptiveConfig::prediction_horizon`.
    confidence_horizon: usize,
    /// Statistics of the most recent analysis, for inspection and testing.
    last_statistics: LandscapeStatistics,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    OptimizationLandscapeAnalyzer<T>
{
    fn new(config: &AdaptiveConfig<T>) -> Result<Self> {
        Ok(Self {
            landscape_features: LandscapeFeatures::default(),
            complexity_estimator: ComplexityEstimator::new(),
            local_geometry: LocalGeometryAnalyzer::new(),
            global_structure: GlobalStructureDetector::new(),
            analysis_cache: HashMap::new(),
            confidence_horizon: config.prediction_horizon.max(2),
            last_statistics: LandscapeStatistics::empty(),
        })
    }

    /// Analyze the optimization landscape from the observed history.
    ///
    /// Every returned number is a statistic of the inputs — see
    /// [`LandscapeStatistics`] for the formulas. Previously this ignored both
    /// arguments and returned `complexity = 0.5`, `difficulty = 0.3`,
    /// `confidence = 0.9`, `strategies = [Adaptive]` on every call.
    ///
    /// The recommended strategy set is derived from where the history sits in
    /// the (complexity, difficulty) plane:
    ///
    /// | condition                          | strategy       |
    /// |------------------------------------|----------------|
    /// | difficulty high, complexity high   | `Exploratory`  |
    /// | difficulty high, complexity low    | `Aggressive`   |
    /// | difficulty low, complexity high    | `Conservative` |
    /// | difficulty low, complexity low     | `Exploitative` |
    /// | anything in between                | `Adaptive`     |
    ///
    /// # Errors
    /// Returns `Err` when both histories are empty — there is nothing to
    /// analyze, and answering with neutral constants is what this method used to
    /// do wrong.
    fn analyze(
        &mut self,
        gradient_history: &[Array1<T>],
        loss_history: &[T],
    ) -> Result<LandscapeAnalysis<T>> {
        if gradient_history.is_empty() && loss_history.is_empty() {
            return Err(crate::error::OptimError::InsufficientData(
                "landscape analysis needs at least one gradient or loss sample".to_string(),
            ));
        }

        let stats = LandscapeStatistics::from_history(gradient_history, loss_history);
        let complexity = stats.complexity();
        let difficulty = stats.difficulty();
        let confidence = stats.analysis_confidence(self.confidence_horizon);

        // Keep the (previously write-only) feature summaries in sync with the
        // statistics that were actually measured.
        self.landscape_features.smoothness =
            scirs2_core::numeric::NumCast::from(1.0 - stats.loss_roughness)
                .unwrap_or_else(|| T::zero());
        self.landscape_features.multimodality =
            scirs2_core::numeric::NumCast::from(stats.reversal_rate).unwrap_or_else(|| T::zero());
        self.landscape_features.noise_level =
            scirs2_core::numeric::NumCast::from(stats.gradient_norm_cv)
                .unwrap_or_else(|| T::zero());
        self.complexity_estimator.computational_complexity =
            scirs2_core::numeric::NumCast::from(complexity).unwrap_or_else(|| T::zero());
        self.complexity_estimator.generalization_complexity =
            scirs2_core::numeric::NumCast::from(difficulty).unwrap_or_else(|| T::zero());

        let high = 0.6;
        let low = 0.4;
        let strategy = match (difficulty >= high, complexity >= high) {
            (true, true) => OptimizationStrategy::Exploratory,
            (true, false) if complexity <= low => OptimizationStrategy::Aggressive,
            (false, true) if difficulty <= low => OptimizationStrategy::Conservative,
            _ if difficulty <= low && complexity <= low => OptimizationStrategy::Exploitative,
            _ => OptimizationStrategy::Adaptive,
        };

        self.last_statistics = stats;

        Ok(LandscapeAnalysis {
            complexity: scirs2_core::numeric::NumCast::from(complexity)
                .unwrap_or_else(|| T::zero()),
            difficulty: scirs2_core::numeric::NumCast::from(difficulty)
                .unwrap_or_else(|| T::zero()),
            recommended_strategies: vec![strategy],
            confidence: scirs2_core::numeric::NumCast::from(confidence)
                .unwrap_or_else(|| T::zero()),
        })
    }

    /// Statistics behind the most recent [`Self::analyze`] call.
    pub fn last_statistics(&self) -> &LandscapeStatistics {
        &self.last_statistics
    }
}
/// Analysis result container
#[derive(Debug, Clone)]
pub struct AnalysisResult<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Analysis timestamp
    timestamp: Instant,
    /// Analysis features
    features: HashMap<String, T>,
    /// Confidence score
    confidence: T,
    /// Analysis metadata
    metadata: HashMap<String, String>,
    /// Complexity score
    complexity_score: T,
    /// Difficulty score
    difficulty_score: T,
    /// Recommended adaptations
    recommended_adaptations: Vec<OptimizationStrategy>,
}
/// Adaptation strategies
#[derive(Debug, Clone, Copy)]
pub enum AdaptationStrategy {
    /// Gradual adaptation
    Gradual,
    /// Rapid adaptation
    Rapid,
    /// Conservative adaptation
    Conservative,
    /// Aggressive adaptation
    Aggressive,
    /// Learned adaptation
    Learned,
}
/// Local geometry analyzer
#[derive(Debug)]
pub struct LocalGeometryAnalyzer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Local minima detector
    local_minima_detector: LocalMinimaDetector<T>,
    /// Saddle point detector
    saddle_point_detector: SaddlePointDetector<T>,
    /// Basin analyzer
    basin_analyzer: BasinAnalyzer<T>,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    LocalGeometryAnalyzer<T>
{
    fn new() -> Self {
        Self {
            local_minima_detector: LocalMinimaDetector::new(),
            saddle_point_detector: SaddlePointDetector::new(),
            basin_analyzer: BasinAnalyzer::new(),
        }
    }
}
/// Configuration for adaptive enhancements
#[derive(Debug, Clone)]
pub struct AdaptiveConfig<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Enable adaptive sequence length
    pub adaptive_sequence_length: bool,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Minimum sequence length
    pub min_sequence_length: usize,
    /// Attention sparsity threshold
    pub attention_sparsity_threshold: T,
    /// Memory budget (MB)
    pub memory_budget: usize,
    /// Enable dynamic head pruning
    pub dynamic_head_pruning: bool,
    /// Enable layer adaptation
    pub layer_adaptation: bool,
    /// Landscape analysis frequency
    pub landscape_analysis_frequency: usize,
    /// Performance prediction horizon
    pub prediction_horizon: usize,
    /// Adaptation learning rate
    pub adaptation_lr: T,
}
// NOTE: `PredictionCache` used to be declared here as a `HashMap` plus a
// `hit_rate` and a `capacity` that were never read or updated. The real,
// capacity-enforcing, hit-rate-tracking cache lives in
// [`crate::adaptive::predictor::PredictionCache`] and is re-exported by
// `adaptive::mod`.
/// Attention optimization result
#[derive(Debug, Clone)]
pub struct AttentionOptimization<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Optimized attention patterns
    pub attention_patterns: Array3<T>,
    /// Sparsity level achieved
    pub sparsitylevel: T,
    /// Memory savings
    pub memory_savings: usize,
    /// Computational speedup
    pub computational_speedup: T,
}
/// Performance prediction result
#[derive(Debug)]
pub struct PerformancePrediction<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Predicted convergence improvement
    pub convergence_improvement: T,
    /// Predicted final performance
    pub final_performance: T,
    /// Prediction confidence
    pub confidence: T,
    /// Uncertainty estimate
    pub uncertainty: T,
}
/// Uncertainty estimation methods
#[derive(Debug, Clone, Copy)]
pub enum UncertaintyMethod {
    /// Monte Carlo dropout
    MonteCarloDropout,
    /// Bayesian neural networks
    BayesianNN,
    /// Ensemble methods
    Ensemble,
    /// Variational inference
    VariationalInference,
}
// NOTE: `PredictorNetwork` used to be declared here with `Array2::zeros`
// weights and no forward pass, and `UncertaintyEstimator` held three hardcoded
// constants (0.1 / 0.05 / 0.15) that nothing computed. Both are replaced by
// [`crate::adaptive::predictor::PredictorNetwork`], which is Xavier-initialized,
// has a real forward pass, is fitted by closed-form ridge regression, and
// derives its uncertainty from the ridge posterior.
/// Optimization pattern
#[derive(Debug, Clone)]
pub struct OptimizationPattern<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Pattern type
    pattern_type: PatternType,
    /// Pattern parameters
    parameters: HashMap<String, T>,
    /// Pattern confidence
    confidence: T,
    /// Pattern applicability
    applicability: PatternApplicability,
}
/// Compression quality metrics
#[derive(Debug, Clone)]
pub struct CompressionQualityMetrics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Reconstruction error
    pub(super) reconstruction_error: T,
    /// Information loss
    pub(super) information_loss: T,
    /// Compression ratio achieved
    pub(super) compression_ratio: T,
    /// Compression time
    pub(super) compression_time: u64,
}
/// Path analysis results
#[derive(Debug, Clone)]
pub struct PathAnalysisResults<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Shortest paths
    shortest_paths: Vec<OptimizationPath<T>>,
    /// Path difficulties
    path_difficulties: Vec<T>,
    /// Connectivity measure
    connectivity_measure: T,
}
/// Detection algorithms
#[derive(Debug, Clone, Copy)]
pub enum MinimaDetectionAlgorithm {
    GradientBased,
    HessianBased,
    TopologyBased,
    SamplingBased,
}
/// Optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum OptimizationStrategy {
    Conservative,
    Aggressive,
    Adaptive,
    Exploratory,
    Exploitative,
}
/// Saddle point representation
#[derive(Debug, Clone)]
pub struct SaddlePoint<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Position
    position: Array1<T>,
    /// Value
    value: T,
    /// Escape directions
    escape_directions: Vec<Array1<T>>,
    /// Instability measure
    instability: T,
}
/// Adaptive Transformer optimizer configuration
#[derive(Debug, Clone)]
pub struct AdaptiveTransformerOptimizerConfig {
    /// Base learned optimizer config
    pub base_config: LearnedOptimizerConfig,
    /// Model dimension
    pub model_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Feed-forward dimension
    pub ff_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Attention dropout rate
    pub attention_dropout: f64,
    /// Feed-forward dropout rate
    pub ff_dropout: f64,
    /// Layer normalization epsilon
    pub layer_norm_eps: f64,
    /// Pre-layer normalization flag
    pub pre_layer_norm: bool,
    /// Positional encoding type
    pub pos_encoding_type: PositionalEncodingType,
    /// Relative position bias flag
    pub relative_position_bias: bool,
    /// Use RoPE (Rotary Position Embedding)
    pub use_rope: bool,
    /// Gradient checkpointing flag
    pub gradient_checkpointing: bool,
    /// Attention optimization configuration
    pub attention_optimization: AttentionOptimization<f64>,
    /// Multi-scale attention flag
    pub multi_scale_attention: bool,
    /// Cross-attention flag
    pub cross_attention: bool,
}
/// Memory usage tracker
#[derive(Debug)]
pub struct MemoryUsageTracker {
    /// Current memory usage (MB)
    current_usage: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Memory budget
    budget: usize,
    /// Usage history
    usage_history: VecDeque<usize>,
}
impl MemoryUsageTracker {
    /// Tracker with a caller-supplied budget (from
    /// `AdaptiveConfig::memory_budget`); `0` falls back to 8192 MB.
    fn with_budget(budget: usize) -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            budget: if budget == 0 { 8192 } else { budget },
            usage_history: VecDeque::new(),
        }
    }

    /// Configured budget.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Peak usage observed so far.
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }
}
/// Local minimum representation
#[derive(Debug, Clone)]
pub struct LocalMinimum<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Position
    position: Array1<T>,
    /// Value
    value: T,
    /// Basin size
    basin_size: T,
    /// Escape difficulty
    escape_difficulty: T,
}
/// Saddle point detector
#[derive(Debug)]
pub struct SaddlePointDetector<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Detection threshold
    threshold: T,
    /// Detected saddle points
    detected_saddles: Vec<SaddlePoint<T>>,
    /// Detection algorithm
    algorithm: SaddleDetectionAlgorithm,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    SaddlePointDetector<T>
{
    fn new() -> Self {
        Self {
            threshold: scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero()),
            detected_saddles: Vec::new(),
            algorithm: SaddleDetectionAlgorithm::EigenvalueBased,
        }
    }
}
/// Architecture search space
#[derive(Debug, Clone)]
pub struct ArchitectureSearchSpace {
    /// Layer count range
    pub(super) layer_count_range: (usize, usize),
    /// Hidden size options
    pub(super) hidden_size_options: Vec<usize>,
    /// Attention head options
    pub(super) attention_head_options: Vec<usize>,
    /// Feed-forward dimension options
    pub(super) ff_dim_options: Vec<usize>,
    /// Activation function options
    pub(super) activation_options: Vec<ActivationType>,
}
/// Symmetry detector
#[derive(Debug)]
pub struct SymmetryDetector<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Detected symmetries
    symmetries: Vec<Symmetry<T>>,
    /// Symmetry types
    symmetry_types: Vec<SymmetryType>,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    SymmetryDetector<T>
{
    fn new() -> Self {
        Self {
            symmetries: Vec::new(),
            symmetry_types: Vec::new(),
        }
    }
}
/// Pattern types
#[derive(Debug, Clone, Copy)]
pub enum PatternType {
    ConvexRegion,
    RavineLike,
    PlateauLike,
    Oscillatory,
    Monotonic,
    Chaotic,
}
/// Convergence metrics for tracking optimization progress
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Rate of convergence
    pub convergence_rate: T,
    /// Stability measure
    pub stability_measure: T,
    /// Plateau detection flag
    pub plateau_detection: bool,
    /// Oscillation measure
    pub oscillation_measure: T,
}
/// Adaptive Transformer Enhancement System
pub struct AdaptiveTransformerEnhancement<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Adaptive sequence processor
    sequence_processor: AdaptiveSequenceProcessor<T>,
    /// Memory-efficient attention manager
    attention_manager: MemoryEfficientAttentionManager<T>,
    /// Dynamic architecture adapter
    architecture_adapter: DynamicArchitectureAdapter<T>,
    /// Optimization landscape analyzer
    landscape_analyzer: OptimizationLandscapeAnalyzer<T>,
    /// Performance predictor
    performance_predictor: TransformerPerformancePredictor<T>,
    /// Adaptive configuration
    adaptive_config: AdaptiveConfig<T>,
    /// Exponential moving average of per-parameter squared gradients.
    ///
    /// This is what makes the adaptive learning rate genuinely per-parameter:
    /// the previous implementation returned `base_lr · scale · 1.1` for even
    /// indices and `base_lr · scale · 0.9` for odd ones, i.e. index-parity noise
    /// that carried no information about the parameter at all.
    grad_second_moment: Array1<T>,
    /// Number of `enhanced_optimize_step` calls, used to bias-correct the
    /// second-moment estimate and to honour
    /// `AdaptiveConfig::landscape_analysis_frequency`.
    step_count: usize,
    /// Most recent landscape analysis, reused between analysis refreshes.
    cached_landscape: Option<LandscapeAnalysis<T>>,
}
impl<
        T: Float
            + Debug
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static
            + std::iter::Sum
            + scirs2_core::numeric::FromPrimitive,
    > AdaptiveTransformerEnhancement<T>
{
    /// Enhance a transformer optimizer for the current optimization task.
    ///
    /// This now *acts on* `transformer` instead of ignoring it: after deriving
    /// the architecture adaptation, the adapted configuration is pushed into the
    /// optimizer via [`TransformerOptimizer::apply_architecture_config`]. That
    /// call rebuilds the optimizer's transformer stack when a dimension actually
    /// changed, and is a no-op when the proposal matches the live configuration.
    ///
    /// The returned [`EnhancementResult::architecture_adaptation`] reports the
    /// proposal; whether it was applied is observable through the optimizer's
    /// own configuration.
    ///
    /// # Errors
    /// Propagates analysis errors (an empty history is an error, not a set of
    /// neutral constants) and any error from reconfiguring the optimizer.
    pub fn enhance_optimizer(
        &mut self,
        transformer: &mut TransformerOptimizer<T>,
        gradient_history: &[Array1<T>],
        losshistory: &[T],
    ) -> Result<EnhancementResult<T>> {
        // Adapt *this* optimizer's architecture, not a default one.
        self.architecture_adapter.sync_with(transformer.config());

        let landscape_analysis = self
            .landscape_analyzer
            .analyze(gradient_history, losshistory)?;
        let sequence_adaptation = self
            .sequence_processor
            .adapt_to_landscape(&landscape_analysis)?;
        let attention_optimization = self
            .attention_manager
            .optimize_attention(&landscape_analysis)?;
        let architecture_adaptation = self.architecture_adapter.adapt_architecture(
            &landscape_analysis,
            &sequence_adaptation,
            &attention_optimization,
        )?;
        let performance_prediction = self
            .performance_predictor
            .predict_improvement(&landscape_analysis, &architecture_adaptation)?;

        // F22: actually push the adaptation into the optimizer we were handed.
        transformer.apply_architecture_config(&architecture_adaptation.adapted_config)?;

        let convergence_metrics = self.calculate_convergence_metrics(losshistory);
        Ok(EnhancementResult {
            sequence_adaptation,
            attention_optimization,
            architecture_adaptation,
            performance_prediction,
            landscape_analysis,
            convergence_metrics,
        })
    }
}
impl<
        T: Float
            + Debug
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static
            + std::iter::Sum,
    > AdaptiveTransformerEnhancement<T>
{
    pub fn new(config: AdaptiveConfig<T>) -> Result<Self> {
        Ok(Self {
            sequence_processor: AdaptiveSequenceProcessor::new(&config)?,
            attention_manager: MemoryEfficientAttentionManager::new(&config)?,
            architecture_adapter: DynamicArchitectureAdapter::new(&config)?,
            landscape_analyzer: OptimizationLandscapeAnalyzer::new(&config)?,
            performance_predictor: TransformerPerformancePredictor::new(&config)?,
            adaptive_config: config,
            grad_second_moment: Array1::zeros(0),
            step_count: 0,
            cached_landscape: None,
        })
    }

    /// Read-only access to the configuration this enhancement was built with.
    pub fn config(&self) -> &AdaptiveConfig<T> {
        &self.adaptive_config
    }

    /// Fit the performance predictor on observed outcomes.
    ///
    /// See [`TransformerPerformancePredictor::train`]. Until this is called,
    /// [`EnhancementResult::performance_prediction`] honestly reports zero means
    /// with unit uncertainty.
    pub fn train_performance_predictor(
        &mut self,
        samples: &[PredictorSample],
    ) -> Result<PredictorFitReport> {
        self.performance_predictor.train(samples)
    }

    /// Whether the performance predictor has been fitted.
    pub fn predictor_is_trained(&self) -> bool {
        self.performance_predictor.is_trained()
    }

    /// Current per-parameter second-moment estimates (empty before the first
    /// [`Self::enhanced_optimize_step`]).
    pub fn gradient_second_moment(&self) -> &Array1<T> {
        &self.grad_second_moment
    }
    /// Enhanced optimization step with adaptive features.
    ///
    /// The landscape is re-analyzed at most once every
    /// `AdaptiveConfig::landscape_analysis_frequency` steps and reused in
    /// between; that field previously configured nothing and the (constant)
    /// analysis was recomputed on every call.
    pub fn enhanced_optimize_step(
        &mut self,
        parameters: &mut Array1<T>,
        gradients: &Array1<T>,
        losshistory: &[T],
        gradient_history: &[Array1<T>],
    ) -> Result<EnhancementResult<T>> {
        let frequency = self.adaptive_config.landscape_analysis_frequency.max(1);
        let refresh = self.cached_landscape.is_none() || self.step_count.is_multiple_of(frequency);
        let landscape = if refresh {
            let fresh = self
                .landscape_analyzer
                .analyze(gradient_history, losshistory)?;
            self.cached_landscape = Some(LandscapeAnalysis {
                complexity: fresh.complexity,
                difficulty: fresh.difficulty,
                recommended_strategies: fresh.recommended_strategies.clone(),
                confidence: fresh.confidence,
            });
            fresh
        } else {
            match &self.cached_landscape {
                Some(cached) => LandscapeAnalysis {
                    complexity: cached.complexity,
                    difficulty: cached.difficulty,
                    recommended_strategies: cached.recommended_strategies.clone(),
                    confidence: cached.confidence,
                },
                None => self
                    .landscape_analyzer
                    .analyze(gradient_history, losshistory)?,
            }
        };
        let sequence_adaptation = self.sequence_processor.adapt_to_landscape(&landscape)?;
        let attention_optimization = self.attention_manager.optimize_attention(&landscape)?;
        let architecture_adaptation = self.architecture_adapter.adapt_architecture(
            &landscape,
            &sequence_adaptation,
            &attention_optimization,
        )?;
        let performance_prediction = self
            .performance_predictor
            .predict_improvement(&landscape, &architecture_adaptation)?;
        self.apply_adaptive_updates(
            parameters,
            gradients,
            &sequence_adaptation,
            &attention_optimization,
            &architecture_adaptation,
        )?;
        Ok(EnhancementResult {
            landscape_analysis: landscape,
            sequence_adaptation,
            attention_optimization,
            architecture_adaptation,
            performance_prediction,
            convergence_metrics: self.calculate_convergence_metrics(losshistory),
        })
    }
    /// Apply adaptive updates to parameters
    /// Apply the adaptive update to `parameters`.
    ///
    /// Two things changed here. First, the composite scale is now a *bounded*
    /// blend rather than a raw triple product: `computational_speedup` can be in
    /// the tens, so multiplying three unbounded factors and dividing by 3 could
    /// produce a scale far above 1 and diverge. Second, the per-parameter rate
    /// is a real RMSProp-style rate over the observed gradient second moment
    /// (see [`Self::calculate_adaptive_learning_rate`]) instead of index parity.
    ///
    /// # Errors
    /// Returns `Err` when `parameters` and `gradients` have different lengths —
    /// previously `zip` silently truncated to the shorter one, leaving the tail
    /// of a longer parameter vector un-updated.
    fn apply_adaptive_updates(
        &mut self,
        parameters: &mut Array1<T>,
        gradients: &Array1<T>,
        sequence_adaptation: &SequenceAdaptation<T>,
        attention_optimization: &AttentionOptimization<T>,
        architecture_adaptation: &ArchitectureAdaptation<T>,
    ) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(crate::error::OptimError::ComputationError(format!(
                "parameter/gradient length mismatch: {} vs {}",
                parameters.len(),
                gradients.len()
            )));
        }

        // Bound each factor to [0, 1] before averaging so the composite scale is
        // itself in [0, 1] and cannot blow the step up.
        let squash = |v: T| -> T {
            let x = v.to_f64().unwrap_or(0.0).abs();
            scirs2_core::numeric::NumCast::from(x / (1.0 + x)).unwrap_or_else(|| T::zero())
        };
        let three: T = scirs2_core::numeric::NumCast::from(3.0).unwrap_or_else(|| T::one());
        let combined_scale = (squash(sequence_adaptation.efficiency_gain)
            + squash(attention_optimization.computational_speedup)
            + squash(architecture_adaptation.expected_improvement))
            / three;

        // Second-moment accumulator tracks the parameter vector's width.
        if self.grad_second_moment.len() != gradients.len() {
            self.grad_second_moment = Array1::zeros(gradients.len());
        }
        let beta: T = scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero());
        let one_minus_beta = T::one() - beta;
        for (slot, &g) in self.grad_second_moment.iter_mut().zip(gradients.iter()) {
            *slot = beta * *slot + one_minus_beta * g * g;
        }

        self.step_count += 1;
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let adaptive_lr = self.calculate_adaptive_learning_rate(i, combined_scale)?;
            *param = *param - adaptive_lr * *grad;
        }
        Ok(())
    }

    /// Per-parameter learning rate for parameter `param_index`.
    ///
    /// RMSProp form: `lr_i = base_lr · scale / (sqrt(v̂_i) + ε)` where `v̂_i` is
    /// the bias-corrected exponential moving average of `g_i²` maintained by
    /// [`Self::apply_adaptive_updates`]. Parameters with a persistently large
    /// gradient therefore get a *smaller* step, which is the whole point of a
    /// per-parameter rate.
    ///
    /// `base_lr` is `AdaptiveConfig::adaptation_lr` — the config field that used
    /// to be ignored in favour of a hardcoded `0.001`.
    ///
    /// # Errors
    /// Returns `Err` when `param_index` is outside the tracked width.
    fn calculate_adaptive_learning_rate(&self, param_index: usize, basescale: T) -> Result<T> {
        if param_index >= self.grad_second_moment.len() {
            return Err(crate::error::OptimError::ComputationError(format!(
                "parameter index {param_index} outside the tracked width {}",
                self.grad_second_moment.len()
            )));
        }
        let base_lr = self.adaptive_config.adaptation_lr;
        let epsilon: T = scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero());

        // Bias correction for the EMA warm-up (Kingma & Ba 2015, eq. 2).
        let beta = 0.9_f64;
        let correction = 1.0 - beta.powi(self.step_count.max(1) as i32);
        let correction_t: T = scirs2_core::numeric::NumCast::from(correction.max(f64::EPSILON))
            .unwrap_or_else(|| T::one());
        let v_hat = self.grad_second_moment[param_index] / correction_t;

        Ok(base_lr * basescale / (v_hat.sqrt() + epsilon))
    }
    /// Calculate convergence metrics
    fn calculate_convergence_metrics(&self, losshistory: &[T]) -> ConvergenceMetrics<T> {
        if losshistory.len() < 2 {
            return ConvergenceMetrics {
                convergence_rate: T::zero(),
                stability_measure: T::zero(),
                plateau_detection: false,
                oscillation_measure: T::zero(),
            };
        }
        let recent_losses = &losshistory[losshistory.len().saturating_sub(10)..];
        let convergence_rate = if recent_losses.len() >= 2 {
            let initial = recent_losses[0];
            let final_loss = recent_losses[recent_losses.len() - 1];
            if initial > T::zero() {
                (initial - final_loss) / initial
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };
        let mean_loss = recent_losses.iter().cloned().sum::<T>()
            / T::from(recent_losses.len()).expect("unwrap failed");
        let variance = recent_losses
            .iter()
            .map(|&loss| {
                let diff = loss - mean_loss;
                diff * diff
            })
            .sum::<T>()
            / T::from(recent_losses.len()).expect("unwrap failed");
        let stability_measure = T::one() / (T::one() + variance);
        let plateau_threshold =
            scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero());
        let plateau_detection = convergence_rate.abs() < plateau_threshold;
        let mut oscillation_sum = T::zero();
        for i in 1..recent_losses.len() {
            oscillation_sum = oscillation_sum + (recent_losses[i] - recent_losses[i - 1]).abs();
        }
        let oscillation_measure = if recent_losses.len() > 1 {
            oscillation_sum / T::from(recent_losses.len() - 1).expect("unwrap failed")
        } else {
            T::zero()
        };
        ConvergenceMetrics {
            convergence_rate,
            stability_measure,
            plateau_detection,
            oscillation_measure,
        }
    }
    /// Update internal state based on optimization progress
    pub fn update_enhancement_state(
        &mut self,
        enhancement_result: &EnhancementResult<T>,
    ) -> Result<()> {
        let cache_key = format!(
            "analysis_{}",
            enhancement_result
                .landscape_analysis
                .complexity
                .to_f64()
                .unwrap_or(0.0)
        );
        self.landscape_analyzer.analysis_cache.insert(
            cache_key,
            AnalysisResult {
                timestamp: Instant::now(),
                features: {
                    let mut features = HashMap::new();
                    features.insert(
                        "complexity".to_string(),
                        enhancement_result.landscape_analysis.complexity,
                    );
                    features.insert(
                        "difficulty".to_string(),
                        enhancement_result.landscape_analysis.difficulty,
                    );
                    features
                },
                complexity_score: enhancement_result.landscape_analysis.complexity,
                difficulty_score: enhancement_result.landscape_analysis.difficulty,
                recommended_adaptations: enhancement_result
                    .landscape_analysis
                    .recommended_strategies
                    .clone(),
                confidence: enhancement_result.landscape_analysis.confidence,
                metadata: HashMap::new(),
            },
        );
        let performance = ArchitecturePerformance {
            convergence_speed: enhancement_result.convergence_metrics.convergence_rate,
            final_performance: T::one() - enhancement_result.performance_prediction.uncertainty,
            memory_efficiency: T::from(enhancement_result.attention_optimization.memory_savings)
                .expect("unwrap failed"),
            computational_cost: T::one()
                / enhancement_result
                    .attention_optimization
                    .computational_speedup,
            adaptation_time: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
        };
        self.architecture_adapter
            .performance_history
            .push_back(performance);
        if self.architecture_adapter.performance_history.len() > 100 {
            self.architecture_adapter.performance_history.pop_front();
        }
        Ok(())
    }
    /// Get enhancement statistics
    pub fn get_enhancement_statistics(&self) -> EnhancementStatistics<T> {
        let avg_complexity = if !self.landscape_analyzer.analysis_cache.is_empty() {
            let sum: T = self
                .landscape_analyzer
                .analysis_cache
                .values()
                .map(|result| result.complexity_score)
                .sum();
            sum / T::from(self.landscape_analyzer.analysis_cache.len()).expect("unwrap failed")
        } else {
            scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero())
        };
        let avg_performance = if !self.architecture_adapter.performance_history.is_empty() {
            let sum: T = self
                .architecture_adapter
                .performance_history
                .iter()
                .map(|perf| perf.final_performance)
                .sum();
            sum / T::from(self.architecture_adapter.performance_history.len())
                .expect("unwrap failed")
        } else {
            scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero())
        };
        EnhancementStatistics {
            total_enhancements: self.landscape_analyzer.analysis_cache.len(),
            average_complexity: avg_complexity,
            average_performance: avg_performance,
            memory_efficiency: scirs2_core::numeric::NumCast::from(0.8)
                .unwrap_or_else(|| T::zero()),
            adaptation_success_rate: scirs2_core::numeric::NumCast::from(0.85)
                .unwrap_or_else(|| T::zero()),
        }
    }
}
/// Basin shapes
#[derive(Debug, Clone, Copy)]
pub enum BasinShape {
    Spherical,
    Ellipsoidal,
    Irregular,
    Narrow,
    Wide,
}
/// Local minima detector
#[derive(Debug)]
pub struct LocalMinimaDetector<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Detection threshold
    threshold: T,
    /// Detected minima
    detected_minima: Vec<LocalMinimum<T>>,
    /// Detection algorithm
    algorithm: MinimaDetectionAlgorithm,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    LocalMinimaDetector<T>
{
    fn new() -> Self {
        Self {
            threshold: scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero()),
            detected_minima: Vec::new(),
            algorithm: MinimaDetectionAlgorithm::GradientBased,
        }
    }
}
/// Compression parameters
#[derive(Debug, Clone)]
pub struct CompressionParams<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Target compression ratio
    pub(super) target_ratio: T,
    /// Quality threshold
    pub(super) quality_threshold: T,
    /// Maximum compression time
    pub(super) max_time: u64,
    /// Compression strength
    pub(super) strength: T,
}
/// Pattern recognizer
#[derive(Debug)]
pub struct PatternRecognizer<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Recognized patterns
    patterns: Vec<OptimizationPattern<T>>,
    /// Pattern library
    pattern_library: PatternLibrary<T>,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    PatternRecognizer<T>
{
    fn new() -> Self {
        Self {
            patterns: Vec::new(),
            pattern_library: PatternLibrary {
                patterns: HashMap::new(),
                pattern_index: HashMap::new(),
                usage_stats: HashMap::new(),
            },
        }
    }
}
/// Gradient characteristics
#[derive(Debug, Clone)]
pub struct GradientCharacteristics<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Gradient norm
    pub(super) gradient_norm: T,
    /// Gradient consistency
    pub(super) consistency: T,
    /// Gradient noise ratio
    pub(super) noise_ratio: T,
    /// Gradient correlation
    pub(super) correlation: T,
}
/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Mish,
    ELU,
    Tanh,
}
