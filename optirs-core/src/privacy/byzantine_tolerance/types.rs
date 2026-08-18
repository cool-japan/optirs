//! Supporting types for [`super::aggregator::ByzantineTolerantAggregator`]:
//! configuration, participant/reputation bookkeeping, the anomaly and
//! statistical-analysis engines, gradient verification, and result types.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

use super::helpers::{
    cosine_similarity, euclidean_distance, from_scalar, l2_norm, to_scalar, RuleFn,
    HISTORY_CAPACITY, MAX_PATTERNS,
};

/// Trust levels for participants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustLevel {
    /// Highly trusted participant
    High,
    /// Moderately trusted participant
    Medium,
    /// Low trust participant
    Low,
    /// Blacklisted participant
    Blacklisted,
}
/// Configuration for [`super::aggregator::ByzantineTolerantAggregator`].
///
/// Use [`ByzantineConfig::validate`] (called automatically by
/// [`super::aggregator::ByzantineTolerantAggregator::new`]) to reject inconsistent settings before
/// they can produce meaningless scores.
#[derive(Debug, Clone)]
pub struct ByzantineConfig {
    /// Maximum number of Byzantine participants to tolerate (`f`).
    pub max_byzantine: usize,
    /// Minimum number of participants required (`n`).
    pub min_participants: usize,
    /// Aggregation method for Byzantine tolerance
    pub aggregation_method: ByzantineAggregationMethod,
    /// Threshold on the combined Byzantine score, in `(0, 1]`. Participants whose
    /// score exceeds it are treated as Byzantine.
    pub anomaly_threshold: f64,
    /// Reputation decay factor in `[0, 1)`.
    ///
    /// Reputation is an exponential moving average of a per-round observation
    /// (`1.0` for honest, `0.0` for Byzantine): `score <- decay * score +
    /// (1 - decay) * observation`. Byzantine rounds additionally use a five-fold
    /// update rate (capped at `1.0`) so misbehaviour is punished faster than good
    /// behaviour is rewarded.
    pub reputation_decay: f64,
    /// Enable gradient verification
    pub gradient_verification: bool,
    /// Statistical outlier detection
    pub outlier_detection: OutlierDetectionMethod,
    /// Minimum fraction of the submitting cohort that must survive detection for
    /// the round to be accepted, in `(0, 1]`.
    pub consensus_threshold: f64,
}
impl Default for ByzantineConfig {
    fn default() -> Self {
        Self {
            max_byzantine: 1,
            min_participants: 5,
            aggregation_method: ByzantineAggregationMethod::CoordinateMedian,
            anomaly_threshold: 0.5,
            reputation_decay: 0.9,
            gradient_verification: true,
            outlier_detection: OutlierDetectionMethod::ZScore,
            consensus_threshold: 0.5,
        }
    }
}
impl ByzantineConfig {
    /// Minimum cohort size required by `aggregation_method` for its published
    /// robustness guarantee, given `max_byzantine`.
    pub fn required_participants(&self) -> usize {
        let f = self.max_byzantine;
        match self.aggregation_method {
            ByzantineAggregationMethod::Krum | ByzantineAggregationMethod::MultiKrum => 2 * f + 3,
            ByzantineAggregationMethod::Bulyan => 4 * f + 3,
            ByzantineAggregationMethod::TrimmedMean => 2 * f + 1,
            ByzantineAggregationMethod::CoordinateMedian
            | ByzantineAggregationMethod::Median
            | ByzantineAggregationMethod::GeometricMedian
            | ByzantineAggregationMethod::FoolsGold
            | ByzantineAggregationMethod::FLAME => 2 * f + 1,
        }
    }
    /// Validate the configuration.
    ///
    /// Rejects the states that used to produce `NaN` confidence scores, silent
    /// no-ops or `usize` underflow inside the Krum family.
    pub fn validate(&self) -> Result<()> {
        if self.min_participants == 0 {
            return Err(OptimError::InvalidConfig(
                "min_participants must be at least 1".to_string(),
            ));
        }
        if !self.anomaly_threshold.is_finite()
            || self.anomaly_threshold <= 0.0
            || self.anomaly_threshold > 1.0
        {
            return Err(OptimError::InvalidConfig(format!(
                "anomaly_threshold must lie in (0, 1], got {}",
                self.anomaly_threshold
            )));
        }
        if !self.reputation_decay.is_finite()
            || self.reputation_decay < 0.0
            || self.reputation_decay >= 1.0
        {
            return Err(OptimError::InvalidConfig(format!(
                "reputation_decay must lie in [0, 1), got {}",
                self.reputation_decay
            )));
        }
        if !self.consensus_threshold.is_finite()
            || self.consensus_threshold <= 0.0
            || self.consensus_threshold > 1.0
        {
            return Err(OptimError::InvalidConfig(format!(
                "consensus_threshold must lie in (0, 1], got {}",
                self.consensus_threshold
            )));
        }
        let required = self.required_participants();
        if self.min_participants < required {
            return Err(OptimError::InvalidConfig(format!(
                "{:?} tolerating {} Byzantine participants requires at least {} participants, \
                 but min_participants is {}",
                self.aggregation_method, self.max_byzantine, required, self.min_participants
            )));
        }
        Ok(())
    }
}
/// Expected gradient properties
#[derive(Debug, Clone)]
pub struct GradientProperties<T: Float + Debug + Send + Sync + 'static> {
    /// Accepted L2 norm range, inclusive
    pub norm_range: (T, T),
    /// Minimum fraction of non-zero coordinates expected in a well-formed update
    pub sparsity_threshold: f64,
    /// Minimum cosine similarity with the previous round's aggregate. The default
    /// of `0.0` only rejects updates pointing against the consensus direction.
    pub direction_consistency: f64,
}
impl<T: Float + Debug + Send + Sync + 'static> Default for GradientProperties<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Float + Debug + Send + Sync + 'static> GradientProperties<T> {
    /// Create new gradient properties.
    ///
    /// Returns permissive defaults when the requested constants are not
    /// representable in `T`.
    pub fn new() -> Self {
        Self {
            norm_range: (T::zero(), T::from(100.0).unwrap_or_else(T::max_value)),
            sparsity_threshold: 0.1,
            direction_consistency: 0.0,
        }
    }
}
/// Per-participant gradient statistics used for temporal anomaly detection.
#[derive(Debug, Clone)]
pub struct GradientStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Running mean gradient per participant (exponential moving average)
    pub mean: HashMap<String, Array1<T>>,
    /// Historical gradient L2 norms per participant
    pub norm_history: HashMap<String, Vec<T>>,
}
impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand> Default
    for GradientStatistics<T>
{
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    GradientStatistics<T>
{
    /// Create new gradient statistics
    pub fn new() -> Self {
        Self {
            mean: HashMap::new(),
            norm_history: HashMap::new(),
        }
    }
    /// Recorded norm history of a participant.
    pub fn norm_history_of(&self, participant_id: &str) -> Option<&[T]> {
        self.norm_history.get(participant_id).map(|v| v.as_slice())
    }
    /// Running mean update of a participant.
    pub fn mean_of(&self, participant_id: &str) -> Option<&Array1<T>> {
        self.mean.get(participant_id)
    }
    /// Update the statistics of `participant_id` with a new gradient.
    pub fn update(&mut self, participant_id: &str, gradient: &Array1<T>) -> Result<()> {
        let norm = l2_norm(gradient);
        let history = self
            .norm_history
            .entry(participant_id.to_string())
            .or_default();
        history.push(norm);
        if history.len() > HISTORY_CAPACITY {
            history.remove(0);
        }
        let alpha: T = to_scalar(0.01)?;
        match self.mean.get(participant_id) {
            Some(mean) if mean.len() == gradient.len() => {
                let updated = mean * (T::one() - alpha) + gradient * alpha;
                self.mean.insert(participant_id.to_string(), updated);
            }
            _ => {
                self.mean
                    .insert(participant_id.to_string(), gradient.clone());
            }
        }
        Ok(())
    }
}
/// Deterministic SplitMix64 pseudo-random generator.
///
/// Used for the randomised parts of this module (isolation-forest splits and
/// FLAME's noise calibration). It is deliberately *not* a cryptographic generator:
/// its purpose is reproducibility, not secrecy.
#[derive(Debug, Clone)]
pub(super) struct SplitMix64 {
    pub(super) state: u64,
}
impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform sample in `[0, 1)`.
    pub(super) fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }
    /// Uniform sample in `[0, n)`; returns `0` when `n == 0`.
    pub(super) fn next_range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
    /// Standard normal sample via the Box-Muller transform.
    pub(super) fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
/// Gradient verification system
pub struct GradientVerifier<T: Float + Debug + Send + Sync + 'static> {
    /// Expected gradient properties
    pub(super) expected_properties: GradientProperties<T>,
    /// Verification rules
    pub(super) verification_rules: Vec<VerificationRule<T>>,
    /// Aggregate accepted in the previous round, used as the reference direction
    /// for the direction-consistency rule
    pub(super) reference_direction: Option<Array1<T>>,
}
impl<T: Float + Debug + Send + Sync + 'static> Default for GradientVerifier<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Float + Debug + Send + Sync + 'static> GradientVerifier<T> {
    /// Create new gradient verifier with the default expected properties.
    pub fn new() -> Self {
        Self::with_properties(GradientProperties::new())
    }
    /// Create a verifier whose rules enforce `properties`.
    pub fn with_properties(properties: GradientProperties<T>) -> Self {
        let (min_norm, max_norm) = properties.norm_range;
        let sparsity_threshold = properties.sparsity_threshold;
        let verification_rules = vec![
            VerificationRule {
                name: "Finite values".to_string(),
                rule_fn: Box::new(|gradient: &Array1<T>| gradient.iter().all(|&x| x.is_finite())),
                weight: 1.0,
            },
            VerificationRule {
                name: "Norm range".to_string(),
                rule_fn: Box::new(move |gradient: &Array1<T>| {
                    let norm = l2_norm(gradient);
                    norm >= min_norm && norm <= max_norm
                }),
                weight: 0.25,
            },
            VerificationRule {
                name: "Sparsity".to_string(),
                rule_fn: Box::new(move |gradient: &Array1<T>| {
                    if gradient.is_empty() {
                        return false;
                    }
                    let non_zero = gradient.iter().filter(|x| **x != T::zero()).count();
                    non_zero as f64 / gradient.len() as f64 >= sparsity_threshold
                }),
                weight: 0.25,
            },
        ];
        Self {
            expected_properties: properties,
            verification_rules,
            reference_direction: None,
        }
    }
    /// Properties enforced by this verifier.
    pub fn expected_properties(&self) -> &GradientProperties<T> {
        &self.expected_properties
    }
    /// Set the reference direction used by the direction-consistency check,
    /// normally the aggregate accepted in the previous round.
    pub fn set_reference_direction(&mut self, direction: Array1<T>) {
        self.reference_direction = Some(direction);
    }
    /// Verify a gradient against all rules and, when a reference direction is
    /// available, against `expected_properties.direction_consistency`.
    pub fn verify_gradient(&self, gradient: &Array1<T>) -> Result<VerificationScore> {
        let mut rule_scores = HashMap::new();
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;
        for rule in &self.verification_rules {
            let score = if (rule.rule_fn)(gradient) { 1.0 } else { 0.0 };
            rule_scores.insert(rule.name.clone(), score);
            weighted_score += score * rule.weight;
            total_weight += rule.weight;
        }
        if let Some(reference) = &self.reference_direction {
            if reference.len() == gradient.len() {
                let similarity = from_scalar(cosine_similarity(reference, gradient)?)?;
                let score = if similarity >= self.expected_properties.direction_consistency {
                    1.0
                } else {
                    0.0
                };
                rule_scores.insert("Direction consistency".to_string(), score);
                weighted_score += score * 0.25;
                total_weight += 0.25;
            }
        }
        let overall_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            1.0
        };
        Ok(VerificationScore {
            score: overall_score,
            rule_scores,
            passed: overall_score >= 0.8,
        })
    }
}
/// Pattern recognition model for detecting malicious behaviour.
///
/// Prototypes are unit-norm gradient directions learned online: honest updates
/// accepted by a round feed `normal_patterns`, updates rejected as Byzantine feed
/// `attack_patterns`.
pub struct PatternModel<T: Float + Debug + Send + Sync + 'static> {
    /// Reference patterns for normal behavior
    pub(super) normal_patterns: Vec<Array1<T>>,
    /// Reference patterns for attack behaviors
    pub(super) attack_patterns: Vec<Array1<T>>,
    /// Cosine similarity above which an observation is merged into an existing
    /// prototype instead of creating a new one
    pub(super) matching_threshold: f64,
}
impl<T: Float + Debug + Send + Sync + 'static> Default for PatternModel<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Float + Debug + Send + Sync + 'static> PatternModel<T> {
    /// Create new pattern model
    pub fn new() -> Self {
        Self {
            normal_patterns: Vec::new(),
            attack_patterns: Vec::new(),
            matching_threshold: 0.8,
        }
    }
    /// Cosine similarity above which an observation is merged into an existing
    /// prototype instead of creating a new one.
    pub fn matching_threshold(&self) -> f64 {
        self.matching_threshold
    }
    /// Number of learned normal and attack prototypes.
    pub fn pattern_counts(&self) -> (usize, usize) {
        (self.normal_patterns.len(), self.attack_patterns.len())
    }
    /// Learn a normal behaviour prototype.
    pub fn learn_normal(&mut self, gradient: &Array1<T>) -> Result<()> {
        let threshold = self.matching_threshold;
        Self::learn_into(&mut self.normal_patterns, gradient, threshold)
    }
    /// Learn an attack prototype.
    pub fn learn_attack(&mut self, gradient: &Array1<T>) -> Result<()> {
        let threshold = self.matching_threshold;
        Self::learn_into(&mut self.attack_patterns, gradient, threshold)
    }
    /// Merge `gradient`'s direction into `patterns`.
    ///
    /// Prototypes are unit-norm directions. An observation whose cosine similarity
    /// to the closest prototype reaches `matching_threshold` is merged into it with
    /// an exponential moving average; otherwise it becomes a new prototype. At most
    /// [`MAX_PATTERNS`] prototypes are retained, the oldest being evicted first.
    pub(super) fn learn_into(
        patterns: &mut Vec<Array1<T>>,
        gradient: &Array1<T>,
        matching_threshold: f64,
    ) -> Result<()> {
        let norm = l2_norm(gradient);
        if !norm.is_finite() || norm <= T::zero() {
            return Ok(());
        }
        let direction: Array1<T> = gradient.map(|&x| x / norm);
        let threshold: T = to_scalar(matching_threshold)?;
        let alpha: T = to_scalar(0.2)?;
        let mut best: Option<(usize, T)> = None;
        for (index, pattern) in patterns.iter().enumerate() {
            if pattern.len() != direction.len() {
                continue;
            }
            let similarity = cosine_similarity(pattern, &direction)?;
            if best.is_none_or(|(_, current)| similarity > current) {
                best = Some((index, similarity));
            }
        }
        match best {
            Some((index, similarity)) if similarity >= threshold => {
                let merged: Array1<T> = patterns[index]
                    .iter()
                    .zip(direction.iter())
                    .map(|(&p, &d)| p * (T::one() - alpha) + d * alpha)
                    .collect();
                let merged_norm = l2_norm(&merged);
                patterns[index] = if merged_norm > T::zero() {
                    merged.map(|&x| x / merged_norm)
                } else {
                    direction
                };
            }
            _ => {
                patterns.push(direction);
                if patterns.len() > MAX_PATTERNS {
                    patterns.remove(0);
                }
            }
        }
        Ok(())
    }
    /// Pattern deviation score in `[0, 1]`, or `None` when nothing comparable has
    /// been learned yet.
    ///
    /// The score is the larger of the novelty relative to the closest normal
    /// prototype and the alignment with the closest known attack prototype. `None`
    /// is returned whenever *no* prototype is dimension-compatible with `gradient`
    /// (including after a model reshape, when prototypes exist but none of them
    /// applies), so the caller never folds a constant into a combined score.
    pub fn compute_pattern_deviation(&self, gradient: &Array1<T>) -> Result<Option<f64>> {
        let normal = Self::best_similarity(&self.normal_patterns, gradient)?;
        let attack = Self::best_similarity(&self.attack_patterns, gradient)?;
        if normal.is_none() && attack.is_none() {
            return Ok(None);
        }
        let novelty = normal.map_or(0.0, |similarity| ((1.0 - similarity) / 2.0).clamp(0.0, 1.0));
        let attack_match = attack.map_or(0.0, |similarity| similarity.clamp(0.0, 1.0));
        Ok(Some(novelty.max(attack_match)))
    }
    /// Highest cosine similarity between `gradient` and any compatible prototype.
    pub(super) fn best_similarity(
        patterns: &[Array1<T>],
        gradient: &Array1<T>,
    ) -> Result<Option<f64>> {
        let mut best: Option<f64> = None;
        for pattern in patterns {
            if pattern.len() != gradient.len() {
                continue;
            }
            let similarity = from_scalar(cosine_similarity(pattern, gradient)?)?;
            if best.is_none_or(|current| similarity > current) {
                best = Some(similarity);
            }
        }
        Ok(best)
    }
    /// Euclidean distance between a gradient and a prototype.
    pub fn compute_pattern_distance(&self, gradient: &Array1<T>, pattern: &Array1<T>) -> Result<T> {
        euclidean_distance(gradient, pattern)
    }
}
/// Verification rule for gradients
pub struct VerificationRule<T: Float + Debug + Send + Sync + 'static> {
    /// Rule name
    pub name: String,
    /// Rule function
    pub rule_fn: RuleFn<T>,
    /// Rule weight in verification
    pub weight: f64,
}
/// Outlier score for participant
#[derive(Debug, Clone)]
pub struct OutlierScore {
    /// Outlier score in `[0, 1]`, higher meaning more outlying
    pub score: f64,
    /// Detection method used
    pub method: OutlierDetectionMethod,
    /// Additional details
    pub details: String,
}
/// Byzantine aggregation result
#[derive(Debug, Clone)]
pub struct ByzantineAggregationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Aggregated gradient
    pub aggregate: Array1<T>,
    /// List of honest participants, sorted by id
    pub honest_participants: Vec<String>,
    /// List of detected Byzantine participants, sorted by id
    pub byzantine_participants: Vec<String>,
    /// Updated reputation scores
    pub reputation_updates: HashMap<String, ReputationScore>,
    /// Aggregation method used
    pub aggregation_method: ByzantineAggregationMethod,
    /// Fraction of the submitting cohort that survived detection
    pub consensus_ratio: f64,
    /// Confidence score of the aggregation, in `[0, 1]`
    pub confidence_score: f64,
}
/// Byzantine-robust aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByzantineAggregationMethod {
    /// Coordinate-wise trimmed mean (Yin et al.), trimming `max_byzantine` values
    /// from each tail.
    TrimmedMean,
    /// Coordinate-wise median
    CoordinateMedian,
    /// Krum algorithm (select most representative gradient)
    Krum,
    /// Multi-Krum (average the `n - f` most representative gradients)
    MultiKrum,
    /// Bulyan (iterative Krum selection followed by median-proximity averaging)
    Bulyan,
    /// FoolsGold (defend against Sybil attacks)
    FoolsGold,
    /// FLAME (clustering, norm-median clipping and calibrated noise)
    FLAME,
    /// Median-based aggregation
    Median,
    /// Geometric median
    GeometricMedian,
}
/// Behavior history for participants, recorded once per aggregation round.
#[derive(Debug, Clone, Default)]
pub struct BehaviorHistory {
    /// History of gradient norms
    pub gradient_norms: Vec<f64>,
    /// History of cosine similarities with the accepted aggregate
    pub gradient_similarities: Vec<f64>,
    /// History of participation patterns (`false` when the participant submitted
    /// but was filtered out before aggregation)
    pub participation_pattern: Vec<bool>,
    /// History of anomaly scores
    pub anomaly_scores: Vec<f64>,
    /// Number of rounds participated
    pub rounds_participated: usize,
}
impl BehaviorHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self::default()
    }
    /// Mean of the recorded anomaly scores, or `None` when nothing was recorded.
    pub fn mean_anomaly_score(&self) -> Option<f64> {
        if self.anomaly_scores.is_empty() {
            return None;
        }
        Some(self.anomaly_scores.iter().sum::<f64>() / self.anomaly_scores.len() as f64)
    }
    /// Mean cosine similarity with the accepted aggregate, or `None`.
    pub fn mean_similarity(&self) -> Option<f64> {
        if self.gradient_similarities.is_empty() {
            return None;
        }
        Some(
            self.gradient_similarities.iter().sum::<f64>()
                / self.gradient_similarities.len() as f64,
        )
    }
    pub(super) fn push_bounded<V>(values: &mut Vec<V>, value: V) {
        values.push(value);
        if values.len() > HISTORY_CAPACITY {
            values.remove(0);
        }
    }
}
/// Anomaly detection engine
pub struct AnomalyDetector<T: Float + Debug + Send + Sync + 'static> {
    /// Score above which a participant is reported as anomalous
    pub(super) threshold: f64,
    /// Historical gradient statistics, keyed per participant
    pub(super) gradient_stats: GradientStatistics<T>,
    /// Pattern recognition model
    pub(super) pattern_model: PatternModel<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    AnomalyDetector<T>
{
    /// Create new anomaly detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            gradient_stats: GradientStatistics::new(),
            pattern_model: PatternModel::new(),
        }
    }
    /// Score above which a participant is reported as anomalous.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
    /// Historical statistics, keyed per participant.
    pub fn gradient_stats(&self) -> &GradientStatistics<T> {
        &self.gradient_stats
    }
    /// Number of learned normal and attack prototypes.
    pub fn pattern_counts(&self) -> (usize, usize) {
        self.pattern_model.pattern_counts()
    }
    /// Detect an anomaly in `gradient`, attributed to `participant_id`.
    ///
    /// The deviation is measured against the participant's *own* history and is
    /// computed **before** the new sample is recorded, so a gradient is never part
    /// of the baseline it is judged against, and one participant's outlier cannot
    /// inflate everyone else's baseline.
    ///
    /// A participant with fewer than three recorded rounds has no usable temporal
    /// baseline and scores `0.0`; cross-sectional detection of first-round attacks
    /// is the statistical outlier detector's job.
    pub fn detect_anomaly(
        &mut self,
        participant_id: &str,
        gradient: &Array1<T>,
    ) -> Result<AnomalyScore> {
        let norm_deviation = self.compute_norm_deviation(participant_id, gradient)?;
        let pattern_deviation = self.pattern_model.compute_pattern_deviation(gradient)?;
        let (combined_score, details) = match pattern_deviation {
            Some(pattern) => (
                (norm_deviation + pattern) / 2.0,
                format!("Norm dev: {norm_deviation:.4}, Pattern dev: {pattern:.4}"),
            ),
            None => (
                norm_deviation,
                format!("Norm dev: {norm_deviation:.4}, no learned patterns yet"),
            ),
        };
        let combined_score = combined_score.clamp(0.0, 1.0);
        self.gradient_stats.update(participant_id, gradient)?;
        Ok(AnomalyScore {
            score: combined_score,
            is_anomalous: combined_score > self.threshold,
            method: "Combined norm and pattern analysis".to_string(),
            details,
        })
    }
    /// Learn a normal behaviour prototype from an accepted update.
    pub fn learn_normal(&mut self, gradient: &Array1<T>) -> Result<()> {
        self.pattern_model.learn_normal(gradient)
    }
    /// Learn an attack prototype from a rejected update.
    pub fn learn_attack(&mut self, gradient: &Array1<T>) -> Result<()> {
        self.pattern_model.learn_attack(gradient)
    }
    /// Norm deviation score in `[0, 1]`, relative to the participant's own history.
    pub(super) fn compute_norm_deviation(
        &self,
        participant_id: &str,
        gradient: &Array1<T>,
    ) -> Result<f64> {
        let Some(history) = self.gradient_stats.norm_history.get(participant_id) else {
            return Ok(0.0);
        };
        if history.len() < 3 {
            return Ok(0.0);
        }
        let gradient_norm = l2_norm(gradient);
        let count: T = to_scalar(history.len() as f64)?;
        let mean_norm = history.iter().fold(T::zero(), |acc, &x| acc + x) / count;
        let variance = history
            .iter()
            .map(|&x| {
                let diff = x - mean_norm;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
            / count;
        let std_norm = variance.sqrt();
        if std_norm > T::zero() {
            let z_score = from_scalar(((gradient_norm - mean_norm) / std_norm).abs())?;
            Ok((z_score / 3.0).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
    /// Compute L2 norm of gradient
    pub fn compute_l2_norm(&self, gradient: &Array1<T>) -> T {
        l2_norm(gradient)
    }
}
/// Reputation score for participants
#[derive(Debug, Clone)]
pub struct ReputationScore {
    /// Current reputation score (0.0 to 1.0)
    pub score: f64,
    /// Number of successful aggregations
    pub successful_aggregations: usize,
    /// Number of detected anomalies
    pub detected_anomalies: usize,
    /// Average gradient quality score, derived from the participant's recorded
    /// anomaly history (`1 - mean(anomaly_score)`).
    pub gradient_quality: f64,
    /// Consistency score across rounds, derived from the participant's recorded
    /// cosine similarity with the accepted aggregate, mapped to `[0, 1]`.
    pub consistency_score: f64,
    /// Trust level
    pub trust_level: TrustLevel,
}
impl Default for ReputationScore {
    fn default() -> Self {
        Self::new()
    }
}
impl ReputationScore {
    /// Create new reputation score with default values
    pub fn new() -> Self {
        Self {
            score: 0.7,
            successful_aggregations: 0,
            detected_anomalies: 0,
            gradient_quality: 0.5,
            consistency_score: 0.5,
            trust_level: TrustLevel::Medium,
        }
    }
}
/// Verification score for gradient
#[derive(Debug, Clone)]
pub struct VerificationScore {
    /// Verification score (0.0 = failed, 1.0 = passed)
    pub score: f64,
    /// Individual rule scores
    pub rule_scores: HashMap<String, f64>,
    /// Overall verification status
    pub passed: bool,
}
/// Cross-sectional statistical measures of one round's gradients.
#[derive(Debug, Clone)]
pub struct StatisticalMeasures<T: Float + Debug + Send + Sync + 'static> {
    /// Mean of gradients
    pub mean: Array1<T>,
    /// Standard deviation (population)
    pub std_dev: Array1<T>,
    /// Median
    pub median: Array1<T>,
    /// First quartile
    pub q1: Array1<T>,
    /// Third quartile
    pub q3: Array1<T>,
    /// Interquartile range (`q3 - q1`)
    pub iqr: Array1<T>,
    /// Standardised third moment. `0` for a constant coordinate.
    pub skewness: Array1<T>,
    /// Standardised fourth moment (Pearson kurtosis, `3` for a normal
    /// distribution). `3` is reported for a constant coordinate, where the
    /// quantity is undefined.
    pub kurtosis: Array1<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> Default for StatisticalMeasures<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Float + Debug + Send + Sync + 'static> StatisticalMeasures<T> {
    /// Create new statistical measures
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(0),
            std_dev: Array1::zeros(0),
            median: Array1::zeros(0),
            q1: Array1::zeros(0),
            q3: Array1::zeros(0),
            iqr: Array1::zeros(0),
            skewness: Array1::zeros(0),
            kurtosis: Array1::zeros(0),
        }
    }
}
/// Statistical analysis engine
pub struct StatisticalAnalysis<T: Float + Debug + Send + Sync + 'static> {
    /// Statistical measures of the most recently analysed cohort
    pub(super) measures: StatisticalMeasures<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> Default for StatisticalAnalysis<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Float + Debug + Send + Sync + 'static> StatisticalAnalysis<T> {
    /// Create new statistical analysis engine
    pub fn new() -> Self {
        Self {
            measures: StatisticalMeasures::new(),
        }
    }
    /// Measures of the most recently analysed cohort.
    pub fn measures(&self) -> &StatisticalMeasures<T> {
        &self.measures
    }
    /// Compute cross-sectional statistical measures for one round's gradients.
    pub fn compute_statistics(
        &mut self,
        gradients: &[&Array1<T>],
    ) -> Result<StatisticalMeasures<T>> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients provided".to_string(),
            ));
        }
        let dim = gradients[0].len();
        for gradient in gradients {
            if gradient.len() != dim {
                return Err(OptimError::DimensionMismatch(format!(
                    "gradient of length {} in a cohort of dimension {dim}",
                    gradient.len()
                )));
            }
        }
        let mut mean = Array1::zeros(dim);
        let mut median = Array1::zeros(dim);
        let mut std_dev = Array1::zeros(dim);
        let mut q1 = Array1::zeros(dim);
        let mut q3 = Array1::zeros(dim);
        let mut iqr = Array1::zeros(dim);
        let mut skewness = Array1::zeros(dim);
        let mut kurtosis = Array1::zeros(dim);
        let count: T = to_scalar(gradients.len() as f64)?;
        let two: T = to_scalar(2.0)?;
        let normal_kurtosis: T = to_scalar(3.0)?;
        for i in 0..dim {
            let mut values: Vec<T> = gradients.iter().map(|g| g[i]).collect();
            let sum: T = values.iter().copied().fold(T::zero(), |acc, x| acc + x);
            mean[i] = sum / count;
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            median[i] = if values.len().is_multiple_of(2) {
                let mid = values.len() / 2;
                (values[mid - 1] + values[mid]) / two
            } else {
                values[values.len() / 2]
            };
            let variance: T = values
                .iter()
                .map(|&x| {
                    let diff = x - mean[i];
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
                / count;
            std_dev[i] = variance.sqrt();
            let q1_idx = values.len() / 4;
            let q3_idx = (3 * values.len() / 4).min(values.len() - 1);
            q1[i] = values[q1_idx];
            q3[i] = values[q3_idx];
            iqr[i] = q3[i] - q1[i];
            if std_dev[i] > T::zero() {
                let mut third = T::zero();
                let mut fourth = T::zero();
                for &value in &values {
                    let z = (value - mean[i]) / std_dev[i];
                    third = third + z * z * z;
                    fourth = fourth + z * z * z * z;
                }
                skewness[i] = third / count;
                kurtosis[i] = fourth / count;
            } else {
                skewness[i] = T::zero();
                kurtosis[i] = normal_kurtosis;
            }
        }
        self.measures = StatisticalMeasures {
            mean,
            std_dev,
            median,
            q1,
            q3,
            iqr,
            skewness,
            kurtosis,
        };
        Ok(self.measures.clone())
    }
}
/// Outlier detection methods.
///
/// Every method reports a score in `[0, 1]`, higher meaning more outlying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierDetectionMethod {
    /// Maximum per-coordinate z-score, normalised by the three-sigma rule.
    ZScore,
    /// Interquartile range method with a standard-deviation fallback for
    /// degenerate (zero-IQR) coordinates.
    IQR,
    /// Isolation forest (Liu et al.) over the current round's cohort.
    IsolationForest,
    /// Local outlier factor (Breunig et al.) over the current round's cohort.
    LocalOutlierFactor,
    /// Diagonal-covariance Mahalanobis distance (root-mean-square z-score).
    ///
    /// Federated cohorts are far smaller than the gradient dimension, so the full
    /// covariance matrix is singular; the diagonal form is used instead.
    MahalanobisDistance,
}
/// Anomaly score for participant
#[derive(Debug, Clone)]
pub struct AnomalyScore {
    /// Anomaly score in `[0, 1]` (0.0 = normal, 1.0 = highly anomalous)
    pub score: f64,
    /// Whether `score` exceeds the detector's threshold
    pub is_anomalous: bool,
    /// Detection method used
    pub method: String,
    /// Additional details
    pub details: String,
}
