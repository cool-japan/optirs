// Byzantine Fault Tolerance for Federated Learning
//
// This module implements Byzantine-robust aggregation algorithms that can
// tolerate malicious participants in federated learning scenarios.
//
// # Determinism
//
// Every algorithm in this module orders its cohort by participant id before doing
// any work, so results never depend on `HashMap` iteration order. The two
// randomised components (FLAME's calibrated noise and the isolation-forest outlier
// detector) draw from an internal, deterministically seeded SplitMix64 generator.
//
// # Threat model notes
//
// The FLAME noise term is a *backdoor mitigation* measure taken from Nguyen et al.,
// 2022. It is **not** a differential-privacy mechanism and provides no formal
// privacy guarantee; use [`crate::privacy`]'s differential privacy machinery for
// that.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

/// Type alias for validation rule function
type RuleFn<T> = Box<dyn Fn(&Array1<T>) -> bool + Send + Sync>;

/// Maximum number of per-participant history samples retained.
const HISTORY_CAPACITY: usize = 1000;

/// Maximum number of behaviour prototypes retained by [`PatternModel`].
const MAX_PATTERNS: usize = 32;

/// Noise multiplier used by the FLAME aggregator (lambda in Nguyen et al., 2022).
const FLAME_NOISE_LAMBDA: f64 = 0.001;

/// Number of trees built by the isolation-forest outlier detector.
const ISOLATION_TREES: usize = 100;

/// Euler-Mascheroni constant, used by the isolation-forest path-length normaliser.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Convert an `f64` constant into `T`, reporting an error when the target
/// floating point type cannot represent it.
fn to_scalar<T: Float>(value: f64) -> Result<T> {
    T::from(value).ok_or_else(|| {
        OptimError::ComputationError(format!(
            "value {value} is not representable in the target floating point type"
        ))
    })
}

/// Convert a `T` value into `f64`, reporting an error when the conversion fails.
fn from_scalar<T: Float>(value: T) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        OptimError::ComputationError("floating point value is not convertible to f64".to_string())
    })
}

/// L2 norm of a gradient.
fn l2_norm<T: Float>(gradient: &Array1<T>) -> T {
    gradient
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt()
}

/// Euclidean distance between two gradients.
fn euclidean_distance<T: Float>(a: &Array1<T>, b: &Array1<T>) -> Result<T> {
    if a.len() != b.len() {
        return Err(OptimError::DimensionMismatch(format!(
            "gradient dimensions don't match: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let mut sum = T::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }

    Ok(sum.sqrt())
}

/// Cosine similarity between two gradients. Returns `0` when either operand has a
/// zero norm (the angle is undefined in that case).
fn cosine_similarity<T: Float>(a: &Array1<T>, b: &Array1<T>) -> Result<T> {
    if a.len() != b.len() {
        return Err(OptimError::DimensionMismatch(format!(
            "gradient dimensions don't match: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let mut dot_product = T::zero();
    let mut norm_a = T::zero();
    let mut norm_b = T::zero();

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product = dot_product + *x * *y;
        norm_a = norm_a + *x * *x;
        norm_b = norm_b + *y * *y;
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a > T::zero() && norm_b > T::zero() {
        Ok(dot_product / (norm_a * norm_b))
    } else {
        Ok(T::zero())
    }
}

/// Convert a gradient into an `f64` vector for the detectors that work in `f64`.
fn to_f64_vec<T: Float>(gradient: &Array1<T>) -> Result<Vec<f64>> {
    gradient.iter().copied().map(from_scalar).collect()
}

/// Deterministic SplitMix64 pseudo-random generator.
///
/// Used for the randomised parts of this module (isolation-forest splits and
/// FLAME's noise calibration). It is deliberately *not* a cryptographic generator:
/// its purpose is reproducibility, not secrecy.
#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Uniform sample in `[0, n)`; returns `0` when `n == 0`.
    fn next_range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Standard normal sample via the Box-Muller transform.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Byzantine fault tolerant aggregator
pub struct ByzantineTolerantAggregator<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for Byzantine tolerance
    config: ByzantineConfig,

    /// Participant reputation scores
    reputation_scores: HashMap<String, ReputationScore>,

    /// History of participant behavior
    behavior_history: HashMap<String, BehaviorHistory>,

    /// Anomaly detection engine
    anomaly_detector: AnomalyDetector<T>,

    /// Statistical analysis engine
    statistics_engine: StatisticalAnalysis<T>,

    /// Gradient verification system
    gradient_verifier: GradientVerifier<T>,

    /// Per-participant sum of all historical updates, used by FoolsGold.
    fools_gold_history: HashMap<String, Array1<T>>,

    /// Number of completed aggregation rounds (seeds FLAME's noise).
    round: u64,
}

/// Configuration for [`ByzantineTolerantAggregator`].
///
/// Use [`ByzantineConfig::validate`] (called automatically by
/// [`ByzantineTolerantAggregator::new`]) to reject inconsistent settings before
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
            // Blanchard et al.: Krum needs n >= 2f + 3.
            ByzantineAggregationMethod::Krum | ByzantineAggregationMethod::MultiKrum => 2 * f + 3,
            // Mhamdi et al.: Bulyan needs n >= 4f + 3.
            ByzantineAggregationMethod::Bulyan => 4 * f + 3,
            // Yin et al.: the trimmed mean needs n > 2f so at least one value survives.
            ByzantineAggregationMethod::TrimmedMean => 2 * f + 1,
            // Median-style and reweighting estimators need a strict honest majority.
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

    fn push_bounded<V>(values: &mut Vec<V>, value: V) {
        values.push(value);
        if values.len() > HISTORY_CAPACITY {
            values.remove(0);
        }
    }
}

/// Anomaly detection engine
pub struct AnomalyDetector<T: Float + Debug + Send + Sync + 'static> {
    /// Score above which a participant is reported as anomalous
    threshold: f64,

    /// Historical gradient statistics, keyed per participant
    gradient_stats: GradientStatistics<T>,

    /// Pattern recognition model
    pattern_model: PatternModel<T>,
}

/// Per-participant gradient statistics used for temporal anomaly detection.
#[derive(Debug, Clone)]
pub struct GradientStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Running mean gradient per participant (exponential moving average)
    pub mean: HashMap<String, Array1<T>>,

    /// Historical gradient L2 norms per participant
    pub norm_history: HashMap<String, Vec<T>>,
}

/// Pattern recognition model for detecting malicious behaviour.
///
/// Prototypes are unit-norm gradient directions learned online: honest updates
/// accepted by a round feed `normal_patterns`, updates rejected as Byzantine feed
/// `attack_patterns`.
pub struct PatternModel<T: Float + Debug + Send + Sync + 'static> {
    /// Reference patterns for normal behavior
    normal_patterns: Vec<Array1<T>>,

    /// Reference patterns for attack behaviors
    attack_patterns: Vec<Array1<T>>,

    /// Cosine similarity above which an observation is merged into an existing
    /// prototype instead of creating a new one
    matching_threshold: f64,
}

/// Statistical analysis engine
pub struct StatisticalAnalysis<T: Float + Debug + Send + Sync + 'static> {
    /// Statistical measures of the most recently analysed cohort
    measures: StatisticalMeasures<T>,
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

/// Gradient verification system
pub struct GradientVerifier<T: Float + Debug + Send + Sync + 'static> {
    /// Expected gradient properties
    expected_properties: GradientProperties<T>,

    /// Verification rules
    verification_rules: Vec<VerificationRule<T>>,

    /// Aggregate accepted in the previous round, used as the reference direction
    /// for the direction-consistency rule
    reference_direction: Option<Array1<T>>,
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

/// Verification rule for gradients
pub struct VerificationRule<T: Float + Debug + Send + Sync + 'static> {
    /// Rule name
    pub name: String,

    /// Rule function
    pub rule_fn: RuleFn<T>,

    /// Rule weight in verification
    pub weight: f64,
}

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Create new Byzantine tolerant aggregator.
    ///
    /// Returns an error when `config` is inconsistent (see
    /// [`ByzantineConfig::validate`]).
    pub fn new(config: ByzantineConfig) -> Result<Self> {
        config.validate()?;
        let anomaly_threshold = config.anomaly_threshold;
        Ok(Self {
            config,
            reputation_scores: HashMap::new(),
            behavior_history: HashMap::new(),
            anomaly_detector: AnomalyDetector::new(anomaly_threshold),
            statistics_engine: StatisticalAnalysis::new(),
            gradient_verifier: GradientVerifier::new(),
            fools_gold_history: HashMap::new(),
            round: 0,
        })
    }

    /// Configuration in use.
    pub fn config(&self) -> &ByzantineConfig {
        &self.config
    }

    /// Number of completed aggregation rounds.
    pub fn round(&self) -> u64 {
        self.round
    }

    /// Reputation record of a participant, if one exists.
    pub fn reputation(&self, participant_id: &str) -> Option<&ReputationScore> {
        self.reputation_scores.get(participant_id)
    }

    /// Recorded behaviour history of a participant, if one exists.
    pub fn behavior_history(&self, participant_id: &str) -> Option<&BehaviorHistory> {
        self.behavior_history.get(participant_id)
    }

    /// Replace the gradient properties enforced by the verification stage.
    pub fn set_gradient_properties(&mut self, properties: GradientProperties<T>) {
        let reference = self.gradient_verifier.reference_direction.clone();
        self.gradient_verifier = GradientVerifier::with_properties(properties);
        self.gradient_verifier.reference_direction = reference;
    }

    /// Perform Byzantine-robust aggregation.
    pub fn byzantine_robust_aggregate(
        &mut self,
        participant_gradients: &HashMap<String, Array1<T>>,
    ) -> Result<ByzantineAggregationResult<T>> {
        // Step 0: reject malformed cohorts up front so no downstream comparator can
        // encounter NaN and no dimension mismatch can slip through.
        Self::validate_cohort(participant_gradients)?;
        let total_submitted = participant_gradients.len();

        // Step 1: Pre-filtering based on reputation
        let filtered_participants = self.filter_by_reputation(participant_gradients)?;
        if filtered_participants.is_empty() {
            return Err(OptimError::InvalidState(
                "every submitting participant is blacklisted".to_string(),
            ));
        }

        // Step 2: Anomaly detection
        let anomaly_results = self.detect_anomalies(&filtered_participants)?;

        // Step 3: Statistical outlier detection
        let outlier_results = self.detect_statistical_outliers(&filtered_participants)?;

        // Step 4: Gradient verification
        let verification_results = if self.config.gradient_verification {
            self.verify_gradients(&filtered_participants)?
        } else {
            HashMap::new()
        };

        // Step 5: Identify Byzantine participants
        let byzantine_participants = self.identify_byzantine_participants(
            &anomaly_results,
            &outlier_results,
            &verification_results,
        )?;

        // Step 6: Select honest participants
        let honest_participants =
            self.select_honest_participants(&filtered_participants, &byzantine_participants)?;

        // Step 6b: consensus gate
        let consensus_ratio = honest_participants.len() as f64 / total_submitted as f64;
        if consensus_ratio < self.config.consensus_threshold {
            return Err(OptimError::InvalidState(format!(
                "only {:.1}% of the cohort survived Byzantine detection, below the configured \
                 consensus threshold of {:.1}%",
                consensus_ratio * 100.0,
                self.config.consensus_threshold * 100.0
            )));
        }

        // Step 7: Perform robust aggregation
        let aggregate = self.perform_robust_aggregation(&honest_participants)?;

        // Step 8: Record behaviour, update reputations and learn behaviour patterns
        let confidence_score =
            self.calculate_confidence_score(&honest_participants, total_submitted, &aggregate)?;
        self.record_behavior(
            participant_gradients,
            &filtered_participants,
            &anomaly_results,
            &aggregate,
        )?;
        self.update_reputations(&honest_participants, &byzantine_participants)?;
        self.learn_patterns(
            &honest_participants,
            &filtered_participants,
            &byzantine_participants,
        )?;
        self.gradient_verifier
            .set_reference_direction(aggregate.clone());
        self.round = self.round.saturating_add(1);

        let mut honest_ids: Vec<String> = honest_participants.keys().cloned().collect();
        honest_ids.sort();

        Ok(ByzantineAggregationResult {
            aggregate,
            honest_participants: honest_ids,
            byzantine_participants,
            reputation_updates: self.get_reputation_updates(),
            aggregation_method: self.config.aggregation_method,
            consensus_ratio,
            confidence_score,
        })
    }

    /// Reject empty, ragged or non-finite cohorts.
    fn validate_cohort(gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        if gradients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No gradients to aggregate".to_string(),
            ));
        }

        let mut expected_dim: Option<usize> = None;
        for (participant_id, gradient) in Self::ordered_cohort(gradients) {
            if gradient.is_empty() {
                return Err(OptimError::InvalidConfig(format!(
                    "participant '{participant_id}' submitted an empty gradient"
                )));
            }
            match expected_dim {
                None => expected_dim = Some(gradient.len()),
                Some(dim) if dim != gradient.len() => {
                    return Err(OptimError::DimensionMismatch(format!(
                        "participant '{participant_id}' submitted a gradient of length {} \
                         while the cohort uses {dim}",
                        gradient.len()
                    )));
                }
                Some(_) => {}
            }
            if !gradient.iter().all(|x| x.is_finite()) {
                return Err(OptimError::InvalidParameter(format!(
                    "participant '{participant_id}' submitted a non-finite gradient"
                )));
            }
        }

        Ok(())
    }

    /// Cohort ordered by participant id, making every algorithm in this module
    /// independent of `HashMap` iteration order.
    fn ordered_cohort(gradients: &HashMap<String, Array1<T>>) -> Vec<(&String, &Array1<T>)> {
        let mut items: Vec<(&String, &Array1<T>)> = gradients.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        items
    }

    /// Filter participants based on reputation scores
    fn filter_by_reputation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut filtered = HashMap::new();

        for (participant_id, gradient) in gradients {
            match self.reputation_scores.get(participant_id) {
                Some(reputation) if reputation.trust_level == TrustLevel::Blacklisted => {}
                // New participants are admitted with medium trust.
                _ => {
                    filtered.insert(participant_id.clone(), gradient.clone());
                }
            }
        }

        Ok(filtered)
    }

    /// Detect anomalies in gradients
    fn detect_anomalies(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, AnomalyScore>> {
        let ordered: Vec<(String, Array1<T>)> = Self::ordered_cohort(gradients)
            .into_iter()
            .map(|(id, g)| (id.clone(), g.clone()))
            .collect();

        let mut anomaly_results = HashMap::new();
        for (participant_id, gradient) in ordered {
            let anomaly_score = self
                .anomaly_detector
                .detect_anomaly(&participant_id, &gradient)?;
            anomaly_results.insert(participant_id, anomaly_score);
        }

        Ok(anomaly_results)
    }

    /// Detect statistical outliers across the current round's cohort.
    fn detect_statistical_outliers(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, OutlierScore>> {
        let ordered = Self::ordered_cohort(gradients);
        let cohort: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let stats = self.statistics_engine.compute_statistics(&cohort)?;

        let mut outlier_results = HashMap::new();
        for (index, (participant_id, _)) in ordered.iter().enumerate() {
            let outlier_score = self.compute_outlier_score(index, &cohort, &stats)?;
            outlier_results.insert((*participant_id).clone(), outlier_score);
        }

        Ok(outlier_results)
    }

    /// Verify gradients using verification rules
    fn verify_gradients(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, VerificationScore>> {
        let mut verification_results = HashMap::new();

        for (participant_id, gradient) in gradients {
            let verification_score = self.gradient_verifier.verify_gradient(gradient)?;
            verification_results.insert(participant_id.clone(), verification_score);
        }

        Ok(verification_results)
    }

    /// Identify Byzantine participants based on multiple criteria.
    ///
    /// Every participant whose combined score exceeds `anomaly_threshold` is
    /// reported; the list is deliberately not capped at `max_byzantine`, because a
    /// round in which more than `f` participants look Byzantine is exactly the case
    /// the caller must be told about (the downstream consensus gate then rejects it).
    fn identify_byzantine_participants(
        &self,
        anomaly_results: &HashMap<String, AnomalyScore>,
        outlier_results: &HashMap<String, OutlierScore>,
        verification_results: &HashMap<String, VerificationScore>,
    ) -> Result<Vec<String>> {
        let mut participant_ids: Vec<&String> = anomaly_results.keys().collect();
        participant_ids.sort();

        let mut byzantine_participants = Vec::new();
        for participant_id in participant_ids {
            let anomaly_score = anomaly_results.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing anomaly score for '{participant_id}'"))
            })?;
            let outlier_score = outlier_results.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing outlier score for '{participant_id}'"))
            })?;
            let verification_score = verification_results.get(participant_id);

            let combined_score =
                self.compute_byzantine_score(anomaly_score, outlier_score, verification_score);

            if combined_score > self.config.anomaly_threshold {
                byzantine_participants.push(participant_id.clone());
            }
        }

        Ok(byzantine_participants)
    }

    /// Select honest participants for aggregation
    fn select_honest_participants(
        &self,
        all_participants: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut honest_participants = HashMap::new();

        for (participant_id, gradient) in all_participants {
            if !byzantine_participants.contains(participant_id) {
                honest_participants.insert(participant_id.clone(), gradient.clone());
            }
        }

        if honest_participants.len() < self.config.min_participants {
            return Err(OptimError::InvalidState(format!(
                "only {} honest participants remain, but {} are required",
                honest_participants.len(),
                self.config.min_participants
            )));
        }

        Ok(honest_participants)
    }

    /// Perform robust aggregation using the configured method.
    ///
    /// `f` is always `config.max_byzantine`: the detection stages that ran before
    /// this point are heuristics, so their removals are not assumed sound and the
    /// configured tolerance is kept in full. `n` is the number of gradients handed
    /// to this call, i.e. the round's participants minus those filtered out by
    /// reputation and minus those flagged Byzantine.
    fn perform_robust_aggregation(
        &mut self,
        honest_gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        let f = self.config.max_byzantine;
        match self.config.aggregation_method {
            ByzantineAggregationMethod::TrimmedMean => {
                self.trimmed_mean_aggregation(honest_gradients, f)
            }
            ByzantineAggregationMethod::CoordinateMedian => {
                self.coordinate_median_aggregation(honest_gradients)
            }
            ByzantineAggregationMethod::Krum => self.krum_aggregation(honest_gradients, f),
            ByzantineAggregationMethod::MultiKrum => {
                self.multi_krum_aggregation(honest_gradients, f)
            }
            ByzantineAggregationMethod::Bulyan => self.bulyan_aggregation(honest_gradients, f),
            ByzantineAggregationMethod::FoolsGold => self.fools_gold_aggregation(honest_gradients),
            ByzantineAggregationMethod::FLAME => self.flame_aggregation(honest_gradients),
            ByzantineAggregationMethod::Median => self.median_aggregation(honest_gradients),
            ByzantineAggregationMethod::GeometricMedian => {
                self.geometric_median_aggregation(honest_gradients)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregation algorithms
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Coordinate-wise trimmed mean (Yin et al., 2018).
    ///
    /// `trim_per_tail` values are dropped from each tail of every coordinate; the
    /// robustness bound requires `trim_per_tail >= f`, hence the pipeline passes
    /// `config.max_byzantine`. Requires `n > 2 * trim_per_tail` so at least one
    /// value survives - smaller cohorts are rejected instead of silently
    /// aggregating to zero.
    fn trimmed_mean_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        trim_per_tail: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ordered = Self::ordered_cohort(gradients);
        let values: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = values.len();

        if n <= 2 * trim_per_tail {
            return Err(OptimError::InvalidState(format!(
                "trimmed mean needs more than {} gradients to trim {} from each tail, got {}",
                2 * trim_per_tail,
                trim_per_tail,
                n
            )));
        }

        let dim = values[0].len();
        let mut result = Array1::zeros(dim);
        let kept = n - 2 * trim_per_tail;
        let divisor: T = to_scalar(kept as f64)?;

        for i in 0..dim {
            let mut coord_values: Vec<T> = values.iter().map(|g| g[i]).collect();
            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let sum: T = coord_values[trim_per_tail..n - trim_per_tail]
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + x);
            result[i] = sum / divisor;
        }

        Ok(result)
    }

    /// Coordinate-wise median aggregation
    fn coordinate_median_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ordered = Self::ordered_cohort(gradients);
        let values: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let dim = values[0].len();
        let mut result = Array1::zeros(dim);
        let two: T = to_scalar(2.0)?;

        for i in 0..dim {
            let mut coord_values: Vec<T> = values.iter().map(|g| g[i]).collect();
            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            result[i] = if coord_values.len().is_multiple_of(2) {
                let mid = coord_values.len() / 2;
                (coord_values[mid - 1] + coord_values[mid]) / two
            } else {
                coord_values[coord_values.len() / 2]
            };
        }

        Ok(result)
    }

    /// Krum scores of every gradient in `grads`.
    ///
    /// The score of gradient `i` is the sum of its `n - f - 2` smallest squared-free
    /// Euclidean distances to the other gradients. The window is computed with
    /// saturating arithmetic and clamped to at least one neighbour, so no call can
    /// underflow even if a caller bypasses the cohort-size gate.
    fn krum_scores(&self, grads: &[&Array1<T>], f: usize) -> Result<Vec<T>> {
        let n = grads.len();
        let take_count = n.saturating_sub(f).saturating_sub(2).max(1);
        let mut scores = Vec::with_capacity(n);

        for (i, gradient) in grads.iter().enumerate() {
            let mut distances = Vec::with_capacity(n.saturating_sub(1));
            for (j, other) in grads.iter().enumerate() {
                if i != j {
                    distances.push(euclidean_distance(gradient, other)?);
                }
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let score = distances
                .iter()
                .take(take_count)
                .copied()
                .fold(T::zero(), |acc, d| acc + d);
            scores.push(score);
        }

        Ok(scores)
    }

    /// Index of the minimum-score gradient; ties resolve to the lowest index.
    fn krum_select(&self, grads: &[&Array1<T>], f: usize) -> Result<usize> {
        let scores = self.krum_scores(grads, f)?;
        let mut best = 0usize;
        for (i, score) in scores.iter().enumerate().skip(1) {
            if *score < scores[best] {
                best = i;
            }
        }
        if scores.is_empty() {
            return Err(OptimError::InvalidState(
                "cannot run Krum on an empty cohort".to_string(),
            ));
        }
        Ok(best)
    }

    /// Reject cohorts smaller than `required` for the named algorithm.
    fn require_cohort(n: usize, required: usize, algorithm: &str, f: usize) -> Result<()> {
        if n < required {
            return Err(OptimError::InvalidState(format!(
                "{algorithm} tolerating {f} Byzantine participants requires at least {required} \
                 gradients, got {n}"
            )));
        }
        Ok(())
    }

    /// Krum aggregation (Blanchard et al., 2017): select the single most
    /// representative gradient. Requires `n >= 2f + 3`.
    fn krum_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        f: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        Self::require_cohort(grads.len(), 2 * f + 3, "Krum", f)?;

        let selected = self.krum_select(&grads, f)?;
        Ok(grads[selected].clone())
    }

    /// Multi-Krum aggregation: average the `n - f` gradients with the lowest Krum
    /// scores. Requires `n >= 2f + 3`.
    fn multi_krum_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        f: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = grads.len();
        Self::require_cohort(n, 2 * f + 3, "Multi-Krum", f)?;

        let k = n.saturating_sub(f).max(1);
        let scores = self.krum_scores(&grads, f)?;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut result: Array1<T> = Array1::zeros(grads[0].len());
        for &index in indices.iter().take(k) {
            result = result + grads[index];
        }
        let divisor: T = to_scalar(k as f64)?;
        Ok(result / divisor)
    }

    /// Bulyan aggregation (El Mhamdi et al., 2018).
    ///
    /// Stage one runs Krum `theta = n - 2f` times, removing the selected gradient
    /// each time. Stage two averages, per coordinate, the `beta = theta - 2f`
    /// values closest to the median of the selection. Requires `n >= 4f + 3`.
    fn bulyan_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        f: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = grads.len();
        Self::require_cohort(n, 4 * f + 3, "Bulyan", f)?;

        let theta = n - 2 * f;
        let beta = theta - 2 * f;

        // Stage 1: iterative Krum with removal.
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut selection: Vec<usize> = Vec::with_capacity(theta);
        for _ in 0..theta {
            if remaining.is_empty() {
                break;
            }
            let subset: Vec<&Array1<T>> = remaining.iter().map(|&i| grads[i]).collect();
            let local = self.krum_select(&subset, f)?;
            selection.push(remaining[local]);
            remaining.remove(local);
        }

        if selection.len() < beta || beta == 0 {
            return Err(OptimError::InvalidState(format!(
                "Bulyan selected {} gradients but needs at least {} for the median stage",
                selection.len(),
                beta.max(1)
            )));
        }

        // Stage 2: per-coordinate average of the beta values closest to the median.
        let dim = grads[0].len();
        let mut result = Array1::zeros(dim);
        let two: T = to_scalar(2.0)?;
        let divisor: T = to_scalar(beta as f64)?;

        for c in 0..dim {
            let mut coord: Vec<T> = selection.iter().map(|&i| grads[i][c]).collect();
            let mut sorted = coord.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let median = if sorted.len().is_multiple_of(2) {
                let mid = sorted.len() / 2;
                (sorted[mid - 1] + sorted[mid]) / two
            } else {
                sorted[sorted.len() / 2]
            };

            // Stable sort by distance to the median keeps ties in selection order.
            coord.sort_by(|a, b| {
                (*a - median)
                    .abs()
                    .partial_cmp(&(*b - median).abs())
                    .unwrap_or(Ordering::Equal)
            });

            let sum = coord
                .iter()
                .take(beta)
                .copied()
                .fold(T::zero(), |acc, x| acc + x);
            result[c] = sum / divisor;
        }

        Ok(result)
    }

    /// Simple median aggregation
    fn median_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        self.coordinate_median_aggregation(gradients)
    }

    /// Geometric median aggregation via Weiszfeld's algorithm.
    fn geometric_median_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ordered = Self::ordered_cohort(gradients);
        let values: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let mut current = values[0].clone();
        let tolerance: T = to_scalar(1e-6)?;

        for _ in 0..100 {
            let mut numerator: Array1<T> = Array1::zeros(current.len());
            let mut denominator = T::zero();

            for &gradient in &values {
                let distance = euclidean_distance(&current, gradient)?;
                if distance > T::zero() {
                    let weight = T::one() / distance;
                    numerator = numerator + gradient * weight;
                    denominator = denominator + weight;
                }
            }

            if denominator <= T::zero() {
                break;
            }

            let new_estimate = numerator / denominator;
            let change = euclidean_distance(&current, &new_estimate)?;
            current = new_estimate;
            if change < tolerance {
                break;
            }
        }

        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// FoolsGold and FLAME
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// FoolsGold aggregation (Fung et al., 2020).
    ///
    /// Sybils share an objective, so their *historical* update directions stay
    /// mutually similar while honest clients diverge. Each participant's updates
    /// are accumulated across rounds, the pairwise cosine similarity matrix of
    /// those histories is built, pardoning re-scales the row of a client that is
    /// only similar to a more-similar client, and the resulting `alpha` is
    /// rescaled and passed through the logit `ln(a / (1 - a)) + 0.5` clipped to
    /// `[0, 1]` to obtain the per-client learning rate.
    fn fools_gold_aggregation(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ids: Vec<String> = {
            let ordered = Self::ordered_cohort(gradients);
            ordered.iter().map(|(id, _)| (*id).clone()).collect()
        };

        // Accumulate this round's updates into the per-client history.
        for participant_id in &ids {
            let gradient = gradients.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing gradient for '{participant_id}'"))
            })?;
            match self.fools_gold_history.get_mut(participant_id) {
                // A dimension change means the model was reshaped; the accumulated
                // history is no longer comparable, so it restarts from this round.
                Some(history) if history.len() == gradient.len() => {
                    *history = &*history + gradient;
                }
                _ => {
                    self.fools_gold_history
                        .insert(participant_id.clone(), gradient.clone());
                }
            }
        }

        let weights = self.compute_fools_gold_weights(&ids)?;

        let dim = gradients
            .get(&ids[0])
            .map(|g| g.len())
            .ok_or_else(|| OptimError::InvalidState("empty FoolsGold cohort".to_string()))?;
        let mut result: Array1<T> = Array1::zeros(dim);
        let mut total_weight = 0.0f64;

        for (participant_id, weight) in ids.iter().zip(weights.iter()) {
            let gradient = gradients.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing gradient for '{participant_id}'"))
            })?;
            let scaled: T = to_scalar(*weight)?;
            result = result + gradient * scaled;
            total_weight += *weight;
        }

        if total_weight > 0.0 {
            let divisor: T = to_scalar(total_weight)?;
            Ok(result / divisor)
        } else {
            // Every client received zero weight, which only happens when all
            // histories are perfectly collinear: honest and Sybil updates are then
            // indistinguishable and there is nothing to discriminate on, so the
            // unweighted mean is returned.
            let mut mean: Array1<T> = Array1::zeros(dim);
            for participant_id in &ids {
                let gradient = gradients.get(participant_id).ok_or_else(|| {
                    OptimError::InvalidState(format!("missing gradient for '{participant_id}'"))
                })?;
                mean = mean + gradient;
            }
            let divisor: T = to_scalar(ids.len() as f64)?;
            Ok(mean / divisor)
        }
    }

    /// FoolsGold per-client learning rates in `[0, 1]`, computed from the
    /// accumulated update histories.
    fn compute_fools_gold_weights(&self, ids: &[String]) -> Result<Vec<f64>> {
        let n = ids.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            // A single client cannot be a Sybil of anyone.
            return Ok(vec![1.0]);
        }

        let histories: Vec<&Array1<T>> = ids
            .iter()
            .map(|id| {
                self.fools_gold_history.get(id).ok_or_else(|| {
                    OptimError::InvalidState(format!("missing FoolsGold history for '{id}'"))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Pairwise cosine similarity of the histories.
        let mut cs = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let similarity = if histories[i].len() == histories[j].len() {
                    from_scalar(cosine_similarity(histories[i], histories[j])?)?
                } else {
                    0.0
                };
                cs[i][j] = similarity;
                cs[j][i] = similarity;
            }
        }

        // Maximum similarity of every client to any other client.
        let row_max = |row: &[f64], skip: usize| -> f64 {
            row.iter()
                .enumerate()
                .filter(|(j, _)| *j != skip)
                .map(|(_, v)| *v)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let v: Vec<f64> = (0..n).map(|i| row_max(&cs[i], i)).collect();

        // Pardoning: a client that merely resembles a very-similar client should not
        // be punished for it.
        let mut pardoned = cs.clone();
        for i in 0..n {
            for j in 0..n {
                if i != j && v[j] > v[i] && v[j] > 0.0 {
                    pardoned[i][j] *= v[i] / v[j];
                }
            }
        }

        let mut alpha: Vec<f64> = (0..n)
            .map(|i| (1.0 - row_max(&pardoned[i], i)).clamp(0.0, 1.0))
            .collect();

        // Rescale so the least-Sybil client keeps a full learning rate.
        let max_alpha = alpha.iter().copied().fold(0.0f64, f64::max);
        if max_alpha > 0.0 {
            for a in alpha.iter_mut() {
                *a = (*a / max_alpha).clamp(0.0, 1.0);
            }
        } else {
            return Ok(vec![0.0; n]);
        }

        // Logit rescale, clipped to [0, 1].
        Ok(alpha
            .into_iter()
            .map(|a| {
                let clamped = a.clamp(1e-9, 0.99);
                let logit = (clamped / (1.0 - clamped)).ln() + 0.5;
                if logit.is_finite() {
                    logit.clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .collect())
    }

    /// FLAME aggregation (Nguyen et al., 2022).
    ///
    /// 1. Deterministic average-linkage agglomerative clustering on cosine
    ///    distances, merging until a cluster reaches `min_cluster_size = n/2 + 1`;
    ///    that cluster is admitted and everything else is discarded.
    /// 2. Norm-median clipping: every admitted update is scaled by
    ///    `min(1, S_t / ||g_i||)` where `S_t` is the median L2 norm over the whole
    ///    cohort, so a scaling attack cannot dominate the mean.
    /// 3. Calibrated Gaussian noise with `sigma = lambda * S_t` is added to the
    ///    clipped mean.
    ///
    /// The noise is a backdoor-mitigation measure and carries **no** differential
    /// privacy guarantee; it is drawn from a deterministic generator seeded by the
    /// round counter, so a given sequence of rounds is reproducible.
    fn flame_aggregation(&self, gradients: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;

        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = grads.len();
        let dim = grads[0].len();

        let admitted = Self::flame_admitted_indices(&grads)?;
        if admitted.is_empty() {
            return Err(OptimError::InvalidState(
                "FLAME clustering admitted no gradients".to_string(),
            ));
        }

        // Median of the cohort's L2 norms.
        let mut norms: Vec<T> = grads.iter().map(|g| l2_norm(g)).collect();
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let two: T = to_scalar(2.0)?;
        let median_norm = if norms.len().is_multiple_of(2) {
            let mid = norms.len() / 2;
            (norms[mid - 1] + norms[mid]) / two
        } else {
            norms[norms.len() / 2]
        };

        // Norm-median clipping followed by the mean of the admitted updates.
        let mut result: Array1<T> = Array1::zeros(dim);
        for &index in &admitted {
            let gradient = grads[index];
            let norm = l2_norm(gradient);
            let gamma = if norm > T::zero() && norm > median_norm {
                median_norm / norm
            } else {
                T::one()
            };
            result = result + gradient * gamma;
        }
        let divisor: T = to_scalar(admitted.len() as f64)?;
        result = result / divisor;

        // Calibrated noise.
        let sigma = from_scalar(median_norm)? * FLAME_NOISE_LAMBDA;
        if sigma > 0.0 {
            let seed = self
                .round
                .wrapping_mul(0x0000_0100_0000_01B3)
                .wrapping_add(n as u64)
                .wrapping_add(0x0000_0F1A_3E5E_ED00);
            let mut rng = SplitMix64::new(seed);
            for value in result.iter_mut() {
                let noise: T = to_scalar(rng.next_gaussian() * sigma)?;
                *value = *value + noise;
            }
        }

        Ok(result)
    }

    /// Indices admitted by FLAME's clustering stage, sorted ascending.
    ///
    /// Deterministic average-linkage agglomerative clustering on cosine distances:
    /// the closest pair of clusters is merged repeatedly (ties resolve to the
    /// lowest index pair) until one cluster reaches `n / 2 + 1` members.
    fn flame_admitted_indices(grads: &[&Array1<T>]) -> Result<Vec<usize>> {
        let n = grads.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let min_size = n / 2 + 1;
        let mut distance = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = 1.0 - from_scalar(cosine_similarity(grads[i], grads[j])?)?;
                distance[i][j] = d;
                distance[j][i] = d;
            }
        }

        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        if clusters[0].len() >= min_size {
            return Ok(clusters[0].clone());
        }

        while clusters.len() > 1 {
            let mut best: Option<(usize, usize, f64)> = None;
            for a in 0..clusters.len() {
                for b in (a + 1)..clusters.len() {
                    let mut sum = 0.0;
                    for &x in &clusters[a] {
                        for &y in &clusters[b] {
                            sum += distance[x][y];
                        }
                    }
                    let linkage = sum / (clusters[a].len() * clusters[b].len()) as f64;
                    if best.is_none_or(|(_, _, current)| linkage < current) {
                        best = Some((a, b, linkage));
                    }
                }
            }

            let Some((a, b, _)) = best else { break };
            let merged = clusters.remove(b);
            clusters[a].extend(merged);
            if clusters[a].len() >= min_size {
                let mut out = clusters[a].clone();
                out.sort_unstable();
                return Ok(out);
            }
        }

        let mut out = clusters.into_iter().next().unwrap_or_default();
        out.sort_unstable();
        Ok(out)
    }

    /// Participant ids admitted by FLAME's clustering stage, sorted ascending.
    pub fn flame_admitted_ids(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Vec<String>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let admitted = Self::flame_admitted_indices(&grads)?;
        Ok(admitted
            .into_iter()
            .map(|index| ordered[index].0.clone())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Outlier detection
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Outlier score in `[0, 1]` for `cohort[index]` under the configured method.
    ///
    /// The match is exhaustive on purpose: adding a variant to
    /// [`OutlierDetectionMethod`] must be a compile error rather than fall into a
    /// wildcard arm.
    fn compute_outlier_score(
        &self,
        index: usize,
        cohort: &[&Array1<T>],
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let gradient = cohort.get(index).ok_or_else(|| {
            OptimError::InvalidState(format!("outlier index {index} is outside the cohort"))
        })?;

        match self.config.outlier_detection {
            OutlierDetectionMethod::ZScore => Self::zscore_outlier_score(gradient, stats),
            OutlierDetectionMethod::IQR => Self::iqr_outlier_score(gradient, stats),
            OutlierDetectionMethod::MahalanobisDistance => {
                Self::mahalanobis_outlier_score(gradient, stats)
            }
            OutlierDetectionMethod::LocalOutlierFactor => Self::lof_outlier_score(index, cohort),
            OutlierDetectionMethod::IsolationForest => {
                Self::isolation_forest_outlier_score(index, cohort)
            }
        }
    }

    /// Maximum per-coordinate z-score, normalised by the three-sigma rule.
    fn zscore_outlier_score(
        gradient: &Array1<T>,
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let mut max_z_score = 0.0f64;

        for i in 0..gradient.len().min(stats.std_dev.len()) {
            if stats.std_dev[i] > T::zero() {
                let z_score = ((gradient[i] - stats.mean[i]) / stats.std_dev[i]).abs();
                let z_score = from_scalar(z_score)?;
                if z_score > max_z_score {
                    max_z_score = z_score;
                }
            }
        }

        Ok(OutlierScore {
            score: (max_z_score / 3.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::ZScore,
            details: format!("Max Z-score: {max_z_score:.4}"),
        })
    }

    /// Interquartile-range score.
    ///
    /// A coordinate on which at least three quarters of the cohort agree has a
    /// zero IQR; instead of dividing by it (which produced `+inf`), such a
    /// coordinate falls back to the absolute deviation from the mean normalised by
    /// the coordinate's standard deviation, and contributes nothing at all when the
    /// coordinate is constant across the cohort.
    fn iqr_outlier_score(
        gradient: &Array1<T>,
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let mut max_score = 0.0f64;
        let mut degenerate = 0usize;

        for i in 0..gradient.len().min(stats.iqr.len()) {
            let value = gradient[i];
            let iqr = stats.iqr[i];

            let score = if iqr > T::zero() {
                let excess = if value < stats.q1[i] {
                    stats.q1[i] - value
                } else if value > stats.q3[i] {
                    value - stats.q3[i]
                } else {
                    T::zero()
                };
                // 1.5 * IQR beyond a quartile is Tukey's classic outlier fence.
                from_scalar(excess / iqr)? / 1.5
            } else {
                degenerate += 1;
                if stats.std_dev[i] > T::zero() {
                    from_scalar((value - stats.mean[i]).abs() / stats.std_dev[i])? / 3.0
                } else {
                    0.0
                }
            };

            if score > max_score {
                max_score = score;
            }
        }

        Ok(OutlierScore {
            score: max_score.clamp(0.0, 1.0),
            method: OutlierDetectionMethod::IQR,
            details: format!(
                "Max IQR score: {max_score:.4} ({degenerate} degenerate coordinate(s))"
            ),
        })
    }

    /// Diagonal-covariance Mahalanobis distance, reported as the root-mean-square
    /// z-score and normalised by the three-sigma rule.
    fn mahalanobis_outlier_score(
        gradient: &Array1<T>,
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let mut accumulated = 0.0f64;
        let mut used = 0usize;

        for i in 0..gradient.len().min(stats.std_dev.len()) {
            if stats.std_dev[i] > T::zero() {
                let z = from_scalar((gradient[i] - stats.mean[i]) / stats.std_dev[i])?;
                accumulated += z * z;
                used += 1;
            }
        }

        let distance = if used == 0 {
            0.0
        } else {
            (accumulated / used as f64).sqrt()
        };

        Ok(OutlierScore {
            score: (distance / 3.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::MahalanobisDistance,
            details: format!(
                "Diagonal Mahalanobis (RMS z) distance: {distance:.4} over {used} coordinate(s)"
            ),
        })
    }

    /// Local outlier factor (Breunig et al., 2000) over the current cohort.
    fn lof_outlier_score(index: usize, cohort: &[&Array1<T>]) -> Result<OutlierScore> {
        let n = cohort.len();
        if n < 3 {
            return Ok(OutlierScore {
                score: 0.0,
                method: OutlierDetectionMethod::LocalOutlierFactor,
                details: format!("cohort of {n} is too small for a local outlier factor"),
            });
        }

        // The neighbourhood is capped at half the cohort: with `k = n - 1` every
        // point neighbours every other, so a single distant gradient dominates every
        // k-distance and LOF collapses to 1 for the whole round.
        let k = ((n - 1) / 2).clamp(1, 20);
        let mut distance = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = from_scalar(euclidean_distance(cohort[i], cohort[j])?)?;
                distance[i][j] = d;
                distance[j][i] = d;
            }
        }

        let mut neighbours: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut k_distance = vec![0.0f64; n];
        for i in 0..n {
            let mut others: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, distance[i][j]))
                .collect();
            others.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            k_distance[i] = others[k - 1].1;
            neighbours.push(others.into_iter().take(k).map(|(j, _)| j).collect());
        }

        // Local reachability density; the epsilon floor keeps duplicated gradients
        // (mean reachability distance of zero) finite.
        let mut lrd = vec![0.0f64; n];
        for i in 0..n {
            let sum: f64 = neighbours[i]
                .iter()
                .map(|&j| k_distance[j].max(distance[i][j]))
                .sum();
            let mean = sum / neighbours[i].len() as f64;
            lrd[i] = 1.0 / mean.max(f64::EPSILON);
        }

        let neighbourhood = &neighbours[index];
        let lof = neighbourhood.iter().map(|&j| lrd[j]).sum::<f64>()
            / (neighbourhood.len() as f64 * lrd[index]);
        let lof = if lof.is_finite() { lof } else { 1.0 };

        Ok(OutlierScore {
            score: (lof - 1.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::LocalOutlierFactor,
            details: format!("Local outlier factor: {lof:.4} (k = {k})"),
        })
    }

    /// Isolation forest (Liu et al., 2008) over the current cohort.
    ///
    /// The splits are drawn from a deterministically seeded generator, so repeated
    /// evaluation of the same cohort yields the same score.
    fn isolation_forest_outlier_score(index: usize, cohort: &[&Array1<T>]) -> Result<OutlierScore> {
        let n = cohort.len();
        if n < 3 {
            return Ok(OutlierScore {
                score: 0.0,
                method: OutlierDetectionMethod::IsolationForest,
                details: format!("cohort of {n} is too small for an isolation forest"),
            });
        }

        let points: Vec<Vec<f64>> = cohort
            .iter()
            .map(|g| to_f64_vec(g))
            .collect::<Result<Vec<_>>>()?;
        let indices: Vec<usize> = (0..n).collect();
        let depth_limit = ((n as f64).log2().ceil() as usize).max(1);

        let mut totals = vec![0.0f64; n];
        let mut rng = SplitMix64::new(0x15_01A7_10F0_2E57);
        for _ in 0..ISOLATION_TREES {
            isolation_path(&points, &indices, 0, depth_limit, &mut rng, &mut totals);
        }

        let expected = totals[index] / ISOLATION_TREES as f64;
        let normaliser = average_path_length(n);
        let raw = if normaliser > 0.0 {
            2f64.powf(-expected / normaliser)
        } else {
            0.5
        };

        Ok(OutlierScore {
            // 0.5 is the neutral isolation score; only the excess is reported so
            // that, like every other detector here, 0 means "not an outlier".
            score: ((raw - 0.5) * 2.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::IsolationForest,
            details: format!("Isolation score: {raw:.4}, mean path length: {expected:.4}"),
        })
    }
}

/// Accumulate isolation-tree path lengths for every point in `idx`.
fn isolation_path(
    points: &[Vec<f64>],
    idx: &[usize],
    depth: usize,
    limit: usize,
    rng: &mut SplitMix64,
    out: &mut [f64],
) {
    let terminate = |out: &mut [f64]| {
        let adjustment = average_path_length(idx.len());
        for &i in idx {
            out[i] += depth as f64 + adjustment;
        }
    };

    if idx.len() <= 1 || depth >= limit {
        terminate(out);
        return;
    }

    let dim = points[idx[0]].len();
    let mut candidates: Vec<(usize, f64, f64)> = Vec::new();
    // `d` indexes the coordinate of every point in `idx`, not a single slice, so
    // there is no iterator form of this scan.
    #[allow(clippy::needless_range_loop)]
    for d in 0..dim {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &i in idx {
            let value = points[i][d];
            if value < lo {
                lo = value;
            }
            if value > hi {
                hi = value;
            }
        }
        if hi > lo {
            candidates.push((d, lo, hi));
        }
    }

    if candidates.is_empty() {
        terminate(out);
        return;
    }

    let (dimension, lo, hi) = candidates[rng.next_range(candidates.len())];
    let split = lo + rng.next_f64() * (hi - lo);

    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in idx {
        if points[i][dimension] < split {
            left.push(i);
        } else {
            right.push(i);
        }
    }

    if left.is_empty() || right.is_empty() {
        terminate(out);
        return;
    }

    isolation_path(points, &left, depth + 1, limit, rng, out);
    isolation_path(points, &right, depth + 1, limit, rng, out);
}

/// Average path length of an unsuccessful search in a binary search tree of `n`
/// nodes, the isolation-forest normalisation constant `c(n)`.
fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    2.0 * ((nf - 1.0).ln() + EULER_MASCHERONI) - 2.0 * (nf - 1.0) / nf
}

// ---------------------------------------------------------------------------
// Scoring, reputation and behaviour tracking
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Combined Byzantine score in `[0, 1]`; every input is already bounded to
    /// `[0, 1]`, so the weights below are a genuine convex combination.
    fn compute_byzantine_score(
        &self,
        anomaly_score: &AnomalyScore,
        outlier_score: &OutlierScore,
        verification_score: Option<&VerificationScore>,
    ) -> f64 {
        let mut combined_score = anomaly_score.score.clamp(0.0, 1.0) * 0.4;
        combined_score += outlier_score.score.clamp(0.0, 1.0) * 0.3;

        if let Some(verification) = verification_score {
            combined_score += (1.0 - verification.score.clamp(0.0, 1.0)) * 0.3;
        }

        combined_score
    }

    /// Confidence in the round's aggregate, in `[0, 1]`.
    ///
    /// The product of two measured quantities: the fraction of the submitting
    /// cohort that survived detection, and how tightly the survivors cluster around
    /// the aggregate (their mean distance to it, relative to the median update
    /// norm).
    fn calculate_confidence_score(
        &self,
        honest_participants: &HashMap<String, Array1<T>>,
        total_submitted: usize,
        aggregate: &Array1<T>,
    ) -> Result<f64> {
        if total_submitted == 0 || honest_participants.is_empty() {
            return Ok(0.0);
        }

        let survivor_fraction =
            (honest_participants.len() as f64 / total_submitted as f64).min(1.0);

        let ordered = Self::ordered_cohort(honest_participants);
        let mut dispersion = 0.0f64;
        let mut norms: Vec<f64> = Vec::with_capacity(ordered.len());
        for (_, gradient) in &ordered {
            dispersion += from_scalar(euclidean_distance(gradient, aggregate)?)?;
            norms.push(from_scalar(l2_norm(gradient))?);
        }
        dispersion /= ordered.len() as f64;

        norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let scale = if norms.len().is_multiple_of(2) {
            let mid = norms.len() / 2;
            (norms[mid - 1] + norms[mid]) / 2.0
        } else {
            norms[norms.len() / 2]
        };

        let consensus = if scale > 0.0 {
            1.0 / (1.0 + dispersion / scale)
        } else if dispersion == 0.0 {
            1.0
        } else {
            0.0
        };

        Ok((survivor_fraction * consensus).clamp(0.0, 1.0))
    }

    /// Record this round's observed behaviour for every submitting participant.
    fn record_behavior(
        &mut self,
        submitted: &HashMap<String, Array1<T>>,
        filtered: &HashMap<String, Array1<T>>,
        anomaly_results: &HashMap<String, AnomalyScore>,
        aggregate: &Array1<T>,
    ) -> Result<()> {
        let ordered: Vec<(String, Array1<T>)> = Self::ordered_cohort(submitted)
            .into_iter()
            .map(|(id, g)| (id.clone(), g.clone()))
            .collect();

        for (participant_id, gradient) in ordered {
            let participated = filtered.contains_key(&participant_id);
            let norm = from_scalar(l2_norm(&gradient))?;
            let similarity = from_scalar(cosine_similarity(&gradient, aggregate)?)?;
            let anomaly = anomaly_results
                .get(&participant_id)
                .map(|score| score.score.clamp(0.0, 1.0));

            let history = self.behavior_history.entry(participant_id).or_default();
            BehaviorHistory::push_bounded(&mut history.participation_pattern, participated);

            if participated {
                history.rounds_participated = history.rounds_participated.saturating_add(1);
                BehaviorHistory::push_bounded(&mut history.gradient_norms, norm);
                BehaviorHistory::push_bounded(&mut history.gradient_similarities, similarity);
                if let Some(anomaly) = anomaly {
                    BehaviorHistory::push_bounded(&mut history.anomaly_scores, anomaly);
                }
            }
        }

        Ok(())
    }

    /// Update participant reputations based on aggregation results.
    ///
    /// Reputation is an exponential moving average driven by `reputation_decay`;
    /// see [`ByzantineConfig::reputation_decay`].
    fn update_reputations(
        &mut self,
        honest_participants: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<()> {
        let decay = self.config.reputation_decay;
        let honest_rate = 1.0 - decay;
        // Misbehaviour is punished faster than good behaviour is rewarded.
        let byzantine_rate = (honest_rate * 5.0).min(1.0);

        let mut honest_ids: Vec<&String> = honest_participants.keys().collect();
        honest_ids.sort();

        for participant_id in honest_ids {
            let quality = self.derived_quality(participant_id);
            let reputation = self
                .reputation_scores
                .entry(participant_id.clone())
                .or_default();

            reputation.successful_aggregations =
                reputation.successful_aggregations.saturating_add(1);
            reputation.score = (decay * reputation.score + honest_rate).clamp(0.0, 1.0);
            if let Some((gradient_quality, consistency)) = quality {
                reputation.gradient_quality = gradient_quality;
                reputation.consistency_score = consistency;
            }

            reputation.trust_level = match reputation.score {
                s if s >= 0.8 => TrustLevel::High,
                s if s >= 0.5 => TrustLevel::Medium,
                _ => TrustLevel::Low,
            };
        }

        for participant_id in byzantine_participants {
            let quality = self.derived_quality(participant_id);
            let reputation = self
                .reputation_scores
                .entry(participant_id.clone())
                .or_default();

            reputation.detected_anomalies = reputation.detected_anomalies.saturating_add(1);
            reputation.score = (reputation.score * (1.0 - byzantine_rate)).clamp(0.0, 1.0);
            if let Some((gradient_quality, consistency)) = quality {
                reputation.gradient_quality = gradient_quality;
                reputation.consistency_score = consistency;
            }

            reputation.trust_level = if reputation.score < 0.1 {
                TrustLevel::Blacklisted
            } else if reputation.score < 0.3 {
                TrustLevel::Low
            } else {
                TrustLevel::Medium
            };
        }

        Ok(())
    }

    /// Gradient quality and consistency derived from the recorded behaviour history.
    fn derived_quality(&self, participant_id: &str) -> Option<(f64, f64)> {
        let history = self.behavior_history.get(participant_id)?;
        let quality = 1.0 - history.mean_anomaly_score()?;
        let consistency = (history.mean_similarity()? + 1.0) / 2.0;
        Some((quality.clamp(0.0, 1.0), consistency.clamp(0.0, 1.0)))
    }

    /// Feed this round's outcome to the behaviour pattern model.
    fn learn_patterns(
        &mut self,
        honest_participants: &HashMap<String, Array1<T>>,
        filtered: &HashMap<String, Array1<T>>,
        byzantine_participants: &[String],
    ) -> Result<()> {
        let honest: Vec<Array1<T>> = Self::ordered_cohort(honest_participants)
            .into_iter()
            .map(|(_, g)| g.clone())
            .collect();
        for gradient in honest {
            self.anomaly_detector.learn_normal(&gradient)?;
        }

        let attacks: Vec<Array1<T>> = byzantine_participants
            .iter()
            .filter_map(|id| filtered.get(id).cloned())
            .collect();
        for gradient in attacks {
            self.anomaly_detector.learn_attack(&gradient)?;
        }

        Ok(())
    }

    /// Compute Euclidean distance between two gradients
    pub fn compute_euclidean_distance(&self, a: &Array1<T>, b: &Array1<T>) -> Result<T> {
        euclidean_distance(a, b)
    }

    /// Compute cosine similarity between two gradients
    pub fn compute_cosine_similarity(&self, a: &Array1<T>, b: &Array1<T>) -> Result<T> {
        cosine_similarity(a, b)
    }

    /// Get reputation updates
    fn get_reputation_updates(&self) -> HashMap<String, ReputationScore> {
        self.reputation_scores.clone()
    }
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

impl Default for ReputationScore {
    fn default() -> Self {
        Self::new()
    }
}

impl ReputationScore {
    /// Create new reputation score with default values
    pub fn new() -> Self {
        Self {
            score: 0.7, // Start with medium trust
            successful_aggregations: 0,
            detected_anomalies: 0,
            gradient_quality: 0.5,
            consistency_score: 0.5,
            trust_level: TrustLevel::Medium,
        }
    }
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

        // Only measured quantities enter the score: before the pattern model has
        // learned anything it contributes nothing at all rather than a constant.
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

        // The new sample joins the baseline only after it has been scored.
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
    fn compute_norm_deviation(&self, participant_id: &str, gradient: &Array1<T>) -> Result<f64> {
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
            // Normalise by the three-sigma rule, as everywhere else in this module.
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
    fn learn_into(
        patterns: &mut Vec<Array1<T>>,
        gradient: &Array1<T>,
        matching_threshold: f64,
    ) -> Result<()> {
        let norm = l2_norm(gradient);
        if !norm.is_finite() || norm <= T::zero() {
            // A zero or malformed update carries no direction to learn from.
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
    /// - including after a model reshape, when prototypes exist but none of them
    /// applies - so the caller never folds a constant into a combined score.
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
    fn best_similarity(patterns: &[Array1<T>], gradient: &Array1<T>) -> Result<Option<f64>> {
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

            // Standardised third and fourth moments, guarding sigma > 0.
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
                // A constant coordinate has no defined shape; report the normal
                // reference values.
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap;

    fn cohort(entries: &[(&str, &[f64])]) -> HashMap<String, Array1<f64>> {
        entries
            .iter()
            .map(|(id, values)| ((*id).to_string(), Array1::from(values.to_vec())))
            .collect()
    }

    fn aggregator(
        method: ByzantineAggregationMethod,
        max_byzantine: usize,
        min_participants: usize,
    ) -> ByzantineTolerantAggregator<f64> {
        let config = ByzantineConfig {
            max_byzantine,
            min_participants,
            aggregation_method: method,
            gradient_verification: false,
            ..ByzantineConfig::default()
        };
        ByzantineTolerantAggregator::new(config).expect("configuration must be valid")
    }

    // -- configuration ------------------------------------------------------

    #[test]
    fn test_byzantine_config_accepts_sound_settings() {
        let config = ByzantineConfig {
            max_byzantine: 2,
            min_participants: 7,
            aggregation_method: ByzantineAggregationMethod::Krum,
            anomaly_threshold: 0.5,
            reputation_decay: 0.9,
            gradient_verification: true,
            outlier_detection: OutlierDetectionMethod::ZScore,
            consensus_threshold: 0.7,
        };

        assert_eq!(config.max_byzantine, 2);
        assert_eq!(config.min_participants, 7);
        assert_eq!(config.required_participants(), 7);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_byzantine_config_rejects_undersized_krum_cohort() {
        // Krum needs n >= 2f + 3 = 7; the old code accepted this and then underflowed.
        let config = ByzantineConfig {
            max_byzantine: 2,
            min_participants: 5,
            aggregation_method: ByzantineAggregationMethod::Krum,
            ..ByzantineConfig::default()
        };
        assert!(config.validate().is_err());
        assert!(ByzantineTolerantAggregator::<f64>::new(config).is_err());
    }

    #[test]
    fn test_byzantine_config_rejects_zero_denominator_state() {
        // This is the config that used to make confidence_score NaN.
        let config = ByzantineConfig {
            max_byzantine: 0,
            min_participants: 0,
            aggregation_method: ByzantineAggregationMethod::Median,
            ..ByzantineConfig::default()
        };
        assert!(config.validate().is_err());

        for bad in [0.0, -0.1, 1.5, f64::NAN] {
            let config = ByzantineConfig {
                consensus_threshold: bad,
                ..ByzantineConfig::default()
            };
            assert!(config.validate().is_err(), "consensus_threshold {bad}");

            let config = ByzantineConfig {
                anomaly_threshold: bad,
                ..ByzantineConfig::default()
            };
            assert!(config.validate().is_err(), "anomaly_threshold {bad}");
        }

        for bad in [1.0, 1.5, -0.1, f64::NAN] {
            let config = ByzantineConfig {
                reputation_decay: bad,
                ..ByzantineConfig::default()
            };
            assert!(config.validate().is_err(), "reputation_decay {bad}");
        }
    }

    #[test]
    fn test_bulyan_config_requires_four_f_plus_three() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 6,
            aggregation_method: ByzantineAggregationMethod::Bulyan,
            ..ByzantineConfig::default()
        };
        assert_eq!(config.required_participants(), 7);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_reputation_score() {
        let mut reputation = ReputationScore::new();
        assert_eq!(reputation.score, 0.7);
        assert_eq!(reputation.successful_aggregations, 0);
        assert_eq!(reputation.trust_level, TrustLevel::Medium);

        reputation.successful_aggregations += 1;
        reputation.score = 0.9;
        assert_eq!(reputation.successful_aggregations, 1);
    }

    // -- input validation ---------------------------------------------------

    #[test]
    fn test_non_finite_gradients_are_rejected() {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 2.0]),
            ("b", &[1.0, f64::NAN]),
            ("c", &[1.0, 2.0]),
        ]);
        assert!(agg.byzantine_robust_aggregate(&gradients).is_err());

        let gradients = cohort(&[("a", &[1.0, 2.0]), ("b", &[1.0, f64::INFINITY])]);
        assert!(agg.byzantine_robust_aggregate(&gradients).is_err());
    }

    #[test]
    fn test_ragged_cohort_is_rejected() {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        let gradients = cohort(&[("a", &[1.0, 2.0]), ("b", &[1.0]), ("c", &[1.0, 2.0])]);
        assert!(agg.byzantine_robust_aggregate(&gradients).is_err());
    }

    // -- trimmed mean (F12, F13) --------------------------------------------

    #[test]
    fn test_trimmed_mean_rejects_cohort_too_small_to_trim() {
        // Regression: n = 2 with one trimmed value per tail used to slice an empty
        // range and silently return an all-zero update.
        let agg = aggregator(ByzantineAggregationMethod::TrimmedMean, 1, 3);
        let gradients = cohort(&[("a", &[1.0, 2.0]), ("b", &[3.0, 4.0])]);

        let result = agg.trimmed_mean_aggregation(&gradients, 1);
        assert!(
            result.is_err(),
            "n = 2 with trim 1 per tail must be an error"
        );

        // n = 3 with trim 1 keeps exactly the median and is accepted.
        let gradients = cohort(&[("a", &[1.0]), ("b", &[2.0]), ("c", &[100.0])]);
        let result = agg
            .trimmed_mean_aggregation(&gradients, 1)
            .expect("n = 3 with trim 1 must succeed");
        assert!((result[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_trims_max_byzantine_per_tail() {
        // Regression for the hardcoded 10% trim: with n = 5 and f = 2, Yin et al.
        // requires two values dropped from each tail (leaving the median alone).
        // The old 10% rule dropped one, letting a Byzantine value survive.
        let agg = aggregator(ByzantineAggregationMethod::TrimmedMean, 2, 5);
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[2.0]),
            ("c", &[3.0]),
            ("d", &[50.0]),
            ("e", &[100.0]),
        ]);

        let robust = agg
            .trimmed_mean_aggregation(&gradients, 2)
            .expect("trimming 2 per tail from 5 gradients must succeed");
        assert!((robust[0] - 3.0).abs() < 1e-12, "got {}", robust[0]);

        // What the old hardcoded 10% (one per tail) would have produced.
        let weak = agg
            .trimmed_mean_aggregation(&gradients, 1)
            .expect("trimming 1 per tail must succeed");
        assert!((weak[0] - 18.333_333_333).abs() < 1e-6, "got {}", weak[0]);
    }

    #[test]
    fn test_trimmed_mean_aggregation() {
        let agg = aggregator(ByzantineAggregationMethod::TrimmedMean, 1, 3);

        let gradients = cohort(&[
            ("client1", &[1.0, 2.0, 3.0]),
            ("client2", &[1.1, 2.1, 3.1]),
            ("client3", &[0.9, 1.9, 2.9]),
            ("client4", &[1.0, 2.0, 3.0]),
            ("client5", &[10.0, 20.0, 30.0]),
        ]);

        let result = agg
            .trimmed_mean_aggregation(&gradients, 1)
            .expect("trimmed mean must succeed");

        // Trimming one value per tail removes both the low and the high extreme.
        assert!(
            (result[0] - 1.033_333_333).abs() < 1e-6,
            "got {}",
            result[0]
        );
        assert!(
            (result[1] - 2.033_333_333).abs() < 1e-6,
            "got {}",
            result[1]
        );
        assert!(
            (result[2] - 3.033_333_333).abs() < 1e-6,
            "got {}",
            result[2]
        );
    }

    #[test]
    fn test_coordinate_median_aggregation() {
        let agg = aggregator(ByzantineAggregationMethod::CoordinateMedian, 1, 3);
        let gradients = cohort(&[
            ("client1", &[1.0, 2.0, 3.0]),
            ("client2", &[2.0, 3.0, 4.0]),
            ("client3", &[3.0, 4.0, 5.0]),
        ]);

        let result = agg
            .coordinate_median_aggregation(&gradients)
            .expect("median must succeed");

        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
    }

    // -- Krum family (F11, F14) ---------------------------------------------

    #[test]
    fn test_krum_rejects_undersized_cohort_without_panicking() {
        // Regression: `participants.len() - max_byzantine - 2` underflowed for
        // f = 2 and 3 survivors, panicking in debug and wrapping in release.
        let agg = aggregator(ByzantineAggregationMethod::Krum, 2, 7);
        let gradients = cohort(&[("a", &[1.0]), ("b", &[2.0]), ("c", &[3.0])]);

        assert!(agg.krum_aggregation(&gradients, 2).is_err());
        assert!(agg.multi_krum_aggregation(&gradients, 2).is_err());
        assert!(agg.bulyan_aggregation(&gradients, 2).is_err());
    }

    #[test]
    fn test_krum_selects_a_representative_gradient() {
        let agg = aggregator(ByzantineAggregationMethod::Krum, 0, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.1, 1.1]),
            ("c", &[100.0, 100.0]),
        ]);

        let result = agg.krum_aggregation(&gradients, 0).expect("krum");
        assert!(
            result[0] < 2.0,
            "Krum must not pick the outlier: {result:?}"
        );
    }

    #[test]
    fn test_multi_krum_averages_the_selected_gradients() {
        let agg = aggregator(ByzantineAggregationMethod::MultiKrum, 1, 5);
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[1.0]),
            ("c", &[1.0]),
            ("d", &[1.0]),
            ("e", &[100.0]),
        ]);

        let result = agg
            .multi_krum_aggregation(&gradients, 1)
            .expect("multi-krum");
        assert!((result[0] - 1.0).abs() < 1e-12, "got {}", result[0]);
    }

    #[test]
    fn test_bulyan_runs_iterative_krum_and_median_selection() {
        // n = 7, f = 1 satisfies n >= 4f + 3. theta = 5, beta = 3.
        let agg = aggregator(ByzantineAggregationMethod::Bulyan, 1, 7);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.0]),
            ("c", &[1.0, 1.0]),
            ("d", &[1.0, 1.0]),
            ("e", &[1.0, 1.0]),
            ("f", &[1.0, 1.0]),
            ("g", &[500.0, -500.0]),
        ]);

        let result = agg.bulyan_aggregation(&gradients, 1).expect("bulyan");
        assert!((result[0] - 1.0).abs() < 1e-12, "got {}", result[0]);
        assert!((result[1] - 1.0).abs() < 1e-12, "got {}", result[1]);
    }

    #[test]
    fn test_bulyan_no_longer_zeroes_the_update() {
        // Regression: the old implementation handed 2 gradients to a trimmed mean
        // that trimmed them both, returning Ok(zeros).
        let agg = aggregator(ByzantineAggregationMethod::Bulyan, 3, 15);
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[2.0]),
            ("c", &[3.0]),
            ("d", &[4.0]),
            ("e", &[5.0]),
        ]);

        // n = 5 < 4f + 3 = 15, so this is an explicit error rather than zeros.
        assert!(agg.bulyan_aggregation(&gradients, 3).is_err());
    }

    // -- outlier detectors (F10, F20) ---------------------------------------

    fn outlier_scores(
        method: OutlierDetectionMethod,
        entries: &[(&str, &[f64])],
    ) -> Vec<OutlierScore> {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        agg.config.outlier_detection = method;
        let gradients = cohort(entries);
        let ordered = ByzantineTolerantAggregator::<f64>::ordered_cohort(&gradients);
        let grads: Vec<&Array1<f64>> = ordered.iter().map(|(_, g)| *g).collect();
        let stats = agg
            .statistics_engine
            .compute_statistics(&grads)
            .expect("statistics");
        (0..grads.len())
            .map(|i| {
                agg.compute_outlier_score(i, &grads, &stats)
                    .expect("outlier score")
            })
            .collect()
    }

    #[test]
    fn test_every_outlier_method_produces_a_bounded_score() {
        // Regression: IsolationForest / LocalOutlierFactor / MahalanobisDistance
        // fell into a wildcard arm that recursed into itself forever.
        let entries: &[(&str, &[f64])] = &[
            ("a", &[1.0, 1.0, 1.0]),
            ("b", &[1.1, 0.9, 1.0]),
            ("c", &[0.9, 1.1, 1.0]),
            ("d", &[1.0, 1.0, 1.1]),
            ("e", &[50.0, -50.0, 50.0]),
        ];

        for method in [
            OutlierDetectionMethod::ZScore,
            OutlierDetectionMethod::IQR,
            OutlierDetectionMethod::IsolationForest,
            OutlierDetectionMethod::LocalOutlierFactor,
            OutlierDetectionMethod::MahalanobisDistance,
        ] {
            let scores = outlier_scores(method, entries);
            assert_eq!(scores.len(), entries.len());
            for score in &scores {
                assert!(score.score.is_finite(), "{method:?} produced {score:?}");
                assert!(
                    (0.0..=1.0).contains(&score.score),
                    "{method:?} produced {score:?}"
                );
                assert_eq!(score.method, method);
            }
            // The planted outlier is index 4 in id order.
            assert!(
                scores[4].score >= scores[0].score,
                "{method:?} did not rank the outlier at least as high as an inlier"
            );
        }
    }

    #[test]
    fn test_iqr_score_is_finite_when_the_iqr_is_zero() {
        // Regression: [1, 1, 1, 1, 10] gives q1 == q3 == 1, so the dissenter used to
        // divide by zero and score +inf.
        let entries: &[(&str, &[f64])] = &[
            ("a", &[1.0]),
            ("b", &[1.0]),
            ("c", &[1.0]),
            ("d", &[1.0]),
            ("e", &[10.0]),
        ];
        let scores = outlier_scores(OutlierDetectionMethod::IQR, entries);
        for score in &scores {
            assert!(score.score.is_finite(), "{score:?}");
        }
        assert!(scores[4].score > 0.0, "the dissenter must score above zero");
    }

    #[test]
    fn test_outlier_detectors_are_deterministic() {
        let entries: &[(&str, &[f64])] = &[
            ("a", &[1.0, 2.0]),
            ("b", &[1.2, 1.9]),
            ("c", &[0.8, 2.1]),
            ("d", &[30.0, -30.0]),
        ];
        for method in [
            OutlierDetectionMethod::IsolationForest,
            OutlierDetectionMethod::LocalOutlierFactor,
        ] {
            let first = outlier_scores(method, entries);
            let second = outlier_scores(method, entries);
            for (a, b) in first.iter().zip(second.iter()) {
                assert_eq!(a.score.to_bits(), b.score.to_bits(), "{method:?}");
            }
        }
    }

    // -- statistics (F23) ---------------------------------------------------

    #[test]
    fn test_skewness_and_kurtosis_are_computed() {
        let mut engine = StatisticalAnalysis::<f64>::new();
        let a = Array1::from(vec![-1.0]);
        let b = Array1::from(vec![0.0]);
        let c = Array1::from(vec![1.0]);
        let measures = engine
            .compute_statistics(&[&a, &b, &c])
            .expect("statistics");

        // Symmetric sample: skewness 0, standardised fourth moment 1.5.
        assert!(measures.skewness[0].abs() < 1e-12);
        assert!((measures.kurtosis[0] - 1.5).abs() < 1e-12);

        // A right-skewed sample must report a positive third moment.
        let a = Array1::from(vec![0.0]);
        let b = Array1::from(vec![0.0]);
        let c = Array1::from(vec![0.0]);
        let d = Array1::from(vec![10.0]);
        let measures = engine
            .compute_statistics(&[&a, &b, &c, &d])
            .expect("statistics");
        assert!(measures.skewness[0] > 0.5, "got {}", measures.skewness[0]);

        // A constant coordinate reports the normal reference values.
        let a = Array1::from(vec![2.0]);
        let b = Array1::from(vec![2.0]);
        let measures = engine.compute_statistics(&[&a, &b]).expect("statistics");
        assert_eq!(measures.skewness[0], 0.0);
        assert_eq!(measures.kurtosis[0], 3.0);
    }

    #[test]
    fn test_quartiles_are_reported() {
        let mut engine = StatisticalAnalysis::<f64>::new();
        let values: Vec<Array1<f64>> = (0..5).map(|i| Array1::from(vec![i as f64])).collect();
        let refs: Vec<&Array1<f64>> = values.iter().collect();
        let measures = engine.compute_statistics(&refs).expect("statistics");

        assert_eq!(measures.q1[0], 1.0);
        assert_eq!(measures.q3[0], 3.0);
        assert_eq!(measures.iqr[0], 2.0);
        assert_eq!(measures.median[0], 2.0);
    }

    // -- FoolsGold (F15) ----------------------------------------------------

    #[test]
    fn test_fools_gold_penalises_sybils() {
        let mut agg = aggregator(ByzantineAggregationMethod::FoolsGold, 1, 3);
        // Three colluding clients share a direction; two honest clients do not.
        let gradients = cohort(&[
            ("honest1", &[1.0, 0.0, 0.0, 0.0]),
            ("honest2", &[0.0, 1.0, 0.0, 0.0]),
            ("sybil1", &[0.0, 0.0, 1.0, 1.0]),
            ("sybil2", &[0.0, 0.0, 1.0, 1.0]),
            ("sybil3", &[0.0, 0.0, 1.0, 1.0]),
        ]);

        let result = agg.fools_gold_aggregation(&gradients).expect("fools gold");

        // The Sybil direction is (0, 0, 1, 1); it must be suppressed relative to the
        // honest directions even though it is the numerical majority.
        let honest_mass = result[0] + result[1];
        let sybil_mass = result[2] + result[3];
        assert!(
            honest_mass > sybil_mass * 10.0,
            "Sybils were not suppressed: {result:?}"
        );

        let ids: Vec<String> = vec![
            "honest1".to_string(),
            "honest2".to_string(),
            "sybil1".to_string(),
            "sybil2".to_string(),
            "sybil3".to_string(),
        ];
        let weights = agg.compute_fools_gold_weights(&ids).expect("weights");
        assert!(weights[0] > 0.9 && weights[1] > 0.9, "{weights:?}");
        assert!(
            weights[2] < 0.1 && weights[3] < 0.1 && weights[4] < 0.1,
            "{weights:?}"
        );
    }

    #[test]
    fn test_fools_gold_is_not_a_plain_mean() {
        // Regression: the old implementation returned reputation-weighted means, so
        // a fresh aggregator produced exactly mean(gradients).
        let mut agg = aggregator(ByzantineAggregationMethod::FoolsGold, 1, 3);
        let gradients = cohort(&[
            ("honest", &[1.0, 0.0]),
            ("sybil1", &[0.0, 1.0]),
            ("sybil2", &[0.0, 1.0]),
            ("sybil3", &[0.0, 1.0]),
        ]);

        let result = agg.fools_gold_aggregation(&gradients).expect("fools gold");
        let mean_second = 0.75;
        assert!(
            (result[1] - mean_second).abs() > 0.1,
            "FoolsGold degenerated into the mean: {result:?}"
        );
    }

    // -- FLAME (F16, F17) ---------------------------------------------------

    #[test]
    fn test_flame_clustering_is_deterministic() {
        // Regression: the cluster seed came from HashMap::iter().next(), so the
        // admitted set changed between processes.
        let agg = aggregator(ByzantineAggregationMethod::FLAME, 1, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.1]),
            ("c", &[0.9, 1.0]),
            ("d", &[-1.0, -1.0]),
            ("e", &[-1.0, -0.9]),
        ]);

        let first = agg.flame_admitted_ids(&gradients).expect("clustering");
        for _ in 0..25 {
            let again = agg.flame_admitted_ids(&gradients).expect("clustering");
            assert_eq!(first, again);
        }
        assert!(first.len() > gradients.len() / 2);
        assert!(first.contains(&"a".to_string()));
        assert!(!first.contains(&"d".to_string()));
    }

    #[test]
    fn test_flame_clips_scaling_attacks() {
        let agg = aggregator(ByzantineAggregationMethod::FLAME, 1, 3);
        // The attacker agrees in direction (so it survives clustering) but inflates
        // its norm by 1000x; norm-median clipping must neutralise that.
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.0]),
            ("c", &[1.0, 1.0]),
            ("d", &[1.0, 1.0]),
            ("attacker", &[1000.0, 1000.0]),
        ]);

        let result = agg.flame_aggregation(&gradients).expect("flame");
        assert!(
            result[0] < 2.0,
            "clipping did not bound the attack: {result:?}"
        );
        assert!(
            result[0] > 0.5,
            "the honest signal was destroyed: {result:?}"
        );
    }

    #[test]
    fn test_flame_noise_is_small_relative_to_the_update() {
        let agg = aggregator(ByzantineAggregationMethod::FLAME, 1, 3);
        let gradients = cohort(&[("a", &[1.0]), ("b", &[1.0]), ("c", &[1.0])]);
        let result = agg.flame_aggregation(&gradients).expect("flame");
        assert!((result[0] - 1.0).abs() < 0.05, "got {}", result[0]);
    }

    // -- anomaly detection (F18, F19) ---------------------------------------

    #[test]
    fn test_anomaly_history_is_per_participant_and_excludes_the_sample() {
        let mut detector = AnomalyDetector::<f64>::new(0.5);

        // Alice builds a history of comparable, slightly varying updates.
        for magnitude in [1.0, 1.1, 0.9, 1.05, 0.95] {
            let score = detector
                .detect_anomaly("alice", &Array1::from(vec![magnitude, 0.0]))
                .expect("anomaly");
            assert!(score.score.is_finite());
        }

        // A huge update from Alice deviates from her own baseline...
        let alice_attack = detector
            .detect_anomaly("alice", &Array1::from(vec![100.0, 0.0]))
            .expect("anomaly");
        assert!(alice_attack.score > 0.0, "{alice_attack:?}");
        assert!(alice_attack.is_anomalous, "{alice_attack:?}");

        // ...and Bob, who has no history of his own, is unaffected by it.
        let bob = detector
            .detect_anomaly("bob", &Array1::from(vec![1.0, 0.0]))
            .expect("anomaly");
        assert_eq!(bob.score, 0.0, "{bob:?}");

        let alice_history = detector
            .gradient_stats()
            .norm_history_of("alice")
            .expect("alice history");
        let bob_history = detector
            .gradient_stats()
            .norm_history_of("bob")
            .expect("bob history");
        assert_eq!(alice_history.len(), 6);
        assert_eq!(bob_history.len(), 1);
    }

    #[test]
    fn test_anomaly_score_has_no_constant_pattern_term() {
        // Regression: compute_pattern_deviation always returned 0.5, so the combined
        // score could never drop below 0.25.
        let mut detector = AnomalyDetector::<f64>::new(0.5);
        for _ in 0..5 {
            let score = detector
                .detect_anomaly("alice", &Array1::from(vec![1.0, 0.0]))
                .expect("anomaly");
            assert!(
                score.score < 0.25,
                "constant pattern term survives: {score:?}"
            );
        }
        assert_eq!(detector.pattern_counts(), (0, 0));
    }

    #[test]
    fn test_pattern_model_learns_and_scores() {
        let mut model = PatternModel::<f64>::new();
        assert_eq!(
            model
                .compute_pattern_deviation(&Array1::from(vec![1.0, 0.0]))
                .expect("deviation"),
            None
        );

        model
            .learn_normal(&Array1::from(vec![1.0, 0.0]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (1, 0));

        // A second, near-identical observation is merged rather than appended.
        model
            .learn_normal(&Array1::from(vec![2.0, 0.05]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (1, 0));

        // A different direction becomes a new prototype.
        model
            .learn_normal(&Array1::from(vec![0.0, 1.0]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (2, 0));

        let aligned = model
            .compute_pattern_deviation(&Array1::from(vec![3.0, 0.0]))
            .expect("deviation")
            .expect("some");
        let opposite = model
            .compute_pattern_deviation(&Array1::from(vec![-1.0, -1.0]))
            .expect("deviation")
            .expect("some");
        assert!(aligned < 0.1, "aligned update scored {aligned}");
        assert!(opposite > aligned, "opposite update scored {opposite}");

        // A known attack direction is flagged even when it is not novel.
        model
            .learn_attack(&Array1::from(vec![0.0, -1.0]))
            .expect("learn");
        assert_eq!(model.pattern_counts(), (2, 1));
        let attack = model
            .compute_pattern_deviation(&Array1::from(vec![0.0, -5.0]))
            .expect("deviation")
            .expect("some");
        assert!(attack > 0.9, "attack direction scored {attack}");
        assert!(model.matching_threshold() > 0.0);
    }

    // -- confidence and reputation (F21, F22) -------------------------------

    #[test]
    fn test_confidence_score_is_finite_and_meaningful() {
        let agg = aggregator(ByzantineAggregationMethod::Median, 0, 1);

        let tight = cohort(&[("a", &[1.0]), ("b", &[1.0]), ("c", &[1.0])]);
        let aggregate = Array1::from(vec![1.0]);
        let tight_score = agg
            .calculate_confidence_score(&tight, 3, &aggregate)
            .expect("confidence");
        assert!(tight_score.is_finite());
        assert!((tight_score - 1.0).abs() < 1e-9, "got {tight_score}");

        let spread = cohort(&[("a", &[1.0]), ("b", &[-3.0]), ("c", &[7.0])]);
        let spread_score = agg
            .calculate_confidence_score(&spread, 6, &aggregate)
            .expect("confidence");
        assert!(spread_score.is_finite());
        assert!(spread_score < tight_score, "got {spread_score}");
        assert!((0.0..=1.0).contains(&spread_score));
    }

    #[test]
    fn test_reputation_uses_the_configured_decay() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 3,
            aggregation_method: ByzantineAggregationMethod::Median,
            reputation_decay: 0.5,
            gradient_verification: false,
            ..ByzantineConfig::default()
        };
        let mut agg = ByzantineTolerantAggregator::<f64>::new(config).expect("config");

        let honest = cohort(&[("good", &[1.0])]);
        agg.update_reputations(&honest, &["bad".to_string()])
            .expect("reputations");

        // Honest: 0.5 * 0.7 + 0.5 = 0.85. Byzantine rate = min(0.5 * 5, 1) = 1.
        let good = agg.reputation("good").expect("good");
        assert!((good.score - 0.85).abs() < 1e-12, "got {}", good.score);
        assert_eq!(good.trust_level, TrustLevel::High);

        let bad = agg.reputation("bad").expect("bad");
        assert!(bad.score.abs() < 1e-12, "got {}", bad.score);
        assert_eq!(bad.trust_level, TrustLevel::Blacklisted);
    }

    #[test]
    fn test_gradient_verifier_enforces_expected_properties() {
        let verifier = GradientVerifier::<f64>::with_properties(GradientProperties {
            norm_range: (0.5, 10.0),
            sparsity_threshold: 0.5,
            direction_consistency: 0.0,
        });
        assert_eq!(verifier.expected_properties().sparsity_threshold, 0.5);

        let good = verifier
            .verify_gradient(&Array1::from(vec![1.0, 1.0]))
            .expect("verify");
        assert!(good.passed, "{good:?}");

        let too_large = verifier
            .verify_gradient(&Array1::from(vec![100.0, 100.0]))
            .expect("verify");
        assert!(too_large.score < good.score, "{too_large:?}");
        assert_eq!(too_large.rule_scores.get("Norm range"), Some(&0.0));

        let too_sparse = verifier
            .verify_gradient(&Array1::from(vec![1.0, 0.0, 0.0, 0.0]))
            .expect("verify");
        assert_eq!(too_sparse.rule_scores.get("Sparsity"), Some(&0.0));

        let mut verifier = verifier;
        verifier.set_reference_direction(Array1::from(vec![1.0, 1.0]));
        let opposed = verifier
            .verify_gradient(&Array1::from(vec![-1.0, -1.0]))
            .expect("verify");
        assert_eq!(opposed.rule_scores.get("Direction consistency"), Some(&0.0));
    }

    // -- full pipeline ------------------------------------------------------

    #[test]
    fn test_full_pipeline_records_behaviour_and_reputations() {
        let config = ByzantineConfig {
            max_byzantine: 1,
            min_participants: 4,
            aggregation_method: ByzantineAggregationMethod::CoordinateMedian,
            outlier_detection: OutlierDetectionMethod::ZScore,
            gradient_verification: true,
            consensus_threshold: 0.5,
            ..ByzantineConfig::default()
        };
        let mut agg = ByzantineTolerantAggregator::<f64>::new(config).expect("config");

        let gradients = cohort(&[
            ("a", &[1.0, 2.0]),
            ("b", &[1.1, 2.1]),
            ("c", &[0.9, 1.9]),
            ("d", &[1.0, 2.0]),
            ("e", &[1.05, 1.95]),
        ]);

        let result = agg
            .byzantine_robust_aggregate(&gradients)
            .expect("aggregation");

        assert!(result.aggregate.iter().all(|x| x.is_finite()));
        assert!(result.confidence_score.is_finite());
        assert!((0.0..=1.0).contains(&result.confidence_score));
        assert!((0.0..=1.0).contains(&result.consensus_ratio));
        assert_eq!(agg.round(), 1);

        let mut expected: Vec<String> = gradients.keys().cloned().collect();
        expected.sort();
        assert_eq!(result.honest_participants, expected);

        for id in &expected {
            let history = agg.behavior_history(id).expect("behaviour history");
            assert_eq!(history.rounds_participated, 1);
            assert_eq!(history.gradient_norms.len(), 1);
            assert_eq!(history.gradient_similarities.len(), 1);
            assert_eq!(history.participation_pattern, vec![true]);

            let reputation = agg.reputation(id).expect("reputation");
            assert_eq!(reputation.successful_aggregations, 1);
            assert!(reputation.gradient_quality.is_finite());
            assert!(reputation.consistency_score > 0.5);
        }

        // The pattern model has now learned from the accepted updates.
        assert!(agg.anomaly_detector.pattern_counts().0 > 0);
    }

    #[test]
    fn test_full_pipeline_is_deterministic() {
        let gradients = cohort(&[
            ("a", &[1.0, 2.0]),
            ("b", &[1.1, 2.1]),
            ("c", &[0.9, 1.9]),
            ("d", &[1.0, 2.0]),
            ("e", &[3.0, -4.0]),
        ]);

        let run = || {
            let config = ByzantineConfig {
                max_byzantine: 1,
                min_participants: 3,
                aggregation_method: ByzantineAggregationMethod::CoordinateMedian,
                outlier_detection: OutlierDetectionMethod::LocalOutlierFactor,
                gradient_verification: true,
                ..ByzantineConfig::default()
            };
            let mut agg = ByzantineTolerantAggregator::<f64>::new(config).expect("config");
            agg.byzantine_robust_aggregate(&gradients)
                .expect("aggregation")
        };

        let first = run();
        let second = run();
        assert_eq!(first.honest_participants, second.honest_participants);
        assert_eq!(first.byzantine_participants, second.byzantine_participants);
        assert_eq!(
            first
                .aggregate
                .to_vec()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            second
                .aggregate
                .to_vec()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_consensus_threshold_gates_a_split_cohort() {
        let gradients = cohort(&[
            ("a", &[1.0]),
            ("b", &[1.0]),
            ("c", &[1.0]),
            ("d", &[900.0]),
            ("e", &[-900.0]),
        ]);

        let build = |consensus_threshold: f64| {
            let config = ByzantineConfig {
                max_byzantine: 1,
                min_participants: 3,
                aggregation_method: ByzantineAggregationMethod::Median,
                anomaly_threshold: 0.1,
                consensus_threshold,
                gradient_verification: false,
                ..ByzantineConfig::default()
            };
            ByzantineTolerantAggregator::<f64>::new(config).expect("config")
        };

        // The two extremes are flagged, so 3 of 5 participants survive.
        let mut permissive = build(0.5);
        let accepted = permissive
            .byzantine_robust_aggregate(&gradients)
            .expect("60% consensus clears a 50% threshold");
        assert_eq!(accepted.byzantine_participants.len(), 2);
        assert!((accepted.consensus_ratio - 0.6).abs() < 1e-12);
        assert!((accepted.aggregate[0] - 1.0).abs() < 1e-12);

        // The same round is rejected when 90% consensus is demanded.
        let mut strict = build(0.9);
        let error = strict
            .byzantine_robust_aggregate(&gradients)
            .expect_err("60% consensus must not clear a 90% threshold");
        assert!(
            format!("{error}").contains("consensus"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_blacklisted_participants_are_filtered() {
        let mut agg = aggregator(ByzantineAggregationMethod::Median, 1, 3);
        for _ in 0..5 {
            agg.update_reputations(&HashMap::new(), &["bad".to_string()])
                .expect("reputations");
        }
        assert_eq!(
            agg.reputation("bad").expect("bad").trust_level,
            TrustLevel::Blacklisted
        );

        let gradients = cohort(&[("good1", &[1.0]), ("good2", &[1.0]), ("bad", &[500.0])]);
        let filtered = agg.filter_by_reputation(&gradients).expect("filter");
        assert_eq!(filtered.len(), 2);
        assert!(!filtered.contains_key("bad"));
    }

    #[test]
    fn test_euclidean_distance() {
        let agg = aggregator(ByzantineAggregationMethod::Krum, 1, 5);

        let a = Array1::from(vec![1.0, 2.0, 3.0]);
        let b = Array1::from(vec![4.0, 5.0, 6.0]);

        let distance = agg
            .compute_euclidean_distance(&a, &b)
            .expect("distance must be computable");
        let expected = (3.0_f64.powi(2) + 3.0_f64.powi(2) + 3.0_f64.powi(2)).sqrt();

        assert!((distance - expected).abs() < 1e-10);
        assert!(agg
            .compute_euclidean_distance(&a, &Array1::from(vec![1.0]))
            .is_err());
    }

    #[test]
    fn test_cosine_similarity_handles_zero_norms() {
        let agg = aggregator(ByzantineAggregationMethod::Krum, 1, 5);
        let a = Array1::from(vec![1.0, 0.0]);
        let zero = Array1::from(vec![0.0, 0.0]);
        assert_eq!(
            agg.compute_cosine_similarity(&a, &zero)
                .expect("similarity"),
            0.0
        );
        assert!((agg.compute_cosine_similarity(&a, &a).expect("similarity") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_geometric_median_converges() {
        let agg = aggregator(ByzantineAggregationMethod::GeometricMedian, 1, 3);
        let gradients = cohort(&[
            ("a", &[1.0, 1.0]),
            ("b", &[1.0, 1.0]),
            ("c", &[1.0, 1.0]),
            ("d", &[100.0, 100.0]),
        ]);
        let result = agg
            .geometric_median_aggregation(&gradients)
            .expect("geometric median");
        assert!(result.iter().all(|x| x.is_finite()));
        assert!((result[0] - 1.0).abs() < 1e-6, "got {}", result[0]);
    }
}
