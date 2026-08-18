// Byzantine Robust Aggregation Module
//
// Byzantine-robust aggregation for federated learning: outlier detection over
// client updates, a reputation system, and a set of robust estimators that
// bound how far a malicious cohort can drag the aggregate.
//
// What changed and why
// --------------------
// An earlier revision of this module dispatched only two of its eight
// configured methods and *silently fell through to a plain arithmetic mean*
// for the rest. Plain averaging has an unbounded breakdown point: a single
// client can move the aggregate anywhere it likes. Selecting `Krum` and
// receiving FedAvg is therefore not a degradation, it is the removal of the
// entire guarantee -- while the configuration still claims it. Every method
// is now either implemented (see [`super::robust_ops`]) or rejected; nothing
// falls through.
//
// Two pieces of state were also declared but never written: `outlier_history`
// (which made [`ByzantineRobustAggregator::compute_robustness_factor`] return
// a constant 1.0, i.e. "no Byzantine behaviour ever observed", regardless of
// what happened) and the statistical analyzer's rolling window (which made
// `window_size` and `adaptive_threshold` dead configuration). Both are now
// populated and bounded.

use super::outlier_tests::{self, OutlierVerdict};
use super::robust_ops::{self, CohortMember, CENTERED_CLIPPING_ITERATIONS};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

/// Maximum number of outlier evaluations retained for
/// [`ByzantineRobustAggregator::compute_robustness_factor`].
pub const OUTLIER_HISTORY_CAPACITY: usize = 1000;

/// Byzantine-robust aggregation algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ByzantineRobustMethod {
    /// Coordinate-wise trimmed mean: discard `trim_ratio / 2` of the values
    /// from each tail of every coordinate before averaging.
    TrimmedMean {
        /// Total fraction of values discarded, split between the two tails.
        trim_ratio: f64,
    },

    /// Coordinate-wise median.
    CoordinateWiseMedian,

    /// Krum: return the single update closest to its `n - f - 2` nearest
    /// peers.
    Krum {
        /// Assumed number of Byzantine clients.
        f: usize,
    },

    /// Multi-Krum: average the `m` updates with the lowest Krum scores.
    MultiKrum {
        /// Assumed number of Byzantine clients.
        f: usize,
        /// Number of updates to average.
        m: usize,
    },

    /// Bulyan: iterated Krum selection followed by a median-proximity
    /// coordinate-wise average.
    Bulyan {
        /// Assumed number of Byzantine clients.
        f: usize,
    },

    /// Centered clipping around a robust centre, clipping each update's
    /// deviation to radius `tau`.
    CenteredClipping {
        /// Clipping radius.
        tau: f64,
    },

    /// FedAvg restricted to the clients whose outlier statistic stays within
    /// `threshold`.
    FedAvgOutlierDetection {
        /// Maximum absolute outlier statistic a client may have and still be
        /// averaged.
        threshold: f64,
    },

    /// Reputation-weighted averaging, with reputations decayed towards their
    /// initial value by `reputation_decay` before use.
    ReputationWeighted {
        /// Per-round mean reversion applied to reputations, in `[0, 1]`.
        reputation_decay: f64,
    },
}

/// Byzantine robustness configuration
#[derive(Debug, Clone)]
pub struct ByzantineRobustConfig {
    /// Aggregation method
    pub method: ByzantineRobustMethod,

    /// Upper bound on the fraction of the cohort that dynamic detection is
    /// allowed to exclude in one round. Must be in `[0, 0.5)`.
    pub expected_byzantine_ratio: f64,

    /// Run outlier detection before aggregating and drop the clients it
    /// flags (up to `expected_byzantine_ratio` of the cohort). Requires
    /// `statistical_tests.enabled`.
    pub dynamic_detection: bool,

    /// Reputation system settings
    pub reputation_system: ReputationSystemConfig,

    /// Statistical tests for outlier detection
    pub statistical_tests: StatisticalTestConfig,
}

/// Reputation system configuration
#[derive(Debug, Clone)]
pub struct ReputationSystemConfig {
    /// Whether reputations are maintained at all.
    pub enabled: bool,
    /// Reputation assigned to a client on first sight, in `[0, 1]`.
    pub initial_reputation: f64,
    /// Per-round mean reversion towards `initial_reputation`, in `[0, 1]`.
    pub reputation_decay: f64,
    /// Floor below which a reputation cannot fall.
    pub min_reputation: f64,
    /// Reputation subtracted when a client is flagged as an outlier.
    pub outlier_penalty: f64,
    /// Reputation added when a client is not flagged.
    pub contribution_bonus: f64,
}

/// Statistical test configuration for outlier detection
#[derive(Debug, Clone)]
pub struct StatisticalTestConfig {
    /// Whether detection runs.
    pub enabled: bool,
    /// Which test decides what counts as an outlier.
    pub test_type: StatisticalTestType,
    /// Significance level for the tests that define a p-value.
    pub significancelevel: f64,
    /// Number of past per-client statistics retained for the adaptive
    /// threshold. Must be non-zero.
    pub window_size: usize,
    /// Estimate the test's location and scale from the retained window in
    /// addition to the current round, so the decision boundary tracks recent
    /// cohort behaviour instead of being re-derived from one round alone.
    pub adaptive_threshold: bool,
}

/// Statistical tests available for outlier detection. See
/// [`super::outlier_tests`] for the definitions and references.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatisticalTestType {
    /// Two-sided z-test against the standard normal.
    ZScore,
    /// Iglewicz-Hoaglin median/MAD score with the 3.5 cut-off.
    ModifiedZScore,
    /// Tukey's 1.5 x IQR fences.
    IQRTest,
    /// Grubbs' test with a Bonferroni correction over the cohort.
    GrubbsTest,
    /// Chauvenet's criterion.
    ChauventCriterion,
}

/// Byzantine-robust aggregation engine
pub struct ByzantineRobustAggregator<
    T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static,
> {
    config: ByzantineRobustConfig,
    client_reputations: HashMap<String, f64>,
    outlier_history: VecDeque<OutlierDetectionResult>,
    statistical_analyzer: StatisticalAnalyzer<T>,
    robust_estimators: RobustEstimators<T>,
    rounds_aggregated: usize,
}

/// Statistical analyzer for outlier detection
pub struct StatisticalAnalyzer<
    T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static,
> {
    window_size: usize,
    significancelevel: f64,
    test_type: StatisticalTestType,
    adaptive_threshold: bool,
    test_statistics: VecDeque<TestStatistic<T>>,
}

/// Diagnostics recorded by the most recent [`ByzantineRobustAggregator::robust_aggregate`].
///
/// These are not caches used to skip work -- every aggregation recomputes from
/// scratch -- they are the audit trail explaining *which* clients the method
/// actually used, which is the only way to tell a working robust aggregator
/// from a plain mean after the fact.
pub struct RobustEstimators<
    T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum + 'static,
> {
    last_trim_count: usize,
    last_median: Option<Array1<T>>,
    krum_scores: HashMap<String, f64>,
    last_contributors: Vec<String>,
    last_excluded: Vec<String>,
}

/// Outlier detection result
#[derive(Debug, Clone)]
pub struct OutlierDetectionResult {
    /// Client the verdict applies to.
    pub clientid: String,
    /// Round in which the verdict was produced.
    pub round: usize,
    /// Whether the client was flagged.
    pub is_outlier: bool,
    /// The test statistic (a z-score, modified z-score, IQR multiple or
    /// Grubbs' G, depending on the configured test).
    pub outlier_score: f64,
    /// Raw input statistic: the client's mean Euclidean distance to the rest
    /// of the cohort.
    pub mean_distance: f64,
    /// Two-sided p-value where the test defines one.
    pub p_value: Option<f64>,
    /// Name of the test that produced the verdict.
    pub detection_method: String,
}

/// Test statistic for outlier detection
#[derive(Debug, Clone)]
pub struct TestStatistic<T: Float + Debug + Send + Sync + 'static> {
    /// The test statistic produced for this client.
    pub statistic_value: T,
    /// The raw input statistic (mean distance to the cohort) that the test
    /// consumed. Retained so the adaptive threshold can pool it with later
    /// rounds.
    pub sample_value: f64,
    /// Two-sided p-value, where the test defines one.
    pub p_value: Option<f64>,
    /// Which test produced it.
    pub test_type: StatisticalTestType,
    /// Client the statistic belongs to.
    pub clientid: String,
}

/// Per-client privacy allocation supplied to
/// [`ByzantineRobustAggregator::robust_aggregate`].
///
/// `utility_weight` is used as the client's weight by the two methods where a
/// weight is unambiguous -- [`ByzantineRobustMethod::ReputationWeighted`] and
/// [`ByzantineRobustMethod::FedAvgOutlierDetection`]. The rank- and
/// geometry-based methods (trimmed mean, median, Krum, Multi-Krum, Bulyan,
/// centered clipping) have no weighted formulation that preserves their
/// robustness proofs, so they ignore it; the allocation is still validated so
/// a malformed one cannot pass silently.
#[derive(Debug, Clone)]
pub struct AdaptivePrivacyAllocation {
    /// Epsilon allocated to the client for this round. Must be positive.
    pub epsilon: f64,
    /// Delta allocated to the client. Must be in `[0, 1)`.
    pub delta: f64,
    /// Non-negative aggregation weight.
    pub utility_weight: f64,
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + 'static
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > ByzantineRobustAggregator<T>
{
    /// Create an aggregator with the default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(ByzantineRobustConfig::default())
    }

    /// Create an aggregator with an explicit configuration, rejecting
    /// configurations whose parameters cannot produce a valid aggregate.
    pub fn with_config(config: ByzantineRobustConfig) -> Result<Self> {
        config.validate()?;
        let statistical_analyzer = StatisticalAnalyzer::with_config(&config.statistical_tests);
        Ok(Self {
            config,
            client_reputations: HashMap::new(),
            outlier_history: VecDeque::new(),
            statistical_analyzer,
            robust_estimators: RobustEstimators::new(),
            rounds_aggregated: 0,
        })
    }

    /// Run outlier detection over a cohort and record the verdicts.
    pub fn detect_byzantine_clients(
        &mut self,
        client_updates: &HashMap<String, Array1<T>>,
        round: usize,
    ) -> Result<Vec<OutlierDetectionResult>> {
        if !self.config.statistical_tests.enabled {
            return Err(OptimError::InvalidConfig(
                "statistical outlier detection is disabled in this configuration".to_string(),
            ));
        }
        let results = self
            .statistical_analyzer
            .detect_outliers(client_updates, round)?;
        for result in results.iter() {
            if self.outlier_history.len() >= OUTLIER_HISTORY_CAPACITY {
                self.outlier_history.pop_front();
            }
            self.outlier_history.push_back(result.clone());
        }
        Ok(results)
    }

    /// Current reputation of each requested client, defaulting to the
    /// configured initial reputation for clients never seen before.
    pub fn get_client_reputations(&self, clients: &[String]) -> HashMap<String, f64> {
        clients
            .iter()
            .map(|client_id| {
                let reputation = self
                    .client_reputations
                    .get(client_id)
                    .copied()
                    .unwrap_or(self.config.reputation_system.initial_reputation);
                (client_id.clone(), reputation)
            })
            .collect()
    }

    /// Aggregate a cohort of client updates using the configured method.
    ///
    /// When `dynamic_detection` is enabled the cohort is first filtered: the
    /// configured statistical test is run, and up to
    /// `floor(n * expected_byzantine_ratio)` of the highest-scoring flagged
    /// clients are removed before the estimator runs. At least one client
    /// always survives.
    ///
    /// Returns an error rather than a plain mean whenever the configured
    /// method cannot be applied to this cohort.
    pub fn robust_aggregate(
        &mut self,
        client_updates: &HashMap<String, Array1<T>>,
        allocations: &HashMap<String, AdaptivePrivacyAllocation>,
    ) -> Result<Array1<T>> {
        let full_cohort = robust_ops::ordered_cohort(client_updates)?;
        let round = self.rounds_aggregated;

        let excluded = if self.config.dynamic_detection {
            self.select_exclusions(client_updates, full_cohort.len(), round)?
        } else {
            HashSet::new()
        };

        let cohort: Vec<CohortMember<'_, T>> = full_cohort
            .iter()
            .filter(|(id, _)| !excluded.contains(*id))
            .copied()
            .collect();
        if cohort.is_empty() {
            return Err(OptimError::InvalidState(
                "dynamic detection excluded every client; refusing to aggregate an empty cohort"
                    .to_string(),
            ));
        }

        let weights = allocation_weights(&cohort, allocations)?;
        let mut detection_excluded: Vec<String> = excluded.into_iter().collect();
        detection_excluded.sort();
        let aggregate = self.apply_method(&cohort, weights.as_deref(), &detection_excluded)?;

        self.rounds_aggregated = self.rounds_aggregated.saturating_add(1);
        Ok(aggregate)
    }

    /// Fraction of recorded outlier evaluations that were *not* flagged.
    ///
    /// Errors when no detection has ever run: reporting a perfect 1.0 for an
    /// empty history would claim evidence of robustness that does not exist.
    pub fn compute_robustness_factor(&self) -> Result<f64> {
        let total = self.outlier_history.len();
        if total == 0 {
            return Err(OptimError::InvalidState(
                "no outlier evaluations have been recorded; run detect_byzantine_clients or \
                 enable dynamic_detection before asking for a robustness factor"
                    .to_string(),
            ));
        }
        let flagged = self
            .outlier_history
            .iter()
            .filter(|result| result.is_outlier)
            .count();
        Ok(1.0 - (flagged as f64 / total as f64))
    }

    /// Coordinate-wise median of a cohort, independent of the configured
    /// method.
    pub fn coordinate_wise_median(
        &self,
        client_updates: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        let cohort = robust_ops::ordered_cohort(client_updates)?;
        robust_ops::coordinate_wise_median(&cohort)
    }

    /// Get current configuration
    pub fn config(&self) -> &ByzantineRobustConfig {
        &self.config
    }

    /// Recorded outlier verdicts, oldest first.
    pub fn outlier_history(&self) -> &VecDeque<OutlierDetectionResult> {
        &self.outlier_history
    }

    /// Diagnostics from the most recent aggregation.
    pub fn robust_estimators(&self) -> &RobustEstimators<T> {
        &self.robust_estimators
    }

    /// Read-only access to the statistical analyzer.
    pub fn statistical_analyzer(&self) -> &StatisticalAnalyzer<T> {
        &self.statistical_analyzer
    }

    /// Number of completed aggregations.
    pub fn rounds_aggregated(&self) -> usize {
        self.rounds_aggregated
    }

    /// Update client reputation after an outlier verdict.
    ///
    /// A no-op when the reputation system is disabled, so that a disabled
    /// system cannot influence a later reputation-weighted aggregate.
    pub fn update_client_reputation(&mut self, client_id: String, is_outlier: bool) {
        if !self.config.reputation_system.enabled {
            return;
        }
        let system = &self.config.reputation_system;
        let current = self
            .client_reputations
            .get(&client_id)
            .copied()
            .unwrap_or(system.initial_reputation);

        let updated = if is_outlier {
            (current - system.outlier_penalty).max(system.min_reputation)
        } else {
            (current + system.contribution_bonus).min(1.0)
        };
        self.client_reputations.insert(client_id, updated);
    }

    /// Move every known reputation `decay` of the way back towards the
    /// configured initial reputation.
    fn decay_reputations(&mut self, decay: f64) {
        if decay <= 0.0 {
            return;
        }
        let initial = self.config.reputation_system.initial_reputation;
        let floor = self.config.reputation_system.min_reputation;
        for reputation in self.client_reputations.values_mut() {
            *reputation = (*reputation + (initial - *reputation) * decay).max(floor);
        }
    }

    /// Run detection and decide which clients to drop, honouring the
    /// `expected_byzantine_ratio` cap.
    fn select_exclusions(
        &mut self,
        client_updates: &HashMap<String, Array1<T>>,
        cohort_size: usize,
        round: usize,
    ) -> Result<HashSet<String>> {
        let detections = self.detect_byzantine_clients(client_updates, round)?;

        if self.config.reputation_system.enabled {
            let decay = self.config.reputation_system.reputation_decay;
            self.decay_reputations(decay);
            for detection in detections.iter() {
                self.update_client_reputation(detection.clientid.clone(), detection.is_outlier);
            }
        }

        let mut flagged: Vec<&OutlierDetectionResult> = detections
            .iter()
            .filter(|detection| detection.is_outlier)
            .collect();
        // Drop the most extreme first, ties broken by client id.
        flagged.sort_by(|a, b| {
            b.outlier_score
                .abs()
                .partial_cmp(&a.outlier_score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.clientid.cmp(&b.clientid))
        });

        let cap = ((cohort_size as f64) * self.config.expected_byzantine_ratio).floor() as usize;
        let cap = cap.min(cohort_size.saturating_sub(1));
        Ok(flagged
            .into_iter()
            .take(cap)
            .map(|detection| detection.clientid.clone())
            .collect())
    }

    /// Dispatch to the configured estimator. Every arm either produces a real
    /// aggregate or an error; there is no fall-through.
    ///
    /// `detection_excluded` lists the clients dynamic detection already
    /// removed, so that the recorded exclusion set is the union of what
    /// detection dropped and what the method itself filtered.
    fn apply_method(
        &mut self,
        cohort: &[CohortMember<'_, T>],
        weights: Option<&[f64]>,
        detection_excluded: &[String],
    ) -> Result<Array1<T>> {
        self.robust_estimators.reset_round();
        self.robust_estimators.last_excluded = detection_excluded.to_vec();
        match self.config.method {
            ByzantineRobustMethod::TrimmedMean { trim_ratio } => {
                let trim = robust_ops::trim_count_for_ratio(cohort.len(), trim_ratio)?;
                self.robust_estimators.last_trim_count = trim;
                self.robust_estimators.last_contributors = cohort_ids(cohort);
                robust_ops::coordinate_wise_trimmed_mean(cohort, trim)
            }
            ByzantineRobustMethod::CoordinateWiseMedian => {
                let median = robust_ops::coordinate_wise_median(cohort)?;
                self.robust_estimators.last_median = Some(median.clone());
                self.robust_estimators.last_contributors = cohort_ids(cohort);
                Ok(median)
            }
            ByzantineRobustMethod::Krum { f } => {
                let scores = robust_ops::krum_scores(cohort, f)?;
                self.robust_estimators.record_krum_scores(cohort, &scores);
                let winner = robust_ops::krum_select(cohort, f)?;
                self.robust_estimators.last_contributors = vec![cohort[winner].0.to_string()];
                Ok(cohort[winner].1.clone())
            }
            ByzantineRobustMethod::MultiKrum { f, m } => {
                let scores = robust_ops::krum_scores(cohort, f)?;
                self.robust_estimators.record_krum_scores(cohort, &scores);
                let selected = robust_ops::multi_krum_indices(cohort, f, m)?;
                let subset: Vec<CohortMember<'_, T>> =
                    selected.iter().map(|&index| cohort[index]).collect();
                self.robust_estimators.last_contributors = cohort_ids(&subset);
                robust_ops::mean(&subset)
            }
            ByzantineRobustMethod::Bulyan { f } => {
                self.robust_estimators.last_contributors = cohort_ids(cohort);
                robust_ops::bulyan(cohort, f)
            }
            ByzantineRobustMethod::CenteredClipping { tau } => {
                self.robust_estimators.last_contributors = cohort_ids(cohort);
                robust_ops::centered_clipping(cohort, tau, CENTERED_CLIPPING_ITERATIONS)
            }
            ByzantineRobustMethod::FedAvgOutlierDetection { threshold } => {
                let verdicts = self.statistical_analyzer.score_cohort(cohort)?;
                let kept: Vec<usize> = verdicts
                    .iter()
                    .enumerate()
                    .filter(|(_, verdict)| verdict.statistic.abs() <= threshold)
                    .map(|(index, _)| index)
                    .collect();
                if kept.is_empty() {
                    return Err(OptimError::InvalidState(format!(
                        "every client's outlier statistic exceeded the threshold {threshold}; \
                         there is nothing left to average"
                    )));
                }
                let subset: Vec<CohortMember<'_, T>> =
                    kept.iter().map(|&index| cohort[index]).collect();
                self.robust_estimators.last_contributors = cohort_ids(&subset);
                self.robust_estimators.last_excluded.extend(
                    (0..cohort.len())
                        .filter(|index| !kept.contains(index))
                        .map(|index| cohort[index].0.to_string()),
                );
                self.robust_estimators.last_excluded.sort();
                self.robust_estimators.last_excluded.dedup();
                match weights {
                    Some(all) => {
                        let subset_weights: Vec<f64> =
                            kept.iter().map(|&index| all[index]).collect();
                        robust_ops::weighted_mean(&subset, &subset_weights)
                    }
                    None => robust_ops::mean(&subset),
                }
            }
            ByzantineRobustMethod::ReputationWeighted { reputation_decay } => {
                if !self.config.reputation_system.enabled {
                    return Err(OptimError::InvalidConfig(
                        "ReputationWeighted aggregation requires reputation_system.enabled; \
                         with the system disabled every client would carry the same implicit \
                         weight, which is plain FedAvg under a robust-sounding name"
                            .to_string(),
                    ));
                }
                self.decay_reputations(reputation_decay);
                let initial = self.config.reputation_system.initial_reputation;
                let mut combined = Vec::with_capacity(cohort.len());
                for (index, (id, _)) in cohort.iter().enumerate() {
                    let reputation = self
                        .client_reputations
                        .get(*id)
                        .copied()
                        .unwrap_or(initial)
                        .max(0.0);
                    let weight = match weights {
                        Some(all) => reputation * all[index],
                        None => reputation,
                    };
                    combined.push(weight);
                }
                self.robust_estimators.last_contributors = cohort_ids(cohort);
                robust_ops::weighted_mean(cohort, &combined)
            }
        }
    }
}

fn cohort_ids<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
) -> Vec<String> {
    cohort.iter().map(|(id, _)| (*id).to_string()).collect()
}

/// Validate `allocations` and project them onto the cohort order.
///
/// Returns `None` when no allocations were supplied (the common case, where
/// every client is weighted equally). When allocations *are* supplied they
/// must cover every cohort member: a partial allocation would silently give
/// the uncovered clients a fabricated default weight.
fn allocation_weights<T: Float + Debug + Send + Sync + 'static>(
    cohort: &[CohortMember<'_, T>],
    allocations: &HashMap<String, AdaptivePrivacyAllocation>,
) -> Result<Option<Vec<f64>>> {
    if allocations.is_empty() {
        return Ok(None);
    }
    let mut weights = Vec::with_capacity(cohort.len());
    for (id, _) in cohort.iter() {
        let allocation = allocations.get(*id).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "privacy allocations were supplied but client {id} is missing from them; a \
                 partial allocation cannot be completed without inventing a weight"
            ))
        })?;
        if !allocation.epsilon.is_finite() || allocation.epsilon <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "client {id} has a non-positive epsilon allocation ({})",
                allocation.epsilon
            )));
        }
        if !allocation.delta.is_finite() || !(0.0..1.0).contains(&allocation.delta) {
            return Err(OptimError::InvalidConfig(format!(
                "client {id} has a delta allocation of {} outside [0, 1)",
                allocation.delta
            )));
        }
        if !allocation.utility_weight.is_finite() || allocation.utility_weight < 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "client {id} has a negative or non-finite utility weight ({})",
                allocation.utility_weight
            )));
        }
        weights.push(allocation.utility_weight);
    }
    Ok(Some(weights))
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum>
    StatisticalAnalyzer<T>
{
    /// Create an analyzer with an explicit window and significance level,
    /// using a two-sided z-test and a per-round (non-adaptive) threshold.
    pub fn new(window_size: usize, significancelevel: f64) -> Self {
        Self {
            window_size: window_size.max(1),
            significancelevel,
            test_type: StatisticalTestType::ZScore,
            adaptive_threshold: false,
            test_statistics: VecDeque::new(),
        }
    }

    /// Create an analyzer from a [`StatisticalTestConfig`].
    pub fn with_config(config: &StatisticalTestConfig) -> Self {
        Self {
            window_size: config.window_size.max(1),
            significancelevel: config.significancelevel,
            test_type: config.test_type,
            adaptive_threshold: config.adaptive_threshold,
            test_statistics: VecDeque::new(),
        }
    }

    /// Detect outliers using the configured statistical test.
    ///
    /// The per-client input statistic is the mean Euclidean distance from
    /// that client's update to every other client's. Cohorts of fewer than
    /// three clients yield no verdicts: with two updates each is exactly as
    /// far from the other as vice versa, so no distance-based test can
    /// distinguish them.
    pub fn detect_outliers(
        &mut self,
        client_updates: &HashMap<String, Array1<T>>,
        round: usize,
    ) -> Result<Vec<OutlierDetectionResult>> {
        let cohort = robust_ops::ordered_cohort(client_updates)?;
        if cohort.len() < 3 {
            return Ok(Vec::new());
        }

        let distances = self.mean_distances(&cohort)?;
        let verdicts = self.evaluate(&distances)?;

        let mut results = Vec::with_capacity(cohort.len());
        for (index, (id, _)) in cohort.iter().enumerate() {
            let verdict = verdicts[index];
            let statistic = T::from(verdict.statistic).ok_or_else(|| {
                OptimError::ComputationError(format!(
                    "test statistic {} is not representable in the target float type",
                    verdict.statistic
                ))
            })?;
            self.push_statistic(TestStatistic {
                statistic_value: statistic,
                sample_value: distances[index],
                p_value: verdict.p_value,
                test_type: self.test_type,
                clientid: (*id).to_string(),
            });
            results.push(OutlierDetectionResult {
                clientid: (*id).to_string(),
                round,
                is_outlier: verdict.is_outlier,
                outlier_score: verdict.statistic,
                mean_distance: distances[index],
                p_value: verdict.p_value,
                detection_method: format!("{:?}", self.test_type),
            });
        }
        Ok(results)
    }

    /// Score an already-ordered cohort without recording anything.
    pub fn score_cohort(&self, cohort: &[CohortMember<'_, T>]) -> Result<Vec<OutlierVerdict>> {
        if cohort.len() < 3 {
            return Ok(cohort
                .iter()
                .map(|_| OutlierVerdict {
                    statistic: 0.0,
                    p_value: None,
                    is_outlier: false,
                })
                .collect());
        }
        let distances = self.mean_distances(cohort)?;
        self.evaluate(&distances)
    }

    /// Statistics retained in the rolling window, oldest first.
    pub fn test_statistics(&self) -> &VecDeque<TestStatistic<T>> {
        &self.test_statistics
    }

    /// Configured window width.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Configured significance level.
    pub fn significance_level(&self) -> f64 {
        self.significancelevel
    }

    /// Mean Euclidean distance from each cohort member to the others.
    fn mean_distances(&self, cohort: &[CohortMember<'_, T>]) -> Result<Vec<f64>> {
        let squared = robust_ops::pairwise_squared_distances(cohort)?;
        let n = cohort.len();
        if n < 2 {
            return Ok(vec![0.0; n]);
        }
        Ok((0..n)
            .map(|i| {
                let total: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| squared[i][j].max(0.0).sqrt())
                    .sum();
                total / (n - 1) as f64
            })
            .collect())
    }

    fn evaluate(&self, distances: &[f64]) -> Result<Vec<OutlierVerdict>> {
        if self.adaptive_threshold && !self.test_statistics.is_empty() {
            let mut pool: Vec<f64> = distances.to_vec();
            pool.extend(
                self.test_statistics
                    .iter()
                    .map(|statistic| statistic.sample_value),
            );
            outlier_tests::evaluate(self.test_type, distances, &pool, self.significancelevel)
        } else {
            outlier_tests::evaluate(self.test_type, distances, distances, self.significancelevel)
        }
    }

    fn push_statistic(&mut self, statistic: TestStatistic<T>) {
        while self.test_statistics.len() >= self.window_size {
            self.test_statistics.pop_front();
        }
        self.test_statistics.push_back(statistic);
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum>
    RobustEstimators<T>
{
    /// Create an empty diagnostics record.
    pub fn new() -> Self {
        Self {
            last_trim_count: 0,
            last_median: None,
            krum_scores: HashMap::new(),
            last_contributors: Vec::new(),
            last_excluded: Vec::new(),
        }
    }

    /// Coordinate-wise trimmed mean at the requested ratio.
    ///
    /// The trim count is derived from the ratio via
    /// [`robust_ops::trim_count_for_ratio`], so a ratio too small to remove
    /// anything from a cohort of this size removes nothing -- and
    /// [`Self::last_trim_count`] says so, rather than the caller having to
    /// guess whether trimming happened.
    pub fn trimmed_mean(
        &mut self,
        client_updates: &HashMap<String, Array1<T>>,
        trim_ratio: f64,
    ) -> Result<Array1<T>> {
        let cohort = robust_ops::ordered_cohort(client_updates)?;
        let trim = robust_ops::trim_count_for_ratio(cohort.len(), trim_ratio)?;
        self.last_trim_count = trim;
        self.last_contributors = cohort_ids(&cohort);
        robust_ops::coordinate_wise_trimmed_mean(&cohort, trim)
    }

    /// Coordinate-wise median, caching the result for inspection.
    pub fn median(&mut self, client_updates: &HashMap<String, Array1<T>>) -> Result<Array1<T>> {
        let cohort = robust_ops::ordered_cohort(client_updates)?;
        let median = robust_ops::coordinate_wise_median(&cohort)?;
        self.last_median = Some(median.clone());
        self.last_contributors = cohort_ids(&cohort);
        Ok(median)
    }

    /// Values removed from each tail by the most recent trimmed mean.
    pub fn last_trim_count(&self) -> usize {
        self.last_trim_count
    }

    /// Most recently computed coordinate-wise median.
    pub fn last_median(&self) -> Option<&Array1<T>> {
        self.last_median.as_ref()
    }

    /// Krum scores from the most recent Krum-family aggregation.
    pub fn krum_scores(&self) -> &HashMap<String, f64> {
        &self.krum_scores
    }

    /// Clients that actually contributed to the most recent aggregate.
    pub fn last_contributors(&self) -> &[String] {
        &self.last_contributors
    }

    /// Clients excluded from the most recent aggregate.
    pub fn last_excluded(&self) -> &[String] {
        &self.last_excluded
    }

    fn reset_round(&mut self) {
        self.last_trim_count = 0;
        self.last_median = None;
        self.krum_scores.clear();
        self.last_contributors.clear();
        self.last_excluded.clear();
    }

    fn record_krum_scores(&mut self, cohort: &[CohortMember<'_, T>], scores: &[f64]) {
        self.krum_scores.clear();
        for ((id, _), &score) in cohort.iter().zip(scores.iter()) {
            self.krum_scores.insert((*id).to_string(), score);
        }
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum> Default
    for RobustEstimators<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl ByzantineRobustConfig {
    /// Reject configurations whose parameters cannot yield a valid aggregate.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..0.5).contains(&self.expected_byzantine_ratio)
            || !self.expected_byzantine_ratio.is_finite()
        {
            return Err(OptimError::InvalidConfig(format!(
                "expected_byzantine_ratio must be in [0, 0.5), got {}",
                self.expected_byzantine_ratio
            )));
        }
        if self.dynamic_detection && !self.statistical_tests.enabled {
            return Err(OptimError::InvalidConfig(
                "dynamic_detection requires statistical_tests.enabled; without a test there is \
                 nothing to detect with"
                    .to_string(),
            ));
        }
        if self.statistical_tests.window_size == 0 {
            return Err(OptimError::InvalidConfig(
                "statistical_tests.window_size must be greater than zero".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.statistical_tests.significancelevel)
            || self.statistical_tests.significancelevel <= 0.0
        {
            return Err(OptimError::InvalidConfig(format!(
                "statistical_tests.significancelevel must be in (0, 1), got {}",
                self.statistical_tests.significancelevel
            )));
        }
        self.reputation_system.validate()?;

        match self.method {
            ByzantineRobustMethod::TrimmedMean { trim_ratio } => {
                if !(0.0..1.0).contains(&trim_ratio) || !trim_ratio.is_finite() {
                    return Err(OptimError::InvalidConfig(format!(
                        "TrimmedMean trim_ratio must be in [0, 1), got {trim_ratio}"
                    )));
                }
            }
            ByzantineRobustMethod::MultiKrum { m, .. } => {
                if m == 0 {
                    return Err(OptimError::InvalidConfig(
                        "MultiKrum must average at least one update (m > 0)".to_string(),
                    ));
                }
            }
            ByzantineRobustMethod::CenteredClipping { tau } => {
                if !tau.is_finite() || tau <= 0.0 {
                    return Err(OptimError::InvalidConfig(format!(
                        "CenteredClipping tau must be positive and finite, got {tau}"
                    )));
                }
            }
            ByzantineRobustMethod::FedAvgOutlierDetection { threshold } => {
                if !threshold.is_finite() || threshold <= 0.0 {
                    return Err(OptimError::InvalidConfig(format!(
                        "FedAvgOutlierDetection threshold must be positive and finite, got \
                         {threshold}"
                    )));
                }
            }
            ByzantineRobustMethod::ReputationWeighted { reputation_decay } => {
                if !(0.0..=1.0).contains(&reputation_decay) || !reputation_decay.is_finite() {
                    return Err(OptimError::InvalidConfig(format!(
                        "ReputationWeighted reputation_decay must be in [0, 1], got \
                         {reputation_decay}"
                    )));
                }
            }
            ByzantineRobustMethod::CoordinateWiseMedian
            | ByzantineRobustMethod::Krum { .. }
            | ByzantineRobustMethod::Bulyan { .. } => {}
        }
        Ok(())
    }
}

impl ReputationSystemConfig {
    /// Reject reputation parameters that cannot produce usable weights.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("initial_reputation", self.initial_reputation),
            ("reputation_decay", self.reputation_decay),
            ("min_reputation", self.min_reputation),
            ("outlier_penalty", self.outlier_penalty),
            ("contribution_bonus", self.contribution_bonus),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OptimError::InvalidConfig(format!(
                    "reputation_system.{name} must be finite and non-negative, got {value}"
                )));
            }
        }
        if self.reputation_decay > 1.0 {
            return Err(OptimError::InvalidConfig(format!(
                "reputation_system.reputation_decay must be in [0, 1], got {}",
                self.reputation_decay
            )));
        }
        if self.initial_reputation < self.min_reputation {
            return Err(OptimError::InvalidConfig(format!(
                "reputation_system.initial_reputation ({}) is below min_reputation ({})",
                self.initial_reputation, self.min_reputation
            )));
        }
        Ok(())
    }
}

impl Default for ByzantineRobustConfig {
    fn default() -> Self {
        Self {
            method: ByzantineRobustMethod::TrimmedMean { trim_ratio: 0.2 },
            expected_byzantine_ratio: 0.1,
            dynamic_detection: false,
            reputation_system: ReputationSystemConfig::default(),
            statistical_tests: StatisticalTestConfig::default(),
        }
    }
}

impl Default for ReputationSystemConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_reputation: 1.0,
            reputation_decay: 0.01,
            min_reputation: 0.1,
            outlier_penalty: 0.5,
            contribution_bonus: 0.1,
        }
    }
}

impl Default for StatisticalTestConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            test_type: StatisticalTestType::ZScore,
            significancelevel: 0.05,
            window_size: 100,
            adaptive_threshold: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn cohort(pairs: &[(&str, Vec<f64>)]) -> HashMap<String, Array1<f64>> {
        pairs
            .iter()
            .map(|(id, values)| ((*id).to_string(), Array1::from(values.clone())))
            .collect()
    }

    /// Ten honest clients clustered near `1.0` plus one wild attacker.
    fn contaminated_cohort() -> HashMap<String, Array1<f64>> {
        let mut updates: HashMap<String, Array1<f64>> = (0..10)
            .map(|i| {
                (
                    format!("honest{i:02}"),
                    Array1::from(vec![1.0 + i as f64 * 0.01, 2.0 + i as f64 * 0.01]),
                )
            })
            .collect();
        updates.insert("attacker".to_string(), Array1::from(vec![1.0e4, -1.0e4]));
        updates
    }

    fn no_allocations() -> HashMap<String, AdaptivePrivacyAllocation> {
        HashMap::new()
    }

    #[test]
    fn test_byzantine_robust_aggregator_creation() {
        assert!(ByzantineRobustAggregator::<f64>::new().is_ok());
    }

    // ---------------------------------------------------------------------
    // F26/F27: every configured method is dispatched; none falls through to
    // a plain mean.
    // ---------------------------------------------------------------------

    #[test]
    fn every_method_produces_a_distinct_real_aggregate_not_the_mean() {
        let updates = contaminated_cohort();
        let plain_mean = {
            let cohort = robust_ops::ordered_cohort(&updates).expect("cohort");
            robust_ops::mean(&cohort).expect("mean")
        };
        // Sanity: the attacker drags the plain mean far away from 1.0.
        assert!(plain_mean[0] > 100.0);

        for method in [
            ByzantineRobustMethod::TrimmedMean { trim_ratio: 0.4 },
            ByzantineRobustMethod::CoordinateWiseMedian,
            ByzantineRobustMethod::Krum { f: 2 },
            ByzantineRobustMethod::MultiKrum { f: 2, m: 5 },
            ByzantineRobustMethod::Bulyan { f: 2 },
            ByzantineRobustMethod::CenteredClipping { tau: 1.0 },
            ByzantineRobustMethod::FedAvgOutlierDetection { threshold: 2.0 },
        ] {
            let mut aggregator =
                ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
                    method,
                    ..ByzantineRobustConfig::default()
                })
                .expect("config");
            let result = aggregator
                .robust_aggregate(&updates, &no_allocations())
                .unwrap_or_else(|error| panic!("{method:?} failed: {error}"));
            assert!(
                result[0].abs() < 10.0,
                "{method:?} returned {} -- it did not resist the attacker",
                result[0]
            );
            assert!(
                (result[0] - plain_mean[0]).abs() > 1.0,
                "{method:?} returned the plain mean, i.e. it fell through"
            );
        }
    }

    #[test]
    fn krum_returns_one_actual_client_update() {
        let updates = contaminated_cohort();
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::Krum { f: 2 },
            ..ByzantineRobustConfig::default()
        })
        .expect("config");
        let result = aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect("krum");

        let contributors = aggregator.robust_estimators().last_contributors().to_vec();
        assert_eq!(contributors.len(), 1, "Krum selects exactly one client");
        assert_ne!(contributors[0], "attacker");
        let chosen = &updates[&contributors[0]];
        assert_eq!(result.to_vec(), chosen.to_vec());

        // Real Krum scores were recorded for every client, and the attacker's
        // is by far the worst.
        let scores = aggregator.robust_estimators().krum_scores();
        assert_eq!(scores.len(), updates.len());
        let attacker = scores["attacker"];
        assert!(scores
            .iter()
            .filter(|(id, _)| id.as_str() != "attacker")
            .all(|(_, &score)| score < attacker));
    }

    #[test]
    fn multi_krum_averages_exactly_m_clients() {
        let updates = contaminated_cohort();
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::MultiKrum { f: 2, m: 4 },
            ..ByzantineRobustConfig::default()
        })
        .expect("config");
        aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect("multi-krum");
        let contributors = aggregator.robust_estimators().last_contributors();
        assert_eq!(contributors.len(), 4);
        assert!(!contributors.iter().any(|id| id == "attacker"));
    }

    #[test]
    fn krum_family_errors_when_the_cohort_is_too_small_instead_of_averaging() {
        let updates = cohort(&[("a", vec![1.0]), ("b", vec![1.1]), ("c", vec![0.9])]);
        for method in [
            ByzantineRobustMethod::Krum { f: 3 },
            ByzantineRobustMethod::MultiKrum { f: 3, m: 2 },
            ByzantineRobustMethod::Bulyan { f: 3 },
        ] {
            let mut aggregator =
                ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
                    method,
                    ..ByzantineRobustConfig::default()
                })
                .expect("config");
            let err = aggregator
                .robust_aggregate(&updates, &no_allocations())
                .expect_err("must not silently average");
            assert!(format!("{err}").contains("needs at least"));
        }
    }

    // ---------------------------------------------------------------------
    // F28: the trim ratio comes from the configuration.
    // ---------------------------------------------------------------------

    #[test]
    fn trim_ratio_is_configuration_driven() {
        // 10 clients: ratio 0.4 trims 2 from each tail, ratio 0.2 trims 1,
        // ratio 0.05 trims none.
        let updates: HashMap<String, Array1<f64>> = (0..10)
            .map(|i| (format!("c{i:02}"), Array1::from(vec![i as f64])))
            .collect();

        for (ratio, expected_trim) in [(0.4, 2_usize), (0.2, 1), (0.05, 0)] {
            let mut aggregator =
                ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
                    method: ByzantineRobustMethod::TrimmedMean { trim_ratio: ratio },
                    ..ByzantineRobustConfig::default()
                })
                .expect("config");
            aggregator
                .robust_aggregate(&updates, &no_allocations())
                .expect("trimmed mean");
            assert_eq!(
                aggregator.robust_estimators().last_trim_count(),
                expected_trim,
                "ratio {ratio} should trim {expected_trim} per tail"
            );
        }
    }

    #[test]
    fn trimmed_mean_actually_removes_the_tails() {
        // Sorted values 0.9, 1.0, 1.1, 10.0. A 50% ratio trims one per tail,
        // leaving the mean of {1.0, 1.1} = 1.05. The previous implementation
        // computed floor(4 * 0.25 / 2) = 0 and returned the plain mean 3.25
        // while its test claimed the outlier had been excluded.
        let mut estimators = RobustEstimators::<f64>::new();
        let updates = cohort(&[
            ("client1", vec![1.0]),
            ("client2", vec![1.1]),
            ("client3", vec![10.0]),
            ("client4", vec![0.9]),
        ]);
        let trimmed = estimators
            .trimmed_mean(&updates, 0.5)
            .expect("trimmed mean");
        assert_eq!(estimators.last_trim_count(), 1);
        assert!((trimmed[0] - 1.05).abs() < 1e-12);

        // With the old 0.25 ratio nothing is trimmed -- and the API now says
        // so instead of implying robustness.
        let untrimmed = estimators
            .trimmed_mean(&updates, 0.25)
            .expect("trimmed mean");
        assert_eq!(estimators.last_trim_count(), 0);
        assert!((untrimmed[0] - 3.25).abs() < 1e-12);
    }

    #[test]
    fn test_coordinate_wise_median() {
        let aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");
        let updates = cohort(&[
            ("client1", vec![1.0, 4.0, 7.0]),
            ("client2", vec![2.0, 5.0, 8.0]),
            ("client3", vec![3.0, 6.0, 9.0]),
        ]);
        let median = aggregator.coordinate_wise_median(&updates).expect("median");
        assert_eq!(median.to_vec(), vec![2.0, 5.0, 8.0]);
    }

    // ---------------------------------------------------------------------
    // F29/F30: no `.expect` on malformed input; errors propagate.
    // ---------------------------------------------------------------------

    #[test]
    fn malformed_cohorts_produce_errors_rather_than_panics() {
        let mut aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");
        let empty: HashMap<String, Array1<f64>> = HashMap::new();
        assert!(aggregator
            .robust_aggregate(&empty, &no_allocations())
            .is_err());

        let ragged = cohort(&[("a", vec![1.0, 2.0]), ("b", vec![1.0])]);
        assert!(aggregator
            .robust_aggregate(&ragged, &no_allocations())
            .is_err());

        let nan = cohort(&[("a", vec![f64::NAN]), ("b", vec![1.0])]);
        let err = aggregator
            .robust_aggregate(&nan, &no_allocations())
            .expect_err("NaN must be rejected");
        assert!(format!("{err}").contains("non-finite"));
    }

    #[test]
    fn invalid_configurations_are_rejected_at_construction() {
        let bad_ratio = ByzantineRobustConfig {
            method: ByzantineRobustMethod::TrimmedMean { trim_ratio: 1.5 },
            ..ByzantineRobustConfig::default()
        };
        assert!(ByzantineRobustAggregator::<f64>::with_config(bad_ratio).is_err());

        let contradictory = ByzantineRobustConfig {
            dynamic_detection: true,
            statistical_tests: StatisticalTestConfig {
                enabled: false,
                ..StatisticalTestConfig::default()
            },
            ..ByzantineRobustConfig::default()
        };
        assert!(ByzantineRobustAggregator::<f64>::with_config(contradictory).is_err());

        let bad_tau = ByzantineRobustConfig {
            method: ByzantineRobustMethod::CenteredClipping { tau: -1.0 },
            ..ByzantineRobustConfig::default()
        };
        assert!(ByzantineRobustAggregator::<f64>::with_config(bad_tau).is_err());

        let bad_byzantine_ratio = ByzantineRobustConfig {
            expected_byzantine_ratio: 0.9,
            ..ByzantineRobustConfig::default()
        };
        assert!(ByzantineRobustAggregator::<f64>::with_config(bad_byzantine_ratio).is_err());
    }

    // ---------------------------------------------------------------------
    // Detection, history and the robustness factor.
    // ---------------------------------------------------------------------

    #[test]
    fn test_outlier_detection() {
        let mut analyzer = StatisticalAnalyzer::<f64>::new(100, 0.05);
        let updates = contaminated_cohort();
        let detections = analyzer.detect_outliers(&updates, 1).expect("detection");

        assert_eq!(detections.len(), updates.len());
        let attacker = detections
            .iter()
            .find(|result| result.clientid == "attacker")
            .expect("attacker verdict");
        assert!(attacker.is_outlier);
        assert!(attacker.p_value.is_some_and(|p| p < 0.05));
        assert_eq!(attacker.detection_method, "ZScore");
        assert!(detections
            .iter()
            .filter(|result| result.clientid != "attacker")
            .all(|result| !result.is_outlier));
    }

    #[test]
    fn detection_populates_the_rolling_window_and_bounds_it() {
        let mut analyzer = StatisticalAnalyzer::<f64>::new(5, 0.05);
        let updates = contaminated_cohort();
        assert!(analyzer.test_statistics().is_empty());

        analyzer.detect_outliers(&updates, 1).expect("round 1");
        // 11 clients into a window of 5 leaves the last 5.
        assert_eq!(analyzer.test_statistics().len(), 5);
        analyzer.detect_outliers(&updates, 2).expect("round 2");
        assert_eq!(analyzer.test_statistics().len(), 5);
        assert!(analyzer
            .test_statistics()
            .iter()
            .all(|statistic| statistic.test_type == StatisticalTestType::ZScore));
    }

    #[test]
    fn detection_needs_at_least_three_clients() {
        let mut analyzer = StatisticalAnalyzer::<f64>::new(100, 0.05);
        let pair = cohort(&[("a", vec![1.0]), ("b", vec![1000.0])]);
        assert!(analyzer
            .detect_outliers(&pair, 1)
            .expect("no verdicts")
            .is_empty());
    }

    #[test]
    fn robustness_factor_reflects_recorded_verdicts() {
        let mut aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");
        // No history: the old implementation reported a perfect 1.0.
        assert!(aggregator.compute_robustness_factor().is_err());

        let updates = contaminated_cohort();
        let detections = aggregator
            .detect_byzantine_clients(&updates, 1)
            .expect("detection");
        assert_eq!(aggregator.outlier_history().len(), detections.len());

        let flagged = detections.iter().filter(|r| r.is_outlier).count();
        let expected = 1.0 - flagged as f64 / detections.len() as f64;
        let factor = aggregator
            .compute_robustness_factor()
            .expect("factor after detection");
        assert!((factor - expected).abs() < 1e-12);
        assert!(factor < 1.0, "one attacker must lower the factor");
    }

    #[test]
    fn detection_errors_when_statistical_tests_are_disabled() {
        let config = ByzantineRobustConfig {
            statistical_tests: StatisticalTestConfig {
                enabled: false,
                ..StatisticalTestConfig::default()
            },
            ..ByzantineRobustConfig::default()
        };
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(config).expect("config");
        let err = aggregator
            .detect_byzantine_clients(&contaminated_cohort(), 1)
            .expect_err("detection is off");
        assert!(format!("{err}").contains("disabled"));
    }

    #[test]
    fn dynamic_detection_excludes_flagged_clients_up_to_the_ratio_cap() {
        let updates = contaminated_cohort();
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::TrimmedMean { trim_ratio: 0.0 },
            dynamic_detection: true,
            expected_byzantine_ratio: 0.2,
            ..ByzantineRobustConfig::default()
        })
        .expect("config");

        let result = aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect("aggregate");
        assert_eq!(
            aggregator.robust_estimators().last_excluded(),
            &["attacker".to_string()]
        );
        // With the attacker gone, a zero-trim mean is already close to 1.0.
        assert!((result[0] - 1.045).abs() < 0.01, "got {}", result[0]);
    }

    #[test]
    fn dynamic_detection_respects_the_exclusion_cap() {
        let updates = contaminated_cohort();
        // A zero ratio permits zero exclusions, so the attacker survives and
        // the untrimmed mean is dragged away.
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::TrimmedMean { trim_ratio: 0.0 },
            dynamic_detection: true,
            expected_byzantine_ratio: 0.0,
            ..ByzantineRobustConfig::default()
        })
        .expect("config");
        let result = aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect("aggregate");
        assert!(aggregator.robust_estimators().last_excluded().is_empty());
        assert!(result[0] > 100.0);
    }

    // ---------------------------------------------------------------------
    // Reputation.
    // ---------------------------------------------------------------------

    #[test]
    fn test_reputation_system() {
        let mut aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");
        let reputations = aggregator.get_client_reputations(&["client1".to_string()]);
        assert_eq!(reputations.get("client1"), Some(&1.0));

        aggregator.update_client_reputation("client1".to_string(), true);
        let updated = aggregator.get_client_reputations(&["client1".to_string()]);
        assert_eq!(updated.get("client1"), Some(&0.5));

        aggregator.update_client_reputation("client2".to_string(), false);
        let good = aggregator.get_client_reputations(&["client2".to_string()]);
        assert_eq!(good.get("client2"), Some(&1.0));
    }

    #[test]
    fn reputation_updates_are_a_no_op_when_the_system_is_disabled() {
        let config = ByzantineRobustConfig {
            reputation_system: ReputationSystemConfig {
                enabled: false,
                ..ReputationSystemConfig::default()
            },
            ..ByzantineRobustConfig::default()
        };
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(config).expect("config");
        aggregator.update_client_reputation("client1".to_string(), true);
        assert_eq!(
            aggregator
                .get_client_reputations(&["client1".to_string()])
                .get("client1"),
            Some(&1.0)
        );
    }

    #[test]
    fn reputation_weighted_aggregation_down_weights_penalised_clients() {
        let updates = cohort(&[("good", vec![0.0]), ("bad", vec![10.0])]);
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::ReputationWeighted {
                reputation_decay: 0.0,
            },
            ..ByzantineRobustConfig::default()
        })
        .expect("config");

        // Equal reputations => the plain mean.
        let balanced = aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect("balanced");
        assert!((balanced[0] - 5.0).abs() < 1e-12);

        // Penalise "bad" twice: 1.0 -> 0.5 -> 0.1 (the floor).
        aggregator.update_client_reputation("bad".to_string(), true);
        aggregator.update_client_reputation("bad".to_string(), true);
        let skewed = aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect("skewed");
        // weights 1.0 and 0.1 => 10 * 0.1 / 1.1
        assert!(
            (skewed[0] - (10.0 * 0.1 / 1.1)).abs() < 1e-12,
            "got {}",
            skewed[0]
        );
    }

    #[test]
    fn reputation_weighted_requires_the_reputation_system() {
        let config = ByzantineRobustConfig {
            method: ByzantineRobustMethod::ReputationWeighted {
                reputation_decay: 0.1,
            },
            reputation_system: ReputationSystemConfig {
                enabled: false,
                ..ReputationSystemConfig::default()
            },
            ..ByzantineRobustConfig::default()
        };
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(config).expect("config");
        let updates = cohort(&[("a", vec![1.0]), ("b", vec![2.0])]);
        let err = aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect_err("must not silently become FedAvg");
        assert!(format!("{err}").contains("reputation_system.enabled"));
    }

    #[test]
    fn reputation_decay_reverts_towards_the_initial_value() {
        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::ReputationWeighted {
                reputation_decay: 0.5,
            },
            ..ByzantineRobustConfig::default()
        })
        .expect("config");
        aggregator.update_client_reputation("bad".to_string(), true);
        assert_eq!(
            aggregator
                .get_client_reputations(&["bad".to_string()])
                .get("bad"),
            Some(&0.5)
        );

        let updates = cohort(&[("bad", vec![1.0]), ("good", vec![1.0])]);
        aggregator
            .robust_aggregate(&updates, &no_allocations())
            .expect("aggregate");
        // 0.5 + (1.0 - 0.5) * 0.5 = 0.75
        let reputation = aggregator.get_client_reputations(&["bad".to_string()])["bad"];
        assert!((reputation - 0.75).abs() < 1e-12, "got {reputation}");
    }

    // ---------------------------------------------------------------------
    // Privacy allocations.
    // ---------------------------------------------------------------------

    #[test]
    fn allocations_weight_the_weighted_methods() {
        let updates = cohort(&[("a", vec![0.0]), ("b", vec![10.0])]);
        let mut allocations = HashMap::new();
        allocations.insert(
            "a".to_string(),
            AdaptivePrivacyAllocation {
                epsilon: 1.0,
                delta: 1e-5,
                utility_weight: 3.0,
            },
        );
        allocations.insert(
            "b".to_string(),
            AdaptivePrivacyAllocation {
                epsilon: 1.0,
                delta: 1e-5,
                utility_weight: 1.0,
            },
        );

        let mut aggregator = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::ReputationWeighted {
                reputation_decay: 0.0,
            },
            ..ByzantineRobustConfig::default()
        })
        .expect("config");
        let result = aggregator
            .robust_aggregate(&updates, &allocations)
            .expect("weighted");
        assert!((result[0] - 2.5).abs() < 1e-12, "got {}", result[0]);
    }

    #[test]
    fn partial_or_malformed_allocations_are_rejected() {
        let updates = cohort(&[("a", vec![0.0]), ("b", vec![10.0])]);
        let mut aggregator = ByzantineRobustAggregator::<f64>::new().expect("aggregator");

        let mut partial = HashMap::new();
        partial.insert(
            "a".to_string(),
            AdaptivePrivacyAllocation {
                epsilon: 1.0,
                delta: 1e-5,
                utility_weight: 1.0,
            },
        );
        let err = aggregator
            .robust_aggregate(&updates, &partial)
            .expect_err("missing client b");
        assert!(format!("{err}").contains("missing from them"));

        let mut negative = partial.clone();
        negative.insert(
            "b".to_string(),
            AdaptivePrivacyAllocation {
                epsilon: 1.0,
                delta: 1e-5,
                utility_weight: -1.0,
            },
        );
        assert!(aggregator.robust_aggregate(&updates, &negative).is_err());

        let mut bad_epsilon = partial;
        bad_epsilon.insert(
            "b".to_string(),
            AdaptivePrivacyAllocation {
                epsilon: 0.0,
                delta: 1e-5,
                utility_weight: 1.0,
            },
        );
        assert!(aggregator.robust_aggregate(&updates, &bad_epsilon).is_err());
    }

    // ---------------------------------------------------------------------
    // Determinism.
    // ---------------------------------------------------------------------

    #[test]
    fn aggregation_is_independent_of_hash_map_order() {
        let updates = contaminated_cohort();
        let mut first = ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::MultiKrum { f: 2, m: 5 },
            ..ByzantineRobustConfig::default()
        })
        .expect("config");
        let reference = first
            .robust_aggregate(&updates, &no_allocations())
            .expect("aggregate");

        for _ in 0..5 {
            let shuffled: HashMap<String, Array1<f64>> = updates
                .iter()
                .map(|(id, update)| (id.clone(), update.clone()))
                .collect();
            let mut aggregator =
                ByzantineRobustAggregator::<f64>::with_config(ByzantineRobustConfig {
                    method: ByzantineRobustMethod::MultiKrum { f: 2, m: 5 },
                    ..ByzantineRobustConfig::default()
                })
                .expect("config");
            let result = aggregator
                .robust_aggregate(&shuffled, &no_allocations())
                .expect("aggregate");
            assert_eq!(result.to_vec(), reference.to_vec());
        }
    }
}
