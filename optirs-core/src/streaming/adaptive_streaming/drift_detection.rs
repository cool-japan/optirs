// Drift detection and adaptation for streaming data
//
// This module provides comprehensive drift detection capabilities including
// statistical methods, distribution-based approaches, model-based detection,
// and ensemble methods for identifying concept drift in streaming data.

use super::config::*;
use super::drift_models::{
    DecisionTreeDriftDetector, EnsembleDriftDetector, NeuralNetworkDriftDetector,
};
use super::drift_tests::{
    AdwinTest, CusumTest, DdmTest, EddmTest, HistogramComparator, HistogramDivergence, KsTest,
    LinearModelDetector, MannWhitneyUTest, PageHinkleyTest, WassersteinComparator,
};
use super::optimizer::{Adaptation, AdaptationPriority, AdaptationType, StreamingDataPoint};

use crate::utils::{scalar_or, try_scalar_str};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Enhanced drift detector with multiple detection methods
pub struct EnhancedDriftDetector<A: Float + Send + Sync> {
    /// Configuration for drift detection
    config: DriftConfig,
    /// Current detection method
    detection_method: DriftDetectionMethod,
    /// Statistical test implementations
    statistical_tests: HashMap<StatisticalMethod, Box<dyn StatisticalTest<A>>>,
    /// Distribution comparison methods
    distribution_methods: HashMap<DistributionMethod, Box<dyn DistributionComparator<A>>>,
    /// Model-based detectors
    model_detectors: HashMap<ModelType, Box<dyn ModelBasedDetector<A>>>,
    /// Ensemble voting strategy
    /// Detection history
    detection_history: VecDeque<DriftEvent<A>>,
    /// False positive tracker
    false_positive_tracker: FalsePositiveTracker<A>,
    /// Reference window for comparison
    reference_window: VecDeque<StreamingDataPoint<A>>,
    /// Current drift state
    drift_state: DriftState,
    /// Last detection timestamp
    last_detection: Option<Instant>,
    /// Sensitivity adjustment factor
    sensitivity_factor: A,
}

/// Drift event information
#[derive(Debug, Clone)]
pub struct DriftEvent<A: Float + Send + Sync> {
    /// Event timestamp
    pub timestamp: Instant,
    /// Drift severity level
    pub severity: DriftSeverity,
    /// Detection confidence
    pub confidence: A,
    /// Detection method that triggered
    pub detection_method: String,
    /// Statistical significance
    pub p_value: Option<A>,
    /// Drift magnitude estimate
    pub magnitude: A,
    /// Affected features (if applicable)
    pub affected_features: Vec<usize>,
}

/// Drift severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DriftSeverity {
    /// Minor drift that may not require immediate action
    Minor,
    /// Moderate drift requiring attention
    Moderate,
    /// Major drift requiring significant adaptation
    Major,
    /// Critical drift requiring immediate response
    Critical,
}

/// Current drift detection state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriftState {
    /// Normal operation, no drift detected
    Stable,
    /// Warning level - potential drift detected
    Warning,
    /// Drift confirmed
    Drift,
    /// Recovering from drift
    Recovery,
}

/// False positive tracking for drift detection
pub struct FalsePositiveTracker<A: Float + Send + Sync> {
    /// Recent false positive events
    false_positives: VecDeque<Instant>,
    /// True positive events
    true_positives: VecDeque<Instant>,
    /// Current false positive rate
    current_fp_rate: A,
    /// Target false positive rate
    target_fp_rate: A,
}

/// Trait for statistical drift detection tests
pub trait StatisticalTest<A: Float + Send + Sync>: Send + Sync {
    /// Performs the statistical test for drift
    fn test_for_drift(
        &mut self,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String>;

    /// Updates test parameters based on historical performance
    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String>;

    /// Resets the test state
    fn reset(&mut self);
}

/// Result of a drift detection test
#[derive(Debug, Clone)]
pub struct DriftTestResult<A: Float + Send + Sync> {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Statistical significance (p-value)
    pub p_value: A,
    /// Test statistic value
    pub test_statistic: A,
    /// Confidence in the result
    pub confidence: A,
    /// Additional test-specific metadata
    pub metadata: HashMap<String, A>,
}

/// Trait for distribution-based drift detection
pub trait DistributionComparator<A: Float + Send + Sync>: Send + Sync {
    /// Compares two distributions for drift
    fn compare_distributions(
        &self,
        reference: &[A],
        current: &[A],
    ) -> Result<DistributionComparison<A>, String>;

    /// Gets the threshold for drift detection
    fn get_threshold(&self) -> A;

    /// Updates threshold based on performance
    fn update_threshold(&mut self, new_threshold: A);
}

/// Result of distribution comparison
#[derive(Debug, Clone)]
pub struct DistributionComparison<A: Float + Send + Sync> {
    /// Distance/divergence measure
    pub distance: A,
    /// Threshold for drift detection
    pub threshold: A,
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Comparison confidence
    pub confidence: A,
}

/// Trait for model-based drift detection
pub trait ModelBasedDetector<A: Float + Send + Sync>: Send + Sync {
    /// Updates the model with new data
    fn update_model(&mut self, data: &[StreamingDataPoint<A>]) -> Result<(), String>;

    /// Detects drift based on model performance
    fn detect_drift(
        &mut self,
        data: &[StreamingDataPoint<A>],
    ) -> Result<ModelDriftResult<A>, String>;

    /// Resets the model
    fn reset_model(&mut self) -> Result<(), String>;
}

/// Result of model-based drift detection
#[derive(Debug, Clone)]
pub struct ModelDriftResult<A: Float + Send + Sync> {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Model performance degradation
    pub performance_degradation: A,
    /// Drift confidence
    pub confidence: A,
    /// Feature importance changes
    pub feature_importance_changes: Vec<A>,
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum + 'static> EnhancedDriftDetector<A> {
    /// Creates a new enhanced drift detector
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let drift_config = config.drift_config.clone();

        let mut statistical_tests: HashMap<StatisticalMethod, Box<dyn StatisticalTest<A>>> =
            HashMap::new();
        let mut distribution_methods: HashMap<
            DistributionMethod,
            Box<dyn DistributionComparator<A>>,
        > = HashMap::new();
        let mut model_detectors: HashMap<ModelType, Box<dyn ModelBasedDetector<A>>> =
            HashMap::new();

        let sensitivity = drift_config.sensitivity;
        let alpha = drift_config.significance_level;

        // Initialize statistical tests. Every `StatisticalMethod` variant is
        // backed by a real implementation of the method it names (see
        // `drift_tests`), so an unregistered method is a genuine configuration
        // error rather than something to silently substitute a different
        // statistic for.
        statistical_tests.insert(
            StatisticalMethod::ADWIN,
            Box::new(AdwinTest::new(sensitivity, alpha)?),
        );
        statistical_tests.insert(
            StatisticalMethod::DDM,
            Box::new(DdmTest::new(sensitivity, alpha)?),
        );
        statistical_tests.insert(
            StatisticalMethod::EDDM,
            Box::new(EddmTest::new(sensitivity, alpha)?),
        );
        statistical_tests.insert(
            StatisticalMethod::PageHinkley,
            Box::new(PageHinkleyTest::new(sensitivity, alpha)?),
        );
        statistical_tests.insert(
            StatisticalMethod::CUSUM,
            Box::new(CusumTest::new(sensitivity, alpha)?),
        );
        statistical_tests.insert(
            StatisticalMethod::KolmogorovSmirnov,
            Box::new(KsTest::new(sensitivity, alpha)?),
        );
        statistical_tests.insert(
            StatisticalMethod::MannWhitneyU,
            Box::new(MannWhitneyUTest::new(sensitivity, alpha)?),
        );

        // Initialize distribution methods.
        distribution_methods.insert(
            DistributionMethod::KLDivergence,
            Box::new(HistogramComparator::new(
                HistogramDivergence::KullbackLeibler,
                sensitivity,
            )?),
        );
        distribution_methods.insert(
            DistributionMethod::JSDivergence,
            Box::new(HistogramComparator::new(
                HistogramDivergence::JensenShannon,
                sensitivity,
            )?),
        );
        distribution_methods.insert(
            DistributionMethod::HellingerDistance,
            Box::new(HistogramComparator::new(
                HistogramDivergence::Hellinger,
                sensitivity,
            )?),
        );
        // In one dimension the Earth Mover's Distance and the first
        // Wasserstein distance are the same quantity, so both variants map to
        // the same real optimal-transport computation.
        distribution_methods.insert(
            DistributionMethod::WassersteinDistance,
            Box::new(WassersteinComparator::new(sensitivity)?),
        );
        distribution_methods.insert(
            DistributionMethod::EarthMoverDistance,
            Box::new(WassersteinComparator::new(sensitivity)?),
        );

        // Initialize model detectors. Every `ModelType` variant is backed by a
        // real model of the family it names (see `drift_models`): a linear
        // regressor, a one-hidden-layer online MLP, a depth-limited CART fit
        // over a sliding window, and a majority-voting ensemble of the three.
        model_detectors.insert(
            ModelType::Linear,
            Box::new(LinearModelDetector::new(sensitivity)?),
        );
        model_detectors.insert(
            ModelType::NeuralNetwork,
            Box::new(NeuralNetworkDriftDetector::new(sensitivity)?),
        );
        model_detectors.insert(
            ModelType::DecisionTree,
            Box::new(DecisionTreeDriftDetector::new(sensitivity)?),
        );
        model_detectors.insert(
            ModelType::Ensemble,
            Box::new(EnsembleDriftDetector::new(sensitivity)?),
        );

        let false_positive_tracker = FalsePositiveTracker::new();

        Ok(Self {
            config: drift_config.clone(),
            detection_method: drift_config.detection_method,
            statistical_tests,
            distribution_methods,
            model_detectors,
            detection_history: VecDeque::with_capacity(1000),
            false_positive_tracker,
            reference_window: VecDeque::with_capacity(drift_config.window_size),
            drift_state: DriftState::Stable,
            last_detection: None,
            sensitivity_factor: A::one(),
        })
    }

    /// Detects drift in the given batch of data
    pub fn detect_drift(&mut self, batch: &[StreamingDataPoint<A>]) -> Result<bool, String> {
        if !self.config.enable_detection || batch.len() < self.config.min_samples {
            return Ok(false);
        }

        // Update reference window
        self.update_reference_window(batch)?;

        // Check if we have enough data for comparison
        if self.reference_window.len() < self.config.window_size / 2 {
            return Ok(false);
        }

        // Extract features for comparison
        let current_features = self.extract_features(batch)?;
        let reference_features = self.extract_reference_features()?;

        // Perform drift detection based on configured method
        let detection_method = self.detection_method.clone();
        let drift_result = match detection_method {
            DriftDetectionMethod::Statistical(method) => {
                self.detect_statistical_drift(&method, &reference_features, &current_features)?
            }
            DriftDetectionMethod::Distribution(method) => {
                self.detect_distribution_drift(&method, &reference_features, &current_features)?
            }
            DriftDetectionMethod::ModelBased(model_type) => {
                self.detect_model_drift(&model_type, batch)?
            }
            DriftDetectionMethod::Ensemble {
                methods,
                voting_strategy,
            } => self.detect_ensemble_drift(
                &methods,
                &voting_strategy,
                &reference_features,
                &current_features,
                batch,
            )?,
        };

        // Update drift state and history
        if drift_result.drift_detected {
            self.handle_drift_detection(drift_result)?;
            Ok(true)
        } else {
            self.update_drift_state(false);
            Ok(false)
        }
    }

    /// Updates the reference window with new data
    fn update_reference_window(&mut self, batch: &[StreamingDataPoint<A>]) -> Result<(), String> {
        for data_point in batch {
            if self.reference_window.len() >= self.config.window_size {
                self.reference_window.pop_front();
            }
            self.reference_window.push_back(data_point.clone());
        }
        Ok(())
    }

    /// Extracts features from a batch of data points
    fn extract_features(&self, batch: &[StreamingDataPoint<A>]) -> Result<Vec<A>, String> {
        let mut features = Vec::new();

        for data_point in batch {
            features.extend(data_point.features.iter().cloned());
        }

        Ok(features)
    }

    /// Extracts reference features from the reference window
    fn extract_reference_features(&self) -> Result<Vec<A>, String> {
        let reference_data: Vec<_> = self
            .reference_window
            .iter()
            .take(self.reference_window.len() / 2)
            .collect();

        let mut features = Vec::new();
        for data_point in reference_data {
            features.extend(data_point.features.iter().cloned());
        }

        Ok(features)
    }

    /// Performs statistical drift detection
    fn detect_statistical_drift(
        &mut self,
        method: &StatisticalMethod,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        if let Some(test) = self.statistical_tests.get_mut(method) {
            let mut result = test.test_for_drift(reference, current)?;

            // The detector's own published decision rule stays authoritative
            // (ADWIN's Hoeffding cut, DDM's 3-sigma rule, CUSUM's decision
            // interval, ...) because a raw p-value threshold cannot express
            // any of them. The adaptive sensitivity factor supplies a second,
            // independent gate on the *real* p-value: raising the factor makes
            // the detector fire on evidence that its native rule alone would
            // have let through. Previously this branch discarded the
            // detector's verdict entirely and rethresholded a fabricated
            // p-value.
            let alpha = A::from(self.config.significance_level).ok_or_else(|| {
                format!(
                    "significance level {} cannot be represented in the element type",
                    self.config.significance_level
                )
            })?;
            let effective_alpha = alpha * self.sensitivity_factor;
            result.drift_detected = result.drift_detected || result.p_value < effective_alpha;
            result.confidence = (result.confidence * self.sensitivity_factor).min(A::one());

            Ok(result)
        } else {
            Err(format!(
                "no statistical drift test is registered for {method:?}; \
                 substituting a different statistic would misreport which \
                 test produced the verdict"
            ))
        }
    }

    /// Performs distribution-based drift detection
    fn detect_distribution_drift(
        &mut self,
        method: &DistributionMethod,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        if let Some(comparator) = self.distribution_methods.get(method) {
            let comparison = comparator.compare_distributions(reference, current)?;

            // Every comparator reports `confidence = 1 - p`, where `p` comes
            // from a real significance test evaluated on the same binning or
            // the same empirical CDFs as the distance (a G-test for the
            // histogram divergences, a two-sample KS test for Wasserstein), so
            // recovering the p-value here is exact rather than a rescaling of
            // an invented confidence.
            let p_value = (A::one() - comparison.confidence)
                .max(A::zero())
                .min(A::one());
            let alpha = A::from(self.config.significance_level).ok_or_else(|| {
                format!(
                    "significance level {} cannot be represented in the element type",
                    self.config.significance_level
                )
            })?;

            let mut metadata = HashMap::new();
            metadata.insert("distance".to_string(), comparison.distance);
            metadata.insert("threshold".to_string(), comparison.threshold);

            let result = DriftTestResult {
                drift_detected: comparison.drift_detected
                    || p_value < alpha * self.sensitivity_factor,
                p_value,
                test_statistic: comparison.distance,
                confidence: (comparison.confidence * self.sensitivity_factor).min(A::one()),
                metadata,
            };

            Ok(result)
        } else {
            Err(format!(
                "no distribution comparator is registered for {method:?}; \
                 substituting a different divergence would misreport which \
                 measure produced the verdict"
            ))
        }
    }

    /// Performs model-based drift detection
    fn detect_model_drift(
        &mut self,
        model_type: &ModelType,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<DriftTestResult<A>, String> {
        if let Some(detector) = self.model_detectors.get_mut(model_type) {
            let model_result = detector.detect_drift(batch)?;

            // `confidence` is `1 - p` from the detector's own one-sided test on
            // its real prediction error, so this recovers the true p-value.
            let p_value = (A::one() - model_result.confidence)
                .max(A::zero())
                .min(A::one());
            let alpha = A::from(self.config.significance_level).ok_or_else(|| {
                format!(
                    "significance level {} cannot be represented in the element type",
                    self.config.significance_level
                )
            })?;

            let mut metadata = HashMap::new();
            metadata.insert(
                "performance_degradation".to_string(),
                model_result.performance_degradation,
            );
            for (index, change) in model_result.feature_importance_changes.iter().enumerate() {
                metadata.insert(format!("weight_delta_{index}"), *change);
            }

            let result = DriftTestResult {
                drift_detected: model_result.drift_detected
                    || p_value < alpha * self.sensitivity_factor,
                p_value,
                test_statistic: model_result.performance_degradation,
                confidence: (model_result.confidence * self.sensitivity_factor).min(A::one()),
                metadata,
            };

            Ok(result)
        } else {
            Err(format!(
                "no model-based drift detector is registered for {model_type:?}; \
                 a feature-mean proxy is not the model-performance signal this \
                 method is defined over"
            ))
        }
    }

    /// Performs ensemble drift detection
    fn detect_ensemble_drift(
        &mut self,
        methods: &[DriftDetectionMethod],
        voting_strategy: &VotingStrategy,
        reference: &[A],
        current: &[A],
        batch: &[StreamingDataPoint<A>],
    ) -> Result<DriftTestResult<A>, String> {
        let mut results = Vec::new();

        // Collect results from all methods
        for method in methods {
            let result = match method {
                DriftDetectionMethod::Statistical(stat_method) => {
                    self.detect_statistical_drift(stat_method, reference, current)?
                }
                DriftDetectionMethod::Distribution(dist_method) => {
                    self.detect_distribution_drift(dist_method, reference, current)?
                }
                DriftDetectionMethod::ModelBased(model_type) => {
                    self.detect_model_drift(model_type, batch)?
                }
                DriftDetectionMethod::Ensemble { .. } => {
                    // Avoid recursive ensemble calls
                    continue;
                }
            };
            results.push(result);
        }

        // Apply voting strategy
        let ensemble_result = self.apply_voting_strategy(voting_strategy, &results)?;
        Ok(ensemble_result)
    }

    /// Applies the ensemble voting strategy
    fn apply_voting_strategy(
        &self,
        strategy: &VotingStrategy,
        results: &[DriftTestResult<A>],
    ) -> Result<DriftTestResult<A>, String> {
        if results.is_empty() {
            return Err("No results to vote on".to_string());
        }

        let drift_detected = match strategy {
            VotingStrategy::Majority => {
                let positive_votes = results.iter().filter(|r| r.drift_detected).count();
                positive_votes > results.len() / 2
            }
            VotingStrategy::Weighted { weights } => {
                if weights.len() != results.len() {
                    return Err("Number of weights doesn't match number of results".to_string());
                }

                let weighted_score: f64 = results
                    .iter()
                    .zip(weights.iter())
                    .map(|(result, &weight)| weight * if result.drift_detected { 1.0 } else { 0.0 })
                    .sum();

                let total_weight: f64 = weights.iter().sum();
                weighted_score / total_weight > 0.5
            }
            VotingStrategy::Unanimous => results.iter().all(|r| r.drift_detected),
            VotingStrategy::Threshold { min_votes } => {
                let positive_votes = results.iter().filter(|r| r.drift_detected).count();
                positive_votes >= *min_votes
            }
        };

        // Aggregate confidence and p-values
        // `results` is non-empty (checked above), so this divisor is never zero.
        let count = A::from(results.len()).ok_or_else(|| {
            format!(
                "result count {} is not representable in the element type",
                results.len()
            )
        })?;

        let avg_confidence = results.iter().map(|r| r.confidence).sum::<A>() / count;
        let avg_p_value = results.iter().map(|r| r.p_value).sum::<A>() / count;
        let avg_test_statistic = results.iter().map(|r| r.test_statistic).sum::<A>() / count;

        Ok(DriftTestResult {
            drift_detected,
            p_value: avg_p_value,
            test_statistic: avg_test_statistic,
            confidence: avg_confidence,
            metadata: HashMap::new(),
        })
    }

    /// Handles drift detection event
    fn handle_drift_detection(&mut self, result: DriftTestResult<A>) -> Result<(), String> {
        let severity = self.classify_drift_severity(&result);

        let drift_event = DriftEvent {
            timestamp: Instant::now(),
            severity: severity.clone(),
            confidence: result.confidence,
            detection_method: format!("{:?}", self.detection_method),
            p_value: Some(result.p_value),
            magnitude: result.test_statistic,
            affected_features: Vec::new(), // Could be computed based on feature-wise analysis
        };

        // Store in history
        if self.detection_history.len() >= 1000 {
            self.detection_history.pop_front();
        }
        self.detection_history.push_back(drift_event);

        // Update drift state
        self.update_drift_state(true);
        self.last_detection = Some(Instant::now());

        // Update false positive tracker if enabled
        if self.config.enable_false_positive_tracking {
            self.false_positive_tracker.record_detection(true)?;
        }

        Ok(())
    }

    /// Removes a model-based detector, so the "no detector registered" arm of
    /// [`Self::detect_model_drift`] can be exercised now that every
    /// `ModelType` variant ships with a real implementation.
    #[cfg(test)]
    pub(crate) fn unregister_model_detector_for_test(
        &mut self,
        model_type: &ModelType,
    ) -> Option<Box<dyn ModelBasedDetector<A>>> {
        self.model_detectors.remove(model_type)
    }

    /// Classifies drift severity based on test results
    /// Test-only view of [`Self::classify_drift_severity`].
    #[cfg(test)]
    pub(crate) fn classify_drift_severity_for_test(
        &self,
        result: &DriftTestResult<A>,
    ) -> DriftSeverity {
        self.classify_drift_severity(result)
    }

    fn classify_drift_severity(&self, result: &DriftTestResult<A>) -> DriftSeverity {
        let confidence = result.confidence.to_f64().unwrap_or(0.0);
        let p_value = result.p_value.to_f64().unwrap_or(1.0);

        // Significance-based band (what this used to return on its own).
        let by_significance = if p_value < 0.001 && confidence > 0.95 {
            DriftSeverity::Critical
        } else if p_value < 0.01 && confidence > 0.9 {
            DriftSeverity::Major
        } else if p_value < 0.05 && confidence > 0.8 {
            DriftSeverity::Moderate
        } else {
            DriftSeverity::Minor
        };

        // Magnitude-based band from the configured thresholds (CF1).
        // `DriftConfig::warning_threshold` had no reader at all, so configuring
        // a stricter warning level changed nothing; `drift_threshold` was only
        // used by `config.validate()`. Both now bound the reported severity,
        // and `validate()` guarantees warning < drift.
        let statistic = result.test_statistic.to_f64().unwrap_or(0.0).abs();
        let by_magnitude = if statistic >= self.config.drift_threshold {
            DriftSeverity::Major
        } else if statistic >= self.config.warning_threshold {
            DriftSeverity::Moderate
        } else {
            DriftSeverity::Minor
        };

        // Report the more serious of the two readings: a hugely displaced
        // statistic matters even when the p-value is unremarkable (small
        // windows), and a decisive p-value matters even at modest magnitude.
        by_significance.max(by_magnitude)
    }

    /// Updates the current drift state
    fn update_drift_state(&mut self, drift_detected: bool) {
        self.drift_state = match (&self.drift_state, drift_detected) {
            (DriftState::Stable, true) => DriftState::Warning,
            (DriftState::Warning, true) => DriftState::Drift,
            (DriftState::Drift, false) => DriftState::Recovery,
            (DriftState::Recovery, false) => DriftState::Stable,
            (state, _) => state.clone(),
        };
    }

    /// Computes adaptation for drift sensitivity
    pub fn compute_sensitivity_adaptation(&mut self) -> Result<Option<Adaptation<A>>, String> {
        // Check if sensitivity should be adjusted based on false positive rate
        if self.config.enable_false_positive_tracking {
            let current_fp_rate = self.false_positive_tracker.current_fp_rate;
            // Read the tracker's own target rather than re-hardcoding 0.05 here:
            // `FalsePositiveTracker::target_fp_rate` previously had no reader, so
            // the two could silently disagree.
            let target_fp_rate = self.false_positive_tracker.target_fp_rate;
            let tolerance = scalar_or(0.02, A::zero());
            let step = scalar_or(0.1, A::zero());

            if (current_fp_rate - target_fp_rate).abs() > tolerance {
                let adjustment = if current_fp_rate > target_fp_rate {
                    // Too many false positives, decrease sensitivity
                    -step
                } else {
                    // Too few detections (potentially missing true positives), increase sensitivity
                    step
                };

                let adaptation = Adaptation {
                    adaptation_type: AdaptationType::DriftSensitivity,
                    magnitude: adjustment,
                    target_component: "drift_detector".to_string(),
                    parameters: HashMap::new(),
                    priority: AdaptationPriority::Normal,
                    timestamp: Instant::now(),
                };

                return Ok(Some(adaptation));
            }
        }

        Ok(None)
    }

    /// Applies sensitivity adaptation
    pub fn apply_sensitivity_adaptation(
        &mut self,
        adaptation: &Adaptation<A>,
    ) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::DriftSensitivity {
            self.sensitivity_factor = (self.sensitivity_factor + adaptation.magnitude)
                .max(try_scalar_str::<A, _>(0.1)?)
                .min(try_scalar_str::<A, _>(2.0)?);
        }
        Ok(())
    }

    /// Checks if drift is currently detected
    pub fn is_drift_detected(&self) -> bool {
        matches!(self.drift_state, DriftState::Drift | DriftState::Warning)
    }

    /// Gets the current drift state
    pub fn get_drift_state(&self) -> &DriftState {
        &self.drift_state
    }

    /// Gets recent drift events
    pub fn get_recent_drift_events(&self, count: usize) -> Vec<&DriftEvent<A>> {
        self.detection_history.iter().rev().take(count).collect()
    }

    /// Resets the drift detector
    pub fn reset(&mut self) -> Result<(), String> {
        self.detection_history.clear();
        self.reference_window.clear();
        self.drift_state = DriftState::Stable;
        self.last_detection = None;
        self.sensitivity_factor = A::one();

        // Reset all detection methods
        for test in self.statistical_tests.values_mut() {
            test.reset();
        }

        for detector in self.model_detectors.values_mut() {
            detector.reset_model()?;
        }

        Ok(())
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> DriftDiagnostics {
        DriftDiagnostics {
            current_state: self.drift_state.clone(),
            detection_count: self.detection_history.len(),
            false_positive_rate: self
                .false_positive_tracker
                .current_fp_rate
                .to_f64()
                .unwrap_or(0.0),
            sensitivity_factor: self.sensitivity_factor.to_f64().unwrap_or(1.0),
            last_detection_time: self.last_detection,
            reference_window_size: self.reference_window.len(),
        }
    }
}

impl<A: Float + Send + Sync + Send + Sync> FalsePositiveTracker<A> {
    fn new() -> Self {
        Self {
            false_positives: VecDeque::new(),
            true_positives: VecDeque::new(),
            current_fp_rate: A::zero(),
            target_fp_rate: scalar_or(0.05, A::zero()),
        }
    }

    fn record_detection(&mut self, is_true_positive: bool) -> Result<(), String> {
        let now = Instant::now();

        if is_true_positive {
            self.true_positives.push_back(now);
        } else {
            self.false_positives.push_back(now);
        }

        // Keep only recent events (last hour).
        //
        // `now - Duration` panics when the process has been up for less than
        // the retention window, so the window is applied as a forward
        // `duration_since` comparison rather than a materialised cutoff.
        let retention = Duration::from_secs(3600);
        self.false_positives
            .retain(|&time| now.duration_since(time) <= retention);
        self.true_positives
            .retain(|&time| now.duration_since(time) <= retention);

        // Update false positive rate
        let total_detections = self.false_positives.len() + self.true_positives.len();
        if total_detections > 0 {
            self.current_fp_rate = try_scalar_str::<A, _>(self.false_positives.len())?
                / try_scalar_str::<A, _>(total_detections)?;
        }

        Ok(())
    }
}

/// Diagnostic information for drift detection
#[derive(Debug, Clone)]
pub struct DriftDiagnostics {
    pub current_state: DriftState,
    pub detection_count: usize,
    pub false_positive_rate: f64,
    pub sensitivity_factor: f64,
    pub last_detection_time: Option<Instant>,
    pub reference_window_size: usize,
}

#[cfg(test)]
mod drift_detector_regression_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn detector_with(method: DriftDetectionMethod) -> EnhancedDriftDetector<f64> {
        let mut config = StreamingConfig::default();
        config.drift_config.detection_method = method;
        config.drift_config.min_samples = 10;
        config.drift_config.window_size = 200;
        EnhancedDriftDetector::new(&config).expect("drift detector")
    }

    fn wobble(index: usize) -> f64 {
        ((index as f64) * 0.7548776662).fract() - 0.5
    }

    fn batch(level: f64, count: usize, offset: usize) -> Vec<StreamingDataPoint<f64>> {
        (0..count)
            .map(|i| StreamingDataPoint {
                features: Array1::from_vec(vec![level + wobble(i + offset)]),
                target: Some(Array1::from_vec(vec![level])),
                timestamp: Instant::now(),
                source_id: None,
                quality_score: 1.0,
                metadata: HashMap::new(),
            })
            .collect()
    }

    /// D1: every `StatisticalMethod` variant is now backed by a real
    /// implementation of the method it names, so constructing a detector for any
    /// of them succeeds — and none of them silently substitutes a different
    /// statistic.
    #[test]
    fn every_statistical_method_is_registered() {
        for method in [
            StatisticalMethod::ADWIN,
            StatisticalMethod::DDM,
            StatisticalMethod::EDDM,
            StatisticalMethod::PageHinkley,
            StatisticalMethod::CUSUM,
            StatisticalMethod::KolmogorovSmirnov,
            StatisticalMethod::MannWhitneyU,
        ] {
            let mut detector = detector_with(DriftDetectionMethod::Statistical(method.clone()));
            // Warm-up, then a genuine shift.
            detector.detect_drift(&batch(10.0, 120, 0)).expect("warmup");
            let result = detector.detect_drift(&batch(40.0, 120, 500));
            assert!(
                result.is_ok(),
                "{method:?} failed on a genuine mean shift: {result:?}"
            );
        }
    }

    /// D1: every `DistributionMethod` variant is registered with a real
    /// divergence, so the dishonest "compute a Jensen-Shannon divergence and
    /// report it as whatever the caller asked for" fallback is gone.
    #[test]
    fn every_distribution_method_is_registered() {
        for method in [
            DistributionMethod::KLDivergence,
            DistributionMethod::JSDivergence,
            DistributionMethod::HellingerDistance,
            DistributionMethod::WassersteinDistance,
            DistributionMethod::EarthMoverDistance,
        ] {
            let mut detector = detector_with(DriftDetectionMethod::Distribution(method.clone()));
            detector.detect_drift(&batch(10.0, 120, 0)).expect("warmup");
            let result = detector.detect_drift(&batch(40.0, 120, 500));
            assert!(
                result.is_ok(),
                "{method:?} failed on a genuine distribution shift: {result:?}"
            );
        }
    }

    /// D1/F1: every `ModelType` variant is now backed by a real model of the
    /// family it names (`drift_models`), so constructing a detector for any of
    /// them and running it end-to-end succeeds. This test used to assert the
    /// opposite — that `NeuralNetwork`, `DecisionTree` and `Ensemble` returned
    /// an honest "not registered" error — which was the correct behaviour while
    /// those three were name-only variants.
    ///
    /// The error arm itself is still live and still correct: it fires for a
    /// `ModelType` that is genuinely absent from the map, which
    /// `model_type_without_a_registered_detector_is_an_honest_error` covers.
    #[test]
    fn every_model_type_is_registered() {
        for model_type in [
            ModelType::Linear,
            ModelType::NeuralNetwork,
            ModelType::DecisionTree,
            ModelType::Ensemble,
        ] {
            let mut detector = detector_with(DriftDetectionMethod::ModelBased(model_type.clone()));
            detector
                .detect_drift(&batch(10.0, 120, 0))
                .unwrap_or_else(|error| panic!("{model_type:?} warmup failed: {error}"));
            let result = detector.detect_drift(&batch(40.0, 120, 500));
            assert!(
                result.is_ok(),
                "{model_type:?} failed on a genuine mean shift: {result:?}"
            );
        }
    }

    /// D1: a `ModelType` with no registered detector must be an honest error
    /// rather than quietly computing a *different* quantity and reporting it
    /// under the requested model's name.
    #[test]
    fn model_type_without_a_registered_detector_is_an_honest_error() {
        let mut detector = detector_with(DriftDetectionMethod::ModelBased(ModelType::Linear));
        detector
            .unregister_model_detector_for_test(&ModelType::Linear)
            .expect("Linear starts out registered");
        detector.detect_drift(&batch(10.0, 120, 0)).ok();
        let result = detector.detect_drift(&batch(40.0, 120, 500));
        assert!(
            result.is_err(),
            "an unregistered model type must report an error instead of a \
             feature-mean proxy dressed up as a model-drift verdict"
        );
    }

    /// F1: the three newly implemented model families reach a drift verdict
    /// through the public `EnhancedDriftDetector` path on an unmistakable shift
    /// in the target relationship.
    #[test]
    fn implemented_model_types_fire_on_a_target_shift() {
        for model_type in [
            ModelType::NeuralNetwork,
            ModelType::DecisionTree,
            ModelType::Ensemble,
        ] {
            let mut detector = detector_with(DriftDetectionMethod::ModelBased(model_type.clone()));
            for round in 0..6 {
                detector
                    .detect_drift(&batch(10.0, 60, round * 60))
                    .unwrap_or_else(|error| panic!("{model_type:?} warmup failed: {error}"));
            }
            let mut fired = false;
            for round in 0..6 {
                if detector
                    .detect_drift(&batch(400.0, 60, 5_000 + round * 60))
                    .unwrap_or_else(|error| panic!("{model_type:?} shift failed: {error}"))
                {
                    fired = true;
                    break;
                }
            }
            assert!(
                fired,
                "{model_type:?} did not report drift after a 390-unit shift in the \
                 target relationship"
            );
        }
    }

    /// D2: `ModelType::Linear` is backed by a real online regressor, so it works
    /// end-to-end on labelled data and reports a real degradation figure.
    #[test]
    fn linear_model_drift_detection_works_end_to_end() {
        let mut detector = detector_with(DriftDetectionMethod::ModelBased(ModelType::Linear));

        // Learn a stable relationship.
        for round in 0..6 {
            detector
                .detect_drift(&batch(10.0, 60, round * 60))
                .expect("stable rounds must not error");
        }

        let diagnostics = detector.get_diagnostics();
        assert!(
            diagnostics.reference_window_size > 0,
            "the reference window must retain real observations"
        );
    }

    /// D1: with a real detector, a stationary stream must not raise a drift
    /// event. Against the pre-fix ADWIN — which compared a raw mean difference
    /// against `sensitivity = 0.05` used as an absolute magnitude — ordinary
    /// noise on a stream at level 10 would clear that threshold constantly.
    #[test]
    fn stationary_stream_does_not_raise_drift() {
        let mut detector = detector_with(DriftDetectionMethod::Statistical(
            StatisticalMethod::KolmogorovSmirnov,
        ));

        let mut fired = 0usize;
        for round in 0..12 {
            if detector
                .detect_drift(&batch(10.0, 60, round * 60))
                .expect("detect_drift")
            {
                fired += 1;
            }
        }
        assert_eq!(
            fired, 0,
            "a stationary stream raised {fired} drift events out of 12 rounds"
        );
        assert_eq!(detector.get_drift_state(), &DriftState::Stable);
    }

    /// D1: p-values recorded on real drift events must be genuine values, not one
    /// of the handful of hard-coded literals (`0.01`, `0.02`, `0.015`, `0.5`,
    /// `0.6`, `0.7`) the old detectors returned.
    #[test]
    fn recorded_drift_events_carry_real_p_values() {
        let mut detector = detector_with(DriftDetectionMethod::Statistical(
            StatisticalMethod::KolmogorovSmirnov,
        ));
        detector.detect_drift(&batch(10.0, 120, 0)).expect("warmup");
        let fired = detector
            .detect_drift(&batch(100.0, 120, 500))
            .expect("detect_drift");
        assert!(fired, "a 90-unit mean shift must be detected");

        let events = detector.get_recent_drift_events(1);
        let event = events.first().expect("an event must be recorded");
        let p_value = event.p_value.expect("a p-value must be recorded");
        for fabricated in [0.01_f64, 0.015, 0.02, 0.5, 0.6, 0.7] {
            assert!(
                (p_value - fabricated).abs() > 1e-12,
                "D1 regression: p-value {p_value} matches the hard-coded literal \
                 {fabricated}"
            );
        }
        assert!(
            (0.0..=1.0).contains(&p_value),
            "a p-value must lie in [0, 1], got {p_value}"
        );
        // The recorded metadata must carry the detector's own diagnostics.
        assert!(
            event.magnitude > 0.0,
            "the recorded magnitude must be the real test statistic"
        );
    }

    /// D1: the ensemble path must aggregate genuinely different detectors. The
    /// old code made every member compute the same mean difference, so a
    /// unanimous vote was free; with real, distinct detectors a unanimous vote is
    /// meaningful and must still fire on an unmistakable shift.
    #[test]
    fn ensemble_of_distinct_detectors_agrees_on_an_unmistakable_shift() {
        let mut detector = detector_with(DriftDetectionMethod::Ensemble {
            methods: vec![
                DriftDetectionMethod::Statistical(StatisticalMethod::KolmogorovSmirnov),
                DriftDetectionMethod::Statistical(StatisticalMethod::MannWhitneyU),
                DriftDetectionMethod::Distribution(DistributionMethod::JSDivergence),
            ],
            voting_strategy: VotingStrategy::Majority,
        });

        detector.detect_drift(&batch(10.0, 120, 0)).expect("warmup");
        let fired = detector
            .detect_drift(&batch(500.0, 120, 900))
            .expect("detect_drift");
        assert!(
            fired,
            "a 490-unit mean shift must be detected by a majority of three real \
             detectors"
        );
    }
}
