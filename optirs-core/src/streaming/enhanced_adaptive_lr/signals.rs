// Real adaptation signals for the enhanced adaptive learning-rate controller
// (findings E1, E3, E4, E5, E6).
//
// The block this file replaces was introduced in the source as "Implementation
// stubs for the various components / In a full implementation, these would
// contain sophisticated algorithms":
//
// * E1/E5 — `resolve_signals` multiplied the hardcoded literal `0.001` by the
//   vote instead of the live learning rate, and never read `signal_weights`,
//   `signal_reliability`, `conflict_resolution`, `voting_history` or
//   `last_decision`. Four of the five conflict-resolution strategies were
//   unreachable.
// * E3 — `DriftAwareAdapter::generate_signal` returned a hardcoded "No drift
//   detected" vote because `drift_detectors` was `vec![]` and the local
//   detector type had no methods.
// * E4 — `memory_pressure` was never written, so the resource signal always
//   read `0.0`, interpreted it as spare capacity and pushed the learning rate
//   up on every single step.
// * E6 — every sub-adapter constructor took the configuration and ignored it.

use super::*;

/// Bound on the retained signal-vote log.
const MAX_VOTING_HISTORY: usize = 512;

/// Bound on the retained per-metric history.
const MAX_METRIC_HISTORY: usize = 512;

fn to_scalar<A: Float>(value: f64) -> A {
    A::from(value).unwrap_or_else(A::zero)
}

fn from_scalar<A: Float>(value: A) -> f64 {
    value.to_f64().unwrap_or(0.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|value| (value - m).powi(2)).sum::<f64>() / values.len() as f64
}

/// Ordinary-least-squares slope against the sample index.
fn slope(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let x_mean = (values.len() - 1) as f64 / 2.0;
    let y_mean = mean(values);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (index, value) in values.iter().enumerate() {
        let dx = index as f64 - x_mean;
        numerator += dx * (value - y_mean);
        denominator += dx * dx;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

// ---------------------------------------------------------------------------
// E1/E5: multi-signal resolution
// ---------------------------------------------------------------------------

impl<A: Float + Default + Clone + Send + Sync> MultiSignalAdaptationStrategy<A> {
    pub(crate) fn new(config: &AdaptiveLRConfig<A>) -> Result<Self> {
        // E6: the configuration is actually read now. Signals the operator
        // disabled get zero weight, so they cannot influence a decision even if
        // something else pushes a vote for them.
        let mut signal_weights = HashMap::new();
        let enabled_weight = A::one();
        let disabled_weight = A::zero();
        signal_weights.insert(
            AdaptationSignalType::GradientMagnitude,
            if config.enable_gradient_adaptation {
                enabled_weight
            } else {
                disabled_weight
            },
        );
        signal_weights.insert(
            AdaptationSignalType::GradientVariance,
            if config.enable_gradient_adaptation {
                enabled_weight
            } else {
                disabled_weight
            },
        );
        signal_weights.insert(
            AdaptationSignalType::LossProgression,
            if config.enable_performance_adaptation {
                enabled_weight
            } else {
                disabled_weight
            },
        );
        signal_weights.insert(
            AdaptationSignalType::AccuracyTrend,
            if config.enable_performance_adaptation {
                enabled_weight
            } else {
                disabled_weight
            },
        );
        signal_weights.insert(
            AdaptationSignalType::ConceptDrift,
            if config.enable_drift_adaptation {
                enabled_weight
            } else {
                disabled_weight
            },
        );
        signal_weights.insert(
            AdaptationSignalType::ResourceUtilization,
            if config.enable_resource_adaptation {
                enabled_weight
            } else {
                disabled_weight
            },
        );
        signal_weights.insert(AdaptationSignalType::ModelComplexity, enabled_weight);
        signal_weights.insert(AdaptationSignalType::DataQuality, enabled_weight);

        Ok(Self {
            signal_weights,
            voting_history: VecDeque::with_capacity(MAX_VOTING_HISTORY),
            conflict_resolution: if config.use_ensemble_voting {
                ConflictResolution::WeightedAverage
            } else {
                ConflictResolution::HighestConfidence
            },
            signal_reliability: HashMap::new(),
            last_decision: None,
        })
    }

    /// Effective weight of a signal: its configured weight times its measured
    /// reliability (seeded neutrally at 1.0 until effectiveness is observed).
    fn effective_weight(&self, signal_type: AdaptationSignalType) -> f64 {
        let configured = self
            .signal_weights
            .get(&signal_type)
            .map(|weight| from_scalar(*weight))
            .unwrap_or(1.0);
        let reliability = self
            .signal_reliability
            .get(&signal_type)
            .map(|value| from_scalar(*value))
            .unwrap_or(1.0);
        (configured * reliability).max(0.0)
    }

    pub(crate) fn resolve_signals(
        &mut self,
        signals: Vec<SignalVote<A>>,
        current_lr: A,
        sensitivity: A,
        use_ensemble_voting: bool,
        _step: usize,
    ) -> Result<AdaptationDecision<A>> {
        // Record every vote, bounded.
        for signal in &signals {
            self.voting_history.push_back(signal.clone());
        }
        while self.voting_history.len() > MAX_VOTING_HISTORY {
            self.voting_history.pop_front();
        }

        if signals.is_empty() {
            let decision = AdaptationDecision {
                // With no signal there is no reason to change anything; the old
                // code snapped the learning rate to a hardcoded 0.001 here.
                new_lr: current_lr,
                lr_multiplier: A::one(),
                contributing_signals: vec![],
                confidence: A::zero(),
                rationale: "No signals available; learning rate left unchanged".to_string(),
                timestamp: Instant::now(),
            };
            self.last_decision = Some(decision.clone());
            return Ok(decision);
        }

        self.conflict_resolution = if use_ensemble_voting {
            self.conflict_resolution
        } else {
            ConflictResolution::HighestConfidence
        };

        // Weighted candidates: (multiplier, weight, confidence, signal type).
        let candidates: Vec<(f64, f64, f64, AdaptationSignalType)> = signals
            .iter()
            .map(|signal| {
                (
                    from_scalar(signal.recommended_lr_change),
                    self.effective_weight(signal.signal_type) * from_scalar(signal.confidence),
                    from_scalar(signal.confidence),
                    signal.signal_type,
                )
            })
            .filter(|(multiplier, weight, _, _)| multiplier.is_finite() && *weight > 0.0)
            .collect();

        if candidates.is_empty() {
            let decision = AdaptationDecision {
                new_lr: current_lr,
                lr_multiplier: A::one(),
                contributing_signals: signals.iter().map(|s| s.signal_type).collect(),
                confidence: A::zero(),
                rationale: "All adaptation signals are disabled or unreliable".to_string(),
                timestamp: Instant::now(),
            };
            self.last_decision = Some(decision.clone());
            return Ok(decision);
        }

        // E1/E5: all five conflict-resolution strategies are reachable now.
        let (raw_multiplier, rationale) = match self.conflict_resolution {
            ConflictResolution::WeightedAverage => {
                let total_weight: f64 = candidates.iter().map(|(_, weight, _, _)| weight).sum();
                let weighted: f64 = candidates
                    .iter()
                    .map(|(multiplier, weight, _, _)| multiplier * weight)
                    .sum();
                (
                    weighted / total_weight,
                    "Reliability-weighted average of adaptation signals".to_string(),
                )
            }
            ConflictResolution::HighestConfidence => {
                let best = candidates
                    .iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .unwrap_or((1.0, 1.0, 0.0, AdaptationSignalType::LossProgression));
                (best.0, format!("Highest-confidence signal ({:?})", best.3))
            }
            ConflictResolution::MajorityVote { threshold } => {
                // Group votes by direction; a direction needs `threshold` of the
                // weight to win, otherwise nothing changes.
                let total_weight: f64 = candidates.iter().map(|(_, weight, _, _)| weight).sum();
                let up: f64 = candidates
                    .iter()
                    .filter(|(multiplier, _, _, _)| *multiplier > 1.0)
                    .map(|(_, weight, _, _)| weight)
                    .sum();
                let down: f64 = candidates
                    .iter()
                    .filter(|(multiplier, _, _, _)| *multiplier < 1.0)
                    .map(|(_, weight, _, _)| weight)
                    .sum();
                let share = |value: f64| {
                    if total_weight > 0.0 {
                        value / total_weight
                    } else {
                        0.0
                    }
                };
                if share(up) >= threshold {
                    let mean_up = mean(
                        &candidates
                            .iter()
                            .filter(|(multiplier, _, _, _)| *multiplier > 1.0)
                            .map(|(multiplier, _, _, _)| *multiplier)
                            .collect::<Vec<f64>>(),
                    );
                    (mean_up, "Majority voted to increase".to_string())
                } else if share(down) >= threshold {
                    let mean_down = mean(
                        &candidates
                            .iter()
                            .filter(|(multiplier, _, _, _)| *multiplier < 1.0)
                            .map(|(multiplier, _, _, _)| *multiplier)
                            .collect::<Vec<f64>>(),
                    );
                    (mean_down, "Majority voted to decrease".to_string())
                } else {
                    (
                        1.0,
                        "No direction reached the majority threshold".to_string(),
                    )
                }
            }
            ConflictResolution::Conservative => {
                // The smallest change any signal asked for.
                let smallest = candidates
                    .iter()
                    .min_by(|a, b| {
                        (a.0 - 1.0)
                            .abs()
                            .partial_cmp(&(b.0 - 1.0).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(multiplier, _, _, _)| *multiplier)
                    .unwrap_or(1.0);
                (
                    smallest,
                    "Conservative resolution: smallest requested change".to_string(),
                )
            }
            ConflictResolution::MetaLearned => {
                // Weight each signal by its *measured* reliability alone, which
                // is what the meta-learned resolution means here: trust the
                // signals that have historically produced improvements.
                let mut total = 0.0;
                let mut weighted = 0.0;
                for (multiplier, _, _, signal_type) in &candidates {
                    let reliability = self
                        .signal_reliability
                        .get(signal_type)
                        .map(|value| from_scalar(*value))
                        .unwrap_or(0.0);
                    total += reliability;
                    weighted += multiplier * reliability;
                }
                if total > 0.0 {
                    (
                        weighted / total,
                        "Meta-learned resolution from measured signal reliability".to_string(),
                    )
                } else {
                    // Nothing has been measured yet; say so instead of guessing.
                    (
                        1.0,
                        "Meta-learned resolution has no measured reliability yet".to_string(),
                    )
                }
            }
        };

        // `adaptation_sensitivity` scales how far a single decision may move the
        // learning rate (it used to be ignored entirely).
        let sensitivity = from_scalar(sensitivity).abs().clamp(0.0, 1.0);
        let scale = if sensitivity > 0.0 { sensitivity } else { 1.0 };
        let multiplier = 1.0 + (raw_multiplier - 1.0) * scale;
        let multiplier = if multiplier.is_finite() && multiplier > 0.0 {
            multiplier
        } else {
            1.0
        };

        let confidence = mean(
            &candidates
                .iter()
                .map(|(_, _, confidence, _)| *confidence)
                .collect::<Vec<f64>>(),
        );

        let decision = AdaptationDecision {
            // The live learning rate is the base, so adaptation composes across
            // steps instead of resetting.
            new_lr: current_lr * to_scalar(multiplier),
            lr_multiplier: to_scalar(multiplier),
            contributing_signals: candidates
                .iter()
                .map(|(_, _, _, signal_type)| *signal_type)
                .collect(),
            confidence: to_scalar(confidence),
            rationale,
            timestamp: Instant::now(),
        };
        self.last_decision = Some(decision.clone());
        Ok(decision)
    }

    pub(crate) fn update_signal_reliability(
        &mut self,
        signal_type: AdaptationSignalType,
        effectiveness: A,
    ) {
        let reliability = self
            .signal_reliability
            .entry(signal_type)
            .or_insert_with(|| A::from(0.5).unwrap_or_else(A::zero));

        // Update reliability using exponential moving average, clamped to a
        // non-negative weight so a bad run cannot invert a signal's vote.
        let alpha = A::from(0.1).unwrap_or_else(A::zero);
        let updated = (*reliability) * (A::one() - alpha) + effectiveness * alpha;
        *reliability = updated.max(A::zero());
    }
}

// ---------------------------------------------------------------------------
// gradient-based signal
// ---------------------------------------------------------------------------

impl<A: Float + Default + Clone + Send + Sync> GradientBasedAdapter<A> {
    pub(crate) fn new(config: &AdaptiveLRConfig<A>) -> Result<Self> {
        // E6: the history window comes from the configuration now.
        let window = config.history_window_size.clamp(4, MAX_METRIC_HISTORY);
        Ok(Self {
            magnitude_history: VecDeque::with_capacity(window),
            direction_variance_history: VecDeque::with_capacity(window),
            norm_statistics: GradientNormStatistics::default(),
            snr_estimator: SignalToNoiseEstimator::default(),
            staleness_detector: GradientStalenessDetector::default(),
        })
    }

    /// Fold a gradient into the rolling statistics without producing a vote.
    pub(crate) fn observe_only(&mut self, gradients: &Array1<A>) {
        let magnitude = gradients
            .iter()
            .map(|&g| g * g)
            .fold(A::zero(), |acc, x| acc + x)
            .sqrt();
        self.push_magnitude(magnitude);
    }

    fn push_magnitude(&mut self, magnitude: A) {
        self.magnitude_history.push_back(magnitude);
        while self.magnitude_history.len() > MAX_METRIC_HISTORY {
            self.magnitude_history.pop_front();
        }
        self.staleness_detector
            .gradient_timestamps
            .push_back(Instant::now());
        while self.staleness_detector.gradient_timestamps.len() > MAX_METRIC_HISTORY {
            self.staleness_detector.gradient_timestamps.pop_front();
        }
        self.refresh_statistics();
    }

    /// Real moments, percentiles, lag-1 autocorrelation and signal-to-noise
    /// estimate over the retained magnitudes (all of which were left at their
    /// `Default` zeros before).
    fn refresh_statistics(&mut self) {
        let values: Vec<f64> = self
            .magnitude_history
            .iter()
            .map(|value| from_scalar(*value))
            .collect();
        if values.is_empty() {
            return;
        }
        let m = mean(&values);
        let var = variance(&values);
        let std_dev = var.sqrt();

        self.norm_statistics.mean = to_scalar(m);
        self.norm_statistics.variance = to_scalar(var);
        if std_dev > f64::EPSILON {
            let third = values.iter().map(|v| (v - m).powi(3)).sum::<f64>() / values.len() as f64;
            let fourth = values.iter().map(|v| (v - m).powi(4)).sum::<f64>() / values.len() as f64;
            self.norm_statistics.skewness = to_scalar(third / std_dev.powi(3));
            self.norm_statistics.kurtosis = to_scalar(fourth / var.powi(2) - 3.0);
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let last = sorted.len() - 1;
        self.norm_statistics.percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]
            .iter()
            .map(|q| to_scalar(sorted[((sorted.len() as f64 * q) as usize).min(last)]))
            .collect();

        if values.len() >= 3 && var > f64::EPSILON {
            let numerator: f64 = values
                .windows(2)
                .map(|pair| (pair[0] - m) * (pair[1] - m))
                .sum();
            let denominator: f64 = values.iter().map(|v| (v - m).powi(2)).sum();
            self.norm_statistics.autocorrelation =
                to_scalar((numerator / denominator).clamp(-1.0, 1.0));
        }

        // Signal-to-noise: the mean magnitude against its own dispersion.
        self.snr_estimator.signal_estimate = to_scalar(m);
        self.snr_estimator.noise_estimate = to_scalar(std_dev);
        let snr = if std_dev > f64::EPSILON {
            m / std_dev
        } else {
            0.0
        };
        self.snr_estimator.snr_history.push_back(to_scalar(snr));
        while self.snr_estimator.snr_history.len() > MAX_METRIC_HISTORY {
            self.snr_estimator.snr_history.pop_front();
        }
    }

    pub(crate) fn generate_signal(
        &mut self,
        gradients: &Array1<A>,
        _step: usize,
    ) -> Result<SignalVote<A>> {
        let magnitude = gradients
            .iter()
            .map(|&g| g * g)
            .fold(A::zero(), |acc, x| acc + x)
            .sqrt();
        self.push_magnitude(magnitude);

        let magnitude_value = from_scalar(magnitude);
        if !magnitude_value.is_finite() {
            return Err(crate::error::OptimError::InvalidParameter(
                "gradient magnitude is not finite".to_string(),
            ));
        }

        // Scale-free adaptation: compare this magnitude with the median of the
        // retained history rather than against absolute constants, so the signal
        // works for any problem scale.
        let median = self
            .norm_statistics
            .percentiles
            .get(2)
            .map(|value| from_scalar(*value))
            .unwrap_or(magnitude_value);
        let snr = from_scalar(self.snr_estimator.signal_estimate)
            / from_scalar(self.snr_estimator.noise_estimate).max(f64::EPSILON);

        let recommended = if median <= f64::EPSILON {
            1.0
        } else {
            let ratio = magnitude_value / median;
            // A gradient much larger than usual means the step is too big.
            (1.0 / ratio.clamp(0.5, 2.0)).clamp(0.8, 1.2)
        };

        // Confidence rises with the amount of history and the signal-to-noise
        // ratio, rather than being the hardcoded 0.7 it used to be.
        let coverage = (self.magnitude_history.len() as f64 / 32.0).min(1.0);
        let confidence = (0.5 * coverage + 0.5 * (snr / (1.0 + snr))).clamp(0.0, 1.0);

        Ok(SignalVote {
            signal_type: AdaptationSignalType::GradientMagnitude,
            recommended_lr_change: to_scalar(recommended),
            confidence: to_scalar(confidence),
            reasoning: format!(
                "gradient magnitude {magnitude_value:.6} vs median {median:.6}, snr {snr:.3}"
            ),
            timestamp: Instant::now(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.magnitude_history.clear();
        self.direction_variance_history.clear();
        self.norm_statistics = GradientNormStatistics::default();
        self.snr_estimator = SignalToNoiseEstimator::default();
        self.staleness_detector = GradientStalenessDetector::default();
    }
}

// ---------------------------------------------------------------------------
// performance-based signal
// ---------------------------------------------------------------------------

impl<A: Float + Default + Clone + Send + Sync> PerformanceBasedAdapter<A> {
    pub(crate) fn new(config: &AdaptiveLRConfig<A>) -> Result<Self> {
        // E6: the plateau band scales with the configured sensitivity.
        let plateau_detector = PlateauDetector {
            plateau_threshold: config.adaptation_sensitivity,
            min_plateau_duration: config.adaptation_frequency.max(2),
            ..PlateauDetector::default()
        };

        let trend_analyzer = PerformanceTrendAnalyzer {
            trend_detection_window: config.history_window_size.clamp(4, 128),
            ..PerformanceTrendAnalyzer::default()
        };

        Ok(Self {
            metric_history: HashMap::new(),
            trend_analyzer,
            plateau_detector,
            overfitting_detector: OverfittingDetector::default(),
            efficiency_tracker: LearningEfficiencyTracker::default(),
        })
    }

    /// Fold a loss into the rolling history without producing a vote.
    pub(crate) fn observe_only(&mut self, loss: A) {
        let history = self.metric_history.entry("loss".to_string()).or_default();
        history.push_back(loss);
        while history.len() > MAX_METRIC_HISTORY {
            history.pop_front();
        }
    }

    pub(crate) fn generate_signal(
        &mut self,
        loss: A,
        metrics: &HashMap<String, A>,
        _step: usize,
    ) -> Result<SignalVote<A>> {
        self.observe_only(loss);
        for (name, value) in metrics {
            let history = self.metric_history.entry(name.clone()).or_default();
            history.push_back(*value);
            while history.len() > MAX_METRIC_HISTORY {
                history.pop_front();
            }
        }

        let losses: Vec<f64> = self
            .metric_history
            .get("loss")
            .map(|history| history.iter().map(|value| from_scalar(*value)).collect())
            .unwrap_or_default();

        let window = self.trend_analyzer.trend_detection_window.max(4);
        let recent: Vec<f64> = losses.iter().rev().take(window).rev().copied().collect();

        if recent.len() < 2 {
            return Err(crate::error::OptimError::InvalidState(
                "not enough loss history for a performance signal".to_string(),
            ));
        }

        let trend = slope(&recent);
        let spread = variance(&recent).sqrt();
        let level = mean(&recent).abs().max(f64::EPSILON);
        let relative_trend = trend / level;
        let relative_spread = spread / level;

        // Plateau detection over the retained window.
        // `adaptation_sensitivity` scales a 1% relative-trend band; using it
        // directly as the band would make a sensitivity of 1.0 classify a 99%
        // relative trend as a plateau.
        let plateau_band =
            (0.01 * from_scalar(self.plateau_detector.plateau_threshold).abs()).clamp(1e-6, 0.5);
        if relative_trend.abs() < plateau_band {
            self.plateau_detector.current_plateau_length += 1;
        } else {
            self.plateau_detector.current_plateau_length = 0;
        }
        let on_plateau = self.plateau_detector.current_plateau_length
            >= self.plateau_detector.min_plateau_duration;
        self.plateau_detector.plateau_confidence = to_scalar(
            (self.plateau_detector.current_plateau_length as f64
                / self.plateau_detector.min_plateau_duration.max(1) as f64)
                .min(1.0),
        );

        // Overfitting detection, when a validation loss is reported.
        self.overfitting_detector.train_loss_history.push_back(loss);
        while self.overfitting_detector.train_loss_history.len() > MAX_METRIC_HISTORY {
            self.overfitting_detector.train_loss_history.pop_front();
        }
        let mut overfitting = false;
        if let Some(validation) = metrics.get("val_loss") {
            self.overfitting_detector
                .val_loss_history
                .push_back(*validation);
            while self.overfitting_detector.val_loss_history.len() > MAX_METRIC_HISTORY {
                self.overfitting_detector.val_loss_history.pop_front();
            }
            let validation_values: Vec<f64> = self
                .overfitting_detector
                .val_loss_history
                .iter()
                .map(|value| from_scalar(*value))
                .collect();
            if validation_values.len() >= 4 {
                // Training improving while validation degrades.
                overfitting = relative_trend < 0.0 && slope(&validation_values) > 0.0;
            }
        }

        // Learning efficiency: loss reduction per step over the window.
        if recent.len() >= 2 {
            let reduction = recent[0] - recent[recent.len() - 1];
            self.efficiency_tracker
                .loss_reduction_per_step
                .push_back(to_scalar(reduction / recent.len() as f64));
            while self.efficiency_tracker.loss_reduction_per_step.len() > MAX_METRIC_HISTORY {
                self.efficiency_tracker.loss_reduction_per_step.pop_front();
            }
            self.efficiency_tracker.efficiency_score = to_scalar(mean(
                &self
                    .efficiency_tracker
                    .loss_reduction_per_step
                    .iter()
                    .map(|value| from_scalar(*value))
                    .collect::<Vec<f64>>(),
            ));
        }

        let trend_type = if overfitting {
            TrendType::Degrading
        } else if on_plateau {
            TrendType::Plateau
        } else if relative_spread > 0.5 {
            TrendType::Volatile
        } else if relative_trend < 0.0 {
            TrendType::Improving
        } else {
            TrendType::Degrading
        };
        self.trend_analyzer.trend_types = vec![trend_type];
        self.trend_analyzer.trend_strength = to_scalar(relative_trend.abs().min(1.0));
        self.efficiency_tracker.efficiency_trend = trend_type;

        let recommended = match trend_type {
            // Stuck: a larger step may escape the plateau.
            TrendType::Plateau => 1.1,
            // Loss rising or validation diverging: shorten the step.
            TrendType::Degrading => 0.9,
            // Oscillating badly: shorten the step more aggressively.
            TrendType::Volatile => 0.8,
            // Making progress: nudge upwards.
            TrendType::Improving => 1.02,
            TrendType::Oscillating => 0.85,
        };

        let coverage = (recent.len() as f64 / window as f64).min(1.0);
        let confidence = (0.4 + 0.6 * coverage).clamp(0.0, 1.0);

        Ok(SignalVote {
            signal_type: AdaptationSignalType::LossProgression,
            recommended_lr_change: to_scalar(recommended),
            confidence: to_scalar(confidence),
            reasoning: format!(
                "{trend_type:?}: relative trend {relative_trend:.5}, spread {relative_spread:.5}"
            ),
            timestamp: Instant::now(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.metric_history.clear();
        self.plateau_detector.current_plateau_length = 0;
        self.overfitting_detector.train_loss_history.clear();
        self.overfitting_detector.val_loss_history.clear();
        self.efficiency_tracker.loss_reduction_per_step.clear();
    }
}

// ---------------------------------------------------------------------------
// E3: drift-aware signal backed by the real detectors
// ---------------------------------------------------------------------------

/// The real detector behind [`ConceptDriftDetector`], selected by
/// [`DriftDetectionMethod`].
#[derive(Debug, Clone)]
pub enum LossDriftDetector<A: Float + Send + Sync> {
    /// Page-Hinkley cumulative-sum test
    PageHinkley(crate::streaming::concept_drift::PageHinkleyDetector<A>),
    /// Adaptive windowing
    Adwin(crate::streaming::concept_drift::AdwinDetector<A>),
    /// Drift detection method over a running-mean error indicator
    Ddm {
        detector: crate::streaming::concept_drift::DdmDetector<A>,
        running_mean: A,
        samples: usize,
    },
}

impl<A: Float + std::iter::Sum + Send + Sync> LossDriftDetector<A> {
    fn for_method(method: DriftDetectionMethod, threshold: A, window: usize) -> Self {
        let warning = threshold * A::from(0.5).unwrap_or_else(A::one);
        match method {
            DriftDetectionMethod::PageHinkley => LossDriftDetector::PageHinkley(
                crate::streaming::concept_drift::PageHinkleyDetector::new(threshold, warning),
            ),
            DriftDetectionMethod::ADWIN | DriftDetectionMethod::KSWIN => {
                // Both are windowing tests; ADWIN's Hoeffding cut is the real
                // implementation available here, and its confidence parameter is
                // derived from the configured threshold.
                let lower = A::from(1e-9).unwrap_or_else(A::zero);
                let upper = A::from(0.5).unwrap_or_else(A::one);
                let delta = if threshold > A::zero() {
                    (A::one() / threshold).clamp(lower, upper)
                } else {
                    upper
                };
                LossDriftDetector::Adwin(crate::streaming::concept_drift::AdwinDetector::new(
                    delta,
                    window.max(16),
                ))
            }
            DriftDetectionMethod::DDM
            | DriftDetectionMethod::EDDM
            | DriftDetectionMethod::Statistical => LossDriftDetector::Ddm {
                detector: crate::streaming::concept_drift::DdmDetector::with_warmup(
                    window.clamp(8, 64),
                ),
                running_mean: A::zero(),
                samples: 0,
            },
        }
    }

    fn update(&mut self, value: A) -> crate::streaming::concept_drift::DriftStatus {
        match self {
            LossDriftDetector::PageHinkley(detector) => detector.update(value),
            LossDriftDetector::Adwin(detector) => detector.update(value),
            LossDriftDetector::Ddm {
                detector,
                running_mean,
                samples,
            } => {
                *samples += 1;
                let count = A::from(*samples).unwrap_or_else(A::one);
                *running_mean = *running_mean + (value - *running_mean) / count;
                detector.update(value > *running_mean)
            }
        }
    }
}

impl<A: Float + Default + Clone + std::iter::Sum + Send + Sync> ConceptDriftDetector<A> {
    /// Build a detector for `method` over a `window`-sample window.
    pub fn new(method: DriftDetectionMethod, threshold: A, window: usize) -> Self {
        Self {
            detection_method: method,
            drift_threshold: threshold,
            window_size: window,
            drift_confidence: A::zero(),
            last_drift_time: None,
            inner: LossDriftDetector::for_method(method, threshold, window),
        }
    }

    /// Feed a value and report whether drift was detected.
    pub fn update(&mut self, value: A) -> crate::streaming::concept_drift::DriftStatus {
        let status = self.inner.update(value);
        self.drift_confidence = match status {
            crate::streaming::concept_drift::DriftStatus::Drift => A::one(),
            crate::streaming::concept_drift::DriftStatus::Warning => {
                A::from(0.5).unwrap_or_else(A::zero)
            }
            crate::streaming::concept_drift::DriftStatus::Stable => A::zero(),
        };
        if status == crate::streaming::concept_drift::DriftStatus::Drift {
            self.last_drift_time = Some(Instant::now());
        }
        status
    }

    /// Confidence of the most recent verdict.
    pub fn confidence(&self) -> A {
        self.drift_confidence
    }

    /// When drift was last detected.
    pub fn last_drift_time(&self) -> Option<Instant> {
        self.last_drift_time
    }
}

impl<A: Float + Default + Clone + std::iter::Sum + Send + Sync> DriftAwareAdapter<A> {
    pub(crate) fn new(config: &AdaptiveLRConfig<A>) -> Result<Self> {
        // E3/E6: three genuinely different detectors over the same stream, sized
        // from the configuration, instead of an empty vector.
        let threshold = A::from(3.0).unwrap_or_else(A::one);
        let window = config.history_window_size.clamp(16, 512);
        let drift_detectors = vec![
            ConceptDriftDetector::new(DriftDetectionMethod::PageHinkley, threshold, window),
            ConceptDriftDetector::new(DriftDetectionMethod::ADWIN, threshold, window),
            ConceptDriftDetector::new(DriftDetectionMethod::DDM, threshold, window),
        ];

        let distribution_tracker = DistributionTracker {
            ..DistributionTracker::default()
        };

        let adaptation_speed = AdaptationSpeedController {
            base_adaptation_rate: config.adaptation_sensitivity,
            current_adaptation_rate: config.adaptation_sensitivity,
            ..AdaptationSpeedController::default()
        };

        Ok(Self {
            drift_detectors,
            distribution_tracker,
            adaptation_speed,
            drift_severity: DriftSeverityAssessor::default(),
        })
    }

    pub(crate) fn generate_signal(
        &mut self,
        gradients: &Array1<A>,
        _step: usize,
    ) -> Result<SignalVote<A>> {
        let magnitude = gradients
            .iter()
            .map(|&g| g * g)
            .fold(A::zero(), |acc, x| acc + x)
            .sqrt();
        if !from_scalar(magnitude).is_finite() {
            return Err(crate::error::OptimError::InvalidParameter(
                "gradient magnitude is not finite".to_string(),
            ));
        }

        let mut drift_votes = 0usize;
        let mut warning_votes = 0usize;
        for detector in self.drift_detectors.iter_mut() {
            match detector.update(magnitude) {
                crate::streaming::concept_drift::DriftStatus::Drift => drift_votes += 1,
                crate::streaming::concept_drift::DriftStatus::Warning => warning_votes += 1,
                crate::streaming::concept_drift::DriftStatus::Stable => {}
            }
        }

        // Distribution shift over the gradient coordinates, measured as a real
        // KL divergence between the current and the reference histogram.
        let divergence = self.distribution_tracker.observe(gradients);

        let detectors = self.drift_detectors.len().max(1) as f64;
        let agreement = (drift_votes as f64 + 0.5 * warning_votes as f64) / detectors;
        let severity = self.drift_severity.assess(agreement, divergence);
        let recommended = from_scalar(severity.recommended_lr_adjustment);
        let rate = self.adaptation_speed.update(agreement);

        // The severity's recommendation is applied at the controller's current
        // adaptation rate, so a fast-moving stream reacts harder.
        let scaled = 1.0 + (recommended - 1.0) * rate;
        let confidence = (0.3 + 0.7 * agreement).clamp(0.0, 1.0);

        Ok(SignalVote {
            signal_type: AdaptationSignalType::ConceptDrift,
            recommended_lr_change: to_scalar(scaled),
            confidence: to_scalar(confidence),
            reasoning: format!(
                "{:?} drift ({drift_votes}/{} detectors, KL {:.4})",
                severity.level,
                self.drift_detectors.len(),
                divergence
            ),
            timestamp: Instant::now(),
        })
    }

    pub(crate) fn reset(&mut self) {
        for detector in self.drift_detectors.iter_mut() {
            detector.drift_confidence = A::zero();
            detector.last_drift_time = None;
            detector.inner = LossDriftDetector::for_method(
                detector.detection_method,
                detector.drift_threshold,
                detector.window_size,
            );
        }
        self.distribution_tracker = DistributionTracker::default();
        self.drift_severity = DriftSeverityAssessor::default();
    }

    /// Whether any detector currently reports drift.
    pub fn drift_detected(&self) -> bool {
        self.drift_detectors
            .iter()
            .any(|detector| from_scalar(detector.drift_confidence) >= 1.0)
    }
}

/// Number of histogram bins used for the distribution-divergence estimate.
const DISTRIBUTION_BINS: usize = 16;

/// Half-width, in reference standard deviations, of the binned range.
const DISTRIBUTION_Z_RANGE: f64 = 4.0;

impl<A: Float + Default + Send + Sync> DistributionTracker<A> {
    /// Fold the gradient coordinates into a histogram over the *reference*
    /// z-scale and return the KL divergence from the running reference
    /// distribution.
    ///
    /// Binning against the reference mean and standard deviation (rather than
    /// each observation's own min/max) is what makes the measure sensitive to a
    /// shift: a per-observation range would renormalise the shift away and
    /// report zero divergence for an arbitrarily displaced distribution.
    pub(crate) fn observe(&mut self, gradients: &Array1<A>) -> f64 {
        let values: Vec<f64> = gradients.iter().map(|value| from_scalar(*value)).collect();
        if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
            return 0.0;
        }

        let reference =
            self.feature_distributions
                .entry(0)
                .or_insert_with(|| FeatureDistribution {
                    mean: A::zero(),
                    variance: A::zero(),
                    histogram: vec![A::zero(); DISTRIBUTION_BINS],
                    last_update: Instant::now(),
                });

        let reference_mean = from_scalar(reference.mean);
        let reference_std = from_scalar(reference.variance).max(0.0).sqrt();
        let scale = if reference_std > f64::EPSILON {
            reference_std
        } else {
            variance(&values).sqrt().max(f64::EPSILON)
        };

        let width = 2.0 * DISTRIBUTION_Z_RANGE / DISTRIBUTION_BINS as f64;
        let mut counts = [0.0f64; DISTRIBUTION_BINS];
        for value in &values {
            let z = ((value - reference_mean) / scale)
                .clamp(-DISTRIBUTION_Z_RANGE, DISTRIBUTION_Z_RANGE);
            let index = (((z + DISTRIBUTION_Z_RANGE) / width) as usize).min(DISTRIBUTION_BINS - 1);
            counts[index] += 1.0;
        }
        let total: f64 = counts.iter().sum();
        let current: Vec<f64> = counts.iter().map(|count| count / total).collect();

        // KL(current || reference), with Laplace smoothing so a zero reference
        // bin cannot produce an infinite divergence.
        let smoothing = 1.0 / DISTRIBUTION_BINS as f64;
        let mut divergence = 0.0;
        let mut initialised = false;
        for (index, probability) in current.iter().enumerate() {
            let previous = from_scalar(reference.histogram[index]);
            if previous > 0.0 {
                initialised = true;
            }
            let q = (previous + smoothing) / (1.0 + 1.0);
            let p = (probability + smoothing) / (1.0 + 1.0);
            if p > 0.0 && q > 0.0 {
                divergence += p * (p / q).ln();
            }
        }

        // Update the reference with an exponential moving average.
        for (index, probability) in current.iter().enumerate() {
            let previous = from_scalar(reference.histogram[index]);
            reference.histogram[index] = to_scalar(0.9 * previous + 0.1 * probability);
        }
        // Track the reference location and spread with the same EMA so the
        // z-scale follows the stream slowly rather than jumping onto it.
        reference.mean = to_scalar(0.9 * reference_mean + 0.1 * mean(&values));
        let observed_variance = variance(&values);
        let previous_variance = from_scalar(reference.variance);
        reference.variance = to_scalar(if previous_variance > 0.0 {
            0.9 * previous_variance + 0.1 * observed_variance
        } else {
            observed_variance
        });
        reference.last_update = Instant::now();

        let divergence = if initialised && divergence.is_finite() {
            divergence.max(0.0)
        } else {
            // The first observation has nothing to diverge from.
            0.0
        };
        self.distribution_drift_score = to_scalar(divergence);
        divergence
    }
}

impl<A: Float + Default + Send + Sync> AdaptationSpeedController<A> {
    /// Accelerate while drift persists, decelerate while it does not, with
    /// momentum so the rate does not jump.
    pub(crate) fn update(&mut self, drift_agreement: f64) -> f64 {
        let base = from_scalar(self.base_adaptation_rate).max(1e-6);
        let acceleration = from_scalar(self.acceleration_factor).max(1.0);
        let deceleration = from_scalar(self.deceleration_factor).clamp(0.0, 1.0);
        let current = from_scalar(self.current_adaptation_rate).max(1e-6);

        let target = if drift_agreement > 0.0 {
            current * acceleration
        } else {
            current * deceleration
        };
        let momentum = from_scalar(self.momentum).clamp(0.0, 0.95);
        let updated = (momentum * current + (1.0 - momentum) * target).clamp(base * 0.1, 1.0);
        self.current_adaptation_rate = to_scalar(updated);
        self.momentum = to_scalar(0.9 * momentum + 0.1 * drift_agreement.clamp(0.0, 1.0));
        updated
    }
}

impl<A: Float + Default + Send + Sync> DriftSeverityAssessor<A> {
    /// Classify drift severity from detector agreement and distribution shift.
    pub(crate) fn assess(&mut self, agreement: f64, divergence: f64) -> DriftSeverityLevel<A> {
        let magnitude = (agreement + divergence.min(1.0)) / 2.0;
        // Dead band: residual sampling noise in the divergence estimate must
        // not be reported as drift on an otherwise stationary stream.
        let magnitude = if magnitude < 0.05 { 0.0 } else { magnitude };
        let (level, adjustment) = if magnitude >= 0.75 {
            (DriftSeverity::Critical, 2.0)
        } else if magnitude >= 0.5 {
            (DriftSeverity::Severe, 1.5)
        } else if magnitude >= 0.25 {
            (DriftSeverity::Moderate, 1.2)
        } else if magnitude > 0.0 {
            (DriftSeverity::Mild, 1.05)
        } else {
            (DriftSeverity::None, 1.0)
        };

        let assessed = DriftSeverityLevel {
            level,
            recommended_lr_adjustment: to_scalar(adjustment),
        };
        self.current_severity = assessed.clone();
        self.severity_history.push_back(assessed.clone());
        while self.severity_history.len() > MAX_METRIC_HISTORY {
            self.severity_history.pop_front();
        }
        if self.severity_levels.len() < 5
            && !self
                .severity_levels
                .iter()
                .any(|existing| existing.level == level)
        {
            self.severity_levels.push(assessed.clone());
        }
        assessed
    }
}

// ---------------------------------------------------------------------------
// E4: resource-aware signal from real measurements only
// ---------------------------------------------------------------------------

impl<A: Float + Default + Clone + Send + Sync> ResourceAwareAdapter<A> {
    pub(crate) fn new(config: &AdaptiveLRConfig<A>) -> Result<Self> {
        let compute_tracker = ComputationTimeTracker {
            time_budget: config.step_time_budget,
            ..ComputationTimeTracker::default()
        };

        Ok(Self {
            memory_tracker: MemoryUsageTracker::default(),
            compute_tracker,
            energy_tracker: EnergyConsumptionTracker::default(),
            throughput_requirements: ThroughputRequirements {
                min_samples_per_second: A::zero(),
                // No throughput has been observed yet; a caller supplies it
                // through `record_throughput`.
                current_throughput: A::zero(),
                throughput_deficit: A::zero(),
            },
            budget_manager: ResourceBudgetManager {
                memory_budget_mb: config.memory_budget_mb.unwrap_or(0.0),
                compute_budget_seconds: config
                    .step_time_budget
                    .map(|budget| budget.as_secs_f64())
                    .unwrap_or(0.0),
                budget_utilization: A::zero(),
                budget_violations: 0,
            },
        })
    }

    /// Record a measured step duration.
    pub(crate) fn record_step_time(&mut self, elapsed: Duration, budget: Option<Duration>) {
        self.compute_tracker.time_budget = budget;
        self.compute_tracker.step_times.push_back(elapsed);
        while self.compute_tracker.step_times.len() > MAX_METRIC_HISTORY {
            self.compute_tracker.step_times.pop_front();
        }
        let seconds: Vec<f64> = self
            .compute_tracker
            .step_times
            .iter()
            .map(|duration| duration.as_secs_f64())
            .collect();
        let average = mean(&seconds);
        self.compute_tracker.average_step_time = Duration::from_secs_f64(average.max(0.0));
        self.compute_tracker.time_pressure = match budget {
            Some(budget) if budget.as_secs_f64() > 0.0 => {
                let pressure = average / budget.as_secs_f64();
                if pressure > 1.0 {
                    self.budget_manager.budget_violations += 1;
                }
                Some(pressure)
            }
            _ => None,
        };
        if self.budget_manager.compute_budget_seconds > 0.0 {
            self.budget_manager.budget_utilization = to_scalar(
                (average / self.budget_manager.compute_budget_seconds).clamp(0.0, f64::MAX),
            );
        }
        // Throughput follows directly from the measured step time.
        if average > 0.0 {
            self.throughput_requirements.current_throughput = to_scalar(1.0 / average);
            self.refresh_throughput_deficit();
        }
    }

    /// Record real memory usage; without a budget there is no pressure to report.
    pub(crate) fn record_memory_usage(&mut self, usage_mb: f64, budget_mb: Option<f64>) {
        self.memory_tracker.current_usage_mb = usage_mb;
        self.memory_tracker.peak_usage_mb = self.memory_tracker.peak_usage_mb.max(usage_mb);
        self.memory_tracker.usage_history.push_back(usage_mb);
        while self.memory_tracker.usage_history.len() > MAX_METRIC_HISTORY {
            self.memory_tracker.usage_history.pop_front();
        }
        if let Some(budget) = budget_mb.filter(|budget| *budget > 0.0) {
            self.budget_manager.memory_budget_mb = budget;
            let pressure = usage_mb / budget;
            if pressure > 1.0 {
                self.budget_manager.budget_violations += 1;
            }
            self.memory_tracker.memory_pressure = Some(pressure);
        } else {
            self.memory_tracker.memory_pressure = None;
        }
    }

    /// Record measured energy consumption for the most recent step.
    pub(crate) fn record_energy(&mut self, joules: f64) {
        self.energy_tracker.energy_per_step.push_back(joules);
        while self.energy_tracker.energy_per_step.len() > MAX_METRIC_HISTORY {
            self.energy_tracker.energy_per_step.pop_front();
        }
        self.energy_tracker.cumulative_energy += joules;
        let average = mean(
            &self
                .energy_tracker
                .energy_per_step
                .iter()
                .copied()
                .collect::<Vec<f64>>(),
        );
        self.energy_tracker.energy_efficiency = if average > 0.0 {
            Some(1.0 / average)
        } else {
            None
        };
    }

    /// Record the observed sample throughput and the requirement it must meet.
    pub(crate) fn record_throughput(&mut self, samples_per_second: A) {
        self.throughput_requirements.current_throughput = samples_per_second;
        self.refresh_throughput_deficit();
    }

    /// Set the minimum throughput the deployment has to satisfy.
    ///
    /// Only the minimum is modelled: `throughput_deficit` — the one quantity the
    /// resource-aware adapter acts on — is measured against it, and the separate
    /// "target" the signature used to take had no reader anywhere.
    pub fn set_throughput_requirement(&mut self, minimum: A) {
        self.throughput_requirements.min_samples_per_second = minimum;
        self.refresh_throughput_deficit();
    }

    fn refresh_throughput_deficit(&mut self) {
        let minimum = from_scalar(self.throughput_requirements.min_samples_per_second);
        let current = from_scalar(self.throughput_requirements.current_throughput);
        self.throughput_requirements.throughput_deficit = to_scalar((minimum - current).max(0.0));
    }

    pub(crate) fn generate_signal(&mut self, _step: usize) -> Result<SignalVote<A>> {
        // E4: only measurements that actually exist take part. Previously the
        // never-written `memory_pressure` of 0.0 was read as "plenty of spare
        // memory" and pushed the learning rate up on every step.
        let mut terms: Vec<(f64, String)> = Vec::new();

        if let Some(pressure) = self.memory_tracker.memory_pressure {
            let recommended = if pressure > 0.8 {
                0.9
            } else if pressure < 0.3 {
                1.05
            } else {
                1.0
            };
            terms.push((recommended, format!("memory pressure {pressure:.2}")));
        }

        if let Some(pressure) = self.compute_tracker.time_pressure {
            // Over the step budget: a smaller learning rate does not make steps
            // faster, but it lets the deployment take more of them before
            // diverging, which is the trade this signal is for.
            let recommended = if pressure > 1.0 {
                0.95
            } else if pressure < 0.5 {
                1.02
            } else {
                1.0
            };
            terms.push((recommended, format!("time pressure {pressure:.2}")));
        }

        let deficit = from_scalar(self.throughput_requirements.throughput_deficit);
        if deficit > 0.0 {
            terms.push((0.95, format!("throughput deficit {deficit:.2}/s")));
        }

        if terms.is_empty() {
            return Err(crate::error::OptimError::InvalidState(
                "no resource measurements have been reported, so no resource signal can be \
                 produced"
                    .to_string(),
            ));
        }

        let recommended = mean(&terms.iter().map(|(value, _)| *value).collect::<Vec<f64>>());
        // Confidence grows with the number of independent measurements backing
        // the vote (three is the maximum this adapter can observe).
        let confidence = (terms.len() as f64 / 3.0).clamp(0.0, 1.0);

        Ok(SignalVote {
            signal_type: AdaptationSignalType::ResourceUtilization,
            recommended_lr_change: to_scalar(recommended),
            confidence: to_scalar(confidence),
            reasoning: terms
                .iter()
                .map(|(_, reason)| reason.clone())
                .collect::<Vec<String>>()
                .join(", "),
            timestamp: Instant::now(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.memory_tracker = MemoryUsageTracker::default();
        self.compute_tracker = ComputationTimeTracker {
            time_budget: self.compute_tracker.time_budget,
            ..ComputationTimeTracker::default()
        };
        self.energy_tracker = EnergyConsumptionTracker::default();
        self.budget_manager.budget_violations = 0;
        self.budget_manager.budget_utilization = A::zero();
    }

    /// Budget violations observed so far.
    pub(crate) fn budget_violations(&self) -> usize {
        self.budget_manager.budget_violations
    }
}
