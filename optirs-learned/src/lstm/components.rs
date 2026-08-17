//! Real implementations of the LSTM optimizer's auxiliary components.
//!
//! This file replaces the block that used to sit at the bottom of `lstm.rs`
//! under the comment *"Placeholder implementations for remaining complex
//! components / These would be fully implemented in a production system"*
//! (findings F6/F7). Concretely, what was there before:
//!
//! | component | old behaviour |
//! |---|---|
//! | `MetaLearner::step` | `Ok(T::zero())` — nothing trained |
//! | `TransferLearner::transfer_to_domain` | all-zero `TransferResults` |
//! | `AdaptiveLearningRateController::compute_lr` | returned `self.current_lr` unchanged, ignoring gradients, loss and history |
//! | `OptimizationStateTracker::update` | empty body |
//!
//! Every one of them now computes something from its inputs. `MetaLearner::step`
//! delegates to the truncated-BPTT meta-trainer in [`super::bptt`], which is
//! where the actual learning happens.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::bptt::{BpttConfig, DiagonalQuadraticTask, MetaTrainer};
use super::{
    AdaptiveLearningRateController, ConvergenceIndicators, DomainSimilarityEstimator,
    FlowStability, GradientAnalyzer, GradientCorrelationTracker, GradientFlowAnalyzer,
    GradientNoiseEstimator, GradientStatistics, HistoryBuffer, InnerLoopState, LRAdaptationParams,
    LSTMNetwork, LossLandscapeAnalyzer, MetaLearner, MetaLearningState, MetaTask, NoiseCharacteristics,
    NoiseType, OptimizationPhase, OptimizationStateTracker, PerformanceTrend, PerformanceTracker,
    SimilarityFunction, StabilityMetrics, TransferLearner, TransferMetrics, TransferResults,
};
use crate::error::{OptimError, Result};
use crate::{LearnedOptimizerConfig, MetaOptimizationStrategy};

/// How many recent losses the LR controller keeps for its trend estimate.
const LR_TREND_WINDOW: usize = 8;

/// How many samples the state tracker keeps in each trend vector.
const STATE_TREND_WINDOW: usize = 32;

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> MetaLearner<T> {
    pub(super) fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        let meta_lr: T =
            scirs2_core::numeric::NumCast::from(config.meta_learning_rate).unwrap_or_else(T::zero);
        Ok(Self {
            strategy: MetaOptimizationStrategy::MAML,
            meta_parameters: HashMap::new(),
            meta_gradients: HashMap::new(),
            task_history: VecDeque::new(),
            meta_state: MetaLearningState {
                meta_step: 0,
                meta_lr,
                adaptation_rate: scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::zero()),
                meta_validation_performance: T::zero(),
                adaptation_history: VecDeque::new(),
                inner_loop_state: InnerLoopState {
                    inner_step: 0,
                    inner_parameters: Array1::zeros(1),
                    inner_optimizer_state: HashMap::new(),
                    inner_performance: T::zero(),
                },
            },
            transfer_learner: TransferLearner {
                source_knowledge: HashMap::new(),
                adaptation_parameters: Array1::zeros(1),
                transfer_metrics: TransferMetrics {
                    efficiency: T::zero(),
                    adaptation_speed: T::zero(),
                    knowledge_retention: T::zero(),
                    negative_transfer_score: T::zero(),
                },
                similarity_estimator: DomainSimilarityEstimator {
                    domain_embeddings: HashMap::new(),
                    similarity_params: Array1::zeros(1),
                    similarity_function: SimilarityFunction::Cosine,
                },
            },
        })
    }

    /// Run one meta-training step over `tasks`, updating `network`'s weights.
    ///
    /// Each `MetaTask`'s recorded trajectory is turned into a differentiable
    /// diagonal-quadratic surrogate
    /// ([`DiagonalQuadraticTask::from_trajectory`], which identifies the
    /// curvature and optimum by per-coordinate least squares on the observed
    /// `(parameters, gradient)` pairs). The LSTM controller is then unrolled on
    /// those surrogates and its weights are updated by truncated
    /// backpropagation through time — see [`MetaTrainer::meta_step`].
    ///
    /// Returns the mean meta-loss (the summed surrogate loss over the unrolled
    /// horizon, averaged over tasks). Previously this returned `T::zero()` and
    /// trained nothing.
    ///
    /// # Errors
    /// Returns `Err` when `tasks` is empty, when no task has a trajectory long
    /// enough to identify a surrogate, or when the surrogates disagree on
    /// dimension.
    pub(super) fn step(
        &mut self,
        tasks: &[MetaTask<T>],
        network: &mut LSTMNetwork<T>,
    ) -> Result<T> {
        if tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "meta-learning step requires at least one task".to_string(),
            ));
        }

        let mut surrogates: Vec<DiagonalQuadraticTask<T>> = Vec::with_capacity(tasks.len());
        let mut weights: Vec<T> = Vec::with_capacity(tasks.len());
        let mut rejected = 0usize;
        for task in tasks {
            match DiagonalQuadraticTask::from_trajectory(&task.training_trajectory) {
                Ok(surrogate) => {
                    surrogates.push(surrogate);
                    weights.push(if task.weight > T::zero() {
                        task.weight
                    } else {
                        T::one()
                    });
                }
                Err(_) => rejected += 1,
            }
        }

        if surrogates.is_empty() {
            return Err(OptimError::InsufficientData(format!(
                "none of the {} supplied tasks had a trajectory from which a \
                 differentiable surrogate could be identified ({rejected} rejected); \
                 each task needs at least two trajectory points with varying parameters",
                tasks.len()
            )));
        }

        let unroll = self
            .meta_state
            .adaptation_history
            .len()
            .clamp(8, 32)
            .max(8);
        let bptt_config = BpttConfig {
            unroll_steps: unroll,
            meta_learning_rate: self.meta_state.meta_lr.to_f64().unwrap_or(1e-3),
            gradient_clip: 1.0,
        };
        let mut trainer = MetaTrainer::new(bptt_config);
        let meta_loss = trainer.meta_step(network, &surrogates, &weights)?;

        // Record what was learned so the state is no longer write-only.
        self.meta_state.meta_step += 1;
        self.meta_state.meta_validation_performance = meta_loss;
        for task in tasks.iter().take(16) {
            self.task_history.push_back(task.clone());
            if self.task_history.len() > 64 {
                self.task_history.pop_front();
            }
        }
        self.meta_parameters.insert(
            "flattened_controller".to_string(),
            Array1::from_vec(super::bptt::flatten_parameters(network)),
        );
        self.meta_gradients.insert(
            "last_meta_gradient".to_string(),
            Array1::from_vec(trainer.last_gradient_vector()),
        );

        Ok(meta_loss)
    }

    /// Number of meta-training steps performed so far.
    pub fn meta_step_count(&self) -> usize {
        self.meta_state.meta_step
    }

    /// Most recent meta-loss.
    pub fn last_meta_loss(&self) -> T {
        self.meta_state.meta_validation_performance
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> TransferLearner<T> {
    /// Adapt an already-meta-trained controller to a new task domain.
    ///
    /// This is a real measurement, not the all-zero struct it used to return:
    ///
    /// 1. Evaluate the controller on the target domain's surrogates *before*
    ///    adapting (`initial_performance` = mean unrolled loss).
    /// 2. Run `adaptation_steps` BPTT meta-steps restricted to the target
    ///    surrogates.
    /// 3. Re-evaluate (`final_performance`).
    /// 4. `transfer_efficiency` = relative loss reduction
    ///    `(initial - final) / initial`, clamped to `[-1, 1]`; negative values
    ///    are recorded as a `negative_transfer_score` rather than hidden.
    ///
    /// # Errors
    /// Returns `Err` when no target task yields an identifiable surrogate.
    pub(super) fn transfer_to_domain(
        &mut self,
        target_tasks: &[MetaTask<T>],
        network: &mut LSTMNetwork<T>,
    ) -> Result<TransferResults<T>> {
        if target_tasks.is_empty() {
            return Err(OptimError::InsufficientData(
                "transfer requires at least one target task".to_string(),
            ));
        }

        let mut surrogates = Vec::new();
        let mut weights = Vec::new();
        for task in target_tasks {
            if let Ok(s) = DiagonalQuadraticTask::from_trajectory(&task.training_trajectory) {
                surrogates.push(s);
                weights.push(if task.weight > T::zero() {
                    task.weight
                } else {
                    T::one()
                });
            }
        }
        if surrogates.is_empty() {
            return Err(OptimError::InsufficientData(
                "no target task had a trajectory from which a surrogate could be identified"
                    .to_string(),
            ));
        }

        let config = BpttConfig {
            unroll_steps: 12,
            meta_learning_rate: 1e-2,
            gradient_clip: 1.0,
        };
        let mut trainer = MetaTrainer::new(config);

        let initial_performance = trainer.evaluate(network, &surrogates, &weights)?;
        let adaptation_steps = 8usize;
        for _ in 0..adaptation_steps {
            trainer.meta_step(network, &surrogates, &weights)?;
        }
        let final_performance = trainer.evaluate(network, &surrogates, &weights)?;

        let efficiency = if initial_performance > T::zero() {
            let ratio = (initial_performance - final_performance) / initial_performance;
            let one = T::one();
            if ratio > one {
                one
            } else if ratio < -one {
                -one
            } else {
                ratio
            }
        } else {
            T::zero()
        };

        self.transfer_metrics.efficiency = efficiency;
        self.transfer_metrics.adaptation_speed = efficiency
            / scirs2_core::numeric::NumCast::from(adaptation_steps.max(1)).unwrap_or_else(T::one);
        self.transfer_metrics.knowledge_retention = if initial_performance > T::zero() {
            let retained = final_performance / initial_performance;
            if retained > T::one() {
                T::zero()
            } else {
                T::one() - retained
            }
        } else {
            T::zero()
        };
        self.transfer_metrics.negative_transfer_score = if efficiency < T::zero() {
            -efficiency
        } else {
            T::zero()
        };

        Ok(TransferResults {
            initial_performance,
            final_performance,
            adaptation_steps,
            transfer_efficiency: efficiency,
        })
    }

    /// Metrics recorded by the most recent [`Self::transfer_to_domain`] call.
    pub fn metrics(&self) -> &TransferMetrics<T> {
        &self.transfer_metrics
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> AdaptiveLearningRateController<T> {
    pub(super) fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        let base_lr: T =
            scirs2_core::numeric::NumCast::from(config.learning_rate).unwrap_or_else(T::zero);
        if base_lr <= T::zero() {
            return Err(OptimError::InvalidConfig(
                "learning_rate must be positive".to_string(),
            ));
        }
        let momentum: T = scirs2_core::numeric::NumCast::from(if config.use_momentum {
            config.momentum_decay
        } else {
            0.0
        })
        .unwrap_or_else(T::zero);

        Ok(Self {
            base_lr,
            current_lr: base_lr,
            adaptation_params: LRAdaptationParams {
                momentum,
                gradient_sensitivity: scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::zero()),
                loss_sensitivity: scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::zero()),
                // Bound the schedule to two orders of magnitude either side of
                // the configured base rate instead of fixed 1e-6 / 0.1 constants.
                min_lr: base_lr / scirs2_core::numeric::NumCast::from(100.0).unwrap_or_else(T::one),
                max_lr: base_lr * scirs2_core::numeric::NumCast::from(100.0).unwrap_or_else(T::one),
                adaptation_rate: scirs2_core::numeric::NumCast::from(0.01)
                    .unwrap_or_else(|| T::zero()),
            },
            lr_history: VecDeque::new(),
            performance_tracker: PerformanceTracker {
                recent_losses: VecDeque::new(),
                trend: PerformanceTrend::Unknown,
                stagnation_counter: 0,
                best_performance: T::infinity(),
                improvement_rate: T::zero(),
            },
            schedule_params: None,
        })
    }

    /// Compute the learning rate for the next step from live signals.
    ///
    /// Three multiplicative factors, all derived from the arguments:
    ///
    /// * **Gradient-norm normalization** — `1 / (1 + s·‖g‖)` with
    ///   `s = gradient_sensitivity`. A large gradient shrinks the step, which is
    ///   the standard defence against the divergence a fixed rate invites.
    /// * **Loss trend** — the sign of the recent loss slope. A falling loss
    ///   grows the rate by `1 + adaptation_rate`, a rising one shrinks it by
    ///   `1 - adaptation_rate`.
    /// * **Stagnation escape** — after `LR_TREND_WINDOW` consecutive steps with
    ///   no improvement on the best loss seen, the rate is boosted by
    ///   `1 + loss_sensitivity` to escape a plateau.
    ///
    /// The result is smoothed with `momentum` against the previous rate and
    /// clamped to `[min_lr, max_lr]`. Previously this returned `self.current_lr`
    /// unchanged and read none of its arguments.
    ///
    /// # Errors
    /// Returns `Err` when `gradients` is empty — there is no signal to adapt to.
    pub(super) fn compute_lr(
        &mut self,
        gradients: &Array1<T>,
        loss: Option<T>,
        history: &HistoryBuffer<T>,
    ) -> Result<T> {
        if gradients.is_empty() {
            return Err(OptimError::InsufficientData(
                "cannot compute an adaptive learning rate from an empty gradient".to_string(),
            ));
        }

        let grad_norm = gradients
            .iter()
            .map(|&g| g * g)
            .fold(T::zero(), |a, b| a + b)
            .sqrt();

        // --- gradient factor ------------------------------------------------
        let grad_factor =
            T::one() / (T::one() + self.adaptation_params.gradient_sensitivity * grad_norm);

        // --- loss trend factor ----------------------------------------------
        if let Some(l) = loss {
            self.performance_tracker.recent_losses.push_back(l);
            while self.performance_tracker.recent_losses.len() > LR_TREND_WINDOW {
                self.performance_tracker.recent_losses.pop_front();
            }
            if l < self.performance_tracker.best_performance {
                self.performance_tracker.best_performance = l;
                self.performance_tracker.stagnation_counter = 0;
            } else {
                self.performance_tracker.stagnation_counter += 1;
            }
        }

        // Prefer the controller's own window; fall back to the shared history
        // buffer when the caller passes no per-step loss.
        let losses: Vec<T> = if self.performance_tracker.recent_losses.len() >= 2 {
            self.performance_tracker.recent_losses.iter().copied().collect()
        } else {
            history
                .losses
                .iter()
                .rev()
                .take(LR_TREND_WINDOW)
                .rev()
                .copied()
                .collect()
        };

        let (trend_factor, improvement_rate) = if losses.len() >= 2 {
            let first = losses[0];
            let last = losses[losses.len() - 1];
            let rate = if first.abs() > T::zero() {
                (first - last) / first.abs()
            } else {
                T::zero()
            };
            let factor = if last < first {
                T::one() + self.adaptation_params.adaptation_rate
            } else if last > first {
                T::one() - self.adaptation_params.adaptation_rate
            } else {
                T::one()
            };
            (factor, rate)
        } else {
            (T::one(), T::zero())
        };
        self.performance_tracker.improvement_rate = improvement_rate;
        self.performance_tracker.trend = if losses.len() < 2 {
            PerformanceTrend::Unknown
        } else if improvement_rate > T::zero() {
            PerformanceTrend::Improving
        } else if improvement_rate < T::zero() {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        };

        // --- stagnation escape ----------------------------------------------
        let stagnation_factor = if self.performance_tracker.stagnation_counter >= LR_TREND_WINDOW {
            T::one() + self.adaptation_params.loss_sensitivity
        } else {
            T::one()
        };

        let target = self.base_lr * grad_factor * trend_factor * stagnation_factor;

        // --- momentum smoothing + clamping ----------------------------------
        let m = self.adaptation_params.momentum;
        let smoothed = m * self.current_lr + (T::one() - m) * target;
        let clamped = if smoothed < self.adaptation_params.min_lr {
            self.adaptation_params.min_lr
        } else if smoothed > self.adaptation_params.max_lr {
            self.adaptation_params.max_lr
        } else {
            smoothed
        };

        self.current_lr = clamped;
        self.lr_history.push_back(clamped);
        while self.lr_history.len() > 1000 {
            self.lr_history.pop_front();
        }
        Ok(clamped)
    }

    /// Current learning rate.
    pub fn current_lr(&self) -> T {
        self.current_lr
    }

    /// Observed performance trend.
    pub fn trend(&self) -> PerformanceTrend {
        self.performance_tracker.trend
    }

    /// Consecutive steps without a new best loss.
    pub fn stagnation_counter(&self) -> usize {
        self.performance_tracker.stagnation_counter
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> OptimizationStateTracker<T> {
    pub(super) fn new() -> Self {
        Self {
            phase: OptimizationPhase::InitialDescent,
            convergence_indicators: ConvergenceIndicators {
                gradient_norm_trend: Vec::new(),
                loss_change_trend: Vec::new(),
                parameter_change_magnitude: T::zero(),
                convergence_probability: T::zero(),
                estimated_steps_to_convergence: None,
            },
            gradient_analyzer: GradientAnalyzer {
                gradient_stats: GradientStatistics {
                    mean_norm: T::zero(),
                    norm_variance: T::zero(),
                    direction_consistency: T::zero(),
                    magnitude_distribution: Vec::new(),
                    component_stats: Array1::zeros(1),
                },
                correlation_tracker: GradientCorrelationTracker {
                    correlation_matrix: Array2::zeros((1, 1)),
                    temporal_correlations: VecDeque::new(),
                    cross_correlations: HashMap::new(),
                },
                noise_estimator: GradientNoiseEstimator {
                    noise_level: T::zero(),
                    signal_to_noise_ratio: T::zero(),
                    noise_characteristics: NoiseCharacteristics {
                        noise_type: NoiseType::White,
                        scale: T::zero(),
                        temporal_correlation: T::zero(),
                        spatial_correlation: T::zero(),
                    },
                },
                flow_analyzer: GradientFlowAnalyzer {
                    flow_field: Array2::zeros((1, 1)),
                    critical_points: Vec::new(),
                    stability: FlowStability::Unknown,
                    attractors: Vec::new(),
                    repellers: Vec::new(),
                },
            },
            landscape_analyzer: LossLandscapeAnalyzer {
                local_curvature: T::zero(),
                hessian_eigenvalues: None,
                roughness: T::zero(),
                basin_size: T::zero(),
                barrier_heights: Vec::new(),
            },
            stability_metrics: StabilityMetrics {
                lyapunov_exponents: Array1::zeros(1),
                stability_margin: T::zero(),
                perturbation_sensitivity: T::zero(),
                robustness_score: T::zero(),
            },
            previous_gradient: None,
            previous_loss: None,
            step_count: 0,
        }
    }

    /// Update every tracked statistic from the observed step.
    ///
    /// The body used to be empty (`// Placeholder state update`), so
    /// `LSTMOptimizer::get_state_analysis` reported the all-zero struct built by
    /// [`Self::new`] forever. Now, per step:
    ///
    /// * `gradient_norm_trend` / `loss_change_trend` get one sample each
    ///   (bounded to `STATE_TREND_WINDOW`).
    /// * `mean_norm` and `norm_variance` are Welford running moments of `‖g‖`.
    /// * `direction_consistency` is the cosine similarity with the previous
    ///   gradient.
    /// * `noise_level` is the norm of the gradient *difference* (a
    ///   finite-difference noise proxy) and `signal_to_noise_ratio` is
    ///   `‖g‖ / noise_level`.
    /// * `local_curvature` is the secant estimate
    ///   `‖g_t − g_{t−1}‖ / ‖Δθ‖` — the directional curvature along the step
    ///   actually taken.
    /// * `parameter_change_magnitude` is `‖Δθ‖`.
    /// * `convergence_probability` is `1 / (1 + ‖g‖)`, and `phase` is derived
    ///   from the gradient-norm trend.
    pub(super) fn update(&mut self, gradients: &Array1<T>, updates: &Array1<T>, loss: Option<T>) {
        if gradients.is_empty() {
            return;
        }
        self.step_count += 1;
        let n_t: T =
            scirs2_core::numeric::NumCast::from(self.step_count).unwrap_or_else(T::one);

        let grad_norm = gradients
            .iter()
            .map(|&g| g * g)
            .fold(T::zero(), |a, b| a + b)
            .sqrt();
        let update_norm = updates
            .iter()
            .map(|&u| u * u)
            .fold(T::zero(), |a, b| a + b)
            .sqrt();

        // --- trends ----------------------------------------------------------
        self.convergence_indicators
            .gradient_norm_trend
            .push(grad_norm);
        if self.convergence_indicators.gradient_norm_trend.len() > STATE_TREND_WINDOW {
            self.convergence_indicators.gradient_norm_trend.remove(0);
        }
        if let (Some(current), Some(previous)) = (loss, self.previous_loss) {
            self.convergence_indicators
                .loss_change_trend
                .push(current - previous);
            if self.convergence_indicators.loss_change_trend.len() > STATE_TREND_WINDOW {
                self.convergence_indicators.loss_change_trend.remove(0);
            }
        }
        self.convergence_indicators.parameter_change_magnitude = update_norm;
        self.convergence_indicators.convergence_probability = T::one() / (T::one() + grad_norm);

        // --- Welford moments of the gradient norm ---------------------------
        let delta = grad_norm - self.gradient_analyzer.gradient_stats.mean_norm;
        self.gradient_analyzer.gradient_stats.mean_norm =
            self.gradient_analyzer.gradient_stats.mean_norm + delta / n_t;
        let delta2 = grad_norm - self.gradient_analyzer.gradient_stats.mean_norm;
        // Running population variance: M2/n, accumulated in `norm_variance`.
        let prev_m2 = self.gradient_analyzer.gradient_stats.norm_variance * (n_t - T::one());
        let m2 = prev_m2 + delta * delta2;
        self.gradient_analyzer.gradient_stats.norm_variance = m2 / n_t;

        self.gradient_analyzer
            .gradient_stats
            .magnitude_distribution
            .push(grad_norm);
        if self
            .gradient_analyzer
            .gradient_stats
            .magnitude_distribution
            .len()
            > STATE_TREND_WINDOW
        {
            self.gradient_analyzer
                .gradient_stats
                .magnitude_distribution
                .remove(0);
        }
        self.gradient_analyzer.gradient_stats.component_stats = gradients.mapv(|g| g.abs());

        // --- direction consistency, noise, curvature ------------------------
        if let Some(previous) = &self.previous_gradient {
            let width = previous.len().min(gradients.len());
            if width > 0 {
                let mut dot = T::zero();
                let mut prev_sq = T::zero();
                let mut diff_sq = T::zero();
                for i in 0..width {
                    let p = previous[i];
                    let g = gradients[i];
                    dot = dot + p * g;
                    prev_sq = prev_sq + p * p;
                    diff_sq = diff_sq + (g - p) * (g - p);
                }
                let prev_norm = prev_sq.sqrt();
                if prev_norm > T::zero() && grad_norm > T::zero() {
                    self.gradient_analyzer.gradient_stats.direction_consistency =
                        dot / (prev_norm * grad_norm);
                }
                let noise = diff_sq.sqrt();
                self.gradient_analyzer.noise_estimator.noise_level = noise;
                self.gradient_analyzer.noise_estimator.signal_to_noise_ratio = if noise > T::zero() {
                    grad_norm / noise
                } else {
                    T::infinity()
                };
                self.gradient_analyzer
                    .noise_estimator
                    .noise_characteristics
                    .scale = noise;
                self.gradient_analyzer
                    .noise_estimator
                    .noise_characteristics
                    .temporal_correlation =
                    self.gradient_analyzer.gradient_stats.direction_consistency;

                // Secant curvature along the step actually taken.
                if update_norm > T::zero() {
                    self.landscape_analyzer.local_curvature = noise / update_norm;
                }
            }
        }

        // --- roughness & stability -------------------------------------------
        let trend = &self.convergence_indicators.gradient_norm_trend;
        if trend.len() >= 3 {
            let mut second_diff = T::zero();
            for w in trend.windows(3) {
                second_diff = second_diff + (w[2] - w[1] - (w[1] - w[0])).abs();
            }
            let count: T = scirs2_core::numeric::NumCast::from(trend.len() - 2)
                .unwrap_or_else(T::one);
            let mean_second = second_diff / count;
            self.landscape_analyzer.roughness = mean_second / (T::one() + mean_second);
        }
        self.stability_metrics.perturbation_sensitivity =
            self.gradient_analyzer.noise_estimator.noise_level;
        self.stability_metrics.stability_margin =
            T::one() / (T::one() + self.landscape_analyzer.local_curvature);
        self.stability_metrics.robustness_score =
            (T::one() + self.gradient_analyzer.gradient_stats.direction_consistency)
                / scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(T::one);

        // --- phase ------------------------------------------------------------
        self.phase = Self::classify_phase(
            trend,
            self.gradient_analyzer.gradient_stats.direction_consistency,
        );

        self.previous_gradient = Some(gradients.clone());
        if let Some(l) = loss {
            self.previous_loss = Some(l);
        }
    }

    /// Classify the optimization phase from the gradient-norm trend.
    ///
    /// * fewer than 4 samples → `InitialDescent`
    /// * norm shrinking fast (last < 25% of first) → `Convergence`
    /// * norm roughly flat and direction consistency low → `Oscillation`
    /// * norm roughly flat and direction consistency high → `Plateau`
    /// * norm growing → `Divergence`
    /// * otherwise → `SteadyProgress`
    fn classify_phase(trend: &[T], consistency: T) -> OptimizationPhase {
        if trend.len() < 4 {
            return OptimizationPhase::InitialDescent;
        }
        let first = trend[0];
        let last = trend[trend.len() - 1];
        if first <= T::zero() {
            return OptimizationPhase::Convergence;
        }
        let ratio = (last / first).to_f64().unwrap_or(1.0);
        let low_consistency = consistency.to_f64().unwrap_or(1.0) < 0.0;
        if ratio < 0.25 {
            OptimizationPhase::Convergence
        } else if ratio > 1.5 {
            OptimizationPhase::Divergence
        } else if ratio > 0.9 {
            if low_consistency {
                OptimizationPhase::Oscillation
            } else {
                OptimizationPhase::Plateau
            }
        } else {
            OptimizationPhase::SteadyProgress
        }
    }

    /// Number of steps observed.
    pub fn observed_steps(&self) -> usize {
        self.step_count
    }
}
