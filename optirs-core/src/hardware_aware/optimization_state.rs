// Real optimizer step path for the hardware-aware optimizer.
//
// Before this module existed, `OptimizationState` tracked a parameter array and
// nothing else: the hardware analysis in [`super`] recommended batch sizes,
// precision and memory strategies but never actually ran an optimization step,
// so a `HardwareAwareOptimizer` could not optimize anything. The state now owns
// a real optimizer behind the crate's [`Optimizer`] trait, an explicit learning
// rate schedule, a step counter and the gradient accumulator that
// [`MemoryStrategy::GradientAccumulation`] configures.

use std::fmt;

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;

use super::{HardwareOptimizationConfig, HardwarePlatform, MemoryStrategy, QuantizationSupport};
use crate::error::{OptimError, Result};
use crate::optimizers::{Adam, Lion, Optimizer, LAMB, SGD};
use crate::schedulers::{ConstantScheduler, LearningRateScheduler};
use crate::utils::scalar_or;

/// Learning rate used when the caller does not supply one.
///
/// `1e-3` is the value every adaptive optimizer in this crate uses as its own
/// default, so a hardware-aware optimizer built with
/// [`super::HardwareAwareOptimizer::new`] behaves like a directly constructed
/// `Adam`/`Lion`/`LAMB` until the caller says otherwise.
pub const DEFAULT_BASE_LEARNING_RATE: f64 = 1e-3;

/// Batch size at or above which a large-batch optimizer is recommended.
///
/// LAMB was introduced precisely because Adam's update stops scaling past a few
/// hundred samples per step (You et al., "Large Batch Optimization for Deep
/// Learning: Training BERT in 76 minutes", arXiv:1904.00962); the TPU and
/// distributed configurations in [`super`] routinely produce batches this large.
const LARGE_BATCH_THRESHOLD: usize = 512;

/// Power budget (watts) below which optimizer state, not compute, is the binding
/// constraint on an edge device.
const LOW_POWER_BUDGET_WATTS: f64 = 5.0;

/// Optimizer families the hardware analysis can recommend.
///
/// The distinguishing property is how much *per-parameter optimizer state* each
/// family needs, because on the memory-constrained platforms this module models
/// that state — not the parameters themselves — is what decides whether a model
/// fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareOptimizerKind {
    /// Momentum SGD: one auxiliary buffer per parameter, and none at all when
    /// momentum is zero. Recommended when optimizer state has to be paid for
    /// over a slow link (CPU offloading) or under a very small power budget.
    Sgd,
    /// Lion: a single momentum buffer and a sign-based update
    /// (Chen et al., "Symbolic Discovery of Optimization Algorithms",
    /// arXiv:2302.06675). Half of Adam's optimizer state, and the sign update
    /// is insensitive to gradient scale, which suits the quantized/low-precision
    /// arithmetic edge devices use.
    Lion,
    /// Adam: first and second moments, two buffers per parameter. The default
    /// when optimizer-state memory is not the binding constraint.
    Adam,
    /// LAMB: Adam's moments plus a layer-wise trust ratio, for the very large
    /// batches TPU and distributed configurations produce.
    Lamb,
}

impl HardwareOptimizerKind {
    /// Number of per-parameter auxiliary buffers this family keeps.
    ///
    /// Multiply by the parameter count and the element size to get the
    /// optimizer-state footprint that has to fit alongside the model.
    pub fn state_buffers_per_parameter(self) -> usize {
        match self {
            Self::Sgd => 1,
            Self::Lion => 1,
            Self::Adam => 2,
            Self::Lamb => 2,
        }
    }

    /// Stable short name, for diagnostics and reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sgd => "sgd",
            Self::Lion => "lion",
            Self::Adam => "adam",
            Self::Lamb => "lamb",
        }
    }

    /// Recommend an optimizer family for a platform and the configuration the
    /// hardware analysis produced for it.
    ///
    /// The rules are ordered from the hardest constraint to the softest:
    ///
    /// 1. Offloading optimizer state to CPU memory, or a sub-`LOW_POWER_BUDGET_WATTS`
    ///    power budget, makes every extra per-parameter buffer expensive: use SGD.
    /// 2. Any other edge device: use Lion, which halves Adam's state and whose
    ///    sign update tolerates quantized arithmetic.
    /// 3. Batches of `LARGE_BATCH_THRESHOLD` or more (TPU, distributed): use LAMB.
    /// 4. Otherwise: Adam.
    pub fn recommend_for<A: Float>(
        platform: &HardwarePlatform,
        config: &HardwareOptimizationConfig<A>,
    ) -> Self {
        let offloading = matches!(config.memory_strategy, MemoryStrategy::CPUOffloading { .. });

        match platform {
            HardwarePlatform::Edge {
                power_budget,
                quantization_support,
                ..
            } => {
                if offloading || *power_budget < LOW_POWER_BUDGET_WATTS {
                    Self::Sgd
                } else if matches!(
                    quantization_support,
                    QuantizationSupport::Int4 | QuantizationSupport::Int8
                ) {
                    Self::Lion
                } else if config.batch_size >= LARGE_BATCH_THRESHOLD {
                    Self::Lamb
                } else {
                    Self::Adam
                }
            }
            _ => {
                if offloading {
                    Self::Sgd
                } else if config.batch_size >= LARGE_BATCH_THRESHOLD {
                    Self::Lamb
                } else {
                    Self::Adam
                }
            }
        }
    }
}

/// Outcome of a single call to [`OptimizationState::step`].
///
/// `applied` is `false` while gradient accumulation is still filling its window:
/// the gradient was recorded but the parameters did not move, which is exactly
/// what [`MemoryStrategy::GradientAccumulation`] asks for and what a caller
/// driving a training loop needs to know.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareStepReport<A: Float> {
    /// Whether an optimizer update was applied to the parameters.
    pub applied: bool,
    /// Learning rate the schedule supplied for this update. While accumulating
    /// this is the rate the *next* update will use.
    pub learning_rate: A,
    /// Number of optimizer updates applied so far.
    pub step_count: usize,
    /// Micro-batches currently held in the gradient accumulator.
    pub accumulated_micro_steps: usize,
}

/// Current optimization state: parameters plus everything needed to move them.
pub struct OptimizationState<A: Float + 'static, D: Dimension + 'static> {
    /// Current parameters.
    parameters: Array<A, D>,
    /// The optimizer that actually performs the update.
    optimizer: Box<dyn Optimizer<A, D> + Send + Sync>,
    /// Which family `optimizer` belongs to.
    optimizer_kind: HardwareOptimizerKind,
    /// Learning rate schedule driving `optimizer` between updates.
    lr_schedule: Box<dyn LearningRateScheduler<A> + Send + Sync>,
    /// The rate this state was constructed with. Reported by
    /// [`OptimizationState::base_learning_rate`] so a caller can tell how far a
    /// schedule has moved; the rate actually in effect always comes from
    /// `lr_schedule`, never from here.
    base_learning_rate: A,
    /// Number of optimizer updates applied.
    step_count: usize,
    /// Micro-batches folded into `gradient_accumulator` since the last update.
    accumulated_micro_steps: usize,
    /// Micro-batches per optimizer update (1 = update on every gradient).
    accumulation_steps: usize,
    /// Running sum of the gradients of the current accumulation window.
    /// Allocated lazily so a state that never accumulates costs nothing.
    gradient_accumulator: Option<Array<A, D>>,
}

impl<A, D> fmt::Debug for OptimizationState<A, D>
where
    A: Float + fmt::Debug + 'static,
    D: Dimension + 'static,
{
    // Hand-written because neither the boxed optimizer nor the boxed schedule is
    // `Debug`; the fields that matter for diagnostics are printed by value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptimizationState")
            .field("parameter_count", &self.parameters.len())
            .field("optimizer_kind", &self.optimizer_kind)
            .field("base_learning_rate", &self.base_learning_rate)
            .field("step_count", &self.step_count)
            .field("accumulation_steps", &self.accumulation_steps)
            .field("accumulated_micro_steps", &self.accumulated_micro_steps)
            .finish()
    }
}

impl<A, D> OptimizationState<A, D>
where
    A: Float + ScalarOperand + std::fmt::Debug + Send + Sync + 'static,
    D: Dimension + 'static,
{
    /// Build a state around `parameters`, using the optimizer family `kind`.
    ///
    /// `accumulation_steps` is clamped to at least 1: zero micro-batches per
    /// update would mean the parameters never move.
    pub fn new(
        parameters: Array<A, D>,
        kind: HardwareOptimizerKind,
        base_learning_rate: A,
        accumulation_steps: usize,
    ) -> Self {
        Self {
            parameters,
            optimizer: build_optimizer(kind, base_learning_rate),
            optimizer_kind: kind,
            lr_schedule: Box::new(ConstantScheduler::new(base_learning_rate)),
            base_learning_rate,
            step_count: 0,
            accumulated_micro_steps: 0,
            accumulation_steps: accumulation_steps.max(1),
            gradient_accumulator: None,
        }
    }

    /// Apply one gradient.
    ///
    /// With `accumulation_steps == 1` this performs an optimizer update
    /// immediately. Otherwise the gradient is summed into the accumulator and
    /// the update happens once the window is full, using the *mean* of the
    /// window so the effective learning rate does not scale with the number of
    /// micro-batches.
    ///
    /// A gradient whose shape differs from the parameters is reported rather
    /// than silently zipped against a truncated view.
    pub fn step(&mut self, gradients: &Array<A, D>) -> Result<HardwareStepReport<A>> {
        if gradients.raw_dim() != self.parameters.raw_dim() {
            return Err(OptimError::DimensionMismatch(format!(
                "hardware-aware step: parameters have shape {:?} but the gradient has shape {:?}",
                self.parameters.raw_dim().slice(),
                gradients.raw_dim().slice()
            )));
        }

        let effective_gradient = if self.accumulation_steps == 1 {
            gradients.clone()
        } else {
            let shape = self.parameters.raw_dim();
            let accumulator = self
                .gradient_accumulator
                .get_or_insert_with(|| Array::zeros(shape));
            for (slot, &g) in accumulator.iter_mut().zip(gradients.iter()) {
                *slot = *slot + g;
            }
            self.accumulated_micro_steps += 1;

            if self.accumulated_micro_steps < self.accumulation_steps {
                return Ok(HardwareStepReport {
                    applied: false,
                    learning_rate: self.lr_schedule.get_learning_rate(),
                    step_count: self.step_count,
                    accumulated_micro_steps: self.accumulated_micro_steps,
                });
            }

            let window = scalar_or(self.accumulated_micro_steps, A::one());
            let averaged = accumulator.mapv(|g| g / window);
            accumulator.fill(A::zero());
            self.accumulated_micro_steps = 0;
            averaged
        };

        // Apply the schedule's current rate, then advance it — the same order a
        // training loop uses, so the first update runs at the configured base
        // rate rather than at an already-decayed one.
        let learning_rate = self.lr_schedule.get_learning_rate();
        self.optimizer.set_learning_rate(learning_rate);
        self.parameters = self.optimizer.step(&self.parameters, &effective_gradient)?;
        self.step_count += 1;
        self.lr_schedule.step();

        Ok(HardwareStepReport {
            applied: true,
            learning_rate,
            step_count: self.step_count,
            accumulated_micro_steps: 0,
        })
    }

    /// Current parameters.
    pub fn parameters(&self) -> &Array<A, D> {
        &self.parameters
    }

    /// Number of optimizer updates applied so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Optimizer family currently in use.
    pub fn optimizer_kind(&self) -> HardwareOptimizerKind {
        self.optimizer_kind
    }

    /// Learning rate the next update will use.
    pub fn learning_rate(&self) -> A {
        self.lr_schedule.get_learning_rate()
    }

    /// Rate this state was constructed with, for comparison against the live
    /// [`OptimizationState::learning_rate`].
    pub fn base_learning_rate(&self) -> A {
        self.base_learning_rate
    }

    /// Micro-batches per optimizer update.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// Micro-batches currently held in the accumulator.
    pub fn accumulated_micro_steps(&self) -> usize {
        self.accumulated_micro_steps
    }

    /// Change the accumulation window.
    ///
    /// Any partially accumulated window is discarded: mixing gradients averaged
    /// over different window sizes would silently change the effective learning
    /// rate of the next update.
    pub fn set_accumulation_steps(&mut self, accumulation_steps: usize) {
        self.accumulation_steps = accumulation_steps.max(1);
        if let Some(accumulator) = self.gradient_accumulator.as_mut() {
            accumulator.fill(A::zero());
        }
        self.accumulated_micro_steps = 0;
    }

    /// Install a learning rate schedule.
    ///
    /// The schedule owns the rate from this point on: every update reads
    /// [`LearningRateScheduler::get_learning_rate`] and pushes it into the
    /// optimizer, so a rate set directly on the optimizer would be overwritten.
    ///
    /// `base_learning_rate` is deliberately left alone: it records what this
    /// state was constructed with, which is the only thing a later comparison
    /// against the live rate can be meaningful against.
    pub fn set_lr_scheduler(&mut self, schedule: Box<dyn LearningRateScheduler<A> + Send + Sync>) {
        self.lr_schedule = schedule;
    }

    /// Replace the optimizer with a freshly built one of family `kind`.
    ///
    /// This discards the accumulated moment state, which is why
    /// [`super::HardwareAwareOptimizer::optimize_for_hardware`] only does it
    /// before the first update.
    pub fn rebuild_optimizer(&mut self, kind: HardwareOptimizerKind) {
        self.optimizer = build_optimizer(kind, self.lr_schedule.get_learning_rate());
        self.optimizer_kind = kind;
    }

    /// Install a caller-supplied optimizer, for the cases the built-in
    /// recommendation does not cover.
    ///
    /// `kind` is what the state will report from
    /// [`OptimizationState::optimizer_kind`]; pass the family `optimizer`
    /// actually belongs to so the reported state stays truthful.
    pub fn set_optimizer(
        &mut self,
        kind: HardwareOptimizerKind,
        optimizer: Box<dyn Optimizer<A, D> + Send + Sync>,
    ) {
        self.optimizer = optimizer;
        self.optimizer_kind = kind;
    }
}

/// Instantiate the optimizer for a family at a given learning rate.
fn build_optimizer<A, D>(
    kind: HardwareOptimizerKind,
    learning_rate: A,
) -> Box<dyn Optimizer<A, D> + Send + Sync>
where
    A: Float + ScalarOperand + std::fmt::Debug + Send + Sync + 'static,
    D: Dimension + 'static,
{
    match kind {
        // A momentum term costs one buffer per parameter and is what makes SGD
        // competitive on the ill-conditioned problems these platforms run; the
        // `Sgd` recommendation is about avoiding Adam's *second* moment, not
        // about running raw gradient descent.
        HardwareOptimizerKind::Sgd => Box::new(SGD::new_with_config(
            learning_rate,
            scalar_or(0.9, A::zero()),
            A::zero(),
        )),
        HardwareOptimizerKind::Lion => Box::new(Lion::new(learning_rate)),
        HardwareOptimizerKind::Adam => Box::new(Adam::new(learning_rate)),
        HardwareOptimizerKind::Lamb => Box::new(LAMB::new(learning_rate)),
    }
}

/// Micro-batches per optimizer update implied by a memory strategy.
///
/// Only [`MemoryStrategy::GradientAccumulation`] asks for accumulation; a
/// `Mixed` strategy takes the largest window any of its components requests, so
/// combining accumulation with another strategy does not silently drop it.
pub(super) fn accumulation_steps_for(strategy: &MemoryStrategy) -> usize {
    match strategy {
        MemoryStrategy::GradientAccumulation { accumulation_steps } => (*accumulation_steps).max(1),
        MemoryStrategy::Mixed { strategies, .. } => strategies
            .iter()
            .map(accumulation_steps_for)
            .max()
            .unwrap_or(1),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedulers::ExponentialDecay;
    use scirs2_core::ndarray::{Array1, Ix1};

    /// Gradient of `f(x) = sum(x_i^2)`.
    fn quadratic_gradient(parameters: &Array1<f64>) -> Array1<f64> {
        parameters.mapv(|x| 2.0 * x)
    }

    /// `f(x) = sum(x_i^2)`.
    fn quadratic_loss(parameters: &Array1<f64>) -> f64 {
        parameters.iter().map(|&x| x * x).sum()
    }

    /// The state must actually run optimizer steps: before this module it only
    /// held a parameter array, so a loss could never move.
    #[test]
    fn steps_reduce_a_quadratic_loss() {
        for kind in [
            HardwareOptimizerKind::Sgd,
            HardwareOptimizerKind::Lion,
            HardwareOptimizerKind::Adam,
            HardwareOptimizerKind::Lamb,
        ] {
            let start = Array1::from_vec(vec![1.0, -2.0, 3.0]);
            let initial_loss = quadratic_loss(&start);
            let mut state: OptimizationState<f64, Ix1> =
                OptimizationState::new(start, kind, 0.05, 1);

            for _ in 0..200 {
                let gradient = quadratic_gradient(state.parameters());
                let report = state.step(&gradient).expect("step must succeed");
                assert!(report.applied, "{} did not apply an update", kind.name());
            }

            let final_loss = quadratic_loss(state.parameters());
            assert_eq!(state.step_count(), 200, "{}", kind.name());
            assert!(
                final_loss < initial_loss * 0.5,
                "{}: loss did not decrease ({initial_loss} -> {final_loss})",
                kind.name()
            );
        }
    }

    /// A gradient of the wrong shape must be reported, not truncated.
    #[test]
    fn mismatched_gradient_shape_is_reported() {
        let mut state: OptimizationState<f64, Ix1> = OptimizationState::new(
            Array1::from_vec(vec![1.0, 2.0]),
            HardwareOptimizerKind::Adam,
            0.01,
            1,
        );
        let error = state
            .step(&Array1::from_vec(vec![1.0, 2.0, 3.0]))
            .expect_err("a shape mismatch must be reported");
        assert!(
            matches!(error, OptimError::DimensionMismatch(_)),
            "{error:?}"
        );
    }

    /// Gradient accumulation must hold the parameters still until the window is
    /// full, then apply the window mean.
    #[test]
    fn gradient_accumulation_updates_once_per_window() {
        let mut state: OptimizationState<f64, Ix1> = OptimizationState::new(
            Array1::from_vec(vec![0.0, 0.0]),
            HardwareOptimizerKind::Sgd,
            0.1,
            3,
        );
        let gradient = Array1::from_vec(vec![1.0, 1.0]);

        for micro in 1..=2 {
            let report = state.step(&gradient).expect("accumulating step");
            assert!(!report.applied, "micro-batch {micro} must not update");
            assert_eq!(report.accumulated_micro_steps, micro);
            assert_eq!(state.parameters()[0], 0.0);
        }

        let report = state.step(&gradient).expect("closing step");
        assert!(report.applied, "the full window must apply an update");
        assert_eq!(state.step_count(), 1);
        assert!(state.parameters()[0] < 0.0);
    }

    /// The window mean, not the window sum, must be applied: three identical
    /// gradients accumulated must move the parameters exactly as far as one
    /// unaccumulated gradient of the same value.
    #[test]
    fn accumulation_applies_the_window_mean() {
        let gradient = Array1::from_vec(vec![1.0, -0.5]);

        let mut direct: OptimizationState<f64, Ix1> = OptimizationState::new(
            Array1::from_vec(vec![0.0, 0.0]),
            HardwareOptimizerKind::Sgd,
            0.1,
            1,
        );
        direct.step(&gradient).expect("direct step");

        let mut accumulated: OptimizationState<f64, Ix1> = OptimizationState::new(
            Array1::from_vec(vec![0.0, 0.0]),
            HardwareOptimizerKind::Sgd,
            0.1,
            3,
        );
        for _ in 0..3 {
            accumulated.step(&gradient).expect("accumulated step");
        }

        for (index, (&direct_value, &accumulated_value)) in direct
            .parameters()
            .iter()
            .zip(accumulated.parameters().iter())
            .enumerate()
        {
            assert!(
                (direct_value - accumulated_value).abs() < 1e-12,
                "coordinate {index}: {direct_value} != {accumulated_value}"
            );
        }
    }

    /// The learning rate schedule must actually drive the optimizer.
    #[test]
    fn the_schedule_drives_the_optimizer_learning_rate() {
        let mut state: OptimizationState<f64, Ix1> = OptimizationState::new(
            Array1::from_vec(vec![1.0]),
            HardwareOptimizerKind::Sgd,
            0.1,
            1,
        );
        state.set_lr_scheduler(Box::new(ExponentialDecay::new(0.1, 0.5, 1)));
        assert!((state.learning_rate() - 0.1).abs() < 1e-12);
        assert!((state.base_learning_rate() - 0.1).abs() < 1e-12);

        let first = state
            .step(&Array1::from_vec(vec![1.0]))
            .expect("first step");
        assert!((first.learning_rate - 0.1).abs() < 1e-12);

        let second = state
            .step(&Array1::from_vec(vec![1.0]))
            .expect("second step");
        assert!(
            second.learning_rate < first.learning_rate,
            "the schedule did not decay: {} -> {}",
            first.learning_rate,
            second.learning_rate
        );
        assert!(
            (state.base_learning_rate() - 0.1).abs() < 1e-12,
            "the construction-time rate must stay put so the decay is measurable"
        );
    }

    /// The memory rationale behind each recommendation must be reported
    /// truthfully: that is the whole basis on which the family is chosen.
    #[test]
    fn each_family_reports_its_optimizer_state_footprint() {
        assert_eq!(HardwareOptimizerKind::Sgd.state_buffers_per_parameter(), 1);
        assert_eq!(HardwareOptimizerKind::Lion.state_buffers_per_parameter(), 1);
        assert_eq!(HardwareOptimizerKind::Adam.state_buffers_per_parameter(), 2);
        assert_eq!(HardwareOptimizerKind::Lamb.state_buffers_per_parameter(), 2);
        assert!(
            HardwareOptimizerKind::Lion.state_buffers_per_parameter()
                < HardwareOptimizerKind::Adam.state_buffers_per_parameter(),
            "Lion is recommended for edge devices precisely because it is cheaper"
        );
    }

    /// The recommendation must follow the hardware constraint, not a constant.
    #[test]
    fn optimizer_recommendation_follows_the_platform() {
        let edge = HardwarePlatform::Edge {
            power_budget: 2.0,
            memory_limit: 256 * 1024 * 1024,
            quantization_support: QuantizationSupport::Int8,
        };
        let mut config: HardwareOptimizationConfig<f64> = HardwareOptimizationConfig {
            batch_size: 16,
            memory_strategy: MemoryStrategy::Standard,
            parallelization: super::super::ParallelizationStrategy::SingleThread,
            precision: super::super::PrecisionStrategy::FP32,
            optimizer_params: std::collections::HashMap::new(),
            communication: None,
        };

        // Very low power: optimizer state is the binding constraint.
        assert_eq!(
            HardwareOptimizerKind::recommend_for(&edge, &config),
            HardwareOptimizerKind::Sgd
        );

        let roomy_edge = HardwarePlatform::Edge {
            power_budget: 30.0,
            memory_limit: 4 * 1024 * 1024 * 1024,
            quantization_support: QuantizationSupport::Int8,
        };
        assert_eq!(
            HardwareOptimizerKind::recommend_for(&roomy_edge, &config),
            HardwareOptimizerKind::Lion
        );

        let gpu = HardwarePlatform::GPU {
            memory: 16 * 1024 * 1024 * 1024,
            compute_units: 80,
            memory_bandwidth: 900.0,
            architecture: super::super::GPUArchitecture::Ampere,
        };
        config.batch_size = 128;
        assert_eq!(
            HardwareOptimizerKind::recommend_for(&gpu, &config),
            HardwareOptimizerKind::Adam
        );

        config.batch_size = 4096;
        assert_eq!(
            HardwareOptimizerKind::recommend_for(&gpu, &config),
            HardwareOptimizerKind::Lamb
        );

        config.memory_strategy = MemoryStrategy::CPUOffloading { offload_ratio: 0.8 };
        assert_eq!(
            HardwareOptimizerKind::recommend_for(&gpu, &config),
            HardwareOptimizerKind::Sgd
        );
    }

    /// A `Mixed` memory strategy must not lose the accumulation window.
    #[test]
    fn accumulation_window_survives_a_mixed_memory_strategy() {
        assert_eq!(accumulation_steps_for(&MemoryStrategy::Standard), 1);
        assert_eq!(
            accumulation_steps_for(&MemoryStrategy::GradientAccumulation {
                accumulation_steps: 0
            }),
            1,
            "a zero window would mean the parameters never move"
        );
        assert_eq!(
            accumulation_steps_for(&MemoryStrategy::Mixed {
                strategies: vec![
                    MemoryStrategy::Standard,
                    MemoryStrategy::GradientAccumulation {
                        accumulation_steps: 4
                    },
                ],
                strategy_weights: vec![0.5, 0.5],
            }),
            4
        );
    }
}
