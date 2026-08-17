// Meta-optimization for the adaptive learning-rate controller (finding E2).
//
// `meta_optimize` used to return the literal `A::from(0.001)` regardless of the
// decision, the step or anything ever measured, and that constant was then
// blended into the learning rate at 30% weight — so with meta-learning enabled
// every adaptation was dragged towards 0.001 no matter what the signals said.
//
// It is now a real UCB1 bandit over a discrete set of learning-rate multipliers,
// using the `arm_rewards` / `arm_counts` fields that were previously dead and
// fed from the effectiveness the caller reports through
// `evaluate_adaptation_effectiveness`.

use super::*;

/// The multipliers the bandit chooses between.
const ARM_MULTIPLIERS: [f64; 5] = [0.5, 0.9, 1.0, 1.1, 2.0];

/// Bound on the retained hyperparameter-update log.
const MAX_HISTORY: usize = 512;

fn to_scalar<A: Float>(value: f64) -> A {
    A::from(value).unwrap_or_else(A::zero)
}

fn from_scalar<A: Float>(value: A) -> f64 {
    value.to_f64().unwrap_or(0.0)
}

impl<A: Float + Default + Clone + Send + Sync> MetaOptimizer<A> {
    pub(crate) fn new(config: &AdaptiveLRConfig<A>) -> Result<Self> {
        // E6: the exploration rate follows the configured sensitivity.
        let sensitivity = from_scalar(config.adaptation_sensitivity).clamp(0.0, 1.0);
        let exploration_strategy = ExplorationStrategy {
            strategy_type: ExplorationStrategyType::UCB1,
            exploration_rate: to_scalar(sensitivity.max(0.01)),
            exploitation_rate: to_scalar(1.0 - sensitivity.max(0.01)),
            ..ExplorationStrategy::default()
        };

        Ok(Self {
            lr_predictor: LearningRatePredictorNetwork::default(),
            optimization_history: VecDeque::with_capacity(MAX_HISTORY),
            exploration_strategy,
            transfer_learner: TransferLearner::default(),
        })
    }

    /// Register a previously solved task whose learning-rate schedule can act as
    /// a prior. With none registered the transfer term contributes nothing.
    pub(crate) fn add_source_task(&mut self, task: TaskData<A>) {
        self.transfer_learner.source_task_data.push(task);
        let count = self.transfer_learner.source_task_data.len();
        self.transfer_learner.transfer_confidence =
            to_scalar((count as f64 / (count as f64 + 4.0)).clamp(0.0, 1.0));
    }

    /// UCB1 arm selection: the arm maximising `mean_reward + sqrt(2 ln n / n_a)`.
    /// Unplayed arms are played first, which is UCB1's own initialisation rule
    /// rather than an arbitrary default.
    fn select_arm(&mut self) -> usize {
        let total_plays: usize = self.exploration_strategy.arm_counts.values().sum();
        for arm in 0..ARM_MULTIPLIERS.len() {
            if self
                .exploration_strategy
                .arm_counts
                .get(&arm)
                .copied()
                .unwrap_or(0)
                == 0
            {
                return arm;
            }
        }

        let total = (total_plays.max(1) as f64).ln();
        let mut best_arm = ARM_MULTIPLIERS.len() / 2;
        let mut best_score = f64::MIN;
        for arm in 0..ARM_MULTIPLIERS.len() {
            let count = self
                .exploration_strategy
                .arm_counts
                .get(&arm)
                .copied()
                .unwrap_or(0)
                .max(1) as f64;
            let reward = self
                .exploration_strategy
                .arm_rewards
                .get(&arm)
                .map(|value| from_scalar(*value))
                .unwrap_or(0.0);
            let exploration = from_scalar(self.exploration_strategy.exploration_rate).max(0.0);
            let score = reward + exploration * (2.0 * total / count).sqrt();
            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }
        best_arm
    }

    /// Suggest a learning rate for this step: the decision's own rate scaled by
    /// the multiplier the bandit currently believes in, blended with any
    /// transferable prior.
    pub(crate) fn meta_optimize(
        &mut self,
        decision: &AdaptationDecision<A>,
        _step: usize,
    ) -> Result<A> {
        let arm = self.select_arm();
        let multiplier = ARM_MULTIPLIERS[arm];
        let suggested = from_scalar(decision.new_lr) * multiplier;

        // Transfer prior: the mean of the next learning rate each similar source
        // task used at this point in its schedule. Contributes nothing (weight
        // zero) while no source task is registered.
        let transfer_weight =
            from_scalar(self.transfer_learner.transfer_confidence).clamp(0.0, 1.0);
        let transfer_prior = if transfer_weight > 0.0 {
            let priors: Vec<f64> = self
                .transfer_learner
                .source_task_data
                .iter()
                .filter_map(|task| task.optimal_lr_sequence.last().copied())
                .map(from_scalar)
                .collect();
            if priors.is_empty() {
                None
            } else {
                Some(priors.iter().sum::<f64>() / priors.len() as f64)
            }
        } else {
            None
        };

        let proposal = match transfer_prior {
            Some(prior) => (1.0 - transfer_weight) * suggested + transfer_weight * prior,
            None => suggested,
        };
        if !proposal.is_finite() || proposal <= 0.0 {
            return Err(crate::error::OptimError::InvalidState(
                "meta-optimization produced a non-positive learning rate".to_string(),
            ));
        }

        // Remember which arm produced this proposal so the reward can be
        // attributed when the caller reports the outcome.
        self.optimization_history.push_back(HyperparameterUpdate {
            timestamp: Instant::now(),
            old_lr: decision.new_lr,
            new_lr: to_scalar(proposal),
            features: Array1::from_vec(vec![
                decision.lr_multiplier,
                decision.confidence,
                to_scalar(multiplier),
                to_scalar(arm as f64),
            ]),
            reward: A::zero(), // filled in by `record_reward`
            exploration_bonus: to_scalar(from_scalar(self.exploration_strategy.exploration_rate)),
        });
        while self.optimization_history.len() > MAX_HISTORY {
            self.optimization_history.pop_front();
        }

        Ok(to_scalar(proposal))
    }

    /// Attribute a measured effectiveness to the arm that produced the most
    /// recent proposal.
    pub(crate) fn record_reward(&mut self, effectiveness: A) {
        let Some(update) = self.optimization_history.back_mut() else {
            return;
        };
        update.reward = effectiveness;
        // The arm index was stored as the fourth feature.
        let Some(arm) = update
            .features
            .iter()
            .nth(3)
            .map(|value| from_scalar(*value))
        else {
            return;
        };
        let arm = arm as usize;
        if arm >= ARM_MULTIPLIERS.len() {
            return;
        }
        let count = self.exploration_strategy.arm_counts.entry(arm).or_insert(0);
        *count += 1;
        let plays = *count as f64;
        let reward = self
            .exploration_strategy
            .arm_rewards
            .entry(arm)
            .or_insert_with(A::zero);
        let previous = from_scalar(*reward);
        *reward = to_scalar(previous + (from_scalar(effectiveness) - previous) / plays);
    }

    /// Measured mean reward per arm.
    pub(crate) fn arm_rewards(&self) -> &HashMap<usize, A> {
        &self.exploration_strategy.arm_rewards
    }

    /// Times each arm has been played.
    pub(crate) fn arm_counts(&self) -> &HashMap<usize, usize> {
        &self.exploration_strategy.arm_counts
    }

    pub(crate) fn reset(&mut self) {
        self.optimization_history.clear();
        self.exploration_strategy.arm_rewards.clear();
        self.exploration_strategy.arm_counts.clear();
    }
}
