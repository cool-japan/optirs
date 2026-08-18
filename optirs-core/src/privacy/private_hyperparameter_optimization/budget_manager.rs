//! Privacy budget accounting for hyperparameter optimization.
//!
//! Extracted from `types.rs` to keep every file under the 2000-line limit. See
//! [`HPOBudgetManager`] for the field-semantics bug this module fixes.

use crate::error::{OptimError, Result};
use crate::privacy::{DifferentialPrivacyConfig, PrivacyBudget};

use super::types::BudgetAllocationStrategy;

/// Fraction of the total epsilon reserved for the final private selection.
///
/// Selecting the best configuration is itself a query against the private data
/// (Liu & Talwar 2019), so it must be paid for out of the same budget as the
/// evaluations rather than being taken for free.
pub const DEFAULT_SELECTION_BUDGET_FRACTION: f64 = 0.1;

/// Adaptive budget controller.
///
/// Tracks the observed score history and turns it into the allocation weights
/// [`HPOBudgetManager`] uses. Before this, `record_performance` pushed onto a
/// vector nothing ever read, and `budget_efficiency` / `allocation_weights` were
/// never written.
pub struct AdaptiveBudgetController {
    /// Historical performance scores, oldest first
    performance_history: Vec<f64>,
    /// Score improvement per unit epsilon, one entry per recorded evaluation
    budget_efficiency: Vec<f64>,
    /// Allocation weight actually applied, one entry per recorded evaluation
    allocation_weights: Vec<f64>,
    /// Learning rate for adaptation
    adaptation_rate: f64,
    /// Number of recent scores considered by the adaptation window
    window: usize,
}

impl AdaptiveBudgetController {
    /// A controller with a 0.1 adaptation rate over a five-score window.
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            budget_efficiency: Vec::new(),
            allocation_weights: Vec::new(),
            adaptation_rate: 0.1,
            window: 5,
        }
    }

    /// Record an observed score.
    pub fn record_performance(&mut self, score: f64) {
        if score.is_finite() {
            self.performance_history.push(score);
        }
    }

    /// Record the epsilon-efficiency of an evaluation and the weight used.
    pub fn record_allocation(&mut self, weight: f64, epsilon: f64) {
        self.allocation_weights.push(weight);
        if epsilon > 0.0 && self.performance_history.len() >= 2 {
            let last = self.performance_history.len() - 1;
            let gain = self.performance_history[last] - self.performance_history[last - 1];
            self.budget_efficiency.push(gain / epsilon);
        }
    }

    /// The adaptation rate.
    pub fn adaptation_rate(&self) -> f64 {
        self.adaptation_rate
    }

    /// Replace the adaptation rate.
    pub fn set_adaptation_rate(&mut self, rate: f64) -> Result<()> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the adaptation rate must be non-negative and finite, got {rate}"
            )));
        }
        self.adaptation_rate = rate;
        Ok(())
    }

    /// Recorded scores, oldest first.
    pub fn performance_history(&self) -> &[f64] {
        &self.performance_history
    }

    /// Recorded score-per-epsilon efficiencies.
    pub fn budget_efficiency(&self) -> &[f64] {
        &self.budget_efficiency
    }

    /// Recorded allocation weights.
    pub fn allocation_weights(&self) -> &[f64] {
        &self.allocation_weights
    }

    /// Improvement of the best score over the adaptation window, normalised by
    /// the window's absolute score scale.
    ///
    /// Returns 0 while there is not enough history to measure an improvement, so
    /// the first trials are allocated uniformly.
    pub fn recent_improvement(&self) -> f64 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }
        let start = self.performance_history.len().saturating_sub(self.window);
        let window = &self.performance_history[start..];
        let first = window[0];
        let last = window[window.len() - 1];
        let scale = window
            .iter()
            .map(|score| score.abs())
            .fold(0.0f64, f64::max)
            .max(f64::EPSILON);
        ((last - first) / scale).clamp(-1.0, 1.0)
    }

    /// Coefficient of variation of the scores over the adaptation window.
    pub fn recent_dispersion(&self) -> f64 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }
        let start = self.performance_history.len().saturating_sub(self.window);
        let window = &self.performance_history[start..];
        let count = window.len() as f64;
        let mean = window.iter().sum::<f64>() / count;
        let variance = window
            .iter()
            .map(|score| (score - mean) * (score - mean))
            .sum::<f64>()
            / count;
        let scale = mean.abs().max(f64::EPSILON);
        (variance.sqrt() / scale).clamp(0.0, 1.5)
    }
}

/// Privacy budget manager for hyperparameter optimization.
///
/// # The accounting bug this replaces
///
/// `get_evaluation_budget` returned a [`PrivacyBudget`] whose *per-trial
/// allocation* was stored in `epsilon_remaining` with `epsilon_consumed: 0.0`,
/// and `record_evaluation` then read `budgetused.epsilon_remaining` and both
/// added it to `epsilon_consumed` and subtracted it from `epsilon_remaining`.
/// The meaning of the field flipped between the two calls, so a caller passing
/// back a genuine budget object corrupted the accounting, and nothing stopped
/// `epsilon_remaining` from going negative.
///
/// # The convention used now
///
/// * The grant returned by [`HPOBudgetManager::get_evaluation_budget`] carries
///   the allocation in **`epsilon_consumed`** -- the amount this evaluation is
///   authorised to spend and, by construction, does spend -- and the manager's
///   remaining budget in `epsilon_remaining`. `record_evaluation` reads the same
///   field, so the meaning never changes.
/// * Delta is a **reporting parameter, not an additive spend**, matching the
///   convention documented at `privacy::mod`. `delta_consumed` is always 0 and
///   `delta_remaining` always carries the reporting delta. The previous code
///   subtracted delta additively and gated `has_budget_remaining` on it, which
///   contradicted the rest of the crate.
/// * Composition across HPO trials is **basic (linear) composition** of
///   pure-epsilon mechanisms: epsilons add. That is always a valid upper bound.
///   The `accounting_method` field carries the *base optimizer's* accountant,
///   which is what governs each individual DP-SGD evaluation; it does not claim
///   that the HPO-level composition is tighter than linear.
pub struct HPOBudgetManager {
    /// Epsilon available to the evaluations (total minus the selection reserve)
    evaluation_epsilon: f64,
    /// Epsilon reserved for the final private selection
    selection_epsilon: f64,
    /// Delta the guarantee is reported at
    reporting_delta: f64,
    /// Epsilon granted and spent so far
    epsilon_spent: f64,
    /// Number of evaluations planned
    num_evaluations: usize,
    /// Number of evaluations recorded
    evaluations_recorded: usize,
    /// Per-evaluation allocations granted so far
    evaluation_allocations: Vec<f64>,
    /// Budget allocation strategy
    allocation_strategy: BudgetAllocationStrategy,
    /// Accounting method of the underlying optimizer
    accounting_method: crate::privacy::AccountingMethod,
    /// Adaptive budget controller
    adaptive_controller: AdaptiveBudgetController,
}

impl HPOBudgetManager {
    /// Create a manager that spends the whole budget on evaluations.
    pub fn new(
        baseconfig: DifferentialPrivacyConfig,
        allocation_strategy: BudgetAllocationStrategy,
        num_evaluations: usize,
    ) -> Result<Self> {
        Self::with_selection_reserve(baseconfig, allocation_strategy, num_evaluations, 0.0)
    }

    /// Create a manager that holds back `selection_fraction` of the budget for
    /// the final private selection.
    pub fn with_selection_reserve(
        baseconfig: DifferentialPrivacyConfig,
        allocation_strategy: BudgetAllocationStrategy,
        num_evaluations: usize,
        selection_fraction: f64,
    ) -> Result<Self> {
        if !baseconfig.target_epsilon.is_finite() || baseconfig.target_epsilon <= 0.0 {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "target_epsilon must be positive and finite, got {}",
                baseconfig.target_epsilon
            )));
        }
        if !baseconfig.target_delta.is_finite() || !(0.0..1.0).contains(&baseconfig.target_delta) {
            return Err(OptimError::InvalidPrivacyConfig(format!(
                "target_delta must lie in [0, 1), got {}",
                baseconfig.target_delta
            )));
        }
        if num_evaluations == 0 {
            return Err(OptimError::InvalidConfig(
                "num_evaluations must be positive".to_string(),
            ));
        }
        if !(0.0..0.5).contains(&selection_fraction) {
            return Err(OptimError::InvalidConfig(format!(
                "the selection budget fraction must lie in [0, 0.5), got {selection_fraction}"
            )));
        }

        let selection_epsilon = baseconfig.target_epsilon * selection_fraction;
        Ok(Self {
            evaluation_epsilon: baseconfig.target_epsilon - selection_epsilon,
            selection_epsilon,
            reporting_delta: baseconfig.target_delta,
            epsilon_spent: 0.0,
            num_evaluations,
            evaluations_recorded: 0,
            evaluation_allocations: Vec::new(),
            allocation_strategy,
            accounting_method: baseconfig.accounting_method,
            adaptive_controller: AdaptiveBudgetController::new(),
        })
    }

    /// Epsilon still available to evaluations.
    pub fn epsilon_remaining(&self) -> f64 {
        (self.evaluation_epsilon - self.epsilon_spent).max(0.0)
    }

    /// Epsilon reserved for the final private selection.
    pub fn selection_epsilon(&self) -> f64 {
        self.selection_epsilon
    }

    /// Epsilon granted and spent on evaluations so far.
    pub fn epsilon_spent(&self) -> f64 {
        self.epsilon_spent
    }

    /// Whether any evaluation budget is left.
    ///
    /// Gated on epsilon only: delta is not consumed additively anywhere in this
    /// crate, and gating on it made the manager report exhaustion after the
    /// first trial.
    pub fn has_budget_remaining(&self) -> Result<bool> {
        Ok(self.epsilon_remaining() > f64::EPSILON
            && self.evaluations_recorded < self.num_evaluations)
    }

    /// Allocation weight for `iteration` under the configured strategy.
    fn allocation_weight(&self, iteration: usize) -> Result<f64> {
        match self.allocation_strategy {
            BudgetAllocationStrategy::Equal => Ok(1.0),
            BudgetAllocationStrategy::Adaptive => {
                // Andrew-style adaptation: spend more when the search is still
                // improving, less when it has plateaued. Bounded to [0.5, 2] so
                // one noisy trial cannot drain the budget.
                let improvement = self.adaptive_controller.recent_improvement();
                Ok(
                    (1.0 + self.adaptive_controller.adaptation_rate() * improvement)
                        .clamp(0.5, 2.0),
                )
            }
            BudgetAllocationStrategy::Hierarchical => {
                // Coarse to fine: geometrically increasing per-trial budget, so
                // later (finer) trials are measured more accurately.
                let stage = (iteration as f64) / (self.num_evaluations.max(4) as f64 / 4.0);
                Ok(2.0f64.powf(stage.min(4.0)))
            }
            BudgetAllocationStrategy::Uncertainty => {
                // Spend in proportion to how uncertain the recent scores are;
                // a flat landscape needs less accuracy to rank.
                let dispersion = self.adaptive_controller.recent_dispersion();
                Ok((0.5 + dispersion).clamp(0.5, 2.0))
            }
            BudgetAllocationStrategy::Bandit => Err(OptimError::UnsupportedOperation(
                "BudgetAllocationStrategy::Bandit needs a per-arm reward history, which is not \
                 available when the budget for the next trial is allocated; use Adaptive or \
                 Uncertainty"
                    .to_string(),
            )),
        }
    }

    /// Grant the budget for evaluation `iteration`.
    ///
    /// The grant is returned with the allocation in `epsilon_consumed`. Pass the
    /// same object back to [`HPOBudgetManager::record_evaluation`].
    pub fn get_evaluation_budget(&mut self, iteration: usize) -> Result<PrivacyBudget> {
        let remaining = self.epsilon_remaining();
        if remaining <= f64::EPSILON {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.epsilon_spent,
                target_epsilon: self.evaluation_epsilon,
            });
        }
        let trials_left = self.num_evaluations.saturating_sub(iteration).max(1);
        let uniform = remaining / trials_left as f64;
        let allocation = (uniform * self.allocation_weight(iteration)?).min(remaining);
        if !allocation.is_finite() || allocation <= 0.0 {
            return Err(OptimError::PrivacyAccountingError(format!(
                "the allocation for evaluation {iteration} came out as {allocation}"
            )));
        }

        self.evaluation_allocations.push(allocation);
        Ok(PrivacyBudget {
            epsilon_consumed: allocation,
            delta_consumed: 0.0,
            epsilon_remaining: remaining,
            delta_remaining: self.reporting_delta,
            steps_taken: self.evaluations_recorded,
            accounting_method: self.accounting_method,
            estimated_steps_remaining: trials_left,
        })
    }

    /// Charge the granted budget and record the observed score.
    pub fn record_evaluation(&mut self, budgetused: &PrivacyBudget, score: f64) -> Result<()> {
        let allocation = budgetused.epsilon_consumed;
        if !allocation.is_finite() || allocation < 0.0 {
            return Err(OptimError::PrivacyAccountingError(format!(
                "an evaluation reported a spend of {allocation}, which is not an epsilon"
            )));
        }
        if allocation > self.epsilon_remaining() + 1e-12 {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: self.epsilon_spent + allocation,
                target_epsilon: self.evaluation_epsilon,
            });
        }
        self.epsilon_spent += allocation;
        self.evaluations_recorded += 1;
        self.adaptive_controller.record_performance(score);
        Ok(())
    }

    /// The budget consumed so far, including the selection reserve once spent.
    pub fn get_total_consumed_budget(&self) -> PrivacyBudget {
        PrivacyBudget {
            epsilon_consumed: self.epsilon_spent,
            delta_consumed: 0.0,
            epsilon_remaining: self.epsilon_remaining(),
            delta_remaining: self.reporting_delta,
            steps_taken: self.evaluations_recorded,
            accounting_method: self.accounting_method,
            estimated_steps_remaining: self
                .num_evaluations
                .saturating_sub(self.evaluations_recorded),
        }
    }

    /// Charge the selection reserve, once the selection has been made.
    pub fn record_selection_spend(&mut self, epsilon: f64) -> Result<()> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(OptimError::PrivacyAccountingError(format!(
                "the selection reported a spend of {epsilon}, which is not an epsilon"
            )));
        }
        if epsilon > self.selection_epsilon + 1e-12 {
            return Err(OptimError::PrivacyBudgetExhausted {
                consumed_epsilon: epsilon,
                target_epsilon: self.selection_epsilon,
            });
        }
        self.epsilon_spent += epsilon;
        self.evaluation_epsilon += epsilon;
        self.selection_epsilon -= epsilon;
        Ok(())
    }

    /// Per-evaluation allocations granted so far.
    pub fn evaluation_allocations(&self) -> &[f64] {
        &self.evaluation_allocations
    }

    /// The recorded score history.
    pub fn performance_history(&self) -> &[f64] {
        self.adaptive_controller.performance_history()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::AccountingMethod;

    fn base_config(target_epsilon: f64) -> DifferentialPrivacyConfig {
        DifferentialPrivacyConfig {
            target_epsilon,
            ..DifferentialPrivacyConfig::default()
        }
    }

    fn manager(strategy: BudgetAllocationStrategy, trials: usize) -> HPOBudgetManager {
        match HPOBudgetManager::new(base_config(1.0), strategy, trials) {
            Ok(manager) => manager,
            Err(err) => panic!("construction failed: {err}"),
        }
    }

    #[test]
    fn the_granted_field_and_the_charged_field_are_the_same_field() {
        // Regression for the accounting bug: the allocation used to be returned
        // in `epsilon_remaining` and charged from `epsilon_remaining` while also
        // being added to `epsilon_consumed`, so the field's meaning flipped
        // between the grant and the charge.
        let mut manager = manager(BudgetAllocationStrategy::Equal, 4);
        let grant = match manager.get_evaluation_budget(0) {
            Ok(grant) => grant,
            Err(err) => panic!("grant failed: {err}"),
        };
        assert!(
            (grant.epsilon_consumed - 0.25).abs() < 1e-12,
            "the grant is carried in epsilon_consumed, got {}",
            grant.epsilon_consumed
        );
        assert!(
            (grant.epsilon_remaining - 1.0).abs() < 1e-12,
            "epsilon_remaining reports the manager's remaining budget, got {}",
            grant.epsilon_remaining
        );

        let ok = manager.record_evaluation(&grant, 0.5);
        assert!(ok.is_ok(), "charge failed");
        assert!((manager.epsilon_spent() - 0.25).abs() < 1e-12);
        assert!((manager.epsilon_remaining() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn equal_allocation_spends_exactly_the_target_over_the_planned_trials() {
        let mut manager = manager(BudgetAllocationStrategy::Equal, 4);
        for iteration in 0..4 {
            assert!(
                match manager.has_budget_remaining() {
                    Ok(remaining) => remaining,
                    Err(err) => panic!("check failed: {err}"),
                },
                "budget must remain at iteration {iteration}"
            );
            let grant = match manager.get_evaluation_budget(iteration) {
                Ok(grant) => grant,
                Err(err) => panic!("grant failed: {err}"),
            };
            let ok = manager.record_evaluation(&grant, iteration as f64);
            assert!(ok.is_ok());
        }
        assert!(
            (manager.epsilon_spent() - 1.0).abs() < 1e-9,
            "spent {}",
            manager.epsilon_spent()
        );
        assert!(manager.epsilon_remaining() < 1e-9);
        assert!(
            !match manager.has_budget_remaining() {
                Ok(remaining) => remaining,
                Err(err) => panic!("check failed: {err}"),
            },
            "the budget must be reported as exhausted"
        );
    }

    #[test]
    fn the_remaining_budget_never_goes_negative() {
        let mut manager = manager(BudgetAllocationStrategy::Equal, 3);
        for iteration in 0..3 {
            let grant = match manager.get_evaluation_budget(iteration) {
                Ok(grant) => grant,
                Err(err) => panic!("grant failed: {err}"),
            };
            let ok = manager.record_evaluation(&grant, 0.0);
            assert!(ok.is_ok());
            assert!(
                manager.epsilon_remaining() >= 0.0,
                "remaining went negative at iteration {iteration}"
            );
        }
        assert!(manager.get_evaluation_budget(3).is_err());
    }

    #[test]
    fn overspending_a_grant_is_refused() {
        let mut manager = manager(BudgetAllocationStrategy::Equal, 2);
        let mut grant = match manager.get_evaluation_budget(0) {
            Ok(grant) => grant,
            Err(err) => panic!("grant failed: {err}"),
        };
        grant.epsilon_consumed = 10.0;
        assert!(
            manager.record_evaluation(&grant, 0.0).is_err(),
            "a caller must not be able to charge more than the budget holds"
        );
        grant.epsilon_consumed = f64::NAN;
        assert!(manager.record_evaluation(&grant, 0.0).is_err());
        grant.epsilon_consumed = -1.0;
        assert!(manager.record_evaluation(&grant, 0.0).is_err());
    }

    #[test]
    fn delta_is_reported_and_never_consumed() {
        // Regression: `record_evaluation` used to subtract delta additively and
        // `has_budget_remaining` gated on it, so the manager reported exhaustion
        // after the first trial.
        let mut manager = manager(BudgetAllocationStrategy::Equal, 100);
        for iteration in 0..10 {
            let grant = match manager.get_evaluation_budget(iteration) {
                Ok(grant) => grant,
                Err(err) => panic!("grant failed: {err}"),
            };
            assert_eq!(grant.delta_consumed, 0.0);
            assert_eq!(grant.delta_remaining, 1e-5);
            let ok = manager.record_evaluation(&grant, 0.0);
            assert!(ok.is_ok());
            assert!(
                match manager.has_budget_remaining() {
                    Ok(remaining) => remaining,
                    Err(err) => panic!("check failed: {err}"),
                },
                "the manager must still have budget at iteration {iteration}"
            );
        }
        let total = manager.get_total_consumed_budget();
        assert_eq!(total.delta_consumed, 0.0);
        assert_eq!(total.delta_remaining, 1e-5);
        assert_eq!(total.steps_taken, 10);
    }

    #[test]
    fn invalid_privacy_parameters_are_refused_at_construction() {
        for epsilon in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                HPOBudgetManager::new(base_config(epsilon), BudgetAllocationStrategy::Equal, 4)
                    .is_err(),
                "target_epsilon = {epsilon} must be refused"
            );
        }
        for delta in [1.0f64, -1e-5, f64::NAN] {
            let config = DifferentialPrivacyConfig {
                target_delta: delta,
                ..DifferentialPrivacyConfig::default()
            };
            assert!(
                HPOBudgetManager::new(config, BudgetAllocationStrategy::Equal, 4).is_err(),
                "target_delta = {delta} must be refused"
            );
        }
        assert!(
            HPOBudgetManager::new(base_config(1.0), BudgetAllocationStrategy::Equal, 0).is_err(),
            "zero evaluations must be refused"
        );
    }

    #[test]
    fn the_selection_reserve_is_held_back_and_then_charged() {
        let mut manager = match HPOBudgetManager::with_selection_reserve(
            base_config(1.0),
            BudgetAllocationStrategy::Equal,
            4,
            0.1,
        ) {
            Ok(manager) => manager,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert!((manager.selection_epsilon() - 0.1).abs() < 1e-12);
        assert!(
            (manager.epsilon_remaining() - 0.9).abs() < 1e-12,
            "only 0.9 may be spent on evaluations, got {}",
            manager.epsilon_remaining()
        );

        for iteration in 0..4 {
            let grant = match manager.get_evaluation_budget(iteration) {
                Ok(grant) => grant,
                Err(err) => panic!("grant failed: {err}"),
            };
            let ok = manager.record_evaluation(&grant, 0.0);
            assert!(ok.is_ok());
        }
        assert!((manager.epsilon_spent() - 0.9).abs() < 1e-9);

        let ok = manager.record_selection_spend(0.1);
        assert!(ok.is_ok(), "the reserve must be chargeable");
        assert!(
            (manager.epsilon_spent() - 1.0).abs() < 1e-9,
            "spent {}",
            manager.epsilon_spent()
        );
        assert!(
            manager.record_selection_spend(0.1).is_err(),
            "the reserve must not be chargeable twice"
        );
    }

    #[test]
    fn an_out_of_range_selection_reserve_is_refused() {
        for fraction in [-0.1f64, 0.5, 1.0, f64::NAN] {
            assert!(
                HPOBudgetManager::with_selection_reserve(
                    base_config(1.0),
                    BudgetAllocationStrategy::Equal,
                    4,
                    fraction
                )
                .is_err(),
                "fraction {fraction} must be refused"
            );
        }
    }

    #[test]
    fn the_bandit_strategy_is_refused_rather_than_faked() {
        let mut manager = manager(BudgetAllocationStrategy::Bandit, 4);
        let message = match manager.get_evaluation_budget(0) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("Bandit allocation is not implementable here"),
        };
        assert!(message.contains("per-arm reward history"), "got: {message}");
    }

    #[test]
    fn every_implemented_strategy_stays_within_the_budget() {
        for strategy in [
            BudgetAllocationStrategy::Equal,
            BudgetAllocationStrategy::Adaptive,
            BudgetAllocationStrategy::Hierarchical,
            BudgetAllocationStrategy::Uncertainty,
        ] {
            let mut manager = manager(strategy, 8);
            let mut granted_total = 0.0;
            for iteration in 0..8 {
                if !match manager.has_budget_remaining() {
                    Ok(remaining) => remaining,
                    Err(err) => panic!("check failed: {err}"),
                } {
                    break;
                }
                let grant = match manager.get_evaluation_budget(iteration) {
                    Ok(grant) => grant,
                    Err(err) => panic!("{strategy:?} grant failed: {err}"),
                };
                assert!(
                    grant.epsilon_consumed > 0.0,
                    "{strategy:?} granted a non-positive epsilon"
                );
                granted_total += grant.epsilon_consumed;
                let ok = manager.record_evaluation(&grant, iteration as f64 * 0.1);
                assert!(ok.is_ok(), "{strategy:?} charge failed");
            }
            assert!(
                granted_total <= 1.0 + 1e-9,
                "{strategy:?} granted {granted_total} of a 1.0 budget"
            );
        }
    }

    #[test]
    fn the_adaptive_controller_measures_real_improvement_and_dispersion() {
        let mut controller = AdaptiveBudgetController::new();
        assert_eq!(controller.recent_improvement(), 0.0);
        assert_eq!(controller.recent_dispersion(), 0.0);

        for score in [0.1f64, 0.2, 0.3, 0.4, 0.5] {
            controller.record_performance(score);
        }
        assert!(
            controller.recent_improvement() > 0.5,
            "a rising series must show improvement, got {}",
            controller.recent_improvement()
        );
        assert_eq!(controller.performance_history().len(), 5);

        let mut flat = AdaptiveBudgetController::new();
        for _ in 0..5 {
            flat.record_performance(0.5);
        }
        assert!(flat.recent_improvement().abs() < 1e-12);
        assert!(flat.recent_dispersion() < 1e-12);

        let mut noisy = AdaptiveBudgetController::new();
        for score in [0.1f64, 0.9, 0.2, 0.8, 0.15] {
            noisy.record_performance(score);
        }
        assert!(
            noisy.recent_dispersion() > flat.recent_dispersion(),
            "a scattered series must be more dispersed than a flat one"
        );

        // Non-finite scores are not recorded, so they cannot poison the window.
        let before = noisy.performance_history().len();
        noisy.record_performance(f64::NAN);
        assert_eq!(noisy.performance_history().len(), before);
        assert!(noisy.set_adaptation_rate(-1.0).is_err());
        assert!(noisy.set_adaptation_rate(0.25).is_ok());
        assert!((noisy.adaptation_rate() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn the_grant_records_the_base_optimizers_accounting_method() {
        let config = DifferentialPrivacyConfig {
            accounting_method: AccountingMethod::RenyiDP,
            ..DifferentialPrivacyConfig::default()
        };
        let mut manager = match HPOBudgetManager::new(config, BudgetAllocationStrategy::Equal, 2) {
            Ok(manager) => manager,
            Err(err) => panic!("construction failed: {err}"),
        };
        let grant = match manager.get_evaluation_budget(0) {
            Ok(grant) => grant,
            Err(err) => panic!("grant failed: {err}"),
        };
        assert!(matches!(grant.accounting_method, AccountingMethod::RenyiDP));
    }
}
