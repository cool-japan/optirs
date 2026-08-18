//! Privacy budget tracking and forecasting.
//!
//! # The defects this replaces
//!
//! * `PrivacyBudgetTracker::allocations` was initialised empty and never
//!   written, so `get_current_allocations()` -- reached from
//!   `EnhancedAuditSystem::get_privacy_budget_status()` -- returned an empty
//!   map no matter how much budget had been spent.
//! * `record_consumption` hardcoded `purpose: "optimization"` for every event
//!   and pushed the record onto an unbounded queue that nothing read.
//! * `alerts` was never written, so a budget could be exhausted silently.
//! * `BudgetForecastingModel` and `PredictionModel` were constructors with no
//!   other methods; `PredictionModel::parameters` was an empty `Vec` and
//!   `model_type` was never matched on.
//!
//! What is implemented: allocations keyed by the event's declared processing
//! purpose, real consumption accounting against them, threshold alerts, and an
//! ordinary-least-squares forecast of the exhaustion time. Model types other
//! than linear regression return [`OptimError::UnsupportedOperation`] rather
//! than silently falling back to a line.

use crate::error::{OptimError, Result};
use std::collections::{HashMap, VecDeque};

use super::types::{
    AlertSeverity, AuditEvent, BudgetAlert, BudgetAlertType, BudgetAllocation, BudgetConsumption,
    ConsumptionPattern, ModelType, PatternType,
};

/// Fraction of an allocation at which a warning is raised.
pub const NEARLY_EXHAUSTED_FRACTION: f64 = 0.8;

/// Purpose recorded for events that declare none.
pub const UNSPECIFIED_PURPOSE: &str = "unspecified";

/// Prediction model for budget forecasting.
pub struct PredictionModel {
    /// Fitted parameters: `[intercept, slope]` for a linear model.
    parameters: Vec<f64>,
    /// Model family.
    model_type: ModelType,
}

impl PredictionModel {
    /// Create an unfitted linear model.
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            model_type: ModelType::LinearRegression,
        }
    }

    /// Create an unfitted model of the requested family.
    pub fn with_model_type(model_type: ModelType) -> Self {
        Self {
            parameters: Vec::new(),
            model_type,
        }
    }

    /// The model family.
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.parameters.len() == 2
    }

    /// Fitted `[intercept, slope]`, if the model has been fitted.
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Fit by ordinary least squares.
    ///
    /// Requires at least two distinct abscissae; a degenerate design matrix is
    /// an error rather than an arbitrary slope.
    pub fn fit(&mut self, xs: &[f64], ys: &[f64]) -> Result<()> {
        match self.model_type {
            ModelType::LinearRegression => {}
            other => {
                return Err(OptimError::UnsupportedOperation(format!(
                    "the {other:?} forecasting model is not implemented; only \
                     ModelType::LinearRegression can be fitted here"
                )))
            }
        }
        if xs.len() != ys.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "{} abscissae and {} ordinates were supplied",
                xs.len(),
                ys.len()
            )));
        }
        if xs.len() < 2 {
            return Err(OptimError::InvalidParameter(
                "at least two observations are needed to fit a trend".to_string(),
            ));
        }
        if !xs.iter().chain(ys.iter()).all(|value| value.is_finite()) {
            return Err(OptimError::InvalidParameter(
                "every observation must be finite to fit a trend".to_string(),
            ));
        }

        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut covariance = 0.0;
        let mut variance = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let dx = x - mean_x;
            covariance += dx * (y - mean_y);
            variance += dx * dx;
        }
        if variance <= 0.0 {
            return Err(OptimError::InvalidParameter(
                "all observations share the same abscissa, so no trend is identifiable".to_string(),
            ));
        }
        let slope = covariance / variance;
        let intercept = mean_y - slope * mean_x;
        self.parameters = vec![intercept, slope];
        Ok(())
    }

    /// Predict the ordinate at `x`.
    pub fn predict(&self, x: f64) -> Result<f64> {
        if !self.is_fitted() {
            return Err(OptimError::InvalidState(
                "the prediction model has not been fitted".to_string(),
            ));
        }
        Ok(self.parameters[0] + self.parameters[1] * x)
    }

    /// Solve `predict(x) = target` for `x`.
    pub fn solve_for(&self, target: f64) -> Result<f64> {
        if !self.is_fitted() {
            return Err(OptimError::InvalidState(
                "the prediction model has not been fitted".to_string(),
            ));
        }
        let slope = self.parameters[1];
        if slope.abs() <= f64::EPSILON {
            return Err(OptimError::InvalidState(
                "the fitted trend is flat, so the target is never reached".to_string(),
            ));
        }
        Ok((target - self.parameters[0]) / slope)
    }
}

impl Default for PredictionModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Budget forecasting model over the consumption history.
pub struct BudgetForecastingModel {
    /// Consumption patterns derived from the history, one per purpose.
    patterns: Vec<ConsumptionPattern>,
    /// Fitted trend of cumulative consumption against time.
    model: PredictionModel,
}

impl BudgetForecastingModel {
    /// Create an unfitted model.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            model: PredictionModel::new(),
        }
    }

    /// Derived consumption patterns.
    pub fn patterns(&self) -> &[ConsumptionPattern] {
        &self.patterns
    }

    /// The fitted trend.
    pub fn model(&self) -> &PredictionModel {
        &self.model
    }

    /// Fit the trend and derive the consumption patterns.
    pub fn fit(&mut self, history: &VecDeque<BudgetConsumption>) -> Result<()> {
        if history.len() < 2 {
            return Err(OptimError::InvalidParameter(format!(
                "a consumption forecast needs at least two records, {} are available",
                history.len()
            )));
        }

        // Cumulative spend against time, in insertion order.
        let mut xs = Vec::with_capacity(history.len());
        let mut ys = Vec::with_capacity(history.len());
        let mut cumulative = 0.0;
        for record in history {
            cumulative += record.epsilon_consumed;
            xs.push(record.timestamp as f64);
            ys.push(cumulative);
        }
        self.model.fit(&xs, &ys)?;

        // Per-purpose patterns.
        let mut by_purpose: HashMap<String, Vec<&BudgetConsumption>> = HashMap::new();
        for record in history {
            by_purpose
                .entry(record.purpose.clone())
                .or_default()
                .push(record);
        }
        let mut patterns = Vec::with_capacity(by_purpose.len());
        for (purpose, records) in by_purpose {
            let first = records.iter().map(|r| r.timestamp).min().unwrap_or(0);
            let last = records.iter().map(|r| r.timestamp).max().unwrap_or(0);
            let window = last.saturating_sub(first).max(1);
            let total: f64 = records.iter().map(|r| r.epsilon_consumed).sum();
            let avg_rate = total / window as f64;
            let peak_rate = records
                .iter()
                .map(|r| r.epsilon_consumed)
                .fold(0.0f64, f64::max);
            patterns.push(ConsumptionPattern {
                id: purpose,
                time_window: window,
                avg_consumption_rate: avg_rate,
                peak_consumption_rate: peak_rate,
                pattern_type: classify_pattern(&records),
            });
        }
        patterns.sort_by(|left, right| left.id.cmp(&right.id));
        self.patterns = patterns;
        Ok(())
    }

    /// Predict the cumulative spend at `timestamp`.
    pub fn predict_cumulative(&self, timestamp: u64) -> Result<f64> {
        self.model.predict(timestamp as f64)
    }

    /// Predict the timestamp at which cumulative spend reaches `target`.
    pub fn predict_exhaustion(&self, target: f64) -> Result<u64> {
        if !target.is_finite() || target < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the exhaustion target must be a non-negative finite epsilon, got {target}"
            )));
        }
        let solved = self.model.solve_for(target)?;
        if solved < 0.0 {
            return Err(OptimError::InvalidState(
                "the fitted trend places exhaustion before the Unix epoch".to_string(),
            ));
        }
        Ok(solved as u64)
    }
}

impl Default for BudgetForecastingModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify a consumption series by its dispersion and regularity.
fn classify_pattern(records: &[&BudgetConsumption]) -> PatternType {
    if records.len() < 3 {
        return PatternType::Irregular;
    }
    let amounts: Vec<f64> = records.iter().map(|r| r.epsilon_consumed).collect();
    let n = amounts.len() as f64;
    let mean = amounts.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return PatternType::Irregular;
    }
    let variance = amounts
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / n;
    let coefficient_of_variation = variance.sqrt() / mean;

    // Inter-arrival regularity: a periodic series has near-constant gaps.
    let mut gaps = Vec::with_capacity(records.len().saturating_sub(1));
    for window in records.windows(2) {
        gaps.push(window[1].timestamp.saturating_sub(window[0].timestamp) as f64);
    }
    let gap_mean = if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().sum::<f64>() / gaps.len() as f64
    };
    let gap_cv = if gap_mean > 0.0 {
        let gap_variance = gaps
            .iter()
            .map(|value| (value - gap_mean) * (value - gap_mean))
            .sum::<f64>()
            / gaps.len() as f64;
        gap_variance.sqrt() / gap_mean
    } else {
        f64::INFINITY
    };

    if coefficient_of_variation < 0.1 {
        PatternType::Steady
    } else if coefficient_of_variation > 1.0 {
        PatternType::Bursty
    } else if gap_cv < 0.1 {
        PatternType::Periodic
    } else {
        PatternType::Irregular
    }
}

/// Privacy budget tracker.
pub struct PrivacyBudgetTracker {
    /// Allocations by purpose.
    allocations: HashMap<String, BudgetAllocation>,
    /// Consumption records, oldest first.
    consumption_history: VecDeque<BudgetConsumption>,
    /// Raised alerts.
    alerts: Vec<BudgetAlert>,
    /// Forecasting model.
    forecasting_model: BudgetForecastingModel,
    /// Cap on retained consumption records.
    history_capacity: usize,
}

impl PrivacyBudgetTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            consumption_history: VecDeque::new(),
            alerts: Vec::new(),
            forecasting_model: BudgetForecastingModel::new(),
            history_capacity: 100_000,
        }
    }

    /// Allocate budget to a purpose.
    pub fn allocate(
        &mut self,
        purpose: impl Into<String>,
        allocated_epsilon: f64,
        allocated_delta: f64,
        timestamp: u64,
        expires_at: Option<u64>,
    ) -> Result<()> {
        let purpose = purpose.into();
        if !allocated_epsilon.is_finite() || allocated_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the epsilon allocated to `{purpose}` must be positive and finite, got \
                 {allocated_epsilon}"
            )));
        }
        if !allocated_delta.is_finite() || !(0.0..1.0).contains(&allocated_delta) {
            return Err(OptimError::InvalidParameter(format!(
                "the delta allocated to `{purpose}` must lie in [0, 1), got {allocated_delta}"
            )));
        }
        let entry = self
            .allocations
            .entry(purpose.clone())
            .or_insert_with(|| BudgetAllocation {
                purpose: purpose.clone(),
                allocated_epsilon: 0.0,
                allocated_delta,
                consumed_epsilon: 0.0,
                consumed_delta: 0.0,
                timestamp,
                expires_at,
            });
        entry.allocated_epsilon += allocated_epsilon;
        entry.allocated_delta = allocated_delta;
        entry.expires_at = expires_at;
        Ok(())
    }

    /// Record the spend described by an audit event.
    ///
    /// `event.privacy_context.epsilon_budget` is read as **the epsilon spent by
    /// the operation this event describes**, and must be finite and
    /// non-negative. The purpose is the event's first declared processing
    /// purpose, so consumption lands against the allocation it belongs to
    /// instead of a hardcoded `"optimization"` bucket.
    pub fn record_consumption(&mut self, event: &AuditEvent) -> Result<()> {
        let epsilon = event.privacy_context.epsilon_budget;
        let delta = event.privacy_context.delta_budget;
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "event `{}` reports epsilon = {epsilon}, which is not a spend that can be \
                 accounted",
                event.id
            )));
        }
        if !delta.is_finite() || !(0.0..1.0).contains(&delta) {
            return Err(OptimError::InvalidParameter(format!(
                "event `{}` reports delta = {delta}, which must lie in [0, 1)",
                event.id
            )));
        }

        let purpose = event
            .data
            .processing_purposes
            .first()
            .cloned()
            .unwrap_or_else(|| UNSPECIFIED_PURPOSE.to_string());

        self.consumption_history.push_back(BudgetConsumption {
            id: format!("consumption_{}", self.consumption_history.len()),
            timestamp: event.timestamp,
            purpose: purpose.clone(),
            epsilon_consumed: epsilon,
            delta_consumed: delta,
            operation: event.data.description.clone(),
        });
        while self.consumption_history.len() > self.history_capacity {
            let _ = self.consumption_history.pop_front();
        }

        let mut alert = None;
        if let Some(allocation) = self.allocations.get_mut(&purpose) {
            allocation.consumed_epsilon += epsilon;
            // Delta is a reporting parameter throughout this crate, not an
            // additively consumed budget; the largest delta seen is recorded
            // so a report can state the parameter the guarantee is quoted at.
            allocation.consumed_delta = allocation.consumed_delta.max(delta);
            let allocated = allocation.allocated_epsilon;
            let consumed = allocation.consumed_epsilon;
            if consumed >= allocated {
                alert = Some((
                    BudgetAlertType::BudgetExhausted,
                    AlertSeverity::Critical,
                    format!(
                        "purpose `{purpose}` has consumed epsilon {consumed:.6} of an allocated \
                         {allocated:.6}"
                    ),
                ));
            } else if consumed >= NEARLY_EXHAUSTED_FRACTION * allocated {
                alert = Some((
                    BudgetAlertType::BudgetNearlyExhausted,
                    AlertSeverity::Warning,
                    format!(
                        "purpose `{purpose}` has consumed epsilon {consumed:.6} of an allocated \
                         {allocated:.6}"
                    ),
                ));
            }
            if let Some(expiry) = allocation.expires_at {
                if event.timestamp > expiry {
                    alert = Some((
                        BudgetAlertType::AllocationExpired,
                        AlertSeverity::Critical,
                        format!(
                            "purpose `{purpose}` was spent at {} but its allocation expired at \
                             {expiry}",
                            event.timestamp
                        ),
                    ));
                }
            }
        } else if epsilon > 0.0 {
            alert = Some((
                BudgetAlertType::ReallocationNeeded,
                AlertSeverity::Warning,
                format!("epsilon {epsilon:.6} was spent on unallocated purpose `{purpose}`"),
            ));
        }

        if let Some((alert_type, severity, message)) = alert {
            self.raise_alert(alert_type, severity, message, event.timestamp);
        }

        // Refit the forecast once there is enough history for it to mean
        // anything; a failure to fit is not a failure to record.
        if self.consumption_history.len() >= 2 {
            let _ = self.forecasting_model.fit(&self.consumption_history);
        }
        Ok(())
    }

    /// Raise an alert.
    fn raise_alert(
        &mut self,
        alert_type: BudgetAlertType,
        severity: AlertSeverity,
        message: String,
        timestamp: u64,
    ) {
        self.alerts.push(BudgetAlert {
            id: format!("budget_alert_{}", self.alerts.len()),
            alert_type,
            message,
            severity,
            timestamp,
            acknowledged: false,
        });
    }

    /// Current allocations, by purpose.
    pub fn get_current_allocations(&self) -> HashMap<String, BudgetAllocation> {
        self.allocations.clone()
    }

    /// Consumption records, oldest first.
    pub fn consumption_history(&self) -> impl Iterator<Item = &BudgetConsumption> {
        self.consumption_history.iter()
    }

    /// Raised alerts.
    pub fn alerts(&self) -> &[BudgetAlert] {
        &self.alerts
    }

    /// Total epsilon recorded as spent.
    pub fn total_epsilon_consumed(&self) -> f64 {
        self.consumption_history
            .iter()
            .map(|record| record.epsilon_consumed)
            .sum()
    }

    /// The forecasting model.
    pub fn forecasting_model(&self) -> &BudgetForecastingModel {
        &self.forecasting_model
    }

    /// Forecast the timestamp at which the allocation for `purpose` runs out.
    pub fn forecast_exhaustion(&self, purpose: &str) -> Result<u64> {
        let allocation = self.allocations.get(purpose).ok_or_else(|| {
            OptimError::InvalidParameter(format!("no budget is allocated to purpose `{purpose}`"))
        })?;
        self.forecasting_model
            .predict_exhaustion(allocation.allocated_epsilon)
    }
}

impl Default for PrivacyBudgetTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::enhanced_audit::types::{AuditEventData, AuditEventType, PrivacyContext};

    fn spend_event(id: &str, purpose: &str, epsilon: f64, timestamp: u64) -> AuditEvent {
        AuditEvent {
            id: id.to_string(),
            timestamp,
            event_type: AuditEventType::PrivacyBudgetConsumption,
            actor: "trainer".to_string(),
            data: AuditEventData {
                description: format!("spent {epsilon}"),
                affected_data_subjects: Vec::new(),
                data_categories: Vec::new(),
                processing_purposes: vec![purpose.to_string()],
                legal_basis: vec!["consent".to_string()],
                technical_measures: vec!["differential_privacy".to_string()],
                metadata: HashMap::new(),
            },
            privacy_context: PrivacyContext {
                epsilon_budget: epsilon,
                delta_budget: 1e-6,
                privacy_mechanism: "dp_sgd".to_string(),
                data_minimization: true,
                purpose_limitation: true,
                storage_limitation: true,
            },
            signature: None,
            compliance_annotations: HashMap::new(),
        }
    }

    #[test]
    fn consumption_lands_against_the_declared_purpose() {
        // Regression: `allocations` was never written, so the budget status
        // was an empty map however much epsilon had been spent, and every
        // record was filed under a hardcoded "optimization" purpose.
        let mut tracker = PrivacyBudgetTracker::new();
        let ok = tracker.allocate("ml_training", 1.0, 1e-5, 0, None);
        assert!(ok.is_ok());
        let ok = tracker.allocate("analytics", 2.0, 1e-5, 0, None);
        assert!(ok.is_ok());

        for step in 0..3u64 {
            let event = spend_event(&format!("e{step}"), "ml_training", 0.1, 100 + step * 10);
            let ok = tracker.record_consumption(&event);
            assert!(ok.is_ok(), "record_consumption failed");
        }

        let allocations = tracker.get_current_allocations();
        assert_eq!(allocations.len(), 2);
        let training = match allocations.get("ml_training") {
            Some(allocation) => allocation,
            None => panic!("the ml_training allocation must be tracked"),
        };
        assert!((training.consumed_epsilon - 0.3).abs() < 1e-12);
        let analytics = match allocations.get("analytics") {
            Some(allocation) => allocation,
            None => panic!("the analytics allocation must be tracked"),
        };
        assert_eq!(analytics.consumed_epsilon, 0.0);
        assert!((tracker.total_epsilon_consumed() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn crossing_the_warning_and_exhaustion_thresholds_raises_alerts() {
        let mut tracker = PrivacyBudgetTracker::new();
        let ok = tracker.allocate("ml_training", 1.0, 1e-5, 0, None);
        assert!(ok.is_ok());

        // 0.5 -> below the warning threshold.
        let ok = tracker.record_consumption(&spend_event("a", "ml_training", 0.5, 10));
        assert!(ok.is_ok());
        assert!(tracker.alerts().is_empty());

        // 0.85 -> nearly exhausted.
        let ok = tracker.record_consumption(&spend_event("b", "ml_training", 0.35, 20));
        assert!(ok.is_ok());
        assert_eq!(tracker.alerts().len(), 1);
        assert!(matches!(
            tracker.alerts()[0].alert_type,
            BudgetAlertType::BudgetNearlyExhausted
        ));

        // 1.05 -> exhausted.
        let ok = tracker.record_consumption(&spend_event("c", "ml_training", 0.2, 30));
        assert!(ok.is_ok());
        assert_eq!(tracker.alerts().len(), 2);
        assert!(matches!(
            tracker.alerts()[1].alert_type,
            BudgetAlertType::BudgetExhausted
        ));
        assert!(matches!(
            tracker.alerts()[1].severity,
            AlertSeverity::Critical
        ));
    }

    #[test]
    fn spending_on_an_unallocated_purpose_is_flagged() {
        let mut tracker = PrivacyBudgetTracker::new();
        let ok = tracker.record_consumption(&spend_event("a", "surprise", 0.1, 10));
        assert!(ok.is_ok());
        assert_eq!(tracker.alerts().len(), 1);
        assert!(matches!(
            tracker.alerts()[0].alert_type,
            BudgetAlertType::ReallocationNeeded
        ));
    }

    #[test]
    fn spending_after_the_allocation_expires_is_flagged() {
        let mut tracker = PrivacyBudgetTracker::new();
        let ok = tracker.allocate("ml_training", 1.0, 1e-5, 0, Some(50));
        assert!(ok.is_ok());
        let ok = tracker.record_consumption(&spend_event("a", "ml_training", 0.1, 100));
        assert!(ok.is_ok());
        assert!(tracker
            .alerts()
            .iter()
            .any(|alert| matches!(alert.alert_type, BudgetAlertType::AllocationExpired)));
    }

    #[test]
    fn a_non_finite_spend_is_refused() {
        let mut tracker = PrivacyBudgetTracker::new();
        assert!(tracker
            .record_consumption(&spend_event("a", "p", f64::INFINITY, 1))
            .is_err());
        assert!(tracker
            .record_consumption(&spend_event("b", "p", -1.0, 1))
            .is_err());
        assert!(tracker
            .record_consumption(&spend_event("c", "p", f64::NAN, 1))
            .is_err());
    }

    #[test]
    fn an_invalid_allocation_is_refused() {
        let mut tracker = PrivacyBudgetTracker::new();
        assert!(tracker.allocate("p", 0.0, 1e-5, 0, None).is_err());
        assert!(tracker.allocate("p", -1.0, 1e-5, 0, None).is_err());
        assert!(tracker.allocate("p", 1.0, 1.0, 0, None).is_err());
        assert!(tracker.allocate("p", 1.0, -1e-5, 0, None).is_err());
    }

    #[test]
    fn least_squares_recovers_an_exact_line() {
        let mut model = PredictionModel::new();
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0, 11.0];
        let ok = model.fit(&xs, &ys);
        assert!(ok.is_ok(), "fit failed");
        assert!((model.parameters()[0] - 3.0).abs() < 1e-12);
        assert!((model.parameters()[1] - 2.0).abs() < 1e-12);
        match model.predict(10.0) {
            Ok(value) => assert!((value - 23.0).abs() < 1e-12),
            Err(err) => panic!("predict failed: {err}"),
        }
        match model.solve_for(23.0) {
            Ok(value) => assert!((value - 10.0).abs() < 1e-12),
            Err(err) => panic!("solve failed: {err}"),
        }
    }

    #[test]
    fn a_degenerate_or_unfitted_model_errors_instead_of_guessing() {
        let mut model = PredictionModel::new();
        assert!(
            model.predict(1.0).is_err(),
            "unfitted model must not predict"
        );
        assert!(
            model.fit(&[1.0], &[1.0]).is_err(),
            "one point is not a trend"
        );
        assert!(
            model.fit(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).is_err(),
            "a degenerate design matrix must be refused"
        );
        assert!(model.fit(&[0.0, 1.0], &[0.0, f64::NAN]).is_err());
        assert!(model.fit(&[0.0, 1.0, 2.0], &[0.0, 1.0]).is_err());
    }

    #[test]
    fn an_unimplemented_model_family_is_refused() {
        for model_type in [
            ModelType::ARIMA,
            ModelType::NeuralNetwork,
            ModelType::RandomForest,
        ] {
            let mut model = PredictionModel::with_model_type(model_type);
            let outcome = model.fit(&[0.0, 1.0], &[0.0, 1.0]);
            assert!(
                outcome.is_err(),
                "{model_type:?} must not silently fall back to a line"
            );
        }
    }

    #[test]
    fn the_forecast_extrapolates_the_observed_spend_rate() {
        let mut tracker = PrivacyBudgetTracker::new();
        let ok = tracker.allocate("ml_training", 1.0, 1e-5, 0, None);
        assert!(ok.is_ok());
        // 0.1 epsilon every 10 seconds starting at t=0: cumulative reaches
        // 1.0 at t = 90 (the 10th record at t=90 makes it 1.0).
        for step in 0..5u64 {
            let ok = tracker.record_consumption(&spend_event(
                &format!("e{step}"),
                "ml_training",
                0.1,
                step * 10,
            ));
            assert!(ok.is_ok());
        }
        let exhaustion = match tracker.forecast_exhaustion("ml_training") {
            Ok(timestamp) => timestamp,
            Err(err) => panic!("forecast failed: {err}"),
        };
        // cumulative(t) = 0.1 + 0.01 t  =>  1.0 at t = 90.
        assert!(
            (80..=100).contains(&exhaustion),
            "forecast placed exhaustion at t={exhaustion}"
        );
        assert!(tracker.forecast_exhaustion("nonexistent").is_err());
    }

    #[test]
    fn consumption_patterns_are_derived_per_purpose() {
        let mut tracker = PrivacyBudgetTracker::new();
        let ok = tracker.allocate("steady", 10.0, 1e-5, 0, None);
        assert!(ok.is_ok());
        for step in 0..6u64 {
            let ok = tracker.record_consumption(&spend_event(
                &format!("s{step}"),
                "steady",
                0.1,
                step * 5,
            ));
            assert!(ok.is_ok());
        }
        let patterns = tracker.forecasting_model().patterns();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].id, "steady");
        assert!(matches!(patterns[0].pattern_type, PatternType::Steady));
        assert!(patterns[0].avg_consumption_rate > 0.0);
        assert!((patterns[0].peak_consumption_rate - 0.1).abs() < 1e-12);
    }

    #[test]
    fn a_bursty_series_is_not_classified_as_steady() {
        let mut model = BudgetForecastingModel::new();
        let mut history = VecDeque::new();
        for (index, amount) in [0.001, 0.001, 5.0, 0.001, 0.001].iter().enumerate() {
            history.push_back(BudgetConsumption {
                id: format!("c{index}"),
                timestamp: index as u64 * 7,
                purpose: "bursty".to_string(),
                epsilon_consumed: *amount,
                delta_consumed: 0.0,
                operation: "op".to_string(),
            });
        }
        let ok = model.fit(&history);
        assert!(ok.is_ok(), "fit failed");
        assert_eq!(model.patterns().len(), 1);
        assert!(matches!(
            model.patterns()[0].pattern_type,
            PatternType::Bursty
        ));
    }

    #[test]
    fn a_forecast_needs_at_least_two_records() {
        let mut model = BudgetForecastingModel::new();
        let mut history = VecDeque::new();
        assert!(model.fit(&history).is_err());
        history.push_back(BudgetConsumption {
            id: "c0".to_string(),
            timestamp: 1,
            purpose: "p".to_string(),
            epsilon_consumed: 0.1,
            delta_consumed: 0.0,
            operation: "op".to_string(),
        });
        assert!(model.fit(&history).is_err());
    }
}
