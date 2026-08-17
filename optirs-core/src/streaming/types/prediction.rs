//! Online-trained prediction model for the predictive streaming engine (T8).
//!
//! The previous implementation kept a `weights` vector that was allocated to
//! zeros and never touched again, and "predicted" by cloning the most recent
//! observed data point. Nothing in the crate ever trained it, so the
//! predictive engine could not anticipate anything: it echoed the present.
//!
//! This module replaces that with a real, incrementally-fitted model.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::functions::to_a_or;
use super::types_2::StreamingDataPoint;

/// Default autoregressive order used by [`PredictiveStreamingEngine`] when no
/// explicit order is configured.
///
/// [`PredictiveStreamingEngine`]: super::primitives::PredictiveStreamingEngine
pub(super) const DEFAULT_MODEL_ORDER: usize = 4;

/// Initial diagonal of the RLS inverse-correlation matrix. A large value
/// encodes "no prior information", which is what makes the first updates move
/// the weights decisively instead of crawling.
const RLS_INITIAL_PRECISION: f64 = 1.0e4;

/// If the trace of the inverse-correlation matrix ever exceeds this bound the
/// matrix is reset to its initial value. This is the standard guard against
/// *covariance windup*: with exponential forgetting, directions of the
/// regressor space that stop being excited make `P` grow like
/// `forgetting_factor^-n`, and an unbounded `P` eventually turns a tiny
/// numerical error into an enormous weight jump.
const RLS_MAX_TRACE: f64 = 1.0e12;

/// Smoothing factor for the running error/scale statistics that
/// [`PredictionModel::confidence`] is derived from.
const ERROR_EWMA_ALPHA: f64 = 0.05;

/// Autoregressive prediction model for streaming data, fitted online.
///
/// The model is a vector autoregression with *shared scalar coefficients*:
/// the next observation vector is predicted from the previous `model_order`
/// observed vectors as
///
/// ```text
/// x[t] ~= a_1 * x[t-1] + a_2 * x[t-2] + ... + a_p * x[t-p] + b
/// ```
///
/// with one scalar coefficient per lag plus an intercept, shared across all
/// coordinates. Sharing the coefficients keeps the parameter count
/// independent of the feature dimensionality (so a wide stream does not need
/// a quadratic number of parameters to be trainable from a short window)
/// while still capturing the temporal structure — trends, decay and
/// oscillation — that a "repeat the last value" predictor cannot represent
/// at all.
///
/// Fitting uses exponentially-weighted **Recursive Least Squares**: every
/// observed coordinate of every new sample performs one exact rank-1 update
/// of the least-squares solution, so the model is always the exact
/// (forgetting-weighted) least-squares fit of everything it has seen. That
/// is what makes it converge in a handful of periods rather than needing a
/// learning-rate schedule.
pub struct PredictionModel<A: Float + Send + Sync> {
    /// Coefficient vector `[a_1, ..., a_p, b]`, length `model_order + 1`.
    pub(super) weights: Array1<A>,
    /// Dimensionality of the observed feature vectors. Zero until the first
    /// observation, then adopted from the data.
    pub(super) featuredim: usize,
    /// Autoregressive order `p` (number of lags).
    pub(super) model_order: usize,
    /// RLS inverse-correlation matrix `P`, `(p + 1) x (p + 1)`.
    pub(super) inverse_correlation: Array2<A>,
    /// Exponential forgetting factor in `(0, 1]`. Values below 1 let the fit
    /// track a drifting stream instead of averaging over all history.
    pub(super) forgetting_factor: A,
    /// Most recent `model_order` observed feature vectors, newest first.
    pub(super) feature_lags: VecDeque<Array1<A>>,
    /// Most recent `model_order` observed targets, newest first. `None`
    /// entries mark unlabelled samples, which disable target prediction.
    pub(super) target_lags: VecDeque<Option<A>>,
    /// Number of coordinate-level RLS updates performed so far.
    pub(super) train_updates: usize,
    /// EWMA of the absolute one-step-ahead prediction error.
    pub(super) error_ewma: A,
    /// EWMA of the absolute observed value, used to normalise `error_ewma`.
    pub(super) scale_ewma: A,
}

impl<A: Float + Send + Sync> PredictionModel<A> {
    /// Create a model of the given autoregressive order with the default
    /// forgetting factor.
    ///
    /// Note that the argument is the autoregressive **order** (number of
    /// lags). The previous signature took a "feature dimension" and sized the
    /// never-trained weight vector with it; the real feature dimensionality
    /// is now adopted from the data on the first observation.
    pub fn new(model_order: usize) -> Result<Self> {
        Self::with_forgetting_factor(model_order, to_a_or(0.99, A::one()))
    }

    /// Create a model with an explicit exponential forgetting factor.
    pub fn with_forgetting_factor(model_order: usize, forgetting_factor: A) -> Result<Self> {
        if model_order == 0 {
            return Err(OptimError::InvalidConfig(
                "prediction model order must be at least 1".to_string(),
            ));
        }
        if !(forgetting_factor > A::zero() && forgetting_factor <= A::one()) {
            return Err(OptimError::InvalidConfig(
                "prediction model forgetting factor must lie in (0, 1]".to_string(),
            ));
        }
        let params = model_order + 1;
        let mut inverse_correlation = Array2::zeros((params, params));
        let initial = to_a_or(RLS_INITIAL_PRECISION, A::one());
        for i in 0..params {
            inverse_correlation[[i, i]] = initial;
        }
        Ok(Self {
            weights: Array1::zeros(params),
            featuredim: 0,
            model_order,
            inverse_correlation,
            forgetting_factor,
            feature_lags: VecDeque::with_capacity(model_order),
            target_lags: VecDeque::with_capacity(model_order),
            train_updates: 0,
            error_ewma: A::zero(),
            scale_ewma: A::zero(),
        })
    }

    /// Number of RLS updates applied so far (one per observed coordinate of
    /// every sample that had a full lag window available).
    pub fn train_updates(&self) -> usize {
        self.train_updates
    }

    /// The fitted autoregressive coefficients `[a_1, ..., a_p, b]`.
    pub fn coefficients(&self) -> &Array1<A> {
        &self.weights
    }

    /// EWMA of the absolute one-step-ahead prediction error measured on live
    /// data *before* each sample was used for training, i.e. an honest
    /// out-of-sample error estimate.
    pub fn mean_absolute_error(&self) -> A {
        self.error_ewma
    }

    /// Measured out-of-sample error relative to the observed signal scale.
    /// Zero for an untrained model (it has made no errors yet because it has
    /// made no predictions).
    pub fn normalized_error(&self) -> A {
        if self.train_updates == 0 {
            return A::zero();
        }
        let eps = to_a_or(1e-12, A::one());
        let normalized = self.error_ewma / (self.scale_ewma + eps);
        if normalized.is_finite() {
            normalized.max(A::zero())
        } else {
            A::zero()
        }
    }

    /// Confidence in `[0, 1]`, derived from the measured out-of-sample error
    /// relative to the observed signal scale: `1 / (1 + error / scale)`. An
    /// untrained model reports zero — it has no basis for any claim.
    pub fn confidence(&self) -> A {
        if self.train_updates == 0 {
            return A::zero();
        }
        let eps = to_a_or(1e-12, A::one());
        let normalized = self.error_ewma / (self.scale_ewma + eps);
        if !normalized.is_finite() {
            return A::zero();
        }
        let confidence = A::one() / (A::one() + normalized);
        confidence.max(A::zero()).min(A::one())
    }

    /// Feed one newly observed data point to the model: measure the
    /// out-of-sample error of the prediction the current weights would have
    /// made for it, then perform the RLS update that folds it into the fit.
    ///
    /// Must be called once per sample, in arrival order.
    pub fn observe(&mut self, point: &StreamingDataPoint<A>) -> Result<()> {
        let dim = point.features.len();
        if dim == 0 {
            return Err(OptimError::InvalidConfig(
                "prediction model cannot observe a zero-dimensional sample".to_string(),
            ));
        }
        if self.featuredim == 0 {
            self.featuredim = dim;
        } else if self.featuredim != dim {
            // A dimensionality change invalidates the lag window (the lags
            // would be indexed past their end). Restart the window rather
            // than silently mixing incompatible observations.
            self.featuredim = dim;
            self.feature_lags.clear();
            self.target_lags.clear();
        }

        if self.feature_lags.len() == self.model_order {
            let alpha = to_a_or(ERROR_EWMA_ALPHA, A::one());
            let mut regressor = Array1::zeros(self.model_order + 1);
            regressor[self.model_order] = A::one();
            for coordinate in 0..dim {
                for lag in 0..self.model_order {
                    // `feature_lags` is newest-first and every entry has been
                    // length-checked against `featuredim` on insertion.
                    regressor[lag] = match self.feature_lags.get(lag) {
                        Some(vector) => match vector.get(coordinate) {
                            Some(&value) => value,
                            None => A::zero(),
                        },
                        None => A::zero(),
                    };
                }
                let observed = point.features[coordinate];
                let predicted = Self::dot(&self.weights, &regressor);
                let error = observed - predicted;
                if error.is_finite() {
                    self.error_ewma = self.error_ewma * (A::one() - alpha) + error.abs() * alpha;
                }
                self.scale_ewma = self.scale_ewma * (A::one() - alpha) + observed.abs() * alpha;
                self.rls_update(&regressor, error);
            }
        }

        self.feature_lags.push_front(point.features.clone());
        while self.feature_lags.len() > self.model_order {
            self.feature_lags.pop_back();
        }
        self.target_lags.push_front(point.target);
        while self.target_lags.len() > self.model_order {
            self.target_lags.pop_back();
        }
        Ok(())
    }

    /// One exponentially-weighted RLS rank-1 update for regressor `u` and
    /// prediction error `e`:
    ///
    /// ```text
    /// g = P u / (lambda + u^T P u)
    /// w = w + g e
    /// P = (P - g (u^T P)) / lambda
    /// ```
    ///
    /// The update is skipped (rather than propagating a non-finite value into
    /// the weights) whenever the denominator is not a usable positive number.
    fn rls_update(&mut self, regressor: &Array1<A>, error: A) {
        if !error.is_finite() {
            return;
        }
        let params = self.weights.len();
        // p_u = P * u
        let mut p_u = Array1::zeros(params);
        for i in 0..params {
            let mut acc = A::zero();
            for j in 0..params {
                acc = acc + self.inverse_correlation[[i, j]] * regressor[j];
            }
            p_u[i] = acc;
        }
        let denominator = self.forgetting_factor + Self::dot(regressor, &p_u);
        if !(denominator.is_finite() && denominator > A::zero()) {
            return;
        }
        let mut gain = Array1::zeros(params);
        for i in 0..params {
            gain[i] = p_u[i] / denominator;
        }
        for i in 0..params {
            let updated = self.weights[i] + gain[i] * error;
            if updated.is_finite() {
                self.weights[i] = updated;
            }
        }
        // P <- (P - gain * (P u)^T) / lambda, then symmetrised: RLS keeps `P`
        // symmetric in exact arithmetic, and re-symmetrising each step stops
        // rounding error from accumulating an asymmetric drift.
        let lambda = self.forgetting_factor;
        for i in 0..params {
            for j in 0..params {
                let value = (self.inverse_correlation[[i, j]] - gain[i] * p_u[j]) / lambda;
                self.inverse_correlation[[i, j]] = value;
            }
        }
        let two = to_a_or(2.0, A::one());
        for i in 0..params {
            for j in (i + 1)..params {
                let averaged =
                    (self.inverse_correlation[[i, j]] + self.inverse_correlation[[j, i]]) / two;
                self.inverse_correlation[[i, j]] = averaged;
                self.inverse_correlation[[j, i]] = averaged;
            }
        }
        let mut trace = A::zero();
        let mut healthy = true;
        for i in 0..params {
            let diagonal = self.inverse_correlation[[i, i]];
            if !diagonal.is_finite() {
                healthy = false;
                break;
            }
            trace = trace + diagonal.abs();
        }
        if !healthy || trace > to_a_or(RLS_MAX_TRACE, A::one()) {
            self.reset_inverse_correlation();
        }
        self.train_updates += 1;
    }

    fn reset_inverse_correlation(&mut self) {
        let params = self.weights.len();
        let initial = to_a_or(RLS_INITIAL_PRECISION, A::one());
        for i in 0..params {
            for j in 0..params {
                self.inverse_correlation[[i, j]] = if i == j { initial } else { A::zero() };
            }
        }
    }

    fn dot(left: &Array1<A>, right: &Array1<A>) -> A {
        left.iter()
            .zip(right.iter())
            .fold(A::zero(), |acc, (&a, &b)| acc + a * b)
    }

    /// Apply the fitted coefficients to a newest-first stack of lagged
    /// scalars, returning the one-step-ahead prediction.
    fn predict_scalar(&self, lags: &[A]) -> A {
        let mut accumulated = self.weights[self.model_order];
        for (lag, &value) in lags.iter().enumerate().take(self.model_order) {
            accumulated = accumulated + self.weights[lag] * value;
        }
        accumulated
    }

    /// Roll the fitted model forward `horizon` steps from `data`, feeding each
    /// prediction back in as the next lag (the standard iterated multi-step
    /// forecast).
    ///
    /// Returns fewer than `horizon` points when the model has not been
    /// trained yet, when `data` is shorter than the autoregressive order, or
    /// when the recursion stops producing finite values — never a fabricated
    /// copy of the present.
    pub fn predict(
        &self,
        data: &VecDeque<StreamingDataPoint<A>>,
        horizon: usize,
    ) -> Result<Vec<StreamingDataPoint<A>>> {
        let mut predictions = Vec::new();
        if data.len() < self.model_order || self.train_updates == 0 {
            return Ok(predictions);
        }
        let dim = match data.back() {
            Some(point) => point.features.len(),
            None => return Ok(predictions),
        };
        if dim == 0 {
            return Ok(predictions);
        }

        // Newest-first lag stacks, mirroring the training layout.
        let mut feature_lags: Vec<Array1<A>> = data
            .iter()
            .rev()
            .take(self.model_order)
            .map(|point| point.features.clone())
            .collect();
        if feature_lags.iter().any(|vector| vector.len() != dim) {
            return Err(OptimError::InvalidConfig(
                "prediction history contains samples of differing dimensionality".to_string(),
            ));
        }
        let mut target_lags: Vec<A> = Vec::with_capacity(self.model_order);
        let targets_available = data
            .iter()
            .rev()
            .take(self.model_order)
            .all(|point| point.target.is_some());
        if targets_available {
            for point in data.iter().rev().take(self.model_order) {
                target_lags.push(point.target.unwrap_or_else(A::zero));
            }
        }

        let step_interval = self.mean_arrival_interval(data);
        let confidence = self.confidence();
        let base_time = data.back().map(|point| point.timestamp);

        for step in 0..horizon {
            let mut predicted_features = Array1::zeros(dim);
            let mut lags = vec![A::zero(); self.model_order];
            let mut finite = true;
            for coordinate in 0..dim {
                for (lag, vector) in feature_lags.iter().enumerate() {
                    lags[lag] = vector[coordinate];
                }
                let value = self.predict_scalar(&lags);
                if !value.is_finite() {
                    finite = false;
                    break;
                }
                predicted_features[coordinate] = value;
            }
            if !finite {
                break;
            }
            let predicted_target = if targets_available {
                let value = self.predict_scalar(&target_lags);
                if value.is_finite() {
                    Some(value)
                } else {
                    None
                }
            } else {
                None
            };

            let mut metadata = HashMap::new();
            metadata.insert("predicted".to_string(), "true".to_string());
            metadata.insert("prediction_step".to_string(), (step + 1).to_string());
            let timestamp = match base_time {
                Some(instant) => instant + step_interval * (step as u32 + 1),
                None => Instant::now(),
            };
            predictions.push(StreamingDataPoint {
                features: predicted_features.clone(),
                target: predicted_target,
                timestamp,
                // The sample weight carries the model's measured confidence,
                // so a consumer that mixes predicted with observed samples
                // discounts the predictions by how well the model actually
                // fits rather than trusting them equally.
                weight: confidence,
                metadata,
            });

            feature_lags.insert(0, predicted_features);
            feature_lags.truncate(self.model_order);
            if targets_available {
                if let Some(target) = predicted_target {
                    target_lags.insert(0, target);
                    target_lags.truncate(self.model_order);
                }
            }
        }
        Ok(predictions)
    }

    /// Mean observed inter-arrival time of the most recent samples, used to
    /// stamp predicted points at realistic future instants instead of a
    /// hardcoded interval.
    fn mean_arrival_interval(&self, data: &VecDeque<StreamingDataPoint<A>>) -> Duration {
        let window = self.model_order_hint().min(data.len());
        let recent: Vec<Instant> = data
            .iter()
            .rev()
            .take(window.max(2))
            .map(|point| point.timestamp)
            .collect();
        if recent.len() < 2 {
            return Duration::ZERO;
        }
        let mut total = Duration::ZERO;
        let mut intervals = 0u32;
        // `recent` is newest-first, so the earlier instant is the later entry.
        for pair in recent.windows(2) {
            total += pair[0].saturating_duration_since(pair[1]);
            intervals += 1;
        }
        if intervals == 0 {
            Duration::ZERO
        } else {
            total / intervals
        }
    }

    /// Number of recent samples inspected when estimating the arrival
    /// interval. Deliberately a small multiple of the model order so a bursty
    /// stream is averaged over roughly the same span the model reasons about.
    fn model_order_hint(&self) -> usize {
        self.model_order.saturating_mul(2).max(2)
    }
}

#[cfg(test)]
mod prediction_model_training_tests {
    use super::*;
    use crate::streaming::types::primitives::{PredictiveStreamingEngine, StreamingConfig};

    /// Two superposed sinusoids of different frequencies. Any linear
    /// combination of two sinusoids satisfies an exact order-4 linear
    /// recurrence, so a correctly fitted AR(4) can reproduce this stream
    /// essentially exactly, while "repeat the previous value" cannot: the
    /// fast component advances 2 radians (about 115 degrees) per sample, so
    /// consecutive values are nowhere near each other.
    fn stream_point(t: usize) -> StreamingDataPoint<f64> {
        let tf = t as f64;
        let fast = (2.0 * tf).sin();
        let slow = 0.6 * (0.7 * tf + 0.4).sin();
        StreamingDataPoint {
            features: Array1::from_vec(vec![fast, slow]),
            target: Some(fast + slow),
            timestamp: Instant::now(),
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    fn train_model(samples: usize) -> (PredictionModel<f64>, VecDeque<StreamingDataPoint<f64>>) {
        let mut model = PredictionModel::new(4).expect("construct model");
        let mut history: VecDeque<StreamingDataPoint<f64>> = VecDeque::new();
        for t in 0..samples {
            let point = stream_point(t);
            model.observe(&point).expect("observe");
            history.push_back(point);
            while history.len() > 32 {
                history.pop_front();
            }
        }
        (model, history)
    }

    /// T8: the model must actually be *fitted* by the data it observes.
    /// A one-step-ahead forecast from the trained model has to beat the old
    /// implementation's behaviour — echoing the most recent observation — by
    /// a wide margin on a stream that genuinely has temporal structure.
    #[test]
    fn trained_model_forecasts_far_better_than_echoing_the_last_sample() {
        const TRAIN: usize = 400;
        let (model, history) = train_model(TRAIN);

        assert!(
            model.train_updates() > 0,
            "T8 regression: the model performed no training updates at all"
        );
        assert!(
            model.coefficients().iter().any(|&w| w.abs() > 1e-6),
            "T8 regression: model weights are still all zero after {TRAIN} \
             observed samples (they were never trained)"
        );

        let truth = stream_point(TRAIN);
        let last_observed = history.back().expect("history is non-empty").clone();

        let forecast = model.predict(&history, 1).expect("predict");
        assert_eq!(
            forecast.len(),
            1,
            "a trained model must produce the requested forecast"
        );

        let model_error: f64 = forecast[0]
            .features
            .iter()
            .zip(truth.features.iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum();
        let echo_error: f64 = last_observed
            .features
            .iter()
            .zip(truth.features.iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum();

        assert!(
            echo_error > 0.5,
            "test would be vacuous: the echo baseline must be genuinely bad \
             on this stream, but its error is only {echo_error}"
        );
        assert!(
            model_error < 1e-3,
            "T8 regression: trained model did not learn the autoregressive \
             structure (absolute forecast error {model_error})"
        );
        assert!(
            model_error < echo_error * 0.01,
            "T8 regression: trained model ({model_error}) is not decisively \
             better than echoing the last sample ({echo_error})"
        );
    }

    /// T8: prediction error must *decrease* as the model sees more data —
    /// the defining property of a model that is learning, and one no
    /// untrained echo predictor can have.
    #[test]
    fn prediction_error_converges_as_training_proceeds() {
        let mut model = PredictionModel::new(4).expect("construct model");
        for t in 0..20 {
            model.observe(&stream_point(t)).expect("observe");
        }
        let early_error = model.mean_absolute_error();
        let early_confidence = model.confidence();
        assert!(
            early_error > 0.0,
            "an untrained model must report a real, nonzero error"
        );

        for t in 20..400 {
            model.observe(&stream_point(t)).expect("observe");
        }
        let late_error = model.mean_absolute_error();
        let late_confidence = model.confidence();

        assert!(
            late_error < early_error * 0.01,
            "T8 regression: prediction error did not converge \
             (early={early_error}, late={late_error})"
        );
        assert!(
            late_confidence > early_confidence,
            "measured confidence must rise as the fit improves \
             (early={early_confidence}, late={late_confidence})"
        );
        assert!(
            late_confidence > 0.99,
            "a model that fits an exactly-representable stream should end up \
             highly confident, got {late_confidence}"
        );
    }

    /// T8: multi-step forecasts must actually roll the model forward. The old
    /// implementation emitted `horizon` identical copies of the most recent
    /// observation.
    #[test]
    fn multi_step_forecast_is_not_a_repeated_copy_of_the_present() {
        let (model, history) = train_model(400);
        let forecast = model.predict(&history, 5).expect("predict");
        assert_eq!(forecast.len(), 5);

        let last_observed = history.back().expect("non-empty").features.clone();
        for (step, point) in forecast.iter().enumerate() {
            let deviation: f64 = point
                .features
                .iter()
                .zip(last_observed.iter())
                .map(|(&p, &l)| (p - l).abs())
                .sum();
            assert!(
                deviation > 1e-3,
                "forecast step {step} is a copy of the last observation"
            );
            let truth = stream_point(400 + step);
            let error: f64 = point
                .features
                .iter()
                .zip(truth.features.iter())
                .map(|(&p, &t)| (p - t).abs())
                .sum();
            assert!(
                error < 1e-2,
                "iterated forecast step {step} diverged from the true \
                 continuation (error {error})"
            );
            assert!(
                point.metadata.get("predicted").map(String::as_str) == Some("true"),
                "predicted points must be tagged as predictions"
            );
        }
        // Successive forecast steps must differ from each other too.
        assert!(
            (forecast[0].features[0] - forecast[1].features[0]).abs() > 1e-3,
            "consecutive forecast steps are identical"
        );
    }

    /// T8: an untrained engine must emit *nothing* rather than a confident
    /// echo of the present. The old code returned `horizon` fabricated points
    /// as soon as three samples had arrived, with no training whatsoever.
    #[test]
    fn engine_emits_no_predictions_until_the_fit_is_trustworthy() {
        let config = StreamingConfig::default();
        let mut engine = PredictiveStreamingEngine::<f64>::new(&config).expect("construct engine");

        let warmup: Vec<StreamingDataPoint<f64>> = (0..6).map(stream_point).collect();
        let predictions = engine.predict_next(&warmup).expect("predict_next");
        assert!(
            predictions.is_empty(),
            "T8 regression: engine emitted {} predictions from a model that \
             has barely seen any data (confidence {})",
            predictions.len(),
            engine.prediction_confidence()
        );

        // Once genuinely fitted, it must start forecasting.
        let rest: Vec<StreamingDataPoint<f64>> = (6..400).map(stream_point).collect();
        let predictions = engine.predict_next(&rest).expect("predict_next");
        assert!(
            !predictions.is_empty(),
            "engine never started forecasting despite a perfectly learnable \
             stream (confidence {})",
            engine.prediction_confidence()
        );
        assert!(
            engine.prediction_error() < 1e-3,
            "engine reports an implausibly large fitted error: {}",
            engine.prediction_error()
        );
        // The confidence carried on each predicted sample must be the real
        // measured confidence, not a placeholder 1.0.
        let confidence = engine.prediction_confidence();
        assert!(
            (predictions[0].weight - confidence).abs() < 1e-12,
            "predicted sample weight must carry the model's measured confidence"
        );
    }

    #[test]
    fn invalid_model_configuration_is_an_error_not_a_panic() {
        assert!(PredictionModel::<f64>::new(0).is_err());
        assert!(PredictionModel::<f64>::with_forgetting_factor(3, 0.0).is_err());
        assert!(PredictionModel::<f64>::with_forgetting_factor(3, 1.5).is_err());
        let mut model = PredictionModel::<f64>::new(2).expect("construct");
        let empty = StreamingDataPoint {
            features: Array1::from_vec(vec![]),
            target: None,
            timestamp: Instant::now(),
            weight: 1.0,
            metadata: HashMap::new(),
        };
        assert!(model.observe(&empty).is_err());
    }
}
