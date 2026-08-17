//! Landscape-and-adaptation driven performance prediction.
//!
//! This is the `TransformerPerformancePredictor` façade that
//! `AdaptiveTransformerEnhancement` holds. The learning machinery it delegates
//! to lives in [`crate::adaptive::predictor`]; this file is only the bridge
//! between the crate's landscape/adaptation types and that machinery.
//!
//! It was split out of `adaptive/types.rs` to keep every file under the
//! 2000-line cap.

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::Result;

use super::predictor::{
    PredictionCache, PredictionFeatures, PredictorFitReport, PredictorNetwork, PredictorSample,
    PREDICTION_FEATURE_COUNT,
};
use super::types::{
    ArchitectureAdaptation, ArchitectureChange, LandscapeAnalysis, OptimizationStrategy,
    PerformancePrediction, UncertaintyMethod,
};
use super::AdaptiveConfig;

/// Performance predictor for transformer variants.
///
/// Wraps the real predictor from [`crate::adaptive::predictor`]: a
/// Xavier-initialized fixed random feature map with ridge-fitted output heads,
/// a bounded prediction cache with real hit accounting, and a predictive
/// standard deviation derived from the ridge posterior.
///
/// Before [`Self::train`] has been called, [`Self::predict_improvement`] answers
/// "no information" (zero means, unit uncertainty, zero confidence) instead of
/// the constants `0.15 / 0.92 / 0.85 / 0.05` it used to fabricate.
#[derive(Debug)]
pub struct TransformerPerformancePredictor<
    T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
> {
    /// Neural predictor network (fixed random feature map + learned heads)
    predictor_network: PredictorNetwork<T>,
    /// Prediction cache
    prediction_cache: PredictionCache,
    /// Uncertainty estimation method (recorded so the choice is inspectable)
    uncertainty_method: UncertaintyMethod,
    /// Prediction horizon, taken from the configuration; also the number of
    /// history samples the analysis confidence saturates at.
    prediction_horizon: usize,
}
impl<T: Float + Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static>
    TransformerPerformancePredictor<T>
{
    /// Build the predictor from the adaptive configuration.
    ///
    /// The configuration is genuinely read: `prediction_horizon` sizes both the
    /// hidden feature map and the cache, and `adaptation_lr` supplies the ridge
    /// coefficient (a smaller adaptation rate means a more conservative, more
    /// strongly regularized fit).
    pub fn new(config: &AdaptiveConfig<T>) -> Result<Self> {
        let horizon = config.prediction_horizon.max(1);
        // Feature-map width scales with the horizon, floored so the map is
        // always wider than the raw feature vector and capped so the ridge solve
        // stays small.
        let hidden = (2 * PREDICTION_FEATURE_COUNT).max(horizon).min(128);
        let feature_width = PREDICTION_FEATURE_COUNT.max(horizon.min(64));
        let ridge = {
            let lr = config.adaptation_lr.to_f64().unwrap_or(0.0).abs();
            let lambda = if lr > 0.0 { lr * 1e-2 } else { 1e-4 };
            scirs2_core::numeric::NumCast::from(lambda.clamp(1e-8, 1.0)).unwrap_or_else(|| T::one())
        };
        Ok(Self {
            predictor_network: PredictorNetwork::new(
                vec![PREDICTION_FEATURE_COUNT, hidden, feature_width],
                ridge,
            )?,
            prediction_cache: PredictionCache::new(8 * horizon),
            uncertainty_method: UncertaintyMethod::Ensemble,
            prediction_horizon: horizon,
        })
    }

    /// Fit the prediction heads on observed `(landscape+adaptation, outcome)`
    /// pairs. Returns the training RMSE of each head.
    pub fn train(&mut self, samples: &[PredictorSample]) -> Result<PredictorFitReport> {
        self.prediction_cache.clear();
        self.predictor_network.fit(samples)
    }

    /// Whether the heads have been fitted.
    pub fn is_trained(&self) -> bool {
        self.predictor_network.is_trained()
    }

    /// Observed cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        self.prediction_cache.hit_rate()
    }

    /// Uncertainty estimation method in use.
    pub fn uncertainty_method(&self) -> UncertaintyMethod {
        self.uncertainty_method
    }

    /// Prediction horizon this predictor was configured with.
    pub fn prediction_horizon(&self) -> usize {
        self.prediction_horizon
    }

    /// Turn a landscape analysis plus a proposed adaptation into the real
    /// feature vector the network consumes.
    pub fn extract_features(
        landscape: &LandscapeAnalysis<T>,
        adaptation: &ArchitectureAdaptation<T>,
    ) -> PredictionFeatures {
        let complexity = landscape.complexity.to_f64().unwrap_or(0.0);
        let difficulty = landscape.difficulty.to_f64().unwrap_or(0.0);

        // Strategy family, folded into three slots.
        let mut conservative = 0.0;
        let mut aggressive = 0.0;
        let mut exploratory = 0.0;
        for strategy in &landscape.recommended_strategies {
            match strategy {
                OptimizationStrategy::Conservative | OptimizationStrategy::Exploitative => {
                    conservative += 1.0
                }
                OptimizationStrategy::Aggressive => aggressive += 1.0,
                OptimizationStrategy::Exploratory => exploratory += 1.0,
                OptimizationStrategy::Adaptive => {
                    // Adaptive sits between the two extremes.
                    conservative += 0.5;
                    aggressive += 0.5;
                }
            }
        }
        let strategy_total = (conservative + aggressive + exploratory).max(1.0);

        // Architecture proposal: start from the adapted config, then let the
        // explicit change list override it (a change list is what the adapter
        // actually promises to apply).
        let mut layers = adaptation.adapted_config.num_transformer_layers as f64;
        let mut hidden = adaptation.adapted_config.model_dimension as f64;
        let mut heads = adaptation.adapted_config.num_attention_heads as f64;
        let mut dropout = adaptation.adapted_config.dropout_rate;
        for change in &adaptation.changes {
            match change {
                ArchitectureChange::LayerCountChange(n) => layers = *n as f64,
                ArchitectureChange::HiddenSizeChange(n) => hidden = *n as f64,
                ArchitectureChange::AttentionHeadChange(n) => heads = *n as f64,
                ArchitectureChange::DropoutChange(d) => dropout = *d,
                ArchitectureChange::ActivationChange(_) => {}
            }
        }

        PredictionFeatures {
            complexity,
            difficulty,
            landscape_confidence: landscape.confidence.to_f64().unwrap_or(0.0),
            expected_improvement: adaptation.expected_improvement.to_f64().unwrap_or(0.0),
            adaptation_confidence: adaptation.confidence.to_f64().unwrap_or(0.0),
            change_count: (adaptation.changes.len() as f64 / 8.0).min(1.0),
            layer_count: (layers / 24.0).min(1.0),
            hidden_size: (hidden / 2048.0).min(1.0),
            head_count: (heads / 32.0).min(1.0),
            dropout: dropout.clamp(0.0, 1.0),
            interaction: complexity * difficulty,
            complexity_sq: complexity * complexity,
            difficulty_sq: difficulty * difficulty,
            strategy_conservative: conservative / strategy_total,
            strategy_aggressive: aggressive / strategy_total,
            strategy_exploratory: exploratory / strategy_total,
        }
    }

    pub fn predict_improvement(
        &mut self,
        landscape: &LandscapeAnalysis<T>,
        adaptation: &ArchitectureAdaptation<T>,
    ) -> Result<PerformancePrediction<T>> {
        let features = Self::extract_features(landscape, adaptation);
        let key = features.cache_key();
        let output = match self.prediction_cache.get(&key) {
            Some(cached) => cached,
            None => {
                let fresh = self.predictor_network.predict(&features)?;
                self.prediction_cache.insert(key, fresh.clone());
                fresh
            }
        };

        let cast =
            |v: f64| -> T { scirs2_core::numeric::NumCast::from(v).unwrap_or_else(|| T::zero()) };
        Ok(PerformancePrediction {
            convergence_improvement: cast(output.convergence_improvement),
            final_performance: cast(output.final_performance),
            confidence: cast(output.confidence),
            uncertainty: cast(output.uncertainty),
        })
    }
}
