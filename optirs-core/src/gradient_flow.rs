//! Gradient Flow Analysis Module
//!
//! Provides tools for analyzing gradient flow through neural network layers,
//! detecting vanishing/exploding gradients, and generating visual reports.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Configuration for gradient flow analysis
#[derive(Debug, Clone)]
pub struct GradientFlowConfig {
    /// Threshold below which gradients are considered vanishing
    pub vanishing_threshold: f64,
    /// Threshold above which gradients are considered exploding
    pub exploding_threshold: f64,
    /// Number of histogram bins for gradient magnitude distribution
    pub histogram_bins: usize,
    /// Maximum number of historical records to keep per layer
    pub max_history: usize,
}

impl Default for GradientFlowConfig {
    fn default() -> Self {
        Self {
            vanishing_threshold: 1e-7,
            exploding_threshold: 1e3,
            histogram_bins: 50,
            max_history: 100,
        }
    }
}

/// Per-layer gradient statistics
#[derive(Debug, Clone)]
pub struct LayerGradientStats<A> {
    /// Name of the layer
    pub layer_name: String,
    /// Mean gradient norm
    pub mean_norm: A,
    /// Maximum gradient norm
    pub max_norm: A,
    /// Minimum gradient norm
    pub min_norm: A,
    /// Variance of gradient magnitudes
    pub variance: A,
    /// Fraction of near-zero elements (sparsity)
    pub sparsity: A,
    /// Histogram of gradient magnitude distribution
    pub histogram: Vec<usize>,
}

/// Overall gradient health status
#[derive(Debug, Clone, PartialEq)]
pub enum GradientHealth {
    /// All layers have healthy gradient flow
    Healthy,
    /// Some layers show concerning gradient behavior
    Warning,
    /// Critical gradient flow issues detected
    Critical,
}

/// Comprehensive gradient health report
#[derive(Debug, Clone)]
pub struct GradientHealthReport {
    /// Layers with vanishing gradients
    pub vanishing_layers: Vec<String>,
    /// Layers with exploding gradients
    pub exploding_layers: Vec<String>,
    /// Layers with healthy gradient flow
    pub healthy_layers: Vec<String>,
    /// Overall health assessment
    pub overall_health: GradientHealth,
    /// Actionable recommendations
    pub recommendations: Vec<String>,
}

/// Gradient flow analyzer for monitoring and diagnosing gradient behavior
pub struct GradientFlowAnalyzer<A> {
    /// Configuration parameters
    config: GradientFlowConfig,
    /// Historical statistics per layer
    layer_stats: HashMap<String, Vec<LayerGradientStats<A>>>,
    /// Ordering of layers for rendering
    layer_order: Vec<String>,
}

impl<A> GradientFlowAnalyzer<A>
where
    A: Float + ScalarOperand + Debug + std::iter::Sum,
{
    /// Create a new gradient flow analyzer with the given configuration
    pub fn new(config: GradientFlowConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
            layer_order: Vec::new(),
        }
    }

    /// Record gradients for a layer and compute statistics
    ///
    /// Computes mean norm, max norm, min norm, variance, sparsity, and
    /// a histogram of gradient magnitudes. The results are stored in the
    /// internal history for later analysis.
    pub fn record_gradients(
        &mut self,
        layer_name: &str,
        gradients: &Array1<A>,
    ) -> Result<LayerGradientStats<A>> {
        let len = gradients.len();
        if len == 0 {
            return Err(OptimError::InvalidParameter(
                "Gradients array must not be empty".to_string(),
            ));
        }

        let len_a = A::from(len).ok_or_else(|| {
            OptimError::ComputationError("Failed to convert length to float".to_string())
        })?;

        // Compute absolute values for magnitude analysis
        let abs_grads: Vec<A> = gradients.iter().map(|&g| g.abs()).collect();

        // Mean norm
        let sum: A = abs_grads.iter().copied().sum();
        let mean_norm = sum / len_a;

        // Max and min norms
        let max_norm = abs_grads
            .iter()
            .copied()
            .fold(A::neg_infinity(), |a, b| if b > a { b } else { a });
        let min_norm = abs_grads
            .iter()
            .copied()
            .fold(A::infinity(), |a, b| if b < a { b } else { a });

        // Variance: E[x^2] - (E[x])^2
        let sum_sq: A = abs_grads.iter().map(|&g| g * g).sum();
        let mean_sq = sum_sq / len_a;
        let variance = mean_sq - mean_norm * mean_norm;
        // Clamp to zero in case of floating point issues
        let variance = if variance < A::zero() {
            A::zero()
        } else {
            variance
        };

        // Sparsity: fraction of elements with magnitude below vanishing threshold
        let vanishing_thresh = A::from(self.config.vanishing_threshold).ok_or_else(|| {
            OptimError::ComputationError(
                "Failed to convert vanishing threshold to float".to_string(),
            )
        })?;
        let near_zero_count = abs_grads.iter().filter(|&&g| g < vanishing_thresh).count();
        let sparsity = A::from(near_zero_count).ok_or_else(|| {
            OptimError::ComputationError("Failed to convert count to float".to_string())
        })? / len_a;

        // Histogram of gradient magnitudes
        let histogram = self.compute_histogram(&abs_grads, max_norm)?;

        let stats = LayerGradientStats {
            layer_name: layer_name.to_string(),
            mean_norm,
            max_norm,
            min_norm,
            variance,
            sparsity,
            histogram,
        };

        // Track layer ordering
        if !self.layer_order.contains(&layer_name.to_string()) {
            self.layer_order.push(layer_name.to_string());
        }

        // Store in history, respecting max_history
        let history = self.layer_stats.entry(layer_name.to_string()).or_default();
        history.push(stats.clone());
        if history.len() > self.config.max_history {
            history.remove(0);
        }

        Ok(stats)
    }

    /// Compute a histogram of gradient magnitudes
    fn compute_histogram(&self, abs_grads: &[A], max_val: A) -> Result<Vec<usize>> {
        let bins = self.config.histogram_bins;
        let mut histogram = vec![0usize; bins];

        if max_val <= A::zero() {
            // All zeros, put everything in first bin
            histogram[0] = abs_grads.len();
            return Ok(histogram);
        }

        for &val in abs_grads {
            let normalized = val / max_val;
            let bin_idx = (normalized
                * A::from(bins).ok_or_else(|| {
                    OptimError::ComputationError("Failed to convert bins to float".to_string())
                })?)
            .to_f64()
            .ok_or_else(|| OptimError::ComputationError("Failed to convert to f64".to_string()))?;
            let bin_idx = (bin_idx as usize).min(bins - 1);
            histogram[bin_idx] += 1;
        }

        Ok(histogram)
    }

    /// Detect layers with vanishing gradients
    ///
    /// Returns names of layers whose most recent mean gradient norm
    /// is below the configured vanishing threshold.
    pub fn detect_vanishing_gradients(&self) -> Vec<String> {
        let threshold = self.config.vanishing_threshold;
        let mut vanishing = Vec::new();

        for (name, stats_history) in &self.layer_stats {
            if let Some(latest) = stats_history.last() {
                let mean_f64 = latest.mean_norm.to_f64().unwrap_or(0.0);
                if mean_f64 < threshold {
                    vanishing.push(name.clone());
                }
            }
        }

        vanishing.sort();
        vanishing
    }

    /// Detect layers with exploding gradients
    ///
    /// Returns names of layers whose most recent max gradient norm
    /// is above the configured exploding threshold.
    pub fn detect_exploding_gradients(&self) -> Vec<String> {
        let threshold = self.config.exploding_threshold;
        let mut exploding = Vec::new();

        for (name, stats_history) in &self.layer_stats {
            if let Some(latest) = stats_history.last() {
                let max_f64 = latest.max_norm.to_f64().unwrap_or(0.0);
                if max_f64 > threshold {
                    exploding.push(name.clone());
                }
            }
        }

        exploding.sort();
        exploding
    }

    /// Generate a comprehensive gradient health report
    ///
    /// Analyzes all tracked layers and produces a report with:
    /// - Lists of vanishing, exploding, and healthy layers
    /// - An overall health assessment
    /// - Actionable recommendations for fixing gradient issues
    pub fn get_health_report(&self) -> GradientHealthReport {
        let vanishing = self.detect_vanishing_gradients();
        let exploding = self.detect_exploding_gradients();

        let mut healthy = Vec::new();
        for name in &self.layer_order {
            if !vanishing.contains(name) && !exploding.contains(name) {
                healthy.push(name.clone());
            }
        }

        let overall_health = if !exploding.is_empty() {
            GradientHealth::Critical
        } else if !vanishing.is_empty() {
            if vanishing.len() > self.layer_order.len() / 2 {
                GradientHealth::Critical
            } else {
                GradientHealth::Warning
            }
        } else {
            GradientHealth::Healthy
        };

        let mut recommendations = Vec::new();

        if !vanishing.is_empty() {
            recommendations.push(format!(
                "Vanishing gradients detected in {} layer(s): consider using residual connections, \
                 batch normalization, or switching to ReLU-family activations.",
                vanishing.len()
            ));
            recommendations
                .push("Consider using gradient scaling or a smaller model depth.".to_string());
        }

        if !exploding.is_empty() {
            recommendations.push(format!(
                "Exploding gradients detected in {} layer(s): apply gradient clipping \
                 (e.g., max norm clipping) or reduce learning rate.",
                exploding.len()
            ));
            recommendations.push(
                "Consider weight initialization with smaller variance (e.g., He or Xavier init)."
                    .to_string(),
            );
        }

        if vanishing.is_empty() && exploding.is_empty() {
            recommendations.push("Gradient flow appears healthy across all layers.".to_string());
        }

        GradientHealthReport {
            vanishing_layers: vanishing,
            exploding_layers: exploding,
            healthy_layers: healthy,
            overall_health,
            recommendations,
        }
    }

    /// Render an SVG flow chart showing gradient magnitudes across layers
    ///
    /// Produces an SVG string with bars representing mean gradient norms
    /// for each layer, color-coded by health status.
    pub fn render_flow_chart(&self) -> Result<String> {
        if self.layer_order.is_empty() {
            return Err(OptimError::InvalidState(
                "No gradient data recorded yet".to_string(),
            ));
        }

        let vanishing = self.detect_vanishing_gradients();
        let exploding = self.detect_exploding_gradients();

        let bar_width = 40;
        let bar_spacing = 10;
        let margin_left = 150;
        let margin_top = 40;
        let chart_width = 400;
        let num_layers = self.layer_order.len();
        let total_height = margin_top + num_layers * (bar_width + bar_spacing) + 40;
        let total_width = margin_left + chart_width + 60;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            total_width, total_height, total_width, total_height
        );
        svg.push('\n');

        // Title
        svg.push_str(&format!(
            r#"  <text x="{}" y="25" text-anchor="middle" font-size="16" font-weight="bold">Gradient Flow Analysis</text>"#,
            total_width / 2
        ));
        svg.push('\n');

        // Find max mean_norm for scaling
        let mut max_mean = 0.0f64;
        for name in &self.layer_order {
            if let Some(history) = self.layer_stats.get(name) {
                if let Some(latest) = history.last() {
                    let val = latest.mean_norm.to_f64().unwrap_or(0.0);
                    if val > max_mean {
                        max_mean = val;
                    }
                }
            }
        }
        if max_mean <= 0.0 {
            max_mean = 1.0;
        }

        for (i, name) in self.layer_order.iter().enumerate() {
            let y = margin_top + i * (bar_width + bar_spacing);

            let mean_val = self
                .layer_stats
                .get(name)
                .and_then(|h| h.last())
                .map(|s| s.mean_norm.to_f64().unwrap_or(0.0))
                .unwrap_or(0.0);

            let bar_len = ((mean_val / max_mean) * chart_width as f64).max(1.0) as usize;

            let color = if exploding.contains(name) {
                "#ff4444" // Red for exploding
            } else if vanishing.contains(name) {
                "#ffaa00" // Orange for vanishing
            } else {
                "#44bb44" // Green for healthy
            };

            // Layer label
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" text-anchor="end" font-size="12" dominant-baseline="middle">{}</text>"#,
                margin_left - 10,
                y + bar_width / 2,
                name
            ));
            svg.push('\n');

            // Bar
            svg.push_str(&format!(
                r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="3" ry="3"/>"#,
                margin_left, y, bar_len, bar_width, color
            ));
            svg.push('\n');

            // Value label
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" font-size="10" dominant-baseline="middle">{:.2e}</text>"#,
                margin_left + bar_len + 5,
                y + bar_width / 2,
                mean_val
            ));
            svg.push('\n');
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Get the history of gradient statistics for a specific layer
    pub fn get_layer_history(&self, layer_name: &str) -> Option<&Vec<LayerGradientStats<A>>> {
        self.layer_stats.get(layer_name)
    }

    /// Clear all recorded gradient history
    pub fn clear_history(&mut self) {
        self.layer_stats.clear();
        self.layer_order.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_record_gradients_basic() {
        let config = GradientFlowConfig::default();
        let mut analyzer = GradientFlowAnalyzer::<f64>::new(config);

        let gradients = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);
        let stats = analyzer
            .record_gradients("layer1", &gradients)
            .expect("Should record gradients");

        assert_eq!(stats.layer_name, "layer1");
        // Mean of |0.1, 0.2, 0.3, 0.4, 0.5| = 0.3
        assert!((stats.mean_norm - 0.3).abs() < 1e-10);
        // Max = 0.5
        assert!((stats.max_norm - 0.5).abs() < 1e-10);
        // Min = 0.1
        assert!((stats.min_norm - 0.1).abs() < 1e-10);
        // Sparsity should be 0 (no values below 1e-7)
        assert!((stats.sparsity - 0.0).abs() < 1e-10);
        // Histogram should sum to gradient count
        let hist_sum: usize = stats.histogram.iter().sum();
        assert_eq!(hist_sum, 5);

        // Verify history was stored
        let history = analyzer.get_layer_history("layer1");
        assert!(history.is_some());
        assert_eq!(history.map(|h| h.len()).unwrap_or(0), 1);
    }

    #[test]
    fn test_detect_vanishing_gradients() {
        let config = GradientFlowConfig {
            vanishing_threshold: 1e-7,
            ..Default::default()
        };
        let mut analyzer = GradientFlowAnalyzer::<f64>::new(config);

        // Normal gradients
        let normal_grads = Array1::from_vec(vec![0.01, 0.02, 0.015, 0.008]);
        analyzer
            .record_gradients("healthy_layer", &normal_grads)
            .expect("Should record");

        // Vanishing gradients
        let tiny_grads = Array1::from_vec(vec![1e-9, 1e-10, 1e-8, 1e-11]);
        analyzer
            .record_gradients("vanishing_layer", &tiny_grads)
            .expect("Should record");

        let vanishing = analyzer.detect_vanishing_gradients();
        assert!(vanishing.contains(&"vanishing_layer".to_string()));
        assert!(!vanishing.contains(&"healthy_layer".to_string()));
    }

    #[test]
    fn test_detect_exploding_gradients() {
        let config = GradientFlowConfig {
            exploding_threshold: 1e3,
            ..Default::default()
        };
        let mut analyzer = GradientFlowAnalyzer::<f64>::new(config);

        // Normal gradients
        let normal_grads = Array1::from_vec(vec![0.5, 1.0, 0.3, 0.8]);
        analyzer
            .record_gradients("normal_layer", &normal_grads)
            .expect("Should record");

        // Exploding gradients
        let huge_grads = Array1::from_vec(vec![5000.0, 10000.0, 3000.0, 8000.0]);
        analyzer
            .record_gradients("exploding_layer", &huge_grads)
            .expect("Should record");

        let exploding = analyzer.detect_exploding_gradients();
        assert!(exploding.contains(&"exploding_layer".to_string()));
        assert!(!exploding.contains(&"normal_layer".to_string()));
    }

    #[test]
    fn test_health_report_generation() {
        let config = GradientFlowConfig::default();
        let mut analyzer = GradientFlowAnalyzer::<f64>::new(config);

        // Healthy layer
        let healthy = Array1::from_vec(vec![0.01, 0.02, 0.015]);
        analyzer
            .record_gradients("fc1", &healthy)
            .expect("Should record");

        // Vanishing layer
        let vanishing = Array1::from_vec(vec![1e-10, 1e-11, 1e-9]);
        analyzer
            .record_gradients("fc2", &vanishing)
            .expect("Should record");

        // Exploding layer
        let exploding = Array1::from_vec(vec![5000.0, 10000.0, 8000.0]);
        analyzer
            .record_gradients("fc3", &exploding)
            .expect("Should record");

        let report = analyzer.get_health_report();

        assert!(report.vanishing_layers.contains(&"fc2".to_string()));
        assert!(report.exploding_layers.contains(&"fc3".to_string()));
        assert!(report.healthy_layers.contains(&"fc1".to_string()));
        assert_eq!(report.overall_health, GradientHealth::Critical);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_render_flow_chart_svg() {
        let config = GradientFlowConfig::default();
        let mut analyzer = GradientFlowAnalyzer::<f64>::new(config);

        let grads1 = Array1::from_vec(vec![0.01, 0.02, 0.015]);
        let grads2 = Array1::from_vec(vec![0.005, 0.003, 0.004]);
        let grads3 = Array1::from_vec(vec![0.1, 0.08, 0.12]);

        analyzer
            .record_gradients("conv1", &grads1)
            .expect("Should record");
        analyzer
            .record_gradients("conv2", &grads2)
            .expect("Should record");
        analyzer
            .record_gradients("fc1", &grads3)
            .expect("Should record");

        let svg = analyzer
            .render_flow_chart()
            .expect("Should render flow chart");

        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("conv1"));
        assert!(svg.contains("conv2"));
        assert!(svg.contains("fc1"));
        assert!(svg.contains("Gradient Flow Analysis"));
    }
}
