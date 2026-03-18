//! Metrics collection for tracking optimization performance in WASM.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// A single optimization step's metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub learning_rate: f64,
    pub grad_norm: f64,
    pub param_norm: f64,
    pub update_norm: f64,
    pub step: usize,
}

/// Metrics collector for tracking optimization across multiple optimizers
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct WasmMetricsCollector {
    optimizers: HashMap<String, Vec<StepMetrics>>,
    step_count: HashMap<String, usize>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmMetricsCollector {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            optimizers: HashMap::new(),
            step_count: HashMap::new(),
        }
    }

    /// Register a new optimizer for tracking
    pub fn register_optimizer(&mut self, name: &str) {
        self.optimizers.entry(name.to_string()).or_default();
        self.step_count.entry(name.to_string()).or_insert(0);
    }

    /// Record metrics for an optimization step
    pub fn update(
        &mut self,
        name: &str,
        learning_rate: f64,
        gradients: &[f64],
        params_before: &[f64],
        params_after: &[f64],
    ) {
        let grad_norm = gradients.iter().map(|x| x * x).sum::<f64>().sqrt();
        let param_norm = params_after.iter().map(|x| x * x).sum::<f64>().sqrt();
        let update_norm = params_before
            .iter()
            .zip(params_after.iter())
            .map(|(a, b)| (b - a) * (b - a))
            .sum::<f64>()
            .sqrt();

        let step = self.step_count.get(name).copied().unwrap_or(0);

        let metrics = StepMetrics {
            learning_rate,
            grad_norm,
            param_norm,
            update_norm,
            step,
        };

        self.optimizers
            .entry(name.to_string())
            .or_default()
            .push(metrics);
        *self.step_count.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Get a summary report as a string
    pub fn summary_report(&self) -> String {
        let mut report = String::from("=== Optimization Metrics Summary ===\n");
        for (name, metrics) in &self.optimizers {
            report.push_str(&format!("\nOptimizer: {}\n", name));
            report.push_str(&format!("  Steps: {}\n", metrics.len()));
            if let Some(last) = metrics.last() {
                report.push_str(&format!("  Last LR: {:.6e}\n", last.learning_rate));
                report.push_str(&format!("  Last Grad Norm: {:.6e}\n", last.grad_norm));
                report.push_str(&format!("  Last Update Norm: {:.6e}\n", last.update_norm));
            }
            if !metrics.is_empty() {
                let avg_grad_norm =
                    metrics.iter().map(|m| m.grad_norm).sum::<f64>() / metrics.len() as f64;
                let avg_update_norm =
                    metrics.iter().map(|m| m.update_norm).sum::<f64>() / metrics.len() as f64;
                report.push_str(&format!("  Avg Grad Norm: {:.6e}\n", avg_grad_norm));
                report.push_str(&format!("  Avg Update Norm: {:.6e}\n", avg_update_norm));
            }
        }
        report
    }

    /// Get summary as JSON string
    pub fn summary_json(&self) -> Result<String, String> {
        serde_json::to_string(&self.optimizers).map_err(|e| e.to_string())
    }

    /// Get metrics for a specific optimizer as JSON
    pub fn optimizer_metrics_json(&self, name: &str) -> Result<String, String> {
        match self.optimizers.get(name) {
            Some(metrics) => serde_json::to_string(metrics).map_err(|e| e.to_string()),
            None => Err(format!("Optimizer '{}' not registered", name)),
        }
    }

    /// Get total number of tracked optimizers
    pub fn optimizer_count(&self) -> usize {
        self.optimizers.len()
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.optimizers.clear();
        self.step_count.clear();
    }

    /// Clear metrics for a specific optimizer
    pub fn clear_optimizer(&mut self, name: &str) {
        if let Some(metrics) = self.optimizers.get_mut(name) {
            metrics.clear();
        }
        if let Some(count) = self.step_count.get_mut(name) {
            *count = 0;
        }
    }
}

impl Default for WasmMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
