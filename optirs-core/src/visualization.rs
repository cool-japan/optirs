// Visualization tools for optimization metrics
//
// This module provides comprehensive visualization capabilities for tracking
// optimization progress, comparing optimizers, and analyzing training dynamics.

#[allow(dead_code)]
use crate::error::{OptimError, Result};
use std::collections::{HashMap, VecDeque};
use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuration for visualization output
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Output directory for saved plots
    pub output_dir: String,

    /// Maximum number of data points to display
    pub max_points: usize,

    /// Update frequency for real-time plots (in steps)
    pub update_frequency: usize,

    /// Enable interactive HTML output
    pub interactive_html: bool,

    /// Enable SVG output for publication
    pub svg_output: bool,

    /// Color scheme for plots
    pub color_scheme: ColorScheme,

    /// Default figure size (width, height)
    pub figure_size: (u32, u32),

    /// DPI for raster outputs
    pub dpi: u32,

    /// Enable grid lines
    pub show_grid: bool,

    /// Enable legends
    pub show_legend: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_dir: "optimization_plots".to_string(),
            max_points: 10000,
            update_frequency: 100,
            interactive_html: true,
            svg_output: false,
            color_scheme: ColorScheme::Default,
            figure_size: (800, 600),
            dpi: 300,
            show_grid: true,
            show_legend: true,
        }
    }
}

/// Color schemes for plots
#[derive(Debug, Clone, Copy)]
pub enum ColorScheme {
    Default,
    Dark,
    Colorblind,
    Publication,
    Vibrant,
}

/// Optimization metric for tracking
#[derive(Debug, Clone)]
pub struct OptimizationMetric {
    /// Metric name
    pub name: String,

    /// Metric values over time
    pub values: VecDeque<f64>,

    /// Timestamps for each value
    pub timestamps: VecDeque<u64>,

    /// Step numbers
    pub steps: VecDeque<usize>,

    /// Target value (if any)
    pub target: Option<f64>,

    /// Whether higher values are better
    pub higher_isbetter: bool,

    /// Units for display
    pub units: String,

    /// Smoothing window size
    pub smoothing_window: usize,
}

impl OptimizationMetric {
    /// Create new metric tracker
    pub fn new(name: String, higher_isbetter: bool, units: String) -> Self {
        Self {
            name,
            values: VecDeque::new(),
            timestamps: VecDeque::new(),
            steps: VecDeque::new(),
            target: None,
            higher_isbetter,
            units,
            smoothing_window: 10,
        }
    }

    /// Add a new value
    pub fn add_value(&mut self, value: f64, step: usize) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.values.push_back(value);
        self.timestamps.push_back(timestamp);
        self.steps.push_back(step);

        // Keep only recent values to avoid memory issues
        while self.values.len() > 50000 {
            self.values.pop_front();
            self.timestamps.pop_front();
            self.steps.pop_front();
        }
    }

    /// Get smoothed values
    pub fn get_smoothed_values(&self) -> Vec<f64> {
        if self.values.len() < self.smoothing_window {
            return self.values.iter().copied().collect();
        }

        let mut smoothed = Vec::new();
        let window = self.smoothing_window.min(self.values.len());

        for i in 0..self.values.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(self.values.len());

            let sum: f64 = self.values.range(start..end).sum();
            let avg = sum / (end - start) as f64;
            smoothed.push(avg);
        }

        smoothed
    }

    /// Get recent improvement
    pub fn get_recent_improvement(&self, windowsize: usize) -> Option<f64> {
        if self.values.len() < windowsize * 2 {
            return None;
        }

        let recent_avg: f64 =
            self.values.iter().rev().take(windowsize).sum::<f64>() / windowsize as f64;
        let older_avg: f64 = self
            .values
            .iter()
            .rev()
            .skip(windowsize)
            .take(windowsize)
            .sum::<f64>()
            / windowsize as f64;

        Some(if self.higher_isbetter {
            recent_avg - older_avg
        } else {
            older_avg - recent_avg
        })
    }
}

/// Optimizer comparison data
#[derive(Debug, Clone)]
pub struct OptimizerComparison {
    /// Optimizer name
    pub name: String,

    /// Performance metrics
    pub metrics: HashMap<String, Vec<f64>>,

    /// Hyperparameters used
    pub hyperparameters: HashMap<String, f64>,

    /// Total training time
    pub training_time: Duration,

    /// Memory usage statistics
    pub memory_stats: MemoryStats,

    /// Convergence information
    pub convergence_info: ConvergenceInfo,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,

    /// Average memory usage (MB)
    pub avg_memory_mb: f64,

    /// Memory efficiency (ops per MB)
    pub memory_efficiency: f64,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether convergence was achieved
    pub converged: bool,

    /// Step at which convergence was achieved
    pub convergence_step: Option<usize>,

    /// Final metric value
    pub final_value: f64,

    /// Best metric value achieved
    pub best_value: f64,

    /// Convergence rate (improvement per step)
    pub convergence_rate: f64,
}

/// Main visualization engine
pub struct OptimizationVisualizer {
    /// Configuration
    config: VisualizationConfig,

    /// Tracked metrics
    metrics: HashMap<String, OptimizationMetric>,

    /// Optimizer comparisons
    comparisons: Vec<OptimizerComparison>,

    /// Real-time dashboard state
    dashboard_state: DashboardState,

    /// Step counter
    current_step: usize,

    /// Last update step
    last_update_step: usize,
}

/// Dashboard state for real-time visualization
#[derive(Debug)]
struct DashboardState {
    /// Active plots
    active_plots: HashMap<String, PlotState>,

    /// Layout configuration
    layout: DashboardLayout,

    /// Update timestamps
    last_update: SystemTime,
}

/// Individual plot state
#[derive(Debug)]
struct PlotState {
    /// Plot type
    plot_type: PlotType,

    /// Data series
    series: Vec<DataSeries>,

    /// Axis configuration
    x_axis: AxisConfig,
    y_axis: AxisConfig,

    /// Plot title
    title: String,
}

/// Types of plots available
#[derive(Debug, Clone, Copy)]
pub enum PlotType {
    Line,
    Scatter,
    Histogram,
    Heatmap,
    Bar,
    Box,
    Violin,
    Surface3D,
}

/// Data series for plotting
#[derive(Debug, Clone)]
pub struct DataSeries {
    /// Series name
    pub name: String,

    /// X values
    pub x_values: Vec<f64>,

    /// Y values
    pub y_values: Vec<f64>,

    /// Z values (for 3D plots)
    pub z_values: Option<Vec<f64>>,

    /// Color
    pub color: String,

    /// Line style
    pub line_style: LineStyle,

    /// Marker style
    pub marker_style: MarkerStyle,
}

/// Line styles
#[derive(Debug, Clone, Copy)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
    None,
}

/// Marker styles
#[derive(Debug, Clone, Copy)]
pub enum MarkerStyle {
    Circle,
    Square,
    Triangle,
    Diamond,
    Plus,
    Cross,
    None,
}

/// Axis configuration
#[derive(Debug, Clone)]
pub struct AxisConfig {
    /// Axis label
    pub label: String,

    /// Scale type
    pub scale: AxisScale,

    /// Range (min, max)
    pub range: Option<(f64, f64)>,

    /// Tick configuration
    pub ticks: TickConfig,
}

/// Axis scale types
#[derive(Debug, Clone, Copy)]
pub enum AxisScale {
    Linear,
    Log,
    Symlog,
}

/// Tick configuration
#[derive(Debug, Clone)]
pub struct TickConfig {
    /// Major tick spacing
    pub major_spacing: Option<f64>,

    /// Minor tick count
    pub minor_count: usize,

    /// Show tick labels
    pub show_labels: bool,
}

/// Dashboard layout
#[derive(Debug, Clone)]
pub struct DashboardLayout {
    /// Number of rows
    pub rows: usize,

    /// Number of columns
    pub cols: usize,

    /// Plot positions
    pub plot_positions: HashMap<String, (usize, usize)>,
}

impl OptimizationVisualizer {
    /// Create new visualization engine
    pub fn new(config: VisualizationConfig) -> Result<Self> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&config.output_dir).map_err(|e| {
            OptimError::InvalidConfig(format!("Failed to create output directory: {e}"))
        })?;

        let dashboard_state = DashboardState {
            active_plots: HashMap::new(),
            layout: DashboardLayout {
                rows: 2,
                cols: 2,
                plot_positions: HashMap::new(),
            },
            last_update: SystemTime::now(),
        };

        Ok(Self {
            config,
            metrics: HashMap::new(),
            comparisons: Vec::new(),
            dashboard_state,
            current_step: 0,
            last_update_step: 0,
        })
    }

    /// Add or update a metric
    pub fn add_metric(&mut self, name: String, value: f64, higher_isbetter: bool, units: String) {
        let metric = self
            .metrics
            .entry(name.clone())
            .or_insert_with(|| OptimizationMetric::new(name, higher_isbetter, units));

        metric.add_value(value, self.current_step);
    }

    /// Set target value for a metric
    pub fn set_target(&mut self, metricname: &str, target: f64) {
        if let Some(metric) = self.metrics.get_mut(metricname) {
            metric.target = Some(target);
        }
    }

    /// Update step counter
    pub fn step(&mut self) {
        self.current_step += 1;

        if self.current_step - self.last_update_step >= self.config.update_frequency {
            if let Err(e) = self.update_dashboard() {
                eprintln!("Failed to update dashboard: {e}");
            }
            self.last_update_step = self.current_step;
        }
    }

    /// Create loss curve plot
    pub fn plot_loss_curve(&self, metricname: &str) -> Result<String> {
        let metric = self
            .metrics
            .get(metricname)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Metric '{metricname}' not found")))?;

        let steps: Vec<f64> = metric.steps.iter().map(|&s| s as f64).collect();
        let values = metric.get_smoothed_values();

        let plotdata = self.create_line_plot(
            &steps,
            &values,
            &format!("{} over Training Steps", metric.name),
            "Training Steps",
            &format!("{} ({})", metric.name, metric.units),
        )?;

        self.save_plot(&plotdata, &format!("{metricname}_curve"))
    }

    /// Create learning rate schedule plot
    pub fn plot_learning_rate_schedule(&self) -> Result<String> {
        if let Some(lr_metric) = self.metrics.get("learning_rate") {
            let steps: Vec<f64> = lr_metric.steps.iter().map(|&s| s as f64).collect();
            let values: Vec<f64> = lr_metric.values.iter().copied().collect();

            let plotdata = self.create_line_plot(
                &steps,
                &values,
                "Learning Rate Schedule",
                "Training Steps",
                "Learning Rate",
            )?;

            self.save_plot(&plotdata, "learning_rate_schedule")
        } else {
            Err(OptimError::InvalidConfig(
                "Learning rate metric not found".to_string(),
            ))
        }
    }

    /// Create optimizer comparison plot
    pub fn plot_optimizer_comparison(&self, metricname: &str) -> Result<String> {
        if self.comparisons.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No optimizer comparisons available".to_string(),
            ));
        }

        let mut plotdata = String::new();

        // HTML header for interactive plot
        if self.config.interactive_html {
            plotdata.push_str(&self.create_html_header("Optimizer Comparison")?);
            plotdata.push_str("<div id='comparison-plot'></div>\n");
            plotdata.push_str("<script>\n");
            plotdata.push_str("const traces = [];\n");

            for comparison in &self.comparisons {
                if let Some(values) = comparison.metrics.get(metricname) {
                    let x_values: Vec<String> = (0..values.len()).map(|i| i.to_string()).collect();
                    writeln!(&mut plotdata,
                        "traces.push({{x: {:?}, y: {:?}, name: '{}', type: 'scatter', mode: 'lines'}});",
                        x_values, values, comparison.name
                    ).unwrap();
                }
            }

            plotdata.push_str("Plotly.newPlot('comparison-plot', traces, {\n");
            plotdata.push_str("  title: 'Optimizer Comparison',\n");
            plotdata.push_str("  xaxis: {title: 'Training Steps'},\n");
            writeln!(&mut plotdata, "  yaxis: {{title: '{metricname}'}}").unwrap();
            plotdata.push_str("});\n");
            plotdata.push_str("</script>\n");
            plotdata.push_str("</body></html>\n");
        }

        self.save_plot(&plotdata, &format!("{metricname}_comparison"))
    }

    /// Create gradient norm visualization
    pub fn plot_gradient_norm(&self) -> Result<String> {
        if let Some(grad_metric) = self.metrics.get("gradient_norm") {
            let steps: Vec<f64> = grad_metric.steps.iter().map(|&s| s as f64).collect();
            let values: Vec<f64> = grad_metric.values.iter().copied().collect();

            let mut plotdata = self.create_line_plot(
                &steps,
                &values,
                "Gradient Norm",
                "Training Steps",
                "Gradient Norm",
            )?;

            // Add log scale for y-axis if values span multiple orders of magnitude
            let max_val = values.iter().fold(0.0f64, |a, &b| a.max(b));
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            if max_val / min_val > 100.0 {
                plotdata = plotdata.replace("yaxis: {", "yaxis: {type: 'log', ");
            }

            self.save_plot(&plotdata, "gradient_norm")
        } else {
            Err(OptimError::InvalidConfig(
                "Gradient norm metric not found".to_string(),
            ))
        }
    }

    /// Create training throughput plot
    pub fn plot_throughput(&self) -> Result<String> {
        if let Some(throughput_metric) = self.metrics.get("throughput") {
            let steps: Vec<f64> = throughput_metric.steps.iter().map(|&s| s as f64).collect();
            let values: Vec<f64> = throughput_metric.values.iter().copied().collect();

            let plotdata = self.create_line_plot(
                &steps,
                &values,
                "Training Throughput",
                "Training Steps",
                "Samples/Second",
            )?;

            self.save_plot(&plotdata, "throughput")
        } else {
            Err(OptimError::InvalidConfig(
                "Throughput metric not found".to_string(),
            ))
        }
    }

    /// Create memory usage visualization
    pub fn plot_memory_usage(&self) -> Result<String> {
        if let Some(memory_metric) = self.metrics.get("memory_usage") {
            let steps: Vec<f64> = memory_metric.steps.iter().map(|&s| s as f64).collect();
            let values: Vec<f64> = memory_metric.values.iter().copied().collect();

            let plotdata = self.create_line_plot(
                &steps,
                &values,
                "Memory Usage",
                "Training Steps",
                "Memory (MB)",
            )?;

            self.save_plot(&plotdata, "memory_usage")
        } else {
            Err(OptimError::InvalidConfig(
                "Memory usage metric not found".to_string(),
            ))
        }
    }

    /// Create hyperparameter sensitivity analysis
    pub fn plot_hyperparameter_sensitivity(
        &self,
        param_name: &str,
        metricname: &str,
    ) -> Result<String> {
        let mut param_values = Vec::new();
        let mut metric_values = Vec::new();

        for comparison in &self.comparisons {
            if let (Some(&param_val), Some(metric_vals)) = (
                comparison.hyperparameters.get(param_name),
                comparison.metrics.get(metricname),
            ) {
                if let Some(&final_metric) = metric_vals.last() {
                    param_values.push(param_val);
                    metric_values.push(final_metric);
                }
            }
        }

        if param_values.is_empty() {
            return Err(OptimError::InvalidConfig(format!(
                "No data available for hyperparameter '{}' and metric '{}'",
                param_name, metricname
            )));
        }

        let plotdata = self.create_scatter_plot(
            &param_values,
            &metric_values,
            &format!("Sensitivity of {} to {}", metricname, param_name),
            param_name,
            metricname,
        )?;

        self.save_plot(
            &plotdata,
            &format!("sensitivity_{}_{}", param_name, metricname),
        )
    }

    /// Create comprehensive dashboard
    pub fn create_dashboard(&self) -> Result<String> {
        let mut dashboard = String::new();

        if self.config.interactive_html {
            dashboard.push_str(&self.create_html_header("Optimization Dashboard")?);

            // Add CSS for layout
            dashboard.push_str(
                r#"
<style>
.dashboard-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 20px;
    height: 100vh;
    padding: 20px;
}
.plot-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
}
.metrics-summary {
    grid-column: span 2;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 20px;
}
</style>
"#,
            );

            // Metrics summary
            dashboard.push_str("<div class='metrics-summary'>\n");
            dashboard.push_str("<h2>Current Metrics</h2>\n");
            dashboard.push_str("<div style='display: flex; gap: 20px;'>\n");

            for (name, metric) in &self.metrics {
                if let Some(&latest_value) = metric.values.back() {
                    writeln!(
                        &mut dashboard,
                        "<div><strong>{}:</strong> {:.4} {}</div>",
                        name, latest_value, metric.units
                    )
                    .unwrap();
                }
            }

            dashboard.push_str("</div></div>\n");

            // Plot containers
            dashboard.push_str("<div class='dashboard-container'>\n");

            let mut plot_id = 0;
            for _ in &self.metrics {
                if plot_id >= 4 {
                    break;
                } // Limit to 4 plots in 2x2 grid

                writeln!(
                    &mut dashboard,
                    "<div class='plot-container'><div id='plot-{}'></div></div>",
                    plot_id
                )
                .unwrap();

                plot_id += 1;
            }

            dashboard.push_str("</div>\n");

            // JavaScript for plots
            dashboard.push_str("<script>\n");

            plot_id = 0;
            for (name, metric) in &self.metrics {
                if plot_id >= 4 {
                    break;
                }

                let steps: Vec<String> = metric.steps.iter().map(|&s| s.to_string()).collect();
                let values: Vec<f64> = metric.values.iter().copied().collect();

                writeln!(&mut dashboard,
                    "Plotly.newPlot('plot-{}', [{{x: {:?}, y: {:?}, type: 'scatter', mode: 'lines', name: '{}'}}], {{title: '{}', xaxis: {{title: 'Steps'}}, yaxis: {{title: '{}'}}}});",
                    plot_id, steps, values, name, name, metric.units
                ).unwrap();

                plot_id += 1;
            }

            dashboard.push_str("</script>\n");
            dashboard.push_str("</body></html>\n");
        }

        self.save_plot(&dashboard, "dashboard")
    }

    /// Update real-time dashboard
    fn update_dashboard(&mut self) -> Result<()> {
        self.dashboard_state.last_update = SystemTime::now();

        // In a real implementation, this would update the live dashboard
        // For now, we'll just regenerate static files
        self.create_dashboard()?;

        Ok(())
    }

    /// Add optimizer comparison data
    pub fn add_optimizer_comparison(&mut self, comparison: OptimizerComparison) {
        self.comparisons.push(comparison);
    }

    /// Export all visualizations
    pub fn export_all(&self) -> Result<Vec<String>> {
        let mut exported_files = Vec::new();

        // Export individual metric plots
        for metricname in self.metrics.keys() {
            if let Ok(filename) = self.plot_loss_curve(metricname) {
                exported_files.push(filename);
            }
        }

        // Export comparisons
        for metricname in ["loss", "accuracy", "throughput"] {
            if let Ok(filename) = self.plot_optimizer_comparison(metricname) {
                exported_files.push(filename);
            }
        }

        // Export specialized plots
        if let Ok(filename) = self.plot_gradient_norm() {
            exported_files.push(filename);
        }

        if let Ok(filename) = self.plot_throughput() {
            exported_files.push(filename);
        }

        if let Ok(filename) = self.plot_memory_usage() {
            exported_files.push(filename);
        }

        // Export dashboard
        if let Ok(filename) = self.create_dashboard() {
            exported_files.push(filename);
        }

        Ok(exported_files)
    }

    /// Helper function to create line plot
    fn create_line_plot(
        &self,
        x_values: &[f64],
        y_values: &[f64],
        title: &str,
        x_label: &str,
        y_label: &str,
    ) -> Result<String> {
        if !self.config.interactive_html {
            return Ok(format!("# {}\nX: {:?}\nY: {:?}", title, x_values, y_values));
        }

        let mut plot = String::new();
        plot.push_str(&self.create_html_header(title)?);
        plot.push_str("<div id='plot'></div>\n");
        plot.push_str("<script>\n");

        writeln!(
            &mut plot,
            "const trace = {{x: {:?}, y: {:?}, type: 'scatter', mode: 'lines', name: '{}'}};",
            x_values, y_values, title
        )
        .unwrap();

        writeln!(&mut plot,
            "Plotly.newPlot('plot', [trace], {{title: '{}', xaxis: {{title: '{}'}}, yaxis: {{title: '{}'}}}});",
            title, x_label, y_label
        ).unwrap();

        plot.push_str("</script></body></html>");

        Ok(plot)
    }

    /// Helper function to create scatter plot
    fn create_scatter_plot(
        &self,
        x_values: &[f64],
        y_values: &[f64],
        title: &str,
        x_label: &str,
        y_label: &str,
    ) -> Result<String> {
        if !self.config.interactive_html {
            return Ok(format!("# {}\nX: {:?}\nY: {:?}", title, x_values, y_values));
        }

        let mut plot = String::new();
        plot.push_str(&self.create_html_header(title)?);
        plot.push_str("<div id='plot'></div>\n");
        plot.push_str("<script>\n");

        writeln!(
            &mut plot,
            "const trace = {{x: {:?}, y: {:?}, type: 'scatter', mode: 'markers', name: '{}'}};",
            x_values, y_values, title
        )
        .unwrap();

        writeln!(&mut plot,
            "Plotly.newPlot('plot', [trace], {{title: '{}', xaxis: {{title: '{}'}}, yaxis: {{title: '{}'}}}});",
            title, x_label, y_label
        ).unwrap();

        plot.push_str("</script></body></html>");

        Ok(plot)
    }

    /// Create HTML header for interactive plots
    fn create_html_header(&self, title: &str) -> Result<String> {
        Ok(format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #plot {{ width: 100%; height: 500px; }}
    </style>
</head>
<body>
    <h1>{}</h1>
"#,
            title, title
        ))
    }

    /// Save plot to file
    fn save_plot(&self, plotdata: &str, filename: &str) -> Result<String> {
        let extension = if self.config.interactive_html {
            "html"
        } else {
            "txt"
        };
        let full_filename = format!("{}.{}", filename, extension);
        let filepath = Path::new(&self.config.output_dir).join(&full_filename);

        let mut file = std::fs::File::create(&filepath).map_err(|e| {
            OptimError::InvalidConfig(format!(
                "Failed to create file {}: {}",
                filepath.display(),
                e
            ))
        })?;

        file.write_all(plotdata.as_bytes()).map_err(|e| {
            OptimError::InvalidConfig(format!(
                "Failed to write to file {}: {}",
                filepath.display(),
                e
            ))
        })?;

        Ok(full_filename)
    }

    /// Get color for plot series
    fn get_color(&self, index: usize) -> String {
        let colors = match self.config.color_scheme {
            ColorScheme::Default => vec![
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22", "#17becf",
            ],
            ColorScheme::Dark => vec![
                "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69",
                "#fccde5", "#d9d9d9", "#bc80bd",
            ],
            ColorScheme::Colorblind => vec![
                "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
                "#CC79A7",
            ],
            ColorScheme::Publication => vec!["#000000", "#333333", "#666666", "#999999", "#CCCCCC"],
            ColorScheme::Vibrant => vec![
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8",
                "#F7DC6F", "#BB8FCE", "#85C1E9",
            ],
        };

        colors[index % colors.len()].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.max_points, 10000);
        assert!(config.interactive_html);
        assert!(config.show_grid);
    }

    #[test]
    fn test_optimization_metric() {
        let mut metric = OptimizationMetric::new("loss".to_string(), false, "nats".to_string());

        metric.add_value(1.0, 0);
        metric.add_value(0.8, 1);
        metric.add_value(0.6, 2);
        metric.add_value(0.4, 3); // Add 4th value to meet windowsize * 2 requirement

        assert_eq!(metric.values.len(), 4);
        assert_eq!(metric.steps.len(), 4);

        let improvement = metric.get_recent_improvement(2);
        assert!(improvement.is_some());
    }

    #[test]
    fn test_visualizer_creation() {
        let config = VisualizationConfig {
            output_dir: "/tmp/test_plots".to_string(),
            ..Default::default()
        };

        let visualizer = OptimizationVisualizer::new(config);
        assert!(visualizer.is_ok());
    }

    #[test]
    fn test_add_metric() {
        let config = VisualizationConfig {
            output_dir: "/tmp/test_plots".to_string(),
            ..Default::default()
        };

        let mut visualizer = OptimizationVisualizer::new(config).unwrap();

        visualizer.add_metric("loss".to_string(), 1.0, false, "nats".to_string());
        visualizer.step();
        visualizer.add_metric("loss".to_string(), 0.8, false, "nats".to_string());

        assert!(visualizer.metrics.contains_key("loss"));
        assert_eq!(visualizer.metrics["loss"].values.len(), 2);
    }

    #[test]
    fn test_optimizer_comparison() {
        let comparison = OptimizerComparison {
            name: "Adam".to_string(),
            metrics: {
                let mut map = HashMap::new();
                map.insert("loss".to_string(), vec![1.0, 0.8, 0.6]);
                map
            },
            hyperparameters: {
                let mut map = HashMap::new();
                map.insert("learning_rate".to_string(), 0.001);
                map
            },
            training_time: Duration::from_secs(120),
            memory_stats: MemoryStats {
                peak_memory_mb: 1024.0,
                avg_memory_mb: 512.0,
                memory_efficiency: 100.0,
            },
            convergence_info: ConvergenceInfo {
                converged: true,
                convergence_step: Some(100),
                final_value: 0.6,
                best_value: 0.6,
                convergence_rate: 0.004,
            },
        };

        assert_eq!(comparison.name, "Adam");
        assert!(comparison.convergence_info.converged);
    }

    #[test]
    fn test_color_schemes() {
        let config = VisualizationConfig {
            color_scheme: ColorScheme::Colorblind,
            output_dir: "/tmp/test_plots".to_string(),
            ..Default::default()
        };

        let visualizer = OptimizationVisualizer::new(config).unwrap();
        let color = visualizer.get_color(0);
        assert_eq!(color, "#000000");
    }
}
