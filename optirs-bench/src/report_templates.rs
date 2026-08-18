//! Benchmark report generator and templating.
//!
//! This module renders the benchmark data types produced by the core
//! benchmarking pipeline ([`BenchmarkReport`], [`OptimizerPerformance`],
//! [`OptimizerComparison`], [`BenchmarkResult`] and
//! [`VisualizationExport`]) into
//! human- and machine-readable reports.
//!
//! Three output formats are supported:
//!
//! * [`ReportFormat::Markdown`] – `#`/`##` headings with GitHub-flavoured tables.
//! * [`ReportFormat::PlainText`] – underlined sections and space-aligned columns.
//! * [`ReportFormat::Csv`] – a header row plus one machine-parseable row per
//!   optimizer.
//!
//! The [`ReportTemplate`] renderer produces an executive summary, a leaderboard
//! style ranking and an optional per-optimizer breakdown.  It complements the
//! free-form text dashboard provided by
//! [`OptimizerDashboard::generate_comparison_report`](crate::visualization::OptimizerDashboard::generate_comparison_report)
//! by offering structured, multi-format output and a stable ranking.
//!
//! ```rust,no_run
//! use optirs_bench::report_templates::{ReportFormat, ReportTemplate, save_report};
//! use std::path::Path;
//!
//! # fn demo<A: scirs2_core::numeric::Float>(report: &optirs_bench::BenchmarkReport<A>) -> optirs_bench::Result<()> {
//! let template = ReportTemplate::new().with_title("Optimizer Shootout");
//! let markdown = template.render(report, None, ReportFormat::Markdown);
//! save_report(Path::new("/tmp/benchmark_report.md"), &markdown)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;

use scirs2_core::numeric::Float;

use crate::visualization::VisualizationExport;
use crate::Result;
use crate::{BenchmarkReport, BenchmarkResult, OptimizerComparison, OptimizerPerformance};

/// Output format for a rendered benchmark report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReportFormat {
    /// GitHub-flavoured Markdown with `#`/`##` headings and pipe tables.
    Markdown,
    /// Plain text with underlined sections and space-aligned columns.
    PlainText,
    /// Comma-separated values: a header row plus one row per optimizer.
    Csv,
}

impl ReportFormat {
    /// Conventional file extension (without leading dot) for this format.
    pub fn extension(self) -> &'static str {
        match self {
            ReportFormat::Markdown => "md",
            ReportFormat::PlainText => "txt",
            ReportFormat::Csv => "csv",
        }
    }

    /// All supported formats, in a stable order.
    pub fn all() -> [ReportFormat; 3] {
        [
            ReportFormat::Markdown,
            ReportFormat::PlainText,
            ReportFormat::Csv,
        ]
    }
}

/// Configurable benchmark report renderer.
///
/// The renderer is intentionally cheap to clone and holds only presentation
/// configuration; the actual benchmark data is borrowed at render time.
#[derive(Debug, Clone)]
pub struct ReportTemplate {
    /// Title rendered as the top-level heading.
    pub title: String,
    /// Whether to emit the executive summary section.
    pub include_summary: bool,
    /// Whether to emit the detailed per-optimizer performance section.
    pub include_per_optimizer: bool,
    /// Number of decimal places used when formatting floating point values.
    pub precision: usize,
}

impl Default for ReportTemplate {
    fn default() -> Self {
        Self {
            title: "Benchmark Report".to_string(),
            include_summary: true,
            include_per_optimizer: true,
            precision: 6,
        }
    }
}

impl ReportTemplate {
    /// Create a report template with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the report title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Toggle the executive summary section.
    pub fn with_summary(mut self, include: bool) -> Self {
        self.include_summary = include;
        self
    }

    /// Toggle the detailed per-optimizer performance section.
    pub fn with_per_optimizer(mut self, include: bool) -> Self {
        self.include_per_optimizer = include;
        self
    }

    /// Set the number of decimal places used for floating point values.
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    /// Render a benchmark report (optionally with a pairwise comparison) in the
    /// requested format.
    ///
    /// For [`ReportFormat::Csv`] the output is the per-optimizer ranking table
    /// only (a header row followed by one row per optimizer); the executive
    /// summary and comparison prose are omitted so the result stays
    /// machine-parseable.
    pub fn render<A: Float>(
        &self,
        report: &BenchmarkReport<A>,
        comparison: Option<&OptimizerComparison<A>>,
        format: ReportFormat,
    ) -> String {
        match format {
            ReportFormat::Csv => self.render_csv(report),
            ReportFormat::Markdown | ReportFormat::PlainText => {
                self.render_human(report, comparison, format)
            }
        }
    }

    /// Render the report in every supported format at once.
    pub fn generate_full_report<A: Float>(
        &self,
        report: &BenchmarkReport<A>,
        comparison: Option<&OptimizerComparison<A>>,
    ) -> HashMap<ReportFormat, String> {
        let mut rendered = HashMap::with_capacity(ReportFormat::all().len());
        for format in ReportFormat::all() {
            rendered.insert(format, self.render(report, comparison, format));
        }
        rendered
    }

    /// Render a per-run detail table from raw [`BenchmarkResult`] records.
    ///
    /// This consumes the individual results (which are not retained inside a
    /// [`BenchmarkReport`]) so callers can attach a fine-grained breakdown to a
    /// summary report.
    pub fn render_run_details<A: Float>(
        &self,
        results: &[BenchmarkResult<A>],
        format: ReportFormat,
    ) -> String {
        let headers = [
            "Optimizer",
            "Function",
            "Converged",
            "Convergence Step",
            "Final Value",
            "Final Grad Norm",
            "Final Error",
            "Iterations",
            "Time (s)",
            "Evaluations",
        ];
        let mut rows = Vec::with_capacity(results.len());
        for result in results {
            let convergence_step = match result.convergence_step {
                Some(step) => step.to_string(),
                None => "-".to_string(),
            };
            rows.push(vec![
                result.optimizername.clone(),
                result.function_name.clone(),
                if result.converged {
                    "yes".to_string()
                } else {
                    "no".to_string()
                },
                convergence_step,
                self.fmt_num(as_f64(result.final_function_value)),
                self.fmt_num(as_f64(result.final_gradient_norm)),
                self.fmt_num(as_f64(result.final_error)),
                result.iterations_taken.to_string(),
                self.fmt_num(result.elapsed_time.as_secs_f64()),
                result.function_evaluations.to_string(),
            ]);
        }
        render_table(&headers, &rows, format)
    }

    /// Render a compact trajectory summary for a single optimizer from an
    /// exported [`VisualizationExport`].
    pub fn render_trajectory_summary<A: Float>(
        &self,
        optimizer_name: &str,
        export: &VisualizationExport<A>,
        format: ReportFormat,
    ) -> String {
        let steps = export.step_indices.len().max(export.loss_history.len());
        let initial_loss = export.loss_history.first().copied().map(as_f64);
        let final_loss = export.loss_history.last().copied().map(as_f64);
        let improvement = match (initial_loss, final_loss) {
            (Some(start), Some(end)) => Some(start - end),
            _ => None,
        };
        let (min_lr, max_lr) = min_max(&export.learning_rate_history);
        let final_lr = export.learning_rate_history.last().copied().map(as_f64);

        let headers = ["Metric", "Value"];
        let rows = vec![
            vec!["Optimizer".to_string(), optimizer_name.to_string()],
            vec!["Steps recorded".to_string(), steps.to_string()],
            vec!["Initial loss".to_string(), self.fmt_opt(initial_loss)],
            vec!["Final loss".to_string(), self.fmt_opt(final_loss)],
            vec!["Total improvement".to_string(), self.fmt_opt(improvement)],
            vec!["Min learning rate".to_string(), self.fmt_opt(min_lr)],
            vec!["Max learning rate".to_string(), self.fmt_opt(max_lr)],
            vec!["Final learning rate".to_string(), self.fmt_opt(final_lr)],
            vec![
                "Parameter groups".to_string(),
                export.parameter_norms.len().to_string(),
            ],
            vec![
                "State snapshots".to_string(),
                export.state_snapshots.len().to_string(),
            ],
        ];
        render_table(&headers, &rows, format)
    }

    /// Render the Markdown / plain-text variant (they share the same structure
    /// and differ only in heading and table syntax).
    fn render_human<A: Float>(
        &self,
        report: &BenchmarkReport<A>,
        comparison: Option<&OptimizerComparison<A>>,
        format: ReportFormat,
    ) -> String {
        let ranked = rank_optimizers(report);
        let mut out = String::new();

        out.push_str(&heading(1, &self.title, format));

        if self.include_summary {
            out.push_str(&heading(2, "Executive Summary", format));
            out.push_str(&self.summary_lines(report, &ranked));
            out.push('\n');
        }

        out.push_str(&heading(2, "Ranking", format));
        if ranked.is_empty() {
            out.push_str("No optimizer performance data available.\n\n");
        } else {
            let headers = [
                "Rank",
                "Optimizer",
                "Success Rate",
                "Avg Final Error",
                "Avg Iterations",
                "Runs",
            ];
            let rows = self.ranking_rows(&ranked);
            out.push_str(&render_table(&headers, &rows, format));
            out.push('\n');
        }

        if self.include_per_optimizer && !ranked.is_empty() {
            out.push_str(&heading(2, "Per-Optimizer Performance", format));
            let headers = [
                "Optimizer",
                "Total Runs",
                "Successful Runs",
                "Success Rate",
                "Avg Final Error",
                "Avg Iterations",
                "Avg Time (s)",
            ];
            let rows = self.per_optimizer_rows(&ranked);
            out.push_str(&render_table(&headers, &rows, format));
            out.push('\n');
        }

        if let Some(comparison) = comparison {
            out.push_str(&heading(2, "Optimizer Comparison", format));
            out.push_str(&self.comparison_lines(comparison));
            out.push('\n');
        }

        out
    }

    /// Render the CSV variant: a header row plus one row per optimizer.
    fn render_csv<A: Float>(&self, report: &BenchmarkReport<A>) -> String {
        let ranked = rank_optimizers(report);
        let headers = [
            "rank",
            "optimizer",
            "success_rate",
            "successful_runs",
            "total_runs",
            "average_final_error",
            "average_iterations",
            "average_time_seconds",
        ];
        let mut rows = Vec::with_capacity(ranked.len());
        for entry in &ranked {
            rows.push(vec![
                entry.rank.to_string(),
                entry.name.clone(),
                self.fmt_num(entry.success_rate),
                entry.successful_runs.to_string(),
                entry.total_runs.to_string(),
                self.fmt_num(entry.average_final_error),
                self.fmt_num(entry.average_iterations),
                self.fmt_num(entry.average_time_secs),
            ]);
        }
        render_table(&headers, &rows, ReportFormat::Csv)
    }

    /// Build the executive-summary bullet list.
    fn summary_lines<A: Float>(
        &self,
        report: &BenchmarkReport<A>,
        ranked: &[RankedOptimizer],
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!("- Total tests run: {}\n", report.total_tests));
        out.push_str(&format!("- Optimizers evaluated: {}\n", ranked.len()));

        if let Some(best) = ranked.first() {
            out.push_str(&format!(
                "- Best optimizer: {} (success rate {:.2}%, average final error {})\n",
                best.name,
                best.success_rate * 100.0,
                self.fmt_num(best.average_final_error),
            ));
        } else {
            out.push_str("- Best optimizer: n/a (no data)\n");
        }

        if ranked.len() > 1 {
            if let Some(worst) = ranked.last() {
                out.push_str(&format!(
                    "- Worst optimizer: {} (success rate {:.2}%, average final error {})\n",
                    worst.name,
                    worst.success_rate * 100.0,
                    self.fmt_num(worst.average_final_error),
                ));
            }
        }

        out
    }

    /// Build the pairwise comparison bullet list from an [`OptimizerComparison`].
    fn comparison_lines<A: Float>(&self, comparison: &OptimizerComparison<A>) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "- Comparing {} (first) vs {} (second)\n",
            comparison.optimizer1, comparison.optimizer2,
        ));
        out.push_str(&format!(
            "- Success rate difference (first - second): {}\n",
            self.fmt_num(comparison.success_rate_diff),
        ));
        out.push_str(&format!(
            "- Average iterations difference (first - second): {}\n",
            self.fmt_num(comparison.avg_iterations_diff),
        ));
        out.push_str(&format!(
            "- Average final error difference (first - second): {}\n",
            self.fmt_num(as_f64(comparison.avg_error_diff)),
        ));
        out
    }

    /// Build the ranking table rows (best first).
    fn ranking_rows(&self, ranked: &[RankedOptimizer]) -> Vec<Vec<String>> {
        ranked
            .iter()
            .map(|entry| {
                vec![
                    entry.rank.to_string(),
                    entry.name.clone(),
                    self.fmt_num(entry.success_rate),
                    self.fmt_num(entry.average_final_error),
                    self.fmt_num(entry.average_iterations),
                    format!("{}/{}", entry.successful_runs, entry.total_runs),
                ]
            })
            .collect()
    }

    /// Build the detailed per-optimizer table rows (best first).
    fn per_optimizer_rows(&self, ranked: &[RankedOptimizer]) -> Vec<Vec<String>> {
        ranked
            .iter()
            .map(|entry| {
                vec![
                    entry.name.clone(),
                    entry.total_runs.to_string(),
                    entry.successful_runs.to_string(),
                    self.fmt_num(entry.success_rate),
                    self.fmt_num(entry.average_final_error),
                    self.fmt_num(entry.average_iterations),
                    self.fmt_num(entry.average_time_secs),
                ]
            })
            .collect()
    }

    /// Format a floating point value at the configured precision.
    fn fmt_num(&self, value: f64) -> String {
        format!("{:.*}", self.precision, value)
    }

    /// Format an optional floating point value, using `-` for `None`.
    fn fmt_opt(&self, value: Option<f64>) -> String {
        match value {
            Some(value) => self.fmt_num(value),
            None => "-".to_string(),
        }
    }
}

/// A single optimizer's summarised metrics with its computed rank.
#[derive(Debug, Clone)]
struct RankedOptimizer {
    /// 1-based rank (1 = best).
    rank: usize,
    /// Optimizer name.
    name: String,
    /// Convergence success rate in `[0, 1]`.
    success_rate: f64,
    /// Number of converged runs.
    successful_runs: usize,
    /// Total number of runs.
    total_runs: usize,
    /// Average final error (lower is better).
    average_final_error: f64,
    /// Average iterations to convergence.
    average_iterations: f64,
    /// Average wall-clock time per run, in seconds.
    average_time_secs: f64,
}

/// Build a ranked, best-first list of optimizers from a [`BenchmarkReport`].
///
/// Optimizers are ordered by convergence success rate (descending), then by
/// average final error (ascending), then by average iterations (ascending),
/// with the optimizer name as a final deterministic tie-breaker.
fn rank_optimizers<A: Float>(report: &BenchmarkReport<A>) -> Vec<RankedOptimizer> {
    let mut entries: Vec<RankedOptimizer> = report
        .optimizer_performance
        .iter()
        .map(|(name, performance)| ranked_from(name, performance))
        .collect();

    entries.sort_by(|a, b| {
        b.success_rate
            .partial_cmp(&a.success_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.average_final_error
                    .partial_cmp(&b.average_final_error)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                a.average_iterations
                    .partial_cmp(&b.average_iterations)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then_with(|| a.name.cmp(&b.name))
    });

    for (index, entry) in entries.iter_mut().enumerate() {
        entry.rank = index + 1;
    }

    entries
}

/// Convert a single [`OptimizerPerformance`] entry into a [`RankedOptimizer`]
/// (rank is assigned later by [`rank_optimizers`]).
fn ranked_from<A: Float>(name: &str, performance: &OptimizerPerformance<A>) -> RankedOptimizer {
    let success_rate = if performance.total_runs > 0 {
        performance.successful_runs as f64 / performance.total_runs as f64
    } else {
        0.0
    };

    RankedOptimizer {
        rank: 0,
        name: name.to_string(),
        success_rate,
        successful_runs: performance.successful_runs,
        total_runs: performance.total_runs,
        average_final_error: as_f64(performance.average_final_error),
        average_iterations: performance.average_iterations,
        average_time_secs: performance.average_time.as_secs_f64(),
    }
}

/// Render a table of pre-formatted string cells in the requested format.
fn render_table(headers: &[&str], rows: &[Vec<String>], format: ReportFormat) -> String {
    match format {
        ReportFormat::Markdown => render_markdown_table(headers, rows),
        ReportFormat::PlainText => render_plaintext_table(headers, rows),
        ReportFormat::Csv => render_csv_table(headers, rows),
    }
}

/// Render a GitHub-flavoured Markdown pipe table.
fn render_markdown_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();

    out.push('|');
    for header in headers.iter().copied() {
        out.push(' ');
        out.push_str(&markdown_escape(header));
        out.push_str(" |");
    }
    out.push('\n');

    out.push('|');
    for _ in headers {
        out.push_str("---|");
    }
    out.push('\n');

    for row in rows {
        out.push('|');
        for cell in row {
            out.push(' ');
            out.push_str(&markdown_escape(cell));
            out.push_str(" |");
        }
        out.push('\n');
    }

    out
}

/// Render a space-aligned plain-text table.
fn render_plaintext_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let columns = headers.len();
    let mut widths = vec![0usize; columns];

    for (index, header) in headers.iter().copied().enumerate() {
        widths[index] = widths[index].max(header.len());
    }
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            if index < columns {
                widths[index] = widths[index].max(cell.len());
            }
        }
    }

    let mut out = String::new();

    for (index, header) in headers.iter().copied().enumerate() {
        if index > 0 {
            out.push_str("  ");
        }
        if index + 1 < columns {
            out.push_str(&pad_right(header, widths[index]));
        } else {
            out.push_str(header);
        }
    }
    out.push('\n');

    let separator_width = widths.iter().sum::<usize>() + 2 * columns.saturating_sub(1);
    out.push_str(&"-".repeat(separator_width));
    out.push('\n');

    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            if index > 0 {
                out.push_str("  ");
            }
            let width = widths.get(index).copied().unwrap_or(0);
            if index + 1 < row.len() {
                out.push_str(&pad_right(cell, width));
            } else {
                out.push_str(cell);
            }
        }
        out.push('\n');
    }

    out
}

/// Render a CSV table with RFC-4180 style quoting.
fn render_csv_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();

    for (index, header) in headers.iter().copied().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&csv_field(header));
    }
    out.push('\n');

    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(&csv_field(cell));
        }
        out.push('\n');
    }

    out
}

/// Render a section heading for the given (human-readable) format.
fn heading(level: usize, text: &str, format: ReportFormat) -> String {
    match format {
        ReportFormat::Markdown => {
            let mut out = String::new();
            out.push_str(&"#".repeat(level.max(1)));
            out.push(' ');
            out.push_str(text);
            out.push_str("\n\n");
            out
        }
        ReportFormat::PlainText => {
            let underline = if level <= 1 { '=' } else { '-' };
            let mut out = String::new();
            out.push_str(text);
            out.push('\n');
            out.push_str(&underline.to_string().repeat(text.len()));
            out.push_str("\n\n");
            out
        }
        ReportFormat::Csv => String::new(),
    }
}

/// Escape a Markdown table cell (pipes would otherwise break the table).
fn markdown_escape(text: &str) -> String {
    text.replace('|', "\\|")
}

/// Quote a CSV field if it contains a comma, quote, or newline.
fn csv_field(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        let escaped = field.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        field.to_string()
    }
}

/// Right-pad a string with spaces to at least `width` bytes.
fn pad_right(text: &str, width: usize) -> String {
    if text.len() >= width {
        text.to_string()
    } else {
        let mut out = String::with_capacity(width);
        out.push_str(text);
        out.push_str(&" ".repeat(width - text.len()));
        out
    }
}

/// Convert a [`Float`] value to `f64`, defaulting to `0.0` on failure.
fn as_f64<A: Float>(value: A) -> f64 {
    value.to_f64().unwrap_or(0.0)
}

/// Compute the `(min, max)` of a slice of [`Float`] values as `f64`.
fn min_max<A: Float>(values: &[A]) -> (Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None);
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        let converted = as_f64(value);
        if converted < min {
            min = converted;
        }
        if converted > max {
            max = converted;
        }
    }
    (Some(min), Some(max))
}

/// Write a rendered report to disk.
///
/// Any parent directories must already exist; the file is created or truncated.
pub fn save_report(path: &Path, content: &str) -> Result<()> {
    std::fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    /// Build a small synthetic benchmark report with three optimizers whose
    /// relative quality is unambiguous (champion > middle > laggard).
    fn synthetic_report() -> BenchmarkReport<f64> {
        let mut optimizer_performance = HashMap::new();
        optimizer_performance.insert(
            "champion".to_string(),
            OptimizerPerformance {
                total_runs: 10,
                successful_runs: 10,
                average_iterations: 25.0,
                average_final_error: 0.001,
                average_time: Duration::from_millis(5),
            },
        );
        optimizer_performance.insert(
            "middle".to_string(),
            OptimizerPerformance {
                total_runs: 10,
                successful_runs: 8,
                average_iterations: 60.0,
                average_final_error: 0.05,
                average_time: Duration::from_millis(12),
            },
        );
        optimizer_performance.insert(
            "laggard".to_string(),
            OptimizerPerformance {
                total_runs: 10,
                successful_runs: 4,
                average_iterations: 120.0,
                average_final_error: 0.5,
                average_time: Duration::from_millis(30),
            },
        );

        BenchmarkReport {
            total_tests: 30,
            optimizer_performance,
        }
    }

    fn synthetic_comparison() -> OptimizerComparison<f64> {
        OptimizerComparison {
            optimizer1: "champion".to_string(),
            optimizer2: "laggard".to_string(),
            success_rate_diff: 0.6,
            avg_iterations_diff: -95.0,
            avg_error_diff: -0.499,
        }
    }

    fn synthetic_results() -> Vec<BenchmarkResult<f64>> {
        vec![
            BenchmarkResult {
                optimizername: "champion".to_string(),
                function_name: "sphere".to_string(),
                converged: true,
                convergence_step: Some(20),
                final_function_value: 0.0001,
                final_gradient_norm: 0.0002,
                final_error: 0.0001,
                iterations_taken: 25,
                elapsed_time: Duration::from_millis(5),
                function_evaluations: 25,
                function_value_history: vec![1.0, 0.5, 0.0001],
                gradient_norm_history: vec![1.0, 0.4, 0.0002],
            },
            BenchmarkResult {
                optimizername: "laggard".to_string(),
                function_name: "rosenbrock".to_string(),
                converged: false,
                convergence_step: None,
                final_function_value: 0.5,
                final_gradient_norm: 0.3,
                final_error: 0.5,
                iterations_taken: 120,
                elapsed_time: Duration::from_millis(30),
                function_evaluations: 120,
                function_value_history: vec![5.0, 2.0, 0.5],
                gradient_norm_history: vec![4.0, 1.0, 0.3],
            },
        ]
    }

    fn synthetic_export() -> VisualizationExport<f64> {
        VisualizationExport {
            step_indices: vec![0, 1, 2, 3],
            loss_history: vec![2.0, 1.0, 0.3, 0.1],
            learning_rate_history: vec![0.1, 0.05, 0.025, 0.0125],
            parameter_norms: vec![vec![1.0, 0.9, 0.8, 0.7]],
            state_snapshots: Vec::new(),
        }
    }

    #[test]
    fn report_template_markdown_contains_title_and_table() {
        let report = synthetic_report();
        let template = ReportTemplate::new().with_title("Synthetic Benchmark Suite");
        let markdown = template.render(&report, None, ReportFormat::Markdown);

        // Title heading present.
        assert!(markdown.contains("# Synthetic Benchmark Suite"));
        // Section headings present.
        assert!(markdown.contains("## Ranking"));
        assert!(markdown.contains("## Executive Summary"));
        // Markdown table delimiter present.
        assert!(markdown.contains("|---"));
        // Best optimizer surfaced in the summary.
        assert!(markdown.contains("champion"));
    }

    #[test]
    fn report_template_ranking_orders_best_first() {
        let report = synthetic_report();
        let ranked = rank_optimizers(&report);

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].name, "champion");
        assert_eq!(ranked[1].name, "middle");
        assert_eq!(ranked[2].name, "laggard");
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[2].rank, 3);

        // The rendered markdown lists the best optimizer before the worst.
        let template = ReportTemplate::new();
        let markdown = template.render(&report, None, ReportFormat::Markdown);
        let champion_pos = markdown
            .find("champion")
            .expect("champion should appear in report");
        let laggard_pos = markdown
            .find("laggard")
            .expect("laggard should appear in report");
        assert!(champion_pos < laggard_pos);
    }

    #[test]
    fn report_template_csv_shape_and_parsing() {
        let report = synthetic_report();
        let template = ReportTemplate::new().with_precision(4);
        let csv = template.render(&report, None, ReportFormat::Csv);

        let lines: Vec<&str> = csv.lines().filter(|line| !line.is_empty()).collect();
        // Header row + one row per optimizer.
        assert_eq!(lines.len(), 1 + 3);

        let header: Vec<&str> = lines[0].split(',').collect();
        assert_eq!(header.len(), 8);
        assert_eq!(header[0], "rank");
        assert_eq!(header[1], "optimizer");
        assert_eq!(header[2], "success_rate");

        // Every data row has the same column count and parseable numerics.
        for line in &lines[1..] {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 8);
            let _rank: usize = fields[0].parse().expect("rank should parse as usize");
            let success_rate: f64 = fields[2].parse().expect("success_rate should parse as f64");
            assert!((0.0..=1.0).contains(&success_rate));
        }

        // The first data row corresponds to the best optimizer.
        let first_row: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(first_row[1], "champion");
    }

    #[test]
    fn report_template_plaintext_has_sections() {
        let report = synthetic_report();
        let comparison = synthetic_comparison();
        let template = ReportTemplate::new().with_title("Plain Report");
        let text = template.render(&report, Some(&comparison), ReportFormat::PlainText);

        assert!(text.contains("Plain Report"));
        assert!(text.contains("Ranking"));
        assert!(text.contains("Optimizer Comparison"));
        // Underline separators used in plain-text headings.
        assert!(text.contains("======"));
        assert!(text.contains("------"));
        // Comparison surfaced the compared optimizer names.
        assert!(text.contains("champion"));
        assert!(text.contains("laggard"));
    }

    #[test]
    fn report_template_generate_full_report_covers_all_formats() {
        let report = synthetic_report();
        let template = ReportTemplate::new();
        let rendered = template.generate_full_report(&report, None);

        assert_eq!(rendered.len(), 3);
        assert!(rendered.contains_key(&ReportFormat::Markdown));
        assert!(rendered.contains_key(&ReportFormat::PlainText));
        assert!(rendered.contains_key(&ReportFormat::Csv));
        assert!(rendered[&ReportFormat::Markdown].contains("# Benchmark Report"));
    }

    #[test]
    fn report_template_run_details_and_trajectory() {
        let template = ReportTemplate::new().with_precision(3);

        let results = synthetic_results();
        let details = template.render_run_details(&results, ReportFormat::Markdown);
        assert!(details.contains("|---"));
        assert!(details.contains("Convergence Step"));
        assert!(details.contains("sphere"));
        assert!(details.contains("rosenbrock"));
        // Non-converged run renders its missing convergence step as `-`.
        assert!(details.contains("| - |"));

        let export = synthetic_export();
        let trajectory =
            template.render_trajectory_summary("champion", &export, ReportFormat::PlainText);
        assert!(trajectory.contains("Steps recorded"));
        assert!(trajectory.contains("Total improvement"));
        // Initial loss 2.0 minus final loss 0.1 = 1.9 improvement at precision 3.
        assert!(trajectory.contains("1.900"));
    }

    #[test]
    fn report_template_save_report_round_trips() {
        let report = synthetic_report();
        let template = ReportTemplate::new();
        let markdown = template.render(&report, None, ReportFormat::Markdown);

        let mut path = std::env::temp_dir();
        let file_name = format!(
            "optirs_bench_report_template_{}.{}",
            std::process::id(),
            ReportFormat::Markdown.extension(),
        );
        path.push(file_name);

        save_report(&path, &markdown).expect("save_report should succeed");
        let read_back = std::fs::read_to_string(&path).expect("temp report should be readable");
        assert_eq!(read_back, markdown);

        std::fs::remove_file(&path).expect("temp report should be removable");
    }

    #[test]
    fn report_template_empty_report_is_safe() {
        let report = BenchmarkReport::<f64> {
            total_tests: 0,
            optimizer_performance: HashMap::new(),
        };
        let template = ReportTemplate::new();

        let markdown = template.render(&report, None, ReportFormat::Markdown);
        assert!(markdown.contains("No optimizer performance data available."));

        let csv = template.render(&report, None, ReportFormat::Csv);
        let lines: Vec<&str> = csv.lines().filter(|line| !line.is_empty()).collect();
        // Only the header row, no data rows.
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn report_format_extension_is_stable() {
        assert_eq!(ReportFormat::Markdown.extension(), "md");
        assert_eq!(ReportFormat::PlainText.extension(), "txt");
        assert_eq!(ReportFormat::Csv.extension(), "csv");
    }
}
