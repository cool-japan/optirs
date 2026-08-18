// Real metrics export (finding M2).
//
// `export_metrics` used to be `Ok(())` with a comment. It now serialises the
// retained snapshots into the configured formats and writes them to the
// configured destinations, returning an explicit unsupported-operation error
// for every format or destination this crate genuinely cannot produce (rather
// than silently reporting success).

use super::alerts::{resolve_snapshot_metric, KNOWN_METRIC_PATHS};
use super::*;
use crate::error::OptimError;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;

/// One exported row: a snapshot timestamp and the metrics that had a value.
type ExportRow = (u64, BTreeMap<String, f64>);

fn sanitize_metric_name(path: &str) -> String {
    path.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn render_json(rows: &[ExportRow]) -> Result<String> {
    let payload: Vec<serde_json::Value> = rows
        .iter()
        .map(|(timestamp, metrics)| {
            let mut object = serde_json::Map::new();
            object.insert("timestamp".to_string(), serde_json::Value::from(*timestamp));
            for (path, value) in metrics {
                object.insert(
                    path.clone(),
                    serde_json::Number::from_f64(*value)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null),
                );
            }
            serde_json::Value::Object(object)
        })
        .collect();
    serde_json::to_string_pretty(&payload)
        .map_err(|error| OptimError::Other(format!("failed to serialise metrics as JSON: {error}")))
}

fn render_csv(rows: &[ExportRow]) -> String {
    let mut columns: BTreeSet<&str> = BTreeSet::new();
    for (_, metrics) in rows {
        for path in metrics.keys() {
            columns.insert(path.as_str());
        }
    }
    let mut out = String::from("timestamp");
    for column in &columns {
        out.push(',');
        out.push_str(column);
    }
    out.push('\n');
    for (timestamp, metrics) in rows {
        out.push_str(&timestamp.to_string());
        for column in &columns {
            out.push(',');
            if let Some(value) = metrics.get(*column) {
                out.push_str(&format!("{value}"));
            }
        }
        out.push('\n');
    }
    out
}

fn render_prometheus(rows: &[ExportRow]) -> String {
    let mut out = String::new();
    // Prometheus scrapes the current value, so only the newest row is emitted.
    if let Some((timestamp, metrics)) = rows.last() {
        for (path, value) in metrics {
            let name = format!("optirs_streaming_{}", sanitize_metric_name(path));
            out.push_str(&format!("# TYPE {name} gauge\n"));
            out.push_str(&format!("{name} {value} {}\n", timestamp * 1000));
        }
    }
    out
}

fn render_influx(rows: &[ExportRow]) -> String {
    let mut out = String::new();
    for (timestamp, metrics) in rows {
        if metrics.is_empty() {
            continue;
        }
        let fields: Vec<String> = metrics
            .iter()
            .map(|(path, value)| format!("{}={value}", sanitize_metric_name(path)))
            .collect();
        out.push_str(&format!(
            "optirs_streaming {} {}\n",
            fields.join(","),
            timestamp * 1_000_000_000
        ));
    }
    out
}

fn format_extension(format: &ExportFormat) -> &'static str {
    match format {
        ExportFormat::Json => "json",
        ExportFormat::Csv => "csv",
        ExportFormat::Prometheus => "prom",
        ExportFormat::InfluxDB => "influx",
        ExportFormat::Parquet | ExportFormat::Custom { .. } => "bin",
    }
}

impl<A: Float + Default + Clone + std::fmt::Debug + Send + Sync> StreamingMetricsCollector<A> {
    /// Export metrics to configured destinations.
    ///
    /// Returns the files actually written. Formats and destinations that this
    /// crate cannot produce yield an error instead of a silent success.
    ///
    /// Honours `ExportConfig::frequency`: a call that arrives sooner than the
    /// configured interval writes nothing. Use [`Self::export_metrics_now`] to
    /// bypass the throttle.
    pub fn export_metrics(&mut self) -> Result<Vec<PathBuf>> {
        let now = SystemTime::now();
        if let Some(last) = self.last_export {
            if saturating_elapsed(now, last) < self.export_config.frequency {
                return Ok(Vec::new());
            }
        }
        self.export_metrics_now()
    }

    /// Export immediately, ignoring `ExportConfig::frequency`.
    pub fn export_metrics_now(&mut self) -> Result<Vec<PathBuf>> {
        self.last_export = Some(SystemTime::now());
        self.export_metrics_inner()
    }

    fn export_metrics_inner(&mut self) -> Result<Vec<PathBuf>> {
        // A byte-level codec was requested for the metric payload. optirs-core
        // has no compression dependency, so writing plain bytes here would
        // quietly violate the configuration.
        match self.historical_data.compression_config.algorithm {
            CompressionAlgorithm::Gzip | CompressionAlgorithm::Lz4 | CompressionAlgorithm::Zstd => {
                if self.historical_data.compression_config.enabled {
                    return Err(OptimError::UnsupportedOperation(format!(
                        "export requested with byte-level compression ({:?}) but optirs-core \
                         bundles no compression codec; use CompressionAlgorithm::Custom for \
                         temporal compression or disable compression",
                        self.historical_data.compression_config.algorithm
                    )));
                }
            }
            CompressionAlgorithm::None | CompressionAlgorithm::Custom => {}
        }

        let rows = self.export_rows();
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let formats = self.export_config.formats.clone();
        let destinations = self.export_config.destinations.clone();
        let mut written = Vec::new();

        for format in &formats {
            let payload = match format {
                ExportFormat::Json => render_json(&rows)?,
                ExportFormat::Csv => render_csv(&rows),
                ExportFormat::Prometheus => render_prometheus(&rows),
                ExportFormat::InfluxDB => render_influx(&rows),
                ExportFormat::Parquet => {
                    return Err(OptimError::UnsupportedOperation(
                        "Parquet export requires a columnar writer that optirs-core does not \
                         depend on"
                            .to_string(),
                    ));
                }
                ExportFormat::Custom { format } => {
                    return Err(OptimError::UnsupportedOperation(format!(
                        "no writer is registered for the custom export format '{format}'"
                    )));
                }
            };

            for destination in &destinations {
                match destination {
                    ExportDestination::File { path } => {
                        let mut file_path = PathBuf::from(path);
                        let extension = format_extension(format);
                        match file_path.extension() {
                            Some(existing) if existing == extension => {}
                            _ => {
                                let name = file_path
                                    .file_name()
                                    .map(|name| name.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| "streaming_metrics".to_string());
                                file_path.set_file_name(format!("{name}.{extension}"));
                            }
                        }
                        if let Some(parent) = file_path.parent() {
                            if !parent.as_os_str().is_empty() {
                                std::fs::create_dir_all(parent).map_err(OptimError::IO)?;
                            }
                        }
                        let mut file = std::fs::File::create(&file_path).map_err(OptimError::IO)?;
                        file.write_all(payload.as_bytes()).map_err(OptimError::IO)?;
                        written.push(file_path);
                    }
                    ExportDestination::Database { .. }
                    | ExportDestination::S3 { .. }
                    | ExportDestination::Http { .. }
                    | ExportDestination::Kafka { .. } => {
                        return Err(OptimError::UnsupportedOperation(
                            "optirs-core has no database, object-store, HTTP or Kafka client; \
                             only file destinations can be written from this crate"
                                .to_string(),
                        ));
                    }
                }
            }
        }

        Ok(written)
    }

    /// Flatten the most recent `batch_size` snapshots into exportable rows.
    fn export_rows(&self) -> Vec<ExportRow> {
        let batch = self.export_config.batch_size.max(1);
        let mut rows: Vec<ExportRow> = Vec::new();
        for (_key, snapshot) in self.historical_data.time_series.iter().rev().take(batch) {
            let mut metrics = BTreeMap::new();
            for path in KNOWN_METRIC_PATHS {
                if let Some(value) = resolve_snapshot_metric(snapshot, path) {
                    if value.is_finite() {
                        metrics.insert((*path).to_string(), value);
                    }
                }
            }
            // Rows carry the second-resolution timestamp, not the micros key.
            rows.push((snapshot.timestamp, metrics));
        }
        rows.reverse();
        rows
    }
}
