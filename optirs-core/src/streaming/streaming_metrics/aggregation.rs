// Historical storage, retention and real aggregation (findings M2, M3).
//
// * `store_snapshot` used to insert into an unbounded `BTreeMap` and never
//   prune, so a long-running process grew until it died.
// * `get_aggregated` returned `Ok(Vec::new())` unconditionally.
// * `get_range` panicked on any `SystemTime` before the Unix epoch.

use super::accumulator::{mean_f64, variance_f64};
use super::alerts::{resolve_snapshot_metric, KNOWN_METRIC_PATHS};
use super::*;
use crate::error::OptimError;

/// Aggregate statistics for one metric path over one bucket.
#[derive(Debug, Clone, Default)]
pub struct AggregatedSeries {
    /// Observations that contributed
    pub count: usize,
    /// Arithmetic mean
    pub mean: Option<f64>,
    /// Median
    pub median: Option<f64>,
    /// Minimum
    pub min: Option<f64>,
    /// Maximum
    pub max: Option<f64>,
    /// Sum
    pub sum: Option<f64>,
    /// Population standard deviation
    pub std_dev: Option<f64>,
    /// Requested percentiles, keyed by percentile rank
    pub percentiles: BTreeMap<u8, f64>,
}

impl AggregatedSeries {
    fn from_values(values: &[f64], functions: &[AggregationFunction]) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut series = Self {
            count: values.len(),
            ..Self::default()
        };
        if values.is_empty() {
            return series;
        }
        let last = sorted.len() - 1;
        for function in functions {
            match function {
                AggregationFunction::Mean => series.mean = Some(mean_f64(values)),
                AggregationFunction::Median => {
                    series.median = Some(sorted[sorted.len() / 2]);
                }
                AggregationFunction::Min => series.min = Some(sorted[0]),
                AggregationFunction::Max => series.max = Some(sorted[last]),
                AggregationFunction::Sum => series.sum = Some(values.iter().sum()),
                AggregationFunction::Count => {}
                AggregationFunction::StdDev => {
                    series.std_dev = Some(variance_f64(values).sqrt());
                }
                AggregationFunction::Percentile(rank) => {
                    let clamped = (*rank).min(100) as f64 / 100.0;
                    let index = ((sorted.len() as f64 * clamped) as usize).min(last);
                    series.percentiles.insert(*rank, sorted[index]);
                }
            }
        }
        series
    }
}

impl<A: Float + Send + Sync> HistoricalMetrics<A> {
    /// Store a snapshot, then enforce retention, storage and compression
    /// limits so the series stays bounded (M3).
    pub(crate) fn store_snapshot(&mut self, snapshot: MetricsSnapshot<A>) -> Result<()> {
        // Keyed at microsecond resolution: a second-resolution key collapsed
        // every sample within a second onto one entry.
        self.time_series.insert(snapshot.timestamp_micros, snapshot);
        self.prune();
        Ok(())
    }

    /// Approximate bytes held by one retained snapshot.
    fn snapshot_bytes() -> u64 {
        std::mem::size_of::<MetricsSnapshot<A>>() as u64 + std::mem::size_of::<u64>() as u64
    }

    /// Drop data outside the retention window, apply the storage-size cap and
    /// (optionally) collapse temporally redundant points.
    pub(crate) fn prune(&mut self) {
        if !self.retention_policy.auto_cleanup {
            // Even with automatic cleanup disabled the hard storage cap is
            // enforced, otherwise the series is unbounded by construction.
            self.enforce_storage_cap();
            return;
        }

        if let Some(&newest) = self.time_series.keys().next_back() {
            // `raw_data_retention` is expressed in seconds; the keys are micros.
            let window_micros = self
                .retention_policy
                .raw_data_retention
                .saturating_mul(MICROS_PER_SEC);
            let cutoff = newest.saturating_sub(window_micros);
            let expired: Vec<u64> = self
                .time_series
                .range(..cutoff)
                .map(|(timestamp, _)| *timestamp)
                .collect();
            for timestamp in expired {
                self.time_series.remove(&timestamp);
            }
        }

        self.compress();
        self.enforce_storage_cap();
    }

    fn enforce_storage_cap(&mut self) {
        let per_snapshot = Self::snapshot_bytes().max(1);
        let max_snapshots = (self.retention_policy.max_storage_size / per_snapshot).max(1) as usize;
        while self.time_series.len() > max_snapshots {
            let Some(&oldest) = self.time_series.keys().next() else {
                break;
            };
            self.time_series.remove(&oldest);
        }
    }

    /// Temporal (lossy) compression: drop interior points whose watched
    /// metrics stay within `lossy_tolerance` of their neighbours, down to at
    /// most `target_ratio` of the stored points.
    ///
    /// Byte-level codecs (`Gzip`/`Lz4`/`Zstd`) are deliberately *not*
    /// reinterpreted as temporal compression: selecting one of them here only
    /// enables the redundancy pass, and asking the export path for that codec
    /// returns an explicit unsupported-operation error.
    fn compress(&mut self) {
        if !self.compression_config.enabled
            || self.compression_config.algorithm == CompressionAlgorithm::None
        {
            return;
        }
        let total = self.time_series.len();
        if total < 3 {
            return;
        }
        let target =
            ((total as f64) * self.compression_config.target_ratio.clamp(0.0, 1.0)).ceil() as usize;
        let target = target.max(2);
        if total <= target {
            return;
        }

        let tolerance = self.compression_config.lossy_tolerance.max(0.0);
        let watched = [
            "performance.accuracy.current_loss",
            "performance.throughput.samples_per_second",
            "performance.latency.end_to_end.mean",
        ];

        let timestamps: Vec<u64> = self.time_series.keys().copied().collect();
        let mut removable: Vec<(f64, u64)> = Vec::new();
        for window in timestamps.windows(3) {
            let (previous, current, next) = (window[0], window[1], window[2]);
            let mut worst = 0.0f64;
            let mut comparable = false;
            for path in watched {
                let before = self
                    .time_series
                    .get(&previous)
                    .and_then(|s| resolve_snapshot_metric(s, path));
                let middle = self
                    .time_series
                    .get(&current)
                    .and_then(|s| resolve_snapshot_metric(s, path));
                let after = self
                    .time_series
                    .get(&next)
                    .and_then(|s| resolve_snapshot_metric(s, path));
                if let (Some(before), Some(middle), Some(after)) = (before, middle, after) {
                    comparable = true;
                    let interpolated = (before + after) / 2.0;
                    let scale = middle.abs().max(interpolated.abs()).max(f64::EPSILON);
                    worst = worst.max((middle - interpolated).abs() / scale);
                }
            }
            if comparable && worst <= tolerance {
                removable.push((worst, current));
            }
        }

        removable.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut remaining = total;
        for (_, timestamp) in removable {
            if remaining <= target {
                break;
            }
            if self.time_series.remove(&timestamp).is_some() {
                remaining -= 1;
            }
        }
    }

    pub(crate) fn get_range(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<Vec<MetricsSnapshot<A>>> {
        // M4: saturating conversion; a pre-epoch `SystemTime` used to panic.
        // Micros, to match the time-series key resolution.
        let start_ts = unix_timestamp_micros(start_time);
        let end_ts = unix_timestamp_micros(end_time);
        if end_ts < start_ts {
            return Err(OptimError::InvalidParameter(
                "metrics range end precedes its start".to_string(),
            ));
        }

        let snapshots = self
            .time_series
            .range(start_ts..=end_ts)
            .map(|(_, snapshot)| snapshot.clone())
            .collect();

        Ok(snapshots)
    }

    /// Bucket the retained snapshots by `period` and reduce each bucket with
    /// the requested aggregation functions.
    pub(crate) fn get_aggregated(
        &self,
        period: AggregationPeriod,
        start_time: SystemTime,
        end_time: SystemTime,
        functions: &[AggregationFunction],
        extra_paths: &[String],
    ) -> Result<Vec<AggregatedMetrics>> {
        let snapshots = self.get_range(start_time, end_time)?;
        if snapshots.is_empty() {
            return Ok(Vec::new());
        }

        let bucket_seconds = period.seconds().max(1);
        let mut buckets: BTreeMap<u64, Vec<&MetricsSnapshot<A>>> = BTreeMap::new();
        for snapshot in &snapshots {
            let bucket = (snapshot.timestamp / bucket_seconds) * bucket_seconds;
            buckets.entry(bucket).or_default().push(snapshot);
        }

        let mut paths: Vec<&str> = KNOWN_METRIC_PATHS.to_vec();
        for path in extra_paths {
            if !paths.contains(&path.as_str()) {
                paths.push(path.as_str());
            }
        }

        let mut aggregated = Vec::with_capacity(buckets.len());
        for (bucket_start, bucket) in buckets {
            let mut series = BTreeMap::new();
            for path in &paths {
                let values: Vec<f64> = bucket
                    .iter()
                    .filter_map(|snapshot| resolve_snapshot_metric(*snapshot, path))
                    .filter(|value| value.is_finite())
                    .collect();
                if values.is_empty() {
                    // Never measured in this bucket: omit rather than zero-fill.
                    continue;
                }
                series.insert(
                    (*path).to_string(),
                    AggregatedSeries::from_values(&values, functions),
                );
            }

            aggregated.push(AggregatedMetrics {
                period,
                period_start: bucket_start,
                period_end: bucket_start + bucket_seconds,
                sample_count: bucket.len(),
                series,
            });
        }

        Ok(aggregated)
    }
}

impl<A: Float + Default + Clone + std::fmt::Debug + Send + Sync> StreamingMetricsCollector<A> {
    /// Get aggregated metrics for a period, using the configured aggregation
    /// functions.
    pub fn get_aggregated_metrics(
        &self,
        period: AggregationPeriod,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<Vec<AggregatedMetrics>> {
        let requested_window = saturating_elapsed(end_time, start_time);
        if requested_window > self.aggregation_config.max_window {
            return Err(OptimError::InvalidParameter(format!(
                "requested aggregation window of {}s exceeds the configured maximum of {}s",
                requested_window.as_secs(),
                self.aggregation_config.max_window.as_secs()
            )));
        }

        let mut functions = self.aggregation_config.default_functions.clone();
        if functions.is_empty() {
            functions.push(AggregationFunction::Mean);
        }
        let extra_paths: Vec<String> = self
            .aggregation_config
            .custom_aggregations
            .keys()
            .cloned()
            .collect();

        let mut aggregated = self.historical_data.get_aggregated(
            period,
            start_time,
            end_time,
            &functions,
            &extra_paths,
        )?;

        // Apply per-metric overrides where the operator asked for a different
        // set of statistics.
        for (path, overrides) in &self.aggregation_config.custom_aggregations {
            if overrides.is_empty() {
                continue;
            }
            let snapshots = self.historical_data.get_range(start_time, end_time)?;
            let bucket_seconds = period.seconds().max(1);
            for bucket in aggregated.iter_mut() {
                let values: Vec<f64> = snapshots
                    .iter()
                    .filter(|snapshot| {
                        (snapshot.timestamp / bucket_seconds) * bucket_seconds
                            == bucket.period_start
                    })
                    .filter_map(|snapshot| resolve_snapshot_metric(snapshot, path))
                    .filter(|value| value.is_finite())
                    .collect();
                if values.is_empty() {
                    continue;
                }
                bucket.series.insert(
                    path.clone(),
                    AggregatedSeries::from_values(&values, overrides),
                );
            }
        }

        Ok(aggregated)
    }

    /// Aggregate over every configured interval, returning one bucket list per
    /// interval that maps onto a known aggregation period.
    pub fn get_aggregated_for_configured_intervals(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<Vec<(Duration, Vec<AggregatedMetrics>)>> {
        let mut results = Vec::new();
        for interval in &self.aggregation_config.intervals {
            let period = match interval.as_secs() {
                0..=60 => AggregationPeriod::Minute,
                61..=3_600 => AggregationPeriod::Hour,
                3_601..=86_400 => AggregationPeriod::Day,
                86_401..=604_800 => AggregationPeriod::Week,
                _ => AggregationPeriod::Month,
            };
            results.push((
                *interval,
                self.get_aggregated_metrics(period, start_time, end_time)?,
            ));
        }
        Ok(results)
    }
}
