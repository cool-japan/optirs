// Search history tracking and analysis

use num_traits::Float;
use std::fmt::Debug;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use crate::error::Result;
use super::evaluation::EvaluationMetrics;

/// Search history tracker
pub struct SearchHistory<T: Float + Debug + Send + Sync + 'static> {
    entries: VecDeque<ArchitectureEntry>,
    events: VecDeque<SearchEvent>,
    max_history_size: usize,
    start_time: Instant,
    convergence_history: Vec<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> SearchHistory<T> {
    pub fn new(max_size: usize) -> Result<Self> {
        Ok(Self {
            entries: VecDeque::new(),
            events: VecDeque::new(),
            max_history_size: max_size,
            start_time: Instant::now(),
            convergence_history: Vec::new(),
        })
    }

    pub fn record_evaluation(&mut self, architecture: String, metrics: EvaluationMetrics) -> Result<()> {
        let entry = ArchitectureEntry {
            architecture,
            metrics,
            timestamp: Instant::now(),
            evaluation_id: self.entries.len(),
        };

        self.entries.push_back(entry);

        if self.entries.len() > self.max_history_size {
            self.entries.pop_front();
        }

        self.record_event(SearchEvent::ArchitectureEvaluated);
        Ok(())
    }

    pub fn record_event(&mut self, event: SearchEvent) {
        self.events.push_back(event);

        if self.events.len() > self.max_history_size {
            self.events.pop_front();
        }
    }

    pub fn record_search_completion(&mut self, duration: Duration) -> Result<()> {
        self.record_event(SearchEvent::SearchCompleted { duration });
        Ok(())
    }

    pub fn total_evaluated(&self) -> usize {
        self.entries.len()
    }

    pub fn total_iterations(&self) -> usize {
        self.events.iter().filter(|e| matches!(e, SearchEvent::IterationCompleted { .. })).count()
    }

    pub fn get_recent_best_performance(&self, window: usize) -> Option<f64> {
        self.entries
            .iter()
            .rev()
            .take(window)
            .map(|entry| entry.metrics.overall_score(&[]))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn get_convergence_history(&self) -> Vec<f64> {
        self.convergence_history.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect()
    }
}

/// Architecture entry in search history
#[derive(Debug, Clone)]
pub struct ArchitectureEntry {
    pub architecture: String,
    pub metrics: EvaluationMetrics,
    pub timestamp: Instant,
    pub evaluation_id: usize,
}

/// Search events
#[derive(Debug, Clone)]
pub enum SearchEvent {
    SearchStarted,
    IterationCompleted { iteration: usize },
    ArchitectureEvaluated,
    PopulationUpdated,
    SearchCompleted { duration: Duration },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics() -> EvaluationMetrics {
        EvaluationMetrics {
            accuracy: 0.8,
            training_time_seconds: 100.0,
            inference_time_ms: 10.0,
            memory_usage_mb: 512,
            flops: 1_000_000,
            parameters: 100_000,
            energy_consumption: 50.0,
            convergence_rate: 0.8,
            robustness_score: 0.7,
            generalization_score: 0.72,
            efficiency_score: 0.6,
            valid: true,
        }
    }

    #[test]
    fn test_search_history() {
        let mut history = SearchHistory::<f32>::new(100).unwrap();

        let metrics = create_test_metrics();
        let result = history.record_evaluation("test_arch".to_string(), metrics);
        assert!(result.is_ok());

        assert_eq!(history.total_evaluated(), 1);
    }
}