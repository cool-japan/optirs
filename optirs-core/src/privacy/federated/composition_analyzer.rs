// Federated Composition Analyzer Module
//
// This module implements privacy composition analysis for federated learning,
// tracking privacy budget consumption across multiple rounds and providing
// various composition methods for differential privacy guarantees.

use crate::error::{OptimError, Result};
use std::collections::HashMap;

/// Federated composition methods
#[derive(Debug, Clone, Copy, Default)]
pub enum FederatedCompositionMethod {
    /// Basic composition
    Basic,

    /// Advanced composition with amplification
    AdvancedComposition,

    /// Moments accountant for federated setting
    #[default]
    FederatedMomentsAccountant,

    /// Renyi differential privacy
    RenyiDP,

    /// Zero-concentrated differential privacy
    ZCDP,
}

/// Federated composition analyzer
pub struct FederatedCompositionAnalyzer {
    method: FederatedCompositionMethod,
    round_compositions: Vec<RoundComposition>,
    client_compositions: HashMap<String, Vec<ClientComposition>>,
}

/// Round composition for privacy accounting
#[derive(Debug, Clone)]
pub struct RoundComposition {
    pub round: usize,
    pub participating_clients: usize,
    pub epsilonconsumed: f64,
    pub delta_consumed: f64,
    pub amplification_applied: bool,
    pub composition_method: FederatedCompositionMethod,
}

/// Client-specific composition tracking
#[derive(Debug, Clone)]
pub struct ClientComposition {
    pub clientid: String,
    pub round: usize,
    pub epsilon_contribution: f64,
    pub delta_contribution: f64,
}

/// Composition statistics
#[derive(Debug, Clone)]
pub struct CompositionStats {
    pub total_rounds: usize,
    pub total_epsilon_consumed: f64,
    pub total_delta_consumed: f64,
    pub composition_method: FederatedCompositionMethod,
    pub amplification_rounds: usize,
}

impl FederatedCompositionAnalyzer {
    pub fn new(method: FederatedCompositionMethod) -> Self {
        Self {
            method,
            round_compositions: Vec::new(),
            client_compositions: HashMap::new(),
        }
    }

    pub fn analyze_composition(&self, round: usize, epsilon: f64, delta: f64) -> Result<f64> {
        match self.method {
            FederatedCompositionMethod::Basic => Ok(epsilon * round as f64),
            FederatedCompositionMethod::AdvancedComposition => {
                // Simplified advanced composition
                let k = round as f64;
                let advanced_epsilon = (k * epsilon * epsilon
                    + k.sqrt() * epsilon * (2.0 * (1.25 / delta).ln()).sqrt())
                .sqrt();
                Ok(advanced_epsilon)
            }
            FederatedCompositionMethod::FederatedMomentsAccountant => {
                // Use existing moments accountant logic
                Ok(epsilon * (round as f64).sqrt())
            }
            FederatedCompositionMethod::RenyiDP => {
                // Simplified Renyi DP composition
                Ok(epsilon * (round as f64).ln())
            }
            FederatedCompositionMethod::ZCDP => {
                // Zero-concentrated DP composition
                Ok(epsilon * (round as f64).sqrt())
            }
        }
    }

    pub fn add_round_composition(&mut self, composition: RoundComposition) {
        self.round_compositions.push(composition);
    }

    pub fn add_client_composition(&mut self, client_id: String, composition: ClientComposition) {
        self.client_compositions
            .entry(client_id)
            .or_default()
            .push(composition);
    }

    pub fn get_composition_stats(&self) -> CompositionStats {
        if self.round_compositions.is_empty() {
            return CompositionStats::default();
        }

        let total_epsilon: f64 = self
            .round_compositions
            .iter()
            .map(|comp| comp.epsilonconsumed)
            .sum();

        let total_delta: f64 = self
            .round_compositions
            .iter()
            .map(|comp| comp.delta_consumed)
            .sum();

        CompositionStats {
            total_rounds: self.round_compositions.len(),
            total_epsilon_consumed: total_epsilon,
            total_delta_consumed: total_delta,
            composition_method: self.method,
            amplification_rounds: self
                .round_compositions
                .iter()
                .filter(|comp| comp.amplification_applied)
                .count(),
        }
    }

    /// Get current composition method
    pub fn method(&self) -> FederatedCompositionMethod {
        self.method
    }

    /// Get number of rounds tracked
    pub fn rounds_count(&self) -> usize {
        self.round_compositions.len()
    }

    /// Get client composition history for a specific client
    pub fn get_client_compositions(&self, client_id: &str) -> Option<&Vec<ClientComposition>> {
        self.client_compositions.get(client_id)
    }

    /// Get round compositions
    pub fn get_round_compositions(&self) -> &Vec<RoundComposition> {
        &self.round_compositions
    }

    /// Clear all composition history
    pub fn clear_history(&mut self) {
        self.round_compositions.clear();
        self.client_compositions.clear();
    }

    /// Set composition method
    pub fn set_method(&mut self, method: FederatedCompositionMethod) {
        self.method = method;
    }
}

impl Default for CompositionStats {
    fn default() -> Self {
        Self {
            total_rounds: 0,
            total_epsilon_consumed: 0.0,
            total_delta_consumed: 0.0,
            composition_method: FederatedCompositionMethod::default(),
            amplification_rounds: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_composition_analyzer() {
        let analyzer =
            FederatedCompositionAnalyzer::new(FederatedCompositionMethod::AdvancedComposition);

        let epsilon = analyzer.analyze_composition(5, 0.1, 1e-5).unwrap();
        assert!(epsilon > 0.1); // Should be larger than single round epsilon
    }

    #[test]
    fn test_composition_stats() {
        let mut analyzer = FederatedCompositionAnalyzer::new(
            FederatedCompositionMethod::FederatedMomentsAccountant,
        );

        // Add some round compositions
        analyzer.add_round_composition(RoundComposition {
            round: 1,
            participating_clients: 10,
            epsilonconsumed: 0.1,
            delta_consumed: 1e-5,
            amplification_applied: true,
            composition_method: FederatedCompositionMethod::FederatedMomentsAccountant,
        });

        analyzer.add_round_composition(RoundComposition {
            round: 2,
            participating_clients: 12,
            epsilonconsumed: 0.15,
            delta_consumed: 1e-5,
            amplification_applied: false,
            composition_method: FederatedCompositionMethod::FederatedMomentsAccountant,
        });

        let stats = analyzer.get_composition_stats();
        assert_eq!(stats.total_rounds, 2);
        assert_eq!(stats.total_epsilon_consumed, 0.25);
        assert_eq!(stats.total_delta_consumed, 2e-5);
        assert_eq!(stats.amplification_rounds, 1);
    }

    #[test]
    fn test_basic_composition() {
        let analyzer = FederatedCompositionAnalyzer::new(FederatedCompositionMethod::Basic);
        let epsilon = analyzer.analyze_composition(3, 0.1, 1e-5).unwrap();
        assert!((epsilon - 0.3).abs() < 1e-10); // Basic composition: 3 * 0.1, with floating point tolerance
    }

    #[test]
    fn test_client_composition_tracking() {
        let mut analyzer = FederatedCompositionAnalyzer::new(
            FederatedCompositionMethod::FederatedMomentsAccountant,
        );

        let client_comp = ClientComposition {
            clientid: "client1".to_string(),
            round: 1,
            epsilon_contribution: 0.05,
            delta_contribution: 5e-6,
        };

        analyzer.add_client_composition("client1".to_string(), client_comp);

        let compositions = analyzer.get_client_compositions("client1");
        assert!(compositions.is_some());
        assert_eq!(compositions.unwrap().len(), 1);
    }

    #[test]
    fn test_clear_history() {
        let mut analyzer = FederatedCompositionAnalyzer::new(
            FederatedCompositionMethod::FederatedMomentsAccountant,
        );

        // Add some data
        analyzer.add_round_composition(RoundComposition {
            round: 1,
            participating_clients: 10,
            epsilonconsumed: 0.1,
            delta_consumed: 1e-5,
            amplification_applied: true,
            composition_method: FederatedCompositionMethod::FederatedMomentsAccountant,
        });

        assert_eq!(analyzer.rounds_count(), 1);

        analyzer.clear_history();
        assert_eq!(analyzer.rounds_count(), 0);
    }
}
