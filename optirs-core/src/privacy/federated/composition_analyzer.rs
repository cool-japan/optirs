// Federated Composition Analyzer Module
//
// This module implements privacy composition analysis for federated learning,
// tracking privacy budget consumption across multiple rounds and providing
// various composition methods for differential privacy guarantees.

use crate::error::{OptimError, Result};
use std::collections::HashMap;

/// Federated composition methods.
///
/// Every variant treats the caller-supplied per-round `(epsilon, delta)` as a
/// black-box `(ε, δ)`-DP guarantee and composes it over `round` applications.
/// Mechanism-specific accountants (moments / RDP) that need the noise multiplier
/// `σ` and the sampling rate `q` cannot be reconstructed from `(ε, δ)` alone; for
/// tight per-step RDP accounting use [`crate::privacy::accountant`] instead.
#[derive(Debug, Clone, Copy, Default)]
pub enum FederatedCompositionMethod {
    /// Basic (linear) composition: `ε' = k·ε`. Always a valid upper bound.
    Basic,

    /// Advanced composition (Dwork–Roth, Thm 3.20):
    /// `ε' = √(2k·ln(1/δ'))·ε + k·ε·(e^ε − 1)`, returned as `min(ε', k·ε)`.
    /// The caller's `delta` plays the role of the advanced-composition slack `δ'`;
    /// the composed mechanism's total failure probability is `k·δ_round + δ'`.
    AdvancedComposition,

    /// Moments-accountant-style composition.
    ///
    /// A true moments accountant needs `σ` and `q`, which `(ε, δ)` does not carry,
    /// so this returns the conservative advanced-composition bound (an upper bound,
    /// never an under-estimate). Use [`crate::privacy::accountant`] for tight RDP.
    #[default]
    FederatedMomentsAccountant,

    /// Rényi-DP-style composition.
    ///
    /// Like [`Self::FederatedMomentsAccountant`], RDP needs `σ`/`q`; this returns the
    /// conservative advanced-composition bound. Use [`crate::privacy::accountant`].
    RenyiDP,

    /// Zero-concentrated DP composition.
    ///
    /// Interprets the per-round `(ε, δ)` as a `ρ`-zCDP guarantee (the largest `ρ`
    /// consistent with `ε = ρ + 2√(ρ·ln(1/δ))`), composes `ρ_total = k·ρ`, and
    /// converts back. Only valid when the per-round mechanism really is `ρ`-zCDP
    /// (e.g. Gaussian); otherwise prefer [`Self::AdvancedComposition`].
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

    /// Compose the per-round `(epsilon, delta)` guarantee over `round` rounds.
    ///
    /// Returns the total `ε` under the configured composition method. See
    /// [`FederatedCompositionMethod`] for the exact bound each variant applies and
    /// its assumptions. Every returned value is a valid *upper* bound on the true
    /// privacy loss for its declared assumptions — the function never under-reports
    /// via the old `ε·√k` / `ε·ln(k)` heuristics, which silently voided the DP
    /// guarantee.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] unless `epsilon > 0`,
    /// `0 < delta < 1`, and `round >= 1`, all finite. A non-finite result is also
    /// rejected so a NaN can never masquerade as a passing budget check.
    pub fn analyze_composition(&self, round: usize, epsilon: f64, delta: f64) -> Result<f64> {
        validate_composition_params(round, epsilon, delta)?;
        let k = round as f64;

        let total_epsilon = match self.method {
            FederatedCompositionMethod::Basic => k * epsilon,
            // Mechanism-specific accountants need σ/q, which (ε, δ) lacks; the
            // conservative advanced-composition bound is the tightest honest
            // answer from (ε, δ) alone and never under-reports.
            FederatedCompositionMethod::AdvancedComposition
            | FederatedCompositionMethod::FederatedMomentsAccountant
            | FederatedCompositionMethod::RenyiDP => {
                advanced_composition_epsilon(k, epsilon, delta)
            }
            FederatedCompositionMethod::ZCDP => zcdp_composition_epsilon(k, epsilon, delta),
        };

        if !total_epsilon.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "composed epsilon is not finite (round={round}, epsilon={epsilon}, delta={delta})"
            )));
        }

        Ok(total_epsilon)
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

/// Validate the per-round composition parameters.
///
/// `epsilon > 0`, `0 < delta < 1`, `round >= 1`, all finite. Rejecting these up
/// front is the single most important guard: without it `delta = 0` makes
/// `ln(1/δ)` infinite and a negative `delta` yields `NaN`, and a `NaN` budget
/// compares `false` against every threshold, so the budget check silently passes.
fn validate_composition_params(round: usize, epsilon: f64, delta: f64) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "epsilon must be a positive finite number, got {epsilon}"
        )));
    }
    if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "delta must lie in the open interval (0, 1), got {delta}"
        )));
    }
    if round < 1 {
        return Err(OptimError::InvalidParameter(
            "round must be at least 1".to_string(),
        ));
    }
    Ok(())
}

/// Advanced composition bound (Dwork–Roth, *The Algorithmic Foundations of
/// Differential Privacy*, Thm 3.20):
///
/// `ε' = √(2k·ln(1/δ'))·ε + k·ε·(e^ε − 1)`, returned as `min(ε', k·ε)`.
///
/// The `min` with basic composition matters at small `k`, where the advanced
/// bound can exceed the trivial linear one. Callers must have validated the
/// parameters (see [`validate_composition_params`]); with `0 < delta < 1` the
/// `ln(1/delta)` term is finite and positive.
fn advanced_composition_epsilon(k: f64, epsilon: f64, delta: f64) -> f64 {
    let advanced =
        (2.0 * k * (1.0 / delta).ln()).sqrt() * epsilon + k * epsilon * (epsilon.exp() - 1.0);
    advanced.min(k * epsilon)
}

/// Zero-concentrated DP composition.
///
/// Interprets the per-round `(ε, δ)` as `ρ`-zCDP with the largest `ρ` consistent
/// with the tight conversion `ε = ρ + 2√(ρ·ln(1/δ))`. Writing `a = ln(1/δ)` and
/// `u = √ρ`, that quadratic solves to `u = √(a + ε) − √a`, so
/// `ρ = (√(a + ε) − √a)²`. zCDP composes additively, `ρ_total = k·ρ`, and converts
/// back with `ε_total = ρ_total + 2√(ρ_total·a)`.
///
/// Taking the largest consistent `ρ` makes the composed `ε` the most conservative
/// (largest) value, so this never under-reports under the zCDP assumption. Callers
/// must have validated the parameters; `a > 0` because `0 < delta < 1`.
fn zcdp_composition_epsilon(k: f64, epsilon: f64, delta: f64) -> f64 {
    let a = (1.0 / delta).ln();
    let u = (a + epsilon).sqrt() - a.sqrt();
    let rho = u * u;
    let rho_total = k * rho;
    rho_total + 2.0 * (rho_total * a).sqrt()
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

        let epsilon = analyzer
            .analyze_composition(5, 0.1, 1e-5)
            .expect("unwrap failed");
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
        let epsilon = analyzer
            .analyze_composition(3, 0.1, 1e-5)
            .expect("unwrap failed");
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
        assert_eq!(compositions.expect("unwrap failed").len(), 1);
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

    /// Golden value: advanced composition at k=1000, ε=0.1, δ=1e-5.
    /// Dwork–Roth Thm 3.20 gives ≈25.69; the old buggy `sqrt(...)` heuristic
    /// returned ≈5.03 (a ~5× under-report that voided the DP guarantee).
    #[test]
    fn test_advanced_composition_golden_value() {
        let analyzer =
            FederatedCompositionAnalyzer::new(FederatedCompositionMethod::AdvancedComposition);
        let eps = analyzer
            .analyze_composition(1000, 0.1, 1e-5)
            .expect("valid params");
        assert!(
            (eps - 25.69).abs() < 0.1,
            "advanced composition = {eps}, expected ≈25.69"
        );
        // Must never fall back to the old under-reporting value.
        assert!(eps > 5.03, "must exceed the old buggy 5.03 under-report");
        // And must never exceed basic composition (k·ε = 100).
        assert!(eps <= 100.0);
    }

    /// Moments/Rényi variants must not under-report: they return the conservative
    /// advanced-composition bound, never the old `ε·√k` (≈3.16) or `ε·ln(k)`.
    #[test]
    fn test_moments_and_renyi_are_conservative() {
        let advanced =
            FederatedCompositionAnalyzer::new(FederatedCompositionMethod::AdvancedComposition)
                .analyze_composition(1000, 0.1, 1e-5)
                .expect("valid");

        for method in [
            FederatedCompositionMethod::FederatedMomentsAccountant,
            FederatedCompositionMethod::RenyiDP,
        ] {
            let eps = FederatedCompositionAnalyzer::new(method)
                .analyze_composition(1000, 0.1, 1e-5)
                .expect("valid");
            assert!(
                (eps - advanced).abs() < 1e-9,
                "{method:?} must equal the advanced-composition bound, got {eps}"
            );
            assert!(eps > 3.16, "{method:?} must exceed old ε·√k under-report");
        }
    }

    /// zCDP composition: proper ε→ρ inversion, composition, and back-conversion.
    /// Golden value at k=1000, ε=0.1, δ=1e-5 is ≈3.38 (hand-computed), pinned so a
    /// future algebra regression in the inversion is caught.
    #[test]
    fn test_zcdp_composition_golden_value() {
        let analyzer = FederatedCompositionAnalyzer::new(FederatedCompositionMethod::ZCDP);
        let one = analyzer.analyze_composition(1, 0.1, 1e-5).expect("valid");
        let many = analyzer
            .analyze_composition(1000, 0.1, 1e-5)
            .expect("valid");
        assert!(one > 0.0 && one.is_finite());
        // A single ρ-round returns an ε at least the input.
        assert!(one >= 0.1 - 1e-9, "single-round zCDP ε = {one}");
        assert!(many > one, "zCDP must accumulate across rounds");
        assert!(
            (many - 3.38).abs() < 0.05,
            "zCDP composition = {many}, expected ≈3.38"
        );
    }

    /// At k=1 advanced composition must not undercut basic composition (the `min`
    /// with `k·ε` is what guarantees this).
    #[test]
    fn test_advanced_composition_single_round_matches_basic() {
        let analyzer =
            FederatedCompositionAnalyzer::new(FederatedCompositionMethod::AdvancedComposition);
        let eps = analyzer.analyze_composition(1, 0.5, 1e-5).expect("valid");
        assert!(
            (eps - 0.5).abs() < 1e-12,
            "k=1 advanced should equal ε, got {eps}"
        );
    }

    /// F36: invalid parameters must be rejected, never silently returned as a NaN
    /// or infinite budget that passes every downstream threshold check.
    #[test]
    fn test_invalid_parameters_are_rejected() {
        let analyzer =
            FederatedCompositionAnalyzer::new(FederatedCompositionMethod::AdvancedComposition);
        // delta = 0 (the old code produced +inf here).
        assert!(analyzer.analyze_composition(10, 0.1, 0.0).is_err());
        // delta >= 1.
        assert!(analyzer.analyze_composition(10, 0.1, 1.0).is_err());
        // negative delta (the old code produced NaN).
        assert!(analyzer.analyze_composition(10, 0.1, -1e-5).is_err());
        // non-positive epsilon.
        assert!(analyzer.analyze_composition(10, 0.0, 1e-5).is_err());
        assert!(analyzer.analyze_composition(10, -0.1, 1e-5).is_err());
        // round = 0.
        assert!(analyzer.analyze_composition(0, 0.1, 1e-5).is_err());
        // non-finite epsilon.
        assert!(analyzer.analyze_composition(10, f64::NAN, 1e-5).is_err());
    }
}
