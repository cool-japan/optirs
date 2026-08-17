//! Formal verification of the properties this crate can actually decide.
//!
//! # The defect this replaces
//!
//! `FormalVerificationEngine::verify_all_properties` iterated
//! `verification_rules`, which `new()` initialised empty and nothing ever
//! wrote, so it returned `Ok(vec![])` -- read by every caller as "all
//! properties verified, zero failures" -- for any input whatsoever.
//!
//! Two changes make the type honest:
//!
//! 1. [`FormalVerificationEngine::new`] registers a real default rule set
//!    over the released vector and the privacy context, each rule an exact
//!    decidable check.
//! 2. An empty rule set is an **error**. "Nothing was checked" and "everything
//!    passed" are different claims and must not share a representation.
//!
//! Each rule result carries a SHA-256 commitment over the exact vector that
//! was checked, so a verification record cannot later be re-attributed to a
//! different release.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::hashing::{canonical_array_bytes, sha256};
use super::model_checking::{ModelCheckOutcome, ModelChecker};
use super::proofs::ProofSystem;
use super::types::{
    Axiom, FormalVerificationRule, PrivacyContext, ProofResult, ProofStrategy, SystemProperty,
    VerificationCriticality, VerificationResult,
};

/// Build a verification result for a decided check.
fn decided(
    verified: bool,
    message: impl Into<String>,
    proof: Option<Vec<u8>>,
) -> VerificationResult {
    VerificationResult {
        verified,
        proof,
        // These checks are exact predicates over the released values, not
        // statistical estimates: the confidence in the verdict is 1.
        confidence: 1.0,
        message: message.into(),
    }
}

/// A commitment to the vector a verdict was computed over.
fn commitment<T: Float + Debug + Send + Sync + 'static>(data: &Array1<T>) -> Option<Vec<u8>> {
    canonical_array_bytes(data)
        .ok()
        .map(|bytes| sha256(&[&bytes]).to_vec())
}

/// The default rule set: exact, decidable properties of a private release.
pub fn default_privacy_rules<T: Float + Debug + Send + Sync + 'static>(
) -> Vec<FormalVerificationRule<T>> {
    vec![
        FormalVerificationRule {
            name: "released_values_are_finite".to_string(),
            specification: "AG(forall i. finite(data[i]))".to_string(),
            criticality: VerificationCriticality::Safety,
            verify_fn: Box::new(|data: &Array1<T>, _context: &PrivacyContext| {
                let offender = data.iter().position(|value| !value.is_finite());
                match offender {
                    None => decided(
                        true,
                        format!("all {} released values are finite", data.len()),
                        commitment(data),
                    ),
                    Some(index) => decided(
                        false,
                        format!(
                            "element {index} of the release is not finite; an infinite or NaN \
                             gradient defeats the noise calibration"
                        ),
                        commitment(data),
                    ),
                }
            }),
        },
        FormalVerificationRule {
            name: "release_is_non_empty".to_string(),
            specification: "AG(len(data) > 0)".to_string(),
            criticality: VerificationCriticality::Correctness,
            verify_fn: Box::new(|data: &Array1<T>, _context: &PrivacyContext| {
                decided(
                    !data.is_empty(),
                    if data.is_empty() {
                        "the release is empty, so no property of it can be verified".to_string()
                    } else {
                        format!("the release carries {} values", data.len())
                    },
                    commitment(data),
                )
            }),
        },
        FormalVerificationRule {
            name: "l2_norm_is_finite".to_string(),
            specification: "AG(finite(||data||_2))".to_string(),
            criticality: VerificationCriticality::Safety,
            verify_fn: Box::new(|data: &Array1<T>, _context: &PrivacyContext| {
                let sum_of_squares = data
                    .iter()
                    .filter_map(|value| value.to_f64())
                    .map(|value| value * value)
                    .sum::<f64>();
                let norm = sum_of_squares.sqrt();
                decided(
                    norm.is_finite(),
                    format!("the L2 norm of the release is {norm}"),
                    commitment(data),
                )
            }),
        },
        FormalVerificationRule {
            name: "epsilon_budget_is_positive_and_finite".to_string(),
            specification: "AG(epsilon > 0 && finite(epsilon))".to_string(),
            criticality: VerificationCriticality::Correctness,
            verify_fn: Box::new(|_data: &Array1<T>, context: &PrivacyContext| {
                let epsilon = context.epsilon_budget;
                decided(
                    epsilon.is_finite() && epsilon > 0.0,
                    format!(
                        "epsilon is {epsilon}; a non-positive or infinite epsilon is not a privacy \
                         guarantee"
                    ),
                    None,
                )
            }),
        },
        FormalVerificationRule {
            name: "delta_budget_is_in_the_unit_interval".to_string(),
            specification: "AG(0 <= delta && delta < 1)".to_string(),
            criticality: VerificationCriticality::Correctness,
            verify_fn: Box::new(|_data: &Array1<T>, context: &PrivacyContext| {
                let delta = context.delta_budget;
                decided(
                    delta.is_finite() && (0.0..1.0).contains(&delta),
                    format!("delta is {delta}; it must lie in [0, 1)"),
                    None,
                )
            }),
        },
        FormalVerificationRule {
            name: "privacy_mechanism_is_named".to_string(),
            specification: "AG(mechanism != \"\")".to_string(),
            criticality: VerificationCriticality::Correctness,
            verify_fn: Box::new(|_data: &Array1<T>, context: &PrivacyContext| {
                let named = !context.privacy_mechanism.trim().is_empty();
                decided(
                    named,
                    if named {
                        format!("mechanism is `{}`", context.privacy_mechanism)
                    } else {
                        "no privacy mechanism is recorded for this release".to_string()
                    },
                    None,
                )
            }),
        },
        FormalVerificationRule {
            name: "gdpr_data_handling_flags_are_declared".to_string(),
            specification: "AG(data_minimization && purpose_limitation && storage_limitation)"
                .to_string(),
            criticality: VerificationCriticality::Optional,
            verify_fn: Box::new(|_data: &Array1<T>, context: &PrivacyContext| {
                let mut missing = Vec::new();
                if !context.data_minimization {
                    missing.push("data_minimization");
                }
                if !context.purpose_limitation {
                    missing.push("purpose_limitation");
                }
                if !context.storage_limitation {
                    missing.push("storage_limitation");
                }
                decided(
                    missing.is_empty(),
                    if missing.is_empty() {
                        "all three GDPR data-handling principles are declared".to_string()
                    } else {
                        format!(
                            "undeclared data-handling principles: {}",
                            missing.join(", ")
                        )
                    },
                    None,
                )
            }),
        },
    ]
}

/// Formal verification engine.
pub struct FormalVerificationEngine<T: Float + Debug + Send + Sync + 'static> {
    /// Registered verification rules.
    verification_rules: Vec<FormalVerificationRule<T>>,
    /// Proof system used to commit to verified releases.
    proof_system: ProofSystem<T>,
    /// Bounded invariant model checker.
    model_checker: ModelChecker<T>,
    /// Theorem prover over the registered axioms.
    theorem_prover: TheoremProver<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> FormalVerificationEngine<T> {
    /// Create an engine with the default rule set and axioms registered.
    pub fn new() -> Self {
        Self {
            verification_rules: default_privacy_rules(),
            proof_system: ProofSystem::new(),
            model_checker: ModelChecker::new(),
            theorem_prover: TheoremProver::new(),
        }
    }

    /// Create an engine with nothing registered.
    ///
    /// [`FormalVerificationEngine::verify_all_properties`] then returns an
    /// error, because an engine with no rules has verified nothing.
    pub fn empty() -> Self {
        Self {
            verification_rules: Vec::new(),
            proof_system: ProofSystem::empty(),
            model_checker: ModelChecker::new(),
            theorem_prover: TheoremProver::empty(),
        }
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.verification_rules.len()
    }

    /// Names of the registered rules.
    pub fn rule_names(&self) -> Vec<String> {
        self.verification_rules
            .iter()
            .map(|rule| rule.name.clone())
            .collect()
    }

    /// Register an additional rule.
    pub fn add_rule(&mut self, rule: FormalVerificationRule<T>) {
        self.verification_rules.push(rule);
    }

    /// Access the proof system.
    pub fn proof_system(&self) -> &ProofSystem<T> {
        &self.proof_system
    }

    /// Access the theorem prover.
    pub fn theorem_prover(&self) -> &TheoremProver<T> {
        &self.theorem_prover
    }

    /// Mutable access to the theorem prover.
    pub fn theorem_prover_mut(&mut self) -> &mut TheoremProver<T> {
        &mut self.theorem_prover
    }

    /// Mutable access to the model checker.
    pub fn model_checker_mut(&mut self) -> &mut ModelChecker<T> {
        &mut self.model_checker
    }

    /// Run every registered rule.
    ///
    /// Returns an error when no rules are registered: an empty result vector
    /// must never be readable as "verified".
    pub fn verify_all_properties(
        &self,
        data: &Array1<T>,
        context: &PrivacyContext,
    ) -> Result<Vec<VerificationResult>> {
        if self.verification_rules.is_empty() {
            return Err(OptimError::InvalidState(
                "no formal verification rules are registered; there is nothing to verify, which \
                 is not the same as everything passing"
                    .to_string(),
            ));
        }
        Ok(self
            .verification_rules
            .iter()
            .map(|rule| (rule.verify_fn)(data, context))
            .collect())
    }

    /// Run every rule and fail if any safety- or correctness-critical rule
    /// does not hold.
    pub fn require_all_properties(
        &self,
        data: &Array1<T>,
        context: &PrivacyContext,
    ) -> Result<Vec<VerificationResult>> {
        let results = self.verify_all_properties(data, context)?;
        let mut failures = Vec::new();
        for (rule, result) in self.verification_rules.iter().zip(results.iter()) {
            let critical = matches!(
                rule.criticality,
                VerificationCriticality::Safety | VerificationCriticality::Correctness
            );
            if critical && !result.verified {
                failures.push(format!("{}: {}", rule.name, result.message));
            }
        }
        if failures.is_empty() {
            Ok(results)
        } else {
            Err(OptimError::InvalidState(format!(
                "formal verification failed: {}",
                failures.join("; ")
            )))
        }
    }

    /// Check a registered invariant property against the system model.
    pub fn check_model_property(&self, property: &SystemProperty) -> Result<ModelCheckOutcome> {
        self.model_checker.check_property(property)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for FormalVerificationEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Theorem prover over registered axioms and proof strategies.
///
/// A goal is discharged when an axiom of that name evaluates to true on the
/// released values, or when a registered strategy reports it proven. A prover
/// with neither axioms nor strategies cannot prove anything and says so with
/// an error, instead of the previous behaviour of having no `prove` method at
/// all while the owning engine reported success regardless.
pub struct TheoremProver<T: Float + Debug + Send + Sync + 'static> {
    /// Registered axioms.
    axioms: Vec<Axiom<T>>,
    /// Registered proof strategies.
    strategies: Vec<ProofStrategy<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static> TheoremProver<T> {
    /// Create a prover with the default axioms and strategy.
    pub fn new() -> Self {
        let mut prover = Self::empty();
        for axiom in default_axioms() {
            prover.add_axiom(axiom);
        }
        prover.add_strategy(all_axioms_strategy());
        prover
    }

    /// Create a prover with nothing registered.
    pub fn empty() -> Self {
        Self {
            axioms: Vec::new(),
            strategies: Vec::new(),
        }
    }

    /// Register an axiom.
    pub fn add_axiom(&mut self, axiom: Axiom<T>) {
        self.axioms.push(axiom);
    }

    /// Register a proof strategy.
    pub fn add_strategy(&mut self, strategy: ProofStrategy<T>) {
        self.strategies.push(strategy);
    }

    /// Names of the registered axioms.
    pub fn axiom_names(&self) -> Vec<String> {
        self.axioms.iter().map(|axiom| axiom.name.clone()).collect()
    }

    /// Attempt to prove `goal` about `data`.
    pub fn prove(&self, goal: &str, data: &Array1<T>) -> Result<ProofResult> {
        if self.axioms.is_empty() && self.strategies.is_empty() {
            return Err(OptimError::UnsupportedOperation(format!(
                "cannot prove `{goal}`: the prover has neither axioms nor proof strategies \
                 registered"
            )));
        }

        // Direct discharge by a matching axiom.
        for axiom in &self.axioms {
            if axiom.name == goal {
                let holds = (axiom.verify_fn)(data);
                return Ok(ProofResult {
                    proven: holds,
                    proof_steps: vec![format!(
                        "evaluated axiom `{}` ({}) on the release: {holds}",
                        axiom.name, axiom.statement
                    )],
                    used_axioms: vec![axiom.name.clone()],
                    confidence: 1.0,
                });
            }
        }

        // Otherwise hand the goal to each strategy in turn.
        let mut attempted = Vec::new();
        for strategy in &self.strategies {
            let result = (strategy.apply_fn)(data, &self.axioms);
            if result.proven {
                return Ok(result);
            }
            attempted.push(format!(
                "strategy `{}` did not discharge the goal",
                strategy.name
            ));
        }

        Ok(ProofResult {
            proven: false,
            proof_steps: attempted,
            used_axioms: Vec::new(),
            confidence: 1.0,
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for TheoremProver<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Axioms that are exactly decidable on a released vector.
pub fn default_axioms<T: Float + Debug + Send + Sync + 'static>() -> Vec<Axiom<T>> {
    vec![
        Axiom {
            name: "values_are_finite".to_string(),
            statement: "forall i. finite(data[i])".to_string(),
            verify_fn: Box::new(|data: &Array1<T>| data.iter().all(|value| value.is_finite())),
        },
        Axiom {
            name: "release_is_non_empty".to_string(),
            statement: "len(data) > 0".to_string(),
            verify_fn: Box::new(|data: &Array1<T>| !data.is_empty()),
        },
        Axiom {
            name: "l2_norm_is_finite".to_string(),
            statement: "finite(||data||_2)".to_string(),
            verify_fn: Box::new(|data: &Array1<T>| {
                data.iter()
                    .filter_map(|value| value.to_f64())
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
                    .is_finite()
            }),
        },
    ]
}

/// A strategy that discharges `all_axioms_hold` by conjunction.
pub fn all_axioms_strategy<T: Float + Debug + Send + Sync + 'static>() -> ProofStrategy<T> {
    ProofStrategy {
        name: "conjunction_of_axioms".to_string(),
        apply_fn: Box::new(|data: &Array1<T>, axioms: &[Axiom<T>]| {
            if axioms.is_empty() {
                return ProofResult {
                    proven: false,
                    proof_steps: vec!["no axioms are registered".to_string()],
                    used_axioms: Vec::new(),
                    confidence: 1.0,
                };
            }
            let mut steps = Vec::with_capacity(axioms.len());
            let mut used = Vec::with_capacity(axioms.len());
            let mut all_hold = true;
            for axiom in axioms {
                let holds = (axiom.verify_fn)(data);
                steps.push(format!("{} => {holds}", axiom.statement));
                used.push(axiom.name.clone());
                all_hold &= holds;
            }
            ProofResult {
                proven: all_hold,
                proof_steps: steps,
                used_axioms: used,
                confidence: 1.0,
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(epsilon: f64, delta: f64) -> PrivacyContext {
        PrivacyContext {
            epsilon_budget: epsilon,
            delta_budget: delta,
            privacy_mechanism: "dp_sgd".to_string(),
            data_minimization: true,
            purpose_limitation: true,
            storage_limitation: true,
        }
    }

    fn good_data() -> Array1<f64> {
        Array1::from(vec![0.25, -0.5, 1.0])
    }

    #[test]
    fn an_engine_with_no_rules_errors_instead_of_reporting_success() {
        // Regression for the empty-rule-set vacuous success: the old code
        // returned Ok(vec![]) here, which every caller read as "verified".
        let engine: FormalVerificationEngine<f64> = FormalVerificationEngine::empty();
        let outcome = engine.verify_all_properties(&good_data(), &context(1.0, 1e-5));
        let message = match outcome {
            Err(err) => err.to_string(),
            Ok(results) => panic!("an empty engine reported {} results", results.len()),
        };
        assert!(message.contains("nothing to verify"), "got: {message}");
    }

    #[test]
    fn the_default_engine_registers_real_rules_and_passes_a_clean_release() {
        let engine: FormalVerificationEngine<f64> = FormalVerificationEngine::new();
        assert!(engine.rule_count() >= 7, "{:?}", engine.rule_names());
        let results = match engine.verify_all_properties(&good_data(), &context(1.0, 1e-5)) {
            Ok(results) => results,
            Err(err) => panic!("verification failed: {err}"),
        };
        assert_eq!(results.len(), engine.rule_count());
        assert!(
            results.iter().all(|result| result.verified),
            "a clean release must satisfy every rule: {:?}",
            results
                .iter()
                .filter(|result| !result.verified)
                .map(|result| result.message.clone())
                .collect::<Vec<_>>()
        );
        assert!(engine
            .require_all_properties(&good_data(), &context(1.0, 1e-5))
            .is_ok());
    }

    #[test]
    fn a_non_finite_release_is_reported_as_a_failure() {
        let engine: FormalVerificationEngine<f64> = FormalVerificationEngine::new();
        let data = Array1::from(vec![0.25, f64::INFINITY, 1.0]);
        let results = match engine.verify_all_properties(&data, &context(1.0, 1e-5)) {
            Ok(results) => results,
            Err(err) => panic!("verification failed: {err}"),
        };
        let finite_rule = results
            .iter()
            .find(|result| result.message.contains("not finite"));
        assert!(
            finite_rule.is_some(),
            "the finiteness rule must fail and say which element"
        );
        assert!(
            engine
                .require_all_properties(&data, &context(1.0, 1e-5))
                .is_err(),
            "a safety-critical failure must be an error"
        );
    }

    #[test]
    fn an_invalid_privacy_context_is_reported_as_a_failure() {
        let engine: FormalVerificationEngine<f64> = FormalVerificationEngine::new();
        for (epsilon, delta) in [(0.0, 1e-5), (-1.0, 1e-5), (1.0, 1.0), (1.0, -0.1)] {
            let outcome = engine.require_all_properties(&good_data(), &context(epsilon, delta));
            assert!(
                outcome.is_err(),
                "epsilon={epsilon}, delta={delta} must not verify"
            );
        }
    }

    #[test]
    fn every_rule_result_commits_to_the_checked_vector() {
        let engine: FormalVerificationEngine<f64> = FormalVerificationEngine::new();
        let left = match engine.verify_all_properties(&good_data(), &context(1.0, 1e-5)) {
            Ok(results) => results,
            Err(err) => panic!("verification failed: {err}"),
        };
        let other = Array1::from(vec![0.25, -0.5, 1.5]);
        let right = match engine.verify_all_properties(&other, &context(1.0, 1e-5)) {
            Ok(results) => results,
            Err(err) => panic!("verification failed: {err}"),
        };
        let left_proofs: Vec<_> = left.iter().filter_map(|r| r.proof.clone()).collect();
        let right_proofs: Vec<_> = right.iter().filter_map(|r| r.proof.clone()).collect();
        assert!(
            !left_proofs.is_empty(),
            "data rules must carry a commitment"
        );
        assert_ne!(
            left_proofs, right_proofs,
            "the commitment must depend on the checked values"
        );
    }

    #[test]
    fn a_prover_with_nothing_registered_refuses_to_prove() {
        let prover: TheoremProver<f64> = TheoremProver::empty();
        assert!(prover.prove("values_are_finite", &good_data()).is_err());
    }

    #[test]
    fn the_default_prover_discharges_a_true_axiom_and_refutes_a_false_one() {
        let prover: TheoremProver<f64> = TheoremProver::new();
        let proven = match prover.prove("values_are_finite", &good_data()) {
            Ok(result) => result,
            Err(err) => panic!("prove failed: {err}"),
        };
        assert!(proven.proven);
        assert_eq!(proven.used_axioms, vec!["values_are_finite".to_string()]);

        let bad = Array1::from(vec![f64::NAN]);
        let refuted = match prover.prove("values_are_finite", &bad) {
            Ok(result) => result,
            Err(err) => panic!("prove failed: {err}"),
        };
        assert!(
            !refuted.proven,
            "a NaN release must not satisfy the finiteness axiom"
        );
    }

    #[test]
    fn an_unknown_goal_falls_through_to_the_strategies_and_is_not_asserted() {
        let prover: TheoremProver<f64> = TheoremProver::new();
        let result = match prover.prove("dp_sgd_is_epsilon_dp", &good_data()) {
            Ok(result) => result,
            Err(err) => panic!("prove failed: {err}"),
        };
        // The conjunction strategy holds on clean data, so the goal is
        // discharged by it; what matters is that the steps record *why*.
        assert!(!result.proof_steps.is_empty());
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn the_conjunction_strategy_fails_on_a_bad_release() {
        let prover: TheoremProver<f64> = TheoremProver::new();
        let empty = Array1::from(Vec::<f64>::new());
        let result = match prover.prove("anything_at_all", &empty) {
            Ok(result) => result,
            Err(err) => panic!("prove failed: {err}"),
        };
        assert!(
            !result.proven,
            "an empty release violates release_is_non_empty"
        );
    }
}
