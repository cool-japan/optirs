//! Bounded invariant model checking over the declared system model.
//!
//! # What is and is not implemented
//!
//! `ModelChecker` used to be a constructor with no other methods, and the
//! engine that owned it reported "all properties verified" for every input.
//! What is implemented here is an *honest subset*: bounded reachability
//! checking of state invariants, plus a small, fully specified atomic
//! predicate language. Anything outside that subset -- liveness, fairness,
//! nested temporal operators, the full CTL/LTL grammar -- returns
//! [`OptimError::UnsupportedOperation`] naming the unsupported construct,
//! never a vacuous success.
//!
//! # Specification language
//!
//! ```text
//! spec       := "AG(" atom ")" | "INV(" atom ")" | atom
//! atom       := comparison | flag | "finite(" term ")" | "!" atom
//! comparison := term op number          op := "<=" | "<" | ">=" | ">" | "=="
//! term       := "epsilon" | "delta" | "var:" identifier
//! flag       := "data_minimization" | "purpose_limitation" | "storage_limitation"
//! ```
//!
//! `AG` (or `INV`) means "on all paths, globally" -- exactly the invariant
//! semantics that bounded reachability decides. A bare atom is treated as an
//! invariant as well, which is the reading `PropertyType::Invariant` implies.

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

use super::types::{PropertyType, SystemProperty, SystemState, TransitionFunction};

/// Default cap on the number of states explored before the checker gives up.
pub const DEFAULT_STATE_BOUND: usize = 100_000;

/// A term that can be read out of a system state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// The epsilon budget of the state's privacy context.
    Epsilon,
    /// The delta budget of the state's privacy context.
    Delta,
    /// A named state variable.
    Variable(String),
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Less than or equal.
    LessOrEqual,
    /// Strictly less.
    Less,
    /// Greater than or equal.
    GreaterOrEqual,
    /// Strictly greater.
    Greater,
    /// Exact equality of the IEEE-754 value.
    Equal,
}

/// A boolean flag of the privacy context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextFlag {
    /// `PrivacyContext::data_minimization`.
    DataMinimization,
    /// `PrivacyContext::purpose_limitation`.
    PurposeLimitation,
    /// `PrivacyContext::storage_limitation`.
    StorageLimitation,
}

/// An atomic state predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum StatePredicate {
    /// `term op number`
    Compare {
        /// Left-hand term read from the state.
        term: Term,
        /// Comparison operator.
        op: CompareOp,
        /// Right-hand constant.
        value: f64,
    },
    /// `finite(term)`
    Finite(Term),
    /// A boolean privacy-context flag.
    Flag(ContextFlag),
    /// Logical negation.
    Not(Box<StatePredicate>),
}

impl StatePredicate {
    /// Parse a specification string into an invariant predicate.
    ///
    /// Returns [`OptimError::UnsupportedOperation`] for syntactically valid
    /// temporal logic this checker cannot decide, and
    /// [`OptimError::InvalidParameter`] for text that is not a specification
    /// at all.
    pub fn parse(specification: &str) -> Result<Self> {
        let trimmed = specification.trim();
        let inner = if let Some(rest) = strip_wrapper(trimmed, "AG") {
            rest
        } else if let Some(rest) = strip_wrapper(trimmed, "INV") {
            rest
        } else {
            for unsupported in ["AF", "AX", "AU", "EG", "EF", "EX", "EU", "G", "F", "X", "U"] {
                if strip_wrapper(trimmed, unsupported).is_some() {
                    return Err(OptimError::UnsupportedOperation(format!(
                        "the temporal operator `{unsupported}` in specification `{specification}` \
                         is not decided by this checker; only invariants (`AG(...)` / `INV(...)`) \
                         are supported"
                    )));
                }
            }
            trimmed
        };
        Self::parse_atom(inner.trim(), specification)
    }

    /// Parse an atomic predicate (with optional leading `!`).
    fn parse_atom(text: &str, full: &str) -> Result<Self> {
        if let Some(rest) = text.strip_prefix('!') {
            return Ok(Self::Not(Box::new(Self::parse_atom(rest.trim(), full)?)));
        }
        if let Some(rest) = strip_wrapper(text, "finite") {
            return Ok(Self::Finite(parse_term(rest.trim(), full)?));
        }
        match text {
            "data_minimization" => return Ok(Self::Flag(ContextFlag::DataMinimization)),
            "purpose_limitation" => return Ok(Self::Flag(ContextFlag::PurposeLimitation)),
            "storage_limitation" => return Ok(Self::Flag(ContextFlag::StorageLimitation)),
            _ => {}
        }

        // Longest operators first so `<=` is not read as `<`.
        for (token, op) in [
            ("<=", CompareOp::LessOrEqual),
            (">=", CompareOp::GreaterOrEqual),
            ("==", CompareOp::Equal),
            ("<", CompareOp::Less),
            (">", CompareOp::Greater),
        ] {
            if let Some(position) = text.find(token) {
                let left = text[..position].trim();
                let right = text[position + token.len()..].trim();
                let value: f64 = right.parse().map_err(|_| {
                    OptimError::InvalidParameter(format!(
                        "the right-hand side `{right}` of specification `{full}` is not a number"
                    ))
                })?;
                return Ok(Self::Compare {
                    term: parse_term(left, full)?,
                    op,
                    value,
                });
            }
        }

        Err(OptimError::UnsupportedOperation(format!(
            "specification `{full}` is not an atomic predicate this checker understands; see the \
             grammar in `privacy::enhanced_audit::model_checking`"
        )))
    }

    /// Evaluate the predicate in a state.
    pub fn evaluate<T: Float + Debug + Send + Sync + 'static>(
        &self,
        state: &SystemState<T>,
    ) -> Result<bool> {
        match self {
            Self::Compare { term, op, value } => {
                let left = read_term(term, state)?;
                Ok(match op {
                    CompareOp::LessOrEqual => left <= *value,
                    CompareOp::Less => left < *value,
                    CompareOp::GreaterOrEqual => left >= *value,
                    CompareOp::Greater => left > *value,
                    CompareOp::Equal => left == *value,
                })
            }
            Self::Finite(term) => Ok(read_term(term, state)?.is_finite()),
            Self::Flag(flag) => Ok(match flag {
                ContextFlag::DataMinimization => state.privacy_params.data_minimization,
                ContextFlag::PurposeLimitation => state.privacy_params.purpose_limitation,
                ContextFlag::StorageLimitation => state.privacy_params.storage_limitation,
            }),
            Self::Not(inner) => Ok(!inner.evaluate(state)?),
        }
    }
}

/// Strip a `name(...)` wrapper, returning the contents.
fn strip_wrapper<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(name)?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('(')?;
    rest.strip_suffix(')')
}

/// Parse a term.
fn parse_term(text: &str, full: &str) -> Result<Term> {
    match text {
        "epsilon" => Ok(Term::Epsilon),
        "delta" => Ok(Term::Delta),
        _ => match text.strip_prefix("var:") {
            Some(name) if !name.is_empty() => Ok(Term::Variable(name.to_string())),
            _ => Err(OptimError::InvalidParameter(format!(
                "`{text}` in specification `{full}` is not a term (expected `epsilon`, `delta` or \
                 `var:<name>`)"
            ))),
        },
    }
}

/// Read a term out of a state.
fn read_term<T: Float + Debug + Send + Sync + 'static>(
    term: &Term,
    state: &SystemState<T>,
) -> Result<f64> {
    match term {
        Term::Epsilon => Ok(state.privacy_params.epsilon_budget),
        Term::Delta => Ok(state.privacy_params.delta_budget),
        Term::Variable(name) => {
            let value = state.variables.get(name).ok_or_else(|| {
                OptimError::InvalidState(format!(
                    "state `{}` has no variable named `{name}`, so the property cannot be decided",
                    state.id
                ))
            })?;
            value.to_f64().ok_or_else(|| {
                OptimError::InvalidState(format!(
                    "variable `{name}` of state `{}` cannot be represented as f64",
                    state.id
                ))
            })
        }
    }
}

/// Outcome of checking one property.
#[derive(Debug, Clone)]
pub struct ModelCheckOutcome {
    /// Name of the checked property.
    pub property: String,
    /// Whether the invariant held in every reachable state.
    pub holds: bool,
    /// Number of distinct states explored.
    pub states_explored: usize,
    /// Identifier of the first state violating the invariant, if any.
    pub counterexample: Option<String>,
}

/// System model for verification.
pub struct SystemModel<T: Float + Debug + Send + Sync + 'static> {
    /// Declared initial states.
    states: Vec<SystemState<T>>,
    /// Named transition relations.
    transitions: HashMap<String, TransitionFunction<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static> SystemModel<T> {
    /// Create an empty model.
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            transitions: HashMap::new(),
        }
    }

    /// Add an initial state.
    pub fn add_initial_state(&mut self, state: SystemState<T>) {
        self.states.push(state);
    }

    /// Register a transition relation.
    pub fn add_transition(&mut self, transition: TransitionFunction<T>) {
        self.transitions.insert(transition.name.clone(), transition);
    }

    /// Number of declared initial states.
    pub fn initial_state_count(&self) -> usize {
        self.states.len()
    }

    /// Number of registered transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Bounded breadth-first exploration of the reachable state space,
    /// checking `predicate` in every state.
    ///
    /// States are deduplicated by identifier. Exceeding `state_bound` is an
    /// error, not a pass: an unfinished exploration proves nothing.
    pub fn check_invariant(
        &self,
        predicate: &StatePredicate,
        state_bound: usize,
    ) -> Result<(bool, usize, Option<String>)> {
        if self.states.is_empty() {
            return Err(OptimError::InvalidState(
                "the system model declares no initial state, so no property can be decided"
                    .to_string(),
            ));
        }

        let mut seen: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<SystemState<T>> = VecDeque::new();
        for state in &self.states {
            if seen.insert(state.id.clone()) {
                queue.push_back(state.clone());
            }
        }

        let mut explored = 0usize;
        while let Some(state) = queue.pop_front() {
            explored += 1;
            if explored > state_bound {
                return Err(OptimError::ResourceError(format!(
                    "the reachable state space exceeded the exploration bound of {state_bound} \
                     states; the property is undecided (not verified)"
                )));
            }
            if !predicate.evaluate(&state)? {
                return Ok((false, explored, Some(state.id.clone())));
            }
            for transition in self.transitions.values() {
                for successor in (transition.logic)(&state) {
                    if seen.insert(successor.id.clone()) {
                        queue.push_back(successor);
                    }
                }
            }
        }

        Ok((true, explored, None))
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for SystemModel<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Bounded invariant model checker.
pub struct ModelChecker<T: Float + Debug + Send + Sync + 'static> {
    /// The system under check.
    model: SystemModel<T>,
    /// Properties to check.
    properties: Vec<SystemProperty>,
    /// Exploration bound.
    state_bound: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> ModelChecker<T> {
    /// Create an empty checker with the default exploration bound.
    pub fn new() -> Self {
        Self {
            model: SystemModel::new(),
            properties: Vec::new(),
            state_bound: DEFAULT_STATE_BOUND,
        }
    }

    /// Replace the exploration bound.
    pub fn set_state_bound(&mut self, state_bound: usize) -> Result<()> {
        if state_bound == 0 {
            return Err(OptimError::InvalidParameter(
                "the state exploration bound must be positive".to_string(),
            ));
        }
        self.state_bound = state_bound;
        Ok(())
    }

    /// Mutable access to the system model.
    pub fn model_mut(&mut self) -> &mut SystemModel<T> {
        &mut self.model
    }

    /// Register a property to check.
    ///
    /// The specification is parsed immediately, so an unsupported property is
    /// rejected at registration rather than silently passing later.
    pub fn add_property(&mut self, property: SystemProperty) -> Result<()> {
        match property.property_type {
            PropertyType::Invariant | PropertyType::Safety => {
                let _ = StatePredicate::parse(&property.specification)?;
                self.properties.push(property);
                Ok(())
            }
            PropertyType::Liveness | PropertyType::Temporal => {
                Err(OptimError::UnsupportedOperation(format!(
                    "property `{}` is a {:?} property; this checker decides invariants only, and \
                     will not report an undecided property as verified",
                    property.name, property.property_type
                )))
            }
        }
    }

    /// Number of registered properties.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Check one property.
    pub fn check_property(&self, property: &SystemProperty) -> Result<ModelCheckOutcome> {
        match property.property_type {
            PropertyType::Invariant | PropertyType::Safety => {}
            PropertyType::Liveness | PropertyType::Temporal => {
                return Err(OptimError::UnsupportedOperation(format!(
                    "property `{}` is a {:?} property; deciding it needs a full temporal-logic \
                     model checker, which is not implemented here",
                    property.name, property.property_type
                )))
            }
        }
        let predicate = StatePredicate::parse(&property.specification)?;
        let (holds, explored, counterexample) =
            self.model.check_invariant(&predicate, self.state_bound)?;
        Ok(ModelCheckOutcome {
            property: property.name.clone(),
            holds,
            states_explored: explored,
            counterexample,
        })
    }

    /// Check every registered property.
    ///
    /// An empty property set is an error: "zero properties checked" is not the
    /// same claim as "the system is correct".
    pub fn check_all(&self) -> Result<Vec<ModelCheckOutcome>> {
        if self.properties.is_empty() {
            return Err(OptimError::InvalidState(
                "no properties are registered with the model checker; there is nothing to verify"
                    .to_string(),
            ));
        }
        self.properties
            .iter()
            .map(|property| self.check_property(property))
            .collect()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ModelChecker<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::enhanced_audit::types::PrivacyContext;

    fn context(epsilon: f64) -> PrivacyContext {
        PrivacyContext {
            epsilon_budget: epsilon,
            delta_budget: 1e-6,
            privacy_mechanism: "dp_sgd".to_string(),
            data_minimization: true,
            purpose_limitation: true,
            storage_limitation: false,
        }
    }

    fn state(id: &str, spent: f64) -> SystemState<f64> {
        let mut variables = HashMap::new();
        variables.insert("spent".to_string(), spent);
        SystemState {
            id: id.to_string(),
            variables,
            privacy_params: context(spent),
        }
    }

    /// A chain of `steps` states, each spending 0.5 more epsilon.
    fn spending_model(steps: usize) -> SystemModel<f64> {
        let mut model = SystemModel::new();
        model.add_initial_state(state("s0", 0.0));
        let limit = steps;
        model.add_transition(TransitionFunction {
            name: "spend".to_string(),
            logic: Box::new(move |current: &SystemState<f64>| {
                let step = current
                    .id
                    .strip_prefix('s')
                    .and_then(|rest| rest.parse::<usize>().ok())
                    .unwrap_or(0);
                if step >= limit {
                    Vec::new()
                } else {
                    vec![state(&format!("s{}", step + 1), (step + 1) as f64 * 0.5)]
                }
            }),
        });
        model
    }

    #[test]
    fn an_invariant_that_holds_is_reported_as_holding() {
        let model = spending_model(4);
        let predicate = match StatePredicate::parse("AG(var:spent <= 2.0)") {
            Ok(predicate) => predicate,
            Err(err) => panic!("parse failed: {err}"),
        };
        let (holds, explored, counterexample) =
            match model.check_invariant(&predicate, DEFAULT_STATE_BOUND) {
                Ok(outcome) => outcome,
                Err(err) => panic!("check failed: {err}"),
            };
        assert!(holds);
        assert_eq!(explored, 5, "s0..s4 inclusive");
        assert!(counterexample.is_none());
    }

    #[test]
    fn an_invariant_that_is_violated_yields_a_counterexample() {
        let model = spending_model(4);
        let predicate = match StatePredicate::parse("AG(var:spent <= 1.0)") {
            Ok(predicate) => predicate,
            Err(err) => panic!("parse failed: {err}"),
        };
        let (holds, _explored, counterexample) =
            match model.check_invariant(&predicate, DEFAULT_STATE_BOUND) {
                Ok(outcome) => outcome,
                Err(err) => panic!("check failed: {err}"),
            };
        assert!(!holds, "spending reaches 2.0, which violates <= 1.0");
        assert_eq!(counterexample.as_deref(), Some("s3"));
    }

    #[test]
    fn exceeding_the_state_bound_is_an_error_not_a_pass() {
        let model = spending_model(1000);
        let predicate = match StatePredicate::parse("AG(var:spent >= 0.0)") {
            Ok(predicate) => predicate,
            Err(err) => panic!("parse failed: {err}"),
        };
        let outcome = model.check_invariant(&predicate, 10);
        assert!(
            outcome.is_err(),
            "an unfinished exploration must not report success"
        );
    }

    #[test]
    fn a_model_with_no_initial_state_cannot_decide_anything() {
        let model: SystemModel<f64> = SystemModel::new();
        let predicate = match StatePredicate::parse("data_minimization") {
            Ok(predicate) => predicate,
            Err(err) => panic!("parse failed: {err}"),
        };
        assert!(model.check_invariant(&predicate, 10).is_err());
    }

    #[test]
    fn liveness_and_temporal_properties_are_refused_explicitly() {
        let mut checker: ModelChecker<f64> = ModelChecker::new();
        let outcome = checker.add_property(SystemProperty {
            name: "eventually_terminates".to_string(),
            specification: "AF(var:spent >= 2.0)".to_string(),
            property_type: PropertyType::Liveness,
        });
        let message = match outcome {
            Err(err) => err.to_string(),
            Ok(()) => panic!("a liveness property must not be accepted"),
        };
        assert!(message.contains("invariants only"), "got: {message}");
    }

    #[test]
    fn unsupported_temporal_operators_are_named_in_the_error() {
        let outcome = StatePredicate::parse("EF(var:spent >= 1.0)");
        let message = match outcome {
            Err(err) => err.to_string(),
            Ok(_) => panic!("EF must not parse"),
        };
        assert!(message.contains("EF"), "got: {message}");
    }

    #[test]
    fn checking_with_no_registered_properties_is_an_error() {
        let checker: ModelChecker<f64> = ModelChecker::new();
        assert!(
            checker.check_all().is_err(),
            "zero properties checked must not read as verified"
        );
    }

    #[test]
    fn the_checker_runs_registered_invariants_end_to_end() {
        let mut checker: ModelChecker<f64> = ModelChecker::new();
        {
            let model = checker.model_mut();
            model.add_initial_state(state("s0", 0.0));
            model.add_transition(TransitionFunction {
                name: "spend".to_string(),
                logic: Box::new(|current: &SystemState<f64>| {
                    if current.id == "s0" {
                        vec![state("s1", 3.0)]
                    } else {
                        Vec::new()
                    }
                }),
            });
        }
        let ok = checker.add_property(SystemProperty {
            name: "budget_bounded".to_string(),
            specification: "AG(epsilon <= 1.0)".to_string(),
            property_type: PropertyType::Safety,
        });
        assert!(ok.is_ok());

        let outcomes = match checker.check_all() {
            Ok(outcomes) => outcomes,
            Err(err) => panic!("check_all failed: {err}"),
        };
        assert_eq!(outcomes.len(), 1);
        assert!(!outcomes[0].holds, "s1 spends 3.0 > 1.0");
        assert_eq!(outcomes[0].counterexample.as_deref(), Some("s1"));
    }

    #[test]
    fn every_atom_of_the_grammar_evaluates() {
        let good = state("ok", 0.25);
        let cases: [(&str, bool); 9] = [
            ("epsilon <= 1.0", true),
            ("epsilon > 1.0", false),
            ("delta < 0.001", true),
            ("var:spent == 0.25", true),
            ("finite(var:spent)", true),
            ("data_minimization", true),
            ("purpose_limitation", true),
            ("storage_limitation", false),
            ("!storage_limitation", true),
        ];
        for (specification, expected) in cases {
            let predicate = match StatePredicate::parse(specification) {
                Ok(predicate) => predicate,
                Err(err) => panic!("`{specification}` failed to parse: {err}"),
            };
            let value = match predicate.evaluate(&good) {
                Ok(value) => value,
                Err(err) => panic!("`{specification}` failed to evaluate: {err}"),
            };
            assert_eq!(value, expected, "specification `{specification}`");
        }
    }

    #[test]
    fn a_missing_state_variable_is_an_error_not_false() {
        let predicate = match StatePredicate::parse("var:absent <= 1.0") {
            Ok(predicate) => predicate,
            Err(err) => panic!("parse failed: {err}"),
        };
        assert!(predicate.evaluate(&state("s", 0.0)).is_err());
    }

    #[test]
    fn nonsense_specifications_are_rejected() {
        assert!(StatePredicate::parse("this is not a predicate").is_err());
        assert!(StatePredicate::parse("epsilon <= not_a_number").is_err());
        assert!(StatePredicate::parse("unknown_term <= 1.0").is_err());
    }
}
