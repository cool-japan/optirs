//! Interactive-preference articulation and constraint-handling scaffolding for constrained/interactive multi-objective optimization.

use crate::nas_engine::{ConstraintHandlingMethod, UserPreferences};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Preference articulation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArticulationMethod {
    /// A priori (before optimization)
    APriori,
    /// Interactive (during optimization)
    Interactive,
    /// A posteriori (after optimization)
    APosteriori,
    /// Progressive (evolving preferences)
    Progressive,
}
/// Constraint directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintDirection {
    LessThanOrEqual,
    GreaterThanOrEqual,
    Equal,
}
/// Constraint function
#[derive(Debug)]
pub struct ConstraintFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Function type
    pub(super) function_type: ConstraintType,
    /// Function parameters
    pub(super) parameters: HashMap<String, T>,
    /// Constraint bound
    pub(super) bound: T,
    /// Constraint direction (<=, >=, )
    pub(super) direction: ConstraintDirection,
}
/// Constraint handler for constrained multi-objective optimization
pub struct ConstraintHandler<T: Float + Debug + Send + Sync + 'static> {
    /// Constraint handling method
    pub(super) method: ConstraintHandlingMethod,
    /// Constraint functions
    pub(super) constraints: Vec<ConstraintFunction<T>>,
    /// Constraint tolerance
    pub(super) tolerance: T,
    /// Penalty parameters
    pub(super) penalty_parameters: PenaltyParameters<T>,
}
/// Types of constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Linear constraint
    Linear,
    /// Quadratic constraint
    Quadratic,
    /// Nonlinear constraint
    Nonlinear,
    /// Resource constraint
    Resource,
    /// Performance constraint
    Performance,
    /// Custom constraint
    Custom,
}
/// Penalty function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyFunctionType {
    /// Linear penalty
    Linear,
    /// Quadratic penalty
    Quadratic,
    /// Exponential penalty
    Exponential,
    /// Logarithmic penalty
    Logarithmic,
}
/// Penalty parameters for constraint handling
#[derive(Debug, Clone)]
pub struct PenaltyParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Static penalty weight
    pub(super) static_weight: T,
    /// Dynamic penalty weight
    pub(super) dynamic_weight: T,
    /// Penalty increase rate
    pub(super) increase_rate: T,
    /// Penalty function type
    pub(super) penalty_function: PenaltyFunctionType,
}
/// Preference handling for interactive optimization
pub struct PreferenceHandler<T: Float + Debug + Send + Sync + 'static> {
    /// User preferences
    pub(super) preferences: UserPreferences<T>,
    /// Preference articulation method
    pub(super) articulation_method: ArticulationMethod,
    /// Decision maker utilities
    pub(super) utilities: Vec<T>,
    /// Preference history
    pub(super) preference_history: Vec<PreferenceSnapshot<T>>,
}
/// Preference snapshot
#[derive(Debug, Clone)]
pub struct PreferenceSnapshot<T: Float + Debug + Send + Sync + 'static> {
    /// Timestamp
    pub(super) timestamp: std::time::SystemTime,
    /// Preference values
    pub(super) preferences: HashMap<String, T>,
    /// Confidence levels
    pub(super) confidence: HashMap<String, T>,
    /// Context information
    pub(super) context: String,
}
