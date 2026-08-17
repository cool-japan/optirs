//! Further MOEA scaffolding: SMS-EMOA and IBEA (indicator-based) optimizers.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::core::{Individual, MultiObjectiveStatistics, ParetoFront};
use super::hypervolume::HypervolumeCalculator;

/// Indicator-Based Evolutionary Algorithm (IBEA)
pub struct IBEA<T: Float + Debug + Send + Sync + 'static> {
    /// Population
    pub(super) population: Vec<Individual<T>>,
    /// Fitness values based on indicators
    pub(super) indicator_fitness: Vec<T>,
    /// Quality indicator
    pub(super) quality_indicator: QualityIndicator<T>,
    /// Scaling factor
    pub(super) scaling_factor: T,
    /// Pareto front
    pub(super) pareto_front: ParetoFront<T>,
    /// Generation counter
    pub(super) generation: usize,
    /// Statistics
    pub(super) statistics: MultiObjectiveStatistics<T>,
}
/// Types of quality indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorType {
    /// Additive epsilon indicator
    AdditiveEpsilon,
    /// Multiplicative epsilon indicator
    MultiplicativeEpsilon,
    /// Hypervolume contribution
    HypervolumeContribution,
    /// R2 indicator
    R2,
}
/// Quality indicators for IBEA
#[derive(Debug)]
pub struct QualityIndicator<T: Float + Debug + Send + Sync + 'static> {
    /// Indicator type
    pub(super) indicator_type: IndicatorType,
    /// Indicator parameters
    pub(super) parameters: HashMap<String, T>,
}
/// SMS-EMOA implementation
pub struct SmsEmoa<T: Float + Debug + Send + Sync + 'static> {
    /// Base population
    pub(super) population: Vec<Individual<T>>,
    /// Pareto front
    pub(super) pareto_front: ParetoFront<T>,
    /// Hypervolume calculator
    pub(super) hypervolume_calculator: HypervolumeCalculator<T>,
    /// Reference point for hypervolume
    pub(super) reference_point: Vec<T>,
    /// Generation counter
    pub(super) generation: usize,
    /// Statistics
    pub(super) statistics: MultiObjectiveStatistics<T>,
}
