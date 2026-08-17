//! Evolutionary hyperparameter search for
//! [`super::OptimizationStrategy::Evolutionary`] (F15).
//!
//! The previous implementation pushed 20 random configurations into
//! `state.population` and then never read it again — the population was
//! write-only, and every suggestion after the first was a mutation of the single
//! best configuration, so there was no recombination and no selection pressure
//! beyond "the best so far".
//!
//! This module implements real selection, crossover and mutation over the
//! population, all bounded by the search space.

use super::support::{
    bounds, denormalize, normalize, sample_parameter, snap_to_discrete, sorted_categorical_names,
    sorted_parameter_names, standard_normal, StrategyRng, SuggestedParameters,
};
use super::{HyperparameterConfiguration, HyperparameterSpace};
use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use scirs2_core::RngExt;
use std::collections::HashMap;

/// Tunables of the evolutionary strategy.
#[derive(Debug, Clone, Copy)]
pub struct EvolutionConfig {
    /// Target population size.
    pub population_size: usize,
    /// Number of individuals compared in each tournament.
    pub tournament_size: usize,
    /// Per-parameter probability of taking the second parent's value.
    pub crossover_rate: f64,
    /// Per-parameter probability of a mutation.
    pub mutation_rate: f64,
    /// Gaussian mutation width as a fraction of the (normalized) axis.
    pub mutation_scale: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            tournament_size: 3,
            crossover_rate: 0.5,
            mutation_rate: 0.3,
            mutation_scale: 0.15,
        }
    }
}

/// Pick the fittest of `tournament_size` random individuals (higher score wins).
fn tournament<'a, T: Float>(
    population: &'a [(&'a HyperparameterConfiguration<T>, f64)],
    config: &EvolutionConfig,
    rng: &mut StrategyRng,
) -> &'a HyperparameterConfiguration<T> {
    let mut best = rng.gen_range(0..population.len());
    for _ in 1..config.tournament_size.max(1) {
        let challenger = rng.gen_range(0..population.len());
        if population[challenger].1 > population[best].1 {
            best = challenger;
        }
    }
    population[best].0
}

/// Produce one offspring by tournament-selecting two parents, recombining them
/// per parameter, and mutating the result inside the declared bounds.
///
/// Returns an error when the population is empty — there is nothing to breed
/// from, and inventing a random configuration here would hide the fact that
/// selection never happened.
pub fn breed<T: Float>(
    space: &HyperparameterSpace<T>,
    population: &[(&HyperparameterConfiguration<T>, f64)],
    config: &EvolutionConfig,
    rng: &mut StrategyRng,
) -> Result<SuggestedParameters<T>> {
    if population.is_empty() {
        return Err(OptimError::InvalidConfig(
            "evolutionary search needs a scored population to select parents from \
             (run the random initial design first)"
                .to_string(),
        ));
    }

    let first = tournament(population, config, rng);
    let second = tournament(population, config, rng);

    let mut parameters = HashMap::new();
    for name in sorted_parameter_names(space) {
        let Some(range) = space.parameter_ranges().get(&name) else {
            continue;
        };
        // Recombine: take either parent's value for this gene.
        let inherited = if rng.gen_range(0.0..1.0) < config.crossover_rate {
            second.parameters.get(&name)
        } else {
            first.parameters.get(&name)
        }
        .or_else(|| first.parameters.get(&name))
        .or_else(|| second.parameters.get(&name))
        .map(|value| value.to_f64().unwrap_or(0.0));

        let mut raw = match inherited {
            Some(value) => value,
            // Neither parent carries this gene: draw it from the prior.
            None => sample_parameter(range, rng),
        };

        // Mutate on the parameter's own axis so a log-scaled range mutates
        // multiplicatively rather than additively.
        if rng.gen_range(0.0..1.0) < config.mutation_rate {
            let unit = normalize(range, raw);
            let mutated = (unit + standard_normal(rng) * config.mutation_scale).clamp(0.0, 1.0);
            raw = denormalize(range, mutated);
        }

        let (lower, upper) = bounds(range);
        raw = snap_to_discrete(range, raw.clamp(lower, upper));
        parameters.insert(
            name,
            scirs2_core::numeric::NumCast::from(raw).unwrap_or_else(T::zero),
        );
    }

    let mut categorical = HashMap::new();
    for name in sorted_categorical_names(space) {
        let Some(options) = space.categorical_options().get(&name) else {
            continue;
        };
        if options.is_empty() {
            continue;
        }
        let inherited = if rng.gen_range(0.0..1.0) < config.crossover_rate {
            second.categorical_parameters.get(&name)
        } else {
            first.categorical_parameters.get(&name)
        }
        .or_else(|| first.categorical_parameters.get(&name))
        .filter(|value| options.contains(value))
        .cloned();

        let chosen = if rng.gen_range(0.0..1.0) < config.mutation_rate || inherited.is_none() {
            options[rng.gen_range(0..options.len())].clone()
        } else {
            inherited.unwrap_or_else(|| options[0].clone())
        };
        categorical.insert(name, chosen);
    }

    Ok((parameters, categorical))
}

/// Trim `population` to the fittest `population_size` individuals (higher score
/// first). Called after every recorded evaluation so the population is a real,
/// bounded, elitist pool rather than an append-only log.
pub fn survivor_selection<T: Float>(
    population: &mut Vec<HyperparameterConfiguration<T>>,
    config: &EvolutionConfig,
) {
    population.sort_by(|a, b| {
        let sa = a
            .score
            .and_then(|s| s.to_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let sb = b
            .score
            .and_then(|s| s.to_f64())
            .unwrap_or(f64::NEG_INFINITY);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    population.truncate(config.population_size.max(1));
}

#[cfg(test)]
mod tests {
    use super::super::{DistributionType, ParameterRange};
    use super::*;
    use scirs2_core::random::Random;

    fn space() -> HyperparameterSpace<f64> {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        space.add_parameter(
            "x".to_string(),
            ParameterRange {
                name: "x".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                distribution: DistributionType::Uniform,
                log_scale: false,
                discrete_values: None,
            },
        );
        space.add_categorical_parameter(
            "opt".to_string(),
            vec!["adam".to_string(), "sgd".to_string()],
        );
        space
    }

    fn config_with(id: &str, x: f64, opt: &str, score: f64) -> HyperparameterConfiguration<f64> {
        let mut parameters = HashMap::new();
        parameters.insert("x".to_string(), x);
        let mut categorical = HashMap::new();
        categorical.insert("opt".to_string(), opt.to_string());
        HyperparameterConfiguration {
            id: id.to_string(),
            parameters,
            categorical_parameters: categorical,
            score: Some(score),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn breeding_requires_a_population() {
        let mut rng = Random::seed(1);
        assert!(breed(&space(), &[], &EvolutionConfig::default(), &mut rng).is_err());
    }

    #[test]
    fn selection_pressure_pulls_offspring_toward_the_fittest_parents() {
        let population = [
            config_with("a", 0.90, "adam", 1.0),
            config_with("b", 0.88, "adam", 0.9),
            config_with("c", 0.10, "sgd", -1.0),
            config_with("d", 0.12, "sgd", -0.9),
        ];
        let scored: Vec<(&HyperparameterConfiguration<f64>, f64)> = population
            .iter()
            .map(|c| (c, c.score.unwrap_or(0.0)))
            .collect();

        let space = space();
        let config = EvolutionConfig {
            mutation_rate: 0.0,
            ..EvolutionConfig::default()
        };
        let mut rng = Random::seed(555);
        let mut values = Vec::new();
        for _ in 0..60 {
            let (parameters, _) = breed(&space, &scored, &config, &mut rng).expect("breed");
            values.push(parameters["x"]);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!(
            mean > 0.6,
            "tournament selection must favour the high-scoring parents, mean = {mean}"
        );
    }

    #[test]
    fn offspring_stay_inside_the_declared_bounds() {
        let population = [config_with("a", 0.99, "adam", 1.0)];
        let scored: Vec<(&HyperparameterConfiguration<f64>, f64)> =
            population.iter().map(|c| (c, 1.0)).collect();
        let space = space();
        let config = EvolutionConfig {
            mutation_rate: 1.0,
            mutation_scale: 5.0,
            ..EvolutionConfig::default()
        };
        let mut rng = Random::seed(31);
        for _ in 0..200 {
            let (parameters, categorical) =
                breed(&space, &scored, &config, &mut rng).expect("breed");
            let x = parameters["x"];
            assert!((0.0..=1.0).contains(&x), "x escaped: {x}");
            assert!(["adam", "sgd"].contains(&categorical["opt"].as_str()));
        }
    }

    #[test]
    fn crossover_mixes_both_parents() {
        let population = [
            config_with("low", 0.0, "adam", 1.0),
            config_with("high", 1.0, "sgd", 1.0),
        ];
        let scored: Vec<(&HyperparameterConfiguration<f64>, f64)> =
            population.iter().map(|c| (c, 1.0)).collect();
        let space = space();
        let config = EvolutionConfig {
            mutation_rate: 0.0,
            ..EvolutionConfig::default()
        };
        let mut rng = Random::seed(808);
        let mut seen_low = false;
        let mut seen_high = false;
        for _ in 0..100 {
            let (parameters, _) = breed(&space, &scored, &config, &mut rng).expect("breed");
            if parameters["x"] == 0.0 {
                seen_low = true;
            }
            if parameters["x"] == 1.0 {
                seen_high = true;
            }
        }
        assert!(
            seen_low && seen_high,
            "recombination must be able to inherit from either parent"
        );
    }

    #[test]
    fn survivor_selection_keeps_the_best_and_bounds_the_population() {
        let mut population: Vec<HyperparameterConfiguration<f64>> = (0..50)
            .map(|i| config_with(&format!("c{i}"), 0.5, "adam", i as f64))
            .collect();
        let config = EvolutionConfig {
            population_size: 10,
            ..EvolutionConfig::default()
        };
        survivor_selection(&mut population, &config);
        assert_eq!(population.len(), 10);
        assert_eq!(population[0].score, Some(49.0));
        assert_eq!(population[9].score, Some(40.0));
    }
}
